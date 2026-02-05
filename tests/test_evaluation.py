"""Tests for the evaluation pipeline (recorder, evaluator, replay)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.evaluation.evaluator import Evaluator, GapMetrics
from src.evaluation.recorder import Recorder
from src.evaluation.replay import Replay
from src.pose_estimator import Point3D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_keypoints():
    """Create a minimal set of 6 upper-body keypoints."""
    names = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
    ]
    return {
        name: Point3D(
            x=0.5, y=0.5, z=0.0, visibility=0.9,
            world_x=0.0, world_y=0.0, world_z=0.0,
        )
        for name in names
    }


# ---------------------------------------------------------------------------
# Recorder tests
# ---------------------------------------------------------------------------

class TestRecorder:
    def test_recorder_creates_jsonl(self, tmp_path):
        recorder = Recorder(output_dir=str(tmp_path))
        session_id = recorder.start_session(config_snapshot={"test": True})

        for i in range(100):
            recorder.record_frame(
                timestamp=1000.0 + i * 0.033,
                frame_number=i,
                keypoints=make_keypoints(),
                pose_confidence=0.9,
                human_angles=np.zeros(8),
                robot_cmd=np.ones(8) * 0.1,
                robot_actual=np.ones(8) * 0.1,
                latency_ms=15.0,
            )

        session_dir = recorder.end_session()

        # Check JSONL file
        frames_path = session_dir / "frames.jsonl"
        assert frames_path.exists()

        lines = frames_path.read_text().strip().split("\n")
        assert len(lines) == 100

        # Verify each line is valid JSON with required fields
        first = json.loads(lines[0])
        assert "timestamp" in first
        assert "frame_number" in first
        assert "keypoints" in first
        assert "human_angles" in first
        assert "robot_cmd" in first
        assert "robot_actual" in first
        assert "gap_b" in first
        assert "gap_b_rmse" in first
        assert "latency_ms" in first
        assert len(first["human_angles"]) == 8
        assert len(first["robot_cmd"]) == 8
        assert len(first["gap_b"]) == 8

    def test_recorder_metadata(self, tmp_path):
        recorder = Recorder(output_dir=str(tmp_path))
        recorder.start_session(config_snapshot={"smoothing_factor": 0.3})

        for i in range(10):
            recorder.record_frame(
                timestamp=1000.0 + i * 0.033,
                frame_number=i,
                keypoints=make_keypoints(),
                pose_confidence=0.85,
                human_angles=np.zeros(8),
                robot_cmd=np.ones(8) * 0.1,
                robot_actual=np.ones(8) * 0.1,
                latency_ms=14.0,
            )

        session_dir = recorder.end_session()

        metadata_path = session_dir / "metadata.json"
        assert metadata_path.exists()

        meta = json.loads(metadata_path.read_text())
        assert meta["total_frames"] == 10
        assert "session_id" in meta
        assert "config_snapshot" in meta
        assert meta["config_snapshot"]["smoothing_factor"] == 0.3
        assert "avg_gap_b_rmse_deg" in meta
        assert "avg_latency_ms" in meta
        assert "avg_pose_confidence" in meta

    def test_recorder_no_keypoints(self, tmp_path):
        """Recording with keypoints=None should write null."""
        recorder = Recorder(output_dir=str(tmp_path))
        recorder.start_session(config_snapshot={})

        recorder.record_frame(
            timestamp=1000.0,
            frame_number=0,
            keypoints=None,
            pose_confidence=0.0,
            human_angles=np.zeros(8),
            robot_cmd=np.zeros(8),
            robot_actual=np.zeros(8),
            latency_ms=10.0,
        )

        session_dir = recorder.end_session()
        line = (session_dir / "frames.jsonl").read_text().strip()
        frame = json.loads(line)
        assert frame["keypoints"] is None


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

class TestEvaluatorGapB:
    def test_gap_b_zero_error(self):
        evaluator = Evaluator(tolerance_deg=15.0)
        cmd = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        actual = cmd.copy()

        gap = evaluator.compute_gap_b(cmd, actual)
        assert gap.rmse_deg == pytest.approx(0.0, abs=1e-6)
        assert gap.max_error_deg == pytest.approx(0.0, abs=1e-6)
        assert gap.accuracy_percent == pytest.approx(100.0)
        assert np.allclose(gap.per_joint_deg, 0.0)

    def test_gap_b_known_error(self):
        evaluator = Evaluator(tolerance_deg=15.0)
        cmd = np.zeros(8)
        # 10 degrees off in radians
        error_rad = np.deg2rad(10.0)
        actual = np.full(8, error_rad)

        gap = evaluator.compute_gap_b(cmd, actual)
        assert gap.rmse_deg == pytest.approx(10.0, abs=0.1)
        assert gap.max_error_deg == pytest.approx(10.0, abs=0.1)
        # accuracy: 100 - (10/15)*100 = 33.3%
        assert gap.accuracy_percent == pytest.approx(33.3, abs=1.0)


class TestEvaluatorStaticPose:
    def test_static_pose_within_tolerance(self):
        evaluator = Evaluator()
        # Human angles match expected (both in the right units)
        human_angles = np.deg2rad(np.array([90, 0, 0, 0, 90, 0, 0, 0], dtype=float))
        expected = {
            "l_shoulder_roll": 90,
            "r_shoulder_roll": 90,
        }
        gap = evaluator.evaluate_static_pose(human_angles, expected, tolerance_deg=10)
        assert gap.rmse_deg == pytest.approx(0.0, abs=0.1)
        assert gap.accuracy_percent == pytest.approx(100.0, abs=1.0)

    def test_static_pose_outside_tolerance(self):
        evaluator = Evaluator()
        # Human angles are 20 degrees off
        human_angles = np.deg2rad(np.array([70, 0, 0, 0, 70, 0, 0, 0], dtype=float))
        expected = {
            "l_shoulder_roll": 90,
            "r_shoulder_roll": 90,
        }
        gap = evaluator.evaluate_static_pose(human_angles, expected, tolerance_deg=10)
        assert gap.rmse_deg == pytest.approx(20.0, abs=0.5)
        # accuracy: 100 - (20/10)*100 = negative → clamped to 0
        assert gap.accuracy_percent == pytest.approx(0.0, abs=1.0)

    def test_static_pose_partial_joints(self):
        """Only specified joints should be evaluated."""
        evaluator = Evaluator()
        human_angles = np.deg2rad(np.array([0, 0, 0, 90, 0, 0, 0, 90], dtype=float))
        expected = {"l_elbow": 90, "r_elbow": 90}
        gap = evaluator.evaluate_static_pose(human_angles, expected, tolerance_deg=8)
        assert gap.rmse_deg == pytest.approx(0.0, abs=0.5)


class TestEvaluatorMotionSequence:
    def _make_jab_def(self):
        return {
            "description": "Left jab",
            "keyframes": [
                {"phase": "guard", "time_ratio": 0.0,
                 "angles_deg": {"l_elbow": 90}, "tolerance_deg": 12},
                {"phase": "extend", "time_ratio": 0.4,
                 "angles_deg": {"l_elbow": 0}, "tolerance_deg": 10},
                {"phase": "retract", "time_ratio": 1.0,
                 "angles_deg": {"l_elbow": 90}, "tolerance_deg": 12},
            ],
            "trajectory_constraints": [
                {"joint": "l_elbow", "pattern": "decrease_then_increase"},
            ],
            "timing": {"duration_range_sec": [0.2, 0.8]},
        }

    def test_motion_sequence_correct(self):
        evaluator = Evaluator()
        jab_def = self._make_jab_def()

        # Simulate a correct jab: elbow goes 90 -> 0 -> 90 over 0.5s
        n_frames = 30
        timestamps = [i * 0.5 / (n_frames - 1) for i in range(n_frames)]
        recorded = []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            # Parabolic: 90 at t=0, 0 at t=0.4-ish, 90 at t=1
            elbow_deg = 90.0 * (2 * t - 1) ** 2  # U-shape
            angles = np.zeros(8)
            angles[3] = np.deg2rad(elbow_deg)  # l_elbow index
            recorded.append(angles)

        result = evaluator.evaluate_motion_sequence(recorded, timestamps, jab_def)
        assert result.overall_score > 40.0
        assert result.timing_pass is True

    def test_motion_sequence_wrong_timing(self):
        evaluator = Evaluator()
        jab_def = self._make_jab_def()

        # Same trajectory but over 2.0 seconds (outside [0.2, 0.8])
        n_frames = 30
        timestamps = [i * 2.0 / (n_frames - 1) for i in range(n_frames)]
        recorded = []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            elbow_deg = 90.0 * (2 * t - 1) ** 2
            angles = np.zeros(8)
            angles[3] = np.deg2rad(elbow_deg)
            recorded.append(angles)

        result = evaluator.evaluate_motion_sequence(recorded, timestamps, jab_def)
        assert result.timing_pass is False


class TestTrajectoryConstraints:
    def test_decrease_then_increase(self):
        series = np.array([90, 70, 50, 30, 10, 5, 10, 30, 50, 70, 90], dtype=float)
        assert Evaluator._check_decrease_then_increase(series)

    def test_decrease_then_increase_monotonic_fail(self):
        series = np.array([90, 80, 70, 60, 50, 40, 30, 20, 10], dtype=float)
        assert not Evaluator._check_decrease_then_increase(series)

    def test_increase_then_decrease(self):
        series = np.array([10, 30, 50, 70, 90, 70, 50, 30, 10], dtype=float)
        assert Evaluator._check_increase_then_decrease(series)

    def test_stable(self):
        series = np.array([88, 90, 92, 91, 89, 90, 90, 91, 89], dtype=float)
        assert Evaluator._check_stable(series, range_deg=20)

    def test_stable_fail(self):
        series = np.array([60, 90, 120, 90, 60], dtype=float)
        assert not Evaluator._check_stable(series, range_deg=10)


class TestRunningStats:
    def test_running_stats(self):
        evaluator = Evaluator(tolerance_deg=15.0)
        cmd = np.zeros(8)
        actual = np.full(8, np.deg2rad(5.0))

        gap = evaluator.compute_gap_b(cmd, actual)
        evaluator.update_running_stats(gap)
        evaluator.update_running_stats(gap)

        stats = evaluator.get_running_stats()
        assert stats.rmse_deg == pytest.approx(5.0, abs=0.1)

    def test_running_stats_reset(self):
        evaluator = Evaluator()
        cmd = np.zeros(8)
        actual = np.full(8, np.deg2rad(10.0))
        gap = evaluator.compute_gap_b(cmd, actual)
        evaluator.update_running_stats(gap)

        evaluator.reset_running_stats()
        stats = evaluator.get_running_stats()
        assert stats.rmse_deg == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Replay tests
# ---------------------------------------------------------------------------

class TestReplay:
    def _create_session(self, tmp_path, smoothing=0.3):
        """Helper: create a fake session with recorded keypoints."""
        recorder = Recorder(output_dir=str(tmp_path))
        recorder.start_session(config_snapshot={"angles": {"smoothing_factor": smoothing}})

        # Create keypoints that produce non-trivial angles
        for i in range(20):
            kp = {
                "left_shoulder": Point3D(0.5, 0.5, 0.0, 0.9, 0.0, 0.0, 0.0),
                "right_shoulder": Point3D(0.5, 0.5, 0.0, 0.9, 0.3, 0.0, 0.0),
                "left_elbow": Point3D(0.5, 0.5, 0.0, 0.9, 0.0, 0.0, -0.15),
                "right_elbow": Point3D(0.5, 0.5, 0.0, 0.9, 0.3, 0.0, -0.15),
                "left_wrist": Point3D(0.5, 0.5, 0.0, 0.9, 0.15, 0.0, -0.15),
                "right_wrist": Point3D(0.5, 0.5, 0.0, 0.9, 0.15, 0.0, -0.15),
            }
            recorder.record_frame(
                timestamp=1000.0 + i * 0.033,
                frame_number=i,
                keypoints=kp,
                pose_confidence=0.9,
                human_angles=np.ones(8) * 0.5,
                robot_cmd=np.ones(8) * 0.6,
                robot_actual=np.ones(8) * 0.55,
                latency_ms=15.0,
            )

        return recorder.end_session()

    def test_replay_load_session(self, tmp_path):
        session_dir = self._create_session(tmp_path)
        replay = Replay()
        frames = replay.load_session(str(session_dir))
        assert len(frames) == 20
        assert "timestamp" in frames[0]
        assert "keypoints" in frames[0]

    def test_replay_different_smoothing(self, tmp_path):
        session_dir = self._create_session(tmp_path)
        replay = Replay()
        original = replay.load_session(str(session_dir))

        # Replay with very different smoothing
        replayed = replay.replay_with_config(original, {
            "angles": {"smoothing_factor": 0.9},
            "mapping": {"mirror_mode": True, "dead_zone_deg": 3.0},
        })

        assert len(replayed) == len(original)

        # Robot_cmd should differ due to different smoothing
        orig_cmds = [f["robot_cmd"] for f in original]
        new_cmds = [f["robot_cmd"] for f in replayed]
        # At least some frames should have different commands
        diffs = [
            np.max(np.abs(np.array(a) - np.array(b)))
            for a, b in zip(orig_cmds, new_cmds)
        ]
        assert max(diffs) > 0.001, "Smoothing change should produce different commands"

    def test_replay_comparison(self, tmp_path):
        session_dir = self._create_session(tmp_path)
        replay = Replay()
        original = replay.load_session(str(session_dir))
        replayed = replay.replay_with_config(original, {
            "angles": {"smoothing_factor": 0.9},
            "mapping": {"mirror_mode": True, "dead_zone_deg": 3.0},
        })

        result = replay.compare(original, replayed)
        assert "original_avg_gap_deg" in result
        assert "new_avg_gap_deg" in result
        assert "improvement_deg" in result
        assert "improvement_percent" in result
        assert "per_joint_improvement" in result
        assert len(result["per_joint_improvement"]) == 8
