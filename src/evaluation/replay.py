"""Offline replay. Re-processes stored session data with different parameters."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.angle_calculator import AngleCalculator, JointAngles
from src.evaluation.evaluator import JOINT_NAMES
from src.motion_mapper import MappingConfig, MotionMapper
from src.pose_estimator import Point3D, PoseResult


class Replay:
    def load_session(self, session_dir: str) -> list[dict]:
        frames_path = Path(session_dir) / "frames.jsonl"
        frames = []
        with open(frames_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))
        return frames

    def replay_with_config(
        self, frames: list[dict], new_config: dict
    ) -> list[dict]:
        angle_cfg = new_config.get("angles", {})
        angle_calculator = AngleCalculator(
            smoothing_factor=angle_cfg.get("smoothing_factor", 0.3),
        )

        map_cfg = new_config.get("mapping", {})
        dead_zone_deg = map_cfg.get("dead_zone_deg", 3.0)
        mapping_config = MappingConfig(
            mirror_mode=map_cfg.get("mirror_mode", True),
            dead_zone=np.deg2rad(dead_zone_deg),
        )
        motion_mapper = MotionMapper(config=mapping_config)

        replayed = []
        for frame in frames:
            kp_data = frame.get("keypoints")
            if kp_data is None:
                replayed.append(dict(frame))
                continue

            # Reconstruct PoseResult from stored keypoints
            keypoints = {}
            for name, vals in kp_data.items():
                wx, wy, wz, vis = vals
                keypoints[name] = Point3D(
                    x=0.5, y=0.5, z=0.0,
                    visibility=vis,
                    world_x=wx, world_y=wy, world_z=wz,
                )

            pose = PoseResult(
                keypoints=keypoints,
                is_valid=True,
                timestamp=frame["timestamp"],
            )

            joint_angles = angle_calculator.calculate(pose)
            if joint_angles is None:
                replayed.append(dict(frame))
                continue

            human_angles = joint_angles.to_array()
            servo_angles = motion_mapper.map(joint_angles)
            robot_cmd = servo_angles.angles

            # Re-use original robot_actual (can't re-simulate PyBullet)
            robot_actual = np.array(frame["robot_actual"])
            gap_b = np.abs(robot_cmd - robot_actual)
            gap_b_rmse = float(np.sqrt(np.mean(gap_b**2)))

            new_frame = dict(frame)
            new_frame["human_angles"] = human_angles.tolist()
            new_frame["robot_cmd"] = robot_cmd.tolist()
            new_frame["gap_b"] = gap_b.tolist()
            new_frame["gap_b_rmse"] = gap_b_rmse
            replayed.append(new_frame)

        return replayed

    def compare(
        self, original_frames: list[dict], replayed_frames: list[dict]
    ) -> dict:
        def avg_gap_deg(frames: list[dict]) -> tuple[float, np.ndarray]:
            if not frames:
                return 0.0, np.zeros(8)
            rmses = [f["gap_b_rmse"] for f in frames]
            per_joint = np.mean([f["gap_b"] for f in frames], axis=0)
            return float(np.rad2deg(np.mean(rmses))), np.rad2deg(per_joint)

        orig_avg, orig_per_joint = avg_gap_deg(original_frames)
        new_avg, new_per_joint = avg_gap_deg(replayed_frames)
        improvement = orig_avg - new_avg
        pct = (improvement / orig_avg * 100.0) if orig_avg > 0 else 0.0

        per_joint_improvement = {}
        for i, name in enumerate(JOINT_NAMES):
            per_joint_improvement[name] = {
                "original_deg": round(float(orig_per_joint[i]), 2),
                "new_deg": round(float(new_per_joint[i]), 2),
                "improvement_deg": round(float(orig_per_joint[i] - new_per_joint[i]), 2),
            }

        return {
            "original_avg_gap_deg": round(orig_avg, 2),
            "new_avg_gap_deg": round(new_avg, 2),
            "improvement_deg": round(improvement, 2),
            "improvement_percent": round(pct, 1),
            "per_joint_improvement": per_joint_improvement,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Replay a session with different config and compare results"
    )
    parser.add_argument(
        "--session", required=True, help="Path to session directory"
    )
    parser.add_argument(
        "--config", required=True, help="Path to new config YAML file"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        new_config = yaml.safe_load(f) or {}

    replay = Replay()
    print(f"Loading session from {args.session}...")
    original = replay.load_session(args.session)
    print(f"Loaded {len(original)} frames")

    print("Replaying with new config...")
    replayed = replay.replay_with_config(original, new_config)

    result = replay.compare(original, replayed)

    print(f"\nOriginal avg Gap B: {result['original_avg_gap_deg']:.1f} deg")
    print(f"New avg Gap B:      {result['new_avg_gap_deg']:.1f} deg")
    print(
        f"Improvement:        {result['improvement_deg']:.1f} deg "
        f"({result['improvement_percent']:.1f}%)"
    )
    print("\nPer-joint changes:")
    for name, data in result["per_joint_improvement"].items():
        print(
            f"  {name:20s}: {data['original_deg']:.1f} deg -> "
            f"{data['new_deg']:.1f} deg ({data['improvement_deg']:+.1f} deg)"
        )


if __name__ == "__main__":
    main()
