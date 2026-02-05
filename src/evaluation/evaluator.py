"""Evaluator. Computes Gap A (detection accuracy) and Gap B (execution accuracy) metrics."""

from dataclasses import dataclass, field

import numpy as np

# Canonical joint order (matches RobotInterface.JOINT_NAMES)
JOINT_NAMES = [
    "l_shoulder_roll",
    "l_shoulder_tilt",
    "l_shoulder_pan",
    "l_elbow",
    "r_shoulder_roll",
    "r_shoulder_tilt",
    "r_shoulder_pan",
    "r_elbow",
]

JOINT_NAME_TO_IDX = {name: i for i, name in enumerate(JOINT_NAMES)}


@dataclass
class GapMetrics:
    per_joint_deg: np.ndarray  # (8,) error per joint in degrees
    rmse_deg: float  # overall RMSE in degrees
    max_error_deg: float  # worst single-joint error
    worst_joint_idx: int  # index of worst joint
    accuracy_percent: float  # 100 - (rmse / tolerance) * 100, clamped [0, 100]


@dataclass
class MotionEvalResult:
    motion_name: str
    keyframe_errors: dict  # {phase_name: GapMetrics}
    trajectory_pass: dict  # {constraint_desc: bool}
    timing_pass: bool
    overall_score: float  # 0-100


class Evaluator:
    def __init__(self, tolerance_deg: float = 15.0):
        self.tolerance_deg = tolerance_deg
        self._running_sum = np.zeros(8)
        self._running_count = 0

    def compute_gap_b(
        self, robot_cmd: np.ndarray, robot_actual: np.ndarray
    ) -> GapMetrics:
        error_rad = np.abs(robot_cmd - robot_actual)
        error_deg = np.rad2deg(error_rad)
        rmse = float(np.sqrt(np.mean(error_deg**2)))
        max_err = float(np.max(error_deg))
        worst_idx = int(np.argmax(error_deg))
        accuracy = max(0.0, min(100.0, 100.0 - (rmse / self.tolerance_deg) * 100.0))
        return GapMetrics(
            per_joint_deg=error_deg,
            rmse_deg=rmse,
            max_error_deg=max_err,
            worst_joint_idx=worst_idx,
            accuracy_percent=accuracy,
        )

    def evaluate_static_pose(
        self,
        human_angles: np.ndarray,
        expected_angles_deg: dict,
        tolerance_deg: float,
    ) -> GapMetrics:
        human_deg = np.rad2deg(human_angles)
        errors = np.zeros(8)
        checked = np.zeros(8, dtype=bool)

        for name, expected_val in expected_angles_deg.items():
            idx = JOINT_NAME_TO_IDX.get(name)
            if idx is None:
                continue
            errors[idx] = abs(human_deg[idx] - expected_val)
            checked[idx] = True

        # Only use checked joints for aggregate metrics
        if not np.any(checked):
            return GapMetrics(
                per_joint_deg=errors,
                rmse_deg=0.0,
                max_error_deg=0.0,
                worst_joint_idx=0,
                accuracy_percent=100.0,
            )

        checked_errors = errors[checked]
        rmse = float(np.sqrt(np.mean(checked_errors**2)))
        max_err = float(np.max(checked_errors))
        worst_idx = int(np.argmax(errors))
        accuracy = max(0.0, min(100.0, 100.0 - (rmse / tolerance_deg) * 100.0))

        return GapMetrics(
            per_joint_deg=errors,
            rmse_deg=rmse,
            max_error_deg=max_err,
            worst_joint_idx=worst_idx,
            accuracy_percent=accuracy,
        )

    def evaluate_motion_sequence(
        self,
        recorded_angles: list[np.ndarray],
        timestamps: list[float],
        motion_def: dict,
    ) -> MotionEvalResult:
        if len(timestamps) < 2:
            return MotionEvalResult(
                motion_name=motion_def.get("description", "unknown"),
                keyframe_errors={},
                trajectory_pass={},
                timing_pass=False,
                overall_score=0.0,
            )

        t0, t1 = timestamps[0], timestamps[-1]
        duration = t1 - t0
        # Normalize timestamps to [0, 1]
        t_norm = [(t - t0) / duration if duration > 0 else 0.0 for t in timestamps]

        # Evaluate keyframes
        keyframe_errors = {}
        keyframe_scores = []
        for kf in motion_def.get("keyframes", []):
            ratio = kf["time_ratio"]
            # Find closest frame
            closest_idx = int(np.argmin([abs(tn - ratio) for tn in t_norm]))
            tolerance = kf.get("tolerance_deg", self.tolerance_deg)
            gap = self.evaluate_static_pose(
                recorded_angles[closest_idx],
                kf["angles_deg"],
                tolerance,
            )
            phase = kf.get("phase", f"t={ratio}")
            keyframe_errors[phase] = gap
            keyframe_scores.append(gap.accuracy_percent)

        # Evaluate trajectory constraints
        trajectory_pass = {}
        for constraint in motion_def.get("trajectory_constraints", []):
            joint_name = constraint["joint"]
            pattern = constraint["pattern"]
            idx = JOINT_NAME_TO_IDX.get(joint_name)
            if idx is None:
                continue

            series_deg = np.array(
                [np.rad2deg(angles[idx]) for angles in recorded_angles]
            )
            desc = f"{joint_name}:{pattern}"

            if pattern == "decrease_then_increase":
                trajectory_pass[desc] = self._check_decrease_then_increase(series_deg)
            elif pattern == "increase_then_decrease":
                trajectory_pass[desc] = self._check_increase_then_decrease(series_deg)
            elif pattern == "stable":
                stable_range = constraint.get("stable_range_deg", 20)
                trajectory_pass[desc] = self._check_stable(series_deg, stable_range)
            else:
                trajectory_pass[desc] = False

        # Evaluate timing
        timing_range = motion_def.get("timing", {}).get("duration_range_sec", [0, 999])
        timing_pass = timing_range[0] <= duration <= timing_range[1]

        # Compute overall score
        kf_score = np.mean(keyframe_scores) if keyframe_scores else 100.0
        traj_count = len(trajectory_pass)
        traj_score = (
            (sum(trajectory_pass.values()) / traj_count * 100.0) if traj_count else 100.0
        )
        timing_score = 100.0 if timing_pass else 50.0

        # Weighted: 50% keyframe accuracy, 30% trajectory, 20% timing
        overall = 0.5 * kf_score + 0.3 * traj_score + 0.2 * timing_score

        return MotionEvalResult(
            motion_name=motion_def.get("description", "unknown"),
            keyframe_errors=keyframe_errors,
            trajectory_pass=trajectory_pass,
            timing_pass=timing_pass,
            overall_score=round(overall, 1),
        )

    def update_running_stats(self, gap_b: GapMetrics) -> GapMetrics:
        self._running_sum += gap_b.per_joint_deg
        self._running_count += 1
        return self.get_running_stats()

    def get_running_stats(self) -> GapMetrics:
        if self._running_count == 0:
            return GapMetrics(
                per_joint_deg=np.zeros(8),
                rmse_deg=0.0,
                max_error_deg=0.0,
                worst_joint_idx=0,
                accuracy_percent=100.0,
            )
        avg = self._running_sum / self._running_count
        rmse = float(np.sqrt(np.mean(avg**2)))
        max_err = float(np.max(avg))
        worst_idx = int(np.argmax(avg))
        accuracy = max(0.0, min(100.0, 100.0 - (rmse / self.tolerance_deg) * 100.0))
        return GapMetrics(
            per_joint_deg=avg,
            rmse_deg=rmse,
            max_error_deg=max_err,
            worst_joint_idx=worst_idx,
            accuracy_percent=accuracy,
        )

    def reset_running_stats(self) -> None:
        self._running_sum = np.zeros(8)
        self._running_count = 0

    @staticmethod
    def _check_decrease_then_increase(series: np.ndarray) -> bool:
        if len(series) < 3:
            return False
        min_idx = int(np.argmin(series))
        # Minimum should not be at the very start or end
        if min_idx == 0 or min_idx == len(series) - 1:
            return False
        # Check that values generally decrease before min and increase after
        before_min = series[0] > series[min_idx]
        after_min = series[-1] > series[min_idx]
        return before_min and after_min

    @staticmethod
    def _check_increase_then_decrease(series: np.ndarray) -> bool:
        if len(series) < 3:
            return False
        max_idx = int(np.argmax(series))
        if max_idx == 0 or max_idx == len(series) - 1:
            return False
        before_max = series[0] < series[max_idx]
        after_max = series[-1] < series[max_idx]
        return before_max and after_max

    @staticmethod
    def _check_stable(series: np.ndarray, range_deg: float) -> bool:
        if len(series) == 0:
            return True
        mean_val = np.mean(series)
        return bool(np.all(np.abs(series - mean_val) <= range_deg))
