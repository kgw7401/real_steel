"""Session recorder. Streams frame data to JSONL and writes session metadata."""

import json
import time
from pathlib import Path

import numpy as np


class Recorder:
    def __init__(self, output_dir: str = "data/sessions"):
        self.output_dir = Path(output_dir)
        self._session_id: str | None = None
        self._session_dir: Path | None = None
        self._file = None
        self._config_snapshot: dict = {}
        self._start_time: float = 0.0
        # Running accumulators (avoid re-reading JSONL at end)
        self._frame_count: int = 0
        self._sum_gap_b_rmse: float = 0.0
        self._sum_latency: float = 0.0
        self._sum_confidence: float = 0.0

    def start_session(self, config_snapshot: dict) -> str:
        self._session_id = time.strftime("%Y-%m-%d_%H-%M-%S")
        self._session_dir = self.output_dir / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._file = open(self._session_dir / "frames.jsonl", "w")
        self._config_snapshot = config_snapshot
        self._start_time = time.time()
        self._frame_count = 0
        self._sum_gap_b_rmse = 0.0
        self._sum_latency = 0.0
        self._sum_confidence = 0.0

        return self._session_id

    def record_frame(
        self,
        timestamp: float,
        frame_number: int,
        keypoints: dict | None,
        pose_confidence: float,
        human_angles: np.ndarray,
        robot_cmd: np.ndarray,
        robot_actual: np.ndarray,
        latency_ms: float,
        motion_label: str | None = None,
        motion_phase: str | None = None,
    ) -> None:
        if self._file is None:
            raise RuntimeError("No active session. Call start_session() first.")

        gap_b = np.abs(robot_cmd - robot_actual)
        gap_b_rmse = float(np.sqrt(np.mean(gap_b**2)))

        # Serialize keypoints: {name: [wx, wy, wz, vis], ...}
        kp_data = None
        if keypoints is not None:
            kp_data = {}
            for name, pt in keypoints.items():
                kp_data[name] = [pt.world_x, pt.world_y, pt.world_z, pt.visibility]

        frame = {
            "timestamp": timestamp,
            "frame_number": frame_number,
            "keypoints": kp_data,
            "pose_confidence": pose_confidence,
            "human_angles": human_angles.tolist(),
            "robot_cmd": robot_cmd.tolist(),
            "robot_actual": robot_actual.tolist(),
            "gap_b": gap_b.tolist(),
            "gap_b_rmse": gap_b_rmse,
            "latency_ms": latency_ms,
            "motion_label": motion_label,
            "motion_phase": motion_phase,
        }

        self._file.write(json.dumps(frame) + "\n")

        # Update accumulators
        self._frame_count += 1
        self._sum_gap_b_rmse += gap_b_rmse
        self._sum_latency += latency_ms
        self._sum_confidence += pose_confidence

    def end_session(self) -> Path:
        if self._file is None or self._session_dir is None:
            raise RuntimeError("No active session.")

        self._file.close()
        self._file = None

        duration = time.time() - self._start_time
        n = max(self._frame_count, 1)

        metadata = {
            "session_id": self._session_id,
            "mode": "simulation",
            "duration_sec": round(duration, 1),
            "total_frames": self._frame_count,
            "config_snapshot": self._config_snapshot,
            "avg_gap_b_rmse_deg": round(np.rad2deg(self._sum_gap_b_rmse / n), 2),
            "avg_latency_ms": round(self._sum_latency / n, 2),
            "avg_pose_confidence": round(self._sum_confidence / n, 2),
        }

        with open(self._session_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return self._session_dir
