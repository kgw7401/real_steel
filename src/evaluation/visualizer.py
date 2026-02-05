"""Evaluation visualizer. Draws real-time accuracy overlays on camera frames."""

import cv2
import numpy as np

from src.evaluation.evaluator import JOINT_NAMES, GapMetrics

# Short labels for display
_JOINT_LABELS = ["L_roll", "L_tilt", "L_pan", "L_elbow", "R_roll", "R_tilt", "R_pan", "R_elbow"]

_PANEL_WIDTH = 200
_GREEN = (0, 200, 0)
_YELLOW = (0, 200, 200)
_RED = (0, 0, 200)
_WHITE = (255, 255, 255)
_DARK_BG = (30, 30, 30)


def _error_color(deg: float) -> tuple[int, int, int]:
    if deg < 5.0:
        return _GREEN
    elif deg < 10.0:
        return _YELLOW
    return _RED


class EvalVisualizer:
    def draw_gap_overlay(self, frame: np.ndarray, gap_b: GapMetrics) -> np.ndarray:
        h, w = frame.shape[:2]

        # Create panel on the right
        panel = np.full((h, _PANEL_WIDTH, 3), _DARK_BG, dtype=np.uint8)

        y = 30
        # Header
        cv2.putText(
            panel, f"GAP B: {gap_b.rmse_deg:.1f} deg",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1,
        )
        y += 10
        cv2.line(panel, (10, y), (_PANEL_WIDTH - 10, y), _WHITE, 1)
        y += 25

        # Per-joint error bars
        bar_max_width = _PANEL_WIDTH - 110
        for i, label in enumerate(_JOINT_LABELS):
            err = gap_b.per_joint_deg[i]
            color = _error_color(err)

            # Label
            cv2.putText(
                panel, f"{label}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _WHITE, 1,
            )

            # Bar (scale: 15deg = full width)
            bar_len = min(int(err / 15.0 * bar_max_width), bar_max_width)
            bar_x = 80
            cv2.rectangle(panel, (bar_x, y - 10), (bar_x + bar_len, y), color, -1)

            # Value
            cv2.putText(
                panel, f"{err:.1f}", (bar_x + bar_max_width + 5, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
            )
            y += 25

        # Accuracy
        y += 10
        acc_color = _error_color(15.0 - gap_b.accuracy_percent / 100.0 * 15.0)
        cv2.putText(
            panel, f"ACC: {gap_b.accuracy_percent:.0f}%",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, acc_color, 2,
        )

        return np.hstack([frame, panel])

    def draw_test_guide(
        self, frame: np.ndarray, instruction: str, countdown: float
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        output = frame.copy()

        # Banner at top
        banner_h = 50
        cv2.rectangle(output, (0, 0), (w, banner_h), _DARK_BG, -1)

        text = f"{instruction}    {countdown:.1f}s"
        cv2.putText(
            output, text,
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, _WHITE, 2,
        )

        return output
