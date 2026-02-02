"""Pose estimation module. Detects human body landmarks using MediaPipe."""

from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision


@dataclass
class Point3D:
    x: float  # normalized [0, 1]
    y: float
    z: float
    visibility: float
    world_x: float  # meters, from pose_world_landmarks
    world_y: float
    world_z: float


@dataclass
class PoseResult:
    keypoints: dict[str, Point3D]
    is_valid: bool
    timestamp: float

    def get_point(self, name: str) -> Point3D | None:
        return self.keypoints.get(name)


class PoseEstimator:
    KEYPOINT_INDICES: dict[int, str] = {
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
    }

    BONES = [
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "right_shoulder"),
    ]

    def __init__(self, model_path: str, min_visibility: float = 0.3):
        self.min_visibility = min_visibility
        self._last_valid_result: PoseResult | None = None
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_poses=1,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def process(self, image: np.ndarray, timestamp: float) -> PoseResult:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.landmarker.detect(mp_image)

        keypoints: dict[str, Point3D] = {}
        is_valid = False

        if result.pose_landmarks and result.pose_world_landmarks:
            landmarks = result.pose_landmarks[0]
            world_landmarks = result.pose_world_landmarks[0]

            for idx, name in self.KEYPOINT_INDICES.items():
                lm = landmarks[idx]
                wlm = world_landmarks[idx]
                keypoints[name] = Point3D(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility,
                    world_x=wlm.x,
                    world_y=wlm.y,
                    world_z=wlm.z,
                )

            is_valid = all(
                keypoints[name].visibility >= self.min_visibility
                for name in self.KEYPOINT_INDICES.values()
            )

        result = PoseResult(
            keypoints=keypoints,
            is_valid=is_valid,
            timestamp=timestamp,
        )

        if is_valid:
            self._last_valid_result = result
            return result

        # Hold last valid pose when detection fails
        if self._last_valid_result is not None:
            return PoseResult(
                keypoints=self._last_valid_result.keypoints,
                is_valid=True,
                timestamp=timestamp,
            )

        return result

    def draw(self, image: np.ndarray, pose: PoseResult) -> np.ndarray:
        output = image.copy()
        if not pose.keypoints:
            return output

        h, w = image.shape[:2]

        # Draw keypoints
        for name, point in pose.keypoints.items():
            if point.visibility >= self.min_visibility:
                cx, cy = int(point.x * w), int(point.y * h)
                color = (0, 255, 0) if point.visibility > 0.8 else (0, 165, 255)
                cv2.circle(output, (cx, cy), 5, color, -1)
                label = name.split("_")[0][0].upper()
                cv2.putText(
                    output,
                    label,
                    (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

        # Draw bones
        for start_name, end_name in self.BONES:
            start = pose.keypoints.get(start_name)
            end = pose.keypoints.get(end_name)
            if (
                start
                and end
                and start.visibility >= self.min_visibility
                and end.visibility >= self.min_visibility
            ):
                pt1 = (int(start.x * w), int(start.y * h))
                pt2 = (int(end.x * w), int(end.y * h))
                cv2.line(output, pt1, pt2, (0, 255, 0), 2)

        return output

    def close(self) -> None:
        self.landmarker.close()
