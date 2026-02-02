"""Joint angle extraction module. Computes robot joint angles from pose landmarks."""

from dataclasses import dataclass

import numpy as np

from src.pose_estimator import Point3D, PoseResult


@dataclass
class JointAngles:
    """Joint angles in radians. Order: roll, tilt, pan, elbow per arm."""

    left_shoulder_roll: float
    left_shoulder_tilt: float
    left_shoulder_pan: float
    left_elbow: float
    right_shoulder_roll: float
    right_shoulder_tilt: float
    right_shoulder_pan: float
    right_elbow: float
    timestamp: float
    valid: np.ndarray  # shape (8,) bool

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.left_shoulder_roll,
                self.left_shoulder_tilt,
                self.left_shoulder_pan,
                self.left_elbow,
                self.right_shoulder_roll,
                self.right_shoulder_tilt,
                self.right_shoulder_pan,
                self.right_elbow,
            ]
        )


class AngleCalculator:
    # Below this threshold, elbow angle snaps to 0.
    # Compensates for MediaPipe depth noise on straight arms (~25-45 deg noise).
    ELBOW_DEAD_ZONE = 0.8  # ~46 degrees

    # When the horizontal projection of the upper arm is less than this fraction
    # of the full arm length, pan angle is geometrically ill-defined (arm nearly vertical).
    # Pan is linearly attenuated to 0 below this ratio.
    PAN_HORIZ_THRESHOLD = 0.3

    KEYPOINT_ORDER = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
    ]

    def __init__(self, smoothing_factor: float = 0.3):
        self.smoothing_factor = smoothing_factor
        self.prev_angles: JointAngles | None = None

    def calculate(self, pose: PoseResult) -> JointAngles | None:
        if not pose.is_valid:
            return None

        pts = np.array([
            [p.world_x, p.world_y, p.world_z]
            for p in (pose.keypoints[k] for k in self.KEYPOINT_ORDER)
        ])
        ls, rs, le, re, lw, rw = pts[0], pts[1], pts[2], pts[3], pts[4], pts[5]

        angles = JointAngles(
            left_shoulder_roll=self._calc_shoulder_roll(ls, le, rs),
            left_shoulder_tilt=self._calc_shoulder_tilt(ls, le),
            left_shoulder_pan=self._calc_shoulder_pan(ls, le, rs),
            left_elbow=self._calc_elbow_angle(ls, le, lw),
            right_shoulder_roll=self._calc_shoulder_roll(rs, re, ls),
            right_shoulder_tilt=self._calc_shoulder_tilt(rs, re),
            right_shoulder_pan=self._calc_shoulder_pan(rs, re, ls),
            right_elbow=self._calc_elbow_angle(rs, re, rw),
            timestamp=pose.timestamp,
            valid=np.ones(8, dtype=bool),
        )

        if self.prev_angles is not None:
            angles = self._smooth(angles, self.prev_angles)

        self.prev_angles = angles
        return angles

    def _to_world_vec(self, p: Point3D) -> np.ndarray:
        return np.array([p.world_x, p.world_y, p.world_z])

    def _calc_shoulder_roll(
        self, shoulder: np.ndarray, elbow: np.ndarray, other_shoulder: np.ndarray
    ) -> float:
        """Abduction angle — how far the arm is spread from the body.
        0=arm hanging at side, +ve=arm spread outward (toward T-pose)."""
        upper_arm = elbow - shoulder

        # Outward direction in the frontal plane (perpendicular to forward axis)
        outward = shoulder - other_shoulder
        outward[2] = 0  # zero out Z (depth) — stay in frontal plane
        out_norm = np.linalg.norm(outward)
        if out_norm < 1e-6:
            return 0.0
        outward = outward / out_norm

        # Project upper arm onto frontal plane (ignore Z/depth)
        arm_frontal = upper_arm.copy()
        arm_frontal[2] = 0

        # Downward component (Y points down in our convention)
        down_component = upper_arm[1]  # positive = downward
        # Outward component
        out_component = np.dot(arm_frontal, outward)

        # Angle from downward: 0 = hanging, pi/2 = spread horizontal
        angle = float(np.arctan2(out_component, down_component))
        return max(angle, 0.0)  # clamp negative (arm crossing body) to 0

    def _calc_shoulder_pan(
        self, shoulder: np.ndarray, elbow: np.ndarray, other_shoulder: np.ndarray
    ) -> float:
        """Horizontal rotation of upper arm from resting (arm at side).
        0=arm at side, +ve=forward (left arm) or -ve=forward (right arm)."""
        # Outward direction: points away from torso center, projected onto XZ plane
        torso_dir = shoulder - other_shoulder
        torso_dir[1] = 0  # zero out Y (vertical)
        norm = np.linalg.norm(torso_dir)
        if norm < 1e-6:
            return 0.0
        torso_dir = torso_dir / norm

        # Upper arm projected onto horizontal plane
        full_arm = elbow - shoulder
        full_length = np.linalg.norm(full_arm)
        upper_arm = full_arm.copy()
        upper_arm[1] = 0
        horiz_length = np.linalg.norm(upper_arm)
        if horiz_length < 1e-6:
            return 0.0
        upper_arm = upper_arm / horiz_length

        dot = np.clip(np.dot(torso_dir, upper_arm), -1.0, 1.0)
        angle = float(np.arccos(dot))

        # Sign from cross product Y component
        cross = np.cross(torso_dir, upper_arm)
        if cross[1] < 0:
            angle = -angle

        # Attenuate pan when arm is nearly vertical (pan is geometrically ill-defined).
        # horiz_ratio → 0 when arm hangs straight down, → 1 when arm is horizontal.
        horiz_ratio = horiz_length / max(full_length, 1e-6)
        if horiz_ratio < self.PAN_HORIZ_THRESHOLD:
            angle *= horiz_ratio / self.PAN_HORIZ_THRESHOLD

        return angle

    # MediaPipe Z (depth) has ~3x more noise than X/Y.
    # Reduce Z's contribution to horizontal distance in tilt calculation.
    DEPTH_WEIGHT = 0.3

    def _calc_shoulder_tilt(
        self, shoulder: np.ndarray, elbow: np.ndarray
    ) -> float:
        """Vertical angle of upper arm from horizontal. 0=horizontal, +ve=raised."""
        upper_arm = elbow - shoulder
        # Down-weight Z (depth) to reduce noise-driven angle compression
        horizontal_dist = np.sqrt(
            upper_arm[0] ** 2 + (self.DEPTH_WEIGHT * upper_arm[2]) ** 2
        )
        # MediaPipe world Y points down, so negate
        angle = float(np.arctan2(-upper_arm[1], horizontal_dist))
        return angle

    def _calc_elbow_angle(
        self, shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray
    ) -> float:
        """Elbow flexion. 0=straight, +ve=bent."""
        upper_arm = shoulder - elbow  # reversed: points from elbow toward shoulder
        forearm = wrist - elbow

        ua_norm = np.linalg.norm(upper_arm)
        fa_norm = np.linalg.norm(forearm)
        if ua_norm < 1e-6 or fa_norm < 1e-6:
            return 0.0

        upper_arm = upper_arm / ua_norm
        forearm = forearm / fa_norm

        dot = np.clip(np.dot(upper_arm, forearm), -1.0, 1.0)
        angle_between = float(np.arccos(dot))

        # Flexion = pi - angle_between (0 when straight, pi when fully folded)
        flexion = np.pi - angle_between

        # Dead zone: snap small angles to 0 (MediaPipe depth noise on straight arms)
        if flexion < self.ELBOW_DEAD_ZONE:
            return 0.0
        return flexion

    def _smooth(self, current: JointAngles, previous: JointAngles) -> JointAngles:
        alpha = self.smoothing_factor
        cur = current.to_array()
        prev = previous.to_array()
        smoothed = alpha * prev + (1 - alpha) * cur
        return JointAngles(
            *smoothed,
            timestamp=current.timestamp,
            valid=current.valid & previous.valid,
        )

    def reset(self) -> None:
        self.prev_angles = None
