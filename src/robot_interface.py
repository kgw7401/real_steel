"""Abstract robot interface. Defines the contract for simulated and real robot implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class JointState:
    positions: np.ndarray  # shape (6,), radians
    velocities: np.ndarray  # shape (6,), rad/s
    timestamp: float  # time.time()


class RobotInterface(ABC):
    NUM_JOINTS = 6
    JOINT_NAMES = [
        "l_shoulder_pan",
        "l_shoulder_tilt",
        "l_elbow",
        "r_shoulder_pan",
        "r_shoulder_tilt",
        "r_elbow",
    ]

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def set_joint_positions(self, positions: np.ndarray) -> None: ...

    @abstractmethod
    def get_joint_state(self) -> JointState: ...

    @abstractmethod
    def home(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...
