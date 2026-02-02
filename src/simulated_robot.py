"""Simulated robot module. PyBullet implementation of RobotInterface using URDF model."""

import time

import numpy as np
import pybullet as p
import pybullet_data

from src.robot_interface import JointState, RobotInterface


class SimulatedRobot(RobotInterface):
    """PyBullet simulation implementation of RobotInterface."""

    # URDF joint names in our canonical order
    _URDF_JOINT_NAMES = [
        "l_shoulder_roll_joint",
        "l_shoulder_tilt_joint",
        "l_shoulder_pan_joint",
        "l_elbow_joint",
        "r_shoulder_roll_joint",
        "r_shoulder_tilt_joint",
        "r_shoulder_pan_joint",
        "r_elbow_joint",
    ]

    def __init__(
        self, urdf_path: str, gui: bool = True, timestep: float = 1 / 240
    ):
        self.urdf_path = urdf_path
        self.gui = gui
        self.timestep = timestep
        self.client: int | None = None
        self.robot_id: int | None = None
        self.joint_indices: list[int] = []

        # Control parameters
        self.max_force = 50.0
        self.position_gain = 0.8
        self.velocity_gain = 1.0

        # Real-time stepping: accumulate wall-clock time and step physics to match
        self._last_step_time: float | None = None

    def connect(self) -> bool:
        try:
            self.client = p.connect(p.GUI if self.gui else p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.8)
            p.setTimeStep(self.timestep)

            self.robot_id = p.loadURDF(
                self.urdf_path,
                basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,
            )

            self._map_joints()
            self.home()

            if self.gui:
                p.resetDebugVisualizerCamera(
                    cameraDistance=0.6,
                    cameraYaw=0,
                    cameraPitch=-15,
                    cameraTargetPosition=[0, 0, 0.55],
                )

            return True

        except Exception as e:
            print(f"Failed to connect to simulation: {e}")
            return False

    def _map_joints(self):
        """Map our joint names to PyBullet joint indices."""
        joint_name_to_index = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            name = info[1].decode("utf-8")
            joint_name_to_index[name] = i

        self.joint_indices = []
        for name in self._URDF_JOINT_NAMES:
            if name in joint_name_to_index:
                self.joint_indices.append(joint_name_to_index[name])
            else:
                raise ValueError(f"Joint {name} not found in URDF")

    def disconnect(self) -> None:
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None

    def set_joint_positions(self, positions: np.ndarray) -> None:
        if len(positions) != self.NUM_JOINTS:
            raise ValueError(
                f"Expected {self.NUM_JOINTS} positions, got {len(positions)}"
            )

        for i, pos in enumerate(positions):
            p.setJointMotorControl2(
                self.robot_id,
                self.joint_indices[i],
                p.POSITION_CONTROL,
                targetPosition=float(pos),
                force=self.max_force,
                positionGain=self.position_gain,
                velocityGain=self.velocity_gain,
            )

    def get_joint_state(self) -> JointState:
        positions = []
        velocities = []

        for idx in self.joint_indices:
            state = p.getJointState(self.robot_id, idx)
            positions.append(state[0])
            velocities.append(state[1])

        return JointState(
            positions=np.array(positions),
            velocities=np.array(velocities),
            timestamp=time.time(),
        )

    def home(self) -> None:
        self.set_joint_positions(np.zeros(self.NUM_JOINTS))
        for _ in range(100):
            self.step()

    def step(self):
        """Advance simulation to match wall-clock time."""
        now = time.time()
        if self._last_step_time is None:
            self._last_step_time = now
            p.stepSimulation()
            return

        elapsed = now - self._last_step_time
        # Cap at 50ms to avoid spiral-of-death when the app hitches
        elapsed = min(elapsed, 0.05)
        steps = int(elapsed / self.timestep)
        for _ in range(max(steps, 1)):
            p.stepSimulation()
        self._last_step_time = now

    def is_connected(self) -> bool:
        return self.client is not None and p.isConnected(self.client)

    def add_debug_controls(self):
        """Add GUI sliders for manual joint control."""
        self.sliders = []
        joint_limits_deg = [
            (-20, 135),   # l_roll
            (-90, 90),    # l_tilt
            (-90, 90),    # l_pan
            (0, 135),     # l_elbow
            (-20, 135),   # r_roll
            (-90, 90),    # r_tilt
            (-90, 90),    # r_pan
            (0, 135),     # r_elbow
        ]

        for i, name in enumerate(self.JOINT_NAMES):
            low, high = joint_limits_deg[i]
            slider = p.addUserDebugParameter(name, low, high, 0)
            self.sliders.append(slider)

        self.reset_button = p.addUserDebugParameter("RESET", 1, 0, 0)

    def read_debug_controls(self) -> np.ndarray:
        """Read current slider values and return as radians."""
        positions_deg = [p.readUserDebugParameter(s) for s in self.sliders]
        return np.deg2rad(positions_deg)
