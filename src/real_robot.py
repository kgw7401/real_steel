"""Real robot module. Serial/ESP32 implementation of RobotInterface."""

import logging
import time

import numpy as np
import serial

from src.robot_interface import JointState, RobotInterface

log = logging.getLogger(__name__)


class RealRobot(RobotInterface):
    """Controls the physical robot via serial communication with ESP32 + PCA9685."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.serial: serial.Serial | None = None
        self.last_positions: np.ndarray = np.zeros(self.NUM_JOINTS)

    def connect(self) -> bool:
        """Open serial, wait for READY, run safe startup sequence."""
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=self.timeout)
            time.sleep(2)  # Wait for ESP32 reset after USB connect

            # Wait for READY (firmware may print boot/sweep messages first)
            if not self._wait_for_ready(timeout=10.0):
                print(f"ESP32 on {self.port} did not send READY")
                self.serial.close()
                self.serial = None
                return False

            # Safe startup: disable -> home -> enable
            self._send("E:0")
            self._send("H")
            self._send("E:1")
            self.last_positions = np.zeros(self.NUM_JOINTS)

            print(f"Connected to ESP32 on {self.port}")
            return True

        except serial.SerialException as e:
            print(f"Failed to connect to {self.port}: {e}")
            self.serial = None
            return False

    def disconnect(self) -> None:
        if self.serial is not None and self.serial.is_open:
            try:
                self._send("E:0")
                self._send("H")
            except serial.SerialException:
                pass  # Best effort on shutdown
            self.serial.close()
        self.serial = None

    def set_joint_positions(self, positions: np.ndarray) -> None:
        if len(positions) != self.NUM_JOINTS:
            raise ValueError(f"Expected {self.NUM_JOINTS} positions, got {len(positions)}")

        if not self.is_connected():
            return

        # Convert radians to degrees
        deg = np.rad2deg(positions)
        angles_str = ",".join(f"{a:.1f}" for a in deg)
        resp = self._send(f"J:{angles_str}")

        if resp is not None and resp.startswith("ERR"):
            log.warning("set_joint_positions: %s", resp)
        else:
            self.last_positions = positions.copy()

    def get_joint_state(self) -> JointState:
        if not self.is_connected():
            return JointState(
                positions=self.last_positions.copy(),
                velocities=np.zeros(self.NUM_JOINTS),
                timestamp=time.time(),
            )

        resp = self._send("Q")

        if resp is not None and resp.startswith("P:"):
            try:
                angles_deg = [float(x) for x in resp[2:].split(",")]
                if len(angles_deg) == self.NUM_JOINTS:
                    positions = np.deg2rad(angles_deg)
                    self.last_positions = positions
                    return JointState(
                        positions=positions,
                        velocities=np.zeros(self.NUM_JOINTS),
                        timestamp=time.time(),
                    )
            except ValueError:
                log.warning("Failed to parse position response: %s", resp)

        # Fallback to last known positions
        return JointState(
            positions=self.last_positions.copy(),
            velocities=np.zeros(self.NUM_JOINTS),
            timestamp=time.time(),
        )

    def home(self) -> None:
        self._send("H")
        self.last_positions = np.zeros(self.NUM_JOINTS)

    def is_connected(self) -> bool:
        return self.serial is not None and self.serial.is_open

    # --- Internal helpers ---

    def _send(self, cmd: str) -> str | None:
        """Send a command and return the response line. Returns None on error."""
        if not self.is_connected():
            return None
        try:
            self.serial.write(f"{cmd}\n".encode())
            resp = self.serial.readline().decode(errors="replace").strip()
            return resp
        except serial.SerialTimeoutException:
            log.warning("Serial timeout sending: %s", cmd)
            return None
        except serial.SerialException as e:
            log.error("Serial error: %s", e)
            self.serial = None
            return None

    def _wait_for_ready(self, timeout: float = 10.0) -> bool:
        """Read lines until READY is found or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self.serial.readline().decode(errors="replace").strip()
                if line:
                    log.debug("ESP32: %s", line)
                if "READY" in line:
                    return True
            except serial.SerialException:
                return False
        return False
