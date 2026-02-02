"""Drive a single servo from the camera pipeline over serial.

Usage:
    python tests/test_single_servo.py --port /dev/tty.usbserial-XXXX
    python tests/test_single_servo.py --port /dev/tty.usbserial-XXXX --joint 3  # left elbow
    python tests/test_single_servo.py --port /dev/tty.usbserial-XXXX --manual   # type angles

Joint indices (mapped angles, post-mirror, in degrees):
    0=L_roll  1=L_tilt  2=L_pan  3=L_elbow
    4=R_roll  5=R_tilt  6=R_pan  7=R_elbow

Default joint is 1 (left shoulder tilt) — easy to see movement.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import serial

from src.angle_calculator import AngleCalculator
from src.camera import Camera
from src.motion_mapper import MappingConfig, MotionMapper
from src.pose_estimator import PoseEstimator


JOINT_LABELS = ["L_roll", "L_tilt", "L_pan", "L_elbow",
                "R_roll", "R_tilt", "R_pan", "R_elbow"]


def connect(port: str, baud: int = 115200) -> serial.Serial:
    print(f"Connecting to {port} at {baud} baud...")
    ser = serial.Serial(port, baud, timeout=2)
    time.sleep(2)  # Wait for ESP32 reset

    # Drain any buffered output, then wait for READY.
    # The startup sweep takes ~3.5s, so allow up to 10s total.
    deadline = time.time() + 10
    while time.time() < deadline:
        line = ser.readline().decode(errors="replace").strip()
        if line:
            print(f"  ESP32: {line}")
        if "READY" in line:
            print("Connected.")
            return ser

    raise RuntimeError("Did not receive READY from ESP32 within 10s")


def send_angle(ser: serial.Serial, angle_deg: float) -> str:
    cmd = f"A:{angle_deg:.1f}\n"
    ser.write(cmd.encode())
    return ser.readline().decode(errors="replace").strip()


def run_manual(ser: serial.Serial):
    """Manual mode: type angles in the terminal."""
    print("\nManual mode. Type an angle in degrees (or 'q' to quit, 'h' to home):")
    while True:
        try:
            val = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if val.lower() == "q":
            break
        if val.lower() == "h":
            ser.write(b"H\n")
            print(f"  {ser.readline().decode(errors='replace').strip()}")
            continue
        try:
            angle = float(val)
            resp = send_angle(ser, angle)
            print(f"  {resp}")
        except ValueError:
            print("  Enter a number, 'h', or 'q'")


def run_camera(ser: serial.Serial, joint_idx: int):
    """Camera mode: drive servo from pose estimation pipeline."""
    print(f"\nCamera mode. Driving servo with joint {joint_idx} ({JOINT_LABELS[joint_idx]})")
    print("Press 'q' or ESC to quit.\n")

    camera = Camera(device_id=0, width=640, height=480, fps=30)
    pose_estimator = PoseEstimator(
        model_path="data/pose_landmarker_lite.task", min_visibility=0.5,
    )
    angle_calculator = AngleCalculator(smoothing_factor=0.3)
    motion_mapper = MotionMapper(
        config=MappingConfig(mirror_mode=True, dead_zone=np.deg2rad(3.0))
    )

    if not camera.open():
        print("Failed to open camera")
        return

    last_print = time.time()
    frame_count = 0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            pose = pose_estimator.process(frame.image, frame.timestamp)
            joint_angles = angle_calculator.calculate(pose)

            servo_deg = None
            if joint_angles is not None:
                servo = motion_mapper.map(joint_angles)
                servo_deg = np.rad2deg(servo.angles[joint_idx])
                resp = send_angle(ser, servo_deg)

                if resp != "" and not resp.startswith("OK"):
                    print(f"  Warning: {resp}")

            # Display
            display = pose_estimator.draw(frame.image, pose)

            if servo_deg is not None:
                cv2.putText(display, f"{JOINT_LABELS[joint_idx]}: {servo_deg:+.1f} deg",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No pose detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Single Servo Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

            # Stats
            frame_count += 1
            now = time.time()
            if now - last_print >= 3.0:
                fps = frame_count / (now - last_print)
                angle_str = f"{servo_deg:+.1f}" if servo_deg is not None else "---"
                print(f"  FPS: {fps:.1f}  |  {JOINT_LABELS[joint_idx]}: {angle_str} deg")
                frame_count = 0
                last_print = now

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        camera.release()
        pose_estimator.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Single servo test via ESP32")
    parser.add_argument("--port", required=True, help="Serial port (e.g. /dev/tty.usbserial-XXXX)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--joint", type=int, default=1,
                        help="Joint index to drive (0-7, default: 1 = L_tilt)")
    parser.add_argument("--manual", action="store_true",
                        help="Manual mode: type angles instead of camera")
    args = parser.parse_args()

    if args.joint < 0 or args.joint > 7:
        print(f"Invalid joint index {args.joint}, must be 0-7")
        sys.exit(1)

    ser = connect(args.port, args.baud)

    try:
        if args.manual:
            run_manual(ser)
        else:
            run_camera(ser, args.joint)
    finally:
        print("Homing and closing...")
        ser.write(b"H\n")
        ser.readline()
        ser.close()
        print("Done.")


if __name__ == "__main__":
    main()
