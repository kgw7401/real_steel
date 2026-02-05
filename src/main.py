"""Main entry point. Runs the shadow boxing control loop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time

import cv2
import numpy as np
import yaml

from src.angle_calculator import AngleCalculator
from src.camera import Camera
from src.motion_mapper import MappingConfig, MotionMapper
from src.pose_estimator import PoseEstimator
from src.profiler import PipelineProfiler


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        print(f"Config not found: {path}, using defaults")
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real Steel — Shadow Boxing Robot")
    parser.add_argument("--sim", action="store_true", help="Use PyBullet simulation")
    parser.add_argument(
        "--config", default="config/settings.yaml", help="Config file path"
    )
    parser.add_argument("--port", default=None, help="Serial port override")
    parser.add_argument(
        "--no-viz", action="store_true", help="Disable camera visualization"
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="Run calibration test mode"
    )
    parser.add_argument(
        "--record", action="store_true", help="Enable data recording to JSONL"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Enable real-time Gap B display overlay"
    )
    parser.add_argument(
        "--run-test",
        choices=["static", "dynamic", "all"],
        default=None,
        help="Run guided test sequence",
    )
    return parser.parse_args()


class TestRunner:
    """Guides user through static poses and dynamic motion sequences for evaluation."""

    def __init__(self, evaluator, recorder, eval_viz, camera, pose_estimator,
                 angle_calculator, motion_mapper, robot, use_sim, no_viz):
        self.evaluator = evaluator
        self.recorder = recorder
        self.eval_viz = eval_viz
        self.camera = camera
        self.pose_estimator = pose_estimator
        self.angle_calculator = angle_calculator
        self.motion_mapper = motion_mapper
        self.robot = robot
        self.use_sim = use_sim
        self.no_viz = no_viz
        self.results = {}

    def run_static_poses(self, config_path: str) -> dict:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        poses = config.get("static_poses", {})
        results = {}

        for pose_name, pose_def in poses.items():
            description = pose_def["description"]
            hold_seconds = pose_def.get("hold_seconds", 3)
            expected = pose_def["expected_angles_deg"]
            tolerance = pose_def.get("tolerance_deg", 15)

            print(f"\n=== Static Pose: {pose_name} ===")
            print(f"    {description}")
            print(f"    Hold for {hold_seconds} seconds...")

            # Collect frames during hold period
            collected_angles = []
            start = time.time()
            while time.time() - start < hold_seconds:
                human_angles, _ = self._process_one_frame(
                    instruction=f"HOLD: {pose_name.upper().replace('_', ' ')}",
                    countdown=hold_seconds - (time.time() - start),
                )
                if human_angles is not None:
                    collected_angles.append(human_angles)

            if not collected_angles:
                print(f"    SKIP (no pose detected)")
                continue

            # Evaluate using average angles during hold
            avg_angles = np.mean(collected_angles, axis=0)
            gap = self.evaluator.evaluate_static_pose(avg_angles, expected, tolerance)
            results[pose_name] = {
                "rmse_deg": gap.rmse_deg,
                "max_error_deg": gap.max_error_deg,
                "accuracy_percent": gap.accuracy_percent,
                "pass": gap.accuracy_percent >= 50.0,
            }
            status = "PASS" if results[pose_name]["pass"] else "FAIL"
            print(f"    {status} — RMSE: {gap.rmse_deg:.1f} deg, Accuracy: {gap.accuracy_percent:.0f}%")

        self.results["static_poses"] = results
        return results

    def run_motion_sequences(self, config_path: str) -> dict:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        sequences = config.get("motion_sequences", {})
        results = {}

        for seq_name, seq_def in sequences.items():
            description = seq_def["description"]
            timing = seq_def.get("timing", {})
            max_duration = timing.get("duration_range_sec", [0, 3])[1]
            capture_duration = max_duration + 1.0  # extra buffer

            print(f"\n=== Motion: {seq_name} ===")
            print(f"    {description}")
            print(f"    Perform the motion when ready (capturing for {capture_duration:.0f}s)...")

            # Collect frames during motion period
            collected_angles = []
            timestamps = []
            start = time.time()
            while time.time() - start < capture_duration:
                human_angles, ts = self._process_one_frame(
                    instruction=f"PERFORM: {seq_name.upper().replace('_', ' ')}",
                    countdown=capture_duration - (time.time() - start),
                )
                if human_angles is not None:
                    collected_angles.append(human_angles)
                    timestamps.append(ts)

            if len(collected_angles) < 3:
                print(f"    SKIP (too few frames)")
                continue

            eval_result = self.evaluator.evaluate_motion_sequence(
                collected_angles, timestamps, seq_def
            )
            results[seq_name] = {
                "overall_score": eval_result.overall_score,
                "timing_pass": eval_result.timing_pass,
                "trajectory_pass": eval_result.trajectory_pass,
                "pass": eval_result.overall_score >= 50.0,
            }
            status = "PASS" if results[seq_name]["pass"] else "FAIL"
            print(f"    {status} — Score: {eval_result.overall_score:.0f}/100")

        self.results["motion_sequences"] = results
        return results

    def save_report(self, output_dir: str) -> Path:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filename = f"benchmark_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        report_path = out_path / filename
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nBenchmark report saved to {report_path}")
        return report_path

    def _process_one_frame(self, instruction: str = "", countdown: float = 0.0):
        """Run one iteration of the pipeline. Returns (human_angles_array, timestamp) or (None, None)."""
        frame, is_new = self.camera.read()
        if frame is None or not is_new:
            if self.use_sim and self.robot is not None:
                self.robot.step()
            return None, None

        pose = self.pose_estimator.process(frame.image, frame.timestamp)
        joint_angles = self.angle_calculator.calculate(pose)

        servo_angles = None
        if joint_angles is not None and self.robot is not None:
            servo_angles = self.motion_mapper.map(joint_angles)
            self.robot.set_joint_positions(servo_angles.angles)

        if self.use_sim and self.robot is not None:
            self.robot.step()

        human_angles = joint_angles.to_array() if joint_angles is not None else None

        # Record if active
        if self.recorder is not None and joint_angles is not None and servo_angles is not None:
            robot_actual = self.robot.get_joint_state().positions
            self.recorder.record_frame(
                timestamp=frame.timestamp,
                frame_number=frame.frame_number,
                keypoints=pose.keypoints if pose.is_valid else None,
                pose_confidence=np.mean([p.visibility for p in pose.keypoints.values()]) if pose.keypoints else 0.0,
                human_angles=joint_angles.to_array(),
                robot_cmd=servo_angles.angles,
                robot_actual=robot_actual,
                latency_ms=0.0,
            )

        # Visualization
        if not self.no_viz:
            display = self.pose_estimator.draw(frame.image, pose)
            if self.eval_viz and instruction:
                display = self.eval_viz.draw_test_guide(display, instruction, countdown)
            cv2.imshow("Real Steel - Camera", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                return None, None

        return human_angles, frame.timestamp


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.calibrate:
        from src.calibration import CalibrationMode
        cal = CalibrationMode(cfg)
        report = cal.run()
        sys.exit(0 if report.overall_pass else 1)

    # Camera setup
    cam_cfg = cfg.get("camera", {})
    camera = Camera(
        device_id=cam_cfg.get("device_id", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        fps=cam_cfg.get("fps", 30),
    )

    # Pose estimator setup
    pose_cfg = cfg.get("pose", {})
    model_path = pose_cfg.get("model_path", "data/pose_landmarker_lite.task")
    pose_estimator = PoseEstimator(
        model_path=model_path,
        min_visibility=pose_cfg.get("min_visibility", 0.5),
    )

    # Angle calculator
    angle_cfg = cfg.get("angles", {})
    angle_calculator = AngleCalculator(
        smoothing_factor=angle_cfg.get("smoothing_factor", 0.3),
    )

    # Motion mapper
    map_cfg = cfg.get("mapping", {})
    dead_zone_deg = map_cfg.get("dead_zone_deg", 3.0)
    mapping_config = MappingConfig(
        mirror_mode=map_cfg.get("mirror_mode", True),
        dead_zone=np.deg2rad(dead_zone_deg),
    )
    motion_mapper = MotionMapper(config=mapping_config)

    # Pipeline profiler
    profiler = PipelineProfiler(
        stages=["camera", "pose", "angles", "mapping", "robot"],
        report_interval=5.0,
    )

    # Robot setup
    robot = None
    use_sim = False
    if args.sim:
        from src.simulated_robot import SimulatedRobot
        sim_cfg = cfg.get("simulation", {})
        urdf_path = sim_cfg.get("urdf_path", "urdf/real_steel.urdf")
        robot = SimulatedRobot(urdf_path=urdf_path, gui=True)
        if not robot.connect():
            print("Failed to connect to simulation")
            sys.exit(1)
        use_sim = True
        print("Simulation connected")
    elif args.port or cfg.get("serial", {}).get("port"):
        from src.real_robot import RealRobot
        port = args.port or cfg["serial"]["port"]
        baud = cfg.get("serial", {}).get("baud", 115200)
        robot = RealRobot(port=port, baud=baud)
        if not robot.connect():
            print(f"Failed to connect to ESP32 on {port}")
            sys.exit(1)
    else:
        print("Specify --sim or --port")
        sys.exit(1)

    # Open camera
    if not camera.open():
        print("Failed to open camera")
        if robot:
            robot.disconnect()
        sys.exit(1)
    print("Camera opened")

    # Evaluation setup
    recorder = None
    evaluator = None
    eval_viz = None

    if args.record or args.run_test:
        from src.evaluation.recorder import Recorder
        recorder = Recorder()
        session_id = recorder.start_session(config_snapshot=cfg)
        print(f"Recording session: {session_id}")

    if args.eval or args.run_test:
        from src.evaluation.evaluator import Evaluator
        from src.evaluation.visualizer import EvalVisualizer
        eval_tolerance = cfg.get("evaluation", {}).get("tolerance_deg", 15.0)
        evaluator = Evaluator(tolerance_deg=eval_tolerance)
        eval_viz = EvalVisualizer()

    # Run test mode
    if args.run_test:
        test_runner = TestRunner(
            evaluator=evaluator,
            recorder=recorder,
            eval_viz=eval_viz,
            camera=camera,
            pose_estimator=pose_estimator,
            angle_calculator=angle_calculator,
            motion_mapper=motion_mapper,
            robot=robot,
            use_sim=use_sim,
            no_viz=args.no_viz,
        )

        if args.run_test in ("static", "all"):
            test_runner.run_static_poses("config/test_poses.yaml")

        if args.run_test in ("dynamic", "all"):
            test_runner.run_motion_sequences("config/test_sequences.yaml")

        test_runner.save_report("data/benchmarks/")

        # Cleanup
        if recorder:
            path = recorder.end_session()
            print(f"Session saved to {path}")
        if robot is not None:
            robot.home()
            robot.disconnect()
        camera.release()
        pose_estimator.close()
        cv2.destroyAllWindows()
        print("Done")
        return

    # PyBullet camera state for keyboard control
    if use_sim:
        import pybullet as pb
        cam_info = pb.getDebugVisualizerCamera()
        cam_dist = cam_info[10]
        cam_yaw = cam_info[8]
        cam_pitch = cam_info[9]
        cam_target = list(cam_info[11])

    # FPS tracking
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0.0

    debug_timer = time.time()
    frame_num = 0

    print("Running pipeline. Press 'q' or ESC to quit.")
    if use_sim:
        print("PyBullet camera: arrow keys to rotate, +/- to zoom")
    print("Joint order: [L_roll, L_tilt, L_pan, L_elbow, R_roll, R_tilt, R_pan, R_elbow]")

    try:
        while True:
            frame_start = time.perf_counter()

            profiler.start("camera")
            frame, is_new_frame = camera.read()
            profiler.stop("camera")

            if frame is None or not is_new_frame:
                # No new frame — keep sim stepping but skip processing
                if use_sim and robot is not None:
                    robot.step()
                continue

            profiler.tick()

            # Pose estimation
            profiler.start("pose")
            pose = pose_estimator.process(frame.image, frame.timestamp)
            profiler.stop("pose")

            # Angle calculation
            profiler.start("angles")
            joint_angles = angle_calculator.calculate(pose)
            profiler.stop("angles")

            # Motion mapping and robot command
            servo_angles = None
            profiler.start("mapping")
            if joint_angles is not None and robot is not None:
                servo_angles = motion_mapper.map(joint_angles)
            profiler.stop("mapping")

            profiler.start("robot")
            if servo_angles is not None and robot is not None:
                robot.set_joint_positions(servo_angles.angles)
            profiler.stop("robot")

            # Step simulation + keyboard camera control
            if use_sim and robot is not None:
                robot.step()

                keys = pb.getKeyboardEvents()
                if pb.B3G_LEFT_ARROW in keys and keys[pb.B3G_LEFT_ARROW] & pb.KEY_IS_DOWN:
                    cam_yaw -= 2
                if pb.B3G_RIGHT_ARROW in keys and keys[pb.B3G_RIGHT_ARROW] & pb.KEY_IS_DOWN:
                    cam_yaw += 2
                if pb.B3G_UP_ARROW in keys and keys[pb.B3G_UP_ARROW] & pb.KEY_IS_DOWN:
                    cam_pitch -= 2
                if pb.B3G_DOWN_ARROW in keys and keys[pb.B3G_DOWN_ARROW] & pb.KEY_IS_DOWN:
                    cam_pitch += 2
                if ord("=") in keys and keys[ord("=")] & pb.KEY_IS_DOWN:
                    cam_dist = max(0.2, cam_dist - 0.05)
                if ord("-") in keys and keys[ord("-")] & pb.KEY_IS_DOWN:
                    cam_dist += 0.05

                pb.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, cam_target)

            # Compute latency for this frame
            latency_ms = (time.perf_counter() - frame_start) * 1000.0

            # Recording
            if recorder and joint_angles is not None and servo_angles is not None and robot is not None:
                robot_actual = robot.get_joint_state().positions
                pose_confidence = (
                    np.mean([p.visibility for p in pose.keypoints.values()])
                    if pose.keypoints else 0.0
                )
                recorder.record_frame(
                    timestamp=frame.timestamp,
                    frame_number=frame.frame_number,
                    keypoints=pose.keypoints if pose.is_valid else None,
                    pose_confidence=pose_confidence,
                    human_angles=joint_angles.to_array(),
                    robot_cmd=servo_angles.angles,
                    robot_actual=robot_actual,
                    latency_ms=latency_ms,
                )

            # Debug print every second
            frame_num += 1
            now = time.time()
            if now - debug_timer >= 1.0:
                debug_timer = now
                if joint_angles is not None and servo_angles is not None and robot is not None:
                    human = np.rad2deg(joint_angles.to_array())
                    mapped = np.rad2deg(servo_angles.angles)
                    actual = np.rad2deg(robot.get_joint_state().positions)
                    fmt = lambda a: ", ".join(f"{v:+6.1f}" for v in a)
                    print(f"--- Frame {frame_num} ---")
                    print(f"  Human:  [{fmt(human)}]")
                    print(f"  Robot:  [{fmt(mapped)}]")
                    print(f"  Actual: [{fmt(actual)}]")
                elif not pose.is_valid:
                    print(f"--- Frame {frame_num} --- (no pose detected)")

            # Visualization
            if not args.no_viz:
                display = pose_estimator.draw(frame.image, pose)

                # Eval overlay
                if evaluator and servo_angles is not None and robot is not None:
                    robot_actual = robot.get_joint_state().positions
                    gap_b = evaluator.compute_gap_b(servo_angles.angles, robot_actual)
                    evaluator.update_running_stats(gap_b)
                    display = eval_viz.draw_gap_overlay(display, evaluator.get_running_stats())

                # FPS counter
                fps_counter += 1
                elapsed = time.time() - fps_timer
                if elapsed >= 5.0:
                    current_fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_timer = time.time()

                # Profiler report (prints every report_interval)
                report = profiler.report()
                if report:
                    print(report)

                # Draw FPS on frame
                cv2.putText(
                    display,
                    f"FPS: {current_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Draw current angles
                if joint_angles is not None:
                    angles_deg = np.rad2deg(joint_angles.to_array())
                    labels = ["LR", "LT", "LP", "LE", "RR", "RT", "RP", "RE"]
                    y_start = 60
                    for i, (label, deg) in enumerate(zip(labels, angles_deg)):
                        cv2.putText(
                            display,
                            f"{label}: {deg:+6.1f}°",
                            (10, y_start + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                cv2.imshow("Real Steel - Camera", display)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted")

    # Cleanup
    print("Shutting down...")
    if recorder:
        path = recorder.end_session()
        print(f"Session saved to {path}")
    if robot is not None:
        robot.home()
        robot.disconnect()
    camera.release()
    pose_estimator.close()
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
