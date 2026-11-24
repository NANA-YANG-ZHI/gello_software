"""Move a UR arm back to a start joint position after teleoperation.

Run this while the UR driver (start_drivers.launch.py) is still up and after the
GELLO bridge has been stopped:

    ros2 run ur_move_to_start move_to_start
    ros2 run ur_move_to_start move_to_start --start-file <path/to/initial_positions.yaml>
    ros2 run ur_move_to_start move_to_start --start 0 -90 90 -90 -90 0 --deg

What it does:
  1. Reads the current arm position from /joint_states.
  2. Deactivates whatever controller is commanding the arm (e.g. the
     forward_position_controller the bridge streams to) and activates the driver's
     scaled_joint_trajectory_controller.
  3. Sends one FollowJointTrajectory goal to the start pose. The controller
     interpolates smoothly from rest to rest, and the pendant speed slider applies.
  4. Optionally (--restore) switches the original controllers back on.

Ctrl-C while moving cancels the goal and the arm stops where it is.
"""

import argparse
import math
import os
import sys
from typing import List, Optional

import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectoryPoint

from ur_move_to_start.planning import (
    conflicting_controllers,
    joint_names,
    load_start_positions,
    max_joint_distance,
    motion_duration,
    positions_by_name,
)

DEFAULT_TRAJECTORY_CONTROLLER = "scaled_joint_trajectory_controller"


class MoveToStartError(RuntimeError):
    pass


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move a UR arm back to its start position.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--start-file",
        help="YAML start pose (ur_description initial_positions.yaml format). "
        "Default: this package's config/start_position.yaml",
    )
    source.add_argument(
        "--start", type=float, nargs=6, metavar="Q", help="Start pose, 6 joint values"
    )
    parser.add_argument("--deg", action="store_true", help="--start values are in degrees")
    parser.add_argument("--tf-prefix", default="", help="Joint name prefix used by the driver")
    parser.add_argument(
        "--controller-manager", default="/controller_manager", help="Controller manager node"
    )
    parser.add_argument(
        "--trajectory-controller",
        default=DEFAULT_TRAJECTORY_CONTROLLER,
        help="Trajectory controller to use (scaled_joint_trajectory_controller respects "
        "the pendant speed slider; joint_trajectory_controller does not)",
    )
    parser.add_argument("--max-vel", type=float, default=0.3, help="Joint speed limit, rad/s")
    parser.add_argument("--max-acc", type=float, default=0.5, help="Joint accel limit, rad/s^2")
    parser.add_argument("--min-duration", type=float, default=2.0, help="Shortest motion, s")
    parser.add_argument(
        "--max-joint-distance",
        type=float,
        default=math.pi,
        help="Refuse to move if any joint is further than this from the target (rad). "
        "Guards against a wrong pose or a 2*pi wrap turning a joint all the way round.",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Re-activate the controllers that were switched off once the move is done",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan, move nothing")
    args = parser.parse_args(argv)
    if args.deg and args.start is None:
        parser.error("--deg only applies to --start")
    return args


def resolve_target(args: argparse.Namespace) -> List[float]:
    if args.start is not None:
        values = list(args.start)
        return [math.radians(v) for v in values] if args.deg else values
    path = args.start_file or os.path.join(
        get_package_share_directory("ur_move_to_start"), "config", "start_position.yaml"
    )
    return load_start_positions(path)


class MoveToStart(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("ur_move_to_start")
        self.args = args
        self.joints = joint_names(args.tf_prefix)
        self._joint_state: Optional[JointState] = None
        self._program_running: Optional[bool] = None

        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            Bool,
            "/io_and_status_controller/robot_program_running",
            self._on_program_running,
            latched,
        )

        cm = args.controller_manager.rstrip("/")
        self.list_client = self.create_client(ListControllers, f"{cm}/list_controllers")
        self.switch_client = self.create_client(SwitchController, f"{cm}/switch_controller")
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            f"/{args.trajectory_controller}/follow_joint_trajectory",
        )

    # --- callbacks ---------------------------------------------------------------

    def _on_joint_state(self, msg: JointState) -> None:
        self._joint_state = msg

    def _on_program_running(self, msg: Bool) -> None:
        self._program_running = msg.data

    # --- helpers -------------------------------------------------------------------

    def _spin_until(self, predicate, timeout: float) -> bool:
        deadline = self.get_clock().now().nanoseconds + int(timeout * 1e9)
        while rclpy.ok() and not predicate():
            if self.get_clock().now().nanoseconds > deadline:
                return False
            rclpy.spin_once(self, timeout_sec=0.05)
        return predicate()

    def _call(self, client, request, timeout: float = 5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise MoveToStartError(f"service {client.srv_name} not available. Is the driver up?")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            raise MoveToStartError(f"service {client.srv_name} did not answer")
        return future.result()

    def current_positions(self) -> List[float]:
        def have_arm_joints() -> bool:
            msg = self._joint_state
            return msg is not None and (
                positions_by_name(msg.name, msg.position, self.joints) is not None
            )

        if not self._spin_until(have_arm_joints, timeout=5.0):
            raise MoveToStartError(
                f"no /joint_states with joints {self.joints} within 5 s. "
                "Is the driver running, and is --tf-prefix right?"
            )
        msg = self._joint_state
        return positions_by_name(msg.name, msg.position, self.joints)

    def check_program_running(self) -> None:
        self._spin_until(lambda: self._program_running is not None, timeout=2.0)
        if self._program_running is False:
            raise MoveToStartError(
                "the External Control program is not running on the robot. "
                "Press play on the teach pendant (or restart it via the dashboard client)."
            )
        if self._program_running is None:
            self.get_logger().warn(
                "could not read robot_program_running; continuing. "
                "If nothing moves, check that External Control is playing on the pendant."
            )

    def list_controllers(self):
        response = self._call(self.list_client, ListControllers.Request())
        return [
            {"name": c.name, "state": c.state, "claimed_interfaces": list(c.claimed_interfaces)}
            for c in response.controller
        ]

    def switch(self, activate: List[str], deactivate: List[str]) -> None:
        if not activate and not deactivate:
            return
        request = SwitchController.Request()
        request.activate_controllers = activate
        request.deactivate_controllers = deactivate
        request.strictness = SwitchController.Request.STRICT
        request.timeout = DurationMsg(sec=5)
        response = self._call(self.switch_client, request, timeout=10.0)
        if not response.ok:
            raise MoveToStartError(
                f"controller switch failed (activate={activate}, deactivate={deactivate})"
            )
        self.get_logger().info(f"controllers: activated {activate}, deactivated {deactivate}")

    def prepare_controllers(self) -> List[str]:
        """Make the trajectory controller the only one commanding the arm.

        Returns the controllers that were deactivated, so they can be restored.
        """
        controllers = self.list_controllers()
        names = {c["name"]: c for c in controllers}
        trajectory = self.args.trajectory_controller
        if trajectory not in names:
            raise MoveToStartError(
                f"controller '{trajectory}' is not loaded. Loaded: {sorted(names)}"
            )
        to_deactivate = conflicting_controllers(controllers, self.joints, keep=trajectory)
        to_activate = [] if names[trajectory]["state"] == "active" else [trajectory]
        self.switch(to_activate, to_deactivate)
        return to_deactivate

    def execute(self, target: List[float], duration: float) -> None:
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            raise MoveToStartError(
                f"action server {self.args.trajectory_controller}/follow_joint_trajectory "
                "not available"
            )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joints
        point = JointTrajectoryPoint()
        point.positions = target
        point.velocities = [0.0] * len(target)
        point.accelerations = [0.0] * len(target)
        point.time_from_start = DurationMsg(
            sec=int(duration), nanosec=int((duration - int(duration)) * 1e9)
        )
        goal.trajectory.points = [point]

        send_future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise MoveToStartError("trajectory goal was rejected")

        result_future = handle.get_result_async()
        try:
            # No timeout: the scaled controller slows down with the speed slider.
            rclpy.spin_until_future_complete(self, result_future)
        except KeyboardInterrupt:
            self.get_logger().warn("interrupted, cancelling the motion")
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
            raise

        wrapped = result_future.result()
        if wrapped is None:
            raise MoveToStartError("no result from the trajectory controller")
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED or wrapped.result.error_code != 0:
            raise MoveToStartError(
                f"motion failed: status={wrapped.status}, "
                f"error_code={wrapped.result.error_code}, {wrapped.result.error_string}"
            )


def fmt(values: List[float]) -> str:
    return "[" + ", ".join(f"{math.degrees(v):7.1f}" for v in values) + "] deg"


def run(node: MoveToStart, args: argparse.Namespace) -> None:
    target = resolve_target(args)
    current = node.current_positions()
    distance = max_joint_distance(current, target)
    duration = motion_duration(current, target, args.max_vel, args.max_acc, args.min_duration)

    print(f"current: {fmt(current)}")
    print(f"target:  {fmt(target)}")
    print(f"max joint distance {math.degrees(distance):.1f} deg, planned duration {duration:.1f} s")

    if distance > args.max_joint_distance:
        raise MoveToStartError(
            f"a joint is {math.degrees(distance):.1f} deg from the target, more than "
            f"--max-joint-distance ({math.degrees(args.max_joint_distance):.1f} deg). "
            "Check the start pose, or raise the limit if this is intended."
        )
    if args.dry_run:
        print("dry run: nothing moved")
        return

    node.check_program_running()

    if not args.yes:
        answer = input("Move the robot to the start position? Make sure the path is clear. [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("cancelled")
            return

    deactivated = node.prepare_controllers()
    try:
        node.execute(target, duration)
        print("reached start position")
    finally:
        if args.restore and deactivated:
            node.switch(activate=deactivated, deactivate=[args.trajectory_controller])


def main(argv=None) -> None:
    argv = sys.argv if argv is None else argv
    args = parse_args(remove_ros_args(argv)[1:])
    rclpy.init(args=argv)
    node = MoveToStart(args)
    exit_code = 0
    try:
        run(node, args)
    except MoveToStartError as e:
        node.get_logger().error(str(e))
        exit_code = 1
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
