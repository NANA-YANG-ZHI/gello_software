"""Move a robot arm smoothly from wherever it is to a start (home) joint position.

Works with anything that follows the :class:`gello.robots.robot.Robot` protocol:
a ``ZMQClientRobot`` talking to ``experiments/launch_nodes.py`` (``sim_ur`` or
``ur``), a ``URRobot`` directly, or a ``MujocoRobotServer``.

The motion is a minimum-jerk joint-space trajectory: every joint starts and
stops at the same time with zero velocity and acceleration, and its duration is
chosen so that no joint exceeds ``max_vel`` (rad/s) or ``max_acc`` (rad/s^2).

Example::

    from gello.zmq_core.robot_node import ZMQClientRobot
    from gello.utils.move_to_start import UR_HOME_JOINTS, move_to_joints

    robot = ZMQClientRobot(port=6001, host="127.0.0.1")
    result = move_to_joints(robot, UR_HOME_JOINTS, max_vel=0.5)
    print(result)
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from gello.robots.robot import Robot

# "home" keyframe of the UR5e in mujoco_menagerie
# (third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml).
UR_HOME_JOINTS = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])

# Reset pose used by experiments/run_env.py for a single UR arm.
UR_RUN_ENV_RESET_JOINTS = np.deg2rad([0, -90, 90, -90, -90, 0])

# Minimum-jerk profile s(t) = 10t^3 - 15t^4 + 6t^5 on t in [0, 1]:
# peak velocity = 1.875 * D / T, peak acceleration = 5.7735 * D / T^2.
_MJ_PEAK_VEL = 1.875
_MJ_PEAK_ACC = 10.0 / np.sqrt(3.0)


@dataclass
class MoveResult:
    success: bool
    message: str
    final_error: float  # max |joint error| over arm joints, rad
    duration: float  # wall-clock seconds spent moving and settling


def min_jerk(s: np.ndarray) -> np.ndarray:
    """Minimum-jerk interpolation fraction for normalized time ``s`` in [0, 1]."""
    s = np.clip(s, 0.0, 1.0)
    return s**3 * (10.0 - 15.0 * s + 6.0 * s**2)


def trajectory_duration(
    start: np.ndarray,
    target: np.ndarray,
    max_vel: float,
    max_acc: float,
    min_duration: float = 0.5,
) -> float:
    """Shortest duration that keeps every joint within the velocity/acceleration limits."""
    distance = float(np.max(np.abs(np.asarray(target) - np.asarray(start)), initial=0.0))
    t_vel = _MJ_PEAK_VEL * distance / max_vel
    t_acc = np.sqrt(_MJ_PEAK_ACC * distance / max_acc)
    return max(t_vel, t_acc, min_duration)


def plan_trajectory(
    start: np.ndarray,
    target: np.ndarray,
    control_hz: float,
    max_vel: float,
    max_acc: float,
    min_duration: float = 0.5,
) -> np.ndarray:
    """Waypoints (N, dof) from ``start`` (excluded) to ``target`` (included)."""
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    duration = trajectory_duration(start, target, max_vel, max_acc, min_duration)
    n = max(int(np.ceil(duration * control_hz)), 1)
    s = np.arange(1, n + 1) / n
    return start + min_jerk(s)[:, None] * (target - start)


def move_to_joints(
    robot: Robot,
    target_joints: Sequence[float],
    *,
    max_vel: float = 0.5,
    max_acc: float = 1.0,
    control_hz: float = 100.0,
    tolerance: float = 0.01,
    settle_timeout: float = 3.0,
    max_tracking_error: Optional[float] = 0.5,
    max_distance: Optional[float] = None,
    gripper: Optional[float] = None,
    verbose: bool = True,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> MoveResult:
    """Move the arm joints to ``target_joints`` along a smooth trajectory.

    Args:
        robot: Robot to command.
        target_joints: Arm joint targets in radians (6 for a UR). If the robot has
            more DoFs (e.g. a gripper as the 7th), the extra DoFs are held
            (or set to ``gripper``).
        max_vel: Joint velocity limit, rad/s.
        max_acc: Joint acceleration limit, rad/s^2.
        control_hz: Rate at which commands are streamed.
        tolerance: Success when every arm joint is within this many radians.
        settle_timeout: Seconds to keep holding the target after the trajectory
            ends while waiting to get within ``tolerance``.
        max_tracking_error: Abort (and hold the measured position) if any arm joint
            lags its command by more than this many radians, e.g. on a collision.
            ``None`` disables the check.
        max_distance: Refuse to move if any joint is further than this from the
            target. ``None`` disables the check.
        gripper: Gripper command (0 = open, 1 = closed) for the extra DoF. ``None``
            keeps the gripper where it is.
        verbose: Print progress.
        clock, sleep: Injectable for testing.
    """
    target = np.asarray(target_joints, dtype=float).reshape(-1)
    n_arm = target.size
    if not np.all(np.isfinite(target)):
        return MoveResult(False, f"target is not finite: {target}", np.inf, 0.0)

    num_dofs = robot.num_dofs()
    if n_arm > num_dofs:
        return MoveResult(
            False,
            f"target has {n_arm} joints but robot only has {num_dofs} DoFs",
            np.inf,
            0.0,
        )

    current = np.asarray(robot.get_joint_state(), dtype=float).reshape(-1)
    extra = current[n_arm:num_dofs].copy()
    if gripper is not None:
        extra[:] = gripper

    def full_command(arm: np.ndarray) -> np.ndarray:
        return np.concatenate([arm, extra])

    start = current[:n_arm]
    distance = float(np.max(np.abs(target - start), initial=0.0))
    if max_distance is not None and distance > max_distance:
        return MoveResult(
            False,
            f"refusing to move: max joint distance {distance:.3f} rad > "
            f"max_distance {max_distance:.3f} rad",
            distance,
            0.0,
        )

    trajectory = plan_trajectory(start, target, control_hz, max_vel, max_acc)
    period = 1.0 / control_hz
    if verbose:
        print(
            f"Moving to start: max joint distance {distance:.3f} rad, "
            f"{len(trajectory) * period:.2f} s at {control_hz:.0f} Hz"
        )

    t0 = clock()
    next_tick = t0

    def wait_for_tick() -> None:
        nonlocal next_tick
        next_tick += period
        remaining = next_tick - clock()
        if remaining > 0:
            sleep(remaining)
        else:
            next_tick = clock()  # running late; don't try to catch up

    def hold(measured: np.ndarray) -> None:
        robot.command_joint_state(full_command(measured[:n_arm]))

    try:
        # 1) Stream the trajectory.
        for waypoint in trajectory:
            robot.command_joint_state(full_command(waypoint))
            wait_for_tick()
            measured = np.asarray(robot.get_joint_state(), dtype=float).reshape(-1)
            lag = float(np.max(np.abs(measured[:n_arm] - waypoint)))
            if max_tracking_error is not None and lag > max_tracking_error:
                hold(measured)
                return MoveResult(
                    False,
                    f"aborted: tracking error {lag:.3f} rad > {max_tracking_error:.3f} rad "
                    "(blocked or not following commands?); holding current position",
                    float(np.max(np.abs(measured[:n_arm] - target))),
                    clock() - t0,
                )

        # 2) Hold the target until the arm settles.
        settle_deadline = clock() + settle_timeout
        while True:
            measured = np.asarray(robot.get_joint_state(), dtype=float).reshape(-1)
            error = float(np.max(np.abs(measured[:n_arm] - target)))
            if error <= tolerance:
                if verbose:
                    print(f"Reached start position (max error {error:.4f} rad)")
                return MoveResult(True, "reached target", error, clock() - t0)
            if clock() >= settle_deadline:
                return MoveResult(
                    False,
                    f"timed out: max joint error {error:.4f} rad > tolerance {tolerance} rad",
                    error,
                    clock() - t0,
                )
            robot.command_joint_state(full_command(target))
            wait_for_tick()
    except KeyboardInterrupt:
        measured = np.asarray(robot.get_joint_state(), dtype=float).reshape(-1)
        hold(measured)
        if verbose:
            print("Interrupted: holding current position")
        raise
