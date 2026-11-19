"""Move a UR arm (real or simulated) back to its start position.

Start the robot node first, in another terminal:

    python experiments/launch_nodes.py --robot sim_ur     # MuJoCo
    python experiments/launch_nodes.py --robot ur --robot-ip <ip>   # real arm

Then:

    python experiments/move_to_start.py                          # UR "home" keyframe
    python experiments/move_to_start.py --pose run-env           # run_env.py reset pose
    python experiments/move_to_start.py --start-joints-deg 0 -90 90 -90 -90 0
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import tyro

from gello.utils.move_to_start import (
    UR_HOME_JOINTS,
    UR_RUN_ENV_RESET_JOINTS,
    move_to_joints,
)
from gello.zmq_core.robot_node import ZMQClientRobot


@dataclass
class Args:
    hostname: str = "127.0.0.1"
    robot_port: int = 6001

    pose: Literal["home", "run-env"] = "home"
    """Named start pose: 'home' = menagerie UR5e keyframe, 'run-env' = run_env.py reset."""
    start_joints: Optional[Tuple[float, ...]] = None
    """Custom start pose in radians (overrides --pose)."""
    start_joints_deg: Optional[Tuple[float, ...]] = None
    """Custom start pose in degrees (overrides --pose)."""

    max_vel: float = 0.5
    """Joint velocity limit, rad/s."""
    max_acc: float = 1.0
    """Joint acceleration limit, rad/s^2."""
    hz: float = 100.0
    tolerance: float = 0.01
    """Success when every joint is within this many radians."""
    gripper: Optional[float] = None
    """Gripper command 0 (open) .. 1 (closed); default keeps it as is."""
    yes: bool = False
    """Skip the confirmation prompt."""


def resolve_target(args: Args) -> np.ndarray:
    if args.start_joints is not None and args.start_joints_deg is not None:
        raise ValueError("Pass only one of --start-joints and --start-joints-deg")
    if args.start_joints is not None:
        return np.array(args.start_joints, dtype=float)
    if args.start_joints_deg is not None:
        return np.deg2rad(np.array(args.start_joints_deg, dtype=float))
    return UR_HOME_JOINTS if args.pose == "home" else UR_RUN_ENV_RESET_JOINTS


def main(args: Args) -> None:
    target = resolve_target(args)
    robot = ZMQClientRobot(port=args.robot_port, host=args.hostname)

    current = np.asarray(robot.get_joint_state(), dtype=float)
    n = len(target)
    np.set_printoptions(precision=1, suppress=True)
    print(f"current (deg): {np.rad2deg(current[:n])}")
    print(f"target  (deg): {np.rad2deg(target)}")

    if not args.yes:
        answer = input("Move the arm to the target? Make sure the path is clear. [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Cancelled")
            return

    result = move_to_joints(
        robot,
        target,
        max_vel=args.max_vel,
        max_acc=args.max_acc,
        control_hz=args.hz,
        tolerance=args.tolerance,
        gripper=args.gripper,
    )
    print(result.message)
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
