"""ROS-free helpers for move_to_start: loading the start pose and sizing the motion."""

import math
from typing import Dict, List, Mapping, Optional, Sequence

import yaml

UR_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# The trajectory controller fits a quintic between the current state (at rest) and a
# goal point with zero velocity and acceleration. For a quintic over distance D and
# duration T: peak velocity = 1.875 * D / T, peak acceleration = 5.7735 * D / T^2.
_QUINTIC_PEAK_VEL = 1.875
_QUINTIC_PEAK_ACC = 10.0 / math.sqrt(3.0)


def joint_names(tf_prefix: str = "") -> List[str]:
    return [tf_prefix + name for name in UR_JOINT_NAMES]


def load_start_positions(path: str) -> List[float]:
    """Read a start pose from YAML, in UR joint order.

    Accepts the flat format of ur_description's initial_positions.yaml
    (``shoulder_pan_joint: 0.0`` ...), optionally nested under one top-level key,
    or a plain list of six numbers.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if isinstance(data, Mapping) and len(data) == 1:
        (only_value,) = data.values()
        if isinstance(only_value, (Mapping, list)):
            data = only_value

    if isinstance(data, list):
        positions = [float(v) for v in data]
    elif isinstance(data, Mapping):
        missing = [name for name in UR_JOINT_NAMES if name not in data]
        if missing:
            raise ValueError(f"{path}: missing joints {missing}")
        positions = [float(data[name]) for name in UR_JOINT_NAMES]
    else:
        raise ValueError(f"{path}: expected a mapping of joint names or a list of 6 numbers")

    if len(positions) != len(UR_JOINT_NAMES):
        raise ValueError(f"{path}: expected {len(UR_JOINT_NAMES)} joints, got {len(positions)}")
    if not all(math.isfinite(p) for p in positions):
        raise ValueError(f"{path}: non-finite joint position in {positions}")
    return positions


def positions_by_name(
    names: Sequence[str], positions: Sequence[float], wanted: Sequence[str]
) -> Optional[List[float]]:
    """Pick ``wanted`` joints out of a JointState-style (names, positions) pair."""
    lookup: Dict[str, float] = dict(zip(names, positions))
    if not all(name in lookup for name in wanted):
        return None
    return [float(lookup[name]) for name in wanted]


def max_joint_distance(current: Sequence[float], target: Sequence[float]) -> float:
    return max((abs(t - c) for c, t in zip(current, target)), default=0.0)


def motion_duration(
    current: Sequence[float],
    target: Sequence[float],
    max_vel: float,
    max_acc: float,
    min_duration: float,
) -> float:
    """Shortest duration (s) keeping every joint under ``max_vel`` and ``max_acc``."""
    if max_vel <= 0 or max_acc <= 0:
        raise ValueError("max_vel and max_acc must be positive")
    distance = max_joint_distance(current, target)
    t_vel = _QUINTIC_PEAK_VEL * distance / max_vel
    t_acc = math.sqrt(_QUINTIC_PEAK_ACC * distance / max_acc)
    return max(t_vel, t_acc, min_duration)


def conflicting_controllers(
    controllers: Sequence[Mapping], joints: Sequence[str], keep: str
) -> List[str]:
    """Names of active controllers (other than ``keep``) that claim a command interface
    of one of ``joints``. Each controller is a mapping with ``name``, ``state`` and
    ``claimed_interfaces``, mirroring controller_manager_msgs/ControllerState.
    """
    prefixes = tuple(joint + "/" for joint in joints)
    result = []
    for controller in controllers:
        if controller["name"] == keep or controller["state"] != "active":
            continue
        if any(iface.startswith(prefixes) for iface in controller["claimed_interfaces"]):
            result.append(controller["name"])
    return result
