import math

import pytest

from ur_move_to_start.planning import (
    UR_JOINT_NAMES,
    conflicting_controllers,
    joint_names,
    load_start_positions,
    max_joint_distance,
    motion_duration,
    positions_by_name,
)

HOME = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]


def write(tmp_path, text):
    path = tmp_path / "pose.yaml"
    path.write_text(text)
    return str(path)


def test_load_flat_initial_positions_format(tmp_path):
    text = "\n".join(f"{n}: {v}" for n, v in reversed(list(zip(UR_JOINT_NAMES, HOME))))
    assert load_start_positions(write(tmp_path, text)) == HOME


def test_load_nested_and_list_formats(tmp_path):
    nested = "start:\n" + "\n".join(f"  {n}: {v}" for n, v in zip(UR_JOINT_NAMES, HOME))
    assert load_start_positions(write(tmp_path, nested)) == HOME
    assert load_start_positions(write(tmp_path, str(HOME))) == HOME


@pytest.mark.parametrize(
    "text",
    ["shoulder_pan_joint: 0.0", "[0, 1, 2]", "[0, 1, 2, 3, 4, .nan]", "just a string"],
)
def test_load_rejects_bad_files(tmp_path, text):
    with pytest.raises(ValueError):
        load_start_positions(write(tmp_path, text))


def test_positions_by_name_reorders_and_handles_missing():
    names = ["elbow_joint", "x", "shoulder_pan_joint"]
    assert positions_by_name(names, [3.0, 9.0, 1.0], ["shoulder_pan_joint", "elbow_joint"]) == [
        1.0,
        3.0,
    ]
    assert positions_by_name(names, [3.0, 9.0, 1.0], ["wrist_1_joint"]) is None


def test_motion_duration_respects_limits():
    start = [0.0] * 6
    target = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    t = motion_duration(start, target, max_vel=0.5, max_acc=100.0, min_duration=0.1)
    assert t == pytest.approx(1.875 * 1.0 / 0.5)
    t = motion_duration(start, target, max_vel=100.0, max_acc=0.5, min_duration=0.1)
    assert 5.7735 * 1.0 / t**2 == pytest.approx(0.5, rel=1e-3)
    assert motion_duration(start, start, 0.3, 0.5, min_duration=2.0) == 2.0
    with pytest.raises(ValueError):
        motion_duration(start, target, 0.0, 1.0, 1.0)


def test_max_joint_distance():
    assert max_joint_distance([0, 0, 2 * math.pi], [0, 0.1, 0]) == pytest.approx(2 * math.pi)


def test_conflicting_controllers_by_claimed_interfaces():
    joints = joint_names("ur_")
    controllers = [
        {"name": "forward_position_controller", "state": "active",
         "claimed_interfaces": [j + "/position" for j in joints]},
        {"name": "scaled_joint_trajectory_controller", "state": "active",
         "claimed_interfaces": [j + "/position" for j in joints]},
        {"name": "forward_velocity_controller", "state": "inactive", "claimed_interfaces": []},
        {"name": "io_and_status_controller", "state": "active",
         "claimed_interfaces": ["ur_gpio/io_async_success"]},
        {"name": "other_arm", "state": "active",
         "claimed_interfaces": ["left_shoulder_pan_joint/position"]},
    ]
    assert conflicting_controllers(
        controllers, joints, keep="scaled_joint_trajectory_controller"
    ) == ["forward_position_controller"]
