from pathlib import Path

import numpy as np
import pytest

from gello.utils.move_to_start import (
    UR_HOME_JOINTS,
    min_jerk,
    move_to_joints,
    move_to_start,
    plan_trajectory,
    trajectory_duration,
)

HZ = 100.0


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class LagRobot:
    """First-order-lag position-controlled robot: arm joints + optional gripper."""

    def __init__(self, q0, clock, alpha=0.5, stuck_joint=None):
        self.q = np.array(q0, dtype=float)
        self.clock = clock
        self.alpha = alpha
        self.stuck_joint = stuck_joint
        self.commands = []

    def num_dofs(self):
        return len(self.q)

    def get_joint_state(self):
        return self.q.copy()

    def command_joint_state(self, cmd):
        cmd = np.asarray(cmd, dtype=float)
        assert cmd.shape == self.q.shape
        self.commands.append(cmd.copy())
        new_q = self.q + self.alpha * (cmd - self.q)
        if self.stuck_joint is not None:
            new_q[self.stuck_joint] = self.q[self.stuck_joint]
        self.q = new_q


def run(robot, clock, target, **kw):
    return move_to_joints(
        robot, target, control_hz=HZ, verbose=False, clock=clock, sleep=clock.sleep, **kw
    )


def test_min_jerk_endpoints():
    assert min_jerk(np.array(0.0)) == 0.0
    assert min_jerk(np.array(1.0)) == pytest.approx(1.0)
    assert min_jerk(np.array(0.5)) == pytest.approx(0.5)


def test_trajectory_respects_limits():
    start = np.zeros(6)
    target = np.array([2.0, -1.0, 0.5, 0, 0, 3.0])
    traj = plan_trajectory(start, target, HZ, max_vel=0.5, max_acc=1.0)
    assert np.allclose(traj[-1], target)
    full = np.vstack([start, traj])
    vel = np.diff(full, axis=0) * HZ
    acc = np.diff(vel, axis=0) * HZ
    assert np.abs(vel).max() <= 0.5 * 1.01
    assert np.abs(acc).max() <= 1.0 * 1.05
    # all joints arrive together
    assert len(traj) == int(np.ceil(trajectory_duration(start, target, 0.5, 1.0) * HZ))


def test_reaches_target_and_holds_gripper():
    clock = FakeClock()
    q0 = np.concatenate([np.random.default_rng(0).uniform(-3, 3, 6), [0.7]])
    robot = LagRobot(q0, clock)
    result = run(robot, clock, UR_HOME_JOINTS)
    assert result.success, result.message
    assert np.allclose(robot.q[:6], UR_HOME_JOINTS, atol=0.01)
    assert all(c[6] == pytest.approx(0.7) for c in robot.commands)


def test_gripper_override():
    clock = FakeClock()
    robot = LagRobot(np.zeros(7), clock)
    result = run(robot, clock, UR_HOME_JOINTS, gripper=0.0)
    assert result.success
    robot2 = LagRobot(np.r_[np.zeros(6), 1.0], FakeClock())
    run(robot2, robot2.clock, UR_HOME_JOINTS, gripper=0.0)
    assert all(c[6] == 0.0 for c in robot2.commands)


def test_move_to_start_defaults_to_home():
    clock = FakeClock()
    robot = LagRobot(np.r_[np.zeros(6), 0.3], clock)
    result = move_to_start(robot, control_hz=HZ, verbose=False, clock=clock, sleep=clock.sleep)
    assert result.success
    assert np.allclose(robot.q[:6], UR_HOME_JOINTS, atol=0.01)


def test_already_at_target():
    clock = FakeClock()
    robot = LagRobot(UR_HOME_JOINTS, clock)
    result = run(robot, clock, UR_HOME_JOINTS)
    assert result.success


def test_aborts_when_blocked():
    clock = FakeClock()
    robot = LagRobot(np.zeros(6), clock, stuck_joint=1)
    result = run(robot, clock, UR_HOME_JOINTS, max_tracking_error=0.2)
    assert not result.success
    assert "tracking error" in result.message
    # last command holds the measured position, not the target
    assert np.allclose(robot.commands[-1], robot.q, atol=0.2)


def test_refuses_far_target():
    clock = FakeClock()
    robot = LagRobot(np.zeros(6), clock)
    result = run(robot, clock, UR_HOME_JOINTS, max_distance=1.0)
    assert not result.success
    assert robot.commands == []


def test_rejects_bad_target():
    clock = FakeClock()
    robot = LagRobot(np.zeros(6), clock)
    assert not run(robot, clock, [np.nan] * 6).success
    assert not run(robot, clock, np.zeros(8)).success


def test_timeout_when_never_converges():
    clock = FakeClock()
    robot = LagRobot(np.zeros(6), clock, alpha=0.0)
    result = run(robot, clock, UR_HOME_JOINTS, max_tracking_error=None, settle_timeout=0.5)
    assert not result.success
    assert "timed out" in result.message


# --------------------------------------------------------------------------- MuJoCo
MENAGERIE = Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"


class SteppedMujocoUR:
    """UR5e + Robotiq 2F-85 from the GELLO sim, stepped synchronously (no viewer)."""

    def __init__(self, q0):
        mujoco = pytest.importorskip("mujoco")
        from gello.robots.sim_robot import build_scene

        self.mujoco = mujoco
        arena = build_scene(
            str(MENAGERIE / "universal_robots_ur5e" / "ur5e.xml"),
            str(MENAGERIE / "robotiq_2f85" / "2f85.xml"),
        )
        assets = {
            a.file.get_vfs_filename(): a.file.contents
            for a in arena.asset.all_children()
            if a.tag == "mesh"
        }
        self.model = mujoco.MjModel.from_xml_string(arena.to_xml_string(), assets)
        self.data = mujoco.MjData(self.model)
        self.nu = self.model.nu
        self.data.qpos[:6] = q0
        self.data.ctrl[:6] = q0
        mujoco.mj_forward(self.model, self.data)
        self.steps_per_cmd = int(round(1.0 / HZ / self.model.opt.timestep))

    def num_dofs(self):
        return self.nu

    def get_joint_state(self):
        return self.data.qpos[: self.nu].copy()

    def command_joint_state(self, cmd):
        cmd = np.array(cmd, dtype=float)
        cmd[-1] *= 255  # same gripper scaling as MujocoRobotServer
        self.data.ctrl[:] = cmd
        for _ in range(self.steps_per_cmd):
            self.mujoco.mj_step(self.model, self.data)


@pytest.mark.skipif(not (MENAGERIE / "universal_robots_ur5e").exists(), reason="no menagerie")
@pytest.mark.parametrize(
    "q0",
    [
        np.zeros(6),  # the "weird" all-zero pose the sim starts in
        np.array([1.0, -2.0, -1.5, 0.5, 1.2, -2.0]),
    ],
)
def test_mujoco_ur_returns_home(q0):
    robot = SteppedMujocoUR(q0)
    clock = FakeClock()
    result = run(robot, clock, UR_HOME_JOINTS, settle_timeout=5.0)
    assert result.success, result.message
    assert np.allclose(robot.get_joint_state()[:6], UR_HOME_JOINTS, atol=0.01)
