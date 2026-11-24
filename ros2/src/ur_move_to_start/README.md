# ur_move_to_start

Brings a UR arm driven by `ur_robot_driver` back to a start joint position, e.g. after
stopping the GELLO bridge.

```bash
# driver still running (start_drivers.launch.py), bridge stopped
ros2 run ur_move_to_start move_to_start                       # config/start_position.yaml
ros2 run ur_move_to_start move_to_start \
  --start-file $(ros2 pkg prefix custom_ur_description)/share/custom_ur_description/config/pushing_demo_initial_positions.yaml
ros2 run ur_move_to_start move_to_start --start 0 -90 90 -90 -90 0 --deg
ros2 run ur_move_to_start move_to_start --dry-run             # print plan only
```

How it works: reads `/joint_states`, deactivates any active controller that commands the
arm joints (e.g. `forward_position_controller`), activates
`scaled_joint_trajectory_controller`, and sends a single `FollowJointTrajectory` goal that
the controller interpolates smoothly from rest to rest. The pendant speed slider applies.
Ctrl-C cancels the goal and stops the arm.

Options: `--max-vel` (rad/s, default 0.3), `--max-acc` (default 0.5), `--min-duration`
(s, default 2), `--max-joint-distance` (refuse if a joint is further than this, default
pi), `--tf-prefix`, `--trajectory-controller`, `--restore` (switch the original
controllers back on afterwards), `--yes` (no prompt).

`--restore` makes the bridge's controller active again. Only use it if the bridge will
sync smoothly when restarted, otherwise the arm may jump to wherever GELLO is.
