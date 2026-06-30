# Record Configuration Examples

Each YAML file is a complete example for one `teleop.type`. Do not mix fields
from different teleoperators: LeRobot/draccus uses a strict typed schema, so
unknown fields should fail fast instead of being ignored.

SpaceMouse examples use USB device indices:

```yaml
teleop:
  type: bi_spacemouse
  left_device_index: 0
  right_device_index: 1
```

Pico examples use arm-to-controller mapping:

```yaml
teleop:
  type: pico
  left_arm_controller: right
  right_arm_controller: left
  R_headset_world: [90.0, 0.0, 90.0]
```

For dual-arm Pico teleoperation:

- Facing the robot: use `left_arm_controller: right`,
  `right_arm_controller: left`, and `R_headset_world: [90.0, 0.0, 90.0]`.
- Standing in the same direction as the robot: use `left_arm_controller: left`,
  `right_arm_controller: right`, and `R_headset_world: [90.0, 0.0, -90.0]`.

For single-arm Pico teleoperation, use `type: pico_single` and select the
controller with `side: left` or `side: right`. `R_headset_world` is configurable
in the same way.
