# Record Configuration Examples

Each YAML is a complete example for one `teleop.type`. Do not mix teleoperator
fields — draccus will reject unknown keys.

**SpaceMouse** — USB device indices:

```yaml
teleop:
  type: bi_spacemouse
  left_device_index: 0
  right_device_index: 1
```

**Pico** — map controllers to arms (`type: pico_single` + `side: left|right` for single-arm):

```yaml
teleop:
  type: pico
  left_arm_controller: right
  right_arm_controller: left
  R_headset_world: [90.0, 0.0, 90.0]
```

Dual-arm presets:

- Facing the robot: `left_arm_controller: right`, `right_arm_controller: left`, `R_headset_world: [90.0, 0.0, 90.0]`
- Same direction as robot: swap controllers, `R_headset_world: [90.0, 0.0, -90.0]`

## CPU affinity (recommended)

Bind **server** and **recording** to separate cores; **recording must not share cores with server processes**.

| Process | Config | Example cores |
|---------|--------|---------------|
| Arm ZMQ server | `rokae_python_wrapper/config/server/*.yaml` → `cpu` | `2`, `6` |
| Gripper server | same → `end_effector.cpu` | `4`, `8` |
| Recording | `export RECORD_TASKSET_CPUS=…` before `rokae_record.sh` | `9` |

```bash
# rokae_python_wrapper/
./scripts/rokae_run.sh --config config/server/dual.example.yaml

# repo root — pick a core not used by server YAML above
export RECORD_TASKSET_CPUS=9
./scripts/record/rokae_record.sh --config_path=config/record/dual_spacemouse_example.yaml
```

Child arm/gripper processes follow `cpu` / `end_effector.cpu` in the server YAML (not `ARM_CONTROL_CPUS` in `rokae_run.sh`, which only pins the launcher). Adjust core IDs for your machine; prefer `isolcpus` when available.
