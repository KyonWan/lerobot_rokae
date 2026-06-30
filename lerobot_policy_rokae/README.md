# lerobot_policy_rokae

Non-intrusive LeRobot policy plugin scaffold for Rokae.
Runtime bridge code has been split out to a dedicated package: `rokae_policy_runtime`.

This package provides:

- A reserved custom policy type: `rokae_custom`
- A processor hub scaffold for multi-policy adaptation
- Placeholder profiles for `act`, `diffusion`, `smolvla`, and `xvla`
- Dual canonical contract stubs (`joint` and `ee`)

## Why this package

`lerobot_robot_rokae` and `lerobot_teleoperator_rokae` should stay focused on hardware and teleoperation.
`lerobot_policy_rokae` centralizes policy-side data adaptation so multiple policies can share one pipeline layer.

## Install

This package is experimental/debug-only for now. It is not installed by the default `requirements.txt` and is not recommended for regular recording or teleoperation users.

From the repo root:

```bash
pip install -r requirements-policy.txt
```

If you also run real hardware:

```bash
pip install -r requirements.txt
```

## Plugin discovery

LeRobot scripts call `register_third_party_plugins()` and auto-import installed packages whose distribution
name starts with:

- `lerobot_robot_`
- `lerobot_camera_`
- `lerobot_teleoperator_`
- `lerobot_policy_`

This package uses `lerobot_policy_rokae`, so it is auto-discovered.

## Current scaffold content

- `configuration_rokae_custom.py`:
  - registers `@PreTrainedConfig.register_subclass("rokae_custom")`
- `modeling_rokae_custom.py`:
  - `RokaeCustomPolicy` skeleton with explicit `NotImplementedError` for model logic
- `processor_rokae_custom.py`:
  - identity pre/post processors as bootstrap
- `processor_hub/`:
  - `contracts.py`: dual-contract stubs
  - `registry.py`: placeholder profiles for `act`/`diffusion`/`smolvla`/`xvla`
  - `steps_*`: feature select, contract map, padding placeholders
  - `factory.py`: unified adaptation pipeline assembler

### Field conventions aligned to Rokae plugins

The processor hub profiles now follow the keys produced by your current robot plugins:

- Single-arm: `joint_pos*`, `cart_pos*`, `gripper_pos`, camera keys
- Bimanual: `left_joint_pos*`, `right_joint_pos*`, `left_cart_pos*`, `right_cart_pos*`,
  `left_gripper_pos`, `right_gripper_pos`, camera keys

After `batch_to_transition`, these appear as observation keys with `observation.` prefix
(e.g. `observation.joint_pos0`, `observation.left_cart_pos3`).

Default profile behavior:

- `act` / `diffusion`: keep joint-space + gripper + images
- `smolvla` / `xvla`: keep cartesian-space + gripper + images

## Minimal acceptance checklist

- `pip install -e lerobot_policy_rokae` succeeds
- `lerobot-train --help` runs without plugin import errors
- `rokae_custom` is discoverable as a policy type
- processor hub profiles include: `act`, `diffusion`, `smolvla`, `xvla`

## Next TODOs

1. Implement real tensor mapping for joint <-> ee contracts.
2. Replace placeholder selection/padding with shape-aware transforms.
3. Add per-policy feature validation against dataset metadata.
4. Add unit tests for profile routing and step serialization.

## Runtime migration note

Old command path:

```bash
python -m lerobot_policy_rokae.cli.run_rokae_openpi --help
```

New command path:

```bash
python -m rokae_policy_runtime.openpi.cli --help
rokae-openpi --help
```
