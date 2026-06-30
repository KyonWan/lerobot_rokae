# rokae_policy_runtime

Runtime bridges between **Rokae** hardware and policy servers (e.g. **OpenPI** over WebSocket). Other policy adapters can live in this package later.

## Install

Requires Python >= 3.10. This package is experimental/debug-only for now. It is not installed by the default `requirements.txt` and is not recommended for regular recording or teleoperation users.

From the **repository root**:

```bash
pip install -r requirements-policy.txt
```

Or from this directory:

```bash
cd rokae_policy_runtime
pip install -e .
```

`pyproject.toml` pulls in `lerobot`, `lerobot_robot_rokae`, `openpi-client`, and common numeric/image deps. If you develop `lerobot` / `lerobot_robot_rokae` from local checkouts, install those in editable mode first so imports resolve.

## OpenPI bridge (single-arm Rokae)

- **CLI:** `rokae-openpi` or `python -m rokae_policy_runtime.openpi.cli`
- **Details (ZMQ server, cameras, observation format):** [src/rokae_policy_runtime/openpi/README.md](src/rokae_policy_runtime/openpi/README.md)

```bash
rokae-openpi --help
```
