"""Meta-package hook: ``pip install -e .`` editable-installs local monorepo members."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.editable_wheel import editable_wheel

ROOT = Path(__file__).resolve().parent


def _pip_install_editable(project_dir: Path, spec: str = ".") -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", spec],
        cwd=project_dir,
    )


def _pip_requested_policy_extra() -> bool:
    """Detect ``pip install -e '.[policy]'`` from invoking pip command line (Linux)."""
    if any("policy" in arg for arg in sys.argv):
        return True
    try:
        with open(f"/proc/{os.getppid()}/cmdline", "rb") as f:
            cmd = f.read().decode(errors="replace").replace("\0", " ")
        return "[policy]" in cmd or ".[policy]" in cmd
    except OSError:
        return False


def _install_local_editable(*, include_policy: bool) -> None:
    _pip_install_editable(ROOT / "lerobot", ".[intelrealsense]")
    _pip_install_editable(ROOT / "rokae_python_wrapper")
    _pip_install_editable(ROOT / "lerobot_robot_rokae")
    _pip_install_editable(ROOT / "lerobot_teleoperator_rokae")
    if include_policy:
        _pip_install_editable(ROOT / "rokae_policy_runtime")
        _pip_install_editable(ROOT / "lerobot_policy_rokae")


class EditableWheelWithLocalMembers(editable_wheel):
    def run(self) -> None:
        editable_wheel.run(self)
        include_policy = _pip_requested_policy_extra()
        _install_local_editable(include_policy=include_policy)


setup(
    cmdclass={"editable_wheel": EditableWheelWithLocalMembers},
)
