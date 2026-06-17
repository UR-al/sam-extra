from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from packaging.version import parse


import_name = {
    "opencv-python": "cv2",
    "py-cpuinfo": "cpuinfo",
    "protobuf": "google.protobuf",
}


def is_installed(package: str, min_version: str | None = None, max_version: str | None = None):
    name = import_name.get(package, package)
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return False

    if spec is None:
        return False

    if not min_version and not max_version:
        return True

    if not min_version:
        min_version = "0.0.0"
    if not max_version:
        max_version = "99999999.99999999.99999999"

    try:
        pkg_version = version(package)
        return parse(min_version) <= parse(pkg_version) <= parse(max_version)
    except Exception:
        return False


def check_environment():
    required = [
        ("sam3", None, None),
        ("torch", None, None),
        ("opencv-python", None, None),
        ("timm", None, None),
        ("einops", None, None),
        ("huggingface_hub", None, None),
        ("iopath", None, None),
        ("safetensors", None, None),
    ]
    missing = [pkg for pkg, low, high in required if not is_installed(pkg, low, high)]
    if missing:
        joined = ", ".join(missing)
        print(
            "[forge_sam3_extension] missing dependencies detected: "
            f"{joined}. Install them manually in the Forge venv before enabling SAM3."
        )


check_environment()


# ---------------------------------------------------------------------------
# v0.8.0 Anima vendor bootstrap — shallow-clone kohya-ss/sd-scripts once.
# ---------------------------------------------------------------------------
# Forge runs install.py as a subprocess at extension load (launch_utils.py),
# so this happens before scripts/!sam3.py ever imports. Failure is non-fatal:
# the Anima panel just hides itself, the rest of the SAM3 extension still
# works.

_ANIMA_REPO = "https://github.com/kohya-ss/sd-scripts.git"
_ANIMA_BRANCH = "main"
_ANIMA_ROOT = Path(__file__).resolve().parent / "anima_vendor"
# Sentinel = the actual entrypoint upstream ships. If the clone was
# interrupted partway through, this file will be missing and the next run
# re-attempts cleanly.
_ANIMA_SENTINEL = _ANIMA_ROOT / "anima_minimal_inference.py"


def _have_git() -> bool:
    try:
        subprocess.check_call(
            ["git", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def ensure_anima_vendor() -> bool:
    """Clone kohya-ss/sd-scripts into ``anima_vendor/`` if the sentinel file
    is missing. Idempotent.

    Returns True when the vendor tree is usable after this call, False when
    bootstrap failed (the Anima panel reads ``anima_available()`` to decide
    whether to render).
    """
    if _ANIMA_SENTINEL.exists():
        return True

    if not _have_git():
        print(
            "[forge_sam3_extension] Anima panel disabled: 'git' not on PATH. "
            "Install git, or clone kohya-ss/sd-scripts manually into "
            f"{_ANIMA_ROOT}",
            file=sys.stderr,
        )
        return False

    _ANIMA_ROOT.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[forge_sam3_extension] cloning sd-scripts → {_ANIMA_ROOT} "
        "(first-run, ~30s)",
        file=sys.stderr,
    )
    try:
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--branch",
                _ANIMA_BRANCH,
                _ANIMA_REPO,
                str(_ANIMA_ROOT),
            ],
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.CalledProcessError as e:
        print(
            f"[forge_sam3_extension] Anima vendor clone failed (exit "
            f"{e.returncode}); the Anima panel will be disabled.",
            file=sys.stderr,
        )
        return False

    return _ANIMA_SENTINEL.exists()


ensure_anima_vendor()


# Anima vendor pulls a few packages that aren't part of the SAM3 core deps.
# We don't auto-pip these (torchvision in particular needs to match the
# installed torch+cuda ABI, and forcing a wrong wheel breaks Forge itself).
# Just emit a one-line breadcrumb so the user knows what to install before
# clicking ▶ Anima Tile-Repair.
_ANIMA_DEPS = (
    "torchvision",  # match torch CUDA build manually
    "imagesize",
    "accelerate",
    "transformers",
    "diffusers",
    "einops",
    "huggingface_hub",
    "safetensors",
)


def check_anima_environment():
    if not _ANIMA_SENTINEL.exists():
        return  # vendor missing — panel won't render anyway
    missing = [pkg for pkg in _ANIMA_DEPS if not is_installed(pkg)]
    if missing:
        joined = " ".join(missing)
        print(
            "[forge_sam3_extension] Anima panel: missing deps "
            f"({', '.join(missing)}). Install in the Forge venv before "
            f"clicking ▶ Anima Tile-Repair, e.g.\n"
            f"   pip install {joined}\n"
            f"(For torchvision, match your installed torch's CUDA build.)",
            file=sys.stderr,
        )


check_anima_environment()
