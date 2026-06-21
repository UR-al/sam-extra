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


# ---------------------------------------------------------------------------
# v0.9.0 LoRA Manager vendor bootstrap — shallow-clone willmiao/
# ComfyUI-Lora-Manager + auto-install its (lightweight, pure-python) deps.
# ---------------------------------------------------------------------------
# Unlike the Anima vendor, the LoRA Manager's extra deps are all small
# pure-python packages (aiohttp-socks, piexif, olefile, natsort, aiosqlite,
# beautifulsoup4) with no torch/cuda ABI coupling, so auto-installing the
# missing ones is safe and was explicitly opted into by the user.

_LM_REPO = "https://github.com/willmiao/ComfyUI-Lora-Manager.git"
_LM_BRANCH = "main"
_LM_ROOT = Path(__file__).resolve().parent / "lora_manager_vendor"
_LM_SENTINEL = _LM_ROOT / "standalone.py"

# pip name -> import name (for is_installed's find_spec check)
_LM_DEP_IMPORT = {
    "aiohttp": "aiohttp",
    "aiohttp-socks": "aiohttp_socks",
    "jinja2": "jinja2",
    "safetensors": "safetensors",
    "piexif": "piexif",
    "Pillow": "PIL",
    "olefile": "olefile",
    "toml": "toml",
    "numpy": "numpy",
    "natsort": "natsort",
    "GitPython": "git",
    "aiosqlite": "aiosqlite",
    "beautifulsoup4": "bs4",
    "platformdirs": "platformdirs",
    "pyyaml": "yaml",
    "brotli": "brotli",
}


def ensure_lora_manager_vendor() -> bool:
    """Clone willmiao/ComfyUI-Lora-Manager into ``lora_manager_vendor/`` if
    the sentinel (standalone.py) is missing. Idempotent.

    Returns True when the vendor tree is usable after this call.
    """
    if _LM_SENTINEL.exists():
        return True

    if not _have_git():
        print(
            "[forge_sam3_extension] LoRA Manager disabled: 'git' not on PATH. "
            "Install git, or clone willmiao/ComfyUI-Lora-Manager manually "
            f"into {_LM_ROOT}",
            file=sys.stderr,
        )
        return False

    _LM_ROOT.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[forge_sam3_extension] cloning ComfyUI-Lora-Manager → {_LM_ROOT} "
        "(first-run, ~20s)",
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
                _LM_BRANCH,
                _LM_REPO,
                str(_LM_ROOT),
            ],
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.CalledProcessError as e:
        print(
            f"[forge_sam3_extension] LoRA Manager clone failed (exit "
            f"{e.returncode}); the Manage tab will be disabled.",
            file=sys.stderr,
        )
        return False

    return _LM_SENTINEL.exists()


def check_lora_manager_environment():
    """Auto-install the LoRA Manager's missing pure-python deps into the
    Forge venv (user opted into auto-install). No version pins — we only add
    packages that are entirely absent so existing Forge packages are never
    downgraded."""
    if not _LM_SENTINEL.exists():
        return

    # Seed import_name so is_installed resolves the non-obvious ones.
    for pip_name, imp in _LM_DEP_IMPORT.items():
        import_name.setdefault(pip_name, imp)

    missing = [pip for pip in _LM_DEP_IMPORT if not is_installed(pip)]
    if not missing:
        return

    joined = " ".join(missing)
    print(
        f"[forge_sam3_extension] LoRA Manager: installing missing deps "
        f"({', '.join(missing)}) into the Forge venv...",
        file=sys.stderr,
    )
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing],
            env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
        )
    except subprocess.CalledProcessError as e:
        print(
            f"[forge_sam3_extension] LoRA Manager dep install failed (exit "
            f"{e.returncode}). Install manually:\n   pip install {joined}",
            file=sys.stderr,
        )


ensure_lora_manager_vendor()
check_lora_manager_environment()
