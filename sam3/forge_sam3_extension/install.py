from __future__ import annotations

import importlib.util
from importlib.metadata import version

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
    ]
    missing = [pkg for pkg, low, high in required if not is_installed(pkg, low, high)]
    if missing:
        joined = ", ".join(missing)
        print(
            "[forge_sam3_extension] missing dependencies detected: "
            f"{joined}. Install them manually in the Forge venv before enabling SAM3."
        )


check_environment()
