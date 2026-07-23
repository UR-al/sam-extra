"""Public SAM3 API with lightweight, lazy imports.

Keeping package initialization small lets standalone extension scripts import
``sam3ext.guidance`` without initializing SAM3 checkpoints, image processing,
or optional model dependencies.
"""

from __future__ import annotations

from .__version__ import __version__

__all__ = [
    "ALL_ARGS",
    "SAM3_NAME",
    "Sam3Args",
    "Sam3Result",
    "__version__",
    "find_checkpoint_options",
    "run_sam3_on_pil",
]


def __getattr__(name: str):
    if name in {"ALL_ARGS", "Sam3Args"}:
        from .args import ALL_ARGS, Sam3Args

        return {"ALL_ARGS": ALL_ARGS, "Sam3Args": Sam3Args}[name]
    if name in {
        "SAM3_NAME",
        "Sam3Result",
        "find_checkpoint_options",
        "run_sam3_on_pil",
    }:
        from .core import (
            SAM3_NAME,
            Sam3Result,
            find_checkpoint_options,
            run_sam3_on_pil,
        )

        return {
            "SAM3_NAME": SAM3_NAME,
            "Sam3Result": Sam3Result,
            "find_checkpoint_options": find_checkpoint_options,
            "run_sam3_on_pil": run_sam3_on_pil,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
