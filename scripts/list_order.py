"""Sampling method and Schedule type lists in family order (v0.33.0).

No Script class, no setting: one ``on_before_ui`` callback sorts Forge's sampler and scheduler lists by
family (``sam3ext/list_order.py``: DPM++ together, Euler together, … the RES4LYF samplers as one block,
"custom" last; Forge's first entries — DPM++ 2M, Automatic — stay first). Forge runs ``before_ui`` after
every script has loaded (so every extension's samplers and schedulers are in the lists) and before it builds
the UI, on each start and each Reload UI, also for ``--nowebui``; the dropdowns, the XYZ choices and
``/sdapi/v1/samplers`` / ``/sdapi/v1/schedulers`` read the lists after that. Names, labels and aliases do
not change, so infotext, API requests and XYZ values work as before.

The file name sorts before ``negpip.py``, which stays the extension's last script.
"""
from __future__ import annotations

from modules import script_callbacks

from sam3ext import list_order


def _on_before_ui() -> None:
    list_order.apply()      # never raises; logs a failure


script_callbacks.on_before_ui(_on_before_ui)
