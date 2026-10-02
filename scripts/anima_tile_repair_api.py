"""Registers the Anima Tile & Repair JSON routes at app start.

``POST /sam-extra/tile-repair`` (+ ``GET .../options``, ``POST .../stop``) run the Tile-Repair panel's
pipeline without the Gradio panel — see ``sam3ext/tile_repair_api.py``. No always-on script, no
UI: this file only hooks ``on_app_started`` (it also fires for ``--nowebui`` API-only launches).
"""
from __future__ import annotations

import sys
import traceback

from modules import script_callbacks

from sam3ext.tile_repair_api import TILE_REPAIR_API_PATH, register_tile_repair_routes


def on_app_started_tile_repair_api(demo, app) -> None:
    try:
        if register_tile_repair_routes(app):
            print(f"[SAM3 Anima] Tile-Repair API: {TILE_REPAIR_API_PATH}")
    except Exception:
        traceback.print_exc(file=sys.stderr)


script_callbacks.on_app_started(on_app_started_tile_repair_api, name="sam-extra-tile-repair-api")
