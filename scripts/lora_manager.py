"""LoRA Manager bridge (introduced in v0.9.0).

Registers two things:

1. Settings (Settings → SAM3 LoRA Manager): tab placement mode (add a
   "Manage" tab next to LoRA, or replace the LoRA tab) + the standalone
   server port.

2. A hidden Gradio "bridge": two invisible buttons + two output textboxes.
   ``javascript/lora_manager.js`` clicks these by elem_id to (a) fetch the
   current config (tab mode / port / availability) at page load and (b)
   lazily spawn the standalone server the first time the user opens the
   Manage tab. The handler returns JSON into the paired textbox, which the
   JS reads back.

The actual server lifecycle lives in ``sam3ext.lora_manager_core``.
"""
from __future__ import annotations

import json
import sys
import traceback

import gradio as gr

from modules import script_callbacks, scripts, shared

from sam3ext.lora_manager_core import (
    DEFAULT_PORT,
    get_or_spawn,
    lora_manager_available,
)


OPT_TAB_MODE = "sam3_lora_manager_tab_mode"
OPT_PORT = "sam3_lora_manager_port"

_TAB_MODE_ADD = "Add Manage tab (keep LoRA)"
_TAB_MODE_REPLACE = "Replace LoRA tab"


def on_ui_settings():
    section = ("sam3_lora_manager", "SAM3 LoRA Manager")
    shared.opts.add_option(
        OPT_TAB_MODE,
        shared.OptionInfo(
            _TAB_MODE_ADD,
            "Manage 탭 배치 (extra-networks strip)",
            gr.Radio,
            {"choices": [_TAB_MODE_ADD, _TAB_MODE_REPLACE]},
            section=section,
        ),
    )
    shared.opts.add_option(
        OPT_PORT,
        shared.OptionInfo(
            DEFAULT_PORT,
            "LoRA Manager standalone 서버 포트 (재시작 후 적용)",
            section=section,
        ),
    )


script_callbacks.on_ui_settings(on_ui_settings)


def _read_opts() -> tuple[str, int]:
    tab_mode = getattr(shared.opts, OPT_TAB_MODE, _TAB_MODE_ADD)
    try:
        port = int(getattr(shared.opts, OPT_PORT, DEFAULT_PORT) or DEFAULT_PORT)
    except (TypeError, ValueError):
        port = DEFAULT_PORT
    return tab_mode, port


def _config_handler() -> str:
    """Cheap — no spawn. JS calls this once on load to decide tab placement."""
    tab_mode, port = _read_opts()
    return json.dumps(
        {
            "available": lora_manager_available(),
            "replace": tab_mode == _TAB_MODE_REPLACE,
            "port": port,
        }
    )


def _spawn_handler() -> str:
    """Lazy spawn — JS calls this the first time the Manage tab is shown."""
    _tab_mode, port = _read_opts()
    try:
        res = get_or_spawn(port)
    except Exception as e:  # pragma: no cover - defensive
        traceback.print_exc(file=sys.stderr)
        res = {"url": "", "port": port, "status": "error", "message": str(e)}
    return json.dumps(res)


class LoraManagerBridge(scripts.Script):
    alwayson = True
    # No accordion — this script contributes only an invisible bridge, so
    # suppress the group wrapper Forge would otherwise draw in the scripts
    # column (see modules/scripts.py: create_group docstring).
    create_group = False

    def title(self):
        return "SAM3 LoRA Manager bridge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # Build the hidden bridge once (t2i pass). The JS addresses these by
        # global elem_id, so a single set serves both txt2img and img2img
        # extra-networks strips.
        if is_img2img:
            return []
        with gr.Group(visible=False):
            cfg_btn = gr.Button(value="lm_config", elem_id="sam3_lm_config_btn")
            cfg_out = gr.Textbox(value="", elem_id="sam3_lm_config_out")
            spawn_btn = gr.Button(value="lm_spawn", elem_id="sam3_lm_spawn_btn")
            spawn_out = gr.Textbox(value="", elem_id="sam3_lm_spawn_out")
        cfg_btn.click(fn=_config_handler, inputs=[], outputs=[cfg_out])
        spawn_btn.click(fn=_spawn_handler, inputs=[], outputs=[spawn_out])
        return []
