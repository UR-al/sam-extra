"""Colorcraft (sam-extra) — latent colour grading during sampling, as an always-on Forge script.

A port of muerrilla/ComfyUI-Colorcraft@d28ac6a (MIT, Copyright (c) 2026 Sahand Ahmadian Tehrani
(Muerrilla)) on the Forge Neo structure of aoleg/ComfyUI-Colorcraft@f00066c (MIT, Oleg Afonin); the
library and every change made to it are in ``sam3ext/colorcraft/`` (see the module headers).

This file is the WebUI seam only:

* ``ui`` builds the panel (``sam3ext.colorcraft.ui``) — one collapsed accordion, ``panel_state.ARG_COUNT``
  (67) script arguments in ``panel_state.ARG_NAMES`` order;
* ``process_before_every_sampling`` attaches the post-CFG function to a clone of the pass's UNet
  (``sam3ext.colorcraft.hook.process``); it is appended last, after every other sam-extra post-CFG
  function of the pass, because this file loads after the ``anima_*`` scripts and Forge calls
  ``process_before_every_sampling`` in load order;
* XYZ axes ``[Colorcraft] …`` (modifier I and the master switch) and two Settings entries;
* the Debug panel's capture (kept from ``postprocess``) and its Refresh button.

The title is not upstream's "Colorcraft", so the two can be installed side by side, and the infotext
key is this extension's own (``SAM Extra Colorcraft``); pasting reads upstream's and the fork's
``Colorcraft`` key too.
"""
from __future__ import annotations

import sys
import traceback
from functools import partial

from modules import script_callbacks, scripts

from sam3ext import layout_lanes
from sam3ext.colorcraft import debug_panel, hook, spec, ui

TITLE = "Colorcraft (sam-extra)"
XYZ_PREFIX = "[Colorcraft]"
SETTINGS_SECTION = ("sam3_colorcraft", "SAM Extra Colorcraft")


def _log(message: str) -> None:
    print(f"[Colorcraft] {message}", file=sys.stderr)


def _warn(message: str) -> None:
    """The Debug renderer's warnings, which carry their own ``[Colorcraft] Debug:`` prefix."""
    print(message, file=sys.stderr)


# ---------------------------------------------------------------------------
# XYZ plot
# ---------------------------------------------------------------------------

XYZ_AXES = (
    ("Enable", "enabled", str),
    ("Strength", "strength", float),
    ("Start", "start", float),
    ("End", "end", float),
    ("Exposure", "exposure", float),
    ("Tone Compression", "tone_compression", float),
    ("Contrast", "contrast", float),
    ("Clarity", "clarity", float),
    ("Sharpness", "sharpness", float),
    ("Temperature", "temperature", float),
    ("Tint", "tint", float),
    ("Vibrance", "vibrance", float),
    ("Saturation", "saturation", float),
    ("Chroma Contrast", "chroma_contrast", float),
)


def _xyz_set(p, x, xs, *, field: str):
    # A new dict each time: xyz_grid hands every cell a shallow copy of p, so a dict found on p may be
    # shared with other cells.
    values = dict(getattr(p, spec.XYZ_ATTR, None) or {})
    values[field] = x
    setattr(p, spec.XYZ_ATTR, values)


def register_xyz_axes(xyz_grid) -> None:
    """Add the ``[Colorcraft]`` axes; on a WebUI reload replace ours in place (same index).

    xyz_grid keeps a chosen axis by its index in ``axis_options``, so the list is never reordered and
    a reload neither duplicates the axes nor leaves them pointing at the old module's functions."""
    bool_choices = lambda: ["True", "False"]  # noqa: E731
    fresh = []
    for label, field, kind in XYZ_AXES:
        kwargs = {"choices": bool_choices} if field == "enabled" else {}
        fresh.append(xyz_grid.AxisOption(f"{XYZ_PREFIX} {label}", kind, partial(_xyz_set, field=field), **kwargs))
    index_by_label = {getattr(option, "label", None): i for i, option in enumerate(xyz_grid.axis_options)}
    for option in fresh:
        index = index_by_label.get(option.label)
        if index is None:
            xyz_grid.axis_options.append(option)
        else:
            xyz_grid.axis_options[index] = option


def _find_xyz_grid():
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            return script.module
    return None


def _on_before_ui() -> None:
    try:
        xyz_grid = _find_xyz_grid()
        if xyz_grid is not None:
            register_xyz_axes(xyz_grid)
    except Exception:
        _log("xyz_grid axis registration failed:\n" + traceback.format_exc())


script_callbacks.on_before_ui(_on_before_ui)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _on_ui_settings() -> None:
    import gradio as gr
    from modules import shared

    shared.opts.add_option(
        hook.OPT_PRE_DD,
        shared.OptionInfo(
            True,
            "Colorcraft + Detail Daemon: 스케줄 위치를 Detail Daemon 이 바꾸기 전 σ 로 찾기",
            gr.Checkbox,
            section=SETTINGS_SECTION,
            infotext=hook.INFOTEXT_PRE_DD,
        ).info(
            "Detail Daemon 은 모델에 넘기는 σ 를 줄입니다. Colorcraft 는 매 호출의 σ 로 스케줄(Start·End 곡선) 위치를 "
            "찾으므로, 줄어든 σ 를 쓰면 위치가 조금 뒤로 밀립니다. 켜면(기본) 샘플러 자신의 σ 로 찾아 스텝에 맞춥니다. "
            "Detail Daemon 을 끈 생성은 켜고 끔에 관계없이 같습니다. 끄면 ComfyUI 에서 두 노드를 이어 쓴 것과 같습니다. "
            "실제로 바뀐 생성에만 infotext 에 남깁니다."
        ),
    )
    shared.opts.add_option(
        hook.OPT_LOG,
        shared.OptionInfo(
            False,
            "Colorcraft: 패스마다 σ 목록과 탭별 스케줄 값을 콘솔에 출력",
            gr.Checkbox,
            section=SETTINGS_SECTION,
        ).info("결과는 같습니다. 스케줄이 어느 스텝에 얼마나 걸리는지 확인할 때만 켜세요."),
    )


script_callbacks.on_ui_settings(_on_ui_settings)


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------


class Colorcraft(scripts.Script):
    # Right under Anima Optimal Scale (-25) in the ANIMA tuning section; UI order only.
    sorting_priority = -23

    @property
    def section(self):
        return layout_lanes.anima_section(bool(getattr(self, "is_img2img", False)))

    def title(self):
        return TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def _elem_id(self, item_id):
        try:
            return self.elem_id(item_id)
        except Exception:
            tab = "img2img" if getattr(self, "is_img2img", False) else "txt2img"
            return f"script_{tab}_colorcraft_samextra_{item_id}"

    def ui(self, is_img2img):
        components, fields = ui.build(self._elem_id, on_debug_refresh=self._on_debug_refresh)
        self.infotext_fields = fields
        return components

    # Debug panel: the generation leaves its capture on ``p``; this tab's script keeps the latest one
    # for the Refresh button (upstream keeps it on the script too).
    debug_capture = None

    def postprocess(self, p, processed, *args):
        state = getattr(p, hook.STATE_ATTR, None)
        capture = state.get("debug") if isinstance(state, dict) else None
        if capture is not None:
            self.debug_capture = capture

    def _on_debug_refresh(self, axes, masks, combos, composite, overlay, axis_style, *mask_values):
        try:
            return debug_panel.render_for_panel(self.debug_capture, axes, masks, combos, composite, overlay,
                                                axis_style, mask_values, warn=_warn)
        except Exception as exc:
            _log(f"debug render failed: {type(exc).__name__}: {exc}")
            return []

    def process_before_every_sampling(self, p, *args, **kwargs):
        try:
            # Forge passes the pass's latent as ``x`` (processing.py); its grid scales the mask blur on
            # Anima SPEED's coarse steps (hook.grid_downscale).
            hook.process(p, args, full_hw=hook.latent_grid(kwargs.get("x")))
        except Exception as exc:  # never break a generation
            _log(f"not applied: {type(exc).__name__}: {exc}")
            params = getattr(p, "extra_generation_params", None)
            if isinstance(params, dict) and spec.INFOTEXT_KEY in params:
                params[spec.STATUS_KEY] = f"not applied: {type(exc).__name__}"
