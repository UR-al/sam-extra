"""Anima Detail Daemon — standalone detail control for Forge Neo.

A SELF-CONTAINED fork of muerrilla's ``sd-webui-Detail-Daemon`` (schedule +
sigma-adjustment math reimplemented), redesigned for an intuitive UX and wired
into our own panel. Independent of SAM3 and of the PAG/APG script; touches no
Forge core file.

How it works
------------
It keeps the injected noise the same but **lowers how much noise is removed at
each step**, which makes the model leave more high-frequency content in → more
detail. Mechanically it multiplies the per-step ``sigma`` the denoiser sees by a
bell-shaped schedule:

    sigma *= 1 - schedule[step] * amount * (cfg_scale if coupled else 1)

Positive ``amount`` → sigma lowered → more detail; negative → smoother / less
bokeh-noise. Zero (or disabled) → exact no-op.

Hook
----
Forge fires ``on_cfg_denoiser(params)`` each denoise step with a
``CFGDenoiserParams`` carrying ``.sigma``, ``.sampling_step`` and
``.total_sampling_steps``. We register one global callback that reads the shared
``_DD`` state (set per-generation by the script's ``process_before_every_sampling``)
and adjusts ``params.sigma`` in place. Everything is guarded; on any error it
leaves sigma untouched, so enabling this can never break a generation.

UX principles (per request)
---------------------------
- Easy by default: **Enable + one "Detail amount" slider + presets**.
- Deep when wanted: the full muerrilla curve (start/end/bias/exponent/offsets/
  fade/smooth/multiplier) lives in an "Advanced" accordion.
- Every automatic behavior is a toggle (e.g. "couple to CFG scale").

Model-agnostic: works on Anima (RF) and any other engine, since it only scales
sampler sigmas.
"""
from __future__ import annotations

import sys
import traceback
from functools import partial

import gradio as gr

from modules import script_callbacks, scripts

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _log(msg: str) -> None:
    print(f"[AnimaDetailDaemon] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Shared state (set per generation; read by the global denoiser callback).
# ---------------------------------------------------------------------------

_DD: dict = {
    "on": False,
    "amount": 0.10,
    "start": 0.2,
    "end": 0.8,
    "bias": 0.5,
    "exponent": 1.0,
    "start_offset": 0.0,
    "end_offset": 0.0,
    "fade": 0.0,
    "smooth": True,
    "multiplier": 1.0,     # global strength on top of the schedule
    "cfg_couple": True,    # multiply the effect by cfg_scale (muerrilla "both")
    "cfg_scale": 1.0,      # captured from p each generation
    "sched": None,         # cached schedule array
    "sched_key": None,     # (steps, params…) the cache was built for
}

_PRESETS = {  # preset → detail amount
    "Subtle": 0.05,
    "Medium": 0.10,
    "Strong": 0.25,
}


# ---------------------------------------------------------------------------
# Schedule construction (faithful reimplementation of muerrilla's make_schedule)
# ---------------------------------------------------------------------------


def _make_schedule(steps, start, end, bias, amount, exponent,
                   start_offset, end_offset, fade, smooth):
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = (
        int(round(x * (steps - 1))) for x in (start, mid, end)
    )

    start_values = np.linspace(0, 1, max(0, mid_idx - start_idx + 1))
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    start_values = start_values ** exponent
    if start_values.any():
        start_values *= (amount - start_offset)
        start_values += start_offset

    end_values = np.linspace(1, 0, max(0, end_idx - mid_idx + 1))
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    end_values = end_values ** exponent
    if end_values.any():
        end_values *= (amount - end_offset)
        end_values += end_offset

    multipliers[start_idx:mid_idx + 1] = start_values
    multipliers[mid_idx:end_idx + 1] = end_values
    multipliers[:start_idx] = start_offset
    multipliers[end_idx + 1:] = end_offset
    multipliers *= 1 - fade
    return multipliers


def _get_schedule(steps: int):
    key = (
        steps, _DD["amount"], _DD["start"], _DD["end"], _DD["bias"],
        _DD["exponent"], _DD["start_offset"], _DD["end_offset"], _DD["fade"],
        _DD["smooth"],
    )
    if _DD["sched"] is not None and _DD["sched_key"] == key:
        return _DD["sched"]
    try:
        sched = _make_schedule(
            steps, _DD["start"], _DD["end"], _DD["bias"], _DD["amount"],
            _DD["exponent"], _DD["start_offset"], _DD["end_offset"],
            _DD["fade"], _DD["smooth"],
        )
    except Exception as e:
        _log(f"schedule build failed: {type(e).__name__}: {e}")
        sched = None
    _DD["sched"] = sched
    _DD["sched_key"] = key
    return sched


# ---------------------------------------------------------------------------
# The global denoiser callback (registered once at import)
# ---------------------------------------------------------------------------


def _denoiser_callback(params) -> None:
    if not _DD["on"] or np is None:
        return
    try:
        sigma = getattr(params, "sigma", None)
        if sigma is None:
            return
        steps = int(getattr(params, "total_sampling_steps", 0) or 0)
        if steps <= 0:
            return
        sched = _get_schedule(steps)
        if sched is None or len(sched) == 0:
            return
        step = int(getattr(params, "sampling_step", 0) or 0)
        idx = min(max(step, 0), len(sched) - 1)
        mult = float(sched[idx]) * float(_DD["multiplier"])
        if mult == 0.0:
            return
        cfg = float(_DD["cfg_scale"]) if _DD["cfg_couple"] else 1.0
        factor = 1.0 - mult * cfg
        # Clamp so a stray large amount can't zero-out or explode the sigma.
        factor = min(3.0, max(0.05, factor))
        params.sigma = sigma * factor
    except Exception as e:
        _log(f"denoiser callback skipped: {type(e).__name__}: {e}")


script_callbacks.on_cfg_denoiser(_denoiser_callback)


# ---------------------------------------------------------------------------
# XYZ plot integration
# ---------------------------------------------------------------------------


def _dd_xyz_set(p, x, xs, *, field: str):
    if not hasattr(p, "_anima_detail_daemon_xyz"):
        p._anima_detail_daemon_xyz = {}
    p._anima_detail_daemon_xyz[field] = x


def _make_dd_xyz_axis() -> None:
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break
    if xyz_grid is None:
        return
    bool_choices = lambda: ["True", "False"]  # noqa: E731
    axis = [
        xyz_grid.AxisOption(
            "[Detail Daemon] Enable", str,
            partial(_dd_xyz_set, field="enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption("[Detail Daemon] Amount", float, partial(_dd_xyz_set, field="amount")),
        xyz_grid.AxisOption("[Detail Daemon] Start", float, partial(_dd_xyz_set, field="start")),
        xyz_grid.AxisOption("[Detail Daemon] End", float, partial(_dd_xyz_set, field="end")),
        xyz_grid.AxisOption("[Detail Daemon] Bias", float, partial(_dd_xyz_set, field="bias")),
    ]
    if not any(a.label.startswith("[Detail Daemon]") for a in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)


def _dd_on_before_ui() -> None:
    try:
        _make_dd_xyz_axis()
    except Exception:
        _log("xyz_grid axis registration failed:\n" + traceback.format_exc())


script_callbacks.on_before_ui(_dd_on_before_ui)


def _as_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


# ---------------------------------------------------------------------------
# The extension script
# ---------------------------------------------------------------------------


class AnimaDetailDaemon(scripts.Script):
    sorting_priority = 97

    def title(self):
        return "Anima Detail Daemon"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Anima Detail Daemon (디테일 조정)", open=False):
            gr.Markdown(
                "매 스텝 **제거하는 노이즈량을 줄여** 디테일·질감을 늘립니다(배경 뽀샤시↓). "
                "추가 forward 없이 sampler sigma만 조정하며 **모든 모델에서 동작**합니다. "
                "양수=디테일↑, 음수=매끈, 0/끄면 완전 무효. PAG·APG와 독립이라 같이 써도 됩니다."
            )
            enabled = gr.Checkbox(
                label="Enable Detail Daemon",
                value=False,
                elem_id="anima_dd_enable",
            )
            preset = gr.Radio(
                label="Preset (Custom이면 아래 Amount 사용)",
                choices=["Custom", "Subtle", "Medium", "Strong"],
                value="Medium",
                elem_id="anima_dd_preset",
            )
            amount = gr.Slider(
                label="Detail amount (음수=매끈 · 양수=디테일↑)",
                minimum=-1.0, maximum=1.0, step=0.01, value=0.10,
                elem_id="anima_dd_amount",
            )
            with gr.Accordion("Detail Daemon Advanced (세부값)", open=False):
                gr.Markdown(
                    "적용 구간과 곡선을 세밀 조정합니다. **start/end**=적용 스텝 구간, "
                    "**bias**=피크 위치, **exponent**=곡률, **offset**=구간 밖 기본값, "
                    "**fade**=전체 감쇠, **smooth**=코사인 스무딩, **multiplier**=스케줄 위 "
                    "전역 강도. **couple to CFG**=원본 'both' 모드(효과×CFG scale)."
                )
                with gr.Row():
                    start = gr.Slider(label="Start", minimum=0.0, maximum=1.0, step=0.01, value=0.2, elem_id="anima_dd_start")
                    end = gr.Slider(label="End", minimum=0.0, maximum=1.0, step=0.01, value=0.8, elem_id="anima_dd_end")
                    bias = gr.Slider(label="Bias", minimum=0.0, maximum=1.0, step=0.01, value=0.5, elem_id="anima_dd_bias")
                exponent = gr.Slider(label="Exponent", minimum=0.0, maximum=10.0, step=0.05, value=1.0, elem_id="anima_dd_exponent")
                with gr.Row():
                    start_offset = gr.Slider(label="Start offset", minimum=-1.0, maximum=1.0, step=0.01, value=0.0, elem_id="anima_dd_start_offset")
                    end_offset = gr.Slider(label="End offset", minimum=-1.0, maximum=1.0, step=0.01, value=0.0, elem_id="anima_dd_end_offset")
                fade = gr.Slider(label="Fade", minimum=0.0, maximum=1.0, step=0.05, value=0.0, elem_id="anima_dd_fade")
                multiplier = gr.Slider(label="Multiplier (전역 강도)", minimum=0.0, maximum=2.0, step=0.05, value=1.0, elem_id="anima_dd_multiplier")
                with gr.Row():
                    smooth = gr.Checkbox(label="Smooth (코사인 스무딩)", value=True, elem_id="anima_dd_smooth")
                    cfg_couple = gr.Checkbox(label="Couple to CFG scale (원본 both 모드)", value=True, elem_id="anima_dd_cfg_couple")
        return [
            enabled, preset, amount, start, end, bias, exponent,
            start_offset, end_offset, fade, multiplier, smooth, cfg_couple,
        ]

    def process_before_every_sampling(self, p, *args, **kwargs):
        if np is None:
            return

        xyz = getattr(p, "_anima_detail_daemon_xyz", {}) or {}

        def _arg(i, default):
            return args[i] if len(args) > i else default

        def _xyz_num(key, cur):
            if key in xyz:
                try:
                    return float(xyz[key])
                except (TypeError, ValueError):
                    return cur
            return cur

        try:
            enabled = bool(_arg(0, False))
        except Exception:
            enabled = False
        if "enabled" in xyz:
            enabled = _as_bool(xyz["enabled"], enabled)

        if not enabled:
            _DD["on"] = False
            return

        try:
            preset = str(_arg(1, "Custom"))
            amount = float(_arg(2, 0.10))
            # A preset overrides the amount slider (Custom = use the slider).
            if preset in _PRESETS:
                amount = _PRESETS[preset]
            amount = _xyz_num("amount", amount)

            _DD.update(
                on=True,
                amount=amount,
                start=_xyz_num("start", float(_arg(3, 0.2))),
                end=_xyz_num("end", float(_arg(4, 0.8))),
                bias=_xyz_num("bias", float(_arg(5, 0.5))),
                exponent=float(_arg(6, 1.0)),
                start_offset=float(_arg(7, 0.0)),
                end_offset=float(_arg(8, 0.0)),
                fade=float(_arg(9, 0.0)),
                multiplier=float(_arg(10, 1.0)),
                smooth=_as_bool(_arg(11, True), True),
                cfg_couple=_as_bool(_arg(12, True), True),
                cfg_scale=float(getattr(p, "cfg_scale", 1.0) or 1.0),
                sched=None,       # force schedule rebuild for this generation
                sched_key=None,
            )
        except Exception as e:
            _DD["on"] = False
            _log(f"bad args, disabling: {type(e).__name__}: {e}")
            return

        if not hasattr(p, "extra_generation_params"):
            p.extra_generation_params = {}
        p.extra_generation_params["Anima Detail Daemon"] = (
            f"amount={_DD['amount']}, range={_DD['start']:.2f}-{_DD['end']:.2f}, "
            f"bias={_DD['bias']}, cfg_couple={_DD['cfg_couple']}"
        )
        _log(
            f"active ✅ amount={_DD['amount']} range={_DD['start']:.2f}-{_DD['end']:.2f} "
            f"bias={_DD['bias']} exp={_DD['exponent']} mult={_DD['multiplier']} "
            f"couple={_DD['cfg_couple']} cfg={_DD['cfg_scale']}"
        )

    def postprocess(self, p, processed, *args):
        _DD["on"] = False
