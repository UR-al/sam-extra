"""Anima Detail Daemon — Detail Daemon for Forge Neo, same numbers as the originals.

Detail Daemon on a Forge hook. The values and the per-call lookup follow the
ComfyUI node Jonseed/ComfyUI-Detail-Daemon (the user's reference) and the Forge
behaviour the node does not cover follows muerrilla/sd-webui-detail-daemon:

- Detail amount: range -5..5, step .01, default 0.10 (ComfyUI node inputs).
  Start .2 / End .8 / Bias .5 / Exponent 1 / offsets 0 / Fade 0 / Smooth on
  are the node's defaults too.
- Strength: ``sigma * max(1e-06, 1 - get_dd_schedule(sigma) * 0.1 * cfg_scale)``
  (node sampler; muerrilla has the same ``* 0.1 * cfg_scale`` without the
  floor). ``cfg_scale`` is always ``p.cfg_scale``, also on the HiRes pass
  (muerrilla ``self.cfg_scale = p.cfg_scale``).
- Schedule position (node): an N-entry schedule for the N steps the sampler
  runs, looked up by the sigma of each model call — nearest sigma, linear
  interpolation between neighbours, untouched outside the sampled range — so
  second-order samplers (Heun, DPM2, DPM++ 2S a, DPM++ SDE) read the curve at
  their midpoints like the node does.
- Hires Pass (muerrilla): off = base pass only, on = HiRes pass only.
- DPM adaptive / HeunPP2 are not supported and turn it off (muerrilla).
- No presets, no global multiplier, no CFG decoupling: the originals have none.
  Their positional slots (1 preset, 10 multiplier, 12 cfg_couple) stay so old
  API payloads still line up, but they are hidden and ignored.

Positive ``amount`` → sigma lowered → more detail; negative → smoother.
Zero (or disabled) → exact no-op.

Hook
----
Forge fires ``on_cfg_denoiser(params)`` for every model call with a
``CFGDenoiserParams`` carrying ``.sigma``, ``.sampling_step``,
``.total_sampling_steps`` and ``.denoiser``. We register one global callback that
reads the shared ``_DD`` state (set per sampling pass by the script's
``process_before_every_sampling``) and scales ``params.sigma`` in place, like
muerrilla (``params.sigma *= ...``), so the rest of ``CFGDenoiser.forward`` (the
NGMS check, soft inpainting's ``MaskBlendArgs``) sees the adjusted sigma too.
The sigma list comes from ``transformer_options['sampling_sigmas']``, which Forge
sets before every k-diffusion run. That is Forge's whole list: txt2img walks all
of it, img2img and the hires pass walk ``sigmas[steps - t_enc - 1:]``, so for the
pass the script set up the node's list is taken from that start (Forge's own
``setup_img2img_steps``), not counted back from the end — the ``DDIM`` schedule
type returns more than ``steps + 1`` sigmas. The callback stays on until
``postprocess`` (muerrilla's does too), so it also sees sampling runs made from
``postprocess_image`` without this script's hook (ADetailer's inner img2img,
img2img-hires-fix); those count their own ``denoiser.steps + 1`` sigmas back
from the end. The CompVis timestep samplers (DDIM, PLMS)
do not set it and have no ComfyUI counterpart; there the position falls back to
muerrilla's model-call counter. Everything is guarded; on any error it leaves
sigma untouched, so enabling this can never break a generation.

Model-agnostic: works on Anima (RF) and any other engine, since it only scales
sampler sigmas.

``_make_schedule`` and ``get_dd_schedule`` are copied from the MIT-licensed
originals (``_make_schedule`` adds a ``max(0, ...)`` guard on the linspace
lengths, nothing else):
  muerrilla/sd-webui-detail-daemon — Copyright (c) 2024 Sahand Ahmadian, MIT License
  Jonseed/ComfyUI-Detail-Daemon — Copyright (c) 2024 Jonseed, MIT License
The MIT text and the commits are in THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import sys
import traceback
from functools import partial

import gradio as gr

from modules import script_callbacks, scripts

from sam3ext import layout_lanes
from sam3ext.guidance.dave_gate import note_pre_dd_sigma

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


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
    "hires": False,        # Hires Pass: False = base pass only, True = HiRes pass only
    "cfg_scale": 1.0,      # p.cfg_scale, captured each sampling pass
    "sched": None,         # cached schedule array
    "sched_key": None,     # (steps, params…) the cache was built for
    "node": None,          # cached sigma lookup of this pass (see _node_lookup)
    "offset": 0,           # where this pass starts in sampling_sigmas (_forge_sampling_offset)
    "offset_p": None,      # the request ``offset`` was worked out for (see _run_offset)
}

# muerrilla's Detail Daemon scales the schedule by a fixed 0.1 before it touches
# sigma, and ComfyUI-Detail-Daemon does the same, so amounts transfer 1:1.
_SIGMA_SCALE = 0.1

# ComfyUI-Detail-Daemon never lets the sigma factor reach zero or go negative:
# ``sigma * max(1e-06, 1.0 - dd_adjustment * cfg_scale)``
# (Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:295). There is
# no upper bound.
_MIN_SIGMA_FACTOR = 1e-06

# muerrilla skips these samplers (scripts/detail_daemon.py:197-199); its position
# came from the model-call count, which these do not keep fixed per step.
_UNSUPPORTED_SAMPLERS = ("DPM adaptive", "HeunPP2")


# ---------------------------------------------------------------------------
# Schedule construction (same as muerrilla's make_schedule and the ComfyUI
# node's make_detail_daemon_schedule)
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


# origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:226-262, copied unchanged
# (MIT, Copyright (c) 2024 Jonseed).
def get_dd_schedule(
    sigma: float,
    sigmas: torch.Tensor,
    dd_schedule: torch.Tensor,
) -> float:
    sched_len = len(dd_schedule)
    if (
        sched_len < 2
        or len(sigmas) < 2
        or sigma <= 0
        or not (sigmas[-1] <= sigma <= sigmas[0])
    ):
        return 0.0
    # First, we find the index of the closest sigma in the list to what the model was
    # called with.
    deltas = (sigmas[:-1] - sigma).abs()
    idx = int(deltas.argmin())
    if (
        (idx == 0 and sigma >= sigmas[0])
        or (idx == sched_len - 1 and sigma <= sigmas[-2])
        or deltas[idx] == 0
    ):
        # Either exact match or closest to head/tail of the DD schedule so we
        # can't interpolate to another schedule item.
        return dd_schedule[idx].item()
    # If we're here, that means the sigma is in between two sigmas in the
    # list.
    idxlow, idxhigh = (idx, idx - 1) if sigma > sigmas[idx] else (idx + 1, idx)
    # We find the low/high neighbor sigmas - our sigma is somewhere between them.
    nlow, nhigh = sigmas[idxlow], sigmas[idxhigh]
    if nhigh - nlow == 0:
        # Shouldn't be possible, but just in case... Avoid divide by zero.
        return dd_schedule[idxlow]
    # Ratio of how close we are to the high neighbor.
    ratio = ((sigma - nlow) / (nhigh - nlow)).clamp(0, 1)
    # Mix the DD schedule high/low items according to the ratio.
    return torch.lerp(dd_schedule[idxlow], dd_schedule[idxhigh], ratio).item()


def _is_img2img_request(p) -> bool:
    """``p`` is a ``StableDiffusionProcessingImg2Img`` (or a subclass).

    Checked by class name so the helper needs no ``modules.processing`` import."""
    return any(
        cls.__name__ == "StableDiffusionProcessingImg2Img"
        for cls in type(p).__mro__
    )


def _forge_sampling_offset(p):
    """Index in Forge's ``sampling_sigmas`` of the first sigma this pass samples.

    txt2img's first pass walks the whole list (``KDiffusionSampler.sample``),
    so the offset is 0. img2img and the hires pass walk
    ``sigmas[steps - t_enc - 1:]`` (modules/sd_samplers_kdiffusion.py:145-148).
    Their ``steps, t_enc`` come from Forge's own ``setup_img2img_steps`` with
    the argument ``processing.py`` passes it: the hires pass uses
    ``hr_second_pass_steps or steps`` (:1552), img2img uses None (:1920).
    Returns None when the offset cannot be worked out; the run then counts back
    like a run the script did not set up (``_run_offset``). It is not counted
    back from the end with ``denoiser.steps`` otherwise, because
    some schedulers (Forge's ``ddim_scheduler``) return more than
    ``steps + 1`` sigmas. Same rule as DAVE's gate
    (``scripts/anima_safe_pag.py`` ``_forge_sampling_offset``)."""
    if getattr(p, "is_hr_pass", False):
        requested = getattr(p, "hr_second_pass_steps", 0) or getattr(p, "steps", None)
    elif _is_img2img_request(p):
        requested = None
    else:
        return 0
    try:
        from modules import sd_samplers_common

        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, requested)
        return int(steps) - int(t_enc) - 1
    except Exception:
        return None


def _executed_sigmas(denoiser):
    """Return Forge's ``sampling_sigmas`` for this k-diffusion run, or None.

    ComfyUI hands the node's sampler the sigmas it samples. Forge stores the
    whole list in ``transformer_options['sampling_sigmas']`` right before it
    samples (sd_samplers_kdiffusion.py ``sample`` / ``sample_img2img``). The
    CompVis timestep samplers (DDIM, PLMS; ``classic_ddim_eps_estimation``)
    never set it, so a value left there by an earlier run is not theirs.
    """
    if denoiser is None or getattr(denoiser, "classic_ddim_eps_estimation", False):
        return None
    try:
        unet = denoiser.p.sd_model.forge_objects.unet
        sigmas = unet.model_options["transformer_options"]["sampling_sigmas"]
    except Exception:
        return None
    if sigmas is None or len(sigmas) < 2:
        return None
    return sigmas


def _run_offset(denoiser, src) -> int:
    """Index in ``src`` of the first sigma this sampling run walks.

    The start ``process_before_every_sampling`` worked out belongs to the
    request it was given (``denoiser.p`` is that ``p`` for the pass's own
    sampling, processing.py:1380/:1546/:1914). The callback is global and stays
    on until ``postprocess``, like muerrilla's, so it also sees runs made from
    ``postprocess_image`` without that hook: ADetailer's inner img2img (a new
    ``p``) and img2img-hires-fix's ``sample_img2img(copy(p), ..., steps=...)``.
    Those count the run's own ``denoiser.steps + 1`` (``launch_sampling(t_enc + 1)``)
    sigmas back from the end, which is the list they walk whenever the scheduler
    returns ``steps + 1`` sigmas; muerrilla also reads the run's own step counts.
    """
    offset = _DD["offset"]
    pass_p = _DD["offset_p"]
    if pass_p is not None and getattr(denoiser, "p", None) is pass_p and isinstance(offset, int):
        return offset
    sampler_steps = int(getattr(denoiser, "steps", 0) or 0)
    if 0 < sampler_steps < len(src) - 1:
        return len(src) - (sampler_steps + 1)
    return 0


def _node_lookup(denoiser):
    """``(sigmas_cpu, dd_schedule, sigma_min, sigma_max)`` as the node's sampler builds them.

    None when Forge gives no sigma list for this run (timestep samplers).
    """
    if torch is None:
        return None
    src = _executed_sigmas(denoiser)
    if src is None:
        return None
    offset = _run_offset(denoiser, src)
    cache = _DD["node"]
    if cache is not None and cache["src"] is src and cache["offset"] == offset:
        return cache["lookup"]
    sigmas = src
    if 0 < offset < len(src) - 1:
        # img2img / hires walk ``sigmas[steps - t_enc - 1:]`` of Forge's list;
        # the node is handed exactly the list its sampler walks.
        sigmas = src[offset:]
    sched = _get_schedule(len(sigmas) - 1)
    # origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:282-288
    dd_schedule = torch.tensor(
        sched if sched is not None else [],
        dtype=torch.float32,
        device="cpu",
    )
    sigmas_cpu = sigmas.detach().clone().cpu()
    sigma_max, sigma_min = float(sigmas_cpu[0]), float(sigmas_cpu[-1]) + 1e-05
    lookup = (sigmas_cpu, dd_schedule, sigma_min, sigma_max)
    _DD["node"] = {"src": src, "offset": offset, "lookup": lookup}
    return lookup


def _schedule_position(params) -> tuple[int, int]:
    """Return (index, schedule length) by model call (muerrilla's formula).

    Only for the timestep samplers, which give no sigma list to look up.
    Forge sets ``state.sampling_step`` only after a step's model call, so it
    trails by one step; the denoiser's own call counter does not.
    """
    step = int(getattr(params, "sampling_step", 0) or 0)
    steps = int(getattr(params, "total_sampling_steps", 0) or 0)
    denoiser = getattr(params, "denoiser", None)
    if denoiser is not None:
        step = max(step, int(getattr(denoiser, "step", 0) or 0))
        steps = max(steps, int(getattr(denoiser, "total_steps", 0) or 0))
        sampler_steps = int(getattr(denoiser, "steps", 0) or 0)
        if sampler_steps > 0:
            steps -= max(steps // sampler_steps - 1, 0)
    return step, steps


def _call_count_value(params):
    step, steps = _schedule_position(params)
    if steps <= 0:
        return None
    sched = _get_schedule(steps)
    if sched is None or len(sched) == 0:
        return None
    return float(sched[min(max(step, 0), len(sched) - 1)])


def _sigma_factor(schedule_value, cfg_scale: float):
    """``1 - s * 0.1 * cfg`` with the ComfyUI node's 1e-06 floor (no ceiling)."""
    return max(_MIN_SIGMA_FACTOR, 1.0 - schedule_value * _SIGMA_SCALE * cfg_scale)


def _scale_sigma(params, factor) -> None:
    """Scale ``params.sigma`` in place, like muerrilla's ``params.sigma *= ...``.

    ``CFGDenoiser.forward`` keeps its own reference to this tensor and reads it
    after the callback (NGMS ``s_min_uncond`` check, ``MaskBlendArgs.sigma``),
    so writing into it is what makes those see the adjusted sigma. Forge's UniPC
    passes a view of its own timestep list instead of a fresh tensor; writing
    into that would move the solver's time steps (batch 1) or raise (batch > 1),
    so a view gets a new tensor, which is what the ComfyUI node always does.
    """
    sigma = params.sigma
    if torch is not None and isinstance(sigma, torch.Tensor) and not sigma._is_view():
        sigma.mul_(factor)
    else:
        params.sigma = sigma * factor


def _sigma_value(sigma) -> float:
    # origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:291
    if hasattr(sigma, "max"):
        return float(sigma.max().detach().cpu())
    return float(sigma)


def _denoiser_callback(params) -> None:
    # The DAVE gate (anima_safe_pag.py) reads this for the same forward: unscaled unless set below.
    note_pre_dd_sigma(None)
    if not _DD["on"] or np is None:
        return
    try:
        sigma = getattr(params, "sigma", None)
        if sigma is None:
            return
        lookup = _node_lookup(getattr(params, "denoiser", None))
        if lookup is not None:
            # origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:290-296
            sigmas_cpu, dd_schedule, sigma_min, sigma_max = lookup
            sigma_float = _sigma_value(sigma)
            if not (sigma_min <= sigma_float <= sigma_max):
                return
            value = get_dd_schedule(sigma_float, sigmas_cpu, dd_schedule)
        else:
            value = _call_count_value(params)
            if value is None:
                return
        factor = _sigma_factor(value, float(_DD["cfg_scale"]))
        if isinstance(factor, float) and factor == 1.0:
            return
        original = sigma.detach().clone() if hasattr(sigma, "detach") else sigma
        _scale_sigma(params, factor)
        note_pre_dd_sigma(original, params.sigma)
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
    @property
    def section(self):
        # Forge 는 사용자 섹션을 설정값 칼럼(#txt2img_settings) 안에 만든다 → 1열 "ANIMA 튜닝" 자리.
        # 설정(sam3_layout_sections)을 끄거나 img2img 면 None 이라 예전과 똑같이 스크립트 컨테이너로 간다.
        return layout_lanes.anima_section(bool(getattr(self, "is_img2img", False)))

    # Sits directly under the SAM3 mask accordion (-30) in the SAM3 extension
    # block (lower sorting_priority = higher up). The sigma callback is global
    # and order-independent, so processing early is safe.
    sorting_priority = -29

    def title(self):
        return "Anima Detail Daemon"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Anima Detail Daemon (디테일 조정)", open=False):
            gr.Markdown(
                "매 스텝 **제거하는 노이즈량을 줄여** 디테일·질감을 늘립니다(배경 뽀샤시↓). "
                "추가 forward 없이 sampler sigma만 조정하며 **모든 모델에서 동작**합니다. "
                "양수=디테일↑, 음수=매끈, 0/끄면 완전 무효. PAG·APG와 독립이라 같이 써도 됩니다. "
                "값·범위·기본값은 ComfyUI Detail Daemon 노드와 **같습니다**(강도 = Amount × 0.1 × CFG scale). "
                "**Hires Pass**를 끄면 기본 패스에만, 켜면 Hires 패스에만 적용합니다"
                "(원본 sd-webui-detail-daemon과 같음)."
            )
            with gr.Row():
                enabled = gr.Checkbox(
                    label="Enable Detail Daemon",
                    value=False,
                    elem_id="anima_dd_enable",
                    elem_classes=["sam3-on"],
                )
                hires = gr.Checkbox(label="Hires Pass", value=False, elem_id="anima_dd_hires")
            # ComfyUI-Detail-Daemon detail_amount: -5..5, step .01, default 0.1.
            amount = gr.Slider(
                label="Detail amount (ComfyUI detail_amount와 같은 값 · 음수=매끈 · 양수=디테일↑)",
                minimum=-5.0, maximum=5.0, step=0.01, value=0.10,
                elem_id="anima_dd_amount",
            )
            with gr.Accordion("Detail Daemon Advanced (세부값)", open=False):
                gr.Markdown(
                    "적용 구간과 곡선을 세밀 조정합니다. **start/end**=적용 스텝 구간, "
                    "**bias**=피크 위치, **exponent**=곡률, **offset**=구간 밖 기본값, "
                    "**fade**=전체 감쇠, **smooth**=코사인 스무딩."
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
                smooth = gr.Checkbox(label="Smooth (코사인 스무딩)", value=True, elem_id="anima_dd_smooth")
            # Old positional slots kept only so older API payloads still line up
            # (1 preset, 10 multiplier, 12 cfg_couple). The originals have none of
            # these knobs, so they are hidden and never read.
            preset = gr.Textbox(label="Preset (unused)", value="Custom", visible=False, elem_id="anima_dd_preset")
            multiplier = gr.Slider(
                label="Multiplier (unused)", minimum=0.0, maximum=2.0, step=0.05, value=1.0,
                visible=False, elem_id="anima_dd_multiplier",
            )
            cfg_couple = gr.Checkbox(
                label="Couple to CFG scale (unused)", value=True, visible=False, elem_id="anima_dd_cfg_couple",
            )
        # Hires Pass is appended last (arg 13) so the older 13-arg order is unchanged.
        return [
            enabled, preset, amount, start, end, bias, exponent,
            start_offset, end_offset, fade, multiplier, smooth, cfg_couple, hires,
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

        # muerrilla: "Selected sampler (...) is not supported." — off for the
        # whole generation, judged by the base sampler like the original.
        sampler_name = getattr(p, "sampler_name", None)
        if sampler_name in _UNSUPPORTED_SAMPLERS:
            _DD["on"] = False
            _log(f"selected sampler ({sampler_name}) is not supported, skipping")
            return

        try:
            # Args 1 (preset), 10 (multiplier) and 12 (cfg_couple) are old
            # slots the originals do not have; they are never read.
            hires = _as_bool(_arg(13, False), False)
            cfg_scale = getattr(p, "cfg_scale", None)
            _DD.update(
                # muerrilla: a daemon runs only on the pass its Hires Pass box
                # names (``daemon['hires'] != self.is_hires_pass → skip``).
                on=bool(getattr(p, "is_hr_pass", False)) == hires,
                amount=_xyz_num("amount", float(_arg(2, 0.10))),
                start=_xyz_num("start", float(_arg(3, 0.2))),
                end=_xyz_num("end", float(_arg(4, 0.8))),
                bias=_xyz_num("bias", float(_arg(5, 0.5))),
                exponent=float(_arg(6, 1.0)),
                start_offset=float(_arg(7, 0.0)),
                end_offset=float(_arg(8, 0.0)),
                fade=float(_arg(9, 0.0)),
                smooth=_as_bool(_arg(11, True), True),
                hires=hires,
                # muerrilla uses p.cfg_scale on every pass (not hr_cfg / refiner_cfg).
                cfg_scale=1.0 if cfg_scale is None else float(cfg_scale),
                sched=None,       # force schedule rebuild for this pass
                sched_key=None,
                node=None,
                # Where this pass's sampler starts in sampling_sigmas, and the
                # request it was worked out for (other runs count back, _run_offset).
                offset=_forge_sampling_offset(p),
                offset_p=p,
            )
        except Exception as e:
            _DD["on"] = False
            _log(f"bad args, disabling: {type(e).__name__}: {e}")
            return

        if not hasattr(p, "extra_generation_params"):
            p.extra_generation_params = {}
        p.extra_generation_params["Anima Detail Daemon"] = (
            f"amount={_DD['amount']}, range={_DD['start']:.2f}-{_DD['end']:.2f}, "
            f"bias={_DD['bias']}, exponent={_DD['exponent']}, "
            f"start_offset={_DD['start_offset']}, end_offset={_DD['end_offset']}, "
            f"fade={_DD['fade']}, smooth={_DD['smooth']}, hires={_DD['hires']}"
        )
        if _DD["on"]:
            _log(
                f"active ✅ amount={_DD['amount']} range={_DD['start']:.2f}-{_DD['end']:.2f} "
                f"bias={_DD['bias']} exp={_DD['exponent']} hires={_DD['hires']} cfg={_DD['cfg_scale']}"
            )

    def postprocess(self, p, processed, *args):
        _DD["on"] = False
        _DD["node"] = None  # drop the sampler's sigma tensor
        _DD["offset_p"] = None  # and the request
