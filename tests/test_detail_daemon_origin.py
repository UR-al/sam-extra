"""Detail Daemon — origin parity with the implementations it was ported from.

Values (amount range/default, curve defaults, ``x 0.1 x cfg``, the 1e-06 floor)
and the per-call schedule lookup (nearest sigma, interpolation, range check)
follow the ComfyUI node Jonseed/ComfyUI-Detail-Daemon, which the user chose as
the reference. What the node does not cover on Forge (Hires Pass, which CFG,
unsupported samplers, writing the sigma in place, the CompVis timestep samplers)
follows muerrilla/sd-webui-detail-daemon.

The origin functions below are copied from those repositories, not re-derived,
so a drift in ``scripts/anima_detail_daemon.py`` shows up as a diff against the
original code. Both origins are MIT licensed:

- ComfyUI-Detail-Daemon — Copyright (c) 2024 Jonseed, MIT License
- sd-webui-detail-daemon — Copyright (c) 2024 Sahand Ahmadian, MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of the
Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""
from __future__ import annotations

import ast
import copy
import functools
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests._forge_checkout import require_forge_file  # noqa: E402

# Building the UI below must not phone home for a gradio version check.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

NODE = "Jonseed/ComfyUI-Detail-Daemon@3394e44afea04ed0188fb37b21f0d9952469766b:detail_daemon_node.py"
MUERRILLA = "muerrilla/sd-webui-detail-daemon@19479998340831d7804fca8efd3f262b54b6373f:scripts/detail_daemon.py"


def _load_dd_module():
    """Load the Detail Daemon script without booting the full WebUI."""
    modules_stub = types.ModuleType("modules")

    class Script:
        pass

    modules_stub.scripts = types.SimpleNamespace(Script=Script, AlwaysVisible=object(), scripts_data=[])
    modules_stub.script_callbacks = types.SimpleNamespace(
        on_cfg_denoiser=lambda fn: None,
        on_before_ui=lambda fn: None,
    )
    old_modules = sys.modules.get("modules")
    sys.modules["modules"] = modules_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "_test_detail_daemon_origin", ROOT / "scripts" / "anima_detail_daemon.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if old_modules is None:
            sys.modules.pop("modules", None)
        else:
            sys.modules["modules"] = old_modules


dd = _load_dd_module()


# ---------------------------------------------------------------------------
# Origin code, copied as-is
# ---------------------------------------------------------------------------

# origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:25-67
# Schedule creation function from https://github.com/muerrilla/sd-webui-detail-daemon
def make_detail_daemon_schedule(
    steps,
    start,
    end,
    bias,
    amount,
    exponent,
    start_offset,
    end_offset,
    fade,
    smooth,
):
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = [
        int(round(x * (steps - 1))) for x in [start, mid, end]
    ]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    start_values = start_values**exponent
    if start_values.any():
        start_values *= amount - start_offset
        start_values += start_offset

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    end_values = end_values**exponent
    if end_values.any():
        end_values *= amount - end_offset
        end_values += end_offset

    multipliers[start_idx : mid_idx + 1] = start_values
    multipliers[mid_idx : end_idx + 1] = end_values
    multipliers[:start_idx] = start_offset
    multipliers[end_idx + 1 :] = end_offset
    multipliers *= 1 - fade

    return multipliers


# origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:226-262
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


# origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:265-310
def detail_daemon_sampler(
    model: object,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    dds_wrapped_sampler: object,
    dds_make_schedule: callable,
    dds_cfg_scale_override: float,
    **kwargs: dict,
) -> torch.Tensor:
    if dds_cfg_scale_override > 0:
        cfg_scale = dds_cfg_scale_override
    else:
        maybe_cfg_scale = getattr(model.inner_model, "cfg", None)
        cfg_scale = (
            float(maybe_cfg_scale) if isinstance(maybe_cfg_scale, (int, float)) else 1.0
        )
    dd_schedule = torch.tensor(
        dds_make_schedule(len(sigmas) - 1),
        dtype=torch.float32,
        device="cpu",
    )
    sigmas_cpu = sigmas.detach().clone().cpu()
    sigma_max, sigma_min = float(sigmas_cpu[0]), float(sigmas_cpu[-1]) + 1e-05

    def model_wrapper(x: torch.Tensor, sigma: torch.Tensor, **extra_args: dict):
        sigma_float = float(sigma.max().detach().cpu())
        if not (sigma_min <= sigma_float <= sigma_max):
            return model(x, sigma, **extra_args)
        dd_adjustment = get_dd_schedule(sigma_float, sigmas_cpu, dd_schedule) * 0.1
        adjusted_sigma = sigma * max(1e-06, 1.0 - dd_adjustment * cfg_scale)
        return model(x, adjusted_sigma, **extra_args)

    for k in (
        "inner_model",
        "sigmas",
    ):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))
    return dds_wrapped_sampler.sampler_function(
        model_wrapper,
        x,
        sigmas,
        **kwargs,
        **dds_wrapped_sampler.extra_options,
    )


# origin: muerrilla/sd-webui-detail-daemon@1947999:scripts/detail_daemon.py:309-338
def muerrilla_make_schedule(steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth):
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = [int(round(x * (steps - 1))) for x in [start, mid, end]]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    start_values = start_values ** exponent
    if start_values.any():
        start_values *= (amount - start_offset)
        start_values += start_offset

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    end_values = end_values ** exponent
    if end_values.any():
        end_values *= (amount - end_offset)
        end_values += end_offset

    multipliers[start_idx:mid_idx+1] = start_values
    multipliers[mid_idx:end_idx+1] = end_values
    multipliers[:start_idx] = start_offset
    multipliers[end_idx+1:] = end_offset
    multipliers *= 1 - fade

    return multipliers


def node_sigma_factor(schedule_value, cfg_scale):
    # origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:294-295
    #   dd_adjustment = get_dd_schedule(sigma_float, sigmas_cpu, dd_schedule) * 0.1
    #   adjusted_sigma = sigma * max(1e-06, 1.0 - dd_adjustment * cfg_scale)
    dd_adjustment = schedule_value * 0.1
    return max(1e-06, 1.0 - dd_adjustment * cfg_scale)


def muerrilla_position(sampling_step, total_sampling_steps, denoiser_step, denoiser_total_steps, denoiser_steps):
    # origin: muerrilla/sd-webui-detail-daemon@1947999:scripts/detail_daemon.py:281-284
    step = max(sampling_step, denoiser_step)
    steps = max(total_sampling_steps, denoiser_total_steps)
    actual_steps = steps - max(steps // denoiser_steps - 1, 0)
    idx = min(step, actual_steps - 1)
    return idx, actual_steps


# origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:324-356
# (DetailDaemonSamplerNode.INPUT_TYPES; the GUI and graph nodes use the same numbers, :76-108)
NODE_INPUTS = {
    "detail_amount": {"default": 0.1, "min": -5.0, "max": 5.0, "step": 0.01},
    "start": {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
    "end": {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
    "bias": {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "exponent": {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05},
    "start_offset": {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
    "end_offset": {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
    "fade": {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
    "smooth": {"default": True},
}

# origin: muerrilla/sd-webui-detail-daemon@1947999:scripts/detail_daemon.py:197
MUERRILLA_UNSUPPORTED_SAMPLERS = ["DPM adaptive", "HeunPP2"]


# ---------------------------------------------------------------------------
# Samplers: the model-call pattern of Forge's k-diffusion samplers
# (modules_forge/packages/k_diffusion/sampling.py sample_euler :176, sample_heun
# :257/:268, sample_dpm_2 :291/:305 with s_churn 0). They only record the sigma
# each model call is made with; ``model(x, <sigma> * s_in)`` is a fresh tensor.
# ---------------------------------------------------------------------------


def euler(model, x, sigmas, **kwargs):
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        model(x, sigmas[i] * s_in)
    return x


def heun(model, x, sigmas, **kwargs):
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        model(x, sigmas[i] * s_in)
        if sigmas[i + 1] != 0:
            model(x, sigmas[i + 1] * s_in)
    return x


def dpm_2(model, x, sigmas, **kwargs):
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        model(x, sigma_hat * s_in)
        if sigmas[i + 1] != 0:
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            model(x, sigma_mid * s_in)
    return x


# Forge's expected model calls per sampler step (Sampler.config.total_steps).
SECOND_ORDER = {"euler": (euler, 1), "heun": (heun, 2), "dpm_2": (dpm_2, 2)}


def flow_sigmas(steps, shift=3.0):
    """Anima flow sigmas: sigma = s*t / (1 + (s - 1)*t), t = 1 - p (Forge k_prediction.py:212-217), float32."""
    t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return (shift * t / (1 + (shift - 1) * t)).to(torch.float32)


# ---------------------------------------------------------------------------
# Forge host code that decides which sigmas a pass walks, read from the Forge
# checkout this extension lives in (not re-derived).
# ---------------------------------------------------------------------------


def _forge_function(relpath, name, namespace):
    """One top-level function of the Forge checkout, executed in ``namespace``.

    Without a Forge checkout (GitHub CI) the calling test is skipped."""
    path = require_forge_file(relpath)
    source = path.read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
            scope = dict(namespace)
            exec(code, scope)  # noqa: S102 - Forge's own function body
            return scope[name]
    raise AssertionError(f"{name} not found in {relpath}")


def _setup_img2img_steps(fix_steps=False):
    # origin: Haoming02/sd-webui-forge-classic@e33f40e4:modules/sd_samplers_common.py:43-52
    return _forge_function(
        "modules/sd_samplers_common.py", "setup_img2img_steps",
        {"opts": types.SimpleNamespace(img2img_fix_steps=fix_steps)},
    )


# origin: Haoming02/sd-webui-forge-classic@e33f40e4:modules/sd_schedulers.py:124-133
# (ddim_scheduler, the "DDIM" schedule type; ComfyUI's "ddim_uniform" is the same list)
@functools.cache
def _forge_ddim():
    return _forge_function("modules/sd_schedulers.py", "ddim_scheduler", {"torch": torch})


def ddim_sigmas(steps, shift=3.0):
    """Forge's DDIM schedule type on Anima's flow model sigmas (k_prediction.py:190-194)."""
    t = torch.arange(1, 1001, 1) / 1000
    inner = types.SimpleNamespace(sigmas=shift * t / (1 + (shift - 1) * t))
    return _forge_ddim()(steps, float(inner.sigmas[0]), float(inner.sigmas[-1]), inner, "cpu")


class StableDiffusionProcessingImg2Img(types.SimpleNamespace):
    """Stands in for modules.processing.StableDiffusionProcessingImg2Img (matched by name)."""


def _forge_modules(fix_steps=False):
    """``modules`` with Forge's real ``setup_img2img_steps`` behind ``sd_samplers_common``."""
    stub = types.ModuleType("modules")
    stub.sd_samplers_common = types.SimpleNamespace(setup_img2img_steps=_setup_img2img_steps(fix_steps))
    return stub


# ---------------------------------------------------------------------------
# Forge side: p, CFGDenoiser and CFGDenoiserParams as the callback sees them
# ---------------------------------------------------------------------------

# Extension positional args → node input name (index 1/10/12 are ignored old slots, 13 is Hires Pass).
ARG_TO_NODE = {
    2: "detail_amount", 3: "start", 4: "end", 5: "bias", 6: "exponent",
    7: "start_offset", 8: "end_offset", 9: "fade", 11: "smooth",
}


def _args(enabled=True, amount=0.1, start=0.2, end=0.8, bias=0.5, exponent=1.0,
          start_offset=0.0, end_offset=0.0, fade=0.0, smooth=True, hires=False,
          preset="Custom", multiplier=1.0, cfg_couple=True):
    return [enabled, preset, amount, start, end, bias, exponent,
            start_offset, end_offset, fade, multiplier, smooth, cfg_couple, hires]


def _curve(args):
    """The node's schedule inputs, in make_detail_daemon_schedule order after ``steps``."""
    return dict(start=args[3], end=args[4], bias=args[5], amount=args[2], exponent=args[6],
                start_offset=args[7], end_offset=args[8], fade=args[9], smooth=args[11])


def _p(cfg_scale=5.0, is_hr_pass=False, sampler_name="Euler", hr_cfg=2.0, refiner_cfg=None,
       sampling_sigmas=None, request_cls=types.SimpleNamespace, **extra):
    # sd_samplers_kdiffusion.py:192/:246 — transformer_options["sampling_sigmas"] = sigmas
    unet = types.SimpleNamespace(model_options={"transformer_options": {}})
    if sampling_sigmas is not None:
        unet.model_options["transformer_options"]["sampling_sigmas"] = sampling_sigmas
    return request_cls(
        cfg_scale=cfg_scale, is_hr_pass=is_hr_pass, sampler_name=sampler_name, hr_cfg=hr_cfg,
        refiner_cfg=refiner_cfg, extra_generation_params={},
        sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=unet)), **extra,
    )


class ForgeDenoiser:
    """The CFGDenoiser fields the callback reads (sd_samplers_cfg_denoiser.py:40-60, :133-134, :171)."""

    def __init__(self, p, *, steps, total_steps, refiner_pass=False, classic=False):
        self.p = p
        self.steps = steps                  # launch_sampling(steps): sampler steps
        self.total_steps = total_steps      # config.total_steps(steps): expected model calls
        self.step = 0                       # model calls done
        self._refiner_pass = refiner_pass
        self.classic_ddim_eps_estimation = classic
        self.seen = []                      # sigma the model got, per call
        self.forward_sigma = []             # forward's own ``sigma`` after the callback

    def __call__(self, x, sigma, **kwargs):
        # CFGDenoiser.forward: CFGDenoiserParams(x, image_cond, sigma, state.sampling_step, ...)
        params = types.SimpleNamespace(
            x=x, image_cond=None, sigma=sigma, sampling_step=max(self.step - 1, 0),
            total_sampling_steps=self.steps, text_cond=None, text_uncond=None, denoiser=self,
        )
        dd._denoiser_callback(params)
        self.seen.append(params.sigma.clone())
        self.forward_sigma.append(sigma.clone())
        self.step += 1
        return x


def forge_run(p, args, sampler, sigmas_run, *, batch=2, calls_per_step=1, refiner_pass=False, classic=False,
              sampler_steps=None, forge_modules=None):
    """One Forge sampling pass: process_before_every_sampling, then the sampler's model calls.

    ``sampler_steps`` is what Forge's ``launch_sampling`` puts on the denoiser
    (txt2img: the requested steps, img2img/hires: ``t_enc + 1``); by default the
    number of steps in ``sigmas_run``. ``forge_modules`` stands in for Forge's
    ``modules`` package while the pass is set up."""
    patched = {} if forge_modules is None else {"modules": forge_modules}
    with mock.patch.dict(sys.modules, patched):
        dd.AnimaDetailDaemon().process_before_every_sampling(p, *args)
    steps = len(sigmas_run) - 1 if sampler_steps is None else sampler_steps
    denoiser = ForgeDenoiser(p, steps=steps, total_steps=steps * calls_per_step,
                             refiner_pass=refiner_pass, classic=classic)
    sampler(denoiser, torch.zeros(batch, 1), sigmas_run)
    return denoiser.seen, denoiser


class NodeModel:
    """The model the node's wrapper calls; records the sigma it is given."""

    def __init__(self, cfg):
        self.inner_model = types.SimpleNamespace(cfg=cfg)
        self.seen = []

    def __call__(self, x, sigma, **extra_args):
        self.seen.append(sigma.clone())
        return x


def node_run(args, sampler, sigmas_run, *, cfg, batch=2):
    """The node's DetailDaemonSamplerNode.go → detail_daemon_sampler, cfg as Forge uses it (p.cfg_scale)."""
    curve = _curve(args)

    def dds_make_schedule(steps):  # origin: detail_daemon_node.py:384-396
        return make_detail_daemon_schedule(
            steps, curve["start"], curve["end"], curve["bias"], curve["amount"], curve["exponent"],
            curve["start_offset"], curve["end_offset"], curve["fade"], curve["smooth"],
        )

    model = NodeModel(cfg)
    detail_daemon_sampler(
        model, torch.zeros(batch, 1), sigmas_run,
        dds_wrapped_sampler=types.SimpleNamespace(sampler_function=sampler, extra_options={}),
        dds_make_schedule=dds_make_schedule,
        dds_cfg_scale_override=cfg,
    )
    return model.seen


def _one_call(p, args, sigmas_run, call_sigma, **kw):
    """Run process_before_every_sampling and one model call at ``call_sigma``; return (given, adjusted)."""
    def sampler(model, x, sigmas, **kwargs):
        model(x, call_sigma * x.new_ones([x.shape[0]]))
        return x

    seen, _ = forge_run(p, args, sampler, sigmas_run, **kw)
    given = call_sigma * torch.ones(2)
    return given, seen[0]


# Schedule parameter sets: (steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth)
EXPLICIT_CASES = [
    (20, 0.2, 0.8, 0.5, 0.1, 1.0, 0.0, 0.0, 0.0, True),      # node defaults
    (28, 0.2, 0.8, 0.5, 0.1, 1.0, 0.0, 0.0, 0.0, True),
    (30, 0.2, 0.8, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, True),
    (1, 0.2, 0.8, 0.5, 0.1, 1.0, 0.0, 0.0, 0.0, True),        # one step
    (2, 0.2, 0.8, 0.5, 0.25, 1.0, 0.0, 0.0, 0.0, False),
    (3, 0.0, 1.0, 0.5, -0.5, 2.0, 0.1, -0.1, 0.0, True),
    (50, 0.0, 1.0, 0.0, 5.0, 1.0, 0.0, 0.0, 0.0, True),       # bias 0, max amount
    (50, 0.0, 1.0, 1.0, -5.0, 1.0, 0.0, 0.0, 0.0, True),      # bias 1, min amount
    (33, 0.9, 0.1, 0.3, 0.2, 1.0, 0.0, 0.0, 0.0, True),       # start > end (node swaps via min)
    (24, 0.5, 0.5, 0.5, 0.3, 1.0, 0.2, -0.2, 0.0, True),      # zero-width window
    (40, 0.1, 0.9, 0.7, 0.4, 0.0, 0.0, 0.0, 0.0, True),       # exponent 0 → flat 1s
    (40, 0.1, 0.9, 0.7, 0.4, 10.0, 0.0, 0.0, 0.0, False),     # exponent max
    (61, 0.25, 0.75, 0.4, 0.15, 0.5, -1.0, 1.0, 0.0, True),   # offsets at the range ends
    (25, 0.2, 0.8, 0.5, 0.1, 1.0, 0.0, 0.0, 1.0, True),       # fade 1 → all zero
    (25, 0.2, 0.8, 0.5, 0.1, 1.0, 0.05, 0.05, 0.35, False),
    (10, 0.0, 0.0, 0.5, 0.3, 1.0, 0.0, 0.1, 0.0, True),       # window at the first step
    (10, 1.0, 1.0, 0.5, 0.3, 1.0, 0.1, 0.0, 0.0, True),       # window at the last step
    (57, 0.33, 0.66, 0.5, 0.0, 1.0, 0.1, -0.1, 0.0, True),    # amount 0 but offsets
    (57, 0.33, 0.66, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, True),     # all zero → start_values.any() False
    (19, 0.15, 0.85, 0.25, 2.5, 1.5, 0.0, 0.0, 0.1, True),
    (39, 0.0, 1.0, 0.5, 0.1, 1.0, 0.0, 0.0, 0.0, True),
    (100, 0.05, 0.95, 0.6, -0.75, 3.0, -0.3, 0.4, 0.5, False),
]


def _case_args(case):
    _, st, ed, bias, amount, exp, so, eo, fade, smooth = case
    return _args(amount=amount, start=st, end=ed, bias=bias, exponent=exp,
                 start_offset=so, end_offset=eo, fade=fade, smooth=smooth)


def _random_cases(n=200, seed=20260925):
    """Random sets inside the node's input ranges, snapped to the node's step sizes."""
    rng = np.random.RandomState(seed)

    def snap(key):
        spec = NODE_INPUTS[key]
        k = rng.randint(0, int(round((spec["max"] - spec["min"]) / spec["step"])) + 1)
        return round(spec["min"] + k * spec["step"], 4)

    cases = []
    for _ in range(n):
        cases.append((
            int(rng.randint(1, 151)), snap("start"), snap("end"), snap("bias"), snap("detail_amount"),
            snap("exponent"), snap("start_offset"), snap("end_offset"), snap("fade"), bool(rng.randint(0, 2)),
        ))
    return cases


ALL_CASES = EXPLICIT_CASES + _random_cases()


def _assert_same_calls(test, ours, node):
    test.assertEqual(len(ours), len(node))
    for k, (a, b) in enumerate(zip(ours, node)):
        test.assertEqual(a.dtype, b.dtype, f"call {k}")
        test.assertTrue(torch.equal(a, b), f"call {k}: ours {a.tolist()} node {b.tolist()}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class ScheduleOriginTests(unittest.TestCase):
    def test_schedule_is_bit_identical_to_the_node(self):
        self.assertGreaterEqual(len(ALL_CASES), 20)
        worst = 0.0
        for case in ALL_CASES:
            with self.subTest(case=case):
                ours = dd._make_schedule(*case)
                node = make_detail_daemon_schedule(*case)
                self.assertEqual(ours.shape, node.shape)
                diff = float(np.max(np.abs(ours - node))) if len(node) else 0.0
                worst = max(worst, diff)
                self.assertTrue(np.array_equal(ours, node), f"max diff {diff}")
        self.assertEqual(worst, 0.0)

    def test_node_schedule_is_muerrillas(self):
        # The node's schedule is muerrilla's, so matching one matches both.
        for case in EXPLICIT_CASES:
            with self.subTest(case=case):
                self.assertTrue(np.array_equal(make_detail_daemon_schedule(*case), muerrilla_make_schedule(*case)))

    def test_lookup_is_the_nodes(self):
        # Same function body as the node's get_dd_schedule, including its quirks
        # (tensor return when two neighbours are equal).
        sigmas = flow_sigmas(12)
        sched = torch.tensor(make_detail_daemon_schedule(12, 0.1, 0.9, 0.4, 0.7, 1.3, 0.1, -0.2, 0.0, True),
                             dtype=torch.float32)
        probes = [float(s) for s in sigmas] + [float(x) for x in torch.linspace(0.0, 1.2, 97)]
        for s in probes:
            with self.subTest(sigma=s):
                self.assertEqual(dd.get_dd_schedule(s, sigmas, sched), get_dd_schedule(s, sigmas, sched))


class StrengthOriginTests(unittest.TestCase):
    def test_first_order_sampler_matches_the_node_sampler(self):
        # Euler calls the model once per step at sigmas[i]: the node's lookup hits schedule[i] exactly.
        for case in EXPLICIT_CASES:
            args = _case_args(case)
            sigmas = flow_sigmas(case[0])
            for cfg in (1.0, 4.5, 7.0):
                with self.subTest(case=case, cfg=cfg):
                    ours, _ = forge_run(_p(cfg_scale=cfg, sampling_sigmas=sigmas), args, euler, sigmas)
                    _assert_same_calls(self, ours, node_run(args, euler, sigmas, cfg=cfg))

    def test_second_order_samplers_match_the_node_sampler(self):
        # Heun / DPM2 call the model 2N-1 times; the node looks each call up by its sigma
        # (N-entry schedule, interpolated between neighbours) instead of counting calls.
        cases = [c for c in EXPLICIT_CASES if c[0] >= 2] + ALL_CASES[len(EXPLICIT_CASES):len(EXPLICIT_CASES) + 40]
        for name in ("heun", "dpm_2"):
            sampler, calls = SECOND_ORDER[name]
            for case in cases:
                args = _case_args(case)
                sigmas = flow_sigmas(case[0])
                for cfg in (1.0, 5.0):
                    with self.subTest(sampler=name, case=case, cfg=cfg):
                        ours, _ = forge_run(_p(cfg_scale=cfg, sampling_sigmas=sigmas), args, sampler, sigmas,
                                            calls_per_step=calls)
                        _assert_same_calls(self, ours, node_run(args, sampler, sigmas, cfg=cfg))

    def test_heun_28_steps_follows_the_node_not_the_call_count(self):
        # Review case: Heun, 28 steps, amount 1.0, cfg 5, Anima flow sigmas. Call 17 is the
        # corrector of step 8 at sigmas[9]: node schedule28[9] → 0.79341; muerrilla's
        # 55-entry call curve gave schedule55[17] → 0.84567.
        args = _args(amount=1.0)
        sigmas = flow_sigmas(28)
        ours, _ = forge_run(_p(cfg_scale=5.0, sampling_sigmas=sigmas), args, heun, sigmas, calls_per_step=2)
        self.assertEqual(len(ours), 55)
        factor = float(ours[17][0]) / float(sigmas[9])
        self.assertAlmostEqual(factor, 0.79341, places=5)
        idx, actual = muerrilla_position(8, 28, 17, 56, 28)
        self.assertEqual((idx, actual), (17, 55))
        call_count = 1 - muerrilla_make_schedule(actual, 0.2, 0.8, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, True)[idx] * .1 * 5.0
        self.assertAlmostEqual(call_count, 0.84567, places=5)

    def test_img2img_runs_the_tail_like_the_node(self):
        # sample_img2img: steps, t_enc = setup_img2img_steps(p, steps); sigma_sched = sigmas[steps - t_enc - 1:];
        # launch_sampling(t_enc + 1), while transformer_options["sampling_sigmas"] keeps the full list
        # (sd_samplers_kdiffusion.py:145-148, :192-195). The node is handed the tail.
        full = flow_sigmas(30)
        for denoise in (0.02, 0.05, 0.5, 0.95):
            p = _p(cfg_scale=4.0, sampling_sigmas=full, request_cls=StableDiffusionProcessingImg2Img,
                   steps=30, denoising_strength=denoise)
            total, t_enc = _setup_img2img_steps()(p, None)
            self.assertEqual(total, 30)
            tail = full[total - t_enc - 1:]
            for name, (sampler, calls) in SECOND_ORDER.items():
                with self.subTest(t_enc=t_enc, sampler=name):
                    ours, _ = forge_run(p, _args(amount=0.8), sampler, tail, calls_per_step=calls,
                                        sampler_steps=t_enc + 1, forge_modules=_forge_modules())
                    _assert_same_calls(self, ours, node_run(_args(amount=0.8), sampler, tail, cfg=4.0))

    def test_sigma_outside_the_sampled_range_is_untouched(self):
        # detail_daemon_node.py:288, :292-293 — sigma_min = sigmas[-1] + 1e-05, sigma_max = sigmas[0]
        sigmas = flow_sigmas(20)
        for s in (1.25, float(sigmas[0]) * 1.0001, 5e-06):
            with self.subTest(sigma=s):
                given, adjusted = _one_call(_p(sampling_sigmas=sigmas), _args(start=0.0, start_offset=0.5,
                                                                              end_offset=0.5, amount=1.0),
                                            sigmas, torch.tensor(s, dtype=torch.float32))
                self.assertTrue(torch.equal(adjusted, given))

    def test_one_step_run_is_a_no_op_like_the_node(self):
        # get_dd_schedule returns 0.0 when the schedule has fewer than 2 entries.
        sigmas = flow_sigmas(1)
        ours, _ = forge_run(_p(sampling_sigmas=sigmas), _args(amount=1.0, start=0.0, end=0.0), euler, sigmas)
        node = node_run(_args(amount=1.0, start=0.0, end=0.0), euler, sigmas, cfg=5.0)
        _assert_same_calls(self, ours, node)
        self.assertTrue(torch.equal(ours[0], sigmas[0] * torch.ones(2)))

    def test_factor_floor_is_the_nodes_1e_06_and_there_is_no_ceiling(self):
        sigmas = flow_sigmas(20)  # 20 steps, defaults: schedule[10] = amount (peak)
        peak = sigmas[10]
        # amount 5, cfg 7: 1 - 5 * 0.1 * 7 = -2.5 → node floor 1e-06 (old clamp gave 0.05)
        given, adjusted = _one_call(_p(cfg_scale=7.0, sampling_sigmas=sigmas), _args(amount=5.0), sigmas, peak)
        self.assertEqual(node_sigma_factor(5.0, 7.0), 1e-06)
        self.assertTrue(torch.equal(adjusted, given * 1e-06))
        # amount -5, cfg 7: 1 + 3.5 = 4.5 → no ceiling (old clamp gave 3.0)
        given, adjusted = _one_call(_p(cfg_scale=7.0, sampling_sigmas=sigmas), _args(amount=-5.0), sigmas, peak)
        self.assertTrue(torch.equal(adjusted, given * node_sigma_factor(-5.0, 7.0)))
        self.assertAlmostEqual(float(adjusted[0]) / float(peak), 4.5, places=6)

    def test_cfg_scale_zero_is_a_no_op_like_the_original(self):
        # muerrilla multiplies by p.cfg_scale as is: cfg 0 → factor 1.
        sigmas = flow_sigmas(20)
        given, adjusted = _one_call(_p(cfg_scale=0.0, sampling_sigmas=sigmas), _args(amount=1.0), sigmas, sigmas[10])
        self.assertTrue(torch.equal(adjusted, given))

    def test_timestep_samplers_keep_muerrillas_call_count_position(self):
        # DDIM / PLMS (CFGDenoiserTimesteps, classic_ddim_eps_estimation) never set sampling_sigmas and have
        # no ComfyUI counterpart, so the position is muerrilla's: model call index on a call-count curve.
        # A sampling_sigmas left over from an earlier k-diffusion run must not be used.
        n = 12
        start, end, bias, amount, exponent, so, eo, fade, smooth = (0.2, 0.8, 0.5, 0.3, 1.0, 0.0, 0.0, 0.0, True)
        args = _args(amount=amount, start=start, end=end, bias=bias, exponent=exponent,
                     start_offset=so, end_offset=eo, fade=fade, smooth=smooth)
        stale = flow_sigmas(40)
        cfg = 5.0

        def calls(model, x, sigmas, **kwargs):  # PLMS-style: 2n-1 model calls, sigma fixed at 1.0
            for _ in range(2 * n - 1):
                model(x, torch.ones(x.shape[0]))
            return x

        ours, _ = forge_run(_p(cfg_scale=cfg, sampling_sigmas=stale), args, calls, flow_sigmas(n),
                            calls_per_step=2, classic=True)
        for call_idx in range(2 * n - 1):
            with self.subTest(call=call_idx):
                idx, actual = muerrilla_position(max(call_idx - 1, 0), n, call_idx, 2 * n, n)
                sched = muerrilla_make_schedule(actual, start, end, bias, amount, exponent, so, eo, fade, smooth)
                # origin: muerrilla/sd-webui-detail-daemon@1947999:scripts/detail_daemon.py:244, :290, :304
                #   params.sigma *= 1 - schedule[idx] * .1 * cfg_scale
                expected = torch.ones(2)
                expected *= 1 - sched[idx] * .1 * cfg
                self.assertTrue(torch.equal(ours[call_idx], expected), (ours[call_idx], expected))


class ForgeWalkedListTests(unittest.TestCase):
    """The node is handed the list Forge's sampler walks, also when a scheduler returns extra sigmas.

    Forge's DDIM schedule type returns ``steps + 2`` sigmas at 24/28/30/32 steps, so
    ``launch_sampling``'s step count (``denoiser.steps``) is not ``len(sigmas) - 1``.
    txt2img walks the whole list; img2img and hires walk it from ``steps - t_enc - 1``
    (Forge's own ``setup_img2img_steps``), never counted back from the end."""

    ARGS = _args(amount=0.8)
    CFG = 4.0

    def test_forge_ddim_schedule_type_returns_extra_sigmas(self):
        for steps in (24, 28, 30, 32):
            with self.subTest(steps=steps):
                self.assertEqual(len(ddim_sigmas(steps)), steps + 2)

    def test_txt2img_ddim_schedule_type_walks_the_whole_list(self):
        # KDiffusionSampler.sample: sigmas = get_sigmas(p, steps), sampling_sigmas = sigmas,
        # launch_sampling(steps, ...) — the sampler walks every sigma (sd_samplers_kdiffusion.py:204-252).
        for steps in (24, 28, 30, 32):
            full = ddim_sigmas(steps)
            for name, (sampler, calls) in SECOND_ORDER.items():
                with self.subTest(steps=steps, sampler=name):
                    p = _p(cfg_scale=self.CFG, sampling_sigmas=full, steps=steps)
                    ours, _ = forge_run(p, self.ARGS, sampler, full, calls_per_step=calls, sampler_steps=steps)
                    _assert_same_calls(self, ours, node_run(self.ARGS, sampler, full, cfg=self.CFG))

    def test_txt2img_ddim_28_schedule_value_at_step_10(self):
        # Review case: DDIM schedule type, 28 steps → 30 sigmas, a 29-entry node schedule.
        # Step 10 is schedule29[10] = 0.05 * 8 (amount 0.8); counting 28 + 1 sigmas back
        # from the end dropped the first sigma and read a 28-entry curve instead.
        full = ddim_sigmas(28)
        p = _p(cfg_scale=self.CFG, sampling_sigmas=full, steps=28)
        given, adjusted = _one_call(p, self.ARGS, full, full[10], sampler_steps=28)
        sched29 = make_detail_daemon_schedule(29, 0.2, 0.8, 0.5, 0.8, 1.0, 0.0, 0.0, 0.0, True)
        self.assertAlmostEqual(float(sched29[10]), 0.4, places=9)
        expected = float(torch.tensor(sched29, dtype=torch.float32)[10])  # the node's float32 dd_schedule
        self.assertTrue(torch.equal(adjusted, given * node_sigma_factor(expected, self.CFG)),
                        (adjusted, given * node_sigma_factor(expected, self.CFG)))

    def test_img2img_ddim_schedule_type_starts_where_forge_starts(self):
        extra = 0
        for steps, denoise in ((28, 0.5), (30, 0.6), (24, 0.75), (32, 0.35)):
            for fix_steps in (False, True):
                p = _p(cfg_scale=self.CFG, request_cls=StableDiffusionProcessingImg2Img,
                       steps=steps, denoising_strength=denoise)
                total, t_enc = _setup_img2img_steps(fix_steps)(p, None)  # processing.py:1920 passes no steps
                full = ddim_sigmas(total)
                extra += len(full) > total + 1
                p.sd_model.forge_objects.unet.model_options["transformer_options"]["sampling_sigmas"] = full
                walked = full[total - t_enc - 1:]
                for name, (sampler, calls) in SECOND_ORDER.items():
                    with self.subTest(steps=steps, denoise=denoise, fix_steps=fix_steps, sampler=name):
                        ours, _ = forge_run(p, self.ARGS, sampler, walked, calls_per_step=calls,
                                            sampler_steps=t_enc + 1, forge_modules=_forge_modules(fix_steps))
                        _assert_same_calls(self, ours, node_run(self.ARGS, sampler, walked, cfg=self.CFG))
        self.assertGreater(extra, 0)  # the cases include lists longer than steps + 1

    def test_hires_pass_starts_where_forge_starts(self):
        # processing.py:1552 — sample_img2img(..., steps=self.hr_second_pass_steps or self.steps)
        args = _args(amount=0.8, hires=True)
        for steps, hr_steps, denoise in ((28, 0, 0.5), (28, 10, 0.5), (30, 0, 0.4), (20, 28, 0.7)):
            p = _p(cfg_scale=self.CFG, is_hr_pass=True, steps=steps, hr_second_pass_steps=hr_steps,
                   denoising_strength=denoise)
            total, t_enc = _setup_img2img_steps()(p, hr_steps or steps)
            full = ddim_sigmas(total)
            p.sd_model.forge_objects.unet.model_options["transformer_options"]["sampling_sigmas"] = full
            walked = full[total - t_enc - 1:]
            for name, (sampler, calls) in SECOND_ORDER.items():
                with self.subTest(steps=steps, hr_steps=hr_steps, denoise=denoise, sampler=name):
                    ours, _ = forge_run(p, args, sampler, walked, calls_per_step=calls,
                                        sampler_steps=t_enc + 1, forge_modules=_forge_modules())
                    _assert_same_calls(self, ours, node_run(args, sampler, walked, cfg=self.CFG))

    def test_offset_is_forges_start_index(self):
        cases = [
            (_p(steps=28, denoising_strength=0.5), None, 0),  # txt2img first pass
            (_p(request_cls=StableDiffusionProcessingImg2Img, steps=28, denoising_strength=0.5), None, None),
            (_p(is_hr_pass=True, steps=28, hr_second_pass_steps=0, denoising_strength=0.5), 28, None),
            (_p(is_hr_pass=True, steps=28, hr_second_pass_steps=10, denoising_strength=0.5), 10, None),
        ]
        for p, requested, expected in cases:
            if expected is None:
                total, t_enc = _setup_img2img_steps()(p, requested)
                expected = total - t_enc - 1
            with self.subTest(img2img=type(p).__name__, is_hr_pass=p.is_hr_pass):
                with mock.patch.dict(sys.modules, {"modules": _forge_modules()}):
                    self.assertEqual(dd._forge_sampling_offset(p), expected)

    def test_without_forge_the_offset_is_unknown(self):
        p = _p(request_cls=StableDiffusionProcessingImg2Img, steps=28, denoising_strength=0.5)
        with mock.patch.dict(sys.modules, {"modules": types.ModuleType("modules")}):
            self.assertIsNone(dd._forge_sampling_offset(p))


class NestedRunTests(unittest.TestCase):
    """Sampling runs this script's ``process_before_every_sampling`` did not set up.

    The cfg-denoiser callback is global and stays on until ``postprocess``, like
    muerrilla's (registered in ``process``, removed in ``postprocess``,
    detail_daemon.py:256-272), so it also sees runs made from
    ``postprocess_image`` (processing.py:1068, before ``postprocess`` at :1180):
    ADetailer's inner img2img with only its selected scripts, and
    img2img-hires-fix's ``sampler.sample_img2img(copy(p), ..., steps=self.steps)``
    (img2img_hires_fix.py:95, :194, :201). The start index worked out for the
    pass belongs to that pass's ``p``; these runs get the list they walk."""

    CFG = 4.0

    def _outer_pass(self, hires):
        """The generation's own pass (txt2img 28 steps, or its hires pass) with DD on for it."""
        args = _args(amount=0.8, hires=hires)
        if hires:
            p = _p(cfg_scale=self.CFG, is_hr_pass=True, steps=28, hr_second_pass_steps=0,
                   denoising_strength=0.5)
            total, t_enc = _setup_img2img_steps()(p, 28)
            full = flow_sigmas(total)
            p.sd_model.forge_objects.unet.model_options["transformer_options"]["sampling_sigmas"] = full
            forge_run(p, args, euler, full[total - t_enc - 1:], sampler_steps=t_enc + 1,
                      forge_modules=_forge_modules())
        else:
            full = flow_sigmas(28)
            p = _p(cfg_scale=self.CFG, sampling_sigmas=full, steps=28)
            forge_run(p, args, euler, full, sampler_steps=28, forge_modules=_forge_modules())
        self.assertTrue(dd._DD["on"])
        return p, args

    def _nested_run(self, p, total, t_enc, sampler, calls):
        """Forge's sample_img2img on ``p`` with no process_before_every_sampling of this script."""
        full = flow_sigmas(total)
        p.sd_model.forge_objects.unet.model_options["transformer_options"]["sampling_sigmas"] = full
        walked = full[total - t_enc - 1:]
        denoiser = ForgeDenoiser(p, steps=t_enc + 1, total_steps=(t_enc + 1) * calls)
        sampler(denoiser, torch.zeros(2, 1), walked)
        return denoiser.seen, walked

    def test_adetailer_inner_img2img_walks_its_own_tail(self):
        # ADetailer: a new StableDiffusionProcessingImg2Img through process_images →
        # sample_img2img(self, ...) with no steps (processing.py:1920).
        for hires in (False, True):
            for steps, denoise in ((28, 0.4), (20, 0.4), (28, 0.3)):
                for name, (sampler, calls) in SECOND_ORDER.items():
                    with self.subTest(outer_hires=hires, steps=steps, denoise=denoise, sampler=name):
                        _, args = self._outer_pass(hires)
                        inner = _p(cfg_scale=self.CFG, request_cls=StableDiffusionProcessingImg2Img,
                                   steps=steps, denoising_strength=denoise)
                        total, t_enc = _setup_img2img_steps()(inner, None)
                        ours, walked = self._nested_run(inner, total, t_enc, sampler, calls)
                        _assert_same_calls(self, ours, node_run(args, sampler, walked, cfg=self.CFG))

    def test_img2img_hires_fix_on_a_copy_walks_its_own_tail(self):
        # img2img-hires-fix: self.p = copy(p); self.p.denoising_strength = ...;
        # sampler.sample_img2img(self.p, ..., steps=self.steps) — steps given, so Forge
        # uses int(steps / denoise) sigmas and walks the last steps + 1 of them.
        for steps, denoise in ((28, 0.4), (12, 0.35), (20, 0.6)):
            for name, (sampler, calls) in SECOND_ORDER.items():
                with self.subTest(steps=steps, denoise=denoise, sampler=name):
                    outer, args = self._outer_pass(False)
                    fix = copy.copy(outer)
                    fix.denoising_strength = denoise
                    total, t_enc = _setup_img2img_steps()(fix, steps)
                    ours, walked = self._nested_run(fix, total, t_enc, sampler, calls)
                    _assert_same_calls(self, ours, node_run(args, sampler, walked, cfg=self.CFG))

    def test_postprocess_drops_the_pass_request(self):
        outer, _ = self._outer_pass(False)
        dd.AnimaDetailDaemon().postprocess(outer, None)
        self.assertIsNone(dd._DD["offset_p"])


class InPlaceTests(unittest.TestCase):
    # origin: muerrilla/sd-webui-detail-daemon@1947999:scripts/detail_daemon.py:304 — ``params.sigma *= ...`` writes
    # into the tensor CFGDenoiser.forward holds as ``sigma``; forward reads it again after the callback
    # (sd_samplers_cfg_denoiser.py:144 NGMS ``0 < sigma[0] < s_min_uncond``, :159 MaskBlendArgs(sigma=sigma)).
    def test_forward_sees_the_adjusted_sigma(self):
        sigmas = flow_sigmas(20)
        seen, denoiser = forge_run(_p(cfg_scale=5.0, sampling_sigmas=sigmas), _args(amount=1.0), euler, sigmas)
        changed = 0
        for k, (model_sigma, forward_sigma) in enumerate(zip(seen, denoiser.forward_sigma)):
            with self.subTest(call=k):
                self.assertTrue(torch.equal(model_sigma, forward_sigma))
                changed += not torch.equal(forward_sigma, sigmas[k] * torch.ones(2))
        self.assertGreater(changed, 0)

    def test_params_sigma_is_the_same_tensor_object(self):
        sigmas = flow_sigmas(20)
        dd.AnimaDetailDaemon().process_before_every_sampling(_p(sampling_sigmas=sigmas), *_args(amount=1.0))
        held = sigmas[10] * torch.ones(2)
        before = held.clone()
        params = types.SimpleNamespace(sigma=held, sampling_step=9, total_sampling_steps=20,
                                       denoiser=ForgeDenoiser(_p(sampling_sigmas=sigmas), steps=20, total_steps=20))
        dd._denoiser_callback(params)
        self.assertIs(params.sigma, held)
        self.assertTrue(torch.equal(held, before * node_sigma_factor(1.0, 5.0)))

    def test_notes_the_unscaled_sigma_for_the_dave_gate(self):
        # scripts/anima_safe_pag.py _dave_gate_open looks this up instead of the scaled sigma.
        from sam3ext.guidance.dave_gate import note_pre_dd_sigma, pre_dd_sigma
        self.addCleanup(note_pre_dd_sigma, None)
        sigmas = flow_sigmas(20)
        for amount, scaled in ((1.0, True), (0.0, False)):
            with self.subTest(amount=amount):
                p = _p(sampling_sigmas=sigmas)
                dd.AnimaDetailDaemon().process_before_every_sampling(p, *_args(amount=amount))
                note_pre_dd_sigma("stale")
                held = sigmas[10] * torch.ones(2)
                before = held.clone()
                params = types.SimpleNamespace(sigma=held, sampling_step=9, total_sampling_steps=20,
                                               denoiser=ForgeDenoiser(p, steps=20, total_steps=20))
                dd._denoiser_callback(params)
                if scaled:
                    self.assertTrue(torch.equal(pre_dd_sigma(params.sigma), before))
                    self.assertFalse(torch.equal(params.sigma, before))
                    self.assertIsNone(pre_dd_sigma(before))   # only the forward it scaled
                else:   # an unscaled forward clears what an earlier forward noted
                    self.assertIsNone(pre_dd_sigma())

    def test_a_view_of_the_samplers_timesteps_is_not_written(self):
        # Forge UniPC (sd_samplers_extra.sample_unipc → uni_pc.py:516-548) passes
        # ``timesteps[k].expand(batch)`` to the model. Writing into it would move the solver's time steps
        # (batch 1) or raise (batch > 1); it gets a new tensor like the node's ``sigma * factor``.
        sigmas = flow_sigmas(20)
        for batch in (1, 2):
            with self.subTest(batch=batch):
                p = _p(sampling_sigmas=sigmas)
                dd.AnimaDetailDaemon().process_before_every_sampling(p, *_args(amount=1.0))
                timesteps = sigmas.clone()
                view = timesteps[10].expand(batch)
                params = types.SimpleNamespace(sigma=view, sampling_step=9, total_sampling_steps=20,
                                               denoiser=ForgeDenoiser(p, steps=20, total_steps=20))
                dd._denoiser_callback(params)
                self.assertTrue(torch.equal(timesteps, sigmas))
                self.assertIsNot(params.sigma, view)
                self.assertTrue(torch.equal(params.sigma, view * node_sigma_factor(1.0, 5.0)))


class HiresGateTests(unittest.TestCase):
    # origin: muerrilla/sd-webui-detail-daemon@1947999:scripts/detail_daemon.py:104 (Hires Pass, default False), :263-267 (pass flag),
    # :275-277 (``if daemon['hires'] != self.is_hires_pass: continue``)
    SIGMAS = flow_sigmas(20)

    def _adjusted(self, p, args, **kw):
        return _one_call(p, args, self.SIGMAS, self.SIGMAS[10], **kw)

    def _applied(self, *, hires, is_hr_pass):
        given, adjusted = self._adjusted(_p(cfg_scale=4.0, is_hr_pass=is_hr_pass, sampling_sigmas=self.SIGMAS),
                                         _args(amount=0.1, hires=hires))
        return not torch.equal(adjusted, given)

    def test_off_runs_on_the_base_pass_only(self):
        self.assertTrue(self._applied(hires=False, is_hr_pass=False))
        self.assertFalse(self._applied(hires=False, is_hr_pass=True))

    def test_on_runs_on_the_hires_pass_only(self):
        self.assertFalse(self._applied(hires=True, is_hr_pass=False))
        self.assertTrue(self._applied(hires=True, is_hr_pass=True))

    def test_hires_pass_strength_uses_p_cfg_scale_not_hr_cfg(self):
        # origin: muerrilla/sd-webui-detail-daemon@1947999:scripts/detail_daemon.py:259 — self.cfg_scale = p.cfg_scale
        # The denoiser carries p like Forge's CFGDenoiser, so an hr_cfg lookup through it would show here.
        p = _p(cfg_scale=4.0, is_hr_pass=True, hr_cfg=9.0, sampling_sigmas=self.SIGMAS)
        given, adjusted = self._adjusted(p, _args(amount=0.1, hires=True))
        self.assertTrue(torch.equal(adjusted, given * node_sigma_factor(0.1, 4.0)))
        self.assertFalse(torch.equal(adjusted, given * node_sigma_factor(0.1, 9.0)))

    def test_refiner_pass_strength_uses_p_cfg_scale_not_refiner_cfg(self):
        # CFGDenoiser.forward sets _refiner_pass and then uses p.refiner_cfg; muerrilla keeps p.cfg_scale.
        p = _p(cfg_scale=4.0, refiner_cfg=9.0, sampling_sigmas=self.SIGMAS)
        given, adjusted = self._adjusted(p, _args(amount=0.1), refiner_pass=True)
        self.assertTrue(torch.equal(adjusted, given * node_sigma_factor(0.1, 4.0)))
        self.assertFalse(torch.equal(adjusted, given * node_sigma_factor(0.1, 9.0)))

    def test_img2img_without_hr_flag_counts_as_base_pass(self):
        p = types.SimpleNamespace(cfg_scale=4.0, sampler_name="Euler", extra_generation_params={})
        dd.AnimaDetailDaemon().process_before_every_sampling(p, *_args(amount=0.1))
        self.assertTrue(dd._DD["on"])
        dd.AnimaDetailDaemon().process_before_every_sampling(p, *_args(amount=0.1, hires=True))
        self.assertFalse(dd._DD["on"])

    def test_infotext_is_written_on_both_passes(self):
        for is_hr_pass in (False, True):
            with self.subTest(is_hr_pass=is_hr_pass):
                p = _p(is_hr_pass=is_hr_pass)
                dd.AnimaDetailDaemon().process_before_every_sampling(p, *_args(hires=True))
                self.assertIn("hires=True", p.extra_generation_params["Anima Detail Daemon"])

    def test_thirteen_arg_payload_means_base_pass_only(self):
        old_args = _args(amount=0.1)[:13]
        given, adjusted = self._adjusted(_p(sampling_sigmas=self.SIGMAS), old_args)
        self.assertFalse(torch.equal(adjusted, given))
        given, adjusted = self._adjusted(_p(is_hr_pass=True, sampling_sigmas=self.SIGMAS), old_args)
        self.assertTrue(torch.equal(adjusted, given))


class UnsupportedSamplerTests(unittest.TestCase):
    # origin: muerrilla/sd-webui-detail-daemon@1947999:scripts/detail_daemon.py:197-199 — returns before any daemon is set up
    SIGMAS = flow_sigmas(20)

    def test_unsupported_samplers_match_the_original_list(self):
        self.assertEqual(list(dd._UNSUPPORTED_SAMPLERS), MUERRILLA_UNSUPPORTED_SAMPLERS)

    def test_unsupported_base_sampler_turns_it_off(self):
        for name in MUERRILLA_UNSUPPORTED_SAMPLERS:
            with self.subTest(sampler=name):
                p = _p(sampler_name=name, sampling_sigmas=self.SIGMAS)
                given, adjusted = _one_call(p, _args(amount=0.5), self.SIGMAS, self.SIGMAS[10])
                self.assertTrue(torch.equal(adjusted, given))
                self.assertFalse(dd._DD["on"])
                self.assertNotIn("Anima Detail Daemon", p.extra_generation_params)

    def test_only_the_base_sampler_is_checked(self):
        # Like the original, which reads p.sampler_name only.
        p = _p(sampler_name="Euler", is_hr_pass=True, hr_sampler_name="DPM adaptive", sampling_sigmas=self.SIGMAS)
        given, adjusted = _one_call(p, _args(amount=0.5, hires=True), self.SIGMAS, self.SIGMAS[10])
        self.assertFalse(torch.equal(adjusted, given))


class ArgsAndUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gradio as gr

        with gr.Blocks():
            cls.components = dd.AnimaDetailDaemon().ui(False)

    def test_fourteen_args_with_hires_last(self):
        self.assertEqual(len(self.components), 14)
        hires = self.components[13]
        self.assertEqual(type(hires).__name__, "Checkbox")
        self.assertEqual(hires.label, "Hires Pass")
        self.assertIs(hires.value, False)
        self.assertTrue(hires.visible)

    def test_old_slots_are_hidden_at_neutral_values(self):
        preset, multiplier, cfg_couple = (self.components[i] for i in (1, 10, 12))
        for comp in (preset, multiplier, cfg_couple):
            self.assertFalse(comp.visible, comp.label)
        self.assertEqual(multiplier.value, 1.0)
        self.assertIs(cfg_couple.value, True)

    def test_visible_inputs_match_the_node_inputs(self):
        for index, name in ARG_TO_NODE.items():
            comp = self.components[index]
            spec = NODE_INPUTS[name]
            with self.subTest(arg=index, node_input=name):
                self.assertTrue(comp.visible)
                self.assertEqual(comp.value, spec["default"])
                if "min" in spec:
                    self.assertEqual((comp.minimum, comp.maximum, comp.step),
                                     (spec["min"], spec["max"], spec["step"]))

    def test_engine_defaults_match_the_node(self):
        # Missing args fall back to the node defaults too.
        p = _p()
        dd.AnimaDetailDaemon().process_before_every_sampling(p, True)
        for key, name in (("amount", "detail_amount"), ("start", "start"), ("end", "end"), ("bias", "bias"),
                          ("exponent", "exponent"), ("start_offset", "start_offset"),
                          ("end_offset", "end_offset"), ("fade", "fade"), ("smooth", "smooth")):
            with self.subTest(key=key):
                self.assertEqual(dd._DD[key], NODE_INPUTS[name]["default"])
        self.assertIs(dd._DD["hires"], False)

    def test_ignored_slots_do_not_change_the_result(self):
        sigmas = flow_sigmas(20)
        _, base = _one_call(_p(sampling_sigmas=sigmas), _args(amount=0.3), sigmas, sigmas[10])
        for extra in (dict(preset="Strong"), dict(preset="Medium"), dict(multiplier=0.0),
                      dict(multiplier=2.0), dict(cfg_couple=False)):
            with self.subTest(**extra):
                _, got = _one_call(_p(sampling_sigmas=sigmas), _args(amount=0.3, **extra), sigmas, sigmas[10])
                self.assertTrue(torch.equal(got, base))

    def test_no_presets_left(self):
        self.assertFalse(hasattr(dd, "_PRESETS"))


def tearDownModule():
    dd._DD["on"] = False
    dd._DD["node"] = None


if __name__ == "__main__":
    unittest.main()
