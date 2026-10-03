# Derived from (MIT License, notices below):
#   howardhx/speed@ca7801c9bdffe681742e9592345bcf4885959be5 comfyui/speed_sampler.py:77-279
#     (_expand_and_align_torch, _segment_callback, sample_speed) — Copyright (c) 2026 Howard Xiao
#   aoleg/ComfyUI-SPEED@a8873591a27f2c1e086a2caf546f9b6aeec62b81 speed_core.py:303-662
#     (sample_speed_core: sampler_kwargs / full_res_sampler_kwargs, 5-D latents) and
#     scripts/speed_forge.py:73-184 (_ShapeSafeRandn, the wrapped sample function)
#     — Copyright (c) 2026 A. Izzuddin Al Faruq (fork work by Oleg Afonin)
#   sorryhyun/ComfyUI-Spectrum-KSampler@b46a364aec3b161b889c9cc26cd976a49eb537ae spd.py:110-213
#     (make_speed_sampler: the respace loop, per-run expansion generator) — Copyright (c) 2026 sorryhyun
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
# OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Modified by sam-extra, 2026-10-03 (part of a GPL-3.0-only extension; the MIT notices stay):
#   * one runner for both plans: "transition" (official/aoleg, only the transition sigma is patched)
#     and "respace" (sorryhyun, the remaining schedule is re-spaced); respace runs the selected
#     sampler segment by segment instead of a built-in Euler loop — with Euler it is the same
#     computation as sorryhyun's loop. It is a geometric match of sorryhyun's SPD only (same grids,
#     re-spaced sigmas and expansion noise, without his Spectrum caching of the full-size tail), and
#     odd latents differ: sorryhyun pads them to even before sampling, Forge samples the odd grid;
#   * every random draw is per image seed, so a batch image equals the same seed generated alone:
#     transition mode draws the expansion noise with numpy ``default_rng(seed + (k + 1) * 10000)``
#     per image (upstream draws one stream for the whole batch), respace mode keeps one
#     ``torch.Generator(seed + 10000)`` per image on the latent's device (sorryhyun: one per batch);
#     the coarse segments' ancestral/SDE noise comes from the host per seed (aoleg fell back to one
#     CPU generator for the batch) and Brownian samplers get a coarse tree per seed;
#   * the Brownian noise sampler Forge built for the full grid is replaced on coarse segments by a
#     coarse one from the host (aoleg dropped it, so SDE samplers seeded their own tree randomly);
#   * img2img/hires (an init latent with sigma0 < 1): the DCT-truncated latent is rescaled to the
#     flow form ``(1 - s0) y + s0 eps`` (truncation leaves the image part sqrt(HW/hw) times too
#     strong) — on by default, switchable (setting ``sam3_speed_img2img_rescale``, see
#     ``SpeedRun.img2img_rescale``). It is the inverse of the amplitude argument kappa rests on (a
#     coarse image embedded in the full grid comes out r times too weak): without it the coarse steps
#     see an image r times too strong and, after kappa, so does the full-size tail. txt2img
#     (sigma0 = 1, pure noise) is the same either way;
#   * the patched schedule is published to the host after every transition (Forge's
#     ``sampling_sigmas``, which Detail Daemon, DAVE, MG/HiGS and HiFlow look sigmas up in);
#   * the run reports what it did (coarse steps, grids, sigmas) for the infotext status;
#   * a sampler that declares the keyword ``STEP_OFFSET_KWARG`` gets each segment's first step index
#     in the pass (Euler (SMEA) Dy CFG++ places its sub-steps by it, sam3ext/extra_samplers/euler_dy.py).
"""The SPEED sampler run: plan -> coarse segments -> spectral expansion(s) -> full-size tail.

``run_speed`` wraps any k-diffusion ``sample_*(model, x, sigmas, extra_args, callback, disable,
**kw)`` and calls it once per segment. Forge specifics (the randn hijack, Brownian samplers,
``sampling_sigmas``, seeds) come through :class:`SpeedHost`, so the run is testable on the CPU.
"""

from __future__ import annotations

import contextlib
import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from . import spectral
from .schedule import Plan, Transition

__all__ = [
    "EXPANSION_SEED_STRIDE",
    "INIT_NOISE_SEED_OFFSET",
    "RESPACE_SEED_OFFSET",
    "STEP_OFFSET_KWARG",
    "RunReport",
    "SpeedHost",
    "SpeedRun",
    "run_speed",
]

# official comfyui/speed_sampler.py:272 / aoleg speed_core.py:655 — seed + (k + 1) * 10000 per transition.
EXPANSION_SEED_STRIDE = 10000
# sorryhyun spd.py:142 — torch.Generator(...).manual_seed(seed + 10_000), one per run.
RESPACE_SEED_OFFSET = 10_000
# sam-extra: the img2img init-noise top-up (CPU torch generator per image).
INIT_NOISE_SEED_OFFSET = 7919
# sam-extra: the keyword a sampler declares to learn where a segment starts in the pass. The run
# calls the sampler once per segment, so a sampler whose behaviour depends on the step index (Euler
# (SMEA) Dy CFG++: sub-steps at the pass's steps 2-3 / 0-1) would otherwise restart it every segment.
# Passed only to a sampler that names it in its signature (never through ``**kwargs``).
STEP_OFFSET_KWARG = "sam_extra_step_offset"


class SpeedHost:
    """What the run needs from the sampling host. The defaults are host-free (tests, scripts)."""

    def seeds(self, batch: int) -> list[int]:
        return list(range(batch))

    def noise_scope(self, full_shape: tuple[int, ...]):
        """Context in which coarse ``randn_like`` calls get per-seed coarse noise (Forge hijack)."""
        return contextlib.nullcontext()

    def coarse_noise_sampler(self, template, x: torch.Tensor, stage: int, sigmas: torch.Tensor):
        """A noise sampler for a coarse segment replacing ``template`` (built for the full grid).

        Return None to let the sampler build its default one.
        """
        return None

    def publish_sigmas(self, sigmas: torch.Tensor) -> None:
        """The pass schedule as it is now (after a transition patched it)."""

    def restore_sigmas(self) -> None:
        """Undo :meth:`publish_sigmas` after the run."""

    def log(self, message: str) -> None:
        pass


@dataclass
class SpeedRun:
    """Per-pass inputs that are not part of the plan."""

    plan: Plan
    seeds: list[int]
    init_latent: bool = False          # img2img / hires: the start state holds an image
    img2img_rescale: bool = True       # the sam-extra init correction (setting)


@dataclass
class RunReport:
    applied: bool = False
    coarse_steps: int = 0
    total_steps: int = 0
    events: list = field(default_factory=list)   # dicts per transition
    init: str = "txt2img"

    def summary(self, plan: Plan) -> str:
        grids = " -> ".join(f"{h}x{w}" for h, w in [plan.first_grid] + [t.grid_to for t in plan.transitions])
        parts = [f"{self.coarse_steps}/{self.total_steps} steps coarse", grids]
        for event in self.events:
            parts.append(
                f"step {event['step']} sigma {event['sigma']:.4f}->{event['aligned']:.4f}"
            )
        if self.init != "txt2img":
            parts.append(f"init={self.init}")
        return ", ".join(parts)


# ------------------------------------------------------------------------------------------------
# Noise (per image seed)
# ------------------------------------------------------------------------------------------------


def _item_layout(x: torch.Tensor) -> tuple[int, ...]:
    """Per-image leading axes (``(C,)`` for 4-D, ``(C, T)`` for 5-D latents)."""
    return tuple(int(v) for v in x.shape[1:-2])


def _numpy_draws(seed: int, lead: tuple[int, ...], shapes: list[tuple[int, int]]) -> list[torch.Tensor]:
    """Upstream draw order for one image: ``np.random.default_rng(seed)``, then for every channel
    (``np.ndindex`` over the official 4-D view — a 5-D ``(C, T)`` image is visited as ``(T, C)``)
    one ``standard_normal`` per entry of ``shapes``, cast to float32."""
    seed = int(seed)
    rng = np.random.default_rng(seed if seed >= 0 else seed % 2 ** 64)   # numpy refuses negative entropy
    if len(lead) == 2:
        view = (lead[1], lead[0])          # official: x.permute(0, 2, 1, 3, 4) -> (B*T, C, h, w)
    else:
        view = lead
    arrays = [np.empty(view + tuple(s), dtype=np.float32) for s in shapes]
    for idx in np.ndindex(*view):
        for k, s in enumerate(shapes):
            arrays[k][idx] = rng.standard_normal(tuple(s)).astype(np.float32)
    out = [torch.from_numpy(a) for a in arrays]
    if len(lead) == 2:
        out = [t.permute(1, 0, 2, 3) for t in out]
    return out


def _torch_draws(generator: torch.Generator, lead: tuple[int, ...], shapes: list[tuple[int, int]], device) -> list[torch.Tensor]:
    """sorryhyun's draw for one image: ``torch.randn((1, C, H, W), generator, device, float32)``."""
    return [
        torch.randn((1, *lead, *s), generator=generator, device=device, dtype=torch.float32)[0]
        for s in shapes
    ]


def _draw_shapes(transform: str, grid_from, grid_to) -> list[tuple[int, int]]:
    if transform == "dwt":
        return [tuple(grid_from)] * 3            # LH, HL, HH (official order)
    if transform == "fft":
        return [tuple(grid_to)] * 2              # real, imaginary
    return [tuple(grid_to)]


def _expand(transform: str, x: torch.Tensor, grid_to, sigma: float, draws: list[torch.Tensor]) -> torch.Tensor:
    if transform == "dct":
        return spectral.expand_dct(x, grid_to, sigma, draws[0])
    if transform == "fft":
        return spectral.expand_fft(x, grid_to, sigma, draws[0], draws[1])
    return spectral.expand_haar(x, sigma, draws[0], draws[1], draws[2])


def _stack(per_image: list[list[torch.Tensor]], device) -> list[torch.Tensor]:
    return [torch.stack([image[k] for image in per_image]).to(device=device) for k in range(len(per_image[0]))]


# ------------------------------------------------------------------------------------------------
# The run
# ------------------------------------------------------------------------------------------------


def _declares(fn: Callable, name: str) -> bool:
    """Does ``fn``'s signature name the keyword ``name`` (not just ``**kwargs``)?"""
    try:
        parameter = inspect.signature(fn).parameters.get(name)
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
    )


def _segment_callback(outer: Callable | None, start: int):
    """Re-base callback step indices to the whole pass (official ``_segment_callback``)."""
    if outer is None:
        return None

    def inner(d):
        d = dict(d)
        d["i"] = d.get("i", 0) + start
        return outer(d)

    return inner


def _init_coarse(x: torch.Tensor, run: SpeedRun, sigma0: float, report: RunReport) -> torch.Tensor:
    plan = run.plan
    grid = plan.first_grid
    if tuple(x.shape[-2:]) == tuple(grid):
        return x
    work = spectral.dct_downscale(x, grid)
    if run.init_latent and sigma0 < 1.0 - 1e-6:
        if run.img2img_rescale:
            big_h, big_w = plan.full_grid
            r_eff = math.sqrt((big_h * big_w) / (grid[0] * grid[1]))
            top_up = sigma0 * math.sqrt(max(0.0, 1.0 - 1.0 / (r_eff * r_eff)))
            noise = []
            for seed in run.seeds:
                gen = torch.Generator(device="cpu").manual_seed(int(seed) + INIT_NOISE_SEED_OFFSET)
                noise.append(torch.randn(tuple(work.shape[1:]), generator=gen, dtype=torch.float32))
            noise = torch.stack(noise).to(device=work.device, dtype=work.dtype)
            work = work / r_eff + top_up * noise
            report.init = f"rescaled 1/{r_eff:.3f}"
        else:
            report.init = "upstream (not rescaled)"
    return work.to(dtype=x.dtype)


def run_speed(
    base_fn: Callable,
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    run: SpeedRun,
    host: SpeedHost | None = None,
    extra_args: dict | None = None,
    callback: Callable | None = None,
    disable=None,
    sampler_kwargs: dict | None = None,
) -> tuple[torch.Tensor, RunReport]:
    """Run ``base_fn`` segment by segment along ``run.plan``; returns ``(x, report)``."""
    host = host or SpeedHost()
    plan = run.plan
    extra_args = {} if extra_args is None else extra_args
    kwargs = dict(sampler_kwargs or {})
    template = kwargs.pop("noise_sampler", None)
    report = RunReport(total_steps=len(sigmas) - 1)
    if x.ndim not in (4, 5):
        raise ValueError(f"SPEED expects a 4-D or 5-D latent, got {tuple(x.shape)}")
    if tuple(x.shape[-2:]) != tuple(plan.full_grid):
        raise ValueError(f"plan grid {plan.full_grid} does not match the latent {tuple(x.shape[-2:])}")
    batch = int(x.shape[0])
    seeds = [int(s) for s in run.seeds]
    if len(seeds) != batch:
        raise ValueError(f"{len(seeds)} seeds for a batch of {batch}")
    lead = _item_layout(x)
    full_shape = tuple(x.shape)
    device = x.device

    if plan.mode == "respace":
        sig = sigmas.detach().clone().float()           # sorryhyun: sigmas.detach().clone().float()
        generators = [
            torch.Generator(device=device).manual_seed(seed + RESPACE_SEED_OFFSET) for seed in seeds
        ]
    else:
        sig = sigmas.clone()                            # official: sigmas.clone()
        generators = []

    sigma0 = float(sigmas[0])
    stage_cache: dict[int, Any] = {}
    pass_offset = _declares(base_fn, STEP_OFFSET_KWARG)

    def segment(x_cur: torch.Tensor, start: int, end: int, stage: int, full: bool) -> torch.Tensor:
        seg = sig[start:end + 1]
        if len(seg) < 2:
            return x_cur
        seg_kwargs = dict(kwargs)
        if pass_offset:
            seg_kwargs[STEP_OFFSET_KWARG] = start
        if template is not None:
            if full:
                seg_kwargs["noise_sampler"] = template
            else:
                if stage not in stage_cache:
                    stage_cache[stage] = host.coarse_noise_sampler(template, x_cur, stage, sigmas)
                if stage_cache[stage] is not None:
                    seg_kwargs["noise_sampler"] = stage_cache[stage]
        return base_fn(
            model, x_cur, seg, extra_args=extra_args,
            callback=_segment_callback(callback, start), disable=disable, **seg_kwargs,
        )

    with host.noise_scope(full_shape):
        try:
            x_cur = _init_coarse(x, run, sigma0, report)
            start = 0
            stage = 0
            for transition in plan.transitions:
                x_cur = segment(x_cur, start, transition.step, stage, full=False)
                x_cur, event = _apply_transition(x_cur, transition, plan, sig, seeds, generators, lead, device, x.dtype)
                report.events.append(event)
                host.publish_sigmas(sig)
                start = transition.step
                stage += 1
            x_cur = segment(x_cur, start, len(sig) - 1, stage, full=True)
        finally:
            host.restore_sigmas()
    if tuple(x_cur.shape) != full_shape:
        raise RuntimeError(f"SPEED ended on {tuple(x_cur.shape)}, expected {full_shape}")
    report.applied = True
    report.coarse_steps = plan.coarse_steps
    return x_cur, report


def _apply_transition(
    x_cur: torch.Tensor, transition: Transition, plan: Plan, sig: torch.Tensor, seeds: list[int],
    generators: list[torch.Generator], lead: tuple[int, ...], device, dtype,
) -> tuple[torch.Tensor, dict]:
    step = transition.step
    r = transition.scale_to / transition.scale_from
    shapes = _draw_shapes(plan.transform, transition.grid_from, transition.grid_to)
    if tuple(x_cur.shape[-2:]) != tuple(transition.grid_from):
        raise RuntimeError(f"SPEED: latent {tuple(x_cur.shape[-2:])} is not the planned {transition.grid_from}")
    if plan.mode == "respace":
        old = float(sig[step])
        new, factor = spectral.respace_alignment(old, r)
        draws = _stack([_torch_draws(g, lead, shapes, device) for g in generators], device)
        expanded = _expand(plan.transform, x_cur, transition.grid_to, old, draws)
        x_new = (expanded * factor).to(dtype=dtype)
        if old > 0.0 and new != old:
            sig[step + 1:] = new * (sig[step + 1:] / old)
        sig[step] = new
        return x_new, {"step": step, "k": transition.k, "sigma": old, "aligned": float(sig[step]),
                       "grid": tuple(transition.grid_to)}
    t = float(sig[step])
    seed_offset = (transition.k + 1) * EXPANSION_SEED_STRIDE
    draws = _stack([_numpy_draws(seed + seed_offset, lead, shapes) for seed in seeds], device)
    expanded = _expand(plan.transform, x_cur, transition.grid_to, t, draws)
    x_new = (spectral.kappa(t, r) * expanded).to(dtype=dtype)
    sig[step] = float(spectral.align_timestep(t, r))
    return x_new, {"step": step, "k": transition.k, "sigma": t, "aligned": float(sig[step]),
                   "grid": tuple(transition.grid_to)}
