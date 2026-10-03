# Copyright 2024 KBlueLeaf
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is derived from Koishi-Star/Euler-Smea-Dyn-Sampler
# (https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler, commit
# d98a504c8419be5274068ad91ca6bbf2e13635a8, ``smea_sampling.py``); the copyright line above is the
# one in that repository's LICENSE (Apache License 2.0, no NOTICE file).
# MODIFIED by sam-extra, 2026-10-03 (Apache-2.0 section 4(b)) — every change is listed in the
# docstring below.
"""Euler Dy CFG++ and Euler SMEA Dy CFG++ — Koishi-Star's Dy/SMEA sub-steps with Forge's CFG++ update.

Upstream (``smea_sampling.py``):

* ``sample_euler_dy`` — Euler with Karras churn; at steps 2 and 3 (``i // 2 == 1``) it first runs
  ``dy_sampling_step``: the model sees the latent's (odd row, odd column) pixel of every 2×2 block,
  a half-resolution latent; one Euler step moves those pixels; they are written back, and the
  step's own Euler update is then added on top (``x = x + d * dt`` with the full-resolution ``d``).
* ``sample_euler_smea_dy`` — after the step's Euler update, at step 0 ``smea_sampling_step``
  (``nearest-exact`` ×1.25 upscale, one Euler step, back to the original size) and at step 1
  ``dy_sampling_step``. Upstream writes the conditions as ``i + 1 // 2 == 0`` / ``== 1``; ``1 // 2``
  binds first, so they are ``i == 0`` / ``i == 1``, which is what runs here. The sub-steps reuse the
  step's ``sigma_hat`` and ``dt`` on the already-updated latent (upstream behaviour, kept).
* ``_Rescaler`` (WebUI branch) — during a sub-step the denoiser's ``init_latent``/``mask``/``nmask``
  are resized (``nearest-exact``) to the sub-step resolution and restored afterwards.

What sam-extra changed:

1. **CFG++.** Every Euler update — the step's and both sub-steps' — is Forge's CFG++ update
   (``sample_euler_ancestral_cfg_pp`` at η = 0, as Forge's built-in Euler CFG++):
   ``x' = α_t·x0 + σ_t·(x − α_s·x0_uncond)/σ_s`` with ``α = σ·e^{λ(σ)}`` from Forge's half-log-SNR
   helpers and the unconditional x0 kept by the same post-CFG hook (``common.install_uncond_capture``).
   Upstream adds the Dy sub-step's change and the step's Euler increment; with CFG++ the step's
   result is ``CFG++(x̂) + (x_dy − x̂)``, the CFG++ step plus the sub-step's change on the pixels it
   touched (exactly upstream's sum when the CFG++ terms are replaced by Euler ones). The step that
   reaches σ = 0 returns the guided x0, as Forge's does. Without an unconditional prediction (none
   was evaluated) the update falls back to the plain Euler one.
2. **Churn rule.** Upstream clamps ``gamma = max(s_churn / N, √2 − 1)``, i.e. at least √2 − 1 on every
   step inside ``[s_tmin, s_tmax]`` even with ``s_churn = 0``. The registered samplers use
   ``min`` — k-diffusion's ``sample_euler`` / Karras et al. 2022 (arXiv:2206.00364) Algorithm 2 —
   so the default ``s_churn = 0`` does not churn. ``churn_rule="max"`` reproduces upstream and is what
   the origin-parity test uses. Upstream's ``eps = randn_like(x) * s_noise`` drawn on every step
   (also when gamma is 0) and its ``x = x − eps·sqrt(σ̂² − σ²)`` are kept; the noise comes from Forge's
   ``k_diffusion.sampling.default_noise_sampler`` (per-image seeds, see ``common``) instead of the
   module's own ``torch.randn_like``.
3. **Flow models** (``prediction_type == "const"``: ``x = (1 − σ)·x0 + σ·n``). Churn runs on the
   epsilon-equivalent noise level ``s = σ/(1 − σ)``: ``ŝ = s·(1 + γ)``, ``x̂ = α̂·(x/α − eps·sqrt(ŝ² − s²))``,
   ``σ̂ = ŝ/(1 + ŝ)``, ``α̂ = 1 − σ̂`` — the epsilon-model rule in those coordinates, never leaving
   σ ∈ [0, 1). A step at σ ≥ 1 (pure noise) is not churned. ``s_tmin``/``s_tmax`` compare the model's
   own σ, as Forge's ``sample_euler`` does. The CFG++ update carries α explicitly, so it is the flow
   update; the non-CFG++ Euler update ``x + d·dt`` is the flow ODE's Euler step as well.
4. **5-D latents** (Anima, Wan: ``[B, C, T, H, W]``). The sub-steps work on the last two dimensions
   (frames untouched); the ×1.25 SMEA resize interpolates each frame on its own. Upstream's
   ``unfold`` gather is written as strided indexing — the same pixels, the same odd-size handling,
   including its bottom-right corner (when both sizes are odd that pixel takes the value above it).
5. **Sub-step inputs.** Forge's masks are 4-D/5-D (A1111's were 3-D); all three ranks are resized.
   ``extra_args["image_cond"]`` (an inpainting model's concat conditioning) is resized too when it
   has the latent's size — Forge only concatenates a matching one — and so are Forge's 5-D
   ``dynamic_args.ref_latents`` of the latent's size (Anima concatenates them to the latent along the
   frame axis, which would otherwise fail at the sub-step's resolution). The sub-step's model options
   carry ``transformer_options["sam_extra_substep"] = "dy" | "smea"`` (``common.SUBSTEP_MARKER``).
   After a sub-step ``after_substep(denoised)`` re-announces the step's full-resolution x0, which
   Forge's sampler uses to restore the live-preview / interrupt latent the sub-step overwrote.
6. **Step counter.** Forge's CFG denoiser counts its calls in ``step`` and reads it for prompt
   editing (``prompt_parser.reconstruct_*_batch(…, self.step)``), skip-early-CFG and NGMS
   (``self.step / self.total_steps``, ``self.step % 2``) and the refiner's switch-by-steps
   (``sd_samplers_common.apply_refiner``). A sub-step is evaluated with ``step`` one lower — as the
   step it belongs to — and leaves the count as it found it, so those schedules stay on the step
   index (upstream: every sub-step shifted them by one call).
7. **Plain steps.** ``substeps=False`` runs no sub-step at all: Euler Dy CFG++ / Euler SMEA Dy CFG++
   are then Forge's Euler CFG++ with the churn above. ``registry`` passes it when the request has
   full-resolution machinery a sub-step cannot follow (``substep_guard``: Forge's Spectrum
   Integrated, Wan 2.2 I2V ``concat_latent``, PiD ``lq_latent``, ``unet.extra_concat_condition``).
8. Host bookkeeping: Forge's ``trange``; a latent smaller than 2×2 skips the Dy sub-step (upstream
   would call the model on an empty tensor).
9. **Segmented runs.** Anima SPEED (``sam3ext/speed/runner.py``) samples one pass as several calls
   of the sampler, one per resolution segment, and hands each call the index of its first step in
   ``sam_extra_step_offset`` (``runner.STEP_OFFSET_KWARG``). The sub-steps are placed by that pass
   index — Dy at the pass's steps 2-3, SMEA at 0-1, on whichever grid the pass is on there — instead
   of repeating at the start of every segment (a SMEA ×1.25 step right after SPEED's expansion).
   Plain sampling never passes it (0).
"""

from __future__ import annotations

import importlib
from functools import partial

import torch
import torch.nn.functional as F

from .common import install_uncond_capture, is_const, k_sampling, model_sampling, with_substep_marker

__all__ = [
    "CHURN_MAX",
    "CHURN_MIN",
    "churn_gamma",
    "euler_dy",
    "sample_euler_dy_cfg_pp",
    "sample_euler_smea_dy_cfg_pp",
    "spatial_interpolate",
]

CHURN_MIN = "min"   # k-diffusion sample_euler, Karras et al. 2022 Algorithm 2
CHURN_MAX = "max"   # Koishi-Star sample_euler_dy / sample_euler_smea_dy
_CHURN_LIMIT = 2 ** 0.5 - 1

_SMEA_SCALE = (1.25, 1.25)        # smea_sampling_step
_RESIZE_MODE = "nearest-exact"    # smea_sampling_step and _Rescaler
_RESCALED_ATTRS = ("init_latent", "mask", "nmask")   # _Rescaler, WebUI branch


def churn_gamma(rule: str, s_churn: float, steps: int, sigma, s_tmin: float, s_tmax: float) -> float:
    """The churn factor γ of one step (0 outside ``[s_tmin, s_tmax]``)."""
    if rule == CHURN_MAX:
        return max(s_churn / steps, _CHURN_LIMIT) if s_tmin <= sigma <= s_tmax else 0.
    if rule == CHURN_MIN:
        return min(s_churn / steps, _CHURN_LIMIT) if s_tmin <= sigma <= s_tmax else 0.
    raise ValueError(f"unknown churn rule: {rule!r}")


def spatial_interpolate(t: torch.Tensor, *, size=None, scale_factor=None, mode: str = _RESIZE_MODE) -> torch.Tensor:
    """``F.interpolate`` over the last two dimensions of a (C,H,W), (B,C,H,W) or (B,C,T,H,W) tensor.

    3-D is upstream's mask handling (``unsqueeze(0)`` … ``squeeze(0)``); 5-D folds the frames into the
    batch so each frame gets exactly the 2-D interpolation (with the same ``scale_factor`` semantics)."""
    if t.ndim == 4:
        return F.interpolate(t, size=size, scale_factor=scale_factor, mode=mode)
    if t.ndim == 3:
        return F.interpolate(t.unsqueeze(0), size=size, scale_factor=scale_factor, mode=mode).squeeze(0)
    if t.ndim == 5:
        b, c, frames, h, w = t.shape
        flat = t.permute(0, 2, 1, 3, 4).reshape(b * frames, c, h, w)
        out = F.interpolate(flat, size=size, scale_factor=scale_factor, mode=mode)
        return out.reshape(b, frames, c, out.shape[-2], out.shape[-1]).permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(f"expected a 3-D, 4-D or 5-D tensor, got shape {tuple(t.shape)}")


def _forge_dynamic_args():
    """Forge's ``backend.args.dynamic_args`` (None outside Forge)."""
    try:
        return getattr(importlib.import_module("backend.args"), "dynamic_args", None)
    except Exception:
        return None


class _SubstepInputs:
    """Upstream ``_Rescaler`` for Forge: resolution-bound inputs follow the sub-step, then come back.

    * the denoiser's ``init_latent`` / ``mask`` / ``nmask`` (upstream's WebUI branch);
    * ``extra_args["image_cond"]`` of the latent's size (an inpainting model's concat conditioning);
    * on 5-D latents, Forge's ``dynamic_args.ref_latents`` of the latent's size: Anima's DiT concatenates
      them to the latent along the frame axis (``backend/nn/anima.py``), so a reference that is not
      resized with the latent fails ``torch.cat``. Other references are left alone (Flux, Qwen and
      Krea embed theirs as separate tokens of any size, on 4-D latents);
    * Forge's denoiser call counter ``model.step`` (module docstring, item 6): one lower while the
      sub-step is evaluated, so the evaluation counts as its step's and the count ends where it was."""

    def __init__(self, model, extra_args: dict, full_shape: tuple, size: tuple, kind: str):
        self.model = model
        self.extra_args = extra_args
        self.full_ndim = len(full_shape)
        self.full_hw = tuple(int(v) for v in full_shape[-2:])
        self.size = tuple(int(v) for v in size)
        self.kind = kind
        self._saved: dict = {}
        self._refs = None
        self._step = None

    def _matches(self, value, ndims) -> bool:
        return torch.is_tensor(value) and value.ndim in ndims and tuple(value.shape[-2:]) == self.full_hw

    def __enter__(self) -> dict:
        self._saved = {}
        for name in _RESCALED_ATTRS:
            if not hasattr(self.model, name):
                continue
            value = getattr(self.model, name)
            self._saved[name] = value
            if torch.is_tensor(value) and value.ndim >= 3:
                setattr(self.model, name, spatial_interpolate(value, size=self.size, mode=_RESIZE_MODE))
        args = with_substep_marker(self.extra_args, self.kind)
        if self._matches(args.get("image_cond"), (4, 5)):
            args["image_cond"] = spatial_interpolate(args["image_cond"], size=self.size, mode=_RESIZE_MODE)
        if self.full_ndim == 5:
            dynamic_args = _forge_dynamic_args()
            refs = getattr(dynamic_args, "ref_latents", None)
            if isinstance(refs, list) and any(self._matches(ref, (5,)) for ref in refs):
                self._refs = (dynamic_args, refs)
                dynamic_args.ref_latents = [
                    spatial_interpolate(ref, size=self.size, mode=_RESIZE_MODE) if self._matches(ref, (5,)) else ref
                    for ref in refs
                ]
        # Last, so nothing above can fail after it: the step's own evaluation already counted.
        step = getattr(self.model, "step", None)
        if isinstance(step, int) and not isinstance(step, bool) and step >= 1:
            self._step = step
            self.model.step = step - 1
        return args

    def __exit__(self, exc_type, exc, tb) -> bool:
        for name, value in self._saved.items():
            setattr(self.model, name, value)
        self._saved = {}
        if self._refs is not None:
            dynamic_args, refs = self._refs
            dynamic_args.ref_latents = refs
            self._refs = None
        if self._step is not None:
            # Forge's forward added one again — or raised (Interrupt) before it did.
            self.model.step = self._step
            self._step = None
        return False


def _evaluate(model, x, sigma, extra_args: dict, capture):
    """One model call → ``(guided x0, unconditional x0 or None)``."""
    if capture is not None:
        capture.value = None
    denoised = model(x, sigma * x.new_ones([x.shape[0]]), **extra_args)
    if capture is None:
        return denoised, None
    uncond = capture.value
    if not torch.is_tensor(uncond) or uncond.shape != denoised.shape:
        uncond = denoised   # no unconditional prediction → the update is plain Euler
    return denoised, uncond


def _cfg_pp_target(ks, lambda_fn, x, denoised, uncond, sigma, sigma_next):
    """Forge's CFG++ Euler step from ``sigma`` to ``sigma_next`` (> 0).

    origin: Forge Neo 2.29.2 modules_forge/packages/k_diffusion/sampling.py
    ``sample_euler_ancestral_cfg_pp`` (``eta = 0`` as ``sample_euler_cfg_pp`` calls it)."""
    alpha_s = sigma * lambda_fn(sigma).exp()
    alpha_t = sigma_next * lambda_fn(sigma_next).exp()
    d = ks.to_d(x, sigma, alpha_s * uncond)  # to noise
    sigma_down, _sigma_up = ks.get_ancestral_step(sigma / alpha_s, sigma_next / alpha_t, eta=0.0)
    sigma_down = alpha_t * sigma_down
    return alpha_t * denoised + sigma_down * d


def _churn(x, sigma, gamma: float, eps, const: bool):
    """Raise the noise level of ``x`` by γ → ``(x̂, σ̂)`` (see the module docstring, item 3)."""
    if not const:
        # Koishi-Star sample_euler_dy: sigma_hat = sigmas[i] * (gamma + 1), x = x - eps * sqrt(…)
        sigma_hat = sigma * (gamma + 1)
        if gamma > 0:
            x = x - eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5
        return x, sigma_hat
    if not gamma > 0 or not (0 < sigma < 1):
        return x, sigma
    alpha = 1 - sigma
    s = sigma / alpha
    s_hat = s * (gamma + 1)
    sigma_hat = s_hat / (1 + s_hat)
    alpha_hat = 1 - sigma_hat
    x = alpha_hat * (x / alpha - eps * (s_hat ** 2 - s ** 2) ** 0.5)
    return x, sigma_hat


def _step_target(ks, lambda_fn, cfg_pp: bool, x, denoised, uncond, sigma_hat, sigma_next, dt):
    """The latent after one Euler (or CFG++) step of ``x`` from ``sigma_hat`` to ``sigma_next``."""
    if not cfg_pp:
        return x + ks.to_d(x, sigma_hat, denoised) * dt
    if sigma_next == 0:
        return denoised
    return _cfg_pp_target(ks, lambda_fn, x, denoised, uncond, sigma_hat, sigma_next)


def _dy_substep(ctx, x, sigma_hat, sigma_next, dt):
    """Upstream ``dy_sampling_step`` on the last two dimensions (any leading shape)."""
    ks, lambda_fn, model, extra_args, capture, cfg_pp = ctx
    height, width = int(x.shape[-2]), int(x.shape[-1])
    m, n = height // 2, width // 2
    if m == 0 or n == 0:
        return x
    c = x[..., 1:2 * m:2, 1:2 * n:2].contiguous()
    with _SubstepInputs(model, extra_args, tuple(x.shape), (m, n), "dy") as args:
        den_c, unc_c = _evaluate(model, c, sigma_hat, args, capture)
    c = _step_target(ks, lambda_fn, cfg_pp, c, den_c, unc_c, sigma_hat, sigma_next, dt)
    out = x.clone()
    out[..., 1:2 * m:2, 1:2 * n:2] = c
    if height % 2 == 1 and width % 2 == 1:
        # Upstream rebuilds the odd row and column, then sets the corner from its column copy
        # (taken after the row was cropped): the bottom-right pixel gets the one above it.
        out[..., -1:, -1:] = x[..., 2 * m - 1:2 * m, -1:]
    return out


def _smea_substep(ctx, x, sigma_hat, sigma_next, dt):
    """Upstream ``smea_sampling_step``: ×1.25 ``nearest-exact``, one step, back to the input size."""
    ks, lambda_fn, model, extra_args, capture, cfg_pp = ctx
    height, width = int(x.shape[-2]), int(x.shape[-1])
    x_up = spatial_interpolate(x, scale_factor=_SMEA_SCALE, mode=_RESIZE_MODE)
    with _SubstepInputs(model, extra_args, tuple(x.shape), tuple(x_up.shape[-2:]), "smea") as args:
        den, unc = _evaluate(model, x_up, sigma_hat, args, capture)
    x_up = _step_target(ks, lambda_fn, cfg_pp, x_up, den, unc, sigma_hat, sigma_next, dt)
    return spatial_interpolate(x_up, size=(height, width), mode=_RESIZE_MODE)


@torch.no_grad()
def euler_dy(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0,
    noise_sampler=None, after_substep=None, *,
    smea: bool = False, cfg_pp: bool = True, churn_rule: str = CHURN_MIN, substeps: bool = True,
    sam_extra_step_offset: int = 0,
):
    """Euler Dy (``smea=False``) or Euler SMEA Dy (``smea=True``); see the module docstring.

    ``cfg_pp=False, churn_rule="max"`` is upstream's ``sample_euler_dy`` / ``sample_euler_smea_dy``
    (the origin-parity tests run both against the verbatim upstream file). ``substeps=False``: the
    plain steps only (module docstring, item 7). ``sam_extra_step_offset``: the pass index of
    ``sigmas[0]`` when the pass is sampled in segments (item 9)."""
    ks = k_sampling()
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = ks.default_noise_sampler(x) if noise_sampler is None else noise_sampler
    sampling = model_sampling(model)
    const = is_const(sampling)
    lambda_fn = partial(ks.sigma_to_half_log_snr, model_sampling=sampling)
    capture = install_uncond_capture(extra_args, ks) if cfg_pp else None
    ctx = (ks, lambda_fn, model, extra_args, capture, cfg_pp)
    steps = len(sigmas) - 1
    offset = int(sam_extra_step_offset or 0)

    for i in ks.trange(steps, disable=disable):
        step = i + offset   # the step's index in the pass (item 9); ``i`` within this call
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        gamma = churn_gamma(churn_rule, s_churn, steps, sigma, s_tmin, s_tmax)
        eps = noise_sampler(sigma, sigma_next) * s_noise
        x, sigma_hat = _churn(x, sigma, gamma, eps, const)
        dt = sigma_next - sigma_hat
        denoised, uncond = _evaluate(model, x, sigma_hat, extra_args, capture)

        if not smea:
            x_dy = x
            if substeps and sigma_next > 0 and step // 2 == 1:
                x_dy = _dy_substep(ctx, x, sigma_hat, sigma_next, dt)
                if after_substep is not None:
                    after_substep(denoised)
            if callback is not None:
                callback({"x": x_dy, "i": i, "sigma": sigma, "sigma_hat": sigma_hat, "denoised": denoised})
            if not cfg_pp:
                # Euler method (upstream: d from the step's own evaluation, added to the Dy result)
                x = x_dy + ks.to_d(x, sigma_hat, denoised) * dt
            else:
                stepped = _step_target(ks, lambda_fn, True, x, denoised, uncond, sigma_hat, sigma_next, dt)
                x = stepped if x_dy is x else stepped + (x_dy - x)
        else:
            x = _step_target(ks, lambda_fn, cfg_pp, x, denoised, uncond, sigma_hat, sigma_next, dt)
            if substeps and sigma_next > 0:
                ran = False
                if step == 1:   # upstream: ``if i + 1 // 2 == 1``
                    x = _dy_substep(ctx, x, sigma_hat, sigma_next, dt)
                    ran = True
                if step == 0:   # upstream: ``if i + 1 // 2 == 0``
                    x = _smea_substep(ctx, x, sigma_hat, sigma_next, dt)
                    ran = True
                if ran and after_substep is not None:
                    after_substep(denoised)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma_hat, "denoised": denoised})
    return x


@torch.no_grad()
def sample_euler_dy_cfg_pp(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0,
    noise_sampler=None, after_substep=None, substeps=True, sam_extra_step_offset=0,
):
    """Euler Dy with the CFG++ update (half-resolution sub-step at steps 2 and 3)."""
    return euler_dy(
        model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
        s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise,
        noise_sampler=noise_sampler, after_substep=after_substep,
        smea=False, cfg_pp=True, churn_rule=CHURN_MIN, substeps=bool(substeps),
        sam_extra_step_offset=sam_extra_step_offset,
    )


@torch.no_grad()
def sample_euler_smea_dy_cfg_pp(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0,
    noise_sampler=None, after_substep=None, substeps=True, sam_extra_step_offset=0,
):
    """Euler SMEA Dy with the CFG++ update (×1.25 sub-step at step 0, half-resolution at step 1)."""
    return euler_dy(
        model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
        s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise,
        noise_sampler=noise_sampler, after_substep=after_substep,
        smea=True, cfg_pp=True, churn_rule=CHURN_MIN, substeps=bool(substeps),
        sam_extra_step_offset=sam_extra_step_offset,
    )
