# Derived from (MIT License, notice below):
#   aoleg/ComfyUI-SPEED@a8873591a27f2c1e086a2caf546f9b6aeec62b81 scripts/speed_forge.py:73-413
#     (_ShapeSafeRandn, _anima_reference_latents_active, _make_speed_func, the inpaint / hires
#     guards, wrapping ``p.sampler.func`` in ``process_before_every_sampling``)
#     — Copyright (c) 2026 A. Izzuddin Al Faruq (the Forge Neo script is Oleg Afonin's work)
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
# Modified by sam-extra, 2026-10-03 (part of a GPL-3.0-only extension; the MIT notice stays):
#   * no class-level state: the settings travel in the wrapper's closure and the per-generation
#     status lives on ``p`` (aoleg kept ``enabled``/``config``/pass counters on the Script class);
#   * the guards run when the sampler actually starts (every script hook has run by then) and each
#     writes its reason to the infotext: masks / inpaint models, Anima reference latents and the
#     other full-size latents Forge hands the DiT through ``dynamic_args`` (aoleg guarded only the
#     Anima ones): Flux Kontext / Flux.2 Klein / Qwen-Image-Edit / Krea 2 reference latents, Wan 2.2
#     I2V ``concat_latent``, PiD ``lq_latent``; ControlNet (incl. Anima ControlNet-LLLite), non-flow
#     predictors (SDXL/SD1.x are refused, aoleg only warned), Forge's built-in Spectrum Integrated,
#     samplers without a sigma schedule or that re-plan it (Restart, UniPC), no coarse steps;
#   * ``_ShapeSafeRandn`` became a per-seed hijack: a coarse ``randn_like`` draws from Forge's own
#     per-image generators (``p.rng``, honouring the "random number source" option) instead of one
#     CPU generator for the whole batch;
#   * Brownian samplers get a coarse BrownianTree per image seed on coarse segments;
#   * the patched schedule is published as ``transformer_options['sampling_sigmas']`` while the run
#     lasts and the original is put back afterwards; each published copy is tagged with the list it
#     stands in for (``sam3ext.guidance.sigmas.mark_republished``) so Colorcraft follows it;
#   * the wrapper applies once, to the request it was attached for (another run through the same
#     sampler object passes through), falls back to plain sampling if SPEED itself fails, and on an
#     interrupt during a coarse segment resizes Forge's ``state.current_latent`` to the full grid.
"""Forge glue for SPEED: settings, guards, per-seed noise, ``sampling_sigmas`` and the sampler wrapper.

Nothing here imports Forge. ``backend.args``, ``modules.rng``, ``modules.shared`` and
``k_diffusion.sampling`` are read from ``sys.modules`` (Forge has loaded them long before a sampler
runs), so the module works in tests with stub objects and never pulls Forge in by itself.
"""

from __future__ import annotations

import contextlib
import functools
import math
import sys
import traceback
from dataclasses import dataclass, field
from typing import Callable

import torch

from sam3ext.guidance.sigmas import mark_republished

from . import schedule, spectral
from .runner import SpeedHost, SpeedRun, run_speed
from .schedule import PlanError

__all__ = [
    "ARG_ALIASES",
    "ARG_NAMES",
    "BROWNIAN_STAGE_SEED_STRIDE",
    "COARSE_FALLBACK_SEED_OFFSET",
    "DEFAULTS",
    "INFOTEXT_IMG2IMG_RESCALE",
    "KEY",
    "OPT_IMG2IMG_RESCALE",
    "OPT_LOG",
    "OWNER",
    "STATUS_KEY",
    "ForgeSpeedHost",
    "SpeedSettings",
    "attach",
    "begin_pass",
    "coerce_settings",
    "make_wrapper",
    "parse_summary",
    "pass_label",
    "record_status",
    "runtime_guard",
]

OWNER = "sam-extra/anima-speed"
KEY = "Anima SPEED"
STATUS_KEY = "Anima SPEED status"
OPT_LOG = "sam3_speed_log"
OPT_IMG2IMG_RESCALE = "sam3_speed_img2img_rescale"
INFOTEXT_IMG2IMG_RESCALE = "Anima SPEED img2img rescale"
_STATE_ATTR = "_sam3_speed_state"
XYZ_ATTR = "_anima_speed_xyz"

# Coarse Brownian trees are seeded away from Forge's full-grid tree (same image seed otherwise).
BROWNIAN_STAGE_SEED_STRIDE = 1_000_003
# Coarse randn_like without Forge's per-image generators (p.rng missing or a different batch).
COARSE_FALLBACK_SEED_OFFSET = 104_729

# Positional script arguments (append-only contract for API callers).
ARG_NAMES = (
    "enabled", "mode", "threshold", "preset", "scales", "delta", "sigma_divisor",
    "manual_sigmas", "adaptive", "transform", "spectrum_A", "spectrum_beta", "seed", "hires",
)
DEFAULTS = {
    "enabled": False,
    "mode": "transition",
    "threshold": "neo_shift",
    "preset": "anima",
    "scales": "0.5,1.0",
    "delta": 0.01,
    "sigma_divisor": 1.03,
    "manual_sigmas": "0.7",
    "adaptive": True,
    "transform": "dct",
    "spectrum_A": schedule.CUSTOM_DEFAULT_A,
    "spectrum_beta": schedule.CUSTOM_DEFAULT_BETA,
    "seed": -1,
    "hires": False,
}
# Samplers that re-plan their own schedule (Restart: karras re-spacing + restarts up to sigma 2,
# expects a final sigma of 0) or need a fixed minimum length (UniPC order), and the k-diffusion
# solvers without a sigma list.
UNSUPPORTED_SAMPLERS = {
    "restart_sampler": "Restart re-plans its own schedule",
    "sample_unipc": "UniPC is not segment-safe",
    "sample_dpm_fast": "DPM fast has no sigma schedule",
    "sample_dpm_adaptive": "DPM adaptive has no sigma schedule",
}


# ------------------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------------------


def _as_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on", "enable", "enabled"):
        return True
    if text in ("false", "0", "no", "off", "disable", "disabled", ""):
        return False
    return default


def _as_float(value, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _fmt(value: float) -> str:
    return format(float(value), ".10g")


@dataclass(frozen=True)
class SpeedSettings:
    enabled: bool = False
    mode: str = "transition"
    threshold: str = "neo_shift"
    preset: str = "anima"
    scales: tuple = (0.5, 1.0)
    delta: float = 0.01
    sigma_divisor: float = 1.03
    manual_sigmas: tuple = (0.7,)
    adaptive: bool = True
    transform: str = "dct"
    spectrum_A: float = schedule.CUSTOM_DEFAULT_A
    spectrum_beta: float = schedule.CUSTOM_DEFAULT_BETA
    seed: int = -1
    hires: bool = False
    error: str | None = None
    xyz: tuple = ()

    def summary(self) -> str:
        """The ``Anima SPEED`` infotext value (``key=value`` pairs, restored by paste)."""
        parts = [
            f"mode={self.mode}",
            f"threshold={self.threshold}",
            f"preset={self.preset}",
            "scales=" + ",".join(_fmt(v) for v in self.scales),
            f"delta={_fmt(self.delta)}",
            f"divisor={_fmt(self.sigma_divisor)}",
            "manual=" + ",".join(_fmt(v) for v in self.manual_sigmas),
            f"adaptive={self.adaptive}",
            f"transform={self.transform}",
        ]
        if self.preset == "custom":
            parts += [f"A={_fmt(self.spectrum_A)}", f"beta={_fmt(self.spectrum_beta)}"]
        parts += [f"seed={self.seed}", f"hires={self.hires}"]
        return "; ".join(parts)


def parse_summary(text) -> dict[str, str]:
    """``Anima SPEED`` infotext value -> ``{key: value}`` (unknown keys kept)."""
    out: dict[str, str] = {}
    for part in str(text or "").split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def _choice(value, choices, default, name, errors):
    text = str(value if value is not None else default).strip().lower()
    if text in choices:
        return text
    errors.append(f"{name} must be one of {', '.join(choices)} (got {value!r})")
    return default


# Named API arguments: the positional names plus the infotext keys.
ARG_ALIASES = {"divisor": "sigma_divisor", "manual": "manual_sigmas", "A": "spectrum_A", "beta": "spectrum_beta"}


def coerce_settings(args, xyz: dict | None = None) -> SpeedSettings:
    """Script args (+ XYZ overrides) -> :class:`SpeedSettings` (``error`` if invalid).

    ``args`` is the positional list (``ARG_NAMES`` order; missing tail = defaults) or, like other
    sam-extra scripts, one dict of named values as the first argument (``ARG_NAMES`` or the
    ``Anima SPEED`` infotext keys; lists are accepted for ``scales``/``manual_sigmas``).

    XYZ fields: ``enabled``, ``mode``, ``manual`` (also selects the manual threshold),
    ``delta`` (a manual threshold becomes ``neo_shift``), ``divisor`` (selects ``neo_shift``),
    ``scale`` (a two-stage ladder ``scale,1.0``).
    """
    args = list(args or ())
    named = None
    if args and isinstance(args[0], dict):
        named = {ARG_ALIASES.get(str(key), str(key)): value for key, value in args[0].items()}

    def arg(name):
        if named is not None:
            return named.get(name, DEFAULTS[name])
        index = ARG_NAMES.index(name)
        return args[index] if index < len(args) else DEFAULTS[name]

    xyz = dict(xyz or {})
    errors: list[str] = []
    enabled = _as_bool(arg("enabled"), False)
    if "enabled" in xyz:
        enabled = _as_bool(xyz["enabled"], enabled)
    mode = _choice(xyz.get("mode", arg("mode")), schedule.MODES, "transition", "mode", errors)
    threshold = _choice(arg("threshold"), schedule.THRESHOLDS, "neo_shift", "threshold", errors)
    preset = str(arg("preset") or DEFAULTS["preset"]).strip()
    if preset not in schedule.PRESETS:
        errors.append(f"unknown spectrum preset {preset!r}")
        preset = DEFAULTS["preset"]
    transform = _choice(arg("transform"), schedule.TRANSFORMS, "dct", "transform", errors)
    delta = _as_float(arg("delta"), DEFAULTS["delta"])
    divisor = _as_float(arg("sigma_divisor"), DEFAULTS["sigma_divisor"])
    manual_text = arg("manual_sigmas")
    scales_text = arg("scales")
    if "manual" in xyz:
        manual_text = xyz["manual"]
        threshold = "manual"
    if "delta" in xyz:
        delta = _as_float(xyz["delta"], delta)
        if threshold == "manual":
            threshold = "neo_shift"
    if "divisor" in xyz:
        divisor = _as_float(xyz["divisor"], divisor)
        threshold = "neo_shift"
    if "scale" in xyz:
        scale = _as_float(xyz["scale"], 0.5)
        scales_text = "1.0" if scale >= 1.0 else f"{_fmt(scale)},1.0"
    try:
        scales = schedule.parse_scales(scales_text)
    except ValueError as exc:
        errors.append(str(exc))
        scales = (0.5, 1.0)
    try:
        manual = schedule.parse_sigmas(manual_text, strictly_decreasing=(mode == "transition"))
    except ValueError as exc:
        manual = (0.7,)
        if threshold == "manual":
            errors.append(str(exc))
    if threshold != "manual":
        if not (0.0 < delta < 1.0):
            errors.append(f"delta must be in (0, 1); got {delta}")
        if threshold == "neo_shift" and not (divisor > 0.0):
            errors.append(f"sigma divisor must be positive; got {divisor}")
    seed_value = _as_float(arg("seed"), -1.0)
    seed = int(seed_value) if seed_value >= 0 else -1
    return SpeedSettings(
        enabled=enabled, mode=mode, threshold=threshold, preset=preset, scales=tuple(scales),
        delta=delta, sigma_divisor=divisor, manual_sigmas=tuple(manual),
        adaptive=_as_bool(arg("adaptive"), True), transform=transform,
        spectrum_A=_as_float(arg("spectrum_A"), DEFAULTS["spectrum_A"]),
        spectrum_beta=_as_float(arg("spectrum_beta"), DEFAULTS["spectrum_beta"]),
        seed=seed, hires=_as_bool(arg("hires"), False),
        error="; ".join(errors) or None, xyz=tuple(sorted(xyz)),
    )


# ------------------------------------------------------------------------------------------------
# Per-generation status (on p)
# ------------------------------------------------------------------------------------------------


@dataclass
class _GenerationState:
    statuses: dict = field(default_factory=dict)


def pass_label(p) -> str:
    if getattr(p, "is_hr_pass", False):
        return "hires"
    if any(cls.__name__ == "StableDiffusionProcessingImg2Img" for cls in type(p).__mro__):
        return "img2img"
    return "base"


def begin_pass(p, label: str) -> _GenerationState:
    """Start the status of a pass. A first pass (txt2img base, img2img) starts a new state, so a
    later batch iteration or an XYZ cell never inherits the previous one; hires adds to it.

    A first pass also drops the ``Anima SPEED img2img rescale`` value: Forge's XYZ grid hands its
    cells ``copy(p)``, which share one ``extra_generation_params`` dict, so a value written by an
    earlier cell's img2img/hires run would otherwise reach a cell whose passes ran no coarse step."""
    state = getattr(p, _STATE_ATTR, None)
    first = label in ("base", "img2img")
    if first or not isinstance(state, _GenerationState):
        state = _GenerationState()
        try:
            setattr(p, _STATE_ATTR, state)
        except Exception:
            pass
    if first:
        params = getattr(p, "extra_generation_params", None)
        if isinstance(params, dict):
            params.pop(INFOTEXT_IMG2IMG_RESCALE, None)
    state.statuses.pop(label, None)
    _write_status(p, state)
    return state


def _write_status(p, state: _GenerationState) -> None:
    params = getattr(p, "extra_generation_params", None)
    if not isinstance(params, dict):
        return
    if state.statuses:
        params[STATUS_KEY] = " | ".join(f"{name}: {value}" for name, value in state.statuses.items())
    else:
        params.pop(STATUS_KEY, None)


def record_status(p, label: str, text: str) -> None:
    """Set the status of ``label`` and rewrite the ``Anima SPEED status`` infotext value."""
    state = getattr(p, _STATE_ATTR, None)
    if not isinstance(state, _GenerationState):
        state = _GenerationState()
        try:
            setattr(p, _STATE_ATTR, state)
        except Exception:
            pass
    state.statuses[label] = text
    _write_status(p, state)


# ------------------------------------------------------------------------------------------------
# Guards
# ------------------------------------------------------------------------------------------------


def _unet(p):
    return getattr(getattr(getattr(p, "sd_model", None), "forge_objects", None), "unet", None)


def _loaded(name: str):
    """A Forge module if it is already imported, else None (never imports it)."""
    return sys.modules.get(name)


def _dynamic_args():
    return getattr(_loaded("backend.args"), "dynamic_args", None)


def _is_anima(p) -> bool:
    model = getattr(p, "sd_model", None)
    if model is not None and any("Anima" in cls.__name__ for cls in type(model).__mro__):
        return True
    return bool(getattr(_dynamic_args(), "anima", False))


def _predictor(p, model):
    inner = getattr(model, "inner_model", None) if model is not None else None
    predictor = getattr(inner, "predictor", None)
    if predictor is None:
        unet = _unet(p)
        predictor = getattr(getattr(unet, "model", None), "predictor", None)
    return predictor


def _diffusion_model(unet):
    return getattr(getattr(unet, "model", None), "diffusion_model", None)


def _conditioning_latents(p, dyn, unet, x) -> str | None:
    """Full-size latents Forge 2.29.2 hands the DiT through ``backend.args.dynamic_args``.

    ``forge_loader`` resets them at every model load and the engines rewrite them before sampling,
    so what is there when the sampler starts is what the DiT reads at every forward:

    * ``ref_latents`` — Anima (``backend/nn/anima.py``: concatenated on the T axis, the coarse grid
      would not even fit), Flux / Flux Kontext / Flux.2 Klein (``flux.py``), Qwen-Image-Edit
      (``qwen.py``) and Nunchaku (``svdq.py``) append them to the image tokens with RoPE positions of
      their own full grid, Krea 2 (``krea.py``) resizes them to the latent: on the coarse grid image
      and reference would no longer line up;
    * ``concat_latent`` — Wan 2.2 I2V / FLF2V (``backend/nn/wan.py``): concatenated with the latent on
      the channel axis whenever it has fewer channels than the DiT's ``in_dim`` (I2V 36, T2V 16);
    * ``lq_latent`` — PiD (``backend/nn/pixeldit/pid.py``): ``lq_latent[0]``, the full-size
      low-quality input, conditions every forward (``dynamic_args.pid`` marks the model).
    """
    if dyn is None:
        return None
    if getattr(dyn, "ref_latents", None):
        if _is_anima(p):
            return (
                "Anima reference latents are present (Anima Edit / img2img referencing) - they are "
                "concatenated onto the latent at full size; turn off 'anima_do_reference' to use SPEED"
            )
        return (
            "reference latents are present (Flux Kontext / Flux.2 Klein / Qwen-Image-Edit / Krea 2 edit) - "
            "the DiT appends them to the image tokens at full size, so on the coarse grid image and "
            "reference positions would not line up"
        )
    concat = getattr(dyn, "concat_latent", None)
    if torch.is_tensor(concat):
        in_dim = getattr(_diffusion_model(unet), "in_dim", None)
        if not isinstance(in_dim, int) or int(x.shape[1]) < in_dim:
            return (
                "Wan 2.2 image-to-video conditioning (concat_latent) is on - the DiT concatenates it with "
                "the latent on the channel axis at full size"
            )
    lq = getattr(dyn, "lq_latent", None)
    lq_first = lq[0] if isinstance(lq, (list, tuple)) and len(lq) > 0 else None
    if getattr(dyn, "pid", False) is True or torch.is_tensor(lq_first):
        return (
            "PiD low-quality latent conditioning (lq_latent) is on - it is the full-size input and "
            "every forward reads it"
        )
    return None


def _spectrum_wrapper(unet) -> bool:
    options = getattr(unet, "model_options", None) or {}
    fn = options.get("model_function_wrapper") if isinstance(options, dict) else None
    if fn is None:
        return False
    qualname = str(getattr(fn, "__qualname__", "") or getattr(fn, "__name__", ""))
    module = str(getattr(fn, "__module__", "") or "")
    return "spectrum_unet_wrapper" in qualname or module.startswith("lib_spectrum")


def runtime_guard(p, model, x, sigmas) -> str | None:
    """Why SPEED must not run on this sampling call, or None. Checked when the sampler starts."""
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or len(sigmas) < 2:
        return "the sampler gives no sigma schedule"
    if not torch.is_tensor(x) or x.ndim not in (4, 5):
        return f"unsupported latent shape {tuple(getattr(x, 'shape', ()))}"
    predictor = _predictor(p, model)
    kind = getattr(predictor, "prediction_type", None)
    if kind != "const":
        return (
            f"not a flow-matching model (prediction_type={kind}) - SDXL/SD1.x eps/v models are not "
            "supported (kappa and the sigma* presets assume flow sigmas in (0, 1])"
        )
    if float(sigmas.max()) > 1.0 + 1e-4:
        return f"sigma_max {float(sigmas.max()):.3f} > 1 is not a flow schedule"
    if (
        getattr(p, "image_mask", None) is not None
        or getattr(p, "mask", None) is not None
        or getattr(model, "mask", None) is not None
        or getattr(getattr(p, "sd_model", None), "is_inpaint", False)
    ):
        return "inpainting / mask (masks and inpaint conditioning are tied to the full latent grid)"
    unet = _unet(p)
    dyn = _dynamic_args()
    reason = _conditioning_latents(p, dyn, unet, x)
    if reason is not None:
        return reason
    if getattr(unet, "controlnet_linked_list", None) is not None:
        return "ControlNet is on (its hint is tied to the full latent grid)"
    if dyn is not None and getattr(dyn, "ACTIVE_LLLITE_DIT", None):
        return "ControlNet-LLLite is on (its hint is tied to the full latent grid)"
    if _spectrum_wrapper(unet):
        return (
            "Forge Spectrum Integrated is on - SPEED steps aside (its stop-caching point comes from "
            "the step count and it can forecast a coarse-sized tensor across the size change)"
        )
    return None


# ------------------------------------------------------------------------------------------------
# Host
# ------------------------------------------------------------------------------------------------


class _ShapeAwareHijack:
    """Stands in for Forge's ``TorchHijack`` while SPEED runs (aoleg ``_ShapeSafeRandn``).

    A full-grid ``randn_like`` goes to the original (Forge's per-image ``p.rng.next()``); any other
    shape gets per-image noise of that shape from ``coarse``."""

    def __init__(self, inner, full_shape: tuple[int, ...], coarse: Callable):
        self._inner = inner
        self._full = tuple(int(v) for v in full_shape)
        self._coarse = coarse

    def __getattr__(self, item):
        if item == "randn_like":
            return self.randn_like
        return getattr(self._inner, item)

    def randn_like(self, x, *args, **kwargs):
        if tuple(int(v) for v in x.shape) == self._full:
            noise = self._inner.randn_like(x)
        else:
            noise = self._coarse(x)
        dtype = kwargs.get("dtype")
        device = kwargs.get("device")
        if dtype is not None or device is not None:
            noise = noise.to(device=device or noise.device, dtype=dtype or noise.dtype)
        return noise


class ForgeSpeedHost(SpeedHost):
    """SpeedHost for one Forge sampling call."""

    def __init__(self, p, model, sigmas: torch.Tensor, settings: SpeedSettings, log: Callable[[str], None]):
        self.p = p
        self.model = model
        self.settings = settings
        self._log = log
        self._fallback_generators: list[torch.Generator] | None = None
        self.image_seeds = self._image_seeds()
        self._options = None
        self._original = None
        self._offset = None
        unet = _unet(p)
        options = getattr(unet, "model_options", None)
        transformer = options.get("transformer_options") if isinstance(options, dict) else None
        whole = transformer.get("sampling_sigmas") if isinstance(transformer, dict) else None
        if torch.is_tensor(whole) and whole.ndim == 1 and len(whole) >= len(sigmas):
            offset = len(whole) - len(sigmas)
            try:
                same = torch.allclose(
                    whole[offset:].detach().float().cpu(), sigmas.detach().float().cpu(), rtol=0.0, atol=1e-7,
                )
            except Exception:
                same = False
            if same:
                self._options, self._original, self._offset = transformer, whole, offset

    # -- seeds ------------------------------------------------------------------------------------
    def _image_seeds(self) -> list[int]:
        seeds = getattr(self.p, "seeds", None)
        try:
            return [int(s) for s in seeds] if seeds else []
        except (TypeError, ValueError):
            return []

    def seeds(self, batch: int) -> list[int]:
        """Seeds of the spectral (expansion / init) noise: the fixed seed, else the image seeds."""
        if self.settings.seed >= 0:
            return [int(self.settings.seed)] * batch
        return self.batch_image_seeds(batch)

    def batch_image_seeds(self, batch: int) -> list[int]:
        seeds = self.image_seeds
        if len(seeds) == batch:
            return list(seeds)
        base = seeds[0] if seeds else 0
        return [base + i for i in range(batch)]

    # -- sampler noise ----------------------------------------------------------------------------
    def _coarse_randn(self, x: torch.Tensor) -> torch.Tensor:
        shape = tuple(int(v) for v in x.shape[1:])
        generators = getattr(getattr(self.p, "rng", None), "generators", None)
        forge_rng = _loaded("modules.rng")
        draw = getattr(forge_rng, "randn_without_seed", None)
        if generators is not None and len(generators) == int(x.shape[0]) and callable(draw):
            try:
                # Forge's own per-image generators and "random number source" (GPU/CPU/NV).
                noise = torch.stack([draw(shape, generator=g) for g in generators])
                return noise.to(device=x.device, dtype=x.dtype)
            except Exception as exc:  # pragma: no cover - Forge API drift
                self._log(f"per-image generators unavailable ({type(exc).__name__}); using seeded CPU noise")
        if self._fallback_generators is None or len(self._fallback_generators) != int(x.shape[0]):
            self._fallback_generators = [
                torch.Generator(device="cpu").manual_seed(seed + COARSE_FALLBACK_SEED_OFFSET)
                for seed in self.batch_image_seeds(int(x.shape[0]))
            ]
        noise = torch.stack([
            torch.randn(shape, generator=g, dtype=torch.float32) for g in self._fallback_generators
        ])
        return noise.to(device=x.device, dtype=x.dtype)

    @contextlib.contextmanager
    def noise_scope(self, full_shape):
        module = sys.modules.get("k_diffusion.sampling")
        previous = getattr(module, "torch", None) if module is not None else None
        if previous is None:
            yield
            return
        module.torch = _ShapeAwareHijack(previous, tuple(full_shape), self._coarse_randn)
        try:
            yield
        finally:
            module.torch = previous

    def coarse_noise_sampler(self, template, x, stage, sigmas):
        cls = type(template)
        if cls.__name__ != "BrownianTreeNoiseSampler":
            cls = getattr(_loaded("k_diffusion.sampling"), "BrownianTreeNoiseSampler", None)
            if cls is None:
                return None
        whole = self._original if torch.is_tensor(self._original) else sigmas
        positive = whole[whole > 0]
        if positive.numel() == 0:
            return None
        seeds = [s + (stage + 1) * BROWNIAN_STAGE_SEED_STRIDE for s in self.batch_image_seeds(int(x.shape[0]))]
        return cls(x, positive.min(), whole.max(), seed=seeds)

    # -- sampling_sigmas ----------------------------------------------------------------------------
    def publish_sigmas(self, sigmas):
        if self._options is None:
            return
        whole = self._original.clone()
        whole[self._offset:] = sigmas.to(device=whole.device, dtype=whole.dtype)
        # A new tensor: tagged as this run's list, so a stage that keys its run on the list object
        # (Colorcraft) keeps following it instead of taking the patched copy for another run.
        self._options["sampling_sigmas"] = mark_republished(whole, self._original)

    def restore_sigmas(self):
        if self._options is not None:
            self._options["sampling_sigmas"] = self._original

    def log(self, message: str) -> None:
        self._log(message)


# ------------------------------------------------------------------------------------------------
# The wrapper
# ------------------------------------------------------------------------------------------------


def _is_interrupt(exc: BaseException) -> bool:
    return type(exc).__name__ == "InterruptedException"


def _resize_current_latent(full_hw: tuple[int, int]) -> None:
    """Forge returns ``state.current_latent`` on an interrupt; keep it at the requested size."""
    state = getattr(_loaded("modules.shared"), "state", None)
    latent = getattr(state, "current_latent", None)
    try:
        if torch.is_tensor(latent) and latent.ndim >= 4 and tuple(latent.shape[-2:]) != tuple(full_hw):
            state.current_latent = spectral.amplitude_resize(latent, full_hw).to(dtype=latent.dtype)
    except Exception:
        pass


def _plan_for(settings: SpeedSettings, x: torch.Tensor, sigmas: torch.Tensor):
    A, beta, ref_latent, ref_shift = schedule.preset_params(
        settings.preset, settings.spectrum_A, settings.spectrum_beta,
    )
    return schedule.build_plan(
        mode=settings.mode, transform=settings.transform, sigmas=sigmas,
        full_grid=(int(x.shape[-2]), int(x.shape[-1])), scales=settings.scales,
        threshold=settings.threshold, delta=settings.delta, A=A, beta=beta, ref_latent=ref_latent,
        ref_shift=ref_shift, adaptive=settings.adaptive, sigma_divisor=settings.sigma_divisor,
        manual_sigmas=settings.manual_sigmas,
    )


def make_wrapper(
    base_fn: Callable,
    p,
    settings: SpeedSettings,
    label: str,
    *,
    img2img_rescale: bool = True,
    log: Callable[[str], None] = lambda message: None,
):
    """``p.sampler.func`` replacement: SPEED for the pass of ``p``, the original otherwise."""
    used = {"done": False}

    @functools.wraps(base_fn)
    def sample_speed(model, x, sigmas=None, extra_args=None, callback=None, disable=None, **kwargs):
        def passthrough():
            if sigmas is None:     # n / sigma_min / sigma_max solvers: Forge never passed a schedule
                return base_fn(model, x, extra_args=extra_args, callback=callback, disable=disable, **kwargs)
            return base_fn(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, **kwargs)

        owner = getattr(model, "p", None)
        if used["done"] or (owner is not None and owner is not p):
            return passthrough()
        used["done"] = True
        reason = runtime_guard(p, model, x, sigmas)
        if reason is not None:
            record_status(p, label, f"skipped - {reason}")
            log(f"{label}: skipped - {reason}")
            return passthrough()
        try:
            plan = _plan_for(settings, x, sigmas)
        except PlanError as exc:
            prefix = "invalid settings" if exc.kind == "invalid" else "skipped"
            record_status(p, label, f"{prefix} - {exc.reason}")
            log(f"{label}: {prefix} - {exc.reason}")
            return passthrough()
        host = ForgeSpeedHost(p, model, sigmas, settings, log)
        init_latent = label in ("hires", "img2img")
        run = SpeedRun(
            plan=plan, seeds=host.seeds(int(x.shape[0])), init_latent=init_latent,
            img2img_rescale=bool(img2img_rescale),
        )
        for note in plan.notes:
            log(f"{label}: {note}")
        log(
            f"{label}: {settings.mode} plan - "
            + ", ".join(
                # the sigmas the run uses at each hand-off (respace: after the earlier re-spacings)
                f"step {t.step}: {t.scale_from:g}->{t.scale_to:g} {t.grid_from[0]}x{t.grid_from[1]}"
                f"->{t.grid_to[0]}x{t.grid_to[1]} (sigma {t.sigma:.4f}->{t.aligned:.4f})"
                for t in plan.transitions
            )
            + f"; {plan.coarse_steps}/{plan.n_steps} steps coarse"
        )
        try:
            out, report = run_speed(
                base_fn, model, x, sigmas, run=run, host=host, extra_args=extra_args,
                callback=callback, disable=disable, sampler_kwargs=kwargs,
            )
        except BaseException as exc:
            if _is_interrupt(exc):
                _resize_current_latent(plan.full_grid)
                record_status(p, label, "interrupted")
                raise
            if not isinstance(exc, Exception):
                raise
            log(f"{label}: SPEED failed, sampling again without it:\n{traceback.format_exc()}")
            record_status(p, label, f"fell back to plain sampling after {type(exc).__name__}: {exc}")
            return passthrough()
        record_status(p, label, f"applied ({settings.mode}) - {report.summary(plan)}")
        if init_latent and plan.first_grid != plan.full_grid and float(sigmas[0]) < 1.0 - 1e-6:
            params = getattr(p, "extra_generation_params", None)
            if isinstance(params, dict):
                params[INFOTEXT_IMG2IMG_RESCALE] = str(bool(img2img_rescale))
        log(f"{label}: done - {report.summary(plan)}")
        return out

    sample_speed._sam3_speed_wrapped = True
    sample_speed._sam3_speed_base = base_fn
    sample_speed._sam3_speed_owner = OWNER
    return sample_speed


def attach(p, settings: SpeedSettings, label: str, *, img2img_rescale: bool = True, log=lambda m: None) -> str | None:
    """Wrap ``p.sampler.func`` for this pass. Returns a skip reason, or None when attached."""
    sampler = getattr(p, "sampler", None)
    func = getattr(sampler, "func", None)
    if sampler is None or not callable(func):
        return "no k-diffusion sampler function on this pass"
    if not any(cls.__name__ == "KDiffusionSampler" for cls in type(sampler).__mro__):
        # Forge's CompVis timestep samplers (DDIM/PLMS) take timesteps, not a sigma schedule.
        return f"{type(sampler).__name__} is not a k-diffusion sampler (DDIM/PLMS-style samplers are not supported)"
    base = getattr(func, "_sam3_speed_base", None)
    if base is not None:
        func = base   # a stale wrapper (same sampler object re-used) is replaced, never stacked
        sampler.func = base
    reason = UNSUPPORTED_SAMPLERS.get(getattr(func, "__name__", ""))
    if reason is not None:
        return reason
    sampler.func = make_wrapper(func, p, settings, label, img2img_rescale=img2img_rescale, log=log)
    return None
