"""The sampler table and its registration in Forge's sampler list.

``register()`` adds one ``SamplerData`` per entry through ``modules.sd_samplers.add_sampler`` (Forge
Neo 2.29.2; older builds without it get the same three steps by hand). Each constructor builds an
``ExtraKDiffusionSampler`` — Forge's ``KDiffusionSampler`` with three differences:

* ``extra_params`` is set from the table. ``KDiffusionSampler.__init__`` looks it up as
  ``sampler_extra_params.get(funcname, [])``, a dict keyed by function *names*
  (``"sample_euler"`` …); a function object never matches, so without this Forge would pass no
  ``s_churn``/``s_noise``/… at all and write no ``Sigma …`` infotext.
* ``initialize`` adds what the table's runtime hook supplies: the ER SDE stage and η from the
  request (``params.resolve``; their infotext is written here, so a hires pass with another sampler
  does not claim them), and for the Dy samplers the ``after_substep`` hook that puts the step's
  full-resolution x0 back into Forge's preview/interrupt slot (``sd_samplers_common.store_latent``)
  — or ``substeps=False`` when the request has machinery the sub-steps cannot follow
  (``substep_guard``: Spectrum Integrated, Wan I2V, PiD, ``extra_concat_condition``; logged once per
  reason, infotext ``Extra Samplers status``).
* CFG++ entries log Forge's own "CFG between 1.0 ~ 2.0 is recommended" warning (as
  ``modules_forge/forge_alter_samplers.AlterSampler``) when ``p.cfg_scale > 2``.

A label that is already in Forge's list and was not registered by this module (another extension,
e.g. aoleg/Neo_ExtraSchedulers which uses the same labels) is left alone and logged once. Our own
entries — the script registers again when Forge reloads the scripts (Reload UI) — are replaced in
place instead of being mistaken for another extension's.
"""

from __future__ import annotations

import inspect
import logging
import sys
from dataclasses import dataclass, field
from typing import Callable

from . import params as sampler_params
from . import substep_guard
from .common import k_sampling
from .dpmpp_4m_sde import sample_dpmpp_4m_sde
from .er_sde import sample_er_sde_ode, sample_er_sde_reverse_time
from .euler_dy import sample_euler_dy_cfg_pp, sample_euler_smea_dy_cfg_pp

__all__ = [
    "LABEL_DPMPP_4M_SDE",
    "LABEL_ER_SDE_ODE",
    "LABEL_ER_SDE_REVERSE_TIME",
    "LABEL_EULER_DY_CFG_PP",
    "LABEL_EULER_SMEA_DY_CFG_PP",
    "OWNER_ATTR",
    "RegistrationReport",
    "SPECS",
    "SamplerSpec",
    "last_report",
    "register",
    "runtime_kwargs",
    "sampler_class",
]

LABEL_ER_SDE_REVERSE_TIME = sampler_params.LABEL_ER_SDE_REVERSE_TIME
LABEL_ER_SDE_ODE = sampler_params.LABEL_ER_SDE_ODE
LABEL_DPMPP_4M_SDE = "DPM++ 4M SDE"
LABEL_EULER_DY_CFG_PP = "Euler Dy CFG++"
LABEL_EULER_SMEA_DY_CFG_PP = "Euler SMEA Dy CFG++"

# Marks the SamplerData constructors made here (how a reload recognises its own entries).
OWNER_ATTR = "_sam_extra_samplers"

_CFG_PP_WARNING = "CFG between 1.0 ~ 2.0 is recommended when using CFG++ samplers"   # Forge's text
_KARRAS_CHURN = ("s_churn", "s_tmin", "s_tmax", "s_noise")   # Forge's sample_euler extra params


@dataclass(frozen=True)
class SamplerSpec:
    label: str
    func: Callable
    aliases: tuple
    options: dict = field(default_factory=dict)   # SamplerData options (scheduler, brownian_noise, …)
    extra_params: tuple = ()                      # names Forge's Sampler.initialize passes from p/opts
    requires: tuple = ()                          # k_diffusion.sampling attributes the sampler calls
    kind: str = ""                                # "er_sde" | "dy" | "" (runtime keyword arguments)
    cfg_pp: bool = False


_ER_SDE_REQUIRES = ("sample_er_sde",)
_DY_REQUIRES = (
    "default_noise_sampler", "sigma_to_half_log_snr", "to_d", "get_ancestral_step",
    "set_model_options_post_cfg_function", "trange",
)
_4M_REQUIRES = (
    "BrownianTreeNoiseSampler", "sigma_to_half_log_snr", "offset_first_sigma_for_snr", "trange",
)

SPECS: tuple = (
    # Forge's built-in "ER SDE" has no options and no extra params; s_noise is the one Forge
    # parameter the reverse-time scaler uses (the ODE forces it to 0, as ComfyUI's node does).
    SamplerSpec(
        LABEL_ER_SDE_REVERSE_TIME, sample_er_sde_reverse_time, ("er_sde_reverse_time",),
        options={}, extra_params=("s_noise",), requires=_ER_SDE_REQUIRES, kind="er_sde",
    ),
    SamplerSpec(
        LABEL_ER_SDE_ODE, sample_er_sde_ode, ("er_sde_ode",),
        options={}, extra_params=(), requires=_ER_SDE_REQUIRES, kind="er_sde",
    ),
    # Same options and parameters as Forge's "DPM++ 3M SDE"; Clybius' pack also discards the
    # penultimate sigma for this sampler (DISCARD_PENULTIMATE_SIGMA_SAMPLERS).
    SamplerSpec(
        LABEL_DPMPP_4M_SDE, sample_dpmpp_4m_sde, ("dpmpp_4m_sde",),
        options={"scheduler": "exponential", "discard_next_to_last_sigma": True, "brownian_noise": True},
        extra_params=("eta", "s_noise"), requires=_4M_REQUIRES,
    ),
    # Koishi-Star registers its samplers with {} options and the Karras churn parameters; Forge's
    # CFG++ entries copy the options of their base sampler (Euler: {}).
    SamplerSpec(
        LABEL_EULER_DY_CFG_PP, sample_euler_dy_cfg_pp, ("euler_dy_cfg_pp",),
        options={}, extra_params=_KARRAS_CHURN, requires=_DY_REQUIRES, kind="dy", cfg_pp=True,
    ),
    SamplerSpec(
        LABEL_EULER_SMEA_DY_CFG_PP, sample_euler_smea_dy_cfg_pp, ("euler_smea_dy_cfg_pp",),
        options={}, extra_params=_KARRAS_CHURN, requires=_DY_REQUIRES, kind="dy", cfg_pp=True,
    ),
)


def _log(message: str) -> None:
    print(f"[ExtraSamplers] {message}", file=sys.stderr)


_LOGGED: set = set()


def _log_once(log, message: str) -> None:
    if message in _LOGGED:
        return
    _LOGGED.add(message)
    log(message)


def _store_latent_hook():
    """``after_substep``: Forge's ``store_latent`` (the CFG denoiser stores every evaluation's x0)."""
    try:
        from modules import sd_samplers_common
    except Exception:
        return None
    store = getattr(sd_samplers_common, "store_latent", None)
    return store if callable(store) else None


def runtime_kwargs(spec: SamplerSpec, p, forge_eta=None) -> dict:
    """Keyword arguments ``ExtraKDiffusionSampler.initialize`` adds for ``spec`` (and its infotext).

    ``forge_eta`` is the request's ancestral η as Forge's ``Sampler.initialize`` resolved it
    (``self.eta``); the ER SDE infotext needs it (``params.record_infotext``)."""
    kwargs: dict = {}
    if spec.kind == "er_sde":
        settings = sampler_params.resolve(p)
        sampler_params.record_infotext(p, spec.label, settings, forge_eta=forge_eta)
        kwargs["max_stage"] = settings.max_stage
        kwargs["er_sde_eta"] = settings.eta
    elif spec.kind == "dy":
        hook = _store_latent_hook()
        if hook is not None:
            kwargs["after_substep"] = hook
        reasons = substep_guard.blocking_reasons(p)
        if reasons:
            kwargs["substeps"] = False
            substep_guard.record_status(p, reasons)
            for reason in reasons:
                _log_once(_log, (
                    f"{spec.label}: sub-steps skipped - {substep_guard.explain(reason)}; it runs plain "
                    f"CFG++ Euler steps instead (infotext '{substep_guard.STATUS_KEY}')."
                ))
    accepted = inspect.signature(spec.func).parameters
    return {name: value for name, value in kwargs.items() if name in accepted}


def _warn_cfg(spec: SamplerSpec, p) -> None:
    if not spec.cfg_pp:
        return
    try:
        cfg = float(getattr(p, "cfg_scale", 0.0) or 0.0)
    except (TypeError, ValueError):
        return
    if cfg > 2.0:
        logging.warning(_CFG_PP_WARNING)


_CLASS_CACHE: dict = {}


def sampler_class():
    """``ExtraKDiffusionSampler`` for the ``KDiffusionSampler`` currently in ``modules.sd_samplers_kdiffusion``."""
    from modules import sd_samplers_kdiffusion

    base = sd_samplers_kdiffusion.KDiffusionSampler
    cached = _CLASS_CACHE.get("class")
    if cached is not None and _CLASS_CACHE.get("base") is base:
        return cached

    class ExtraKDiffusionSampler(base):
        def __init__(self, spec: SamplerSpec, sd_model):
            super().__init__(spec.func, sd_model)
            # Forge keys sampler_extra_params by function *name*; a function object never matches.
            self.extra_params = list(spec.extra_params)
            self.sam_extra_spec = spec

        def initialize(self, p) -> dict:
            kwargs = super().initialize(p)
            # Forge's initialize has just resolved self.eta (p.eta, else the eta_ancestral setting).
            kwargs.update(runtime_kwargs(self.sam_extra_spec, p, forge_eta=getattr(self, "eta", None)))
            return kwargs

        def sample(self, p, *args, **kwargs):
            _warn_cfg(self.sam_extra_spec, p)
            return super().sample(p, *args, **kwargs)

        def sample_img2img(self, p, *args, **kwargs):
            _warn_cfg(self.sam_extra_spec, p)
            return super().sample_img2img(p, *args, **kwargs)

    ExtraKDiffusionSampler.__qualname__ = "ExtraKDiffusionSampler"
    _CLASS_CACHE["class"] = ExtraKDiffusionSampler
    _CLASS_CACHE["base"] = base
    return ExtraKDiffusionSampler


def _constructor(spec: SamplerSpec):
    def constructor(model):
        return sampler_class()(spec, model)

    setattr(constructor, OWNER_ATTR, True)
    constructor.sam_extra_spec = spec
    return constructor


@dataclass
class RegistrationReport:
    added: list = field(default_factory=list)
    replaced: list = field(default_factory=list)
    skipped_foreign: list = field(default_factory=list)
    skipped_missing: dict = field(default_factory=dict)   # label -> missing k_diffusion names
    error: str | None = None

    @property
    def active(self) -> list:
        return [*self.added, *self.replaced]


_LAST_REPORT: RegistrationReport | None = None


def last_report() -> RegistrationReport | None:
    return _LAST_REPORT


def _is_ours(data) -> bool:
    return bool(getattr(getattr(data, "constructor", None), OWNER_ATTR, False))


def _rebuild(sd_samplers) -> None:
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
    sd_samplers.set_samplers()


def _add(sd_samplers, data) -> None:
    add = getattr(sd_samplers, "add_sampler", None)
    if callable(add):
        add(data)
        return
    # Forge builds without add_sampler: the same steps (modules/sd_samplers.py add_sampler).
    if data.name not in [x.name for x in sd_samplers.all_samplers]:
        sd_samplers.all_samplers.append(data)
        _rebuild(sd_samplers)


def _replace(sd_samplers, old, data) -> None:
    samplers = sd_samplers.all_samplers
    for index, item in enumerate(samplers):
        if item is old:
            samplers[index] = data
            break
    _rebuild(sd_samplers)


def register(log=_log) -> RegistrationReport:
    """Add every ``SPECS`` entry to Forge's sampler list (see the module docstring)."""
    global _LAST_REPORT
    report = RegistrationReport()
    _LAST_REPORT = report
    try:
        from modules import sd_samplers, sd_samplers_common
    except Exception as exc:  # pragma: no cover - outside Forge
        report.error = f"{type(exc).__name__}: {exc}"
        _log_once(log, f"not registered: Forge's sampler modules are unavailable ({report.error})")
        return report
    try:
        ks = k_sampling()
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
        _log_once(log, f"not registered: k_diffusion.sampling is unavailable ({report.error})")
        return report

    existing = {data.name: data for data in sd_samplers.all_samplers}
    for spec in SPECS:
        current = existing.get(spec.label)
        if current is not None and not _is_ours(current):
            report.skipped_foreign.append(spec.label)
            _log_once(log, f'"{spec.label}" is already in the sampler list (another extension); keeping that one.')
            continue
        missing = [name for name in spec.requires if not hasattr(ks, name)]
        if missing:
            report.skipped_missing[spec.label] = missing
            _log_once(log, f'"{spec.label}" not registered: this Forge has no k_diffusion.sampling.{", ".join(missing)}.')
            continue
        data = sd_samplers_common.SamplerData(spec.label, _constructor(spec), list(spec.aliases), dict(spec.options))
        if current is not None:
            _replace(sd_samplers, current, data)
            report.replaced.append(spec.label)
        else:
            _add(sd_samplers, data)
            report.added.append(spec.label)

    cached = getattr(sd_samplers, "get_sampler_and_scheduler", None)
    cache_clear = getattr(cached, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()   # functools.cache: forget any lookup made before these names existed
    return report
