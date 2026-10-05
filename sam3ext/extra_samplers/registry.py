"""The sampler table and its registration in Forge's sampler list.

``register()`` adds one ``SamplerData`` per entry through ``modules.sd_samplers.add_sampler`` (Forge
Neo 2.29.2; older builds without it get the same three steps by hand). Each constructor builds an
``ExtraKDiffusionSampler`` — Forge's ``KDiffusionSampler`` with three differences:

* ``extra_params`` is set from the table. ``KDiffusionSampler.__init__`` looks it up as
  ``sampler_extra_params.get(funcname, [])``, a dict keyed by function *names*
  (``"sample_euler"`` …); a function object never matches, so without this Forge would pass no
  ``s_churn``/``s_noise``/… at all and write no ``Sigma …`` infotext.
* ``initialize`` adds what the table's runtime hook supplies: the ER SDE stage, η and noise window
  from the request (``params.resolve``; their infotext is written here, so a hires pass with another
  sampler does not claim them — a window the model cannot place, because its predictor has no
  ``percent_to_sigma``, is dropped with an ``Extra Samplers status`` part and one log line), and for
  the Dy samplers the ``after_substep`` hook that puts the step's
  full-resolution x0 back into Forge's preview/interrupt slot (``sd_samplers_common.store_latent``)
  — or ``substeps=False`` when the request has machinery the sub-steps cannot follow
  (``substep_guard``: Spectrum Integrated, Wan I2V, PiD, ``extra_concat_condition``; logged once per
  reason, infotext ``Extra Samplers status``).
* CFG++ entries log Forge's own "CFG between 1.0 ~ 2.0 is recommended" warning (as
  ``modules_forge/forge_alter_samplers.AlterSampler``) when ``p.cfg_scale > 2``.

A label that is already in Forge's list and was not registered by this module (another extension,
e.g. aoleg/Neo_ExtraSchedulers which uses the same labels) is left alone and logged once. Our own
entries — the script registers again when Forge reloads the scripts (Reload UI) — are replaced in
place instead of being mistaken for another extension's. An entry whose Forge function is missing
(``requires``: ``k_diffusion.sampling`` attributes; ``forge_requires``: ``"module:attribute"`` or
``"module:attribute(parameter, …)"`` resolved with ``importlib``) is not registered and logged once.

The table, by group (labels are module-level literals here and in ``params``):

* ER SDE — ``ER SDE (Reverse-time)``, ``ER SDE (ODE)``, ``ER SDE (Tunable)`` (``er_sde``, ``er_sde_window``);
* Dy — ``Euler Dy CFG++``, ``Euler SMEA Dy CFG++``, ``Euler Dy``, ``Euler SMEA Dy`` (``euler_dy``);
* DPM++ in λ — ``DPM++ 4M SDE`` (``dpmpp_4m_sde``), ``DPM++ 2M SDE Heun``, ``DPM++ 2M (flow ODE)``,
  ``DPM++ 2M Heun (flow ODE)``, ``DPM++ 3M (flow ODE)`` (``dpmpp_flow``);
* multistep ODE — ``UniPC bh2`` (``unipc``), ``IPNDM``, ``IPNDM_V``, ``DEIS`` (``ipndm_deis``);
* CFG++ — ``CFG++ UD10 AB`` (``cfgpp_ud10_ab``);
* Restart — ``Restart (flow)`` (``restart_flow``).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
from dataclasses import dataclass, field
from typing import Callable

from . import params as sampler_params
from . import substep_guard
from .cfgpp_ud10_ab import sample_cfgpp_ud10_ab
from .common import k_sampling
from .dpmpp_4m_sde import sample_dpmpp_4m_sde
from .dpmpp_flow import (
    sample_dpmpp_2m_flow_ode,
    sample_dpmpp_2m_heun_flow_ode,
    sample_dpmpp_2m_sde_heun,
    sample_dpmpp_3m_flow_ode,
)
from .er_sde import sample_er_sde_ode, sample_er_sde_reverse_time, sample_er_sde_tunable
from .euler_dy import sample_euler_dy, sample_euler_dy_cfg_pp, sample_euler_smea_dy, sample_euler_smea_dy_cfg_pp
from .ipndm_deis import sample_deis, sample_ipndm, sample_ipndm_v
from .restart_flow import sample_restart_flow
from .unipc import sample_unipc_bh2

__all__ = [
    "LABEL_CFGPP_UD10_AB",
    "LABEL_DPMPP_2M_FLOW_ODE",
    "LABEL_DPMPP_2M_HEUN_FLOW_ODE",
    "LABEL_DPMPP_2M_SDE_HEUN",
    "LABEL_DPMPP_3M_FLOW_ODE",
    "LABEL_DPMPP_4M_SDE",
    "LABEL_DEIS",
    "LABEL_ER_SDE_ODE",
    "LABEL_ER_SDE_REVERSE_TIME",
    "LABEL_ER_SDE_TUNABLE",
    "LABEL_EULER_DY",
    "LABEL_EULER_DY_CFG_PP",
    "LABEL_EULER_SMEA_DY",
    "LABEL_EULER_SMEA_DY_CFG_PP",
    "LABEL_IPNDM",
    "LABEL_IPNDM_V",
    "LABEL_RESTART_FLOW",
    "LABEL_UNIPC_BH2",
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
LABEL_ER_SDE_TUNABLE = sampler_params.LABEL_ER_SDE_TUNABLE
LABEL_DPMPP_4M_SDE = "DPM++ 4M SDE"
LABEL_EULER_DY_CFG_PP = "Euler Dy CFG++"
LABEL_EULER_SMEA_DY_CFG_PP = "Euler SMEA Dy CFG++"
LABEL_EULER_DY = "Euler Dy"
LABEL_EULER_SMEA_DY = "Euler SMEA Dy"
LABEL_DPMPP_2M_SDE_HEUN = "DPM++ 2M SDE Heun"
LABEL_DPMPP_2M_FLOW_ODE = "DPM++ 2M (flow ODE)"
LABEL_DPMPP_2M_HEUN_FLOW_ODE = "DPM++ 2M Heun (flow ODE)"
LABEL_DPMPP_3M_FLOW_ODE = "DPM++ 3M (flow ODE)"
LABEL_UNIPC_BH2 = "UniPC bh2"
LABEL_CFGPP_UD10_AB = "CFG++ UD10 AB"
LABEL_IPNDM = "IPNDM"
LABEL_IPNDM_V = "IPNDM_V"
LABEL_DEIS = "DEIS"
LABEL_RESTART_FLOW = "Restart (flow)"

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
    forge_requires: tuple = ()                    # "module:attribute" / "module:attribute(param, …)" it calls


_ER_SDE_REQUIRES = ("sample_er_sde",)
_DY_REQUIRES = (
    "default_noise_sampler", "sigma_to_half_log_snr", "to_d", "get_ancestral_step",
    "set_model_options_post_cfg_function", "trange",
)
_4M_REQUIRES = (
    "BrownianTreeNoiseSampler", "sigma_to_half_log_snr", "offset_first_sigma_for_snr", "trange",
)
# Forge's UniPC (modules/sd_samplers_extra.py) with its ``variant`` argument.
_UNIPC_FORGE_REQUIRES = ("modules.sd_samplers_extra:sample_unipc(variant)",)
_UD10_REQUIRES = (
    "sigma_to_half_log_snr", "to_d", "linear_multistep_coeff", "set_model_options_post_cfg_function", "trange",
)
# Forge's vendored zju-pi DEIS coefficients (modules_forge/packages/k_diffusion/deis.py, unused by Forge itself).
_DEIS_FORGE_REQUIRES = ("k_diffusion.deis:get_deis_coeff_list",)

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
    # ComfyUI's "ER-SDE" scaler with η (= Forge's built-in "ER SDE" at η 1, window off).
    SamplerSpec(
        LABEL_ER_SDE_TUNABLE, sample_er_sde_tunable, ("er_sde_tunable",),
        options={}, extra_params=("s_noise",), requires=_ER_SDE_REQUIRES, kind="er_sde",
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
    # The same core without CFG++ (plain Euler steps, the ``min`` churn rule) under Koishi-Star's own
    # labels; reForge's ``k_*`` names as extra aliases.
    SamplerSpec(
        LABEL_EULER_DY, sample_euler_dy, ("euler_dy", "k_euler_dy"),
        options={}, extra_params=_KARRAS_CHURN, requires=_DY_REQUIRES, kind="dy", cfg_pp=False,
    ),
    SamplerSpec(
        LABEL_EULER_SMEA_DY, sample_euler_smea_dy, ("euler_smea_dy", "k_euler_smea_dy"),
        options={}, extra_params=_KARRAS_CHURN, requires=_DY_REQUIRES, kind="dy", cfg_pp=False,
    ),
    # Forge's own DPM-Solver++ functions (dpmpp_flow). No scheduler hint on any of them: Automatic then
    # takes the model's schedule (Normal on flow models), never Forge's "exponential" for its 2M/3M SDE.
    # The Heun SDE entry has Forge's 2M SDE options; the 3M ODE keeps Forge's 3M discard flag, so it is
    # Forge's "DPM++ 3M SDE" at Eta 0.
    SamplerSpec(
        LABEL_DPMPP_2M_SDE_HEUN, sample_dpmpp_2m_sde_heun, ("dpmpp_2m_sde_heun", "k_dpmpp_2m_sde_heun"),
        options={"brownian_noise": True}, extra_params=("eta", "s_noise"), requires=("sample_dpmpp_2m_sde",),
    ),
    SamplerSpec(
        LABEL_DPMPP_2M_FLOW_ODE, sample_dpmpp_2m_flow_ode, ("dpmpp_2m_flow_ode",),
        options={}, extra_params=(), requires=("sample_dpmpp_2m_sde",),
    ),
    SamplerSpec(
        LABEL_DPMPP_2M_HEUN_FLOW_ODE, sample_dpmpp_2m_heun_flow_ode, ("dpmpp_2m_heun_flow_ode",),
        options={}, extra_params=(), requires=("sample_dpmpp_2m_sde",),
    ),
    SamplerSpec(
        LABEL_DPMPP_3M_FLOW_ODE, sample_dpmpp_3m_flow_ode, ("dpmpp_3m_flow_ode",),
        options={"discard_next_to_last_sigma": True}, extra_params=(), requires=("sample_dpmpp_3m_sde",),
    ),
    # Forge's UniPC with variant "bh2", with the options of Forge's "UniPC" entry.
    SamplerSpec(
        LABEL_UNIPC_BH2, sample_unipc_bh2, ("uni_pc_bh2",),
        options={"discard_next_to_last_sigma": True}, extra_params=(), forge_requires=_UNIPC_FORGE_REQUIRES,
    ),
    # ComfyUI's CFG++ history sampler (cfgpp_ud10_ab): a CFG++ entry, so Forge's warning above CFG 2.
    SamplerSpec(
        LABEL_CFGPP_UD10_AB, sample_cfgpp_ud10_ab, ("cfgpp_ud10_ab",),
        options={}, extra_params=(), requires=_UD10_REQUIRES, cfg_pp=True,
    ),
    # zju-pi's multistep ODE samplers as ComfyUI runs them (ipndm_deis; IPNDM_V with the coeff4 typo fixed);
    # ComfyUI's names as aliases.
    SamplerSpec(LABEL_IPNDM, sample_ipndm, ("ipndm",), options={}, extra_params=(), requires=("trange",)),
    SamplerSpec(LABEL_IPNDM_V, sample_ipndm_v, ("ipndm_v",), options={}, extra_params=(), requires=("trange",)),
    SamplerSpec(
        LABEL_DEIS, sample_deis, ("deis",),
        options={}, extra_params=(), requires=("trange",), forge_requires=_DEIS_FORGE_REQUIRES,
    ),
    # Restart from the paper with the flow forward kernel (restart_flow). Heun steps: Forge's "Heun" options
    # (second_order → total_steps 2n); s_noise for the restart noise. No scheduler hint (Forge's "Restart" has
    # karras, which ignores a flow model's shift).
    SamplerSpec(
        LABEL_RESTART_FLOW, sample_restart_flow, ("restart_flow",),
        options={"second_order": True}, extra_params=("s_noise",), requires=("to_d", "default_noise_sampler", "trange"),
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


def _er_sde_window(spec: SamplerSpec, p, settings, predictor) -> tuple:
    """``(window or None, skipped)`` for an ER SDE entry: the ``(start, end)`` to pass when the request asks
    for the window, the entry takes it and η > 0 (η = 0 is the ODE, which has none); ``skipped`` when it
    had to be dropped because ``predictor`` (Forge's ``model_wrap.predictor``) has no ``percent_to_sigma``
    — then the request's ``Extra Samplers status`` says so and the console once."""
    if not settings.noise_window or settings.eta <= 0:
        return None, False
    if "er_sde_window" not in inspect.signature(spec.func).parameters:
        return None, False
    if not callable(getattr(predictor, "percent_to_sigma", None)):
        substep_guard.record_status_part(p, "noise_window", sampler_params.WINDOW_SKIPPED_STATUS)
        _log_once(_log, (
            f"{spec.label}: the ER SDE noise window is skipped - this model's predictor has no percent_to_sigma; "
            f"it runs with noise on every step (infotext '{substep_guard.STATUS_KEY}')."
        ))
        return None, True
    return (settings.noise_start, settings.noise_end), False


def runtime_kwargs(spec: SamplerSpec, p, forge_eta=None, predictor=None) -> dict:
    """Keyword arguments ``ExtraKDiffusionSampler.initialize`` adds for ``spec`` (and its infotext).

    ``forge_eta`` is the request's ancestral η as Forge's ``Sampler.initialize`` resolved it
    (``self.eta``); the ER SDE infotext needs it (``params.record_infotext``). ``predictor`` is the
    sampler's ``model_wrap.predictor`` (the ER SDE noise window needs its ``percent_to_sigma``)."""
    kwargs: dict = {}
    if spec.kind == "er_sde":
        settings = sampler_params.resolve(p)
        window, skipped = _er_sde_window(spec, p, settings, predictor)
        sampler_params.record_infotext(p, spec.label, settings, forge_eta=forge_eta, window_skipped=skipped)
        kwargs["max_stage"] = settings.max_stage
        kwargs["er_sde_eta"] = settings.eta
        if window is not None:
            kwargs["er_sde_window"] = window
    elif spec.kind == "dy":
        hook = _store_latent_hook()
        if hook is not None:
            kwargs["after_substep"] = hook
        reasons = substep_guard.blocking_reasons(p)
        if reasons:
            kwargs["substeps"] = False
            substep_guard.record_status(p, reasons)
            steps = "plain CFG++ Euler steps" if spec.cfg_pp else "plain Euler steps"
            for reason in reasons:
                _log_once(_log, (
                    f"{spec.label}: sub-steps skipped - {substep_guard.explain(reason)}; it runs {steps} "
                    f"instead (infotext '{substep_guard.STATUS_KEY}')."
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
            predictor = getattr(getattr(self, "model_wrap", None), "predictor", None)
            kwargs.update(runtime_kwargs(self.sam_extra_spec, p, forge_eta=getattr(self, "eta", None),
                                         predictor=predictor))
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
    skipped_missing: dict = field(default_factory=dict)   # label -> missing k_diffusion names / forge_requires
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


def _parse_forge_requirement(requirement: str) -> tuple:
    """``"module:attribute(param, …)"`` → ``(module, attribute, (param, …))`` (the parameters are optional)."""
    module_name, _sep, rest = requirement.partition(":")
    attribute, _sep, params = rest.partition("(")
    names = tuple(name.strip() for name in params.rstrip(")").split(",") if name.strip())
    return module_name.strip(), attribute.strip(), names


def _forge_requirement_met(requirement: str) -> bool:
    """Whether Forge has ``module.attribute`` (and it takes every listed parameter)."""
    module_name, attribute, params = _parse_forge_requirement(requirement)
    try:
        target = getattr(importlib.import_module(module_name), attribute)
    except Exception:
        return False
    if not params:
        return True
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False
    return all(name in parameters for name in params)


def _describe_forge_requirement(requirement: str) -> str:
    module_name, attribute, params = _parse_forge_requirement(requirement)
    text = f"{module_name}.{attribute}"
    if params:
        text += (" with the parameters " if len(params) > 1 else " with the parameter ") + ", ".join(
            f"'{name}'" for name in params)
    return text


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
        missing = [requirement for requirement in spec.forge_requires if not _forge_requirement_met(requirement)]
        if missing:
            report.skipped_missing[spec.label] = missing
            described = "; ".join(_describe_forge_requirement(requirement) for requirement in missing)
            _log_once(log, f'"{spec.label}" not registered: this Forge has no {described}.')
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
