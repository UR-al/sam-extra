"""Register the extra schedulers in Forge 2.29.2's ``modules.sd_schedulers``.

Forge keeps three views of its schedulers (modules/sd_schedulers.py:268-289):

* ``all_schedulers`` — every scheduler (the choices of Settings → "Hide Schedulers");
* ``schedulers`` — those not hidden by that setting (the Schedule type / Hires schedule type
  dropdowns, the XYZ "Schedule type" axis, ``/sdapi/v1/schedulers``);
* ``schedulers_map`` — name and label → scheduler, what ``KDiffusionSampler.get_sigmas`` and the
  infotext parser look up.

``register_schedulers`` appends a ``Scheduler`` to ``all_schedulers`` and, unless its label is
hidden, to ``schedulers``, then rebuilds ``schedulers_map`` with Forge's own formula (names first,
labels over them) — updated in place, so a module that kept a reference to the dict sees the new
entries and keys another extension put there stay. ``sd_samplers.get_sampler_and_scheduler`` is
``functools.cache``-d; its cache is cleared so a lookup made before the registration does not keep
returning the fallback scheduler. ``sd_samplers_kdiffusion.k_diffusion_scheduler`` (name →
function, built from ``schedulers`` at import) gets the visible entries too.

A label or name that already exists — Forge's own, another extension's (for example
aoleg/Neo_ExtraSchedulers installed next to this one) — is left alone and logged once; ours from an
earlier import (Reload UI re-runs the script file, while this module stays imported) is
recognised by the owner tag on the function and not added twice. Names, labels and aliases are
compared case-insensitively.

Aliases: the labels aoleg/Neo_ExtraSchedulers' README writes for these schedulers, lowercased
(``README_LABELS``), so "Schedule type: cosine-exponential blend" in that extension's images resolves
to ours. Forge reads ``Scheduler.aliases`` only for legacy "Sampler: <sampler> <scheduler>" infotext;
the infotext lookup (``sd_samplers.get_sampler_and_scheduler``) and ``get_sigmas`` look names up in
``schedulers_map``, so each alias is also put there. An alias equal to the scheduler's own name or label
adds nothing and is left out; one taken by someone else is skipped and logged once, like a name.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable

from . import schedulers

OWNER = "sam-extra/extra-schedulers"
# The script's title (sam3ext/ui_extra_schedulers.TITLE), so the console tells this extension's lines
# from aoleg/Neo_ExtraSchedulers' when both are installed.
LOG_PREFIX = "[Extra Schedulers (sam-extra)]"


@dataclass(frozen=True)
class SchedulerSpec:
    name: str
    label: str
    function: Callable
    default_rho: float = -1.0
    need_inner_model: bool = False
    aliases: tuple[str, ...] = ()


# The six labels the README of aoleg/Neo_ExtraSchedulers (ca55a59b, "Adds six new schedulers") writes
# — "cosine", "cosine-exponential blend", "phi", "Laplace", "Karras Dynamic", "custom" — lowercased, by
# our name. Only the README was read; the extension's code (no license) was not, so these are the
# documented names, not necessarily the strings its code writes to infotext.
README_LABELS: dict[str, str] = {
    "cosine": "cosine",
    "cosine_exponential": "cosine-exponential blend",
    "phi": "phi",
    "laplace": "laplace",
    "karras_dynamic": "karras dynamic",
    "custom": "custom",
}

# reForge's label (Panchovix/stable-diffusion-webui-reForge 25a5fb38f), so "Schedule type: React Cosinusoidal
# DynSF" in reForge images pastes here; the name is the one the desktop app is meant to compile to for ComfyUI
# once its pack has a handler for it.
LABEL_REACT_DYNSF = "React Cosinusoidal DynSF"
# The name "flow_cosmos_rho7" is the schedule option of KeithZ117/Comfyui-anima-sampler (effba3c5), and the name the
# desktop app is meant to compile to for ComfyUI once its pack has a handler for it. It needs Forge's model
# (need_inner_model) to tell flow from eps/v.
LABEL_FLOW_COSMOS_RHO7 = "Flow Cosmos rho7"
# This extension's own: Karras Dynamic's per-step rho on Flow Cosmos rho7's sigma~ range (flow models), Karras Dynamic
# itself on eps/v ones — with Flow Cosmos rho7's three accordion values. Needs Forge's model for the same reason.
LABEL_FLOW_COSMOS_DYNAMIC = "Flow Cosmos Dynamic"

# The labels are the ones specified for infotext compatibility ("Schedule type: <label>") with
# aoleg/Neo_ExtraSchedulers; that extension has no license and its code was not read, so the match
# rests on the specification, not on its source. Names are this extension's own (API/lookup keys).
# React Cosinusoidal DynSF, Flow Cosmos rho7 and Flow Cosmos Dynamic (appended after the six) are not aoleg's: no README
# alias. Forge shows them in family order anyway (sam3ext/list_order.py).
SCHEDULER_SPECS: tuple[SchedulerSpec, ...] = (
    SchedulerSpec("cosine", "Cosine", schedulers.cosine, aliases=(README_LABELS["cosine"],)),
    SchedulerSpec("cosine_exponential", "CosineExponential blend", schedulers.cosine_exponential_blend,
                  aliases=(README_LABELS["cosine_exponential"],)),
    SchedulerSpec("phi", "Phi", schedulers.phi, aliases=(README_LABELS["phi"],)),
    SchedulerSpec("laplace", "Laplace", schedulers.laplace, aliases=(README_LABELS["laplace"],)),
    SchedulerSpec("karras_dynamic", "Karras Dynamic", schedulers.karras_dynamic,
                  default_rho=schedulers.KARRAS_DYNAMIC_RHO, aliases=(README_LABELS["karras_dynamic"],)),
    SchedulerSpec("custom", "custom", schedulers.custom, aliases=(README_LABELS["custom"],)),
    SchedulerSpec("react_cosinusoidal_dynsf", LABEL_REACT_DYNSF, schedulers.react_cosinusoidal_dynsf),
    SchedulerSpec("flow_cosmos_rho7", LABEL_FLOW_COSMOS_RHO7, schedulers.flow_cosmos_rho7, need_inner_model=True),
    SchedulerSpec("flow_cosmos_dynamic", LABEL_FLOW_COSMOS_DYNAMIC, schedulers.flow_cosmos_dynamic,
                  need_inner_model=True),
)

SCHEDULER_LABELS = tuple(spec.label for spec in SCHEDULER_SPECS)
LABEL_LAPLACE = "Laplace"
LABEL_CUSTOM = "custom"
# The schedulers that read the accordion's Flow Cosmos rho / sigma max / sigma min.
FLOW_COSMOS_LABELS = (LABEL_FLOW_COSMOS_RHO7, LABEL_FLOW_COSMOS_DYNAMIC)

for _spec in SCHEDULER_SPECS:
    _spec.function._sam_extra_owner = OWNER


@dataclass
class RegistrationReport:
    added: list[str] = field(default_factory=list)
    hidden: list[str] = field(default_factory=list)      # added, but hidden by Settings → Hide Schedulers
    already: list[str] = field(default_factory=list)     # ours from an earlier import
    skipped: list[str] = field(default_factory=list)     # name or label taken by someone else
    skipped_aliases: list[str] = field(default_factory=list)   # alias taken by someone else (scheduler still added)
    error: str | None = None


_LOGGED: set[str] = set()


def _log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", file=sys.stderr)


def _log_once(key: str, message: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        _log(message)


def is_ours(scheduler) -> bool:
    return getattr(getattr(scheduler, "function", None), "_sam_extra_owner", None) == OWNER


def _keys(scheduler) -> list[str]:
    keys = [getattr(scheduler, "name", None), getattr(scheduler, "label", None), *(getattr(scheduler, "aliases", None) or [])]
    return [str(key).lower() for key in keys if key]


def _hidden_labels() -> set[str]:
    try:
        from modules import shared

        return set(getattr(shared.opts, "hide_schedulers", None) or [])
    except Exception:
        return set()


def register_schedulers(sd_schedulers=None, *, sd_samplers=None, sd_samplers_kdiffusion=None,
                        hidden_labels=None) -> RegistrationReport:
    """Add ``SCHEDULER_SPECS`` to Forge. The module arguments exist for tests; in Forge they are
    imported from ``modules``. Never raises — a failure is logged and returned in the report."""
    report = RegistrationReport()
    try:
        if sd_schedulers is None:
            from modules import sd_schedulers
        if hidden_labels is None:
            hidden_labels = _hidden_labels()
        hidden_labels = set(hidden_labels)

        scheduler_cls = sd_schedulers.Scheduler
        all_schedulers = sd_schedulers.all_schedulers
        visible = sd_schedulers.schedulers

        schedulers_map = sd_schedulers.schedulers_map
        taken: dict[str, object] = {}
        for existing in [*all_schedulers, *visible]:
            for key in _keys(existing):
                taken.setdefault(key, existing)

        for spec in SCHEDULER_SPECS:
            owners = [taken[key] for key in (spec.name.lower(), spec.label.lower()) if key in taken]
            if owners:
                if all(is_ours(owner) for owner in owners):
                    report.already.append(spec.label)
                else:
                    report.skipped.append(spec.label)
                    other = owners[0]
                    _log_once(
                        f"skip:{spec.label}",
                        f"'{spec.label}' is not added: Schedule type name/label "
                        f"'{getattr(other, 'label', other)}' ({getattr(other, 'name', '?')}) already exists "
                        "(Forge or another extension). The existing scheduler is used for that label.",
                    )
                continue
            aliases = _free_aliases(spec, taken, schedulers_map, report)
            scheduler = scheduler_cls(
                spec.name, spec.label, spec.function,
                default_rho=spec.default_rho, need_inner_model=spec.need_inner_model, aliases=aliases or None,
            )
            all_schedulers.append(scheduler)
            if spec.label in hidden_labels:
                report.hidden.append(spec.label)
            elif visible is not all_schedulers:
                visible.append(scheduler)
            for key in _keys(scheduler):
                taken.setdefault(key, scheduler)
            report.added.append(spec.label)

        # Forge's own formula (sd_schedulers.py:289), written into the existing dict, plus our aliases
        # (Forge's formula has none; setdefault keeps a key someone else already put in the map).
        rebuilt = {**{x.name: x for x in visible}, **{x.label: x for x in visible}}
        schedulers_map.update(rebuilt)
        for scheduler in visible:
            if is_ours(scheduler):
                for alias in getattr(scheduler, "aliases", None) or ():
                    schedulers_map.setdefault(alias, scheduler)

        if report.added:
            _clear_sampler_lookup_cache(sd_samplers)
            _mirror_kdiffusion_map(sd_samplers_kdiffusion, visible)
    except Exception as exc:  # never break Forge's startup
        report.error = f"{type(exc).__name__}: {exc}"
        _log_once(f"error:{report.error}", f"schedulers not registered: {report.error}")
    return report


def _free_aliases(spec: SchedulerSpec, taken: dict, schedulers_map: dict, report: RegistrationReport) -> list[str]:
    """The spec's aliases that are new keys: not its own name or label, and not taken (case-insensitively,
    in the lists; exactly, in the map) by a scheduler that is not ours — those are skipped and logged once."""
    free = []
    for alias in spec.aliases:
        if alias in (spec.name, spec.label) or alias in free:
            continue
        other = taken.get(alias.lower())
        if other is None:
            other = schedulers_map.get(alias)
        if other is not None and not is_ours(other):
            report.skipped_aliases.append(alias)
            _log_once(
                f"alias:{alias}",
                f"alias '{alias}' of '{spec.label}' is not added: Schedule type '{getattr(other, 'label', other)}' "
                f"({getattr(other, 'name', '?')}) already uses it (Forge or another extension).",
            )
            continue
        free.append(alias)
    return free


def _clear_sampler_lookup_cache(sd_samplers) -> None:
    try:
        if sd_samplers is None:
            from modules import sd_samplers
        clear = getattr(getattr(sd_samplers, "get_sampler_and_scheduler", None), "cache_clear", None)
        if callable(clear):
            clear()
    except Exception as exc:
        _log_once("cache", f"could not clear Forge's sampler/scheduler lookup cache: {type(exc).__name__}: {exc}")


def _mirror_kdiffusion_map(sd_samplers_kdiffusion, visible) -> None:
    try:
        if sd_samplers_kdiffusion is None:
            from modules import sd_samplers_kdiffusion
        mapping = getattr(sd_samplers_kdiffusion, "k_diffusion_scheduler", None)
        if isinstance(mapping, dict):
            for scheduler in visible:
                if is_ours(scheduler):
                    mapping.setdefault(scheduler.name, scheduler.function)
    except Exception:
        pass


def summary(report: RegistrationReport) -> str | None:
    """One console line for the startup log (None when there is nothing new to say)."""
    parts = []
    if report.added:
        shown = [label for label in report.added if label not in report.hidden]
        if shown:
            parts.append("added to Schedule type: " + ", ".join(shown))
        if report.hidden:
            parts.append("hidden by Settings → Hide Schedulers: " + ", ".join(report.hidden))
    if report.skipped:
        parts.append("skipped (name taken): " + ", ".join(report.skipped))
    if report.skipped_aliases:
        parts.append("aliases skipped (taken): " + ", ".join(report.skipped_aliases))
    if report.error:
        parts.append("error: " + report.error)
    return " · ".join(parts) or None
