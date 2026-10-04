"""Family order for Forge's Sampling method and Schedule type lists (v0.33.0).

Forge lists samplers and schedulers in registration order: its own, then each extension's in load order
(this extension's Extra Samplers and Extra Schedulers, RES4LYF, sd_forge_neo_extra_samplers …), so the
DPM++ or Euler variants end up far apart. ``apply()`` sorts the lists by family once every script is
loaded and before Forge builds the UI (``scripts/list_order.py``, ``script_callbacks.on_before_ui``; Forge
runs that callback on every start and every Reload UI, also for ``--nowebui``). No setting, no toggle.

Only the order changes. No entry is added, removed or renamed; names, labels, aliases, options and every
lookup in Forge's maps stay as they were. The lists are reordered in place (Forge 2.29.2,
modules/sd_samplers.py and modules/sd_schedulers.py):

* ``sd_samplers.all_samplers`` (and ``samplers`` / ``samplers_for_img2img`` when they are other lists) —
  the txt2img/img2img Sampling method dropdowns (modules/processing_scripts/sampler.py), the Hires sampling
  method dropdown (modules/ui.py), the XYZ "Sampler" / "Hires sampler" choices (scripts/xyz_grid.py lambdas),
  ``/sdapi/v1/samplers`` and Settings → Hide samplers. Then Forge's own ``set_samplers()`` runs (what
  ``add_sampler`` does after each registration) and ``all_samplers_map`` is rebuilt with Forge's formula.
* ``sd_schedulers.all_schedulers`` and ``sd_schedulers.schedulers`` (RES4LYF appends to ``schedulers``
  only) — the Schedule type dropdowns, the Hires schedule type dropdown, the XYZ "Schedule type" choices,
  ``/sdapi/v1/schedulers`` and Settings → Hide schedulers. ``schedulers_map`` is rebuilt in the new order
  (Forge's formula, then the keys outside it, such as this extension's aliases).

A rebuilt map keeps every key it had with the value it had: a key that would map to another entry after the
rebuild (two entries sharing a name, alias or label) gets its old value back and is logged, and the keys
extensions put there themselves stay. ``sd_samplers.get_sampler_and_scheduler`` (``functools.cache``) is
cleared: it walks the scheduler list for legacy "Sampler: <sampler> <scheduler>" infotext.

Every run sorts what a fresh start would. On Reload UI Forge keeps these modules loaded, so the lists come
back in the previous run's order, ``set_samplers()`` (modules/initialize.py ``initialize_rest``) rebuilds
``samplers_map`` over that order with the new Settings → Hide samplers (a Reload UI setting), and the scripts
register again (this extension's scheduler registration writes Forge's ``schedulers_map`` formula over it).
So a run first puts each list back in registration order — the order Forge alone has; it is recorded on
Forge's module the first time (``REGISTRATION_ATTR``, gone with the module) and an entry seen later is
appended, as Forge appends it — and a shared map key that Forge's formula gave to another entry only because
it ran over the sorted order gets the registration-order entry back. Then it sorts: the same lists, maps and
defaults as a fresh start with the same Settings.

Rules, per list (``new_order``):

1. Forge's first entry stays first — Forge's default (the dropdowns' initial value, the fallback of
   ``get_sampler_and_scheduler`` and ``find_sampler_config(None)``): DPM++ 2M and Automatic. In the sampler
   lists the first entry (in registration order) Settings → Hide samplers does not hide stays the first
   visible one as well (the txt2img/img2img dropdowns start on it), also after a Reload UI that changed it.
2. An entry whose label is in the table (exactly as Forge shows it) goes to its family, in the table's
   order. Entries that are not installed are simply not there.
3. A sampler whose label ends in "(RES4LYF)" goes to one block after the families, in its current order.
4. Any other entry whose label starts with a label or prefix of a family (case-insensitive; the longest
   match wins) goes to the end of that family, in its current relative order.
5. The rest keep their relative order at the very end — in the scheduler list before "custom", which is
   placed last.

"Flux Realistic" is the label Forge gives "DPM++ 2s a RF" (the same entry, ``sample_dpmpp_2s_ancestral_RF``)
when ``forbidden_knowledge`` is set, so it has that slot.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Iterable, Sequence

LOG_PREFIX = "[sam-extra]"
REGISTRATION_ATTR = "_sam3_registration_order"     # on modules.sd_samplers / modules.sd_schedulers


@dataclass(frozen=True)
class Family:
    labels: tuple[str, ...]
    prefixes: tuple[str, ...] = ()     # more name starts than the labels themselves


@dataclass(frozen=True)
class Suffix:
    suffix: str


class _Rest:
    def __repr__(self) -> str:
        return "REST"


REST = _Rest()

RES4LYF_SUFFIX = "(RES4LYF)"

SAMPLER_SECTIONS: tuple = (
    Family(("DPM++ 2M", "DPM++ 2M SDE", "DPM++ 2M SDE Heun", "DPM++ 3M SDE", "DPM++ 4M SDE", "DPM++ SDE",
            "DPM++ 2s a RF", "Flux Realistic", "DPM++ 2M (flow ODE)", "DPM++ 2M Heun (flow ODE)",
            "DPM++ 3M (flow ODE)", "DPM++ 2M CFG++", "DPM++ SDE CFG++", "DPM2"), prefixes=("DPM",)),
    Family(("Euler", "Euler a", "Euler A2", "Euler CFG++", "Euler a CFG++", "Euler Dy", "Euler SMEA Dy",
            "Euler Dy CFG++", "Euler SMEA Dy CFG++", "CFG++ UD10 AB")),
    Family(("ER SDE", "ER SDE (Reverse-time)", "ER SDE (ODE)", "ER SDE (Tunable)")),
    Family(("Res Multistep", "Res Multistep CFG++", "Res Multistep A", "Res Multistep A CFG++")),
    Family(("Heun", "EXP Heun 2 x0", "EXP Heun 2 x0 SDE"), prefixes=("EXP Heun",)),
    Family(("LMS", "IPNDM", "IPNDM_V", "DEIS")),
    Family(("UniPC", "UniPC bh2")),
    Family(("Restart", "Restart (flow)")),
    Family(("LCM", "DDIM", "PLMS", "Kohaku LoNyu Yog", "Gradient Estimation", "Gradient Estimation CFG++",
            "SEEDS 2", "SEEDS 3", "SA Solver", "SA Solver PECE"), prefixes=("SEEDS",)),
    Suffix(RES4LYF_SUFFIX),
    REST,
)

SCHEDULER_SECTIONS: tuple = (
    Family(("Automatic",)),
    Family(("Simple", "Normal", "Uniform", "SGM Uniform", "DDIM")),
    Family(("Beta", "Beta57 (RES4LYF)")),
    Family(("Linear Quadratic", "KL Optimal")),
    Family(("Karras", "Karras Dynamic", "Flow Cosmos rho7", "Flow Cosmos Dynamic"), prefixes=("Flow Cosmos",)),
    Family(("Exponential", "Polyexponential", "Cosine", "CosineExponential blend", "Phi")),
    Family(("Laplace", "React Cosinusoidal DynSF", "Tan (RES4LYF)")),
    Family(("Align Your Steps", "Turbo", "Bong Tangent", "FlowMatchEulerDiscrete", "Flux2")),
    REST,
    Family(("custom",)),
)


def new_order(labels: Sequence[str], sections: Sequence, *, keep: Iterable = (0,)) -> list[int]:
    """The indices of ``labels`` in family order (a permutation of ``range(len(labels))``).

    ``keep``: indices that stay in front, in their current order (rules in the module docstring); ``None``
    and indices outside the list are ignored."""
    n = len(labels)
    kept = sorted({i for i in keep if i is not None and 0 <= i < n})
    explicit: dict[str, tuple[int, int]] = {}
    starts: list[tuple[str, int]] = []
    suffixes: list[tuple[str, int]] = []
    rest = len(sections)
    for index, section in enumerate(sections):
        if isinstance(section, Family):
            for position, label in enumerate(section.labels):
                explicit.setdefault(label, (index, position))
            starts.extend((start.casefold(), index) for start in (*section.labels, *section.prefixes))
        elif isinstance(section, Suffix):
            suffixes.append((section.suffix.casefold(), index))
        elif section is REST:
            rest = index

    def rank(i: int) -> tuple:
        label = str(labels[i])
        if label in explicit:
            section, position = explicit[label]
            return section, 0, position, i
        folded = label.casefold()
        for suffix, section in suffixes:
            if folded.endswith(suffix):
                return section, 1, 0, i
        matches = [(len(start), -section) for start, section in starts if folded.startswith(start)]
        if matches:
            return -max(matches)[1], 1, 0, i
        return rest, 1, 0, i

    skip = set(kept)
    return kept + sorted((i for i in range(n) if i not in skip), key=rank)


@dataclass
class Report:
    samplers_moved: bool = False        # this run changed the sampler order Forge shows
    schedulers_moved: bool = False
    restored_keys: list[str] = field(default_factory=list)   # map keys a rebuild would have pointed elsewhere
    undone_keys: list[str] = field(default_factory=list)     # shared keys the formula over the old order moved
    errors: list[str] = field(default_factory=list)


def _log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", file=sys.stderr)


def _distinct_lists(*candidates) -> list[list]:
    seen: set[int] = set()
    out = []
    for candidate in candidates:
        if isinstance(candidate, list) and id(candidate) not in seen:
            seen.add(id(candidate))
            out.append(candidate)
    return out


def _reorder(items: list, labels: Sequence[str], sections: Sequence, keep: Iterable) -> bool:
    order = new_order(labels, sections, keep=keep)
    if order == list(range(len(items))):
        return False
    items[:] = [items[i] for i in order]
    return True


def _registration_record(owner) -> dict:
    """Entry key → position in the order Forge registered the entries in, kept on the Forge module that holds
    the lists (it stays loaded across Reload UI, and the lists with it)."""
    record = getattr(owner, REGISTRATION_ATTR, None)
    if not isinstance(record, dict):
        record = {}
        setattr(owner, REGISTRATION_ATTR, record)
    return record


def _restore_registration_order(items: list, key, record: dict) -> None:
    """Put ``items`` back in registration order (replaced as a whole). A key not seen before is appended in
    its current order: Forge and the extensions only append (a Reload UI that adds an extension, too)."""
    keys = [key(item) for item in items]
    for k in keys:
        record.setdefault(k, len(record))
    order = sorted(range(len(items)), key=lambda i: record[keys[i]])
    if order != list(range(len(items))):
        items[:] = [items[i] for i in order]


def _sampler_key(item):
    return item.name


def _scheduler_key(item):
    return item.name, item.label


def _sampler_alias_formula(items) -> dict:
    out = {}     # modules/sd_samplers.py set_samplers()
    for item in items:
        out[item.name.lower()] = item.name
        for alias in item.aliases:
            out[alias.lower()] = item.name
    return out


def _scheduler_formula(items) -> dict:
    return {**{x.name: x for x in items}, **{x.label: x for x in items}}     # sd_schedulers.py:289


def _undo_formula(mapping, formula, before: Sequence, after: Sequence, report: Report, *, owner: str) -> None:
    """``formula`` (Forge's: a later entry wins a shared key) over the ``before`` order gives a shared key to
    another entry than over the ``after`` (registration) order: where ``mapping`` holds the ``before`` result,
    it gets the ``after`` one. Any other value (a key an extension set itself) stays."""
    if not isinstance(mapping, dict):
        return
    was, now = formula(before), formula(after)
    for key, value in was.items():
        if key in now and not _same(now[key], value) and key in mapping and _same(mapping[key], value):
            mapping[key] = now[key]
            report.undone_keys.append(f"{owner}[{key!r}]")


def _moved(before: Sequence, after: Sequence) -> bool:
    return len(before) != len(after) or any(a is not b for a, b in zip(before, after))


def _copy(mapping) -> dict | None:
    return dict(mapping) if isinstance(mapping, dict) else None


def _first_visible(labels: Sequence[str], hidden) -> int | None:
    return next((i for i, label in enumerate(labels) if label not in hidden), None)


def _same(a, b) -> bool:
    return a is b or (isinstance(a, str) and a == b)


def _rebuild_map(mapping: dict, rebuilt: dict, snapshot: dict, report: Report, *, owner: str) -> None:
    """``mapping`` := ``rebuilt`` (Forge's formula over the new order), then every key of ``snapshot`` with its
    old value: a key the rebuild points elsewhere (two entries sharing it) gets its old value back and is
    reported; keys outside the formula (aliases, keys extensions added themselves) stay, after the others."""
    ordered = dict(rebuilt)
    for key, value in snapshot.items():
        if key in ordered and not _same(ordered[key], value):
            report.restored_keys.append(f"{owner}[{key!r}]")
        ordered[key] = value
    mapping.clear()
    mapping.update(ordered)


def order_samplers(sd_samplers, report: Report) -> None:
    all_samplers = sd_samplers.all_samplers
    lists = _distinct_lists(all_samplers, getattr(sd_samplers, "samplers", None),
                            getattr(sd_samplers, "samplers_for_img2img", None))
    if not isinstance(all_samplers, list):
        return
    entry = [list(items) for items in lists]
    record = _registration_record(sd_samplers)
    for items in lists:
        _restore_registration_order(items, _sampler_key, record)
    _undo_formula(getattr(sd_samplers, "samplers_map", None), _sampler_alias_formula, entry[0], all_samplers,
                  report, owner="samplers_map")

    hidden = set(getattr(sd_samplers, "samplers_hidden", None) or ())
    all_map_before = _copy(getattr(sd_samplers, "all_samplers_map", None))
    alias_map_before = _copy(getattr(sd_samplers, "samplers_map", None))
    sorted_any = False
    for items in lists:
        names = [item.name for item in items]
        keep = (0, _first_visible(names, hidden))
        sorted_any |= _reorder(items, names, SAMPLER_SECTIONS, keep)
    report.samplers_moved = any(_moved(before, items) for before, items in zip(entry, lists))
    if not sorted_any:
        return

    set_samplers = getattr(sd_samplers, "set_samplers", None)
    if callable(set_samplers):
        set_samplers()            # Forge's own rebuild: samplers / samplers_for_img2img / samplers_map
    all_map = getattr(sd_samplers, "all_samplers_map", None)
    if all_map_before is not None and isinstance(all_map, dict):
        rebuilt = {item.name: item for item in sd_samplers.all_samplers}     # Forge's formula, new order
        _rebuild_map(all_map, rebuilt, all_map_before, report, owner="all_samplers_map")
    alias_map = getattr(sd_samplers, "samplers_map", None)
    if alias_map_before is not None and isinstance(alias_map, dict):
        _rebuild_map(alias_map, dict(alias_map), alias_map_before, report, owner="samplers_map")


def order_schedulers(sd_schedulers, report: Report) -> None:
    visible = sd_schedulers.schedulers
    lists = _distinct_lists(getattr(sd_schedulers, "all_schedulers", None), visible)
    entry = [list(items) for items in lists]
    entry_visible = list(visible) if isinstance(visible, list) else None
    record = _registration_record(sd_schedulers)
    for items in lists:
        _restore_registration_order(items, _scheduler_key, record)
    mapping = getattr(sd_schedulers, "schedulers_map", None)
    if entry_visible is not None:
        _undo_formula(mapping, _scheduler_formula, entry_visible, visible, report, owner="schedulers_map")

    sorted_any = False
    for items in lists:
        labels = [item.label for item in items]
        sorted_any |= _reorder(items, labels, SCHEDULER_SECTIONS, (0,))
    report.schedulers_moved = any(_moved(before, items) for before, items in zip(entry, lists))
    if not sorted_any:
        return
    if isinstance(mapping, dict):
        _rebuild_map(mapping, _scheduler_formula(visible), dict(mapping), report, owner="schedulers_map")


def _clear_lookup_cache(sd_samplers) -> None:
    clear = getattr(getattr(sd_samplers, "get_sampler_and_scheduler", None), "cache_clear", None)
    if callable(clear):
        clear()


def apply(sd_samplers=None, sd_schedulers=None, *, log=_log) -> Report:
    """Sort Forge's sampler and scheduler lists by family (module docstring). The module arguments exist
    for tests; in Forge they are imported from ``modules``. Never raises: a failure is logged (each list is
    replaced as a whole, so none is left half sorted)."""
    report = Report()
    try:
        if sd_samplers is None:
            from modules import sd_samplers
        order_samplers(sd_samplers, report)
    except Exception as exc:
        report.errors.append(f"samplers: {type(exc).__name__}: {exc}")
    try:
        if sd_schedulers is None:
            from modules import sd_schedulers
        order_schedulers(sd_schedulers, report)
    except Exception as exc:
        report.errors.append(f"schedulers: {type(exc).__name__}: {exc}")
    try:
        if report.samplers_moved or report.schedulers_moved or report.undone_keys:
            _clear_lookup_cache(sd_samplers)
    except Exception as exc:
        report.errors.append(f"lookup cache: {type(exc).__name__}: {exc}")
    if report.samplers_moved or report.schedulers_moved:
        moved = [name for name, flag in (("Sampling method", report.samplers_moved),
                                         ("Schedule type", report.schedulers_moved)) if flag]
        log(" and ".join(moved) + (" lists" if len(moved) > 1 else " list") + " sorted by family.")
    if report.restored_keys:
        log("Sampling method / Schedule type family order: kept the previous entry for "
            + ", ".join(report.restored_keys) + " (two entries share that key).")
    for error in report.errors:
        log(f"Sampling method / Schedule type family order not applied ({error}).")
    return report
