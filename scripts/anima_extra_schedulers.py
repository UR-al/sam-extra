"""Extra Schedulers — Cosine, CosineExponential blend, Phi, Laplace, Karras Dynamic and a safe custom scheduler.

The schedulers live in ``sam3ext/extra_schedulers/`` and are added to Forge's Schedule type list
when this script is loaded (``registry.register_schedulers`` — before Forge builds the UI, so the
dropdowns, the XYZ "Schedule type" axis and the API list include them). This always-on script
holds the accordion with the values two of them need (custom expression / sigma list, Laplace
μ/β), hands them to the schedulers for each generation (``process``), records them in infotext
and registers XYZ axes. Arguments, infotext keys and the controls are in
``sam3ext/ui_extra_schedulers.py``.

Clean reimplementation: the labels are the ones specified for infotext compatibility with the
unlicensed aoleg/Neo_ExtraSchedulers (fork of the equally unlicensed DenOfEquity/webUI_ExtraSchedulers),
but no code of either repository was read or used — only their READMEs, for the user-facing names.
"""
from __future__ import annotations

import math
import sys
import traceback
from functools import partial

from modules import script_callbacks, scripts

from sam3ext import layout_lanes
from sam3ext import ui_extra_schedulers as ues
from sam3ext.extra_schedulers import registry
from sam3ext.extra_schedulers import settings as es_settings
from sam3ext.extra_schedulers.expression import ExpressionError, compile_expression
from sam3ext.extra_schedulers.sigma_list import SigmaListError, parse_sigma_list


def _log(message: str) -> None:
    print(f"{registry.LOG_PREFIX} {message}", file=sys.stderr)


# Registered at load time: Forge loads every script before it builds the UI.
_REPORT = registry.register_schedulers()
_SUMMARY = registry.summary(_REPORT)
if _SUMMARY:
    _log(_SUMMARY)


# ---------------------------------------------------------------------------
# XYZ plot
# ---------------------------------------------------------------------------
# The script title, not the shorter "[Extra Schedulers]": aoleg/Neo_ExtraSchedulers' axis names are
# unknown (its code was not read), and a same-prefix axis already in the list would make
# _register_xyz_axes skip ours, or show two axes with one label.
XYZ_PREFIX = f"[{ues.TITLE}]"
XYZ_LAPLACE_MU = f"{XYZ_PREFIX} Laplace mu"
XYZ_LAPLACE_BETA = f"{XYZ_PREFIX} Laplace beta"
XYZ_CUSTOM_EXPRESSION = f"{XYZ_PREFIX} Custom expression"
XYZ_CUSTOM_SIGMAS = f"{XYZ_PREFIX} Custom sigma list"


def _xyz_overrides(p) -> dict:
    overrides = getattr(p, ues.XYZ_ATTR, None)
    if not isinstance(overrides, dict):
        overrides = {}
        setattr(p, ues.XYZ_ATTR, overrides)
    return overrides


def _xyz_set_number(p, x, xs, *, field: str):
    _xyz_overrides(p)[field] = x


def _xyz_set_custom(p, x, xs, *, mode: str):
    # A custom-text axis also picks the mode it is for; with both axes the later-applied one wins.
    overrides = _xyz_overrides(p)
    overrides["custom_mode"] = mode
    overrides["custom_expression" if mode == es_settings.MODE_EXPRESSION else "custom_sigmas"] = x


# Checked before the grid starts (xyz_grid calls ``confirm`` for every axis), like Forge's own
# confirm_samplers. XYZ splits axis values at commas unless a value is in double quotes.
_QUOTE_HINT = " (put each value that contains commas in double quotes)"


def _confirm_expressions(p, values):
    for value in values:
        try:
            compile_expression(value)
        except ExpressionError as exc:
            raise ValueError(f"{XYZ_CUSTOM_EXPRESSION}: {value!r}: {exc}{_QUOTE_HINT}") from None


def _confirm_sigma_lists(p, values):
    for value in values:
        try:
            parse_sigma_list(value)
        except SigmaListError as exc:
            raise ValueError(f"{XYZ_CUSTOM_SIGMAS}: {value!r}: {exc}{_QUOTE_HINT}") from None


def _confirm_range(label: str, low: float, high: float):
    """XYZ confirm for a Laplace axis: every value within the accordion slider's range (ComfyUI
    LaplaceScheduler's), checked before the grid starts — the generation itself would clamp it, and the
    grid would show the unclamped number."""

    def confirm(p, values):
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = math.nan
            if not (math.isfinite(number) and low <= number <= high):
                raise ValueError(f"{label}: {value!r} is outside {low:g} … {high:g} (the slider's range)")

    return confirm


def make_xyz_axes(xyz_grid) -> list:
    return [
        xyz_grid.AxisOption(
            XYZ_LAPLACE_MU, float, partial(_xyz_set_number, field="laplace_mu"),
            confirm=_confirm_range(XYZ_LAPLACE_MU, es_settings.LAPLACE_MU_MIN, es_settings.LAPLACE_MU_MAX),
        ),
        xyz_grid.AxisOption(
            XYZ_LAPLACE_BETA, float, partial(_xyz_set_number, field="laplace_beta"),
            confirm=_confirm_range(XYZ_LAPLACE_BETA, es_settings.LAPLACE_BETA_MIN, es_settings.LAPLACE_BETA_MAX),
        ),
        xyz_grid.AxisOption(
            XYZ_CUSTOM_EXPRESSION, str, partial(_xyz_set_custom, mode=es_settings.MODE_EXPRESSION),
            confirm=_confirm_expressions,
        ),
        xyz_grid.AxisOption(
            XYZ_CUSTOM_SIGMAS, str, partial(_xyz_set_custom, mode=es_settings.MODE_SIGMAS),
            confirm=_confirm_sigma_lists,
        ),
    ]


def _register_xyz_axes() -> None:
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break
    if xyz_grid is None:
        return
    if not any(str(axis.label).startswith(XYZ_PREFIX) for axis in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(make_xyz_axes(xyz_grid))


def _on_before_ui() -> None:
    try:
        _register_xyz_axes()
    except Exception:
        _log("xyz_grid axis registration failed:\n" + traceback.format_exc())


script_callbacks.on_before_ui(_on_before_ui)


# ---------------------------------------------------------------------------
# The always-on script
# ---------------------------------------------------------------------------
class ExtraSchedulers(scripts.Script):
    # Right under Anima 3.8B (-35) at the top of the "ANIMA 튜닝" section, i.e. just below Forge's
    # own sampler settings; in img2img (and with the sections setting off) above SAM3 (-30).
    sorting_priority = -34

    @property
    def section(self):
        return layout_lanes.anima_section(bool(getattr(self, "is_img2img", False)))

    def title(self):
        return ues.TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        controls = ues.build_controls(self.elem_id)
        self.infotext_fields = ues.infotext_fields(controls)
        return list(controls)

    def process(self, p, *args):
        # Forge reports and swallows exceptions from ``process``, so the state is set first and
        # from values that cannot fail: a broken argument must not leave the previous
        # generation's expression in place.
        try:
            current = es_settings.with_overrides(ues.coerce_args(args), getattr(p, ues.XYZ_ATTR, None))
        except Exception as exc:
            _log(f"bad arguments, using the defaults: {type(exc).__name__}: {exc}")
            current = es_settings.DEFAULTS
        es_settings.set_active(current)
        try:
            ues.write_infotext(p, current)
        except Exception as exc:
            _log(f"infotext not written: {type(exc).__name__}: {exc}")
