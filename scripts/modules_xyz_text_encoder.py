"""XYZ axis ``[Modules] Text Encoder`` (sam-extra) — compare text encoders the way the stock ``VAE`` axis compares VAEs.

The logic is ``sam3ext/xyz_text_encoder.py``; this file is the WebUI seam:

* ``on_before_ui`` adds the axis to Forge's ``scripts/xyz_grid.py`` (appended; on Reload UI replaced at the same index —
  xyz_grid keeps a chosen axis by its index, see ``scripts/colorcraft.py``). Both tabs. ``confirm`` checks every value
  before the first cell, ``apply`` stores the cell's module list on the cell's copy of ``p``.
* ``XyzTextEncoderHook`` (always on, no UI, like ``scripts/negpip.py``) — in ``before_process``, which
  ``process_images`` calls after every axis was applied and before it saves the settings it restores, the cell gets a
  NEW ``override_settings`` with ``forge_additional_modules`` (and the checkpoint a Checkpoint axis chose). Forge then
  does what it does for an API override: ``modules_change`` / ``checkpoint_change`` without saving, a model reload,
  the real list in ``Module N``, and the user's selection back in its ``finally`` — also when the cell fails or is
  interrupted. config.json and the dropdown are never touched.

The file name sorts before ``negpip.py``, which stays the extension's last script (metadata.ini).
"""
from __future__ import annotations

import os
import sys
import traceback

from modules import script_callbacks, scripts

from sam3ext import xyz_text_encoder as xte

TITLE = "XYZ Text Encoder (sam-extra)"


def _log(message: str) -> None:
    print(f"[XYZ Text Encoder] {message}", file=sys.stderr)


# ---------------------------------------------------------------------------------------------------------------------
# Forge, read when used (nothing of it is imported at load time)
# ---------------------------------------------------------------------------------------------------------------------

def _opts():
    from modules import shared

    return shared.opts


def _main_entry():
    from modules_forge import main_entry

    return main_entry


def _module_list():
    """Forge's ``module_list`` (file name -> full path) and ``module_map`` (file name -> relative path). Forge fills them
    while it builds the UI and on every page load; when they are empty (an API-only start) they are filled once here."""
    main_entry = _main_entry()
    if not main_entry.module_list:
        try:
            main_entry.refresh_models()
        except Exception:
            _log("refreshing the VAE / Text Encoder list failed:\n" + traceback.format_exc())
    if not main_entry.module_list:
        raise xte.AxisValueError(f"{xte.AXIS_LABEL}: Forge's VAE / Text Encoder list is empty "
                                 "(no file in models/VAE or models/text_encoder)")
    return main_entry.module_list, main_entry.module_map


def _text_encoder_roots() -> list:
    roots = []
    try:
        from modules import paths

        roots.append(os.path.join(paths.models_path, "text_encoder"))
    except Exception:
        pass
    try:
        from modules import shared

        roots.extend(getattr(shared.cmd_opts, "text_encoder_dirs", None) or [])
    except Exception:
        pass
    return roots


def _classifier():
    roots = _text_encoder_roots()
    return lambda path: xte.classify_file(path, te_roots=roots)


def _selection() -> list:
    return list(getattr(_opts(), xte.MODULES_KEY, None) or [])


def _plan(p, value):
    module_list, _ = _module_list()
    base = xte.effective_base(getattr(p, "override_settings", None), _selection())
    return xte.plan(value, base=base, module_list=module_list, classify=_classifier())


# ---------------------------------------------------------------------------------------------------------------------
# XYZ axis
# ---------------------------------------------------------------------------------------------------------------------

def _apply(p, x, xs):
    # Raises on a bad value (confirm has already checked them all), which stops the grid like the stock axes do.
    modules = _plan(p, x)
    setattr(p, xte.CELL_ATTR, None if modules is None else {"value": x, "modules": modules})


def _confirm(p, xs):
    for x in xs:
        _plan(p, x)


def _format(p, opt, x):
    try:
        module_list, _ = _module_list()
        current = xte.effective_base(getattr(p, "override_settings", None), _selection())
        return xte.format_label(x, module_list=module_list, current=current, classify=_classifier())
    except Exception:
        return f"{xte.LABEL}: {x}"


def _choices():
    try:
        module_list, module_map = _module_list()
    except xte.AxisValueError:
        return [xte.CURRENT]
    return xte.choices(module_list, module_map, _selection(), _classifier(),
                       use_short=bool(getattr(_opts(), "sd_modules_dropdown_use_short", True)))


def make_axis(xyz_grid):
    return xyz_grid.AxisOption(xte.AXIS_LABEL, str, _apply, format_value=_format, confirm=_confirm, cost=xte.COST,
                               choices=_choices)


def register_xyz_axes(xyz_grid) -> None:
    """Append the axis; on a WebUI reload replace ours in place (same index), never reorder the list."""
    option = make_axis(xyz_grid)
    for index, existing in enumerate(xyz_grid.axis_options):
        if getattr(existing, "label", None) == option.label:
            xyz_grid.axis_options[index] = option
            return
    xyz_grid.axis_options.append(option)


def _find_xyz_grid():
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            return script.module
    return None


def _on_before_ui() -> None:
    try:
        xyz_grid = _find_xyz_grid()
        if xyz_grid is not None:
            register_xyz_axes(xyz_grid)
    except Exception:
        _log("xyz_grid axis registration failed:\n" + traceback.format_exc())


script_callbacks.on_before_ui(_on_before_ui)


# ---------------------------------------------------------------------------------------------------------------------
# The cell's settings
# ---------------------------------------------------------------------------------------------------------------------

def apply_cell(p) -> bool:
    """Give the cell its module list through ``p.override_settings``; ``True`` when something was set.

    A no-op (one attribute read) for every generation that is not a cell of this axis."""
    cell = getattr(p, xte.CELL_ATTR, None)
    if not cell:
        return False
    from modules import sd_models

    opts = _opts()
    module_list = _main_entry().module_list
    settings = getattr(p, "override_settings", None) or {}
    base = xte.normalise(xte.effective_base(settings, _selection()), module_list)
    carry = xte.checkpoint_to_carry(sd_models.model_data.forge_loading_parameters, sd_models.select_checkpoint())
    new = xte.cell_overrides(settings, cell["modules"], base, carry)
    if new is None:
        return False
    # process_images restores only the keys it finds in opts.data (a fresh config may not have them yet).
    for key in new.keys() - settings.keys():
        if key not in opts.data:
            value = getattr(opts, key, None)
            opts.data[key] = list(value) if isinstance(value, (list, tuple)) else value
    p.override_settings = new
    p.override_settings_restore_afterwards = True
    # process_images pops "sd_vae" (the stock VAE axis's value) from the dict it is handed — now ``new``. Without this
    # axis it would come out of the shared dict; take it out there too, so nothing is left behind that Forge removes.
    if xte.VAE_KEY in settings and settings is not new:
        settings.pop(xte.VAE_KEY, None)
    return True


class XyzTextEncoderHook(scripts.Script):
    def title(self):
        return TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return None

    def before_process(self, p, *args):
        try:
            apply_cell(p)
        except Exception:
            # Forge would only log it (errors.report) and render the cell with the dropdown's encoder; say so plainly.
            _log("the cell's text encoder was NOT applied:\n" + traceback.format_exc())
