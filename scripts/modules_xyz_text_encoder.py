"""XYZ axis ``[Modules] Text Encoder`` (sam-extra) — compare text encoders the way the stock ``VAE`` axis compares VAEs.

The logic is ``sam3ext/xyz_text_encoder.py`` (values, slots, the cell's list) and ``sam3ext/xyz_text_encoder_cell.py``
(handing the list to Forge for one cell); this file is the WebUI seam:

* ``on_before_ui`` adds the axis to Forge's ``scripts/xyz_grid.py`` (appended; on Reload UI replaced at the same index —
  xyz_grid keeps a chosen axis by its index, see ``scripts/colorcraft.py``). Both tabs. ``confirm`` checks every value
  before the first cell (and that two Text Encoder axes do not pick the same slot), ``apply`` stores the cell's module
  list on the cell's copy of ``p`` - built on the list of an earlier Text Encoder axis of the same cell.
* ``XyzTextEncoderHook`` (always on, no UI, like ``scripts/negpip.py``) — ``before_process`` hands the cell's list to
  Forge's model loader (the loading parameters, as the stock Checkpoint axis does) and to the ``Module N`` infotext,
  ``process`` / ``postprocess_image`` keep the infotext list the cell's, ``postprocess`` puts both back. Forge's
  settings - config.json and the dropdown - are never written.

The file name sorts before ``negpip.py``, which stays the extension's last script (metadata.ini).
"""
from __future__ import annotations

import os
import sys
import traceback

from modules import script_callbacks, scripts

from sam3ext import xyz_text_encoder as xte
from sam3ext import xyz_text_encoder_cell as cell_runtime

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


def _checkpoint(p=None):
    """The checkpoint the slot check reads (``xte.Checkpoint``, ``None`` when unknown): an API caller's
    ``override_settings`` checkpoint, else the Checkpoint dropdown's. A Checkpoint axis's values are not looked at."""
    try:
        from modules import sd_models

        info = None
        name = (getattr(p, "override_settings", None) or {}).get(xte.CHECKPOINT_KEY) if p is not None else None
        if name:
            info = sd_models.checkpoint_aliases.get(name)
        if info is None:
            info = sd_models.select_checkpoint()
        return xte.checkpoint_file(getattr(info, "filename", None), getattr(info, "name", None))
    except Exception:
        return None


def _overlap_error(slots) -> xte.AxisValueError:
    names = ", ".join(xte.slot_name(slot) for slot in sorted(slots) if slot != xte.UNKNOWN_SLOT) or "the same"
    return xte.AxisValueError(f"{xte.AXIS_LABEL} is on two axes that both pick {names} encoders - give each slot one "
                              "axis (for Flux: CLIP-L files on one axis, T5 files on the other)")


# ---------------------------------------------------------------------------------------------------------------------
# XYZ axis
# ---------------------------------------------------------------------------------------------------------------------

def _apply(p, x, xs):
    # Raises on a bad value (confirm has already checked them all), which stops the grid like the stock axes do.
    module_list, _ = _module_list()
    classify = _classifier()
    slots = xte.pick_slots(x, module_list, classify)
    earlier = getattr(p, xte.CELL_ATTR, None)
    if not slots:
        # "Use Forge current": mark the copy as a cell of the axis; an earlier Text Encoder axis's list stays
        if not hasattr(p, xte.CELL_ATTR):
            setattr(p, xte.CELL_ATTR, None)
        return
    if earlier:
        both = xte.overlap(earlier["slots"], slots)
        if both:
            raise _overlap_error(both)
        origin, base = earlier["origin"], earlier["modules"]
    else:
        origin = base = xte.normalise(xte.effective_base(getattr(p, "override_settings", None), _selection()),
                                      module_list)
    added = [entry for entry in base if entry not in origin]
    modules = xte.plan(x, base=base, module_list=module_list, classify=classify, checkpoint=_checkpoint(p), added=added)
    setattr(p, xte.CELL_ATTR, {"value": x, "modules": modules, "origin": origin,
                               "slots": slots | (earlier["slots"] if earlier else frozenset())})


def _confirm(p, xs):
    module_list, _ = _module_list()
    classify = _classifier()
    checkpoint = _checkpoint(p)
    base = xte.effective_base(getattr(p, "override_settings", None), _selection())
    axis_slots = frozenset()
    for x in xs:
        xte.plan(x, base=base, module_list=module_list, classify=classify, checkpoint=checkpoint)
        axis_slots |= xte.pick_slots(x, module_list, classify)
    earlier = list(getattr(p, xte.CONFIRMED_ATTR, None) or ())
    for other in earlier:
        both = xte.overlap(other, axis_slots)
        if both:
            raise _overlap_error(both)
    setattr(p, xte.CONFIRMED_ATTR, [*earlier, axis_slots])


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
                       use_short=bool(getattr(_opts(), "sd_modules_dropdown_use_short", True)), checkpoint=_checkpoint())


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
# The cell
# ---------------------------------------------------------------------------------------------------------------------

class XyzTextEncoderHook(scripts.Script):
    def title(self):
        return TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return None

    def before_process(self, p, *args):
        try:
            cell_runtime.before_process(p)
        except Exception:
            # Forge would only log it (errors.report) and render the cell with the dropdown's encoder; say so plainly.
            _log("the cell's text encoder was NOT applied:\n" + traceback.format_exc())

    def process(self, p, *args):
        try:
            cell_runtime.process(p)
        except Exception:
            _log("the cell's Module N list could not be set:\n" + traceback.format_exc())

    def postprocess_image(self, p, pp, *args):
        try:
            cell_runtime.reassert(p)
        except Exception:
            _log("the cell's Module N list could not be set:\n" + traceback.format_exc())

    def postprocess(self, p, processed, *args):
        try:
            cell_runtime.postprocess(p)
        except Exception:
            _log("putting the loading parameters back after the cell failed:\n" + traceback.format_exc())
