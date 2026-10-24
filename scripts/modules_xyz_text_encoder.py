"""XYZ axis ``[Modules] Text Encoder`` (sam-extra) — compare text encoders the way the stock ``VAE`` axis compares VAEs.

The logic is ``sam3ext/xyz_text_encoder.py`` (values, slots, the cell's list) and ``sam3ext/xyz_text_encoder_cell.py``
(handing the list to Forge for one cell); this file is the WebUI seam:

* ``on_before_ui`` adds the axis to Forge's ``scripts/xyz_grid.py`` (appended; on Reload UI replaced at the same index —
  xyz_grid keeps a chosen axis by its index, see ``scripts/colorcraft.py``). Both tabs. ``confirm`` checks every value
  before the first cell (and that two Text Encoder axes do not pick the same slot), ``apply`` stores the cell's values
  on the cell's copy of ``p`` (after an earlier Text Encoder axis of the same cell) with the list they make.
* ``on_before_ui`` also replaces Forge's own ``VAE`` axis in place (same index, label, choices with "None", cost,
  labels): a copy whose ``apply`` runs Forge's ``apply_vae`` and marks the cell (``sam3ext/xyz_vae.py``). Forge 2.30
  swaps the VAE into the loaded model; the hook sends a VAE Forge cannot swap (a 2D Qwen2D VAE on a model built with a
  3D Wan VAE, or the reverse - Forge's strict load fails and xyz_grid leaves the cell black) to the model loader
  instead, like a Text Encoder cell, and cells Forge can swap keep Forge's swap. Forge 2.31's axis loads the model with
  the VAE in the cell's module list itself; the copy writes that list again so that the value is its only VAE.
* ``XyzTextEncoderHook`` (always on, no UI, like ``scripts/negpip.py``) — ``before_process`` builds the cell's list from
  the values (now that every axis of the cell has been applied) and hands it to Forge's model loader (the loading
  parameters, as the stock Checkpoint axis of Forge 2.30 does) and to the ``Module N`` infotext, ``process`` /
  ``postprocess_image`` keep the infotext list the cell's, ``postprocess`` puts both back. sam-extra never writes
  Forge's settings (config.json, the dropdown); from Forge 2.31 the stock Checkpoint and VAE axes put their own values
  into them for the cell (``set_config`` in ``process_images``), as they do without sam-extra.

The file name sorts before ``negpip.py``, which stays the extension's last script (metadata.ini).
"""
from __future__ import annotations

import copy
import os
import sys
import traceback

from modules import script_callbacks, scripts

from sam3ext import xyz_text_encoder as xte
from sam3ext import xyz_text_encoder_cell as cell_runtime
from sam3ext import xyz_vae as xvae

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
    while it builds the UI and on every page load; when they are empty (an API-only start) they are filled once here.
    ``module_map`` came with Forge 2.30-6 (with the dropdown's relative-path option); before it - 2.29.2, 2.30-2 - it is
    ``{}`` here and the dropdown values are file names, as that Forge's dropdown shows them."""
    main_entry = _main_entry()
    if not main_entry.module_list:
        try:
            main_entry.refresh_models()
        except Exception:
            _log("refreshing the VAE / Text Encoder list failed:\n" + traceback.format_exc())
    if not main_entry.module_list:
        raise xte.AxisValueError(f"{xte.AXIS_LABEL}: Forge's VAE / Text Encoder list is empty "
                                 "(no file in models/VAE or models/text_encoder)")
    return main_entry.module_list, getattr(main_entry, "module_map", None) or {}


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


def _checkpoint_named(name=None):
    """``xte.Checkpoint`` of the checkpoint ``name`` (an ``override_settings`` value); without one, or for a name Forge
    does not know (``process_images`` drops such an override), the Checkpoint dropdown's. ``None`` when unknown."""
    try:
        from modules import sd_models

        info = sd_models.checkpoint_aliases.get(name) if name else None
        if info is None:
            info = sd_models.select_checkpoint()
        return xte.checkpoint_file(getattr(info, "filename", None), getattr(info, "name", None))
    except Exception:
        return None


def _checkpoint(p=None):
    """The checkpoint ``p`` renders with as far as its ``override_settings`` say now (``xte.Checkpoint``, ``None`` when
    unknown): an API caller's checkpoint - or, from Forge 2.31, the stock Checkpoint axis's once it has been applied to
    the cell (2.30 loads the axis's checkpoint without writing it there) - else the Checkpoint dropdown's."""
    name = (getattr(p, "override_settings", None) or {}).get(xte.CHECKPOINT_KEY) if p is not None else None
    return _checkpoint_named(name)


def _start_of(p) -> dict:
    settings = getattr(p, "override_settings", None) or {}
    modules = settings.get(xte.MODULES_KEY)
    return {"modules": list(modules) if isinstance(modules, (list, tuple)) else None,
            "checkpoint": settings.get(xte.CHECKPOINT_KEY)}


def _known_checkpoint(name):
    """``name`` (an ``override_settings`` value) when Forge knows it, else the Checkpoint dropdown's title read now -
    the checkpoint ``_checkpoint_named`` gives for either, as a name that does not read the dropdown again. ``name``
    when the dropdown cannot be read."""
    try:
        from modules import sd_models

        if name and sd_models.checkpoint_aliases.get(name) is not None:
            return name
        return getattr(sd_models.select_checkpoint(), "title", None) or name
    except Exception:
        return name


def _grid_start(p) -> dict:
    """The module list and checkpoint the grid starts from, which ``confirm`` checked the values against (``confirm``
    keeps them on the grid's ``p`` before its first cell): the module list of the grid's ``override_settings`` (an API
    caller's; ``None`` - the dropdown's), and the checkpoint of it or else the Checkpoint dropdown's, read then. Every
    cell shares that dict, and Forge 2.31's stock VAE / Checkpoint axes write their values into it - so while a cell's
    axes are applied it may hold the previous cell's; 2.31 can also leave a cell's checkpoint in the dropdown
    (``process_images`` puts back only keys config.json has, and nothing for an API caller's
    ``override_settings_restore_afterwards=False``). A caller that skips ``confirm``: what ``p`` has now."""
    start = getattr(p, xte.GRID_START_ATTR, None)
    return start if isinstance(start, dict) else _start_of(p)


def _overlap_error(slots) -> xte.AxisValueError:
    names = ", ".join(xte.slot_name(slot) for slot in sorted(slots) if slot != xte.UNKNOWN_SLOT) or "the same"
    return xte.AxisValueError(f"{xte.AXIS_LABEL} is on two axes that both pick {names} encoders - give each slot one "
                              "axis (for Flux: CLIP-L files on one axis, T5 files on the other)")


# ---------------------------------------------------------------------------------------------------------------------
# XYZ axis
# ---------------------------------------------------------------------------------------------------------------------

def _apply(p, x, xs):
    # Raises on a bad value (confirm has already checked them all), which stops the grid like the stock axes do. The
    # list is checked on the grid's starting selection and checkpoint - what confirm checked, never a value an earlier
    # cell left in the shared override_settings or (Forge 2.31) in the Checkpoint dropdown; the hook builds it again when
    # the cell starts (_build_cell).
    module_list, _ = _module_list()
    classify = _classifier()
    slots = xte.pick_slots(x, module_list, classify)
    earlier = getattr(p, xte.CELL_ATTR, None)
    if not slots:
        # "Use Forge current": mark the copy as a cell of the axis; an earlier Text Encoder axis's list stays
        if not hasattr(p, xte.CELL_ATTR):
            setattr(p, xte.CELL_ATTR, None)
        return
    start = _grid_start(p)
    if earlier:
        both = xte.overlap(earlier["slots"], slots)
        if both:
            raise _overlap_error(both)
        origin, values = earlier["origin"], (*earlier["values"], x)
    else:
        origin = xte.normalise(xte.effective_base({xte.MODULES_KEY: start["modules"]}, _selection()), module_list)
        values = (x,)
    modules = xte.plan_values(values, base=origin, module_list=module_list, classify=classify,
                              checkpoint=_checkpoint_named(start["checkpoint"]))
    setattr(p, xte.CELL_ATTR, {"value": x, "values": values, "modules": modules, "origin": origin,
                               "slots": slots | (earlier["slots"] if earlier else frozenset())})


def _build_cell(p, cell):
    """The cell's list when it starts (the hook's ``before_process``, after every axis of the cell): the values of its
    Text Encoder axes on the selection the cell has now - with Forge 2.31's stock VAE axis, the cell's VAE list in its
    ``override_settings`` whichever axis was applied first - and the checkpoint it loads (2.31's Checkpoint axis writes
    it there too). On Forge 2.30 that is what ``apply`` built. A value that does not fit the cell's checkpoint (a grid
    mixing model families on 2.31) gets the list ``apply`` checked - the one 2.30 renders - and a console line, rather
    than stopping the grid."""
    values = cell.get("values")
    if not values:
        return cell["modules"]
    module_list, _ = _module_list()
    classify = _classifier()
    base = xte.effective_base(getattr(p, "override_settings", None), _selection())
    try:
        return xte.plan_values(values, base=base, module_list=module_list, classify=classify, checkpoint=_checkpoint(p))
    except xte.AxisValueError as exc:
        checked = _checkpoint_named(_grid_start(p)["checkpoint"])
        modules = xte.plan_values(values, base=base, module_list=module_list, classify=classify, checkpoint=checked)
        name = f'"{checked.name}"' if checked is not None else "the one the values were checked against"
        _log(f"{exc} - this cell's list is made as for the checkpoint {name}, as the axis did before Forge 2.31")
        return modules


def _confirm(p, xs):
    module_list, _ = _module_list()
    classify = _classifier()
    if not isinstance(getattr(p, xte.GRID_START_ATTR, None), dict):
        start = _start_of(p)
        # without a checkpoint of the caller's, the dropdown's now - before a cell can leave its own there
        start["checkpoint"] = _known_checkpoint(start["checkpoint"])
        setattr(p, xte.GRID_START_ATTR, start)
    start = _grid_start(p)
    checkpoint = _checkpoint_named(start["checkpoint"])
    base = xte.effective_base({xte.MODULES_KEY: start["modules"]}, _selection())
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


# ---------------------------------------------------------------------------------------------------------------------
# Forge's VAE axis, replaced in place
# ---------------------------------------------------------------------------------------------------------------------

# On our copy of Forge's VAE option: the apply of Forge's option it replaced.
VAE_STOCK_APPLY = "_sam_extra_stock_apply"


def _vae_cell_modules(settings, x, before) -> None:
    """Forge 2.31: the cell's list the stock axis wrote, again with exactly ``x`` as the VAE
    (``xvae.axis_cell_modules``); Forge's list stays when ``x`` is not a file of Forge's VAE list. The dropdown's list
    (``_selection()``) counts at the grid's first cell only: Forge may leave an earlier cell's list in it."""
    from modules import sd_vae

    vae_dict = getattr(sd_vae, "vae_dict", None) or {}
    if x in (None, "None"):
        path = None
    elif x in vae_dict:
        path = vae_dict[x]
    else:
        return
    module_list, _ = _module_list()
    settings[xte.MODULES_KEY] = xvae.axis_cell_modules(before, path, _selection(), module_list=module_list,
                                                       classify=xte.classify_file, vae_names=set(vae_dict))


def make_vae_apply(stock_apply):
    """Forge's ``apply_vae``, then the mark the hook decides the cell by.

    Forge 2.30's writes the file to ``p.override_settings["sd_vae"]`` (swapped into the loaded model in place; the
    hook sends a file Forge cannot swap to the model loader) - left as it is. Forge 2.31's writes the cell's whole
    module list to ``override_settings["forge_additional_modules"]`` instead (the model is loaded with it, so both VAE
    families render); that list keeps a VAE of the selection that is not one of the axis's values, and the last VAE of
    the sorted list wins, so it is written again with exactly the value as the VAE, on the grid's starting list."""

    def apply(p, x, xs):
        settings = getattr(p, "override_settings", None)
        before = settings.get(xte.MODULES_KEY) if isinstance(settings, dict) else None
        stock_apply(p, x, xs)
        if isinstance(settings, dict) and settings.get(xte.MODULES_KEY) is not before:
            try:
                _vae_cell_modules(settings, x, before)       # Forge 2.31: the stock axis wrote the cell's list
            except Exception:
                print("[XYZ VAE] setting the cell's VAE failed - Forge's own list is used:\n" + traceback.format_exc(),
                      file=sys.stderr)
        setattr(p, xvae.CELL_ATTR, {"value": x})

    return apply


def replace_vae_axis(xyz_grid) -> None:
    """Put a copy of Forge's ``VAE`` axis option at its index - the same object otherwise (label, type, choices with
    "None", cost, labels, confirm, img2img / txt2img class) - whose ``apply`` also marks the cell. Again on a WebUI
    reload: the copy wraps Forge's apply, never an earlier copy's.

    Forge 2.31's ``VAE ("state_dict" only)`` axis (the in-place swap of 2.30's ``VAE`` axis, kept for speed) is not
    replaced: its values must be VAEs of the family the model was built with - another family fails Forge's strict load
    and leaves the cell black, as with 2.30's own axis."""
    for index, existing in enumerate(xyz_grid.axis_options):
        if getattr(existing, "label", None) != xvae.AXIS_LABEL:
            continue
        stock_apply = getattr(existing, VAE_STOCK_APPLY, None) or existing.apply
        option = copy.copy(existing)
        option.apply = make_vae_apply(stock_apply)
        setattr(option, VAE_STOCK_APPLY, stock_apply)
        xyz_grid.axis_options[index] = option


def _find_xyz_grid():
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            return script.module
    return None


def _on_before_ui() -> None:
    try:
        xyz_grid = _find_xyz_grid()
    except Exception:
        _log("finding xyz_grid failed:\n" + traceback.format_exc())
        return
    if xyz_grid is None:
        return
    try:
        register_xyz_axes(xyz_grid)
    except Exception:
        _log("xyz_grid axis registration failed:\n" + traceback.format_exc())
    try:
        replace_vae_axis(xyz_grid)
    except Exception:
        print("[XYZ VAE] replacing Forge's VAE axis failed - Forge's own axis stays:\n" + traceback.format_exc(),
              file=sys.stderr)


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
            cell_runtime.before_process(p, build=_build_cell)
        except Exception:
            # Forge would only log it (errors.report) and render the cell with the dropdown's encoder (a VAE Forge
            # cannot swap then fails in Forge's swap); say so plainly.
            _log("the cell's text encoder / VAE module list was NOT applied:\n" + traceback.format_exc())

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
