"""One cell of the XYZ ``[Modules] Text Encoder`` axis - or of Forge's ``VAE`` axis when its VAE cannot be swapped in
place - while Forge renders it: the Forge-facing half (the pure logic is ``sam3ext/xyz_text_encoder.py`` and
``sam3ext/xyz_vae.py``; ``scripts/modules_xyz_text_encoder.py`` calls these from its always-on hook).

A cell's module list comes from the Text Encoder axis (the selection with an encoder replaced), from the VAE axis
(``sam3ext/xyz_vae_cell.py``: the selection - or the Text Encoder axis's list - with the VAE replaced, only when Forge's
own in-place swap would fail), or from both; it is loaded the same way whichever axis made it. "The selection" is the
one the cell has when it starts (``before_process``, after every axis of the cell): from Forge 2.31 the stock VAE axis
puts the cell's VAE list into the cell's ``override_settings`` and the stock Checkpoint axis its checkpoint, and either
may be applied after the Text Encoder axis.

The cell's module list never goes into Forge's settings (``shared.opts``). Forge writes those to config.json whenever a
page loads (``main_entry.on_preset_change``), the VAE / Text Encoder dropdown changes or Settings are applied - which
can happen while a grid runs - and loads what it saved on the next start. (From Forge 2.31 the stock Checkpoint and
VAE axes do put their own values into the settings for the cell - ``process_images``' ``set_config`` - with or without
sam-extra; the Text Encoder axis's encoders still never go there.) Instead the list goes:

* into the loading parameters the next model load reads (``sd_models.model_data.forge_loading_parameters``), where
  Forge 2.30's stock ``Checkpoint name`` axis puts its checkpoint (xyz_grid ``refresh_loading_params_for_xyz_grid``);
  the checkpoint already there (the dropdown's, or the Checkpoint axis's) stays - and the one 2.31's ``set_config``
  rebuilds with (the Checkpoint axis's) is kept too;
* into the ``Module N`` infotext list (``processing._overridden_modules``, the list Forge's own VAE override uses). When
  the stock VAE axis rebuilds that list from the settings, ``process`` swaps the cell's encoders back in.

Forge rebuilds the loading parameters from the settings in a few places while a cell runs: an API request's
``override_settings`` (``set_config``), a Hires checkpoint / module switch and its restore, the Refiner switch and its
restore. All of them call ``main_entry.refresh_model_loading_parameters``, which ``install`` wraps once: while a cell
runs in the task that started it, a rebuild from the module list the cell started from gets the cell's list (and the
cell's checkpoint, when the rebuild took the settings' one) - what those places load when the cell's list is chosen in
the dropdown. A rebuild from another list (Hires VAE / TE modules, PiD, the user changing the dropdown mid-grid) stays
as Forge made it.

``finish`` (from the hook's ``postprocess``; for a cell that failed, when the next cell of the axis or the next task
starts) puts the loading parameters and the infotext list back if they are still the cell's - the parameters to what
Forge itself last built (before the cell, or the rebuild the cell's list stood in for: a preset switched mid-cell
stays). Passes inside a cell (ADetailer, SAM3 refine) render with the cell's model, as they would with the dropdown's.
"""
from __future__ import annotations

import functools
import sys
import traceback
import weakref
from dataclasses import dataclass

from . import xyz_text_encoder as xte
from . import xyz_vae as xvae
from . import xyz_vae_cell

WRAPPER_MARKER = "_sam_extra_xyz_text_encoder_wrapper"


def _log(message: str) -> None:
    print(f"[XYZ Text Encoder] {message}", file=sys.stderr)


# The attributes the axes set on a cell's copy of ``p`` (copies of it - passes inside the cell - share them).
MARK_ATTRS = (xte.CELL_ATTR, xvae.CELL_ATTR)


@dataclass
class Cell:
    p_ref: object               # () -> the cell's p (a weak reference)
    marks: tuple                # the axes' attributes on the cell's p (MARK_ATTRS; copies of the cell's p share them)
    task: object                # shared.state.time_start of the task the cell runs in
    modules: list               # the cell's list
    base: list                  # the list it was made from (the settings', or an API caller's)
    settings_checkpoint: object  # the settings' checkpoint when the cell started
    cell_checkpoint: object     # the checkpoint in the loading parameters when the cell started
    underlying: dict            # what Forge would load without the cell: the parameters before it, or Forge's latest
    #                             rebuild that the cell's list stood in for
    params: dict                # the cell's loading parameters (the latest)
    overridden: list            # the cell's Module N list (the latest)
    previous_overridden: object
    installed: list             # every Module N list set while the cell ran (the cell's and its passes')


_STATE: Cell | None = None


def state() -> Cell | None:
    return _STATE


def _forge():
    from modules import processing, sd_models, shared
    from modules_forge import main_entry

    return processing, sd_models, shared, main_entry


def _task(shared):
    return getattr(getattr(shared, "state", None), "time_start", None)


def _running(shared, cell: Cell) -> bool:
    task_state = getattr(shared, "state", None)
    return (task_state is not None and getattr(task_state, "time_start", None) == cell.task
            and bool(getattr(task_state, "job", "")))


def _same_checkpoint(a, b) -> bool:
    if a is b:
        return True
    return a is not None and b is not None and getattr(a, "filename", None) == getattr(b, "filename", None)


def _reference(p):
    try:
        return weakref.ref(p)
    except TypeError:
        return lambda: p


def _marks(p) -> tuple:
    return tuple(mark for mark in (getattr(p, attr, None) for attr in MARK_ATTRS) if mark is not None)


def _ours(p, cell: Cell) -> bool:
    """``p`` is the cell's, or a copy of it (a pass inside the cell)."""
    return any(mark is own for mark in _marks(p) for own in cell.marks)


def _owner(p) -> Cell | None:
    cell = _STATE
    if cell is not None and _ours(p, cell):
        return cell
    return None


# ---------------------------------------------------------------------------------------------------------------------
# The hook's calls
# ---------------------------------------------------------------------------------------------------------------------

def before_process(p, build=None) -> bool:
    """End a cell that is over, start this one; ``True`` when the cell's list was handed to Forge.

    ``build(p, cell)`` makes the Text Encoder cell's list now that every axis of the cell has been applied (the script's
    ``_build_cell``: the selection and checkpoint the cell has at this moment); once per cell, the list is kept in the
    mark (``xte.BUILT``), so a pass inside the cell (a copy of p) uses it. Without ``build``: the list ``apply`` made.

    A no-op (a few attribute reads) for a generation that is not a cell of either axis while no cell runs."""
    has_attr = any(hasattr(p, attr) for attr in MARK_ATTRS)
    cell = getattr(p, xte.CELL_ATTR, None)
    vae = getattr(p, xvae.CELL_ATTR, None)
    if _STATE is not None:
        if _ours(p, _STATE):
            return False                  # a copy of the running cell's p: a pass inside the cell
        _, _, shared, _ = _forge()
        if has_attr or _task(shared) != _STATE.task:
            finish()                      # the next cell, or a new task after a cell that failed
    if cell and build is not None and not cell.get(xte.BUILT):
        cell["modules"] = build(p, cell)
        cell[xte.BUILT] = True
    modules = list(cell["modules"]) if cell and cell["modules"] is not None else None
    planned = None
    if isinstance(vae, dict) and not vae.get(xvae.SEEN):
        vae[xvae.SEEN] = True             # decided once: a pass inside the cell (a copy of p) is not decided again
        try:
            planned = xyz_vae_cell.plan(p, modules)
        except Exception:
            xyz_vae_cell._log("checking whether Forge can swap the cell's VAE in place failed - Forge's own swap is "
                              "used:\n" + traceback.format_exc())
        if planned is not None:
            modules = planned
    if modules is None:
        return False
    handed = begin(p, modules)
    if handed and planned is not None:
        xyz_vae_cell.take_over(p)         # only now: if begin failed, Forge's swap still runs (and fails loudly)
    return handed


def begin(p, modules) -> bool:
    """Hand ``modules`` (the cell's list) to Forge's model loader and ``Module N`` list for the cell of ``p``."""
    global _STATE
    processing, sd_models, shared, main_entry = _forge()
    install(main_entry)
    module_list = main_entry.module_list
    selection = getattr(shared.opts, xte.MODULES_KEY, None) or []
    base = xte.normalise(xte.effective_base(getattr(p, "override_settings", None), selection), module_list)
    modules = list(modules)
    if modules == base:
        return False
    data = sd_models.model_data
    current = data.forge_loading_parameters
    if not isinstance(current, dict) or current.get("checkpoint_info") is None:
        main_entry.refresh_model_loading_parameters()     # what Forge does at startup
        current = data.forge_loading_parameters
    try:
        settings_checkpoint = sd_models.select_checkpoint()
    except Exception:
        settings_checkpoint = None
    params = dict(current)
    params["additional_modules"] = list(modules)
    overridden = list(modules)
    _STATE = Cell(p_ref=_reference(p), marks=_marks(p), task=_task(shared), modules=modules, base=base,
                  settings_checkpoint=settings_checkpoint, cell_checkpoint=current.get("checkpoint_info"),
                  underlying=current, params=params, overridden=overridden,
                  previous_overridden=getattr(processing, "_overridden_modules", None), installed=[overridden])
    data.forge_loading_parameters = params
    processing._overridden_modules = overridden
    return True


def process(p) -> None:
    """Start of ``process_images_inner``: after Forge's VAE override rebuilt the ``Module N`` list from the settings
    (the VAE axis swapping in place on a Text Encoder cell, or a pass inside the cell with a VAE of its own), put the
    cell's entries back into it."""
    cell = _STATE
    if cell is None:
        return
    processing, _, shared, main_entry = _forge()
    owned = _ours(p, cell)
    if not owned:
        # a pass inside the cell: only while the settings are the ones the cell was made from (not PiD's own list)
        selection = getattr(shared.opts, xte.MODULES_KEY, None) or []
        if _task(shared) != cell.task or xte.normalise(selection, main_entry.module_list) != cell.base:
            return
    current = getattr(processing, "_overridden_modules", None)
    if current is cell.overridden:
        return
    new = list(cell.modules) if current is None else xte.swap_entries(current, cell.base, cell.modules)
    processing._overridden_modules = new
    cell.installed.append(new)
    if owned:
        cell.overridden = new


def reassert(p) -> None:
    """``postprocess_image`` (before the image's infotext): a pass inside the cell with its own VAE override resets
    Forge's ``Module N`` list when it ends; the cell's image still gets the cell's list."""
    cell = _owner(p)
    if cell is None:
        return
    from modules import processing

    if getattr(processing, "_overridden_modules", None) is not cell.overridden:
        processing._overridden_modules = cell.overridden


def postprocess(p) -> None:
    cell = _STATE
    if cell is not None and cell.p_ref() is p:
        finish()


def finish() -> None:
    """Put the loading parameters and the ``Module N`` list back if they are still the cell's."""
    global _STATE
    cell, _STATE = _STATE, None
    if cell is None:
        return
    from modules import processing, sd_models

    data = sd_models.model_data
    if data.forge_loading_parameters is cell.params:
        data.forge_loading_parameters = cell.underlying
    current = getattr(processing, "_overridden_modules", None)
    if any(current is installed for installed in cell.installed):
        processing._overridden_modules = cell.previous_overridden


# ---------------------------------------------------------------------------------------------------------------------
# Forge's rebuilds of the loading parameters
# ---------------------------------------------------------------------------------------------------------------------

def install(main_entry) -> None:
    """Wrap ``main_entry.refresh_model_loading_parameters`` once (main_entry's own callers find the wrapper through
    the module's globals, the rest through the module attribute)."""
    original = getattr(main_entry, "refresh_model_loading_parameters", None)
    if original is None or getattr(original, WRAPPER_MARKER, False):
        return

    @functools.wraps(original)
    def refresh_model_loading_parameters(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("refresh", True) is False:     # Forge's "refresh=False": nothing was rebuilt
            return result
        try:
            after_refresh()
        except Exception:
            _log("keeping the cell's text encoder after Forge rebuilt its loading parameters failed:\n"
                 + traceback.format_exc())
        return result

    setattr(refresh_model_loading_parameters, WRAPPER_MARKER, True)
    main_entry.refresh_model_loading_parameters = refresh_model_loading_parameters


def after_refresh() -> None:
    cell = _STATE
    if cell is None:
        return
    _, sd_models, shared, main_entry = _forge()
    if not _running(shared, cell):
        return
    data = sd_models.model_data
    params = data.forge_loading_parameters
    if params is cell.params or not isinstance(params, dict):
        return
    if xte.normalise(params.get("additional_modules") or [], main_entry.module_list) != cell.base:
        return
    new = dict(params)
    new["additional_modules"] = list(cell.modules)
    if _same_checkpoint(params.get("checkpoint_info"), cell.settings_checkpoint):
        new["checkpoint_info"] = cell.cell_checkpoint
    data.forge_loading_parameters = new
    cell.underlying, cell.params = params, new
