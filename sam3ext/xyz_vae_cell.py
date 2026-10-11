"""One cell of Forge's X/Y/Z ``VAE`` axis (replaced in place by sam-extra) - the Forge-facing half of
``sam3ext/xyz_vae.py``.

``plan`` runs in the always-on hook's ``before_process`` (``sam3ext/xyz_text_encoder_cell.py``), after every axis of
the cell has been applied and before ``process_images`` loads the model:

* the VAE the cell's model has: when the model Forge will use is the one already loaded (the cell's loading
  parameters are the ones it was built from), its ``first_stage_model`` itself; otherwise the VAE Forge will build
  from the cell's module list (the [Modules] Text Encoder axis's list when it has one) or the checkpoint
  (``xyz_vae.predicted_shapes``, from headers);
* when the picked file's tensors are exactly that VAE's, nothing changes: Forge's own swap renders the cell, as without
  sam-extra (the same path, the same calls);
* otherwise - a VAE of the other family (2D Qwen2D on a model built with a 3D Wan VAE, or the reverse), or one that VAE
  cannot take as it is - the cell's module list with the VAE replaced is returned: the hook hands it to the model loader
  the way the [Modules] Text Encoder axis's cells are loaded (loading parameters and the ``Module N`` list only - no
  setting, no config.json - and both put back after the cell), and ``take_over`` takes ``sd_vae`` out of the cell's
  ``override_settings`` (Forge's swap would raise and leave a black cell), so the cell renders what choosing the file
  in the "VAE / Text Encoder" dropdown renders.

When the headers cannot tell (a pickle VAE, an unreadable file, a format Forge converts first) the cell is left to
Forge, as before.
"""
from __future__ import annotations

import os
import sys

from . import xyz_text_encoder as xte
from . import xyz_vae as xvae


def _log(message: str) -> None:
    print(f"[XYZ VAE] {message}", file=sys.stderr)


def _forge():
    from modules import sd_models, shared
    from modules_forge import main_entry

    try:
        from modules import sd_vae
    except Exception:
        sd_vae = None
    return sd_models, shared, main_entry, sd_vae


def _uses_api_model(settings) -> bool:
    """An API caller's ``override_settings`` that change the model inside ``process_images`` (after this hook)."""
    return bool(settings.get(xte.CHECKPOINT_KEY)) or isinstance(settings.get(xte.MODULES_KEY), (list, tuple))


def _checkpoint_info(settings, current, sd_models):
    name = settings.get(xte.CHECKPOINT_KEY)
    if name:
        info = getattr(sd_models, "checkpoint_aliases", {}).get(name)
        if info is not None:
            return info
    info = current.get("checkpoint_info") if isinstance(current, dict) else None
    if info is not None:
        return info
    try:
        return sd_models.select_checkpoint()
    except Exception:
        return None


def model_vae(p, entries, te_modules, base):
    """``(shapes, where)`` of the VAE the cell's model is built with - ``where`` is ``"loaded"`` (read from the model in
    memory) or ``"predicted"`` (from the headers); ``(None, …)`` when it cannot be told."""
    sd_models, _, _, sd_vae = _forge()
    settings = getattr(p, "override_settings", None) or {}
    data = sd_models.model_data
    current = data.forge_loading_parameters
    if (not _uses_api_model(settings) and isinstance(current, dict) and current.get("checkpoint_info") is not None):
        expected = current
        if te_modules is not None and list(te_modules) != list(base):
            expected = dict(current)
            expected["additional_modules"] = list(te_modules)    # what the hook's begin hands Forge for the encoder
        if str(expected) == getattr(data, "forge_hash", None):  # forge_model_reload loads nothing new
            first_stage = getattr(getattr(data, "sd_model", None), "first_stage_model", None)
            state_dict = getattr(first_stage, "state_dict", None)
            if callable(state_dict):
                return xvae.module_shapes(state_dict()), "loaded"
    info = _checkpoint_info(settings, current, sd_models)
    names = set(getattr(sd_vae, "vae_dict", None) or ())
    checkpoint_shapes = xvae.checkpoint_file_shapes(getattr(info, "filename", None))
    return xvae.predicted_shapes(entries, checkpoint_shapes, classify=xte.classify_file, vae_names=names), "predicted"


def plan(p, te_modules=None):
    """The cell's module list when its VAE has to be loaded with the model (once the list is handed to Forge,
    ``take_over`` takes ``sd_vae`` out of the cell's ``override_settings``); ``None``: Forge's own swap renders the
    cell."""
    settings = getattr(p, "override_settings", None)
    if not isinstance(settings, dict):
        return None
    override = settings.get(xvae.OVERRIDE_KEY)
    if not isinstance(override, str) or override in ("", "None"):
        return None
    sd_models, shared, main_entry, sd_vae = _forge()
    module_list = getattr(main_entry, "module_list", None) or {}
    if not module_list:
        return None
    name = os.path.basename(override)
    path = module_list[name] if name in module_list and xvae.same_file(module_list[name], override) else override
    selection = getattr(shared.opts, xte.MODULES_KEY, None) or []
    base = xte.normalise(xte.effective_base(settings, selection), module_list)
    entries = list(te_modules) if te_modules is not None else list(base)
    if any(xvae.same_file(entry, path) for entry in entries):
        return None                    # the model's own VAE: Forge swaps nothing
    picked = xvae.file_shapes(path)
    if picked is None:
        return None                    # a pickle / unreadable file: Forge decides, as before
    target, where = model_vae(p, entries, te_modules, base)
    if target is None or xvae.swappable(target, picked):
        return None
    names = set(getattr(sd_vae, "vae_dict", None) or ())
    modules = xvae.replace_vae(entries, path, xte.classify_file, names)
    model = "the loaded model's VAE" if where == "loaded" else "the VAE the model is built with"
    _log(f"{name} is a {xvae.describe(picked)} and {model} is a {xvae.describe(target)} - Forge cannot swap it in place "
         f"({xvae.mismatch(target, picked)}), so this cell loads the model with it, as choosing it in the "
         "VAE / Text Encoder dropdown does")
    return modules


def take_over(p) -> None:
    """The cell's list has been handed to the model loader: Forge's in-place swap must not run on top of it."""
    settings = getattr(p, "override_settings", None)
    if isinstance(settings, dict):
        settings.pop(xvae.OVERRIDE_KEY, None)
