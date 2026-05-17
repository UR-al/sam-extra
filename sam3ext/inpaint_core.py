"""Reusable building blocks for SAM3-driven inpaint passes.

This module owns the i2i orchestration that was originally inlined in
``scripts/!sam3.py``. Keeping it framework-agnostic (no Gradio, no script
runner state) lets both the in-flight ``postprocess_image`` callback and the
new post-generation "Refine" panel share the same code path.
"""
from __future__ import annotations

import sys
import traceback
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image

from modules import shared
from modules.processing import StableDiffusionProcessingImg2Img, process_images


SCRIPT_EXCLUDE_FILENAMES = frozenset({"!sam3", "sam3_mask"})


# Maps the user-facing inpainting_fill label (matches webui's img2img
# "Masked content" dropdown) to the int value StableDiffusionProcessingImg2Img
# accepts. Accept the int directly too for backward compat with code paths
# that already pass an int.
_INPAINTING_FILL_MAP = {
    "fill": 0,
    "original": 1,
    "latent noise": 2,
    "latent nothing": 3,
}


def _resolve_inpainting_fill(raw, default: int = 2) -> int:
    if raw is None:
        return default
    if isinstance(raw, bool):  # bool is subclass of int, exclude explicitly
        return default
    if isinstance(raw, int):
        return raw if 0 <= raw <= 3 else default
    try:
        return _INPAINTING_FILL_MAP[str(raw).strip().lower()]
    except KeyError:
        return default


@contextmanager
def pause_total_tqdm():
    """Hide the outer total-tqdm bar while inner i2i passes run.

    Earlier version had an ``except Exception: yield`` block that also
    swallowed exceptions raised THROUGH the yield (from inside the user's
    ``with`` body), then yielded a second time — producing
    ``RuntimeError: generator didn't stop after throw()`` whenever an inner
    pass raised. Guard only the import here; exceptions from inside the
    yield propagate normally.
    """
    try:
        from modules.shared import opts
    except Exception:
        yield
        return
    with patch.dict(opts.data, {"multiple_tqdm": False}, clear=False):
        yield


def copy_prompt(prompt_value: Any, fallback: str) -> str:
    text = str(prompt_value or "").strip()
    return text or str(fallback or "")


def apply_prompt_sr(p, text: str) -> str:
    pairs = getattr(p, "_sam3_xyz_prompt_sr", None) or []
    for pair in pairs:
        text = text.replace(pair.s, pair.r)
    return text


def script_args_copy(script_args):
    script_args = script_args or []
    type_ = type(script_args)
    result = []
    for arg in script_args:
        try:
            copied = copy(arg)
        except TypeError:
            copied = arg
        result.append(copied)
    return type_(result)


def script_filter(p, exclude_filenames=SCRIPT_EXCLUDE_FILENAMES):
    """Clone ``p.scripts`` with SAM3 itself stripped, keeping every other
    alwayson script (notably ControlNet) intact.
    """
    script_runner = copy(getattr(p, "scripts", None))
    script_args = script_args_copy(getattr(p, "script_args", []))
    if script_runner is None:
        return None, script_args

    filtered = []
    for script_object in getattr(script_runner, "alwayson_scripts", []):
        filename = Path(getattr(script_object, "filename", "")).stem.lower()
        if filename in exclude_filenames:
            continue
        filtered.append(script_object)

    script_runner.alwayson_scripts = filtered
    return script_runner, script_args


def get_sampler(p, args: dict[str, Any]) -> str | None:
    if args.get("sam3_use_sampler"):
        sampler = args.get("sam3_sampler", "Use same sampler")
        if sampler != "Use same sampler":
            return sampler
    return getattr(p, "sampler_name", None)


def get_scheduler(p, args: dict[str, Any]) -> dict:
    """Sampler and scheduler used to be coupled to a single ``sam3_use_sampler``
    toggle — v0.6.0 splits them. ``sam3_use_scheduler=True`` (or the legacy
    ``sam3_use_sampler=True``) opts into the args-supplied scheduler;
    otherwise inherit the parent ``p.scheduler``."""
    if not hasattr(p, "scheduler"):
        return {}
    opted_in = args.get("sam3_use_scheduler") or args.get("sam3_use_sampler")
    if opted_in:
        scheduler = args.get("sam3_scheduler", "Use same scheduler")
        if scheduler not in (None, "", "Use same scheduler"):
            return {"scheduler": scheduler}
    return {"scheduler": getattr(p, "scheduler")}


def get_seed(p, args: dict[str, Any]) -> int:
    """Return the seed to pass to the i2i pass.

    - In-flight: opt-in via ``sam3_use_seed``; otherwise inherit ``p.seed``.
    - Refine path passes ``sam3_use_seed=True`` unconditionally so the
      Refine panel's seed (default -1 = random) always wins.
    """
    if args.get("sam3_use_seed"):
        try:
            return int(args.get("sam3_seed", -1))
        except (TypeError, ValueError):
            return -1
    return getattr(p, "seed", -1) if p is not None else -1


def get_noise_multiplier(p, args: dict[str, Any]) -> float:
    if args.get("sam3_use_noise_multiplier"):
        return float(args.get("sam3_noise_multiplier", 1.0))
    return getattr(p, "initial_noise_multiplier", None) or 1.0


def build_i2i(p, image: Image.Image, args: dict[str, Any]) -> StableDiffusionProcessingImg2Img:
    width = int(args["sam3_inpaint_width"]) if args.get("sam3_use_inpaint_width_height") else int(getattr(p, "width", image.width))
    height = int(args["sam3_inpaint_height"]) if args.get("sam3_use_inpaint_width_height") else int(getattr(p, "height", image.height))
    steps = int(args["sam3_steps"]) if args.get("sam3_use_steps") else int(getattr(p, "steps", 28))
    cfg_scale = float(args["sam3_cfg_scale"]) if args.get("sam3_use_cfg_scale") else float(getattr(p, "cfg_scale", 7.0))
    sampler_name = get_sampler(p, args)
    noise_multiplier = get_noise_multiplier(p, args)
    version_args = get_scheduler(p, args)
    seed = get_seed(p, args)
    # In-flight inpaint is usually a face/detailer style refinement of a
    # small region — keep the masked area's original pixels by default so
    # the model has color context to work from. Refine path (drastic
    # changes like clothes swap) defaults to ``latent noise`` instead.
    # Both paths honor an explicit ``sam3_inpainting_fill`` from args.
    inpainting_fill = _resolve_inpainting_fill(args.get("sam3_inpainting_fill"), default=1)

    p2 = StableDiffusionProcessingImg2Img(
        init_images=[image],
        resize_mode=0,
        denoising_strength=float(args["sam3_denoising_strength"]),
        mask=None,
        mask_blur=int(args["sam3_mask_blur"]),
        inpainting_fill=inpainting_fill,
        inpaint_full_res=bool(args["sam3_inpaint_only_masked"]),
        inpaint_full_res_padding=int(args["sam3_inpaint_only_masked_padding"]),
        inpainting_mask_invert=0,
        initial_noise_multiplier=noise_multiplier,
        sd_model=p.sd_model,
        outpath_samples=p.outpath_samples,
        outpath_grids=p.outpath_grids,
        prompt="",
        negative_prompt="",
        styles=getattr(p, "styles", []),
        seed=seed,
        subseed=getattr(p, "subseed", -1),
        subseed_strength=getattr(p, "subseed_strength", 0),
        seed_resize_from_h=getattr(p, "seed_resize_from_h", 0),
        seed_resize_from_w=getattr(p, "seed_resize_from_w", 0),
        sampler_name=sampler_name,
        batch_size=1,
        n_iter=1,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=bool(args.get("sam3_restore_face", False)),
        tiling=getattr(p, "tiling", False),
        extra_generation_params=dict(getattr(p, "extra_generation_params", {})),
        do_not_save_samples=True,
        do_not_save_grid=True,
        **version_args,
    )
    # Match StableDiffusionProcessing's default cached_c / cached_uc shape
    # (length 3) — earlier length-2 risked IndexError if webui or a script
    # ever reads index 2.
    p2.cached_c = [None, None, None]
    p2.cached_uc = [None, None, None]
    p2.scripts, p2.script_args = script_filter(p)
    p2._sam3_inner = True
    p2.all_hr_prompts = [""]
    p2.all_hr_negative_prompts = [""]
    return p2


def _find_controlnet_script(p2: StableDiffusionProcessingImg2Img):
    runner = getattr(p2, "scripts", None)
    if runner is None:
        return None
    for script_object in getattr(runner, "alwayson_scripts", []):
        filename = Path(getattr(script_object, "filename", "")).stem.lower()
        if filename == "controlnet":
            return script_object
    return None


def _disabled_cn_unit(template_unit, ControlNetUnit):
    """Build a fresh disabled unit, preferring to copy the template (which
    preserves any user-set defaults like resize_mode) and just flipping
    ``enabled`` off. Falls back to constructing from scratch if copying fails.
    """
    try:
        disabled = copy(template_unit)
        disabled.enabled = False
        return disabled
    except Exception:
        return ControlNetUnit(enabled=False, module="None", model="None")


def inject_controlnet_unit(p2: StableDiffusionProcessingImg2Img, cn_args: dict[str, Any] | None) -> None:
    """Patch ``p2.script_args`` so the inpaint pass runs with a SAM3-supplied
    ControlNet unit in slot 0.

    No-op when ``cn_args`` is ``None`` / ``sam3_cn_enable`` is false, when the
    ControlNet extension is not importable, or when the ControlNet script is
    not registered in ``p2.scripts`` (e.g. ``control_net_unit_count == 0``).
    """
    if not cn_args or not cn_args.get("sam3_cn_enable"):
        return

    try:
        from lib_controlnet.external_code import ControlNetUnit
        from lib_controlnet import global_state as _cn_state
    except Exception as exc:
        print(f"[-] SAM3: ControlNet extension not loaded, skipping CN injection ({exc})", file=sys.stderr)
        return

    # Re-register models/sam3/ entries on the CN filename dict in case it was
    # reset between UI build time and now (refresh button in the standard
    # ControlNet UI calls update_controlnet_filenames() which wipes the dict).
    try:
        from .ui import _scan_sam3_dir_for_cn_models

        extras = _scan_sam3_dir_for_cn_models()
        if extras:
            _cn_state.controlnet_filename_dict.update(extras)
    except Exception:
        pass

    cn_script = _find_controlnet_script(p2)
    if cn_script is None:
        print("[-] SAM3: ControlNet script not present in alwayson_scripts; skipping CN injection.", file=sys.stderr)
        return

    args_from = getattr(cn_script, "args_from", None)
    args_to = getattr(cn_script, "args_to", None)
    if args_from is None or args_to is None or args_to <= args_from:
        print(
            "[-] SAM3: ControlNet has no unit slots in script_args "
            f"(args_from={args_from}, args_to={args_to}); is control_net_unit_count == 0?",
            file=sys.stderr,
        )
        return

    module_name = str(cn_args.get("sam3_cn_module", "inpaint_only"))
    model_name = str(cn_args.get("sam3_cn_model", "None"))

    # LLLite anima inpaint variants take a 4-channel (RGB + mask) cond and
    # need the mask tensor to survive preprocessing. ``inpaint_*`` preprocessors
    # discard the mask (return None) and rewrite cond, which breaks the
    # ``assert isinstance(mask, torch.Tensor)`` in the LLLite forward. Force a
    # pass-through preprocessor in that case.
    lower_model = model_name.lower()
    if "lllite" in lower_model and "inpaint" in lower_model and module_name.startswith("inpaint"):
        print(
            f"[-] SAM3: LLLite inpaint model '{model_name}' is incompatible with "
            f"preprocessor '{module_name}' (preprocessor strips the mask); "
            f"overriding to 'None' so the mask reaches the LLLite forward.",
            file=sys.stderr,
        )
        module_name = "None"

    sam3_unit = ControlNetUnit(
        enabled=True,
        module=module_name,
        model=model_name,
        weight=float(cn_args.get("sam3_cn_weight", 1.0)),
        guidance_start=float(cn_args.get("sam3_cn_guidance_start", 0.0)),
        guidance_end=float(cn_args.get("sam3_cn_guidance_end", 1.0)),
        pixel_perfect=bool(cn_args.get("sam3_cn_pixel_perfect", True)),
        control_mode=str(cn_args.get("sam3_cn_control_mode", "Balanced")),
        resize_mode=str(cn_args.get("sam3_cn_resize_mode", "Crop and Resize")),
        processor_res=int(cn_args.get("sam3_cn_processor_res", 512)),
        threshold_a=float(cn_args.get("sam3_cn_threshold_a", -1.0)),
        threshold_b=float(cn_args.get("sam3_cn_threshold_b", -1.0)),
        image=None,  # → ControlNet falls back to p2.init_images[0] + p2.image_mask
    )

    script_args = p2.script_args
    args_type = type(script_args)
    cn_slot = list(script_args[args_from:args_to])

    if cn_args.get("sam3_cn_override_external"):
        cn_slot = [_disabled_cn_unit(unit, ControlNetUnit) for unit in cn_slot]

    cn_slot[0] = sam3_unit
    p2.script_args = args_type(list(script_args[:args_from]) + cn_slot + list(script_args[args_to:]))


def build_standalone_i2i(
    image: Image.Image,
    args: dict[str, Any],
    *,
    sd_model,
    outpath_samples: str,
    outpath_grids: str,
    scripts_runner,
    script_args,
) -> StableDiffusionProcessingImg2Img:
    """Build a fresh ``StableDiffusionProcessingImg2Img`` for the post-inpaint
    refine flow (no ``p`` to inherit from).

    The caller supplies the SD model, output paths, and a scripts runner
    (typically ``modules.scripts.scripts_txt2img`` with SAM3 stripped). Defaults
    for steps/cfg/sampler come from ``args`` directly — there is no parent
    process to fall back to.
    """
    width = int(args["sam3_inpaint_width"]) if args.get("sam3_use_inpaint_width_height") else int(image.width)
    height = int(args["sam3_inpaint_height"]) if args.get("sam3_use_inpaint_width_height") else int(image.height)
    steps = int(args.get("sam3_steps", 28))
    cfg_scale = float(args.get("sam3_cfg_scale", 7.0))
    sampler_name = args.get("sam3_sampler", "Use same sampler")
    if sampler_name in (None, "", "Use same sampler"):
        sampler_name = "Euler a"
    noise_multiplier = float(args.get("sam3_noise_multiplier", 1.0))
    scheduler = args.get("sam3_scheduler", "Use same scheduler")
    version_args = {} if scheduler in (None, "", "Use same scheduler") else {"scheduler": scheduler}
    try:
        seed = int(args.get("sam3_seed", -1))
    except (TypeError, ValueError):
        seed = -1

    # ``inpainting_fill`` semantics (matches webui's img2img inpaint dropdown):
    #   0 = fill          (gray fill in masked area)
    #   1 = original      (keep original pixels in masked area as init)
    #   2 = latent noise  (random latent in masked area)  ← Refine default
    #   3 = latent nothing (zeros in masked area)
    # For "shirt → nude" style drastic transforms, ``2`` gives the cleanest
    # separation from the original at sampling start. ``1`` (the previous
    # default) was leaving original-color bias even at denoise=1 — visible
    # as a flat-color paint over the unchanged garment.
    inpainting_fill = _resolve_inpainting_fill(args.get("sam3_inpainting_fill"))

    p2 = StableDiffusionProcessingImg2Img(
        init_images=[image],
        resize_mode=0,
        denoising_strength=float(args["sam3_denoising_strength"]),
        mask=None,
        mask_blur=int(args["sam3_mask_blur"]),
        inpainting_fill=inpainting_fill,
        inpaint_full_res=bool(args["sam3_inpaint_only_masked"]),
        inpaint_full_res_padding=int(args["sam3_inpaint_only_masked_padding"]),
        inpainting_mask_invert=0,
        initial_noise_multiplier=noise_multiplier,
        sd_model=sd_model,
        outpath_samples=outpath_samples,
        outpath_grids=outpath_grids,
        prompt="",
        negative_prompt="",
        styles=[],
        seed=seed,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        sampler_name=sampler_name,
        batch_size=1,
        n_iter=1,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=bool(args.get("sam3_restore_face", False)),
        tiling=False,
        extra_generation_params={},
        do_not_save_samples=False,  # post-mode results are user-triggered: save them
        do_not_save_grid=True,
        **version_args,
    )
    p2.cached_c = [None, None, None]
    p2.cached_uc = [None, None, None]
    p2.scripts = scripts_runner
    p2.script_args = script_args
    p2._sam3_inner = True
    p2.all_hr_prompts = [""]
    p2.all_hr_negative_prompts = [""]
    return p2


def _find_sampler_script(runner):
    """Locate Forge's built-in ScriptSampler — the script whose ``setup(p)``
    method silently overwrites ``p.steps`` / ``p.sampler_name`` /
    ``p.scheduler`` with the values from its own UI widgets. We need to
    override its slot in script_args so our intended sampler/steps survive.
    """
    for s in getattr(runner, "alwayson_scripts", []) or []:
        if Path(getattr(s, "filename", "")).stem.lower() != "sampler":
            continue
        try:
            if (s.title() or "").lower() == "sampler":
                return s
        except Exception:
            pass
    return None


def override_sampler_script_slot(p2, args: dict[str, Any]) -> None:
    """ScriptSampler.setup runs early in process_images and does
    ``p.steps = steps; p.sampler_name = sampler_name; p.scheduler = scheduler``
    using whatever values sit in its 3-slot script_args window. Without an
    override, those values come from inherited t2i state (in-flight) or
    component defaults (standalone refine), which silently clobber what we
    set via the constructor.

    For EACH of (steps, sampler, scheduler), we patch only when the
    corresponding ``sam3_use_*`` flag in ``args`` is True. This way:

    - In-flight: the user's per-slot override toggles
      (``sam3_use_steps`` / ``sam3_use_sampler`` / ``sam3_use_scheduler``)
      decide what overrides the t2i defaults.
    - Standalone refine: ``run_sam3_refine`` sets all three flags True so
      the Refine panel's chosen values always win.
    """
    sampler_script = _find_sampler_script(getattr(p2, "scripts", None))
    if sampler_script is None:
        return
    af = getattr(sampler_script, "args_from", None)
    at = getattr(sampler_script, "args_to", None)
    if af is None or at is None or at - af < 3:
        return

    use_steps = bool(args.get("sam3_use_steps"))
    use_sampler = bool(args.get("sam3_use_sampler"))
    # Back-compat: pre-0.6 the sampler toggle did both. Honor either flag.
    use_scheduler = bool(args.get("sam3_use_scheduler") or args.get("sam3_use_sampler"))

    if not (use_steps or use_sampler or use_scheduler):
        return

    script_args = p2.script_args
    args_type = type(script_args)
    patched = list(script_args)
    before = (patched[af + 0], patched[af + 1], patched[af + 2])

    if use_steps:
        patched[af + 0] = int(args.get("sam3_steps", before[0]))
    if use_sampler:
        sampler_name = str(args.get("sam3_sampler") or "Euler a")
        if sampler_name in ("", "Use same sampler"):
            sampler_name = "Euler a"
        patched[af + 1] = sampler_name
    if use_scheduler:
        scheduler = str(args.get("sam3_scheduler") or "Automatic")
        if scheduler in ("", "Use same scheduler"):
            scheduler = "Automatic"
        patched[af + 2] = scheduler

    after = (patched[af + 0], patched[af + 1], patched[af + 2])
    if before == after:
        return
    p2.script_args = args_type(patched) if not isinstance(script_args, list) else patched
    # NOTE for users wondering why the inpaint pass is slower than they
    # remember: pre-v0.6.0 the SAM3 panel's "Use separate ..." toggles
    # were silently ignored (ScriptSampler.setup re-applied the t2i
    # values). v0.6.0 fixed that, so a heavier sampler/scheduler chosen
    # here (e.g. ER SDE + Bong Tangent) actually runs in-flight now and
    # is heavier than the default DPM++ 2M that used to silently win.
    # Uncheck the toggles in the SAM3 panel to revert to t2i defaults.
    print(
        f"[-] SAM3: patched ScriptSampler slot (args[{af}:{at}]) — "
        f"before {before} → after {after} "
        f"(use_steps={use_steps}, use_sampler={use_sampler}, use_scheduler={use_scheduler})",
        file=sys.stderr,
    )


def _disable_inherited_cn_units(script_args, runner) -> None:
    """Walk ``runner.alwayson_scripts`` for the ControlNet script, find its
    args_from..args_to window, and replace every ControlNetUnit in that window
    with a disabled copy.

    Why: the t2i scripts runner's ControlNet inputs are ``gr.State`` objects
    initialized with the *user's currently-configured t2i CN units* (whatever
    they had set up before clicking Refine). Without this, a t2i CN unit
    with LLLite would silently run during Refine even when the Refine panel
    has cn_enable=False — causing surprise compute (LLLite 112 modules per
    attention layer) and OOM. ``inject_controlnet_unit`` will turn slot 0
    back on with the SAM3-supplied unit if the user opted in.
    """
    if not script_args:
        return
    cn_script = None
    for script_object in getattr(runner, "alwayson_scripts", []):
        if Path(getattr(script_object, "filename", "")).stem.lower() == "controlnet":
            cn_script = script_object
            break
    if cn_script is None:
        return
    af = getattr(cn_script, "args_from", None)
    at = getattr(cn_script, "args_to", None)
    if af is None or at is None or at <= af:
        return
    try:
        from lib_controlnet.external_code import ControlNetUnit
    except Exception:
        return
    for i in range(af, min(at, len(script_args))):
        existing = script_args[i]
        try:
            disabled = copy(existing)
            disabled.enabled = False
            script_args[i] = disabled
        except Exception:
            script_args[i] = ControlNetUnit(enabled=False, module="None", model="None")


def build_standalone_scripts_runner():
    """Clone the t2i script runner with SAM3 itself stripped (avoids
    recursion). Returns ``(runner, script_args)`` ready to drop onto a
    standalone ``p2``, or ``(None, None)`` if the t2i runner has not been
    built yet (e.g., refine is invoked before any UI has rendered).

    ``scripts_txt2img.inputs`` is a list of *Gradio components*, not values —
    a real Generate click lets Gradio resolve those to current UI values. For
    our standalone path there is no such click, so we collapse each component
    to its default ``.value``. For most alwayson scripts that means "do
    nothing"; for ControlNet, the .value is the user's currently-configured
    t2i CN unit which we then forcibly disable via
    ``_disable_inherited_cn_units`` so the Refine panel's CN settings are
    authoritative.
    """
    from modules import scripts as _scripts

    source = getattr(_scripts, "scripts_txt2img", None)
    if source is None:
        return None, None

    runner = copy(source)
    filtered = []
    for script_object in getattr(source, "alwayson_scripts", []):
        filename = Path(getattr(script_object, "filename", "")).stem.lower()
        if filename in SCRIPT_EXCLUDE_FILENAMES:
            continue
        filtered.append(script_object)
    runner.alwayson_scripts = filtered

    components = getattr(source, "inputs", None) or []
    script_args = [getattr(component, "value", None) for component in components]

    # Slot 0 is the "Script" selectable-script-index dropdown (type="index").
    # In a real Generate click Gradio converts the selected choice ("None") to
    # int 0 *before* fn invocation. We bypass that, so use the int directly.
    # Third-party extensions read p.script_args[0] and assume int — leaving the
    # raw "None" string in place crashes them (api-payload-display, etc.).
    if script_args:
        script_args[0] = 0

    # Force all inherited ControlNet units off. The Refine panel's CN
    # configuration is the only one that should drive the refine pass; without
    # this, a t2i CN unit with LLLite would keep running even when the Refine
    # panel says cn_enable=False — which is exactly the "Refine is slow and
    # the LLLite isn't supposed to be on" case the v0.6.1 user reported.
    _disable_inherited_cn_units(script_args, runner)

    return runner, script_args


def run_sam3_refine(
    image: Image.Image,
    args: dict[str, Any],
    *,
    sd_model,
    outpath_samples: str,
    outpath_grids: str,
    override_settings: dict[str, Any] | None = None,
) -> list[tuple[Image.Image, str]]:
    """Standalone equivalent of ``run_inpaint_passes`` for the post-generation
    Refine panel: no ``p`` to inherit from, runs once per detected mask, and
    returns ``(image, infotext)`` pairs (one per mask) instead of chaining
    them.

    The ``infotext`` is ``processed.infotexts[0]`` (or ``processed.info`` as
    fallback) — the exact "Parameters: ..." string that gets embedded in the
    saved PNG. The handler stitches these into the gallery's
    ``generation_info`` JSON so clicking the new image in the gallery shows
    the real transformed prompt, not the original t2i prompt.

    Returns ``[]`` when SAM3 finds nothing or every pass is interrupted.
    """
    from .core import run_sam3_on_pil, unload_sam3

    # Standalone refine always overrides the t2i sampler/steps/scheduler/seed
    # — there's no parent process to inherit from. Set the use_* flags so
    # override_sampler_script_slot patches all three ScriptSampler slots.
    args = dict(args)
    args.setdefault("sam3_use_steps", True)
    args.setdefault("sam3_use_sampler", True)
    args.setdefault("sam3_use_scheduler", True)
    args.setdefault("sam3_use_seed", True)

    scripts_runner, script_args_template = build_standalone_scripts_runner()
    if scripts_runner is None:
        print("[-] SAM3 Refine: t2i scripts runner not initialized; aborting.", file=sys.stderr)
        return []

    allow_huggingface = not getattr(shared.cmd_opts, "sam3_no_huggingface", False)

    shared.state.textinfo = "SAM3 Refine: running detection..."
    sam3_result = run_sam3_on_pil(
        image=image,
        prompt=args["sam3_prompt"],
        threshold=float(args["sam3_threshold"]),
        checkpoint_value=args["sam3_checkpoint"],
        device=args["sam3_device"],
        allow_huggingface=allow_huggingface,
        mask_dilation=int(args.get("sam3_mask_dilation", 0)),
        mask_hull=bool(args.get("sam3_mask_hull", False)),
    )

    if args.get("sam3_unload_after"):
        unload_sam3()
        print("[-] SAM3 Refine: model unloaded from VRAM (re-loads on next detection).", file=sys.stderr)

    masks_source = sam3_result.masks if args.get("sam3_mask_mode") == "Individual" else None
    masks = [sam3_result.mask] if not masks_source else masks_source
    if not masks or not any(np_any(m) for m in masks):
        print("[-] SAM3 Refine: detection returned an empty mask; nothing to do.", file=sys.stderr)
        return []

    # Diagnostic: per-mask coverage so the user can see if SAM3 caught a tiny
    # sliver vs the whole garment, plus the key inpaint knobs in effect.
    try:
        import numpy as np

        for i, m in enumerate(masks, start=1):
            arr = np.asarray(m)
            nonzero = int((arr > 127).sum()) if arr.size else 0
            total = int(arr.size) if arr.size else 1
            pct = 100.0 * nonzero / max(total, 1)
            print(
                f"[-] SAM3 Refine: mask {i}/{len(masks)} coverage {pct:.1f}% "
                f"({nonzero}/{total} pixels)",
                file=sys.stderr,
            )
    except Exception:
        pass
    fill_resolved = _resolve_inpainting_fill(args.get("sam3_inpainting_fill"))
    fill_label = {0: "fill", 1: "original", 2: "latent noise", 3: "latent nothing"}.get(fill_resolved, "?")
    print(
        f"[-] SAM3 Refine: inpaint settings — denoise={args.get('sam3_denoising_strength')}, "
        f"mask_blur={args.get('sam3_mask_blur')}, only_masked={args.get('sam3_inpaint_only_masked')}, "
        f"fill={fill_resolved} ({fill_label}), "
        f"steps={args.get('sam3_steps')}, cfg={args.get('sam3_cfg_scale')}, "
        f"sampler={args.get('sam3_sampler')!r}, scheduler={args.get('sam3_scheduler')!r}, "
        f"cn_enable={args.get('sam3_cn_enable')}, cn_model={args.get('sam3_cn_model')!r}, "
        f"cn_weight={args.get('sam3_cn_weight')}",
        file=sys.stderr,
    )

    prompt = copy_prompt(args.get("sam3_inpaint_prompt"), "")
    negative_prompt = copy_prompt(args.get("sam3_negative_prompt"), "")

    results: list[tuple[Image.Image, str]] = []
    shared.state.job_count += len(masks)

    with pause_total_tqdm():
        for index, mask in enumerate(masks, start=1):
            if shared.state.interrupted or shared.state.skipped:
                break
            shared.state.textinfo = f"SAM3 Refine: pass {index}/{len(masks)} — preparing"
            p2 = build_standalone_i2i(
                image,
                args,
                sd_model=sd_model,
                outpath_samples=outpath_samples,
                outpath_grids=outpath_grids,
                scripts_runner=scripts_runner,
                script_args=list(script_args_template),
            )
            if override_settings:
                p2.override_settings = dict(override_settings)
            p2.image_mask = mask
            p2.prompt = prompt
            p2.negative_prompt = negative_prompt
            inject_controlnet_unit(p2, args)
            override_sampler_script_slot(p2, args)
            shared.state.textinfo = f"SAM3 Refine: pass {index}/{len(masks)} — sampling"
            print(
                f"[-] SAM3 Refine pass {index}: p2 BEFORE process_images — "
                f"sampler_name={p2.sampler_name!r}, scheduler={p2.scheduler!r}, "
                f"steps={p2.steps}, cfg_scale={p2.cfg_scale}, "
                f"denoising_strength={p2.denoising_strength}, "
                f"inpainting_fill={getattr(p2, 'inpainting_fill', None)}, "
                f"inpaint_full_res={getattr(p2, 'inpaint_full_res', None)}, "
                f"image_mask_set={p2.image_mask is not None}",
                file=sys.stderr,
            )
            processed = None
            try:
                processed = process_images(p2)
            except Exception:
                error = traceback.format_exc()
                print(
                    f"[-] SAM3 Refine: pass {index} failed inside process_images "
                    f"(this is a webui-side error; common causes: OOM during VAE "
                    f"decode, an alwayson script adding images to x_samples_ddim, "
                    f"or batch/seed list mismatch). Skipping this mask.\n{error}",
                    file=sys.stderr,
                )
            finally:
                p2.close()

            if processed is None:
                continue
            if not processed.images:
                print(f"[-] SAM3 Refine: pass {index} returned no images.", file=sys.stderr)
                continue
            # Use the per-image infotext when available (Forge populates this
            # in processed.infotexts) — falls back to processed.info for older
            # paths / single-image batches.
            info_text = ""
            try:
                if getattr(processed, "infotexts", None):
                    info_text = processed.infotexts[0] or ""
                if not info_text:
                    info_text = getattr(processed, "info", "") or ""
            except Exception:
                info_text = ""
            # Show what process_images actually USED after fix_p_invalid_sampler_and_scheduler
            # and any script-side overrides. The "Sampler:" / "Schedule type:"
            # lines extracted from infotext are authoritative.
            import re as _re

            actual_sampler = _re.search(r"Sampler:\s*([^,\n]+)", info_text)
            actual_scheduler = _re.search(r"Schedule type:\s*([^,\n]+)", info_text)
            actual_steps = _re.search(r"Steps:\s*(\d+)", info_text)
            actual_cfg = _re.search(r"CFG scale:\s*([\d.]+)", info_text)
            print(
                f"[-] SAM3 Refine pass {index}: ACTUAL infotext — "
                f"sampler={actual_sampler.group(1).strip() if actual_sampler else '???'!r}, "
                f"scheduler={actual_scheduler.group(1).strip() if actual_scheduler else '???'!r}, "
                f"steps={actual_steps.group(1) if actual_steps else '???'}, "
                f"cfg={actual_cfg.group(1) if actual_cfg else '???'}",
                file=sys.stderr,
            )
            results.append((processed.images[0].convert("RGB"), info_text))
            print(f"[-] SAM3 Refine: pass {index} completed.", file=sys.stderr)

    return results


def np_any(mask) -> bool:
    import numpy as np

    return bool(np.any(np.asarray(mask)))


def run_inpaint_passes(
    p,
    image: Image.Image,
    masks: list,
    prompt: str,
    negative_prompt: str,
    args: dict[str, Any],
    cn_args: dict[str, Any] | None = None,
) -> Image.Image:
    """Run SAM3 inpaint passes sequentially, chaining the output of each pass
    into the next.

    Returns the final composited image (or the original image when no mask
    yields a valid pass).
    """
    current_image = image.convert("RGB") if isinstance(image, Image.Image) else image
    shared.state.job_count += len(masks)

    with pause_total_tqdm():
        for index, mask in enumerate(masks, start=1):
            if shared.state.interrupted or shared.state.skipped:
                break
            p2 = build_i2i(p, current_image, args)
            p2.image_mask = mask
            p2.init_images[0] = current_image
            p2.prompt = prompt
            p2.negative_prompt = negative_prompt
            inject_controlnet_unit(p2, cn_args)
            # P0 fix: ScriptSampler.setup would otherwise overwrite our
            # sampler/steps/scheduler back to the inherited t2i values,
            # silently negating the SAM3 panel's per-slot "Use separate ..."
            # toggles. The helper checks those flags itself and patches only
            # the opted-in slots.
            override_sampler_script_slot(p2, args)
            try:
                processed = process_images(p2)
            except Exception:
                error = traceback.format_exc()
                print(f"[-] SAM3: inpaint pass {index} failed:\n{error}", file=sys.stderr)
                raise
            finally:
                p2.close()

            if not processed.images:
                print(f"[-] SAM3: inpaint pass {index} returned no images.", file=sys.stderr)
                break
            print(f"[-] SAM3: inpaint pass {index} completed.", file=sys.stderr)
            current_image = processed.images[0].convert("RGB")

    return current_image
