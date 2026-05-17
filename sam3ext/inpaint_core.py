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


@contextmanager
def pause_total_tqdm():
    """Hide the outer total-tqdm bar while inner i2i passes run."""
    try:
        from modules.shared import opts
        with patch.dict(opts.data, {"multiple_tqdm": False}, clear=False):
            yield
    except Exception:
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
    if not hasattr(p, "scheduler"):
        return {}
    if args.get("sam3_use_sampler"):
        scheduler = args.get("sam3_scheduler", "Use same scheduler")
        if scheduler != "Use same scheduler":
            return {"scheduler": scheduler}
    return {"scheduler": getattr(p, "scheduler")}


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

    p2 = StableDiffusionProcessingImg2Img(
        init_images=[image],
        resize_mode=0,
        denoising_strength=float(args["sam3_denoising_strength"]),
        mask=None,
        mask_blur=int(args["sam3_mask_blur"]),
        inpainting_fill=1,
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
        seed=getattr(p, "seed", -1),
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
    p2.cached_c = [None, None]
    p2.cached_uc = [None, None]
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
    if sampler_name == "Use same sampler":
        sampler_name = "Euler a"
    noise_multiplier = float(args.get("sam3_noise_multiplier", 1.0))
    scheduler = args.get("sam3_scheduler", "Use same scheduler")
    version_args = {} if scheduler == "Use same scheduler" else {"scheduler": scheduler}

    p2 = StableDiffusionProcessingImg2Img(
        init_images=[image],
        resize_mode=0,
        denoising_strength=float(args["sam3_denoising_strength"]),
        mask=None,
        mask_blur=int(args["sam3_mask_blur"]),
        inpainting_fill=1,
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
        seed=-1,
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
    p2.cached_c = [None, None]
    p2.cached_uc = [None, None]
    p2.scripts = scripts_runner
    p2.script_args = script_args
    p2._sam3_inner = True
    p2.all_hr_prompts = [""]
    p2.all_hr_negative_prompts = [""]
    return p2


def build_standalone_scripts_runner():
    """Clone the t2i script runner with SAM3 itself stripped (avoids
    recursion). Returns ``(runner, script_args)`` ready to drop onto a
    standalone ``p2``, or ``(None, None)`` if the t2i runner has not been
    built yet (e.g., refine is invoked before any UI has rendered).

    ``scripts_txt2img.inputs`` is a list of *Gradio components*, not values —
    a real Generate click lets Gradio resolve those to current UI values. For
    our standalone path there is no such click, so we collapse each component
    to its default ``.value``. For most alwayson scripts that means "do
    nothing"; for ControlNet that means a list of disabled
    ``ControlNetUnit`` slots ready for ``inject_controlnet_unit`` to overwrite
    slot 0.
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

    return runner, script_args


def run_sam3_refine(
    image: Image.Image,
    args: dict[str, Any],
    *,
    sd_model,
    outpath_samples: str,
    outpath_grids: str,
) -> list[Image.Image]:
    """Standalone equivalent of ``run_inpaint_passes`` for the post-generation
    Refine panel: no ``p`` to inherit from, runs once per detected mask, and
    returns the resulting images (one per mask) instead of chaining them.

    Returns ``[]`` when SAM3 finds nothing or every pass is interrupted.
    """
    from .core import run_sam3_on_pil

    scripts_runner, script_args_template = build_standalone_scripts_runner()
    if scripts_runner is None:
        print("[-] SAM3 Refine: t2i scripts runner not initialized; aborting.", file=sys.stderr)
        return []

    from modules import shared as _shared
    allow_huggingface = not getattr(_shared.cmd_opts, "sam3_no_huggingface", False)

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
        from .core import unload_sam3

        unload_sam3()
        print("[-] SAM3 Refine: model unloaded from VRAM (re-loads on next detection).", file=sys.stderr)

    masks_source = sam3_result.masks if args.get("sam3_mask_mode") == "Individual" else None
    masks = [sam3_result.mask] if not masks_source else masks_source
    if not masks or not any(np_any(m) for m in masks):
        print("[-] SAM3 Refine: detection returned an empty mask; nothing to do.", file=sys.stderr)
        return []

    prompt = copy_prompt(args.get("sam3_inpaint_prompt"), "")
    negative_prompt = copy_prompt(args.get("sam3_negative_prompt"), "")

    results: list[Image.Image] = []
    _shared.state.job_count += len(masks)

    with pause_total_tqdm():
        for index, mask in enumerate(masks, start=1):
            if _shared.state.interrupted or _shared.state.skipped:
                break
            p2 = build_standalone_i2i(
                image,
                args,
                sd_model=sd_model,
                outpath_samples=outpath_samples,
                outpath_grids=outpath_grids,
                scripts_runner=scripts_runner,
                script_args=list(script_args_template),
            )
            p2.image_mask = mask
            p2.prompt = prompt
            p2.negative_prompt = negative_prompt
            inject_controlnet_unit(p2, args)
            try:
                from modules.processing import process_images as _process_images

                processed = _process_images(p2)
            except Exception:
                error = traceback.format_exc()
                print(f"[-] SAM3 Refine: pass {index} failed:\n{error}", file=sys.stderr)
                raise
            finally:
                p2.close()

            if not processed.images:
                print(f"[-] SAM3 Refine: pass {index} returned no images.", file=sys.stderr)
                continue
            results.append(processed.images[0].convert("RGB"))
            print(f"[-] SAM3 Refine: pass {index} completed.", file=sys.stderr)

    return results


def np_any(mask) -> bool:
    import numpy as _np

    return bool(_np.any(_np.asarray(mask)))


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
