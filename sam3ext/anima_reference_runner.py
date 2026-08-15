"""Forge adapter for Feature 6: Anima Character Reference / ReStyler.

The geometry and prompt rules live in :mod:`anima_reference_core`.  This
module is the deliberately thin Forge-facing adapter: it translates one
validated request into a standalone img2img job, temporarily enables Forge
Neo's native Anima reference path, and returns both the full split canvas and
the cropped target panel.

No Forge core file is modified.  The two fixed invariants are intentional:

* batch size is one because Anima's reference latent is process-global;
* ``inpaint_full_res`` is false so the model sees reference and target panels
  together.  Cropping to only the mask would remove the reference panel.

Everything else that changes the workflow is represented on
``ReferenceGenerationRequest`` and is exposed by the Feature 6 UI.
"""
from __future__ import annotations

import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from PIL import Image

from .anima_reference_core import (
    PreparedReferenceCanvas,
    ReferenceCanvasConfig,
    compose_reference_prompt,
    crop_reference_result,
    prepare_reference_canvas,
)


@dataclass(frozen=True)
class ReferenceGenerationRequest:
    """Complete user-editable state for one Feature 6 run."""

    reference_image: Image.Image
    canvas: ReferenceCanvasConfig = field(default_factory=ReferenceCanvasConfig)

    prompt: str = ""
    negative_prompt: str = ""
    prefix_enabled: bool = True
    prefix_text: str = "split screen, multiple views"
    prefix_strength: float = 1.2
    extra_prefix: str = ""
    extra_suffix: str = ""

    edit_lora_enabled: bool = True
    edit_lora_name: str = "AnimeEditV2"
    edit_lora_strength: float = 0.72
    extend_lora_enabled: bool = True
    extend_lora_name: str = "Extend Image (Anima Edit) v1"
    extend_lora_strength: float = 0.4
    require_enabled_loras: bool = True

    checkpoint_override: str = "Use current"
    override_additional_modules: bool = False
    additional_modules: tuple[str, ...] = ()

    steps: int = 30
    cfg_scale: float = 5.0
    shift: float = 3.0
    sampler: str = "Euler a"
    scheduler: str = "Simple"
    denoising_strength: float = 1.0
    resize_mode: str = "Just Resize"
    inpainting_fill: str = "latent noise"
    mask_blur: int = 0
    mask_round: bool = True
    mask_invert: bool = False
    inpainting_mask_weight: float = 1.0
    initial_noise_multiplier: float = 1.0

    eta: float = 1.0
    s_min_uncond: float = 0.0
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = 0.0
    s_noise: float = 1.0

    seed: int = -1
    seed_step: int = 1
    candidate_count: int = 1
    restore_faces: bool = False
    native_reference_enabled: bool = True

    save_target: bool = True
    save_generated_canvas: bool = False
    save_input_canvas: bool = False
    save_mask: bool = False

    def validate(self) -> "ReferenceGenerationRequest":
        self.canvas.validate()
        image = self.reference_image
        if image is None or image.width < 1 or image.height < 1:
            raise ValueError("A non-empty reference image is required.")
        if not 1 <= int(self.steps) <= 1000:
            raise ValueError("Steps must be between 1 and 1000.")
        if not 0.0 <= float(self.cfg_scale) <= 100.0:
            raise ValueError("CFG scale must be between 0 and 100.")
        if not 0.0 <= float(self.shift) <= 100.0:
            raise ValueError("Shift must be between 0 and 100.")
        if not -100.0 <= float(self.prefix_strength) <= 100.0:
            raise ValueError("Prefix strength must be between -100 and 100.")
        if not 0.0 <= float(self.denoising_strength) <= 1.0:
            raise ValueError("Denoising strength must be between 0 and 1.")
        if not 0 <= int(self.mask_blur) <= 1024:
            raise ValueError("Mask blur must be between 0 and 1024.")
        if not 0.0 <= float(self.inpainting_mask_weight) <= 1.0:
            raise ValueError("Inpainting mask weight must be between 0 and 1.")
        if not 0.0 <= float(self.initial_noise_multiplier) <= 10.0:
            raise ValueError("Initial noise multiplier must be between 0 and 10.")
        if not 0.0 <= float(self.eta) <= 10.0:
            raise ValueError("Eta must be between 0 and 10.")
        if not 0.0 <= float(self.s_min_uncond) <= 1000.0:
            raise ValueError("s_min_uncond must be between 0 and 1000.")
        if not 0.0 <= float(self.s_churn) <= 1000.0:
            raise ValueError("s_churn must be between 0 and 1000.")
        if not 0.0 <= float(self.s_tmin) <= 1000.0:
            raise ValueError("s_tmin must be between 0 and 1000.")
        if not 0.0 <= float(self.s_tmax) <= 10000.0:
            raise ValueError("s_tmax must be between 0 and 10000.")
        if not 0.0 <= float(self.s_noise) <= 10.0:
            raise ValueError("s_noise must be between 0 and 10.")
        if not 1 <= int(self.candidate_count) <= 64:
            raise ValueError("Candidate count must be between 1 and 64.")
        if not str(self.sampler or "").strip():
            raise ValueError("Sampler is required.")
        if not str(self.scheduler or "").strip():
            raise ValueError("Scheduler is required.")
        if self.resize_mode not in (
            "Just Resize",
            "Crop and Resize",
            "Resize and Fill",
        ):
            raise ValueError(f"Unsupported resize mode: {self.resize_mode!r}")
        if self.inpainting_fill not in (
            "fill",
            "original",
            "latent noise",
            "latent nothing",
        ):
            raise ValueError(
                f"Unsupported masked content: {self.inpainting_fill!r}"
            )
        if self.edit_lora_enabled and not str(self.edit_lora_name).strip():
            raise ValueError("Edit LoRA is enabled but its name is empty.")
        if self.extend_lora_enabled and not str(
            self.extend_lora_name
        ).strip():
            raise ValueError("Extend LoRA is enabled but its name is empty.")
        for name, value in (
            ("Edit LoRA strength", self.edit_lora_strength),
            ("Extend LoRA strength", self.extend_lora_strength),
        ):
            if not -10.0 <= float(value) <= 10.0:
                raise ValueError(f"{name} must be between -10 and 10.")
        return self

    @property
    def internal_prompt(self) -> str:
        return compose_reference_prompt(
            self.prompt,
            prefix_enabled=self.prefix_enabled,
            prefix_text=self.prefix_text,
            prefix_strength=self.prefix_strength,
            edit_lora_enabled=self.edit_lora_enabled,
            edit_lora_name=self.edit_lora_name,
            edit_lora_strength=self.edit_lora_strength,
            extend_lora_enabled=self.extend_lora_enabled,
            extend_lora_name=self.extend_lora_name,
            extend_lora_strength=self.extend_lora_strength,
            extra_prefix=self.extra_prefix,
            extra_suffix=self.extra_suffix,
        )


@dataclass(frozen=True)
class ReferenceGenerationOutput:
    target_image: Image.Image
    generated_canvas: Image.Image
    infotext: str
    seed: int


@dataclass(frozen=True)
class ReferenceGenerationResult:
    prepared: PreparedReferenceCanvas
    outputs: tuple[ReferenceGenerationOutput, ...]


def candidate_seed(base_seed: int, seed_step: int, index: int) -> int:
    """Return a deterministic candidate seed, preserving Forge's ``-1``."""

    seed = int(base_seed)
    if seed < 0:
        return -1
    return seed + int(seed_step) * int(index)


def build_processing_args(
    request: ReferenceGenerationRequest,
    *,
    seed: int,
) -> dict[str, Any]:
    """Translate the deep request model to the existing standalone i2i seam."""

    return {
        "sam3_use_inpaint_width_height": False,
        "sam3_inpaint_width": request.canvas.output_width,
        "sam3_inpaint_height": request.canvas.output_height,
        "sam3_steps": int(request.steps),
        "sam3_cfg_scale": float(request.cfg_scale),
        "sam3_sampler": str(request.sampler),
        "sam3_scheduler": str(request.scheduler),
        "sam3_seed": int(seed),
        "sam3_noise_multiplier": float(request.initial_noise_multiplier),
        "sam3_inpainting_fill": str(request.inpainting_fill),
        "sam3_resize_mode": str(request.resize_mode),
        "sam3_mask_invert": bool(request.mask_invert),
        "sam3_denoising_strength": float(request.denoising_strength),
        "sam3_mask_blur": int(request.mask_blur),
        # Invariant: only-masked preprocessing would crop away the reference.
        "sam3_inpaint_only_masked": False,
        "sam3_inpaint_only_masked_padding": 0,
        "sam3_restore_face": bool(request.restore_faces),
    }


def _is_anima_engine(model: Any) -> bool:
    if model is None:
        return False
    cls = type(model)
    return (
        cls.__name__.lower() == "anima"
        or cls.__module__.endswith(".anima")
        or hasattr(model, "text_processing_engine_anima")
    )


def _clear_reference_state(
    model: Any = None,
    *,
    dynamic_args_obj: Any = None,
) -> None:
    try:
        if dynamic_args_obj is None:
            from backend.args import dynamic_args

            dynamic_args_obj = dynamic_args
        dynamic_args_obj.ref_latents.clear()
        dynamic_args_obj.is_referencing = False
    except Exception:
        pass
    try:
        if model is not None and hasattr(model, "clear_references"):
            model.clear_references()
    except Exception:
        pass


@contextmanager
def native_anima_reference(
    enabled: bool,
    model: Any = None,
    *,
    opts_data: dict[str, Any] | None = None,
    dynamic_args_obj: Any = None,
) -> Iterator[None]:
    """Temporarily set Forge's Anima reference option without saving config."""

    using_forge_opts = opts_data is None
    if using_forge_opts:
        from modules import shared

        opts_data = shared.opts.data
    missing = object()
    previous = opts_data.get("anima_do_reference", missing)
    _clear_reference_state(
        model,
        dynamic_args_obj=dynamic_args_obj,
    )
    opts_data["anima_do_reference"] = bool(enabled)
    try:
        yield
    finally:
        active_model = model
        if using_forge_opts:
            try:
                from modules import shared

                active_model = getattr(shared, "sd_model", model)
            except Exception:
                pass
        _clear_reference_state(
            active_model,
            dynamic_args_obj=dynamic_args_obj,
        )
        if previous is missing:
            opts_data.pop("anima_do_reference", None)
        else:
            opts_data["anima_do_reference"] = previous


def _infotext_from_processed(processed: Any) -> str:
    try:
        infotexts = list(getattr(processed, "infotexts", None) or [])
        if infotexts:
            return str(infotexts[0] or "")
        return str(getattr(processed, "info", "") or "")
    except Exception:
        return ""


def _save_image(
    image: Image.Image,
    *,
    outpath: str,
    seed: int,
    prompt: str,
    infotext: str,
    processing: Any,
    suffix: str,
) -> None:
    from modules import images, shared

    images.save_image(
        image,
        outpath,
        "",
        seed,
        prompt,
        getattr(shared.opts, "samples_format", "png"),
        info=infotext,
        p=processing,
        suffix=suffix,
    )


def _extra_generation_params(
    request: ReferenceGenerationRequest,
    prepared: PreparedReferenceCanvas,
) -> dict[str, Any]:
    return {
        "SAM3 Feature": "6 - Anima Character Reference",
        "Anima Native Reference": bool(request.native_reference_enabled),
        "Reference placement": request.canvas.placement,
        "Reference target size": (
            f"{request.canvas.output_width}x{request.canvas.output_height}"
        ),
        "Reference composite size": (
            f"{prepared.canvas.width}x{prepared.canvas.height}"
        ),
        "Reference target region scale": request.canvas.target_region_scale,
        "Reference composite megapixels": request.canvas.composite_megapixels,
        "Reference dimension multiple": request.canvas.dimension_multiple,
        "Reference resize filter": request.canvas.resize_filter,
        "Reference mask overlap": request.canvas.mask_overlap,
        "Reference target color": request.canvas.target_color,
        "Reference matte color": request.canvas.reference_matte_color,
        "Reference prefix": (
            f"({request.prefix_text}:{request.prefix_strength:g})"
            if request.prefix_enabled
            else "disabled"
        ),
        "Reference Edit LoRA": (
            f"{request.edit_lora_name}:{request.edit_lora_strength:g}"
            if request.edit_lora_enabled
            else "disabled"
        ),
        "Reference Extend LoRA": (
            f"{request.extend_lora_name}:{request.extend_lora_strength:g}"
            if request.extend_lora_enabled
            else "disabled"
        ),
        "Reference mask round": bool(request.mask_round),
        "Reference mask invert": bool(request.mask_invert),
        "Reference mask weight": request.inpainting_mask_weight,
        "Reference seed step": request.seed_step,
    }


def run_anima_reference(
    request: ReferenceGenerationRequest,
    *,
    sd_model: Any = None,
    outpath_samples: str | None = None,
    outpath_grids: str | None = None,
) -> ReferenceGenerationResult:
    """Run Feature 6 through Forge's native Anima img2img/reference path."""

    request.validate()
    prepared = prepare_reference_canvas(request.reference_image, request.canvas)

    from modules import shared
    from modules.processing import process_images

    from .inpaint_core import build_standalone_i2i, pause_total_tqdm

    model = sd_model or getattr(shared, "sd_model", None)
    if model is None:
        raise RuntimeError("No Forge model is loaded.")

    checkpoint_override = str(request.checkpoint_override or "Use current")
    if checkpoint_override == "Use current" and not _is_anima_engine(model):
        raise RuntimeError(
            "Feature 6 requires an Anima model. Load Anima or choose an "
            "Anima checkpoint override."
        )

    sample_path = outpath_samples or getattr(
        shared.opts, "outdir_txt2img_samples", "outputs/txt2img-images"
    )
    grid_path = outpath_grids or getattr(
        shared.opts, "outdir_txt2img_grids", "outputs/txt2img-grids"
    )

    override_settings: dict[str, Any] = {}
    if checkpoint_override not in ("", "Use current"):
        override_settings["sd_model_checkpoint"] = checkpoint_override
    if request.override_additional_modules:
        override_settings["forge_additional_modules"] = list(
            request.additional_modules
        )

    outputs: list[ReferenceGenerationOutput] = []
    candidate_count = int(request.candidate_count)
    if shared.state.job_count < 0:
        shared.state.job_count = candidate_count
    else:
        shared.state.job_count += candidate_count
    shared.state.job = "Anima Character Reference"

    try:
        with native_anima_reference(request.native_reference_enabled, model):
            with pause_total_tqdm():
                for index in range(candidate_count):
                    if shared.state.interrupted or shared.state.skipped:
                        break
                    seed = candidate_seed(request.seed, request.seed_step, index)
                    shared.state.job = (
                        f"Anima Reference {index + 1}/{candidate_count}"
                    )
                    shared.state.textinfo = (
                        f"Anima Reference: candidate {index + 1}/"
                        f"{candidate_count} - preparing"
                    )

                    p2 = build_standalone_i2i(
                        prepared.canvas,
                        build_processing_args(request, seed=seed),
                        sd_model=model,
                        outpath_samples=sample_path,
                        outpath_grids=grid_path,
                        # A clean runner is intentional.  Third-party scripts
                        # can hold stale t2i state or inject another reference.
                        scripts_runner=None,
                        script_args=None,
                    )
                    p2.prompt = request.internal_prompt
                    p2.negative_prompt = str(request.negative_prompt or "")
                    p2.image_mask = prepared.mask
                    p2.distilled_cfg_scale = float(request.shift)
                    p2.mask_round = bool(request.mask_round)
                    p2.inpainting_mask_weight = float(
                        request.inpainting_mask_weight
                    )
                    p2.eta = float(request.eta)
                    p2.s_min_uncond = float(request.s_min_uncond)
                    p2.s_churn = float(request.s_churn)
                    p2.s_tmin = float(request.s_tmin)
                    p2.s_tmax = float(request.s_tmax)
                    p2.s_noise = float(request.s_noise)
                    p2.override_settings = dict(override_settings)
                    # Save the cropped target ourselves; otherwise Forge
                    # would save the temporary split-screen canvas.
                    p2.do_not_save_samples = True
                    p2.do_not_save_grid = True
                    p2.extra_generation_params.update(
                        _extra_generation_params(request, prepared)
                    )

                    processed = None
                    try:
                        shared.state.textinfo = (
                            f"Anima Reference: candidate {index + 1}/"
                            f"{candidate_count} - sampling"
                        )
                        processed = process_images(p2)
                        active_model = getattr(shared, "sd_model", None)
                        if not _is_anima_engine(active_model):
                            raise RuntimeError(
                                "The selected checkpoint did not load as an "
                                "Anima model; reference output was discarded."
                            )
                        if processed is None or not processed.images:
                            raise RuntimeError(
                                "Forge returned no image for the reference job."
                            )

                        generated = processed.images[0].convert("RGB")
                        target = crop_reference_result(
                            generated,
                            prepared,
                            resize_filter=request.canvas.resize_filter,
                        )
                        infotext = _infotext_from_processed(processed)
                        actual_seed = seed
                        try:
                            if getattr(p2, "all_seeds", None):
                                actual_seed = int(p2.all_seeds[0])
                        except Exception:
                            pass

                        if request.save_target:
                            _save_image(
                                target,
                                outpath=sample_path,
                                seed=actual_seed,
                                prompt=request.internal_prompt,
                                infotext=infotext,
                                processing=p2,
                                suffix="-anima-reference",
                            )
                        if request.save_generated_canvas:
                            _save_image(
                                generated,
                                outpath=sample_path,
                                seed=actual_seed,
                                prompt=request.internal_prompt,
                                infotext=infotext,
                                processing=p2,
                                suffix="-anima-reference-canvas",
                            )
                        if request.save_input_canvas and index == 0:
                            _save_image(
                                prepared.canvas,
                                outpath=sample_path,
                                seed=actual_seed,
                                prompt=request.internal_prompt,
                                infotext=infotext,
                                processing=p2,
                                suffix="-anima-reference-input",
                            )
                        if request.save_mask and index == 0:
                            _save_image(
                                prepared.mask.convert("RGB"),
                                outpath=sample_path,
                                seed=actual_seed,
                                prompt=request.internal_prompt,
                                infotext=infotext,
                                processing=p2,
                                suffix="-anima-reference-mask",
                            )

                        outputs.append(
                            ReferenceGenerationOutput(
                                target_image=target,
                                generated_canvas=generated,
                                infotext=infotext,
                                seed=actual_seed,
                            )
                        )
                        shared.state.textinfo = (
                            f"Anima Reference: candidate {index + 1}/"
                            f"{candidate_count} - done"
                        )
                    except Exception:
                        print(
                            "[-] Anima Character Reference failed:\n"
                            f"{traceback.format_exc()}",
                            file=sys.stderr,
                        )
                        raise
                    finally:
                        p2.close()
                        # Anima decode normally clears this already.  Keep a
                        # defensive per-candidate dynamic clear, but do not
                        # call model.clear_references() here: Forge's base
                        # implementation also flushes the allocator and would
                        # add avoidable stalls between candidates.
                        _clear_reference_state()
    finally:
        shared.state.textinfo = ""

    return ReferenceGenerationResult(
        prepared=prepared,
        outputs=tuple(outputs),
    )
