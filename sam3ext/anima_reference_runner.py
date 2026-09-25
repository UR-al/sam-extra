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

``ReferenceGenerationRequest`` carries the whole workflow; the Feature 6 UI
exposes the main inputs plus an Expert section for everything else.

Defaults reproduce the public Anima ReStyler v1.2 workflow
(animaRestyler_v12.json): a solid ``#000000`` target panel kept as-is
(``inpainting_fill="original"``) at denoise 1.0, AnimeEditV2 at 0.72, the
Extend LoRA bypassed, Euler a / Simple / 30 steps / CFG 5.
"""
from __future__ import annotations

import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Iterator

from PIL import Image

from .anima_ipa.options import IpaOptions
from .anima_reference_core import (
    PreparedReferenceCanvas,
    ReferenceCanvasConfig,
    compose_reference_prompt,
    crop_reference_result,
    prepare_reference_canvas,
)
from .anima_reference_recipe import MODE_I2I, MODE_IPA, MODES


@dataclass(frozen=True)
class ReferenceGenerationRequest:
    """Complete state for one Feature 6 run (defaults = ReStyler v1.2)."""

    reference_image: Image.Image
    canvas: ReferenceCanvasConfig = field(default_factory=ReferenceCanvasConfig)

    prompt: str = ""
    negative_prompt: str = ""
    prefix_enabled: bool = True
    prefix_text: str = "split screen, multiple views"
    prefix_strength: float = 1.2

    edit_lora_enabled: bool = True
    edit_lora_name: str = "AnimeEditV2"
    edit_lora_strength: float = 0.72
    extend_lora_enabled: bool = False
    extend_lora_name: str = "Extend Image (Anima Edit) v1"
    extend_lora_strength: float = 0.4

    steps: int = 30
    cfg_scale: float = 5.0
    shift: float = 3.0
    sampler: str = "Euler a"
    scheduler: str = "Simple"
    denoising_strength: float = 1.0
    inpainting_fill: str = "original"
    mask_blur: int = 0
    initial_noise_multiplier: float = 1.0
    # None keeps Forge's own Settings value instead of overriding it.
    eta: float | None = None
    s_min_uncond: float | None = None

    seed: int = -1
    seed_step: int = 1
    candidate_count: int = 2
    native_reference_enabled: bool = True

    # "i2i" = 기존 split-screen 캔버스, "ipa" = IP-Adapter 주입
    mode: str = MODE_I2I
    ipa: IpaOptions = field(default_factory=IpaOptions)

    # Recorded in the infotext so GPU A/B runs can be compared.
    keep_scope: str = "identity"
    sampling_source: str = "recipe"

    save_target: bool = True
    save_generated_canvas: bool = False
    save_input_canvas: bool = False
    save_mask: bool = False

    def validate(self) -> "ReferenceGenerationRequest":
        self.canvas.validate()
        if self.mode not in MODES:
            raise ValueError(f"Unsupported reference mode: {self.mode!r}")
        if self.mode == MODE_IPA:
            self.ipa.validate()
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
        if not 0.0 <= float(self.initial_noise_multiplier) <= 10.0:
            raise ValueError("Initial noise multiplier must be between 0 and 10.")
        if self.eta is not None and not 0.0 <= float(self.eta) <= 10.0:
            raise ValueError("Eta must be between 0 and 10.")
        if self.s_min_uncond is not None and not (
            0.0 <= float(self.s_min_uncond) <= 1000.0
        ):
            raise ValueError("s_min_uncond must be between 0 and 1000.")
        if not 1 <= int(self.candidate_count) <= 64:
            raise ValueError("Candidate count must be between 1 and 64.")
        if not str(self.sampler or "").strip():
            raise ValueError("Sampler is required.")
        if not str(self.scheduler or "").strip():
            raise ValueError("Scheduler is required.")
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
    def sampling_prompt(self) -> str:
        """샘플러에 실제로 들어가는 프롬프트.

        IPA 는 캔버스가 없으므로 split-screen 접두사도 편집 LoRA 도 붙이지 않는다.
        """
        if self.mode == MODE_IPA:
            return str(self.prompt or "")
        return self.internal_prompt

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
        )


@dataclass(frozen=True)
class ReferenceGenerationOutput:
    target_image: Image.Image
    generated_canvas: Image.Image
    infotext: str
    seed: int


@dataclass(frozen=True)
class ReferenceGenerationResult:
    # IPA 모드에는 캔버스도 마스크도 없으므로 ``prepared`` 가 None 이다.
    prepared: PreparedReferenceCanvas | None = None
    outputs: tuple[ReferenceGenerationOutput, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


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
    """Translate the request to the existing standalone i2i seam."""

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
        # The canvas already has the inference size, so resizing is a no-op.
        "sam3_resize_mode": "Just Resize",
        # Inverting would regenerate the reference and crop an empty panel.
        "sam3_mask_invert": False,
        "sam3_denoising_strength": float(request.denoising_strength),
        "sam3_mask_blur": int(request.mask_blur),
        # Invariant: only-masked preprocessing would crop away the reference.
        "sam3_inpaint_only_masked": False,
        "sam3_inpaint_only_masked_padding": 0,
        "sam3_restore_face": False,
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


def _model_block_count(model: Any) -> int | None:
    try:
        diffusion = model.forge_objects.unet.model.diffusion_model
    except AttributeError:
        return None
    blocks = getattr(diffusion, "blocks", None)
    try:
        return len(blocks) if blocks is not None else None
    except TypeError:
        return None


def _reset_image_stitch_cache() -> None:
    """Make ImageStitch re-encode its references on the next txt2img run.

    Feature 6 clears the process-global reference latents; ImageStitch would
    otherwise see an unchanged parameter cache and skip re-encoding.
    """

    try:
        from modules import scripts
    except Exception:
        return
    for data in list(getattr(scripts, "scripts_data", None) or []):
        script_class = getattr(data, "script_class", None)
        if (
            script_class is not None
            and script_class.__name__ == "ImageStitch"
            and hasattr(script_class, "cached_parameters")
        ):
            script_class.cached_parameters = None


def _panel_diagnostics(
    request: "ReferenceGenerationRequest",
    prepared: PreparedReferenceCanvas,
) -> tuple[tuple[int, int], float]:
    left, top, right, bottom = prepared.target_box
    panel = (right - left, bottom - top)
    upscale = max(
        request.canvas.output_width / panel[0],
        request.canvas.output_height / panel[1],
    )
    return panel, round(upscale, 2)


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
    *,
    model_blocks: int | None,
) -> dict[str, Any]:
    panel, upscale = _panel_diagnostics(request, prepared)
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
        "Reference seed step": request.seed_step,
        "Reference masked content": request.inpainting_fill,
        "Reference target panel": f"{panel[0]}x{panel[1]}",
        "Reference upscale": f"{upscale:.2f}",
        "Reference keep": request.keep_scope,
        "Reference sampling": request.sampling_source,
        "Reference model blocks": (
            str(model_blocks) if model_blocks is not None else "unknown"
        ),
    }


# Only used for legacy v1 checkpoints; bundled v2 checkpoints ignore it.
# Matches DEFAULT_ADAPTER in scripts/anima_3_8b.py.
_ANIMA38_ADAPTER = "Anima-3.8B-expanded_adapter.safetensors"


def _anima38_runtime() -> tuple[Any, str | None]:
    try:
        from .anima38.runtime import shared_runtime

        return shared_runtime(), None
    except Exception as exc:  # pragma: no cover - depends on Forge/torch
        return None, f"{type(exc).__name__}: {exc}"


def _anima38_encoder_present() -> bool:
    """Return whether Qwen3.5's text encoder is actually on disk.

    ``Anima3BRuntime.install()`` never loads Qwen3.5 itself: it is only
    reached lazily from ``process_images`` -> ``get_learned_conditioning``.
    So a v2 bundle without the encoder would otherwise install "successfully"
    and only fail deep inside sampling.
    """

    try:
        from .anima38.files import qwen35_models

        return bool(qwen35_models())
    except Exception:
        return False


def _install_anima38(
    runtime: Any,
    error: str | None,
    processing: Any,
    model: Any,
) -> str:
    """Install the 3.8B Semantic Connector for bundled v2 checkpoints.

    Feature 6 runs without a scripts runner, so the Anima38 script's own
    process_batch never fires; this does the same install for the job.
    """

    if runtime is None:
        return f"unavailable ({error})" if error else "unavailable"
    try:
        if not runtime.is_v2_bundle(model):
            return "not a 3.8B v2 bundle"
    except Exception as exc:
        print(
            f"[-] Feature 6: Anima 3.8B check failed:\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return f"check failed ({type(exc).__name__})"
    if not _anima38_encoder_present():
        return (
            "missing encoder (qwen35_4b.safetensors not found in "
            "models/text_encoder)"
        )
    try:
        runtime.install(processing, _ANIMA38_ADAPTER, 1.0, None)
    except FileNotFoundError as exc:
        _restore_anima38(runtime, processing)
        return f"missing encoder ({exc})"
    except Exception as exc:
        print(
            f"[-] Feature 6: Anima 3.8B install failed:\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        _restore_anima38(runtime, processing)
        return f"install failed ({type(exc).__name__}: {exc})"
    return "v2 bundle"


def _restore_anima38(runtime: Any, processing: Any) -> None:
    try:
        runtime.restore(processing)
    except Exception:
        print(
            f"[-] Feature 6: Anima 3.8B restore failed:\n{traceback.format_exc()}",
            file=sys.stderr,
        )


# IP-Adapter 방식에 3.8B v2 커넥터를 설치할지 — scripts/anima_3_8b.py 가 같은 이름으로
# 'SAM Extra Anima 3.8B' 설정 섹션에 등록한다. 기본은 켬 — 이어붙이기 방식·txt2img 와 같은 조건.
# 끄면 0.21 까지처럼 0.6B 조건으로 샘플링한다(결과가 바뀜).
OPT_IPA_ANIMA38 = "sam3_anima38_reference_ipa"
DEFAULT_IPA_ANIMA38 = True
_ANIMA38_RUNTIME_MODULE = f"{__package__}.anima38.runtime"
# 설정이 꺼져 있을 때 diagnostics 에 남기는 값 (infotext 에는 아무것도 남기지 않는다 — 예전 결과와 같다)
ANIMA38_IPA_OFF = "off"


def _ipa_anima38_wanted(request: ReferenceGenerationRequest, opts: Any) -> bool:
    """요청이 정했으면 그 값, 아니면 Forge 설정(기본 켬)."""
    explicit = request.ipa.anima38_connector
    if explicit is not None:
        return bool(explicit)
    try:
        return bool(getattr(opts, OPT_IPA_ANIMA38, DEFAULT_IPA_ANIMA38))
    except Exception:
        return DEFAULT_IPA_ANIMA38


def _release_foreign_anima38(runtime: Any, processing: Any) -> None:
    """3.8B 설치 순서의 첫 단계 — 다른 생성이 예외로 끝나 남긴 설치를 내린다.

    ``scripts/anima_3_8b.py`` 의 ``_release_stale_install`` 과 같은 공용 함수
    (``Anima3BRuntime.release_foreign_install``)를 쓴다. 커넥터를 안 쓰는 잡도 부른다 —
    안 그러면 txt2img 가 남긴 v2 조건 래퍼로 이 잡의 조건이 만들어진다(스크립트의 Bypass 와 같은 규칙).
    그때는 무거운 임포트 없이 이미 떠 있는 공용 런타임만 본다.
    """
    if runtime is None:
        module = sys.modules.get(_ANIMA38_RUNTIME_MODULE)
        runtime = getattr(module, "_SHARED_RUNTIME", None)
    release = getattr(runtime, "release_foreign_install", None)
    if release is None:
        return

    def report(exc: Exception) -> None:
        print(
            "[-] Feature 6: stale Anima 3.8B restore failed — "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

    try:
        release(processing, on_error=report)
    except Exception as exc:  # pragma: no cover - 복구가 실패해도 이번 잡은 진행한다
        report(exc)


def _ipa_session(sd_model, image, options):
    """IP-Adapter 주입 세션. 테스트에서 통째로 갈아끼울 수 있게 얇게 감싼다."""
    from .anima_ipa.runtime import shared_runtime

    return shared_runtime().session(sd_model, image, options)


def run_anima_reference(
    request: ReferenceGenerationRequest,
    *,
    sd_model: Any = None,
    outpath_samples: str | None = None,
    outpath_grids: str | None = None,
) -> ReferenceGenerationResult:
    """Run Feature 6 through Forge's native Anima img2img/reference path."""

    request.validate()
    if request.mode == MODE_IPA:
        return _run_ipa_reference(
            request,
            sd_model=sd_model,
            outpath_samples=outpath_samples,
            outpath_grids=outpath_grids,
        )
    prepared = prepare_reference_canvas(request.reference_image, request.canvas)

    from modules import shared
    from modules.processing import process_images

    from .inpaint_core import build_standalone_i2i, pause_total_tqdm

    if sd_model is None:
        # Forge loads the selected checkpoint lazily in process_images: right
        # after a restart shared.sd_model is a FakeInitialModel placeholder, and
        # after a dropdown change it is still the previous model. Do that load
        # first so the Anima check sees the selected checkpoint (it returns
        # immediately when that checkpoint is already loaded).
        from modules import sd_models

        sd_models.forge_model_reload()

    model = sd_model or getattr(shared, "sd_model", None)
    if model is None:
        raise RuntimeError("No Forge model is loaded.")

    if not _is_anima_engine(model):
        raise RuntimeError("Feature 6 requires an Anima model. Load Anima first.")

    model_blocks = _model_block_count(model)
    interrupted = False
    anima38_runtime, anima38_error = _anima38_runtime()
    anima38_label = "not attempted"

    sample_path = outpath_samples or getattr(
        shared.opts, "outdir_txt2img_samples", "outputs/txt2img-images"
    )
    grid_path = outpath_grids or getattr(
        shared.opts, "outdir_txt2img_grids", "outputs/txt2img-grids"
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
                        interrupted = True
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
                    if request.eta is not None:
                        p2.eta = float(request.eta)
                    if request.s_min_uncond is not None:
                        p2.s_min_uncond = float(request.s_min_uncond)
                    # Save the cropped target ourselves; otherwise Forge
                    # would save the temporary split-screen canvas.
                    p2.do_not_save_samples = True
                    p2.do_not_save_grid = True
                    p2.extra_generation_params.update(
                        _extra_generation_params(
                            request, prepared, model_blocks=model_blocks
                        )
                    )
                    anima38_label = _install_anima38(
                        anima38_runtime, anima38_error, p2, model
                    )
                    anima38_installed = anima38_label == "v2 bundle"
                    p2.extra_generation_params["Reference Anima38"] = anima38_label

                    processed = None
                    try:
                        shared.state.textinfo = (
                            f"Anima Reference: candidate {index + 1}/"
                            f"{candidate_count} - sampling"
                        )
                        processed = process_images(p2)
                        if shared.state.interrupted or shared.state.skipped:
                            # A stopped job returns a half-denoised canvas;
                            # never save it or show it as a candidate.
                            interrupted = True
                            break
                        active_model = getattr(shared, "sd_model", None)
                        if not _is_anima_engine(active_model):
                            raise RuntimeError(
                                "The active model is not Anima after sampling; "
                                "the reference output was discarded."
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
                        if anima38_installed:
                            _restore_anima38(anima38_runtime, p2)
                        p2.close()
                        # Clear the process-global reference latents between
                        # candidates.  model.clear_references() is not called
                        # here because Forge's implementation also flushes the
                        # allocator and would stall every candidate.
                        _clear_reference_state()
    finally:
        shared.state.textinfo = ""
        _reset_image_stitch_cache()

    panel, upscale = _panel_diagnostics(request, prepared)
    return ReferenceGenerationResult(
        prepared=prepared,
        outputs=tuple(outputs),
        diagnostics={
            "mode": MODE_I2I,
            "target_panel": panel,
            "upscale": upscale,
            "model_blocks": model_blocks,
            "interrupted": interrupted,
            "anima38": anima38_label,
        },
    )


def _run_ipa_reference(
    request: ReferenceGenerationRequest,
    *,
    sd_model: Any = None,
    outpath_samples: str | None = None,
    outpath_grids: str | None = None,
) -> ReferenceGenerationResult:
    """IP-Adapter 방식: 캔버스 없이 txt2img 를 돌리고 샘플링만 주입으로 감싼다.

    주입은 ``p.sample()`` 안에서만 산다. Forge 가 ``sample()`` 첫머리에서 패처를 갈아끼우기
    때문에(modules/processing.py:1542) 그 직전에 끼워 넣어야 하고, 잡이 끝나면 저절로 풀린다.
    """
    from modules import shared
    from modules.processing import process_images

    from .inpaint_core import build_standalone_t2i, pause_total_tqdm

    if sd_model is None:
        # i2i 경로와 같은 이유: 재시작 직후 shared.sd_model 은 자리만 잡은 가짜다.
        from modules import sd_models

        sd_models.forge_model_reload()

    model = sd_model or getattr(shared, "sd_model", None)
    if model is None:
        raise RuntimeError("No Forge model is loaded.")
    if not _is_anima_engine(model):
        raise RuntimeError(
            "IP-Adapter 방식은 Anima 전용입니다. Anima 체크포인트를 먼저 불러오세요."
        )

    model_blocks = _model_block_count(model)
    # 3.8B v2 커넥터(감사 M10). 이 잡은 스크립트 러너 없이 돌아 Anima38 스크립트가 붙지 않는다 —
    # 토글이 켜져 있을 때만 이어붙이기 방식과 같은 설치(_install_anima38)를 후보마다 한다.
    anima38_wanted = _ipa_anima38_wanted(request, shared.opts)
    anima38_runtime, anima38_error = (
        _anima38_runtime() if anima38_wanted else (None, None)
    )
    anima38_label = "not attempted" if anima38_wanted else ANIMA38_IPA_OFF
    # 복제 블록 정책(lineage/all/split)은 잡 시작 때 한 번 정한다 — 후보마다 설정을 다시 읽으면 도중에
    # 설정을 바꿨을 때 infotext 와 실제 주입이 어긋날 수 있다. 요청이 정했으면 그 값, 아니면 Forge 설정.
    duplicate_policy = request.ipa.resolved_duplicate_policy(shared.opts)
    ipa_options = replace(request.ipa, duplicate_policy=duplicate_policy)
    sample_path = outpath_samples or getattr(
        shared.opts, "outdir_txt2img_samples", "outputs/txt2img-images"
    )
    grid_path = outpath_grids or getattr(
        shared.opts, "outdir_txt2img_grids", "outputs/txt2img-grids"
    )

    outputs: list[ReferenceGenerationOutput] = []
    # 주입 세션이 알려 주는 어댑터 깊이(모델과 다르면 블록 계보 매핑이 걸린 것이다).
    seen_adapter_blocks: list[int] = []
    candidate_count = int(request.candidate_count)
    if shared.state.job_count < 0:
        shared.state.job_count = candidate_count
    else:
        shared.state.job_count += candidate_count
    shared.state.job = "Anima Character Reference"

    interrupted = False
    try:
        with pause_total_tqdm():
            for index in range(candidate_count):
                if shared.state.interrupted or shared.state.skipped:
                    interrupted = True
                    break
                seed = candidate_seed(request.seed, request.seed_step, index)
                shared.state.job = (
                    f"Anima Reference (IP-Adapter) {index + 1}/{candidate_count}"
                )
                shared.state.textinfo = (
                    f"Anima Reference: candidate {index + 1}/{candidate_count} - sampling"
                )

                p2 = build_standalone_t2i(
                    request,
                    seed=seed,
                    sd_model=model,
                    outpath_samples=sample_path,
                    outpath_grids=grid_path,
                )
                p2.prompt = request.sampling_prompt
                p2.negative_prompt = str(request.negative_prompt or "")
                p2.distilled_cfg_scale = float(request.shift)
                p2.extra_generation_params.update(
                    {
                        "SAM3 Feature": 6,
                        "SAM3 Reference Mode": MODE_IPA,
                        "SAM3 IPA Strength": float(request.ipa.strength),
                        "SAM3 IPA Ref Size": int(request.ipa.ref_size),
                    }
                )
                if request.ipa.separate_cfg:
                    p2.extra_generation_params["SAM3 IPA CFG"] = float(
                        request.ipa.cfg_scale
                    )

                original_sample = p2.sample

                def sample_with_ipa(*args, _original=original_sample, _p=p2, **kwargs):
                    with _ipa_session(
                        _p.sd_model, request.reference_image, ipa_options
                    ) as adapter_blocks:
                        # 깊이가 다르면 블록 계보 매핑이 걸린 것이다 — 결과에 남긴다(복제 블록 정책도 함께).
                        if adapter_blocks and model_blocks and adapter_blocks != model_blocks:
                            seen_adapter_blocks.append(int(adapter_blocks))
                            _p.extra_generation_params["SAM3 IPA Blocks"] = (
                                f"{int(adapter_blocks)}→{model_blocks}"
                            )
                            _p.extra_generation_params["SAM3 IPA Duplicates"] = duplicate_policy
                        return _original(*args, **kwargs)

                p2.sample = sample_with_ipa

                # 스크립트와 같은 순서: 남은 설치 내리기 → (토글이 켜졌으면) 설치 → 끝나면 복원.
                # 커넥터는 조건(setup_conds)을 만들 때 쓰이므로 process_images 앞에서 단다. DiT forward
                # 래퍼는 IP 주입 껍데기가 _keep_foreign_dit_forward 로 체인째 부른다(주입과 함께 산다).
                _release_foreign_anima38(anima38_runtime, p2)
                anima38_installed = False
                if anima38_wanted:
                    anima38_label = _install_anima38(
                        anima38_runtime, anima38_error, p2, model
                    )
                    anima38_installed = anima38_label == "v2 bundle"
                    p2.extra_generation_params["Reference Anima38"] = anima38_label

                try:
                    processed = process_images(p2)
                    if shared.state.interrupted or shared.state.skipped:
                        # 멈춘 잡은 절반만 디노이즈된 그림을 준다 — 후보로 세지 않는다.
                        interrupted = True
                        break
                    if processed is None or not processed.images:
                        raise RuntimeError(
                            "Forge returned no image for the reference job."
                        )

                    target = processed.images[0].convert("RGB")
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
                            prompt=p2.prompt,
                            infotext=infotext,
                            processing=p2,
                            suffix="",
                        )
                    outputs.append(
                        ReferenceGenerationOutput(
                            target_image=target,
                            # 캔버스가 없다 — 생성물이 곧 결과다.
                            generated_canvas=target,
                            infotext=infotext,
                            seed=actual_seed,
                        )
                    )
                except Exception:
                    print(
                        "[-] Anima Character Reference (IP-Adapter) failed:\n"
                        f"{traceback.format_exc()}",
                        file=sys.stderr,
                    )
                    raise
                finally:
                    if anima38_installed:
                        _restore_anima38(anima38_runtime, p2)
                    close = getattr(p2, "close", None)
                    if callable(close):
                        close()
    finally:
        shared.state.textinfo = ""

    return ReferenceGenerationResult(
        prepared=None,
        outputs=tuple(outputs),
        diagnostics={
            "mode": MODE_IPA,
            "model_blocks": model_blocks,
            "ipa_adapter_blocks": seen_adapter_blocks[-1] if seen_adapter_blocks else None,
            "interrupted": interrupted,
            "ipa_strength": float(request.ipa.strength),
            "ipa_ref_size": int(request.ipa.ref_size),
            "ipa_separate_cfg": bool(request.ipa.separate_cfg),
            "ipa_duplicate_policy": duplicate_policy,
            "anima38_connector": anima38_wanted,
            "anima38": anima38_label,
        },
    )
