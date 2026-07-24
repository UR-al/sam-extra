"""Anima Tile-Repair core — framework-agnostic wrapper around the
kohya-ss/sd-scripts Anima inference pipeline.

This module is the v0.8.0 entry point for the Anima Tile-Repair panel that
lives below the SAM3 in-flight accordion. It does NOT import Gradio (UI lives
in ``ui_anima.py``).

Pipeline ↔ ComfyUI workflow equivalence
---------------------------------------
The user-supplied ComfyUI workflow runs:

    LoadImage(source)
      ├── ResizeImagesByShorterEdge → VAEEncode → KSampler.latent_image
      │     (denoise=1.0; latent_image is shape-only — sampling starts from
      │      pure noise)
      └── AnimaLLLiteApply(image=source, strength=1.0, start=0, end=1)
            → KSampler conditioning

We reproduce the exact same effect by:

1. Calling the vendor's ``anima_minimal_inference_control_net_lllite``
   monkey-patches so ``load_dit_model`` attaches ``ControlNetLLLiteDiT`` and
   ``generate_body`` sets ``cond_image = source`` before sampling.
2. Driving the standard ``ami.generate()`` loop (Flow Matching, ``flow_shift``,
   ``infer_steps``, ``guidance_scale``).
3. Decoding the bf16 latent via the Qwen-Image AutoencoderKL.

No init-image plumbing is required — the source image flows into the network
via the LLLite cond_image, exactly like ComfyUI's AnimaLLLiteApply node.
"""
from __future__ import annotations

import gc
import os
import random
import sys
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

EXTENSION_ROOT = Path(__file__).resolve().parent.parent
ANIMA_VENDOR = EXTENSION_ROOT / "anima_vendor"
# Sentinel matches install.ensure_anima_vendor's check — same file
# upstream ships at repo root.
ANIMA_SENTINEL = ANIMA_VENDOR / "anima_minimal_inference.py"
ANIMA_LLLITE_SENTINEL = ANIMA_VENDOR / "anima_minimal_inference_control_net_lllite.py"


# ---------------------------------------------------------------------------
# Public dataclass — the click handler builds one of these from widget values.
# ---------------------------------------------------------------------------


@dataclass
class AnimaTileRepairArgs:
    """All settings the panel collects. Mirrors the field set we exposed in
    ``ui_anima.ANIMA_ARG_KEYS`` so adding/removing a widget only touches the
    UI module and this dataclass in lockstep."""

    # Models
    lllite_model: str = "None"          # basename under models/ControlNet/
    dit_override: str = "Use Forge current"
    te_override: str = "Use Forge current"
    vae_override: str = "Use Forge current"
    # LoRA stack (4 fixed slots, bypass by leaving "None" / weight 0)
    lora_slots: list[tuple[str, float]] = field(default_factory=lambda: [("None", 0.0)] * 4)
    # Prompts
    positive: str = ""
    negative: str = ""
    # Sampler (vendor defaults — no Turbo LoRA assumed)
    steps: int = 50
    cfg: float = 3.5
    flow_shift: float = 5.0
    seed: int = -1
    # Output size
    width: int = 1024
    height: int = 1024
    # LLLite conditioning schedule
    lllite_strength: float = 1.0
    lllite_start: float = 0.0
    lllite_end: float = 1.0
    lllite_multiplier: float = 1.0
    # Housekeeping
    unload_forge_before: bool = True
    insert_mode: str = "After selected"
    # Restoration mode + PiD (Pixel Diffusion Decoder) option
    restore_mode: str = "Anima Tile-Repair"   # or "PiD Upscale"
    pid_checkpoint: str = ""
    pid_scale: float = 4.0
    pid_steps: int = 8
    pid_degrade: float = 0.4


# ---------------------------------------------------------------------------
# Vendor bootstrap (sys.path injection)
# ---------------------------------------------------------------------------

_VENDOR_INJECTED = False


def _ensure_vendor_importable() -> bool:
    """Add ``anima_vendor/`` to sys.path so ``import anima_minimal_inference``
    and ``from library import ...`` resolve. Idempotent.

    Also ensures ``anima_vendor/networks/__init__.py`` exists so the directory
    is unambiguously a regular package rather than a namespace package — this
    matters because Forge's ``extensions-builtin/sd_forge_lora/networks.py``
    is a *single-file* module also named ``networks`` and it's already cached
    in ``sys.modules`` at extension-load time. See ``_vendor_sys_modules``
    below for the runtime swap that hides that cached module during anima
    inference.
    """
    global _VENDOR_INJECTED
    if _VENDOR_INJECTED:
        return True
    if not ANIMA_SENTINEL.exists():
        return False

    # Force the vendor networks/ to be a regular package so Python's package
    # finder doesn't have to do namespace-package logic alongside whatever
    # other 'networks' module is already in sys.modules.
    nets_init = ANIMA_VENDOR / "networks" / "__init__.py"
    if not nets_init.exists():
        try:
            nets_init.parent.mkdir(parents=True, exist_ok=True)
            nets_init.write_text("", encoding="utf-8")
        except Exception:
            pass

    p = str(ANIMA_VENDOR)
    if p not in sys.path:
        sys.path.insert(0, p)
    _VENDOR_INJECTED = True
    return True


@contextmanager
def _vendor_sys_modules():
    """Hide Forge's ``sd_forge_lora.networks`` (and any other ``library`` /
    ``networks`` collision) from sys.modules so the vendor's namespace
    packages are findable for the duration of an anima inference call.

    Why: Forge's builtin LoRA extension imports its own
    ``extensions-builtin/sd_forge_lora/networks.py`` at startup. That puts a
    *module* (single .py file) named ``networks`` into ``sys.modules`` that
    stays there forever. When the vendor's ``library/lora_utils.py`` then
    does ``from networks.loha import ...``, Python finds the cached module
    instead of our vendor directory and raises::

        ModuleNotFoundError: No module named 'networks.loha'; 'networks' is
        not a package

    The fix is to pop the cached ``networks`` (and any submodule) on entry,
    let the vendor's import find the package directory, and restore the
    original modules on exit so the LoRA extension still works on the next
    t2i Generate.

    ``library`` gets the same treatment defensively even though no known
    shadow currently exists.
    """
    saved: dict[str, Any] = {}
    for key in list(sys.modules):
        if (
            key == "networks"
            or key.startswith("networks.")
            or key == "library"
            or key.startswith("library.")
        ):
            saved[key] = sys.modules.pop(key)
    try:
        yield
    finally:
        # Drop whatever the vendor injected (so sd_forge_lora's import
        # finds its single-file networks again) then restore the previously
        # cached modules.
        for key in list(sys.modules):
            if (
                key == "networks"
                or key.startswith("networks.")
                or key == "library"
                or key.startswith("library.")
            ):
                sys.modules.pop(key, None)
        sys.modules.update(saved)


def anima_available() -> bool:
    """Cheap probe used by ui_anima.py to decide whether to render. Doesn't
    actually import the vendor (would be slow at UI build time)."""
    return ANIMA_SENTINEL.exists()


# ---------------------------------------------------------------------------
# Path resolvers (Forge model-folder scanners)
# ---------------------------------------------------------------------------


def _models_path() -> Path | None:
    try:
        from modules import paths

        return Path(paths.models_path)
    except Exception:
        return None


_PID_NONE = "(no PiD checkpoint found)"


def list_pid_checkpoints() -> list[str]:
    """Forge checkpoints whose name contains 'PiD'. Forge Neo auto-enables PiD
    mode (backend/loader.py: dynamic_args.pid = 'PiD' in repo_name) when such a
    checkpoint is loaded; PiD then runs as an img2img restoration/upscale.
    Returns a sentinel when none are present so the dropdown isn't empty."""
    out: list[str] = []
    try:
        from modules import sd_models

        for title in sd_models.checkpoint_tiles():
            if "pid" in str(title).lower():
                out.append(title)
    except Exception:
        pass
    return out or [_PID_NONE]


def list_dit_choices() -> list[str]:
    """Anima DiT checkpoints. Filters ``models/Stable-diffusion/`` by
    ``anima`` substring to avoid polluting the dropdown with every SDXL
    checkpoint. ``Use Forge current`` always wins as the first entry."""
    out = ["Use Forge current"]
    root = _models_path()
    if root is not None:
        sd_dir = root / "Stable-diffusion"
        if sd_dir.is_dir():
            out.extend(
                sorted(p.name for p in sd_dir.glob("*.safetensors") if "anima" in p.name.lower())
            )
    return out


def list_te_choices() -> list[str]:
    out = ["Use Forge current"]
    root = _models_path()
    if root is not None:
        te_dir = root / "text_encoder"
        if te_dir.is_dir():
            out.extend(sorted(p.name for p in te_dir.glob("*.safetensors")))
    return out


def list_vae_choices() -> list[str]:
    out = ["Use Forge current"]
    try:
        from modules import sd_vae

        sd_vae.refresh_vae_list()
        out.extend(sorted(sd_vae.vae_dict.keys()))
    except Exception:
        # If we can't refresh the dict, fall back to a folder scan so the
        # dropdown isn't empty.
        root = _models_path()
        if root is not None:
            vae_dir = root / "VAE"
            if vae_dir.is_dir():
                out.extend(sorted(p.name for p in vae_dir.glob("*.safetensors")))
    return out


def list_lllite_choices() -> list[str]:
    """LLLite checkpoints. Forge's ControlNet folder is the user's convention
    (``models/ControlNet/animaTileRepair_v10.safetensors`` etc.)."""
    out = ["None"]
    root = _models_path()
    if root is None:
        return out
    # Some Forge variants use lowercase 'controlnet', user's tree uses both.
    for variant in ("ControlNet", "controlnet"):
        cn_dir = root / variant
        if not cn_dir.is_dir():
            continue
        # Match anima*/lllite* substrings so generic SDXL CN checkpoints
        # don't clutter the dropdown. ``saftensors`` typo handled too.
        for ext in (".safetensors", ".saftensors"):
            for p in sorted(cn_dir.glob(f"*{ext}")):
                lower = p.name.lower()
                if "anima" in lower or "lllite" in lower:
                    out.append(p.name)
        # only one variant — first hit wins
        if len(out) > 1:
            break
    return out


def default_te_choice(choices: list[str]) -> str:
    """Pick a sensible default Text Encoder for the dropdown. Anima needs a
    Qwen3 TE — auto-select a file that looks like one so the panel works out
    of the box (the user's anima_baseV10_txt.safetensors etc.) instead of the
    always-failing 'Use Forge current'."""
    for hint in ("qwen3", "qwen_3", "qwen", "anima", "_txt", "text_encoder", "te"):
        for c in choices:
            if c != "Use Forge current" and hint in c.lower():
                return c
    return choices[0] if choices else "Use Forge current"


def default_vae_choice(choices: list[str]) -> str:
    """Pick a sensible default VAE — Anima needs the Qwen-Image VAE, not the
    SDXL one Forge holds."""
    for hint in ("qwen_image", "qwen-image", "qwen", "anima"):
        for c in choices:
            if c != "Use Forge current" and hint in c.lower():
                return c
    return choices[0] if choices else "Use Forge current"


def list_lora_choices() -> list[str]:
    out = ["None"]
    root = _models_path()
    if root is not None:
        for variant in ("Lora", "lora"):
            lora_dir = root / variant
            if not lora_dir.is_dir():
                continue
            out.extend(sorted(p.name for p in lora_dir.glob("*.safetensors")))
            break
    return out


# ---------------------------------------------------------------------------
# "Use Forge current" → concrete path mapping
# ---------------------------------------------------------------------------


def _resolve_forge_current_dit() -> str | None:
    try:
        from modules import sd_models

        ci = sd_models.select_checkpoint()
        return ci.filename if ci is not None else None
    except Exception:
        return None


def _resolve_forge_current_vae() -> str | None:
    try:
        from modules import sd_vae

        # Forge stores the in-memory VAE path here when one is loaded.
        path = getattr(sd_vae, "loaded_vae_file", None)
        if path:
            return str(path)
    except Exception:
        pass
    return None


def _resolve_lllite_path(name: str) -> str | None:
    """Map an LLLite dropdown choice (basename) back to an absolute path."""
    if not name or name == "None":
        return None
    root = _models_path()
    if root is None:
        return None
    for variant in ("ControlNet", "controlnet"):
        cn_dir = root / variant
        candidate = cn_dir / name
        if candidate.is_file():
            return str(candidate)
    return None


def _resolve_te_path(name: str) -> str | None:
    if not name or name == "Use Forge current":
        return None
    root = _models_path()
    if root is None:
        return None
    candidate = root / "text_encoder" / name
    return str(candidate) if candidate.is_file() else None


def _resolve_vae_path(name: str) -> str | None:
    if not name or name == "Use Forge current":
        return None
    try:
        from modules import sd_vae

        return sd_vae.vae_dict.get(name)
    except Exception:
        return None


def _resolve_lora_path(name: str) -> str | None:
    if not name or name == "None":
        return None
    root = _models_path()
    if root is None:
        return None
    for variant in ("Lora", "lora"):
        candidate = root / variant / name
        if candidate.is_file():
            return str(candidate)
    return None


# ---------------------------------------------------------------------------
# Forge SD model swap
# ---------------------------------------------------------------------------


@contextmanager
def forge_sd_unloaded():
    """Push the current Forge SD model out of VRAM during Anima inference,
    then let Forge lazy-reload on the next t2i sampling step.

    CRITICAL distinction (verified against backend/memory_management.py +
    modules/sd_models.py):

    - ``backend.memory_management.unload_all_models()`` clears VRAM without
      destroying ``model_data.sd_model``. ``model_data.forge_hash`` is
      preserved, so the next ``forge_model_reload()`` early-returns and the
      user does NOT pay a full disk reload.
    - ``sd_models.unload_model_weights()`` NUKES the in-memory model
      (replaces with FakeInitialModel + clears forge_hash). The next t2i
      would have to re-read the checkpoint from disk. Do not call it from
      here.
    """
    try:
        from backend import memory_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
    except Exception:
        # If Forge's memory layer isn't importable we just skip — Anima's
        # own loader will create VRAM pressure that Forge will resolve on
        # its next sampling step.
        pass
    try:
        yield
    finally:
        try:
            from backend import memory_management as mm

            mm.soft_empty_cache()
            gc.collect()
        except Exception:
            pass


@contextmanager
def _nullctx():
    yield


# ---------------------------------------------------------------------------
# argparse.Namespace builder (matches anima_minimal_inference's CLI shape)
# ---------------------------------------------------------------------------


def _build_anima_args(repair: AnimaTileRepairArgs, control_image_path: str) -> SimpleNamespace:
    """Build the argparse-Namespace shape that the vendor's
    ``anima_minimal_inference`` + LLLite monkey-patch expect.

    ``control_image_path`` is a temp PNG of the source image (the LLLite
    extension reads it from disk via ``--control_image``).
    """
    ns = SimpleNamespace()

    # --- prompts ---
    ns.prompt = repair.positive
    ns.negative_prompt = repair.negative or ""

    # --- sampling ---
    ns.image_size = [int(repair.height), int(repair.width)]  # [H, W] per vendor convention
    ns.infer_steps = int(repair.steps)
    ns.guidance_scale = float(repair.cfg)
    ns.flow_shift = float(repair.flow_shift)
    # For a random request (-1) pick a concrete seed ourselves instead of
    # handing the vendor ``None``. The vendor would draw its own internal
    # random seed that we can't read back, so the infotext would record -1 and
    # the image would be irreproducible. An explicit int is used verbatim.
    ns.seed = int(repair.seed) if repair.seed >= 0 else random.randint(0, 2**31 - 1)

    # --- model paths ---
    dit = (
        _resolve_forge_current_dit()
        if repair.dit_override == "Use Forge current"
        else None
    )
    if repair.dit_override != "Use Forge current":
        root = _models_path()
        if root is not None:
            dit = str(root / "Stable-diffusion" / repair.dit_override)
    ns.dit = dit  # vendor key is --dit

    # Text encoder — vendor key is --text_encoder
    if repair.te_override == "Use Forge current":
        ns.text_encoder = None
    else:
        ns.text_encoder = _resolve_te_path(repair.te_override)

    # VAE — vendor key is --vae
    if repair.vae_override == "Use Forge current":
        ns.vae = _resolve_forge_current_vae()
    else:
        ns.vae = _resolve_vae_path(repair.vae_override)
    ns.vae_chunk_size = None
    ns.vae_disable_cache = False
    ns.qwen_image_vae_2d = False

    # --- LoRA stack (parallel lists, vendor takes nargs='*') ---
    weights: list[str] = []
    multipliers: list[float] = []
    for name, weight in repair.lora_slots:
        path = _resolve_lora_path(name)
        if path is None:
            continue
        if weight == 0.0:
            continue
        weights.append(path)
        multipliers.append(float(weight))
    ns.lora_weight = weights or None
    # None (not scalar 1.0) when empty — the vendor's
    # load_safetensors_with_lora_and_fp8 / load_anima_model handle None
    # safely; a stray scalar would break the TE-LoRA list path.
    ns.lora_multiplier = multipliers if multipliers else None
    ns.include_patterns = None
    ns.exclude_patterns = None

    # --- LLLite specific ---
    ns.lllite_weights = _resolve_lllite_path(repair.lllite_model)
    ns.control_image = control_image_path
    ns.lllite_multiplier = float(repair.lllite_multiplier)
    # Architecture overrides — let the loader read them from metadata.
    ns.lllite_cond_emb_dim = None
    ns.lllite_mlp_dim = None
    ns.lllite_target_layers = None
    ns.lllite_cond_dim = None
    ns.lllite_cond_resblocks = None
    ns.lllite_use_aspp = None
    ns.lllite_cond_in_channels = None
    ns.lllite_inpaint_masked_input = None
    ns.mask_image = None  # 3-channel tile-repair has no inpaint mask

    # --- precision / runtime ---
    ns.fp8 = False
    ns.fp8_scaled = False
    ns.text_encoder_cpu = False
    ns.device = None  # auto cuda/cpu
    ns.attn_mode = "torch"  # sdpa-equivalent; verified the only safe option
    ns.output_type = "images"
    ns.no_metadata = False
    ns.latent_path = None
    ns.lycoris = False

    # --- save / batch flags (we never actually save via vendor) ---
    ns.save_path = tempfile.gettempdir()  # vendor checks; we ignore its file
    ns.from_file = None
    ns.interactive = False

    return ns


# ---------------------------------------------------------------------------
# Output conversion
# ---------------------------------------------------------------------------


def _tensor_to_pil(pixels: torch.Tensor) -> Image.Image:
    """Convert the vendor's decoded pixel tensor to a PIL RGB image.

    vendor decode_latent returns either BCTHW (T=1) or BCHW depending on the
    Qwen-Image AutoencoderKL flavor. Both layouts collapse to a single
    HxWx3 numpy array in [0, 1].
    """
    t = pixels
    if t.ndim == 5:
        t = t[0, :, 0]  # B C T H W → C H W
    elif t.ndim == 4:
        t = t[0]
    elif t.ndim == 3:
        pass  # already C H W
    else:
        raise ValueError(f"unexpected pixel tensor rank {t.ndim}")
    t = t.float().clamp(-1.0, 1.0)
    # Anima decode_to_pixels returns either [-1,1] or [0,1] depending on the
    # vae flavor. Detect and normalize.
    if float(t.min()) < -0.01:
        t = (t + 1.0) * 0.5
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _build_infotext(repair: AnimaTileRepairArgs, seed_used: int) -> str:
    """One-line infotext for the gallery splice, modeled after Forge's
    standard ``Parameters:`` string so the gallery sidebar shows something
    meaningful when the user clicks the refined thumbnail."""
    parts = [
        repair.positive or "",
        f"Negative prompt: {repair.negative}" if repair.negative else None,
        (
            f"Steps: {repair.steps}, "
            f"CFG scale: {repair.cfg}, "
            f"Seed: {seed_used}, "
            f"Size: {repair.width}x{repair.height}, "
            f"Flow shift: {repair.flow_shift}, "
            f"LLLite: {repair.lllite_model} (mult {repair.lllite_multiplier}, "
            f"strength {repair.lllite_strength}, "
            f"sched {repair.lllite_start:.2f}-{repair.lllite_end:.2f})"
        ),
        "Anima Tile-Repair: on",
    ]
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Top-level entry called by ui_anima.handle_anima_click
# ---------------------------------------------------------------------------


def run_tile_repair(
    source: Image.Image,
    repair: AnimaTileRepairArgs,
) -> list[tuple[Image.Image, str]]:
    """Single-pass Anima Tile-Repair against ``source``. Returns the same
    ``[(pil, infotext)]`` shape as ``inpaint_core.run_sam3_refine`` so the
    gallery splice logic in ``ui_anima.handle_anima_click`` is byte-for-byte
    identical to ``ui_refine.handle_refine_click``.

    Raises ``RuntimeError`` on vendor missing — the UI catches and shows it as
    a red status banner.
    """
    if not _ensure_vendor_importable():
        raise RuntimeError(
            "Anima vendor missing. Re-run install.py or clone "
            "kohya-ss/sd-scripts into anima_vendor/."
        )
    if not ANIMA_LLLITE_SENTINEL.exists():
        raise RuntimeError(
            "Vendor present but anima_minimal_inference_control_net_lllite.py "
            "is missing. Re-clone the vendor tree."
        )
    if repair.lllite_model in (None, "", "None"):
        raise RuntimeError(
            "Pick an LLLite model (e.g. animaTileRepair_v10.safetensors)."
        )

    out_pairs: list[tuple[Image.Image, str]] = []
    control_image_path: str | None = None
    # _vendor_sys_modules() pops sd_forge_lora's cached ``networks`` module so
    # the vendor's namespace package is discoverable. The swap MUST stay
    # active for the whole inference call because the vendor lazily imports
    # submodules during sampling. The ``with`` block guarantees restore on
    # both success and exception paths.
    with _vendor_sys_modules():
        # Apply the LLLite monkey-patches by importing the extension module.
        # It patches ami.parse_args / parse_prompt_line / load_dit_model /
        # generate_body — the four hooks we need to make the LLLite cond
        # image reach the DiT.
        try:
            import anima_minimal_inference_control_net_lllite  # type: ignore  # noqa: F401
            import anima_minimal_inference as ami  # type: ignore
        except ModuleNotFoundError as e:
            # Re-throw with a humanized message so the UI status banner
            # tells the user exactly what to pip install instead of a raw
            # traceback. install.py already breadcrumbed at extension load
            # but the user may have started Forge before reading stderr.
            missing = getattr(e, "name", str(e))
            raise RuntimeError(
                f"Anima 의존성 누락: '{missing}'. Forge venv에서\n"
                f"   pip install {missing}\n"
                f"실행 후 Forge 재시작하세요. "
                f"(torchvision은 torch CUDA 빌드와 같은 버전으로 맞추세요.)"
            ) from e

        # Vendor ami.generate() expects the source image as a file on disk
        # (--control_image path). Save the gallery PIL to a tempfile.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            source.convert("RGB").save(tf, format="PNG")
            control_image_path = tf.name

        try:
            # IMPORTANT: dimensions must be divisible by 32 (vendor
            # check_inputs raises otherwise). Snap to nearest multiple of 32.
            repair.width = max(256, (int(repair.width) // 32) * 32)
            repair.height = max(256, (int(repair.height) // 32) * 32)

            args = _build_anima_args(repair, control_image_path)

            # Resolve "Use Forge current" DiT if still None — vendor will
            # balk without a DiT path.
            if not args.dit:
                args.dit = _resolve_forge_current_dit()
            if not args.dit:
                raise RuntimeError(
                    "No DiT checkpoint resolved. Either select a model in "
                    "the panel or load one in Forge first."
                )

            # Anima REQUIRES a Qwen3 text encoder + a Qwen-Image VAE. Neither
            # is what Forge holds as "current" (Forge's VAE is SDXL-shaped →
            # strict load fails; there's no standalone Qwen3 TE concept). Fail
            # early with an actionable message instead of an opaque ValueError
            # deep in the vendor's tokenizer/VAE loader.
            if not args.text_encoder:
                raise RuntimeError(
                    "Anima는 Qwen3 Text Encoder가 필수입니다. 패널의 "
                    "'SAM3 Anima Text Encoder Override' 드롭다운에서 "
                    "models/text_encoder/ 의 Qwen3 .safetensors를 선택하세요. "
                    "('Use Forge current'는 Anima용 TE를 제공하지 못합니다.)"
                )
            if not args.vae:
                raise RuntimeError(
                    "Anima는 Qwen-Image VAE가 필수입니다. 패널의 "
                    "'SAM3 Anima VAE Override' 드롭다운에서 Qwen-Image VAE를 "
                    "명시적으로 선택하세요. ('Use Forge current'는 SDXL VAE라 "
                    "strict load에서 실패합니다.)"
                )

            ctx = forge_sd_unloaded() if repair.unload_forge_before else _nullctx()
            with ctx:
                # Set the tokenize/encode strategies up front (vendor main()
                # does this; we have to do it ourselves because we're
                # bypassing main()).
                from library import strategy_anima, strategy_base  # type: ignore

                strategy_base.TokenizeStrategy.set_strategy(
                    strategy_anima.AnimaTokenizeStrategy(
                        qwen3_path=args.text_encoder,
                        t5_tokenizer_path=None,
                        qwen3_max_length=512,
                        t5_max_length=512,
                    )
                )
                strategy_base.TextEncodingStrategy.set_strategy(
                    strategy_anima.AnimaTextEncodingStrategy()
                )

                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                args.device = device

                gen_settings = ami.get_generation_settings(args)
                latent = ami.generate(args, gen_settings)

                # Decode — vendor loads VAE separately to keep DiT in VRAM
                # during sampling and frees it before the VAE pass.
                from library import anima_train_utils  # type: ignore

                vae = anima_train_utils.load_qwen_image_vae(
                    args, device="cpu", disable_mmap=True
                )
                vae.to(torch.bfloat16).eval()
                pixels = ami.decode_latent(vae, latent, device)
                pil = _tensor_to_pil(pixels)

                seed_used = int(args.seed) if args.seed is not None else -1
                out_pairs.append((pil, _build_infotext(repair, seed_used)))
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            # Humanize the most common load failure: a non-Anima DiT/VAE was
            # selected (Forge's SDXL checkpoint/VAE) → state_dict key/shape
            # mismatch deep in the loader.
            msg = str(e).lower()
            if "size mismatch" in msg or "missing key" in msg or "unexpected key" in msg:
                raise RuntimeError(
                    "모델 로드 실패 — 선택한 DiT/VAE가 Anima(Qwen-Image) 체크포인트가 "
                    "아닐 수 있습니다. Anima DiT + Qwen-Image VAE + Qwen3 TE를 사용하세요."
                ) from e
            raise
        finally:
            if control_image_path:
                try:
                    os.unlink(control_image_path)
                except Exception:
                    pass

    return out_pairs


def run_pid_upscale(
    source: Image.Image,
    *,
    pid_checkpoint: str,
    scale: float = 4.0,
    degrade_sigma: float = 0.4,
    steps: int = 8,
    sampler: str = "Euler",
) -> list[tuple[Image.Image, str]]:
    """PiD (Pixel Diffusion Decoder) restoration/upscale via Forge Neo's NATIVE
    pipeline. PiD is a first-class diffusion engine (backend/diffusion_engine/
    pid.py); Forge auto-enables PiD mode when a checkpoint whose name contains
    'PiD' is loaded (backend/loader.py). It runs as img2img where
    ``denoising_strength`` is reinterpreted as the degrade sigma and the mask
    must be None (modules/processing.py PiD branches). So we just run a
    standalone img2img with the PiD checkpoint swapped in via override_settings;
    Forge handles all the PiD specifics.

    Returns ``[(pil, infotext)]`` — same shape as run_tile_repair so the gallery
    splice in ui_anima is reused.
    """
    if not pid_checkpoint or pid_checkpoint in ("", "None", _PID_NONE):
        raise RuntimeError(
            "PiD 체크포인트가 없습니다. nvidia/PiD 모델(파일명/repo에 'PiD' 포함)을 "
            "models/Stable-diffusion/ 에 넣고 새로고침하세요."
        )

    from modules import shared
    from modules.processing import process_images

    from .inpaint_core import (
        build_standalone_i2i,
        build_standalone_scripts_runner,
        override_sampler_script_slot,
        pause_total_tqdm,
    )

    sd_model = getattr(shared, "sd_model", None)
    if sd_model is None:
        raise RuntimeError("PiD: no SD model loaded in Forge.")
    outpath_samples = getattr(shared.opts, "outdir_txt2img_samples", "outputs/txt2img-images")
    outpath_grids = getattr(shared.opts, "outdir_txt2img_grids", "outputs/txt2img-grids")

    runner, script_args = build_standalone_scripts_runner()
    if runner is None:
        raise RuntimeError("PiD: img2img scripts runner not initialized.")

    src = source.convert("RGB")
    # PiD targets pixel space (3, H, W); snap to /32 like the rest of the pipeline.
    w = max(256, (int(src.width * scale) // 32) * 32)
    h = max(256, (int(src.height * scale) // 32) * 32)

    # Synthesize the sam3_* args build_standalone_i2i expects. PiD-relevant bits:
    # denoising_strength → degrade sigma (Forge reinterprets it), no mask,
    # explicit target size, CFG 1.0 (PiD is condition-driven).
    args: dict[str, Any] = {
        "sam3_use_inpaint_width_height": True,
        "sam3_inpaint_width": w,
        "sam3_inpaint_height": h,
        "sam3_use_steps": True,
        "sam3_steps": int(steps),
        "sam3_use_cfg_scale": True,
        "sam3_cfg_scale": 1.0,
        "sam3_use_sampler": True,
        "sam3_sampler": sampler or "Euler",
        "sam3_use_scheduler": True,
        "sam3_scheduler": "Automatic",
        "sam3_use_seed": True,
        "sam3_seed": -1,
        "sam3_noise_multiplier": 1.0,
        "sam3_denoising_strength": float(degrade_sigma),
        "sam3_inpainting_fill": "original",
        "sam3_inpaint_only_masked": False,
        "sam3_inpaint_only_masked_padding": 0,
        "sam3_mask_blur": 0,
        "sam3_resize_mode": "Just Resize",
        "sam3_mask_invert": False,
        "sam3_restore_face": False,
    }

    results: list[tuple[Image.Image, str]] = []
    with pause_total_tqdm():
        p2 = build_standalone_i2i(
            src,
            args,
            sd_model=sd_model,
            outpath_samples=outpath_samples,
            outpath_grids=outpath_grids,
            scripts_runner=runner,
            script_args=list(script_args),
        )
        # Swap to the PiD checkpoint for this pass; Forge restores after.
        p2.override_settings = {"sd_model_checkpoint": pid_checkpoint}
        p2.image_mask = None  # PiD asserts no mask
        p2.prompt = ""
        p2.negative_prompt = ""
        override_sampler_script_slot(p2, args)
        try:
            processed = process_images(p2)
        finally:
            p2.close()
        if processed is not None and processed.images:
            info = ""
            try:
                if getattr(processed, "infotexts", None):
                    info = processed.infotexts[0] or ""
                if not info:
                    info = getattr(processed, "info", "") or ""
            except Exception:
                info = ""
            if not info:
                info = f"PiD upscale x{scale}, degrade σ {degrade_sigma}\nPiD: {pid_checkpoint}"
            results.append((processed.images[0].convert("RGB"), info))
    return results
