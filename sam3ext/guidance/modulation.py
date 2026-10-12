"""Anima CLIP-L modulation-guidance loader and projection helpers.

This is an extension-only Forge port of the Anima/Cosmos path from
``Anzhc/Anima-Mod-Guidance-ComfyUI-Node``, itself based on
``quickjkee/modulation-guidance``. Anima does not contain a pooled CLIP text
path, so the port loads a separate CLIP-L encoder and the published
``yresearch/cosmos-pooled`` adapter.

The expensive CLIP and adapter operations happen once per prompt/generation on
CPU. Sampling only keeps the final per-block AdaLN vectors, which are tiny.
"""

from __future__ import annotations

import gc
import hashlib
import os
import threading
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


ADAPTER_URL = (
    "https://huggingface.co/yresearch/cosmos-pooled/resolve/main/"
    "checkpoint_4000.pt"
)
ADAPTER_FILE_NAME = "checkpoint_4000.pt"
ADAPTER_SHA256 = (
    "27d4a33c817fb9ab9602b571f01f429bf34ddf96ab8b73fa4ed682f3266f84ea"
)
ADAPTER_SIZE = 170_609_044
RECOMMENDED_CLIP_MARKER = "Anzhc"

EXPECTED_ADAPTER_KEYS = (
    "scales",
    "text_embedder_clip.linear_1.weight",
    "text_embedder_clip.linear_1.bias",
    "text_embedder_clip.linear_2.weight",
    "text_embedder_clip.linear_2.bias",
)

_CACHE_LOCK = threading.RLock()
_CPU_ADAPTER_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_VERIFIED_OFFICIAL_ADAPTERS: set[tuple] = set()
_CLIP_CACHE_KEY: tuple | None = None
_CLIP_CACHE_VALUE: tuple | None = None
_POOLED_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_MAX_POOLED_CACHE = 64


def _file_signature(path: os.PathLike) -> tuple:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns)


def list_clip_l_models(models_root: os.PathLike) -> list[str]:
    """List standalone CLIP-L safetensors from Forge's text_encoder folder."""
    directory = Path(models_root) / "text_encoder"
    if not directory.is_dir():
        return []
    try:
        from safetensors import safe_open
    except Exception:
        return []

    names = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != ".safetensors":
            continue
        try:
            # Header-only inspection avoids loading hundreds of MB merely to
            # populate a dropdown, and prevents CLIP-G/Qwen encoders from
            # appearing as apparently valid choices.
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                token_keys = [
                    key
                    for key in handle.keys()
                    if key.endswith(
                        "text_model.embeddings.token_embedding.weight"
                    )
                ]
                compatible = any(
                    tuple(handle.get_slice(key).get_shape())[-1:] == (768,)
                    for key in token_keys
                )
            if compatible:
                names.append(path.name)
        except Exception:
            continue
    return sorted(
        names,
        key=lambda name: (
            RECOMMENDED_CLIP_MARKER.lower() not in name.lower(),
            "clip" not in name.lower() or "l" not in name.lower(),
            name.lower(),
        ),
    )


def resolve_clip_path(models_root: os.PathLike, clip_model: str) -> Path:
    value = str(clip_model or "").strip()
    if not value:
        raise RuntimeError(
            "CLIP-L model is empty. Put the Anzhc NoobAI CLIP-L safetensors "
            "under models/text_encoder and select it."
        )
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(models_root) / "text_encoder" / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise RuntimeError(f"CLIP-L model not found: {candidate}")
    if candidate.suffix.lower() != ".safetensors":
        raise RuntimeError(
            "Only safetensors CLIP-L files are accepted by modulation guidance."
        )
    return candidate


def default_adapter_path(models_root: os.PathLike) -> Path:
    return (
        Path(models_root)
        / "anima_modulation_guidance"
        / ADAPTER_FILE_NAME
    )


def _download_official_adapter(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(ADAPTER_URL, timeout=120) as response:
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
        size = temporary.stat().st_size
        actual_hash = digest.hexdigest()
        if size != ADAPTER_SIZE or actual_hash != ADAPTER_SHA256:
            raise RuntimeError(
                "Downloaded modulation adapter failed integrity validation: "
                f"size={size}, sha256={actual_hash}."
            )
        os.replace(temporary, destination)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_adapter_path(
    models_root: os.PathLike,
    adapter_mode: str,
    adapter_path: str = "",
) -> Path:
    mode = str(adapter_mode or "Auto-download official").strip().lower()
    if mode in {"auto", "auto-download", "auto-download official"}:
        resolved = default_adapter_path(models_root)
        # Concurrent requests share one Forge process. Serialize first-use
        # download/rename so queued generations cannot race on the same
        # ``.download`` file.
        with _CACHE_LOCK:
            downloaded = False
            if not resolved.is_file():
                _download_official_adapter(resolved)
                downloaded = True
            elif resolved.stat().st_size != ADAPTER_SIZE:
                raise RuntimeError(
                    f"Official adapter path has an unexpected size: {resolved}. "
                    "Move it away or choose Local file explicitly."
                )
            signature = _file_signature(resolved)
            if downloaded:
                _VERIFIED_OFFICIAL_ADAPTERS.add(signature)
            elif signature not in _VERIFIED_OFFICIAL_ADAPTERS:
                actual_hash = _sha256_file(resolved)
                if actual_hash != ADAPTER_SHA256:
                    raise RuntimeError(
                        "Official modulation adapter failed SHA-256 validation: "
                        f"{resolved}. Move it away so it can be downloaded again."
                    )
                _VERIFIED_OFFICIAL_ADAPTERS.add(signature)
        return resolved.resolve()

    if mode in {"local", "local file", "local_file"}:
        value = str(adapter_path or "").strip()
        if not value:
            raise RuntimeError("Local adapter mode requires an adapter path.")
        resolved = Path(value).expanduser().resolve()
        if not resolved.is_file():
            raise RuntimeError(f"Modulation adapter not found: {resolved}")
        return resolved

    raise RuntimeError(f"Unsupported adapter mode: {adapter_mode}")


def load_adapter_cpu(path: os.PathLike) -> dict[str, torch.Tensor]:
    """Safely load and cache only the tensor keys used by the public adapter."""
    signature = _file_signature(path)
    with _CACHE_LOCK:
        cached = _CPU_ADAPTER_CACHE.get(signature)
        if cached is not None:
            return cached

    try:
        raw = torch.load(
            signature[0],
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to safely load modulation adapter '{signature[0]}': {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise RuntimeError(
            "Invalid modulation adapter: expected a tensor dictionary."
        )

    missing = [key for key in EXPECTED_ADAPTER_KEYS if key not in raw]
    if missing:
        raise RuntimeError(
            "Modulation adapter is missing required keys: "
            + ", ".join(missing)
        )

    state: dict[str, torch.Tensor] = {}
    for key in EXPECTED_ADAPTER_KEYS:
        value = raw[key]
        if not torch.is_tensor(value):
            raise RuntimeError(
                f"Modulation adapter key '{key}' is not a tensor."
            )
        state[key] = value.detach().float().cpu().contiguous()

    with _CACHE_LOCK:
        # A replaced file at the same path must not retain its old tensors.
        for old_key in list(_CPU_ADAPTER_CACHE):
            if old_key[0] == signature[0] and old_key != signature:
                _CPU_ADAPTER_CACHE.pop(old_key, None)
        _CPU_ADAPTER_CACHE[signature] = state
    return state


def validate_adapter_for_model(
    state: dict[str, torch.Tensor],
    diffusion_model,
) -> dict[str, int]:
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None or len(blocks) <= 0:
        raise RuntimeError(
            "Unsupported Anima internals: missing transformer blocks."
        )

    num_blocks = len(blocks)
    # ComfyUI's Cosmos class exposes ``model_channels``. Forge Neo's current
    # backend.nn.anima.Anima does not retain that constructor argument, but
    # every Block exposes the equivalent ``x_dim``. Keep both layouts valid
    # without editing the Forge backend.
    model_channels = getattr(diffusion_model, "model_channels", None)
    if not isinstance(model_channels, int) or model_channels <= 0:
        model_channels = getattr(blocks[0], "x_dim", None)
    if not isinstance(model_channels, int) or model_channels <= 0:
        normalized_shape = getattr(
            getattr(diffusion_model, "t_embedding_norm", None),
            "normalized_shape",
            None,
        )
        if isinstance(normalized_shape, (tuple, list)) and len(normalized_shape) == 1:
            model_channels = int(normalized_shape[0])
    if not isinstance(model_channels, int) or model_channels <= 0:
        raise RuntimeError(
            "Unsupported Anima internals: could not infer model channel width."
        )

    adaln_dim = model_channels * 3
    scales = state["scales"]
    l1_w = state["text_embedder_clip.linear_1.weight"]
    l1_b = state["text_embedder_clip.linear_1.bias"]
    l2_w = state["text_embedder_clip.linear_2.weight"]
    l2_b = state["text_embedder_clip.linear_2.bias"]

    if tuple(scales.shape) != (num_blocks, adaln_dim):
        raise RuntimeError(
            "Adapter/model block shape mismatch: "
            f"adapter={tuple(scales.shape)}, expected=({num_blocks}, {adaln_dim})."
        )
    if l1_w.ndim != 2:
        raise RuntimeError("Adapter CLIP linear_1 weight must be rank 2.")
    pooled_dim = int(l1_w.shape[1])
    if tuple(l1_w.shape) != (adaln_dim, pooled_dim):
        raise RuntimeError("Adapter CLIP linear_1 weight shape is invalid.")
    if tuple(l1_b.shape) != (adaln_dim,):
        raise RuntimeError("Adapter CLIP linear_1 bias shape is invalid.")
    if tuple(l2_w.shape) != (adaln_dim, adaln_dim):
        raise RuntimeError("Adapter CLIP linear_2 weight shape is invalid.")
    if tuple(l2_b.shape) != (adaln_dim,):
        raise RuntimeError("Adapter CLIP linear_2 bias shape is invalid.")

    return {
        "num_blocks": num_blocks,
        "adaln_dim": adaln_dim,
        "pooled_dim": pooled_dim,
    }


def _normalize_clip_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize common HF/Forge prefixes to ``CLIPTextModel`` keys."""
    if "text_model.embeddings.token_embedding.weight" in state:
        return state

    prefixes = (
        "transformer.",
        "cond_stage_model.transformer.",
        "conditioner.embedders.0.transformer.",
        "text_encoders.clip_l.transformer.",
        "clip_l.transformer.",
    )
    for prefix in prefixes:
        marker = prefix + "text_model.embeddings.token_embedding.weight"
        if marker in state:
            return {
                key[len(prefix):]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
    raise RuntimeError(
        "Unsupported CLIP-L key layout. Use a Hugging Face-format CLIP-L "
        "safetensors such as Anzhc NoobAI11 CLIP L Anime."
    )


def _load_clip_encoder(
    clip_path: Path,
    forge_root: os.PathLike,
):
    global _CLIP_CACHE_KEY, _CLIP_CACHE_VALUE
    signature = _file_signature(clip_path)
    with _CACHE_LOCK:
        if _CLIP_CACHE_KEY == signature and _CLIP_CACHE_VALUE is not None:
            return _CLIP_CACHE_VALUE

    try:
        from safetensors.torch import load_file
        from transformers import (
            CLIPTextConfig,
            CLIPTextModel,
            CLIPTokenizer,
        )
        from transformers.modeling_utils import no_init_weights
    except Exception as exc:
        raise RuntimeError(
            "CLIP-L modulation requires transformers and safetensors."
        ) from exc

    assets = (
        Path(forge_root)
        / "backend"
        / "huggingface"
        / "stabilityai"
        / "stable-diffusion-xl-base-1.0"
    )
    config_dir = assets / "text_encoder"
    tokenizer_dir = assets / "tokenizer"
    if not config_dir.is_dir() or not tokenizer_dir.is_dir():
        raise RuntimeError(
            f"Forge CLIP-L config/tokenizer assets are missing under {assets}."
        )

    config = CLIPTextConfig.from_pretrained(
        config_dir,
        local_files_only=True,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
    )
    with no_init_weights():
        model = CLIPTextModel(config)

    raw_state = load_file(str(clip_path), device="cpu")
    normalized = _normalize_clip_state_dict(raw_state)
    model_keys = set(model.state_dict().keys())
    filtered = {key: value for key, value in normalized.items() if key in model_keys}
    result = model.load_state_dict(filtered, strict=False, assign=True)
    if result.missing_keys:
        raise RuntimeError(
            "CLIP-L model is missing required weights: "
            + ", ".join(result.missing_keys[:8])
        )
    # CPU float32 is predictable on every supported Torch build. The
    # recommended Anzhc encoder is already float32, so this is normally a no-op.
    if next(model.parameters()).dtype != torch.float32:
        model.float()
    model.eval()

    value = (model, tokenizer, signature)
    with _CACHE_LOCK:
        _CLIP_CACHE_KEY = signature
        _CLIP_CACHE_VALUE = value
        _POOLED_CACHE.clear()
    return value


def encode_clip_pooled(
    clip_path: os.PathLike,
    prompts: Iterable[str],
    forge_root: os.PathLike,
) -> torch.Tensor:
    """Return one pooled CLIP-L vector per prompt on CPU float32."""
    prompt_tuple = tuple(str(prompt or "") for prompt in prompts)
    if not prompt_tuple:
        raise RuntimeError("At least one CLIP prompt is required.")

    model, tokenizer, signature = _load_clip_encoder(
        Path(clip_path).resolve(),
        forge_root,
    )
    cache_key = (signature, prompt_tuple)
    with _CACHE_LOCK:
        cached = _POOLED_CACHE.get(cache_key)
        if cached is not None:
            _POOLED_CACHE.move_to_end(cache_key)
            return cached.clone()

    tokens = tokenizer(
        list(prompt_tuple),
        padding="max_length",
        truncation=True,
        max_length=int(model.config.max_position_embeddings),
        return_tensors="pt",
    )
    with torch.inference_mode():
        output = model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
        )
        pooled = output.pooler_output.detach().float().cpu().contiguous()
    if pooled.ndim != 2 or not bool(torch.isfinite(pooled).all()):
        raise RuntimeError(
            f"CLIP-L returned an invalid pooled tensor: {tuple(pooled.shape)}."
        )

    with _CACHE_LOCK:
        _POOLED_CACHE[cache_key] = pooled
        _POOLED_CACHE.move_to_end(cache_key)
        while len(_POOLED_CACHE) > _MAX_POOLED_CACHE:
            _POOLED_CACHE.popitem(last=False)
    return pooled.clone()


def project_block_modulations(
    pooled: torch.Tensor,
    adapter_state: dict[str, torch.Tensor],
    diffusion_model,
    weight: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Project base/positive/negative CLIP vectors to per-block AdaLN deltas."""
    meta = validate_adapter_for_model(adapter_state, diffusion_model)
    pooled = pooled.detach().float().cpu()
    if tuple(pooled.shape) != (3, meta["pooled_dim"]):
        raise RuntimeError(
            "Expected three pooled CLIP-L vectors with shape "
            f"(3, {meta['pooled_dim']}), got {tuple(pooled.shape)}."
        )

    projected = F.linear(
        pooled,
        adapter_state["text_embedder_clip.linear_1.weight"],
        adapter_state["text_embedder_clip.linear_1.bias"],
    )
    projected = F.silu(projected)
    projected = F.linear(
        projected,
        adapter_state["text_embedder_clip.linear_2.weight"],
        adapter_state["text_embedder_clip.linear_2.bias"],
    )
    modulation = projected[0] + float(weight) * (
        projected[1] - projected[2]
    )
    block_modulations = (
        adapter_state["scales"] * modulation.unsqueeze(0)
    ).contiguous()
    if not bool(torch.isfinite(block_modulations).all()):
        raise RuntimeError("Projected modulation contains NaN or Inf.")
    return block_modulations, meta


def prepare_block_modulations(
    *,
    forge_root: os.PathLike,
    models_root: os.PathLike,
    clip_model: str,
    prompts: tuple[str, str, str],
    adapter_mode: str,
    adapter_path: str,
    diffusion_model,
    weight: float,
) -> tuple[torch.Tensor, dict]:
    clip_path = resolve_clip_path(models_root, clip_model)
    resolved_adapter = resolve_adapter_path(
        models_root,
        adapter_mode,
        adapter_path,
    )
    pooled = encode_clip_pooled(clip_path, prompts, forge_root)
    adapter_state = load_adapter_cpu(resolved_adapter)
    block_modulations, meta = project_block_modulations(
        pooled,
        adapter_state,
        diffusion_model,
        weight,
    )
    return block_modulations, {
        **meta,
        "clip_path": str(clip_path),
        "adapter_path": str(resolved_adapter),
    }


def clear_modulation_caches() -> bool:
    """Release the persistent ~CLIP+adapter CPU cache on script unload (and for the '모델 내리기' button).

    Returns True when anything was cached."""
    global _CLIP_CACHE_KEY, _CLIP_CACHE_VALUE
    with _CACHE_LOCK:
        held = bool(_CPU_ADAPTER_CACHE or _POOLED_CACHE) or _CLIP_CACHE_VALUE is not None
        _CPU_ADAPTER_CACHE.clear()
        _VERIFIED_OFFICIAL_ADAPTERS.clear()
        _POOLED_CACHE.clear()
        _CLIP_CACHE_KEY = None
        _CLIP_CACHE_VALUE = None
    gc.collect()
    return held


__all__ = [
    "ADAPTER_FILE_NAME",
    "ADAPTER_SHA256",
    "ADAPTER_SIZE",
    "ADAPTER_URL",
    "clear_modulation_caches",
    "default_adapter_path",
    "encode_clip_pooled",
    "list_clip_l_models",
    "load_adapter_cpu",
    "prepare_block_modulations",
    "project_block_modulations",
    "resolve_adapter_path",
    "resolve_clip_path",
    "validate_adapter_for_model",
]
