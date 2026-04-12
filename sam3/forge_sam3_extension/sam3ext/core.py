from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


SAM3_NAME = "SAM3 Mask"
EXTENSION_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = EXTENSION_ROOT / "outputs"


def _safe_import_webui_modules():
    try:
        from modules import paths  # type: ignore
    except Exception:
        return None
    return paths


@dataclass
class Sam3Result:
    mask: Image.Image
    masks: list[Image.Image]
    overlay: Image.Image
    boxes: list[list[float]]
    scores: list[float]
    device: str
    checkpoint: str


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def find_checkpoint_options() -> list[str]:
    paths = _safe_import_webui_modules()
    candidates: list[Path] = []
    if paths is not None:
        models_root = Path(paths.models_path)
        candidates.extend(sorted((models_root / "sam3").glob("*.pt")))
    candidates.extend(sorted((EXTENSION_ROOT / "models").glob("*.pt")))

    seen: set[str] = set()
    result: list[str] = []
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(str(path))
    if not result:
        result.append("models/sam3.pt")
    return result


def resolve_checkpoint_path(checkpoint_value: str, allow_huggingface: bool = True) -> Path | None:
    value = (checkpoint_value or "").strip()
    if not value:
        return None if allow_huggingface else Path("models/sam3.pt")
    if value.lower() in {"auto", "huggingface"}:
        return None if allow_huggingface else Path("models/sam3.pt")

    path = Path(value)
    if path.is_absolute():
        return path

    paths = _safe_import_webui_modules()
    if paths is not None:
        models_root = Path(paths.models_path) / "sam3"
        candidate = models_root / value
        if candidate.exists():
            return candidate
    candidate = EXTENSION_ROOT / value
    if candidate.exists():
        return candidate
    return path


@lru_cache(maxsize=4)
def _load_model_bundle(checkpoint_key: str, device: str):
    checkpoint_path = None if checkpoint_key == "__hf__" else Path(checkpoint_key)
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise RuntimeError("SAM3 package is not installed in the Forge environment.") from exc

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        load_from_HF=checkpoint_path is None,
    )
    processor = Sam3Processor(model, device=device)
    return model, processor


def _to_numpy(value) -> np.ndarray:
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _split_masks(masks: np.ndarray, height: int, width: int) -> list[np.ndarray]:
    if masks.size == 0:
        return []
    masks = masks.astype(bool)
    if masks.ndim == 2:
        return [masks]
    flat = masks.reshape((-1, height, width))
    return [mask for mask in flat if np.any(mask)]


def run_sam3_on_pil(
    image: Image.Image,
    prompt: str,
    threshold: float,
    checkpoint_value: str,
    device: str,
    allow_huggingface: bool = True,
) -> Sam3Result:
    import torch

    resolved_device = _resolve_device(device)
    checkpoint_path = resolve_checkpoint_path(checkpoint_value, allow_huggingface=allow_huggingface)
    if checkpoint_path is not None and checkpoint_path.suffix.lower() != ".pt":
        raise RuntimeError(f"Unsupported SAM3 checkpoint: {checkpoint_path.name}. Expected a .pt file.")
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")

    checkpoint_key = "__hf__" if checkpoint_path is None else str(checkpoint_path.resolve())
    _, processor = _load_model_bundle(checkpoint_key, resolved_device)
    processor.set_confidence_threshold(float(threshold))

    pil_image = image.convert("RGB")
    if resolved_device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = processor.set_image(pil_image)
            state = processor.set_text_prompt(prompt=prompt, state=state)
    else:
        state = processor.set_image(pil_image)
        state = processor.set_text_prompt(prompt=prompt, state=state)

    boxes = _to_numpy(state.get("boxes", []))
    scores = _to_numpy(state.get("scores", []))
    masks = _to_numpy(state.get("masks", []))

    rgb = np.asarray(pil_image)
    h, w = rgb.shape[:2]
    if masks.size == 0:
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        individual_masks: list[Image.Image] = []
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
    else:
        split_masks = _split_masks(masks, h, w)
        if split_masks:
            combined_mask = np.any(np.stack(split_masks, axis=0), axis=0).astype(np.uint8) * 255
            individual_masks = [Image.fromarray(mask.astype(np.uint8) * 255, mode="L") for mask in split_masks]
        else:
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            individual_masks = []

    overlay = rgb.copy()
    overlay[combined_mask > 0] = (
        overlay[combined_mask > 0] * 0.35 + np.array([30, 210, 255]) * 0.65
    ).astype(np.uint8)

    import cv2

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        overlay = np.ascontiguousarray(overlay)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 120, 255), 2)
        label = f"{float(scores[idx]):.2f}" if idx < len(scores) else "n/a"
        cv2.putText(
            overlay,
            label,
            (x1, max(y1 - 10, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 120, 255),
            2,
            cv2.LINE_AA,
        )

    return Sam3Result(
        mask=Image.fromarray(combined_mask, mode="L"),
        masks=individual_masks,
        overlay=Image.fromarray(overlay),
        boxes=boxes.astype(float).tolist(),
        scores=[float(score) for score in scores.tolist()],
        device=resolved_device,
        checkpoint=str(checkpoint_path) if checkpoint_path else "facebook/sam3::sam3.pt",
    )


def write_artifacts(result: Sam3Result, seed: int | None) -> dict[str, str]:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sam3_{seed}" if seed is not None else "sam3"
    index = 1
    while True:
        suffix = "" if index == 1 else f"_{index}"
        mask_path = output_dir / f"{stem}_face_mask{suffix}.png"
        overlay_path = output_dir / f"{stem}_face_overlay{suffix}.png"
        meta_path = output_dir / f"{stem}_face_prompt{suffix}.json"
        if not mask_path.exists() and not overlay_path.exists() and not meta_path.exists():
            break
        index += 1

    result.mask.save(mask_path)
    result.overlay.save(overlay_path)
    for idx, mask in enumerate(result.masks, start=1):
        single_mask_path = output_dir / f"{stem}_face_mask_{idx:02d}{suffix}.png"
        mask.save(single_mask_path)
    meta_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "device": result.device,
                "checkpoint": result.checkpoint,
                "boxes": result.boxes,
                "scores": result.scores,
                "mask_count": len(result.masks),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"mask": str(mask_path), "overlay": str(overlay_path), "meta": str(meta_path)}
