from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


SAM3_NAME = "SAM3 Mask"
EXTENSION_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = EXTENSION_ROOT / "outputs"
SUPPORTED_CHECKPOINT_SUFFIXES = (".pt", ".safetensors")


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


def _ensure_bpe_vocab() -> str | None:
    target = EXTENSION_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if target.exists():
        return str(target)

    # Try to find it in other common Forge locations
    paths = _safe_import_webui_modules()
    if paths is not None:
        webui_root = Path(paths.models_path).parent
        # Candidate 1: forge_legacy_preprocessors (OneFormer data)
        candidate = (
            webui_root
            / "extensions-builtin"
            / "forge_legacy_preprocessors"
            / "annotator"
            / "oneformer"
            / "oneformer"
            / "data"
            / "bpe_simple_vocab_16e6.txt.gz"
        )
        if candidate.exists():
            return str(candidate)

    # Fallback to downloading if possible, using a known public repo that has this exact file
    try:
        from huggingface_hub import hf_hub_download

        target.parent.mkdir(parents=True, exist_ok=True)
        return hf_hub_download(
            repo_id="facebook/sam2",
            filename="sam2/assets/bpe_simple_vocab_16e6.txt.gz",
            local_dir=str(EXTENSION_ROOT),
        )
    except Exception:
        pass

    return None


def find_checkpoint_options() -> list[str]:
    """Return SAM3 detection checkpoint *filenames* (basenames, not full
    paths) for the UI dropdown.

    Filters to files whose name starts with ``sam3`` so non-SAM3 weights
    stored next to them (e.g. ``anima-lllite-inpainting-v2.safetensors``,
    which lives in the same ``models/sam3/`` folder for the CN dropdown)
    don't pollute the SAM3 Checkpoint selector. ``resolve_checkpoint_path``
    handles basename → absolute path resolution at load time.
    """
    paths = _safe_import_webui_modules()
    seen_names: set[str] = set()
    result: list[str] = []

    def _consider(path: Path) -> None:
        name = path.name
        if not name.lower().startswith("sam3"):
            return
        if name in seen_names:
            return
        seen_names.add(name)
        result.append(name)

    if paths is not None:
        models_root = Path(paths.models_path)
        for suffix in SUPPORTED_CHECKPOINT_SUFFIXES:
            for p in sorted((models_root / "sam3").glob(f"*{suffix}")):
                _consider(p)
            for p in sorted(models_root.glob(f"sam3*{suffix}")):
                _consider(p)
    for suffix in SUPPORTED_CHECKPOINT_SUFFIXES:
        for p in sorted((EXTENSION_ROOT / "models").glob(f"*{suffix}")):
            _consider(p)

    if not result:
        result.append("sam3.pt")
    return result


def resolve_checkpoint_path(checkpoint_value: str, allow_huggingface: bool = True) -> Path | None:
    value = (checkpoint_value or "").strip()
    if not value:
        return None if allow_huggingface else Path("sam3.pt")
    if value.lower() in {"auto", "huggingface"}:
        return None if allow_huggingface else Path("sam3.pt")

    path = Path(value)
    if path.is_absolute():
        return path

    paths = _safe_import_webui_modules()
    if paths is not None:
        webui_root = Path(paths.models_path).parent
        models_root = Path(paths.models_path)
        basename = path.name
        candidates = [
            models_root / "sam3" / value,
            models_root / "sam3" / basename,
            models_root / value,
            webui_root / value,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    candidate = EXTENSION_ROOT / value
    if candidate.exists():
        return candidate
    candidate = EXTENSION_ROOT / "models" / path.name
    if candidate.exists():
        return candidate
    return path


def _load_state_dict_from_file(checkpoint_path: Path) -> dict:
    suffix = checkpoint_path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(checkpoint_path))

    import torch

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    return ckpt


def _apply_sam3_checkpoint(model, checkpoint_path: Path) -> None:
    ckpt = _load_state_dict_from_file(checkpoint_path)

    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
    }
    if getattr(model, "inst_interactive_predictor", None) is not None:
        sam3_image_ckpt.update(
            {
                k.replace("tracker.", "inst_interactive_predictor.model."): v
                for k, v in ckpt.items()
                if "tracker" in k
            }
        )
    if not sam3_image_ckpt:
        sam3_image_ckpt = dict(ckpt)

    missing, unexpected = model.load_state_dict(sam3_image_ckpt, strict=False)
    if missing:
        print(
            f"[-] SAM3: loaded {checkpoint_path.name} with {len(missing)} missing key(s).",
            file=sys.stderr,
        )
    if unexpected:
        print(
            f"[-] SAM3: loaded {checkpoint_path.name} with {len(unexpected)} unexpected key(s).",
            file=sys.stderr,
        )


def unload_sam3() -> None:
    """Evict every cached SAM3 model bundle and reclaim VRAM.

    SAM3 is ~3.5 GB on GPU. Once loaded the ``@lru_cache(maxsize=4)`` keeps
    it pinned for the lifetime of the process, which leaves the downstream
    inpaint sampler fighting for memory (and triggers Forge's
    ``--reserve-vram`` warning on smaller GPUs). When ``sam3_unload_after``
    is on we call this between detection and the inpaint pass: next
    detection re-loads from disk (~3-5 s) but the sampler gets the full
    free VRAM in the meantime.
    """
    import gc

    _load_model_bundle.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# maxsize=2: each cached SAM3 bundle is ~3.5 GB on GPU. 4 was excessive;
# 2 lets the user A/B between sam3.pt and sam3.safetensors without holding
# stale entries forever. ``unload_sam3()`` clears this for full VRAM reclaim.
@lru_cache(maxsize=2)
def _load_model_bundle(checkpoint_key: str, device: str):
    checkpoint_path = None if checkpoint_key == "__hf__" else Path(checkpoint_key)
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise RuntimeError("SAM3 package is not installed in the Forge environment.") from exc

    bpe_path = _ensure_bpe_vocab()

    suffix = checkpoint_path.suffix.lower() if checkpoint_path else ""
    if checkpoint_path is not None and suffix != ".pt":
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            checkpoint_path=None,
            load_from_HF=False,
        )
        _apply_sam3_checkpoint(model, checkpoint_path)
    else:
        model = build_sam3_image_model(
            bpe_path=bpe_path,
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


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return int((x2 - x1) * (y2 - y1))


def _restrict_mask_to_box(mask: np.ndarray, box: np.ndarray, height: int, width: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
    pad = max(8, int(round(max(x2 - x1, y2 - y1) * 0.08)))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)

    clipped = np.zeros_like(mask, dtype=bool)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped


def _split_prompt_groups(prompt: str) -> list[list[str]]:
    """Split into independent groups with `/`, then OR-tokens with `,;|\\n` inside each group.

    Example: "face, eyes, hair / hand" -> [["face", "eyes", "hair"], ["hand"]]
    Each group becomes a single mask (its tokens OR'd); groups stay separate so that
    Individual mask mode can inpaint each group in its own pass.
    """
    groups: list[list[str]] = []
    for raw_group in re.split(r"/", prompt or ""):
        tokens = [t.strip() for t in re.split(r"[,;|\n]", raw_group)]
        tokens = [t for t in tokens if t]
        if tokens:
            groups.append(tokens)
    return groups


def _dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.astype(bool)
    import cv2

    k = 2 * int(px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    return dilated.astype(bool)


def _convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    """Wrap each connected region of ``mask`` in its convex hull.

    Useful for hair / fur / antennae where SAM3 catches the main silhouette
    but misses thin strands that stick out — the hull naturally includes the
    air between strands so a follow-up inpaint can redraw the whole shape.

    Per-component (not one global hull over all components) so distinct
    detected regions stay separate.
    """
    if not np.any(mask):
        return mask.astype(bool)
    import cv2

    mask_u8 = mask.astype(np.uint8) * 255
    num_labels, labels = cv2.connectedComponents(mask_u8)
    out = np.zeros_like(mask_u8)
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hull = cv2.convexHull(np.vstack(contours))
        cv2.fillPoly(out, [hull], 255)
    return (out > 0).astype(bool)


def _clean_split_masks(masks: np.ndarray, boxes: np.ndarray, height: int, width: int) -> list[np.ndarray]:
    split_masks = _split_masks(masks, height, width)
    if not split_masks:
        return []
    if boxes.size == 0:
        return split_masks

    box_rects = [tuple(int(round(v)) for v in box.tolist()) for box in boxes]
    cleaned_masks: list[np.ndarray] = []
    for mask in split_masks:
        mask_box = _mask_bbox(mask)
        if mask_box is None:
            continue

        best_index = max(
            range(len(box_rects)),
            key=lambda idx: _intersection_area(mask_box, box_rects[idx]),
        )
        cleaned = _restrict_mask_to_box(mask, boxes[best_index], height, width)
        if np.any(cleaned):
            cleaned_masks.append(cleaned)
    return cleaned_masks


def run_sam3_on_pil(
    image: Image.Image,
    prompt: str,
    threshold: float,
    checkpoint_value: str,
    device: str,
    allow_huggingface: bool = True,
    mask_dilation: int = 0,
    mask_hull: bool = False,
) -> Sam3Result:
    import torch

    resolved_device = _resolve_device(device)
    checkpoint_path = resolve_checkpoint_path(checkpoint_value, allow_huggingface=allow_huggingface)
    if checkpoint_path is not None and checkpoint_path.suffix.lower() not in SUPPORTED_CHECKPOINT_SUFFIXES:
        raise RuntimeError(
            f"Unsupported SAM3 checkpoint: {checkpoint_path.name}. "
            f"Expected one of: {', '.join(SUPPORTED_CHECKPOINT_SUFFIXES)}."
        )
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")

    checkpoint_key = "__hf__" if checkpoint_path is None else str(checkpoint_path.resolve())
    _, processor = _load_model_bundle(checkpoint_key, resolved_device)
    processor.set_confidence_threshold(float(threshold))

    pil_image = image.convert("RGB")
    rgb = np.asarray(pil_image)
    h, w = rgb.shape[:2]

    groups = _split_prompt_groups(prompt) or [[prompt or ""]]

    group_masks: list[np.ndarray] = []
    all_boxes: list[np.ndarray] = []
    all_scores: list[float] = []

    def _run_prompt(sub_prompt: str):
        if resolved_device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(pil_image)
                state = processor.set_text_prompt(prompt=sub_prompt, state=state)
        else:
            state = processor.set_image(pil_image)
            state = processor.set_text_prompt(prompt=sub_prompt, state=state)

        boxes_np = _to_numpy(state.get("boxes", []))
        scores_np = _to_numpy(state.get("scores", []))
        masks_np = _to_numpy(state.get("masks", []))
        if masks_np.size == 0:
            return [], boxes_np, scores_np

        split = _clean_split_masks(masks_np, boxes_np, h, w)
        return split, boxes_np, scores_np

    for group_tokens in groups:
        group_split: list[np.ndarray] = []
        for sub_prompt in group_tokens:
            split, boxes_np, scores_np = _run_prompt(sub_prompt)
            group_split.extend(split)
            if boxes_np.size:
                for box in boxes_np:
                    all_boxes.append(box.astype(float))
            if scores_np.size:
                for score in scores_np.tolist():
                    all_scores.append(float(score))

        if not group_split:
            continue
        group_mask = np.any(np.stack(group_split, axis=0), axis=0)
        if mask_hull:
            group_mask = _convex_hull_mask(group_mask)
        group_mask = _dilate_mask(group_mask, mask_dilation)
        if np.any(group_mask):
            group_masks.append(group_mask)

    if group_masks:
        combined_mask = np.any(np.stack(group_masks, axis=0), axis=0).astype(np.uint8) * 255
        individual_masks = [Image.fromarray(mask.astype(np.uint8) * 255, mode="L") for mask in group_masks]
    else:
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        individual_masks = []

    boxes_out = np.stack(all_boxes, axis=0) if all_boxes else np.zeros((0, 4), dtype=np.float32)
    scores_out = np.array(all_scores, dtype=np.float32) if all_scores else np.zeros((0,), dtype=np.float32)

    overlay = rgb.copy()
    overlay[combined_mask > 0] = (
        overlay[combined_mask > 0] * 0.35 + np.array([30, 210, 255]) * 0.65
    ).astype(np.uint8)

    import cv2

    for idx, box in enumerate(boxes_out):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        overlay = np.ascontiguousarray(overlay)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 120, 255), 2)
        label = f"{float(scores_out[idx]):.2f}" if idx < len(scores_out) else "n/a"
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
        boxes=boxes_out.astype(float).tolist(),
        scores=[float(score) for score in scores_out.tolist()],
        device=resolved_device,
        checkpoint=str(checkpoint_path) if checkpoint_path else "facebook/sam3::sam3.pt",
    )


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_for_filename(text: str, max_len: int = 32) -> str:
    """Reduce a free-form prompt to a filesystem-safe slug for artifact
    filenames. Returns ``"mask"`` when empty so we never produce ``__``."""
    if not text:
        return "mask"
    cleaned = _FILENAME_SAFE.sub("_", text.strip())
    cleaned = cleaned.strip("_") or "mask"
    return cleaned[:max_len]


def write_artifacts(result: Sam3Result, seed: int | None, label: str | None = None) -> dict[str, str]:
    """Persist the SAM3 mask/overlay/meta to disk.

    ``label`` (typically the detect prompt) becomes part of the filename so
    runs with different masks don't all read as ``..._face_...``. Falls back
    to ``"mask"`` when no label is given.
    """
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _sanitize_for_filename(label or "")
    stem = f"sam3_{seed}" if seed is not None else "sam3"
    # Safeguard: cap at 10000 to avoid runaway loops if the output dir gets
    # thousands of artifacts and the while True can't find a free slot fast.
    suffix = ""
    mask_path = overlay_path = meta_path = None
    for index in range(1, 10001):
        suffix = "" if index == 1 else f"_{index}"
        mask_path = output_dir / f"{stem}_{slug}_mask{suffix}.png"
        overlay_path = output_dir / f"{stem}_{slug}_overlay{suffix}.png"
        meta_path = output_dir / f"{stem}_{slug}_prompt{suffix}.json"
        if not mask_path.exists() and not overlay_path.exists() and not meta_path.exists():
            break
    else:
        # Hit the cap — fall back to a timestamp suffix so saves still succeed.
        import time as _time

        suffix = f"_{int(_time.time() * 1000)}"
        mask_path = output_dir / f"{stem}_{slug}_mask{suffix}.png"
        overlay_path = output_dir / f"{stem}_{slug}_overlay{suffix}.png"
        meta_path = output_dir / f"{stem}_{slug}_prompt{suffix}.json"

    result.mask.save(mask_path)
    result.overlay.save(overlay_path)
    for idx, mask in enumerate(result.masks, start=1):
        single_mask_path = output_dir / f"{stem}_{slug}_mask_{idx:02d}{suffix}.png"
        mask.save(single_mask_path)
    meta_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "device": result.device,
                "checkpoint": result.checkpoint,
                "label": label,
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
