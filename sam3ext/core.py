from __future__ import annotations

import json
import re
import sys
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


SAM3_NAME = "SAM3 Mask"
EXTENSION_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = EXTENSION_ROOT / "outputs"
SUPPORTED_CHECKPOINT_SUFFIXES = (".pt", ".safetensors")
# Hugging Face facebook/sam3 가 내려 주는 파일 이름 = 체크포인트가 하나도 없을 때 드롭다운 기본값. 이 이름을
# 로컬에서 못 찾으면(allow_huggingface 일 때만) HF 자동 다운로드로 해석한다(감사 M3).
HF_CHECKPOINT_REPO = "facebook/sam3"
HF_CHECKPOINT_NAME = "sam3.pt"


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
    if allow_huggingface and len(path.parts) == 1 and path.name.lower() == HF_CHECKPOINT_NAME:
        # 로컬에 없는 기본값 'sam3.pt' — README 대로 HF(facebook/sam3) 에서 받는다. 다른 이름은 오류 그대로.
        return None
    return path


def _missing_checkpoint_message(checkpoint_path: Path, allow_huggingface: bool) -> str:
    message = (
        f"SAM3 checkpoint not found: {checkpoint_path}. Put it in <webui>/models/sam3/ "
        f"(official weights: https://huggingface.co/{HF_CHECKPOINT_REPO})."
    )
    if not allow_huggingface and checkpoint_path.name.lower() == HF_CHECKPOINT_NAME:
        message += (
            f" Automatic download from Hugging Face ({HF_CHECKPOINT_REPO}) is off because "
            "--sam3-no-huggingface is set."
        )
    return message


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


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# Forge 설정(Settings → SAM3 Mask): 'Unload after' 뒤 번들을 CPU RAM 에 보관할지. 기본 True = 보관(지금 동작).
# 끄면 예전처럼 완전히 해제한다 — RAM 약 3.4 GB 가 비는 대신 다음 검출마다 새로 빌드(이미지당 2.5~5 초).
# 등록은 scripts/!sam3.py 의 on_ui_settings.
OPT_UNLOAD_KEEP_IN_RAM = "sam3_unload_keep_in_ram"


def _keep_bundle_in_ram() -> bool:
    """Forge 설정값. 설정이 없거나(Forge 밖·아직 저장 안 됨) 읽을 수 없으면 기본값(보관)."""
    try:
        from modules import shared  # type: ignore

        return bool(getattr(shared.opts, OPT_UNLOAD_KEEP_IN_RAM, True))
    except Exception:
        return True


def describe_unload(kept: bool, *, after_failure: bool = False) -> str:
    """``unload_sam3()`` 결과를 로그 한 줄로 — 실제로 한 일(CPU RAM 보관 / 완전 해제)을 말한다."""
    if kept:
        head, tail = "model moved from VRAM to CPU RAM", "moves back on next detection"
    else:
        head, tail = "model released from VRAM and RAM", "reloads on next detection"
    return f"{head}{' after the failure' if after_failure else ''} ({tail})."


def unload_sam3(keep_in_ram: bool | None = None) -> bool:
    """SAM3 번들을 VRAM 에서 내린다 — 기본은 버리지 않고 CPU 로만 옮겨 캐시에 남긴다.

    SAM3 is ~3.5 GB on GPU. ``sam3_unload_after`` 가 켜져 있으면 검출과 인페인트 사이에 이것을 불러
    샘플러가 VRAM 을 다 쓰게 한다. 예전엔 여기서 번들을 버려서 다음 이미지마다 무작위 초기화 →
    3.45 GB torch.load → .cuda() 를 처음부터 다시 했다(이미지당 2.5~5 초, 효율 감사 [1]). 이제 다음
    검출은 CPU 사본을 원래 장치로 옮기기만 한다 — 값을 그대로 복사하므로 결과는 비트 동일하다.
    대가는 RAM 약 3.4 GB. 진짜 해제는 ``release_sam3``(체크포인트·장치가 바뀔 때, 스크립트 언로드 때).

    ``keep_in_ram`` 이 None 이면 Forge 설정(``OPT_UNLOAD_KEEP_IN_RAM``, 기본 True)을 따른다. False 면 예전처럼
    ``release_sam3`` 로 완전히 해제한다. 돌려주는 값: 번들이 지금 CPU RAM 에 보관돼 있으면 True.
    """
    import gc

    if keep_in_ram is None:
        keep_in_ram = _keep_bundle_in_ram()
    if not keep_in_ram:
        release_sam3()
        return False

    global _BUNDLE_OFFLOADED
    with _BUNDLE_LOCK:
        if _BUNDLE is not None and _BUNDLE_OFFLOADED is None:
            try:
                _BUNDLE_OFFLOADED = _offload_bundle(*_BUNDLE)
            except Exception:
                # 반쯤 옮겨진 번들은 믿을 수 없다 — 예전처럼 통째로 버린다(다음 검출이 새로 빌드).
                release_sam3()
        kept = _BUNDLE is not None and _BUNDLE_OFFLOADED is not None
    gc.collect()
    _empty_cuda_cache()
    return kept


def drop_offloaded_sam3() -> bool:
    """CPU RAM 에 보관 중인('Unload after' 로 내려간) 번들만 버린다 — 설정을 끈 즉시 RAM 을 비우려고.

    장치에 올라가 있는 번들(검출 중이거나 'Unload after' 를 끈 경우)은 건드리지 않는다. 버렸으면 True.
    """
    with _BUNDLE_LOCK:
        if _BUNDLE is None or _BUNDLE_OFFLOADED is None:
            return False
        release_sam3()
        return True


def release_sam3() -> None:
    """캐시한 SAM3 번들을 진짜로 버리고 VRAM 을 회수한다.

    체크포인트(경로·파일 내용)나 장치가 바뀌었을 때(``_get_model_bundle``), 스크립트가 언로드될 때
    (Reload UI — ``scripts/!sam3.py`` 의 ``on_script_unloaded``) 부른다. 버리기 전에 GPU 텐서의 저장소를
    푼다 — 번들을 다른 누가 아직 쥐고 있어도(예: 검출 중 난 예외의 traceback 프레임) VRAM 은 비게.
    """
    import gc

    global _BUNDLE, _BUNDLE_KEY, _BUNDLE_OFFLOADED
    with _BUNDLE_LOCK:
        bundle = _BUNDLE
        _BUNDLE = None
        _BUNDLE_KEY = None
        _BUNDLE_OFFLOADED = None
        if bundle is not None:
            try:
                _discard_bundle(*bundle)
            except Exception:
                pass
        del bundle
    gc.collect()
    _empty_cuda_cache()


# 캐시한 SAM3 번들 하나(약 3.5 GB). 16 GB GPU 에서 Anima/Qwen + LLLite 가 이미 OOM 경계라 여러 개는 두지
# 않는다. 키 = (체크포인트 키, 파일 mtime_ns·크기, 장치) — 같은 경로의 파일이 바뀌어도 새로 빌드한다.
# _BUNDLE_OFFLOADED 가 None 이 아니면 번들은 CPU 에 내려가 있고, 값은 {텐서 id: 원래 장치} 다.
_BUNDLE_LOCK = threading.RLock()
_BUNDLE_KEY: tuple | None = None
_BUNDLE: tuple[Any, Any] | None = None
_BUNDLE_OFFLOADED: dict[int, Any] | None = None

# nn.Module 의 이 속성들은 _apply 가 다룬다 — 나머지 일반 속성만 훑는다.
_MODULE_SLOT_ATTRS = frozenset({"_parameters", "_buffers", "_modules"})


def _map_nested_tensors(value: Any, fn: Callable, depth: int = 0) -> Any:
    """``value`` 안의 텐서(일반 텐서, dict·list·tuple 안, 3단계까지)에 ``fn`` 을 적용한다.

    dict·list 는 제자리에서 바꾸고, 텐서·tuple 은 바뀐 새 객체를 돌려준다(안 바뀌면 같은 객체).
    nn.Parameter 와 nn.Module 은 건드리지 않는다 — 모듈의 ``_apply`` 가 맡는다.
    """
    import torch

    if isinstance(value, torch.nn.Parameter) or isinstance(value, torch.nn.Module):
        return value
    if isinstance(value, torch.Tensor):
        return fn(value)
    if depth >= 3:
        return value
    if isinstance(value, dict):
        for key, item in list(value.items()):
            mapped = _map_nested_tensors(item, fn, depth + 1)
            if mapped is not item:
                value[key] = mapped
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            mapped = _map_nested_tensors(item, fn, depth + 1)
            if mapped is not item:
                value[index] = mapped
        return value
    if isinstance(value, tuple):
        items = [_map_nested_tensors(item, fn, depth + 1) for item in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        return type(value)._make(items) if hasattr(type(value), "_make") else type(value)(items)
    return value


def _map_bundle_tensors(model, processor, fn: Callable) -> None:
    """번들의 모든 텐서에 ``fn`` 을 적용한다: 파라미터·버퍼(``model._apply``)와 state_dict 밖 텐서.

    state_dict 밖 텐서 — sam3 가 ``__init__`` 에서 ``device="cuda"`` 로 바로 만든 일반 속성
    (``PositionEmbeddingSine.cache`` 4개 약 86 MB, ``TransformerDecoder.compilable_cord_cache``),
    forward 중에 채우는 캐시(``cache``·``coord_cache``), ``Sam3Processor.find_stage`` 의 id 텐서.
    ``nn.Module.to`` 는 이것들을 옮기지 않아서, 번들을 CPU 에 남기면 GPU 에 그대로 남는다.
    """
    model._apply(fn)
    for module in model.modules():
        attrs = vars(module)
        for name, value in list(attrs.items()):
            if name in _MODULE_SLOT_ATTRS:
                continue
            mapped = _map_nested_tensors(value, fn)
            if mapped is not value:
                attrs[name] = mapped
    for holder in (processor, getattr(processor, "find_stage", None)):
        if holder is None or not hasattr(holder, "__dict__"):
            continue
        attrs = vars(holder)
        for name, value in list(attrs.items()):
            mapped = _map_nested_tensors(value, fn)
            if mapped is not value:
                attrs[name] = mapped


def _park_bundle_tensors(model, processor, needs_park: Callable, park: Callable, origin: Callable) -> dict[int, Any]:
    """``needs_park`` 인 텐서를 ``park`` 로 바꾸고 {텐서 id: ``origin(원래 텐서)``} 를 돌려준다.

    파라미터는 ``_apply`` 가 같은 Parameter 객체의 ``.data`` 만 바꾸므로 Parameter 의 id 를, 나머지는
    새로 생긴 텐서(슬롯에 들어가 내려가 있는 동안 살아 있음)의 id 를 기록한다 — ``_restore_bundle`` 이
    같은 슬롯을 훑으며 그 id 로 찾아 ``tensor.to(origin)`` 한다.
    """
    import torch

    saved: dict[int, Any] = {}

    def fn(tensor):
        if not needs_park(tensor):
            return tensor
        moved = park(tensor)
        saved[id(tensor) if isinstance(tensor, torch.nn.Parameter) else id(moved)] = origin(tensor)
        return moved

    _map_bundle_tensors(model, processor, fn)
    return saved


def _offload_bundle(model, processor) -> dict[int, Any]:
    """번들의 CPU 밖 텐서를 전부 CPU 로 옮기고 {텐서 id: 원래 장치} 를 돌려준다."""
    return _park_bundle_tensors(
        model,
        processor,
        needs_park=lambda tensor: tensor.device.type != "cpu",
        park=lambda tensor: tensor.to("cpu"),
        origin=lambda tensor: tensor.device,
    )


def _discard_bundle(model, processor) -> None:
    """버릴 번들의 GPU 텐서를 저장소 없는 meta 텐서로 바꿔 VRAM 만 푼다 — CPU 로 복사(D2H 3.4 GB)하지 않는다.
    이 번들은 다시 쓰지 않는다(CPU 에 있는 텐서는 참조가 끊기면 풀린다)."""
    _park_bundle_tensors(
        model,
        processor,
        needs_park=lambda tensor: tensor.device.type not in ("cpu", "meta"),
        park=lambda tensor: tensor.to("meta"),
        origin=lambda tensor: tensor.device,
    )


def _restore_bundle(model, processor, saved: dict[int, Any]) -> None:
    """``_offload_bundle`` 이 옮긴 텐서만 기록한 원래 장치로 되돌린다(값은 그대로 복사)."""

    def back(tensor):
        device = saved.get(id(tensor))
        return tensor if device is None else tensor.to(device)

    _map_bundle_tensors(model, processor, back)


def _checkpoint_identity(checkpoint_key: str) -> tuple:
    if checkpoint_key == "__hf__":
        return ()
    try:
        stat = Path(checkpoint_key).stat()
    except OSError:
        return ()
    return (stat.st_mtime_ns, stat.st_size)


def _get_model_bundle(checkpoint_key: str, device: str):
    """(model, processor) — 캐시에 있으면 그대로(CPU 에 내려가 있으면 원래 장치로 되돌려서), 없거나
    체크포인트·장치가 바뀌었으면 옛 번들을 해제하고 새로 빌드한다."""
    global _BUNDLE, _BUNDLE_KEY, _BUNDLE_OFFLOADED
    key = (checkpoint_key, _checkpoint_identity(checkpoint_key), device)
    with _BUNDLE_LOCK:
        if _BUNDLE is not None and _BUNDLE_KEY != key:
            release_sam3()
        if _BUNDLE is None:
            bundle = _load_model_bundle(checkpoint_key, device)
            _BUNDLE, _BUNDLE_KEY, _BUNDLE_OFFLOADED = bundle, key, None
            return bundle
        if _BUNDLE_OFFLOADED is not None:
            try:
                _restore_bundle(*_BUNDLE, _BUNDLE_OFFLOADED)
            except BaseException:
                # 되돌리다 실패(OOM 등) — 반쯤 올라간 번들은 버리고 오류를 올린다. 다음 검출은 새로 빌드.
                release_sam3()
                raise
            _BUNDLE_OFFLOADED = None
        return _BUNDLE


@contextmanager
def _preserve_torch_tf32():
    """Forge 프로세스 전역의 TF32 설정을 지킨다.

    ``sam3/model_builder.py`` 는 import 될 때 ``_setup_tf32()`` 로 ``torch.backends.cuda.matmul.allow_tf32`` 와
    ``torch.backends.cudnn.allow_tf32`` 를 True 로 바꾼다(Ampere 이상). 전역 설정이라 SAM3 를 한 번 쓴 뒤로는
    Forge 의 모든 생성에서 fp32 행렬곱(LyCORIS DoRA 합치기, Anima 3.8B 커넥터 등)이 TF32 로 돌아 같은 시드·
    설정의 결과가 SAM3 사용 전과 달라졌다. import·빌드 동안 바뀐 값을 원래대로 되돌린다.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - Forge 밖
        yield
        return
    matmul_precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


@contextmanager
def _sam3_tf32():
    """SAM3 검출 동안에만 SAM3 가 의도한 TF32 를 켠다(예전 마스크와 같게) — 끝나면 Forge 설정으로 되돌린다."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        yield
        return
    with _preserve_torch_tf32():
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        yield


def _load_model_bundle(checkpoint_key: str, device: str):
    with _preserve_torch_tf32():
        return _build_model_bundle(checkpoint_key, device)


def _build_model_bundle(checkpoint_key: str, device: str):
    checkpoint_path = None if checkpoint_key == "__hf__" else Path(checkpoint_key)
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise RuntimeError("SAM3 package is not installed in the Forge environment.") from exc

    bpe_path = _ensure_bpe_vocab()

    if checkpoint_path is None:
        # 첫 사용이면 HF 캐시로 약 3.45 GB 를 받는다 — 멈춘 것처럼 보이지 않게 한 줄 알린다(번들을 빌드할 때만).
        print(
            f"[SAM3] no local SAM3 checkpoint — loading {HF_CHECKPOINT_NAME} from Hugging Face {HF_CHECKPOINT_REPO} "
            "(first use downloads ~3.45 GB into the HF cache; --sam3-no-huggingface turns this off)."
        )
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
        if value.dtype == torch.bool:
            # SAM3 의 masks 는 bool — float 로 바꿔 4배 크기로 옮긴 뒤 다시 bool 로 되돌릴 필요가 없다.
            return value.detach().cpu().numpy()
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


def _mask_roi(mask: np.ndarray, pad: int) -> tuple[int, int, int, int] | None:
    """``mask`` 의 bbox 를 ``pad`` 만큼 넓혀 이미지 안으로 자른 (y0, y1, x0, x1). 빈 마스크면 None."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask[rows[0]:rows[-1] + 1].any(axis=0))
    height, width = mask.shape[:2]
    return (
        max(0, int(rows[0]) - pad),
        min(height, int(rows[-1]) + 1 + pad),
        max(0, int(cols[0]) - pad),
        min(width, int(cols[-1]) + 1 + pad),
    )


# OpenCV dilate 는 면적이 2^20 픽셀 이상일 때만 여러 스레드로 나눈다(cv2 5.0 실측: 1024×1023 은 1스레드,
# 1024×1024 부터 병렬). 병렬일 때 속도는 모양·스레드 수에 따라 들쭉날쭉해서(24스레드 k=129: 1536² 17.6 ms,
# 1536×683 125 ms) 큰 커널에서 '조금 작은 ROI' 는 전체 프레임보다 몇 배 느릴 수 있다.
_CV_PARALLEL_MIN_PIXELS = 1 << 20


def _cv_roi_is_cheaper(roi: tuple[int, int, int, int], shape: tuple[int, ...], threads: int) -> bool:
    """ROI 만 dilate 하는 편이 확실히 싸면 True, 아니면 예전처럼 전체 프레임(False).

    프레임이 병렬 문턱보다 작으면 둘 다 1스레드라 일이 적은 ROI 가 이긴다. 프레임이 문턱 이상이면 ROI 가
    1스레드 영역(문턱 미만)이고 ROI×threads ≤ 프레임일 때만 — 1스레드 ROI 가 이상적인 병렬 전체보다 싸다.
    어느 쪽이든 결과는 같다(비트 동일); 속도만 고른다.
    """
    y0, y1, x0, x1 = roi
    frame = int(shape[0]) * int(shape[1])
    if frame < _CV_PARALLEL_MIN_PIXELS:
        return True
    area = (y1 - y0) * (x1 - x0)
    return area < _CV_PARALLEL_MIN_PIXELS and area * max(1, int(threads)) <= frame


def _dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.astype(bool)
    import cv2

    k = 2 * int(px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # 반지름 px 커널(중심 앵커)은 bbox 밖으로 px 까지만 번진다 → bbox±px ROI 만 팽창해도 픽셀 단위로 같다
    # (ROI 바깥은 원래 0 이고, dilate 의 기본 테두리는 max 에 영향이 없다). 효율 감사 [8].
    # 변환은 예전과 같은 astype(np.uint8) 로 — bbox 는 any() 라 그보다 좁지 않다.
    roi = _mask_roi(mask, int(px)) if mask.ndim == 2 else None
    if roi is None:
        if mask.ndim == 2:  # 빈 마스크
            return np.zeros(mask.shape, dtype=bool)
    elif _cv_roi_is_cheaper(roi, mask.shape, cv2.getNumThreads()):
        y0, y1, x0, x1 = roi
        out = np.zeros(mask.shape, dtype=bool)
        out[y0:y1, x0:x1] = cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), kernel).astype(bool)
        return out
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)  # 예전 경로(전체 프레임)
    return dilated.astype(bool)


def _edge_aware_edges(image_rgb: np.ndarray, canny_low: int = 100, canny_high: int = 200) -> np.ndarray:
    """``_edge_aware_dilate`` 가 막는 벽: Canny 가장자리를 3×3 로 1px 두껍게 한 bool 맵(전체 프레임).

    Canny 는 이미지 전체에서 돌려야 ROI 테두리 결과가 같으므로 잘라서 돌리지 않는다. run_sam3_on_pil 은
    같은 rgb 로 그룹마다 부르므로 run 안에서 한 번만 계산해 ``edges=`` 로 넘긴다.
    """
    import cv2

    if image_rgb.ndim == 3 and image_rgb.shape[2] >= 3:
        gray = cv2.cvtColor(image_rgb[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    # Canny returns 1-pixel-thick edges. Our 3×3 dilation can jump over a
    # single-pixel edge diagonally (3-connected expansion). Thicken the
    # edge map by 1 px so the wall is 3-thick and impossible to skip.
    edges = cv2.Canny(gray, canny_low, canny_high)
    return cv2.dilate(edges, kernel) > 0  # bool, thickened


def _edge_aware_dilate(
    mask: np.ndarray,
    image_rgb: np.ndarray,
    max_px: int,
    canny_low: int = 100,
    canny_high: int = 200,
    *,
    edges: np.ndarray | None = None,
) -> np.ndarray:
    """Grow ``mask`` outward up to ``max_px`` pixels, but stop the expansion
    at strong image edges (detected via Canny on the original RGB image).

    Use case: SAM3's silhouette frequently lands a few pixels INSIDE the
    actual garment boundary. A plain morphological dilation expands by N
    px everywhere — into the body, into adjacent objects, into the
    background — which is too aggressive. This routine grows the mask
    only into regions where there's no edge, which means it walks up to
    the natural object outline (e.g. a shirt's collar / hem) and stops
    there. Catches outline residue without bleeding into the next object.

    Iterative single-pixel dilation: at each step, candidate new pixels =
    dilation - mask; keep only those that aren't on an edge. Bails early
    when no new pixels can be added.

    ``edges`` 는 같은 이미지·임계값으로 만든 ``_edge_aware_edges`` 결과(생략하면 여기서 계산).
    """
    if max_px <= 0 or image_rgb is None:
        return mask.astype(bool)
    import cv2

    out = mask.astype(bool)
    if tuple(image_rgb.shape[:2]) != out.shape:
        raise ValueError(f"edge-aware dilate: image {image_rgb.shape[:2]} != mask {out.shape}")
    # 반복 max_px 번에 3×3 팽창이라 bbox 밖으로 max_px 까지만 자란다 → bbox±(max_px+1) ROI 안에서만
    # 반복해도 픽셀 단위로 같다(예전엔 반복마다 전체 프레임 7패스). 효율 감사 [8].
    roi = _mask_roi(out, int(max_px) + 1)
    if roi is None:
        return out
    if edges is None:
        edges = _edge_aware_edges(image_rgb, canny_low, canny_high)
    elif edges.shape != out.shape:
        raise ValueError(f"edge-aware dilate: edges {edges.shape} != mask {out.shape}")
    y0, y1, x0, x1 = roi
    sub = out[y0:y1, x0:x1]  # view — 아래 |= 가 out 에 바로 쓴다
    blocked = edges[y0:y1, x0:x1]
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(int(max_px)):
        dilated = cv2.dilate(sub.astype(np.uint8), kernel) > 0
        new_pixels = dilated & ~sub & ~blocked
        if not new_pixels.any():
            break
        sub |= new_pixels
    return out


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
    # 예전엔 성분마다 (labels == label) 전체 프레임 비교 + findContours 였다(성분 288개에 1.26 초, 효율 감사 [8]).
    # findContours 한 번으로 모든 성분의 바깥 경계를 얻는다. 8-연결 성분 하나에 바깥 경계는 정확히 하나라
    # 예전 '성분 하나만 남긴 이미지의 RETR_EXTERNAL' 과 같은 점들이다. RETR_EXTERNAL 이 아니라 RETR_CCOMP 인
    # 이유: 다른 성분의 구멍 안에 든 성분도 최상위(부모 -1)로 나와 예전처럼 자기 hull 을 따로 그린다.
    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask_u8)
    if hierarchy is None:
        return out > 0
    for contour, link in zip(contours, hierarchy[0]):
        if link[3] >= 0:  # 구멍 경계 — 바깥 경계 hull 안에 있으므로 예전에도 쓰지 않았다
            continue
        hull = cv2.convexHull(contour)
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


def _manual_mask_result(image: Image.Image, user_mask: np.ndarray) -> Sam3Result:
    """Return the same pre-blur manual mask contract without loading SAM3.

    The existing fallback uses the scribble after text-side hull/outline/
    dilation, so those operations must not be added to this manual-only path.
    Inpaint blur and inversion still happen downstream in Forge.
    """
    pil_image = image.convert("RGB")
    rgb = np.asarray(pil_image)
    h, w = rgb.shape[:2]
    manual = np.asarray(user_mask, dtype=bool)
    if manual.ndim != 2:
        raise ValueError("SAM3 manual mask must be a two-dimensional array")
    if manual.shape != (h, w):
        resized = Image.fromarray(manual.astype(np.uint8) * 255, mode="L").resize(
            (w, h), Image.NEAREST
        )
        manual = np.asarray(resized) > 127
    mask = Image.fromarray(manual.astype(np.uint8) * 255, mode="L")
    overlay = rgb.copy()
    overlay[manual] = (
        overlay[manual] * 0.35 + np.array([30, 210, 255]) * 0.65
    ).astype(np.uint8)
    print("[-] SAM3: manual mask only; text detection and checkpoint loading skipped", file=sys.stderr)
    return Sam3Result(
        mask=mask,
        masks=[mask.copy()] if bool(manual.any()) else [],
        overlay=Image.fromarray(overlay),
        boxes=[],
        scores=[],
        device="manual",
        checkpoint="not used (manual mask)",
    )


def run_sam3_on_pil(
    image: Image.Image,
    prompt: str,
    threshold: float,
    checkpoint_value: str,
    device: str,
    allow_huggingface: bool = True,
    mask_dilation: int = 0,
    mask_hull: bool = False,
    mask_outline_px: int = 0,
    exclude_prompt: str = "",
    user_mask: np.ndarray | None = None,
) -> Sam3Result:
    # Target empty + no Exclude means the user requested their drawn mask.
    # Branch before Torch import/device checks/checkpoint download or load.
    if user_mask is not None and not (prompt or "").strip() and not (exclude_prompt or "").strip():
        return _manual_mask_result(image, user_mask)

    import torch

    resolved_device = _resolve_device(device)
    checkpoint_path = resolve_checkpoint_path(checkpoint_value, allow_huggingface=allow_huggingface)
    if checkpoint_path is not None and checkpoint_path.suffix.lower() not in SUPPORTED_CHECKPOINT_SUFFIXES:
        raise RuntimeError(
            f"Unsupported SAM3 checkpoint: {checkpoint_path.name}. "
            f"Expected one of: {', '.join(SUPPORTED_CHECKPOINT_SUFFIXES)}."
        )
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(_missing_checkpoint_message(checkpoint_path, allow_huggingface))

    checkpoint_key = "__hf__" if checkpoint_path is None else str(checkpoint_path.resolve())
    _, processor = _get_model_bundle(checkpoint_key, resolved_device)
    processor.set_confidence_threshold(float(threshold))

    pil_image = image.convert("RGB")
    rgb = np.asarray(pil_image)
    h, w = rgb.shape[:2]

    groups = _split_prompt_groups(prompt) or [[prompt or ""]]

    group_masks: list[np.ndarray] = []
    all_boxes: list[np.ndarray] = []
    all_scores: list[float] = []
    # Outline(edge-aware) 의 두꺼운 Canny 맵 — 이 run 의 rgb 로 그룹 공통이라 한 번만 만든다(효율 감사 [8]).
    # 지역 변수라 run 이 끝나면 버려진다(다른 이미지·설정과 섞일 일 없음).
    outline_edges: np.ndarray | None = None

    # 이미지 백본(ViT)은 이미지당 한 번만 — 예전엔 검출·제외 토큰마다 set_image 를 다시 돌렸다(기존 M2).
    # set_text_prompt 는 state["backbone_out"].update(텍스트 특징) 로 덮어쓰고 geometric_prompt·결과 키를
    # state 에 넣으므로, 토큰마다 state 와 backbone_out 을 얕은 복사해 set_image 직후 모양 그대로 넘긴다.
    image_state: dict | None = None

    def _run_prompt(sub_prompt: str):
        nonlocal image_state
        if resolved_device == "cuda":
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            autocast = nullcontext()
        with _sam3_tf32(), autocast:
            if image_state is None:
                image_state = processor.set_image(pil_image)
            state = dict(image_state)
            state["backbone_out"] = dict(image_state["backbone_out"])
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
        # Edge-aware expansion BEFORE plain dilation: snap to real object
        # outline first, then apply user-requested px padding on top.
        if mask_outline_px > 0:
            if outline_edges is None:
                outline_edges = _edge_aware_edges(rgb)
            group_mask = _edge_aware_dilate(group_mask, rgb, mask_outline_px, edges=outline_edges)
        group_mask = _dilate_mask(group_mask, mask_dilation)
        if np.any(group_mask):
            group_masks.append(group_mask)

    # Intersection narrowing with the user's drawn mask. Runs AFTER text
    # detect+hull+dilation so SAM3 is allowed to expand to the real object
    # boundary first, then we clip down to the user's hand-drawn ROI.
    # Rule (per user spec):
    #   M_intersect = M_user & M_sam3_combined
    #   target = M_intersect if M_intersect.any() else M_user
    # → SAM3 snaps the rough scribble to the actual object edge, but if
    #   SAM3 missed the area entirely (e.g. text prompt didn't match,
    #   threshold too high) the user's scribble is preserved as-is, so a
    #   manual mask never silently collapses to nothing.
    # The narrowed result becomes a single group mask, so downstream
    # exclude / individual-vs-combined logic still works unchanged.
    if user_mask is not None:
        if user_mask.shape != (h, w):
            um_pil = Image.fromarray(user_mask.astype(np.uint8) * 255, mode="L").resize(
                (w, h), Image.NEAREST
            )
            user_mask = np.asarray(um_pil) > 127
        sam3_combined = (
            np.any(np.stack(group_masks, axis=0), axis=0)
            if group_masks
            else np.zeros((h, w), dtype=bool)
        )
        intersection = user_mask & sam3_combined
        if bool(intersection.any()):
            group_masks = [intersection]
            print(
                f"[-] SAM3: user mask ({int(user_mask.sum())}px) ∩ SAM3 "
                f"({int(sam3_combined.sum())}px) → narrowed to {int(intersection.sum())}px",
                file=sys.stderr,
            )
        else:
            group_masks = [user_mask]
            print(
                f"[-] SAM3: user mask ({int(user_mask.sum())}px) did not "
                f"overlap SAM3 detect — using user mask as-is",
                file=sys.stderr,
            )

    # Exclude prompt: detect regions to PROTECT, subtract from every group
    # mask. Use case: detect="clothes" expands via outline/hull and starts
    # eating into face/eyes; exclude="face, eyes" detects those and
    # subtracts → mask covers clothes only, not face. No dilation/hull on
    # the exclude side — we want a tight mask of what to KEEP.
    exclude_text = (exclude_prompt or "").strip()
    if exclude_text and group_masks:
        exclude_groups = _split_prompt_groups(exclude_text)
        exclude_pieces: list[np.ndarray] = []
        for ex_tokens in exclude_groups:
            for sub_prompt in ex_tokens:
                split, _, _ = _run_prompt(sub_prompt)
                exclude_pieces.extend(split)
        if exclude_pieces:
            exclude_combined = np.any(np.stack(exclude_pieces, axis=0), axis=0)
            print(
                f"[-] SAM3: exclude prompt {exclude_text!r} detected "
                f"{int(exclude_combined.sum())} px to protect",
                file=sys.stderr,
            )
            group_masks = [(m & ~exclude_combined) for m in group_masks]
            group_masks = [m for m in group_masks if np.any(m)]
        else:
            print(
                f"[-] SAM3: exclude prompt {exclude_text!r} matched nothing — "
                f"main mask unchanged",
                file=sys.stderr,
            )

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
    mask_count = len(result.masks)

    def _individual_paths(sfx: str) -> list[Path]:
        return [
            output_dir / f"{stem}_{slug}_mask_{idx:02d}{sfx}.png"
            for idx in range(1, mask_count + 1)
        ]

    suffix = ""
    mask_path = overlay_path = meta_path = None
    for index in range(1, 10001):
        suffix = "" if index == 1 else f"_{index}"
        mask_path = output_dir / f"{stem}_{slug}_mask{suffix}.png"
        overlay_path = output_dir / f"{stem}_{slug}_overlay{suffix}.png"
        meta_path = output_dir / f"{stem}_{slug}_prompt{suffix}.json"
        # Also check the per-mask filenames for this suffix: the combined
        # mask/overlay/meta could be absent (deleted) while individual masks
        # from a prior run still occupy the slot, which we'd otherwise clobber.
        if (
            not mask_path.exists()
            and not overlay_path.exists()
            and not meta_path.exists()
            and not any(p.exists() for p in _individual_paths(suffix))
        ):
            break
    else:
        # Hit the cap — fall back to a timestamp suffix so saves still succeed.
        import time as _time

        suffix = f"_{int(_time.time() * 1000)}"
        mask_path = output_dir / f"{stem}_{slug}_mask{suffix}.png"
        overlay_path = output_dir / f"{stem}_{slug}_overlay{suffix}.png"
        meta_path = output_dir / f"{stem}_{slug}_prompt{suffix}.json"

    result.mask.save(mask_path)
    # overlay 는 1536×2048 RGB 사진이라 zlib 기본 레벨(6)로는 postprocess 안에서 약 0.28 초(CPU 실측) —
    # 레벨 2 는 약 0.11 초에 파일은 +9%(레벨 1 은 0.095 초지만 +54%). PNG 는 무손실이라 픽셀은 같다(감사 supp-06).
    result.overlay.save(overlay_path, compress_level=2)
    for single_mask_path, mask in zip(_individual_paths(suffix), result.masks):
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
