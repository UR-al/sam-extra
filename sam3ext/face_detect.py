"""SAM3 '얼굴 안에서만 찾기'의 얼굴 검출 — ADetailer 가 쓰는 ``models/adetailer`` 의 ultralytics YOLO 상자 모델.

* 모델 폴더: Forge ``models/adetailer``(ADetailer 가 모델을 두는 곳) + ADetailer 설정 ``ad_extra_models_dir``(``|`` 로
  여러 폴더)가 있으면 그 폴더들. 아무것도 받지 않는다 — 파일이 없으면 넣을 곳을 알려 주는 오류로 멈춘다(ADetailer 처럼
  Hugging Face 에서 받지 않음).
* 불러오기: ADetailer 와 같은 ``ultralytics.YOLO(경로)`` → ``model(image, conf=…)`` 의 ``boxes.xyxy`` · ``boxes.conf``.
  ultralytics 는 ADetailer 가 Forge venv 에 깔아 둔 것을 쓴다(없으면 오류 메시지로 알림). 장치는 ADetailer 처럼 비워 두어
  (``""``) ultralytics 가 지금 CUDA 장치를 고르게 하고, SAM3 장치가 cpu 면 ``"cpu"`` — 장치 번호를 주면 ultralytics 가
  ``torch.cuda.set_device`` 로 프로세스 기본 장치를 바꾸기 때문이다.

이 모듈은 불러올 때 torch·ultralytics 를 import 하지 않는다(테스트는 ``detect_faces`` 를 가짜로 바꾼다).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "face_yolov8n.pt"
DEFAULT_CONFIDENCE = 0.3
MODEL_SUFFIX = ".pt"
ADETAILER_FOLDER = "adetailer"
# ADetailer 설정(Settings → ADetailer → Extra paths to scan adetailer models) — 있으면 같은 폴더도 본다.
OPT_EXTRA_MODEL_DIRS = "ad_extra_models_dir"


class FaceDetectorUnavailable(RuntimeError):
    """얼굴 검출기를 쓸 수 없음(모델 파일 없음, ultralytics 없음, 받아야 하는 이름) — 메시지에 할 일을 적는다."""


@dataclass
class FaceDetection:
    """검출기 출력: 이미지 좌표 상자(x0, y0, x1, y1)와 신뢰도, 실제로 읽은 모델 파일."""

    boxes: list[list[float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    model_path: str = ""


def _models_root() -> Path | None:
    try:
        from modules import paths  # type: ignore

        return Path(paths.models_path)
    except Exception:
        return None


def _extra_model_dirs() -> list[Path]:
    try:
        from modules import shared  # type: ignore

        raw = shared.opts.data.get(OPT_EXTRA_MODEL_DIRS, "") or ""
    except Exception:
        return []
    return [Path(part.strip()) for part in str(raw).split("|") if part.strip()]


def model_dirs() -> list[Path]:
    """얼굴 모델을 찾는 폴더들(있는 것만, 순서대로): ``models/adetailer``, ADetailer 의 추가 폴더."""
    dirs: list[Path] = []
    root = _models_root()
    if root is not None:
        dirs.append(root / ADETAILER_FOLDER)
    dirs.extend(_extra_model_dirs())
    seen: set[str] = set()
    out: list[Path] = []
    for folder in dirs:
        key = str(folder).lower()
        if key in seen or not folder.is_dir():
            continue
        seen.add(key)
        out.append(folder)
    return out


def _is_box_model(path: Path) -> bool:
    # YOLO-World(``*-world*``)는 클래스 이름을 따로 줘야 한다 — 상자만 쓰는 이 기능에는 맞지 않아 뺀다.
    return path.is_file() and path.suffix.lower() == MODEL_SUFFIX and "-world" not in path.stem.lower()


def _scan(folder: Path) -> list[Path]:
    try:
        return sorted((p for p in folder.rglob(f"*{MODEL_SUFFIX}") if _is_box_model(p)), key=lambda p: p.name.lower())
    except OSError:
        return []


def find_face_models() -> list[str]:
    """드롭다운 목록(파일 이름): 기본 모델 → 나머지 ``face_*`` → 그 밖의 YOLO 상자 모델(hand_yolov8n 등). 없으면 기본 이름만."""
    names: list[str] = []
    for folder in model_dirs():
        for path in _scan(folder):
            if path.name not in names:
                names.append(path.name)

    def key(name: str) -> tuple[int, str]:
        lowered = name.lower()
        if lowered == DEFAULT_MODEL:
            return (0, lowered)
        return (1 if lowered.startswith("face") else 2, lowered)

    names.sort(key=key)
    return names or [DEFAULT_MODEL]


def resolve_face_model(name: str) -> Path | None:
    """모델 이름(또는 절대 경로) → 있는 파일 경로. 못 찾으면 None — 받지 않는다."""
    value = str(name or "").strip() or DEFAULT_MODEL
    path = Path(value)
    if path.is_absolute():
        return path if _is_box_model(path) else None
    for folder in model_dirs():
        direct = folder / value
        if _is_box_model(direct):
            return direct
        for candidate in _scan(folder):
            if candidate.name.lower() == path.name.lower():
                return candidate
    return None


def missing_model_message(name: str) -> str:
    where = " or ".join(str(d) for d in model_dirs())
    if not where:
        root = _models_root()
        where = str(root / ADETAILER_FOLDER) if root is not None else "<webui>/models/adetailer"
    return (
        f"face crop: face detector model {name!r} not found — put it in {where} "
        "(the same folder ADetailer uses; it is never downloaded automatically)."
    )


def _guard_no_download(path: Path) -> None:
    """ultralytics 는 'yolov5s.pt' 같은 이름을 'yolov5su.pt' 로 바꿔 없으면 GitHub 에서 받는다 — 그런 이름이면 멈춘다."""
    try:
        from ultralytics.utils.checks import check_yolov5u_filename  # type: ignore
    except Exception:
        return
    try:
        renamed = str(check_yolov5u_filename(str(path), verbose=False))
    except TypeError:   # verbose 인자가 없는 판
        renamed = str(check_yolov5u_filename(str(path)))
    if renamed != str(path) and not Path(renamed).exists():
        raise FaceDetectorUnavailable(
            f"face crop: ultralytics would rename {path.name!r} to {Path(renamed).name!r} and download it — "
            "rename the file (e.g. add 'u' as ultralytics expects) or pick another model."
        )


def ultralytics_device(sam3_device: str) -> str:
    """SAM3 장치 → ultralytics ``device`` 인자. cpu 면 "cpu", 아니면 "" (ADetailer 와 같음 — 지금 CUDA 장치)."""
    return "cpu" if str(sam3_device or "").strip().lower() == "cpu" else ""


def _to_list(value: Any) -> list:
    if value is None:
        return []
    for attr in ("detach", "cpu", "numpy"):
        if hasattr(value, attr):
            value = getattr(value, attr)()
    return list(value.tolist()) if hasattr(value, "tolist") else list(value)


def detect_faces(image, model_name: str, confidence: float, device: str) -> FaceDetection:
    """``image``(PIL RGB)에서 얼굴 상자를 찾는다. 모델 파일이 없거나 ultralytics 가 없으면 ``FaceDetectorUnavailable``."""
    path = resolve_face_model(model_name)
    if path is None:
        raise FaceDetectorUnavailable(missing_model_message(str(model_name or DEFAULT_MODEL)))
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise FaceDetectorUnavailable(
            "face crop: the 'ultralytics' package is not installed in the Forge venv — install ADetailer "
            "(it brings ultralytics) or `pip install ultralytics` in the Forge venv."
        ) from exc
    _guard_no_download(path)
    model = YOLO(str(path))
    try:
        pred = model(image, conf=float(confidence), device=ultralytics_device(device), verbose=False)
    finally:
        del model
    if not pred:
        return FaceDetection(model_path=str(path))
    boxes = getattr(pred[0], "boxes", None)
    if boxes is None:
        return FaceDetection(model_path=str(path))
    xyxy = [[float(v) for v in row] for row in _to_list(boxes.xyxy)]
    scores = [float(v) for v in _to_list(boxes.conf)]
    if len(scores) != len(xyxy):
        scores = (scores + [float("nan")] * len(xyxy))[: len(xyxy)]
    return FaceDetection(boxes=xyxy, scores=scores, model_path=str(path))
