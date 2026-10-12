"""SAM3 '얼굴 안에서만 찾기'(Face crop)의 순수 기하 — 얼굴 상자 → 정사각 crop, crop 마스크 → 전체 이미지 좌표.

numpy 와 ``face_detect`` 의 기본값만 쓴다(torch·ultralytics·Forge 없음). 흐름은 ``sam3ext.core.run_sam3_on_pil`` 이 짠다:

1. 얼굴 검출기(``sam3ext.face_detect``)가 준 상자를 ``face_crop_box`` 로 넓히고(옆·아래 20 %, 위 40 % — 이마·머리카락),
   중심을 지킨 정사각형으로 만들어 이미지 안으로 민다(한 변이 이미지보다 길면 그 축만 이미지 크기로 자른다).
2. SAM3 를 crop 마다 그대로 돌린다 — crop 은 자르기만 하고 크기를 바꾸지 않으므로(SAM3 처리기가 전체 이미지 때처럼
   작업 크기 1008×1008 로 늘려 보고, 마스크·상자는 crop 크기로 돌려준다) crop 좌표 → 이미지 좌표는 평행 이동뿐이다
   (``box_to_image`` / ``box_to_crop``).
3. crop 마스크를 ``paste_mask`` 로 이미지 크기 bool 마스크에 붙인다(비트 단위로 같은 자리).
4. crop 이 겹치면 한 얼굴의 crop 에 옆 얼굴의 눈·머리카락이 들어온다 — 검출 조각마다 ``assign_owner`` 로 주인 얼굴을
   정해(조각 무게중심에서 가장 가까운 얼굴, 얼굴 크기로 나눈 거리) 그 얼굴의 마스크에만 더한다. crop 의 안쪽 가장자리
   (이미지 테두리가 아닌 변)에 잘린 조각은 그 너머에서 이어지는 다른 crop 의 조각과 한 물체로 묶어 묶음 전체의 무게중심으로
   정한다(``assign_owners``) — 옆 얼굴의 머리카락 끝이 작은 얼굴의 crop 에 잘려 들어와도 그 얼굴의 패스에 섞이지 않는다.
   조각을 버리지는 않으므로 모든 얼굴을 합친(Combined) 마스크는 주인을 정하지 않은 합집합과 같다.
5. 한 얼굴에 검출기 상자가 둘 남으면(ultralytics NMS 는 IoU 0.7 미만이면 둘 다 둔다) 부위가 두 패스로 갈린다 — 계획에서
   ``same_face`` 인 상자는 점수가 높은 하나만 남긴다(``dedupe_faces``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Collection, Mapping, Sequence

import numpy as np

from .face_detect import DEFAULT_CONFIDENCE, DEFAULT_MODEL

# 얼굴 상자를 넓히는 비율(상자의 폭·높이 기준). face_yolov8 상자는 눈썹 위·턱 근처에서 끝나서, 이마·머리카락·귀가
# 들어오게 위는 더 넓힌다. (나중에 설정으로 뺄 수 있게 여기 한 곳에 둔다.)
EXPAND_SIDE = 0.20
EXPAND_TOP = 0.40
EXPAND_BOTTOM = 0.20
# 아주 작은 얼굴(멀리 있는 인물)도 SAM3 가 볼 주변이 조금은 있게 — crop 한 변의 최솟값(px, 이미지가 더 작으면 이미지 크기).
MIN_CROP_SIDE = 32
# 같은 얼굴로 볼 두 검출기 상자의 IoU 하한(또는 한 상자의 중심이 다른 상자 안). 서로 다른 두 얼굴의 상자가 이만큼
# 겹치는 일은 드물고, 겹치더라도 crop 이 거의 같아 부위를 나눌 근거가 없다.
SAME_FACE_IOU = 0.3

Box = Sequence[float]   # (x0, y0, x1, y1) — 이미지 픽셀 좌표, x1·y1 은 끝(포함하지 않음)


@dataclass(frozen=True)
class CropBox:
    """이미지 안의 정수 crop 영역 ``[x0, x1) × [y0, y1)`` — PIL ``Image.crop`` 의 상자와 같다."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1


def _finite_box(box: Box) -> tuple[float, float, float, float] | None:
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    return x0, y0, x1, y1


def clip_box(box: Box, image_size: tuple[int, int]) -> tuple[float, float, float, float] | None:
    """상자를 이미지(``image_size`` = (폭, 높이), PIL 순서) 안으로 자른다. 비거나 숫자가 아니면 None."""
    finite = _finite_box(box)
    if finite is None:
        return None
    width, height = image_size
    x0, y0, x1, y1 = finite
    x0, x1 = max(0.0, min(x0, x1)), min(float(width), max(x0, x1))
    y0, y1 = max(0.0, min(y0, y1)), min(float(height), max(y0, y1))
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return x0, y0, x1, y1


def expand_box(
    box: Box,
    *,
    side: float = EXPAND_SIDE,
    top: float = EXPAND_TOP,
    bottom: float = EXPAND_BOTTOM,
) -> tuple[float, float, float, float]:
    """얼굴 상자를 넓힌다: 왼쪽·오른쪽은 각각 폭의 ``side``, 위는 높이의 ``top``, 아래는 높이의 ``bottom``."""
    x0, y0, x1, y1 = (float(v) for v in box)
    w, h = x1 - x0, y1 - y0
    return (x0 - w * side, y0 - h * top, x1 + w * side, y1 + h * bottom)


def _fit_axis(start: int, length: int, limit: int) -> tuple[int, int]:
    """길이 ``length`` 구간을 ``[0, limit)`` 안으로 민다(길이는 그대로). 이미지보다 길면 이미지 전체."""
    if length >= limit:
        return 0, limit
    start = min(max(start, 0), limit - length)
    return start, start + length


def square_crop(
    box: Box,
    image_size: tuple[int, int],
    *,
    min_side: int = MIN_CROP_SIDE,
) -> CropBox | None:
    """``box`` 를 감싸는 정사각 정수 crop(중심 유지)을 이미지 안으로 민다.

    한 변 = ceil(max(폭, 높이, ``min_side``)). 이미지 밖으로 나가면 잘라 내지 않고 안쪽으로 밀어 정사각형과 내용을 지킨다.
    한 변이 이미지의 폭(또는 높이)보다 길 때만 그 축이 이미지 크기로 줄어 정사각형이 아니게 된다.
    """
    finite = _finite_box(box)
    if finite is None:
        return None
    width, height = (int(v) for v in image_size)
    if width <= 0 or height <= 0:
        return None
    x0, y0, x1, y1 = finite
    side = int(math.ceil(max(x1 - x0, y1 - y0, float(min_side), 1.0) - 1e-9))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left = int(math.floor(cx - side / 2.0 + 1e-9))
    top = int(math.floor(cy - side / 2.0 + 1e-9))
    left, right = _fit_axis(left, side, width)
    top, bottom = _fit_axis(top, side, height)
    return CropBox(left, top, right, bottom)


def face_crop_box(box: Box, image_size: tuple[int, int]) -> CropBox | None:
    """검출기 상자 하나 → SAM3 가 볼 crop: 이미지 안으로 자르고 → 넓히고 → 정사각형 → 이미지 안으로. 쓸 수 없으면 None."""
    clipped = clip_box(box, image_size)
    if clipped is None:
        return None
    return square_crop(expand_box(clipped), image_size)


def grow_crop(crop: CropBox, margin: int, image_size: tuple[int, int]) -> CropBox:
    """crop 을 사방 ``margin`` px 넓힌 정사각 영역(중심 유지, 이미지 안으로 밂) — Exclude 를 찾을 곳.

    Outline · Dilation 은 마스크를 crop 밖으로 최대 ``margin`` px(체스판 거리) 번지게 한다. 이 영역은 그 범위
    (crop ± ``margin`` 을 이미지로 자른 것)를 늘 덮는다 — 이미지 끝에서 밀려도 반대쪽으로 그만큼 더 덮는다.
    ``margin`` 이 0 이하면 crop 그대로.
    """
    margin = int(margin)
    if margin <= 0:
        return crop
    grown = square_crop(
        (crop.x0 - margin, crop.y0 - margin, crop.x1 + margin, crop.y1 + margin), image_size, min_side=1
    )
    return grown if grown is not None else crop


def box_to_image(box: Box, crop: CropBox) -> list[float]:
    """crop 좌표의 상자(SAM3 출력) → 이미지 좌표. crop 은 크기를 바꾸지 않으므로 평행 이동."""
    x0, y0, x1, y1 = (float(v) for v in box)
    return [x0 + crop.x0, y0 + crop.y0, x1 + crop.x0, y1 + crop.y0]


def box_to_crop(box: Box, crop: CropBox) -> list[float]:
    """이미지 좌표의 상자 → crop 좌표(``box_to_image`` 의 역)."""
    x0, y0, x1, y1 = (float(v) for v in box)
    return [x0 - crop.x0, y0 - crop.y0, x1 - crop.x0, y1 - crop.y0]


def paste_mask(crop_mask: np.ndarray, crop: CropBox, image_shape: tuple[int, int]) -> np.ndarray:
    """crop 크기 마스크를 이미지 크기(``image_shape`` = (높이, 폭), numpy 순서) bool 마스크의 crop 자리에 붙인다.

    crop 밖은 False. 마스크 크기가 crop 과 다르면(다른 처리기) 최근접 이웃으로 crop 크기에 맞춘 뒤 붙인다.
    """
    height, width = (int(v) for v in image_shape)
    if not (0 <= crop.x0 < crop.x1 <= width and 0 <= crop.y0 < crop.y1 <= height):
        raise ValueError(f"face crop {crop.as_tuple()} is outside the image {width}x{height}")
    piece = np.asarray(crop_mask).astype(bool)
    if piece.ndim != 2:
        raise ValueError(f"face crop mask must be 2-D, got shape {piece.shape}")
    if piece.shape != (crop.height, crop.width):
        from PIL import Image

        resized = Image.fromarray(piece.astype(np.uint8) * 255, mode="L").resize(
            (crop.width, crop.height), Image.NEAREST
        )
        piece = np.asarray(resized) > 127
    full = np.zeros((height, width), dtype=bool)
    full[crop.y0:crop.y1, crop.x0:crop.x1] = piece
    return full


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """마스크 픽셀의 무게중심 (x, y) — 픽셀 중심 기준(+0.5). 빈 마스크면 None."""
    ys, xs = np.nonzero(np.asarray(mask))
    if ys.size == 0:
        return None
    return float(xs.mean()) + 0.5, float(ys.mean()) + 0.5


def assign_owner(
    point: tuple[float, float],
    face_boxes: Sequence[Box],
    crops: Sequence[CropBox],
    source: int | Collection[int],
) -> int:
    """crop ``source`` 에서 찾은 조각(무게중심 ``point``)의 주인 얼굴 번호.

    후보 = 무게중심을 crop 안에 둔 얼굴들(+ ``source`` — 조각은 그 crop 안에 있으므로 늘 후보). 그중 얼굴 상자 중심까지의
    거리를 얼굴 크기(상자의 긴 변)로 나눈 값이 가장 작은 얼굴. 같으면 ``source``, 그다음 번호가 작은 얼굴.
    ``source`` 는 여러 crop 에 걸친 조각 묶음(``assign_owners``)이면 그 crop 번호들이다.
    """
    sources = {int(source)} if isinstance(source, (int, np.integer)) else {int(s) for s in source}
    px, py = point
    candidates = sources | {i for i, crop in enumerate(crops) if crop.contains(px, py)}

    def score(index: int) -> tuple[float, bool, int]:
        x0, y0, x1, y1 = (float(v) for v in face_boxes[index])
        size = max(x1 - x0, y1 - y0, 1.0)
        dist = math.hypot(px - (x0 + x1) / 2.0, py - (y0 + y1) / 2.0) / size
        return (dist, index not in sources, index)

    return min(sorted(candidates), key=score)


def _spread(line: np.ndarray) -> np.ndarray:
    """가장자리를 따라 한 칸씩 넓힌다(8-연결 — 대각선으로 이어지는 픽셀도 닿은 것으로)."""
    out = line.copy()
    out[1:] |= line[:-1]
    out[:-1] |= line[1:]
    return out


def edge_contact(piece: np.ndarray, crop: CropBox) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """crop 의 안쪽 가장자리에 닿은 조각이 그 너머에서 이어질 자리 — crop 바로 바깥 한 줄의 픽셀.

    ``piece`` 는 이미지 좌표의 bool 마스크(``paste_mask`` 결과). 안쪽 가장자리 = 이미지 테두리가 아닌 crop 의 변 — 거기
    닿은 조각은 crop 이 잘랐을 수 있다. 돌려주는 값은 ((y0, y1, x0, x1), 그 범위의 bool 마스크)이고, 안쪽 가장자리에
    닿지 않았으면 None.
    """
    height, width = piece.shape
    lines: list[tuple[int, int, int, int, np.ndarray]] = []   # (y0, y1, x0, x1, 바깥 한 줄)
    if crop.x0 > 0:
        line = piece[crop.y0:crop.y1, crop.x0]
        if line.any():
            lines.append((crop.y0, crop.y1, crop.x0 - 1, crop.x0, _spread(line)[:, None]))
    if crop.x1 < width:
        line = piece[crop.y0:crop.y1, crop.x1 - 1]
        if line.any():
            lines.append((crop.y0, crop.y1, crop.x1, crop.x1 + 1, _spread(line)[:, None]))
    if crop.y0 > 0:
        line = piece[crop.y0, crop.x0:crop.x1]
        if line.any():
            lines.append((crop.y0 - 1, crop.y0, crop.x0, crop.x1, _spread(line)[None, :]))
    if crop.y1 < height:
        line = piece[crop.y1 - 1, crop.x0:crop.x1]
        if line.any():
            lines.append((crop.y1, crop.y1 + 1, crop.x0, crop.x1, _spread(line)[None, :]))
    if not lines:
        return None
    y0 = min(item[0] for item in lines)
    y1 = max(item[1] for item in lines)
    x0 = min(item[2] for item in lines)
    x1 = max(item[3] for item in lines)
    contact = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for ly0, ly1, lx0, lx1, values in lines:
        contact[ly0 - y0:ly1 - y0, lx0 - x0:lx1 - x0] |= values
    return (y0, y1, x0, x1), contact


def assign_owners(
    pieces: Sequence[np.ndarray],
    sources: Sequence[int],
    face_boxes: Sequence[Box],
    crops: Sequence[CropBox],
) -> list[int]:
    """같은 검출 토큰으로 찾은 조각들(이미지 좌표, ``sources[i]`` = 찾은 crop)의 주인 얼굴 번호.

    crop 의 안쪽 가장자리에 잘린 조각은 그 너머 한 줄(``edge_contact``)에 걸친 다른 crop 의 조각과 한 물체로 묶는다 —
    한 crop 에서 잘려 보인 물체의 나머지는 이웃 crop 이 본다. 묶음은 합친 마스크의 무게중심으로, 혼자인 조각은 제 무게중심으로
    ``assign_owner`` 에 묻는다. 같은 crop 의 조각끼리는 묶지 않는다(SAM3 가 따로 낸 검출).
    """
    count = len(pieces)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(count):
        contact = edge_contact(pieces[i], crops[sources[i]])
        if contact is None:
            continue
        (y0, y1, x0, x1), where = contact
        for j in range(count):
            if sources[j] == sources[i] or find(i) == find(j):
                continue
            if np.any(pieces[j][y0:y1, x0:x1] & where):
                parent[find(j)] = find(i)

    owners = [0] * count
    clusters: dict[int, list[int]] = {}
    for index in range(count):
        clusters.setdefault(find(index), []).append(index)
    for members in clusters.values():
        if len(members) == 1:
            joined = pieces[members[0]]
        else:
            joined = np.any(np.stack([pieces[i] for i in members], axis=0), axis=0)
        point = mask_centroid(joined)
        if point is None:
            continue
        owner = assign_owner(point, face_boxes, crops, [sources[i] for i in members])
        for index in members:
            owners[index] = owner
    return owners


def order_faces(boxes: Sequence[Box]) -> list[int]:
    """얼굴 번호를 왼쪽 → 오른쪽(같으면 위 → 아래) 순서로 — Individual 패스 순서가 검출기 점수에 따라 바뀌지 않게."""
    def key(index: int) -> tuple[float, float, int]:
        x0, y0, x1, y1 = (float(v) for v in boxes[index])
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, index)

    return sorted(range(len(boxes)), key=key)


@dataclass
class FaceCropInfo:
    """한 이미지의 얼굴 crop 계획과 결과 요약 — ``Sam3Result.face_crop`` 에 실려 상태 줄·infotext·아티팩트 meta 가 된다.

    ``boxes``·``scores``·``crops`` 는 같은 순서(왼쪽 → 오른쪽)이고, 이미지 밖·빈 상자는 이미 뺐다.
    """

    model: str
    confidence: float
    boxes: list[list[float]]
    scores: list[float]
    crops: list[CropBox]

    @property
    def count(self) -> int:
        return len(self.crops)

    def status(self) -> str:
        """콘솔 한 줄(앞의 '[-] SAM3: ' 는 부르는 쪽이 붙인다)."""
        where = f"{self.model} (confidence {float(self.confidence):.2f})"
        if not self.crops:
            return f"face crop — no faces found by {where}; nothing to mask."
        sizes = ", ".join(f"{c.width}x{c.height}@{c.x0},{c.y0}" for c in self.crops)
        return f"face crop — {self.count} face(s) by {where}; SAM3 runs on each crop: {sizes}"

    def as_meta(self) -> dict:
        return {
            "model": self.model,
            "confidence": float(self.confidence),
            "faces": [
                {"box": [float(v) for v in box], "score": float(score), "crop": list(crop.as_tuple())}
                for box, score, crop in zip(self.boxes, self.scores, self.crops)
            ],
        }


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


def face_crop_kwargs(args: Mapping[str, Any]) -> dict[str, Any]:
    """SAM3 인자 dict(``Sam3Args`` 키) → ``run_sam3_on_pil`` 의 얼굴 crop 키워드.

    꺼져 있으면 ``{}`` — 예전과 똑같은 호출이 된다(In-flight·🎯·Refine 이 같이 쓴다).
    """
    if not _truthy(args.get("sam3_face_crop", False)):
        return {}
    try:
        confidence = float(args.get("sam3_face_crop_confidence", DEFAULT_CONFIDENCE))
    except (TypeError, ValueError):
        confidence = DEFAULT_CONFIDENCE
    if not math.isfinite(confidence):
        confidence = DEFAULT_CONFIDENCE
    return {
        "face_crop": True,
        "face_crop_model": str(args.get("sam3_face_crop_model") or "").strip() or DEFAULT_MODEL,
        "face_crop_confidence": min(1.0, max(0.0, confidence)),
    }


def box_iou(a: Box, b: Box) -> float:
    """두 상자 (x0, y0, x1, y1) 의 IoU(겹친 넓이 / 합친 넓이). 넓이가 없으면 0."""
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _centre_inside(a: Box, b: Box) -> bool:
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    cx, cy = (ax0 + ax1) / 2.0, (ay0 + ay1) / 2.0
    return bx0 <= cx < bx1 and by0 <= cy < by1


def same_face(a: Box, b: Box) -> bool:
    """두 검출기 상자가 한 얼굴인가 — IoU 가 ``SAME_FACE_IOU`` 이상이거나 한 상자의 중심이 다른 상자 안."""
    return box_iou(a, b) >= SAME_FACE_IOU or _centre_inside(a, b) or _centre_inside(b, a)


def dedupe_faces(boxes: Sequence[Box], scores: Sequence[float]) -> list[int]:
    """한 얼굴에 겹친 상자는 점수가 높은 하나만 — 남길 상자 번호(원래 순서).

    점수 높은 순서(같거나 숫자가 아니면 앞 번호 먼저)로 보며, 이미 남긴 상자와 ``same_face`` 인 상자는 버린다.
    """
    def rank(index: int) -> tuple[bool, float, int]:
        score = float(scores[index]) if index < len(scores) else float("nan")
        finite = math.isfinite(score)
        return (not finite, -score if finite else 0.0, index)

    kept: list[int] = []
    for index in sorted(range(len(boxes)), key=rank):
        if not any(same_face(boxes[index], boxes[other]) for other in kept):
            kept.append(index)
    return sorted(kept)


def plan_face_crops(
    boxes: Sequence[Box],
    scores: Sequence[float],
    image_size: tuple[int, int],
    *,
    model: str,
    confidence: float,
) -> FaceCropInfo:
    """검출기 상자들 → 쓸 수 있는 얼굴만, 왼쪽 → 오른쪽 순서의 crop 계획(``image_size`` = (폭, 높이)).

    한 얼굴에 겹친 상자(``same_face``)는 점수가 높은 하나만 남긴다 — 둘 다 두면 그 얼굴의 부위가 두 패스로 갈린다.
    """
    kept_boxes: list[list[float]] = []
    kept_scores: list[float] = []
    for index, box in enumerate(boxes):
        clipped = clip_box(box, image_size)
        if clipped is None:
            continue
        kept_boxes.append(list(clipped))
        kept_scores.append(float(scores[index]) if index < len(scores) else float("nan"))
    unique = dedupe_faces(kept_boxes, kept_scores)
    kept_boxes = [kept_boxes[i] for i in unique]
    kept_scores = [kept_scores[i] for i in unique]
    order = order_faces(kept_boxes)
    ordered_boxes = [kept_boxes[i] for i in order]
    ordered_scores = [kept_scores[i] for i in order]
    crops = [square_crop(expand_box(box), image_size) for box in ordered_boxes]
    keep = [i for i, crop in enumerate(crops) if crop is not None]
    return FaceCropInfo(
        model=str(model),
        confidence=float(confidence),
        boxes=[ordered_boxes[i] for i in keep],
        scores=[ordered_scores[i] for i in keep],
        crops=[crops[i] for i in keep],
    )
