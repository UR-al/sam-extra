"""Anima VAE DeGrid — NAFNet 잔차(residual) 모델로 Qwen/Wan VAE 격자 무늬를 지우는 계산(순수 torch, Forge 없음).

모델: DraconicDragon/NAFNet-VAE-DeGrid (Apache-2.0) — NAFNet-small(width 32, enc 2·2·4·8, middle 12, dec 2·2·2·2)을
(PNG, 그 PNG 를 Qwen VAE 로 인코드·디코드한 것) 쌍으로 학습했다. 학습 코드(DraconicDragon/NAFNet-c
``basicsr/models/qdg_model.py`` 48-49·78-79줄)가 ``output = lq + net_g(lq)`` 라서 ``net_g`` 의 출력 — spandrel NAFNet
forward 그대로, 안쪽의 ``x + inp`` 까지 포함 — 이 곧 **잔차**다. 이미지가 아니므로 입력에 더해야 한다.

적용 모드 — ComfyUI-NAFNet-Residual(Apache-2.0, commit e15460d) ``nafnet_node.py`` 132-149줄(= ``patch.py`` 48-62줄):

    Full                  result = image + delta
    Dark Pixels Mainly    result = image + delta.clamp_min(0)   양의 잔차 — 어두운 점을 밝힌다
    Bright Pixels Mainly  result = image + delta.clamp_max(0)   음의 잔차 — 밝은 점을 어둡게 한다
    그 뒤 torch.clamp(result, 0, 1) 한 번(145-149줄).

강도(strength)는 이 확장이 더한 것으로, 고른 잔차에 곱한 뒤 더한다(``image + s·f(delta)``). 강도 1 이면 노드 팩과
연산이 같다. [0, 1] 로 자르는 것은 마지막 이미지 한 번뿐이다 — 잔차도, 중간 합도 자르지 않는다.
ComfyUI 코어 경로(Load Upscale Model → Upscale Image → Image Blend)가 Dark Pixels Mainly 와 같은 이유는 spandrel
``ImageModelDescriptor.__call__`` 이 모델 출력을 ``clamp_(0, 1)`` 해 음의 잔차가 사라지기 때문이다. 그래서 여기서는
descriptor 를 부르지 않고 모델(forward)을 직접 부른다(노드 팩의 ``get_raw_nafnet_call`` 과 같은 이유).

타일: 노드 팩은 ComfyUI ``comfy.utils.tiled_scale``(tile 512, overlap 32)로 잔차를 만들고, 메모리가 모자라면(OOM) 타일을
반으로 줄여 128 까지 다시 한다(nafnet_node.py 87-128줄). ``tiled_residual`` 은 그 함수(Forge 는
``backend/patcher/vae.py`` 에 ComfyUI v0.3.64 판을 옮겨 둠)와 같은 타일 위치·feather 가중치로 붙인다(배율 1 전용).
NAFNet 의 채널 어텐션(SCA)은 타일 전체의 평균을 쓰므로 타일로 나누면 나누지 않은 결과와 조금 다르다 — 노드 팩과
같은 결과를 내려면 같은 타일 크기(512)를 쓴다.

패딩: 노드 팩은 타일을 모델에 그대로 넣어 spandrel 이 16 배수로 0 을 채우고, 그 때문에 16 배수가 아닌 크기에서는
오른쪽·아래 가장자리 잔차가 커진다. 여기서는 타일마다 반사 패딩으로 배수를 맞추고 자른다(``call_padded``). 16 배수
크기(표준 Anima 해상도는 타일 512·256·128 의 모든 조각이 16 배수)에서는 모델을 그대로 불러 노드 팩과 같다.
"""
from __future__ import annotations

import itertools
import math
from typing import Callable

import numpy as np
import torch
from PIL import Image

MODE_FULL = "full"
MODE_DARK = "dark"
MODE_BRIGHT = "bright"
MODES = (MODE_FULL, MODE_DARK, MODE_BRIGHT)

# 노드 팩의 이름 그대로 — infotext 값으로도 쓴다(언어와 무관하게 고정).
MODE_LABELS = {
    MODE_FULL: "Full",
    MODE_DARK: "Dark Pixels Mainly",
    MODE_BRIGHT: "Bright Pixels Mainly",
}

# 노드 팩 패치 설정값(patch.py 13-18줄)과 짧은 별칭도 받는다.
_MODE_ALIASES = {
    "full": MODE_FULL,
    "dark": MODE_DARK,
    "dark_pixels": MODE_DARK,
    "dark pixels": MODE_DARK,
    "bright": MODE_BRIGHT,
    "bright_pixels": MODE_BRIGHT,
    "bright pixels": MODE_BRIGHT,
}

DEFAULT_MODE = MODE_FULL
DEFAULT_STRENGTH = 1.0
STRENGTH_MIN = 0.0
STRENGTH_MAX = 1.5

DEFAULT_TILE = 512   # 노드 팩 nafnet_node.py 87줄
TILE_OVERLAP = 32    # 88줄
MIN_TILE = 128       # 127줄 — 이보다 작아지면 포기
MAX_TILE = 4096

# NAFNet-small(인코더 4단)이 안에서 맞추는 배수 — spandrel ``padder_size = 2 ** len(encoders)``. 모델에 값이 없을 때 쓴다.
PAD_MULTIPLE = 16

# 출력이 잔차가 아니라 이미지인지 — 실제 v1.1·다른 DeGrid 파인튜닝은 |평균| 0.1~0.7/255(최대 수십/255)이고 입력과의
# 상관이 |r| 0.2 안팎 이하다(CPU 실측, Anima 512² 조각 4장: -0.09~+0.19). 복원 이미지를 내는 NAFNet 은 |평균| 이 이미지
# 밝기(수십~백여/255)이고 상관이 1 에 가깝다.
IMAGE_LIKE_ABS_MEAN = 25 / 255       # 잔차가 평균 25 단계를 넘으면 이미지로 본다
IMAGE_LIKE_MIN_ABS_MEAN = 2 / 255    # 이 아래면 상관을 보지 않는다(거의 검은 이미지에서 헛판정 방지)
IMAGE_LIKE_CORRELATION = 0.9


def normalize_mode(value) -> str | None:
    """모드 키·노드 팩 이름·UI 라벨('Full (전체)' 처럼 괄호 설명이 붙은 것) → 모드 키. 모르면 None."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    if text in _MODE_ALIASES:
        return _MODE_ALIASES[text]
    for key, label in MODE_LABELS.items():
        if text == label.lower():
            return key
    return None


def coerce_strength(value, default: float = DEFAULT_STRENGTH) -> float:
    """강도 → [0, 1.5] 안의 float. 읽을 수 없거나 NaN 이면 ``default``."""
    try:
        strength = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(strength):
        return float(default)
    return min(max(strength, STRENGTH_MIN), STRENGTH_MAX)


def coerce_tile(value, default: int = DEFAULT_TILE) -> int:
    """타일 크기 → 0(나누지 않음) 또는 [128, 4096](1~127 은 128). 읽을 수 없으면 ``default``."""
    try:
        tile = int(float(value))
    except (TypeError, ValueError):
        return int(default)
    if tile <= 0:
        return 0
    return min(max(tile, MIN_TILE), MAX_TILE)


# ---------------------------------------------------------------------------
# 잔차 적용
# ---------------------------------------------------------------------------


def select_residual(delta: torch.Tensor, mode: str) -> torch.Tensor:
    """모드가 쓰는 잔차 — Full 전체, Dark 양수만, Bright 음수만(노드 팩 nafnet_node.py 132-139줄)."""
    if mode == MODE_FULL:
        return delta
    if mode == MODE_DARK:
        return delta.clamp_min(0)
    if mode == MODE_BRIGHT:
        return delta.clamp_max(0)
    raise ValueError(f"Unknown DeGrid mode: {mode!r}")


def apply_residual(image: torch.Tensor, delta: torch.Tensor, mode: str, strength: float = 1.0) -> torch.Tensor:
    """``image + strength·f(delta)`` — 자르지 않는다(``finalize`` 가 마지막에 한 번)."""
    selected = select_residual(delta, mode)
    if strength != 1.0:
        selected = selected * float(strength)
    return image + selected


def finalize(result: torch.Tensor) -> torch.Tensor:
    """마지막 이미지만 [0, 1] 로 자른다(노드 팩 145-149줄)."""
    return torch.clamp(result, min=0.0, max=1.0)


def output_looks_like_image(image: torch.Tensor, raw: torch.Tensor) -> bool:
    """모델 출력(``raw``)이 잔차가 아니라 이미지인가 — 파일 키로는 DeGrid 와 일반 복원 NAFNet(SIDD width 32 노이즈 제거는
    구성까지 같다)을 가를 수 없어, 출력으로 거른다. |평균| 이 25/255 를 넘거나, 2/255 를 넘으면서 입력과의 상관이
    0.9 를 넘으면 이미지다. 분산이 0 이라 상관을 정할 수 없으면 잔차로 본다."""
    mean_abs = float(raw.abs().mean()) if raw.numel() else 0.0
    if mean_abs > IMAGE_LIKE_ABS_MEAN:
        return True
    if not mean_abs > IMAGE_LIKE_MIN_ABS_MEAN:
        return False
    # float32 로 충분하다(문턱 0.9) — hires 2432×3712 에서도 사본 두 개 약 0.2 GB.
    a = image.detach().reshape(-1).to(dtype=torch.float32)
    b = raw.detach().reshape(-1).to(device=a.device, dtype=torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(a.norm()) * float(b.norm())
    if not denominator > 0.0:
        return False
    return float(torch.dot(a, b)) / denominator > IMAGE_LIKE_CORRELATION


# ---------------------------------------------------------------------------
# 타일 (ComfyUI tiled_scale, 배율 1)
# ---------------------------------------------------------------------------


def tile_starts(size: int, tile: int, overlap: int) -> list[int]:
    """한 축의 타일 시작 위치 — ComfyUI 와 같다: ``range(0, size - overlap, tile - overlap)`` 를
    ``[0, size - overlap]`` 로 자른다. 크기가 타일 이하면 [0]."""
    if size <= tile:
        return [0]
    return [max(0, min(size - overlap, pos)) for pos in range(0, size - overlap, tile - overlap)]


def _feather_mask(height: int, width: int, overlap: int, dtype: torch.dtype, device) -> torch.Tensor:
    """타일 가장자리 overlap 칸을 (t+1)/overlap 로 낮추는 가중치(ComfyUI 와 같음 — 타일이 overlap 이하인 축은 그대로)."""
    mask = torch.ones((1, 1, height, width), dtype=dtype, device=device)
    for dim, length in ((2, height), (3, width)):
        feather = int(overlap)
        if feather <= 0 or feather >= length:
            continue
        for t in range(feather):
            a = (t + 1) / feather
            mask.narrow(dim, t, 1).mul_(a)
            mask.narrow(dim, length - 1 - t, 1).mul_(a)
    return mask


def tiled_residual(
    x: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    tile: int = DEFAULT_TILE,
    overlap: int = TILE_OVERLAP,
    out_device="cpu",
) -> torch.Tensor:
    """``fn``(같은 크기의 잔차를 돌려주는 함수)을 타일마다 불러 feather 가중 평균으로 붙인다. ``x``: [B,3,H,W].

    ``tile`` 이 0 이하이거나 이미지가 타일 안에 들어가면 한 번에 부른다. 결과는 ``out_device`` 의 float32.
    """
    batch, _channels, height, width = x.shape
    out = torch.empty((batch, x.shape[1], height, width), dtype=torch.float32, device=out_device)
    for b in range(batch):
        sample = x[b : b + 1]
        if tile <= 0 or (height <= tile and width <= tile):
            out[b : b + 1] = fn(sample).to(device=out_device, dtype=torch.float32)
            continue
        acc = torch.zeros((1, x.shape[1], height, width), dtype=torch.float32, device=out_device)
        weight = torch.zeros_like(acc)
        for top, left in itertools.product(tile_starts(height, tile, overlap), tile_starts(width, tile, overlap)):
            h = min(tile, height - top)
            w = min(tile, width - left)
            piece = fn(sample[:, :, top : top + h, left : left + w]).to(device=out_device, dtype=torch.float32)
            mask = _feather_mask(h, w, overlap, torch.float32, out_device).expand_as(piece)
            acc[:, :, top : top + h, left : left + w].add_(piece * mask)
            weight[:, :, top : top + h, left : left + w].add_(mask)
        out[b : b + 1] = acc / weight
    return out


def pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """[B,C,H,W] 의 오른쪽·아래를 ``multiple`` 배수로 반사 패딩한다(이미 배수면 ``x`` 그대로). 채울 칸이 그 축 길이
    이상이라 반사할 수 없는 아주 작은 입력은 가장자리를 복제한다."""
    multiple = int(multiple)
    height, width = int(x.shape[-2]), int(x.shape[-1])
    pad_h = (-height) % multiple if multiple > 1 else 0
    pad_w = (-width) % multiple if multiple > 1 else 0
    if not pad_h and not pad_w:
        return x
    mode = "reflect" if pad_h < height and pad_w < width else "replicate"
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode=mode)


def call_padded(fn: Callable[[torch.Tensor], torch.Tensor], piece: torch.Tensor, multiple: int) -> torch.Tensor:
    """``fn`` 을 ``multiple`` 배수로 반사 패딩한 입력으로 부르고 원래 크기로 자른다.

    spandrel NAFNet 은 16 배수가 아니면 안에서 **0** 으로 채운다(``check_image_size``). 이 모델은 ``ending(x) + inp`` 가
    잔차가 되도록 입력을 상쇄하게 학습돼 있어, 이미지 옆 0 띠가 강한 경계가 되고 오른쪽·아래 가장자리 잔차가 커진다
    (v1.1 CPU 실측: 250x190 오른쪽 아래 56/255, 반사 패딩 1.6/255 — 256x192 는 1.8/255). 저자의 단독 추론
    스크립트(NAFNet-c ``infer.py``)도 반사로 채운다. 배수인 입력(표준 Anima 크기의 모든 타일)은 그대로 부르므로 노드
    팩(ComfyUI-NAFNet-Residual)과 같다.
    """
    height, width = int(piece.shape[-2]), int(piece.shape[-1])
    padded = pad_to_multiple(piece, multiple)
    if padded is piece:
        return fn(piece)
    return fn(padded)[..., :height, :width]


def is_oom_error(exc: BaseException) -> bool:
    """OOM 판정 — torch 의 OOM 예외이거나 메시지에 'out of memory'."""
    oom_type = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return "out of memory" in str(exc).lower()


def tiled_residual_with_oom_retry(
    x: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    tile: int = DEFAULT_TILE,
    overlap: int = TILE_OVERLAP,
    out_device="cpu",
    is_oom: Callable[[BaseException], bool] = is_oom_error,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> tuple[torch.Tensor, int]:
    """OOM 이면 타일을 반으로 줄여 다시 한다(노드 팩 94-128줄). 128 보다 작아지면 마지막 예외를 그대로 올린다.

    ``tile`` 0(나누지 않음)이 OOM 이면 긴 변의 절반부터 줄인다. OOM 이 아닌 예외는 곧바로 올린다.
    ``(잔차, 실제로 쓴 타일 크기)`` 를 돌려준다.
    """
    height, width = int(x.shape[-2]), int(x.shape[-1])
    current = int(tile)
    while True:
        try:
            return tiled_residual(x, fn, tile=current, overlap=overlap, out_device=out_device), current
        except Exception as exc:
            if not is_oom(exc):
                raise
            base = current if current > 0 else max(height, width)
            current = base // 2
            if current < MIN_TILE:
                raise
            if on_retry is not None:
                on_retry(current, exc)


# ---------------------------------------------------------------------------
# PIL ↔ 텐서
# ---------------------------------------------------------------------------


def pil_to_tensor(image: Image.Image) -> tuple[torch.Tensor, Image.Image | None]:
    """PIL → ([1,3,H,W] float32 [0,1] RGB, 알파 채널 또는 None). RGBA·LA 의 알파는 그대로 되돌려 붙인다."""
    alpha = None
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")
        rgb = rgba.convert("RGB")
    elif image.mode != "RGB":
        rgb = image.convert("RGB")
    else:
        rgb = image
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))[None]
    return tensor, alpha


def tensor_to_pil(tensor: torch.Tensor, alpha: Image.Image | None = None) -> Image.Image:
    """[1,3,H,W] [0,1] → PIL RGB(알파가 있으면 RGBA). 8비트는 반올림한다 — 잔차 0 인 픽셀이 원본과 같은 값이 되게
    (ComfyUI SaveImage·Forge 저장은 버림(astype)이라 k/255·255 가 k 보다 조금 작으면 한 단계 내려간다)."""
    array = tensor.detach()[0].to(device="cpu", dtype=torch.float32).numpy().transpose(1, 2, 0)
    array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    out = Image.fromarray(array, "RGB")
    if alpha is not None:
        out.putalpha(alpha)
    return out


def residual_stats(delta: torch.Tensor, before: Image.Image | None = None, after: Image.Image | None = None) -> dict:
    """로그용 숫자 — 잔차 |평균|·최대(8비트 단계), 8비트 값이 바뀐 픽셀 비율."""
    stats = {
        "abs_mean_255": float(delta.abs().mean()) * 255.0,
        "abs_max_255": float(delta.abs().max()) * 255.0 if delta.numel() else 0.0,
    }
    if before is not None and after is not None:
        a = np.asarray(before.convert("RGB"), dtype=np.int16)
        b = np.asarray(after.convert("RGB"), dtype=np.int16)
        stats["changed_pixels"] = float(np.any(a != b, axis=-1).mean())
    return stats
