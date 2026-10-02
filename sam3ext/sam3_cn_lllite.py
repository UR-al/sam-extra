"""SAM3 ControlNet unit — Anima ControlNet-LLLite 판별과 전처리기(module) 가드 (순수 로직, torch·Forge 없음).

SAM3 인페인트 패스의 CN 유닛은 ``image=None`` 이라 Forge ControlNet 이 ``p2.init_images[0]`` 을 cond 로,
``p2.image_mask`` 를 mask 로 쓴다(``inpaint_core.inject_controlnet_unit``). 전처리기가 그 cond 를 바꾼다.

원본(kohya sd-scripts ``anima_minimal_inference_control_net_lllite.py``, kohya ComfyUI-Anima-LLLite
``AnimaLLLiteApply_sdscripts``)은 LLLite 에 **사용자가 준 제어 이미지**(canny·lineart·depth 맵, Tile & Repair 면
고칠 그림)를 그대로 준다:
  * 채널 수는 가중치 메타데이터 ``lllite.cond_in_channels`` (없으면 3) 로 정한다. 3 이 표준(모든 제어 종류),
    4 는 인페인트다(sd-scripts docs/anima_train_control_net_lllite.md 메타데이터 표).
  * 3채널은 마스크를 버린다(``cond_in_channels != 4`` → mask 무시). 4채널(인페인트)은 RGB + 마스크다.

SAM3 CN 유닛의 cond 는 언제나 인페인트 입력 이미지라, lineart·canny·depth LLLite 에는 전처리기가 곧 그 맵을
만드는 유일한 단계다 — 그래서 전처리기는 **필요한 경우에만** 바꾼다:
  * **Tile & Repair** (3채널, 고칠 그림 자체를 받는다) → 언제나 ``None``. 기본 ``inpaint_only`` 는 고칠 영역을
    ``cond*(1-mask) - mask`` 로 −1 로 비운다(LLLite 입력으로는 ``*2-1`` 뒤 −3) — 복구할 내용이 사라진다.
  * **그 밖의 Anima LLLite** (3채널 lineart·canny·depth 등, 4채널 인페인트) → ``inpaint_*`` 만 ``None``.
    3채널: ``inpaint_*`` 는 마스크를 제어 이미지에 구워 넣는다 — 원본 3채널은 마스크를 쓰지 않는다.
    4채널: ``inpaint_*`` 는 마스크를 버리고(None 반환) Forge 내장 LLLite forward 의
    ``assert isinstance(mask, torch.Tensor)`` 가 깨진다. 다른 전처리기(lineart_anime 등)는 고른 그대로 둔다.
  * Anima LLLite 가 아니면(SDXL controllllite 포함) 건드리지 않는다.

판별 순서: safetensors 헤더(JSON, torch 없이) → 없으면 파일 이름.
  * Anima LLLite 인가: 키가 ``lllite_dit`` 로 시작한다 — Forge 내장
    ``ControlLLLiteAnimaPatcher.try_build_from_state_dict`` 가 Anima 패처를 고르는 기준과 같다.
  * 채널: 메타데이터 ``lllite.cond_in_channels`` (원본) → ``lllite_conditioning1.conv1.weight`` 의
    입력 채널(Forge 내장 ``infer_anima_config``) → 3 (원본 기본값).
  * Tile & Repair 인가(3채널만): 메타데이터 ``modelspec.title`` 에 'tile' 이 들어 있다 — civitai 2708551
    v1.0 = ``anima_tiled_lllite_v1``, v2.0 = ``anima_tile_multitask_v1``. 제목이 없으면 파일 이름에 'tile'.

앱(UR_IV)의 거울: ``core/sam3_cn_names.py`` (``lllite_channels_from_name`` · ``lllite_tile_repair_from_name`` ·
``lllite_module_override``)와 ``frontend/src/utils/sam3ControlNet.ts`` — 이름 규칙은 세 곳이 같아야 한다.
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any, NamedTuple, Optional

NONE = "None"

# 원본 기본값: kohya-ss/ComfyUI-Anima-LLLite@b7495bd8:nodes.py:168,
# kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:325-329
META_COND_IN_CHANNELS = "lllite.cond_in_channels"
DEFAULT_COND_IN_CHANNELS = 3
# Forge extensions-builtin/sd_forge_controlllite/scripts/forge_controllllite.py
# (ControlLLLiteAnimaPatcher.try_build_from_state_dict: any(k.startswith("lllite_dit")))
ANIMA_KEY_PREFIX = "lllite_dit"
COND_CONV1_KEY = "lllite_conditioning1.conv1.weight"
# Tile & Repair (civitai 2708551) 헤더의 modelspec.title — v1.0 'anima_tiled_lllite_v1', v2.0 'anima_tile_multitask_v1'
META_TITLE = "modelspec.title"
TILE_REPAIR_MARK = "tile"

# safetensors 는 헤더를 100 MB 로 제한한다 — 그보다 크면 safetensors 파일이 아니다.
_MAX_HEADER_BYTES = 100 * 1024 * 1024


def read_safetensors_header(path: Any) -> Optional[dict]:
    """safetensors 파일의 JSON 헤더(텐서 모양·``__metadata__``). 읽을 수 없거나 safetensors 가 아니면 None.

    파일 앞 8바이트(리틀 엔디언 u64 = 헤더 길이)와 그 뒤 JSON 만 읽는다 — 가중치는 읽지 않는다.
    """
    if not path:
        return None
    try:
        file = Path(str(path))
        if file.suffix.lower() != ".safetensors" or not file.is_file():
            return None
        with file.open("rb") as handle:
            raw_len = handle.read(8)
            if len(raw_len) != 8:
                return None
            (length,) = struct.unpack("<Q", raw_len)
            if length <= 0 or length > _MAX_HEADER_BYTES:
                return None
            data = handle.read(length)
        if len(data) != length:
            return None
        header = json.loads(data.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return header if isinstance(header, dict) else None


def lllite_channels_from_header(header: Optional[dict]) -> Optional[int]:
    """헤더 → Anima LLLite 의 cond 입력 채널 수(3/4…). Anima LLLite 가 아니면 None."""
    if not isinstance(header, dict):
        return None
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not any(str(key).startswith(ANIMA_KEY_PREFIX) for key in tensors):
        return None
    meta = header.get("__metadata__")
    if isinstance(meta, dict) and meta.get(META_COND_IN_CHANNELS) is not None:
        try:
            return int(meta[META_COND_IN_CHANNELS])
        except (TypeError, ValueError):
            pass
    conv = tensors.get(COND_CONV1_KEY)
    shape = conv.get("shape") if isinstance(conv, dict) else None
    if isinstance(shape, list) and len(shape) >= 2:
        try:
            return int(shape[1])
        except (TypeError, ValueError):
            pass
    return DEFAULT_COND_IN_CHANNELS


def _flat(value: Any) -> str:
    """대소문자·구분자를 무시한 비교용 글자(``anima_tile-repair`` = ``animaTileRepair``)."""
    text = "" if value is None else str(value).strip().casefold()
    return re.sub(r"[^0-9a-z]", "", text)


def lllite_channels_from_name(name: Any) -> Optional[int]:
    """파일 이름 → 채널 추정(헤더를 못 읽을 때만). Anima LLLite 로 보이지 않으면 None.

    * 'anima' + 'lllite' + 'inpaint'             → 4  (예: anima-lllite-inpainting-v2)
    * 'anima' + ('lllite' 또는 'tilerepair')      → 3  (예: animaTileRepair_v20, anima_lllite_lineart_v1)
    'anima' 가 없으면 None — SDXL ``kohya_controllllite_xl_*`` 는 'controllllite' 에 'lllite' 가 들어 있어도
    Anima 패처가 아니다(헤더에 ``lllite_dit`` 키가 없다).
    """
    flat = _flat(name)
    if not flat or flat == NONE.casefold() or "anima" not in flat:
        return None
    if "lllite" in flat and "inpaint" in flat:
        return 4
    if "lllite" in flat or "tilerepair" in flat:
        return 3
    return None


def lllite_tile_repair_from_name(name: Any) -> bool:
    """이름으로 본 3채널 Anima LLLite 가 Tile & Repair 인가 — 이름에 'tile' (animaTileRepair_*, anima_tiled_lllite_*)."""
    return lllite_channels_from_name(name) == 3 and TILE_REPAIR_MARK in _flat(name)


def lllite_tile_repair_from_header(header: Optional[dict], name: Any = "") -> bool:
    """헤더로 본 3채널 Anima LLLite 가 Tile & Repair 인가 — ``modelspec.title`` 에 'tile', 제목이 없으면 이름."""
    if lllite_channels_from_header(header) != 3:
        return False
    meta = header.get("__metadata__")
    title = meta.get(META_TITLE) if isinstance(meta, dict) else None
    if title is not None and str(title).strip():
        return TILE_REPAIR_MARK in _flat(title)
    return TILE_REPAIR_MARK in _flat(name)


class AnimaLLLite(NamedTuple):
    """SAM3 CN 모델이 Anima LLLite 일 때 — cond 채널 수와 Tile & Repair 여부."""

    channels: int
    tile_repair: bool = False


def anima_lllite(model_name: Any, model_path: Any = None) -> Optional[AnimaLLLite]:
    """모델 → Anima LLLite 정보. 헤더를 읽으면 헤더가 정답(Anima 인지·채널), 못 읽으면 이름으로. 아니면 None."""
    header = read_safetensors_header(model_path)
    if header is not None:
        channels = lllite_channels_from_header(header)
        tile = lllite_tile_repair_from_header(header, model_name)
    else:
        channels = lllite_channels_from_name(model_name)
        tile = lllite_tile_repair_from_name(model_name)
    return None if channels is None else AnimaLLLite(channels, tile)


def lllite_channels(model_name: Any, model_path: Any = None) -> Optional[int]:
    """모델 → Anima LLLite 채널 수(``anima_lllite`` 의 채널만)."""
    info = anima_lllite(model_name, model_path)
    return None if info is None else info.channels


def forced_cn_module(module: Any, lllite: Optional[AnimaLLLite], model_name: Any = "") -> tuple[str, Optional[str]]:
    """(쓸 전처리기, 바꿨다면 그 이유) — 바꾸지 않으면 이유는 None. ``lllite`` 는 ``anima_lllite`` 결과."""
    current = "" if module is None else str(module)
    if lllite is None or current == NONE:
        return current, None
    if lllite.tile_repair:
        return NONE, (
            f"Anima Tile & Repair ControlNet-LLLite '{model_name}' takes the unprocessed image to repair "
            f"(like kohya sd-scripts / ComfyUI-Anima-LLLite); overriding preprocessor '{current}' to 'None'"
            + (" (inpaint_* would blank the region to repair)." if current.startswith("inpaint") else ".")
        )
    if not current.startswith("inpaint"):
        return current, None
    if lllite.channels == 4:
        return NONE, (
            f"LLLite inpaint model '{model_name}' is incompatible with preprocessor '{current}' "
            f"(preprocessor strips the mask); overriding to 'None' so the mask reaches the LLLite forward."
        )
    return NONE, (
        f"{lllite.channels}-channel Anima ControlNet-LLLite '{model_name}' ignores the mask (like kohya "
        f"ComfyUI-Anima-LLLite); preprocessor '{current}' would blank the masked region of the control image, "
        f"overriding to 'None'."
    )


__all__ = [
    "ANIMA_KEY_PREFIX", "AnimaLLLite", "COND_CONV1_KEY", "DEFAULT_COND_IN_CHANNELS", "META_COND_IN_CHANNELS",
    "META_TITLE", "NONE", "TILE_REPAIR_MARK", "anima_lllite", "forced_cn_module", "lllite_channels",
    "lllite_channels_from_header", "lllite_channels_from_name", "lllite_tile_repair_from_header",
    "lllite_tile_repair_from_name", "read_safetensors_header",
]
