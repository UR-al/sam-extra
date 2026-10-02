"""DoRA 추론 방식 — Forge/Comfy 의 ``weight_decompose`` 를 LyCORIS 학습 공식·fp32 계산으로 바꿔 끼운다.

Forge Neo 는 ``dora_scale`` 이 든 LoRA/LoKr/LoHa/GLoRA 를 ComfyUI 와 같은
``modules_forge/packages/comfy/weight_adapter/base.py:40`` ``weight_decompose`` 로 합친다. 학습한 LyCORIS
(``lycoris/modules/lokr.py`` ``apply_weight_decompose``)와 두 군데가 다르다.

1. 출력 축(``wd_on_output=True`` — LyCORIS 기본값, ``dora_scale`` [out,1])에서 Forge 는 **원본** 가중치의
   노름 ``‖W₀‖`` 로 나누고(base.py:47) LyCORIS 는 **합친** 가중치의 노름 ``‖W₀+ΔW‖`` 로 나눈다. 입력 축
   ([1,in])은 둘 다 ``‖W₀+ΔW‖`` 라 공식은 같다.
2. Forge 는 가중치를 ``lora_compute_dtype`` 로 합친다(backend/patcher/base.py:457). 대부분의 GPU(RTX 20xx·Volta
   이후, Windows 의 GTX 10xx, ROCm·DirectML·MPS)에서 fp16 이고 CPU·``--force-fp32``·GTX 16xx 만 fp32 다. 그래서
   노름·eps(fp16 9.8e-4)·곱셈이 fp16 이다. DoRA 는 가중치 전체를 다시 스케일하므로 이 반올림이 ΔW 에 비해 크게
   불어난다. LyCORIS 는 ``dora_scale`` dtype(학습 시 fp32)으로 계산하고 eps 는 fp32 값이다.

모드
- ``forge``      : 순정. 원래 함수를 그대로 부른다(비트 단위 동일).
- ``forge_fp32`` : Forge 공식 그대로, 계산만 fp32. 출력 축에서는 ``‖W₀‖`` 가 아주 작은 행이 fp16 eps 의 완충
  없이 그대로 커져서 순정보다 학습에서 더 멀어지는 레이어도 있다(실측: b39 q_proj). 입력 축 파일에 알맞다.
- ``lycoris``    : LyCORIS 학습 공식 + fp32. 강도 1.0 에서 학습 때 합쳐지던 가중치와 같다(실측: fp32 버퍼에서
  LyCORIS ``get_merged_weight`` 와 오차 0, 최종 bf16 반올림 하한의 1~3% 이내).
- ``no_magnitude``: 크기 보정을 모든 블록에서 끈다(DoRA 끔). ``dora_scale`` 이 없는 파일을 Forge 가 합치는 식
  (lora.py·lokr.py·loha.py·glora.py 의 ``weight += function(((strength * alpha) * lora_diff).type(weight.dtype))``)
  을 그대로 써서, dora_scale 을 지운 파일을 순정 Forge 에 넣은 것과 같다. 학습한 크기 조정분이 빠지므로 학습
  결과와 다르다 — DoRA 없이 학습한 LoKr 이라면 어떻게 나올지 근사해 보는 실험용.

강도(``<lora:x:0.7>``)는 두 fp32 모드 모두 Forge 처럼 ``W₀ + s·(W_dora − W₀)`` 로 섞는다. LyCORIS 자체의
multiplier 는 ΔW 를 늘 100% 넣고 크기 보정만 보간해서 강도 0 에서도 ``W₀+ΔW`` 가 남는데
(lokr.py ``get_merged_weight``/``apply_weight_decompose``), 그 동작은 따르지 않는다.

각 어댑터 모듈이 ``from .base import ... weight_decompose`` 로 이름을 복사해 두므로(lora.py:8, lokr.py:8,
loha.py:8, glora.py:8) base 가 아니라 **어댑터 모듈마다** 바꿔 끼운다. OFT/BOFT 는 alpha 자리에 OFT 제약을,
OFTv2 는 강도를 넘기므로(oft.py:81, oftv2.py:204) 건드리지 않는다.

일반 패치 경로(``patch_weight_to_device``)에서는 결과를 fp32 로 돌려줘서 Forge 가 한 번만 저장 dtype 으로
반올림하게 한다. 오프셋 패치(narrow 뷰)와 Low VRAM·온라인 LoRA 경로는 원래 함수처럼 제자리에 쓴다 — 뒤의 두
경로는 ΔW 가 이미 저장 dtype 으로 계산된 채 넘어오고, 매 forward 마다 다시 합친다.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Any, Callable, NamedTuple

LOGGER = logging.getLogger("sam-extra.dora-infer-mode")

MODE_FORGE = "forge"
MODE_FORGE_FP32 = "forge_fp32"
MODE_LYCORIS = "lycoris"
MODE_NO_MAGNITUDE = "no_magnitude"
MODES = (MODE_FORGE, MODE_FORGE_FP32, MODE_LYCORIS, MODE_NO_MAGNITUDE)

# infotext — Forge 파서(modules/infotext_utils.py re_param_code)는 키에 [\w\s\-/] 만 읽는다. 순정은 기록하지 않는다.
INFOTEXT_KEY = "DoRA mode"
INFOTEXT_VALUES = {MODE_FORGE_FP32: "Forge fp32", MODE_LYCORIS: "LyCORIS", MODE_NO_MAGNITUDE: "No magnitude"}

WEIGHT_ADAPTER_PACKAGE = "modules_forge.packages.comfy.weight_adapter"
TARGET_ADAPTERS = ("lora", "lokr", "loha", "glora")

# sd_model 에 붙이는 표시: 지금 합쳐 둔(또는 다음에 합칠) LoRA 가 어느 상태로 만들어졌는지. 모듈 전역이 아니라
# 모델에 두어야 "Reload UI"·확장 끄기처럼 전역만 초기화되는 경우에도 낡은 가중치를 알아본다.
# 상태는 (DoRA 방식, 끼워 넣은 블록 정책 — sam3ext.anima_lora_blocks.DUPLICATE_*) 튜플이다. 표시가 없는 모델은
# "알 수 없음"으로 본다 — 스크립트의 process() 를 거치지 않고 새로 올라온 모델(✨ 루프 안 재로드, hires 체크포인트,
# 🎯·Refine·IPA)이 어떤 상태로 합쳐졌는지 모르기 때문이다. 새 모델은 LoRA 를 어차피 새로 합치므로 비용은 거의 없다.
MERGED_STATE_ATTR = "_sam3_dora_merged_state"
STOCK_STATE: tuple = (MODE_FORGE, "keep")

_PATCH_OWNER = "sam-extra.dora-infer-mode.v1"
_OWNER_ATTR = "_sam3_dora_owner"
_ORIGINAL_ATTR = "_sam3_dora_original"

_STATE: dict[str, Any] = {
    "mode": MODE_FORGE,
    "calls": 0,
    "layer_ids": set(),
    "fallbacks": 0,
    "fallback_reason": None,
}
_INSTALLED: dict[str, tuple[Any, Callable]] = {}  # 어댑터 이름 → (모듈, 이번 로드가 끼운 래퍼)


class Counters(NamedTuple):
    layers: int              # fp32 로 합친 서로 다른 DoRA 레이어 수
    calls: int               # 합친 횟수(Low VRAM·온라인 LoRA 는 매 forward 라 레이어 수보다 크다)
    fallbacks: int           # 실패해서 순정으로 합친 횟수
    fallback_reason: str | None


# ── 모드 ──

_ALIASES = {
    "forge": MODE_FORGE,
    "comfy": MODE_FORGE,
    "forge/comfy": MODE_FORGE,
    "stock": MODE_FORGE,
    "off": MODE_FORGE,
    "forge_fp32": MODE_FORGE_FP32,
    "forge fp32": MODE_FORGE_FP32,
    "lycoris": MODE_LYCORIS,
    "no_magnitude": MODE_NO_MAGNITUDE,
    "no magnitude": MODE_NO_MAGNITUDE,
}


def normalize_mode(value: Any) -> str | None:
    """모드 키·UI 라벨·XYZ 값·infotext 값 → 모드 키. 모르는 값은 None."""
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return None
    if text in _ALIASES:
        return _ALIASES[text]
    if text.startswith("lycoris"):
        return MODE_LYCORIS
    if text.startswith(("dora off", "dora 끔", "no magnitude", "no_magnitude")):
        return MODE_NO_MAGNITUDE
    if text.startswith(("forge", "comfy")):
        return MODE_FORGE_FP32 if "fp32" in text else MODE_FORGE
    return None


def current_mode() -> str:
    return _STATE["mode"]


def set_mode(mode: Any) -> bool:
    """모드를 바꾸고, 실제로 바뀌었는지 돌려준다. 모르는 값은 ValueError."""
    normalized = normalize_mode(mode)
    if normalized is None:
        raise ValueError(f"unknown DoRA inference mode: {mode!r}")
    changed = normalized != _STATE["mode"]
    _STATE["mode"] = normalized
    return changed


def take_counters() -> Counters:
    """이번 생성의 계산 기록을 돌려주고 비운다."""
    counters = Counters(
        layers=len(_STATE["layer_ids"]),
        calls=int(_STATE["calls"]),
        fallbacks=int(_STATE["fallbacks"]),
        fallback_reason=_STATE["fallback_reason"],
    )
    _STATE["calls"] = 0
    _STATE["layer_ids"] = set()
    _STATE["fallbacks"] = 0
    _STATE["fallback_reason"] = None
    return counters


# ── 계산 ──

def decompose(dora_scale, weight, lora_diff, alpha, strength, function, *, merged_norm: bool, inplace: bool = True):
    """DoRA 를 fp32 로 계산한다.

    ``inplace`` 면 결과를 ``weight`` 에 제자리로 쓰고 ``weight`` 를 돌려준다 — 오프셋 패치(합쳐진 qkv 등)는 narrow
    뷰를 넘기고 반환값을 버리기 때문이다(backend/patcher/lora.py:47-65). ``weight`` 는 끝의 ``copy_`` 한 번으로만
    바뀌므로 도중에 실패하면 손대지 않은 상태로 남는다. ``inplace`` 가 거짓이면 fp32 결과 텐서를 그대로 돌려준다.

    ``merged_norm`` 이 참이면 출력 축도 ``‖W₀+ΔW‖`` 로 나누고(LyCORIS), 거짓이면 ``‖W₀‖``(Forge/Comfy).
    """
    import torch

    f32 = torch.float32
    w0 = weight.to(f32)  # 읽기 전용 — weight 가 이미 fp32 면 같은 텐서다
    diff = function(lora_diff.to(f32) * alpha)
    merged = w0 + diff.to(f32)
    del diff

    scale = dora_scale.to(device=merged.device, dtype=f32)
    if scale.shape[0] == merged.shape[0]:  # 출력 축 — Forge 와 같은 판정(base.py:45)
        source = merged if merged_norm else w0
        norm = source.reshape(source.shape[0], -1).norm(dim=1, keepdim=True)
        norm = norm.reshape(source.shape[0], *[1] * (source.dim() - 1))
    else:  # 입력 축 — Forge·LyCORIS 모두 합친 가중치
        norm = merged.transpose(0, 1).reshape(merged.shape[1], -1).norm(dim=1, keepdim=True)
        norm = norm.reshape(merged.shape[1], *[1] * (merged.dim() - 1)).transpose(0, 1)
    norm = norm + torch.finfo(f32).eps  # LyCORIS 는 fp32 eps 를 더한다(lokr.py _dora_eps)

    merged.mul_(scale / norm)
    if strength != 1.0:
        merged.sub_(w0).mul_(float(strength)).add_(w0)
    if not inplace:
        return merged
    weight.copy_(merged)
    return weight


def _can_return_fp32(weight, intermediate_dtype) -> bool:
    """일반 패치 경로인지. 그때만 fp32 결과를 돌려줘도 호출자가 받아 쓴다.

    patch_weight_to_device 는 merge_lora_to_weight 를 computation_dtype 없이(= fp32) 부르고 반환값을 저장
    dtype 으로 한 번 반올림한다(base.py:465-468, set_weight 도 입력을 자기 dtype 으로 바꾼다). Low VRAM·온라인
    LoRA 는 computation_dtype=weight.dtype 이라 여기서 걸러지고, narrow 뷰(오프셋 패치)는 ``_base`` 가 있다.
    """
    import torch

    return (
        getattr(weight, "_base", None) is None
        and intermediate_dtype == torch.float32
        and weight.dtype != torch.float32
    )


def _is_oom(exc: BaseException) -> bool:
    # Forge 안에서는 이미 import 돼 있다. 여기서 새로 import 하지 않는다(테스트에서 CUDA 를 건드리지 않게).
    memory_management = sys.modules.get("backend.memory_management")
    checker = getattr(memory_management, "is_oom", None)
    if callable(checker):
        try:
            return bool(checker(exc))
        except Exception:
            pass
    try:
        import torch

        oom = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", None)
        if oom is not None and isinstance(exc, oom):
            return True
    except Exception:
        pass
    return "out of memory" in str(exc).lower()


def _soft_empty_cache() -> None:
    memory_management = sys.modules.get("backend.memory_management")
    empty = getattr(memory_management, "soft_empty_cache", None)
    if callable(empty):
        try:
            empty()
        except Exception:
            pass


def _record_fallback(mode: str, reason: str) -> None:
    _STATE["fallbacks"] += 1
    if _STATE["fallback_reason"] is None:
        _STATE["fallback_reason"] = reason
        # Forge 의 어댑터도 계산 오류를 logging.error 로 남긴다(lokr.py:155).
        LOGGER.error("[sam-extra] DoRA %s: %s - falling back to stock merge for this layer", mode, reason)


def _make_wrapper(original: Callable) -> Callable:
    def weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype, function):
        mode = _STATE["mode"]
        if mode == MODE_FORGE:
            return original(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype, function)
        if mode == MODE_NO_MAGNITUDE:
            # dora_scale 이 없을 때 어댑터가 하는 것과 같은 한 줄(제자리 덧셈 — 오프셋 뷰도 그대로 동작).
            weight += function(((strength * alpha) * lora_diff).type(weight.dtype))
            _STATE["calls"] += 1
            _STATE["layer_ids"].add(id(dora_scale))
            return weight
        failure: tuple[bool, str] | None = None
        out = None
        try:
            out = decompose(
                dora_scale, weight, lora_diff, alpha, strength, function,
                merged_norm=(mode == MODE_LYCORIS),
                inplace=not _can_return_fp32(weight, intermediate_dtype),
            )
        except Exception as exc:
            failure = (_is_oom(exc), f"{type(exc).__name__}: {exc}")
        if failure is None:
            _STATE["calls"] += 1
            _STATE["layer_ids"].add(id(dora_scale))
            return out
        # except 절을 벗어난 뒤라 fp32 임시 텐서를 쥔 프레임은 이미 풀렸다. Forge 는 패치 중 OOM 을 재시도하지
        # 않고 생성을 멈추므로(base.py:792-797), 메모리를 비우고 순정(fp16, 임시 메모리가 더 적음)으로 합친다.
        oom, reason = failure
        if oom:
            _soft_empty_cache()
        _record_fallback(mode, ("OOM " if oom else "") + reason)
        return original(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype, function)

    setattr(weight_decompose, _OWNER_ATTR, _PATCH_OWNER)
    setattr(weight_decompose, _ORIGINAL_ATTR, original)
    return weight_decompose


# ── 설치 ──

def _load_adapter_module(name: str):
    try:
        return importlib.import_module(f"{WEIGHT_ADAPTER_PACKAGE}.{name}")
    except Exception:
        return None


def install(modules: dict[str, Any] | None = None) -> int:
    """어댑터 모듈마다 ``weight_decompose`` 를 래퍼로 바꾸고, 새로 끼운 수를 돌려준다. 여러 번 불러도 같다.

    ``modules`` 는 테스트용 ``{어댑터 이름: 모듈}``. None 이면 Forge 패키지에서 가져온다.
    """
    count = 0
    for name in TARGET_ADAPTERS:
        module = modules.get(name) if modules is not None else _load_adapter_module(name)
        if module is None:
            continue
        current = getattr(module, "weight_decompose", None)
        if not callable(current):
            continue
        mine = _INSTALLED.get(name)
        if mine is not None and mine[0] is module and current is mine[1]:
            continue
        if getattr(current, _OWNER_ATTR, None) == _PATCH_OWNER:
            # "Reload scripts" 뒤 남은 예전 로드의 래퍼 — 그 모드 상태는 이제 아무도 못 바꾸므로 원본에 새로 끼운다.
            original = getattr(current, _ORIGINAL_ATTR, None)
            if not callable(original):
                continue
        else:
            original = current
        wrapper = _make_wrapper(original)
        setattr(module, "weight_decompose", wrapper)
        _INSTALLED[name] = (module, wrapper)
        count += 1
    return count


def uninstall() -> int:
    """이번 로드가 끼운 래퍼를 원본으로 되돌리고, 되돌린 수를 돌려준다.

    다른 확장이 위에 감싸서 떼어 낼 수 없는 래퍼도 순정으로 통과하도록 모드를 먼저 순정으로 돌린다. 순정이 아닌
    방식으로 합쳐 둔 가중치가 남아 있을 수 있으므로(Reload UI 는 모델을 다시 올리지 않는다) 현재 모델의 LoRA·조건
    캐시도 버린다 — 확장을 끄고 다시 불러오면 process() 가 다시 돌지 않기 때문이다.
    """
    _STATE["mode"] = MODE_FORGE
    restored = 0
    for name, (module, wrapper) in list(_INSTALLED.items()):
        if getattr(module, "weight_decompose", None) is wrapper:
            setattr(module, "weight_decompose", getattr(wrapper, _ORIGINAL_ATTR))
            restored += 1
        _INSTALLED.pop(name, None)
    _invalidate_current_model_if_dirty()
    return restored


def _invalidate_current_model_if_dirty() -> None:
    shared = sys.modules.get("modules.shared")
    try:
        sd_model = getattr(shared, "sd_model", None) if shared is not None else None
    except Exception:
        sd_model = None
    if sd_model is None or getattr(sd_model, MERGED_STATE_ATTR, None) == STOCK_STATE:
        return
    invalidate_lora_cache(None, sd_model)
    clear_cond_caches(None, _processing_classes())
    setattr(sd_model, MERGED_STATE_ATTR, STOCK_STATE)


def _processing_classes() -> tuple:
    processing = sys.modules.get("modules.processing")
    return tuple(
        klass
        for klass in (
            getattr(processing, "StableDiffusionProcessing", None),
            getattr(processing, "StableDiffusionProcessingTxt2Img", None),
        )
        if klass is not None
    )


def tag_model(sd_model, state: tuple) -> None:
    """새로 올라온 모델에 표시를 붙인다(Forge on_model_loaded). 이후 LoRA 는 지금 전역 상태로 합쳐진다."""
    if sd_model is not None:
        try:
            setattr(sd_model, MERGED_STATE_ATTR, tuple(state))
        except Exception:
            pass


def installed_adapters() -> tuple[str, ...]:
    return tuple(name for name in TARGET_ADAPTERS if name in _INSTALLED)


# ── 캐시 ──

_COND_CACHE_NAMES = ("cached_c", "cached_uc", "cached_hr_c", "cached_hr_uc")


def invalidate_lora_cache(p=None, sd_model=None) -> None:
    """Forge 가 합쳐 둔 LoRA 가중치를 다음 생성에서 새로 만들게 하고, 조건 캐시를 비운다.

    ``networks.load_networks`` 는 LoRA 목록 해시가 같으면 아무것도 하지 않는다(sd_forge_lora/networks.py:186-188).
    해시를 ``None`` 으로 두면 어떤 목록과도 같지 않아서 — ``str([])`` 는 LoRA 없는 프롬프트의 해시와 같아 이전 LoRA
    가 남는다 — 원본 패처에서 다시 만든다(LoRA Block Weight 확장과 같은 방법, lora_block_weight.py:702). 새 패처는
    새 ``patches_uuid`` 를 가져서 다음 로드 때 새 방식으로 다시 합친다.
    """
    models: list[Any] = []
    for candidate in (sd_model, getattr(p, "sd_model", None)):
        if candidate is not None and not any(candidate is seen for seen in models):
            models.append(candidate)
    for model in models:
        if hasattr(model, "current_lora_hash"):
            model.current_lora_hash = None
    if p is not None:
        # img2img 에서 바꿔도 txt2img 의 hires 조건 캐시(클래스 목록)까지 비운다 — p 의 MRO 에는 없다.
        clear_cond_caches(p, _processing_classes())


def clear_cond_caches(p=None, classes: tuple = ()) -> None:
    """조건 캐시를 **제자리에서** 비운다.

    캐시 키(processing.py:446 ``cached_params``)에는 DoRA 방식이 없어서 텍스트 인코더 쪽 DoRA(Anima 의 llm_adapter
    키 등)로 만든 조건이 남는다. 목록을 새로 바꾸면 안 된다 — XYZ 칸은 ``copy(p)`` 로 원래 p 의 목록을 같이 쥐고
    있어서(xyz_grid.py:769) 바꾼 뒤에도 옛 목록의 조건을 다시 쓴다. 길이(2 또는 3)는 그대로 둔다.
    """
    lists: list[list] = []

    def collect(value) -> None:
        if isinstance(value, list) and not any(value is seen for seen in lists):
            lists.append(value)

    targets = list(classes)
    if p is not None:
        for name in _COND_CACHE_NAMES:
            collect(getattr(p, name, None))
        targets.extend(type(p).__mro__)
    for klass in targets:
        for name in _COND_CACHE_NAMES:
            collect(vars(klass).get(name) if hasattr(klass, "__dict__") else None)
    for value in lists:
        value[:] = [None] * len(value)


def sync_merged_state(p, sd_model, state: tuple) -> bool:
    """모델에 붙은 표시가 이번 생성의 상태와 다르거나 없으면 캐시를 버리고 표시를 바꾼다. 버렸으면 True."""
    model = sd_model if sd_model is not None else getattr(p, "sd_model", None)
    if model is None:
        return False
    if getattr(model, MERGED_STATE_ATTR, None) == state:
        return False
    invalidate_lora_cache(p, model)
    setattr(model, MERGED_STATE_ATTR, state)
    return True


__all__ = [
    "Counters",
    "INFOTEXT_KEY",
    "INFOTEXT_VALUES",
    "MERGED_STATE_ATTR",
    "MODES",
    "MODE_FORGE",
    "MODE_FORGE_FP32",
    "MODE_LYCORIS",
    "MODE_NO_MAGNITUDE",
    "STOCK_STATE",
    "TARGET_ADAPTERS",
    "clear_cond_caches",
    "current_mode",
    "decompose",
    "install",
    "installed_adapters",
    "invalidate_lora_cache",
    "normalize_mode",
    "set_mode",
    "sync_merged_state",
    "tag_model",
    "take_counters",
    "uninstall",
]
