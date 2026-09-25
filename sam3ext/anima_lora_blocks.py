"""ANIMA LoRA block-layout compatibility for Forge.

ANIMA Base 1.0, 2.9B, and 3.8B use the same 2048-wide DiT block
implementation at different depths (28, 40, and 52 blocks).  Forge remaps
some upward combinations; this module owns the complete compatibility matrix
without modifying Forge itself:

* 28 <-> 40 (Base 1.0 <-> 2.9B)
* 28 <-> 52 (Base 1.0 <-> 3.8B)
* 40 <-> 52 (2.9B <-> 3.8B)

Mappings are expressed as ``target index -> source index``.  Expansion clones
the preceding source-lineage block into each inserted position.  Contraction
selects the original lineage positions and deliberately drops generation-only
blocks.  Only complete, contiguous LoRA layouts are remapped: guessing the
source architecture of a sparse/partial LoRA would silently attach weights to
the wrong semantic depth.  A sparse LoRA is still passed through untouched
when Forge's own rule (smallest layout holding its highest block index) names
the active model, because no conversion is needed there and vanilla Forge
loads it as-is; every other sparse case is refused with a UI notice instead
of the silent skip Forge would otherwise perform.  The opt-in setting
``sam3_anima_sparse_lora_forge_guess`` restores Forge's guess instead: the
sparse LoRA is converted from that inferred layout with this module's tables
(inserted-block policy included), and the generation's infotext records it.
"""

from __future__ import annotations

import importlib
import logging
import re
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, MutableMapping


LOGGER = logging.getLogger("sam-extra.anima-lora-blocks")

ANIMA_BASE_BLOCKS = 28
ANIMA_29B_BLOCKS = 40
ANIMA_38B_BLOCKS = 52
SUPPORTED_BLOCK_COUNTS = (
    ANIMA_BASE_BLOCKS,
    ANIMA_29B_BLOCKS,
    ANIMA_38B_BLOCKS,
)

LAYOUT_NAMES = {
    ANIMA_BASE_BLOCKS: "ANIMA Base 1.0",
    ANIMA_29B_BLOCKS: "ANIMA 2.9B",
    ANIMA_38B_BLOCKS: "ANIMA 3.8B",
}

_BLOCK_KEY_RE = re.compile(
    r"^(lora_unet_blocks_|diffusion_model\.blocks\.)(\d+)(.*)$"
)

# The semantic connector is bundled only into ANIMA 3.8B v1.1.  Qwen3.5 is a
# separate encoder and must not be classified by DiT block count, so only
# exact connector namespaces are filtered during a downward projection.
_ANIMA_38B_ONLY_KEY_PREFIXES = (
    "net.anima_v2_connector.",
    "diffusion_model.anima_v2_connector.",
    "lora_unet_anima_v2_connector_",
)

# Forge's Base 1.0 -> 2.9B mapping.  Repeated entries are the twelve blocks
# inserted by the 40-block model and inherit the preceding Base block.
_ANIMA_28_TO_40 = (
    0,
    1,
    1,
    2,
    3,
    3,
    4,
    5,
    5,
    6,
    7,
    7,
    8,
    9,
    9,
    10,
    11,
    11,
    12,
    13,
    14,
    14,
    15,
    16,
    16,
    17,
    18,
    18,
    19,
    20,
    20,
    21,
    22,
    22,
    23,
    24,
    24,
    25,
    26,
    27,
)

# These are the 28 original Base lineage positions inside the 40-block model.
# This must not be derived by taking the first occurrence in _ANIMA_28_TO_40:
# Forge's source checkpoints identify these exact blocks as the shared weights.
_ANIMA_40_TO_28 = (
    0,
    1,
    3,
    4,
    6,
    7,
    9,
    10,
    12,
    13,
    15,
    16,
    18,
    19,
    20,
    22,
    23,
    25,
    26,
    28,
    29,
    31,
    32,
    34,
    35,
    37,
    38,
    39,
)

# Anima-3.8B-v1.1.safetensors records this expansion in its own metadata:
# old_block_count=40, new_block_count=52, source_method="Anima-2.9B /
# LLaMA-Pro block expansion", and the following insertion/source pairs.
_ANIMA_38B_INSERTED_TO_29B_SOURCE = {
    3: 2,
    7: 5,
    11: 8,
    15: 11,
    19: 14,
    23: 17,
    27: 20,
    31: 23,
    35: 26,
    39: 29,
    43: 32,
    47: 35,
}


def _make_expansion_mapping(
    source_count: int,
    inserted_to_source: dict[int, int],
) -> tuple[int, ...]:
    """Build a target->source expansion mapping and validate every position."""

    target_count = source_count + len(inserted_to_source)
    mapping: list[int] = []
    next_source = 0
    for target_index in range(target_count):
        if target_index in inserted_to_source:
            source_index = inserted_to_source[target_index]
            if source_index != next_source - 1:
                raise ValueError(
                    "ANIMA inserted block must clone the immediately preceding "
                    f"source block: target={target_index}, source={source_index}, "
                    f"expected={next_source - 1}"
                )
            mapping.append(source_index)
            continue
        if next_source >= source_count:
            raise ValueError("ANIMA expansion consumed more source blocks than exist")
        mapping.append(next_source)
        next_source += 1

    if next_source != source_count:
        raise ValueError(
            f"ANIMA expansion consumed {next_source}/{source_count} source blocks"
        )
    return tuple(mapping)


def _make_contraction_mapping(
    target_count: int,
    source_count: int,
    inserted_positions: set[int],
) -> tuple[int, ...]:
    """Return target->source positions while excluding inserted source blocks."""

    mapping = tuple(
        source_index
        for source_index in range(source_count)
        if source_index not in inserted_positions
    )
    if len(mapping) != target_count:
        raise ValueError(
            f"ANIMA contraction produced {len(mapping)} positions, expected {target_count}"
        )
    return mapping


_ANIMA_40_TO_52 = _make_expansion_mapping(
    ANIMA_29B_BLOCKS,
    _ANIMA_38B_INSERTED_TO_29B_SOURCE,
)
_ANIMA_52_TO_40 = _make_contraction_mapping(
    ANIMA_29B_BLOCKS,
    ANIMA_38B_BLOCKS,
    set(_ANIMA_38B_INSERTED_TO_29B_SOURCE),
)

# Compose target->source mappings through the 40-block lineage.
_ANIMA_28_TO_52 = tuple(
    _ANIMA_28_TO_40[source_40] for source_40 in _ANIMA_40_TO_52
)
_ANIMA_52_TO_28 = tuple(
    _ANIMA_52_TO_40[source_40] for source_40 in _ANIMA_40_TO_28
)

BLOCK_MAPPINGS: dict[tuple[int, int], tuple[int, ...]] = {
    (ANIMA_BASE_BLOCKS, ANIMA_29B_BLOCKS): _ANIMA_28_TO_40,
    (ANIMA_29B_BLOCKS, ANIMA_BASE_BLOCKS): _ANIMA_40_TO_28,
    (ANIMA_29B_BLOCKS, ANIMA_38B_BLOCKS): _ANIMA_40_TO_52,
    (ANIMA_38B_BLOCKS, ANIMA_29B_BLOCKS): _ANIMA_52_TO_40,
    (ANIMA_BASE_BLOCKS, ANIMA_38B_BLOCKS): _ANIMA_28_TO_52,
    (ANIMA_38B_BLOCKS, ANIMA_BASE_BLOCKS): _ANIMA_52_TO_28,
}


# 상향 변환에서 끼워 넣은(복제된) 블록을 어떻게 채울지. 3.8B 의 끼워 넣은 블록은 출력 투영이 원본과 무관하고
# 행이 훨씬 짧아서(중앙값 0.4배), 2.9B 에서 학습한 DoRA 의 절대 크기(dora_scale)를 그대로 복제하면 행 배율이
# 레이어별 중앙값 1.3~11.8배가 된다. 학습량과 거의 무관하게(1에포크 파일도 비슷한 크기로) 생기는 변화다.
DUPLICATE_KEEP = "keep"          # 그대로 복제(Forge 기본)
DUPLICATE_ADDITIVE = "additive"  # 복제본에서 dora_scale 만 뺀다 → W₀ + ΔW (크기 보정 없음)
DUPLICATE_SKIP = "skip"          # 복제본을 만들지 않는다 → 끼워 넣은 블록은 LoRA 없음(모든 LoRA)
# 약한 복사(Civitai "Anima 2B LoRA Bridge"의 copy_prev 방식): 복제본을 덧셈형으로 두고 ΔW 를 strength 배로 줄여,
# 범위(어텐션만 / 어텐션+MLP / 전체) 안의 모듈에만 넣는다. 브리지 권장값은 강도 0.08~0.18, 어텐션만.
DUPLICATE_WEAK = "weak"
DUPLICATE_POLICIES = (DUPLICATE_KEEP, DUPLICATE_ADDITIVE, DUPLICATE_SKIP, DUPLICATE_WEAK)
_DORA_SCALE_SUFFIX = ".dora_scale"

WEAK_SCOPE_ATTN = "attn"          # self_attn·cross_attn 투영만
WEAK_SCOPE_ATTN_MLP = "attn_mlp"  # + mlp
WEAK_SCOPE_ALL = "all"            # 블록 안의 모든 모듈(모듈레이션·노름 포함)
WEAK_SCOPES = (WEAK_SCOPE_ATTN, WEAK_SCOPE_ATTN_MLP, WEAK_SCOPE_ALL)
DEFAULT_WEAK_STRENGTH = 0.12

_DUPLICATE_POLICY = DUPLICATE_KEEP
_WEAK = {"strength": DEFAULT_WEAK_STRENGTH, "scope": WEAK_SCOPE_ATTN}

# 부분(sparse) LoRA 순정 추측 변환 토글(Forge 설정, 기본 끔). 등록은 scripts/anima_lora_blocks.py.
OPT_SPARSE_FORGE_GUESS = "sam3_anima_sparse_lora_forge_guess"
SPARSE_GUESS_LABEL = "부분 LoRA 순정 추측 변환"
# 추측 변환한 생성의 infotext 키. 설정의 infotext 이름이기도 해서, 붙여 넣으면 이 설정을 켜는 덮어쓰기가 된다.
INFOTEXT_SPARSE_GUESS_KEY = "Anima sparse LoRA"
SPARSE_GUESS_HINT = (
    f"설정 → SAM Extra LoRA 의 '{SPARSE_GUESS_LABEL}'을 켜면 순정 Forge 처럼 판정 레이아웃에서 변환해 "
    "로드합니다(블록 대응이 틀릴 수 있음)."
)
# 이번 LoRA 로드 묶음(Forge current_lora_hash)에서 토글에 따라 결과가 갈리는 부분 LoRA 처리를 모델에 적어 둔다.
SPARSE_STATE_ATTR = "_sam3_anima_sparse_lora_state"

# 부분 LoRA 의 모양: 앞 N블록만(접두) / 중간이 빔 / 지원 레이아웃으로 판정 불가(인덱스 52 이상).
SPARSE_PREFIX = "prefix"
SPARSE_GAP = "gap"
SPARSE_UNKNOWN = "unknown"

# 어댑터 인자 이름(키의 마지막 부분). 약한 복사는 모듈마다 아래 우선순위의 인자 하나만 strength 배 해서 ΔW 를
# 정확히 strength 배로 만든다: LoRA up·B(ΔW=up@down), LoKr w1 또는 w1_a(ΔW=kron(w1,w2)), LoHa w1_a
# (ΔW=(w1a@w1b)⊙(w2a@w2b)), 전체 diff. 이 중 하나도 없는 모듈은 어떻게 줄일지 몰라 복사하지 않는다.
_ADAPTER_PARAMS = (
    "lora_up.weight", "lora_down.weight", "lora_mid.weight", "lora_A.weight", "lora_B.weight",
    "lora.up.weight", "lora.down.weight", "alpha", "dora_scale",
    "lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b", "lokr_t2",
    "hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "hada_t1", "hada_t2",
    "diff", "diff_b",
)
_WEAK_SCALE_PRIORITY = ("lora_up.weight", "lora_B.weight", "lora.up.weight", "lokr_w1", "lokr_w1_a", "hada_w1_a", "diff")


def normalize_duplicate_policy(value: Any) -> str | None:
    """정책 키·UI 라벨·XYZ 값·infotext 값 → 정책 키. 모르는 값은 None."""
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return None
    for policy in DUPLICATE_POLICIES:
        if policy in text:
            return policy
    if "약한" in text:
        return DUPLICATE_WEAK
    if "덧셈" in text:
        return DUPLICATE_ADDITIVE
    if "넣지 않" in text:
        return DUPLICATE_SKIP
    if "그대로" in text or "순정" in text:
        return DUPLICATE_KEEP
    return None


def set_duplicate_policy(policy: Any) -> bool:
    """전역 정책을 바꾸고, 바뀌었는지 돌려준다. 모르는 값은 ValueError."""
    global _DUPLICATE_POLICY
    normalized = normalize_duplicate_policy(policy)
    if normalized is None:
        raise ValueError(f"unknown duplicate-block policy: {policy!r}")
    changed = normalized != _DUPLICATE_POLICY
    _DUPLICATE_POLICY = normalized
    return changed


def current_duplicate_policy() -> str:
    return _DUPLICATE_POLICY


def normalize_weak_scope(value: Any) -> str | None:
    """범위 키·UI 라벨 → attn / attn_mlp / all. 모르는 값은 None."""
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return None
    if "mlp" in text:
        return WEAK_SCOPE_ATTN_MLP
    if text.startswith("all") or "전체" in text:
        return WEAK_SCOPE_ALL
    if "attn" in text or "어텐션" in text:
        return WEAK_SCOPE_ATTN
    return None


def set_weak_copy(strength: Any, scope: Any) -> bool:
    """약한 복사의 강도(0~2, 소수 셋째 자리)와 범위를 바꾸고, 바뀌었는지 돌려준다."""
    try:
        value = round(min(2.0, max(0.0, float(strength))), 3)
    except (TypeError, ValueError):
        value = DEFAULT_WEAK_STRENGTH
    area = normalize_weak_scope(scope) or WEAK_SCOPE_ATTN
    changed = (value, area) != (_WEAK["strength"], _WEAK["scope"])
    _WEAK["strength"], _WEAK["scope"] = value, area
    return changed


def current_weak_copy() -> tuple[float, str]:
    return _WEAK["strength"], _WEAK["scope"]


def policy_state_key(policy: str | None = None) -> str:
    """정책을 한 문자열로. 약한 복사는 강도·범위까지 담는다(infotext 값과 캐시 상태 비교에 같이 쓴다)."""
    current = normalize_duplicate_policy(policy) or _DUPLICATE_POLICY
    if current != DUPLICATE_WEAK:
        return current
    return f"weak {_WEAK['strength']:g} {_WEAK['scope']}"


def parse_weak_key(value: Any) -> tuple[float | None, str | None]:
    """'weak 0.12 attn' → (0.12, 'attn'). 없는 항목은 None."""
    import re as _re

    text = str(value if value is not None else "")
    match = _re.search(r"[-+]?\d*\.?\d+", text)
    strength = float(match.group()) if match else None
    rest = text[match.end():] if match else text
    return strength, normalize_weak_scope(rest)


def _split_module_param(suffix: str) -> tuple[str, str | None]:
    for param in sorted(_ADAPTER_PARAMS, key=len, reverse=True):
        if suffix.endswith("." + param):
            return suffix[: -len(param) - 1], param
    return suffix, None


def _in_weak_scope(module: str, scope: str) -> bool:
    if scope == WEAK_SCOPE_ALL:
        return True
    name = module.lower()
    if "modulation" in name or "norm" in name:
        return False
    if "self_attn" in name or "cross_attn" in name:
        return True
    return scope == WEAK_SCOPE_ATTN_MLP and "mlp" in name


@dataclass(frozen=True)
class BlockRemapResult:
    """Observable result of one in-place LoRA compatibility pass."""

    source_blocks: int | None
    target_blocks: int
    supported: bool
    remapped: bool
    moved_adapter_keys: int = 0
    dropped_auxiliary_keys: int = 0
    duplicate_policy: str = DUPLICATE_KEEP
    dropped_dora_keys: int = 0          # additive·weak: 복제본에서 뺀 dora_scale 수
    skipped_duplicate_keys: int = 0     # skip: 만들지 않은 복제 키 수 / weak: 범위 밖·줄일 수 없는 키 수
    weak_strength: float = 0.0
    weak_scope: str = ""
    weak_scaled_modules: int = 0        # weak: strength 배로 넣은 모듈 수
    # 순정 Forge 규칙으로 판정한 원본 레이아웃(가장 큰 블록 인덱스+1 이 들어가는 가장 작은 레이아웃).
    # 완전한 레이아웃이면 source_blocks 와 같고, 부분(sparse) LoRA 는 이 값이 target_blocks 와 같을 때만 통과한다.
    inferred_blocks: int | None = None
    # 부분 LoRA 를 토글(OPT_SPARSE_FORGE_GUESS)로 추측 변환했는지. True 면 source_blocks 는 판정값(추측)이다.
    guessed: bool = False

    @property
    def duplicated_blocks(self) -> int:
        if not self.remapped or self.source_blocks is None:
            return 0
        return max(0, self.target_blocks - self.source_blocks)

    @property
    def dropped_blocks(self) -> int:
        if not self.remapped or self.source_blocks is None:
            return 0
        return max(0, self.source_blocks - self.target_blocks)


def anima_lora_block_indices(lora: MutableMapping[str, Any]) -> set[int]:
    """Return the exact ANIMA DiT block indices represented by a LoRA."""

    return {
        int(match.group(2))
        for key in lora
        if (match := _BLOCK_KEY_RE.match(key)) is not None
    }


def detect_anima_lora_layout(lora: MutableMapping[str, Any]) -> int | None:
    """Detect a complete supported layout; refuse sparse or non-contiguous sets."""

    indices = anima_lora_block_indices(lora)
    for block_count in SUPPORTED_BLOCK_COUNTS:
        if indices == set(range(block_count)):
            return block_count
    return None


def infer_forge_lora_layout(indices: set[int]) -> int | None:
    """순정 Forge(networks.process_anima)와 같은 규칙으로 원본 레이아웃을 판정한다.

    가장 큰 블록 인덱스+1 이 들어가는 가장 작은 지원 레이아웃. 블록 키가 없거나 52 를 넘으면 None.
    부분 LoRA 에서는 추측일 뿐이므로(앞 21블록만 있는 LoRA 는 28·40·52 어느 쪽일 수도 있다) 기본은 변환 근거로
    쓰지 않고, 판정이 현재 모델과 같을 때 '변환 불필요' 로 통과시키는 데만 쓴다. 설정 토글
    (OPT_SPARSE_FORGE_GUESS)을 켜면 순정처럼 이 판정에서 변환한다.
    """

    if not indices:
        return None
    needed = max(indices) + 1
    return next((size for size in SUPPORTED_BLOCK_COUNTS if needed <= size), None)


def sparse_lora_kind(indices: set[int]) -> str:
    """부분 LoRA 의 모양(안내문 갈래). 완전한 레이아웃은 호출자가 먼저 거른다.

    접두(0 부터 빈틈없이 앞 N블록)는 판정이 가장 큰 인덱스로만 정해져 원래 모델이 더 클 수 있고, 중간이 빈 것은
    빠진 자리가 원래 어디였는지 모른다. 52 이상 인덱스는 어떤 레이아웃으로도 판정하지 못한다.
    """

    if infer_forge_lora_layout(indices) is None:
        return SPARSE_UNKNOWN
    return SPARSE_PREFIX if indices == set(range(max(indices) + 1)) else SPARSE_GAP


def sparse_forge_guess_enabled() -> bool:
    """Forge 설정 토글(부분 LoRA 순정 추측 변환). Forge 밖·미등록·bool 이 아닌 값(스텁)은 끔."""

    try:
        from modules import shared

        value = getattr(getattr(shared, "opts", None), OPT_SPARSE_FORGE_GUESS, False)
    except Exception:
        return False
    return value is True


def _forge_sd_model() -> Any:
    try:
        from modules import shared

        return getattr(shared, "sd_model", None)
    except Exception:
        return None


def _sparse_state_for(model: Any) -> dict | None:
    """모델에 적힌 부분 LoRA 기록이 지금 합쳐진 LoRA 묶음(current_lora_hash)의 것일 때만 돌려준다."""

    state = getattr(model, SPARSE_STATE_ATTR, None) if model is not None else None
    if not isinstance(state, dict):
        return None
    if state.get("hash") != getattr(model, "current_lora_hash", object()):
        return None
    return state


def _record_sparse_decision(name: str, result: "BlockRemapResult", enabled: bool) -> None:
    """토글에 따라 결과가 갈리는 부분 LoRA 를 합칠 때마다 모델에 적는다(infotext·토글 변경 감지용).

    ``networks.load_networks`` 는 새 해시를 모델에 적은 뒤 LoRA 를 하나씩 합치므로(sd_forge_lora/networks.py
    load_networks) 해시가 다르면 새 묶음으로 보고 다시 시작한다. 같은 해시를 다시 합칠 때(캐시 무효화 뒤)는
    이름별로 덮어써, 토글을 끄고 다시 합친 LoRA 가 추측 기록에 남지 않게 한다.
    """

    model = _forge_sd_model()
    if model is None:
        return
    try:
        state = _sparse_state_for(model)
        if state is None:
            state = {"hash": getattr(model, "current_lora_hash", None), "enabled": enabled, "guessed": {}}
        state["enabled"] = enabled
        if result.guessed:
            state["guessed"][name] = f"{result.source_blocks}->{result.target_blocks}"
        else:
            state["guessed"].pop(name, None)
        setattr(model, SPARSE_STATE_ATTR, state)
    except Exception:
        LOGGER.debug("[sam-extra] could not record the sparse ANIMA LoRA state", exc_info=True)


def sparse_guess_record(model: Any) -> dict[str, str]:
    """지금 합쳐진 LoRA 묶음에서 추측 변환한 LoRA {파일 이름: '28->40'}. 없거나 옛 묶음이면 빈 dict."""

    state = _sparse_state_for(model)
    return dict(state["guessed"]) if state else {}


def sparse_cache_stale(model: Any, enabled: bool) -> bool:
    """지금 합쳐진 LoRA 묶음에 토글이 영향을 준 부분 LoRA 가 있고, 그때의 토글이 지금과 다르면 True.

    Forge 는 LoRA 목록 해시가 같으면 다시 합치지 않아서(설정을 바꾸거나 infotext 덮어쓰기로 켜도) 옛 결과가
    남는다 — 이때 호출자가 캐시를 버린다.
    """

    state = _sparse_state_for(model)
    return bool(state) and bool(state.get("enabled")) != bool(enabled)


def forget_sparse_state(model: Any) -> None:
    try:
        if model is not None and hasattr(model, SPARSE_STATE_ATTR):
            delattr(model, SPARSE_STATE_ATTR)
    except Exception:
        pass


def _move_llm_adapter_keys(lora: MutableMapping[str, Any]) -> int:
    """Mirror Forge's ANIMA LLM-adapter namespace migration."""

    moved = 0
    for key in list(lora):
        if key.startswith("diffusion_model.llm_adapter"):
            lora[key.replace("diffusion_model", "text_encoders.qwen3_06b", 1)] = (
                lora.pop(key)
            )
            moved += 1
        elif key.startswith("lora_unet_llm_adapter"):
            lora[key.replace("lora_unet_llm_adapter", "lora_te_llm_adapter", 1)] = (
                lora.pop(key)
            )
            moved += 1
    return moved


def _drop_38b_only_keys(lora: MutableMapping[str, Any]) -> int:
    dropped = 0
    for key in list(lora):
        lowered = key.lower()
        if lowered.startswith(_ANIMA_38B_ONLY_KEY_PREFIXES):
            lora.pop(key)
            dropped += 1
    return dropped


def _38b_only_keys(lora: MutableMapping[str, Any]) -> list[str]:
    return [
        key
        for key in lora
        if key.lower().startswith(_ANIMA_38B_ONLY_KEY_PREFIXES)
    ]


def _clone_value(value: Any) -> Any:
    clone = getattr(value, "clone", None)
    return clone() if callable(clone) else value


def _remap_blocks_in_place(
    lora: MutableMapping[str, Any],
    mapping: tuple[int, ...],
    duplicate_policy: str = DUPLICATE_KEEP,
    weak_strength: float = DEFAULT_WEAK_STRENGTH,
    weak_scope: str = WEAK_SCOPE_ATTN,
) -> tuple[int, int, int]:
    """Replace all block keys using one validated target->source mapping.

    같은 원본이 두 번째 이후로 나오는 타깃이 끼워 넣은 블록이다. 각 매핑은 원본 번호가 늘어나는 순서이고
    삽입은 바로 앞 원본의 복제라서 원래 계보 블록이 늘 먼저 온다. 돌려주는 값은
    (뺀 dora_scale 수, 만들지 않은 키 수, 약한 복사로 넣은 모듈 수). 28→52 처럼 두 세대를 합성한 매핑에서도 약한
    복사는 끼워 넣은 모든 블록에 같은 강도로 한 번만 들어간다(세대마다 곱해지지 않는다).
    """

    source_blocks: dict[int, list[tuple[str, str, Any]]] = {}
    for key in list(lora):
        match = _BLOCK_KEY_RE.match(key)
        if match is None:
            continue
        source_blocks.setdefault(int(match.group(2)), []).append(
            (match.group(1), match.group(3), lora.pop(key))
        )

    emitted_sources: set[int] = set()
    dropped_dora = 0
    skipped = 0
    scaled_modules = 0
    for target_index, source_index in enumerate(mapping):
        duplicate_source = source_index in emitted_sources
        # 추측 변환한 부분 LoRA 는 빠진 원본 블록이 있다 — 그 자리는 순정처럼 비워 둔다.
        entries_at_source = source_blocks.get(source_index, ())
        if duplicate_source and duplicate_policy == DUPLICATE_WEAK:
            groups: dict[str, list[tuple[str, str, Any, str | None]]] = {}
            for prefix, suffix, value in entries_at_source:
                module, param = _split_module_param(suffix)
                groups.setdefault(module, []).append((prefix, suffix, value, param))
            for module, entries in groups.items():
                params = {entry[3] for entry in entries}
                scale_param = next((name for name in _WEAK_SCALE_PRIORITY if name in params), None)
                if scale_param is None or not _in_weak_scope(module, weak_scope):
                    skipped += len(entries)
                    continue
                for prefix, suffix, value, param in entries:
                    if param == "dora_scale":
                        dropped_dora += 1
                        continue
                    copied = _clone_value(value)
                    if param in (scale_param, "diff_b"):
                        copied = copied * weak_strength
                    lora[f"{prefix}{target_index}{suffix}"] = copied
                scaled_modules += 1
            continue
        for prefix, suffix, value in entries_at_source:
            if duplicate_source and duplicate_policy == DUPLICATE_SKIP:
                skipped += 1
                continue
            if (
                duplicate_source
                and duplicate_policy == DUPLICATE_ADDITIVE
                and suffix.endswith(_DORA_SCALE_SUFFIX)
            ):
                dropped_dora += 1
                continue
            lora[f"{prefix}{target_index}{suffix}"] = (
                _clone_value(value) if duplicate_source else value
            )
        emitted_sources.add(source_index)
    return dropped_dora, skipped, scaled_modules


def remap_anima_lora_for_model(
    lora: MutableMapping[str, Any],
    target_blocks: int,
    duplicate_policy: str | None = None,
    sparse_forge_guess: bool = False,
) -> BlockRemapResult:
    """Mutate one loaded LoRA for the active ANIMA model block count.

    The function is the module's main interface and has no Forge dependency,
    making the exact conversion matrix directly testable.  Unsupported sparse
    block layouts are not remapped and are reported with ``supported=False``;
    Forge's independent LLM-adapter namespace normalization still applies.

    ``duplicate_policy`` 가 None 이면 전역 정책(:func:`set_duplicate_policy`)을 쓴다. 상향 변환의 끼워 넣은
    블록에만 적용되고, 하향 변환·같은 블록 수에는 영향이 없다.

    ``sparse_forge_guess`` (설정 토글, 기본 False)가 True 면 판정이 현재 모델과 다른 부분 LoRA 를 거부하지 않고
    순정 Forge 처럼 판정 레이아웃에서 변환한다 — 대응표·끼워 넣은 블록 정책은 완전한 레이아웃과 같고, 빠진 원본
    블록의 자리는 비워 둔다. 순정은 큰 판정 → 작은 모델을 거부하지만 여기서는 이 확장의 하향 표로 축소한다.
    판정 불가(인덱스 52 이상)·알 수 없는 타깃은 그대로 거부한다.
    """

    policy = normalize_duplicate_policy(duplicate_policy) or _DUPLICATE_POLICY
    source_indices = anima_lora_block_indices(lora)
    source_blocks = detect_anima_lora_layout(lora)
    inferred_blocks = infer_forge_lora_layout(source_indices)
    guessed = bool(
        sparse_forge_guess
        and source_indices
        and source_blocks is None
        and inferred_blocks is not None
        and inferred_blocks != target_blocks
        and target_blocks in SUPPORTED_BLOCK_COUNTS
    )
    # Forge performs this namespace migration for every ANIMA LoRA, regardless
    # of its DiT block layout.  Keep that independent normalization even when
    # sparse block indices cannot be remapped safely.
    moved = _move_llm_adapter_keys(lora)

    # Never guess sparse DiT block semantics unless the user opted in.  Only the
    # independent LLM key normalization above is allowed on this path.  The one
    # exception is a sparse LoRA whose Forge-rule layout already equals the
    # target: nothing needs converting, so it continues below exactly like a
    # same-layout LoRA (vanilla Forge loads it as-is; refusing it here silently
    # dropped the LoRA).
    if source_indices and source_blocks is None and inferred_blocks != target_blocks and not guessed:
        return BlockRemapResult(
            source_blocks=None,
            target_blocks=target_blocks,
            supported=False,
            remapped=False,
            moved_adapter_keys=moved,
            inferred_blocks=inferred_blocks,
        )
    if target_blocks not in SUPPORTED_BLOCK_COUNTS:
        return BlockRemapResult(
            source_blocks=source_blocks,
            target_blocks=target_blocks,
            supported=False,
            remapped=False,
            moved_adapter_keys=moved,
            inferred_blocks=inferred_blocks,
        )

    auxiliary_keys = _38b_only_keys(lora)
    # Never empty Forge's state dict.  load_lora_for_models() divides its
    # unmatched count by len(lora), so an empty connector-only LoRA would turn
    # a clean incompatibility report into ZeroDivisionError.  Keeping these
    # keys lets Forge reject the non-empty, wholly unmatched LoRA safely.
    if (
        target_blocks < ANIMA_38B_BLOCKS
        and not source_indices
        and auxiliary_keys
        and len(auxiliary_keys) == len(lora)
    ):
        return BlockRemapResult(
            source_blocks=None,
            target_blocks=target_blocks,
            supported=False,
            remapped=False,
            moved_adapter_keys=moved,
        )

    dropped_auxiliary = 0

    # Connector weights have no destination below 52 blocks.  At this
    # point the layout is either complete or contains another compatible
    # namespace, so removal cannot create the empty-dict Forge crash guarded
    # above.
    if target_blocks < ANIMA_38B_BLOCKS:
        dropped_auxiliary = _drop_38b_only_keys(lora)

    if not anima_lora_block_indices(lora):
        return BlockRemapResult(
            source_blocks=None,
            target_blocks=target_blocks,
            supported=True,
            remapped=False,
            moved_adapter_keys=moved,
            dropped_auxiliary_keys=dropped_auxiliary,
        )

    # 완전한 같은 레이아웃, 또는 순정 판정이 타깃과 같은 부분 LoRA(source_blocks None): 블록 키는 그대로.
    if not guessed and (source_blocks is None or source_blocks == target_blocks):
        return BlockRemapResult(
            source_blocks=source_blocks,
            target_blocks=target_blocks,
            supported=True,
            remapped=False,
            moved_adapter_keys=moved,
            dropped_auxiliary_keys=dropped_auxiliary,
            inferred_blocks=inferred_blocks,
        )

    # 추측 변환이면 판정 레이아웃을 원본으로 본다(guessed 가 True 면 inferred_blocks 는 None 이 아니다).
    layout = inferred_blocks if guessed else source_blocks
    assert layout is not None
    mapping = BLOCK_MAPPINGS[(layout, target_blocks)]
    weak_strength, weak_scope = current_weak_copy()
    dropped_dora, skipped, scaled = _remap_blocks_in_place(lora, mapping, policy, weak_strength, weak_scope)
    return BlockRemapResult(
        source_blocks=layout,
        target_blocks=target_blocks,
        supported=True,
        remapped=True,
        moved_adapter_keys=moved,
        dropped_auxiliary_keys=dropped_auxiliary,
        duplicate_policy=policy,
        dropped_dora_keys=dropped_dora,
        skipped_duplicate_keys=skipped,
        weak_strength=weak_strength if policy == DUPLICATE_WEAK else 0.0,
        weak_scope=weak_scope if policy == DUPLICATE_WEAK else "",
        weak_scaled_modules=scaled,
        inferred_blocks=inferred_blocks,
        guessed=guessed,
    )


_PATCH_OWNER = "sam-extra.anima-lora-blocks.v1"
_PATCHED_MODULE: ModuleType | None = None
_KNOWN_FORGE_MODULE: ModuleType | None = None
_ORIGINAL_PROCESS_ANIMA: Any = None


def _is_forge_lora_module(module: Any) -> bool:
    if not callable(getattr(module, "process_anima", None)):
        return False
    if not callable(getattr(module, "load_lora_for_models", None)):
        return False
    module_file = str(getattr(module, "__file__", "")).replace("\\", "/").lower()
    return not module_file or module_file.endswith("/sd_forge_lora/networks.py")


def _find_forge_lora_module() -> ModuleType | None:
    current = sys.modules.get("networks")
    if current is not None and _is_forge_lora_module(current):
        return current

    for module in tuple(sys.modules.values()):
        if module is not None and _is_forge_lora_module(module):
            return module

    # Anima Tile-Repair temporarily removes Forge's single-file ``networks``
    # module from sys.modules to import its vendored ``networks`` package.  A
    # UI reload during that window can still reattach to the live Forge module
    # through this direct reference; the context manager later restores the
    # very same object.
    if _KNOWN_FORGE_MODULE is not None and _is_forge_lora_module(
        _KNOWN_FORGE_MODULE
    ):
        return _KNOWN_FORGE_MODULE

    try:
        imported = importlib.import_module("networks")
    except Exception:
        return None
    return imported if _is_forge_lora_module(imported) else None


def _format_remap_message(result: BlockRemapResult) -> str:
    assert result.source_blocks is not None
    source_name = LAYOUT_NAMES[result.source_blocks]
    target_name = LAYOUT_NAMES[result.target_blocks]
    if result.duplicated_blocks:
        action = f"duplicating {result.duplicated_blocks} inserted lineage blocks"
        if result.duplicate_policy == DUPLICATE_WEAK:
            action = (
                f"weak copy x{result.weak_strength:g} ({result.weak_scope}) into "
                f"{result.duplicated_blocks} inserted blocks: {result.weak_scaled_modules} modules, "
                f"{result.skipped_duplicate_keys} keys out of scope, {result.dropped_dora_keys} DoRA magnitudes dropped"
            )
        elif result.skipped_duplicate_keys:
            action = (
                f"leaving {result.duplicated_blocks} inserted blocks without LoRA "
                f"({result.skipped_duplicate_keys} keys skipped)"
            )
        elif result.dropped_dora_keys:
            action += (
                f", additive on inserted blocks ({result.dropped_dora_keys} "
                "DoRA magnitudes dropped)"
            )
    else:
        action = f"dropping {result.dropped_blocks} source-only blocks"
    if result.guessed:
        # 추측 변환은 콘솔 경고 한 줄로 — 무엇을 추측했는지와 틀릴 수 있다는 것까지 같은 줄에 담는다.
        return (
            f"[sam-extra] Guessing sparse ANIMA LoRA layout like Forge ({OPT_SPARSE_FORGE_GUESS}): "
            f"treating it as {result.source_blocks}-Block and Re-Mapping to {result.target_blocks}-Block "
            f"({source_name} -> {target_name}; {action}); block correspondence may be wrong"
        )
    return (
        "[sam-extra] Re-Mapping "
        f"{result.source_blocks}-Block ANIMA LoRA to {result.target_blocks}-Block "
        f"({source_name} -> {target_name}; {action})"
    )


def _notify_ui(message: str, info: bool = False) -> None:
    """LoRA 를 건너뛸 때(gr.Warning) 또는 추측 변환했을 때(info=True, gr.Info) 사용자에게 알린다.

    Forge 의 LoRA 로딩은 Generate 이벤트 스레드 안에서 돌므로 토스트가 UI 에 뜬다. 이벤트 밖(테스트·API)
    에서는 gradio 가 파이썬 warnings 로 돌리고, gradio 가 없거나 실패하면 콘솔 로그(이미 남김)만으로 끝낸다.
    """

    try:
        import gradio as gr

        (gr.Info if info else gr.Warning)(message)
    except Exception:
        LOGGER.debug("[sam-extra] could not raise a UI warning", exc_info=True)


_LORA_LOADER_NAME = "load_lora_for_models"


def _caller_lora_filename() -> str:
    """호출 스택에서 순정 load_lora_for_models 의 filename 인자를 최선으로 읽는다. 없으면 빈 문자열.

    process_anima(lora, blocks) 시그니처에는 파일 이름이 없어 안내문에 어떤 LoRA 인지 적으려면 이 방법뿐이다.
    프레임 수에 기대지 않고(다른 확장의 래퍼가 몇 겹 끼어도 된다) 이름이 load_lora_for_models 이고 filename 이
    인자인 가장 가까운 프레임만 본다. 이름만 같은 다른 지역 변수로 엉뚱한 이름을 적지 않도록 추측은 하지 않는다.
    """

    frame = None
    try:
        frame = sys._getframe(1)
        while frame is not None:
            code = frame.f_code
            if code.co_name == _LORA_LOADER_NAME:
                arg_count = code.co_argcount + code.co_kwonlyargcount
                if "filename" in code.co_varnames[:arg_count]:
                    value = frame.f_locals.get("filename")
                    return value if isinstance(value, str) else ""
            frame = frame.f_back
    except Exception:
        pass
    finally:
        del frame  # 프레임 참조 순환을 남기지 않는다.
    return ""


def _describe_indices(indices: list[int]) -> str:
    if not indices:
        return "없음"
    if indices == list(range(indices[0], indices[-1] + 1)):
        return f"{indices[0]}~{indices[-1]}"
    return f"{indices[0]}~{indices[-1]} 중 {len(indices)}개"


def _layout_label(blocks: int | None) -> str:
    if blocks is None:
        return f"지원하지 않는 레이아웃(블록 인덱스 {ANIMA_38B_BLOCKS} 이상)"
    return f"{LAYOUT_NAMES.get(blocks, str(blocks) + ' blocks')}({blocks}블록)"


def _caller_lora_basename() -> str:
    return _caller_lora_filename().replace("\\", "/").rsplit("/", 1)[-1]


def _skip_notice(reason: str, basename: str | None = None) -> str:
    if basename is None:
        basename = _caller_lora_basename()
    name = f" '{basename}'" if basename else ""
    return f"Anima LoRA{name} 건너뜀: {reason}"


def _sparse_refusal_reason(indices: list[int], inferred: int | None, target_blocks: int) -> str:
    """부분 LoRA 거부 사유 — 접두 / 중간이 빔 / 판정 불가 세 갈래. 앞 두 갈래만 토글을 안내한다."""

    kind = sparse_lora_kind(set(indices))
    if kind == SPARSE_UNKNOWN:
        return (
            f"DiT 블록 인덱스 {indices[-1]} 이 있어(지원 레이아웃은 최대 {ANIMA_38B_BLOCKS}블록) 레이아웃을 "
            "판정할 수 없습니다. 순정 Forge 도 판정하지 못하는 경우라 추측 변환 설정으로도 로드하지 않습니다."
        )
    # 판정 레이아웃을 '쓸 모델'로 권하지 않는다: 앞 블록만 담은 접두 LoRA 는 2.9B·3.8B 에서 학습한
    # 것일 수도 있어 판정(가장 큰 인덱스 기준)이 원래 모델과 다를 수 있다.
    if kind == SPARSE_PREFIX:
        problem = (
            f"DiT 블록 {_describe_indices(indices)}(앞 {len(indices)}블록)만 담은 접두 LoRA 라 순정 Forge 규칙"
            f"(가장 큰 블록 인덱스 기준)으로는 {_layout_label(inferred)}으로 판정되지만, 더 큰 모델에서 앞 블록만 "
            f"학습했을 수도 있어 현재 모델 {_layout_label(target_blocks)}과의 블록 대응을 확정할 수 없습니다."
        )
    else:
        problem = (
            f"DiT 블록 {_describe_indices(indices)}만 있고 중간(또는 앞)이 빈 부분 LoRA 라 빠진 자리가 원래 어디였는지 "
            f"확정할 수 없습니다(순정 판정 {_layout_label(inferred)}, 현재 모델 {_layout_label(target_blocks)})."
        )
    return f"{problem} {SPARSE_GUESS_HINT}"


def install_forge_lora_block_hook(
    networks_module: ModuleType | None = None,
) -> bool:
    """Patch Forge's narrow ``process_anima`` seam; return True when installed."""

    global _PATCHED_MODULE, _KNOWN_FORGE_MODULE, _ORIGINAL_PROCESS_ANIMA

    module = networks_module or _find_forge_lora_module()
    if module is None or not _is_forge_lora_module(module):
        return False

    _KNOWN_FORGE_MODULE = module

    current = module.process_anima
    if getattr(current, "_sam3_anima_block_owner", None) == _PATCH_OWNER:
        _PATCHED_MODULE = module
        _ORIGINAL_PROCESS_ANIMA = getattr(
            current,
            "_sam3_anima_block_original",
            _ORIGINAL_PROCESS_ANIMA,
        )
        return False


    # If another extension wrapped our already-installed callable, installing
    # again above it would create a stale nested SAM3 hook that cannot be
    # cleanly removed on UI reload.  Keep the existing lower hook instead.
    if _PATCHED_MODULE is module and _ORIGINAL_PROCESS_ANIMA is not None:
        LOGGER.warning(
            "[sam-extra] ANIMA LoRA hook is already installed below another "
            "wrapper; skipping a duplicate installation"
        )
        return False

    original = current
    log = getattr(module, "logger", LOGGER)

    def process_anima_with_38b(lora, blocks):
        target_blocks = int(blocks)
        guess_enabled = sparse_forge_guess_enabled()
        source_indices = sorted(anima_lora_block_indices(lora))
        result = remap_anima_lora_for_model(lora, target_blocks, sparse_forge_guess=guess_enabled)
        complete = source_indices == list(range(len(source_indices))) and len(source_indices) in SUPPORTED_BLOCK_COUNTS
        basename = None
        # 토글에 따라 결과가 갈리는 부분 LoRA(판정이 현재 모델과 다름)만 기록한다 — infotext·토글 변경 감지용.
        if (
            source_indices
            and not complete
            and target_blocks in SUPPORTED_BLOCK_COUNTS
            and result.inferred_blocks is not None
            and result.inferred_blocks != target_blocks
        ):
            basename = _caller_lora_basename()
            _record_sparse_decision(basename or "?", result, guess_enabled)
        if result.supported:
            if result.guessed:
                log.warning(_format_remap_message(result))
                name = f" '{basename}'" if basename else ""
                _notify_ui(
                    f"Anima LoRA{name} 추측 변환: DiT 블록 "
                    f"{_describe_indices(source_indices)}만 담은 부분 LoRA 를 순정 Forge 규칙대로 "
                    f"{_layout_label(result.source_blocks)}으로 보고 {_layout_label(target_blocks)}로 "
                    "변환했습니다 — 블록 대응이 틀릴 수 있습니다.",
                    info=True,
                )
            elif result.remapped:
                log.warning(_format_remap_message(result))
            elif result.source_blocks is None and result.inferred_blocks is not None:
                indices = sorted(anima_lora_block_indices(lora))
                log.info(
                    "[sam-extra] Loading sparse ANIMA LoRA as-is: %d of %d "
                    "blocks (indices %s) already match %s",
                    len(indices),
                    target_blocks,
                    _describe_indices(indices),
                    LAYOUT_NAMES.get(target_blocks, f"{target_blocks} blocks"),
                )
            if result.dropped_auxiliary_keys:
                log.warning(
                    "[sam-extra] Dropped %d ANIMA 3.8B-only connector "
                    "LoRA keys while projecting to %s; this direction is lossy",
                    result.dropped_auxiliary_keys,
                    LAYOUT_NAMES.get(target_blocks, f"{target_blocks} blocks"),
                )
            return True
        if target_blocks in SUPPORTED_BLOCK_COUNTS:
            indices = sorted(anima_lora_block_indices(lora))
            if not indices and _38b_only_keys(lora):
                log.warning(
                    "[sam-extra] ANIMA 3.8B-only connector LoRA "
                    "cannot be projected to %s; keys were left intact so "
                    "Forge can reject it safely",
                    LAYOUT_NAMES[target_blocks],
                )
                _notify_ui(_skip_notice(
                    "3.8B 전용 Semantic Connector 키만 있어 "
                    f"{_layout_label(target_blocks)} 에는 넣을 곳이 없습니다."
                ))
            else:
                log.warning(
                    "[sam-extra] Refusing to guess a sparse/unsupported ANIMA "
                    "LoRA layout with indices %s for %s (Forge's own rule would "
                    "call it %s); DiT block keys were left unchanged",
                    indices,
                    LAYOUT_NAMES[target_blocks],
                    _layout_label(result.inferred_blocks),
                )
                _notify_ui(_skip_notice(
                    _sparse_refusal_reason(indices, result.inferred_blocks, target_blocks),
                    basename,
                ))
            # This adapter owns all known targets.  Delegating a sparse layout
            # would let Forge infer a generation from only a contiguous prefix.
            return False
        # Let a future Forge teach us a newer layout, but contain implementations
        # that may raise after partially writing remapped keys.
        before_fallback = dict(lora)
        try:
            return original(lora, blocks)
        except IndexError:
            lora.clear()
            lora.update(before_fallback)
            log.warning(
                "[sam-extra] Forge could not handle unknown ANIMA target "
                "%s; its partial LoRA changes were rolled back",
                target_blocks,
            )
            _notify_ui(_skip_notice(
                f"Forge 가 {target_blocks}블록 모델로 변환하지 못했습니다(변경 사항은 되돌림)."
            ))
            return False

    process_anima_with_38b._sam3_anima_block_owner = _PATCH_OWNER
    process_anima_with_38b._sam3_anima_block_original = original
    module.process_anima = process_anima_with_38b
    _PATCHED_MODULE = module
    _ORIGINAL_PROCESS_ANIMA = original
    return True


def uninstall_forge_lora_block_hook() -> bool:
    """Restore Forge's original callable when this extension owns the patch."""

    global _PATCHED_MODULE, _ORIGINAL_PROCESS_ANIMA

    module = _PATCHED_MODULE
    current = getattr(module, "process_anima", None) if module is not None else None
    restored = bool(
        module is not None
        and _ORIGINAL_PROCESS_ANIMA is not None
        and getattr(current, "_sam3_anima_block_owner", None) == _PATCH_OWNER
    )
    if restored:
        module.process_anima = _ORIGINAL_PROCESS_ANIMA
    _PATCHED_MODULE = None
    _ORIGINAL_PROCESS_ANIMA = None
    return restored


def forge_hook_installed() -> bool:
    """이 확장의 블록 변환이 Forge 에 걸려 있는지(끼워 넣은 블록 정책이 적용될 수 있는지)."""

    return _PATCHED_MODULE is not None and _ORIGINAL_PROCESS_ANIMA is not None


__all__ = [
    "ANIMA_29B_BLOCKS",
    "ANIMA_38B_BLOCKS",
    "ANIMA_BASE_BLOCKS",
    "BLOCK_MAPPINGS",
    "BlockRemapResult",
    "DUPLICATE_ADDITIVE",
    "DUPLICATE_KEEP",
    "DUPLICATE_POLICIES",
    "DUPLICATE_SKIP",
    "DUPLICATE_WEAK",
    "DEFAULT_WEAK_STRENGTH",
    "INFOTEXT_SPARSE_GUESS_KEY",
    "OPT_SPARSE_FORGE_GUESS",
    "SPARSE_GAP",
    "SPARSE_GUESS_HINT",
    "SPARSE_GUESS_LABEL",
    "SPARSE_PREFIX",
    "SPARSE_STATE_ATTR",
    "SPARSE_UNKNOWN",
    "SUPPORTED_BLOCK_COUNTS",
    "WEAK_SCOPES",
    "WEAK_SCOPE_ALL",
    "WEAK_SCOPE_ATTN",
    "WEAK_SCOPE_ATTN_MLP",
    "anima_lora_block_indices",
    "current_duplicate_policy",
    "current_weak_copy",
    "detect_anima_lora_layout",
    "forge_hook_installed",
    "forget_sparse_state",
    "infer_forge_lora_layout",
    "install_forge_lora_block_hook",
    "normalize_duplicate_policy",
    "normalize_weak_scope",
    "parse_weak_key",
    "policy_state_key",
    "remap_anima_lora_for_model",
    "set_duplicate_policy",
    "set_weak_copy",
    "sparse_cache_stale",
    "sparse_forge_guess_enabled",
    "sparse_guess_record",
    "sparse_lora_kind",
    "uninstall_forge_lora_block_hook",
]
