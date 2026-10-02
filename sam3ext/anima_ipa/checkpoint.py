"""어댑터 체크포인트의 구조를 키·shape·메타데이터만으로 알아낸다.

버전이 다른 어댑터를 코드 수정 없이 받아내기 위한 것이다. torch 를 쓰지 않으므로 가짜 키
목록만으로 전부 테스트할 수 있다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

SIGLIP_HIDDEN_SIZE = 768
DEFAULT_BLOCKS = 28

UNSUPPORTED_INJECT_BEFORE_MLP = "블록 MLP 앞 주입(ip_inject_before_mlp)"
UNSUPPORTED_INDEPENDENT_Q = "독립 IP Q 투영(shared_ip_q_proj)"

_BLOCK_RE = re.compile(r"blocks\.(\d+)\.(ip_k_proj|ip_v_proj|adaln_ip)")
_LORA_RE = re.compile(
    r"lora\.base_model\.model\.blocks\.(\d+)\.cross_attn\.(\w+)"
    r"\.lora_([AB])(?:\.default)?\.weight"
)


@dataclass(frozen=True)
class AdapterSpec:
    """어댑터 한 개의 구조. ``supported`` 가 False 면 설치하지 않는다."""

    num_blocks: int
    embed_dim: int            # SigLIP 토큰 차원(보통 768)
    inner_dim: int            # DiT x_dim
    shared_projection: bool
    compressor: tuple[int, int] | None      # (쿼리 수, 레이어 수)
    has_self_attn: bool
    has_siglip_norm: bool
    has_null_tokens: bool
    lora_blocks: tuple[int, ...]
    lora_rank: int
    norm_keys: bool
    unsupported: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return not self.unsupported


def _truthy(value: object) -> bool:
    return str(value).strip().lower() == "true"


def inspect_adapter(
    keys: Iterable[str],
    shapes: Mapping[str, tuple[int, ...]],
    metadata: Mapping[str, str],
) -> AdapterSpec:
    keys = list(keys)
    key_set = set(keys)

    block_indices = {int(m.group(1)) for m in map(_BLOCK_RE.match, keys) if m}
    num_blocks = max(block_indices) + 1 if block_indices else DEFAULT_BLOCKS

    shared = any(key.startswith("shared_ip_k_proj") for key in keys)
    if shared:
        probe = shapes.get("shared_ip_k_proj.expand.weight") or shapes.get(
            "shared_ip_k_proj.0.weight"
        )
    else:
        probe = next(
            (
                shapes[f"blocks.{i}.ip_k_proj.weight"]
                for i in range(num_blocks)
                if f"blocks.{i}.ip_k_proj.weight" in shapes
            ),
            None,
        )
    if probe:
        inner_dim, embed_dim = probe[0], probe[1]
    else:
        inner_dim = embed_dim = SIGLIP_HIDDEN_SIZE

    compressor = None
    queries = shapes.get("siglip_compressor.queries")
    if queries is not None:
        layers = max(
            (
                int(key.split(".")[2]) + 1
                for key in keys
                if key.startswith("siglip_compressor.layers.")
            ),
            default=2,
        )
        compressor = (int(queries[0]), int(layers))

    lora: dict[int, set[str]] = {}
    lora_rank = 0
    for key in keys:
        if not key.startswith("lora."):
            continue
        match = _LORA_RE.match(key)
        if match is None:
            continue
        lora.setdefault(int(match.group(1)), set()).add(match.group(2))
        if match.group(3) == "A":
            shape = shapes.get(key)
            if shape:
                lora_rank = max(lora_rank, int(shape[0]))

    unsupported: list[str] = []
    if _truthy(metadata.get("ip_inject_before_mlp", "False")):
        unsupported.append(UNSUPPORTED_INJECT_BEFORE_MLP)
    if any(key.startswith("shared_ip_q_proj") for key in keys):
        unsupported.append(UNSUPPORTED_INDEPENDENT_Q)

    return AdapterSpec(
        num_blocks=num_blocks,
        embed_dim=int(embed_dim),
        inner_dim=int(inner_dim),
        shared_projection=shared,
        compressor=compressor,
        has_self_attn=any(key.startswith("ip_self_attn.") for key in keys),
        has_siglip_norm=any(key.startswith("siglip_norm.") for key in keys),
        has_null_tokens="null_tokens" in key_set,
        lora_blocks=tuple(sorted(lora)),
        lora_rank=lora_rank,
        norm_keys=_truthy(metadata.get("ip_norm_keys", "False")),
        unsupported=tuple(unsupported),
    )
