"""v2 run id 를 조건 텐서 안에 새기는 마커.

Semantic Connector v2 는 프롬프트를 인코딩할 때 '실행(run)'을 등록해 두고, 디노이징 매 스텝의
forward 에서 그 run 의 의미 특징으로 컨텍스트를 다시 만든다. 원본은 run id 를 ``{"crossattn", "vector"}``
dict 의 ``vector``(→ ``y``) 로 실어 보냈는데, 그 dict 는 Forge 의 네이티브 계약(줄마다 텐서 하나인 list)과
달라 ``get_learned_conditioning`` 을 감싸는 다른 확장(NegPiP 등)이 ``assert isinstance(conds, list)`` 로
죽었다. 그래서 run id 를 각 줄 텐서의 **마지막 토큰 행**에 새긴다.

- 값은 {0, 1, 512, 1024} 뿐이라 fp16/bf16 캐스트에도 정확히 보존된다.
- 부호만 비교에서 버리므로(NegPiP 는 토큰 행에 -1 을 곱한다) 부호 반전에도 살아남는다.
- 마커가 없는 행(네거티브 프롬프트 등 순정 경로)은 -1 로 읽혀 그대로 둔다.
"""

from __future__ import annotations

import torch

MAGIC = (1024.0, 512.0)
BITS = 20          # run id < 1,048,576
WIDTH = 2 + BITS   # 필요한 채널 수


def stamp_run_id(placeholder: torch.Tensor, run_id: int) -> torch.Tensor:
    """``[1, L, C]`` 자리표시 텐서의 마지막 행을 마커로 바꾼 사본을 돌려준다."""
    if placeholder.ndim != 3 or placeholder.shape[-1] < WIDTH:
        raise ValueError("placeholder must be [1, L, C] with C >= %d" % WIDTH)
    if not 0 <= run_id < (1 << BITS):
        raise ValueError("run id out of marker range: %r" % run_id)
    row = torch.zeros(placeholder.shape[-1], dtype=placeholder.dtype, device=placeholder.device)
    row[0] = MAGIC[0]
    row[1] = MAGIC[1]
    for bit in range(BITS):
        row[2 + bit] = float((run_id >> bit) & 1)
    stamped = placeholder.clone()
    stamped[:, -1] = row
    return stamped


def read_run_ids(context: torch.Tensor) -> torch.Tensor:
    """``[B, L, C]`` 컨텍스트에서 행별 run id 를 읽는다. 마커가 없으면 -1."""
    batch = context.shape[0]
    none = torch.full((batch,), -1, dtype=torch.long, device=context.device)
    if context.ndim != 3 or context.shape[-1] < WIDTH:
        return none
    rows = context[:, -1, :WIDTH].detach().abs().to(torch.float32)
    is_marker = (rows[:, 0] == MAGIC[0]) & (rows[:, 1] == MAGIC[1])
    bits = rows[:, 2:].round().clamp(0.0, 1.0).to(torch.long)
    weights = (1 << torch.arange(BITS, device=context.device, dtype=torch.long))
    ids = (bits * weights).sum(dim=-1)
    return torch.where(is_marker, ids, none)


def as_rows(context: torch.Tensor) -> torch.Tensor:
    """Forge 는 줄마다 ``[1, L, C]`` 를 stack 해 ``[B, 1, L, C]`` 로 넘기기도 한다 — ``[B, L, C]`` 뷰로 편다."""
    if context.ndim == 3:
        return context
    return context.reshape(context.shape[0], -1, context.shape[-1])
