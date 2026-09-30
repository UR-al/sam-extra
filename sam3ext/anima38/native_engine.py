"""Forge 0.6B TE 엔진 두 세대에서 3.8B 커넥터가 쓰는 원본 입력(qwen 은닉 상태·T5 id·T5 가중치)을 뽑는다.

- 옛 엔진 ``AnimaTextProcessingEngine``: ``tokenize_line``·``process_tokens`` (runtime._native_inputs 의 옛 경로).
- 새 엔진 ``Qwen06Engine`` (Forge 21886f41 "Rewrite TextProcessingEngine", ComfyUI v0.36 sd1_clip 이식):
  토크나이저 두 개의 ``tokenize_with_weights`` 와 ``text_encoder.encode_token_weights`` — 여기서 그 한 줄 계산을 따라 한다.
"""
from __future__ import annotations

import torch

ONE_CHUNK_ERROR = "Anima 3.8B expects one prompt chunk."


def is_legacy_engine(engine) -> bool:
    """옛 엔진(tokenize_line 있음)이면 True."""
    return hasattr(engine, "tokenize_line")


def qwen06_native_inputs(engine, line: str, device, dtype):
    """Qwen06Engine.__call__ 의 한 줄 분(_preprocess 앞까지)을 그대로 — Forge 21886f41 기준.

    돌려주는 모양·장치·dtype 은 옛 경로와 같다: source [1,T,D] (dtype), target_ids [1,L] long (끝 토큰 포함),
    target_weights [1,L,1] (dtype). 가중치는 옛 경로처럼 파이썬 float 에서 바로 dtype 으로 만든다.
    """
    name = engine.emphasis.name   # 새 엔진은 속성이 opts.emphasis 를 매번 읽는다 — 한 번만 읽어 둘이 어긋나지 않게
    n = name == "None"
    i = name == "Ignore"

    qwen_chunk = engine.qwen_tokenizer.tokenize_with_weights(line, disable_weights=n)
    t5_chunk = engine.t5_tokenizer.tokenize_with_weights(line, disable_weights=n)
    # 새 토크나이저는 max_length=INF 라 사실상 한 조각 — 옛 경로와 같은 오류로 막아 둔다(TE 를 돌리기 전에)
    if len(qwen_chunk) != 1 or len(t5_chunk) != 1:
        raise RuntimeError(ONE_CHUNK_ERROR)

    qwen_chunk = [[(x[0], 1.0) for x in inner] for inner in qwen_chunk]
    cond = engine.text_encoder.encode_token_weights(qwen_chunk)[0]

    target_ids = torch.tensor([x[0] for x in t5_chunk[0]], device=device, dtype=torch.long).unsqueeze(0)
    target_weights = torch.tensor(
        [1.0 if i else x[1] for x in t5_chunk[0]],
        device=device,
        dtype=dtype,
    ).reshape(1, -1, 1)
    return cond.to(device=device, dtype=dtype), target_ids, target_weights
