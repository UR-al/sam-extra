"""Forge 0.6B TE 엔진 두 세대에서 3.8B 커넥터가 쓰는 원본 입력(qwen 은닉 상태·T5 id·T5 가중치)을 뽑는다.

- 옛 엔진 ``AnimaTextProcessingEngine``: ``tokenize_line``·``process_tokens`` (runtime._native_inputs 의 옛 경로).
- 새 엔진 ``Qwen06Engine`` (Forge 21886f41 "Rewrite TextProcessingEngine", ComfyUI v0.36 sd1_clip 이식):
  토크나이저 두 개의 ``tokenize_with_weights`` 와 ``text_encoder.encode_token_weights`` — 여기서 그 한 줄 계산을 따라 한다.

엔진이 달린 모델 속성 이름도 Forge 판마다 다르다(``anima_text_engine``):

- 2.29.1(ceff5168)까지: ``sd_model.text_processing_engine_anima``.
- 2.29.2(46365871)부터: ``sd_model.text_processing_engine_qwen`` — 같은 이름을 Flux2(Klein)·Krea2·Qwen-Image·Z-Image 엔진도
  쓴다(Krea2·Qwen-Image 는 그 전부터). 이름만으로는 Anima 인지 알 수 없어, T5 토크나이저(``t5_tokenizer``)를 함께 가진 엔진만
  Anima 로 본다 — 옛·새 Anima 엔진 둘 다 갖고, 다른 모델의 엔진은 갖지 않는다.
"""
from __future__ import annotations

import torch

ONE_CHUNK_ERROR = "Anima 3.8B expects one prompt chunk."

LEGACY_ENGINE_ATTR = "text_processing_engine_anima"   # Forge ~2.29.1 — Anima 만 쓰던 이름
SHARED_ENGINE_ATTR = "text_processing_engine_qwen"    # Forge 2.29.2~ — Anima·Flux2·Krea2·Qwen-Image·Z-Image 공용
ANIMA_ENGINE_ATTRS = (SHARED_ENGINE_ATTR, LEGACY_ENGINE_ATTR)


def is_anima_text_engine(engine) -> bool:
    """Anima 0.6B 텍스트 엔진(옛 AnimaTextProcessingEngine·새 Qwen06Engine)인가 — 둘만 T5 토크나이저를 쥔다."""
    return engine is not None and hasattr(engine, "t5_tokenizer")


def anima_text_engine(model):
    """모델의 Anima 텍스트 엔진. Anima 가 아니거나 엔진이 없으면 None.

    옛 이름은 Anima 에만 있으므로 그대로 믿고, 공용 이름은 Anima 엔진일 때만 돌려준다(Z-Image·Flux2 등을 Anima 로
    잘못 알아보지 않게)."""
    if model is None:
        return None
    engine = getattr(model, LEGACY_ENGINE_ATTR, None)
    if engine is not None:
        return engine
    engine = getattr(model, SHARED_ENGINE_ATTR, None)
    return engine if is_anima_text_engine(engine) else None


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


def emphasis_infotext(engine, lines, uses_emphasis) -> str | None:
    """이 엔진의 __call__ 이 'Emphasis' 생성 정보에 쓸 값 — 쓰지 않으면 None. uses_emphasis 는 Forge 의 함수를 받는다.

    - 옛 엔진(Forge ad88b6b4 까지): 어느 줄이든 emphasis 를 쓰면 지금 방식 이름을 쓴다.
    - 새 Qwen06Engine(21886f41~): 어느 줄이든 emphasis 를 쓰고 방식이 None/Ignore 일 때만 쓴다.
    """
    if not any(uses_emphasis(line) for line in lines):
        return None
    name = engine.emphasis.name
    if is_legacy_engine(engine) or name in ("None", "Ignore"):
        return name
    return None
