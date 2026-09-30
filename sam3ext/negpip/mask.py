# Copyright (C) 2025 hako-mikan
# Copyright (C) 2026 Haoming02
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU Affero General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version. This program is distributed in the hope
# that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License
# for more details. You should have received a copy of the GNU Affero General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>. Full text: sam3ext/negpip/LICENSE
#
# NegPiP — vendored into sam-extra (forge_sam3_extension). Upstream: https://github.com/Haoming02/sd-forge-negpip
# @ 0585496 (archived 2026-09-30); this file is _build_negpip_mask from lib_negpip/anima.py.
# MODIFIED by sam-extra, 2026-09-30 (AGPL-3.0 section 5a): moved into its own Forge-free module; supports both
# Forge Anima text engines (old AnimaTextProcessingEngine path from upstream b3673ce, new Qwen06Engine path
# from 0585496); new engine follows the engine's emphasis mode (None/Ignore -> no negative rows).
"""NegPiP 의 Anima 행 마스크 — Forge 0.6B TE 엔진 두 세대 공용. Forge 를 import 하지 않는다.

Anima 조건 행 i 는 엔진이 T5 토큰 i 에 가중치를 곱한 결과다. 마스크는 **엔진이 실제로 곱한 가중치** 가 음수인 행만 -1 이다.
NegPiP 는 그 행에 -1 을 한 번 더 곱해 K 의 부호를 되돌리고 V 에만 음수를 남긴다. 그래서 행 정렬과 부호가 모두 그 엔진이 조건을
만든 토큰화와 같아야 한다.

- 옛 엔진 ``AnimaTextProcessingEngine`` (Forge ad88b6b4 까지): ``tokenize_line`` 의 ``t5_multipliers`` — 상류 b3673ce 와 같다.
  이 엔진은 ``Ignore`` 에서도 가중치를 곱하고(NegPiP 동작), ``None`` 이면 괄호를 글자로 읽어 가중치가 모두 1 이다(마스크도 1).
- 새 엔진 ``Qwen06Engine`` (Forge 21886f41~): ``t5_tokenizer.tokenize_with_weights`` — 상류 0585496 과 같다. 다만 이 엔진은
  ``None`` 이면 가중치를 파싱하지 않고(괄호·``:-1`` 이 토큰으로 남아 행 수부터 다르다), ``Ignore`` 면 괄호는 먹되 T5 가중치를 1.0 으로
  둔다 — 어느 쪽도 행을 뒤집지 않으므로 마스크는 모두 1 이다. 상류 0585496 은 여기서도 가중치를 파싱해 ``None`` 이면 행이 어긋나고,
  ``Ignore`` 면 K 만 뒤집혀(V 는 양수) NegPiP 와 반대 뜻이 된다.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# 새 엔진에서 음수 가중치가 조건에 곱해지지 않는 emphasis (Qwen06Engine.__call__ 의 n·i)
NEW_ENGINE_NO_WEIGHTS = ("None", "Ignore")
# 옛 엔진에서 그런 emphasis (parse_prompt_attention 이 None 만 글자 그대로 읽는다)
OLD_ENGINE_NO_WEIGHTS = ("None",)


def is_legacy_engine(engine) -> bool:
    """옛 엔진(tokenize_line 있음)이면 True — sam3ext.anima38.native_engine.is_legacy_engine 과 같은 판별."""
    return hasattr(engine, "tokenize_line")


def negpip_effective(engine, emphasis_name: str) -> bool:
    """이 엔진이 emphasis_name 방식에서 음수 가중치를 조건에 곱하는가 — 아니면 Anima NegPiP 는 뜻이 없다."""
    blocked = OLD_ENGINE_NO_WEIGHTS if is_legacy_engine(engine) else NEW_ENGINE_NO_WEIGHTS
    return emphasis_name not in blocked


def engine_t5_multipliers(engine, line: str) -> list[float]:
    """엔진이 이 줄의 조건 행(T5 토큰, 끝 토큰 포함)에 곱하는 가중치. 가중치를 쓰지 않는 방식이면 빈 목록."""
    if is_legacy_engine(engine):
        # 옛 엔진 — 상류 b3673ce 그대로. tokenize_line 은 엔진이 쥔 emphasis 로 파싱한다(__call__ 이 방금 opts 로 맞춘 값).
        multipliers = []
        for chunk in engine.tokenize_line(line):
            multipliers.extend(getattr(chunk, "t5_multipliers", []))
        return multipliers
    # 새 엔진 — 속성이 opts.emphasis 를 매번 읽는다. __call__ 과 같이 한 번 읽어 판단한다.
    if engine.emphasis.name in NEW_ENGINE_NO_WEIGHTS:
        return []
    # __call__ 의 t5_chunk 와 같은 호출(disable_weights=n=False), 첫 조각만 (max_length=INF 라 조각은 하나)
    chunk = engine.t5_tokenizer.tokenize_with_weights(line, disable_weights=False)
    return list(map(lambda x: x[1], chunk[0]))


def build_negpip_mask(
    text_processing_engine,
    line: str,
    token_length: int,
    device: torch.device,
    dtype: torch.dtype,
):
    multipliers = engine_t5_multipliers(text_processing_engine, line)

    if len(multipliers) == 0:
        return torch.ones(token_length, device=device, dtype=dtype)

    weights = torch.tensor(multipliers, device=device, dtype=dtype)
    ones = torch.ones_like(weights)
    mask = torch.where(weights < 0, -ones, ones)

    if mask.shape[0] < token_length:
        mask = F.pad(mask, (0, token_length - mask.shape[0]), value=1.0)
    elif mask.shape[0] > token_length:
        mask = mask[:token_length]

    return mask
