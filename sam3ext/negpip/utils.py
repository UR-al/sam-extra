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
# @ 0585496 (archived 2026-09-30), lib_negpip/utils.py.
# MODIFIED by sam-extra, 2026-09-30 (AGPL-3.0 section 5a): import paths (lib_negpip -> sam3ext.negpip); added
# clip_fragment_specials / clip_word_rows (SD1/SDXL rows of the negative-term condition on Forge's new ClipEngine,
# used by NegPiP._cond_dealer in scripts/negpip.py). The upstream functions below are unchanged.

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

from sam3ext.negpip import IS_NEO

NEG_PATTERN = re.compile(r"\(\s*(?:[^\\(:)]|\\[\(\)])+?\s*\:\s*-\s*\d*\.?\d+\s*\)")


def reset_prompt_cache(p: "StableDiffusionProcessing"):
    c = 3 if IS_NEO else 2

    p.cached_c = [None] * c
    p.cached_uc = [None] * c
    if hasattr(p, "cached_hr_c"):
        p.cached_hr_c = [None] * c
        p.cached_hr_uc = [None] * c


def hr_dealer(p: "StableDiffusionProcessing") -> tuple[bool, bool]:
    return (
        bool(getattr(p, "hr_prompts", None)),
        bool(getattr(p, "hr_negative_prompts", None)),
    )


def has_negative(prompt: str) -> bool:
    return bool(re.search(NEG_PATTERN, prompt))


def have_negative(prompts: list[str]) -> bool:
    return any(has_negative(p) for p in prompts)


def any_negative(p: "StableDiffusionProcessing") -> bool:
    return any(
        [
            have_negative(p.prompts),
            have_negative(p.negative_prompts),
            have_negative(getattr(p, "hr_prompts", None) or ""),
            have_negative(getattr(p, "hr_negative_prompts", None) or ""),
        ]
    )


# ---- sam-extra: SD1/SDXL 음수 항 조건의 행 (Forge 새 ClipEngine) ----
# 상류 _cond_dealer 는 "(단어:w)" 조건에서 cond[1 : token_len + 2] 를 잘라 [단어…, EOS] 행을 얻는다. 옛 엔진(classic_engine,
# Forge ad88b6b4 까지)의 tokenize 는 add_special_tokens=False 라 청크가 [BOS, 단어…, EOS…] 이고 token_len 은 단어 수다.
# 새 엔진(sd_engine.ClipEngine, 21886f41~)의 tokenize 는 그 인자를 넘기지 않아 프롬프트 조각마다 [BOS, 단어…, EOS] 가 붙는다 —
# 청크가 [BOS, BOS, 단어…, EOS, EOS…], token_len 이 BOS·EOS 까지 세어 같은 자르기가 [BOS, 단어…, EOS, EOS] 가 되고, NegPiP 가
# V 를 뒤집는 행에 BOS(어텐션 싱크) 행이 들어간다. 판별은 판 번호가 아니라 엔진 자신으로 한다(빈 글자를 토큰화해 특수 토큰이
# 나오는가) — Forge 가 고치면 저절로 옛 자르기로 돌아간다.


def clip_fragment_specials(engine) -> frozenset[int] | None:
    """engine.tokenize 가 조각마다 특수 토큰을 붙이면 그 토큰 id 들(엔진의 시작·끝 토큰 포함), 아니면 None.

    옛 엔진·A1111 식 엔진(빈 글자 → 토큰 없음), tokenize 가 없거나 실패하는 엔진은 None — 상류 자르기를 그대로 쓴다.
    """
    tokenize = getattr(engine, "tokenize", None)
    if tokenize is None:
        return None
    try:
        ids = list(tokenize([""])[0])
    except Exception:
        return None
    if not ids:
        return None
    specials = set(ids)
    for name in ("id_start", "id_end"):
        value = getattr(engine, name, None)
        if isinstance(value, int):
            specials.add(value)
    return frozenset(specials)


def clip_word_rows(chunks, specials: frozenset[int]) -> list[int]:
    """tokenize_line 청크에서 옛 엔진 자르기와 같은 뜻의 조건 행 번호: 단어 토큰 행 + 마지막 단어 바로 뒤 끝 토큰 행.

    청크는 조건에서 len(chunk.tokens)(77) 행씩 이어진다. 특수 토큰(시작·끝·조각마다 붙은 것)이 아닌 토큰 — 단어와 임베딩 자리
    — 의 행을 모으고, 옛 엔진이 [단어…, EOS] 로 끝 토큰 하나를 붙이던 것처럼 마지막 단어 다음 행을 붙인다. 단어가 없으면(공백뿐)
    옛 엔진의 cond[1:2] 처럼 앞쪽 시작 토큰들 바로 다음 행 하나다.
    """
    rows: list[int] = []
    offset = 0
    for chunk in chunks:
        tokens = list(chunk.tokens)
        rows.extend(offset + j for j, token in enumerate(tokens) if token not in specials)
        offset += len(tokens)
    if rows:
        return rows + [rows[-1] + 1]
    tokens = list(chunks[0].tokens) if chunks else []
    lead = 0
    while lead < len(tokens) and tokens[lead] == tokens[0]:
        lead += 1
    return [max(lead, 1)]
