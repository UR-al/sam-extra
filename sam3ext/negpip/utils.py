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
# used by NegPiP._cond_dealer in scripts/negpip.py); clip_row_specials / encoded_clip_rows / ClipRowMismatch (the same row
# rule on the old engine too, rows taken from the chunks of the exact "(text:-w)" string that was encoded, row-count check)
# and snapshot_prompts / restore_prompts (NegPiP stands down with the prompts it had edited put back); join_term_rows /
# active_rows / context_rows (SD1/SDXL: every negative term of a prompt, per batch item, same schedule rule on both sides,
# used by scripts/negpip.py and sd.py). The upstream functions below are unchanged.

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
# 나오는가) — Forge 가 고치면 저절로 옛 엔진과 같은 특수 토큰(id_start·id_end)으로 돌아간다(clip_row_specials).
# 행은 인코딩한 바로 그 글자 "(단어:-w)" 의 청크에서 센다. 상류는 맨 글자(가중치 1)를 토큰화해 셌는데, Forge 파서는 묶음 가중치를
# BREAK 표시(-1)에도 곱하고 엔진은 가중치가 정확히 -1 인 BREAK 에서만 청크를 나눈다 — "(cat BREAK dog:2)" 는 한 청크, 맨 글자
# "cat BREAK dog" 는 두 청크라 행 번호가 조건과 어긋났다(새 엔진 IndexError, 옛 엔진 채움 EOS·BOS 행).


def clip_fragment_specials(engine) -> frozenset[int] | None:
    """engine.tokenize 가 조각마다 특수 토큰을 붙이면 그 토큰 id 들(엔진의 시작·끝 토큰 포함), 아니면 None.

    옛 엔진·A1111 식 엔진(빈 글자 → 토큰 없음), tokenize 가 없거나 실패하는 엔진은 None — 그때 행 규칙은 엔진의
    id_start·id_end 만 건너뛴다(clip_row_specials).
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


def clip_row_specials(engine) -> frozenset[int] | None:
    """_cond_dealer 가 행을 고를 때 건너뛸 특수 토큰 id — 옛·새 엔진 공용. 모르면 None(상류 자르기 cond[1:token_len+2]).

    새 엔진은 조각마다 붙는 특수 토큰(clip_fragment_specials), 옛 엔진·A1111 식 엔진은 청크를 감싸는 id_start·id_end 다.
    옛 엔진 한 청크 항에서 clip_word_rows 는 상류 자르기와 같은 행이고, 여러 청크(BREAK·75 토큰 초과)에서는 상류 자르기가
    잡던 청크 경계의 채움 EOS·다음 청크 BOS 대신 단어 행만 고른다 — 두 엔진이 같은 행을 고른다.
    """
    specials = clip_fragment_specials(engine)
    if specials is not None:
        return specials
    ids = [getattr(engine, name, None) for name in ("id_start", "id_end")]
    if all(isinstance(value, int) for value in ids):
        return frozenset(ids)
    return None


class ClipRowMismatch(ValueError):
    """토큰화한 청크의 행 수가 인코딩한 조건의 행 수와 다르다 — 행 번호가 조건에 맞지 않는다."""


def encoded_clip_rows(chunks, specials: frozenset[int], row_count: int) -> list[int]:
    """인코딩한 글자의 청크에서 고른 조건 행(clip_word_rows). 청크가 조건과 행 수부터 다르면 ClipRowMismatch.

    방어용 검사다 — 행은 인코딩한 바로 그 글자를 토큰화해 고르므로 보통은 맞는다. 틀린 행을 조용히 잡거나 IndexError 를
    내는 대신 NegPiP 가 이 생성에서 물러나게 한다(scripts/negpip.py process_batch).
    """
    total = sum(len(chunk.tokens) for chunk in chunks)
    if total != row_count:
        raise ClipRowMismatch(f"tokenized {total} rows, encoded {row_count} rows")
    return clip_word_rows(chunks, specials)


def snapshot_prompts(p: "StableDiffusionProcessing") -> list[tuple[list[str], list[str]]]:
    """_getScheduledNegPip 이 음수 항을 지우기 전의 프롬프트 목록들 — 물러날 때 restore_prompts 로 되돌린다."""
    names = ("prompts", "negative_prompts", "hr_prompts", "hr_negative_prompts")
    lists = [getattr(p, name, None) for name in names]
    return [(prompts, list(prompts)) for prompts in lists if isinstance(prompts, list)]


def restore_prompts(snapshot: list[tuple[list[str], list[str]]]) -> None:
    for prompts, saved in snapshot:
        prompts[:] = saved


# ---- sam-extra: SD1/SDXL 음수 항 조건을 항목(배치 안 프롬프트)마다, 항 전부 ----
# 상류 0585496 의 _calc_conds 는 항마다 텐서 하나를 만들고(hako-mikan 원본은 한 영역의 항들을 torch.cat 한 텐서 하나), 훅은 그 목록의
# [0] 만 썼다 — 두 번째 항부터는 프롬프트에서 지워지고 어텐션에는 들어가지 않았다. 또 배치 항목 0 의 스케줄만 골라 batch_size 로
# 반복했다. 편입본은 항목·스케줄 줄마다 그 줄의 항들을 이어 붙인 행 하나(join_term_rows)를 만들고, 스텝마다 항목별로 고른다(active_rows).


def join_term_rows(terms: list[tuple["torch.Tensor", int]]) -> tuple["torch.Tensor | None", int]:
    """한 스케줄 줄의 음수 항 조건들(_cond_dealer 의 ([행 수, D], 행 수))을 토큰 축으로 이은 (행 | None, 행 수)."""
    terms = [(rows, count) for rows, count in terms if count > 0]
    if not terms:
        return None, 0
    import torch

    rows = torch.cat([rows for rows, _ in terms], 0) if len(terms) > 1 else terms[0][0]
    return rows, rows.shape[0]


def active_rows(schedules, sampling_step: int) -> tuple[list, list[int]]:
    """항목마다 이 스텝에 덧붙일 (행 | None) 과 행 수 — 상류와 같은 문턱(step >= sampling_step + 2)을 넘는 첫 스케줄 줄.

    상류는 긍정 쪽만 첫 줄에서 멈추고 부정 쪽은 break 가 안쪽 반복에 있어 뒤 줄들의 첫 항까지 모았다(hako-mikan 원본부터) —
    부정 프롬프트의 `[(x:-1):5]` 가 0 스텝부터 걸렸다. 두 쪽이 같은 규칙을 쓴다. 맞는 줄이 없으면 그 항목은 덧붙이지 않는다.
    """
    rows: list = []
    counts: list[int] = []
    for schedule in schedules or ():
        entry = (None, 0)
        for step, item in schedule:
            if step >= sampling_step + 2:
                entry = item
                break
        rows.append(entry[0])
        counts.append(entry[1])
    return rows, counts


def context_rows(cond) -> int:
    """CFGDenoiserParams.text_cond/text_uncond 의 문맥 행 수(SDXL 은 dict 의 crossattn) — 모르면 0."""
    if isinstance(cond, dict) or hasattr(cond, "keys"):
        try:
            cond = cond["crossattn"]
        except (KeyError, TypeError):
            return 0
    shape = getattr(cond, "shape", None)
    if shape is None or len(shape) < 2:
        return 0
    return int(shape[1])
