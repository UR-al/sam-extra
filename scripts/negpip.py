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
# @ 0585496 (archived 2026-09-30), scripts/negpip.py.
# MODIFIED by sam-extra, 2026-09-30 (AGPL-3.0 section 5a): import paths (lib_negpip -> sam3ext.negpip); patch state
# shared across script reloads (sam3ext.negpip.PATCHED); stands down when a separately installed NegPiP is loaded
# (sam3ext/negpip/coexist.py); on Anima, stays off when the emphasis mode does not apply negative weights
# (sam3ext/negpip/mask.py); on SD1/SDXL with Forge's new ClipEngine (which wraps every prompt fragment in BOS/EOS),
# _cond_dealer takes the same condition rows as on the old engine ([words..., EOS], not [BOS, words..., EOS, EOS]) —
# detected from the engine itself (sam3ext/negpip/utils.py clip_fragment_specials). _cond_dealer tokenizes the exact
# "(text:-w)" string it encodes (upstream tokenized the bare text, whose chunk layout differs when BREAK sits inside a
# weighted group — IndexError on the new engine, padding/BOS rows on the old one) and, on both engines, picks the word
# rows plus the EOS after them by skipping the engine's special tokens (sam3ext/negpip/utils.py clip_row_specials /
# encoded_clip_rows; single-chunk terms: same rows as the upstream cond[1:token_len+2] slice except under emphasis "None",
# where the literal "(text:w)" characters are all taken, multi-chunk terms: no chunk
# padding/BOS rows); if the tokenized rows ever disagree with the encoded rows it stands down for that generation and puts
# back the prompts it had edited. SD1/SDXL conditions per batch item with every negative term (upstream: one tensor per
# term and the hook used only the first; only batch item 0's schedule, repeated batch_size times): _cond_dealer returns
# the un-repeated [rows, D], _calc_conds joins the terms of each schedule line (utils.join_term_rows, identical lines
# share one tensor), denoiser_callback picks each item's line with the same rule on the cond and uncond side (upstream's
# uncond break sat in the inner loop — utils.active_rows) and records the native context lengths for sd.py; the
# 'NegPiP Enable' line reports the largest joined row count of any item. Load order after sd-dynamic-thresholding via
# metadata.ini. File name (scripts/negpip.py — ADetailer's ad_script_names selects always-on scripts by this stem),
# title, UI and behaviour otherwise unchanged.
# MODIFIED by sam-extra, 2026-10-02: the Anima emphasis gate looks the text engine up with
# sam3ext.anima38.native_engine.anima_text_engine (Forge 2.29.2 renamed sd_model.text_processing_engine_anima to the
# text_processing_engine_qwen name that Flux2/Krea2/Qwen-Image/Z-Image share).

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

import torch
from sam3ext.anima38.native_engine import anima_text_engine
from sam3ext.negpip import INCOMPATIBLE_EXTENSIONS, IS_NEO, PATCHED
from sam3ext.negpip.anima import patch_anima_negpip
from sam3ext.negpip.coexist import BUILTIN_ATTR, standalone_reason, warn_once
from sam3ext.negpip.mask import negpip_effective
from sam3ext.negpip.sd import patch_sd_negpip
from sam3ext.negpip.utils import (
    NEG_PATTERN,
    active_rows,
    any_negative,
    ClipRowMismatch,
    clip_row_specials,
    context_rows,
    encoded_clip_rows,
    hr_dealer,
    join_term_rows,
    reset_prompt_cache,
    restore_prompts,
    snapshot_prompts,
)

from modules import scripts, shared
from modules.prompt_parser import (
    SdConditioning,
    get_learned_conditioning,
    get_learned_conditioning_prompt_schedules,
)
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser


def _verify_ext(p: " StableDiffusionProcessing"):
    for ext in p.scripts.scripts:
        if ext.title() not in INCOMPATIBLE_EXTENSIONS:
            continue
        if p.script_args[ext.args_from] is True:
            return False

    return True


def _emphasis_name() -> str:
    """Forge 엔진이 쓸 emphasis 이름 — 모르는 값은 엔진처럼 Original 로 (get_current_option)."""
    from backend.text_processing import emphasis

    return emphasis.get_current_option(shared.opts.emphasis).name


class NegPiP(scripts.Script):
    # sam-extra: 패키지의 목록을 쓴다 — Reload UI 로 클래스가 새로 생겨도 패치 상태를 잊지 않는다
    _patched: list[bool] = PATCHED

    def __init__(self):
        # sam-extra: 따로 설치된 NegPiP 와 구별하는 표시 (sam3ext/negpip/coexist.py)
        setattr(self, BUILTIN_ATTR, True)

        self.active: bool = False

        self.is_xl: bool
        self.is_anima: bool
        self.is_hr: bool

        self.tokenizer: torch.nn.Module
        # sam-extra: _cond_dealer 가 행을 고를 때 건너뛸 엔진의 특수 토큰 id (utils.clip_row_specials) — 모르면 None(상류 자르기)
        self.clip_specials: frozenset[int] | None = None

        self.has_hr_p: bool
        self.has_hr_n: bool
        self.rev: bool
        self.batch_size: int

        # sam-extra: 항목(배치 안 프롬프트)마다 이 스텝에 덧붙일 음수 항 행 [행 수, D](항 전부를 이은 것) 또는 None 과 그 행 수.
        # *_all 은 항목마다 [(스케줄 줄 끝 스텝, (행 | None, 행 수)), ...] — 상류는 항마다 텐서였고 훅이 [0] 만 썼다
        self.conds: list[torch.Tensor | None]
        self.c_len: int
        self.c_tokens: list[int]
        self.c_native: int = 0
        self.conds_all: list[list[tuple[int, tuple[torch.Tensor | None, int]]]]
        self.hr_conds_all: list[list[tuple[int, tuple[torch.Tensor | None, int]]]]

        self.unconds: list[torch.Tensor | None]
        self.uc_len: int
        self.uc_tokens: list[int]
        self.uc_native: int = 0
        self.unconds_all: list[list[tuple[int, tuple[torch.Tensor | None, int]]]]
        self.hr_unconds_all: list[list[tuple[int, tuple[torch.Tensor | None, int]]]]

        on_cfg_denoiser(self.denoiser_callback)

    def reset(self):
        self.active = False

        self.is_xl = False
        self.is_anima = False
        self.is_hr = False

        self.tokenizer = None

        self.conds = None
        self.c_tokens = None
        self.conds_all = None
        self.hr_conds_all = None

        self.unconds = None
        self.uc_tokens = None
        self.unconds_all = None
        self.hr_unconds_all = None

        patch_sd_negpip(None, NegPiP, unpatch=True)
        patch_anima_negpip(NegPiP, unpatch=True)

    def title(self):
        return "NegPiP"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return None

    def process_batch(self, p: "StableDiffusionProcessing", *args, **kwargs):
        self.reset()

        if not any_negative(p):
            return

        # sam-extra: 따로 설치된 sd-forge-negpip 가 있으면 내장은 쉰다 — 두 번 걸리지 않게 (경고는 한 번)
        if (reason := standalone_reason(p, NegPiP._patched)) is not None:
            warn_once(reason)
            return

        if not _verify_ext(p):
            print("NegPiP Disabled")
            return

        if IS_NEO and not p.sd_model.is_webui_legacy_model():
            self.is_anima = type(p.sd_model).__name__ == "Anima"
            if self.is_anima:
                # sam-extra: 엔진이 음수 가중치를 조건에 곱하지 않는 emphasis 면 NegPiP 는 뜻이 없다 (mask.py)
                name = _emphasis_name()
                if not negpip_effective(anima_text_engine(p.sd_model), name):
                    print(f"NegPiP Disabled (Emphasis: {name})")
                    self.is_anima = False
                    return
                patch_anima_negpip(NegPiP)
                reset_prompt_cache(p)
                p.extra_generation_params["NegPiP"] = True
                self.active = True
            return

        self.is_xl = p.sd_model.is_sdxl
        self.batch_size = p.batch_size
        self.has_hr_p, self.has_hr_n = hr_dealer(p)
        self.rev = p.sampler_name not in ("DDIM", "PLMS", "UniPC")

        if IS_NEO:
            self.tokenizer = (
                p.sd_model.text_processing_engine_l.tokenize_line
                if self.is_xl
                else p.sd_model.text_processing_engine.tokenize_line
            )
        else:
            self.tokenizer = (
                p.sd_model.conditioner.embedders[0].tokenize_line
                if self.is_xl
                else p.sd_model.cond_stage_model.tokenize_line
            )
        # sam-extra: 엔진의 특수 토큰(새 엔진은 조각마다 붙는 BOS/EOS 까지)을 엔진에 물어 둔다 — _cond_dealer 가 옛·새 엔진에서
        # 같은 [단어…, EOS] 행을 고른다 (utils.clip_row_specials)
        self.clip_specials = clip_row_specials(getattr(self.tokenizer, "__self__", None))

        # sam-extra: 음수 항 조건의 행 수가 토큰화와 어긋나면(ClipRowMismatch) 이 생성에서 물러나며 지운 음수 항을 되돌린다
        prompts_before = snapshot_prompts(p)
        try:
            nip = self._getScheduledNegPip(p.prompts, p.steps)
            pin = self._getScheduledNegPip(p.negative_prompts, p.steps)

            self.conds_all = self._calc_conds(p, nip)
            self.unconds_all = self._calc_conds(p, pin)

            hr_steps: int = getattr(p, "hr_second_pass_steps", 0) or p.steps

            if self.has_hr_p:
                hr_nip = self._getScheduledNegPip(p.hr_prompts, hr_steps)
                self.hr_conds_all = self._calc_conds(p, hr_nip)

            if self.has_hr_n:
                hr_pin = self._getScheduledNegPip(p.hr_negative_prompts, hr_steps)
                self.hr_unconds_all = self._calc_conds(p, hr_pin)
        except ClipRowMismatch as exc:
            restore_prompts(prompts_before)
            print(f"NegPiP Disabled (condition rows: {exc})")
            return

        def calcChunks(a: int, b: int) -> int:
            return a // b if a % b == 0 else a // b + 1

        self.c_len = calcChunks(self.tokenizer(p.prompts[0])[1], 75)
        self.uc_len = calcChunks(self.tokenizer(p.negative_prompts[0])[1], 75)

        patch_sd_negpip(self, NegPiP)
        reset_prompt_cache(p)
        p.extra_generation_params["NegPiP"] = True
        self.active = True

        # sam-extra: 상류는 항목 0 의 첫 항 행 수만 찍었다(항목 0 에 항이 없으면 줄이 없음) — 어느 항목·줄이든 이은 행 수의 최대
        for side, schedules in (("Positive", self.conds_all), ("Negative", self.unconds_all)):
            rows = max((count for schedule in schedules for _, (_, count) in schedule), default=0)
            if rows > 0:
                print(f"NegPiP Enable ({side}: {rows})")

    def postprocess(self, *args, **kwargs):
        self.reset()

    def before_hr(self, *args, **kwargs):
        self.is_hr = True

    def denoiser_callback(self, params: CFGDenoiserParams):
        if (not self.active) or self.is_anima:
            return

        if self.is_hr and self.has_hr_p:
            conds = self.hr_conds_all
        else:
            conds = self.conds_all

        if self.is_hr and self.has_hr_n:
            unconds = self.hr_unconds_all
        else:
            unconds = self.unconds_all

        # sam-extra: 항목마다 그 항목의 스케줄에서 고른다(상류는 항목 0 만) — 긍정·부정 같은 규칙 (utils.active_rows)
        self.conds, self.c_tokens = active_rows(conds, params.sampling_step)
        self.unconds, self.uc_tokens = active_rows(unconds, params.sampling_step)
        # sam-extra: 이 스텝 조건의 원래 문맥 길이 — Forge 가 cond·uncond 를 한 배치로 묶으며 lcm 으로 반복한 문맥을 훅이 알아본다
        self.c_native = context_rows(getattr(params, "text_cond", None))
        self.uc_native = context_rows(getattr(params, "text_uncond", None))

    @staticmethod
    def _getScheduledNegPip(
        prompts: list[str], steps: list[int]
    ) -> list[list[tuple[int, list[tuple[str, float]]]]]:
        """extract the prompts with negative weights"""

        output = []

        scheduled = get_learned_conditioning_prompt_schedules(prompts, steps)
        for i, batch_schedule in enumerate(scheduled):
            stepout = []

            for step, prompt in batch_schedule:
                neg_matches: list[str] = re.findall(NEG_PATTERN, prompt)
                neg_targets = []

                for minusmatch in neg_matches:
                    prompts[i] = prompts[i].replace(minusmatch, "")
                    neg_targets.append(minusmatch.strip("(").strip(")"))

                neg_targets: list[tuple[str, str]] = [x.split(":") for x in neg_targets]
                text_weights: list[tuple[str, float]] = []

                for text, weight in neg_targets:
                    if text.strip() in ("BREAK", "AND", "ADDCOL", "ADDROW"):
                        continue
                    if (weight := float(weight)) < 0.0:
                        text_weights.append((text, weight))

                stepout.append((step, text_weights))

            output.append(stepout)

        return output

    def _cond_dealer(
        self, p: "StableDiffusionProcessing", target: tuple[str, float]
    ) -> tuple[torch.Tensor, int]:
        conds = []

        # sam-extra: 행은 인코딩한 바로 그 글자에서 센다. 상류는 맨 글자 target[0](가중치 1)를 토큰화했는데, 묶음 가중치가 1 이
        # 아니면 Forge 파서가 BREAK 를 청크 구분이 아닌 글자로 남겨 청크 배치가 조건과 달랐다 (utils.py 머리 주석)
        text = f"({target[0]}:{-target[1]})"

        input = SdConditioning(
            [text],
            width=p.width,
            height=p.height,
        )

        cond = get_learned_conditioning(p.sd_model, input, p.steps)

        # cond[0][0] 은 첫 스케줄 줄의 조건 — 그 줄의 글자를 토큰화한다 (항 안의 [a|b] 등도 인코딩과 같게)
        encoded = get_learned_conditioning_prompt_schedules([text], p.steps)[0][0][1]
        chunks, token_len = self.tokenizer(encoded)

        rows = cond[0][0].cond if not self.is_xl else cond[0][0].cond["crossattn"]
        if self.clip_specials is not None:
            # sam-extra: 엔진의 특수 토큰을 건너뛴 [단어…, EOS] 행 — 옛·새 엔진이 같은 행 (utils.encoded_clip_rows)
            conds.append(rows[encoded_clip_rows(chunks, self.clip_specials, rows.shape[0]), :])
        else:
            conds.append(rows[1 : token_len + 2, :])

        # sam-extra: 반복하지 않은 [행 수, D] — 항목마다 제 행을 쓰므로 batch_size 로 늘리지 않는다 (훅이 행마다 붙인다)
        conds = torch.cat(conds, 0)
        return conds, conds.shape[0]

    def _calc_conds(
        self,
        p: "StableDiffusionProcessing",
        targetlist: list[list[tuple[int, list[tuple[str, float]]]]],
    ) -> list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]:
        # sam-extra: 스케줄 줄마다 항 전부를 이은 (행 | None, 행 수) 하나 (utils.join_term_rows). 같은 항·같은 줄은 한 번만
        # 인코딩해 같은 텐서를 나눠 쓴다 — 훅은 같은 텐서를 붙이는 행들을 한 번에 계산한다
        outconds = []
        term_cache: dict[tuple[str, float], tuple[torch.Tensor, int]] = {}
        line_cache: dict[tuple[tuple[str, float], ...], tuple[torch.Tensor | None, int]] = {}
        for batch in targetlist:
            stepconds = []
            for step, regions in batch:
                key = tuple(regions)
                if key not in line_cache:
                    for targets in regions:
                        if targets not in term_cache:
                            term_cache[targets] = self._cond_dealer(p, targets)
                    line_cache[key] = join_term_rows([term_cache[targets] for targets in regions])
                stepconds.append((step, line_cache[key]))
            outconds.append(stepconds)
        return outconds
