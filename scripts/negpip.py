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
# detected from the engine itself (sam3ext/negpip/utils.py clip_fragment_specials); old engine unchanged. Load order
# after sd-dynamic-thresholding via metadata.ini. File name (scripts/negpip.py — ADetailer's ad_script_names selects
# always-on scripts by this stem), title, UI and behaviour otherwise unchanged.

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

import torch
from sam3ext.negpip import INCOMPATIBLE_EXTENSIONS, IS_NEO, PATCHED
from sam3ext.negpip.anima import patch_anima_negpip
from sam3ext.negpip.coexist import BUILTIN_ATTR, standalone_reason, warn_once
from sam3ext.negpip.mask import negpip_effective
from sam3ext.negpip.sd import patch_sd_negpip
from sam3ext.negpip.utils import (
    NEG_PATTERN,
    any_negative,
    clip_fragment_specials,
    clip_word_rows,
    hr_dealer,
    reset_prompt_cache,
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
        # sam-extra: 새 ClipEngine 처럼 조각마다 특수 토큰을 붙이는 엔진이면 그 id 들 (utils.clip_fragment_specials)
        self.clip_specials: frozenset[int] | None = None

        self.has_hr_p: bool
        self.has_hr_n: bool
        self.rev: bool
        self.batch_size: int

        self.conds: list[torch.Tensor]
        self.c_len: int
        self.c_tokens: list[int]
        self.conds_all: list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]
        self.hr_conds_all: list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]

        self.unconds: list[torch.Tensor]
        self.uc_len: int
        self.uc_tokens: list[int]
        self.unconds_all: list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]
        self.hr_unconds_all: list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]

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
                if not negpip_effective(p.sd_model.text_processing_engine_anima, name):
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
            # sam-extra: 엔진이 조각마다 BOS/EOS 를 붙이는지 엔진에 물어 둔다 — _cond_dealer 가 옛 엔진과 같은 행을 고른다
            self.clip_specials = clip_fragment_specials(getattr(self.tokenizer, "__self__", None))
        else:
            self.tokenizer = (
                p.sd_model.conditioner.embedders[0].tokenize_line
                if self.is_xl
                else p.sd_model.cond_stage_model.tokenize_line
            )

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

        def calcChunks(a: int, b: int) -> int:
            return a // b if a % b == 0 else a // b + 1

        self.c_len = calcChunks(self.tokenizer(p.prompts[0])[1], 75)
        self.uc_len = calcChunks(self.tokenizer(p.negative_prompts[0])[1], 75)

        patch_sd_negpip(self, NegPiP)
        reset_prompt_cache(p)
        p.extra_generation_params["NegPiP"] = True
        self.active = True

        if len(self.conds_all[0][0][1]) > 0:
            print(f"NegPiP Enable (Positive: {self.conds_all[0][0][1][0][1]})")
        if len(self.unconds_all[0][0][1]) > 0:
            print(f"NegPiP Enable (Negative: {self.unconds_all[0][0][1][0][1]})")

    def postprocess(self, *args, **kwargs):
        self.reset()

    def before_hr(self, *args, **kwargs):
        self.is_hr = True

    def denoiser_callback(self, params: CFGDenoiserParams):
        if (not self.active) or self.is_anima:
            return

        conds_list = []
        tokens_list = []

        if self.is_hr and self.has_hr_p:
            conds = self.hr_conds_all
        else:
            conds = self.conds_all

        if conds is not None:
            for step, regions in conds[0]:
                if step >= params.sampling_step + 2:
                    for conds, tokens in regions:
                        conds_list.append(conds)
                        tokens_list.append(tokens)
                    break
            self.conds = conds_list
            self.c_tokens = tokens_list

        unconds_list = []
        uc_tokens_list = []

        if self.is_hr and self.has_hr_n:
            unconds = self.hr_unconds_all
        else:
            unconds = self.unconds_all

        if unconds is not None:
            for step, regions in unconds[0]:
                if step >= params.sampling_step + 2:
                    for unconds, uc_tokens in regions:
                        unconds_list.append(unconds)
                        uc_tokens_list.append(uc_tokens)
                        break
            self.unconds = unconds_list
            self.uc_tokens = uc_tokens_list

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

        input = SdConditioning(
            [f"({target[0]}:{-target[1]})"],
            width=p.width,
            height=p.height,
        )

        cond = get_learned_conditioning(p.sd_model, input, p.steps)

        chunks, token_len = self.tokenizer(target[0])

        if self.clip_specials is not None:
            # sam-extra: 새 ClipEngine — [BOS, 단어…, EOS, EOS] 대신 옛 엔진과 같은 [단어…, EOS] 행 (utils.clip_word_rows)
            rows = cond[0][0].cond if not self.is_xl else cond[0][0].cond["crossattn"]
            conds.append(rows[clip_word_rows(chunks, self.clip_specials), :])
        else:
            conds.append(
                cond[0][0].cond[1 : token_len + 2, :]
                if not self.is_xl
                else cond[0][0].cond["crossattn"][1 : token_len + 2, :]
            )

        conds = torch.cat(conds, 0).unsqueeze(0)
        return conds.repeat(self.batch_size, 1, 1), conds.shape[1]

    def _calc_conds(
        self,
        p: "StableDiffusionProcessing",
        targetlist: list[list[tuple[int, list[tuple[str, float]]]]],
    ) -> list[list[tuple[int, list[tuple[torch.Tensor, int]]]]]:
        outconds = []
        for batch in targetlist:
            stepconds = []
            for step, regions in batch:
                regionconds = []
                for targets in regions:
                    conds, c_tokens = self._cond_dealer(p, targets)
                    regionconds.append((conds, c_tokens))
                stepconds.append((step, regionconds))
            outconds.append(stepconds)
        return outconds
