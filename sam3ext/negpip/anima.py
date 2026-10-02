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
# @ 0585496 (archived 2026-09-30), lib_negpip/anima.py.
# MODIFIED by sam-extra, 2026-09-30 (AGPL-3.0 section 5a): import paths (lib_negpip -> sam3ext.negpip);
# _build_negpip_mask now calls sam3ext/negpip/mask.py, which supports both Forge Anima text engines and the
# engine's emphasis mode. negpip_learned_conditioning returns one {"crossattn", "c_negpip_mask"} dict per schedule line
# (sam3ext/negpip/mask.py negpip_line_conds) instead of torch.stack-ing every line into one dict — Forge's per-line
# contract, so prompt-editing variants longer than 512 rows with different lengths no longer fail with "stack expects
# each tensor to be equal size". The other hooks are unchanged.
# MODIFIED by sam-extra, 2026-10-02: the Anima text engine is looked up with sam3ext.anima38.native_engine.anima_text_engine
# (Forge 2.29.2 renamed sd_model.text_processing_engine_anima to the text_processing_engine_qwen name that Flux2/Krea2/
# Qwen-Image/Z-Image share; upstream read the old attribute directly).

# https://github.com/david419kr/sd-webui-negpip/blob/main/scripts/negpip.py

from functools import wraps
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from scripts.negpip import NegPiP

    from backend.diffusion_engine.anima import Anima as AnimaEngine
    from backend.nn.anima import Anima
    from backend.text_processing.anima_engine import Qwen06Engine
    from modules.prompt_parser import SdConditioning

import torch
from einops import rearrange

from backend.nn.anima import SelfCrossAttention
from backend.sampling import condition, sampling_function
from modules import shared
from sam3ext.anima38.native_engine import anima_text_engine
from sam3ext.negpip.mask import build_negpip_mask, negpip_line_conds


def patch_anima_negpip(cls: "NegPiP", *, unpatch=False):
    if unpatch != cls._patched[1]:
        return

    cls._patched[1] = not cls._patched[1]

    model: "AnimaEngine" = shared.sd_model
    dit: "Anima" = model.forge_objects.unet.model.diffusion_model
    _hook_get_learned_conditioning(model, unpatch)
    _hook_dit_forward(dit, unpatch)
    _hook_forwards(unpatch)
    _hook_compile_conditions(unpatch)


# ================================================================================ #


def _hook_get_learned_conditioning(model: "AnimaEngine", remove: bool):
    if remove:
        if hasattr(model, "orig_forward"):
            model.get_learned_conditioning = model.orig_forward
            del model.orig_forward
        return

    model.orig_forward = model.get_learned_conditioning

    engine: "Qwen06Engine" = anima_text_engine(model)

    @torch.inference_mode()
    @wraps(model.orig_forward)
    def negpip_learned_conditioning(prompt: "SdConditioning"):
        conds = model.orig_forward(prompt)
        assert isinstance(conds, list)
        assert len(prompt) == len(conds)
        assert all(isinstance(cond, torch.Tensor) for cond in conds)

        # sam-extra: 줄마다 dict 하나(길이도 줄마다) — 상류는 모든 줄을 torch.stack 해, 512 행을 넘는 프롬프트 편집 줄들의
        # 길이가 다르면 조건 단계에서 죽었다. 스텝마다 줄 고르기·배치 맞추기는 prompt_parser 가 순정과 같게 한다 (mask.py)
        lines, _count = negpip_line_conds(engine, prompt, conds, build_mask=_build_negpip_mask)

        if _count > 0:
            key = "Negative" if prompt.is_negative_prompt else "Positive"
            print(f"NegPiP Enable ({key}: {_count})")

        return lines

    model.get_learned_conditioning = negpip_learned_conditioning


def _build_negpip_mask(
    text_processing_engine: "Qwen06Engine",
    line: str,
    token_length: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
):
    # sam-extra: 옛·새 엔진 모두, 그 엔진의 emphasis 방식대로 (sam3ext/negpip/mask.py)
    return build_negpip_mask(text_processing_engine, line, token_length, device, dtype)


def _hook_dit_forward(dit: "Anima", remove: bool):
    if remove:
        if hasattr(dit, "orig_forward"):
            if getattr(dit.forward, "_negpip", False):
                dit.forward = dit.orig_forward
            del dit.orig_forward
        return

    dit.orig_forward = dit.forward

    @torch.inference_mode()
    @wraps(dit.orig_forward)
    def negpip_forward(
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        transformer_options = kwargs.get("transformer_options", {})

        negpip_mask = kwargs.get("c_negpip_mask", None)
        if negpip_mask is None:
            negpip_mask = torch.ones(
                context.shape[0],
                context.shape[1],
                1,
                device=context.device,
                dtype=context.dtype,
            )

        transformer_options["negpip_mask"] = negpip_mask
        kwargs["transformer_options"] = transformer_options

        return dit.orig_forward(x, timesteps, context, padding_mask, **kwargs)

    negpip_forward._negpip = True
    dit.forward = negpip_forward


def _hook_forwards(remove: bool):
    if remove:
        if hasattr(SelfCrossAttention, "negpip_orig_forward"):
            if getattr(SelfCrossAttention.forward, "_negpip", False):
                SelfCrossAttention.forward = SelfCrossAttention.negpip_orig_forward
            del SelfCrossAttention.negpip_orig_forward
        return

    SelfCrossAttention.negpip_orig_forward = SelfCrossAttention.forward

    @torch.inference_mode()
    @wraps(SelfCrossAttention.negpip_orig_forward)
    def negpip_forward(
        self: SelfCrossAttention,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
        transformer_options: Optional[dict] = {},
    ):
        if self.is_SelfAttn:
            return self.negpip_orig_forward(x, context, rope_emb, transformer_options)

        negpip_mask: torch.Tensor = transformer_options.get("negpip_mask", None)

        q = self.q_proj(x)
        context_k = x if context is None else context
        context_v = context_k
        if negpip_mask is not None:
            assert negpip_mask.ndim == context_v.ndim
            if (batch := (x.size(0) // negpip_mask.size(0))) > 1:
                negpip_mask = negpip_mask.repeat(batch, 1, 1)
            context_v = context_v * negpip_mask.to(context_v)

        k = self.k_proj(context_k)
        v = self.v_proj(context_v)

        q, k, v = map(
            lambda t: rearrange(
                t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim
            ),
            (q, k, v),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        if self.is_SelfAttn and rope_emb is not None:
            q = self.apply_rotary_pos_emb(q, rope_emb)
            k = self.apply_rotary_pos_emb(k, rope_emb)

        return self.compute_attention(q, k, v, transformer_options=transformer_options)

    negpip_forward._negpip = True
    SelfCrossAttention.forward = negpip_forward


def _hook_compile_conditions(remove: bool):
    if remove:
        if hasattr(condition, "orig_forward"):
            condition.compile_conditions = condition.orig_forward
            sampling_function.compile_conditions = condition.orig_forward
            del condition.orig_forward
        return

    condition.orig_forward = condition.compile_conditions

    @wraps(condition.orig_forward)
    def compile_conditions(cond):
        if cond is None:
            return None

        if isinstance(cond, dict) and "crossattn" in cond and "vector" not in cond:
            cross_attn = cond["crossattn"]
            model_conds = {"c_crossattn": condition.ConditionCrossAttn(cross_attn)}
            if "c_negpip_mask" in cond:
                model_conds["c_negpip_mask"] = condition.Condition(
                    cond["c_negpip_mask"]
                )
            return [dict(cross_attn=cross_attn, model_conds=model_conds)]

        return condition.orig_forward(cond)

    condition.compile_conditions = compile_conditions
    sampling_function.compile_conditions = compile_conditions
