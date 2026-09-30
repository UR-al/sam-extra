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
# @ 0585496 (archived 2026-09-30), lib_negpip/sd.py.
# MODIFIED by sam-extra, 2026-09-30 (AGPL-3.0 section 5a): import paths (lib_negpip -> sam3ext.negpip); _hook_forward's
# forward reads Forge's own chunk labels (transformer_options["cond_or_uncond"], passed to attn2 by the Forge sampler) instead
# of guessing cond/uncond from the sampler name, x.shape[0] == 2*batch_size, the per-module Counter and context lengths
# (upstream's guess stays only as a fallback for callers without the labels, with a one-time warning); appends to each batch
# row the negative-term rows of that row's own prompt with every term (upstream: the first term of batch item 0 on every
# row), runs rows without terms through the original forward, repeats the appended rows when Forge lcm-repeated the
# context, and appends them to a separately passed value as well. Counter, patch_sd_negpip and _main_forward unchanged.

from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.negpip import NegPiP

    from backend.nn.unet import CrossAttention
    from backend.nn.unet import IntegratedUNet2DConditionModel as UNet

import torch

from sam3ext.negpip import IS_NEO
from modules import shared

if IS_NEO:
    from backend.attention import attention_function as optimized_attention
else:
    from ldm_patched.ldm.modules.attention import optimized_attention


def patch_sd_negpip(instance: "NegPiP", cls: "NegPiP", *, unpatch=False):
    if unpatch != cls._patched[0]:
        return

    cls._patched[0] = not cls._patched[0]

    unet: "UNet" = shared.sd_model.forge_objects.unet.model.diffusion_model
    for name, module in unet.named_modules():
        if "attn2" in name and module.__class__.__name__ == "CrossAttention":
            _hook_forward(instance, module, unpatch)


# ================================================================================ #


class Counter:
    def __init__(self, xl: bool):
        self.count: int = 0
        self.limit: int = 70 if xl else 16
        self.p: bool = True

    def counter(self) -> bool:
        outpn = self.p

        self.count += 1
        if self.count == self.limit:
            self.p = not self.p
            self.count = 0

        return outpn


# ---- sam-extra: 배치의 행마다 cond/uncond 와 항목을 Forge 가 넘기는 표시로 가려 그 행의 음수 항을 붙인다 ----
# 상류 0585496 은 (1) 조건 목록의 [0](항목 0 의 첫 항)만 모든 행에 붙이고, (2) cond/uncond 를 x.shape[0] == 2*batch_size 와 샘플러
# 이름(rev), 아니면 모듈마다의 호출 수(Counter, 한도 16/70 은 원래 UNet 전체의 attn2 수라 모듈마다 두면 16/70 패스마다 뒤집힘)와
# 문맥 길이로 추정했다 — Forge 는 [U, C] 로 묶거나(샘플러와 무관), 메모리가 모자라면 U·C 를 따로, 문맥 길이의 lcm 비가 4 를 넘으면
# C·U 를 따로, CFG 1(·Skip Early CFG·NGMS)이면 C 만, AND 프롬프트면 C 조각을 여럿 돌린다. 그 조각 순서를 Forge 가
# transformer_options["cond_or_uncond"] 로 attn2 까지 넘기므로(backend/sampling/sampling_function.py calc_cond_uncond_batch →
# backend/nn/unet.py BasicTransformerBlock, ad88b6b4·ceff5168 같음) 편입본은 그것을 읽는다. 조각은 x.shape[0] // len 행씩이고
# 조각 안의 행은 배치 항목 순서다(compile_conditions·compile_weighted_conditions).

COND, UNCOND = 0, 1  # Forge calc_cond_uncond_batch 의 조각 표시

_warned: set[str] = set()


def _warn_once(key: str, text: str) -> None:
    """프로세스당 한 번 — 매 호출(스텝 × attn2 수)마다 콘솔을 채우지 않는다."""
    if key in _warned:
        return
    _warned.add(key)
    print(f"[sam-extra] WARNING: NegPiP: {text}")


def _chunk_labels(cls: "NegPiP", counter: Counter, x, context, kwargs) -> list[int] | None:
    """이 호출의 배치 조각마다 COND/UNCOND. 표시가 없는 호출자면 상류의 추정 그대로(경고 한 번), 못 고르면 None."""
    options = kwargs.get("transformer_options")
    labels = options.get("cond_or_uncond") if isinstance(options, dict) else None
    if labels and x.shape[0] % len(labels) == 0:
        return list(labels)

    _warn_once(
        "labels",
        "cross-attention was called without transformer_options['cond_or_uncond'] (an older Forge or a caller "
        "other than Forge's sampler); falling back to upstream's cond/uncond guess (sampler name, call counter, "
        "context length), which can pick the wrong half when cond and uncond run as separate batches.",
    )
    if x.shape[0] == cls.batch_size * 2:
        return [UNCOND, COND] if cls.rev else [COND, UNCOND]
    concon = counter.counter()
    if context.shape[1] == cls.c_len * 77 and concon:
        return [COND]
    if context.shape[1] == cls.uc_len * 77 and concon:
        return [UNCOND]
    return None


def _item_rows(items, counts, count: int) -> list[tuple["torch.Tensor | None", int]]:
    """조각의 행마다 (덧붙일 행 | None, 행 수). 행 r 은 항목 r % 항목 수 — Forge 가 조건을 조각의 행 수에 맞추는 규칙과 같다
    (backend/sampling/condition.py repeat_to_batch_size: 반복 후 자르기). 보통은 행 수 = 항목 수(배치 크기)라 행 r = 항목 r."""
    items = list(items or ())
    counts = list(counts or ())
    if len(items) == count:
        return list(zip(items, counts))
    if all(item is None for item in items):
        return [(None, 0)] * count
    if all(item is items[0] for item in items):
        return [(items[0], counts[0])] * count
    _warn_once(
        "items",
        f"a batch chunk has {count} rows for {len(items)} prompts with different negative terms (not Forge's usual "
        "layout); pairing row r with prompt r % n as Forge's repeat_to_batch_size does.",
    )
    return [(items[r % len(items)], counts[r % len(items)]) for r in range(count)]


def _context_repeat(context, native: int) -> int:
    """Forge 가 cond·uncond 를 한 배치로 묶으며 문맥을 lcm 길이로 반복했으면 그 배수(ConditionCrossAttn.concat), 아니면 1.

    같은 행을 k 번 반복한 문맥은 어텐션 결과가 원래와 같지만, 뒤에 붙인 음수 항 행은 softmax 몫이 1/k 로 준다 — 붙일 행도 k 번
    반복하면 원래 문맥 + 음수 항과 정확히 같다. native 는 이 스텝 조건의 원래 길이(denoiser_callback 의 text_cond/text_uncond).
    """
    length = context.shape[1]
    if context.shape[0] == 0 or native <= 0 or length <= native or length % native:
        return 1
    k = length // native
    tiles = context[:1].reshape(1, k, native, -1)
    return k if bool((tiles == tiles[:, :1]).all()) else 1


def _row_groups(cls: "NegPiP", labels: list[int], x, context) -> list[tuple["torch.Tensor | None", int, int, list[int]]]:
    """덧붙일 것이 같은 행끼리 — [(덧붙일 행 | None, V 를 뒤집을 행 수, 반복, [행 번호…])]."""
    per_chunk = x.shape[0] // len(labels)
    groups: dict = {}
    for chunk, label in enumerate(labels):
        if label == COND:
            items, counts, native = cls.conds, cls.c_tokens, cls.c_native
        elif label == UNCOND:
            items, counts, native = cls.unconds, cls.uc_tokens, cls.uc_native
        else:
            items, counts, native = None, None, 0
        start = chunk * per_chunk
        repeat = None
        for offset, (extra, count) in enumerate(_item_rows(items, counts, per_chunk)):
            if extra is None or not count:
                key = None
            else:
                if repeat is None:
                    repeat = _context_repeat(context[start : start + per_chunk], native)
                key = (id(extra), repeat)
            if key not in groups:
                groups[key] = (extra, count, repeat or 1, []) if key is not None else (None, 0, 1, [])
            groups[key][3].append(start + offset)
    return list(groups.values())


def _group_forward(cls: "NegPiP", module: "CrossAttention", x, context, value, extra, tokens, repeat, args, kwargs):
    if extra is None:
        # 음수 항이 없는 행 — NegPiP 가 없을 때와 똑같이 원래 forward
        return module.orig_forward(x, context, value, None, *args, **kwargs)
    extra = extra.to(device=context.device, dtype=context.dtype)
    if repeat > 1:
        extra = extra.repeat(repeat, 1)
    extra = extra.unsqueeze(0).expand(context.shape[0], -1, -1)
    context = torch.cat([context, extra], 1)
    if value is not None:
        # attn2_patch 가 value 를 따로 넘기면 거기에도 같은 행을 붙인다 — 상류는 K 만 길어져 모양이 어긋났다
        value = torch.cat([value, extra.to(value.dtype)], 1)
    return _main_forward(cls, module, x, context, value, None, tokens * repeat)


def _batch_to(tensor, rows: int):
    if tensor is None or tensor.shape[0] == rows:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(rows, *tensor.shape[1:])
    return None


def _hook_forward(cls: "NegPiP", module: "CrossAttention", remove: bool):
    if remove:
        if hasattr(module, "orig_forward"):
            module.forward = module.orig_forward
            del module.orig_forward
        return

    counter = Counter(cls.is_xl)

    module.orig_forward = module.forward

    @torch.inference_mode()
    @wraps(module.orig_forward)
    def forward(x, context=None, value=None, mask=None, *args, **kwargs):
        if context is None:
            return module.orig_forward(x, context, value, mask, *args, **kwargs)

        rows = x.shape[0]
        context_b, value_b = _batch_to(context, rows), _batch_to(value, rows)
        labels = _chunk_labels(cls, counter, x, context, kwargs)
        groups = _row_groups(cls, labels, x, context if context_b is None else context_b) if labels else []
        if all(extra is None for extra, *_ in groups):
            return module.orig_forward(x, context, value, mask, *args, **kwargs)

        if mask is not None or context_b is None or (value is not None and value_b is None):
            _warn_once(
                "shape",
                "cross-attention was called with an attention mask or a context batch that does not match the "
                "input; NegPiP is skipped for such calls.",
            )
            return module.orig_forward(x, context, value, mask, *args, **kwargs)

        if len(groups) == 1:
            extra, tokens, repeat, _ = groups[0]
            return _group_forward(cls, module, x, context_b, value_b, extra, tokens, repeat, args, kwargs)

        out = None
        for extra, tokens, repeat, indices in groups:
            index = torch.tensor(indices, device=x.device)
            y = _group_forward(
                cls,
                module,
                x.index_select(0, index),
                context_b.index_select(0, index),
                None if value_b is None else value_b.index_select(0, index),
                extra,
                tokens,
                repeat,
                args,
                kwargs,
            )
            if out is None:
                out = y.new_empty((rows, *y.shape[1:]))
            out.index_copy_(0, index, y)
        return out

    module.forward = forward


@torch.inference_mode()
def _main_forward(cls: "NegPiP", attn: "CrossAttention", x, ctx, value, mask, tokens):
    q = attn.to_q(x)
    ctx = ctx.to(x.dtype)
    k = attn.to_k(ctx)

    if value is not None:
        v = attn.to_v(value)
        del value
    else:
        v = attn.to_v(ctx)

    if cls.active:
        if tokens:
            v[:, -tokens:, :] = -v[:, -tokens:, :]

    out = optimized_attention(q, k, v, attn.heads, mask)
    return attn.to_out(out)
