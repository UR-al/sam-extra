"""Anima DiT 에 IP-Adapter 를 얹는다 — Forge 코어 파일은 건드리지 않는다.

설치는 DiT·블록·cross_attn 의 **껍데기 복제본**에만 한다(:func:`shell`). 가중치 텐서는 원본과
공유하고 모듈 딕셔너리만 복사하므로, 되돌리기는 패처 참조를 원래대로 두는 것으로 끝난다.

토큰은 ``transformer_options["sam3_ip_tokens"]`` 로 흘러온다.

상류(ComfyUI-Anima_IP-Adapter)와 다른 점:

* ``strength`` 는 토큰이 아니라 게이트에 곱한다 — 선형이고 null 토큰과 대칭이다.
* 전방 훅이 잡은 활성값을 블록 forward 끝에서 바로 버린다.
* 지원하지 않는 구조(:mod:`.checkpoint` 의 ``unsupported``)는 조용히 다르게 돌지 않고 거부한다.
"""
from __future__ import annotations

import copy
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn

TOKENS_KEY = "sam3_ip_tokens"

_LORA_RE = re.compile(
    r"lora\.base_model\.model\.blocks\.(\d+)\.cross_attn\.(\w+)"
    r"\.lora_([AB])(?:\.default)?\.weight"
)


@dataclass(frozen=True)
class Injection:
    """한 번의 생성에 쓰는 토큰과 세기."""

    tokens: torch.Tensor
    null_tokens: torch.Tensor
    gate_scale: float = 1.0
    separate_cfg: bool = False
    cfg_scale: float = 4.0


class _LoRALinear(nn.Module):
    """base(x) + (x @ A.T @ B.T) * scale. 복제본에만 끼우므로 되돌리기가 보장된다."""

    def __init__(self, base: nn.Linear, lora_a: torch.Tensor, lora_b: torch.Tensor,
                 scale: float = 1.0):
        super().__init__()
        self.base = base
        self.register_buffer("lora_a", lora_a, persistent=False)
        self.register_buffer("lora_b", lora_b, persistent=False)
        self.scale = float(scale)

    def forward(self, x):
        lora_a = self.lora_a.to(device=x.device, dtype=x.dtype)
        lora_b = self.lora_b.to(device=x.device, dtype=x.dtype)
        return self.base(x) + (x @ lora_a.T @ lora_b.T) * self.scale


class _SharedProj(nn.Module):
    """공유 투영: Linear → GELU → Linear."""

    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.expand = nn.Linear(dim, inner)
        self.act = nn.GELU()
        self.project = nn.Linear(inner, inner)

    def forward(self, x):
        return self.project(self.act(self.expand(x)))


class _SharedRef(nn.Module):
    """블록이 공유 모듈을 부르게 하는 얇은 참조(입력 장치로 옮겨 준다)."""

    def __init__(self, shared: nn.Module):
        super().__init__()
        self._shared = [shared]     # 리스트에 담아 파라미터 중복 등록을 피한다

    def forward(self, x):
        shared = self._shared[0]
        shared.to(device=x.device, dtype=x.dtype)
        return shared(x)


def shell(module: nn.Module) -> nn.Module:
    """가중치는 공유하고 모듈 딕셔너리만 복사한 껍데기."""
    copied = copy.copy(module)
    copied._modules = module._modules.copy()
    copied._parameters = module._parameters.copy()
    copied._buffers = module._buffers.copy()
    copied._forward_pre_hooks = module._forward_pre_hooks.copy()
    return copied


# 28블록 베이스에서 학습한 것을 더 깊은 Anima 에 얹을 때 쓰는 블록 계보. 표 자체는
# sam3ext/anima_lora_blocks.py 의 BLOCK_MAPPINGS 이고, 그 값은 Forge 본체가 Edit LoRA 에
# 쓰는 것과 같다(extensions-builtin/sd_forge_lora/networks.py:60-64).
#
# 그 표에는 (52,28) 같은 **축소** 키도 함께 들어 있다. 축소는 어댑터 블록을 버리므로
# (52→28 은 24개를 버린다) 여기서는 상향 세 쌍만 허용한다 — Forge 본체도 하향은 거부한다.
UPWARD_LINEAGE: frozenset[tuple[int, int]] = frozenset({(28, 40), (28, 52), (40, 52)})


def _block_plan(adapter_blocks: int, model_blocks: int) -> tuple[int, ...] | None:
    """모델 블록 j 가 쓸 어댑터 블록 번호. ``None`` 이면 지원하지 않는 조합이다."""
    if adapter_blocks == model_blocks:
        return tuple(range(model_blocks))
    if (adapter_blocks, model_blocks) not in UPWARD_LINEAGE:
        return None
    from ..anima_lora_blocks import BLOCK_MAPPINGS

    plan = BLOCK_MAPPINGS.get((adapter_blocks, model_blocks))
    if plan is None or len(plan) != model_blocks:
        return None
    return tuple(plan)


def ensure_compatible(spec, dit) -> None:
    """어댑터와 모델이 안 맞으면 숫자를 보여 주고 멈춘다."""
    if spec.unsupported:
        raise RuntimeError(
            "이 IP-Adapter 는 아직 지원하지 않는 기능을 씁니다: "
            + ", ".join(spec.unsupported)
        )
    blocks = getattr(dit, "blocks", None)
    if blocks is None or not len(blocks):
        raise RuntimeError(
            "IP-Adapter 는 Anima 전용입니다 — 이 모델에는 DiT 블록이 없습니다."
        )
    if spec.num_blocks > len(blocks):
        raise RuntimeError(
            f"어댑터는 블록 {spec.num_blocks} 개용인데 지금 모델은 {len(blocks)} 개입니다."
        )
    if _block_plan(spec.num_blocks, len(blocks)) is None:
        raise RuntimeError(
            f"어댑터는 블록 {spec.num_blocks} 개용인데 지금 모델은 {len(blocks)} 개이고, "
            "이 조합의 ANIMA 블록 계보 매핑이 없습니다."
        )
    model_dim = int(getattr(blocks[0], "x_dim", 0) or 0)
    if model_dim and spec.inner_dim != model_dim:
        raise RuntimeError(
            f"어댑터 차원은 {spec.inner_dim} 인데 지금 모델은 {model_dim} 입니다. "
            "다른 Anima 베이스용 어댑터입니다."
        )


def _linear(in_features, out_features, *, bias, device, dtype) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=bias).to(device=device, dtype=dtype)


def _copy_into(module: nn.Module, weights, prefix: str) -> int:
    """``prefix`` 로 시작하는 가중치를 옮기고 몇 개를 옮겼는지 준다.

    개수를 돌려주는 이유: 하나도 못 옮기면 그 모듈은 랜덤 초기화로 남는데, adaln 게이트의
    bias 가 0.1 이라 **무동작이 아니라 랜덤 투영이 잔차에 더해진다**. 부르는 쪽이 멈춰야 한다.
    """
    copied = 0
    for name, param in module.named_parameters():
        key = f"{prefix}.{name}"
        if key in weights:
            param.data.copy_(weights[key].to(dtype=param.dtype, device=param.device))
            copied += 1
    return copied


def _apply_lora(block, index: int, weights) -> None:
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in weights.items():
        match = _LORA_RE.match(key)
        if match is None or int(match.group(1)) != index:
            continue
        grouped.setdefault(match.group(2), {})[match.group(3)] = value
    for name, pair in grouped.items():
        if "A" not in pair or "B" not in pair:
            continue
        base = getattr(block.cross_attn, name, None)
        if not isinstance(base, nn.Linear):
            continue
        setattr(block.cross_attn, name, _LoRALinear(base, pair["A"], pair["B"]))


def _patch_forward(block) -> bool:
    if getattr(block, "_sam3_ip_patched", False):
        return False

    def capture(_module, args):
        block._sam3_ip_x = args[0]

    block.cross_attn.register_forward_pre_hook(capture)
    original = block.forward

    def forward(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, **kwargs):
        options = kwargs.get("transformer_options") or {}
        result = original(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, **kwargs)
        x_cross = getattr(block, "_sam3_ip_x", None)
        block._sam3_ip_x = None          # 활성값을 붙잡아 두지 않는다
        tokens = options.get(TOKENS_KEY)
        if tokens is None or x_cross is None:
            return result

        batch, frames, height, width, _ = x_B_T_H_W_D.shape
        heads = block.cross_attn.n_heads
        head_dim = block.cross_attn.head_dim
        tokens = tokens.to(device=x_cross.device, dtype=x_cross.dtype)
        if tokens.shape[0] != batch:
            tokens = tokens.expand(batch, -1, -1)

        query = block.cross_attn.q_proj(x_cross)
        query = query.reshape(batch, -1, heads, head_dim).permute(0, 2, 1, 3)
        query = block.cross_attn.q_norm(query)
        key = block.ip_k_proj(tokens).reshape(batch, -1, heads, head_dim).permute(0, 2, 1, 3)
        value = block.ip_v_proj(tokens).reshape(batch, -1, heads, head_dim).permute(0, 2, 1, 3)
        key_norm = getattr(block, "ip_k_norm", None)
        if key_norm is not None:
            key = key_norm(key)

        attended = F.scaled_dot_product_attention(query, key, value)
        ip_out = attended.permute(0, 2, 1, 3).reshape(
            batch, frames * height * width, heads * head_dim
        )

        gate = block.adaln_ip(emb_B_T_D) * float(getattr(block, "sam3_ip_gate_scale", 1.0))
        # 토큰이 통째로 0 이면(= null 이 0 인 경우) 기여를 끈다.
        live = (tokens.abs().sum(dim=(1, 2)) > 1e-6).to(gate.dtype).reshape(batch, 1, 1)
        gate = gate * live
        return result + gate.reshape(batch, frames, 1, 1, -1) * ip_out.reshape(
            batch, frames, height, width, -1
        )

    block.forward = forward
    block._sam3_ip_patched = True
    return True


def install(dit, spec, weights, *, use_lora: bool = True, gate_scale: float = 1.0) -> int:
    """블록마다 ip_k_proj/ip_v_proj/adaln_ip 를 만들고 forward 를 감싼다. 패치한 블록 수를 준다."""
    parameters = list(dit.parameters())
    device = parameters[0].device if parameters else torch.device("cpu")
    dtype = parameters[0].dtype if parameters else torch.float32

    shared_k = shared_v = None
    if spec.shared_projection:
        shared_k = _SharedProj(spec.embed_dim, spec.inner_dim).to(device=device, dtype=dtype)
        shared_v = _SharedProj(spec.embed_dim, spec.inner_dim).to(device=device, dtype=dtype)
        _copy_into(shared_k, weights, "shared_ip_k_proj")
        _copy_into(shared_v, weights, "shared_ip_v_proj")
        dit.sam3_shared_ip_k_proj = shared_k
        dit.sam3_shared_ip_v_proj = shared_v

    plan = _block_plan(spec.num_blocks, len(dit.blocks))
    if plan is None:
        raise RuntimeError(
            f"어댑터는 블록 {spec.num_blocks} 개용인데 지금 모델은 {len(dit.blocks)} 개이고, "
            "이 조합의 ANIMA 블록 계보 매핑이 없습니다."
        )
    if len(plan) != len(dit.blocks):
        raise RuntimeError(
            f"블록 계보 매핑의 길이가 {len(plan)} 인데 모델 블록은 {len(dit.blocks)} 개입니다."
        )

    patched = 0
    for index, block in enumerate(dit.blocks):
        if getattr(block, "_sam3_ip_patched", False):
            continue
        # 깊은 모델에서는 여러 블록이 같은 어댑터 블록을 쓴다(28→52 는 12개가 3번씩).
        source = plan[index]

        if spec.shared_projection:
            block.ip_k_proj = _SharedRef(shared_k)
            block.ip_v_proj = _SharedRef(shared_v)
        else:
            block.ip_k_proj = _linear(
                spec.embed_dim, spec.inner_dim, bias=True, device=device, dtype=dtype
            )
            block.ip_v_proj = _linear(
                spec.embed_dim, spec.inner_dim, bias=True, device=device, dtype=dtype
            )
            nn.init.normal_(block.ip_k_proj.weight, std=1.0 / math.sqrt(spec.embed_dim))
            nn.init.zeros_(block.ip_k_proj.bias)
            nn.init.normal_(block.ip_v_proj.weight, std=1.0 / math.sqrt(spec.embed_dim))
            nn.init.zeros_(block.ip_v_proj.bias)
            copied = _copy_into(block.ip_k_proj, weights, f"blocks.{source}.ip_k_proj")
            copied += _copy_into(block.ip_v_proj, weights, f"blocks.{source}.ip_v_proj")
            if not copied:
                raise RuntimeError(
                    f"어댑터에 blocks.{source} 의 K/V 가중치가 없습니다 — "
                    "비어 있는 블록이 있는 어댑터는 쓸 수 없습니다."
                )

        adaln = nn.Sequential(
            nn.SiLU(),
            _linear(spec.inner_dim, spec.inner_dim, bias=True, device=device, dtype=dtype),
        )
        nn.init.zeros_(adaln[1].weight)
        nn.init.constant_(adaln[1].bias, 0.1)
        block.adaln_ip = adaln
        if not _copy_into(block.adaln_ip, weights, f"blocks.{source}.adaln_ip"):
            raise RuntimeError(
                f"어댑터에 blocks.{source} 의 게이트(adaln_ip)가 없습니다 — "
                "비어 있는 블록이 있는 어댑터는 쓸 수 없습니다."
            )

        if spec.norm_keys:
            block.ip_k_norm = nn.LayerNorm(
                block.cross_attn.head_dim, elementwise_affine=False, eps=1e-6
            ).to(device=device, dtype=dtype)

        block.sam3_ip_gate_scale = float(gate_scale)
        if use_lora and source in spec.lora_blocks:
            _apply_lora(block, source, weights)

        patched += int(_patch_forward(block))
    return patched


def cfg_correction(denoised, cond_with_ip, cond_without_ip, *, cond_scale, ip_cfg_scale):
    """독립 IP CFG 보정.

    기본 CFG 는 ``uncond + cfg * (cond_with_ip - uncond)`` 를 준다. 우리가 원하는 것은
    ``uncond + cfg * (cond_without_ip - uncond) + (ip - 1) * (cond_with_ip - cond_without_ip)``
    이므로 차이만큼 더한다.
    """
    factor = float(ip_cfg_scale) - 1.0 - float(cond_scale)
    return denoised + factor * (cond_with_ip - cond_without_ip)


class InjectionHandler:
    """unet 래퍼 + post-CFG. cond 에는 실제 토큰, uncond 에는 null 토큰을 싣는다."""

    def __init__(self, injection: Injection):
        self.injection = injection
        self._cond_without_ip = None

    def __call__(self, apply_model, args):
        model_input = args["input"]
        timestep = args["timestep"]
        conditioning = dict(args["c"])
        cond_or_uncond = args.get("cond_or_uncond")

        batch = model_input.shape[0]
        device = model_input.device
        tokens = self.injection.tokens.to(device=device)
        null_tokens = self.injection.null_tokens.to(device=device)
        batched = tokens.expand(batch, -1, -1).clone()
        if cond_or_uncond is not None:
            for index, is_uncond in enumerate(cond_or_uncond):
                if index < batch and is_uncond:
                    batched[index] = null_tokens[0]

        options = dict(conditioning.get("transformer_options") or {})
        options[TOKENS_KEY] = batched
        conditioning["transformer_options"] = options

        if not self.injection.separate_cfg:
            return apply_model(model_input, timestep, **conditioning)

        output = apply_model(model_input, timestep, **conditioning)

        # IP 없는 예측을 한 번 더 — post_cfg 에서 둘의 차이를 따로 스케일한다.
        without = dict(conditioning)
        without_options = dict(options)
        without_options[TOKENS_KEY] = null_tokens.expand(batch, -1, -1)
        without["transformer_options"] = without_options
        baseline = apply_model(model_input, timestep, **without)

        order = list(cond_or_uncond or [0])
        chunks = baseline.chunk(len(order)) if order else (baseline,)
        for index, is_uncond in enumerate(order):
            if not is_uncond and index < len(chunks):
                self._cond_without_ip = chunks[index]
                break
        return output

    def post_cfg(self, args):
        if not self.injection.separate_cfg or self._cond_without_ip is None:
            return args["denoised"]
        with_ip = args["cond_denoised"]
        without = self._cond_without_ip.to(device=with_ip.device, dtype=with_ip.dtype)
        return cfg_correction(
            args["denoised"], with_ip, without,
            cond_scale=args["cond_scale"], ip_cfg_scale=self.injection.cfg_scale,
        )

    def to(self, *args, **kwargs):
        return self


@contextmanager
def patched_unet(sd_model, spec, weights, injection: Injection, *,
                 use_lora: bool = True) -> Iterator[None]:
    """샘플링 동안만 주입된 패처를 끼워 넣는다.

    ``sample()`` 이 첫머리에서 복사해 가는 ``forge_objects_after_applying_lora.unet`` 을 잠시
    바꾼다(modules/processing.py:1542). 원본 패처와 원본 DiT 는 건드리지 않으므로, 빠져나올 때
    참조 하나만 되돌리면 흔적이 남지 않는다.
    """
    objects = sd_model.forge_objects_after_applying_lora
    original = objects.unet
    dit = original.get_model_object("diffusion_model")
    ensure_compatible(spec, dit)

    patched_model = original.clone()
    copied = shell(dit)
    copied.blocks = nn.ModuleList([shell(block) for block in dit.blocks])
    for block in copied.blocks:
        block.cross_attn = shell(block.cross_attn)
    install(copied, spec, weights, use_lora=use_lora, gate_scale=injection.gate_scale)

    handler = InjectionHandler(injection)
    patched_model.add_object_patch("diffusion_model", copied)
    patched_model.set_model_unet_function_wrapper(handler)
    patched_model.set_model_sampler_post_cfg_function(handler.post_cfg)

    objects.unet = patched_model
    try:
        yield
    finally:
        objects.unet = original
