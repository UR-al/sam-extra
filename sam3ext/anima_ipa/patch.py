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

from .options import (  # noqa: F401 - 예전처럼 patch 에서도 이 이름들을 쓸 수 있게 다시 내보낸다
    DEFAULT_DUPLICATE_POLICY,
    DUPLICATE_POLICIES,
    OPT_DUPLICATE_POLICY,
    duplicate_policy_from_opts,
    normalize_duplicate_policy,
)

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
    """가중치는 공유하고 모듈 딕셔너리만 복사한 껍데기.

    다른 확장이 원본에 얹어 둔 **인스턴스** ``forward``(Safe PAG 의 블록 래퍼, Anima 3.8B 런타임의
    DiT 래퍼)는 원본에 바인딩된 메서드를 닫아 두고 있다. 그대로 따라오면 껍데기를 불러도 원본 블록이
    돌아 껍데기 cross_attn 의 훅이 한 번도 안 불리고 주입이 조용히 빠진다. 껍데기는 클래스 forward 를
    쓴다 — anima38 런타임이 "항상 클래스의 순정 forward" 를 부르는 것과 같은 이유다.
    """
    copied = copy.copy(module)
    copied.__dict__.pop("forward", None)
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


# 28블록 어댑터를 더 깊은 모델에 얹을 때 복제 블록을 어떻게 다룰지(lineage/all/split)의 뜻과 기본값은
# options.py 에 있다 — UI·설정 등록·요청 조립이 torch 없이 쓰기 때문이다.


def _duplicate_gates(plan: tuple[int, ...], policy: str) -> list[float | None]:
    """모델 블록마다 게이트 배율. None 이면 그 블록에는 IPA 를 걸지 않는다.

    ``lineage`` 는 각 어댑터 블록의 첫 번째 대응 블록만 1.0, 끼워 넣은 복제는 None.
    ``split`` 은 같은 어댑터 블록을 쓰는 n 개가 1/n 씩. ``all`` 은 모두 1.0.
    """
    counts: dict[int, int] = {}
    for source in plan:
        counts[source] = counts.get(source, 0) + 1
    seen: set[int] = set()
    gates: list[float | None] = []
    for source in plan:
        first = source not in seen
        seen.add(source)
        if policy == "lineage":
            gates.append(1.0 if first else None)
        elif policy == "split":
            gates.append(1.0 / counts[source])
        else:
            gates.append(1.0)
    return gates


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
        block._sam3_ip_ran = True        # 껍데기 블록이 돌았다는 표시 — 오류 힌트를 가르는 데 쓴다
        x_cross = getattr(block, "_sam3_ip_x", None)
        block._sam3_ip_x = None          # 활성값을 붙잡아 두지 않는다
        tokens = options.get(TOKENS_KEY)
        if tokens is None:
            return result
        if x_cross is None:
            # 토큰은 왔는데 훅이 안 불렸다 — forward 가 이 껍데기의 cross_attn 을 거치지 않았다.
            # 조용히 주입 없는 결과를 내놓으면 사용자는 "참조가 안 먹는 IP-Adapter" 만 본다.
            raise RuntimeError(
                "IP 주입이 실행되지 않았습니다 — 블록 forward 가 껍데기 cross_attn 을 거치지 "
                "않았습니다(다른 확장의 forward 패치가 원본 블록에 남아 있을 수 있습니다)."
            )
        block._sam3_ip_applied = True    # 한 번이라도 주입했다는 표시 — InjectionHandler 가 본다

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


def install(dit, spec, weights, *, use_lora: bool = True, gate_scale: float = 1.0,
            duplicate_policy: str = DEFAULT_DUPLICATE_POLICY) -> int:
    """블록마다 ip_k_proj/ip_v_proj/adaln_ip 를 만들고 forward 를 감싼다. 패치한 블록 수를 준다.

    ``duplicate_policy`` 는 어댑터보다 깊은 모델에서 복제 블록을 어떻게 다룰지(:data:`DUPLICATE_POLICIES`).
    블록 수가 같으면 복제가 없으므로 어느 값이든 결과가 같다.
    """
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

    gates = _duplicate_gates(plan, normalize_duplicate_policy(duplicate_policy) or DEFAULT_DUPLICATE_POLICY)

    patched = 0
    for index, block in enumerate(dit.blocks):
        if getattr(block, "_sam3_ip_patched", False):
            continue
        gate = gates[index]
        if gate is None:
            continue   # lineage: 끼워 넣은 복제 블록 — IPA 없음(어댑터 LoRA 도 걸지 않는다)
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

        block.sam3_ip_gate_scale = float(gate_scale) * float(gate)
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
    """unet 래퍼 + post-CFG. cond 에는 실제 토큰, uncond 에는 null 토큰을 싣는다.

    ``blocks`` 를 주면 첫 호출 뒤 그중 하나라도 실제로 주입했는지 확인한다 — DiT forward 가
    껍데기 블록을 아예 거치지 않는 배선이면 블록 쪽 검사가 불릴 기회조차 없기 때문이다.
    """

    def __init__(self, injection: Injection, blocks=None):
        self.injection = injection
        self._cond_without_ip = None
        self._blocks = blocks
        self._verified = blocks is None

    def _verify_injected(self) -> None:
        if self._verified:
            return
        if not any(getattr(block, "_sam3_ip_applied", False) for block in self._blocks):
            # 원본 DiT 의 인스턴스 forward 래퍼는 _keep_foreign_dit_forward 가 이미 살려 두므로
            # 그것을 원인으로 지목하지 않는다. 껍데기 블록이 돌았는지로 두 경우를 가른다.
            if any(getattr(block, "_sam3_ip_ran", False) for block in self._blocks):
                raise RuntimeError(
                    "IP 주입이 실행되지 않았습니다 — 껍데기 블록은 돌았지만 transformer_options 에 "
                    f"IP 토큰({TOKENS_KEY})이 실려 오지 않았습니다(중간의 다른 확장 unet/DiT 래퍼가 "
                    "transformer_options 를 새 딕셔너리로 바꿔 넘겼을 수 있습니다)."
                )
            raise RuntimeError(
                "IP 주입이 실행되지 않았습니다 — 샘플러가 부른 모델이 IP-Adapter 껍데기 블록을 한 번도 "
                "돌지 않았습니다(diffusion_model 객체 패치가 적용되지 않았거나, 다른 확장이 원본 "
                "DiT·블록을 직접 부르고 있을 수 있습니다)."
            )
        self._verified = True

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
            output = apply_model(model_input, timestep, **conditioning)
            self._verify_injected()
            return output

        output = apply_model(model_input, timestep, **conditioning)
        self._verify_injected()

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


def _keep_foreign_dit_forward(dit, copied) -> bool:
    """원본 DiT 에 다른 확장의 인스턴스 forward(Anima 3.8B 런타임·NegPiP 의 DiT 래퍼)가 있으면
    껍데기가 그 체인을 그대로 부르되, 도는 동안만 원본의 ``blocks`` 를 껍데기 블록으로 바꿔 둔다.

    그 래퍼들은 원본에 바인딩된 메서드를 닫아 두고 있어 껍데기가 물려받으면 원본 블록이 돈다
    (주입 무효). 반대로 :func:`shell` 처럼 그냥 버리면 3.8B v2 커넥터의 조건 확장이 껍데기에서
    빠진다. 체인은 결국 클래스 forward 의 ``for block in self.blocks`` 에 닿으므로, 그 동안
    ``blocks`` 만 바꾸면 둘 다 산다. 호출이 끝나면(예외여도) 원본 ``blocks`` 를 되돌린다.

    외부 forward 는 세션 시작 때 고정해 두지 않고 **호출마다** 원본에서 다시 찾는다. 도중에
    래퍼가 풀리면(NegPiP 의 reset 등) 낡은 클로저 대신 껍데기의 클래스 forward 로 돌고, 도중에
    생기거나 바뀌면 지금 것을 따른다. 외부 forward 가 없을 때의 결과는 클래스 forward 그대로다.
    돌려주는 값은 설치 시점에 외부 forward 가 있었는지다.
    """
    shell_blocks = copied.blocks
    native = type(copied).forward

    def forward(*args, **kwargs):
        foreign = dit.__dict__.get("forward")
        if foreign is None:
            return native(copied, *args, **kwargs)
        original_blocks = dit.blocks
        dit.blocks = shell_blocks
        try:
            return foreign(*args, **kwargs)
        finally:
            dit.blocks = original_blocks

    copied.forward = forward
    return "forward" in dit.__dict__


def _undo_object_patches(patcher) -> None:
    """이 복제 패처가 모델에 걸어 둔 객체 패치(``diffusion_model`` = 껍데기 DiT)를 원래대로 되돌린다.

    Forge 는 같은 모델의 다른 복제를 올릴 때 이전 복제를 ``detach(unpatch_all=False)`` 로만 떼어 내고
    (backend/memory_management.py load_models_gpu) 객체 패치는 되돌리지 않는다. 복제들은 모델 객체와
    ``object_patches_backup`` 을 공유하므로, 그대로 두면 IPA 가 끝난 뒤에도 원본 KModel 의 diffusion_model 이
    껍데기로 남는다. 그러면 다음 생성에서 LoRA 키가 맞지 않아 Forge 가 LoRA 를 통째로 버리고
    (``[LORA] Mismatch``), IPA 어댑터의 cross_attn LoRA 래퍼도 남아 결과가 달라졌다(2026-09-23 GPU 확인).
    가중치 패치는 원본 패처와 같은 것이라 건드리지 않는다.
    """
    backup = getattr(patcher, "object_patches_backup", None)
    patches = getattr(patcher, "object_patches", None)
    model = getattr(patcher, "model", None)
    if not backup or not patches or model is None:
        return
    for key in list(patches):
        if key not in backup:
            continue
        original = backup.pop(key)
        try:
            from backend import utils as forge_utils

            forge_utils.set_attr_raw(model, key, original)
        except ImportError:  # Forge 밖(테스트) — 점 없는 이름만 쓴다
            setattr(model, key, original)


@contextmanager
def patched_unet(sd_model, spec, weights, injection: Injection, *,
                 use_lora: bool = True, duplicate_policy: str | None = None) -> Iterator[None]:
    """샘플링 동안만 주입된 패처를 끼워 넣는다.

    ``sample()`` 이 첫머리에서 복사해 가는 ``forge_objects_after_applying_lora.unet`` 을 잠시
    바꾼다(modules/processing.py:1542). 원본 패처와 원본 DiT 는 건드리지 않으므로, 빠져나올 때
    참조 하나만 되돌리면 흔적이 남지 않는다. (예외: 원본 DiT 에 다른 확장의 forward 래퍼가
    있으면 :func:`_keep_foreign_dit_forward` 가 호출 동안만 원본의 ``blocks`` 를 바꿔 둔다.)

    ``duplicate_policy`` 가 None 이면 Forge 설정(``sam3_ipa_duplicate_policy``)을 읽는다.
    """
    if duplicate_policy is None:
        duplicate_policy = duplicate_policy_from_opts()
    objects = sd_model.forge_objects_after_applying_lora
    original = objects.unet
    dit = original.get_model_object("diffusion_model")
    ensure_compatible(spec, dit)

    patched_model = original.clone()
    copied = shell(dit)
    copied.blocks = nn.ModuleList([shell(block) for block in dit.blocks])
    for block in copied.blocks:
        block.cross_attn = shell(block.cross_attn)
    install(copied, spec, weights, use_lora=use_lora, gate_scale=injection.gate_scale,
            duplicate_policy=duplicate_policy)
    _keep_foreign_dit_forward(dit, copied)

    handler = InjectionHandler(injection, blocks=copied.blocks)
    patched_model.add_object_patch("diffusion_model", copied)
    patched_model.set_model_unet_function_wrapper(handler)
    patched_model.set_model_sampler_post_cfg_function(handler.post_cfg)

    objects.unet = patched_model
    try:
        yield
    finally:
        objects.unet = original
        _undo_object_patches(patched_model)
