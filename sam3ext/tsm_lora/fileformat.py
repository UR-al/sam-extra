"""TSM 파일 형식 — 헤더 읽기, 키 판별, 검증, 파일 캐시.

LyCORIS 3.2 ``TSMModule`` 을 ``ss_saved_format=tsm-native`` 로 저장한 파일의 모듈 하나(``lora_unet_blocks_3_self_attn_q_proj``)
는 이 키들을 가진다::

    <모듈>.experts.<s>.<i>.down.weight   [r, in]    s = 스케일(0 = 코어), i = 그 스케일의 전문가
    <모듈>.experts.<s>.<i>.up.weight     [out, r]   LyCORIS 가 저장 전에 scalar 를 곱해 둔다(읽으면 scalar 1)
    <모듈>.router_fc.weight              [n_ctx, in] (라우터 입력 input) 또는 [n_ctx, r] (bottleneck)
    <모듈>.router_fc.bias                [n_ctx]
    <모듈>.timestep_embed.weight         [T, n_ctx]
    <모듈>.alpha                         0차원

``n_scales`` 는 전문가 키에서(스케일별 전문가 수), ``T`` 는 ``timestep_embed`` 의 행 수, 라우터 입력 방식은
``router_fc`` 의 너비에서 읽는다. 단계(1 = 전문가 합, 2 = 라우터 게이트)는 키에 남지 않고 메타데이터
``ss_network_args.tsm_stage`` 에만 있다 — 없으면 라우터가 0 이 아닌 파일만 2단계로 본다(1단계 파일의 라우터는 정확히 0).

전문가가 LoKr 인 TSM-LoKr 파일(``ss_lycoris_algo=tsm-lokr``)은 :mod:`.lokr_format` 이 읽는다. 여기의 판별·분리·캐시 함수
(``is_tsm_state_dict`` · ``split_state_dict`` · ``tsm_file_from_state_dict`` · ``load_tsm_file``)는 두 종류를 모두 다룬다.
"""
from __future__ import annotations

import json
import math
import os
import re
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch


TSM_SAVED_FORMAT = "tsm-native"
TSM_ALGO = "tsm"
DEFAULT_NUM_TIMESTEPS = 1000
ROUTER_INPUT = "input"
ROUTER_BOTTLENECK = "bottleneck"

# 학습기가 TSM 에 넘긴 σ·1000 의 dtype. sd-scripts 는 timesteps 를 weight_dtype 으로 돌려준다
# (flux_train_utils.get_noisy_model_input_and_timesteps: ``timesteps.to(dtype=weight_dtype)``) — mixed_precision bf16 이면
# bf16 으로 반올림된 값을 lycoris_algo_fixes.set_timestep 이 round 한다. 그래서 512 이상에서는 t 가 4 단위로만 나오고,
# 2단계의 timestep_embed 는 그 행만 학습된다(나머지는 0 그대로).
TIMESTEP_FP32 = "fp32"
TIMESTEP_BF16 = "bf16"
TIMESTEP_FP16 = "fp16"
TIMESTEP_DTYPES = (TIMESTEP_FP32, TIMESTEP_BF16, TIMESTEP_FP16)

STAGE_FROM_METADATA = "metadata"
STAGE_FROM_ROUTER = "router"          # 메타데이터 없음 — 0 이 아닌 라우터가 있어 2단계
STAGE_FROM_ZERO_ROUTER = "zero-router"  # 메타데이터 없음 — 라우터가 모두 0 이라 1단계

_EXPERT_RE = re.compile(r"^(?P<mod>.+?)\.experts\.(?P<s>\d+)\.(?P<i>\d+)\.(?P<w>up|down)\.weight$")
_AUX_SUFFIXES = ("alpha", "router_fc.weight", "router_fc.bias", "timestep_embed.weight")
_MAX_HEADER_BYTES = 256 * 1024 * 1024


class TsmFormatError(ValueError):
    """TSM 파일을 읽을 수 없다. 메시지는 한국어 — 영어 순서로 한 줄."""

    def __init__(self, korean: str, english: str) -> None:
        self.korean = korean
        self.english = english
        super().__init__(f"{korean} / {english}")


def expert_index(t: int, n: int, num_timesteps: int = DEFAULT_NUM_TIMESTEPS) -> int:
    """tsm.py ``_get_expert_index``: ``ceil(t / T * n)`` (파이썬 float 식 — 정수 ceil 이면 n=25·50·100 에서 갈린다)을
    [1, n] 로 자르고 0부터 센다."""
    i = math.ceil(t / num_timesteps * n)
    return max(1, min(i, n)) - 1


# ── 헤더 ──

def read_header(path: str | os.PathLike) -> tuple[dict[str, str], dict[str, dict]]:
    """safetensors 헤더만 읽는다(텐서는 읽지 않는다). (메타데이터, {키: {"dtype", "shape", ...}})."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                raise TsmFormatError("safetensors 헤더가 없습니다", "missing safetensors header")
            (size,) = struct.unpack("<Q", raw)
            if size <= 0 or size > _MAX_HEADER_BYTES:
                raise TsmFormatError(f"safetensors 헤더 길이가 이상합니다({size})", f"bad safetensors header size {size}")
            header = json.loads(fh.read(size).decode("utf-8"))
    except TsmFormatError:
        raise
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise TsmFormatError(
            f"safetensors 헤더를 읽지 못했습니다: {exc}", f"could not read the safetensors header: {exc}"
        ) from exc
    if not isinstance(header, dict):
        raise TsmFormatError("safetensors 헤더가 dict 가 아닙니다", "safetensors header is not a JSON object")
    meta = header.pop("__metadata__", None) or {}
    return {str(k): str(v) for k, v in meta.items()}, header


def network_args(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """``ss_network_args`` (JSON 문자열 또는 dict) → dict. 없거나 깨졌으면 빈 dict."""
    raw = (metadata or {}).get("ss_network_args")
    if isinstance(raw, Mapping):
        return dict(raw)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def metadata_algo(metadata: Mapping[str, Any] | None) -> str | None:
    """메타데이터에 적힌 LyCORIS algo(소문자). 없으면 None."""
    meta = metadata or {}
    algo = meta.get("ss_lycoris_algo") or network_args(meta).get("algo")
    text = str(algo or "").strip().lower()
    return text or None


def metadata_says_tsm(metadata: Mapping[str, Any] | None) -> bool:
    """메타데이터가 TSM(LoRA 전문가) 또는 TSM-LoKr 파일이라고 하는가."""
    meta = metadata or {}
    if str(meta.get("ss_saved_format", "")).strip().lower() == TSM_SAVED_FORMAT:
        return True
    if metadata_algo(meta) == TSM_ALGO:
        return True
    from .lokr_format import metadata_says_tsm_lokr

    return metadata_says_tsm_lokr(meta)


def metadata_may_hide_tsm(metadata: Mapping[str, Any] | None) -> bool:
    """메타데이터만으로는 TSM 이 아니라고 단정할 수 없는가(키를 봐야 하는가).

    kohya 순정 LoRA(``networks.lora`` 등)나 다른 algo 를 적은 LyCORIS 파일은 False. 메타데이터가 없거나 LyCORIS 인데
    algo 가 없으면 True.
    """
    meta = metadata or {}
    if metadata_says_tsm(meta):
        return True
    if metadata_algo(meta) is not None:
        return False
    module = str(meta.get("ss_network_module", "")).strip().lower()
    return not module or "lycoris" in module


# ── 키 판별 ──

def tsm_module_names(keys: Iterable[str]) -> set[str]:
    """전문가 키(``experts.S.I.up/down.weight``)와 라우터 또는 시간 임베딩 키를 함께 가진 모듈 이름."""
    experts: set[str] = set()
    aux: set[str] = set()
    for key in keys:
        match = _EXPERT_RE.match(key)
        if match is not None:
            experts.add(match.group("mod"))
            continue
        for suffix in ("router_fc.weight", "timestep_embed.weight"):
            if key.endswith("." + suffix):
                aux.add(key[: -len(suffix) - 1])
    return experts & aux


def is_tsm_state_dict(sd_or_keys: Mapping[str, Any] | Iterable[str]) -> bool:
    """TSM(LoRA 전문가) 또는 TSM-LoKr 키가 있는가."""
    from .lokr_format import tsm_lokr_module_names

    keys = list(sd_or_keys.keys() if isinstance(sd_or_keys, Mapping) else sd_or_keys)
    return bool(tsm_module_names(keys)) or bool(tsm_lokr_module_names(keys))


def _module_of(key: str, modules: set[str]) -> str | None:
    match = _EXPERT_RE.match(key)
    if match is not None:
        return match.group("mod") if match.group("mod") in modules else None
    for suffix in _AUX_SUFFIXES:
        if key.endswith("." + suffix):
            name = key[: -len(suffix) - 1]
            return name if name in modules else None
    return None


def split_state_dict(sd: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(TSM 모듈의 키, 나머지 키). TSM 과 일반 LoRA 키가 섞인 파일에서 나머지는 Forge 에 그대로 넘긴다.

    TSM-LoKr 모듈은 그 모듈의 ``alpha`` · ``dora_scale`` · 라우터 키까지 떼어 Forge 가 보지 않게 한다."""
    from . import lokr_format as lf

    modules = tsm_module_names(sd.keys())
    lokr_modules = lf.tsm_lokr_module_names(sd.keys())
    tsm: dict[str, Any] = {}
    rest: dict[str, Any] = {}
    for key, value in sd.items():
        claimed = (modules and _module_of(key, modules) is not None) or (
            lokr_modules and lf.module_of(key, lokr_modules) is not None
        )
        (tsm if claimed else rest)[key] = value
    return tsm, rest


# ── 파일 표현 ──

@dataclass(eq=False)
class TsmModule:
    """대상 Linear 하나의 전문가 묶음(CPU 텐서, 파일 dtype 그대로)."""

    kind = "lora"   # TSM-LoKr 는 lokr_format.TsmLokrModule (kind "lokr")

    name: str
    n_scales: tuple[int, ...]
    rank: int
    in_features: int
    out_features: int
    scale: float                              # alpha / rank (alpha 0 → rank)
    num_timesteps: int
    downs: tuple[tuple[torch.Tensor, ...], ...]  # [스케일][전문가] → [r, in]
    ups: tuple[tuple[torch.Tensor, ...], ...]    # [스케일][전문가] → [out, r]
    router_w: torch.Tensor | None = None      # [n_ctx, in] 또는 [n_ctx, r]
    router_b: torch.Tensor | None = None      # [n_ctx]
    embed: torch.Tensor | None = None         # [T, n_ctx]
    router_input: str = ROUTER_INPUT
    timestep_dtype: str = TIMESTEP_FP32       # 학습 때 σ·1000 이 거친 dtype(:func:`timestep_dtype_from_metadata`)

    @property
    def n_ctx(self) -> int:
        return len(self.n_scales) - 1

    @property
    def has_router(self) -> bool:
        return self.router_w is not None and self.router_b is not None and self.embed is not None

    def router_is_zero(self) -> bool:
        if not self.has_router:
            return True
        return all(bool((tensor == 0).all()) for tensor in (self.router_w, self.router_b, self.embed))

    def expert_indices(self, t: int) -> tuple[int, ...]:
        return tuple(expert_index(t, n, self.num_timesteps) for n in self.n_scales)

    def fused_bytes(self, element_size: int = 2) -> int:
        """한 시간 구간의 활성 전문가(스케일마다 하나)를 이어 붙인 down·up 의 크기."""
        width = self.rank * len(self.n_scales)
        return width * (self.in_features + self.out_features) * element_size


@dataclass(eq=False)
class TsmFile:
    path: str
    name: str
    modules: dict[str, TsmModule]
    stage: int
    stage_source: str
    n_scales: tuple[int, ...]             # 가장 많은 모듈의 구성(모듈마다 다를 수 있다 — mixed_scales)
    rank: int
    num_timesteps: int
    router_input: str
    metadata: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    mixed_scales: bool = False
    timestep_dtype: str = TIMESTEP_FP32

    expert_kind = "lora"   # TSM-LoKr 파일은 lokr_format.TsmLokrFile ("lokr")

    def scales_label(self) -> str:
        return "+".join(str(n) for n in self.n_scales) + ("*" if self.mixed_scales else "")

    def kind_label(self) -> str:
        """전문가 종류: LoRA 는 ``r8``, TSM-LoKr 는 ``lokr f4 dora``."""
        return f"r{self.rank}"

    def summary(self) -> str:
        """infotext·로그용 한 줄: ``s1 8+1 r8`` (스케일 구성이 모듈마다 다르면 ``8+1*``, TSM-LoKr 는 ``s1 8+1 lokr f4 dora``)."""
        return f"s{self.stage} {self.scales_label()} {self.kind_label()}"

    def fused_bytes(self, element_size: int = 2) -> int:
        return sum(module.fused_bytes(element_size) for module in self.modules.values())


def timestep_dtype_from_metadata(metadata: Mapping[str, Any] | None) -> str:
    """``ss_mixed_precision`` → 학습 때 TSM 이 받은 σ·1000 의 dtype. 적혀 있지 않으면 fp32(반올림 없음).

    sd-scripts ``prepare_dtype``: bf16 → bfloat16, fp16 → float16, 그 밖(no) → float32. fp8_base 여도 timesteps 는
    weight_dtype(위 값)으로 온다.
    """
    text = str((metadata or {}).get("ss_mixed_precision", "") or "").strip().lower()
    if text in ("bf16", "bfloat16"):
        return TIMESTEP_BF16
    if text in ("fp16", "float16", "half"):
        return TIMESTEP_FP16
    return TIMESTEP_FP32


def _stage_from_metadata(args: Mapping[str, Any]) -> int | None:
    raw = args.get("tsm_stage")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        stage = int(str(raw).strip())
    except ValueError:
        stage = -1
    if stage not in (1, 2):
        raise TsmFormatError(
            f"메타데이터의 tsm_stage 값 {raw!r} 을 모릅니다(1 또는 2)", f"unknown tsm_stage {raw!r} in metadata (expected 1 or 2)"
        )
    return stage


def _parse_n_scales(raw: Any) -> tuple[int, ...] | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        text = raw.strip()
        try:
            value = json.loads(text)
        except ValueError:
            value = [part for part in re.split(r"[,\s\[\]]+", text) if part]
    else:
        value = raw
    if isinstance(value, (int, float)):
        value = [value]
    try:
        return tuple(int(v) for v in value)
    except (TypeError, ValueError):
        return None


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in getattr(tensor, "shape", ()))


def _build_module(name: str, items: dict[str, Any], router_hint: str | None) -> TsmModule:
    experts: dict[int, dict[int, dict[str, Any]]] = {}
    prefix = name + "."
    for key, value in items.items():
        match = _EXPERT_RE.match(key)
        if match is not None:
            experts.setdefault(int(match.group("s")), {}).setdefault(int(match.group("i")), {})[match.group("w")] = value
    if sorted(experts) != list(range(len(experts))):
        raise TsmFormatError(
            f"{name}: 스케일 번호가 0 부터 이어지지 않습니다({sorted(experts)})",
            f"{name}: scale indices are not contiguous from 0 ({sorted(experts)})",
        )
    n_scales: list[int] = []
    downs: list[tuple[torch.Tensor, ...]] = []
    ups: list[tuple[torch.Tensor, ...]] = []
    rank = in_features = out_features = None
    for s in range(len(experts)):
        indices = sorted(experts[s])
        if indices != list(range(len(indices))):
            raise TsmFormatError(
                f"{name}: 스케일 {s} 의 전문가 번호가 0 부터 이어지지 않습니다({indices})",
                f"{name}: expert indices of scale {s} are not contiguous from 0 ({indices})",
            )
        scale_downs, scale_ups = [], []
        for i in indices:
            pair = experts[s][i]
            if "down" not in pair or "up" not in pair:
                raise TsmFormatError(
                    f"{name}: 전문가 {s}.{i} 에 up/down 중 하나가 없습니다", f"{name}: expert {s}.{i} lacks up or down"
                )
            down, up = pair["down"], pair["up"]
            dshape, ushape = _shape(down), _shape(up)
            if len(dshape) != 2 or len(ushape) != 2:
                raise TsmFormatError(
                    f"{name}: Linear 전문가만 지원합니다(down {dshape}, up {ushape} — Conv TSM 은 아님)",
                    f"{name}: only Linear experts are supported (down {dshape}, up {ushape}; no conv TSM)",
                )
            r, fan_in = dshape
            fan_out, r_up = ushape
            if r != r_up:
                raise TsmFormatError(
                    f"{name}: 전문가 {s}.{i} 의 rank 가 맞지 않습니다(down {r}, up {r_up})",
                    f"{name}: expert {s}.{i} rank mismatch (down {r}, up {r_up})",
                )
            if rank is None:
                rank, in_features, out_features = r, fan_in, fan_out
            elif (r, fan_in, fan_out) != (rank, in_features, out_features):
                raise TsmFormatError(
                    f"{name}: 전문가 {s}.{i} 의 모양이 다른 전문가와 다릅니다({dshape}, {ushape})",
                    f"{name}: expert {s}.{i} shape differs from the others ({dshape}, {ushape})",
                )
            scale_downs.append(down)
            scale_ups.append(up)
        n_scales.append(len(indices))
        downs.append(tuple(scale_downs))
        ups.append(tuple(scale_ups))
    assert rank is not None and in_features is not None and out_features is not None

    alpha_value = items.get(prefix + "alpha")
    alpha = float(alpha_value.item() if hasattr(alpha_value, "item") else alpha_value) if alpha_value is not None else float(rank)
    if alpha == 0:
        alpha = float(rank)

    router_w = items.get(prefix + "router_fc.weight")
    router_b = items.get(prefix + "router_fc.bias")
    embed = items.get(prefix + "timestep_embed.weight")
    n_ctx = len(n_scales) - 1
    router_input = ROUTER_INPUT
    if router_w is not None or router_b is not None or embed is not None:
        if router_w is None or router_b is None or embed is None:
            raise TsmFormatError(
                f"{name}: router_fc.weight·router_fc.bias·timestep_embed.weight 가 다 있지 않습니다",
                f"{name}: router_fc.weight, router_fc.bias and timestep_embed.weight must all be present",
            )
        wshape, bshape, eshape = _shape(router_w), _shape(router_b), _shape(embed)
        if len(wshape) != 2 or wshape[0] != n_ctx or bshape != (n_ctx,) or len(eshape) != 2 or eshape[1] != n_ctx or eshape[0] < 1:
            raise TsmFormatError(
                f"{name}: 라우터 모양이 전문가 스케일 수({len(n_scales)})와 맞지 않습니다"
                f"(router_fc {wshape}/{bshape}, timestep_embed {eshape})",
                f"{name}: router shapes do not match {len(n_scales)} scales "
                f"(router_fc {wshape}/{bshape}, timestep_embed {eshape})",
            )
        width = wshape[1]
        if width == in_features and width == rank:
            router_input = router_hint if router_hint in (ROUTER_INPUT, ROUTER_BOTTLENECK) else ROUTER_INPUT
        elif width == in_features:
            router_input = ROUTER_INPUT
        elif width == rank:
            router_input = ROUTER_BOTTLENECK
        else:
            raise TsmFormatError(
                f"{name}: router_fc 너비 {width} 가 입력({in_features})도 rank({rank})도 아닙니다",
                f"{name}: router_fc width {width} is neither the input ({in_features}) nor the rank ({rank})",
            )
        num_timesteps = eshape[0]
    else:
        num_timesteps = DEFAULT_NUM_TIMESTEPS
    return TsmModule(
        name=name,
        n_scales=tuple(n_scales),
        rank=int(rank),
        in_features=int(in_features),
        out_features=int(out_features),
        scale=alpha / float(rank),
        num_timesteps=int(num_timesteps),
        downs=tuple(downs),
        ups=tuple(ups),
        router_w=router_w,
        router_b=router_b,
        embed=embed,
        router_input=router_input,
    )


def build_tsm_file(
    sd: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    path: str = "",
    name: str = "",
) -> TsmFile:
    """state dict(+메타데이터) → 검증한 :class:`TsmFile`. TSM 이 아닌 키는 무시한다(호출자가 따로 다룬다)."""
    meta = {str(k): str(v) for k, v in (metadata or {}).items()}
    args = network_args(meta)
    algo = metadata_algo(meta)
    if algo is not None and algo != TSM_ALGO:
        raise TsmFormatError(
            f"LyCORIS algo 가 '{algo}' 입니다(TSM 아님)", f"LyCORIS algo is '{algo}', not tsm"
        )
    modules_names = sorted(tsm_module_names(sd.keys()))
    if not modules_names:
        raise TsmFormatError(
            "TSM 키(experts.S.I.up/down.weight + router_fc/timestep_embed)가 없습니다",
            "no TSM keys (experts.S.I.up/down.weight with router_fc/timestep_embed)",
        )
    grouped: dict[str, dict[str, Any]] = {module: {} for module in modules_names}
    module_set = set(modules_names)
    for key, value in sd.items():
        owner = _module_of(key, module_set)
        if owner is not None:
            grouped[owner][key] = value
    hint = str(args.get("tsm_router_input", "") or "").strip().lower() or None
    modules = {module: _build_module(module, grouped[module], hint) for module in modules_names}

    # 스케일 구성은 모듈마다 달라도 계산에는 문제가 없다(모듈별로 따로 고른다). 학습기는 한 값을 쓰므로 메타데이터에
    # tsm_n_scales 가 있으면 모든 모듈이 그와 같아야 하고, 없으면 가장 많은 구성을 파일 요약에 쓴다.
    counts: dict[tuple[int, ...], int] = {}
    for module in modules.values():
        counts[module.n_scales] = counts.get(module.n_scales, 0) + 1
    n_scales = max(counts, key=lambda key: (counts[key], key))
    for module in modules.values():
        if any(n < 1 for n in module.n_scales):
            raise TsmFormatError(f"{module.name}: 빈 스케일이 있습니다({module.n_scales})", f"{module.name}: empty expert scale ({module.n_scales})")
    declared = _parse_n_scales(args.get("tsm_n_scales"))
    if declared is not None:
        odd = [m.name for m in modules.values() if m.n_scales != declared]
        if odd:
            raise TsmFormatError(
                f"메타데이터 tsm_n_scales {list(declared)} 와 파일의 전문가 {list(modules[odd[0]].n_scales)} 가 다릅니다({odd[0]})",
                f"metadata tsm_n_scales {list(declared)} differs from the experts in the file "
                f"{list(modules[odd[0]].n_scales)} ({odd[0]})",
            )
    mixed_scales = len(counts) > 1
    timesteps = {m.num_timesteps for m in modules.values()}
    if len(timesteps) != 1:
        raise TsmFormatError(
            f"모듈마다 timestep_embed 행 수(T)가 다릅니다({sorted(timesteps)})",
            f"timestep_embed rows (T) differ between modules ({sorted(timesteps)})",
        )
    num_timesteps = timesteps.pop()
    has_router = any(m.has_router for m in modules.values())
    declared_text = str(args.get("tsm_num_timesteps", "") or "").strip()
    if declared_text.isdigit() and int(declared_text) > 0:
        declared_t = int(declared_text)
        if not has_router:
            # 라우터가 없으면 T 를 적은 곳은 메타데이터뿐이다.
            for module in modules.values():
                module.num_timesteps = declared_t
            num_timesteps = declared_t
        elif declared_t != num_timesteps:
            raise TsmFormatError(
                f"메타데이터 tsm_num_timesteps {declared_t} 와 timestep_embed 행 수 {num_timesteps} 가 다릅니다",
                f"metadata tsm_num_timesteps {declared_t} differs from timestep_embed rows {num_timesteps}",
            )

    warnings: list[str] = []
    if mixed_scales:
        warnings.append("모듈마다 전문가 스케일 구성이 다릅니다: " + ", ".join(
            f"{'+'.join(str(n) for n in key)} x{count}" for key, count in sorted(counts.items(), key=lambda kv: -kv[1])
        ))
    dim = str(meta.get("ss_network_dim", "") or "").strip()
    ranks = sorted({m.rank for m in modules.values()})
    if dim.isdigit() and ranks != [int(dim)]:
        # 프리셋이 모듈별 dim 을 바꿀 수 있어 오류로 보지 않는다.
        warnings.append(f"ss_network_dim={dim} 인데 rank 는 {ranks} 입니다")

    stage = _stage_from_metadata(args)
    nonzero_router = any(not m.router_is_zero() for m in modules.values())
    if stage is not None:
        source = STAGE_FROM_METADATA
    elif nonzero_router:
        stage, source = 2, STAGE_FROM_ROUTER
    else:
        stage, source = 1, STAGE_FROM_ZERO_ROUTER
    if stage == 2 and len(n_scales) > 1:
        missing = [m.name for m in modules.values() if not m.has_router]
        if missing:
            raise TsmFormatError(
                f"2단계 파일인데 라우터가 없는 모듈이 있습니다({missing[0]} 등 {len(missing)}개)",
                f"stage-2 file but {len(missing)} modules have no router (e.g. {missing[0]})",
            )
        if not nonzero_router:
            warnings.append("2단계 파일인데 라우터가 모두 0 입니다 — 문맥 전문가가 빠지고 코어 전문가만 쓰입니다(2단계 학습 0 스텝?)")
    if stage == 1 and source == STAGE_FROM_METADATA and nonzero_router:
        warnings.append("1단계 파일인데 라우터가 0 이 아닙니다 — 1단계 식(전문가 합)으로 돌립니다")

    timestep_dtype = timestep_dtype_from_metadata(meta)
    for module in modules.values():
        module.timestep_dtype = timestep_dtype

    routers = {m.router_input for m in modules.values() if m.has_router}
    router_input = routers.pop() if len(routers) == 1 else (ROUTER_INPUT if not routers else "mixed")
    ranks_all = {m.rank for m in modules.values()}
    return TsmFile(
        path=str(path),
        name=name or os.path.splitext(os.path.basename(str(path)))[0] or "tsm",
        modules=modules,
        stage=int(stage),
        stage_source=source,
        n_scales=n_scales,
        rank=max(ranks_all),
        num_timesteps=int(num_timesteps),
        router_input=router_input,
        metadata=meta,
        warnings=warnings,
        mixed_scales=mixed_scales,
        timestep_dtype=timestep_dtype,
    )


# ── 파일 캐시 ──

_CACHE_LOCK = threading.Lock()
_CACHE: "OrderedDict[tuple[str, int, int], TsmFile]" = OrderedDict()
CACHE_SIZE = 2


def _stat_key(path: str) -> tuple[str, int, int]:
    full = os.path.abspath(path)
    stat = os.stat(full)
    return os.path.normcase(full), int(stat.st_mtime_ns), int(stat.st_size)


def cached_tsm_file(path: str) -> TsmFile | None:
    try:
        key = _stat_key(path)
    except OSError:
        return None
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
        return hit


def _remember(path: str, tsm: TsmFile) -> None:
    try:
        key = _stat_key(path)
    except OSError:
        return
    with _CACHE_LOCK:
        _CACHE[key] = tsm
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)


def build_any_tsm_file(
    sd: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    path: str = "",
    name: str = "",
) -> TsmFile:
    """TSM-LoKr 키가 있으면 :func:`.lokr_format.build_tsm_lokr_file`, 아니면 :func:`build_tsm_file` (LoRA 전문가).

    두 종류의 키가 한 파일에 섞여 있으면 거부한다(학습기는 algo 하나로 저장한다)."""
    from . import lokr_format as lf

    keys = list(sd.keys())
    if not lf.tsm_lokr_module_names(keys):
        return build_tsm_file(sd, metadata, path=path, name=name)
    if tsm_module_names(keys):
        raise TsmFormatError(
            "TSM(LoRA 전문가)과 TSM-LoKr 키가 한 파일에 섞여 있습니다",
            "the file mixes TSM (LoRA expert) and TSM-LoKr keys",
        )
    return lf.build_tsm_lokr_file(sd, metadata, path=path, name=name)


def tsm_file_from_state_dict(path: str, sd: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> TsmFile:
    """이미 메모리에 읽힌 state dict(Forge 로더가 읽은 것)로 만든다. 같은 파일(경로·mtime·크기)은 캐시를 쓴다."""
    hit = cached_tsm_file(path) if path else None
    if hit is not None:
        return hit
    if metadata is None and path:
        try:
            metadata, _ = read_header(path)
        except TsmFormatError:
            metadata = {}
    tsm = build_any_tsm_file(sd, metadata, path=path)
    if path:
        _remember(path, tsm)
    return tsm


def load_tsm_file(path: str) -> TsmFile:
    """파일을 CPU 로 읽어 검증한다(캐시). safetensors 만 받는다."""
    hit = cached_tsm_file(path)
    if hit is not None:
        return hit
    if not str(path).lower().endswith(".safetensors"):
        raise TsmFormatError("TSM 은 .safetensors 파일만 읽습니다", "TSM files must be .safetensors")
    metadata, header = read_header(path)
    if not is_tsm_state_dict(header.keys()):
        raise TsmFormatError(
            "TSM 키(experts.S.I.up/down.weight + router_fc/timestep_embed)가 없습니다",
            "no TSM keys (experts.S.I.up/down.weight with router_fc/timestep_embed)",
        )
    from safetensors.torch import load_file

    sd = load_file(str(path), device="cpu")
    tsm, _rest = split_state_dict(sd)
    result = build_any_tsm_file(tsm, metadata, path=str(path))
    _remember(path, result)
    return result


def release_cache() -> bool:
    """읽어 둔 TSM 파일(CPU 텐서)을 버린다. 놓은 것이 있었으면 True."""
    with _CACHE_LOCK:
        had = bool(_CACHE)
        _CACHE.clear()
    return had
