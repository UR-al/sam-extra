"""TSM-LoKr 파일 형식 — sd-scripts 포크 ``algo=tsm-lokr`` (``--lycoris_algo_fixes``) 가 저장한 파일.

TSM 과 같은 시간 구간 전문가인데, 전문가 하나가 LoRA(up·down)가 아니라 LoKr(크로네커 곱)이다. 모듈 하나
(``lora_unet_blocks_3_mlp_layer1``)의 키::

    <모듈>.experts.<s>.<i>.lokr_w1      [a, c]       (또는 .lokr_w1_a [a, r] + .lokr_w1_b [r, c] — decompose_both)
    <모듈>.experts.<s>.<i>.lokr_w2      [b, d]       (또는 .lokr_w2_a [b, r] + .lokr_w2_b [r, d] — full_matrix 아님)
    <모듈>.alpha                        0차원        (w1·w2 가 둘 다 꽉 찬 행렬이면 스케일 1, 아니면 alpha / r)
    <모듈>.dora_scale                   [out, 1]     (DoRA 출력 축 크기 m — 모든 전문가가 같이 쓴다; DoRA 없으면 없음)
    <모듈>.router_fc.weight             [n_ctx, in]  (라우터 입력은 input 만 — full-matrix LoKr 에는 bottleneck 이 없다)
    <모듈>.router_fc.bias               [n_ctx]
    <모듈>.timestep_embed.weight        [T, n_ctx]

``out = a·b``, ``in = c·d``, ``kron(w1, w2)[i·b+k, j·d+l] = w1[i, j]·w2[k, l]`` (LyCORIS ``functional/lokr.py``).
``(a, b)`` 는 LyCORIS ``factorization(out, factor)``, ``(c, d)`` 는 ``factorization(in, factor)`` 이다.

메타데이터(설계 §3.7)는 꼭 있어야 한다 — 단계·시간 dtype·DoRA 축을 키에서 추측하지 않는다::

    ss_lycoris_algo = tsm-lokr
    ss_saved_format = tsm-lokr-native
    ss_tsm_expert   = lokr
    ss_tsm_lokr     = {"version": 1, "n_scales": [8, 1], "num_timesteps": 1000, "stage": 1, "router_input": "input",
                       "factors": [4, 4], "full_matrix": true, "decompose_both": false, "dora": "out",
                       "timestep_dtype": "bf16", "timestep_scale": 1000, "pooling": "sample",
                       "multiplier": "forge-blend", "gate_init": 0.0}

``factors`` 는 스케일마다 하나(``len == len(n_scales)``) 또는 [코어, 문맥] 두 개(문맥 스케일이 모두 같은 값)다.
v1 은 ``dora`` 가 ``out`` · ``none`` 만, ``router_input`` 은 ``input`` 만 받는다(학습기 v1 과 같다).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

import torch

from .fileformat import (
    ROUTER_INPUT,
    TIMESTEP_DTYPES,
    TsmFile,
    TsmFormatError,
    expert_index,
    metadata_algo,
    network_args,
)

TSM_LOKR_ALGO = "tsm-lokr"
TSM_LOKR_SAVED_FORMAT = "tsm-lokr-native"
TSM_LOKR_META_KEY = "ss_tsm_lokr"
TSM_LOKR_VERSION = 1
DORA_OUT = "out"
DORA_NONE = "none"
MULTIPLIER_FORGE_BLEND = "forge-blend"
POOLING_SAMPLE = "sample"

_FACTOR_KEYS = ("lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b")
_EXPERT_RE = re.compile(
    r"^(?P<mod>.+?)\.experts\.(?P<s>\d+)\.(?P<i>\d+)\.(?P<w>lokr_w1|lokr_w1_a|lokr_w1_b|lokr_w2|lokr_w2_a|lokr_w2_b)$"
)
AUX_SUFFIXES = ("alpha", "dora_scale", "router_fc.weight", "router_fc.bias", "timestep_embed.weight")
_REQUIRED_CONFIG = {
    "version": int, "n_scales": list, "num_timesteps": int, "stage": int, "router_input": str, "factors": list,
    "full_matrix": bool, "decompose_both": bool, "dora": str, "timestep_dtype": str, "timestep_scale": (int, float),
    "pooling": str, "multiplier": str, "gate_init": (int, float),
}


# ── 키 판별 ──

def tsm_lokr_module_names(keys: Iterable[str]) -> set[str]:
    """LoKr 전문가 키(``experts.S.I.lokr_*``)와 라우터 또는 시간 임베딩 키를 함께 가진 모듈 이름."""
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


def is_tsm_lokr_state_dict(sd_or_keys: Mapping[str, Any] | Iterable[str]) -> bool:
    keys = sd_or_keys.keys() if isinstance(sd_or_keys, Mapping) else sd_or_keys
    return bool(tsm_lokr_module_names(keys))


def module_of(key: str, modules: set[str]) -> str | None:
    """키가 ``modules`` 중 하나의 TSM-LoKr 키(전문가·alpha·dora_scale·라우터)면 그 모듈 이름."""
    match = _EXPERT_RE.match(key)
    if match is not None:
        return match.group("mod") if match.group("mod") in modules else None
    for suffix in AUX_SUFFIXES:
        if key.endswith("." + suffix):
            name = key[: -len(suffix) - 1]
            return name if name in modules else None
    return None


def saved_formats(metadata: Mapping[str, Any] | None) -> list[str]:
    """``ss_saved_format`` 의 형식 목록 — 포크는 모듈 종류별 형식을 ``+`` 로 잇는다(``native+tsm-lokr-native``)."""
    text = str((metadata or {}).get("ss_saved_format", "") or "").strip().lower()
    return [part for part in text.split("+") if part]


def metadata_says_tsm_lokr(metadata: Mapping[str, Any] | None) -> bool:
    meta = metadata or {}
    if TSM_LOKR_SAVED_FORMAT in saved_formats(meta):
        return True
    return metadata_algo(meta) == TSM_LOKR_ALGO


# ── LyCORIS 규칙 ──

def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """LyCORIS 3.2 ``functional/general.py`` ``factorization`` 그대로 — (작은 값, 큰 값)."""
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


# ── 설정(메타데이터 JSON) ──

def parse_config(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """``ss_tsm_lokr`` v1 을 엄격하게 읽는다. 모르는 키는 ``config["unknown_keys"]`` 로 돌려준다(경고용)."""
    raw = (metadata or {}).get(TSM_LOKR_META_KEY)
    if raw is None or str(raw).strip() == "":
        raise TsmFormatError(
            f"TSM-LoKr 파일에 메타데이터 {TSM_LOKR_META_KEY} 가 없습니다(단계·시간 dtype·DoRA 축을 알 수 없음)",
            f"TSM-LoKr file lacks the {TSM_LOKR_META_KEY} metadata (stage, timestep dtype and DoRA axis are unknown)",
        )
    try:
        value = raw if isinstance(raw, Mapping) else json.loads(str(raw))
    except ValueError as exc:
        raise TsmFormatError(
            f"{TSM_LOKR_META_KEY} 가 JSON 이 아닙니다: {exc}", f"{TSM_LOKR_META_KEY} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise TsmFormatError(f"{TSM_LOKR_META_KEY} 가 JSON 객체가 아닙니다", f"{TSM_LOKR_META_KEY} is not a JSON object")
    config = dict(value)
    version = config.get("version")
    if isinstance(version, bool) or version != TSM_LOKR_VERSION:
        raise TsmFormatError(
            f"{TSM_LOKR_META_KEY} 버전 {version!r} 을 모릅니다(이 확장은 {TSM_LOKR_VERSION} 만 읽습니다)",
            f"unknown {TSM_LOKR_META_KEY} version {version!r} (this extension reads version {TSM_LOKR_VERSION})",
        )
    for key, kind in _REQUIRED_CONFIG.items():
        if key not in config:
            raise TsmFormatError(f"{TSM_LOKR_META_KEY} 에 '{key}' 가 없습니다", f"{TSM_LOKR_META_KEY} lacks '{key}'")
        item = config[key]
        if (isinstance(item, bool) and kind is not bool) or not isinstance(item, kind):
            raise TsmFormatError(
                f"{TSM_LOKR_META_KEY} 의 '{key}' 값 {item!r} 의 형식이 틀립니다",
                f"{TSM_LOKR_META_KEY} '{key}' has the wrong type ({item!r})",
            )

    def bad(key: str, why_ko: str, why_en: str) -> TsmFormatError:
        return TsmFormatError(
            f"{TSM_LOKR_META_KEY} 의 '{key}' = {config[key]!r}: {why_ko}",
            f"{TSM_LOKR_META_KEY} '{key}' = {config[key]!r}: {why_en}",
        )

    n_scales = config["n_scales"]
    if not n_scales or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in n_scales):
        raise bad("n_scales", "1 이상의 정수 목록이어야 합니다", "must be a list of integers >= 1")
    if config["num_timesteps"] < 1:
        raise bad("num_timesteps", "1 이상이어야 합니다", "must be >= 1")
    if config["stage"] not in (1, 2):
        raise bad("stage", "1 또는 2 여야 합니다", "must be 1 or 2")
    if config["router_input"] != ROUTER_INPUT:
        raise bad("router_input", "TSM-LoKr 는 input 만 씁니다", "TSM-LoKr only uses input")
    factors = config["factors"]
    if any(isinstance(f, bool) or not isinstance(f, int) or f == 0 for f in factors) or len(factors) not in (
        len(n_scales), 2
    ) or (len(factors) == 2 and len(n_scales) == 1):
        raise bad("factors", "스케일마다 하나 또는 [코어, 문맥] 의 0 이 아닌 정수여야 합니다",
                  "must be one non-zero integer per scale or [core, context]")
    if config["dora"] not in (DORA_OUT, DORA_NONE):
        raise bad("dora", "out 또는 none 만 읽습니다(입력 축 DoRA 는 v1 에 없음)", "only out or none (no input-axis DoRA in v1)")
    if config["timestep_dtype"] not in TIMESTEP_DTYPES:
        raise bad("timestep_dtype", f"{'/'.join(TIMESTEP_DTYPES)} 중 하나여야 합니다", f"must be one of {TIMESTEP_DTYPES}")
    if float(config["timestep_scale"]) != 1000.0:
        raise bad("timestep_scale", "1000 만 지원합니다(t = σ·1000)", "only 1000 is supported (t = sigma*1000)")
    if config["pooling"] != POOLING_SAMPLE:
        raise bad("pooling", "sample 만 지원합니다", "only sample is supported")
    if config["multiplier"] != MULTIPLIER_FORGE_BLEND:
        raise bad("multiplier", "forge-blend 만 지원합니다", "only forge-blend is supported")
    per_scale = list(factors) if len(factors) == len(n_scales) else [factors[0]] + [factors[1]] * (len(n_scales) - 1)
    config["scale_factors"] = tuple(int(f) for f in per_scale)
    config["n_scales"] = tuple(int(n) for n in n_scales)
    config["unknown_keys"] = sorted(set(config) - set(_REQUIRED_CONFIG) - {"scale_factors", "unknown_keys"})
    return config


# ── 파일 표현 ──

def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in getattr(tensor, "shape", ()))


@dataclass(eq=False)
class LokrExpert:
    """전문가 하나(CPU 텐서, 파일 dtype). 꽉 찬 인자는 ``w1``/``w2``, 저랭크면 ``*_a``·``*_b``."""

    shape: tuple[int, int, int, int]          # (a, c, b, d): w1 [a, c], w2 [b, d]
    scale: float
    w1: torch.Tensor | None = None
    w1_a: torch.Tensor | None = None
    w1_b: torch.Tensor | None = None
    w2: torch.Tensor | None = None
    w2_a: torch.Tensor | None = None
    w2_b: torch.Tensor | None = None

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(t for t in (self.w1, self.w1_a, self.w1_b, self.w2, self.w2_a, self.w2_b) if t is not None)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())

    def full_factors(self, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu"):
        """(w1 [a, c], w2 [b, d]) 를 ``dtype``·``device`` 로. 저랭크 인자는 그 dtype 으로 곱한다."""
        if self.w1 is not None:
            w1 = self.w1.to(device=device, dtype=dtype)
        else:
            w1 = self.w1_a.to(device=device, dtype=dtype) @ self.w1_b.to(device=device, dtype=dtype)
        if self.w2 is not None:
            w2 = self.w2.to(device=device, dtype=dtype)
        else:
            w2 = self.w2_a.to(device=device, dtype=dtype) @ self.w2_b.to(device=device, dtype=dtype)
        return w1, w2


@dataclass(eq=False)
class TsmLokrModule:
    """대상 Linear 하나의 TSM-LoKr(CPU 텐서, 파일 dtype)."""

    name: str
    n_scales: tuple[int, ...]
    in_features: int
    out_features: int
    num_timesteps: int
    factors: tuple[int, ...]                              # 스케일마다 LoKr factor
    experts: tuple[tuple[LokrExpert, ...], ...]           # [스케일][전문가]
    dora_scale: torch.Tensor | None = None                # [out, 1]
    router_w: torch.Tensor | None = None                  # [n_ctx, in]
    router_b: torch.Tensor | None = None
    embed: torch.Tensor | None = None                     # [T, n_ctx]
    alpha: float = 1.0
    timestep_dtype: str = "fp32"
    router_input: str = ROUTER_INPUT
    base: "TsmLokrModule | None" = None                   # without_dora() 보기면 원래 모듈(장치 사본을 같이 쓴다)

    kind = "lokr"
    rank = 0          # LoRA TsmModule 과 같은 이름(LoKr 에는 공용 rank 가 없다)

    @property
    def n_ctx(self) -> int:
        return len(self.n_scales) - 1

    @property
    def has_router(self) -> bool:
        return self.router_w is not None and self.router_b is not None and self.embed is not None

    @property
    def has_dora(self) -> bool:
        return self.dora_scale is not None

    @property
    def root(self) -> "TsmLokrModule":
        return self.base if self.base is not None else self

    def router_is_zero(self) -> bool:
        if not self.has_router:
            return True
        return all(bool((tensor == 0).all()) for tensor in (self.router_w, self.router_b, self.embed))

    def expert_indices(self, t: int) -> tuple[int, ...]:
        return tuple(expert_index(t, n, self.num_timesteps) for n in self.n_scales)

    def without_dora(self) -> "TsmLokrModule":
        """같은 텐서를 쓰고 DoRA 크기만 뺀 보기(끼워 넣은 블록의 additive · weak 정책)."""
        if self.dora_scale is None:
            return self
        return replace(self, dora_scale=None, base=self.root)

    def interval_bytes(self) -> int:
        """한 시간 구간의 활성 전문가(스케일마다 가장 큰 것 하나) — 장치 사본의 크기(파일 dtype)."""
        return sum(max(e.nbytes() for e in scale) for scale in self.experts)

    def fused_bytes(self, element_size: int = 2) -> int:
        """장치에 두는 것: 활성 전문가 + DoRA 크기(fp32) + 라우터(fp32) + 2단계 노름 항(fp32, 코어 구간마다)."""
        extra = self.out_features * 4 if self.dora_scale is not None else 0
        if self.has_router:
            extra += (self.router_w.numel() + self.router_b.numel() + self.embed.numel()) * 4
            n_ctx = self.n_ctx
            extra += self.n_scales[0] * self.out_features * 4 * (1 + n_ctx + n_ctx * n_ctx)
        return self.interval_bytes() + extra

    def merged_bytes(self, element_size: int = 2) -> int:
        return self.out_features * self.in_features * int(element_size)


@dataclass(eq=False)
class TsmLokrFile(TsmFile):
    """TSM-LoKr 파일. :class:`TsmFile` 과 같은 이름으로 읽힌다(``rank`` 는 0)."""

    factors: tuple[int, ...] = ()
    full_matrix: bool = True
    dora: str = DORA_OUT
    config: dict[str, Any] = field(default_factory=dict)

    expert_kind = "lokr"

    def kind_label(self) -> str:
        """infotext·로그용: ``lokr f4 dora`` (코어·문맥 factor 가 다르면 ``f4/8``)."""
        distinct = list(dict.fromkeys(self.factors))
        factor = "/".join(str(f) for f in distinct) if distinct else "?"
        return f"lokr f{factor} {'dora' if self.dora == DORA_OUT else 'nodora'}"

    def fused_bytes(self, element_size: int = 2) -> int:
        return sum(module.fused_bytes(element_size) for module in self.modules.values())


def _build_expert(name: str, s: int, i: int, parts: dict[str, Any], alpha: float | None) -> LokrExpert:
    where = f"{name}: 전문가 {s}.{i}"
    where_en = f"{name}: expert {s}.{i}"
    has_w1 = "lokr_w1" in parts
    has_w1_ab = "lokr_w1_a" in parts or "lokr_w1_b" in parts
    has_w2 = "lokr_w2" in parts
    has_w2_ab = "lokr_w2_a" in parts or "lokr_w2_b" in parts
    if has_w1 == has_w1_ab or has_w2 == has_w2_ab or (has_w1_ab and not ("lokr_w1_a" in parts and "lokr_w1_b" in parts)) or (
        has_w2_ab and not ("lokr_w2_a" in parts and "lokr_w2_b" in parts)
    ):
        raise TsmFormatError(
            f"{where} 의 인자가 맞지 않습니다({sorted(parts)} — lokr_w1 또는 lokr_w1_a+b, lokr_w2 또는 lokr_w2_a+b 하나씩)",
            f"{where_en} has an invalid factor set ({sorted(parts)}; expected lokr_w1 or lokr_w1_a+b and lokr_w2 or lokr_w2_a+b)",
        )
    for key, value in parts.items():
        if len(_shape(value)) != 2:
            raise TsmFormatError(
                f"{where}.{key} 가 2차원이 아닙니다({_shape(value)} — Linear 만 지원, Conv/Tucker 아님)",
                f"{where_en}.{key} is not 2-D ({_shape(value)}; Linear only, no conv/Tucker)",
            )
    rank = None
    if has_w1:
        a, c = _shape(parts["lokr_w1"])
    else:
        (a, r1), (r1b, c) = _shape(parts["lokr_w1_a"]), _shape(parts["lokr_w1_b"])
        if r1 != r1b:
            raise TsmFormatError(f"{where}: lokr_w1_a·b 의 rank 가 다릅니다({r1}, {r1b})", f"{where_en}: lokr_w1_a/b rank mismatch ({r1}, {r1b})")
        rank = r1
    if has_w2:
        b, d = _shape(parts["lokr_w2"])
    else:
        (b, r2), (r2b, d) = _shape(parts["lokr_w2_a"]), _shape(parts["lokr_w2_b"])
        if r2 != r2b:
            raise TsmFormatError(f"{where}: lokr_w2_a·b 의 rank 가 다릅니다({r2}, {r2b})", f"{where_en}: lokr_w2_a/b rank mismatch ({r2}, {r2b})")
        if rank is not None and rank != r2:
            raise TsmFormatError(f"{where}: w1·w2 의 rank 가 다릅니다({rank}, {r2})", f"{where_en}: w1/w2 rank mismatch ({rank}, {r2})")
        rank = r2
    # Forge·LyCORIS LoKr 규칙: 둘 다 꽉 찬 행렬이면 스케일 1, 아니면 alpha / rank (alpha 가 없거나 0 이면 rank).
    if rank is None:
        scale = 1.0
    else:
        value = float(alpha) if alpha else float(rank)
        scale = value / float(rank)
    return LokrExpert(
        shape=(int(a), int(c), int(b), int(d)),
        scale=scale,
        w1=parts.get("lokr_w1"),
        w1_a=parts.get("lokr_w1_a"),
        w1_b=parts.get("lokr_w1_b"),
        w2=parts.get("lokr_w2"),
        w2_a=parts.get("lokr_w2_a"),
        w2_b=parts.get("lokr_w2_b"),
    )


def _build_module(name: str, items: dict[str, Any], config: Mapping[str, Any]) -> TsmLokrModule:
    prefix = name + "."
    found: dict[int, dict[int, dict[str, Any]]] = {}
    for key, value in items.items():
        match = _EXPERT_RE.match(key)
        if match is not None:
            found.setdefault(int(match.group("s")), {}).setdefault(int(match.group("i")), {})[match.group("w")] = value
    scales = sorted(found)
    if scales != list(range(len(scales))):
        raise TsmFormatError(
            f"{name}: 스케일 번호가 0 부터 이어지지 않습니다({scales})", f"{name}: scale indices are not contiguous from 0 ({scales})"
        )
    n_scales = tuple(len(found[s]) for s in scales)
    if n_scales != tuple(config["n_scales"]):
        raise TsmFormatError(
            f"{name}: 전문가 {list(n_scales)} 가 메타데이터 n_scales {list(config['n_scales'])} 와 다릅니다",
            f"{name}: experts {list(n_scales)} differ from the metadata n_scales {list(config['n_scales'])}",
        )
    alpha_value = items.get(prefix + "alpha")
    alpha = float(alpha_value.item() if hasattr(alpha_value, "item") else alpha_value) if alpha_value is not None else None
    factors = tuple(config["scale_factors"])
    experts: list[tuple[LokrExpert, ...]] = []
    out_features = in_features = None
    for s in scales:
        indices = sorted(found[s])
        if indices != list(range(len(indices))):
            raise TsmFormatError(
                f"{name}: 스케일 {s} 의 전문가 번호가 0 부터 이어지지 않습니다({indices})",
                f"{name}: expert indices of scale {s} are not contiguous from 0 ({indices})",
            )
        row = []
        for i in indices:
            expert = _build_expert(name, s, i, found[s][i], alpha)
            a, c, b, d = expert.shape
            if out_features is None:
                out_features, in_features = a * b, c * d
            elif (a * b, c * d) != (out_features, in_features):
                raise TsmFormatError(
                    f"{name}: 전문가 {s}.{i} 의 크기({a * b}x{c * d})가 다른 전문가({out_features}x{in_features})와 다릅니다",
                    f"{name}: expert {s}.{i} size {a * b}x{c * d} differs from the others ({out_features}x{in_features})",
                )
            want = (factorization(a * b, factors[s]), factorization(c * d, factors[s]))
            if ((a, b), (c, d)) != want:
                raise TsmFormatError(
                    f"{name}: 전문가 {s}.{i} 의 인자 모양 w1 {a}x{c} · w2 {b}x{d} 이 factor {factors[s]} 와 맞지 않습니다"
                    f"(기대 w1 {want[0][0]}x{want[1][0]} · w2 {want[0][1]}x{want[1][1]})",
                    f"{name}: expert {s}.{i} factor shapes w1 {a}x{c}, w2 {b}x{d} do not match factor {factors[s]} "
                    f"(expected w1 {want[0][0]}x{want[1][0]}, w2 {want[0][1]}x{want[1][1]})",
                )
            if config["full_matrix"] and (expert.w1 is None or expert.w2 is None):
                raise TsmFormatError(
                    f"{name}: full_matrix 인데 전문가 {s}.{i} 에 저랭크 인자가 있습니다",
                    f"{name}: full_matrix is set but expert {s}.{i} has low-rank factors",
                )
            if not config["decompose_both"] and expert.w1 is None:
                raise TsmFormatError(
                    f"{name}: decompose_both 가 아닌데 전문가 {s}.{i} 의 w1 이 저랭크입니다",
                    f"{name}: decompose_both is off but expert {s}.{i} has a low-rank w1",
                )
            row.append(expert)
        experts.append(tuple(row))
    assert out_features is not None and in_features is not None

    dora = items.get(prefix + "dora_scale")
    if config["dora"] == DORA_OUT:
        if dora is None:
            raise TsmFormatError(f"{name}: DoRA(out) 파일인데 dora_scale 이 없습니다", f"{name}: DoRA (out) file without dora_scale")
        if _shape(dora) != (out_features, 1):
            raise TsmFormatError(
                f"{name}: dora_scale 모양 {_shape(dora)} 이 출력 축 [{out_features}, 1] 이 아닙니다",
                f"{name}: dora_scale shape {_shape(dora)} is not the output axis [{out_features}, 1]",
            )
    elif dora is not None:
        raise TsmFormatError(f"{name}: dora=none 인데 dora_scale 이 있습니다", f"{name}: dora=none but dora_scale is present")

    router_w = items.get(prefix + "router_fc.weight")
    router_b = items.get(prefix + "router_fc.bias")
    embed = items.get(prefix + "timestep_embed.weight")
    n_ctx = len(n_scales) - 1
    if router_w is None or router_b is None or embed is None:
        raise TsmFormatError(
            f"{name}: router_fc.weight·router_fc.bias·timestep_embed.weight 가 다 있지 않습니다",
            f"{name}: router_fc.weight, router_fc.bias and timestep_embed.weight must all be present",
        )
    wshape, bshape, eshape = _shape(router_w), _shape(router_b), _shape(embed)
    T = int(config["num_timesteps"])
    if wshape != (n_ctx, in_features) or bshape != (n_ctx,) or eshape != (T, n_ctx):
        raise TsmFormatError(
            f"{name}: 라우터 모양이 맞지 않습니다(router_fc {wshape}/{bshape}, timestep_embed {eshape} — "
            f"기대 [{n_ctx}, {in_features}]/[{n_ctx}], [{T}, {n_ctx}])",
            f"{name}: router shapes do not match (router_fc {wshape}/{bshape}, timestep_embed {eshape}; "
            f"expected [{n_ctx}, {in_features}]/[{n_ctx}], [{T}, {n_ctx}])",
        )
    return TsmLokrModule(
        name=name,
        n_scales=n_scales,
        in_features=int(in_features),
        out_features=int(out_features),
        num_timesteps=T,
        factors=factors,
        experts=tuple(experts),
        dora_scale=dora,
        router_w=router_w,
        router_b=router_b,
        embed=embed,
        alpha=float(alpha) if alpha is not None else 1.0,
        timestep_dtype=str(config["timestep_dtype"]),
    )


def build_tsm_lokr_file(
    sd: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    path: str = "",
    name: str = "",
) -> TsmLokrFile:
    """state dict(+메타데이터) → 검증한 :class:`TsmLokrFile`. TSM-LoKr 가 아닌 키는 무시한다."""
    meta = {str(k): str(v) for k, v in (metadata or {}).items()}
    algo = metadata_algo(meta)
    if algo is not None and algo != TSM_LOKR_ALGO:
        raise TsmFormatError(f"LyCORIS algo 가 '{algo}' 입니다(tsm-lokr 아님)", f"LyCORIS algo is '{algo}', not tsm-lokr")
    saved = saved_formats(meta)
    if saved and TSM_LOKR_SAVED_FORMAT not in saved:
        raise TsmFormatError(
            f"ss_saved_format 이 '{'+'.join(saved)}' 입니다({TSM_LOKR_SAVED_FORMAT} 없음)",
            f"ss_saved_format is '{'+'.join(saved)}' (no {TSM_LOKR_SAVED_FORMAT})",
        )
    expert_kind = str(meta.get("ss_tsm_expert", "") or "").strip().lower()
    if expert_kind and expert_kind != "lokr":
        raise TsmFormatError(f"ss_tsm_expert 가 '{expert_kind}' 입니다(lokr 아님)", f"ss_tsm_expert is '{expert_kind}', not lokr")
    config = parse_config(meta)
    names = sorted(tsm_lokr_module_names(sd.keys()))
    if not names:
        raise TsmFormatError(
            "TSM-LoKr 키(experts.S.I.lokr_w1/w2 + router_fc/timestep_embed)가 없습니다",
            "no TSM-LoKr keys (experts.S.I.lokr_w1/w2 with router_fc/timestep_embed)",
        )
    module_set = set(names)
    grouped: dict[str, dict[str, Any]] = {n: {} for n in names}
    for key, value in sd.items():
        owner = module_of(key, module_set)
        if owner is not None:
            grouped[owner][key] = value
    modules = {n: _build_module(n, grouped[n], config) for n in names}

    args = network_args(meta)
    declared = str(args.get("tsm_stage", "") or "").strip()
    stage = int(config["stage"])
    if declared and declared != str(stage):
        raise TsmFormatError(
            f"ss_network_args 의 tsm_stage={declared} 가 {TSM_LOKR_META_KEY} 의 stage={stage} 와 다릅니다",
            f"ss_network_args tsm_stage={declared} differs from {TSM_LOKR_META_KEY} stage={stage}",
        )
    warnings: list[str] = []
    if config["unknown_keys"]:
        warnings.append(f"{TSM_LOKR_META_KEY} 의 모르는 키를 무시합니다: {', '.join(config['unknown_keys'])}")
    n_scales = tuple(config["n_scales"])
    nonzero_router = any(not m.router_is_zero() for m in modules.values())
    if stage == 2 and len(n_scales) > 1 and not nonzero_router:
        warnings.append("2단계 파일인데 라우터가 모두 0 입니다 — 문맥 전문가가 빠지고 코어 전문가만 쓰입니다(2단계 학습 0 스텝?)")
    if stage == 1 and nonzero_router:
        warnings.append("1단계 파일인데 라우터가 0 이 아닙니다 — 1단계 식(전문가 합)으로 돌립니다")
    return TsmLokrFile(
        path=str(path),
        name=name or os.path.splitext(os.path.basename(str(path)))[0] or "tsm-lokr",
        modules=modules,
        stage=stage,
        stage_source="metadata",
        n_scales=n_scales,
        rank=0,
        num_timesteps=int(config["num_timesteps"]),
        router_input=ROUTER_INPUT,
        metadata=meta,
        warnings=warnings,
        mixed_scales=False,
        timestep_dtype=str(config["timestep_dtype"]),
        factors=tuple(config["scale_factors"]),
        full_matrix=bool(config["full_matrix"]),
        dora=str(config["dora"]),
        config=config,
    )
