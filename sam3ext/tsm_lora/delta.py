"""TSM Δ 계산 — 학습(LyCORIS 3.2.0.dev55 ``tsm.py`` + sd-scripts 포크 ``library/lycoris_algo_fixes.py``)과 같은 식.

대상 Linear 하나, 샘플 하나에 대해::

    t        = round(q(fp32(σ) * 1000) · T / 1000)  float32 곱 → 학습 dtype q(bf16 등, 메타데이터 ss_mixed_precision)
                                                 로 반올림 → 파이썬 round(짝수 쪽), [0, T] 로 자름
    idx_s(t) = clamp(ceil(t / T * n_s), 1, n_s) - 1  파이썬 float 식(정수 ceil 아님)
    E(x)     = up @ (down @ x)                      가중치는 x.dtype 으로
    1단계:  Δ = scale · mult · Σ_s E_{s, idx_s(t)}(x)
    2단계:  Δ = scale · mult · ( E_{0, idx_0(t)}(x) + Σ_{j≥1} g_{j-1} · E_{j, idx_j(t)}(x) )
            z = 그 샘플 입력의 채널 외 모든 축 평균(fp32)   (bottleneck 방식이면 코어 down(x), 너비 r)
            g = router_fc(z) + timestep_embed[clamp(t-1, 0, T-1)]   (시그모이드 없음, mult 를 곱하지 않음)

샘플별 풀링이 기본이다 — 학습은 배치 1 이었고, Forge 가 cond·uncond 를 한 배치로 묶어도(또는 따로 돌려도) 같은 값이
나온다. ``POOL_BATCH`` 는 포크의 LyCORIS forward 가 같은 t 를 가진 샘플을 함께 평균하던 동작(비교용)이다.

계산은 시간 구간의 활성 전문가(스케일마다 하나)를 이어 붙인 down ``[S·r, in]`` · up ``[out, S·r]`` 한 쌍으로 한다.
1단계는 선형이라 합과 같고, 2단계는 중간값의 문맥 조각에 게이트를 곱한다. 이어 붙인 텐서는 장치·dtype 별로
:class:`FusedCache` 가 시간 구간이 바뀔 때만(전체 스케줄에서 8번) 다시 만든다.

Forge 훅은 :class:`Slot` 으로 돈다 — 대상 Linear 에 걸린 TSM 하나의 상수(scale·강도, 단계, 라우터 방식)와 지금 구간의
장치 텐서를 쥐고, Linear 출력에 ``out.addmm_(mid, upᵀ, alpha=scale·강도)`` 로 바로 더한다(Δ·스케일 곱·합 텐서를 따로
만들지 않는다). :func:`module_delta` 는 같은 식의 기준 구현(Δ 를 돌려줌 — 테스트·행마다 t 가 다른 드문 호출용)이다.
2단계 라우터 평균은 ``mean(dtype=float32)`` 로 낸다 — GPU 에서 입력의 fp32 사본을 만들지 않는다(CPU 는 값이 비트 단위로
같다). 같은 입력을 받는 Linear(self_attn q·k·v, cross_attn k·v)는 :class:`PoolCache` 로 평균을 한 번만 낸다.
"""
from __future__ import annotations

import weakref
from collections import OrderedDict
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from .fileformat import (  # noqa: F401  (td.expert_index 로도 쓴다)
    ROUTER_BOTTLENECK,
    TIMESTEP_BF16,
    TIMESTEP_FP16,
    TIMESTEP_FP32,
    TsmModule,
    expert_index,
)

POOL_SAMPLE = "sample"
POOL_BATCH = "batch"
POOLINGS = (POOL_SAMPLE, POOL_BATCH)
TRAIN_MAX_TIMESTEP = 1000.0  # 학습기가 σ 에 곱한 값(anima_train_network: timesteps = sigmas * 1000)


# ── 시간 ──

def timestep_values(timesteps: torch.Tensor | Sequence[float] | float, multiplier: float = 1.0) -> list[float]:
    """DiT 가 받은 timestep 인자 → 학습기가 쓰던 ``σ·1000`` 값(float32 곱)들. 배치 행마다 하나.

    Forge 의 Anima 는 ``PredictionDiscreteFlow.timestep(σ) = σ·multiplier`` (multiplier 1.0) 를 넘기므로 σ 를 되찾을
    때 multiplier 로 나눈다(1.0 이면 나누지 않는다 — 값이 비트 단위로 그대로).
    """
    if isinstance(timesteps, torch.Tensor):
        values = timesteps.detach().to(dtype=torch.float32).reshape(-1)
        if values.device.type != "cpu":
            values = values.cpu()
    else:
        seq = [timesteps] if isinstance(timesteps, (int, float)) else list(timesteps)
        values = torch.tensor([float(v) for v in seq], dtype=torch.float32)
    if float(multiplier) != 1.0:
        values = values / torch.tensor(float(multiplier), dtype=torch.float32)
    values = values * torch.tensor(TRAIN_MAX_TIMESTEP, dtype=torch.float32)
    return [float(v) for v in values.tolist()]


_TORCH_TIMESTEP_DTYPES = {TIMESTEP_BF16: torch.bfloat16, TIMESTEP_FP16: torch.float16}


def quantize_value(value: float, timestep_dtype: str | None = None) -> float:
    """학습기가 ``σ·1000`` (float32) 을 weight_dtype 으로 바꾼 값(round-to-nearest-even). fp32 면 그대로."""
    dtype = _TORCH_TIMESTEP_DTYPES.get(timestep_dtype or TIMESTEP_FP32)
    if dtype is None:
        return float(value)
    return float(torch.tensor(float(value), dtype=torch.float32).to(dtype).item())


def t_from_value(value: float, num_timesteps: int = 1000, timestep_dtype: str | None = None) -> int:
    """lycoris_algo_fixes.set_timestep: ``round(v · T / 1000)`` 을 [0, T] 로 자른다(파이썬 round = 짝수 쪽).

    ``v`` 는 학습 때처럼 ``timestep_dtype`` 으로 먼저 반올림한다(bf16 학습이면 512 이상에서 4 단위)."""
    v = quantize_value(value, timestep_dtype)
    t = int(round(v * num_timesteps / TRAIN_MAX_TIMESTEP))
    return max(0, min(int(num_timesteps), t))


def sigma_to_t(sigma: float, num_timesteps: int = 1000, multiplier: float = 1.0, timestep_dtype: str | None = None) -> int:
    return t_from_value(timestep_values([sigma], multiplier)[0], num_timesteps, timestep_dtype)


def embed_row(t: int, num_timesteps: int = 1000) -> int:
    return min(max(int(t) - 1, 0), int(num_timesteps) - 1)


# ── 이어 붙인 전문가 캐시 ──

class FusedCache:
    """(모듈, dtype, 장치) 마다 지금 시간 구간의 이어 붙인 (down, up). 구간이 바뀌면 그 자리를 바꿔 끼운다."""

    def __init__(self) -> None:
        self._entries: dict[tuple[int, torch.dtype, str], tuple[tuple[int, ...], torch.Tensor, torch.Tensor]] = {}
        self._modules: dict[int, TsmModule] = {}   # id 가 다른 객체에 다시 쓰이지 않게 참조를 쥔다
        # 2단계 라우터(router_fc·timestep_embed 전체)의 장치 사본 — Linear 호출마다 CPU→GPU 복사를 하지 않게.
        self._routers: dict[tuple[int, torch.dtype, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.builds = 0

    def get(self, module: TsmModule, indices: tuple[int, ...], dtype: torch.dtype, device: torch.device):
        key = (id(module), dtype, str(device))
        hit = self._entries.get(key)
        if hit is not None and hit[0] == indices:
            return hit[1], hit[2]
        down, up = fuse_experts(module, indices, dtype, device)
        self._entries[key] = (indices, down, up)
        self._modules[id(module)] = module
        self.builds += 1
        return down, up

    def router(self, module: TsmModule, dtype: torch.dtype, device: torch.device):
        """(router_w, router_b, timestep_embed) 를 ``dtype``·``device`` 로 — 한 번만 옮긴다."""
        key = (id(module), dtype, str(device))
        hit = self._routers.get(key)
        if hit is None:
            assert module.router_w is not None and module.router_b is not None and module.embed is not None
            hit = tuple(upload([t], 0, dtype, device) for t in (module.router_w, module.router_b, module.embed))
            self._routers[key] = hit
            self._modules[id(module)] = module
        return hit

    def nbytes(self) -> int:
        fused = sum(d.numel() * d.element_size() + u.numel() * u.element_size() for _i, d, u in self._entries.values())
        routers = sum(t.numel() * t.element_size() for parts in self._routers.values() for t in parts)
        return fused + routers

    def clear(self) -> bool:
        had = bool(self._entries or self._routers)
        self._entries.clear()
        self._routers.clear()
        self._modules.clear()
        return had


def upload(parts: Sequence[torch.Tensor], dim: int, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """CPU 텐서들을 ``dim`` 으로 이어 붙여 ``dtype`` 으로 바꾼 뒤 ``device`` 로 한 번에 옮긴다(연속).

    GPU 로는 고정(pinned) 메모리에서 ``non_blocking`` 으로 보낸다 — 페이지 가능 메모리에서의 복사는 호스트를 GPU 큐가 빌
    때까지 세워, 시간 구간이 바뀌는 DiT 호출에서 대상 Linear 마다(640번) 동기화가 생긴다. 고정 블록은 torch 의 호스트
    캐시가 복사가 끝날 때까지 다시 쓰지 않는다. 고정할 수 없으면 예전처럼 동기 복사. 값은 전문가마다 옮겨 이어 붙이던
    것과 같다(dtype 바꾸기는 원소별)."""
    cpu = parts[0] if len(parts) == 1 else torch.cat(list(parts), dim)
    device = torch.device(device)
    if cpu.device.type == "cpu":
        cpu = cpu.to(dtype=dtype).contiguous()
        if device.type == "cpu":
            return cpu
        if device.type == "cuda":
            try:
                return cpu.pin_memory().to(device, non_blocking=True)
            except Exception:  # pragma: no cover - 고정 메모리를 못 받는 환경
                pass
    return cpu.to(device=device, dtype=dtype).contiguous()


def fuse_experts(module: TsmModule, indices: Sequence[int], dtype: torch.dtype, device: torch.device | str):
    """스케일마다 고른 전문가의 down 을 행으로, up 을 열로 이어 붙인다 → (``[S·r, in]``, ``[out, S·r]``)."""
    downs = [module.downs[s][i] for s, i in enumerate(indices)]
    ups = [module.ups[s][i] for s, i in enumerate(indices)]
    return upload(downs, 0, dtype, device), upload(ups, 1, dtype, device)


# ── Δ ──

def _router_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def pooled_mean(src: torch.Tensor, width: int, pooling: str, rdtype: torch.dtype) -> torch.Tensor:
    """라우터 입력: 샘플마다(``[B, width]``) 또는 배치 전체(``[1, width]``) 채널 외 모든 축의 평균, ``rdtype`` 으로.

    ``.to(fp32).mean()`` 이 아니라 ``mean(dtype=fp32)`` — CUDA 의 bf16/fp16 → fp32 줄이기는 입력을 읽으며 fp32 로 더해
    전체 fp32 사본(1024² mlp_layer2 입력이면 약 268 MB)을 만들지 않는다. CPU 는 안에서 같은 사본을 만들어 값이 같다."""
    src = src.detach()
    rows = 1 if pooling == POOL_BATCH else src.shape[0]
    return src.reshape(rows, -1, width).mean(1, dtype=rdtype)


class PoolCache:
    """DiT 호출 하나 안에서 같은 입력 텐서의 라우터 평균을 다시 쓴다 — self_attn 의 q·k·v 는 같은 x, cross_attn 의 k·v 는
    같은 context 를 받는다(Forge ``SelfCrossAttention.compute_qkv``).

    마지막 하나만 쥔다(q·k·v / k·v 는 잇달아 불린다). 텐서는 약한 참조로만 쥐어 그 수명을 늘리지 않고, 같은 객체가 살아
    있을 때만 맞춘다 — 풀린 텐서의 id·주소를 새 텐서가 다시 받아도 틀리게 맞지 않는다. 버전 카운터가 있으면(추론 텐서가
    아니면) 제자리 수정도 가린다. DiT 호출의 시작·끝에서 비운다."""

    __slots__ = ("_ref", "_sig", "_z", "hits", "misses")

    def __init__(self) -> None:
        self._ref = None
        self._sig = None
        self._z = None
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        self._ref = self._sig = self._z = None

    def mean(self, src: torch.Tensor, width: int, pooling: str, rdtype: torch.dtype) -> torch.Tensor:
        # 추론 텐서(Forge 는 inference_mode)는 버전을 세지 않는다 — ``_version`` 이 던지는 예외는 비싸므로 먼저 가린다.
        version = None if src.is_inference() else src._version
        sig = (version, src.data_ptr(), src.shape, width, pooling, rdtype)
        ref = self._ref
        if ref is not None and ref() is src and self._sig == sig:
            self.hits += 1
            return self._z
        z = pooled_mean(src, width, pooling, rdtype)
        self.misses += 1
        try:
            self._ref = weakref.ref(src)
        except TypeError:  # pragma: no cover - Tensor 는 weakref 를 받는다
            self.clear()
            return z
        self._sig, self._z = sig, z
        return z


def gates(
    module: TsmModule,
    pool_src: torch.Tensor,
    t: int,
    pooling: str = POOL_SAMPLE,
    cache: "FusedCache | None" = None,
) -> torch.Tensor:
    """라우터 게이트. 샘플별이면 ``[B, n_ctx]``, 배치 전체면 ``[1, n_ctx]`` (fp32, fp64 입력이면 fp64)."""
    assert module.router_w is not None and module.router_b is not None and module.embed is not None
    width = module.router_w.shape[1]
    rdtype = _router_dtype(pool_src.dtype)
    z = pooled_mean(pool_src, width, pooling, rdtype)
    device = z.device
    row = embed_row(t, module.num_timesteps)
    if cache is not None:
        w, b, embed = cache.router(module, rdtype, device)
        e = embed[row]
    else:
        w = module.router_w.to(device=device, dtype=rdtype)
        b = module.router_b.to(device=device, dtype=rdtype)
        e = module.embed[row].to(device=device, dtype=rdtype)
    return F.linear(z, w, b) + e


def _delta_same_t(
    module: TsmModule,
    x: torch.Tensor,
    t: int,
    stage: int,
    multiplier: float,
    pooling: str,
    cache: FusedCache | None,
) -> torch.Tensor:
    indices = module.expert_indices(t)
    if cache is not None:
        down, up = cache.get(module, indices, x.dtype, x.device)
    else:
        down, up = fuse_experts(module, indices, x.dtype, x.device)
    mid = F.linear(x, down)                                 # [..., S·r]
    if stage == 2 and module.n_ctx > 0:
        r = module.rank
        src = mid[..., :r] if module.router_input == ROUTER_BOTTLENECK else x
        g = gates(module, src, t, pooling, cache)           # [G, n_ctx]
        factor = torch.ones((g.shape[0], r * len(module.n_scales)), dtype=g.dtype, device=g.device)
        factor[:, r:] = g.repeat_interleave(r, dim=1)
        shape = (g.shape[0],) + (1,) * (mid.dim() - 2) + (factor.shape[1],)
        mid = mid * factor.to(mid.dtype).reshape(shape)
    out = F.linear(mid, up)
    return out * (module.scale * float(multiplier))


def can_add_in_place(out: torch.Tensor, x: torch.Tensor) -> bool:
    """Linear 출력에 Δ 를 제자리로 더해도 되는지 — 새로 만든 출력(F.linear 결과)이면 된다.

    dtype 이 입력과 같아야 하고(장치 사본·중간값이 입력 dtype), 연속이어야 하며(2-D 로 보기), 기울기를 쓰지 않고, 추론 텐서면
    지금도 inference_mode 여야 한다. 아니면 새 텐서를 만드는 쪽으로 간다."""
    return (
        isinstance(out, torch.Tensor)
        and out.dtype is x.dtype
        and not out.requires_grad
        and out.is_contiguous()
        and (not out.is_inference() or torch.is_inference_mode_enabled())
    )


class Slot:
    """대상 Linear 하나에 걸린 TSM 하나 — 훅이 호출마다 다시 셈하지 않도록 상수와 지금 구간의 장치 텐서를 쥔다.

    ``alpha`` (scale·강도)는 up 에 접지 않고 ``addmm_`` 의 alpha 로 준다 — GEMM 이 fp32 로 모은 값에 곱하고 한 번만
    반올림하므로, 같은 모듈을 다른 강도로 거는 약한 복사 블록도 장치 사본 하나를 같이 쓴다. ``sig`` 는 세션의 (T, 학습
    timestep dtype) 묶음 번호 — DiT 호출마다 그 묶음의 t 를 한 번만 셈한다."""

    __slots__ = (
        "module", "multiplier", "stage", "alpha", "sig", "in_features", "rank", "n_ctx", "n_parts", "gated",
        "bottleneck", "ikey", "_indices", "_dtype", "_device", "_down", "_up", "_router", "_router_key",
    )

    def __init__(self, module: TsmModule, multiplier: float, stage: int, sig: int = 0) -> None:
        self.module = module
        self.multiplier = float(multiplier)
        self.stage = int(stage)
        self.alpha = module.scale * float(multiplier)
        self.sig = sig
        self.ikey = 0   # 세션의 (sig, 스케일 구성) 번호 — DiT 호출마다 전문가 번호를 한 번만 셈한다
        self.in_features = module.in_features
        self.rank = module.rank
        self.n_ctx = module.n_ctx
        self.n_parts = len(module.n_scales)
        self.gated = stage == 2 and module.n_ctx > 0
        self.bottleneck = module.router_input == ROUTER_BOTTLENECK
        self.forget()

    def forget(self) -> None:
        """장치 텐서 참조를 놓는다(세션 release — 이것까지 놓아야 VRAM 이 실제로 풀린다)."""
        self._indices = self._dtype = self._device = None
        self._down = self._up = None
        self._router = self._router_key = None

    def holds_device_tensors(self) -> bool:
        return self._down is not None or self._router is not None

    def weights(self, x: torch.Tensor, t: int, cache: FusedCache | None,
                indices: tuple[int, ...] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """지금 구간의 이어 붙인 (down, up) — 구간·dtype·장치가 그대로면 쥔 것을 그대로(대부분의 호출)."""
        if indices is None:
            indices = self.module.expert_indices(t)
        dtype, device = x.dtype, x.device
        if indices != self._indices or dtype is not self._dtype or device != self._device:
            if cache is not None:
                self._down, self._up = cache.get(self.module, indices, dtype, device)
            else:
                self._down, self._up = fuse_experts(self.module, indices, dtype, device)
            self._indices, self._dtype, self._device = indices, dtype, device
        return self._down, self._up

    def _router_parts(self, rdtype: torch.dtype, device: torch.device, cache: FusedCache | None):
        key = (rdtype, device)
        if self._router is None or self._router_key != key:
            module = self.module
            if cache is not None:
                self._router = cache.router(module, rdtype, device)
            else:
                self._router = tuple(t.to(device=device, dtype=rdtype) for t in (module.router_w, module.router_b, module.embed))
            self._router_key = key
        return self._router

    def gates(self, x: torch.Tensor, mid: torch.Tensor, t: int, pooling: str, cache: FusedCache | None,
              pool: PoolCache | None) -> torch.Tensor:
        """:func:`gates` 와 같은 값 ``[G, n_ctx]``. bottleneck 이면 코어 down(x) = ``mid`` 의 앞 r 열을 평균한다."""
        rdtype = _router_dtype(x.dtype)
        if self.bottleneck:
            z = pooled_mean(mid.view(x.shape[0], -1, mid.shape[-1])[..., : self.rank], self.rank, pooling, rdtype)
        elif pool is not None:
            z = pool.mean(x, self.in_features, pooling, rdtype)
        else:
            z = pooled_mean(x, self.in_features, pooling, rdtype)
        w, b, embed = self._router_parts(rdtype, z.device, cache)
        return F.linear(z, w, b) + embed[embed_row(t, self.module.num_timesteps)]

    def add(self, out: torch.Tensor, x: torch.Tensor, t: int, pooling: str = POOL_SAMPLE,
            cache: FusedCache | None = None, pool: PoolCache | None = None, in_place: bool | None = None,
            indices: tuple[int, ...] | None = None) -> torch.Tensor:
        """``out + Δ`` (모든 행이 같은 t). ``in_place`` 면 ``out`` 을 고쳐 그대로 돌려준다(None 이면 :func:`can_add_in_place`).

        ``mid = x·downᵀ`` (``[M, S·r]``) 하나만 새로 만든다 — 2단계 게이트는 그 문맥 조각에 제자리로 곱한다."""
        down, up = self.weights(x, t, cache, indices)
        mid = F.linear(x.reshape(-1, self.in_features), down)
        if self.gated:
            g = self.gates(x, mid, t, pooling, cache, pool).to(mid.dtype)   # [G, n_ctx] — G 는 배치 또는 1(배치 전체)
            rows = g.shape[0]
            mid.view(rows, -1, self.n_parts, self.rank).narrow(2, 1, self.n_ctx).mul_(g.view(rows, 1, self.n_ctx, 1))
        if in_place is None:
            in_place = can_add_in_place(out, x)
        width = out.shape[-1]
        if in_place:
            out.view(-1, width).addmm_(mid, up.t(), alpha=self.alpha)
            return out
        if out.dtype is mid.dtype:
            return torch.addmm(out.reshape(-1, width), mid, up.t(), alpha=self.alpha).view(out.shape)
        delta = F.linear(mid, up) * self.alpha
        return out + delta.view(out.shape).to(out.dtype)


def module_delta(
    module: TsmModule,
    x: torch.Tensor,
    ts: Sequence[int | None] | int | None,
    stage: int,
    multiplier: float = 1.0,
    pooling: str = POOL_SAMPLE,
    cache: FusedCache | None = None,
) -> torch.Tensor:
    """Linear 입력 ``x`` (배치 축 0) 에 대한 TSM Δ (``x.dtype``). 베이스 Linear 출력에 더한다.

    ``ts`` 는 배치 행마다의 정수 t (하나만 주면 모든 행에 같다). ``None`` 은 LyCORIS 처럼 T(마지막 전문가)로 본다 —
    Forge 런타임은 t 를 모르면 Δ 를 아예 더하지 않으므로 이 값은 학습 쪽 대조용이다.
    """
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage!r}")
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
    T = module.num_timesteps
    batch = x.shape[0] if x.dim() >= 2 else 1
    if ts is None or isinstance(ts, int):
        ts = [ts] * batch
    rows = [T if t is None else max(0, min(T, int(t))) for t in ts]
    if len(rows) == 1 and batch > 1:
        rows = rows * batch
    if len(rows) != batch:
        raise ValueError(f"{len(rows)} timesteps for a batch of {batch}")
    groups: "OrderedDict[int, list[int]]" = OrderedDict()
    for row, t in enumerate(rows):
        groups.setdefault(t, []).append(row)
    if len(groups) == 1:
        return _delta_same_t(module, x, rows[0], stage, multiplier, pooling, cache)
    out = x.new_empty(tuple(x.shape[:-1]) + (module.out_features,))
    for t, members in groups.items():
        index = torch.tensor(members, device=x.device, dtype=torch.long)
        out[index] = _delta_same_t(module, x.index_select(0, index), t, stage, multiplier, pooling, cache)
    return out


def total_delta(
    items: Iterable[tuple[TsmModule, float, int]],
    x: torch.Tensor,
    ts: Sequence[int | None] | int | None,
    pooling: str = POOL_SAMPLE,
    cache: FusedCache | None = None,
) -> torch.Tensor | None:
    """한 Linear 에 걸린 TSM 들(파일이 여럿일 수 있다)의 Δ 합. 걸린 것이 없으면 None."""
    result = None
    for module, multiplier, stage in items:
        if multiplier == 0:
            continue
        delta = module_delta(module, x, ts, stage, multiplier, pooling, cache)
        result = delta if result is None else result + delta
    return result
