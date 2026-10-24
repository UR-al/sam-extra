"""TSM-LoKr 계산 — 학습(sd-scripts 포크 ``library/tsm_lokr.py``, 설계 §3.2)과 같은 식.

대상 Linear 하나(기본 가중치 W ``[out, in]``), 샘플 하나(정수 t)에 대해 — 전문가 e 의 ``K_e = s_e·kron(w1_e, w2_e)``,
코어 ``C = K_{0, idx_0(t)}``, 문맥 ``X_j = K_{j, idx_j(t)}``, DoRA 크기 ``m [out, 1]`` (모든 전문가가 같이 쓴다)::

    D(V)   = V · (m / (‖V‖_행 + eps))          eps = fp32 기계 엡실론, 행 노름(dim=1), 모두 fp32
    1단계:  V = W + (C + Σ_j X_j)               W' = D(V)          (DoRA 없으면 W' = V)
    2단계:  V_b = W + C + Σ_j g_bj · X_j        g_b = router_fc(z_b) + timestep_embed[clamp(t-1)] (시그모이드 없음)
            ‖V_b‖² = a + 2 Σ_j g_bj b_j + Σ_jk g_bj g_bk c_jk   a = ‖W+C‖², b_j = ⟨W+C, X_j⟩, c_jk = ⟨X_j, X_k⟩ (행마다)
            y_b = (m / (√‖V_b‖² + eps)) · (x_b (W+C)ᵀ + Σ_j g_bj · x_b X_jᵀ)
    강도 μ: W'_μ = W + μ·(W' − W)               (Forge 블렌드 — 학습기도 같은 식; μ = 1 이면 W')

계산 길(설계 §6.3):

* **가중치 함수**(Forge ``Linear.weight_function`` — ``backend/operations.py`` ``get_weight_and_bias`` ·
  ``weights_manual_cast``): Forge 가 넘긴 W 로 그 구간의 병합 가중치를 만들어 돌려준다. 1단계는 이것으로 끝(행렬곱은 Forge 가
  한 번). 2단계는 ``W + C`` 를 돌려주고 출력 훅이 문맥 항(크로네커 곱 행렬곱, 기본 GEMM 의 1/factor)과 샘플별 행 스케일을
  더한다. 노름 항 a·b·c 는 그 함수가 받은 W 로 (모듈, 코어 구간) 마다 한 번 셈한다.
* **출력 훅**(``weight_function`` 이 없는 Linear — 순정 ``nn.Linear`` 등): 기본 출력 ``out = x Wᵀ`` 에 크로네커 곱
  행렬곱으로 ``K x`` 를 더하고 노름(모듈·구간마다 한 번, ``module.weight`` 로)으로 행 스케일을 곱한다.
* 한 호출 안에서 행마다 t 가 다르면(드묾) 그 행 묶음마다 병합 가중치로 다시 계산한다(두 길 모두).

크로네커 곱: ``kron(w1 [a,c], w2 [b,d])[i·b+k, j·d+l] = w1[i,j]·w2[k,l]``; 행렬곱 ``kron(w1, w2) x`` 는 x 를 ``[c, d]``
로 보고 ``w1 · (X · w2ᵀ)`` (가중치를 만들지 않는다).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn.functional as F

from .delta import POOL_BATCH, POOL_SAMPLE, POOLINGS, PoolCache, embed_row, pooled_mean
from .lokr_format import TsmLokrModule

EPS = float(torch.finfo(torch.float32).eps)
MODE_WEIGHT_FN = "weight_function"
MODE_HOOK = "hook"


# ── 크로네커 ──

def kron4(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """``torch.kron(w1, w2)`` 와 비트 단위로 같은 값(곱 하나씩) — ``[a·b, c·d]``."""
    a, c = w1.shape
    b, d = w2.shape
    return (w1[:, None, :, None] * w2[None, :, None, :]).reshape(a * b, c * d)


def kron_mv(x2d: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """``x2d [N, c·d] @ kron(w1, w2)ᵀ`` → ``[N, a·b]`` (가중치를 만들지 않는다)."""
    a, c = w1.shape
    b, d = w2.shape
    n = x2d.shape[0]
    y = torch.matmul(x2d.reshape(n, c, d), w2.t())        # [N, c, b]
    return torch.matmul(w1, y).reshape(n, a * b)            # [N, a, b]


def _wide_factor(w: torch.Tensor) -> torch.Tensor:
    return w if w.dtype in (torch.float32, torch.float64) else w.float()


def expert_sum(experts: Sequence[tuple[torch.Tensor, torch.Tensor, float]]) -> torch.Tensor:
    """``Σ s_e·kron4(w1_e, w2_e)`` (fp32, fp64 인자면 fp64 — 스케일 순서대로 왼쪽부터 더한다, 학습기 rebuild 와 같은 순서).

    새 텐서 하나에 제자리로 모은다(IEEE 덧셈·곱셈은 교환 법칙이 정확히 성립 — 값은 ``total + s·kron`` 과 비트 단위로 같다)."""
    total = None
    for w1, w2, scale in experts:
        w1, w2 = _wide_factor(w1), _wide_factor(w2)
        if total is not None and w1.dtype != total.dtype:
            w1, w2 = w1.to(total.dtype), w2.to(total.dtype)
        term = kron4(w1, w2)
        if float(scale) != 1.0:
            term.mul_(float(scale))
        if total is None:
            total = term
        else:
            total.add_(term)
            del term
    assert total is not None
    return total


def _compute_dtype(weight: torch.Tensor) -> torch.dtype:
    return torch.float64 if weight.dtype == torch.float64 else torch.float32


def merged_weight(
    weight: torch.Tensor,
    experts: Sequence[tuple[torch.Tensor, torch.Tensor, float]],
    dora_scale: torch.Tensor | None,
    multiplier: float = 1.0,
    eps: float = EPS,
) -> torch.Tensor:
    """1단계 병합 가중치 ``W + μ·(D(W + ΣK) − W)`` (``weight`` 의 dtype). 계산은 fp32(fp64 입력이면 fp64).

    학습기 ``_rebuild_tsm_lokr`` 와 같은 순서: ``V = W + (ΣK)`` → ``V·(m / (‖V‖ + eps))`` → μ ≠ 1 이면 블렌드 → 캐스트.
    모두 ΣK 텐서 하나에 제자리로 한다 — 연산마다 피연산자가 같고 교환만 바뀌어 값은 비트 단위로 같고, 가중치 크기의 fp32
    임시 텐서는 ΣK 와 항 하나(가중치 원소당 8 바이트 + 결과 캐스트)뿐이다."""
    cdtype = _compute_dtype(weight)
    v = expert_sum(experts)
    if v.dtype != cdtype:
        v = v.to(cdtype)
    v.add_(weight)                                    # W + ΣK (bf16 W 는 fp32 로 올려 더한다 — 정확한 승격)
    if dora_scale is not None:
        m = dora_scale.to(device=v.device, dtype=cdtype)
        v.mul_(m / (torch.linalg.vector_norm(v, dim=1, keepdim=True) + eps))
    if float(multiplier) != 1.0:
        v.sub_(weight).mul_(float(multiplier)).add_(weight)
    return v.to(weight.dtype)


def core_weight(weight: torch.Tensor, core: tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """2단계의 ``W + C`` (``weight`` 의 dtype)."""
    cdtype = _compute_dtype(weight)
    v = expert_sum([core])
    if v.dtype != cdtype:
        v = v.to(cdtype)
    return v.add_(weight).to(weight.dtype)


def stage2_terms(
    weight: torch.Tensor,
    core: tuple[torch.Tensor, torch.Tensor, float],
    contexts: Sequence[tuple[torch.Tensor, torch.Tensor, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(a ``[out]``, b ``[n_ctx, out]``, c ``[n_ctx, n_ctx, out]``) — ``M = W + C`` 와 ``X_j`` 의 행별 내적(fp32/fp64)."""
    cdtype = _compute_dtype(weight)
    m = expert_sum([core])
    m = (m if m.dtype == cdtype else m.to(cdtype)).add_(weight)
    xs = [expert_sum([ctx]).to(cdtype) for ctx in contexts]
    a = (m * m).sum(1)
    out = m.shape[0]
    b = torch.empty((len(xs), out), dtype=cdtype, device=m.device)
    c = torch.empty((len(xs), len(xs), out), dtype=cdtype, device=m.device)
    for j, xj in enumerate(xs):
        b[j] = (m * xj).sum(1)
        for k in range(j, len(xs)):
            c[j, k] = (xj * xs[k]).sum(1)
            c[k, j] = c[j, k]
    return a, b, c


def stage2_row_scale(
    g: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, dora_scale: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """샘플별 DoRA 행 스케일 ``m / (√(a + 2 Σ g_j b_j + Σ g_j g_k c_jk) + eps)`` → ``[G, out]`` (g 의 dtype)."""
    g = g.to(a.dtype)
    m = dora_scale.to(device=a.device, dtype=a.dtype).reshape(1, -1)
    if g.shape[1] == 1:
        g0 = g[:, :1]
        q = a + 2 * g0 * b[0] + g0 * g0 * c[0, 0]
    else:
        q = a + 2 * (g @ b) + torch.einsum("bj,bk,jko->bo", g, g, c)
    return m / (torch.sqrt(q.clamp_min(0)) + eps)


def gates_from_parts(
    z: torch.Tensor, router: tuple[torch.Tensor, torch.Tensor, torch.Tensor], t: int, num_timesteps: int
) -> torch.Tensor:
    w, b, embed = router
    return F.linear(z, w, b) + embed[embed_row(t, num_timesteps)]


# ── 장치 사본 ──

class LokrCache:
    """세션 하나의 TSM-LoKr 장치 사본. 전문가 인자는 파일 dtype 그대로 (원래 모듈, 장치) 마다 지금 구간의 것만 쥔다 —
    구간이 바뀌면 그 모듈의 앞 구간 코어 전문가를 놓는다(문맥 전문가는 구간이 같으면 그대로). 약한 복사 블록(DoRA 를 뺀 보기)도
    원래 모듈의 사본을 같이 쓴다. 크로네커 행렬곱용 입력 dtype 사본은 필요할 때만(2단계 문맥 · 출력 훅) 만든다."""

    def __init__(self) -> None:
        self._experts: dict[tuple[int, str], dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, float]]] = {}
        self._cast: dict[tuple[int, str, torch.dtype], dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, float]]] = {}
        self._dora: dict[tuple[int, str], torch.Tensor] = {}
        self._routers: dict[tuple[int, torch.dtype, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._modules: dict[int, TsmLokrModule] = {}
        self.uploads = 0

    def _upload(self, tensor: torch.Tensor, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
        from .delta import upload

        return upload([tensor], 0, dtype or tensor.dtype, device)

    def experts(
        self, module: TsmLokrModule, indices: tuple[int, ...], device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        """스케일마다 ``indices`` 의 전문가 (w1, w2, scale) — 파일 dtype, 저랭크 인자는 파일 dtype 으로 곱해 둔다."""
        root = module.root
        key = (id(root), str(device))
        held = self._experts.get(key)
        wanted = {(s, i) for s, i in enumerate(indices)}
        if held is None:
            held = self._experts[key] = {}
            self._modules[id(root)] = root
        elif not wanted.issubset(held):
            for stale in [k for k in held if k not in wanted]:
                del held[stale]
            for cast_key, cast in self._cast.items():
                if cast_key[:2] == key:
                    for stale in [k for k in cast if k not in wanted]:
                        del cast[stale]
        result = []
        for s, i in enumerate(indices):
            entry = held.get((s, i))
            if entry is None:
                expert = root.experts[s][i]
                dtype = expert.tensors()[0].dtype
                if expert.w1 is not None and expert.w2 is not None:
                    w1 = self._upload(expert.w1, device)
                    w2 = self._upload(expert.w2, device)
                else:
                    wide = torch.float64 if dtype == torch.float64 else torch.float32
                    w1, w2 = expert.full_factors(wide, "cpu")
                    w1, w2 = self._upload(w1, device, dtype), self._upload(w2, device, dtype)
                entry = held[(s, i)] = (w1, w2, expert.scale)
                self.uploads += 1
            result.append(entry)
        return result

    def experts_as(
        self, module: TsmLokrModule, indices: tuple[int, ...], device: torch.device, dtype: torch.dtype, scales: Sequence[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        """``scales`` 의 전문가를 ``dtype`` 으로(크로네커 행렬곱의 입력 dtype). 파일 dtype 과 같으면 사본 없이 그대로."""
        base = self.experts(module, indices, device)
        if all(base[s][0].dtype == dtype for s in scales):
            return [base[s] for s in scales]
        root = module.root
        key = (id(root), str(device), dtype)
        held = self._cast.setdefault(key, {})
        result = []
        for s in scales:
            entry = held.get((s, indices[s]))
            if entry is None:
                w1, w2, scale = base[s]
                entry = held[(s, indices[s])] = (w1.to(dtype), w2.to(dtype), scale)
            result.append(entry)
        return result

    def dora(self, module: TsmLokrModule, device: torch.device) -> torch.Tensor | None:
        if module.dora_scale is None:
            return None
        root = module.root
        key = (id(root), str(device))
        hit = self._dora.get(key)
        if hit is None:
            wide = torch.float64 if root.dora_scale.dtype == torch.float64 else torch.float32
            hit = self._dora[key] = self._upload(root.dora_scale, device, wide)
            self._modules[id(root)] = root
        return hit

    def router(self, module: TsmLokrModule, dtype: torch.dtype, device: torch.device):
        root = module.root
        key = (id(root), dtype, str(device))
        hit = self._routers.get(key)
        if hit is None:
            hit = tuple(self._upload(t, device, dtype) for t in (root.router_w, root.router_b, root.embed))
            self._routers[key] = hit
            self._modules[id(root)] = root
        return hit

    def nbytes(self) -> int:
        total = 0
        for group in (*self._experts.values(), *self._cast.values()):
            for w1, w2, _s in group.values():
                total += w1.numel() * w1.element_size() + w2.numel() * w2.element_size()
        total += sum(t.numel() * t.element_size() for t in self._dora.values())
        total += sum(t.numel() * t.element_size() for parts in self._routers.values() for t in parts)
        return total

    def clear(self) -> bool:
        had = bool(self._experts or self._cast or self._dora or self._routers)
        self._experts.clear()
        self._cast.clear()
        self._dora.clear()
        self._routers.clear()
        self._modules.clear()
        return had


# ── 슬롯 ──

class LokrSlot:
    """대상 Linear 하나에 걸린 TSM-LoKr 하나. 세션이 만들고, Forge 가중치 함수(:meth:`weight_fn`)와 출력 훅
    (:meth:`finish`)이 쓴다. 장치 사본은 세션의 :class:`LokrCache`, 노름 항·병합 캐시는 이 슬롯(대상 Linear 의 W 로 셈)."""

    kind = "lokr"

    def __init__(self, module: TsmLokrModule, multiplier: float, stage: int, sig: int = 0, mode: str = MODE_WEIGHT_FN,
                 merge_cache: bool = False) -> None:
        self.module = module
        self.multiplier = float(multiplier)
        self.stage = int(stage)
        self.sig = sig
        self.ikey = 0
        self.mode = mode
        self.merge_cache = bool(merge_cache)
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.n_ctx = module.n_ctx
        self.gated = self.stage == 2 and module.n_ctx > 0
        self.forget()

    def forget(self) -> None:
        """장치 텐서 참조를 놓는다(세션 release)."""
        self._terms: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._norms: dict[tuple, torch.Tensor] = {}
        self._merged = None          # (key, tensor) — 병합 캐시
        self._call = None            # 가중치 함수가 돈 CallState
        self._call_terms = None      # 그 호출의 2단계 노름 항(가중치 함수가 받은 W 로 셈한 것)
        self._seen_w = None          # 행마다 t 가 달라 가중치 함수가 그대로 돌려준 W (그 호출의 출력 훅이 다시 쓰고 놓는다)
        self._cache_ok = False

    def holds_device_tensors(self) -> bool:
        return bool(self._terms or self._norms or self._merged is not None or self._seen_w is not None
                    or self._call_terms is not None)

    def nbytes(self) -> int:
        total = sum(t.numel() * t.element_size() for parts in self._terms.values() for t in parts)
        total += sum(t.numel() * t.element_size() for t in self._norms.values())
        if self._merged is not None:
            total += self._merged[1].numel() * self._merged[1].element_size()
        return total

    # 구성 요소

    def _experts(self, indices, device, cache: LokrCache):
        return cache.experts(self.module, indices, device)

    def _terms_for(self, weight: torch.Tensor, indices, cache: LokrCache, key_extra=()):
        key = (tuple(indices), weight.dtype, str(weight.device), *key_extra)
        hit = self._terms.get(key)
        if hit is None:
            experts = self._experts(indices, weight.device, cache)
            hit = self._terms[key] = stage2_terms(weight, experts[0], experts[1:])
        return hit

    def _norm_for(self, weight: torch.Tensor, indices, cache: LokrCache) -> torch.Tensor:
        """출력 훅 1단계: ``‖W + ΣK‖_행`` (fp32/fp64 ``[out]``)."""
        key = (tuple(indices), weight.dtype, str(weight.device))
        hit = self._norms.get(key)
        if hit is None:
            cdtype = _compute_dtype(weight)
            v = expert_sum(self._experts(indices, weight.device, cache))
            v = (v if v.dtype == cdtype else v.to(cdtype)).add_(weight)
            hit = self._norms[key] = torch.linalg.vector_norm(v, dim=1)
        return hit

    def gates(self, x: torch.Tensor, t: int, pooling: str, cache: LokrCache, pool: PoolCache | None) -> torch.Tensor:
        rdtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        z = pool.mean(x, self.in_features, pooling, rdtype) if pool is not None else pooled_mean(x, self.in_features, pooling, rdtype)
        router = cache.router(self.module, rdtype, z.device)
        return gates_from_parts(z, router, t, self.module.num_timesteps)

    # Forge 가중치 함수

    def weight_for(self, weight: torch.Tensor, indices: tuple[int, ...], cache: LokrCache) -> torch.Tensor:
        """모든 행이 같은 t 일 때 이 Linear 에 넘길 가중치(``weight`` 의 dtype). 2단계면 노름 항도 이 W 로 정해 둔다."""
        if self.gated and cache.dora(self.module, weight.device) is not None:
            self._call_terms = self._terms_for(weight, indices, cache)
        if self.merge_cache and self._cache_ok:
            key = (tuple(indices), weight.dtype, str(weight.device))
            if self._merged is not None and self._merged[0] == key:
                return self._merged[1]
        experts = self._experts(indices, weight.device, cache)
        if self.gated:
            result = core_weight(weight, experts[0])
        else:
            result = merged_weight(weight, experts, cache.dora(self.module, weight.device), self.multiplier)
        if self.merge_cache and self._cache_ok:
            self._merged = ((tuple(indices), weight.dtype, str(weight.device)), result)
        return result

    # 출력 — 출력 쪽 계산은 넓은 dtype(fp32, fp64 입력이면 fp64)의 누산기 하나에 모으고 마지막에 한 번만 반올림한다.

    @staticmethod
    def _wide(dtype: torch.dtype) -> torch.dtype:
        return torch.float64 if dtype == torch.float64 else torch.float32

    def _store(self, out: torch.Tensor, acc: torch.Tensor, bias: torch.Tensor | None, in_place: bool) -> torch.Tensor:
        """누산기(편향 없음 ``[N, out]``) → 출력 dtype. ``in_place`` 면 ``out`` 에 써서 그대로 돌려준다."""
        if bias is not None:
            acc.add_(bias.to(device=acc.device, dtype=acc.dtype))
        if in_place:
            out.view(-1, self.out_features).copy_(acc)
            return out
        return acc.to(out.dtype).view(out.shape)

    def _base(self, out: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        """편향을 뺀 출력 ``[N, out]`` (출력 dtype — 편향이 없으면 보기만)."""
        flat = out.reshape(-1, self.out_features)
        return flat if bias is None else flat - bias.to(flat.dtype)

    def _gate_acc(self, acc: torch.Tensor, base: torch.Tensor, x: torch.Tensor, t: int, indices, terms, pooling: str,
                  cache: LokrCache, pool: PoolCache | None, sub_core: bool) -> torch.Tensor:
        """2단계: ``acc = x (W+C)ᵀ`` (넓은 dtype) → 제자리로 ``W'_μ x``.

        ``D x = s ⊙ (x(W+C)ᵀ + Σ g_j x X_jᵀ)`` (샘플별 게이트 g, DoRA 행 스케일 s), ``W'_μ x = μ·D x + (1−μ)·W x``.
        ``W x`` 는 ``base`` (``sub_core`` 면 ``base − C x`` — 가중치 함수가 C 를 이미 합친 출력)."""
        width = self.out_features
        g = self.gates(x, t, pooling, cache, pool)                       # [G, n_ctx] fp32/fp64
        rows = g.shape[0]
        x2d = x.reshape(-1, self.in_features)
        av = acc.view(rows, -1, width)
        contexts = cache.experts_as(self.module, indices, x.device, x.dtype, range(1, 1 + self.n_ctx))
        for j, (w1, w2, scale) in enumerate(contexts):
            av.addcmul_(kron_mv(x2d, w1, w2).view(rows, -1, width), (g[:, j] * scale).to(acc.dtype).view(rows, 1, 1))
        m = cache.dora(self.module, x.device)
        if m is not None:
            a, b, c = terms
            s = stage2_row_scale(g.to(a.device), a, b, c, m)               # [G, out]
            av.mul_(s.to(device=acc.device, dtype=acc.dtype).view(rows, 1, width))
        mu = self.multiplier
        if mu != 1.0:
            acc.mul_(mu).add_(base, alpha=1.0 - mu)
            if sub_core:
                w1, w2, scale = cache.experts_as(self.module, indices, x.device, x.dtype, (0,))[0]
                acc.add_(kron_mv(x2d, w1, w2), alpha=-(1.0 - mu) * scale)
        return acc

    def _rows_from_weight(self, weight: torch.Tensor, x: torch.Tensor, t: int, pooling: str, cache: LokrCache,
                          bias: torch.Tensor | None) -> torch.Tensor:
        """기본 가중치 ``weight`` 에서 이 행들(모두 같은 t)의 출력을 처음부터 — 행마다 t 가 다른 호출용."""
        indices = self.module.expert_indices(t)
        if not self.gated:
            experts = self._experts(indices, weight.device, cache)
            w = merged_weight(weight, experts, cache.dora(self.module, weight.device), self.multiplier)
            return F.linear(x, w.to(x.dtype), None if bias is None else bias.to(x.dtype))
        y1 = F.linear(x, core_weight(weight, self._experts(indices, weight.device, cache)[0]).to(x.dtype))
        terms = self._terms_for(weight, indices, cache) if cache.dora(self.module, weight.device) is not None else None
        base = y1.reshape(-1, self.out_features)
        acc = self._gate_acc(base.to(self._wide(y1.dtype), copy=True), base, x, t, indices, terms, pooling, cache, None, True)
        return self._store(y1, acc, bias, in_place=True)

    def finish(self, out: torch.Tensor, x: torch.Tensor, call, t: int | None, ts: Sequence[int] | None, pooling: str,
               cache: LokrCache, pool: PoolCache | None, in_place: bool, target=None) -> torch.Tensor | None:
        """Linear 출력 → 이 TSM-LoKr 를 적용한 출력. 적용하지 못하면 None(호출자가 건너뛴 것으로 센다).

        ``t`` 는 모든 행이 같을 때의 t, 아니면 ``ts`` 가 행별 t. ``target`` 은 그 Linear(출력 훅 길의 ``weight`` · 편향)."""
        bias = getattr(target, "bias", None) if target is not None else None
        if bias is not None and not isinstance(bias, torch.Tensor):
            bias = None
        if self.mode == MODE_WEIGHT_FN:
            if self._call is not call:
                return None
            if t is not None:
                if not self.gated:
                    return out                      # 가중치 함수가 이미 병합 가중치로 계산했다
                terms, self._call_terms = self._call_terms, None
                if terms is None and self.module.has_dora:
                    return None
                base = self._base(out, bias)
                acc = self._gate_acc(base.to(self._wide(out.dtype), copy=True), base, x, t, self.module.expert_indices(t),
                                     terms, pooling, cache, pool, sub_core=True)
                return self._store(out, acc, bias, in_place)
            weight = self._seen_w
            self._seen_w = None
            if weight is None:
                return None
            return self._regroup(out, x, ts, weight, pooling, cache, bias, in_place)
        # 출력 훅 길 — 기본 가중치는 target.weight, out = x Wᵀ (+ 편향)
        weight = getattr(target, "weight", None)
        if not isinstance(weight, torch.Tensor):
            return None
        if t is None:
            return self._regroup(out, x, ts, weight, pooling, cache, bias, in_place)
        indices = self.module.expert_indices(t)
        x2d = x.reshape(-1, self.in_features)
        base = self._base(out, bias)
        acc = base.to(self._wide(out.dtype), copy=True)
        if self.gated:
            w1, w2, scale = cache.experts_as(self.module, indices, x.device, x.dtype, (0,))[0]
            acc.add_(kron_mv(x2d, w1, w2), alpha=scale)                        # x (W+C)ᵀ
            terms = self._terms_for(weight, indices, cache) if self.module.has_dora else None
            acc = self._gate_acc(acc, base, x, t, indices, terms, pooling, cache, pool, sub_core=False)
            return self._store(out, acc, bias, in_place)
        mu = self.multiplier
        m = cache.dora(self.module, x.device)
        experts = cache.experts_as(self.module, indices, x.device, x.dtype, range(len(self.module.n_scales)))
        if m is None:
            for w1, w2, scale in experts:                                       # W x + μ·K x
                acc.add_(kron_mv(x2d, w1, w2), alpha=mu * scale)
            return self._store(out, acc, bias, in_place)
        norm = self._norm_for(weight, indices, cache)
        s = (m.reshape(-1).to(norm.dtype) / (norm + EPS)).to(device=acc.device, dtype=acc.dtype)   # [out]
        acc.mul_((1.0 - mu) + mu * s)                                           # (1−μ)·W x + μ·s ⊙ W x
        for w1, w2, scale in experts:                                           # + μ·s ⊙ K x
            acc.addcmul_(kron_mv(x2d, w1, w2), (mu * scale) * s)
        return self._store(out, acc, bias, in_place)

    def _regroup(self, out, x, ts, weight, pooling, cache, bias, in_place):
        """행마다 t 가 다르면 같은 t 의 행 묶음마다 기본 가중치에서 다시 계산한다."""
        if ts is None or len(ts) != x.shape[0]:
            return None
        groups: "OrderedDict[int, list[int]]" = OrderedDict()
        for row, t in enumerate(ts):
            groups.setdefault(int(t), []).append(row)
        y = out if in_place else out.clone()
        for t, members in groups.items():
            index = torch.tensor(members, device=x.device, dtype=torch.long)
            rows = self._rows_from_weight(weight, x.index_select(0, index), t, pooling, cache, bias)
            y.index_copy_(0, index, rows.to(y.dtype))
        return y

    def note_weight(self, call, weight: torch.Tensor | None) -> None:
        """가중치 함수가 이 호출에서 돌았다고 적는다. 행마다 t 가 다르면(``weight`` 가 주어지면) 그 W 를 출력 훅까지 쥔다."""
        self._call = call
        self._seen_w = weight


def reference_output(
    module: TsmLokrModule,
    weight: torch.Tensor,
    x: torch.Tensor,
    ts: Sequence[int],
    stage: int,
    multiplier: float = 1.0,
    pooling: str = POOL_SAMPLE,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """기준 구현: 샘플(또는 묶음)마다 dense ``V_b`` 를 만들어 ``x W'_μᵀ`` (CPU 대조·행마다 t 가 다른 경우의 기준)."""
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
    cache = LokrCache()
    device = x.device
    out = torch.empty(tuple(x.shape[:-1]) + (module.out_features,), dtype=x.dtype, device=device)
    rows = list(range(x.shape[0]))
    groups = [[r] for r in rows] if pooling == POOL_SAMPLE or stage == 1 else None
    if groups is None:
        by_t: "OrderedDict[int, list[int]]" = OrderedDict()
        for r in rows:
            by_t.setdefault(int(ts[r]), []).append(r)
        groups = list(by_t.values())
    cdtype = _compute_dtype(weight)
    for members in groups:
        t = int(ts[members[0]])
        indices = module.expert_indices(t)
        experts = cache.experts(module, indices, device)
        xs = x[members]
        m = cache.dora(module, device)
        if stage == 1 or module.n_ctx == 0:
            w = merged_weight(weight, experts, m, multiplier)
        else:
            rdtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            z = xs.reshape(1, -1, module.in_features).to(rdtype).mean(1)
            g = gates_from_parts(z, cache.router(module, rdtype, device), t, module.num_timesteps)[0]
            v = weight.to(cdtype) + expert_sum([experts[0]]).to(cdtype)
            for j in range(module.n_ctx):
                v = v + g[j].to(cdtype) * expert_sum([experts[1 + j]]).to(cdtype)
            d = v * (m.to(cdtype) / (torch.linalg.vector_norm(v, dim=1, keepdim=True) + EPS)) if m is not None else v
            w0 = weight.to(cdtype)
            if multiplier != 1.0:
                d = w0 + multiplier * (d - w0)
            w = d.to(weight.dtype)
        out[members] = F.linear(xs, w.to(x.dtype), None if bias is None else bias.to(x.dtype))
    return out
