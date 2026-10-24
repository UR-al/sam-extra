"""TSM 런타임 — 생성 한 번의 세션, DiT·Linear 훅, Forge LoRA 로더 가로채기.

흐름(Forge 쪽 연결은 ``scripts/tsm_lora.py``):

1. ``<lora:이름:강도>`` 로 TSM 파일을 적으면 Forge ``networks.load_networks`` 가 파일을 읽어
   ``load_lora_for_models`` 를 부른다. :func:`install_loader_hook` 이 감싼 그 함수가 TSM 키를 떼어 내
   (``fileformat.split_state_dict``) 모델에 기록하고(:func:`records_for` — Forge ``current_lora_hash`` 별), 남은 키만
   순정 로더에 넘긴다. 해시가 같아 Forge 가 다시 읽지 않을 때도, 하이레스 LoRA(다른 해시)도 기록으로 찾는다.
2. 샘플링 패스마다(``process_before_every_sampling``) 기록 + 아코디언에서 고른 파일로 :class:`TsmSession` 을 만들어
   UNet 복제본의 ``transformer_options`` 에 넣고, DiT 에 훅을 건다(:class:`HookManager`).
3. DiT forward pre-hook 이 그 호출의 ``transformer_options`` 에서 세션을, 두 번째 인자(timestep = σ·multiplier)에서
   행별 t 를 읽어(호스트 동기화는 DiT 호출마다 이 한 번) (T, dtype) 묶음마다 정수 t 로 바꿔 스레드 로컬에 둔다 — 세션이
   없는 호출(Tile-Repair 의 벤더 루프 등)은 비운다. 대상 Linear 의 forward hook 이 그때만 Linear 출력에 Δ 를 제자리로
   더한다(:class:`~.delta.Slot`). post-hook 이 다시 비운다.
4. 샘플러의 ``launch_sampling`` 이 끝나면(:func:`wrap_sampler_release` — 정상 · 중단 · 예외 모두, VAE 디코드 전) 장치
   캐시를 놓는다. 다음 패스(하이레스)는 다시 만든다. ``postprocess`` 에서 훅을 뗀다.

Forge 의 일반 LoRA 는 가중치에 합쳐지거나(온라인·lowvram 이면 Linear 안에서) 그대로 돌고, TSM 은 그 Linear 의
출력에 Δ 를 더하므로 둘이 학습 때처럼 더해진다.

TSM-LoKr(``algo=tsm-lokr``, :mod:`.lokr_delta`)은 DoRA 노름이 병합 가중치 전체에 걸려 출력에 Δ 를 더하는 식으로는 안
된다. Forge Linear(``weight_function`` 목록이 있는 것)에는 forward pre-hook 이 인스턴스 목록 끝에 우리 가중치 함수를
붙여(클래스 목록은 건드리지 않는다) Forge 가 그 호출의 가중치로 병합 가중치 ``W + μ·(D(W + ΔW_t) − W)`` 를 쓰게 하고,
2단계는 출력 훅이 게이트 항과 샘플별 DoRA 행 스케일을 더한다. ``weight_function`` 이 없는 Linear 는 출력 훅만으로
(크로네커 곱 행렬곱 + 노름) 계산한다 — 어느 길인지는 바인딩 때 정해지고 세션 로그에 한 번 적는다. 가중치 함수는 세션이
없으면 받은 가중치를 그대로 돌려주고, 장치 캐시를 놓을 때(:meth:`TsmSession.release`) · 훅을 뗄 때 목록에서 빠진다.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import weakref
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Iterable

import torch

from . import blocks as tb
from . import delta as td
from . import lokr_delta as tld
from .fileformat import TsmFile, TsmFormatError, split_state_dict, tsm_file_from_state_dict

LOGGER = logging.getLogger("sam-extra.tsm-lora")

OPTION_KEY = "sam3_tsm"           # transformer_options 키
RECORDS_ATTR = "_sam3_tsm_lora_records"
SOURCE_PROMPT = "prompt"
SOURCE_PICKER = "picker"

STAGE_AUTO = "auto"
STAGE_1 = "1"
STAGE_2 = "2"
STAGE_OFF = "off"
STAGE_CHOICES = (STAGE_AUTO, STAGE_1, STAGE_2, STAGE_OFF)


def _emit(message: str) -> None:
    """생성 경로에서 부르므로 cp949 콘솔의 UnicodeEncodeError 로 생성을 죽이지 않는다."""
    text = f"[TSM LoRA] {message}"
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            print(text.encode(encoding, "backslashreplace").decode(encoding, "replace"))
        except Exception:
            pass
    except Exception:
        pass


_LOG: Callable[[str], None] = _emit


def set_logger(fn: Callable[[str], None] | None) -> None:
    global _LOG
    _LOG = fn or _emit


def normalize_stage(value: Any) -> str | None:
    """단계 덮어쓰기 값·UI 라벨·XYZ 값·infotext 값 → auto / 1 / 2 / off. 모르면 None."""
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return None
    if text in STAGE_CHOICES:
        return text
    if text.startswith("auto") or "자동" in text:
        return STAGE_AUTO
    if text.startswith("off") or "끔" in text:
        return STAGE_OFF
    if text.startswith(("1", "s1", "stage 1", "stage1")) or "1단계" in text:
        return STAGE_1
    if text.startswith(("2", "s2", "stage 2", "stage2")) or "2단계" in text:
        return STAGE_2
    return None


def normalize_pooling(value: Any) -> str | None:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return None
    if text.startswith(td.POOL_SAMPLE) or "샘플" in text or text.startswith("per"):
        return td.POOL_SAMPLE
    if text.startswith(td.POOL_BATCH) or "배치" in text or "joint" in text:
        return td.POOL_BATCH
    return None


def resolve_stage(tsm: TsmFile, override: str | None) -> int | None:
    """파일 단계와 덮어쓰기 → 쓸 단계. ``off`` 면 None."""
    override = normalize_stage(override) or STAGE_AUTO
    if override == STAGE_OFF:
        return None
    if override == STAGE_1:
        return 1
    if override == STAGE_2:
        return 2
    return tsm.stage


# ── 세션 ──

@dataclass
class TsmEntry:
    tsm: TsmFile
    multiplier: float
    source: str
    stage: int = 1
    label: str = ""

    def describe(self) -> str:
        return f"{self.label or self.tsm.name}:{self.multiplier:g}"


@dataclass
class BindReport:
    entries: list[TsmEntry] = field(default_factory=list)
    modules: int = 0
    missing: int = 0
    shape_mismatch: int = 0
    refused: list[tuple[str, str]] = field(default_factory=list)   # (파일 이름, 사유)
    notes: list[str] = field(default_factory=list)
    lokr_modes: dict[str, int] = field(default_factory=dict)      # TSM-LoKr 계산 길별 모듈 수(weight_function / hook)
    unsupported: int = 0                                           # TSM-LoKr 를 걸 수 없는 모듈(실수 가중치가 없는 순정 Linear)


def lokr_mode_for(target: Any) -> str | None:
    """TSM-LoKr 를 이 Linear 에 거는 길. Forge Linear(``weight_function`` 목록) → 가중치 함수, 실수 2차원 가중치가 있는
    다른 Linear → 출력 훅, 둘 다 아니면 None(걸 수 없음)."""
    if isinstance(getattr(target, "weight_function", None), list):
        return tld.MODE_WEIGHT_FN
    weight = getattr(target, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.dim() == 2 and weight.dtype.is_floating_point:
        return tld.MODE_HOOK
    return None


class TsmSession:
    """생성의 한 샘플링 패스. DiT 모듈 id → 그 모듈에 걸린 TSM 목록과 장치 캐시를 쥔다."""

    def __init__(
        self,
        entries: Iterable[TsmEntry],
        pooling: str = td.POOL_SAMPLE,
        timestep_multiplier: float = 1.0,
        lokr_merge_cache: bool = False,
    ) -> None:
        self.entries = list(entries)
        self.pooling = pooling if pooling in td.POOLINGS else td.POOL_SAMPLE
        self.timestep_multiplier = float(timestep_multiplier or 1.0)
        self.lokr_merge_cache = bool(lokr_merge_cache)
        self.cache = td.FusedCache()
        self.lokr_cache = tld.LokrCache()
        self.pool = td.PoolCache()
        self.dit_id: int | None = None
        self.bound: dict[int, list[tuple[Any, float, int]]] = {}
        self.slots: dict[int, tuple[td.Slot, ...]] = {}   # 훅이 쓰는 것 — 강도 0 은 뺐다(TSM LoRA 전문가)
        self.lokr_slots: dict[int, tuple[tld.LokrSlot, ...]] = {}   # TSM-LoKr — 모듈마다 하나
        self.sigs: list[tuple[int, str]] = []            # (T, 학습 timestep dtype) 묶음 — Slot.sig 가 가리킨다
        self.index_keys: list[tuple[int, tuple[int, ...]]] = []   # (sig, 스케일 구성) — Slot.ikey 가 가리킨다
        self.targets: dict[int, Any] = {}
        self.report = BindReport()
        self.calls = 0          # Δ 를 더한 DiT 호출 수
        self.skipped_calls = 0  # 세션은 있는데 t 를 못 읽거나 배치가 안 맞아 건너뛴 Linear 호출
        self.seen_t: set[int] = set()
        self._warned: set[str] = set()
        self._t_memo: dict[tuple[tuple[float, ...], int, str], list[int]] = {}

    # 바인딩

    def bind(self, dit: Any, policy: str | None = None, weak: tuple[float, str] | None = None) -> BindReport:
        dit = tb.unwrap_dit(dit)
        report = BindReport()
        self.dit_id = id(dit)
        self.bound.clear()
        self.slots.clear()
        self.lokr_slots.clear()
        self.sigs.clear()
        self.index_keys.clear()
        self.targets.clear()
        if not tb.is_anima_dit(dit):
            report.refused = [(entry.tsm.name, "not an Anima DiT") for entry in self.entries]
            self.report = report
            return report
        table = tb.dit_module_table(dit)
        target_blocks = tb.dit_block_count(dit)
        binding = tb.Binding()
        for entry in self.entries:
            try:
                plan = tb.plan_file(entry.tsm, target_blocks, policy, weak)
            except TsmFormatError as exc:
                report.refused.append((entry.tsm.name, str(exc)))
                continue
            before_missing = len(binding.missing)
            before_shape = len(binding.shape_mismatch)
            trial = tb.bind_plan(plan, table, entry.multiplier, entry.stage)
            unmatched = len(trial.missing) + len(trial.shape_mismatch)
            total = sum(len(v) for v in plan.targets.values()) or 1
            if unmatched / total > 0.5:
                example = (trial.missing + trial.shape_mismatch)[0]
                report.refused.append(
                    (entry.tsm.name, f"{unmatched}/{total} modules do not match this DiT (e.g. {example})")
                )
                continue
            if getattr(entry.tsm, "expert_kind", "lora") == "lokr":
                # DoRA 노름은 병합 가중치 전체에 걸린다 — 한 Linear 에 TSM-LoKr 둘을 겹치는 식은 정하지 않았다.
                taken = [key for key in trial.modules if any(
                    getattr(m, "kind", "lora") == "lokr" for m, _mult, _st in binding.modules.get(key, (None, []))[1])]
                if taken:
                    report.refused.append(
                        (entry.tsm.name, f"another TSM-LoKr file already uses {len(taken)} of these modules "
                                         "(one TSM-LoKr per Linear)")
                    )
                    continue
            tb.bind_plan(plan, table, entry.multiplier, entry.stage, binding)
            report.missing += len(binding.missing) - before_missing
            report.shape_mismatch += len(binding.shape_mismatch) - before_shape
            report.entries.append(entry)
            if plan.describe():
                report.notes.append(f"{entry.tsm.name}: {plan.describe()}")
            if plan.text_side:
                report.notes.append(f"{entry.tsm.name}: {len(plan.text_side)} text-encoder modules ignored (no timestep there)")
        for key, (module, items) in binding.modules.items():
            lora_items = [item for item in items if getattr(item[0], "kind", "lora") != "lokr"]
            lokr_items = [item for item in items if getattr(item[0], "kind", "lora") == "lokr"]
            mode = lokr_mode_for(module) if lokr_items else None
            if lokr_items and mode is None:
                report.unsupported += 1
                lokr_items = []
                if not lora_items:
                    continue
            self.targets[key] = module
            self.bound[key] = lora_items + lokr_items
            slots = tuple(self._slot(m, mult, stage) for m, mult, stage in lora_items if mult != 0)
            if slots:
                self.slots[key] = slots
            lokr = tuple(self._slot(m, mult, stage, mode) for m, mult, stage in lokr_items if mult != 0)
            if lokr:
                self.lokr_slots[key] = lokr
                report.lokr_modes[mode] = report.lokr_modes.get(mode, 0) + 1
        if report.unsupported:
            report.notes.append(f"TSM-LoKr: {report.unsupported} modules have no float weight or weight_function - left out")
        if report.lokr_modes:
            report.notes.append(
                "TSM-LoKr: " + ", ".join(f"{count} modules via {mode}" for mode, count in sorted(report.lokr_modes.items()))
            )
        report.modules = len(self.bound)
        self.report = report
        return report

    def _slot(self, module: Any, multiplier: float, stage: int, mode: str | None = None):
        key = (int(module.num_timesteps), str(module.timestep_dtype))
        if key not in self.sigs:
            self.sigs.append(key)
        if getattr(module, "kind", "lora") == "lokr":
            slot = tld.LokrSlot(module, multiplier, stage, self.sigs.index(key), mode or tld.MODE_WEIGHT_FN,
                                self.lokr_merge_cache)
        else:
            slot = td.Slot(module, multiplier, stage, self.sigs.index(key))
        ikey = (slot.sig, tuple(module.n_scales))
        if ikey not in self.index_keys:
            self.index_keys.append(ikey)
        slot.ikey = self.index_keys.index(ikey)
        return slot

    def transient_bytes(self, tokens: int, element_size: int = 2) -> int:
        """Linear 호출 하나가 잠깐 더 쓰는 메모리의 위 한계(호출은 차례로 돌아 서로 쌓이지 않는다 — 가장 큰 것 하나).

        ``tokens`` = 배치 × 토큰 수. 중간값 ``mid [tokens, S·r]`` 에, 출력에 제자리로 더하지 못할 때 만드는 새 출력
        ``[tokens, out]`` 을 더한다. 2단계 라우터 평균은 ``[배치, in]`` (fp32) 뿐이라 작다 — 입력의 fp32 사본은 없다."""
        worst = 0
        seen: set[int] = set()
        for items in self.bound.values():
            for module, _mult, _stage in items:
                if id(module) in seen:
                    continue
                seen.add(id(module))
                if getattr(module, "kind", "lora") == "lokr":
                    # 가중치 쪽: 2단계 노름 항(구간마다 처음 한 번 — fp32 M·X·곱 셋)이 병합(ΣK·항 둘 + 결과 캐스트)보다 크다.
                    # 출력 쪽: fp32 누산기 + 크로네커 항 + 새 출력 [tokens, out].
                    weight = module.out_features * module.in_features
                    worst = max(worst, weight * (12 + element_size), int(tokens) * module.out_features * (4 + 2 * element_size))
                    continue
                width = module.rank * len(module.n_scales)
                worst = max(worst, int(tokens) * (width + module.out_features) * element_size)
        return worst

    def fused_bytes(self, element_size: int = 2) -> int:
        seen: set[int] = set()
        total = 0
        for items in self.bound.values():
            for module, _mult, _stage in items:
                if id(module) not in seen:
                    seen.add(id(module))
                    total += module.fused_bytes(element_size)
        if self.lokr_merge_cache:
            # 병합 캐시: 대상 Linear 마다 지금 구간의 병합 가중치 하나
            total += sum(slot.module.merged_bytes(element_size) for slots in self.lokr_slots.values() for slot in slots)
        return total

    # 실행

    def timestep_rows(self, timesteps: Any) -> list[float] | None:
        if timesteps is None:
            return None
        try:
            return td.timestep_values(timesteps, self.timestep_multiplier)
        except Exception:
            return None

    def warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            _LOG(message)

    def begin_call(self, values: list[float]) -> "CallState":
        """DiT 호출 하나의 행별 t — 묶음(T, dtype)마다 한 번 셈해 둔다(Linear 마다 다시 셈하지 않는다)."""
        self.pool.clear()
        ts = [tuple(self.rows_t(values, T, dtype)) for T, dtype in self.sigs]
        uniform = [row[0] if len(set(row)) == 1 else None for row in ts]
        indices = []
        for sig, n_scales in self.index_keys:
            t, T = uniform[sig], self.sigs[sig][0]
            indices.append(None if t is None else tuple(td.expert_index(t, n, T) for n in n_scales))
        return CallState(len(values), ts, uniform, indices)

    def apply(self, module_id: int, x: Any, out: Any, call: "CallState | None", target: Any = None) -> torch.Tensor | None:
        """Linear 훅: 출력에 Δ 를 더한 텐서(대개 ``out`` 자체를 제자리로 고친 것). 걸 것이 없으면 None.

        모든 행이 같은 t 면(보통 — cond·uncond 는 같은 σ) :meth:`~.delta.Slot.add` 로, 행마다 t 가 다르면 기준 식
        :func:`~.delta.module_delta` 로 더한다. 건너뛰는 경우는 :meth:`delta_for` 와 같다. TSM-LoKr 는 먼저
        :meth:`~.lokr_delta.LokrSlot.finish` 로(가중치 함수가 이미 병합했으면 2단계 게이트 항만), ``target`` 은 그 Linear."""
        slots = self.slots.get(module_id)
        lokr = self.lokr_slots.get(module_id)
        if (slots is None and lokr is None) or call is None or not isinstance(x, torch.Tensor) or x.dim() < 2:
            return None
        if not isinstance(out, torch.Tensor):
            return None
        result = None
        in_place = None
        if lokr is not None:
            result = self._apply_lokr(lokr, x, out, call, target)
            if result is not None:
                in_place = td.can_add_in_place(result, x)
        for slot in slots or ():
            if x.shape[-1] != slot.in_features:
                self.skipped_calls += 1
                self.warn_once(
                    f"width:{slot.module.name}",
                    f"{slot.module.name}: 입력 너비 {x.shape[-1]} 가 TSM 의 {slot.module.in_features} 와 달라 건너뜁니다",
                )
                continue
            current = out if result is None else result
            t = call.uniform[slot.sig]
            if t is not None:
                if in_place is None:
                    in_place = td.can_add_in_place(current, x)
                indices = call.indices[slot.ikey] if call.indices else None
                result = slot.add(current, x, t, self.pooling, self.cache, self.pool, in_place, indices)
                if result is not current:   # 새로 만든 결과 — 다음 TSM 이 제자리로 더해도 되는지 다시 본다
                    in_place = td.can_add_in_place(result, x)
                continue
            ts = call.ts[slot.sig]
            if len(ts) != x.shape[0]:
                self.skipped_calls += 1
                self.warn_once(
                    "batch",
                    f"배치 {x.shape[0]} 와 timestep {len(ts)} 개가 맞지 않아 이 호출은 TSM 없이 돕니다",
                )
                continue
            delta = td.module_delta(slot.module, x, list(ts), slot.stage, slot.multiplier, self.pooling, self.cache)
            result = current + delta.to(current.dtype)
            in_place = td.can_add_in_place(result, x)
        return result

    def _apply_lokr(self, slots, x: torch.Tensor, out: torch.Tensor, call: "CallState", target: Any) -> torch.Tensor | None:
        result = None
        for slot in slots:
            current = out if result is None else result
            if x.shape[-1] != slot.in_features:
                self.skipped_calls += 1
                self.warn_once(
                    f"width:{slot.module.name}",
                    f"{slot.module.name}: 입력 너비 {x.shape[-1]} 가 TSM-LoKr 의 {slot.in_features} 와 달라 건너뜁니다",
                )
                continue
            t = call.uniform[slot.sig]
            ts = call.ts[slot.sig]
            if t is None and len(ts) != x.shape[0]:
                self.skipped_calls += 1
                self.warn_once("batch", f"배치 {x.shape[0]} 와 timestep {len(ts)} 개가 맞지 않아 이 호출은 TSM 없이 돕니다")
                continue
            done = slot.finish(current, x, call, t, None if t is not None else list(ts), self.pooling, self.lokr_cache,
                               self.pool, td.can_add_in_place(current, x), target)
            if done is None:
                self.skipped_calls += 1
                self.warn_once(
                    f"lokr:{slot.mode}",
                    f"{slot.module.name}: TSM-LoKr 를 적용하지 못했습니다(Forge 가 이 Linear 의 weight_function 을 부르지 않음 — "
                    "다른 확장이 Linear forward 를 바꿨을 수 있음). 이 호출은 TSM-LoKr 없이 돕니다",
                )
                continue
            result = done
        return result

    def delta_for(self, module_id: int, x: torch.Tensor, values: list[float]) -> torch.Tensor | None:
        items = self.bound.get(module_id)
        if not items:
            return None
        if not isinstance(x, torch.Tensor) or x.dim() < 2:
            return None
        batch = x.shape[0]
        if len(values) != batch:
            if len(set(values)) == 1:
                values = values[:1] * batch
            else:
                self.skipped_calls += 1
                self.warn_once(
                    "batch",
                    f"배치 {batch} 와 timestep {len(values)} 개가 맞지 않아 이 호출은 TSM 없이 돕니다",
                )
                return None
        result = None
        for module, multiplier, stage in items:
            if multiplier == 0 or getattr(module, "kind", "lora") != "lora":
                continue   # TSM-LoKr 에는 출력에 더하는 Δ 가 없다(DoRA 가 병합 가중치 전체에 걸린다)
            if x.shape[-1] != module.in_features:
                self.skipped_calls += 1
                self.warn_once(
                    f"width:{module.name}",
                    f"{module.name}: 입력 너비 {x.shape[-1]} 가 TSM 의 {module.in_features} 와 달라 건너뜁니다",
                )
                continue
            ts = self.rows_t(values, module.num_timesteps, module.timestep_dtype)
            delta = td.module_delta(module, x, ts, stage, multiplier, self.pooling, self.cache)
            result = delta if result is None else result + delta
        return result

    def rows_t(self, values: list[float], num_timesteps: int, timestep_dtype: str) -> list[int]:
        """행별 정수 t — DiT 호출 하나에서 모든 Linear 가 같은 값을 쓰므로 (값, T, dtype) 별로 한 번만 계산한다."""
        key = (tuple(values), int(num_timesteps), str(timestep_dtype))
        ts = self._t_memo.get(key)
        if ts is None:
            if len(self._t_memo) > 64:
                self._t_memo.clear()
            ts = [td.t_from_value(v, num_timesteps, timestep_dtype) for v in values]
            self._t_memo[key] = ts
            self.seen_t.update(ts)
        return ts

    def release(self) -> bool:
        """장치 사본을 놓는다 — 캐시와, 그것을 가리키던 Slot 참조·라우터 평균까지(남은 참조가 있으면 VRAM 이 안 풀린다).
        TSM-LoKr 의 장치 사본·노름 항·병합 캐시도 놓고, 그 Linear 들의 weight_function 목록에서 우리 함수를 뺀다
        (다음 패스의 호출이 있으면 pre-hook 이 다시 붙인다)."""
        held = False
        for slots in self.slots.values():
            for slot in slots:
                held = slot.holds_device_tensors() or held
                slot.forget()
        for slots in self.lokr_slots.values():
            for slot in slots:
                held = slot.holds_device_tensors() or held
                slot.forget()
        if self.lokr_slots:
            WEIGHT_FNS.remove(self.lokr_slots.keys())
        held = self.lokr_cache.clear() or held
        self.pool.clear()
        return self.cache.clear() or held

    def device_bytes(self) -> int:
        """지금 쥔 장치 사본의 크기(LoRA 전문가 캐시 + TSM-LoKr 사본·노름 항·병합 캐시)."""
        total = self.cache.nbytes() + self.lokr_cache.nbytes()
        return total + sum(slot.nbytes() for slots in self.lokr_slots.values() for slot in slots)


class CallState:
    """DiT 호출 하나의 t. ``ts[sig]`` 행별 t, ``uniform[sig]`` 모든 행이 같으면 그 t(아니면 None),
    ``indices[ikey]`` 그 t 의 스케일별 전문가 번호(행마다 t 가 다르면 None)."""

    __slots__ = ("batch", "ts", "uniform", "indices")

    def __init__(self, batch: int, ts: list[tuple[int, ...]], uniform: list[int | None],
                 indices: list[tuple[int, ...] | None] | None = None) -> None:
        self.batch = batch
        self.ts = ts
        self.uniform = uniform
        self.indices = indices if indices is not None else []


# ── 훅 ──

class _Active(threading.local):
    def __init__(self) -> None:
        self.session: TsmSession | None = None
        self.values: list[float] | None = None
        self.call: CallState | None = None


_ACTIVE = _Active()


def active_session() -> TsmSession | None:
    return _ACTIVE.session


def _dit_pre_hook(module, args, kwargs):
    options = kwargs.get("transformer_options") if isinstance(kwargs, dict) else None
    session = options.get(OPTION_KEY) if isinstance(options, dict) else None
    if not isinstance(session, TsmSession) or not session.bound:
        _ACTIVE.session, _ACTIVE.values, _ACTIVE.call = None, None, None
        return None
    timesteps = args[1] if len(args) > 1 else kwargs.get("timesteps")
    values = session.timestep_rows(timesteps)
    if not values:
        session.warn_once("timestep", "DiT 호출에 timestep 이 없어 TSM 없이 돕니다(학습의 'T 로 대체'는 따르지 않음)")
        _ACTIVE.session, _ACTIVE.values, _ACTIVE.call = None, None, None
        return None
    session.calls += 1
    _ACTIVE.session, _ACTIVE.values, _ACTIVE.call = session, values, session.begin_call(values)
    return None


def _dit_post_hook(module, args, output):
    """``always_call`` 로 걸어 DiT forward 가 던져도(OOM 등) 돈다 — 그때 ``output`` 은 None 이다.

    던졌으면 이 생성의 postprocess 가 불리지 않을 수 있으므로 세션의 장치 캐시를 바로 놓는다(다음 호출이 있으면
    :class:`~.delta.FusedCache` 가 다시 만든다)."""
    session = _ACTIVE.session
    _ACTIVE.session, _ACTIVE.values, _ACTIVE.call = None, None, None
    if session is not None:
        session.pool.clear()
    if output is None and session is not None:
        try:
            session.release()
        except Exception:
            pass
    return None


def _register_post_hook(dit: Any, hook: Callable) -> Any:
    try:
        return dit.register_forward_hook(hook, always_call=True)
    except TypeError:  # pragma: no cover - torch < 2.0
        return dit.register_forward_hook(hook)


def _linear_hook(module, args, output):
    session = _ACTIVE.session
    if session is None or not args:
        return None
    return session.apply(id(module), args[0], output, _ACTIVE.call, module)


# ── TSM-LoKr: Forge weight_function ──

_MISSING = object()


class _LokrWeightFn:
    """Forge ``Linear.weight_function`` 목록 끝에 붙는 우리 함수(Linear 마다 하나). 활성 세션이 그 Linear 에 TSM-LoKr 를
    걸었을 때만 병합 가중치를 돌려주고, 아니면 받은 가중치를 그대로 돌려준다."""

    __slots__ = ("module_id", "__weakref__")
    _sam3_tsm_lokr = True

    def __init__(self, module_id: int) -> None:
        self.module_id = module_id

    def __call__(self, weight):
        session = _ACTIVE.session
        call = _ACTIVE.call
        if session is None or call is None or not isinstance(weight, torch.Tensor):
            return weight
        slots = session.lokr_slots.get(self.module_id)
        if not slots:
            return weight
        for slot in slots:
            if slot.mode != tld.MODE_WEIGHT_FN:
                continue
            t = call.uniform[slot.sig]
            if t is None:
                slot.note_weight(call, weight)   # 행마다 t 가 다르다 — 출력 훅이 행 묶음마다 이 W 로 다시 계산한다
                continue
            indices = call.indices[slot.ikey] if call.indices else slot.module.expert_indices(t)
            weight = slot.weight_for(weight, indices, session.lokr_cache)
            slot.note_weight(call, None)
        return weight


class WeightFnRegistry:
    """우리가 Forge Linear 에 붙인 가중치 함수들. 클래스 목록(``ForgeWeights.weight_function``)은 건드리지 않고 늘
    인스턴스 목록을 새로 만들어 넣는다. 뗄 때는 지금 목록에서 우리 함수만 뺀다(그 사이 Forge 가 목록을 바꿨어도 Forge 의
    것은 그대로) — 처음에 인스턴스 목록이 없었고 남는 것이 없으면 인스턴스 속성을 지워 클래스 목록으로 돌아간다."""

    def __init__(self) -> None:
        self._entries: dict[int, tuple[Callable[[], Any], _LokrWeightFn, bool]] = {}

    def ensure(self, module: Any) -> tuple[_LokrWeightFn, bool]:
        """목록 끝에 우리 함수가 있게 한다. (함수, 목록에 우리 함수만 있는가)."""
        key = id(module)
        entry = self._entries.get(key)
        if entry is None or entry[0]() is not module:
            had_instance = isinstance(getattr(module, "__dict__", {}).get("weight_function", _MISSING), list)
            entry = (_weak(module), _LokrWeightFn(key), had_instance)
            self._entries[key] = entry
        fn = entry[1]
        current = getattr(module, "weight_function", None)
        if not isinstance(current, list):
            return fn, False
        if not current or current[-1] is not fn:
            current = [f for f in current if f is not fn] + [fn]
            module.weight_function = current
        return fn, len(current) == 1

    def installed(self, module: Any) -> bool:
        entry = self._entries.get(id(module))
        current = getattr(module, "weight_function", None)
        return entry is not None and isinstance(current, list) and any(f is entry[1] for f in current)

    def remove(self, module_ids: Iterable[int] | None = None) -> int:
        keys = list(self._entries) if module_ids is None else [k for k in module_ids if k in self._entries]
        removed = 0
        for key in keys:
            ref, fn, had_instance = self._entries.pop(key)
            module = ref()
            if module is None:
                continue
            current = module.__dict__.get("weight_function", _MISSING) if hasattr(module, "__dict__") else _MISSING
            if not isinstance(current, list) or not any(f is fn for f in current):
                continue
            rest = [f for f in current if f is not fn]
            if not rest and not had_instance:
                try:
                    del module.weight_function
                except AttributeError:
                    module.weight_function = rest
            else:
                module.weight_function = rest
            removed += 1
        return removed

    def __len__(self) -> int:
        return len(self._entries)


WEIGHT_FNS = WeightFnRegistry()


def _lokr_pre_hook(module, args):
    session = _ACTIVE.session
    if session is None:
        return None
    slots = session.lokr_slots.get(id(module))
    if not slots or slots[0].mode != tld.MODE_WEIGHT_FN:
        return None
    _fn, only = WEIGHT_FNS.ensure(module)
    for slot in slots:
        slot._cache_ok = only   # 병합 캐시는 Forge 가 넘기는 W 가 이 Linear 의 가중치 그대로일 때(다른 함수 없음)만
    return None


def _weak(module: Any) -> Callable[[], Any]:
    try:
        return weakref.ref(module)
    except TypeError:  # pragma: no cover - nn.Module 은 weakref 를 받는다
        return lambda: module


def _remove_handles(handles: Iterable[Any]) -> None:
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass


class HookManager:
    """DiT 마다 pre/post 훅 한 쌍 + 세션이 묶은 Linear 마다 forward 훅 하나. 같은 모듈에 두 번 걸지 않는다.

    모듈은 약한 참조로만 쥔다(하이레스 체크포인트 전환 때 앞 DiT 를 붙잡지 않는다). 키는 ``id`` 라서, 앞 모듈이 풀린
    뒤 새 모듈이 같은 id 를 받으면 참조가 가리키는 객체로 가려 다시 건다.
    """

    def __init__(self) -> None:
        self._dit_handles: dict[int, tuple[Callable[[], Any], list[Any]]] = {}
        self._linear_handles: dict[int, tuple[Callable[[], Any], Any]] = {}
        self._pre_handles: dict[int, tuple[Callable[[], Any], Any]] = {}   # TSM-LoKr 가중치 함수를 붙이는 pre-hook

    def attach(self, dit: Any, session: TsmSession) -> int:
        dit = tb.unwrap_dit(dit)
        key = id(dit)
        held = self._dit_handles.get(key)
        if held is None or held[0]() is not dit:
            if held is not None:
                _remove_handles(held[1])
            self._dit_handles[key] = (
                _weak(dit),
                [dit.register_forward_pre_hook(_dit_pre_hook, with_kwargs=True), _register_post_hook(dit, _dit_post_hook)],
            )
        added = 0
        for module_id, module in session.targets.items():
            held_linear = self._linear_handles.get(module_id)
            if held_linear is not None and held_linear[0]() is module:
                continue
            if held_linear is not None:
                _remove_handles([held_linear[1]])
            self._linear_handles[module_id] = (_weak(module), module.register_forward_hook(_linear_hook))
            added += 1
        for module_id, slots in session.lokr_slots.items():
            module = session.targets.get(module_id)
            if module is None or slots[0].mode != tld.MODE_WEIGHT_FN:
                continue
            held_pre = self._pre_handles.get(module_id)
            if held_pre is not None and held_pre[0]() is module:
                continue
            if held_pre is not None:
                _remove_handles([held_pre[1]])
            self._pre_handles[module_id] = (_weak(module), module.register_forward_pre_hook(_lokr_pre_hook))
        return added

    def installed(self) -> bool:
        return bool(self._dit_handles or self._linear_handles or self._pre_handles)

    def pre_hook_count(self) -> int:
        return len(self._pre_handles)

    def counts(self) -> tuple[int, int]:
        return len(self._dit_handles), len(self._linear_handles)

    def remove_all(self) -> bool:
        had = self.installed()
        for _ref, handles in self._dit_handles.values():
            _remove_handles(handles)
        _remove_handles(handle for _ref, handle in self._linear_handles.values())
        _remove_handles(handle for _ref, handle in self._pre_handles.values())
        self._dit_handles.clear()
        self._linear_handles.clear()
        self._pre_handles.clear()
        WEIGHT_FNS.remove()
        _ACTIVE.session, _ACTIVE.values, _ACTIVE.call = None, None, None
        return had


HOOKS = HookManager()
_SESSIONS: list[TsmSession] = []


def remember_session(session: TsmSession) -> None:
    """샘플링 패스는 차례로 돈다 — 새 패스가 시작되면 앞 패스들(앞 batch count·1차 패스)의 장치 캐시는 다시 쓰이지
    않으므로 놓는다. 그러지 않으면 batch count × (1차 + 하이레스) 만큼 캐시가 VRAM 에 쌓인다. 세션 자체(보고용 통계)는
    postprocess 까지 남긴다."""
    for previous in _SESSIONS:
        if previous is not session:
            previous.release()
    if session not in _SESSIONS:
        _SESSIONS.append(session)


def release_device_caches() -> bool:
    """세션들의 장치 캐시만 놓는다(훅·세션은 그대로). 패스의 샘플링이 끝나면 VAE 디코드 전에 부른다
    (:func:`wrap_sampler_release`). 다음 DiT 호출이 있으면 다시 만든다."""
    released = False
    for session in _SESSIONS:
        released = session.release() or released
    return released


SAMPLER_MARK = "_sam3_tsm_release_after_sampling"


def wrap_sampler_release(sampler: Any) -> str:
    """Forge 샘플러 객체의 ``launch_sampling`` 을 감싸, 그 패스의 샘플링이 끝나면 장치 캐시를 놓는다.

    Forge 의 모든 샘플러(k-diffusion · CompVis)는 ``sample`` · ``sample_img2img`` 안에서 ``self.launch_sampling(steps,
    func)`` 로 스텝을 돌리고, 돌아온 뒤에야 ``processing`` 이 VAE 로 디코드한다(1차 패스 뒤 latent 가 아닌 업스케일러의
    디코드, 하이레스 패스 안의 디코드, img2img · txt2img 의 마지막 디코드). ``post_sample`` 은 하이레스 디코드 뒤에 불려
    늦다. 인스턴스 속성으로 감싸므로 그 샘플러 객체에만 걸리고, 중단(``InterruptedException`` 은 launch_sampling 이 삼킨다)
    · 예외에도 ``finally`` 로 놓는다. 같은 객체(img2img 의 batch count)에는 한 번만 감싼다.

    돌려주는 값: ``"installed"`` · ``"already"`` · ``"unavailable"`` (샘플러가 없거나 감쌀 수 없음 — post_sample 이 대신)."""
    original = getattr(sampler, "launch_sampling", None) if sampler is not None else None
    if not callable(original):
        return "unavailable"
    if getattr(sampler, SAMPLER_MARK, False):
        return "already"

    def launch_sampling(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            try:
                release_device_caches()
            except Exception:
                LOGGER.debug("TSM cache release after sampling failed", exc_info=True)

    launch_sampling._sam3_tsm_release = True
    launch_sampling.__wrapped__ = original
    try:
        sampler.launch_sampling = launch_sampling
        setattr(sampler, SAMPLER_MARK, True)
    except Exception:
        return "unavailable"
    return "installed"


def finish_generation() -> tuple[list[TsmSession], bool]:
    """세션들의 장치 캐시를 놓고 훅을 뗀다. (끝난 세션들, 훅이 있었는지)."""
    sessions = list(_SESSIONS)
    _SESSIONS.clear()
    for session in sessions:
        session.release()
    return sessions, HOOKS.remove_all()


def release_all() -> bool:
    """'모델 내리기' — 훅·장치 캐시·읽어 둔 파일을 모두 놓는다."""
    from .fileformat import release_cache

    sessions, hooked = finish_generation()
    cached = release_cache()
    return bool(sessions) or hooked or cached


# ── Forge LoRA 로더 가로채기 ──

_LOADER_OWNER = "sam-extra.tsm-lora.v1"
_LOADER_MODULE: ModuleType | None = None


def _record_store(model: Any) -> dict | None:
    """모델에 적힌 기록이 지금 LoRA 묶음(current_lora_hash)의 것이면 그 dict."""
    store = getattr(model, RECORDS_ATTR, None) if model is not None else None
    if not isinstance(store, dict):
        return None
    if store.get("hash") != getattr(model, "current_lora_hash", object()):
        return None
    return store


def record_prompt_tsm(model: Any, filename: str, strength: float, tsm: TsmFile | None, error: str = "") -> None:
    """``load_networks`` 가 새 해시를 모델에 적은 뒤 파일마다 부르는 로더 안에서 부른다. 해시가 바뀌면 새로 시작하고,
    같은 해시를 다시 읽으면(캐시 무효화 뒤) 파일 이름별로 덮어쓴다."""
    if model is None:
        return
    store = _record_store(model)
    if store is None:
        store = {"hash": getattr(model, "current_lora_hash", None), "items": {}}
    store["items"][os.path.normcase(os.path.abspath(filename)) if filename else "?"] = {
        "filename": filename,
        "strength": float(strength),
        "tsm": tsm,
        "error": error,
    }
    try:
        setattr(model, RECORDS_ATTR, store)
    except Exception:
        LOGGER.debug("could not record TSM LoRA on the model", exc_info=True)


def records_for(model: Any) -> list[dict]:
    """지금 LoRA 묶음에서 ``<lora:…>`` 로 읽힌 TSM 파일 기록. 옛 묶음이면 빈 목록."""
    store = _record_store(model)
    return list(store["items"].values()) if store else []


def forget_records(model: Any) -> None:
    try:
        if model is not None and hasattr(model, RECORDS_ATTR):
            delattr(model, RECORDS_ATTR)
    except Exception:
        pass


def _forge_sd_model() -> Any:
    try:
        from modules import shared

        return getattr(shared, "sd_model", None)
    except Exception:
        return None


def _is_anima_loader_run(module: ModuleType, model: Any) -> bool:
    dynamic = getattr(getattr(module, "dynamic_args", None), "anima", None)
    if dynamic is not None:
        return bool(dynamic)
    dit = getattr(getattr(model, "model", None), "diffusion_model", None)
    return tb.is_anima_dit(dit)


def make_loader_wrapper(original: Callable, module: ModuleType, notify: Callable[[str], None] | None = None) -> Callable:
    def load_lora_for_models(model, clip, lora, strength_model, strength_clip, filename="default", online_mode=False, *args, **kwargs):
        try:
            tsm_part, rest = split_state_dict(lora) if isinstance(lora, dict) else ({}, lora)
        except Exception:
            tsm_part, rest = {}, lora
        if not tsm_part:
            return original(model, clip, lora, strength_model, strength_clip, filename, online_mode, *args, **kwargs)
        base = os.path.basename(str(filename))
        sd_model = _forge_sd_model()
        if not _is_anima_loader_run(module, model):
            message = f"'{base}' 은(는) TSM 파일인데 지금 모델이 Anima 가 아니라 건너뜁니다"
            _LOG(message)
            if notify is not None:
                notify(message)
            record_prompt_tsm(sd_model, str(filename), strength_model, None, "not an Anima model")
        else:
            try:
                tsm = tsm_file_from_state_dict(str(filename), tsm_part)
                record_prompt_tsm(sd_model, str(filename), strength_model, tsm)
                _LOG(
                    f"'{base}' ({tsm.summary()}, t {tsm.timestep_dtype}, {len(tsm.modules)} modules) — "
                    f"sam-extra 가 샘플링 중에 적용합니다(강도 {strength_model:g})"
                )
                for warning in tsm.warnings:
                    _LOG(f"'{base}': {warning}")
            except TsmFormatError as exc:
                message = f"'{base}' TSM 파일을 읽지 못해 건너뜁니다: {exc}"
                _LOG(message)
                if notify is not None:
                    notify(message)
                record_prompt_tsm(sd_model, str(filename), strength_model, None, str(exc))
        if rest:
            return original(model, clip, rest, strength_model, strength_clip, filename, online_mode, *args, **kwargs)
        return model, clip

    load_lora_for_models._sam3_tsm_owner = _LOADER_OWNER
    load_lora_for_models._sam3_tsm_original = original
    load_lora_for_models.__wrapped__ = original
    return load_lora_for_models


def install_loader_hook(module: ModuleType | None = None, notify: Callable[[str], None] | None = None) -> bool:
    """Forge ``networks.load_lora_for_models`` 를 감싼다(멱등). 새로 감쌌으면 True."""
    global _LOADER_MODULE
    if module is None:
        from .. import anima_lora_blocks as alb

        module = alb._find_forge_lora_module()
    if module is None or not callable(getattr(module, "load_lora_for_models", None)):
        return False
    current = module.load_lora_for_models
    if getattr(current, "_sam3_tsm_owner", None) == _LOADER_OWNER:
        _LOADER_MODULE = module
        return False
    module.load_lora_for_models = make_loader_wrapper(current, module, notify)
    _LOADER_MODULE = module
    return True


def loader_hook_installed(module: ModuleType | None = None) -> bool:
    module = module or _LOADER_MODULE
    current = getattr(module, "load_lora_for_models", None) if module is not None else None
    return getattr(current, "_sam3_tsm_owner", None) == _LOADER_OWNER


def uninstall_loader_hook(module: ModuleType | None = None) -> bool:
    """우리 래퍼가 맨 위에 있을 때만 원래 함수로 되돌린다(다른 확장이 위에 감쌌으면 그대로 둔다)."""
    global _LOADER_MODULE
    module = module or _LOADER_MODULE
    if module is None:
        return False
    current = getattr(module, "load_lora_for_models", None)
    if getattr(current, "_sam3_tsm_owner", None) != _LOADER_OWNER:
        return False
    module.load_lora_for_models = current._sam3_tsm_original
    if module is _LOADER_MODULE:
        _LOADER_MODULE = None
    return True
