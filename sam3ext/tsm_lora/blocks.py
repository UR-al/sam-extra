"""TSM 모듈 이름 → Forge Anima DiT 의 Linear, 그리고 28/40/52 블록 변환.

LyCORIS(kohya) 이름은 ``lora_unet_`` + DiT 안의 모듈 경로에서 ``.`` 을 ``_`` 로 바꾼 것이다:
``lora_unet_blocks_3_adaln_modulation_mlp_2`` = ``blocks[3].adaln_modulation_mlp[2]``. 손으로 쓴 표 없이
``dit.named_modules()`` 를 같은 규칙으로 바꿔 찾는다.

블록 수가 다른 모델(2.9B 파일을 3.8B 에, 등)은 ``anima_lora_blocks.BLOCK_MAPPINGS`` (타깃 → 원본) 를 쓰고,
끼워 넣은 블록은 같은 정책을 따른다(``DoRA 추론 방식`` 아코디언의 '끼워 넣은 블록'):

* keep / additive — 앞 원본 블록의 TSM 을 그대로 쓴다(TSM 에는 DoRA 크기 보정이 없어 둘이 같다).
* skip — 끼워 넣은 블록에는 넣지 않는다.
* weak — 범위 안의 모듈에만 Δ 를 강도 배로 넣는다(라우터 게이트는 그대로, 2단계에서도 Δ 전체가 강도 배).

TSM-LoKr(DoRA 크기 ``dora_scale`` 이 있다)는 keep 이 크기까지 복사해 DoRA 를 적용하고, additive 는 끼워 넣은 블록에서 크기를
뺀다(``W + μ·ΔW_t``), weak 는 범위 안에서 크기 없이 ``W + 강도·ΔW_t`` 다(``TsmLokrModule.without_dora``).

완전한 레이아웃(28·40·52 블록이 빈틈없이)만 변환한다. 부분 파일은 순정 판정이 지금 모델과 같을 때만 그대로 쓰고,
나머지는 거부한다(블록 대응을 추측하지 않는다).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .. import anima_lora_blocks as alb
from .fileformat import TsmFile, TsmFormatError, TsmModule

LORA_PREFIX = "lora_unet_"
_BLOCK_NAME_RE = re.compile(r"^lora_unet_blocks_(\d+)_(.+)$")
# Forge 가 텍스트 인코더 쪽으로 옮기는 LLM 어댑터 — 거기에는 시간 단계가 없다.
_TEXT_SIDE_PREFIXES = ("lora_unet_llm_adapter", "lora_te_")


def split_block_name(name: str) -> tuple[int, str] | None:
    match = _BLOCK_NAME_RE.match(name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def module_key(path: str) -> str:
    """DiT 안의 모듈 경로 → LyCORIS 이름(``blocks.3.self_attn.q_proj`` → ``lora_unet_blocks_3_self_attn_q_proj``)."""
    return LORA_PREFIX + path.replace(".", "_")


def unwrap_dit(dit: Any) -> Any:
    """torch.compile 이 감싼 모듈이면 원래 모듈."""
    return getattr(dit, "_orig_mod", dit)


def is_anima_dit(dit: Any) -> bool:
    """Forge ``backend/nn/anima.py`` 의 ``Anima`` 모양인지(블록 목록 + self_attn.q_proj + adaLN 모듈레이션)."""
    dit = unwrap_dit(dit)
    blocks = getattr(dit, "blocks", None)
    try:
        first = blocks[0]
    except Exception:
        return False
    attn = getattr(first, "self_attn", None)
    return (
        attn is not None
        and hasattr(attn, "q_proj")
        and hasattr(first, "cross_attn")
        and hasattr(first, "adaln_modulation_mlp")
        and hasattr(first, "mlp")
    )


def dit_block_count(dit: Any) -> int:
    return len(unwrap_dit(dit).blocks)


def dit_module_table(dit: Any) -> dict[str, Any]:
    """{LyCORIS 이름: 모듈} — 이름이 있는 모든 하위 모듈(호출 가능한 잎이 아니어도 담는다)."""
    table: dict[str, Any] = {}
    for path, module in unwrap_dit(dit).named_modules():
        if path:
            table[module_key(path)] = module
    return table


@dataclass
class FilePlan:
    """파일 하나를 지금 DiT 에 거는 계획: {DiT 쪽 LyCORIS 이름: [(TsmModule, 배율)]}."""

    targets: dict[str, list[tuple[TsmModule, float]]] = field(default_factory=dict)
    source_blocks: int | None = None
    target_blocks: int = 0
    remapped: bool = False
    policy: str = alb.DUPLICATE_KEEP
    duplicated_blocks: int = 0
    skipped_inserted: int = 0
    text_side: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.remapped or self.source_blocks is None:
            return ""
        if self.source_blocks > self.target_blocks:
            return f"{self.source_blocks}->{self.target_blocks} blocks (dropping {self.source_blocks - self.target_blocks})"
        text = f"{self.source_blocks}->{self.target_blocks} blocks ({self.duplicated_blocks} inserted: {self.policy}"
        if self.skipped_inserted:
            text += f", {self.skipped_inserted} modules left out"
        return text + ")"


def plan_file(
    tsm: TsmFile,
    target_blocks: int,
    policy: str | None = None,
    weak: tuple[float, str] | None = None,
) -> FilePlan:
    """모듈 이름을 타깃 블록 수에 맞춘다. 레이아웃을 정할 수 없으면 :class:`TsmFormatError`."""
    policy = alb.normalize_duplicate_policy(policy) or alb.current_duplicate_policy()
    weak_strength, weak_scope = weak if weak is not None else alb.current_weak_copy()
    plan = FilePlan(target_blocks=int(target_blocks), policy=policy)
    by_block: dict[int, list[tuple[str, TsmModule]]] = {}
    for name, module in tsm.modules.items():
        if name.startswith(_TEXT_SIDE_PREFIXES):
            plan.text_side.append(name)
            continue
        split = split_block_name(name)
        if split is None:
            plan.targets.setdefault(name, []).append((module, 1.0))
            continue
        by_block.setdefault(split[0], []).append((split[1], module))
    if not by_block:
        return plan

    indices = set(by_block)
    layout = alb.detect_anima_lora_layout({f"lora_unet_blocks_{i}_x": None for i in indices})
    inferred = alb.infer_forge_lora_layout(indices)
    plan.source_blocks = layout if layout is not None else inferred
    same = layout == target_blocks or (layout is None and inferred == target_blocks)
    if same or target_blocks not in alb.SUPPORTED_BLOCK_COUNTS:
        if not same and max(indices) >= target_blocks:
            raise TsmFormatError(
                f"블록 {max(indices)} 이 있는데 지금 모델은 {target_blocks}블록입니다",
                f"file has block {max(indices)} but the model has {target_blocks} blocks",
            )
        for index, entries in by_block.items():
            for suffix, module in entries:
                plan.targets.setdefault(f"lora_unet_blocks_{index}_{suffix}", []).append((module, 1.0))
        return plan
    if layout is None:
        raise TsmFormatError(
            f"부분 TSM 파일입니다(블록 {min(indices)}~{max(indices)} 중 {len(indices)}개) — "
            f"{target_blocks}블록 모델과의 블록 대응을 정할 수 없습니다",
            f"sparse TSM file ({len(indices)} blocks between {min(indices)} and {max(indices)}): "
            f"cannot map it onto a {target_blocks}-block model",
        )
    mapping = alb.BLOCK_MAPPINGS[(layout, target_blocks)]
    plan.remapped = True
    emitted: set[int] = set()
    for target_index, source_index in enumerate(mapping):
        entries = by_block.get(source_index, ())
        inserted = source_index in emitted
        emitted.add(source_index)
        if inserted:
            plan.duplicated_blocks += 1
        for suffix, module in entries:
            factor = 1.0
            if inserted and policy == alb.DUPLICATE_SKIP:
                plan.skipped_inserted += 1
                continue
            if inserted and policy == alb.DUPLICATE_WEAK:
                if not alb._in_weak_scope(suffix, weak_scope):
                    plan.skipped_inserted += 1
                    continue
                factor = float(weak_strength)
            if inserted and policy in (alb.DUPLICATE_ADDITIVE, alb.DUPLICATE_WEAK) and hasattr(module, "without_dora"):
                module = module.without_dora()   # TSM-LoKr: 끼워 넣은 블록은 DoRA 크기 없이
            plan.targets.setdefault(f"lora_unet_blocks_{target_index}_{suffix}", []).append((module, factor))
    return plan


@dataclass
class Binding:
    """DiT 모듈에 붙일 TSM: {id(모듈): (모듈, [(TsmModule, 배율, 단계)])}."""

    modules: dict[int, tuple[Any, list[tuple[TsmModule, float, int]]]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    shape_mismatch: list[str] = field(default_factory=list)

    def add(self, target: Any, item: tuple[TsmModule, float, int]) -> None:
        self.modules.setdefault(id(target), (target, []))[1].append(item)

    def __len__(self) -> int:
        return len(self.modules)


def _shape_ok(target: Any, module: TsmModule) -> bool:
    """Linear 모양을 확인할 수 있으면 확인한다. GGUF·혼합 정밀도 Linear 처럼 모를 때는 통과(실행 때 x 너비를 다시 본다)."""
    fan_in = getattr(target, "in_features", None)
    fan_out = getattr(target, "out_features", None)
    if isinstance(fan_in, int) and isinstance(fan_out, int):
        return (fan_in, fan_out) == (module.in_features, module.out_features)
    weight = getattr(target, "weight", None)
    shape = tuple(getattr(weight, "shape", ()) or ())
    if len(shape) == 2 and getattr(weight, "dtype", None) is not None and str(weight.dtype).startswith("torch.float"):
        return shape == (module.out_features, module.in_features)
    return True


def bind_plan(plan: FilePlan, table: dict[str, Any], multiplier: float, stage: int, binding: Binding | None = None) -> Binding:
    binding = binding if binding is not None else Binding()
    for name, items in plan.targets.items():
        target = table.get(name)
        if target is None or not callable(target):
            binding.missing.append(name)
            continue
        for module, factor in items:
            if not _shape_ok(target, module):
                binding.shape_mismatch.append(name)
                continue
            binding.add(target, (module, float(multiplier) * factor, stage))
    return binding
