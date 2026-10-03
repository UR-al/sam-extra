"""Forge 가 UI 를 만들 때 always-on 스크립트 묶음에 이름표를 붙인다.

프런트엔드(javascript/notebook_lanes.js)는 이 클래스만 보고 항목을 알아본다. 라벨 글자는 ko_KR 이 바꾸고
``component-N`` id 는 실행마다 달라지므로 둘 다 쓰지 않는다. 키는 스크립트 파일 이름에서 만든다 — 영어이고,
확장을 다시 설치해도 같다.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

ANIMA_SECTION = "sam3_anima"
OPT_LAYOUT_SECTIONS = "sam3_layout_sections"
ANIMA_PRIORITY = -35


def sections_enabled() -> bool:
    """설정이 없거나 Forge 밖(테스트)이면 켜진 것으로 본다."""
    try:
        from modules import shared

        return bool(getattr(shared.opts, OPT_LAYOUT_SECTIONS, True))
    except Exception:
        return True


def anima_section(is_img2img: bool) -> str | None:
    """Anima 튜닝 스크립트가 들어갈 Forge 사용자 섹션. img2img 는 지금 그대로 둔다."""
    if is_img2img or not sections_enabled():
        return None
    return ANIMA_SECTION


def anima_priority(is_img2img: bool, fallback: int) -> int:
    """섹션이 켜졌을 때만 Anima 3.8B 가 묶음 맨 앞에 온다(UI 순서에만 쓰이는 값)."""
    if is_img2img or not sections_enabled():
        return fallback
    return ANIMA_PRIORITY


def slot_key(filename: str) -> str:
    """스크립트 파일 이름 → 안정적인 키. ``!adetailer.py`` → ``adetailer``."""
    stem = os.path.splitext(os.path.basename(str(filename or "")))[0]
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")


@dataclass(frozen=True)
class OnRule:
    """켜짐 체크박스를 어떻게 찾을지. 하나만 채운다."""

    elem_class: str | None = None      # 이 CSS 클래스를 단 체크박스 전부
    label_equals: str | None = None    # Python 라벨이 정확히 이것
    label_prefix: str | None = None    # Python 라벨이 이것으로 시작
    none: bool = False                 # 켜짐 개념 없음 — 표시하지 않는다


@dataclass(frozen=True)
class Slot:
    lane: str
    exp: bool = False
    on: OnRule | None = None


# 묶음(lane): anima = 1열 ANIMA 튜닝, det = 2열 디테일러, lora·etc = 2열 "더 보기", hidden = 컨트롤 없는 빈 묶음.
# sam-extra 자기 스크립트는 켜기 체크박스에 소스에서 직접 sam3-on 클래스를 달아 두므로 on 규칙이 필요 없다.
REGISTRY: dict[str, Slot] = {
    "anima-3-8b": Slot("anima"),
    "anima-detail-daemon": Slot("anima"),
    "anima-skimmed-cfg": Slot("anima"),
    "anima-safe-pag": Slot("anima"),
    # CFG-Zero* optimized-scale(실험). ANIMA 섹션에 들어가는데 등록이 없으면 "도구·실험" 으로 분류돼 더 보기를 닫으면
    # 숨고, 바로 아래에 오도록 정렬한 Colorcraft 와 떨어진다.
    "anima-cfg-optimal-scale": Slot("anima", exp=True),
    "colorcraft": Slot("anima"),       # 샘플링 중 latent 색 보정(모든 모델 — 벡터는 Anima·Flux·Flux2 계열)
    "anima-vae-2x": Slot("anima", exp=True),
    "anima-speed": Slot("anima", exp=True),   # SPEED — 저해상도 선행 샘플링(실험)
    "anima-extra-schedulers": Slot("anima", on=OnRule(none=True)),   # custom·Laplace 값만 — 켜짐 개념 없음
    # 샘플러 값(ER SDE max stage·eta)만 담은 묶음이라 켜짐 개념이 없다 — 샘플러 드롭다운에서 고르면 쓰인다.
    # 파일은 anima_extra_samplers.py — Panchovix/sd_forge_neo_extra_samplers 가 이미 scripts/extra_samplers.py 다.
    "anima-extra-samplers": Slot("anima", on=OnRule(none=True)),
    "sam3": Slot("det"),
    "anima-vae-degrid": Slot("det"),   # 저장 직전 후처리 — SAM3·ADetailer 와 같은 묶음
    "anima-ref-poc": Slot("etc"),
    "adetailer": Slot("det"),
    "lora-block-weight": Slot("lora"),
    "dora-infer-mode": Slot("lora"),
    "controlnet": Slot("lora", on=OnRule(elem_class="cnet-unit-enabled")),
    "dynamic-prompting": Slot("lora", on=OnRule(label_equals="Dynamic Prompts enabled")),
    "dynamic-thresholding": Slot("lora", on=OnRule(elem_class="dynthres-enabled")),
    "img2img-hires-fix": Slot("etc"),
    "model-keyword": Slot("etc", on=OnRule(label_prefix="Model Keyword Enabled")),
    "api-payload-display": Slot("etc", on=OnRule(none=True)),
    "sparse": Slot("etc"),
    "forge-never-oom": Slot("etc", on=OnRule(label_prefix="Enabled for ")),
    "image-stitch": Slot("etc"),
    "spectrum": Slot("etc"),
    "compile": Slot("etc", on=OnRule(none=True)),
}

DEFAULT_LANE = "etc"


def lane_for(key: str) -> str:
    """등록표에 없는 확장은 "도구·실험" 으로 간다."""
    slot = REGISTRY.get(key)
    return slot.lane if slot else DEFAULT_LANE


_GUESS_RE = re.compile(r"^(enable|enabled)\b|\benabled\s*$", re.IGNORECASE)


def _add_classes(component, *names) -> None:
    """Gradio 는 elem_classes 를 None·문자열·리스트로 들고 있을 수 있다. 속성 자체를 새로 대입해야 설정에 실린다."""
    current = getattr(component, "elem_classes", None)
    if isinstance(current, str):
        current = [current]
    values = list(current or [])
    for name in names:
        if name not in values:
            values.append(name)
    component.elem_classes = values


def _has_class(component, name: str) -> bool:
    current = getattr(component, "elem_classes", None)
    if isinstance(current, str):
        return current == name
    return name in (current or [])


def _walk(block, depth: int = 0, limit: int = 6):
    for child in getattr(block, "children", None) or []:
        yield child, depth
        if depth < limit:
            yield from _walk(child, depth + 1, limit)


def _head_accordion(group, controls):
    """헤더로 쓸 아코디언. InputAccordion 모양이면 그 아코디언, 아니면 깊이 3 이내 첫 아코디언."""
    import gradio as gr

    for control in controls or []:
        accordion = getattr(control, "accordion", None)
        if accordion is not None and getattr(control, "accordion_id", None):
            return accordion
    best = None
    for child, depth in _walk(group, limit=3):
        if isinstance(child, gr.Accordion) and (best is None or depth < best[1]):
            best = (child, depth)
    return best[0] if best else None


def _mark_on_controls(controls, rule: OnRule | None) -> tuple[bool, bool]:
    """(표시할 켜짐 컨트롤이 있는가, 추측으로 찾았는가)"""
    import gradio as gr

    if any(_has_class(c, "sam3-on") or _has_class(c, "sam3-on-radio") for c in controls or []):
        return True, False          # sam-extra 가 소스에서 직접 붙인 경우
    if rule is not None and rule.none:
        return False, False
    if rule is not None:
        found = False
        for control in controls or []:
            label = str(getattr(control, "label", "") or "").strip()
            if rule.elem_class and _has_class(control, rule.elem_class):
                _add_classes(control, "sam3-on")
                found = True
            elif rule.label_equals and label == rule.label_equals:
                _add_classes(control, "sam3-on")
                found = True
            elif rule.label_prefix and label.startswith(rule.label_prefix.strip()):
                _add_classes(control, "sam3-on")
                found = True
        if found:
            return True, False
    for control in controls or []:          # InputAccordion 의 숨은 값 체크박스가 제출되는 값이다
        if getattr(control, "accordion_id", None) and getattr(control, "accordion", None) is not None:
            _add_classes(control, "sam3-on")
            return True, False
    for control in controls or []:          # 마지막 수단: 이름이 Enable… 인 체크박스
        if isinstance(control, gr.Checkbox) and _GUESS_RE.search(str(getattr(control, "label", "") or "")):
            _add_classes(control, "sam3-on")
            return True, True
    return False, False


def tag_runner(runner) -> int:
    """always-on 스크립트 묶음에 sam3-slot / sam3-slot--키 / sam3-lane--묶음 클래스를 붙이고, 붙인 수를 돌려준다.

    라벨·elem_id·값·컨트롤 순서는 건드리지 않는다. 여러 번 불러도 결과가 같다.
    """
    tagged = 0
    seen: dict[str, int] = {}
    for script in getattr(runner, "alwayson_scripts", None) or []:
        group = getattr(script, "group", None)
        if group is None or not getattr(script, "create_group", True):
            continue
        if getattr(script, "section", None) not in (None, ANIMA_SECTION):
            continue                        # 샘플러·시드·Hires 같은 Forge 기본 섹션은 건드리지 않는다
        base_key = slot_key(getattr(script, "filename", ""))
        if not base_key:
            continue
        seen[base_key] = seen.get(base_key, 0) + 1
        key = base_key
        if seen[base_key] > 1:
            # 확장이 자기 UI 를 두 번 만드는 경우(예: LoRA Block Weight). 핀·칩이 섞이지 않게 키는 다르게 주되,
            # 묶음은 처음 것과 같은 자리에 둔다 — 안 그러면 두 번째만 "도구·실험" 맨 아래로 떨어진다.
            key = f"{base_key}--{seen[base_key]}"
        controls = list(getattr(script, "controls", None) or [])
        lane = "hidden" if not controls else lane_for(base_key)
        slot = REGISTRY.get(base_key)
        _add_classes(group, "sam3-slot", f"sam3-slot--{key}", f"sam3-lane--{lane}")
        if slot is not None and slot.exp:
            _add_classes(group, "sam3-exp")
        head = _head_accordion(group, controls)
        if head is not None:
            _add_classes(head, "sam3-head")
        if lane != "hidden":
            _, guessed = _mark_on_controls(controls, slot.on if slot else None)
            if guessed:
                _add_classes(group, "sam3-on-guess")
        tagged += 1
    for script in getattr(runner, "selectable_scripts", None) or []:
        # 드롭다운으로 고르는 스크립트 패널. 키도 묶음도 없고, CSS 가 2열 "스크립트" 자리에 놓을 수 있게만 표시한다.
        group = getattr(script, "group", None)
        if group is not None:
            _add_classes(group, "sam3-script-panel")
    return tagged
