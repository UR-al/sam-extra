"""'기본값 저장' 버튼의 프리셋 쪽 — 지금 UI 값을 활성 UI Preset 의 Forge 설정 22개에 쓴다(Gradio·Forge 를 import 하지 않는 순수 로직).

왜 필요한가: Forge 는 페이지를 열 때와 UI Preset 을 바꿀 때마다 modules_forge/main_entry.py 의 on_preset_change 로
체크포인트·VAE/TE·Low Bits·스텝·샘플러·스케줄러·크기·CFG·Distilled CFG(Shift)·배치 22칸을 그 프리셋의 설정
(``forge_checkpoint_{p}`` … ``{p}_i2i_batch_size``)으로 덮어쓴다. 그래서 Settings → Defaults 의 Apply(ui-config)만
저장하면 이 칸들은 다음에 열 때 프리셋 값으로 돌아간다 — 버튼은 Apply 를 누르고(javascript/notebook_save_defaults.js)
같은 클릭에서 이 모듈로 활성 프리셋의 22개 설정도 쓴다.

값은 main_entry 가 쓰는 바로 그 컴포넌트에서 온다(``FIELDS`` 의 순서 = forge_main_entry 의 output_targets 순서).
저장 형식은 on_preset_change 가 다시 읽어 드롭다운·슬라이더에 넣을 수 있는 그대로다:

* ``forge_checkpoint_{p}`` — 체크포인트 드롭다운 값 그대로(checkpoint_change 처럼).
* ``forge_additional_modules_{p}`` — modules_change 처럼 파일 이름을 main_entry.module_list 의 전체 경로로 바꿔 정렬한
  목록(on_preset_change 는 basename 으로 되돌려 드롭다운에 넣는다).
* ``forge_unet_storage_dtype_{p}`` — Low Bits 선택지 이름.
* 샘플러는 이름, 스케줄러는 표시 이름(label) — 설정의 드롭다운 선택지(list_samplers / list_schedulers)가 그렇다.
* 스텝·크기·배치는 정수, CFG·Distilled CFG(Shift)는 실수. 범위는 그 설정의 슬라이더 범위(presets.register)로 본다.

프리셋 값 0 은 "덮어쓰지 않음"이다: on_preset_change 는 스텝 3 · 크기 4 · CFG 3 칸(``SKIP_IF_NOT_POSITIVE``)을
``v > 0`` 일 때만 화면에 넣고 아니면 ``gr.skip()`` 으로 화면 값을 둔다. 그런 칸의 저장된 값이 Forge 가 건너뛰는 값(0 이하 ·
NaN — ``forge_skips``)이면 버튼은 그 값을 그대로 두고(``Plan.kept``, 바뀐 항목으로 세지 않음) 화면 값은 같은 클릭의
Defaults Apply 가 ui-config 에 저장한다 — 화면 값으로 바꿔 쓰면 0 이 뜻하던 "화면 값 유지"가 "이 값으로 덮어쓰기"로 바뀐다.
그대로 두는 칸은 프리셋에 쓰지 않으므로 슬라이더 범위 검사도 하지 않는다. 이미 0 보다 큰 칸은 예전처럼 화면 값을 쓴다
(화면 값이 0 이면 0 — 다음부터 덮어쓰지 않음).

활성 프리셋(UI Preset 드롭다운 값)만 쓰고 다른 프리셋은 건드리지 않는다. Forge 에 등록되지 않은 키(예: Distilled
CFG·Shift 가 없는 sd/klein/qwen 의 ``{p}_*_dcfg``)는 건너뛴다 — on_preset_change 는 그 값을 쓰지 않거나(슬라이더를
숨김) 기본값으로 읽는다. 하나라도 검사를 통과하지 못하면 아무것도 쓰지 않는다. 쓰는 도중(opts.set / opts.save)에
실패하면 메모리의 값도 되돌린다.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

# (설정 키 틀, 종류, 메시지에 쓰는 이름) — modules_forge/main_entry.py forge_main_entry 의 output_targets 순서
FIELDS: tuple[tuple[str, str, str], ...] = (
    ("forge_checkpoint_{p}", "checkpoint", "Checkpoint"),
    ("forge_additional_modules_{p}", "modules", "VAE / Text Encoder"),
    ("forge_unet_storage_dtype_{p}", "dtype", "Diffusion in Low Bits"),
    ("{p}_t2i_step", "int", "txt2img Steps"),
    ("{p}_t2i_hr_step", "int", "txt2img Hires steps"),
    ("{p}_i2i_step", "int", "img2img Steps"),
    ("{p}_t2i_sampler", "sampler", "txt2img Sampler"),
    ("{p}_i2i_sampler", "sampler", "img2img Sampler"),
    ("{p}_t2i_scheduler", "scheduler", "txt2img Schedule type"),
    ("{p}_i2i_scheduler", "scheduler", "img2img Schedule type"),
    ("{p}_t2i_width", "int", "txt2img Width"),
    ("{p}_i2i_width", "int", "img2img Width"),
    ("{p}_t2i_height", "int", "txt2img Height"),
    ("{p}_i2i_height", "int", "img2img Height"),
    ("{p}_t2i_cfg", "float", "txt2img CFG Scale"),
    ("{p}_t2i_hr_cfg", "float", "txt2img Hires CFG Scale"),
    ("{p}_i2i_cfg", "float", "img2img CFG Scale"),
    ("{p}_t2i_dcfg", "float", "txt2img Distilled CFG / Shift"),
    ("{p}_t2i_hr_dcfg", "float", "txt2img Hires Distilled CFG / Shift"),
    ("{p}_i2i_dcfg", "float", "img2img Distilled CFG / Shift"),
    ("{p}_t2i_batch_size", "int", "txt2img Batch size"),
    ("{p}_i2i_batch_size", "int", "img2img Batch size"),
)

# FIELDS[3:] 의 UI 컴포넌트 — forge_main_entry 가 get_a1111_ui_component(tab, label) 로 찾는 그대로, 같은 순서.
# (앞의 셋은 main_entry.ui_checkpoint / ui_vae / ui_forge_unet_dtype)
UI_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("txt2img", "Steps"), ("txt2img", "Hires steps"), ("img2img", "Steps"),
    ("txt2img", "sampler_name"), ("img2img", "sampler_name"),
    ("txt2img", "scheduler"), ("img2img", "scheduler"),
    ("txt2img", "Size-1"), ("img2img", "Size-1"), ("txt2img", "Size-2"), ("img2img", "Size-2"),
    ("txt2img", "CFG scale"), ("txt2img", "Hires CFG Scale"), ("img2img", "CFG scale"),
    ("txt2img", "Distilled CFG Scale"), ("txt2img", "Hires Distilled CFG Scale"), ("img2img", "Distilled CFG Scale"),
    ("txt2img", "Batch size"), ("img2img", "Batch size"),
)

# on_preset_change 가 ``gr.update(value=v) if (v := getattr(opts, key, …)) > 0 else gr.skip()`` 로 읽는 칸 — 값이 0 이하면
# 화면 값을 덮어쓰지 않는다(tests/test_save_defaults.py 가 Forge 의 실제 함수에서 이 10개를 다시 뽑아 비교한다).
SKIP_IF_NOT_POSITIVE: frozenset[str] = frozenset((
    "{p}_t2i_step", "{p}_t2i_hr_step", "{p}_i2i_step",
    "{p}_t2i_width", "{p}_i2i_width", "{p}_t2i_height", "{p}_i2i_height",
    "{p}_t2i_cfg", "{p}_t2i_hr_cfg", "{p}_i2i_cfg",
))

REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MISSING = object()


class SaveDefaultsError(Exception):
    """사용자에게 그대로 보여 줄 한국어 메시지."""


@dataclass
class Host:
    """Forge 쪽 사실들 — 실제 값은 sam3ext/ui_save_defaults.py forge_host(), 테스트는 가짜."""

    opts: Any                                   # shared.opts (data, data_labels, set, save)
    config_filename: str                        # shared.config_filename
    presets: Sequence[str]                      # PresetArch.choices()
    checkpoint_known: Callable[[str], bool]     # sd_models.get_closet_checkpoint_match(name) is not None
    module_paths: Mapping[str, str]             # main_entry.module_list: 파일 이름 → 전체 경로
    dtypes: Sequence[str]                       # main_entry.forge_unet_storage_dtype_options 의 키
    samplers: Sequence[str]                     # [x.name for x in shared_items.list_samplers()]
    schedulers: Sequence[str]                   # shared_items.list_schedulers() (label)
    frozen: bool = False                        # cmd_opts.freeze_settings


@dataclass
class Change:
    key: str
    label: str
    old: Any
    new: Any

    @property
    def changed(self) -> bool:
        return self.old != self.new


@dataclass
class Kept:
    """프리셋 값이 Forge 가 건너뛰는 값(0 = 덮어쓰지 않음)이라 그대로 둔 칸."""

    key: str
    label: str
    value: Any


@dataclass
class Plan:
    preset: str
    changes: list[Change] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)   # Forge 에 없는 키
    kept: list[Kept] = field(default_factory=list)     # SKIP_IF_NOT_POSITIVE 중 저장된 값을 Forge 가 건너뛰는 칸


def forge_skips(stored: Any) -> bool:
    """on_preset_change 의 ``… if v > 0 else gr.skip()`` 그대로: ``v > 0`` 이 거짓(0 · 음수 · NaN)이면 Forge 는 그 칸의
    화면 값을 덮어쓰지 않는다. 0 과 비교할 수 없는 값(글자 · None · 목록 — Forge 의 그 줄은 TypeError)은 건너뛰는 값이 아니다:
    예전처럼 화면 값으로 바꿔 써서 고친다."""
    try:
        return not (stored > 0)
    except TypeError:
        return False


def _number(label: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SaveDefaultsError(f"{label}: 숫자가 아닙니다 ({value!r})")
    try:
        number = float(value)
    except OverflowError:
        raise SaveDefaultsError(f"{label}: 너무 큰 값입니다") from None
    if not math.isfinite(number):
        raise SaveDefaultsError(f"{label}: 유한한 숫자가 아닙니다 ({value!r})")
    return number


def _slider_range(info: Any) -> tuple[float, float] | None:
    args = getattr(info, "component_args", None)
    if isinstance(args, dict) and "minimum" in args and "maximum" in args:
        return float(args["minimum"]), float(args["maximum"])
    return None


def _in_range(label: str, value: float, info: Any) -> None:
    bounds = _slider_range(info)
    if bounds is not None and not bounds[0] <= value <= bounds[1]:
        lo, hi = (int(b) if float(b).is_integer() else b for b in bounds)
        shown = int(value) if float(value).is_integer() else value
        raise SaveDefaultsError(f"{label}: {shown} 은(는) Forge 설정 범위 {lo}~{hi} 밖입니다")


def _restricted(info: Any) -> bool:
    """Options.set 이 조용히 받지 않는 설정(do_not_save, visible=False 로 막힌 설정)."""
    if getattr(info, "do_not_save", False):
        return True
    args = getattr(info, "component_args", None)
    return isinstance(args, dict) and args.get("visible", True) is False


def coerce(kind: str, label: str, value: Any, info: Any, host: Host) -> Any:
    """UI 값 하나를 그 설정이 저장하는 모양으로 — 맞지 않으면 SaveDefaultsError."""
    if kind == "int":
        number = _number(label, value)
        if not number.is_integer():
            raise SaveDefaultsError(f"{label}: 정수가 아닙니다 ({value!r})")
        _in_range(label, number, info)
        return int(number)
    if kind == "float":
        number = _number(label, value)
        _in_range(label, number, info)
        return number
    if kind == "modules":
        names = [] if value is None else value      # 아무것도 고르지 않은 multiselect
        if not isinstance(names, (list, tuple)) or not all(isinstance(name, str) for name in names):
            raise SaveDefaultsError(f"{label}: 파일 이름 목록이 아닙니다 ({value!r})")
        paths = []
        for name in names:
            base = name.replace("\\", "/").rsplit("/", 1)[-1]   # modules_change 의 os.path.basename
            if base not in host.module_paths:
                raise SaveDefaultsError(
                    f"{label}: '{base}' 을(를) Forge 모듈 목록에서 찾지 못했습니다 (모델 목록을 새로 고친 뒤 다시)"
                )
            paths.append(host.module_paths[base])
        return sorted(paths)
    if value is None or value == "":
        raise SaveDefaultsError(f"{label}: 값이 비어 있습니다")
    if not isinstance(value, str):
        raise SaveDefaultsError(f"{label}: 글자가 아닙니다 ({value!r})")
    if kind == "checkpoint":
        if not host.checkpoint_known(value):
            raise SaveDefaultsError(f"{label}: '{value}' 을(를) Forge 체크포인트 목록에서 찾지 못했습니다")
        return value
    choices = {"dtype": host.dtypes, "sampler": host.samplers, "scheduler": host.schedulers}.get(kind)
    if choices is None:
        raise ValueError(f"unknown field kind {kind!r}")
    if value not in choices:
        raise SaveDefaultsError(f"{label}: '{value}' 은(는) Forge 의 선택지에 없습니다")
    return value


def plan(preset: Any, values: Sequence[Any], host: Host) -> Plan:
    """검사만 한다(아무것도 쓰지 않음). 문제가 있으면 첫 메시지(와 나머지 개수)로 SaveDefaultsError."""
    if not isinstance(preset, str) or preset not in host.presets:
        raise SaveDefaultsError(f"UI Preset '{preset}' 을(를) Forge 가 모릅니다")
    if len(values) != len(FIELDS):
        raise SaveDefaultsError(f"값 {len(values)}개를 받았습니다 — {len(FIELDS)}개여야 합니다 (Forge 버전을 확인하세요)")
    result = Plan(preset)
    errors: list[str] = []
    labels = getattr(host.opts, "data_labels", {})
    data = getattr(host.opts, "data", {})
    for (template, kind, label), value in zip(FIELDS, values):
        key = template.format(p=preset)
        info = labels.get(key)
        if info is None or _restricted(info):
            result.skipped.append(key)
            continue
        old = data.get(key, getattr(info, "default", None))
        if template in SKIP_IF_NOT_POSITIVE and forge_skips(old):
            # 0 = "화면 값을 덮어쓰지 않음" 그대로 — 화면 값은 같은 클릭의 Defaults Apply 가 ui-config 에 저장한다
            result.kept.append(Kept(key, label, old))
            continue
        try:
            new = coerce(kind, label, value, info, host)
        except SaveDefaultsError as error:
            errors.append(str(error))
            continue
        result.changes.append(Change(key, label, old, new))
    if errors:
        more = f" (외 {len(errors) - 1}개)" if len(errors) > 1 else ""
        raise SaveDefaultsError(errors[0] + more)
    return result


def apply(result: Plan, host: Host) -> None:
    """설정에 쓰고 config 파일을 저장한다. 실패하면 메모리의 값을 되돌리고 SaveDefaultsError."""
    if host.frozen:
        raise SaveDefaultsError("Forge 가 --freeze-settings 로 설정을 잠가 두어 프리셋 값을 쓰지 못합니다")
    opts = host.opts
    before = {change.key: opts.data.get(change.key, _MISSING) for change in result.changes}
    try:
        for change in result.changes:
            value = list(change.new) if isinstance(change.new, list) else change.new
            opts.set(change.key, value)
            if opts.data.get(change.key, _MISSING) != change.new:
                raise SaveDefaultsError(f"{change.label}: Forge 가 값을 받지 않았습니다 ({change.key})")
        opts.save(host.config_filename)
    except Exception as error:
        for key, value in before.items():
            if value is _MISSING:
                opts.data.pop(key, None)
            else:
                opts.data[key] = value
        if isinstance(error, SaveDefaultsError):
            raise
        raise SaveDefaultsError(f"설정 파일에 쓰지 못했습니다: {type(error).__name__}: {error}") from error


def clean_request_id(request_id: Any) -> str:
    text = request_id if isinstance(request_id, str) else ""
    return text if REQUEST_ID_RE.match(text) else ""


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)                       # 예전 설정 파일에 들어 있던 값일 수 있다(JSON 에 NaN 금지)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def save_preset(request_id: Any, preset: Any, values: Sequence[Any], host: Host) -> dict:
    """버튼 한 번의 프리셋 쪽 결과(JSON 으로 돌려줄 dict). 검사·쓰기 실패는 ok=False 와 메시지 — 던지지 않는다."""
    request = clean_request_id(request_id)
    try:
        result = plan(preset, values, host)
        apply(result, host)
    except SaveDefaultsError as error:
        return {"ok": False, "request": request, "preset": preset if isinstance(preset, str) else None,
                "error": str(error)}
    changed = [change for change in result.changes if change.changed]
    return {
        "ok": True,
        "request": request,
        "preset": result.preset,
        "written": len(result.changes),
        "changed": len(changed),
        "changes": [{"key": c.key, "label": c.label, "old": _jsonable(c.old), "new": _jsonable(c.new)} for c in changed],
        "skipped": list(result.skipped),
        "kept": [{"key": k.key, "label": k.label, "value": _jsonable(k.value)} for k in result.kept],
    }


def to_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)
