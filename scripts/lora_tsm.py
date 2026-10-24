"""TSM LoRA (시간 단계 전문가) — LyCORIS ``algo=tsm`` 파일과 sd-scripts 포크의 ``algo=tsm-lokr`` 파일(전문가가 LoKr +
DoRA)을 Anima 에서 학습과 같은 식으로 돌린다.

계산·파일·훅은 ``sam3ext/tsm_lora/``. 여기는 UI·infotext·XYZ 축·설치 연결만 한다. 파일 이름이 ``lora_tsm.py`` 인
것은 ``negpip.py`` 가 확장의 마지막 스크립트로 남아야 해서다(tests/test_negpip_script.py). ADetailer 는 자기 패스에 넣을
스크립트를 파일 이름(stem)으로 고르므로, ADetailer 패스에도 TSM 을 걸려면 그 설정의 스크립트 이름에 ``lora_tsm`` 을 더한다.

두 가지 방법으로 건다(둘 다 써도 된다 — 같은 파일이면 프롬프트 쪽만 쓴다):

* 프롬프트의 ``<lora:파일:강도>`` — Forge 로더가 읽은 TSM 키를 가로채 샘플링 중에 적용한다. 아코디언을 꺼 두어도
  적용된다(그대로 두면 Forge 가 'LoRA mismatch' 로 건너뛰는 파일이다).
* 아코디언의 파일 칸 — 프롬프트에 적지 않고 하나 더 건다.

API(``alwayson_scripts["TSM LoRA"]``)는 위치 인자 ``[enabled, lora, strength, stage, pooling, lokr_merge_cache]`` 도,
키 이름을 적은 dict 하나도 받는다: ``{"args": [{"enabled": true, "lora": "smoke_tsm_stage1", "strength": 0.8,
"stage": "auto", "pooling": "sample"}]}``. stage 는 ``auto`` / ``1`` / ``2`` / ``off``, pooling 은 ``sample`` / ``batch``
또는 UI 라벨, ``lokr_merge_cache`` 는 TSM-LoKr 의 시간 구간별 병합 가중치 캐시(기본 켬 — 사용자 10-24). 뒤의 인자는
빼도 된다(기본값).
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any

import gradio as gr

from modules import script_callbacks, scripts

from sam3ext import anima_lora_blocks as alb
from sam3ext.tsm_lora import blocks as tb
from sam3ext.tsm_lora import delta as td
from sam3ext.tsm_lora import fileformat as tf
from sam3ext.tsm_lora import listing as tl
from sam3ext.tsm_lora import runtime as tr

try:
    from modules.ui_components import InputAccordion
except Exception:  # pragma: no cover - 옛 Forge 는 InputAccordion 이 없다
    InputAccordion = None


TSM_NAME = "TSM LoRA"

NONE_CHOICE = "없음"

LABEL_STAGE_AUTO = "자동 (파일 메타데이터)"
LABEL_STAGE_1 = "1단계 식 (전문가 합)"
LABEL_STAGE_2 = "2단계 식 (라우터 게이트)"
LABEL_STAGE_OFF = "끔 (TSM 빼고 · 비교용)"
STAGE_CHOICES = [LABEL_STAGE_AUTO, LABEL_STAGE_1, LABEL_STAGE_2, LABEL_STAGE_OFF]
_LABEL_BY_STAGE = {
    tr.STAGE_AUTO: LABEL_STAGE_AUTO,
    tr.STAGE_1: LABEL_STAGE_1,
    tr.STAGE_2: LABEL_STAGE_2,
    tr.STAGE_OFF: LABEL_STAGE_OFF,
}

LABEL_POOL_SAMPLE = "샘플별 (학습과 같음 · 기본)"
LABEL_POOL_BATCH = "배치 전체 (LyCORIS 원본 forward)"
POOL_CHOICES = [LABEL_POOL_SAMPLE, LABEL_POOL_BATCH]
_LABEL_BY_POOL = {td.POOL_SAMPLE: LABEL_POOL_SAMPLE, td.POOL_BATCH: LABEL_POOL_BATCH}

ARG_NAMES = ("enabled", "lora", "strength", "stage", "pooling", "lokr_merge_cache")  # 뒤에만 덧붙인다(API 인자 위치 고정)
DEFAULT_STRENGTH = 1.0
DEFAULT_LOKR_MERGE_CACHE = True  # TSM-LoKr 병합 캐시 기본 켬(사용자 10-24: VRAM 약 5 GB 더 쓰고 가장 빠름)
STRENGTH_RANGE = (-2.0, 3.0)

INFOTEXT_LORA_KEY = "TSM LoRA"         # 아코디언 파일: "이름:강도" (프롬프트 쪽은 프롬프트에 이미 있다)
INFOTEXT_INFO_KEY = "TSM LoRA info"    # 적용한 모든 TSM: "이름 s1 8+1 r8 x0.8; …" (읽기용)
INFOTEXT_STAGE_KEY = "TSM stage"       # 단계 덮어쓰기(자동이 아닐 때만)
INFOTEXT_POOL_KEY = "TSM pooling"      # 라우터 입력 묶음(샘플별이 아닐 때만)
INFOTEXT_LOKR_CACHE_KEY = "TSM LoKr merge cache"   # TSM-LoKr 병합 캐시가 기본값(켬)과 다를 때만("off")
_INFOTEXT_KEYS = (INFOTEXT_LORA_KEY, INFOTEXT_INFO_KEY, INFOTEXT_STAGE_KEY, INFOTEXT_POOL_KEY, INFOTEXT_LOKR_CACHE_KEY)

# XYZ 격자 범례는 Forge 기본 글꼴(Roboto)로 그려져 한글이 네모로 나온다 — 선택지는 영어로.
XYZ_STAGE_LABEL = "[TSM] Stage"
XYZ_STAGE_CHOICES = list(tr.STAGE_CHOICES)
XYZ_POOL_LABEL = "[TSM] Router pooling"
XYZ_POOL_CHOICES = list(td.POOLINGS)
XYZ_STRENGTH_LABEL = "[TSM] Strength"  # 아코디언 파일의 강도(프롬프트 쪽은 Prompt S/R 로)
_XYZ_STAGE_ATTR = "_sam3_tsm_stage_xyz"
_XYZ_POOL_ATTR = "_sam3_tsm_pool_xyz"
_XYZ_STRENGTH_ATTR = "_sam3_tsm_strength_xyz"
# 가중치를 다시 합치지 않는다(샘플링 중 훅만 바뀐다) — 값싼 축.
XYZ_AXIS_COST = 0.2

_STATE_ATTR = "_sam3_tsm_state"
_LAST: dict[str, Any] = {"state": None}
_LISTING: dict[str, str] = {}


def _log(message: str) -> None:
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


tr.set_logger(_log)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _clamp_strength(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return round(min(STRENGTH_RANGE[1], max(STRENGTH_RANGE[0], number)), 4)


def _lora_name(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text in ("", NONE_CHOICE, "None", "none") else text


def coerce_args(args: tuple | list) -> dict[str, Any]:
    """스크립트 인자 → {enabled, lora, strength, stage, pooling, lokr_merge_cache}. 모르거나 빠진 값은 UI 기본값."""
    if args and isinstance(args[0], dict):
        raw = args[0]
    else:
        raw = {name: value for name, value in zip(ARG_NAMES, args)}
    strength = _clamp_strength(raw.get("strength", DEFAULT_STRENGTH))
    return {
        "enabled": _truthy(raw.get("enabled", False)),
        "lora": _lora_name(raw.get("lora")),
        "strength": DEFAULT_STRENGTH if strength is None else strength,
        "stage": tr.normalize_stage(raw.get("stage")) or tr.STAGE_AUTO,
        "pooling": tr.normalize_pooling(raw.get("pooling")) or td.POOL_SAMPLE,
        "lokr_merge_cache": _truthy(raw.get("lokr_merge_cache", DEFAULT_LOKR_MERGE_CACHE)),
    }


def resolve_state(p, args: tuple | list) -> dict[str, Any]:
    """이번 생성의 상태. 단계·묶음 덮어쓰기는 아코디언이 켜졌을 때만, XYZ 값은 늘 아코디언보다 먼저다."""
    state = coerce_args(args)
    if not state["enabled"]:
        state["stage"], state["pooling"] = tr.STAGE_AUTO, td.POOL_SAMPLE
        state["lokr_merge_cache"] = DEFAULT_LOKR_MERGE_CACHE  # 프롬프트 <lora:...> 만 쓸 때도 기본값
    state["stage"] = tr.normalize_stage(getattr(p, _XYZ_STAGE_ATTR, None)) or state["stage"]
    state["pooling"] = tr.normalize_pooling(getattr(p, _XYZ_POOL_ATTR, None)) or state["pooling"]
    xyz_strength = _clamp_strength(getattr(p, _XYZ_STRENGTH_ATTR, None))
    if xyz_strength is not None:
        state["strength"] = xyz_strength
    return state


# ── 파일 목록 ──

def _lora_dirs() -> list[str]:
    try:
        from modules import shared

        opts = getattr(shared, "cmd_opts", None)
        dirs = [getattr(opts, "lora_dir", None), *(getattr(opts, "lora_dirs", None) or [])]
    except Exception:
        return []
    return [d for d in dirs if d]


def _network_rows(refresh: bool = False) -> list[tuple[str, str, Any]] | None:
    module = alb._find_forge_lora_module()
    if module is None:
        return None
    if refresh:
        try:
            module.list_available_networks()
        except Exception:
            _log("LoRA 목록을 다시 읽지 못했습니다:\n" + traceback.format_exc())
    networks = getattr(module, "available_networks", None)
    if not isinstance(networks, dict) or not networks:
        return None
    rows = []
    for name, net in networks.items():
        filename = getattr(net, "filename", None)
        if filename:
            rows.append((str(name), str(filename), getattr(net, "metadata", None)))
    return rows


def list_tsm_loras(refresh: bool = False, scan_dirs: bool = False) -> dict[str, str]:
    """{Forge LoRA 이름: 경로}. Forge 목록(메타데이터 캐시)을 쓰고, 없으면 ``scan_dirs`` 일 때 LoRA 폴더를 돈다."""
    rows = _network_rows(refresh)
    if rows is None and scan_dirs:
        rows = tl.walk_safetensors(_lora_dirs())
    if rows is None:
        return dict(_LISTING)
    found = tl.find_tsm_files(rows)
    _LISTING.clear()
    _LISTING.update(found)
    return found


def _resolve_path(name: str) -> str | None:
    if not name:
        return None
    if name in _LISTING:
        return _LISTING[name]
    if os.path.isfile(name):
        return name
    found = list_tsm_loras(refresh=False, scan_dirs=True)
    if name in found:
        return found[name]
    module = alb._find_forge_lora_module()
    for table_name in ("available_networks", "available_network_aliases"):
        table = getattr(module, table_name, None) if module is not None else None
        net = table.get(name) if isinstance(table, dict) else None
        if net is not None and getattr(net, "filename", None):
            return str(net.filename)
    return None


def _choices() -> list[str]:
    return [NONE_CHOICE, *_LISTING.keys()]


# ── infotext ──

def _paste_enabled(params: dict) -> bool:
    # 켜져 있을 때만 기록하는 키들. 'TSM LoRA info' 는 프롬프트 쪽만 쓴 이미지에도 남으므로 보지 않는다.
    return any(key in params for key in (INFOTEXT_LORA_KEY, INFOTEXT_STAGE_KEY, INFOTEXT_POOL_KEY, INFOTEXT_LOKR_CACHE_KEY))


def _split_lora_value(value: Any) -> tuple[str, float | None]:
    text = str(value or "").strip()
    name, sep, weight = text.rpartition(":")
    if sep and _clamp_strength(weight) is not None:
        return name.strip(), _clamp_strength(weight)
    return text, None


def _paste_lora(params: dict) -> str | None:
    if not _paste_enabled(params):
        return None
    if INFOTEXT_LORA_KEY not in params:
        return NONE_CHOICE
    name, _weight = _split_lora_value(params.get(INFOTEXT_LORA_KEY))
    return name or NONE_CHOICE


def _paste_strength(params: dict) -> float | None:
    if INFOTEXT_LORA_KEY not in params:
        return None
    return _split_lora_value(params.get(INFOTEXT_LORA_KEY))[1]


def _paste_stage(params: dict) -> str | None:
    if not _paste_enabled(params):
        return None
    stage = tr.normalize_stage(params.get(INFOTEXT_STAGE_KEY)) or tr.STAGE_AUTO
    return _LABEL_BY_STAGE[stage]


def _paste_pooling(params: dict) -> str | None:
    if not _paste_enabled(params):
        return None
    pooling = tr.normalize_pooling(params.get(INFOTEXT_POOL_KEY)) or td.POOL_SAMPLE
    return _LABEL_BY_POOL[pooling]


def _paste_lokr_cache(params: dict) -> bool | None:
    if not _paste_enabled(params):
        return None
    value = params.get(INFOTEXT_LOKR_CACHE_KEY)
    return DEFAULT_LOKR_MERGE_CACHE if value is None else _truthy(value)


def _clear_infotext(p) -> None:
    params = getattr(p, "extra_generation_params", None)
    if isinstance(params, dict):
        # XYZ 칸들은 이 dict 하나를 같이 쓴다(xyz_grid.py) — 앞 칸의 키를 지운다.
        for key in _INFOTEXT_KEYS:
            params.pop(key, None)


_APPLIED_ATTR = "_sam3_tsm_applied"


def _record_infotext(p, state: dict[str, Any], entries: list[tr.TsmEntry]) -> None:
    """적용한 TSM 을 infotext 에. 하이레스 패스가 다른 LoRA 를 걸어도 앞 패스의 기록은 남긴다(process 에서만 비운다)."""
    params = getattr(p, "extra_generation_params", None)
    if not isinstance(params, dict):
        return
    applied = getattr(p, _APPLIED_ATTR, None)
    if not isinstance(applied, dict):
        applied = {}
    for e in entries:
        text = f"{e.label} s{e.stage} {e.tsm.scales_label()} {e.tsm.kind_label()} x{e.multiplier:g}"
        applied.setdefault(text, e.source)
    try:
        setattr(p, _APPLIED_ATTR, applied)
    except Exception:
        pass
    picker = [e for e in entries if e.source == tr.SOURCE_PICKER]
    if picker:
        params[INFOTEXT_LORA_KEY] = f"{picker[0].label}:{picker[0].multiplier:g}"
    if applied:
        params[INFOTEXT_INFO_KEY] = "; ".join(applied)
    if state["stage"] != tr.STAGE_AUTO:
        params[INFOTEXT_STAGE_KEY] = state["stage"]
    if state["pooling"] != td.POOL_SAMPLE:
        params[INFOTEXT_POOL_KEY] = state["pooling"]
    # 기본값(켬)과 다를 때만 남긴다 — 껐을 때 "off"
    if bool(state.get("lokr_merge_cache")) != DEFAULT_LOKR_MERGE_CACHE and any(
            getattr(e.tsm, "expert_kind", "lora") == "lokr" for e in entries):
        params[INFOTEXT_LOKR_CACHE_KEY] = "on" if state.get("lokr_merge_cache") else "off"


# ── 설치 ──

def _install_loader() -> None:
    try:
        if tr.install_loader_hook(notify=alb._notify_ui):
            print("[sam-extra] TSM LoRA loader hook installed (<lora:...> TSM files are applied by sam-extra).")
    except Exception:
        print("[-] sam-extra: failed to install the TSM LoRA loader hook:\n" + traceback.format_exc(), file=sys.stderr)


def _on_unloaded() -> None:
    try:
        tr.uninstall_loader_hook()
    except Exception:
        pass
    try:
        tr.finish_generation()
    except Exception:
        pass


# networks(sd_forge_lora)는 내장 확장이라 보통 먼저 읽힌다. on_before_ui 는 멱등 예비.
_install_loader()
script_callbacks.on_before_ui(_install_loader)

try:
    script_callbacks.on_script_unloaded(_on_unloaded)
except AttributeError:  # pragma: no cover - 옛 Forge
    pass


# ── XYZ ──

def _xyz_set_stage(p, x, xs) -> None:
    setattr(p, _XYZ_STAGE_ATTR, x)


def _xyz_set_pool(p, x, xs) -> None:
    setattr(p, _XYZ_POOL_ATTR, x)


def _xyz_set_strength(p, x, xs) -> None:
    setattr(p, _XYZ_STRENGTH_ATTR, x)


def _make_xyz_axis() -> None:
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break
    if xyz_grid is None:
        return
    existing = {str(a.label) for a in xyz_grid.axis_options}
    for label, setter, choices in (
        (XYZ_STAGE_LABEL, _xyz_set_stage, XYZ_STAGE_CHOICES),
        (XYZ_POOL_LABEL, _xyz_set_pool, XYZ_POOL_CHOICES),
    ):
        if label not in existing:
            xyz_grid.axis_options.append(
                xyz_grid.AxisOption(label, str, setter, cost=XYZ_AXIS_COST, choices=lambda c=choices: list(c))
            )
    if XYZ_STRENGTH_LABEL not in existing:
        xyz_grid.axis_options.append(xyz_grid.AxisOption(XYZ_STRENGTH_LABEL, float, _xyz_set_strength, cost=XYZ_AXIS_COST))


def _on_before_ui_xyz() -> None:
    try:
        _make_xyz_axis()
    except Exception:
        _log("xyz_grid axis registration failed:\n" + traceback.format_exc())


script_callbacks.on_before_ui(_on_before_ui_xyz)


# ── 적용 ──

def _lora_model(p):
    # 로더 훅은 shared.sd_model(= networks.load_networks 의 current_sd)에 기록한다.
    model = alb._forge_sd_model()
    return model if model is not None else getattr(p, "sd_model", None)


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _base(path: str) -> str:
    return os.path.splitext(os.path.basename(str(path)))[0]


def collect_entries(p, state: dict[str, Any]) -> list[tr.TsmEntry]:
    """프롬프트 쪽 기록 + 아코디언 파일 → 적용할 TSM 목록(단계는 아직 정하지 않는다)."""
    entries: list[tr.TsmEntry] = []
    for record in tr.records_for(_lora_model(p)):
        tsm = record.get("tsm")
        if tsm is None:
            continue
        entries.append(tr.TsmEntry(tsm, float(record["strength"]), tr.SOURCE_PROMPT, label=_base(record["filename"])))
    if not (state["enabled"] and state["lora"]):
        return entries
    path = _resolve_path(state["lora"])
    if path is None:
        _log(f"'{state['lora']}' 파일을 LoRA 폴더에서 찾지 못했습니다 — 아코디언 파일은 건너뜁니다.")
        return entries
    if any(_same_path(path, e.tsm.path) for e in entries if e.tsm.path):
        _log(f"'{state['lora']}' 은(는) 프롬프트의 <lora:…> 에도 있어 프롬프트 쪽 강도만 씁니다.")
        return entries
    try:
        tsm = tf.load_tsm_file(path)
    except tf.TsmFormatError as exc:
        message = f"'{state['lora']}' TSM 파일을 읽지 못해 건너뜁니다: {exc}"
        _log(message)
        alb._notify_ui(message)
        return entries
    entries.append(tr.TsmEntry(tsm, float(state["strength"]), tr.SOURCE_PICKER, label=state["lora"]))
    return entries


def _element_size(unet, dit) -> int:
    """장치 캐시는 Linear 입력 dtype(= 계산 dtype)으로 만든다 — fp8·GGUF 저장 dtype 이 아니라."""
    dtype = getattr(getattr(unet, "model", None), "computation_dtype", None)
    if dtype is None:
        dtype = getattr(tb.unwrap_dit(dit), "dtype", None)
    try:
        import torch

        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            return 2
        return max(2, torch.empty((), dtype=dtype).element_size())
    except Exception:
        return 2


def sampling_tokens(p, latent, dit) -> int:
    """이 패스의 DiT 한 호출이 Linear 에 넣는 행 수(배치 × 토큰)의 위 한계 — 임시 메모리 예약용.

    ``latent`` 는 ``process_before_every_sampling`` 의 ``x`` (1차 · 하이레스는 그 패스 크기의 latent, img2img 는 init_latent)
    ``[B, C, (T,) H, W]``. 토큰 = T · ⌈H/patch⌉ · ⌈W/patch⌉, cond · uncond 를 한 배치로 돌리는 경우를 따라 배치는 2 배."""
    patch = getattr(tb.unwrap_dit(dit), "patch_spatial", 2)
    try:
        patch = max(1, int(patch))
    except (TypeError, ValueError):
        patch = 2
    shape = tuple(getattr(latent, "shape", ()) or ())
    if len(shape) >= 4:
        batch, height, width = int(shape[0]), int(shape[-2]), int(shape[-1])
        frames = 1
        for size in shape[2:-2]:
            frames *= int(size)
    else:
        batch = int(getattr(p, "batch_size", 1) or 1)
        height = int(getattr(p, "height", 1024) or 1024) // 8
        width = int(getattr(p, "width", 1024) or 1024) // 8
        frames = 1
    tokens = max(1, frames) * -(-height // patch) * -(-width // patch)
    return 2 * max(1, batch) * tokens


def apply_tsm(p, state: dict[str, Any], latent=None) -> tr.TsmSession | None:
    """이 샘플링 패스의 TSM 세션을 만들어 UNet 복제본에 싣고 DiT 에 훅을 건다. 걸 것이 없으면 None."""
    entries = collect_entries(p, state)
    zero = [e for e in entries if e.multiplier == 0]
    if zero:
        # 강도 0 은 Δ 가 0 — 훅을 걸지 않아 꺼 둔 것과 똑같이 돈다.
        _log("강도 0 이라 건너뜁니다: " + ", ".join(e.label or e.tsm.name for e in zero))
        entries = [e for e in entries if e.multiplier != 0]
    if not entries:
        return None
    if state["stage"] == tr.STAGE_OFF:
        _record_infotext(p, state, [])
        _log(f"단계 계산 '끔' — TSM {len(entries)}개를 빼고 생성합니다(비교용).")
        return None
    for entry in entries:
        entry.stage = tr.resolve_stage(entry.tsm, state["stage"]) or entry.tsm.stage
        if entry.stage == 2 and entry.tsm.stage == 1 and all(m.router_is_zero() for m in entry.tsm.modules.values()):
            _log(f"'{entry.label}': 1단계 파일(라우터 0)을 2단계 식으로 돌립니다 — 문맥 전문가가 빠집니다.")
    sd_model = getattr(p, "sd_model", None)
    forge_objects = getattr(sd_model, "forge_objects", None)
    unet = getattr(forge_objects, "unet", None)
    dit = getattr(getattr(unet, "model", None), "diffusion_model", None)
    if unet is None or dit is None or not tb.is_anima_dit(dit):
        _log("지금 모델이 Anima 가 아니라 TSM 을 건너뜁니다.")
        return None
    multiplier = getattr(getattr(unet.model, "predictor", None), "multiplier", 1.0)
    try:
        multiplier = float(multiplier)
    except (TypeError, ValueError):
        multiplier = 1.0
    session = tr.TsmSession(entries, state["pooling"], multiplier, lokr_merge_cache=bool(state.get("lokr_merge_cache")))
    report = session.bind(dit, alb.current_duplicate_policy(), alb.current_weak_copy())
    for name, reason in report.refused:
        message = f"'{name}' 을(를) 이 모델에 걸 수 없어 건너뜁니다: {reason}"
        _log(message)
        alb._notify_ui(message)
    for note in report.notes:
        _log(note)
    if report.missing or report.shape_mismatch:
        _log(f"DiT 에 없는 모듈 {report.missing}개, 모양이 다른 모듈 {report.shape_mismatch}개는 건너뜁니다.")
    if not session.bound:
        return None
    if getattr(dit, "_orig_mod", None) is not None:
        _log("DiT 가 torch.compile 로 감싸여 있습니다 — 스텝마다 바뀌는 TSM 훅 때문에 다시 컴파일될 수 있습니다.")
    unet = unet.clone()
    if hasattr(unet, "set_transformer_option"):
        unet.set_transformer_option(tr.OPTION_KEY, session)
    else:  # pragma: no cover - 옛 Forge
        unet.model_options.setdefault("transformer_options", {})[tr.OPTION_KEY] = session
    element_size = _element_size(unet, dit)
    cache_bytes = session.fused_bytes(element_size)
    # Linear 호출 하나의 임시 텐서(중간값 · 제자리로 못 더할 때의 새 출력)도 예약한다 — lowvram 이 그만큼 남겨 두게.
    transient_bytes = session.transient_bytes(sampling_tokens(p, latent, dit), element_size)
    if hasattr(unet, "add_extra_preserved_memory_during_sampling"):
        unet.add_extra_preserved_memory_during_sampling(cache_bytes + transient_bytes)
    p.sd_model.forge_objects.unet = unet
    tr.HOOKS.attach(dit, session)
    # 훅을 건 뒤에는 DiT 모듈을 쥘 일이 없다 — 하이레스 체크포인트 전환이나 예외로 끝난 생성 뒤에 앞 DiT 를 붙잡지 않게.
    session.targets.clear()
    tr.remember_session(session)
    # 이 패스의 샘플링이 끝나면(VAE 디코드 전) 장치 캐시를 놓는다 — post_sample 은 하이레스 디코드 뒤라 늦다.
    release_hook = tr.wrap_sampler_release(getattr(p, "sampler", None))
    _record_infotext(p, state, report.entries)
    files = ", ".join(
        f"{e.label} s{e.stage} {e.tsm.kind_label()} x{e.multiplier:g} t:{e.tsm.timestep_dtype}" for e in report.entries
    )
    _log(
        f"적용: {files} — 모듈 {report.modules}개, 장치 캐시 약 {cache_bytes / 2**20:.0f} MB"
        f"(+ 호출당 임시 약 {transient_bytes / 2**20:.0f} MB 예약), 라우터 묶음 {state['pooling']}"
        + ("" if release_hook != "unavailable" else " — 샘플러를 감싸지 못해 캐시는 post_sample 에서 놓습니다")
    )
    return session


def _finish_and_report() -> None:
    sessions, _hooked = tr.finish_generation()
    for session in sessions:
        if session.calls and session.seen_t:
            ts = sorted(session.seen_t)
            _log(f"DiT 호출 {session.calls}번에 적용(t {ts[0]}…{ts[-1]}, 서로 다른 t {len(ts)}개)")
        elif session.bound and not session.calls:
            _log("세션을 걸었지만 DiT 호출에 쓰이지 않았습니다(다른 확장이 transformer_options 를 바꿨을 수 있음).")
        if session.skipped_calls:
            _log(f"Linear 호출 {session.skipped_calls}번은 TSM 없이 돌았습니다(배치·너비 불일치).")


class TsmLora(scripts.Script):
    def title(self):
        return TSM_NAME

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        label = "TSM LoRA (시간 단계 전문가 · 실험)"
        try:
            list_tsm_loras()
        except Exception:
            _log("TSM 파일 목록을 만들지 못했습니다:\n" + traceback.format_exc())
        if InputAccordion is not None:
            context = InputAccordion(False, label=label, elem_id=self.elem_id("tsm_lora"))
        else:  # pragma: no cover
            context = gr.Accordion(label, open=False)
        with context as enabled:
            if InputAccordion is None:  # pragma: no cover
                enabled = gr.Checkbox(label="Enable", value=False, elem_classes=["sam3-on"])
            gr.Markdown(
                "**TSM**(LyCORIS `algo=tsm`)은 시간 구간마다 다른 LoRA 전문가를 쓰는 LoRA 입니다. Forge 순정 로더는 이 "
                "파일의 키를 몰라 건너뛰므로, sam-extra 가 샘플링 스텝마다 σ 로 전문가를 골라 대상 Linear 출력에 "
                "더합니다(학습과 같은 식 · `t = round(σ·1000)`, bf16 으로 학습한 파일은 학습 때처럼 σ·1000 을 bf16 으로 반올림).\n\n"
                "- 프롬프트에 `<lora:파일이름:강도>` 로 적은 TSM 파일은 **이 아코디언을 꺼 두어도** 적용됩니다. 아래 파일 "
                "칸은 프롬프트에 적지 않고 하나 더 걸 때 씁니다(같은 파일이면 프롬프트 쪽만 씁니다).\n"
                "- **단계 계산** — 자동은 파일 메타데이터(`tsm_stage`)를 따릅니다. 1단계 파일은 코어 전문가와 문맥 "
                "전문가를 그대로 더하고, 2단계 파일은 라우터 게이트로 문맥 전문가를 섞습니다. 1단계 파일을 2단계 식으로 "
                "돌리면 라우터가 0 이라 문맥 전문가가 빠집니다(비교용). **끔**은 TSM 을 빼고 생성합니다.\n"
                "- **라우터 입력 묶음**(2단계만) — 샘플별은 cond·uncond 를 따로 평균해 학습(배치 1)과 같습니다. "
                "배치 전체는 LyCORIS forward 가 같은 t 의 샘플을 함께 평균하던 방식입니다.\n"
                "- **2단계 학습** — 같은 설정(`algo=tsm`, 같은 preset · dim · alpha · `tsm_n_scales`)에 `tsm_stage=2` 를 "
                "주고 1단계 파일을 `--network_weights` 로 넘깁니다(`--resume` 아님). 라우터와 시간 임베딩만 학습되고 "
                "전문가는 고정됩니다. 0 스텝의 2단계 파일은 게이트가 0 이라 코어 전문가만 쓰입니다.\n"
                "- Anima(Base · 2.9B · 3.8B) 전용입니다. 블록 수가 다른 모델에 얹으면 'DoRA 추론 방식'의 끼워 넣은 "
                "블록 정책을 따릅니다. 가중치에 합치지 않으므로 Forge 의 일반 LoRA 와 함께 써도 됩니다.\n"
                "- **TSM-LoKr**(sd-scripts 포크 `algo=tsm-lokr` — 전문가가 LoKr, DoRA 출력 축)도 같은 방법으로 겁니다. DoRA "
                "노름이 병합 가중치 전체에 걸리므로 Forge Linear 의 weight_function 으로 그 구간의 병합 가중치 "
                "`W + 강도·(DoRA(W + ΔW_t) − W)` 를 넘깁니다(학습과 같은 fp32 노름). 한 Linear 에는 TSM-LoKr 하나만 걸립니다.\n"
                "- **TSM-LoKr 병합 캐시** — 지금 시간 구간의 병합 가중치를 GPU 에 쥐고 구간이 바뀔 때만 다시 만듭니다. DiT "
                "호출마다의 병합(1024² 기준 약 0.05–0.09 초 · 추정)을 아끼는 대신 VRAM 을 대상 가중치만큼(2.9B 약 5.2 GB) 더 "
                "씁니다. 기본은 켬 — VRAM 이 빠듯하면 끄세요."
            )
            with gr.Row():
                lora = gr.Dropdown(
                    label="TSM 파일 (models/Lora · --lora-dirs)",
                    choices=_choices(),
                    value=NONE_CHOICE,
                    allow_custom_value=True,
                    elem_id=self.elem_id("tsm_lora_file"),
                )
                refresh = gr.Button("🔄", elem_id=self.elem_id("tsm_lora_refresh"), scale=0, min_width=40)
                strength = gr.Slider(
                    label="강도",
                    minimum=STRENGTH_RANGE[0],
                    maximum=STRENGTH_RANGE[1],
                    step=0.05,
                    value=DEFAULT_STRENGTH,
                    elem_id=self.elem_id("tsm_lora_strength"),
                )
            stage = gr.Radio(
                label="단계 계산",
                choices=STAGE_CHOICES,
                value=LABEL_STAGE_AUTO,
                elem_id=self.elem_id("tsm_lora_stage"),
            )
            pooling = gr.Radio(
                label="라우터 입력 묶음 (2단계만)",
                choices=POOL_CHOICES,
                value=LABEL_POOL_SAMPLE,
                elem_id=self.elem_id("tsm_lora_pooling"),
            )
            lokr_merge_cache = gr.Checkbox(
                label="TSM-LoKr 병합 캐시 (VRAM 더 씀 · 빠름)",
                value=DEFAULT_LOKR_MERGE_CACHE,
                elem_id=self.elem_id("tsm_lora_lokr_merge_cache"),
            )

        def _refresh(current):
            found = list_tsm_loras(refresh=True, scan_dirs=True)
            value = current if current in found or current == NONE_CHOICE else NONE_CHOICE
            _log(f"TSM 파일 {len(found)}개를 찾았습니다.")
            return gr.update(choices=[NONE_CHOICE, *found.keys()], value=value)

        refresh.click(fn=_refresh, inputs=[lora], outputs=[lora], show_progress=False)
        self.infotext_fields = [
            (enabled, _paste_enabled),
            (lora, _paste_lora),
            (strength, _paste_strength),
            (stage, _paste_stage),
            (pooling, _paste_pooling),
            (lokr_merge_cache, _paste_lokr_cache),
        ]
        return [enabled, lora, strength, stage, pooling, lokr_merge_cache]

    def process(self, p, *args):
        # extra_networks.activate(processing.py — <lora:…> 로드)보다 먼저 돈다.
        if getattr(p, "_sam3_inner", False):
            # SAM3 인페인트·Refine 의 내부 패스 — 바깥 생성의 상태를 그대로 쓴다(인자가 없거나 기본값이다).
            return
        tr.finish_generation()  # 예외로 끝난 이전 생성이 남긴 훅·캐시
        _install_loader()
        state = resolve_state(p, args)
        _clear_infotext(p)
        try:
            setattr(p, _STATE_ATTR, state)
            setattr(p, _APPLIED_ATTR, {})
        except Exception:
            pass
        _LAST["state"] = state

    def process_before_every_sampling(self, p, *args, **kwargs):
        # 1차·하이레스·img2img 패스마다, 그 패스의 LoRA 활성화 뒤에 돈다.
        state = getattr(p, _STATE_ATTR, None)
        if state is None:
            state = _LAST["state"] if getattr(p, "_sam3_inner", False) and _LAST["state"] else resolve_state(p, args)
        try:
            apply_tsm(p, state, kwargs.get("x"))
        except Exception as exc:
            _log(f"적용하지 못해 TSM 없이 생성합니다 — {type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stderr)

    def post_sample(self, p, ps, *args):
        # 예비 — 보통은 apply_tsm 이 감싼 샘플러의 launch_sampling 이 패스마다 이미 놓았다(하이레스 디코드는 이보다 먼저
        # 돈다). 샘플러를 감싸지 못한 경우(post_sample 다음의 마지막 디코드 전) 여기서 놓는다. 훅·통계는 postprocess 까지.
        try:
            tr.release_device_caches()
        except Exception:
            pass

    def postprocess(self, p, processed, *args):
        _finish_and_report()
