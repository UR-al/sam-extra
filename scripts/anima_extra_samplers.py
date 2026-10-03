"""Extra Samplers — registers the samplers of ``sam3ext/extra_samplers`` and holds the ER SDE values.

At import (before Forge builds its sampler dropdowns) ``registry.register()`` adds five entries to
Forge's sampler list: ``ER SDE (Reverse-time)``, ``ER SDE (ODE)``, ``DPM++ 4M SDE``,
``Euler Dy CFG++``, ``Euler SMEA Dy CFG++`` (code, origins and licences in
``sam3ext/extra_samplers/__init__.py``). They are picked in Forge's own Sampler / Hires sampler
dropdowns, the XYZ ``Sampler`` axis and the API ``sampler_name``.

This always-on script adds the collapsed ``Extra Samplers`` accordion with the two values only the
ER SDE pair reads:

* ``ER SDE max stage`` (1-3, default 3) — infotext ``ER SDE max stage`` when not 3;
* ``ER SDE eta`` (0-10, default 1) — Reverse-time only, infotext ``ER SDE eta`` when not 1 (or when
  the request's Forge ``Eta`` is not 1, see ``params.record_infotext``).

They are stored on the request in ``process`` (so both passes of a hires request use them) and the
sampler writes the infotext when it runs. Pasting an infotext restores them (missing keys → the
defaults; an ``ER SDE (Reverse-time)`` infotext with Forge's ``Eta`` but no ``ER SDE eta`` — written
by aoleg/Neo_ExtraSchedulers — gives η from ``Eta``). XYZ axes: ``[Extra Samplers] ER SDE max stage``
and ``[Extra Samplers] ER SDE eta``. API: ``alwayson_scripts["Extra Samplers"]["args"] =
[max_stage, eta]`` (both optional).

Everything else comes from Forge: ``Eta`` (DPM++ 4M SDE) and ``Sigma churn / tmin / tmax / noise``
(Settings → Sampler Parameters, shown with ``--adv-samplers``; also infotext/override settings)
reach the samplers through Forge's ``Sampler.initialize``, which writes their infotext.

File name: not ``extra_samplers.py`` — Panchovix/sd_forge_neo_extra_samplers ships a script of that
name, and Forge keys the module name, ui-config (``customscript/<file>/…``), the startup timer and
error reports by it, as do ADetailer's script filter (stem) and ``layout_lanes`` (lane key
``anima-extra-samplers``). It sorts before ``negpip.py``, which must stay this extension's last script.
"""

from __future__ import annotations

import sys
import traceback
from functools import partial

import gradio as gr

from modules import script_callbacks, scripts

from sam3ext import layout_lanes
from sam3ext.extra_samplers import params as sampler_params
from sam3ext.extra_samplers import registry

TITLE = "Extra Samplers"
XYZ_PREFIX = "[Extra Samplers]"
XYZ_MAX_STAGE = f"{XYZ_PREFIX} ER SDE max stage"
XYZ_ETA = f"{XYZ_PREFIX} ER SDE eta"


def _log(msg: str) -> None:
    print(f"[ExtraSamplers] {msg}", file=sys.stderr)


# Register now: Forge loads the scripts before it builds the UI (and the dropdown choices).
try:
    registry.register(log=_log)
except Exception:  # never break the UI over a sampler table
    _log("registration failed:\n" + traceback.format_exc())


# ---------------------------------------------------------------------------
# XYZ plot
# ---------------------------------------------------------------------------


def _xyz_set(p, x, xs, *, field: str):
    store = getattr(p, sampler_params.XYZ_ATTR, None)
    if not isinstance(store, dict):
        store = {}
        setattr(p, sampler_params.XYZ_ATTR, store)
    store[field] = x


def _make_xyz_axes() -> None:
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break
    if xyz_grid is None:
        return
    if any(str(axis.label).startswith(XYZ_PREFIX) for axis in xyz_grid.axis_options):
        return
    stages = [str(stage) for stage in range(sampler_params.MAX_STAGE_MIN, sampler_params.MAX_STAGE_MAX + 1)]
    xyz_grid.axis_options.extend([
        xyz_grid.AxisOption(
            XYZ_MAX_STAGE, int, partial(_xyz_set, field="max_stage"), choices=lambda: list(stages),
        ),
        xyz_grid.AxisOption(XYZ_ETA, float, partial(_xyz_set, field="eta")),
    ])


def _on_before_ui() -> None:
    try:
        _make_xyz_axes()
    except Exception:
        _log("xyz_grid axis registration failed:\n" + traceback.format_exc())


script_callbacks.on_before_ui(_on_before_ui)


# ---------------------------------------------------------------------------
# UI text
# ---------------------------------------------------------------------------

_HELP = (
    "Forge 샘플러 목록에 **ER SDE (Reverse-time)** · **ER SDE (ODE)** · **DPM++ 4M SDE** · "
    "**Euler Dy CFG++** · **Euler SMEA Dy CFG++** 를 더합니다 — 고르는 곳은 Forge 의 Sampler 드롭다운입니다.\n\n"
    "- 아래 두 값은 **ER SDE 두 항목에만** 쓰입니다(Forge 내장 **ER SDE** 는 그대로). **max stage** 는 풀이 차수"
    "(1 = 지수 Euler, 2·3 = 다단계 보정, 기본 3), **eta** 는 Reverse-time 의 잡음 척도 h(λ)=λ^(η+1) 의 지수"
    "(기본 1 = 고전 역시간 SDE, 0 = ODE 와 같음, 클수록 스텝마다 잡음↑ · 너무 크면 결과가 깨질 수 있음). "
    "ODE 는 잡음을 넣지 않습니다.\n"
    "- Forge 의 Settings → Sampler Parameters(`--adv-samplers` 로 켜야 보임) 값도 그대로 쓰입니다: **sigma noise** 는 "
    "Reverse-time·4M SDE·Dy 의 churn 잡음에, **Eta for k-diffusion samplers** 는 DPM++ 4M SDE 에, "
    "**sigma churn/tmin/tmax** 는 Dy 두 항목에(기본 churn 0 = 재잡음 없음). 바꾼 값은 Forge 가 infotext 에 남깁니다.\n"
    "- **CFG++** 두 항목은 CFG 1~2 를 권장합니다. Dy 는 2·3번째 스텝에, SMEA Dy 는 0번째(×1.25 해상도 — VRAM 더 씀)·"
    "1번째 스텝에 모델을 한 번 더 부릅니다. Anima 같은 flow 모델에서도 돕니다. Forge 의 **Spectrum Integrated**, "
    "Wan I2V·PiD 처럼 전체 해상도 입력을 쓰는 경우에는 이 보조 스텝 없이 일반 CFG++ 스텝으로 돌고, infotext "
    "`Extra Samplers status` 에 이유를 남깁니다."
)


def _status_markdown() -> str:
    report = registry.last_report()
    if report is None:
        return ""
    lines = []
    if report.error:
        lines.append(f"⚠ 샘플러를 등록하지 못했습니다: `{report.error}`")
    if report.skipped_foreign:
        names = ", ".join(f"`{label}`" for label in report.skipped_foreign)
        lines.append(f"⚠ 다른 확장이 먼저 등록한 이름이라 건너뜀: {names} — 이 이름은 그 확장의 샘플러가 돌고, "
                     "위 두 값은 이 확장의 ER SDE 에만 쓰입니다.")
    if report.skipped_missing:
        names = ", ".join(f"`{label}`" for label in report.skipped_missing)
        lines.append(f"⚠ 이 Forge 의 k-diffusion 에 필요한 함수가 없어 건너뜀: {names}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------


class ExtraSamplers(scripts.Script):
    # Directly under Anima 3.8B (-35) in the ANIMA lane; the values are only read when one of the
    # samplers runs, so the processing order does not matter.
    sorting_priority = -32

    @property
    def section(self):
        # Forge puts user sections in the parameters column (#txt2img_settings) → the "ANIMA 튜닝"
        # place in column 1, near the sampler. None (img2img, or sam3_layout_sections off) keeps it
        # in the scripts container as before.
        return layout_lanes.anima_section(bool(getattr(self, "is_img2img", False)))

    def title(self):
        return TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(TITLE, open=False):
            gr.Markdown(_HELP)
            status = _status_markdown()
            if status:
                gr.Markdown(status)
            with gr.Row():
                max_stage = gr.Slider(
                    label="ER SDE max stage",
                    minimum=sampler_params.MAX_STAGE_MIN,
                    maximum=sampler_params.MAX_STAGE_MAX,
                    step=1,
                    value=sampler_params.DEFAULT_MAX_STAGE,
                    elem_id=self.elem_id("er_sde_max_stage"),
                )
                eta = gr.Slider(
                    label="ER SDE eta",
                    minimum=sampler_params.ETA_MIN,
                    maximum=sampler_params.ETA_MAX,
                    step=sampler_params.ETA_STEP,
                    value=sampler_params.DEFAULT_ETA,
                    elem_id=self.elem_id("er_sde_eta"),
                )
        self.infotext_fields = [
            (max_stage, sampler_params.paste_max_stage),
            (eta, sampler_params.paste_eta),
        ]
        return [max_stage, eta]

    def process(self, p, *args):
        try:
            settings = sampler_params.settings_from_args(args, getattr(p, sampler_params.XYZ_ATTR, None))
        except Exception as exc:  # pragma: no cover - coerce_* never raise; keep generation alive anyway
            _log(f"bad arguments, using the defaults: {type(exc).__name__}: {exc}")
            settings = sampler_params.ErSdeSettings()
        sampler_params.apply_to(p, settings)
