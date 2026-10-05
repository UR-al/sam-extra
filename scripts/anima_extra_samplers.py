"""Extra Samplers — registers the samplers of ``sam3ext/extra_samplers`` and holds the ER SDE values.

At import (before Forge builds its sampler dropdowns) ``registry.register()`` adds eighteen entries to
Forge's sampler list (code, origins and licences in ``sam3ext/extra_samplers/__init__.py``):

* ER SDE — ``ER SDE (Reverse-time)``, ``ER SDE (ODE)``, ``ER SDE (Tunable)``;
* Dy — ``Euler Dy CFG++``, ``Euler SMEA Dy CFG++``, ``Euler Dy``, ``Euler SMEA Dy``;
* DPM++ in λ — ``DPM++ 4M SDE``, ``DPM++ 2M SDE Heun``, ``DPM++ 2M (flow ODE)``, ``DPM++ 2M Heun (flow ODE)``,
  ``DPM++ 3M (flow ODE)``;
* multistep ODE — ``UniPC bh2``, ``IPNDM``, ``IPNDM_V``, ``DEIS``;
* CFG++ — ``CFG++ UD10 AB``;
* Restart — ``Restart (flow)``.

They are picked in Forge's own Sampler / Hires sampler dropdowns, the XYZ ``Sampler`` axis and the API
``sampler_name``.

This always-on script adds the collapsed ``Extra Samplers`` accordion with the five values only the
three ER SDE entries of this extension read (``params``):

* ``ER SDE max stage`` (1-3, default 3) — infotext ``ER SDE max stage`` when not 3;
* ``ER SDE eta`` (0-10, default 1) — Reverse-time and Tunable, infotext ``ER SDE eta`` when not 1 (or
  when the request's Forge ``Eta`` is not 1, see ``params.record_infotext``);
* ``ER SDE noise window`` (checkbox, off) with ``ER SDE noise start`` / ``ER SDE noise end`` (sampling
  percentages 0-1, default 0.2 / 0.8) — Reverse-time and Tunable inject noise only on the steps that
  land inside the window; infotext ``ER SDE noise window: 0.2-0.8`` when on (and η > 0).

They are stored on the request in ``process`` (so both passes of a hires request use them) and the
sampler writes the infotext when it runs. Pasting an infotext restores them (missing keys → the
defaults; an ``ER SDE (Reverse-time)`` infotext with Forge's ``Eta`` but no ``ER SDE eta`` — written
by aoleg/Neo_ExtraSchedulers — gives η from ``Eta``). XYZ axes: ``[Extra Samplers] ER SDE max stage``,
``[Extra Samplers] ER SDE eta`` and ``[Extra Samplers] ER SDE noise window`` (values ``off`` or
``start-end``, e.g. ``0.1-0.9``; anything else is refused before the grid starts). API:
``alwayson_scripts["Extra Samplers"]["args"] = [max_stage, eta, noise_window, noise_start, noise_end]``
(every one optional; requests with the old two arguments run unchanged, with the window off).

Everything else comes from Forge: ``Eta`` (DPM++ 4M SDE, DPM++ 2M SDE Heun) and ``Sigma churn / tmin / tmax /
noise`` (Settings → Sampler Parameters, shown with ``--adv-samplers``; also infotext/override settings)
reach the samplers through Forge's ``Sampler.initialize``, which writes their infotext. The three
"flow ODE" entries take no Eta (they are η = 0 by definition) and write none.

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
XYZ_NOISE_WINDOW = f"{XYZ_PREFIX} ER SDE noise window"


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


def _xyz_set_noise_window(p, x, xs):
    """Store ``(enabled, start, end)`` of an ``off`` / ``start-end`` cell (``_xyz_confirm_noise_window``
    has refused anything else before the grid started)."""
    parsed = sampler_params.parse_noise_window(x)
    if parsed is not None:
        _xyz_set(p, parsed, xs, field="noise_window")


def _xyz_confirm_noise_window(p, xs):
    """Forge's ``AxisOption.confirm``: every value must be ``off`` or ``start-end`` with 0 ≤ start, end ≤ 1."""
    for x in xs:
        if sampler_params.parse_noise_window(x) is None:
            raise ValueError(
                f'"{XYZ_NOISE_WINDOW}" value "{x}" is not "{sampler_params.NOISE_WINDOW_OFF}" or '
                '"start-end" (sampling percentages 0-1, e.g. 0.2-0.8)'
            )


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
        xyz_grid.AxisOption(XYZ_NOISE_WINDOW, str, _xyz_set_noise_window, confirm=_xyz_confirm_noise_window),
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
    "Forge 샘플러 목록에 18종을 더합니다 — 고르는 곳은 Forge 의 Sampler 드롭다운입니다.\n"
    "**ER SDE**: ER SDE (Reverse-time) · ER SDE (ODE) · ER SDE (Tunable) / **Dy**: Euler Dy CFG++ · "
    "Euler SMEA Dy CFG++ · Euler Dy · Euler SMEA Dy / **DPM++ (λ)**: DPM++ 4M SDE · DPM++ 2M SDE Heun · "
    "DPM++ 2M (flow ODE) · DPM++ 2M Heun (flow ODE) · DPM++ 3M (flow ODE) / **다단계 ODE**: UniPC bh2 · IPNDM · "
    "IPNDM_V · DEIS / **CFG++**: CFG++ UD10 AB / **Restart**: Restart (flow). "
    "새 항목은 모두 스케줄러 힌트가 없어 Automatic 이 모델 기본 스케줄(flow 모델은 Normal)을 씁니다. "
    "Anima 에서 화질은 확인하지 않았습니다.\n\n"
    "- 아래 값은 **이 확장의 ER SDE 세 항목에만** 쓰입니다(Forge 내장 **ER SDE** 는 그대로). **max stage** 는 풀이 "
    "차수(1 = 지수 Euler, 2·3 = 다단계 보정, 기본 3). **eta** 는 잡음 척도의 지수(기본 1, 0 = ODE 와 같음, 클수록 "
    "스텝마다 잡음↑ · 너무 크면 결과가 깨질 수 있음):\n"
    "  - **ER SDE (Reverse-time)** — h(λ)=λ^(η+1): η 1 = 고전 역시간 SDE.\n"
    "  - **ER SDE (Tunable)** — h(λ)=λ·(e^(λ^0.3)+10)^η (ComfyUI 의 ER-SDE): η 1·잡음 구간 끔 = Forge 내장 "
    "**ER SDE** 와 같은 결과.\n"
    "  - **ER SDE (ODE)** — h(λ)=λ: 잡음을 넣지 않습니다(eta·잡음 구간 무시).\n"
    "- **잡음 구간**: 켜면 시작\\~끝(샘플링 진행률 0–1, 기본 0.2–0.8 — Anima shift 3 에서 σ 0.92\\~0.43)에서만 SDE "
    "잡음을 넣고 나머지 스텝은 ODE 로 풉니다. 시작이 끝보다 크거나 같으면 잡음 없음. (Reverse-time·Tunable)\n"
    "- Forge 의 Settings → Sampler Parameters(`--adv-samplers` 로 켜야 보임) 값도 그대로 쓰입니다: **sigma noise** 는 "
    "Reverse-time·Tunable·4M SDE·2M SDE Heun·Restart (flow)·Dy 의 churn 잡음에, **Eta for k-diffusion samplers** 는 "
    "DPM++ 4M SDE·2M SDE Heun 에, **sigma churn/tmin/tmax** 는 Dy 네 항목에(기본 churn 0 = 재잡음 없음). 바꾼 값은 "
    "Forge 가 infotext 에 남깁니다.\n"
    "- **CFG++** 두 Dy 항목은 CFG 1\\~2 를 권장합니다. Dy 는 2·3번째 스텝에, SMEA Dy 는 0번째(×1.25 해상도 — VRAM 더 씀)·"
    "1번째 스텝에 모델을 한 번 더 부릅니다. Anima 같은 flow 모델에서도 돕니다. Forge 의 **Spectrum Integrated**, "
    "Wan I2V·PiD 처럼 전체 해상도 입력을 쓰는 경우에는 이 보조 스텝 없이 일반 (CFG++) Euler 스텝으로 돌고, infotext "
    "`Extra Samplers status` 에 이유를 남깁니다.\n"
    "- **Euler Dy · Euler SMEA Dy** — CFG++ 판과 같은 보조 스텝, 보통 CFG 로 씁니다. churn 0(기본)이면 다시 잡음을 "
    "넣지 않습니다(reForge 의 같은 이름 샘플러는 매 스텝 churn — 여기는 k-diffusion 의 min 규칙, flow 에서도 σ<1).\n"
    "- **DPM++ 2M SDE Heun** — Forge 의 DPM++ 2M SDE 에 Heun 보정. Eta·sigma noise·Brownian 잡음은 Forge 설정 그대로.\n"
    "- **flow ODE 셋** — 잡음 없는 DPM-Solver++ 를 모델의 λ 좌표(flow 면 log((1−σ)/σ))에서 풉니다(Res Multistep 의 "
    "flow 판 비교용). Eta 를 쓰지 않습니다. **DPM++ 3M (flow ODE)** 는 Forge 의 DPM++ 3M SDE 를 Eta 0 으로 돌린 것과 "
    "같습니다(끝에서 두 번째 σ 버림도 같음).\n"
    "- **UniPC bh2** — Forge 의 UniPC 를 bh2 변형으로(나머지는 Forge 의 UniPC 와 같음).\n"
    "- **IPNDM · IPNDM_V · DEIS** — 스텝당 모델 1회의 다단계 ODE. IPNDM·DEIS 는 ComfyUI 와 같은 결과이고, IPNDM_V 는 "
    "0.33.1 에서 원본(zju-pi, ComfyUI 도 같음)의 4차 계수 오타를 고쳐 스텝 간격 비율이 일정할 때만 ComfyUI 와 "
    "같습니다(반올림 차이 안에서). 고친 뒤 shift 3 스케줄에서는 IPNDM_V 와 IPNDM 의 정확도가 비슷합니다(CPU 시험). "
    "IPNDM_V 는 **Linear Quadratic** 에서 여전히 못 씁니다(아래). DEIS 는 계수 계산에 "
    "CPU 를 조금 더 씁니다.\n"
    "- **Linear Quadratic 스케줄**(스텝 앞 절반이 σ 1 가까이에 몰렸다가 간격이 갑자기 커짐): UniPC bh2 와 Forge 내장 "
    "UniPC 는 이 스케줄에서 불안정합니다 — 코드 결함이 아니라 조합의 성질이라(ComfyUI 도 같음) 고치지 않았습니다. "
    "Simple · Normal · SGM Uniform 은 CPU 시험에서 문제없었습니다. **IPNDM_V** 도 이 스케줄에는 쓰지 마세요: 0.33.1 의 "
    "계수 수정으로 초록 잡음만 없어졌고 그림은 여전히 타고 번집니다(10-05 GPU 확인, ComfyUI 도 같음) — 아주 작은 스텝 뒤 "
    "간격이 3.7배로 뛰는 곳을 넘어 4차로 외삽하는 방식 자체의 성질입니다. IPNDM_V 는 **Simple** 로 쓰고(GPU 에서 깨끗함, "
    "Normal 은 거칢), Linear Quadratic 에는 고정 계수의 **IPNDM** 을 쓰세요.\n"
    "- **CFG++ UD10 AB** — Anima 용 ComfyUI 샘플러(CFG 2 권장, CFG 1 이면 이력 섞인 Euler). CFG 2 를 넘으면 Forge 가 "
    "CFG++ 경고를 남깁니다.\n"
    "- **Restart (flow)** — Heun 스텝에 Restart(논문 알고리즘)를 flow 에 맞게(α=1−σ) 더한 것. 20 스텝 이상에서 "
    "σ/(1−σ) 0.1\\~2 구간(flow σ 0.09\\~0.67)을 한 번(36 스텝부터 두 번) 다시 잡음 넣고 스케줄 그대로 되짚어 풉니다 — 모델 "
    "호출이 Heun 보다 많습니다(Anima 28 스텝에서 약 1.4배). Forge 내장 **Restart** 는 flow 모델에서 σ>1 로 잡음을 넣어 "
    "깨지므로 flow 모델에서는 Settings → Hide samplers 로 숨기기를 권합니다."
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
                     "아래 값은 이 확장의 ER SDE 에만 쓰입니다.")
    if report.skipped_missing:
        names = ", ".join(f"`{label}`" for label in report.skipped_missing)
        lines.append(f"⚠ 이 Forge 에 필요한 함수가 없어 건너뜀: {names}")
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
            with gr.Row():
                noise_window = gr.Checkbox(
                    label="ER SDE noise window",
                    value=sampler_params.DEFAULT_NOISE_WINDOW,
                    elem_id=self.elem_id("er_sde_noise_window"),
                )
                noise_start = gr.Slider(
                    label="ER SDE noise start",
                    minimum=sampler_params.NOISE_PERCENT_MIN,
                    maximum=sampler_params.NOISE_PERCENT_MAX,
                    step=sampler_params.NOISE_PERCENT_STEP,
                    value=sampler_params.DEFAULT_NOISE_START,
                    elem_id=self.elem_id("er_sde_noise_start"),
                )
                noise_end = gr.Slider(
                    label="ER SDE noise end",
                    minimum=sampler_params.NOISE_PERCENT_MIN,
                    maximum=sampler_params.NOISE_PERCENT_MAX,
                    step=sampler_params.NOISE_PERCENT_STEP,
                    value=sampler_params.DEFAULT_NOISE_END,
                    elem_id=self.elem_id("er_sde_noise_end"),
                )
        self.infotext_fields = [
            (max_stage, sampler_params.paste_max_stage),
            (eta, sampler_params.paste_eta),
            (noise_window, sampler_params.paste_noise_window),
            (noise_start, sampler_params.paste_noise_start),
            (noise_end, sampler_params.paste_noise_end),
        ]
        return [max_stage, eta, noise_window, noise_start, noise_end]

    def process(self, p, *args):
        try:
            settings = sampler_params.settings_from_args(args, getattr(p, sampler_params.XYZ_ATTR, None))
        except Exception as exc:  # pragma: no cover - coerce_* never raise; keep generation alive anyway
            _log(f"bad arguments, using the defaults: {type(exc).__name__}: {exc}")
            settings = sampler_params.ErSdeSettings()
        sampler_params.apply_to(p, settings)
