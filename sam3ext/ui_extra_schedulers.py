"""Extra Schedulers — script arguments, infotext keys, paste fields and the Gradio accordion.

API (``alwayson_scripts["Extra Schedulers (sam-extra)"]``): positional ``[custom_mode, custom_expression,
custom_sigmas, laplace_mu, laplace_beta]`` or one dict with those keys; missing values take the
defaults. ``custom_mode`` is ``expression`` or ``sigmas`` (the UI labels "Expression" and "Sigma
list" work too). The values only matter when the generation's Schedule type (or Hires schedule
type) is ``custom`` or ``Laplace``.

The script title is "Extra Schedulers (sam-extra)", not the accordion's "Extra Schedulers": the API
finds an always-on script by its title (the first match, case-insensitively), and
aoleg/Neo_ExtraSchedulers has an accordion of that name. Forge derives the element ids from the
title too (``script_txt2img_extra_schedulers_samextra_<item>``).

Infotext — written only for a scheduler of this extension the generation uses (main or hires pass),
so an image made with another scheduler — including another extension's under the same label —
carries none of these keys:

* ``Custom scheduler expression`` / ``Custom scheduler sigmas`` — the custom scheduler's text
  (whitespace collapsed), whichever mode was used;
* ``Laplace mu`` / ``Laplace beta`` — always both when Laplace was used (like Forge's
  ``Beta schedule alpha/beta``).

Pasting (PNG Info, Send to) restores the mode and text from whichever custom key is present, and
μ/β from their keys — or their defaults when the image used Laplace without them. Fields of a
scheduler the image did not use are left as they are. A pasted expression is only text in a
textbox; it is checked by the safe parser when a generation uses it.
"""
from __future__ import annotations

from typing import Callable

from sam3ext.extra_schedulers import settings as es_settings
from sam3ext.extra_schedulers.registry import LABEL_CUSTOM, LABEL_LAPLACE, is_ours

TITLE = "Extra Schedulers (sam-extra)"   # API alwayson_scripts key, element ids, XYZ axis prefix
ACCORDION_LABEL = "Extra Schedulers"
ARG_NAMES =("custom_mode", "custom_expression", "custom_sigmas", "laplace_mu", "laplace_beta")  # 뒤에만 덧붙인다

KEY_CUSTOM_EXPRESSION = "Custom scheduler expression"
KEY_CUSTOM_SIGMAS = "Custom scheduler sigmas"
KEY_LAPLACE_MU = "Laplace mu"
KEY_LAPLACE_BETA = "Laplace beta"
INFOTEXT_KEYS = (KEY_CUSTOM_EXPRESSION, KEY_CUSTOM_SIGMAS, KEY_LAPLACE_MU, KEY_LAPLACE_BETA)

MODE_LABELS = {es_settings.MODE_EXPRESSION: "Expression", es_settings.MODE_SIGMAS: "Sigma list"}
MODE_CHOICES = [MODE_LABELS[mode] for mode in es_settings.MODES]

# XYZ apply functions store their cell's values on ``p`` under this attribute; ``process`` reads them.
XYZ_ATTR = "_sam3_extra_schedulers_xyz"

USE_SAME_SCHEDULER = "Use same scheduler"


# ── arguments ──
def coerce_args(args) -> es_settings.ExtraSchedulerSettings:
    """Script arguments (positional, or one dict) → settings. Never raises."""
    args = list(args or ())
    if args and isinstance(args[0], dict):
        raw = dict(args[0])
    else:
        raw = dict(zip(ARG_NAMES, args))
    return es_settings.coerce(raw)


def normalize_text(text) -> str:
    return " ".join(str(text or "").split())


def format_number(value: float) -> str:
    """Shortest text that reads back as the same float (0.5 → '0.5', 1/3 → '0.3333333333333333')."""
    return repr(float(value))


# ── which of our schedulers a generation uses ──
def lookup_scheduler(name):
    """Forge's scheduler for a name or label (the API also accepts names, e.g. ``laplace``) — the same
    ``schedulers_map`` lookup ``KDiffusionSampler.get_sigmas`` makes; None when there is none."""
    if not name:
        return None
    try:
        from modules import sd_schedulers

        return sd_schedulers.schedulers_map.get(name)
    except Exception:
        return None


def schedulers_in_use(p) -> set[str]:
    """Labels of this extension's schedulers that the main pass and, with Hires. fix on, the hires
    pass run.

    Only a name Forge resolves to one of ours counts. A label another extension registered first
    (ours was skipped, e.g. aoleg/Neo_ExtraSchedulers installed alongside) runs that extension's
    scheduler with its own values, and a hidden one runs none (get_sigmas falls back to the model's
    sigmas) — writing this accordion's values for them would record values that were not used.
    """
    names = [getattr(p, "scheduler", None)]
    if getattr(p, "enable_hr", False):
        hires = getattr(p, "hr_scheduler", None)
        if hires and hires != USE_SAME_SCHEDULER:
            names.append(hires)
    labels = set()
    for name in names:
        scheduler = lookup_scheduler(name)
        if is_ours(scheduler):
            labels.add(scheduler.label)
    return labels


# ── infotext ──
def infotext_items(current: es_settings.ExtraSchedulerSettings, labels) -> dict:
    items = {}
    labels = set(labels or ())
    if LABEL_CUSTOM in labels:
        if current.custom_mode == es_settings.MODE_SIGMAS:
            items[KEY_CUSTOM_SIGMAS] = normalize_text(current.custom_sigmas)
        else:
            items[KEY_CUSTOM_EXPRESSION] = normalize_text(current.custom_expression)
    if LABEL_LAPLACE in labels:
        items[KEY_LAPLACE_MU] = format_number(current.laplace_mu)
        items[KEY_LAPLACE_BETA] = format_number(current.laplace_beta)
    return items


def write_infotext(p, current: es_settings.ExtraSchedulerSettings) -> dict:
    """Replace this script's keys in ``p.extra_generation_params``.

    The old keys go first: XYZ cells are shallow copies of one ``p`` that share the dict, so a key
    from a cell that used Laplace would otherwise stay on the next cell's images.
    """
    params = getattr(p, "extra_generation_params", None)
    if not isinstance(params, dict):
        params = {}
        p.extra_generation_params = params
    for key in INFOTEXT_KEYS:
        params.pop(key, None)
    items = infotext_items(current, schedulers_in_use(p))
    params.update(items)
    return items


# ── pasting (PNG Info / Send to) ──
def _labels_in_params(params: dict) -> set[str]:
    """Schedule types named in pasted infotext, as our labels where Forge resolves them to one of ours
    (a name or a fork-README alias such as ``laplace``), else as written."""
    labels = set()
    for key in ("Schedule type", "Hires schedule type"):
        value = params.get(key)
        if value and value != USE_SAME_SCHEDULER:
            scheduler = lookup_scheduler(str(value))
            labels.add(scheduler.label if is_ours(scheduler) else str(value))
    return labels


def paste_mode(params: dict):
    if KEY_CUSTOM_SIGMAS in params:
        return MODE_LABELS[es_settings.MODE_SIGMAS]
    if KEY_CUSTOM_EXPRESSION in params:
        return MODE_LABELS[es_settings.MODE_EXPRESSION]
    return None


def paste_expression(params: dict):
    value = params.get(KEY_CUSTOM_EXPRESSION)
    return None if value is None else str(value)


def paste_sigmas(params: dict):
    value = params.get(KEY_CUSTOM_SIGMAS)
    return None if value is None else str(value)


def _paste_laplace(params: dict, key: str, coerce: Callable, default: float):
    if key in params:
        return coerce(params[key])
    if LABEL_LAPLACE in _labels_in_params(params):
        return default
    return None


def paste_laplace_mu(params: dict):
    return _paste_laplace(params, KEY_LAPLACE_MU, es_settings.coerce_laplace_mu, es_settings.LAPLACE_MU_DEFAULT)


def paste_laplace_beta(params: dict):
    return _paste_laplace(params, KEY_LAPLACE_BETA, es_settings.coerce_laplace_beta, es_settings.LAPLACE_BETA_DEFAULT)


# ── Gradio ──
HELP = (
    "Schedule type 목록에 **Cosine · CosineExponential blend · Phi · Laplace · Karras Dynamic · custom** 이 "
    "추가됩니다. 이 아코디언은 그중 **custom** 과 **Laplace** 의 값만 담으며, 다른 스케줄러를 고르면 쓰이지 않습니다.\n\n"
    "- **custom 식** — 스텝마다 시그마를 계산합니다. 변수 `m`(sigma min) · `M`(sigma max) · `n`(스텝 수) · "
    "`s`(이번 스텝, 0부터) · `x`(= s / (n − 1)), 상수 `phi` · `pi` · `e`, 연산 `+ − * / **`, 함수 "
    "`abs sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil min max`. "
    "값은 0보다 커야 하고 마지막 0은 자동으로 붙습니다. 기본값 `M * (m / M) ** x` 는 Exponential 과 같은 곡선입니다. "
    "Anima·Flux 같은 flow 모델은 시그마가 `M`(= 1)을 넘지 않게 쓰세요.\n"
    "- **custom 시그마 목록** — `[1.0, 0.6, 0.25, 0.1, 0.0]` 처럼 적으면 스텝 수에 맞춰 로그-선형 보간합니다(Forge 의 "
    "Align Your Steps 와 같은 방식). 1.0 으로 시작해 0.0 으로 끝나는 목록은 sigma max ~ sigma min 사이로 늘려 쓰고, "
    "그 밖의 목록은 시그마 값 그대로 씁니다(끝의 0.0 은 마지막 0).\n"
    "- **Laplace μ / β** — ComfyUI LaplaceScheduler 와 같은 값(기본 0 / 0.5). 시그마가 e^μ 근처에 모이고 β 가 클수록 "
    "넓게 퍼집니다. 노드처럼 sigma min/max 로 잘랐을 때 같은 시그마가 되풀이되면(flow 모델은 기본 μ 0 에서 앞쪽 절반이 1) "
    "같은 곡선 중 그 범위 안의 구간에 스텝을 고르게 다시 놓습니다. flow 모델도 μ 는 0 그대로 두고(음수 μ 는 물 빠진 이미지), "
    "샘플러는 Euler 계열을 쓰세요(Res Multistep 은 마지막 큰 스텝에서 잔 입자가 남음).\n"
    "- **flow 모델**(Anima 등)에는 Cosine · CosineExponential blend · Phi · Karras Dynamic 이 맞지 않습니다 — Forge 의 "
    "Karras · Exponential 처럼 물 빠진 듯 뿌옇게 나옵니다(SD · SDXL 용). Simple · Beta · Linear Quadratic · Laplace 나 "
    "시간 shift 를 넣은 custom 식을 쓰세요.\n"
    "- **Karras Dynamic** 의 ρ(기본 7)는 Karras 와 같이 쓰는 Settings → Sampler Parameters → rho(`--adv-samplers` 일 때 "
    "보임) 또는 붙여 넣은 `Schedule rho` 로 바뀝니다. 약 4 보다 작으면 시그마가 도중에 다시 올라가므로, 그런 스케줄은 "
    "쓰지 않고 생성을 오류로 멈춥니다.\n\n"
    "식·목록은 이미지 정보에 남고 붙여 넣으면 되살아납니다. 식은 안전한 계산기로만 읽으며(파이썬 코드로 실행하지 않음), "
    "잘못된 식이면 생성이 오류로 멈춥니다."
)


def build_controls(elem_id: Callable[[str], str]):
    """(custom_mode, custom_expression, custom_sigmas, laplace_mu, laplace_beta)."""
    import gradio as gr

    with gr.Accordion(ACCORDION_LABEL, open=False, elem_id=elem_id("accordion")):
        gr.Markdown(HELP)
        mode = gr.Radio(
            label="custom: mode",
            choices=MODE_CHOICES,
            value=MODE_LABELS[es_settings.DEFAULTS.custom_mode],
            elem_id=elem_id("custom_mode"),
        )
        expression = gr.Textbox(
            label="custom: expression",
            value=es_settings.DEFAULT_EXPRESSION,
            lines=1,
            max_lines=3,
            placeholder="m + (M - m) * (1 - x) ** 3",
            elem_id=elem_id("custom_expression"),
        )
        sigmas = gr.Textbox(
            label="custom: sigma list",
            value=es_settings.DEFAULT_SIGMAS,
            lines=1,
            max_lines=3,
            placeholder="[1.0, 0.6, 0.25, 0.1, 0.0]",
            elem_id=elem_id("custom_sigmas"),
        )
        with gr.Row():
            laplace_mu = gr.Slider(
                label="Laplace mu",
                minimum=es_settings.LAPLACE_MU_MIN,
                maximum=es_settings.LAPLACE_MU_MAX,
                step=0.1,
                value=es_settings.LAPLACE_MU_DEFAULT,
                elem_id=elem_id("laplace_mu"),
            )
            laplace_beta = gr.Slider(
                label="Laplace beta",
                minimum=es_settings.LAPLACE_BETA_MIN,
                maximum=es_settings.LAPLACE_BETA_MAX,
                step=0.1,
                value=es_settings.LAPLACE_BETA_DEFAULT,
                elem_id=elem_id("laplace_beta"),
            )
    return mode, expression, sigmas, laplace_mu, laplace_beta


def infotext_fields(controls) -> list:
    mode, expression, sigmas, laplace_mu, laplace_beta = controls
    return [
        (mode, paste_mode),
        (expression, paste_expression),
        (sigmas, paste_sigmas),
        (laplace_mu, paste_laplace_mu),
        (laplace_beta, paste_laplace_beta),
    ]
