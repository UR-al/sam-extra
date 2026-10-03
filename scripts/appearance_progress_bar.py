"""SAM Extra Progress Bar — Settings section and the ``GET /sam-extra/progress`` route.

A smoother txt2img/img2img progress bar, ported from diamfang/sd-webui-smooth-progress@7fe5810
(``scripts/smooth-progress.py`` registers upstream's settings-less route; the route logic lives in
``sam3ext/progress_api.py``, the page in ``javascript/progress_bar.js``). No Script class, no
generation hook: this file only registers Forge Settings and, at app start (also for ``--nowebui``),
the route. Upstream licence:

    MIT License

    Copyright (c) 2026 diamfang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Changes made by sam-extra (2026-10-03): upstream kept its options in each browser's localStorage behind
a ⚙️ popover; here they are Forge options (``sam3_progress_*``, Settings → SAM Extra Progress Bar),
applied live by the page through ``onOptionsChanged``. The bar is off by default. Upstream's colour
presets keep only the solid ones next to the theme colours (no gradients or animated colours), its
animation-speed option (animated presets only) is gone, the height is clamped to 10-50 px and all four
interruption styles are selectable.
"""
from __future__ import annotations

import sys
import traceback

import gradio as gr

from modules import script_callbacks, shared

from sam3ext import progress_api as pa


def on_ui_settings() -> None:
    from modules.ui_components import FormColorPicker

    section = pa.SETTINGS_SECTION
    shared.opts.add_option(
        pa.OPT_ENABLED,
        shared.OptionInfo(
            pa.DEFAULTS[pa.OPT_ENABLED],
            "부드러운 진행 막대 사용 (txt2img·img2img, Settings 저장 후 즉시 적용)",
            gr.Checkbox,
            section=section,
        ).info(
            "켜면 갤러리 위에 이 확장의 진행 막대를 그리고, 이 막대가 그리는 작업 동안 Forge 기본 막대는 숨깁니다(지우지 "
            "않음 — 그 막대를 기다리는 확장이 있습니다). 켜기 전에 시작한 작업은 Forge 막대로 보입니다. 그 탭에서 시작한 "
            "작업만 보여 주고, 작업 중에만 서버에 묻습니다. "
            "sd-webui-smooth-progress 가 함께 설치돼 있으면 이 막대는 켜지지 않습니다."
        ),
    )
    shared.opts.add_option(
        pa.OPT_SMOOTHNESS,
        shared.OptionInfo(
            pa.DEFAULTS[pa.OPT_SMOOTHNESS],
            "진행 막대: 부드러움",
            gr.Radio,
            {"choices": list(pa.SMOOTHNESS_CHOICES)},
            section=section,
        ).info(
            "Smooth > Accurate 는 남은 시간(ETA)에 맞춰 일정한 속도로 채우고, Smooth < Accurate 는 스텝마다 실제 "
            "값에 붙습니다. 스텝이 적은 생성에서는 부드러운 쪽일수록 실제보다 앞서거나 뒤처질 수 있습니다. 끝 신호 "
            "전에는 99.2% 에서 멈춥니다."
        ),
    )
    shared.opts.add_option(
        pa.OPT_TEXT_FORMAT,
        shared.OptionInfo(
            pa.DEFAULTS[pa.OPT_TEXT_FORMAT],
            "진행 막대: 글자",
            gr.Dropdown,
            {"choices": list(pa.TEXT_FORMAT_CHOICES)},
            section=section,
        ).info(
            "Steps 는 지금 패스의 스텝이고, 배치·Hires 처럼 패스가 여럿이면 [2/4] 처럼 몇 번째 패스인지 앞에 붙습니다. "
            "% 와 ETA 는 작업 전체 기준입니다. 대기 중에는 Forge 의 대기열 글자(In queue: 1/2)를 보여 줍니다."
        ),
    )
    shared.opts.add_option(
        pa.OPT_TEXT_ALIGN,
        shared.OptionInfo(
            pa.DEFAULTS[pa.OPT_TEXT_ALIGN],
            "진행 막대: 글자 위치 (0 왼쪽 · 50 가운데 · 100 오른쪽)",
            gr.Slider,
            dict(pa.ALIGN_RANGE),
            section=section,
        ),
    )
    shared.opts.add_option(
        pa.OPT_AFTER_FINISH,
        shared.OptionInfo(
            pa.DEFAULTS[pa.OPT_AFTER_FINISH],
            "진행 막대: 끝난 뒤",
            gr.Radio,
            {"choices": list(pa.AFTER_FINISH_CHOICES)},
            section=section,
        ).info("그대로 둠·글자만 숨김이면 다음 작업 전까지(처음 열었을 때도) 채워진 막대가 남습니다."),
    )
    shared.opts.add_option(
        pa.OPT_FADE_SECONDS,
        shared.OptionInfo(
            pa.DEFAULTS[pa.OPT_FADE_SECONDS],
            "진행 막대: 서서히 숨기는 시간 (초)",
            gr.Slider,
            dict(pa.FADE_RANGE),
            section=section,
        ),
    )
    shared.opts.add_option(
        pa.OPT_INTERRUPT_STYLE,
        shared.OptionInfo(
            pa.DEFAULTS[pa.OPT_INTERRUPT_STYLE],
            "진행 막대: 중단했을 때",
            gr.Radio,
            {"choices": list(pa.INTERRUPT_STYLE_CHOICES)},
            section=section,
        ).info(
            "Interrupt(또는 Esc)로 멈춘 작업은 멈춘 자리에 '중단됨' 을 남깁니다. Skip 은 다음 배치로 넘어갈 뿐이라 "
            "중단으로 보지 않습니다. 빨간색은 테마의 오류 색입니다."
        ),
    )
    shared.opts.add_option(
        pa.OPT_HEIGHT,
        shared.OptionInfo(
            pa.DEFAULTS[pa.OPT_HEIGHT],
            "진행 막대: 높이 (px)",
            gr.Slider,
            dict(pa.HEIGHT_RANGE),
            section=section,
        ).info("10~50 px 로 제한합니다(설정 파일·API 로 넣은 값도)."),
    )
    shared.opts.add_option(
        pa.OPT_COLOR,
        shared.OptionInfo(
            pa.DEFAULTS[pa.OPT_COLOR],
            "진행 막대: 색",
            gr.Dropdown,
            {"choices": list(pa.COLOR_CHOICES)},
            section=section,
        ).info(
            "테마 강조색은 SAM Extra Appearance 테마(Forge Default 면 Forge 의 강조색)를 따릅니다. 막대 위 글자색은 "
            "막대 색의 밝기에 맞춰 고릅니다."
        ),
    )
    shared.opts.add_option(
        pa.OPT_CUSTOM_COLOR,
        shared.OptionInfo(
            pa.DEFAULTS[pa.OPT_CUSTOM_COLOR],
            "진행 막대: 직접 지정 색 (색이 '직접 지정' 일 때)",
            FormColorPicker,
            {},
            section=section,
        ),
    )


def on_app_started_progress_api(demo, app) -> None:
    try:
        if pa.register_progress_routes(app):
            print(f"[SAM Extra] progress bar API: {pa.PROGRESS_API_PATH}")
    except Exception:
        traceback.print_exc(file=sys.stderr)


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_app_started(on_app_started_progress_api, name="sam-extra-progress-api")
