"""SAM Extra MCP — Forge Settings for the stdio MCP server shipped in ``mcp_server/``.

The server runs outside Forge in its own uv environment (the MCP SDK needs a newer pydantic than
Forge pins) and reads these four switches from Forge's settings file before every state-changing
tool call (``mcp_server/sam_extra_mcp/policy.py``). Nothing else is registered here: no route, no
UI, no import of the server package.

The keys and defaults below are the ones ``policy.PERMISSIONS`` enforces; they are written out
here because Forge must not import the server package (``tests/test_mcp_settings_script.py``
keeps the two in step).
"""
from __future__ import annotations

import os

import gradio as gr

from modules import script_callbacks, shared


SECTION = ("sam3_mcp", "SAM Extra MCP")

OPT_ALLOW_GENERATE = "sam3_mcp_allow_generate"
OPT_ALLOW_MODEL_SWITCH = "sam3_mcp_allow_model_switch"
OPT_ALLOW_INTERRUPT = "sam3_mcp_allow_interrupt"
OPT_ALLOW_DOWNLOAD = "sam3_mcp_allow_download"

# Not resolved: when the extension folder is a junction inside Forge's extensions folder, the
# server must be registered through that path so it can find this Forge's settings file.
MCP_PROJECT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp_server")


def registration_command(project: str = MCP_PROJECT) -> str:
    """The Claude Code command that registers this installation's server, user scope."""
    return f'claude mcp add --scope user sam-extra -- uv run --project "{project}" sam-extra-mcp'


OPTIONS = (
    (
        OPT_ALLOW_GENERATE,
        True,
        "MCP: 이미지 생성 허용 (generate — txt2img·img2img)",
        "Forge 밖에서 uv 로 도는 MCP 서버(확장의 mcp_server)가 에이전트 요청으로 생성하게 합니다. 네 스위치 모두 "
        "Apply settings 하면 다음 도구 호출부터 적용됩니다(재시작 불필요). 등록(사용자 범위): "
        + registration_command()
        + " — 첫 실행은 uv 가 환경을 만들려고 네트워크가 필요합니다.",
    ),
    (
        OPT_ALLOW_MODEL_SWITCH,
        False,
        "MCP: 체크포인트 바꾸기 허용 (models load — 프리셋·VAE·텍스트 인코더 함께)",
        "인스턴스 전체에 적용됩니다 — 웹 UI 를 쓰는 사람에게도 바뀐 모델이 보입니다.",
    ),
    (
        OPT_ALLOW_INTERRUPT,
        False,
        "MCP: 생성 중단·건너뛰기 허용 (progress interrupt·skip)",
        "웹 UI 에서 시작한 생성도 멈춥니다.",
    ),
    (
        OPT_ALLOW_DOWNLOAD,
        False,
        "MCP: VAE·텍스트 인코더 내려받기 허용 (module_download)",
        "models 폴더에 수 GB 파일을 씁니다. 받은 크기와(허깅페이스가 알려 주면) SHA256 이 맞을 때만 남기고, 있는 "
        "파일은 덮어쓰지 않습니다. 끄면 받을 곳만 알려 주고 네트워크에는 묻지도 않습니다.",
    ),
)


def _option(default: bool, label: str, info: str):
    try:
        # restrict_api: Forge builds that pass is_api=True on POST /sdapi/v1/options refuse API writes
        # to these keys. (2.29.2 passes is_api=False there, so the server's own refusal is what holds.)
        option = shared.OptionInfo(default, label, gr.Checkbox, section=SECTION, restrict_api=True)
    except TypeError:  # pragma: no cover - an OptionInfo without restrict_api
        option = shared.OptionInfo(default, label, gr.Checkbox, section=SECTION)
    return option.info(info)


def on_ui_settings() -> None:
    for key, default, label, info in OPTIONS:
        shared.opts.add_option(key, _option(default, label, info))


script_callbacks.on_ui_settings(on_ui_settings)
