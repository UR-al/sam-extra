"""TIPO 프롬프트 확장 — txt2img 도구 줄의 🪄 버튼과 스타일 줄 아래 설정 칸.

🪄 를 누르면 지금 프롬프트를 TIPO(KBlueLeaf/TIPO-v2.1-1B-A200M)로 확장해 프롬프트 칸에 다시 쓴다. 사용자가 적은 텍스트는
그대로 두고 새 태그와 설명만 뒤에 붙인다(규칙은 sam3ext/tipo/prompt.py). 확장은 Forge 대기열 잠금 안에서 돌아 이미지 생성과
겹치지 않는다. 가중치는 "모델 받기"를 눌러야 받는다.

확장은 두 단계다: 결과를 만들어 숨은 State 에 두고, 다음 단계가 지금 프롬프트 칸의 값을 다시 읽어 누를 때와 같을 때만
넣는다 — 생성이 끝나길 기다리는 동안 고친 내용을 덮어쓰지 않게. 장치 선택은 javascript/tipo_device.js 가 브라우저에 기억한다.
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tipo import prompt as tp
from .tipo.runtime import MODEL_BYTES_LABEL, shared_runtime

BUTTON_ICON = "🪄"
BUTTON_ELEM_ID = "txt2img_tipo_expand"
BUTTON_TOOLTIP = "TIPO 로 프롬프트 확장 (설정: 스타일 줄 아래 'TIPO 프롬프트 확장')"
DEVICES = ("GPU", "CPU")
DEVICE_DEFAULT = "GPU"
# 새 토큰 한도 — 모델이 보통 먼저 EOS 로 끝내므로 넉넉히. 태그 36/48/72개 ≈ 110/140/210 토큰, 설명은 문장 몇 개가 더 붙는다.
MAX_NEW_TOKENS = {
    tp.MODE_TAGS: {tp.LENGTH_SHORT: 192, tp.LENGTH_NORMAL: 256, tp.LENGTH_LONG: 384},
    tp.MODE_TAGS_NL: {tp.LENGTH_SHORT: 384, tp.LENGTH_NORMAL: 576, tp.LENGTH_LONG: 896},
    tp.MODE_NL: {tp.LENGTH_SHORT: 256, tp.LENGTH_NORMAL: 384, tp.LENGTH_LONG: 640},
}
SOURCE_NOTE = (
    "TIPO-v2.1-1B-A200M by [KBlueLeaf](https://huggingface.co/KBlueLeaf/TIPO-v2.1-1B-A200M) "
    "(Kohaku License 1.0) — 모델 코드는 KohakUwULLM(Apache-2.0)"
)
MISSING_MESSAGE = f"TIPO 모델이 없습니다 — **모델 받기**({MODEL_BYTES_LABEL})를 눌러 주세요."
READY_MESSAGE = "🪄 를 누르면 지금 프롬프트를 확장합니다."
CHANGED_MESSAGE = "확장하는 동안 프롬프트가 바뀌어 결과를 넣지 않았습니다 — 다시 🪄 를 눌러 주세요."
BUSY_MESSAGE = "다른 창에서 모델을 받는 중입니다 — 끝난 뒤 🪄 를 누르면 됩니다."
REFRESH_TOKENS_JS = "function(){ if (typeof update_txt2img_tokens === 'function') update_txt2img_tokens(); }"

_TAG_INDEX: tp.TagIndex | None = None


def _log(message: str) -> None:
    text = f"[TIPO] {message}"
    try:
        print(text, file=sys.stderr)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"), file=sys.stderr)


def _tag_index() -> tp.TagIndex:
    """tagcomplete 의 danbooru CSV(카테고리 포함) — 처음 누를 때 한 번 읽는다."""
    global _TAG_INDEX
    if _TAG_INDEX is None:
        extensions_dir = Path(__file__).resolve().parents[2]
        _TAG_INDEX = tp.load_tag_index(tp.default_tag_csv_paths(extensions_dir))
    return _TAG_INDEX


@dataclass
class TipoPanel:
    accordion: Any
    mode: Any
    length: Any
    allow_new_names: Any
    device: Any
    seed: Any
    undo_button: Any
    download_button: Any
    status: Any
    undo_state: Any
    pending: Any


def build_tipo_panel(model_missing: bool) -> TipoPanel:
    import gradio as gr

    with gr.Accordion("TIPO 프롬프트 확장", open=False, elem_id="sam3_tipo_accordion") as accordion:
        with gr.Row():
            mode = gr.Radio(list(tp.MODES), value=tp.MODE_TAGS_NL, label="확장 방식", elem_id="sam3_tipo_mode")
            length = gr.Radio(list(tp.LENGTHS), value=tp.LENGTH_NORMAL, label="길이", elem_id="sam3_tipo_length")
        with gr.Row():
            allow_new_names = gr.Checkbox(False, label="새 작가·캐릭터·작품 허용", elem_id="sam3_tipo_allow_names")
            device = gr.Radio(list(DEVICES), value=DEVICE_DEFAULT, label="장치", elem_id="sam3_tipo_device")
            seed = gr.Number(-1, label="시드 (-1 = 매번 다르게)", precision=0, elem_id="sam3_tipo_seed")
        with gr.Row():
            undo_button = gr.Button("↩ 되돌리기", size="sm", elem_id="sam3_tipo_undo")
            download_button = gr.Button(
                f"모델 받기 ({MODEL_BYTES_LABEL})", size="sm", visible=model_missing, elem_id="sam3_tipo_download",
            )
        status = gr.Markdown(MISSING_MESSAGE if model_missing else READY_MESSAGE, elem_id="sam3_tipo_status")
        gr.Markdown(SOURCE_NOTE, elem_id="sam3_tipo_source")
        undo_state = gr.State(None)
        pending = gr.State(None)
    return TipoPanel(accordion, mode, length, allow_new_names, device, seed, undo_button, download_button,
                     status, undo_state, pending)


def create_tipo_button():
    from modules.ui_components import ToolButton

    return ToolButton(BUTTON_ICON, elem_id=BUTTON_ELEM_ID, tooltip=BUTTON_TOOLTIP)


def max_new_tokens(mode: str, length: str) -> int:
    budgets = MAX_NEW_TOKENS.get(mode, MAX_NEW_TOKENS[tp.MODE_TAGS_NL])
    return budgets.get(length, budgets[tp.LENGTH_NORMAL])


def handle_expand(prompt, width, height, mode, length, allow_new_names, device, seed, *, runtime=None, index=None):
    """출력: 적용 대기({before, after} 또는 None), 상태 줄, 모델 받기 버튼. 프롬프트 칸은 handle_apply 가 쓴다."""
    import gradio as gr

    runtime = runtime or shared_runtime()
    prompt = prompt or ""

    def unchanged(message, download=None):
        return None, message, download if download is not None else gr.update()

    if not prompt.strip():
        return unchanged("프롬프트를 먼저 적어 주세요.")
    if runtime.missing_files():
        return unchanged(MISSING_MESSAGE, gr.update(visible=True))
    mode = mode if mode in tp.MODES else tp.MODE_TAGS_NL
    length = length if length in tp.LENGTHS else tp.LENGTH_NORMAL
    device = device if device in DEVICES else DEVICE_DEFAULT
    index = index or _tag_index()
    parts = tp.analyze_prompt(prompt, index)
    try:
        aspect = float(width) / float(height) if width and height else None
    except (TypeError, ValueError):
        aspect = None
    tipo_input = tp.build_tipo_prompt(parts, mode, length, aspect)
    try:
        result = runtime.generate(
            tipo_input, requested_device=device, max_new_tokens=max_new_tokens(mode, length),
            seed=int(seed) if seed is not None else -1,
        )
    except Exception as exc:
        _log(f"expand failed:\n{traceback.format_exc()}")
        return unchanged(f"TIPO 실행 실패 — {type(exc).__name__}: {exc}")

    new_tags, description = tp.parse_tipo_output(mode, result.text, finished=result.finished)
    text, added = tp.assemble_prompt(prompt, parts, new_tags, description, mode, bool(allow_new_names), index)
    device_label = "GPU" if result.device == "cuda" else "CPU"
    notes = [note for note in (result.note, None if result.finished else "토큰 한도에서 멈춰 잘린 끝부분은 뺐습니다") if note]
    tail = f"시드 {result.seed} · {device_label} · {result.seconds:.1f}초" + "".join(f" — {note}" for note in notes)
    if text == prompt:
        return unchanged(f"TIPO 가 새로 붙일 것을 찾지 못했습니다 ({tail}).")
    summary = [f"태그 {len(added)}개"]
    if description and mode != tp.MODE_TAGS:
        summary.append("설명")
    return {"before": prompt, "after": text}, f"{' + '.join(summary)} 추가 · {tail}", gr.update(visible=False)


def handle_apply(current, pending, undo_state):
    """확장 결과를 넣는다 — 프롬프트 칸이 누를 때 그대로일 때만. 출력: 프롬프트, 되돌리기 상태, 상태 줄, 적용 대기."""
    import gradio as gr

    if not pending:
        return gr.update(), undo_state, gr.update(), None
    if (current or "") != pending["before"]:
        return gr.update(), undo_state, CHANGED_MESSAGE, None
    return pending["after"], pending["before"], gr.update(), None


def handle_undo(undo_state, prompt):
    import gradio as gr

    if undo_state is None:
        return gr.update(), None, "되돌릴 확장이 없습니다."
    return undo_state, None, "확장 전 프롬프트로 되돌렸습니다."


def handle_download(*, runtime=None):
    import gradio as gr

    runtime = runtime or shared_runtime()
    if not runtime.missing_files():
        return "TIPO 모델이 이미 있습니다. " + READY_MESSAGE, gr.update(visible=False)
    try:
        if not runtime.download():
            return BUSY_MESSAGE, gr.update(visible=False)
    except Exception as exc:
        _log(f"download failed:\n{traceback.format_exc()}")
        return f"모델 받기 실패 — {type(exc).__name__}: {exc}", gr.update(visible=True)
    return "모델을 받았습니다. " + READY_MESSAGE, gr.update(visible=False)


def wire_tipo(button, panel: TipoPanel, prompt, width, height) -> None:
    """확장은 Forge 대기열 잠금 안에서(이미지 생성과 겹치지 않게). 받기는 잠그지 않는다(몇 분 걸려도 생성은 막지 않게)."""
    from .ui_anima_reference import _run_exclusive

    def expand(prompt_value, width_value, height_value, mode, length, allow, device, seed):
        return _run_exclusive(
            "sam3_tipo_expand",
            lambda: handle_expand(prompt_value, width_value, height_value, mode, length, allow, device, seed),
        )

    button.click(
        fn=expand,
        inputs=[prompt, width, height, panel.mode, panel.length, panel.allow_new_names, panel.device, panel.seed],
        outputs=[panel.pending, panel.status, panel.download_button],
        show_progress="minimal",
    ).then(
        fn=handle_apply, inputs=[prompt, panel.pending, panel.undo_state],
        outputs=[prompt, panel.undo_state, panel.status, panel.pending], show_progress=False,
    ).then(fn=None, js=REFRESH_TOKENS_JS, show_progress=False)
    panel.undo_button.click(
        fn=handle_undo, inputs=[panel.undo_state, prompt], outputs=[prompt, panel.undo_state, panel.status],
        show_progress=False,
    ).then(fn=None, js=REFRESH_TOKENS_JS, show_progress=False)
    panel.download_button.click(fn=handle_download, inputs=[], outputs=[panel.status, panel.download_button])
