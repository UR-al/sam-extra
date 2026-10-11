"""'모델 내리기' 버튼 — notebook 레이아웃 탭 줄에서 '기본값 저장' 바로 오른쪽의 버튼(javascript/notebook_unload_models.js)의 Forge 쪽.

누르면 생성 중이 아닐 때만 Forge 의 모델(Settings 의 'Unload All Models' 와 같은 함수)과 sam-extra 가 메모리에 쥔 모델
(SAM3 · Anima 3.8B · IP-Adapter · DeGrid · TIPO …, sam3ext/unload_registry.py)을 모두 내리고 VRAM 을 비운 뒤 버튼 옆에
"VRAM 사용 19.6 GB → 3.1 GB (Forge 모델 · SAM3)" 처럼 보여 준다. 순서·거절 규칙은 sam3ext/unload_models.py,
Gradio 연결은 sam3ext/ui_unload_models.py.

이 파일은 연결 시점(footer 의 on_after_component)만 등록한다. Script 클래스는 없다 — 스크립트 칸에 아무것도 놓지 않는다.
파일 이름은 negpip.py 보다 앞에 정렬되어야 한다(같은 확장 안에서 negpip.py 가 마지막 스크립트: metadata.ini).
"""
from modules import script_callbacks

from sam3ext import ui_unload_models

script_callbacks.on_after_component(ui_unload_models.on_after_component)
