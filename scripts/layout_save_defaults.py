"""'기본값 저장' 버튼 — notebook 레이아웃 탭 줄 오른쪽 끝의 버튼 하나(javascript/notebook_save_defaults.js)의 Forge 쪽.

한 번 누르면 확인 없이 두 가지를 저장한다:

1. Forge 의 Settings → Defaults 'Apply'(#ui_defaults_apply, modules/ui_loadsave.py) — 페이지가 그 버튼을 눌러
   ui-config 파일에 지금 화면 값을 쓰고, Forge 가 그리는 결과 글에서 바뀐 개수를 읽는다.
2. 활성 UI Preset 의 설정 22개(forge_checkpoint_{p} … {p}_i2i_batch_size) — Forge 가 페이지를 열 때와 프리셋을
   바꿀 때마다 그 칸들을 이 값으로 덮어쓰기 때문이다(modules_forge/main_entry.py on_preset_change). 프리셋 값이
   0 인(= Forge 가 화면 값을 덮어쓰지 않는) 스텝·크기·CFG 는 0 그대로 둔다 — 그 화면 값은 1 의 ui-config 로 남는다.
   검사·저장은 sam3ext/save_defaults.py, Gradio 연결은 sam3ext/ui_save_defaults.py.

이 파일은 연결 시점(footer 의 on_after_component)만 등록한다. Script 클래스는 없다 — 스크립트 칸에 아무것도 놓지 않는다.
파일 이름은 negpip.py 보다 앞에 정렬되어야 한다(같은 확장 안에서 negpip.py 가 마지막 스크립트: metadata.ini).
"""
from modules import script_callbacks

from sam3ext import ui_save_defaults

script_callbacks.on_after_component(ui_save_defaults.on_after_component)
