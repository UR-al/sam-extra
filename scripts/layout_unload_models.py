"""'모델 내리기' 버튼 — notebook 레이아웃 탭 줄에서 '기본값 저장' 바로 오른쪽의 버튼(javascript/notebook_unload_models.js)의 Forge 쪽.

누르면 생성 중이 아닐 때만 Forge 의 모델(Settings 의 'Unload All Models' 와 같은 함수)과 Forge 가 그 밖에 쥔 업스케일러 ·
TAESD · 얼굴 복원 모델, sam-extra 가 메모리에 쥔 모델(SAM3 · Anima 3.8B · IP-Adapter · DeGrid · TIPO …,
sam3ext/unload_registry.py)을 모두 내리고 VRAM 을 비운 뒤 버튼 아래에 "VRAM 사용 19.6 GB → 3.1 GB (Forge 모델 · SAM3)" 처럼
보여 준다. 순서·거절 규칙은 sam3ext/unload_models.py, Gradio 연결은 sam3ext/ui_unload_models.py.

이 파일이 하는 일은 둘이다:

* 연결 시점(footer 의 on_after_component)을 등록한다.
* ``UnloadModelsReload`` — 화면에 아무것도 놓지 않는 AlwaysVisible 스크립트(``ui`` 는 빈 목록, ``create_group = False``).
  Forge 의 ✨(갤러리 이미지 고해상도)는 process_images 에서 체크포인트 다시 읽기를 건너뛰므로, 버튼으로 내린 뒤라면 그보다
  먼저 도는 ``before_process`` 에서 고른 체크포인트를 읽는다(``unload_models.reload_for_upscale`` — ✨ 이고 체크포인트가 없을
  때만). txt2img 에만 있다(✨ 는 txt2img 갤러리에만 있다).

파일 이름은 negpip.py 보다 앞에 정렬되어야 한다(같은 확장 안에서 negpip.py 가 마지막 스크립트: metadata.ini).
"""
from modules import script_callbacks, scripts

from sam3ext import ui_unload_models, unload_models

script_callbacks.on_after_component(ui_unload_models.on_after_component)


class UnloadModelsReload(scripts.Script):
    # 화면에 놓는 것이 없다 — Forge 가 스크립트 칸에 빈 묶음을 그리지 않게(modules/scripts.py create_group)
    create_group = False

    def title(self):
        return "SAM Extra unload models (reload for hires fix)"

    def show(self, is_img2img):
        return False if is_img2img else scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def before_process(self, p, *args):
        from modules import sd_models

        unload_models.reload_for_upscale(p, sd_models)
