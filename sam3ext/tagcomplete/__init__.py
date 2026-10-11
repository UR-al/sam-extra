# sam-extra 자체 코드(GPL-3.0-only) — 상류 tagcomplete 코드가 아니다. 이 폴더의 편입 파일(MIT, 머리 주석 참고)과 달리 새로 쓴 파일이다.
"""내장 Tag Autocomplete — sd-webui-tagcomplete-neo 를 sam-extra 에 합친 것 (all-in-one).

편입한 상류 파일 (MIT, Copyright (c) 2022 Dominik Reh; Forge Neo 포크 2026 Eduardo Abreu — 전문 ``LICENSE``):

- ``tag_autocomplete_helper.py``·``shared_paths.py``·``model_keyword_support.py``·``tag_frequency_db.py`` — 상류
  ``scripts/`` 의 같은 이름 파일. 바꾼 곳은 각 파일 머리와 THIRD_PARTY_NOTICES.md 에 적었다.
- ``javascript/*.js`` — 상류 ``javascript/`` 15개, 바꾸지 않음. Forge 가 자동으로 싣지 않는 하위 폴더라
  ``javascript/tagcomplete_loader.js``(확장 루트)가 붙인다.

이 확장의 코드: ``paths.py``(경로), ``coexist.py``(따로 설치된 tagcomplete 와 함께 돌지 않기 · 설정 스위치 · 옮기기 안내 ·
로더가 읽는 숨은 설정 ``sam3_tac_builtin_boot``), ``boot.py``(헬퍼 시작 · 그 시작 정보), ``scripts/builtin_tagcomplete.py``
(Forge 진입점).

이 패키지를 import 해도 아무 일도 일어나지 않는다(파일을 쓰거나 콜백을 거는 것은 ``boot.start_helper()`` 뿐).
"""

UPSTREAM_REPO = "https://github.com/eduardoabreu81/sd-webui-tagcomplete-neo"
UPSTREAM_COMMIT = "8e0d81fd0e25fd2cf98fb4d34d097df9f71a082b"
UPSTREAM_LABEL = "eduardoabreu81/sd-webui-tagcomplete-neo@8e0d81f"
ORIGINAL_REPO = "https://github.com/DominikDoom/a1111-sd-webui-tagcomplete"
