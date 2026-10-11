# sam-extra 자체 코드(GPL-3.0-only) — 상류 tagcomplete 코드가 아니다.
"""내장 Tag Autocomplete 의 경로. 표준 라이브러리만 쓴다 — Forge 를 import 하지 않아 TIPO·옮기기 도구·테스트도 쓴다.

``Path.resolve()`` 대신 ``os.path.abspath`` 를 쓴다: 확장 폴더가 정션·심볼릭 링크여도 Forge 가 아는 철자(data_path 아래)
그대로 남아야 Gradio 의 ``file=`` 허용 경로 검사가 태그 파일을 403 으로 막지 않는다.

테스트는 ``shared_paths`` 를 import 하기 전에 ``TAGS_DIR`` 을 임시 폴더로 바꿀 수 있다(``shared_paths`` 가 import 할 때
이 모듈의 값을 읽는다).
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(os.path.abspath(__file__)).parent
EXT_ROOT = PACKAGE_DIR.parents[1]
# the 15 upstream JS files (Forge never auto-loads a subfolder of javascript/; the loader adds them)
JS_DIR = PACKAGE_DIR / "javascript"
# tag lists, chants, tag_frequency.db and temp/ — the same layout as upstream's tags/ folder
TAGS_DIR = EXT_ROOT / "tags"
TAGS_TEMP_DIR = TAGS_DIR / "temp"
TAG_FREQUENCY_DB = TAGS_DIR / "tag_frequency.db"
