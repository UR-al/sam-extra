# sam-extra 자체 코드(GPL-3.0-only) — 상류 tagcomplete 코드가 아니다.
"""내장 Tag Autocomplete 와 따로 설치된 tagcomplete 가 함께 돌지 않게 한다 · 설정 스위치와 안내 (Forge 를 import 하지 않음).

둘이 함께 돌면: 같은 ``tac_*`` 설정을 두 번 등록하고, ``/tacapi/v1/*`` 경로가 두 번 붙으며(먼저 붙은 쪽이 받음),
``<forge>/tmp/tagAutocompletePath.txt`` 를 서로 덮어쓰고, 페이지에서는 두 번째 사본의 const/class 파일이
"Identifier has already been declared" 로 실패해 반만 초기화된다. 그래서 따로 설치된 사본이 켜져 있으면 내장이 쉰다
(파이썬: 헬퍼를 시작하지 않음 · 로더: JS 를 붙이지 않음 — ``javascript/tagcomplete_loader.js`` 도 페이지에서 한 번 더 본다).

판별: Forge 의 ``extensions.active()`` 가운데 이 확장이 아닌 것의 폴더에 상류의 파일(``javascript/tagAutocomplete.js`` 나
``scripts/tag_autocomplete_helper.py``)이 있으면 — 폴더 이름과 무관하게 a1111-sd-webui-tagcomplete 계열 모두. Forge 는 꺼진
확장의 스크립트·JS 를 싣지 않으므로 Extensions 탭에서 끈 사본은 세지 않는다.
"""

from __future__ import annotations

import html
import os
from pathlib import Path

OPT_ENABLED = "sam3_builtin_tagcomplete_enabled"
OPT_NOTE = "sam3_tac_builtin_note"
# upstream's section (tag_autocomplete_helper.on_ui_settings TAC_SECTION) — the switch and the note sit on top of it
SECTION = ("tac", "Tag Autocomplete")
UPSTREAM_MARKERS = ("javascript/tagAutocomplete.js", "scripts/tag_autocomplete_helper.py")
# tac_* settings that name a file in tags/ ("None" = no file)
TAG_FILE_SETTINGS = ("tac_tagFile", "tac_extra.extraFile", "tac_translation.translationFile", "tac_chantFile")

REASON_OFF = "turned off in Settings -> Tag Autocomplete"

_warned: set[str] = set()


def enabled_setting(opts) -> bool:
    """``sam3_builtin_tagcomplete_enabled`` — 기본 켬. 문자열로 저장된 값도 읽는다(scripts/negpip.py 의 스위치와 같은 규칙)."""
    value = getattr(opts, OPT_ENABLED, True)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "off", "no"}:
            return False
        return True
    return True if value is None else bool(value)


def _same_folder(a, b) -> bool:
    try:
        if os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b)):
            return True
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except (OSError, ValueError, TypeError):
        return False


def standalone_reason(active_extensions, ext_root) -> str | None:
    """켜진 확장 중 따로 설치된 tagcomplete 가 있으면 그 이유, 없으면 None. ``active_extensions`` 는 Forge 의
    ``Extension`` 들(``name``·``path``)."""
    for ext in active_extensions or ():
        path = getattr(ext, "path", None)
        if not path or _same_folder(path, ext_root):
            continue
        for marker in UPSTREAM_MARKERS:
            try:
                found = (Path(path) / marker).is_file()
            except OSError:
                found = False
            if found:
                name = getattr(ext, "name", None) or os.path.basename(os.path.normpath(path))
                return f"another tagcomplete extension is enabled ({name}: {path})"
    return None


def missing_tag_files(opts, tags_dir) -> list[tuple[str, str]]:
    """``(설정 키, 파일 이름)`` — tac_* 설정이 고른 파일이 tags/ 에 없는 것. 상류는 이때 콘솔에 404 만 남기고 그 목록을
    통째로 건너뛴다(태그 파일이 없으면 extra 파일도 읽지 않음)."""
    missing = []
    for key in TAG_FILE_SETTINGS:
        value = getattr(opts, key, None)
        if not isinstance(value, str) or not value.strip() or value.strip() == "None":
            continue
        try:
            present = (Path(tags_dir) / value.strip()).is_file()
        except (OSError, ValueError):
            present = False
        if not present:
            missing.append((key, value.strip()))
    return missing


def _print(text: str) -> None:
    try:   # 레거시 콘솔 코드페이지(cp949 등)에서 경로의 글자로 시작이 죽지 않게
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"))


def warn_once(reason: str) -> None:
    """내장이 쉬는 이유를 프로세스당 한 번 — Reload UI 마다 콘솔을 채우지 않는다."""
    if reason in _warned:
        return
    _warned.add(reason)
    text = f"[sam-extra] Tag Autocomplete: built-in copy not started - {reason}."
    if reason.startswith("another tagcomplete"):
        text += (" sd-webui-tagcomplete-neo is now part of sam-extra: remove that extension (or disable it in the"
                 " Extensions tab) and restart Forge so the two never run together. Its tac_* settings stay as they are.")
    _print(text)


def warn_missing_files_once(missing, tags_dir) -> None:
    for key, name in missing:
        token = f"missing:{key}:{name}"
        if token in _warned:
            continue
        _warned.add(token)
        _print(f"[sam-extra] Tag Autocomplete: {key} = '{name}' is not in {tags_dir} - pick another file in "
               "Settings -> Tag Autocomplete (the list stays empty until then).")


def note_html(state: dict, tags_dir, missing=()) -> str:
    """Settings → Tag Autocomplete 맨 위의 안내(저장되지 않는 OptionHTML)."""
    code = state.get("code")
    reason = html.escape(str(state.get("reason") or ""))
    where = html.escape(str(tags_dir))
    if code == "ok":
        text = ("<b>sam-extra 내장 Tag Autocomplete</b> — sd-webui-tagcomplete-neo(@8e0d81f, MIT)를 sam-extra 에 합친 것입니다. "
                "아래 설정은 따로 설치하던 확장과 같은 키(<code>tac_*</code>)라 값이 그대로 이어집니다. "
                f"태그 파일은 <code>{where}</code> 에서 읽습니다.")
    elif code == "standalone":
        text = (f"따로 설치된 tagcomplete 가 켜져 있어 <b>내장 Tag Autocomplete 는 쉬고 있습니다</b> ({reason}). "
                "아래 <code>tac_*</code> 설정은 그 확장의 것입니다. 내장을 쓰려면 그 확장을 지우거나 Extensions 탭에서 끄고 "
                "Forge 를 다시 시작하세요 — 둘이 함께 돌지는 않습니다.")
    elif code == "off":
        text = ("<b>내장 Tag Autocomplete 가 꺼져 있습니다</b>(아래 스위치). <code>tac_*</code> 설정 값은 config.json 에 그대로 "
                "남아 있어, 켜고 Forge 를 다시 시작하면 그대로 돌아옵니다.")
    else:
        text = (f"<b>내장 Tag Autocomplete 를 시작하지 못했습니다</b> ({reason}). "
                "콘솔의 <code>[sam-extra] Tag Autocomplete</code> 줄을 보세요.")
    if missing:
        items = ", ".join(f"<code>{html.escape(key)}</code> = <code>{html.escape(name)}</code>" for key, name in missing)
        text += f"<br>⚠ 고른 파일이 태그 폴더에 없습니다: {items} — 아래에서 있는 파일을 다시 고르세요(그 전까지 그 목록은 비어 있습니다)."
    return text


def register_settings() -> None:
    """``on_ui_settings`` — sam3_* 키(안내·스위치)만 등록한다. ``tac_*`` 는 헬퍼(상류 on_ui_settings)가 등록한다."""
    from modules import shared   # Forge only (called by Forge's settings callback)

    from sam3ext.tagcomplete import boot, paths

    option_html = getattr(shared, "OptionHTML", None)
    if option_html is not None:
        # file names refer to our tags/ only while the built-in copy runs (otherwise to the other copy's folder)
        missing = missing_tag_files(shared.opts, paths.TAGS_DIR) if boot.STATE.get("code") == "ok" else []
        note = option_html(note_html(boot.STATE, paths.TAGS_DIR, missing))
        note.section = SECTION
        shared.opts.add_option(OPT_NOTE, note)
    shared.opts.add_option(
        OPT_ENABLED,
        shared.OptionInfo(True, "내장 Tag Autocomplete 사용 (sam-extra)", section=SECTION).info(
            "끄면 다시 시작한 뒤부터 내장 태그 자동완성(추천 목록 · /tacapi 경로 · 사용 횟수 기록)이 돌지 않습니다. "
            "따로 설치한 tagcomplete 가 켜져 있으면 이 값과 관계없이 내장은 쉽니다."
        ).needs_restart(),
    )
