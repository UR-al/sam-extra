# sam-extra 자체 코드(GPL-3.0-only) — 상류 tagcomplete 코드가 아니다.
"""내장 Tag Autocomplete 와 따로 설치된 tagcomplete 가 함께 돌지 않게 한다 · 설정 스위치와 안내 (Forge 를 import 하지 않음).

둘이 함께 돌면: 같은 ``tac_*`` 설정을 두 번 등록하고, ``/tacapi/v1/*`` 경로가 두 번 붙으며(먼저 붙은 쪽이 받음),
``<forge>/tmp/tagAutocompletePath.txt`` 를 서로 덮어쓰고, 페이지에서는 두 번째 사본의 const/class 파일이
"Identifier has already been declared" 로 실패해 반만 초기화된다. 그래서 따로 설치된 사본이 켜져 있으면 내장이 쉰다
(파이썬: 헬퍼를 시작하지 않음 · 로더: JS 를 붙이지 않음 — ``javascript/tagcomplete_loader.js`` 도 페이지에서 한 번 더 본다).

판별: Forge 의 ``extensions.active()`` 가운데 이 확장이 아닌 것의 폴더에 상류의 파일(``javascript/tagAutocomplete.js`` 나
``scripts/tag_autocomplete_helper.py``)이 있으면 — 폴더 이름과 무관하게 a1111-sd-webui-tagcomplete 계열 모두. Forge 는 꺼진
확장의 스크립트·JS 를 싣지 않으므로 Extensions 탭에서 끈 사본은 세지 않는다.

안내(콘솔 · 설정 안내 줄)는 예전 사본을 치우기 **전에** ``tools/migrate_tagcomplete.py`` 로 사용자 파일을 옮기라고, 그 사본의
경로를 넣은 명령 그대로 알려 준다. 그 폴더의 ``tags/`` 에는 git 이 담지 않는 사용자 목록과 사용 횟수 DB 가 있다 — 먼저
지우면 사라지고, 먼저 끄고 재시작하면 내장이 빈 ``tag_frequency.db`` 를 만든다(옮기기 도구는 그 빈 DB 를 옆으로 치우고 옮긴다).

``tac_*`` 가 고른 파일이 tags/ 에 없으면 콘솔 · 안내 줄이 그 이름을 알린다. ``tac_tagFile`` 이면 기본 목록
``DEFAULT_TAG_FILE`` 로 대신 추천한다(``tag_file_fallback`` → 시작 정보 → 로더가 페이지의 ``opts`` 에서만 바꿈 —
config.json 은 그대로).
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

OPT_ENABLED = "sam3_builtin_tagcomplete_enabled"
OPT_NOTE = "sam3_tac_builtin_note"
# hidden, never saved: the loader's start-up information (boot.manifest_json()) — it reaches the page in Forge's `opts`
OPT_BOOT = "sam3_tac_builtin_boot"
# upstream's section (tag_autocomplete_helper.on_ui_settings TAC_SECTION) — the switch and the note sit on top of it
SECTION = ("tac", "Tag Autocomplete")
# Forge's own section for settings that are not on the Settings page (modules/shared_options.py "Hidden Options")
HIDDEN_SECTION = (None, "Hidden Options")
UPSTREAM_MARKERS = ("javascript/tagAutocomplete.js", "scripts/tag_autocomplete_helper.py")
# tac_* settings that name a file in tags/ ("None" = no file)
TAG_FILE_SETTINGS = ("tac_tagFile", "tac_extra.extraFile", "tac_translation.translationFile", "tac_chantFile")
# the built-in's default tac_tagFile (tag_autocomplete_helper.on_ui_settings — the parity test keeps the two equal); a
# tac_tagFile that is not in tags/ is read as this file on the page (tag_file_fallback), the setting is not changed
TAG_FILE_SETTING = "tac_tagFile"
DEFAULT_TAG_FILE = "danbooru_2026-10-01_pt20-ia-dd.csv"
MIGRATION_TOOL = Path("tools") / "migrate_tagcomplete.py"
OLD_FOLDER_PLACEHOLDER = "<old sd-webui-tagcomplete-neo folder>"

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


def tagcomplete_copies(extensions, ext_root) -> list[tuple[str, str]]:
    """``(이름, 폴더)`` — ``extensions``(Forge 의 ``Extension`` 들: ``name``·``path``) 가운데 이 확장이 아니면서 상류 파일이
    있는 것, 받은 순서대로."""
    found = []
    for ext in extensions or ():
        path = getattr(ext, "path", None)
        if not path or _same_folder(path, ext_root):
            continue
        for marker in UPSTREAM_MARKERS:
            try:
                present = (Path(path) / marker).is_file()
            except OSError:
                present = False
            if present:
                found.append((getattr(ext, "name", None) or os.path.basename(os.path.normpath(path)), str(path)))
                break
    return found


def standalone_copy(active_extensions, ext_root) -> tuple[str, str] | None:
    """켜진 확장 중 따로 설치된 tagcomplete 의 ``(이름, 폴더)``, 없으면 None."""
    copies = tagcomplete_copies(active_extensions, ext_root)
    return copies[0] if copies else None


def copy_reason(copy: tuple[str, str]) -> str:
    """``standalone_copy`` 의 ``(이름, 폴더)`` → 내장이 쉬는 이유(콘솔 · 안내 줄 · 로더)."""
    name, path = copy
    return f"another tagcomplete extension is enabled ({name}: {path})"


def standalone_reason(active_extensions, ext_root) -> str | None:
    """켜진 확장 중 따로 설치된 tagcomplete 가 있으면 그 이유, 없으면 None."""
    copy = standalone_copy(active_extensions, ext_root)
    return None if copy is None else copy_reason(copy)


def _quoted(value) -> str:
    text = str(value)
    return f'"{text}"' if (" " in text or "\t" in text) else text


def migration_command(source=None, *, python=None, ext_root=None) -> str:
    """예전 폴더의 사용자 파일을 이 확장의 ``tags/`` 로 옮기는 명령(Forge 를 끈 채로) — 이 Forge 의 파이썬, 절대 경로.
    ``source`` 가 없으면 자리 표시(``<old sd-webui-tagcomplete-neo folder>``)."""
    if ext_root is None:
        from sam3ext.tagcomplete import paths   # standard library only

        ext_root = paths.EXT_ROOT
    python = python or sys.executable or "python"
    tool = Path(ext_root) / MIGRATION_TOOL
    src = str(source) if source else OLD_FOLDER_PLACEHOLDER
    return f"{_quoted(python)} -I {_quoted(tool)} --src {_quoted(src)} --apply"


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


def tag_file_fallback(opts, tags_dir) -> dict | None:
    """``tac_tagFile`` 이 고른 파일이 tags/ 에 없고 기본 목록(``DEFAULT_TAG_FILE``)은 있으면
    ``{"key": "tac_tagFile", "missing": <고른 이름>, "use": DEFAULT_TAG_FILE}``, 아니면 None.

    로더(``javascript/tagcomplete_loader.js``)가 이것을 시작 정보로 받아, 페이지의 ``opts`` 에서만 그 값을 기본 목록으로
    바꿔 읽게 한다 — 상류는 없는 파일이면 추천 목록을 통째로 비운다. config.json 의 값과 Settings 화면은 그대로다
    ("None" 은 사용자가 고른 '목록 없음'이라 그대로 둔다). ``missing`` 은 설정 값 그대로다 — 로더가 페이지의 값과 똑같은지 본다."""
    for key, name in missing_tag_files(opts, tags_dir):
        if key != TAG_FILE_SETTING or name == DEFAULT_TAG_FILE:
            continue
        try:
            available = (Path(tags_dir) / DEFAULT_TAG_FILE).is_file()
        except (OSError, ValueError):
            available = False
        if available:
            return {"key": TAG_FILE_SETTING, "missing": getattr(opts, TAG_FILE_SETTING), "use": DEFAULT_TAG_FILE}
    return None


def _print(text: str) -> None:
    try:   # 레거시 콘솔 코드페이지(cp949 등)에서 경로의 글자로 시작이 죽지 않게
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"))


def standalone_steps(source) -> str:
    """콘솔: 따로 설치된 사본에서 내장으로 바꾸는 순서 — 옮기기가 치우기보다 먼저."""
    return ("sd-webui-tagcomplete-neo is now part of sam-extra. To switch to the built-in copy without losing your tag lists"
            " and usage counts (that folder's tags/ holds files git does not track):\n"
            "  1. stop Forge;\n"
            "  2. copy them over (run it once without --apply to see the plan first):\n"
            f"     {migration_command(source)}\n"
            "  3. then move that extension's folder out of extensions/ (safer than deleting it) or disable it in the"
            " Extensions tab;\n"
            "  4. start Forge. Its tac_* settings stay as they are.")


def warn_once(reason: str, source=None) -> None:
    """내장이 쉬는 이유를 프로세스당 한 번 — Reload UI 마다 콘솔을 채우지 않는다. ``source`` 는 따로 설치된 사본의 폴더."""
    if reason in _warned:
        return
    _warned.add(reason)
    text = f"[sam-extra] Tag Autocomplete: built-in copy not started - {reason}."
    if reason.startswith("another tagcomplete"):
        text += "\n" + standalone_steps(source)
    _print(text)


def _falls_back(fallback, key, name) -> bool:
    """``(key, name)`` of ``missing_tag_files`` (name stripped) is the one ``fallback`` replaces."""
    return bool(fallback) and fallback.get("key") == key and str(fallback.get("missing")).strip() == name


def warn_missing_files_once(missing, tags_dir, source=None, fallback=None) -> None:
    """고른 파일이 tags/ 에 없다는 줄(키마다 프로세스당 한 번)과, 새로 알린 것이 있으면 옮기기 명령 한 줄. ``source`` 는
    Forge 가 아는 예전 사본(꺼진 것)의 폴더 — 없으면 명령에 자리 표시. ``fallback`` 은 ``tag_file_fallback`` 의 값 — 그
    키의 줄은 목록이 비는 대신 기본 목록으로 추천한다고 알린다."""
    printed = False
    for key, name in missing:
        token = f"missing:{key}:{name}"
        if token in _warned:
            continue
        _warned.add(token)
        printed = True
        if _falls_back(fallback, key, name):
            _print(f"[sam-extra] Tag Autocomplete: {key} = '{name}' is not in {tags_dir} - suggestions use the default"
                   f" list '{fallback['use']}' instead until you pick another file in Settings -> Tag Autocomplete"
                   " (the setting itself is not changed).")
            continue
        _print(f"[sam-extra] Tag Autocomplete: {key} = '{name}' is not in {tags_dir} - pick another file in "
               "Settings -> Tag Autocomplete (the list stays empty until then).")
    if printed:
        _print("[sam-extra] Tag Autocomplete: if your lists were in an old sd-webui-tagcomplete(-neo) folder, stop Forge and "
               f"copy them over first (without --apply it only shows the plan):\n  {migration_command(source)}")


def _migration_html(source) -> str:
    return f"<code>{html.escape(migration_command(source))}</code>"


def note_html(state: dict, tags_dir, missing=(), fallback=None) -> str:
    """Settings → Tag Autocomplete 맨 위의 안내(저장되지 않는 OptionHTML). ``fallback`` 은 ``tag_file_fallback`` 의 값."""
    code = state.get("code")
    reason = html.escape(str(state.get("reason") or ""))
    where = html.escape(str(tags_dir))
    source = state.get("source")
    if code == "ok":
        text = ("<b>sam-extra 내장 Tag Autocomplete</b> — sd-webui-tagcomplete-neo(@8e0d81f, MIT)를 sam-extra 에 합친 것입니다. "
                "아래 설정은 따로 설치하던 확장과 같은 키(<code>tac_*</code>)라 값이 그대로 이어집니다. "
                f"태그 파일은 <code>{where}</code> 에서 읽습니다.")
    elif code == "standalone":
        text = (f"따로 설치된 tagcomplete 가 켜져 있어 <b>내장 Tag Autocomplete 는 쉬고 있습니다</b> ({reason}). "
                "아래 <code>tac_*</code> 설정은 그 확장의 것입니다. 내장으로 바꾸려면 — 그 폴더의 <code>tags/</code> 에 있는 "
                "사용자 목록·사용 횟수는 git 에 없어, 폴더를 먼저 지우면 사라집니다: "
                "① Forge 를 끄고 ② 먼저 옮기세요(<code>--apply</code> 없이 돌리면 옮길 목록만 보여 줍니다) "
                f"{_migration_html(source)} "
                "③ 그다음 그 확장 폴더를 extensions 밖으로 옮기거나(지우는 것보다 안전) Extensions 탭에서 끄고 "
                "④ Forge 를 다시 시작하세요 — 둘이 함께 돌지는 않습니다.")
    elif code == "off":
        text = ("<b>내장 Tag Autocomplete 가 꺼져 있습니다</b>(아래 스위치). <code>tac_*</code> 설정 값은 config.json 에 그대로 "
                "남아 있어, 켜고 Forge 를 다시 시작하면 그대로 돌아옵니다.")
    else:
        text = (f"<b>내장 Tag Autocomplete 를 시작하지 못했습니다</b> ({reason}). "
                "콘솔의 <code>[sam-extra] Tag Autocomplete</code> 줄을 보세요.")
    if missing:
        items = ", ".join(f"<code>{html.escape(key)}</code> = <code>{html.escape(name)}</code>" for key, name in missing)
        empty = [key for key, name in missing if not _falls_back(fallback, key, name)]
        text += f"<br>⚠ 고른 파일이 태그 폴더에 없습니다: {items} — 아래에서 있는 파일을 다시 고르세요"
        if len(empty) < len(missing):
            text += (f". 그 전까지 <code>{html.escape(fallback['key'])}</code> 는 기본 목록 "
                     f"<code>{html.escape(fallback['use'])}</code> 로 추천합니다(설정 값은 바꾸지 않음)")
            if empty:
                text += f", 나머지({', '.join(f'<code>{html.escape(key)}</code>' for key in empty)})는 비어 있습니다"
            text += "."
        else:
            text += "(그 전까지 그 목록은 비어 있습니다)."
        text += f" 예전 tagcomplete 폴더에 있던 목록이면 Forge 를 끄고 먼저 옮기세요: {_migration_html(source)}"
    return text


def boot_option(shared, fallback=None):
    """숨은 설정 ``sam3_tac_builtin_boot`` — 값은 ``boot.manifest_json(fallback)``. Settings 화면에 없고(Forge 의 "Hidden
    Options" 섹션) config.json 에도 저장되지 않으며(``do_not_save``), 페이지의 ``opts`` 로만 로더에 간다."""
    from sam3ext.tagcomplete import boot

    info = shared.OptionInfo(boot.manifest_json(fallback), "sam-extra built-in Tag Autocomplete start-up (not saved)",
                             section=HIDDEN_SECTION)
    info.do_not_save = True
    return info


def register_settings() -> None:
    """``on_ui_settings`` — sam3_* 키(안내·스위치·숨은 시작 정보)만 등록한다. ``tac_*`` 는 헬퍼(상류 on_ui_settings)가 등록한다."""
    from modules import shared   # Forge only (called by Forge's settings callback)

    from sam3ext.tagcomplete import boot, paths

    # file names refer to our tags/ only while the built-in copy runs (otherwise to the other copy's folder)
    running = boot.STATE.get("code") == "ok"
    fallback = tag_file_fallback(shared.opts, paths.TAGS_DIR) if running else None
    option_html = getattr(shared, "OptionHTML", None)
    if option_html is not None:
        missing = missing_tag_files(shared.opts, paths.TAGS_DIR) if running else []
        note = option_html(note_html(boot.STATE, paths.TAGS_DIR, missing, fallback))
        note.section = SECTION
        shared.opts.add_option(OPT_NOTE, note)
    shared.opts.add_option(
        OPT_ENABLED,
        shared.OptionInfo(True, "내장 Tag Autocomplete 사용 (sam-extra)", section=SECTION).info(
            "끄면 다시 시작한 뒤부터 내장 태그 자동완성(추천 목록 · /tacapi 경로 · 사용 횟수 기록)이 돌지 않습니다. "
            "따로 설치한 tagcomplete 가 켜져 있으면 이 값과 관계없이 내장은 쉽니다."
        ).needs_restart(),
    )
    shared.opts.add_option(OPT_BOOT, boot_option(shared, fallback))
