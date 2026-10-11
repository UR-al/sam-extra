# sam-extra 자체 코드(GPL-3.0-only) — 상류 tagcomplete 코드가 아니다.
"""내장 Tag Autocomplete 의 Forge 진입점 — sd-webui-tagcomplete-neo(@ 8e0d81f, MIT)를 sam-extra 안에서 시작한다.

Forge 가 이 파일을 불러올 때마다(시작 · Reload UI):

1. 스위치가 켜져 있고 따로 설치된 tagcomplete 가 켜져 있지 않으면 상류 헬퍼를 시작한다(``boot.start_helper``) — 상류와 같은
   tac_* 설정 57개(``tac_tagFile`` 의 기본값만 ``danbooru_2026-10-01_pt20-ia-dd.csv``), ``/tacapi/v1/*`` 경로 15개, 임시 파일,
   사용 횟수 DB(``tags/tag_frequency.db``). 고른 태그 파일이 ``tags/`` 에 없으면 콘솔에 알리고 기본 목록으로 대신 추천한다.
   아니면 이유를 콘솔에 한 번 남기고 아무것도 걸지 않는다 — 로더도 JS 를 붙이지 않는다. 따로 설치된 사본이 이유면, 그 폴더를
   치우기 전에 ``tools/migrate_tagcomplete.py`` 로 사용자 파일을 옮기는 명령을 함께 남긴다.
2. Settings → Tag Autocomplete 에 sam3_* 두 줄(안내 · ``sam3_builtin_tagcomplete_enabled`` 스위치)을 tac_* 보다 먼저 올리고,
   로더(``javascript/tagcomplete_loader.js``)가 읽는 시작 정보를 숨은 설정 ``sam3_tac_builtin_boot`` 로 올린다 — 페이지의
   ``opts`` 에 실려 오므로 ``app_started`` 에 붙는 경로를 기다리지 않는다.

파일 이름은 ``negpip.py`` 보다 앞에 정렬되어야 한다(tests/test_integration_load_order.py: negpip.py 가 이 확장의 마지막 스크립트).
상류 헬퍼는 ``scripts/`` 에 두지 않는다 — Forge 가 ``scripts/`` 의 모든 .py 를 스크립트로 실행하기 때문이다.
"""

from modules import script_callbacks, shared

from sam3ext.tagcomplete import boot, coexist, paths


def _extensions(enabled_only: bool) -> list:
    try:
        from modules import extensions
    except ImportError:   # stand-ins without Forge's extension list (tests)
        return []
    try:
        return list(extensions.active() if enabled_only else getattr(extensions, "extensions", ()))
    except Exception as exc:  # noqa: BLE001
        print(f"[sam-extra] Tag Autocomplete: could not list the extensions ({type(exc).__name__}: {exc})")
        return []


def _start() -> None:
    if not coexist.enabled_setting(shared.opts):
        boot.set_state("off", coexist.REASON_OFF)
    else:
        standalone = coexist.standalone_copy(_extensions(enabled_only=True), paths.EXT_ROOT)
        if standalone is not None:
            boot.set_state("standalone", coexist.copy_reason(standalone), source=standalone[1])
        else:
            failure = boot.start_helper()
            # an old copy Forge still lists (disabled): its folder goes into the migration command of the notices
            old = coexist.tagcomplete_copies(_extensions(enabled_only=False), paths.EXT_ROOT)
            boot.set_state("failed" if failure else "ok", failure, source=old[0][1] if old else None)
    if boot.STATE["code"] == "ok":
        coexist.warn_missing_files_once(coexist.missing_tag_files(shared.opts, paths.TAGS_DIR), paths.TAGS_DIR,
                                        boot.STATE["source"], coexist.tag_file_fallback(shared.opts, paths.TAGS_DIR))
    elif boot.STATE["code"] != "failed":   # a failure already printed its traceback
        coexist.warn_once(boot.STATE["reason"], boot.STATE["source"])


# registered before the helper's own on_ui_settings, so the note and the switch come first in the section
script_callbacks.on_ui_settings(coexist.register_settings)
_start()
