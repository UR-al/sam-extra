"""Built-in Tag Autocomplete: the Forge entry script (scripts/builtin_tagcomplete.py), the coexistence rule
(sam3ext/tagcomplete/coexist.py) and the loader's boot route (sam3ext/tagcomplete/boot.py).

The entry script runs against Forge stand-ins with every folder in a temporary directory
(tests/_tagcomplete_support.sandboxed): nothing here writes into the extension's tags/ or a Forge install.

* Default: the helper starts — settings note and switch first in the "Tag Autocomplete" section, then the 57 tac_*;
  the boot route and the 15 /tacapi/v1/* routes once per app (also when app start fires twice); temp files and
  tag_frequency.db in the sandbox.
* Another tagcomplete enabled (any folder with upstream's files, e.g. the old sd-webui-tagcomplete-neo): nothing
  starts, one console line, the note explains it; the boot route still answers (enabled: false) so the page's loader
  does not add the built-in JS. sam-extra's own folder never counts.
* Switch off: nothing starts, the tac_* values stay untouched (not registered, config.json keeps them).
* A helper that fails to import does not stop the script; the boot route reports the failure.
* A tac_* file that is not in tags/ (the user's tac_tagFile names ...-20260907.csv, on disk is ...-v2.csv) is named
  on the console and in the note.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests import _tagcomplete_support as tac  # noqa: E402

ENTRY = ROOT / "scripts" / "builtin_tagcomplete.py"


def load_entry(box, generation=1):
    """Run scripts/builtin_tagcomplete.py as Forge does (a new module object each load); returns printed text."""
    def run():
        spec = importlib.util.spec_from_file_location(f"_tac_entry_{generation}", ENTRY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    _module, printed = tac.run_quietly(run)
    return printed


def fake_extension(root: Path, name: str, *, js=True, helper=True):
    folder = root / "forge" / "extensions" / name
    if js:
        (folder / "javascript").mkdir(parents=True, exist_ok=True)
        (folder / "javascript" / "tagAutocomplete.js").write_text("// upstream", encoding="utf-8")
    if helper:
        (folder / "scripts").mkdir(parents=True, exist_ok=True)
        (folder / "scripts" / "tag_autocomplete_helper.py").write_text("# upstream", encoding="utf-8")
    folder.mkdir(parents=True, exist_ok=True)
    return types.SimpleNamespace(name=name, path=str(folder), enabled=True)


def app_routes(app):
    return [(route.path, method) for route in app.routes for method in sorted(getattr(route, "methods", None) or [])]


class EntryScriptTests(unittest.TestCase):
    def setUp(self):
        from sam3ext.tagcomplete import coexist

        coexist._warned.clear()
        self.addCleanup(coexist._warned.clear)

    def _settings(self, box):
        for callback in box.callbacks.named("on_ui_settings"):
            tac.run_quietly(callback)
        return box.shared.opts.added

    def test_default_starts_the_helper(self):
        from fastapi import FastAPI

        from sam3ext.tagcomplete import boot

        with tac.sandboxed(tags_files=("danbooru.csv", "extra-quality-tags.csv", "demo-chants.json")) as box:
            printed = load_entry(box)
            self.assertEqual(boot.STATE, {"enabled": True, "code": "ok", "reason": None})
            self.assertIsNotNone(box.helper, "the helper module was imported")
            self.assertEqual(printed.count("[sam-extra]"), 0, printed)
            keys = [key for key, _ in self._settings(box)]
            self.assertEqual(keys[:2], ["sam3_tac_builtin_note", "sam3_builtin_tagcomplete_enabled"])
            self.assertEqual(tuple(keys[2:]), tac.TAC_KEYS)
            for _key, info in box.shared.opts.added:
                self.assertEqual(list(info.section), tac.SECTION)
            note = box.shared.opts.added[0][1]
            self.assertTrue(note.do_not_save)
            self.assertIn("sam-extra 내장 Tag Autocomplete", note.default)
            self.assertNotIn("⚠", note.default)
            switch = box.shared.opts.added[1][1]
            self.assertIs(switch.default, True)
            self.assertIn(("needs_restart",), switch.flags)

            app = FastAPI()
            for _ in range(2):   # app start can fire again on the same app
                for callback in box.callbacks.named("on_app_started"):
                    tac.run_quietly(callback, None, app)
            routes = app_routes(app)
            self.assertEqual([r for r in routes if r[0].startswith("/tacapi/")], list(tac.TACAPI_ROUTES))
            self.assertEqual(routes.count((boot.BOOT_PATH, "GET")), 1)

            written = sorted(p.relative_to(box.root).as_posix() for p in box.root.rglob("*") if p.is_file())
            for path in ("forge/tmp/tagAutocompletePath.txt", "forge/tmp/modelKeywordPath.txt", "tags/tag_frequency.db",
                         "tags/temp/wc.txt", "tags/temp/lora.txt", "tags/temp/known_lora_hashes.txt"):
                self.assertIn(path, written)
            self.assertEqual((box.forge / "tmp" / "tagAutocompletePath.txt").read_text(encoding="utf-8"),
                             box.tags.as_posix(), "the JS reads its tag base path from here")

    def test_reload_starts_a_fresh_helper_and_registers_its_callbacks_again(self):
        from sam3ext.tagcomplete import boot

        with tac.sandboxed() as box:
            load_entry(box, 1)
            first = box.helper
            box.callbacks.registered.clear()   # Forge clears the callbacks before it loads the scripts again
            load_entry(box, 2)
            self.assertIsNot(box.helper, first)
            kinds = [name for name, _ in box.callbacks.registered]
            self.assertEqual(kinds.count("on_ui_settings"), 2)   # ours + the helper's
            self.assertEqual(kinds.count("on_app_started"), 2)   # boot route + /tacapi
            self.assertEqual(boot.STATE["code"], "ok")

    def test_another_tagcomplete_makes_the_builtin_stand_down(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from sam3ext.tagcomplete import boot, paths

        with tac.sandboxed() as box:
            other = fake_extension(box.root, "sd-webui-tagcomplete-neo")
            own = types.SimpleNamespace(name="forge_sam3_extension", path=str(paths.EXT_ROOT), enabled=True)
            box.extensions.active = lambda: [own, other]
            printed = load_entry(box)
            self.assertEqual(boot.STATE["code"], "standalone")
            self.assertFalse(boot.STATE["enabled"])
            self.assertIn("sd-webui-tagcomplete-neo", boot.STATE["reason"])
            self.assertIsNone(box.helper, "the helper did not start")
            self.assertEqual(printed.count("[sam-extra] Tag Autocomplete: built-in copy not started"), 1, printed)
            self.assertIn("remove that extension", printed)
            self.assertFalse((box.forge / "tmp" / "tagAutocompletePath.txt").exists(), "the other copy's file is left alone")
            self.assertFalse((box.tags / "temp").exists())
            keys = [key for key, _ in self._settings(box)]
            self.assertEqual(keys, ["sam3_tac_builtin_note", "sam3_builtin_tagcomplete_enabled"])
            self.assertIn("쉬고 있습니다", box.shared.opts.added[0][1].default)

            app = FastAPI()
            for callback in box.callbacks.named("on_app_started"):
                callback(None, app)
            self.assertEqual([path for path, _ in app_routes(app) if path.startswith(("/tacapi", "/sam3-"))],
                             [boot.BOOT_PATH])
            answer = TestClient(app).get(boot.BOOT_PATH, headers={"X-SAM3-Notebook": "1"}).json()
            self.assertEqual(answer["enabled"], False)
            self.assertEqual(answer["scripts"], [])
            self.assertIn("another tagcomplete", answer["reason"])

            # Reload UI: the reason is not printed again
            box.callbacks.registered.clear()
            self.assertNotIn("not started", load_entry(box, 2))

    def test_either_upstream_file_marks_a_copy_and_our_folder_never_does(self):
        from sam3ext.tagcomplete import coexist, paths

        with tac.sandboxed() as box:
            js_only = fake_extension(box.root, "a1111-sd-webui-tagcomplete", helper=False)
            py_only = fake_extension(box.root, "tagcomplete-fork", js=False)
            unrelated = fake_extension(box.root, "sd-dynamic-prompts", js=False, helper=False)
            own = types.SimpleNamespace(name="forge_sam3_extension", path=str(paths.EXT_ROOT))
            self.assertIsNone(coexist.standalone_reason([own, unrelated], paths.EXT_ROOT))
            self.assertIn("a1111-sd-webui-tagcomplete", coexist.standalone_reason([own, js_only], paths.EXT_ROOT))
            self.assertIn("tagcomplete-fork", coexist.standalone_reason([py_only], paths.EXT_ROOT))
            # our package has the upstream JS too, but below sam3ext/ — and the folder itself is skipped
            self.assertTrue((paths.JS_DIR / "tagAutocomplete.js").is_file())
            spelled = types.SimpleNamespace(name="x", path=str(paths.EXT_ROOT) + os.sep)
            self.assertIsNone(coexist.standalone_reason([spelled], paths.EXT_ROOT))
            self.assertIsNone(coexist.standalone_reason(None, paths.EXT_ROOT))

    def test_switch_off_starts_nothing(self):
        from sam3ext.tagcomplete import boot

        for value in (False, "false", "Off", 0):
            with self.subTest(value=value), tac.sandboxed(tac.Opts(sam3_builtin_tagcomplete_enabled=value)) as box:
                from sam3ext.tagcomplete import coexist

                coexist._warned.clear()
                printed = load_entry(box)
                self.assertEqual(boot.STATE["code"], "off")
                self.assertIsNone(box.helper)
                self.assertIn("turned off in Settings -> Tag Autocomplete", printed)
                keys = [key for key, _ in self._settings(box)]
                self.assertEqual(keys, ["sam3_tac_builtin_note", "sam3_builtin_tagcomplete_enabled"])
                self.assertIn("꺼져 있습니다", box.shared.opts.added[0][1].default)
                self.assertFalse((box.forge / "tmp" / "tagAutocompletePath.txt").exists())
        for value in (True, "true", None, "unexpected"):
            with self.subTest(value=value), tac.sandboxed(tac.Opts(sam3_builtin_tagcomplete_enabled=value)) as box:
                load_entry(box)
                self.assertEqual(boot.STATE["code"], "ok")

    def test_a_failing_helper_does_not_stop_the_script(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from sam3ext.tagcomplete import boot

        with tac.sandboxed() as box:
            del sys.modules["modules.paths"]   # the helper's shared_paths cannot import it
            delattr(sys.modules["modules"], "paths")
            load_entry(box)
            self.assertEqual(boot.STATE["code"], "failed")
            self.assertTrue(boot.STATE["reason"].startswith("failed to start: "), boot.STATE)
            self.assertEqual(sys.modules["modules.errors"].reported,
                             ["[sam-extra] Tag Autocomplete: the built-in helper failed to start"])
            self.assertIsNone(box.helper)
            app = FastAPI()
            for callback in box.callbacks.named("on_app_started"):
                callback(None, app)
            answer = TestClient(app).get(boot.BOOT_PATH, headers={"X-SAM3-Notebook": "1"}).json()
            self.assertFalse(answer["enabled"])
            self.assertTrue(answer["reason"].startswith("failed to start"))
            keys = [key for key, _ in self._settings(box)]
            self.assertIn("시작하지 못했습니다", box.shared.opts.added[0][1].default)
            self.assertEqual(keys[:2], ["sam3_tac_builtin_note", "sam3_builtin_tagcomplete_enabled"])

    def test_a_tag_file_missing_from_tags_is_reported(self):
        opts = tac.Opts(tac_tagFile="KR_danbooru_tags_with_description-20260907.csv",
                        **{"tac_extra.extraFile": "danbooru_e621_merged.csv", "tac_chantFile": "demo-chants.json",
                           "tac_translation.translationFile": "None"})
        with tac.sandboxed(opts, tags_files=("danbooru_e621_merged.csv", "demo-chants.json")) as box:
            printed = load_entry(box)
            self.assertIn("tac_tagFile = 'KR_danbooru_tags_with_description-20260907.csv' is not in", printed)
            self.assertNotIn("tac_extra.extraFile", printed)
            self.assertNotIn("tac_chantFile", printed)
            self._settings(box)
            note = box.shared.opts.added[0][1].default
            self.assertIn("KR_danbooru_tags_with_description-20260907.csv", note)
            self.assertIn("⚠", note)
            box.callbacks.registered.clear()
            self.assertNotIn("is not in", load_entry(box, 2), "printed once per process")


class BootRouteTests(unittest.TestCase):
    def setUp(self):
        from sam3ext.tagcomplete import boot

        saved = dict(boot.STATE)
        self.addCleanup(lambda: (boot.STATE.clear(), boot.STATE.update(saved)))

    def _client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from sam3ext.tagcomplete import boot

        app = FastAPI()
        self.assertTrue(boot.register_route(None, app))
        self.assertFalse(boot.register_route(None, app), "once per app")
        return TestClient(app)

    def test_manifest_lists_the_15_files_in_forge_order(self):
        from sam3ext.tagcomplete import boot, paths

        boot.set_state("ok", None)
        response = self._client().get(boot.BOOT_PATH, headers={"X-SAM3-Notebook": "1"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("no-store", response.headers["cache-control"])
        body = response.json()
        self.assertEqual(body["enabled"], True)
        self.assertIsNone(body["reason"])
        self.assertEqual(body["upstream"], "eduardoabreu81/sd-webui-tagcomplete-neo@8e0d81f")
        self.assertEqual(len(body["scripts"]), 15)
        names = []
        for url in body["scripts"]:
            self.assertTrue(url.startswith("file="), url)
            path, _, mtime = url[len("file="):].rpartition("?")
            float(mtime)
            self.assertTrue(os.path.samefile(path if os.path.isabs(path) else os.path.join(os.getcwd(), path),
                                             paths.JS_DIR / os.path.basename(path)))
            names.append(os.path.basename(path))
        self.assertEqual(tuple(names), tac.JS_ORDER)

    def test_the_route_needs_the_same_origin_header(self):
        from sam3ext.tagcomplete import boot

        boot.set_state("ok", None)
        self.assertEqual(self._client().get(boot.BOOT_PATH).status_code, 403)

    def test_disabled_lists_no_files(self):
        from sam3ext.tagcomplete import boot

        boot.set_state("off", "turned off in Settings -> Tag Autocomplete")
        body = self._client().get(boot.BOOT_PATH, headers={"X-SAM3-Notebook": "1"}).json()
        self.assertEqual(body, {"enabled": False, "reason": "turned off in Settings -> Tag Autocomplete",
                                "upstream": "eduardoabreu81/sd-webui-tagcomplete-neo@8e0d81f", "scripts": []})

    def test_forge_webpath_is_used_when_present(self):
        from sam3ext.tagcomplete import boot

        module = types.ModuleType("modules.ui_gradio_extensions")
        module.webpath = lambda fn: "file=FORGE/" + os.path.basename(fn)
        package = types.ModuleType("modules")
        package.__path__ = []
        package.ui_gradio_extensions = module
        saved = {name: sys.modules.get(name) for name in ("modules", "modules.ui_gradio_extensions")}
        sys.modules.update({"modules": package, "modules.ui_gradio_extensions": module})
        try:
            boot.set_state("ok", None)
            self.assertEqual(boot.manifest()["scripts"][0], "file=FORGE/__globals.js")
        finally:
            for name, value in saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


class TipoTagIndexTests(unittest.TestCase):
    def test_tipo_reads_the_builtin_tags_folder(self):
        from sam3ext.tagcomplete import paths
        from sam3ext.tipo import prompt as tp

        self.assertEqual(tp.default_tag_csv_paths(), [paths.TAGS_DIR / "danbooru_2025.csv", paths.TAGS_DIR / "danbooru.csv"])
        for path in tp.default_tag_csv_paths():
            self.assertTrue(path.is_file(), path)   # both ship in git
        source = (ROOT / "sam3ext" / "ui_tipo.py").read_text(encoding="utf-8")
        self.assertNotIn("sd-webui-tagcomplete-neo", source)


if __name__ == "__main__":
    unittest.main()
