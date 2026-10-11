"""Built-in Tag Autocomplete: the Forge entry script (scripts/builtin_tagcomplete.py), the coexistence rule and its
notices (sam3ext/tagcomplete/coexist.py) and the loader's start-up information (sam3ext/tagcomplete/boot.py).

The entry script runs against Forge stand-ins with every folder in a temporary directory
(tests/_tagcomplete_support.sandboxed): nothing here writes into the extension's tags/ or a Forge install.

* Default: the helper starts — settings note and switch first in the "Tag Autocomplete" section, then the 57 tac_*;
  the 15 /tacapi/v1/* routes once per app (also when app start fires twice); temp files and tag_frequency.db in the
  sandbox. The loader's start-up information is the hidden, never saved setting ``sam3_tac_builtin_boot`` (it reaches
  the page in Forge's ``opts``): the entry script registers no ``app_started`` callback of its own, so nothing the
  loader needs waits for the end of Forge's start (before, a ``/sam3-tagcomplete/boot`` route added in app_started
  was not there yet for a page opened during Forge's last seconds of start-up, and the built-in never started on it).
* Another tagcomplete enabled (any folder with upstream's files, e.g. the old sd-webui-tagcomplete-neo): nothing
  starts; one console line and the note give the steps in order — stop Forge, copy the user's files with
  tools/migrate_tagcomplete.py (the exact command, with that folder), only then move the old folder aside or disable
  it — because that folder's tags/ holds what git does not track; the boot information says enabled: false so the
  page's loader does not add the built-in JS. sam-extra's own folder never counts.
* Switch off: nothing starts, the tac_* values stay untouched (not registered, config.json keeps them).
* A helper that fails to import does not stop the script; the boot information reports the failure.
* A tac_* file that is not in tags/ (the user's tac_tagFile names ...-20260907.csv, on disk is ...-v2.csv) is named
  on the console and in the note, with the migration command (with the old copy's folder when Forge still lists it).
"""

from __future__ import annotations

import html
import importlib.util
import json
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
HIDDEN = [None, "Hidden Options"]


def load_entry(box, generation=1):
    """Run scripts/builtin_tagcomplete.py as Forge does (a new module object each load); returns printed text."""
    def run():
        spec = importlib.util.spec_from_file_location(f"_tac_entry_{generation}", ENTRY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    _module, printed = tac.run_quietly(run)
    return printed


def fake_extension(root: Path, name: str, *, js=True, helper=True, enabled=True):
    folder = root / "forge" / "extensions" / name
    if js:
        (folder / "javascript").mkdir(parents=True, exist_ok=True)
        (folder / "javascript" / "tagAutocomplete.js").write_text("// upstream", encoding="utf-8")
    if helper:
        (folder / "scripts").mkdir(parents=True, exist_ok=True)
        (folder / "scripts" / "tag_autocomplete_helper.py").write_text("# upstream", encoding="utf-8")
    folder.mkdir(parents=True, exist_ok=True)
    return types.SimpleNamespace(name=name, path=str(folder), enabled=enabled)


def app_routes(app):
    return [(route.path, method) for route in app.routes for method in sorted(getattr(route, "methods", None) or [])]


def migration_command(source):
    from sam3ext.tagcomplete import coexist, paths

    command = coexist.migration_command(source)
    tool = paths.EXT_ROOT / "tools" / "migrate_tagcomplete.py"
    assert str(tool) in command and sys.executable in command and command.endswith(" --apply"), command
    return command


class EntryScriptTests(unittest.TestCase):
    def setUp(self):
        from sam3ext.tagcomplete import coexist

        coexist._warned.clear()
        self.addCleanup(coexist._warned.clear)

    def _settings(self, box):
        for callback in box.callbacks.named("on_ui_settings"):
            tac.run_quietly(callback)
        return box.shared.opts.added

    def _boot(self, box):
        """The hidden setting the page's loader reads (Forge puts every option's value into the page's opts)."""
        from sam3ext.tagcomplete import coexist

        (info,) = [info for key, info in box.shared.opts.added if key == coexist.OPT_BOOT]
        self.assertEqual(list(info.section), HIDDEN, "not on the Settings page (Forge skips section id None)")
        self.assertTrue(info.do_not_save, "never written to config.json")
        self.assertIsInstance(info.default, str)
        return json.loads(info.default)

    def test_default_starts_the_helper(self):
        from fastapi import FastAPI

        from sam3ext.tagcomplete import boot

        with tac.sandboxed(tags_files=("danbooru.csv", "extra-quality-tags.csv", "demo-chants.json")) as box:
            printed = load_entry(box)
            self.assertEqual(boot.STATE, {"enabled": True, "code": "ok", "reason": None, "source": None})
            self.assertIsNotNone(box.helper, "the helper module was imported")
            self.assertEqual(printed.count("[sam-extra]"), 0, printed)
            keys = [key for key, _ in self._settings(box)]
            self.assertEqual(keys[:3], ["sam3_tac_builtin_note", "sam3_builtin_tagcomplete_enabled", "sam3_tac_builtin_boot"])
            self.assertEqual(tuple(keys[3:]), tac.TAC_KEYS)
            for key, info in box.shared.opts.added:
                self.assertEqual(list(info.section), HIDDEN if key == "sam3_tac_builtin_boot" else tac.SECTION, key)
            note = box.shared.opts.added[0][1]
            self.assertTrue(note.do_not_save)
            self.assertIn("sam-extra 내장 Tag Autocomplete", note.default)
            self.assertNotIn("⚠", note.default)
            switch = box.shared.opts.added[1][1]
            self.assertIs(switch.default, True)
            self.assertIn(("needs_restart",), switch.flags)
            started = self._boot(box)
            self.assertEqual(started, boot.manifest())
            self.assertEqual((started["enabled"], started["reason"], len(started["scripts"])), (True, None, 15))

            app = FastAPI()
            for _ in range(2):   # app start can fire again on the same app
                for callback in box.callbacks.named("on_app_started"):
                    tac.run_quietly(callback, None, app)
            routes = app_routes(app)
            self.assertEqual([r for r in routes if r[0].startswith("/tacapi/")], list(tac.TACAPI_ROUTES))
            self.assertEqual([r for r in routes if r[0].startswith("/sam3-")], [], "no route of our own")

            written = sorted(p.relative_to(box.root).as_posix() for p in box.root.rglob("*") if p.is_file())
            for path in ("forge/tmp/tagAutocompletePath.txt", "forge/tmp/modelKeywordPath.txt", "tags/tag_frequency.db",
                         "tags/temp/wc.txt", "tags/temp/lora.txt", "tags/temp/known_lora_hashes.txt"):
                self.assertIn(path, written)
            self.assertEqual((box.forge / "tmp" / "tagAutocompletePath.txt").read_text(encoding="utf-8"),
                             box.tags.as_posix(), "the JS reads its tag base path from here")

    def test_the_loader_needs_nothing_from_app_started(self):
        """Regression: the start-up information was a route added in app_started, which Forge runs after the browser
        opens and after /internal/ping (the reload after Reload UI) answers; a page loaded in that window waited
        2.5 s, then gave up for good. Now it is a setting, there when the scripts load; app_started only carries the
        helper's own /tacapi routes (as upstream)."""
        with tac.sandboxed() as box:
            load_entry(box)
            app_started = [callback.__name__ for callback in box.callbacks.named("on_app_started")]
            self.assertEqual(app_started, ["_api_tac_once"], "only the upstream helper's /tacapi/v1/* routes")
            self._settings(box)
            self.assertTrue(self._boot(box)["enabled"])
        source = (ROOT / "javascript" / "tagcomplete_loader.js").read_text(encoding="utf-8")
        self.assertNotIn("fetch(", source)
        self.assertIn('"sam3_tac_builtin_boot"', source)
        self.assertIn("onOptionsAvailable", source)

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
            self.assertEqual(kinds.count("on_app_started"), 1)   # the helper's /tacapi
            self.assertEqual(boot.STATE["code"], "ok")

    def test_another_tagcomplete_makes_the_builtin_stand_down(self):
        from fastapi import FastAPI

        from sam3ext.tagcomplete import boot, paths

        with tac.sandboxed() as box:
            other = fake_extension(box.root, "sd-webui-tagcomplete-neo")
            own = types.SimpleNamespace(name="forge_sam3_extension", path=str(paths.EXT_ROOT), enabled=True)
            box.extensions.active = lambda: [own, other]
            printed = load_entry(box)
            self.assertEqual(boot.STATE["code"], "standalone")
            self.assertFalse(boot.STATE["enabled"])
            self.assertIn("sd-webui-tagcomplete-neo", boot.STATE["reason"])
            self.assertEqual(boot.STATE["source"], other.path)
            self.assertIsNone(box.helper, "the helper did not start")
            self.assertEqual(printed.count("[sam-extra] Tag Autocomplete: built-in copy not started"), 1, printed)
            self.assertFalse((box.forge / "tmp" / "tagAutocompletePath.txt").exists(), "the other copy's file is left alone")
            self.assertFalse((box.tags / "temp").exists())
            keys = [key for key, _ in self._settings(box)]
            self.assertEqual(keys, ["sam3_tac_builtin_note", "sam3_builtin_tagcomplete_enabled", "sam3_tac_builtin_boot"])
            self.assertIn("쉬고 있습니다", box.shared.opts.added[0][1].default)
            answer = self._boot(box)
            self.assertEqual(answer["enabled"], False)
            self.assertEqual(answer["scripts"], [])
            self.assertIn("another tagcomplete", answer["reason"])

            app = FastAPI()
            for callback in box.callbacks.named("on_app_started"):
                callback(None, app)
            self.assertEqual([path for path, _ in app_routes(app) if path.startswith(("/tacapi", "/sam3-"))], [])

            # Reload UI: the reason is not printed again
            box.callbacks.registered.clear()
            self.assertNotIn("not started", load_entry(box, 2))

    def test_the_standalone_notices_say_migrate_first_with_the_exact_command(self):
        """Regression: the console line and the note said "remove that extension (or disable it in the Extensions
        tab) and restart" — removing loses the user's untracked lists and tag_frequency.db; disabling makes the
        built-in create an empty tag_frequency.db first. Both now give the migration command (this Forge's python,
        the tool's and the detected folder's absolute paths) before the removal step."""
        from sam3ext.tagcomplete import paths

        with tac.sandboxed() as box:
            other = fake_extension(box.root, "sd-webui-tagcomplete-neo")
            box.extensions.active = lambda: [other]
            printed = load_entry(box)
            command = migration_command(other.path)
            self.assertIn(f"--src {other.path} --apply" if " " not in other.path else f'--src "{other.path}" --apply',
                          command)
            self.assertIn(command, printed)
            for earlier, later in (("1. stop Forge", "2. copy them over"), ("2. copy them over", command),
                                   (command, "3. then move that extension's folder out of extensions/"),
                                   ("3. then move", "4. start Forge")):
                self.assertLess(printed.index(earlier), printed.index(later), (earlier, later))
            self.assertIn("files git does not track", printed)
            self.assertNotIn("remove that extension", printed)
            self._settings(box)
            note = box.shared.opts.added[0][1].default
            self.assertIn(html.escape(command), note)
            self.assertLess(note.index("① Forge 를 끄고"), note.index(html.escape(command)))
            self.assertLess(note.index(html.escape(command)), note.index("③ 그다음 그 확장 폴더를"))
            self.assertIn("git 에 없어", note)
            self.assertTrue((paths.EXT_ROOT / "tools" / "migrate_tagcomplete.py").is_file(), "the named tool exists")

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
            self.assertEqual(coexist.tagcomplete_copies([own, unrelated, py_only, js_only], paths.EXT_ROOT),
                             [("tagcomplete-fork", py_only.path), ("a1111-sd-webui-tagcomplete", js_only.path)])
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
                self.assertEqual(keys, ["sam3_tac_builtin_note", "sam3_builtin_tagcomplete_enabled", "sam3_tac_builtin_boot"])
                self.assertIn("꺼져 있습니다", box.shared.opts.added[0][1].default)
                self.assertEqual(self._boot(box)["enabled"], False)
                self.assertFalse((box.forge / "tmp" / "tagAutocompletePath.txt").exists())
        for value in (True, "true", None, "unexpected"):
            with self.subTest(value=value), tac.sandboxed(tac.Opts(sam3_builtin_tagcomplete_enabled=value)) as box:
                load_entry(box)
                self.assertEqual(boot.STATE["code"], "ok")

    def test_a_failing_helper_does_not_stop_the_script(self):
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
            keys = [key for key, _ in self._settings(box)]
            self.assertIn("시작하지 못했습니다", box.shared.opts.added[0][1].default)
            self.assertEqual(keys[:3], ["sam3_tac_builtin_note", "sam3_builtin_tagcomplete_enabled", "sam3_tac_builtin_boot"])
            answer = self._boot(box)
            self.assertFalse(answer["enabled"])
            self.assertTrue(answer["reason"].startswith("failed to start"))

    def test_a_tag_file_missing_from_tags_is_reported(self):
        from sam3ext.tagcomplete import coexist

        opts = tac.Opts(tac_tagFile="KR_danbooru_tags_with_description-20260907.csv",
                        **{"tac_extra.extraFile": "danbooru_e621_merged.csv", "tac_chantFile": "demo-chants.json",
                           "tac_translation.translationFile": "None"})
        with tac.sandboxed(opts, tags_files=("danbooru_e621_merged.csv", "demo-chants.json")) as box:
            printed = load_entry(box)
            self.assertIn("tac_tagFile = 'KR_danbooru_tags_with_description-20260907.csv' is not in", printed)
            self.assertNotIn("tac_extra.extraFile", printed)
            self.assertNotIn("tac_chantFile", printed)
            # no old copy that Forge knows of: the command has a placeholder for the folder
            command = migration_command(None)
            self.assertIn(coexist.OLD_FOLDER_PLACEHOLDER, command)
            self.assertIn(command, printed)
            self.assertLess(printed.index("is not in"), printed.index(command))
            self._settings(box)
            note = box.shared.opts.added[0][1].default
            self.assertIn("KR_danbooru_tags_with_description-20260907.csv", note)
            self.assertIn("⚠", note)
            self.assertIn(html.escape(command), note)
            box.callbacks.registered.clear()
            self.assertNotIn("is not in", load_entry(box, 2), "printed once per process")

    def test_a_disabled_old_copy_is_named_in_the_missing_file_notice(self):
        """The old copy was disabled before its files were copied (what the old console line said): the built-in runs,
        the user's tag file is not in tags/ yet — the notice names that folder in the migration command."""
        from sam3ext.tagcomplete import boot

        opts = tac.Opts(tac_tagFile="KR_danbooru_tags_with_description-20260907-v2.csv")
        with tac.sandboxed(opts) as box:
            old = fake_extension(box.root, "sd-webui-tagcomplete-neo", enabled=False)
            box.extensions.extensions = [old]   # Forge lists it, active() does not
            printed = load_entry(box)
            self.assertEqual(boot.STATE["code"], "ok", "a disabled copy does not stop the built-in")
            self.assertEqual(boot.STATE["source"], old.path)
            command = migration_command(old.path)
            self.assertIn(command, printed)
            self._settings(box)
            self.assertIn(html.escape(command), box.shared.opts.added[0][1].default)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        from sam3ext.tagcomplete import boot

        saved = dict(boot.STATE)
        self.addCleanup(lambda: (boot.STATE.clear(), boot.STATE.update(saved)))

    def test_manifest_lists_the_15_files_in_forge_order(self):
        from sam3ext.tagcomplete import boot, paths

        boot.set_state("ok", None)
        body = json.loads(boot.manifest_json())
        self.assertEqual(body, boot.manifest())
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

    def test_disabled_lists_no_files(self):
        from sam3ext.tagcomplete import boot

        boot.set_state("off", "turned off in Settings -> Tag Autocomplete")
        self.assertEqual(json.loads(boot.manifest_json()),
                         {"enabled": False, "reason": "turned off in Settings -> Tag Autocomplete",
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
