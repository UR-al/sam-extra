"""Built-in Tag Autocomplete keeps sd-webui-tagcomplete-neo's settings — the user's 57 ``tac_*`` values in
config.json keep working after the separate extension is removed. One intended difference: the default of
``tac_tagFile`` is the newer Danbooru list ``danbooru_2026-10-01_pt20-ia-dd.csv`` (shipped in tags/) instead of
``danbooru.csv``; a saved value in config.json is used as before.

* Against the installed upstream copy (skipped on a PC without one, e.g. CI): the settings the vendored
  ``on_ui_settings`` registers — keys, order, defaults, labels, section, components and their arguments, refresh
  functions, ``.info()`` texts, ``needs_restart`` — equal those of the installed ``scripts/tag_autocomplete_helper.py``,
  both evaluated from the source's AST (tests/_tagcomplete_support.settings_from_source), except exactly the fields in
  ``INTENDED_DIFFERENCES``. The installed copy is sd-webui-tagcomplete-neo 8e0d81f; the main session moves it to
  <forge>/_removed_extensions, where this test still finds it.
* Always: the registrations are the 57 keys of ``TAC_KEYS`` (the development PC's config.json, 2026-10-11) in one
  section ``("tac", "Tag Autocomplete")``; the helper really registers them when sam-extra starts it
  (``boot.start_helper`` against Forge stand-ins); the default files exist in the shipped tags/ folder.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests import _tagcomplete_support as tac  # noqa: E402

DEFAULT_TAG_FILE = "danbooru_2026-10-01_pt20-ia-dd.csv"
# (key, field) -> (upstream, built-in): every way the registrations may differ from the installed upstream helper
INTENDED_DIFFERENCES = {("tac_tagFile", "default"): ("danbooru.csv", DEFAULT_TAG_FILE)}


def differences(theirs: list[dict], ours: list[dict]) -> dict:
    """``{(key, field): (upstream value, our value)}`` for registrations with the same keys in the same order."""
    found = {}
    for upstream_record, mine in zip(theirs, ours):
        for field in sorted(set(upstream_record) | set(mine)):
            if upstream_record.get(field) != mine.get(field):
                found[(mine["key"], field)] = (upstream_record.get(field), mine.get(field))
    return found


class UpstreamParityTests(unittest.TestCase):
    def test_vendored_registration_equals_the_installed_upstream_helper(self):
        upstream = tac.upstream_helper()
        if upstream is None:
            self.skipTest("no installed sd-webui-tagcomplete-neo on this PC (extensions/ or _removed_extensions/)")
        theirs = tac.settings_from_source(upstream.read_text(encoding="utf-8"))
        ours = tac.settings_from_source(tac.HELPER.read_text(encoding="utf-8"))
        self.assertEqual([r["key"] for r in ours], [r["key"] for r in theirs], "keys and their order")
        self.assertEqual(differences(theirs, ours), INTENDED_DIFFERENCES, "only the tac_tagFile default differs")
        self.assertEqual(len(theirs), 57)

    def test_the_comparison_sees_any_other_difference(self):
        """The helper above would report a label, section, flag or another default that drifted from upstream."""
        records = tac.settings_from_source(tac.HELPER.read_text(encoding="utf-8"))
        changed = [dict(r) for r in records]
        changed[0]["label"] = "Tag file"
        changed[5]["flags"] = []
        changed[-1]["section"] = ["x", "y"]
        found = differences(records, changed)
        self.assertEqual(set(found), {("tac_tagFile", "label"), (records[5]["key"], "flags"),
                                      (records[-1]["key"], "section")})


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = tac.settings_from_source(tac.HELPER.read_text(encoding="utf-8"))

    def test_the_57_known_keys_in_one_section(self):
        self.assertEqual(tuple(r["key"] for r in self.records), tac.TAC_KEYS)
        self.assertEqual(len(tac.TAC_KEYS), 57)
        for record in self.records:
            with self.subTest(key=record["key"]):
                self.assertEqual(record["section"], tac.SECTION)

    def test_a_few_defaults_and_labels(self):
        by_key = {r["key"]: r for r in self.records}
        self.assertEqual(by_key["tac_tagFile"]["default"], DEFAULT_TAG_FILE)
        self.assertEqual(by_key["tac_tagFile"]["label"], "Tag filename")
        self.assertEqual(by_key["tac_tagFile"]["component"], "Dropdown")
        self.assertEqual(by_key["tac_tagFile"]["component_args"], {"choices": tac.FIXED_CSV_CHOICES})
        self.assertEqual(by_key["tac_tagFile"]["refresh"], "update_tag_files")
        self.assertEqual(by_key["tac_chantFile"]["refresh"], "update_json_files")
        self.assertEqual(by_key["tac_refreshTempFiles"]["refresh"], "refresh_temp_files")
        self.assertEqual(by_key["tac_active"]["default"], True)
        self.assertEqual(by_key["tac_maxResults"]["default"], 5)
        self.assertEqual(by_key["tac_keymap"]["component"], "Code")
        self.assertIn(("needs_restart",), by_key["tac_delayTime"]["flags"])
        self.assertEqual(by_key["tac_modelSortOrder"]["component_args"],
                         {"choices": ["Name", "Date Modified (newest first)", "Date Modified (oldest first)"]})

    def test_default_files_ship_in_tags(self):
        by_key = {r["key"]: r for r in self.records}
        for key in ("tac_tagFile", "tac_extra.extraFile", "tac_chantFile"):
            with self.subTest(key=key):
                self.assertTrue((ROOT / "tags" / by_key[key]["default"]).is_file(), by_key[key]["default"])
        self.assertEqual(by_key["tac_translation.translationFile"]["default"], "None")

    def test_the_fallback_names_the_registered_default(self):
        """A tac_tagFile that is not in tags/ is read as this file on the page (coexist.tag_file_fallback)."""
        from sam3ext.tagcomplete import coexist

        by_key = {r["key"]: r for r in self.records}
        self.assertEqual(coexist.DEFAULT_TAG_FILE, by_key["tac_tagFile"]["default"])
        self.assertEqual(coexist.TAG_FILE_SETTING, "tac_tagFile")


class RuntimeRegistrationTests(unittest.TestCase):
    """What Forge gets when sam-extra starts the helper — the same 57 registrations, from the vendored module."""

    def test_the_started_helper_registers_the_same_settings(self):
        from sam3ext.tagcomplete import boot

        expected = tac.settings_from_source(tac.HELPER.read_text(encoding="utf-8"))
        with tac.sandboxed() as box:
            failure, _printed = tac.run_quietly(boot.start_helper)
            self.assertIsNone(failure)
            helper = box.helper
            helper.csv_files_withnone[:] = tac.FIXED_CSV_CHOICES   # the lambdas read these module lists
            helper.json_files_withnone[:] = tac.FIXED_JSON_CHOICES
            (settings,) = box.callbacks.named("on_ui_settings")
            settings()
            got = [tac.record(key, info) for key, info in box.shared.opts.added]
        for record in got:   # the sandboxed module's own functions
            if record["refresh"] is not None:
                self.assertIn(record["refresh"], ("update_tag_files", "update_json_files", "refresh_temp_files"))
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
