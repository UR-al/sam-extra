"""The vendored Tag Autocomplete helper (sam3ext/tagcomplete/tag_autocomplete_helper.py) behind its HTTP routes, started
the way sam-extra starts it, against Forge stand-ins in a temporary folder (tests/_tagcomplete_support.sandboxed).

* ``GET /tacapi/v1/wildcard-contents`` (sam-extra change): serves .txt files of the wildcard folders the helper lists
  in wc.txt / wce.txt — upstream's own request shapes keep working — and nothing else: no other folder, no "..", no
  absolute file name, no other suffix (upstream served any existing file on the PC to anyone who could reach Forge).
* The usage counts round-trip through ``tags/tag_frequency.db`` (schema version 1, as upstream).
* ``POST /tacapi/v1/refresh-temp-files`` rewrites lora.txt / wc.txt from the folders.
* ``_api_tac_once``: a second app start on the same app adds nothing.
"""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests import _tagcomplete_support as tac  # noqa: E402


class HelperRouteTests(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from sam3ext.tagcomplete import boot

        self.box = self.enterContext(tac.sandboxed())
        forge = self.box.forge
        # Forge's wildcard folder and an extension's wildcards/ (both exist before the helper is imported)
        (forge / "scripts" / "wildcards" / "hair").mkdir(parents=True)
        (forge / "scripts" / "wildcards" / "colors.txt").write_bytes(b"red\nblue\n# comment\n")
        (forge / "scripts" / "wildcards" / "hair" / "styles.txt").write_bytes(b"bob\n")
        (forge / "scripts" / "wildcards" / "notes.json").write_text("{}", encoding="utf-8")
        (forge / "extensions" / "dp" / "wildcards").mkdir(parents=True)
        (forge / "extensions" / "dp" / "wildcards" / "ext.txt").write_bytes(b"ext\n")
        (forge / "secret.txt").write_text("secret", encoding="utf-8")
        (forge / "models" / "Lora" / "style_a.safetensors").write_bytes(b"\x00" * 16)
        failure, _ = tac.run_quietly(boot.start_helper)
        self.assertIsNone(failure)
        self.app = FastAPI()
        (self.api,) = [cb for name, cb in self.box.callbacks.registered
                       if name == "on_app_started" and cb.__name__ == "_api_tac_once"]
        self.api(None, self.app)
        self.client = TestClient(self.app)

    def _wildcard(self, basepath, filename):
        return self.client.get("/tacapi/v1/wildcard-contents", params={"basepath": str(basepath), "filename": filename})

    def test_routes_once_per_app(self):
        self.api(None, self.app)
        routes = [(r.path, m) for r in self.app.routes if r.path.startswith("/tacapi/") for m in sorted(r.methods)]
        self.assertEqual(routes, list(tac.TACAPI_ROUTES))

    def test_listed_wildcard_files_are_served(self):
        forge = self.box.forge
        wildcards = forge / "scripts" / "wildcards"
        self.assertEqual(self._wildcard(wildcards, "colors.txt").text, "red\nblue\n# comment\n")
        self.assertEqual(self._wildcard(wildcards, "hair/styles.txt").text, "bob\n")
        self.assertEqual(self._wildcard(wildcards.as_posix(), "hair/styles.txt").status_code, 200)
        self.assertEqual(self._wildcard(forge / "extensions" / "dp" / "wildcards", "ext.txt").text, "ext\n")
        # the JS asks with the base paths of wce.txt — they are the ones the route accepts
        wce = (self.box.tags / "temp" / "wce.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn((forge / "extensions" / "dp" / "wildcards").as_posix(), wce)

    def test_anything_else_is_404(self):
        forge = self.box.forge
        wildcards = forge / "scripts" / "wildcards"
        cases = [
            (forge, "secret.txt"),                              # a folder that is not a wildcard folder
            (wildcards, "../../secret.txt"),                    # leaving the folder
            (wildcards, "..\\..\\secret.txt"),
            (wildcards, str(forge / "secret.txt")),             # an absolute file name replaces the base
            (wildcards, "notes.json"),                          # not a wildcard file
            (wildcards, "missing.txt"),
            (ROOT, "README.md"),
            ("", "colors.txt"),
        ]
        for basepath, filename in cases:
            with self.subTest(basepath=str(basepath), filename=filename):
                self.assertEqual(self._wildcard(basepath, filename).status_code, 404)

    def test_usage_counts_round_trip_through_the_database(self):
        for _ in range(3):
            self.assertEqual(self.client.post("/tacapi/v1/increase-use-count",
                                              params={"tagname": "1girl", "ttype": 0, "neg": False}).status_code, 200)
        answer = self.client.get("/tacapi/v1/get-use-count", params={"tagname": "1girl", "ttype": 0, "neg": False})
        self.assertEqual(answer.json()["result"][0], 3)
        listed = self.client.post("/tacapi/v1/get-use-count-list",
                                  json={"tagNames": ["1girl", "solo"], "tagTypes": [0, 0], "neg": False}).json()
        self.assertEqual([row[:3] for row in listed["result"]], [["1girl", 0, 3], ["solo", 0, 0]])
        database = self.box.tags / "tag_frequency.db"
        with sqlite3.connect(database) as connection:
            self.assertEqual(connection.execute("SELECT value FROM db_data WHERE key='version'").fetchone(), ("1",))
            self.assertEqual(connection.execute("SELECT count_pos FROM tag_frequency WHERE name='1girl'").fetchone(), (3,))
        connection.close()

    def test_refresh_rewrites_the_temp_lists(self):
        (self.box.forge / "models" / "Lora" / "style_b.safetensors").write_bytes(b"\x00" * 16)
        (self.box.forge / "scripts" / "wildcards" / "new.txt").write_text("x\n", encoding="utf-8")
        self.assertEqual(self.client.post("/tacapi/v1/refresh-temp-files").status_code, 200)
        lora = (self.box.tags / "temp" / "lora.txt").read_text(encoding="utf-8")
        self.assertIn('"style_a.safetensors"', lora)
        self.assertIn('"style_b.safetensors"', lora)
        wc = (self.box.tags / "temp" / "wc.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(wc[0], "scripts/wildcards")
        self.assertTrue(any(line.startswith('"new.txt"') for line in wc[1:]), wc)


if __name__ == "__main__":
    unittest.main()
