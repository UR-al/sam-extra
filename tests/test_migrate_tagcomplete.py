"""tools/migrate_tagcomplete.py — copies a user's tag lists, tag_frequency.db and the model-keyword hash cache from an
old sd-webui-tagcomplete(-neo) folder into sam-extra's tags/, on temporary folders only.

Dry run by default; never overwrites (identical = done, different = conflict, exit 2); never changes the source;
SHA-256 checked for every copy, the database opened read-only afterwards (integrity, schema, rows); temp/ files the
helper rebuilds are not copied; a progress JSONL lets a second run skip finished files; preconditions fail with
exit 1 before anything is written.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# by path: a "tools" package elsewhere on sys.path must not shadow the extension's tools/ folder
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("sam3_migrate_tagcomplete", ROOT / "tools" / "migrate_tagcomplete.py")
migrate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = migrate   # dataclasses look their module up there
_spec.loader.exec_module(migrate)


def make_database(path: Path, rows) -> None:
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("CREATE TABLE db_data (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute("CREATE TABLE tag_frequency (name TEXT NOT NULL, type INT NOT NULL, count_pos INT, "
                           "count_neg INT, last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (name, type))")
        connection.execute("INSERT INTO db_data VALUES ('version', 1)")
        connection.executemany("INSERT INTO tag_frequency (name, type, count_pos, count_neg) VALUES (?, ?, ?, ?)", rows)
    connection.close()


def tree_hashes(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(folder.rglob("*")) if p.is_file()}


class MigrateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sam3_migrate_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        old = self.tmp / "extensions" / "sd-webui-tagcomplete-neo"
        (old / "scripts").mkdir(parents=True)
        (old / "scripts" / "tag_autocomplete_helper.py").write_text("# upstream", encoding="utf-8")
        tags = old / "tags"
        (tags / "temp").mkdir(parents=True)
        (tags / "danbooru.csv").write_bytes(b"1girl,0,100,\n")                       # shipped, same bytes
        (tags / "extra-quality-tags.csv").write_bytes(b"masterpiece,5,0,\n")          # shipped, user edited it
        (tags / "KR_tags-v2.csv").write_bytes("﻿소녀,0,1,\r\n".encode("utf-8"))  # the user's own list
        (tags / "e621.csv").write_bytes(b"canine,5,1,\n")                             # upstream list sam-extra does not ship
        (tags / "noob-chants.json").write_bytes(b"[]")
        make_database(tags / "tag_frequency.db", [("1girl", 0, 5, 0), ("solo", 0, 2, 1)])
        (tags / "temp" / "known_lora_hashes.txt").write_bytes(b"a.safetensors,1234abcd,1.0\n")
        (tags / "temp" / "lora.txt").write_bytes(b'"a.safetensors","a",,"a"\n')       # rebuilt at every start
        (tags / "temp" / "wc.txt").write_bytes(b"scripts/wildcards\n")
        self.old, self.src_tags = old, tags
        self.dst = self.tmp / "forge_sam3_extension" / "tags"
        self.dst.mkdir(parents=True)
        (self.dst / "danbooru.csv").write_bytes(b"1girl,0,100,\n")
        (self.dst / "extra-quality-tags.csv").write_bytes(b"masterpiece,5,0,\nbest quality,5,0,\n")
        self.progress = self.tmp / "scratch" / "migration.jsonl"

    def run_tool(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = migrate.main(["--src", str(self.old), "--dst", str(self.dst), "--progress", str(self.progress), *extra])
        return code, out.getvalue()

    def test_dry_run_writes_nothing_and_prints_the_plan(self):
        before = tree_hashes(self.dst)
        source = tree_hashes(self.old)
        code, out = self.run_tool()
        self.assertEqual(code, 2, "the edited shipped file is a conflict")
        self.assertEqual(tree_hashes(self.dst), before)
        self.assertEqual(tree_hashes(self.old), source)
        self.assertFalse(self.progress.exists())
        self.assertIn("DRY RUN", out)
        self.assertIn(f"source: {self.src_tags}", out)
        self.assertIn(f"target: {self.dst}", out)
        for line in ("copy      KR_tags-v2.csv", "copy      e621.csv", "copy      noob-chants.json",
                     "copy      tag_frequency.db", "copy      temp/known_lora_hashes.txt",
                     "identical danbooru.csv", "conflict  extra-quality-tags.csv"):
            self.assertIn(line, out)
        self.assertIn("[schema 1, 2 rows]", out)
        self.assertIn(str(self.dst / "KR_tags-v2.csv"), out, "absolute target paths")
        self.assertIn("not copied (rebuilt at every start): temp/lora.txt, temp/wc.txt", out)
        self.assertIn("7 files: conflict 1, copy 5, identical 1; to copy", out)

    def test_apply_copies_verifies_and_never_overwrites(self):
        source = tree_hashes(self.old)
        edited = (self.dst / "extra-quality-tags.csv").read_bytes()
        code, out = self.run_tool("--apply")
        self.assertEqual(code, 2)
        self.assertEqual(tree_hashes(self.old), source, "the source is read only")
        for name in ("KR_tags-v2.csv", "e621.csv", "noob-chants.json", "tag_frequency.db", "temp/known_lora_hashes.txt"):
            with self.subTest(name=name):
                self.assertEqual((self.dst / name).read_bytes(), (self.src_tags / name).read_bytes())
                self.assertEqual((self.dst / name).stat().st_mtime_ns // 10**6,
                                 (self.src_tags / name).stat().st_mtime_ns // 10**6, "mtime kept")
                self.assertIn(f"copied    {name} -> {self.dst / name}  (sha256 verified", out)
        self.assertEqual((self.dst / "extra-quality-tags.csv").read_bytes(), edited, "the conflict is left alone")
        self.assertFalse((self.dst / "temp" / "lora.txt").exists())
        self.assertFalse(list(self.dst.rglob("*.partial")))
        self.assertEqual(migrate.database_summary(self.dst / "tag_frequency.db"), ("1", 2))
        self.assertIn("7 files: conflict 1, copied 5, identical 1; copied", out)
        entries = [json.loads(line) for line in self.progress.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sorted(e["name"] for e in entries),
                         sorted(["KR_tags-v2.csv", "e621.csv", "noob-chants.json", "tag_frequency.db",
                                 "temp/known_lora_hashes.txt", "danbooru.csv"]))

        # a second run: everything is done, nothing is copied again
        code, out = self.run_tool("--apply")
        self.assertEqual(code, 2)
        self.assertNotIn("copied    ", out)
        self.assertIn("conflict 1, done 6", out)

    def test_without_conflicts_the_exit_code_is_0(self):
        (self.dst / "extra-quality-tags.csv").unlink()
        code, _ = self.run_tool("--apply")
        self.assertEqual(code, 0)
        self.assertEqual((self.dst / "extra-quality-tags.csv").read_bytes(), b"masterpiece,5,0,\n")

    def test_no_cache_leaves_the_hash_cache(self):
        (self.dst / "extra-quality-tags.csv").unlink()
        code, out = self.run_tool("--apply", "--no-cache")
        self.assertEqual(code, 0)
        self.assertFalse((self.dst / "temp").exists())
        self.assertNotIn("known_lora_hashes", out)

    def test_an_existing_database_is_not_replaced(self):
        make_database(self.dst / "tag_frequency.db", [])   # e.g. the built-in started once before the copy
        code, out = self.run_tool("--apply")
        self.assertEqual(code, 2)
        self.assertEqual(migrate.database_summary(self.dst / "tag_frequency.db"), ("1", 0))
        self.assertIn("conflict  tag_frequency.db", out)
        self.assertIn("it has 0 rows, schema 1", out)

    def test_a_database_in_the_middle_of_a_write_is_not_copied(self):
        (self.src_tags / "tag_frequency.db-journal").write_bytes(b"x")
        code, out = self.run_tool("--apply")
        self.assertEqual(code, 2)
        self.assertFalse((self.dst / "tag_frequency.db").exists())
        self.assertFalse((self.dst / "tag_frequency.db-journal").exists())
        self.assertIn("blocked   tag_frequency.db", out)
        self.assertIn("stop Forge and run again", out)

    def test_a_file_that_appears_while_copying_is_not_overwritten(self):
        (self.dst / "extra-quality-tags.csv").unlink()
        real_place = migrate._place

        def racing(temporary, final):
            if final.name == "e621.csv":
                final.write_bytes(b"someone else's")
            real_place(temporary, final)

        with mock.patch.object(migrate, "_place", racing):
            code, out = self.run_tool("--apply")
        self.assertEqual(code, 2)
        self.assertEqual((self.dst / "e621.csv").read_bytes(), b"someone else's")
        self.assertIn("conflict  e621.csv: appeared in the target while copying", out)
        self.assertFalse(list(self.dst.rglob("*.partial")))

    def test_a_changed_copy_is_not_put_in_place(self):
        (self.dst / "extra-quality-tags.csv").unlink()
        real_copy = shutil.copy2

        def corrupt(src, dst, *args, **kwargs):
            real_copy(src, dst, *args, **kwargs)
            if Path(src).name == "e621.csv":
                Path(dst).write_bytes(b"corrupt")

        with mock.patch.object(migrate.shutil, "copy2", corrupt):
            code, out = self.run_tool("--apply")
        self.assertEqual(code, 2)
        self.assertFalse((self.dst / "e621.csv").exists())
        self.assertIn("FAILED    e621.csv", out)
        self.assertFalse(list(self.dst.rglob("*.partial")))

    def test_preconditions(self):
        cases = {
            "missing source": (self.tmp / "nope", self.dst),
            "not tagcomplete": (self.tmp / "forge_sam3_extension", self.dst),
            "missing target": (self.old, self.tmp / "nope"),
            "same folder": (self.old, self.src_tags),
        }
        for label, (src, dst) in cases.items():
            with self.subTest(label):
                before = tree_hashes(self.tmp)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = migrate.main(["--src", str(src), "--dst", str(dst), "--apply"])
                self.assertEqual(code, 1)
                self.assertIn("ABORT", out.getvalue())
                self.assertEqual(tree_hashes(self.tmp), before)

    def test_the_default_target_is_this_extensions_tags_folder(self):
        self.assertTrue(os.path.samefile(migrate.DEFAULT_DST, ROOT / "tags"))
        self.assertTrue(migrate.DEFAULT_DST.is_dir())
        self.assertEqual(migrate.DATABASE, "tag_frequency.db")


if __name__ == "__main__":
    unittest.main()
