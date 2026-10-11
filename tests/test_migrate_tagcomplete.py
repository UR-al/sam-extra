"""tools/migrate_tagcomplete.py — copies a user's tag lists, tag_frequency.db and the model-keyword hash cache from an
old sd-webui-tagcomplete(-neo) folder into sam-extra's tags/, on temporary folders only.

Dry run by default; never overwrites (identical = done, different = conflict, exit 2); never changes the source;
SHA-256 checked for every copy, the database opened read-only afterwards (integrity, schema, rows); temp/ files the
helper rebuilds are not copied; a progress JSONL lets a second run skip finished files; preconditions fail with
exit 1 before anything is written.

The built-in may have started before the copy (the old extension disabled first): its empty tag_frequency.db (0 rows,
same schema, no unfinished write) and empty temp/known_lora_hashes.txt are set aside as ``*.empty-<time>.bak`` and
replaced — checked again right before, and put back when the copy fails. A database with rows stays a conflict whose
message names the way out; a different non-empty hash cache is kept without counting as a conflict.
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

    def test_a_database_with_rows_is_not_replaced_and_the_way_out_is_named(self):
        target = self.dst / "tag_frequency.db"
        make_database(target, [("1girl", 0, 1, 0)])   # the built-in has been counting for a while
        before = target.read_bytes()
        code, out = self.run_tool("--apply")
        self.assertEqual(code, 2)
        self.assertEqual(target.read_bytes(), before)
        self.assertIn("conflict  tag_frequency.db", out)
        self.assertIn("it has 1 rows, schema 1; the old one is not merged in", out)
        self.assertIn(f"To use the old counts (2 rows) instead: stop Forge, move {target} aside and run again", out)
        self.assertFalse(list(self.dst.glob("*.bak")))

    def test_the_builtins_empty_database_is_set_aside_and_replaced(self):
        """The old extension was disabled first, so the built-in started once and made an empty tag_frequency.db: the
        usage counts must still arrive (they were a conflict before, exit 2, nothing copied)."""
        (self.dst / "extra-quality-tags.csv").unlink()
        target = self.dst / "tag_frequency.db"
        make_database(target, [])
        empty = target.read_bytes()
        code, out = self.run_tool()
        self.assertEqual(code, 0)
        self.assertEqual(target.read_bytes(), empty, "a dry run changes nothing")
        self.assertIn("replace   tag_frequency.db", out)
        self.assertIn("the one there is empty (0 rows, schema 1", out)

        code, out = self.run_tool("--apply")
        self.assertEqual(code, 0, out)
        self.assertEqual(migrate.database_summary(target), ("1", 2))
        self.assertEqual(target.read_bytes(), (self.src_tags / "tag_frequency.db").read_bytes())
        (aside,) = self.dst.glob("tag_frequency.db.empty-*.bak")
        self.assertEqual(aside.read_bytes(), empty, "set aside, never deleted")
        self.assertIn(f"replaced  tag_frequency.db -> {target}  (sha256 verified, schema 1, 2 rows; "
                      f"the empty one is now {aside})", out)
        entries = [json.loads(line) for line in self.progress.read_text(encoding="utf-8").splitlines()]
        (entry,) = [e for e in entries if e["name"] == "tag_frequency.db"]
        self.assertEqual((entry["action"], entry["aside"]), ("replaced", str(aside)))
        code, out = self.run_tool("--apply")
        self.assertEqual(code, 0)
        self.assertIn("done      tag_frequency.db", out)

    def test_an_empty_database_is_left_alone_while_written_or_with_another_schema(self):
        (self.dst / "extra-quality-tags.csv").unlink()
        target = self.dst / "tag_frequency.db"
        make_database(target, [])
        empty = target.read_bytes()
        (self.dst / "tag_frequency.db-journal").write_bytes(b"x")   # Forge is writing to it
        code, out = self.run_tool("--apply")
        self.assertEqual(code, 2)
        self.assertEqual(target.read_bytes(), empty)
        self.assertIn("conflict  tag_frequency.db", out)
        self.assertIn("the one there has an unfinished write (tag_frequency.db-journal) - stop Forge and run again", out)
        (self.dst / "tag_frequency.db-journal").unlink()
        self.assertEqual(migrate.database_summary(target), ("1", 0))
        connection = sqlite3.connect(target)
        with connection:
            connection.execute("UPDATE db_data SET value = 2 WHERE key = 'version'")
        connection.close()
        code, out = self.run_tool("--apply")
        self.assertEqual(code, 2)
        self.assertEqual(migrate.database_summary(target), ("2", 0))
        self.assertIn("the one there is empty but has schema 2 (the old one: 1) - left as it is", out)
        self.assertFalse(list(self.dst.glob("*.bak")))

    def test_a_placeholder_that_changes_after_the_plan_is_left_alone(self):
        (self.dst / "extra-quality-tags.csv").unlink()
        target = self.dst / "tag_frequency.db"
        make_database(target, [])
        real_replace = migrate.replace_one

        def forge_still_running(item):
            if item.name == "tag_frequency.db":
                connection = sqlite3.connect(target)
                with connection:
                    connection.execute("INSERT INTO tag_frequency (name, type, count_pos, count_neg) VALUES ('x', 0, 1, 0)")
                connection.close()
            real_replace(item)

        with mock.patch.object(migrate, "replace_one", forge_still_running):
            code, out = self.run_tool("--apply")
        self.assertEqual(code, 2)
        self.assertEqual(migrate.database_summary(target), ("1", 1))
        self.assertIn(f"conflict  tag_frequency.db: {target} changed since the plan - left as it is", out)
        self.assertFalse(list(self.dst.glob("*.bak")))

    def test_a_failed_copy_puts_the_placeholder_back(self):
        (self.dst / "extra-quality-tags.csv").unlink()
        target = self.dst / "tag_frequency.db"
        make_database(target, [])
        empty = target.read_bytes()
        real_copy = shutil.copy2

        def corrupt(src, dst, *args, **kwargs):
            real_copy(src, dst, *args, **kwargs)
            if Path(src).name == "tag_frequency.db":
                Path(dst).write_bytes(b"corrupt")

        with mock.patch.object(migrate.shutil, "copy2", corrupt):
            code, out = self.run_tool("--apply")
        self.assertEqual(code, 2)
        self.assertEqual(target.read_bytes(), empty)
        self.assertIn("FAILED    tag_frequency.db", out)
        self.assertFalse(list(self.dst.glob("*.bak")), "renamed back")
        self.assertFalse(list(self.dst.rglob("*.partial")))

    def test_the_hash_cache_replaces_an_empty_one_and_never_counts_as_a_conflict(self):
        (self.dst / "extra-quality-tags.csv").unlink()
        cache = self.dst / "temp" / "known_lora_hashes.txt"
        cache.parent.mkdir()
        cache.write_bytes(b"")   # model_keyword_support touches it at the built-in's first start
        code, out = self.run_tool("--apply")
        self.assertEqual(code, 0, out)
        self.assertEqual(cache.read_bytes(), (self.src_tags / "temp" / "known_lora_hashes.txt").read_bytes())
        (aside,) = cache.parent.glob("known_lora_hashes.txt.empty-*.bak")
        self.assertEqual(aside.read_bytes(), b"")
        self.assertIn("replaced  temp/known_lora_hashes.txt", out)

        cache.write_bytes(b"b.safetensors,99999999,2.0\n")   # the built-in has hashed other LoRAs since
        code, out = self.run_tool("--apply")
        self.assertEqual(code, 0, out)
        self.assertEqual(cache.read_bytes(), b"b.safetensors,99999999,2.0\n")
        self.assertIn("kept      temp/known_lora_hashes.txt", out)
        self.assertIn("a cache the helper rebuilds", out)

    def test_after_the_builtin_started_first_everything_still_arrives(self):
        """The order the old console line led to: disable the old copy, restart (the real helper makes its empty
        tag_frequency.db and known_lora_hashes.txt in tags/), then migrate. Before: 2 conflicts, exit 2, no counts."""
        from sam3ext.tagcomplete import boot
        from tests import _tagcomplete_support as tac

        with tac.sandboxed() as box:
            failure, _ = tac.run_quietly(boot.start_helper)
            self.assertIsNone(failure)
            self.assertEqual(migrate.database_summary(box.tags / "tag_frequency.db"), ("1", 0))
            self.assertEqual((box.tags / "temp" / "known_lora_hashes.txt").read_bytes(), b"")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = migrate.main(["--src", str(self.old), "--dst", str(box.tags), "--apply"])
            self.assertEqual(code, 0, out.getvalue())
            self.assertEqual(migrate.database_summary(box.tags / "tag_frequency.db"), ("1", 2))
            self.assertEqual((box.tags / "temp" / "known_lora_hashes.txt").read_bytes(),
                             (self.src_tags / "temp" / "known_lora_hashes.txt").read_bytes())
            self.assertEqual(out.getvalue().count("replaced  "), 2, out.getvalue())
            self.assertNotIn("conflict", out.getvalue())

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
