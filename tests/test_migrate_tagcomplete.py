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


# ---------------------------------------------------------------------------------------------------------------------
# --set-tagfile: tac_tagFile in a config.json (temporary copies only)
# ---------------------------------------------------------------------------------------------------------------------

DEFAULT_TAG_FILE = "danbooru_2026-10-01_pt20-ia-dd.csv"
OLD_TAG_FILE = "KR_danbooru_tags_with_description-20260907.csv"


def forge_config(settings: dict) -> bytes:
    """config.json as Forge writes it on Windows: json.dump(indent=4, ensure_ascii=False) through a text file (CRLF)."""
    return json.dumps(settings, indent=4, ensure_ascii=False).replace("\n", "\r\n").encode("utf-8")


SETTINGS = {
    "VERSION_UID": "PY313",
    "outdir_samples": "C:\\출력\\그림",          # a non-ASCII path, written as is
    "sd_model_checkpoint": "anima\\animaOfficial.safetensors [abc]",
    "tac_tagFile": OLD_TAG_FILE,
    "tac_active": True,
    "tac_maxResults": 5,
    "cfg_scale": 4.5,
    "tac_keymap": "{\n    \"MoveUp\": \"ArrowUp\"\n}",   # a JSON text inside a string
    "nested": {"tac_tagFile": "not this one", "list": [1, 2.0, None, "a \"quoted\" tac_tagFile"]},
    "note": "\"tac_tagFile\": \"also not this one\"",
    "tac_chantFile": "demo-chants.json",
    "empty": None,
}


class SetTagFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sam3_settagfile_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.forge = self.tmp / "forge"
        self.forge.mkdir()
        self.config = self.forge / "config.json"
        self.original = forge_config(SETTINGS)
        self.config.write_bytes(self.original)
        self.tags = self.tmp / "forge_sam3_extension" / "tags"
        self.tags.mkdir(parents=True)
        for name in (DEFAULT_TAG_FILE, "danbooru.csv"):
            (self.tags / name).write_bytes(b"1girl,0,1,\n")
        # a port nothing listens on (bound, then closed) for the "is Forge running" probe
        probe = __import__("socket").socket()
        probe.bind(("127.0.0.1", 0))
        self.free_port = probe.getsockname()[1]
        probe.close()

    def run_tool(self, *extra, name=DEFAULT_TAG_FILE, port=None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = migrate.main(["--set-tagfile", name, "--config", str(self.config), "--dst", str(self.tags),
                                 "--forge-port", str(port or self.free_port), *extra])
        return code, out.getvalue()

    def backups(self):
        return sorted(p.name for p in self.forge.iterdir() if p.name.startswith("config.json.bak-tagfile-"))

    def expected_bytes(self):
        return self.original.replace(f'"tac_tagFile": "{OLD_TAG_FILE}"'.encode(),
                                     f'"tac_tagFile": "{DEFAULT_TAG_FILE}"'.encode())

    def test_dry_run_prints_old_and_new_and_writes_nothing(self):
        code, out = self.run_tool()
        self.assertEqual(code, 0)
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertEqual(self.backups(), [])
        self.assertEqual(sorted(p.name for p in self.forge.iterdir()), ["config.json"])
        self.assertIn("DRY RUN", out)
        self.assertIn(f"tac_tagFile: '{OLD_TAG_FILE}' -> '{DEFAULT_TAG_FILE}'", out)
        self.assertIn(f"would write the backup {self.forge / 'config.json.bak-tagfile-'}", out)
        self.assertIn(f"config: {self.config}", out)

    def test_apply_needs_forge_stopped(self):
        code, out = self.run_tool("--apply")
        self.assertEqual(code, 1)
        self.assertIn("--apply needs --forge-stopped", out)
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertEqual(self.backups(), [])

    def test_apply_changes_only_that_value_after_a_verified_backup(self):
        before = self.config.stat().st_mtime_ns
        code, out = self.run_tool("--apply", "--forge-stopped")
        self.assertEqual(code, 0, out)
        new = self.config.read_bytes()
        self.assertEqual(new, self.expected_bytes(), "every other byte as it was")
        self.assertEqual(new.count(b"\r\n"), self.original.count(b"\r\n"))
        self.assertNotIn(b"\n", new.replace(b"\r\n", b""), "CRLF kept")
        loaded = json.loads(new.decode("utf-8"))
        self.assertEqual(list(loaded), list(SETTINGS), "key order")
        self.assertEqual(loaded, {**SETTINGS, "tac_tagFile": DEFAULT_TAG_FILE})
        (backup,) = self.backups()
        self.assertRegex(backup, r"^config\.json\.bak-tagfile-\d{8}-\d{6}$")
        self.assertEqual((self.forge / backup).read_bytes(), self.original)
        self.assertEqual((self.forge / backup).stat().st_mtime_ns // 10**6, before // 10**6, "the backup keeps the time")
        self.assertIn(f"tac_tagFile: '{OLD_TAG_FILE}' -> '{DEFAULT_TAG_FILE}' written to {self.config}", out)
        self.assertIn(f"backup: {self.forge / backup}", out)
        self.assertFalse(list(self.forge.glob("*.partial")))

        code, out = self.run_tool("--apply", "--forge-stopped")
        self.assertEqual(code, 0)
        self.assertIn("is already", out)
        self.assertEqual(len(self.backups()), 1, "no second backup")
        self.assertEqual(self.config.read_bytes(), new)

    def test_refused_while_something_answers_on_the_forge_port(self):
        import socket

        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen()
        self.addCleanup(server.close)
        port = server.getsockname()[1]
        for extra in ((), ("--apply", "--forge-stopped")):
            with self.subTest(extra=extra):
                code, out = self.run_tool(*extra, port=port)
                self.assertEqual(code, 1)
                self.assertIn(f"Forge seems to be running (127.0.0.1:{port} answers)", out)
                self.assertEqual(self.config.read_bytes(), self.original)
                self.assertEqual(self.backups(), [])

    def test_every_given_port_is_probed_and_7860_is_the_default(self):
        self.assertEqual(migrate.DEFAULT_FORGE_PORTS, (7860,))
        with mock.patch.object(migrate, "forge_listening", return_value=[]) as probe:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                migrate.main(["--set-tagfile", DEFAULT_TAG_FILE, "--config", str(self.config), "--dst", str(self.tags)])
            self.assertEqual(probe.call_args.args[0], [7860])
            with contextlib.redirect_stdout(out):
                migrate.main(["--set-tagfile", DEFAULT_TAG_FILE, "--config", str(self.config), "--dst", str(self.tags),
                              "--forge-port", "7860", "--forge-port", "17860"])
            self.assertEqual(probe.call_args.args[0], [7860, 17860])

    def test_refused_while_the_usage_counts_have_an_unfinished_write(self):
        (self.tags / "tag_frequency.db-journal").write_bytes(b"")
        code, out = self.run_tool("--apply", "--forge-stopped")
        self.assertEqual(code, 1)
        self.assertIn("unfinished write", out)
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertEqual(self.backups(), [])

    def test_the_name_must_be_a_csv_in_the_tags_folder(self):
        for name, message in (("gone.csv", "is not in"), ("sub/danbooru.csv", "not a path"),
                              ("..\\danbooru.csv", "not a path"), ("demo-chants.json", ".csv tag list"),
                              ("", "not a path"), (" danbooru.csv", "not a path")):
            with self.subTest(name=name):
                code, out = self.run_tool("--apply", "--forge-stopped", name=name)
                self.assertEqual(code, 1)
                self.assertIn(message, out)
                self.assertEqual(self.config.read_bytes(), self.original)
                self.assertEqual(self.backups(), [])
        code, _ = self.run_tool("--apply", "--forge-stopped", name="danbooru.csv")
        self.assertEqual(code, 0, "any list that is in tags/")

    def test_a_config_without_the_key_gets_it_as_the_last_member(self):
        settings = {key: value for key, value in SETTINGS.items() if key != "tac_tagFile"}
        original = forge_config(settings)
        self.config.write_bytes(original)
        code, out = self.run_tool("--apply", "--forge-stopped")
        self.assertEqual(code, 0, out)
        self.assertIn(f"tac_tagFile: (not set) -> '{DEFAULT_TAG_FILE}'", out)
        new = self.config.read_bytes()
        self.assertEqual(new, original[:-3] + f',\r\n    "tac_tagFile": "{DEFAULT_TAG_FILE}"\r\n}}'.encode(),
                         "after the last member, with its line break, indentation and ': '")
        self.assertEqual(json.loads(new.decode("utf-8")), {**settings, "tac_tagFile": DEFAULT_TAG_FILE})

    def test_a_byte_order_mark_and_lf_files_stay_as_they_are(self):
        original = b"\xef\xbb\xbf" + json.dumps(SETTINGS, indent=2, ensure_ascii=False).encode("utf-8")
        self.config.write_bytes(original)
        code, out = self.run_tool("--apply", "--forge-stopped")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.config.read_bytes(),
                         original.replace(f'"tac_tagFile": "{OLD_TAG_FILE}"'.encode(),
                                          f'"tac_tagFile": "{DEFAULT_TAG_FILE}"'.encode()))

    def test_files_it_cannot_edit_safely_are_left_alone(self):
        cases = {
            "duplicate key": b'{\r\n    "tac_tagFile": "a.csv",\r\n    "tac_tagFile": "b.csv"\r\n}',
            "not a string": b'{\r\n    "tac_tagFile": 3\r\n}',
            "broken JSON": b'{\r\n    "tac_tagFile": "a.csv",\r\n}',
            "text after it": b'{"tac_tagFile": "a.csv"} x',
            "not an object": b'["tac_tagFile"]',
            "no settings": b"{}",
            "not UTF-8": b'{"tac_tagFile": "\xff.csv"}',
        }
        for label, data in cases.items():
            with self.subTest(label):
                self.config.write_bytes(data)
                code, out = self.run_tool("--apply", "--forge-stopped")
                self.assertEqual(code, 1)
                self.assertIn("ABORT", out)
                self.assertEqual(self.config.read_bytes(), data)
                self.assertEqual(self.backups(), [])
        self.config.unlink()
        code, out = self.run_tool("--apply", "--forge-stopped")
        self.assertEqual((code, "--config is not a file" in out), (1, True))

    def test_a_config_changed_after_it_was_read_is_not_overwritten(self):
        real = shutil.copystat
        changed = forge_config({**SETTINGS, "cfg_scale": 7})

        def meanwhile(src, dst, *args, **kwargs):   # Forge saving its settings right after the backup
            real(src, dst, *args, **kwargs)
            Path(src).write_bytes(changed)

        with mock.patch.object(migrate.shutil, "copystat", meanwhile):
            code, out = self.run_tool("--apply", "--forge-stopped")
        self.assertEqual(code, 1)
        self.assertIn("changed while this ran", out)
        self.assertEqual(self.config.read_bytes(), changed)
        (backup,) = self.backups()
        self.assertEqual((self.forge / backup).read_bytes(), self.original)

    def test_with_tag_file_touches_only_the_top_level_value(self):
        text = self.original.decode("utf-8")
        new, old = migrate.with_tag_file(text, "x.csv")
        self.assertEqual(old, OLD_TAG_FILE)
        start = text.index(f'"tac_tagFile": "{OLD_TAG_FILE}"') + len('"tac_tagFile": ')
        self.assertEqual(new, text[:start] + '"x.csv"' + text[start + len(OLD_TAG_FILE) + 2:])
        self.assertEqual(json.loads(new)["nested"], SETTINGS["nested"])
        self.assertEqual(json.loads(new)["note"], SETTINGS["note"])
        compact, old = migrate.with_tag_file('{"a":1,"b":[2]}', "x.csv")
        self.assertEqual((compact, old), ('{"a":1,"b":[2],"tac_tagFile":"x.csv"}', None))
        single, _ = migrate.with_tag_file('{\n\t"a": 1\n}\n', "한글.csv")
        self.assertEqual(single, '{\n\t"a": 1,\n\t"tac_tagFile": "한글.csv"\n}\n')

    def test_set_tagfile_and_copying_are_separate_runs(self):
        for argv in (["--set-tagfile", DEFAULT_TAG_FILE, "--src", str(self.tmp), "--config", str(self.config)],
                     ["--set-tagfile", DEFAULT_TAG_FILE],
                     ["--src", str(self.tmp), "--config", str(self.config)],
                     ["--src", str(self.tmp), "--forge-stopped"],
                     ["--set-tagfile", DEFAULT_TAG_FILE, "--config", str(self.config), "--forge-port", "70000"],
                     ["--set-tagfile", DEFAULT_TAG_FILE, "--config", str(self.config), "--forge-port", "x"],
                     []):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    migrate.main(argv)
                self.assertEqual(raised.exception.code, 2)
        self.assertEqual(self.config.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
