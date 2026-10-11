# sam-extra 자체 코드(GPL-3.0-only) — 상류 tagcomplete 코드가 아니다.
"""Copy a user's Tag Autocomplete data from an old sd-webui-tagcomplete(-neo) folder into sam-extra's tags/.

sd-webui-tagcomplete-neo is built into sam-extra (sam3ext/tagcomplete). What the old extension kept in its tags/
folder and git does not ship — the user's own tag lists, the upstream lists sam-extra does not track, the usage counts
(tag_frequency.db) and the model-keyword hash cache (temp/known_lora_hashes.txt) — is copied over by this tool, once,
before the old folder is removed. Everything else in temp/ is rebuilt by the helper at every start and is not copied.

    python tools/migrate_tagcomplete.py --src <old extension folder> [--dst <sam-extra>/tags]
                                        [--progress <file.jsonl>] [--apply] [--no-cache]

* Dry run unless ``--apply``: prints what would be copied, with absolute paths, sizes and SHA-256.
* Never overwrites: a file that already exists in ``--dst`` with the same bytes counts as done; with other bytes it is
  a conflict, left alone and reported (exit code 2). A copy is written under a temporary name, its SHA-256 is checked
  against the source, then it is renamed into place with an operation that fails if the name exists by then.
* Never changes ``--src`` (read only). Copies keep the source's modification time.
* tag_frequency.db is copied only when no ``-journal`` / ``-wal`` file sits next to it (an unfinished write — stop Forge
  and run again), and the copy is opened read-only afterwards: integrity check, schema version, row count.
* ``--progress`` appends one JSON line per finished file; a second run skips those whose copy is still intact.
* Exit codes: 0 done (or nothing to do), 1 a precondition failed (nothing written), 2 conflicts or failed files.

Standard library only; run it with the Forge venv's ``python -I``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

EXT_ROOT = Path(os.path.abspath(__file__)).parents[1]
DEFAULT_DST = EXT_ROOT / "tags"
UPSTREAM_MARKERS = ("scripts/tag_autocomplete_helper.py", "javascript/tagAutocomplete.js")
DATABASE = "tag_frequency.db"
CACHE = "temp/known_lora_hashes.txt"
CHUNK = 1 << 20


class PreconditionError(Exception):
    pass


@dataclass
class Item:
    name: str          # relative to the tags folders, POSIX form
    src: Path
    dst: Path
    size: int
    sha256: str
    action: str = ""   # copy · identical · conflict · done · blocked
    note: str = ""


def _print(text: str = "") -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:   # legacy console code pages (cp949 ...)
        print(text.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_folder(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def check_preconditions(src: Path, dst: Path) -> Path:
    """The source tags/ folder; raises PreconditionError with the reason."""
    if not src.is_dir():
        raise PreconditionError(f"--src is not a folder: {src}")
    if not any((src / marker).is_file() for marker in UPSTREAM_MARKERS):
        raise PreconditionError(f"--src does not look like a tagcomplete extension (no {' or '.join(UPSTREAM_MARKERS)}): {src}")
    tags = src / "tags"
    if not tags.is_dir():
        raise PreconditionError(f"--src has no tags folder: {tags}")
    if not dst.is_dir():
        raise PreconditionError(f"--dst is not a folder: {dst}")
    if _same_folder(tags, dst):
        raise PreconditionError(f"--src tags folder and --dst are the same folder: {dst}")
    return tags


SQLITE_SIDE_FILES = ("-journal", "-wal", "-shm")


def collect(src_tags: Path, dst: Path, include_cache: bool) -> list[Item]:
    names = sorted(entry.name for entry in src_tags.iterdir()
                   if entry.is_file() and not entry.name.endswith(SQLITE_SIDE_FILES))
    if include_cache and (src_tags / CACHE).is_file():
        names.append(CACHE)
    items = []
    for name in names:
        source = src_tags / name
        items.append(Item(name=name, src=source, dst=dst / name, size=source.stat().st_size, sha256=sha256_of(source)))
    return items


def _journal(path: Path) -> Path | None:
    for suffix in ("-journal", "-wal"):
        candidate = path.with_name(path.name + suffix)
        if candidate.exists():
            return candidate
    return None


def database_summary(path: Path) -> tuple[str, int]:
    """(schema version, rows) of a tag_frequency.db, opened read-only."""
    uri = Path(os.path.abspath(path)).as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        status = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if status != "ok":
            raise sqlite3.DatabaseError(f"integrity check: {status}")
        row = connection.execute("SELECT value FROM db_data WHERE key = 'version'").fetchone()
        rows = connection.execute("SELECT COUNT(*) FROM tag_frequency").fetchone()[0]
    finally:
        connection.close()
    return (str(row[0]) if row else "?"), int(rows)


def load_progress(path: Path | None) -> dict:
    done = {}
    if path is None or not path.is_file():
        return done
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict) and "name" in entry and "sha256" in entry:
                done[entry["name"]] = entry["sha256"]
    return done


def append_progress(path: Path | None, item: Item) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"name": item.name, "sha256": item.sha256, "bytes": item.size, "action": item.action,
                                 "dst": str(item.dst)}, ensure_ascii=False) + "\n")


def plan(items: list[Item], progress: dict) -> None:
    for item in items:
        if item.name == DATABASE and _journal(item.src) is not None:
            item.action, item.note = "blocked", f"unfinished write ({_journal(item.src).name}) - stop Forge and run again"
            continue
        if item.dst.exists():
            same = item.dst.is_file() and item.dst.stat().st_size == item.size and sha256_of(item.dst) == item.sha256
            if same:
                item.action = "done" if progress.get(item.name) == item.sha256 else "identical"
            else:
                item.action = "conflict"
                item.note = "a different file with this name is already there - left as it is"
                if item.name == DATABASE:
                    try:
                        version, rows = database_summary(item.dst)
                        item.note += f" (it has {rows} rows, schema {version}; the old one is not merged in)"
                    except sqlite3.Error as error:
                        item.note += f" ({error})"
            continue
        item.action = "copy"
        if item.name == DATABASE:
            try:
                version, rows = database_summary(item.src)
                item.note = f"schema {version}, {rows} rows"
            except sqlite3.Error as error:
                item.action, item.note = "blocked", f"not a readable tag_frequency.db: {error}"


def _place(temporary: Path, final: Path) -> None:
    """Rename into place; fails (FileExistsError) when ``final`` exists — never overwrites."""
    if os.name == "nt":
        os.rename(temporary, final)
    else:
        os.link(temporary, final)
        os.unlink(temporary)


def copy_one(item: Item) -> None:
    item.dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = item.dst.with_name(f".{item.dst.name}.{uuid.uuid4().hex[:8]}.partial")
    try:
        shutil.copy2(item.src, temporary)
        if sha256_of(temporary) != item.sha256:
            raise OSError("the copy's SHA-256 differs from the source (source changed while copying?)")
        _place(temporary, item.dst)
    finally:
        if temporary.exists():
            temporary.unlink()
    if sha256_of(item.dst) != item.sha256:
        raise OSError(f"SHA-256 of {item.dst} differs after the copy")
    if item.name == DATABASE:
        version, rows = database_summary(item.dst)
        source_version, source_rows = database_summary(item.src)
        if (version, rows) != (source_version, source_rows):
            raise sqlite3.DatabaseError(f"copy has schema {version} / {rows} rows, source {source_version} / {source_rows}")
        item.note = f"schema {version}, {rows} rows"


def _mib(size: int) -> str:
    return f"{size / (1 << 20):.2f} MiB"


def run(src: Path, dst: Path, apply: bool, include_cache: bool, progress_path: Path | None) -> int:
    src, dst = Path(os.path.abspath(src)), Path(os.path.abspath(dst))
    try:
        src_tags = check_preconditions(src, dst)
    except PreconditionError as error:
        _print(f"[migrate_tagcomplete] ABORT: {error}")
        return 1
    progress = load_progress(progress_path)
    items = collect(src_tags, dst, include_cache)
    plan(items, progress)

    _print(f"[migrate_tagcomplete] {'APPLY' if apply else 'DRY RUN (add --apply to copy)'}")
    _print(f"  source: {src_tags}")
    _print(f"  target: {dst}")
    if progress_path is not None:
        _print(f"  progress: {Path(os.path.abspath(progress_path))}")
    if (src_tags / "temp").is_dir():
        rebuilt = sorted(p.name for p in (src_tags / "temp").iterdir() if p.is_file() and f"temp/{p.name}" != CACHE)
        if rebuilt:
            _print(f"  not copied (rebuilt at every start): temp/{', temp/'.join(rebuilt)}")
    for item in items:
        line = f"  {item.action:<9} {item.name}  {_mib(item.size)}  sha256 {item.sha256[:16]}  -> {item.dst}"
        _print(line + (f"  [{item.note}]" if item.note else ""))

    failed = []
    if apply:
        for item in items:
            if item.action != "copy":
                continue
            try:
                copy_one(item)
            except FileExistsError:
                item.action, item.note = "conflict", "appeared in the target while copying - left as it is"
                _print(f"  conflict  {item.name}: {item.note}")
            except (OSError, sqlite3.Error) as error:
                item.action, item.note = "failed", str(error)
                failed.append(item)
            else:
                item.action = "copied"
                append_progress(progress_path, item)
                _print(f"  copied    {item.name} -> {item.dst}  (sha256 verified{', ' + item.note if item.note else ''})")
        for item in items:
            if item.action == "identical":
                append_progress(progress_path, item)
        for item in failed:
            _print(f"  FAILED    {item.name}: {item.note}")

    counts = {}
    for item in items:
        counts[item.action] = counts.get(item.action, 0) + 1
    to_copy = [item for item in items if item.action in ("copy", "copied")]
    summary = ", ".join(f"{key} {value}" for key, value in sorted(counts.items()))
    _print(f"[migrate_tagcomplete] {len(items)} files: {summary}; "
           f"{'copied' if apply else 'to copy'} {_mib(sum(i.size for i in to_copy))} into {dst}")
    if any(item.action in ("conflict", "blocked", "failed") for item in items):
        return 2
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", required=True, type=Path, help="the old extension folder (e.g. extensions/sd-webui-tagcomplete-neo)")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help=f"sam-extra's tags folder (default: {DEFAULT_DST})")
    parser.add_argument("--progress", type=Path, default=None, help="JSONL file of finished files (appended)")
    parser.add_argument("--apply", action="store_true", help="copy (without it: only show the plan)")
    parser.add_argument("--no-cache", action="store_true", help="leave temp/known_lora_hashes.txt (a rebuildable cache)")
    args = parser.parse_args(argv)
    return run(args.src, args.dst, args.apply, not args.no_cache, args.progress)


if __name__ == "__main__":
    sys.exit(main())
