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
* Two exceptions, both placeholders the built-in copy makes when it starts before the copy (e.g. after the old extension
  was disabled first): an empty tag_frequency.db (0 rows, the old one's schema, no unfinished write next to it) and an
  empty temp/known_lora_hashes.txt are renamed aside (``<name>.empty-<time>.bak`` next to it, never deleted), then the
  old file is copied ("replace"). A tag_frequency.db with rows stays a conflict - the message says how to use the old
  counts instead. A different, non-empty temp/known_lora_hashes.txt is kept and reported ("kept", not a conflict): it is
  a cache the helper rebuilds.
* Never changes ``--src`` (read only). Copies keep the source's modification time.
* tag_frequency.db is copied only when no ``-journal`` / ``-wal`` file sits next to it (an unfinished write — stop Forge
  and run again), and the copy is opened read-only afterwards: integrity check, schema version, row count.
* ``--progress`` appends one JSON line per finished file; a second run skips those whose copy is still intact.
* Exit codes: 0 done (or nothing to do), 1 a precondition failed (nothing written), 2 conflicts, blocked or failed files.

A separate, opt-in run sets the tag list Forge uses (``tac_tagFile``) in a config.json — e.g. to the built-in default
``danbooru_2026-10-01_pt20-ia-dd.csv`` when config.json still names another list (a saved value wins over a default):

    python tools/migrate_tagcomplete.py --set-tagfile <name> --config <forge>/config.json [--dst <sam-extra>/tags]
                                        [--forge-port <port> ...] [--apply --forge-stopped]

* ``<name>`` is a .csv file in ``--dst`` (what Settings -> Tag Autocomplete lists), not a path.
* Dry run unless ``--apply``: prints ``tac_tagFile: 'old' -> 'new'`` and the backup it would write. ``--apply`` also needs
  ``--forge-stopped``: a running Forge keeps its settings in memory and writes all of them back to config.json when
  settings are saved, which would undo the edit (and Forge would not use it before a restart anyway).
* Refused (exit 1, nothing written) while Forge looks alive: something answers on 127.0.0.1 at a ``--forge-port``
  (default 7860; repeat it for every port this Forge is started with), or tag_frequency.db in ``--dst`` has an
  unfinished write (``-journal`` / ``-wal``) next to it.
* Writes ``config.json.bak-tagfile-<time>`` next to it first (never overwriting; the bytes are checked), then changes
  only that value: every other byte of the file stays as it was — key order, indentation, line endings, other values.
  A config.json without the key gets it as a new last member, written like the member before it. The new file is
  written under a temporary name, checked, then renamed over config.json; config.json changed since it was read stops
  the run before that.
* Exit codes: 0 done, already set, or a dry run that --apply would carry out; 1 a precondition failed (nothing changed).

Standard library only; run it with the Forge venv's ``python -I``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import sys
import time
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
    action: str = ""   # copy · replace · identical · done · kept · conflict · blocked (then: copied · replaced · failed)
    note: str = ""
    dst_size: int | None = None      # replace: the placeholder the plan saw (it must still be that file)
    dst_sha256: str | None = None
    aside: Path | None = None        # replaced: where the placeholder went


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
        entry = {"name": item.name, "sha256": item.sha256, "bytes": item.size, "action": item.action, "dst": str(item.dst)}
        if item.aside is not None:
            entry["aside"] = str(item.aside)
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _remember_target(item: Item) -> None:
    item.dst_size = item.dst.stat().st_size
    item.dst_sha256 = sha256_of(item.dst)


def _plan_database_conflict(item: Item) -> None:
    """The target already has a different tag_frequency.db: replace it when it is the built-in's empty placeholder."""
    pending = _journal(item.dst)
    if pending is not None:   # not even opened: a hot journal would make sqlite try to roll it back
        item.action = "conflict"
        item.note = f"the one there has an unfinished write ({pending.name}) - stop Forge and run again"
        return
    try:
        version, rows = database_summary(item.dst)
    except sqlite3.Error as error:
        item.action, item.note = "conflict", f"a different file with this name is already there - left as it is ({error})"
        return
    try:
        source_version, source_rows = database_summary(item.src)
    except sqlite3.Error as error:
        item.action, item.note = "blocked", f"not a readable tag_frequency.db: {error}"
        return
    if rows == 0 and version == source_version:
        item.action = "replace"
        item.note = (f"the one there is empty (0 rows, schema {version} - the built-in made it when it first started): "
                     f"it is renamed aside, then the old one (schema {source_version}, {source_rows} rows) is copied")
        _remember_target(item)
        return
    item.action = "conflict"
    if rows == 0:
        item.note = (f"the one there is empty but has schema {version} (the old one: {source_version}) - left as it is; "
                     f"stop Forge, move {item.dst} aside and run again to use the old one")
    else:
        item.note = (f"a different file with this name is already there - left as it is (it has {rows} rows, schema "
                     f"{version}; the old one is not merged in). To use the old counts ({source_rows} rows) instead: "
                     f"stop Forge, move {item.dst} aside and run again")


def plan(items: list[Item], progress: dict) -> None:
    for item in items:
        if item.name == DATABASE and _journal(item.src) is not None:
            item.action, item.note = "blocked", f"unfinished write ({_journal(item.src).name}) - stop Forge and run again"
            continue
        if item.dst.exists():
            same = item.dst.is_file() and item.dst.stat().st_size == item.size and sha256_of(item.dst) == item.sha256
            if same:
                item.action = "done" if progress.get(item.name) == item.sha256 else "identical"
            elif item.name == DATABASE and item.dst.is_file():
                _plan_database_conflict(item)
            elif item.name == CACHE and item.dst.is_file() and item.dst.stat().st_size == 0:
                item.action = "replace"
                item.note = ("the one there is empty (the built-in made it when it first started): it is renamed aside, "
                             "then the old one is copied")
                _remember_target(item)
            elif item.name == CACHE and item.dst.is_file():
                item.action = "kept"
                item.note = "a different hash cache is already there - kept (a cache the helper rebuilds; nothing is lost)"
            else:
                item.action = "conflict"
                item.note = "a different file with this name is already there - left as it is"
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


def _aside_path(path: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.name}.empty-{stamp}.bak")
    if candidate.exists():
        candidate = path.with_name(f"{path.name}.empty-{stamp}-{uuid.uuid4().hex[:8]}.bak")
    return candidate


def replace_one(item: Item) -> None:
    """Rename the placeholder the plan saw aside (never deleted, never overwriting), then copy; after a failed copy the
    placeholder goes back. Raises FileExistsError when the target is no longer that placeholder."""
    try:
        unchanged = (item.dst.is_file() and item.dst.stat().st_size == item.dst_size
                     and sha256_of(item.dst) == item.dst_sha256)
    except OSError:
        unchanged = False
    if not unchanged:
        raise FileExistsError(f"{item.dst} changed since the plan")
    if item.name == DATABASE and _journal(item.dst) is not None:
        raise FileExistsError(f"{item.dst} has an unfinished write next to it - stop Forge and run again")
    aside = _aside_path(item.dst)
    _place(item.dst, aside)
    try:
        copy_one(item)
    except BaseException:
        if not item.dst.exists():
            _place(aside, item.dst)
        raise
    item.aside = aside


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
            if item.action not in ("copy", "replace"):
                continue
            replacing = item.action == "replace"
            if replacing:
                item.note = ""
            try:
                if replacing:
                    replace_one(item)
                else:
                    copy_one(item)
            except FileExistsError as error:
                item.action = "conflict"
                item.note = (f"{error} - left as it is" if replacing
                             else "appeared in the target while copying - left as it is")
                _print(f"  conflict  {item.name}: {item.note}")
            except (OSError, sqlite3.Error) as error:
                item.action, item.note = "failed", str(error)
                failed.append(item)
            else:
                item.action = "replaced" if replacing else "copied"
                append_progress(progress_path, item)
                details = "sha256 verified" + (", " + item.note if item.note else "")
                if replacing:
                    details += f"; the empty one is now {item.aside}"
                _print(f"  {item.action:<9} {item.name} -> {item.dst}  ({details})")
        for item in items:
            if item.action == "identical":
                append_progress(progress_path, item)
        for item in failed:
            _print(f"  FAILED    {item.name}: {item.note}")

    counts = {}
    for item in items:
        counts[item.action] = counts.get(item.action, 0) + 1
    to_copy = [item for item in items if item.action in ("copy", "copied", "replace", "replaced")]
    summary = ", ".join(f"{key} {value}" for key, value in sorted(counts.items()))
    _print(f"[migrate_tagcomplete] {len(items)} files: {summary}; "
           f"{'copied' if apply else 'to copy'} {_mib(sum(i.size for i in to_copy))} into {dst}")
    if any(item.action in ("conflict", "blocked", "failed") for item in items):
        return 2
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# --set-tagfile: tac_tagFile in Forge's config.json
# ---------------------------------------------------------------------------------------------------------------------

TAG_FILE_KEY = "tac_tagFile"
DEFAULT_FORGE_PORTS = (7860,)
_JSON_SPACE = " \t\n\r"
_BOM = "﻿"


class ConfigError(ValueError):
    pass


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index] in _JSON_SPACE:
        index += 1
    return index


def top_level_members(text: str) -> list[tuple[str, int, int, int]]:
    """``(key, key start, value start, value end)`` of the top-level object's members, in file order (offsets into
    ``text``; ``key start`` is the opening quote). Raises ConfigError when ``text`` is not one JSON object."""
    decoder = json.JSONDecoder()
    index = _skip_space(text, 1 if text.startswith(_BOM) else 0)
    if index >= len(text) or text[index] != "{":
        raise ConfigError("not a JSON object")
    index = _skip_space(text, index + 1)
    members = []
    try:
        if index < len(text) and text[index] == "}":
            index += 1
        else:
            while True:
                if index >= len(text) or text[index] != '"':
                    raise ConfigError(f"expected a key at offset {index}")
                key, after_key = json.decoder.scanstring(text, index + 1)
                colon = _skip_space(text, after_key)
                if colon >= len(text) or text[colon] != ":":
                    raise ConfigError(f"expected ':' at offset {colon}")
                start = _skip_space(text, colon + 1)
                _value, end = decoder.raw_decode(text, start)
                members.append((key, index, start, end))
                index = _skip_space(text, end)
                if index < len(text) and text[index] == ",":
                    index = _skip_space(text, index + 1)
                    continue
                if index < len(text) and text[index] == "}":
                    index += 1
                    break
                raise ConfigError(f"expected ',' or '}}' at offset {index}")
    except json.JSONDecodeError as error:
        raise ConfigError(str(error)) from None
    if _skip_space(text, index) != len(text):
        raise ConfigError(f"text after the JSON object at offset {index}")
    return members


def with_tag_file(text: str, name: str) -> tuple[str, str | None]:
    """``(new text, old value or None)`` — ``text`` with only the value of the top-level ``tac_tagFile`` replaced by
    ``name`` (every other character as it was), or, without that key, with it added as the last member in the style
    of the member before it (its line break, indentation and ``": "``)."""
    members = top_level_members(text)
    found = [member for member in members if member[0] == TAG_FILE_KEY]
    if len(found) > 1:
        raise ConfigError(f"{TAG_FILE_KEY} appears {len(found)} times - fix config.json by hand")
    value = json.dumps(name, ensure_ascii=False)   # as Forge writes config.json (json.dump, ensure_ascii=False)
    if found:
        _key, _key_start, start, end = found[0]
        old = json.loads(text[start:end])
        if not isinstance(old, str):
            raise ConfigError(f"{TAG_FILE_KEY} is not a string there: {text[start:end]}")
        return text[:start] + value + text[end:], old
    if not members:
        raise ConfigError("it has no settings yet - start Forge once so it writes them, then run this again")
    _key, key_start, start, end = members[-1]
    lead = key_start
    while lead > 0 and text[lead - 1] in _JSON_SPACE:
        lead -= 1
    # the key's own end: the separator is what lies between it and the value (e.g. ': ')
    _name, after_key = json.decoder.scanstring(text, key_start + 1)
    member = text[lead:key_start] + json.dumps(TAG_FILE_KEY) + text[after_key:start] + value
    return text[:end] + "," + member + text[end:], None


def forge_listening(ports, host: str = "127.0.0.1", timeout: float = 0.5) -> list[int]:
    """The ports in ``ports`` where something accepts a TCP connection on ``host``."""
    busy = []
    for port in ports:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                busy.append(port)
        except OSError:
            continue
    return busy


def check_tag_file_name(name: str, tags: Path) -> str | None:
    """Why ``name`` cannot be set (None when it can): a bare .csv file name that exists in ``tags``."""
    if not name or name in (".", "..") or any(sep in name for sep in ("/", "\\", ":")) or name != name.strip():
        return f"--set-tagfile takes a file name in {tags}, not a path: {name!r}"
    if not name.lower().endswith(".csv"):
        return f"--set-tagfile takes a .csv tag list (Settings -> Tag Autocomplete lists the *.csv in {tags}): {name!r}"
    if not (tags / name).is_file():
        return f"{name} is not in {tags} - the setting would name a missing file"
    return None


def _backup_path(config: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = config.with_name(f"{config.name}.bak-tagfile-{stamp}")
    if candidate.exists():
        candidate = config.with_name(f"{config.name}.bak-tagfile-{stamp}-{uuid.uuid4().hex[:8]}")
    return candidate


def _write_new(path: Path, data: bytes) -> None:
    """``data`` under a temporary name next to ``path`` (checked), then renamed over ``path``."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.partial")
    try:
        with open(temporary, "xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.read_bytes() != data:
            raise OSError(f"{temporary} does not read back as written")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _quote(value) -> str:
    return "(not set)" if value is None else repr(value)


def set_tag_file(config: Path, name: str, tags: Path, apply: bool, forge_stopped: bool, ports) -> int:
    config, tags = Path(os.path.abspath(config)), Path(os.path.abspath(tags))
    _print(f"[migrate_tagcomplete] SET {TAG_FILE_KEY} - {'APPLY' if apply else 'DRY RUN (add --apply --forge-stopped to write)'}")
    _print(f"  config: {config}")
    _print(f"  tags:   {tags}")

    def abort(reason: str) -> int:
        _print(f"[migrate_tagcomplete] ABORT: {reason} - nothing changed")
        return 1

    problem = check_tag_file_name(name, tags)
    if problem:
        return abort(problem)
    if not config.is_file():
        return abort(f"--config is not a file: {config}")
    original = config.read_bytes()
    try:
        text = original.decode("utf-8")
        new_text, old = with_tag_file(text, name)
        before, after = json.loads(text.lstrip(_BOM)), json.loads(new_text.lstrip(_BOM))
    except (UnicodeDecodeError, ValueError) as error:
        return abort(f"{config} cannot be edited safely: {error}")
    expected = dict(before)
    expected[TAG_FILE_KEY] = name
    if after != expected or list(after) != list(expected):
        return abort("internal check failed: the edit would change more than tac_tagFile")
    _print(f"  {TAG_FILE_KEY}: {_quote(old)} -> {name!r}")
    if old == name:
        _print(f"[migrate_tagcomplete] {TAG_FILE_KEY} is already {name!r} - nothing to change")
        return 0
    if apply and not forge_stopped:
        return abort("--apply needs --forge-stopped: stop Forge first (it writes every setting it holds back into "
                     "config.json when settings are saved, which would undo this edit), then confirm with --forge-stopped")
    busy = forge_listening(ports)
    if busy:
        where = ", ".join(f"127.0.0.1:{port}" for port in busy)
        return abort(f"Forge seems to be running ({where} answers) - stop it first (another program on that port: "
                     "pass the ports this Forge uses with --forge-port)")
    pending = _journal(tags / DATABASE)
    if pending is not None:
        return abort(f"{pending} exists (an unfinished write to the usage counts) - Forge seems to be running; stop it first")
    data = new_text.encode("utf-8")
    backup = _backup_path(config)
    if not apply:
        _print(f"  would write the backup {backup} ({len(original)} bytes), then config.json with only this value "
               f"changed ({len(original)} -> {len(data)} bytes)")
        _print("[migrate_tagcomplete] DRY RUN - add --apply --forge-stopped to write (Forge must be stopped)")
        return 0
    try:
        with open(backup, "xb") as handle:
            handle.write(original)
        shutil.copystat(config, backup)
        if backup.read_bytes() != original:
            raise OSError(f"{backup} does not read back as config.json")
    except OSError as error:
        return abort(f"could not write the backup: {error}")
    _print(f"  backup: {backup} (sha256 {hashlib.sha256(original).hexdigest()[:16]}, verified)")
    if config.read_bytes() != original:
        return abort(f"{config} changed while this ran (is Forge running?) - the backup is the earlier version")
    try:
        _write_new(config, data)
    except OSError as error:
        return abort(f"could not write {config}: {error} (the backup is {backup})")
    if config.read_bytes() != data:
        return abort(f"{config} does not read back as written - restore it from {backup}")
    _print(f"[migrate_tagcomplete] {TAG_FILE_KEY}: {_quote(old)} -> {name!r} written to {config} "
           f"(backup: {backup}); start Forge to use it")
    return 0


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"not a TCP port: {value!r}")
    return port


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, default=None, help="the old extension folder (e.g. extensions/sd-webui-tagcomplete-neo)")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help=f"sam-extra's tags folder (default: {DEFAULT_DST})")
    parser.add_argument("--progress", type=Path, default=None, help="JSONL file of finished files (appended)")
    parser.add_argument("--apply", action="store_true", help="copy / write (without it: only show the plan)")
    parser.add_argument("--no-cache", action="store_true", help="leave temp/known_lora_hashes.txt (a rebuildable cache)")
    parser.add_argument("--set-tagfile", metavar="NAME", default=None,
                        help=f"instead of copying: set {TAG_FILE_KEY} in --config to NAME, a .csv in --dst")
    parser.add_argument("--config", type=Path, default=None, help="Forge's config.json (with --set-tagfile)")
    parser.add_argument("--forge-stopped", action="store_true",
                        help="with --set-tagfile --apply: confirms that Forge is not running")
    parser.add_argument("--forge-port", type=_port, action="append", default=None, metavar="PORT",
                        help="with --set-tagfile: a port this Forge listens on, refused while 127.0.0.1:PORT answers "
                             f"(repeatable; default {', '.join(map(str, DEFAULT_FORGE_PORTS))})")
    args = parser.parse_args(argv)
    if args.set_tagfile is not None:
        if args.src is not None:
            parser.error("--set-tagfile and --src are separate runs: copy the files first, then set the tag file")
        if args.config is None:
            parser.error("--set-tagfile needs --config <forge>/config.json")
        ports = args.forge_port if args.forge_port else list(DEFAULT_FORGE_PORTS)
        return set_tag_file(args.config, args.set_tagfile, args.dst, args.apply, args.forge_stopped, ports)
    if args.src is None:
        parser.error("--src is required (or --set-tagfile NAME --config <forge>/config.json)")
    if args.config is not None or args.forge_stopped or args.forge_port:
        parser.error("--config, --forge-stopped and --forge-port go with --set-tagfile")
    return run(args.src, args.dst, args.apply, not args.no_cache, args.progress)


if __name__ == "__main__":
    sys.exit(main())
