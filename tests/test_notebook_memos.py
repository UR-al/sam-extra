from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from sam3ext import notebook_memos
from sam3ext.notebook_memos import (
    MAX_MEMOS,
    MEMO_API_PATH,
    MemoCapacityError,
    MemoConflictError,
    MemoCorruptError,
    MemoNotFoundError,
    MemoSchemaError,
    MemoStore,
    parse_timestamp,
    register_memo_routes,
)
from sam3ext.notebook_store import (
    NOTEBOOK_API_PATH,
    NotebookStore,
    register_notebook_routes,
)

HEADERS = {"X-SAM3-Notebook": "1"}


def iso(delta: timedelta = timedelta(0)) -> str:
    """A JS-style ``toISOString()`` value relative to now."""

    moment = datetime.now(UTC) + delta
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def stored_memo(memo_id: str, *, deleted: bool = False, minutes_ago: int = 60) -> dict:
    stamp = iso(-timedelta(minutes=minutes_ago))
    return {
        "id": memo_id,
        "title": memo_id,
        "text": "" if deleted else f"text of {memo_id}",
        "created_at": stamp,
        "updated_at": stamp,
        "deleted": deleted,
    }


class MemoStoreTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.path = Path(self._directory.name) / "memos.json"
        self.store = MemoStore(self.path)

    def write_raw(self, memos: list[dict], *, revision: int = 7) -> None:
        self.path.write_text(
            json.dumps({"schema_version": 1, "revision": revision, "memos": memos}),
            encoding="utf-8",
        )

    def test_upsert_creates_memo_and_keeps_client_timestamp_verbatim(self):
        stamp = iso(-timedelta(seconds=5))
        result = self.store.upsert(
            "memo-1",
            {"title": "제목", "text": "본문", "updated_at": stamp, "base_updated_at": None},
        )

        self.assertEqual(result["revision"], 1)
        memo = result["memo"]
        self.assertEqual(memo["id"], "memo-1")
        self.assertEqual(memo["title"], "제목")
        self.assertEqual(memo["text"], "본문")
        self.assertEqual(memo["updated_at"], stamp)
        self.assertEqual(memo["created_at"], stamp)
        self.assertIs(memo["deleted"], False)

        on_disk = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["schema_version"], 1)
        self.assertEqual(on_disk["revision"], 1)
        self.assertEqual(on_disk["memos"][0]["text"], "본문")
        # Compact UTF-8, like notebook.json.
        self.assertIn("본문", self.path.read_text(encoding="utf-8"))
        self.assertNotIn("\n  ", self.path.read_text(encoding="utf-8"))

    def test_optional_created_at_is_kept_for_new_memos_only(self):
        created = iso(-timedelta(days=2))
        first = self.store.upsert(
            "memo-1", {"title": "a", "text": "", "created_at": created}
        )["memo"]
        self.assertEqual(first["created_at"], created)
        second = self.store.upsert(
            "memo-1",
            {"title": "b", "text": "", "created_at": iso(), "base_updated_at": first["updated_at"]},
        )["memo"]
        self.assertEqual(second["created_at"], created)

    def test_missing_or_bad_client_timestamp_uses_server_now(self):
        before = datetime.now(UTC)
        for value in (None, "", "not a date", "2026-09-25T05:00:00", 12345):
            memo = self.store.upsert(
                "memo-x", {"title": "t", "text": "", "updated_at": value}
            )["memo"]
            self.assertGreaterEqual(parse_timestamp(memo["updated_at"]), before)

    def test_far_future_client_timestamp_is_not_trusted(self):
        memo = self.store.upsert(
            "memo-1", {"title": "t", "text": "", "updated_at": iso(timedelta(days=3))}
        )["memo"]
        self.assertLess(
            parse_timestamp(memo["updated_at"]),
            datetime.now(UTC) + timedelta(minutes=1),
        )

    def test_non_utc_offset_is_converted_to_utc(self):
        local = (datetime.now(UTC) - timedelta(minutes=1)).astimezone(
            timezone(timedelta(hours=9))
        )
        memo = self.store.upsert(
            "memo-1", {"title": "t", "text": "", "updated_at": local.isoformat()}
        )["memo"]
        self.assertTrue(memo["updated_at"].endswith("+00:00"))
        self.assertEqual(parse_timestamp(memo["updated_at"]), local.astimezone(UTC))

    def test_edit_never_moves_updated_at_backwards(self):
        first = self.store.upsert(
            "memo-1", {"title": "t", "text": "", "updated_at": iso()}
        )["memo"]
        older = iso(-timedelta(minutes=5))
        second = self.store.upsert(
            "memo-1",
            {"title": "t2", "text": "", "updated_at": older, "base_updated_at": first["updated_at"]},
        )["memo"]
        self.assertGreater(
            parse_timestamp(second["updated_at"]),
            parse_timestamp(first["updated_at"]),
        )

    def test_list_is_newest_first_and_hides_tombstones_unless_asked(self):
        self.write_raw(
            [
                stored_memo("old", minutes_ago=30),
                stored_memo("new", minutes_ago=1),
                stored_memo("gone", deleted=True, minutes_ago=2),
            ]
        )

        listed = self.store.list_memos()
        self.assertEqual(listed["revision"], 7)
        self.assertEqual([memo["id"] for memo in listed["memos"]], ["new", "old"])
        everything = self.store.list_memos(include_deleted=True)
        self.assertEqual(
            [memo["id"] for memo in everything["memos"]], ["new", "gone", "old"]
        )
        self.assertIs(everything["memos"][1]["deleted"], True)

    def test_list_orders_by_instant_not_by_string(self):
        # 10:00+09:00 is 01:00Z, older than 02:00Z even though it sorts later.
        self.write_raw(
            [
                {**stored_memo("kst"), "updated_at": "2026-09-25T10:00:00+09:00"},
                {**stored_memo("utc"), "updated_at": "2026-09-25T02:00:00Z"},
            ]
        )
        self.assertEqual(
            [memo["id"] for memo in self.store.list_memos()["memos"]], ["utc", "kst"]
        )

    def test_matching_base_saves_and_stale_base_conflicts_with_stored_copy(self):
        first = self.store.upsert("memo-1", {"title": "a", "text": "one"})["memo"]
        second = self.store.upsert(
            "memo-1",
            {"title": "a", "text": "two", "base_updated_at": first["updated_at"]},
        )
        self.assertEqual(second["revision"], 2)
        self.assertEqual(second["memo"]["text"], "two")

        with self.assertRaises(MemoConflictError) as caught:
            self.store.upsert(
                "memo-1",
                {"title": "a", "text": "stale", "base_updated_at": first["updated_at"]},
            )
        self.assertEqual(caught.exception.memo["text"], "two")
        self.assertEqual(self.store.list_memos()["memos"][0]["text"], "two")
        self.assertEqual(self.store.load()["revision"], 2)

    def test_base_in_other_format_but_same_instant_is_not_a_conflict(self):
        first = self.store.upsert(
            "memo-1", {"title": "a", "text": "", "updated_at": "2026-09-25T02:00:00.000Z"}
        )["memo"]
        self.assertEqual(first["updated_at"], "2026-09-25T02:00:00.000Z")
        saved = self.store.upsert(
            "memo-1",
            {"title": "b", "text": "", "base_updated_at": "2026-09-25T11:00:00+09:00"},
        )
        self.assertEqual(saved["memo"]["title"], "b")

    def test_null_base_is_an_unconditional_upsert(self):
        self.store.upsert("memo-1", {"title": "a", "text": "one"})
        saved = self.store.upsert(
            "memo-1", {"title": "a", "text": "forced", "base_updated_at": None}
        )
        self.assertEqual(saved["memo"]["text"], "forced")

    def test_missing_fields_keep_stored_values(self):
        self.store.upsert("memo-1", {"title": "keep me", "text": "body"})
        saved = self.store.upsert("memo-1", {"text": "new body"})["memo"]
        self.assertEqual(saved["title"], "keep me")
        self.assertEqual(saved["text"], "new body")

    def test_invalid_ids_and_fields_are_rejected(self):
        for bad_id in ("", "-starts-with-dash", "has space", "x" * 81, "슬래시"):
            with self.assertRaisesRegex(ValueError, "Memo id"):
                self.store.upsert(bad_id, {"title": "", "text": ""})
        with self.assertRaisesRegex(ValueError, "must be an object"):
            self.store.upsert("memo-1", ["not", "an", "object"])
        with self.assertRaisesRegex(ValueError, "character limit"):
            self.store.upsert("memo-1", {"title": "t" * 121, "text": ""})
        with self.assertRaisesRegex(ValueError, "character limit"):
            self.store.upsert("memo-1", {"title": "", "text": "x" * 100_001})
        with self.assertRaisesRegex(ValueError, "must be a string"):
            self.store.upsert("memo-1", {"title": {"nested": 1}, "text": ""})
        with self.assertRaisesRegex(ValueError, "base_updated_at"):
            self.store.upsert("memo-1", {"title": "", "text": "", "base_updated_at": "yesterday"})
        self.assertFalse(self.path.exists())
        # The limits themselves are accepted.
        saved = self.store.upsert("A.b_c:d-9", {"title": "t" * 120, "text": "x" * 100_000})
        self.assertEqual(len(saved["memo"]["text"]), 100_000)

    def test_delete_leaves_tombstone_newer_than_the_memo(self):
        memo = self.store.upsert(
            "memo-1", {"title": "title stays", "text": "secret", "updated_at": iso(timedelta(minutes=5))}
        )["memo"]
        deleted = self.store.delete("memo-1")

        self.assertEqual(deleted["revision"], 2)
        tombstone = deleted["memo"]
        self.assertIs(tombstone["deleted"], True)
        self.assertEqual(tombstone["text"], "")
        self.assertEqual(tombstone["title"], "title stays")
        self.assertGreater(
            parse_timestamp(tombstone["updated_at"]),
            parse_timestamp(memo["updated_at"]),
        )
        self.assertEqual(self.store.list_memos()["memos"], [])
        self.assertNotIn("secret", self.path.read_text(encoding="utf-8"))

    def test_repeated_delete_is_idempotent_and_unknown_id_is_not_found(self):
        self.store.upsert("memo-1", {"title": "", "text": ""})
        first = self.store.delete("memo-1")
        again = self.store.delete("memo-1")
        self.assertEqual(again, first)
        self.assertEqual(self.store.load()["revision"], 2)
        with self.assertRaises(MemoNotFoundError):
            self.store.delete("memo-unknown")

    def test_tombstone_conflicts_with_stale_base_and_null_base_revives(self):
        memo = self.store.upsert("memo-1", {"title": "t", "text": "one"})["memo"]
        self.store.delete("memo-1")

        with self.assertRaises(MemoConflictError) as caught:
            self.store.upsert(
                "memo-1", {"title": "t", "text": "edit", "base_updated_at": memo["updated_at"]}
            )
        self.assertIs(caught.exception.memo["deleted"], True)

        revived = self.store.upsert("memo-1", {"title": "t", "text": "back"})["memo"]
        self.assertIs(revived["deleted"], False)
        self.assertEqual(revived["created_at"], memo["created_at"])
        self.assertEqual([m["id"] for m in self.store.list_memos()["memos"]], ["memo-1"])

    def test_delete_with_stale_base_keeps_the_newer_edit(self):
        seen = self.store.upsert("memo-1", {"title": "t", "text": "v1"})["memo"]
        newer = self.store.upsert(
            "memo-1", {"title": "t", "text": "newer edit", "base_updated_at": seen["updated_at"]}
        )["memo"]

        with self.assertRaises(MemoConflictError) as caught:
            self.store.delete("memo-1", seen["updated_at"])
        self.assertEqual(caught.exception.memo["text"], "newer edit")
        self.assertEqual(self.store.list_memos()["memos"][0]["text"], "newer edit")
        self.assertEqual(self.store.load()["revision"], 2)

        with self.assertRaises(ValueError):
            self.store.delete("memo-1", "yesterday")
        # The version the client saw (or no base at all) deletes as before.
        tombstone = self.store.delete("memo-1", newer["updated_at"])["memo"]
        self.assertIs(tombstone["deleted"], True)
        # Deleting an existing tombstone with a stale base stays idempotent.
        self.assertEqual(self.store.delete("memo-1", seen["updated_at"])["memo"], tombstone)

    def test_lone_surrogates_are_replaced_instead_of_failing_the_write(self):
        saved = self.store.upsert("memo-1", {"title": "cut \ud83d", "text": "a\udc00b \U0001F600"})
        self.assertEqual(saved["memo"]["title"], "cut �")
        self.assertEqual(saved["memo"]["text"], "a�b \U0001F600")
        stored = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(stored["memos"][0]["title"], "cut �")

    def test_oldest_tombstones_are_pruned_to_make_room(self):
        memos = [stored_memo(f"live-{i}") for i in range(MAX_MEMOS - 2)]
        memos.append(stored_memo("tomb-old", deleted=True, minutes_ago=500))
        memos.append(stored_memo("tomb-new", deleted=True, minutes_ago=5))
        self.write_raw(memos)

        self.store.upsert("fresh", {"title": "", "text": ""})

        ids = {memo["id"] for memo in self.store.load()["memos"]}
        self.assertEqual(len(ids), MAX_MEMOS)
        self.assertIn("fresh", ids)
        self.assertNotIn("tomb-old", ids)
        self.assertIn("tomb-new", ids)

    def test_full_store_refuses_new_memos_but_allows_edits(self):
        self.write_raw([stored_memo(f"live-{i}") for i in range(MAX_MEMOS)])

        with self.assertRaises(MemoCapacityError):
            self.store.upsert("one-too-many", {"title": "", "text": ""})
        self.assertEqual(self.store.load()["revision"], 7)

        edited = self.store.upsert("live-3", {"title": "edited", "text": ""})
        self.assertEqual(edited["memo"]["title"], "edited")

    def test_store_byte_limit_is_enforced_before_writing(self):
        with mock.patch.object(notebook_memos, "MAX_MEMO_STORE_BYTES", 200):
            with self.assertRaises(MemoCapacityError):
                self.store.upsert("memo-1", {"title": "", "text": "x" * 500})
        self.assertFalse(self.path.exists())

    def test_atomic_save_keeps_previous_revision_as_backup(self):
        self.store.upsert("memo-1", {"title": "", "text": "first"})
        self.store.upsert("memo-1", {"title": "", "text": "second"})

        self.assertTrue(self.store.backup_path.exists())
        backup = json.loads(self.store.backup_path.read_text(encoding="utf-8"))
        self.assertEqual(backup["revision"], 1)
        self.assertEqual(backup["memos"][0]["text"], "first")
        leftovers = [p.name for p in self.path.parent.iterdir() if p.suffix == ".tmp"]
        self.assertEqual(leftovers, [])

    def test_corrupt_primary_falls_back_to_backup(self):
        self.store.upsert("memo-1", {"title": "", "text": "safe"})
        self.store.upsert("memo-1", {"title": "", "text": "latest"})
        self.path.write_text("{broken", encoding="utf-8")

        self.assertEqual(self.store.list_memos()["memos"][0]["text"], "safe")
        # Saving from the recovered copy must not replace the good backup.
        self.store.upsert("memo-2", {"title": "", "text": ""})
        backup = json.loads(self.store.backup_path.read_text(encoding="utf-8"))
        self.assertEqual(backup["memos"][0]["text"], "safe")

    def test_damaged_primary_is_kept_aside_before_it_is_replaced(self):
        self.store.upsert("memo-1", {"title": "", "text": "safe"})
        self.store.upsert("memo-1", {"title": "", "text": "latest"})
        damaged = '{"schema_version":1,"revision":2,"memos":[{"id":"memo-1","text":"latest"'
        self.path.write_text(damaged, encoding="utf-8")

        self.store.upsert("memo-2", {"title": "", "text": ""})
        self.store.upsert("memo-2", {"title": "", "text": "again"})

        kept = sorted(self.path.parent.glob("memos.json.corrupt-*"))
        self.assertEqual(len(kept), 1)   # only the first write after the fallback
        self.assertEqual(kept[0].read_text(encoding="utf-8"), damaged)
        self.assertEqual(self.store.list_memos()["memos"][0]["text"], "again")

    def test_failed_copy_of_damaged_primary_aborts_the_write(self):
        self.store.upsert("memo-1", {"title": "", "text": "safe"})
        self.store.upsert("memo-1", {"title": "", "text": "latest"})
        self.path.write_text("{broken", encoding="utf-8")

        with mock.patch.object(notebook_memos.shutil, "copy2", side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                self.store.upsert("memo-2", {"title": "", "text": ""})
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{broken")

    def test_corrupt_primary_without_backup_is_not_overwritten(self):
        self.path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(MemoCorruptError):
            self.store.list_memos()
        with self.assertRaises(MemoCorruptError):
            self.store.upsert("memo-1", {"title": "", "text": ""})
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{broken")

    def test_future_schema_is_not_downgraded(self):
        future = {"schema_version": 2, "revision": 3, "memos": []}
        self.path.write_text(json.dumps(future), encoding="utf-8")
        with self.assertRaises(MemoSchemaError):
            self.store.list_memos()
        with self.assertRaises(MemoSchemaError):
            self.store.upsert("memo-1", {"title": "", "text": ""})
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), future)

    def test_loaded_duplicates_and_bad_ids_keep_their_text(self):
        self.write_raw(
            [
                stored_memo("same"),
                {**stored_memo("same"), "text": "second copy"},
                {**stored_memo("x"), "id": "bad id with spaces", "text": "orphan"},
            ]
        )
        memos = self.store.load()["memos"]
        self.assertEqual(len({memo["id"] for memo in memos}), 3)
        self.assertIn("second copy", [memo["text"] for memo in memos])
        self.assertIn("orphan", [memo["text"] for memo in memos])

    def test_memo_without_timestamps_reads_the_same_every_time(self):
        self.write_raw([{"id": "memo-1", "title": "hand edited", "text": "x"}])
        first = self.store.list_memos()["memos"][0]
        second = self.store.list_memos()["memos"][0]
        self.assertEqual(first, second)
        self.assertEqual(parse_timestamp(first["updated_at"]).year, 1970)

    def test_default_path_sits_next_to_notebook_json(self):
        self.assertEqual(
            notebook_memos.default_memo_path().name, notebook_memos.MEMO_FILE_NAME
        )
        self.assertEqual(
            notebook_memos.default_memo_path().parent,
            notebook_memos.default_notebook_path().parent,
        )
        self.assertEqual(
            notebook_memos.memo_path_for_notebook(NotebookStore(self.path.parent / "notebook.json")),
            self.path,
        )


class MemoRouteTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)

    def client_for(self, app: FastAPI) -> TestClient:
        client = TestClient(app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def notebook_app(self) -> FastAPI:
        app = FastAPI()
        register_notebook_routes(app, NotebookStore(self.directory / "notebook.json"))
        return app

    def test_notebook_registration_adds_memo_routes_next_to_notebook_json(self):
        app = self.notebook_app()
        memo_routes = [
            (route.path, sorted(route.methods))
            for route in app.routes
            if getattr(route, "path", "").startswith(MEMO_API_PATH)
        ]
        self.assertEqual(
            sorted(memo_routes),
            [
                (MEMO_API_PATH, ["GET"]),
                (MEMO_API_PATH + "/{memo_id}", ["DELETE"]),
                (MEMO_API_PATH + "/{memo_id}", ["PUT"]),
            ],
        )
        # A second call registers nothing new.
        self.assertFalse(
            register_notebook_routes(app, NotebookStore(self.directory / "notebook.json"))
        )
        self.assertFalse(register_memo_routes(app))
        self.assertEqual(
            len([r for r in app.routes if getattr(r, "path", "").startswith(MEMO_API_PATH)]),
            3,
        )

        client = self.client_for(app)
        response = client.put(
            MEMO_API_PATH + "/memo-1", headers=HEADERS, json={"title": "t", "text": "x"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue((self.directory / "memos.json").exists())
        # The preset Notebook revision is untouched by memo writes.
        notebook = client.get(NOTEBOOK_API_PATH, headers=HEADERS)
        self.assertEqual(notebook.json()["revision"], 0)

    def test_get_put_delete_round_trip(self):
        client = self.client_for(self.notebook_app())

        empty = client.get(MEMO_API_PATH, headers=HEADERS)
        self.assertEqual(empty.status_code, 200)
        self.assertEqual(empty.json(), {"revision": 0, "memos": []})
        self.assertIn("no-store", empty.headers["cache-control"])

        stamp = iso(-timedelta(seconds=3))
        created = client.put(
            MEMO_API_PATH + "/memo-1",
            headers=HEADERS,
            json={"title": "제목", "text": "본문", "updated_at": stamp, "base_updated_at": None},
        )
        self.assertEqual(created.status_code, 200)
        self.assertEqual(created.json()["revision"], 1)
        memo = created.json()["memo"]
        self.assertEqual(memo["updated_at"], stamp)
        self.assertEqual(
            set(memo), {"id", "title", "text", "created_at", "updated_at", "deleted"}
        )

        updated = client.put(
            MEMO_API_PATH + "/memo-1",
            headers=HEADERS,
            json={"title": "제목", "text": "둘", "base_updated_at": stamp},
        )
        self.assertEqual(updated.status_code, 200)

        conflict = client.put(
            MEMO_API_PATH + "/memo-1",
            headers=HEADERS,
            json={"title": "제목", "text": "낡은 편집", "base_updated_at": stamp},
        )
        self.assertEqual(conflict.status_code, 409)
        self.assertIn("detail", conflict.json())
        self.assertEqual(conflict.json()["memo"]["text"], "둘")

        listed = client.get(MEMO_API_PATH, headers=HEADERS).json()
        self.assertEqual(listed["revision"], 2)
        self.assertEqual([m["text"] for m in listed["memos"]], ["둘"])

        deleted = client.delete(MEMO_API_PATH + "/memo-1", headers=HEADERS)
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json()["revision"], 3)
        self.assertIs(deleted.json()["memo"]["deleted"], True)
        self.assertEqual(deleted.json()["memo"]["text"], "")

        self.assertEqual(client.get(MEMO_API_PATH, headers=HEADERS).json()["memos"], [])
        with_tombstones = client.get(
            MEMO_API_PATH, headers=HEADERS, params={"include_deleted": "1"}
        ).json()["memos"]
        self.assertEqual([m["id"] for m in with_tombstones], ["memo-1"])

        missing = client.delete(MEMO_API_PATH + "/memo-unknown", headers=HEADERS)
        self.assertEqual(missing.status_code, 404)

    def test_delete_base_query_refuses_to_wipe_an_unseen_edit(self):
        client = self.client_for(self.notebook_app())
        url = MEMO_API_PATH + "/memo-1"
        seen = client.put(url, headers=HEADERS, json={"title": "t", "text": "v1"}).json()["memo"]
        client.put(
            url,
            headers=HEADERS,
            json={"title": "t", "text": "newer edit from app", "base_updated_at": seen["updated_at"]},
        )

        stale = client.delete(url, headers=HEADERS, params={"base_updated_at": seen["updated_at"]})
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["memo"]["text"], "newer edit from app")
        self.assertIn("detail", stale.json())

        bogus = client.delete(url, headers=HEADERS, params={"base_updated_at": "soon"})
        self.assertEqual(bogus.status_code, 400)

        current = client.get(MEMO_API_PATH, headers=HEADERS).json()["memos"][0]["updated_at"]
        done = client.delete(url, headers=HEADERS, params={"base_updated_at": current})
        self.assertEqual(done.status_code, 200)
        self.assertIs(done.json()["memo"]["deleted"], True)

    def test_js_string_cut_inside_an_emoji_is_stored_not_rejected(self):
        client = self.client_for(self.notebook_app())
        body = b'{"title":"x\\ud83d (copy)","text":"t","base_updated_at":null}'
        response = client.put(
            MEMO_API_PATH + "/memo-copy",
            headers={**HEADERS, "Content-Type": "application/json"},
            content=body,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["memo"]["title"], "x� (copy)")

    def test_ids_with_dots_and_colons_route_correctly(self):
        client = self.client_for(self.notebook_app())
        response = client.put(
            MEMO_API_PATH + "/app:memo.2026-09-25_1", headers=HEADERS, json={"title": "", "text": ""}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["memo"]["id"], "app:memo.2026-09-25_1")

    def test_requests_without_notebook_header_are_refused(self):
        client = self.client_for(self.notebook_app())
        self.assertEqual(client.get(MEMO_API_PATH).status_code, 403)
        self.assertEqual(
            client.put(MEMO_API_PATH + "/memo-1", json={"title": "", "text": ""}).status_code,
            403,
        )
        self.assertEqual(client.delete(MEMO_API_PATH + "/memo-1").status_code, 403)
        self.assertFalse((self.directory / "memos.json").exists())

    def test_bad_requests_are_400_or_413(self):
        client = self.client_for(self.notebook_app())
        url = MEMO_API_PATH + "/memo-1"

        non_object = client.put(url, headers=HEADERS, json=[])
        self.assertEqual(non_object.status_code, 400)
        self.assertIn("must be an object", non_object.json()["detail"])

        broken = client.put(
            url, headers={**HEADERS, "Content-Type": "application/json"}, content=b"{broken"
        )
        self.assertEqual(broken.status_code, 400)

        bad_id = client.put(
            MEMO_API_PATH + "/-bad", headers=HEADERS, json={"title": "", "text": ""}
        )
        self.assertEqual(bad_id.status_code, 400)
        self.assertEqual(client.delete(MEMO_API_PATH + "/-bad", headers=HEADERS).status_code, 400)

        too_long = client.put(url, headers=HEADERS, json={"title": "t" * 121, "text": ""})
        self.assertEqual(too_long.status_code, 400)

        with mock.patch.object(notebook_memos, "MAX_MEMO_REQUEST_BYTES", 16):
            too_large = client.put(url, headers=HEADERS, json={"title": "", "text": "x" * 64})
        self.assertEqual(too_large.status_code, 413)
        self.assertFalse((self.directory / "memos.json").exists())

    def test_full_store_is_413(self):
        (self.directory / "memos.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "revision": 1,
                    "memos": [stored_memo(f"live-{i}") for i in range(MAX_MEMOS)],
                }
            ),
            encoding="utf-8",
        )
        client = self.client_for(self.notebook_app())
        response = client.put(
            MEMO_API_PATH + "/memo-new", headers=HEADERS, json={"title": "", "text": ""}
        )
        self.assertEqual(response.status_code, 413)

    def test_corrupt_and_future_storage_are_reported(self):
        path = self.directory / "memos.json"
        path.write_text("{broken", encoding="utf-8")
        client = self.client_for(self.notebook_app())
        response = client.get(MEMO_API_PATH, headers=HEADERS)
        self.assertEqual(response.status_code, 500)
        self.assertIn("damaged", response.json()["detail"])

        path.write_text(json.dumps({"schema_version": 9, "memos": []}), encoding="utf-8")
        self.assertEqual(client.get(MEMO_API_PATH, headers=HEADERS).status_code, 409)
        self.assertEqual(
            client.put(
                MEMO_API_PATH + "/memo-1", headers=HEADERS, json={"title": "", "text": ""}
            ).status_code,
            409,
        )

    def test_storage_oserrors_become_bounded_500_responses(self):
        class BrokenStore:
            def list_memos(self, *, include_deleted=False):
                raise OSError("private read path")

            def upsert(self, memo_id, payload):
                raise OSError("private write path")

            def delete(self, memo_id, base_updated_at=None):
                raise OSError("private delete path")

        app = FastAPI()
        register_memo_routes(app, BrokenStore())
        client = self.client_for(app)
        with self.assertLogs("sam3ext.notebook_memos", level="ERROR"):
            read = client.get(MEMO_API_PATH, headers=HEADERS)
        self.assertEqual(read.status_code, 500)
        self.assertEqual(read.json()["detail"], "Memo storage I/O failed while reading")
        self.assertNotIn("private", read.text)
        with self.assertLogs("sam3ext.notebook_memos", level="ERROR"):
            write = client.put(
                MEMO_API_PATH + "/memo-1", headers=HEADERS, json={"title": "", "text": ""}
            )
        self.assertEqual(write.status_code, 500)
        self.assertNotIn("private", write.text)
        with self.assertLogs("sam3ext.notebook_memos", level="ERROR"):
            delete = client.delete(MEMO_API_PATH + "/memo-1", headers=HEADERS)
        self.assertEqual(delete.status_code, 500)
        self.assertNotIn("private", delete.text)

    def test_routes_reuse_gradio_login_guard(self):
        app = FastAPI()

        @app.get("/login_check")
        def login_check(x_test_auth: str | None = Header(default=None)):
            if x_test_auth != "ok":
                raise HTTPException(status_code=401, detail="Not authenticated")

        register_notebook_routes(app, NotebookStore(self.directory / "notebook.json"))
        client = self.client_for(app)
        self.assertEqual(client.get(MEMO_API_PATH, headers=HEADERS).status_code, 401)
        self.assertEqual(
            client.put(
                MEMO_API_PATH + "/memo-1", headers=HEADERS, json={"title": "", "text": ""}
            ).status_code,
            401,
        )
        self.assertEqual(
            client.delete(MEMO_API_PATH + "/memo-1", headers=HEADERS).status_code, 401
        )
        authed = {**HEADERS, "X-Test-Auth": "ok"}
        self.assertEqual(client.get(MEMO_API_PATH, headers=authed).status_code, 200)
        self.assertEqual(
            client.put(
                MEMO_API_PATH + "/memo-1", headers=authed, json={"title": "", "text": ""}
            ).status_code,
            200,
        )

    def test_memo_registration_failure_keeps_notebook_routes(self):
        app = FastAPI()
        with mock.patch.object(
            notebook_memos, "register_memo_routes", side_effect=RuntimeError("boom")
        ):
            with self.assertLogs("sam3ext.notebook_memos", level="ERROR"):
                self.assertTrue(
                    register_notebook_routes(
                        app, NotebookStore(self.directory / "notebook.json")
                    )
                )
        client = self.client_for(app)
        self.assertEqual(client.get(NOTEBOOK_API_PATH, headers=HEADERS).status_code, 200)
        self.assertEqual(client.get(MEMO_API_PATH, headers=HEADERS).status_code, 404)


if __name__ == "__main__":
    unittest.main()
