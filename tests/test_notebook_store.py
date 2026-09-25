from __future__ import annotations

import ast
import contextlib
import json
import secrets
import sys
import tempfile
import types
import unittest
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.testclient import TestClient

from sam3ext.notebook_store import (
    NOTEBOOK_API_PATH,
    NotebookConflictError,
    NotebookCorruptError,
    NotebookSchemaError,
    NotebookStore,
    _forge_api_auth_dependencies,
    empty_notebook,
    register_notebook_routes,
    sanitize_notebook,
)

# <Forge>/extensions/forge_sam3_extension/tests/this file → <Forge>/modules/api/api.py
FORGE_API_PY = Path(__file__).resolve().parents[3] / "modules" / "api" / "api.py"
NOTEBOOK_HEADERS = {"X-SAM3-Notebook": "1"}


@contextlib.contextmanager
def forge_api_auth(value, *, loaded=True, api=True, nowebui=False):
    """Forge started with ``--api-auth value``: its loaded ``modules.shared.cmd_opts`` carries it.

    ``api``/``nowebui``: whether Forge built its API at all (``--api`` / ``--nowebui``).
    ``loaded=False``: no Forge at all. Only this one ``sys.modules`` entry is swapped.
    """

    missing = object()
    saved = sys.modules.get("modules.shared", missing)
    if loaded:
        sys.modules["modules.shared"] = types.SimpleNamespace(
            cmd_opts=types.SimpleNamespace(api_auth=value, api=api, nowebui=nowebui))
    else:
        sys.modules.pop("modules.shared", None)
    try:
        yield
    finally:
        if saved is missing:
            sys.modules.pop("modules.shared", None)
        else:
            sys.modules["modules.shared"] = saved


def forge_api_class(api_auth: str):
    """Forge's own ``Api.add_api_route`` and ``Api.auth``, compiled from modules/api/api.py as is."""

    tree = ast.parse(FORGE_API_PY.read_text(encoding="utf-8"))
    api = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    methods = [node for node in api.body
               if isinstance(node, ast.FunctionDef) and node.name in ("add_api_route", "auth")]
    module = ast.Module(body=[ast.ClassDef(name="ForgeApi", bases=[], keywords=[], body=methods,
                                           decorator_list=[], type_params=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Depends": Depends, "HTTPBasic": HTTPBasic, "HTTPBasicCredentials": HTTPBasicCredentials,
        "HTTPException": HTTPException, "compare_digest": secrets.compare_digest,
        "shared": types.SimpleNamespace(cmd_opts=types.SimpleNamespace(api_auth=api_auth)),
    }
    exec(compile(module, str(FORGE_API_PY), "exec"), namespace)
    return namespace["ForgeApi"]


class NotebookStoreTests(unittest.TestCase):
    def test_sanitize_keeps_supported_multi_entry_presets(self):
        payload = {
            "version": 1,
            "revision": "4",
            "presets": [
                {
                    "id": "preset-1",
                    "name": "  Test preset  ",
                    "entries": [
                        {
                            "id": "entry-x",
                            "target": "xyz_x",
                            "axis_type": "CFG Scale",
                            "value": "4, 5, 6",
                        },
                        {
                            "id": "entry-prompt",
                            "target": "prompt",
                            "value": "masterpiece",
                        },
                    ],
                }
            ],
        }

        result = sanitize_notebook(payload)

        self.assertEqual(result["version"], 1)
        self.assertEqual(result["revision"], 4)
        self.assertEqual(result["presets"][0]["name"], "Test preset")
        self.assertEqual(
            [entry["target"] for entry in result["presets"][0]["entries"]],
            ["xyz_x", "prompt"],
        )
        self.assertEqual(
            result["presets"][0]["entries"][0]["axis_type"],
            "CFG Scale",
        )

    def test_future_schema_and_unsupported_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema version"):
            sanitize_notebook({"version": 99, "presets": []})
        with self.assertRaisesRegex(ValueError, "Unsupported Notebook target"):
            sanitize_notebook(
                {
                    "version": 1,
                    "presets": [
                        {
                            "id": "one",
                            "name": "bad",
                            "entries": [
                                {
                                    "id": "bad-entry",
                                    "target": "unsupported",
                                    "value": "not silently dropped",
                                }
                            ],
                        }
                    ],
                }
            )

    def test_atomic_save_increments_revision_and_keeps_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notebook.json"
            store = NotebookStore(path)

            first = store.save(
                {
                    **empty_notebook(),
                    "presets": [
                        {"id": "one", "name": "preset1", "entries": []}
                    ],
                },
                expected_revision=0,
            )
            second_payload = json.loads(json.dumps(first))
            second_payload["presets"][0]["name"] = "renamed"
            second = store.save(
                second_payload,
                expected_revision=first["revision"],
            )

            self.assertEqual(first["revision"], 1)
            self.assertEqual(second["revision"], 2)
            self.assertEqual(store.load()["presets"][0]["name"], "renamed")
            self.assertTrue(store.backup_path.exists())
            backup = json.loads(store.backup_path.read_text(encoding="utf-8"))
            self.assertEqual(backup["revision"], 1)
            self.assertNotIn("\n  ", path.read_text(encoding="utf-8"))

    def test_stale_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = NotebookStore(Path(directory) / "notebook.json")
            store.save(empty_notebook(), expected_revision=0)

            with self.assertRaises(NotebookConflictError):
                store.save(empty_notebook(), expected_revision=0)

    def test_corrupt_primary_falls_back_to_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notebook.json"
            store = NotebookStore(path)
            first = store.save(
                {
                    **empty_notebook(),
                    "presets": [
                        {"id": "one", "name": "safe", "entries": []}
                    ],
                },
                expected_revision=0,
            )
            next_payload = json.loads(json.dumps(first))
            next_payload["presets"][0]["name"] = "latest"
            store.save(next_payload, expected_revision=1)
            path.write_text("{broken", encoding="utf-8")

            self.assertEqual(store.load()["presets"][0]["name"], "safe")

    def test_save_after_corrupt_primary_does_not_overwrite_good_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notebook.json"
            store = NotebookStore(path)
            first = store.save(
                {
                    **empty_notebook(),
                    "presets": [
                        {"id": "one", "name": "safe", "entries": []}
                    ],
                },
                expected_revision=0,
            )
            second_payload = json.loads(json.dumps(first))
            second_payload["presets"][0]["name"] = "newer"
            store.save(second_payload, expected_revision=1)
            path.write_text("{broken", encoding="utf-8")

            recovered = store.load()
            recovered["presets"][0]["name"] = "recovered"
            saved = store.save(recovered, expected_revision=1)

            backup = json.loads(store.backup_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["revision"], 2)
            self.assertEqual(saved["presets"][0]["name"], "recovered")
            self.assertEqual(backup["presets"][0]["name"], "safe")

    def test_corrupt_primary_without_backup_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notebook.json"
            path.write_text("{broken", encoding="utf-8")
            store = NotebookStore(path)

            with self.assertRaises(NotebookCorruptError):
                store.load()
            with self.assertRaises(NotebookCorruptError):
                store.save(empty_notebook(), expected_revision=0)
            self.assertEqual(path.read_text(encoding="utf-8"), "{broken")

    def test_newer_primary_schema_is_not_downgraded_from_old_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notebook.json"
            store = NotebookStore(path)
            store.save(empty_notebook(), expected_revision=0)
            store.save(
                {**empty_notebook(revision=1), "presets": []},
                expected_revision=1,
            )
            future = {
                "version": 2,
                "revision": 99,
                "updated_at": "",
                "presets": [],
            }
            path.write_text(json.dumps(future), encoding="utf-8")

            with self.assertRaises(NotebookSchemaError):
                store.load()
            with self.assertRaises(NotebookSchemaError):
                store.save(empty_notebook(), expected_revision=0)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), future)

    def test_routes_are_idempotent_and_support_get_put(self):
        with tempfile.TemporaryDirectory() as directory:
            app = FastAPI()
            store = NotebookStore(Path(directory) / "notebook.json")
            self.assertTrue(register_notebook_routes(app, store))
            self.assertFalse(register_notebook_routes(app, store))

            matches = [
                route for route in app.routes if route.path == NOTEBOOK_API_PATH
            ]
            self.assertEqual(len(matches), 2)
            methods = {method for route in matches for method in route.methods}
            self.assertIn("GET", methods)
            self.assertIn("PUT", methods)

            with TestClient(app) as client:
                loaded = client.get(
                    NOTEBOOK_API_PATH,
                    headers={"X-SAM3-Notebook": "1"},
                )
                self.assertEqual(loaded.status_code, 200)
                self.assertEqual(loaded.json()["revision"], 0)

                saved = client.put(
                    NOTEBOOK_API_PATH,
                    headers={"X-SAM3-Notebook": "1"},
                    json={
                        "version": 1,
                        "revision": 0,
                        "presets": [
                            {
                                "id": "one",
                                "name": "HTTP",
                                "entries": [],
                            }
                        ],
                    },
                )
                self.assertEqual(saved.status_code, 200)
                self.assertEqual(saved.json()["revision"], 1)

                stale = client.put(
                    NOTEBOOK_API_PATH,
                    headers={"X-SAM3-Notebook": "1"},
                    json={"version": 1, "revision": 0, "presets": []},
                )
                self.assertEqual(stale.status_code, 409)

    def test_put_rejects_cross_origin_shape_and_non_object_json(self):
        with tempfile.TemporaryDirectory() as directory:
            app = FastAPI()
            register_notebook_routes(
                app,
                NotebookStore(Path(directory) / "notebook.json"),
            )
            with TestClient(app) as client:
                missing_get_header = client.get(NOTEBOOK_API_PATH)
                self.assertEqual(missing_get_header.status_code, 403)

                missing_header = client.put(
                    NOTEBOOK_API_PATH,
                    json={"version": 1, "revision": 0, "presets": []},
                )
                self.assertEqual(missing_header.status_code, 403)

                non_object = client.put(
                    NOTEBOOK_API_PATH,
                    headers={"X-SAM3-Notebook": "1"},
                    json=[],
                )
                self.assertEqual(non_object.status_code, 400)
                self.assertIn("must be an object", non_object.json()["detail"])

    def test_route_reports_corrupt_storage_instead_of_empty_notebook(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notebook.json"
            path.write_text("{broken", encoding="utf-8")
            app = FastAPI()
            register_notebook_routes(app, NotebookStore(path))

            with TestClient(app) as client:
                response = client.get(
                    NOTEBOOK_API_PATH,
                    headers={"X-SAM3-Notebook": "1"},
                )
                self.assertEqual(response.status_code, 500)
                self.assertIn("damaged", response.json()["detail"])
                save = client.put(
                    NOTEBOOK_API_PATH,
                    headers={"X-SAM3-Notebook": "1"},
                    json={
                        "version": 1,
                        "revision": 0,
                        "presets": [],
                    },
                )
                self.assertEqual(save.status_code, 500)
            self.assertEqual(path.read_text(encoding="utf-8"), "{broken")

    def test_routes_reuse_gradio_login_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            app = FastAPI()

            @app.get("/login_check")
            def login_check(x_test_auth: str | None = Header(default=None)):
                if x_test_auth != "ok":
                    raise HTTPException(status_code=401, detail="Not authenticated")

            register_notebook_routes(
                app,
                NotebookStore(Path(directory) / "notebook.json"),
            )
            with TestClient(app) as client:
                self.assertEqual(client.get(NOTEBOOK_API_PATH).status_code, 401)
                self.assertEqual(
                    client.get(
                        NOTEBOOK_API_PATH,
                        headers={
                            "X-Test-Auth": "ok",
                            "X-SAM3-Notebook": "1",
                        },
                    ).status_code,
                    200,
                )
                self.assertEqual(
                    client.put(
                        NOTEBOOK_API_PATH,
                        headers={
                            "X-Test-Auth": "ok",
                            "X-SAM3-Notebook": "1",
                        },
                        json={
                            "version": 1,
                            "revision": 0,
                            "presets": [],
                        },
                    ).status_code,
                    200,
                )

    def test_route_converts_storage_oserrors_to_bounded_500_responses(self):
        class ReadFailureStore:
            def load(self):
                raise OSError("private read path")

        class WriteFailureStore:
            def load(self):
                return empty_notebook()

            def save(self, payload, *, expected_revision=None):
                raise OSError("private write path")

        read_app = FastAPI()
        register_notebook_routes(read_app, ReadFailureStore())
        with TestClient(read_app) as client:
            response = client.get(
                NOTEBOOK_API_PATH,
                headers={"X-SAM3-Notebook": "1"},
            )
            self.assertEqual(response.status_code, 500)
            self.assertEqual(
                response.json()["detail"],
                "Notebook storage I/O failed while reading",
            )
            self.assertNotIn("private read path", response.text)

        write_app = FastAPI()
        register_notebook_routes(write_app, WriteFailureStore())
        with TestClient(write_app) as client:
            response = client.put(
                NOTEBOOK_API_PATH,
                headers={"X-SAM3-Notebook": "1"},
                json={"version": 1, "revision": 0, "presets": []},
            )
            self.assertEqual(response.status_code, 500)
            self.assertEqual(
                response.json()["detail"],
                "Notebook storage I/O failed while writing",
            )
            self.assertNotIn("private write path", response.text)


class ForgeApiAuthTests(unittest.TestCase):
    """``--api-auth`` guards the Notebook routes the way Forge guards ``/sdapi``."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.store = NotebookStore(Path(self._directory.name) / "notebook.json")

    def test_nowebui_api_auth_guards_the_notebook_routes(self):
        app = FastAPI()   # api_only_worker: no Gradio /login_check
        with forge_api_auth("user:secret", api=False, nowebui=True):
            register_notebook_routes(app, self.store)
        body = {"version": 1, "revision": 0, "presets": []}
        with TestClient(app) as client:
            for auth in (None, ("user", "wrong")):
                with self.subTest(auth=auth):
                    extra = {} if auth is None else {"auth": auth}
                    read = client.get(NOTEBOOK_API_PATH, headers=NOTEBOOK_HEADERS, **extra)
                    write = client.put(NOTEBOOK_API_PATH, headers=NOTEBOOK_HEADERS, json=body, **extra)
                    self.assertEqual((read.status_code, write.status_code), (401, 401))
                    self.assertEqual(read.headers["www-authenticate"], "Basic")
            self.assertFalse(self.store.path.exists())
            ok = {"headers": NOTEBOOK_HEADERS, "auth": ("user", "secret")}
            self.assertEqual(client.get(NOTEBOOK_API_PATH, **ok).status_code, 200)
            self.assertEqual(client.put(NOTEBOOK_API_PATH, json=body, **ok).status_code, 200)

    def test_gradio_login_and_api_auth_are_both_required(self):
        app = FastAPI()

        @app.get("/login_check")
        def login_check(x_test_auth: str | None = Header(default=None)):
            if x_test_auth != "ok":
                raise HTTPException(status_code=401, detail="Not authenticated")

        with forge_api_auth("user:secret"):
            register_notebook_routes(app, self.store)
        logged_in = {**NOTEBOOK_HEADERS, "X-Test-Auth": "ok"}
        with TestClient(app) as client:
            self.assertEqual(client.get(NOTEBOOK_API_PATH, headers=logged_in).status_code, 401)
            self.assertEqual(
                client.get(NOTEBOOK_API_PATH, headers=NOTEBOOK_HEADERS, auth=("user", "secret")).status_code,
                401)
            self.assertEqual(
                client.get(NOTEBOOK_API_PATH, headers=logged_in, auth=("user", "secret")).status_code, 200)

    def test_without_api_auth_the_browser_page_needs_no_basic_credentials(self):
        app = FastAPI()
        with forge_api_auth(None):
            register_notebook_routes(app, self.store)
        with TestClient(app) as client:
            self.assertEqual(client.get(NOTEBOOK_API_PATH, headers=NOTEBOOK_HEADERS).status_code, 200)

    def test_api_auth_without_forge_api_guards_nothing_like_forge(self):
        # webui_worker builds Forge's Api only under --api: without it --api-auth guards no route.
        app = FastAPI()
        with forge_api_auth("user:secret", api=False, nowebui=False):
            register_notebook_routes(app, self.store)
        with TestClient(app) as client:
            self.assertEqual(client.get(NOTEBOOK_API_PATH, headers=NOTEBOOK_HEADERS).status_code, 200)

    def test_outside_forge_there_is_no_api_auth_setting(self):
        app = FastAPI()
        with forge_api_auth(None, loaded=False):
            register_notebook_routes(app, self.store)
        with TestClient(app) as client:
            self.assertEqual(client.get(NOTEBOOK_API_PATH, headers=NOTEBOOK_HEADERS).status_code, 200)

    @unittest.skipUnless(FORGE_API_PY.is_file(), "Forge's modules/api/api.py is not next to the extension")
    def test_the_guard_answers_exactly_like_forge_api_auth(self):
        # origin: <Forge>/modules/api/api.py Api.add_api_route + Api.auth (applied under --api-auth)
        api_auth = "user:secret,second:pass2"
        app = FastAPI()
        forge = forge_api_class(api_auth)()
        forge.app = app
        forge.credentials = dict(pair.split(":") for pair in api_auth.split(","))   # Api.__init__
        forge.add_api_route("/forge", lambda: {"ok": True}, methods=["GET"])
        app.add_api_route("/extension", lambda: {"ok": True}, methods=["GET"],
                          dependencies=_forge_api_auth_dependencies(api_auth))
        cases = (None, ("user", "secret"), ("second", "pass2"), ("user", "pass2"), ("user", ""),
                 ("nobody", "secret"), ("", ""), ("user", "secret "))
        with TestClient(app) as client:
            for auth in cases:
                with self.subTest(auth=auth):
                    extra = {} if auth is None else {"auth": auth}
                    forge_answer = client.get("/forge", **extra)
                    ours = client.get("/extension", **extra)
                    self.assertEqual(
                        (ours.status_code, ours.headers.get("www-authenticate"), ours.json()),
                        (forge_answer.status_code, forge_answer.headers.get("www-authenticate"),
                         forge_answer.json()),
                    )
            bad_header = {"Authorization": "Basic not-base64"}
            self.assertEqual(client.get("/extension", headers=bad_header).status_code,
                             client.get("/forge", headers=bad_header).status_code)


if __name__ == "__main__":
    unittest.main()
