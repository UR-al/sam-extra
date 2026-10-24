"""GET /sam3-lora/lite (sam3ext/lora_lite.py): Forge's LoRA list without the safetensors headers.

The reference every answer is checked against is built from ``/sdapi/v1/loras`` items the way
its clients reduce them (AI Studio Pro's ``get_loras``): name/alias/path as is, "activation text"
else "ss_activation_text", and ``ss_tag_frequency`` (JSON string or dict) merged with
``dict.update`` and sorted by count, descending and stable. ``ForgeOriginTests`` runs Forge's own
``sd_forge_lora`` code — the folder scan, ``NetworkOnDisk``, ``read_metadata_from_safetensors``
and the ``/sdapi/v1/loras`` route itself — on real safetensors headers in a temp folder, and
compares the two routes item by item on one app.
"""
from __future__ import annotations

import ast
import contextlib
import json
import os
import sys
import tempfile
import threading
import time
import types
import typing
import unittest
from pathlib import Path
from unittest import mock

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from sam3ext import lora_lite
from sam3ext.lora_lite import (
    LORA_LITE_PATH,
    TAG_FREQUENCY_TOP,
    LoraLiteBuilder,
    activation_text,
    lora_lite_data,
    register_lora_lite_route,
    tag_frequency_top,
)
from tests._forge_checkout import require_forge_file

ROOT = Path(__file__).resolve().parents[1]


# -- the reference: what a /sdapi/v1/loras client derives ------------------------------------

def reference_item(full: dict) -> dict:
    """One /sdapi/v1/loras item reduced the way AI Studio Pro's get_loras reads it."""
    metadata = full.get("metadata") or {}
    act = metadata.get("activation text", "") or metadata.get("ss_activation_text", "")
    tag_freq = metadata.get("ss_tag_frequency", {})
    if isinstance(tag_freq, str):
        try:
            tag_freq = json.loads(tag_freq)
        except Exception:
            tag_freq = {}
    merged = {}
    if isinstance(tag_freq, dict):
        for ds_tags in tag_freq.values():
            if isinstance(ds_tags, dict):
                merged.update(ds_tags)
    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return {
        "name": full["name"],
        "alias": full["alias"],
        "path": full["path"],
        "activation_text": act if isinstance(act, str) else "",
        "tag_frequency_top": [[tag, count] for tag, count in ranked[:32]],
    }


def reference_payload(full_list: list[dict]) -> dict:
    items = [reference_item(item) for item in full_list]
    return {"version": 1, "count": len(items), "items": items}


@contextlib.contextmanager
def forge_api_auth(value):
    """Forge started with ``--api --api-auth value`` (the stand-in of tests/test_lora_routes.py)."""

    missing = object()
    saved = sys.modules.get("modules.shared", missing)
    sys.modules["modules.shared"] = types.SimpleNamespace(
        cmd_opts=types.SimpleNamespace(api_auth=value, api=True, nowebui=False))
    try:
        yield
    finally:
        if saved is missing:
            sys.modules.pop("modules.shared", None)
        else:
            sys.modules["modules.shared"] = saved


# -- fake Forge networks ----------------------------------------------------------------------

class FakeNetworkOnDisk:
    """The attributes /sdapi/v1/loras serialises (sd_forge_lora/network.py NetworkOnDisk)."""

    def __init__(self, name: str, filename: str, metadata: dict | None = None) -> None:
        self.name = name
        self.filename = filename
        self.metadata = metadata if metadata is not None else {}
        self.alias = self.metadata.get("ss_output_name", self.name)


def create_lora_json(obj) -> dict:
    """sd_forge_lora/scripts/lora_script.py create_lora_json (ForgeOriginTests runs the real one)."""
    return {"name": obj.name, "alias": obj.alias, "path": obj.filename, "metadata": obj.metadata}


def tag_groups(*groups: dict) -> dict:
    return {f"{10 - i}_set{i}": dict(group) for i, group in enumerate(groups)}


MANY_TAGS = {f"tag{i:02d}": (50 - i // 3) for i in range(45)}   # ties in threes, 45 > 32
SAMPLE_METADATA = [
    # (name, filename, metadata)
    ("styleA", "D:/LoRA/styleA/styleA.safetensors", {
        "ss_output_name": "styleA_v2",
        # Forge leaves a JSON string it could not parse as a string; this one is valid JSON but
        # starts with a space, so it reaches clients as a string.
        "ss_tag_frequency": " " + json.dumps(tag_groups(
            {"1girl": 40, "solo": 35, "smile": 12},
            {"solo": 3, "outdoors": 12, "night": 12},
        )),
        "ss_datasets": json.dumps([{"subsets": [{"image_dir": "x" * 2000}]}]),
    }),
    ("charB", "D:/LoRA/charB/charB.safetensors", {
        "activation text": "",
        "ss_activation_text": "charB, red hair",
        "ss_tag_frequency": tag_groups({"charB": 20, "red hair": 20}),
    }),
    ("charC", "D:/LoRA/charC/charC.safetensors", {
        "activation text": "trigger one, trigger two",
        "ss_activation_text": "ignored",
        "ss_tag_frequency": tag_groups(MANY_TAGS),
    }),
    ("bare", "D:/LoRA/bare.pt", {}),
    ("broken", "D:/LoRA/broken.safetensors", {"ss_tag_frequency": "{not json"}),
    ("odd", "D:/LoRA/odd.safetensors", {"ss_tag_frequency": {"10_a": {"a": 1}, "9_x": "not a dict", "8_b": {"b": 2}}}),
]


def fake_networks(entries=SAMPLE_METADATA, *, rescan=None):
    """A stand-in for Forge's ``networks`` module: available_networks + list_available_networks."""
    module = types.SimpleNamespace(available_networks={}, rescans=0)

    def fill(source):
        module.available_networks.clear()
        for name, filename, metadata in source:
            module.available_networks[name] = FakeNetworkOnDisk(name, filename, json.loads(json.dumps(metadata)))

    def list_available_networks():
        module.rescans += 1
        fill(rescan if rescan is not None else entries)

    fill(entries)
    module.list_available_networks = list_available_networks
    return module


def full_list(networks) -> list[dict]:
    return [create_lora_json(obj) for obj in networks.available_networks.values()]


def lite_app(networks, *, auth_dependencies=(), builder=None) -> FastAPI:
    app = FastAPI()
    register_lora_lite_route(
        app, auth_dependencies=list(auth_dependencies), networks=networks,
        builder=builder or LoraLiteBuilder())
    return app


# -- field helpers ----------------------------------------------------------------------------

class ActivationTextTests(unittest.TestCase):
    def test_activation_text_then_ss_activation_text_then_empty(self):
        self.assertEqual(activation_text({"activation text": "a, b", "ss_activation_text": "c"}), "a, b")
        self.assertEqual(activation_text({"activation text": "", "ss_activation_text": "c"}), "c")
        self.assertEqual(activation_text({"ss_activation_text": "c"}), "c")
        self.assertEqual(activation_text({}), "")
        self.assertEqual(activation_text(None), "")

    def test_the_value_a_client_reads_wins_and_only_text_is_kept(self):
        # A client of /sdapi/v1/loras reads md.get("activation text", "") or md.get("ss_activation_text", "").
        # Forge parses a "{"-led header value into an object: that object is what such a client gets (and has
        # no words in), so it is "" here too — not the text below it, or the two lists would disagree.
        self.assertEqual(activation_text({"activation text": {"x": 1}, "ss_activation_text": "c"}), "")
        self.assertEqual(activation_text({"activation text": {}, "ss_activation_text": "c"}), "c")
        self.assertEqual(activation_text({"activation text": "", "ss_activation_text": {"x": 1}}), "")
        self.assertEqual(activation_text({"activation text": 5}), "")

    def test_every_case_matches_the_reference(self):
        cases = [
            {"activation text": "a, b", "ss_activation_text": "c"},
            {"activation text": "", "ss_activation_text": "c"},
            {"activation text": " , ", "ss_activation_text": "c"},
            {"activation text": {"x": 1}, "ss_activation_text": "c"},
            {"activation text": {}, "ss_activation_text": "c"},
            {"activation text": None, "ss_activation_text": "c"},
            {"activation text": "", "ss_activation_text": {"x": 1}},
            {"ss_activation_text": 0},
            {},
        ]
        for metadata in cases:
            with self.subTest(metadata=metadata):
                expected = reference_item({"name": "n", "alias": "n", "path": "p", "metadata": metadata})
                self.assertEqual(activation_text(metadata), expected["activation_text"])


class TagFrequencyTests(unittest.TestCase):
    def assertMatchesReference(self, metadata):
        expected = reference_item({"name": "n", "alias": "n", "path": "p", "metadata": metadata})
        self.assertEqual(tag_frequency_top(metadata), expected["tag_frequency_top"])

    def test_json_string_and_dict_give_the_same_list(self):
        groups = tag_groups({"1girl": 40, "solo": 35, "smile": 12}, {"solo": 3, "outdoors": 12})
        as_dict = tag_frequency_top({"ss_tag_frequency": groups})
        self.assertEqual(as_dict, tag_frequency_top({"ss_tag_frequency": json.dumps(groups)}))
        self.assertMatchesReference({"ss_tag_frequency": groups})

    def test_later_dataset_replaces_the_count_and_the_first_position_stays(self):
        groups = tag_groups({"1girl": 40, "solo": 35, "smile": 12}, {"solo": 3, "outdoors": 12, "night": 12})
        self.assertEqual(
            tag_frequency_top({"ss_tag_frequency": groups}),
            [["1girl", 40], ["smile", 12], ["outdoors", 12], ["night", 12], ["solo", 3]],
        )

    def test_ties_keep_merge_order_and_the_list_stops_at_32(self):
        top = tag_frequency_top({"ss_tag_frequency": tag_groups(MANY_TAGS)})
        self.assertEqual(TAG_FREQUENCY_TOP, 32)
        self.assertEqual(len(top), 32)
        self.assertEqual([tag for tag, _ in top], [f"tag{i:02d}" for i in range(32)])
        self.assertMatchesReference({"ss_tag_frequency": tag_groups(MANY_TAGS)})

    def test_nothing_to_rank_gives_an_empty_list(self):
        for metadata in ({}, None, {"ss_tag_frequency": "{not json"}, {"ss_tag_frequency": "[1, 2]"},
                         {"ss_tag_frequency": 7}, {"ss_tag_frequency": {}},
                         {"ss_tag_frequency": {"10_x": "nope"}}):
            with self.subTest(metadata=metadata):
                self.assertEqual(tag_frequency_top(metadata), [])

    def test_groups_that_are_not_dicts_are_skipped(self):
        metadata = {"ss_tag_frequency": {"10_a": {"a": 1}, "9_x": "not a dict", "8_b": {"b": 2}}}
        self.assertEqual(tag_frequency_top(metadata), [["b", 2], ["a", 1]])
        self.assertMatchesReference(metadata)

    def test_a_count_that_cannot_be_compared_drops_only_its_tag(self):
        metadata = {"ss_tag_frequency": tag_groups({"a": 3, "b": "many", "c": 5})}
        self.assertEqual(tag_frequency_top(metadata), [["c", 5], ["a", 3]])

    def test_a_count_json_cannot_carry_drops_only_its_tag(self):
        metadata = {"ss_tag_frequency": tag_groups({"a": 3, "d": float("nan"), "e": float("inf"), "c": 5})}
        top = tag_frequency_top(metadata)
        self.assertEqual(sorted(tag for tag, _ in top), ["a", "c"])
        json.dumps(top, allow_nan=False)


# -- builder and memo -------------------------------------------------------------------------

class BuilderTests(unittest.TestCase):
    def test_payload_matches_the_reference_in_sdapi_order(self):
        networks = fake_networks()
        data = lora_lite_data(networks=networks, builder=LoraLiteBuilder())
        self.assertEqual(data, reference_payload(full_list(networks)))
        self.assertEqual([item["name"] for item in data["items"]], [name for name, _, _ in SAMPLE_METADATA])
        self.assertEqual(data["items"][0]["alias"], "styleA_v2")
        self.assertEqual(set(data["items"][0]), {"name", "alias", "path", "activation_text", "tag_frequency_top"})
        json.dumps(data, allow_nan=False)

    def test_tag_lists_are_memoised_per_metadata_object(self):
        networks = fake_networks()
        builder = LoraLiteBuilder()
        with mock.patch.object(lora_lite, "tag_frequency_top", wraps=lora_lite.tag_frequency_top) as ranker:
            first = lora_lite_data(networks=networks, builder=builder)
            self.assertEqual(ranker.call_count, len(SAMPLE_METADATA))
            second = lora_lite_data(networks=networks, builder=builder)
            self.assertEqual(ranker.call_count, len(SAMPLE_METADATA), "unchanged objects are not re-ranked")
            self.assertEqual(first, second)

            # Forge's rescan builds new objects with new metadata: every LoRA that has an
            # ss_tag_frequency is ranked again (one without has nothing to rank either way).
            lora_lite_data(refresh=True, networks=networks, builder=builder)
            with_tags = sum(1 for _, _, metadata in SAMPLE_METADATA if "ss_tag_frequency" in metadata)
            self.assertEqual(ranker.call_count, len(SAMPLE_METADATA) + with_tags)

            # A value replaced in place on one object re-ranks that LoRA only.
            networks.available_networks["charB"].metadata["ss_tag_frequency"] = tag_groups({"new": 9})
            third = lora_lite_data(networks=networks, builder=builder)
            self.assertEqual(ranker.call_count, len(SAMPLE_METADATA) + with_tags + 1)
        self.assertEqual(third["items"][1]["tag_frequency_top"], [["new", 9]])

    def test_memo_keeps_only_the_loras_of_the_last_answer(self):
        networks = fake_networks()
        builder = LoraLiteBuilder()
        lora_lite_data(networks=networks, builder=builder)
        del networks.available_networks["charC"]
        lora_lite_data(networks=networks, builder=builder)
        self.assertNotIn("D:/LoRA/charC/charC.safetensors", builder._memo)
        self.assertEqual(len(builder._memo), len(SAMPLE_METADATA) - 1)

    def test_an_answer_cannot_change_the_next_one(self):
        networks = fake_networks()
        builder = LoraLiteBuilder()
        first = lora_lite_data(networks=networks, builder=builder)
        first["items"][0]["tag_frequency_top"][0][1] = -1
        first["items"][0]["tag_frequency_top"].clear()
        self.assertEqual(lora_lite_data(networks=networks, builder=builder), reference_payload(full_list(networks)))

    def test_refresh_rescans_before_reading(self):
        networks = fake_networks(rescan=SAMPLE_METADATA[:2])
        data = lora_lite_data(networks=networks, builder=LoraLiteBuilder())
        self.assertEqual((networks.rescans, data["count"]), (0, len(SAMPLE_METADATA)))
        data = lora_lite_data(refresh=True, networks=networks, builder=LoraLiteBuilder())
        self.assertEqual((networks.rescans, data["count"]), (1, 2))
        self.assertEqual(data, reference_payload(full_list(networks)))

    def test_concurrent_refreshes_run_one_at_a_time(self):
        # Forge's list_available_networks() clears its tables and refills them file by file. Two at once
        # (two ?refresh=1 in the threadpool) interleave, and a caller can answer a half-filled table.
        networks = fake_networks()
        refill = networks.list_available_networks
        running, overlaps = [], []

        def slow_rescan():
            running.append(1)
            if len(running) > 1:
                overlaps.append(len(running))
            try:
                networks.available_networks.clear()
                time.sleep(0.05)
                refill()
            finally:
                running.pop()

        networks.list_available_networks = slow_rescan
        answers = []
        threads = [threading.Thread(target=lambda: answers.append(
            lora_lite_data(refresh=True, networks=networks, builder=LoraLiteBuilder()))) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual(overlaps, [], "rescans overlapped")
        self.assertEqual([data["count"] for data in answers], [len(SAMPLE_METADATA)] * 3)
        self.assertEqual(networks.rescans, 3)

    def test_pathlike_paths_are_strings(self):
        networks = fake_networks([("p", Path("D:/LoRA/p.safetensors"), {})])
        (item,) = lora_lite_data(networks=networks, builder=LoraLiteBuilder())["items"]
        self.assertEqual(item["path"], os.fspath(Path("D:/LoRA/p.safetensors")))

    def test_no_loras(self):
        networks = fake_networks([])
        self.assertEqual(lora_lite_data(networks=networks, builder=LoraLiteBuilder()),
                         {"version": 1, "count": 0, "items": []})


# -- the HTTP route ---------------------------------------------------------------------------

class RouteTests(unittest.TestCase):
    def test_route_answers_the_reference(self):
        networks = fake_networks()
        with TestClient(lite_app(networks)) as client:
            response = client.get(LORA_LITE_PATH)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(response.headers["cache-control"], "no-store, max-age=0")
        self.assertEqual(response.json(), reference_payload(full_list(networks)))

    def test_refresh_query(self):
        networks = fake_networks(rescan=SAMPLE_METADATA[:3])
        with TestClient(lite_app(networks)) as client:
            for query, rescans, count in (("", 0, 6), ("?refresh=0", 0, 6), ("?refresh=", 0, 6),
                                          ("?refresh=1", 1, 3), ("?refresh=true", 2, 3)):
                with self.subTest(query=query):
                    response = client.get(LORA_LITE_PATH + query)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual((networks.rescans, response.json()["count"]), (rescans, count))
                    self.assertEqual(response.json(), reference_payload(full_list(networks)))

    def test_without_forge_lora_support_it_answers_503(self):
        app = FastAPI()
        register_lora_lite_route(app, auth_dependencies=[])

        def missing():
            raise ImportError("No module named 'networks'")

        with mock.patch.object(lora_lite, "_forge_networks", side_effect=missing):
            with TestClient(app) as client:
                response = client.get(LORA_LITE_PATH)
        self.assertEqual(response.status_code, 503)
        self.assertIn("sd_forge_lora", response.json()["detail"])

    def test_registered_once_get_only_out_of_the_schema(self):
        app = FastAPI()
        self.assertTrue(register_lora_lite_route(app, auth_dependencies=[], networks=fake_networks()))
        self.assertFalse(register_lora_lite_route(app, auth_dependencies=[], networks=fake_networks()))
        routes = [(r.path, r.name, tuple(sorted(r.methods)), r.include_in_schema)
                  for r in app.routes if r.path == LORA_LITE_PATH]
        self.assertEqual(routes, [(LORA_LITE_PATH, "sam3-lora-lite", ("GET",), False)])
        with TestClient(app) as client:
            self.assertEqual(client.post(LORA_LITE_PATH).status_code, 405)
            self.assertNotIn(LORA_LITE_PATH, client.get("/openapi.json").json().get("paths", {}))

    def test_auth_dependencies_guard_it_and_no_notebook_header_is_needed(self):
        def deny():
            raise HTTPException(status_code=401, detail="denied")

        networks = fake_networks()
        with TestClient(lite_app(networks, auth_dependencies=[Depends(deny)])) as client:
            self.assertEqual(client.get(LORA_LITE_PATH + "?refresh=1").status_code, 401)
        self.assertEqual(networks.rescans, 0, "a refused request rescans nothing")
        with TestClient(lite_app(networks)) as client:
            self.assertEqual(client.get(LORA_LITE_PATH).status_code, 200)

    def test_forge_api_auth_guards_it_like_sdapi(self):
        app = FastAPI()
        with forge_api_auth("user:secret"):
            register_lora_lite_route(app, networks=fake_networks(), builder=LoraLiteBuilder())
        with TestClient(app) as client:
            self.assertEqual(client.get(LORA_LITE_PATH).status_code, 401)
            self.assertEqual(client.get(LORA_LITE_PATH, auth=("user", "wrong")).status_code, 401)
            self.assertEqual(client.get(LORA_LITE_PATH, auth=("user", "secret")).status_code, 200)

    def test_the_lora_manager_script_registers_it(self):
        source = (ROOT / "scripts" / "lora_manager.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        started = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "_on_app_started")
        calls = [node.func.id for node in ast.walk(started)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertIn("register_lora_routes", calls)
        self.assertIn("register_lora_lite_route", calls)
        tries = [node for node in started.body if isinstance(node, ast.Try)]
        self.assertEqual(len(tries), 2, "each registration fails on its own")


# -- against Forge's own code -----------------------------------------------------------------

def _forge_tree(relpath: str) -> tuple[Path, ast.Module]:
    path = require_forge_file(relpath)
    return path, ast.parse(path.read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str, body=None) -> ast.FunctionDef:
    return next(node for node in (body if body is not None else tree.body)
                if isinstance(node, ast.FunctionDef) and node.name == name)


def _exec(nodes: list[ast.stmt], filename: Path, namespace: dict) -> dict:
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(filename), "exec"), namespace)
    return namespace


def write_safetensors(path: Path, metadata: dict) -> None:
    """A safetensors header with ``__metadata__`` (all values strings, as the format requires)."""
    header = json.dumps({"__metadata__": metadata}).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(header).to_bytes(8, "little") + header)


class ForgeOriginTests(unittest.TestCase):
    """Forge's scan, NetworkOnDisk, metadata reader and /sdapi/v1/loras, run as is."""

    @classmethod
    def setUpClass(cls):
        cls.networks_py, cls.networks_tree = _forge_tree("extensions-builtin/sd_forge_lora/networks.py")
        cls.network_py, cls.network_tree = _forge_tree("extensions-builtin/sd_forge_lora/network.py")
        cls.script_py, cls.script_tree = _forge_tree("extensions-builtin/sd_forge_lora/scripts/lora_script.py")
        cls.models_py, cls.models_tree = _forge_tree("modules/sd_models.py")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.lora_dir = base / "LoRA"
        self.extra_dir = base / "extra"
        self.forge = self._load_forge()

    def _load_forge(self):
        errors = types.SimpleNamespace(
            report=lambda *a, **k: None, display=lambda *a, **k: None)
        reader = _exec([_function(self.models_tree, "read_metadata_from_safetensors")],
                       self.models_py, {"errors": errors})["read_metadata_from_safetensors"]

        def walk_files(path, allowed_extensions=None):
            for root, _dirs, files in sorted(os.walk(path)):
                for filename in sorted(files):
                    if allowed_extensions is None or os.path.splitext(filename)[1].lower() in allowed_extensions:
                        yield os.path.join(root, filename)

        shared = types.SimpleNamespace(
            cmd_opts=types.SimpleNamespace(lora_dir=str(self.lora_dir), lora_dirs=[str(self.extra_dir)], api=True),
            opts=types.SimpleNamespace(lora_preferred_name="Alias from file"),
            walk_files=walk_files,
        )
        network_module = types.ModuleType("network")
        network_module.__dict__.update({
            "os": os, "errors": errors, "shared": shared,
            "cache": types.SimpleNamespace(cached_data_for_file=lambda _s, _t, _f, func: func()),
            "hashes": types.SimpleNamespace(sha256_from_cache=lambda *a, **k: None),
            "sd_models": types.SimpleNamespace(read_metadata_from_safetensors=reader),
        })
        network_class = next(node for node in self.network_tree.body
                             if isinstance(node, ast.ClassDef) and node.name == "NetworkOnDisk")
        _exec([network_class], self.network_py, network_module.__dict__)

        networks_module = types.ModuleType("networks")
        networks_module.__dict__.update({
            "os": os, "errors": errors, "shared": shared, "network": network_module,
            "Optional": typing.Optional,
            "available_networks": {}, "available_network_aliases": {},
            "available_network_hash_lookup": {}, "forbidden_network_aliases": set(),
        })
        _exec([_function(self.networks_tree, "process_network_files"),
               _function(self.networks_tree, "list_available_networks")],
              self.networks_py, networks_module.__dict__)

        # lora_script.py: create_lora_json and api_networks sit under `if shared.cmd_opts.api:`.
        api_block = next(node for node in self.script_tree.body if isinstance(node, ast.If))
        script_ns = _exec(
            [_function(self.script_tree, "create_lora_json", api_block.body),
             _function(self.script_tree, "api_networks", api_block.body)],
            self.script_py,
            {"network": network_module, "networks": networks_module,
             "gr": types.SimpleNamespace(Blocks=object), "FastAPI": FastAPI})
        return types.SimpleNamespace(networks=networks_module, api_networks=script_ns["api_networks"])

    def _library(self):
        for name, _filename, metadata in SAMPLE_METADATA:
            folder = self.lora_dir / name
            if name == "bare":
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "bare.pt").write_bytes(b"")
                continue
            stored = {key: value if isinstance(value, str) else json.dumps(value)
                      for key, value in metadata.items()}
            write_safetensors(folder / f"{name}.safetensors", stored)
        # Same name in the second folder: Forge keeps the first position and the later object.
        write_safetensors(self.extra_dir / "dup" / "charB.safetensors",
                          {"ss_output_name": "charB_extra", "ss_activation_text": "from extra"})
        write_safetensors(self.extra_dir / "zeta.safetensors",
                          {"ss_tag_frequency": json.dumps(tag_groups({"z": 1, "y": 1}, {"x": 1}))})

    def _app(self):
        app = FastAPI()
        self.forge.api_networks(None, app)
        register_lora_lite_route(app, auth_dependencies=[], networks=self.forge.networks,
                                 builder=LoraLiteBuilder())
        return app

    def test_lite_matches_sdapi_loras_item_by_item(self):
        self._library()
        self.forge.networks.list_available_networks()
        with TestClient(self._app()) as client:
            full = client.get("/sdapi/v1/loras").json()
            lite = client.get(LORA_LITE_PATH).json()
        self.assertEqual(len(full), len(SAMPLE_METADATA) + 1)
        self.assertEqual(lite, reference_payload(full))
        names = [item["name"] for item in lite["items"]]
        # Folder walk order, and the duplicate charB keeps its first position.
        self.assertEqual(names, ["bare", "broken", "charB", "charC", "odd", "styleA", "zeta"])
        by_name = dict(zip(names, lite["items"]))
        charb = by_name["charB"]
        self.assertEqual((charb["alias"], charb["activation_text"]), ("charB_extra", "from extra"))
        self.assertTrue(charb["path"].endswith(os.path.join("dup", "charB.safetensors")))
        # Forge parsed the JSON-looking strings to dicts; the space-led one stayed a string.
        objs = self.forge.networks.available_networks
        self.assertIsInstance(objs["charC"].metadata["ss_tag_frequency"], dict)
        self.assertIsInstance(objs["styleA"].metadata["ss_tag_frequency"], str)
        self.assertEqual(by_name["styleA"]["tag_frequency_top"][:2], [["1girl", 40], ["smile", 12]])
        self.assertEqual(by_name["styleA"]["alias"], "styleA_v2")
        self.assertEqual(by_name["charC"]["activation_text"], "trigger one, trigger two")
        self.assertEqual(len(by_name["charC"]["tag_frequency_top"]), 32)
        self.assertEqual(by_name["zeta"]["tag_frequency_top"], [["z", 1], ["y", 1], ["x", 1]])
        self.assertEqual(by_name["bare"], {"name": "bare", "alias": "bare", "path": by_name["bare"]["path"],
                                           "activation_text": "", "tag_frequency_top": []})
        self.assertNotIn("ss_datasets", json.dumps(lite))

    def test_refresh_is_the_rescan_refresh_loras_runs(self):
        self._library()
        self.forge.networks.list_available_networks()
        with TestClient(self._app()) as client:
            before = client.get(LORA_LITE_PATH).json()["count"]
            write_safetensors(self.lora_dir / "late" / "late.safetensors", {"ss_activation_text": "late"})
            self.assertEqual(client.get(LORA_LITE_PATH).json()["count"], before)
            refreshed = client.get(LORA_LITE_PATH + "?refresh=1").json()
            full = client.get("/sdapi/v1/loras").json()
        self.assertEqual(refreshed["count"], before + 1)
        self.assertEqual(refreshed, reference_payload(full))


if __name__ == "__main__":
    unittest.main()
