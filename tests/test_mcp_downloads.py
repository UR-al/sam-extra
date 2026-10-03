"""sam-extra MCP server: module downloads are verified, and the module table knows sam-extra's modules.

- A download is kept only when its size matches what the host announced and, where Hugging Face
  states it (X-Linked-ETag on the /resolve/ redirect), its SHA256 too; unverifiable downloads are
  refused. Hugging Face is simulated with an httpx mock transport - nothing leaves the machine.
- The module audit knows sam-extra's Anima 3.8B files (Qwen3.5-4B, the expanded adapter) and the
  Qwen2D VAE family, which upstream reported as leftovers from another preset.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _mcp_support import HAS_HTTPX, httpx, make_service

from sam_extra_mcp.forgeneo import fetcher
from sam_extra_mcp.forgeneo.client import ApiResult
from sam_extra_mcp.forgeneo.modules import ARCH_MODULES, audit
from sam_extra_mcp.forgeneo.profile import _module_health
from sam_extra_mcp.policy import ALLOW_DOWNLOAD

FILE_URL = "https://huggingface.co/org/repo/resolve/main/split_files/text_encoders/te.safetensors"
CDN_URL = "https://cdn-lfs.example/objects/abc"


def safetensors_blob(payload: bytes = b"\x00" * 64) -> bytes:
    header = json.dumps({"__metadata__": {"format": "pt"}}).encode("utf-8")
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + payload


class HuggingFace:
    """/resolve/ answers HEAD with a redirect carrying X-Linked-Size / X-Linked-ETag, as the Hub does."""

    def __init__(self, content: bytes, *, announced_size: int | None = None, sha256: str | None = "auto",
                 linked_headers: bool = True, cdn_length: bool = True, head_status: int = 302):
        self.content = content
        self.announced = len(content) if announced_size is None else announced_size
        self.sha256 = hashlib.sha256(content).hexdigest() if sha256 == "auto" else sha256
        self.linked_headers = linked_headers
        self.cdn_length = cdn_length
        self.head_status = head_status
        self.requests: list[tuple[str, str]] = []
        self.on_get = None

    def transport(self):
        return httpx.MockTransport(self.handle)

    def handle(self, request):
        url = str(request.url)
        self.requests.append((request.method, url))
        if url.startswith("https://huggingface.co/"):
            if self.head_status >= 400:
                return httpx.Response(self.head_status)
            headers = {"location": CDN_URL}
            if self.linked_headers:
                headers["x-linked-size"] = str(self.announced)
                if self.sha256:
                    headers["x-linked-etag"] = f'"{self.sha256}"'
            return httpx.Response(302, headers=headers)
        if url == CDN_URL:
            headers = {"content-length": str(self.announced)} if self.cdn_length else {}
            if request.method == "HEAD":
                return httpx.Response(200, headers=headers)
            if callable(self.on_get):
                self.on_get()
            return httpx.Response(200, content=self.content)
        return httpx.Response(404)


@unittest.skipUnless(HAS_HTTPX, "httpx is not installed")
class FetcherVerificationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def fetch(self, hub: HuggingFace, name: str = "te.safetensors", permitted: bool = True) -> dict:
        return fetcher.fetch(FILE_URL, str(self.folder), name, permitted=permitted, transport=hub.transport())

    def leftovers(self) -> list[str]:
        return sorted(path.name for path in self.folder.iterdir())

    def test_size_and_sha256_come_from_the_redirect(self):
        content = safetensors_blob()
        hub = HuggingFace(content)
        intent = fetcher.plan(FILE_URL, str(self.folder), "te.safetensors", transport=hub.transport())
        self.assertIsNone(intent.blocked)
        self.assertEqual(intent.size_bytes, len(content))
        self.assertEqual(intent.sha256, hashlib.sha256(content).hexdigest())
        self.assertEqual(intent.verification, "size and sha256")
        self.assertEqual({method for method, _ in hub.requests}, {"HEAD"}, "the size probe never downloads")
        self.assertEqual(len(hub.requests), 1, "the Hub's own headers were enough")

    def test_verified_download_lands_in_place(self):
        content = safetensors_blob()
        result = self.fetch(HuggingFace(content))
        self.assertIs(result["ok"], True, result)
        self.assertEqual(result["verified"], "size and sha256")
        self.assertEqual(result["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual((self.folder / "te.safetensors").read_bytes(), content)
        self.assertEqual(self.leftovers(), ["te.safetensors"])

    def test_size_only_when_no_hash_is_stated(self):
        content = safetensors_blob()
        result = self.fetch(HuggingFace(content, linked_headers=False))
        self.assertIs(result["ok"], True, result)
        self.assertEqual(result["verified"], "size only")

    def test_sha256_mismatch_is_discarded(self):
        result = self.fetch(HuggingFace(safetensors_blob(), sha256="0" * 64))
        self.assertIs(result["ok"], False)
        self.assertIn("SHA256 mismatch", result["error"])
        self.assertEqual(self.leftovers(), [])

    def test_a_short_download_is_discarded(self):
        content = safetensors_blob()
        result = self.fetch(HuggingFace(content, announced_size=len(content) + 10, sha256=None))
        self.assertIs(result["ok"], False)
        self.assertIn("incomplete", result["error"])
        self.assertEqual(self.leftovers(), [])

    def test_a_stream_past_the_announced_size_is_cut_off(self):
        content = safetensors_blob()
        result = self.fetch(HuggingFace(content, announced_size=len(content) - 10, sha256=None))
        self.assertIs(result["ok"], False)
        self.assertIn("more than the announced", result["error"])
        self.assertEqual(self.leftovers(), [])

    def test_unknown_size_is_refused_before_downloading(self):
        hub = HuggingFace(safetensors_blob(), linked_headers=False, cdn_length=False)
        result = self.fetch(hub)
        self.assertIs(result["ok"], False)
        self.assertIn("could not be verified", result["error"])
        self.assertNotIn("GET", [method for method, _ in hub.requests])
        self.assertEqual(self.leftovers(), [])

    def test_something_that_is_not_safetensors_is_discarded(self):
        result = self.fetch(HuggingFace(b"<html>login required</html>"))
        self.assertIs(result["ok"], False)
        self.assertIn("not a valid safetensors", result["error"])
        self.assertEqual(self.leftovers(), [])

    def test_not_permitted_means_no_request_at_all(self):
        hub = HuggingFace(safetensors_blob())
        result = self.fetch(hub, permitted=False)
        self.assertIs(result["ok"], False)
        self.assertEqual(hub.requests, [])

    def test_plan_without_probe_stays_on_this_machine(self):
        hub = HuggingFace(safetensors_blob())
        intent = fetcher.plan(FILE_URL, str(self.folder), "te.safetensors", probe=False, transport=hub.transport())
        self.assertEqual(hub.requests, [])
        self.assertIsNone(intent.blocked)
        self.assertEqual(intent.verification, "not checked yet")
        self.assertIs(intent.as_dict()["probed"], False)

    def test_a_failing_host_blocks_the_plan(self):
        intent = fetcher.plan(FILE_URL, str(self.folder), "te.safetensors", transport=HuggingFace(b"", head_status=404).transport())
        self.assertIn("cannot check the file", intent.blocked)

    def test_a_file_that_appears_meanwhile_is_left_alone(self):
        content = safetensors_blob()
        hub = HuggingFace(content)
        hub.on_get = lambda: (self.folder / "te.safetensors").write_bytes(b"someone else's file")
        result = self.fetch(hub)
        self.assertIs(result["ok"], False)
        self.assertIn("appeared during the download", result["error"])
        self.assertEqual((self.folder / "te.safetensors").read_bytes(), b"someone else's file")
        self.assertEqual(self.leftovers(), ["te.safetensors"])

    def test_safetensors_check_runs_before_placement_and_a_late_file_survives(self):
        # The destination appears after the download finished, right before the move: the move
        # itself must refuse (there is no separate check to slip past any more).
        content = safetensors_blob()
        hub = HuggingFace(content)
        real_check = fetcher._looks_like_safetensors

        def check_then_race(path):
            (self.folder / "te.safetensors").write_bytes(b"arrived at the last moment")
            return real_check(path)

        with mock.patch.object(fetcher, "_looks_like_safetensors", check_then_race):
            result = self.fetch(hub)
        self.assertIs(result["ok"], False)
        self.assertIn("appeared during the download", result["error"])
        self.assertEqual((self.folder / "te.safetensors").read_bytes(), b"arrived at the last moment")
        self.assertEqual(self.leftovers(), ["te.safetensors"])

    def test_safetensors_check(self):
        good = self.folder / "good.safetensors"
        good.write_bytes(safetensors_blob())
        self.assertTrue(fetcher._looks_like_safetensors(str(good)))
        for name, blob in (("zero.bin", struct.pack("<Q", 0)), ("huge.bin", struct.pack("<Q", 1 << 40)),
                           ("short.bin", struct.pack("<Q", 64) + b"{}"), ("list.bin", struct.pack("<Q", 2) + b"[]")):
            path = self.folder / name
            path.write_bytes(blob)
            self.assertFalse(fetcher._looks_like_safetensors(str(path)), name)

    def test_etag_forms(self):
        digest = "a" * 64
        self.assertEqual(fetcher._sha256_from_etag(f'"{digest}"'), digest)
        self.assertEqual(fetcher._sha256_from_etag(f'W/"{digest.upper()}"'), digest)
        self.assertIsNone(fetcher._sha256_from_etag('"5d41402abc4b2a76b9719d911017c592"'))  # an md5-like git etag
        self.assertIsNone(fetcher._sha256_from_etag(None))


class PlacementTests(unittest.TestCase):
    """place_new_file: moving the finished download into place never replaces a file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.partial = self.folder / "x.safetensors.part"
        self.target = self.folder / "x.safetensors"
        self.partial.write_bytes(b"new download")

    def tearDown(self):
        self._tmp.cleanup()

    def strategies(self):
        # rename refuses an existing name only on Windows; the hard link refuses it everywhere.
        return (True, False) if os.name == "nt" else (False,)

    def test_places_the_file_and_drops_the_partial_name(self):
        for rename in self.strategies():
            with self.subTest(rename=rename):
                self.partial.write_bytes(b"new download")
                self.target.unlink(missing_ok=True)
                self.assertIs(fetcher.place_new_file(str(self.partial), str(self.target), rename_refuses_existing=rename), True)
                self.assertEqual(self.target.read_bytes(), b"new download")
                self.assertFalse(self.partial.exists())

    def test_never_replaces_an_existing_file(self):
        for rename in self.strategies():
            with self.subTest(rename=rename):
                self.partial.write_bytes(b"new download")
                self.target.write_bytes(b"someone else's file")
                self.assertIs(fetcher.place_new_file(str(self.partial), str(self.target), rename_refuses_existing=rename), False)
                self.assertEqual(self.target.read_bytes(), b"someone else's file")
                self.assertTrue(self.partial.exists(), "the caller discards the partial download")

    def test_without_hard_links_it_falls_back_to_check_then_replace(self):
        refused = OSError(errno.EPERM, "hard links are not supported here")
        with mock.patch.object(fetcher.os, "link", side_effect=refused):
            self.target.write_bytes(b"someone else's file")
            self.assertIs(fetcher.place_new_file(str(self.partial), str(self.target), rename_refuses_existing=False), False)
            self.assertEqual(self.target.read_bytes(), b"someone else's file")
            self.target.unlink()
            self.assertIs(fetcher.place_new_file(str(self.partial), str(self.target), rename_refuses_existing=False), True)
        self.assertEqual(self.target.read_bytes(), b"new download")

    def test_the_default_strategy_follows_the_platform(self):
        import inspect

        default = inspect.signature(fetcher.place_new_file).parameters["rename_refuses_existing"].default
        self.assertIs(default, os.name == "nt")


ANIMA_LISTING = [
    {"model_name": "qwen_image_vae.safetensors", "filename": "/models/VAE/qwen_image_vae.safetensors"},
    {"model_name": "Qwen2D-Anime-dense_epoch_1.safetensors", "filename": "/models/VAE/Qwen2D-Anime-dense_epoch_1.safetensors"},
    {"model_name": "qwen_3_06b_base.safetensors", "filename": "/models/text_encoder/qwen_3_06b_base.safetensors"},
    {"model_name": "qwen35_4b.safetensors", "filename": "/models/text_encoder/qwen35_4b.safetensors"},
    {"model_name": "Anima-3.8B-expanded_adapter.safetensors", "filename": "/models/text_encoder/Anima-3.8B-expanded_adapter.safetensors"},
    {"model_name": "qwen3vl_4b_bf16.safetensors", "filename": "/models/text_encoder/qwen3vl_4b_bf16.safetensors"},
]


class _Listing:
    def modules(self):
        return ApiResult(True, data=ANIMA_LISTING)


class SamExtraModuleTableTests(unittest.TestCase):
    def test_anima_3_8b_setup_is_not_called_a_leftover(self):
        selected = ["qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors", "qwen35_4b.safetensors",
                    "Anima-3.8B-expanded_adapter.safetensors"]
        report = audit("anima", selected, ANIMA_LISTING)
        self.assertNotIn("unrecognised", report)
        self.assertEqual(report["missing"], [])
        self.assertEqual([item["module"] for item in report["extras_loaded"]],
                         ["qwen35_4b.safetensors", "Anima-3.8B-expanded_adapter.safetensors"])
        self.assertIn("sam-extra", report["extras_loaded"][0]["what"])
        health = _module_health(_Listing(), {"forge_additional_modules_anima": selected}, "anima")
        self.assertIs(health["healthy"], True)
        self.assertEqual(health["extras_loaded"], ["qwen35_4b.safetensors", "Anima-3.8B-expanded_adapter.safetensors"])

    def test_qwen35_does_not_stand_in_for_the_qwen3_encoder(self):
        report = audit("anima", ["qwen_image_vae.safetensors", "qwen35_4b.safetensors"], ANIMA_LISTING)
        missing = [item["need"] for item in report["missing"]]
        self.assertEqual(missing, ["Qwen3 0.6B"])
        self.assertEqual(report["missing"][0]["candidates_installed"], ["qwen_3_06b_base.safetensors"])
        self.assertNotIn("unrecognised", report)

    def test_other_variants_of_the_markers(self):
        extras = ARCH_MODULES["anima"].extras
        for name in ("Qwen3.5-4B-bf16.safetensors", "qwen3_5_4b_fp8.safetensors", "anima_3_8b_adapter.safetensors"):
            self.assertTrue(any(extra.matches(name) for extra in extras), name)

    def test_qwen2d_vae_family_fills_the_qwen_image_vae_slot(self):
        report = audit("anima", ["Qwen2D-Anime-dense_epoch_1.safetensors", "qwen_3_06b_base.safetensors"], ANIMA_LISTING)
        self.assertEqual(report["missing"], [])
        self.assertNotIn("unrecognised", report)

    def test_a_real_leftover_is_still_reported(self):
        report = audit("anima", ["qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors", "qwen3vl_4b_bf16.safetensors"],
                       ANIMA_LISTING)
        self.assertEqual(report["unrecognised"]["modules"], ["qwen3vl_4b_bf16.safetensors"])

    def test_extras_are_anima_only(self):
        report = audit("krea", ["qwen_image_vae.safetensors", "qwen3vl_4b_bf16.safetensors", "qwen35_4b.safetensors"],
                       ANIMA_LISTING)
        self.assertNotIn("extras_loaded", report)
        self.assertEqual(report["unrecognised"]["modules"], ["qwen35_4b.safetensors"])


@unittest.skipUnless(HAS_HTTPX, "httpx is not installed")
class ServiceDownloadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_allowed_download_goes_into_the_folder_forge_scans(self):
        content = safetensors_blob()
        hub = HuggingFace(content)
        service, fake, tree = make_service(self.base, settings={ALLOW_DOWNLOAD: True}, download_transport=hub.transport())
        self.addCleanup(service.close)
        listing = service.module_download()
        self.assertIs(listing["downloads_allowed"], True)
        self.assertEqual(listing["models_dir"], str(tree["forge"] / "models"))
        plan = service.module_download(label="Qwen3 0.6B base")
        self.assertIs(plan["plan"]["probed"], True)
        self.assertEqual(plan["plan"]["verification"], "size and sha256")
        self.assertNotIn("GET", [method for method, _ in hub.requests])
        result = service.module_download(label="Qwen3 0.6B base", confirm=True)
        self.assertIs(result["ok"], True, result)
        target = tree["forge"] / "models" / "text_encoder" / "qwen_3_06b_base.safetensors"
        self.assertEqual(target.read_bytes(), content)
        self.assertEqual(result["verified"], "size and sha256")

    def test_models_folder_comes_from_the_installation_not_from_module_paths(self):
        # A module path with a "vae" folder higher up used to be cut there, sending the download
        # to <base>/VAE. The target is now <Forge data>/models/<category>, whatever Forge lists.
        misleading = self.base / "vae" / "elsewhere" / "models" / "VAE"
        misleading.mkdir(parents=True)
        (self.base / "VAE").mkdir(exist_ok=True)
        service, fake, tree = make_service(self.base, settings={})
        self.addCleanup(service.close)
        fake.sd_modules = [{"model_name": "x.safetensors", "filename": str(misleading / "x.safetensors")}]
        fake.options["forge_additional_modules_anima"] = [str(misleading / "x.safetensors")]
        plan = service.module_download(label="Qwen-Image VAE")
        self.assertEqual(plan["plan"]["destination"],
                         str(tree["forge"] / "models" / "VAE" / "qwen_image_vae.safetensors"))
        listing = service.module_download()
        self.assertEqual(listing["models_dir"], str(tree["forge"] / "models"))

    def test_models_folder_override(self):
        shared = self.base / "shared-models"
        (shared / "VAE").mkdir(parents=True)
        service, fake, tree = make_service(self.base, settings={}, env={"SAM_EXTRA_MCP_MODELS_DIR": str(shared)})
        self.addCleanup(service.close)
        plan = service.module_download(label="Qwen-Image VAE")
        self.assertEqual(plan["plan"]["destination"], str(shared / "VAE" / "qwen_image_vae.safetensors"))

    def test_a_missing_models_folder_is_reported_not_guessed(self):
        service, fake, tree = make_service(self.base, settings={},
                                           env={"SAM_EXTRA_MCP_MODELS_DIR": str(self.base / "nowhere")})
        self.addCleanup(service.close)
        result = service.module_download(label="Qwen-Image VAE")
        self.assertIs(result["ok"], False)
        self.assertIn("cannot locate the Forge models folder", result["error"])

    def test_folder_links_are_not_fetched(self):
        service, fake, _ = make_service(self.base, settings={ALLOW_DOWNLOAD: True})
        self.addCleanup(service.close)
        result = service.module_download(preset="krea", label="Qwen3-VL 4B (bf16 / fp8_scaled)", confirm=True)
        self.assertIs(result["fetchable"], False)

    def test_unknown_label_lists_the_choices(self):
        service, fake, _ = make_service(self.base, settings={ALLOW_DOWNLOAD: True})
        self.addCleanup(service.close)
        result = service.module_download(label="nope", confirm=True)
        self.assertIs(result["ok"], False)
        self.assertIn("Qwen3 0.6B base", result["choices"])


if __name__ == "__main__":
    unittest.main()
