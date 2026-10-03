"""sam-extra MCP server: generation results - the upstream defects and their fixes.

- Without FORGE_PATH_MAP upstream found no output folder and answered ok:false after Forge had
  rendered; here the folder comes from the extension location, and whatever is not found on disk
  is saved from the API response into the fallback folder.
- Upstream returned any new file in the output folder, so a concurrent generation (web UI,
  another agent) came back as ours; here files are matched by infotext, seed and time.
- The API ignores the "Output Directory" override (outdir_samples); results are looked for where
  the API actually saves.

Every test drives the real service against a fake Forge API that writes real files.
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from _mcp_support import (
    HAS_HTTPX,
    infotext,
    jpeg_bytes,
    make_service,
    png_bytes,
    webp_bytes,
)

from sam_extra_mcp.forgeneo import generate as gen
from sam_extra_mcp.forgeneo.client import ApiResult
from sam_extra_mcp.forgeneo.config import Config
from sam_extra_mcp.forgeneo.infotext import read_generation_metadata, read_parameters_from_bytes


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


@unittest.skipUnless(HAS_HTTPX, "httpx is not installed")
class GenerateServiceTests(_Tmp):
    def service(self, **kwargs):
        service, fake, tree = make_service(self.base, settings={}, **kwargs)
        self.addCleanup(service.close)
        return service, fake, tree

    def test_results_are_found_without_a_path_map(self):
        service, fake, tree = self.service()
        result = service.generate("a lighthouse at dusk", steps=4, seed=77)
        self.assertIs(result["ok"], True, result)
        self.assertEqual(result["delivery"], "filesystem")
        self.assertEqual(len(result["files"]), 1)
        self.assertTrue(result["files"][0].startswith(str(tree["output"]).replace("\\", "/")))
        self.assertEqual(result["images"][0]["seed"], 77)
        self.assertEqual(result["images"][0]["matched_by"], gen.MATCH_INFOTEXT)

    def test_a_concurrent_generation_is_not_returned(self):
        service, fake, _ = self.service()

        def someone_else(fake):
            folder = fake.outdir("txt2img")
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "90000-424242.png").write_bytes(png_bytes(infotext("a castle", 424242)))
            # Same prompt, different seed: still someone else's.
            (folder / "90001-555.png").write_bytes(png_bytes(infotext("a lighthouse at dusk", 555)))

        fake.during_generation = someone_else
        result = service.generate("a lighthouse at dusk", steps=4, seed=77, batch_size=2)
        self.assertIs(result["ok"], True, result)
        self.assertEqual([image["seed"] for image in result["images"]], [77, 78])
        names = [os.path.basename(path) for path in result["files"]]
        self.assertNotIn("90000-424242.png", names)
        self.assertNotIn("90001-555.png", names)
        self.assertEqual(result["ignored_new_files"]["count"], 2)

    def test_files_without_metadata_are_matched_by_seed_in_the_name(self):
        service, fake, _ = self.service()
        fake.embed = False  # enable_pnginfo off, no .txt either

        def someone_else(fake):
            folder = fake.outdir("txt2img")
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "90000-424242.png").write_bytes(png_bytes(None))

        fake.during_generation = someone_else
        result = service.generate("a cat", steps=4, seed=31337)
        self.assertIs(result["ok"], True, result)
        self.assertEqual(result["delivery"], "filesystem")
        self.assertEqual(result["images"][0]["matched_by"], gen.MATCH_SEED_FILENAME)
        self.assertTrue(result["files"][0].endswith("-31337.png"))
        self.assertTrue(any("could not be told apart" in warning for warning in result["warnings"]))

    def test_txt_sidecar_identifies_the_file(self):
        service, fake, _ = self.service()
        fake.embed = False
        fake.save_txt = True
        result = service.generate("a cat", steps=4, seed=4242)
        self.assertEqual(result["images"][0]["matched_by"], gen.MATCH_INFOTEXT)
        self.assertNotIn("warnings", result)

    def test_nothing_saved_by_forge_still_delivers_the_images(self):
        """samples_save off (or an unknown folder): the API response is saved instead."""
        service, fake, tree = self.service()
        fake.save_files = False
        result = service.generate("a cat", steps=4, seed=9, batch_size=2)
        self.assertIs(result["ok"], True, result)
        self.assertEqual(result["delivery"], "api response")
        fallback = tree["forge"] / "sam-extra" / "mcp" / "outputs"
        self.assertEqual(len(result["files"]), 2)
        for path, seed in zip(result["files"], (9, 10)):
            self.assertTrue(path.startswith(str(fallback).replace("\\", "/")))
            self.assertTrue(path.endswith(".png"))
            text, _ = read_generation_metadata(path)
            self.assertIn(f"Seed: {seed}", text)
        self.assertEqual(sorted(p.name for p in fallback.iterdir()), sorted(os.path.basename(p) for p in result["files"]))

    def test_unknown_output_folder_falls_back_instead_of_failing(self):
        # Forge reports an absolute outdir this machine cannot reach.
        service, fake, tree = self.service()
        fake.options["outdir_txt2img_samples"] = str(self.base / "elsewhere" / "deep" / "folder")
        fake.save_files = False
        result = service.generate("a cat", steps=4, seed=5)
        self.assertIs(result["ok"], True, result)
        self.assertEqual(result["delivery"], "api response")
        self.assertIn("Forge's output folder is not readable from here", result["warnings"])

    def test_some_on_disk_some_from_the_response(self):
        service, fake, _ = self.service()
        original = fake.write_output

        def lose_the_second(folder, seed, text, *, suffix=""):
            if seed == 21:
                return None
            return original(folder, seed, text, suffix=suffix)

        fake.write_output = lose_the_second
        result = service.generate("a cat", steps=4, seed=20, batch_size=2)
        self.assertIs(result["ok"], True, result)
        self.assertEqual(result["delivery"], "filesystem + api response")
        by_seed = {image["seed"]: image["matched_by"] for image in result["images"]}
        self.assertEqual(by_seed[20], gen.MATCH_INFOTEXT)
        self.assertEqual(by_seed[21], gen.SAVED_BY_INFOTEXT)

    def test_grid_is_never_returned(self):
        service, fake, _ = self.service()
        fake.save_files = False
        result = service.generate("a cat", steps=4, seed=1, batch_size=3)
        self.assertEqual([image["seed"] for image in result["images"]], [1, 2, 3])

    def test_response_without_metadata_is_matched_by_position(self):
        service, fake, _ = self.service()
        fake.save_files = False
        fake.embed = False
        result = service.generate("a cat", steps=4, seed=40, batch_size=2)
        self.assertIs(result["ok"], True, result)
        self.assertEqual([(image["seed"], image["matched_by"]) for image in result["images"]],
                         [(40, gen.SAVED_BY_POSITION), (41, gen.SAVED_BY_POSITION)])

    def test_jpeg_results_are_identified_by_their_exif(self):
        service, fake, _ = self.service()
        fake.samples_format = "jpg"

        def someone_else(fake):
            folder = fake.outdir("txt2img")
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "90000-1.jpg").write_bytes(jpeg_bytes(infotext("a castle", 1)))

        fake.during_generation = someone_else
        result = service.generate("a cat", steps=4, seed=600)
        self.assertEqual(result["delivery"], "filesystem")
        self.assertTrue(result["files"][0].endswith("-600.jpg"))
        self.assertEqual(result["images"][0]["matched_by"], gen.MATCH_INFOTEXT)
        self.assertEqual(result["ignored_new_files"]["count"], 1)

    def test_webp_saved_from_the_response_keeps_its_format(self):
        service, fake, _ = self.service()
        fake.samples_format = "webp"
        fake.save_files = False
        result = service.generate("a cat", steps=4, seed=700)
        self.assertTrue(result["files"][0].endswith(".webp"), result)
        self.assertEqual(result["images"][0]["matched_by"], gen.SAVED_BY_INFOTEXT)

    def test_auxiliary_saves_are_listed_with_the_image(self):
        service, fake, _ = self.service()
        fake.extra_saves = ["-before-highres-fix"]
        result = service.generate("a cat", steps=4, seed=800)
        image = result["images"][0]
        self.assertTrue(image["file"].endswith("-800.png"))
        self.assertEqual(len(image["related_files"]), 1)
        self.assertTrue(image["related_files"][0].endswith("-800-before-highres-fix.png"))
        self.assertEqual(len(result["files"]), 1)

    def test_api_results_ignore_the_output_directory_override(self):
        service, fake, tree = self.service()
        override = self.base / "ui-only"
        override.mkdir()
        fake.options["outdir_samples"] = str(override)
        result = service.generate("a cat", steps=4, seed=900)
        self.assertEqual(result["delivery"], "filesystem", result)
        self.assertTrue(result["files"][0].startswith(str(tree["output"]).replace("\\", "/")))
        # The history follows the operator's own generations into the override folder.
        self.assertEqual(service.history.output_dir, str(override).replace("\\", "/"))

    def test_img2img_results_come_from_the_img2img_folder(self):
        service, fake, tree = self.service()
        init = self.base / "init.png"
        init.write_bytes(png_bytes(None))
        result = service.generate("repaint", init_image=str(init), denoising_strength=1.7, steps=4, seed=12)
        self.assertEqual(result["mode"], "img2img")
        self.assertEqual(result["delivery"], "filesystem", result)
        self.assertIn("/output/img2img-images/", result["files"][0])
        sent = fake.calls("POST", "/sdapi/v1/img2img")[0][2]
        self.assertEqual(sent["denoising_strength"], 1.0)
        self.assertEqual(base64.b64decode(sent["init_images"][0]), init.read_bytes())

    def test_out_of_range_requests_are_clamped_and_reported(self):
        service, fake, _ = self.service()
        result = service.generate("a cat", steps=5000, width=99999, height=16, batch_size=64, seed=1)
        sent = fake.calls("POST", "/sdapi/v1/txt2img")[0][2]
        self.assertEqual((sent["steps"], sent["width"], sent["height"], sent["batch_size"]), (150, 4096, 64, 8))
        self.assertEqual(result["adjusted"]["batch_size"], {"asked": 64, "used": 8})
        self.assertEqual(set(result["adjusted"]), {"steps", "width", "height", "batch_size"})

    def test_empty_prompt_is_refused_before_forge(self):
        service, fake, _ = self.service()
        self.assertIs(service.generate("   ")["ok"], False)
        self.assertEqual(fake.calls("POST"), [])

    def test_forge_errors_are_reported(self):
        service, fake, _ = self.service()
        fake.fail_generation = (500, "CUDA out of memory")
        result = service.generate("a cat", steps=4)
        self.assertIs(result["ok"], False)
        self.assertIn("HTTP 500", result["error"])
        self.assertIn("CUDA out of memory", result["error"])

    def test_generation_refreshes_the_history(self):
        service, fake, _ = self.service()
        profile = service.model_profile()
        self.assertEqual(profile["samples_observed"], 0)
        for seed in range(6):
            self.assertIs(service.generate("1girl, solo, masterpiece", steps=11, seed=seed)["ok"], True)
        profile = service.model_profile()
        self.assertEqual(profile["samples_observed"], 6, "generations made after start-up are seen")
        self.assertEqual(profile["recommended"]["steps"], 11.0)

    def test_unknown_progress_action_is_an_error(self):
        service, fake, _ = self.service()
        result = service.progress(action="stop")
        self.assertIs(result["ok"], False)
        self.assertEqual(result["choices"], ["status", "interrupt", "skip"])
        self.assertEqual(fake.calls("POST"), [])

    def test_progress_skips_the_preview_image(self):
        service, fake, _ = self.service()
        self.assertIs(service.progress()["ok"], True)
        self.assertEqual([params for path, params in fake.queries if path == "/sdapi/v1/progress"],
                         [{"skip_current_image": "true"}])


class MatchingUnitTests(_Tmp):
    def expected(self, *seeds, prompt="a cat"):
        return [gen.ExpectedImage(i, seed, prompt, infotext(prompt, seed)) for i, seed in enumerate(seeds)]

    def test_infotext_beats_seed(self):
        files = []
        for name, data in (("a-5.png", png_bytes(infotext("a cat", 5))), ("b.png", png_bytes(infotext("a cat", 5, steps=99)))):
            path = self.base / name
            path.write_bytes(data)
            files.append((time.time(), str(path)))
        chosen, foreign, unattributed = gen._match_files(files, self.expected(5))
        self.assertEqual(os.path.basename(chosen[0].path), "a-5.png")
        self.assertEqual(chosen[0].matched_by, gen.MATCH_INFOTEXT)
        self.assertEqual(len(chosen[0].related), 1)

    def test_a_grid_in_the_samples_folder_never_stands_in_for_an_image(self):
        grid = self.base / "grid-0001.png"
        grid.write_bytes(png_bytes(infotext("a cat", 5)))  # Forge writes the first image's infotext
        chosen, foreign, unattributed = gen._match_files([(time.time(), str(grid))], self.expected(5))
        self.assertEqual((chosen, foreign, unattributed), ({}, (), ()))

    def test_adetailer_previews_are_auxiliary(self):
        self.assertTrue(gen._is_auxiliary("00001-5-ad-preview-1.png"))
        self.assertTrue(gen._is_auxiliary("00001-5-ad-preview.png"))
        self.assertTrue(gen._is_auxiliary("00001-5-ad-before.png"))
        self.assertFalse(gen._is_auxiliary("00001-5.png"))

    def test_seed_with_a_different_prompt_is_foreign(self):
        path = self.base / "x.png"
        path.write_bytes(png_bytes(infotext("a dog", 5)))
        chosen, foreign, _ = gen._match_files([(time.time(), str(path))], self.expected(5))
        self.assertEqual(chosen, {})
        self.assertEqual(len(foreign), 1)

    def test_seed_in_name_needs_word_boundaries(self):
        path = self.base / "00012-1234567.png"
        path.write_bytes(png_bytes(None))
        self.assertIsNone(gen._match_filename(str(path), self.expected(12)))
        self.assertEqual(gen._match_filename(str(path), self.expected(1234567))[0], 0)

    def test_crlf_sidecar_matches(self):
        image = self.base / "00001-5.png"
        image.write_bytes(png_bytes(None))
        (self.base / "00001-5.txt").write_bytes(infotext("a cat", 5).replace("\n", "\r\n").encode("utf-8") + b"\r\n")
        chosen, _, _ = gen._match_files([(time.time(), str(image))], self.expected(5))
        self.assertEqual(chosen[0].matched_by, gen.MATCH_INFOTEXT)

    def test_files_outside_the_request_window_are_ignored(self):
        old = self.base / "old.png"
        old.write_bytes(png_bytes(infotext("a cat", 5)))
        past = time.time() - 60
        os.utime(old, (past, past))
        late = self.base / "late.png"
        late.write_bytes(png_bytes(infotext("a cat", 5)))
        future = time.time() + 60
        os.utime(late, (future, future))
        started = time.time() - 1
        self.assertEqual(gen._new_media(str(self.base), set(), started, time.time()), [])

    def test_expected_images_follow_index_of_first_image(self):
        info = {"index_of_first_image": 1, "infotexts": ["grid", "one", "two"], "all_seeds": [7, 8],
                "all_prompts": ["p", "p"]}
        expected = gen.expected_images(info, {"prompt": "p"}, 3)
        self.assertEqual([(e.index, e.seed, e.infotext) for e in expected], [(0, 7, "one"), (1, 8, "two")])
        no_info = gen.expected_images(None, {"prompt": "p"}, 2)
        self.assertEqual([(e.index, e.seed) for e in no_info], [(0, None), (1, None)])

    def test_saved_files_never_overwrite(self):
        blob = png_bytes(None)
        first = gen._write_new_file(str(self.base), "same", blob)
        second = gen._write_new_file(str(self.base), "same", blob)
        self.assertNotEqual(first, second)
        self.assertTrue(second.endswith("same-1.png"))
        self.assertIsNone(gen._write_new_file(str(self.base), "unknown", b"not an image"))

    def test_image_extension_from_signature(self):
        self.assertEqual(gen.image_extension(png_bytes(None)), ".png")
        self.assertEqual(gen.image_extension(jpeg_bytes(None)), ".jpg")
        self.assertEqual(gen.image_extension(webp_bytes(None)), ".webp")
        self.assertEqual(gen.image_extension(b"\x00\x00\x00\x1cftypavif" + b"\x00" * 8), ".avif")
        self.assertIsNone(gen.image_extension(b"GIF89a"))

    def test_run_generation_without_a_fallback_reports_the_render(self):
        class _Client:
            config = Config()

            def txt2img(self, payload):
                info = {"all_seeds": [3], "infotexts": [infotext("a cat", 3)], "all_prompts": ["a cat"]}
                return ApiResult(True, data={"images": [base64.b64encode(png_bytes(None)).decode()], "info": json.dumps(info)})

        result = gen.run_generation(_Client(), {"prompt": "a cat"}, None, fallback_dir=None)
        self.assertIs(result.ok, False)
        self.assertIn("Forge generated 1 image(s) (seeds 3)", result.error)

    def test_init_image_guards(self):
        video = self.base / "clip.mp4"
        video.write_bytes(b"\x00" * 16)
        self.assertIn("not a recognised image", gen.encode_init_image(str(video))[1])
        big = self.base / "big.png"
        big.write_bytes(b"x")
        with mock.patch.object(gen, "MAX_INIT_IMAGE_BYTES", 0):
            self.assertIn("larger than", gen.encode_init_image(str(big))[1])


class PixelBudgetTests(_Tmp):
    """One generate call may render at most config.max_pixels (SAM_EXTRA_MCP_MAX_PIXELS, 2048x2048x4)."""

    def test_requested_pixels(self):
        cases = (
            ({"width": 1024, "height": 1024, "batch_size": 2}, 2 * 1024 * 1024),
            ({"width": 512, "height": 512, "batch_size": 2, "n_iter": 3}, 6 * 512 * 512),
            ({"width": 1024, "height": 1024, "enable_hr": True, "hr_scale": 2}, 2048 * 2048),
            ({"width": 1024, "height": 1024, "enable_hr": True}, 2048 * 2048),  # Forge's default scale
            ({"width": 1024, "height": 512, "enable_hr": True, "hr_resize_x": 2048, "hr_resize_y": 0}, 2048 * 1024),
            ({"width": 1024, "height": 512, "enable_hr": True, "hr_resize_x": 0, "hr_resize_y": 1024}, 2048 * 1024),
            ({"width": 1024, "height": 1024, "enable_hr": True, "hr_scale": 0.5}, 1024 * 1024),  # first pass is larger
            ({"width": 1024, "height": 1024, "enable_hr": False, "hr_scale": 4}, 1024 * 1024),
            ({"width": "x", "height": None, "batch_size": "nan"}, 0),
        )
        for payload, expected in cases:
            with self.subTest(payload=payload):
                self.assertEqual(gen.requested_pixels(payload), expected)

    def test_the_default_budget(self):
        self.assertEqual(Config().max_pixels, 2048 * 2048 * 4)
        self.assertEqual(Config.from_env({}).max_pixels, 2048 * 2048 * 4)

    @unittest.skipUnless(HAS_HTTPX, "httpx is not installed")
    def test_four_2048_images_fit_and_five_do_not(self):
        service, fake, _ = make_service(self.base, settings={})
        self.addCleanup(service.close)
        self.assertIs(service.generate("a cat", width=2048, height=2048, batch_size=4, steps=4, seed=1)["ok"], True)
        refused = service.generate("a cat", width=2048, height=2048, batch_size=5, steps=4, seed=1)
        self.assertIs(refused["ok"], False)
        self.assertIn("megapixels", refused["error"])
        self.assertIn("SAM_EXTRA_MCP_MAX_PIXELS", refused["error"])
        self.assertEqual((refused["pixels"], refused["max_pixels"]), (5 * 2048 * 2048, 4 * 2048 * 2048))
        self.assertEqual(len(fake.calls("POST", "/sdapi/v1/txt2img")), 1, "the refused call never reached Forge")

    @unittest.skipUnless(HAS_HTTPX, "httpx is not installed")
    def test_the_operator_can_change_the_budget(self):
        service, fake, _ = make_service(self.base, settings={}, env={"SAM_EXTRA_MCP_MAX_PIXELS": "300_000"})
        self.addCleanup(service.close)
        self.assertEqual(service.capabilities()["server"]["config"]["max_pixels_per_generate"], 300_000)
        self.assertIs(service.generate("a cat", width=512, height=512, steps=4, seed=1)["ok"], True)
        self.assertIs(service.generate("a cat", width=512, height=512, batch_size=2, steps=4, seed=1)["ok"], False)
        self.assertEqual(len(fake.calls("POST", "/sdapi/v1/txt2img")), 1)

    @unittest.skipUnless(HAS_HTTPX, "httpx is not installed")
    def test_sizes_taken_from_the_profile_count_too(self):
        # No size given: the profile supplies the preset's 832x1216 (1.01 MP), over a 1 MP budget.
        service, fake, _ = make_service(self.base, settings={}, env={"SAM_EXTRA_MCP_MAX_PIXELS": "1000000"})
        self.addCleanup(service.close)
        refused = service.generate("a cat", steps=4, seed=1)
        self.assertIs(refused["ok"], False, refused)
        self.assertEqual(refused["pixels"], 832 * 1216)
        self.assertIn("width", refused["defaults_applied"])
        self.assertEqual(fake.calls("POST", "/sdapi/v1/txt2img"), [])

    def test_unusable_settings_keep_the_default(self):
        from sam_extra_mcp.forgeneo.config import DEFAULT_MAX_PIXELS, parse_max_pixels

        self.assertEqual(parse_max_pixels("16_777_216"), 16_777_216)
        self.assertEqual(parse_max_pixels(" 1.6e7 "), 16_000_000)
        self.assertEqual(parse_max_pixels("8,000,000"), 8_000_000)
        for raw in (None, "", "0", "-5", "lots", "inf", "nan"):
            self.assertEqual(parse_max_pixels(raw), DEFAULT_MAX_PIXELS, raw)


def _untouchable(*args, **kwargs):
    raise AssertionError(f"the file system was touched: {args!r}")


class InitImagePathTests(_Tmp):
    """img2img reads the agent's `init_image` path: it must stay a file on this PC.

    On Windows, merely asking about \\\\host\\share\\x.png makes the OS contact that host (SMB,
    or WebDAV over HTTP) with the user's NTLM credentials and puts the name in DNS - network
    traffic of the agent's choosing, generation being allowed by default. Every test here
    replaces the file-system calls, so a regression fails instead of reaching for a host
    (the .invalid names never resolve either).
    """

    NETWORK_PATHS = ["//host.invalid/share/x.png"] + (
        [r"\\host.invalid\share\x.png", r"\\?\UNC\host.invalid\share\x.png", r"\\.\pipe\x.png",
         r"\\host.invalid@SSL\DavWWWRoot\x.png"]
        if os.name == "nt" else []
    )

    def test_network_and_device_paths_are_refused_untouched(self):
        with mock.patch("os.path.isfile", _untouchable), mock.patch("os.path.getsize", _untouchable), \
                mock.patch("builtins.open", _untouchable):
            for path in self.NETWORK_PATHS:
                with self.subTest(path=path):
                    data, error = gen.encode_init_image(path)
                    self.assertIsNone(data)
                    self.assertIn("not a network share or device path", error)

    @unittest.skipUnless(os.name == "nt", "the \\??\\ prefix is a Win32 spelling")
    def test_nt_namespace_spelling_is_opened_as_a_local_name(self):
        # \??\UNC\host\share\x.png carries no leading pair, yet Win32 hands \??\ names to the NT
        # namespace as they are - os.path.isfile(r"\??\C:\Windows\win.ini") is True. The file is
        # looked up through its normalised absolute path (C:\??\...), which names nothing.
        seen = []

        def isfile(path):
            seen.append(path)
            return False

        with mock.patch("os.path.isfile", isfile), mock.patch("builtins.open", _untouchable):
            data, error = gen.encode_init_image(r"\??\UNC\host.invalid\share\x.png")
        self.assertIsNone(data)
        self.assertIn("not found", error)
        self.assertEqual(len(seen), 1)
        self.assertRegex(seen[0], r"^[A-Za-z]:\\", "looked up as a local name, not as \\??\\UNC\\...")

    def test_the_suffix_is_checked_before_the_file_system(self):
        with mock.patch("os.path.isfile", _untouchable):
            data, error = gen.encode_init_image(str(self.base / "secrets.txt"))
        self.assertIsNone(data)
        self.assertIn("not a recognised image", error)

    def test_local_files_still_work_including_non_ascii_paths(self):
        folder = self.base / "한글 폴더"
        folder.mkdir()
        image = folder / "참고 이미지.png"
        image.write_bytes(png_bytes(None))
        data, error = gen.encode_init_image(str(image))
        self.assertIsNone(error)
        self.assertEqual(base64.b64decode(data), image.read_bytes())
        try:
            relative = os.path.relpath(image)
        except ValueError:  # a temp folder on another drive has no relative path
            return
        self.assertIsNone(gen.encode_init_image(relative)[1])

    @unittest.skipUnless(HAS_HTTPX, "httpx is not installed")
    def test_generate_refuses_before_any_request(self):
        service, fake, _ = make_service(self.base, settings={})
        self.addCleanup(service.close)
        with mock.patch("os.path.isfile", _untouchable):
            result = service.generate("repaint", init_image="//host.invalid/share/x.png")
        self.assertIs(result["ok"], False)
        self.assertIn("not a network share or device path", result["error"])
        self.assertEqual(fake.requests, [])


class MetadataReaderTests(_Tmp):
    TEXT = infotext("café, 1girl, 고양이", 5)

    def test_png_text_chunk_is_latin1_per_spec(self):
        path = self.base / "a.png"
        path.write_bytes(png_bytes("café latte\nSteps: 1, Seed: 5"))
        self.assertEqual(read_generation_metadata(str(path))[0], "café latte\nSteps: 1, Seed: 5")

    def test_png_itxt_carries_utf8(self):
        path = self.base / "a.png"
        path.write_bytes(png_bytes(self.TEXT))
        self.assertEqual(read_generation_metadata(str(path)), (self.TEXT, "png_chunk"))

    def test_compressed_itxt_and_ztxt(self):
        import struct
        import zlib

        def chunk(kind, body):
            return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

        head = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", b"\x00" * 13)
        itxt = head + chunk(b"iTXt", b"parameters\x00\x01\x00\x00\x00" + zlib.compress(self.TEXT.encode("utf-8"))) + chunk(b"IEND", b"")
        ztxt = head + chunk(b"zTXt", b"parameters\x00\x00" + zlib.compress(b"plain text\nSteps: 2")) + chunk(b"IEND", b"")
        self.assertEqual(read_parameters_from_bytes(itxt), (self.TEXT, "png_chunk"))
        self.assertEqual(read_parameters_from_bytes(ztxt), ("plain text\nSteps: 2", "png_chunk"))

    def test_jpeg_exif_user_comment(self):
        for order in (">", "<"):
            path = self.base / f"a{order == '<'}.jpg"
            path.write_bytes(jpeg_bytes(self.TEXT, order=order))
            self.assertEqual(read_generation_metadata(str(path)), (self.TEXT, "exif_user_comment"))

    def test_webp_exif_with_and_without_header(self):
        for flag in (False, True):
            path = self.base / f"a{flag}.webp"
            path.write_bytes(webp_bytes(self.TEXT, with_exif_header=flag))
            self.assertEqual(read_generation_metadata(str(path)), (self.TEXT, "exif_user_comment"))

    def test_jpeg_without_exif_uses_the_sidecar(self):
        path = self.base / "a.jpg"
        path.write_bytes(jpeg_bytes(None))
        (self.base / "a.txt").write_text("from the sidecar\nSteps: 3", encoding="utf-8")
        self.assertEqual(read_generation_metadata(str(path)), ("from the sidecar\nSteps: 3", "txt_sidecar"))

    def test_garbage_is_not_metadata(self):
        self.assertEqual(read_parameters_from_bytes(b"\xff\xd8\xff\xe1\x00"), (None, "none"))
        self.assertEqual(read_parameters_from_bytes(b"RIFF\x00\x00\x00\x00WEBPXXXX"), (None, "none"))
        self.assertEqual(read_parameters_from_bytes(b""), (None, "none"))


if __name__ == "__main__":
    unittest.main()
