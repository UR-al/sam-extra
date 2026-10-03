# Derived from https://github.com/eduardoabreu81/forgeneo-mcp at commit a103dc5 (tests/*.py)
# for sam-extra (tests/test_mcp_forgeneo_upstream.py).
#
# Copyright (c) 2026 Eduardo Abreu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Changes in sam-extra (2026-10-03): the sixteen upstream test files merged into one module,
# one TestCase per file, pytest turned into unittest; the adaptations are listed below.

"""forgeneo-mcp's own test suite (tests/ at commit a103dc5), ported from pytest to unittest.

One TestCase per upstream file, same test names, against the vendored copy in
``mcp_server/sam_extra_mcp/forgeneo``. Adaptations, each marked where it happens:

- ``tmp_path`` / ``monkeypatch`` became ``tempfile`` / ``unittest.mock``;
- no test reaches the network: upstream's unreachable-download test connected to 127.0.0.1:1,
  here a mock transport refuses the connection; the disk-space test patches ``remote_file``
  (sam-extra's replacement for ``remote_size``);
- ``fetcher.fetch`` now needs the policy decision (``permitted=True``) - the behaviour under test
  is unchanged;
- ``Config()`` still returns None without a mapping (``same_machine`` defaults to False);
  sam-extra's same-machine mapping is tested in test_mcp_forge_paths.py;
- the dialect cache points at a temporary folder instead of the real ``~/.forgeneo-mcp``;
- test_no_local_statistics: test_readme_prose_carries_no_environment_statistics and
  test_readme_guard_reads_the_file are not ported (they read upstream's README, which is not
  vendored); the runtime-text check covers the vendored package and sam-extra's own modules.
  That makes 128 of upstream's 130 tests.
"""
from __future__ import annotations

import ast
import base64
import os
import re
import struct
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from unittest import mock

from _mcp_support import HAS_HTTPX, MCP_PROJECT, httpx

import sam_extra_mcp.forgeneo.dialects as dialects
from sam_extra_mcp.forgeneo import downloads, fetcher, identity
from sam_extra_mcp.forgeneo.client import ApiResult
from sam_extra_mcp.forgeneo.config import Config, _parse_path_map
from sam_extra_mcp.forgeneo.generate import _new_files, _normalise, _trim_info, build_payload
from sam_extra_mcp.forgeneo.history import HistoryIndex, _looks_danbooru, _summarise, normalise_checkpoint
from sam_extra_mcp.forgeneo.identity import DialectResolution, lora_ecosystem, resolve
from sam_extra_mcp.forgeneo.infotext import parse_infotext, read_generation_metadata
from sam_extra_mcp.forgeneo.modules import ARCH_MODULES, audit, classify, requirements_for
from sam_extra_mcp.forgeneo.presets import defaults_for, detect_lineage, looks_like_accelerator
from sam_extra_mcp.forgeneo.profile import (
    _module_health,
    assess_turbo,
    instance_defaults,
    preset_for_checkpoint,
    switch_checkpoint,
)


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        identity.configure(self.tmp_path / "dialect-cache")

    def tearDown(self):
        identity.configure(None)
        self._tmp.cleanup()


# -- tests/test_checkpoint_switch.py -------------------------------------------------------------

SWITCH_OPTIONS = {
    "sd_model_checkpoint": "Anime/animeMix_v10.safetensors",
    "forge_preset": "anima",
    "forge_checkpoint_anima": "Anime/animeMix_v10.safetensors",
    "forge_checkpoint_krea": "Krea/kreaMix_v20.safetensors",
    "forge_additional_modules_anima": ["/models/VAE/anime_vae.safetensors"],
    "forge_additional_modules_krea": ["/models/VAE/krea_vae.safetensors"],
}
SWITCH_CONFLICTING = dict(SWITCH_OPTIONS, forge_checkpoint_anima="Krea/museMix_v35.gguf")


class _SwitchClient:
    def __init__(self, options=None, ok=True):
        self._options = options if options is not None else dict(SWITCH_OPTIONS)
        self._ok = ok
        self.posted: dict | None = None

    def options(self):
        return ApiResult(self._ok, data=self._options, error=None if self._ok else "unreachable")

    def set_options(self, values):
        self.posted = values
        return ApiResult(True, data={})


class CheckpointSwitchTests(unittest.TestCase):
    """Switching checkpoint has to carry the architecture's modules along."""

    def test_finds_the_preset_that_owns_a_checkpoint(self):
        self.assertEqual(preset_for_checkpoint(SWITCH_OPTIONS, "Krea/kreaMix_v20.safetensors"), "krea")
        self.assertEqual(preset_for_checkpoint(SWITCH_OPTIONS, "kreaMix_v20.safetensors"), "krea")
        self.assertIsNone(preset_for_checkpoint(SWITCH_OPTIONS, "unknownModel.safetensors"))

    def test_switching_architecture_brings_its_modules(self):
        client = _SwitchClient()
        result = switch_checkpoint(client, "Krea/kreaMix_v20.safetensors")
        self.assertIs(result["ok"], True)
        self.assertEqual(client.posted["sd_model_checkpoint"], "Krea/kreaMix_v20.safetensors")
        self.assertEqual(client.posted["forge_preset"], "krea")
        self.assertEqual(client.posted["forge_additional_modules"], ["/models/VAE/krea_vae.safetensors"])
        self.assertEqual(result["preset"], "krea")

    def test_switching_within_the_same_architecture_leaves_modules_alone(self):
        client = _SwitchClient()
        result = switch_checkpoint(client, "Anime/animeMix_v10.safetensors")
        self.assertEqual(client.posted, {"sd_model_checkpoint": "Anime/animeMix_v10.safetensors"})
        self.assertEqual(result["applied"], {})

    def test_unclaimed_checkpoint_warns_instead_of_guessing(self):
        client = _SwitchClient()
        result = switch_checkpoint(client, "somethingNew.safetensors")
        self.assertIs(result["ok"], True)
        self.assertEqual(result["architecture_confidence"], "unknown")
        self.assertNotIn("forge_preset", client.posted)
        self.assertTrue(any("nothing identifies this checkpoint" in note for note in result["notes"]))

    def test_unreachable_instance_reports_rather_than_switching(self):
        client = _SwitchClient(ok=False)
        result = switch_checkpoint(client, "whatever.safetensors")
        self.assertIs(result["ok"], False)
        self.assertIsNone(client.posted)

    def test_switch_always_flags_the_instance_wide_effect(self):
        result = switch_checkpoint(_SwitchClient(), "Krea/kreaMix_v20.safetensors")
        self.assertIn("web UI", result["warning"])

    def test_conflicting_signals_do_not_switch_preset(self):
        client = _SwitchClient(dict(SWITCH_CONFLICTING))
        result = switch_checkpoint(client, "Krea/museMix_v35.gguf")
        self.assertIs(result["ok"], True)
        self.assertEqual(result["architecture_confidence"], "conflicting")
        self.assertNotIn("forge_preset", client.posted)
        self.assertTrue(any("cannot tell which architecture" in note for note in result["notes"]))

    def test_explicit_preset_overrides_inference(self):
        client = _SwitchClient(dict(SWITCH_CONFLICTING))
        result = switch_checkpoint(client, "Krea/museMix_v35.gguf", preset="krea")
        self.assertEqual(result["architecture_confidence"], "stated by caller")
        self.assertEqual(client.posted["forge_preset"], "krea")
        self.assertEqual(client.posted["forge_additional_modules"], ["/models/VAE/krea_vae.safetensors"])

    def test_agreeing_signals_are_reported_as_such(self):
        client = _SwitchClient()
        result = switch_checkpoint(client, "Krea/kreaMix_v20.safetensors")
        self.assertEqual(result["architecture_confidence"], "registry and folder agree")
        self.assertEqual(result["signals"], {"preset_registry": "krea", "folder": "krea"})


# -- tests/test_config_and_presets.py --------------------------------------------------------------

class ConfigAndPresetsTests(unittest.TestCase):
    def test_path_map_parsing_normalises_separators(self):
        pairs = _parse_path_map(r"I:\forge=\\host\I\forge ; C:\x=//host/C/x")
        self.assertEqual(pairs[0][0], "i:/forge")
        self.assertEqual(pairs[0][1], "//host/I/forge")
        self.assertEqual(len(pairs), 2)

    def test_localise_translates_windows_path(self):
        config = Config(path_map=_parse_path_map(r"I:\sd-webui-forge-neo=//host/I/sd-webui-forge-neo"))
        local = config.localise(r"I:\sd-webui-forge-neo\models\Lora\a.safetensors")
        self.assertEqual(local, "//host/I/sd-webui-forge-neo/models/Lora/a.safetensors")

    def test_localise_returns_none_without_mapping(self):
        # Upstream semantics hold for a Config that is not known to share the machine.
        self.assertIsNone(Config().localise(r"I:\forge\models\a.safetensors"))

    def test_localise_is_case_insensitive_on_prefix(self):
        config = Config(path_map=_parse_path_map(r"i:\forge=//host/I/forge"))
        self.assertIsNotNone(config.localise(r"I:\Forge\models\x.safetensors"))

    def test_accelerator_detection_ignores_substring_false_positives(self):
        self.assertIs(looks_like_accelerator("SomeLora", ("hyper-realistic", "character")), False)
        self.assertIs(looks_like_accelerator("MegaThick", ("hyperass", "curvy")), False)

    def test_accelerator_detection_matches_exact_tags(self):
        self.assertIs(looks_like_accelerator("whatever", ("assets", "distillation", "dmd2")), True)
        self.assertIs(looks_like_accelerator("turbo-accel-lora-v1", ()), True)

    def test_arch_defaults_known_and_unknown(self):
        anima = defaults_for("anima")
        self.assertTrue(anima is not None and anima.sampler == "ER SDE")
        self.assertIsNone(defaults_for("nonexistent"))
        self.assertIsNone(defaults_for(None))

    def test_wan_is_the_only_video_arch(self):
        self.assertIs(defaults_for("wan").is_video, True)
        self.assertIs(defaults_for("anima").is_video, False)

    def test_lineage_detection_from_name(self):
        self.assertEqual(detect_lineage("ponyDiffusionV6XL.safetensors"), "pony")
        self.assertEqual(detect_lineage("waiIllustrious_v14"), "illustrious")
        self.assertIsNone(detect_lineage("animeMix_v10Turbo"))


# -- tests/test_dialects.py ----------------------------------------------------------------------

class _FakeLora:
    def __init__(self, base_model):
        self.base_model = base_model


class DialectsTests(_TempDirCase):
    """Dialect resolution has to behave on a clean install."""

    def test_quality_tags_differ_by_lineage(self):
        self.assertEqual(dialects.PONY.quality_prefix[0], "score_9")
        self.assertIn("masterpiece", dialects.ILLUSTRIOUS.quality_prefix)
        self.assertEqual(dialects.NATURAL.quality_prefix, ())
        self.assertEqual(dialects.NATURAL.negative_baseline, ())

    def test_architecture_implies_dialect_when_unambiguous(self):
        self.assertIs(dialects.for_architecture("krea"), dialects.NATURAL)
        self.assertIs(dialects.for_architecture("flux"), dialects.NATURAL)
        self.assertIs(dialects.for_architecture("anima"), dialects.ANIMA)

    def test_xl_is_ambiguous_and_implies_nothing(self):
        self.assertIsNone(dialects.for_architecture("xl"))

    def test_declared_base_model_maps_to_dialect(self):
        self.assertIs(dialects.from_declared_base("Illustrious"), dialects.ILLUSTRIOUS)
        self.assertIs(dialects.from_declared_base("NoobAI XL"), dialects.ILLUSTRIOUS)
        self.assertIs(dialects.from_declared_base("Pony"), dialects.PONY)
        self.assertIs(dialects.from_declared_base("Anima"), dialects.ANIMA)
        self.assertIsNone(dialects.from_declared_base(None))

    def test_observed_prompts_detect_pony_by_score_ladder(self):
        self.assertIs(dialects.from_observed_prompts(["score_9, score_8_up, 1girl, solo"] * 6), dialects.PONY)

    def test_observed_prompts_detect_prose(self):
        prompts = ["a photograph of a lighthouse at dusk with storm clouds"] * 6
        self.assertIs(dialects.from_observed_prompts(prompts), dialects.NATURAL)

    def test_lora_ecosystem_needs_a_clear_majority(self):
        mixed = [_FakeLora("Illustrious")] * 3 + [_FakeLora("Pony")] * 3
        self.assertIsNone(lora_ecosystem(mixed)[0])
        leaning = [_FakeLora("Illustrious")] * 8 + [_FakeLora("Pony")] * 1
        dialect, detail = lora_ecosystem(leaning)
        self.assertIs(dialect, dialects.ILLUSTRIOUS)
        self.assertIn("8 of 9", detail)

    def test_lora_ecosystem_ignores_tiny_samples(self):
        self.assertIsNone(lora_ecosystem([_FakeLora("Illustrious")] * 2)[0])

    def test_clean_install_on_xl_returns_unknown_with_alternatives(self):
        result = resolve(identifier="abc123", architecture="xl")
        self.assertIsInstance(result, DialectResolution)
        self.assertIs(result.known, False)
        self.assertEqual(result.confidence, "unknown")
        self.assertIn("pony", result.alternatives)
        self.assertIn("illustrious", result.alternatives)

    def test_clean_install_on_known_architecture_still_answers(self):
        result = resolve(identifier="abc123", architecture="krea")
        self.assertIs(result.dialect, dialects.NATURAL)
        self.assertEqual(result.source, "architecture")

    def test_declared_base_outranks_architecture(self):
        result = resolve(identifier="x", architecture="xl", declared_base="Pony")
        self.assertIs(result.dialect, dialects.PONY)
        self.assertEqual(result.confidence, "high")

    def test_anima_quality_prefix_matches_official_card(self):
        self.assertEqual(dialects.ANIMA.quality_prefix, ("masterpiece", "best quality", "score_7", "safe"))
        self.assertIn("chromatic aberration", dialects.ANIMA.negative_baseline)
        self.assertIn("score_1", dialects.ANIMA.negative_baseline)

    def test_anima_knows_it_cannot_do_realism(self):
        self.assertIn("photorealism", dialects.ANIMA.avoid)

    def test_anima_tag_style_prefers_spaces(self):
        self.assertIn("spaces instead of underscores", dialects.ANIMA.tag_style)
        self.assertIn("@", dialects.ANIMA.artist_syntax)

    def test_tagged_detection_survives_space_separated_tags(self):
        prompt = "masterpiece, best quality, 1girl, solo, long hair, blue eyes, school uniform"
        self.assertIs(dialects._looks_tagged(prompt), True)

    def test_prose_is_not_mistaken_for_tags(self):
        prompt = (
            "a photograph of a weathered fisherman standing on a harbour dock at first light, "
            "shot on 85mm, natural overcast light"
        )
        self.assertIs(dialects._looks_tagged(prompt), False)


# -- tests/test_downloads.py -----------------------------------------------------------------------

class _Usage:
    def __init__(self, free):
        self.total = free * 2
        self.used = free
        self.free = free


class DownloadsTests(_TempDirCase):
    """Nothing may start without an explicit confirmation, and everything that could go wrong is
    checked first."""

    def test_blob_pages_become_direct_file_urls(self):
        entry = downloads.for_architecture("flux")[0]
        self.assertIn("/blob/", entry.url)
        self.assertIn("/resolve/", entry.direct_url)
        self.assertNotIn("/blob/", entry.direct_url)

    def test_entries_know_where_they_belong(self):
        for arch in downloads.CATALOGUE:
            for entry in downloads.for_architecture(arch):
                self.assertIn(entry.target_folder, ("models/VAE", "models/text_encoder"))
                if entry.is_directory:
                    self.assertNotIn("filename", entry.as_dict())
                else:
                    self.assertTrue(entry.filename.endswith(".safetensors"))

    def test_every_architecture_with_requirements_has_somewhere_to_get_them(self):
        for arch, spec in ARCH_MODULES.items():
            if spec.requirements:
                self.assertTrue(downloads.for_architecture(arch), f"{arch} has requirements but no download entry")

    def test_matching_finds_an_entry_for_a_reported_gap(self):
        hits = downloads.matching("flux", "CLIP-L")
        self.assertTrue(hits and hits[0].label == "CLIP-L")
        self.assertEqual(downloads.matching("flux", "Wan 2.1 VAE"), ())

    def test_sdxl_entry_names_every_lineage_it_serves(self):
        entry = downloads.for_architecture("xl")[0]
        self.assertIn("pony", entry.note.lower())
        self.assertIn("illustrious", entry.note.lower())

    def test_plan_refuses_when_the_file_is_already_there(self):
        existing = self.tmp_path / "clip_l.safetensors"
        existing.write_bytes(b"already here")
        intent = fetcher.plan("https://example.invalid/clip_l.safetensors", str(self.tmp_path), "clip_l.safetensors")
        self.assertIs(intent.already_present, True)
        self.assertIn("already exists", intent.blocked)

    def test_plan_refuses_an_unreachable_folder(self):
        missing = str(self.tmp_path / "nope" / "deeper")
        intent = fetcher.plan("https://example.invalid/x.safetensors", missing, "x.safetensors")
        self.assertTrue(intent.blocked and "does not exist" in intent.blocked)

    def test_plan_refuses_when_the_disk_would_be_left_full(self):
        # sam-extra: remote_file replaces remote_size (it also reports the SHA256).
        with mock.patch.object(fetcher, "remote_file", lambda url, transport=None: fetcher.RemoteFile(9 * 1024 ** 3, None)), \
                mock.patch.object(fetcher.shutil, "disk_usage", lambda path: _Usage(9 * 1024 ** 3)):
            intent = fetcher.plan("https://example.invalid/big.safetensors", str(self.tmp_path), "big.safetensors")
        self.assertTrue(intent.blocked and "not enough room" in intent.blocked)

    def test_fetch_stops_at_a_blocked_plan(self):
        existing = self.tmp_path / "x.safetensors"
        existing.write_bytes(b"present")
        result = fetcher.fetch("https://example.invalid/x.safetensors", str(self.tmp_path), "x.safetensors", permitted=True)
        self.assertIs(result["ok"], False)
        self.assertEqual(existing.read_bytes(), b"present")

    @unittest.skipUnless(HAS_HTTPX, "httpx is not installed")
    def test_partial_downloads_never_look_like_finished_files(self):
        # sam-extra: a mock transport refuses the connection instead of dialling 127.0.0.1:1.
        def refuse(request):
            raise httpx.ConnectError("connection refused", request=request)

        result = fetcher.fetch(
            "http://127.0.0.1:1/never.safetensors", str(self.tmp_path), "never.safetensors",
            permitted=True, transport=httpx.MockTransport(refuse),
        )
        self.assertIs(result["ok"], False)
        self.assertEqual(sorted(p.name for p in self.tmp_path.iterdir() if p.name != "dialect-cache"), [])

    def test_directory_links_are_marked_unfetchable(self):
        krea = downloads.for_architecture("krea")
        folder_entry = next(entry for entry in krea if entry.is_directory)
        self.assertIn("/tree/", folder_entry.url)
        payload = folder_entry.as_dict()
        self.assertIs(payload["fetchable"], False)
        self.assertIn("pick the build", payload["why"])

    def test_krea_offers_both_valid_encoders(self):
        labels = [entry.label for entry in downloads.for_architecture("krea")]
        self.assertTrue(any("Qwen3-VL 4B" in label for label in labels))
        self.assertTrue(any(label == "Qwen3 4B" for label in labels))

    def test_at_least_one_encoder_per_architecture_is_directly_fetchable(self):
        from sam_extra_mcp.forgeneo.modules import TEXT_ENCODER

        for arch, spec in ARCH_MODULES.items():
            if not any(requirement.kind == TEXT_ENCODER for requirement in spec.requirements):
                continue
            entries = [
                entry for entry in downloads.for_architecture(arch)
                if entry.kind == TEXT_ENCODER and not entry.is_directory
            ]
            self.assertTrue(entries, f"{arch} has no directly fetchable text encoder")


# -- tests/test_generate.py ------------------------------------------------------------------------

class GenerateTests(_TempDirCase):
    def test_payload_omits_unset_parameters(self):
        payload = build_payload("a prompt")
        self.assertNotIn("steps", payload)
        self.assertNotIn("cfg_scale", payload)
        self.assertNotIn("sampler_name", payload)
        self.assertIs(payload["save_images"], True)

    def test_payload_includes_given_parameters(self):
        payload = build_payload("a prompt", steps=10, cfg_scale=1.5, sampler_name="ER SDE", scheduler="Beta")
        self.assertEqual(payload["steps"], 10)
        self.assertEqual(payload["cfg_scale"], 1.5)
        self.assertEqual(payload["sampler_name"], "ER SDE")
        self.assertEqual(payload["scheduler"], "Beta")

    def test_new_files_ignores_infotext_sidecars(self):
        started = time.time()
        image = self.tmp_path / "00000-model - 832x1216.png"
        sidecar = self.tmp_path / "00000-model - 832x1216.txt"
        image.write_bytes(b"fake")
        sidecar.write_text("Steps: 10")
        found = _new_files(str(self.tmp_path), before=set(), started=started)
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].endswith(".png"))

    def test_new_files_skips_preexisting(self):
        started = time.time()
        old = self.tmp_path / "old.png"
        old.write_bytes(b"fake")
        before = {str(old)}
        new = self.tmp_path / "new.png"
        new.write_bytes(b"fake")
        found = _new_files(str(self.tmp_path), before=before, started=started)
        self.assertEqual(len(found), 1)
        self.assertEqual(os.path.basename(found[0]), "new.png")

    def test_normalise_preserves_unc_prefix(self):
        self.assertEqual(_normalise("//host/I/forge" + chr(92) + "output" + chr(92) + "a.png"), "//host/I/forge/output/a.png")
        self.assertEqual(_normalise("C:" + chr(92) + "forge" + chr(92) + "a.png"), "C:/forge/a.png")

    def test_trim_info_drops_noise(self):
        trimmed = _trim_info({"prompt": "x", "seed": 1, "irrelevant": "y" * 5000})
        self.assertEqual(trimmed, {"prompt": "x", "seed": 1})

    def test_usable_target_accepts_folder_not_created_yet(self):
        from sam_extra_mcp.forgeneo.generate import _usable_target

        self.assertIs(_usable_target(str(self.tmp_path)), True)
        self.assertIs(_usable_target(str(self.tmp_path / "img2img-images")), True)
        self.assertIs(_usable_target(str(self.tmp_path / "missing" / "deeper")), False)

    def test_encode_init_image_rejects_bad_input(self):
        from sam_extra_mcp.forgeneo.generate import encode_init_image

        data, error = encode_init_image(str(self.tmp_path / "nope.png"))
        self.assertTrue(data is None and "not found" in error)
        not_image = self.tmp_path / "script.py"
        not_image.write_text("print()")
        data, error = encode_init_image(str(not_image))
        self.assertTrue(data is None and "not a recognised image" in error)
        empty = self.tmp_path / "empty.png"
        empty.write_bytes(b"")
        data, error = encode_init_image(str(empty))
        self.assertTrue(data is None and "empty" in error)

    def test_encode_init_image_returns_base64(self):
        from sam_extra_mcp.forgeneo.generate import encode_init_image

        path = self.tmp_path / "image.png"
        path.write_bytes(b"binary-content")
        data, error = encode_init_image(str(path))
        self.assertIsNone(error)
        self.assertEqual(base64.b64decode(data), b"binary-content")


# -- tests/test_history.py -------------------------------------------------------------------------

class HistoryTests(unittest.TestCase):
    def test_normalise_strips_folder_and_extension(self):
        self.assertEqual(normalise_checkpoint(r"Anima\animeMix_v10.safetensors"), "animemix_v10")
        self.assertEqual(normalise_checkpoint("animeMix_v10"), "animemix_v10")

    def test_normalise_handles_forward_slashes_and_hash_suffix(self):
        self.assertEqual(normalise_checkpoint("sub/dir/model.ckpt"), "model")
        self.assertEqual(normalise_checkpoint("model_v1 [abc123]"), "model_v1")

    def test_normalise_is_safe_on_empty_input(self):
        self.assertEqual(normalise_checkpoint(""), "")

    def test_summarise_uses_median_and_display_name(self):
        entries = [
            parse_infotext(f"a prompt\nSteps: {steps}, CFG scale: 1.5, Sampler: ER SDE, Model: animeMix_v10Turbo")
            for steps in (10, 11, 11, 12)
        ]
        regime = _summarise("animemix_v10turbo", (), entries)
        self.assertEqual(regime.checkpoint, "animeMix_v10Turbo")
        self.assertEqual(regime.steps, 11.0)
        self.assertEqual(regime.cfg, 1.5)
        self.assertEqual(regime.sampler, "ER SDE")
        self.assertEqual(regime.samples, 4)

    def test_danbooru_detection(self):
        self.assertIs(_looks_danbooru("score_9, score_8_up, 1girl"), True)
        self.assertIs(_looks_danbooru("masterpiece, best quality, solo"), True)
        self.assertIs(_looks_danbooru("long_hair, blue_eyes, standing, smile"), True)
        self.assertIs(_looks_danbooru("a photograph of a mountain at sunrise"), False)


# -- tests/test_infotext.py ------------------------------------------------------------------------

INFOTEXT_SAMPLE = (
    "1girl, solo, <lora:turbo-accel-v2:1> <lora:my_style:0.75> masterpiece\n"
    "Negative prompt: worst quality, blurry\n"
    "Steps: 11, Sampler: ER SDE, Schedule type: Beta, CFG scale: 1.5, Seed: 42, "
    'Size: 1024x1024, Model hash: abc123, Model: animeMix_v10Turbo, Version: f2.0'
)


class InfotextTests(unittest.TestCase):
    def test_splits_prompt_and_negative(self):
        info = parse_infotext(INFOTEXT_SAMPLE)
        self.assertTrue(info.prompt.startswith("1girl, solo"))
        self.assertEqual(info.negative, "worst quality, blurry")

    def test_reads_sampling_parameters(self):
        info = parse_infotext(INFOTEXT_SAMPLE)
        self.assertEqual(info.steps, 11.0)
        self.assertEqual(info.cfg, 1.5)
        self.assertEqual(info.sampler, "ER SDE")
        self.assertEqual(info.scheduler, "Beta")
        self.assertEqual(info.checkpoint, "animeMix_v10Turbo")

    def test_extracts_loras_with_weights(self):
        self.assertEqual(parse_infotext(INFOTEXT_SAMPLE).loras, (("turbo-accel-v2", 1.0), ("my_style", 0.75)))

    def test_handles_prompt_without_negative(self):
        info = parse_infotext("a mountain at sunrise\nSteps: 20, CFG scale: 7, Model: foo")
        self.assertEqual(info.prompt, "a mountain at sunrise")
        self.assertEqual(info.negative, "")
        self.assertEqual(info.checkpoint, "foo")

    def test_empty_text_is_safe(self):
        info = parse_infotext("")
        self.assertEqual(info.prompt, "")
        self.assertEqual(info.params, {})
        self.assertEqual(info.loras, ())

    def test_prompt_only_text_has_no_params(self):
        info = parse_infotext("just a prompt with no metadata")
        self.assertEqual(info.prompt, "just a prompt with no metadata")
        self.assertEqual(info.params, {})


# -- tests/test_instance_defaults.py ---------------------------------------------------------------

INSTANCE_OPTIONS = {
    "anima_t2i_step": 10.0,
    "anima_t2i_cfg": 1.0,
    "anima_t2i_dcfg": 3.0,
    "anima_t2i_sampler": "ER SDE",
    "anima_t2i_scheduler": "Beta",
    "anima_t2i_width": 832,
    "anima_t2i_height": 1216,
    "anima_i2i_width": 0,
    "anima_i2i_sampler": "ER SDE",
}


class InstanceDefaultsTests(unittest.TestCase):
    def test_reads_per_architecture_defaults(self):
        live = instance_defaults(INSTANCE_OPTIONS, "anima")
        self.assertEqual(live["step"], 10.0)
        self.assertEqual(live["dcfg"], 3.0)
        self.assertEqual(live["sampler"], "ER SDE")
        self.assertEqual(live["width"], 832)
        self.assertEqual(live["height"], 1216)

    def test_zero_dimensions_mean_auto_not_zero(self):
        live = instance_defaults(INSTANCE_OPTIONS, "anima", mode="i2i")
        self.assertNotIn("width", live)
        self.assertEqual(live["sampler"], "ER SDE")

    def test_unknown_preset_yields_nothing(self):
        self.assertEqual(instance_defaults(INSTANCE_OPTIONS, "flux"), {})
        self.assertEqual(instance_defaults(INSTANCE_OPTIONS, None), {})

    def test_payload_sends_distilled_cfg_when_given(self):
        self.assertEqual(build_payload("prompt", distilled_cfg_scale=3.0)["distilled_cfg_scale"], 3.0)

    def test_payload_omits_distilled_cfg_when_unset(self):
        self.assertNotIn("distilled_cfg_scale", build_payload("prompt"))


# -- tests/test_metadata_sources.py ----------------------------------------------------------------

METADATA_PARAMS = "a prompt\nSteps: 10, CFG scale: 1.5, Sampler: ER SDE, Model: someModel"


def _upstream_png_bytes(text: str | None) -> bytes:
    out = bytearray(b"\x89PNG\r\n\x1a\n")

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    out += chunk(b"IHDR", b"\x00" * 13)
    if text is not None:
        out += chunk(b"tEXt", b"parameters\x00" + text.encode("utf-8"))
    out += chunk(b"IEND", b"")
    return bytes(out)


class MetadataSourcesTests(_TempDirCase):
    """enable_pnginfo and save_txt are independent Forge settings; history must survive any
    combination of them."""

    def test_reads_embedded_chunk_when_present(self):
        path = self.tmp_path / "image.png"
        path.write_bytes(_upstream_png_bytes(METADATA_PARAMS))
        text, source = read_generation_metadata(str(path))
        self.assertEqual(source, "png_chunk")
        self.assertIn("Steps: 10", text)

    def test_falls_back_to_txt_when_pnginfo_disabled(self):
        path = self.tmp_path / "image.png"
        path.write_bytes(_upstream_png_bytes(None))
        (self.tmp_path / "image.txt").write_text(METADATA_PARAMS, encoding="utf-8")
        text, source = read_generation_metadata(str(path))
        self.assertEqual(source, "txt_sidecar")
        self.assertIn("Steps: 10", text)

    def test_prefers_embedded_chunk_over_sidecar(self):
        path = self.tmp_path / "image.png"
        path.write_bytes(_upstream_png_bytes(METADATA_PARAMS))
        (self.tmp_path / "image.txt").write_text("Steps: 99", encoding="utf-8")
        text, source = read_generation_metadata(str(path))
        self.assertEqual(source, "png_chunk")
        self.assertIn("Steps: 10", text)

    def test_non_png_formats_use_the_sidecar(self):
        path = self.tmp_path / "image.jpg"
        path.write_bytes(b"not really a jpeg")
        (self.tmp_path / "image.txt").write_text(METADATA_PARAMS, encoding="utf-8")
        text, source = read_generation_metadata(str(path))
        self.assertEqual(source, "txt_sidecar")
        self.assertIn("Sampler: ER SDE", text)

    def test_reports_none_when_both_carriers_are_missing(self):
        path = self.tmp_path / "image.png"
        path.write_bytes(_upstream_png_bytes(None))
        text, source = read_generation_metadata(str(path))
        self.assertIsNone(text)
        self.assertEqual(source, "none")

    def test_empty_sidecar_counts_as_missing(self):
        path = self.tmp_path / "image.png"
        path.write_bytes(_upstream_png_bytes(None))
        (self.tmp_path / "image.txt").write_text("   ", encoding="utf-8")
        text, source = read_generation_metadata(str(path))
        self.assertIsNone(text)
        self.assertEqual(source, "none")


# -- tests/test_module_health.py -------------------------------------------------------------------

HEALTH_MODULES = [
    {"model_name": "qwen_image_vae.safetensors", "filename": "/models/VAE/qwen_image_vae.safetensors"},
    {"model_name": "qwen_3_06b_base.safetensors", "filename": "/models/text_encoder/qwen_3_06b_base.safetensors"},
    {"model_name": "qwen3vl_4b_fp8_scaled.safetensors", "filename": "/models/text_encoder/qwen3vl_4b_fp8_scaled.safetensors"},
]


class _ModulesClient:
    def __init__(self, ok=True):
        self._ok = ok

    def modules(self):
        return ApiResult(self._ok, data=HEALTH_MODULES, error=None if self._ok else "unreachable")


class ModuleHealthTests(unittest.TestCase):
    def test_healthy_preset_reports_no_problems(self):
        options = {"forge_additional_modules_anima": [
            "/models/VAE/qwen_image_vae.safetensors", "/models/text_encoder/qwen_3_06b_base.safetensors"]}
        health = _module_health(_ModulesClient(), options, "anima")
        self.assertIs(health["healthy"], True)
        self.assertEqual(health["problems"], [])
        self.assertEqual(len(health["loaded"]), 2)

    def test_drifted_preset_names_the_installed_file_that_fits(self):
        options = {"forge_additional_modules_anima": [
            "/models/VAE/qwen_image_vae.safetensors", "/models/text_encoder/qwen3vl_4b_fp8_scaled.safetensors"]}
        health = _module_health(_ModulesClient(), options, "anima")
        self.assertIs(health["healthy"], False)
        self.assertTrue(any("qwen_3_06b_base.safetensors is installed and fits" in p for p in health["problems"]))
        self.assertTrue(any("does not ask for" in p for p in health["problems"]))

    def test_missing_with_nothing_installed_says_so(self):
        health = _module_health(_ModulesClient(), {"forge_additional_modules_flux": []}, "flux")
        self.assertTrue(any("none is installed" in problem for problem in health["problems"]))

    def test_unknown_architecture_is_skipped_quietly(self):
        self.assertIs(_module_health(_ModulesClient(), {}, "nonexistent")["checked"], False)

    def test_unreachable_listing_does_not_fabricate_a_verdict(self):
        health = _module_health(_ModulesClient(ok=False), {}, "anima")
        self.assertIs(health["checked"], False)
        self.assertNotIn("healthy", health)

    def test_no_preset_means_no_check(self):
        self.assertIs(_module_health(_ModulesClient(), {}, None)["checked"], False)


# -- tests/test_modules.py -------------------------------------------------------------------------

AVAILABLE = [
    {"model_name": "ae.safetensors", "filename": "/models/VAE/ae.safetensors"},
    {"model_name": "qwen_image_vae.safetensors", "filename": "/models/VAE/qwen_image_vae.safetensors"},
    {"model_name": "custom_vae_v10.safetensors", "filename": "/models/VAE/Sub/custom_vae_v10.safetensors"},
    {"model_name": "hdrVAE_fp32.safetensors", "filename": "/models/VAE/Anime/hdrVAE_fp32.safetensors"},
    {"model_name": "qwen_3_06b_base.safetensors", "filename": "/models/text_encoder/qwen_3_06b_base.safetensors"},
    {"model_name": "qwen3vl_4b_fp8_scaled.safetensors", "filename": "/models/text_encoder/qwen3vl_4b_fp8_scaled.safetensors"},
]


class ModulesTests(unittest.TestCase):
    def test_classifies_by_folder_not_by_name(self):
        self.assertEqual(classify("/models/VAE/anything.safetensors"), "vae")
        self.assertEqual(classify("/models/text_encoder/anything.safetensors"), "text_encoder")
        self.assertIsNone(classify("/models/Stable-diffusion/model.safetensors"))

    def test_flux_vae_pattern_is_anchored_but_accepts_rebuilds(self):
        flux_vae = ARCH_MODULES["flux"].requirements[0]
        self.assertIs(flux_vae.matches("ae.safetensors"), True)
        self.assertIs(flux_vae.matches("ultrafluxVAEImproved_v10.safetensors"), True)
        self.assertIs(flux_vae.matches("qwen_image_vae.safetensors"), False)
        self.assertIs(flux_vae.matches("custom_vae_v10.safetensors"), False)

    def test_klein_covers_both_sizes(self):
        vae, encoder = requirements_for("klein").requirements
        self.assertTrue(vae.matches("ae.safetensors") and vae.matches("flux2-vae.safetensors"))
        self.assertTrue(encoder.matches("qwen_3_4b_bf16.safetensors"))
        self.assertTrue(encoder.matches("qwen_3_8b_fp8mixed.safetensors"))

    def test_every_forge_preset_has_requirements(self):
        from sam_extra_mcp.forgeneo.presets import ARCH_DEFAULTS

        self.assertLessEqual(set(ARCH_DEFAULTS), set(ARCH_MODULES))

    def test_unclassifiable_module_is_not_accused(self):
        self.assertNotIn("unrecognised", audit("anima", ["some_unlisted_file.safetensors"], AVAILABLE))

    def test_detects_a_foreign_text_encoder_left_behind(self):
        report = audit("anima", ["qwen_image_vae.safetensors", "qwen3vl_4b_fp8_scaled.safetensors"], AVAILABLE)
        self.assertEqual(report["unrecognised"]["modules"], ["qwen3vl_4b_fp8_scaled.safetensors"])
        self.assertIn("Qwen3 0.6B", [item["need"] for item in report["missing"]])
        self.assertEqual([item["loaded"] for item in report["satisfied"]], ["qwen_image_vae.safetensors"])

    def test_unrecognised_is_phrased_as_a_question_not_a_verdict(self):
        report = audit("anima", ["hdrVAE_fp32.safetensors"], AVAILABLE)
        self.assertIn("check", report["unrecognised"])
        self.assertIn("community build", report["unrecognised"]["why"])

    def test_missing_module_points_at_an_installed_candidate(self):
        report = audit("anima", [], AVAILABLE)
        encoder = next(item for item in report["missing"] if item["kind"] == "text_encoder")
        self.assertEqual(encoder["candidates_installed"], ["qwen_3_06b_base.safetensors"])
        self.assertEqual(encoder["action"], "select one")

    def test_missing_module_with_nothing_installed_says_download(self):
        report = audit("flux", [], AVAILABLE)
        clip = next(item for item in report["missing"] if item["need"] == "CLIP-L")
        self.assertEqual(clip["candidates_installed"], [])
        self.assertEqual(clip["action"], "download one")

    def test_unspecified_vae_is_reported_not_guessed(self):
        report = audit("pid", [], AVAILABLE)
        self.assertIn("does not name which one", report["vae"]["status"])
        self.assertTrue(report["vae"]["installed_vaes"])

    def test_krea_vae_comes_from_the_merged_wiki_cell(self):
        report = audit("krea", ["qwen_image_vae.safetensors", "qwen3vl_4b_fp8_scaled.safetensors"], AVAILABLE)
        self.assertEqual(report["missing"], [])
        self.assertNotIn("vae", report)
        self.assertEqual(len(report["satisfied"]), 2)

    def test_sd_and_sdxl_follow_the_reference(self):
        for arch, pattern in (("sd", "vae-ft-mse"), ("xl", "sdxl-vae")):
            spec = requirements_for(arch)
            self.assertEqual(len(spec.requirements), 1)
            requirement = spec.requirements[0]
            self.assertEqual(requirement.kind, "vae")
            self.assertIs(requirement.optional, False)
            self.assertTrue(requirement.matches(f"{pattern}-something.safetensors"))

    def test_unknown_architecture_says_so(self):
        self.assertIs(audit("nonexistent", [], AVAILABLE)["known"], False)

    def test_every_architecture_accounts_for_its_vae(self):
        from sam_extra_mcp.forgeneo.modules import VAE

        for arch, spec in ARCH_MODULES.items():
            requires_vae = any(requirement.kind == VAE for requirement in spec.requirements)
            self.assertTrue(requires_vae or spec.vae_unspecified, f"{arch} neither requires a VAE nor explains it")

    def test_notes_do_not_add_claims_the_reference_lacks(self):
        for arch in ("sd", "xl"):
            note = ARCH_MODULES[arch].note.lower()
            for invented in ("optional", "upgrade", "built-in", "baked"):
                self.assertNotIn(invented, note)

    def test_unspecified_vae_says_required_not_absent(self):
        self.assertTrue(audit("pid", [], AVAILABLE)["vae"]["status"].startswith("required"))

    def test_xl_note_covers_every_lineage(self):
        note = ARCH_MODULES["xl"].note.lower()
        for lineage in ("pony", "illustrious", "animagine"):
            self.assertIn(lineage, note)


# -- tests/test_no_local_statistics.py -------------------------------------------------------------

PACKAGE = MCP_PROJECT / "sam_extra_mcp"
# Upstream's keyword fields, plus the ones sam-extra's modules use for runtime text.
RUNTIME_FIELDS = frozenset(
    {"notes", "avoid", "structure", "tag_style", "artist_syntax", "weighting", "detail", "label", "note", "action"}
)
STATISTIC_PATTERNS = (
    re.compile(r"\d+\s*%"),
    re.compile(r"\b\d+\s+of\s+\d+\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*\+?\s*-?\s*(checkpoint|lora|model|generation|image|file)s?\b", re.IGNORECASE),
    re.compile(r"\b(most|majority|typically|usually)\s+\d+", re.IGNORECASE),
)
ALLOWED = ("year 2023", "year 2025", "score_1", "score_9", "two sentences", "8-12 steps", "CFG 1")


def _runtime_strings() -> list[tuple[str, int, str, str]]:
    found: list[tuple[str, int, str, str]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.keyword) or node.arg not in RUNTIME_FIELDS:
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if isinstance(value, str):
                found.append((path.name, node.value.lineno, node.arg, value))
    return found


class NoLocalStatisticsTests(unittest.TestCase):
    """Guard against shipping one environment's measurements as universal advice."""

    def test_runtime_text_carries_no_environment_statistics(self):
        offenders = []
        for filename, lineno, field_name, text in _runtime_strings():
            cleaned = text
            for allowed in ALLOWED:
                cleaned = cleaned.replace(allowed, "")
            for pattern in STATISTIC_PATTERNS:
                match = pattern.search(cleaned)
                if match:
                    offenders.append(f"{filename}:{lineno} [{field_name}] -> {match.group(0)!r} in {text[:90]!r}")
                    break
        self.assertFalse(offenders, "\n  ".join(offenders))

    def test_the_guard_actually_catches_the_original_mistake(self):
        regression = (
            "true photorealism on the stock model. Community merges are a different matter: on one "
            "207-checkpoint Anima collection, 23% declared realism tags."
        )
        self.assertTrue(any(pattern.search(regression) for pattern in STATISTIC_PATTERNS))

    def test_qualitative_replacement_passes(self):
        replacement = (
            "merges are a different matter: many are tuned towards semi-realism, so judge the loaded "
            "checkpoint by its own tags rather than by the family"
        )
        self.assertFalse(any(pattern.search(replacement) for pattern in STATISTIC_PATTERNS))

    def test_audit_actually_reads_something(self):
        self.assertGreaterEqual(len(_runtime_strings()), 10)

    def test_readme_guard_catches_the_lora_count(self):
        # Upstream's README checks themselves are not ported (its README is not vendored);
        # this one only proves the "300+ LoRAs" shape is caught.
        self.assertTrue(any(pattern.search("so 300+ LoRAs cost nothing") for pattern in STATISTIC_PATTERNS))


# -- tests/test_preset_priority.py -----------------------------------------------------------------

def _priority_entry(steps, cfg, sampler="ER SDE", lora=None):
    prompt = "a prompt" if not lora else f"a prompt <lora:{lora}:1>"
    return parse_infotext(
        f"{prompt}\nSteps: {steps}, CFG scale: {cfg}, Sampler: {sampler}, Schedule type: Beta, Model: someModel"
    )


def _priority_index(entries_by_accel):
    index = HistoryIndex(None)
    index._built = True  # noqa: SLF001 - fixture bypasses the filesystem scan
    key = normalise_checkpoint("someModel")
    for accelerators, entries in entries_by_accel.items():
        regime = _summarise(key, accelerators, entries)
        index._regimes[(key, accelerators)] = regime  # noqa: SLF001
        index._by_checkpoint[key].append(regime)  # noqa: SLF001
        for entry in entries:
            for name, weight in entry.loras:
                index._lora_usage[name] += 1  # noqa: SLF001
                index._lora_weights[name].append(weight)  # noqa: SLF001
    return index


class PresetPriorityTests(unittest.TestCase):
    def test_habit_is_unknown_without_history(self):
        habit = HistoryIndex(None).accelerator_habit()
        self.assertIs(habit["known"], False)
        self.assertIsNone(habit["rate"])

    def test_habit_reports_a_dominant_accelerator(self):
        index = _priority_index({
            ("turbo-accel-v2",): [_priority_entry(10, 1.5, lora="turbo-accel-v2")] * 9,
            (): [_priority_entry(32, 4.0)],
        })
        habit = index.accelerator_habit()
        self.assertIs(habit["known"], True)
        self.assertEqual(habit["rate"], 0.9)
        self.assertEqual(habit["common"][0]["name"], "turbo-accel-v2")
        self.assertEqual(habit["common"][0]["typical_weight"], 1.0)

    def test_habit_reports_absence_of_accelerators(self):
        habit = _priority_index({(): [_priority_entry(32, 4.0)] * 10}).accelerator_habit()
        self.assertEqual(habit["rate"], 0.0)
        self.assertEqual(habit["common"], [])

    def test_habit_counts_generations_not_regimes(self):
        habit = _priority_index({
            ("acc",): [_priority_entry(10, 1.5, lora="acc")] * 3,
            (): [_priority_entry(32, 4.0)] * 7,
        }).accelerator_habit()
        self.assertEqual(habit["generations"], 10)
        self.assertEqual(habit["rate"], 0.3)


# -- tests/test_regime_selection.py ----------------------------------------------------------------

def _regime_entry(steps, cfg, lora=None):
    prompt = "a prompt" if not lora else f"a prompt <lora:{lora}:1>"
    return parse_infotext(f"{prompt}\nSteps: {steps}, CFG scale: {cfg}, Sampler: ER SDE, Model: animeMix_v10")


def _index_with_regimes():
    index = HistoryIndex(None)
    index._built = True  # noqa: SLF001 - test fixture, bypassing the filesystem scan
    key = normalise_checkpoint("animeMix_v10")
    with_turbo = ("turbo-accel-v2",)
    regimes = {
        (key, with_turbo): _summarise(key, with_turbo, [_regime_entry(10, 1.5, "turbo-accel-v2")] * 7),
        (key, ()): _summarise(key, (), [_regime_entry(30, 4.0)]),
    }
    index._regimes = regimes  # noqa: SLF001
    for (_, _), regime in regimes.items():
        index._by_checkpoint[key].append(regime)  # noqa: SLF001
    return index


class RegimeSelectionTests(unittest.TestCase):
    """Regime selection must not let a stray outlier outvote the representative sample."""

    def test_unspecified_accelerators_picks_the_most_observed_regime(self):
        regime = _index_with_regimes().regime_for("animeMix_v10")
        self.assertEqual(regime.samples, 7)
        self.assertEqual(regime.accelerators, ("turbo-accel-v2",))
        self.assertEqual(regime.steps, 10.0)

    def test_explicit_empty_tuple_asks_for_no_accelerator(self):
        regime = _index_with_regimes().regime_for("animeMix_v10", accelerators=())
        self.assertEqual(regime.accelerators, ())
        self.assertEqual(regime.steps, 30.0)

    def test_explicit_combination_matches_exactly(self):
        self.assertEqual(_index_with_regimes().regime_for("animeMix_v10", accelerators=("turbo-accel-v2",)).samples, 7)

    def test_unknown_combination_falls_back_to_most_observed(self):
        self.assertEqual(_index_with_regimes().regime_for("animeMix_v10", accelerators=("never-seen-lora",)).samples, 7)

    def test_regimes_for_lists_all_sorted_by_samples(self):
        self.assertEqual([r.samples for r in _index_with_regimes().regimes_for("animeMix_v10")], [7, 1])

    def test_unknown_checkpoint_returns_nothing(self):
        self.assertIsNone(_index_with_regimes().regime_for("neverSeen"))


# -- tests/test_shift_detection.py -----------------------------------------------------------------

def _shift_entry(extra=""):
    return parse_infotext(f"a prompt\nSteps: 10, Sampler: ER SDE, CFG scale: 1.5, Model: someModel{extra}")


class ShiftDetectionTests(unittest.TestCase):
    def test_detects_architecture_that_uses_shift(self):
        regime = _summarise("somemodel", (), [_shift_entry(", Shift: 3.0") for _ in range(6)])
        self.assertIs(regime.uses_shift, True)
        self.assertEqual(regime.shift, 3.0)

    def test_detects_architecture_that_ignores_shift(self):
        regime = _summarise("somemodel", (), [_shift_entry() for _ in range(6)])
        self.assertIs(regime.uses_shift, False)
        self.assertIsNone(regime.shift)

    def test_distilled_cfg_scale_counts_as_shift(self):
        regime = _summarise("somemodel", (), [_shift_entry(", Distilled CFG Scale: 3.5") for _ in range(4)])
        self.assertIs(regime.uses_shift, True)
        self.assertEqual(regime.shift, 3.5)

    def test_shift_is_the_median_of_observations(self):
        entries = [_shift_entry(", Shift: 3.0"), _shift_entry(", Shift: 3.5"), _shift_entry(", Shift: 3.5")]
        self.assertEqual(_summarise("somemodel", (), entries).shift, 3.5)


# -- tests/test_turbo_assessment.py ----------------------------------------------------------------

ANIMA = defaults_for("anima")
KLEIN = defaults_for("klein")
ZIT = defaults_for("zit")


class TurboAssessmentTests(unittest.TestCase):
    """Turbo assessment must behave sanely on a clean install."""

    def test_clean_install_unknown_checkpoint_is_unknown_not_no(self):
        result = assess_turbo("someModel_v1", (), ANIMA, observed_steps=32.0, samples=0)
        self.assertEqual(result.state, "unknown")
        self.assertEqual(result.confidence, "low")
        self.assertIs(result.architecture_is_distilled, False)

    def test_distilled_architecture_is_not_reported_as_checkpoint_turbo(self):
        for arch in (KLEIN, ZIT):
            result = assess_turbo("plainName", (), arch, observed_steps=float(arch.steps), samples=0)
            self.assertIs(result.architecture_is_distilled, True)
            self.assertEqual(result.state, "unknown")

    def test_name_hint_gives_medium_confidence_without_history(self):
        result = assess_turbo("animeMix_v10Turbo", (), ANIMA, observed_steps=None, samples=0)
        self.assertEqual(result.state, "yes")
        self.assertEqual(result.confidence, "medium")

    def test_accelerator_in_use_is_high_confidence(self):
        result = assess_turbo("plainName", ("turbo-accel-lora-v1",), ANIMA, 32.0, 0)
        self.assertEqual((result.state, result.confidence), ("yes", "high"))

    def test_history_far_below_preset_is_high_confidence(self):
        result = assess_turbo("plainName", (), ANIMA, observed_steps=11.0, samples=38)
        self.assertEqual((result.state, result.confidence), ("yes", "high"))

    def test_history_in_line_with_preset_is_a_confident_no(self):
        result = assess_turbo("plainName", (), ANIMA, observed_steps=30.0, samples=20)
        self.assertEqual((result.state, result.confidence), ("no", "medium"))

    def test_single_observation_is_not_enough_to_decide(self):
        self.assertEqual(assess_turbo("plainName", (), ANIMA, observed_steps=10.0, samples=1).state, "unknown")

    def test_no_preset_match_still_answers(self):
        result = assess_turbo("plainName", (), None, observed_steps=None, samples=0)
        self.assertEqual(result.state, "unknown")
        self.assertIs(result.architecture_is_distilled, False)


if __name__ == "__main__":
    unittest.main()
