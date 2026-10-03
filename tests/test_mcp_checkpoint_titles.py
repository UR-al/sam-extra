"""sam-extra MCP server: Forge's checkpoint titles end in " [shorthash]", after the extension.

Once Forge knows a checkpoint's hash, its title - the value of ``sd_model_checkpoint`` and of the
``/sdapi/v1/sd-models`` "title", and what ``models`` lists - is ``name.safetensors [0123456789]``,
while the infotext of a generation records ``Model: name``. Upstream stripped the extension before
the bracketed hash, so a hashed title never matched the history: ``model_profile`` reported no past
generations for a checkpoint with hundreds of them and fell back to preset defaults (seen on a live
Forge 2.29.2). Forge's per-preset record (``forge_checkpoint_<preset>``) keeps whichever form its
dropdown held, so a checkpoint switch by title lost the architecture signal the same way.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _mcp_support import DEFAULT_OPTIONS, HAS_HTTPX, infotext, make_service, png_bytes

from sam_extra_mcp.forgeneo.history import normalise_checkpoint
from sam_extra_mcp.forgeneo.profile import preset_for_checkpoint


class NormaliseTitleTests(unittest.TestCase):
    def test_a_hashed_title_matches_the_infotext_name(self):
        self.assertEqual(normalise_checkpoint("Anima-3.8B-v1.1.safetensors [4a458d26b2]"), "anima-3.8b-v1.1")
        self.assertEqual(normalise_checkpoint("Anima-3.8B-v1.1"), "anima-3.8b-v1.1")
        self.assertEqual(normalise_checkpoint(r"Anima\animeMix_v10.safetensors [0123456789]"), "animemix_v10")
        self.assertEqual(normalise_checkpoint("Krea/museMix_v35.gguf [abcdef0123]"), "musemix_v35")
        self.assertEqual(normalise_checkpoint("sub/model.ckpt [abc123]"), "model")

    def test_upstream_forms_are_unchanged(self):
        self.assertEqual(normalise_checkpoint(r"Anima\animeMix_v10.safetensors"), "animemix_v10")
        self.assertEqual(normalise_checkpoint("model_v1 [abc123]"), "model_v1")
        self.assertEqual(normalise_checkpoint(""), "")

    def test_a_bracket_in_the_name_reads_the_same_from_both_sources(self):
        infotext_name = normalise_checkpoint("model [v2]")
        self.assertEqual(normalise_checkpoint("model [v2].safetensors [0123456789]"), infotext_name)
        self.assertEqual(normalise_checkpoint("model [v2].safetensors"), infotext_name)


class PresetRegistryTitleTests(unittest.TestCase):
    OPTIONS = {
        "forge_checkpoint_anima": "Anima-3.8B-v1.1.safetensors",
        "forge_checkpoint_krea": "krea2Anime_v15.safetensors [0123456789]",
    }

    def test_hash_on_one_side_only_still_matches(self):
        self.assertEqual(preset_for_checkpoint(self.OPTIONS, "Anima-3.8B-v1.1.safetensors [4a458d26b2]"), "anima")
        self.assertEqual(preset_for_checkpoint(self.OPTIONS, "krea2Anime_v15.safetensors"), "krea")
        self.assertEqual(preset_for_checkpoint(self.OPTIONS, "krea2Anime_v15.safetensors [0123456789]"), "krea")

    def test_a_different_checkpoint_is_still_no_match(self):
        self.assertIsNone(preset_for_checkpoint(self.OPTIONS, "Anima-3.8B-v1.0.safetensors [4a458d26b2]"))
        self.assertIsNone(preset_for_checkpoint(self.OPTIONS, "krea2Anime_v15.gguf"))


@unittest.skipUnless(HAS_HTTPX, "httpx is required for the fake Forge transport")
class ServiceTitleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_model_profile_uses_the_history_of_a_hashed_title(self):
        options = dict(DEFAULT_OPTIONS, sd_model_checkpoint="Anima/animeMix_v10.safetensors [abc123]")
        service, _fake, tree = make_service(self.base, options=options)
        self.addCleanup(service.close)
        folder = tree["forge"] / "output" / "txt2img-images" / "2026-10-03"
        folder.mkdir(parents=True)
        for seed in (1, 2, 3):
            text = infotext("1girl, solo", seed, steps=11, cfg=1.5, model="animeMix_v10")
            (folder / f"0000{seed}-{seed}.png").write_bytes(png_bytes(text))

        profile = service.model_profile()
        self.assertIs(profile["ok"], True)
        self.assertEqual(profile["samples_observed"], 3)
        self.assertEqual(profile["recommended"]["steps"], 11)
        self.assertEqual(profile["recommended"]["cfg"], 1.5)
        self.assertIn("history (3 generations)", profile["parameter_source"])

    def test_loading_by_title_keeps_the_registry_signal(self):
        krea_modules = ["/models/VAE/krea_vae.safetensors"]
        options = dict(
            DEFAULT_OPTIONS,
            forge_checkpoint_krea="kreaRoot_v1.safetensors",
            forge_additional_modules_krea=krea_modules,
        )
        service, fake, _tree = make_service(
            self.base, options=options, settings={"sam3_mcp_allow_model_switch": True}
        )
        self.addCleanup(service.close)

        result = service.models(action="load", name="kreaRoot_v1.safetensors [0123456789]")
        self.assertIs(result["ok"], True, result)
        self.assertEqual(result["signals"]["preset_registry"], "krea")
        self.assertEqual(result["preset"], "krea")
        self.assertEqual(fake.options["forge_preset"], "krea")
        self.assertEqual(fake.options["forge_additional_modules"], krea_modules)


if __name__ == "__main__":
    unittest.main()
