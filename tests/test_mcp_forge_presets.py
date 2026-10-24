"""sam-extra MCP server: the vendored architecture tables follow Forge's own preset list.

- ``presets.ARCH_DEFAULTS`` holds Forge's sampler / scheduler / steps / CFG / shift / frames for every
  preset ``modules_forge/presets.py`` defines (checked against the Forge checkout - 2.30.1 has
  ``qwen21``, 2.31 adds ``moss``), and each of them has a module entry and a prompt dialect.
- MOSS-Image (Forge 2.31): natural-language dialect, the JSON layout prompt, Qwen3.5 9B + Flux.2 VAE.
  Qwen-Image-2.1's prompt-rewriting models (also Qwen3.5 9B) do not pass as the MOSS encoder.
- Qwen-Image-2.1: Qwen3-VL 8B and its own VAE, as the Forge wiki lists them.

The Forge checkout is next to the extension, else ``$SAM3_FORGE_ROOT`` (tests/_forge_checkout); without
one (CI) the Forge comparisons skip. Nothing here touches the network or a GPU.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _mcp_support import HAS_HTTPX  # noqa: E402,F401  (puts mcp_server on sys.path)

from sam_extra_mcp.forgeneo import dialects, downloads  # noqa: E402
from sam_extra_mcp.forgeneo.modules import ARCH_MODULES, audit, requirements_for  # noqa: E402
from sam_extra_mcp.forgeneo.presets import ARCH_DEFAULTS, PROMPT_STYLE, defaults_for  # noqa: E402
from sam_extra_mcp.forgeneo.profile import preset_from_folder  # noqa: E402
from tests._forge_checkout import require_forge_file  # noqa: E402


def _forge_presets():
    path = require_forge_file("modules_forge/presets.py")
    if not path.is_file():
        raise AssertionError(f"Forge moved its preset table: {path}")
    spec = importlib.util.spec_from_file_location("_forge_presets_for_mcp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)   # top level imports only enum; register() (gradio) is not called
    return module


class ForgePresetTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forge = _forge_presets()
        cls.names = cls.forge.PresetArch.choices()

    def test_every_forge_preset_is_in_the_table_with_forges_values(self):
        forge = self.forge
        for arch in forge.PresetArch:
            with self.subTest(preset=arch.name):
                ours = ARCH_DEFAULTS.get(arch.name)
                self.assertIsNotNone(ours, f"Forge preset '{arch.name}' is missing from ARCH_DEFAULTS")
                self.assertEqual(ours.sampler, forge.SAMPLERS[arch])
                self.assertEqual(ours.scheduler, forge.SCHEDULERS[arch])
                self.assertEqual(ours.steps, forge.STEPS[arch])
                self.assertEqual(ours.cfg, forge.CFG[arch])
                self.assertEqual(ours.shift, forge.SHIFT.get(arch))
                self.assertEqual(ours.frames, forge.is_video(arch.name))

    def test_every_forge_preset_has_modules_and_a_dialect(self):
        for name in self.names:
            with self.subTest(preset=name):
                self.assertIsNotNone(requirements_for(name))
                if name in dialects.AMBIGUOUS_ARCHITECTURES:
                    self.assertIsNone(dialects.for_architecture(name))
                else:
                    self.assertIsNotNone(dialects.for_architecture(name))

    def test_the_table_has_no_preset_forge_lacks_beyond_moss(self):
        # moss is the one entry ahead of Forge 2.30.1; everything else must still exist in Forge.
        self.assertLessEqual(set(ARCH_DEFAULTS) - {"moss"}, set(self.names))


class ForgeMossLoaderTests(unittest.TestCase):
    """The module note and the download source rest on Forge reading Qwen3.5 9B only in the
    ``model.language_model.*`` layout (backend/loader.py replace_state_dict, Forge 2.31)."""

    def test_forge_reads_the_moss_encoder_only_in_the_model_language_model_layout(self):
        source = require_forge_file("backend/loader.py").read_text(encoding="utf-8")
        if "qwen35_9b" not in source:
            self.skipTest("Forge 2.30.x: no MOSS-Image support")
        self.assertIn('"model.language_model.layers.0.linear_attn.A_log" in asd', source)
        self.assertIn('"model.language_model.layers.0.input_layernorm.weight" in asd', source)
        self.assertIn('qwen35_9b.transformer.{k}"] = v', source)


class MossTests(unittest.TestCase):
    def test_defaults(self):
        moss = defaults_for("moss")
        self.assertEqual((moss.sampler, moss.scheduler, moss.steps, moss.cfg, moss.shift), ("Euler", "Simple", 50, 3.0, 5.0))
        self.assertFalse(moss.is_video)
        self.assertIs(defaults_for("MOSS"), moss)

    def test_prompt_dialect_is_natural_language(self):
        self.assertIs(dialects.for_architecture("moss"), dialects.NATURAL)
        self.assertIs(dialects.from_declared_base("MOSS-Image"), dialects.NATURAL)
        style = PROMPT_STYLE["moss"]
        for words in ("natural language", "JSON", "chat template", "not applied"):
            self.assertIn(words, style)

    def test_requirements(self):
        vae, encoder = requirements_for("moss").requirements
        self.assertTrue(vae.matches("flux2-vae.safetensors") and vae.matches("flux2-vae_small.safetensors"))
        self.assertFalse(vae.matches("ae.safetensors"))
        for name in ("qwen3.5_9b_bf16.safetensors", "qwen3.5_9b_int8_convrot.safetensors",
                     "qwen3.5_9b_moss_image_sft_bf16.safetensors"):
            self.assertTrue(encoder.matches(name), name)
        for name in ("qwen3.5_9b_qwen_image_2.1_pe_t2i_w4a8.safetensors",   # Qwen-Image-2.1 prompt rewriter
                     "qwen35_4b.safetensors", "qwen_3_8b.safetensors", "qwen3vl_8b_bf16.safetensors"):
            self.assertFalse(encoder.matches(name), name)
        self.assertIn("model.language_model", requirements_for("moss").note)

    def test_audit_offers_only_the_real_encoder(self):
        available = [
            {"model_name": "flux2-vae.safetensors", "filename": "/f/models/VAE/flux2-vae.safetensors"},
            {"model_name": "qwen3.5_9b_bf16.safetensors", "filename": "/f/models/text_encoder/qwen3.5_9b_bf16.safetensors"},
            {"model_name": "qwen3.5_9b_qwen_image_2.1_pe_t2i_w4a8.safetensors",
             "filename": "/f/models/text_encoder/qwen3.5_9b_qwen_image_2.1_pe_t2i_w4a8.safetensors"},
        ]
        report = audit("moss", ["flux2-vae.safetensors"], available)
        self.assertEqual([item["loaded"] for item in report["satisfied"]], ["flux2-vae.safetensors"])
        (missing,) = report["missing"]
        self.assertEqual(missing["candidates_installed"], ["qwen3.5_9b_bf16.safetensors"])
        full = audit("moss", ["flux2-vae.safetensors", "qwen3.5_9b_bf16.safetensors"], available)
        self.assertEqual(full["missing"], [])

    def test_downloads_come_from_the_comfy_org_repackage(self):
        entries = downloads.for_architecture("moss")
        self.assertTrue(entries)
        for entry in entries:
            self.assertIn("huggingface.co/Comfy-Org/MOSS-Image/blob/main/", entry.url)
            self.assertFalse(entry.is_directory)
        for need in ("Flux.2 VAE", "Qwen3.5 9B"):
            with self.subTest(need=need):
                self.assertTrue(downloads.matching("moss", need))
        self.assertEqual({e.kind for e in downloads.matching("moss", "Qwen3.5 9B")}, {"text_encoder"})

    def test_folder_alias(self):
        options = {"forge_checkpoint_moss": None, "forge_checkpoint_qwen21": None, "forge_checkpoint_xl": None}
        self.assertEqual(preset_from_folder(options, "MOSS-Image/moss_image_sft_bf16.safetensors"), "moss")
        self.assertEqual(preset_from_folder(options, "Qwen-Image-2.1/qwen_image_2.1_bf16.safetensors"), "qwen21")


class QwenImage21Tests(unittest.TestCase):
    def test_defaults(self):
        q = defaults_for("qwen21")
        self.assertEqual((q.sampler, q.scheduler, q.steps, q.cfg, q.shift), ("Euler", "Simple", 32, 2.5, None))
        self.assertIs(dialects.for_architecture("qwen21"), dialects.NATURAL)
        self.assertIn("chat template", PROMPT_STYLE["qwen21"])

    def test_requirements_and_downloads(self):
        vae, encoder = requirements_for("qwen21").requirements
        self.assertTrue(vae.matches("qwen_image_2.1_vae_bf16.safetensors"))
        self.assertFalse(vae.matches("qwen_image_vae.safetensors"))
        self.assertTrue(encoder.matches("qwen3vl_8b_int8_convrot.safetensors"))
        self.assertFalse(encoder.matches("qwen3vl_4b_fp8_scaled.safetensors"))
        # The older Qwen-Image preset must not take the 2.1 VAE either.
        self.assertFalse(requirements_for("qwen").requirements[0].matches("qwen_image_2.1_vae_bf16.safetensors"))
        for need in ("Qwen-Image-2.1 VAE", "Qwen3-VL 8B"):
            with self.subTest(need=need):
                hits = downloads.matching("qwen21", need)
                self.assertTrue(hits)
                for entry in hits:
                    self.assertIn("huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/", entry.url)
        self.assertIn("qwen21", ARCH_MODULES)


if __name__ == "__main__":
    unittest.main()
