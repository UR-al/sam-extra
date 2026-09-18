from __future__ import annotations

import unittest

from sam3ext.anima_reference_recipe import (
    FILL_ORIGINAL,
    FILL_SMEAR,
    KEEP_IDENTITY,
    KEEP_OUTFIT,
    KEEP_STYLE,
    RECIPE_OUTPUT_SIZE,
    WARN_LBW,
    WARN_NEGPIP,
    WARN_WILDCARD,
    DetectedLoras,
    SamplingSettings,
    choose_fallback_index,
    detect_reference_loras,
    edit_strength_for_scope,
    fill_settings,
    forge_preset_sampling,
    is_reference_output,
    lora_stem,
    prepare_inherited_prompt,
    prompt_syntax_warnings,
    resolve_output_size,
    strip_reference_tokens,
)


class LoraDetectionTests(unittest.TestCase):
    def test_stem_keeps_version_dots_and_drops_real_extensions(self):
        self.assertEqual(lora_stem("anima/AnimeEdit v1.2.safetensors"), "AnimeEdit v1.2")
        self.assertEqual(lora_stem(r"anima\Extend Image (Anima Edit) v1"), "Extend Image (Anima Edit) v1")
        self.assertEqual(lora_stem("style.v2.pt"), "style.v2")

    def test_prefers_exact_animeeditv2_and_finds_extend(self):
        found = detect_reference_loras(
            ["zeta", "anima/AnimeEditV1", "AnimeEditV2", "Extend Image (Anima Edit) v1"]
        )
        self.assertEqual(found, DetectedLoras(edit="AnimeEditV2", extend="Extend Image (Anima Edit) v1"))

    def test_extend_lora_is_never_taken_as_edit(self):
        found = detect_reference_loras(["Extend Image (Anima Edit) v1"])
        self.assertEqual(found, DetectedLoras(edit=None, extend="Extend Image (Anima Edit) v1"))

    def test_other_edit_names_are_accepted_when_v2_is_absent(self):
        found = detect_reference_loras(["anima_edit_v1.2", "style_a"])
        self.assertEqual(found.edit, "anima_edit_v1.2")
        self.assertIsNone(found.extend)


class PromptTests(unittest.TestCase):
    def test_reference_tokens_are_removed_from_inherited_prompt(self):
        cases = [
            (
                "<lora:AnimeEditV2:0.72>, (split screen, multiple views:1.2), 1girl, <lora:styleA:0.5>",
                "1girl, <lora:styleA:0.5>",
            ),
            ("1girl, <lora:anima/Extend Image (Anima Edit) v1:0.4>, smile", "1girl, smile"),
            ("(split screen, multiple views, split screen:1.2), cat", "cat"),
            ("(before and after:1.1), cat", "(before and after:1.1), cat"),
        ]
        for prompt, expected in cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(strip_reference_tokens(prompt), expected)

    def test_style_scope_also_drops_style_loras(self):
        prompt = "1girl, <lora:styleA:0.5>, <lyco:b:1>"
        self.assertEqual(prepare_inherited_prompt(prompt, KEEP_STYLE), "1girl")
        self.assertEqual(prepare_inherited_prompt(prompt, KEEP_OUTFIT), prompt)
        self.assertEqual(prepare_inherited_prompt(prompt, KEEP_IDENTITY), prompt)

    def test_syntax_that_needs_other_scripts_is_reported(self):
        self.assertEqual(prompt_syntax_warnings("1girl, __hair__"), [WARN_WILDCARD])
        self.assertEqual(prompt_syntax_warnings("{red|blue} hair"), [WARN_WILDCARD])
        self.assertEqual(prompt_syntax_warnings("(bad hands:-1.0)"), [WARN_NEGPIP])
        self.assertEqual(prompt_syntax_warnings("<lora:x:1:lbw=0,1>"), [WARN_LBW])
        self.assertEqual(prompt_syntax_warnings("1girl, smile"), [])


class ScopeSizeFillTests(unittest.TestCase):
    def test_outfit_and_style_raise_edit_strength_to_at_least_point_eight(self):
        self.assertEqual(edit_strength_for_scope(KEEP_IDENTITY, 0.72), 0.72)
        self.assertEqual(edit_strength_for_scope(KEEP_OUTFIT, 0.72), 0.8)
        self.assertEqual(edit_strength_for_scope(KEEP_STYLE, 0.72), 0.8)
        self.assertEqual(edit_strength_for_scope(KEEP_OUTFIT, 1.0), 1.0)

    def test_output_size_follows_txt2img_unless_custom(self):
        self.assertEqual(resolve_output_size(False, 960, 1088, 832, 1216), (832, 1216))
        self.assertEqual(resolve_output_size(True, 704, 896, 832, 1216), (704, 896))
        self.assertEqual(resolve_output_size(False, 960, 1088, None, None), RECIPE_OUTPUT_SIZE)
        self.assertEqual(resolve_output_size(False, 960, 1088, 0, 0), RECIPE_OUTPUT_SIZE)
        self.assertEqual(resolve_output_size(False, 960, 1088, "832", "1216.0"), (832, 1216))

    def test_fill_modes_map_to_forge_masked_content(self):
        self.assertEqual(fill_settings(FILL_ORIGINAL, "#000000"), ("original", "#000000"))
        self.assertEqual(fill_settings(FILL_SMEAR, "#ffffff"), ("latent noise", "#ffffff"))
        self.assertEqual(fill_settings("unknown", ""), ("original", "#000000"))


class SamplingAndFallbackTests(unittest.TestCase):
    def test_forge_anima_i2i_preset_is_read_from_options(self):
        opts = {
            "anima_i2i_sampler": "ER SDE",
            "anima_i2i_scheduler": "Beta57 (RES4LYF)",
            "anima_i2i_step": 50,
            "anima_i2i_cfg": 5,
            "anima_i2i_dcfg": 3,
        }
        self.assertEqual(
            forge_preset_sampling(opts, SamplingSettings()),
            SamplingSettings("ER SDE", "Beta57 (RES4LYF)", 50, 5.0, 3.0),
        )

    def test_missing_preset_values_fall_back(self):
        fallback = SamplingSettings("Euler a", "Simple", 30, 5.0, 3.0)
        self.assertEqual(forge_preset_sampling({"anima_i2i_step": ""}, fallback), fallback)

    def test_fallback_skips_previous_reference_outputs(self):
        marker = "Steps: 30, SAM3 Feature: 6 - Anima Character Reference"
        self.assertTrue(is_reference_output(marker))
        self.assertFalse(is_reference_output("Steps: 30"))
        self.assertEqual(choose_fallback_index(3, 1, ["a", "b", "c"]), 1)
        self.assertEqual(choose_fallback_index(3, -1, ["a", "b", marker]), 1)
        self.assertEqual(choose_fallback_index(3, 2, ["a", "b", marker]), 1)
        self.assertEqual(choose_fallback_index(2, "x", [marker, marker]), -1)
        self.assertEqual(choose_fallback_index(0, -1, []), -1)


if __name__ == "__main__":
    unittest.main()
