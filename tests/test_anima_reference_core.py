from __future__ import annotations

import unittest

from PIL import Image

from sam3ext.anima_reference_core import (
    ReferenceCanvasConfig,
    compose_reference_prompt,
    crop_reference_result,
    prepare_reference_canvas,
)


class AnimaReferenceCanvasTests(unittest.TestCase):
    def test_right_layout_creates_exact_target_mask_and_aligned_canvas(self):
        reference = Image.new("RGB", (640, 960), "red")
        prepared = prepare_reference_canvas(
            reference,
            ReferenceCanvasConfig(
                output_width=960,
                output_height=1088,
                composite_megapixels=1.4,
                dimension_multiple=16,
                placement="right",
                target_color="#123456",
            ),
        )

        width, height = prepared.canvas.size
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)
        self.assertEqual(prepared.reference_box[0], 0)
        self.assertEqual(prepared.reference_box[2], prepared.target_box[0])
        self.assertEqual(prepared.target_box[2], width)
        self.assertEqual(prepared.mask.getpixel((0, height // 2)), 0)
        self.assertEqual(prepared.mask.getpixel((width - 1, height // 2)), 255)
        self.assertEqual(
            prepared.canvas.getpixel((width - 1, height // 2)),
            (0x12, 0x34, 0x56),
        )

    def test_left_layout_and_overlap_do_not_change_crop_region(self):
        reference = Image.new("RGB", (512, 512), "blue")
        prepared = prepare_reference_canvas(
            reference,
            ReferenceCanvasConfig(
                output_width=768,
                output_height=1024,
                placement="left",
                mask_overlap=32,
                composite_megapixels=0,
            ),
        )

        self.assertEqual(prepared.target_box[0], 0)
        self.assertGreater(prepared.mask_box[2], prepared.target_box[2])
        self.assertEqual(prepared.mask.getpixel((0, 10)), 255)
        self.assertEqual(
            prepared.mask.getpixel((prepared.canvas.width - 1, 10)),
            0,
        )

    def test_crop_returns_exact_requested_size_even_if_generated_size_differs(self):
        prepared = prepare_reference_canvas(
            Image.new("RGB", (400, 800), "white"),
            ReferenceCanvasConfig(
                output_width=832,
                output_height=1216,
                placement="right",
                composite_megapixels=0,
            ),
        )
        generated = prepared.canvas.resize(
            (prepared.canvas.width * 2, prepared.canvas.height * 2)
        )
        result = crop_reference_result(generated, prepared)
        self.assertEqual(result.size, (832, 1216))

    def test_transparent_reference_uses_user_matte_color(self):
        reference = Image.new("RGBA", (128, 128), (255, 0, 0, 0))
        prepared = prepare_reference_canvas(
            reference,
            ReferenceCanvasConfig(
                composite_megapixels=0,
                reference_matte_color="#00ff00",
            ),
        )
        x = (prepared.reference_box[0] + prepared.reference_box[2]) // 2
        y = prepared.canvas.height // 2
        self.assertEqual(prepared.canvas.getpixel((x, y)), (0, 255, 0))

    def test_invalid_values_fail_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "Output width"):
            prepare_reference_canvas(
                Image.new("RGB", (64, 64)),
                ReferenceCanvasConfig(output_width=1),
            )
        with self.assertRaises(ValueError):
            prepare_reference_canvas(
                Image.new("RGB", (64, 64)),
                ReferenceCanvasConfig(target_color="not-a-color"),
            )


class AnimaReferencePromptTests(unittest.TestCase):
    def test_every_prompt_part_is_configurable(self):
        prompt = compose_reference_prompt(
            "standing in rain",
            prefix_enabled=True,
            prefix_text="split screen, multiple views",
            prefix_strength=2.0,
            edit_lora_enabled=True,
            edit_lora_name=r"anima\AnimeEditV2.safetensors",
            edit_lora_strength=0.65,
            extend_lora_enabled=True,
            extend_lora_name="Extend Image (Anima Edit) v1.safetensors",
            extend_lora_strength=0.35,
            extra_prefix="score_7",
            extra_suffix="night",
        )

        self.assertIn("<lora:anima/AnimeEditV2:0.65>", prompt)
        self.assertIn("<lora:Extend Image (Anima Edit) v1:0.35>", prompt)
        self.assertIn("(split screen, multiple views:2)", prompt)
        self.assertIn("score_7", prompt)
        self.assertTrue(prompt.endswith("night"))

    def test_all_automatic_prompt_parts_can_be_disabled(self):
        prompt = compose_reference_prompt(
            "plain prompt",
            prefix_enabled=False,
            edit_lora_enabled=False,
            edit_lora_name="ignored.safetensors",
            extend_lora_enabled=False,
            extend_lora_name="ignored.safetensors",
        )
        self.assertEqual(prompt, "plain prompt")


if __name__ == "__main__":
    unittest.main()
