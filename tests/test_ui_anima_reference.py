from __future__ import annotations

import json
import unittest

import gradio as gr
from PIL import Image

from sam3ext.ui_anima_reference import (
    REFERENCE_ARG_KEYS,
    _merge_gallery,
    build_anima_reference_panel,
    missing_enabled_loras,
    request_from_values,
)


class AnimaReferenceUiTests(unittest.TestCase):
    def _panel(self):
        with gr.Blocks():
            return build_anima_reference_panel(["Euler a"], ["Simple"])

    def test_every_request_key_has_one_editable_widget(self):
        panel = self._panel()
        widgets = panel.all_widgets()

        self.assertEqual(len(widgets), len(REFERENCE_ARG_KEYS))
        self.assertEqual(
            [widget is getattr(panel, key) for widget, key in zip(widgets, REFERENCE_ARG_KEYS)],
            [True] * len(REFERENCE_ARG_KEYS),
        )

    def test_component_defaults_map_to_typed_request(self):
        panel = self._panel()
        values = [widget.value for widget in panel.all_widgets()]
        image = Image.new("RGB", (512, 768), "white")
        values[REFERENCE_ARG_KEYS.index("reference_image")] = image

        request, gallery_mode, insert_mode = request_from_values(
            tuple(values),
            reference_image=image,
            main_prompt="main positive",
            main_negative="main negative",
        )

        self.assertEqual(request.prompt, "main positive")
        self.assertEqual(request.negative_prompt, "main negative")
        self.assertEqual(request.steps, 30)
        self.assertEqual(request.canvas.output_width, 960)
        self.assertEqual(gallery_mode, "Target only")
        self.assertEqual(insert_mode, "At end")

    def test_non_default_values_all_reach_the_request_model(self):
        panel = self._panel()
        keyed = {
            key: widget.value
            for key, widget in zip(REFERENCE_ARG_KEYS, panel.all_widgets())
        }
        image = Image.new("RGBA", (333, 555), (1, 2, 3, 4))
        keyed.update(
            {
                "reference_image": image,
                "output_width": 704,
                "output_height": 896,
                "placement": "left",
                "target_region_scale": 1.25,
                "target_color": "#123456",
                "reference_matte_color": "#654321",
                "composite_megapixels": 2.2,
                "dimension_multiple": 32,
                "resize_filter": "bicubic",
                "mask_overlap": 17,
                "inherit_main_prompt": False,
                "prompt": "panel positive",
                "inherit_main_negative": False,
                "negative_prompt": "panel negative",
                "prefix_enabled": False,
                "prefix_text": "custom prefix",
                "prefix_strength": 1.7,
                "extra_prefix": "before",
                "extra_suffix": "after",
                "edit_lora_enabled": False,
                "edit_lora_name": "edit-custom",
                "edit_lora_strength": 0.61,
                "extend_lora_enabled": False,
                "extend_lora_name": "extend-custom",
                "extend_lora_strength": 0.31,
                "require_enabled_loras": False,
                "checkpoint_override": "checkpoint-custom",
                "override_additional_modules": True,
                "additional_modules": ["vae.safetensors", "te.safetensors"],
                "steps": 44,
                "cfg_scale": 4.4,
                "shift": 2.8,
                "sampler": "sampler-custom",
                "scheduler": "scheduler-custom",
                "denoising_strength": 0.88,
                "resize_mode": "Resize and Fill",
                "inpainting_fill": "original",
                "mask_blur": 9,
                "mask_round": False,
                "mask_invert": True,
                "inpainting_mask_weight": 0.73,
                "initial_noise_multiplier": 0.91,
                "eta": 0.66,
                "s_min_uncond": 0.2,
                "s_churn": 1.1,
                "s_tmin": 0.3,
                "s_tmax": 8.0,
                "s_noise": 0.95,
                "seed": 1000,
                "seed_step": 7,
                "candidate_count": 3,
                "restore_faces": True,
                "native_reference_enabled": False,
                "save_target": False,
                "save_generated_canvas": True,
                "save_input_canvas": True,
                "save_mask": True,
                "gallery_content": "Target + generated canvas",
                "insert_mode": "Replace gallery",
            }
        )
        values = tuple(keyed[key] for key in REFERENCE_ARG_KEYS)
        request, gallery_mode, insert_mode = request_from_values(
            values,
            reference_image=image,
            main_prompt="ignored main",
            main_negative="ignored negative",
        )

        self.assertEqual(
            request.canvas,
            request.canvas.__class__(
                output_width=704,
                output_height=896,
                placement="left",
                target_region_scale=1.25,
                target_color="#123456",
                reference_matte_color="#654321",
                composite_megapixels=2.2,
                dimension_multiple=32,
                resize_filter="bicubic",
                mask_overlap=17,
            ),
        )
        expected = {
            "prompt": "panel positive",
            "negative_prompt": "panel negative",
            "prefix_enabled": False,
            "prefix_text": "custom prefix",
            "prefix_strength": 1.7,
            "extra_prefix": "before",
            "extra_suffix": "after",
            "edit_lora_enabled": False,
            "edit_lora_name": "edit-custom",
            "edit_lora_strength": 0.61,
            "extend_lora_enabled": False,
            "extend_lora_name": "extend-custom",
            "extend_lora_strength": 0.31,
            "require_enabled_loras": False,
            "checkpoint_override": "checkpoint-custom",
            "override_additional_modules": True,
            "additional_modules": ("vae.safetensors", "te.safetensors"),
            "steps": 44,
            "cfg_scale": 4.4,
            "shift": 2.8,
            "sampler": "sampler-custom",
            "scheduler": "scheduler-custom",
            "denoising_strength": 0.88,
            "resize_mode": "Resize and Fill",
            "inpainting_fill": "original",
            "mask_blur": 9,
            "mask_round": False,
            "mask_invert": True,
            "inpainting_mask_weight": 0.73,
            "initial_noise_multiplier": 0.91,
            "eta": 0.66,
            "s_min_uncond": 0.2,
            "s_churn": 1.1,
            "s_tmin": 0.3,
            "s_tmax": 8.0,
            "s_noise": 0.95,
            "seed": 1000,
            "seed_step": 7,
            "candidate_count": 3,
            "restore_faces": True,
            "native_reference_enabled": False,
            "save_target": False,
            "save_generated_canvas": True,
            "save_input_canvas": True,
            "save_mask": True,
        }
        for name, value in expected.items():
            self.assertEqual(getattr(request, name), value, name)
        self.assertEqual(gallery_mode, "Target + generated canvas")
        self.assertEqual(insert_mode, "Replace gallery")

    def test_missing_lora_guard_uses_installed_set_not_suggestions(self):
        panel = self._panel()
        values = [widget.value for widget in panel.all_widgets()]
        image = Image.new("RGB", (128, 128))
        values[REFERENCE_ARG_KEYS.index("reference_image")] = image
        request, _, _ = request_from_values(
            tuple(values),
            reference_image=image,
            main_prompt="",
            main_negative="",
        )

        self.assertEqual(
            missing_enabled_loras(request, available=[]),
            ["AnimeEditV2", "Extend Image (Anima Edit) v1"],
        )
        self.assertEqual(
            missing_enabled_loras(
                request,
                available=["AnimeEditV2", "Extend Image (Anima Edit) v1"],
            ),
            [],
        )

    def test_gallery_replace_is_transactional_and_updates_infotexts(self):
        old = [Image.new("RGB", (8, 8), "red")]
        new = [Image.new("RGB", (8, 8), "blue")]
        updated, payload = _merge_gallery(
            old,
            0,
            new,
            ["new info"],
            json.dumps({"infotexts": ["old info"]}),
            "Replace gallery",
        )

        self.assertEqual(updated, new)
        self.assertEqual(json.loads(payload)["infotexts"], ["new info"])


if __name__ == "__main__":
    unittest.main()
