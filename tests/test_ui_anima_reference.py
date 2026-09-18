from __future__ import annotations

import json
import sys
import types
import unittest
from unittest import mock

import gradio as gr
from PIL import Image

from sam3ext import ui_anima_reference as ui
from sam3ext.anima_reference_core import prepare_reference_canvas
from sam3ext.anima_reference_recipe import (
    WARN_EXTEND_MISSING,
    WARN_WILDCARD,
    DetectedLoras,
)
from sam3ext.anima_reference_runner import (
    ReferenceGenerationOutput,
    ReferenceGenerationResult,
)
from sam3ext.ui_anima_reference import (
    REFERENCE_ARG_KEYS,
    _merge_gallery,
    _run_exclusive,
    build_anima_reference_panel,
    request_from_values,
    resolve_reference_image,
)

# Labels of Forge txt2img widgets whose saved ui-config values used to leak
# into Feature 6 (ui-config.json:41, 751, 2977).
_TXT2IMG_LABELS = {
    "Denoising strength",
    "Steps",
    "Masked content",
    "Sampler",
    "Scheduler",
    "Seed",
    "CFG scale",
}


def _panel():
    with gr.Blocks():
        return build_anima_reference_panel(["Euler a", "ER SDE"], ["Simple"])


def _defaults(panel, **changes):
    keyed = {key: widget.value for key, widget in zip(REFERENCE_ARG_KEYS, panel.all_widgets())}
    keyed.update(changes)
    return tuple(keyed[key] for key in REFERENCE_ARG_KEYS)


def _map(values, **kwargs):
    image = Image.new("RGB", (512, 768), "white")
    options = dict(
        reference_image=image,
        main_prompt="1girl, smile",
        main_negative="lowres",
        txt2img_width=832,
        txt2img_height=1216,
        available_loras=["AnimeEditV2"],
        forge_opts={},
    )
    options.update(kwargs)
    return request_from_values(values, **options)


class PanelTests(unittest.TestCase):
    def test_every_request_key_has_one_widget(self):
        panel = _panel()
        widgets = panel.all_widgets()
        self.assertEqual(len(widgets), len(REFERENCE_ARG_KEYS))
        for widget, key in zip(widgets, REFERENCE_ARG_KEYS):
            self.assertIs(widget, getattr(panel, key), key)

    def test_labels_are_prefixed_so_txt2img_values_cannot_leak_in(self):
        panel = _panel()
        labels = [widget.label for widget in panel.all_widgets()]
        for label in labels:
            self.assertTrue(label.startswith("[Ref] "), label)
            self.assertNotIn(label, _TXT2IMG_LABELS)
        self.assertEqual(len(labels), len(set(labels)))

    def test_reference_image_widget_keeps_alpha_for_the_matte_composite(self):
        # F1: Gradio 4.40's gr.Image defaults image_mode="RGB" and flattens
        # transparency in preprocess() before resolve_reference_image ever
        # sees the upload; the widget must opt into RGBA explicitly.
        panel = _panel()
        self.assertEqual(panel.reference_image.image_mode, "RGBA")


class RequestMappingTests(unittest.TestCase):
    def test_defaults_follow_txt2img_size_and_the_restyler_recipe(self):
        request, detected, warnings = _map(_defaults(_panel()))

        self.assertEqual((request.canvas.output_width, request.canvas.output_height), (832, 1216))
        self.assertEqual(request.inpainting_fill, "original")
        self.assertEqual(request.canvas.target_color, "#000000")
        self.assertEqual(request.denoising_strength, 1.0)
        self.assertEqual(request.edit_lora_name, "AnimeEditV2")
        self.assertEqual(request.edit_lora_strength, 0.72)
        self.assertFalse(request.extend_lora_enabled)
        self.assertEqual(
            (request.sampler, request.scheduler, request.steps, request.cfg_scale, request.shift),
            ("Euler a", "Simple", 30, 5.0, 3.0),
        )
        self.assertEqual(request.candidate_count, 2)
        self.assertEqual(request.prompt, "1girl, smile")
        self.assertEqual(request.negative_prompt, "lowres")
        self.assertEqual(request.keep_scope, "identity")
        self.assertEqual(request.sampling_source, "recipe")
        self.assertEqual(detected, DetectedLoras(edit="AnimeEditV2", extend=None))
        self.assertEqual(warnings, [])

    def test_custom_size_overrides_txt2img(self):
        values = _defaults(_panel(), custom_size_enabled=True, output_width=704, output_height=896)
        request, _, _ = _map(values)
        self.assertEqual((request.canvas.output_width, request.canvas.output_height), (704, 896))

    def test_panel_prompt_is_used_verbatim_and_inherited_prompt_is_cleaned(self):
        main = "<lora:AnimeEditV2:0.72>, (split screen, multiple views:1.2), 1girl, <lora:styleA:0.5>"
        request, _, _ = _map(_defaults(_panel()), main_prompt=main)
        self.assertEqual(request.prompt, "1girl, <lora:styleA:0.5>")

        request, _, _ = _map(_defaults(_panel(), keep_scope="style"), main_prompt=main)
        self.assertEqual(request.prompt, "1girl")

        values = _defaults(_panel(), prompt="custom <lora:AnimeEditV2:1>")
        request, _, _ = _map(values, main_prompt=main)
        self.assertEqual(request.prompt, "custom <lora:AnimeEditV2:1>")

    def test_outfit_scope_raises_edit_strength(self):
        request, _, _ = _map(_defaults(_panel(), keep_scope="outfit"))
        self.assertEqual(request.edit_lora_strength, 0.8)
        self.assertEqual(request.keep_scope, "outfit")

    def test_forge_preset_replaces_panel_sampling(self):
        opts = {
            "anima_i2i_sampler": "ER SDE",
            "anima_i2i_scheduler": "Beta57 (RES4LYF)",
            "anima_i2i_step": 50,
            "anima_i2i_cfg": 5,
            "anima_i2i_dcfg": 3,
        }
        request, _, _ = _map(_defaults(_panel(), follow_forge_preset=True), forge_opts=opts)
        self.assertEqual(
            (request.sampler, request.scheduler, request.steps, request.shift, request.sampling_source),
            ("ER SDE", "Beta57 (RES4LYF)", 50, 3.0, "forge_preset"),
        )

    def test_missing_extend_is_dropped_with_a_warning(self):
        request, detected, warnings = _map(_defaults(_panel(), extend_lora_enabled=True))
        self.assertFalse(request.extend_lora_enabled)
        self.assertIsNone(detected.extend)
        self.assertIn(WARN_EXTEND_MISSING, warnings)

        request, _, warnings = _map(
            _defaults(_panel(), extend_lora_enabled=True),
            available_loras=["AnimeEditV2", "Extend Image (Anima Edit) v1"],
        )
        self.assertTrue(request.extend_lora_enabled)
        self.assertEqual(request.extend_lora_name, "Extend Image (Anima Edit) v1")
        self.assertEqual(warnings, [])

    def test_missing_edit_lora_is_reported_not_guessed(self):
        request, detected, _ = _map(_defaults(_panel()), available_loras=[])
        self.assertIsNone(detected.edit)
        self.assertEqual(request.edit_lora_name, "")

    def test_smear_fill_and_debug_saves(self):
        values = _defaults(_panel(), fill_mode="smear", save_debug_images=True)
        request, _, _ = _map(values)
        self.assertEqual(request.inpainting_fill, "latent noise")
        self.assertTrue(request.save_generated_canvas)
        self.assertTrue(request.save_input_canvas)
        self.assertTrue(request.save_mask)

    def test_inherited_prompt_syntax_is_warned(self):
        _, _, warnings = _map(_defaults(_panel()), main_prompt="1girl, __hair__")
        self.assertEqual(warnings, [WARN_WILDCARD])

    def test_radio_widgets_default_to_labels_and_map_to_request_fields(self):
        # F6: Forge's ui_loadsave.radio_choices compares a saved value with
        # the display labels, so (label, value) tuple choices never restore.
        panel = _panel()
        self.assertEqual(panel.keep_scope.value, "정체성")
        self.assertEqual(panel.fill_mode.value, "원본 (단색 그대로)")

        values = _defaults(
            _panel(), keep_scope="정체성+의상", fill_mode="번진 이미지"
        )
        request, _, _ = _map(values)
        self.assertEqual(request.keep_scope, "outfit")
        self.assertEqual(request.inpainting_fill, "latent noise")

        # Internal values (what a saved ui-config, or an old caller, sends)
        # must still work.
        values = _defaults(_panel(), keep_scope="outfit", fill_mode="smear")
        request, _, _ = _map(values)
        self.assertEqual(request.keep_scope, "outfit")
        self.assertEqual(request.inpainting_fill, "latent noise")


class GalleryTests(unittest.TestCase):
    def test_gallery_replace_is_transactional_and_updates_infotexts(self):
        old = [Image.new("RGB", (8, 8), "red")]
        new = [Image.new("RGB", (8, 8), "blue")]
        updated, payload = _merge_gallery(
            old, 0, new, ["new info"], json.dumps({"infotexts": ["old info"]}), "Replace gallery"
        )
        self.assertEqual(updated, new)
        self.assertEqual(json.loads(payload)["infotexts"], ["new info"])


class StatusFormattingTests(unittest.TestCase):
    """F4: status-line diagnostics required by spec 5.1 / 5.2-4."""

    def _status_text(self, diagnostics, **map_kwargs):
        request, detected, warnings = _map(_defaults(_panel()), **map_kwargs)
        prepared = prepare_reference_canvas(request.reference_image, request.canvas)
        output = ReferenceGenerationOutput(
            target_image=Image.new(
                "RGB", (request.canvas.output_width, request.canvas.output_height)
            ),
            generated_canvas=prepared.canvas,
            infotext="",
            seed=1,
        )
        result = ReferenceGenerationResult(
            prepared=prepared, outputs=(output,), diagnostics=diagnostics
        )
        return ui.format_reference_status(result, request, detected, warnings, "uploaded")

    def test_block_count_other_than_28_reports_the_compat_hook(self):
        text = self._status_text({"model_blocks": 52, "anima38": "v2 bundle"})
        self.assertIn(
            "Edit LoRA 28블록 → 52블록 모델 (블록 호환 훅으로 적용)", text
        )

    def test_default_28_block_model_has_no_compat_hook_line(self):
        text = self._status_text({"model_blocks": 28, "anima38": "not attempted"})
        self.assertNotIn("블록 호환 훅", text)

    def test_v1_38b_checkpoint_gets_an_explicit_warning(self):
        text = self._status_text(
            {"model_blocks": 52, "anima38": "not a 3.8B v2 bundle"}
        )
        self.assertIn(
            "3.8B v1 체크포인트는 캐릭터 레퍼런스에서 어댑터를 쓰지 않고 "
            "기본 Anima로 진행했습니다",
            text,
        )
        # Not a connector failure, so it stays visible in the green line too.
        self.assertIn("3.8B 커넥터: not a 3.8B v2 bundle", text)

    def test_anima38_failure_labels_move_to_the_warning_line_not_the_green_one(self):
        for label in (
            "unavailable (ImportError: no torch)",
            "missing encoder (qwen35_4b.safetensors not found in models/text_encoder)",
            "install failed (RuntimeError: boom)",
            "check failed (RuntimeError)",
        ):
            text = self._status_text({"model_blocks": 28, "anima38": label})
            green, marker, warning = text.partition("⚠")
            self.assertTrue(marker, text)
            self.assertNotIn(label, green, label)
            self.assertIn(label, warning, label)
            self.assertIn("3.8B 커넥터: 꺼짐", green, label)


_F6 = "Steps: 30, SAM3 Feature: 6 - Anima Character Reference"


def _extras(width=832, height=1216, info=None):
    return ("1girl", "lowres", info or json.dumps({"infotexts": ["Steps: 20"]}), width, height)


def _fake_result(request):
    prepared = prepare_reference_canvas(request.reference_image, request.canvas)
    output = ReferenceGenerationOutput(
        target_image=Image.new("RGB", (request.canvas.output_width, request.canvas.output_height)),
        generated_canvas=prepared.canvas,
        infotext=_F6,
        seed=7,
    )
    return ReferenceGenerationResult(
        prepared=prepared,
        outputs=(output,),
        diagnostics={
            "target_panel": (688, 1008),
            "upscale": 1.21,
            "model_blocks": 52,
            "interrupted": False,
            "anima38": "v2 bundle",
        },
    )


class ReferenceSourceTests(unittest.TestCase):
    def test_upload_is_kept_as_is_including_transparency(self):
        upload = Image.new("RGBA", (10, 20), (0, 0, 0, 0))
        image, index, source = resolve_reference_image(upload, [], -1, "")
        self.assertIs(image, upload)
        self.assertEqual((index, source), (-1, "uploaded"))

    def test_gallery_fallback_skips_previous_reference_outputs(self):
        first = Image.new("RGB", (8, 8), "red")
        second = Image.new("RGB", (8, 8), "blue")
        info = json.dumps({"infotexts": ["Steps: 20", _F6]})
        image, index, source = resolve_reference_image(None, [first, second], 1, info)
        self.assertEqual((index, source), (0, "gallery"))
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_no_usable_image_is_reported(self):
        info = json.dumps({"infotexts": [_F6]})
        self.assertEqual(
            resolve_reference_image(None, [Image.new("RGB", (8, 8))], -1, info),
            (None, -1, "none"),
        )


class ExclusiveRunTests(unittest.TestCase):
    def _fake_modules(self, events):
        class Lock:
            def __enter__(self):
                events.append("lock")
                return self

            def __exit__(self, *exc):
                events.append("unlock")
                return False

        package = types.ModuleType("modules")
        package.__path__ = []
        shared = types.ModuleType("modules.shared")
        shared.state = types.SimpleNamespace(
            begin=lambda job: events.append(("begin", job)),
            end=lambda: events.append("end"),
        )
        call_queue = types.ModuleType("modules.call_queue")
        call_queue.queue_lock = Lock()
        package.shared = shared
        package.call_queue = call_queue
        return {"modules": package, "modules.shared": shared, "modules.call_queue": call_queue}

    def test_run_holds_the_queue_lock_and_starts_a_fresh_job(self):
        events = []
        with mock.patch.dict(sys.modules, self._fake_modules(events)):
            result = _run_exclusive("job-x", lambda: events.append("run") or "ok")
        self.assertEqual(result, "ok")
        self.assertEqual(events, ["lock", ("begin", "job-x"), "run", "end", "unlock"])

    def test_job_is_ended_even_when_the_run_fails(self):
        events = []

        def boom():
            raise RuntimeError("x")

        with mock.patch.dict(sys.modules, self._fake_modules(events)):
            with self.assertRaises(RuntimeError):
                _run_exclusive("job-y", boom)
        self.assertEqual(events, ["lock", ("begin", "job-y"), "end", "unlock"])


class StopReferenceTests(unittest.TestCase):
    """F7: Stop must not interrupt an unrelated job holding the queue."""

    def _fake_modules(self, job):
        package = types.ModuleType("modules")
        package.__path__ = []
        shared = types.ModuleType("modules.shared")
        shared.state = types.SimpleNamespace(job=job, interrupted=False, skipped=False)
        package.shared = shared
        return {"modules": package, "modules.shared": shared}, shared

    def test_stop_interrupts_a_feature_6_job(self):
        for job in (
            "sam3_character_reference",
            "Anima Reference 2/3",
            "Anima Character Reference",
        ):
            modules, shared = self._fake_modules(job)
            with mock.patch.dict(sys.modules, modules):
                ui._stop_reference()
            self.assertTrue(shared.state.interrupted, job)
            self.assertTrue(shared.state.skipped, job)

    def test_stop_leaves_an_unrelated_job_running(self):
        modules, shared = self._fake_modules("txt2img")
        with mock.patch.dict(sys.modules, modules):
            ui._stop_reference()
        self.assertFalse(shared.state.interrupted)
        self.assertFalse(shared.state.skipped)


class HandlerTests(unittest.TestCase):
    def _click(self, gallery, values, loras=("AnimeEditV2",), run=None):
        captured = {}

        def fake_run(request):
            captured["request"] = request
            return (run or _fake_result)(request)

        with (
            mock.patch.object(ui, "run_anima_reference", side_effect=fake_run) as run_mock,
            mock.patch.object(ui, "list_installed_lora_names", return_value=list(loras)),
            mock.patch.object(ui, "_run_exclusive", side_effect=lambda job, fn: fn()),
            mock.patch.object(ui, "_forge_opts_data", return_value={}),
        ):
            outputs = ui.handle_anima_reference_click(gallery, -1, *values, *_extras())
        return outputs, captured, run_mock

    def test_gallery_fallback_is_used_pinned_and_reported(self):
        gallery = [Image.new("RGB", (64, 96), "blue")]
        outputs, captured, _ = self._click(gallery, _defaults(_panel()))

        self.assertEqual(len(outputs), 7)
        self.assertEqual(captured["request"].canvas.output_width, 832)
        self.assertEqual(len(outputs[0]), 2)
        self.assertEqual(outputs[6]["value"].size, (64, 96))
        self.assertIn("v2 bundle", outputs[1])
        self.assertIn("52", outputs[1])
        self.assertIn(_F6, json.loads(outputs[3])["infotexts"])

    def test_uploaded_reference_is_not_pinned_again(self):
        values = list(_defaults(_panel()))
        values[REFERENCE_ARG_KEYS.index("reference_image")] = Image.new("RGB", (32, 32))
        outputs, _, _ = self._click([], tuple(values))
        self.assertNotIn("value", outputs[6])

    def test_missing_edit_lora_blocks_before_running(self):
        gallery = [Image.new("RGB", (64, 96), "blue")]
        outputs, _, run_mock = self._click(gallery, _defaults(_panel()), loras=())
        self.assertIn("AnimeEditV2", outputs[1])
        run_mock.assert_not_called()

    def test_warnings_reach_the_status(self):
        gallery = [Image.new("RGB", (64, 96), "blue")]
        values = _defaults(_panel(), extend_lora_enabled=True)
        outputs, _, _ = self._click(gallery, values)
        self.assertIn(WARN_EXTEND_MISSING, outputs[1])

    def test_preview_uses_the_same_fallback(self):
        gallery = [Image.new("RGB", (64, 96), "blue")]
        canvas, mask, status = ui.preview_reference_layout(
            gallery, -1, *_defaults(_panel()), *_extras()
        )
        self.assertIsInstance(canvas, Image.Image)
        self.assertIsInstance(mask, Image.Image)
        self.assertIn("갤러리", status)


class WiringTests(unittest.TestCase):
    def test_generate_and_preview_receive_the_full_input_vector(self):
        with gr.Blocks() as demo:
            panel = build_anima_reference_panel(["Euler a"], ["Simple"])
            gallery = gr.Gallery()
            prompt = gr.Textbox()
            negative = gr.Textbox()
            html_info = gr.HTML()
            info = gr.Textbox()
            width = gr.Slider(minimum=64, maximum=2048, value=832)
            height = gr.Slider(minimum=64, maximum=2048, value=1216)
            ui.wire_anima_reference_panel(
                panel,
                gallery=gallery,
                main_prompt=prompt,
                main_negative=negative,
                html_info=html_info,
                generation_info=info,
                width=width,
                height=height,
                selected_index_js="(...args) => args",
            )
        by_name = {fn.name: fn for fn in demo.fns.values()}
        run = by_name["handle_anima_reference_click"]
        self.assertEqual(len(run.inputs), 2 + len(REFERENCE_ARG_KEYS) + 5)
        self.assertIs(run.inputs[-2], width)
        self.assertIs(run.inputs[-1], height)
        self.assertEqual(len(run.outputs), 7)
        self.assertIs(run.outputs[6], panel.reference_image)
        preview = by_name["preview_reference_layout"]
        self.assertEqual(len(preview.inputs), len(run.inputs))

    def test_missing_size_components_fall_back_to_hidden_numbers(self):
        with gr.Blocks() as demo:
            panel = build_anima_reference_panel(["Euler a"], ["Simple"])
            ui.wire_anima_reference_panel(
                panel,
                gallery=gr.Gallery(),
                main_prompt=gr.Textbox(),
                main_negative=gr.Textbox(),
                html_info=gr.HTML(),
                generation_info=gr.Textbox(),
                width=None,
                height=None,
                selected_index_js="(...args) => args",
            )
        run = {fn.name: fn for fn in demo.fns.values()}["handle_anima_reference_click"]
        self.assertIs(run.inputs[-2], panel.txt2img_width_fallback)
        self.assertIs(run.inputs[-1], panel.txt2img_height_fallback)


if __name__ == "__main__":
    unittest.main()
