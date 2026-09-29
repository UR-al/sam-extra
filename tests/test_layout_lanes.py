"""layout_lanes — 스크립트 파일 이름으로 만든 키, 묶음표, Gradio elem_classes 태깅.

프런트엔드(javascript/notebook_lanes.js)는 여기서 붙인 클래스만 보고 항목을 알아본다. 라벨 글자는 ko_KR 이
바꾸고 component-N id 는 실행마다 달라지므로 둘 다 쓰지 않는다.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext import layout_lanes as ll  # noqa: E402


class SlotKeyTests(unittest.TestCase):
    def test_key_comes_from_the_file_name(self):
        cases = [
            ("C:/forge/extensions/aadetailer-neoforge/scripts/!adetailer.py", "adetailer"),
            ("/forge/extensions/x/scripts/lora_block_weight.py", "lora-block-weight"),
            ("/forge/extensions/forge_sam3_extension/scripts/!sam3.py", "sam3"),
            ("/forge/extensions/forge_sam3_extension/scripts/anima_3_8b.py", "anima-3-8b"),
            ("/forge/extensions-builtin/x/scripts/forge_never_oom.py", "forge-never-oom"),
        ]
        for path, key in cases:
            with self.subTest(path=path):
                self.assertEqual(ll.slot_key(path), key)


class RegistryTests(unittest.TestCase):
    LANES = {"anima", "det", "lora", "etc", "hidden"}

    def test_every_entry_has_a_known_lane(self):
        self.assertTrue(ll.REGISTRY)
        for key, slot in ll.REGISTRY.items():
            with self.subTest(key=key):
                self.assertIn(slot.lane, self.LANES)

    def test_expected_placement(self):
        expected = {
            "anima-3-8b": "anima", "anima-detail-daemon": "anima", "anima-skimmed-cfg": "anima",
            "anima-safe-pag": "anima", "anima-vae-2x": "anima",
            "sam3": "det", "adetailer": "det", "anima-vae-degrid": "det",
            "lora-block-weight": "lora", "dora-infer-mode": "lora", "controlnet": "lora", "dynamic-prompting": "lora",
            "dynamic-thresholding": "lora",
            "anima-ref-poc": "etc", "api-payload-display": "etc", "compile": "etc",
        }
        for key, lane in expected.items():
            with self.subTest(key=key):
                self.assertEqual(ll.REGISTRY[key].lane, lane)

    def test_only_vae_2x_is_marked_experimental(self):
        marked = {key for key, slot in ll.REGISTRY.items() if slot.exp}
        self.assertEqual(marked, {"anima-vae-2x"})

    def test_unknown_extensions_fall_into_tools(self):
        self.assertEqual(ll.lane_for("some-new-extension"), "etc")

    def test_the_option_key_is_the_one_the_settings_page_registers(self):
        self.assertEqual(ll.OPT_LAYOUT_SECTIONS, "sam3_layout_sections")


class _FakeScript:
    def __init__(self, filename, group, controls, section=None, create_group=True):
        self.filename = filename
        self.group = group
        self.controls = controls
        self.section = section
        self.create_group = create_group


class _FakeRunner:
    def __init__(self, scripts):
        self.alwayson_scripts = scripts
        self.selectable_scripts = []


def _accordion_group(label, *, enable_label="Enable Foo", elem_classes=None, input_accordion=False):
    """Forge 스크립트 하나가 만드는 묶음을 실제 Gradio 컴포넌트로 흉내 낸다."""
    with gr.Group() as group:
        with gr.Accordion(label, open=False) as acc:
            box = gr.Checkbox(label=enable_label, value=False, elem_classes=elem_classes)
            if input_accordion:
                value_box = gr.Checkbox(label=label, value=False, visible=False)
                value_box.accordion_id = "fake-accordion"
                value_box.accordion = acc
                return group, [value_box, box]
    return group, [box]


class TagRunnerTests(unittest.TestCase):
    def test_group_gets_slot_lane_and_head_classes(self):
        with gr.Blocks():
            group, controls = _accordion_group("A디테일러", input_accordion=True)
            runner = _FakeRunner([_FakeScript("/x/scripts/!adetailer.py", group, controls)])
            tagged = ll.tag_runner(runner)
        self.assertEqual(tagged, 1)
        self.assertIn("sam3-slot", group.elem_classes)
        self.assertIn("sam3-slot--adetailer", group.elem_classes)
        self.assertIn("sam3-lane--det", group.elem_classes)

    def test_input_accordion_value_checkbox_is_marked_on(self):
        with gr.Blocks():
            group, controls = _accordion_group("A디테일러", input_accordion=True)
            runner = _FakeRunner([_FakeScript("/x/scripts/!adetailer.py", group, controls)])
            ll.tag_runner(runner)
        value_box, plain_box = controls
        self.assertIn("sam3-on", value_box.elem_classes)
        self.assertNotIn("sam3-on", plain_box.elem_classes or [])

    def test_registry_rule_beats_the_guess(self):
        with gr.Blocks():
            group, controls = _accordion_group(
                "ControlNet Integrated", enable_label="Enable", elem_classes=["cnet-unit-enabled"]
            )
            runner = _FakeRunner([_FakeScript("/x/scripts/controlnet.py", group, controls)])
            ll.tag_runner(runner)
        self.assertIn("sam3-on", controls[0].elem_classes)
        self.assertNotIn("sam3-on-guess", group.elem_classes)

    def test_guess_is_marked(self):
        with gr.Blocks():
            group, controls = _accordion_group("새 확장", enable_label="Enable Foo")
            runner = _FakeRunner([_FakeScript("/x/scripts/brand_new.py", group, controls)])
            ll.tag_runner(runner)
        self.assertIn("sam3-on", controls[0].elem_classes)
        self.assertIn("sam3-on-guess", group.elem_classes)
        self.assertIn("sam3-lane--etc", group.elem_classes)

    def test_rule_none_means_no_on_control(self):
        with gr.Blocks():
            group, controls = _accordion_group("API payload", enable_label="Enabled")
            runner = _FakeRunner([_FakeScript("/x/scripts/api_payload_display.py", group, controls)])
            ll.tag_runner(runner)
        self.assertNotIn("sam3-on", controls[0].elem_classes or [])

    def test_empty_group_is_hidden_and_core_sections_are_skipped(self):
        with gr.Blocks():
            with gr.Group() as empty:
                pass
            with gr.Group() as core:
                gr.Checkbox(label="Enable", value=False)
            runner = _FakeRunner([
                _FakeScript("/x/scripts/negpip.py", empty, []),
                _FakeScript("/x/scripts/sampler.py", core, [], section="sampler"),
                _FakeScript("/x/scripts/nogroup.py", None, [], create_group=False),
            ])
            tagged = ll.tag_runner(runner)
        self.assertEqual(tagged, 1)
        self.assertIn("sam3-lane--hidden", empty.elem_classes)
        self.assertNotIn("sam3-slot", core.elem_classes or [])

    def test_running_twice_adds_no_duplicates_and_keeps_labels(self):
        with gr.Blocks():
            group, controls = _accordion_group("SAM3 Mask", enable_label="Enable SAM3")
            runner = _FakeRunner([_FakeScript("/x/scripts/!sam3.py", group, controls)])
            ll.tag_runner(runner)
            ll.tag_runner(runner)
        self.assertEqual(group.elem_classes.count("sam3-slot"), 1)
        self.assertEqual(controls[0].label, "Enable SAM3")

    def test_the_anima_section_is_included(self):
        """섹션이 None 이 아닌 스크립트를 건너뛰면 1열 ANIMA 묶음이 통째로 사라진다."""
        with gr.Blocks():
            group, controls = _accordion_group("Anima 3.8B", enable_label="Enable", input_accordion=True)
            runner = _FakeRunner([_FakeScript("/x/scripts/anima_3_8b.py", group, controls, section="sam3_anima")])
            self.assertEqual(ll.tag_runner(runner), 1)
        self.assertIn("sam3-lane--anima", group.elem_classes)

    def test_multi_feature_rules(self):
        with gr.Blocks():
            with gr.Group() as group:
                with gr.Accordion("Never OOM Integrated", open=False):
                    unet = gr.Checkbox(label="Enabled for UNet", value=False)
                    vae = gr.Checkbox(label="Enabled for VAE ", value=False)
            runner = _FakeRunner([_FakeScript("/x/scripts/forge_never_oom.py", group, [unet, vae])])
            ll.tag_runner(runner)
        self.assertIn("sam3-on", unet.elem_classes)
        self.assertIn("sam3-on", vae.elem_classes, "끝 공백이 있는 라벨도 잡는다")

    def test_elem_classes_may_arrive_as_none_or_str(self):
        with gr.Blocks():
            group, controls = _accordion_group("Model Keyword", enable_label="Model Keyword Enabled")
            group.elem_classes = "gradio-group"          # Gradio 는 문자열로도 들어온다
            controls[0].elem_classes = None
            runner = _FakeRunner([_FakeScript("/x/scripts/model_keyword.py", group, controls)])
            ll.tag_runner(runner)
        self.assertIn("gradio-group", group.elem_classes)
        self.assertIn("sam3-slot--model-keyword", group.elem_classes)
        self.assertIn("sam3-on", controls[0].elem_classes)

    def test_a_duplicate_script_stays_in_the_same_lane(self):
        """확장이 자기 UI 를 두 번 만들면(예: LoRA Block Weight) 두 번째도 같은 묶음에 둔다 — 키만 다르게."""
        with gr.Blocks():
            first, first_controls = _accordion_group("로라 블록 웨이트", input_accordion=True)
            second, second_controls = _accordion_group("로라 블록 웨이트", input_accordion=True)
            runner = _FakeRunner([
                _FakeScript("/x/scripts/lora_block_weight.py", first, first_controls),
                _FakeScript("/x/scripts/lora_block_weight.py", second, second_controls),
            ])
            self.assertEqual(ll.tag_runner(runner), 2)
        self.assertIn("sam3-slot--lora-block-weight", first.elem_classes)
        self.assertIn("sam3-lane--lora", first.elem_classes)
        self.assertIn("sam3-lane--lora", second.elem_classes, "두 번째도 같은 묶음")
        keys = [c for c in second.elem_classes if c.startswith("sam3-slot--")]
        self.assertEqual(len(keys), 1)
        self.assertNotEqual(keys[0], "sam3-slot--lora-block-weight", "핀·칩이 섞이지 않게 키는 다르다")

    def test_selectable_scripts_get_only_the_panel_class(self):
        """드롭다운으로 고르는 스크립트(X/Y/Z 등)는 2열 '스크립트' 자리에 놓이게만 표시한다."""
        with gr.Blocks():
            with gr.Group() as panel:
                gr.Checkbox(label="Enable", value=False)
            runner = _FakeRunner([])
            runner.selectable_scripts = [_FakeScript("/x/scripts/xyz_grid.py", panel, [])]
            self.assertEqual(ll.tag_runner(runner), 0, "선택형은 칸 수에 세지 않는다")
        self.assertIn("sam3-script-panel", panel.elem_classes)
        self.assertNotIn("sam3-slot", panel.elem_classes)


class ConfigTests(unittest.TestCase):
    def test_classes_reach_the_gradio_config(self):
        with gr.Blocks() as demo:
            group, controls = _accordion_group("SAM3 Mask", enable_label="Enable SAM3")
            runner = _FakeRunner([_FakeScript("/x/scripts/!sam3.py", group, controls)])
            ll.tag_runner(runner)
        config = demo.get_config_file()
        classes = [c["props"].get("elem_classes") for c in config["components"] if c["id"] == group._id]
        self.assertIn("sam3-slot--sam3", classes[0])


if __name__ == "__main__":
    unittest.main()
