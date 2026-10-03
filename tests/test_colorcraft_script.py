"""Colorcraft script (scripts/colorcraft.py) — panel, argument layout, XYZ axes, settings, lanes.

The panel is built with the real Gradio; the WebUI ``modules`` is a stub.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from sam3ext import layout_lanes  # noqa: E402
from sam3ext.colorcraft import hook, spec, ui  # noqa: E402

SCRIPT = ROOT / "scripts" / "colorcraft.py"


def _stub_modules(registered=None, xyz=None):
    registered = registered if registered is not None else {}
    modules = types.ModuleType("modules")
    modules.__path__ = []

    class Script:
        pass

    data = []
    if xyz is not None:
        data.append(types.SimpleNamespace(script_class=type("XYZ", (), {"__module__": "xyz_grid.py"}), module=xyz))
    modules.scripts = types.SimpleNamespace(Script=Script, AlwaysVisible=object(), scripts_data=data)
    modules.script_callbacks = types.SimpleNamespace(
        on_before_ui=lambda fn: registered.setdefault("before_ui", []).append(fn),
        on_ui_settings=lambda fn: registered.setdefault("ui_settings", []).append(fn),
    )
    return modules


def load_script(registered=None, xyz=None):
    modules = _stub_modules(registered, xyz)
    with mock.patch.dict(sys.modules, {"modules": modules, "modules.scripts": modules.scripts,
                                       "modules.script_callbacks": modules.script_callbacks}):
        spec_ = importlib.util.spec_from_file_location("_test_colorcraft_script", SCRIPT)
        module = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(module)
    return module


def build_ui(module, is_img2img=False):
    script = module.Colorcraft()
    script.is_img2img = is_img2img
    with gr.Blocks():
        controls = script.ui(is_img2img)
    return script, controls


class ScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_script()
        cls.script, cls.controls = build_ui(cls.module)

    def test_title_and_visibility(self):
        script = self.module.Colorcraft()
        self.assertEqual(script.title(), "Colorcraft (sam-extra)")
        self.assertNotEqual(script.title(), "Colorcraft")       # upstream's and the fork's title
        self.assertIs(script.show(False), self.module.scripts.AlwaysVisible)
        self.assertIs(script.show(True), self.module.scripts.AlwaysVisible)
        self.assertEqual(script.sorting_priority, -23)

    def test_argument_count_and_order(self):
        self.assertEqual(spec.ARG_COUNT, 579)
        self.assertEqual(len(self.controls), spec.ARG_COUNT)
        self.assertEqual(len(spec.arg_names()), spec.ARG_COUNT)
        self.assertEqual(spec.arg_names()[:6], ["enabled", "masking", "I.active", "I.kind", "I.pass", "I.mask"])
        self.assertEqual(spec.arg_names()[-3:], ["C5.normalize", "debug", "debug_step"])
        for index, path in enumerate(spec.arg_names()):
            elem = path.replace(".", "_")
            self.assertTrue(self.controls[index].elem_id.endswith("_" + elem), (index, path, self.controls[index].elem_id))
        # offsets used by the hook
        self.assertEqual((spec.MODIFIER_OFFSET, spec.LEAF_OFFSET, spec.COMBO_OFFSET, spec.DEBUG_OFFSET),
                         (2, 442, 542, 577))
        self.assertEqual((spec.N_MODIFIER, spec.N_LEAF, spec.N_COMBO), (44, 10, 7))

    def test_controls_carry_the_field_tables(self):
        names = spec.arg_names()
        for index, path in enumerate(names):
            tag, _, name = path.partition(".")
            if not name:
                f = {f.name: f for f in spec.GLOBAL_FIELDS + spec.DEBUG_FIELDS}[tag]
            elif tag in spec.MODIFIER_TAGS:
                f = spec.MODIFIER_BY_NAME[name]
            elif tag in spec.MASK_TAGS:
                f = spec.LEAF_BY_NAME[name]
            else:
                f = spec.COMBO_BY_NAME[name]
            control = self.controls[index]
            with self.subTest(path=path):
                self.assertEqual(control.label, f.label)
                self.assertTrue(getattr(control, "do_not_save_to_config", False))
                if f.kind == spec.FLOAT:
                    self.assertIsInstance(control, gr.Slider)
                    self.assertEqual((control.minimum, control.maximum, control.step), (f.minimum, f.maximum, f.step))
                    self.assertEqual(control.value, f.default)
                elif f.kind == spec.BOOL:
                    self.assertIsInstance(control, gr.Checkbox)
                    expected = spec.modifier_default(spec.MODIFIER_TAGS.index(tag), name) if tag in spec.MODIFIER_TAGS else f.default
                    self.assertEqual(control.value, expected)
                else:
                    self.assertIsInstance(control, (gr.Dropdown, gr.Radio))
                    self.assertEqual(control.value, f.default)
        self.assertEqual(self.controls[names.index("I.active")].value, True)
        self.assertEqual(self.controls[names.index("II.active")].value, False)
        self.assertIsInstance(self.controls[names.index("I.pass")], gr.Radio)
        combo_b = self.controls[names.index("C3.mask_b")]
        self.assertEqual([c[0] if isinstance(c, tuple) else c for c in combo_b.choices], spec.combo_ref_choices(2))

    def test_default_arguments_are_a_no_op(self):
        config = spec.config_from_args([c.value for c in self.controls])
        self.assertFalse(config.enabled)
        config.enabled = True
        self.assertEqual(spec.build_chain(config, False).entries, [])

    def test_enable_checkbox_carries_sam3_on(self):
        enable = self.controls[0]
        self.assertEqual(enable.label, "Enable Colorcraft")
        self.assertIn("sam3-on", enable.elem_classes)
        self.assertIn("sam3-on--colorcraft", enable.elem_classes)
        others = [c for c in self.controls[1:] if "sam3-on" in (getattr(c, "elem_classes", None) or [])]
        self.assertEqual(others, [])

    def test_infotext_fields_cover_every_argument(self):
        fields = self.script.infotext_fields
        self.assertEqual(len(fields), spec.ARG_COUNT)
        self.assertEqual([component for component, _ in fields], list(self.controls))
        params = {spec.INFOTEXT_KEY: "v1;mods=II;II.kind=Punch;II.clarity=0.4"}
        values = {component.elem_id: getter(params) for component, getter in fields}
        self.assertTrue(values[self.controls[0].elem_id])
        names = spec.arg_names()
        self.assertEqual(values[self.controls[names.index("II.kind")].elem_id], "Punch")
        self.assertEqual(values[self.controls[names.index("II.clarity")].elem_id], 0.4)
        self.assertIs(values[self.controls[names.index("I.active")].elem_id], False)

    def test_section_follows_the_layout_setting(self):
        script = self.module.Colorcraft()
        script.is_img2img = False
        with mock.patch.object(layout_lanes, "sections_enabled", return_value=True):
            self.assertEqual(script.section, layout_lanes.ANIMA_SECTION)
        script.is_img2img = True
        self.assertIsNone(script.section)

    def test_kind_shows_only_its_nodes_controls(self):
        def visible(label):
            updates = ui._kind_updates(gr, label)
            groups = dict(zip(ui._GROUP_ORDER, [u["visible"] for u in updates[:len(ui._GROUP_ORDER)]]))
            gates = dict(zip(("more_colors", "color_shift"), [u["visible"] for u in updates[len(ui._GROUP_ORDER):-1]]))
            return groups, gates, updates[-1]["visible"]

        groups, gates, mask = visible("Advanced")
        self.assertTrue(all(groups.values()))
        self.assertTrue(all(gates.values()))
        self.assertTrue(mask)
        groups, gates, mask = visible("Basic")
        self.assertEqual({g for g, shown in groups.items() if shown}, {"contrast", "color_shift"})
        self.assertFalse(any(gates.values()))
        self.assertFalse(mask)
        for label, shown_groups in (("Luma", {"luma"}), ("Chroma", {"chroma"}), ("Chroma Plus", {"chroma_plus"}),
                                    ("Punch", {"contrast", "detail"}), ("Shift", {"color_shift"})):
            groups, gates, mask = visible(label)
            with self.subTest(kind=label):
                self.assertEqual({g for g, shown in groups.items() if shown}, shown_groups)
                self.assertFalse(any(gates.values()))
                self.assertTrue(mask)

    def test_every_group_holds_exactly_its_fields(self):
        """The panel groups the Types show are the field groups the chain builder reads."""
        for kind, groups in spec.KIND_GROUPS.items():
            panel = {f.name for f in spec.MODIFIER_FIELDS if f.group in groups}
            params = {panel_name for _, panel_name in spec._KIND_PARAMS[kind]}
            with self.subTest(kind=kind):
                self.assertTrue(params <= panel, params - panel)
                gates = set(spec.KIND_GATES.get(kind, ()))
                # the two Advanced-node gate checkboxes sit in their groups but are hidden for other Types
                hidden_gates = {"more_colors", "color_shift"} - gates
                self.assertEqual(panel - params - gates - hidden_gates, set())

    def test_debug_panel(self):
        names = spec.arg_names()
        debug, step = self.controls[names.index("debug")], self.controls[names.index("debug_step")]
        self.assertEqual((debug.label, debug.value), ("Capture debug latent", False))
        self.assertEqual((step.label, step.minimum, step.maximum, step.step, step.value), ("Debug Step", 0, 50, 1, 5))
        params = {spec.INFOTEXT_KEY: "v1;mods=I;I.exposure=0.1"}
        self.assertIsNone(spec.paste_value(params, "debug"))      # a tool: pasting leaves it alone
        self.assertIsNone(spec.paste_value(params, "debug_step"))
        config = spec.config_from_args(spec.config_to_args(spec.default_config())[:-2] + [True, 12])
        self.assertEqual((config.debug, config.debug_step), (True, 12.0))

    def test_postprocess_keeps_the_capture_for_the_refresh_button(self):
        script = self.module.Colorcraft()
        self.assertIsNone(script.debug_capture)
        script.postprocess(types.SimpleNamespace(), None)
        self.assertIsNone(script.debug_capture)
        marker = object()
        script.postprocess(types.SimpleNamespace(**{hook.STATE_ATTR: {"debug": marker}}), None)
        self.assertIs(script.debug_capture, marker)
        script.postprocess(types.SimpleNamespace(**{hook.STATE_ATTR: {"passes": {}}}), None)
        self.assertIs(script.debug_capture, marker)                # a run without a capture keeps the last one
        with mock.patch.object(self.module.debug_panel, "render_for_panel", side_effect=RuntimeError("x")), \
                mock.patch.object(self.module, "_log"):
            self.assertEqual(script._on_debug_refresh([], [], [], "none", "white", "colormap"), [])

    def test_ui_builds_for_both_tabs(self):
        _, img2img_controls = build_ui(self.module, is_img2img=True)
        self.assertEqual(len(img2img_controls), spec.ARG_COUNT)
        self.assertTrue(img2img_controls[0].elem_id.startswith("script_img2img_"))
        self.assertTrue(self.controls[0].elem_id.startswith("script_txt2img_"))


class SliderScriptTests(unittest.TestCase):
    """javascript/colorcraft_sliders.js (upstream's unclamped number fields) finds this accordion."""

    def _forge_elem_id(self):
        forge = ROOT.parents[1] / "modules" / "scripts.py"
        if not forge.is_file():
            self.skipTest("the Forge checkout is not next to this extension")
        import ast
        import re
        tree = ast.parse(forge.read_text(encoding="utf-8"))
        script = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Script")
        method = next(n for n in script.body if isinstance(n, ast.FunctionDef) and n.name == "elem_id")
        scope = {"re": re}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(forge), "exec"), scope)  # noqa: S102
        return scope["elem_id"]

    def test_scope_selector_is_the_accordion_forge_builds(self):
        forge_elem_id = self._forge_elem_id()
        module = load_script()
        source = (ROOT / "javascript" / "colorcraft_sliders.js").read_text(encoding="utf-8")
        for is_img2img in (False, True):
            script = module.Colorcraft()
            script.is_img2img = is_img2img
            expected = forge_elem_id(script, "accordion")
            with self.subTest(img2img=is_img2img):
                self.assertIn("#" + expected, source)
                self.assertEqual(script._elem_id("accordion"), expected)   # the fallback without Forge


class LayoutLaneTests(unittest.TestCase):
    def test_registered_in_the_anima_lane(self):
        self.assertEqual(layout_lanes.slot_key("/x/extensions/forge_sam3_extension/scripts/colorcraft.py"), "colorcraft")
        self.assertEqual(layout_lanes.lane_for("colorcraft"), "anima")
        self.assertFalse(layout_lanes.REGISTRY["colorcraft"].exp)

    def test_tagging_finds_our_head_accordion_and_on_checkbox(self):
        module = load_script()
        script = module.Colorcraft()
        script.is_img2img = False
        with gr.Blocks():
            with gr.Group() as group:
                controls = script.ui(False)
        runner = types.SimpleNamespace(
            alwayson_scripts=[types.SimpleNamespace(filename=str(SCRIPT), group=group, controls=controls,
                                                    section=None, create_group=True)],
            selectable_scripts=[])
        self.assertEqual(layout_lanes.tag_runner(runner), 1)
        self.assertIn("sam3-slot--colorcraft", group.elem_classes)
        self.assertIn("sam3-lane--anima", group.elem_classes)
        self.assertNotIn("sam3-on-guess", group.elem_classes)
        head = layout_lanes._head_accordion(group, controls)
        self.assertEqual(head.label, "Colorcraft (latent 색 보정 · ComfyUI-Colorcraft)")
        self.assertIn("sam3-head", head.elem_classes)

    def test_frontend_short_label(self):
        source = (ROOT / "javascript" / "notebook_lanes.js").read_text(encoding="utf-8")
        self.assertIn('"colorcraft": "Colorcraft"', source)


class _AxisOption:
    def __init__(self, label, type, apply, **kwargs):
        self.label, self.type, self.apply, self.kwargs = label, type, apply, kwargs


class XyzTests(unittest.TestCase):
    def _xyz(self):
        return types.SimpleNamespace(AxisOption=_AxisOption,
                                     axis_options=[_AxisOption("Nothing", str, None), _AxisOption("Seed", int, None)])

    def test_axes(self):
        xyz = self._xyz()
        registered = {}
        load_script(registered, xyz)
        registered["before_ui"][0]()
        labels = [a.label for a in xyz.axis_options]
        ours = [label for label in labels if label.startswith("[Colorcraft] ")]
        self.assertEqual(ours, ["[Colorcraft] Enable", "[Colorcraft] Strength", "[Colorcraft] Start",
                                "[Colorcraft] End", "[Colorcraft] Exposure", "[Colorcraft] Tone Compression",
                                "[Colorcraft] Contrast", "[Colorcraft] Clarity", "[Colorcraft] Sharpness",
                                "[Colorcraft] Temperature", "[Colorcraft] Tint", "[Colorcraft] Vibrance",
                                "[Colorcraft] Saturation", "[Colorcraft] Chroma Contrast"])
        by_label = {a.label: a for a in xyz.axis_options}
        self.assertEqual(by_label["[Colorcraft] Enable"].kwargs["choices"](), ["True", "False"])
        self.assertIs(by_label["[Colorcraft] Exposure"].type, float)
        p = types.SimpleNamespace()
        by_label["[Colorcraft] Exposure"].apply(p, 0.3, [0.3])
        by_label["[Colorcraft] Enable"].apply(p, "True", ["True"])
        self.assertEqual(getattr(p, spec.XYZ_ATTR), {"exposure": 0.3, "enabled": "True"})
        fields = {label.split("] ", 1)[1] for label in ours}
        self.assertEqual(len(fields), len(spec.XYZ_MODIFIER_FIELDS) + 1)
        # xyz_grid gives every cell a shallow copy of p: a dict already on p is never written into
        shared = {"tint": 0.1}
        cell = types.SimpleNamespace(**{spec.XYZ_ATTR: shared})
        by_label["[Colorcraft] Tint"].apply(cell, 0.4, [0.4])
        self.assertEqual(shared, {"tint": 0.1})
        self.assertEqual(getattr(cell, spec.XYZ_ATTR), {"tint": 0.4})

    def test_reload_does_not_duplicate_and_keeps_indices(self):
        xyz = self._xyz()
        first = {}
        load_script(first, xyz)
        first["before_ui"][0]()
        xyz.axis_options.append(_AxisOption("[Other] Axis", str, None))     # an extension registered after us
        before = [a.label for a in xyz.axis_options]
        old_apply = xyz.axis_options[before.index("[Colorcraft] Exposure")].apply
        second = {}
        load_script(second, xyz)                                           # WebUI "Reload UI": a new module
        second["before_ui"][0]()
        after = [a.label for a in xyz.axis_options]
        self.assertEqual(after, before)
        new_apply = xyz.axis_options[after.index("[Colorcraft] Exposure")].apply
        self.assertIsNot(new_apply, old_apply)                             # points at the reloaded module
        p = types.SimpleNamespace()
        new_apply(p, 0.2, [0.2])
        self.assertEqual(getattr(p, spec.XYZ_ATTR), {"exposure": 0.2})

    def test_without_xyz_grid_nothing_happens(self):
        registered = {}
        load_script(registered)
        registered["before_ui"][0]()   # no xyz_grid among scripts_data: no error


class SettingsTests(unittest.TestCase):
    def test_options(self):
        registered = {}
        module = load_script(registered)
        added = {}

        class OptionInfo:
            def __init__(self, default, label, component, section=None, infotext=None, **kwargs):
                self.default, self.label, self.component = default, label, component
                self.section, self.infotext = section, infotext
                self.help = None

            def info(self, text):
                self.help = text
                return self

        shared = types.SimpleNamespace(opts=types.SimpleNamespace(add_option=lambda key, info: added.__setitem__(key, info)),
                                       OptionInfo=OptionInfo)
        with mock.patch.dict(sys.modules, {"modules": types.SimpleNamespace(shared=shared),
                                           "modules.shared": shared}):
            registered["ui_settings"][0]()
        self.assertEqual(set(added), {hook.OPT_PRE_DD, hook.OPT_LOG})
        self.assertEqual(added[hook.OPT_PRE_DD].default, True)
        self.assertEqual(added[hook.OPT_PRE_DD].infotext, "SAM Extra Colorcraft pre-DD sigma")
        self.assertEqual(added[hook.OPT_LOG].default, False)
        self.assertIsNone(added[hook.OPT_LOG].infotext)
        for info in added.values():
            self.assertEqual(info.section, ("sam3_colorcraft", "SAM Extra Colorcraft"))
            self.assertTrue(info.help)
        self.assertTrue(all(key.startswith("sam3_") for key in added))
        self.assertEqual(module.SETTINGS_SECTION, ("sam3_colorcraft", "SAM Extra Colorcraft"))


if __name__ == "__main__":
    unittest.main()
