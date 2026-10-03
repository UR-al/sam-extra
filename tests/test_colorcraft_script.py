"""Colorcraft script (scripts/colorcraft.py) — panel, argument layout, XYZ axes, settings, lanes.

The panel is built with the real Gradio; the WebUI ``modules`` is a stub. The panel is the shared editor
(v0.32.0): 67 script arguments in ``panel_state.ARG_NAMES`` order, js-only editor events.
"""

from __future__ import annotations

import collections
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
from sam3ext.colorcraft import hook, panel_state, spec, ui  # noqa: E402

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


def build_both(module):
    """txt2img and img2img in ONE Blocks, as Forge builds them: ``(demo, {tab: (script, controls)})``."""
    built = {}
    with gr.Blocks() as demo:
        for tab in ("txt2img", "img2img"):
            script = module.Colorcraft()
            script.is_img2img = tab == "img2img"
            built[tab] = (script, script.ui(script.is_img2img))
    return demo, built


def build_wired():
    """Both tabs through ``ui.build`` with ``parts`` (every named component) and a Debug callback, in ONE Blocks:
    ``(demo, {tab: (controls, fields, parts)})``."""
    built = {}
    with gr.Blocks() as demo:
        for tab in ("txt2img", "img2img"):
            parts = {}
            controls, fields = ui.build(lambda item, tab=tab: f"script_{tab}_colorcraft_samextra_{item}",
                                        on_debug_refresh=lambda *a: [], parts=parts)
            built[tab] = (controls, fields, parts)
    return demo, built


def _ids(comps):
    return [c._id for c in comps]


def _targets(dep):
    return [list(t) for t in dep["targets"]]


def _field_of(name):
    kind, _, field = name.partition(".")
    if not field:
        return {f.name: f for f in spec.GLOBAL_FIELDS + spec.DEBUG_FIELDS}.get(kind)
    return {"mod": spec.MODIFIER_BY_NAME, "leaf": spec.LEAF_BY_NAME, "combo": spec.COMBO_BY_NAME}[kind][field]


def _suffix(name):
    return name.replace(".", "_")


class ScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_script()
        cls.script, cls.controls = build_ui(cls.module)
        cls.demo, cls.both = build_both(cls.module)
        cls.wired_demo, cls.wired = build_wired()

    def test_title_and_visibility(self):
        script = self.module.Colorcraft()
        self.assertEqual(script.title(), "Colorcraft (sam-extra)")
        self.assertNotEqual(script.title(), "Colorcraft")       # upstream's and the fork's title
        self.assertIs(script.show(False), self.module.scripts.AlwaysVisible)
        self.assertIs(script.show(True), self.module.scripts.AlwaysVisible)
        self.assertEqual(script.sorting_priority, -23)

    def test_argument_count_and_order(self):
        self.assertEqual(len(self.controls), panel_state.ARG_COUNT)
        self.assertEqual(panel_state.ARG_COUNT, 67)
        for index, name in enumerate(panel_state.ARG_NAMES):
            elem = "script_txt2img_colorcraft_samextra_" + _suffix(name)
            self.assertEqual(self.controls[index].elem_id, elem, (index, name))
        # v0.31.0's positional layout stays the wrapped-list / direct-caller format
        self.assertEqual(spec.ARG_COUNT, 579)
        self.assertEqual(len(spec.arg_names()), spec.ARG_COUNT)
        self.assertEqual(spec.arg_names()[:6], ["enabled", "masking", "I.active", "I.kind", "I.pass", "I.mask"])
        self.assertEqual(spec.arg_names()[-3:], ["C5.normalize", "debug", "debug_step"])
        self.assertEqual((spec.MODIFIER_OFFSET, spec.LEAF_OFFSET, spec.COMBO_OFFSET, spec.DEBUG_OFFSET),
                         (2, 442, 542, 577))
        self.assertEqual((spec.N_MODIFIER, spec.N_LEAF, spec.N_COMBO), (44, 10, 7))

    def test_controls_carry_the_field_tables(self):
        for index, name in enumerate(panel_state.ARG_NAMES):
            control = self.controls[index]
            f = _field_of(name)
            if f is None:                                  # the hidden state and ref
                self.assertIsInstance(control, gr.Textbox, name)
                continue
            with self.subTest(name=name):
                self.assertEqual(control.label, f.label)
                if f.kind == spec.FLOAT:
                    self.assertIsInstance(control, gr.Slider)
                    self.assertEqual((control.minimum, control.maximum, control.step), (f.minimum, f.maximum, f.step))
                    self.assertEqual(control.value, f.default)
                elif f.kind == spec.BOOL:
                    self.assertIsInstance(control, gr.Checkbox)
                    expected = spec.modifier_default(0, f.name) if name.startswith("mod.") else f.default
                    self.assertEqual(control.value, expected)
                else:
                    self.assertIsInstance(control, (gr.Dropdown, gr.Radio))
                    self.assertEqual(control.value, f.default)
        names = list(panel_state.ARG_NAMES)
        self.assertIs(self.controls[names.index("mod.active")].value, True)      # modifier I's defaults
        self.assertIsInstance(self.controls[names.index("mod.pass")], gr.Radio)
        for side in ("mask_a", "mask_b"):
            combo = self.controls[names.index(f"combo.{side}")]
            self.assertEqual([c[0] if isinstance(c, tuple) else c for c in combo.choices],
                             spec.combo_ref_choices(spec.COMBO_COUNT))
        # nothing the panel creates goes to ui-config.json (Accordions, Groups, Rows and Columns stay as before)
        made = [b for b in self.demo.blocks.values() if isinstance(b, gr.components.Component)]
        self.assertGreater(len(made), 2 * 67)
        missing = [getattr(b, "elem_id", None) or type(b).__name__ for b in made
                   if not getattr(b, "do_not_save_to_config", False)]
        self.assertEqual(missing, [])

    def test_default_arguments_are_a_no_op(self):
        args = [c.value for c in self.controls]
        self.assertEqual(args, panel_state.default_args())
        config = panel_state.config_from_script_args(args)
        self.assertFalse(config.enabled)
        self.assertFalse(panel_state.args_enabled(args))
        config.enabled = True
        self.assertEqual(spec.build_chain(config, False).entries, [])

    def test_enable_checkbox_carries_sam3_on(self):
        enable = self.controls[0]
        self.assertEqual(enable.label, "Enable Colorcraft")
        self.assertIn("sam3-on", enable.elem_classes)
        self.assertIn("sam3-on--colorcraft", enable.elem_classes)
        others = [b for b in self.demo.blocks.values() if "sam3-on" in (getattr(b, "elem_classes", None) or [])]
        self.assertEqual([b.elem_id for b in others], ["script_txt2img_colorcraft_samextra_enabled",
                                                        "script_img2img_colorcraft_samextra_enabled"])

    def test_hidden_carriers(self):
        state, ref = self.controls[panel_state.STATE_INDEX], self.controls[panel_state.REF_INDEX]
        for control, value, label in ((state, "", "Colorcraft state"), (ref, "I|M1|0", "Colorcraft editor")):
            self.assertIsInstance(control, gr.Textbox)
            self.assertIs(control.visible, False)
            self.assertEqual((control.value, control.label), (value, label))

    def test_infotext_fields(self):
        fields = self.script.infotext_fields
        self.assertEqual(len(fields), 65)
        by_id = {c.elem_id: c for c, _ in fields}
        pre = "script_txt2img_colorcraft_samextra_"
        expected = [pre + s for s in ("enabled", "masking", "state", "ref", "modifier_select", "mask_select",
                                      "summary", "leaf_group", "combo_group", "mod_reset", "mask_reset")]
        expected += [pre + "mod_" + n for n in panel_state.MOD_NAMES]
        expected += [pre + "leaf_" + n for n in panel_state.LEAF_NAMES]
        self.assertEqual([c.elem_id for c, _ in fields], expected)
        self.assertEqual(len(by_id), 65)
        # the debug capture, debug step and the combo editor are never pasted
        controls = set(map(id, self.controls))
        pasted = {id(c) for c, _ in fields}
        names = [n for n, c in zip(panel_state.ARG_NAMES, self.controls) if id(c) in controls - pasted]
        self.assertEqual(names, ["debug", "debug_step"] + [f"combo.{n}" for n in panel_state.COMBO_NAMES])

        params = {spec.INFOTEXT_KEY: "v1;mods=II;II.kind=Punch;II.clarity=0.4"}
        values = {component.elem_id: getter(params) for component, getter in fields}
        self.assertIs(values[pre + "enabled"], True)
        rev = panel_state.parse_state(values[pre + "state"])[1]
        self.assertEqual(values[pre + "ref"], f"II|M1|{rev}")
        self.assertEqual(values[pre + "modifier_select"]["value"], "II")
        self.assertEqual(values[pre + "mask_select"]["value"], "M1")
        self.assertEqual(values[pre + "mod_kind"], "Punch")
        self.assertEqual(values[pre + "mod_clarity"], 0.4)
        self.assertIs(values[pre + "mod_active"], True)
        self.assertEqual(values[pre + "mod_reset"]["value"], "Reset II")
        args = [values.get(c.elem_id) for c in self.controls]
        args[3:5] = [False, 5]                                          # Debug values: not pasted
        args[panel_state.EDITOR_OFFSET + 54:] = [f.default for f in spec.COMBO_FIELDS]
        got = panel_state.config_from_script_args(args)
        want = spec.decode(params)
        self.assertEqual(spec.to_infotext(got), spec.to_infotext(want))
        self.assertEqual(spec.config_to_args(got)[:-2], spec.config_to_args(want)[:-2])
        # an infotext without the key: Enable off, the rest left alone
        self.assertEqual([getter({"Steps": "20"}) for _, getter in fields], [False] + [None] * 64)

    def test_paste_writes_the_view_of_the_pasted_look(self):
        """The paste's selectors, groups, Reset labels and summary (one Python response, no js involved)."""
        _, fields, parts = self.wired["txt2img"]
        params = {spec.INFOTEXT_KEY: "v1;mods=II;masking=1;masks=M3;combos=;II.kind=Luma;II.mask=M3;"
                                     "II.exposure=0.2;M3.blur=4"}
        by = {id(c): getter(params) for c, getter in fields}
        config = spec.decode(params)
        mods, masks, html = panel_state.overview(config)
        self.assertEqual(by[id(parts["mod_select"])], {"__type__": "update", "choices": mods, "value": "II"})
        self.assertEqual(by[id(parts["mask_select"])], {"__type__": "update", "choices": masks, "value": "M3"})
        self.assertEqual(by[id(parts["summary"])], html)
        self.assertIn("II Luma ➜ M3", html)
        self.assertEqual(by[id(parts["leaf_group"])], {"__type__": "update", "visible": True})
        self.assertEqual(by[id(parts["combo_group"])], {"__type__": "update", "visible": False})
        self.assertEqual(by[id(parts["mod_reset"])], {"__type__": "update", "value": "Reset II"})
        self.assertEqual(by[id(parts["mask_reset"])], {"__type__": "update", "value": "Reset M3"})
        self.assertEqual(by[id(parts["leaf"]["blur"])], 4.0)
        self.assertEqual(by[id(parts["mod"]["exposure"])], 0.2)

    def test_first_paint(self):
        """What the page shows before any event: I and M1 in the editors, the leaf editor, the default labels."""
        for tab, (controls, _, parts) in self.wired.items():
            with self.subTest(tab=tab):
                mods, masks, html = panel_state.overview(spec.default_config())
                self.assertEqual((parts["mod_select"].value, parts["mask_select"].value), ("I", "M1"))
                self.assertEqual((parts["mod_select"].choices, parts["mask_select"].choices), (mods, masks))
                self.assertEqual((parts["mod_select"].info, parts["mask_select"].info), (ui.MOD_MARKS, ui.MASK_MARKS))
                self.assertEqual(parts["summary"].value, html)
                self.assertEqual((parts["leaf_group"].visible, parts["combo_group"].visible), (True, False))
                self.assertEqual((parts["mod_reset"].value, parts["mask_reset"].value), ("Reset I", "Reset M1"))
                self.assertEqual(controls[panel_state.REF_INDEX].value, "I|M1|0")

    def test_section_follows_the_layout_setting(self):
        script = self.module.Colorcraft()
        script.is_img2img = False
        with mock.patch.object(layout_lanes, "sections_enabled", return_value=True):
            self.assertEqual(script.section, layout_lanes.ANIMA_SECTION)
        script.is_img2img = True
        self.assertIsNone(script.section)

    def test_kind_tables_and_groups(self):
        """The Type visibility runs in the browser (tests/js/colorcraft_editor.test.mjs) from spec's tables."""
        schema = panel_state.schema()
        self.assertEqual(schema["kindGroups"], {k: list(v) for k, v in spec.KIND_GROUPS.items()})
        self.assertEqual(schema["kindGates"], {k: list(v) for k, v in spec.KIND_GATES.items()})
        self.assertEqual(schema["kindTakesMask"], dict(spec.KIND_TAKES_MASK))
        self.assertEqual(ui.GROUP_ORDER, ("luma", "contrast", "detail", "chroma", "chroma_plus", "color_shift", "dev"))
        parts = {}
        with gr.Blocks() as demo:
            ui.build(lambda item: f"script_txt2img_colorcraft_samextra_{item}", parts=parts)
        self.assertEqual(tuple(parts["groups"]), ui.GROUP_ORDER)
        groups_of = {g: {f.name for f in spec.MODIFIER_FIELDS if f.group == g} for g in ui.GROUP_ORDER}
        for group, block in parts["groups"].items():
            inside = {name for name, comp in parts["mod"].items() if _contains(block, comp)}
            self.assertEqual(inside, groups_of[group], group)
        kind_dep = next(d for d in demo.get_config_file()["dependencies"] if ".kind(" in (d.get("js") or ""))
        expected = [parts["groups"][g]._id for g in ui.GROUP_ORDER]
        expected += [parts["mod"][n]._id for n in ("more_colors", "color_shift", "mask")]
        self.assertEqual(kind_dep["outputs"], expected)
        self.assertEqual(kind_dep["inputs"], [parts["mod"]["kind"]._id])

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
        debug, step = self.controls[3], self.controls[4]
        self.assertEqual(panel_state.ARG_NAMES[3:5], ("debug", "debug_step"))
        self.assertEqual((debug.label, debug.value), ("Capture debug latent", False))
        self.assertEqual((step.label, step.minimum, step.maximum, step.step, step.value), ("Debug Step", 0, 50, 1, 5))
        params = {spec.INFOTEXT_KEY: "v1;mods=I;I.exposure=0.1"}
        self.assertIsNone(spec.paste_value(params, "debug"))      # a tool: pasting leaves it alone
        self.assertIsNone(spec.paste_value(params, "debug_step"))
        self.assertNotIn(id(debug), {id(c) for c, _ in self.script.infotext_fields})
        self.assertNotIn(id(step), {id(c) for c, _ in self.script.infotext_fields})
        args = panel_state.default_args()
        args[3:5] = [True, 12]
        config = panel_state.config_from_script_args(args)
        self.assertEqual((config.debug, config.debug_step), (True, 12.0))
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
        self.assertEqual(len(img2img_controls), 67)
        self.assertEqual(len(self.controls), 67)
        self.assertTrue(img2img_controls[0].elem_id.startswith("script_img2img_"))
        self.assertTrue(self.controls[0].elem_id.startswith("script_txt2img_"))
        for tab, (_, controls) in self.both.items():
            self.assertEqual(len(controls), 67, tab)

    def test_no_duplicate_elem_ids(self):
        ids = [getattr(b, "elem_id", None) for b in self.demo.blocks.values()]
        counts = collections.Counter(i for i in ids if i)
        self.assertEqual([i for i, n in counts.items() if n > 1], [])
        pre = "script_txt2img_colorcraft_samextra_"
        for renamed in ("masking_panel", "debug_panel", "masking", "debug"):
            self.assertEqual(counts[pre + renamed], 1, renamed)

    def test_js_wiring_is_exact(self):
        """colorcraft_editor.js reads its inputs and writes its outputs by position: every component of every
        js-only event, and its triggers, in order. The fallback's ref index and output count are in the js."""
        deps = self.wired_demo.get_config_file()["dependencies"]
        for tab, (controls, _, p) in self.wired.items():
            editors = controls[panel_state.EDITOR_OFFSET:]
            head = [p["enabled"], p["masking"], p["state"], p["ref"]]
            view = [p["mod_select"], p["mask_select"], p["summary"]]
            mine = [d for d in deps if d.get("js") and f"('{tab}'" in d["js"]]
            sync, reset_mod, reset_mask, labels, kind = mine
            with self.subTest(tab=tab, dep="sync"):
                self.assertEqual(_targets(sync), [[p["mod_select"]._id, "change"], [p["mask_select"]._id, "change"],
                                                   [p["state"]._id, "change"]])
                self.assertEqual(sync["inputs"], _ids([p["mod_select"], p["mask_select"]] + head + editors))
                self.assertEqual(sync["outputs"], _ids([p["state"], p["ref"]] + editors + [
                    p["leaf_group"], p["combo_group"], p["mod_reset"], p["mask_reset"]] + view))
                self.assertIn(f"window.samextraColorcraft.sync('{tab}', a)", sync["js"])
                self.assertIn("String(a[5])", sync["js"])                    # ref
                self.assertIn("Array(67)", sync["js"])                       # 70 outputs, the last 3 are the view
            for dep, section, button in ((reset_mod, "mod", p["mod_reset"]), (reset_mask, "mask", p["mask_reset"])):
                with self.subTest(tab=tab, dep=f"reset {section}"):
                    self.assertEqual(_targets(dep), [[button._id, "click"]])
                    self.assertEqual(dep["inputs"], _ids(head + editors))
                    self.assertEqual(dep["outputs"], _ids([p["state"], p["ref"]] + editors + view))
                    self.assertIn(f"window.samextraColorcraft.reset('{tab}', '{section}', a)", dep["js"])
                    self.assertIn("String(a[3])", dep["js"])
                    self.assertIn("Array(63)", dep["js"])
            with self.subTest(tab=tab, dep="labels"):
                self.assertEqual(_targets(labels), [[c._id, "input"] for c in editors]
                                 + [[p["enabled"]._id, "change"], [p["masking"]._id, "change"]])
                self.assertEqual(labels["inputs"], _ids(head + editors))
                self.assertEqual(labels["outputs"], _ids(view))
                self.assertIn(f"window.samextraColorcraft.labels('{tab}', a)", labels["js"])
                self.assertIn("String(a[3])", labels["js"])
                self.assertIn("Array(0)", labels["js"])
            with self.subTest(tab=tab, dep="kind"):
                self.assertEqual(kind["js"], f"(...a) => window.samextraColorcraft ? "
                                             f"window.samextraColorcraft.kind('{tab}', a) : undefined")
                self.assertEqual(kind["inputs"], [p["mod"]["kind"]._id])
                self.assertEqual(kind["outputs"], _ids([p["groups"][g] for g in ui.GROUP_ORDER]
                                                      + [p["mod"][n] for n in ("more_colors", "color_shift", "mask")]))
            refresh = [d for d in deps if d.get("backend_fn") and _targets(d) == [[p["refresh"]._id, "click"]]]
            with self.subTest(tab=tab, dep="debug refresh"):
                self.assertEqual(len(refresh), 1)
                self.assertEqual(refresh[0]["inputs"][6:], _ids(head + editors))
                self.assertEqual(refresh[0]["outputs"], [p["gallery"]._id])

    def test_a11y_anchors_match_the_editor_js(self):
        """colorcraft_editor.js names the selector (and Pass) radio groups, describes every editor control by these
        elem ids and mounts its hidden description/status node in the summary block; style.css hides that node.
        tests/js/colorcraft_editor_a11y.test.mjs builds its Gradio-shaped panel from the same labels."""
        source = (ROOT / "javascript" / "colorcraft_editor.js").read_text(encoding="utf-8")
        a11y_test = (ROOT / "tests" / "js" / "colorcraft_editor_a11y.test.mjs").read_text(encoding="utf-8")
        self.assertIn("const prefix = (tab) => `script_${tab}_colorcraft_samextra_`;", source)
        self.assertIn('for (const name of ["modifier_select", "mask_select", "mod_pass"])', source)
        self.assertIn("doc.getElementById(`${pre}summary`)", source)
        for kind in ("mod", "leaf", "combo"):
            self.assertIn(f"doc.getElementById(`${{pre}}{kind}_${{f.name}}`)", source)
        self.assertIn(f'const MOD_MARKS = "{ui.MOD_MARKS}";', a11y_test)
        self.assertIn(f'const MASK_MARKS = "{ui.MASK_MARKS}";', a11y_test)
        for tab, (_, _, p) in self.wired.items():
            pre = f"script_{tab}_colorcraft_samextra_"
            with self.subTest(tab=tab):
                self.assertIsInstance(p["summary"], gr.HTML)
                self.assertEqual(p["summary"].elem_id, pre + "summary")
                for key, elem, label, info in (("mod_select", "modifier_select", "수정자", ui.MOD_MARKS),
                                               ("mask_select", "mask_select", "마스크 · 조합", ui.MASK_MARKS)):
                    radio = p[key]
                    self.assertIsInstance(radio, gr.Radio)
                    self.assertEqual((radio.elem_id, radio.label, radio.info), (pre + elem, label, info))
                self.assertIsInstance(p["mod"]["pass"], gr.Radio)
                for kind, names in (("mod", panel_state.MOD_NAMES), ("leaf", panel_state.LEAF_NAMES),
                                    ("combo", panel_state.COMBO_NAMES)):
                    for name in names:
                        self.assertEqual(p[kind][name].elem_id, f"{pre}{kind}_{name}")
        css = (ROOT / "style.css").read_text(encoding="utf-8")
        begin, end = "/* samextra-colorcraft:begin */", "/* samextra-colorcraft:end */"
        self.assertIn(begin, css)
        block = css[css.index(begin):css.index(end)]
        self.assertIn('const SR_CLASS = "samextra-cc-sr";', source)
        self.assertIn(".samextra-cc-sr {", block)
        self.assertNotRegex(block, r"(?m)^\s*(display|visibility)\s*:")    # a live region must stay rendered

    def test_js_dependencies(self):
        deps = self.demo.get_config_file()["dependencies"]
        js_only = [d for d in deps if not d.get("backend_fn")]
        python = [d for d in deps if d.get("backend_fn")]
        self.assertEqual(len(js_only), 10)
        self.assertEqual(len(python), 2)                          # Debug Refresh, one per tab
        rows = []
        for tab in ("txt2img", "img2img"):
            mine = [d for d in js_only if f"('{tab}'" in d["js"]]
            self.assertEqual(len(mine), 5, tab)
            for d in mine:
                self.assertFalse(d["queue"])
                self.assertEqual(d["show_progress"], "hidden")
                self.assertFalse(d["show_api"])
                rows.append((tab, d["js"], len(d["inputs"]), len(d["outputs"]), len(d["targets"])))
        expected = []
        for tab in ("txt2img", "img2img"):
            expected += [
                (tab, ui._js(tab, "sync", ref_at=5, n_out=70), 67, 70, 3),
                (tab, ui._js(tab, "reset", "mod", ref_at=3, n_out=66), 65, 66, 1),
                (tab, ui._js(tab, "reset", "mask", ref_at=3, n_out=66), 65, 66, 1),
                (tab, ui._js(tab, "labels", ref_at=3, n_out=3), 65, 3, 63),
                (tab, ui._js(tab, "kind"), 1, 10, 1),
            ]
        self.assertEqual(rows, expected)
        for d in python:
            self.assertEqual((len(d["inputs"]), len(d["outputs"])), (71, 1))


def _contains(block, target):
    stack = [block]
    while stack:
        node = stack.pop()
        if node is target:
            return True
        stack.extend(getattr(node, "children", []) or [])
    return False


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
