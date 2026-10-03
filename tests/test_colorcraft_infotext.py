"""Colorcraft infotext — this extension's key, and pasting upstream's and the fork's ``Colorcraft`` key.

* Our ``SAM Extra Colorcraft`` value round-trips (also through Forge's own infotext quoting and
  parsing, read from the checkout) and restores exactly what each tab's Type runs.
* Upstream's Forge format ``MOD(...)MASK(...)COMBO(...)`` is produced with upstream's own writer
  functions (pinned copy) the way its ``process()`` writes it, and read here — as the string, and as
  the dict upstream's own paste callback turns it into when that extension is installed too.
* The fork's ``v1;name=value`` format is produced with the fork's own ``to_infotext`` (pinned copy).
* Numerically: the pasted settings render bit-identically to what the original settings meant —
  upstream's Forge tab (the Advanced node with Chroma Plus and Color Shift applied) and the fork's
  chain, both evaluated by upstream's node (the latest math).
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _support():
    spec_ = importlib.util.spec_from_file_location("_colorcraft_support", ROOT / "tests" / "_colorcraft_support.py")
    module = sys.modules.get(spec_.name)
    if module is None:
        module = importlib.util.module_from_spec(spec_)
        sys.modules[spec_.name] = module
        spec_.loader.exec_module(module)
    return module


S = _support()

from sam3ext.colorcraft import spec  # noqa: E402

UPSTREAM_NAMES = ("MODIFIER_FIELDS", "MODIFIER_VALUE_FIELDS", "LEAF_FIELDS", "COMBO_FIELDS", "BOOL_FIELDS",
                  "STR_FIELDS", "MODIFIER_COUNT", "MASK_COUNT", "COMBO_COUNT", "ROMAN", "write_entries",
                  "parse_entries", "_cast_field", "combine_infotext_sections", "extract_infotext_section",
                  "parse_infotext", "compute_mask_activeness")


def _upstream():
    return S.upstream_forge_functions(*UPSTREAM_NAMES)


def upstream_tab(**values):
    """One upstream Forge tab, every MODIFIER_FIELDS value (upstream's widget defaults + ``values``)."""
    tab = {"active": True, "mask": "none", "strength": 1.0, "start": 0.5, "end": 0.75, "advanced": False,
           "exponent": 0.0, "bias": 0.5, "start_off": 0.0, "end_off": 0.0, "smooth": True, "exposure": 0.0,
           "tone_compression": 0.0, "contrast": 0.0, "clarity": 0.0, "sharpness": 0.0, "temperature": 0.0,
           "tint": 0.0, "vibrance": 0.0, "saturation": 0.0, "chroma_contrast": 0.0, "chroma_center": 0.0,
           "temp_plus_tint": 0.0, "temp_minus_tint": 0.0, "lab_a": 0.0, "lab_b": 0.0, "lab_a_plus_b": 0.0,
           "lab_a_minus_b": 0.0, "color_shift_amount": 0.0, "color_shift_mode": "default",
           "color_shift_red": 0.0, "color_shift_green": 0.0, "color_shift_blue": 0.0,
           "color_shift_brightness": 0.0, "hires": False}
    tab.update(values)
    return tab


def upstream_leaf(**values):
    leaf = {"mask_axis": "exposure", "mask_mode": "highs", "mask_strength": 1.0, "mask_width": 0.0,
            "mask_center": 0.0, "mask_hardness": 1.0, "blur": 0.0, "spread": 0.0, "contrast": 0.0,
            "normalize": False}
    leaf.update(values)
    return leaf


def upstream_combo(**values):
    combo = {"mask_a": "none", "mask_b": "none", "operation": "and", "blur": 0.0, "spread": 0.0,
             "contrast": 0.0, "normalize": False}
    combo.update(values)
    return combo


def upstream_infotext(up, tabs, leaves=None, combos=None, masking_enabled=True):
    """What upstream's ``Script.process`` writes — its own writer functions, its process() logic.

    origin: muerrilla/ComfyUI-Colorcraft@d28ac6a:scripts/colorcraft.py:705-763 (reproduced step by step)."""
    tabs = list(tabs) + [upstream_tab(active=False)] * (up["MODIFIER_COUNT"] - len(tabs))
    leaves = (leaves or {})
    combos = (combos or {})
    leaf_list = [leaves.get(f"M{i + 1}", upstream_leaf()) for i in range(up["MASK_COUNT"])]
    combo_list = [combos.get(f"C{i + 1}", upstream_combo()) for i in range(up["COMBO_COUNT"])]
    mod_mask_selections = [tab["mask"] for tab in tabs]
    modifier_active_list = [tab["active"] for tab in tabs]
    leaf_active, combo_active = up["compute_mask_activeness"](
        mod_mask_selections, [c["mask_a"] for c in combo_list], [c["mask_b"] for c in combo_list],
        modifier_active_list)
    mod_entries = []
    for i, d in enumerate(tabs):
        d = dict(d)
        if not d["active"]:
            continue
        if not d["advanced"]:
            d["bias"], d["exponent"], d["start_off"], d["end_off"] = 0.5, 0.0, 0.0, 0.0
        mod_entries.append((up["ROMAN"][i], {**d, "active_flag": True}))
    leaf_entries = [(f"M{i + 1}", leaf_list[i]) for i in range(up["MASK_COUNT"]) if leaf_active[i]]
    combo_entries = [(f"C{i + 1}", combo_list[i]) for i in range(up["COMBO_COUNT"]) if combo_active[i]]
    mod_str = up["write_entries"](mod_entries, up["MODIFIER_VALUE_FIELDS"])
    if masking_enabled:
        mask_str = up["write_entries"](leaf_entries, up["LEAF_FIELDS"]) if leaf_entries else ""
        combo_str = up["write_entries"](combo_entries, up["COMBO_FIELDS"]) if combo_entries else ""
    else:
        mask_str, combo_str = "", ""
    return up["combine_infotext_sections"](mod_str, mask_str, combo_str)


def _forge_infotext_tools():
    """Forge's own ``quote``/``unquote`` and the infotext key/value regex (modules/infotext_utils.py)."""
    path = S.FORGE / "modules" / "infotext_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scope = {"json": __import__("json"), "re": re}
    keep = [node for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in ("quote", "unquote"))
            or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in ("re_param_code", "re_param")
                                                     for t in node.targets))]
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
    return scope


def _through_forge(tools, params):
    """``create_infotext``'s last line, then ``parse_generation_parameters``' key/value loop."""
    line = ", ".join(f"{k}: {tools['quote'](v)}" for k, v in params.items())
    return {k: tools["unquote"](v) for k, v in tools["re_param"].findall(line)}


def _chain_view(config, is_hires):
    """What a pass runs: the chain entries, comparable across configs."""
    chain = spec.build_chain(config, is_hires)
    return chain.tags, chain.entries


class OwnFormatTests(unittest.TestCase):
    def test_round_trip_restores_what_every_scenario_runs(self):
        for scenario in S.SCENARIOS:
            config = S.scenario_config(scenario)
            text = spec.to_infotext(config)
            restored = spec.parse_own(text)
            with self.subTest(scenario=scenario.label):
                self.assertTrue(text.startswith("v1;mods="))
                for hires in (False, True):
                    self.assertEqual(_chain_view(restored, hires), _chain_view(config, hires))
                self.assertEqual(spec.to_infotext(restored), text)

    def test_only_the_types_controls_are_written(self):
        config = spec.default_config()
        config.enabled = True
        tab = config.modifiers[0]
        tab.update(kind="Luma", exposure=0.3, contrast=0.9, lab_a=0.5, mask="M1")
        # masking off: the selection is kept (it is the tab's state), M1 itself is not in use
        self.assertEqual(spec.to_infotext(config), "v1;mods=I;I.kind=Luma;I.mask=M1;I.exposure=0.3")
        config.masking = True
        self.assertEqual(spec.to_infotext(config),
                         "v1;mods=I;masking=1;masks=M1;combos=;I.kind=Luma;I.mask=M1;I.exposure=0.3")
        tab.update(kind="Basic")                                    # Basic takes no mask
        self.assertEqual(spec.to_infotext(config), "v1;mods=I;masking=1;masks=;combos=;I.kind=Basic;I.contrast=0.9")
        tab.update(kind="Advanced", more_colors=False, color_shift=False, color_shift_amount=0.4)
        text = spec.to_infotext(config)
        for part in ("I.more_colors=0", "I.lab_a=0.5", "I.color_shift=0", "I.color_shift_amount=0.4",
                     "I.contrast=0.9", "I.exposure=0.3"):
            self.assertIn(part, text)
        self.assertNotIn("I.kind=", text)                           # Advanced is the default type

    def test_masks_written_are_the_ones_in_use(self):
        config = spec.default_config()
        config.enabled, config.masking = True, True
        config.modifiers[0].update(exposure=0.3, mask="C2")
        config.leaves[0].update(mask_axis="hue")
        config.leaves[3].update(mask_axis="tint", blur=4.0)
        config.leaves[6].update(mask_axis="lab-a")               # not referenced
        config.combos[0].update(mask_a="M1", mask_b="M4", operation="or")
        config.combos[1].update(mask_a="C1", mask_b="M1", operation="xor")
        text = spec.to_infotext(config)
        self.assertIn("masks=M1,M4", text)
        self.assertIn("combos=C1,C2", text)
        self.assertNotIn("M7.", text)
        restored = spec.parse_own(text)
        self.assertEqual(_chain_view(restored, False), _chain_view(config, False))

    def test_survives_forges_infotext_quoting(self):
        S.require_forge(self)
        tools = _forge_infotext_tools()
        for scenario in S.SCENARIOS:
            config = S.scenario_config(scenario)
            params = {"Steps": 28, "Sampler": "Euler", spec.INFOTEXT_KEY: spec.to_infotext(config),
                      spec.STATUS_KEY: "base: krea2 (Wan21); mods I+II; 28 evals, 14 edited | hires: x", "Seed": 1}
            parsed = _through_forge(tools, params)
            with self.subTest(scenario=scenario.label):
                self.assertEqual(parsed[spec.INFOTEXT_KEY], params[spec.INFOTEXT_KEY])
                self.assertEqual(parsed[spec.STATUS_KEY], params[spec.STATUS_KEY])
                self.assertEqual(parsed["Seed"], "1")

    def test_values_off_the_slider_grid_round_trip_exactly(self):
        """Typed (unclamped) numbers, XYZ and API values come back as they were — not cut to six significant
        digits, and a tiny value is not rounded to 0; binary noise still prints short."""
        config = spec.default_config()
        config.enabled = True
        config.modifiers[0].update(exposure=1.23456789, tone_compression=1e-7, end=0.987654321,
                                   contrast=-123.456789, temperature=0.1 + 0.2)
        text = spec.to_infotext(config)
        parts = text.split(";")
        for part in ("I.exposure=1.23456789", "I.tone_compression=1e-07", "I.end=0.987654321",
                     "I.contrast=-123.456789", "I.temperature=0.3"):
            self.assertIn(part, parts)
        restored = spec.parse_own(text)
        for name in ("exposure", "tone_compression", "end", "contrast"):
            self.assertEqual(restored.modifiers[0][name], config.modifiers[0][name], name)
        for hires in (False, True):
            self.assertEqual(_chain_view(restored, hires)[0], _chain_view(config, hires)[0])

    def test_bad_values_fall_back_to_defaults(self):
        config = spec.parse_own("v1;mods=I,XI,II;I.exposure=abc;I.kind=Nope;I.pass=hires;II.contrast=nan;"
                                "M1.mask_mode=sideways;C1.mask_a=C3;junk;=;I.unknown=1")
        self.assertTrue(config.modifiers[0]["active"])
        self.assertTrue(config.modifiers[1]["active"])
        self.assertEqual(config.modifiers[0]["exposure"], 0.0)
        self.assertEqual(config.modifiers[0]["kind"], "Advanced")
        self.assertEqual(config.modifiers[0]["pass"], "Hires")      # case-insensitive choice
        self.assertEqual(config.modifiers[1]["contrast"], 0.0)
        self.assertEqual(config.leaves[0]["mask_mode"], "highs")
        self.assertEqual(config.combos[0]["mask_a"], "none")        # C1 cannot reference a later combo


class PasteTests(unittest.TestCase):
    def test_no_colorcraft_key_turns_it_off_and_leaves_the_rest(self):
        params = {"Steps": "28"}
        self.assertFalse(spec.paste_value(params, "enabled"))
        for path in ("masking", "I.exposure", "M1.mask_axis", "C1.operation"):
            self.assertIsNone(spec.paste_value(params, path))

    def test_paste_replaces_every_field(self):
        config = spec.default_config()
        config.enabled = True
        config.modifiers[0].update(kind="Chroma", temperature=0.4)
        params = {spec.INFOTEXT_KEY: spec.to_infotext(config)}
        values = [spec.paste_value(params, path) for path in spec.arg_names()]
        self.assertEqual(len(values), spec.ARG_COUNT)
        restored = spec.config_from_args(values)
        self.assertEqual(_chain_view(restored, False), _chain_view(config, False))
        self.assertTrue(values[0])
        self.assertIs(spec.paste_value(params, "II.active"), False)

    def test_our_key_wins_over_a_foreign_one(self):
        ours = spec.default_config()
        ours.enabled = True
        ours.modifiers[0].update(exposure=0.2)
        params = {spec.INFOTEXT_KEY: spec.to_infotext(ours), "Colorcraft": "v1;tint=0.5"}
        self.assertEqual(spec.paste_value(params, "I.exposure"), 0.2)
        self.assertEqual(spec.paste_value(params, "I.tint"), 0.0)

    def test_unknown_foreign_value_is_ignored(self):
        params = {"Colorcraft": "something else"}
        self.assertFalse(spec.paste_value(params, "enabled"))
        self.assertIsNone(spec.paste_value(params, "I.exposure"))

    def test_decode_is_memoised_per_value(self):
        params = {spec.INFOTEXT_KEY: "v1;mods=I;I.exposure=0.1"}
        self.assertIs(spec.decode(params), spec.decode(dict(params)))

    def test_an_unreadable_foreign_value_is_ignored_without_raising(self):
        """A dict of another shape under ``Colorcraft`` (left by some other extension's paste callback) reads
        as an unknown value, once — Forge would log a traceback for every panel field that raised."""
        for raw in ({"modifiers": {"I": None}}, {"modifiers": "I:..."}, {"masks": 5}):
            params = {spec.FOREIGN_KEY: raw}
            spec._DECODE_CACHE.clear()
            with self.subTest(raw=raw), mock.patch.object(spec, "parse_upstream", wraps=spec.parse_upstream) as parse:
                values = [spec.paste_value(params, path) for path in spec.arg_names()]
                self.assertIs(values[0], False)
                self.assertTrue(all(value is None for value in values[1:]))
                self.assertEqual(parse.call_count, 1)


class CompactApiArgumentTests(unittest.TestCase):
    """The API may send the whole set-up as the first script argument instead of 579 positional values:
    this extension's infotext string, or a dict of ``arg_names()`` paths. Forge's API fills the other
    positions with the panel defaults (modules/api/api.py ``init_default_script_args``)."""

    @staticmethod
    def _rest():
        return spec.config_to_args(spec.default_config())[1:]

    def test_infotext_string_equals_the_positional_arguments(self):
        for scenario in S.SCENARIOS:
            config = S.scenario_config(scenario)
            text = spec.to_infotext(config)
            positional = spec.config_from_args(spec.config_to_args(config))
            compact = spec.config_from_args([text] + self._rest())
            with self.subTest(scenario=scenario.label):
                self.assertTrue(spec.is_compact_arg(text))
                self.assertTrue(spec.args_enabled([text] + self._rest()))
                for hires in (False, True):
                    self.assertEqual(_chain_view(compact, hires), _chain_view(positional, hires))

    def test_the_string_form_reads_like_a_paste(self):
        text = "  v1 ; mods=II;II.kind=Luma;II.exposure=0.3;junk;XI.exposure=1"
        self.assertTrue(spec.is_compact_arg(text))
        config = spec.config_from_args([text, True, *self._rest()[1:]])   # positional values are ignored
        self.assertTrue(config.enabled)
        self.assertFalse(config.masking)
        self.assertEqual([m["active"] for m in config.modifiers[:3]], [False, True, False])   # tab I too
        self.assertEqual((config.modifiers[1]["kind"], config.modifiers[1]["exposure"]), ("Luma", 0.3))

    def test_dict_of_argument_paths(self):
        mapping = {"I.kind": "Luma", "I.exposure": 0.3, "II.active": True, "II.kind": "chroma",
                   "II.vibrance": 0.2, "II.mask": "M2", "masking": "1", "M2.mask_axis": "hue", "C1.mask_a": "C3",
                   "I.pass": None, "nonsense": 1, "I.nope": 2, "XI.exposure": 1.0, "II.contrast": float("nan"),
                   "M2.mask_mode": "sideways"}
        self.assertTrue(spec.is_compact_arg(mapping))
        self.assertTrue(spec.args_enabled([mapping]))           # sending the dict asks for Colorcraft
        config = spec.config_from_args([mapping] + self._rest())
        expected = spec.default_config()                        # modifier I is Active by default
        expected.enabled, expected.masking = True, True
        expected.modifiers[0].update(kind="Luma", exposure=0.3)
        expected.modifiers[1].update(active=True, kind="Chroma", vibrance=0.2, mask="M2")
        expected.leaves[1]["mask_axis"] = "hue"
        self.assertEqual(config.combos[0]["mask_a"], "none")   # C1 cannot reference a later combo
        self.assertEqual(spec.to_infotext(config), spec.to_infotext(expected))
        for hires in (False, True):
            self.assertEqual(_chain_view(config, hires), _chain_view(expected, hires))

    def test_a_dict_can_switch_it_off(self):
        self.assertFalse(spec.args_enabled([{"enabled": False, "I.exposure": 0.3}]))
        self.assertFalse(spec.config_from_args([{"enabled": "false", "I.exposure": 0.3}]).enabled)

    def test_the_panels_own_first_argument_is_read_as_before(self):
        self.assertFalse(spec.args_enabled([]))
        self.assertFalse(spec.args_enabled([False] + self._rest()))
        self.assertTrue(spec.args_enabled([True] + self._rest()))
        self.assertTrue(spec.args_enabled(["True"]))
        for text in ("v1;exposure=0.3", "MOD(I:none)MASK()COMBO()", "something", ""):
            with self.subTest(text=text):                       # the fork's and upstream's strings are not ours
                self.assertFalse(spec.is_compact_arg(text))
                self.assertFalse(spec.args_enabled([text]))


class UpstreamFormatTests(unittest.TestCase):
    """muerrilla's Forge script: ``Colorcraft: MOD(I:…;II:…)MASK(M1:…)COMBO(C1:…)``."""

    @classmethod
    def setUpClass(cls):
        cls.up = _upstream()

    def _sample(self):
        tabs = [
            upstream_tab(mask="C1", strength=0.8, start=0.1, end=0.9, advanced=True, exponent=1.2, bias=0.4,
                         start_off=0.05, end_off=-0.05, smooth=False, exposure=0.3, contrast=0.2, vibrance=0.4,
                         temperature=-0.3, lab_a=0.2, color_shift_amount=0.3, color_shift_mode="legacy",
                         color_shift_red=0.5, color_shift_brightness=-0.1, clarity=0.2, hires=False),
            upstream_tab(active=False, exposure=0.9),
            upstream_tab(mask="M3", tint=0.4, chroma_contrast=0.5, chroma_center=-0.2, hires=True),
        ]
        leaves = {"M1": upstream_leaf(mask_axis="hue", mask_mode="range", mask_width=0.5, blur=8.0, normalize=True),
                  "M2": upstream_leaf(mask_axis="clarity", mask_mode="lows", contrast=2.0),
                  "M3": upstream_leaf(mask_axis="saturation", mask_mode="protect range", mask_center=0.3),
                  "M9": upstream_leaf(mask_axis="tint")}
        combos = {"C1": upstream_combo(mask_a="M1", mask_b="M2", operation="subtract", spread=0.4)}
        return tabs, leaves, combos

    def _expected(self, tabs, leaves, combos, masking):
        config = spec.blank_config()
        config.enabled = True
        for i, tab in enumerate(tabs):
            if not tab["active"]:
                continue
            modifier = config.modifiers[i]
            modifier.update(active=True, kind="Advanced", more_colors=True, color_shift=True,
                            **{"pass": "Hires" if tab["hires"] else "Base"})
            values = dict(tab)
            if not values["advanced"]:
                values.update(bias=0.5, exponent=0.0, start_off=0.0, end_off=0.0)
            for name, value in values.items():
                if name in spec.MODIFIER_BY_NAME and name != "active":
                    modifier[name] = value
        written_leaves, written_combos = set(), set()
        if masking:
            for i, tab in enumerate(tabs):
                if tab["active"] and tab["mask"] != "none":
                    written_leaves.add(tab["mask"])
            for tag in list(written_leaves):
                if tag in combos:
                    written_combos.add(tag)
                    written_leaves.discard(tag)
                    written_leaves |= {combos[tag]["mask_a"], combos[tag]["mask_b"]}
        for tag in written_leaves:
            config.leaves[spec.MASK_TAGS.index(tag)].update(leaves[tag])
        for tag in written_combos:
            config.combos[spec.COMBO_TAGS.index(tag)].update(combos[tag])
        config.masking = bool(written_leaves or written_combos)
        return config

    def test_string_form(self):
        tabs, leaves, combos = self._sample()
        for masking in (True, False):
            text = upstream_infotext(self.up, tabs, leaves, combos, masking_enabled=masking)
            with self.subTest(masking=masking):
                self.assertTrue(spec.is_upstream_text(text))
                restored = spec.decode({"Colorcraft": text})
                expected = self._expected(tabs, leaves, combos, masking)
                self.assertEqual(restored.modifiers, expected.modifiers)
                self.assertEqual(restored.leaves, expected.leaves)
                self.assertEqual(restored.combos, expected.combos)
                self.assertEqual(restored.masking, expected.masking)
                self.assertTrue(restored.enabled)

    def test_dict_form_from_upstreams_own_paste_callback(self):
        tabs, leaves, combos = self._sample()
        text = upstream_infotext(self.up, tabs, leaves, combos)
        params = {"Colorcraft": text}
        self.up["parse_infotext"](text, params)            # upstream's on_infotext_pasted callback
        self.assertIsInstance(params["Colorcraft"], dict)
        from_dict = spec.decode(params)
        from_text = spec.decode({"Colorcraft": text})
        self.assertEqual(from_dict.modifiers, from_text.modifiers)
        self.assertEqual(from_dict.leaves, from_text.leaves)
        self.assertEqual(from_dict.combos, from_text.combos)
        self.assertEqual(from_dict.masking, from_text.masking)
        self.assertTrue(spec.paste_value(params, "enabled"))

    def test_our_reader_matches_upstreams_parse(self):
        tabs, leaves, combos = self._sample()
        text = upstream_infotext(self.up, tabs, leaves, combos)
        params = {"Colorcraft": text}
        self.up["parse_infotext"](text, params)
        self.assertEqual(spec.upstream_dict(text), params["Colorcraft"])

    def test_older_and_newer_field_counts_keep_what_lines_up(self):
        tabs, _, _ = self._sample()
        text = upstream_infotext(self.up, tabs[:1], masking_enabled=False)
        values = text[len("MOD(I:"):text.index(")MASK")].split(",")
        shorter = "MOD(I:" + ",".join(values[:12]) + ")MASK()COMBO()"
        config = spec.parse_upstream(shorter)
        self.assertEqual(config.modifiers[0]["exposure"], 0.3)        # 11th value
        self.assertEqual(config.modifiers[0]["pass"], "Base")
        longer = "MOD(I:" + ",".join(values + ["extra"]) + ")MASK()COMBO()"
        self.assertEqual(spec.parse_upstream(longer).modifiers[0]["exposure"], 0.3)

    def test_pasted_upstream_settings_render_like_upstreams_forge_tab(self):
        """Upstream's Forge tab is the Advanced node with Chroma Plus and Color Shift applied: rendering the
        pasted settings through our hook is bit-identical to that node."""
        S.require_forge(self)
        tabs, leaves, combos = self._sample()
        config = spec.decode({"Colorcraft": upstream_infotext(self.up, tabs, leaves, combos)})
        mods = []
        for i, tab in enumerate(tabs):
            if tab["active"] and not tab["hires"]:
                values = {k: v for k, v in config.modifiers[i].items() if k not in ("active", "kind", "mask")}
                mods.append(S.Mod("advanced", values, mask=config.modifiers[i]["mask"]))
        leaf_values = {tag: config.leaves[spec.MASK_TAGS.index(tag)] for tag in spec.MASK_TAGS}
        combo_values = {tag: config.combos[spec.COMBO_TAGS.index(tag)] for tag in spec.COMBO_TAGS}
        scenario = S.Scenario("upstream-paste", mods, leaf_values, combo_values, config.masking)
        self.assertTrue(all(m.values["more_colors"] and m.values["color_shift"] for m in mods))
        moved, total, _ = S.parity_run(self, "krea2", scenario, S.flow_sigmas(12))
        self.assertGreater(moved, total // 2)


class ForkFormatTests(unittest.TestCase):
    """aoleg's fork: ``Colorcraft: v1;exposure=0.3;…`` (its spec.to_infotext, defaults omitted)."""

    @classmethod
    def setUpClass(cls):
        cls.fork = S.load_fork()

    def _values(self, **changes):
        P = self.fork.params
        values = P.defaults()
        values.update({p["name"]: p["default"] for p in P.RUNTIME_PARAMS})
        values.update(changes)
        return values

    def _text(self, **changes):
        text = self.fork.spec.to_infotext(self._values(**changes))
        self.assertTrue(spec.is_fork_text(text), text)
        return text

    def test_reader_uses_the_forks_defaults(self):
        self.assertEqual(spec.fork_values("v1"), {k: v for k, v in self._values().items()})
        text = self._text(exposure=0.3, mask_axis="hue", masking=True, more_colors=True, apply_to_hr=False)
        self.assertEqual(spec.fork_values(text), self.fork.spec.from_infotext(text))

    def test_mapping(self):
        config = spec.decode({"Colorcraft": self._text(
            exposure=0.3, tone_compression=0.1, vibrance=0.4, saturation=0.5, chroma_contrast=0.2,
            chroma_center=0.25, more_colors=True, lab_a=0.2, color_shift=True, color_shift_amount=0.3,
            mode="legacy", red=0.4, brightness=-0.1, recenter_override=True, recenter=0.8, apply_to_hr=False)})
        tab = config.modifiers[0]
        self.assertTrue(config.enabled)
        self.assertTrue(tab["active"])
        self.assertEqual(tab["kind"], "Advanced")
        self.assertEqual(tab["pass"], "Base")
        self.assertEqual(tab["chroma_center"], -0.5)                 # fraction 0.25 -> 2*0.25 - 1
        self.assertEqual((tab["color_shift_mode"], tab["color_shift_red"], tab["color_shift_brightness"]),
                         ("legacy", 0.4, -0.1))
        self.assertTrue(tab["more_colors"] and tab["color_shift"] and tab["recenter_override"])
        self.assertEqual(spec.decode({"Colorcraft": self._text()}).modifiers[0]["pass"], "Both")
        self.assertEqual(spec.decode({"Colorcraft": self._text()}).modifiers[0]["chroma_center"], 0.0)

    def test_masks_combine_blur_and_preview(self):
        config = spec.decode({"Colorcraft": self._text(
            exposure=0.4, masking=True, mask_axis="hue", mask_mode="range", mask_width=0.5,
            mask_combine=True, mask_operation="xor", mask_b_axis="tint", mask_b_mode="lows",
            mask_blur_radius=12.0, mask_spread=0.3)})
        self.assertTrue(config.masking)
        self.assertEqual(config.modifiers[0]["mask"], "C1")
        self.assertEqual((config.leaves[0]["mask_axis"], config.leaves[0]["mask_mode"]), ("hue", "range"))
        self.assertEqual((config.leaves[1]["mask_axis"], config.leaves[1]["mask_mode"]), ("tint", "lows"))
        self.assertEqual((config.combos[0]["mask_a"], config.combos[0]["mask_b"], config.combos[0]["operation"]),
                         ("M1", "M2", "xor"))
        self.assertEqual((config.combos[0]["blur"], config.combos[0]["spread"]), (12.0, 0.3))
        preview = spec.decode({"Colorcraft": self._text(masking=True, mask_preview=True,
                                                        mask_preview_color="green")}).modifiers[0]
        self.assertEqual((preview["kind"], preview["color_shift_mode"], preview["color_shift_amount"]),
                         ("Shift", "legacy", 1.0))
        self.assertEqual((preview["start"], preview["end"]), (1.0, 1.0))
        self.assertEqual((preview["color_shift_red"], preview["color_shift_green"], preview["color_shift_blue"]),
                         (-0.5, 0.5, -0.5))

    def test_pasted_fork_settings_render_like_the_forks_chain(self):
        """The fork's chain (its spec.build_chain) evaluated by upstream's node — with the fork's Chroma Center
        mapped onto upstream's scale — is bit-identical to our hook on the pasted settings."""
        S.require_forge(self)
        origin = S.load_origin()
        cases = {
            "advanced": dict(exposure=0.3, contrast=0.2, vibrance=0.4, chroma_contrast=0.3, chroma_center=0.3,
                             temperature=-0.2, more_colors=True, lab_b=0.2, color_shift=True,
                             color_shift_amount=0.3, red=0.4, start=0.1, end=0.9, bias=0.3, advanced=False),
            "shaped-masked": dict(exposure=0.4, tint=0.2, advanced=True, exponent=1.5, start_off=0.1, end_off=0.05,
                                  masking=True, mask_axis="exposure", mask_mode="highs", mask_hardness=2.0,
                                  mask_combine=True, mask_operation="or", mask_b_axis="hue", mask_b_mode="range",
                                  mask_b_width=0.4, mask_blur_radius=16.0, mask_spread=0.5),
            "preview": dict(masking=True, mask_axis="temperature", mask_mode="split", mask_preview=True,
                            mask_preview_color="blue"),
        }
        walked = S.flow_sigmas(10)
        for label, changes in cases.items():
            with self.subTest(case=label):
                values = self._values(**changes)
                chain = self.fork.spec.build_chain(values)
                for entry in chain:
                    if "chroma_center" in entry["params"]:
                        entry["params"] = dict(entry["params"], chroma_center=2.0 * entry["params"]["chroma_center"] - 1.0)
                config = spec.decode({"Colorcraft": self.fork.spec.to_infotext(values)})
                p = S.make_p("krea2")
                vae = p.sd_model.forge_objects.vae
                latent_format = p.sd_model.model_config.latent_format
                callback = S.attach_ours(p, spec.config_to_args(config))
                unet = p.sd_model.forge_objects.unet
                unet.model_options["transformer_options"]["sampling_sigmas"] = walked
                with S.oracle_family(origin, "krea2"):
                    oracle = S.oracle_callback(origin, chain, S.FakeVAE(vae.latent_channels, vae.latent_dim), walked,
                                               latent_format)
                    moved = 0
                    for k, sigma in enumerate(S.midpoints(walked)):
                        x0 = S.latent("krea2", 500 + k)
                        ours = callback(S.forge_args(x0.clone(), sigma, unet))
                        theirs = oracle(x0.clone(), sigma)
                        self.assertTrue(torch.equal(ours, theirs), f"{label} eval {k}")
                        moved += not torch.equal(ours, x0)
                self.assertGreater(moved, 0)


if __name__ == "__main__":
    unittest.main()
