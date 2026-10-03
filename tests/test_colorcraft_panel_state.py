"""Colorcraft shared editor — the panel's 67 script arguments (sam3ext/colorcraft/panel_state.py).

* Layout and the hidden state's codec (``""`` = defaults, JSON with ``v``/``rev``, defaults left out).
* The same ``Config`` as v0.31.0: random looks in three layouts (fresh state, stale state + editors, rev
  mismatch) decode to exactly what v0.31.0's 579 positional values decode to — values, Python types,
  infotext and the chain for both passes; XYZ applies after decoding.
* API forms through Forge's ``init_script_args`` (compact string/dict, ``[enabled(, masking)]``, the refused
  579-value list, the wrapped list, 579 values passed straight to the hook, states without rev).
* Paste, UI (``_parse_info``) and API (``apply_infotext``) casts: own, upstream and fork infotext.
* The at-a-glance labels and summary (Korean text pinned), the generated schema file and the copy.
"""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from sam3ext.colorcraft import panel_state as P  # noqa: E402
from sam3ext.colorcraft import spec, ui  # noqa: E402
from tests import _colorcraft_panel_support as PS  # noqa: E402

same_config = PS.same_config


class LayoutTests(unittest.TestCase):
    def test_argument_layout(self):
        self.assertEqual(P.ARG_COUNT, 67)
        self.assertEqual(len(P.ARG_NAMES), 67)
        self.assertEqual(P.ARG_NAMES[:6], ("enabled", "masking", "state", "debug", "debug_step", "ref"))
        self.assertEqual((P.STATE_INDEX, P.REF_INDEX, P.EDITOR_OFFSET, P.EDITOR_COUNT), (2, 5, 6, 61))
        self.assertEqual(P.ARG_NAMES[6:50], tuple(f"mod.{f.name}" for f in spec.MODIFIER_FIELDS))
        self.assertEqual(P.ARG_NAMES[50:60], tuple(f"leaf.{f.name}" for f in spec.LEAF_FIELDS))
        self.assertEqual(P.ARG_NAMES[60:], tuple(f"combo.{f.name}" for f in spec.COMBO_FIELDS))
        self.assertEqual((P.DEFAULT_STATE, P.DEFAULT_REF), ("", "I|M1|0"))
        self.assertEqual(len(P.STATE_PATHS), 575)
        self.assertEqual(list(P.STATE_PATHS), spec.arg_names()[2:-2])
        self.assertEqual(len(P.default_args()), 67)

    def test_ref(self):
        self.assertEqual(P.parse_ref("III|C2|3fa9c1d2e4b0"), ("III", "C2", "3fa9c1d2e4b0"))
        self.assertEqual(P.parse_ref(P.make_ref("X", "M10", "0")), ("X", "M10", "0"))
        for bad in (None, "", "I|M1", "I|M1|", "XI|M1|r", "I|C6|r", "I|none|r", "I|M1|r|x", 5):
            self.assertIsNone(P.parse_ref(bad), bad)

    def test_paste_target_is_never_a_combo(self):
        config = spec.default_config()
        config.masking = True
        config.modifiers[0]["active"] = False
        config.modifiers[3].update(active=True, kind="Luma", mask="C1")
        config.combos[0].update(mask_a="M4", mask_b="M2")
        self.assertEqual(P.paste_target(config), ("IV", "M2"))           # the first LEAF in use
        self.assertEqual(P.paste_target(spec.default_config()), ("I", "M1"))


class StateCodecTests(unittest.TestCase):
    def test_parse_state(self):
        self.assertEqual(P.parse_state(""), ({}, "0"))
        self.assertEqual(P.parse_state(None), ({}, "0"))
        self.assertEqual(P.parse_state({"rev": "r", "I.exposure": 0.3}), ({"I.exposure": 0.3}, "r"))
        self.assertEqual(P.parse_state({"I.exposure": 0.3}), ({"I.exposure": 0.3}, None))
        self.assertEqual(P.parse_state('  {"v": 1, "rev": "7", "M1.blur": 2}'), ({"M1.blur": 2}, "7"))
        self.assertEqual(P.parse_state('{"v": 1.0, "rev": "r"}'), ({}, "r"))          # JSON 1.0 is 1 in JS too
        dropped = {"v": 1, "rev": "r", "enabled": True, "masking": True, "debug": True, "debug_step": 9,
                   "I.nope": 1, "foo": 2, "C6.blur": 3, "II.kind": "Luma"}
        self.assertEqual(P.parse_state(json.dumps(dropped)), ({"II.kind": "Luma"}, "r"))
        for bad in ("{not json", "[1, 2]", '{"v": 2}', '{"v": "1"}', '{"v": true}', '{"v": null}', "hello", 5, True,
                    [1]):
            with self.assertRaises(P.UnreadableState, msg=repr(bad)):
                P.parse_state(bad)
        self.assertTrue(issubclass(P.UnreadableState, P.RefusedArgs))
        self.assertTrue(issubclass(P.LegacyPositionalArgs, P.RefusedArgs))
        self.assertTrue(issubclass(P.RefusedArgs, ValueError))

    def test_only_a_writer_rev_is_a_rev(self):
        """A rev is a non-empty string without ``|`` (12 hex chars from either writer). Numbers, ``""`` and the
        initial ``"0"`` in a state that is not the empty default count as no rev, so such a state (an API caller's
        own JSON) is never overlaid by the editors -- and colorcraft_editor.js reads it the same way."""
        for rev in (7, 1.0, 0, "0", "", "a|b", None, True, ["r"]):
            with self.subTest(rev=rev):
                self.assertEqual(P.parse_state(json.dumps({"v": 1, "rev": rev, "I.exposure": 0.3})),
                                 ({"I.exposure": 0.3}, None))
                self.assertEqual(P.parse_state({"rev": rev}), ({}, None))
        self.assertEqual(P.parse_state(""), ({}, P.INITIAL_REV))                      # only "" / None carry "0"
        self.assertEqual(P.parse_state('{"rev": "3fa9c1d2e4b0"}'), ({}, "3fa9c1d2e4b0"))

    def test_dump_state(self):
        self.assertEqual(P.dump_state(spec.default_config(), "r"), '{"v":1,"rev":"r"}')
        config = spec.default_config()
        config.modifiers[0]["exposure"] = -0.0                            # kept apart from the default 0.0
        config.modifiers[1]["tint"] = math.nan                            # never stored: coerce reads the default
        config.modifiers[0]["active"] = False                             # I's default is True
        config.leaves[2]["mask_axis"] = "hue"
        text = P.dump_state(config, "abc")
        self.assertEqual(text, '{"v":1,"rev":"abc","I.active":false,"I.exposure":-0.0,"M3.mask_axis":"hue"}')
        got = P.config_from_state(False, False, text)
        self.assertEqual(repr(got.modifiers[0]["exposure"]), "-0.0")
        self.assertIs(got.modifiers[0]["active"], False)
        self.assertEqual(got.modifiers[1]["tint"], 0.0)
        self.assertEqual(P.parse_state(text)[1], "abc")
        self.assertNotIn("|", P.new_rev())
        self.assertEqual(len(P.new_rev()), 12)

    def test_null_means_the_default(self):
        got = P.config_from_state(True, False, '{"v":1,"rev":"r","I.exposure":null,"I.active":null}')
        same_config(self, got, P.config_from_state(True, False, ""))

    def test_random_configs_round_trip(self):
        for n, config in enumerate(PS.random_configs(400, 400)):
            text = P.dump_state(config, f"r{n}")
            got = P.config_from_state(config.enabled, config.masking, text, config.debug, config.debug_step)
            same_config(self, got, config, f"#{n}")
            self.assertEqual(P.parse_state(text)[1], f"r{n}")

    def test_head_values_are_coerced_like_v031(self):
        for enabled, masking, debug, step in ((1, 0, "True", "12"), ("true", "off", 0, 3.0), (None, None, None, None)):
            got = P.config_from_state(enabled, masking, "", debug, step)
            want = spec.config_from_args([enabled, masking] + spec.config_to_args(spec.default_config())[2:-2]
                                         + [debug if debug is not None else False, step if step is not None else 5])
            self.assertEqual((got.enabled, got.masking, got.debug, repr(got.debug_step)),
                             (want.enabled, want.masking, want.debug, repr(want.debug_step)))


class SameConfigTests(unittest.TestCase):
    """What the panel sends decodes to v0.31.0's Config for the same look, whatever the editors show."""

    def test_random_configs_equal_v031(self):
        rng = random.Random(20261003)
        for n in range(400):
            config = PS.random_config(rng)
            mod = rng.choice(spec.MODIFIER_TAGS)
            mask = rng.choice(spec.MASK_TAGS + spec.COMBO_TAGS)
            fresh = P.args_from_config(config, mod, mask, rev="r1")
            stale_args = PS.stale_args(config, mod, mask, "r1")
            for label, args in (("fresh", fresh), ("stale state + editors", stale_args)):
                got = P.config_from_script_args(args)
                same_config(self, got, config, f"#{n} {label}")
                self.assertEqual(spec.to_infotext(got), spec.to_infotext(config))
                for hires in (False, True):
                    self.assertEqual(repr(spec.build_chain(got, hires)), repr(spec.build_chain(config, hires)))
            # a rev mismatch never lays the editors over (a paste won / an API state): the pure state
            mismatch = list(stale_args)
            mismatch[P.REF_INDEX] = P.make_ref(mod, mask, "other")
            pure = P.config_from_state(config.enabled, config.masking, mismatch[P.STATE_INDEX], config.debug,
                                       config.debug_step)
            same_config(self, P.config_from_script_args(mismatch), spec.config_from_args(spec.config_to_args(pure)))

    def test_only_the_editor_in_use_is_read(self):
        config = spec.default_config()
        config.enabled = True
        config = spec.config_from_args(spec.config_to_args(config))     # v0.31.0's coercion (debug_step 5.0)
        leaf_args = P.args_from_config(config, "I", "M1")
        leaf_args[P.EDITOR_OFFSET + 44 + 10] = "M3"                      # combo editor slot: ignored on M1
        same_config(self, P.config_from_script_args(leaf_args), config)
        combo_args = P.args_from_config(config, "I", "C1")
        combo_args[P.EDITOR_OFFSET + 44] = "hue"                          # leaf editor slot: ignored on C1
        same_config(self, P.config_from_script_args(combo_args), config)

    def test_xyz_after_decoding(self):
        for config in PS.random_configs(7, 50):
            for xyz in ({"exposure": 0.25, "enabled": "True"}, {"tint": -0.4}, {"enabled": "False"}):
                got = spec.apply_xyz(P.config_from_script_args(P.args_from_config(config)), xyz)
                same_config(self, got, spec.apply_xyz(config, xyz))


class ApiFormTests(unittest.TestCase):
    def test_compact_forms_unchanged(self):
        config = spec.default_config()
        config.enabled = True
        config.modifiers[0]["exposure"] = 0.3
        text = spec.to_infotext(config)
        same_config(self, P.config_from_script_args(PS.forge_api_fill([text])), spec.parse_own(text))
        self.assertTrue(P.args_enabled(PS.forge_api_fill([text])))
        d = {"I.exposure": 0.3, "M1.mask_axis": "hue"}
        same_config(self, P.config_from_script_args(PS.forge_api_fill([d])), spec.config_from_mapping(d))
        self.assertTrue(P.args_enabled(PS.forge_api_fill([d])))
        self.assertFalse(P.args_enabled(PS.forge_api_fill([{"enabled": False, "I.exposure": 0.3}])))

    def test_short_requests_mean_what_they_meant(self):
        for req, enabled, masking in (([True], True, False), ([True, True], True, True), ([False], False, False)):
            args = PS.forge_api_fill(req)
            got = P.config_from_script_args(args)
            same_config(self, got, spec.config_from_args(req), str(req))
            self.assertEqual((got.enabled, got.masking), (enabled, masking))
            self.assertEqual(P.args_enabled(args), enabled)

    def test_old_579_positional_is_refused_through_forge(self):
        config = PS.random_config(random.Random(3))
        config.enabled = True
        old = spec.config_to_args(config)
        with self.assertRaises(P.LegacyPositionalArgs) as caught:
            P.config_from_script_args(PS.forge_api_fill(old))
        self.assertEqual(str(caught.exception), P.LEGACY_NOTE)
        self.assertEqual(P.LEGACY_NOTE, "v0.31.0 positional arguments (579 values) are no longer read - send the "
                                         "infotext string or a dict of argument paths as the first argument, or the "
                                         "old list wrapped in one list")
        # ... read exactly as v0.31.0 when wrapped in one list, or when a Python caller passes all 579 directly
        same_config(self, P.config_from_script_args(PS.forge_api_fill([old])), config)
        self.assertTrue(P.args_enabled(PS.forge_api_fill([old])))
        self.assertTrue(P.args_enabled([tuple(old)]))
        same_config(self, P.config_from_script_args([tuple(old)]), config)
        same_config(self, P.config_from_script_args(old), config)
        off = list(old)
        off[0] = False
        self.assertFalse(P.args_enabled(PS.forge_api_fill([off])))

    def test_api_state_without_rev_is_never_overlaid(self):
        args = PS.forge_api_fill([True, False, json.dumps({"I.exposure": 0.3, "M1.mask_axis": "hue"})])
        got = P.config_from_script_args(args)
        self.assertEqual(got.modifiers[0]["exposure"], 0.3)
        self.assertEqual(got.leaves[0]["mask_axis"], "hue")
        got = P.config_from_script_args(PS.forge_api_fill([True, False, {"I.exposure": 0.3}]))
        self.assertEqual(got.modifiers[0]["exposure"], 0.3)

    def test_old_579_positional_with_null_or_text_in_slot_2_is_refused(self):
        """v0.31.0 read None in a modifier slot as its default, so slot 2 (I.active) may be null, or a string;
        cut to 67 by Forge such a list must still be refused, not read as an empty panel state."""
        config = PS.random_config(random.Random(3))
        config.enabled = True
        config.modifiers[0]["exposure"] = 0.3
        old = spec.config_to_args(config)
        for slot2 in (None, "true", "", 1):
            legacy = list(old)
            legacy[2] = slot2
            with self.subTest(slot2=slot2):
                with self.assertRaises(P.LegacyPositionalArgs):
                    P.config_from_script_args(PS.forge_api_fill(legacy))
                # all 579 straight to the hook: v0.31.0's reader, null = I's default
                same_config(self, P.config_from_script_args(legacy), spec.config_from_args(legacy))
        # recognised by slots 3-5 alone (v0.31.0's I.kind, I.pass, I.mask, spelled as v0.31.0's coerce accepts)
        for head in ([True, False, None, "Advanced"], [True, False, None, "chroma_plus"],
                     [True, False, None, None, "hires"], [True, False, None, None, None, "M1"],
                     [True, False, None, None, None, "none"], [True, False, "", False, 5, "C2"]):
            with self.subTest(head=head), self.assertRaises(P.LegacyPositionalArgs):
                P.config_from_script_args(PS.forge_api_fill(head))
        # the panel's own head never looks like it
        for head in ([True, False, None], [True, False, "", True, 12, "II|C1|r"], [True, False, "", None, None, None],
                     [True, False, "", "true", "7", "I|M1|0"]):
            with self.subTest(head=head):
                P.config_from_script_args(PS.forge_api_fill(head))

    def test_state_with_the_initial_rev_is_never_overlaid(self):
        """Forge fills ``ref`` with "I|M1|0"; a non-empty state carrying rev 0 must not get modifier I's and M1's
        default editors laid over it."""
        for rev in (0, "0"):
            state = {"v": 1, "rev": rev, "I.exposure": 0.3, "M1.mask_axis": "hue", "II.exposure": 0.2}
            for sent in (json.dumps(state), state):
                with self.subTest(rev=rev, sent=type(sent).__name__):
                    got = P.config_from_script_args(PS.forge_api_fill([True, False, sent]))
                    self.assertEqual(got.modifiers[0]["exposure"], 0.3)
                    self.assertEqual(got.leaves[0]["mask_axis"], "hue")
                    self.assertEqual(got.modifiers[1]["exposure"], 0.2)

    def test_unreadable_state(self):
        for bad in ("{not json", '{"v": 2}', "hello"):
            with self.assertRaises(P.UnreadableState, msg=bad):
                P.config_from_script_args(PS.forge_api_fill([True, False, bad]))
        with self.assertRaises(P.UnreadableState):
            P.config_from_script_args(PS.forge_api_fill([True, False, [1, 2]]))
        with self.assertRaises(P.LegacyPositionalArgs):
            P.config_from_script_args(PS.forge_api_fill([True, False, 0.5]))

    def test_short_panel_layouts_are_not_overlaid(self):
        # fewer than 67 arguments: the state alone (no editors to lay over)
        state = P.dump_state(spec.default_config(), "r")
        got = P.config_from_script_args([True, False, state, False, 5, "I|M1|r"] + [None] * 10)
        same_config(self, got, P.config_from_state(True, False, state))


class PasteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gr.Blocks():
            cls.components, cls.fields = ui.build(PS.elem_id("txt2img"), on_debug_refresh=lambda *a: [])

    def ui_paste(self, params, args):
        """Forge infotext_utils._parse_info, then the script args the next Generate sends."""
        args = list(args)
        index = {id(c): i for i, c in enumerate(self.components)}
        for comp, key in self.fields:
            v = key(params)
            if v is None or id(comp) not in index:
                continue
            if isinstance(v, dict):
                v = v.get("value")
            valtype = type(comp.value)
            args[index[id(comp)]] = False if (valtype is bool and v == "False") else valtype(v)
        return args

    def api_paste(self, params):
        """modules/api/api.py apply_infotext -> mentioned_script_args -> init_script_args."""
        index = {id(c): i for i, c in enumerate(self.components)}
        mentioned = {}
        for comp, key in self.fields:
            if id(comp) not in index:
                continue
            v = key(params)
            if v is None:
                continue
            target = type(comp.value)
            if not isinstance(v, target):
                v = target(v)
            mentioned[index[id(comp)]] = v
        args = P.default_args()
        for i, v in mentioned.items():
            args[i] = v
        return args

    def test_own_upstream_fork_paste(self):
        texts = []
        for config in PS.random_configs(11, 60):
            config.enabled = True
            texts.append({spec.INFOTEXT_KEY: spec.to_infotext(config)})
        texts.append({spec.FOREIGN_KEY: "MOD(1,0.5,0.75,0.3,0,0,0,0,0,0,0,0)"})
        texts.append({spec.FOREIGN_KEY: {"modifiers": {"I": {"exposure": 0.3, "start": 0.0, "end": 1.0},
                                                   "III": {"tint": 0.2, "hires": True, "mask": "M1"}},
                                     "masks": {"M1": {"mask_axis": "hue"}}}})
        texts.append({spec.FOREIGN_KEY: "v1;mods=I;I.exposure=0.4"})
        decoded = 0
        for params in texts:
            want = spec.decode(params)
            if want is None:
                continue
            decoded += 1
            start = P.default_args()
            start[P.EDITOR_OFFSET + P.MOD_NAMES.index("exposure")] = 2.0       # an unsaved edit before the paste
            for args in (self.ui_paste(params, start), self.api_paste(params)):
                got = P.config_from_script_args(args)
                self.assertEqual(spec.to_infotext(got), spec.to_infotext(want), params)
                for hires in (False, True):
                    self.assertEqual(repr(spec.build_chain(got, hires)), repr(spec.build_chain(want, hires)))
        self.assertEqual(decoded, 63)

    def test_one_paste_one_rev(self):
        params = {spec.INFOTEXT_KEY: "v1;mods=II;II.kind=Punch;II.clarity=0.4"}
        values = {id(c): key(params) for c, key in self.fields}
        state = values[id(self.components[P.STATE_INDEX])]
        ref = values[id(self.components[P.REF_INDEX])]
        self.assertEqual(ref, f"II|M1|{P.parse_state(state)[1]}")

    def test_no_key_turns_off_and_leaves_the_rest(self):
        start = P.args_from_config(PS.random_config(random.Random(5)))
        got = self.ui_paste({"Steps": "20"}, start)
        self.assertIs(got[0], False)
        self.assertEqual(got[1:], start[1:])


class OverviewTests(unittest.TestCase):
    """The selector labels and the summary line (javascript/colorcraft_editor.js renders the same bytes)."""

    def _labels(self, choices):
        return [label for label, _ in choices]

    def test_default(self):
        mods, masks, html = P.overview(spec.default_config())
        self.assertEqual(self._labels(mods), ["I ●"] + spec.MODIFIER_TAGS[1:])
        self.assertEqual([value for _, value in mods], spec.MODIFIER_TAGS)
        self.assertEqual(self._labels(masks), spec.MASK_TAGS + spec.COMBO_TAGS)
        self.assertEqual(html, '<div class="samextra-cc-summary">꺼짐 — 켠 수정자: I Advanced</div>')

    def test_nested_combo_in_use(self):
        config = spec.default_config()
        config.enabled = config.masking = True
        config.modifiers[1]["exposure"] = 0.2                            # edited but off: ○
        config.modifiers[2].update(active=True, kind="Chroma", mask="C2", **{"pass": "Hires"})
        config.combos[0].update(mask_a="M1", mask_b="M2")
        config.combos[1].update(mask_a="C1", mask_b="M3")
        config.leaves[4]["blur"] = 3.0                                    # edited, not reached: ○
        mods, masks, html = P.overview(config)
        self.assertEqual(self._labels(mods)[:4], ["I ●", "II ○", "III ●", "IV"])
        self.assertEqual(self._labels(masks), ["M1 ●", "M2 ●", "M3 ●", "M4", "M5 ○", "M6", "M7", "M8", "M9", "M10",
                                               "C1 ●", "C2 ●", "C3", "C4", "C5"])
        self.assertEqual(html, '<div class="samextra-cc-summary">켠 수정자: I Advanced, III Chroma ➜ C2 (Hires) · '
                               '쓰는 마스크: M1, M2, M3, C1, C2</div>')

    def test_incomplete_combo_is_not_in_use(self):
        """A combo whose A and B do not both resolve runs unmasked (spec.build_mask_specs, the hook's
        "incomplete combo, unmasked"): no ● for it or for what it names, and the summary says so."""
        config = spec.default_config()
        config.enabled = config.masking = True
        config.modifiers[2].update(active=True, kind="Luma", mask="C1")
        config.modifiers[3].update(active=True, kind="Chroma", mask="C2")
        config.combos[0].update(mask_a="M3")                              # B stays none
        config.combos[1].update(mask_a="C1", mask_b="M4")                 # names the incomplete C1
        self.assertEqual(spec.build_chain(config, False).unresolved_masks, [])   # no-op entries are left out
        mods, masks, html = P.overview(config)
        self.assertEqual(self._labels(mods)[:4], ["I ●", "II", "III ●", "IV ●"])
        self.assertEqual(self._labels(masks), spec.MASK_TAGS + ["C1 ○", "C2 ○", "C3", "C4", "C5"])
        self.assertEqual(html, '<div class="samextra-cc-summary">켠 수정자: I Advanced, III Luma ➜ C1(미완성), '
                               'IV Chroma ➜ C2(미완성) · 쓰는 마스크: 없음</div>')
        config.modifiers[2]["exposure"] = config.modifiers[3]["saturation"] = 0.2
        self.assertEqual(spec.build_chain(config, False).unresolved_masks, ["III->C1", "IV->C2"])
        config.combos[0]["mask_b"] = "M5"                                 # now both resolve
        self.assertEqual(spec.build_chain(config, False).unresolved_masks, [])
        mods, masks, html = P.overview(config)
        self.assertEqual([label for label in self._labels(masks) if label.endswith("●")],
                         ["M3 ●", "M4 ●", "M5 ●", "C1 ●", "C2 ●"])
        self.assertEqual(html, '<div class="samextra-cc-summary">켠 수정자: I Advanced, III Luma ➜ C1, '
                               'IV Chroma ➜ C2 · 쓰는 마스크: M3, M4, M5, C1, C2</div>')

    def test_disabled_and_none_active(self):
        config = spec.default_config()
        config.masking = True
        config.modifiers[0]["active"] = False
        config.modifiers[1].update(active=True, kind="Luma", mask="M2")
        self.assertEqual(P.overview(config)[2],
                         '<div class="samextra-cc-summary">꺼짐 — 켠 수정자: II Luma ➜ M2 · 쓰는 마스크: M2</div>')
        config.enabled = True
        config.modifiers[1]["active"] = False
        self.assertEqual(P.overview(config)[2], '<div class="samextra-cc-summary">켠 수정자 없음 · 쓰는 마스크: 없음</div>')
        config.masking = False
        config.modifiers[1]["active"] = True
        self.assertEqual(P.overview(config)[2], '<div class="samextra-cc-summary">켠 수정자: II Luma</div>')


class SchemaFileTests(unittest.TestCase):
    def test_schema_file_is_current(self):
        # Compared with LF: a core.autocrlf=true checkout (this repo's Windows clones) writes the file with CRLF in the
        # working tree — git stores LF either way, and write_schema_js() writes LF.
        data = (ROOT / "javascript" / "colorcraft_schema.js").read_bytes()
        self.assertEqual(data.replace(b"\r\n", b"\n"), P.schema_js().encode("utf-8"),
                         "regenerate: venv python -c \"from sam3ext.colorcraft import panel_state; "
                         "panel_state.write_schema_js()\"")
        self.assertFalse(data.startswith(b"\xef\xbb\xbf"))

    def test_write_schema_js(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = P.write_schema_js(Path(tmp) / "schema.js")
            self.assertEqual(target.read_bytes(), P.schema_js().encode("utf-8"))

    def test_schema_carries_spec_tables(self):
        s = P.schema()
        self.assertEqual(s["kindGroups"], {k: list(v) for k, v in spec.KIND_GROUPS.items()})
        self.assertEqual(s["kindGates"], {k: list(v) for k, v in spec.KIND_GATES.items()})
        self.assertEqual(s["kindTakesMask"], dict(spec.KIND_TAKES_MASK))
        self.assertEqual(s["kindIds"], dict(spec.KIND_IDS))
        self.assertEqual(s["comboRefChoices"], [spec.combo_ref_choices(i) for i in range(spec.COMBO_COUNT)])
        self.assertEqual(s["activeDefaults"], [True] + [False] * 9)
        self.assertEqual(tuple(s["groupOrder"]), P.GROUP_ORDER)


class CopyTests(unittest.TestCase):
    def test_no_tilde_pairs_in_the_panel_text(self):
        """Gradio 4.40 Markdown renders a ``~ … ~`` pair as strikethrough (v0.31.0's "M1~M10 … C1~C5")."""
        for name in ("INTRO", "MASK_HELP", "DEBUG_HELP"):
            self.assertNotIn("~", getattr(ui, name), name)
        self.assertIn("M1–M10", ui.INTRO)
        self.assertIn("C1–C5", ui.INTRO)
        self.assertIn("수정자는 마스크 없이", ui.MASK_HELP)
        # the selector sits below the intro; the panel has no tabs any more
        self.assertIn("아래 **수정자** 선택 줄에서", ui.INTRO)
        for name in ("INTRO", "MASK_HELP"):
            self.assertNotIn("위 선택 줄", getattr(ui, name), name)
            self.assertNotIn("탭", getattr(ui, name), name)


if __name__ == "__main__":
    unittest.main()
