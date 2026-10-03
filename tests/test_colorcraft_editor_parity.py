"""Colorcraft shared editor — the browser side (javascript/colorcraft_editor.js) against the Python reader.

Runs the real JS under node (tests/js/colorcraft_bridge.mjs, colorcraft_session.mjs; only ``node:vm`` and
``node:crypto``) and decodes what it writes with ``panel_state``:

* JS ``commit`` equals the Python overlay of the same editors (150 random looks);
* JS ``overview`` equals ``panel_state.overview``: selector labels and summary html byte for byte (150);
* session model: 60 random sessions × 40 steps of edits, selector switches (``sync``) and Resets (``reset``),
  applied with Gradio's update rules — after every step the 67 arguments a Generate would send decode to
  exactly the values the user set;
* JS ``parseState`` reads hand-written states like ``panel_state.parse_state`` (version, rev, dropped keys);
* the page's js strings (``ui._js``) without colorcraft_editor.js, or with it throwing: the visible fallback.
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.colorcraft import panel_state as P  # noqa: E402
from sam3ext.colorcraft import spec, ui  # noqa: E402
from tests import _colorcraft_panel_support as PS  # noqa: E402


@unittest.skipIf(PS.NODE is None, "node is not on PATH (the editor's JS runs under node)")
class EditorParityTests(unittest.TestCase):
    def test_js_commit_equals_python_overlay(self):
        rng = random.Random(99)
        cases = []
        for _ in range(150):
            config = PS.random_config(rng)
            mod = rng.choice(spec.MODIFIER_TAGS)
            mask = rng.choice(spec.MASK_TAGS + spec.COMBO_TAGS)
            state = P.dump_state(PS.stale(config, mod, mask), "r")
            cases.append((config, mod, mask, state, P.editor_values(config, mod, mask)))
        out = PS.bridge([{"op": "commit", "state": s, "ref": P.make_ref(m, k, "r"), "editors": e}
                         for _, m, k, s, e in cases])
        self.assertEqual(len(out), len(cases))
        for (config, mod, mask, state, editors), committed in zip(cases, out):
            overlay = P.config_from_script_args([config.enabled, config.masking, state, config.debug,
                                                 config.debug_step, P.make_ref(mod, mask, "r")] + editors)
            from_js = P.config_from_script_args([config.enabled, config.masking, committed, config.debug,
                                                 config.debug_step])
            PS.same_config(self, from_js, overlay, f"{mod}/{mask}")
            PS.same_config(self, from_js, config, f"{mod}/{mask}")

    def test_js_overview_equals_python(self):
        cases = PS.random_configs(42, 150)
        jobs = [{"op": "overview", "state": P.dump_state(c, "r"), "ref": "I|M1|r",
                 "editors": P.editor_values(c, "I", "M1"), "enabled": c.enabled, "masking": c.masking} for c in cases]
        for config, got in zip(cases, PS.bridge(jobs)):
            mods, masks, html = P.overview(config)
            self.assertEqual(got["mods"], [list(x) for x in mods])
            self.assertEqual(got["masks"], [list(x) for x in masks])
            self.assertEqual(got["html"], html)

    def test_session_model(self):
        sessions = PS.node_json("colorcraft_session.mjs", {"seed": 7, "sessions": 60, "steps": 40})
        checked = 0
        for s, records in enumerate(sessions):
            for k, rec in enumerate(records):
                args, truth = rec["args"], rec["truth"]
                self.assertEqual(len(args), P.ARG_COUNT)
                got = P.config_from_script_args(args)
                want = spec.config_from_mapping(dict(truth))
                want.enabled, want.masking = bool(args[0]), bool(args[1])
                PS.same_config(self, got, want, f"session {s} step {k}")
                checked += 1
        self.assertEqual(checked, 60 * 40)

    def test_js_reads_states_like_python(self):
        states = ["", "  {}", '{"v": true}', '{"v": 1.0, "rev": "r"}', '{"v": "1"}', '{"v": 2}', '{"v": null}', "[1]",
                  "{bad", '{"v": 1} x', "null", '"x"', "5", '{"rev": 1.0, "I.exposure": 0.3}', '{"rev": 7}',
                  '{"rev": "0", "I.exposure": 0.3}', '{"rev": 0}', '{"rev": ""}', '{"rev": "a|b"}', '{"rev": null}',
                  '{"v": 1, "rev": "3fa9c1d2e4b0", "enabled": true, "debug": 1, "I.nope": 1, "C6.blur": 2, '
                  '"M1.blur": 2, "II.kind": "Luma"}']
        got = PS.bridge([{"op": "parse", "state": s} for s in states])
        for text, js in zip(states, got):
            try:
                paths, rev = P.parse_state(text)
                want = {"map": paths, "rev": rev}
            except P.UnreadableState:
                want = None
            self.assertEqual(js, want, text)

    def test_fallback_when_the_editor_is_missing_or_fails(self):
        """Without javascript/colorcraft_editor.js (or with it throwing: no schema) the selectors go back to the
        nodes the editors show and the summary says why -- never a selector on one node and the editors on
        another. Run through the page's own js strings."""
        import gradio as gr

        parts = {}
        with gr.Blocks() as demo:
            controls, _ = ui.build(PS.elem_id("txt2img"), parts=parts)
        sync, reset_mod, reset_mask, labels, kind = [d for d in demo.get_config_file()["dependencies"] if d.get("js")]
        editors = P.editor_values(spec.default_config(), "III", "C2")
        head = [True, True, "", "III|C2|r"]
        note = f'<div class="samextra-cc-summary">{ui.JS_FAILED}</div>'
        skip = {"__type__": "update"}
        calls = [(sync, ["II", "M1"] + head + editors), (reset_mod, head + editors), (reset_mask, head + editors),
                 (labels, head + editors)]
        for files, logged in ((["colorcraft_schema.js"], 1), (["colorcraft_editor.js"], 2)):
            jobs = [{"op": "call_page", "files": files, "js": d["js"], "args": args, "nOutputs": len(d["outputs"])}
                    for d, args in calls]
            jobs.append({"op": "call_page", "files": files, "js": labels["js"], "args": [True, True, "", ""] + editors,
                         "nOutputs": 3})
            jobs.append({"op": "call_page", "files": files, "js": kind["js"], "args": ["Luma"], "nOutputs": 10})
            got = PS.bridge(jobs)
            for (dep, _), result in zip(calls, got):
                with self.subTest(files=files, js=dep["js"][:80]):
                    out = result["out"]
                    self.assertEqual(len(out), len(dep["outputs"]))
                    self.assertEqual(out[:-3], [skip] * (len(out) - 3))
                    self.assertEqual(out[-3:], [{"__type__": "update", "value": "III"},
                                                {"__type__": "update", "value": "C2"}, note])
                    self.assertEqual(len(result["errors"]), logged, result["errors"])
                    self.assertTrue(result["errors"][-1].startswith("[Colorcraft] javascript/colorcraft_editor.js"))
            self.assertEqual(got[-2]["out"], [skip, skip, note])                # no readable ref: selectors alone
            self.assertEqual(got[-1]["out"], [])                                 # kind: no update
        # with both files the same strings run the editor
        args = ["II", "M1", True, True, "", P.DEFAULT_REF] + P.default_args()[P.EDITOR_OFFSET:]
        full = PS.bridge([{"op": "call", "js": sync["js"], "args": args, "nOutputs": 70}])[0]
        self.assertEqual(len(full), 70)
        self.assertEqual(full[1], f"II|M1|{P.parse_state(full[0])[1]}")
        self.assertNotIn(note, full)


if __name__ == "__main__":
    unittest.main()
