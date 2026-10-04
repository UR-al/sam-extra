"""Extra Schedulers — origin parity of Laplace with ComfyUI itself.

The oracle is upstream ComfyUI (GPL-3.0, the same license as this extension) at
``comfyanonymous/ComfyUI@36c0b0a687e5e6d7b55e3e61ab24262ffc0f2508``, kept verbatim in
``tests/_origin_comfyui_laplace.py``: ``get_sigmas_laplace`` (comfy/k_diffusion/sampling.py:52-59)
and the ``LaplaceScheduler`` node (comfy_extras/nodes_custom_sampler.py:112-133). The SHA-256 of
each verbatim block is pinned below.

What must match:

* the extension's ``get_sigmas_laplace`` is the upstream function unchanged (same source text);
* the Laplace scheduler's sigmas equal the node's output bit for bit, for SD and flow sigma ranges,
  plus the final 0 every Forge scheduler ends with (the node returns ``steps`` sigmas without it —
  ComfyUI's SamplerCustom then stops at sigma_min);
* the one other intended difference: where the node's clamp to [sigma_min, sigma_max] repeats a
  sigma (``REPEAT_CASES``), the extension spreads the steps over the part of the curve inside that
  range, or stops with an error when no such part is left (the spread itself is tested in
  tests/test_extra_schedulers_schedulers.py);
* the accordion's μ/β defaults, ranges and slider steps equal the node's inputs.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
import sys
import unittest
from pathlib import Path

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext import ui_extra_schedulers as ues  # noqa: E402
from sam3ext.extra_schedulers import schedulers as sch  # noqa: E402
from sam3ext.extra_schedulers import settings as st  # noqa: E402

ORIGIN_FILE = ROOT / "tests" / "_origin_comfyui_laplace.py"
ORIGIN_COMMIT = "comfyanonymous/ComfyUI@36c0b0a687e5e6d7b55e3e61ab24262ffc0f2508"
# SHA-256 of each verbatim block (the upstream lines, LF line endings, one trailing newline).
BLOCK_SHA256 = {
    "comfy/k_diffusion/sampling.py:52-59": "8cbf050bf6ab79c84d9ca715c433eb596913ab061750d588925e4cc716a3ee4e",
    "comfy_extras/nodes_custom_sampler.py:112-133": "1202a02ea91bc512036a28886ee5e89446e2b0a1b77895d4e685f4cba070cff7",
}
# SHA-256 of the whole upstream files at that commit (recorded when the blocks were cut out of them).
FILE_SHA256 = {
    "comfy/k_diffusion/sampling.py": "cfc4221ed0bc1f84a6f5689d3ec440cad3008c2541ab8f5e2599d1a73fac74ce",
    "comfy_extras/nodes_custom_sampler.py": "b0ba1521c72475e06fed15db004274ed7c8bb2bf73f1d58849faf6f9f9d264fe",
}
MARKER = re.compile(r"^# ---- upstream (\S+) \(verbatim\) ----\n", re.M)


def origin_blocks() -> dict[str, str]:
    text = ORIGIN_FILE.read_text(encoding="utf-8")
    matches = list(MARKER.finditer(text))
    blocks = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks[match.group(1)] = text[match.end():end].rstrip("\n") + "\n"
    return blocks


class _IO:
    """Stand-in for ``comfy_api.latest.io``: records the node schema instead of building a UI."""

    class ComfyNode:
        pass

    class Schema:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Inputs:
        def __init__(self, kind):
            self.kind = kind

        def Input(self, name, **options):  # noqa: N802 - upstream name
            return {"kind": self.kind, "name": name, **options}

    Int = _Inputs("int")
    Float = _Inputs("float")

    class Sigmas:
        @staticmethod
        def Output():  # noqa: N802 - upstream name
            return "SIGMAS"

    class NodeOutput:
        def __init__(self, *values):
            self.values = values


def load_origin():
    spec = importlib.util.spec_from_file_location("_origin_comfyui_laplace", ORIGIN_FILE)
    module = importlib.util.module_from_spec(spec)
    module.io = _IO
    module.k_diffusion_sampling = module      # the node calls k_diffusion_sampling.get_sigmas_laplace
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ORIGIN = load_origin()

CASES = [
    # (steps, sigma_min, sigma_max, mu, beta) — the node's output has no repeated sigma
    (20, 0.0291675, 14.614642, 0.0, 0.5),      # the node's defaults (SD/SDXL sigmas)
    (1, 0.0291675, 14.614642, 0.0, 0.5),
    (2, 0.0291675, 14.614642, 0.0, 0.5),
    (7, 0.0291675, 14.614642, 1.3, 0.25),
    (40, 0.001, 1.0, -3.5, 0.8),               # flow-model sigmas (Anima/Flux)
    (9, 0.001, 1.0, -2.0, 0.5),
]
# The node's clamp repeats a sigma here: the extension spreads the steps (None) or stops (an error).
REPEAT_CASES = [
    ((30, 0.0291675, 14.614642, -2.0, 2.0), None),
    ((16, 0.001, 1.0, -1.0, 0.5), None),
    ((28, 0.001, 1.0, 0.0, 0.5), None),        # flow sigmas with the default mu: half the steps at 1
    ((12, 0.0291675, 14.614642, 10.0, 10.0), None),   # the ends of the node's ranges
    ((25, 0.0291675, 14.614642, 0.0, 0.0), sch.ExtraSchedulerError),     # beta 0: every step at exp(mu)
    ((12, 0.0291675, 14.614642, -10.0, 0.1), sch.ExtraSchedulerError),   # the whole curve below sigma_min
]


class OriginCopyTests(unittest.TestCase):
    def test_upstream_blocks_are_verbatim(self):
        blocks = origin_blocks()
        self.assertEqual(set(blocks), set(BLOCK_SHA256))
        for key, digest in BLOCK_SHA256.items():
            with self.subTest(block=key):
                self.assertEqual(hashlib.sha256(blocks[key].encode("utf-8")).hexdigest(), digest)

    def test_header_names_the_commit_and_the_license(self):
        text = ORIGIN_FILE.read_text(encoding="utf-8")
        self.assertIn(ORIGIN_COMMIT, text)
        self.assertIn("GPL-3.0", text)
        for path, digest in FILE_SHA256.items():
            self.assertIn(path, text)
            self.assertIn(digest, text)

    def test_extension_copy_of_get_sigmas_laplace_is_unchanged(self):
        source = (ROOT / "sam3ext" / "extra_schedulers" / "schedulers.py").read_text(encoding="utf-8")
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "get_sigmas_laplace")
        ours = ast.get_source_segment(source, node) + "\n"
        self.assertEqual(ours, origin_blocks()["comfy/k_diffusion/sampling.py:52-59"])
        self.assertIn(f"origin: {ORIGIN_COMMIT}:comfy/k_diffusion/sampling.py:52-59", source)


class LaplaceParityTests(unittest.TestCase):
    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)

    def _ours(self, steps, sigma_min, sigma_max, mu, beta):
        st.set_active(st.ExtraSchedulerSettings(laplace_mu=mu, laplace_beta=beta))
        return sch.laplace(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device="cpu")

    def test_bit_for_bit_with_comfyui_plus_the_final_zero(self):
        for steps, lo, hi, mu, beta in CASES:
            with self.subTest(steps=steps, sigma_min=lo, sigma_max=hi, mu=mu, beta=beta):
                upstream = ORIGIN.get_sigmas_laplace(steps, lo, hi, mu=mu, beta=beta)
                self.assertFalse(bool((upstream[1:] == upstream[:-1]).any()), "a case without a repeat")
                ours = self._ours(steps, lo, hi, mu, beta)
                self.assertEqual(ours.dtype, upstream.dtype)
                self.assertTrue(torch.equal(ours[:-1], upstream))
                self.assertEqual(ours.shape, (steps + 1,))
                self.assertEqual(float(ours[-1]), 0.0)

    def test_where_the_node_repeats_a_sigma_the_steps_are_spread_or_refused(self):
        for (steps, lo, hi, mu, beta), error in REPEAT_CASES:
            with self.subTest(steps=steps, sigma_min=lo, sigma_max=hi, mu=mu, beta=beta):
                upstream = ORIGIN.get_sigmas_laplace(steps, lo, hi, mu=mu, beta=beta)
                self.assertTrue(bool((upstream[1:] == upstream[:-1]).any()), "the node repeats a sigma")
                if error is not None:
                    with self.assertRaises(error):
                        self._ours(steps, lo, hi, mu, beta)
                    continue
                ours = self._ours(steps, lo, hi, mu, beta)
                self.assertEqual(ours.shape, (steps + 1,))
                self.assertTrue(bool((ours[1:] < ours[:-1]).all()), "every step lowers sigma")
                # The same curve: every value is one the node's formula gives somewhere on 0 … 1.
                self.assertLessEqual(float(ours[0]), float(upstream[0]))
                self.assertGreaterEqual(float(ours[-2]), float(upstream[-1]))

    def test_through_the_comfyui_node(self):
        node = ORIGIN.LaplaceScheduler
        for steps, lo, hi, mu, beta in CASES:   # cases without a repeated sigma
            with self.subTest(steps=steps, mu=mu, beta=beta):
                output = node.execute(steps, hi, lo, mu, beta)
                (upstream,) = output.values
                self.assertEqual(upstream.shape, (steps,), "the node returns no final 0")
                self.assertTrue(torch.equal(self._ours(steps, lo, hi, mu, beta)[:-1], upstream))
        self.assertIs(node.get_sigmas.__func__, node.execute.__func__)   # the node's legacy entry point


class NodeInputParityTests(unittest.TestCase):
    """The accordion's Laplace values and sliders are the node's inputs."""

    @classmethod
    def setUpClass(cls):
        schema = ORIGIN.LaplaceScheduler.define_schema()
        cls.inputs = {item["name"]: item for item in schema.kwargs["inputs"]}

    def test_defaults_and_ranges(self):
        mu, beta = self.inputs["mu"], self.inputs["beta"]
        self.assertEqual((mu["default"], mu["min"], mu["max"]),
                         (st.LAPLACE_MU_DEFAULT, st.LAPLACE_MU_MIN, st.LAPLACE_MU_MAX))
        self.assertEqual((beta["default"], beta["min"], beta["max"]),
                         (st.LAPLACE_BETA_DEFAULT, st.LAPLACE_BETA_MIN, st.LAPLACE_BETA_MAX))
        self.assertEqual(st.DEFAULTS.laplace_mu, mu["default"])
        self.assertEqual(st.DEFAULTS.laplace_beta, beta["default"])

    def test_api_values_are_clamped_to_the_node_ranges(self):
        mu, beta = self.inputs["mu"], self.inputs["beta"]
        self.assertEqual(st.coerce_laplace_mu(99), mu["max"])
        self.assertEqual(st.coerce_laplace_mu(-99), mu["min"])
        self.assertEqual(st.coerce_laplace_beta(-1), beta["min"])
        self.assertEqual(st.coerce_laplace_beta(42), beta["max"])

    def test_sliders(self):
        with gr.Blocks():
            controls = ues.build_controls(lambda item: f"t_{item}")
        self.assertEqual(len(controls), len(ues.ARG_NAMES))
        mu_slider, beta_slider = (controls[ues.ARG_NAMES.index(name)] for name in ("laplace_mu", "laplace_beta"))
        for slider, spec in ((mu_slider, self.inputs["mu"]), (beta_slider, self.inputs["beta"])):
            with self.subTest(slider=slider.label):
                self.assertEqual(slider.value, spec["default"])
                self.assertEqual(slider.minimum, spec["min"])
                self.assertEqual(slider.maximum, spec["max"])
                self.assertEqual(slider.step, spec["step"])

    def test_node_defaults_for_the_sigma_range_are_sd_sigmas(self):
        # The node's sigma_max/min defaults are SD/SDXL's; in Forge they come from the model instead.
        self.assertAlmostEqual(self.inputs["sigma_max"]["default"], 14.614642)
        self.assertAlmostEqual(self.inputs["sigma_min"]["default"], 0.0291675)
        self.assertEqual(self.inputs["steps"]["default"], 20)


if __name__ == "__main__":
    unittest.main()
