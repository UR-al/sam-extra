from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_skim_module():
    """Load the extension script without booting the full WebUI."""
    modules_stub = types.ModuleType("modules")

    class Script:
        pass

    modules_stub.scripts = types.SimpleNamespace(
        Script=Script,
        AlwaysVisible=object(),
        scripts_data=[],
    )
    modules_stub.shared = types.SimpleNamespace(
        state=types.SimpleNamespace(sampling_step=0, sampling_steps=20)
    )

    old_modules = sys.modules.get("modules")
    sys.modules["modules"] = modules_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "_test_anima_skimmed_cfg", ROOT / "scripts" / "anima_skimmed_cfg.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if old_modules is None:
            sys.modules.pop("modules", None)
        else:
            sys.modules["modules"] = old_modules


def _reference_skim(x_orig, cond, uncond, cond_scale, skimming_scale,
                    disable_flipping_filter=False):
    """Upstream Extraltodeus/Skimmed_CFG maths, transcribed for comparison."""
    cond = cond.clone()
    denoised = x_orig - (
        (x_orig - uncond) + cond_scale * ((x_orig - cond) - (x_orig - uncond))
    )
    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    matching_diff_after = (
        cond.sign() == (cond * cond_scale - uncond * (cond_scale - 1)).sign()
    )
    if disable_flipping_filter:
        outer_influence = matching_pred_signs & matching_diff_after
    else:
        deviation_influence = denoised.sign() == (denoised - x_orig).sign()
        outer_influence = (
            matching_pred_signs & matching_diff_after & deviation_influence
        )
    low_cfg_denoised_outer = x_orig - (
        (x_orig - uncond)
        + skimming_scale * ((x_orig - cond) - (x_orig - uncond))
    )
    difference = denoised - low_cfg_denoised_outer
    cond[outer_influence] = cond[outer_influence] - (
        difference[outer_influence] / cond_scale
    )
    return cond


class SkimmedCFGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skim = _load_skim_module()

    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(1, 4, 8, 8)
        self.cond = torch.randn(1, 4, 8, 8)
        self.uncond = torch.randn(1, 4, 8, 8)
        self.skim._SKIM.update(
            on=False, skimming_cfg=7.0, full_skim_negative=False,
            disable_flipping_filter=False, start=0.0, end=1.0, flip_at=0.0,
            steps=0, warned=False,
        )

    def _args(self, cond_scale=8.0):
        denoised = self.uncond + cond_scale * (self.cond - self.uncond)
        return {
            "denoised": denoised,
            "input": self.x,
            "cond_denoised": self.cond,
            "uncond_denoised": self.uncond,
            "cond_scale": cond_scale,
        }

    def test_matches_upstream_reference_maths(self):
        for flip in (False, True):
            with self.subTest(disable_flipping_filter=flip):
                out = self.skim._skim_predictions(
                    self.x, self.cond, self.uncond, 8.0, 7.0, flip
                )
                expected = _reference_skim(
                    self.x, self.cond, self.uncond, 8.0, 7.0, flip
                )
                torch.testing.assert_close(out, expected)

    def test_predictions_are_not_mutated_in_place(self):
        cond_before = self.cond.clone()
        self.skim._skim_predictions(
            self.x, self.cond, self.uncond, 8.0, 7.0, False
        )
        torch.testing.assert_close(self.cond, cond_before)

    def test_equal_scales_leave_predictions_untouched(self):
        """skimming_cfg == cond_scale has nothing to pull back."""
        out = self.skim._skim_predictions(
            self.x, self.cond, self.uncond, 8.0, 8.0, False
        )
        torch.testing.assert_close(out, self.cond)

    def test_disabled_hook_preserves_incoming(self):
        args = self._args()
        out = self.skim._post_cfg(args)
        self.assertIs(out, args["denoised"])

    def test_enabled_hook_changes_the_result(self):
        self.skim._SKIM.update(on=True)
        args = self._args()
        out = self.skim._post_cfg(args)
        self.assertEqual(out.shape, args["denoised"].shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertFalse(torch.allclose(out, args["denoised"]))
        self.assertEqual(self.skim._SKIM["steps"], 1)

    def test_cfg_scale_one_is_skipped(self):
        self.skim._SKIM.update(on=True)
        args = self._args(cond_scale=1.0)
        out = self.skim._post_cfg(args)
        self.assertIs(out, args["denoised"])

    def test_percent_range_gates_the_hook(self):
        self.skim._SKIM.update(on=True, start=0.5, end=1.0)
        args = self._args()
        # shared.state stub reports step 0 of 20 -> 0%
        self.assertIs(self.skim._post_cfg(args), args["denoised"])

    def test_negative_skimming_cfg_follows_the_live_scale(self):
        self.skim._SKIM.update(on=True, skimming_cfg=-1.0)
        args = self._args(cond_scale=8.0)
        out = self.skim._post_cfg(args)
        self.assertTrue(torch.isfinite(out).all())

    def test_full_skim_negative_runs(self):
        self.skim._SKIM.update(on=True, full_skim_negative=True)
        args = self._args()
        out = self.skim._post_cfg(args)
        self.assertTrue(torch.isfinite(out).all())

    def test_skim_is_published_to_downstream_guidance(self):
        """Upstream is a pre-CFG node, so later guidance must see the skim."""
        # skimming_cfg 3 < cond_scale - 1, so BOTH predictions get skimmed;
        # at the default 7 with CFG 8 the positive is skimmed at an identical
        # scale and only the negative changes.
        self.skim._SKIM.update(on=True, skimming_cfg=3.0)
        args = self._args()
        cond_before = args["cond_denoised"].clone()
        uncond_before = args["uncond_denoised"].clone()

        out = self.skim._post_cfg(args)

        # Forge reuses these tensors for every later post-CFG consumer.
        self.assertFalse(torch.allclose(args["cond_denoised"], cond_before))
        self.assertFalse(torch.allclose(args["uncond_denoised"], uncond_before))
        # A downstream consumer rebuilding the base from the published
        # predictions must land on our result, not on the unskimmed one.
        rebuilt = args["uncond_denoised"] + 8.0 * (
            args["cond_denoised"] - args["uncond_denoised"]
        )
        torch.testing.assert_close(rebuilt, out)

    def test_gated_out_steps_publish_nothing(self):
        self.skim._SKIM.update(on=True, start=0.5, end=1.0)
        args = self._args()
        cond_before = args["cond_denoised"].clone()

        self.skim._post_cfg(args)

        torch.testing.assert_close(args["cond_denoised"], cond_before)

    def test_missing_uncond_preserves_incoming(self):
        self.skim._SKIM.update(on=True)
        args = self._args()
        args["uncond_denoised"] = torch.zeros(1, 4, 4, 4)  # shape mismatch
        self.assertIs(self.skim._post_cfg(args), args["denoised"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
