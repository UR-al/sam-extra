"""Experimental PAG strength envelope (Forge option sam3_guidance_pag_cosine_envelope, default off).

This extension's own curve, not a published PAG variant: ``sin²(π·u)`` over PAG's linear sigma
window, 0 at both ends and the configured scale in the middle; PAG term only (SEG/SLG unchanged).
From the 2026-10-02 review (docs/review_proposals_20261002/pag_envelope).
"""

from __future__ import annotations

import math
import types
import unittest
from unittest import mock

import gradio as gr
import torch

from tests.test_anima_safe_pag import _load_pag_module, _load_real_forge_anima


class PagEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _load_pag_module()

    def setUp(self):
        self.saved = dict(self.pag._STATE)
        self.addCleanup(self._restore)

    def _restore(self):
        self.pag._STATE.clear()
        self.pag._STATE.update(self.saved)

    def _window(self, on=True, hi=10.0, lo=2.0):
        self.pag._STATE.update(pag_cosine_envelope=on, sigma_hi=hi, sigma_lo=lo)

    def test_factor_is_sin_squared_over_the_linear_sigma_window(self):
        self._window()
        factor = self.pag._pag_envelope_factor
        for sigma, expected in ((10.0, 0.0), (8.0, 0.5), (6.0, 1.0), (4.0, 0.5), (2.0, 0.0),
                                (11.0, 0.0), (1.0, 0.0)):
            with self.subTest(sigma=sigma):
                self.assertAlmostEqual(factor(sigma), expected, places=12)
        self.assertAlmostEqual(factor(torch.tensor([7.0, 3.0])), math.sin(math.pi * 3 / 8) ** 2)
        self._window(hi=2.0, lo=10.0)   # reversed bounds are swapped
        self.assertAlmostEqual(factor(6.0), 1.0)

    def test_off_or_unusable_bounds_keep_the_constant_strength(self):
        self._window(on=False)
        self.assertEqual(self.pag._pag_envelope_factor(10.0), 1.0)
        for hi, lo, sigma in ((None, 2.0, 5.0), (10.0, None, 5.0), (5.0, 5.0, 5.0),
                              (float("nan"), 2.0, 5.0), (10.0, 2.0, float("inf"))):
            with self.subTest(hi=hi, lo=lo, sigma=sigma):
                self._window(hi=hi, lo=lo)
                self.assertEqual(self.pag._pag_envelope_factor(sigma), 1.0)

    def test_window_ends_get_no_pag_weak_rows(self):
        p = self.pag
        p._STATE.update(attn_method="pag", attn_scale=3.0, attn_targets={1})
        self._window()
        self.assertTrue(p._attn_rows_in_range(torch.tensor([6.0])))
        self.assertFalse(p._attn_rows_in_range(torch.tensor([10.0])), "factor 0 at the upper end")
        self.assertFalse(p._attn_rows_in_range(torch.tensor([2.0])), "factor 0 at the lower end")
        self._window(on=False)
        self.assertTrue(p._attn_rows_in_range(torch.tensor([10.0])), "off: the inclusive window as before")
        p._STATE.update(attn_method="seg")
        self._window()
        with mock.patch.object(p, "_percent_in_range", return_value=True):
            self.assertTrue(p._attn_rows_in_range(torch.tensor([10.0])), "SEG keeps its own gate")

    def test_only_the_pag_term_is_scaled(self):
        p = self.pag
        torch.manual_seed(0)
        cond = torch.randn(1, 4, 6, 6)
        weak_attn = torch.randn(1, 4, 6, 6)
        weak_slg = torch.randn(1, 4, 6, 6)
        base = torch.randn(1, 4, 6, 6)
        p._STATE.update(attn_method="pag", attn_scale=3.0, slg_scale=2.0, rescale=0.0,
                        cond_raw=cond, attn_raw=weak_attn, slg_raw=weak_slg, delta_logged=True)
        args = {"cond_denoised": cond, "sigma": torch.tensor([8.0])}
        self._window()
        got = p._apply_perturbation(args, base)
        expected = base + 3.0 * 0.5 * (cond - weak_attn) + 2.0 * (cond - weak_slg)
        torch.testing.assert_close(got, expected)
        self._window(on=False)
        torch.testing.assert_close(
            p._apply_perturbation(args, base), base + 3.0 * (cond - weak_attn) + 2.0 * (cond - weak_slg),
        )
        p._STATE.update(attn_method="seg")
        self._window()
        torch.testing.assert_close(
            p._apply_perturbation(args, base), base + 3.0 * (cond - weak_attn) + 2.0 * (cond - weak_slg),
            msg="SEG is not enveloped",
        )

    def test_infotext_records_the_option_and_whether_a_sigma_window_exists(self):
        p = self.pag
        record = []
        anima = _load_real_forge_anima(record)
        dit = anima.Anima(in_channels=4, out_channels=4, patch_spatial=2, patch_temporal=1,
                          model_channels=48, crossattn_emb_channels=32, adaln_lora_dim=16,
                          num_blocks=6, num_heads=4)

        def run(predictor):
            class Unet:
                def __init__(self):
                    self.model = types.SimpleNamespace(diffusion_model=dit, predictor=predictor)
                    self.model_options = {}

                def clone(self):
                    return Unet()

                def set_model_unet_function_wrapper(self, fn):
                    pass

                def set_model_sampler_pre_cfg_function(self, fn):
                    pass

                def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
                    pass

            Anima = type("Anima", (), {})
            model = Anima()
            model.forge_objects = types.SimpleNamespace(unet=Unet())
            request = types.SimpleNamespace(sd_model=model, extra_generation_params={}, steps=20)
            process = p.AnimaSafePAG()
            with gr.Blocks():
                inputs = process.ui(False)
            args = [component.value for component in inputs]
            args[0] = True
            args[4] = "3"
            stub = types.SimpleNamespace(opts=types.SimpleNamespace(**{p.OPT_PAG_COSINE: True}))
            try:
                with mock.patch.object(p, "shared", stub):
                    process.process_before_every_sampling(request, *args)
                return dict(request.extra_generation_params), p._STATE["pag_cosine_envelope"]
            finally:
                process.postprocess(request, None)
                p._teardown_global_patches()

        flow = types.SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent)
        params, state = run(flow)
        self.assertTrue(state)
        self.assertEqual(params[p.INFOTEXT_PAG_COSINE], "True")
        self.assertEqual(params[p.INFOTEXT_PAG_ENVELOPE_EFFECTIVE], "cosine-squared in linear sigma window")
        params, _ = run(types.SimpleNamespace())
        self.assertEqual(params[p.INFOTEXT_PAG_ENVELOPE_EFFECTIVE],
                         "constant fallback: no usable sigma window")


if __name__ == "__main__":
    unittest.main()
