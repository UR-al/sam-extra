"""Math and lifecycle invariants for the optional Anima Guidance Suite."""

from __future__ import annotations

import unittest

import torch

from sam3ext.guidance.cns import color_noise_wavelet
from sam3ext.guidance.cwm_smc import (
    apply_cwm_error,
    apply_smc_error,
    compose_cfg,
)
from sam3ext.guidance.dave import apply_dave
from sam3ext.guidance.dcw import apply_dcw
from sam3ext.guidance.haar import haar_dwt2d, haar_idwt2d, pad_even
from sam3ext.guidance.runtime import GuidanceRuntime


class GuidanceMathTests(unittest.TestCase):
    def _assert_haar_roundtrip(self, shape):
        source = torch.randn(*shape)
        padded, (height, width) = pad_even(source)
        reconstructed = haar_idwt2d(*haar_dwt2d(padded))[
            ..., :height, :width
        ]

        self.assertEqual(reconstructed.shape, source.shape)
        self.assertEqual(reconstructed.dtype, source.dtype)
        torch.testing.assert_close(reconstructed, source, rtol=1e-6, atol=1e-6)

    def test_haar_roundtrip_4d_even_and_odd(self):
        self._assert_haar_roundtrip((2, 4, 8, 10))
        self._assert_haar_roundtrip((2, 4, 7, 9))

    def test_haar_roundtrip_5d_odd(self):
        self._assert_haar_roundtrip((1, 3, 2, 7, 9))

    def test_dcw_zero_is_bitwise_identity(self):
        denoised = torch.randn(1, 4, 7, 9)
        live = torch.randn_like(denoised)

        out = apply_dcw(denoised, live, torch.tensor([1.0]), 0.0, 0.0)

        self.assertIs(out, denoised)

    def test_dcw_supports_4d_and_5d_odd_latents(self):
        for shape in ((1, 4, 7, 9), (1, 4, 2, 7, 9)):
            denoised = torch.randn(*shape)
            live = torch.randn_like(denoised)

            out = apply_dcw(
                denoised, live, torch.tensor([1.0]), 0.1, 0.02
            )

            self.assertEqual(out.shape, denoised.shape)
            self.assertEqual(out.dtype, denoised.dtype)
            self.assertTrue(torch.isfinite(out).all())

    def test_cwm_zero_alphas_matches_standard_cfg_error(self):
        error = torch.randn(2, 4, 7, 9)

        out = apply_cwm_error(
            error,
            sigma=torch.tensor([1.0, 0.5]),
            effective_scale=3.5,
            alpha_low=0.0,
            alpha_high=0.0,
        )

        torch.testing.assert_close(out, error * 3.5)

    def test_smc_zero_parameters_are_neutral(self):
        error = torch.randn(2, 4, 5, 7)
        previous = torch.randn_like(error)

        out, new_state = apply_smc_error(
            error, previous, lambda_value=0.0, k_value=0.0
        )

        self.assertIs(out, error)
        torch.testing.assert_close(new_state, error)

    def test_neutral_cwm_and_smc_compose_as_standard_cfg(self):
        cond = torch.randn(2, 4, 5, 7)
        uncond = torch.randn_like(cond)
        scale = 4.25

        for mode in ("cwm", "smc", "smc+cwm"):
            out, _ = compose_cfg(
                cond=cond,
                uncond=uncond,
                sigma=torch.tensor([1.0, 0.5]),
                effective_scale=scale,
                mode=mode,
                alpha_low=0.0,
                alpha_high=0.0,
                smc_lambda=0.0,
                smc_k=0.0,
                smc_previous=None,
            )
            torch.testing.assert_close(
                out, uncond.float() + scale * (cond.float() - uncond.float())
            )

    def test_smc_output_stays_finite(self):
        error = torch.randn(2, 16, 7, 9)

        out, state = apply_smc_error(
            error, None, lambda_value=6.0, k_value=0.2
        )

        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(state).all())

    def test_smc_and_cwm_zero_nonfinite_values_like_reference(self):
        error = torch.tensor([[[[float("nan"), float("inf"), -float("inf")]]]])

        smc_out, state = apply_smc_error(
            error, None, lambda_value=6.0, k_value=0.2
        )
        cwm_out = apply_cwm_error(
            error,
            sigma=torch.tensor([1.0]),
            effective_scale=4.0,
            alpha_low=0.3,
            alpha_high=0.15,
        )

        self.assertTrue(torch.isfinite(smc_out).all())
        self.assertTrue(torch.isfinite(state).all())
        self.assertTrue(torch.isfinite(cwm_out).all())
        self.assertEqual(torch.count_nonzero(smc_out).item(), 0)
        self.assertEqual(torch.count_nonzero(cwm_out).item(), 0)

    def test_dave_zero_is_bitwise_identity_and_mean_is_attenuated(self):
        source = torch.randn(2, 3, 5, 7)
        self.assertIs(apply_dave(source, 0.0), source)

        attenuation = 0.3
        expected = source - attenuation * source.mean(
            dim=(1, 2), keepdim=True
        )
        out = apply_dave(source, attenuation)

        torch.testing.assert_close(out, expected)

    def test_cns_zero_is_identity_and_coloring_is_deterministic(self):
        noise = torch.randn(1, 4, 7, 9)
        live = torch.randn_like(noise)
        self.assertIs(color_noise_wavelet(noise, live, strength=0.0), noise)

        first = color_noise_wavelet(noise, live, strength=1.0)
        second = color_noise_wavelet(noise, live, strength=1.0)
        partial = color_noise_wavelet(noise, live, strength=0.5)

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, noise))
        self.assertAlmostEqual(
            float(first.std()), float(noise.std()), delta=1e-5
        )
        self.assertAlmostEqual(
            float(partial.std()), float(noise.std()), delta=1e-5
        )

    def test_cns_recoloring_does_not_consume_rng(self):
        noise = torch.randn(1, 4, 7, 9)
        live = torch.randn_like(noise)
        before = torch.random.get_rng_state().clone()

        color_noise_wavelet(noise, live, strength=1.0)

        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))


class GuidanceRuntimeTests(unittest.TestCase):
    def test_reset_pass_releases_all_generation_sized_tensors(self):
        state = {
            "attn_raw": torch.ones(1),
            "slg_raw": torch.ones(1),
            "adg_skipped": True,
            "attn_spatial_shape": (1, 2, 3),
            "step_open": True,
        }
        apg = {"avg": torch.ones(1), "last_sigma": 1.0}
        runtime = GuidanceRuntime(
            state=state,
            apg=apg,
            adg={},
            smc_prev=torch.ones(1),
            cns_x_t=torch.ones(1),
            cns_noise_calls=4,
        )

        runtime.reset_pass()

        self.assertIsNone(apg["avg"])
        self.assertIsNone(apg["last_sigma"])
        self.assertIsNone(runtime.smc_prev)
        self.assertIsNone(runtime.cns_x_t)
        self.assertEqual(runtime.cns_noise_calls, 0)
        self.assertIsNone(state["attn_raw"])
        self.assertIsNone(state["slg_raw"])
        self.assertFalse(state["adg_skipped"])
        self.assertIsNone(state["attn_spatial_shape"])
        self.assertFalse(state["step_open"])
        self.assertEqual(state["active"], 0)


if __name__ == "__main__":
    unittest.main()
