"""sam3ext.speed.spectral — the pure-torch DCT/FFT/Haar used by SPEED at runtime.

scipy and PyWavelets are only used here, as references: the extension never imports them for SPEED.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.speed import spectral  # noqa: E402

try:
    from scipy.fft import dctn, idctn
except ImportError:  # pragma: no cover - scipy ships with the Forge venv
    dctn = idctn = None

try:
    import pywt
except ImportError:  # pragma: no cover
    pywt = None


@unittest.skipIf(dctn is None, "scipy not installed")
class DctAgainstScipyTests(unittest.TestCase):
    def test_dct2_and_idct2_match_scipy_ortho_type2(self):
        rng = np.random.default_rng(0)
        for shape in ((8, 8), (3, 13, 7), (2, 4, 16, 24), (1, 4, 1, 21, 9), (5, 1)):
            with self.subTest(shape=shape):
                a = rng.standard_normal(shape)
                x = torch.from_numpy(a)
                np.testing.assert_allclose(spectral.dct2(x).numpy(), dctn(a, type=2, norm="ortho", axes=(-2, -1)), atol=1e-12)
                np.testing.assert_allclose(spectral.idct2(x).numpy(), idctn(a, type=2, norm="ortho", axes=(-2, -1)), atol=1e-12)

    def test_float32_inputs_use_a_float32_basis(self):
        x = torch.randn(2, 3, 10, 6)
        out = spectral.dct2(x)
        self.assertEqual(out.dtype, torch.float32)
        np.testing.assert_allclose(out.numpy(), dctn(x.numpy(), type=2, norm="ortho", axes=(-2, -1)), atol=2e-5)

    def test_dct_downscale_is_scipy_truncation(self):
        a = np.random.default_rng(1).standard_normal((2, 4, 30, 22)).astype(np.float32)
        for size in ((15, 11), (11, 8), (30, 22), (1, 1)):
            with self.subTest(size=size):
                h, w = size
                expected = idctn(dctn(a, type=2, norm="ortho", axes=(-2, -1))[..., :h, :w], type=2, norm="ortho", axes=(-2, -1))
                out = spectral.dct_downscale(torch.from_numpy(a), size)
                self.assertEqual(out.dtype, torch.float64)          # work dtype off MPS
                np.testing.assert_allclose(out.numpy(), expected, atol=1e-5)
        with self.assertRaises(ValueError):
            spectral.dct_downscale(torch.zeros(1, 1, 4, 4), (8, 4))


class BasisCacheTests(unittest.TestCase):
    def setUp(self):
        spectral.clear_basis_cache()

    def test_same_tensor_per_size_device_dtype(self):
        a = spectral.dct_basis(16, "cpu", torch.float64)
        self.assertIs(a, spectral.dct_basis(16, torch.device("cpu"), torch.float64))
        self.assertIsNot(a, spectral.dct_basis(16, "cpu", torch.float32))
        self.assertIsNot(a, spectral.dct_basis(17, "cpu", torch.float64))
        identity = a @ a.transpose(0, 1)
        torch.testing.assert_close(identity, torch.eye(16, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_cache_is_bounded(self):
        for n in range(1, 40):
            spectral.dct_basis(n, "cpu", torch.float64)
        self.assertLessEqual(len(spectral._BASIS_CACHE), spectral._BASIS_CACHE_LIMIT)
        self.assertIn((39, "cpu", torch.float64), spectral._BASIS_CACHE)    # most recent kept

    def test_work_dtype(self):
        self.assertEqual(spectral.work_dtype("cpu"), torch.float64)
        self.assertEqual(spectral.work_dtype(torch.device("mps")), torch.float32)

    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            spectral.dct_basis(0)


class ExpansionTests(unittest.TestCase):
    def test_dct_expand_keeps_the_low_block_and_fills_the_rest(self):
        x = torch.randn(1, 2, 6, 5, dtype=torch.float64)
        noise = torch.randn(1, 2, 12, 10, dtype=torch.float64)
        out = spectral.expand_dct(x, (12, 10), 0.4, noise)
        coeffs = spectral.dct2(out)
        torch.testing.assert_close(coeffs[..., :6, :5], spectral.dct2(x))
        expected_hf = 0.4 * noise
        mask = torch.ones(12, 10, dtype=torch.bool)
        mask[:6, :5] = False
        torch.testing.assert_close(coeffs[..., mask], expected_hf[..., mask])

    def test_zero_sigma_is_band_limited_upsampling_with_amplitude_one_over_r(self):
        flat = torch.full((1, 1, 8, 8), 3.0, dtype=torch.float64)
        out = spectral.expand_dct(flat, (16, 16), 0.0, torch.zeros(1, 1, 16, 16))
        torch.testing.assert_close(out, torch.full((1, 1, 16, 16), 1.5, dtype=torch.float64))   # 3 / r, r = 2

    def test_white_noise_round_trip_stays_white(self):
        """T_Phi of unit white noise, expanded at sigma = 1 with fresh unit noise: unit white noise again
        (kappa(1, r) = 1), the property behind the paper's alignment."""
        g = torch.Generator().manual_seed(0)
        x = torch.randn(64, 4, 32, 32, generator=g, dtype=torch.float64)
        coarse = spectral.dct_downscale(x, (16, 16))
        self.assertAlmostEqual(float(coarse.var()), 1.0, delta=0.02)
        expanded = spectral.expand_dct(coarse, (32, 32), 1.0, torch.randn(64, 4, 32, 32, generator=g, dtype=torch.float64))
        out = spectral.kappa(1.0, 2.0) * expanded
        self.assertAlmostEqual(float(out.var()), 1.0, delta=0.02)
        self.assertAlmostEqual(float(out.mean()), 0.0, delta=0.01)

    def test_alignment_reproduces_the_flow_form(self):
        """Expanding (1-t) y + t eps at time t and scaling by kappa gives (1-t~) up(y) + t~ eps' (Eq. 5-6)."""
        t, r = 0.8, 2.0
        t_tilde = spectral.align_timestep(t, r)
        # Signal part, exactly: kappa (1 - t) / r = 1 - t~.
        y = torch.linspace(-1, 1, 16, dtype=torch.float64).view(1, 1, 4, 4)
        signal = spectral.kappa(t, r) * spectral.expand_dct((1 - t) * y, (8, 8), t, torch.zeros(1, 1, 8, 8))
        torch.testing.assert_close(signal, (1 - t_tilde) * spectral.amplitude_resize(y, (8, 8)))
        # Noise part: unit coarse noise at t plus the sigma-scaled fill is white with variance t~**2.
        g = torch.Generator().manual_seed(3)
        eps = torch.randn(4096, 1, 4, 4, generator=g, dtype=torch.float64)
        noise = torch.randn(4096, 1, 8, 8, generator=g, dtype=torch.float64)
        out = spectral.kappa(t, r) * spectral.expand_dct(t * eps, (8, 8), t, noise)
        self.assertAlmostEqual(float(out.var()), t_tilde ** 2, delta=0.01)
        per_pixel = out.var(dim=0)
        self.assertLess(float((per_pixel - t_tilde ** 2).abs().max()), 0.06)    # white: no pixel left out

    @unittest.skipIf(pywt is None, "PyWavelets not installed")
    def test_haar_matches_pywt_waverec2(self):
        rng = np.random.default_rng(4)
        ll = rng.standard_normal((3, 5, 7))
        lh, hl, hh = (rng.standard_normal((3, 5, 7)) for _ in range(3))
        sigma = 0.6
        expected = np.stack([
            pywt.waverec2([ll[i], (sigma * lh[i], sigma * hl[i], sigma * hh[i])], "haar", mode="periodization")
            for i in range(3)
        ])
        out = spectral.expand_haar(torch.from_numpy(ll), sigma, *(torch.from_numpy(a) for a in (lh, hl, hh)))
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-12)

    def test_fft_matches_the_official_numpy_formula(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal((2, 6, 9)).astype(np.float32)
        nr = rng.standard_normal((2, 12, 15)).astype(np.float32)
        ni = rng.standard_normal((2, 12, 15)).astype(np.float32)
        t = 0.55
        expected = []
        for i in range(2):     # official utils.py _fft_expand_np @ ca7801c9, one channel at a time
            x_src = np.fft.fftshift(np.fft.fft2(x[i], norm="ortho"))
            x_big = np.fft.fftshift(t * (nr[i] + 1j * ni[i]))
            x_big[3:3 + 6, 3:3 + 9] = x_src
            expected.append(np.fft.ifft2(np.fft.ifftshift(x_big), norm="ortho").real)
        out = spectral.expand_fft(torch.from_numpy(x), (12, 15), t, torch.from_numpy(nr), torch.from_numpy(ni))
        np.testing.assert_allclose(out.numpy(), np.stack(expected), atol=1e-5)

    def test_targets_smaller_than_the_source_are_refused(self):
        with self.assertRaises(ValueError):
            spectral.expand_dct(torch.zeros(1, 1, 8, 8), (4, 8), 0.5, torch.zeros(1, 1, 4, 8))
        with self.assertRaises(ValueError):
            spectral.expand_fft(torch.zeros(1, 1, 8, 8), (8, 4), 0.5, torch.zeros(1, 1, 8, 4), torch.zeros(1, 1, 8, 4))
        with self.assertRaises(ValueError):
            spectral.expand_dct(torch.zeros(1, 1, 4, 4), (8, 8), 0.5, torch.zeros(1, 2, 8, 8))


class AmplitudeAndAlignmentTests(unittest.TestCase):
    def test_amplitude_resize_keeps_the_mean(self):
        x = torch.randn(2, 3, 9, 12, dtype=torch.float64) + 2.0
        for size in ((18, 24), (5, 6), (9, 12)):
            with self.subTest(size=size):
                out = spectral.amplitude_resize(x, size)
                self.assertEqual(tuple(out.shape[-2:]), size)
                torch.testing.assert_close(out.mean(dim=(-2, -1)), x.mean(dim=(-2, -1)))

    def test_kappa_and_alignment(self):
        self.assertAlmostEqual(spectral.kappa(0.0, 2.0), 2.0)
        self.assertAlmostEqual(spectral.kappa(1.0, 2.0), 1.0)
        for t in (0.1, 0.5, 0.93):
            for r in (1.5, 2.0, 4.0):
                self.assertAlmostEqual(spectral.align_timestep(t, r), t * spectral.kappa(t, r))
                aligned, factor = spectral.respace_alignment(t, r)
                self.assertAlmostEqual(aligned, spectral.align_timestep(t, r), places=14)
                self.assertAlmostEqual(factor, spectral.kappa(t, r), places=14)
                self.assertTrue(t < aligned < 1.0)

    def test_snap(self):
        self.assertEqual(spectral.snap(62.5, 2), 62)     # round-half-even like upstream
        self.assertEqual(spectral.snap(0.3, 2), 2)
        self.assertEqual(spectral.snap(64, 2), 64)


if __name__ == "__main__":
    unittest.main()
