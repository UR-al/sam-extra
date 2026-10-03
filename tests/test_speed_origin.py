"""Origin-parity tests for SPEED (sam3ext/speed) against verbatim upstream copies, on the CPU.

Oracles (each file pins the SHA-256 of its upstream text below):

* ``tests/_origin_speed_spectral_utils.py`` — aoleg/ComfyUI-SPEED@a8873591 ``spectral_utils.py`` with
  the one-line FFT fix upstream howardhx/speed made in ca7801c9 (documented in its header);
* ``tests/_origin_speed_core.py`` — aoleg ``speed_core.py`` (presets, adaptive delta, neo_shift
  divisor, ``_resolve_transitions``, ``sample_speed_core``), loaded in a synthetic package so its
  ``.spectral_utils`` import is the fixed oracle above;
* ``tests/_origin_speed_official_utils.py`` — howardhx/speed@ca7801c9 ``utils.py`` (the fixed FFT);
* ``tests/_origin_spd_core.py`` / ``tests/_origin_spd_sampler.py`` — sorryhyun/ComfyUI-Spectrum-
  KSampler@b46a364a ``spd_core.py`` (``spectral_expand``, ``dct_lowpass_init``) and ``spd.py``
  (``make_speed_sampler``'s respace loop, ``comfy`` stubbed).

What must match: presets and sigma* maths, the transition step lookup (adaptive caps, divisor,
manual), the transition-mode run (segment shapes, patched sigmas exactly, output to float precision:
upstream's DCT is float32 scipy, ours float64 torch), the FFT/DWT expansions, and the respace-mode
run step by step (model inputs, sigmas exactly, output). Batches differ on purpose (per-image seeds):
each image of our batch equals the upstream run of that image's seed alone.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.speed import runner, schedule, spectral  # noqa: E402

ORACLES = {
    # file: (marker, sha256 of the text after the marker)
    "_origin_speed_spectral_utils.py": (
        "# ---- upstream spectral_utils.py below (verbatim except the FFT line) ----",
        "f335f3361216d2bf784fdb22c067cb927250193b7e792554e2e11d388a51ab0f",
    ),
    "_origin_speed_core.py": (
        "# ---- upstream speed_core.py below (verbatim) ----",
        "db7507bc4920a9a030d7a5c3bbe2defc935a2375536571c8c26f4c61dd3bf10a",
    ),
    "_origin_speed_official_utils.py": (
        "# ---- upstream utils.py below (verbatim) ----",
        "04ee4607eaff90350bcdaabe62d16f2cca5100128b0c97c3173cc381e3df222a",
    ),
    "_origin_spd_core.py": (
        "# ---- upstream _vendor/networks/spd_core.py below (verbatim) ----",
        "3a029e7eac90fe259fff7fec82ec703af8308e0f827533358f15a3a5bd020484",
    ),
    "_origin_spd_sampler.py": (
        "# ---- upstream spd.py below (verbatim) ----",
        "46abefeec611c65df2fbb9ef058626df4730b1c9ca043147fef928f7b349f533",
    ),
}
# aoleg/ComfyUI-SPEED@a8873591:spectral_utils.py as published (before the one-line FFT fix).
AOLEG_SPECTRAL_UTILS_UPSTREAM_SHA256 = "958da38a7aa9e144ce0b8d2ac30806136de193ac4d8e95a82907d081508c2838"
FFT_FIXED_LINE = "        X_big = np.fft.fftshift(t * (nr + 1j * ni))\n"
FFT_UPSTREAM_LINE = "        X_big = np.fft.fftshift(t * (nr + 1j * ni) / np.sqrt(2.0))\n"


def _body(name: str) -> str:
    marker, _ = ORACLES[name]
    text = (TESTS / name).read_bytes().decode("utf-8").replace("\r\n", "\n")
    _, sep, body = text.partition(marker + "\n")
    if not sep:
        raise AssertionError(f"{name}: marker line missing")
    return body


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _stub_modules(stubs):
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_aoleg():
    """aoleg ``speed_core`` inside a synthetic package (its ``.spectral_utils`` = the fixed oracle)."""
    pkg_name = "_speed_origin_pkg"
    if f"{pkg_name}.speed_core" in sys.modules:
        return sys.modules[f"{pkg_name}.spectral_utils"], sys.modules[f"{pkg_name}.speed_core"]
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(TESTS)]
    sys.modules[pkg_name] = pkg
    spectral_utils = _load(f"{pkg_name}.spectral_utils", TESTS / "_origin_speed_spectral_utils.py")
    core = _load(f"{pkg_name}.speed_core", TESTS / "_origin_speed_core.py")
    return spectral_utils, core


def _load_official():
    try:
        import pywt  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:  # pragma: no cover - both ship with the Forge venv
        raise unittest.SkipTest("PyWavelets / PyYAML not installed") from None
    if "_speed_origin_official_utils" in sys.modules:
        return sys.modules["_speed_origin_official_utils"]
    return _load("_speed_origin_official_utils", TESTS / "_origin_speed_official_utils.py")


def _load_sorryhyun():
    if "_speed_origin_spd" in sys.modules:
        return sys.modules["_speed_origin_spd_core"], sys.modules["_speed_origin_spd"]
    core = _load("_speed_origin_spd_core", TESTS / "_origin_spd_core.py")
    networks = types.ModuleType("networks")
    networks.__path__ = []
    networks.spd_core = core
    with _stub_modules({"networks": networks, "networks.spd_core": core}):
        spd = _load("_speed_origin_spd", TESTS / "_origin_spd_sampler.py")
    return core, spd


def _kd_to_d(x, sigma, denoised):
    # origin: comfy/k_diffusion/sampling.py to_d (= Forge modules_forge/packages/k_diffusion/sampling.py:53-55)
    return (x - denoised) / sigma.reshape(sigma.shape + (1,) * (x.ndim - sigma.ndim))


def _comfy_stubs():
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    samplers = types.ModuleType("comfy.samplers")
    samplers.KSAMPLER = lambda fn, **kwargs: types.SimpleNamespace(sampler_function=fn)
    k_diffusion = types.ModuleType("comfy.k_diffusion")
    k_diffusion.__path__ = []
    kds = types.ModuleType("comfy.k_diffusion.sampling")
    kds.to_d = _kd_to_d
    utils = types.ModuleType("comfy.utils")
    utils.model_trange = lambda n, disable=None: range(n)
    comfy.samplers, comfy.k_diffusion, comfy.utils = samplers, k_diffusion, utils
    k_diffusion.sampling = kds
    return {
        "comfy": comfy, "comfy.samplers": samplers, "comfy.k_diffusion": k_diffusion,
        "comfy.k_diffusion.sampling": kds, "comfy.utils": utils,
    }


class _SpectrumState:
    """The bits of sorryhyun's SpectrumState the SPEED loop touches."""

    def __init__(self):
        self.active = True
        self.num_steps = 0
        self.resets = 0

    def reset(self):
        self.resets += 1


# ------------------------------------------------------------------------------------------------
# Schedules and a deterministic "model"
# ------------------------------------------------------------------------------------------------


def flux_schedule(steps: int, mu: float = 1.15) -> torch.Tensor:
    """aoleg tests/test_math.py ``_schedule``: exp(mu)-shifted, uniform in t."""
    shift = math.exp(mu)
    ts = [1.0 - i / steps for i in range(steps)]
    return torch.tensor([shift * t / (shift * t + 1.0 - t) for t in ts] + [0.0])


def anima_schedule(steps: int, shift: float = 3.0, beta_spacing: bool = False) -> torch.Tensor:
    """Discrete-flow (time_snr_shift) sigmas, uniform or Beta(0.6, 0.6) spaced like Forge's Beta."""
    if beta_spacing:
        from scipy import stats

        ts = [float(stats.beta.ppf(1.0 - i / steps, 0.6, 0.6)) for i in range(steps)]
    else:
        ts = [1.0 - i / steps for i in range(steps)]
    return torch.tensor([shift * t / (1.0 + (shift - 1.0) * t) for t in ts] + [0.0])


def denoiser(x: torch.Tensor, sigma: torch.Tensor, **_kwargs) -> torch.Tensor:
    """Any-shape deterministic x0 estimate (smooth in x and sigma)."""
    s = sigma.reshape(sigma.shape + (1,) * (x.ndim - 1)).to(x.dtype)
    return torch.tanh(x) * (0.6 - 0.3 * s) + 0.05 * x.mean(dim=tuple(range(1, x.ndim)), keepdim=True)


class Recorder:
    """k-diffusion ``sample_euler`` (s_churn = 0), recording every segment it is called on."""

    def __init__(self):
        self.calls = []
        self.__name__ = "sample_euler"

    def __call__(self, model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        extra_args = {} if extra_args is None else extra_args
        self.calls.append((tuple(x.shape), sigmas.detach().clone()))
        s_in = x.new_ones([x.shape[0]])
        for i in range(len(sigmas) - 1):
            sigma_hat = sigmas[i]
            denoised = model(x, sigma_hat * s_in, **extra_args)
            d = _kd_to_d(x, sigma_hat * s_in, denoised)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
        return x


def _ours(x, sigmas, *, mode, scales, threshold, seeds, transform="dct", manual=(), delta=0.01,
          preset="flux", adaptive=False, divisor=1.0, A=None, beta=None):
    if preset == "custom" or A is not None:
        A_, beta_, ref_latent, ref_shift = A, beta, schedule.REFERENCE_LATENT, None
    else:
        A_, beta_, ref_latent, ref_shift = schedule.preset_params(preset)
    plan = schedule.build_plan(
        mode=mode, transform=transform, sigmas=sigmas, full_grid=tuple(x.shape[-2:]), scales=scales,
        threshold=threshold, delta=delta, A=A_, beta=beta_, ref_latent=ref_latent, ref_shift=ref_shift,
        adaptive=adaptive, sigma_divisor=divisor, manual_sigmas=manual,
    )
    rec = Recorder()
    seen = []
    out, report = runner.run_speed(
        rec, denoiser, x, sigmas, run=runner.SpeedRun(plan=plan, seeds=list(seeds)),
        callback=lambda d: seen.append(d["i"]), disable=True,
    )
    return out, rec, seen, plan, report


# ------------------------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------------------------


class OracleIntegrityTests(unittest.TestCase):
    def test_every_oracle_is_the_pinned_upstream_text(self):
        for name, (_marker, digest) in ORACLES.items():
            with self.subTest(oracle=name):
                body = _body(name)
                self.assertEqual(hashlib.sha256(body.encode("utf-8")).hexdigest(), digest)

    def test_aoleg_copy_differs_from_upstream_by_the_fft_line_only(self):
        body = _body("_origin_speed_spectral_utils.py")
        self.assertEqual(body.count(FFT_FIXED_LINE), 1)
        self.assertNotIn(FFT_UPSTREAM_LINE, body)
        upstream = body.replace(FFT_FIXED_LINE, FFT_UPSTREAM_LINE)
        self.assertEqual(hashlib.sha256(upstream.encode("utf-8")).hexdigest(), AOLEG_SPECTRAL_UTILS_UPSTREAM_SHA256)

    def test_the_fixed_line_is_the_official_one(self):
        official = _body("_origin_speed_official_utils.py")
        self.assertIn(FFT_FIXED_LINE, official)
        self.assertNotIn("np.sqrt(2.0)", official)


class PresetAndThresholdParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.utils, cls.core = _load_aoleg()

    def test_presets_and_constants_are_aolegs(self):
        self.assertEqual(schedule.PRESETS, self.core._PRESETS)
        self.assertEqual(schedule.REFERENCE_LATENT, self.core.REFERENCE_LATENT)
        self.assertEqual(schedule.MAX_COARSE_FRACTION, self.core.MAX_COARSE_FRACTION)
        self.assertEqual(set(schedule.PRESET_NAMES), set(self.core._PRESETS))
        self.assertEqual(schedule.PRESET_NAMES[0], "anima")

    def test_spectrum_maths(self):
        u = self.utils
        for omega in (4.0, 16.0, 32.0, 64.0, 96.0):
            for A, beta in ((203.615097, 1.915461), (8664.998524, 2.422687), (520.495848, 1.915461)):
                P = schedule.power_spectrum(omega, A, beta)
                self.assertEqual(P, u.power_spectrum(omega, A, beta))
                for delta in (0.001, 0.01, 0.05, 0.5):
                    t = schedule.activation_time(P, delta)
                    self.assertEqual(t, u.activation_time(P, delta))
                    self.assertAlmostEqual(schedule.equivalent_delta(t, A, beta, omega), delta, places=9)
                    self.assertEqual(schedule.equivalent_delta(t, A, beta, omega), u.equivalent_delta(t, A, beta, omega))
        for t in (0.5, 0.85, 0.932, 0.99):
            for shift in (1.0, 3.0, 3.158193, 8.0):
                self.assertEqual(schedule.reference_coarse_fraction(t, shift), u.reference_coarse_fraction(t, shift))
        with self.assertRaises(ValueError):
            schedule.activation_time(1.0, 1.0)

    def test_delta_optimal_transitions(self):
        for scales in ([0.5, 1.0], [0.25, 0.5, 1.0], [0.37, 1.0], [0.3, 0.6, 0.8, 1.0]):
            for H, W in ((128, 128), (104, 152), (64, 64), (192, 256)):
                with self.subTest(scales=scales, hw=(H, W)):
                    self.assertEqual(
                        schedule.delta_optimal_transitions(scales, 0.01, 203.615097, 1.915461, H, W),
                        self.utils.delta_optimal_transitions(scales, 0.01, 203.615097, 1.915461, H, W),
                    )

    def test_parsing(self):
        for text in ("0.5,1.0", "0.25, 0.5 ,1", " 0.37,1.0 "):
            self.assertEqual(list(schedule.parse_scales(text)), self.core._parse_scales(text))
        for bad in ("0.5,0.9", "0.5,0.25,1.0", "0,1.0", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.core._parse_scales(bad)
                with self.assertRaises(ValueError):
                    schedule.parse_scales(bad)
        for text in ("0.85", "0.95,0.85", "0.9, 0.7, 0.2"):
            self.assertEqual(list(schedule.parse_sigmas(text)), self.core._parse_sigmas(text))
        for bad in ("1.0", "0", "0.7,0.8", "0.8,0.8"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.core._parse_sigmas(bad)
                with self.assertRaises(ValueError):
                    schedule.parse_sigmas(bad)
        self.assertEqual(schedule.parse_scales("1/2,1"), (0.5, 1.0))   # official parse_scales fractions

    def _schedules(self):
        out = []
        for steps in (8, 20, 32, 40):
            out.append((f"flux{steps}", flux_schedule(steps)))
            out.append((f"flux{steps}_mu6.7", flux_schedule(steps, 6.70)))
            out.append((f"anima{steps}", anima_schedule(steps)))
            out.append((f"anima{steps}_beta", anima_schedule(steps, beta_spacing=True)))
        out.append(("linear20", torch.linspace(1.0, 0.0, 21)))
        return out

    def test_resolve_transitions_matrix(self):
        cases = 0
        raised = 0
        for name, sigmas in self._schedules():
            for preset in ("flux", "flux2", "anima", "krea-2", "wan21", "custom"):
                A, beta, ref_latent, ref_shift = schedule.preset_params(preset, 203.615097, 1.915461)
                for scales in ([0.5, 1.0], [0.25, 0.5, 1.0]):
                    for latent in (64, 128, 192, 384):
                        for adaptive in (False, True):
                            for divisor in (1.0, 1.03, 0.9):
                                for delta in (0.005, 0.01, 0.05):
                                    tp = schedule.delta_threshold_plan(
                                        len(sigmas) - 1, scales, delta, A, beta, latent, latent,
                                        adaptive=adaptive, ref_latent=ref_latent, ref_shift=ref_shift,
                                        sigma_divisor=divisor,
                                    )
                                    actual = schedule.static_transitions(sigmas, scales, tp.values, tp.caps)
                                    try:
                                        expected = self.core._resolve_transitions(
                                            sigmas, scales, delta, A, beta, latent, latent,
                                            adaptive=adaptive, ref_latent=ref_latent, ref_shift=ref_shift,
                                            sigma_divisor=divisor,
                                        )
                                    except ValueError:
                                        # aoleg raises when a divisor < 1 lifts sigma* to >= 1 under a
                                        # ref_shift cap; ours keeps that threshold uncapped (it fires at
                                        # step 0, so build_plan drops the stage).
                                        self.assertTrue(adaptive and ref_shift and divisor < 1.0)
                                        for value, cap in zip(tp.values, tp.caps):
                                            if value >= 1.0:
                                                self.assertIsNone(cap)
                                        self.assertTrue(any(v >= 1.0 for v in tp.values))
                                        raised += 1
                                        continue
                                    if actual != expected:
                                        self.fail(f"{name} {preset} {scales} {latent} adaptive={adaptive} "
                                                  f"divisor={divisor} delta={delta}: {actual} != {expected}")
                                    cases += 1
        self.assertGreater(cases, 1000)
        self.assertGreater(raised, 0, "the divisor-above-one case is part of the matrix")

    def test_adaptive_log_matches(self):
        sigmas = flux_schedule(40)
        logged = []
        self.core._resolve_transitions(sigmas, [0.5, 1.0], 0.01, 203.615097, 1.915461, 192, 192,
                                       adaptive=True, ref_latent=128, sigma_divisor=1.03, log_fn=logged.append)
        tp = schedule.delta_threshold_plan(40, [0.5, 1.0], 0.01, 203.615097, 1.915461, 192, 192,
                                           adaptive=True, ref_latent=128, sigma_divisor=1.03)
        self.assertEqual(list(tp.notes), logged[:2])

    def test_resolve_manual_matrix(self):
        for name, sigmas in self._schedules():
            for scales, manual in (([0.5, 1.0], [0.85]), ([0.5, 1.0], [0.7]), ([0.25, 0.5, 1.0], [0.95, 0.85]),
                                   ([0.5, 1.0], [0.02]), ([0.25, 0.5, 1.0], [0.6, 0.3])):
                with self.subTest(schedule=name, manual=manual):
                    self.assertEqual(
                        schedule.static_transitions(sigmas, scales, manual),
                        self.core._resolve_manual(sigmas, scales, manual),
                    )


class TransitionModeRunParityTests(unittest.TestCase):
    """aoleg ``sample_speed_core`` (official segmentation, FFT-fixed spectral utils) vs ``run_speed``."""

    @classmethod
    def setUpClass(cls):
        cls.utils, cls.core = _load_aoleg()

    def _theirs(self, x, sigmas, *, seed, **kw):
        rec = Recorder()
        seen = []
        out = self.core.sample_speed_core(
            rec, denoiser, x, sigmas, extra_args={}, callback=lambda d: seen.append(d["i"]), disable=True,
            seed=seed, **kw,
        )
        return out, rec, seen

    def _compare(self, x, sigmas, theirs_kw, ours_kw, seed=1234, atol=2e-4):
        out_t, rec_t, seen_t = self._theirs(x, sigmas, seed=seed, **theirs_kw)
        out_o, rec_o, seen_o, plan, report = _ours(x, sigmas, mode="transition", seeds=[seed], **ours_kw)
        self.assertEqual([shape for shape, _ in rec_o.calls], [shape for shape, _ in rec_t.calls])
        for (_, sig_o), (_, sig_t) in zip(rec_o.calls, rec_t.calls):
            self.assertTrue(torch.equal(sig_o, sig_t), f"{sig_o} != {sig_t}")
        self.assertEqual(seen_o, seen_t)
        self.assertEqual(tuple(out_o.shape), tuple(out_t.shape))
        torch.testing.assert_close(out_o, out_t, atol=atol, rtol=1e-4)
        self.assertTrue(report.applied)
        return out_o, plan

    def test_manual_two_stage_dct_4d_and_5d(self):
        sigmas = anima_schedule(20)
        for shape in ((1, 4, 32, 32), (1, 4, 1, 24, 40), (1, 3, 2, 16, 16)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, generator=torch.Generator().manual_seed(7))
                self._compare(
                    x, sigmas,
                    dict(transform="dct", mode="manual", scales=[0.5, 1.0], manual_sigmas=[0.9]),
                    dict(scales=(0.5, 1.0), threshold="manual", manual=(0.9,)),
                )

    def test_every_transform(self):
        sigmas = flux_schedule(16)
        x = torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(3))
        for transform in ("dct", "fft", "dwt"):
            with self.subTest(transform=transform):
                self._compare(
                    x, sigmas,
                    dict(transform=transform, mode="manual", scales=[0.25, 0.5, 1.0], manual_sigmas=[0.97, 0.9]),
                    dict(transform=transform, scales=(0.25, 0.5, 1.0), threshold="manual", manual=(0.97, 0.9)),
                )

    def test_delta_optimal_presets_adaptive_and_neo_divisor(self):
        sigmas = anima_schedule(32, beta_spacing=True)
        x = torch.randn(1, 4, 1, 64, 48, generator=torch.Generator().manual_seed(11))
        for preset in ("anima", "flux", "custom"):
            A, beta, ref_latent, ref_shift = schedule.preset_params(preset, 203.615097, 1.915461)
            for divisor in (1.0, 1.03):
                with self.subTest(preset=preset, divisor=divisor):
                    self._compare(
                        x, sigmas,
                        dict(transform="dct", mode="delta_optimal", scales=[0.5, 1.0], delta=0.01, spectrum_A=A,
                             spectrum_beta=beta, adaptive_delta=True, ref_latent=ref_latent, ref_shift=ref_shift,
                             sigma_divisor=divisor),
                        dict(scales=(0.5, 1.0), threshold="neo_shift" if divisor != 1.0 else "delta_optimal",
                             delta=0.01, preset=preset, adaptive=True, divisor=divisor,
                             **({"A": A, "beta": beta} if preset == "custom" else {})),
                    )

    def test_non_dyadic_scale_and_odd_grid(self):
        sigmas = torch.linspace(1.0, 0.0, 13)
        x = torch.randn(1, 4, 27, 33, generator=torch.Generator().manual_seed(5))
        out, plan = self._compare(
            x, sigmas,
            dict(transform="dct", mode="manual", scales=[0.37, 1.0], manual_sigmas=[0.6]),
            dict(scales=(0.37, 1.0), threshold="manual", manual=(0.6,)),
        )
        self.assertEqual(plan.first_grid, (round(0.37 * 27), round(0.37 * 33)))

    def test_each_batch_image_is_the_upstream_single_seed_run(self):
        sigmas = anima_schedule(12)
        x = torch.randn(3, 4, 16, 16, generator=torch.Generator().manual_seed(9))
        seeds = [100, 101, 7777]
        out, *_ = _ours(x, sigmas, mode="transition", seeds=seeds, scales=(0.5, 1.0), threshold="manual", manual=(0.8,))
        for b, seed in enumerate(seeds):
            single, _, _ = self._theirs(
                x[b:b + 1], sigmas, seed=seed, transform="dct", mode="manual", scales=[0.5, 1.0], manual_sigmas=[0.8],
            )
            torch.testing.assert_close(out[b:b + 1], single, atol=2e-4, rtol=1e-4)
        batch_theirs, _, _ = self._theirs(
            x, sigmas, seed=seeds[0], transform="dct", mode="manual", scales=[0.5, 1.0], manual_sigmas=[0.8],
        )
        self.assertFalse(torch.allclose(out[1:], batch_theirs[1:], atol=1e-3),
                         "upstream draws one stream for the batch; ours is per image")

    def test_full_res_kwargs_and_segment_kwargs(self):
        """aoleg's contract: sampler kwargs reach every segment, the full-grid noise sampler only the last."""
        calls = []

        def fake(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, noise_sampler=None):
            calls.append((tuple(x.shape), s_churn, noise_sampler))
            return x

        sentinel = object()
        sigmas = torch.linspace(1.0, 0.0, 21)
        plan = schedule.build_plan(mode="transition", transform="dct", sigmas=sigmas, full_grid=(32, 32),
                                   scales=(0.5, 1.0), threshold="manual", manual_sigmas=(0.72,))
        runner.run_speed(fake, None, torch.randn(1, 4, 32, 32), sigmas, run=runner.SpeedRun(plan=plan, seeds=[0]),
                         sampler_kwargs={"s_churn": 0.25, "noise_sampler": sentinel})
        self.assertEqual([c[0] for c in calls], [(1, 4, 16, 16), (1, 4, 32, 32)])
        self.assertEqual([c[1] for c in calls], [0.25, 0.25])
        self.assertIsNone(calls[0][2])          # the default host has no coarse sampler
        self.assertIs(calls[1][2], sentinel)


class ExpansionOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.utils, cls.core = _load_aoleg()

    def _draws(self, seed, lead, shapes):
        return runner._numpy_draws(seed, lead, shapes)

    def test_dct_fft_dwt_against_numpy_oracles(self):
        x = np.random.default_rng(0).standard_normal((1, 4, 8, 12)).astype(np.float32)
        t, seed = 0.83, 4242
        cases = {
            "dct": (self.utils._dct_expand_np(x, (16, 24), t, seed), [(16, 24)]),
            "fft": (self.utils._fft_expand_np(x, (16, 24), t, seed), [(16, 24), (16, 24)]),
            "dwt": (self.utils._dwt_expand_np(x, t, seed), [(8, 12)] * 3),
        }
        for name, (expected, shapes) in cases.items():
            with self.subTest(transform=name):
                draws = [d.unsqueeze(0) for d in self._draws(seed, (4,), shapes)]
                actual = runner._expand(name, torch.from_numpy(x), (16, 24), t, draws)
                np.testing.assert_allclose(actual.numpy(), expected, atol=2e-5, rtol=1e-5)

    def test_official_fft_equals_the_fixed_aoleg_line(self):
        official = _load_official()
        x = np.random.default_rng(1).standard_normal((2, 3, 6, 10)).astype(np.float32)
        np.testing.assert_array_equal(
            official._fft_expand_np(x, (12, 20), 0.7, 99), self.utils._fft_expand_np(x, (12, 20), 0.7, 99),
        )
        np.testing.assert_array_equal(
            official._dct_expand_np(x, (12, 20), 0.7, 99), self.utils._dct_expand_np(x, (12, 20), 0.7, 99),
        )

    def test_fft_fix_restores_the_noise_variance(self):
        """With a zero source the expansion is pure fill: variance t**2 (fixed) vs t**2 / 2 (aoleg@a8873591)."""
        body = _body("_origin_speed_spectral_utils.py").replace(FFT_FIXED_LINE, FFT_UPSTREAM_LINE)
        namespace: dict = {"__name__": "_speed_origin_unfixed"}
        exec(compile(body, "aoleg_spectral_utils_unfixed", "exec"), namespace)  # noqa: S102 - pinned oracle text
        zero = np.zeros((1, 16, 4, 4), dtype=np.float32)
        fixed = self.utils._fft_expand_np(zero, (64, 64), 1.0, 5)
        unfixed = namespace["_fft_expand_np"](zero, (64, 64), 1.0, 5)
        np.testing.assert_allclose(unfixed, fixed / np.sqrt(2.0), atol=1e-6)
        self.assertAlmostEqual(float(fixed.var()), 1.0, delta=0.05)
        draws = [d.unsqueeze(0) for d in runner._numpy_draws(5, (16,), [(64, 64), (64, 64)])]
        ours = runner._expand("fft", torch.from_numpy(zero), (64, 64), 1.0, draws)
        self.assertAlmostEqual(float(ours.var()), float(fixed.var()), places=4)

    def test_dwt_ratio_rule_is_the_official_one(self):
        """aoleg (= official utils.spectral_expand_and_align) refuses dwt with r != 2 at the expansion,
        after the coarse steps ran; the plan refuses the same ladder before sampling."""
        with self.assertRaises(ValueError) as ctx:
            self.core._expand_and_align_torch(torch.zeros(1, 4, 2, 2), 0.26, 0.5, 0.9, "dwt", 0, 8, 8)
        self.assertIn("DWT requires r=2", str(ctx.exception))
        with self.assertRaises(schedule.PlanError):
            schedule.build_plan(mode="transition", transform="dwt", sigmas=anima_schedule(20), full_grid=(8, 8),
                                scales=(0.26, 0.5, 1.0), threshold="manual", manual_sigmas=(0.9, 0.6))

    def test_initial_downscale(self):
        x = torch.randn(2, 4, 1, 30, 22, generator=torch.Generator().manual_seed(2))
        for scale in (0.5, 0.37, 0.25):
            with self.subTest(scale=scale):
                expected = self.core._initial_dct_downscale(x, scale)
                grid = schedule.stage_grids("transition", [scale, 1.0], 30, 22)[0]
                actual = spectral.dct_downscale(x, grid).to(x.dtype)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)


class RespaceModeParityTests(unittest.TestCase):
    """sorryhyun ``spd_core`` primitives and the ``make_speed_sampler`` loop vs respace mode."""

    @classmethod
    def setUpClass(cls):
        cls.core, cls.spd = _load_sorryhyun()

    def test_dct_helpers(self):
        x = torch.randn(2, 4, 12, 18, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(spectral.dct2(x.double()).float(), self.core.dct2(x), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(spectral.idct2(x.double()).float(), self.core.idct2(x), atol=1e-5, rtol=1e-5)
        for v, m in ((62.5, 2), (31.0, 2), (0.4, 2), (13.0, 2), (64.0, 2)):
            self.assertEqual(spectral.snap(v, m), self.core._snap(v, m))

    def test_lowpass_init_grids(self):
        for H, W in ((64, 48), (104, 152), (50, 38), (128, 128)):
            for scale in (0.5, 0.25, 0.75):
                with self.subTest(hw=(H, W), scale=scale):
                    x5 = torch.randn(1, 4, 1, H, W, generator=torch.Generator().manual_seed(H))
                    expected = self.core.dct_lowpass_init(x5, scale, 2)
                    grid = schedule.stage_grids("respace", [scale, 1.0], H, W)[0]
                    self.assertEqual(tuple(expected.shape[-2:]), grid)
                    # sorryhyun multiplies float32 DCT matrices; ours is float64 (the float32 error grows with H).
                    torch.testing.assert_close(spectral.dct_downscale(x5, grid).float(), expected, atol=3e-4, rtol=1e-3)

    def test_spectral_expand(self):
        x5 = torch.randn(1, 4, 1, 16, 12, generator=torch.Generator().manual_seed(1))
        for sigma in (0.95, 0.7, 0.31):
            with self.subTest(sigma=sigma):
                gen = torch.Generator().manual_seed(77)
                expected, aligned = self.core.spectral_expand(x5, sigma, 0.5, 1.0, 32, 24, 2, gen)
                new, factor = spectral.respace_alignment(sigma, 2.0)
                self.assertEqual(aligned, new)
                draws = runner._torch_draws(torch.Generator().manual_seed(77), (4, 1), [(32, 24)], torch.device("cpu"))
                ours = runner._expand("dct", x5, (32, 24), sigma, [draws[0].unsqueeze(0)]) * factor
                torch.testing.assert_close(ours.float(), expected, atol=2e-5, rtol=1e-5)

    def _theirs(self, x, sigmas, stages, trans, seed):
        # sorryhyun's own normalisation (one sigma is repeated for every hand-off), then the loop.
        explicit = list(trans) if len(trans) == len(stages) - 1 else None
        stages_r, trans_r, active = self.spd.resolve_spd_schedule(stages, explicit, 1.0, trans[0])
        self.assertTrue(active)
        with _stub_modules(_comfy_stubs()):
            sampler = self.spd.make_speed_sampler(_SpectrumState(), stages_r, trans_r, seed)
        calls = []

        def model(x_in, sigma, **kw):
            calls.append((tuple(x_in.shape), float(sigma[0])))
            return denoiser(x_in, sigma, **kw)

        seen = []
        out = sampler.sampler_function(model, x, sigmas, extra_args={}, callback=lambda d: seen.append(d["i"]),
                                       disable=True)
        return out, calls, seen

    def _ours(self, x, sigmas, stages, trans, seed):
        plan = schedule.build_plan(mode="respace", transform="dct", sigmas=sigmas, full_grid=tuple(x.shape[-2:]),
                                   scales=stages, threshold="manual", manual_sigmas=trans)
        calls = []

        def model(x_in, sigma, **kw):
            calls.append((tuple(x_in.shape), float(sigma[0])))
            return denoiser(x_in, sigma, **kw)

        seen = []
        out, report = runner.run_speed(Recorder(), model, x, sigmas, run=runner.SpeedRun(plan=plan, seeds=[seed]),
                                       callback=lambda d: seen.append(d["i"]), disable=True)
        return out, calls, seen

    def test_loop_parity(self):
        cases = [
            ((1, 4, 1, 32, 24), anima_schedule(20), [0.5, 1.0], [0.7]),
            ((1, 4, 32, 32), flux_schedule(24), [0.5, 1.0], [0.85]),
            ((1, 4, 1, 48, 32), anima_schedule(28, beta_spacing=True), [0.25, 0.5, 1.0], [0.9, 0.6]),
            ((1, 4, 1, 40, 40), anima_schedule(30), [0.5, 0.75, 1.0], [0.6]),       # one sigma, every hand-off
            ((1, 4, 1, 64, 48), torch.linspace(1.0, 0.0, 26), [0.5, 1.0], [0.7]),
        ]
        for shape, sigmas, stages, trans in cases:
            with self.subTest(shape=shape, stages=stages, trans=trans):
                x = torch.randn(shape, generator=torch.Generator().manual_seed(len(stages) * 31 + shape[-1]))
                out_t, calls_t, seen_t = self._theirs(x, sigmas, stages, trans, 2024)
                out_o, calls_o, seen_o = self._ours(x, sigmas, stages, trans, 2024)
                self.assertEqual(calls_o, calls_t)      # shapes and sigmas of every model call, exactly
                self.assertEqual(seen_o, seen_t)
                torch.testing.assert_close(out_o, out_t, atol=2e-4, rtol=1e-4)

    def test_batch_image_equals_the_single_seed_loop(self):
        sigmas = anima_schedule(16)
        x = torch.randn(2, 4, 1, 24, 24, generator=torch.Generator().manual_seed(4))
        plan = schedule.build_plan(mode="respace", transform="dct", sigmas=sigmas, full_grid=(24, 24),
                                   scales=(0.5, 1.0), threshold="manual", manual_sigmas=(0.7,))
        out, _ = runner.run_speed(Recorder(), denoiser, x, sigmas, run=runner.SpeedRun(plan=plan, seeds=[5, 6]))
        for b, seed in enumerate((5, 6)):
            single, _, _ = self._theirs(x[b:b + 1], sigmas, [0.5, 1.0], [0.7], seed)
            torch.testing.assert_close(out[b:b + 1], single, atol=2e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
