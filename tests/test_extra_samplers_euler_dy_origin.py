"""Origin-parity tests for ``Euler Dy CFG++`` / ``Euler SMEA Dy CFG++`` (``sam3ext.extra_samplers.euler_dy``).

The oracle is Koishi-Star's ``smea_sampling.py`` (Euler-Smea-Dyn-Sampler@d98a504, Apache-2.0),
loaded verbatim from ``tests/_origin_koishi_smea_sampling.py`` (SHA-256 pinned below) on its WebUI
backend, with Forge Neo's real ``k_diffusion.sampling`` as the ``sampling`` it calls.

The port has exactly two switches against upstream, both covered here:

* ``churn_rule``: upstream clamps ``gamma = max(s_churn/N, √2−1)``; the registered samplers use
  k-diffusion's ``min``. ``churn_rule="max", cfg_pp=False`` must be upstream bit for bit (Dy and SMEA
  Dy, even and odd latent sizes, the WebUI ``_Rescaler`` inputs); with ``min`` the result must still
  be upstream's wherever the two rules give the same γ.
* ``cfg_pp``: the Euler updates become Forge's CFG++ update. Without a Dy step the sampler must be
  Forge's own ``sample_euler_cfg_pp`` bit for bit; with them it must equal an independent
  transcription of the documented algorithm, on epsilon and rectified-flow models.

Plus the Forge-side additions: 5-D latents (each frame as its own 4-D run), flow churn in
epsilon-equivalent coordinates, Forge-shaped masks / ``image_cond`` following the sub-step, the
``sam_extra_substep`` marker, the ``after_substep`` hook, and the noise draws.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


def _fixtures():
    name = "_extra_samplers_fixtures"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


fx = _fixtures()

from sam3ext.extra_samplers import common, euler_dy  # noqa: E402

ORIGIN_FILE = "_origin_koishi_smea_sampling.py"
ORIGIN_MARKER = "# ---- upstream smea_sampling.py below (verbatim) ----\n"
# SHA-256 of Koishi-Star/Euler-Smea-Dyn-Sampler@d98a504c8419be5274068ad91ca6bbf2e13635a8:smea_sampling.py
# with CRLF normalised to LF (git blob fa56ced4ca4d0a760f762aead7dcb42f04997ea6; the raw CRLF file is
# f4bc0081…3f0e2b).
ORIGIN_SHA256 = "110d29a6ad0ea4af9432e82285ccc6c4ef6817c5d82a39a90651ec1088a439c7"

SHAPES_4D = [(1, 4, 8, 8), (2, 3, 9, 7), (1, 4, 8, 7), (1, 4, 7, 8)]
LIMIT = 2 ** 0.5 - 1


class OriginCopyTests(unittest.TestCase):
    def test_origin_copy_is_upstream_verbatim(self):
        body = fx.origin_body(ORIGIN_FILE, ORIGIN_MARKER)
        self.assertEqual(fx.sha256(body), ORIGIN_SHA256)

    def test_upstream_uses_max_and_the_registered_samplers_use_min(self):
        body = fx.origin_body(ORIGIN_FILE, ORIGIN_MARKER)
        upstream_rule = "gamma = max(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0."
        self.assertEqual(body.count(upstream_rule), 4)   # euler_dy, smea_dy, and the two negative variants
        for public in (euler_dy.sample_euler_dy_cfg_pp, euler_dy.sample_euler_smea_dy_cfg_pp):
            source = inspect.getsource(public)
            with self.subTest(sampler=public.__name__):
                self.assertIn("cfg_pp=True, churn_rule=CHURN_MIN", source)
        for public in (euler_dy.sample_euler_dy, euler_dy.sample_euler_smea_dy):
            source = inspect.getsource(public)
            with self.subTest(sampler=public.__name__):
                self.assertIn("cfg_pp=False, churn_rule=CHURN_MIN", source)

    def test_the_plain_entries_have_upstreams_names_and_the_cfg_pp_entries_signature(self):
        for plain, cfg_pp, upstream_name in (
            (euler_dy.sample_euler_dy, euler_dy.sample_euler_dy_cfg_pp, "def sample_euler_dy("),
            (euler_dy.sample_euler_smea_dy, euler_dy.sample_euler_smea_dy_cfg_pp, "def sample_euler_smea_dy("),
        ):
            with self.subTest(sampler=plain.__name__):
                self.assertEqual(inspect.signature(plain), inspect.signature(cfg_pp))
                self.assertIn(upstream_name, fx.origin_body(ORIGIN_FILE, ORIGIN_MARKER))


class ChurnRuleTests(unittest.TestCase):
    def test_churn_gamma_is_the_two_inline_formulas(self):
        sigmas = [torch.tensor(v, dtype=torch.float64) for v in (0.02, 0.5, 3.0, 14.6)]
        for s_churn in (0.0, 0.3, 1.0, 2.5, 40.0):
            for steps in (4, 10, 30):
                for s_tmin, s_tmax in ((0.0, float("inf")), (0.4, 5.0), (100.0, 200.0)):
                    for sigma in sigmas:
                        # Koishi-Star (smea_sampling.py:126) and k-diffusion / Forge sample_euler (sampling.py:167)
                        upstream = max(s_churn / steps, 2 ** 0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.
                        kdiff = min(s_churn / steps, 2 ** 0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.
                        self.assertEqual(euler_dy.churn_gamma("max", s_churn, steps, sigma, s_tmin, s_tmax), upstream)
                        self.assertEqual(euler_dy.churn_gamma("min", s_churn, steps, sigma, s_tmin, s_tmax), kdiff)
        with self.assertRaises(ValueError):
            euler_dy.churn_gamma("median", 0.0, 4, sigmas[0], 0.0, 1.0)

    def test_the_rules_agree_only_outside_the_window_or_at_the_clamp(self):
        sigma = torch.tensor(1.0)
        self.assertEqual(euler_dy.churn_gamma("min", 0.0, 8, sigma, 0.0, float("inf")), 0.0)
        self.assertEqual(euler_dy.churn_gamma("max", 0.0, 8, sigma, 0.0, float("inf")), LIMIT)
        at_clamp = LIMIT * 8
        self.assertEqual(euler_dy.churn_gamma("min", at_clamp, 8, sigma, 0.0, float("inf")), LIMIT)
        self.assertEqual(euler_dy.churn_gamma("max", at_clamp, 8, sigma, 0.0, float("inf")), LIMIT)


class _ForgeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = fx.forge_k_sampling()
        cls.origin = fx.load_koishi_origin(cls.ks)

    def setUp(self):
        self.enterContext(fx.installed_k_sampling(self.ks))


def _a1111_inputs(model, shape, seed):
    """WebUI-shaped inpainting inputs on the denoiser: init_latent (B,C,H,W), mask/nmask (C,H,W)."""
    model.init_latent = fx.seeded(shape, seed + 1)
    mask = (fx.seeded(shape[1:], seed + 2) > 0).to(torch.float64)
    model.mask = mask
    model.nmask = 1 - mask


class KoishiParityTests(_ForgeBase):
    """``cfg_pp=False`` against the verbatim upstream samplers."""

    def _pair(self, smea, shape, *, churn_rule="max", s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"),
              s_noise=1.0, steps=6, inpaint=False, seed=0):
        sigmas = fx.eps_sigmas(steps)
        x = fx.seeded(shape, seed) * float(sigmas[0])
        upstream_fn = self.origin.sample_euler_smea_dy if smea else self.origin.sample_euler_dy
        origin_model, model = fx.ToyModel(fx.EpsSampling()), fx.ToyModel(fx.EpsSampling())
        if inpaint:
            _a1111_inputs(origin_model, shape, seed)
            _a1111_inputs(model, shape, seed)
        originals = (model.init_latent, model.mask, model.nmask)
        origin_seen, seen = [], []
        with fx.origin_torch(self.origin, fx.NoiseSequence(x, seed + 50)):
            origin = upstream_fn(origin_model, x.clone(), sigmas, extra_args={}, disable=True,
                                 callback=lambda d: origin_seen.append((d["x"], d["denoised"], d["sigma_hat"])),
                                 s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise)
        noise = fx.NoiseSequence(x, seed + 50)
        ours = euler_dy.euler_dy(model, x.clone(), sigmas, extra_args={}, disable=True,
                                 callback=lambda d: seen.append((d["x"], d["denoised"], d["sigma_hat"])),
                                 s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise,
                                 noise_sampler=noise, smea=smea, cfg_pp=False, churn_rule=churn_rule)
        self.assertEqual(noise.calls, steps)   # upstream draws on every step
        self.assertEqual((model.init_latent, model.mask, model.nmask), originals)
        self.assertTrue(all(a is b for a, b in zip((model.init_latent, model.mask, model.nmask), originals)))
        return ours, origin, seen, origin_seen, model, origin_model

    def _assert_same(self, ours, origin, seen, origin_seen, model, origin_model):
        self.assertTrue(torch.equal(ours, origin))
        self.assertEqual(len(seen), len(origin_seen))
        for (x, den, sh), (ox, oden, osh) in zip(seen, origin_seen):
            self.assertTrue(torch.equal(x, ox))
            self.assertTrue(torch.equal(den, oden))
            self.assertTrue(torch.equal(torch.as_tensor(sh), torch.as_tensor(osh)))
        self.assertEqual([(c.shape, c.sigma, c.init_latent, c.mask) for c in model.calls],
                         [(c.shape, c.sigma, c.init_latent, c.mask) for c in origin_model.calls])

    def test_max_rule_is_upstream_euler_dy(self):
        for shape in SHAPES_4D:
            for inpaint in (False, True):
                with self.subTest(shape=shape, inpaint=inpaint):
                    self._assert_same(*self._pair(False, shape, inpaint=inpaint))

    def test_max_rule_is_upstream_euler_smea_dy(self):
        for shape in SHAPES_4D:
            for inpaint in (False, True):
                with self.subTest(shape=shape, inpaint=inpaint):
                    self._assert_same(*self._pair(True, shape, inpaint=inpaint))

    def test_max_rule_with_churn_settings(self):
        for smea in (False, True):
            for s_churn, s_tmin, s_tmax, s_noise in ((20.0, 0.0, float("inf"), 1.0), (0.0, 0.5, 5.0, 0.8)):
                with self.subTest(smea=smea, s_churn=s_churn, s_tmin=s_tmin):
                    self._assert_same(*self._pair(smea, (1, 4, 8, 6), s_churn=s_churn, s_tmin=s_tmin,
                                                  s_tmax=s_tmax, s_noise=s_noise))

    def test_min_rule_is_upstream_where_both_rules_agree(self):
        for smea in (False, True):
            with self.subTest(smea=smea, case="window excludes every sigma"):
                self._assert_same(*self._pair(smea, (2, 3, 9, 7), churn_rule="min", s_tmin=1e9, s_tmax=2e9))
            with self.subTest(smea=smea, case="s_churn at the clamp"):
                # N = 4: s_churn/N == √2−1 exactly, so min and max both give √2−1
                self._assert_same(*self._pair(smea, (1, 4, 8, 8), churn_rule="min", s_churn=LIMIT * 4, steps=4))

    def test_the_churn_rule_is_the_only_deviation(self):
        """At the defaults (s_churn 0) ``min`` never churns: the run equals upstream with churn kept
        out of the window, and differs from upstream's default, which churns every step."""
        for smea in (False, True):
            with self.subTest(smea=smea):
                ours_min, _, _, _, _, _ = self._pair(smea, (1, 4, 8, 8), churn_rule="min")
                _, upstream_no_churn, _, _, _, _ = self._pair(smea, (1, 4, 8, 8), s_tmin=1e9, s_tmax=2e9)
                _, upstream_default, _, _, _, _ = self._pair(smea, (1, 4, 8, 8))
                self.assertTrue(torch.equal(ours_min, upstream_no_churn))
                self.assertFalse(torch.equal(ours_min, upstream_default))

    def test_the_registered_samplers_are_the_core_with_cfg_pp_and_min(self):
        sigmas = fx.flow_sigmas(6)
        x = fx.seeded((1, 4, 8, 8), 3)
        for public, smea in ((euler_dy.sample_euler_dy_cfg_pp, False), (euler_dy.sample_euler_smea_dy_cfg_pp, True)):
            with self.subTest(sampler=public.__name__):
                a = public(fx.ToyModel(fx.FlowSampling(), cond_scale=1.8), x.clone(), sigmas, disable=True,
                           s_churn=0.7, noise_sampler=fx.NoiseSequence(x, 9))
                b = euler_dy.euler_dy(fx.ToyModel(fx.FlowSampling(), cond_scale=1.8), x.clone(), sigmas, disable=True,
                                      s_churn=0.7, noise_sampler=fx.NoiseSequence(x, 9),
                                      smea=smea, cfg_pp=True, churn_rule="min")
                self.assertTrue(torch.equal(a, b))

    def test_the_registered_plain_samplers_are_the_core_without_cfg_pp_and_min(self):
        for flow in (True, False):
            sigmas = fx.flow_sigmas(6)[1:] if flow else fx.eps_sigmas(6)
            x = fx.seeded((1, 4, 8, 8), 3) * (1.0 if flow else float(sigmas[0]))
            sampling = fx.FlowSampling if flow else fx.EpsSampling
            for public, smea in ((euler_dy.sample_euler_dy, False), (euler_dy.sample_euler_smea_dy, True)):
                with self.subTest(sampler=public.__name__, flow=flow):
                    a_model, b_model = fx.ToyModel(sampling(), cond_scale=1.8), fx.ToyModel(sampling(), cond_scale=1.8)
                    a = public(a_model, x.clone(), sigmas, disable=True, s_churn=0.7,
                               noise_sampler=fx.NoiseSequence(x, 9))
                    b = euler_dy.euler_dy(b_model, x.clone(), sigmas, disable=True, s_churn=0.7,
                                          noise_sampler=fx.NoiseSequence(x, 9), smea=smea, cfg_pp=False,
                                          churn_rule="min")
                    self.assertTrue(torch.equal(a, b))
                    # no CFG++ hook: the plain entries leave Forge's CFG-1 optimisation alone
                    self.assertTrue(all("sampler_post_cfg_function" not in call.model_options
                                        for call in a_model.calls))

    def test_the_plain_entries_at_their_defaults_are_upstream_without_churn(self):
        """``Euler Dy`` / ``Euler SMEA Dy`` with their defaults (s_churn 0, the ``min`` rule: no churn) equal the
        verbatim upstream samplers run with ``s_tmin`` above every σ (upstream's ``max`` rule then gives γ = 0
        too) — bit for bit, on ε and flow models, even and odd latent sizes; the noise upstream draws on every
        step is replayed to ours (``fx.ReplayNoise``)."""
        for flow in (False, True):
            for shape in SHAPES_4D:
                sigmas = fx.flow_sigmas(7) if flow else fx.eps_sigmas(7)
                x = fx.seeded(shape, 40) * (1.0 if flow else float(sigmas[0]))
                sampling = fx.FlowSampling if flow else fx.EpsSampling
                for public, upstream_fn in ((euler_dy.sample_euler_dy, self.origin.sample_euler_dy),
                                            (euler_dy.sample_euler_smea_dy, self.origin.sample_euler_smea_dy)):
                    with self.subTest(sampler=public.__name__, flow=flow, shape=shape):
                        drawn = fx.NoiseSequence(x, 41)
                        origin_model, model = fx.ToyModel(sampling()), fx.ToyModel(sampling())
                        with fx.origin_torch(self.origin, drawn):
                            upstream = upstream_fn(origin_model, x.clone(), sigmas, extra_args={}, disable=True,
                                                   s_tmin=1e9, s_tmax=2e9)
                        replay = fx.ReplayNoise(drawn.drawn)
                        ours = public(model, x.clone(), sigmas, extra_args={}, disable=True, noise_sampler=replay)
                        self.assertTrue(torch.equal(ours, upstream))
                        self.assertEqual(replay.calls, len(sigmas) - 1)
                        self.assertEqual([(c.shape, c.sigma) for c in model.calls],
                                         [(c.shape, c.sigma) for c in origin_model.calls])

    def test_without_substeps_the_plain_entries_are_forges_euler(self):
        """``substeps=False`` (substep_guard) and no churn: Forge's own ``sample_euler`` bit for bit."""
        for flow in (False, True):
            sigmas = fx.flow_sigmas(7) if flow else fx.eps_sigmas(7)
            x = fx.seeded((2, 4, 6, 5), 42) * (1.0 if flow else float(sigmas[0]))
            sampling = fx.FlowSampling if flow else fx.EpsSampling
            for public in (euler_dy.sample_euler_dy, euler_dy.sample_euler_smea_dy):
                with self.subTest(sampler=public.__name__, flow=flow):
                    model = fx.ToyModel(sampling(), cond_scale=2.2)
                    ours = public(model, x.clone(), sigmas, disable=True, noise_sampler=fx.NoiseSequence(x),
                                  substeps=False)
                    self.assertEqual(len(model.calls), len(sigmas) - 1)
                    forge = self.ks.sample_euler(fx.ToyModel(sampling(), cond_scale=2.2), x.clone(), sigmas,
                                                 disable=True)
                    self.assertTrue(torch.equal(ours, forge))

    def test_churn_on_the_plain_entries_follows_the_min_rule(self):
        """ε: γ = min(s_churn/N, √2 − 1) on the first step (Karras); flow: the churned σ̂ stays below 1."""
        steps = 6
        for public in (euler_dy.sample_euler_dy, euler_dy.sample_euler_smea_dy):
            with self.subTest(sampler=public.__name__, model="eps"):
                sigmas = fx.eps_sigmas(steps)
                model = fx.ToyModel(fx.EpsSampling())
                public(model, fx.seeded((1, 4, 8, 8), 43) * float(sigmas[0]), sigmas, disable=True, s_churn=1.5,
                       noise_sampler=fx.NoiseSequence(torch.zeros(1, 4, 8, 8, dtype=torch.float64)))
                gamma = min(1.5 / steps, LIMIT)
                self.assertEqual(model.calls[0].sigma, float(sigmas[0] * (gamma + 1)))
            with self.subTest(sampler=public.__name__, model="flow"):
                sigmas = fx.flow_sigmas(steps + 1)[1:]
                model = fx.ToyModel(fx.FlowSampling())
                public(model, fx.seeded((1, 4, 8, 8), 44), sigmas, disable=True, s_churn=40.0,
                       noise_sampler=fx.NoiseSequence(torch.zeros(1, 4, 8, 8, dtype=torch.float64)))
                main = [call.sigma for call in model.calls if call.marker is None]
                self.assertTrue(all(0.0 < s < 1.0 for s in main), main)
                self.assertGreater(main[0], float(sigmas[0]))   # churned (γ = √2 − 1)


# ---------------------------------------------------------------------------
# CFG++
# ---------------------------------------------------------------------------


def _toy_pair(x, sigma, cond_scale):
    s = sigma.reshape(1) if sigma.ndim == 0 else sigma
    cond, uncond = fx.toy_x0(x, s, 1.0), fx.toy_x0(x, s, -1.0)
    return uncond + (cond - uncond) * cond_scale, uncond


def _alpha(sigma, flow):
    return 1 - sigma if flow else torch.ones_like(sigma)


def _cfg_pp(x, guided, uncond, sigma, sigma_next, flow):
    if sigma_next == 0:
        return guided
    return _alpha(sigma_next, flow) * guided + sigma_next * (x - _alpha(sigma, flow) * uncond) / sigma


def _odd_corner(out, x):
    h, w = x.shape[-2:]
    if h % 2 == 1 and w % 2 == 1:
        out[..., -1, -1] = x[..., 2 * (h // 2) - 1, -1]
    return out


def _reference(x, sigmas, *, smea, flow, cond_scale):
    """Euler (SMEA) Dy CFG++ transcribed from the module docstring (4-D, no churn)."""
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        guided, uncond = _toy_pair(x, sigma, cond_scale)
        h, w = x.shape[-2:]
        m, n = h // 2, w // 2
        if not smea:
            delta = torch.zeros_like(x)
            if sigma_next > 0 and i in (2, 3):
                c = x[..., 1:2 * m:2, 1:2 * n:2]
                c_guided, c_uncond = _toy_pair(c, sigma, cond_scale)
                dy = x.clone()
                dy[..., 1:2 * m:2, 1:2 * n:2] = _cfg_pp(c, c_guided, c_uncond, sigma, sigma_next, flow)
                delta = _odd_corner(dy, x) - x
            x = _cfg_pp(x, guided, uncond, sigma, sigma_next, flow) + delta
        else:
            x = _cfg_pp(x, guided, uncond, sigma, sigma_next, flow)
            if sigma_next > 0 and i == 1:
                c = x[..., 1:2 * m:2, 1:2 * n:2]
                c_guided, c_uncond = _toy_pair(c, sigma, cond_scale)
                dy = x.clone()
                dy[..., 1:2 * m:2, 1:2 * n:2] = _cfg_pp(c, c_guided, c_uncond, sigma, sigma_next, flow)
                x = _odd_corner(dy, x)
            if sigma_next > 0 and i == 0:
                up = F.interpolate(x, scale_factor=(1.25, 1.25), mode="nearest-exact")
                u_guided, u_uncond = _toy_pair(up, sigma, cond_scale)
                up = _cfg_pp(up, u_guided, u_uncond, sigma, sigma_next, flow)
                x = F.interpolate(up, size=(h, w), mode="nearest-exact")
    return x


class CfgPlusPlusTests(_ForgeBase):
    def _run(self, smea, flow, x, sigmas, cond_scale=2.2, **kwargs):
        sampling = fx.FlowSampling() if flow else fx.EpsSampling()
        model = fx.ToyModel(sampling, cond_scale=cond_scale)
        kwargs.setdefault("noise_sampler", fx.NoiseSequence(x, 1))
        out = euler_dy.euler_dy(model, x.clone(), sigmas, extra_args={}, disable=True,
                                smea=smea, cfg_pp=True, churn_rule="min", **kwargs)
        return out, model

    def test_without_a_substep_it_is_forges_euler_cfg_pp(self):
        for flow in (False, True):
            full = fx.flow_sigmas(8) if flow else fx.eps_sigmas(8)
            for sigmas in (full[:3], torch.cat([full[:2], full[-1:]])):   # two steps: i = 0, 1
                x = fx.seeded((2, 4, 6, 5), 2) * (1.0 if flow else float(sigmas[0]))
                with self.subTest(flow=flow, last=float(sigmas[-1])):
                    ours, _ = self._run(False, flow, x, sigmas)
                    sampling = fx.FlowSampling() if flow else fx.EpsSampling()
                    forge = self.ks.sample_euler_cfg_pp(fx.ToyModel(sampling, cond_scale=2.2), x.clone(), sigmas,
                                                        extra_args={}, disable=True)
                    self.assertTrue(torch.equal(ours, forge))

    def test_the_plain_mode_is_forges_euler_cfg_pp_on_every_step(self):
        """``substeps=False`` (substep_guard: Spectrum, Wan I2V, PiD …): no extra call anywhere, and the
        whole run is Forge's own ``sample_euler_cfg_pp`` bit for bit."""
        for smea in (False, True):
            for flow in (False, True):
                sigmas = fx.flow_sigmas(7) if flow else fx.eps_sigmas(7)
                x = fx.seeded((2, 4, 6, 5), 2) * (1.0 if flow else float(sigmas[0]))
                with self.subTest(smea=smea, flow=flow):
                    ours, model = self._run(smea, flow, x, sigmas, substeps=False)
                    self.assertEqual(len(model.calls), len(sigmas) - 1)
                    self.assertTrue(all(call.marker is None for call in model.calls))
                    sampling = fx.FlowSampling() if flow else fx.EpsSampling()
                    forge = self.ks.sample_euler_cfg_pp(fx.ToyModel(sampling, cond_scale=2.2), x.clone(), sigmas,
                                                        extra_args={}, disable=True)
                    self.assertTrue(torch.equal(ours, forge))

    def test_euler_dy_cfg_pp_is_the_documented_algorithm(self):
        for flow in (False, True):
            for shape in SHAPES_4D:
                sigmas = fx.flow_sigmas(7) if flow else fx.eps_sigmas(7)
                x = fx.seeded(shape, 4) * (1.0 if flow else float(sigmas[0]))
                with self.subTest(flow=flow, shape=shape):
                    ours, model = self._run(False, flow, x, sigmas)
                    reference = _reference(x.clone(), sigmas, smea=False, flow=flow, cond_scale=2.2)
                    torch.testing.assert_close(ours, reference, rtol=1e-12, atol=1e-12)
                    self.assertEqual(len(model.calls), (len(sigmas) - 1) + 2)

    def test_euler_smea_dy_cfg_pp_is_the_documented_algorithm(self):
        for flow in (False, True):
            for shape in SHAPES_4D:
                sigmas = fx.flow_sigmas(7) if flow else fx.eps_sigmas(7)
                x = fx.seeded(shape, 5) * (1.0 if flow else float(sigmas[0]))
                with self.subTest(flow=flow, shape=shape):
                    ours, model = self._run(True, flow, x, sigmas)
                    reference = _reference(x.clone(), sigmas, smea=True, flow=flow, cond_scale=2.2)
                    torch.testing.assert_close(ours, reference, rtol=1e-12, atol=1e-12)
                    up = (math.floor(shape[2] * 1.25), math.floor(shape[3] * 1.25))
                    self.assertEqual(model.calls[1].shape[-2:], up)                      # step 0: SMEA
                    self.assertEqual(model.calls[3].shape[-2:], (shape[2] // 2, shape[3] // 2))  # step 1: Dy

    def test_the_cfg_pp_hook_is_forges(self):
        x = fx.seeded((1, 4, 6, 6), 6)
        extra_args = {"model_options": {"transformer_options": {"keep": 1}}}
        euler_dy.sample_euler_dy_cfg_pp(fx.ToyModel(fx.FlowSampling(), cond_scale=1.5), x, fx.flow_sigmas(5),
                                        extra_args=extra_args, disable=True, noise_sampler=fx.NoiseSequence(x))
        options = extra_args["model_options"]
        self.assertTrue(options["disable_cfg1_optimization"])
        (capture,) = options["sampler_post_cfg_function"]
        self.assertIsInstance(capture, common.UncondCapture)
        self.assertEqual(options["transformer_options"], {"keep": 1})

    def test_without_an_unconditional_prediction_the_update_is_euler(self):
        sigmas = fx.flow_sigmas(6)
        x = fx.seeded((1, 4, 8, 8), 7)
        for smea in (False, True):
            with self.subTest(smea=smea):
                # cond_scale None: the toy runs no CFG and no post-CFG hook → nothing captured
                plain_cfg_pp = euler_dy.euler_dy(fx.ToyModel(fx.FlowSampling()), x.clone(), sigmas, disable=True,
                                                 noise_sampler=fx.NoiseSequence(x), smea=smea, cfg_pp=True)
                euler = euler_dy.euler_dy(fx.ToyModel(fx.FlowSampling()), x.clone(), sigmas, disable=True,
                                          noise_sampler=fx.NoiseSequence(x), smea=smea, cfg_pp=False,
                                          churn_rule="min")
                torch.testing.assert_close(plain_cfg_pp, euler, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Forge-side additions
# ---------------------------------------------------------------------------


class FiveDimensionalTests(_ForgeBase):
    def test_each_frame_of_a_5d_latent_is_its_own_4d_run(self):
        sigmas = fx.flow_sigmas(7)
        for smea in (False, True):
            for shape in ((2, 4, 3, 8, 8), (1, 3, 2, 9, 7)):
                with self.subTest(smea=smea, shape=shape):
                    x = fx.seeded(shape, 8)
                    noise = fx.NoiseSequence(x, 12)
                    five = euler_dy.euler_dy(fx.ToyModel(fx.FlowSampling(), cond_scale=1.7), x.clone(), sigmas,
                                             disable=True, s_churn=1.2, noise_sampler=noise, smea=smea)
                    frames = []
                    for t in range(shape[2]):
                        replay = fx.ReplayNoise([n[:, :, t] for n in noise.drawn])
                        frames.append(euler_dy.euler_dy(
                            fx.ToyModel(fx.FlowSampling(), cond_scale=1.7), x[:, :, t].clone(), sigmas,
                            disable=True, s_churn=1.2, noise_sampler=replay, smea=smea))
                    self.assertTrue(torch.equal(five, torch.stack(frames, dim=2)))

    def test_spatial_interpolate_ranks(self):
        x = fx.seeded((2, 3, 4, 9, 7), 9)
        five = euler_dy.spatial_interpolate(x, scale_factor=(1.25, 1.25), mode="nearest-exact")
        for t in range(4):
            torch.testing.assert_close(
                five[:, :, t], F.interpolate(x[:, :, t], scale_factor=(1.25, 1.25), mode="nearest-exact"),
                rtol=0, atol=0)
        mask = fx.seeded((3, 9, 7), 10)
        self.assertTrue(torch.equal(
            euler_dy.spatial_interpolate(mask, size=(4, 3)),
            F.interpolate(mask.unsqueeze(0), size=(4, 3), mode="nearest-exact").squeeze(0)))
        with self.assertRaises(ValueError):
            euler_dy.spatial_interpolate(torch.zeros(9, 7), size=(3, 3))


class ChurnTests(_ForgeBase):
    def _first_call(self, sampling, sigmas, x, s_churn, noise, cfg_pp=True):
        seen = {}

        class Recording(fx.ToyModel):
            def __call__(inner, xt, sigma, **kw):
                seen.setdefault("x", xt.clone())
                seen.setdefault("sigma", sigma.clone())
                return super().__call__(xt, sigma, **kw)

        euler_dy.euler_dy(Recording(sampling, cond_scale=1.5), x.clone(), sigmas, disable=True,
                          s_churn=s_churn, noise_sampler=noise, cfg_pp=cfg_pp)
        return seen["x"], seen["sigma"]

    def test_flow_churn_uses_the_epsilon_equivalent_noise_level(self):
        for cfg_pp in (True, False):   # the CFG++ entries and the plain Euler Dy / Euler SMEA Dy
            with self.subTest(cfg_pp=cfg_pp):
                sigmas = fx.flow_sigmas(5)[1:]       # starts below 1
                x = fx.seeded((1, 4, 6, 6), 11)
                noise = fx.NoiseSequence(x, 13)
                s_churn = 3.0                         # min(3/4, √2−1) = √2−1
                x_hat, sigma_hat = self._first_call(fx.FlowSampling(), sigmas, x, s_churn, noise, cfg_pp=cfg_pp)
                gamma = euler_dy.churn_gamma("min", s_churn, len(sigmas) - 1, sigmas[0], 0.0, float("inf"))
                self.assertEqual(gamma, LIMIT)
                sigma = sigmas[0]
                s = sigma / (1 - sigma)
                s_hat = s * (1 + gamma)
                expected_sigma = s_hat / (1 + s_hat)
                expected_x = (1 - expected_sigma) * (x / (1 - sigma) - noise.drawn[0] * (s_hat ** 2 - s ** 2) ** 0.5)
                torch.testing.assert_close(sigma_hat, expected_sigma.reshape(1), rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(x_hat, expected_x, rtol=1e-12, atol=1e-12)
                self.assertLess(float(sigma_hat), 1.0)
                self.assertGreater(float(sigma_hat), float(sigma))

    def test_flow_at_sigma_one_is_not_churned(self):
        sigmas = fx.flow_sigmas(5)
        self.assertEqual(float(sigmas[0]), 1.0)
        x = fx.seeded((1, 4, 6, 6), 14)
        x_hat, sigma_hat = self._first_call(fx.FlowSampling(), sigmas, x, 3.0, fx.NoiseSequence(x))
        self.assertEqual(float(sigma_hat), 1.0)
        self.assertTrue(torch.equal(x_hat, x))

    def test_epsilon_churn_is_karras(self):
        sigmas = fx.eps_sigmas(4)
        x = fx.seeded((1, 4, 6, 6), 15) * float(sigmas[0])
        noise = fx.NoiseSequence(x, 16)
        x_hat, sigma_hat = self._first_call(fx.EpsSampling(), sigmas, x, 0.8, noise)
        gamma = 0.8 / 4
        expected_sigma = sigmas[0] * (gamma + 1)
        torch.testing.assert_close(sigma_hat, expected_sigma.reshape(1), rtol=0, atol=0)
        torch.testing.assert_close(x_hat, x - noise.drawn[0] * (expected_sigma ** 2 - sigmas[0] ** 2) ** 0.5,
                                   rtol=0, atol=0)

    def test_noise_is_drawn_on_every_step(self):
        for s_churn in (0.0, 2.0):
            x = fx.seeded((1, 4, 6, 6), 17)
            noise = fx.NoiseSequence(x)
            euler_dy.sample_euler_smea_dy_cfg_pp(fx.ToyModel(fx.FlowSampling(), cond_scale=1.5), x, fx.flow_sigmas(6),
                                                 disable=True, s_churn=s_churn, noise_sampler=noise)
            self.assertEqual(noise.calls, 6)

    def test_default_noise_comes_from_forges_noise_sampler(self):
        calls = []
        original = self.ks.default_noise_sampler

        def patched(x):
            calls.append(tuple(x.shape))
            return original(x)

        self.ks.default_noise_sampler = patched
        try:
            x = fx.seeded((1, 4, 6, 6), 18)
            euler_dy.sample_euler_dy_cfg_pp(fx.ToyModel(fx.FlowSampling(), cond_scale=1.5), x, fx.flow_sigmas(4),
                                            disable=True)
        finally:
            self.ks.default_noise_sampler = original
        self.assertEqual(calls, [(1, 4, 6, 6)])


class SubstepInputTests(_ForgeBase):
    def _inpaint_model(self, shape, flow=True):
        model = fx.ToyModel(fx.FlowSampling() if flow else fx.EpsSampling(), cond_scale=1.6)
        model.init_latent = fx.seeded(shape, 20)
        mask_shape = (1, shape[1], *([1] * (len(shape) - 4)), shape[-2], shape[-1])   # Forge: (1,C,H,W) / (1,C,1,H,W)
        mask = (fx.seeded(mask_shape, 21) > 0).to(torch.float64)
        model.mask, model.nmask = mask, 1 - mask
        return model

    def test_inpainting_inputs_and_image_cond_follow_the_substep_and_come_back(self):
        for smea in (False, True):
            for shape in ((1, 4, 8, 6), (1, 4, 1, 8, 6)):
                with self.subTest(smea=smea, shape=shape):
                    model = self._inpaint_model(shape)
                    originals = (model.init_latent, model.mask, model.nmask)
                    image_cond = fx.seeded((shape[0], 5, *shape[2:]), 22)
                    extra_args = {"image_cond": image_cond}
                    x = fx.seeded(shape, 23)
                    euler_dy.euler_dy(model, x, fx.flow_sigmas(6), extra_args=extra_args, disable=True,
                                      noise_sampler=fx.NoiseSequence(x), smea=smea)
                    sub = [call for call in model.calls if call.marker is not None]
                    main = [call for call in model.calls if call.marker is None]
                    self.assertEqual(len(main), 6)
                    self.assertEqual([c.marker for c in sub], ["smea", "dy"] if smea else ["dy", "dy"])
                    for call in sub:
                        hw = call.shape[-2:]
                        self.assertNotEqual(hw, shape[-2:])
                        self.assertEqual(call.init_latent[-2:], hw)
                        self.assertEqual(call.mask[-2:], hw)
                        self.assertEqual(call.image_cond[-2:], hw)
                        self.assertEqual(len(call.init_latent), len(shape))
                    for call in main:
                        self.assertEqual(call.shape, shape)
                        self.assertEqual(call.image_cond[-2:], shape[-2:])
                    self.assertTrue(all(a is b for a, b in zip((model.init_latent, model.mask, model.nmask), originals)))
                    self.assertIs(extra_args["image_cond"], image_cond)

    def _with_forge_references(self, refs):
        dynamic_args = types.SimpleNamespace(ref_latents=refs)
        backend = types.ModuleType("backend")
        backend.__path__ = []
        args_module = types.ModuleType("backend.args")
        args_module.dynamic_args = dynamic_args
        self.enterContext(fx.stub_modules({"backend": backend, "backend.args": args_module}))
        return dynamic_args

    def _recording_model(self, dynamic_args, seen):
        class Recording(fx.ToyModel):
            def __call__(inner, x, sigma, **kwargs):
                seen.append((tuple(x.shape[-2:]), [tuple(ref.shape) for ref in dynamic_args.ref_latents]))
                return super().__call__(x, sigma, **kwargs)

        return Recording(fx.FlowSampling(), cond_scale=1.6)

    def test_anima_reference_latents_follow_a_5d_substep_and_come_back(self):
        same, other = fx.seeded((1, 4, 1, 8, 6), 30), fx.seeded((1, 4, 1, 5, 5), 31)
        refs = [same, other]
        dynamic_args = self._with_forge_references(refs)
        for smea in (False, True):
            with self.subTest(smea=smea):
                seen = []
                x = fx.seeded((1, 4, 1, 8, 6), 32)
                euler_dy.euler_dy(self._recording_model(dynamic_args, seen), x, fx.flow_sigmas(6), disable=True,
                                  noise_sampler=fx.NoiseSequence(x), smea=smea)
                self.assertEqual(len(seen), 8)
                for hw, ref_shapes in seen:
                    self.assertEqual(ref_shapes[0][-2:], hw)            # concatenated along T → follows
                    self.assertEqual(ref_shapes[1], (1, 4, 1, 5, 5))    # another size: not Anima's concat
                self.assertIs(dynamic_args.ref_latents, refs)
                self.assertIs(refs[0], same)

    def test_4d_references_are_left_alone(self):
        refs = [fx.seeded((1, 4, 8, 6), 33)]
        dynamic_args = self._with_forge_references(refs)
        seen = []
        x = fx.seeded((1, 4, 8, 6), 34)
        euler_dy.sample_euler_dy_cfg_pp(self._recording_model(dynamic_args, seen), x, fx.flow_sigmas(6),
                                        disable=True, noise_sampler=fx.NoiseSequence(x))
        self.assertTrue(all(ref_shapes == [(1, 4, 8, 6)] for _hw, ref_shapes in seen))
        self.assertIs(dynamic_args.ref_latents, refs)

    def test_a_dummy_image_cond_is_left_alone(self):
        model = fx.ToyModel(fx.FlowSampling(), cond_scale=1.6)
        dummy = torch.zeros(1, 5, 1, 1, dtype=torch.float64)
        x = fx.seeded((1, 4, 8, 8), 24)
        euler_dy.sample_euler_dy_cfg_pp(model, x, fx.flow_sigmas(6), extra_args={"image_cond": dummy},
                                        disable=True, noise_sampler=fx.NoiseSequence(x))
        self.assertTrue(all(call.image_cond == (1, 5, 1, 1) for call in model.calls))

    def test_the_substep_marker_rides_on_model_options(self):
        model = fx.ToyModel(fx.FlowSampling(), cond_scale=1.6)
        x = fx.seeded((1, 4, 8, 8), 25)
        extra_args = {"model_options": {"transformer_options": {"sampling_sigmas": "kept"}}}
        euler_dy.sample_euler_dy_cfg_pp(model, x, fx.flow_sigmas(6), extra_args=extra_args, disable=True,
                                        noise_sampler=fx.NoiseSequence(x))
        sub = [call for call in model.calls if call.marker is not None]
        self.assertEqual(len(sub), 2)
        for call in sub:
            options = call.model_options
            self.assertEqual(options["transformer_options"][common.SUBSTEP_MARKER], "dy")
            self.assertEqual(options["transformer_options"]["sampling_sigmas"], "kept")
            # the CFG++ capture is shared with the step's own evaluation
            self.assertIs(options["sampler_post_cfg_function"], extra_args["model_options"]["sampler_post_cfg_function"])
        self.assertNotIn(common.SUBSTEP_MARKER, extra_args["model_options"]["transformer_options"])

    def test_after_substep_gets_the_steps_full_resolution_denoised(self):
        for smea, expected_steps in ((False, [2, 3]), (True, [0, 1])):
            with self.subTest(smea=smea):
                model = fx.ToyModel(fx.FlowSampling(), cond_scale=1.6)
                x = fx.seeded((1, 4, 8, 8), 26)
                per_step, hooked = {}, []
                euler_dy.euler_dy(model, x, fx.flow_sigmas(6), disable=True, noise_sampler=fx.NoiseSequence(x),
                                  smea=smea, callback=lambda d: per_step.__setitem__(d["i"], d["denoised"]),
                                  after_substep=lambda latent: hooked.append((latent, len(model.calls))))
                self.assertEqual(len(hooked), 2)
                for (latent, calls_so_far), step in zip(hooked, expected_steps):
                    self.assertIs(latent, per_step[step])
                    self.assertEqual(tuple(latent.shape), (1, 4, 8, 8))
                    self.assertEqual(model.calls[calls_so_far - 1].marker, "smea" if smea and step == 0 else "dy")

    def test_a_latent_smaller_than_2x2_skips_the_dy_substep(self):
        model = fx.ToyModel(fx.FlowSampling(), cond_scale=1.6)
        x = fx.seeded((1, 4, 1, 6), 27)
        out = euler_dy.sample_euler_dy_cfg_pp(model, x, fx.flow_sigmas(6), disable=True,
                                              noise_sampler=fx.NoiseSequence(x))
        self.assertEqual(len(model.calls), 6)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
