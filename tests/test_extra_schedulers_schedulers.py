"""Extra Schedulers — the scheduler functions against independent implementations of their formulas.

The oracles below are written separately with numpy (float64) from the formulas, not by calling
the code under test. Where Forge has a matching function (k-diffusion ``get_sigmas_exponential`` and
``get_sigmas_karras``, ``sd_schedulers._loglinear_interp``) its real code is cut out of the Forge
checkout and run too (skipped when no Forge checkout is found).
"""
from __future__ import annotations

import ast
import contextlib
import io
import math
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import _extra_samplers_fixtures as fixtures  # noqa: E402
import _extra_schedulers_support as support  # noqa: E402
from sam3ext.extra_schedulers import schedulers as sch  # noqa: E402
from sam3ext.extra_schedulers import settings as st  # noqa: E402
from sam3ext.extra_schedulers import sigma_list as sl  # noqa: E402

PHI = (1 + 5 ** 0.5) / 2
# (sigma_min, sigma_max) of an SD/SDXL epsilon model and of a flow model (Anima/Flux sigmas ≤ 1).
RANGES = {"eps": (0.0291675, 14.614642), "flow": (0.0010, 1.0), "narrow": (0.5, 0.75)}
STEPS = (1, 2, 3, 4, 7, 20, 64)


def _p(n):
    return np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(max(n, 0))


def oracle_cosine(n, lo, hi):
    p = _p(n)
    return lo + 0.5 * (hi - lo) * (1 - np.cos(np.pi * (1 - np.sqrt(p))))


def oracle_exponential(n, lo, hi):
    return np.exp(np.linspace(np.log(hi), np.log(lo), n))


def oracle_blend(n, lo, hi):
    p = _p(n)
    return (1 - p) * oracle_cosine(n, lo, hi) + p * oracle_exponential(n, lo, hi)


def oracle_phi(n, lo, hi):
    return lo + (hi - lo) * (1 - _p(n)) ** (PHI ** 2)


def oracle_karras_dynamic(n, lo, hi, rho=7.0):
    out = []
    for i, ramp in enumerate(_p(n)):
        r = rho + 2 * np.cos(2 * np.pi * i / n)
        out.append((hi ** (1 / r) + ramp * (lo ** (1 / r) - hi ** (1 / r))) ** r)
    return np.array(out)


def oracle_loglinear(values, count):
    """Log-linear interpolation of a list onto ``count`` evenly spaced positions."""
    positions = np.linspace(0, 1, len(values))
    return np.exp(np.interp(np.linspace(0, 1, count), positions, np.log(np.asarray(values, dtype=np.float64))))


def oracle_loglinear_reversed(values, count):
    """The same, in the order Forge's ``_loglinear_interp`` works (reverse, interpolate, reverse back).
    Equal to ``oracle_loglinear`` for ``count >= 2``; for one position it gives the last value."""
    return oracle_loglinear(list(values)[::-1], count)[::-1]


def _check(test, result, expected, *, rtol=2e-6):
    test.assertEqual(result.dtype, torch.float32)
    test.assertEqual(result.shape, (len(expected) + 1,))
    test.assertEqual(float(result[-1]), 0.0, "the final sigma is 0")
    np.testing.assert_allclose(result[:-1].double().numpy(), expected, rtol=rtol, atol=0)


class ClosedFormTests(unittest.TestCase):
    CASES = {
        "Cosine": (sch.cosine, oracle_cosine),
        "CosineExponential blend": (sch.cosine_exponential_blend, oracle_blend),
        "Phi": (sch.phi, oracle_phi),
        "Karras Dynamic": (sch.karras_dynamic, oracle_karras_dynamic),
    }

    def test_against_independent_formulas(self):
        for label, (function, oracle) in self.CASES.items():
            for range_name, (lo, hi) in RANGES.items():
                for n in STEPS:
                    with self.subTest(scheduler=label, range=range_name, steps=n):
                        _check(self, function(n=n, sigma_min=lo, sigma_max=hi, device="cpu"), oracle(n, lo, hi))

    def test_endpoints_are_sigma_max_and_sigma_min(self):
        for label, (function, _oracle) in self.CASES.items():
            for lo, hi in RANGES.values():
                sigmas = function(n=12, sigma_min=lo, sigma_max=hi, device="cpu")
                with self.subTest(scheduler=label, range=(lo, hi)):
                    self.assertAlmostEqual(float(sigmas[0]), hi, delta=hi * 1e-6)
                    self.assertAlmostEqual(float(sigmas[-2]), lo, delta=lo * 1e-6)

    def test_strictly_decreasing(self):
        for label, (function, _oracle) in self.CASES.items():
            for lo, hi in RANGES.values():
                sigmas = function(n=30, sigma_min=lo, sigma_max=hi, device="cpu")
                with self.subTest(scheduler=label, range=(lo, hi)):
                    self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))

    def test_single_step_and_no_step(self):
        for label, (function, _oracle) in self.CASES.items():
            with self.subTest(scheduler=label):
                self.assertEqual(function(n=1, sigma_min=0.03, sigma_max=14.6, device="cpu").tolist(),
                                 [torch.tensor(14.6).item(), 0.0])
                self.assertEqual(function(n=0, sigma_min=0.03, sigma_max=14.6, device="cpu").tolist(), [0.0])

    def test_the_shapes_differ_where_the_descriptions_say(self):
        lo, hi = RANGES["eps"]
        n = 20
        cosine = sch.cosine(n, lo, hi).double().numpy()[:-1]
        exponential = oracle_exponential(n, lo, hi)
        blend = sch.cosine_exponential_blend(n, lo, hi).double().numpy()[:-1]
        # "Initial drop is relatively slow": Cosine keeps more noise early than the exponential.
        self.assertGreater(cosine[1], exponential[1])
        # "starts cosine, ends up exponential (long tail)".
        self.assertAlmostEqual(blend[0], cosine[0])
        self.assertLess(abs(blend[-2] - exponential[-2]), abs(cosine[-2] - exponential[-2]))

    def test_karras_dynamic_rho_reaches_the_formula(self):
        lo, hi = RANGES["eps"]
        for rho in (3.5, 7.0, 12.0):
            with self.subTest(rho=rho):
                _check(self, sch.karras_dynamic(10, lo, hi, rho=rho), oracle_karras_dynamic(10, lo, hi, rho))

    def test_karras_dynamic_rejects_rho_that_makes_an_exponent_non_positive(self):
        for rho in (2.0, 1.0, 0.0, -3.0, float("nan")):
            with self.subTest(rho=rho):
                with self.assertRaises(sch.KarrasDynamicError) as ctx:
                    sch.karras_dynamic(8, 0.03, 14.6, rho=rho)
                self.assertIsInstance(ctx.exception, ValueError)
                self.assertIn("rho", str(ctx.exception))

    def test_karras_dynamic_just_above_the_minimum_rho(self):
        # rho_i = rho + 2cos(2πi/n) gets down to 1e-6 here. Two steps: the second exponent is rho − 2, where
        # (σmin/σmax)^(1/rho_i) underflows to 0 — the ramp still ends at σmin, not at sigma 0.
        two = sch.karras_dynamic(2, 0.03, 14.6, rho=2.000001)
        self.assertEqual(two.tolist(), [torch.tensor(14.6).item(), torch.tensor(0.03).item(), 0.0])
        # Eight steps: the near-zero exponent at i = 4 lifts that step back to ≈ σmax (σmax^(1/rho_i) would
        # overflow in the textbook form) — a rising schedule, refused with a readable error.
        with self.assertRaises(sch.KarrasDynamicError) as ctx:
            sch.karras_dynamic(8, 0.03, 14.6, rho=2.000001)
        self.assertIn("rises", str(ctx.exception))

    def test_karras_dynamic_refuses_a_rising_schedule(self):
        lo, hi = RANGES["eps"]
        oracle = oracle_karras_dynamic(20, lo, hi, 3.0)
        self.assertTrue((np.diff(oracle) > 0).any(), "the formula itself rises for rho 3")
        with self.assertRaises(sch.KarrasDynamicError) as ctx:
            sch.karras_dynamic(20, lo, hi, rho=3.0)
        message = str(ctx.exception)
        self.assertTrue(message.startswith("Extra Schedulers: Karras Dynamic with rho 3 does not fall over 20 steps"))
        self.assertIn("Raise rho to about 4 or more: Settings → Sampler Parameters → rho", message)
        self.assertIsInstance(ctx.exception, sch.ExtraSchedulerError)
        self.assertNotIsInstance(ctx.exception, sch.CustomSchedulerError)

    def test_karras_dynamic_from_rho_4_never_rises(self):
        ranges = {**RANGES, "ztsnr": (0.0291675, 4500.0)}
        for rho in (4.0, 4.5, 7.0, 12.0):
            for range_name, (lo, hi) in ranges.items():
                with self.subTest(rho=rho, range=range_name):
                    for n in range(1, 151):
                        sigmas = sch.karras_dynamic(n, lo, hi, rho=rho)   # raises if any step rises or stays
                        self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))

    def test_karras_dynamic_ends_exactly_at_sigma_min(self):
        lo, hi = RANGES["eps"]
        for n in (2, 3, 7, 20):
            with self.subTest(steps=n):
                self.assertEqual(float(sch.karras_dynamic(n, lo, hi)[-2]), float(torch.tensor(lo)))

    def test_karras_dynamic_with_sigma_min_above_sigma_max_says_so(self):
        with self.assertRaises(sch.KarrasDynamicError) as ctx:
            sch.karras_dynamic(10, 14.6, 0.03)
        self.assertIn("larger sigma_min", str(ctx.exception))
        with self.assertRaises(sch.KarrasDynamicError) as ctx:
            sch.karras_dynamic(10, 2.0, 2.0)
        self.assertIn("the same sigma_min", str(ctx.exception))

    def test_a_schedule_that_keeps_sigma_where_it_was_is_refused(self):
        # sigma_min == sigma_max (Settings → Sampler Parameters overrides) makes every closed form flat.
        for function in (sch.cosine, sch.cosine_exponential_blend, sch.phi):
            with self.subTest(function=function.__name__):
                with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                    function(6, 2.0, 2.0)
                self.assertIn("at steps 0 and 1", str(ctx.exception))
                self.assertIn("Res Multistep", str(ctx.exception))
                self.assertEqual(function(1, 2.0, 2.0).tolist(), [2.0, 0.0], "one step has no neighbour")

    def test_karras_dynamic_names_a_flat_step(self):
        flat = torch.tensor([14.6, 3.0, 3.0, 0.03, 0.0])
        with self.assertRaises(sch.KarrasDynamicError) as ctx:
            sch._check_karras_dynamic_falls(flat, 4, 7.0, 0.03, 14.6)
        self.assertIn("sigma stays at 3 at step 2", str(ctx.exception))
        self.assertIn("Raise rho", str(ctx.exception))

    def test_logarithmic_schedules_need_positive_sigmas(self):
        for function in (sch.cosine_exponential_blend, sch.karras_dynamic):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function(5, 0.0, 14.6)

    def test_device_argument(self):
        self.assertEqual(sch.phi(4, 0.03, 14.6, device=torch.device("cpu")).device.type, "cpu")


@unittest.skipUnless(support.forge_root(), "Forge checkout not found")
class ForgeFunctionTests(unittest.TestCase):
    """The same comparisons against Forge's own k-diffusion functions (run from the Forge files)."""

    @classmethod
    def setUpClass(cls):
        namespace = {"torch": torch, "math": math}
        support.forge_function("modules_forge/packages/k_diffusion/sampling.py", "append_zero", namespace)
        cls.exponential = staticmethod(
            support.forge_function("modules_forge/packages/k_diffusion/sampling.py", "get_sigmas_exponential", namespace)
        )
        cls.karras = staticmethod(
            support.forge_function("modules_forge/packages/k_diffusion/sampling.py", "get_sigmas_karras", namespace)
        )

    def test_blend_ends_on_forges_exponential_schedule(self):
        for lo, hi in RANGES.values():
            for n in (2, 5, 20):
                forge = self.exponential(n, lo, hi).double().numpy()
                p = _p(n)
                expected = (1 - p) * oracle_cosine(n, lo, hi) + p * forge[:-1]
                with self.subTest(range=(lo, hi), steps=n):
                    # Forge's exponential is float32; the blend uses it at weight p.
                    _check(self, sch.cosine_exponential_blend(n, lo, hi), expected, rtol=2e-6)

    def test_karras_dynamic_meets_forges_karras_where_the_cosine_term_is_zero(self):
        # rho_i = rho + 2cos(2πi/n): at i = n/4 and 3n/4 the cosine is 0, so the step is Karras(rho).
        lo, hi = RANGES["eps"]
        for n in (4, 8, 20):
            karras = self.karras(n, lo, hi, rho=7.0).double().numpy()
            dynamic = sch.karras_dynamic(n, lo, hi).double().numpy()
            for i in (n // 4, 3 * n // 4):
                with self.subTest(steps=n, step=i):
                    self.assertAlmostEqual(dynamic[i] / karras[i], 1.0, delta=1e-6)
            with self.subTest(steps=n, step=0):
                self.assertAlmostEqual(dynamic[0] / karras[0], 1.0, delta=1e-6)


class CustomExpressionTests(unittest.TestCase):
    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)

    def _custom(self, expression, n=10, lo=0.0291675, hi=14.614642):
        st.set_active(st.ExtraSchedulerSettings(custom_mode=st.MODE_EXPRESSION, custom_expression=expression))
        return sch.custom(n=n, sigma_min=lo, sigma_max=hi, device="cpu")

    def test_default_expression_is_the_exponential_schedule(self):
        for lo, hi in RANGES.values():
            with self.subTest(range=(lo, hi)):
                _check(self, self._custom(st.DEFAULT_EXPRESSION, 15, lo, hi), oracle_exponential(15, lo, hi))

    def test_the_default_settings_use_the_default_expression(self):
        _check(self, sch.custom(9, 0.03, 14.6), oracle_exponential(9, 0.03, 14.6))

    def test_expressions_reproduce_the_closed_forms(self):
        lo, hi = RANGES["eps"]
        cases = {
            "m + 0.5 * (M - m) * (1 - cos(pi * (1 - sqrt(x))))": oracle_cosine,
            "m + (M - m) * (1 - x) ** (phi ** 2)": oracle_phi,
        }
        for text, oracle in cases.items():
            with self.subTest(text=text):
                _check(self, self._custom(text, 20, lo, hi), oracle(20, lo, hi))

    def test_all_variables(self):
        sigmas = self._custom("M - s * (M - m) / (n - 1) + 0 * x", 5, 1.0, 5.0)
        self.assertEqual(sigmas.tolist(), [5.0, 4.0, 3.0, 2.0, 1.0, 0.0])

    def test_single_step_uses_x_zero(self):
        self.assertEqual(self._custom("M * (1 - x) + m * x", 1, 0.5, 2.0).tolist(), [2.0, 0.0])

    def test_a_zero_or_negative_sigma_before_the_end_is_an_error(self):
        for text in ("M * (1 - x)", "m - 1", "0 * x"):
            with self.subTest(text=text):
                with self.assertRaises(sch.CustomSchedulerError) as ctx:
                    self._custom(text, 6)
                self.assertIn("above 0", str(ctx.exception))

    def test_a_flat_part_is_an_error(self):
        # A constant, or a max() that holds sigma_min for the last steps: zero-length steps (NaN with Res Multistep).
        for text, step in (("M", 0), ("max(m, M * (1 - 2 * x))", 5)):
            with self.subTest(text=text):
                with self.assertRaises(sch.CustomSchedulerError) as ctx:
                    self._custom(text, 10)
                self.assertIn(f"at steps {step} and {step + 1}", str(ctx.exception))
                self.assertIn("expression", str(ctx.exception))
        falling = self._custom("m + (M - m) * (1 - x)", 10)
        self.assertTrue(bool((falling[1:] < falling[:-1]).all()))

    def test_a_rising_expression_is_still_used(self):
        # Only a step that keeps sigma is refused; a custom schedule may rise (the user's choice).
        sigmas = self._custom("m + (M - m) * abs(cos(1.5 * pi * x))", 7)   # M, …, m, …, M, …, m
        self.assertTrue(bool((sigmas[1:-1] > sigmas[:-2]).any()), "this expression rises somewhere")

    def test_values_beyond_float32_are_an_error(self):
        with self.assertRaises(sch.CustomSchedulerError) as ctx:
            self._custom("M * 1e300", 4)
        self.assertIn("float32", str(ctx.exception))
        self.assertEqual(float(self._custom("M * 1e30 * (2 - x)", 4)[0]), float(torch.tensor(2 * 14.614642 * 1e30)))

    def test_values_below_the_float32_range_are_an_error(self):
        # Above 0 in float64, but 0 or a subnormal once stored in Forge's float32 sigmas — a sigma of 0
        # before the last step makes the samplers divide by zero.
        self.assertEqual(float(torch.tensor(0.0291675 * 1e-60, dtype=torch.float32)), 0.0)
        for text in ("m * 1e-60", "M * 1e-40", "M * (m / M) ** (x * 30)"):
            with self.subTest(text=text):
                with self.assertRaises(sch.CustomSchedulerError) as ctx:
                    self._custom(text, 10)
                self.assertIn("too small for Forge's float32 sigmas", str(ctx.exception))
        sigmas = self._custom("m * 1e-36 * (2 - x)", 4)   # 5.8e-38 … 2.9e-38: tiny, but still normal float32s
        self.assertTrue(bool((sigmas[:-1] > 0).all()))

    def test_invalid_expressions_raise_a_readable_error(self):
        for text in ("__import__('os')", "M +", "y", "sqrt(-1)", "x ** 99"):
            with self.subTest(text=text):
                with self.assertRaises(sch.CustomSchedulerError) as ctx:
                    self._custom(text)
                self.assertIn("Extra Schedulers", str(ctx.exception))
                self.assertIsInstance(ctx.exception, ValueError)


class CustomSigmaListTests(unittest.TestCase):
    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)

    def _custom(self, text, n, lo=0.0291675, hi=14.614642, interp=oracle_loglinear):
        st.set_active(st.ExtraSchedulerSettings(custom_mode=st.MODE_SIGMAS, custom_sigmas=text))
        with mock.patch.object(sl, "forge_loglinear_interp", return_value=interp):
            return sch.custom(n=n, sigma_min=lo, sigma_max=hi, device="cpu")

    def test_normalised_list_is_scaled_between_sigma_max_and_sigma_min(self):
        lo, hi = RANGES["eps"]
        values = [1.0, 0.6, 0.25, 0.1, 0.0]
        scaled = [lo + v * (hi - lo) for v in values]
        _check(self, self._custom("[1.0, 0.6, 0.25, 0.1, 0.0]", 5, lo, hi), np.array(scaled))   # same count: as is
        _check(self, self._custom("[1.0, 0.6, 0.25, 0.1, 0.0]", 12, lo, hi), oracle_loglinear(scaled, 12))

    def test_other_lists_are_sigmas_as_they_are(self):
        values = [14.6, 6.0, 2.0, 0.5, 0.03]
        _check(self, self._custom("14.6, 6, 2, 0.5, 0.03", 9), oracle_loglinear(values, 9))
        _check(self, self._custom("(14.6 6 2 0.5 0.03)", 5), np.array(values))

    def test_a_trailing_zero_is_the_final_zero(self):
        values = [10.0, 3.0, 1.0, 0.1]
        _check(self, self._custom("[10, 3, 1, 0.1, 0]", 7), oracle_loglinear(values, 7))
        _check(self, self._custom("[10, 3, 1, 0.1, 0]", 4), np.array(values))

    def test_flow_ranges(self):
        lo, hi = RANGES["flow"]
        values = [1.0, 0.5, 0.0]
        scaled = [lo + v * (hi - lo) for v in values]
        _check(self, self._custom("[1.0, 0.5, 0.0]", 8, lo, hi), oracle_loglinear(scaled, 8))

    def test_a_single_step_starts_at_the_first_value(self):
        # One step from the list's first sigma, like the expression mode (x = 0) and Karras/Exponential —
        # not from sigma_min, where Forge's interpolation onto one position lands.
        lo, hi = RANGES["eps"]
        interp = oracle_loglinear_reversed
        self.assertAlmostEqual(float(interp([hi, 3.0, lo], 1)[0]), lo, places=12)
        for text, first, low, high in (("[1.0, 0.6, 0.25, 0.1, 0.0]", hi, lo, hi), ("[14.6, 3, 0.03, 0]", 14.6, lo, hi),
                                       ("[1.0, 0.0]", hi, lo, hi)):
            with self.subTest(text=text):
                self.assertEqual(self._custom(text, 1, low, high, interp=interp).tolist(),
                                 [torch.tensor(first).item(), 0.0])
        _check(self, self._custom("[14.6, 3, 0.03]", 7, interp=interp), oracle_loglinear([14.6, 3, 0.03], 7))

    def test_values_below_the_float32_range_are_an_error(self):
        for text in ("[1e-50, 1e-60]", "[14.6, 1e-46]", "[14.6, 3, 1e-40, 0]"):
            with self.subTest(text=text):
                with self.assertRaises(sch.CustomSchedulerError) as ctx:
                    self._custom(text, 6)
                self.assertIn("too small for Forge's float32 sigmas", str(ctx.exception))

    def test_the_default_list(self):
        self.assertEqual(sl.parse_sigma_list(st.DEFAULT_SIGMAS), [1.0, 0.6, 0.25, 0.1, 0.0])

    def test_a_repeated_sigma_is_an_error(self):
        # As is (same count) and after the log-linear interpolation, which keeps a repeat flat.
        for text, n in (("[14.6, 3, 3, 0.03]", 4), ("[1.0, 1.0, 0.5, 0.0]", 9)):
            with self.subTest(text=text, steps=n):
                with self.assertRaises(sch.CustomSchedulerError) as ctx:
                    self._custom(text, n)
                self.assertIn("sigma list", str(ctx.exception))
                self.assertIn("Res Multistep", str(ctx.exception))

    def test_rejected_lists(self):
        bad = {
            "text": "abc",
            "one number": "[1.0]",
            "empty": "[]",
            "negative": "[1.0, -0.5, 0.0]",
            "nan": "[1.0, nan, 0.0]",
            "inf": "[inf, 1.0]",
            "huge": "[1e999, 1.0]",
            "unclosed": "[1.0, 0.5",
            "mismatched": "[1.0, 0.5)",
            "expression": "[1.0, 2*0.5]",
            "call": "[__import__('os'), 1]",
            "hex": "[0x10, 1]",
            "non-ASCII digits": "[٣, 1]",
            "underscore digits": "[1_0, 1]",
            "zero inside a raw list": "[5, 0, 1]",
            "only a zero after the final zero is dropped": "[3, 0, 0]",
            "too many": "[" + ", ".join(["1"] * (sl.MAX_VALUES + 1)) + "]",
            "too long": "[" + "1" * (sl.MAX_TEXT_LENGTH + 1) + ", 1]",
            "beyond float32": "[1e300, 1e299]",
        }
        for name, text in bad.items():
            with self.subTest(name=name):
                with self.assertRaises(sch.CustomSchedulerError):
                    self._custom(text, 6)

    def test_parser_errors_are_value_errors(self):
        with self.assertRaises(ValueError):
            sl.parse_sigma_list("[1, x]")
        with self.assertRaises(sl.SigmaListError):
            sl.parse_sigma_list(None)

    def test_without_forge_the_interpolation_error_is_readable(self):
        st.set_active(st.ExtraSchedulerSettings(custom_mode=st.MODE_SIGMAS, custom_sigmas="[1.0, 0.5, 0.0]"))
        with support.stub_modules({"modules": None}):
            with self.assertRaises(sch.CustomSchedulerError) as ctx:
                sch.custom(n=7, sigma_min=0.03, sigma_max=14.6)
        self.assertIn("log-linear", str(ctx.exception))


@unittest.skipUnless(support.forge_root(), "Forge checkout not found")
class ForgeInterpolationTests(unittest.TestCase):
    """The list mode with Forge's real ``_loglinear_interp`` (cut out of modules/sd_schedulers.py)."""

    @classmethod
    def setUpClass(cls):
        cls.interp = staticmethod(support.forge_function("modules/sd_schedulers.py", "_loglinear_interp", {"np": np}))

    def test_forge_interpolation_matches_the_oracle(self):
        for values in ([14.6, 6.0, 2.0, 0.5, 0.03], [1.0, 0.7, 0.2, 0.01], [3.0, 3.0]):
            for count in (2, 5, 13, 40):
                with self.subTest(values=values, count=count):
                    np.testing.assert_allclose(self.interp(values, count), oracle_loglinear(values, count), rtol=1e-12)

    def test_list_mode_uses_forges_function(self):
        st.set_active(st.ExtraSchedulerSettings(custom_mode=st.MODE_SIGMAS, custom_sigmas="[1.0, 0.6, 0.25, 0.1, 0.0]"))
        self.addCleanup(st.reset)
        lo, hi = RANGES["eps"]
        scaled = [lo + v * (hi - lo) for v in (1.0, 0.6, 0.25, 0.1, 0.0)]
        with mock.patch.object(sl, "forge_loglinear_interp", return_value=self.interp):
            result = sch.custom(n=11, sigma_min=lo, sigma_max=hi)
        _check(self, result, np.asarray(self.interp(scaled, 11), dtype=np.float64), rtol=1e-7)

    def test_one_position_of_forges_interpolation_is_the_last_value(self):
        # Why a single step does not go through the interpolation (sigma_list.sigmas_from_list).
        (only,) = self.interp([14.6, 3.0, 0.03], 1)
        self.assertAlmostEqual(float(only), 0.03, places=12)
        for count in (1, 2, 7):   # the stand-in the list tests use for one position is Forge's order
            np.testing.assert_allclose(self.interp([14.6, 3.0, 0.03], count),
                                       oracle_loglinear_reversed([14.6, 3.0, 0.03], count), rtol=1e-12)
        st.set_active(st.ExtraSchedulerSettings(custom_mode=st.MODE_SIGMAS, custom_sigmas="[14.6, 3, 0.03]"))
        self.addCleanup(st.reset)
        with mock.patch.object(sl, "forge_loglinear_interp", return_value=self.interp):
            self.assertEqual(sch.custom(n=1, sigma_min=0.03, sigma_max=14.6).tolist(), [torch.tensor(14.6).item(), 0.0])


class LaplaceSettingsTests(unittest.TestCase):
    """Laplace reads μ/β from the generation's settings (ComfyUI parity is in the origin tests)."""

    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)

    def test_mu_and_beta_come_from_the_active_settings(self):
        base = sch.laplace(12, 0.03, 14.6)
        st.set_active(st.ExtraSchedulerSettings(laplace_mu=1.0, laplace_beta=0.3))
        moved = sch.laplace(12, 0.03, 14.6)
        self.assertFalse(torch.equal(base, moved))
        expected = torch.cat([sch.get_sigmas_laplace(12, 0.03, 14.6, mu=1.0, beta=0.3), torch.zeros(1)])
        self.assertTrue(torch.equal(moved, expected))

    def test_flow_models_use_the_part_of_the_curve_below_sigma_max(self):
        # With sigma_max = 1 the default μ 0 puts e^μ at sigma_max: ComfyUI's clamp gives the first half of
        # the steps sigma 1 (zero-length steps — Res Multistep returned a black image on Anima 3.8B). The
        # steps are spread over x = ½ … 1 instead, where the curve is e^(β·log(2 − 2x + ε)).
        lo, hi = RANGES["flow"]
        comfy = sch.get_sigmas_laplace(20, lo, hi)
        self.assertEqual(comfy[:10].tolist(), [1.0] * 10)
        sigmas = sch.laplace(20, lo, hi)
        x = np.linspace((1 - 1e-5) / 2, 1.0, 20)   # where the curve is 1 (ε/2 left of ½) … the end
        expected = np.minimum(np.exp(0.5 * np.log(np.abs(2 - 2 * x) + 1e-5)), hi)
        _check(self, sigmas, expected, rtol=5e-6)
        self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))
        self.assertEqual(float(sigmas[0]), 1.0)

    def test_a_negative_mu_on_flow_sigmas_is_comfyuis_schedule(self):
        lo, hi = RANGES["flow"]
        st.set_active(st.ExtraSchedulerSettings(laplace_mu=-2.0))
        lowered = sch.laplace(20, lo, hi)
        self.assertTrue(torch.equal(lowered[:-1], sch.get_sigmas_laplace(20, lo, hi, mu=-2.0)), "no repeat, no spread")
        self.assertEqual(float(lowered[0]), 1.0)

    def test_the_spread_finds_the_ends_of_the_range_on_the_curve(self):
        # An independent search: bisection on the float64 curve for where it crosses sigma_max and sigma_min.
        def curve(x, mu, beta):
            if x == 0.5:
                return math.exp(mu)
            return math.exp(mu - beta * math.copysign(1.0, 0.5 - x) * math.log(1 - 2 * abs(0.5 - x) + 1e-5))

        def crossing(sigma, mu, beta):
            low, high = 0.0, 1.0   # the curve falls from x = 0 to x = 1
            if curve(low, mu, beta) <= sigma:
                return low
            if curve(high, mu, beta) >= sigma:
                return high
            for _ in range(200):
                mid = (low + high) / 2
                low, high = (mid, high) if curve(mid, mu, beta) > sigma else (low, mid)
            return (low + high) / 2

        for n, lo, hi, mu, beta in ((30, 0.0291675, 14.614642, -2.0, 2.0), (16, 0.001, 1.0, -1.0, 0.5),
                                    (12, 0.0291675, 14.614642, 10.0, 10.0), (28, 0.003, 1.0, 0.0, 0.5),
                                    (24, 0.001, 1.0, 1.5, 1.0)):
            with self.subTest(n=n, mu=mu, beta=beta):
                st.set_active(st.ExtraSchedulerSettings(laplace_mu=mu, laplace_beta=beta))
                ours = sch.laplace(n, lo, hi)
                x = np.linspace(crossing(hi, mu, beta), crossing(lo, mu, beta), n)
                expected = np.clip([curve(float(v), mu, beta) for v in x], lo, hi)
                _check(self, ours, expected, rtol=2e-4)   # ComfyUI's float32 log(2 − 2x + ε) loses digits near x = 1
                self.assertTrue(bool((ours[1:] < ours[:-1]).all()))

    def test_no_part_of_the_curve_in_the_range_is_an_error(self):
        lo, hi = RANGES["eps"]
        for mu, beta, what in ((0.0, 0.0, "beta 0"), (-10.0, 0.1, "mu -10"), (10.0, 0.1, "mu 10")):
            with self.subTest(mu=mu, beta=beta):
                st.set_active(st.ExtraSchedulerSettings(laplace_mu=mu, laplace_beta=beta))
                with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                    sch.laplace(12, lo, hi)
                message = str(ctx.exception)
                self.assertIn(what, message)
                self.assertIn("Raise beta, or move mu", message)
                self.assertIn("Res Multistep", message)

    def test_shape_and_final_zero(self):
        for n in (1, 2, 9, 40):
            sigmas = sch.laplace(n, 0.03, 14.6)
            with self.subTest(steps=n):
                self.assertEqual(sigmas.shape, (n + 1,))
                self.assertEqual(float(sigmas[-1]), 0.0)
                self.assertTrue(bool((sigmas[:-1] >= 0.03 - 1e-7).all() and (sigmas[:-1] <= 14.6 + 1e-6).all()))


def _anima_range() -> tuple:
    """(sigma_min, sigma_max) of Anima (shift 3) as Forge's ``PredictionDiscreteFlow`` builds its float32 table
    (backend/modules/k_prediction.py: ``time_snr_shift(shift, (arange(1, 1001) / 1000 * 1000) / 1000)``) and
    ``KDiffusionSampler.get_sigmas`` hands it on (``sigmas[0].item()``, ``sigmas[-1].item()``)."""
    t = (torch.arange(1, 1001, 1) / 1000) * 1000 / 1000
    table = 3.0 * t / (1 + (3.0 - 1) * t)
    return table[0].item(), table[-1].item()


def oracle_react(n, lo, hi, factor=2.15):
    """React Cosinusoidal DynSF from its one-line formula (float64)."""
    p = _p(n)
    return hi * ((lo + (hi - lo) * np.cos(np.pi * p / 2)) / hi) ** (factor * p)


class ReactDynSFTests(unittest.TestCase):
    RANGES = {"anima": _anima_range(), "sdxl": RANGES["eps"]}

    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)

    def test_the_factor_is_reforges_default(self):
        self.assertEqual(sch.REACT_DYNSF_FACTOR, 2.15)

    def test_values_are_the_formula_in_float64_stored_as_float32(self):
        for range_name, (lo, hi) in self.RANGES.items():
            for n in (1, 2, 3, 28, 50, 150):
                with self.subTest(range=range_name, steps=n):
                    ours = sch.react_cosinusoidal_dynsf(n=n, sigma_min=lo, sigma_max=hi, device="cpu")
                    expected = torch.tensor(np.append(oracle_react(n, lo, hi), 0.0), dtype=torch.float64)
                    self.assertTrue(torch.equal(ours, expected.to(torch.float32)))
                    self.assertEqual(ours.dtype, torch.float32)
                    self.assertEqual(float(ours[0]), float(torch.tensor(hi, dtype=torch.float32)))
        self.assertEqual(sch.react_cosinusoidal_dynsf(0, 0.03, 14.6).tolist(), [0.0])
        self.assertEqual(sch.react_cosinusoidal_dynsf(1, 0.03, 14.6).tolist(), [torch.tensor(14.6).item(), 0.0])

    def test_the_shape_on_anima_shift_3_with_28_steps(self):
        """The scout's numbers: high-σ-heavy, and one nearly wasted last step."""
        lo, hi = self.RANGES["anima"]
        self.assertAlmostEqual(lo, 0.002994012, places=9)
        sigmas = sch.react_cosinusoidal_dynsf(28, lo, hi)[:-1].double()
        self.assertEqual(int((sigmas > 0.5).sum()), 17)
        self.assertEqual(round(sorted(sigmas.tolist())[len(sigmas) // 2], 3), 0.72)
        self.assertEqual(int((sigmas < 0.05).sum()), 4)
        self.assertAlmostEqual(float(sigmas[-1]), 3.749e-6, delta=0.001e-6)
        # = σmax·(σmin/σmax)^2.15 (cos(π/2) is 6e-17, not 0)
        self.assertAlmostEqual(float(sigmas[-1]), hi * (lo / hi) ** 2.15, delta=1e-12)

    def test_it_is_the_custom_expression_with_factor_2_15(self):
        """The same formula through the custom scheduler's safe evaluator (a different code path): equal bit for
        bit on both ranges, so other factors can be had by editing 2.15 in the expression."""
        st.set_active(st.ExtraSchedulerSettings(custom_mode=st.MODE_EXPRESSION,
                                                custom_expression="M*((m+(M-m)*cos(x*pi/2))/M)**(2.15*x)"))
        for range_name, (lo, hi) in self.RANGES.items():
            for n in (2, 3, 28, 50, 150):
                with self.subTest(range=range_name, steps=n):
                    self.assertTrue(torch.equal(sch.custom(n, lo, hi), sch.react_cosinusoidal_dynsf(n, lo, hi)))

    def test_strictly_falling_until_float32_runs_out_of_room_near_sigma_max(self):
        """σ₁ ≈ σmax·(1 − 2.65·x³) for x = 1/(n − 1): in float32 σ₀ == σ₁ from 448 steps on Anima's range and from
        489 on SDXL's — the existing flat-step rule then refuses the schedule (ExtraSchedulerError) instead of
        handing a zero-length step to the sampler."""
        for range_name, (lo, hi), last_good in (("anima", self.RANGES["anima"], 447), ("sdxl", self.RANGES["sdxl"], 488)):
            with self.subTest(range=range_name):
                for n in range(2, last_good + 1):
                    sigmas = sch.react_cosinusoidal_dynsf(n, lo, hi)
                    self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()), n)
                with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                    sch.react_cosinusoidal_dynsf(last_good + 1, lo, hi)
                self.assertIn("React Cosinusoidal DynSF gives sigma", str(ctx.exception))
                self.assertIn("at steps 0 and 1", str(ctx.exception))

    def test_needs_positive_sigmas(self):
        for lo, hi in ((0.0, 1.0), (-0.1, 1.0), (0.003, 0.0)):
            with self.subTest(lo=lo, hi=hi):
                with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                    sch.react_cosinusoidal_dynsf(10, lo, hi)
                self.assertIn("needs sigma_min and sigma_max above 0", str(ctx.exception))

    def test_device_argument(self):
        self.assertEqual(sch.react_cosinusoidal_dynsf(4, 0.03, 14.6, device=torch.device("cpu")).device.type, "cpu")


# ---------------------------------------------------------------------------
# Flow Cosmos rho7
# ---------------------------------------------------------------------------
# The custom expression of the GPU A/B run on Anima (scout VERIFIED.md C3); the scheduler must give its numbers.
COSMOS_AB_EXPRESSION = "(80**(1/7)+x*(0.002**(1/7)-80**(1/7)))**7/(1+(80**(1/7)+x*(0.002**(1/7)-80**(1/7)))**7)"
COSMOS_STEPS = (1, 2, 3, 28, 50, 150)
# float32 runs out of room near the top (Δt ≈ 0.0665/(n − 1) against an ulp of 5.96e-8): the first two equal
# neighbours appear at this step count (steps 467/468, t ≈ 0.987626), found by a float32 scan in the design run.
COSMOS_FIRST_FLAT_STEPS = 1_119_764


def upstream_flow_cosmos_rho(steps, order=7.0, sigma_max=80.0, sigma_min=0.002):
    """Test reference only, not shipped: the numbers of KeithZ117/Comfyui-anima-sampler @ effba3c5 (MIT)
    ``cosmos_schedules.py`` ``build_flow_cosmos_rho_sigmas`` (lines 157-178) with its helpers
    ``rho_space_descending`` and ``_cosmos_rflow_time``, written out here in their operation order (steps >= 2)."""
    start_root, end_root = sigma_max ** (1 / order), sigma_min ** (1 / order)
    step = (end_root - start_root) / (steps - 1)
    times = []
    for i in range(steps):
        sigma = (start_root + step * i) ** order
        clamped = min(float(sigma), float(sigma_max))
        times.append(0.0 if sigma <= 0.0 else clamped / (clamped + 1.0))
    return times + [0.0]


def karras_formula(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Test stand-in for Forge's ``k_diffusion.sampling.get_sigmas_karras``: Karras et al. (arXiv:2206.00364) eq. 5
    in torch float32 with the final 0. Forge's own function is used in FlowCosmosRho7ForgeTests."""
    p = torch.linspace(0, 1, n, device=device)
    top, bottom = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
    return torch.cat([(top + p * (bottom - top)) ** rho, torch.zeros(1, device=device)])


_MISSING = object()


def _k_sampling_with(get_sigmas_karras):
    """A ``k_diffusion.sampling`` stand-in with only what Flow Cosmos rho7 calls on an eps/v model."""
    module = types.ModuleType("k_diffusion.sampling")
    if get_sigmas_karras is not _MISSING:
        module.get_sigmas_karras = get_sigmas_karras
    return module


def _with_prediction(kind):
    """What Forge hands a need_inner_model scheduler, reduced to what is read: ``model_wrap.predictor``."""
    return types.SimpleNamespace(predictor=types.SimpleNamespace(prediction_type=kind))


FLOW_MODEL, EPS_MODEL, V_MODEL = _with_prediction("const"), _with_prediction("epsilon"), _with_prediction("v_prediction")


class FlowCosmosRho7Tests(unittest.TestCase):
    FLOW_RANGES = {"anima": _anima_range(), "flow": RANGES["flow"], "narrow flow": (0.05, 0.9)}

    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)
        patcher = mock.patch.object(sch, "_flow_cosmos_not_flow_logged", True)   # quiet, except where tested
        patcher.start()
        self.addCleanup(patcher.stop)
        # Forge's k_diffusion.sampling, which the eps/v branch calls at run time: a recording stand-in here
        self.karras_calls = []

        def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
            self.karras_calls.append((n, sigma_min, sigma_max, rho, device))
            return karras_formula(n, sigma_min, sigma_max, rho=rho, device=device)

        self.enterContext(fixtures.installed_k_sampling(_k_sampling_with(get_sigmas_karras)))

    def test_the_constants_are_module_level_literals(self):
        """Cosmos-Predict2's EDM range and ρ; the desktop app pins them (and the log line) by AST. They are the
        defaults of the accordion's Flow Cosmos values. The log line is a template for the rho of the run (D19); with
        the default it is the line decided for D18."""
        expected = {"FLOW_COSMOS_SIGMA_MAX": 80.0, "FLOW_COSMOS_SIGMA_MIN": 0.002, "FLOW_COSMOS_RHO": 7.0,
                    "FLOW_COSMOS_NOT_FLOW_LOG": "[Extra Schedulers] Flow Cosmos rho7: not a flow model - using Karras "
                                                "rho {rho:g} on the model's own sigma range"}
        tree = ast.parse((ROOT / "sam3ext" / "extra_schedulers" / "schedulers.py").read_text(encoding="utf-8"))
        for name, value in expected.items():
            with self.subTest(name=name):
                literal = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                               and [getattr(t, "id", None) for t in node.targets] == [name])
                self.assertIsInstance(literal, ast.Constant)
                self.assertEqual(literal.value, value)
                self.assertEqual(getattr(sch, name), value)
        self.assertEqual(sch.FLOW_COSMOS_NOT_FLOW_LOG.format(rho=st.FLOW_COSMOS_RHO_DEFAULT),
                         "[Extra Schedulers] Flow Cosmos rho7: not a flow model - using Karras rho 7 on the model's "
                         "own sigma range")
        defaults = st.DEFAULTS
        self.assertEqual((defaults.flow_cosmos_rho, defaults.flow_cosmos_sigma_max, defaults.flow_cosmos_sigma_min),
                         (sch.FLOW_COSMOS_RHO, sch.FLOW_COSMOS_SIGMA_MAX, sch.FLOW_COSMOS_SIGMA_MIN))

    def test_flow_values_are_the_upstream_numbers_bit_for_bit(self):
        """Same float32 list as KeithZ117's builder (inline reference above) on every flow range — the list does not
        depend on the model's range. In float64 the two differ in the last bit for most n (the reference adds
        i·((r_min − r_max)/(n − 1)), the scheduler x·(r_min − r_max) like the A/B expression). In float32 they are
        equal for every n up to 30,000 (checked on the development PC); at very large n one element can differ by
        1 float32 ulp (e.g. n = 180,836 at index 166,745) — the A/B expression's operation order is the one kept."""
        for range_name, (lo, hi) in self.FLOW_RANGES.items():
            for n in sorted(set(COSMOS_STEPS[1:]) | set(range(2, 301))):
                with self.subTest(range=range_name, steps=n):
                    ours = sch.flow_cosmos_rho7(n=n, sigma_min=lo, sigma_max=hi, inner_model=FLOW_MODEL, device="cpu")
                    expected = torch.tensor(upstream_flow_cosmos_rho(n), dtype=torch.float32)
                    self.assertTrue(torch.equal(ours, expected))
                    self.assertEqual(ours.dtype, torch.float32)

    def test_flow_values_are_the_ab_custom_expression_bit_for_bit(self):
        """The A/B expression through the custom scheduler's safe evaluator (another code path): the same float64
        values before the cast, so the same float32 list, on every flow range including Anima's."""
        st.set_active(st.ExtraSchedulerSettings(custom_mode=st.MODE_EXPRESSION, custom_expression=COSMOS_AB_EXPRESSION))
        for range_name, (lo, hi) in self.FLOW_RANGES.items():
            for n in sorted(set(COSMOS_STEPS) | set(range(1, 201))):
                with self.subTest(range=range_name, steps=n):
                    self.assertTrue(torch.equal(sch.custom(n, lo, hi),
                                                sch.flow_cosmos_rho7(n, lo, hi, inner_model=FLOW_MODEL)))
                    self.assertEqual(sch.sigmas_from_expression(COSMOS_AB_EXPRESSION, n, lo, hi),
                                     sch._flow_cosmos_times(n))

    def test_the_flow_list_from_the_formula(self):
        """t = s/(1 + s) of s = (80^(1/7) + p(0.002^(1/7) − 80^(1/7)))^7: 80/81 first, ≈ 0.002/1.002 last, one step ⇒
        just 80/81 (p = 0, like the other schedulers), no step ⇒ only the final 0."""
        lo, hi = self.FLOW_RANGES["anima"]
        top = float(torch.tensor(80.0 / 81.0, dtype=torch.float32))
        for n in COSMOS_STEPS:
            with self.subTest(steps=n):
                sigmas = sch.flow_cosmos_rho7(n, lo, hi, inner_model=FLOW_MODEL)
                self.assertEqual(sigmas.shape, (n + 1,))
                self.assertEqual(float(sigmas[0]), top)
                self.assertEqual(float(sigmas[-1]), 0.0)
                if n > 1:
                    s_min = (0.002 ** (1 / 7)) ** 7
                    self.assertEqual(float(sigmas[-2]), float(torch.tensor(s_min / (1 + s_min), dtype=torch.float32)))
                    self.assertAlmostEqual(float(sigmas[-2]), 0.002 / 1.002, delta=1e-9)
        self.assertEqual(sch.flow_cosmos_rho7(1, lo, hi, inner_model=FLOW_MODEL).tolist(), [top, 0.0])
        self.assertEqual(sch.flow_cosmos_rho7(0, lo, hi, inner_model=FLOW_MODEL).tolist(), [0.0])
        # high-σ-heavy, like any Karras ramp seen in flow time: 28 steps → 17 above 0.5, 5 below 0.05
        sigmas = sch.flow_cosmos_rho7(28, lo, hi, inner_model=FLOW_MODEL)[:-1]
        self.assertEqual(int((sigmas > 0.5).sum()), 17)
        self.assertEqual(int((sigmas < 0.05).sum()), 5)

    def test_strictly_falling_until_float32_runs_out_of_room(self):
        lo, hi = self.FLOW_RANGES["anima"]
        for n in range(2, 1001):
            sigmas = sch.flow_cosmos_rho7(n, lo, hi, inner_model=FLOW_MODEL)
            self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()), n)
        sigmas = sch.flow_cosmos_rho7(COSMOS_FIRST_FLAT_STEPS - 1, lo, hi, inner_model=FLOW_MODEL)
        self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))
        with self.assertRaises(sch.ExtraSchedulerError) as ctx:
            sch.flow_cosmos_rho7(COSMOS_FIRST_FLAT_STEPS, lo, hi, inner_model=FLOW_MODEL)
        self.assertNotIsInstance(ctx.exception, (sch.CustomSchedulerError, sch.KarrasDynamicError))
        self.assertIn("Extra Schedulers: Flow Cosmos rho7 gives sigma 0.987626 at steps 467 and 468", str(ctx.exception))
        self.assertIn(sch.ZERO_LENGTH_STEP, str(ctx.exception))

    def test_eps_and_v_models_get_forges_karras_rho_7_on_the_range_forge_passes(self):
        """Not a flow model: Forge's Karras (the stand-in from setUp; Forge's own function: the Forge class below)
        with ρ 7 on whichever sigma range Forge passes — the model's, or the Settings sigma min / max overrides."""
        for model in (EPS_MODEL, V_MODEL):
            for lo, hi in (RANGES["eps"], (0.1, 20.0), (0.0291675, 1.0)):
                for n in COSMOS_STEPS:
                    with self.subTest(kind=model.predictor.prediction_type, range=(lo, hi), steps=n):
                        self.karras_calls.clear()
                        ours = sch.flow_cosmos_rho7(n, lo, hi, inner_model=model)
                        self.assertEqual(self.karras_calls, [(n, lo, hi, 7.0, "cpu")])
                        self.assertTrue(torch.equal(ours, karras_formula(n, lo, hi)))
                        self.assertEqual(ours.dtype, torch.float32)
                        if n > 1:
                            self.assertAlmostEqual(float(ours[0]) / hi, 1.0, delta=1e-6)
                            self.assertAlmostEqual(float(ours[-2]) / lo, 1.0, delta=1e-6)

    def test_the_eps_v_branch_calls_forges_karras_at_run_time(self):
        """Forge's ``k_diffusion.sampling.get_sigmas_karras`` is looked up at each call and its own tensor is the
        schedule — nothing of it is copied here (THIRD_PARTY_NOTICES), so it is Forge's Karras by construction."""
        calls = []
        forge_list = torch.tensor([9.0, 4.0, 1.0, 0.0])

        def get_sigmas_karras(*args, **kwargs):
            calls.append((args, kwargs))
            return forge_list

        cpu = torch.device("cpu")
        with fixtures.installed_k_sampling(_k_sampling_with(get_sigmas_karras)):
            sigmas = sch.flow_cosmos_rho7(3, 0.5, 9.0, inner_model=EPS_MODEL, device=cpu)
        self.assertTrue(torch.equal(sigmas, forge_list))
        self.assertEqual(calls, [((3, 0.5, 9.0), {"rho": 7.0, "device": cpu})])
        self.karras_calls.clear()
        sch.flow_cosmos_rho7(3, 0.5, 9.0, inner_model=V_MODEL)   # the module installed now, not one bound before
        self.assertEqual(self.karras_calls, [(3, 0.5, 9.0, 7.0, "cpu")])
        # Forge's list still goes through the flat-step rule
        with fixtures.installed_k_sampling(_k_sampling_with(lambda *a, **k: torch.tensor([9.0, 4.0, 4.0, 0.0]))):
            with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                sch.flow_cosmos_rho7(3, 0.5, 9.0, inner_model=EPS_MODEL)
        self.assertIn("Flow Cosmos rho7 gives sigma 4 at steps 1 and 2", str(ctx.exception))

    def test_without_forges_karras_an_eps_v_model_stops_and_a_flow_model_does_not_need_it(self):
        flow_list = sch._with_final_zero(sch._flow_cosmos_times(10), "cpu")
        absent = {"no module": None, "no function": _k_sampling_with(_MISSING), "not callable": _k_sampling_with(None)}
        for case, module in absent.items():
            stderr = io.StringIO()
            with self.subTest(case=case), support.stub_modules({"k_diffusion.sampling": module}), \
                    mock.patch.object(sch, "_flow_cosmos_not_flow_logged", False), contextlib.redirect_stderr(stderr):
                with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                    sch.flow_cosmos_rho7(10, 0.03, 14.6, inner_model=EPS_MODEL)
                self.assertIn("k_diffusion.sampling.get_sigmas_karras", str(ctx.exception))
                self.assertTrue(torch.equal(sch.flow_cosmos_rho7(10, 0.03, 1.0, inner_model=FLOW_MODEL), flow_list))
                self.assertEqual(stderr.getvalue(), "", "no 'using Karras' line when Forge's Karras cannot run")

    def test_not_a_flow_model_is_logged_once_per_process(self):
        stderr = io.StringIO()
        with mock.patch.object(sch, "_flow_cosmos_not_flow_logged", False), contextlib.redirect_stderr(stderr):
            sch.flow_cosmos_rho7(10, 0.03, 1.0, inner_model=FLOW_MODEL)
            self.assertEqual(stderr.getvalue(), "", "a flow model logs nothing")
            sch.flow_cosmos_rho7(10, 0.03, 14.6, inner_model=EPS_MODEL)
            sch.flow_cosmos_rho7(20, 0.03, 14.6, inner_model=V_MODEL)
            sch.flow_cosmos_rho7(10, 0.03, 14.6, inner_model=EPS_MODEL)
        self.assertEqual(stderr.getvalue(), "[Extra Schedulers] Flow Cosmos rho7: not a flow model - using Karras rho 7 "
                                            "on the model's own sigma range\n")

    def test_without_a_model_parameterisation_it_stops(self):
        """Forge always passes its model_wrap (need_inner_model); anything that cannot tell flow from eps/v stops
        with ExtraSchedulerError instead of guessing."""
        for inner_model in (None, types.SimpleNamespace(), types.SimpleNamespace(predictor=types.SimpleNamespace())):
            with self.subTest(inner_model=inner_model):
                with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                    sch.flow_cosmos_rho7(10, 0.03, 1.0, inner_model=inner_model)
                self.assertIn("Flow Cosmos rho7 needs Forge's model", str(ctx.exception))
        with self.assertRaises(sch.ExtraSchedulerError):
            sch.flow_cosmos_rho7(10, 0.03, 1.0)

    def test_needs_positive_sigmas(self):
        for model in (FLOW_MODEL, EPS_MODEL):
            for lo, hi in ((0.0, 1.0), (-0.1, 14.6), (0.003, 0.0)):
                with self.subTest(kind=model.predictor.prediction_type, lo=lo, hi=hi):
                    with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                        sch.flow_cosmos_rho7(10, lo, hi, inner_model=model)
                    self.assertIn("Flow Cosmos rho7 needs sigma_min and sigma_max above 0", str(ctx.exception))

    def test_a_flat_karras_range_is_refused(self):
        """Settings → sigma min = sigma max on an eps model: every step the same sigma — refused like the others."""
        with self.assertRaises(sch.ExtraSchedulerError) as ctx:
            sch.flow_cosmos_rho7(10, 2.0, 2.0, inner_model=EPS_MODEL)
        self.assertIn("Flow Cosmos rho7 gives sigma 2 at steps 0 and 1", str(ctx.exception))

    def test_device_argument(self):
        for model in (FLOW_MODEL, EPS_MODEL):
            sigmas = sch.flow_cosmos_rho7(4, 0.03, 1.0, inner_model=model, device=torch.device("cpu"))
            self.assertEqual(sigmas.device.type, "cpu")


def cosmos_expression(rho, ve_sigma_max, ve_sigma_min):
    """The A/B custom expression with other numbers in place of 7, 80 and 0.002 (repr: the exact floats)."""
    r, top, bottom = repr(float(rho)), repr(float(ve_sigma_max)), repr(float(ve_sigma_min))
    ramp = f"({top}**(1/{r})+x*({bottom}**(1/{r})-{top}**(1/{r})))**{r}"
    return f"{ramp}/(1+{ramp})"


def oracle_flow_cosmos(n, rho, ve_sigma_max, ve_sigma_min):
    """Independent float64 form (numpy): s = σ̃max((1 − p) + p(σ̃min/σ̃max)^(1/ρ))^ρ, t = 1/(1 + 1/s) — the same values
    as s/(1 + s) of the ρ-ramp, written differently (so its rounding differs)."""
    p = _p(n)
    s = ve_sigma_max * ((1.0 - p) + p * (ve_sigma_min / ve_sigma_max) ** (1.0 / rho)) ** rho
    return 1.0 / (1.0 + 1.0 / s)


def _float32_ulps(a, b) -> int:
    """Largest distance in float32 units in the last place between two float32 arrays of positive numbers."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return int(np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64)).max(initial=0))


def _cosmos_settings(rho=st.FLOW_COSMOS_RHO_DEFAULT, sigma_max=st.FLOW_COSMOS_SIGMA_MAX_DEFAULT,
                     sigma_min=st.FLOW_COSMOS_SIGMA_MIN_DEFAULT, **others):
    return st.ExtraSchedulerSettings(flow_cosmos_rho=rho, flow_cosmos_sigma_max=sigma_max,
                                     flow_cosmos_sigma_min=sigma_min, **others)


class FlowCosmosKnobTests(unittest.TestCase):
    """D19: the accordion's ρ, σ̃ max and σ̃ min, read when the scheduler runs (``settings.active()``, like Laplace)."""

    RHOS = (1.0, 2.5, 5.0, 7.0, 9.3, 15.0)
    # (σ̃ max, σ̃ min): Cosmos's, ×3 (Anima's shift 3), the slider corners, SDXL's model range, a narrow one
    VE_RANGES = ((80.0, 0.002), (240.0, 0.006), (1.0, 0.0001), (1000.0, 1.0), (1000.0, 0.0001),
                 (14.614642, 0.0291675), (1.5, 0.75))
    STEPS = (1, 2, 3, 10, 28, 151)
    ANIMA = _anima_range()

    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)
        patcher = mock.patch.object(sch, "_flow_cosmos_not_flow_logged", True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.karras_calls = []

        def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
            self.karras_calls.append((n, sigma_min, sigma_max, rho, device))
            return karras_formula(n, sigma_min, sigma_max, rho=rho, device=device)

        self.enterContext(fixtures.installed_k_sampling(_k_sampling_with(get_sigmas_karras)))

    def flow(self, n, settings=None, lo=None, hi=None):
        if settings is not None:
            st.set_active(settings)
        lo, hi = (self.ANIMA if lo is None else (lo, hi))
        return sch.flow_cosmos_rho7(n, lo, hi, inner_model=FLOW_MODEL)

    def test_the_defaults_are_the_list_from_before(self):
        """Defaults (reset, or the same numbers set explicitly) == the D18 list == the A/B custom expression, bit for
        bit — also through the script's coercion of an old request without these values; and the A/B expression is
        ``cosmos_expression`` with the defaults."""
        ab_settings = st.ExtraSchedulerSettings(custom_mode=st.MODE_EXPRESSION, custom_expression=COSMOS_AB_EXPRESSION)
        same = (st.DEFAULTS, _cosmos_settings(7.0, 80.0, 0.002), st.coerce({}),
                st.coerce({"custom_mode": "expression", "laplace_mu": 0.0, "laplace_beta": 0.5}),
                st.coerce({"flow_cosmos_rho": "7", "flow_cosmos_sigma_max": 80, "flow_cosmos_sigma_min": "0.002"}))
        for n in sorted(set(COSMOS_STEPS) | set(range(1, 121))):
            with self.subTest(steps=n):
                st.set_active(ab_settings)   # the default Flow Cosmos values
                ab = sch.custom(n, *self.ANIMA)
                self.assertTrue(torch.equal(self.flow(n), ab))
                for settings in same:
                    self.assertTrue(torch.equal(self.flow(n, settings), ab))
                if n > 1:
                    self.assertTrue(torch.equal(ab, torch.tensor(upstream_flow_cosmos_rho(n), dtype=torch.float32)))
                self.assertEqual(sch.sigmas_from_expression(cosmos_expression(7, 80, 0.002), n, *self.ANIMA),
                                 sch.sigmas_from_expression(COSMOS_AB_EXPRESSION, n, *self.ANIMA))

    def test_the_values_are_the_custom_expression_with_those_numbers_bit_for_bit(self):
        """Operation order: the A/B expression with ρ, σ̃ max and σ̃ min in place of 7, 80 and 0.002, through the safe
        evaluator (another code path), gives the same float64 values, so the same float32 list."""
        for rho in self.RHOS:
            for top, bottom in self.VE_RANGES:
                expression = cosmos_expression(rho, top, bottom)
                for n in self.STEPS:
                    with self.subTest(rho=rho, sigma_max=top, sigma_min=bottom, steps=n):
                        settings = _cosmos_settings(rho, top, bottom, custom_expression=expression)
                        st.set_active(settings)
                        self.assertEqual(sch.sigmas_from_expression(expression, n, *self.ANIMA),
                                         sch._flow_cosmos_times(n, rho, bottom, top))
                        self.assertTrue(torch.equal(self.flow(n), sch.custom(n, *self.ANIMA)))

    def test_a_grid_of_values_against_an_independent_formula(self):
        """float64 against ``oracle_flow_cosmos`` (another form of the formula), within the error the ramp's
        subtraction allows — σ̃max^(1/ρ) + p(σ̃min^(1/ρ) − σ̃max^(1/ρ)) loses about ρ·ε·(σ̃max/σ̃min)^(1/ρ) relative at
        the end (the A/B expression's order, also k-diffusion's Karras order); then the list is the float32 rounding of
        those float64 values with the final 0, within 1 float32 ulp of the oracle's rounding."""
        eps = float(np.finfo(np.float64).eps)
        for rho in self.RHOS:
            for top, bottom in self.VE_RANGES:
                rtol = 64 * eps * rho * (top / bottom) ** (1.0 / rho)
                for n in self.STEPS:
                    with self.subTest(rho=rho, sigma_max=top, sigma_min=bottom, steps=n):
                        times = np.array(sch._flow_cosmos_times(n, rho, bottom, top), dtype=np.float64)
                        expected = oracle_flow_cosmos(n, rho, top, bottom)
                        np.testing.assert_allclose(times, expected, rtol=rtol, atol=0)
                        sigmas = self.flow(n, _cosmos_settings(rho, top, bottom))
                        self.assertEqual(sigmas.dtype, torch.float32)
                        self.assertEqual(sigmas.shape, (n + 1,))
                        self.assertEqual(float(sigmas[-1]), 0.0)
                        np.testing.assert_array_equal(sigmas[:-1].numpy(), times.astype(np.float32))
                        self.assertLessEqual(_float32_ulps(sigmas[:-1].numpy(), expected), 1)
                        # σ̃max/(1 + σ̃max) first, σ̃min/(1 + σ̃min) last, strictly falling in between
                        self.assertLessEqual(_float32_ulps(sigmas[:1].numpy(), [top / (1.0 + top)]), 1)
                        if n > 1:
                            self.assertLessEqual(_float32_ulps(sigmas[-2:-1].numpy(), [bottom / (1.0 + bottom)]), 1)
                            self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))

    def test_scaling_the_range_is_the_flow_time_shift(self):
        """t' = k·t/(1 + (k − 1)·t) ⇔ σ̃' = k·σ̃, and k·σ̃ of every ramp value is the ramp over k·σ̃min … k·σ̃max: Anima's
        shift 3 applied to the default list is the list of σ̃ 0.006 … 240 (float64, to rounding)."""
        for k in (3.0, 2.0, 0.5):
            for rho in (7.0, 4.0):
                for n in (2, 3, 28, 151):
                    with self.subTest(k=k, rho=rho, steps=n):
                        base = np.array(sch._flow_cosmos_times(n, rho, 0.002, 80.0))
                        scaled = np.array(sch._flow_cosmos_times(n, rho, 0.002 * k, 80.0 * k))
                        np.testing.assert_allclose(k * base / (1.0 + (k - 1.0) * base), scaled, rtol=1e-13)

    def test_the_values_are_read_when_the_scheduler_runs(self):
        first = self.flow(28)
        moved = self.flow(28, _cosmos_settings(5.0, 240.0, 0.006))
        self.assertFalse(torch.equal(first, moved))
        self.assertTrue(torch.equal(moved, sch._with_final_zero(sch._flow_cosmos_times(28, 5.0, 0.006, 240.0), "cpu")))
        st.reset()
        self.assertTrue(torch.equal(self.flow(28), first))

    def test_on_a_flow_model_the_models_range_is_still_not_used(self):
        st.set_active(_cosmos_settings(4.5, 300.0, 0.01))
        lists = [sch.flow_cosmos_rho7(20, lo, hi, inner_model=FLOW_MODEL)
                 for lo, hi in (self.ANIMA, RANGES["flow"], (0.05, 0.9), RANGES["eps"])]
        for sigmas in lists[1:]:
            self.assertTrue(torch.equal(sigmas, lists[0]))

    def test_eps_and_v_models_get_forges_karras_with_the_rho(self):
        """Not a flow model: Forge's Karras with the accordion's ρ on the range Forge passes; σ̃ max / σ̃ min are not
        used there (even values a flow model would refuse)."""
        for model in (EPS_MODEL, V_MODEL):
            for rho in (1.0, 3.3, 7.0, 15.0):
                for lo, hi in (RANGES["eps"], (0.1, 20.0)):
                    for ve in ((80.0, 0.002), (240.0, 0.006), (1.0, 1.0)):
                        with self.subTest(kind=model.predictor.prediction_type, rho=rho, range=(lo, hi), ve=ve):
                            st.set_active(_cosmos_settings(rho, *ve))
                            self.karras_calls.clear()
                            ours = sch.flow_cosmos_rho7(28, lo, hi, inner_model=model)
                            self.assertEqual(self.karras_calls, [(28, lo, hi, rho, "cpu")])
                            self.assertTrue(torch.equal(ours, karras_formula(28, lo, hi, rho=rho)))

    def test_the_log_line_names_the_rho_of_the_first_eps_v_run(self):
        stderr = io.StringIO()
        with mock.patch.object(sch, "_flow_cosmos_not_flow_logged", False), contextlib.redirect_stderr(stderr):
            st.set_active(_cosmos_settings(rho=5.5))
            sch.flow_cosmos_rho7(10, 0.03, 14.6, inner_model=EPS_MODEL)
            st.reset()
            sch.flow_cosmos_rho7(10, 0.03, 14.6, inner_model=EPS_MODEL)
        self.assertEqual(stderr.getvalue(), "[Extra Schedulers] Flow Cosmos rho7: not a flow model - using Karras "
                                            "rho 5.5 on the model's own sigma range\n")

    def test_a_rho_that_is_not_finite_and_above_0_stops_the_generation(self):
        """The sliders keep ρ in 1 … 15 (``settings``), so only a value set past them gets here; both branches."""
        for rho in (0.0, -1.0, -0.0, math.nan, math.inf, -math.inf):
            for model in (FLOW_MODEL, EPS_MODEL):
                with self.subTest(rho=rho, kind=model.predictor.prediction_type):
                    st.set_active(_cosmos_settings(rho=rho))
                    self.karras_calls.clear()
                    with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                        sch.flow_cosmos_rho7(10, 0.03, 14.6, inner_model=model)
                    self.assertNotIsInstance(ctx.exception, (sch.CustomSchedulerError, sch.KarrasDynamicError))
                    self.assertIn("Flow Cosmos rho7 needs a finite rho above 0", str(ctx.exception))
                    self.assertIn(sch.FLOW_COSMOS_VALUES_SOURCE, str(ctx.exception))
                    self.assertEqual(self.karras_calls, [], "Forge's Karras is not called")

    def test_a_sigma_range_that_is_not_0_below_min_below_max_stops_a_flow_model(self):
        """0 < σ̃ min < σ̃ max, both finite. From the sliders only σ̃ min = σ̃ max = 1 (where their ranges meet) gets
        here."""
        bad = ((1.0, 1.0), (80.0, 80.0), (1.0, 0.5), (0.0, 80.0), (-0.002, 80.0), (math.nan, 80.0), (0.002, math.nan),
               (0.002, math.inf), (math.inf, math.inf), (-math.inf, 80.0), (0.002, 0.0))
        for sigma_min, sigma_max in bad:
            with self.subTest(sigma_min=sigma_min, sigma_max=sigma_max):
                st.set_active(_cosmos_settings(7.0, sigma_max, sigma_min))
                with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                    self.flow(10)
                self.assertNotIsInstance(ctx.exception, (sch.CustomSchedulerError, sch.KarrasDynamicError))
                self.assertIn("Flow Cosmos rho7 needs 0 < sigma~ min < sigma~ max, both finite", str(ctx.exception))
                self.assertIn(sch.FLOW_COSMOS_VALUES_SOURCE, str(ctx.exception))
        for n in (0, 1):   # also without steps to fall: the values are checked, not only the result
            with self.subTest(steps=n):
                with self.assertRaises(sch.ExtraSchedulerError):
                    self.flow(n, _cosmos_settings(7.0, 1.0, 1.0))
        # just inside: no error
        self.assertTrue(torch.isfinite(self.flow(10, _cosmos_settings(7.0, 1.0, 0.9999))).all())

    def test_values_past_the_sliders_that_overflow_stop_the_generation(self):
        st.set_active(_cosmos_settings(rho=0.001))   # 80 ** 1000 overflows a float
        with self.assertRaises(sch.ExtraSchedulerError) as ctx:
            self.flow(10)
        self.assertIn("Flow Cosmos rho7 with rho 0.001 on sigma~ 0.002 … 80 overflows", str(ctx.exception))
        # eps/v: Forge's Karras raises OverflowError itself (14.6 ** 1000) — the same error, naming the range it got
        for model in (EPS_MODEL, V_MODEL):
            with self.subTest(kind=model.predictor.prediction_type):
                with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                    sch.flow_cosmos_rho7(10, 0.03, 14.6, inner_model=model)
                self.assertNotIsInstance(ctx.exception, (sch.CustomSchedulerError, sch.KarrasDynamicError))
                self.assertIn("Flow Cosmos rho7 with rho 0.001 on sigma 0.03 … 14.6 overflows", str(ctx.exception))
                self.assertIn(sch.FLOW_COSMOS_VALUES_SOURCE, str(ctx.exception))

    def test_a_sigma_below_float32_tiny_before_the_final_0_stops_the_generation(self):
        """Values past the sliders (only reachable without the coercion) can round the last sigma before the final 0
        to 0 — the ramp's subtraction cancels at p = 1 for a tiny ρ (both branches) or a σ̃ min far below float64's
        resolution of σ̃ max^(1/ρ) (flow) — or make Forge's float32 Karras overflow to NaN. The flat-step rule does
        not see [..., 0, 0]; like the custom scheduler, every sigma before the final 0 must be a float32 number of at
        least the smallest normal float32 (Forge's samplers divide by sigma)."""
        cases = (("flow", FLOW_MODEL, _cosmos_settings(rho=0.05)), ("eps", EPS_MODEL, _cosmos_settings(rho=0.05)),
                 ("v", V_MODEL, _cosmos_settings(rho=0.05)), ("eps NaN", EPS_MODEL, _cosmos_settings(rho=0.02)),
                 ("flow", FLOW_MODEL, _cosmos_settings(sigma_min=5e-324)),
                 ("flow", FLOW_MODEL, _cosmos_settings(sigma_min=1e-300)),
                 ("flow", FLOW_MODEL, _cosmos_settings(rho=1.0, sigma_min=1e-300)))
        for kind, model, settings in cases:
            with self.subTest(kind=kind, rho=settings.flow_cosmos_rho, sigma_min=settings.flow_cosmos_sigma_min):
                st.set_active(settings)
                with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                    sch.flow_cosmos_rho7(28, 0.0291675, 14.614642, inner_model=model)
                message = str(ctx.exception)
                self.assertNotIsInstance(ctx.exception, (sch.CustomSchedulerError, sch.KarrasDynamicError))
                self.assertIn("Extra Schedulers: Flow Cosmos rho7 gives sigma ", message)
                self.assertIn(f"the smallest usable value is {sch._FLOAT32_TINY:.4g}", message)
                self.assertIn(sch.FLOW_COSMOS_VALUES_SOURCE, message)
        # the rule itself, on lists no branch makes from the sliders: inf, NaN, negative, subnormal, 0 — the final 0 is fine
        for head in ([math.inf, 0.5], [math.nan, 0.5], [0.5, -0.25], [0.5, 1e-40], [0.5, 0.0], [0.5, 0.25, 0.0]):
            with self.subTest(rule=head):
                with self.assertRaises(sch.ExtraSchedulerError):
                    sch._check_flow_cosmos_usable(torch.tensor([*head, 0.0], dtype=torch.float32), 7.0, "test")
        sch._check_flow_cosmos_usable(torch.tensor([0.5, sch._FLOAT32_TINY, 0.0], dtype=torch.float32), 7.0, "test")
        sch._check_flow_cosmos_usable(torch.tensor([0.0], dtype=torch.float32), 7.0, "test")   # n = 0
        # σ̃ min 1e-30 is tiny but still a normal float32 at the end: a usable, strictly falling list
        sigmas = self.flow(28, _cosmos_settings(sigma_min=1e-30))
        self.assertGreaterEqual(float(sigmas[-2]), sch._FLOAT32_TINY)
        self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))
        # the slider corners are far from it on both branches
        for rho in (st.FLOW_COSMOS_RHO_MIN, st.FLOW_COSMOS_RHO_MAX):
            for top, bottom in ((st.FLOW_COSMOS_SIGMA_MAX_MAX, st.FLOW_COSMOS_SIGMA_MIN_MIN),
                                (st.FLOW_COSMOS_SIGMA_MAX_MIN, st.FLOW_COSMOS_SIGMA_MIN_MIN)):
                for model in (FLOW_MODEL, EPS_MODEL):
                    with self.subTest(corner=(rho, top, bottom), kind=model.predictor.prediction_type):
                        st.set_active(_cosmos_settings(rho, top, bottom))
                        sigmas = sch.flow_cosmos_rho7(1000, 0.0291675, 14.614642, inner_model=model)
                        self.assertGreater(float(sigmas[-2]), 1e-5)

    def test_a_flat_step_from_a_narrow_range_names_these_values(self):
        """The flat-step rule is unchanged; on a flow model its message points at these values, not at Settings →
        sigma min / max (which a flow model's list does not use)."""
        st.set_active(_cosmos_settings(7.0, 1.0, 0.9999))
        self.assertTrue(bool((self.flow(500)[1:] < self.flow(500)[:-1]).all()))
        with self.assertRaises(sch.ExtraSchedulerError) as ctx:
            self.flow(1000)
        message = str(ctx.exception)
        self.assertIn("Extra Schedulers: Flow Cosmos rho7 gives sigma 0.5 at steps 3 and 4", message)
        self.assertIn("rho 7, sigma~ 0.9999 … 1 — " + sch.FLOW_COSMOS_VALUES_SOURCE, message)
        self.assertIn(sch.ZERO_LENGTH_STEP, message)
        self.assertNotIn("Settings → Sampler Parameters", message)
        # where it starts (README: "σ̃ 0.9999~1 은 841 스텝부터"): 840 steps fall strictly, 841 have a flat step, for ρ
        # at both slider ends and the default (a scan of every slider ρ 1.0 … 15.0 found 841 for all of them)
        for rho in (st.FLOW_COSMOS_RHO_MIN, 7.0, st.FLOW_COSMOS_RHO_MAX):
            with self.subTest(rho=rho):
                st.set_active(_cosmos_settings(rho, 1.0, 0.9999))
                sigmas = self.flow(840)
                self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))
                with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                    self.flow(841)
                self.assertIn("Extra Schedulers: Flow Cosmos rho7 gives sigma 0.499989 at steps ", str(ctx.exception))
        # the eps/v branch keeps the Settings note (its range is the one Forge passes)
        with self.assertRaises(sch.ExtraSchedulerError) as ctx:
            sch.flow_cosmos_rho7(10, 2.0, 2.0, inner_model=EPS_MODEL)
        self.assertIn("Settings → Sampler Parameters → sigma min / sigma max", str(ctx.exception))


@unittest.skipUnless(support.forge_root(), "Forge checkout not found")
class FlowCosmosRho7ForgeTests(unittest.TestCase):
    """Against Forge's own code: k-diffusion's ``get_sigmas_karras``, ``ForgeScheduleLinker`` (Forge's
    ``model_wrap``, what ``get_sigmas`` passes as ``inner_model``) around Forge's real predictors."""

    @classmethod
    def setUpClass(cls):
        sampling_path = "modules_forge/packages/k_diffusion/sampling.py"
        namespace = {"torch": torch, "math": math}
        append_zero = support.forge_function(sampling_path, "append_zero", namespace)
        cls.karras = staticmethod(support.forge_function(sampling_path, "get_sigmas_karras", namespace))
        linker_namespace = {"torch": torch, "nn": torch.nn, "sampling": types.SimpleNamespace(append_zero=append_zero)}
        cls.linker = staticmethod(support.forge_function("modules_forge/packages/k_diffusion/external.py",
                                                         "ForgeScheduleLinker", linker_namespace))
        prediction = fixtures.forge_k_prediction()
        cls.predictors = {
            "flow (Anima, shift 3)": fixtures.forge_flow_predictor(3.0),
            "flow (shift 1)": fixtures.forge_flow_predictor(1.0),
            "eps (SDXL)": fixtures.forge_eps_predictor(),
            "v (SDXL betas)": prediction.Prediction(sigma_data=1.0, prediction_type="v_prediction",
                                                    beta_schedule="linear", linear_start=0.00085, linear_end=0.012,
                                                    timesteps=1000),
        }

    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)
        patcher = mock.patch.object(sch, "_flow_cosmos_not_flow_logged", True)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Forge's k_diffusion.sampling, executed from Forge's file: the eps/v branch calls its get_sigmas_karras
        self.enterContext(fixtures.installed_k_sampling(fixtures.forge_k_sampling()))

    def _model(self, name):
        model_wrap = self.linker(self.predictors[name])
        # KDiffusionSampler.get_sigmas: m_sigma_min, m_sigma_max = model_wrap.sigmas[0].item(), [-1].item()
        return model_wrap, model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

    def test_forges_flow_predictors_get_the_flow_list(self):
        for name in ("flow (Anima, shift 3)", "flow (shift 1)"):
            model_wrap, lo, hi = self._model(name)
            for n in COSMOS_STEPS[1:]:
                with self.subTest(model=name, steps=n):
                    ours = sch.flow_cosmos_rho7(n=n, sigma_min=lo, sigma_max=hi, inner_model=model_wrap, device="cpu")
                    self.assertTrue(torch.equal(ours, torch.tensor(upstream_flow_cosmos_rho(n), dtype=torch.float32)))
        self.assertEqual(round(self._model("flow (Anima, shift 3)")[1], 9), 0.002994012)

    def test_forges_eps_and_v_predictors_get_forges_karras_bit_for_bit(self):
        for name in ("eps (SDXL)", "v (SDXL betas)"):
            model_wrap, lo, hi = self._model(name)
            for n in COSMOS_STEPS:
                with self.subTest(model=name, steps=n):
                    ours = sch.flow_cosmos_rho7(n=n, sigma_min=lo, sigma_max=hi, inner_model=model_wrap, device="cpu")
                    self.assertTrue(torch.equal(ours, self.karras(n, lo, hi, rho=7.0)))
                    self.assertTrue(torch.equal(ours, self.karras(n, lo, hi)))   # Forge's Karras default rho
            # Settings → sigma min / sigma max overrides reach it like Forge's Karras
            ours = sch.flow_cosmos_rho7(28, 0.1, 10.0, inner_model=model_wrap)
            self.assertTrue(torch.equal(ours, self.karras(28, 0.1, 10.0)))
        self.assertAlmostEqual(self._model("eps (SDXL)")[2], 14.614642, places=5)

    def test_the_accordion_values_with_forges_predictors(self):
        """D19 with Forge's code: on eps/v Forge's Karras gets the accordion's ρ (σ̃ range unused); on flow the list is the
        formula with ρ, σ̃ max, σ̃ min, whatever the model's shift."""
        for rho, top, bottom in ((5.0, 240.0, 0.006), (1.0, 1000.0, 0.0001), (15.0, 1.0, 0.5)):
            st.set_active(_cosmos_settings(rho, top, bottom))
            for name in ("eps (SDXL)", "v (SDXL betas)"):
                model_wrap, lo, hi = self._model(name)
                for n in COSMOS_STEPS:
                    with self.subTest(model=name, rho=rho, steps=n):
                        ours = sch.flow_cosmos_rho7(n=n, sigma_min=lo, sigma_max=hi, inner_model=model_wrap)
                        self.assertTrue(torch.equal(ours, self.karras(n, lo, hi, rho=rho)))
            expected = {n: sch._with_final_zero(sch._flow_cosmos_times(n, rho, bottom, top), "cpu")
                        for n in COSMOS_STEPS}
            for name in ("flow (Anima, shift 3)", "flow (shift 1)"):
                model_wrap, lo, hi = self._model(name)
                for n in COSMOS_STEPS:
                    with self.subTest(model=name, rho=rho, steps=n):
                        ours = sch.flow_cosmos_rho7(n=n, sigma_min=lo, sigma_max=hi, inner_model=model_wrap)
                        self.assertTrue(torch.equal(ours, expected[n]))

    def test_values_past_the_sliders_with_forges_karras(self):
        """Forge's own get_sigmas_karras with a ρ past the sliders: its OverflowError (ρ 0.001), its 0 before the final 0
        (ρ 0.05) and its NaN (ρ 0.02, float32 overflow) all stop the generation with ExtraSchedulerError."""
        for name in ("eps (SDXL)", "v (SDXL betas)"):
            model_wrap, lo, hi = self._model(name)
            for rho, words in ((0.001, "overflows"), (0.05, "gives sigma 0 at step 27"),
                               (0.02, "gives sigma nan at step 0")):
                with self.subTest(model=name, rho=rho):
                    st.set_active(_cosmos_settings(rho=rho))
                    with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                        sch.flow_cosmos_rho7(n=28, sigma_min=lo, sigma_max=hi, inner_model=model_wrap)
                    self.assertIn(words, str(ctx.exception))

    def test_a_tiny_settings_sigma_min_that_forges_karras_cannot_end_on_stops_the_generation(self):
        """Reachable within the sliders on eps/v: Settings → sigma min set very small. Forge's float32 Karras then ends
        on a 0 before the final 0 (ρ 1 and 1e-7: the ramp's subtraction cancels) or on a subnormal (ρ 7 and 1e-40),
        which Forge's samplers divide by; Flow Cosmos rho7 refuses it like the custom scheduler. Just above, it is
        Forge's Karras bit for bit."""
        model_wrap, _, hi = self._model("eps (SDXL)")
        for rho, sigma_min, forge_last in ((1.0, 1e-7, 0.0), (7.0, 1e-40, None)):
            with self.subTest(rho=rho, sigma_min=sigma_min):
                forge = self.karras(28, sigma_min, hi, rho=rho)
                self.assertLess(float(forge[-2]), sch._FLOAT32_TINY)
                if forge_last is not None:
                    self.assertEqual(float(forge[-2]), forge_last)
                st.set_active(_cosmos_settings(rho=rho))
                with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                    sch.flow_cosmos_rho7(n=28, sigma_min=sigma_min, sigma_max=hi, inner_model=model_wrap)
                self.assertIn("at step 27 (Forge's Karras on sigma ", str(ctx.exception))
                self.assertIn("Settings → Sampler Parameters → sigma min / sigma max; rho ", str(ctx.exception))
        for rho, sigma_min in ((1.0, 1e-5), (7.0, 1e-37)):
            with self.subTest(rho=rho, sigma_min=sigma_min):
                st.set_active(_cosmos_settings(rho=rho))
                ours = sch.flow_cosmos_rho7(n=28, sigma_min=sigma_min, sigma_max=hi, inner_model=model_wrap)
                self.assertTrue(torch.equal(ours, self.karras(28, sigma_min, hi, rho=rho)))


# ---------------------------------------------------------------------------
# Flow Cosmos Dynamic (D20)
# ---------------------------------------------------------------------------
def oracle_flow_cosmos_dynamic(n, rho, ve_sigma_max, ve_sigma_min):
    """Independent float64 form (numpy): the textbook Karras ramp with Karras Dynamic's per-step exponent,
    s_i = (σ̃max^(1/ρᵢ) + p(σ̃min^(1/ρᵢ) − σ̃max^(1/ρᵢ)))^ρᵢ, ρᵢ = ρ + 2cos(2πi/n), then t = 1/(1 + 1/s) — the same values
    as the scheduler's ratio form and s/(1 + s), written differently (so its rounding differs)."""
    i = np.arange(n, dtype=np.float64)
    r = rho + 2.0 * np.cos(2.0 * np.pi * i / max(n, 1))
    p = _p(n)
    s = (ve_sigma_max ** (1.0 / r) + p * (ve_sigma_min ** (1.0 / r) - ve_sigma_max ** (1.0 / r))) ** r
    return 1.0 / (1.0 + 1.0 / s)


FLOW_COSMOS_DYNAMIC_FIRST_FLAT_STEPS = 1_277_194


def _dynamic_settings(rho=st.FLOW_COSMOS_RHO_DEFAULT, sigma_max=st.FLOW_COSMOS_SIGMA_MAX_DEFAULT,
                      sigma_min=st.FLOW_COSMOS_SIGMA_MIN_DEFAULT):
    return _cosmos_settings(rho, sigma_max, sigma_min)


class FlowCosmosDynamicTests(unittest.TestCase):
    """D20: Karras Dynamic's per-step ρ on Flow Cosmos rho7's σ̃ range (flow), Karras Dynamic itself (eps/v); the same
    three accordion values as Flow Cosmos rho7."""

    RHOS = (4.0, 5.0, 7.0, 9.3, 15.0)          # from about 4 it falls (below, Karras Dynamic's rule stops it)
    VE_RANGES = FlowCosmosKnobTests.VE_RANGES
    STEPS = (1, 2, 3, 4, 10, 28, 151)
    ANIMA = _anima_range()

    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)
        for flag in ("_flow_cosmos_not_flow_logged", "_flow_cosmos_dynamic_not_flow_logged"):
            patcher = mock.patch.object(sch, flag, True)   # quiet, except where tested
            patcher.start()
            self.addCleanup(patcher.stop)

    def flow(self, n, settings=None, lo=None, hi=None):
        if settings is not None:
            st.set_active(settings)
        lo, hi = (self.ANIMA if lo is None else (lo, hi))
        return sch.flow_cosmos_dynamic(n, lo, hi, inner_model=FLOW_MODEL)

    def test_the_log_line_is_a_module_level_literal_parallel_to_flow_cosmos_rho7s(self):
        tree = ast.parse((ROOT / "sam3ext" / "extra_schedulers" / "schedulers.py").read_text(encoding="utf-8"))
        literal = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                       and [getattr(t, "id", None) for t in node.targets] == ["FLOW_COSMOS_DYNAMIC_NOT_FLOW_LOG"])
        self.assertIsInstance(literal, ast.Constant)
        self.assertEqual(literal.value, "[Extra Schedulers] Flow Cosmos Dynamic: not a flow model - using Karras Dynamic "
                                        "rho {rho:g} on the model's own sigma range")
        self.assertEqual(sch.FLOW_COSMOS_DYNAMIC_NOT_FLOW_LOG.replace("Flow Cosmos Dynamic", "Flow Cosmos rho7")
                         .replace("Karras Dynamic", "Karras"), sch.FLOW_COSMOS_NOT_FLOW_LOG)

    def test_a_grid_of_values_against_an_independent_formula(self):
        """float64 against ``oracle_flow_cosmos_dynamic`` within the ramp's rounding (the subtraction loses about
        ρᵢ·ε·(σ̃max/σ̃min)^(1/ρᵢ), largest at the smallest ρᵢ = ρ − 2); the list is the float32 rounding of those values
        with the final 0, within 1 float32 ulp of the oracle's; σ̃max/(1 + σ̃max) first and σ̃min/(1 + σ̃min) last exactly
        (Karras Dynamic's ramp gives σ̃max at p = 0 and σ̃min at p = 1 exactly); strictly falling."""
        eps = float(np.finfo(np.float64).eps)
        for rho in self.RHOS:
            for top, bottom in self.VE_RANGES:
                rtol = 64 * eps * (rho + 2) * (top / bottom) ** (1.0 / (rho - 2))
                for n in self.STEPS:
                    with self.subTest(rho=rho, sigma_max=top, sigma_min=bottom, steps=n):
                        times, bad = sch._flow_cosmos_dynamic_times(n, rho, bottom, top)
                        self.assertIsNone(bad)
                        times = np.array(times, dtype=np.float64)
                        expected = oracle_flow_cosmos_dynamic(n, rho, top, bottom)
                        np.testing.assert_allclose(times, expected, rtol=rtol, atol=0)
                        sigmas = self.flow(n, _dynamic_settings(rho, top, bottom))
                        self.assertEqual(sigmas.dtype, torch.float32)
                        self.assertEqual(sigmas.shape, (n + 1,))
                        self.assertEqual(float(sigmas[-1]), 0.0)
                        np.testing.assert_array_equal(sigmas[:-1].numpy(), times.astype(np.float32))
                        self.assertLessEqual(_float32_ulps(sigmas[:-1].numpy(), expected), 1)
                        self.assertEqual(float(sigmas[0]), float(np.float32(top / (1.0 + top))))
                        if n > 1:
                            self.assertEqual(float(sigmas[-2]), float(np.float32(bottom / (1.0 + bottom))))
                        self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))

    def test_the_ramp_is_karras_dynamics_own(self):
        """The σ̃ values are Karras Dynamic's (the same function, ``_karras_dynamic_values``): Karras Dynamic on the σ̃
        range is their float32 list bit for bit, and t is s/(1 + s) of each in float64."""
        for rho in (4.0, 7.0, 12.5):
            for top, bottom in ((80.0, 0.002), (240.0, 0.006), (1000.0, 0.0001)):
                for n in (1, 2, 5, 28, 64, 151):
                    with self.subTest(rho=rho, sigma_max=top, sigma_min=bottom, steps=n):
                        values, bad = sch._karras_dynamic_values(n, bottom, top, rho)
                        self.assertIsNone(bad)
                        self.assertTrue(torch.equal(sch._with_final_zero(values, "cpu"),
                                                    sch.karras_dynamic(n, bottom, top, rho=rho)))
                        times, _bad = sch._flow_cosmos_dynamic_times(n, rho, bottom, top)
                        self.assertEqual(times, [s / (1.0 + s) for s in values])
        # the defaults of the helper are Flow Cosmos rho7's (ρ 7 on 0.002 … 80)
        self.assertEqual(sch._flow_cosmos_dynamic_times(28), sch._flow_cosmos_dynamic_times(28, 7.0, 0.002, 80.0))

    def test_it_equals_flow_cosmos_rho7_only_where_the_maths_says_so(self):
        """ρᵢ = ρ only where cos(2πi/n) = 0, so with the same ρ and σ̃ range the two lists meet at the first step (σ̃max),
        the last one (σ̃min) and, when 4 divides n, at steps n/4 and 3n/4 (the same value in another operation order —
        equal there in float32 too). Elsewhere σ̃ is lower than Flow Cosmos rho7's where ρᵢ > ρ (first and last quarter)
        and higher where ρᵢ < ρ (middle half): the weighted power mean ((1 − p) + p·r^q)^(1/q) of 1 and r = σ̃min/σ̃max
        grows with q = 1/ρᵢ. Checked in float64 (σ̃) and on the float32 lists."""
        for rho in (4.0, 7.0, 15.0):
            for top, bottom in ((80.0, 0.002), (240.0, 0.006), (1.0, 0.0001), (1.5, 0.75)):
                st.set_active(_dynamic_settings(rho, top, bottom))
                for n in range(1, 161):
                    with self.subTest(rho=rho, sigma_max=top, sigma_min=bottom, steps=n):
                        meet = {0, n - 1} | ({n // 4, 3 * n // 4} if n % 4 == 0 else set())
                        dynamic = self.flow(n)[:-1].numpy()
                        rho7 = sch.flow_cosmos_rho7(n, *self.ANIMA, inner_model=FLOW_MODEL)[:-1].numpy()
                        self.assertEqual(set(np.nonzero(dynamic == rho7)[0].tolist()), meet)
                        s_dynamic, _bad = sch._karras_dynamic_values(n, bottom, top, rho)
                        r_top, r_bottom = top ** (1.0 / rho), bottom ** (1.0 / rho)
                        s_rho7 = [(r_top + p * (r_bottom - r_top)) ** rho for p in sch._positions(n)]
                        for i in sorted(set(range(n)) - meet):
                            lower = math.cos(2.0 * math.pi * i / n) > 0.0
                            self.assertEqual(s_dynamic[i] < s_rho7[i], lower, i)
                            self.assertNotEqual(s_dynamic[i], s_rho7[i], i)

    def test_strictly_falling_with_the_defaults_until_float32_runs_out_of_room(self):
        for n in range(1, 1001):
            sigmas = self.flow(n)
            self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()), n)
        self.assertEqual(self.flow(1).tolist(), [float(np.float32(80.0 / 81.0)), 0.0])
        self.assertEqual(self.flow(0).tolist(), [0.0])
        # the first flat step (module docstring; found by bisection, the 100 counts below it fall too): later than Flow
        # Cosmos rho7's 1,119,764 — near the start ρᵢ ≈ ρ + 2 makes the first steps longer
        sigmas = self.flow(FLOW_COSMOS_DYNAMIC_FIRST_FLAT_STEPS - 1)
        self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))
        with self.assertRaises(sch.ExtraSchedulerError) as ctx:
            self.flow(FLOW_COSMOS_DYNAMIC_FIRST_FLAT_STEPS)
        self.assertNotIsInstance(ctx.exception, (sch.CustomSchedulerError, sch.KarrasDynamicError))
        self.assertIn("Extra Schedulers: Flow Cosmos Dynamic gives sigma 0.987626 at steps 476 and 477 (rho 7, sigma~ "
                      "0.002 … 80 — " + sch.FLOW_COSMOS_VALUES_SOURCE, str(ctx.exception))

    def test_from_rho_4_it_falls_on_the_slider_ranges(self):
        """Karras Dynamic's finding holds on the σ̃ ranges: from ρ 4 no step rises or stays (n 1 … 150)."""
        for rho in (4.0, 4.5, 7.0, 15.0):
            for top, bottom in ((80.0, 0.002), (1000.0, 0.0001), (1.0, 0.0001), (240.0, 0.006)):
                with self.subTest(rho=rho, sigma_max=top, sigma_min=bottom):
                    st.set_active(_dynamic_settings(rho, top, bottom))
                    for n in range(1, 151):
                        sigmas = self.flow(n)   # raises if any step rises or stays
                        self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))

    def test_a_rising_schedule_stops_the_generation_naming_the_accordion(self):
        """Below ρ ≈ 4 a small ρᵢ lifts the middle steps (as in Karras Dynamic): refused, on both branches, with
        Karras Dynamic's error pointing at the accordion's ρ (not Settings → rho, which does not reach it)."""
        st.set_active(_dynamic_settings(rho=3.0))
        oracle = oracle_flow_cosmos_dynamic(20, 3.0, 80.0, 0.002)
        self.assertTrue((np.diff(oracle) > 0).any(), "the formula itself rises for rho 3")
        for model, lo, hi, words in ((FLOW_MODEL, *self.ANIMA, "sigma rises from 0.9704 to 0.972 at step 7"),
                                     (EPS_MODEL, *RANGES["eps"], "sigma rises from 6.408 to 6.544 at step 7")):
            with self.subTest(kind=model.predictor.prediction_type):
                with self.assertRaises(sch.KarrasDynamicError) as ctx:
                    sch.flow_cosmos_dynamic(20, lo, hi, inner_model=model)
                message = str(ctx.exception)
                self.assertTrue(message.startswith("Extra Schedulers: Flow Cosmos Dynamic with rho 3 does not fall over "
                                                   "20 steps — "), message)
                self.assertIn(words, message)
                self.assertTrue(message.endswith(f"Raise rho to about 4 or more: {sch.FLOW_COSMOS_VALUES_SOURCE}."))
                self.assertNotIn("Settings → Sampler Parameters → rho", message)
        # every ρ from 2.1 to 3.8 rises for some n ≤ 300 on the default σ̃ range; 3.9 does not (the docstring's numbers)
        for rho, rises in ((3.8, True), (3.9, False)):
            with self.subTest(rho=rho):
                st.set_active(_dynamic_settings(rho=rho))
                found = False
                for n in range(1, 301):
                    try:
                        self.flow(n)
                    except sch.KarrasDynamicError:
                        found = True
                        break
                self.assertEqual(found, rises)

    def test_rho_must_be_above_2_on_both_branches(self):
        for rho in (2.0, 1.5, 1.0, 0.0, -3.0, math.nan):
            for model in (FLOW_MODEL, EPS_MODEL, V_MODEL):
                with self.subTest(rho=rho, kind=model.predictor.prediction_type):
                    st.set_active(_dynamic_settings(rho=rho))
                    with self.assertRaises(sch.KarrasDynamicError) as ctx:
                        sch.flow_cosmos_dynamic(10, 0.03, 14.6, inner_model=model)
                    self.assertEqual(str(ctx.exception),
                                     f"Extra Schedulers: Flow Cosmos Dynamic needs rho above 2 (its per-step exponent goes "
                                     f"down to rho - 2), got {rho:g}. Set rho to about 4 or more: "
                                     f"{sch.FLOW_COSMOS_VALUES_SOURCE}.")
        # the slider's lower part (1 … 2) gets here from the UI; Flow Cosmos rho7 takes those values
        st.set_active(_dynamic_settings(rho=st.FLOW_COSMOS_RHO_MIN))
        sch.flow_cosmos_rho7(10, *self.ANIMA, inner_model=FLOW_MODEL)
        with self.assertRaises(sch.KarrasDynamicError):
            self.flow(10)

    def test_eps_and_v_models_get_karras_dynamic_bit_for_bit_with_the_accordion_rho(self):
        """Not a flow model: exactly Karras Dynamic (``karras_dynamic``) on the range Forge passes with the accordion's
        ρ; σ̃ max / σ̃ min are not used there (even values a flow model would refuse)."""
        for model in (EPS_MODEL, V_MODEL):
            for rho in (4.0, 5.5, 7.0, 12.0, 15.0):
                for lo, hi in (RANGES["eps"], (0.1, 20.0), (0.0291675, 4500.0), RANGES["flow"]):
                    for ve in ((80.0, 0.002), (1.0, 1.0)):
                        st.set_active(_dynamic_settings(rho, *ve))
                        for n in (0, 1, 2, 3, 7, 28, 64, 150):
                            with self.subTest(kind=model.predictor.prediction_type, rho=rho, range=(lo, hi), ve=ve,
                                              steps=n):
                                ours = sch.flow_cosmos_dynamic(n, lo, hi, inner_model=model)
                                self.assertTrue(torch.equal(ours, sch.karras_dynamic(n, lo, hi, rho=rho)))
                                self.assertEqual(ours.dtype, torch.float32)
        # Karras Dynamic's other errors keep their wording, with this scheduler's name
        st.reset()
        with self.assertRaises(sch.KarrasDynamicError) as ctx:
            sch.flow_cosmos_dynamic(10, 14.6, 0.03, inner_model=EPS_MODEL)
        self.assertEqual(str(ctx.exception), "Extra Schedulers: Flow Cosmos Dynamic cannot fall from sigma_max 0.03 to a "
                                             "larger sigma_min 14.6 (Settings → Sampler Parameters → sigma min / sigma max)")

    def test_a_tiny_settings_sigma_min_on_eps_v_is_karras_dynamic_too(self):
        """Karras Dynamic ends exactly on Forge's sigma min; set very small in Settings it becomes 0 (1e-46) or a
        subnormal (1e-40, 1e-38) in float32. On an eps/v model Flow Cosmos Dynamic is exactly Karras Dynamic (D20), so
        it returns the same list there as well — no rule of Flow Cosmos rho7's is added on this branch."""
        hi = RANGES["eps"][1]
        for rho in (4.0, 7.0, 15.0):
            st.set_active(_dynamic_settings(rho))
            for sigma_min in (1e-46, 1e-40, 1e-38, 1e-37, 1e-30, 1e-5):
                with self.subTest(rho=rho, sigma_min=sigma_min):
                    expected = sch.karras_dynamic(28, sigma_min, hi, rho=rho)
                    if sigma_min <= 1e-38:
                        self.assertLess(float(expected[-2]), sch._FLOAT32_TINY)
                    for model in (EPS_MODEL, V_MODEL):
                        self.assertTrue(torch.equal(sch.flow_cosmos_dynamic(28, sigma_min, hi, inner_model=model),
                                                    expected))

    def test_a_flow_model_keeps_flow_cosmos_rho7s_float32_rule(self):
        """Past the sliders (σ̃ min 0.0001 … 1): a σ̃ min whose flow time is 0 or a subnormal in float32 stops a flow
        model with Flow Cosmos rho7's rule — the sigma Forge's samplers would divide by."""
        for ve_min, words in ((1e-46, "gives sigma 0 at step 27"), (1e-40, "gives sigma 9.99995e-41 at step 27")):
            with self.subTest(ve_min=ve_min):
                st.set_active(_dynamic_settings(7.0, 80.0, ve_min))
                with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                    sch.flow_cosmos_dynamic(28, *self.ANIMA, inner_model=FLOW_MODEL)
                self.assertNotIsInstance(ctx.exception, (sch.CustomSchedulerError, sch.KarrasDynamicError))
                self.assertIn(f"Extra Schedulers: Flow Cosmos Dynamic {words} (sigma~ {ve_min:g} … 80; rho 7 — "
                              + sch.FLOW_COSMOS_VALUES_SOURCE, str(ctx.exception))
        st.set_active(_dynamic_settings(7.0, 80.0, 1e-37))
        self.assertGreaterEqual(float(sch.flow_cosmos_dynamic(28, *self.ANIMA, inner_model=FLOW_MODEL)[-2]),
                                sch._FLOAT32_TINY)

    def test_the_values_are_read_when_the_scheduler_runs_and_the_models_range_is_not_used(self):
        first = self.flow(28)
        moved = self.flow(28, _dynamic_settings(5.0, 240.0, 0.006))
        self.assertFalse(torch.equal(first, moved))
        times, _bad = sch._flow_cosmos_dynamic_times(28, 5.0, 0.006, 240.0)
        self.assertTrue(torch.equal(moved, sch._with_final_zero(times, "cpu")))
        lists = [sch.flow_cosmos_dynamic(20, lo, hi, inner_model=FLOW_MODEL)
                 for lo, hi in (self.ANIMA, RANGES["flow"], (0.05, 0.9), RANGES["eps"])]
        for sigmas in lists[1:]:
            self.assertTrue(torch.equal(sigmas, lists[0]))
        st.reset()
        self.assertTrue(torch.equal(self.flow(28), first))

    def test_a_sigma_range_that_is_not_0_below_min_below_max_stops_a_flow_model(self):
        for sigma_min, sigma_max in ((1.0, 1.0), (1.0, 0.5), (0.0, 80.0), (math.nan, 80.0), (0.002, math.inf)):
            with self.subTest(sigma_min=sigma_min, sigma_max=sigma_max):
                with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                    self.flow(10, _dynamic_settings(7.0, sigma_max, sigma_min))
                self.assertNotIsInstance(ctx.exception, (sch.CustomSchedulerError, sch.KarrasDynamicError))
                self.assertIn("Flow Cosmos Dynamic needs 0 < sigma~ min < sigma~ max, both finite", str(ctx.exception))
                self.assertIn(sch.FLOW_COSMOS_VALUES_SOURCE, str(ctx.exception))
                # eps/v does not use them
                sch.flow_cosmos_dynamic(10, *RANGES["eps"], inner_model=EPS_MODEL)

    def test_a_flat_step_from_a_narrow_range_names_these_values(self):
        """The flat-step rule runs before the falling rule, so a float32 flat step from a narrow σ̃ range gets its own
        message naming the accordion values; σ̃ 0.9999 … 1 first has one at 841 steps, as for Flow Cosmos rho7."""
        for rho in (st.FLOW_COSMOS_RHO_MAX, 7.0, 4.0):
            with self.subTest(rho=rho):
                st.set_active(_dynamic_settings(rho, 1.0, 0.9999))
                sigmas = self.flow(840)
                self.assertTrue(bool((sigmas[1:] < sigmas[:-1]).all()))
                with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                    self.flow(841)
                message = str(ctx.exception)
                self.assertNotIsInstance(ctx.exception, sch.KarrasDynamicError)
                self.assertIn("Extra Schedulers: Flow Cosmos Dynamic gives sigma 0.499989 at steps ", message)
                self.assertIn(f"rho {rho:g}, sigma~ 0.9999 … 1 — " + sch.FLOW_COSMOS_VALUES_SOURCE, message)
                self.assertIn(sch.ZERO_LENGTH_STEP, message)

    def test_not_a_flow_model_is_logged_once_per_process_with_the_first_runs_rho(self):
        stderr = io.StringIO()
        with mock.patch.object(sch, "_flow_cosmos_dynamic_not_flow_logged", False), \
                mock.patch.object(sch, "_flow_cosmos_not_flow_logged", False), contextlib.redirect_stderr(stderr):
            self.flow(10)
            self.assertEqual(stderr.getvalue(), "", "a flow model logs nothing")
            st.set_active(_dynamic_settings(rho=5.5))
            sch.flow_cosmos_dynamic(10, 0.03, 14.6, inner_model=EPS_MODEL)
            st.reset()
            sch.flow_cosmos_dynamic(20, 0.03, 14.6, inner_model=V_MODEL)
            self.assertEqual(stderr.getvalue(), "[Extra Schedulers] Flow Cosmos Dynamic: not a flow model - using Karras "
                                                "Dynamic rho 5.5 on the model's own sigma range\n")
            # Flow Cosmos rho7 keeps its own line (its own flag)
            with fixtures.installed_k_sampling(_k_sampling_with(karras_formula)):
                sch.flow_cosmos_rho7(10, 0.03, 14.6, inner_model=EPS_MODEL)
            self.assertEqual(stderr.getvalue().count("not a flow model"), 2)

    def test_without_a_model_parameterisation_it_stops(self):
        for inner_model in (None, types.SimpleNamespace(), types.SimpleNamespace(predictor=types.SimpleNamespace())):
            with self.subTest(inner_model=inner_model):
                with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                    sch.flow_cosmos_dynamic(10, 0.03, 1.0, inner_model=inner_model)
                self.assertIn("Flow Cosmos Dynamic needs Forge's model", str(ctx.exception))

    def test_needs_positive_sigmas(self):
        for model in (FLOW_MODEL, EPS_MODEL):
            for lo, hi in ((0.0, 1.0), (-0.1, 14.6), (0.003, 0.0)):
                with self.subTest(kind=model.predictor.prediction_type, lo=lo, hi=hi):
                    with self.assertRaises(sch.ExtraSchedulerError) as ctx:
                        sch.flow_cosmos_dynamic(10, lo, hi, inner_model=model)
                    self.assertIn("Flow Cosmos Dynamic needs sigma_min and sigma_max above 0", str(ctx.exception))

    def test_device_argument(self):
        for model in (FLOW_MODEL, EPS_MODEL):
            sigmas = sch.flow_cosmos_dynamic(4, 0.03, 1.0, inner_model=model, device=torch.device("cpu"))
            self.assertEqual(sigmas.device.type, "cpu")


@unittest.skipUnless(support.forge_root(), "Forge checkout not found")
class FlowCosmosDynamicForgeTests(unittest.TestCase):
    """With Forge's ``ForgeScheduleLinker`` around Forge's real predictors (what ``get_sigmas`` passes as
    ``inner_model``): flow predictors get the flow list whatever their shift, eps/v ones Karras Dynamic."""

    # Flow Cosmos rho7's setUpClass run for this class (Forge's linker and predictors), and its _model
    setUpClass = classmethod(FlowCosmosRho7ForgeTests.setUpClass.__func__)
    _model = FlowCosmosRho7ForgeTests._model

    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)
        patcher = mock.patch.object(sch, "_flow_cosmos_dynamic_not_flow_logged", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_forges_predictors(self):
        for rho, top, bottom in ((7.0, 80.0, 0.002), (5.0, 240.0, 0.006), (15.0, 1.0, 0.5)):
            st.set_active(_dynamic_settings(rho, top, bottom))
            for n in (1, 2, 3, 28, 50, 150):
                times, _bad = sch._flow_cosmos_dynamic_times(n, rho, bottom, top)
                expected = sch._with_final_zero(times, "cpu")
                for name in ("flow (Anima, shift 3)", "flow (shift 1)"):
                    model_wrap, lo, hi = self._model(name)
                    with self.subTest(model=name, rho=rho, steps=n):
                        self.assertTrue(torch.equal(sch.flow_cosmos_dynamic(n, lo, hi, inner_model=model_wrap), expected))
                for name in ("eps (SDXL)", "v (SDXL betas)"):
                    model_wrap, lo, hi = self._model(name)
                    with self.subTest(model=name, rho=rho, steps=n):
                        self.assertTrue(torch.equal(sch.flow_cosmos_dynamic(n, lo, hi, inner_model=model_wrap),
                                                    sch.karras_dynamic(n, lo, hi, rho=rho)))


if __name__ == "__main__":
    unittest.main()
