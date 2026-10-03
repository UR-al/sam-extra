"""Extra Schedulers — the scheduler functions against independent implementations of their formulas.

The oracles below are written separately with numpy (float64) from the formulas, not by calling
the code under test. Where Forge has a matching function (k-diffusion ``get_sigmas_exponential`` and
``get_sigmas_karras``, ``sd_schedulers._loglinear_interp``) its real code is cut out of the Forge
checkout and run too (skipped when no Forge checkout is found).
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

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


if __name__ == "__main__":
    unittest.main()
