"""Extra Schedulers — the custom scheduler's expression parser (security first, then correctness).

The expression is restored from infotext, i.e. from any PNG someone hands the user, so the parser
must never execute it: every construct outside the whitelist is rejected before evaluation, and
nothing that passes can run Python code, build big integers or loop.
"""
from __future__ import annotations

import builtins
import math
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext.extra_schedulers import expression as ex  # noqa: E402

VARS = {"m": 0.0292, "M": 14.6146, "n": 20.0, "s": 3.0, "x": 3 / 19}


def _value(text, **overrides):
    env = dict(VARS)
    env.update(overrides)
    return ex.compile_expression(text).evaluate(**env)


class RejectsCodeTests(unittest.TestCase):
    """Nothing that could run code, reach objects or loop gets through compile_expression."""

    REJECTED = {
        "import call": "__import__('os').system('echo pwned')",
        "import name": "__import__",
        "builtins": "__builtins__",
        "dunder name": "__class__",
        "attribute on a name": "os.system",
        "attribute on a literal": "(1).real",
        "class walk": "().__class__.__bases__[0].__subclasses__()",
        "attribute of an allowed name": "m.real",
        "attribute of a call": "sqrt(4).hex",
        "subscript": "x[0]",
        "list": "[1, 2]",
        "tuple": "(1, 2)",
        "dict": "{1: 2}",
        "set": "{1}",
        "lambda": "(lambda: 1)()",
        "lambda value": "lambda: 1",
        "list comprehension": "[i for i in range(3)]",
        "set comprehension": "{i for i in range(3)}",
        "dict comprehension": "{i: i for i in range(3)}",
        "generator": "sum(i for i in range(10**9))",
        "eval": "eval('1')",
        "exec": "exec('x=1')",
        "open": "open('C:/x.txt', 'w')",
        "getattr": "getattr(1, 'real')",
        "compile": "compile('1', 'x', 'eval')",
        "unknown function": "sum([1, 2])",
        "range": "range(10)",
        "string": "'abc'",
        "bytes": "b'abc'",
        "f-string": "f'{m}'",
        "True": "True",
        "None": "None",
        "ellipsis": "...",
        "complex": "1j",
        "ternary": "1 if x else 2",
        "comparison": "x < 1",
        "chained comparison": "0 < x < 1",
        "and": "x and 1",
        "or": "x or 1",
        "not": "not x",
        "invert": "~1",
        "floor division": "7 // 2",
        "modulo": "7 % 2",
        "shift": "1 << 70",
        "bit and": "1 & 3",
        "matmul": "m @ M",
        "walrus": "(y := 1)",
        "keyword argument": "log(x=1)",
        "starred argument": "max(*[1, 2])",
        "function used as value": "max",
        "method call on function": "max.__call__(1, 2)",
        "call of a call": "sqrt(4)(2)",
        "await": "await x",
        "yield": "(yield 1)",
        "statement": "x = 1",
        "two statements": "1; 2",
        "unknown variable": "y + 1",
        "lookalike variable": "N + 1",
    }

    def test_every_construct_outside_the_whitelist_is_rejected(self):
        for name, text in self.REJECTED.items():
            with self.subTest(name=name, text=text):
                with self.assertRaises(ex.ExpressionError):
                    ex.compile_expression(text)

    def test_nothing_is_executed_while_rejecting(self):
        # If any path evaluated the text, __import__ or open would be called.
        calls = []
        real_import = builtins.__import__

        def spy_import(name, *args, **kwargs):
            calls.append(name)
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", side_effect=spy_import), \
                mock.patch.object(builtins, "open", side_effect=AssertionError("open called")):
            for text in ("__import__('os')", "open('x')", "().__class__", "exec('1')"):
                with self.assertRaises(ex.ExpressionError):
                    ex.compile_expression(text)
        self.assertNotIn("os", calls)

    def test_compiled_form_holds_no_code_objects(self):
        compiled = ex.compile_expression("m + (M - m) * (1 - x) ** phi + max(sqrt(x), 0.5)")

        def walk(node):
            yield node
            if isinstance(node, tuple):
                for child in node:
                    yield from walk(child)

        kinds = {type(item) for item in walk(compiled._tree)}
        self.assertLessEqual(kinds, {tuple, str, float})

    def test_non_text_is_rejected(self):
        for value in (None, 1, 1.5, b"m", ["m"], {"m": 1}):
            with self.subTest(value=value):
                with self.assertRaises(ex.ExpressionError):
                    ex.compile_expression(value)

    def test_empty_and_blank_are_rejected(self):
        for text in ("", "   ", "\n\t"):
            with self.assertRaises(ex.ExpressionError):
                ex.compile_expression(text)

    def test_null_bytes_are_rejected(self):
        with self.assertRaises(ex.ExpressionError):
            ex.compile_expression("m\x00+1")


class BoundsTests(unittest.TestCase):
    """Huge exponents and time bombs fail fast instead of hanging or exhausting memory."""

    def _fails_fast(self, text, *, at="compile"):
        start = time.perf_counter()
        with self.assertRaises(ex.ExpressionError):
            if at == "compile":
                ex.compile_expression(text)
            else:
                _value(text)
        self.assertLess(time.perf_counter() - start, 0.5, text[:60])

    def test_huge_exponents_are_rejected(self):
        for text in ("9**9**9", "9**9**9**9**9", "2 ** 65", "x ** 100", "(10**10)**(10**10)", "2 ** -65",
                     "M ** (M * 10)", "e ** 1000"):
            with self.subTest(text=text):
                self._fails_fast(text, at="evaluate")

    def test_exponent_bound_is_inclusive(self):
        self.assertEqual(_value("2 ** 64"), 2.0 ** 64)
        self.assertEqual(_value("2 ** -64"), 2.0 ** -64)

    def test_overflow_and_non_finite_values_fail(self):
        for text in ("exp(1000)", "exp(exp(exp(10)))", "1e308 * 10", "-1e308 * 10", "(2 ** 64) ** 64",
                     "cosh(1000)", "1e308 + 1e308"):
            with self.subTest(text=text):
                self._fails_fast(text, at="evaluate")

    def test_big_literals_fail_at_compile_time(self):
        self._fails_fast("1" * 400)            # an int too large for a float
        self._fails_fast("1e999")              # parses as inf
        self._fails_fast("-1e999")

    def test_length_limit_is_checked_before_parsing(self):
        long_text = "+".join(["1"] * 300)      # 599 characters of a valid expression
        with mock.patch.object(ex.ast, "parse", side_effect=AssertionError("parsed")):
            self._fails_fast(long_text)
        longest = "0." + "1" * (ex.MAX_LENGTH - 2)   # exactly MAX_LENGTH characters, one node
        self.assertEqual(len(longest), ex.MAX_LENGTH)
        self.assertAlmostEqual(_value(longest), 1 / 9)
        self._fails_fast(longest + "1")

    def test_node_limit(self):
        text = "+".join(["x"] * 150)            # 299 characters, 299 nodes
        self._fails_fast(text)

    def test_deep_nesting_fails_cleanly(self):
        for text in ("-" * 400 + "1", "(" * 240 + "1" + ")" * 240, "sqrt(" * 60 + "1" + ")" * 60,
                     "+" * 450 + "1"):
            with self.subTest(text=text[:20]):
                self._fails_fast(text)

    def test_many_call_arguments(self):
        self._fails_fast("max(" + ",".join(["1"] * 9) + ")")
        self.assertEqual(_value("max(" + ",".join(["1"] * 8) + ")"), 1.0)

    def test_repeated_evaluation_cost_is_bounded(self):
        compiled = ex.compile_expression("+".join(["sqrt(x+1)"] * 30))
        start = time.perf_counter()
        for step in range(2000):
            compiled.evaluate(**ex.step_variables(step, 2000, 0.03, 14.6))
        self.assertLess(time.perf_counter() - start, 2.0)


class MathErrorTests(unittest.TestCase):
    def test_domain_errors_become_expression_errors(self):
        for text in ("sqrt(-1)", "log(0)", "log(-1)", "(-8) ** (1/3)", "0 ** -1", "acos(2)", "log(8, 1)",
                     "1 / 0", "m / (x - x)", "log(x - x)"):
            with self.subTest(text=text):
                with self.assertRaises(ex.ExpressionError):
                    _value(text)

    def test_missing_or_non_finite_variables(self):
        compiled = ex.compile_expression("m + x")
        with self.assertRaises(ex.ExpressionError):
            compiled.evaluate(m=1.0, M=2.0, n=3.0, s=0.0)
        with self.assertRaises(ex.ExpressionError):
            compiled.evaluate(m=float("nan"), M=2.0, n=3.0, s=0.0, x=0.0)

    def test_error_type_is_a_value_error(self):
        self.assertTrue(issubclass(ex.ExpressionError, ValueError))


class CorrectnessTests(unittest.TestCase):
    def test_precedence_follows_python(self):
        self.assertEqual(_value("2 + 3 * 4 ** 2"), 50.0)
        self.assertEqual(_value("-2 ** 2"), -4.0)
        self.assertEqual(_value("(-2) ** 2"), 4.0)
        self.assertEqual(_value("2 ** -1"), 0.5)
        self.assertEqual(_value("2 ** 3 ** 2"), 512.0)
        self.assertEqual(_value("8 / 2 / 2"), 2.0)
        self.assertEqual(_value("1 - 2 - 3"), -4.0)
        self.assertEqual(_value("+3"), 3.0)
        self.assertEqual(_value("--3"), 3.0)

    def test_variables_and_constants(self):
        self.assertEqual(_value("m"), VARS["m"])
        self.assertEqual(_value("M"), VARS["M"])
        self.assertEqual(_value("n"), 20.0)
        self.assertEqual(_value("s"), 3.0)
        self.assertAlmostEqual(_value("x"), 3 / 19)
        self.assertAlmostEqual(_value("phi"), (1 + math.sqrt(5)) / 2)
        self.assertEqual(_value("phi"), ex.GOLDEN_RATIO)
        self.assertEqual(_value("pi"), math.pi)
        self.assertEqual(_value("e"), math.e)

    def test_functions(self):
        cases = {
            "abs(-2.5)": 2.5, "sqrt(16)": 4.0, "exp(0)": 1.0, "log(e)": 1.0, "log(8, 2)": 3.0, "log2(8)": 3.0,
            "log10(1000)": 3.0, "sin(0)": 0.0, "cos(0)": 1.0, "tan(0)": 0.0, "asin(1)": math.pi / 2,
            "acos(1)": 0.0, "atan(1)": math.pi / 4, "atan2(1, 1)": math.pi / 4, "sinh(0)": 0.0, "cosh(0)": 1.0,
            "tanh(0)": 0.0, "floor(2.7)": 2.0, "ceil(2.1)": 3.0, "min(3, 1, 2)": 1.0, "max(3, 1, 2)": 3.0,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertAlmostEqual(_value(text), expected, places=12)

    def test_typical_scheduler_expressions(self):
        m, M = VARS["m"], VARS["M"]
        x = VARS["x"]
        self.assertAlmostEqual(_value("M * (m / M) ** x"), M * (m / M) ** x, places=12)
        self.assertAlmostEqual(_value("m + (M - m) * (1 - x) ** 3"), m + (M - m) * (1 - x) ** 3, places=12)
        self.assertAlmostEqual(
            _value("m + (M - m) * (1 - x) ** (phi ** 2)"), m + (M - m) * (1 - x) ** (ex.GOLDEN_RATIO ** 2), places=12,
        )
        self.assertAlmostEqual(
            _value("m + 0.5 * (M - m) * (1 - cos(pi * (1 - sqrt(x))))"),
            m + 0.5 * (M - m) * (1 - math.cos(math.pi * (1 - math.sqrt(x)))),
            places=12,
        )

    def test_number_literals(self):
        self.assertEqual(_value("1_000"), 1000.0)
        self.assertEqual(_value("0x10"), 16.0)
        self.assertEqual(_value(".5"), 0.5)
        self.assertEqual(_value("1e-3"), 0.001)

    def test_whitespace_and_newlines(self):
        self.assertEqual(ex.compile_expression("  M *\n (m / M) ** x \t").source, "M * (m / M) ** x")
        self.assertEqual(_value("1 +\n 2"), 3.0)

    def test_results_are_floats(self):
        for text in ("1", "floor(2.5)", "max(1, 2)", "2 ** 2"):
            self.assertIs(type(_value(text)), float)

    def test_step_variables(self):
        self.assertEqual(ex.step_variables(0, 1, 0.1, 2.0)["x"], 0.0)
        self.assertEqual(ex.step_variables(4, 5, 0.1, 2.0)["x"], 1.0)
        variables = ex.step_variables(2, 5, 0.1, 2.0)
        self.assertEqual(variables, {"m": 0.1, "M": 2.0, "n": 5.0, "s": 2.0, "x": 0.5})

    def test_compiled_expression_is_reusable_and_immutable(self):
        compiled = ex.compile_expression("M * (m / M) ** x")
        first = [compiled.evaluate(**ex.step_variables(i, 5, 0.03, 14.6)) for i in range(5)]
        second = [compiled.evaluate(**ex.step_variables(i, 5, 0.03, 14.6)) for i in range(5)]
        self.assertEqual(first, second)
        with self.assertRaises(Exception):
            compiled.source = "1"


if __name__ == "__main__":
    unittest.main()
