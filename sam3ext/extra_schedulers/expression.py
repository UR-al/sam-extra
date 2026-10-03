"""Safe arithmetic expressions for the ``custom`` scheduler — no ``eval``, no ``exec``.

The expression comes back from infotext (PNG parameters, "Send to", the API), so it is
untrusted text. ``compile_expression`` parses it with ``ast.parse(mode="eval")`` and converts the
tree into a small private form, rejecting every node that is not on the whitelist below.
Nothing in the text is ever compiled or executed as Python; ``Expression.evaluate`` walks the
checked form with plain Python floats.

Allowed
    * numbers: ``int``/``float`` literals (``bool``, ``complex``, ``str``, ``bytes``, ``None`` and
      ``...`` are rejected); every literal is converted to a finite float.
    * names: the variables ``m`` (sigma_min), ``M`` (sigma_max), ``n`` (number of steps), ``s``
      (this step, from 0) and ``x`` (``s / (n - 1)``, 0 when ``n == 1``), and the constants
      ``phi`` (golden ratio), ``pi`` and ``e``.
    * operators: binary ``+ - * / **`` and unary ``-`` / ``+``.
    * calls of the functions in ``FUNCTIONS`` by bare name, positional arguments only, with
      the argument count checked.

Everything else is rejected before anything is evaluated: attribute access (``().__class__``),
subscripts, names outside the whitelist (``__import__``, ``open``), lambdas, comprehensions and
generators, comparisons, boolean operators, conditionals, f-strings, walrus, starred and
keyword arguments, and calls of anything that is not a whitelisted bare name.

Bounds
    * at most ``MAX_LENGTH`` characters (checked before parsing), ``MAX_NODES`` syntax nodes and
      nesting depth ``MAX_DEPTH``;
    * ``**`` exponents must stay within ``±MAX_EXPONENT``. The check uses the evaluated
      exponent, so ``9**9**9`` fails on the inner ``9**9``;
    * every intermediate value must be a finite float — overflow, division by zero and domain
      errors (``sqrt(-1)``, ``log(0)``, a negative base with a fractional exponent) raise
      ``ExpressionError``.

All arithmetic is float arithmetic with a bounded number of operations per step, so there is no
big-integer or repetition "time bomb": an evaluation costs at most ``MAX_NODES`` float operations.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Callable

MAX_LENGTH = 500
MAX_NODES = 200
MAX_DEPTH = 32
MAX_EXPONENT = 64.0
MAX_CALL_ARGS = 8

GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0

VARIABLES = ("m", "M", "n", "s", "x")
CONSTANTS = {"phi": GOLDEN_RATIO, "pi": math.pi, "e": math.e}


class ExpressionError(ValueError):
    """The text is not an allowed expression, or evaluating it gave no finite number."""


def _log(value, base=None):
    return math.log(value) if base is None else math.log(value, base)


def _min(*values):
    return min(values)


def _max(*values):
    return max(values)


# name -> (function, minimum argument count, maximum argument count)
FUNCTIONS: dict[str, tuple[Callable[..., float], int, int]] = {
    "abs": (math.fabs, 1, 1),
    "sqrt": (math.sqrt, 1, 1),
    "exp": (math.exp, 1, 1),
    "log": (_log, 1, 2),
    "log2": (math.log2, 1, 1),
    "log10": (math.log10, 1, 1),
    "sin": (math.sin, 1, 1),
    "cos": (math.cos, 1, 1),
    "tan": (math.tan, 1, 1),
    "asin": (math.asin, 1, 1),
    "acos": (math.acos, 1, 1),
    "atan": (math.atan, 1, 1),
    "atan2": (math.atan2, 2, 2),
    "sinh": (math.sinh, 1, 1),
    "cosh": (math.cosh, 1, 1),
    "tanh": (math.tanh, 1, 1),
    "floor": (math.floor, 1, 1),
    "ceil": (math.ceil, 1, 1),
    "min": (_min, 2, MAX_CALL_ARGS),
    "max": (_max, 2, MAX_CALL_ARGS),
}

_BINARY_OPERATORS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Pow: "**",
}

# Private checked form: ("num", value) | ("var", name) | ("neg", node)
# | ("bin", operator, left, right) | ("call", name, (args...)).
_NUM, _VAR, _NEG, _BIN, _CALL = "num", "var", "neg", "bin", "call"


def normalize_source(text) -> str:
    """Whitespace-normalised source: tabs and newlines become spaces (so a pasted multi-line
    expression still parses), runs of spaces collapse and the ends are stripped."""
    if not isinstance(text, str):
        raise ExpressionError("the expression must be text")
    return " ".join(text.split())


def _describe_node(node) -> str:
    return type(node).__name__


class _Converter:
    def __init__(self):
        self.nodes = 0

    def convert(self, node, depth: int):
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise ExpressionError(f"the expression is too long (more than {MAX_NODES} parts)")
        if depth > MAX_DEPTH:
            raise ExpressionError(f"the expression is nested too deeply (more than {MAX_DEPTH} levels)")

        if isinstance(node, ast.Constant):
            value = node.value
            # bool is a subclass of int; exact type checks keep True/False out.
            if type(value) not in (int, float):
                raise ExpressionError(f"only numbers are allowed as literals, not {type(value).__name__}")
            try:
                number = float(value)
            except OverflowError:
                raise ExpressionError("a number in the expression is too large") from None
            if not math.isfinite(number):
                raise ExpressionError("numbers in the expression must be finite")
            return (_NUM, number)

        if isinstance(node, ast.Name):
            if node.id in VARIABLES:
                return (_VAR, node.id)
            if node.id in CONSTANTS:
                return (_NUM, CONSTANTS[node.id])
            raise ExpressionError(
                f"unknown name {node.id!r}; allowed names are "
                f"{', '.join(VARIABLES + tuple(CONSTANTS))} and the functions {', '.join(FUNCTIONS)}"
            )

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return (_NEG, self.convert(node.operand, depth + 1))
            if isinstance(node.op, ast.UAdd):
                return self.convert(node.operand, depth + 1)
            raise ExpressionError(f"the operator {_describe_node(node.op)} is not allowed")

        if isinstance(node, ast.BinOp):
            operator = _BINARY_OPERATORS.get(type(node.op))
            if operator is None:
                raise ExpressionError(
                    f"the operator {_describe_node(node.op)} is not allowed (use + - * / **)"
                )
            return (_BIN, operator, self.convert(node.left, depth + 1), self.convert(node.right, depth + 1))

        if isinstance(node, ast.Call):
            func = node.func
            if not isinstance(func, ast.Name) or func.id not in FUNCTIONS:
                raise ExpressionError(
                    "only these functions can be called by name: " + ", ".join(FUNCTIONS)
                )
            if node.keywords:
                raise ExpressionError(f"{func.id}() takes no keyword arguments")
            if any(isinstance(arg, ast.Starred) for arg in node.args):
                raise ExpressionError(f"{func.id}() does not accept *arguments")
            _fn, low, high = FUNCTIONS[func.id]
            if not low <= len(node.args) <= high:
                expected = str(low) if low == high else f"{low} to {high}"
                raise ExpressionError(f"{func.id}() takes {expected} arguments, got {len(node.args)}")
            args = tuple(self.convert(arg, depth + 1) for arg in node.args)
            return (_CALL, func.id, args)

        raise ExpressionError(f"{_describe_node(node)} is not allowed in a scheduler expression")


def _finite(value: float, what: str) -> float:
    if not math.isfinite(value):
        raise ExpressionError(f"{what} gave a value that is not finite")
    return value


def _evaluate(node, env: dict) -> float:
    kind = node[0]
    if kind == _NUM:
        return node[1]
    if kind == _VAR:
        return env[node[1]]
    if kind == _NEG:
        return -_evaluate(node[1], env)
    if kind == _BIN:
        operator = node[1]
        left = _evaluate(node[2], env)
        right = _evaluate(node[3], env)
        if operator == "+":
            return _finite(left + right, "an addition")
        if operator == "-":
            return _finite(left - right, "a subtraction")
        if operator == "*":
            return _finite(left * right, "a multiplication")
        if operator == "/":
            if right == 0.0:
                raise ExpressionError("division by zero")
            return _finite(left / right, "a division")
        # operator == "**"
        if abs(right) > MAX_EXPONENT:
            raise ExpressionError(f"exponent {right:g} is outside ±{MAX_EXPONENT:g}")
        # math.pow raises instead of returning a complex number for a negative base with a
        # fractional exponent, and raises OverflowError instead of hanging on huge results.
        return _finite(math.pow(left, right), "a power")
    # kind == _CALL
    name = node[1]
    function = FUNCTIONS[name][0]
    args = [_evaluate(arg, env) for arg in node[2]]
    return _finite(float(function(*args)), f"{name}()")


@dataclass(frozen=True)
class Expression:
    """A checked expression. ``source`` is the normalised text that was parsed."""

    source: str
    _tree: tuple

    def evaluate(self, **variables: float) -> float:
        env = {}
        for name in VARIABLES:
            if name not in variables:
                raise ExpressionError(f"missing value for {name!r}")
            value = float(variables[name])
            if not math.isfinite(value):
                raise ExpressionError(f"{name} is not finite")
            env[name] = value
        try:
            return _evaluate(self._tree, env)
        except ExpressionError:
            raise
        except ZeroDivisionError:
            raise ExpressionError("division by zero") from None
        except OverflowError:
            raise ExpressionError("a value in the expression overflowed") from None
        except (ValueError, TypeError) as exc:
            raise ExpressionError(f"math error in the expression: {exc}") from None


def compile_expression(text) -> Expression:
    """Parse and check ``text``; raise ``ExpressionError`` if it is not an allowed expression."""
    source = normalize_source(text)
    if not source:
        raise ExpressionError("the expression is empty")
    if len(source) > MAX_LENGTH:
        raise ExpressionError(f"the expression is longer than {MAX_LENGTH} characters")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"cannot parse the expression: {exc.msg}") from None
    except (ValueError, MemoryError, RecursionError) as exc:
        raise ExpressionError(f"cannot parse the expression: {type(exc).__name__}") from None
    return Expression(source, _Converter().convert(tree.body, 1))


def step_variables(step: int, steps: int, sigma_min: float, sigma_max: float) -> dict:
    """Variables for step ``step`` (0-based) of ``steps``: m, M, n, s and x."""
    x = step / (steps - 1) if steps > 1 else 0.0
    return {"m": float(sigma_min), "M": float(sigma_max), "n": float(steps), "s": float(step), "x": x}
