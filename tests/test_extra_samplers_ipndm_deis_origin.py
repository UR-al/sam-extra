"""Origin-parity tests for ``IPNDM`` / ``IPNDM_V`` / ``DEIS`` (``sam3ext/extra_samplers/ipndm_deis.py``).

The oracle is ComfyUI's own code, verbatim (``tests/_origin_comfyui_ipndm_deis.py``: ``sample_ipndm``,
``sample_ipndm_v``, ``sample_deis`` — zju-pi/diff-sampler under Apache 2, adapted by ComfyUI — and the whole
``deis.py`` at comfyanonymous/ComfyUI@387f98aa; SHA-256 of every block pinned below). The extension side runs
on Forge Neo's real ``k_diffusion.sampling`` and Forge's vendored ``k_diffusion/deis.py`` (both executed from
Forge's files). What must match: the sampled latent, bit for bit, on ε and flow models, 4-D and 5-D latents,
every order the samplers run; Forge's DEIS coefficients and ComfyUI's, coefficient for coefficient.

One documented exception (sam-extra 0.33.1, change 4 of ``ipndm_deis.py``): IPNDM_V's order-4 weight ``coeff4``
ends in ``h_n_2 / h_n_3`` where upstream has the typo ``h_n_1 / h_n_2``. The step code is upstream's but for that
one token pair (``test_the_step_code_is_upstreams_but_the_coeff4_token``); IPNDM_V equals ComfyUI's ``ipndm_v``
bit for bit at orders 2-3 on every list and at order 4 on lists whose consecutive step ratios are equal (equal
steps, a halving list), and on every list it equals ComfyUI's function with that token fixed. Why: on Anima's
Linear Quadratic 28 list upstream's weights sum to about −174 at step 15 — not even a constant velocity is
integrated — while the fixed ones sum to 1 at every step (``LinearQuadraticTests``). Beyond parity: they solve the
flow ODE (convergence on the exact denoiser of Gaussian data; on Linear Quadratic no worse than Euler) and the
callback sees the current latent.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import io
import sys
import textwrap
import tokenize
import unittest
from pathlib import Path

import torch


def _fixtures():
    name = "_extra_samplers_fixtures"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


fx = _fixtures()

from sam3ext.extra_samplers import ipndm_deis  # noqa: E402

ORIGIN_FILE = "_origin_comfyui_ipndm_deis.py"
# SHA-256 of each verbatim block (upstream comfyanonymous/ComfyUI@387f98aa lines, LF).
ORIGIN_BLOCK_SHA256 = {
    "comfy/k_diffusion/sampling.py:1173-1330": "d9a32e74c577669705a8437174cbdbcc298bea0897447588694e022967e054a9",
    "comfy/k_diffusion/deis.py:1-120": "8d4e5057a062b77ef2e33af8208a2d6a3c3fbb431315a4c37689a307d66bc807",
}
STEPS = (1, 2, 3, 4, 5, 8, 28)

ORIGIN = fx.load_comfy_ipndm_deis_origin()
_PAIRS = {
    "ipndm": (ipndm_deis.sample_ipndm, ORIGIN.sample_ipndm),
    "ipndm_v": (ipndm_deis.sample_ipndm_v, ORIGIN.sample_ipndm_v),
    "deis": (ipndm_deis.sample_deis, ORIGIN.sample_deis),
}

# The one exception to upstream's step code (ipndm_deis.py change 4, 0.33.1): coeff4's last factor. temp2 multiplies
# the third divided difference, whose last weight is ``-q * h_n_2 / h_n_3`` (coeff3 carries the matching
# ``q * (1 + h_n_2 / h_n_3)``); upstream wrote ``h_n_1 / h_n_2`` there — the same number only when
# ``h_n_1 / h_n_2 == h_n_2 / h_n_3``.
COEFF4_UPSTREAM_LINE = "coeff4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 / h_n_2"
COEFF4_UPSTREAM = ("h_n_1", "/", "h_n_2")
COEFF4_FIXED = ("h_n_2", "/", "h_n_3")

# Lists whose consecutive step ratios are equal in floating point (the coeff4 fix is the identity there):
# equal steps, and a list whose every step is half the one before (both exact in binary).
CONSTANT_RATIO = ("equal", "halving")

# Anima's Linear Quadratic 28 list, literally: ComfyUI's ``calculate_sigmas(model_sampling, "linear_quadratic",
# 28)`` on CPU (float32), the same list Forge's Linear Quadratic gives Anima. Fourteen equal steps, then the step
# ratio jumps to 3.7 and 2.5 (steps 14-15): the list on which upstream's coeff4 typo turned images into noise.
LQ28 = (
    1.0, 0.9982143044471741, 0.9964285492897034, 0.9946428537368774, 0.9928571581840515,
    0.9910714030265808, 0.9892857074737549, 0.987500011920929, 0.9857142567634583, 0.9839285612106323,
    0.9821428656578064, 0.9803571701049805, 0.9785714149475098, 0.9767857193946838, 0.9750000238418579,
    0.968367338180542, 0.9520407915115356, 0.9260203838348389, 0.8903061151504517, 0.844897985458374,
    0.7897959351539612, 0.7250000238418579, 0.6505101919174194, 0.5663265585899353, 0.4724489748477936,
    0.3688775599002838, 0.25561225414276123, 0.13265305757522583, 0.0,
)


def _comfy_ipndm_v_with_the_coeff4_fix():
    """ComfyUI's verbatim ``sample_ipndm_v`` with only coeff4's last factor replaced — the oracle where the fix
    changes the result. Built from the origin copy's own source, so nothing else can differ."""
    source = inspect.getsource(ORIGIN.sample_ipndm_v)
    assert source.count(COEFF4_UPSTREAM_LINE) == 1
    fixed_line = COEFF4_UPSTREAM_LINE[: -len(" ".join(COEFF4_UPSTREAM))] + " ".join(COEFF4_FIXED)
    namespace = dict(vars(ORIGIN))
    exec(compile(source.replace(COEFF4_UPSTREAM_LINE, fixed_line), ORIGIN.__file__, "exec"), namespace)
    return namespace["sample_ipndm_v"]


COMFY_IPNDM_V_FIXED = _comfy_ipndm_v_with_the_coeff4_fix()


def _step_tokens(fn) -> list:
    """``(type, string)`` tokens of a sampler's step code — from ``order = min(...)`` to the statement before the
    history update — without comments, blank lines or spacing."""
    source = textwrap.dedent(inspect.getsource(fn))
    loop = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.For))
    assert "buffer_model" in ast.unparse(loop.body[-1])      # the history update (``_keep`` here, inline upstream)
    start = next(index for index, stmt in enumerate(loop.body)
                 if isinstance(stmt, ast.Assign) and [ast.unparse(t) for t in stmt.targets] == ["order"])
    statements = loop.body[start:-1]
    lines = source.splitlines(keepends=True)[statements[0].lineno - 1: statements[-1].end_lineno]
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}
    return [(tok.type, "" if tok.type in (tokenize.INDENT, tokenize.DEDENT) else tok.string)
            for tok in tokenize.generate_tokens(io.StringIO(textwrap.dedent("".join(lines))).readline)
            if tok.type not in skip]


def _constant_ratio_sigmas(kind: str, flow: bool, steps: int, dtype) -> torch.Tensor:
    """``equal`` steps of 1/32 (flow) or 1/2 (ε) down to 0, or a ``halving`` list from 1 (flow) or 16 (ε), then 0."""
    if kind == "equal":
        unit = 2.0 ** -5 if flow else 0.5
        values = [(steps - k) * unit for k in range(steps + 1)]
    else:
        values = [(1.0 if flow else 16.0) * 2.0 ** -k for k in range(steps)] + [0.0]
    return torch.tensor(values, dtype=dtype)


def _problem(flow: bool, steps: int, dtype, five_d: bool = False, seed: int = 0, lists: str = "uneven"):
    """``uneven``: the shift-3 flow list or the Karras ε list (step ratios all differ); else a constant-ratio list."""
    sampling = fx.FlowSampling(shift=3.0) if flow else fx.EpsSampling()
    if lists == "uneven":
        sigmas = fx.flow_sigmas(steps, dtype=dtype) if flow else fx.eps_sigmas(steps, dtype=dtype)
    else:
        sigmas = _constant_ratio_sigmas(lists, flow, steps, dtype)
    shape = (1, 4, 2, 6, 5) if five_d else (2, 4, 6, 5)
    x = fx.seeded(shape, seed, dtype=dtype) * (1.0 if flow else float(sigmas[0]))
    return sampling, sigmas, x


def _parity_lists(name: str, max_order: int) -> tuple:
    """The lists on which a sampler must equal ComfyUI's verbatim function: the uneven ones for IPNDM and DEIS; for
    IPNDM_V the constant-ratio ones, and the uneven ones too below order 4 (coeff4 is an order-4 weight) — at order 4
    there it is ComfyUI's with the coeff4 fix (``test_ipndm_v_is_comfyuis_with_the_coeff4_fix_on_every_list``)."""
    if name != "ipndm_v":
        return ("uneven",)
    return CONSTANT_RATIO if max_order >= 4 else ("uneven", *CONSTANT_RATIO)


def _ode_error(fn, sigmas: torch.Tensor, seed: int = 3) -> float:
    """Largest final error on the exact denoiser of Gaussian data (flow), against the closed-form ODE solution."""
    model = fx.GaussianModel(fx.FlowSampling(shift=3.0))
    x = fx.seeded((64, 1, 1, 1), seed, dtype=sigmas.dtype)
    exact = model.ode_transport(x.double(), sigmas[0].double(), torch.tensor(0.0, dtype=torch.float64))
    return float((fn(model, x.clone(), sigmas, extra_args={}, disable=True).double() - exact).abs().max())


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = fx.forge_k_sampling()
        cls.deis = fx.forge_deis()

    def setUp(self):
        self.enterContext(fx.installed_k_sampling(self.ks))
        self.enterContext(fx.installed_k_diffusion_deis(self.deis))


class OriginCopyTests(_Base):
    def test_every_block_is_upstream_verbatim(self):
        blocks = fx.origin_blocks(ORIGIN_FILE)
        self.assertEqual(set(blocks), set(ORIGIN_BLOCK_SHA256))
        for key, digest in ORIGIN_BLOCK_SHA256.items():
            with self.subTest(block=key):
                self.assertEqual(fx.sha256(blocks[key]), digest)
        sampling_block = blocks["comfy/k_diffusion/sampling.py:1173-1330"]
        self.assertEqual(sampling_block.count("#From https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py\n"
                                              "#under Apache 2 license"), 3)
        self.assertEqual(sampling_block.count(COEFF4_UPSTREAM_LINE), 1)

    def test_the_step_code_is_upstreams_but_the_coeff4_token(self):
        """Comments and spacing aside, each sampler's step code (``order = …`` up to the history update) is
        upstream's, token for token — IPNDM_V's but for coeff4's last factor, ``h_n_1 / h_n_2`` → ``h_n_2 / h_n_3``
        (change 4); nothing else may differ."""
        for name, (ours, origin) in _PAIRS.items():
            with self.subTest(sampler=name):
                upstream = _step_tokens(origin)
                expected = list(upstream)
                if name == "ipndm_v":
                    strings = [string for _type, string in upstream]
                    start = strings.index("coeff4")
                    self.assertEqual(strings[start + 1], "=")
                    end = next(index for index in range(start, len(upstream)) if upstream[index][0] == tokenize.NEWLINE)
                    self.assertEqual(tuple(strings[end - 3:end]), COEFF4_UPSTREAM)
                    expected[end - 3:end] = [(kind, string) for (kind, _old), string in zip(upstream[end - 3:end], COEFF4_FIXED)]
                    self.assertEqual(sum(a != b for a, b in zip(expected, upstream)), 2)   # the two names
                self.assertEqual(_step_tokens(ours), expected)

    def test_the_defaults_are_comfyuis(self):
        for name, (ours, origin) in _PAIRS.items():
            with self.subTest(sampler=name):
                ours_params = inspect.signature(ours).parameters
                origin_params = inspect.signature(origin).parameters
                self.assertEqual(ours_params["max_order"].default, origin_params["max_order"].default)
        self.assertEqual((ipndm_deis.IPNDM_MAX_ORDER, ipndm_deis.DEIS_MAX_ORDER, ipndm_deis.DEIS_MODE), (4, 3, "tab"))
        self.assertEqual(inspect.signature(ipndm_deis.sample_deis).parameters["deis_mode"].default, "tab")

    def test_forges_deis_coefficients_are_comfyuis(self):
        """Forge's vendored deis.py differs from ComfyUI's only in formatting and an unused lambda."""
        for sigmas in (fx.flow_sigmas(12), fx.flow_sigmas(12)[1:], fx.eps_sigmas(12), fx.flow_sigmas(9, dtype=torch.float32)):
            for max_order in (1, 2, 3, 4):
                with self.subTest(first=float(sigmas[0]), dtype=sigmas.dtype, max_order=max_order):
                    forge = self.deis.get_deis_coeff_list(sigmas, max_order, deis_mode="tab")
                    comfy = ORIGIN.get_deis_coeff_list(sigmas, max_order, deis_mode="tab")
                    self.assertEqual([len(c) for c in forge], [len(c) for c in comfy])
                    for a_list, b_list in zip(forge, comfy):
                        for a, b in zip(a_list, b_list):
                            self.assertTrue(torch.equal(a, b) or (torch.isnan(a) and torch.isnan(b)))


class OriginParityTests(_Base):
    def _same(self, ours, origin, flow, dtype, five_d, steps, max_order, lists):
        sampling, sigmas, x = _problem(flow, steps, dtype, five_d, lists=lists)
        expected = origin(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True, max_order=max_order)
        model = fx.ToyModel(sampling)
        got = ours(model, x.clone(), sigmas, extra_args={}, disable=True, max_order=max_order)
        self.assertTrue(torch.isfinite(got).all())
        self.assertTrue(torch.equal(got, expected))
        self.assertEqual(len(model.calls), steps)   # one model call per step

    def test_the_constant_ratio_lists_have_bit_equal_step_ratios(self):
        """Where the coeff4 fix is the identity: ``h_n_1 / h_n_2 == h_n_2 / h_n_3`` in the list's own dtype."""
        for kind in CONSTANT_RATIO:
            for flow in (False, True):
                for dtype in (torch.float32, torch.float64):
                    with self.subTest(kind=kind, flow=flow, dtype=dtype):
                        sigmas = _constant_ratio_sigmas(kind, flow, 28, dtype)
                        h = sigmas[1:] - sigmas[:-1]
                        self.assertTrue((h < 0).all())
                        ratios = h[1:-2] / h[:-3]   # h_(k+1) / h_k over h_0 .. h_25, the steps order 4 looks back on
                        self.assertTrue(torch.equal(ratios, ratios[:1].expand_as(ratios)))

    def test_every_sampler_is_comfyuis_bit_for_bit(self):
        """IPNDM and DEIS on the shift-3 flow and Karras ε lists; IPNDM_V there at orders 2-3 (coeff4 is an order-4
        weight) and at every order on the constant-ratio lists, where it is ComfyUI's ``ipndm_v`` bit for bit."""
        for name, (ours, origin) in _PAIRS.items():
            for flow in (False, True):
                for dtype in (torch.float32, torch.float64):
                    for five_d in (False, True):
                        for steps in STEPS:
                            for max_order in (2, 3, 4):
                                for lists in _parity_lists(name, max_order):
                                    with self.subTest(sampler=name, flow=flow, dtype=dtype, five_d=five_d, steps=steps,
                                                      max_order=max_order, lists=lists):
                                        self._same(ours, origin, flow, dtype, five_d, steps, max_order, lists)

    def test_ipndm_v_is_comfyuis_with_the_coeff4_fix_on_every_list(self):
        """On every list IPNDM_V is ComfyUI's ``sample_ipndm_v`` with coeff4's token fixed, bit for bit. On the
        uneven lists that differs from upstream from the first order-4 step on (5+ steps) — the only results the
        fix changes; on the constant-ratio lists it does not."""
        ours, origin = _PAIRS["ipndm_v"]
        for flow in (False, True):
            for dtype in (torch.float32, torch.float64):
                for five_d in (False, True):
                    for steps in STEPS:
                        for lists in ("uneven", *CONSTANT_RATIO):
                            with self.subTest(flow=flow, dtype=dtype, five_d=five_d, steps=steps, lists=lists):
                                self._same(ours, COMFY_IPNDM_V_FIXED, flow, dtype, five_d, steps, 4, lists)
                                sampling, sigmas, x = _problem(flow, steps, dtype, five_d, lists=lists)
                                got = ours(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True)
                                upstream = origin(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True)
                                self.assertEqual(torch.equal(got, upstream), lists != "uneven" or steps < 5)

    def test_the_registered_defaults_are_comfyuis_defaults(self):
        for name, (ours, origin) in _PAIRS.items():
            for flow in (False, True):
                for lists in _parity_lists(name, ipndm_deis.IPNDM_MAX_ORDER):
                    with self.subTest(sampler=name, flow=flow, lists=lists):
                        sampling, sigmas, x = _problem(flow, 28, torch.float32, lists=lists)
                        expected = origin(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True)
                        got = ours(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True)
                        self.assertTrue(torch.equal(got, expected))
        for flow in (False, True):
            with self.subTest(sampler="ipndm_v", flow=flow, lists="uneven", oracle="coeff4 fixed"):
                sampling, sigmas, x = _problem(flow, 28, torch.float32)
                expected = COMFY_IPNDM_V_FIXED(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True)
                got = ipndm_deis.sample_ipndm_v(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True)
                self.assertTrue(torch.equal(got, expected))

    def test_order_one_is_euler_where_upstream_raises(self):
        """Upstream's history update indexes an empty list at ``max_order = 1``; here order 1 keeps no history
        and is Forge's Euler step (on a grid that does not end at 0, where upstream would denoise)."""
        sampling, sigmas, x = _problem(True, 8, torch.float64)
        for name, (ours, origin) in _PAIRS.items():
            with self.subTest(sampler=name):
                with self.assertRaises(IndexError):
                    origin(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True, max_order=1)
                euler = self.ks.sample_euler(fx.ToyModel(sampling), x.clone(), sigmas[:-1], extra_args={}, disable=True)
                got = ours(fx.ToyModel(sampling), x.clone(), sigmas[:-1], extra_args={}, disable=True, max_order=1)
                self.assertTrue(torch.equal(got, euler))
                # orders above 4 are clamped to 4 (upstream would skip the update)
                four = ours(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True, max_order=4)
                self.assertTrue(torch.equal(ours(fx.ToyModel(sampling), x.clone(), sigmas, extra_args={}, disable=True,
                                                 max_order=9), four))

    def test_the_callback_sees_the_current_latent(self):
        for name, (ours, _origin) in _PAIRS.items():
            with self.subTest(sampler=name):
                sampling, sigmas, x = _problem(True, 6, torch.float64)
                inputs, seen = [], []

                class Recorder(fx.ToyModel):
                    def __call__(self, latent, sigma, **kwargs):
                        inputs.append(latent.clone())
                        return super().__call__(latent, sigma, **kwargs)

                ours(Recorder(sampling), x.clone(), sigmas, extra_args={}, disable=True, callback=seen.append)
                self.assertEqual([info["i"] for info in seen], list(range(6)))
                for info, latent in zip(seen, inputs):
                    self.assertTrue(torch.equal(info["x"], latent))
                self.assertFalse(torch.equal(seen[-1]["x"], x))


class FlowOdeTests(_Base):
    """On the exact denoiser of Gaussian data (closed-form ODE solution) on Anima's shift-3 grid."""

    def _error(self, fn, steps):
        return _ode_error(fn, fx.flow_sigmas(steps))

    def test_they_solve_the_flow_ode(self):
        errors = {name: [self._error(ours, n) for n in (16, 32, 64)] for name, (ours, _o) in _PAIRS.items()}
        euler = [self._error(self.ks.sample_euler, n) for n in (16, 32, 64)]
        for name, errs in errors.items():
            with self.subTest(sampler=name, errors=errs):
                self.assertLess(errs[1], errs[0] / 3)
                self.assertLess(errs[2], errs[1] / 3)
                for ours, plain in zip(errs, euler):
                    self.assertLess(ours, plain)
        # On this slowly varying grid the variable-step weights (coeff4 fixed) and the fixed AB4 weights are about
        # equally accurate: within 5% either way at 16/32/64 steps. Upstream's coeff4 typo happened to make IPNDM_V
        # 10-20% more accurate here (its weights still sum to 0.99-1.0 on this grid) — the same typo breaks the
        # Linear Quadratic list (LinearQuadraticTests).
        for v, fixed in zip(errors["ipndm_v"], errors["ipndm"]):
            self.assertLess(abs(v / fixed - 1), 0.05)


class LinearQuadraticTests(_Base):
    """Why IPNDM_V carries the coeff4 fix: Anima's Linear Quadratic 28 list (``LQ28``), whose step ratio jumps."""

    @staticmethod
    def _weight_sums(fn, sigmas: torch.Tensor) -> torch.Tensor:
        """Constant-velocity probe: with ``D = x − σ·v`` every derivative is ``v``, so a step moves ``x`` by
        ``(σ_next − σ)·(Σ weights)·v`` — exactly right when the weights sum to 1. Returns that sum per step (rows)
        as each latent element sees it (columns)."""
        v = torch.tensor([[0.7, -1.3, 0.45, 2.1]], dtype=sigmas.dtype)
        inputs = []

        def model(x, sigma, **kwargs):
            inputs.append(x.clone())
            return x - sigma.reshape(-1, 1).to(x.dtype) * v

        out = fn(model, fx.seeded((1, 4), 5, dtype=sigmas.dtype), sigmas, extra_args={}, disable=True)
        latents = inputs + [out]
        return torch.cat([(latents[i + 1] - latents[i]) / ((sigmas[i + 1] - sigmas[i]) * v)
                          for i in range(len(sigmas) - 1)])

    def test_the_weights_sum_to_one_at_every_step(self):
        sigmas = torch.tensor(LQ28, dtype=torch.float64)
        sums = self._weight_sums(ipndm_deis.sample_ipndm_v, sigmas)
        self.assertEqual(tuple(sums.shape), (28, 4))
        self.assertLessEqual(float((sums - 1).abs().max()), 1e-9)
        # The reason for the fix: upstream's weights (ComfyUI's verbatim ipndm_v) are right through the 14 equal
        # steps and wrong from the jump on — about -174 at step 15 (σ 0.968 → 0.952), so the latent moves 174 times
        # the step, the wrong way; Forge and ComfyUI both drew green noise there.
        upstream = self._weight_sums(ORIGIN.sample_ipndm_v, sigmas)
        per_step = upstream.mean(dim=1)
        self.assertLess(float((upstream - per_step[:, None]).abs().max()), 1e-6)   # every element sees one sum
        self.assertLess(float((per_step[:15] - 1).abs().max()), 1e-3)
        self.assertAlmostEqual(float(per_step[15]), -174.0, delta=0.5)
        self.assertEqual(int((per_step - 1).abs().argmax()), 15)

    def test_it_solves_the_flow_ode_no_worse_than_euler(self):
        """The exact denoiser of Gaussian data on LQ28 (float64, and float32 as Forge samples): IPNDM_V's error is at
        most Euler's (and about IPNDM's); upstream's formula is more than ten times Euler's."""
        ours, origin = _PAIRS["ipndm_v"]
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                sigmas = torch.tensor(LQ28, dtype=dtype)
                errors = {"ipndm_v": _ode_error(ours, sigmas), "euler": _ode_error(self.ks.sample_euler, sigmas),
                          "ipndm": _ode_error(ipndm_deis.sample_ipndm, sigmas), "upstream": _ode_error(origin, sigmas)}
                self.assertLessEqual(errors["ipndm_v"], errors["euler"], errors)
                self.assertLess(abs(errors["ipndm_v"] / errors["ipndm"] - 1), 0.05, errors)
                self.assertGreater(errors["upstream"], 10 * errors["euler"], errors)


if __name__ == "__main__":
    unittest.main()
