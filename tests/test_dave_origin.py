"""Origin-parity tests for the DAVE early-step gate, block default and strength threshold.

The oracle is the upstream ComfyUI node itself: its step lookup, tau gate and
block-pooling rule are copied below verbatim (MIT, sorryhyun/ComfyUI-Anima-DAVE),
together with the ComfyUI/Forge code that decides which sigmas a sampler visits.
Each copy carries an ``origin:`` comment with repository, commit, file and lines.

Upstream finds the forward's sigma in the schedule the sampler walks and runs
DAVE while that index is below ``k = max(1, min(n, round(tau·n)))``. On Anima
(flow, shift 3) with tau 0.10 that is steps {0,1} at 20/25 steps, {0,1,2} at
28/30 and 0..4 at 50 — not the lagging step fraction's 0-2 / 0-3 / 0-3 / 0-5.
An off-schedule sigma (a second-order midpoint) is step 0 and always on.
Expected tables come from running the original node on CPU
(scratchpad origin_parity/dave/golden_gate.py, 2026-09-25).

Forge keeps its whole sigma list in ``sampling_sigmas`` and walks it from
``steps - t_enc - 1`` on. The tests take that offset from Forge's real
``setup_img2img_steps`` and check it against Forge's ``DDIM`` schedule type,
which returns ``steps + 2`` sigmas at 24/28/30/32 steps
(scratchpad origin_parity/davef_review/ddim_sched_tail.py).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.guidance import dave_gate  # noqa: E402


def _load_base_tests():
    """The PAG test module's loaders, under a private name (not collected twice)."""
    spec = importlib.util.spec_from_file_location(
        "_dave_origin_base_tests", ROOT / "tests" / "test_anima_safe_pag.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_tests()


# ---------------------------------------------------------------------------
# Upstream oracle — MIT License, Copyright (c) 2026 Seunghyun Ji
# (ComfyUI-Anima-DAVE; full notice in THIRD_PARTY_NOTICES.md). Verbatim.
# ---------------------------------------------------------------------------


# origin: sorryhyun/ComfyUI-Anima-DAVE@83143e8d84768e25f72755ec00ea00ded07ee06e:nodes.py:91-106
def _current_step(transformer_options: dict, timestep: torch.Tensor):
    """(step_index, n_steps) for the current forward, mapped against the schedule.

    Mirrors ComfyUI's own ``isclose(sample_sigmas, sigma_now)`` mapping
    (``comfy/context_windows.py``). Returns ``(None, None)`` if the schedule is
    unavailable (custom samplers that don't publish ``sample_sigmas``).
    """
    sched = transformer_options.get("sample_sigmas")
    if sched is None or len(sched) < 2:
        return None, None
    cur = transformer_options.get("sigmas", timestep)
    cur0 = cur.flatten()[0].to(sched.device)
    matches = torch.isclose(sched, cur0, rtol=1e-4, atol=1e-6)
    nz = torch.nonzero(matches).flatten()
    step = int(nz[0].item()) if nz.numel() else 0
    return step, len(sched) - 1


def _origin_gate(tau_f, topts, t):
    # origin: sorryhyun/ComfyUI-Anima-DAVE@83143e8d:nodes.py:199-208 (dave_apply_wrapper body)
    # Decide the early-step gate for this forward. tau == 0 → every step.
    if tau_f > 0.0:
        step, n_steps = _current_step(topts, t)
        if step is None:  # no schedule published → run every step (safe default)
            gate = True
        else:
            k = max(1, min(n_steps, round(tau_f * n_steps)))
            gate = step < k
    else:
        gate = True
    return gate


def _origin_pooled(strength, weight):
    # origin: sorryhyun/ComfyUI-Anima-DAVE@83143e8d:nodes.py:165-166
    atten = np.clip(float(strength) * weight, 0.0, 1.0)
    pooled = [(i, float(a)) for i, a in enumerate(atten) if a > 1e-3]
    return pooled


# origin: sorryhyun/ComfyUI-Anima-DAVE@83143e8d:dave_alpha.npz — ``weight`` float32[28],
# 1.0 on flat blocks 8..18 and 0.0 elsewhere; loaded as float64 (nodes.py:67-72).
_ORIGIN_WEIGHT = np.array([0.0] * 8 + [1.0] * 11 + [0.0] * 9, dtype=np.float32).astype(np.float64)
# origin: nodes.py:124-127, 135-138 (strength / tau inputs)
_ORIGIN_STRENGTH = {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}
_ORIGIN_TAU = {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}


# Active steps per (tau, steps) from the original node's wrapper on a
# shift-3 simple schedule (golden_gate.py). Always a prefix 0..count-1.
ORIGIN_ACTIVE_STEP_COUNT = {
    0.05: {8: 1, 20: 1, 24: 1, 25: 1, 28: 1, 30: 2, 32: 2, 40: 2, 45: 2, 50: 2},
    0.1: {8: 1, 20: 2, 24: 2, 25: 2, 28: 3, 30: 3, 32: 3, 40: 4, 45: 4, 50: 5},
    0.15: {8: 1, 20: 3, 24: 4, 25: 4, 28: 4, 30: 4, 32: 5, 40: 6, 45: 7, 50: 8},
    0.3: {8: 2, 20: 6, 24: 7, 25: 8, 28: 8, 30: 9, 32: 10, 40: 12, 45: 14, 50: 15},
    1.0: {8: 8, 20: 20, 24: 24, 25: 25, 28: 28, 30: 30, 32: 32, 40: 40, 45: 45, 50: 50},
}


# ---------------------------------------------------------------------------
# Host code: Anima's flow sigmas and the schedules a sampler walks.
# ---------------------------------------------------------------------------


def _time_snr_shift(alpha, t):
    # origin: Comfy-Org/ComfyUI@387f98aa2822f684b8597959a52a467d88cc4806:comfy/model_sampling.py:289-292
    # (same body in Haoming02/sd-webui-forge-classic@e33f40e4:backend/modules/k_prediction.py:32-35)
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def _flow_sigmas(shift=3.0, timesteps=1000):
    # origin: Comfy-Org/ComfyUI@387f98aa:comfy/model_sampling.py:307-311, 327-328
    # (= Haoming02/sd-webui-forge-classic@e33f40e4:backend/modules/k_prediction.py:190-194)
    return _time_snr_shift(shift, torch.arange(1, timesteps + 1, 1) / timesteps)


_SIGMAS = _flow_sigmas()
_FORGE = ROOT.parents[1]


def _forge_function(relpath, name, namespace):
    """One top-level function of the Forge checkout, executed in ``namespace``.

    The body is Forge's own source (read with ``ast``), so the tests follow the
    host code the gate relies on without importing Forge's module graph."""
    source = (_FORGE / relpath).read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            code = compile(module, str(_FORGE / relpath), "exec")
            scope = dict(namespace)
            exec(code, scope)  # noqa: S102 - Forge's own function body
            return scope[name]
    raise AssertionError(f"{name} not found in {relpath}")


def _setup_img2img_steps(fix_steps=False):
    # origin: Haoming02/sd-webui-forge-classic@e33f40e4:modules/sd_samplers_common.py:43-52
    return _forge_function(
        "modules/sd_samplers_common.py", "setup_img2img_steps",
        {"opts": types.SimpleNamespace(img2img_fix_steps=fix_steps)},
    )


# origin: Haoming02/sd-webui-forge-classic@e33f40e4:modules/sd_schedulers.py:124-133
# (ddim_scheduler, Forge's "DDIM" schedule type; the same list as
# Comfy-Org/ComfyUI@387f98aa:comfy/samplers.py:654-669, "ddim_uniform")
_FORGE_DDIM = _forge_function("modules/sd_schedulers.py", "ddim_scheduler", {"torch": torch})


def _ddim_sigmas(steps):
    inner = types.SimpleNamespace(sigmas=_SIGMAS)
    return _FORGE_DDIM(steps, float(_SIGMAS[0]), float(_SIGMAS[-1]), inner, "cpu")


def _simple_sigmas(steps):
    # origin: Comfy-Org/ComfyUI@387f98aa:comfy/samplers.py:645-652 (simple_scheduler)
    # (= Haoming02/sd-webui-forge-classic@e33f40e4:modules/sd_schedulers.py:49-55)
    sigs = []
    ss = len(_SIGMAS) / steps
    for x in range(steps):
        sigs += [float(_SIGMAS[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def _forge_img2img(steps, denoise, schedule=_simple_sigmas, *, requested=None, fix_steps=False):
    """``(sampling_sigmas, sampled, offset, sampling_steps)`` of a Forge img2img / hires run.

    origin: Haoming02/sd-webui-forge-classic@e33f40e4:modules/sd_samplers_kdiffusion.py:145-148
    (steps, t_enc = setup_img2img_steps(p, steps); sigma_sched = sigmas[steps - t_enc - 1:]),
    :192 (sampling_sigmas = the whole list), :194-195 + sd_samplers_common.py:435-438
    (launch_sampling(t_enc + 1) → state.sampling_steps). ``requested`` is the
    hires pass's ``hr_second_pass_steps or steps`` (processing.py:1552); img2img
    passes None (:1920).
    """
    request = types.SimpleNamespace(steps=steps, denoising_strength=denoise)
    total, t_enc = _setup_img2img_steps(fix_steps)(request, requested)
    sigmas = schedule(total)
    offset = total - t_enc - 1
    return sigmas, sigmas[offset:], offset, t_enc + 1


def _comfy_denoise(steps, denoise):
    # origin: Comfy-Org/ComfyUI@387f98aa:comfy/samplers.py:1431-1441 (KSampler.set_steps)
    new_steps = int(steps / denoise)
    return _simple_sigmas(new_steps)[-(steps + 1):]


def _origin_active(tau, sample_sigmas, sigmas_now):
    return [
        _origin_gate(tau, {"sample_sigmas": sample_sigmas, "sigmas": s}, s)
        for s in sigmas_now
    ]


def _rows(value, batch=2):
    """A forward's ``transformer_options['sigmas']`` — the sigma for every batch row."""
    return torch.full((batch,), float(value))


# ---------------------------------------------------------------------------
# 1) sam3ext.guidance.dave_gate == upstream gate
# ---------------------------------------------------------------------------


class DaveGateOriginTests(unittest.TestCase):
    STEPS = (8, 20, 24, 25, 28, 30, 32, 40, 45, 50)
    TAUS = (0.05, 0.1, 0.15, 0.3, 1.0)

    def test_step_tables_match_origin(self):
        for tau in self.TAUS:
            for steps in self.STEPS:
                sched = _simple_sigmas(steps)
                now = [_rows(s) for s in sched[:-1]]
                with self.subTest(tau=tau, steps=steps):
                    origin = _origin_active(tau, sched, now)
                    ours = [dave_gate.gate_active(tau, sched, s) for s in now]
                    self.assertEqual(ours, origin)
                    count = ORIGIN_ACTIVE_STEP_COUNT[tau][steps]
                    self.assertEqual(origin, [True] * count + [False] * (steps - count))

    def test_tau_010_windows(self):
        for steps, expected in ((20, {0, 1}), (25, {0, 1}), (28, {0, 1, 2}),
                                (30, {0, 1, 2}), (50, {0, 1, 2, 3, 4})):
            sched = _simple_sigmas(steps)
            with self.subTest(steps=steps):
                active = {
                    i for i, s in enumerate(sched[:-1])
                    if dave_gate.gate_active(0.1, sched, _rows(s))
                }
                self.assertEqual(active, expected)

    def test_window_steps_is_python_round(self):
        # round-half-even like upstream: 0.1·25 = 2.5 → 2, 0.15·30 = 4.5 → 4.
        self.assertEqual(dave_gate.window_steps(0.1, 25), 2)
        self.assertEqual(dave_gate.window_steps(0.15, 30), 4)
        self.assertEqual(dave_gate.window_steps(0.05, 8), 1)   # max(1, ...)
        self.assertEqual(dave_gate.window_steps(1.0, 20), 20)  # min(n, ...)
        for tau in (0.01, 0.05, 0.1, 0.25, 0.5, 0.99):
            for n in range(1, 60):
                self.assertEqual(dave_gate.window_steps(tau, n),
                                 max(1, min(n, round(tau * n))))

    def test_off_schedule_sigma_is_step_zero_so_on(self):
        sched = _simple_sigmas(30)
        # step 5 exact: past k = 3 → off; the (5, 6) midpoint is off-schedule → step 0 → on.
        exact = _rows(sched[5])
        mid = _rows((sched[5] + sched[6]) / 2)
        self.assertFalse(dave_gate.gate_active(0.1, sched, exact))
        self.assertTrue(dave_gate.gate_active(0.1, sched, mid))
        for i in range(29):
            mid = _rows((sched[i] + sched[i + 1]) / 2)
            with self.subTest(midpoint=i):
                self.assertTrue(_origin_gate(0.1, {"sample_sigmas": sched, "sigmas": mid}, mid))
                self.assertTrue(dave_gate.gate_active(0.1, sched, mid))

    def test_isclose_tolerance_matches_origin(self):
        sched = _simple_sigmas(30)
        for factor in (1 + 5e-5, 1 - 5e-5, 1 + 5e-4, 1 - 5e-4, 1 + 1e-4, 1.0):
            sigma = (sched[5] * factor).reshape(1)
            with self.subTest(factor=factor):
                self.assertEqual(
                    dave_gate.gate_active(0.1, sched, sigma),
                    _origin_gate(0.1, {"sample_sigmas": sched, "sigmas": sigma}, sigma),
                )
        # golden_gate.py: ×(1+5e-5) still step 5 (off), ×(1+5e-4) off-schedule (on).
        self.assertFalse(dave_gate.gate_active(0.1, sched, (sched[5] * (1 + 5e-5)).reshape(1)))
        self.assertTrue(dave_gate.gate_active(0.1, sched, (sched[5] * (1 + 5e-4)).reshape(1)))

    def test_first_matching_entry_wins(self):
        sched = torch.tensor([1.0, 0.8, 0.8, 0.8, 0.5, 0.0])
        sigma = _rows(0.8)
        self.assertEqual(dave_gate.current_step(sched, sigma),
                         _current_step({"sample_sigmas": sched}, sigma))
        self.assertEqual(dave_gate.current_step(sched, sigma), (1, 5))

    def test_no_schedule_short_schedule_and_tau_zero_are_on(self):
        sigma = torch.tensor([0.1])
        self.assertTrue(_origin_gate(0.1, {"sigmas": sigma}, sigma))
        self.assertTrue(dave_gate.gate_active(0.1, None, sigma))
        short = torch.tensor([1.0])
        self.assertTrue(_origin_gate(0.1, {"sample_sigmas": short, "sigmas": sigma}, sigma))
        self.assertTrue(dave_gate.gate_active(0.1, short, sigma))
        sched = _simple_sigmas(30)
        late = _rows(sched[29])
        self.assertTrue(_origin_gate(0.0, {"sample_sigmas": sched, "sigmas": late}, late))
        self.assertTrue(dave_gate.gate_active(0.0, sched, late))
        self.assertFalse(dave_gate.gate_active(0.1, sched, late))

    def test_list_schedule_and_float_sigma_match_tensor_path(self):
        for steps in (20, 30, 50):
            sched = _simple_sigmas(steps)
            listed = [float(v) for v in sched]
            for i in range(steps):
                with self.subTest(steps=steps, step=i):
                    self.assertEqual(
                        dave_gate.gate_active(0.1, listed, float(sched[i])),
                        dave_gate.gate_active(0.1, sched, _rows(sched[i])),
                    )

    def test_sigma_dtype_is_cast_to_the_schedule(self):
        sched = _simple_sigmas(25)
        for i in (0, 1, 2, 10):
            sigma64 = sched[i].reshape(1).to(torch.float64)
            with self.subTest(step=i):
                self.assertEqual(dave_gate.gate_active(0.1, sched, sigma64),
                                 dave_gate.gate_active(0.1, sched, sched[i].reshape(1)))

    def test_attenuation_threshold_matches_origin_pool(self):
        for strength in (0.0, 0.001, 0.0011, 0.3, 1.0, 1.7, -0.5):
            with self.subTest(strength=strength):
                self.assertEqual(dave_gate.attenuation_active(strength),
                                 bool(_origin_pooled(strength, _ORIGIN_WEIGHT)))
        # golden_gate.py: 0.001 → passthrough, 0.0011 → wrapper installed.
        self.assertFalse(dave_gate.attenuation_active(0.001))
        self.assertTrue(dave_gate.attenuation_active(0.0011))

    def test_default_blocks_are_the_shipped_mask(self):
        pag = _BASE._load_pag_module()
        pooled = [i for i, _a in _origin_pooled(0.3, _ORIGIN_WEIGHT)]
        self.assertEqual(sorted(pag._parse_blocks(dave_gate.DEFAULT_BLOCKS, 28)), pooled)
        self.assertEqual(pooled, list(range(8, 19)))


# ---------------------------------------------------------------------------
# 2) Forge's walked slice — the schedule ComfyUI would publish
# ---------------------------------------------------------------------------


def _walked_active(tau, sched, offset):
    """(origin, ours) active flags over the sigmas Forge walks from ``offset``."""
    walked = sched[offset:] if offset else sched
    now = [_rows(s) for s in walked[:-1]]
    origin = _origin_active(tau, walked, now)
    ours = [dave_gate.gate_active(tau, sched, s, offset) for s in now]
    return origin, ours


class SampledScheduleTests(unittest.TestCase):
    def test_forge_img2img_uses_the_walked_slice(self):
        for steps, denoise in ((20, 0.5), (30, 0.6), (15, 0.35), (28, 0.75)):
            full, sampled, offset, sampling_steps = _forge_img2img(steps, denoise)
            with self.subTest(steps=steps, denoise=denoise):
                self.assertTrue(torch.equal(dave_gate.sampled_schedule(full, offset), sampled))
                origin, ours = _walked_active(0.1, full, offset)
                self.assertEqual(ours, origin)
                count = dave_gate.window_steps(0.1, sampling_steps)
                self.assertEqual(origin, [True] * count + [False] * (len(sampled) - 1 - count))

    def test_ddim_schedule_has_more_than_steps_plus_one_sigmas(self):
        # Why the offset is needed: Forge's "DDIM" type on Anima's 1000-sigma table.
        for steps, length in ((20, 21), (24, 26), (25, 26), (28, 30), (30, 32), (32, 34), (50, 51)):
            with self.subTest(steps=steps):
                self.assertEqual(len(_ddim_sigmas(steps)), length)

    def test_ddim_txt2img_matches_origin(self):
        # txt2img walks the whole list: offset 0.
        for steps, expected in ((24, [0, 1]), (28, [0, 1, 2]), (30, [0, 1, 2]), (32, [0, 1, 2])):
            sched = _ddim_sigmas(steps)
            with self.subTest(steps=steps):
                origin, ours = _walked_active(0.1, sched, 0)
                self.assertEqual(ours, origin)
                self.assertEqual([i for i, on in enumerate(origin) if on], expected)

    def test_ddim_img2img_matches_origin(self):
        for steps, denoise in ((28, 0.5), (30, 0.6), (24, 0.75)):
            full, _sampled, offset, _sampling_steps = _forge_img2img(steps, denoise, _ddim_sigmas)
            with self.subTest(steps=steps, denoise=denoise):
                self.assertEqual(offset, steps - int(denoise * steps) - 1)
                origin, ours = _walked_active(0.1, full, offset)
                self.assertEqual(ours, origin)
                self.assertEqual([i for i, on in enumerate(origin) if on], [0, 1])

    def test_ddim_hires_matches_origin(self):
        # Hires passes ``hr_second_pass_steps or steps`` → the fix-steps branch.
        for steps, denoise in ((28, 0.5), (30, 0.35)):
            full, _sampled, offset, sampling_steps = _forge_img2img(
                steps, denoise, _ddim_sigmas, requested=steps)
            with self.subTest(steps=steps, denoise=denoise):
                self.assertEqual(sampling_steps, steps)
                origin, ours = _walked_active(0.1, full, offset)
                self.assertEqual(ours, origin)

    def test_counting_back_sampling_steps_misses_the_first_ddim_sigma(self):
        # The rule this replaces: keep the last ``sampling_steps + 1`` entries.
        sched = _ddim_sigmas(28)
        tail = sched[len(sched) - (28 + 1):]
        self.assertEqual(float(tail[0]), float(sched[1]))
        full, sampled, _offset, sampling_steps = _forge_img2img(28, 0.5, _ddim_sigmas)
        tail = full[len(full) - (sampling_steps + 1):]
        self.assertEqual(len(tail), len(sampled) - 1)

    def test_without_the_slice_img2img_would_never_run(self):
        full, sampled, offset, sampling_steps = _forge_img2img(20, 0.5)
        self.assertEqual((offset, sampling_steps), (9, 11))
        # The first sampled sigma is index 9 of the whole list: past k = 2 there.
        self.assertFalse(dave_gate.gate_active(0.1, full, _rows(sampled[0])))
        self.assertTrue(dave_gate.gate_active(0.1, full, _rows(sampled[0]), offset))

    def test_fix_steps_img2img_matches_comfy_denoise(self):
        # Forge "img2img_fix_steps" samples the same slice ComfyUI does.
        for steps, denoise, expected in ((20, 0.5, 2), (30, 0.4, 3)):
            full, sampled, offset, _sampling_steps = _forge_img2img(steps, denoise, fix_steps=True)
            with self.subTest(steps=steps, denoise=denoise):
                self.assertTrue(torch.equal(sampled, _comfy_denoise(steps, denoise)))
                origin, ours = _walked_active(0.1, full, offset)
                self.assertEqual(ours, origin)
                self.assertEqual(sum(origin), expected)

    def test_no_offset_keeps_the_whole_list(self):
        sched = _simple_sigmas(20)
        for offset in (None, 0, "x"):
            with self.subTest(offset=offset):
                self.assertIs(dave_gate.sampled_schedule(sched, offset), sched)
        self.assertIsNone(dave_gate.sampled_schedule(None, 5))
        self.assertTrue(torch.equal(dave_gate.sampled_schedule(sched, 5), sched[5:]))


# ---------------------------------------------------------------------------
# 3) Forge wiring: the Anima block wrapper in scripts/anima_safe_pag.py
# ---------------------------------------------------------------------------


class _State:
    def __init__(self, sampling_step=0, sampling_steps=0):
        self.sampling_step = sampling_step
        self.sampling_steps = sampling_steps


class BlockWrapperGateTests(unittest.TestCase):
    TARGETS = {1, 2, 3}

    @classmethod
    def setUpClass(cls):
        cls.pag = _BASE._load_pag_module()

    def setUp(self):
        p = self.pag
        saved_dave = dict(p._DAVE)
        saved_state = {key: p._STATE.get(key) for key in ("any_b0", "slg_b0", "slg_b1", "dedup_until")}
        self.addCleanup(p._DAVE.update, saved_dave)
        self.addCleanup(p._STATE.update, saved_state)
        p._STATE.update(any_b0=None, slg_b0=None, slg_b1=None, dedup_until=None)
        p._DAVE.update(
            on=True, strength=0.3, tau=0.1, targets=set(self.TARGETS), steps=0,
            gate=dave_gate.ForwardGateCache(), schedule_ok=True, offset=0,
        )
        self.block_out = torch.randn(2, 1, 4, 5, 8, generator=torch.Generator().manual_seed(0)) + 1.0

    def _block(self, idx=1):
        return self.pag._make_block_wrapper(idx, lambda *a, **k: self.block_out.clone())

    def _forward(self, options, *, positional=False, state=None, blocks=(1,)):
        """One model forward over ``blocks``; returns which blocks DAVE changed."""
        x = torch.zeros(2, 1, 4, 5, 8)
        emb = torch.zeros(2, 1, 8)
        ctx = torch.zeros(2, 3, 8)
        shared = types.SimpleNamespace(state=state or _State())
        applied = []
        with mock.patch.object(self.pag, "shared", shared):
            for idx in blocks:
                block = self._block(idx)
                if positional:
                    out = block(x, emb, ctx, None, None, None, options)
                else:
                    out = block(x, emb, ctx, transformer_options=options)
                applied.append(not torch.equal(out, self.block_out))
        return applied

    def _run(self, sched, sigmas_now, sampling_steps, offset=0, **kwargs):
        self.pag._DAVE["offset"] = offset  # set by the attach (_forge_sampling_offset)
        steps = []
        for i, sigma in enumerate(sigmas_now):
            options = {"sampling_sigmas": sched, "sigmas": _rows(sigma)}
            state = _State(max(0, i - 1), sampling_steps)  # Forge: one step late
            steps.append(self._forward(options, state=state, **kwargs)[0])
        return steps

    def test_txt2img_steps_match_origin_not_step_fraction(self):
        for steps in (20, 25, 28, 30, 50):
            sched = _simple_sigmas(steps)
            with self.subTest(steps=steps):
                self.assertEqual(self._run(sched, sched[:-1], steps),
                                 _origin_active(0.1, sched, [_rows(s) for s in sched[:-1]]))

    def test_img2img_walks_from_the_attach_offset(self):
        full, sampled, offset, sampling_steps = _forge_img2img(20, 0.5)
        applied = self._run(full, sampled[:-1], sampling_steps, offset)
        self.assertEqual(applied, _origin_active(0.1, sampled, [_rows(s) for s in sampled[:-1]]))
        self.assertEqual(applied, [True] + [False] * (len(sampled) - 2))

    def test_ddim_schedule_type_matches_origin(self):
        # Forge's "DDIM" type returns steps + 2 sigmas here; the gate must not
        # count back from shared.state.sampling_steps.
        for steps in (28, 30):
            sched = _ddim_sigmas(steps)
            with self.subTest(txt2img=steps):
                applied = self._run(sched, sched[:-1], steps, 0)
                self.assertEqual(applied, _origin_active(0.1, sched, [_rows(s) for s in sched[:-1]]))
                self.assertEqual(sum(applied), 3)
        full, sampled, offset, sampling_steps = _forge_img2img(28, 0.5, _ddim_sigmas)
        applied = self._run(full, sampled[:-1], sampling_steps, offset)
        self.assertEqual(applied, _origin_active(0.1, sampled, [_rows(s) for s in sampled[:-1]]))
        self.assertEqual(sum(applied), 2)

    def _run_nested(self, pass_offset, inner_steps, inner_denoise, schedule=_simple_sigmas):
        """A run the script did not attach for: ADetailer's inner img2img from ``postprocess_image``.

        The attach set ``pass_offset`` for the outer request; the inner run brings its own
        request and ``launch_sampling`` count through ``on_cfg_denoiser`` before its blocks run.
        """
        pag = self.pag
        outer_p, inner_p = object(), object()
        pag._DAVE.update(offset=pass_offset, offset_p=outer_p, run_p=None, run_steps=0)
        full, sampled, _offset, sampling_steps = _forge_img2img(inner_steps, inner_denoise, schedule)
        params = types.SimpleNamespace(denoiser=types.SimpleNamespace(p=inner_p, steps=sampling_steps))
        pag._dave_run_callback(params)
        applied = []
        for i, sigma in enumerate(sampled[:-1]):
            options = {"sampling_sigmas": full, "sigmas": _rows(sigma)}
            applied.append(self._forward(options, state=_State(max(0, i - 1), sampling_steps))[0])
        return applied, _origin_active(0.1, sampled, [_rows(s) for s in sampled[:-1]])

    def test_adetailer_inner_run_walks_its_own_tail(self):
        # What the outer pass's attach left: txt2img 0, an img2img/hires start, unknown (None).
        for pass_offset in (0, 27, None):
            for inner_steps, inner_denoise in ((28, 0.4), (20, 0.4), (28, 0.3)):
                with self.subTest(pass_offset=pass_offset, inner=(inner_steps, inner_denoise)):
                    applied, origin = self._run_nested(pass_offset, inner_steps, inner_denoise)
                    self.assertEqual(applied, origin)
                    self.assertTrue(applied[0], "the original runs DAVE on the inner run's first step")

    def test_the_pass_own_run_keeps_the_attach_offset(self):
        # Forge's "DDIM" type returns steps + 2 sigmas: counting back would miss the first one,
        # so the pass's own sampling (denoiser.p is the attached p) keeps the attach offset.
        pag = self.pag
        outer_p = object()
        full, sampled, offset, sampling_steps = _forge_img2img(28, 0.5, _ddim_sigmas)
        pag._DAVE.update(offset=offset, offset_p=outer_p, run_p=None, run_steps=0)
        pag._dave_run_callback(types.SimpleNamespace(
            denoiser=types.SimpleNamespace(p=outer_p, steps=sampling_steps)))
        self.assertEqual(pag._dave_offset(full), offset)
        self.assertNotEqual(len(full) - (sampling_steps + 1), offset)
        applied = self._run(full, sampled[:-1], sampling_steps, offset)
        self.assertEqual(applied, _origin_active(0.1, sampled, [_rows(s) for s in sampled[:-1]]))

    def test_run_callback_does_nothing_while_dave_is_off(self):
        pag = self.pag
        pag._DAVE.update(on=False, run_p=None, run_steps=0)
        pag._dave_run_callback(types.SimpleNamespace(
            denoiser=types.SimpleNamespace(p=object(), steps=12)))
        self.assertIsNone(pag._DAVE["run_p"])
        self.assertEqual(pag._DAVE["run_steps"], 0)

    def test_detail_daemon_scaled_sigma_is_looked_up_before_scaling(self):
        # Detail Daemon hands the model sigma * (1 - adj*cfg); that value is on no schedule, so the
        # original gate calls it step 0 and DAVE runs every step (tau=1.0 look). The pre-DD option
        # (default on) looks up the sigma Detail Daemon noted before scaling it.
        from sam3ext.guidance.dave_gate import note_pre_dd_sigma
        self.addCleanup(note_pre_dd_sigma, None)
        sched = _simple_sigmas(20)

        def run(pre_dd):
            self.pag._DAVE.update(pre_dd=pre_dd, gate=dave_gate.ForwardGateCache(), offset=0)
            applied = []
            for i, sigma in enumerate(sched[:-1]):
                note_pre_dd_sigma(_rows(sigma), _rows(float(sigma) * 0.97))
                options = {"sampling_sigmas": sched, "sigmas": _rows(float(sigma) * 0.97)}
                applied.append(self._forward(options, state=_State(max(0, i - 1), 20))[0])
            return applied

        self.assertEqual(run(True), _origin_active(0.1, sched, [_rows(s) for s in sched[:-1]]))
        self.assertEqual(run(False), [True] * 20)   # the originals chained

    def test_a_note_from_another_forward_is_ignored(self):
        from sam3ext.guidance.dave_gate import note_pre_dd_sigma
        self.addCleanup(note_pre_dd_sigma, None)
        sched = _simple_sigmas(20)
        # Stale: Detail Daemon scaled step 0 earlier; this forward (step 10, unscaled) carries its own sigma.
        note_pre_dd_sigma(_rows(sched[0]), _rows(float(sched[0]) * 0.97))
        options = {"sampling_sigmas": sched, "sigmas": _rows(sched[10])}
        self.assertEqual(self._forward(options, state=_State(10, 20)), [False])

    def test_unknown_offset_uses_the_whole_list(self):
        sched = _simple_sigmas(25)
        applied = self._run(sched, sched[:-1], 25, None)
        self.assertEqual(applied, _origin_active(0.1, sched, [_rows(s) for s in sched[:-1]]))

    def test_midpoint_sigma_is_on(self):
        sched = _simple_sigmas(30)
        mid = (sched[10] + sched[11]) / 2
        options = {"sampling_sigmas": sched, "sigmas": _rows(mid)}
        self.assertEqual(self._forward(options, state=_State(10, 30)), [True])
        options = {"sampling_sigmas": sched, "sigmas": _rows(sched[10])}
        self.assertEqual(self._forward(options, state=_State(10, 30)), [False])

    def test_positional_transformer_options(self):
        sched = _simple_sigmas(25)
        for i, expected in ((0, True), (1, True), (2, False)):
            options = {"sampling_sigmas": sched, "sigmas": _rows(sched[i])}
            with self.subTest(step=i):
                self.assertEqual(self._forward(options, positional=True, state=_State(0, 25)),
                                 [expected])

    def test_one_lookup_per_forward_shared_by_the_pooled_blocks(self):
        sched = _simple_sigmas(25)
        gate = self.pag._DAVE["gate"]
        for i in range(4):
            options = {"sampling_sigmas": sched, "sigmas": _rows(sched[i])}
            applied = self._forward(options, state=_State(0, 25), blocks=(0, 1, 2, 3, 4))
            with self.subTest(step=i):
                on = i < 2
                self.assertEqual(applied, [False, on, on, on, False])
                self.assertEqual(gate.lookups, i + 1)
        self.assertEqual(self.pag._DAVE["steps"], 2 * len(self.TARGETS))

    def test_strength_is_the_applied_attenuation(self):
        sched = _simple_sigmas(20)
        options = {"sampling_sigmas": sched, "sigmas": _rows(sched[0])}
        with mock.patch.object(self.pag, "shared", types.SimpleNamespace(state=_State(0, 20))):
            out = self._block(1)(torch.zeros(1), transformer_options=options)
        # origin: sorryhyun/ComfyUI-Anima-DAVE@83143e8d:nodes.py:84-86 (_dc_hook)
        dims = tuple(range(1, self.block_out.ndim - 1))
        mu = self.block_out.float().mean(dim=dims, keepdim=True)
        self.assertTrue(torch.equal(out, (self.block_out.float() - 0.3 * mu).to(self.block_out.dtype)))

    def test_timestep_sampler_uses_forge_position_not_a_stale_list(self):
        p = self.pag
        p._DAVE["schedule_ok"] = False
        stale = _simple_sigmas(8)
        applied = []
        for step in range(20):
            # A list left by an earlier k-diffusion run; the sigma happens to match step 0.
            options = {"sampling_sigmas": stale, "sigmas": _rows(stale[0])}
            applied.append(self._forward(options, state=_State(step, 20))[0])
        # k = round(0.1 · 20) = 2 on Forge's (lagging) step position.
        self.assertEqual(applied, [True, True] + [False] * 18)

    def test_tau_zero_runs_every_step(self):
        self.pag._DAVE["tau"] = 0.0
        sched = _simple_sigmas(20)
        self.assertEqual(self._run(sched, sched[:-1], 20), [True] * 20)

    def test_sampler_publishes_sigmas(self):
        p = self.pag
        timesteps = types.SimpleNamespace(
            sampler=types.SimpleNamespace(
                model_wrap_cfg=types.SimpleNamespace(classic_ddim_eps_estimation=True)))
        kdiff = types.SimpleNamespace(
            sampler=types.SimpleNamespace(
                model_wrap_cfg=types.SimpleNamespace(classic_ddim_eps_estimation=False)))
        self.assertFalse(p._sampler_publishes_sigmas(timesteps))
        self.assertTrue(p._sampler_publishes_sigmas(kdiff))
        self.assertTrue(p._sampler_publishes_sigmas(types.SimpleNamespace()))


# ---------------------------------------------------------------------------
# 4) Attach: empty block field, strength threshold, sampler kind
# ---------------------------------------------------------------------------


class DaveAttachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _BASE._load_pag_module()
        cls.anima = _BASE._load_real_forge_anima([])

    def _process(self, *, blocks="8-18", strength=0.3, tau=0.1, sampler=None,
                 request_cls=types.SimpleNamespace, fix_steps=None, **fields):
        """process_before_every_sampling with only DAVE on; returns (_DAVE copy, infotext).

        ``fix_steps`` (not None) puts Forge's ``setup_img2img_steps`` behind
        ``modules.sd_samplers_common`` for the attach-time offset."""
        p = self.pag
        torch.manual_seed(0)
        dit = self.anima.Anima(
            in_channels=4, out_channels=4, patch_spatial=2, patch_temporal=1,
            model_channels=48, crossattn_emb_channels=32, adaln_lora_dim=16,
            num_blocks=28, num_heads=4,
        )

        class DummyUnet:
            def __init__(self):
                self.model = types.SimpleNamespace(diffusion_model=dit, predictor=None)
                self.model_options = {}

            def clone(self):
                return DummyUnet()

            def set_model_unet_function_wrapper(self, fn):
                self.model_options["model_function_wrapper"] = fn

            def set_model_sampler_pre_cfg_function(self, fn):
                pass

            def set_model_sampler_post_cfg_function(self, fn):
                pass

        model = type("Anima", (), {})()
        model.forge_objects = types.SimpleNamespace(unet=DummyUnet())
        request = request_cls(
            sd_model=model, extra_generation_params={}, steps=20, is_hr_pass=False,
        )
        for key, value in fields.items():
            setattr(request, key, value)
        if sampler is not None:
            request.sampler = sampler
        process = p.AnimaSafePAG()
        with gr.Blocks():
            inputs = process.ui(False)
        args = [component.value for component in inputs]
        self.assertEqual(inputs[34].elem_id, "anima_guidance_dave_blocks")
        args[0] = False
        args[31], args[32], args[33], args[34] = True, strength, tau, blocks
        forge_modules = {} if fix_steps is None else {"modules": _forge_modules_stub(fix_steps)}
        try:
            with mock.patch.object(p, "shared", types.SimpleNamespace(opts=types.SimpleNamespace())),                     mock.patch.dict(sys.modules, forge_modules):
                process.process_before_every_sampling(request, *args)
            return dict(p._DAVE), dict(request.extra_generation_params)
        finally:
            process.postprocess(request, None)
            p._teardown_global_patches()

    def test_ui_defaults_are_upstream(self):
        with gr.Blocks():
            inputs = self.pag.AnimaSafePAG().ui(False)
        strength, tau, blocks = inputs[32], inputs[33], inputs[34]
        self.assertEqual((strength.value, strength.minimum, strength.maximum, strength.step),
                         tuple(_ORIGIN_STRENGTH[k] for k in ("default", "min", "max", "step")))
        self.assertEqual((tau.value, tau.minimum, tau.maximum, tau.step),
                         tuple(_ORIGIN_TAU[k] for k in ("default", "min", "max", "step")))
        self.assertEqual(blocks.value, dave_gate.DEFAULT_BLOCKS)

    def test_empty_block_field_is_8_to_18(self):
        pooled = {i for i, _a in _origin_pooled(0.3, _ORIGIN_WEIGHT)}
        for spec in ("", "   ", "8-18"):
            with self.subTest(spec=spec):
                state, params = self._process(blocks=spec)
                self.assertTrue(state["on"])
                self.assertEqual(state["targets"], pooled)
                self.assertIn(f"blocks={sorted(pooled)}", params["Anima DAVE"])
        state, _params = self._process(blocks="18")
        self.assertEqual(state["targets"], {18})

    def test_strength_threshold_is_1e_3(self):
        for strength, on in ((0.0, False), (0.001, False), (0.0011, True), (0.3, True)):
            with self.subTest(strength=strength):
                state, params = self._process(strength=strength)
                self.assertIs(state["on"], on)
                self.assertIs(state["on"], bool(_origin_pooled(strength, _ORIGIN_WEIGHT)))
                self.assertIs("Anima DAVE" in params, on)

    def test_sampler_kind_selects_the_gate_source(self):
        timesteps = types.SimpleNamespace(
            model_wrap_cfg=types.SimpleNamespace(classic_ddim_eps_estimation=True))
        state, _params = self._process(sampler=timesteps)
        self.assertFalse(state["schedule_ok"])
        state, _params = self._process()
        self.assertTrue(state["schedule_ok"])
        self.assertIsInstance(state["gate"], dave_gate.ForwardGateCache)

    def test_attach_records_the_walked_offset(self):
        state, _params = self._process()
        self.assertEqual(state["offset"], 0)  # txt2img first pass
        state, _params = self._process(
            request_cls=StableDiffusionProcessingImg2Img, fix_steps=False,
            steps=28, denoising_strength=0.5)
        self.assertEqual(state["offset"], _forge_img2img(28, 0.5)[2])
        state, _params = self._process(
            fix_steps=False, is_hr_pass=True, hr_second_pass_steps=0,
            steps=28, denoising_strength=0.5)
        self.assertEqual(state["offset"], _forge_img2img(28, 0.5, requested=28)[2])


# ---------------------------------------------------------------------------
# 5) Attach-time offset: Forge's own setup_img2img_steps, per pass kind
# ---------------------------------------------------------------------------


class StableDiffusionProcessingImg2Img(types.SimpleNamespace):
    """Stands in for modules.processing.StableDiffusionProcessingImg2Img (matched by name)."""


def _forge_modules_stub(fix_steps):
    stub = types.ModuleType("modules")
    stub.sd_samplers_common = types.SimpleNamespace(
        setup_img2img_steps=_setup_img2img_steps(fix_steps))
    return stub


class ForgeSamplingOffsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _BASE._load_pag_module()

    def _offset(self, request, fix_steps=False):
        with mock.patch.dict(sys.modules, {"modules": _forge_modules_stub(fix_steps)}):
            return self.pag._forge_sampling_offset(request)

    def test_txt2img_first_pass_walks_the_whole_list(self):
        request = types.SimpleNamespace(steps=28, denoising_strength=0.7, is_hr_pass=False)
        self.assertEqual(self._offset(request), 0)

    def test_img2img_uses_setup_img2img_steps_without_steps(self):
        # processing.py:1920 — sample_img2img(..., steps=None)
        for steps, denoise in ((28, 0.5), (30, 0.6), (24, 0.75), (20, 1.0)):
            for fix_steps in (False, True):
                request = StableDiffusionProcessingImg2Img(
                    steps=steps, denoising_strength=denoise, is_hr_pass=False)
                with self.subTest(steps=steps, denoise=denoise, fix_steps=fix_steps):
                    self.assertEqual(self._offset(request, fix_steps),
                                     _forge_img2img(steps, denoise, fix_steps=fix_steps)[2])

    def test_img2img_subclass_counts(self):
        class Inpaint(StableDiffusionProcessingImg2Img):
            pass

        request = Inpaint(steps=28, denoising_strength=0.5, is_hr_pass=False)
        self.assertEqual(self._offset(request), 13)

    def test_hires_passes_second_pass_steps_or_steps(self):
        # processing.py:1552 — steps=self.hr_second_pass_steps or self.steps
        for hr_steps, requested in ((0, 28), (10, 10)):
            request = types.SimpleNamespace(
                steps=28, hr_second_pass_steps=hr_steps, denoising_strength=0.5, is_hr_pass=True)
            with self.subTest(hr_second_pass_steps=hr_steps):
                self.assertEqual(self._offset(request),
                                 _forge_img2img(28, 0.5, requested=requested)[2])

    def test_without_forge_the_offset_is_unknown(self):
        request = StableDiffusionProcessingImg2Img(
            steps=28, denoising_strength=0.5, is_hr_pass=False)
        with mock.patch.dict(sys.modules, {"modules": types.ModuleType("modules")}):
            self.assertIsNone(self.pag._forge_sampling_offset(request))


if __name__ == "__main__":
    unittest.main()
