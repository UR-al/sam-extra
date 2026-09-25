"""Origin-parity tests for the Anima Skimmed CFG script.

The oracle is the upstream ComfyUI node itself: its maths and its
``CFG_Skimming_Single_Scale_Pre_CFG`` patch are copied below (Apache-2.0,
Extraltodeus/Skimmed_CFG), together with the ComfyUI and Forge code that runs
around a pre-CFG patch. Each copy carries an ``origin:`` comment with the
repository, commit, file and lines it was taken from.

The Forge hook must give the same predictions and the same CFG result as
upstream's patch followed by ComfyUI's ``cfg_function``, bit for bit, on every
step of a real Anima (flow, shift 3) schedule.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import io
import math
import sys
import textwrap
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "anima_skimmed_cfg.py"


def _load_skim_module():
    """Load the extension script without booting the full WebUI."""
    modules_stub = types.ModuleType("modules")

    class Script:
        pass

    modules_stub.scripts = types.SimpleNamespace(
        Script=Script,
        AlwaysVisible=object(),
        scripts_data=[],
    )
    modules_stub.shared = types.SimpleNamespace(
        state=types.SimpleNamespace(sampling_step=0, sampling_steps=20)
    )
    modules_stub.script_callbacks = types.SimpleNamespace(
        on_before_ui=lambda fn: None
    )

    old_modules = sys.modules.get("modules")
    sys.modules["modules"] = modules_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "_test_anima_skimmed_cfg", SCRIPT
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if old_modules is None:
            sys.modules.pop("modules", None)
        else:
            sys.modules["modules"] = old_modules


# ---------------------------------------------------------------------------
# Upstream oracle (Apache-2.0).
# origin: Extraltodeus/Skimmed_CFG@d83005832ac42783adfd6f4ae96f6ef6406d1a74:skimmed_CFG.py:9-54
# Kept as text so the test can also prove the extension's copy is unmodified.
# ---------------------------------------------------------------------------

_ORIGIN_MATHS = '''
@torch.no_grad()
def get_skimming_mask(
    x_orig,
    cond,
    uncond,
    cond_scale,
    return_denoised=False,
    disable_flipping_filter=False,
):
    denoised = x_orig - (
        (x_orig - uncond) + cond_scale * ((x_orig - cond) - (x_orig - uncond))
    )
    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    matching_diff_after = (
        cond.sign() == (cond * cond_scale - uncond * (cond_scale - 1)).sign()
    )

    if disable_flipping_filter:
        outer_influence = matching_pred_signs & matching_diff_after
    else:
        deviation_influence = denoised.sign() == (denoised - x_orig).sign()
        outer_influence = (
            matching_pred_signs & matching_diff_after & deviation_influence
        )

    if return_denoised:
        return outer_influence, denoised
    else:
        return outer_influence


@torch.no_grad()
def skimmed_CFG(
    x_orig, cond, uncond, cond_scale, skimming_scale, disable_flipping_filter=False
):
    outer_influence, denoised = get_skimming_mask(
        x_orig, cond, uncond, cond_scale, True, disable_flipping_filter
    )
    low_cfg_denoised_outer = x_orig - (
        (x_orig - uncond) + skimming_scale * ((x_orig - cond) - (x_orig - uncond))
    )
    low_cfg_denoised_outer_difference = denoised - low_cfg_denoised_outer
    cond[outer_influence] = cond[outer_influence] - (
        low_cfg_denoised_outer_difference[outer_influence] / cond_scale
    )
    return cond
'''

_ORIGIN = {"torch": torch}
exec(compile(_ORIGIN_MATHS, "<Skimmed_CFG skimmed_CFG.py:9-54>", "exec"), _ORIGIN)


def _origin_execute(model_sampling, skimming_cfg, full_skim_negative,
                    disable_flipping_filter, start_at_percentage,
                    end_at_percentage, flip_at_percentage):
    """``CFG_Skimming_Single_Scale_Pre_CFG.execute`` minus the model clone.

    origin: Extraltodeus/Skimmed_CFG@d83005832ac42783adfd6f4ae96f6ef6406d1a74:skimmed_CFG.py:141-197
    """
    skimmed_CFG = _ORIGIN["skimmed_CFG"]
    start_at_sigma = model_sampling.percent_to_sigma(start_at_percentage)
    end_at_sigma = model_sampling.percent_to_sigma(end_at_percentage)
    flip_at_sigma = model_sampling.percent_to_sigma(flip_at_percentage)

    @torch.no_grad()
    def pre_cfg_patch(args):
        conds_out = args["conds_out"]
        cond_scale = args["cond_scale"]
        x_orig = args["input"]
        sigma = args["sigma"][0].item()

        # Fix: Use >= instead of > for proper boundary checking
        if (
            not torch.any(conds_out[1])
            or sigma <= end_at_sigma
            or sigma >= start_at_sigma
        ):
            return conds_out

        practical_scale = cond_scale if skimming_cfg < 0 else skimming_cfg

        flip_filter = disable_flipping_filter
        if flip_at_percentage > 0 and sigma > flip_at_sigma:
            flip_filter = not disable_flipping_filter

        conds_out[1] = skimmed_CFG(
            x_orig,
            conds_out[1],
            conds_out[0],
            cond_scale,
            practical_scale if not full_skim_negative else 0,
            flip_filter,
        )
        conds_out[0] = skimmed_CFG(
            x_orig,
            conds_out[0],
            conds_out[1],
            cond_scale - 1,
            practical_scale,
            flip_filter,
        )
        return conds_out

    return pre_cfg_patch


# origin: Extraltodeus/Skimmed_CFG@d83005832ac42783adfd6f4ae96f6ef6406d1a74:skimmed_CFG.py:5-6,94-133
_ORIGIN_MAX_SCALE = 10
_ORIGIN_STEP_STEP = 2
_ORIGIN_INPUTS = {
    "skimming_cfg": dict(default=7.0, min=0.0, max=_ORIGIN_MAX_SCALE,
                         step=1.0 / _ORIGIN_STEP_STEP),
    "start_at_percentage": dict(default=0.0, min=0.0, max=1.0, step=0.01),
    "end_at_percentage": dict(default=1.0, min=0.0, max=1.0, step=0.01),
    "flip_at_percentage": dict(default=0.0, min=0.0, max=1.0, step=0.01),
}


# ---------------------------------------------------------------------------
# Host code around the patch.
# ---------------------------------------------------------------------------


def _time_snr_shift(alpha, t):
    # origin: Comfy-Org/ComfyUI@387f98aa2822f684b8597959a52a467d88cc4806:comfy/model_sampling.py:289-292
    # (same body in Haoming02/sd-webui-forge-classic@e33f40e4:backend/modules/k_prediction.py:32-35)
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


class _FlowSampling:
    """Anima's flow sampling (shift 3, multiplier 1).

    ComfyUI ``ModelSamplingDiscreteFlow`` and Forge ``PredictionDiscreteFlow``
    share this ``percent_to_sigma``.
    """

    def __init__(self, shift=3.0, multiplier=1.0, timesteps=1000):
        self.shift = shift
        self.multiplier = multiplier
        # origin: Comfy-Org/ComfyUI@387f98aa:comfy/model_sampling.py:307-311 (set_parameters)
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier)
        self.sigmas = ts

    def sigma(self, timestep):
        # origin: Comfy-Org/ComfyUI@387f98aa:comfy/model_sampling.py:327-328
        return _time_snr_shift(self.shift, timestep / self.multiplier)

    def percent_to_sigma(self, percent):
        # origin: Comfy-Org/ComfyUI@387f98aa:comfy/model_sampling.py:331-336
        # (= Haoming02/sd-webui-forge-classic@e33f40e4:backend/modules/k_prediction.py:212-217)
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return _time_snr_shift(self.shift, 1.0 - percent)


class _SkimCheckSampling(_FlowSampling):
    """The sigma table ``origin_parity/skim_check.py`` used for the plan's numbers.

    It evaluates the shift in Python doubles and stores float32, where ComfyUI
    evaluates it in float32. The two tables differ in the last bit of most
    entries; for the gates below that matters only at simple step 9, whose
    sigma is the flip point itself (t = 0.7 -> 0.875): 0.875 here, 0.87499994
    in ComfyUI's table.
    """

    def __init__(self):
        super().__init__()
        self.sigmas = torch.tensor(
            [_time_snr_shift(self.shift, i / 1000.0) for i in range(1, 1001)],
            dtype=torch.float32,
        )


class _EpsSampling:
    """Only the saturating ends of a discrete (eps/v) schedule."""

    def percent_to_sigma(self, percent):
        # origin: Haoming02/sd-webui-forge-classic@e33f40e4:backend/modules/k_prediction.py:142-148
        # (ComfyUI ModelSamplingDiscrete has the same ends)
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        return 1.0  # not reached by the tests below


def _simple_sigmas(model_sampling, steps):
    # origin: Comfy-Org/ComfyUI@387f98aa:comfy/samplers.py:645-652 (simple_scheduler)
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def _beta_sigmas(model_sampling, steps, alpha=0.6, beta=0.6):
    # origin: Comfy-Org/ComfyUI@387f98aa:comfy/samplers.py:696-708 (beta_scheduler)
    import numpy
    import scipy.stats

    total_timesteps = (len(model_sampling.sigmas) - 1)
    ts = 1 - numpy.linspace(0, 1, steps, endpoint=False)
    ts = numpy.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps)

    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs += [float(model_sampling.sigmas[int(t)])]
        last_t = t
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def _karras_sigmas(model_sampling, steps, rho=7.0):
    # origin: Comfy-Org/ComfyUI@387f98aa:comfy/k_diffusion/sampling.py:23-29 (get_sigmas_karras,
    # called with sigma_min/sigma_max = model_sampling.sigmas[0]/[-1], comfy/samplers.py:1368)
    sigma_min = float(model_sampling.sigmas[0])
    sigma_max = float(model_sampling.sigmas[-1])
    ramp = torch.linspace(0, 1, steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def _comfy_step(patch, x, cond_pred, uncond_pred, cond_scale, timestep,
                model_options):
    """ComfyUI: pre-CFG patches, then ``cfg_function`` (no post-CFG).

    origin: Comfy-Org/ComfyUI@387f98aa:comfy/samplers.py:621-626 (pre-CFG loop)
            and :592-598 (cfg_function)
    """
    out = [cond_pred.clone(), uncond_pred.clone()]
    args = {"conds": None, "conds_out": out, "cond_scale": cond_scale,
            "timestep": timestep, "input": x, "sigma": timestep, "model": None,
            "model_options": model_options}
    out = patch(args)
    cond_pred, uncond_pred = out[0], out[1]
    if "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred,
                "cond_scale": cond_scale, "timestep": timestep, "input": x,
                "sigma": timestep, "cond_denoised": cond_pred,
                "uncond_denoised": uncond_pred, "model": None,
                "model_options": model_options, "input_cond": None,
                "input_uncond": None}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale
    return cfg_result, cond_pred, uncond_pred


def _forge_step(model, x, cond_pred, uncond_pred, cond_scale, timestep,
                model_options, cond=None):
    """Forge: CFG step, then the post-CFG list (where the skim hook sits).

    origin: Haoming02/sd-webui-forge-classic@e33f40e4:backend/sampling/sampling_function.py:293,308-318
    Returns the final result and the prediction tensors later hooks would see.
    """
    cond = [{}] if cond is None else cond
    cond_pred = cond_pred.clone()
    uncond_pred = uncond_pred.clone()
    edit_strength = sum((item["strength"] if "strength" in item else 1) for item in cond)
    if "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep, "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    elif not math.isclose(edit_strength, 1.0):
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale * edit_strength
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": None, "cond_scale": cond_scale, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred, "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)
    return cfg_result, cond_pred, uncond_pred


def _rescale_cfg(multiplier, is_flow):
    """ComfyUI RescaleCFG's ``sampler_cfg_function``.

    origin: Comfy-Org/ComfyUI@387f98aa:comfy_extras/nodes_model_advanced.py:291-324
    """

    def rescale_cfg(args):
        x_orig = args["input"]
        cond_scale = args["cond_scale"]

        if is_flow:
            x_0_cond = args["cond_denoised"]
            x_0_uncond = args["uncond_denoised"]
            x_0_cfg = x_0_uncond + cond_scale * (x_0_cond - x_0_uncond)
            dims = tuple(range(1, x_0_cond.ndim))
            ro_pos = x_0_cond.std(dim=dims, keepdim=True)
            ro_cfg = x_0_cfg.std(dim=dims, keepdim=True).clamp(min=1e-8)
            x_0_rescaled = x_0_cfg * (ro_pos / ro_cfg)
            x_0_final = multiplier * x_0_rescaled + (1.0 - multiplier) * x_0_cfg
            return x_orig - x_0_final

        cond = args["cond"]
        uncond = args["uncond"]
        sigma = args["sigma"]
        sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))

        x = x_orig / (sigma * sigma + 1.0)
        cond = ((x - (x_orig - cond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)
        uncond = ((x - (x_orig - uncond)) * (sigma ** 2 + 1.0) ** 0.5) / (sigma)

        x_cfg = uncond + cond_scale * (cond - uncond)
        ro_pos = torch.std(cond, dim=(1, 2, 3), keepdim=True)
        ro_cfg = torch.std(x_cfg, dim=(1, 2, 3), keepdim=True)

        x_rescaled = x_cfg * (ro_pos / ro_cfg)
        x_final = multiplier * x_rescaled + (1.0 - multiplier) * x_cfg

        return x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)

    return rescale_cfg


def _pattern(bits):
    return "".join("#" if b else "." for b in bits)


# Upstream gate tables, Anima flow shift 3, 30 steps: which steps the upstream
# node skims ("apply") and, of those, which run with the filter flipped.
# Generated by origin_parity/skim_check.py (upstream gate on the ComfyUI
# simple/beta/karras schedules, sigma table of _SkimCheckSampling).
_GOLDEN = {
    "simple": {
        (0.0, 1.0, 0.0): (".#############################", ".............................."),
        (0.0, 1.0, 0.3): (".#############################", ".#########...................."),
        (0.2, 0.8, 0.0): (".......##################.....", ".............................."),
    },
    "beta": {
        (0.0, 1.0, 0.0): (".#############################", ".............................."),
        (0.0, 1.0, 0.3): (".#############################", ".##########..................."),
        (0.2, 0.8, 0.0): (".........#############........", ".............................."),
    },
    "karras": {
        (0.0, 1.0, 0.0): (".#############################", ".............................."),
        (0.0, 1.0, 0.3): (".#############################", ".............................."),
        (0.2, 0.8, 0.0): (".#####........................", ".............................."),
    },
}


class SkimmedCFGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skim = _load_skim_module()
        from sam3ext.guidance import skimmed_cfg as vendored

        cls.vendored = vendored
        cls.flow = _FlowSampling()
        cls.model = types.SimpleNamespace(predictor=cls.flow)

    def setUp(self):
        # The sigma-window line is logged once per pass; tests start hundreds.
        real_log = self.skim._log
        self.skim._log = lambda message: None
        self.addCleanup(setattr, self.skim, "_log", real_log)
        self._real_log = real_log
        torch.manual_seed(0)
        self.x = torch.randn(1, 4, 8, 8)
        self.cond = torch.randn(1, 4, 8, 8)
        self.uncond = torch.randn(1, 4, 8, 8)
        self._configure()

    def _configure(self, on=False, **overrides):
        state = dict(
            on=on, skimming_cfg=7.0, full_skim_negative=False,
            disable_flipping_filter=False, start=0.0, end=1.0, flip_at=0.0,
            steps=0, warned=False, sigmas=None,
        )
        state.update(overrides)
        self.skim._SKIM.update(state)

    def _args(self, cond_scale=8.0, sigma=0.5, model=None, **extra):
        denoised = self.uncond + (self.cond - self.uncond) * cond_scale
        args = {
            "denoised": denoised,
            "cond": [{}],
            "uncond": None,
            "input": self.x,
            "cond_denoised": self.cond,
            "uncond_denoised": self.uncond,
            "cond_scale": cond_scale,
            "sigma": torch.tensor([sigma]),
            "model": self.model if model is None else model,
            "model_options": {},
        }
        args.update(extra)
        return args

    def _hook_options(self, **extra):
        return {"sampler_post_cfg_function": [self.skim._post_cfg], **extra}

    # -- vendored code ------------------------------------------------------

    def test_vendored_maths_is_upstream_source_unmodified(self):
        tree = ast.parse(_ORIGIN_MATHS)
        for node in tree.body:
            with self.subTest(function=node.name):
                ours = ast.parse(textwrap.dedent(
                    inspect.getsource(getattr(self.vendored, node.name))
                )).body[0]
                self.assertEqual(ast.dump(ours), ast.dump(node))

    def test_broken_vendored_import_fails_the_script_load(self):
        """A missing vendored module must not turn the hook into a silent no-op.

        It used to share ``import torch``'s try/except, so any import error set
        ``torch = None``: the checkbox stayed, nothing ran, nothing was logged.
        """
        name = "sam3ext.guidance.skimmed_cfg"
        saved = sys.modules.get(name)

        def restore():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

        self.addCleanup(restore)
        sys.modules[name] = None  # makes ``from ... import`` raise ImportError
        with self.assertRaises(ImportError):
            _load_skim_module()

    def test_vendored_maths_matches_upstream_bitwise(self):
        for seed in range(4):
            g = torch.Generator().manual_seed(seed)
            x, cond, uncond = (torch.randn(2, 16, 1, 12, 10, generator=g) for _ in range(3))
            for scale in (8.0, 4.5, 1.5, 0.5, -2.0):
                for skimming in (7.0, 3.0, 0.0, 12.0):
                    for flip in (False, True):
                        with self.subTest(seed=seed, scale=scale, skim=skimming, flip=flip):
                            ours = self.vendored.skimmed_CFG(
                                x, cond.clone(), uncond, scale, skimming, flip)
                            ref = _ORIGIN["skimmed_CFG"](
                                x, cond.clone(), uncond, scale, skimming, flip)
                            self.assertTrue(torch.equal(ours, ref))

    # -- whole step vs upstream --------------------------------------------

    def _assert_bitwise(self, ours, ref):
        """``torch.equal`` that also counts NaN == NaN (same positions).

        inf compares equal to inf of the same sign, so the NaN mask plus
        ``torch.equal`` on the rest is exact for every non-finite value.
        """
        self.assertEqual(ours.dtype, ref.dtype)
        self.assertEqual(ours.shape, ref.shape)
        nan = torch.isnan(ours)
        self.assertTrue(torch.equal(nan, torch.isnan(ref)))
        zero = torch.zeros((), dtype=ours.dtype)
        self.assertTrue(torch.equal(torch.where(nan, zero, ours),
                                    torch.where(nan, zero, ref)))

    def _assert_step_matches_upstream(self, *, config, sigma, cond_scale,
                                      cfg_function=None, dtype=torch.float32,
                                      model=None, sampling=None, poison=None):
        model = self.model if model is None else model
        sampling = self.flow if sampling is None else sampling
        self._configure(on=True, **config)
        g = torch.Generator().manual_seed(11)
        shape = (2, 16, 1, 16, 16)
        x = torch.randn(shape, generator=g).to(dtype)
        cond = torch.randn(shape, generator=g).to(dtype)
        uncond = (cond + 0.4 * torch.randn(shape, generator=g)).to(dtype)
        if poison is not None:
            poison(x, cond, uncond)
        timestep = torch.tensor([sigma, sigma])

        extra = {} if cfg_function is None else {"sampler_cfg_function": cfg_function}
        patch = _origin_execute(
            sampling, config.get("skimming_cfg", 7.0),
            config.get("full_skim_negative", False),
            config.get("disable_flipping_filter", False),
            config.get("start", 0.0), config.get("end", 1.0),
            config.get("flip_at", 0.0),
        )
        ref, ref_c, ref_u = _comfy_step(patch, x, cond, uncond, cond_scale,
                                        timestep, dict(extra))
        out, c, u = _forge_step(model, x, cond, uncond, cond_scale, timestep,
                                self._hook_options(**extra))
        self._assert_bitwise(out, ref)
        self._assert_bitwise(c, ref_c)
        self._assert_bitwise(u, ref_u)
        return out, x, cond, uncond

    def test_step_matches_upstream_node_bitwise(self):
        configs = (
            dict(skimming_cfg=7.0),
            dict(skimming_cfg=3.0),
            dict(skimming_cfg=0.0),
            dict(skimming_cfg=-1.0),
            dict(skimming_cfg=3.0, full_skim_negative=True),
            dict(skimming_cfg=2.0, disable_flipping_filter=True),
            dict(skimming_cfg=2.0, flip_at=0.3),
            dict(skimming_cfg=-1.0, full_skim_negative=True, flip_at=0.3,
                 disable_flipping_filter=True),
            dict(skimming_cfg=3.0, start=0.2, end=0.8),
        )
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for config in configs:
                for sigma in (1.0, 0.97, 0.9, 0.6, 0.3, 0.05):
                    for cond_scale in (8.0, 4.0, 1.5):
                        with self.subTest(dtype=dtype, config=config,
                                          sigma=sigma, cond_scale=cond_scale):
                            self._assert_step_matches_upstream(
                                config=config, sigma=sigma,
                                cond_scale=cond_scale, dtype=dtype,
                            )

    def test_rescale_cfg_is_applied_to_the_skimmed_predictions(self):
        """A registered sampler_cfg_function survives, fed the skimmed preds."""
        for is_flow in (True, False):
            with self.subTest(is_flow=is_flow):
                rescale = _rescale_cfg(0.7, is_flow)
                out, x, cond, uncond = self._assert_step_matches_upstream(
                    config=dict(skimming_cfg=3.0), sigma=0.6, cond_scale=8.0,
                    cfg_function=rescale,
                )
                self.assertEqual(self.skim._SKIM["steps"], 1)
                timestep = torch.tensor([0.6, 0.6])
                # The skim changed the result...
                unskimmed, _, _ = _forge_step(
                    self.model, x, cond, uncond, 8.0, timestep,
                    {"sampler_cfg_function": rescale})
                self.assertFalse(torch.allclose(out, unskimmed))
                # ...and RescaleCFG was not replaced by the linear combine.
                self._configure(on=True, skimming_cfg=3.0)
                linear, _, _ = _forge_step(
                    self.model, x, cond, uncond, 8.0, timestep,
                    self._hook_options())
                self.assertFalse(torch.allclose(out, linear))

    def test_forge_edit_strength_is_kept(self):
        """Forge scales the linear combine by the conds' strength; so do we."""
        self._configure(on=True, skimming_cfg=3.0)
        args = self._args(cond=[{"strength": 0.5}])
        cond, uncond = self.cond.clone(), self.uncond.clone()
        out = self.skim._post_cfg(args)
        patch = _origin_execute(self.flow, 3.0, False, False, 0.0, 1.0, 0.0)
        conds_out = patch({"conds_out": [cond, uncond], "cond_scale": 8.0,
                           "input": self.x, "sigma": torch.tensor([0.5])})
        # origin: Haoming02/sd-webui-forge-classic@e33f40e4:backend/sampling/sampling_function.py:311-312
        expected = conds_out[1] + (conds_out[0] - conds_out[1]) * 8.0 * 0.5
        self.assertTrue(torch.equal(out, expected))

    def test_non_finite_values_pass_through_like_upstream(self):
        """No sanitising: inf/NaN come out where upstream puts them.

        Upstream's patch (Extraltodeus/Skimmed_CFG@d8300583:skimmed_CFG.py:160-197)
        and ComfyUI's ``cfg_function`` (Comfy-Org/ComfyUI@387f98aa:comfy/samplers.py:592-598)
        pass non-finite values through; the extension used to zero them with
        ``torch.nan_to_num`` after the CFG step (parity plan 6.2 #8, removed).
        """

        def cond_inf(x, cond, uncond):
            cond[0, 0, 0, 0, 0] = math.inf
            cond[0, 1, 0, 2, 3] = -math.inf

        def cond_nan(x, cond, uncond):
            cond[1, 3, 0, 5, 7] = math.nan

        def uncond_inf_nan(x, cond, uncond):
            uncond[1, 0, 0, 0, 0] = math.inf
            uncond[0, 2, 0, 1, 1] = math.nan

        def input_inf(x, cond, uncond):
            # Finite predictions; the skim itself computes inf - inf there.
            x[0, :, 0, :4, :4] = math.inf

        configs = (
            dict(skimming_cfg=3.0),
            dict(skimming_cfg=3.0, disable_flipping_filter=True),
            dict(skimming_cfg=-1.0, full_skim_negative=True),
        )
        cases = [(poison, config)
                 for poison in (cond_inf, cond_nan, uncond_inf_nan)
                 for config in configs]
        # With the flipping filter on, whether the skim touches an inf-input
        # element depends on torch.sign(NaN) (0 on current torch), so only the
        # filter-off config is guaranteed to carry the NaN into the output.
        cases.append((input_inf, configs[1]))
        cfg_functions = (None, _rescale_cfg(0.7, True))
        for poison, config in cases:
            for cfg_function in cfg_functions:
                with self.subTest(poison=poison.__name__, config=config,
                                  rescale=cfg_function is not None):
                    out, _, _, _ = self._assert_step_matches_upstream(
                        config=config, sigma=0.6, cond_scale=8.0,
                        cfg_function=cfg_function, poison=poison,
                    )
                    # The step was skimmed (the old sanitiser ran only then)
                    # and upstream's result really is non-finite somewhere.
                    self.assertEqual(self.skim._SKIM["steps"], 1)
                    self.assertFalse(bool(torch.isfinite(out).all()))

    # -- gate --------------------------------------------------------------

    def _schedule(self, name, steps=30, sampling=None):
        sampling = self.flow if sampling is None else sampling
        if name == "beta":
            try:
                import scipy.stats  # noqa: F401
            except ImportError:  # pragma: no cover - scipy ships with Forge
                self.skipTest("scipy is not installed")
            return _beta_sigmas(sampling, steps)
        if name == "karras":
            return _karras_sigmas(sampling, steps)
        return _simple_sigmas(sampling, steps)

    def _gate_table(self, sampling, golden):
        """Run every step of each schedule through the hook and the upstream
        oracle (bitwise equal), and compare the skim/flip pattern to golden."""
        counts = {}
        for name, cases in golden.items():
            sigmas = self._schedule(name, sampling=sampling)[:-1]  # no call at 0
            self.assertEqual(len(sigmas), 30)
            for (start, end, flip), (apply_golden, flip_golden) in cases.items():
                with self.subTest(schedule=name, start=start, end=end, flip=flip):
                    applied, flipped = [], []
                    s_start, s_end, s_flip = self.vendored.skim_sigmas(
                        self.flow.percent_to_sigma, start, end, flip)
                    for sigma in sigmas:
                        self._assert_step_matches_upstream(
                            config=dict(skimming_cfg=3.0, start=start, end=end,
                                        flip_at=flip),
                            sigma=float(sigma), cond_scale=8.0,
                        )
                        hit = self.skim._SKIM["steps"] > 0
                        applied.append(hit)
                        s = torch.tensor([float(sigma)])[0].item()
                        flipped.append(hit and self.vendored.flip_filter_at(
                            s, False, flip, s_flip))
                    self.assertEqual(_pattern(applied), apply_golden)
                    self.assertEqual(_pattern(flipped), flip_golden)
                    counts[(name, start, end, flip)] = (sum(applied), sum(flipped))
        return counts

    def test_gate_tables_match_upstream_on_30_step_anima_schedules(self):
        counts = self._gate_table(_SkimCheckSampling(), _GOLDEN)

        # Numbers quoted in the parity plan (origin_parity/skim_check.py).
        for name in ("simple", "beta", "karras"):
            self.assertEqual(counts[(name, 0.0, 1.0, 0.0)][0], 29)
        self.assertEqual(
            [counts[(n, 0.2, 0.8, 0.0)][0] for n in ("simple", "beta", "karras")],
            [18, 13, 5])
        self.assertEqual(
            [counts[(n, 0.0, 1.0, 0.3)][1] for n in ("simple", "beta", "karras")],
            [9, 10, 0])

    def test_gate_tables_on_comfyui_float32_sigma_table(self):
        """Same gate on ComfyUI's own float32 table: identical except simple
        step 9, which lands 6e-8 below the flip sigma and so is not flipped —
        in upstream as well (the step is still compared bitwise)."""
        golden = {name: dict(cases) for name, cases in _GOLDEN.items()}
        golden["simple"][(0.0, 1.0, 0.3)] = (
            ".#############################", ".########.....................")
        self.assertEqual(float(_simple_sigmas(self.flow, 30)[9]), 0.8749999403953552)
        counts = self._gate_table(self.flow, golden)
        self.assertEqual(
            [counts[(n, 0.0, 1.0, 0.3)][1] for n in ("simple", "beta", "karras")],
            [8, 10, 0])

    def test_first_flow_step_is_not_skimmed(self):
        """Upstream quirk: percent_to_sigma(0) == first sigma == 1.0."""
        self._configure(on=True)
        args = self._args(sigma=1.0)
        cond_before = self.cond.clone()
        self.assertIs(self.skim._post_cfg(args), args["denoised"])
        self.assertTrue(torch.equal(self.cond, cond_before))
        self.assertEqual(self.skim._SKIM["steps"], 0)

    def test_first_discrete_step_is_skimmed(self):
        """Eps/v models map 0% to a huge sigma, so upstream skims step 0."""
        self._configure(on=True, skimming_cfg=3.0)
        args = self._args(sigma=14.6146, model=types.SimpleNamespace(
            predictor=_EpsSampling()))
        out = self.skim._post_cfg(args)
        self.assertIsNot(out, args["denoised"])
        self.assertEqual(self.skim._SKIM["steps"], 1)

    def test_window_bounds_are_strict(self):
        """Neither bound is skimmed: 50% is sigma 0.75 exactly at shift 3."""
        mid = self.flow.percent_to_sigma(0.5)
        self.assertEqual(mid, 0.75)
        cases = (
            # (start, end, sigma, skimmed)
            (0.0, 0.5, 0.75, False),
            (0.0, 0.5, 0.7501, True),
            (0.0, 0.5, 0.7499, False),
            (0.0, 0.5, 1.0, False),
            (0.5, 1.0, 0.75, False),
            (0.5, 1.0, 0.7499, True),
            (0.5, 1.0, 0.7501, False),
        )
        for start, end, sigma, expect in cases:
            with self.subTest(start=start, end=end, sigma=sigma):
                self._configure(on=True, start=start, end=end)
                self.skim._post_cfg(self._args(sigma=sigma))
                self.assertEqual(self.skim._SKIM["steps"], int(expect))

    def test_flip_happens_before_the_flip_point(self):
        s_flip = self.flow.percent_to_sigma(0.3)
        self.assertTrue(self.vendored.flip_filter_at(s_flip + 1e-3, False, 0.3, s_flip))
        self.assertFalse(self.vendored.flip_filter_at(s_flip, False, 0.3, s_flip))
        self.assertFalse(self.vendored.flip_filter_at(s_flip - 1e-3, False, 0.3, s_flip))
        self.assertFalse(self.vendored.flip_filter_at(s_flip + 1e-3, True, 0.3, s_flip))
        self.assertFalse(self.vendored.flip_filter_at(0.99, False, 0.0, 1.0))

    def test_reversed_window_skims_nothing(self):
        """No swap: upstream's start 0.8 / end 0.2 never passes its gate."""
        script = self.skim.AnimaSkimmedCFG()
        unet = _FakeUnet()
        p = types.SimpleNamespace(sd_model=types.SimpleNamespace(
            forge_objects=types.SimpleNamespace(unet=unet)))
        script.process_before_every_sampling(p, True, 3.0, False, False, 0.8, 0.2, 0.0)
        self.assertEqual((self.skim._SKIM["start"], self.skim._SKIM["end"]), (0.8, 0.2))
        hook = p.sd_model.forge_objects.unet.model_options["sampler_post_cfg_function"][0]
        self.assertIs(hook, self.skim._post_cfg)
        for sigma in self._schedule("simple")[:-1]:
            self.skim._post_cfg(self._args(sigma=float(sigma)))
        self.assertEqual(self.skim._SKIM["steps"], 0)

    def test_arguments_are_not_clamped(self):
        script = self.skim.AnimaSkimmedCFG()
        p = types.SimpleNamespace(sd_model=types.SimpleNamespace(
            forge_objects=types.SimpleNamespace(unet=_FakeUnet())))
        p._anima_skimmed_cfg_xyz = {"flip_at": "1.5", "start": "-0.2"}
        script.process_before_every_sampling(p, True, -1.0, True, False, 0.0, 1.0, 0.3)
        self.assertEqual(self.skim._SKIM["flip_at"], 1.5)
        self.assertEqual(self.skim._SKIM["start"], -0.2)
        self.assertEqual(self.skim._SKIM["skimming_cfg"], -1.0)
        self.assertIsNone(self.skim._SKIM["sigmas"])

    def test_sigmas_follow_the_sampling_predictor(self):
        self._configure(on=True, start=0.2, end=0.8)
        self.skim._post_cfg(self._args(sigma=0.5))
        cached_predictor, sigmas = self.skim._SKIM["sigmas"]
        self.assertIs(cached_predictor, self.flow)
        self.assertEqual(sigmas, self.vendored.skim_sigmas(
            self.flow.percent_to_sigma, 0.2, 0.8, 0.0))
        other = _FlowSampling(shift=1.0)
        self.skim._post_cfg(self._args(
            sigma=0.5, model=types.SimpleNamespace(predictor=other)))
        self.assertIs(self.skim._SKIM["sigmas"][0], other)

    def test_missing_predictor_skips(self):
        self._configure(on=True)
        args = self._args(model=types.SimpleNamespace())
        self.assertIs(self.skim._post_cfg(args), args["denoised"])
        self.assertTrue(self.skim._SKIM["warned"])

    # -- CFG 1 and missing negatives ----------------------------------------

    def test_zero_negative_prediction_is_left_alone(self):
        """Upstream: ``not torch.any(conds_out[1])`` — Forge's CFG 1 path."""
        self._configure(on=True)
        self.uncond = torch.zeros_like(self.cond)
        args = self._args()
        self.assertIs(self.skim._post_cfg(args), args["denoised"])
        self.assertEqual(self.skim._SKIM["steps"], 0)

    def test_cfg_scale_one_is_skipped(self):
        """Host guard: upstream would divide by ``cond_scale - 1 == 0``."""
        self._configure(on=True)
        args = self._args(cond_scale=1.0)
        self.assertIs(self.skim._post_cfg(args), args["denoised"])

    def test_missing_uncond_preserves_incoming(self):
        self._configure(on=True)
        args = self._args()
        args["uncond_denoised"] = torch.zeros(1, 4, 4, 4)  # shape mismatch
        self.assertIs(self.skim._post_cfg(args), args["denoised"])

    def test_disabled_hook_preserves_incoming(self):
        args = self._args()
        self.assertIs(self.skim._post_cfg(args), args["denoised"])

    # -- publishing ----------------------------------------------------------

    def test_skim_is_published_to_downstream_guidance(self):
        """Upstream is a pre-CFG node, so later guidance must see the skim."""
        self._configure(on=True, skimming_cfg=3.0)
        args = self._args()
        cond_before = args["cond_denoised"].clone()
        uncond_before = args["uncond_denoised"].clone()

        out = self.skim._post_cfg(args)

        self.assertFalse(torch.allclose(args["cond_denoised"], cond_before))
        self.assertFalse(torch.allclose(args["uncond_denoised"], uncond_before))
        rebuilt = args["uncond_denoised"] + (
            args["cond_denoised"] - args["uncond_denoised"]) * 8.0
        self.assertTrue(torch.equal(rebuilt, out))

    def test_gated_out_steps_publish_nothing(self):
        self._configure(on=True, start=0.5, end=1.0)
        args = self._args(sigma=0.9)  # before the 50% sigma (0.75)
        cond_before = args["cond_denoised"].clone()
        self.assertIs(self.skim._post_cfg(args), args["denoised"])
        self.assertTrue(torch.equal(args["cond_denoised"], cond_before))

    def test_prepend_is_first_deduplicated_and_preserves_other_callbacks(self):
        calls = []

        def other_a(args):
            calls.append("a")
            return args["denoised"]

        def stale_skim(args):
            calls.append("stale")
            return args["denoised"]

        stale_skim._sam_extra_post_cfg_owner = self.skim._POST_CFG_OWNER

        def other_b(args):
            calls.append("b")
            return args["denoised"]

        unet = types.SimpleNamespace(
            model_options={
                "unrelated": 7,
                "sampler_post_cfg_function": [other_a, stale_skim, other_b],
            }
        )

        self.skim._prepend_post_cfg_function(unet)
        callbacks = unet.model_options["sampler_post_cfg_function"]

        self.assertIs(callbacks[0], self.skim._post_cfg)
        self.assertEqual(callbacks[1:], [other_a, other_b])
        self.assertEqual(
            sum(
                getattr(fn, "_sam_extra_post_cfg_owner", None)
                == self.skim._POST_CFG_OWNER
                for fn in callbacks
            ),
            1,
        )
        self.assertEqual(unet.model_options["unrelated"], 7)

    # -- UI --------------------------------------------------------------------

    def test_slider_ranges_follow_upstream_inputs(self):
        """Steps, defaults and ranges of the node's inputs (-1 kept on purpose)."""
        sliders = {}
        for node in ast.walk(ast.parse(SCRIPT.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "Slider"):
                kw = {k.arg: k.value for k in node.keywords}
                sliders[ast.literal_eval(kw["elem_id"])] = {
                    key: ast.literal_eval(kw[key])
                    for key in ("minimum", "maximum", "step", "value")
                }
        expected = {
            "anima_skim_start": "start_at_percentage",
            "anima_skim_end": "end_at_percentage",
            "anima_skim_flip_at": "flip_at_percentage",
            "anima_skim_cfg": "skimming_cfg",
        }
        for elem_id, name in expected.items():
            with self.subTest(slider=elem_id):
                origin = _ORIGIN_INPUTS[name]
                ours = sliders[elem_id]
                self.assertEqual(ours["step"], origin["step"])
                self.assertEqual(ours["value"], origin["default"])
                self.assertEqual(ours["maximum"], origin["max"])
                if name == "skimming_cfg":
                    # Host difference: -1 = "use the live CFG" (the preset nodes).
                    self.assertEqual(ours["minimum"], -1.0)
                else:
                    self.assertEqual(ours["minimum"], origin["min"])

    def test_logging_survives_a_legacy_console_encoding(self):
        """A log line must never be able to abort a generation.

        Several messages carry non-ASCII characters (em dash, check mark) and
        _log is called from inside the post-CFG hook, so on a Windows console
        using a legacy code page print() raising UnicodeEncodeError would kill
        the sampler run. Reproduce that stdout and assert the hook still returns.
        """

        class LegacyStream(io.TextIOBase):
            encoding = "cp949"

            def write(self, text):  # noqa: D401 - stream protocol
                text.encode("cp949")  # raises on characters cp949 lacks
                return len(text)

        self.skim._log = self._real_log
        self._configure(on=True)
        args = self._args(cond_scale=1.0)
        original = sys.stdout
        sys.stdout = LegacyStream()
        try:
            with self.assertRaises(UnicodeEncodeError):
                sys.stdout.write("em dash — here")
            out = self.skim._post_cfg(args)
        finally:
            sys.stdout = original

        self.assertIs(out, args["denoised"])
        self.assertTrue(self.skim._SKIM["warned"])


class _FakeUnet:
    def __init__(self, model_options=None):
        self.model_options = dict(model_options or {})

    def clone(self):
        return _FakeUnet(self.model_options)


if __name__ == "__main__":
    unittest.main(verbosity=2)
