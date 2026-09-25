"""Origin-parity tests for DCW(+a) — DCW, RDC, CWM and SMC in the Anima Guidance suite.

The oracle is the upstream ComfyUI node itself, ``namemechan/ComfyUI-DCW@66aaf9dd``
``dcw_node.py``, loaded verbatim from ``tests/_origin_comfyui_dcw.py`` (GPL-3.0, the same
license as this extension; its SHA-256 is pinned below), plus ComfyUI's ``cfg_function`` that
calls the node's hooks (copied below with an ``origin:`` comment). The Forge side runs this
script's real attach path and Forge's real ``sampling_function_inner``.

What must match upstream:

* INPUT_TYPES — every numeric input's default, min, max and step (Forge UI sliders), the
  server-side clamps and omitted-argument fallbacks, the SMC preset table and constants.
* Registration — ``disable_cfg1_optimization`` exactly when upstream installs its cfg hook:
  SMC on, or CWM with a non-zero alpha, and no other ``sampler_cfg_function``.
* Runtime — per-step outputs over a multi-step schedule (SMC ``e_prev`` and the RDC EMA
  carried) at CFG 4 and CFG 1, the scale taken from ``cond_scale``, SMC/CWM stepping aside
  for another ``sampler_cfg_function`` while DCW still applies, and RDC living inside DCW.

Kept host differences (parity plan §3.4): the SMC preset is a master checkbox plus a list
without ``Off``, DCW/CWM are off by default (always-on script), Forge computes the CFG bases
in x0 space inside one post-CFG hook (float rounding only), APG/ADG are not upstream. On ADG's
cond-only steps DCW still runs and SMC keeps ``e_prev``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import math
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.guidance import cwm_smc, dcw  # noqa: E402

ORIGIN_FILE = ROOT / "tests" / "_origin_comfyui_dcw.py"
ORIGIN_MARKER = "# ---- upstream dcw_node.py below (verbatim) ----\n"
# SHA-256 of namemechan/ComfyUI-DCW@66aaf9dddb03bad031c1e8443e255a811008e477:dcw_node.py with
# CRLF normalised to LF (the raw upstream file is b38af7e1…83b0f).
ORIGIN_SHA256 = "dc051505d025fde9209bbf802127b94ccad228ca69306b0edac49c2208a7d76e"


def _load_origin():
    spec = importlib.util.spec_from_file_location("_origin_comfyui_dcw", ORIGIN_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_base_tests():
    """The PAG test module's loaders (script without WebUI, Forge's real sampler)."""
    spec = importlib.util.spec_from_file_location(
        "_dcw_origin_base_tests", ROOT / "tests" / "test_anima_safe_pag.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ORIGIN = _load_origin()
_BASE = _load_base_tests()


# ---------------------------------------------------------------------------
# ComfyUI host — GPL-3.0, comfyanonymous/ComfyUI. Verbatim cfg_function.
# origin: comfyanonymous/ComfyUI@387f98aa:comfy/samplers.py:592-605
# ---------------------------------------------------------------------------


def cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep, model_options={}, cond=None, uncond=None):
    if "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options, "input_cond": cond, "input_uncond": uncond}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "cond_scale": cond_scale, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)

    return cfg_result


def _comfy_step(model_options, x, sigma, cond_pred, uncond_pred, cond_scale):
    """ComfyUI ``sampling_function`` around ``cfg_function`` for one model call.

    origin: comfyanonymous/ComfyUI@387f98aa:comfy/samplers.py:609-613 — at CFG 1 the uncond
    batch is dropped unless ``disable_cfg1_optimization``; ``_calc_cond_batch`` then leaves
    that output at zeros (samplers.py:235-236)."""
    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:  # noqa: E712
        uncond_pred = torch.zeros_like(x)
    return cfg_function(None, cond_pred, uncond_pred, cond_scale, x, sigma, model_options=model_options)


class _ComfyModel:
    """ModelPatcher stand-in: ``clone()`` and ``model_options``; ``.model`` for SMC Auto."""

    def __init__(self, model_options=None, inner=None):
        self.model_options = dict(model_options or {})
        self.model = inner if inner is not None else type("Anima", (), {})()

    def clone(self):
        return _ComfyModel(self.model_options, self.model)


def _origin_patch(config, model_options=None):
    """Upstream ``DCWModelPatch.patch`` for one parity config; returns the patched options."""
    model = _ComfyModel(model_options)
    with mock.patch("builtins.print"):
        (patched,) = ORIGIN.DCWModelPatch().patch(
            model,
            config["lambda_l"], config["lambda_h"], config["dcw_enabled"],
            config["alpha_l"], config["alpha_h"], config["cwm_enabled"],
            config["smc_preset"], config["smc_lambda"], config["smc_k"],
            config["rdc_tau"], config["rdc_alpha_ll"], config["rdc_alpha_hh"],
        )
    return patched.model_options


# ---------------------------------------------------------------------------
# Forge host
# ---------------------------------------------------------------------------


class _ForgeUnet:
    """Forge ModelPatcher option handling.

    origin: Forge backend/patcher/base.py:65-69, 244-245 (set_model_options_post_cfg_function)."""

    def __init__(self, model_options=None):
        self.model_options = dict(model_options or {})
        self.post_cfg_calls = []

    def clone(self):
        return _ForgeUnet(self.model_options)

    def set_model_sampler_post_cfg_function(self, post_cfg_function, disable_cfg1_optimization=False):
        self.post_cfg_calls.append(disable_cfg1_optimization)
        self.model_options["sampler_post_cfg_function"] = self.model_options.get("sampler_post_cfg_function", []) + [post_cfg_function]
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_pre_cfg_function(self, pre_cfg_function, disable_cfg1_optimization=False):
        self.model_options["sampler_pre_cfg_function"] = self.model_options.get("sampler_pre_cfg_function", []) + [pre_cfg_function]

    def set_model_unet_function_wrapper(self, unet_wrapper_function):
        self.model_options["model_function_wrapper"] = unet_wrapper_function


class _ForgeModel:
    """``model.apply_model`` returns the condition tensor itself, so Forge's real batching
    yields ``cond_pred``/``uncond_pred`` bit-exactly."""

    @staticmethod
    def memory_required(shape):
        return shape[0]

    @staticmethod
    def apply_model(x, timestep, *, bias, transformer_options):
        return bias.clone()


class _CountingForgeModel(_ForgeModel):
    """Counts the latent rows Forge evaluates (1 per cond, 1 per uncond at batch 1)."""

    def __init__(self):
        self.rows = 0

    def apply_model(self, x, timestep, *, bias, transformer_options):
        self.rows += int(x.shape[0])
        return bias.clone()


# UI/script-argument index of each upstream input (scripts/anima_safe_pag.py ui()).
UI_INDEX = {
    "alpha_l": 24, "alpha_h": 25, "smc_lambda": 26, "smc_k": 27,
    "lambda_l": 29, "lambda_h": 30,
    "rdc_tau": 59, "rdc_alpha_ll": 60, "rdc_alpha_hh": 61,
}
# XYZ override key → upstream input.
XYZ_KEY = {
    "cwm_alpha_low": "alpha_l", "cwm_alpha_high": "alpha_h",
    "smc_lambda": "smc_lambda", "smc_k": "smc_k",
    "dcw_lambda_low": "lambda_l", "dcw_lambda_high": "lambda_h",
    "rdc_tau": "rdc_tau", "rdc_alpha_ll": "rdc_alpha_ll", "rdc_alpha_hh": "rdc_alpha_hh",
}

BASE_CONFIG = dict(
    lambda_l=0.05, lambda_h=0.01, dcw_enabled=False,
    alpha_l=0.0, alpha_h=0.0, cwm_enabled=False,
    smc_preset="Off", smc_lambda=6.0, smc_k=0.1,
    rdc_tau=0.0, rdc_alpha_ll=0.03, rdc_alpha_hh=0.0,
)

PARITY_CONFIGS = {
    "dcw_default": dict(dcw_enabled=True),
    "dcw_rdc": dict(dcw_enabled=True, rdc_tau=0.1, rdc_alpha_ll=0.05, rdc_alpha_hh=0.01),
    "rdc_only": dict(dcw_enabled=True, lambda_l=0.0, lambda_h=0.0, rdc_tau=0.2),
    "cwm": dict(cwm_enabled=True, alpha_l=0.3, alpha_h=0.15),
    "cwm_wide": dict(cwm_enabled=True, alpha_l=1.8, alpha_h=-0.5),
    "smc_auto": dict(smc_preset="Auto"),
    "smc_cwm_dcw_rdc": dict(
        smc_preset="Custom", smc_lambda=4.0, smc_k=0.3,
        cwm_enabled=True, alpha_l=0.2, alpha_h=0.1,
        dcw_enabled=True, lambda_l=-0.08, lambda_h=0.02,
        rdc_tau=0.15, rdc_alpha_ll=0.03,
    ),
}


class DcwOriginTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _BASE._load_pag_module()
        cls.sampler, cls.Condition, cls.memory = _BASE._load_forge_sampler()

    def setUp(self):
        _BASE.AnimaSafePagTests.setUp(self)

    # -- Forge attach -------------------------------------------------------

    def _ui_values(self):
        with gr.Blocks():
            inputs = self.pag.AnimaSafePAG().ui(False)
        return inputs, [component.value for component in inputs]

    def _forge_args(self, config):
        config = {**BASE_CONFIG, **config}
        _inputs, args = self._ui_values()
        args[28] = config["dcw_enabled"]
        args[43] = config["cwm_enabled"]
        args[57] = config["smc_preset"] != "Off"
        if config["smc_preset"] != "Off":
            args[56] = config["smc_preset"]
        for key, index in UI_INDEX.items():
            args[index] = config[key]
        return args

    def _attach(self, args, *, unet=None, xyz=None, steps=20, **request_fields):
        """``request_fields`` are StableDiffusionProcessing attributes such as ``cfg_scale``,
        ``is_hr_pass``/``hr_cfg`` or the refiner fields; omitted, the pass CFG is unknown."""
        model = type("Anima", (), {})()
        model.forge_objects = types.SimpleNamespace(unet=unet or _ForgeUnet())
        request = types.SimpleNamespace(
            sd_model=model, extra_generation_params={}, steps=steps, **request_fields,
        )
        if xyz:
            request._anima_safe_pag_xyz = dict(xyz)
        with mock.patch.object(self.pag, "_log"):
            self.pag.AnimaSafePAG().process_before_every_sampling(request, *args)
        return request, model.forge_objects.unet

    def _forge_step(self, model_options, x, sigma, cond_pred, uncond_pred, cond_scale, cond_extra=None,
                    model=None):
        """One Forge model call. ``uncond_pred=None`` is Forge at CFG == 1, which encodes no
        negative prompt (modules/processing.py:481-483, hires :1606-1608)."""
        Condition = self.Condition
        cond = [{"model_conds": {"bias": Condition(cond_pred)}, **(cond_extra or {})}]
        uncond = None if uncond_pred is None else [{"model_conds": {"bias": Condition(uncond_pred)}}]
        with mock.patch.dict(sys.modules, {"backend.sampling.sampling_function": self.sampler}):
            return self.sampler.sampling_function_inner(
                model or _ForgeModel(), x, sigma, uncond, cond, cond_scale, model_options,
            )

    @staticmethod
    def _schedule(shape=(1, 4, 8, 8), seed=0):
        generator = torch.Generator().manual_seed(seed)
        sigmas = (1.0, 0.92, 0.8, 0.64, 0.45, 0.28, 0.12, 0.04)
        steps = []
        for sigma in sigmas:
            x = torch.randn(shape, generator=generator) * (1.0 + sigma)
            cond = torch.randn(shape, generator=generator)
            uncond = cond * 0.6 + 0.4 * torch.randn(shape, generator=generator)
            steps.append((torch.tensor([sigma]), x, cond, uncond))
        return steps


class InputTypesOriginTests(DcwOriginTestCase):
    def test_vendored_upstream_file_is_unchanged(self):
        text = ORIGIN_FILE.read_text(encoding="utf-8")
        self.assertIn(ORIGIN_MARKER, text)
        body = text.split(ORIGIN_MARKER, 1)[1]
        self.assertEqual(hashlib.sha256(body.encode("utf-8")).hexdigest(), ORIGIN_SHA256)

    def test_ui_sliders_match_upstream_input_types(self):
        """origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:630-801 (INPUT_TYPES)."""
        required = ORIGIN.DCWModelPatch.INPUT_TYPES()["required"]
        inputs, _values = self._ui_values()
        for key, index in UI_INDEX.items():
            spec = required[key][1]
            slider = inputs[index]
            with self.subTest(input=key, elem_id=slider.elem_id):
                self.assertEqual(
                    (slider.value, slider.minimum, slider.maximum, slider.step),
                    (spec["default"], spec["min"], spec["max"], spec["step"]),
                )

    def test_upstream_inputs_are_all_covered(self):
        required = ORIGIN.DCWModelPatch.INPUT_TYPES()["required"]
        numeric = {key for key, value in required.items() if value[0] == "FLOAT"}
        self.assertEqual(numeric, set(UI_INDEX))
        self.assertEqual(
            {key for key, value in required.items() if value[0] == "BOOLEAN"},
            {"dcw_enabled", "cwm_enabled"},
        )
        self.assertEqual(
            list(inspect.signature(ORIGIN.DCWModelPatch.patch).parameters)[1:],
            ["model", "lambda_l", "lambda_h", "dcw_enabled", "alpha_l", "alpha_h",
             "cwm_enabled", "smc_preset", "smc_lambda", "smc_k",
             "rdc_tau", "rdc_alpha_ll", "rdc_alpha_hh"],
        )

    def test_documented_host_differences_in_the_switches(self):
        """Always-on script: DCW/CWM start off (upstream True: dcw_node.py:669-672, 706-709);
        SMC Off is the master checkbox, so the list starts at Auto (dcw_node.py:712-727)."""
        required = ORIGIN.DCWModelPatch.INPUT_TYPES()["required"]
        inputs, _values = self._ui_values()
        self.assertTrue(required["dcw_enabled"][1]["default"])
        self.assertTrue(required["cwm_enabled"][1]["default"])
        self.assertEqual(inputs[28].elem_id, "anima_guidance_dcw_enable")
        self.assertFalse(inputs[28].value)
        self.assertEqual(inputs[43].elem_id, "anima_guidance_cwm_enable")
        self.assertFalse(inputs[43].value)
        self.assertEqual(required["smc_preset"][1]["default"], "Off")
        self.assertFalse(inputs[57].value)
        choices = [
            choice[0] if isinstance(choice, (tuple, list)) else choice
            for choice in inputs[56].choices
        ]
        self.assertEqual(
            choices, [name for name in ORIGIN.SMC_PRESET_NAMES if name != "Off"],
        )

    def test_rdc_has_no_visible_switch_and_arg_58_keeps_its_slot(self):
        """Upstream: no RDC toggle, rdc_tau alone (dcw_node.py:752-756)."""
        inputs, _values = self._ui_values()
        placeholder = inputs[58]
        self.assertEqual(placeholder.elem_id, "anima_guidance_rdc_enable")
        self.assertFalse(placeholder.visible)
        self.assertIs(placeholder.value, True)
        self.assertTrue(getattr(placeholder, "do_not_save_to_config", False))
        self.assertEqual(len(inputs), 62)

    def test_smc_presets_and_constants_match_upstream(self):
        """origin: dcw_node.py:75-94 (presets, names), 243-290 (_ch_energy_weight, _SMC_CLAMP_MIN)."""
        self.assertEqual(cwm_smc.SMC_PRESETS, ORIGIN._SMC_PRESETS)
        self.assertEqual(list(cwm_smc.SMC_PRESET_NAMES), ORIGIN.SMC_PRESET_NAMES)
        # The later module-level assignment (1e-8) is the one upstream's SMC uses.
        self.assertEqual(ORIGIN._SMC_CLAMP_MIN, 1e-8)
        self.assertEqual(cwm_smc.SMC_NORM_EPS, ORIGIN._SMC_CLAMP_MIN)
        self.assertEqual(cwm_smc.SMC_DELTA_FLOOR, ORIGIN._SMC_CLAMP_MIN)
        ours = inspect.signature(dcw._channel_energy_weight).parameters
        theirs = inspect.signature(ORIGIN._ch_energy_weight).parameters
        self.assertEqual(
            (ours["clamp_low"].default, ours["clamp_high"].default),
            (theirs["clamp_lo"].default, theirs["clamp_hi"].default),
        )
        apply_defaults = inspect.signature(ORIGIN.apply_dcw).parameters
        ours_dcw = inspect.signature(dcw.apply_dcw).parameters
        for key in ("rdc_alpha_ll", "rdc_alpha_hh", "rdc_tau"):
            self.assertEqual(ours_dcw[key].default, apply_defaults[key].default)

    def test_server_side_clamps_are_the_upstream_domains(self):
        """XYZ/API values outside INPUT_TYPES clamp to upstream min/max; non-finite values fall
        back to the upstream default."""
        required = ORIGIN.DCWModelPatch.INPUT_TYPES()["required"]
        args = self._forge_args(dict(
            dcw_enabled=True, cwm_enabled=True, smc_preset="Custom", rdc_tau=0.1,
        ))

        def resolved():
            return {
                "alpha_l": self.pag._CFG["alpha_low"], "alpha_h": self.pag._CFG["alpha_high"],
                "smc_lambda": self.pag._CFG["smc_lambda"], "smc_k": self.pag._CFG["smc_k"],
                "lambda_l": self.pag._DCW["lambda_low"], "lambda_h": self.pag._DCW["lambda_high"],
                "rdc_tau": self.pag._DCW["rdc_tau"], "rdc_alpha_ll": self.pag._DCW["rdc_alpha_ll"],
                "rdc_alpha_hh": self.pag._DCW["rdc_alpha_hh"],
            }

        for label, pick in (("above", "max"), ("below", "min")):
            with self.subTest(side=label):
                self.setUp()
                shift = 10.0 if pick == "max" else -10.0
                xyz = {key: required[name][1][pick] + shift for key, name in XYZ_KEY.items()}
                self._attach(args, xyz=xyz)
                self.assertEqual(
                    resolved(), {name: required[name][1][pick] for name in UI_INDEX},
                )
        with self.subTest(side="nan"):
            # Non-finite API values (an XYZ NaN keeps the UI value instead).
            self.setUp()
            nan_args = list(args)
            for index in UI_INDEX.values():
                nan_args[index] = float("nan")
            self._attach(nan_args)
            self.assertEqual(
                resolved(), {name: required[name][1]["default"] for name in UI_INDEX},
            )
        with self.subTest(side="inside"):
            self.setUp()
            self._attach(args, xyz={"cwm_alpha_low": 1.8, "dcw_lambda_high": -0.25})
            self.assertEqual(self.pag._CFG["alpha_low"], 1.8)
            self.assertEqual(self.pag._DCW["lambda_high"], -0.25)

    def test_omitted_arguments_fall_back_to_upstream_defaults(self):
        """Short API/infotext argument lists stop before an index: the missing value is the
        upstream default (dcw_node.py:636-777)."""
        _inputs, values = self._ui_values()

        args = list(values[:29])
        args[28] = True
        self._attach(args)
        self.assertEqual((self.pag._DCW["lambda_low"], self.pag._DCW["lambda_high"]), (0.05, 0.01))
        self.assertEqual(self.pag._DCW["rdc_tau"], 0.0)
        self.assertFalse(self.pag._DCW["rdc_on"])

        self.setUp()
        args = list(values[:24])
        args[22] = "CWM"
        _request, unet = self._attach(args)
        self.assertEqual((self.pag._CFG["alpha_low"], self.pag._CFG["alpha_high"]), (0.0, 0.0))
        # Upstream: CWM with both alphas 0 installs no cfg hook and so no CFG 1 flag.
        self.assertEqual(unet.post_cfg_calls, [False])

        self.setUp()
        args = list(values[:27])
        args[22] = "SMC"
        self._attach(args)
        self.assertEqual((self.pag._CFG["smc_lambda"], self.pag._CFG["smc_k"]), (6.0, 0.1))


class RdcGateOriginTests(DcwOriginTestCase):
    def test_rdc_follows_dcw_enabled_and_tau(self):
        """origin: dcw_node.py:817-819 — ``rdc_on = rdc_tau > 0``, ``dcw_active =
        dcw_enabled and (lambda_l or lambda_h or rdc_on)``; RDC runs only in the DCW hook."""
        for dcw_enabled in (False, True):
            for lambdas in ((0.05, 0.01), (0.0, 0.0)):
                for tau in (0.0, 0.2):
                    config = dict(dcw_enabled=dcw_enabled, lambda_l=lambdas[0],
                                  lambda_h=lambdas[1], rdc_tau=tau)
                    with self.subTest(**config):
                        self.setUp()
                        self._attach(self._forge_args(config))
                        rdc_on = tau > 0.0
                        dcw_active = dcw_enabled and (lambdas != (0.0, 0.0) or rdc_on)
                        self.assertEqual(self.pag._DCW["on"], dcw_active)
                        self.assertEqual(self.pag._DCW["rdc_on"], dcw_enabled and rdc_on)
                        self.assertEqual(self.pag._STATE["requested_rdc"], dcw_enabled and rdc_on)
                        origin_options = _origin_patch({**BASE_CONFIG, **config})
                        self.assertEqual(
                            dcw_active,
                            len(origin_options.get("sampler_post_cfg_function", [])) == 1,
                        )

    def test_legacy_rdc_slot_only_vetoes(self):
        """Arg 58 (old RDC toggle) False from an API caller or the legacy XYZ axis turns RDC
        off; True — what this UI always sends — leaves the upstream gate alone."""
        args = self._forge_args(dict(dcw_enabled=True, rdc_tau=0.2))
        self._attach(args)
        self.assertTrue(self.pag._DCW["rdc_on"])

        self.setUp()
        args[58] = False
        self._attach(args)
        self.assertFalse(self.pag._DCW["rdc_on"])
        self.assertTrue(self.pag._DCW["on"])

        self.setUp()
        args[58] = True
        self._attach(args, xyz={"rdc_enabled": "False"})
        self.assertFalse(self.pag._DCW["rdc_on"])

        self.setUp()
        args[28] = False
        self._attach(args, xyz={"rdc_enabled": "True"})
        self.assertFalse(self.pag._DCW["rdc_on"])


class RegistrationOriginTests(DcwOriginTestCase):
    CASES = {
        "nothing": {},
        "dcw_only": dict(dcw_enabled=True),
        "dcw_rdc": dict(dcw_enabled=True, rdc_tau=0.1),
        "cwm_zero_alpha": dict(cwm_enabled=True),
        "cwm_alpha": dict(cwm_enabled=True, alpha_l=0.3),
        "cwm_off_with_alpha": dict(alpha_l=0.3, alpha_h=0.2),
        "smc_auto": dict(smc_preset="Auto"),
        "smc_custom_k0": dict(smc_preset="Custom", smc_k=0.0),
        "smc_cwm_dcw": dict(smc_preset="Flux", cwm_enabled=True, alpha_h=0.2, dcw_enabled=True),
    }

    def test_disable_cfg1_optimization_matches_upstream(self):
        """origin: dcw_node.py:817-878 — the flag comes with the cfg hook."""
        for name, config in self.CASES.items():
            for foreign_cfg in (False, True):
                with self.subTest(case=name, foreign_sampler_cfg_function=foreign_cfg):
                    self.setUp()
                    existing = {"sampler_cfg_function": lambda args: args["cond"]} if foreign_cfg else {}
                    origin_options = _origin_patch({**BASE_CONFIG, **config}, existing)
                    _request, unet = self._attach(
                        self._forge_args(config), unet=_ForgeUnet(existing),
                    )
                    expected = bool(origin_options.get("disable_cfg1_optimization", False))
                    self.assertEqual(
                        bool(unet.model_options.get("disable_cfg1_optimization", False)),
                        expected,
                    )
                    if unet.post_cfg_calls:
                        self.assertEqual(unet.post_cfg_calls, [expected])

    def test_flag_only_on_a_pass_configured_at_cfg_one(self):
        """Forge-only: in a CFG≠1 pass the only cond_scale=1 steps are Forge's "Ignore Negative
        Prompt during Early Steps" / NGMS skips (modules/sd_samplers_cfg_denoiser.py:141-148),
        which rely on the cfg1 shortcut (backend/sampling/sampling_function.py:295). The flag
        follows the pass's own CFG: cfg_scale, hr_cfg on hires, refiner_cfg after a refiner
        switch (sd_samplers_cfg_denoiser.py:136-139). Without a readable CFG the upstream
        registration applies (test above)."""
        config = dict(cwm_enabled=True, alpha_l=0.3, smc_preset="Auto")
        refiner = dict(refiner_checkpoint_info=object(), refiner_switch_at=0.8)
        cases = {
            "cfg_4": (dict(cfg_scale=4.0), False),
            "cfg_1": (dict(cfg_scale=1.0), True),
            "hires_at_1_after_cfg_4": (dict(cfg_scale=4.0, is_hr_pass=True, hr_cfg=1.0), True),
            "hires_at_4_after_cfg_1": (dict(cfg_scale=1.0, is_hr_pass=True, hr_cfg=4.0), False),
            "refiner_at_1": (dict(cfg_scale=4.0, refiner_cfg=1.0, **refiner), True),
            "refiner_cfg_unset": (dict(cfg_scale=4.0, refiner_cfg=None, **refiner), False),
            "refiner_cfg_without_refiner": (
                dict(cfg_scale=4.0, refiner_cfg=1.0, refiner_checkpoint_info=None,
                     refiner_switch_at=None),
                False,
            ),
        }
        for name, (fields, expected) in cases.items():
            with self.subTest(case=name):
                self.setUp()
                _request, unet = self._attach(self._forge_args(config), **fields)
                self.assertEqual(unet.post_cfg_calls, [expected])
                self.assertEqual(
                    bool(unet.model_options.get("disable_cfg1_optimization", False)), expected,
                )

    def test_apg_alone_keeps_the_cfg1_shortcut(self):
        """APG is not upstream; it is skipped at CFG≈1 and never forces an uncond pass."""
        args = self._forge_args({})
        args[12] = True
        _request, unet = self._attach(args)
        self.assertEqual(unet.post_cfg_calls, [False])
        self.assertNotIn("disable_cfg1_optimization", unet.model_options)


class RuntimeOriginTests(DcwOriginTestCase):
    def _compare(self, config, cond_scale, *, shape=(1, 4, 8, 8), existing=None, cond_extra=None):
        full = {**BASE_CONFIG, **config}
        origin_options = dict(_origin_patch(full, existing))  # fresh per KSampler run
        _request, unet = self._attach(
            self._forge_args(config), unet=_ForgeUnet(existing), cfg_scale=cond_scale,
        )
        forge_options = unet.model_options
        worst = 0.0
        for index, (sigma, x, cond, uncond) in enumerate(self._schedule(shape)):
            with mock.patch("builtins.print"):
                expected = _comfy_step(origin_options, x, sigma, cond, uncond, cond_scale)
            actual = self._forge_step(forge_options, x, sigma, cond, uncond, cond_scale, cond_extra)
            worst = max(worst, float((actual - expected).abs().max()))
            with self.subTest(step=index):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
        return worst

    def test_per_step_outputs_match_upstream_at_cfg_4_and_cfg_1(self):
        for name, config in PARITY_CONFIGS.items():
            for cond_scale in (4.0, 1.0):
                with self.subTest(config=name, cfg=cond_scale):
                    self.setUp()
                    self._compare(config, cond_scale)

    def test_odd_latent_sizes_match_upstream(self):
        """Cosmos latents can be odd (dcw_node.py:135-151 reflect padding)."""
        for name in ("dcw_rdc", "smc_cwm_dcw_rdc"):
            with self.subTest(config=name):
                self.setUp()
                self._compare(PARITY_CONFIGS[name], 4.0, shape=(1, 4, 7, 9))

    def test_cfg_one_changes_the_image_like_upstream(self):
        """Upstream SMC/CWM act at CFG 1 (disable_cfg1_optimization); they are not a no-op
        when an uncond exists (a CFG≈1 pass with an encoded negative, e.g. a refiner at 1)."""
        sigma, x, cond, uncond = self._schedule()[2]
        _request, unet = self._attach(self._forge_args(PARITY_CONFIGS["cwm"]), cfg_scale=1.0)
        out = self._forge_step(unet.model_options, x, sigma, cond, uncond, 1.0)
        self.assertFalse(torch.allclose(out, cond))
        self.assertEqual(self.pag._CFG["steps"], 1)

    def test_forge_skip_negative_steps_stay_negative_free_in_a_cfg_4_pass(self):
        """Forge-only, no ComfyUI counterpart: "Ignore Negative Prompt during Early Steps" and
        NGMS force cond_scale=1 on some steps of a CFG 4 pass (sd_samplers_cfg_denoiser.py:
        141-148). Those steps keep Forge's cond-only result — no uncond forward, SMC/CWM not
        applied at w=1 — while the pass's real CFG 4 steps still get them."""
        config = dict(cwm_enabled=True, alpha_l=0.3, alpha_h=0.15, smc_preset="Auto")
        _request, unet = self._attach(self._forge_args(config), cfg_scale=4.0)
        self.assertNotIn("disable_cfg1_optimization", unet.model_options)
        schedule = self._schedule()

        sigma, x, cond, uncond = schedule[1]
        model = _CountingForgeModel()
        with mock.patch.object(self.pag, "_log"):
            skipped = self._forge_step(unet.model_options, x, sigma, cond, uncond, 1.0, model=model)
        self.assertEqual(model.rows, 1)
        torch.testing.assert_close(skipped, cond, rtol=0, atol=0)
        self.assertEqual(self.pag._CFG["steps"], 0)
        self.assertIsNone(self.pag._RUNTIME.smc_prev)

        sigma, x, cond, uncond = schedule[2]
        model = _CountingForgeModel()
        guided = self._forge_step(unet.model_options, x, sigma, cond, uncond, 4.0, model=model)
        self.assertEqual(model.rows, 2)
        self.assertEqual(self.pag._CFG["steps"], 1)
        self.assertFalse(torch.allclose(guided, uncond + 4.0 * (cond - uncond)))

    def test_forge_cfg_exactly_one_has_no_uncond_so_smc_cwm_step_aside(self):
        """Host difference: at CFG == 1 Forge encodes no negative prompt at all
        (modules/processing.py:481-483; hires :1606-1608), so disable_cfg1_optimization has no
        uncond to evaluate and ``uncond_denoised`` is zeros. SMC/CWM must not run on that
        (CWM would reweight the whole prediction); the cond prediction is kept and DCW still
        corrects it like upstream's every-step post-CFG hook."""
        config = dict(
            cwm_enabled=True, alpha_l=0.3, alpha_h=0.15, smc_preset="Auto", dcw_enabled=True,
        )
        _request, unet = self._attach(self._forge_args(config), cfg_scale=1.0)
        self.assertTrue(unet.model_options["disable_cfg1_optimization"])
        sigma, x, cond, _uncond = self._schedule()[3]
        model = _CountingForgeModel()
        with mock.patch.object(self.pag, "_log") as log:
            out = self._forge_step(unet.model_options, x, sigma, cond, None, 1.0, model=model)
        self.assertEqual(model.rows, 1)
        expected = ORIGIN.apply_dcw(cond, x, sigma, 0.05, 0.01)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=2e-5)
        self.assertEqual(self.pag._CFG["steps"], 0)
        self.assertEqual(self.pag._DCW["dcw_steps"], 1)
        self.assertIsNone(self.pag._RUNTIME.smc_prev)
        messages = [call.args[0] for call in log.call_args_list]
        self.assertTrue(any("(SMC/CWM) skipped" in m and "no uncond" in m for m in messages), messages)

    def test_scale_is_cond_scale_not_a_fit_of_the_incoming_result(self):
        """An earlier post-CFG hook (e.g. MaHiRo) may rescale the incoming result; upstream's
        cfg hook still combines at ``args["cond_scale"]`` (dcw_node.py:848-866)."""
        config = PARITY_CONFIGS["cwm"]
        origin_options = _origin_patch({**BASE_CONFIG, **config})
        _request, unet = self._attach(self._forge_args(config))
        sigma, x, cond, uncond = self._schedule()[1]

        def earlier_hook(args):
            return args["denoised"] * 1.3

        options = dict(unet.model_options)
        options["sampler_post_cfg_function"] = [earlier_hook, *options["sampler_post_cfg_function"]]
        actual = self._forge_step(options, x, sigma, cond, uncond, 4.0)
        expected = _comfy_step(dict(origin_options), x, sigma, cond, uncond, 4.0)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
        self.assertEqual(self.pag._CFG["effective_scale"], 4.0)

    def test_forge_edit_strength_multiplies_the_scale_like_vanilla_cfg(self):
        """Forge's linear CFG multiplies by edit_strength (sampling_function.py:293, 307-310);
        the base uses the same product, so CWM runs at cond_scale × edit_strength."""
        config = PARITY_CONFIGS["cwm"]
        _request, unet = self._attach(self._forge_args(config))
        sigma, x, cond, uncond = self._schedule()[1]
        actual = self._forge_step(
            unet.model_options, x, sigma, cond, uncond, 4.0, cond_extra={"strength": 1.5},
        )
        origin_options = dict(_origin_patch({**BASE_CONFIG, **config}))
        expected = _comfy_step(origin_options, x, sigma, cond, uncond, 6.0)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
        self.assertEqual(self.pag._CFG["effective_scale"], 6.0)

    def test_foreign_sampler_cfg_function_keeps_its_result_and_dcw_still_applies(self):
        """origin: dcw_node.py:834-841, 880-889 — SMC/CWM skipped with one warning, DCW appended."""

        def rescale_like(args):  # a stand-in sampler_cfg_function (x - pred space)
            cond, uncond, scale = args["cond"], args["uncond"], args["cond_scale"]
            return uncond + scale * (cond - uncond) * 0.9

        existing = {"sampler_cfg_function": rescale_like}
        config = PARITY_CONFIGS["smc_cwm_dcw_rdc"]
        with mock.patch.object(self.pag, "_log") as log:
            self._compare(config, 4.0, existing=existing)
        warnings = [call.args[0] for call in log.call_args_list if "sampler_cfg_function" in call.args[0]]
        self.assertEqual(len(warnings), 1)
        self.assertIn("SMC/CWM skipped", warnings[0])
        self.assertEqual(self.pag._CFG["steps"], 0)
        self.assertGreater(self.pag._DCW["dcw_steps"], 0)
        self.assertIsNone(self.pag._RUNTIME.smc_prev)

    def test_foreign_sampler_cfg_function_keeps_pag_running(self):
        p = self.pag
        cond = torch.tensor([[2.0, -1.0]])
        weak = torch.tensor([[1.5, -0.5]])
        base = torch.tensor([[7.0, 3.0]])
        p._STATE.update(on=True, attn_raw=weak, attn_scale=2.0, rescale=0.0)
        p._CFG.update(smc_on=True, cwm_on=True, alpha_low=0.3, alpha_high=0.1)
        with mock.patch.object(p, "_log"):
            out = p._post_cfg({
                "denoised": base,
                "cond_denoised": cond,
                "uncond_denoised": torch.zeros_like(cond),
                "cond_scale": 4.0,
                "sigma": torch.tensor([1.0]),
                "model_options": {"sampler_cfg_function": object()},
            })
        torch.testing.assert_close(out, base + 2.0 * (cond - weak))
        self.assertEqual(p._STATE["applied_steps"], 1)


class AdaptiveGuidanceDcwTests(DcwOriginTestCase):
    def test_adg_cond_only_step_still_gets_dcw_and_keeps_smc_state(self):
        """ADG is not upstream. On its cond-only steps the incoming result is the cond
        prediction; DCW still corrects it like upstream's every-step post-CFG hook, APG
        momentum is cleared and SMC's e_prev survives for the next guided step."""
        args = self._forge_args(dict(dcw_enabled=True, smc_preset="Auto"))
        args[17] = True   # Adaptive Guidance
        args[18] = 0.0    # skip from the first step
        args[19] = 0      # never keep an uncond step
        _request, unet = self._attach(args)
        self.assertIn("model_function_wrapper", unet.model_options)
        previous = torch.ones(1, 4, 8, 8)
        self.pag._RUNTIME.smc_prev = previous
        self.pag._APG.update(avg=torch.ones(1), last_sigma=1.0)

        sigma, x, cond, uncond = self._schedule()[3]
        out = self._forge_step(unet.model_options, x, sigma, cond, uncond, 4.0)

        expected = ORIGIN.apply_dcw(cond, x, sigma, 0.05, 0.01)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=2e-5)
        self.assertEqual(self.pag._STATE["adg_skipped_steps"], 1)
        self.assertEqual(self.pag._DCW["dcw_steps"], 1)
        self.assertEqual(self.pag._CFG["steps"], 0)
        self.assertIs(self.pag._RUNTIME.smc_prev, previous)
        self.assertIsNone(self.pag._APG["avg"])


if __name__ == "__main__":
    unittest.main()
