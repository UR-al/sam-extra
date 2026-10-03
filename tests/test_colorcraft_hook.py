"""Colorcraft Forge hook (sam3ext/colorcraft/hook.py) — attach/detach, passes, offsets, safety.

The post-CFG function goes on a clone of the pass's UNet in ``process_before_every_sampling``, tagged
with this script's owner, after every other sam-extra post-CFG function. These tests drive the real
``hook.process`` with Forge's ModelPatcher option semantics, Forge's real latent formats and
``setup_img2img_steps`` from the Forge checkout, and compare against upstream's node where the
behaviour is the node's.
"""

from __future__ import annotations

import gc
import importlib.util
import sys
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _support():
    spec_ = importlib.util.spec_from_file_location("_colorcraft_support", ROOT / "tests" / "_colorcraft_support.py")
    module = sys.modules.get(spec_.name)
    if module is None:
        module = importlib.util.module_from_spec(spec_)
        sys.modules[spec_.name] = module
        spec_.loader.exec_module(module)
    return module


S = _support()

from sam3ext.colorcraft import engine, hook, schedule, spec  # noqa: E402
from sam3ext.guidance import dave_gate  # noqa: E402

Mod, Scenario = S.Mod, S.Scenario


def _config_args(*mods, masking=False, enabled=True, leaves=None, combos=None):
    return S.scenario_args(Scenario("t", list(mods), leaves or {}, combos or {}, masking)) if enabled else \
        spec.config_to_args(spec.default_config())


def _p(family="krea2", **fields):
    return S.make_p(family, **fields)


def _attach(p, args, modules_stub=None):
    return S.attach_ours(p, args, modules_stub=modules_stub)


def _sigmas(steps=10):
    return S.flow_sigmas(steps)


EXPOSURE = Mod("advanced", {"start": 0.0, "end": 1.0, "exposure": 0.4})


class OffTests(unittest.TestCase):
    """Off means off: no clone, no metadata, nothing on ``p``."""

    def setUp(self):
        S.require_forge(self)

    def _assert_untouched(self, p):
        self.assertEqual(p.sd_model.forge_objects.unet.clones, 0)
        self.assertEqual(p.extra_generation_params, {})
        self.assertFalse(hasattr(p, hook.STATE_ATTR))
        self.assertNotIn("sampler_post_cfg_function", p.sd_model.forge_objects.unet.model_options)

    def test_disabled(self):
        p = _p()
        self.assertIsNone(_attach(p, _config_args(EXPOSURE, enabled=False)))
        self._assert_untouched(p)

    def test_default_arguments_are_off(self):
        p = _p()
        self.assertIsNone(_attach(p, [c for c in spec.config_to_args(spec.default_config())]))
        self._assert_untouched(p)
        self.assertIsNone(_attach(p, []))
        self._assert_untouched(p)

    def test_enabled_but_nothing_to_do(self):
        cases = {
            "no active tab": spec.config_to_args(_inactive_config()),
            "every amount 0": _config_args(Mod("advanced", {"start": 0.0, "end": 1.0})),
            "strength 0": _config_args(Mod("advanced", {"strength": 0.0, "exposure": 0.5})),
            "gated off": _config_args(Mod("advanced", {"more_colors": False, "lab_a": 0.5, "color_shift": False,
                                                       "color_shift_amount": 0.5})),
            "hires-only tab on the base pass": _config_args(Mod("advanced", {"pass": "Hires", "exposure": 0.5})),
        }
        for label, args in cases.items():
            with self.subTest(case=label):
                p = _p()
                self.assertIsNone(_attach(p, args))
                self._assert_untouched(p)

    def test_sam3_and_adetailer_inner_passes_are_skipped(self):
        for attr in ("_sam3_inner", "_ad_inner"):
            with self.subTest(inner=attr):
                p = _p(**{attr: True})
                self.assertIsNone(_attach(p, _config_args(EXPOSURE)))
                self._assert_untouched(p)


def _inactive_config():
    config = spec.default_config()
    config.enabled = True
    config.modifiers[0]["active"] = False
    config.modifiers[0]["exposure"] = 0.5
    return config


class AttachTests(unittest.TestCase):
    def setUp(self):
        S.require_forge(self)

    def test_attached_last_after_every_other_post_cfg_function(self):
        """Skimmed CFG (prepends), Optimal Scale and the guidance suite (append) and a foreign function stay;
        a stale Colorcraft callback is replaced; ours runs last."""
        skimmed = lambda a: a["denoised"]  # noqa: E731
        skimmed._sam_extra_post_cfg_owner = "sam-extra/anima-skimmed-cfg"
        optimal = lambda a: a["denoised"]  # noqa: E731
        optimal._sam_extra_optimal_scale_owner = "sam-extra/anima-optimal-scale"
        pag = lambda a: a["denoised"]  # noqa: E731
        foreign = lambda a: a["denoised"]  # noqa: E731
        stale = lambda a: a["denoised"]  # noqa: E731
        setattr(stale, hook.OWNER_ATTR, hook.OWNER)
        unet = S.FakeUnet({"transformer_options": {},
                           "sampler_post_cfg_function": [skimmed, optimal, pag, stale, foreign]})
        p = _p(unet=unet)
        callback = _attach(p, _config_args(EXPOSURE))
        installed = p.sd_model.forge_objects.unet.model_options["sampler_post_cfg_function"]
        self.assertEqual(installed[:4], [skimmed, optimal, pag, foreign])
        self.assertIs(installed[-1], callback)
        self.assertEqual(sum(hook.is_owned(fn) for fn in installed), 1)
        # the original UNet is untouched (the clone carries the hook)
        self.assertEqual(unet.model_options["sampler_post_cfg_function"], [skimmed, optimal, pag, stale, foreign])

    def test_scripts_registering_post_cfg_functions_load_before_colorcraft(self):
        """Forge calls process_before_every_sampling in load order (alphabetical within the extension), so
        appending makes Colorcraft the last sam-extra post-CFG function of the pass."""
        registering = sorted(
            path.name for path in (ROOT / "scripts").glob("*.py")
            if path.name != "colorcraft.py"
            and "sampler_post_cfg_function" in path.read_text(encoding="utf-8"))
        self.assertIn("anima_cfg_optimal_scale.py", registering)
        self.assertIn("anima_safe_pag.py", registering)
        for name in registering:
            self.assertLess(name, "colorcraft.py", name)
        metadata = (ROOT / "metadata.ini").read_text(encoding="utf-8")
        self.assertNotIn("colorcraft", metadata.lower())

    def test_forge_runs_process_before_every_sampling_in_load_order(self):
        """The ordering above relies on Forge's ScriptRunner: its effective process_before_every_sampling (the
        last definition in the class — Forge 2.29.2 defines it twice) walks ``self.alwayson_scripts``, i.e.
        load order, not the user-sortable callback order. If Forge changes that, re-check the ordering."""
        import ast
        source = (S.FORGE / "modules" / "scripts.py").read_text(encoding="utf-8")
        runner = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "ScriptRunner")
        methods = [n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "process_before_every_sampling"]
        loop = next(n for n in ast.walk(methods[-1]) if isinstance(n, ast.For))
        self.assertEqual(ast.unparse(loop.iter), "self.alwayson_scripts")

    def test_detach_removes_only_our_callback(self):
        ours = lambda a: a["denoised"]  # noqa: E731
        setattr(ours, hook.OWNER_ATTR, hook.OWNER)
        foreign = lambda a: a["denoised"]  # noqa: E731
        unet = S.FakeUnet({"transformer_options": {}, "sampler_post_cfg_function": [foreign, ours]})
        p = _p(unet=unet, extra_generation_params={spec.INFOTEXT_KEY: "kept"})
        self.assertTrue(hook.detach_owned(p))
        self.assertEqual(p.sd_model.forge_objects.unet.model_options["sampler_post_cfg_function"], [foreign])
        self.assertEqual(p.extra_generation_params, {spec.INFOTEXT_KEY: "kept"})
        p2 = _p()
        self.assertFalse(hook.detach_owned(p2))
        self.assertEqual(p2.sd_model.forge_objects.unet.clones, 0)

    def test_metadata_written_at_attach(self):
        p = _p()
        _attach(p, _config_args(EXPOSURE))
        self.assertEqual(p.extra_generation_params[spec.INFOTEXT_KEY], "v1;mods=I;I.start=0;I.end=1;I.exposure=0.4")
        self.assertEqual(p.extra_generation_params[spec.STATUS_KEY],
                         "base: krea2 (Wan21); mods I; pending model evaluation")

    def test_no_unet_is_reported(self):
        p = _p()
        p.sd_model.forge_objects.unet = None
        self.assertIsNone(_attach(p, _config_args(EXPOSURE)))
        self.assertIn("no UNet", p.extra_generation_params[spec.STATUS_KEY])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        S.require_forge(self)

    def _attached(self, mods, family="krea2", steps=10, **p_fields):
        p = _p(family, **p_fields)
        callback = _attach(p, _config_args(*mods))
        unet = p.sd_model.forge_objects.unet
        unet.model_options["transformer_options"]["sampling_sigmas"] = _sigmas(steps)
        return p, callback, unet

    def test_outside_the_window_is_the_incoming_object(self):
        p, callback, unet = self._attached([Mod("advanced", {"start": 0.5, "end": 0.75, "exposure": 0.4})])
        sigmas = _sigmas(10)
        for index in (0, 1, 2, 9):
            x0 = S.latent("krea2", index)
            self.assertIs(callback(S.forge_args(x0, float(sigmas[index]), unet)), x0)
        x0 = S.latent("krea2", 5)
        self.assertIsNot(callback(S.forge_args(x0, float(sigmas[5]), unet)), x0)
        self.assertIn("5 evals, 1 edited", p.extra_generation_params[spec.STATUS_KEY])

    def test_no_in_place_writes(self):
        mods = [Mod("advanced", {"start": 0.0, "end": 1.0, "exposure": 0.3, "contrast": 0.4, "vibrance": 0.5,
                                 "color_shift_amount": 0.3, "color_shift_red": 0.5}, mask="M1"),
                Mod("shift", {"start": 0.0, "end": 1.0, "color_shift_amount": 0.4, "color_shift_green": 0.3})]
        p = _p()
        callback = _attach(p, _config_args(*mods, masking=True,
                                           leaves={"M1": {"mask_axis": "hue", "mask_mode": "range", "blur": 16.0}}))
        unet = p.sd_model.forge_objects.unet
        unet.model_options["transformer_options"]["sampling_sigmas"] = _sigmas(10)
        for sigma in S.midpoints(_sigmas(10)):
            x0 = S.latent("krea2", 7)
            args = S.forge_args(x0, sigma, unet)
            saved = {k: (v.clone(), v._version) for k, v in args.items() if torch.is_tensor(v)}
            out = callback(args)
            self.assertIsNot(out, x0)
            for key, (value, version) in saved.items():
                self.assertTrue(torch.equal(args[key], value), key)
                self.assertEqual(args[key]._version, version, key)

    def test_half_precision_is_graded_in_float32_and_returned_in_its_dtype(self):
        mods = [Mod("advanced", {"start": 0.0, "end": 1.0, "exposure": 0.3, "contrast": 0.4, "vibrance": 0.5,
                                 "temperature": -0.2, "chroma_contrast": 0.3})]
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                p, callback, unet = self._attached(mods)
                sigma = float(_sigmas(10)[4])
                x0 = S.latent("krea2", 11, dtype=dtype)
                out = callback(S.forge_args(x0, sigma, unet))
                self.assertEqual(out.dtype, dtype)
                self.assertEqual(out.shape, x0.shape)
                reference = callback(S.forge_args(x0.float(), sigma, unet))
                self.assertEqual(reference.dtype, torch.float32)
                self.assertTrue(torch.equal(out, reference.to(dtype)))
                self.assertFalse(torch.equal(out, x0))

    def test_an_exception_passes_the_input_through_and_is_reported(self):
        p, callback, unet = self._attached([EXPOSURE])
        x0 = S.latent("krea2", 1)
        with mock.patch.object(hook, "apply_chain", side_effect=RuntimeError("boom")), \
                mock.patch.object(hook, "_log") as log:
            self.assertIs(callback(S.forge_args(x0, 0.5, unet)), x0)
            self.assertIs(callback(S.forge_args(x0, 0.4, unet)), x0)
        self.assertEqual(log.call_count, 1)
        status = p.extra_generation_params[spec.STATUS_KEY]
        self.assertIn("error RuntimeError x2 (input passed through)", status)
        # the pass carries on once the cause is gone
        self.assertIsNot(callback(S.forge_args(x0, 0.5, unet)), x0)

    def test_a_non_finite_result_is_passed_through(self):
        p, callback, unet = self._attached([EXPOSURE])
        x0 = S.latent("krea2", 1)
        with mock.patch.object(hook, "apply_chain", side_effect=lambda *a, **k: torch.full_like(x0, float("nan"))):
            self.assertIs(callback(S.forge_args(x0, 0.5, unet)), x0)
        self.assertIn("non-finite result passed through x1", p.extra_generation_params[spec.STATUS_KEY])

    def test_anchors_are_encoded_once_before_sampling(self):
        mods = [Mod("advanced", {"start": 0.0, "end": 1.0, "contrast": 0.3, "color_shift_amount": 0.4,
                                 "color_shift_red": 0.5, "pass": "Both"}),
                Mod("basic", {"start": 0.0, "end": 1.0, "contrast": -0.2, "color_shift_amount": 0.3,
                              "color_shift_red": 0.5, "pass": "Both"}),
                Mod("shift", {"start": 0.0, "end": 1.0, "color_shift_amount": 0.3, "color_shift_blue": 0.2,
                              "pass": "Both"})]
        p = _p()
        vae = p.sd_model.forge_objects.vae
        callback = _attach(p, _config_args(*mods))
        # neutral grey (contrast) + (0.5, 0, 0, 0) + (0, 0, 0.2, 0)
        self.assertEqual(vae.calls, 3)
        unet = p.sd_model.forge_objects.unet
        unet.model_options["transformer_options"]["sampling_sigmas"] = _sigmas(10)
        for sigma in S.midpoints(_sigmas(10)):
            callback(S.forge_args(S.latent("krea2", 3), sigma, unet))
        self.assertEqual(vae.calls, 3)
        # the hires pass of the same request reuses them
        p.is_hr_pass = True
        p.sd_model.forge_objects.unet = S.FakeUnet()
        self.assertIsNotNone(_attach(p, _config_args(*mods)))
        self.assertEqual(vae.calls, 3)
        # a hires checkpoint with another VAE encodes its own
        other = S.FakeVAE(16, 3)
        p.sd_model.forge_objects.vae = other
        p.sd_model.forge_objects.unet = S.FakeUnet()
        self.assertIsNotNone(_attach(p, _config_args(*mods)))
        self.assertEqual(other.calls, 3)

    def test_anchors_use_vae_encode_and_process_in_never_encode_first_stage(self):
        p = _p()
        p.sd_model.encode_first_stage = mock.Mock(side_effect=AssertionError("encode_first_stage"))
        p.sd_model.get_first_stage_encoding = mock.Mock(side_effect=AssertionError("get_first_stage_encoding"))
        latent_format = p.sd_model.model_config.latent_format
        with mock.patch.object(type(latent_format), "process_in", autospec=True,
                               side_effect=lambda self, x: (x - self.latents_mean) / self.latents_std) as process_in:
            _attach(p, _config_args(Mod("advanced", {"contrast": 0.3})))
        self.assertEqual(p.sd_model.forge_objects.vae.calls, 1)
        self.assertEqual(process_in.call_count, 1)
        shaped = process_in.call_args.args[1]
        self.assertEqual(tuple(shaped.shape), (1, 16, 1, 1, 1))   # Wan21: rank 3 from the 5-D encode

    def test_anchor_failure_turns_contrast_and_shift_off_only(self):
        p = _p()
        p.sd_model.forge_objects.vae = types.SimpleNamespace(encode=mock.Mock(side_effect=RuntimeError("no vae")))
        callback = _attach(p, _config_args(Mod("advanced", {"start": 0.0, "end": 1.0, "contrast": 0.4,
                                                            "exposure": 0.3})))
        self.assertIsNotNone(callback)
        status = p.extra_generation_params[spec.STATUS_KEY]
        self.assertIn("anchors failed (RuntimeError): Contrast/Color Shift off", status)
        unet = p.sd_model.forge_objects.unet
        unet.model_options["transformer_options"]["sampling_sigmas"] = _sigmas(10)
        x0 = S.latent("krea2", 2)
        out = callback(S.forge_args(x0, 0.5, unet))
        self.assertIsNot(out, x0)
        self.assertNotIn("error", p.extra_generation_params[spec.STATUS_KEY])
        # contrast-only on a model without a basis: nothing left to apply, nothing attached
        p2 = _p(None, latent_format=S.latent_formats().SDXL(), vae=types.SimpleNamespace(
            encode=mock.Mock(side_effect=RuntimeError("no vae"))))
        self.assertIsNone(_attach(p2, _config_args(Mod("basic", {"contrast": 0.4}))))
        self.assertEqual(p2.sd_model.forge_objects.unet.clones, 0)
        self.assertIn("nothing to apply", p2.extra_generation_params[spec.STATUS_KEY])

    def test_foreign_sampling_runs_are_passed_through(self):
        p, callback, unet = self._attached([EXPOSURE])
        x0 = S.latent("krea2", 1)
        self.assertIsNot(callback(S.forge_args(x0, 0.5, unet)), x0)
        unet.model_options["transformer_options"]["sampling_sigmas"] = _sigmas(6)   # another run's list
        self.assertIs(callback(S.forge_args(x0, 0.5, unet)), x0)
        self.assertIn("other sampling run passed through x1", p.extra_generation_params[spec.STATUS_KEY])

    def test_a_stale_list_is_not_this_samplers(self):
        """A k-diffusion run publishes a new list; the one left in the options by an earlier run is not used."""
        stale = _sigmas(30)
        unet = S.FakeUnet({"transformer_options": {"sampling_sigmas": stale}})
        p = _p(unet=unet)
        callback = _attach(p, _config_args(EXPOSURE))
        clone = p.sd_model.forge_objects.unet
        self.assertIs(clone.model_options["transformer_options"]["sampling_sigmas"], stale)
        x0 = S.latent("krea2", 1)
        self.assertIs(callback(S.forge_args(x0, 0.5, clone)), x0)
        self.assertIn("no sigma list: passed through", p.extra_generation_params[spec.STATUS_KEY])
        clone.model_options["transformer_options"]["sampling_sigmas"] = _sigmas(10)
        self.assertIsNot(callback(S.forge_args(x0, 0.5, clone)), x0)

    def test_compvis_samplers_use_upstreams_forge_step_counter(self):
        """DDIM/PLMS (classic_ddim_eps_estimation) publish no sigma list: the step comes from the denoiser,
        with upstream's Forge arithmetic (scripts/colorcraft.py denoiser_callback)."""
        denoiser = types.SimpleNamespace(classic_ddim_eps_estimation=True, step=0, total_steps=10, steps=10)
        stale = _sigmas(30)
        unet = S.FakeUnet({"transformer_options": {"sampling_sigmas": stale}})
        p = _p(unet=unet, sampler=types.SimpleNamespace(model_wrap_cfg=denoiser))
        mod = Mod("advanced", {"start": 0.2, "end": 0.8, "exposure": 0.4})
        callback = _attach(p, _config_args(mod))
        clone = p.sd_model.forge_objects.unet
        entries = spec.build_chain(spec.config_from_args(_config_args(mod)), False).entries
        expected_schedule = engine.build_entries(entries, 10)[0][0]
        for step in range(10):
            denoiser.step = step
            x0 = S.latent("krea2", step)
            out = callback(S.forge_args(x0, 0.123, clone))
            current, actual = schedule.forge_step_position(0, 0, step, 10, 10)
            self.assertEqual((current, actual), (step, 10))
            value = float(expected_schedule[current])
            if value == 0:
                self.assertIs(out, x0)
            else:
                reference = engine.apply_chain(engine.build_entries(entries, 10), x0, [value], "krea2",
                                               hook.basis_on("krea2", x0.device, x0.dtype), None, 8)
                self.assertTrue(torch.equal(out, reference), step)

    def test_forge_step_position_is_upstreams_arithmetic(self):
        # step = max(sampling_step, denoiser.step); steps = max(total, denoiser.total_steps);
        # actual = steps - max(steps // denoiser.steps - 1, 0); current = min(step, actual - 1)
        self.assertEqual(schedule.forge_step_position(3, 20, 5, 20, 20), (5, 20))
        self.assertEqual(schedule.forge_step_position(0, 0, 7, 40, 20), (7, 39))   # 2 calls per step
        self.assertEqual(schedule.forge_step_position(0, 0, 50, 20, 20), (19, 20))
        self.assertEqual(schedule.forge_step_position(0, 0, 0, 0, 0), (None, None))

    def test_detail_daemons_scaled_sigma(self):
        """Pre-DD on (default): the schedule follows the sampler's sigma; off: the sigma the model saw."""
        # exponent 1: a ramp, so a shifted lookup reads a different value (exponent 0 is a flat plateau)
        mod = Mod("advanced", {"start": 0.0, "end": 1.0, "advanced": True, "exponent": 1.0, "exposure": 0.4})
        sigmas = _sigmas(10)
        original, scaled = sigmas[3].reshape(1).clone(), (sigmas[3] * 0.9).reshape(1).clone()
        x0 = S.latent("krea2", 4)
        results = {}
        for pre_dd in (True, False):
            p = _p()
            with mock.patch.object(hook, "_option", side_effect=lambda name, default: pre_dd if name == hook.OPT_PRE_DD else default):
                callback = _attach(p, _config_args(mod))
            unet = p.sd_model.forge_objects.unet
            unet.model_options["transformer_options"]["sampling_sigmas"] = sigmas
            args = S.forge_args(x0, 0.0, unet)
            args["sigma"] = scaled.expand(2).clone()
            dave_gate.note_pre_dd_sigma(original, args["sigma"])
            try:
                results[pre_dd] = callback(args)
            finally:
                dave_gate.note_pre_dd_sigma(None)
            self.assertEqual(p.extra_generation_params[hook.INFOTEXT_PRE_DD], str(pre_dd))
            plain = callback(S.forge_args(x0, float(original if pre_dd else scaled), unet))
            self.assertTrue(torch.equal(results[pre_dd], plain))
        self.assertFalse(torch.equal(results[True], results[False]))

    def test_pre_dd_key_only_when_detail_daemon_scaled(self):
        p, callback, unet = self._attached([EXPOSURE])
        callback(S.forge_args(S.latent("krea2", 1), 0.5, unet))
        self.assertNotIn(hook.INFOTEXT_PRE_DD, p.extra_generation_params)


class PassTests(unittest.TestCase):
    def setUp(self):
        S.require_forge(self)

    def test_base_hires_both(self):
        mods = [Mod("advanced", {"pass": "Base", "exposure": 0.3}), Mod("advanced", {"pass": "Hires", "tint": 0.3}),
                Mod("advanced", {"pass": "Both", "contrast": 0.3})]
        p = _p()
        _attach(p, _config_args(*mods))
        self.assertIn("base: krea2 (Wan21); mods I+III", p.extra_generation_params[spec.STATUS_KEY])
        p.is_hr_pass = True
        p.sd_model.forge_objects.unet = S.FakeUnet()
        _attach(p, _config_args(*mods))
        self.assertEqual(p.extra_generation_params[spec.STATUS_KEY],
                         "base: krea2 (Wan21); mods I+III; pending model evaluation | "
                         "hires: krea2 (Wan21); mods II+III; pending model evaluation")

    def test_a_new_base_pass_starts_a_new_record(self):
        mods = [Mod("advanced", {"pass": "Both", "exposure": 0.3})]
        p = _p()
        _attach(p, _config_args(*mods))
        p.is_hr_pass = True
        p.sd_model.forge_objects.unet = S.FakeUnet()
        _attach(p, _config_args(*mods))
        p.is_hr_pass = False   # next n_iter batch, same request
        p.sd_model.forge_objects.unet = S.FakeUnet()
        _attach(p, _config_args(*mods))
        self.assertEqual(p.extra_generation_params[spec.STATUS_KEY],
                         "base: krea2 (Wan21); mods I; pending model evaluation")

    def test_xyz_overrides_modifier_one(self):
        config = spec.default_config()
        config.modifiers[0]["active"] = False
        p = _p()
        setattr(p, spec.XYZ_ATTR, {"enabled": "True", "exposure": 0.25, "strength": 0.5})
        self.assertIsNotNone(_attach(p, spec.config_to_args(config)))
        self.assertEqual(p.extra_generation_params[spec.INFOTEXT_KEY], "v1;mods=I;I.strength=0.5;I.exposure=0.25")
        p2 = _p()
        setattr(p2, spec.XYZ_ATTR, {"enabled": "False"})
        self.assertIsNone(_attach(p2, _config_args(EXPOSURE)))

    def test_compact_api_argument_attaches_like_the_positional_ones(self):
        """API: the infotext string or a dict of argument paths in the first slot (the other slots hold the
        panel defaults, as Forge's API fills them) grades exactly like the 579 positional values."""
        mods = [Mod("advanced", {"start": 0.0, "end": 1.0, "exposure": 0.3, "contrast": 0.2}),
                Mod("chroma", {"start": 0.0, "end": 1.0, "vibrance": 0.4})]
        positional = _config_args(*mods)
        text = spec.to_infotext(spec.config_from_args(positional))
        rest = spec.config_to_args(spec.default_config())[1:]
        mapping = {"I.exposure": 0.3, "I.contrast": 0.2, "I.start": 0.0, "I.end": 1.0, "II.active": True,
                   "II.kind": "Chroma", "II.vibrance": 0.4, "II.start": 0.0, "II.end": 1.0}
        sigma = float(_sigmas(10)[3])
        outputs, infotexts = [], []
        for args in (positional, [text] + rest, [mapping] + rest):
            p = _p()
            callback = _attach(p, args)
            self.assertIsNotNone(callback)
            unet = p.sd_model.forge_objects.unet
            unet.model_options["transformer_options"]["sampling_sigmas"] = _sigmas(10)
            outputs.append(callback(S.forge_args(S.latent("krea2", 3), sigma, unet)))
            infotexts.append(p.extra_generation_params[spec.INFOTEXT_KEY])
        self.assertEqual(infotexts[1:], [infotexts[0], infotexts[0]])
        for out in outputs[1:]:
            self.assertTrue(torch.equal(out, outputs[0]))
        p = _p()
        self.assertIsNone(_attach(p, [{"enabled": False, "I.exposure": 0.3}] + rest))
        self.assertEqual(p.extra_generation_params, {})
        p = _p()                                         # XYZ overrides still apply on top of the compact form
        setattr(p, spec.XYZ_ATTR, {"exposure": 0.5})
        self.assertIsNotNone(_attach(p, [text] + rest))
        self.assertIn("I.exposure=0.5", p.extra_generation_params[spec.INFOTEXT_KEY].split(";"))


class _Format:
    """A latent format stand-in (the anchor cache keys on the format's class name)."""

    def process_in(self, latent):
        return latent * 0.5 + 0.25


class AnchorCacheTests(unittest.TestCase):
    """Anchors are kept across requests for the same VAE (XYZ cells, API calls): bit-identical to a fresh
    encode, bounded, and never reused for another VAE, other weight patches, dtype, device or format."""

    COLORS = [engine.NEUTRAL_COLOR, (0.5, 0.0, 0.0, 0.0), (0.0, 0.2, -0.1, 0.1)]
    CPU = torch.device("cpu")

    def setUp(self):
        hook.clear_anchor_cache()
        self.addCleanup(hook.clear_anchor_cache)

    def _encode(self, vae, colors=None, fmt=None, request=None):
        return hook.encode_anchors(vae, fmt or _Format(), colors or self.COLORS, {} if request is None else request,
                                   device=self.CPU)

    def test_a_cached_anchor_is_the_uncached_one(self):
        vae = S.FakeVAE(16, 3)
        first = self._encode(vae)
        again = self._encode(vae)                                   # the next request
        self.assertEqual(vae.calls, 3)
        hook.clear_anchor_cache()
        fresh = self._encode(vae)                                   # the uncached path
        self.assertEqual(vae.calls, 6)
        for key, anchor in fresh.items():
            self.assertIs(again[key], first[key])
            self.assertTrue(torch.equal(again[key], anchor))
            self.assertEqual(again[key].dtype, anchor.dtype)

    def test_another_vae_patches_dtype_device_or_format_encode_their_own(self):
        vae = S.FakeVAE(16, 3)
        vae.patcher = types.SimpleNamespace(patches_uuid="a")
        vae.vae_dtype = torch.bfloat16
        one = self.COLORS[:1]
        self._encode(vae, one)
        other = S.FakeVAE(16, 3)
        self._encode(other, one)
        self.assertEqual((vae.calls, other.calls), (1, 1))         # another VAE object
        vae.patcher.patches_uuid = "b"                              # weight patches added to this VAE
        self._encode(vae, one)
        vae.vae_dtype = torch.float16                               # another compute dtype
        self._encode(vae, one)
        self._encode(vae, one, fmt=types.new_class("Flux", (_Format,))())   # another latent format
        self.assertEqual(vae.calls, 4)
        self._encode(vae, one)                                      # nothing changed: cached
        self.assertEqual(vae.calls, 4)
        module = vae.first_stage_model
        colour = hook.color_key(one[0])
        self.assertNotEqual(hook._anchor_key(vae, module, _Format(), torch.device("cpu"), colour),
                            hook._anchor_key(vae, module, _Format(), torch.device("cuda", 0), colour))

    def test_an_entry_never_answers_for_a_vae_that_is_gone(self):
        vae = S.FakeVAE(16, 3)
        anchors = self._encode(vae, self.COLORS[:1])
        (key, (_, anchor)), = hook._ANCHOR_CACHE.items()
        # as if the encoder it was made with had died and this one got its id
        hook._ANCHOR_CACHE[key] = (weakref.ref(S.FakeEncoderModule()), torch.zeros_like(anchor))
        again = self._encode(vae, self.COLORS[:1])
        self.assertEqual(vae.calls, 2)
        self.assertTrue(torch.equal(next(iter(again.values())), next(iter(anchors.values()))))
        gone = S.FakeVAE(16, 3)
        self._encode(gone)
        self.assertEqual(len(hook._ANCHOR_CACHE), 4)
        del gone                                                    # a model reload drops the old VAE
        gc.collect()
        self._encode(vae, self.COLORS[1:2])
        self.assertEqual(len(hook._ANCHOR_CACHE), 2)                # its three anchors went with it

    def test_bounded_least_recently_used_first_out(self):
        vae = S.FakeVAE(16, 3)
        colors = [(i / 100.0, 0.0, 0.0, 0.0) for i in range(hook.ANCHOR_CACHE_LIMIT + 8)]
        self._encode(vae, colors)
        self.assertEqual(len(hook._ANCHOR_CACHE), hook.ANCHOR_CACHE_LIMIT)
        self._encode(vae, colors[-1:])
        self.assertEqual(vae.calls, len(colors))                    # the newest is kept
        self._encode(vae, colors[:1])
        self.assertEqual(vae.calls, len(colors) + 1)                # the oldest was dropped

    def test_an_encoder_without_weak_references_is_cached_for_the_request_only(self):
        vae = S.FakeVAE(16, 3)
        vae.first_stage_model = object()
        request = {}
        self._encode(vae, self.COLORS[:1], request=request)
        self._encode(vae, self.COLORS[:1], request=request)        # the hires pass of the same request
        self.assertEqual(vae.calls, 1)
        self._encode(vae, self.COLORS[:1])                          # the next request
        self.assertEqual(vae.calls, 2)
        self.assertEqual(len(hook._ANCHOR_CACHE), 0)

    def test_xyz_cells_with_the_same_vae_do_not_encode_again(self):
        """Each XYZ cell is a new request on the same model: the second one encodes nothing and grades exactly
        like the first, and like a request after the cache was cleared (the uncached path)."""
        S.require_forge(self)
        mods = [Mod("advanced", {"start": 0.0, "end": 1.0, "contrast": 0.3, "color_shift_amount": 0.4,
                                 "color_shift_red": 0.5}),
                Mod("shift", {"start": 0.0, "end": 1.0, "color_shift_amount": 0.3, "color_shift_blue": 0.2})]
        first = _p()
        vae = first.sd_model.forge_objects.vae
        sigma = float(_sigmas(10)[2])

        def grade(p):
            callback = _attach(p, _config_args(*mods))
            unet = p.sd_model.forge_objects.unet
            unet.model_options["transformer_options"]["sampling_sigmas"] = _sigmas(10)
            return callback(S.forge_args(S.latent("krea2", 4), sigma, unet))

        outputs = [grade(first)]
        self.assertEqual(vae.calls, 3)
        outputs.append(grade(_p(vae=vae)))
        self.assertEqual(vae.calls, 3)
        hook.clear_anchor_cache()
        outputs.append(grade(_p(vae=vae)))
        self.assertEqual(vae.calls, 6)
        for out in outputs[1:]:
            self.assertTrue(torch.equal(out, outputs[0]))


class OffsetTests(unittest.TestCase):
    """img2img and the hires pass sample ``sigmas[steps - t_enc - 1:]``; the node is handed that slice."""

    SCENARIO = S.SCENARIOS[[s.label for s in S.SCENARIOS].index("stack-luma-chroma-punch-shaping")]

    def setUp(self):
        S.require_forge(self)

    def test_img2img_matches_the_node_on_the_sliced_list(self):
        checked = 0
        for steps, denoise in ((28, 0.5), (20, 0.75), (30, 0.35)):
            for fix_steps in (False, True):
                p = _p(request_cls=S.StableDiffusionProcessingImg2Img, steps=steps, denoising_strength=denoise)
                total, t_enc = S.setup_img2img_steps(fix_steps)(p, None)
                full = S.flow_sigmas(total)
                walked = full[total - t_enc - 1:]
                with self.subTest(steps=steps, denoise=denoise, fix_steps=fix_steps):
                    moved, total_evals, _ = S.parity_run(
                        self, "krea2", self.SCENARIO, walked, full=full,
                        p_fields={"steps": steps, "denoising_strength": denoise},
                        modules_stub=S.forge_modules(fix_steps), request_cls=S.StableDiffusionProcessingImg2Img)
                    self.assertGreater(moved, 0)
                    checked += 1
        self.assertEqual(checked, 6)

    def test_hires_pass_matches_the_node_on_the_sliced_list(self):
        for steps, hr_steps, denoise in ((28, 0, 0.5), (28, 10, 0.5), (20, 28, 0.7)):
            p = _p(is_hr_pass=True, steps=steps, hr_second_pass_steps=hr_steps, denoising_strength=denoise)
            total, t_enc = S.setup_img2img_steps()(p, hr_steps or steps)
            full = S.flow_sigmas(total)
            walked = full[total - t_enc - 1:]
            scenario = Scenario(self.SCENARIO.label, [Mod(m.kind, dict(m.values, **{"pass": "Hires"}), m.mask)
                                                      for m in self.SCENARIO.mods])
            with self.subTest(steps=steps, hr_steps=hr_steps, denoise=denoise):
                moved, _, p_out = S.parity_run(
                    self, "zimage", scenario, walked, full=full,
                    p_fields={"is_hr_pass": True, "steps": steps, "hr_second_pass_steps": hr_steps,
                              "denoising_strength": denoise})
                self.assertGreater(moved, 0)
                self.assertIn("hires: zimage (Flux)", p_out.extra_generation_params[spec.STATUS_KEY])

    def test_the_slice_matters(self):
        """Building the schedule on Forge's whole list (no offset) would read different values."""
        p = _p(request_cls=S.StableDiffusionProcessingImg2Img, steps=28, denoising_strength=0.5)
        total, t_enc = S.setup_img2img_steps()(p, None)
        full = S.flow_sigmas(total)
        walked = full[total - t_enc - 1:]
        mod = self.SCENARIO.mods[0]
        sched = spec.schedule_of(dict(spec.default_config().modifiers[0], **mod.values))
        on_walked = schedule.make_schedule(len(walked) - 1, amount=sched["strength"],
                                           **{k: sched[k] for k in engine._SCHEDULE_KEYS})
        on_full = schedule.make_schedule(len(full) - 1, amount=sched["strength"],
                                         **{k: sched[k] for k in engine._SCHEDULE_KEYS})
        values_walked = [schedule.sigma_to_value(float(s), walked, on_walked) for s in walked[:-1]]
        values_full = [schedule.sigma_to_value(float(s), full, on_full) for s in walked[:-1]]
        self.assertNotEqual(values_walked, values_full)

    def test_unknown_offset_finds_the_runs_first_sigma(self):
        """Without Forge (offset None) the walked part starts at the first sigma the run evaluates."""
        p = _p(request_cls=S.StableDiffusionProcessingImg2Img, steps=28, denoising_strength=0.5)
        total, t_enc = S.setup_img2img_steps()(p, None)
        full = S.flow_sigmas(total)
        walked = full[total - t_enc - 1:]
        moved, _, p_out = S.parity_run(
            self, "krea2", self.SCENARIO, walked, full=full,
            p_fields={"steps": 28, "denoising_strength": 0.5}, modules_stub=types.ModuleType("modules"),
            request_cls=S.StableDiffusionProcessingImg2Img)
        self.assertGreater(moved, 0)
        self.assertIn(f"offset from first sigma ({total - t_enc - 1})", p_out.extra_generation_params[spec.STATUS_KEY])


class UnsupportedModelTests(unittest.TestCase):
    """SD 1.5 / SDXL: only the vector-free controls (contrast, colour shift) apply — like the node."""

    def setUp(self):
        S.require_forge(self)

    def test_sdxl_matches_the_node(self):
        formats = S.latent_formats()
        origin = S.load_origin()
        scenario = Scenario("sdxl", [
            Mod("advanced", {"start": 0.0, "end": 1.0, "exposure": 0.5, "contrast": 0.4, "vibrance": 0.3,
                             "color_shift_amount": 0.3, "color_shift_red": 0.4}, mask="M1"),
            Mod("basic", {"start": 0.0, "end": 1.0, "contrast": -0.3, "color_shift_amount": 0.2,
                          "color_shift_blue": 0.5}),
            Mod("luma", {"start": 0.0, "end": 1.0, "exposure": 0.6}),
        ], leaves={"M1": {"mask_axis": "exposure"}}, masking=True)
        for name in ("SDXL", "SD15"):
            latent_format = getattr(formats, name)()
            vae = S.FakeVAE(4, 2)
            p = _p(None, latent_format=latent_format, vae=vae)
            callback = _attach(p, S.scenario_args(scenario))
            self.assertIsNotNone(callback)
            status = p.extra_generation_params[spec.STATUS_KEY]
            self.assertIn(f"no basis ({name})", status)
            self.assertIn("vector controls off on this model: only Contrast and Color Shift apply", status)
            self.assertIn("mods I+II+III", status)
            unet = p.sd_model.forge_objects.unet
            walked = _sigmas(10)
            unet.model_options["transformer_options"]["sampling_sigmas"] = walked
            oracle = S.oracle_callback(origin, S.oracle_chain(scenario, origin.nodes), S.FakeVAE(4, 2), walked,
                                       latent_format)
            for k, sigma in enumerate(S.midpoints(walked)):
                g = torch.Generator().manual_seed(k)
                x0 = torch.randn(2, 4, 16, 16, generator=g)
                ours = callback(S.forge_args(x0.clone(), sigma, unet))
                theirs = oracle(x0.clone(), sigma)
                with self.subTest(model=name, eval=k):
                    self.assertTrue(torch.equal(ours, theirs))
                    self.assertFalse(torch.equal(ours, x0))

    def test_vector_only_settings_attach_nothing_but_say_why(self):
        p = _p(None, latent_format=S.latent_formats().SDXL(), vae=S.FakeVAE(4, 2))
        self.assertIsNone(_attach(p, _config_args(Mod("luma", {"exposure": 0.5}))))
        self.assertEqual(p.sd_model.forge_objects.unet.clones, 0)
        status = p.extra_generation_params[spec.STATUS_KEY]
        self.assertIn("vector controls off on this model", status)
        self.assertIn("nothing to apply", status)
        self.assertIn(spec.INFOTEXT_KEY, p.extra_generation_params)


class ScriptSafetyTests(unittest.TestCase):
    def test_process_before_every_sampling_never_raises(self):
        script_module = _load_script()
        p = types.SimpleNamespace(extra_generation_params={spec.INFOTEXT_KEY: "v1;mods=I"})
        with mock.patch.object(script_module.hook, "process", side_effect=ValueError("bad")), \
                mock.patch.object(script_module, "_log"):
            script_module.Colorcraft().process_before_every_sampling(p, *_config_args(EXPOSURE))
        self.assertEqual(p.extra_generation_params[spec.STATUS_KEY], "not applied: ValueError")


def _load_script():
    modules = types.ModuleType("modules")
    modules.__path__ = []

    class Script:
        pass

    modules.scripts = types.SimpleNamespace(Script=Script, AlwaysVisible=object(), scripts_data=[])
    modules.script_callbacks = types.SimpleNamespace(on_before_ui=lambda fn: None, on_ui_settings=lambda fn: None)
    with mock.patch.dict(sys.modules, {"modules": modules, "modules.scripts": modules.scripts,
                                       "modules.script_callbacks": modules.script_callbacks}):
        spec_ = importlib.util.spec_from_file_location("_test_colorcraft_script_hook", ROOT / "scripts" / "colorcraft.py")
        module = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
