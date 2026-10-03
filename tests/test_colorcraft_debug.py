"""Colorcraft Debug panel — capture during sampling, render on demand (upstream's Forge Debug panel).

* sam3ext/colorcraft/debug.py is upstream's lib_colorcraft/debug.py unchanged (syntax trees compared).
* The renderer is upstream's ``build_debug_gallery_images`` (scripts/colorcraft.py:820-905): the pinned
  method itself is executed against a fake ``self`` and must give the same captions and pixels.
* The hook captures the pre-edit x0 once, at the debug step, only with the panel's capture on, on a
  model with a basis; batches of one size accumulate, another size starts fresh.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

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

from sam3ext.colorcraft import debug as debug_mod  # noqa: E402
from sam3ext.colorcraft import debug_panel, hook, panel_state, spec, ui  # noqa: E402

Mod, Scenario = S.Mod, S.Scenario


def _fake_decode(latent, width, height, is_5d=False):
    """Deterministic stand-in for the VAE decode: first three channels squashed to RGB."""
    images = []
    for b in range(latent.shape[0]):
        rgb = torch.sigmoid(latent[b, :3].float()).permute(1, 2, 0).numpy()
        images.append(Image.fromarray((rgb * 255).astype("uint8"), mode="RGB").resize((width, height),
                                                                                     Image.BILINEAR))
    return images


def _upstream_renderer(origin):
    """Upstream's ``Script.build_debug_gallery_images`` (pinned copy), executable on a fake ``self``."""
    tree = ast.parse(S.origin_body(S.ORIGIN_DIR / "scripts" / "colorcraft.py"))
    script = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Script")
    method = next(node for node in script.body
                  if isinstance(node, ast.FunctionDef) and node.name == "build_debug_gallery_images")
    scope = {
        "torch": torch, "print": lambda *a, **k: None,
        "DEBUG_CAPTION_MARKER": "​[colorcraft-debug]", "VAE_DOWNSCALE_FACTOR": 8,
        "decode_debug_latent": _fake_decode,
        "chroma_axes": origin.vectors.chroma_axes,
        "compute_hue_projection": origin.masking.compute_hue_projection,
        "compute_saturation_projection": origin.masking.compute_saturation_projection,
        "resolve_mask_tensor": origin.masking.resolve_mask_tensor,
    }
    for name in ("DEBUG_COMPOSITE_COLORS", "downscale_latent_for_storage", "render_hue_images",
                 "compute_axis_projection", "debug_tensor_to_images", "build_curve_infos", "collect_leaves",
                 "composite_mask_images"):
        scope[name] = getattr(origin.debug, name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "upstream build_debug_gallery_images", "exec"), scope)  # noqa: S102
    return scope["build_debug_gallery_images"]


def _mask_config():
    config = spec.default_config()
    config.leaves[0].update(mask_axis="hue", mask_mode="range", mask_center=0.3, mask_width=0.6, blur=8.0)
    config.leaves[1].update(mask_axis="clarity", mask_mode="split", mask_hardness=3.0)
    config.leaves[2].update(mask_axis="saturation", mask_mode="protect range", normalize=True, contrast=1.5)
    config.combos[0].update(mask_a="M1", mask_b="M3", operation="and", spread=0.4)
    config.combos[1].update(mask_a="C1", mask_b="none")          # incomplete
    return config


class UnchangedDebugModuleTests(unittest.TestCase):
    def test_functions_and_tables_are_upstreams(self):
        origin = S.load_origin()
        theirs = ast.parse(inspect.getsource(origin.debug))
        ours = ast.parse(inspect.getsource(debug_mod))
        their_functions = {n.name: ast.dump(n) for n in theirs.body if isinstance(n, ast.FunctionDef)}
        our_functions = {n.name: ast.dump(n) for n in ours.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(our_functions, their_functions)
        for name in ("DEBUG_COMPOSITE_COLORS", "DEBUG_OVERLAY_COLORS", "DEBUG_AXIS_STYLES"):
            self.assertEqual(getattr(debug_mod, name), getattr(origin.debug, name))
        self.assertTrue((debug_mod.DEBUG_COLORMAP_LUT == origin.debug.DEBUG_COLORMAP_LUT).all())
        self.assertTrue((debug_mod.DEBUG_HUE_LUT == origin.debug.DEBUG_HUE_LUT).all())

    def test_panel_constants_are_upstreams(self):
        scope = S.upstream_forge_functions("DEBUG_AXIS_OPTIONS", "DEBUG_CAPTION_MARKER")
        self.assertEqual(debug_panel.DEBUG_AXIS_OPTIONS, scope["DEBUG_AXIS_OPTIONS"])
        self.assertEqual(debug_panel.DEBUG_CAPTION_MARKER, scope["DEBUG_CAPTION_MARKER"])

    def test_panel_widgets_are_upstreams(self):
        """Debug Step and the three dropdown defaults (upstream scripts/colorcraft.py:614-626)."""
        widgets = {}
        for node in ast.walk(ast.parse(S.origin_body(S.ORIGIN_DIR / "scripts" / "colorcraft.py"))):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id.startswith("gr_debug_") and isinstance(node.value, ast.Call)):
                widgets[node.targets[0].id] = {kw.arg: kw.value for kw in node.value.keywords}
        step = widgets["gr_debug_step"]
        field = spec.DEBUG_FIELDS[1]
        self.assertEqual((field.minimum, field.maximum, field.step, field.default, field.label),
                         tuple(ast.literal_eval(step[k]) for k in ("minimum", "maximum", "step", "value", "label")))
        self.assertEqual((debug_panel.DEBUG_STEP_MIN, debug_panel.DEBUG_STEP_MAX, debug_panel.DEBUG_STEP_DEFAULT),
                         (0, 50, 5))
        self.assertEqual(ast.literal_eval(widgets["gr_debug_axis_style"]["value"]), "colormap")
        self.assertEqual(ast.literal_eval(widgets["gr_debug_composite_color"]["value"]), "none")
        self.assertEqual(ast.literal_eval(widgets["gr_debug_overlay_color"]["value"]), "white")


class RendererParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.origin = S.load_origin()
        cls.upstream = staticmethod(_upstream_renderer(cls.origin))

    def _both(self, family, axes, masks, combos, composite, overlay="white", style="colormap"):
        g = torch.Generator().manual_seed(5)
        latent = torch.randn(2, S.FAMILIES[family][1], 12, 10, generator=g)
        capture = debug_panel.Capture(latent, False, family)
        leaf_specs, combo_specs = spec.build_mask_specs(_mask_config())
        warnings = []
        ours = debug_panel.build_debug_gallery_images(capture, axes, masks, combos, composite, overlay, style,
                                                      leaf_specs, combo_specs, decode=_fake_decode,
                                                      warn=warnings.append)
        fake_self = types.SimpleNamespace(
            debug_latent=latent, debug_latent_is_5d=False,
            cur_basis={k: v.clone() for k, v in hook.load_family_basis(family).items()},
            dev=self.origin.basis.resolve_dev({}, family))
        theirs = self.upstream(fake_self, axes, masks, combos, composite, overlay, style, leaf_specs, combo_specs)
        return ours, theirs, warnings

    def _assert_same(self, ours, theirs):
        self.assertEqual(len(ours), len(theirs))
        for (img_a, cap_a), (img_b, cap_b) in zip(ours, theirs):
            self.assertEqual(cap_a, cap_b)
            self.assertEqual(img_a.size, img_b.size)
            self.assertTrue(np.array_equal(np.asarray(img_a), np.asarray(img_b)), cap_a)

    def test_axes_masks_and_combos_render_like_upstream(self):
        for family in ("krea2", "zimage"):
            for composite in ("none", "red"):
                with self.subTest(family=family, composite=composite):
                    ours, theirs, warnings = self._both(
                        family, debug_panel.DEBUG_AXIS_OPTIONS, ["M1", "M2", "M3"], ["C1", "C2"], composite)
                    self._assert_same(ours, theirs)
                    # 14 axes + 3 masks + 1 complete combo, two images each (batch 2)
                    self.assertEqual(len(ours), 2 * (14 + 3 + 1))
                    self.assertTrue(all(caption.startswith(debug_panel.DEBUG_CAPTION_MARKER + " ") for _, caption in ours))
                    self.assertEqual(ours[0][0].size, (10 * 8, 12 * 8))
                    self.assertTrue(any("C2 has an unresolved reference" in w for w in warnings))
                    self.assertTrue(any("36 images" in w for w in warnings))

    def test_greyscale_and_overlay_none(self):
        ours, theirs, _ = self._both("krea2", ["exposure", "hue (weighted)"], ["M2"], [], "none", "none", "greyscale")
        self._assert_same(ours, theirs)

    def test_flux2_uses_its_own_downscale(self):
        g = torch.Generator().manual_seed(1)
        capture = debug_panel.Capture(torch.randn(1, 128, 6, 5, generator=g), False, "flux2")
        images = debug_panel.build_debug_gallery_images(capture, ["exposure"], [], [], "none", "white", "colormap",
                                                        {}, {}, warn=lambda *_: None)
        self.assertEqual(images[0][0].size, (5 * 16, 6 * 16))

    def test_nothing_captured_or_no_basis(self):
        warnings = []
        self.assertEqual(debug_panel.build_debug_gallery_images(None, ["exposure"], [], [], "none", "white",
                                                                "colormap", {}, {}, warn=warnings.append), [])
        capture = debug_panel.Capture(torch.zeros(1, 4, 4, 4), False, None)
        self.assertEqual(debug_panel.build_debug_gallery_images(capture, ["exposure"], [], [], "none", "white",
                                                                "colormap", {}, {}, warn=warnings.append), [])
        self.assertEqual(len(warnings), 2)

    def test_a_failed_decode_falls_back_to_plain_masks(self):
        capture = debug_panel.Capture(torch.randn(1, 16, 8, 8), False, "krea2")
        leaf_specs, combo_specs = spec.build_mask_specs(_mask_config())
        warnings = []

        def broken(*args):
            raise RuntimeError("no model")

        images = debug_panel.build_debug_gallery_images(capture, [], ["M1"], [], "red", "white", "colormap",
                                                        leaf_specs, combo_specs, decode=broken,
                                                        warn=warnings.append)
        self.assertEqual(len(images), 1)
        self.assertTrue(any("could not decode" in w for w in warnings))

    def test_refresh_reads_the_panels_current_mask_values(self):
        config = _mask_config()
        mask_values = []
        for leaf in config.leaves:
            mask_values += [leaf[f.name] for f in spec.LEAF_FIELDS]
        for combo in config.combos:
            mask_values += [combo[f.name] for f in spec.COMBO_FIELDS]
        capture = debug_panel.Capture(torch.randn(1, 16, 8, 8, generator=torch.Generator().manual_seed(2)),
                                      False, "krea2")
        decoded = []
        model = types.SimpleNamespace(decode_first_stage=lambda latent: decoded.append(latent) or
                                      torch.zeros(latent.shape[0], 3, latent.shape[-2] * 8, latent.shape[-1] * 8))
        devices = types.SimpleNamespace(dtype_vae=torch.float32)
        modules = types.SimpleNamespace(devices=devices, shared=types.SimpleNamespace(device="cpu"))
        with mock.patch.dict(sys.modules, {"modules": modules, "modules.devices": devices,
                                           "modules.shared": modules.shared}):
            images = debug_panel.render_for_panel(capture, ["tint"], ["M1"], ["C1"], "green", "white", "colormap",
                                                  mask_values, sd_model=model, warn=lambda *_: None)
        self.assertEqual([caption.split(" ", 1)[1] for _, caption in images], ["axis:tint", "mask:M1", "combo:C1"])
        self.assertEqual(len(decoded), 1)
        leaf_specs, combo_specs = spec.build_mask_specs(config)
        expected = debug_panel.build_debug_gallery_images(capture, ["tint"], ["M1"], ["C1"], "none", "white",
                                                          "colormap", leaf_specs, combo_specs, warn=lambda *_: None)
        self.assertTrue(np.array_equal(np.asarray(images[0][0]), np.asarray(expected[0][0])))


class PanelRefreshAdapterTests(unittest.TestCase):
    """The shared editor's Refresh (``ui.build``) hands ``render_for_panel`` v0.31.0's ``mask_values`` (every leaf,
    then every combo) of the effective config: the hidden state with the visible editors laid over."""

    @classmethod
    def setUpClass(cls):
        import gradio as gr

        cls.calls = []
        parts = {}
        with gr.Blocks() as demo:
            components, _ = ui.build(lambda item: f"script_txt2img_colorcraft_samextra_{item}",
                                     on_debug_refresh=lambda *a: cls.calls.append(a) or a, parts=parts)
        fns = demo.fns.values() if isinstance(demo.fns, dict) else demo.fns
        python = [f for f in fns if f.fn is not None]
        assert len(python) == 1, python
        cls.handler = staticmethod(python[0].fn)

    def _refresh(self, args, panel=("tint",), masks=("M1",), combos=("C1",)):
        head = [list(panel), list(masks), list(combos), "none", "white", "colormap"]
        enabled, masking, state, _, _, ref = args[:6]
        return self.handler(*head, enabled, masking, state, ref, *args[panel_state.EDITOR_OFFSET:])

    def test_renders_like_v031_with_an_edit_only_in_the_visible_leaf_editor(self):
        config = _mask_config()
        config.enabled = config.masking = True
        stale = config.copy()
        stale.leaves[0] = {f.name: f.default for f in spec.LEAF_FIELDS}       # M1's edits are only in the editor
        args = ([True, True, panel_state.dump_state(stale, "r1"), False, 5, "I|M1|r1"]
                + panel_state.editor_values(config, "I", "M1"))
        got = self._refresh(args)
        v031 = spec.config_to_args(config)[spec.LEAF_OFFSET:spec.DEBUG_OFFSET]
        self.assertEqual([repr(v) for v in got[6:]], [repr(v) for v in v031])
        self.assertEqual(got[:6], (["tint"], ["M1"], ["C1"], "none", "white", "colormap"))
        capture = debug_panel.Capture(torch.randn(1, 16, 8, 8, generator=torch.Generator().manual_seed(4)),
                                      False, "krea2")
        ours = debug_panel.render_for_panel(capture, ["tint"], ["M1"], ["C1"], "none", "white", "colormap", got[6:],
                                            warn=lambda *_: None)
        theirs = debug_panel.render_for_panel(capture, ["tint"], ["M1"], ["C1"], "none", "white", "colormap", v031,
                                              warn=lambda *_: None)
        self.assertEqual([c for _, c in ours], [c for _, c in theirs])
        self.assertEqual(len(ours), 3)
        for (a, _), (b, _) in zip(ours, theirs):
            self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))
        # the stale state alone would draw another M1
        stale_only = debug_panel.render_for_panel(capture, [], ["M1"], [], "none", "white", "colormap",
                                                  spec.config_to_args(stale)[spec.LEAF_OFFSET:spec.DEBUG_OFFSET],
                                                  warn=lambda *_: None)
        self.assertFalse(np.array_equal(np.asarray(stale_only[0][0]), np.asarray(ours[1][0])))

    def test_an_unreadable_state_draws_the_defaults(self):
        args = panel_state.default_args()
        args[panel_state.STATE_INDEX] = "{not json"
        got = self._refresh(args)
        self.assertEqual(list(got[6:]), spec.config_to_args(spec.default_config())[spec.LEAF_OFFSET:spec.DEBUG_OFFSET])


class CaptureTests(unittest.TestCase):
    def setUp(self):
        S.require_forge(self)

    def _attach(self, mods, debug_step=3, family="krea2", masking=False, **p_fields):
        p = S.make_p(family, **p_fields)
        config = S.scenario_config(Scenario("debug", mods, {}, {}, masking))
        config.debug, config.debug_step = True, debug_step
        callback = S.attach_ours(p, spec.config_to_args(config))
        return p, callback

    def test_captures_the_pre_edit_x0_once_at_the_debug_step(self):
        p, callback = self._attach([Mod("advanced", {"start": 0.0, "end": 1.0, "exposure": 0.4})], debug_step=3)
        unet = p.sd_model.forge_objects.unet
        walked = S.flow_sigmas(10)
        unet.model_options["transformer_options"]["sampling_sigmas"] = walked
        seen = {}
        for k, sigma in enumerate([float(walked[i]) for i in range(10)] + [float(walked[3])]):
            x0 = S.latent("krea2", 40 + k)
            out = callback(S.forge_args(x0, sigma, unet))
            seen[k] = (x0, out)
        capture = getattr(p, hook.STATE_ATTR)["debug"]
        x0, out = seen[3]
        self.assertTrue(torch.equal(capture.latent, x0.squeeze(2)))
        self.assertFalse(torch.equal(capture.latent, out.squeeze(2)))
        self.assertTrue(capture.is_5d)
        self.assertEqual(capture.family, "krea2")
        self.assertEqual(capture.latent.shape[0], 2)              # the repeated step-3 call did not add more
        self.assertIn("debug latent captured at step 3", p.extra_generation_params[spec.STATUS_KEY])

    def test_debug_step_is_clamped_to_the_last_step(self):
        p, callback = self._attach([Mod("advanced", {"start": 0.0, "end": 1.0, "exposure": 0.4})], debug_step=50)
        unet = p.sd_model.forge_objects.unet
        walked = S.flow_sigmas(6)
        unet.model_options["transformer_options"]["sampling_sigmas"] = walked
        for i in range(6):
            callback(S.forge_args(S.latent("krea2", i), float(walked[i]), unet))
        self.assertIn("captured at step 5", p.extra_generation_params[spec.STATUS_KEY])

    def test_off_means_no_capture(self):
        p = S.make_p("krea2")
        callback = S.attach_ours(p, S.scenario_args(Scenario("x", [Mod("advanced", {"exposure": 0.4})])))
        unet = p.sd_model.forge_objects.unet
        unet.model_options["transformer_options"]["sampling_sigmas"] = S.flow_sigmas(10)
        for sigma in S.midpoints(S.flow_sigmas(10)):
            callback(S.forge_args(S.latent("krea2", 1), sigma, unet))
        self.assertNotIn("debug", getattr(p, hook.STATE_ATTR))

    def test_capture_only_when_nothing_is_set(self):
        p, callback = self._attach([Mod("advanced", {})], debug_step=2)
        self.assertIsNotNone(callback)
        self.assertNotIn(spec.INFOTEXT_KEY, p.extra_generation_params)
        unet = p.sd_model.forge_objects.unet
        walked = S.flow_sigmas(8)
        unet.model_options["transformer_options"]["sampling_sigmas"] = walked
        for i in range(8):
            x0 = S.latent("krea2", i)
            self.assertIs(callback(S.forge_args(x0, float(walked[i]), unet)), x0)
        self.assertIn("debug capture only", p.extra_generation_params[spec.STATUS_KEY])
        self.assertIn("debug", getattr(p, hook.STATE_ATTR))

    def test_no_basis_no_capture(self):
        p = S.make_p(None, latent_format=S.latent_formats().SDXL(), vae=S.FakeVAE(4, 2))
        config = spec.default_config()
        config.enabled, config.debug = True, True
        self.assertIsNone(S.attach_ours(p, spec.config_to_args(config)))
        status = p.extra_generation_params[spec.STATUS_KEY]
        self.assertIn("debug capture needs a model with a basis", status)

    def test_batches_accumulate_and_another_size_starts_fresh(self):
        mods = [Mod("advanced", {"start": 0.0, "end": 1.0, "exposure": 0.4, "pass": "Both"})]
        config = S.scenario_config(Scenario("debug", mods))
        config.debug, config.debug_step = True, 1
        args = spec.config_to_args(config)
        p = S.make_p("krea2")
        walked = S.flow_sigmas(4)
        for batch in range(2):
            p.sd_model.forge_objects.unet = S.FakeUnet()
            callback = S.attach_ours(p, args)
            unet = p.sd_model.forge_objects.unet
            unet.model_options["transformer_options"]["sampling_sigmas"] = walked
            for i in range(4):
                callback(S.forge_args(S.latent("krea2", 10 * batch + i), float(walked[i]), unet))
        self.assertEqual(getattr(p, hook.STATE_ATTR)["debug"].latent.shape[0], 4)
        p.is_hr_pass = True
        p.sd_model.forge_objects.unet = S.FakeUnet()
        callback = S.attach_ours(p, args)
        unet = p.sd_model.forge_objects.unet
        unet.model_options["transformer_options"]["sampling_sigmas"] = walked
        for i in range(4):
            callback(S.forge_args(S.latent("krea2", 99, size=32), float(walked[i]), unet))
        self.assertEqual(tuple(getattr(p, hook.STATE_ATTR)["debug"].latent.shape), (2, 16, 32, 32))

    def test_step_counter_samplers_capture_too(self):
        denoiser = types.SimpleNamespace(classic_ddim_eps_estimation=True, step=0, total_steps=8, steps=8)
        p, callback = self._attach([Mod("advanced", {"start": 0.0, "end": 1.0, "exposure": 0.4})], debug_step=4,
                                   sampler=types.SimpleNamespace(model_wrap_cfg=denoiser))
        unet = p.sd_model.forge_objects.unet
        xs = {}
        for step in range(8):
            denoiser.step = step
            xs[step] = S.latent("krea2", 70 + step)
            callback(S.forge_args(xs[step], 0.5, unet))
        self.assertTrue(torch.equal(getattr(p, hook.STATE_ATTR)["debug"].latent, xs[4].squeeze(2)))


if __name__ == "__main__":
    unittest.main()
