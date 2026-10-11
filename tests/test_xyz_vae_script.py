"""XYZ ``VAE`` axis — Forge's axis replaced in place by scripts/modules_xyz_text_encoder.py, against a stub ``modules``.

Registration (the same index, label, type, choices, cost, labels, class; Forge's apply wrapped once, also on Reload UI;
isolated from the Text Encoder axis's registration), the wrapped ``apply`` (Forge's first, then the mark), and the
always-on hook on a VAE cell: a VAE of the other family goes to the loader (loading parameters, ``Module N``, ``sd_vae``
taken out, all put back after the cell), anything Forge can swap - or that the headers cannot tell - is left to Forge
untouched; a failing check or hand-over leaves Forge's swap in place. Forge's own code runs in
tests/test_xyz_vae_origin.py.
"""
from __future__ import annotations

import contextlib
import copy
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import xyz_text_encoder as xte  # noqa: E402
from sam3ext import xyz_text_encoder_cell as cell_runtime  # noqa: E402
from sam3ext import xyz_vae as xvae  # noqa: E402
from sam3ext import xyz_vae_cell  # noqa: E402
from tests import _xyz_text_encoder_support as tfx  # noqa: E402
from tests import _xyz_vae_support as fx  # noqa: E402
from tests.test_xyz_text_encoder_script import AxisOption, Forge, P, load_script  # noqa: E402

V3 = "qwen_image_vae.safetensors"
V3HDR = "qwen_image_vae_HDR_fp32.safetensors"
V2HDR = "qwen_image_vae_2dHDR_fp32.safetensors"
TE = "qwen_3_06b_base.safetensors"
TE_FP32 = "qwen_3_06b_base_full_fp32.safetensors"


def stock_apply_vae(calls):
    """xyz_grid's apply_vae as Forge has it (``find_vae``: "None" stays, else ``sd_vae.vae_dict[name]``)."""

    def apply(p, x, xs):
        calls.append((p, x, list(xs)))
        p.override_settings["sd_vae"] = "None" if x in (None, "None") else STOCK_VAE_DICT[x]

    return apply


STOCK_VAE_DICT: dict = {}


class AxisOptionTxt2Img(AxisOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_img2img = False


def forge_xyz(calls, cls=AxisOption):
    choices = lambda: ["None", *sorted(STOCK_VAE_DICT)]  # noqa: E731
    label = lambda p, opt, x: f"{opt.label}: {x}"  # noqa: E731
    vae = cls("VAE", str, stock_apply_vae(calls), format_value=label, cost=0.7, choices=choices)
    return types.SimpleNamespace(AxisOption=AxisOption, axis_options=[
        AxisOption("Nothing", str, None), AxisOption("Seed", int, None), vae, AxisOption("Clip skip", int, None)])


class RegistrationTests(unittest.TestCase):
    def test_replaced_in_place(self):
        module, _ = load_script()
        calls = []
        xyz = forge_xyz(calls)
        stock = xyz.axis_options[2]
        before = [option.label for option in xyz.axis_options]
        module.replace_vae_axis(xyz)
        option = xyz.axis_options[2]
        self.assertEqual([o.label for o in xyz.axis_options], before, "same labels, same order")
        self.assertIsNot(option, stock)
        self.assertIs(type(option), type(stock))
        for attr in ("label", "type", "format_value", "confirm", "cost", "choices", "prepare"):
            self.assertIs(getattr(option, attr), getattr(stock, attr), attr)
        self.assertEqual(option.cost, 0.7)
        self.assertEqual(option.choices(), stock.choices())
        self.assertIsNot(option.apply, stock.apply)
        self.assertIs(getattr(option, module.VAE_STOCK_APPLY), stock.apply)

    def test_apply_is_forges_then_the_mark(self):
        module, _ = load_script()
        calls = []
        xyz = forge_xyz(calls)
        module.replace_vae_axis(xyz)
        STOCK_VAE_DICT[V3] = "C:/m/VAE/" + V3
        try:
            p = P()
            xs = [V3, "None"]
            xyz.axis_options[2].apply(p, V3, xs)
            self.assertEqual(calls, [(p, V3, xs)])
            self.assertEqual(p.override_settings, {"sd_vae": "C:/m/VAE/" + V3})
            self.assertEqual(getattr(p, xvae.CELL_ATTR), {"value": V3})
            q = P()
            xyz.axis_options[2].apply(q, "None", xs)
            self.assertEqual(q.override_settings, {"sd_vae": "None"})
            # Forge's KeyError for an unknown file stops the grid as before, with no mark
            r = P()
            with self.assertRaises(KeyError):
                xyz.axis_options[2].apply(r, "missing.safetensors", xs)
            self.assertFalse(hasattr(r, xvae.CELL_ATTR))
        finally:
            STOCK_VAE_DICT.clear()

    def test_wrapped_once_also_on_reload(self):
        calls = []
        xyz = forge_xyz(calls)
        stock_apply = xyz.axis_options[2].apply
        module, _ = load_script()
        module.replace_vae_axis(xyz)
        module.replace_vae_axis(xyz)
        reloaded, _ = load_script()
        reloaded.replace_vae_axis(xyz)
        option = xyz.axis_options[2]
        self.assertIs(getattr(option, reloaded.VAE_STOCK_APPLY), stock_apply)
        STOCK_VAE_DICT[V3] = "C:/m/VAE/" + V3
        try:
            option.apply(P(), V3, [V3])
        finally:
            STOCK_VAE_DICT.clear()
        self.assertEqual(len(calls), 1, "Forge's apply runs once")

    def test_the_tab_class_is_kept(self):
        module, _ = load_script()
        xyz = forge_xyz([], AxisOptionTxt2Img)
        module.replace_vae_axis(xyz)
        self.assertIsInstance(xyz.axis_options[2], AxisOptionTxt2Img)
        self.assertFalse(xyz.axis_options[2].is_img2img)

    def test_no_vae_axis(self):
        module, _ = load_script()
        xyz = types.SimpleNamespace(AxisOption=AxisOption, axis_options=[AxisOption("Nothing", str, None)])
        module.replace_vae_axis(xyz)
        self.assertEqual([o.label for o in xyz.axis_options], ["Nothing"])

    def test_on_before_ui_does_both_and_each_alone(self):
        calls = []
        xyz = forge_xyz(calls)
        registered = {}
        module, _ = load_script(registered, xyz)
        registered["before_ui"][0]()
        self.assertEqual([o.label for o in xyz.axis_options],
                         ["Nothing", "Seed", "VAE", "Clip skip", xte.AXIS_LABEL])
        self.assertTrue(hasattr(xyz.axis_options[2], module.VAE_STOCK_APPLY))
        # the Text Encoder axis failing to register does not keep Forge's VAE axis unreplaced (and the reverse)
        xyz = forge_xyz(calls)
        registered = {}
        module, _ = load_script(registered, xyz)
        stderr = io.StringIO()
        with mock.patch.object(module, "register_xyz_axes", side_effect=RuntimeError("te")), \
                contextlib.redirect_stderr(stderr):
            registered["before_ui"][0]()
        self.assertTrue(hasattr(xyz.axis_options[2], module.VAE_STOCK_APPLY))
        self.assertIn("xyz_grid axis registration failed", stderr.getvalue())
        xyz = forge_xyz(calls)
        registered = {}
        module, _ = load_script(registered, xyz)
        stderr = io.StringIO()
        with mock.patch.object(module, "replace_vae_axis", side_effect=RuntimeError("vae")), \
                contextlib.redirect_stderr(stderr):
            registered["before_ui"][0]()
        self.assertEqual(xyz.axis_options[-1].label, xte.AXIS_LABEL)
        self.assertIn("[XYZ VAE] replacing Forge's VAE axis failed - Forge's own axis stays", stderr.getvalue())


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)


class FakeVae:
    def __init__(self, spec):
        self.spec = spec

    def state_dict(self):
        return {key: FakeTensor(shape) for key, shape in self.spec.items()}


class HookCase(unittest.TestCase):
    UI = (V3, TE)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        models = Path(self.tmp.name) / "models"
        module_list = fx.write_user_vaes(models / "VAE")
        for name in (TE, TE_FP32):
            module_list[name] = str(fx.write_vae(models / "text_encoder" / name, tfx.CASES["qwen3_06b"][0], 0.5))
        self.forge = Forge(models, module_list=module_list, modules=self.UI)
        self.module, self.scripts = load_script()
        self.hook = self.module.XyzTextEncoderHook()
        calls = []
        self.xyz = forge_xyz(calls)
        self.module.register_xyz_axes(self.xyz)
        self.module.replace_vae_axis(self.xyz)
        self.vae_axis = self.xyz.axis_options[2]
        self.te_axis = self.xyz.axis_options[-1]
        STOCK_VAE_DICT.update({name: path for name, path in module_list.items() if "VAE" in Path(path).parts})
        xte.clear_cache()
        cell_runtime._STATE = None

    def tearDown(self):
        cell_runtime._STATE = None
        STOCK_VAE_DICT.clear()
        xte.clear_cache()
        self.tmp.cleanup()

    def load(self, spec):
        """The model Forge has loaded from the current loading parameters, its VAE of layout ``spec``."""
        data = self.forge.sd_models.model_data
        data.sd_model = types.SimpleNamespace(first_stage_model=FakeVae(spec))
        data.forge_hash = str(data.forge_loading_parameters)

    def cell(self, parent, vae, te=None):
        pc = copy.copy(parent)
        if te is not None:
            self.te_axis.apply(pc, te, [te])
        self.vae_axis.apply(pc, vae, [vae])
        self.hook.before_process(pc)
        return pc


class HookTests(HookCase):
    def test_another_family_goes_to_the_loader_and_back(self):
        self.load(fx.tiny_3d())
        parent = P({"CLIP_stop_at_last_layers": 2})
        settings = parent.override_settings
        before = self.forge.params
        stderr = io.StringIO()
        with self.forge.installed(), contextlib.redirect_stderr(stderr):
            pc = self.cell(parent, V2HDR)
            self.assertEqual(self.forge.params["additional_modules"], self.forge.paths(V2HDR, TE))
            self.assertIs(self.forge.params["checkpoint_info"], before["checkpoint_info"])
            self.assertEqual(self.forge.processing._overridden_modules, self.forge.paths(V2HDR, TE))
            self.assertNotIn("sd_vae", pc.override_settings, "Forge's in-place swap does not run")
            self.assertIs(pc.override_settings, settings)
            self.assertEqual(self.forge.opts.forge_additional_modules, self.forge.paths(V3, TE))
            self.hook.process(pc)
            self.hook.postprocess_image(pc, None)
            self.hook.postprocess(pc, None)
        self.assertIs(self.forge.params, before)
        self.assertIsNone(self.forge.processing._overridden_modules)
        self.assertIsNone(cell_runtime.state())
        self.assertEqual(settings, {"CLIP_stop_at_last_layers": 2})
        self.assertEqual(self.forge.opts.saved, [])
        self.assertIn(f"[XYZ VAE] {V2HDR} is a 2D Qwen2D VAE", stderr.getvalue())
        self.assertIn("the loaded model's VAE is a 3D Wan / Qwen-Image VAE", stderr.getvalue())

    def test_what_forge_can_swap_is_left_to_forge(self):
        self.load(fx.tiny_3d())
        for value, expected in ((V3HDR, STOCK_VAE_DICT.get(V3HDR)), ("None", "None"), (V3, None)):
            with self.subTest(value=value):
                before = self.forge.params
                parent = P()
                stderr = io.StringIO()
                with self.forge.installed(), contextlib.redirect_stderr(stderr):
                    pc = self.cell(parent, value)
                self.assertIs(self.forge.params, before)
                self.assertEqual(pc.override_settings["sd_vae"], expected or STOCK_VAE_DICT[V3])
                self.assertIsNone(self.forge.processing._overridden_modules)
                self.assertIsNone(cell_runtime.state())
                self.assertEqual(stderr.getvalue(), "")

    def test_predicted_from_the_selection_when_another_model_is_loaded(self):
        """The loaded model is not the one the cell gets (its loading parameters changed): the selection's VAE file
        tells what Forge will build."""
        self.load(fx.tiny_2d())
        self.forge.sd_models.model_data.forge_hash = "another model"
        with self.forge.installed(), contextlib.redirect_stderr(io.StringIO()) as stderr:
            pc = self.cell(P(), V3HDR)
            self.assertIn("sd_vae", pc.override_settings)
            self.assertIsNone(cell_runtime.state())
            pc = self.cell(P(), V2HDR)
            self.assertEqual(self.forge.params["additional_modules"], self.forge.paths(V2HDR, TE))
        self.assertIn("the VAE the model is built with is a 3D", stderr.getvalue())

    def test_what_the_headers_cannot_tell_is_left_to_forge(self):
        pickle = Path(self.tmp.name) / "models" / "VAE" / "old.pt"
        pickle.write_bytes(b"\x80\x02}q\x00.")
        self.forge.module_list["old.pt"] = STOCK_VAE_DICT["old.pt"] = str(pickle)
        self.load(fx.tiny_3d())
        with self.forge.installed(), contextlib.redirect_stderr(io.StringIO()):
            pc = self.cell(P(), "old.pt")
        self.assertEqual(pc.override_settings["sd_vae"], str(pickle))
        self.assertIsNone(cell_runtime.state())
        # nothing loaded and the selection's VAE a pickle file: nothing to compare with
        self.forge.opts.data["forge_additional_modules"] = self.forge.paths("old.pt", TE)
        self.forge.sd_models.model_data.forge_hash = ""
        with self.forge.installed(), contextlib.redirect_stderr(io.StringIO()):
            pc = self.cell(P(), V2HDR)
        self.assertEqual(pc.override_settings["sd_vae"], STOCK_VAE_DICT[V2HDR])
        self.assertIsNone(cell_runtime.state())

    def test_an_empty_module_list_is_left_to_forge(self):
        self.load(fx.tiny_3d())
        self.forge.module_list.clear()
        with self.forge.installed(), contextlib.redirect_stderr(io.StringIO()):
            pc = self.cell(P(), V2HDR)
        self.assertIn("sd_vae", pc.override_settings)
        self.assertIsNone(cell_runtime.state())

    def test_a_pass_inside_the_cell_is_not_decided_again(self):
        self.load(fx.tiny_3d())
        with self.forge.installed(), contextlib.redirect_stderr(io.StringIO()):
            pc = self.cell(P(), V2HDR)
            state = cell_runtime.state()
            inner = copy.copy(pc)                          # ADetailer / SAM3 refine: a copy of the cell's p
            inner.override_settings = {"sd_vae": STOCK_VAE_DICT[V3HDR]}
            with mock.patch.object(xyz_vae_cell, "plan", side_effect=AssertionError("decided twice")):
                self.assertFalse(cell_runtime.before_process(inner))
            self.assertIs(cell_runtime.state(), state)
            self.hook.postprocess(pc, None)
        self.assertIsNone(cell_runtime.state())

    def test_a_failing_check_leaves_forges_swap(self):
        self.load(fx.tiny_3d())
        stderr = io.StringIO()
        with self.forge.installed(), contextlib.redirect_stderr(stderr), \
                mock.patch.object(xyz_vae_cell, "model_vae", side_effect=OSError("disk")):
            pc = self.cell(P(), V2HDR, te="qwen_3_06b_base_full_fp32")
        self.assertEqual(pc.override_settings["sd_vae"], STOCK_VAE_DICT[V2HDR])
        self.assertIn("[XYZ VAE] checking whether Forge can swap the cell's VAE in place failed", stderr.getvalue())
        # the Text Encoder axis's list still goes to the loader
        self.assertEqual(self.forge.params["additional_modules"], self.forge.paths(V3, TE_FP32))

    def test_a_failing_hand_over_leaves_forges_swap(self):
        self.load(fx.tiny_3d())
        stderr = io.StringIO()
        with self.forge.installed(), contextlib.redirect_stderr(stderr), \
                mock.patch.object(cell_runtime, "begin", side_effect=RuntimeError("loader")):
            pc = self.cell(P(), V2HDR)
        self.assertEqual(pc.override_settings["sd_vae"], STOCK_VAE_DICT[V2HDR], "an error, not the selection's VAE")
        self.assertIn("NOT applied", stderr.getvalue())

    def test_with_a_text_encoder_cell(self):
        """The Text Encoder axis's list with the VAE replaced; the loaded model is the encoder's (its parameters)."""
        with self.forge.installed(), contextlib.redirect_stderr(io.StringIO()):
            data = self.forge.sd_models.model_data
            expected = dict(data.forge_loading_parameters)
            expected["additional_modules"] = self.forge.paths(V3, TE_FP32)
            data.sd_model = types.SimpleNamespace(first_stage_model=FakeVae(fx.tiny_3d()))
            data.forge_hash = str(expected)
            pc = self.cell(P(), V2HDR, te="qwen_3_06b_base_full_fp32")
            self.assertEqual(self.forge.params["additional_modules"], self.forge.paths(V2HDR, TE_FP32))
            self.assertEqual(self.forge.processing._overridden_modules, self.forge.paths(V2HDR, TE_FP32))
            self.assertNotIn("sd_vae", pc.override_settings)
            self.hook.postprocess(pc, None)
            # same family on the encoder's model: Forge swaps, the cell's list is the encoder's
            pc = self.cell(P(), V3HDR, te="qwen_3_06b_base_full_fp32")
            self.assertEqual(self.forge.params["additional_modules"], self.forge.paths(V3, TE_FP32))
            self.assertEqual(pc.override_settings["sd_vae"], STOCK_VAE_DICT[V3HDR])
            self.hook.postprocess(pc, None)
        self.assertIsNone(cell_runtime.state())


class TwoDimensionalSelectionTests(HookCase):
    UI = (V2HDR, TE)

    def test_the_reverse(self):
        self.load(fx.tiny_2d())
        with self.forge.installed(), contextlib.redirect_stderr(io.StringIO()):
            pc = self.cell(P(), V3HDR)
            self.assertEqual(self.forge.params["additional_modules"], self.forge.paths(V3HDR, TE))
            self.hook.postprocess(pc, None)
            pc = self.cell(P(), "Qwen2D-Anime-dense_epoch_1.safetensors")
            self.assertIn("sd_vae", pc.override_settings)
        self.assertIsNone(cell_runtime.state())


if __name__ == "__main__":
    unittest.main()
