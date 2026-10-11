"""XYZ ``[Modules] Text Encoder`` against Forge's own code (Forge Neo 2.30; skipped without a Forge checkout).

1. **Slots** — Forge's real ``backend/loader.py`` ``replace_state_dict`` runs on the same headers as
   ``sam3ext.xyz_text_encoder.te_slots`` (tensors are stand-ins with a ``shape``): the ``text_encoders.<slot>.`` keys it
   writes are the slots te_slots gives, for every branch, the files Forge rejects, and the safetensors files in the
   checkout's own ``models/text_encoder`` (headers only).
2. **A grid through Forge's restore code** — Forge's ``processing.process_images`` / ``manage_model_and_prompt_cache``,
   ``sysinfo.set_config``, ``main_entry.modules_change`` / ``checkpoint_change`` / ``refresh_model_loading_parameters``,
   ``sd_models.get_closet_checkpoint_match`` / ``select_checkpoint`` and xyz_grid's ``apply_checkpoint`` / ``apply_vae``
   / ``find_vae`` / ``refresh_loading_params_for_xyz_grid`` run against stand-ins for the model loader, the sampler and
   the VAE swap, driven like xyz_grid's ``cell`` (shallow copy, X/Y/Z applied outside the try, ``process_images``).
   Checked per cell: the checkpoint and modules the model is loaded with, the ``Module N`` list of the infotext; after
   the grid: Forge's settings and loading parameters are the user's again, config.json was not saved, the parent's
   ``override_settings`` is untouched — with a Checkpoint axis in both axis orders, a VAE axis, "Use Forge current"
   between values, a failing cell and an API caller's own module list.
3. **Source guards** — the Forge code the axis relies on is still shaped the same way.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import io
import logging
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import xyz_text_encoder as xte  # noqa: E402
from tests import _xyz_text_encoder_support as fx  # noqa: E402
from tests._extra_schedulers_support import forge_function, forge_root, forge_source  # noqa: E402
from tests._forge_checkout import require_forge_file  # noqa: E402
from tests.test_xyz_text_encoder_script import AxisOption, load_script  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# 1. Slots: Forge's replace_state_dict
# ---------------------------------------------------------------------------------------------------------------------

class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]


GUESS = types.SimpleNamespace(vae_key_prefix=["first_stage_model."], text_encoder_key_prefix=["text_encoders."])
# checkpoint keys that make replace_state_dict pick a model type (where CLIP-L / CLIP-G weights go)
SDXL = {"model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": [640, 2048]}
FLUX = {"model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale": [128]}
CLIP_CONTEXT = {
    "clip_l_hf": FLUX, "clip_l_sd1_prefixed": FLUX, "clip_l_sdxl_prefixed": FLUX, "clip_l_flux_prefixed": FLUX,
    "clip_g_resblocks": SDXL, "clip_g_sdxl_resblocks": SDXL, "clip_other_width": SDXL, "clip_l_and_g_in_one": SDXL,
}
# formats Forge converts with torch (key renames + torch.cat) — checked by key name and shape only (test_xyz_text_encoder)
NEEDS_TORCH = {"clip_g_hf", "clip_l_resblocks"}
PREFIX_SLOTS = (
    ("cond_stage_model.transformer.", "clip_l"),
    ("conditioner.embedders.0.transformer.", "clip_l"),
    ("conditioner.embedders.0.model.transformer.", "clip_g"),
    ("conditioner.embedders.1.model.transformer.", "clip_g"),
)


class SlotOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_forge_file("backend/loader.py")
        namespace = {"os": os, "torch": types.SimpleNamespace(Tensor=object)}
        forge_function("backend/state_dict.py", "state_dict_prefix_replace", namespace)
        cls.replace = staticmethod(forge_function("backend/loader.py", "replace_state_dict", namespace))

    def forge_slots(self, shapes, checkpoint=None):
        checkpoint = checkpoint or {}
        sd = {key: FakeTensor(shape) for key, shape in checkpoint.items()}
        asd = {key: FakeTensor(shape) for key, shape in shapes.items()}
        out = self.replace(dict(sd), asd, GUESS, "file.safetensors")
        slots = set()
        for key in out:
            if key in sd:
                continue
            if key.startswith("text_encoders."):
                slots.add(key[len("text_encoders."):].split(".", 1)[0])
            for prefix, slot in PREFIX_SLOTS:
                if key.startswith(prefix):
                    slots.add(slot)
        return slots

    def test_every_case_matches_forge(self):
        for name, (shapes, metadata, expected) in fx.CASES.items():
            if name in NEEDS_TORCH:
                continue
            with self.subTest(case=name):
                header = fx.header(shapes, metadata)
                if expected == fx.RAISES:
                    with self.assertRaises((AssertionError, KeyError)):
                        self.forge_slots(shapes)
                    with self.assertRaises(xte.ForgeLoadError):
                        xte.te_slots(header)
                    continue
                forge = self.forge_slots(shapes, CLIP_CONTEXT.get(name))
                self.assertEqual(forge, set(xte.te_slots(header)))
                self.assertEqual(forge, set(expected))

    def test_clip_goes_nowhere_without_a_clip_family(self):
        # where a CLIP file goes depends on the checkpoint: nowhere for Anima / Krea2 (te_slots names what it IS)
        self.assertEqual(self.forge_slots(fx.CASES["clip_l_hf"][0]), set())
        self.assertEqual(self.forge_slots(fx.CASES["clip_l_hf"][0], SDXL), {"clip_l"})

    def test_the_checkouts_text_encoder_files(self):
        """The safetensors files in the checkout's models/text_encoder: Forge's slots from their real headers."""
        folder = forge_root() / "models" / "text_encoder"
        files = sorted(folder.rglob("*.safetensors")) + sorted(folder.rglob("*.sft")) if folder.is_dir() else []
        if not files:
            self.skipTest(f"no safetensors file in {folder}")
        for path in files:
            with self.subTest(file=path.name):
                header = xte.read_safetensors_header(path)
                self.assertIsInstance(header, dict)
                kind = xte.classify_header(path.name, header)
                if kind.kind == "vae":
                    continue
                shapes = {key: value["shape"] for key, value in header.items() if key != "__metadata__"}
                if kind.kind == "invalid":
                    with self.assertRaises((AssertionError, KeyError)):
                        self.forge_slots(shapes)
                    continue
                # CLIP weights go nowhere without a CLIP checkpoint (model type "-"): compared by name above
                expected = set(xte.te_slots(header)) - {"clip_l", "clip_g"}
                self.assertEqual(self.forge_slots(shapes), expected)
                if kind.kind in ("qwen35", "adapter"):
                    self.assertEqual(expected, set(), "Forge would use it as an encoder")


# ---------------------------------------------------------------------------------------------------------------------
# 2. A grid through Forge's settings / reload / restore code
# ---------------------------------------------------------------------------------------------------------------------

VAE = "qwen_image_vae.safetensors"
VAE_HDR = "qwen_image_vae_HDR_fp32.safetensors"
TE = "qwen_3_06b_base.safetensors"
TE_FP32 = "qwen_3_06b_base_full_fp32.safetensors"
TE_FP16 = "qwen_3_06b_base_fp16_uncensored.safetensors"
CKPT_A = "anima.safetensors"
CKPT_B = "anima_other.safetensors"
NOTHING = AxisOption("Nothing", str, lambda p, x, xs: None)


class P:
    """A processing object as far as process_images uses it here; xyz_grid copies it per cell."""

    def __init__(self, scripts, override_settings=None):
        self.scripts = scripts
        self.override_settings = {} if override_settings is None else override_settings
        self.override_settings_restore_afterwards = True
        self.styles = []
        self.seed = 1
        self.prompt_cache_clears = 0

    def clear_prompt_cache(self):
        self.prompt_cache_clears += 1


class Grid:
    """Forge's code from the checkout, wired together, with stand-ins for the loader, the sampler and the VAE swap."""

    def __init__(self, models_path):
        self.module_list, self.module_map = fx.write_models(models_path)
        self.models_path = str(models_path)
        self.loads, self.cells, self.unloads, self.vae_swaps, self.reported = [], [], 0, [], []
        self.fail_cells = set()
        self.swapped_vae = None
        infos = [fx.CheckpointInfo(CKPT_A, "aaaa"), fx.CheckpointInfo(CKPT_B, "bbbb")]
        self.ckpt = {info.name: info for info in infos}
        self.ui_modules = self.paths(VAE, TE)
        self.opts = fx.FakeOpts(
            {"sd_model_checkpoint": self.ckpt[CKPT_A].title, "forge_additional_modules": list(self.ui_modules),
             "forge_unet_storage_dtype": "Automatic", "CLIP_stop_at_last_layers": 1},
            {"forge_additional_modules": [], "sd_modules_dropdown_use_short": True})

        shared = self.shared = types.ModuleType("modules.shared")
        shared.opts = self.opts
        shared.cmd_opts = types.SimpleNamespace(text_encoder_dirs=[])
        shared.sd_model = None

        sd_models = self.sd_models = types.ModuleType("modules.sd_models")
        aliases = {alias: info for info in infos for alias in info.ids}
        sd_models.__dict__.update(checkpoint_aliases=aliases, checkpoints_list={i.title: i for i in infos}, re=re,
                                  sys=sys, shared=shared, re_strip_checksum=re.compile(r"\s*\[[^]]+]\s*$"))
        forge_function("modules/sd_models.py", "get_closet_checkpoint_match", sd_models.__dict__)
        forge_function("modules/sd_models.py", "select_checkpoint", sd_models.__dict__)
        sd_models.model_data = types.SimpleNamespace(forge_loading_parameters={}, forge_hash="", sd_model=None)

        proc = self.proc = types.ModuleType("modules.processing")

        main_entry = self.main_entry = types.ModuleType("modules_forge.main_entry")
        main_entry.__dict__.update(
            shared=shared, sd_models=sd_models, module_list=self.module_list, module_map=self.module_map, os=os,
            torch=types.SimpleNamespace(float16="float16", bfloat16="bfloat16"),
            forge_unet_storage_dtype_options={"Automatic": (None, False)}, logger=logging.getLogger("xyz-te-test"),
            print_json=lambda **kwargs: None, dynamic_args=types.SimpleNamespace(online_lora=False), processing=proc)
        for name in ("modules_change", "checkpoint_change", "refresh_model_loading_parameters"):
            forge_function("modules_forge/main_entry.py", name, main_entry.__dict__)
        main_entry.refresh_models = lambda: ([], sorted(self.module_list))

        sysinfo = types.ModuleType("modules.sysinfo")
        sysinfo.Any = Any
        forge_function("modules/sysinfo.py", "set_config", sysinfo.__dict__)

        self.vae_dict = {name: path for name, path in self.module_list.items() if os.sep + "VAE" + os.sep in path}
        sd_vae = types.SimpleNamespace(vae_dict=self.vae_dict, reload_vae_weights=self._reload_vae,
                                       restore_vae_weights=self._restore_vae)
        proc.__dict__.update(
            opts=self.opts, shared=shared, sd_models=sd_models, set_config=sysinfo.set_config, os=os, sd_vae=sd_vae,
            sd_samplers=types.SimpleNamespace(fix_p_invalid_sampler_and_scheduler=lambda p: None),
            profiling=types.SimpleNamespace(Profiler=contextlib.nullcontext), process_images_inner=self._inner,
            forge_model_reload=self._forge_model_reload, memory_management=types.SimpleNamespace(
                unload_all_models=self._unload_all), need_global_unload=False, _overridden_modules=None,
            StableDiffusionProcessing=object, Processed=object, Optional=Optional)
        forge_function("modules/processing.py", "manage_model_and_prompt_cache", proc.__dict__)
        forge_function("modules/processing.py", "process_images", proc.__dict__)

        modules = self.modules = types.ModuleType("modules")
        modules.__path__ = []
        paths = types.ModuleType("modules.paths")
        paths.models_path = self.models_path
        modules.shared, modules.sd_models, modules.processing, modules.paths, modules.sysinfo = (
            shared, sd_models, proc, paths, sysinfo)
        modules_forge = types.ModuleType("modules_forge")
        modules_forge.__path__ = []
        modules_forge.main_entry = main_entry
        self.sys_modules = {"modules": modules, "modules.shared": shared, "modules.sd_models": sd_models,
                            "modules.processing": proc, "modules.paths": paths, "modules.sysinfo": sysinfo,
                            "modules_forge": modules_forge, "modules_forge.main_entry": main_entry}

        xyz = self.xyz = types.ModuleType("xyz_grid.py")
        xyz.__dict__.update(modules=modules, opts=self.opts, shared=shared, select_checkpoint=sd_models.select_checkpoint,
                            model_data=sd_models.model_data, sd_vae=sd_vae, StableDiffusionProcessing=object)
        for name in ("refresh_loading_params_for_xyz_grid", "find_vae", "apply_vae", "apply_checkpoint"):
            forge_function("scripts/xyz_grid.py", name, xyz.__dict__)

        self.script, _ = load_script()
        self.hook = self.script.XyzTextEncoderHook()
        self.te_axis = self.script.make_axis(types.SimpleNamespace(AxisOption=AxisOption))
        self.ckpt_axis = AxisOption("Checkpoint name", str, xyz.apply_checkpoint, cost=1.0)
        self.vae_axis = AxisOption("VAE", str, xyz.apply_vae, cost=0.7)
        self.runner = types.SimpleNamespace(before_process=self._before_process)

        with mock.patch.dict(sys.modules, self.sys_modules):
            main_entry.refresh_model_loading_parameters()     # Forge's state after startup
        self.ui_loading = self.describe(sd_models.model_data.forge_loading_parameters)

    # -- stand-ins below Forge's code --
    def paths(self, *names):
        return sorted(self.module_list[name] for name in names)

    @staticmethod
    def describe(params):
        return params["checkpoint_info"].name, tuple(sorted(os.path.basename(m) for m in params["additional_modules"]))

    def _forge_model_reload(self):
        data = self.sd_models.model_data
        current = str(data.forge_loading_parameters)
        if data.forge_hash == current:
            return data.sd_model, False
        loaded = self.describe(data.forge_loading_parameters)
        self.loads.append(loaded)
        data.sd_model = types.SimpleNamespace(loaded=loaded)
        data.forge_hash = current
        self.shared.sd_model = data.sd_model
        return data.sd_model, True

    def _unload_all(self):
        self.unloads += 1

    def _reload_vae(self, vae):
        if vae in (None, "None"):
            return False
        self.swapped_vae = os.path.basename(vae)
        return True

    def _restore_vae(self):
        self.swapped_vae = None

    def _inner(self, p):
        index = len(self.cells)
        modules = self.proc.__dict__["_overridden_modules"] or self.opts.forge_additional_modules
        self.cells.append({
            "model": self.sd_models.model_data.sd_model.loaded,
            "loading": self.describe(self.sd_models.model_data.forge_loading_parameters),
            "module_n": sorted(os.path.splitext(os.path.basename(m))[0] for m in modules),
            "vae": self.swapped_vae,
        })
        if index in self.fail_cells:
            raise RuntimeError(f"cell {index} failed")
        return types.SimpleNamespace(images=[index])

    def _before_process(self, p):
        # modules/scripts.py ScriptRunner.before_process: every always-on script, errors reported
        try:
            self.hook.before_process(p)
        except Exception as exc:
            self.reported.append(exc)

    # -- xyz_grid's run / cell --
    def run(self, p, x=(NOTHING, [None]), y=(NOTHING, [None]), z=(NOTHING, [None])):
        (x_opt, xs), (y_opt, ys), (z_opt, zs) = x, y, z
        with mock.patch.dict(sys.modules, self.sys_modules), contextlib.redirect_stdout(io.StringIO()):
            for opt, values in (x, y, z):
                if opt.confirm:
                    opt.confirm(p, values)

            def cell(xv, yv, zv):
                pc = copy.copy(p)
                pc.styles = pc.styles[:]
                x_opt.apply(pc, xv, xs)
                y_opt.apply(pc, yv, ys)
                z_opt.apply(pc, zv, zs)
                try:
                    self.proc.process_images(pc)
                except Exception:
                    pass   # errors.display; the grid goes on

            # xyz_grid's loop order: the costliest axis outside (strict ">"), else z, y, x
            first, second = "z", "y"
            if x_opt.cost > y_opt.cost and x_opt.cost > z_opt.cost:
                first, second = "x", ("y" if y_opt.cost > z_opt.cost else "z")
            elif y_opt.cost > x_opt.cost and y_opt.cost > z_opt.cost:
                first, second = "y", ("x" if x_opt.cost > z_opt.cost else "z")
            elif z_opt.cost > x_opt.cost and z_opt.cost > y_opt.cost:
                first, second = "z", ("x" if x_opt.cost > y_opt.cost else "y")
            third = ({"x", "y", "z"} - {first, second}).pop()
            values = {"x": xs, "y": ys, "z": zs}
            for a in values[first]:
                for b in values[second]:
                    for c in values[third]:
                        picked = {first: a, second: b, third: c}
                        cell(picked["x"], picked["y"], picked["z"])
            self.xyz.refresh_loading_params_for_xyz_grid()    # after the grid


def expected(ckpt, *modules):
    return ckpt, tuple(sorted(modules))


def stems(*names):
    return sorted(os.path.splitext(name)[0] for name in names)


class GridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for rel in ("modules/processing.py", "modules/sysinfo.py", "modules_forge/main_entry.py", "modules/sd_models.py",
                    "scripts/xyz_grid.py"):
            require_forge_file(rel)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        xte.clear_cache()
        self.grid = Grid(Path(self.tmp.name) / "models")
        self.p = P(self.grid.runner)
        self.parent_settings = self.p.override_settings

    def tearDown(self):
        xte.clear_cache()
        self.tmp.cleanup()

    def assertRestored(self, parent_settings=None):
        g = self.grid
        self.assertEqual(g.opts.data["forge_additional_modules"], g.ui_modules)
        self.assertEqual(g.opts.data["sd_model_checkpoint"], g.ckpt[CKPT_A].title)
        self.assertEqual(g.describe(g.sd_models.model_data.forge_loading_parameters), g.ui_loading)
        self.assertEqual(g.ui_loading, expected(CKPT_A, VAE, TE))
        self.assertEqual(g.opts.saved, [], "config.json is never saved")
        self.assertIs(self.p.override_settings, self.parent_settings)
        self.assertEqual(self.p.override_settings, parent_settings or {})
        self.assertFalse(hasattr(self.p, xte.CELL_ATTR))
        self.assertEqual(g.reported, [])
        self.assertIsNone(g.proc.__dict__["_overridden_modules"])

    def models(self):
        return [cell["model"] for cell in self.grid.cells]

    def test_text_encoder_only(self):
        values = ["Use Forge current", "qwen_3_06b_base_full_fp32", "qwen_3_06b_base_fp16_uncensored.safetensors", TE]
        self.grid.run(self.p, x=(self.grid.te_axis, values))
        self.assertEqual(self.models(), [expected(CKPT_A, VAE, TE), expected(CKPT_A, VAE, TE_FP32),
                                         expected(CKPT_A, VAE, TE_FP16), expected(CKPT_A, VAE, TE)])
        self.assertEqual([cell["module_n"] for cell in self.grid.cells],
                         [stems(VAE, TE), stems(VAE, TE_FP32), stems(VAE, TE_FP16), stems(VAE, TE)])
        for cell in self.grid.cells:
            self.assertEqual(cell["loading"], cell["model"])
        self.assertRestored()

    def test_checkpoint_axis_in_both_orders(self):
        te_values = ["Use Forge current", "qwen_3_06b_base_full_fp32"]
        ckpt_values = [CKPT_A, CKPT_B]
        want = {expected(CKPT_A, VAE, TE), expected(CKPT_A, VAE, TE_FP32), expected(CKPT_B, VAE, TE),
                expected(CKPT_B, VAE, TE_FP32)}
        for order in ("checkpoint_x", "checkpoint_y"):
            with self.subTest(order=order):
                self.tearDown()
                self.setUp()
                g = self.grid
                if order == "checkpoint_x":
                    g.run(self.p, x=(g.ckpt_axis, ckpt_values), y=(g.te_axis, te_values))
                else:
                    g.run(self.p, x=(g.te_axis, te_values), y=(g.ckpt_axis, ckpt_values))
                self.assertEqual(len(g.cells), 4)
                self.assertEqual(set(self.models()), want)
                # the Checkpoint axis stays the outer loop (cost 1.0 > 0.9)
                self.assertEqual([m[0] for m in self.models()], [CKPT_A, CKPT_A, CKPT_B, CKPT_B])
                for cell in g.cells:
                    self.assertEqual(cell["module_n"], stems(*cell["model"][1]))
                self.assertRestored()

    def test_vae_axis(self):
        g = self.grid
        g.run(self.p, x=(g.vae_axis, [VAE, VAE_HDR]), y=(g.te_axis, ["Use Forge current", "qwen_3_06b_base_full_fp32"]))
        # the text encoder (0.9) is the outer loop, the VAE (0.7) is swapped on the loaded model
        self.assertEqual(self.models(), [expected(CKPT_A, VAE, TE), expected(CKPT_A, VAE, TE),
                                         expected(CKPT_A, VAE, TE_FP32), expected(CKPT_A, VAE, TE_FP32)])
        self.assertEqual([cell["vae"] for cell in g.cells], [None, VAE_HDR, None, VAE_HDR])
        self.assertEqual([cell["module_n"] for cell in g.cells],
                         [stems(VAE, TE), stems(VAE_HDR, TE), stems(VAE, TE_FP32), stems(VAE_HDR, TE_FP32)])
        self.assertEqual(len(g.loads), 2, "each text encoder is loaded once")
        self.assertIsNone(g.swapped_vae)
        # xyz's apply_vae writes "sd_vae" into the shared dict; it is gone after the cell, as without the axis
        self.assertRestored()

    def test_current_between_values(self):
        g = self.grid
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32", "Use Forge current", "qwen_3_06b_base_fp16_uncensored"]))
        self.assertEqual(self.models(), [expected(CKPT_A, VAE, TE_FP32), expected(CKPT_A, VAE, TE),
                                         expected(CKPT_A, VAE, TE_FP16)])
        self.assertRestored()

    def test_a_failing_cell_is_restored(self):
        g = self.grid
        g.fail_cells = {0}
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32", "Use Forge current"]))
        self.assertEqual(self.models(), [expected(CKPT_A, VAE, TE_FP32), expected(CKPT_A, VAE, TE)])
        self.assertRestored()

    def test_restored_also_when_the_caller_keeps_its_settings(self):
        # an API caller may send override_settings_restore_afterwards=False; the axis's cells are restored anyway
        g = self.grid
        self.p.override_settings_restore_afterwards = False
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32", "Use Forge current"]))
        self.assertEqual(self.models(), [expected(CKPT_A, VAE, TE_FP32), expected(CKPT_A, VAE, TE)])
        self.assertFalse(self.p.override_settings_restore_afterwards)
        self.assertRestored()

    def test_a_fresh_config_is_restored_too(self):
        # no "forge_additional_modules" in config.json yet: process_images would not restore a key opts.data lacks
        g = self.grid
        del g.opts.data["forge_additional_modules"]
        with mock.patch.dict(sys.modules, g.sys_modules), contextlib.redirect_stdout(io.StringIO()):
            g.main_entry.refresh_model_loading_parameters()
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32"]))
        self.assertEqual(self.models(), [expected(CKPT_A, TE_FP32)])
        self.assertEqual(g.opts.forge_additional_modules, [])
        self.assertEqual(g.describe(g.sd_models.model_data.forge_loading_parameters), expected(CKPT_A))
        self.assertEqual(g.opts.saved, [])

    def test_same_encoder_twice_is_loaded_once(self):
        g = self.grid
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32", "qwen_3_06b_base_full_fp32"]))
        self.assertEqual(g.loads, [expected(CKPT_A, VAE, TE_FP32)])
        # the restore after the first cell asks for a global unload, paid by the second (not a disk reload)
        self.assertEqual(g.unloads, 1)
        self.assertRestored()

    def test_api_callers_module_list_is_the_base(self):
        g = self.grid
        api = {"forge_additional_modules": [VAE_HDR, TE]}
        self.p = P(g.runner, api)
        self.parent_settings = api
        g.run(self.p, x=(g.te_axis, ["Use Forge current", "qwen_3_06b_base_full_fp32"]))
        self.assertEqual(self.models(), [expected(CKPT_A, VAE_HDR, TE), expected(CKPT_A, VAE_HDR, TE_FP32)])
        self.assertRestored(parent_settings={"forge_additional_modules": [VAE_HDR, TE]})

    def test_a_bad_value_stops_before_the_first_cell(self):
        g = self.grid
        with self.assertRaisesRegex(ValueError, "is a Qwen3-VL 4B encoder"):
            g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32", "qwen3vl_4b_bf16"]))
        self.assertEqual(g.cells, [])
        self.assertRestored()


# ---------------------------------------------------------------------------------------------------------------------
# 3. Source guards
# ---------------------------------------------------------------------------------------------------------------------

def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise LookupError(name)


class SourceGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_forge_file("modules/processing.py")

    def test_before_process_runs_before_the_settings_are_stored(self):
        source = forge_source("modules/processing.py")
        node = _function(ast.parse(source), "process_images")
        segment = ast.get_source_segment(source, node)
        self.assertLess(segment.index("p.scripts.before_process(p)"), segment.index("stored_opts = "))
        self.assertIn("if k in opts.data}", segment, "only keys in opts.data are restored")
        self.assertIn("if p.override_settings_restore_afterwards:", segment)
        self.assertIn("set_config(stored_opts, save_config=False)", segment)
        self.assertIn('_vae_override = p.override_settings.pop("sd_vae", None)', segment)
        self.assertIn("_overridden_modules = shared.opts.forge_additional_modules.copy()", segment)

    def test_cell_applies_the_axes_outside_the_try_before_process_images(self):
        source = forge_source("scripts/xyz_grid.py")
        cell = _function(ast.parse(source), "cell")
        applies, try_index = [], None
        for index, statement in enumerate(cell.body):
            text = ast.get_source_segment(source, statement)
            if isinstance(statement, ast.Expr) and ".apply(pc, " in text:
                applies.append(index)
            if isinstance(statement, ast.Try) and "process_images(pc)" in text:
                try_index = index
        self.assertEqual(len(applies), 3)
        self.assertIsNotNone(try_index)
        self.assertLess(max(applies), try_index)
        self.assertIn("pc = copy(p)", ast.get_source_segment(source, cell))

    def test_set_config_sends_modules_through_modules_change(self):
        source = forge_source("modules/sysinfo.py")
        segment = ast.get_source_segment(source, _function(ast.parse(source), "set_config"))
        self.assertIn('if k == "forge_additional_modules":', segment)
        self.assertIn("main_entry.modules_change(v, preset=None, save=False, refresh=False)", segment)
        self.assertIn("main_entry.checkpoint_change(v, preset=None, save=False, refresh=False)", segment)

    def test_infotext_module_list(self):
        source = forge_source("modules/processing.py")
        segment = ast.get_source_segment(source, _function(ast.parse(source), "create_infotext"))
        self.assertIn("for i, m in enumerate(_overridden_modules or shared.opts.forge_additional_modules):", segment)
        self.assertIn('generation_params[f"Module {i+1}"] = os.path.splitext(os.path.basename(m))[0]', segment)

    def test_axis_option_signature(self):
        source = forge_source("scripts/xyz_grid.py")
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AxisOption")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        self.assertEqual([a.arg for a in init.args.args],
                         ["self", "label", "type", "apply", "format_value", "confirm", "cost", "choices", "prepare"])

    def test_module_list_is_keyed_by_file_name(self):
        source = forge_source("modules_forge/main_entry.py")
        tree = ast.parse(source)
        finder = ast.get_source_segment(source, _function(tree, "find_files_with_extensions"))
        self.assertIn("found_files[file] = full_path", finder)
        change = ast.get_source_segment(source, _function(tree, "modules_change"))
        self.assertIn("module_name = os.path.basename(v)", change)
        self.assertIn("modules.sort()", change)


if __name__ == "__main__":
    unittest.main()
