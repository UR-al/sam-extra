"""XYZ ``VAE`` axis (Forge's, replaced in place) against Forge's own code (Forge Neo 2.30 / 2.31; skipped without a
Forge checkout).

1. **A grid through Forge's code, on CPU** — Forge's ``modules/sd_vae.py`` (the whole file: ``refresh_vae_list``, the
   strict in-place swap ``reload_vae_weights`` / ``_load_vae_dict`` and its restore), ``processing.process_images`` /
   ``manage_model_and_prompt_cache``, ``sd_models.forge_model_reload`` / ``SdModelData`` / ``FakeInitialModel``,
   ``backend/utils.py`` ``load_torch_file``, ``backend/state_dict.py`` ``load_state_dict``, ``sysinfo.set_config``,
   ``main_entry.modules_change`` / ``checkpoint_change`` / ``refresh_model_loading_parameters`` and xyz_grid's
   ``AxisOption`` class, its ``VAE`` axis (the ``axis_options`` entry itself), ``apply_vae`` / ``find_vae`` /
   ``apply_checkpoint`` / ``format_value_add_label`` run against tiny real VAE files (a few tensors each, every tensor
   holding the file's own value). The stand-in for ``forge_loader`` builds the VAE the way ``backend/loader.py`` does:
   from the last VAE file of the module list (else the checkpoint's), the 2D ``Qwen2DVAE`` layout when the file has
   ``post_quant_conv.weight``, else the 3D ``WanVAE`` layout; the stand-in for ``process_images_inner`` "decodes" by
   reading which file's weights the VAE holds. Driven like xyz_grid's ``cell`` (a failing cell is an empty image).
   The user's grid (Anima, base 3D ``qwen_image_vae``, Y = 3D / 2D / 3D / 2D / 2D) leaves three black cells with
   Forge 2.30's axis and none with the replacement, also in another order and on a 2D base; cells Forge can swap take
   Forge's path call for call; restore after a failing cell, a failing load, an interrupt; labels and ``Module N``;
   the [Modules] Text Encoder and Checkpoint axes in every position; an API caller's module list; a VAE built into the
   checkpoint. Forge 2.31's axis loads the model with the VAE in the cell's module list (no black cell, nothing
   swapped in place) - the same images, other loads; the replacement then only makes the value the cell's one VAE
   (Forge's own list keeps a VAE of the selection that is not one of the values, and its "None" can lose the
   selection's VAE).
2. **Forge's real VAE classes** (a subprocess, CPU, ``--cpu``): tiny ``WanVAE`` / ``Qwen2DVAE`` - ``vae_family`` matches
   the class Forge's loader picks, and ``swappable`` is exactly whether Forge's ``_load_vae_dict`` loads a file into a
   model; the same for the checkout's own ``models/VAE`` files (headers only, full-size models on the meta device).
3. **Source guards** — the Forge code the replacement relies on is still shaped the same way.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import gc
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest import mock

import safetensors
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import xyz_text_encoder as xte  # noqa: E402
from sam3ext import xyz_text_encoder_cell as cell_runtime  # noqa: E402
from sam3ext import xyz_vae as xvae  # noqa: E402
from tests import _xyz_text_encoder_support as tfx  # noqa: E402
from tests import _xyz_vae_support as fx  # noqa: E402
from tests._extra_schedulers_support import forge_function, forge_root, forge_source  # noqa: E402
from tests._forge_checkout import require_forge_file  # noqa: E402
from tests.test_xyz_text_encoder_origin import forge_axes_write_settings  # noqa: E402
from tests.test_xyz_text_encoder_script import load_script  # noqa: E402

V3 = "qwen_image_vae.safetensors"
V2HDR = "qwen_image_vae_2dHDR_fp32.safetensors"
V3HDR = "qwen_image_vae_HDR_fp32.safetensors"
V2D = "Qwen2D-Anime-dense_epoch_1.safetensors"
V2E = "Qwen2D-Anime-dense-440k_ema_epoch_1.safetensors"
VALUE = {name: spec[2] for name, spec in fx.USER_VAES.items()}
FAMILY = {name: spec[0] for name, spec in fx.USER_VAES.items()}
TE = "qwen_3_06b_base.safetensors"
TE_FP32 = "qwen_3_06b_base_full_fp32.safetensors"
CKPT_A = "anima.safetensors"
CKPT_B = "anima_other.safetensors"
CKPT_AIO = "anima_with_vae.safetensors"
AIO_VALUE = 7.0
WIDE_2D = "Qwen2D-wide.safetensors"        # a 2D file Forge's 2D layout cannot take (other latent width)

FORGE_FILES = ("modules/processing.py", "modules/sysinfo.py", "modules_forge/main_entry.py", "modules/sd_models.py",
               "modules/sd_vae.py", "modules/util.py", "scripts/xyz_grid.py", "backend/utils.py",
               "backend/state_dict.py", "backend/loader.py")


def stems(*names):
    return sorted(os.path.splitext(name)[0] for name in names)


def _forge_axis_option(source: str, label: str) -> str:
    """The source of the ``axis_options`` entry with ``label`` in Forge's scripts/xyz_grid.py."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "axis_options" for t in node.targets):
            for element in node.value.elts:
                if (isinstance(element, ast.Call) and element.args and isinstance(element.args[0], ast.Constant)
                        and element.args[0].value == label):
                    return ast.get_source_segment(source, element)
    raise LookupError(label)


def module_from_shapes(shapes: dict) -> torch.nn.Module:
    """A module whose parameters have exactly these names and shapes (the "class" of a VAE layout)."""
    root = torch.nn.Module()
    for key, shape in shapes.items():
        *parents, leaf = key.split(".")
        node = root
        for part in parents:
            if part not in node._modules:
                node.add_module(part, torch.nn.Module())
            node = node._modules[part]
        node.register_parameter(leaf, torch.nn.Parameter(torch.zeros(shape), requires_grad=False))
    return root


class Engine:
    """The model object forge_model_reload sets up (the attributes it and the hook read)."""

    def __init__(self, first_stage_model, family, loaded):
        self.first_stage_model = first_stage_model
        self.family = family
        self.loaded = loaded
        self.forge_objects = types.SimpleNamespace(vae=types.SimpleNamespace(upscale_ratio=8))


class CheckpointInfo(tfx.CheckpointInfo):
    def __init__(self, name, shorthash, filename):
        super().__init__(name, shorthash, filename)
        self.sha256 = shorthash * 16

    def calculate_shorthash(self):
        return self.ids[2]


class Shared:
    """``modules.shared``: ``sd_model`` reads the loaded model like Forge's (a property over model_data)."""

    def __init__(self, opts, model_data):
        self.opts = opts
        self._model_data = model_data
        self.cmd_opts = types.SimpleNamespace(text_encoder_dirs=[], vae_dirs=[], embeddings_dir=None)
        self.config_filename = "config.json"
        self.state = types.SimpleNamespace(time_start=1.0, job="1 out of 5", interrupted=False,
                                           stopping_generation=False)
        self.natural_sort_key = None

    @property
    def sd_model(self):
        return self._model_data.sd_model


class P:
    def __init__(self, scripts, override_settings=None):
        self.scripts = scripts
        self.override_settings = {} if override_settings is None else override_settings
        self.override_settings_restore_afterwards = True
        self.styles = []
        self.seed = 1
        self.nested = False

    def clear_prompt_cache(self):
        pass


class VaeGrid:
    """Forge's code from the checkout, wired together, over tiny real VAE files."""

    def __init__(self, folder, ui=(V3, TE), extra=None):
        models = Path(folder) / "models"
        self.models_path = str(models)
        self.module_list, self.module_map = {}, {}
        for name, path in fx.write_user_vaes(models / "VAE").items():
            self.module_list[name], self.module_map[name] = path, name
        for name, spec in {TE: tfx.CASES["qwen3_06b"][0], TE_FP32: tfx.CASES["qwen3_06b"][0]}.items():
            self.module_list[name] = str(fx.write_vae(models / "text_encoder" / name, spec, 0.5))
            self.module_map[name] = name
        for name, (shapes, value) in (extra or {}).items():
            self.module_list[name] = str(fx.write_vae(models / "VAE" / name, shapes, value))
            self.module_map[name] = name
        unet = {"net.blocks.0.self_attn.q_proj.weight": [2, 2]}
        ckpt_dir = models / "Stable-diffusion"
        infos = [CheckpointInfo(CKPT_A, "aaaa", str(fx.write_vae(ckpt_dir / CKPT_A, unet, 0.0))),
                 CheckpointInfo(CKPT_B, "bbbb", str(fx.write_vae(ckpt_dir / CKPT_B, unet, 0.0))),
                 CheckpointInfo(CKPT_AIO, "cccc", str(fx.write_vae(
                     ckpt_dir / CKPT_AIO, {**unet, **{f"vae.{k}": v for k, v in fx.tiny_3d().items()}}, AIO_VALUE)))]
        self.ckpt = {info.name: info for info in infos}
        self.loads, self.cells, self.nested, self.reported, self.errors = [], [], [], [], []
        self.swaps, self.grid_module_n, self.settings_seen, self.labels = [], [], [], []
        self.applied, self.seen_axes, self.current_axes = [], [], {}   # {axis label: value} per cell / settings_seen
        self.fail_cells, self.events = set(), {}
        self.stderr = io.StringIO()
        self.ui_modules = self.paths(*ui)
        self.opts = tfx.FakeOpts(
            {"sd_model_checkpoint": self.ckpt[CKPT_A].title, "forge_additional_modules": list(self.ui_modules),
             "forge_unet_storage_dtype": "Automatic", "sd_vae": "Automatic"},
            tfx.forge_selection_defaults(forge_source("modules/shared_options.py")))

        # modules.sd_models: Forge's SdModelData / FakeInitialModel / forge_model_reload / checkpoint lookup
        sd_models = self.sd_models = types.ModuleType("modules.sd_models")
        forge_function("modules/sd_models.py", "FakeInitialModel", sd_models.__dict__)
        forge_function("modules/sd_models.py", "SdModelData", sd_models.__dict__)
        sd_models.model_data = sd_models.SdModelData()
        shared = self.shared = Shared(self.opts, sd_models.model_data)
        util = {"re": re, "os": os}
        shared.natural_sort_key = forge_function("modules/util.py", "natural_sort_key", util)
        aliases = {alias: info for info in infos for alias in info.ids}
        sd_models.__dict__.update(checkpoint_aliases=aliases, checkpoints_list={i.title: i for i in infos}, re=re,
                                  sys=sys, shared=shared, re_strip_checksum=re.compile(r"\s*\[[^]]+]\s*$"),
                                  model_path=str(ckpt_dir))
        forge_function("modules/sd_models.py", "get_closet_checkpoint_match", sd_models.__dict__)
        forge_function("modules/sd_models.py", "select_checkpoint", sd_models.__dict__)

        backend_utils = {"torch": torch, "safetensors": safetensors, "DISABLE_MMAP": False, "MMAP_TORCH_FILES": False}
        self.load_torch_file = forge_function("backend/utils.py", "load_torch_file", backend_utils)
        self.forge_load_state_dict = forge_function("backend/state_dict.py", "load_state_dict", {"torch": torch})

        proc = self.proc = types.ModuleType("modules.processing")
        memory = types.SimpleNamespace(unload_all_models=lambda: None, soft_empty_cache=lambda: None,
                                       logger=logging.getLogger("xyz-vae-test"))
        reload_ns = {"torch": torch, "model_data": sd_models.model_data, "gc": gc, "memory_management": memory,
                     "FakeInitialModel": sd_models.FakeInitialModel, "Timer": _Timer,
                     "dynamic_args": types.SimpleNamespace(), "cmd_opts": shared.cmd_opts,
                     "forge_loader": self._forge_loader, "shared": shared, "processing": proc,
                     "script_callbacks": types.SimpleNamespace(model_loaded_callback=lambda model: None)}
        sd_models.forge_model_reload = forge_function("modules/sd_models.py", "forge_model_reload", reload_ns)

        main_entry = self.main_entry = types.ModuleType("modules_forge.main_entry")
        main_entry.__dict__.update(   # module_map only where this Forge has it (2.30-6): 2.29.2 / 2.30-2 run without
            tfx.forge_module_lists(forge_source("modules_forge/main_entry.py"), self.module_list, self.module_map),
            shared=shared, sd_models=sd_models, os=os,
            torch=types.SimpleNamespace(float16="float16", bfloat16="bfloat16"),
            forge_unet_storage_dtype_options={"Automatic": (None, False)}, logger=logging.getLogger("xyz-vae-test"),
            print_json=lambda **kwargs: None, dynamic_args=types.SimpleNamespace(online_lora=False), processing=proc)
        for name in ("modules_change", "checkpoint_change", "refresh_model_loading_parameters"):
            forge_function("modules_forge/main_entry.py", name, main_entry.__dict__)
        main_entry.refresh_models = lambda: ([], sorted(self.module_list))

        sysinfo = types.ModuleType("modules.sysinfo")
        sysinfo.Any = Any
        forge_function("modules/sysinfo.py", "set_config", sysinfo.__dict__)

        modules = self.modules = types.ModuleType("modules")
        modules.__path__ = []
        paths = types.ModuleType("modules.paths")
        paths.models_path = self.models_path
        hashes = types.SimpleNamespace(sha256=lambda filename, title: None)
        modules.shared, modules.sd_models, modules.processing, modules.paths, modules.sysinfo, modules.hashes = (
            shared, sd_models, proc, paths, sysinfo, hashes)
        backend = types.ModuleType("backend")
        backend.__path__ = []
        backend.memory_management = memory
        backend.utils = types.SimpleNamespace(load_torch_file=self.load_torch_file)
        # modules.sd_vae: Forge's whole file
        sd_vae = self.sd_vae = types.ModuleType("modules.sd_vae")
        with mock.patch.dict(sys.modules, {"modules": modules, "modules.shared": shared, "modules.paths": paths,
                                           "modules.sd_models": sd_models, "modules.hashes": hashes,
                                           "backend": backend, "backend.memory_management": memory,
                                           "backend.utils": backend.utils}):
            exec(compile(forge_source("modules/sd_vae.py"), "<forge:modules/sd_vae.py>", "exec"), sd_vae.__dict__)
        modules.sd_vae = sd_vae
        sd_vae.refresh_vae_list()
        self.forge_reload_vae_weights = sd_vae.reload_vae_weights
        sd_vae.reload_vae_weights = self._reload_vae_weights

        proc.__dict__.update(
            opts=self.opts, shared=shared, sd_models=sd_models, set_config=sysinfo.set_config, os=os, sd_vae=sd_vae,
            sd_samplers=types.SimpleNamespace(fix_p_invalid_sampler_and_scheduler=lambda p: None),
            profiling=types.SimpleNamespace(Profiler=contextlib.nullcontext), process_images_inner=self._inner,
            forge_model_reload=sd_models.forge_model_reload, memory_management=memory, need_global_unload=False,
            _overridden_modules=None, StableDiffusionProcessing=object, Processed=object, Optional=Optional)
        forge_function("modules/processing.py", "manage_model_and_prompt_cache", proc.__dict__)
        forge_function("modules/processing.py", "process_images", proc.__dict__)

        modules_forge = types.ModuleType("modules_forge")
        modules_forge.__path__ = []
        modules_forge.main_entry = main_entry
        self.sys_modules = {"modules": modules, "modules.shared": shared, "modules.sd_models": sd_models,
                            "modules.processing": proc, "modules.paths": paths, "modules.sysinfo": sysinfo,
                            "modules.sd_vae": sd_vae, "modules_forge": modules_forge,
                            "modules_forge.main_entry": main_entry}

        # xyz_grid: Forge's AxisOption, the functions its VAE / Checkpoint axes use, and its "VAE" / "Nothing" entries
        xyz = self.xyz = types.ModuleType("xyz_grid.py")
        xyz.__dict__.update(modules=modules, opts=self.opts, shared=shared, select_checkpoint=sd_models.select_checkpoint,
                            model_data=sd_models.model_data, sd_vae=sd_vae, StableDiffusionProcessing=object, T=Any,
                            os=os, get_closet_checkpoint_match=sd_models.get_closet_checkpoint_match,
                            checkpoints_list=sd_models.checkpoints_list)
        for name in ("format_value_add_label", "do_nothing", "format_nothing", "format_remove_path",
                     "refresh_loading_params_for_xyz_grid", "find_vae", "apply_vae", "apply_checkpoint",
                     "AxisOption"):
            forge_function("scripts/xyz_grid.py", name, xyz.__dict__)
        xyz_source = forge_source("scripts/xyz_grid.py")
        self.nothing = eval(_forge_axis_option(xyz_source, "Nothing"), xyz.__dict__)
        self.stock_vae_axis = eval(_forge_axis_option(xyz_source, "VAE"), xyz.__dict__)
        # Forge's cost for it (1.0 on 2.30, 2.0 from 2.31 - AxisOption reads opts.xyz_use_cost there)
        cost = next(ast.literal_eval(k.value) for k in ast.parse(_forge_axis_option(xyz_source, "Checkpoint name"),
                                                                 mode="eval").body.keywords if k.arg == "cost")
        self.ckpt_axis = xyz.AxisOption("Checkpoint name", str, xyz.apply_checkpoint, cost=cost,
                                        format_value=xyz.format_remove_path)
        xyz.axis_options = [self.nothing, self.stock_vae_axis]

        self.script, _ = load_script()
        self.script.register_xyz_axes(xyz)
        self.script.replace_vae_axis(xyz)
        self.vae_axis = xyz.axis_options[1]
        self.te_axis = xyz.axis_options[2]
        self.hook = self.script.XyzTextEncoderHook()
        self.runner = types.SimpleNamespace(before_process=self._before_process, process=self._process,
                                            postprocess_image=self._postprocess_image, postprocess=self._postprocess)
        with mock.patch.dict(sys.modules, self.sys_modules):
            main_entry.refresh_model_loading_parameters()     # Forge's state after startup
        self.ui_loading = self.describe(sd_models.model_data.forge_loading_parameters)
        self.initial_settings = copy.deepcopy(self.opts.data)

    # -- helpers --
    def paths(self, *names):
        return sorted(self.module_list[name] for name in names)

    @staticmethod
    def describe(params):
        return params["checkpoint_info"].name, tuple(sorted(os.path.basename(m) for m in params["additional_modules"]))

    @property
    def params(self):
        return self.sd_models.model_data.forge_loading_parameters

    def settings(self):
        """Forge's settings without the runtime key forge_model_reload writes (``sd_checkpoint_hash``)."""
        return {k: v for k, v in self.opts.data.items() if k != "sd_checkpoint_hash"}

    # -- stand-ins below Forge's code --
    def _forge_loader(self, checkpoint, additional_state_dicts=None):
        """backend/loader.py as far as the VAE goes: replace_state_dict's VAE branch (each VAE file replaces the VAE,
        else the checkpoint's own ``vae.`` keys stay), the class by ``post_quant_conv.weight``, Forge's
        ``load_state_dict`` on a model built on the meta device (``no_init_weights``)."""
        vae = {k[len("vae."):]: v for k, v in self.load_torch_file(checkpoint).items() if k.startswith("vae.")} or None
        for path in additional_state_dicts or []:
            sd = self.load_torch_file(path)
            if "decoder.conv_in.weight" in sd or "decoder.middle.0.residual.0.gamma" in sd:
                vae = sd
        if vae is None:
            raise ValueError("You do not have VAE state dict!")
        family = xvae.FAMILY_2D if "post_quant_conv.weight" in vae else xvae.FAMILY_3D
        with torch.device("meta"):
            first_stage = module_from_shapes(fx.FAMILY_SHAPES[family]())
        with contextlib.redirect_stdout(io.StringIO()):
            self.forge_load_state_dict(first_stage, dict(vae))
        loaded = (os.path.basename(checkpoint), tuple(sorted(os.path.basename(m) for m in additional_state_dicts or [])))
        self.loads.append(loaded)
        return Engine(first_stage, family, loaded)

    def _reload_vae_weights(self, vae):
        self.swaps.append(os.path.basename(vae) if vae not in (None, "None") else vae)
        return self.forge_reload_vae_weights(vae)

    def _inner(self, p):
        """process_images_inner as far as the hook sees it, and the VAE "decode": which file's weights it holds."""
        record = self.nested if p.nested else self.cells
        index = len(record)
        self.runner.process(p)
        if not p.nested:
            self.settings_seen.append(self.settings())
            self.seen_axes.append(dict(self.current_axes))
            for event in self.events.get(index, ()):
                event(self, p)
        self.runner.postprocess_image(p, None)
        model = self.shared.sd_model
        values = {float(t.flatten()[0]) for t in model.first_stage_model.state_dict().values()}
        modules = self.proc.__dict__["_overridden_modules"] or self.opts.forge_additional_modules
        record.append({"family": model.family, "vae": values.pop() if len(values) == 1 else sorted(values),
                       "model": model.loaded,
                       "module_n": sorted(os.path.splitext(os.path.basename(m))[0] for m in modules)})
        if not p.nested and index in self.fail_cells:
            raise RuntimeError(f"cell {index} failed")
        result = types.SimpleNamespace(images=[index])
        self.runner.postprocess(p, result)
        return result

    def _call(self, method, *args):
        try:
            method(*args)
        except Exception as exc:
            self.reported.append(exc)

    def _before_process(self, p):
        self._call(self.hook.before_process, p)

    def _process(self, p):
        self._call(self.hook.process, p)

    def _postprocess_image(self, p, pp):
        self._call(self.hook.postprocess_image, p, pp)

    def _postprocess(self, p, processed):
        self._call(self.hook.postprocess, p, processed)

    # -- xyz_grid's run / cell --
    def run(self, p, x=None, y=None, z=None):
        nothing = (self.nothing, [0])
        (x_opt, xs), (y_opt, ys), (z_opt, zs) = x or nothing, y or nothing, z or nothing
        with (mock.patch.dict(sys.modules, self.sys_modules), contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(self.stderr)):
            for opt, values in ((x_opt, xs), (y_opt, ys), (z_opt, zs)):
                if opt.confirm:
                    opt.confirm(p, values)
            self.labels = [[opt.format_value(p, opt, v) for v in values]
                           for opt, values in ((x_opt, xs), (y_opt, ys), (z_opt, zs))]

            def cell(xv, yv, zv):
                if self.shared.state.interrupted:
                    self.cells.append({"skipped": True})       # xyz_grid's cell returns an empty Processed
                    return
                self.current_axes = {opt.label: v for opt, v in ((x_opt, xv), (y_opt, yv), (z_opt, zv))}
                self.applied.append(self.current_axes)
                pc = copy.copy(p)
                pc.styles = pc.styles[:]
                x_opt.apply(pc, xv, xs)
                y_opt.apply(pc, yv, ys)
                z_opt.apply(pc, zv, zs)
                before = len(self.cells)
                try:
                    self.proc.process_images(pc)
                except Exception as exc:                        # errors.display; Image.new: a black cell
                    self.errors.append(exc)
                    if len(self.cells) == before:
                        self.cells.append({"black": True})
                    else:
                        self.cells[-1]["black"] = True
                modules = self.proc.__dict__["_overridden_modules"] or self.opts.forge_additional_modules
                self.grid_module_n.append(stems(*map(os.path.basename, modules)))   # create_infotext(pc) after it

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

    def stock_writes(self, axes, base_modules) -> dict:
        """What Forge 2.31's stock Checkpoint / VAE axes put into the cell's ``override_settings`` (and, through
        ``set_config``, into Forge's settings for the cell) - the VAE as the replaced axis writes it: exactly the value
        on ``base_modules``. Nothing on 2.30."""
        out = {}
        if not forge_axes_write_settings():
            return out
        if axes.get("Checkpoint name"):
            out["sd_model_checkpoint"] = self.sd_models.get_closet_checkpoint_match(axes["Checkpoint name"]).name
        if "VAE" in axes:
            vae = axes["VAE"]
            kept = [m for m in base_modules if vae in (None, "None") or os.path.basename(m) not in FAMILY]
            out["forge_additional_modules"] = sorted(set(kept + ([] if vae in (None, "None") else
                                                                 [self.module_list[vae]])))
        return out

    def settings_during(self, axes) -> dict:
        """Forge's settings while the cell with ``axes`` renders: the user's (2.31: with the stock axes' values)."""
        out = {k: v for k, v in self.initial_settings.items() if k != "sd_checkpoint_hash"}
        for key, value in self.stock_writes(axes, out.get("forge_additional_modules") or []).items():
            if out.get(key) != value:      # set_config skips an unchanged value
                out[key] = copy.deepcopy(value)
        return out

    def generate(self, p=None):
        """A generation outside a grid (txt2img without a script); its record."""
        self.current_axes = {}
        with (mock.patch.dict(sys.modules, self.sys_modules), contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(self.stderr)):
            self.proc.process_images(p or P(self.runner))
        return self.cells[-1]


class _Timer:
    def record(self, *args, **kwargs):
        pass

    def summary(self):
        return ""


def portable(value, root: str):
    """``value`` with the grid's temporary models folder written as ``<models>`` (two grids compare equal)."""
    if isinstance(value, str):     # also as repr() writes it (str(dict) of the loading parameters)
        return value.replace(root.replace("\\", "\\\\"), "<models>").replace(root, "<models>")
    if isinstance(value, dict):
        return {portable(k, root): portable(v, root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(portable(v, root) for v in value)
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {portable(str(value), root)}"
    return value


def cells_of(grid, *keys):
    keys = keys or ("family", "vae", "module_n")
    return [{key: cell.get(key) for key in keys} if not (cell.get("black") or cell.get("skipped"))
            else ("black" if cell.get("black") else "skipped") for cell in grid.cells]


def made(name, te=TE):
    """What a cell rendered with VAE ``name`` (and encoder ``te``) shows: its family, its weights, its Module N."""
    return {"family": FAMILY[name], "vae": VALUE[name], "module_n": stems(name, te)}


class GridCase(unittest.TestCase):
    UI = (V3, TE)
    EXTRA = None

    @classmethod
    def setUpClass(cls):
        for rel in FORGE_FILES:
            require_forge_file(rel)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        xte.clear_cache()
        cell_runtime._STATE = None
        self.grid = VaeGrid(self.tmp.name, self.UI, self.EXTRA)
        self.p = P(self.grid.runner)
        self.parent_settings = self.p.override_settings

    def tearDown(self):
        cell_runtime._STATE = None
        xte.clear_cache()
        self.tmp.cleanup()

    def assertRestored(self, parent_settings=None):
        """Forge's settings and loading parameters are the user's again, config.json was not saved, nothing of the
        hook is left. During each cell a page load would save the user's settings - from Forge 2.31 with the stock
        Checkpoint / VAE axes' values in them (Forge's own ``set_config``), and the last cell's stay in the parent's
        ``override_settings`` (every cell shares that dict)."""
        g = self.grid
        initial = {k: v for k, v in g.initial_settings.items() if k != "sd_checkpoint_hash"}
        self.assertEqual(g.settings(), initial, "Forge's settings are the user's")
        self.assertEqual(len(g.seen_axes), len(g.settings_seen))
        for seen, axes in zip(g.settings_seen, g.seen_axes):
            self.assertEqual(seen, g.settings_during(axes), "a page load during a cell saves these")
        self.assertEqual(g.opts.saved, [], "config.json is never saved")
        self.assertEqual(g.describe(g.params), g.ui_loading)
        self.assertIs(self.p.override_settings, self.parent_settings)
        parent = dict(parent_settings or {})
        if g.applied:
            parent.update(g.stock_writes(g.applied[-1], initial.get("forge_additional_modules") or []))
        self.assertEqual({k: list(v) if isinstance(v, list) else v for k, v in self.p.override_settings.items()}, parent)
        for attr in (xte.CELL_ATTR, xvae.CELL_ATTR):
            self.assertFalse(hasattr(self.p, attr))
        self.assertEqual(g.reported, [])
        self.assertIsNone(g.proc.__dict__["_overridden_modules"])
        self.assertIsNone(cell_runtime.state())
        self.assertIsNone(g.sd_vae.base_vae, "Forge's swap restored its VAE")
        self.assertIsNone(g.sd_vae.loaded_vae_file)

    def assertNextGenerationIsTheUsers(self):
        g = self.grid
        cell = g.generate()
        ui = [os.path.basename(m) for m in g.ui_modules]
        vae = next(name for name in ui if name in FAMILY)
        self.assertEqual(cell["model"], g.ui_loading)
        self.assertEqual((cell["family"], cell["vae"]), (FAMILY[vae], VALUE[vae]))
        self.assertEqual(cell["module_n"], stems(*ui))


class UsersGridTests(GridCase):
    """Anima with qwen_image_vae (3D) + qwen_3_06b_base, a generation done, then Y = the five VAEs."""

    def setUp(self):
        super().setUp()
        self.grid.generate()                       # the user had generated before: the model is loaded
        self.grid.loads.clear()
        self.grid.cells.clear()

    def test_forges_own_axis_on_the_users_grid(self):
        """The bug, reproduced through Forge 2.30's code: strict load_state_dict of a 2D file into the 3D VAE raises,
        xyz_grid draws an empty image for the cell. Forge 2.31's axis loads the model with each VAE instead: every
        cell renders (the selection's VAE is one of the values here)."""
        g = self.grid
        g.run(self.p, y=(g.stock_vae_axis, list(fx.USER_ORDER)))
        if forge_axes_write_settings():
            self.assertEqual(cells_of(g), [made(name) for name in fx.USER_ORDER])
            self.assertEqual(g.errors, [])
            self.assertEqual(g.loads, [(CKPT_A, tuple(sorted((name, TE)))) for name in fx.USER_ORDER[1:]])
            self.assertEqual(g.swaps, [])
            return
        self.assertEqual(cells_of(g), [made(V3), "black", made(V3HDR), "black", "black"])
        self.assertEqual([type(e).__name__ for e in g.errors], ["RuntimeError"] * 3)
        self.assertIn("Missing key(s) in state_dict", str(g.errors[0]))
        self.assertEqual(g.loads, [], "Forge's axis never reloads the model")

    def test_the_replaced_axis_renders_every_cell(self):
        g = self.grid
        g.run(self.p, y=(g.vae_axis, list(fx.USER_ORDER)))
        self.assertEqual(cells_of(g), [made(V3), made(V2HDR), made(V3HDR), made(V2D), made(V2E)])
        self.assertEqual(g.errors, [])
        log = g.stderr.getvalue()
        if forge_axes_write_settings():
            # Forge 2.31: every VAE loads the model with it (the first is the loaded model's), nothing is swapped
            self.assertEqual(g.loads, [(CKPT_A, tuple(sorted((name, TE)))) for name in fx.USER_ORDER[1:]])
            self.assertEqual(g.swaps, [])
            self.assertEqual([cell["model"] for cell in g.cells],
                             [(CKPT_A, tuple(sorted((name, TE)))) for name in fx.USER_ORDER])
            self.assertEqual(log, "")
            self.assertRestored()
            self.assertNextGenerationIsTheUsers()
            return
        # the 2D cells load the model with their VAE; the 3D HDR cell is Forge's in-place swap on the user's model
        self.assertEqual(g.loads, [(CKPT_A, tuple(sorted((V2HDR, TE)))), (CKPT_A, tuple(sorted((V3, TE)))),
                                   (CKPT_A, tuple(sorted((V2D, TE)))), (CKPT_A, tuple(sorted((V2E, TE))))])
        self.assertEqual(g.swaps, [None, V3HDR], "Forge's swap runs for the 3D cells only (None: the loaded VAE)")
        self.assertEqual([cell["model"] for cell in g.cells],
                         [(CKPT_A, tuple(sorted(m))) for m in ((V3, TE), (V2HDR, TE), (V3, TE), (V2D, TE), (V2E, TE))])
        # the console says why, naming both families (the first 2D cell against the loaded model)
        self.assertIn(f"[XYZ VAE] {V2HDR} is a 2D Qwen2D VAE ({fx.Z} ch) and the loaded model's VAE is a 3D Wan / "
                      f"Qwen-Image VAE ({fx.Z} ch)", log)
        self.assertIn(f"[XYZ VAE] {V2E} is a 2D Qwen2D VAE ({fx.Z} ch) and the VAE the model is built with is a 3D",
                      log)
        self.assertEqual(log.count("[XYZ VAE]"), 3)
        self.assertRestored()
        self.assertNextGenerationIsTheUsers()

    def test_another_order(self):
        """3D / 2D / 2D / 3D / 2D: two 2D cells in a row each load their own VAE; the 3D cell after them swaps."""
        g = self.grid
        order = [V3, V2HDR, V2D, V3HDR, V2E]
        g.run(self.p, y=(g.vae_axis, order))
        self.assertEqual(cells_of(g), [made(name) for name in order])
        self.assertEqual(g.swaps, [] if forge_axes_write_settings() else [None, V3HDR])
        self.assertRestored()
        self.assertNextGenerationIsTheUsers()

    def test_labels_and_infotext(self):
        g = self.grid
        g.run(self.p, y=(g.vae_axis, list(fx.USER_ORDER)))
        # Forge's labels (format_value_add_label), the same as Forge's axis gives
        self.assertEqual(g.labels[1], [f"VAE: {name}" for name in fx.USER_ORDER])
        self.assertEqual(g.labels[1], [g.stock_vae_axis.format_value(self.p, g.stock_vae_axis, name)
                                       for name in fx.USER_ORDER])
        # each image's Module N: the files the cell used; after each cell (the grid image's infotext): the user's
        self.assertEqual([cell["module_n"] for cell in g.cells], [stems(name, TE) for name in fx.USER_ORDER])
        self.assertEqual(g.grid_module_n, [stems(V3, TE)] * 5)

    def test_a_failing_2d_cell_is_put_back(self):
        g = self.grid
        g.fail_cells = {1}
        g.run(self.p, y=(g.vae_axis, list(fx.USER_ORDER)))
        self.assertEqual(cells_of(g)[1], "black")
        self.assertEqual(cells_of(g)[2:], [made(V3HDR), made(V2D), made(V2E)])
        self.assertRestored()
        self.assertNextGenerationIsTheUsers()

    def test_a_failing_last_cell_is_put_back_by_the_next_task(self):
        g = self.grid
        g.fail_cells = {4}
        g.run(self.p, y=(g.vae_axis, list(fx.USER_ORDER)))
        self.assertEqual(cells_of(g)[4], "black")
        # xyz_grid rebuilt the loading parameters after the grid; the Module N list waits for the next task (2.30:
        # the 2D cell went to the loader through the hook; 2.31 loads it without the hook)
        self.assertEqual(g.describe(g.params), g.ui_loading)
        self.assertEqual(cell_runtime.state() is None, forge_axes_write_settings())
        g.shared.state.time_start, g.shared.state.job = 2.0, "task(2)"
        self.assertNextGenerationIsTheUsers()
        self.assertIsNone(cell_runtime.state())
        self.assertIsNone(g.proc.__dict__["_overridden_modules"])

    def test_a_vae_the_model_cannot_load_fails_like_the_dropdown(self):
        """A 2D file of another latent width: Forge's own load fails as it would from the dropdown (an error, not a
        silently wrong VAE); the next cell is fine."""
        self.tearDown()
        self.EXTRA = {WIDE_2D: (fx.tiny_2d(z=8), 6.0)}
        self.setUp()
        g = self.grid
        g.run(self.p, y=(g.vae_axis, [WIDE_2D, V2HDR, V3HDR]))
        self.assertEqual(cells_of(g), ["black", made(V2HDR), made(V3HDR)])
        self.assertIn("size mismatch", str(g.errors[0]))
        self.assertRestored()
        self.assertNextGenerationIsTheUsers()

    def test_interrupted(self):
        g = self.grid

        def interrupt(grid, p):
            grid.shared.state.interrupted = True

        g.events[1] = [interrupt]
        g.run(self.p, y=(g.vae_axis, list(fx.USER_ORDER)))
        self.assertEqual(cells_of(g), [made(V3), made(V2HDR), "skipped", "skipped", "skipped"])
        self.assertRestored()
        g.shared.state.interrupted = False
        self.assertNextGenerationIsTheUsers()

    def test_same_family_cells_take_forges_path_call_for_call(self):
        """Values Forge can swap: the run with the replaced axis is the run with Forge's axis - the same loads, swaps,
        images, Module N lists, settings and override_settings at every step."""
        forge_231 = forge_axes_write_settings()
        # Forge 2.31: values whose list Forge's own axis gets right (its "None" drops the selection's VAE when that VAE
        # is one of the values - test_exactly_the_value_unlike_forges_own_axis)
        values = [V3HDR, V3, V3HDR] if forge_231 else [V3HDR, "None", V3, V3HDR]
        traces = []
        for axis in ("stock_vae_axis", "vae_axis"):
            self.tearDown()
            self.setUp()
            g = self.grid
            g.run(self.p, y=(getattr(g, axis), values))
            traces.append(portable({
                "cells": g.cells, "loads": g.loads, "swaps": g.swaps, "grid_module_n": g.grid_module_n,
                "settings": g.settings_seen, "errors": g.errors,
                # (2.31: the last cell's module list stays there - Forge's own unsorted, the replacement's sorted as
                # modules_change keeps it)
                "parent": {k: sorted(v) if isinstance(v, list) else v for k, v in self.p.override_settings.items()},
                "log": g.stderr.getvalue(), "labels": g.labels, "saved": g.opts.saved, "after": g.settings(),
                "loading": g.describe(g.params), "hash": g.sd_models.model_data.forge_hash}, g.models_path))
        self.assertEqual(traces[0], traces[1])
        if forge_231:
            self.assertEqual(traces[0]["swaps"], [])
            self.assertEqual(traces[0]["loads"], [(CKPT_A, tuple(sorted((name, TE)))) for name in values])
        else:
            self.assertEqual(traces[0]["swaps"], [V3HDR, "None", None, V3HDR])
            self.assertEqual(traces[0]["loads"], [])
        self.assertEqual(traces[0]["log"], "")
        self.assertRestored()

    def test_exactly_the_value_unlike_forges_own_axis(self):
        """Forge 2.31's own axis drops from the selection only the files that are values of the axis: its "None" cell
        loses the selection's VAE when that VAE is one of the values (no VAE file: the checkpoint has none, the load
        fails). The replaced axis renders the selection's VAE there, as "None" does on 2.30."""
        values = [V3HDR, "None", V3]
        cells, errors = {}, {}
        for axis in ("stock_vae_axis", "vae_axis"):
            self.tearDown()
            self.setUp()
            g = self.grid
            g.run(self.p, y=(getattr(g, axis), values))
            cells[axis], errors[axis] = cells_of(g, "family", "vae"), [str(e) for e in g.errors]
            module_n = [cell.get("module_n") for cell in g.cells]
        rendered = [{"family": FAMILY[name], "vae": VALUE[name]} for name in (V3HDR, V3, V3)]
        self.assertEqual(cells["vae_axis"], rendered)
        self.assertEqual(errors["vae_axis"], [])
        if forge_axes_write_settings():
            self.assertEqual(module_n, [stems(V3HDR, TE), stems(V3, TE), stems(V3, TE)])
            self.assertEqual(cells["stock_vae_axis"], [rendered[0], "black", rendered[2]])
            self.assertEqual(errors["stock_vae_axis"], ["You do not have VAE state dict!"])
        else:      # (2.30's Module N of a "None" cell leaves the selection's VAE out - Forge's own)
            self.assertEqual(cells["stock_vae_axis"], cells["vae_axis"])
        self.assertRestored()

    def test_with_the_text_encoder_axis_in_every_position(self):
        """[Modules] Text Encoder x VAE (3D, 2D, 3D HDR): every cell gets both, whichever of X / Y / Z each axis is on
        (the Text Encoder axis stays the outer loop by cost); the same images in every arrangement."""
        te_values = ["Use Forge current", "qwen_3_06b_base_full_fp32"]
        vae_values = [V3, V2HDR, V3HDR]
        want = {(te, vae): made(vae, TE if te == "Use Forge current" else TE_FP32)
                for te in te_values for vae in vae_values}
        for te_at, vae_at in (("x", "y"), ("y", "x"), ("x", "z"), ("z", "x"), ("y", "z"), ("z", "y")):
            with self.subTest(text_encoder=te_at, vae=vae_at):
                self.tearDown()
                self.setUp()
                g = self.grid
                axes = {te_at: (g.te_axis, te_values), vae_at: (g.vae_axis, vae_values)}
                g.run(self.p, **axes)
                order = [(te, vae) for te in te_values for vae in vae_values]
                self.assertEqual(dict(zip(order, cells_of(g))), want)
                self.assertEqual(g.errors, [])
                self.assertRestored()
                self.assertNextGenerationIsTheUsers()

    def test_with_the_checkpoint_axis(self):
        g = self.grid
        g.run(self.p, x=(g.ckpt_axis, [CKPT_A, CKPT_B]), y=(g.vae_axis, [V2HDR, V3]))
        self.assertEqual(cells_of(g), [made(V2HDR), made(V3), made(V2HDR), made(V3)])
        self.assertEqual([cell["model"][0] for cell in g.cells], [CKPT_A, CKPT_A, CKPT_B, CKPT_B])
        self.assertRestored()
        self.assertNextGenerationIsTheUsers()

    def test_an_api_callers_module_list(self):
        """An API request's own override_settings["forge_additional_modules"] (2D VAE, another encoder) is the cell's
        base; Forge applies and restores it (set_config), the cell's list stands in for it. Forge 2.31's own axis
        builds every cell on the dropdown's list instead (and writes it over the caller's in that dict); the replaced
        axis keeps the caller's list as the base of every cell."""
        g = self.grid
        api = {"forge_additional_modules": [V2HDR, TE_FP32]}
        self.p = P(g.runner, api)
        self.parent_settings = api
        g.run(self.p, y=(g.vae_axis, [V3, V2D]))
        self.assertEqual(cells_of(g), [made(V3, TE_FP32), made(V2D, TE_FP32)])
        self.assertEqual(g.settings(), {k: v for k, v in g.initial_settings.items() if k != "sd_checkpoint_hash"})
        if forge_axes_write_settings():
            self.assertEqual(g.swaps, [])
            self.assertEqual([cell["model"] for cell in g.cells],
                             [(CKPT_A, tuple(sorted((V3, TE_FP32)))), (CKPT_A, tuple(sorted((V2D, TE_FP32))))])
            self.assertEqual(api, {"forge_additional_modules": g.paths(V2D, TE_FP32)}, "the last cell's list")
        else:
            self.assertEqual(g.swaps, [V2D])
            self.assertEqual([cell["model"] for cell in g.cells],
                             [(CKPT_A, tuple(sorted((V3, TE_FP32)))), (CKPT_A, tuple(sorted((V2HDR, TE_FP32))))])
            self.assertEqual(api, {"forge_additional_modules": [V2HDR, TE_FP32]})
        self.assertEqual(g.describe(g.params), g.ui_loading)
        self.assertIsNone(cell_runtime.state())


class TwoDimensionalBaseTests(GridCase):
    """The reverse: the selection's VAE is the 2D qwen_image_vae_2dHDR_fp32."""
    UI = (V2HDR, TE)

    def test_the_same_five_values(self):
        g = self.grid
        g.generate()
        g.loads.clear()
        g.cells.clear()
        g.run(self.p, y=(g.vae_axis, list(fx.USER_ORDER)))
        self.assertEqual(cells_of(g), [made(V3), made(V2HDR), made(V3HDR), made(V2D), made(V2E)])
        if forge_axes_write_settings():     # each VAE loads the model with it
            self.assertEqual(g.loads, [(CKPT_A, tuple(sorted((name, TE)))) for name in fx.USER_ORDER])
            self.assertEqual(g.swaps, [])
        else:
            self.assertEqual(g.loads, [(CKPT_A, tuple(sorted((V3, TE)))), (CKPT_A, tuple(sorted((V2HDR, TE)))),
                                       (CKPT_A, tuple(sorted((V3HDR, TE)))), (CKPT_A, tuple(sorted((V2HDR, TE))))])
            self.assertEqual(g.swaps, [None, V2D, V2E])
        self.assertRestored()
        self.assertNextGenerationIsTheUsers()

    def test_forges_own_axis_on_a_2d_base(self):
        """Forge 2.30's axis leaves the 3D cells black; 2.31's renders them (the selection's VAE is one of the values)."""
        g = self.grid
        g.run(self.p, y=(g.stock_vae_axis, list(fx.USER_ORDER)))
        if forge_axes_write_settings():
            self.assertEqual(cells_of(g), [made(name) for name in fx.USER_ORDER])
        else:
            self.assertEqual(cells_of(g), ["black", made(V2HDR), "black", made(V2D), made(V2E)])

    def test_a_selection_vae_that_is_not_a_value(self):
        """3D values on the 2D selection, which is not one of them: every cell renders its value. Forge 2.31's own axis
        keeps the selection's 2D VAE in each cell's list and the loader takes the last VAE of the sorted list: the
        qwen_image_vae cell renders the 2D selection's VAE ("." < "_"), the HDR one its value ("2" < "H"). (2.30's own
        axis cannot swap a 3D file into the 2D VAE.)"""
        g = self.grid
        g.run(self.p, y=(g.vae_axis, [V3, V3HDR]))
        self.assertEqual(cells_of(g), [made(V3), made(V3HDR)])
        self.assertRestored()
        self.tearDown()
        self.setUp()
        g = self.grid
        g.run(self.p, y=(g.stock_vae_axis, [V3, V3HDR]))
        if forge_axes_write_settings():
            self.assertEqual(cells_of(g, "family", "vae"), [{"family": "2d", "vae": VALUE[V2HDR]},
                                                            {"family": "3d", "vae": VALUE[V3HDR]}])
            self.assertEqual([cell["model"] for cell in g.cells],
                             [(CKPT_A, tuple(sorted((V2HDR, name, TE)))) for name in (V3, V3HDR)])
        else:
            self.assertEqual(cells_of(g), ["black", "black"])

    def test_first_cell_before_any_model_is_loaded(self):
        """Nothing loaded yet: the VAE the model will be built with comes from the headers (2.30; 2.31 loads the
        model with the VAE whatever it is)."""
        g = self.grid
        g.run(self.p, y=(g.vae_axis, [V3, V2D]))
        self.assertEqual(cells_of(g), [made(V3), made(V2D)])
        if forge_axes_write_settings():
            self.assertEqual(g.stderr.getvalue(), "")
        else:
            self.assertIn("the VAE the model is built with is a 2D Qwen2D VAE", g.stderr.getvalue())
        self.assertRestored()


class CheckpointVaeTests(GridCase):
    """No VAE file in the selection: the checkpoint has its own (3D) VAE."""
    UI = (TE,)

    def setUp(self):
        super().setUp()
        g = self.grid
        g.opts.data["sd_model_checkpoint"] = g.ckpt[CKPT_AIO].title
        with mock.patch.dict(sys.modules, g.sys_modules), contextlib.redirect_stdout(io.StringIO()):
            g.main_entry.refresh_model_loading_parameters()
        g.ui_loading = g.describe(g.params)
        g.initial_settings = copy.deepcopy(g.opts.data)

    def test_a_2d_file_replaces_the_checkpoints_vae(self):
        g = self.grid
        g.run(self.p, y=(g.vae_axis, [V2HDR, V3HDR, "None"]))
        self.assertEqual(cells_of(g), [made(V2HDR), {"family": "3d", "vae": VALUE[V3HDR], "module_n": stems(TE, V3HDR)},
                                       {"family": "3d", "vae": AIO_VALUE, "module_n": stems(TE)}])
        if forge_axes_write_settings():     # each file loads the model with it; "None": the checkpoint's own VAE
            self.assertEqual([cell["model"] for cell in g.cells],
                             [(CKPT_AIO, tuple(sorted((V2HDR, TE)))), (CKPT_AIO, tuple(sorted((V3HDR, TE)))),
                              (CKPT_AIO, (TE,))])
            self.assertEqual(g.swaps, [])
        else:
            self.assertEqual([cell["model"] for cell in g.cells],
                             [(CKPT_AIO, tuple(sorted((V2HDR, TE)))), (CKPT_AIO, (TE,)), (CKPT_AIO, (TE,))])
            self.assertEqual(g.swaps, [V3HDR, "None"])
        self.assertRestored()


# ---------------------------------------------------------------------------------------------------------------------
# 2. Forge's real VAE classes (subprocess)
# ---------------------------------------------------------------------------------------------------------------------

PROBE = r'''
import ast, json, os, sys, tempfile, types
forge, ext = sys.argv[1], sys.argv[2]
sys.dont_write_bytecode = True
sys.path[:0] = [forge, os.path.join(forge, "modules_forge", "packages"), ext]
sys.argv = [sys.argv[0], "--cpu"]
import torch
from safetensors.torch import load_file, save_file
from backend.nn.wan_vae import WanVAE
from backend.nn.wan_vae_2d import Qwen2DVAE
from sam3ext import xyz_vae
from sam3ext import xyz_text_encoder as xte

source = open(os.path.join(forge, "modules", "sd_vae.py"), encoding="utf-8").read()
namespace = {"torch": torch}
for node in ast.parse(source).body:
    target = getattr(getattr(node, "target", None), "id", None)
    if (isinstance(node, ast.AnnAssign) and target == "vae_ignore_keys") or (
            isinstance(node, ast.FunctionDef) and node.name == "_load_vae_dict"):
        exec(compile(ast.Module([node], []), "<forge:modules/sd_vae.py>", "exec"), namespace)
load_vae_dict = namespace["_load_vae_dict"]

def forge_loads(model, sd):
    try:
        load_vae_dict(types.SimpleNamespace(first_stage_model=model), sd)
        return True
    except RuntimeError:
        return False

out = {"tiny": [], "pairs": [], "real": [], "real_files": []}
tmp = tempfile.mkdtemp()
models = {
    "3d": WanVAE(base_dim=8, z_dim=4, dim_mult=[1, 2], num_res_blocks=1, attn_scales=[], temporal_downsample=[False]),
    "3d_z8": WanVAE(base_dim=8, z_dim=8, dim_mult=[1, 2], num_res_blocks=1, attn_scales=[], temporal_downsample=[False]),
    "2d": Qwen2DVAE(base_dim=8, z_dim=4, dim_mult=(1, 2), num_res_blocks=1),
    "2d_z8": Qwen2DVAE(base_dim=8, z_dim=8, dim_mult=(1, 2), num_res_blocks=1),
}
files = {}
for name, model in models.items():
    path = os.path.join(tmp, name + ".safetensors")
    save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}, path)
    files[name] = path
    sd = model.state_dict()
    out["tiny"].append({"name": name, "class": type(model).__name__, "family": xyz_vae.vae_family(xte.read_header(path)),
                        "forge_picks_2d": "post_quant_conv.weight" in sd, "keys": len(sd)})
for model_name, model in models.items():
    for file_name, path in files.items():
        out["pairs"].append({"model": model_name, "file": file_name,
                             "swappable": xyz_vae.swappable(xyz_vae.module_shapes(model.state_dict()),
                                                            xyz_vae.file_shapes(path)),
                             "forge_loads": forge_loads(model, load_file(path))})

DTYPES = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
folder = os.path.join(forge, "models", "VAE")
config = os.path.join(forge, "backend", "huggingface", "circlestone-labs", "Anima", "vae")
names = sorted(n for n in os.listdir(folder) if n.endswith(".safetensors")) if os.path.isdir(folder) else []
headers = {n: xte.read_header(os.path.join(folder, n)) for n in names}
names = [n for n in names if xyz_vae.vae_family(headers[n]) in ("3d", "2d")]
if names and os.path.isdir(config):
    with torch.device("meta"):
        bases = {"3d": WanVAE.from_config(WanVAE.load_config(config)), "2d": Qwen2DVAE.from_config({})}
    for name in names:
        out["real_files"].append({"name": name, "family": xyz_vae.vae_family(headers[name]),
                                  "forge_picks_2d": "post_quant_conv.weight" in headers[name]})
        meta = {k: torch.empty(v["shape"], dtype=DTYPES.get(v.get("dtype"), torch.float32), device="meta")
                for k, v in headers[name].items() if k != "__metadata__"}
        for base_name, base in bases.items():
            out["real"].append({"model": base_name, "file": name,
                                "swappable": xyz_vae.swappable(xyz_vae.module_shapes(base.state_dict()),
                                                               xyz_vae.file_shapes(os.path.join(folder, name))),
                                "forge_loads": forge_loads(base, dict(meta))})
print("PROBE-JSON " + json.dumps(out))
'''


class RealClassTests(unittest.TestCase):
    """Forge's ``WanVAE`` / ``Qwen2DVAE`` and its ``_load_vae_dict``, in a subprocess (Forge's backend is imported there
    only; nothing of it enters this test process)."""

    @classmethod
    def setUpClass(cls):
        for rel in ("backend/nn/wan_vae.py", "backend/nn/wan_vae_2d.py", "modules/sd_vae.py"):
            require_forge_file(rel)
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "probe.py"
            script.write_text(textwrap.dedent(PROBE), encoding="utf-8")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
            done = subprocess.run([sys.executable, "-B", str(script), str(forge_root()), str(ROOT)], env=env,
                                  capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        lines = [line for line in done.stdout.splitlines() if line.startswith("PROBE-JSON ")]
        if done.returncode != 0 or not lines:
            raise AssertionError(f"probe failed ({done.returncode}):\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}")
        cls.report = json.loads(lines[-1][len("PROBE-JSON "):])

    def test_family_is_the_class_forges_loader_picks(self):
        tiny = self.report["tiny"]
        self.assertEqual({row["name"]: row["family"] for row in tiny},
                         {"3d": "3d", "3d_z8": "3d", "2d": "2d", "2d_z8": "2d"})
        for row in tiny:
            with self.subTest(model=row["name"]):
                self.assertEqual(row["forge_picks_2d"], row["family"] == "2d")
                self.assertEqual(row["class"], "Qwen2DVAE" if row["family"] == "2d" else "WanVAE")

    def test_swappable_is_forges_strict_load(self):
        pairs = self.report["pairs"]
        self.assertEqual(len(pairs), 16)
        for row in pairs:
            with self.subTest(model=row["model"], file=row["file"]):
                self.assertEqual(row["swappable"], row["forge_loads"])
                self.assertEqual(row["swappable"], row["model"] == row["file"])

    def test_the_checkouts_vae_files(self):
        """The safetensors VAEs in the checkout's models/VAE (headers only, full-size Anima 3D / Qwen2D models on the
        meta device): the swap the axis predicts is the one Forge's strict load does."""
        if not self.report["real"]:
            self.skipTest("no Wan / Qwen2D safetensors VAE in the checkout's models/VAE")
        for row in self.report["real_files"]:
            with self.subTest(file=row["name"]):
                self.assertEqual(row["forge_picks_2d"], row["family"] == "2d")
                if row["name"] in FAMILY:
                    self.assertEqual(row["family"], FAMILY[row["name"]])
        for row in self.report["real"]:
            with self.subTest(model=row["model"], file=row["file"]):
                self.assertEqual(row["swappable"], row["forge_loads"])


# ---------------------------------------------------------------------------------------------------------------------
# 3. Source guards
# ---------------------------------------------------------------------------------------------------------------------

def _function(source, name):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise LookupError(name)


def _in_order(test, text, *parts):
    positions = [text.index(part) for part in parts]
    test.assertEqual(positions, sorted(positions), parts)


class SourceGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for rel in FORGE_FILES:
            require_forge_file(rel)

    def test_the_in_place_swap_is_a_strict_load(self):
        source = forge_source("modules/sd_vae.py")
        load = _function(source, "_load_vae_dict")
        self.assertIn('sd = {k: v for k, v in vae_sd.items() if k[0:4] != "loss" and k not in vae_ignore_keys}', load)
        self.assertIn("model.first_stage_model.load_state_dict(sd)", load)
        self.assertNotIn("strict", load)
        ignored = next(node for node in ast.parse(source).body if isinstance(node, ast.AnnAssign)
                       and getattr(node.target, "id", None) == "vae_ignore_keys")
        self.assertEqual(ast.literal_eval(ignored.value), set(xvae.IGNORED_KEYS))
        reload = _function(source, "reload_vae_weights")
        _in_order(self, reload, 'if vae in (None, "None"):', "store_base_vae(shared.sd_model)",
                  "vae_sd = utils.load_torch_file(vae)", "_load_vae_dict(shared.sd_model, vae_sd)")

    def test_process_images_swaps_after_the_model_load(self):
        segment = _function(forge_source("modules/processing.py"), "process_images")
        # 2.30 skips the swap for the selection's own VAE (``override = None``); 2.31 swaps the override as it is
        override = "override" if "if sd_vae.reload_vae_weights(override):" in segment else "_vae_override"
        _in_order(self, segment, "p.scripts.before_process(p)", '_vae_override = p.override_settings.pop("sd_vae", None)',
                  "manage_model_and_prompt_cache(p)", "if _vae_override is not None:",
                  f"if sd_vae.reload_vae_weights({override}):", f"_overridden_modules.append({override})",
                  "res = process_images_inner(p)", "finally:", "sd_vae.restore_vae_weights()")

    def test_the_model_is_reloaded_only_when_the_loading_parameters_change(self):
        segment = _function(forge_source("modules/sd_models.py"), "forge_model_reload")
        _in_order(self, segment, "current_hash = str(model_data.forge_loading_parameters)",
                  "if model_data.forge_hash == current_hash:", "return model_data.sd_model, False",
                  'additional_state_dicts = model_data.forge_loading_parameters.get("additional_modules", [])',
                  "sd_model = forge_loader(state_dict, additional_state_dicts=additional_state_dicts)",
                  "model_data.forge_hash = current_hash")

    def test_the_loader_builds_the_vae_class_from_the_file(self):
        source = forge_source("backend/loader.py")
        _in_order(self, source, 'if cls_name in ["AutoencoderKLWan", "AutoencoderKLQwenImage"]:',
                  'if "post_quant_conv.weight" in state_dict:  # 2D',
                  "from backend.nn.wan_vae_2d import Qwen2DVAE as WanVAE", "from backend.nn.wan_vae import WanVAE")
        replace = _function(source, "replace_state_dict")
        self.assertIn('if "decoder.conv_in.weight" in asd or "decoder.middle.0.residual.0.gamma" in asd:', replace)
        for key in xvae.CONVERTED_KEYS:
            self.assertIn(f'if "{key}" in asd:', replace)
        self.assertIn(f'"{xvae.WAN_KEY}"', replace)
        loader = _function(source, "forge_loader")
        self.assertIn("state_dicts, estimated_config = split_state_dict(sd, additional_state_dicts=additional_state_dicts)",
                      loader)
        split = _function(source, "split_state_dict")
        self.assertIn("for asd in additional_state_dicts:", split)
        self.assertIn("sd = replace_state_dict(sd, _asd, guess, asd)", split)

    def test_checkpoint_vae_prefixes(self):
        source = forge_source("modules_forge/packages/huggingface_guess/model_list.py")
        prefixes = set()
        for node in ast.walk(ast.parse(source)):
            if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "vae_key_prefix"
                                                     for t in node.targets)):
                prefixes.update(ast.literal_eval(node.value))
        self.assertTrue(prefixes)
        self.assertEqual(prefixes - set(xvae.CHECKPOINT_VAE_PREFIXES), set())

    def test_forges_vae_axis(self):
        source = forge_source("scripts/xyz_grid.py")
        self.assertEqual(_forge_axis_option(source, "VAE"),
                         'AxisOption("VAE", str, apply_vae, cost=0.7, choices=lambda: ["None", '
                         '*sorted(sd_vae.vae_dict.keys())])')
        apply_vae = _function(source, "apply_vae")
        if forge_axes_write_settings():
            # 2.31: the cell's module list (the dropdown's without the axis's values, plus find_vae's file - "None" for
            # "None"), which the replaced axis writes again; its old swap is the "state_dict" axis, not replaced
            self.assertIn("modules: list[str] = [m for m in opts.forge_additional_modules if not to_replace(m)]",
                          apply_vae)
            self.assertIn('p.override_settings["forge_additional_modules"] = modules + [find_vae(x)]', apply_vae)
            self.assertNotIn("sd_vae", apply_vae)
            self.assertIn('p.override_settings["sd_vae"] = find_vae(x)', _function(source, "apply_vae_fast"))
            self.assertIn("AxisOption('VAE (\"state_dict\" only)', str, apply_vae_fast, cost=0.3", source)
        else:
            self.assertIn('p.override_settings["sd_vae"] = find_vae(x)', apply_vae)
        self.assertIn('return "None"', _function(source, "find_vae"))
        self.assertIn("return sd_vae.vae_dict[name]", _function(source, "find_vae"))
        labels = re.findall(r'AxisOption(?:Txt2Img|Img2Img)?\("VAE"', source)
        self.assertEqual(len(labels), 1, "one VAE axis to replace")


if __name__ == "__main__":
    unittest.main()
