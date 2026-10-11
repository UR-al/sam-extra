"""XYZ ``[Modules] Text Encoder`` against Forge's own code (Forge Neo 2.30; skipped without a Forge checkout).

On Forge 2.29.2 (no Qwen3-VL 8B encoder yet — ``_forge_reads_qwen3vl_8b``) Forge rejects an 8B file that te_slots,
written for 2.30, puts in the ``qwen3vl_8b`` slot; the tests expect that rejection there instead of the slot.

1. **Slots** — Forge's real ``backend/loader.py`` ``replace_state_dict`` runs on the same headers as
   ``sam3ext.xyz_text_encoder.te_slots`` (tensors are stand-ins with a ``shape``): the ``text_encoders.<slot>.`` keys it
   writes are the slots te_slots gives, for every branch, the files Forge rejects, and the safetensors files in the
   checkout's own ``models/text_encoder`` (headers only). GGUF files go through Forge's ``gguf_remapping`` first; the
   stdlib GGUF reader gives the names and shapes Forge's vendored ``GGUFReader`` + ``get_orig_shape`` give. Checkpoints:
   Forge's ``preprocess_state_dict`` + ``replace_state_dict`` say which encoder slots a checkpoint has built in (a pick
   overwrites them) and which CLIP slots its family takes at all.
2. **A grid through Forge's code** — Forge's ``processing.process_images`` / ``manage_model_and_prompt_cache``,
   ``sysinfo.set_config``, ``main_entry.modules_change`` / ``checkpoint_change`` / ``refresh_model_loading_parameters``,
   ``sd_models.get_closet_checkpoint_match`` / ``select_checkpoint``, xyz_grid's ``apply_checkpoint`` / ``apply_vae`` /
   ``find_vae`` / ``refresh_loading_params_for_xyz_grid``, the Hires fix's model switch (the statements of
   ``StableDiffusionProcessingTxt2Img.sample``) and the Refiner's (``sd_samplers_common.apply_refiner``) run against
   stand-ins for the model loader, the sampler and the VAE swap, driven like xyz_grid's ``cell`` (shallow copy, X/Y/Z
   applied outside the try, ``process_images``; the hook's ``process`` / ``postprocess_image`` / ``postprocess`` where
   ``process_images_inner`` calls them). Checked per cell: the checkpoint and modules the model is loaded with, the
   ``Module N`` list of the infotext, and what a page load (``on_preset_change`` saves the settings) would write to
   config.json at that moment; after the grid: Forge's settings and loading parameters are the user's, config.json was
   not saved, the parent's ``override_settings`` is untouched — with a Checkpoint axis in both axis orders, a VAE axis,
   the axis on two axes, "Use Forge current" between values, a failing cell, an API caller's own module list and
   checkpoint, the dropdown changed mid-cell, Hires / Refiner model switches and passes inside a cell.
3. **Source guards** — the Forge code the axis relies on or mirrors is still shaped the same way (replace_state_dict's
   probe keys, widths and slot names, its model-type tables, gguf_remapping's renames, the hooks' order).
"""
from __future__ import annotations

import ast
import contextlib
import copy
import importlib.util
import io
import itertools
import logging
import os
import re
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import gguf_header  # noqa: E402
from sam3ext import xyz_text_encoder as xte  # noqa: E402
from sam3ext import xyz_text_encoder_cell as cell_runtime  # noqa: E402
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

    def dequantize_as_pytorch_parameter(self):   # gguf_remapping's embedding tables
        return self


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


def _forge_loader_namespace():
    namespace = {"os": os, "torch": types.SimpleNamespace(Tensor=object)}
    forge_function("backend/state_dict.py", "state_dict_prefix_replace", namespace)
    forge_function("backend/loader.py", "replace_state_dict", namespace)
    forge_function("backend/loader.py", "preprocess_state_dict", namespace)
    return namespace


def _gguf_modules():
    """``backend.loader_gguf`` with Forge's real ``gguf_remapping`` (its torch-only branches are not reached here)."""
    module = types.ModuleType("backend.loader_gguf")
    module.__dict__.update(torch=types.SimpleNamespace(), dequantize=None)
    forge_function("backend/loader_gguf.py", "gguf_remapping", module.__dict__)
    backend = types.ModuleType("backend")
    backend.__path__ = []
    backend.loader_gguf = module
    return {"backend": backend, "backend.loader_gguf": module}


def _forge_reads_qwen3vl_8b() -> bool:
    """Forge Neo 2.30 added the Qwen3-VL 8B text encoder (Qwen-Image 2.1, slot ``qwen3vl_8b``); 2.29.2's
    replace_state_dict asserts that every Qwen3-VL file is the 4B one. te_slots follows 2.30 and names an 8B file's
    slot. On 2.29.2 no model family has that slot (test_slot_names checks it), so the axis refuses such a file as an
    encoder the model does not have, as it does on 2.30 for a model without that slot."""
    return "qwen3vl_8b" in forge_source("backend/loader.py")


class SlotOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_forge_file("backend/loader.py")
        cls.namespace = _forge_loader_namespace()
        cls.replace = staticmethod(cls.namespace["replace_state_dict"])

    def forge_slots(self, shapes, checkpoint=None, path="file.safetensors"):
        checkpoint = checkpoint or {}
        sd = {key: FakeTensor(shape) for key, shape in checkpoint.items()}
        asd = {key: FakeTensor(shape) for key, shape in shapes.items()}
        with mock.patch.dict(sys.modules, _gguf_modules()):
            out = self.replace(dict(sd), asd, GUESS, path)
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
                if set(expected) == {"qwen3vl_8b"} and not _forge_reads_qwen3vl_8b():
                    with self.assertRaises(AssertionError):     # Forge 2.29.2: only the 4B one
                        self.forge_slots(shapes)
                    self.assertEqual(set(xte.te_slots(header)), set(expected))
                    continue
                forge = self.forge_slots(shapes, CLIP_CONTEXT.get(name))
                self.assertEqual(forge, set(xte.te_slots(header)))
                self.assertEqual(forge, set(expected))

    def test_clip_goes_nowhere_without_a_clip_family(self):
        # where a CLIP file goes depends on the checkpoint: nowhere for Anima / Krea2 (te_slots names what it IS)
        self.assertEqual(self.forge_slots(fx.CASES["clip_l_hf"][0]), set())
        self.assertEqual(self.forge_slots(fx.CASES["clip_l_hf"][0], SDXL), {"clip_l"})

    def test_gguf_files_through_forges_remapping(self):
        """GGUF files on disk: the names and shapes our reader gives, through Forge's gguf_remapping and
        replace_state_dict, land in the slots classify_file says."""
        with tempfile.TemporaryDirectory() as folder:
            xte.clear_cache()
            for name, (shapes, expected) in fx.GGUF_CASES.items():
                with self.subTest(case=name):
                    path = fx.write_gguf(Path(folder) / f"{name}.gguf", shapes)
                    read = gguf_header.read_gguf_shapes(path)
                    self.assertEqual(self.forge_slots(read, path=str(path)), set(expected))
                    kind = xte.classify_file(path)
                    self.assertEqual(set(kind.slots), set(expected))
            xte.clear_cache()

    def test_the_checkouts_text_encoder_files(self):
        """The safetensors and GGUF files in the checkout's models/text_encoder: Forge's slots from their real headers."""
        folder = forge_root() / "models" / "text_encoder"
        files = []
        if folder.is_dir():
            for pattern in ("*.safetensors", "*.sft", "*.gguf"):
                files += sorted(folder.rglob(pattern))
        if not files:
            self.skipTest(f"no safetensors / GGUF file in {folder}")
        for path in files:
            with self.subTest(file=path.name):
                header = xte.read_header(str(path))
                self.assertIsInstance(header, dict)
                kind = xte.classify_header(path.name, header)
                if kind.kind == "vae":
                    continue
                shapes = {key: value["shape"] for key, value in header.items() if key != "__metadata__"}
                if path.suffix == ".gguf":
                    shapes = gguf_header.read_gguf_shapes(path)
                if kind.kind == "invalid":
                    with self.assertRaises((AssertionError, KeyError)):
                        self.forge_slots(shapes, path=str(path))
                    continue
                # CLIP weights go nowhere without a CLIP checkpoint (model type "-"): compared by name above
                expected = set(xte.te_slots(header)) - {"clip_l", "clip_g"}
                if expected == {"qwen3vl_8b"} and not _forge_reads_qwen3vl_8b():
                    with self.assertRaises(AssertionError):     # Forge 2.29.2: only the 4B one
                        self.forge_slots(shapes, path=str(path))
                    continue
                self.assertEqual(self.forge_slots(shapes, path=str(path)), expected)
                if kind.kind in ("qwen35", "adapter"):
                    self.assertEqual(expected, set(), "Forge would use it as an encoder")


class GgufReaderOriginTests(unittest.TestCase):
    """The stdlib reader against Forge's vendored gguf-py ``GGUFReader`` and ``backend/loader_gguf.get_orig_shape``,
    the way ``backend/utils.py`` ``load_torch_file`` gives each tensor its shape."""

    @classmethod
    def setUpClass(cls):
        package = require_forge_file("modules_forge/packages/gguf/gguf_reader.py").parent
        try:
            import numpy  # noqa: F401
        except ImportError:   # pragma: no cover - the Forge venv has numpy
            raise unittest.SkipTest("numpy is not installed")
        gguf = types.ModuleType("gguf")
        gguf.__path__ = [str(package)]
        stubs = {"gguf": gguf}
        with mock.patch.dict(sys.modules, stubs):
            constants = cls._load("gguf.constants", package / "constants.py")
            quants = types.ModuleType("gguf.quants")
            quants.__dict__.update(GGML_QUANT_SIZES=constants.GGML_QUANT_SIZES,
                                   GGMLQuantizationType=constants.GGMLQuantizationType)
            forge_function("modules_forge/packages/gguf/quants.py", "quant_shape_to_byte_shape", quants.__dict__)
            sys.modules["gguf.quants"] = quants
            reader = cls._load("gguf.gguf_reader", package / "gguf_reader.py")
            stubs = {name: module for name, module in sys.modules.items() if name == "gguf" or name.startswith("gguf.")}
        gguf.GGUFValueType = constants.GGUFValueType
        cls.stubs = stubs
        cls.GGUFReader = reader.GGUFReader
        namespace = {"gguf": gguf, "torch": types.SimpleNamespace(Size=tuple)}
        cls.get_orig_shape = staticmethod(forge_function("backend/loader_gguf.py", "get_orig_shape", namespace))

    @staticmethod
    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def forge_shapes(self, path):
        with mock.patch.dict(sys.modules, self.stubs):
            reader = self.GGUFReader(str(path))
            out = {}
            for tensor in reader.tensors:
                name = str(tensor.name)
                shape = self.get_orig_shape(reader, name)
                if shape is None:
                    shape = tuple(int(v) for v in reversed(tensor.shape))
                out[name] = tuple(shape)
            return out

    def test_same_names_and_shapes(self):
        kv = [("general.alignment", fx.GGUF_UINT32, 32), ("general.name", fx.GGUF_STRING, "t5"),
              ("tokenizer.tokens", fx.GGUF_ARRAY, (fx.GGUF_STRING, ["<pad>", "▁the", "</s>"] * 20)),
              ("comfy.gguf.orig_shape.c", fx.GGUF_ARRAY, (fx.GGUF_UINT32, [9])),
              ("comfy.gguf.orig_shape.d", fx.GGUF_ARRAY, (fx.GGUF_INT32, []))]
        cases = {
            "plain": dict(shapes={"enc.blk.0.attn_k.weight": [4, 8], "token_embd.weight": [3, 2], "n": [5]}),
            "orig shapes": dict(shapes={"a": [4, 2], "b": [6], "c": [2], "d": [3]},
                                orig_shapes={"a": [2, 2, 2], "b": [32128, 1]}, kv=kv),
            "version 2": dict(shapes={"a": [2, 3]}, version=2),
        }
        with tempfile.TemporaryDirectory() as folder:
            for name, kwargs in cases.items():
                with self.subTest(case=name):
                    path = fx.write_gguf(Path(folder) / "f.gguf", with_data=True, **kwargs)
                    self.assertEqual(gguf_header.read_gguf_shapes(path), self.forge_shapes(path))

    def test_what_forge_cannot_open(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, kwargs in {"big-endian (NumPy 2)": dict(order=">"), "version 1": dict(version=1),
                                 "bad magic": dict(magic=b"GGUG")}.items():
                with self.subTest(case=name):
                    path = fx.write_gguf(Path(folder) / "f.gguf", {"a": [2, 3]}, with_data=True, **kwargs)
                    with self.assertRaises(Exception):
                        self.forge_shapes(path)
                    self.assertIsNone(gguf_header.read_gguf_shapes(path))
            path = fx.write_gguf(Path(folder) / "f.gguf", {"a": [2], "b": [3]}, raw_names={"b": b"a"}, with_data=True)
            with self.assertRaises(Exception):
                self.forge_shapes(path)
            self.assertIsNone(gguf_header.read_gguf_shapes(path))


class CheckpointOriginTests(unittest.TestCase):
    """``checkpoint_profile`` against Forge: ``preprocess_state_dict``, then ``replace_state_dict`` with a CLIP-L,
    CLIP-G, T5 and Qwen3 4B file - a slot is built in when the pick overwrites or deletes checkpoint keys, and Forge
    loads CLIP-L / CLIP-G for the checkpoint when the pick is written anywhere."""

    PICKS = {"clip_l": fx.CASES["clip_l_hf"][0], "clip_g": fx.CASES["clip_g_resblocks"][0],
             "t5xxl": fx.CASES["t5xxl"][0], "qwen3_4b": fx.CASES["qwen3_4b"][0]}

    @classmethod
    def setUpClass(cls):
        require_forge_file("backend/loader.py")
        namespace = _forge_loader_namespace()
        cls.replace = staticmethod(namespace["replace_state_dict"])
        cls.preprocess = staticmethod(namespace["preprocess_state_dict"])

    def test_every_checkpoint(self):
        for case, (shapes, model_type, slots) in fx.CHECKPOINTS.items():
            profile = xte.checkpoint_profile(case, fx.header(shapes))
            for slot, pick in self.PICKS.items():
                with self.subTest(checkpoint=case, pick=slot):
                    sd = self.preprocess({key: FakeTensor(shape) for key, shape in shapes.items()})
                    asd = {key: FakeTensor(shape) for key, shape in pick.items()}
                    out = self.replace(dict(sd), asd, GUESS, "pick.safetensors")
                    changed = {key for key in sd if key not in out or out[key] is not sd[key]}
                    written = {key for key in out if key not in sd or out[key] is not sd[key]}
                    self.assertEqual(bool(changed), slot in profile.slots)
                    self.assertEqual(bool(written), profile.loads(slot))
            self.assertEqual(profile.model_type, model_type)


# ---------------------------------------------------------------------------------------------------------------------
# 2. A grid through Forge's settings / reload / restore code
# ---------------------------------------------------------------------------------------------------------------------

VAE = "qwen_image_vae.safetensors"
VAE_HDR = "qwen_image_vae_HDR_fp32.safetensors"
TE = "qwen_3_06b_base.safetensors"
TE_FP32 = "qwen_3_06b_base_full_fp32.safetensors"
TE_FP16 = "qwen_3_06b_base_fp16_uncensored.safetensors"
AE = "ae.safetensors"
CLIP_L = "clip_l.safetensors"
CLIP_L_OTHER = "clip_l_other.safetensors"
T5_16 = "t5xxl_fp16.safetensors"
T5_8 = "t5xxl_fp8.safetensors"
T5_GGUF = "t5-v1_1-xxl-encoder-Q8_0.gguf"
CKPT_A = "anima.safetensors"
CKPT_B = "anima_other.safetensors"
NOTHING = AxisOption("Nothing", str, lambda p, x, xs: None)


class P:
    """A processing object as far as process_images and the Hires switch use it here; xyz_grid copies it per cell."""

    def __init__(self, scripts, override_settings=None):
        self.scripts = scripts
        self.override_settings = {} if override_settings is None else override_settings
        self.override_settings_restore_afterwards = True
        self.styles = []
        self.seed = 1
        self.prompt_cache_clears = 0
        self.hr_additional_modules = ["Use same choices"]
        self.hr_checkpoint_name = "Use same checkpoint"
        self.nested = False

    def clear_prompt_cache(self):
        self.prompt_cache_clears += 1


def _statements(rel: str, function: str, first: str, last, name: str, args: str, namespace: dict):
    """Forge's statements of ``function`` (``Class.method`` or a function) from the assignment to ``first`` through the
    next statement for which ``last(statement)`` is true, as a function ``name(args)`` executed in ``namespace``."""
    source = forge_source(rel)
    tree = ast.parse(source)
    node = tree
    for part in function.split("."):
        node = next(n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part)
    body = node.body
    start = next(i for i, s in enumerate(body) if isinstance(s, (ast.Assign, ast.AnnAssign)) and any(
        isinstance(t, ast.Name) and t.id == first for t in (s.targets if isinstance(s, ast.Assign) else [s.target])))
    end = next(i for i, s in enumerate(body) if i > start and last(s))
    lines = source.splitlines()[body[start].lineno - 1:body[end].end_lineno]
    code = f"def {name}({args}):\n" + textwrap.indent(textwrap.dedent("\n".join(lines)), "    ") + "\n"
    exec(compile(code, f"<forge:{rel}:{function}>", "exec"), namespace)
    return namespace[name]


class Grid:
    """Forge's code from the checkout, wired together, with stand-ins for the loader, the sampler and the VAE swap."""

    def __init__(self, models_path, ui=(VAE, TE)):
        self.module_list, self.module_map = fx.write_models(models_path)
        self.models_path = str(models_path)
        self.loads, self.cells, self.nested, self.unloads, self.vae_swaps, self.reported = [], [], [], 0, [], []
        self.fail_cells = set()
        self.events = {}            # cell index -> [callable(grid, p)] run inside the cell
        self.settings_seen = []     # what a page load would save during each cell (opts.data)
        self.swapped_vae = None
        infos = [fx.CheckpointInfo(CKPT_A, "aaaa"), fx.CheckpointInfo(CKPT_B, "bbbb")]
        self.ckpt = {info.name: info for info in infos}
        self.ui_modules = self.paths(*ui)
        self.opts = fx.FakeOpts(
            {"sd_model_checkpoint": self.ckpt[CKPT_A].title, "forge_additional_modules": list(self.ui_modules),
             "forge_unet_storage_dtype": "Automatic", "CLIP_stop_at_last_layers": 1},
            {"forge_additional_modules": [], "sd_modules_dropdown_use_short": True})

        shared = self.shared = types.ModuleType("modules.shared")
        shared.opts = self.opts
        shared.cmd_opts = types.SimpleNamespace(text_encoder_dirs=[])
        shared.sd_model = None
        shared.config_filename = "config.json"
        shared.state = types.SimpleNamespace(time_start=1.0, job="1 out of 4")

        sd_models = self.sd_models = types.ModuleType("modules.sd_models")
        aliases = {alias: info for info in infos for alias in info.ids}
        sd_models.__dict__.update(checkpoint_aliases=aliases, checkpoints_list={i.title: i for i in infos}, re=re,
                                  sys=sys, shared=shared, re_strip_checksum=re.compile(r"\s*\[[^]]+]\s*$"))
        forge_function("modules/sd_models.py", "get_closet_checkpoint_match", sd_models.__dict__)
        forge_function("modules/sd_models.py", "select_checkpoint", sd_models.__dict__)
        sd_models.model_data = types.SimpleNamespace(forge_loading_parameters={}, forge_hash="", sd_model=None)
        sd_models.forge_model_reload = self._forge_model_reload

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
        # the Hires fix's model switch and the Refiner's, as Forge has them
        self.hires_reload = _statements(
            "modules/processing.py", "StableDiffusionProcessingTxt2Img.sample", "fp_checkpoint",
            lambda s: isinstance(s, ast.If) and isinstance(s.test, ast.Name) and s.test.id == "reload",
            "hires_reload", "self", {"shared": shared, "main_entry": main_entry, "sd_models": sd_models})
        self.dynamic_args = types.SimpleNamespace(loading_refiner=False)
        self.refiner_switch = _statements(
            "modules/sd_samplers_common.py", "apply_refiner", "original_checkpoint", lambda s: isinstance(s, ast.Try),
            "refiner_switch",
            "refiner_checkpoint_info, cfg_denoiser",
            {"shared": shared, "main_entry": main_entry, "sd_models": sd_models, "dynamic_args": self.dynamic_args})

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
        self.runner = types.SimpleNamespace(before_process=self._before_process, process=self._process,
                                            postprocess_image=self._postprocess_image, postprocess=self._postprocess)

        with mock.patch.dict(sys.modules, self.sys_modules):
            main_entry.refresh_model_loading_parameters()     # Forge's state after startup
        self.ui_loading = self.describe(sd_models.model_data.forge_loading_parameters)
        self.initial_settings = copy.deepcopy(self.opts.data)

    # -- stand-ins below Forge's code --
    def paths(self, *names):
        return sorted(self.module_list[name] for name in names)

    @staticmethod
    def describe(params):
        return params["checkpoint_info"].name, tuple(sorted(os.path.basename(m) for m in params["additional_modules"]))

    @property
    def params(self):
        return self.sd_models.model_data.forge_loading_parameters

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
        """process_images_inner as far as the hook sees it: ``process`` first, the sampling (the cell's events), each
        image's ``postprocess_image`` and infotext (``create_infotext``'s Module N list), ``postprocess`` last."""
        record = self.nested if p.nested else self.cells
        index = len(record)
        self.runner.process(p)
        if not p.nested:
            self.settings_seen.append(copy.deepcopy(self.opts.data))
            for event in self.events.get(index, ()):
                event(self, p)
        self.runner.postprocess_image(p, None)
        modules = self.proc.__dict__["_overridden_modules"] or self.opts.forge_additional_modules
        record.append({
            "model": self.sd_models.model_data.sd_model.loaded,
            "loading": self.describe(self.sd_models.model_data.forge_loading_parameters),
            "module_n": sorted(os.path.splitext(os.path.basename(m))[0] for m in modules),
            "vae": self.swapped_vae,
        })
        if not p.nested and index in self.fail_cells:
            raise RuntimeError(f"cell {index} failed")
        result = types.SimpleNamespace(images=[index])
        self.runner.postprocess(p, result)
        return result

    def _call(self, method, *args):
        # modules/scripts.py ScriptRunner: every always-on script, errors reported
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

    def generate(self, p):
        """A generation outside a grid (txt2img without a script)."""
        with mock.patch.dict(sys.modules, self.sys_modules), contextlib.redirect_stdout(io.StringIO()):
            self.proc.process_images(p)


def expected(ckpt, *modules):
    return ckpt, tuple(sorted(modules))


def stems(*names):
    return sorted(os.path.splitext(name)[0] for name in names)


class GridCase(unittest.TestCase):
    UI = (VAE, TE)

    @classmethod
    def setUpClass(cls):
        for rel in ("modules/processing.py", "modules/sysinfo.py", "modules_forge/main_entry.py", "modules/sd_models.py",
                    "modules/sd_samplers_common.py", "scripts/xyz_grid.py"):
            require_forge_file(rel)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        xte.clear_cache()
        cell_runtime._STATE = None
        self.grid = Grid(Path(self.tmp.name) / "models", self.UI)
        self.p = P(self.grid.runner)
        self.parent_settings = self.p.override_settings

    def tearDown(self):
        cell_runtime._STATE = None
        xte.clear_cache()
        self.tmp.cleanup()

    def assertRestored(self, parent_settings=None, settings_during=None):
        g = self.grid
        self.assertEqual(g.opts.data, g.initial_settings, "Forge's settings are the user's")
        for seen in g.settings_seen:
            self.assertEqual(seen, settings_during or g.initial_settings, "a page load during a cell saves these")
        self.assertEqual(g.describe(g.params), g.ui_loading)
        self.assertEqual(g.opts.saved, [], "config.json is never saved")
        self.assertIs(self.p.override_settings, self.parent_settings)
        self.assertEqual(self.p.override_settings, parent_settings or {})
        self.assertFalse(hasattr(self.p, xte.CELL_ATTR))
        self.assertEqual(g.reported, [])
        self.assertIsNone(g.proc.__dict__["_overridden_modules"])
        self.assertIsNone(cell_runtime.state())

    def models(self):
        return [cell["model"] for cell in self.grid.cells]


class GridTests(GridCase):
    def test_text_encoder_only(self):
        values = ["Use Forge current", "qwen_3_06b_base_full_fp32", "qwen_3_06b_base_fp16_uncensored.safetensors", TE,
                  "None"]
        self.grid.run(self.p, x=(self.grid.te_axis, values))
        self.assertEqual(self.models(), [expected(CKPT_A, VAE, TE), expected(CKPT_A, VAE, TE_FP32),
                                         expected(CKPT_A, VAE, TE_FP16), expected(CKPT_A, VAE, TE),
                                         expected(CKPT_A, VAE, TE)])
        self.assertEqual([cell["module_n"] for cell in self.grid.cells],
                         [stems(VAE, TE), stems(VAE, TE_FP32), stems(VAE, TE_FP16), stems(VAE, TE), stems(VAE, TE)])
        for cell in self.grid.cells:
            self.assertEqual(cell["loading"], cell["model"])
        self.assertRestored()

    def test_settings_are_never_written_during_a_cell(self):
        """What on_preset_change (every page load) would save to config.json while a cell renders is the user's
        selection - also with a Checkpoint axis - so the next start loads the user's model."""
        g = self.grid
        g.run(self.p, x=(g.ckpt_axis, [CKPT_B]), y=(g.te_axis, ["qwen_3_06b_base_full_fp32"]))
        self.assertEqual(self.models(), [expected(CKPT_B, VAE, TE_FP32)])
        snapshot = g.settings_seen[0]
        self.assertEqual(snapshot["forge_additional_modules"], g.ui_modules)
        self.assertEqual(snapshot["sd_model_checkpoint"], g.ckpt[CKPT_A].title)
        self.assertRestored()
        # a restart from what that page load wrote: Forge's startup refresh loads the user's model
        g.opts.data.clear()
        g.opts.data.update(snapshot)
        with mock.patch.dict(sys.modules, g.sys_modules), contextlib.redirect_stdout(io.StringIO()):
            g.main_entry.refresh_model_loading_parameters()
        self.assertEqual(g.describe(g.params), expected(CKPT_A, VAE, TE))

    def test_page_load_save_during_a_cell(self):
        g = self.grid
        g.events[0] = [lambda grid, p: grid.opts.save("config.json")]   # on_preset_change's save, mid-cell
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32"]))
        self.assertEqual(g.opts.written[0]["forge_additional_modules"], g.ui_modules)
        self.assertEqual(g.opts.written[0]["sd_model_checkpoint"], g.ckpt[CKPT_A].title)
        self.assertEqual(self.models(), [expected(CKPT_A, VAE, TE_FP32)])

    def test_dropdown_change_mid_cell_is_kept(self):
        """The user picks other modules while a cell renders (ui_vae.input -> modules_change, saved): the rest of the
        grid builds on the new selection and nothing puts the old one back."""
        g = self.grid
        g.events[0] = [lambda grid, p: grid.main_entry.modules_change([VAE_HDR, TE], "anima")]
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32", "qwen_3_06b_base_fp16_uncensored"]))
        new = g.paths(VAE_HDR, TE)
        self.assertEqual(self.models(), [expected(CKPT_A, VAE, TE_FP32), expected(CKPT_A, VAE_HDR, TE_FP16)])
        self.assertEqual(g.opts.data["forge_additional_modules"], new)
        self.assertEqual(g.opts.data["forge_additional_modules_anima"], new)
        self.assertEqual(g.opts.written[-1]["forge_additional_modules"], new)
        self.assertEqual(g.describe(g.params), expected(CKPT_A, VAE_HDR, TE))
        self.assertIsNone(cell_runtime.state())

    def test_preset_switched_mid_cell_is_kept(self):
        """A UI preset whose modules are the selection's but whose checkpoint is another (main_entry._load_presets):
        the rest of the cell renders the new checkpoint with the cell's encoder, and the next cell starts from it."""
        g = self.grid

        def load_preset(grid, p):
            grid.main_entry.modules_change([VAE, TE], "other", save=False, refresh=False)
            grid.main_entry.checkpoint_change(CKPT_B, "other", save=True, refresh=True)
            grid.batch_two = grid._forge_model_reload()[0].loaded

        g.events[0] = [load_preset]
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32", "Use Forge current"]))
        self.assertEqual(g.batch_two, expected(CKPT_B, VAE, TE_FP32))
        self.assertEqual(g.loads, [expected(CKPT_A, VAE, TE_FP32), expected(CKPT_B, VAE, TE_FP32),
                                   expected(CKPT_B, VAE, TE)])
        self.assertEqual(g.opts.data["sd_model_checkpoint"], CKPT_B)
        self.assertEqual(g.opts.written[-1]["forge_additional_modules"], g.ui_modules)
        self.assertEqual(g.describe(g.params), expected(CKPT_B, VAE, TE))
        self.assertIsNone(cell_runtime.state())

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
        # xyz's apply_vae writes "sd_vae" into the shared dict; Forge pops it in each cell, as without the axis
        self.assertRestored()

    def test_current_between_values(self):
        g = self.grid
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32", "Use Forge current",
                                     "qwen_3_06b_base_fp16_uncensored"]))
        self.assertEqual(self.models(), [expected(CKPT_A, VAE, TE_FP32), expected(CKPT_A, VAE, TE),
                                         expected(CKPT_A, VAE, TE_FP16)])
        self.assertRestored()

    def test_a_failing_cell_is_put_back(self):
        g = self.grid
        g.fail_cells = {0}
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32", "Use Forge current"]))
        self.assertEqual(self.models(), [expected(CKPT_A, VAE, TE_FP32), expected(CKPT_A, VAE, TE)])
        self.assertEqual([cell["module_n"] for cell in g.cells], [stems(VAE, TE_FP32), stems(VAE, TE)])
        self.assertRestored()

    def test_a_failing_last_cell_is_put_back_by_the_next_task(self):
        g = self.grid
        g.fail_cells = {1}
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32", "qwen_3_06b_base_fp16_uncensored"]))
        self.assertEqual(self.models(), [expected(CKPT_A, VAE, TE_FP32), expected(CKPT_A, VAE, TE_FP16)])
        # xyz_grid rebuilt the loading parameters from the settings after the grid; the task ends
        self.assertEqual(g.describe(g.params), expected(CKPT_A, VAE, TE))
        g.shared.state.job = ""
        # between tasks the user goes to other modules and back: Forge's rebuild is left alone
        with mock.patch.dict(sys.modules, g.sys_modules), contextlib.redirect_stdout(io.StringIO()):
            g.main_entry.modules_change([VAE_HDR, TE], None, save=False)
            g.main_entry.modules_change([VAE, TE], None, save=False)
        self.assertEqual(g.describe(g.params), expected(CKPT_A, VAE, TE))
        # the next task: its generation gets the dropdown's model and Module N list
        g.shared.state.time_start, g.shared.state.job = 2.0, "task(2)"
        g.generate(P(g.runner))
        self.assertEqual(g.cells[-1]["model"], expected(CKPT_A, VAE, TE))
        self.assertEqual(g.cells[-1]["module_n"], stems(VAE, TE))
        self.assertIsNone(g.proc.__dict__["_overridden_modules"])
        self.assertIsNone(cell_runtime.state())

    def test_put_back_also_when_the_caller_keeps_its_settings(self):
        # an API caller may send override_settings_restore_afterwards=False; the axis writes no setting anyway
        g = self.grid
        self.p.override_settings_restore_afterwards = False
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32", "Use Forge current"]))
        self.assertEqual(self.models(), [expected(CKPT_A, VAE, TE_FP32), expected(CKPT_A, VAE, TE)])
        self.assertFalse(self.p.override_settings_restore_afterwards)
        self.assertRestored()

    def test_a_fresh_config_gets_no_key(self):
        # no "forge_additional_modules" in config.json yet: it stays out of it
        g = self.grid
        del g.opts.data["forge_additional_modules"]
        with mock.patch.dict(sys.modules, g.sys_modules), contextlib.redirect_stdout(io.StringIO()):
            g.main_entry.refresh_model_loading_parameters()
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32"]))
        self.assertEqual(self.models(), [expected(CKPT_A, TE_FP32)])
        self.assertNotIn("forge_additional_modules", g.opts.data)
        self.assertNotIn("forge_additional_modules", g.settings_seen[0])
        self.assertEqual(g.describe(g.params), expected(CKPT_A))
        self.assertEqual(g.opts.saved, [])

    def test_same_encoder_twice_is_loaded_once(self):
        g = self.grid
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32", "qwen_3_06b_base_full_fp32"]))
        self.assertEqual(g.loads, [expected(CKPT_A, VAE, TE_FP32)])
        self.assertEqual(g.unloads, 0)
        self.assertRestored()

    def test_api_callers_module_list_and_checkpoint(self):
        g = self.grid
        api = {"forge_additional_modules": [VAE_HDR, TE], "sd_model_checkpoint": CKPT_B}
        self.p = P(g.runner, api)
        self.parent_settings = api
        g.run(self.p, x=(g.te_axis, ["Use Forge current", "qwen_3_06b_base_full_fp32"]))
        self.assertEqual(self.models(), [expected(CKPT_B, VAE_HDR, TE), expected(CKPT_B, VAE_HDR, TE_FP32)])
        self.assertEqual([cell["module_n"] for cell in g.cells], [stems(VAE_HDR, TE), stems(VAE_HDR, TE_FP32)])
        # the caller's own override is applied (and put back) by Forge, as without the axis
        during = dict(g.initial_settings, forge_additional_modules=g.paths(VAE_HDR, TE), sd_model_checkpoint=CKPT_B)
        self.assertRestored(parent_settings={"forge_additional_modules": [VAE_HDR, TE], "sd_model_checkpoint": CKPT_B},
                            settings_during=during)

    def test_a_bad_value_stops_before_the_first_cell(self):
        g = self.grid
        with self.assertRaisesRegex(ValueError, "is a Qwen3-VL 4B encoder"):
            g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32", "qwen3vl_4b_bf16"]))
        self.assertEqual(g.cells, [])
        self.assertRestored()


class GridPassesTests(GridCase):
    """Forge's own model switches inside a cell (Hires fix, Refiner) and passes inside a cell (ADetailer, SAM3)."""

    def hires(self, checkpoint=None, modules=None, reload_after=True):
        def event(grid, p):
            p.hr_checkpoint_name = checkpoint or "Use same checkpoint"
            p.hr_additional_modules = modules or ["Use same choices"]
            grid.hires_reload(p)
            if reload_after:
                grid.batch_two = grid._forge_model_reload()[0].loaded    # the next batch's first pass
        return event

    def test_hires_checkpoint_keeps_the_cells_encoder(self):
        g = self.grid
        g.events[0] = [self.hires(checkpoint=CKPT_B)]
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32"]))
        # first pass, the Hires pass on its own checkpoint with the same choices, the next batch
        self.assertEqual(g.loads, [expected(CKPT_A, VAE, TE_FP32), expected(CKPT_B, VAE, TE_FP32),
                                   expected(CKPT_A, VAE, TE_FP32)])
        self.assertEqual(g.batch_two, expected(CKPT_A, VAE, TE_FP32))
        self.assertEqual(self.grid.cells[0]["module_n"], stems(VAE, TE_FP32))
        self.assertRestored()

    def test_hires_modules_of_its_own(self):
        g = self.grid
        g.events[0] = [self.hires(modules=[VAE_HDR, TE])]
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32"]))
        # the Hires pass uses the modules picked for it; the next batch is the cell's again
        self.assertEqual(g.loads, [expected(CKPT_A, VAE, TE_FP32), expected(CKPT_A, VAE_HDR, TE),
                                   expected(CKPT_A, VAE, TE_FP32)])
        self.assertRestored()

    def test_hires_with_a_checkpoint_axis(self):
        g = self.grid
        g.events[0] = [self.hires(checkpoint=CKPT_B)]
        g.run(self.p, x=(g.ckpt_axis, [CKPT_B]), y=(g.te_axis, ["qwen_3_06b_base_full_fp32"]))
        # The Hires checkpoint is the cell's (no reload); Forge's restore after it rebuilds from the settings'
        # checkpoint (the dropdown's), and the cell keeps the Checkpoint axis's for the next batch
        self.assertEqual(g.loads, [expected(CKPT_B, VAE, TE_FP32)])
        self.assertEqual(g.batch_two, expected(CKPT_B, VAE, TE_FP32))
        self.assertRestored()

    def test_refiner_keeps_the_cells_encoder(self):
        g = self.grid
        refiner = types.SimpleNamespace(short_title=CKPT_B)

        def event(grid, p):
            grid.refiner_switch(refiner, types.SimpleNamespace(model_wrap=object()))
            grid.batch_two = grid._forge_model_reload()[0].loaded

        g.events[0] = [event]
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32"]))
        self.assertEqual(g.loads, [expected(CKPT_A, VAE, TE_FP32), expected(CKPT_B, VAE, TE_FP32),
                                   expected(CKPT_A, VAE, TE_FP32)])
        self.assertFalse(g.dynamic_args.loading_refiner)
        self.assertRestored()

    def test_passes_inside_a_cell_use_the_cells_model(self):
        g = self.grid

        def adetailer_like(grid, p):
            nested = P(grid.runner, dict(p.override_settings))      # a new img2img p (ADetailer, SAM3 refine)
            nested.nested = True
            grid.proc.process_images(nested)
            again = copy.copy(p)                                     # a copy of the cell's p
            again.nested = True
            grid.proc.process_images(again)

        g.events[0] = [adetailer_like]
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32", "Use Forge current"]))
        self.assertEqual(g.loads, [expected(CKPT_A, VAE, TE_FP32), expected(CKPT_A, VAE, TE)])
        self.assertEqual([cell["model"] for cell in g.nested], [expected(CKPT_A, VAE, TE_FP32)] * 2)
        self.assertEqual([cell["module_n"] for cell in g.nested], [stems(VAE, TE_FP32)] * 2)
        self.assertEqual(g.cells[0]["module_n"], stems(VAE, TE_FP32))
        self.assertRestored()

    def test_a_pass_with_its_own_vae_does_not_change_the_cells_module_list(self):
        g = self.grid

        def pass_with_vae(grid, p):
            nested = P(grid.runner, {"sd_vae": grid.vae_dict[VAE_HDR]})
            nested.nested = True
            grid.proc.process_images(nested)

        g.events[0] = [pass_with_vae]
        g.run(self.p, x=(g.te_axis, ["qwen_3_06b_base_full_fp32"]))
        self.assertEqual(g.nested[0]["module_n"], stems(VAE_HDR, TE_FP32))
        self.assertEqual(g.cells[0]["module_n"], stems(VAE, TE_FP32))
        self.assertRestored()


class FluxGridTests(GridCase):
    UI = (AE, CLIP_L, T5_16)

    def test_text_encoder_axis_on_x_and_y(self):
        g = self.grid
        g.run(self.p, x=(g.te_axis, ["clip_l", "clip_l_other"]), y=(g.te_axis, ["Use Forge current", "t5xxl_fp8"]))
        # equal costs: y outer, x inner
        self.assertEqual(self.models(), [expected(CKPT_A, AE, CLIP_L, T5_16), expected(CKPT_A, AE, CLIP_L_OTHER, T5_16),
                                         expected(CKPT_A, AE, CLIP_L, T5_8), expected(CKPT_A, AE, CLIP_L_OTHER, T5_8)])
        self.assertEqual([cell["module_n"] for cell in g.cells], [stems(*cell["model"][1]) for cell in g.cells])
        self.assertRestored()

    def test_one_slot_on_two_axes_stops_before_the_first_cell(self):
        g = self.grid
        with self.assertRaisesRegex(ValueError, "on two axes that both pick T5-XXL encoders"):
            g.run(self.p, x=(g.te_axis, ["t5xxl_fp16", "t5xxl_fp8"]), y=(g.te_axis, ["Use Forge current", "t5xxl_fp8"]))
        self.assertEqual(g.cells, [])


class FluxGgufGridTests(GridCase):
    UI = (AE, CLIP_L, T5_GGUF)

    def test_a_t5_replaces_the_gguf_t5(self):
        g = self.grid
        g.run(self.p, x=(g.te_axis, ["Use Forge current", "t5xxl_fp16"]))
        self.assertEqual(self.models(), [expected(CKPT_A, AE, CLIP_L, T5_GGUF), expected(CKPT_A, AE, CLIP_L, T5_16)])
        self.assertEqual(g.cells[1]["module_n"], stems(AE, CLIP_L, T5_16))
        self.assertRestored()


# ---------------------------------------------------------------------------------------------------------------------
# 3. Source guards
# ---------------------------------------------------------------------------------------------------------------------

def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise LookupError(name)


def _segment(rel, name):
    source = forge_source(rel)
    return ast.get_source_segment(source, _function(ast.parse(source), name))


def _in_order(test, text, *parts):
    positions = [text.index(part) for part in parts]
    test.assertEqual(positions, sorted(positions), parts)


def _dict_literal(fn, name):
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise LookupError(name)


def _branch_values(expr):
    """The constants an assigned expression can take: a constant, or an ``a if … else b`` chain of them."""
    if isinstance(expr, ast.Constant):
        return {expr.value}
    if isinstance(expr, ast.IfExp):
        return _branch_values(expr.body) | _branch_values(expr.orelse)
    raise AssertionError(f"not a constant choice: {ast.dump(expr)}")


class SourceGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_forge_file("modules/processing.py")

    def test_process_images_hooks_and_settings(self):
        source = forge_source("modules/processing.py")
        segment = _segment("modules/processing.py", "process_images")
        _in_order(self, segment, "p.scripts.before_process(p)", "stored_opts = ",
                  "set_config(p.override_settings, is_api=True, run_callbacks=False, save_config=False)",
                  "manage_model_and_prompt_cache(p)", "_overridden_modules = shared.opts.forge_additional_modules.copy()",
                  "res = process_images_inner(p)", "finally:", "_overridden_modules = None")
        self.assertIn('_vae_override = p.override_settings.pop("sd_vae", None)', segment)
        inner = ast.get_source_segment(source, _function(ast.parse(source), "process_images_inner"))
        # process() before the batch loop's model reload, postprocess_image before each image's infotext, postprocess last
        _in_order(self, inner, "p.scripts.process(p)", "sd_models.forge_model_reload()",
                  "p.scripts.postprocess_image(p, pp)", "info=infotext(i), p=p)", "p.scripts.postprocess(p, res)")

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
        refresh = _segment("scripts/xyz_grid.py", "refresh_loading_params_for_xyz_grid")
        self.assertIn("model_data.forge_loading_parameters = dict(checkpoint_info=checkpoint_info, "
                      "additional_modules=shared.opts.forge_additional_modules", refresh)

    def test_rebuilds_of_the_loading_parameters_go_through_main_entry(self):
        segment = _segment("modules/sysinfo.py", "set_config")
        self.assertIn('if k == "forge_additional_modules":', segment)
        self.assertIn("main_entry.modules_change(v, preset=None, save=False, refresh=False)", segment)
        self.assertIn("main_entry.checkpoint_change(v, preset=None, save=False, refresh=False)", segment)
        self.assertIn("main_entry.refresh_model_loading_parameters()", segment)
        refresh = _segment("modules_forge/main_entry.py", "refresh_model_loading_parameters")
        self.assertIn("model_data.forge_loading_parameters = dict(checkpoint_info=checkpoint_info, "
                      "additional_modules=shared.opts.forge_additional_modules", refresh)
        # main_entry's own callers find the wrapper through the module's globals
        for name in ("modules_change", "checkpoint_change"):
            self.assertIn("refresh_model_loading_parameters(refresh=refresh)",
                          _segment("modules_forge/main_entry.py", name))
        # every rebuild in processing / the refiner calls it through the module (the wrapper is a module attribute)
        for rel in ("modules/processing.py", "modules/sd_samplers_common.py"):
            text = forge_source(rel)
            self.assertIn("from modules_forge import main_entry", text, rel)
            self.assertIn("main_entry.refresh_model_loading_parameters()", text, rel)
            self.assertNotRegex(text, r"(?<![.\w])refresh_model_loading_parameters\(", rel)
            self.assertNotIn("import refresh_model_loading_parameters", text, rel)

    def test_infotext_module_list(self):
        segment = _segment("modules/processing.py", "create_infotext")
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


class LoaderGuardTests(unittest.TestCase):
    """What ``te_slots`` / ``checkpoint_profile`` / ``gguf_remap`` mirror. A new or renamed branch, slot, width or
    rename in Forge fails here instead of the axis quietly refusing that family's encoders."""

    @classmethod
    def setUpClass(cls):
        require_forge_file("backend/loader.py")
        cls.source = forge_source("backend/loader.py")
        cls.fn = _function(ast.parse(cls.source), "replace_state_dict")

    def test_probe_keys(self):
        probes, reads = set(), set()
        for node in ast.walk(self.fn):
            if (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn))
                    and isinstance(node.left, ast.Constant) and isinstance(node.comparators[0], ast.Name)
                    and node.comparators[0].id == "asd"):
                probes.add(node.left.value)
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "asd"
                    and isinstance(node.slice, ast.Constant)):
                reads.add(node.slice.value)
        self.assertEqual(probes, {xte.T5_KEY, xte.T5_QWEIGHT_KEY, xte.GEMMA2_KEY, xte.Q_NORM_KEY, xte.QWEN3VL_KEY,
                                  xte.QWEN25_KEY, xte.POST_ATTN_KEY, xte.QWEN25_VISUAL_KEY, *xte.VAE_KEYS})
        self.assertEqual(reads, {xte.T5_SHARED_KEY, xte.QWEN3VL_MERGER_KEY, xte.QWEN25_KEY, xte.POST_ATTN_KEY,
                                 "first_stage_model.decoder.conv_in.weight"})
        self.assertEqual(tuple(ast.literal_eval(_dict_literal(self.fn, "CLIP_G"))), xte.CLIP_G_KEYS)
        self.assertEqual(tuple(ast.literal_eval(_dict_literal(self.fn, "CLIP_L"))), xte.CLIP_L_KEYS)
        # the order of the text-encoder branches (an elif chain: the first match wins)
        order = [self.source.index(f'"{key}" in asd') for key in (xte.T5_KEY, xte.T5_QWEIGHT_KEY, xte.GEMMA2_KEY,
                                                                   xte.QWEN3VL_KEY, xte.QWEN25_KEY, xte.POST_ATTN_KEY,
                                                                   xte.QWEN25_VISUAL_KEY)]
        self.assertEqual(order, sorted(order))

    def test_widths(self):
        widths = set()
        for node in ast.walk(self.fn):
            if isinstance(node, ast.Compare) and all(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
                widths |= {c.value for c in node.comparators if isinstance(c, ast.Constant) and isinstance(c.value, int)}
            if isinstance(node, ast.MatchValue) and isinstance(node.value, ast.Constant):
                widths.add(node.value.value)
        # 1280 / 768: CLIP-G / CLIP-L; UMT5 vocabulary; Qwen3-VL 4B merger; Qwen2.5 k_proj bias; Qwen3 0.6B / 4B;
        # Ministral 3; the legacy model types (768 / 1280 / 2048); VAE format 0's 4 latent channels
        self.assertEqual(widths, {1280, 768, xte.UMT5_VOCAB, 2560, 512, 1024, 3072, 2048, 4})

    def test_slot_names(self):
        assigned = {}
        for node in ast.walk(self.fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                assigned.setdefault(node.targets[0].id, []).append((node.lineno, node.value))
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                assigned.setdefault(node.target.id, []).append((node.lineno, node.value))
        slots = set()
        for node in ast.walk(self.fn):
            if not (isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.FormattedValue)
                    and isinstance(node.values[0].value, ast.Name)
                    and node.values[0].value.id == "text_encoder_key_prefix"):
                continue
            choices = []
            for part in node.values[1:]:     # up to the first "." - the slot; the rest is the key inside it
                if isinstance(part, ast.Constant):
                    choices.append([part.value])
                    if "." in part.value:
                        break
                else:
                    name = part.value.id
                    latest = max((entry for entry in assigned[name] if entry[0] < node.lineno), key=lambda e: e[0])
                    choices.append([str(value) for value in _branch_values(latest[1])])
            for combo in itertools.product(*choices):
                slots.add("".join(combo).split(".", 1)[0])
        expected = set(xte.SLOT_NAMES) - {"clip_l", "clip_g"}
        if not _forge_reads_qwen3vl_8b():   # Forge 2.29.2: no Qwen3-VL 8B slot, and no model family that reads one
            expected.discard("qwen3vl_8b")
            self.assertNotIn("qwen3vl_8b", forge_source("modules_forge/packages/huggingface_guess/model_list.py"))
        self.assertEqual(slots, expected)

    def test_model_type_tables(self):
        self.assertEqual(ast.literal_eval(_dict_literal(self.fn, "prefix_L")), xte.PREFIX_L)
        self.assertEqual(ast.literal_eval(_dict_literal(self.fn, "prefix_G")), xte.PREFIX_G)
        constants = {name: ast.literal_eval(_dict_literal(self.fn, name)) for name in
                     ("flux_test_key", "flux_test_key_weight", "svdq_test_key", "legacy_test_key")}
        self.assertEqual(constants["legacy_test_key"], xte.LEGACY_TEST_KEY)
        self.assertEqual((constants["flux_test_key"], constants["flux_test_key_weight"], constants["svdq_test_key"]),
                         xte.FLUX_TEST_KEYS)
        match = next(node for node in ast.walk(self.fn) if isinstance(node, ast.Match))
        cases = {case.pattern.value.value: case.body[0].value.value for case in match.cases}
        self.assertEqual(cases, xte.LEGACY_TYPES)
        self.assertIn("sd[legacy_test_key].shape[1]", ast.get_source_segment(self.source, match))
        preprocess = _segment("backend/loader.py", "preprocess_state_dict")
        self.assertIn(f"k.startswith({xte.UNET_PREFIXES!r})", preprocess.replace("'", '"').replace(
            '("model.diffusion_model.", "net.")', repr(xte.UNET_PREFIXES)))
        self.assertIn('f"model.diffusion_model.{k}"', preprocess)

    def test_gguf_remapping(self):
        source = forge_source("backend/loader_gguf.py")
        fn = _function(ast.parse(source), "gguf_remapping")
        for name, ours in (("gguf_t5_format", xte.GGUF_T5_FORMAT), ("gguf_llm_format", xte.GGUF_LLM_FORMAT),
                           ("gguf_clip_vision_format", xte.GGUF_CLIP_VISION_FORMAT)):
            self.assertEqual(tuple(ast.literal_eval(_dict_literal(fn, name)).items()), ours, name)
        tests = [(node.lineno, node.left.value) for node in ast.walk(fn)
                 if isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant)
                 and isinstance(node.comparators[0], ast.Name) and node.comparators[0].id == "state_dict"]
        self.assertEqual([key for _, key in sorted(tests)],
                         [xte.GGUF_T5_TEST_KEY, xte.GGUF_LLM_TEST_KEY, xte.GGUF_PATCH_EMBED_KEY,
                          xte.GGUF_CLIP_VISION_TEST_KEY, "visual.blocks.0.attn_q.weight"])
        self.assertIn('if path.endswith("gguf"):', ast.get_source_segment(self.source, self.fn))
        # load_torch_file: comfy's orig_shape, else the GGUF dimensions reversed
        utils = forge_source("backend/utils.py")
        self.assertIn("if (shape := get_orig_shape(reader, tensor_name)) is None:", utils)
        self.assertIn("shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))", utils)
        self.assertIn('field_key = f"comfy.gguf.orig_shape.{tensor_name}"', _segment("backend/loader_gguf.py",
                                                                                    "get_orig_shape"))


if __name__ == "__main__":
    unittest.main()
