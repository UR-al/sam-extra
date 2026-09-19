"""Anima 3.8B (Qwen3.5 / v2) 편입 — 번들 판별·경로·스크립트 인자·실패 시 순정 유지."""
from __future__ import annotations

import gc
import importlib.util
import json
import re
import sys
import tempfile
import types
import unittest
import weakref
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext.anima38 import files as anima_files  # noqa: E402
from sam3ext.anima38 import marker as anima_marker  # noqa: E402


def _write_safetensors(path: Path, metadata: dict[str, str] | None, keys=("net.x.weight",)) -> None:
    tensors = {key: torch.zeros(2, 2) for key in keys}
    save_file(tensors, str(path), metadata=metadata)


class BundleDetectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_v2_bundle_is_recognised_by_metadata_not_filename(self):
        bundle = self.root / "whatever-name.safetensors"
        _write_safetensors(bundle, {
            "architecture": anima_files.BUNDLE_ARCHITECTURE,
            "anima_v2_bundle_format": anima_files.BUNDLE_FORMAT,
            "anima_v2_connector_prefix": anima_files.CONNECTOR_PREFIX,
        })
        meta = anima_files.bundle_metadata(bundle)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["anima_v2_connector_prefix"], "net.anima_v2_connector.")

    def test_plain_checkpoint_and_missing_file_are_not_bundles(self):
        plain = self.root / "Anima-2.9B.safetensors"
        _write_safetensors(plain, {"format": "pt"})
        self.assertIsNone(anima_files.bundle_metadata(plain))
        self.assertIsNone(anima_files.bundle_metadata(self.root / "nope.safetensors"))

    def test_qwen35_discovery_by_filename_marker(self):
        te = self.root / "text_encoder"
        te.mkdir()
        _write_safetensors(te / "qwen35_4b.safetensors", None)
        _write_safetensors(te / "qwen_3_06b_base.safetensors", None)
        original = anima_files.text_encoder_roots
        anima_files.text_encoder_roots = lambda: [te]
        try:
            found = anima_files.qwen35_models()
        finally:
            anima_files.text_encoder_roots = original
        self.assertEqual(list(found), ["qwen35_4b.safetensors"])


class UnusedQwen35ModuleTests(unittest.TestCase):
    """VAE/Text Encoder 목록에 qwen35_4b 를 넣어 두면 Forge 는 모델을 불러올 때마다 4.8 GB 를 통째로 읽고
    (키에 model. 접두어가 없어) 버린다. 3.8B 는 런타임이 파일을 알아서 찾으므로 로더에서 빼 준다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.qwen35 = self.root / "qwen35_4b.safetensors"
        _write_safetensors(self.qwen35, None, keys=("embed_tokens.weight", "layers.0.linear_attn.A_log"))
        self.qwen3_06b = self.root / "qwen_3_06b_base.safetensors"
        _write_safetensors(self.qwen3_06b, None, keys=("model.layers.0.post_attention_layernorm.weight",))

    def test_only_a_qwen35_file_forge_would_ignore_is_unused(self):
        self.assertTrue(anima_files.is_unused_qwen35_module(self.qwen35))
        self.assertFalse(anima_files.is_unused_qwen35_module(self.qwen3_06b))
        forge_format = self.root / "qwen35_4b_forge.safetensors"   # Forge 가 언젠가 읽게 될 형식이면 건드리지 않는다
        _write_safetensors(forge_format, None, keys=("model.layers.0.input_layernorm.weight",))
        self.assertFalse(anima_files.is_unused_qwen35_module(forge_format))
        self.assertFalse(anima_files.is_unused_qwen35_module(self.root / "qwen35_4b.gguf"))

    def test_loader_filter_drops_it_before_forge_reads_it(self):
        from sam3ext.anima38 import loader_filter

        calls = []
        loader = types.SimpleNamespace(
            split_state_dict=lambda path, additional_state_dicts=None: calls.append(additional_state_dicts) or "split",
        )
        vae = str(self.root / "qwen_image_vae.safetensors")
        loader_filter.install(loader)
        loader_filter.install(loader)   # 두 번 설치해도 한 겹
        result = loader.split_state_dict("ckpt", additional_state_dicts=[vae, str(self.qwen35), str(self.qwen3_06b)])
        self.assertEqual(result, "split")
        self.assertEqual(calls, [[vae, str(self.qwen3_06b)]])

    def test_script_installs_the_filter_on_forges_loader(self):
        original = lambda path, additional_state_dicts=None: additional_state_dicts
        loader = types.ModuleType("backend.loader")
        loader.split_state_dict = original
        with mock.patch.dict(sys.modules, {"backend.loader": loader}):
            _load_script()
        self.assertIsNot(loader.split_state_dict, original)
        self.assertEqual(loader.split_state_dict("ckpt", additional_state_dicts=[str(self.qwen35)]), [])

    def test_console_that_cannot_print_the_path_does_not_break_loading(self):
        from sam3ext.anima38 import loader_filter

        loader = types.SimpleNamespace(split_state_dict=lambda path, additional_state_dicts=None: "split")
        loader_filter.install(loader)
        printed = []

        def console_print(text):
            printed.append(text)
            if len(printed) == 1:   # 코드페이지에 없는 문자 — 문자 집합에 기대지 않고 첫 출력을 실패시킨다
                raise UnicodeEncodeError("cp949", text, 0, 1, "illegal multibyte sequence")

        path = self.root / "モデルー_qwen35_4b.safetensors"
        _write_safetensors(path, None, keys=("layers.0.x",))
        with mock.patch("builtins.print", side_effect=console_print):
            self.assertEqual(loader.split_state_dict("ckpt", additional_state_dicts=[str(path)]), "split")
        self.assertEqual(len(printed), 2, "실패하면 ASCII 로 한 번 더 찍는다")
        self.assertIn("\\u30fc", printed[1])   # '\u30fc' \uac00 ASCII \uc774\uc2a4\ucf00\uc774\ud504\ub85c \ucc0d\ud614\ub2e4
        self.assertTrue(printed[1].isascii())


class RunIdMarkerTests(unittest.TestCase):
    """run id 마커 — NegPiP 같은 래퍼가 조건 텐서를 바꿔도 forward 에서 run 을 찾을 수 있어야 한다."""

    def _placeholder(self, seed=0, length=512, channels=1024, dtype=torch.float32):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(1, length, channels, generator=g).to(dtype)

    def test_roundtrip_and_untouched_rows(self):
        base = self._placeholder()
        stamped = anima_marker.stamp_run_id(base, 4242)
        self.assertEqual(anima_marker.read_run_ids(stamped).tolist(), [4242])
        torch.testing.assert_close(stamped[:, :-1], base[:, :-1], msg="마지막 행만 바뀐다")
        self.assertEqual(anima_marker.read_run_ids(base).tolist(), [-1], "마커 없는 순정 텐서는 -1")

    def test_survives_negpip_sign_flip_and_half_precision(self):
        stamped = anima_marker.stamp_run_id(self._placeholder(), 1_000_000 - 1)
        flipped = stamped * -1.0                     # NegPiP: 토큰 행 × -1
        self.assertEqual(anima_marker.read_run_ids(flipped).tolist(), [999_999])
        for dtype in (torch.float16, torch.bfloat16):
            self.assertEqual(anima_marker.read_run_ids(stamped.to(dtype)).tolist(), [999_999], str(dtype))

    def test_mixed_batch_keeps_native_rows_as_minus_one(self):
        a = anima_marker.stamp_run_id(self._placeholder(1), 7)
        native = self._placeholder(2)
        b = anima_marker.stamp_run_id(self._placeholder(3), 8)
        batch = torch.cat([a, native, b, native])      # cond / uncond 가 한 배치에 섞이는 CFG 상황
        self.assertEqual(anima_marker.read_run_ids(batch).tolist(), [7, -1, 8, -1])

    def test_reads_markers_from_forge_stacked_4d_context(self):
        a = anima_marker.stamp_run_id(self._placeholder(1), 5)
        b = self._placeholder(2)
        stacked = torch.stack([a, b])                    # reconstruct_cond_batch: [B, 1, 512, C]
        self.assertEqual(stacked.shape, (2, 1, 512, 1024))
        rows = anima_marker.as_rows(stacked)
        self.assertEqual(rows.shape, (2, 512, 1024))
        self.assertEqual(anima_marker.read_run_ids(rows).tolist(), [5, -1])
        self.assertEqual(anima_marker.read_run_ids(stacked).tolist(), [-1, -1], "4D 는 그대로 읽으면 마커가 안 보인다 — as_rows 가 필요")
        self.assertEqual(rows.reshape(stacked.shape).shape, stacked.shape)

    def test_rejects_out_of_range_and_narrow_tensors(self):
        with self.assertRaises(ValueError):
            anima_marker.stamp_run_id(self._placeholder(), 1 << anima_marker.BITS)
        narrow = torch.zeros(2, 16, anima_marker.WIDTH - 1)
        self.assertEqual(anima_marker.read_run_ids(narrow).tolist(), [-1, -1])


class PathTests(unittest.TestCase):
    def test_roots_point_at_this_extension_and_its_forge(self):
        self.assertEqual(anima_files.extension_root(), ROOT)
        self.assertEqual(anima_files.forge_root(), ROOT.parents[1], "extensions/<ext> 의 두 단계 위가 Forge 루트")

    def test_bundled_tokenizer_is_shipped_in_assets(self):
        tokenizer = anima_files.tokenizer_dir()
        self.assertEqual(tokenizer, ROOT / "assets" / "qwen35_tokenizer")
        self.assertTrue((tokenizer / "tokenizer.json").is_file())
        self.assertTrue((tokenizer / "tokenizer_config.json").is_file())


def _load_script(script_callbacks=None):
    """Forge 없이 scripts/anima_3_8b.py 를 로드한다 (modules 스텁)."""
    modules_stub = types.ModuleType("modules")
    if script_callbacks is not None:
        modules_stub.script_callbacks = script_callbacks

    class Script:
        def __init__(self):
            pass

    modules_stub.scripts = types.SimpleNamespace(Script=Script, AlwaysVisible=object())
    ui_components = types.ModuleType("modules.ui_components")
    ui_components.InputAccordion = None
    saved = {name: sys.modules.get(name) for name in ("modules", "modules.ui_components", "gradio")}
    sys.modules["modules"] = modules_stub
    sys.modules["modules.ui_components"] = ui_components
    sys.modules.setdefault("gradio", types.ModuleType("gradio"))
    try:
        spec = importlib.util.spec_from_file_location("_test_anima_3_8b", ROOT / "scripts" / "anima_3_8b.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _load_lifecycle_runtime():
    """Load the real runtime, replacing only model-loading/Forge dependencies."""
    stubs = {}
    for name in ("backend", "backend.memory_management", "backend.operations",
                 "backend.patcher", "backend.patcher.clip"):
        module = types.ModuleType(name)
        module.__path__ = []
        stubs[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(stubs[parent], child, module)
    stubs["backend.memory_management"].current_loaded_models = []
    stubs["backend.memory_management"].soft_empty_cache = lambda: None
    stubs["backend.operations"].ForgeOperations = object
    stubs["backend.operations"].using_forge_operations = nullcontext
    stubs["backend.patcher.clip"].CLIP = object
    for name, exports in {
        "adapter": ("ProgressiveCrossAdapter",),
        "qwen35": ("Qwen35HybridModel",),
        "semantic_v2": ("BundledV2Models", "QualityAnchoredSemanticConnectorV2"),
        "tokenizer": ("Qwen35Tokenizer",),
    }.items():
        module = types.ModuleType(f"sam3ext.anima38.{name}")
        for export in exports:
            setattr(module, export, object)
        stubs[module.__name__] = module
    name = "sam3ext.anima38._test_lifecycle_runtime"
    spec = importlib.util.spec_from_file_location(name, ROOT / "sam3ext/anima38/runtime.py")
    module = importlib.util.module_from_spec(spec)
    stubs[name] = module
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    # install() 은 Qwen3.5 파일부터 확인한다 — 이 PC 의 models/text_encoder 에 기대지 않게 (CI 에는 없다)
    module.qwen35_models = lambda: {"qwen35_4b.safetensors": "qwen35_4b.safetensors"}
    return module


class _LifecycleUnet:
    def __init__(self):
        class Diffusion:
            def forward(self, *args, **kwargs):
                return "native"
        self.model = types.SimpleNamespace(diffusion_model=Diffusion())
        self.extra_model_patchers_during_sampling = []

    def add_extra_torch_module_during_sampling(self, module, **kwargs):
        patcher = types.SimpleNamespace(model=module)
        self.extra_model_patchers_during_sampling.append(patcher)
        return patcher


class _LifecycleModel:
    filename = "test-bundle.safetensors"
    text_processing_engine_anima = object()

    def __init__(self):
        self.forge_objects = types.SimpleNamespace(unet=_LifecycleUnet(), clip=object())

    def get_learned_conditioning(self, prompt):
        return []


class _LifecycleProcessing:
    """Forge's real sd_model property follows shared.sd_model, even for old p."""

    def __init__(self, shared):
        self.shared = shared
        self.extra_generation_params = {}
        self.cached_c = self.cached_uc = self.cached_hr_c = self.cached_hr_uc = [1, 2, 3]

    @property
    def sd_model(self):
        return self.shared.sd_model


class RuntimeLifecycleTests(unittest.TestCase):
    def test_failed_generation_then_model_switch_bypass_releases_original_unet(self):
        module = _load_lifecycle_runtime()
        runtime = module.Anima3BRuntime()
        script = _load_script().Anima38Script()
        script._runtime = runtime
        first_model = _LifecycleModel()
        shared = types.SimpleNamespace(sd_model=first_model)
        first = _LifecycleProcessing(shared)
        unrelated_patcher = object()
        owner = first_model.forge_objects.unet
        owner.extra_model_patchers_during_sampling.append(unrelated_patcher)
        metadata = {"anima_v2_adapter_filename": "test-adapter.safetensors"}
        with mock.patch.object(module, "bundle_metadata", return_value=metadata), mock.patch.object(
            runtime, "_load_v2_models", return_value=object(),
        ), mock.patch.object(runtime, "_unload_patchers") as unload:
            script.process_batch(first, {"enabled": False})
            installed_patcher = runtime._v2_sampling_patcher
            self.assertIsNotNone(installed_patcher)
            for key in ("cached_c", "cached_uc", "cached_hr_c", "cached_hr_uc"):
                setattr(first, key, ["stale marker", 1, 2])
            # No postprocess: sampling failed. Forge now replaces shared.sd_model.
            shared.sd_model = _LifecycleModel()
            second = _LifecycleProcessing(shared)
            script.process_batch(second, {"bypass": True})
            self.assertEqual(owner.extra_model_patchers_during_sampling, [unrelated_patcher])
            self.assertEqual(second.sd_model.forge_objects.unet.extra_model_patchers_during_sampling, [])
            unload.assert_called_once_with(installed_patcher)
        self.assertIsNone(script._installed_for)
        self.assertIsNone(runtime._v2_sampling_patcher)
        for key in ("cached_c", "cached_uc", "cached_hr_c", "cached_hr_uc"):
            self.assertEqual(getattr(first, key), [None, None, None])

    def test_direct_restore_cleans_owner_and_sampling_clone_and_is_idempotent(self):
        module = _load_lifecycle_runtime()
        runtime = module.Anima3BRuntime()
        model = _LifecycleModel()
        shared = types.SimpleNamespace(sd_model=model)
        first = _LifecycleProcessing(shared)
        owner = model.forge_objects.unet
        with mock.patch.object(module, "bundle_metadata", return_value={"bundle": "v2"}), mock.patch.object(
            runtime, "_load_v2_models", return_value=object(),
        ), mock.patch.object(runtime, "_unload_patchers") as unload:
            runtime.install(first, "adapter.safetensors", 1.0, None)
            installed_patcher = runtime._v2_sampling_patcher
            clone = _LifecycleUnet()
            clone.extra_model_patchers_during_sampling = owner.extra_model_patchers_during_sampling.copy()
            model.forge_objects.unet = clone
            second = _LifecycleProcessing(shared)
            first.cached_c = ["old marker", 1, 2]
            runtime.restore(second)
            runtime.restore(second)
            self.assertEqual(owner.extra_model_patchers_during_sampling, [])
            self.assertEqual(clone.extra_model_patchers_during_sampling, [])
            unload.assert_called_once_with(installed_patcher)
        self.assertEqual(first.cached_c, [None, None, None])
        self.assertEqual(second.cached_c, [None, None, None])
        self.assertEqual(runtime._v2_sampling_unets, [])
        self.assertIsNone(runtime._installed_processing)

    def test_partial_install_failure_releases_its_extra_patcher(self):
        module = _load_lifecycle_runtime()
        runtime = module.Anima3BRuntime()
        script = _load_script().Anima38Script()
        script._runtime = runtime
        first = _LifecycleProcessing(types.SimpleNamespace(sd_model=_LifecycleModel()))
        # Force failure after _install_v2 has attached its sampling patcher.
        first.extra_generation_params = None
        with mock.patch.object(module, "bundle_metadata", return_value={"bundle": "v2"}), mock.patch.object(
            runtime, "_load_v2_models", return_value=object(),
        ), mock.patch.object(runtime, "_unload_patchers") as unload, mock.patch("traceback.print_exc"):
            script.process_batch(first, {"enabled": False})
            self.assertEqual(first.sd_model.forge_objects.unet.extra_model_patchers_during_sampling, [])
            self.assertEqual(unload.call_count, 1)
        self.assertIsNone(script._installed_for)
        self.assertIsNone(runtime._v2_sampling_patcher)
        self.assertIsNone(runtime._installed_processing)
        self.assertFalse(runtime._cond_active)
        self.assertFalse(runtime._v2_active)


class ModelRetentionTests(unittest.TestCase):
    """공용 런타임은 프로세스 내내 산다. 이전 모델(3.8B 면 DiT·TE·VAE 약 8 GiB)을 붙잡으면 체크포인트를
    바꿔도(XYZ 체크포인트 축 포함) Forge 가 RAM 에서 못 비운다."""

    def test_runtime_does_not_keep_a_switched_out_checkpoint_alive(self):
        module = _load_lifecycle_runtime()
        runtime = module.Anima3BRuntime()
        script = _load_script().Anima38Script()
        script._runtime = runtime
        shared = types.SimpleNamespace(sd_model=_LifecycleModel())
        p = _LifecycleProcessing(shared)
        with mock.patch.object(module, "bundle_metadata", return_value={"anima_v2_adapter_filename": "a"}),                 mock.patch.object(runtime, "_load_v2_models", return_value=object()):
            script.process_batch(p, {})
            script.postprocess(p, None)
        old_model = weakref.ref(shared.sd_model)
        old_dit = weakref.ref(shared.sd_model.forge_objects.unet.model.diffusion_model)
        shared.sd_model = _LifecycleModel()   # Forge 가 체크포인트를 바꾼다
        del p
        gc.collect()
        self.assertIsNone(old_model(), "런타임이 이전 sd_model 을 붙잡고 있다")
        self.assertIsNone(old_dit(), "런타임이 이전 DiT 를 붙잡고 있다")

    def test_crashed_generation_does_not_keep_the_old_checkpoint_alive_either(self):
        # 샘플링이 예외로 끝나면 postprocess(restore) 가 안 불린다 — 그 상태로 체크포인트를 바꿔도 Forge 가
        # 이전 모델을 비울 수 있어야 한다 (다음 생성의 process_batch 보다 모델 재로드가 먼저다).
        module = _load_lifecycle_runtime()
        runtime = module.Anima3BRuntime()
        script = _load_script().Anima38Script()
        script._runtime = runtime
        shared = types.SimpleNamespace(sd_model=_LifecycleModel())
        p = _LifecycleProcessing(shared)
        with mock.patch.object(module, "bundle_metadata", return_value={"anima_v2_adapter_filename": "a"}), \
                mock.patch.object(runtime, "_load_v2_models", return_value=object()):
            script.process_batch(p, {})   # postprocess 없이 끝난다
        old_model = weakref.ref(shared.sd_model)
        old_dit = weakref.ref(shared.sd_model.forge_objects.unet.model.diffusion_model)
        shared.sd_model = _LifecycleModel()
        gc.collect()
        self.assertIsNone(old_model(), "런타임이 이전 sd_model 을 붙잡고 있다")
        self.assertIsNone(old_dit(), "런타임이 이전 DiT 를 붙잡고 있다")

    def test_loaded_model_callback_drops_connector_built_for_another_model(self):
        module = _load_lifecycle_runtime()
        runtime = module.Anima3BRuntime()
        old_adapter, new_adapter = torch.nn.Module(), torch.nn.Module()
        runtime._v2_key = ("bundle", id(old_adapter))
        runtime._v2_models = types.SimpleNamespace(native_adapter=torch.nn.Module())   # 커넥터 전용 사본
        runtime._v2_source = weakref.ref(old_adapter)
        runtime._adapter_key = ("v1", id(old_adapter))
        runtime._adapter = types.SimpleNamespace(native_adapter=old_adapter)

        def model_with(adapter):
            clip = types.SimpleNamespace(cond_stage_model=types.SimpleNamespace(
                qwen3_06b=types.SimpleNamespace(llm_adapter=adapter)))
            return types.SimpleNamespace(forge_objects=types.SimpleNamespace(clip=clip))

        runtime.release_stale_caches(model_with(old_adapter))   # 같은 모델 — 그대로 둔다
        self.assertIsNotNone(runtime._v2_models)
        runtime.release_stale_caches(model_with(new_adapter))
        self.assertEqual((runtime._v2_models, runtime._v2_key, runtime._adapter, runtime._adapter_key), (None,) * 4)
        runtime._v2_models = types.SimpleNamespace(native_adapter=torch.nn.Module())
        runtime._v2_source = weakref.ref(new_adapter)
        runtime.release_stale_caches(object())                 # Anima 가 아닌 모델
        self.assertIsNone(runtime._v2_models)


    def test_script_releases_caches_when_forge_loads_another_model(self):
        registered = []
        callbacks = types.SimpleNamespace(on_model_loaded=registered.append)
        _load_script(script_callbacks=callbacks)
        self.assertEqual(len(registered), 1)
        runtime = mock.Mock()
        fake_module = types.ModuleType("sam3ext.anima38.runtime")
        fake_module._SHARED_RUNTIME = runtime
        new_model = object()
        with mock.patch.dict(sys.modules, {"sam3ext.anima38.runtime": fake_module}):
            registered[0](new_model)
        runtime.release_stale_caches.assert_called_once_with(new_model)


class EncodeReuseTests(unittest.TestCase):
    """같은 프롬프트를 생성·batch count·Feature 6 후보마다 Qwen3.5-4B(4.45 GiB 를 GPU 로 올렸다 내림)로
    다시 인코딩하지 않는다."""

    def setUp(self):
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        self.load_gpu = mock.Mock()
        self.module.memory_management.load_model_gpu = self.load_gpu
        self.qwen_clip = types.SimpleNamespace(patcher=types.SimpleNamespace(load_device="cpu"))
        adapter = types.SimpleNamespace(embed=types.SimpleNamespace(weight=torch.zeros(1, dtype=torch.float16)))
        self.native_clip = types.SimpleNamespace(
            cond_stage_model=types.SimpleNamespace(qwen3_06b=types.SimpleNamespace(llm_adapter=adapter)),
            patcher=types.SimpleNamespace(load_device="cpu", offload_device="cpu"),
        )
        self.runtime._qwen_path = "qwen35_4b.safetensors"
        self.semantic_calls = []

        def semantic_layers(model, tokenizer, line, device):
            self.semantic_calls.append(line)
            return [torch.full((1, 2, 4), float(len(self.semantic_calls)))] * 4, torch.ones(1, 2)

        for name, value in (
            ("_load_qwen", mock.Mock(return_value=("qwen", "tokenizer", self.qwen_clip))),
            ("_semantic_layers", semantic_layers),
            ("_native_inputs", lambda engine, line, device, dtype: (torch.zeros(1, 2, 4), torch.zeros(1, 2), torch.ones(1, 2, 1))),
        ):
            patcher = mock.patch.object(self.runtime, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _qwen_loads(self):
        return sum(1 for call in self.load_gpu.call_args_list if call.args[0] is self.qwen_clip.patcher)

    def test_unchanged_prompt_skips_qwen35(self):
        _, _, first = self.runtime._extract_prompt_features(object(), self.native_clip, ["girl", "cat"])
        _, _, second = self.runtime._extract_prompt_features(object(), self.native_clip, ["girl", "cat"])
        self.assertEqual(self.semantic_calls, ["girl", "cat"])
        self.assertEqual(self._qwen_loads(), 1, "두 번째는 Qwen3.5 를 GPU 로 올리지 않는다")
        torch.testing.assert_close(second[1][0][0], first[1][0][0])
        self.runtime._extract_prompt_features(object(), self.native_clip, ["girl", "dog"])
        self.assertEqual(self.semantic_calls, ["girl", "cat", "dog"])
        self.assertEqual(self._qwen_loads(), 2)

    def test_native_negative_keeps_forges_shared_negative_cache(self):
        model = _LifecycleModel()
        p = _LifecycleProcessing(types.SimpleNamespace(sd_model=model))
        shared_uc, shared_c = [None] * 3, [None] * 3
        p.cached_uc = p.cached_hr_uc = shared_uc
        p.cached_c = shared_c
        with mock.patch.object(self.module, "bundle_metadata", return_value={"anima_v2_adapter_filename": "a"}),                 mock.patch.object(self.runtime, "_load_v2_models", return_value=object()):
            self.runtime.install(p, "a", 1.0, None)
        self.assertIs(p.cached_uc, shared_uc, "마커 없는 순정 부정 조건은 Forge 의 공용 캐시를 그대로 쓴다")
        self.assertIsNot(p.cached_c, shared_c, "마커가 든 긍정 조건은 이 생성 전용 캐시")
        self.runtime.restore(p)
        p.cached_uc = shared_uc
        with mock.patch.object(self.module, "bundle_metadata", return_value={"anima_v2_adapter_filename": "a"}),                 mock.patch.object(self.runtime, "_load_v2_models", return_value=object()):
            self.runtime.install(p, "a", 1.0, 1.0)   # 부정도 v2 → 마커가 든다
        self.assertIsNot(p.cached_uc, shared_uc)


class LoraCloneTests(unittest.TestCase):
    """LoRA 세트가 바뀌면 Forge 는 forge_objects_original.unet 을 복제해 새 UNet 으로 샘플링한다(설치가
    process_images 보다 앞서는 Feature 6, batch count 사이 LoRA 가 바뀌는 dynamic prompts). 커넥터 패처가
    그 복제본에 없으면 1.2 GiB 커넥터가 메모리 관리 밖에서 매 스텝 CPU→GPU 로 흘러간다."""

    def test_connector_patcher_follows_forges_lora_clone_and_is_purged_on_restore(self):
        module = _load_lifecycle_runtime()
        runtime = module.Anima3BRuntime()
        model = _LifecycleModel()
        clip = model.forge_objects.clip
        original = model.forge_objects.unet

        def clone_of(source):   # UnetPatcher.clone: 모델 공유 + extra 패처 목록 복사
            clone = _LifecycleUnet()
            clone.model = source.model
            clone.extra_model_patchers_during_sampling = source.extra_model_patchers_during_sampling.copy()
            return clone

        def use(unet):
            model.forge_objects = types.SimpleNamespace(unet=unet, clip=clip)
            model.forge_objects_after_applying_lora = types.SimpleNamespace(unet=unet, clip=clip)

        model.forge_objects_original = types.SimpleNamespace(unet=original, clip=clip)
        style_clone = clone_of(original)   # 지난 생성의 LoRA 복제본
        use(style_clone)
        p = _LifecycleProcessing(types.SimpleNamespace(sd_model=model))
        with mock.patch.object(module, "bundle_metadata", return_value={"anima_v2_adapter_filename": "a"}),                 mock.patch.object(runtime, "_load_v2_models", return_value=object()):
            runtime.install(p, "a", 1.0, None)
        patcher = runtime._v2_sampling_patcher
        edit_clone = clone_of(original)    # process_images 안에서 Edit LoRA 로 새로 복제
        use(edit_clone)

        def carries(unet):
            return sum(1 for item in unet.extra_model_patchers_during_sampling if item is patcher)

        self.assertEqual(carries(edit_clone), 1, "샘플링할 복제본에 커넥터 패처가 있어야 Forge 가 GPU 에 올린다")
        self.assertEqual(carries(style_clone), 1)
        runtime.restore(p)
        self.assertEqual([carries(u) for u in (original, style_clone, edit_clone)], [0, 0, 0])


class MissingEncoderFailFastTests(unittest.TestCase):
    """Qwen3.5 는 샘플링 직전(setup_conds)에 게으르게 로드된다. 파일이 없으면 install 에서 바로 실패해야
    스크립트가 경고 한 번 후 순정 Anima 로 계속 간다 — 안 그러면 모든 3.8B 생성이 setup_conds 에서 죽는다."""

    def setUp(self):
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        self.model = _LifecycleModel()
        self.p = _LifecycleProcessing(types.SimpleNamespace(sd_model=self.model))
        self.v2_metadata = {"anima_v2_adapter_filename": "test-adapter.safetensors"}

    def _assert_nothing_patched(self):
        self.assertNotIn("get_learned_conditioning", vars(self.model))
        self.assertEqual(self.model.forge_objects.unet.extra_model_patchers_during_sampling, [])
        self.assertIsNone(self.runtime._installed_processing)

    def test_v2_bundle_without_qwen35_fails_before_patching(self):
        with mock.patch.object(self.module, "bundle_metadata", return_value=self.v2_metadata),                 mock.patch.object(self.module, "qwen35_models", return_value={}),                 mock.patch.object(self.runtime, "_load_v2_models", return_value=object()):
            with self.assertRaises(FileNotFoundError):
                self.runtime.install(self.p, "a", 1.0, None)
        self._assert_nothing_patched()

    def test_v1_without_the_adapter_file_fails_before_patching(self):
        with mock.patch.object(self.module, "bundle_metadata", return_value=None),                 mock.patch.object(self.module, "qwen35_models", return_value={"qwen35_4b.safetensors": "x"}),                 mock.patch.object(self.module, "adapters", return_value={}):
            with self.assertRaises(FileNotFoundError):
                self.runtime.install(self.p, "missing.safetensors", 1.0, None)
        self._assert_nothing_patched()

    def test_script_warns_once_and_generates_natively(self):
        script = _load_script().Anima38Script()
        script._runtime = self.runtime
        with mock.patch.object(self.module, "bundle_metadata", return_value=self.v2_metadata),                 mock.patch.object(self.module, "qwen35_models", return_value={}),                 mock.patch.object(self.runtime, "_load_v2_models", return_value=object()):
            script.process_batch(self.p, {})
        self.assertTrue(script._warned_missing)
        self.assertIsNone(script._installed_for)
        self._assert_nothing_patched()


class _FakeRuntime:
    def __init__(self, v2: bool, fail: Exception | None = None):
        self.v2 = v2
        self.fail = fail
        self.installs: list[tuple] = []
        self.restores = 0
        self._installed_processing = None

    def is_v2_bundle(self, sd_model):
        return self.v2

    def install(self, p, adapter, strength, negative_strength):
        if self.fail is not None:
            raise self.fail
        self.installs.append((adapter, strength, negative_strength))
        self._installed_processing = p

    def restore(self, p):
        self.restores += 1
        self._installed_processing = None

    def install_is_current(self, p):
        return self._installed_processing is p

    def _sync_adapter_lora(self, clip):
        self.synced = getattr(self, "synced", []) + [clip]

    def ensure_attached(self, p):
        self.attached = getattr(self, "attached", 0) + 1


class ScriptArgTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_script()

    def test_sam3_inner_pass_inherits_the_outer_install(self):
        # SAM3 가 생성 도중 돌리는 인페인트 패스(build_i2i)는 같은 스크립트 인스턴스를 공유한다.
        script = self.module.Anima38Script()
        script._runtime = _FakeRuntime(v2=True)
        outer = types.SimpleNamespace(sd_model=object())
        script.process_batch(outer, {})
        inner = types.SimpleNamespace(sd_model=outer.sd_model, _sam3_inner=True, _sam3_outer=outer)
        script.process_batch(inner, {})
        script.postprocess(inner, None)
        self.assertEqual((len(script._runtime.installs), script._runtime.restores), (1, 0))
        self.assertIs(script._installed_for, outer, "그 뒤의 ADetailer 패스도 v2 를 받아야 한다")
        script.postprocess(outer, None)
        self.assertEqual(script._runtime.restores, 1)

    def test_sam3_pass_without_a_live_outer_install_installs_for_itself(self):
        script = self.module.Anima38Script()
        script._runtime = _FakeRuntime(v2=True)
        stale_outer = types.SimpleNamespace(sd_model=object())
        refine = types.SimpleNamespace(sd_model=object(), _sam3_inner=True, _sam3_outer=stale_outer)
        script.process_batch(refine, {})
        self.assertEqual(len(script._runtime.installs), 1)
        self.assertIs(script._installed_for, refine)
        script.postprocess(refine, None)
        self.assertEqual(script._runtime.restores, 1)

    def test_crash_in_one_tab_does_not_leak_v2_into_a_bypass_run_in_the_other(self):
        # txt2img·img2img 는 스크립트 인스턴스가 다르지만 런타임은 하나다. 샘플링 중 예외로 postprocess 가
        # 안 불리면 txt2img 의 설치가 켜진 채 남는다 — img2img 의 Bypass 생성이 v2 로 돌면 안 된다.
        runtime = _FakeRuntime(v2=True)
        txt2img, img2img = self.module.Anima38Script(), self.module.Anima38Script()
        txt2img._runtime = runtime
        crashed = types.SimpleNamespace(sd_model=object())
        txt2img.process_batch(crashed, {})          # postprocess 없이 끝난다
        fake_module = types.ModuleType("sam3ext.anima38.runtime")
        fake_module._SHARED_RUNTIME = runtime       # img2img 인스턴스는 런타임을 아직 안 잡았다
        with mock.patch.dict(sys.modules, {"sam3ext.anima38.runtime": fake_module}):
            img2img.process_batch(types.SimpleNamespace(sd_model=object()), {"bypass": True})
        self.assertEqual(runtime.restores, 1)
        self.assertIsNone(runtime._installed_processing)

    def test_batch_count_iterations_keep_the_install(self):
        # n_iter 마다 process_batch 가 불린다 — 같은 생성이면 내렸다 다시 달지 않아야 조건 캐시·run 이 이어진다.
        script = self.module.Anima38Script()
        script._runtime = _FakeRuntime(v2=True)
        p = types.SimpleNamespace(sd_model=object())
        for _ in range(3):
            script.process_batch(p, {})
        self.assertEqual((len(script._runtime.installs), script._runtime.restores), (1, 0))
        script.process_batch(types.SimpleNamespace(sd_model=object()), {})   # 다음 생성
        self.assertEqual((len(script._runtime.installs), script._runtime.restores), (2, 1))

    def test_positional_and_dict_args_agree(self):
        coerce = self.module.coerce_args
        self.assertEqual(coerce((True, "a.safetensors", 0.5, True, 0.25)),
                         {"enabled": True, "adapter": "a.safetensors", "strength": 0.5, "negative": True, "negative_strength": 0.25, "bypass": False})
        self.assertEqual(coerce(({"enabled": "true", "strength": "0.5", "negative": 1},)),
                         {"enabled": True, "adapter": self.module.DEFAULT_ADAPTER, "strength": 0.5, "negative": True, "negative_strength": 1.0, "bypass": False})
        self.assertEqual(coerce(()), dict(self.module.ARG_DEFAULTS))
        self.assertEqual(coerce(({"strength": 99, "junk": 1},))["strength"], 2.0, "강도는 0~2 로 자른다")

    def test_v2_bundle_activates_even_when_disabled(self):
        script = self.module.Anima38Script()
        script._runtime = _FakeRuntime(v2=True)
        script.process_batch(types.SimpleNamespace(sd_model=object()), False, "x", 1.0, False, 1.0)
        self.assertEqual(script._runtime.installs, [("x", 1.0, None)])
        script.postprocess(types.SimpleNamespace(sd_model=object()), None)
        self.assertEqual(script._runtime.restores, 1)

    def test_v1_needs_the_checkbox_and_passes_negative_strength(self):
        script = self.module.Anima38Script()
        script._runtime = _FakeRuntime(v2=False)
        p = types.SimpleNamespace(sd_model=object())
        script.process_batch(p, False, "x", 1.0, False, 1.0)
        self.assertEqual(script._runtime.installs, [], "v1 은 켜야 돈다")
        script.process_batch(p, {"enabled": True, "adapter": "v1.safetensors", "strength": 0.8, "negative": True, "negative_strength": 0.4})
        self.assertEqual(script._runtime.installs, [("v1.safetensors", 0.8, 0.4)])

    def test_bypass_turns_the_v2_bundle_off(self):
        script = self.module.Anima38Script()
        script._runtime = _FakeRuntime(v2=True)
        p = types.SimpleNamespace(sd_model=object())
        script.process_batch(p, {"bypass": True})
        self.assertEqual(script._runtime.installs, [], "bypass 면 v2 번들도 순정")
        script.process_batch(p, False, "x", 1.0, False, 1.0, True)   # 위치 인자 6번째
        self.assertEqual(script._runtime.installs, [])
        script.process_batch(p, False, "x", 1.0, False, 1.0, False)
        self.assertEqual(len(script._runtime.installs), 1)

    def test_stale_patch_from_a_failed_run_is_restored_before_bypass(self):
        script = self.module.Anima38Script()
        script._runtime = _FakeRuntime(v2=True)
        p = types.SimpleNamespace(sd_model=object())
        script.process_batch(p, False, "x", 1.0, False, 1.0)   # 설치됨, postprocess 는 안 불림(샘플링 예외 상황)
        self.assertEqual(len(script._runtime.installs), 1)
        script.process_batch(p, {"bypass": True})
        self.assertEqual(script._runtime.restores, 1, "다음 생성이 bypass 라도 남은 패치를 먼저 원복")
        self.assertIsNone(script._installed_for)

    def test_restore_leaves_the_patch_in_place_but_inert(self):
        """다른 확장(NegPiP)이 같은 속성을 비LIFO 로 해제해도 사고가 없게 — 우리는 되돌리지 않는다."""
        script = self.module.Anima38Script()
        script._runtime = _FakeRuntime(v2=True)
        p = types.SimpleNamespace(sd_model=object())
        script.process_batch(p, False, "x", 1.0, False, 1.0)
        script.postprocess(p, None)
        self.assertEqual(script._runtime.restores, 1, "postprocess 는 런타임에 원복을 알린다")
        self.assertIsNone(script._installed_for)

    def test_missing_qwen35_does_not_kill_the_generation(self):
        script = self.module.Anima38Script()
        script._runtime = _FakeRuntime(v2=True, fail=FileNotFoundError("qwen35_4b.safetensors was not found"))
        p = types.SimpleNamespace(sd_model=object())
        script.process_batch(p, False, "x", 1.0, False, 1.0)   # raise 하지 않는다
        self.assertEqual(script._runtime.restores, 1, "실패하면 바로 원복해 순정 Anima 로 간다")
        self.assertTrue(script._warned_missing)


class _FakeConnector(torch.nn.Module):
    """_load_v2_models 가 이름을 바꿔 넣는 파라미터 뱅크만 가진 커넥터."""

    def __init__(self, native_adapter, **config):
        super().__init__()
        self.quality_anchor = torch.nn.Module()
        self.quality_anchor.parameter_bank = torch.nn.Module()
        self.quality_anchor.parameter_bank.layer_mix_logits = torch.nn.Parameter(torch.empty(4))
        self.semantic_resampler = torch.nn.Module()
        self.semantic_resampler.parameter_bank = torch.nn.Module()
        bank = self.semantic_resampler.parameter_bank
        bank.query_tokens = torch.nn.Parameter(torch.empty(1, 2, 8))
        bank.layer_embeddings = torch.nn.Parameter(torch.empty(4, 1, 8))


class _FakeBundledModels(torch.nn.Module):
    def __init__(self, native_adapter, connector):   # semantic_v2.BundledV2Models 와 같은 모양
        super().__init__()
        self.native_adapter = native_adapter
        self.connector = connector


class _FakeAdapter(torch.nn.Module):
    def __init__(self, native_adapter):
        super().__init__()
        self.parameter_bank = torch.nn.Module()
        self.parameter_bank.layer_mix_logits = torch.nn.Parameter(torch.zeros(4))


class _FakeQwen(torch.nn.Module):
    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.RMSNorm(8)


class _FakeLLMAdapter(torch.nn.Module):
    """Forge 의 LLMAdapter 처럼 인자 없이 만들어지는 native llm_adapter."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(4, 8)


class ConnectorAdapterLoraTests(unittest.TestCase):
    """v2 커넥터는 샘플링 때 llm_adapter 를 다시 돌린다. TE 에 걸린 LoRA 의 llm_adapter 몫이 거기 빠지면
    (anima-rl 처럼 llm_adapter 키가 든 LoRA) Bypass·v1 과 결과가 달라진다. TE 모듈을 같이 패치하면 두
    패처가 같은 가중치를 제자리 패치해 이중 적용되므로, 커넥터는 번들 원본으로 만든 자기 사본을 쓴다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        self.te_adapter = _FakeLLMAdapter()
        with torch.no_grad():
            self.te_adapter.embed.weight.fill_(5.0)   # TE 쪽은 지금 LoRA 로 제자리 패치된 상태일 수 있다

    def _bundle(self, with_adapter=True):
        path = Path(self.tmp.name) / "bundle.safetensors"
        prefix = anima_files.CONNECTOR_PREFIX
        tensors = {
            prefix + "quality_anchor.layer_mix_logits": torch.zeros(4),
            prefix + "semantic_resampler.query_tokens": torch.zeros(1, 2, 8),
            prefix + "semantic_resampler.layer_embeddings": torch.zeros(4, 1, 8),
        }
        if with_adapter:
            tensors["net.llm_adapter.embed.weight"] = torch.ones(4, 8)
        save_file(tensors, str(path))
        return str(path)

    def _load(self, path):
        native_clip = types.SimpleNamespace(cond_stage_model=types.SimpleNamespace(
            qwen3_06b=types.SimpleNamespace(llm_adapter=self.te_adapter)))
        metadata = {"anima_v2_adapter_architecture": anima_files.V2_ARCHITECTURE}
        with mock.patch.object(self.module, "QualityAnchoredSemanticConnectorV2", _FakeConnector),                 mock.patch.object(self.module, "BundledV2Models", _FakeBundledModels),                 mock.patch.object(self.module, "using_forge_operations", lambda **_: nullcontext()),                 mock.patch.object(self.runtime, "_require_anima", return_value=(None, native_clip)):
            return self.runtime._load_v2_models(object(), path, metadata)

    def test_connector_gets_a_private_adapter_with_the_bundles_original_weights(self):
        models = self._load(self._bundle())
        self.assertIsNot(models.native_adapter, self.te_adapter)
        torch.testing.assert_close(models.native_adapter.embed.weight, torch.ones(4, 8))
        self.assertEqual(float(self.te_adapter.embed.weight[0, 0]), 5.0, "TE 모듈은 건드리지 않는다")
        self.assertIs(self._load(self._bundle()), models, "같은 모델이면 캐시")

    def test_bundle_without_adapter_weights_falls_back_to_sharing(self):
        self.assertIs(self._load(self._bundle(with_adapter=False)).native_adapter, self.te_adapter)

    def test_llm_adapter_lora_patches_follow_the_te_onto_the_private_copy(self):
        models = self._load(self._bundle())
        patcher = types.SimpleNamespace(patches={"connector.x.weight": ["keep"]}, patches_uuid="u0")
        self.runtime._v2_sampling_patcher = patcher
        patch = ("lora-a",)
        te_patches = {"qwen3_06b.llm_adapter.blocks.0.q.weight": [patch], "qwen3_06b.model.layers.0.w": [("te",)]}
        native_clip = types.SimpleNamespace(patcher=types.SimpleNamespace(patches=te_patches))
        self.runtime._sync_adapter_lora(native_clip)
        self.assertEqual(patcher.patches, {"connector.x.weight": ["keep"], "native_adapter.blocks.0.q.weight": [patch]})
        applied = patcher.patches_uuid
        self.assertNotEqual(applied, "u0", "uuid 가 바뀌어야 Forge 가 다음 로드 때 다시 패치한다")
        self.runtime._sync_adapter_lora(native_clip)
        self.assertEqual(patcher.patches_uuid, applied, "그대로면 다시 패치하지 않는다")
        native_clip.patcher.patches = {}
        self.runtime._sync_adapter_lora(native_clip)
        self.assertEqual(patcher.patches, {"connector.x.weight": ["keep"]})
        self.assertIsNotNone(models)

    def test_shared_adapter_never_receives_te_patches(self):
        self._load(self._bundle(with_adapter=False))
        patcher = types.SimpleNamespace(patches={}, patches_uuid="u0")
        self.runtime._v2_sampling_patcher = patcher
        native_clip = types.SimpleNamespace(patcher=types.SimpleNamespace(
            patches={"qwen3_06b.llm_adapter.blocks.0.q.weight": [("lora",)]}))
        self.runtime._sync_adapter_lora(native_clip)
        self.assertEqual((patcher.patches, patcher.patches_uuid), ({}, "u0"), "공유 모듈이면 TE 패처와 이중 적용된다")


class InferenceModeLoaderTests(unittest.TestCase):
    """Forge 는 process_batch·setup_conds 를 torch.inference_mode() 안에서 부르고, 추가 패처 언로드
    (postprocess 의 model.to(offload)) 는 그 밖에서 한다. inference 텐서로 만든 가중치가 밖에서 옮겨지면
    버전 카운터 없는 파라미터가 되어, 다음 샘플링의 인덱싱이 "Inference tensors do not track version
    counter." 로 죽는다 (3.8B 에서 Feature 6 실행 시 semantic_v2.py 의 layer_embeddings[index])."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        self.native_adapter = torch.nn.Module()
        self.native_adapter.embed = torch.nn.Embedding(4, 8)

    def _assert_survives_forge_offload(self, model, probe):
        for name, param in model.named_parameters():
            self.assertFalse(param.is_inference(), name)
        model.to(torch.float16)   # Forge 언로드와 같은 자리: inference_mode 밖
        with torch.inference_mode():
            probe(model)[0]        # 샘플링 중 semantic_v2 가 하는 인덱싱

    def test_v2_connector_loaded_inside_inference_mode_survives_offload(self):
        path = Path(self.tmp.name) / "bundle.safetensors"
        prefix = anima_files.CONNECTOR_PREFIX
        save_file({
            prefix + "quality_anchor.layer_mix_logits": torch.zeros(4),
            prefix + "semantic_resampler.query_tokens": torch.zeros(1, 2, 8),
            prefix + "semantic_resampler.layer_embeddings": torch.zeros(4, 1, 8),
        }, str(path))
        native_clip = types.SimpleNamespace(cond_stage_model=types.SimpleNamespace(
            qwen3_06b=types.SimpleNamespace(llm_adapter=self.native_adapter)))
        metadata = {"anima_v2_adapter_architecture": anima_files.V2_ARCHITECTURE}
        with mock.patch.object(self.module, "QualityAnchoredSemanticConnectorV2", _FakeConnector), \
                mock.patch.object(self.module, "BundledV2Models", _FakeBundledModels), \
                mock.patch.object(self.module, "using_forge_operations", lambda **_: nullcontext()), \
                mock.patch.object(self.runtime, "_require_anima", return_value=(None, native_clip)):
            with torch.inference_mode():   # Anima38Script.process_batch → install
                models = self.runtime._load_v2_models(object(), str(path), metadata)
        self._assert_survives_forge_offload(
            models, lambda m: m.connector.semantic_resampler.parameter_bank.layer_embeddings)

    def test_v1_adapter_loaded_inside_inference_mode_survives_offload(self):
        path = Path(self.tmp.name) / "adapter.safetensors"
        save_file({"layer_mix_logits": torch.zeros(4)}, str(path))
        with mock.patch.object(self.module, "adapters", return_value={"a": str(path)}), \
                mock.patch.object(self.module, "ProgressiveCrossAdapter", _FakeAdapter):
            with torch.inference_mode():   # encode() 는 @torch.inference_mode()
                adapter = self.runtime._load_adapter("a", self.native_adapter)
        self._assert_survives_forge_offload(adapter, lambda m: m.parameter_bank.layer_mix_logits)

    def test_qwen_loaded_inside_inference_mode_survives_offload(self):
        path = Path(self.tmp.name) / "qwen35_4b.safetensors"
        save_file({"model.norm.weight": torch.ones(8)}, str(path))
        with mock.patch.object(self.module, "qwen35_models", return_value={path.name: str(path)}), \
                mock.patch.object(self.module, "Qwen35HybridModel", _FakeQwen), \
                mock.patch.object(self.module, "Qwen35Tokenizer", object), \
                mock.patch.object(self.module, "CLIP", lambda **kw: types.SimpleNamespace(**kw)), \
                mock.patch.object(self.module, "using_forge_operations", lambda **_: nullcontext()):
            with torch.inference_mode():   # setup_conds → get_learned_conditioning
                model, _, _ = self.runtime._load_qwen()
        self._assert_survives_forge_offload(model, lambda m: m.model.norm.weight)


class _ReferenceModel:
    """backend/diffusion_engine/anima.py 의 Anima 가 레퍼런스에 쓰는 필드만 가진 모델."""

    filename = "bundle.safetensors"

    def __init__(self):
        self.text_processing_engine_anima = object()
        self.forge_objects = types.SimpleNamespace(clip=types.SimpleNamespace(patcher=object()))
        self.ref_latents = ["stitch"]   # ImageStitch 가 넣는 레퍼런스
        self.ini_latent = "canvas"      # img2img 캔버스 (encode_first_stage)

    def get_learned_conditioning(self, prompt):
        return ["native"]


class _Prompt(list):
    def __init__(self, lines, negative=False):
        super().__init__(lines)
        self.is_negative_prompt = negative


class NativeReferenceHandoffTests(unittest.TestCase):
    """v1/v2 경로는 순정 get_learned_conditioning 을 건너뛰지만, 샘플링 때 DiT 가 읽는
    dynamic_args.ref_latents 는 거기서만 채워진다 — 3.8B 에서 Feature 6 레퍼런스가 꺼지던 원인."""

    def setUp(self):
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        self.runtime._active_bundle_metadata = {"architecture": "bundle"}
        self.dynamic_args = types.SimpleNamespace(ref_latents=["stale"])
        self.opts = types.SimpleNamespace(anima_do_reference=True)
        state = mock.patch.object(
            self.module, "_anima_reference_state", return_value=(self.dynamic_args, self.opts),
        )
        state.start()
        self.addCleanup(state.stop)
        self.model = _ReferenceModel()

    def _encode(self, prompt, negative_strength=None):
        with mock.patch.object(self.runtime, "_encode_v2", return_value=["v2"]) as encode_v2:
            result = self.runtime.encode(self.model, prompt, "a", 1.0, negative_strength)
        return result, encode_v2

    def test_positive_v2_prompt_hands_canvas_and_stitch_references_to_the_dit(self):
        result, encode_v2 = self._encode(_Prompt(["girl"]))
        self.assertEqual(result, ["v2"])
        encode_v2.assert_called_once()
        self.assertEqual(self.dynamic_args.ref_latents, ["canvas", "stitch"])
        self.assertIsNone(self.model.ini_latent, "캔버스는 한 번만 쓰인다 (순정과 같다)")

    def test_reference_option_off_clears_leftover_references(self):
        self.opts.anima_do_reference = False
        self._encode(_Prompt(["girl"]))
        self.assertEqual(self.dynamic_args.ref_latents, [])
        self.assertEqual(self.model.ini_latent, "canvas")

    def test_negative_prompt_leaves_references_alone(self):
        self._encode(_Prompt(["bad"], negative=True), negative_strength=1.0)
        self.assertEqual(self.dynamic_args.ref_latents, ["stale"])
        self.assertEqual(self.model.ini_latent, "canvas")


class PartialLoadNormTests(unittest.TestCase):
    """VRAM 이 모자라 Forge 가 Qwen3.5 를 부분 로드하면 Forge ops 가 아닌 RMSNorm 은 CPU 에 남는다.
    입력(GPU)과 장치가 달라도 forward 가 가중치를 입력 장치로 옮겨야 한다 — meta 가 GPU 역할."""

    def test_norms_follow_the_input_device(self):
        from sam3ext.anima38 import layers

        for norm_class in (layers.RMSNorm, layers.Qwen35RMSNorm):
            with self.subTest(norm=norm_class.__name__):
                norm = norm_class(8)   # 가중치는 CPU 에 남아 있다
                out = norm(torch.ones(2, 8, device="meta", dtype=torch.float16))
                self.assertEqual((out.device.type, out.dtype, tuple(out.shape)), ("meta", torch.float16, (2, 8)))


class StatusAndPasteTests(unittest.TestCase):
    """3.8B 가 켜졌는지·Bypass 인지·왜 꺼졌는지를 infotext 와 아코디언에서 알 수 있어야 하고, PNG Info 로
    붙여 넣으면 같은 설정으로 다시 생성돼야 한다 (Bypass 로 만든 이미지를 다시 뽑으면 v2 가 켜지던 문제)."""

    def setUp(self):
        self.module = _load_script()

    def _run(self, runtime, args):
        script = self.module.Anima38Script()
        script._runtime = runtime
        p = types.SimpleNamespace(sd_model=object(), extra_generation_params={})
        script.process_batch(p, args)
        return script, p

    def test_status_is_written_to_infotext(self):
        missing = FileNotFoundError("qwen35_4b.safetensors was not found in models/text_encoder.")
        cases = [
            (_FakeRuntime(v2=True), {}, "v2 bundle"),
            (_FakeRuntime(v2=False), {"enabled": True}, "v1 adapter"),
            (_FakeRuntime(v2=True), {"bypass": True}, "bypass"),
            (_FakeRuntime(v2=True, fail=missing), {}, "off: qwen35_4b.safetensors was not found in models/text_encoder."),
            (_FakeRuntime(v2=True, fail=RuntimeError("boom")), {}, "off: install failed (RuntimeError)"),
        ]
        for runtime, args, expected in cases:
            with self.subTest(expected=expected), mock.patch("traceback.print_exc"):
                script, p = self._run(runtime, args)
                self.assertEqual(p.extra_generation_params.get("Anima38"), expected)
                self.assertEqual(script._last_status, expected)

    def test_plain_anima_leaves_no_marker(self):
        for args in ({}, {"bypass": True}):
            with self.subTest(args=args):
                _, p = self._run(_FakeRuntime(v2=False), args)
                self.assertNotIn("Anima38", p.extra_generation_params)

    def _forge_round_trip(self, params):
        """Forge 가 infotext 를 쓰고(k: quote(v)) 다시 읽는 방식 그대로 — modules/infotext_utils.py 의
        re_param_code·quote/unquote. 키에 '.' 이 있으면 이 정규식이 키를 잘라 버린다."""
        def quote(text):
            text = str(text)
            return text if not any(c in text for c in ",\n:") else json.dumps(text, ensure_ascii=False)

        line = ", ".join(f"{key}: {quote(value)}" for key, value in params.items())
        pattern = re.compile(r'\s*([\w\s\-\/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')
        parsed = {}
        for key, value in pattern.findall(line):
            if value[:1] == '"' and value[-1:] == '"':
                value = json.loads(value)
            parsed[key] = value
        return parsed

    def test_png_info_restores_what_the_generation_recorded(self):
        m = self.module
        script, p = self._run(_FakeRuntime(v2=True), {"bypass": True})
        self.assertIs(m._paste_bypass(self._forge_round_trip(p.extra_generation_params)), True)

        module = _load_lifecycle_runtime()
        runtime = module.Anima3BRuntime()
        v1 = _LifecycleProcessing(types.SimpleNamespace(sd_model=_LifecycleModel()))
        with mock.patch.object(module, "bundle_metadata", return_value=None), \
                mock.patch.object(module, "adapters", return_value={"a, v1.safetensors": "x"}):
            runtime.install(v1, "a, v1.safetensors", 0.7, 1.3)
        parsed = self._forge_round_trip(v1.extra_generation_params)
        self.assertIs(m._paste_v1_enabled(parsed), True)
        self.assertEqual(m._paste_v1_adapter(parsed), "a, v1.safetensors")
        self.assertEqual(m._paste_v1_strength(parsed), 0.7)
        self.assertIs(m._paste_negative(parsed), True)
        self.assertEqual(m._paste_negative_strength(parsed), 1.3)

        v2 = _LifecycleProcessing(types.SimpleNamespace(sd_model=_LifecycleModel()))
        with mock.patch.object(module, "bundle_metadata", return_value={"anima_v2_adapter_filename": "bundle-adapter"}), \
                mock.patch.object(runtime, "_load_v2_models", return_value=object()):
            runtime.install(v2, "a", 1.0, None)
        parsed = self._forge_round_trip({**v2.extra_generation_params, "Anima38": "v2 bundle"})
        self.assertIs(m._paste_bypass(parsed), False)
        self.assertIs(m._paste_negative(parsed), False)
        self.assertIs(m._paste_v1_enabled(parsed), False)
        self.assertIsNone(m._paste_v1_adapter(parsed), "v2 어댑터 이름은 v1 드롭다운에 없다")
        self.assertIsNone(m._paste_v1_strength(parsed))

    def test_images_saved_with_the_old_dotted_keys_still_paste(self):
        m = self.module
        legacy = self._forge_round_trip({
            "Anima 3.8B": "bypass",
            "Anima 3.8B adapter": "a.safetensors",
            "Anima 3.8B strength": 0.8,
            "Anima 3.8B architecture": anima_files.ARCHITECTURE,
            "Anima 3.8B negative strength": 1.0,
        })
        self.assertIn("8B", legacy, "Forge 파서가 옛 키를 이렇게 자른다")
        self.assertIs(m._paste_bypass(legacy), True)
        self.assertIs(m._paste_v1_enabled(legacy), True)
        self.assertEqual(m._paste_v1_adapter(legacy), "a.safetensors")
        self.assertEqual(m._paste_v1_strength(legacy), 0.8)
        self.assertIs(m._paste_negative(legacy), True)

    def test_images_without_3_8b_records_leave_the_ui_alone(self):
        m = self.module
        parsed = self._forge_round_trip({"Steps": 30, "Sampler": "Euler a"})
        for paste in (m._paste_bypass, m._paste_negative, m._paste_negative_strength,
                      m._paste_v1_enabled, m._paste_v1_adapter, m._paste_v1_strength):
            self.assertIsNone(paste(parsed), paste.__name__)

    def test_plain_checkpoint_run_updates_the_last_status_without_infotext(self):
        script, p = self._run(_FakeRuntime(v2=False), {})
        self.assertEqual(script._last_status, "off: not a 3.8B checkpoint")
        self.assertNotIn("Anima38", p.extra_generation_params)

    def test_status_panel_reports_encoder_model_and_last_run(self):
        with mock.patch.object(anima_files, "qwen35_models", return_value={"qwen35_4b.safetensors": "C:/m/qwen35_4b.safetensors"}), \
                mock.patch.object(anima_files, "bundle_metadata", return_value={"architecture": "bundle"}):
            text = self.module._status_markdown("C:/m/Anima-3.8B-v1.1.safetensors", "v2 bundle")
        self.assertIn("qwen35_4b.safetensors", text)
        self.assertIn("자동", text)
        self.assertIn("v2 번들", text)
        self.assertIn("v2 bundle", text)
        with mock.patch.object(anima_files, "qwen35_models", return_value={}), \
                mock.patch.object(anima_files, "bundle_metadata", return_value=None):
            text = self.module._status_markdown("C:/m/anima-base.safetensors", None)
        self.assertIn("없음", text)
        self.assertIn("3.8B 번들이 아닙니다", text)
        text = self.module._status_markdown(None, None)
        self.assertIn("아직 선택된 체크포인트가 없습니다", text)

    def test_status_panel_uses_the_selected_checkpoint_before_the_first_generation(self):
        # Forge 는 첫 생성 전까지 shared.sd_model 이 FakeInitialModel(체크포인트 정보 없음)이다
        info = types.SimpleNamespace(filename="C:/m/Anima-3.8B-v1.1.safetensors")
        placeholder = types.SimpleNamespace()
        loading = {"checkpoint_info": info}
        self.assertEqual(self.module._selected_checkpoint_path(placeholder, loading), info.filename)
        loaded = types.SimpleNamespace(sd_checkpoint_info=types.SimpleNamespace(filename="C:/m/loaded.safetensors"))
        self.assertEqual(self.module._selected_checkpoint_path(loaded, loading), info.filename,
                         "로드된 모델보다 드롭다운에서 고른 체크포인트가 다음 생성에 쓰인다")
        self.assertEqual(self.module._selected_checkpoint_path(loaded, {}), "C:/m/loaded.safetensors")
        self.assertIsNone(self.module._selected_checkpoint_path(placeholder, {}))

    def test_install_records_encoder_file_and_negative_path(self):
        module = _load_lifecycle_runtime()
        runtime = module.Anima3BRuntime()
        p = _LifecycleProcessing(types.SimpleNamespace(sd_model=_LifecycleModel()))
        with mock.patch.object(module, "bundle_metadata", return_value={"anima_v2_adapter_filename": "a"}), \
                mock.patch.object(runtime, "_load_v2_models", return_value=object()):
            runtime.install(p, "a", 1.0, None)
            self.assertEqual(p.extra_generation_params["Anima38 encoder"], "qwen35_4b.safetensors")
            self.assertEqual(p.extra_generation_params["Anima38 negative"], "native")
            runtime.install(p, "a", 1.0, 1.0)
            self.assertEqual(p.extra_generation_params["Anima38 negative"], "connector")


_V2_WIDTH = anima_marker.WIDTH + 10   # 마커가 들어갈 폭보다 넓게
_V2_TOKENS = 6


def _v2_features(count):
    """_extract_prompt_features 대역 — 줄마다 (source, target_ids, target_weights) 와 (Qwen3.5 4개 층, mask)."""
    native_rows = [
        (
            torch.full((1, _V2_TOKENS, _V2_WIDTH), float(index + 1)),
            torch.arange(_V2_TOKENS).unsqueeze(0),
            torch.ones(1, _V2_TOKENS, 1),
        )
        for index in range(count)
    ]
    semantic_rows = [([torch.zeros(1, 3, 16)] * 4, torch.ones(1, 3)) for _ in range(count)]
    return object(), native_rows, semantic_rows


def _anima_cond_wrapper(runtime, model, below):
    """install() 이 거는 것과 같은 조건 래퍼 — 표시(_anima3b_patch)와 그 아래 함수(_anima3b_below)."""

    def patched(prompt):
        return runtime._conditioning_entry(model, prompt)

    patched._anima3b_patch = runtime
    patched._anima3b_below = below
    runtime._wrappers.add(patched)
    return patched


class _CondModel:
    """조건 진입점용 Anima — 클래스의 순정 get_learned_conditioning 은 받은 줄을 남긴다."""

    filename = "bundle.safetensors"
    text_processing_engine_anima = object()

    def __init__(self):
        self.forge_objects = types.SimpleNamespace(clip=types.SimpleNamespace(
            patcher=types.SimpleNamespace(offload_device="cpu")))
        self.native_lines = []

    def get_learned_conditioning(self, prompt):
        self.native_lines.append(list(prompt))
        return [torch.full((1, 512, _V2_WIDTH), 7.0) for _ in prompt]


class ConditioningEntryTests(unittest.TestCase):
    """get_learned_conditioning 자리의 진입점 — 켜져 있으면 긍정을 v2 로 인코딩하고, 순정 부정·꺼진 상태는
    클래스의 순정 함수(또는 살아 있는 NegPiP)로 넘긴다. 다른 확장의 낡은 인스턴스 래퍼로 새면 안 된다."""

    def setUp(self):
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        state = mock.patch.object(self.module, "_anima_reference_state", return_value=(
            types.SimpleNamespace(ref_latents=[]), types.SimpleNamespace(anima_do_reference=False)))
        state.start()
        self.addCleanup(state.stop)
        self.model = _CondModel()
        self.below = mock.Mock(return_value=["below"])   # 설치 당시 우리 아래에 있던 인스턴스 래퍼
        self.model.get_learned_conditioning = _anima_cond_wrapper(self.runtime, self.model, self.below)

    def _activate(self):
        self.runtime._cond_active = True
        self.runtime._cond_params = ("a", 1.0, None)
        self.runtime._active_bundle_metadata = {"anima_v2_adapter_filename": "a"}

    def test_positive_v2_prompt_returns_one_marked_tensor_per_line(self):
        self._activate()
        features = _v2_features(2)
        with mock.patch.object(self.runtime, "_extract_prompt_features", return_value=features):
            conds = self.runtime._conditioning_entry(self.model, _Prompt(["girl", "cat"]))
        self.assertIsInstance(conds, list, "Forge 네이티브 계약: 줄마다 텐서 하나")
        self.assertEqual([tuple(cond.shape) for cond in conds], [(1, 512, _V2_WIDTH)] * 2)
        run_ids = [int(anima_marker.read_run_ids(cond)[0]) for cond in conds]
        self.assertEqual(len(set(run_ids)), 2, "줄마다 run 이 따로")
        for (source, _, _), cond, run_id in zip(features[1], conds, run_ids):
            self.assertIn(run_id, self.runtime._v2_runs)
            self.assertIs(self.runtime._v2_runs[run_id].source, source)
            torch.testing.assert_close(cond[:, :_V2_TOKENS], source, msg="자리표시는 순정 source")
        self.assertEqual(self.model.native_lines, [])
        self.below.assert_not_called()

    def test_native_negative_goes_to_the_class_method_not_an_instance_wrapper(self):
        self._activate()
        with mock.patch.object(self.runtime, "_extract_prompt_features") as extract:
            conds = self.model.get_learned_conditioning(_Prompt(["bad"], negative=True))
        extract.assert_not_called()
        self.below.assert_not_called()
        self.assertEqual(self.model.native_lines, [["bad"]])
        self.assertEqual(anima_marker.read_run_ids(conds[0]).tolist(), [-1], "순정 부정엔 마커가 없다")
        self.assertEqual(self.runtime._v2_runs, {})

    def test_inactive_entry_uses_the_wrapper_below_only_while_negpip_is_patched(self):
        prompt = _Prompt(["girl"])
        self.model.orig_forward = _CondModel.get_learned_conditioning.__get__(self.model)   # NegPiP 패치 중
        self.assertEqual(self.model.get_learned_conditioning(prompt), ["below"])
        self.below.assert_called_once_with(prompt)
        self.assertEqual(self.model.native_lines, [])
        del self.model.orig_forward   # NegPiP 가 풀렸다 — 아래 래퍼는 이제 낡았다
        self.model.get_learned_conditioning(prompt)
        self.assertEqual(self.below.call_count, 1)
        self.assertEqual(self.model.native_lines, [["girl"]])


class _RecordingDiT:
    """클래스 forward 가 받은 context·kwargs 를 남기는 DiT."""

    def __init__(self):
        self.calls = []

    def forward(self, x, timesteps, context, *args, **kwargs):
        self.calls.append(types.SimpleNamespace(context=context, args=args, kwargs=kwargs))
        return "native"


class V2ForwardTests(unittest.TestCase):
    """샘플링 forward — 마커가 있는 행만 커넥터 출력으로 바꾸고, 나머지(CFG 의 순정 부정)는 그대로 둔다."""

    def setUp(self):
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        self.connector_timesteps = []

        def connector(source, target_ids, semantic, semantic_source_mask=None, timesteps=None):
            self.connector_timesteps.append(timesteps)
            return torch.full((source.shape[0], _V2_TOKENS, _V2_WIDTH), 3.0)

        self.models = types.SimpleNamespace(
            connector=connector,
            native_adapter=types.SimpleNamespace(embed=types.SimpleNamespace(weight=torch.zeros(1))),
        )
        self.model = _LifecycleModel()
        self.dit = _RecordingDiT()
        self.model.forge_objects.unet.model.diffusion_model = self.dit
        self.p = _LifecycleProcessing(types.SimpleNamespace(sd_model=self.model))
        self.uncond = torch.randn(1, 512, _V2_WIDTH, generator=torch.Generator().manual_seed(0))
        self.x = torch.zeros(2, 4)
        self.t = torch.tensor([0.5, 0.5])

    def _install(self):
        def load(sd_model, path, metadata):   # 실제 _load_v2_models 처럼 런타임에 둔다
            self.runtime._v2_models = self.models
            return self.models

        with mock.patch.object(self.runtime, "_load_v2_models", side_effect=load):
            self.runtime._install_v2(self.p, "bundle.safetensors", {})

    def _encode(self, count=1):
        with mock.patch.object(self.runtime, "_extract_prompt_features", return_value=_v2_features(count)):
            return self.runtime._encode_v2(object(), object(), ["line"] * count)

    @staticmethod
    def _expanded():
        """커넥터 출력(3.0, 토큰 수만큼)을 512 로 채운 한 줄."""
        row = torch.zeros(1, 512, _V2_WIDTH)
        row[:, :_V2_TOKENS] = 3.0
        return row

    def test_cfg_batch_replaces_only_the_marked_row(self):
        self._install()
        (cond,) = self._encode()
        self.dit.forward(self.x, self.t, torch.cat([cond, self.uncond]))
        (call,) = self.dit.calls
        self.assertEqual(tuple(call.context.shape), (2, 512, _V2_WIDTH))
        torch.testing.assert_close(call.context[:1], self._expanded())
        self.assertTrue(torch.equal(call.context[1:], self.uncond), "순정 부정 행은 그대로")
        self.assertEqual(len(self.connector_timesteps), 1)

    def test_forge_stacked_4d_context_keeps_its_shape(self):
        self._install()
        (cond,) = self._encode()
        context = torch.stack([cond, self.uncond])   # reconstruct_cond_batch: [B, 1, 512, C]
        self.dit.forward(self.x, self.t, context)
        seen = self.dit.calls[0].context
        self.assertEqual(seen.shape, context.shape)
        torch.testing.assert_close(seen[0], self._expanded())
        self.assertTrue(torch.equal(seen[1], self.uncond))

    def test_short_run_ids_and_timesteps_repeat_over_the_batch(self):
        self.runtime._v2_models = self.models
        (cond,) = self._encode()
        run_id = int(anima_marker.read_run_ids(cond)[0])
        context = torch.cat([cond, self.uncond, cond, self.uncond])
        out = self.runtime._expand_v2_context(context, torch.tensor([10.0, 20.0]), torch.tensor([run_id, -1]))
        self.assertEqual(tuple(out.shape), (4, 512, _V2_WIDTH))
        for row in (0, 2):
            torch.testing.assert_close(out[row : row + 1], self._expanded())
        for row in (1, 3):
            self.assertTrue(torch.equal(out[row : row + 1], self.uncond))
        (timesteps,) = self.connector_timesteps   # 같은 run 의 두 행은 커넥터 한 번에
        torch.testing.assert_close(timesteps, torch.tensor([10.0, 10.0]))

    def test_unknown_run_id_raises(self):
        self._install()
        stale = anima_marker.stamp_run_id(torch.zeros(1, 512, _V2_WIDTH), 999)   # 등록되지 않은 run
        with self.assertRaisesRegex(RuntimeError, "999"):
            self.dit.forward(self.x, self.t, torch.cat([stale, self.uncond]))
        self.assertEqual(self.dit.calls, [])

    def test_inactive_runtime_passes_the_context_through(self):
        self._install()
        (cond,) = self._encode()
        self.runtime._v2_active = False   # restore 뒤에도 래퍼는 남아 투명해야 한다
        context = torch.cat([cond, self.uncond])
        self.dit.forward(self.x, self.t, context)
        self.assertIs(self.dit.calls[0].context, context)
        self.assertEqual(self.connector_timesteps, [])

    def test_negpip_mask_hits_only_marked_rows_and_fills_a_missing_option(self):
        self._install()
        (cond,) = self._encode()
        context = torch.cat([cond, self.uncond])
        mask = torch.ones(2, 512, 1)
        mask[0, :2] = -1.0   # 긍정 줄 앞 두 토큰이 NegPiP 음수 가중치
        mask[1] = -1.0       # 순정 부정 행은 NegPiP 가 이미 곱해 두었다 — 다시 곱하지 않는다
        self.dit.forward(self.x, self.t, context, c_negpip_mask=mask)
        call = self.dit.calls[-1]
        torch.testing.assert_close(call.context[:1], self._expanded() * mask[:1])
        self.assertTrue(torch.equal(call.context[1:], self.uncond))
        self.assertIs(call.kwargs["c_negpip_mask"], mask)
        self.assertIs(call.kwargs["transformer_options"]["negpip_mask"], mask)

        existing = torch.ones(2, 512, 1)   # NegPiP 가 위에서 이미 넣어 둔 값
        self.dit.forward(self.x, self.t, context, c_negpip_mask=mask,
                         transformer_options={"negpip_mask": existing, "keep": 1})
        options = self.dit.calls[-1].kwargs["transformer_options"]
        self.assertIs(options["negpip_mask"], existing)
        self.assertEqual(options["keep"], 1)

    def test_second_install_does_not_wrap_again(self):
        self._install()
        wrapper = self.dit.forward
        self.assertIs(getattr(wrapper, "_anima3b_patch", None), self.runtime)
        self.runtime._v2_active = False
        self._install()
        self.assertIs(self.dit.forward, wrapper)
        self.assertNotIn("orig_forward", vars(self.dit))
        self.assertTrue(self.runtime._v2_active)

    def test_install_under_negpip_does_not_add_a_second_wrapper(self):
        from functools import wraps

        for copies_tag in (False, True):   # 실제 NegPiP 는 @wraps 로 우리 표시까지 복사한다
            with self.subTest(copies_tag=copies_tag):
                dit = self.dit = _RecordingDiT()
                self.model.forge_objects.unet.model.diffusion_model = dit
                self._install()
                ours = dit.forward

                def negpip_forward(x, timesteps, context, padding_mask=None, **kwargs):
                    return dit.orig_forward(x, timesteps, context, padding_mask, **kwargs)

                if copies_tag:
                    negpip_forward = wraps(ours)(negpip_forward)
                dit.orig_forward, dit.forward = ours, negpip_forward   # NegPiP 가 우리 위에 설치됨
                self.runtime._v2_active = False
                self._install()
                self.assertIs(dit.forward, negpip_forward)
                self.assertIs(dit.orig_forward, ours)
                self.assertTrue(self.runtime._v2_active)
                (cond,) = self._encode()
                dit.forward(self.x[:1], self.t[:1], cond)
                (call,) = dit.calls
                torch.testing.assert_close(call.context, self._expanded())


class V2LoaderValidationTests(unittest.TestCase):
    """번들의 메타데이터·텐서가 v2 커넥터와 맞지 않으면 커넥터를 만들기 전에 실패하고, 맞으면 번들이 적어 둔
    크기로 커넥터를 만든다."""

    V2 = {"anima_v2_adapter_architecture": anima_files.V2_ARCHITECTURE}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        self.native_adapter = torch.nn.Module()
        self.native_adapter.embed = torch.nn.Embedding(4, 8)
        self.configs = []

    def _bundle(self, name="bundle.safetensors", connector=True):
        tensors = {"net.x.weight": torch.zeros(2, 2)}
        if connector:
            prefix = anima_files.CONNECTOR_PREFIX
            tensors.update({
                prefix + "quality_anchor.layer_mix_logits": torch.zeros(4),
                prefix + "semantic_resampler.query_tokens": torch.zeros(1, 2, 8),
                prefix + "semantic_resampler.layer_embeddings": torch.zeros(4, 1, 8),
            })
        path = Path(self.tmp.name) / name
        save_file(tensors, str(path))
        return str(path)

    def _load(self, path, metadata):
        configs = self.configs

        class RecordingConnector(_FakeConnector):
            def __init__(self, native_adapter, **config):
                configs.append(config)
                super().__init__(native_adapter, **config)

        native_clip = types.SimpleNamespace(cond_stage_model=types.SimpleNamespace(
            qwen3_06b=types.SimpleNamespace(llm_adapter=self.native_adapter)))
        with mock.patch.object(self.module, "QualityAnchoredSemanticConnectorV2", RecordingConnector), \
                mock.patch.object(self.module, "BundledV2Models", _FakeBundledModels), \
                mock.patch.object(self.module, "using_forge_operations", lambda **_: nullcontext()), \
                mock.patch.object(self.runtime, "_require_anima", return_value=(None, native_clip)):
            return self.runtime._load_v2_models(object(), path, metadata)

    def _assert_nothing_built(self):
        self.assertEqual(self.configs, [])
        self.assertIsNone(self.runtime._v2_models)
        self.assertIsNone(self.runtime._v2_key)

    def test_non_v2_architecture_is_rejected(self):
        for metadata in ({"anima_v2_adapter_architecture": anima_files.ARCHITECTURE}, {}):
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(RuntimeError, "not Semantic Connector v2"):
                    self._load(self._bundle(), metadata)
                self._assert_nothing_built()

    def test_bundle_without_connector_tensors_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "no connector tensors"):
            self._load(self._bundle(connector=False), self.V2)
        self._assert_nothing_built()

    def test_bundle_metadata_sizes_reach_the_connector(self):
        self._load(self._bundle("defaults.safetensors"), self.V2)
        self._load(self._bundle("custom.safetensors"), {
            **self.V2,
            "anima_v2_adapter_semantic_query_tokens": "32",
            "anima_v2_adapter_semantic_resampler_blocks": "3",
            "anima_v2_adapter_semantic_resampler_dim": "1024",
            "anima_v2_adapter_semantic_resampler_heads": "8",
            "anima_v2_adapter_semantic_resampler_mlp_hidden_dim": "2816",
        })
        self.assertEqual(self.configs, [
            {"num_queries": 64, "resampler_blocks": 6, "resampler_dim": 2048, "resampler_heads": 16,
             "mlp_hidden_dim": 5632},
            {"num_queries": 32, "resampler_blocks": 3, "resampler_dim": 1024, "resampler_heads": 8,
             "mlp_hidden_dim": 2816},
        ])


class NegPipAboveUsTests(unittest.TestCase):
    """NegPiP 가 우리 조건 래퍼 *위* 에 설치되면(스크립트 순서상 흔하다) functools.wraps 가 우리 래퍼의
    __dict__(표시 속성 포함)를 NegPiP 래퍼에 복사한다. 표시 속성으로 알아보면 NegPiP 래퍼를 우리 것으로
    오인해 마스킹을 대신 적용하고 dict 를 돌려주어, NegPiP 의 ``assert isinstance(conds, list)`` 가 죽는다."""

    def test_negpip_wrapper_above_us_gets_a_plain_list(self):
        module = _load_lifecycle_runtime()
        runtime = module.Anima3BRuntime()
        model = _LifecycleModel()
        p = _LifecycleProcessing(types.SimpleNamespace(sd_model=model))
        with mock.patch.object(module, "bundle_metadata", return_value={"anima_v2_adapter_filename": "a"}), \
                mock.patch.object(runtime, "_load_v2_models", return_value=object()):
            runtime.install(p, "a", 1.0, None)
        ours = model.get_learned_conditioning

        # NegPiP 설치 (lib_negpip/anima.py 와 같은 모양)
        model.orig_forward = model.get_learned_conditioning

        @wraps(model.orig_forward)
        def negpip_learned_conditioning(prompt):
            conds = model.orig_forward(prompt)
            assert isinstance(conds, list), type(conds)
            return {"crossattn": torch.stack(conds), "negpip": True}

        model.get_learned_conditioning = negpip_learned_conditioning
        self.assertIs(getattr(negpip_learned_conditioning, "_anima3b_patch", None), runtime, "wraps 가 표시를 복사한다")
        self.assertIs(runtime._our_cond_wrapper(model), ours)

        lib = types.ModuleType("lib_negpip")
        lib.__path__ = []
        anima = types.ModuleType("lib_negpip.anima")
        anima._build_negpip_mask = lambda engine, line, n, device, dtype: torch.ones(n, device=device, dtype=dtype)
        reference = (types.SimpleNamespace(ref_latents=[]), types.SimpleNamespace(anima_do_reference=False))
        with mock.patch.dict(sys.modules, {"lib_negpip": lib, "lib_negpip.anima": anima}), \
                mock.patch.object(module, "_anima_reference_state", return_value=reference), \
                mock.patch.object(runtime, "_extract_prompt_features", return_value=_v2_features(1)):
            result = model.get_learned_conditioning(_Prompt(["girl"]))
        self.assertTrue(result["negpip"], "NegPiP 가 자기 마스킹을 한 번만 한다")


class FastPathLivenessTests(unittest.TestCase):
    """batch count 사이 설치 유지(B7)와 SAM3 안쪽 패스 물려받기(A3)는 설치가 아직 살아 있을 때만 맞다.
    Forge 는 Hires 체크포인트·Refiner 가 있으면 반복마다 모델을 새로 불러오고(processing.py 의
    forge_model_reload), NegPiP 가 앞 순서면 반복마다 자기 패치를 원복·재패치하며 우리 래퍼를 떨군다."""

    def setUp(self):
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        self.script = _load_script().Anima38Script()
        self.script._runtime = self.runtime
        self.shared = types.SimpleNamespace(sd_model=_LifecycleModel())
        self.p = _LifecycleProcessing(self.shared)
        patches = (
            mock.patch.object(self.module, "bundle_metadata", return_value={"anima_v2_adapter_filename": "a"}),
            mock.patch.object(self.runtime, "_load_v2_models", return_value=object()),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _live_on(self, model):
        dit = model.forge_objects.unet.model.diffusion_model
        return self.runtime._cond_installed(model) and (
            self.runtime._patched(dit.forward) or self.runtime._patched(getattr(dit, "orig_forward", None))
        )

    def test_model_reloaded_between_iterations_gets_a_fresh_install(self):
        self.script.process_batch(self.p, {})
        self.shared.sd_model = _LifecycleModel()   # Hires 체크포인트 → 다음 반복 전에 1차 모델을 새로 로드
        self.p.cached_c = ["marked conds of the old model", 1, 2]
        self.script.process_batch(self.p, {})
        self.assertTrue(self._live_on(self.shared.sd_model), "새로 불러온 모델에 다시 설치돼야 한다")
        self.assertEqual(self.p.cached_c, [None, None, None], "옛 설치의 마커 조건을 다시 쓰지 않는다")

    def test_negpip_first_repatch_between_iterations_is_reattached_without_losing_runs(self):
        model = self.shared.sd_model
        self.script.process_batch(self.p, {})
        self.runtime._v2_runs[7] = "run of iteration 0"
        # NegPiP 가 앞 순서: 반복마다 자기 패치를 원복(우리 래퍼도 같이 빠진다)하고 순정 위에 다시 건다
        native = type(model).get_learned_conditioning.__get__(model, type(model))
        model.orig_forward = native

        def negpip(prompt):
            return model.orig_forward(prompt)

        model.get_learned_conditioning = negpip
        dit = model.forge_objects.unet.model.diffusion_model
        dit.forward = type(dit).forward.__get__(dit, type(dit))
        self.assertFalse(self._live_on(model))
        self.script.process_batch(self.p, {})
        self.assertTrue(self._live_on(model))
        self.assertIs(self.runtime._our_cond_wrapper(model)._anima3b_below, negpip)
        self.assertEqual(self.runtime._v2_runs.get(7), "run of iteration 0", "같은 생성 — run 을 지우지 않는다")

    def test_sam3_inner_pass_reattaches_under_the_outer_install(self):
        model = self.shared.sd_model
        self.script.process_batch(self.p, {})
        model.orig_forward = type(model).get_learned_conditioning.__get__(model, type(model))
        model.get_learned_conditioning = lambda prompt: model.orig_forward(prompt)   # NegPiP 가 안쪽 패스에서 재패치
        inner = types.SimpleNamespace(sd_model=model, _sam3_inner=True, _sam3_outer=self.p, extra_generation_params={})
        self.script.process_batch(inner, {})
        self.assertTrue(self._live_on(model))
        self.assertIs(self.runtime._installed_processing, self.p, "바깥 설치는 그대로")


class SamplingTimeLoraSyncTests(unittest.TestCase):
    """Forge 는 하이레스 조건을 인코딩하려고 LoRA 세트를 바꿨다가 1차 샘플링 전에 되돌리고, ADetailer 는 자기
    프롬프트로 인코딩한다. 인코딩 때 맞춘 llm_adapter 패치로 샘플링하면 엉뚱한 LoRA 세트가 커넥터에 걸린다 —
    샘플링 직전(process_before_every_sampling, 그 패스의 LoRA 활성화 뒤)에 다시 맞춘다."""

    def setUp(self):
        self.module = _load_script()

    def _p(self):
        clip = object()
        return types.SimpleNamespace(
            sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(clip=clip)),
            extra_generation_params={},
        ), clip

    def test_every_sampling_pass_of_our_generation_resyncs(self):
        script = self.module.Anima38Script()
        script._runtime = _FakeRuntime(v2=True)
        p, clip = self._p()
        script.process_batch(p, {})
        script.process_before_every_sampling(p, x=None)   # 1차
        script.process_before_every_sampling(p, x=None)   # 하이레스
        self.assertEqual(script._runtime.synced, [clip, clip])
        inner, inner_clip = self._p()
        inner._sam3_outer = p
        script.process_batch(inner, {})
        script.process_before_every_sampling(inner, x=None)
        self.assertEqual(script._runtime.synced[-1], inner_clip)

    def test_other_generations_are_left_alone(self):
        script = self.module.Anima38Script()
        script._runtime = _FakeRuntime(v2=True)
        p, _ = self._p()
        script.process_before_every_sampling(p, x=None)
        self.assertEqual(getattr(script._runtime, "synced", []), [])


class _MixedDtypeAdapter(torch.nn.Module):
    """Forge 의 TE llm_adapter: 임베딩은 fp32 로 남고 나머지는 저장 dtype(bf16)."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(4, 8)
        self.proj = torch.nn.Linear(8, 8)


class PrivateAdapterDtypeTests(unittest.TestCase):
    def test_private_copy_matches_the_te_dtypes_per_parameter(self):
        module = _load_lifecycle_runtime()
        te = _MixedDtypeAdapter()
        te.proj.to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "bundle.safetensors")
            save_file({
                "net.llm_adapter.embed.weight": torch.ones(4, 8, dtype=torch.bfloat16),
                "net.llm_adapter.proj.weight": torch.ones(8, 8, dtype=torch.bfloat16),
                "net.llm_adapter.proj.bias": torch.ones(8, dtype=torch.bfloat16),
            }, path)
            with mock.patch.object(module, "using_forge_operations", lambda **_: nullcontext()):
                private = module.Anima3BRuntime._private_native_adapter(te, path, te.embed.weight.dtype)
        self.assertEqual(private.embed.weight.dtype, torch.float32)
        self.assertEqual(private.proj.weight.dtype, torch.bfloat16, "통째로 fp32 로 만들면 RAM·VRAM 이 약 200 MB 더 든다")
        self.assertEqual(private.proj.bias.dtype, torch.bfloat16)


class SharedRuntimeTests(unittest.TestCase):
    def test_shared_runtime_is_one_instance_per_process(self):
        module = _load_lifecycle_runtime()
        self.assertIs(module.shared_runtime(), module.shared_runtime())

    def test_script_uses_the_shared_runtime(self):
        sentinel = object()
        fake = types.ModuleType("sam3ext.anima38.runtime")
        fake.shared_runtime = lambda: sentinel
        script_module = _load_script()
        with mock.patch.dict(sys.modules, {"sam3ext.anima38.runtime": fake}):
            first = script_module.Anima38Script()._get_runtime()
            second = script_module.Anima38Script()._get_runtime()
        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)


if __name__ == "__main__":
    unittest.main()
