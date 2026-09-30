"""Anima 3.8B (Qwen3.5 / v2) 편입 — 번들 판별·경로·스크립트 인자·실패 시 순정 유지."""
from __future__ import annotations

import gc
import importlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import types
import unittest
import weakref
from contextlib import ExitStack, nullcontext
from functools import wraps
from pathlib import Path
from unittest import mock
from uuid import uuid4

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
            # 목록에서 떼기만 한다 — 커넥터 언로드·재적재 왕복을 하지 않는다(퇴출은 Forge 가, 모델 교체는 on_model_loaded 가)
            unload.assert_not_called()
        self.assertIsNone(script._installed_for)
        self.assertIsNone(runtime._v2_sampling_patcher)
        self.assertIs(runtime._v2_connector_patcher, installed_patcher)
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
            unload.assert_not_called()
        self.assertIs(runtime._v2_connector_patcher, installed_patcher, "다음 설치가 그대로 다시 단다")
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
            self.assertEqual(unload.call_count, 0, "목록에서 떼기만 — 패처는 번들 캐시와 함께 남는다")
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


_REAL_RELEASE = []


def _real_release_foreign_install():
    """실제 Anima3BRuntime.release_foreign_install (한 번만 로드)."""
    if not _REAL_RELEASE:
        _REAL_RELEASE.append(_load_lifecycle_runtime().Anima3BRuntime.release_foreign_install)
    return _REAL_RELEASE[0]


class ReleaseForeignInstallTests(unittest.TestCase):
    """설치 순서의 첫 단계 — 스크립트(_release_stale_install)와 Feature 6 IP-Adapter 가 같이 쓰는 공용 함수."""

    def setUp(self):
        module = _load_lifecycle_runtime()
        self.runtime = module.Anima3BRuntime()
        self.restored = []
        self.runtime.restore = self.restored.append

    def test_another_generations_install_is_restored_and_returned(self):
        stale, current = object(), object()
        self.runtime._installed_processing = stale
        self.assertIs(self.runtime.release_foreign_install(current), stale)
        self.assertEqual(self.restored, [stale])

    def test_nothing_installed_or_our_own_install_is_left_alone(self):
        current = object()
        self.assertIsNone(self.runtime.release_foreign_install(current))
        self.runtime._installed_processing = current
        self.assertIsNone(self.runtime.release_foreign_install(current))
        self.assertEqual(self.restored, [])

    def test_a_failing_restore_goes_to_on_error_or_raises(self):
        stale = object()
        self.runtime._installed_processing = stale

        def boom(p):
            raise RuntimeError("boom")

        self.runtime.restore = boom
        errors = []
        self.assertIs(self.runtime.release_foreign_install(object(), on_error=errors.append), stale)
        self.assertEqual([str(exc) for exc in errors], ["boom"])
        with self.assertRaises(RuntimeError):
            self.runtime.release_foreign_install(object())

    def test_the_script_uses_the_shared_function(self):
        script = _load_script().Anima38Script()
        stale, current = types.SimpleNamespace(sd_model=object()), types.SimpleNamespace(sd_model=object())
        self.runtime._installed_processing = stale
        script._runtime = self.runtime
        script._installed_for = stale
        with mock.patch.object(self.runtime, "release_foreign_install",
                               wraps=self.runtime.release_foreign_install) as shared:
            script._release_stale_install(current)
        shared.assert_called_once()
        self.assertIs(shared.call_args.args[0], current)
        self.assertEqual(self.restored, [stale])
        self.assertIsNone(script._installed_for, "방금 내린 설치를 _safe_restore 가 한 번 더 내리지 않게")

    def test_the_script_logs_a_failing_stale_restore_and_goes_on(self):
        script_module = _load_script()
        script = script_module.Anima38Script()
        stale = types.SimpleNamespace(sd_model=object())
        self.runtime._installed_processing = stale

        def boom(p):
            raise RuntimeError("boom")

        self.runtime.restore = boom
        script._runtime = self.runtime
        script._installed_for = stale
        with mock.patch.object(script_module, "_log") as log:
            script._release_stale_install(types.SimpleNamespace(sd_model=object()))
        self.assertIn("stale restore failed", log.call_args.args[0])
        self.assertIsNone(script._installed_for)


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

    def release_foreign_install(self, p, on_error=None):
        # 가짜에 로직을 다시 쓰지 않는다 — 실제 런타임의 공용 함수를 이 가짜 위에서 그대로 돌린다.
        return _real_release_foreign_install()(self, p, on_error=on_error)

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


class _UnetSlot:
    """약한 참조가 되는 UNet 자리 대역(_attach_sampling_patcher 가 weakref 로 기록한다)."""

    def __init__(self, **values):
        self.__dict__.update(values)


class StaleIpaShellTests(unittest.TestCase):
    """지난 IP-Adapter 잡의 껍데기 DiT 가 Forge 객체 패치로 ``model.diffusion_model`` 에 남아 있어도(클론 전환은
    detach(unpatch_all=False) 라 되돌리지 않는다) v2 래퍼는 ``object_patches_backup`` 의 진짜 DiT 에 걸린다.
    껍데기에 걸면 다음 샘플링의 진짜 DiT 에 커넥터가 없어 조건 확장이 조용히 빠진다."""

    setUp = V2ForwardTests.setUp
    _install = V2ForwardTests._install
    _encode = V2ForwardTests._encode
    _expanded = staticmethod(V2ForwardTests._expanded)

    def _leave_stale_shell(self, owner=None):
        """IP 잡 적재 뒤의 상태: 모델 속성은 껍데기, 원본은 클론끼리 공유하는 backup 에."""
        unet = self.model.forge_objects.unet
        owner = unet if owner is None else owner
        owner.object_patches_backup = {"diffusion_model": self.dit}
        self.shell = _RecordingDiT()
        unet.model.diffusion_model = self.shell

    def _assert_real_dit_expands(self):
        self.assertTrue(self.runtime._patched(self.dit.__dict__.get("forward")), "진짜 DiT 에 래퍼")
        self.assertNotIn("forward", self.shell.__dict__, "껍데기는 건드리지 않는다")
        (cond,) = self._encode()
        self.dit.forward(self.x[:1], self.t[:1], cond)
        (call,) = self.dit.calls
        torch.testing.assert_close(call.context, self._expanded())

    def test_install_wraps_the_backed_up_real_dit(self):
        self._leave_stale_shell()
        self._install()
        self._assert_real_dit_expands()

    def test_a_backup_on_another_slot_of_the_same_model_counts(self):
        unet = self.model.forge_objects.unet
        original = _UnetSlot(model=unet.model, extra_model_patchers_during_sampling=[])
        self.model.forge_objects_original = types.SimpleNamespace(unet=original)
        self._leave_stale_shell(owner=original)
        self._install()
        self._assert_real_dit_expands()

    def test_a_backup_of_another_kmodel_is_ignored(self):
        other = _UnetSlot(
            model=types.SimpleNamespace(diffusion_model=_RecordingDiT()),
            object_patches_backup={"diffusion_model": _RecordingDiT()},
            extra_model_patchers_during_sampling=[],
        )
        self.model.forge_objects_original = types.SimpleNamespace(unet=other)
        self._install()
        self.assertTrue(self.runtime._patched(self.dit.__dict__.get("forward")))
        self.assertNotIn("forward", other.object_patches_backup["diffusion_model"].__dict__)

    def test_without_an_object_patch_the_target_is_unchanged(self):
        self.model.forge_objects.unet.object_patches_backup = {}
        self._install()
        self.assertTrue(self.runtime._patched(self.dit.__dict__.get("forward")))

    def test_ensure_attached_rewraps_the_real_dit(self):
        self._install()
        del self.dit.forward   # 다른 확장이 체인을 되돌리며 우리 래퍼까지 떨군 상태
        self._leave_stale_shell()
        self.runtime.ensure_attached(self.p)
        self._assert_real_dit_expands()


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


class ConnectorPatcherReuseTests(unittest.TestCase):
    """커넥터(약 1.6 GB)는 번들 캐시(_load_v2_models)에 묶인 패처 하나로 생성마다 다시 단다. 예전엔 설치마다 새
    ModelPatcher 를 만들고 restore 에서 강제 언로드해 생성마다 D2H·H2D 왕복을 했다. Forge LoadedModel 은 패처를
    약한 참조로 쥐므로 런타임이 패처를 붙잡아야 GPU 에 남은 커넥터가 '죽은 모델'이 되지 않는다."""

    def setUp(self):
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        self.model = _LifecycleModel()
        self.created = []
        unet = self.model.forge_objects.unet

        class Patcher(types.SimpleNamespace):   # ModelPatcher 처럼 약한 참조가 되는 패처
            pass

        def add(module, **kwargs):
            patcher = Patcher(model=module)
            unet.extra_model_patchers_during_sampling.append(patcher)
            self.created.append(patcher)
            return patcher

        unet.add_extra_torch_module_during_sampling = add
        self.models = object()
        patches = (
            mock.patch.object(self.module, "bundle_metadata", return_value={"anima_v2_adapter_filename": "a"}),
            mock.patch.object(self.runtime, "_load_v2_models", side_effect=lambda *a: self.models),
            mock.patch.object(self.runtime, "_unload_patchers"),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _generate(self):
        p = _LifecycleProcessing(types.SimpleNamespace(sd_model=self.model))
        self.runtime.install(p, "a", 1.0, None)
        patcher = self.runtime._v2_sampling_patcher
        self.runtime.restore(p)
        return patcher

    def _carried(self, patcher):
        unet = self.model.forge_objects.unet
        return sum(1 for item in unet.extra_model_patchers_during_sampling if item is patcher)

    def test_same_bundle_reuses_one_patcher_and_restore_only_detaches(self):
        first = self._generate()
        self.assertEqual(self._carried(first), 0, "restore 는 목록에서 뗀다")
        second = self._generate()
        third_p = _LifecycleProcessing(types.SimpleNamespace(sd_model=self.model))
        self.runtime.install(third_p, "a", 1.0, None)
        self.assertIs(second, first)
        self.assertIs(self.runtime._v2_sampling_patcher, first)
        self.assertEqual(self._carried(first), 1, "다시 달 때 중복으로 쌓이지 않는다")
        self.assertEqual(len(self.created), 1, "ModelPatcher 는 한 번만 만든다")
        self.runtime._unload_patchers.assert_not_called()

    def test_forge_loaded_model_stays_alive_between_generations(self):
        loaded = weakref.ref(self._generate())   # Forge LoadedModel._model 과 같은 약한 참조
        self.created.clear()
        gc.collect()
        self.assertIsNotNone(loaded(), "런타임이 붙잡지 않으면 is_dead() → Forge 가 '메모리 누수'로 치운다")

    def test_other_bundle_and_model_load_unload_the_old_patcher(self):
        old = self._generate()
        self.models = object()   # 다른 번들 → _load_v2_models 가 새 커넥터를 만든다
        new = self._generate()
        self.assertIsNot(new, old)
        self.runtime._unload_patchers.assert_called_once_with(old)
        self.runtime._v2_source = None   # on_model_loaded: 다른 모델 → 커넥터 캐시를 놓는다
        self.runtime.release_stale_caches(object())
        self.assertIsNone(self.runtime._v2_connector_patcher)
        self.assertEqual(self.runtime._unload_patchers.call_args_list[-1], mock.call(new))

    def _install(self):
        p = _LifecycleProcessing(types.SimpleNamespace(sd_model=self.model))
        self.runtime.install(p, "a", 1.0, None)
        return p, self.runtime._v2_sampling_patcher

    def test_restore_unloads_an_orphaned_sampling_patcher(self):
        # 설치 도중 Forge 가 다른 모델을 로드하면(on_model_loaded) 커넥터 패처를 놓는다. 그 뒤 샘플링이 고아 패처를
        # 다시 올렸을 수 있다 — 런타임이 더는 쥐지 않으니 restore 가 예전처럼 내린다(떼기만 하면 Forge 의 죽은 모델)
        p, orphan = self._install()
        self.runtime.release_stale_caches(object())
        self.assertIsNone(self.runtime._v2_connector_patcher)
        self.runtime._unload_patchers.reset_mock()
        self.runtime.restore(p)
        self.runtime._unload_patchers.assert_called_once_with(orphan)
        self.assertEqual(self._carried(orphan), 0)
        self.assertIsNone(self.runtime._v2_sampling_patcher)

    def test_orphan_is_told_apart_from_the_reused_connector(self):
        p, orphan = self._install()
        self.runtime.release_stale_caches(object())
        reused = _ClonablePatcher(model=object())   # 그사이 새 커넥터 패처를 쥐었다 — 이건 남긴다
        self.runtime._v2_connector_patcher = reused
        self.runtime._unload_patchers.reset_mock()
        self.runtime.restore(p)
        self.runtime._unload_patchers.assert_called_once_with(orphan)
        self.assertIs(self.runtime._v2_connector_patcher, reused)

    def test_a_clone_of_the_reused_connector_is_not_an_orphan(self):
        # 같은 커넥터 모듈의 복제 패처를 내리면 Forge 는 재사용 패처까지 내린다(_same_patcher 는 복제를 같은 것으로 본다)
        p, reused = self._install()
        self.runtime._v2_sampling_patcher = _ClonablePatcher(model=reused.model)
        self.runtime.restore(p)
        self.runtime._unload_patchers.assert_not_called()
        self.assertIs(self.runtime._v2_connector_patcher, reused)


class _ClonablePatcher(types.SimpleNamespace):
    """ModelPatcher.is_clone 을 가진 패처 대역 (같은 모듈이면 복제)."""

    def is_clone(self, other):
        return self.model is getattr(other, "model", None)


class _FakeNativeEngine:
    """AnimaTextProcessingEngine 의 tokenize_line·process_tokens 대역 — 출력이 emphasis 와 TE 가중치(LoRA)에 따라 바뀐다."""

    def __init__(self, width=4):
        self.emphasis = types.SimpleNamespace(name="Original")
        self.te_weight = torch.linspace(0.5, 1.5, width)
        self.forward_calls = 0

    def tokenize_line(self, line):
        tokens = [ord(ch) % 97 + 1 for ch in line] or [0]
        boost = 1.1 if self.emphasis.name == "Original" and "(" in line else 1.0
        return [types.SimpleNamespace(
            qwen_tokens=tokens, qwen_multipliers=[1.0] * len(tokens),
            t5_tokens=tokens + [1], t5_multipliers=[boost] * len(tokens) + [1.0],
        )]

    def process_tokens(self, batch_tokens, batch_multipliers):
        self.forward_calls += 1
        tokens = torch.tensor(batch_tokens[0], dtype=torch.float32)
        return (torch.sin(tokens)[:, None] * self.te_weight[None, :]).unsqueeze(0)


def _reference_extract_prompt_features(runtime, native_engine, native_clip, prompt):
    """고치기 전 _extract_prompt_features 의 계산 그대로 — 캐시 없이 줄마다 TE·Qwen3.5 를 새로 돌린다(비교 기준)."""
    qwen, tokenizer, qwen_clip = runtime._load_qwen()
    native_adapter = native_clip.cond_stage_model.qwen3_06b.llm_adapter
    dtype = native_adapter.embed.weight.dtype
    offload_device = native_clip.patcher.offload_device
    native_rows = []
    for line in prompt:
        source, target_ids, target_weights = runtime._native_inputs(
            native_engine, str(line), native_clip.patcher.load_device, dtype,
        )
        native_rows.append((source.to(offload_device), target_ids.to(offload_device), target_weights.to(offload_device)))
    lines = [str(line) for line in prompt]
    rows = {}
    for line in dict.fromkeys(lines):
        semantic, mask = runtime._semantic_layers(qwen, tokenizer, line, qwen_clip.patcher.load_device)
        rows[line] = ([state.to(offload_device, dtype=dtype) for state in semantic], mask.to(offload_device))
    return native_adapter, native_rows, [rows[line] for line in lines]


def _semantic_for(model, tokenizer, line, device):
    return [torch.full((1, 3, 4), float(len(line) + index)) for index in range(4)], torch.ones(1, 3)


class NativeRowCacheTests(unittest.TestCase):
    """0.6B TE 줄 캐시 — 같은 줄·같은 TE 상태면 TE 를 GPU 로 올리지 않는다(예전엔 생성마다 1.75 GB 왕복).
    결과는 캐시 없이 매번 돌린 옛 계산과 비트 단위로 같아야 하고, LoRA·emphasis·DoRA 방식이 바뀌면 다시 돈다."""

    engine_factory = _FakeNativeEngine   # 새 Forge 엔진(Qwen06Engine) 대역으로 같은 검사를 다시 돌리는 하위 클래스가 바꾼다

    def setUp(self):
        self.module = _load_lifecycle_runtime()
        self.load_gpu = mock.Mock()
        self.module.memory_management.load_model_gpu = self.load_gpu
        self.engine = self.engine_factory()
        adapter = types.SimpleNamespace(embed=types.SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32)))
        self.clip = types.SimpleNamespace(
            cond_stage_model=types.SimpleNamespace(qwen3_06b=types.SimpleNamespace(llm_adapter=adapter)),
            patcher=types.SimpleNamespace(load_device="cpu", offload_device="cpu", patches_uuid=uuid4()),
        )
        self.qwen_clip = types.SimpleNamespace(patcher=types.SimpleNamespace(load_device="cpu"))
        self.sd_model = types.SimpleNamespace(current_lora_hash="[]")
        self.runtime = self._runtime()

    def _runtime(self):
        runtime = self.module.Anima3BRuntime()
        runtime._qwen_path = "qwen35_4b.safetensors"
        for name, value in (
            ("_load_qwen", mock.Mock(return_value=("qwen", "tokenizer", self.qwen_clip))),
            ("_semantic_layers", _semantic_for),
        ):
            patcher = mock.patch.object(runtime, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return runtime

    def _te_loads(self):
        return sum(1 for call in self.load_gpu.call_args_list if call.args[0] is self.clip.patcher)

    def _assert_same_as_old(self, prompt, te_loads):
        new = self.runtime._extract_prompt_features(self.engine, self.clip, prompt, sd_model=self.sd_model)
        old = _reference_extract_prompt_features(self.runtime, self.engine, self.clip, prompt)
        self.assertEqual(len(new[1]), len(old[1]))
        for new_row, old_row in zip(new[1], old[1]):
            for new_tensor, old_tensor in zip(new_row, old_row):
                self.assertEqual(new_tensor.dtype, old_tensor.dtype)
                self.assertTrue(torch.equal(new_tensor, old_tensor))
        for (new_states, new_mask), (old_states, old_mask) in zip(new[2], old[2]):
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(new_states, old_states)))
            self.assertTrue(torch.equal(new_mask, old_mask))
        self.assertEqual(self._te_loads(), te_loads)
        return new

    def _lora(self, scale):
        """Forge load_networks: LoRA 세트가 바뀌면 원본 TE 패처를 복제해 add_patches — 가중치와 uuid 가 바뀐다."""
        self.engine.te_weight = self.engine.te_weight * scale
        self.clip.patcher.patches_uuid = uuid4()

    def test_same_lines_skip_the_te_and_stay_bit_identical(self):
        prompt = ["girl, (smile:1.2)", "cat", "girl, (smile:1.2)"]
        first = self._assert_same_as_old(prompt, te_loads=1)
        self.assertEqual(self.engine.forward_calls, 2 + 3, "중복 줄은 한 번(새) + 줄마다(옛 기준)")
        second = self._assert_same_as_old(prompt, te_loads=1)
        self.assertIs(second[1][0][0], first[1][0][0], "두 번째는 TE 를 올리지 않고 캐시를 쓴다")
        self._assert_same_as_old(["cat", "dog"], te_loads=2)   # 새 줄이 있을 때만 TE 를 올린다

    def test_lora_emphasis_and_dora_changes_recompute(self):
        prompt = ["girl, (smile:1.2)"]
        self._assert_same_as_old(prompt, te_loads=1)
        self._lora(1.5)
        self._assert_same_as_old(prompt, te_loads=2)
        self._assert_same_as_old(prompt, te_loads=2)
        self.engine.emphasis = types.SimpleNamespace(name="None")
        self._assert_same_as_old(prompt, te_loads=3)
        # DoRA 추론 방식 표시만 바뀌어도(가중치가 달라졌다고 본다) 다른 키 — uuid 가 그대로인 최악의 경우
        from sam3ext import dora_infer_mode as dim
        setattr(self.sd_model, dim.MERGED_STATE_ATTR, (dim.MODE_LYCORIS, "keep"))
        self.engine.te_weight = self.engine.te_weight + 0.25
        self._assert_same_as_old(prompt, te_loads=4)
        self._assert_same_as_old(prompt, te_loads=4)

    def test_dora_invalidation_signal_drops_the_cache(self):
        from sam3ext import dora_infer_mode as dim
        prompt = ["girl"]
        self._assert_same_as_old(prompt, te_loads=1)
        # DoRA 추론 방식이 바뀌면 dora_infer_mode 가 current_lora_hash=None 과 조건 캐시 비우기로 알린다
        self.assertTrue(dim.sync_merged_state(None, self.sd_model, (dim.MODE_FORGE_FP32, "keep")))
        self.assertIsNone(self.sd_model.current_lora_hash)
        self.engine.te_weight = self.engine.te_weight * 0.5   # 다시 합친 가중치(uuid 는 아직 그대로라고 가정)
        self._assert_same_as_old(prompt, te_loads=2)
        self.assertEqual(len(self.runtime._native_cache), 0, "무효화 신호를 보면 캐시를 비우고 저장하지 않는다")
        self.sd_model.current_lora_hash = "[]"   # Forge load_networks 가 다시 합쳤다
        self._lora(1.0)
        self._assert_same_as_old(prompt, te_loads=3)
        self._assert_same_as_old(prompt, te_loads=3)

    def test_patcher_without_uuid_and_model_switch_do_not_reuse(self):
        self.clip.patcher.patches_uuid = None
        self._assert_same_as_old(["girl"], te_loads=1)
        self._assert_same_as_old(["girl"], te_loads=2)
        self.clip.patcher.patches_uuid = uuid4()
        self._assert_same_as_old(["girl"], te_loads=3)
        self.runtime.release_stale_caches(object())   # 다른 체크포인트
        self.assertEqual(len(self.runtime._native_cache), 0)

    def test_te_dtype_change_recomputes(self):
        # 키의 dtype 필드 — 같은 TE 패처(uuid)·엔진이라도 llm_adapter dtype 이 바뀌면 줄 dtype 이 달라진다
        prompt = ["girl, (smile:1.2)"]
        self._assert_same_as_old(prompt, te_loads=1)
        adapter = self.clip.cond_stage_model.qwen3_06b.llm_adapter
        adapter.embed.weight = torch.zeros(1, dtype=torch.float16)
        rows = self._assert_same_as_old(prompt, te_loads=2)
        self.assertEqual(rows[1][0][0].dtype, torch.float16)
        self._assert_same_as_old(prompt, te_loads=2)

    def test_other_engine_recomputes(self):
        # 키의 엔진 id 필드 — 같은 TE 패처(uuid)·emphasis 라도 다른 엔진이면 다른 줄 (값이 다른 엔진으로 확인)
        prompt = ["girl, (smile:1.2)"]
        self._assert_same_as_old(prompt, te_loads=1)
        other = self.engine_factory()
        other.te_weight = self.engine.te_weight * 2.0
        self.engine = other
        self._assert_same_as_old(prompt, te_loads=2)
        self._assert_same_as_old(prompt, te_loads=2)

    def test_cache_is_bounded(self):
        for index in range(self.module.NATIVE_CACHE_LINES + 5):
            self.runtime._extract_prompt_features(self.engine, self.clip, [f"line {index}"], sd_model=self.sd_model)
        self.assertEqual(len(self.runtime._native_cache), self.module.NATIVE_CACHE_LINES)

    def test_encode_v2_placeholders_and_runs_match_the_old_path(self):
        self.engine = self.engine_factory(width=_V2_WIDTH)   # 마커가 들어갈 폭
        old_runtime = self._runtime()
        reference = lambda engine, clip, prompt, sd_model=None: _reference_extract_prompt_features(  # noqa: E731
            old_runtime, engine, clip, prompt)
        prompt = ["girl, (smile:1.2)", "girl, (smile:1.2)", "cat"]
        with mock.patch.object(old_runtime, "_extract_prompt_features", side_effect=reference):
            for _ in range(2):   # 두 번째는 새 경로가 캐시로 답한다
                new = self.runtime._encode_v2(self.engine, self.clip, prompt, sd_model=self.sd_model)
                old = old_runtime._encode_v2(self.engine, self.clip, prompt)
                self.assertEqual(len(new), len(old))
                for new_cond, old_cond in zip(new, old):
                    self.assertTrue(torch.equal(new_cond, old_cond))
                    run_id = int(anima_marker.read_run_ids(new_cond)[0])
                    new_run, old_run = self.runtime._v2_runs[run_id], old_runtime._v2_runs[run_id]
                    for field in ("source", "target_ids", "target_weights", "semantic_mask"):
                        self.assertTrue(torch.equal(getattr(new_run, field), getattr(old_run, field)), field)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(new_run.semantic, old_run.semantic)))
        self.assertEqual(self._te_loads(), 1)


class _MemPatcher:
    def __init__(self, size, loaded=0.0, device="cuda:0"):
        self.load_device = device
        self._size, self._loaded = size, loaded

    def model_size(self):
        return self._size

    def loaded_size(self):
        return self._loaded


class _MemUnet(_MemPatcher):
    def __init__(self, size, extras=()):
        super().__init__(size)
        self.extra_model_patchers_during_sampling = list(extras)
        self.extra_preserved_memory_during_sampling = 0
        self.shapes = []

    def memory_required(self, shape):
        self.shapes.append(list(shape))
        return 3.0


class EncoderResidencyTests(unittest.TestCase):
    """TE·Qwen3.5 는 여유 VRAM 이 다음 샘플링(Forge load_models_gpu 가 비우려는 양)보다 넉넉할 때만 Forge
    LoadedModel 로 남긴다. 모자라거나 셀 수 없으면 예전처럼 바로 내린다 — unpatch_weights=False 로 목록에서만
    빼는 방식(Forge 계산 밖 VRAM)은 쓰지 않는다."""

    def setUp(self):
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        mm = self.module.memory_management
        self.free = 0.0
        mm.get_free_memory = lambda device=None: self.free
        mm.minimum_inference_memory = lambda: 2.0
        mm.extra_reserved_memory = lambda: 1.0
        self.unet = _MemUnet(10.0, extras=[_MemPatcher(2.0)])   # DiT 미적재 10 + 커넥터 2
        self.runtime._installed_processing = types.SimpleNamespace(
            sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=self.unet)),
            width=1024, height=1024, batch_size=1, enable_hr=True, hr_upscale_to_x=1536, hr_upscale_to_y=1536,
        )
        self.te = _MemPatcher(1.75, loaded=1.75)
        self.qwen = _MemPatcher(5.0)
        unload = mock.patch.object(self.runtime, "_unload_patchers")
        self.unload = unload.start()
        self.addCleanup(unload.stop)

    def test_kept_only_when_sampling_headroom_is_free(self):
        # 12 × 1.1 + max(2, 3 + 0 + 1) = 17.2
        self.free = 17.3
        self.runtime._release_encoder(self.te)
        self.unload.assert_not_called()
        self.assertEqual(self.unet.shapes, [[2, 16, 192, 192]], "하이레스면 큰 쪽 해상도로 잰다")
        self.free = 17.1
        self.runtime._release_encoder(self.te)
        self.unload.assert_called_once_with(self.te)

    def test_upcoming_qwen_needs_room_too(self):
        self.free = 17.3
        self.runtime._release_encoder(self.te, upcoming=(self.qwen,))
        self.unload.assert_called_once_with(self.te)
        self.unload.reset_mock()
        self.free = 22.8   # 17.2 + 5 × 1.1
        self.runtime._release_encoder(self.te, upcoming=(self.qwen,))
        self.unload.assert_not_called()

    def test_loaded_models_and_other_devices_need_no_room(self):
        self.unet._loaded = 10.0
        self.unet.extra_model_patchers_during_sampling = [_MemPatcher(2.0, device="cpu")]
        self.free = 4.0   # 추론 몫만
        self.runtime._release_encoder(self.te)
        self.unload.assert_not_called()

    def test_unknown_state_falls_back_to_unloading(self):
        self.free = 1e12
        self.runtime._installed_processing = None
        self.runtime._release_encoder(self.te)
        self.unload.assert_called_once_with(self.te)
        self.unload.reset_mock()
        self.runtime._installed_processing = types.SimpleNamespace(
            sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=self.unet)), width=64, height=64)
        del self.module.memory_management.get_free_memory   # 옛 Forge·스텁
        self.runtime._release_encoder(self.te)
        self.unload.assert_called_once_with(self.te)

    def _features_setup(self):
        load_gpu = mock.Mock()
        self.module.memory_management.load_model_gpu = load_gpu
        adapter = types.SimpleNamespace(embed=types.SimpleNamespace(weight=torch.zeros(1)))
        te = _MemPatcher(1.75, loaded=1.75, device="cpu")
        te.offload_device, te.patches_uuid = "cpu", uuid4()
        clip = types.SimpleNamespace(
            cond_stage_model=types.SimpleNamespace(qwen3_06b=types.SimpleNamespace(llm_adapter=adapter)), patcher=te)
        qwen_clip = types.SimpleNamespace(patcher=_MemPatcher(5.0, device="cpu"))
        self.runtime._qwen_path = "qwen35_4b.safetensors"
        for name, value in (
            ("_load_qwen", mock.Mock(return_value=("qwen", "tokenizer", qwen_clip))),
            ("_semantic_layers", _semantic_for),
        ):
            patcher = mock.patch.object(self.runtime, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return load_gpu, clip, qwen_clip

    def test_repeat_generation_moves_nothing(self):
        load_gpu, clip, qwen_clip = self._features_setup()
        self.free = 1e12
        engine = _FakeNativeEngine()
        self.runtime._extract_prompt_features(engine, clip, ["girl"], sd_model=types.SimpleNamespace(current_lora_hash="[]"))
        self.assertEqual([call.args[0] for call in load_gpu.call_args_list], [clip.patcher, qwen_clip.patcher])
        self.unload.assert_not_called()
        self.runtime._extract_prompt_features(engine, clip, ["girl"], sd_model=types.SimpleNamespace(current_lora_hash="[]"))
        self.assertEqual(load_gpu.call_count, 2, "같은 프롬프트 — TE·Qwen3.5 어느 쪽도 GPU 로 옮기지 않는다")

    def test_setting_off_unloads_even_with_room(self):
        self.free = 1e12
        with mock.patch.object(self.module, "keep_resident", return_value=False):
            self.runtime._release_encoder(self.te)
        self.unload.assert_called_once_with(self.te)
        self.unload.reset_mock()
        with mock.patch.object(self.module, "keep_resident", return_value=True):
            self.runtime._release_encoder(self.te)
        self.unload.assert_not_called()

    def test_setting_off_unloads_te_and_qwen_right_after_encoding_with_the_same_rows(self):
        load_gpu, clip, qwen_clip = self._features_setup()
        self.free = 1e12
        engine = _FakeNativeEngine()
        sd_model = types.SimpleNamespace(current_lora_hash="[]")
        with mock.patch.object(self.module, "keep_resident", return_value=False):
            new = self.runtime._extract_prompt_features(engine, clip, ["girl"], sd_model=sd_model)
        self.assertEqual(self.unload.call_args_list, [mock.call(clip.patcher), mock.call(qwen_clip.patcher)])
        old = _reference_extract_prompt_features(self.runtime, engine, clip, ["girl"])
        for new_tensor, old_tensor in zip(new[1][0], old[1][0]):
            self.assertTrue(torch.equal(new_tensor, old_tensor))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(new[2][0][0], old[2][0][0])))

    def test_failed_encoding_still_unloads(self):
        _, clip, _ = self._features_setup()
        self.free = 1e12
        with mock.patch.object(self.runtime, "_native_inputs", side_effect=RuntimeError("CUDA out of memory")):
            with self.assertRaises(RuntimeError):
                self.runtime._extract_prompt_features(_FakeNativeEngine(), clip, ["girl"])
        self.unload.assert_called_once_with(clip.patcher)


class _WeakAdapter:
    """llm_adapter 대역 — weakref 가 되는 객체(_v2_source 는 약한 참조)."""

    def __init__(self):
        self.embed = types.SimpleNamespace(weight=torch.zeros(1))


class SharedAdapterFallbackTests(unittest.TestCase):
    """번들에 llm_adapter 원본이 없어 커넥터가 TE 모듈을 같이 쓰는 폴백. 두 패처가 같은 가중치를 제자리 패치하므로
    TE 가 상주하면 커넥터가 TE LoRA 의 llm_adapter 몫이 합쳐진 가중치를 본다(여유 VRAM 에 따라 결과가 바뀜).
    이 경우엔 예전처럼 TE 를 인코딩마다 내리고, 커넥터도 restore 에서 내린다."""

    def setUp(self):
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        mm = self.module.memory_management
        mm.get_free_memory = lambda device=None: 1e12   # 여유는 넉넉 — 공유가 아니면 TE 를 남길 상황
        mm.minimum_inference_memory = lambda: 2.0
        mm.extra_reserved_memory = lambda: 1.0
        self.load_gpu = mock.Mock()
        mm.load_model_gpu = self.load_gpu
        unet = _MemUnet(10.0)
        self.runtime._installed_processing = types.SimpleNamespace(
            sd_model=types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=unet)),
            width=1024, height=1024, batch_size=1,
        )
        self.adapter = _WeakAdapter()
        te = _MemPatcher(1.75, loaded=1.75, device="cpu")
        te.offload_device, te.patches_uuid = "cpu", uuid4()
        self.clip = types.SimpleNamespace(
            cond_stage_model=types.SimpleNamespace(qwen3_06b=types.SimpleNamespace(llm_adapter=self.adapter)),
            patcher=te,
        )
        qwen_clip = types.SimpleNamespace(patcher=_MemPatcher(5.0, device="cpu"))
        self.runtime._qwen_path = "qwen35_4b.safetensors"
        for name, value in (
            ("_load_qwen", mock.Mock(return_value=("qwen", "tokenizer", qwen_clip))),
            ("_semantic_layers", _semantic_for),
            ("_unload_patchers", mock.Mock()),
        ):
            patcher = mock.patch.object(self.runtime, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.unload = self.runtime._unload_patchers
        self.engine = _FakeNativeEngine()   # 캐시 키에 엔진 id 가 든다 — 한 엔진으로
        self.sd_model = types.SimpleNamespace(current_lora_hash="[]")

    def _te_unloads(self):
        return sum(1 for call in self.unload.call_args_list if call.args == (self.clip.patcher,))

    def _extract(self):
        return self.runtime._extract_prompt_features(self.engine, self.clip, ["girl"], sd_model=self.sd_model)

    def test_private_copy_keeps_the_te_resident(self):
        self.runtime._v2_models = types.SimpleNamespace(native_adapter=_WeakAdapter())   # 번들 원본 사본
        self._extract()
        self._extract()
        self.assertEqual(self._te_unloads(), 0)
        self.assertFalse(self.runtime._connector_shares_te(self.clip))

    def test_shared_module_unloads_the_te_after_every_encoding_even_from_cache(self):
        self.runtime._v2_models = types.SimpleNamespace(native_adapter=self.adapter)
        self.assertTrue(self.runtime._connector_shares_te(self.clip))
        first = self._extract()
        self.assertEqual(self._te_unloads(), 1, "여유 VRAM 이 넉넉해도 예전처럼 내린다")
        # 두 번째는 줄 캐시로 TE 를 올리지 않지만, 지난 비 v2 생성에서 남은 TE 가 있을 수 있어 내린다
        second = self._extract()
        self.assertEqual(self._te_unloads(), 2)
        te_loads = sum(1 for call in self.load_gpu.call_args_list if call.args[0] is self.clip.patcher)
        self.assertEqual(te_loads, 1, "줄 캐시는 공유 폴백에서도 그대로 쓴다")
        for new_tensor, old_tensor in zip(second[1][0], first[1][0]):
            self.assertTrue(torch.equal(new_tensor, old_tensor))

    def test_release_native_routes_by_sharing(self):
        self.runtime._v2_models = types.SimpleNamespace(native_adapter=self.adapter)
        with mock.patch.object(self.runtime, "_release_encoder") as release:
            self.runtime._release_native(self.clip)
            release.assert_not_called()
            self.assertEqual(self._te_unloads(), 1)
            self.runtime._v2_models = None   # v2 번들을 쓴 적 없음 — 여유 기준
            self.runtime._release_native(self.clip, upcoming=("qwen",))
            release.assert_called_once_with(self.clip.patcher, upcoming=("qwen",))
            self.assertEqual(self._te_unloads(), 1)


class SharedConnectorRestoreTests(unittest.TestCase):
    """공유 폴백이면 restore 가 커넥터 패처를 예전처럼 내린다(다음 인코딩의 TE 적재·언로드가 공유 모듈을 옮겨
    '적재됨' 커넥터와 장치가 어긋나지 않게). 패처 객체는 계속 재사용한다."""

    def setUp(self):
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        self.model = _LifecycleModel()
        self.created = []
        unet = self.model.forge_objects.unet

        class Patcher(types.SimpleNamespace):
            pass

        def add(module, **kwargs):
            patcher = Patcher(model=module)
            unet.extra_model_patchers_during_sampling.append(patcher)
            self.created.append(patcher)
            return patcher

        unet.add_extra_torch_module_during_sampling = add
        self.te_adapter = _WeakAdapter()
        patches = (
            mock.patch.object(self.module, "bundle_metadata", return_value={"anima_v2_adapter_filename": "a"}),
            mock.patch.object(self.runtime, "_load_v2_models", side_effect=lambda *a: self.models),
            mock.patch.object(self.runtime, "_unload_patchers"),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _generate(self):
        p = _LifecycleProcessing(types.SimpleNamespace(sd_model=self.model))
        self.runtime.install(p, "a", 1.0, None)
        self.runtime._v2_source = weakref.ref(self.te_adapter)   # _load_v2_models 가 기록하는 원본 TE 모듈
        patcher = self.runtime._v2_sampling_patcher
        self.runtime.restore(p)
        return patcher

    def test_shared_connector_is_unloaded_at_restore_but_reused(self):
        self.models = types.SimpleNamespace(native_adapter=self.te_adapter)
        first = self._generate()
        self.runtime._unload_patchers.assert_called_once_with(first)
        second = self._generate()
        self.assertIs(second, first)
        self.assertEqual(len(self.created), 1)
        self.assertEqual(self.runtime._unload_patchers.call_count, 2)

    def test_private_connector_stays_resident(self):
        self.models = types.SimpleNamespace(native_adapter=_WeakAdapter())
        self._generate()
        self._generate()
        self.runtime._unload_patchers.assert_not_called()


def _reference_expand_v2_context(runtime, context, timesteps, run_ids):
    """고치기 전 _expand_v2_context 그대로 — 스텝마다 run 텐서를 다시 .to() 하고 run 마다 unique·nonzero·tolist 로
    동기화한다(비교 기준). 새 구현과 같은 run·커넥터를 읽는다."""
    connector = runtime._v2_models.connector
    flat_ids = run_ids.reshape(-1).to(dtype=torch.long)
    if flat_ids.numel() != context.shape[0]:
        repeats = (context.shape[0] + flat_ids.numel() - 1) // flat_ids.numel()
        flat_ids = flat_ids.repeat(repeats)[: context.shape[0]]
    timestep_rows = timesteps.reshape(-1)
    if timestep_rows.numel() != context.shape[0]:
        repeats = (context.shape[0] + timestep_rows.numel() - 1) // timestep_rows.numel()
        timestep_rows = timestep_rows.repeat(repeats)[: context.shape[0]]
    outputs = [None] * context.shape[0]
    dtype = runtime._v2_models.native_adapter.embed.weight.dtype
    for run_id in flat_ids.unique().tolist():
        if int(run_id) < 0:
            for row_index in (flat_ids == run_id).nonzero(as_tuple=False).reshape(-1).tolist():
                outputs[row_index] = context[row_index : row_index + 1]
            continue
        run = runtime._v2_runs.get(int(run_id))
        if run is None:
            raise RuntimeError(f"Anima v2 conditioning run {run_id} is unavailable; re-encode the prompt.")
        indices = (flat_ids == run_id).nonzero(as_tuple=False).reshape(-1)
        count = indices.numel()
        source = run.source.to(context.device, dtype=dtype).expand(count, -1, -1)
        target_ids = run.target_ids.to(context.device).expand(count, -1)
        semantic = [state.to(context.device, dtype=dtype).expand(count, -1, -1) for state in run.semantic]
        semantic_mask = run.semantic_mask.to(context.device).expand(count, -1)
        expanded = connector(
            source, target_ids, semantic,
            semantic_source_mask=semantic_mask,
            timesteps=timestep_rows[indices].to(context.device),
        )
        weights = run.target_weights.to(context.device, dtype=expanded.dtype).expand(count, -1, -1)
        expanded = expanded * weights[:, : expanded.shape[1]]
        if expanded.shape[1] < 512:
            expanded = torch.nn.functional.pad(expanded, (0, 0, 0, 512 - expanded.shape[1]))
        for output_index, row_index in enumerate(indices.tolist()):
            outputs[row_index] = expanded[output_index : output_index + 1]
    return torch.cat(outputs).to(dtype=context.dtype)


def _v2_rich_features(count, semantic_dtype=torch.float32, seed=0):
    """줄마다 값이 다른 _extract_prompt_features 대역 — 커넥터 입력이 하나라도 어긋나면 출력이 달라지게."""
    generator = torch.Generator().manual_seed(seed)
    native_rows, semantic_rows = [], []
    for index in range(count):
        native_rows.append((
            torch.randn(1, _V2_TOKENS, _V2_WIDTH, generator=generator),
            torch.randint(0, 50, (1, _V2_TOKENS), generator=generator),
            torch.rand(1, _V2_TOKENS, 1, generator=generator) + 0.5,
        ))
        length = 3 + index % 4
        semantic_rows.append((
            [torch.randn(1, length, 16, generator=generator).to(semantic_dtype) for _ in range(4)],
            torch.ones(1, length, dtype=torch.long),
        ))
    return object(), native_rows, semantic_rows


def _mixing_connector(calls):
    """모든 입력(source·T5 id·의미 특징 4층·mask·timestep)이 출력에 섞이는 커넥터 대역. 받은 입력을 남긴다."""

    def connector(source, target_ids, semantic, semantic_source_mask=None, timesteps=None):
        calls.append(types.SimpleNamespace(source=source, target_ids=target_ids, semantic=list(semantic),
                                           mask=semantic_source_mask, timesteps=timesteps))
        mixed = sum((index + 1) * state.sum(dim=(1, 2)) for index, state in enumerate(semantic))
        extra = mixed + semantic_source_mask.sum(dim=1).to(source.dtype) + timesteps.to(source.dtype) * 0.125
        ids = target_ids[:, :, None].to(source.dtype)   # 출력 길이는 실제 커넥터처럼 T5 토큰 수
        return source.sum(dim=1, keepdim=True) * 1.5 + ids * 0.25 + extra[:, None, None]

    return connector


class V2StepCacheTests(unittest.TestCase):
    """스텝마다 부르는 커넥터 경로 — run 텐서는 샘플링 장치에 한 번만 올리고, run 묶기는 forward 한 번에 동기화 한 번.
    결과는 고치기 전 구현(_reference_expand_v2_context)과 비트 단위로 같아야 한다."""

    def setUp(self):
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        self.calls = []
        self._models(torch.float32)
        self.model = _LifecycleModel()
        self.dit = _RecordingDiT()
        self.model.forge_objects.unet.model.diffusion_model = self.dit
        self.p = _LifecycleProcessing(types.SimpleNamespace(sd_model=self.model))
        self.uncond = torch.randn(1, 512, _V2_WIDTH, generator=torch.Generator().manual_seed(7))

    def _models(self, dtype):
        """커넥터 dtype(= llm_adapter embed dtype). float64 면 .to() 가 실제 사본을 만든다 — CPU 에서 캐시가 보인다."""
        self.models = types.SimpleNamespace(
            connector=_mixing_connector(self.calls),
            native_adapter=types.SimpleNamespace(embed=types.SimpleNamespace(weight=torch.zeros(1, dtype=dtype))),
        )
        self.runtime._v2_models = self.models

    def _install(self):
        def load(sd_model, path, metadata):
            self.runtime._v2_models = self.models
            return self.models

        with mock.patch.object(self.runtime, "_load_v2_models", side_effect=load):
            self.runtime._install_v2(self.p, "bundle.safetensors", {})

    def _encode(self, count, seed=0, semantic_dtype=torch.float32):
        features = _v2_rich_features(count, semantic_dtype, seed)
        with mock.patch.object(self.runtime, "_extract_prompt_features", return_value=features):
            conds = self.runtime._encode_v2(object(), object(), ["line"] * count)
        return conds, [int(anima_marker.read_run_ids(cond)[0]) for cond in conds]

    def _assert_same(self, context, timesteps, run_ids):
        old = _reference_expand_v2_context(self.runtime, context, timesteps, run_ids)
        new = self.runtime._expand_v2_context(context, timesteps, run_ids)
        self.assertEqual(new.dtype, old.dtype)
        self.assertTrue(torch.equal(new, old))
        return new

    def test_every_row_layout_matches_the_old_implementation_step_after_step(self):
        for dtype in (torch.float32, torch.float64):
            for semantic_dtype in (torch.float32, torch.bfloat16):
                with self.subTest(dtype=dtype, semantic=semantic_dtype):
                    self.runtime = self.module.Anima3BRuntime()
                    self._models(dtype)
                    conds, (a, b, c) = self._encode(3, semantic_dtype=semantic_dtype)
                    rows = {a: conds[0], b: conds[1], c: conds[2], -1: self.uncond}
                    layouts = (
                        [a], [a, -1], [a, -1, a, -1], [a, b, -1, -1], [a, b, a, c],
                        [a, a, -1, a],   # 등간격이 아닌 행 — 인덱스 텐서로 고른다
                        [-1, -1], [c, -1, b, a], [-1, a, a, a],
                    )
                    for layout in layouts:
                        context = torch.cat([rows[run_id] for run_id in layout])
                        for start in (0.9, 0.5, 0.1):   # 같은 run 을 여러 스텝 — 두 번째부터 장치 사본으로
                            timesteps = torch.linspace(start, start + 0.3, len(layout))
                            self._assert_same(context, timesteps, torch.tensor(layout))
                    # 짧은 run id·timestep 은 배치만큼 되풀이 (CFG 의 [c, uc] → [c, uc, c, uc])
                    context = torch.cat([conds[0], self.uncond, conds[0], self.uncond])
                    self._assert_same(context, torch.tensor([10.0, 20.0]), torch.tensor([a, -1]))

    def test_forward_matches_the_old_implementation(self):
        self._install()
        conds, _ = self._encode(2, semantic_dtype=torch.bfloat16)
        for context in (
            torch.cat([conds[0], self.uncond]),
            torch.stack([conds[0], conds[1], self.uncond]),   # reconstruct_cond_batch 의 [B, 1, 512, C]
        ):
            for step in range(3):
                timesteps = torch.full((context.shape[0],), 0.8 - 0.2 * step)
                self.dit.forward(torch.zeros(context.shape[0], 4), timesteps, context)
                rows = anima_marker.as_rows(context)
                expected = _reference_expand_v2_context(
                    self.runtime, rows, timesteps, anima_marker.read_run_ids(rows)).reshape(context.shape)
                self.assertTrue(torch.equal(self.dit.calls[-1].context, expected))

    def test_run_tensors_reach_the_device_once(self):
        self._models(torch.float64)   # 실제 사본이 생기는 캐스트 — GPU 의 H2D 자리
        self._install()
        (cond,), _ = self._encode(1)
        context = torch.cat([cond, self.uncond])
        self.dit.forward(torch.zeros(2, 4), torch.tensor([0.9, 0.9]), context)
        self.dit.forward(torch.zeros(2, 4), torch.tensor([0.4, 0.4]), context)
        first, second = self.calls[-2], self.calls[-1]
        self.assertEqual(first.source.dtype, torch.float64)
        self.assertEqual(second.source.data_ptr(), first.source.data_ptr(), "둘째 스텝은 첫 스텝의 사본을 쓴다")
        for left, right in zip(first.semantic, second.semantic):
            self.assertEqual(left.dtype, torch.float64)
            self.assertEqual(right.data_ptr(), left.data_ptr())
        self.assertFalse(torch.equal(first.timesteps, second.timesteps), "timestep 은 스텝마다 새 값")

    def test_one_host_sync_per_forward(self):
        self._install()
        conds, _ = self._encode(2)
        context = torch.cat([conds[0], self.uncond, conds[1], conds[0], self.uncond])
        timesteps = torch.full((5,), 0.5)
        counts = {}

        def counting(name):
            original = getattr(torch.Tensor, name)

            def wrapper(tensor, *args, **kwargs):
                counts[name] = counts.get(name, 0) + 1
                return original(tensor, *args, **kwargs)

            return wrapper

        def run(call):
            counts.clear()
            with ExitStack() as stack:
                for name in ("tolist", "item", "__bool__", "nonzero", "unique"):
                    stack.enter_context(mock.patch.object(torch.Tensor, name, counting(name)))
                call()
            return dict(counts)

        self.assertEqual(run(lambda: self.dit.forward(torch.zeros(5, 4), timesteps, context)), {"tolist": 1})
        run_ids = anima_marker.read_run_ids(context)
        old = run(lambda: _reference_expand_v2_context(self.runtime, context, timesteps, run_ids))
        self.assertGreater(sum(old.values()), 4, "옛 구현은 run 마다 nonzero·tolist 로 동기화했다")

    def test_device_copies_are_bounded_and_rebuilt_identically(self):
        self._models(torch.float64)
        cap = self.module.V2_DEVICE_CACHE_RUNS
        conds, ids = self._encode(cap + 3)
        step = torch.tensor([0.5])
        for cond, run_id in zip(conds, ids):
            self._assert_same(cond, step, torch.tensor([run_id]))
        self.assertEqual(len(self.runtime._v2_device_runs), cap)
        self.assertTrue(all(not self.runtime._v2_runs[run_id].device_copies for run_id in ids[:3]))
        self.assertTrue(all(self.runtime._v2_runs[run_id].device_copies for run_id in ids[3:]))
        self._assert_same(conds[0], step, torch.tensor([ids[0]]))   # 쫓겨난 run 도 다시 쓰면 같은 값
        self.assertEqual(len(self.runtime._v2_device_runs), cap)

        runs = dict(self.runtime._v2_runs)
        self.runtime.release_stale_caches(object())   # on_model_loaded — 사본만 놓고 run 은 둔다(하이레스가 쓴다)
        self.assertEqual(self.runtime._v2_runs, runs)
        self.assertEqual(len(self.runtime._v2_device_runs), 0)
        self.assertTrue(all(not run.device_copies for run in runs.values()))

    def test_restore_drops_copies_and_a_reused_run_id_sees_its_own_tensors(self):
        self._models(torch.float64)
        self._install()
        (first,), (run_id,) = self._encode(1, seed=1)
        self.dit.forward(torch.zeros(1, 4), torch.tensor([0.5]), first)
        old_run = self.runtime._v2_runs[run_id]
        self.assertTrue(old_run.device_copies)
        self.runtime.restore(self.p)
        self.assertEqual(self.runtime._v2_runs, {})
        self.assertEqual(len(self.runtime._v2_device_runs), 0)
        self.assertFalse(old_run.device_copies, "restore 는 VRAM 사본도 놓는다")

        self._install()
        (second,), (again,) = self._encode(1, seed=2)
        self.assertEqual(again, run_id, "카운터가 0 부터 — 같은 id 가 다른 run")
        self.dit.forward(torch.zeros(1, 4), torch.tensor([0.5]), second)
        expected = _reference_expand_v2_context(self.runtime, second, torch.tensor([0.5]), torch.tensor([run_id]))
        self.assertTrue(torch.equal(self.dit.calls[-1].context, expected))

    def test_clear_drops_runs_and_their_copies(self):
        self._models(torch.float64)
        conds, ids = self._encode(2)
        self._assert_same(conds[0], torch.tensor([0.5]), torch.tensor([ids[0]]))
        run = self.runtime._v2_runs[ids[0]]
        self.runtime._clear_v2_runs()   # install·restore 가 부른다
        self.assertEqual(self.runtime._v2_runs, {})
        self.assertFalse(run.device_copies)

    def test_placeholders_live_on_the_te_device_and_runs_stay_on_the_host(self):
        clip = types.SimpleNamespace(patcher=types.SimpleNamespace(load_device="meta"))
        features = _v2_rich_features(2)
        with mock.patch.object(self.runtime, "_extract_prompt_features", return_value=features):
            conds = self.runtime._encode_v2(object(), clip, ["a", "b"])
        self.assertEqual([cond.device.type for cond in conds], ["meta", "meta"])
        for run in self.runtime._v2_runs.values():
            self.assertEqual(run.source.device.type, "cpu")

    def test_native_conditioning_keeps_the_forge_device(self):
        class MetaModel(_CondModel):
            def get_learned_conditioning(self, prompt):
                self.native_lines.append(list(prompt))
                return [torch.zeros(1, 512, _V2_WIDTH, device="meta") for _ in prompt]

        model = MetaModel()
        conds = self.runtime.encode(model, _Prompt(["bad"], negative=True), "a", 1.0, None)
        self.assertEqual([cond.device.type for cond in conds], ["meta"], "순정 Anima 처럼 TE 장치 그대로")
        self.assertEqual(model.native_lines, [["bad"]])


def _bf16_semantic_for(model, tokenizer, line, device):
    """Qwen3.5 처럼 bf16 로 계산된 4개 층."""
    generator = torch.Generator().manual_seed(len(line))
    return [torch.randn(1, 3, 4, generator=generator).to(torch.bfloat16) for _ in range(4)], torch.ones(1, 3)


class SemanticStorageTests(unittest.TestCase):
    """의미 특징 캐시 — bf16 로 계산된 층은 bf16 로 두고(RAM 절반), 쓸 때 fp32 로 올린다. bf16→fp32 는 정확하므로
    옛 경로(저장할 때 fp32)와 커넥터 입력·출력이 비트 단위로 같다. 정확하지 않은 캐스트는 예전처럼 저장할 때."""

    def setUp(self):
        self.module = _load_lifecycle_runtime()
        self.module.memory_management.load_model_gpu = mock.Mock()
        self.engine = _FakeNativeEngine(width=_V2_WIDTH)
        self.sd_model = types.SimpleNamespace(current_lora_hash="[]")
        self.qwen_clip = types.SimpleNamespace(patcher=types.SimpleNamespace(load_device="cpu"))

    def _clip(self, dtype):
        adapter = types.SimpleNamespace(embed=types.SimpleNamespace(weight=torch.zeros(1, dtype=dtype)))
        return types.SimpleNamespace(
            cond_stage_model=types.SimpleNamespace(qwen3_06b=types.SimpleNamespace(llm_adapter=adapter)),
            patcher=types.SimpleNamespace(load_device="cpu", offload_device="cpu", patches_uuid=uuid4()),
        )

    def _runtime(self, semantic=_bf16_semantic_for):
        runtime = self.module.Anima3BRuntime()
        runtime._qwen_path = "qwen35_4b.safetensors"
        for name, value in (
            ("_load_qwen", mock.Mock(return_value=("qwen", "tokenizer", self.qwen_clip))),
            ("_semantic_layers", semantic),
        ):
            patcher = mock.patch.object(runtime, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return runtime

    def test_bf16_layers_are_cached_as_bf16_and_upcast_exactly(self):
        clip = self._clip(torch.float32)
        runtime, reference = self._runtime(), self._runtime()
        prompt = ["girl, (smile:1.2)", "cat"]
        new = runtime._extract_prompt_features(self.engine, clip, prompt, sd_model=self.sd_model)
        old = _reference_extract_prompt_features(reference, self.engine, clip, prompt)
        for (new_states, new_mask), (old_states, old_mask) in zip(new[2], old[2]):
            for new_state, old_state in zip(new_states, old_states):
                self.assertEqual(new_state.dtype, torch.bfloat16)
                self.assertEqual(old_state.dtype, torch.float32)
                self.assertTrue(torch.equal(new_state.to(torch.float32), old_state))
            self.assertTrue(torch.equal(new_mask, old_mask))
        for states, _ in runtime._semantic_cache.values():
            self.assertTrue(all(state.dtype == torch.bfloat16 for state in states))

    def test_whole_v2_path_matches_the_old_path(self):
        clip = self._clip(torch.float32)
        runtime, old_runtime = self._runtime(), self._runtime()
        calls = []
        models = types.SimpleNamespace(
            connector=_mixing_connector(calls),
            native_adapter=types.SimpleNamespace(embed=types.SimpleNamespace(weight=torch.zeros(1))),
        )
        runtime._v2_models = old_runtime._v2_models = models
        reference = lambda engine, clip_, prompt, sd_model=None: _reference_extract_prompt_features(  # noqa: E731
            old_runtime, engine, clip_, prompt)
        prompt = ["girl, (smile:1.2)", "girl, (smile:1.2)", "cat"]
        uncond = torch.randn(1, 512, _V2_WIDTH, generator=torch.Generator().manual_seed(3))
        with mock.patch.object(old_runtime, "_extract_prompt_features", side_effect=reference):
            for _ in range(2):   # 두 번째 인코딩은 새 경로가 캐시(bf16)로 답한다
                new_conds = runtime._encode_v2(self.engine, clip, prompt, sd_model=self.sd_model)
                old_conds = old_runtime._encode_v2(self.engine, clip, prompt)
                for new_cond, old_cond in zip(new_conds, old_conds):
                    self.assertTrue(torch.equal(new_cond, old_cond))
                new_context = torch.cat([*new_conds, uncond])
                old_context = torch.cat([*old_conds, uncond])
                for step in range(3):
                    timesteps = torch.full((4,), 0.9 - 0.3 * step)
                    new = runtime._expand_v2_context(
                        new_context, timesteps, anima_marker.read_run_ids(new_context))
                    old = _reference_expand_v2_context(
                        old_runtime, old_context, timesteps, anima_marker.read_run_ids(old_context))
                    self.assertTrue(torch.equal(new, old))

    def test_inexact_casts_still_happen_when_stored(self):
        cases = (
            (torch.float16, _bf16_semantic_for, torch.float16),   # bf16→fp16 은 값이 바뀔 수 있다 — 예전처럼
            (torch.float32, _semantic_for, torch.float32),        # 같은 dtype
        )
        for dtype, semantic, stored in cases:
            with self.subTest(dtype=dtype):
                clip = self._clip(dtype)
                runtime, reference = self._runtime(semantic), self._runtime(semantic)
                new = runtime._extract_prompt_features(self.engine, clip, ["girl"], sd_model=self.sd_model)
                old = _reference_extract_prompt_features(reference, self.engine, clip, ["girl"])
                for new_state, old_state in zip(new[2][0][0], old[2][0][0]):
                    self.assertEqual(new_state.dtype, stored)
                    self.assertTrue(torch.equal(new_state, old_state))

    def test_storage_dtype_rule(self):
        rule = self.module._semantic_storage_dtype
        self.assertIs(rule(torch.bfloat16, torch.float32), torch.bfloat16)
        self.assertIs(rule(torch.float16, torch.float32), torch.float16)
        self.assertIs(rule(torch.float32, torch.float32), torch.float32)
        self.assertIs(rule(torch.bfloat16, torch.float16), torch.float16)
        self.assertIs(rule(torch.float16, torch.bfloat16), torch.bfloat16)
        self.assertIs(rule(torch.float32, torch.bfloat16), torch.bfloat16)


class KeepResidentSettingTests(unittest.TestCase):
    """Forge 설정 — TE·Qwen3.5·커넥터를 생성 사이 VRAM 에 남길지. 켜짐(기본)은 지금 동작, 끄면 예전처럼 TE·Qwen3.5 는
    인코딩 직후, 커넥터는 생성이 끝날 때(restore) 내린다. 같은 GPU 로 학습할 때 끈다."""

    def setUp(self):
        self.module = _load_lifecycle_runtime()

    @staticmethod
    def _forge_opts(opts):
        modules = types.ModuleType("modules")
        modules.__path__ = []
        shared = types.ModuleType("modules.shared")
        shared.opts = opts
        modules.shared = shared
        return mock.patch.dict(sys.modules, {"modules": modules, "modules.shared": shared})

    def test_setting_is_read_from_forge_opts_and_defaults_to_keeping(self):
        name = self.module.OPT_KEEP_RESIDENT
        with self._forge_opts(types.SimpleNamespace(**{name: False})):
            self.assertFalse(self.module.keep_resident())
        with self._forge_opts(types.SimpleNamespace(**{name: True})):
            self.assertTrue(self.module.keep_resident())
        with self._forge_opts(types.SimpleNamespace()):   # 설정 저장 전
            self.assertTrue(self.module.keep_resident())
        with self._forge_opts(None):   # opts 로드 전
            self.assertTrue(self.module.keep_resident())
        with mock.patch.dict(sys.modules, {"modules": types.ModuleType("modules")}):   # Forge 밖
            sys.modules.pop("modules.shared", None)
            self.assertTrue(self.module.keep_resident())

    def _connector_setup(self):
        runtime = self.module.Anima3BRuntime()
        model = _LifecycleModel()
        te, qwen = types.SimpleNamespace(name="te"), types.SimpleNamespace(name="qwen")
        model.forge_objects.clip = types.SimpleNamespace(patcher=te)
        runtime._qwen_clip = types.SimpleNamespace(patcher=qwen)
        models = types.SimpleNamespace(native_adapter=_WeakAdapter())   # 번들 원본 사본 — 공유 폴백 아님
        patches = (
            mock.patch.object(self.module, "bundle_metadata", return_value={"anima_v2_adapter_filename": "a"}),
            mock.patch.object(runtime, "_load_v2_models", return_value=models),
            mock.patch.object(runtime, "_unload_patchers"),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        return runtime, model, te, qwen

    def _generate(self, runtime, model):
        p = _LifecycleProcessing(types.SimpleNamespace(sd_model=model))
        runtime.install(p, "a", 1.0, None)
        connector = runtime._v2_sampling_patcher
        runtime.restore(p)
        return connector

    def test_setting_off_unloads_connector_qwen_and_te_when_the_generation_ends(self):
        runtime, model, te, qwen = self._connector_setup()
        with mock.patch.object(self.module, "keep_resident", return_value=False):
            first = self._generate(runtime, model)
            runtime._unload_patchers.assert_called_once_with(first, qwen, te)
            second = self._generate(runtime, model)
        self.assertIs(second, first, "패처 객체는 그대로 재사용한다 (내리기만)")
        self.assertEqual(runtime._unload_patchers.call_count, 2)

    def test_setting_on_keeps_them(self):
        runtime, model, _, _ = self._connector_setup()
        with mock.patch.object(self.module, "keep_resident", return_value=True):
            self._generate(runtime, model)
            self._generate(runtime, model)
        runtime._unload_patchers.assert_not_called()

    def test_plain_generation_restore_unloads_nothing_even_when_off(self):
        # 3.8B 가 아닌 생성의 postprocess 도 restore 를 부른다 — Forge 의 TE 를 건드리면 안 된다
        runtime, model, _, _ = self._connector_setup()
        with mock.patch.object(self.module, "keep_resident", return_value=False):
            runtime.restore(_LifecycleProcessing(types.SimpleNamespace(sd_model=model)))
        runtime._unload_patchers.assert_not_called()

    def test_script_registers_the_setting_with_the_current_behaviour_as_default(self):
        registered = []
        callbacks = types.SimpleNamespace(on_model_loaded=lambda fn: None, on_ui_settings=registered.append)
        script_module = _load_script(script_callbacks=callbacks)
        self.assertEqual(len(registered), 1)
        self.assertEqual(script_module.OPT_KEEP_RESIDENT, self.module.OPT_KEEP_RESIDENT)
        added = {}

        class OptionInfo:
            def __init__(self, default, label, component=None, component_args=None, section=None):
                self.default, self.label, self.component, self.section = default, label, component, section
                self.comment = ""

            def info(self, text):
                self.comment = text
                return self

        opts = types.SimpleNamespace(add_option=lambda key, info: added.__setitem__(key, info))
        with self._forge_opts(opts):
            sys.modules["modules.shared"].OptionInfo = OptionInfo
            with mock.patch.object(script_module, "gr", types.SimpleNamespace(Checkbox="checkbox")):
                registered[0]()
        info = added[self.module.OPT_KEEP_RESIDENT]
        self.assertIs(info.default, True)
        self.assertEqual(info.component, "checkbox")
        text = info.label + info.comment
        for needle in ("TE", "Qwen3.5", "커넥터", "6~8 GB", "학습"):
            self.assertIn(needle, text)

    def test_script_without_the_settings_callback_still_loads(self):
        _load_script(script_callbacks=types.SimpleNamespace(on_model_loaded=lambda fn: None))


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


# ---------------------------------------------------------------------------------------------------------------
# Semantic Connector v2 속도 경로 — fp32 상주(connector_fp32)·run 캐시(connector_cache).
# 옛 경로(Forge manual cast: bf16 저장 가중치를 호출마다 fp32 로 캐스트해 계산, 스텝마다 전부 다시 계산)를 실제 Forge
# 모듈(CPU, --cpu)로 돌려 기준으로 삼고, 새 경로 출력이 그와 torch.equal 인지 본다. 가중치 파일 없이 작은 차원·무작위
# 가중치로 돈다. 실제 3.8B 번들 가중치 비교는 SAM3_RUN_FORGE_INTEGRATION_TESTS=1 일 때만(ConnectorRealWeightsTests).

from sam3ext.anima38 import connector_cache, connector_fp32  # noqa: E402

FORGE_ROOT = ROOT.parents[1]
_REAL_FORGE: dict = {}
_FORGE_OWNED = ("backend", "modules", "modules_forge", "gguf")
_FORGE_BOUND = ("sam3ext.anima38.semantic_v2", "sam3ext.anima38.adapter")
_V2_DIM, _V2_SOURCE_DIM = 32, 24


def _forge_owned(name: str) -> bool:
    return name.split(".")[0] in _FORGE_OWNED or name in _FORGE_BOUND


class _RealForge(types.SimpleNamespace):
    def active(self):
        """실제 backend·semantic_v2 모듈을 잠시 sys.modules 에 둔다 — 게으른 import(ModelPatcher 등)와 모듈 조회용.
        우리 이름만 넣고 뺀다(patch.dict 처럼 그 사이 새로 import 된 모듈까지 지우지 않는다)."""
        modules = self.modules

        class _Active:
            def __enter__(inner):
                inner.previous = {name: sys.modules.get(name) for name in modules}
                sys.modules.update(modules)
                return self

            def __exit__(inner, *exc):
                for name, module in inner.previous.items():
                    if module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = module
                return False

        return _Active()


def _real_forge() -> _RealForge:
    """실제 Forge backend(operations·nn.anima·patcher.base·memory_management)와 semantic_v2 를 CPU 로 한 번 불러
    모듈 객체만 쥔다. sys.modules·sys.path·argv 는 원래대로 — 다른 테스트의 backend/modules 스텁과 섞이지 않게."""
    if "forge" in _REAL_FORGE:
        if _REAL_FORGE["forge"] is None:
            raise unittest.SkipTest(_REAL_FORGE["reason"])
        return _REAL_FORGE["forge"]
    if not (FORGE_ROOT / "backend" / "operations.py").is_file():
        _REAL_FORGE.update(forge=None, reason="Forge 본체가 없다")
        raise unittest.SkipTest(_REAL_FORGE["reason"])
    saved = {name: module for name, module in sys.modules.items() if _forge_owned(name)}
    saved_path, saved_argv = list(sys.path), sys.argv
    for name in saved:
        del sys.modules[name]
    try:
        sys.path[:0] = [str(FORGE_ROOT), str(FORGE_ROOT / "modules_forge" / "packages")]
        sys.argv = [saved_argv[0] if saved_argv else "test", "--cpu"]   # backend.args: GPU 를 건드리지 않는다
        loaded = dict(
            ops=importlib.import_module("backend.operations"),
            anima=importlib.import_module("backend.nn.anima"),
            base=importlib.import_module("backend.patcher.base"),
            mm=importlib.import_module("backend.memory_management"),
            semantic_v2=importlib.import_module("sam3ext.anima38.semantic_v2"),
        )
        modules = {name: module for name, module in sys.modules.items()
                   if name.split(".")[0] == "backend" or name in _FORGE_BOUND}
    except Exception as exc:  # pragma: no cover - Forge 버전 차이
        _REAL_FORGE.update(forge=None, reason=f"Forge backend 를 CPU 로 불러오지 못함: {exc!r}")
        raise unittest.SkipTest(_REAL_FORGE["reason"])
    finally:
        sys.path[:] = saved_path
        sys.argv = saved_argv
        for name in [name for name in sys.modules if _forge_owned(name)]:
            del sys.modules[name]
        sys.modules.update(saved)
    _REAL_FORGE["forge"] = _RealForge(modules=modules, **loaded)
    return _REAL_FORGE["forge"]


def _set_path(root, name: str, value) -> None:
    *parents, leaf = name.split(".")
    for part in parents:
        root = getattr(root, part)
    setattr(root, leaf, value)


def _v2_bundle(forge, seed=0, blocks=3):
    """작은 실제 커넥터 번들(BundledV2Models) — 런타임처럼 Forge ops(manual cast)로 만들고 실제 번들과 같은 dtype
    배치(llm_adapter embed 만 fp32, 나머지 bf16)의 무작위 가중치를 넣는다. 계산 dtype 은 embed 의 fp32."""
    generator = torch.Generator().manual_seed(seed)
    with forge.ops.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
        native = forge.anima.LLMAdapter(source_dim=_V2_DIM, target_dim=_V2_DIM, model_dim=_V2_DIM,
                                        num_layers=blocks, num_heads=4)
        connector = forge.semantic_v2.QualityAnchoredSemanticConnectorV2(
            native, semantic_source_dim=_V2_SOURCE_DIM, num_queries=5, resampler_blocks=2,
            resampler_dim=16, resampler_heads=2, mlp_hidden_dim=20)
    models = forge.semantic_v2.BundledV2Models(native, connector)
    with torch.no_grad():
        for name, parameter in list(models.named_parameters()):
            value = torch.randn(parameter.shape, generator=generator) * 0.3
            dtype = torch.float32 if name.endswith("embed.weight") else torch.bfloat16
            _set_path(models, name, torch.nn.Parameter(value.to(dtype), requires_grad=False))
    return models.eval().requires_grad_(False)


def _v2_run_inputs(count=1, seed=1, masked=True, lengths=(7, 9, 6)):
    """커넥터 한 run 의 입력 — 런타임처럼 한 줄을 count 행으로 expand 한 뷰(source·T5 id·의미 특징 4층·mask)."""
    generator = torch.Generator().manual_seed(seed)
    source_tokens, target_tokens, semantic_tokens = lengths
    source = torch.randn(1, source_tokens, _V2_DIM, generator=generator).expand(count, -1, -1)
    target_ids = torch.randint(0, 32128, (1, target_tokens), generator=generator).expand(count, -1)
    semantic = [torch.randn(1, semantic_tokens, _V2_SOURCE_DIM, generator=generator).expand(count, -1, -1)
                for _ in range(4)]
    mask = torch.ones(1, semantic_tokens, dtype=torch.long)
    if masked:
        mask[0, -2] = 0
    return source, target_ids, semantic, mask.expand(count, -1)


def _v2_schedule(count):
    """스텝 timestep — 끝값(1·0) 포함, 행마다 다른 timestep 도 (CFG 행이 다른 sigma 를 받는 경우)."""
    steps = [torch.full((count,), value) for value in (1.0, 0.93, 0.71, 0.5, 0.29, 0.07, 0.0)]
    steps.append(torch.linspace(0.9, 0.2, count))
    return steps


def _old_outputs(models, inputs, schedule):
    source, target_ids, semantic, mask = inputs
    with torch.inference_mode():
        return [models.connector(source, target_ids, semantic, semantic_source_mask=mask, timesteps=t)
                for t in schedule]


def _forge_modules(models):
    return [module for module in models.modules() if hasattr(module, "parameters_manual_cast")]


class ConnectorOldPathReferenceTests(unittest.TestCase):
    """비교 기준(옛 경로)이 정말 지금 샘플링 경로인지 — Forge manual cast 로 bf16 가중치를 호출마다 fp32 로 캐스트해
    계산하고, 같은 입력이면 늘 같은 값이다."""

    def test_reference_is_forge_manual_cast_of_bf16_weights(self):
        forge = _real_forge()
        models = _v2_bundle(forge)
        forge_modules = _forge_modules(models)
        self.assertTrue(forge_modules and all(module.parameters_manual_cast for module in forge_modules))
        for name, parameter in models.named_parameters():
            self.assertEqual(parameter.dtype, torch.float32 if name.endswith("embed.weight") else torch.bfloat16, name)
        inputs = _v2_run_inputs(count=2)
        with mock.patch.object(forge.ops, "weights_manual_cast", wraps=forge.ops.weights_manual_cast) as cast:
            first = _old_outputs(models, inputs, _v2_schedule(2))
        self.assertGreater(cast.call_count, len(forge_modules), "스텝마다 모듈마다 캐스트한다")
        second = _old_outputs(models, inputs, _v2_schedule(2))
        for left, right in zip(first, second):
            self.assertEqual(left.dtype, torch.float32)
            self.assertTrue(torch.equal(left, right))
        self.assertFalse(torch.equal(first[0], first[3]), "timestep 이 출력에 들어간다")


class ConnectorFp32ResidencyTests(unittest.TestCase):
    """(a) fp32 상주 — 올린 뒤 한 번 fp32 로 바꿔 둔 가중치로 같은 fp32 계산. 옛 경로와 비트 단위로 같아야 한다."""

    def setUp(self):
        self.forge = _real_forge()
        self.cpu = torch.device("cpu")
        setting = mock.patch.object(connector_fp32, "connector_fp32", return_value=True)
        setting.start()
        self.addCleanup(setting.stop)

    def _unet(self):
        added = []
        unet = types.SimpleNamespace(load_device=self.cpu, offload_device=self.cpu, added=added,
                                     add_extra_model_patcher_during_sampling=added.append)
        return unet

    def _patcher(self, models, **kwargs):
        with self.forge.active():
            return connector_fp32.make_connector_patcher(self._unet(), models, **kwargs)

    def _assert_same_as_old(self, models, reference, inputs, schedule):
        new = _old_outputs(models, inputs, schedule)   # 같은 upstream forward — 가중치 상태만 다르다
        for index, (left, right) in enumerate(zip(reference, new)):
            self.assertEqual(left.dtype, right.dtype)
            self.assertTrue(torch.equal(left, right), f"step {index}")

    def test_converted_weights_give_the_manual_cast_result_bit_for_bit(self):
        for count, masked in ((1, True), (2, False), (3, True)):
            with self.subTest(count=count, masked=masked):
                models = _v2_bundle(self.forge, seed=count)
                inputs = _v2_run_inputs(count=count, masked=masked, seed=10 + count)
                schedule = _v2_schedule(count)
                reference = _old_outputs(models, inputs, schedule)
                with self.forge.active():
                    plan = connector_fp32.Fp32Plan(models, torch.float32)
                grown = plan.convert(models, self.cpu)
                self.assertEqual(grown, plan.growth)
                self.assertTrue(all(p.dtype == torch.float32 for p in models.parameters()))
                self.assertFalse(any(module.parameters_manual_cast for module in _forge_modules(models)))
                with mock.patch.object(self.forge.ops, "weights_manual_cast", wraps=self.forge.ops.weights_manual_cast) as cast:
                    self._assert_same_as_old(models, reference, inputs, schedule)
                self.assertEqual(cast.call_count, 0, "스텝마다 캐스트하지 않는다")

    def test_revert_restores_the_stored_weights_and_manual_cast_exactly(self):
        models = _v2_bundle(self.forge)
        before = {name: parameter.detach().clone() for name, parameter in models.named_parameters()}
        with self.forge.active():
            plan = connector_fp32.Fp32Plan(models, torch.float32)
        plan.convert(models, self.cpu)
        self.assertEqual(plan.revert(), plan.growth)
        for name, parameter in models.named_parameters():
            self.assertEqual(parameter.dtype, before[name].dtype, name)
            self.assertTrue(torch.equal(parameter, before[name]), name)
        self.assertTrue(all(module.parameters_manual_cast for module in _forge_modules(models)))
        self.assertFalse(plan.active)
        self.assertEqual(plan.revert(), 0, "두 번 되돌려도 그대로")

    def test_conversion_inside_inference_mode_makes_normal_parameters(self):
        models = _v2_bundle(self.forge)
        with self.forge.active():
            plan = connector_fp32.Fp32Plan(models, torch.float32)
        with torch.inference_mode():
            plan.convert(models, self.cpu)
        self.assertFalse(any(parameter.is_inference() for parameter in models.parameters()))
        plan.revert()   # 인퍼런스 모드 밖(Forge 의 postprocess 언로드 자리)에서 되돌려도 된다
        self.assertFalse(any(parameter.is_inference() for parameter in models.parameters()))

    def test_forge_patcher_loads_converts_and_reports_the_fp32_size(self):
        models = _v2_bundle(self.forge)
        inputs, schedule = _v2_run_inputs(count=2), _v2_schedule(2)
        reference = _old_outputs(models, inputs, schedule)
        with self.forge.active():
            storage = self.forge.mm.module_size(models)
            patcher = self._patcher(models)
            plan = connector_fp32.plan_of(models)
            self.assertIsInstance(patcher, self.forge.base.ModelPatcher)
            self.assertEqual(plan.storage_size, storage)
            self.assertEqual(patcher.model_size(), storage, "바꾸기 전에는 원래 크기 — Forge 판단이 기본 패처와 같다")
            self.assertGreater(plan.growth, 0)
            gained = patcher.partially_load(self.cpu, 1e32)
            self.assertEqual(patcher.loaded_size(), patcher.model_size(), "다 올리면 Forge 가 보는 적재 크기 = fp32 크기")
            self.assertEqual(gained, patcher.model_size())
            self.assertEqual(self.forge.mm.module_size(models), patcher.model_size())
            self.assertTrue(plan.active)
            self.assertEqual(patcher.model_size(), storage + plan.growth, "바꾼 뒤에는 fp32 크기")
            self.assertEqual(patcher.partially_load(self.cpu, 1e32), 0, "이미 다 올라가 있으면 아무것도 안 한다")
        self._assert_same_as_old(models, reference, inputs, schedule)
        with mock.patch.object(connector_fp32, "connector_fp32", return_value=False):
            self.assertEqual(patcher.sync_fp32_setting(), plan.growth)
            self.assertEqual(patcher.model_size(), storage, "설정을 끄고 되돌리면 원래 크기")
            self.assertEqual(patcher.loaded_size(), storage)

    def test_llm_adapter_lora_is_merged_in_bf16_then_widened(self):
        """LoRA 의 llm_adapter 몫(_sync_adapter_lora 가 옮긴 패치)은 Forge 가 원래 dtype 에 합치고 반올림한다 — fp32 로
        바꾼 뒤에 합치면 반올림이 빠져 값이 달라진다. 순정 ModelPatcher(옛 경로)와 같아야 한다."""
        old_models, new_models = _v2_bundle(self.forge), _v2_bundle(self.forge)
        keys = ("native_adapter.blocks.1.cross_attn.q_proj.weight", "connector.v2_attentions.2.o_proj.weight",
                "native_adapter.blocks.0.mlp.0.bias")
        generator = torch.Generator().manual_seed(5)
        diffs = {key: torch.randn(old_models.get_parameter(key).shape, generator=generator) * 0.01 for key in keys}
        inputs, schedule = _v2_run_inputs(count=2), _v2_schedule(2)
        with self.forge.active():
            old = self.forge.base.ModelPatcher(old_models, self.cpu, self.cpu)
            new = self._patcher(new_models)
            for patcher in (old, new):
                patcher.patches = {key: [(1.0, (diff,), 1.0, None, None)] for key, diff in diffs.items()}
                patcher.patches_uuid = uuid4()
                patcher.partially_load(self.cpu, 1e32)
        self.assertTrue(connector_fp32.plan_of(new_models).active)
        self.assertEqual(old_models.get_parameter(keys[0]).dtype, torch.bfloat16)
        self.assertEqual(new_models.get_parameter(keys[0]).dtype, torch.float32)
        reference = _old_outputs(old_models, inputs, schedule)
        self._assert_same_as_old(new_models, reference, inputs, schedule)
        with self.forge.active():
            new.unpatch_model(self.cpu, unpatch_weights=True)
            old.unpatch_model(self.cpu, unpatch_weights=True)
        pristine = _v2_bundle(self.forge)
        for name, parameter in new_models.named_parameters():
            self.assertEqual(parameter.dtype, pristine.get_parameter(name).dtype, name)
            self.assertTrue(torch.equal(parameter, pristine.get_parameter(name)), name)

    def test_unload_and_partial_unload_return_to_the_stored_dtype(self):
        models = _v2_bundle(self.forge)
        inputs, schedule = _v2_run_inputs(), _v2_schedule(1)
        reference = _old_outputs(models, inputs, schedule)
        with self.forge.active():
            patcher = self._patcher(models)
            plan = connector_fp32.plan_of(models)
            patcher.partially_load(self.cpu, 1e32)
            # Forge 가 조금만 비우라고 하면 fp32→bf16 되돌림으로 충분 — 가중치는 그대로 올라가 있다
            freed = patcher.partially_unload(self.cpu, memory_to_free=plan.growth // 2)
            self.assertEqual(freed, plan.growth)
            self.assertFalse(plan.active)
            self.assertFalse(models.model_lowvram)
            self.assertEqual(patcher.loaded_size(), plan.storage_size)
            self._assert_same_as_old(models, reference, inputs, schedule)
            patcher.partially_load(self.cpu, 1e32)   # 다음 샘플링 — 다시 fp32
            self.assertTrue(plan.active)
            self.assertEqual(patcher.loaded_size(), patcher.model_size())
            patcher.detach()   # Forge 언로드(model_unload → detach → unpatch_model)
            self.assertFalse(plan.active)
            self.assertEqual(patcher.loaded_size(), 0)
            self.assertTrue(all(p.dtype == torch.bfloat16 for n, p in models.named_parameters()
                                if not n.endswith("embed.weight")))
            self.assertTrue(all(module.parameters_manual_cast for module in _forge_modules(models)))
        self._assert_same_as_old(models, reference, inputs, schedule)

    def test_setting_off_is_the_plain_forge_path(self):
        models = _v2_bundle(self.forge)
        with self.forge.active():
            patcher = self._patcher(models)
            plan = connector_fp32.plan_of(models)
            patcher.partially_load(self.cpu, 1e32)
            self.assertTrue(plan.active)
            with mock.patch.object(connector_fp32, "connector_fp32", return_value=False):
                self.assertEqual(patcher.sync_fp32_setting(), plan.growth)   # 설치 때 — 지난 생성의 fp32 를 되돌린다
                self.assertFalse(plan.active)
                self.assertEqual(patcher.loaded_size(), patcher.model_size())
                self.assertEqual(patcher.partially_load(self.cpu, 1e32), 0)
                self.assertFalse(plan.active, "꺼져 있으면 바꾸지 않는다")
                self.assertTrue(all(module.parameters_manual_cast for module in _forge_modules(models)))

    def test_without_room_for_fp32_the_manual_cast_path_stays(self):
        models = _v2_bundle(self.forge)
        with self.forge.active():
            patcher = self._patcher(models)
            plan = connector_fp32.plan_of(models)
            with mock.patch.object(connector_fp32, "connector_fp32", return_value=False):
                patcher.partially_load(self.cpu, 1e32)   # bf16 로 다 올라가 있다
            self.assertFalse(models.model_lowvram)
            # Forge 가 더 쓸 수 있다고 준 몫(extra_memory)이 fp32 증가분보다 작으면 바꾸지 않는다
            with mock.patch("builtins.print") as printed:
                self.assertEqual(patcher.partially_load(self.cpu, plan.growth // 2), 0)
            self.assertIn("skipped", printed.call_args[0][0])
            self.assertFalse(plan.active)
            self.assertEqual(patcher.loaded_size(), plan.storage_size)
            self.assertTrue(all(module.parameters_manual_cast for module in _forge_modules(models)))
            self.assertEqual(patcher.partially_load(self.cpu, plan.growth), plan.growth, "자리가 나면 바꾼다")
            self.assertTrue(plan.active)

    def test_room_between_bf16_and_fp32_loads_like_the_plain_patcher(self):
        """Forge 가 준 여유가 bf16 크기 이상·fp32 크기 미만이면 순정 ModelPatcher(옛 경로)처럼 bf16 로 다 올린다(full_load)
        — fp32 크기를 미리 보고해 lowvram 부분 적재가 되면 안 된다. 바꾸지 못한 채로는 덜 올라간 몫(offloaded)도 0."""
        inputs, schedule = _v2_run_inputs(count=2), _v2_schedule(2)
        for blocks in (3, 6):
            for fraction in (0.1, 0.5, 0.9):
                with self.subTest(blocks=blocks, fraction=fraction):
                    plain_models, models = _v2_bundle(self.forge, blocks=blocks), _v2_bundle(self.forge, blocks=blocks)
                    reference = _old_outputs(plain_models, inputs, schedule)
                    with self.forge.active():
                        plain = self.forge.base.ModelPatcher(plain_models, self.cpu, self.cpu)
                        patcher = self._patcher(models)
                        plan = connector_fp32.plan_of(models)
                        self.assertEqual(patcher.model_size(), plain.model_size())
                        extra = plan.storage_size + int(plan.growth * fraction)
                        plain_gained = plain.partially_load(self.cpu, extra)
                        with mock.patch("builtins.print"):
                            gained = patcher.partially_load(self.cpu, extra)
                        self.assertFalse(plain_models.model_lowvram)
                        self.assertEqual(models.model_lowvram, plain_models.model_lowvram)
                        self.assertEqual(gained, plain_gained)
                        self.assertEqual(patcher.loaded_size(), plain.loaded_size())
                        self.assertEqual(patcher.model_size(), plain.model_size())
                        self.assertFalse(plan.active, "자리가 모자라면 bf16 그대로")
                        self.assertFalse(any(hasattr(m, "prev_parameters_manual_cast") for m in _forge_modules(models)))
                        loaded = self.forge.mm.LoadedModel(patcher)
                        self.assertEqual(loaded.model_offloaded_memory(), 0, "Forge 가 커넥터를 덜 올라갔다고 보지 않는다")
                        self.assertEqual(loaded.model_offloaded_memory(),
                                         self.forge.mm.LoadedModel(plain).model_offloaded_memory())
                    self._assert_same_as_old(models, reference, inputs, schedule)

    def test_room_for_bf16_and_fp32_loads_then_converts(self):
        models = _v2_bundle(self.forge)
        with self.forge.active():
            patcher = self._patcher(models)
            plan = connector_fp32.plan_of(models)
            with mock.patch("builtins.print"):
                gained = patcher.partially_load(self.cpu, plan.storage_size + plan.growth)
            self.assertTrue(plan.active)
            self.assertFalse(models.model_lowvram)
            self.assertEqual(gained, plan.storage_size + plan.growth)
            self.assertEqual(patcher.loaded_size(), patcher.model_size())
            self.assertEqual(self.forge.mm.LoadedModel(patcher).model_offloaded_memory(), 0)

    def test_unbounded_room_still_respects_the_free_vram(self):
        """HIGH_VRAM(여유 1e32)이어도 실제 빈 VRAM 에서 샘플링 최소 몫을 남긴 만큼까지만 바꾼다 — model_size 가 fp32 몫을
        미리 보고하지 않으니 Forge 가 그만큼 비워 두지 않는다. 장치는 CPU 지만 GPU 로 보이게 해 확인한다."""
        mm = self.forge.mm
        reserve = mm.minimum_inference_memory()
        for spare, converts in ((-1, False), (0, True), (1, True)):
            with self.subTest(spare=spare):
                models = _v2_bundle(self.forge)
                with self.forge.active():
                    patcher = self._patcher(models)
                    plan = connector_fp32.plan_of(models)
                    free = reserve + plan.growth + spare * (plan.growth // 2)
                    with mock.patch.object(mm, "is_device_cpu", return_value=False),                             mock.patch.object(mm, "get_free_memory", return_value=free),                             mock.patch("builtins.print"):
                        patcher.partially_load(self.cpu, 1e32)
                    self.assertFalse(models.model_lowvram)
                    self.assertEqual(plan.active, converts)
                    self.assertEqual(patcher.loaded_size(), patcher.model_size())

    def test_shared_fallback_leaves_the_te_llm_adapter_alone(self):
        models = _v2_bundle(self.forge)
        with self.forge.active():
            patcher = self._patcher(models, skip_prefixes=("native_adapter.",))
            patcher.partially_load(self.cpu, 1e32)
        for name, parameter in models.named_parameters():
            if name.startswith("native_adapter."):
                self.assertNotEqual(parameter.dtype, torch.float32 if not name.endswith("embed.weight") else None, name)
            else:
                self.assertEqual(parameter.dtype, torch.float32, name)
        self.assertTrue(all(module.parameters_manual_cast for module in models.native_adapter.modules()
                            if hasattr(module, "parameters_manual_cast")))

    def test_real_forge_load_models_gpu_flow(self):
        """Forge 의 load_models_gpu → LoadedModel.model_load → partially_load 흐름(CPU 장치)으로 바꾸고, 언로드로 되돌린다."""
        models = _v2_bundle(self.forge)
        inputs, schedule = _v2_run_inputs(count=2), _v2_schedule(2)
        reference = _old_outputs(models, inputs, schedule)
        mm = self.forge.mm
        with self.forge.active():
            patcher = self._patcher(models)
            before = list(mm.current_loaded_models)
            try:
                mm.load_models_gpu([patcher])
                self.assertTrue(connector_fp32.plan_of(models).active)
                self._assert_same_as_old(models, reference, inputs, schedule)
            finally:
                for loaded in [item for item in mm.current_loaded_models if item.model is patcher]:
                    loaded.model_unload()
                    mm.current_loaded_models.remove(loaded)
            self.assertEqual(mm.current_loaded_models, before)
        self.assertFalse(connector_fp32.plan_of(models).active)
        self._assert_same_as_old(models, reference, inputs, schedule)

    def test_make_patcher_falls_back_without_a_forge_unet(self):
        models = _v2_bundle(self.forge)
        self.assertIsNone(connector_fp32.make_connector_patcher(object(), models))
        self.assertIsNone(connector_fp32.plan_of(models))


class ConnectorRunCacheTests(unittest.TestCase):
    """(b) run 캐시 — timestep 무관 계산을 첫 스텝에 한 번. 모든 스텝·모든 timestep 에서 upstream forward 와 같아야 한다."""

    def setUp(self):
        self.forge = _real_forge()

    def _cached_outputs(self, models, inputs, schedule):
        source, target_ids, semantic, mask = inputs
        with self.forge.active(), torch.inference_mode():
            self.assertTrue(connector_cache.supports(models.connector))
            cache = connector_cache.prepare(models.connector, source, target_ids, semantic, semantic_source_mask=mask)
            return cache, [connector_cache.forward(models.connector, cache, t) for t in schedule]

    def test_every_step_matches_the_upstream_forward(self):
        for fp32 in (False, True):
            for count, masked in ((1, True), (2, True), (2, False)):
                with self.subTest(fp32=fp32, count=count, masked=masked):
                    models = _v2_bundle(self.forge, seed=count)
                    inputs = _v2_run_inputs(count=count, masked=masked, seed=20 + count)
                    schedule = _v2_schedule(count)
                    reference = _old_outputs(models, inputs, schedule)
                    if fp32:
                        with self.forge.active():
                            connector_fp32.Fp32Plan(models, torch.float32).convert(models, torch.device("cpu"))
                    _, new = self._cached_outputs(models, inputs, schedule)
                    for index, (left, right) in enumerate(zip(reference, new)):
                        self.assertEqual(left.dtype, right.dtype)
                        self.assertTrue(torch.equal(left, right), f"step {index}")

    def test_cached_tensors_are_never_written_by_the_steps(self):
        models = _v2_bundle(self.forge)
        inputs = _v2_run_inputs(count=2)
        source, target_ids, semantic, mask = inputs
        with self.forge.active(), torch.inference_mode():
            cache = connector_cache.prepare(models.connector, source, target_ids, semantic, semantic_source_mask=mask)
            snapshot = [tensor.clone() for tensor in cache.tensors()]
            for t in _v2_schedule(2):
                connector_cache.forward(models.connector, cache, t)
        for before, after in zip(snapshot, cache.tensors()):
            self.assertTrue(torch.equal(before, after), "TransformerBlock 의 제자리 add_ 가 캐시를 건드리면 안 된다")
        self.assertGreater(cache.nbytes, 0)

    def test_the_cache_holds_only_timestep_free_values(self):
        """같은 run 을 다른 timestep 뒤에 준비해도 캐시가 같다 — 준비는 timestep 을 받지도 않는다."""
        models = _v2_bundle(self.forge)
        inputs = _v2_run_inputs(count=1)
        first, _ = self._cached_outputs(models, inputs, [torch.tensor([0.9])])
        second, _ = self._cached_outputs(models, inputs, [torch.tensor([0.1])])
        for left, right in zip(first.tensors(), second.tensors()):
            self.assertTrue(torch.equal(left, right))

    def test_unknown_shapes_and_hooks_fall_back_to_the_upstream_forward(self):
        models = _v2_bundle(self.forge)
        with self.forge.active():
            self.assertTrue(connector_cache.supports(models.connector))
            self.assertFalse(connector_cache.supports(lambda *a, **k: None))
            self.assertFalse(connector_cache.supports(torch.nn.Linear(2, 2)))
            handle = models.native_adapter.blocks[1].register_forward_hook(lambda *a: None)
            self.assertFalse(connector_cache.supports(models.connector), "건너뛸 블록에 훅")
            handle.remove()
            handle = models.connector.semantic_resampler.blocks[0].cross_attention.register_forward_pre_hook(
                lambda *a: None)
            self.assertFalse(connector_cache.supports(models.connector))
            handle.remove()
            models.connector.quality_anchor.semantic_attentions[0].forward = lambda *a, **k: None
            self.assertFalse(connector_cache.supports(models.connector), "인스턴스 forward 덮어쓰기")
            del models.connector.quality_anchor.semantic_attentions[0].forward
            self.assertTrue(connector_cache.supports(models.connector))
            handle = torch.nn.modules.module.register_module_forward_hook(lambda *a: None)
            try:
                self.assertFalse(connector_cache.global_hooks_clear())
                self.assertFalse(connector_cache.supports(models.connector))
            finally:
                handle.remove()
            self.assertTrue(connector_cache.global_hooks_clear())


def _v2_real_features(count, seed=0):
    """_extract_prompt_features 대역 — 작은 실제 커넥터 차원에 맞춘 줄마다 다른 입력(의미 특징 길이도 다르게)."""
    generator = torch.Generator().manual_seed(seed)
    native_rows, semantic_rows = [], []
    for index in range(count):
        tokens = 5 + index % 3
        native_rows.append((
            torch.randn(1, tokens + 1, _V2_DIM, generator=generator),
            torch.randint(0, 32128, (1, tokens), generator=generator),
            torch.rand(1, tokens, 1, generator=generator) + 0.5,
        ))
        length = 4 + index % 4
        mask = torch.ones(1, length, dtype=torch.long)
        mask[0, 0] = 0 if index % 2 else 1
        semantic_rows.append((
            [torch.randn(1, length, _V2_SOURCE_DIM, generator=generator).to(torch.bfloat16) for _ in range(4)],
            mask,
        ))
    return object(), native_rows, semantic_rows


class ConnectorRunCacheRuntimeTests(unittest.TestCase):
    """런타임의 _expand_v2_context 가 run 캐시를 run 의 장치 사본 곁에 두고 쓴다 — 옛 구현(커넥터를 스텝마다 통째로)
    과 모든 스텝에서 같고, LoRA·커넥터 교체·restore·LRU·설정·공유 폴백에서 버리거나 쓰지 않는다."""

    def setUp(self):
        self.forge = _real_forge()
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        self.models = _v2_bundle(self.forge)
        self.te_adapter = torch.nn.Module()   # TE 의 llm_adapter — 커넥터는 번들 사본을 쓴다(공유 폴백 아님)
        self.runtime._v2_models = self.models
        self.runtime._v2_source = weakref.ref(self.te_adapter)
        self.runtime._v2_sampling_patcher = types.SimpleNamespace(patches_uuid=uuid4(), patches={})
        self.native_clip = types.SimpleNamespace(patcher=types.SimpleNamespace(patches={}))   # llm_adapter LoRA 없음
        self.uncond = torch.randn(1, 512, _V2_DIM, generator=torch.Generator().manual_seed(3))
        active = self.forge.active()
        active.__enter__()
        self.addCleanup(active.__exit__, None, None, None)
        prepare = mock.patch.object(connector_cache, "prepare", wraps=connector_cache.prepare)
        self.prepare = prepare.start()
        self.addCleanup(prepare.stop)

    def _encode(self, count, seed=0):
        with mock.patch.object(self.runtime, "_extract_prompt_features", return_value=_v2_real_features(count, seed)):
            conds = self.runtime._encode_v2(object(), self.native_clip, ["line"] * count)
        return conds, [int(anima_marker.read_run_ids(cond)[0]) for cond in conds]

    def _step(self, context, timesteps, run_ids):
        with torch.inference_mode():
            old = _reference_expand_v2_context(self.runtime, context, timesteps, run_ids)
            new = self.runtime._expand_v2_context(context, timesteps, run_ids)
        self.assertEqual(new.dtype, old.dtype)
        self.assertTrue(torch.equal(new, old))
        return new

    def _run_caches(self, run_id):
        return [key for key in self.runtime._v2_runs[run_id].device_copies if key[0] == "run_cache"]

    def test_every_step_and_layout_matches_the_old_expand(self):
        conds, (a, b) = self._encode(2)
        rows = {a: conds[0], b: conds[1], -1: self.uncond}
        layouts = ([a, -1], [a, b], [a, -1, a, -1], [b, a, -1])
        for layout in layouts:
            context = torch.cat([rows[run_id] for run_id in layout])
            for step, value in enumerate((0.95, 0.6, 0.3, 0.0)):
                timesteps = torch.full((len(layout),), value)
                if step == 3:
                    timesteps = torch.linspace(0.8, 0.1, len(layout))
                self._step(context, timesteps, torch.tensor(layout))
        # 한 run·한 행 수마다 준비는 한 번 — 스텝이 늘어도 다시 하지 않는다
        # (a: 1행·2행, b: 1행 — [a, b] 와 [b, a, -1] 의 b 는 같은 1행 캐시)
        self.assertEqual(self.prepare.call_count, 3)
        self.assertEqual(len(self._run_caches(a)), 2)
        self.assertEqual(len(self._run_caches(b)), 1)

    def test_fp32_resident_weights_and_run_cache_together_match(self):
        conds, (a,) = self._encode(1)
        context = torch.cat([conds[0], self.uncond])
        reference = []
        for value in (0.9, 0.4, 0.0):
            with torch.inference_mode():
                reference.append(_reference_expand_v2_context(
                    self.runtime, context, torch.tensor([value, value]), torch.tensor([a, -1])))
        connector_fp32.Fp32Plan(self.models, torch.float32).convert(self.models, torch.device("cpu"))
        for value, expected in zip((0.9, 0.4, 0.0), reference):
            with torch.inference_mode():
                new = self.runtime._expand_v2_context(context, torch.tensor([value, value]), torch.tensor([a, -1]))
            self.assertTrue(torch.equal(new, expected))

    def test_lora_or_connector_change_rebuilds_and_drops_the_old_cache(self):
        conds, (a,) = self._encode(1)
        step = (conds[0], torch.tensor([0.5]), torch.tensor([a]))
        self._step(*step)
        self._step(*step)
        self.assertEqual(self.prepare.call_count, 1)
        self.runtime._v2_sampling_patcher.patches_uuid = uuid4()   # _sync_adapter_lora 가 llm_adapter LoRA 를 옮김
        self._step(*step)
        self.assertEqual(self.prepare.call_count, 2)
        self.assertEqual(len(self._run_caches(a)), 1, "옛 가중치의 캐시는 버린다")
        self.models.current_weight_patches_uuid = uuid4()   # Forge 가 다른 패치로 다시 올림
        self._step(*step)
        self.assertEqual(self.prepare.call_count, 3)
        self.runtime._v2_models = _v2_bundle(self.forge, seed=9)   # 다른 번들(설치가 run 을 비우지만 키도 다르다)
        self._step(*step)
        self.assertEqual(self.prepare.call_count, 4)
        self.assertEqual(len(self._run_caches(a)), 1)

    def test_restore_model_load_and_eviction_release_the_cache(self):
        conds, ids = self._encode(3)
        for cond, run_id in zip(conds, ids):
            self._step(cond, torch.tensor([0.5]), torch.tensor([run_id]))
        runs = [self.runtime._v2_runs[run_id] for run_id in ids]
        self.assertTrue(all(self._run_caches(run_id) for run_id in ids))
        with mock.patch.object(self.module, "V2_DEVICE_CACHE_RUNS", 2):
            self._step(conds[0], torch.tensor([0.4]), torch.tensor([ids[0]]))
        self.assertFalse(runs[1].device_copies, "LRU 로 밀려난 run 은 캐시도 놓는다")
        self.runtime.release_stale_caches(types.SimpleNamespace())   # on_model_loaded — 사본·캐시를 놓는다
        self.assertTrue(all(not run.device_copies for run in runs))
        self.runtime._v2_models = self.models   # (다른 모델이라 커넥터도 놓았다 — 다시 쥐고 이어 간다)
        self.runtime._v2_source = weakref.ref(self.te_adapter)
        self._step(conds[0], torch.tensor([0.3]), torch.tensor([ids[0]]))
        self.assertTrue(self._run_caches(ids[0]))
        self.runtime._clear_v2_runs()   # install·restore 가 부른다
        self.assertEqual(self.runtime._v2_runs, {})
        self.assertTrue(all(not run.device_copies for run in runs))

    def test_shared_fallback_setting_off_and_over_budget_use_the_plain_connector(self):
        conds, (a,) = self._encode(1)
        step = (conds[0], torch.tensor([0.5]), torch.tensor([a]))
        self.runtime._v2_source = weakref.ref(self.models.native_adapter)   # 공유 폴백 — TE 패처가 가중치를 바꾼다
        self._step(*step)
        self.runtime._v2_source = weakref.ref(self.te_adapter)
        with mock.patch.object(connector_cache, "run_cache_enabled", return_value=False):
            self._step(*step)
        self.assertEqual(self.prepare.call_count, 0)
        self.assertEqual(self._run_caches(a), [])
        with mock.patch.object(self.module, "V2_RUN_CACHE_BYTES", 0):
            self._step(*step)   # 준비는 하지만(같은 계산) 남기지 않는다
            self._step(*step)
        self.assertEqual(self.prepare.call_count, 2)
        self.assertEqual(self._run_caches(a), [])
        self._step(*step)
        self.assertEqual(len(self._run_caches(a)), 1)


class _ForgeLikeUnet(_LifecycleUnet):
    """Forge UnetPatcher 처럼 추가 패처를 받는 UNet (장치는 CPU)."""

    def __init__(self):
        super().__init__()
        self.load_device = self.offload_device = torch.device("cpu")

    def add_extra_model_patcher_during_sampling(self, patcher):
        self.extra_model_patchers_during_sampling.append(patcher)


class ConnectorSpeedInstallTests(unittest.TestCase):
    """설치가 Forge UNet 이면 fp32 상주 패처를 달고(공유 폴백이면 TE 모듈은 빼고), 설정을 끄면 설치 때 되돌린다."""

    def setUp(self):
        self.forge = _real_forge()
        self.module = _load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        self.model = _LifecycleModel()
        self.model.forge_objects.unet = _ForgeLikeUnet()
        self.models = _v2_bundle(self.forge)
        self.te_adapter = torch.nn.Module()
        active = self.forge.active()
        active.__enter__()
        self.addCleanup(active.__exit__, None, None, None)

    def _install(self, source):
        def load(sd_model, path, metadata):
            self.runtime._v2_models = self.models
            self.runtime._v2_source = weakref.ref(source)
            return self.models

        p = _LifecycleProcessing(types.SimpleNamespace(sd_model=self.model))
        with mock.patch.object(self.module, "bundle_metadata", return_value={"anima_v2_adapter_filename": "a"}), \
                mock.patch.object(self.runtime, "_load_v2_models", side_effect=load):
            self.runtime.install(p, "a", 1.0, None)
        return p

    def test_forge_unet_gets_the_fp32_patcher(self):
        p = self._install(self.te_adapter)
        patcher = self.runtime._v2_sampling_patcher
        self.assertIsInstance(patcher, self.forge.base.ModelPatcher)
        self.assertTrue(hasattr(patcher, "sync_fp32_setting"))
        self.assertIs(patcher.model, self.models)
        self.assertEqual(connector_fp32.plan_of(self.models).skip_prefixes, ())
        unet = self.model.forge_objects.unet
        self.assertEqual(sum(1 for item in unet.extra_model_patchers_during_sampling if item is patcher), 1)
        with mock.patch.object(connector_fp32, "connector_fp32", return_value=True):
            patcher.partially_load(torch.device("cpu"), 1e32)
        self.assertTrue(connector_fp32.plan_of(self.models).active)
        self.runtime.restore(p)
        with mock.patch.object(connector_fp32, "connector_fp32", return_value=False):
            self._install(self.te_adapter)   # 설정을 끄고 다음 생성 — 같은 패처를 다시 달며 fp32 를 되돌린다
        self.assertIs(self.runtime._v2_sampling_patcher, patcher)
        self.assertFalse(connector_fp32.plan_of(self.models).active)

    def test_shared_fallback_keeps_the_te_module_in_its_dtype(self):
        self._install(self.models.native_adapter)
        self.assertEqual(connector_fp32.plan_of(self.models).skip_prefixes, ("native_adapter.",))


class ConnectorSpeedSettingTests(unittest.TestCase):
    """Forge 설정 두 개 — 'SAM Extra Anima 3.8B' 섹션, 기본 켬(결과는 같고 빨라지는 쪽), 끄면 예전 경로."""

    def test_readers_follow_forge_opts_and_default_on(self):
        for reader, name in ((connector_fp32.connector_fp32, connector_fp32.OPT_CONNECTOR_FP32),
                             (connector_cache.run_cache_enabled, connector_cache.OPT_CONNECTOR_RUN_CACHE)):
            with self.subTest(name=name):
                for opts, expected in ((types.SimpleNamespace(**{name: False}), False),
                                       (types.SimpleNamespace(**{name: True}), True),
                                       (types.SimpleNamespace(), True), (None, True)):
                    with KeepResidentSettingTests._forge_opts(opts):
                        self.assertIs(reader(), expected)

    def test_script_registers_both_settings(self):
        registered = []
        callbacks = types.SimpleNamespace(on_model_loaded=lambda fn: None, on_ui_settings=registered.append)
        script_module = _load_script(script_callbacks=callbacks)
        self.assertEqual(script_module.OPT_CONNECTOR_FP32, connector_fp32.OPT_CONNECTOR_FP32)
        self.assertEqual(script_module.OPT_CONNECTOR_RUN_CACHE, connector_cache.OPT_CONNECTOR_RUN_CACHE)
        added = {}

        class OptionInfo:
            def __init__(self, default, label, component=None, component_args=None, section=None):
                self.default, self.label, self.component, self.section = default, label, component, section
                self.comment = ""

            def info(self, text):
                self.comment = text
                return self

        opts = types.SimpleNamespace(add_option=lambda key, info: added.__setitem__(key, info))
        with KeepResidentSettingTests._forge_opts(opts):
            sys.modules["modules.shared"].OptionInfo = OptionInfo
            with mock.patch.object(script_module, "gr", types.SimpleNamespace(Checkbox="checkbox")):
                registered[0]()
        fp32 = added[connector_fp32.OPT_CONNECTOR_FP32]
        cache = added[connector_cache.OPT_CONNECTOR_RUN_CACHE]
        for info in (fp32, cache):
            self.assertIs(info.default, True)
            self.assertEqual(info.component, "checkbox")
            self.assertEqual(info.section, ("sam3_anima38", "SAM Extra Anima 3.8B"))
            self.assertIn("같습니다", info.comment)
        self.assertIn("1.5 GB", fp32.label)
        self.assertIn("fp32", fp32.label)
        self.assertIn("빨라짐", fp32.label)
        self.assertIn("MB", cache.label)


class ReferenceIpaSettingTests(unittest.TestCase):
    """캐릭터 레퍼런스 IP-Adapter 의 3.8B 커넥터 토글 — 이어붙이기 방식과 같은 조건이 기본(켬)."""

    def test_script_registers_the_toggle_under_the_runner_name(self):
        from sam3ext import anima_reference_runner

        registered = []
        callbacks = types.SimpleNamespace(on_model_loaded=lambda fn: None, on_ui_settings=registered.append)
        script_module = _load_script(script_callbacks=callbacks)
        self.assertEqual(script_module.OPT_IPA_ANIMA38, anima_reference_runner.OPT_IPA_ANIMA38)
        added = {}

        class OptionInfo:
            def __init__(self, default, label, component=None, component_args=None, section=None):
                self.default, self.label, self.component, self.section = default, label, component, section
                self.comment = ""

            def info(self, text):
                self.comment = text
                return self

        opts = types.SimpleNamespace(add_option=lambda key, info: added.__setitem__(key, info))
        with KeepResidentSettingTests._forge_opts(opts):
            sys.modules["modules.shared"].OptionInfo = OptionInfo
            with mock.patch.object(script_module, "gr", types.SimpleNamespace(Checkbox="checkbox")):
                registered[0]()
        info = added[script_module.OPT_IPA_ANIMA38]
        self.assertIs(info.default, True)
        self.assertIs(info.default, anima_reference_runner.DEFAULT_IPA_ANIMA38)
        self.assertEqual(info.component, "checkbox")
        self.assertEqual(info.section, ("sam3_anima38", "SAM Extra Anima 3.8B"))
        self.assertIn("결과가 바뀜", info.label)
        self.assertIn("Reference Anima38", info.comment)


_REAL_BUNDLE = Path(os.environ.get(
    "SAM3_ANIMA38_BUNDLE", str(FORGE_ROOT / "models" / "Stable-diffusion" / "Anima-3.8B-v1.1.safetensors")))


@unittest.skipUnless(
    os.environ.get("SAM3_RUN_FORGE_INTEGRATION_TESTS") == "1" and _REAL_BUNDLE.is_file(),
    "실제 3.8B 번들 가중치 비교 — SAM3_RUN_FORGE_INTEGRATION_TESTS=1 (번들: SAM3_ANIMA38_BUNDLE 또는 기본 경로)",
)
class ConnectorRealWeightsTests(unittest.TestCase):
    """실제 번들 가중치(읽기 전용 safe_open)로 만든 실제 크기 커넥터 — CPU 에서 옛 경로와 새 경로가 같다(수십 초, RAM 약 6 GB)."""

    def test_real_bundle_connector_matches_bit_for_bit(self):
        forge = _real_forge()
        module = _load_lifecycle_runtime()
        runtime = module.Anima3BRuntime()
        with forge.ops.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=True):
            te_adapter = forge.anima.LLMAdapter()
        with torch.no_grad():   # Forge TE 와 같은 dtype 배치 — embed 만 fp32
            for name, parameter in list(te_adapter.named_parameters()):
                dtype = torch.float32 if name == "embed.weight" else torch.bfloat16
                _set_path(te_adapter, name, torch.nn.Parameter(torch.zeros(parameter.shape, dtype=dtype),
                                                               requires_grad=False))
        native_clip = types.SimpleNamespace(cond_stage_model=types.SimpleNamespace(
            qwen3_06b=types.SimpleNamespace(llm_adapter=te_adapter)))
        metadata = anima_files.bundle_metadata(_REAL_BUNDLE)
        self.assertIsNotNone(metadata)
        with forge.active(), \
                mock.patch.object(module, "using_forge_operations", forge.ops.using_forge_operations), \
                mock.patch.object(module, "QualityAnchoredSemanticConnectorV2",
                                  forge.semantic_v2.QualityAnchoredSemanticConnectorV2), \
                mock.patch.object(module, "BundledV2Models", forge.semantic_v2.BundledV2Models), \
                mock.patch.object(runtime, "_require_anima", return_value=(None, native_clip)):
            models = runtime._load_v2_models(object(), str(_REAL_BUNDLE), metadata)
        self.assertIsNot(models.native_adapter, te_adapter, "번들 원본 llm_adapter 사본")
        generator = torch.Generator().manual_seed(0)
        tokens = 24
        source = torch.randn(1, tokens, 1024, generator=generator).expand(2, -1, -1)
        target_ids = torch.randint(0, 32128, (1, tokens), generator=generator).expand(2, -1)
        semantic = [torch.randn(1, tokens, 2560, generator=generator).expand(2, -1, -1) for _ in range(4)]
        mask = torch.ones(1, tokens, dtype=torch.long).expand(2, -1)
        schedule = [torch.tensor([0.9, 0.9]), torch.tensor([0.4, 0.2])]
        reference = _old_outputs(models, (source, target_ids, semantic, mask), schedule)
        with forge.active():
            plan = connector_fp32.Fp32Plan(models, torch.float32)
            self.assertAlmostEqual(plan.growth / 1e9, 1.5, delta=0.1, msg="VRAM 추가량 약 1.5 GB")
            plan.convert(models, torch.device("cpu"))
            with torch.inference_mode():
                self.assertTrue(connector_cache.supports(models.connector))
                cache = connector_cache.prepare(models.connector, source, target_ids, semantic, semantic_source_mask=mask)
                cached = [connector_cache.forward(models.connector, cache, t) for t in schedule]
        fp32 = _old_outputs(models, (source, target_ids, semantic, mask), schedule)
        for left, right, both in zip(reference, fp32, cached):
            self.assertTrue(torch.equal(left, right))
            self.assertTrue(torch.equal(left, both))


if __name__ == "__main__":
    unittest.main()
