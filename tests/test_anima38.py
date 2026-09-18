"""Anima 3.8B (Qwen3.5 / v2) 편입 — 번들 판별·경로·스크립트 인자·실패 시 순정 유지."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
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


def _load_script():
    """Forge 없이 scripts/anima_3_8b.py 를 로드한다 (modules 스텁)."""
    modules_stub = types.ModuleType("modules")

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
        self.assertIsNone(runtime._v2_sampling_unet)
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


class _FakeRuntime:
    def __init__(self, v2: bool, fail: Exception | None = None):
        self.v2 = v2
        self.fail = fail
        self.installs: list[tuple] = []
        self.restores = 0

    def is_v2_bundle(self, sd_model):
        return self.v2

    def install(self, p, adapter, strength, negative_strength):
        if self.fail is not None:
            raise self.fail
        self.installs.append((adapter, strength, negative_strength))

    def restore(self, p):
        self.restores += 1


class ScriptArgTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_script()

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
    def __init__(self, native_adapter, connector):
        super().__init__()
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
