"""TSM LoRA 스크립트(scripts/lora_tsm.py) — 인자 변환, infotext 붙여 넣기, XYZ 축, UI 인자 순서, 샘플링 패스마다의 적용과
뒷정리(가짜 UnetPatcher + 스텁 DiT), 프롬프트 쪽·아코디언 파일 합치기.

Forge 모듈은 sys.modules 스텁으로 넣는다(test_dora_infer_mode 와 같은 방식).
"""
from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext import anima_lora_blocks as alb  # noqa: E402
from sam3ext.tsm_lora import delta as td  # noqa: E402
from sam3ext.tsm_lora import fileformat as tf  # noqa: E402
from sam3ext.tsm_lora import runtime as tr  # noqa: E402
from tests.test_tsm_lora import (  # noqa: E402
    ForgeLikeLinear,
    StubDiT,
    _save_safetensors,
    _SysModules,
    forge_like,
    lokr_meta,
    oracle_delta,
    oracle_lokr_rows,
    stage_meta,
    stub_lokr_sd,
    stub_tsm_sd,
)

HAS_GRADIO = importlib.util.find_spec("gradio") is not None


def _load_script():
    modules_stub = types.ModuleType("modules")

    class Script:
        def elem_id(self, item_id):
            return f"script_{item_id}"

    callbacks = {"before_ui": [], "unloaded": []}
    modules_stub.scripts = types.SimpleNamespace(Script=Script, AlwaysVisible=object(), scripts_data=[])
    modules_stub.script_callbacks = types.SimpleNamespace(
        on_before_ui=callbacks["before_ui"].append,
        on_script_unloaded=callbacks["unloaded"].append,
    )
    modules_stub.shared = types.SimpleNamespace(sd_model=None, cmd_opts=types.SimpleNamespace(lora_dir="", lora_dirs=[]))
    modules_stub.ui_components = types.SimpleNamespace(InputAccordion=None)
    with _SysModules(modules=modules_stub, modules__ui_components=modules_stub.ui_components,
                     modules__shared=modules_stub.shared):
        spec = importlib.util.spec_from_file_location("_test_tsm_lora_script", ROOT / "scripts" / "lora_tsm.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module, modules_stub, callbacks


class FakeUnet:
    """Forge UnetPatcher 의 쓰는 부분만: clone(model_options 의 dict 는 깊은 복사, 객체는 참조)."""

    def __init__(self, dit, multiplier=1.0):
        self.model = types.SimpleNamespace(diffusion_model=dit, predictor=types.SimpleNamespace(multiplier=multiplier))
        self.model_options = {"transformer_options": {}}
        self.extra_preserved_memory_during_sampling = 0
        self.clones = 0

    def clone(self):
        n = FakeUnet.__new__(FakeUnet)
        n.model = self.model
        n.model_options = {"transformer_options": dict(self.model_options["transformer_options"])}
        n.extra_preserved_memory_during_sampling = self.extra_preserved_memory_during_sampling
        n.clones = self.clones + 1
        return n

    def set_transformer_option(self, k, v):
        self.model_options["transformer_options"][k] = v

    def add_extra_preserved_memory_during_sampling(self, memory_in_bytes):
        self.extra_preserved_memory_during_sampling += memory_in_bytes


@unittest.skipUnless(HAS_GRADIO, "gradio is not installed")
class ScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod, cls.stub, cls.callbacks = _load_script()

    def setUp(self):
        torch.manual_seed(0)
        self.dit = StubDiT(blocks=2).double()
        for param in self.dit.parameters():
            param.requires_grad_(False)
        self.sd = stub_tsm_sd(self.dit, n_scales=(8, 1), rank=2, seed=5, zero_router=True)  # 실제 1단계 파일처럼
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name, "smoke_tsm.safetensors"))
        _save_safetensors(self.path, self.sd, stage_meta(1))
        self.sd2 = stub_tsm_sd(self.dit, n_scales=(4, 2, 1), rank=2, seed=9)
        self.path2 = str(Path(self.tmp.name, "second_tsm.safetensors"))
        _save_safetensors(self.path2, self.sd2, stage_meta(2))
        self.mod._LISTING.clear()
        self.mod._LISTING.update({"smoke_tsm": self.path, "second_tsm": self.path2})
        self.logs = []
        self._log = self.mod._log
        self.mod._log = self.logs.append
        tr.set_logger(self.logs.append)
        self._notify = alb._notify_ui
        self.notified = []
        alb._notify_ui = lambda message, info=False: self.notified.append(message)
        tf.release_cache()

    def tearDown(self):
        tr.finish_generation()
        tf.release_cache()
        alb._notify_ui = self._notify
        self.mod._log = self._log
        tr.set_logger(None)
        self.mod._LAST["state"] = None
        self.tmp.cleanup()

    def _p(self, unet=None, inner=False):
        unet = unet or FakeUnet(self.dit)
        model = types.SimpleNamespace(current_lora_hash="[]", forge_objects=types.SimpleNamespace(unet=unet))
        p = types.SimpleNamespace(sd_model=model, extra_generation_params={})
        if inner:
            p._sam3_inner = True
        return p

    def _run(self, p, *args):
        self.stub.shared.sd_model = p.sd_model
        with _SysModules(modules=self.stub, modules__shared=self.stub.shared):
            script = self.mod.TsmLora()
            script.process(p, *args)
            script.process_before_every_sampling(p, *args)
        return p.sd_model.forge_objects.unet

    def _postprocess(self, p):
        with _SysModules(modules=self.stub, modules__shared=self.stub.shared):
            self.mod.TsmLora().postprocess(p, None)

    def _forward(self, unet, sigma=0.3):
        x = torch.randn(2, 1, 2, 3, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
        ctx = torch.randn(2, 5, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
        options = copy.copy(unet.model_options["transformer_options"])
        return self.dit(x, torch.tensor([sigma, sigma], dtype=torch.float32), context=ctx, transformer_options=options)

    # 인자

    def test_coerce_args_positional_dict_and_defaults(self):
        m = self.mod
        self.assertEqual(m.ARG_NAMES, ("enabled", "lora", "strength", "stage", "pooling", "lokr_merge_cache"))
        default = m.coerce_args(())
        self.assertEqual(default, {"enabled": False, "lora": "", "strength": 1.0, "stage": "auto", "pooling": "sample",
                                   "lokr_merge_cache": True})
        positional = m.coerce_args((True, "smoke_tsm", 0.8, m.LABEL_STAGE_2, m.LABEL_POOL_BATCH))
        self.assertEqual(positional, {"enabled": True, "lora": "smoke_tsm", "strength": 0.8, "stage": "2", "pooling": "batch",
                                      "lokr_merge_cache": True})
        api = m.coerce_args(({"enabled": "true", "lora": "없음", "strength": "9", "stage": "off", "lokr_merge_cache": "true"},))
        self.assertEqual(api, {"enabled": True, "lora": "", "strength": 3.0, "stage": "off", "pooling": "sample",
                               "lokr_merge_cache": True})
        self.assertTrue(m.coerce_args((True, "x", 1.0, "auto", "sample", True))["lokr_merge_cache"])
        self.assertFalse(m.coerce_args((True, "x", 1.0, "auto", "sample", False))["lokr_merge_cache"])
        # 아코디언이 꺼져 있으면(프롬프트 <lora:...> 만) 병합 캐시는 기본값(켬) — 아코디언의 끔은 아코디언을 켰을 때만
        self.assertTrue(m.resolve_state(self._p(), (False, "x", 1.0, "auto", "sample", False))["lokr_merge_cache"])
        self.assertFalse(m.resolve_state(self._p(), (True, "x", 1.0, "auto", "sample", False))["lokr_merge_cache"])
        # infotext 붙여넣기: 키가 없으면 기본값(켬), "off" 면 끔
        self.assertTrue(m._paste_lokr_cache({m.INFOTEXT_LORA_KEY: "x:1"}))
        self.assertFalse(m._paste_lokr_cache({m.INFOTEXT_LORA_KEY: "x:1", m.INFOTEXT_LOKR_CACHE_KEY: "off"}))
        self.assertEqual(m.coerce_args((True, "x", "nan"))["strength"], 1.0)

    def test_resolve_state_xyz_wins_and_disabled_resets_overrides(self):
        m = self.mod
        p = self._p()
        state = m.resolve_state(p, (False, "smoke_tsm", 0.5, m.LABEL_STAGE_1, m.LABEL_POOL_BATCH))
        self.assertEqual((state["stage"], state["pooling"]), ("auto", "sample"))
        setattr(p, m._XYZ_STAGE_ATTR, "2")
        setattr(p, m._XYZ_POOL_ATTR, "batch")
        setattr(p, m._XYZ_STRENGTH_ATTR, 0.25)
        state = m.resolve_state(p, (False, "smoke_tsm", 0.5))
        self.assertEqual((state["stage"], state["pooling"], state["strength"]), ("2", "batch", 0.25))

    # infotext

    def test_paste_round_trip(self):
        m = self.mod
        full = {"TSM LoRA": "smoke_tsm:0.8", "TSM stage": "2", "TSM pooling": "batch", "TSM LoRA info": "x"}
        self.assertTrue(m._paste_enabled(full))
        self.assertEqual(m._paste_lora(full), "smoke_tsm")
        self.assertEqual(m._paste_strength(full), 0.8)
        self.assertEqual(m._paste_stage(full), m.LABEL_STAGE_2)
        self.assertEqual(m._paste_pooling(full), m.LABEL_POOL_BATCH)
        only_stage = {"TSM stage": "off"}
        self.assertTrue(m._paste_enabled(only_stage))
        self.assertEqual(m._paste_lora(only_stage), m.NONE_CHOICE)
        self.assertEqual(m._paste_stage(only_stage), m.LABEL_STAGE_OFF)
        self.assertIsNone(m._paste_strength(only_stage))
        prompt_only = {"TSM LoRA info": "smoke_tsm s1 8+1 r2 x1"}  # 프롬프트 쪽만 쓴 이미지 — 아코디언은 꺼진 채
        self.assertFalse(m._paste_enabled(prompt_only))
        self.assertIsNone(m._paste_lora(prompt_only))
        self.assertIsNone(m._paste_stage(prompt_only))
        self.assertIsNone(m._paste_lokr_cache(prompt_only))
        self.assertTrue(m._paste_lokr_cache({"TSM LoKr merge cache": "on"}))
        self.assertFalse(m._paste_lokr_cache({"TSM LoKr merge cache": "off"}))
        self.assertTrue(m._paste_lokr_cache(full))  # 키가 없으면 기본값(켬, 10-24)
        self.assertEqual(m._split_lora_value("dir/a:b:1.5"), ("dir/a:b", 1.5))
        self.assertEqual(m._split_lora_value("plain"), ("plain", None))

    def test_ui_returns_all_arguments_in_order(self):
        import gradio as gr

        m = self.mod
        script = m.TsmLora()
        with gr.Blocks():
            components = script.ui(False)
        self.assertEqual(len(components), len(m.ARG_NAMES))
        self.assertEqual([c for c, _ in script.infotext_fields], components)
        self.assertEqual(components[1].value, m.NONE_CHOICE)
        self.assertEqual(components[2].value, 1.0)
        self.assertEqual(components[3].value, m.LABEL_STAGE_AUTO)
        self.assertEqual(components[4].value, m.LABEL_POOL_SAMPLE)
        self.assertIs(components[5].value, True)  # TSM-LoKr 병합 캐시 기본 켬(10-24)

    def test_xyz_axes_are_registered_once_with_a_low_cost(self):
        m = self.mod
        created = []

        class AxisOption:
            def __init__(self, label, type, apply, format_value=None, confirm=None, cost=0.0, choices=None, prepare=None):
                self.label, self.cost, self.choices = label, cost, choices
                created.append(self)

        fake = types.SimpleNamespace(axis_options=[], AxisOption=AxisOption)
        entry = types.SimpleNamespace(script_class=types.SimpleNamespace(__module__="xyz_grid.py"), module=fake)
        original = self.stub.scripts.scripts_data
        self.stub.scripts.scripts_data = [entry]
        try:
            m._make_xyz_axis()
            m._make_xyz_axis()
        finally:
            self.stub.scripts.scripts_data = original
        labels = sorted(a.label for a in fake.axis_options)
        self.assertEqual(labels, sorted([m.XYZ_STAGE_LABEL, m.XYZ_POOL_LABEL, m.XYZ_STRENGTH_LABEL]))
        stage_axis = next(a for a in fake.axis_options if a.label == m.XYZ_STAGE_LABEL)
        self.assertEqual(stage_axis.choices(), ["auto", "1", "2", "off"])
        for axis in fake.axis_options:
            self.assertTrue(0 < axis.cost < 0.7)
            self.assertTrue(all(ord(ch) < 128 for ch in axis.label))  # 범례 글꼴

    def test_registers_callbacks(self):
        self.assertIn(self.mod._on_unloaded, self.callbacks["unloaded"])
        self.assertIn(self.mod._install_loader, self.callbacks["before_ui"])
        self.assertIn(self.mod._on_before_ui_xyz, self.callbacks["before_ui"])

    # 적용

    def test_picker_file_is_applied_for_the_pass_and_removed_afterwards(self):
        m = self.mod
        base = self._forward(FakeUnet(self.dit))
        p = self._p()
        original_unet = p.sd_model.forge_objects.unet
        unet = self._run(p, True, "smoke_tsm", 0.8)
        self.assertIsNot(unet, original_unet)
        session = unet.model_options["transformer_options"][tr.OPTION_KEY]
        self.assertIsInstance(session, tr.TsmSession)
        self.assertNotIn(tr.OPTION_KEY, original_unet.model_options["transformer_options"])
        self.assertEqual(len(session.bound), 32)
        self.assertGreater(unet.extra_preserved_memory_during_sampling, 0)
        self.assertEqual(p.extra_generation_params[m.INFOTEXT_LORA_KEY], "smoke_tsm:0.8")
        self.assertEqual(p.extra_generation_params[m.INFOTEXT_INFO_KEY], "smoke_tsm s1 8+1 r2 x0.8")
        self.assertNotIn(m.INFOTEXT_STAGE_KEY, p.extra_generation_params)
        on = self._forward(unet)
        self.assertGreater((on - base).abs().max().item(), 1e-4)
        # 한 Linear 를 직접 대조
        layer = self.dit.blocks[0].mlp.layer1
        x = torch.randn(2, 3, 16, dtype=torch.float64)
        got = session.delta_for(id(layer), x, td.timestep_values(torch.tensor([0.3, 0.3])))
        expected = oracle_delta(self.sd, "lora_unet_blocks_0_mlp_layer1", x, [300, 300], 1, 0.8)
        torch.testing.assert_close(got, expected)
        self._postprocess(p)
        self.assertEqual(tr.HOOKS.counts(), (0, 0))
        torch.testing.assert_close(self._forward(unet), base, rtol=0, atol=0)
        self.assertTrue(any("DiT 호출 1번" in line for line in self.logs))

    def test_tsm_lokr_picker_file_uses_forge_weight_functions_and_the_merge_cache(self):
        m = self.mod
        dit = forge_like(self.dit)
        sd = stub_lokr_sd(dit, seed=7)
        path = str(Path(self.tmp.name, "lokr_tsm.safetensors"))
        _save_safetensors(path, sd, lokr_meta(1))
        m._LISTING["lokr_tsm"] = path
        for merge_cache in (False, True):
            p = self._p(FakeUnet(dit))
            unet = self._run(p, True, "lokr_tsm", 0.8, m.LABEL_STAGE_AUTO, m.LABEL_POOL_SAMPLE, merge_cache)
            session = unet.model_options["transformer_options"][tr.OPTION_KEY]
            self.assertEqual((len(session.lokr_slots), session.lokr_merge_cache), (32, merge_cache))
            self.assertEqual(p.extra_generation_params[m.INFOTEXT_INFO_KEY], "lokr_tsm s1 8+1 lokr f4 dora x0.8")
            self.assertEqual(p.extra_generation_params.get(m.INFOTEXT_LOKR_CACHE_KEY), None if merge_cache else "off")
            reserved = unet.extra_preserved_memory_during_sampling
            self.assertGreaterEqual(reserved, session.fused_bytes(2))
            outputs = {}
            hooks = [mod.register_forward_hook(lambda mod, a, o, n=n: outputs.__setitem__(n, (a[0].clone(), o.clone())))
                     for n, mod in dit.named_modules() if isinstance(mod, ForgeLikeLinear)]
            try:
                self._forward(unet)
            finally:
                for h in hooks:
                    h.remove()
            inp, out = outputs["blocks.0.mlp.layer1"]
            want = oracle_lokr_rows(sd, "lora_unet_blocks_0_mlp_layer1", dit.blocks[0].mlp.layer1.weight, inp, [300, 300], 1, 0.8)
            torch.testing.assert_close(out, want, rtol=1e-10, atol=1e-10)
            self.assertTrue(any("lokr f4 dora" in line for line in self.logs))
            self.assertTrue(any("32 modules via weight_function" in line for line in self.logs))
            self._postprocess(p)
            self.assertFalse(any("weight_function" in mod.__dict__ for mod in dit.modules() if isinstance(mod, ForgeLikeLinear)))
            self.assertEqual(tr.HOOKS.pre_hook_count(), 0)

    def test_prompt_records_and_picker_combine_and_duplicates_use_the_prompt(self):
        m = self.mod
        p = self._p()
        tsm2 = tf.load_tsm_file(self.path2)
        tr.record_prompt_tsm(p.sd_model, self.path2, 0.6, tsm2)
        unet = self._run(p, True, "smoke_tsm", 1.0)
        session = unet.model_options["transformer_options"][tr.OPTION_KEY]
        self.assertEqual([(e.label, e.source, e.stage, e.multiplier) for e in session.entries],
                         [("second_tsm", "prompt", 2, 0.6), ("smoke_tsm", "picker", 1, 1.0)])
        self.assertEqual(p.extra_generation_params[m.INFOTEXT_INFO_KEY],
                         "second_tsm s2 4+2+1 r2 x0.6; smoke_tsm s1 8+1 r2 x1")
        tr.finish_generation()
        # 아코디언에서 같은 파일을 고르면 프롬프트 쪽만
        p = self._p()
        tr.record_prompt_tsm(p.sd_model, self.path2, 0.6, tsm2)
        unet = self._run(p, True, "second_tsm", 1.0)
        session = unet.model_options["transformer_options"][tr.OPTION_KEY]
        self.assertEqual([e.source for e in session.entries], ["prompt"])
        self.assertNotIn(m.INFOTEXT_LORA_KEY, p.extra_generation_params)
        self.assertTrue(any("프롬프트" in line for line in self.logs))

    def test_prompt_tsm_applies_with_the_accordion_off_and_stage_off_skips(self):
        m = self.mod
        p = self._p()
        tr.record_prompt_tsm(p.sd_model, self.path, 1.0, tf.load_tsm_file(self.path))
        unet = self._run(p, False, "없음")
        self.assertIn(tr.OPTION_KEY, unet.model_options["transformer_options"])
        tr.finish_generation()
        p = self._p()
        tr.record_prompt_tsm(p.sd_model, self.path, 1.0, tf.load_tsm_file(self.path))
        original = p.sd_model.forge_objects.unet
        unet = self._run(p, True, "없음", 1.0, m.LABEL_STAGE_OFF)
        self.assertIs(unet, original)
        self.assertEqual(p.extra_generation_params[m.INFOTEXT_STAGE_KEY], "off")
        self.assertEqual(tr.HOOKS.counts(), (0, 0))

    def test_zero_strength_hooks_nothing_and_post_sample_frees_the_device_cache(self):
        base = self._forward(FakeUnet(self.dit))
        p = self._p()
        tr.record_prompt_tsm(p.sd_model, self.path2, 0.0, tf.load_tsm_file(self.path2))
        original = p.sd_model.forge_objects.unet
        unet = self._run(p, True, "smoke_tsm", 0.0)
        self.assertIs(unet, original)                      # 강도 0 = 꺼 둔 것과 같다(훅도 복제본도 없다)
        self.assertEqual(tr.HOOKS.counts(), (0, 0))
        torch.testing.assert_close(self._forward(unet), base, rtol=0, atol=0)
        # 한쪽만 0 이면 나머지만 건다
        p = self._p()
        tr.record_prompt_tsm(p.sd_model, self.path2, 0.0, tf.load_tsm_file(self.path2))
        unet = self._run(p, True, "smoke_tsm", 0.5)
        session = unet.model_options["transformer_options"][tr.OPTION_KEY]
        self.assertEqual([(e.label, e.multiplier) for e in session.entries], [("smoke_tsm", 0.5)])
        self.assertEqual(session.targets, {})               # 훅을 건 뒤 DiT 모듈을 쥐지 않는다
        self._forward(unet)
        self.assertGreater(session.cache.nbytes(), 0)
        with _SysModules(modules=self.stub, modules__shared=self.stub.shared):
            self.mod.TsmLora().post_sample(p, types.SimpleNamespace(samples=None))
        self.assertEqual(session.cache.nbytes(), 0)
        self.assertEqual(tr.HOOKS.counts(), (1, 32))        # 훅은 postprocess 까지
        self._postprocess(p)
        self.assertEqual(tr.HOOKS.counts(), (0, 0))

    def _forge_pass(self, p, sampler, latent, *args, process=True):
        """processing 순서: (process) → process_before_every_sampling(x=latent) → sampler.sample(launch_sampling) → 디코드."""
        self.stub.shared.sd_model = p.sd_model
        p.sampler = sampler
        with _SysModules(modules=self.stub, modules__shared=self.stub.shared):
            script = self.mod.TsmLora()
            if process:
                script.process(p, *args)
            script.process_before_every_sampling(p, *args, x=latent, noise=latent, c=None, uc=None)
        unet = p.sd_model.forge_objects.unet
        session = unet.model_options["transformer_options"][tr.OPTION_KEY]
        sampler.step = lambda i: self._forward(unet, sigma=(0.9, 0.5, 0.1)[i % 3])
        sampler.sample(p)
        return unet, session

    def test_each_pass_releases_its_device_copy_before_the_vae_decode(self):
        from tests.test_tsm_lora import FakeForgeSampler

        log = []
        p = self._p()
        latent = torch.zeros(1, 16, 8, 8)
        first_sampler = FakeForgeSampler(None, log)
        unet, first = self._forward_pass_checked(p, first_sampler, latent, log)
        # 예약 = 장치 캐시 + Linear 호출 하나의 임시(배치 2 · 4×4 토큰 = 32 행, 중간값 4 + 가장 큰 출력 48)
        self.assertEqual(unet.extra_preserved_memory_during_sampling, first.fused_bytes(2) + 32 * (4 + 48) * 2)
        self.assertEqual(self.mod.sampling_tokens(p, latent, self.dit), 32)
        # 하이레스 패스(새 샘플러 · 더 큰 latent): 다시 만들고, 하이레스 디코드(post_sample 보다 먼저) 전에 다시 놓는다
        p.sd_model.forge_objects.unet = FakeUnet(self.dit)
        hires_sampler = FakeForgeSampler(None, log)
        unet, hires = self._forward_pass_checked(p, hires_sampler, torch.zeros(1, 16, 12, 12), log, process=False)
        self.assertIsNot(hires, first)
        self.assertEqual(unet.extra_preserved_memory_during_sampling, hires.fused_bytes(2) + 2 * 36 * (4 + 48) * 2)
        self.assertTrue(getattr(first_sampler.launch_sampling, "_sam3_tsm_release", False))
        self.assertTrue(getattr(hires_sampler.launch_sampling, "_sam3_tsm_release", False))
        self._postprocess(p)

    def _forward_pass_checked(self, p, sampler, latent, log, process=True):
        unet, session = self._forge_pass(p, sampler, latent, True, "smoke_tsm", 1.0, process=process)
        self.assertEqual(log[-1], ("sampled", "latent"))
        self.assertGreater(session.calls, 0)
        # 여기가 processing 의 decode_latent_batch 자리 — post_sample 은 아직 불리지 않았다
        self.assertEqual(session.cache.nbytes(), 0)
        self.assertFalse(any(s.holds_device_tensors() for slots in session.slots.values() for s in slots))
        return unet, session

    def test_img2img_batch_count_reuses_the_wrapped_sampler(self):
        from tests.test_tsm_lora import FakeForgeSampler

        log = []
        sampler = FakeForgeSampler(None, log)   # img2img 는 init 에서 만든 샘플러를 batch count 내내 쓴다
        for n in range(2):
            p = self._p()
            unet, session = self._forward_pass_checked(p, sampler, torch.zeros(1, 16, 8, 8), log)
            self.assertFalse(getattr(sampler.launch_sampling.__wrapped__, "_sam3_tsm_release", False))
            self._postprocess(p)

    def test_stage_override_and_pooling_reach_the_session(self):
        m = self.mod
        p = self._p()
        unet = self._run(p, True, "smoke_tsm", 1.0, m.LABEL_STAGE_2, m.LABEL_POOL_BATCH)
        session = unet.model_options["transformer_options"][tr.OPTION_KEY]
        self.assertEqual((session.entries[0].stage, session.pooling), (2, "batch"))
        self.assertEqual(p.extra_generation_params[m.INFOTEXT_STAGE_KEY], "2")
        self.assertEqual(p.extra_generation_params[m.INFOTEXT_POOL_KEY], "batch")
        self.assertTrue(any("문맥 전문가가 빠집니다" in line for line in self.logs))

    def test_non_anima_model_and_missing_file_are_skipped_with_a_log(self):
        p = self._p(unet=FakeUnet(torch.nn.Sequential(torch.nn.Linear(2, 2))))
        original = p.sd_model.forge_objects.unet
        self.assertIs(self._run(p, True, "smoke_tsm", 1.0), original)
        self.assertTrue(any("Anima 가 아니라" in line for line in self.logs))
        p = self._p()
        original = p.sd_model.forge_objects.unet
        self.assertIs(self._run(p, True, "does_not_exist", 1.0), original)
        self.assertTrue(any("찾지 못했습니다" in line for line in self.logs))
        self.assertEqual(p.extra_generation_params, {})

    def test_broken_picker_file_notifies(self):
        bad = dict(self.sd)
        bad["lora_unet_blocks_0_mlp_layer1.router_fc.weight"] = torch.zeros(1, 3)
        path = str(Path(self.tmp.name, "bad_tsm.safetensors"))
        _save_safetensors(path, bad, stage_meta(1))
        self.mod._LISTING["bad_tsm"] = path
        p = self._p()
        original = p.sd_model.forge_objects.unet
        self.assertIs(self._run(p, True, "bad_tsm", 1.0), original)
        self.assertEqual(len(self.notified), 1)
        self.assertIn("router_fc", self.notified[0])

    def test_inner_pass_reuses_the_outer_state(self):
        m = self.mod
        outer = self._p()
        self._run(outer, True, "smoke_tsm", 0.5, m.LABEL_STAGE_2)
        inner = self._p(inner=True)
        unet = self._run(inner)  # 내부 패스는 인자가 없다
        session = unet.model_options["transformer_options"][tr.OPTION_KEY]
        self.assertEqual((session.entries[0].multiplier, session.entries[0].stage), (0.5, 2))

    def test_hires_pass_keeps_the_first_pass_infotext(self):
        m = self.mod
        p = self._p()
        tr.record_prompt_tsm(p.sd_model, self.path2, 0.6, tf.load_tsm_file(self.path2))
        self._run(p, False)
        first = p.extra_generation_params[m.INFOTEXT_INFO_KEY]
        p.sd_model.current_lora_hash = "hires"  # 하이레스는 다른 LoRA 묶음 — TSM 없음
        p.sd_model.forge_objects.unet = FakeUnet(self.dit)
        with _SysModules(modules=self.stub, modules__shared=self.stub.shared):
            self.mod.TsmLora().process_before_every_sampling(p)
        self.assertEqual(p.extra_generation_params[m.INFOTEXT_INFO_KEY], first)
        self.assertNotIn(tr.OPTION_KEY, p.sd_model.forge_objects.unet.model_options["transformer_options"])

    def test_process_cleans_a_previous_crashed_generation_and_xyz_cells(self):
        m = self.mod
        p = self._p()
        self._run(p, True, "smoke_tsm", 1.0)
        self.assertEqual(tr.HOOKS.counts()[0], 1)
        p.extra_generation_params[m.INFOTEXT_STAGE_KEY] = "2"  # 앞 XYZ 칸이 남긴 키
        with _SysModules(modules=self.stub, modules__shared=self.stub.shared):
            self.mod.TsmLora().process(p, False)
        self.assertEqual(tr.HOOKS.counts(), (0, 0))
        self.assertNotIn(m.INFOTEXT_STAGE_KEY, p.extra_generation_params)
        self.assertNotIn(m.INFOTEXT_INFO_KEY, p.extra_generation_params)

    def test_unload_hook_releases_everything(self):
        p = self._p()
        self._run(p, True, "smoke_tsm", 1.0)
        self.assertTrue(tr.release_all())
        self.assertEqual(tr.HOOKS.counts(), (0, 0))
        self.assertIsNone(tf.cached_tsm_file(self.path))
        self.assertFalse(tr.release_all())

    def test_listing_uses_forge_metadata(self):
        m = self.mod
        nets = {
            "smoke_tsm": types.SimpleNamespace(filename=self.path, metadata={"ss_saved_format": "tsm-native"}),
            "plain": types.SimpleNamespace(filename=self.path2, metadata={"ss_network_module": "networks.lora"}),
            "nometa": types.SimpleNamespace(filename=self.path2, metadata={}),
        }
        fake = types.ModuleType("networks")
        fake.__file__ = "/forge/extensions-builtin/sd_forge_lora/networks.py"
        fake.process_anima = lambda lora, blocks: True
        fake.load_lora_for_models = lambda *a, **k: None
        fake.available_networks = nets
        original = alb._find_forge_lora_module
        alb._find_forge_lora_module = lambda: fake
        try:
            found = m.list_tsm_loras()
        finally:
            alb._find_forge_lora_module = original
        self.assertEqual(found, {"nometa": self.path2, "smoke_tsm": self.path})
        self.assertEqual(m._choices(), [m.NONE_CHOICE, "nometa", "smoke_tsm"])


if __name__ == "__main__":
    unittest.main()
