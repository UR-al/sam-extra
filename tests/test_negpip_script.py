"""내장 NegPiP 스크립트(scripts/negpip.py)의 계약·공존 가드·emphasis 게이트.

- 계약: 제목 "NegPiP", AlwaysVisible, ui() None (UR_IV 앱과 inpaint_core 가 제목·인자 0개로 다룬다).
- 공존: 따로 설치된 sd-forge-negpip 가 로드돼 있으면(스크립트 객체·패치 흔적) 내장은 쉬고 경고는 한 번.
- Anima: 엔진이 음수 가중치를 곱하지 않는 emphasis(새 엔진 None/Ignore, 옛 엔진 None)면 패치하지 않는다.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sam3ext.negpip as negpip_pkg  # noqa: E402
from sam3ext.negpip import coexist  # noqa: E402

SCRIPT = ROOT / "scripts" / "negpip.py"
ALWAYS_VISIBLE = object()


class _Script:
    def __init__(self):
        pass


def _load_script():
    """Forge 없이 스크립트를 불러온다 — modules·훅 모듈만 대역, sam3ext.negpip 의 나머지는 실제."""
    calls = []

    def patch_anima_negpip(cls, *, unpatch=False):   # 상류와 같은 플래그 규칙
        if unpatch != cls._patched[1]:
            return
        cls._patched[1] = not cls._patched[1]
        calls.append(("anima", unpatch))

    def patch_sd_negpip(instance, cls, *, unpatch=False):
        if unpatch != cls._patched[0]:
            return
        cls._patched[0] = not cls._patched[0]
        calls.append(("sd", unpatch))

    modules = types.ModuleType("modules")
    modules.__path__ = []
    modules.scripts = types.SimpleNamespace(Script=_Script, AlwaysVisible=ALWAYS_VISIBLE)
    modules.shared = types.SimpleNamespace(opts=types.SimpleNamespace(emphasis="Original"))
    prompt_parser = types.ModuleType("modules.prompt_parser")
    prompt_parser.SdConditioning = list
    prompt_parser.get_learned_conditioning = None
    prompt_parser.get_learned_conditioning_prompt_schedules = None
    callbacks = types.ModuleType("modules.script_callbacks")
    callbacks.CFGDenoiserParams = object
    callbacks.on_cfg_denoiser = lambda callback: None
    anima = types.ModuleType("sam3ext.negpip.anima")
    anima.patch_anima_negpip = patch_anima_negpip
    sd = types.ModuleType("sam3ext.negpip.sd")
    sd.patch_sd_negpip = patch_sd_negpip
    emphasis = types.ModuleType("backend.text_processing.emphasis")
    known = {"None", "Ignore", "Original", "No norm"}
    emphasis.get_current_option = lambda name: types.SimpleNamespace(name=name if name in known else "Original")
    backend = types.ModuleType("backend")
    backend.__path__ = []
    text_processing = types.ModuleType("backend.text_processing")
    text_processing.__path__ = []
    text_processing.emphasis = emphasis
    stubs = {
        "modules": modules, "modules.scripts": modules.scripts, "modules.shared": modules.shared,
        "modules.prompt_parser": prompt_parser, "modules.script_callbacks": callbacks,
        "sam3ext.negpip.anima": anima, "sam3ext.negpip.sd": sd,
        "backend": backend, "backend.text_processing": text_processing,
        "backend.text_processing.emphasis": emphasis,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("_test_builtin_negpip", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    module.IS_NEO = True   # 이 PC 의 Forge Neo 와 같게 (테스트 프로세스엔 backend 가 없어 패키지는 False 로 읽는다)
    module._test_calls = calls
    module._test_stubs = stubs
    return module


class _NewEngine:
    """Qwen06Engine 대역 (tokenize_line 없음). 진짜처럼 T5 토크나이저를 쥔다 — 공용 속성 이름에서 Anima 엔진을 가르는 표시."""

    emphasis = types.SimpleNamespace(name="Original")
    t5_tokenizer = object()


class _QwenOnlyEngine:
    """Z-Image·Flux2 처럼 text_processing_engine_qwen 를 같이 쓰는 엔진 대역 — T5 토크나이저가 없다."""

    emphasis = types.SimpleNamespace(name="Original")


class _OldEngine:
    """AnimaTextProcessingEngine 대역."""

    def tokenize_line(self, line):
        return []


class _DiT:
    def __init__(self, modules=()):
        self._modules_list = list(modules)

    def named_modules(self):
        return [(f"m{i}", m) for i, m in enumerate(self._modules_list)]


def _anima_model(engine, dit=None, attr="text_processing_engine_anima"):
    """attr: Forge ~2.29.1 은 text_processing_engine_anima, 2.29.2~ 는 text_processing_engine_qwen."""
    model = type("Anima", (), {})()
    model.is_webui_legacy_model = lambda: False
    model.is_sdxl = False
    setattr(model, attr, engine)
    model.forge_objects = types.SimpleNamespace(
        unet=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=dit or _DiT())),
    )
    return model


class _Foreign:
    """따로 설치된 sd-forge-negpip 의 스크립트 객체 대역."""

    filename = r"C:\forge\extensions\sd-forge-negpip\scripts\negpip.py"
    args_from = 0

    def title(self):
        return "NegPiP"


class _Other:
    args_from = 0

    def __init__(self, title="Other"):
        self._title = title

    def title(self):
        return self._title


class ScriptTestCase(unittest.TestCase):
    def setUp(self):
        self.module = _load_script()
        self.shared = self.module._test_stubs["modules.shared"]
        self._saved_patched = list(negpip_pkg.PATCHED)
        negpip_pkg.PATCHED[:] = [False, False]
        self.addCleanup(self._restore)
        coexist._warned = False
        self.script = self.module.NegPiP()

    def _restore(self):
        negpip_pkg.PATCHED[:] = self._saved_patched
        coexist._warned = False

    def _p(self, *, engine=None, others=(), prompts=("girl, (bad:-1)",), model=None):
        scripts = [self.script, *others]
        return types.SimpleNamespace(
            sd_model=model or _anima_model(engine or _NewEngine()),
            scripts=types.SimpleNamespace(alwayson_scripts=scripts, scripts=scripts),
            script_args=[None],
            prompts=list(prompts),
            negative_prompts=[""],
            extra_generation_params={},
            cached_c=[1, 2, 3], cached_uc=[1, 2, 3],
        )

    def _run(self, p):
        out = io.StringIO()
        backend = {k: v for k, v in self.module._test_stubs.items() if k.startswith("backend")}
        with contextlib.redirect_stdout(out), mock.patch.dict(sys.modules, backend):
            self.script.process_batch(p)
        return out.getvalue()


class ScriptContractTests(ScriptTestCase):
    def test_title_visibility_and_ui_match_the_standalone(self):
        self.assertEqual(self.script.title(), "NegPiP")
        self.assertIs(self.script.show(False), ALWAYS_VISIBLE)
        self.assertIs(self.script.show(True), ALWAYS_VISIBLE)
        self.assertIsNone(self.script.ui(False))
        self.assertIsNone(self.script.ui(True))

    def test_instance_is_marked_builtin_and_shares_the_package_patch_state(self):
        self.assertTrue(getattr(self.script, coexist.BUILTIN_ATTR))
        self.assertIs(self.module.NegPiP._patched, negpip_pkg.PATCHED)
        # 다시 불러와도(Reload UI) 같은 목록 — 패치 상태를 잊지 않는다
        self.assertIs(_load_script().NegPiP._patched, negpip_pkg.PATCHED)

    def test_patch_state_starts_unpatched(self):
        # 새로 불러온 패키지의 PATCHED 는 [SD, Anima] 모두 False — 아니면 첫 생성의 patch/unpatch 가 뒤집혀 훅이 안 걸린다
        spec = importlib.util.spec_from_file_location("_fresh_sam3ext_negpip", ROOT / "sam3ext" / "negpip" / "__init__.py")
        fresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh)
        self.assertEqual(fresh.PATCHED, [False, False])
        self.assertIsNot(fresh.PATCHED, negpip_pkg.PATCHED)

    def test_file_sorts_after_our_other_scripts(self):
        # Forge 는 확장 안 스크립트를 파일 이름순으로 돈다. 예전 sd-forge-negpip 는 폴더 이름순으로 우리 스크립트 전부의
        # 뒤였다 — 3.8B 런타임(anima_3_8b.py)이 검증된 순서(NegPiP 가 위)를 그대로 지키려고 마지막 자리를 지킨다.
        names = sorted(p.name for p in (ROOT / "scripts").glob("*.py"))
        self.assertIn("anima_3_8b.py", names)
        self.assertEqual(names[-1], SCRIPT.name, names)

    def test_file_stem_is_negpip_for_adetailer(self):
        # ADetailer 는 자기 패스에 넣을 always-on 스크립트를 파일 이름(stem)으로 고른다(!adetailer.py script_filter,
        # ad_only_selected_scripts 기본 켬). 기본 ad_script_names 에 'negpip' 이 있다 — 이름을 바꾸면 ADetailer 패스에서
        # NegPiP 가 빠져 (x:-1) 이 엔진에 음수 가중치로 그대로 들어간다.
        self.assertEqual(SCRIPT.stem, "negpip")
        args = ROOT.parents[0] / "aadetailer-neoforge" / "adetailer" / "args.py"
        if not args.is_file():
            self.skipTest(f"ADetailer 가 설치돼 있지 않다: {args}")
        import ast

        tree = ast.parse(args.read_text(encoding="utf-8"))
        default = next(
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_script_default" for t in node.targets)
        )
        self.assertIn(SCRIPT.stem, ast.literal_eval(default))

    def test_anima_generation_patches_and_records(self):
        p = self._p()
        self._run(p)
        self.assertEqual(self.module._test_calls, [("anima", False)])
        self.assertTrue(self.script.active)
        self.assertIs(p.extra_generation_params.get("NegPiP"), True)
        self.assertTrue(all(value is None for value in p.cached_c), "조건 캐시를 비운다(상류 그대로)")

    def test_no_negative_weight_does_nothing(self):
        p = self._p(prompts=("girl, (good:1.2)",))
        self._run(p)
        self.assertEqual(self.module._test_calls, [])
        self.assertNotIn("NegPiP", p.extra_generation_params)


class ForgeCoupleTests(ScriptTestCase):
    """상류 _verify_ext — Forge Couple 이 켜져 있으면 NegPiP 는 쉰다('NegPiP Disabled')."""

    def test_enabled_forge_couple_disables_negpip(self):
        p = self._p(others=[_Other("Forge Couple")])
        p.script_args = [True]
        out = self._run(p)
        self.assertIn("NegPiP Disabled", out)
        self.assertEqual(self.module._test_calls, [])
        self.assertFalse(self.script.active)
        self.assertNotIn("NegPiP", p.extra_generation_params)

    def test_disabled_forge_couple_does_not(self):
        p = self._p(others=[_Other("Forge Couple")])
        p.script_args = [False]
        out = self._run(p)
        self.assertNotIn("NegPiP Disabled", out)
        self.assertEqual(self.module._test_calls, [("anima", False)])


class CoexistenceGuardTests(ScriptTestCase):
    def test_standalone_script_makes_the_builtin_stand_down_with_one_warning(self):
        first, second = self._p(others=[_Foreign()]), self._p(others=[_Foreign()])
        out = self._run(first) + self._run(second)
        self.assertEqual(self.module._test_calls, [])
        self.assertFalse(self.script.active)
        self.assertNotIn("NegPiP", first.extra_generation_params)
        self.assertEqual(out.count("built-in NegPiP is standing down"), 1)
        self.assertIn("sd-forge-negpip", out)

    def test_no_warning_without_negative_weights(self):
        out = self._run(self._p(others=[_Foreign()], prompts=("girl",)))
        self.assertEqual(out, "")
        self.assertFalse(coexist._warned)

    def test_other_scripts_and_the_builtin_itself_are_not_foreign(self):
        broken = _Other()
        broken.title = lambda: 1 / 0
        runner = types.SimpleNamespace(alwayson_scripts=[self.script, _Other(), broken])
        self.assertEqual(coexist.foreign_scripts(runner), [])
        self._run(self._p(others=[_Other()]))
        self.assertEqual(self.module._test_calls, [("anima", False)])

    def test_runner_without_alwayson_list_falls_back_to_scripts(self):
        runner = types.SimpleNamespace(scripts=[_Foreign()])
        self.assertEqual(len(coexist.foreign_scripts(runner)), 1)
        self.assertEqual(coexist.foreign_scripts(None), [])

    def test_live_foreign_patch_on_the_model_makes_the_builtin_stand_down(self):
        cases = {
            "conditioning hook": lambda m: setattr(m, "orig_forward", object()),
            "dit forward hook": lambda m: setattr(m.forge_objects.unet.model.diffusion_model, "orig_forward", object()),
        }
        for name, apply in cases.items():
            with self.subTest(hook=name):
                self.module._test_calls.clear()
                coexist._warned = False
                model = _anima_model(_NewEngine())
                apply(model)
                out = self._run(self._p(model=model))
                self.assertEqual(self.module._test_calls, [])
                self.assertIn("another NegPiP patch is active", out)

    def test_module_level_patches_are_seen(self):
        attn2 = types.SimpleNamespace(orig_forward=object())   # SD attn2 인스턴스 훅
        cross = type("SelfCrossAttention", (), {"negpip_orig_forward": object()})()   # Anima 클래스 훅
        for module in (attn2, cross):
            with self.subTest(module=type(module).__name__):
                self.assertTrue(coexist.foreign_patch_live(_anima_model(_NewEngine(), _DiT([object(), module]))))
        self.assertFalse(coexist.foreign_patch_live(_anima_model(_NewEngine(), _DiT([object()]))))
        self.assertFalse(coexist.foreign_patch_live(None))

    def test_our_own_patch_is_not_foreign(self):
        model = _anima_model(_NewEngine())
        model.orig_forward = object()
        p = types.SimpleNamespace(sd_model=model, scripts=types.SimpleNamespace(alwayson_scripts=[self.script]))
        self.assertIsNone(coexist.standalone_reason(p, [False, True]))
        self.assertIsNotNone(coexist.standalone_reason(p, [False, False]))

    def test_loaded_lib_negpip_module_alone_is_not_a_foreign_negpip(self):
        # 문서화한 규칙(coexist.py): 모듈 경로는 경고 문구에만 쓴다 — 확장을 꺼도 lib_negpip 는 재시작 전까지 sys.modules 에
        # 남으므로, 그걸로 판별하면 내장까지 영영 꺼진다
        stale = types.ModuleType("lib_negpip")
        stale.__file__ = r"C:\forge\extensions\sd-forge-negpip\lib_negpip\__init__.py"
        with mock.patch.dict(sys.modules, {"lib_negpip": stale}):
            p = self._p()
            self.assertIsNone(coexist.standalone_reason(p, [False, False]))
            out = self._run(p)
        self.assertNotIn("standing down", out)
        self.assertEqual(self.module._test_calls, [("anima", False)])
        self.assertTrue(self.script.active)

    def test_warning_mentions_the_script_path(self):
        reason = coexist.standalone_reason(self._p(others=[_Foreign()]), [False, False])
        self.assertIn(_Foreign.filename, reason)


class AnimaEmphasisGateTests(ScriptTestCase):
    def _gate(self, engine, emphasis):
        self.shared.opts.emphasis = emphasis
        self.module._test_calls.clear()
        p = self._p(engine=engine)
        out = self._run(p)
        return bool(self.module._test_calls), out, p

    def test_new_engine_none_and_ignore_stay_off(self):
        for name in ("None", "Ignore"):
            with self.subTest(emphasis=name):
                patched, out, p = self._gate(_NewEngine(), name)
                self.assertFalse(patched)
                self.assertIn(f"NegPiP Disabled (Emphasis: {name})", out)
                self.assertNotIn("NegPiP", p.extra_generation_params)
                self.assertFalse(self.script.active)

    def test_new_engine_weighted_modes_patch(self):
        for name in ("Original", "No norm", "something unknown"):
            with self.subTest(emphasis=name):
                patched, _, _ = self._gate(_NewEngine(), name)
                self.assertTrue(patched)
                self.script.reset()

    def test_old_engine_only_none_stays_off(self):
        # 옛 엔진은 Ignore 에서도 가중치를 곱한다(Forge ad88b6b4 anima_preprocess)
        for name, expected in (("None", False), ("Ignore", True), ("Original", True)):
            with self.subTest(emphasis=name):
                patched, _, _ = self._gate(_OldEngine(), name)
                self.assertIs(patched, expected)
                self.script.reset()


class NewForgeEngineAttributeTests(ScriptTestCase):
    """Forge 2.29.2 는 Anima 텍스트 엔진을 text_processing_engine_qwen(Flux2·Krea2·Qwen-Image·Z-Image 공용)에 단다.
    옛 이름만 읽으면 Anima 생성에서 게이트가 AttributeError 로 죽는다 — anima_text_engine 이 두 이름을 다 찾는다."""

    NEW_NAME = "text_processing_engine_qwen"

    def test_gate_reads_the_engine_on_the_new_name(self):
        engine = _NewEngine()
        p = self._p(model=_anima_model(engine, attr=self.NEW_NAME))
        with mock.patch.object(self.module, "negpip_effective", wraps=self.module.negpip_effective) as gate:
            self._run(p)
        gate.assert_called_once()
        self.assertIs(gate.call_args.args[0], engine)
        self.assertEqual(self.module._test_calls, [("anima", False)])
        self.assertTrue(self.script.active)
        self.assertIs(p.extra_generation_params.get("NegPiP"), True)

    def test_new_name_gates_like_the_old_name(self):
        for name in ("Original", "No norm", "None", "Ignore"):
            results = {}
            for attr in ("text_processing_engine_anima", self.NEW_NAME):
                self.shared.opts.emphasis = name
                self.module._test_calls.clear()
                self._run(self._p(model=_anima_model(_NewEngine(), attr=attr)))
                results[attr] = bool(self.module._test_calls)
                self.script.reset()
            with self.subTest(emphasis=name):
                self.assertEqual(results[self.NEW_NAME], results["text_processing_engine_anima"], results)

    def test_engine_without_t5_tokenizer_on_the_new_name_is_not_read_as_anima(self):
        # 공용 이름의 다른 모델 엔진은 Anima 엔진으로 넘기지 않는다(None) — 게이트는 Anima 클래스 이름으로만 들어온다
        from sam3ext.anima38.native_engine import anima_text_engine

        self.assertIsNone(anima_text_engine(_anima_model(_QwenOnlyEngine(), attr=self.NEW_NAME)))
        self.assertIsNotNone(anima_text_engine(_anima_model(_NewEngine(), attr=self.NEW_NAME)))


if __name__ == "__main__":
    unittest.main()
