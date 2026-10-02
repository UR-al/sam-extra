"""3.8B v1/v2 경로의 'Emphasis' 생성 정보 — 순정 엔진 __call__ 과 같은 규칙.

v1/v2 는 Forge 엔진 __call__ 을 건너뛰어 순정 엔진이 남기는 'Emphasis' 기록이 빠졌다. 규칙은 엔진마다 다르다:
- 옛 AnimaTextProcessingEngine(Forge ad88b6b4 까지): 방식을 opts 에서 다시 읽고, 어느 줄이든 emphasis 를 쓰면 그 이름.
- 새 Qwen06Engine(21886f41~): 어느 줄이든 emphasis 를 쓰고 방식이 None/Ignore 일 때만.
실제 Qwen06Engine 과의 대조는 test_negpip_vendor.RealForgeEngineTests 에 있다.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_anima38 as base  # noqa: E402
from sam3ext.anima38.native_engine import emphasis_infotext  # noqa: E402

MODES = ("Original", "No norm", "None", "Ignore")


def _uses_emphasis(line: str) -> bool:
    return "(" in line or "[" in line


class _NewEngine:
    def __init__(self, name="Original"):
        self.opts = types.SimpleNamespace(emphasis=name)

    @property
    def emphasis(self):
        return types.SimpleNamespace(name=self.opts.emphasis)


class _OldEngine:
    def __init__(self, name="Original"):
        self.emphasis = types.SimpleNamespace(name=name)

    def tokenize_line(self, line):   # 옛 엔진 판별용
        return []


def _state(opts_emphasis="Original"):
    emphasis = types.SimpleNamespace(
        get_current_option=lambda name: (lambda: types.SimpleNamespace(name=name)),
        uses_emphasis=_uses_emphasis,
    )
    dynamic_args = types.SimpleNamespace(last_extra_generation_params={})
    return emphasis, dynamic_args, types.SimpleNamespace(emphasis=opts_emphasis)


class EmphasisInfotextRuleTests(unittest.TestCase):
    def test_new_engine_writes_only_none_and_ignore(self):
        for name in MODES:
            with self.subTest(emphasis=name):
                engine = _NewEngine(name)
                expected = name if name in ("None", "Ignore") else None
                self.assertEqual(emphasis_infotext(engine, ["a", "(b:1.2)"], _uses_emphasis), expected)
                self.assertIsNone(emphasis_infotext(engine, ["a", "b"], _uses_emphasis))

    def test_old_engine_writes_every_mode_when_emphasis_is_used(self):
        for name in MODES:
            with self.subTest(emphasis=name):
                engine = _OldEngine(name)
                self.assertEqual(emphasis_infotext(engine, ["(b:1.2)"], _uses_emphasis), name)
                self.assertIsNone(emphasis_infotext(engine, ["plain"], _uses_emphasis))

    def test_no_lines(self):
        self.assertIsNone(emphasis_infotext(_OldEngine(), [], _uses_emphasis))


class RecordEmphasisTests(unittest.TestCase):
    def setUp(self):
        self.module = base._load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()

    def _record(self, engine, lines, state):
        with mock.patch.object(self.module, "_emphasis_state", return_value=state):
            self.runtime._record_emphasis(engine, base._Prompt(lines))
        return state[1].last_extra_generation_params

    def test_new_engine(self):
        for name in MODES:
            with self.subTest(emphasis=name):
                params = self._record(_NewEngine(name), ["x, (y:-1)"], _state(name))
                self.assertEqual(params, {"Emphasis": name} if name in ("None", "Ignore") else {})

    def test_old_engine_rereads_opts_like_its_call(self):
        engine = _OldEngine("Original")   # 엔진이 쥔 방식은 낡았다 — 순정 __call__ 은 opts 에서 다시 읽는다
        params = self._record(engine, ["x, (y:1.2)"], _state("No norm"))
        self.assertEqual(engine.emphasis.name, "No norm")
        self.assertEqual(params, {"Emphasis": "No norm"})

    def test_old_engine_without_emphasis_writes_nothing_but_still_rereads(self):
        engine = _OldEngine("Original")
        params = self._record(engine, ["plain"], _state("None"))
        self.assertEqual(engine.emphasis.name, "None")
        self.assertEqual(params, {})

    def test_outside_forge_is_a_no_op(self):
        engine = _OldEngine("Original")
        with mock.patch.object(self.module, "_emphasis_state", return_value=None):
            self.runtime._record_emphasis(engine, base._Prompt(["(a:1.2)"]))
        self.assertEqual(engine.emphasis.name, "Original")

    def test_state_is_none_without_forge(self):
        self.assertIsNone(self.module._emphasis_state())


class _Model:
    def __init__(self, engine):
        self.text_processing_engine_anima = engine
        self.forge_objects = types.SimpleNamespace(clip=types.SimpleNamespace(patcher=object()))

    def get_learned_conditioning(self, prompt):
        return ["native"]


class EncodeWiringTests(unittest.TestCase):
    """v1·v2 경로에서만 기록한다 — 강도 0 은 순정 함수(엔진 __call__)가 스스로 기록한다."""

    def setUp(self):
        self.module = base._load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        self.engine = _NewEngine("Ignore")
        self.model = _Model(self.engine)
        self.state = _state("Ignore")
        patches = [
            mock.patch.object(self.module, "_emphasis_state", return_value=self.state),
            mock.patch.object(self.runtime, "_hand_off_native_reference"),
            mock.patch.object(self.runtime, "_release_native"),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    @property
    def params(self):
        return self.state[1].last_extra_generation_params

    def test_v2_path_records(self):
        self.runtime._active_bundle_metadata = {"architecture": "bundle"}
        with mock.patch.object(self.runtime, "_encode_v2", return_value="v2") as encode_v2:
            result = self.runtime.encode(self.model, base._Prompt(["(a:-1)"]), "x", 1.0, None)
        self.assertEqual(result, "v2")
        encode_v2.assert_called_once()
        self.assertEqual(self.params, {"Emphasis": "Ignore"})

    def test_v1_path_records_before_extracting(self):
        seen = {}

        def extract(*args, **kwargs):
            seen.update(self.params)
            raise RuntimeError("stop")

        with mock.patch.object(self.runtime, "_extract_prompt_features", side_effect=extract), \
                mock.patch.object(self.runtime, "_unload_patchers"):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                self.runtime.encode(self.model, base._Prompt(["(a:-1)"]), "x", 1.0, None)
        self.assertEqual(seen, {"Emphasis": "Ignore"})

    def test_zero_strength_leaves_it_to_the_native_engine(self):
        result = self.runtime.encode(self.model, base._Prompt(["(a:-1)"], negative=True), "x", 1.0, None)
        self.assertEqual(result, ["native"])
        self.assertEqual(self.params, {})


if __name__ == "__main__":
    unittest.main()
