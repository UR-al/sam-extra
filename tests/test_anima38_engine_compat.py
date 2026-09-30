"""Anima 3.8B 가 Forge 0.6B TE 엔진 두 세대 모두에서 도는지.

Forge 21886f41 "Rewrite TextProcessingEngine" 이 AnimaTextProcessingEngine(tokenize_line·process_tokens)을
Qwen06Engine(ComfyUI sd1_clip 이식: tokenize_with_weights·encode_token_weights)으로 바꿨다. 옛 호출이 남아 있으면
3.8B 생성이 전부 AttributeError 로 죽는다.
"""
from __future__ import annotations

import ast
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import test_anima38 as base  # noqa: E402  (클래스를 import 하면 그 테스트가 두 번 돈다 — 모듈로)
from sam3ext.anima38 import native_engine  # noqa: E402

FORGE_ANIMA_ENGINE = ROOT.parents[1] / "backend" / "text_processing" / "anima_engine.py"
FORGE_COMFY = ROOT.parents[1] / "backend" / "text_processing" / "_comfy.py"

_PAD = 151643   # Qwen06Engine 의 qwen pad
_END = 1        # T5 끝 토큰


def _tok(text: str) -> list[int]:
    return [ord(ch) % 89 + 2 for ch in text]   # 0·1(끝 토큰)과 겹치지 않게


def _parse(text: str, disable: bool) -> list[tuple[str, float]]:
    """'(단어:1.2)' 만 읽는 작은 가중치 파서 — disable 이면 원문 그대로 1.0 (ComfyUI disable_weights 와 같은 뜻)."""
    if disable:
        return [(text, 1.0)]
    out = []
    for m in re.finditer(r"\(([^():]*):([0-9.]+)\)|([^()]+)", text):
        out.append((m.group(3), 1.0) if m.group(3) is not None else (m.group(1), float(m.group(2))))
    return out


class _FakeTokenizer:
    """SDTokenizer.tokenize_with_weights 대역 — 호출을 적는다. chunks>1 이면 조각을 더 돌려준다."""

    def __init__(self, end_token=None, pad_token=None, chunks=1):
        self.end_token, self.pad_token, self.chunks = end_token, pad_token, chunks
        self.calls = []

    def tokenize_with_weights(self, text, **kwargs):
        self.calls.append((text, dict(kwargs)))
        pairs = [(t, w) for seg, w in _parse(text, kwargs.get("disable_weights", False)) for t in _tok(seg)]
        if not pairs and self.end_token is None:
            pairs = [(self.pad_token, 1.0)]   # min_length=1
        if self.end_token is not None:
            pairs.append((self.end_token, 1.0))
        return [pairs] + [[(self.end_token or self.pad_token, 1.0)]] * (self.chunks - 1)


class _FakeClipModel:
    """SDClipModel.encode_token_weights 대역 — 가중치가 1.0 이 아니면 ComfyUI 처럼 섞는다(강제 1.0 을 빠뜨리면 값이 바뀐다)."""

    def __init__(self, owner):
        self.owner = owner
        self.calls = []

    def encode_token_weights(self, token_weight_pairs):
        self.calls.append(token_weight_pairs)
        tokens = torch.tensor([t for t, _ in token_weight_pairs[0]], dtype=torch.float32)
        weights = torch.tensor([w for _, w in token_weight_pairs[0]], dtype=torch.float32)
        z = (torch.sin(tokens)[:, None] * self.owner.te_weight[None, :]).unsqueeze(0)
        mixed = (z - 0.5) * weights[None, :, None] + 0.5   # ComfyUI 처럼 1.0 이 아닌 토큰만 바꾼다
        z = torch.where((weights != 1.0)[None, :, None], mixed, z)
        return z, None, {"attention_mask": torch.ones(1, tokens.shape[0], dtype=torch.long)}


class _FakeQwen06Engine:
    """Qwen06Engine 대역 — tokenize_line/process_tokens 가 없고, emphasis 는 opts 를 매번 읽는 속성."""

    def __init__(self, width=4, chunks=1):
        self.opts = types.SimpleNamespace(emphasis="Original")
        self.te_weight = torch.linspace(0.5, 1.5, width)
        self.qwen_tokenizer = _FakeTokenizer(pad_token=_PAD, chunks=chunks)
        self.t5_tokenizer = _FakeTokenizer(end_token=_END, chunks=chunks)
        self.text_encoder = _FakeClipModel(self)

    @property
    def emphasis(self):
        return types.SimpleNamespace(name=self.opts.emphasis)   # 매번 새 객체 (Forge 와 같게)

    @emphasis.setter
    def emphasis(self, value):   # NativeRowCacheTests 가 engine.emphasis = … 로 바꾼다 → opts 를 바꾼 것으로
        self.opts.emphasis = value.name

    @property
    def forward_calls(self):
        return len(self.text_encoder.calls)


class Qwen06InputsTests(unittest.TestCase):
    """새 경로가 Qwen06Engine.__call__ 의 한 줄 계산(_preprocess 앞까지)과 같은지 — 기대값은 식에서 따로 만든다."""

    LINE = "girl, (smile:1.2)"

    def setUp(self):
        self.module = base._load_lifecycle_runtime()
        self.runtime_cls = self.module.Anima3BRuntime
        self.engine = _FakeQwen06Engine()

    def _run(self, dtype=torch.float32):
        return self.runtime_cls._native_inputs(self.engine, self.LINE, "cpu", dtype)

    def _expected_source(self, text_ids):
        ids = torch.tensor(text_ids, dtype=torch.float32)
        return (torch.sin(ids)[:, None] * self.engine.te_weight[None, :]).unsqueeze(0)

    def test_original_emphasis_matches_the_forge_formula(self):
        source, ids, weights = self._run()
        plain, smile = _tok("girl, "), _tok("smile")
        self.assertEqual(ids.tolist(), [plain + smile + [_END]])
        self.assertEqual(weights.reshape(-1).tolist(), [1.0] * 6 + [1.2000000476837158] * 5 + [1.0])
        self.assertTrue(torch.equal(source, self._expected_source(plain + smile)), "qwen 가중치는 1.0 으로 강제")
        # 호출 모양: 두 토크나이저 한 번씩 disable_weights=False, 인코더에는 가중치 1.0 한 조각
        self.assertEqual(self.engine.qwen_tokenizer.calls, [(self.LINE, {"disable_weights": False})])
        self.assertEqual(self.engine.t5_tokenizer.calls, [(self.LINE, {"disable_weights": False})])
        self.assertEqual(self.engine.text_encoder.calls, [[[(t, 1.0) for t in plain + smile]]])

    def test_shapes_devices_dtypes_match_the_old_path(self):
        source, ids, weights = self._run(dtype=torch.float16)
        length = len(_tok("girl, smile")) + 1
        self.assertEqual((tuple(source.shape), source.dtype), ((1, length - 1, 4), torch.float16))
        self.assertEqual((tuple(ids.shape), ids.dtype), ((1, length), torch.long))
        self.assertEqual((tuple(weights.shape), weights.dtype), ((1, length, 1), torch.float16))
        self.assertEqual({t.device.type for t in (source, ids, weights)}, {"cpu"})
        # 가중치는 옛 경로처럼 파이썬 float 에서 바로 dtype 으로
        self.assertTrue(torch.equal(weights.reshape(-1), torch.tensor([1.0] * 6 + [1.2] * 5 + [1.0], dtype=torch.float16)))

    def test_none_emphasis_disables_weight_parsing(self):
        self.engine.opts.emphasis = "None"
        source, ids, weights = self._run()
        for tokenizer in (self.engine.qwen_tokenizer, self.engine.t5_tokenizer):
            self.assertEqual(tokenizer.calls, [(self.LINE, {"disable_weights": True})])
        self.assertEqual(ids.tolist(), [_tok(self.LINE) + [_END]], "괄호까지 글자 그대로")
        self.assertTrue(bool((weights == 1.0).all()))
        self.assertTrue(torch.equal(source, self._expected_source(_tok(self.LINE))))

    def test_ignore_emphasis_parses_but_keeps_t5_weights_at_one(self):
        self.engine.opts.emphasis = "Ignore"
        _, ids, weights = self._run()
        self.assertEqual(self.engine.t5_tokenizer.calls, [(self.LINE, {"disable_weights": False})])
        self.assertEqual(ids.tolist(), [_tok("girl, ") + _tok("smile") + [_END]], "괄호는 먹는다")
        self.assertEqual(weights.reshape(-1).tolist(), [1.0] * ids.shape[1])

    def test_empty_line_uses_pad_and_end_tokens(self):
        source, ids, weights = self.runtime_cls._native_inputs(self.engine, "", "cpu", torch.float32)
        self.assertEqual(ids.tolist(), [[_END]])
        self.assertEqual(self.engine.text_encoder.calls, [[[(_PAD, 1.0)]]])
        self.assertEqual(tuple(source.shape), (1, 1, 4))
        self.assertEqual(weights.reshape(-1).tolist(), [1.0])

    def test_more_than_one_chunk_raises_before_the_encoder(self):
        self.engine = _FakeQwen06Engine(chunks=2)
        with self.assertRaisesRegex(RuntimeError, "one prompt chunk"):
            self._run()
        self.assertEqual(self.engine.text_encoder.calls, [])

    def test_old_engine_still_takes_the_old_path(self):
        engine = base._FakeNativeEngine()
        with mock.patch.object(engine, "tokenize_line", wraps=engine.tokenize_line) as tokenize_line, \
                mock.patch.object(self.module, "qwen06_native_inputs") as new_path:
            source, ids, weights = self.runtime_cls._native_inputs(engine, self.LINE, "cpu", torch.float16)
        new_path.assert_not_called()
        tokenize_line.assert_called_once_with(self.LINE)
        self.assertEqual(engine.forward_calls, 1)
        chunk = base._FakeNativeEngine().tokenize_line(self.LINE)[0]
        expected = engine.process_tokens([chunk.qwen_tokens], [chunk.qwen_multipliers])[0].unsqueeze(0)
        self.assertTrue(torch.equal(source, expected.to(torch.float16)))
        self.assertTrue(torch.equal(ids, torch.tensor([chunk.t5_tokens], dtype=torch.long)))
        self.assertTrue(torch.equal(weights, torch.tensor(chunk.t5_multipliers, dtype=torch.float16).reshape(1, -1, 1)))

    def test_old_engine_multi_chunk_still_raises(self):
        engine = base._FakeNativeEngine()
        engine.tokenize_line = lambda line: [object(), object()]
        with self.assertRaisesRegex(RuntimeError, "one prompt chunk"):
            self.runtime_cls._native_inputs(engine, self.LINE, "cpu", torch.float32)

    def test_dispatch_is_by_the_old_api(self):
        self.assertTrue(native_engine.is_legacy_engine(base._FakeNativeEngine()))
        self.assertFalse(native_engine.is_legacy_engine(_FakeQwen06Engine()))


class Qwen06NativeRowCacheTests(base.NativeRowCacheTests):
    """줄 캐시 검사 전부를 새 엔진 대역으로 — 적중·LoRA·emphasis(opts)·DoRA·엔진 교체가 옛 엔진과 같이 동작하는지."""

    engine_factory = _FakeQwen06Engine

    def test_opts_emphasis_change_is_a_cache_miss(self):
        # 새 엔진은 emphasis 를 엔진이 쥐지 않고 opts 에서 읽는다 — opts 만 바꿔도 키가 달라져야 한다
        prompt = ["girl, (smile:1.2)"]
        first = self._assert_same_as_old(prompt, te_loads=1)
        self.engine.opts.emphasis = "Ignore"
        second = self._assert_same_as_old(prompt, te_loads=2)
        self.assertFalse(torch.equal(first[1][0][2], second[1][0][2]), "Ignore 는 T5 가중치를 1.0 으로")
        self.engine.opts.emphasis = "Original"
        self._assert_same_as_old(prompt, te_loads=2)   # 예전 키로 돌아오면 다시 적중


class NegPipEngineGuardTests(unittest.TestCase):
    """NegPiP 의 _build_negpip_mask 는 아직 engine.tokenize_line 을 부른다 — 새 엔진이면 경고 한 번 후 조건을 그대로."""

    def setUp(self):
        self.module = base._load_lifecycle_runtime()
        self.runtime = self.module.Anima3BRuntime()
        lib = types.ModuleType("lib_negpip")
        lib.__path__ = []
        anima = types.ModuleType("lib_negpip.anima")

        def _build_negpip_mask(engine, line, n, device, dtype):   # 상류와 같이 옛 API 를 먼저 부른다
            engine.tokenize_line(line)
            return torch.ones(n, device=device, dtype=dtype)

        anima._build_negpip_mask = _build_negpip_mask
        patcher = mock.patch.dict(sys.modules, {"lib_negpip": lib, "lib_negpip.anima": anima})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _apply(self, engine):
        model = types.SimpleNamespace(text_processing_engine_anima=engine)
        conds = [torch.ones(1, 3, 4)]
        return conds, self.runtime._apply_negpip(model, base._Prompt(["girl, [:bad:-1]"]), conds)

    def test_new_engine_warns_once_and_keeps_conds(self):
        engine = _FakeQwen06Engine()
        with self.assertLogs(self.module.logger, "WARNING") as logs:
            conds, result = self._apply(engine)
            again_conds, again = self._apply(engine)
        self.assertIs(result, conds)
        self.assertIs(again, again_conds)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("NegPiP", logs.output[0])

    def test_old_engine_still_masks(self):
        conds, result = self._apply(base._FakeNativeEngine())
        self.assertIsInstance(result, dict)
        self.assertEqual(tuple(result["c_negpip_mask"].shape), (1, 3, 1))

    def test_attribute_error_on_an_old_engine_is_not_swallowed(self):
        engine = base._FakeNativeEngine()
        sys.modules["lib_negpip.anima"]._build_negpip_mask = lambda *args: engine.missing_attribute
        with self.assertRaises(AttributeError):
            self._apply(engine)


def _methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}


def _self_attrs(func: ast.FunctionDef) -> set[str]:
    return {
        target.attr
        for node in ast.walk(func) if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
    }


class ForgeQwen06EngineContractTests(unittest.TestCase):
    """설치된 Forge 의 Qwen06Engine 이 새 경로가 기대는 이름·호출을 아직 가지고 있는지 (AST, Forge 없는 CI 는 건너뜀).

    깨지면 Forge 가 엔진을 다시 바꾼 것 — sam3ext/anima38/native_engine.py 를 새 __call__ 에 맞춰 고친다."""

    def setUp(self):
        if not FORGE_ANIMA_ENGINE.is_file():
            self.skipTest(f"Forge 소스 없음: {FORGE_ANIMA_ENGINE}")
        tree = ast.parse(FORGE_ANIMA_ENGINE.read_text(encoding="utf-8"))
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        if "Qwen06Engine" not in classes:
            if any("tokenize_line" in _methods(cls) for cls in classes.values()):
                self.skipTest("21886f41 이전 Forge (옛 엔진 — 옛 경로가 쓰인다)")
            self.fail(f"Anima 엔진 클래스를 찾지 못했다: {sorted(classes)}")
        self.engine = classes["Qwen06Engine"]
        self.methods = _methods(self.engine)

    def test_engine_has_the_attributes_the_new_path_reads(self):
        self.assertNotIn("tokenize_line", self.methods, "옛 API 가 돌아왔다면 디스패치가 옛 경로를 탄다")
        self.assertLessEqual({"text_encoder", "qwen_tokenizer", "t5_tokenizer"}, _self_attrs(self.methods["__init__"]))
        emphasis = self.methods.get("emphasis")
        self.assertIsNotNone(emphasis)
        self.assertIn("property", {getattr(d, "id", None) for d in emphasis.decorator_list})

    def test_call_still_uses_the_mirrored_formula(self):
        call = self.methods["__call__"]
        calls = [node for node in ast.walk(call) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        tokenize = [c for c in calls if c.func.attr == "tokenize_with_weights"]
        self.assertEqual(sorted(c.func.value.attr for c in tokenize), ["qwen_tokenizer", "t5_tokenizer"])
        self.assertTrue(all(any(k.arg == "disable_weights" for k in c.keywords) for c in tokenize))
        self.assertTrue(any(c.func.attr == "encode_token_weights" and c.func.value.attr == "text_encoder" for c in calls))
        strings = {node.value for node in ast.walk(call) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        self.assertLessEqual({"None", "Ignore"}, strings)
        # qwen 가중치를 1.0 으로 강제하는 (x[0], 1.0) 과 t5 첫 조각만 쓰는 t5_chunk[0]
        source = ast.get_source_segment(FORGE_ANIMA_ENGINE.read_text(encoding="utf-8"), call)
        self.assertIn("(x[0], 1.0)", source)
        self.assertIn("t5_chunk[0]", source)

    def test_comfy_helpers_exist(self):
        if not FORGE_COMFY.is_file():
            self.skipTest(f"Forge 소스 없음: {FORGE_COMFY}")
        tree = ast.parse(FORGE_COMFY.read_text(encoding="utf-8"))
        classes = {node.name: _methods(node) for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertIn("tokenize_with_weights", classes.get("SDTokenizer", {}))
        self.assertIn("encode_token_weights", classes.get("ClipTokenWeightEncoder", {}))


if __name__ == "__main__":
    unittest.main()
