"""내장 NegPiP(sam3ext/negpip — Haoming02/sd-forge-negpip 0585496 편입)의 순수 로직과 상류 동일성·라이선스 고지.

- NEG_PATTERN·any_negative: 상류 0585496 그대로(escaped 괄호, 앞의 일반 괄호를 삼키지 않는 새 패턴).
- Anima 마스크: 옛 엔진(tokenize_line → t5_multipliers, 상류 b3673ce)·새 엔진(t5_tokenizer.tokenize_with_weights, 0585496).
  마스크는 엔진이 조건 행에 실제로 곱한 가중치의 부호다 — 새 엔진의 emphasis None/Ignore 는 음수 행이 없다.
- 실제 Forge Qwen06Engine + 실제 Anima 토크나이저(CPU)로 같은 규칙을 확인한다(없으면 건너뜀).
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import logging
import os
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext.negpip import mask as negpip_mask  # noqa: E402
from sam3ext.negpip import utils as negpip_utils  # noqa: E402

FORGE_ROOT = ROOT.parents[1]
VENDOR = ROOT / "sam3ext" / "negpip"
SCRIPT = ROOT / "scripts" / "negpip.py"
ANIMA_TOKENIZERS = FORGE_ROOT / "backend" / "huggingface" / "circlestone-labs" / "Anima"

# 상류 0585496 의 NEG_PATTERN (lib_negpip/utils.py) — 바꾸지 않았다
UPSTREAM_NEG_PATTERN = r"\(\s*(?:[^\\(:)]|\\[\(\)])+?\s*\:\s*-\s*\d*\.?\d+\s*\)"

# 상류 0585496 함수 본문의 SHA-256 앞 16자 (ast.get_source_segment, LF). 편입하며 바꾸지 않은 것만 — 훅이 그대로인지 지킨다.
# 바꾼 것(anima._build_negpip_mask·_hook_get_learned_conditioning, sd._hook_forward, NegPiP.__init__·process_batch·
# denoiser_callback·_cond_dealer·_calc_conds)은 THIRD_PARTY_NOTICES.md 와 파일 머리에 적었다. _cond_dealer 가 한 청크 항에서
# 옛 엔진의 상류 자르기와 같은 행임을 test_negpip_clip_rows 가, Anima 훅이 줄별 조건을 돌려주는 것을 test_negpip_anima_schedule
# 이, SD1/SDXL 훅이 Forge 의 조각 표시로 행마다 제 항목의 음수 항 전부를 붙이는 것을 test_negpip_sd_batching 이 확인한다.
UPSTREAM_FUNCTION_HASHES = {
    "sam3ext/negpip/anima.py": {
        "patch_anima_negpip": "9b25a350d0e5f66d",
        "_hook_dit_forward": "5cb4e3529587da7f",
        "_hook_forwards": "267b1f26acb1a129",
        "_hook_compile_conditions": "79a8194e3171278b",
    },
    "sam3ext/negpip/sd.py": {
        "patch_sd_negpip": "75917b5ce83eb773",
        "Counter": "1009e90596b56daa",
        "_main_forward": "0ffc3eaaf0697c4b",
    },
    "sam3ext/negpip/utils.py": {
        "reset_prompt_cache": "16fe7471d65424d9",
        "hr_dealer": "fd0936ec346e894b",
        "has_negative": "7055daf2084db740",
        "have_negative": "d283363ff349553b",
        "any_negative": "9b3ef5ead8b0a200",
    },
    "scripts/negpip.py": {
        "_verify_ext": "dc1f625e9f3f811f",
        "NegPiP.reset": "243ac1088eb19b67",
        "NegPiP.title": "d17b04287a49e607",
        "NegPiP.show": "ad1aaf3258e5dfaa",
        "NegPiP.ui": "bd3674fe74081e30",
        "NegPiP.postprocess": "393b912221ad2f89",
        "NegPiP.before_hr": "02f0d9f535b538b4",
        "NegPiP._getScheduledNegPip": "f7f57a1293668259",
    },
}
# 상류 LICENSE (AGPL-3.0 전문) 의 SHA-256 — 줄 끝을 LF 로 맞춘 바이트 (다른 원본 고정 테스트처럼; Windows
# autocrlf 체크아웃은 CRLF 로 바꿔 놓는다)
LICENSE_SHA256 = "ce3fb82d9ee80a1cb0e548f9b0560ead52378ee2b4e826fd9459b4acf7075d89"


def _p(prompts=(), negative_prompts=(), hr_prompts=None, hr_negative_prompts=None):
    p = types.SimpleNamespace(prompts=list(prompts), negative_prompts=list(negative_prompts))
    if hr_prompts is not None:
        p.hr_prompts = list(hr_prompts)
    if hr_negative_prompts is not None:
        p.hr_negative_prompts = list(hr_negative_prompts)
    return p


class NegPatternTests(unittest.TestCase):
    def test_pattern_is_upstream_0585496(self):
        self.assertEqual(negpip_utils.NEG_PATTERN.pattern, UPSTREAM_NEG_PATTERN)

    def test_cases(self):
        cases = {
            "(foo:-1)": ["(foo:-1)"],
            "(foo:-1.5)": ["(foo:-1.5)"],
            "( foo bar : - .5 )": ["( foo bar : - .5 )"],
            "(red:-1), (blue:-2)": ["(red:-1)", "(blue:-2)"],
            "((foo:-1))": ["(foo:-1)"],
            "(x (y:-1))": ["(y:-1)"],
            # 새 패턴: 앞의 일반 괄호를 삼키지 않는다 (옛 패턴 [^:]+ 는 '(b) c (d:-1)' 을 잡았다)
            "a (b) c (d:-1)": ["(d:-1)"],
            # escaped 괄호는 대상 안의 글자로 읽는다
            "(foo \\(bar\\):-1)": ["(foo \\(bar\\):-1)"],
            "(\\(x\\):-0.3)": ["(\\(x\\):-0.3)"],
            # escaped 괄호로 둘러싼 것은 가중치 문법이 아니다
            "\\(foo:-1\\)": [],
            "(foo:1)": [],
            "(foo:-1": [],
            "(a:b:-1)": [],
            "[foo:-1]": [],
            "(foo:-1e2)": [],
            "(foo)": [],
            "": [],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(negpip_utils.NEG_PATTERN.findall(text), expected)
                self.assertEqual(negpip_utils.has_negative(text), bool(expected))


class AnyNegativeTests(unittest.TestCase):
    def test_each_prompt_list_counts(self):
        self.assertFalse(negpip_utils.any_negative(_p(["a, (b:1.2)"], ["c"])))
        self.assertTrue(negpip_utils.any_negative(_p(["a, (b:-1)"], ["c"])))
        self.assertTrue(negpip_utils.any_negative(_p(["a"], ["(c:-0.5)"])))
        self.assertTrue(negpip_utils.any_negative(_p(["a"], ["c"], hr_prompts=["(h:-1)"])))
        self.assertTrue(negpip_utils.any_negative(_p(["a"], ["c"], hr_negative_prompts=["(h:-1)"])))

    def test_missing_or_empty_hr_prompts(self):
        self.assertFalse(negpip_utils.any_negative(_p(["a"], ["b"], hr_prompts=[], hr_negative_prompts=None)))
        self.assertFalse(negpip_utils.any_negative(_p(["a"], ["b"])))

    def test_hr_dealer(self):
        self.assertEqual(negpip_utils.hr_dealer(_p(hr_prompts=["x"])), (True, False))
        self.assertEqual(negpip_utils.hr_dealer(_p()), (False, False))


# ---- Anima 마스크 (가짜 엔진) ----

def _tok(text: str) -> list[int]:
    return [ord(ch) % 89 + 2 for ch in text]


class _FakeTokenizer:
    """SDTokenizer.tokenize_with_weights 대역 — '(단어:w)' 만 읽는다. disable_weights 면 원문 그대로."""

    END = 1

    def __init__(self):
        self.calls = []

    def tokenize_with_weights(self, text, **kwargs):
        import re

        self.calls.append((text, dict(kwargs)))
        if kwargs.get("disable_weights", False):
            parsed = [(text, 1.0)]
        else:
            parsed = [
                (m.group(3), 1.0) if m.group(3) is not None else (m.group(1), float(m.group(2)))
                for m in re.finditer(r"\(([^():]*):(-?[0-9.]+)\)|([^()]+)", text)
            ]
        pairs = [(t, w) for seg, w in parsed for t in _tok(seg)]
        return [pairs + [(self.END, 1.0)]]


class _FakeNewEngine:
    """Qwen06Engine 대역 — tokenize_line 이 없고 emphasis 는 opts 를 매번 읽는 속성."""

    def __init__(self, emphasis="Original"):
        self.opts = types.SimpleNamespace(emphasis=emphasis)
        self.t5_tokenizer = _FakeTokenizer()

    @property
    def emphasis(self):
        return types.SimpleNamespace(name=self.opts.emphasis)


class _FakeOldEngine:
    """AnimaTextProcessingEngine 대역 — tokenize_line 이 t5_multipliers 를 가진 조각을 돌려준다."""

    def __init__(self, multipliers):
        self.multipliers = multipliers
        self.lines = []
        self.emphasis = types.SimpleNamespace(name="Original")

    def tokenize_line(self, line):
        self.lines.append(line)
        return [types.SimpleNamespace(t5_tokens=list(range(len(self.multipliers))), t5_multipliers=self.multipliers)]


class MaskBuilderTests(unittest.TestCase):
    LINE = "girl, (bad:-1), (good:1.2)"

    def _expected_new(self, rows):
        plain, bad, sep, good = _tok("girl, "), _tok("bad"), _tok(", "), _tok("good")
        signs = [1.0] * len(plain) + [-1.0] * len(bad) + [1.0] * (len(sep) + len(good) + 1)
        return torch.tensor((signs + [1.0] * rows)[:rows])

    def test_new_engine_follows_the_t5_weights_and_pads(self):
        engine = _FakeNewEngine()
        mask = negpip_mask.build_negpip_mask(engine, self.LINE, 64, "cpu", torch.float32)
        self.assertTrue(torch.equal(mask, self._expected_new(64)))
        self.assertEqual(engine.t5_tokenizer.calls, [(self.LINE, {"disable_weights": False})], "엔진 __call__ 과 같은 호출")

    def test_new_engine_crops_to_the_row_count(self):
        mask = negpip_mask.build_negpip_mask(_FakeNewEngine(), self.LINE, 8, "cpu", torch.float16)
        self.assertEqual(mask.dtype, torch.float16)
        self.assertTrue(torch.equal(mask, self._expected_new(8).to(torch.float16)))

    def test_no_norm_is_like_original(self):
        mask = negpip_mask.build_negpip_mask(_FakeNewEngine("No norm"), self.LINE, 64, "cpu", torch.float32)
        self.assertTrue(torch.equal(mask, self._expected_new(64)))

    def test_new_engine_none_and_ignore_are_all_ones_without_tokenizing(self):
        # None: 엔진은 가중치를 파싱하지 않는다(괄호가 토큰으로 남아 행 수부터 다르다) — 음수 행이 없다.
        # Ignore: 괄호는 먹지만 T5 가중치를 1.0 으로 둔다 — 엔진이 뒤집지 않은 행을 NegPiP 가 뒤집으면 K 만 음수가 된다.
        for name in ("None", "Ignore"):
            with self.subTest(emphasis=name):
                engine = _FakeNewEngine(name)
                mask = negpip_mask.build_negpip_mask(engine, self.LINE, 32, "cpu", torch.float32)
                self.assertTrue(torch.equal(mask, torch.ones(32)))
                self.assertEqual(engine.t5_tokenizer.calls, [])

    def test_old_engine_uses_t5_multipliers_like_b3673ce(self):
        engine = _FakeOldEngine([1.0, -1.0, -0.5, 1.2, 1.0])
        mask = negpip_mask.build_negpip_mask(engine, self.LINE, 7, "cpu", torch.float32)
        self.assertEqual(mask.tolist(), [1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(engine.lines, [self.LINE])

    def test_old_engine_ignore_still_masks(self):
        # 옛 엔진은 Ignore 에서도 t5 가중치를 곱한다(파서는 None 만 글자 그대로) — 엔진이 준 가중치를 그대로 따른다
        engine = _FakeOldEngine([-1.0, 1.0])
        engine.emphasis = types.SimpleNamespace(name="Ignore")
        self.assertEqual(negpip_mask.build_negpip_mask(engine, "x", 2, "cpu", torch.float32).tolist(), [-1.0, 1.0])

    def test_zero_weight_does_not_flip_the_row(self):
        # 상류 그대로 weights < 0 — '(x:0)' 은 엔진이 그 행을 0 으로 만들 뿐 음수 가중치가 아니다(-0.0 도 음수가 아님)
        engine = _FakeOldEngine([1.0, 0.0, -0.0, -1.0])
        self.assertEqual(negpip_mask.build_negpip_mask(engine, "x", 4, "cpu", torch.float32).tolist(), [1.0, 1.0, 1.0, -1.0])
        line = "girl, (x:0), (y:-0), (bad:-1)"
        mask = negpip_mask.build_negpip_mask(_FakeNewEngine(), line, 64, "cpu", torch.float32)
        plain, x, sep, y, bad = _tok("girl, "), _tok("x"), _tok(", "), _tok("y"), _tok("bad")
        signs = [1.0] * (len(plain) + len(x) + len(sep) + len(y) + len(sep)) + [-1.0] * len(bad) + [1.0]
        self.assertEqual(mask.tolist(), (signs + [1.0] * 64)[:64])

    def test_empty_multipliers_give_ones(self):
        engine = _FakeOldEngine([])
        self.assertTrue(torch.equal(negpip_mask.build_negpip_mask(engine, "", 4, "cpu", torch.float32), torch.ones(4)))

    def test_negpip_effective(self):
        new, old = _FakeNewEngine(), _FakeOldEngine([])
        table = {
            ("new", "Original"): True, ("new", "No norm"): True, ("new", "None"): False, ("new", "Ignore"): False,
            ("old", "Original"): True, ("old", "No norm"): True, ("old", "None"): False, ("old", "Ignore"): True,
        }
        for (kind, name), expected in table.items():
            with self.subTest(engine=kind, emphasis=name):
                engine = new if kind == "new" else old
                self.assertIs(negpip_mask.negpip_effective(engine, name), expected)

    def test_legacy_dispatch_matches_the_3_8b_runtime(self):
        from sam3ext.anima38 import native_engine

        for engine in (_FakeNewEngine(), _FakeOldEngine([])):
            self.assertIs(negpip_mask.is_legacy_engine(engine), native_engine.is_legacy_engine(engine))

    def test_mask_module_does_not_import_forge(self):
        tree = ast.parse((VENDOR / "mask.py").read_text(encoding="utf-8"))
        imported = {
            (node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
            for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertLessEqual(imported, {"__future__", "torch"})

    def test_anima_hook_calls_the_shared_builder(self):
        source = (VENDOR / "anima.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_negpip_mask")
        calls = {n.func.id for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertEqual(calls, {"build_negpip_mask"})
        self.assertIn("from sam3ext.negpip.mask import build_negpip_mask", source)


# ---- 실제 Forge Qwen06Engine + 실제 Anima 토크나이저 (CPU) ----

@contextlib.contextmanager
def _forge_text_stubs():
    """Forge 의 backend.text_processing 을 실제 파일로, 나머지(memory_management·args·opts)만 대역으로 잠시 올린다."""
    names = [n for n in sys.modules if n in ("backend", "modules") or n.startswith(("backend.", "modules."))]
    saved = {n: sys.modules.pop(n) for n in names}
    backend = types.ModuleType("backend")
    backend.__path__ = [str(FORGE_ROOT / "backend")]
    mm = types.ModuleType("backend.memory_management")
    mm.intermediate_device = lambda: torch.device("cpu")
    mm.text_encoder_device = lambda: torch.device("cpu")
    mm.logger = logging.getLogger("negpip-test")
    args = types.ModuleType("backend.args")
    args.dynamic_args = types.SimpleNamespace(last_extra_generation_params={})
    tp = types.ModuleType("backend.text_processing")
    tp.__path__ = [str(FORGE_ROOT / "backend" / "text_processing")]
    modules = types.ModuleType("modules")
    modules.__path__ = []
    shared = types.ModuleType("modules.shared")
    shared.opts = types.SimpleNamespace(emphasis="Original")
    backend.memory_management, backend.args, backend.text_processing = mm, args, tp
    modules.shared = shared
    stubs = {"backend": backend, "backend.memory_management": mm, "backend.args": args,
             "backend.text_processing": tp, "modules": modules, "modules.shared": shared}
    sys.modules.update(stubs)
    try:
        yield args.dynamic_args, shared.opts
    finally:
        for name in [n for n in sys.modules if n in ("backend", "modules") or n.startswith(("backend.", "modules."))]:
            del sys.modules[name]
        sys.modules.update(saved)


class _FakeQwen(torch.nn.Module):
    """0.6B TE 자리 — 토크나이저·파서는 실제, 트랜스포머만 작은 가짜."""

    num_layers = 2

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(151936, 4)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None,
                final_layer_norm_intermediate=None, dtype=None, embeds_info=None):
        return embeds, None


class RealForgeEngineTests(unittest.TestCase):
    """실제 Qwen06Engine.__call__ 이 조건에 곱한 T5 가중치와 내장 마스크·3.8B Emphasis 기록 규칙이 같은가."""

    LINES = (
        "",
        "girl, (smile:1.2), [blush]",
        "1girl, solo, (aqua hair:-1.0), smile",
        "(red:-1), (blue:-1.5) eyes, (bad hands:-0.5)",
        "a (b) c (d:-1)",
        "\\(escaped\\) text, (nested (x:-1.5):1.1)",
        "(foo \\(bar\\):-1), plain",
    )

    @classmethod
    def setUpClass(cls):
        engine_file = FORGE_ROOT / "backend" / "text_processing" / "anima_engine.py"
        if not engine_file.is_file() or not (ANIMA_TOKENIZERS / "tokenizer_2").is_dir():
            raise unittest.SkipTest("Forge 소스·Anima 토크나이저 없음")
        if "class Qwen06Engine" not in engine_file.read_text(encoding="utf-8"):
            raise unittest.SkipTest("21886f41 이전 Forge (옛 엔진)")
        try:
            from transformers import Qwen2Tokenizer, T5TokenizerFast
        except ImportError as exc:   # pragma: no cover - Forge venv 밖
            raise unittest.SkipTest(f"transformers 없음: {exc}")
        from sam3ext.anima38.native_engine import emphasis_infotext

        cls.emphasis_infotext = staticmethod(emphasis_infotext)
        qtok = Qwen2Tokenizer.from_pretrained(str(ANIMA_TOKENIZERS / "tokenizer"))
        ttok = T5TokenizerFast.from_pretrained(str(ANIMA_TOKENIZERS / "tokenizer_2"))
        cls._stubs = _forge_text_stubs()
        cls.dynamic_args, cls.opts = cls._stubs.__enter__()
        try:
            cls.emphasis = importlib.import_module("backend.text_processing.emphasis")
            engine_module = importlib.import_module("backend.text_processing.anima_engine")
            cls.engine = engine_module.Qwen06Engine(_FakeQwen(), qtok, ttok)
        except BaseException:
            cls._stubs.__exit__(*sys.exc_info())
            raise
        cls.captured = []
        cls.engine._preprocess = lambda cond, ids, weights: cls.captured.append((ids, weights)) or cond

    @classmethod
    def tearDownClass(cls):
        cls._stubs.__exit__(None, None, None)

    def _call(self, line):
        self.captured.clear()
        self.dynamic_args.last_extra_generation_params.clear()
        self.engine([line])
        ids, weights = self.captured[0]
        return ids.reshape(-1), weights.reshape(-1)

    def test_mask_is_the_sign_of_the_weights_the_engine_applied(self):
        negatives = 0
        for name in ("Original", "No norm", "None", "Ignore"):
            self.opts.emphasis = name
            for line in self.LINES:
                with self.subTest(emphasis=name, line=line):
                    ids, weights = self._call(line)
                    mask = negpip_mask.build_negpip_mask(self.engine, line, 512, "cpu", torch.float32)
                    expected = torch.ones(512)
                    expected[: weights.shape[0]] = torch.where(weights < 0, -1.0, 1.0)
                    self.assertTrue(torch.equal(mask, expected))
                    if name in ("None", "Ignore"):
                        self.assertTrue(bool((mask == 1.0).all()))
                    negatives += int((mask < 0).sum())
        self.assertGreater(negatives, 0, "음수 가중치 줄이 실제로 -1 행을 만든다(무의미한 비교가 아님)")

    def test_none_changes_the_token_rows(self):
        # None 에서 상류 0585496 처럼 가중치를 파싱하면 행 수부터 어긋난다 — 그래서 마스크를 만들지 않는다
        line = "1girl, (aqua hair:-1.0), smile"
        self.opts.emphasis = "None"
        literal, _ = self._call(line)
        parsed = self.engine.t5_tokenizer.tokenize_with_weights(line)[0]
        self.assertNotEqual(literal.shape[0], len(parsed))

    def test_3_8b_emphasis_infotext_matches_the_engine_call(self):
        for name in ("Original", "No norm", "None", "Ignore"):
            self.opts.emphasis = name
            for line in self.LINES:
                with self.subTest(emphasis=name, line=line):
                    self._call(line)
                    written = self.dynamic_args.last_extra_generation_params.get("Emphasis")
                    self.assertEqual(self.emphasis_infotext(self.engine, [line], self.emphasis.uses_emphasis), written)


# ---- 상류와 같은지 · 라이선스 고지 ----

def _function_hashes(path: Path) -> dict[str, str]:
    source = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    tree = ast.parse(source)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = hashlib.sha256(ast.get_source_segment(source, node).encode("utf-8")).hexdigest()[:16]
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    segment = ast.get_source_segment(source, sub)
                    out[f"{node.name}.{sub.name}"] = hashlib.sha256(segment.encode("utf-8")).hexdigest()[:16]
    return out


class UpstreamParityTests(unittest.TestCase):
    """편입하며 바꾸지 않은 함수는 상류 0585496 과 글자까지 같다 — 바꾼 함수는 목록에서 빼고 파일 머리·THIRD_PARTY_NOTICES 에 적는다."""

    def test_unmodified_functions_match_upstream(self):
        for rel, expected in UPSTREAM_FUNCTION_HASHES.items():
            actual = _function_hashes(ROOT / rel)
            for name, digest in expected.items():
                with self.subTest(file=rel, function=name):
                    self.assertEqual(actual.get(name), digest)

    def test_upstream_imports_were_rewritten(self):
        for path in [*VENDOR.glob("*.py"), SCRIPT]:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            modules = {
                node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
            } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
            with self.subTest(file=path.name):
                self.assertFalse({m for m in modules if m.split(".")[0] == "lib_negpip"})

    def test_script_keeps_the_upstream_file_name(self):
        # 상류와 같은 scripts/negpip.py — ADetailer ad_script_names 가 stem 'negpip' 으로 고른다. Forge 는 확장 스크립트를
        # '확장 이름/파일 이름' 으로 구별하고 경로로 불러와(sys.modules 에 넣지 않음) 따로 설치된 확장의 같은 이름과 부딪히지 않는다.
        self.assertTrue(SCRIPT.is_file())
        self.assertEqual(SCRIPT.relative_to(ROOT).as_posix(), "scripts/negpip.py")
        self.assertFalse((ROOT / "scripts" / "sam_extra_negpip.py").exists())


class LicenseNoticeTests(unittest.TestCase):
    """AGPL-3.0 편입 조건: 원래 고지 유지, 수정 고지(§5a, 날짜), 라이선스 전문 동봉, THIRD_PARTY_NOTICES 기록."""

    AGPL_FILES = ("__init__.py", "anima.py", "sd.py", "utils.py", "mask.py")

    def test_licence_text_ships_next_to_the_code(self):
        data = (VENDOR / "LICENSE").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(data).hexdigest(), LICENSE_SHA256)
        self.assertTrue(data.lstrip().startswith(b"GNU AFFERO GENERAL PUBLIC LICENSE"))

    def test_every_vendored_file_keeps_the_agpl_header_and_a_modification_notice(self):
        for path in [*(VENDOR / name for name in self.AGPL_FILES), SCRIPT]:
            text = path.read_text(encoding="utf-8")
            with self.subTest(file=path.name):
                self.assertIn("Copyright (C) 2025 hako-mikan", text)
                self.assertIn("Copyright (C) 2026 Haoming02", text)
                self.assertIn("GNU Affero General Public License", text)
                self.assertIn("MODIFIED by sam-extra, 2026-09-30", text)
                self.assertIn("https://github.com/Haoming02/sd-forge-negpip", text)
                self.assertIn("0585496", text)

    def test_every_file_in_the_folder_is_accounted_for(self):
        own = {"coexist.py"}   # sam-extra 자체 코드(GPL-3.0) — 파일 머리에 밝힌다
        names = {p.name for p in VENDOR.iterdir() if p.is_file()}
        self.assertEqual(names - {"LICENSE"}, set(self.AGPL_FILES) | own)
        self.assertIn("sam-extra 자체 코드", (VENDOR / "coexist.py").read_text(encoding="utf-8"))

    def test_licence_statements_do_not_claim_the_whole_tree_is_gpl_only(self):
        # 편입한 NegPiP 파일은 AGPL-3.0-or-later — 저장소 전체를 GPL-3.0-only 라고 적으면 틀린 고지다
        import json

        self.assertEqual(json.loads((ROOT / "package.json").read_text(encoding="utf-8"))["license"],
                         "GPL-3.0-only AND AGPL-3.0-or-later")
        for rel in ("README.md", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            with self.subTest(file=rel):
                self.assertNotIn("이 확장 전체는", text)
                self.assertNotIn("이 확장 전체의 라이선스", text)
                self.assertIn("AGPL-3.0-or-later", text)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        section = readme[readme.index("## 라이선스"):]
        self.assertIn("결합된 작업", section, "AGPL-3.0 13조 네트워크 요건은 결합된 작업 그 자체에 적용(GPL-3.0 13조)")
        self.assertIn("the combination as such", (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8"))

    def test_third_party_notices_record_the_vendoring(self):
        text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        for needle in ("sd-forge-negpip", "AGPL-3.0", "0585496", "hako-mikan", "Haoming02", "sam3ext/negpip/LICENSE",
                       "2026-09-30", "scripts/negpip.py"):
            with self.subTest(needle=needle):
                self.assertTrue(needle in text, f"THIRD_PARTY_NOTICES.md 에 {needle!r} 가 없다")


if __name__ == "__main__":
    unittest.main()
