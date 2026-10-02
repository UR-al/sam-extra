"""내장 NegPiP 의 Anima 조건 훅이 Forge 의 줄별 계약을 지키는가 — 프롬프트 편집 줄 길이가 512 를 넘어 서로 달라도.

Forge 의 ``model.get_learned_conditioning(SdConditioning)`` 은 스케줄 줄마다 항목 하나를 돌려준다(prompt_parser 가 list 면
``conds[i]``, dict 면 ``{k: v[i]}``). Anima 엔진(새 Qwen06Engine·옛 AnimaTextProcessingEngine)은 줄마다 ``max(512, T5 토큰 수)`` 행이라
``[짧은:아주 긴:0.5]`` 같은 줄들은 길이가 다르다. 순정은 스텝마다 한 줄을 골라 문제없다. 상류 0585496 의 훅은 모든 줄을 torch.stack
해 dict 로 돌려줘 ``stack expects each tensor to be equal size`` 로 조건 단계에서 죽었다. 내장은 줄마다 dict 하나를 돌려준다.

실제 Forge Qwen06Engine + 실제 Anima Qwen2/T5 토크나이저 + 실제 modules/prompt_parser.py (CPU). 가짜는 TE 트랜스포머 하나뿐 —
preprocess_text_embeds 가 T5 토큰마다 행 하나 [id, 1] 을 낸다. 없으면 건너뛴다.
"""
from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.util
import io
import logging
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext.anima38.native_engine import ANIMA_ENGINE_ATTRS  # noqa: E402
from sam3ext.negpip import mask as negpip_mask  # noqa: E402

FORGE = ROOT.parents[1]
ANIMA_HOOK = ROOT / "sam3ext" / "negpip" / "anima.py"
ANIMA_MODEL = FORGE / "backend" / "diffusion_engine" / "anima.py"
ANIMA_TOKENIZERS = FORGE / "backend" / "huggingface" / "circlestone-labs" / "Anima"
PROMPT_PARSER = FORGE / "modules" / "prompt_parser.py"

LONG = "(bad:-1), [short:" + "word " * 600 + ":0.5]"     # 스텝 10 까지 512 행, 그 뒤 600 단어 넘는 행
SHORT = "(bad:-1), [short:long words:0.5]"               # 두 줄 모두 512 행 (엔진이 512 로 채움)
STEPS = 20


@contextlib.contextmanager
def _forge_stubs():
    """backend.text_processing·prompt_parser 는 실제 파일, 나머지(memory_management·args·opts·훅이 import 하는 이름)만 대역."""
    names = [n for n in sys.modules if n in ("backend", "modules") or n.startswith(("backend.", "modules."))]
    saved = {n: sys.modules.pop(n) for n in names}

    def module(name, path=None, **attrs):
        m = types.ModuleType(name)
        if path is not None:
            m.__path__ = path
        m.__dict__.update(attrs)
        return m

    backend = module("backend", [str(FORGE / "backend")])
    mm = module("backend.memory_management", intermediate_device=lambda: torch.device("cpu"),
                text_encoder_device=lambda: torch.device("cpu"), logger=logging.getLogger("negpip-anima-schedule"))
    args = module("backend.args", dynamic_args=types.SimpleNamespace(last_extra_generation_params={}))
    tp = module("backend.text_processing", [str(FORGE / "backend" / "text_processing")])
    nn = module("backend.nn", [])
    nn_anima = module("backend.nn.anima", SelfCrossAttention=type("SelfCrossAttention", (), {}))
    sampling = module("backend.sampling", [])
    sampling.condition = module("backend.sampling.condition")
    sampling.sampling_function = module("backend.sampling.sampling_function")
    modules = module("modules", [str(FORGE / "modules")])
    shared = module("modules.shared", opts=types.SimpleNamespace(emphasis="Original"), sd_model=None)
    backend.memory_management, backend.args, backend.text_processing, backend.nn = mm, args, tp, nn
    modules.shared = shared
    sys.modules.update({
        m.__name__: m for m in (backend, mm, args, tp, nn, nn_anima, sampling, sampling.condition,
                                sampling.sampling_function, modules, shared)
    })
    try:
        yield
    finally:
        for name in [n for n in sys.modules if n in ("backend", "modules") or n.startswith(("backend.", "modules."))]:
            del sys.modules[name]
        sys.modules.update(saved)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


class _FakeQwen(torch.nn.Module):
    """0.6B TE 자리 — 토크나이저·파서·엔진은 실제. 어댑터처럼 T5 토큰마다 행 하나(폭 2: [T5 id, 1])를 낸다."""

    num_layers = 2

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(151936, 4)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None,
                final_layer_norm_intermediate=None, dtype=None, embeds_info=None):
        return embeds, None

    def preprocess_text_embeds(self, cond, t5_ids):
        return torch.stack([t5_ids.float(), torch.ones_like(t5_ids, dtype=torch.float32)], dim=-1)


class AnimaScheduledConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        engine_file = FORGE / "backend" / "text_processing" / "anima_engine.py"
        if not engine_file.is_file() or not (ANIMA_TOKENIZERS / "tokenizer_2").is_dir() or not PROMPT_PARSER.is_file():
            raise unittest.SkipTest("Forge 소스·Anima 토크나이저 없음")
        if "class Qwen06Engine" not in engine_file.read_text(encoding="utf-8"):
            raise unittest.SkipTest("21886f41 이전 Forge (옛 엔진)")
        # 설치된 Forge 의 Anima 가 엔진을 다는 속성 — 2.29.1 까지 text_processing_engine_anima, 2.29.2 부터 _qwen
        model_source = ANIMA_MODEL.read_text(encoding="utf-8") if ANIMA_MODEL.is_file() else ""
        cls.engine_attr = next((a for a in ANIMA_ENGINE_ATTRS if f"self.{a} =" in model_source), None)
        if cls.engine_attr is None:
            raise unittest.SkipTest(f"Anima 엔진 속성을 모른다: {ANIMA_MODEL}")
        try:
            import einops  # noqa: F401  (훅 모듈이 import)
            from transformers import Qwen2Tokenizer, T5TokenizerFast
        except ImportError as exc:   # pragma: no cover - Forge venv 밖
            raise unittest.SkipTest(f"transformers/einops 없음: {exc}")
        qtok = Qwen2Tokenizer.from_pretrained(str(ANIMA_TOKENIZERS / "tokenizer"))
        ttok = T5TokenizerFast.from_pretrained(str(ANIMA_TOKENIZERS / "tokenizer_2"))
        cls._stubs = _forge_stubs()
        cls._stubs.__enter__()
        try:
            engine_module = importlib.import_module("backend.text_processing.anima_engine")
            cls.engine = engine_module.Qwen06Engine(_FakeQwen(), qtok, ttok)
            cls.pp = _load("modules.prompt_parser", PROMPT_PARSER)
            cls.hook = _load("_test_negpip_anima_schedule_hook", ANIMA_HOOK)
        except BaseException:
            cls._stubs.__exit__(*sys.exc_info())
            raise

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("_test_negpip_anima_schedule_hook", None)
        cls._stubs.__exit__(None, None, None)

    def _model(self, *, hooked, attr=None):
        engine = self.engine

        class Model:
            def get_learned_conditioning(self, prompt):
                return engine(prompt)

        model = Model()
        setattr(model, attr or self.engine_attr, engine)
        if hooked:
            self.hook._hook_get_learned_conditioning(model, False)
        return model

    def _conditioning(self, model, prompts, *, multicond=False):
        prompts = self.pp.SdConditioning(prompts)
        with contextlib.redirect_stdout(io.StringIO()):
            if multicond:
                return self.pp.get_multicond_learned_conditioning(model, prompts, STEPS)
            return self.pp.get_learned_conditioning(model, prompts, STEPS)

    def _mask(self, line, rows):
        return negpip_mask.build_negpip_mask(self.engine, line, rows, "cpu", torch.float32).unsqueeze(-1)

    def test_native_schedule_lines_differ_in_length(self):
        # 전제 — 순정 엔진이 줄마다 다른 길이를 돌려준다(이게 없으면 아래 비교는 뜻이 없다)
        native = self._conditioning(self._model(hooked=False), [LONG])[0]
        self.assertEqual([entry.end_at_step for entry in native], [10, 20])
        self.assertEqual(native[0].cond.shape[1], 512)
        self.assertGreater(native[1].cond.shape[1], 512)

    def test_hook_keeps_each_schedule_line_with_its_own_length(self):
        for prompt in (LONG, SHORT):
            native = self._conditioning(self._model(hooked=False), [prompt])[0]
            hooked = self._conditioning(self._model(hooked=True), [prompt])[0]
            lines = self.pp.get_learned_conditioning_prompt_schedules([prompt], STEPS)[0]
            self.assertEqual(len(hooked), len(native))
            for (end, line), n, h in zip(lines, native, hooked):
                with self.subTest(prompt=prompt[:20], end=end):
                    self.assertEqual(h.end_at_step, n.end_at_step)
                    rows = n.cond.reshape(-1, n.cond.shape[-1])
                    mask = self._mask(line, rows.shape[0])
                    self.assertGreater(int((mask < 0).sum()), 0, "(bad:-1) 행이 뒤집힌다")
                    self.assertEqual(tuple(h.cond["c_negpip_mask"].shape), (rows.shape[0], 1))
                    self.assertTrue(torch.equal(h.cond["c_negpip_mask"], mask))
                    self.assertTrue(torch.equal(h.cond["crossattn"], rows * mask))

    def test_each_step_matches_native_times_the_mask(self):
        native = self._conditioning(self._model(hooked=False), [LONG], multicond=True)
        hooked = self._conditioning(self._model(hooked=True), [LONG], multicond=True)
        lines = self.pp.get_learned_conditioning_prompt_schedules([LONG], STEPS)[0]
        for step, line, expected_rows in ((5, lines[0][1], 512), (15, lines[1][1], None)):
            with self.subTest(step=step):
                _, n = self.pp.reconstruct_multicond_batch(native, step)
                _, h = self.pp.reconstruct_multicond_batch(hooked, step)
                rows = n.reshape(1, -1, n.shape[-1])
                if expected_rows is not None:
                    self.assertEqual(rows.shape[1], expected_rows, "512 행 이하 줄은 순정과 같은 512 행 — 패딩을 늘리지 않는다")
                mask = self._mask(line, rows.shape[1]).unsqueeze(0)
                self.assertEqual(tuple(h["crossattn"].shape), tuple(rows.shape))
                self.assertTrue(torch.equal(h["c_negpip_mask"], mask))
                self.assertTrue(torch.equal(h["crossattn"], rows * mask))
        self.assertGreater(self.pp.reconstruct_multicond_batch(hooked, 15)[1]["crossattn"].shape[1], 512)

    def test_batch_items_of_different_length_pad_cond_and_mask_alike(self):
        # 배치 안 다른 길이는 prompt_parser.stack_conds 가 끝 행 반복으로 맞춘다 — 조건과 마스크에 같게 적용돼 행이 맞는다
        hooked = self._conditioning(self._model(hooked=True), [LONG, "short (x:-1)"], multicond=True)
        _, h = self.pp.reconstruct_multicond_batch(hooked, 15)
        crossattn, mask = h["crossattn"], h["c_negpip_mask"]
        self.assertEqual(crossattn.shape[:2], mask.shape[:2])
        self.assertEqual(crossattn.shape[0], 2)
        self.assertGreater(crossattn.shape[1], 512)
        self.assertTrue(torch.equal(crossattn[1, 512:], crossattn[1, 511:512].expand(crossattn.shape[1] - 512, -1)))
        self.assertTrue(torch.equal(mask[1, 512:], mask[1, 511:512].expand(mask.shape[1] - 512, -1)))
        self.assertGreater(int((mask[0] < 0).sum()), 0)
        self.assertGreater(int((mask[1] < 0).sum()), 0)

    def test_hook_finds_the_engine_on_either_forge_attribute_name(self):
        # Forge 2.29.2 는 엔진을 text_processing_engine_qwen 에 단다(2.29.1 까지 text_processing_engine_anima) — 같은 조건·마스크
        hooked = {attr: self._conditioning(self._model(hooked=True, attr=attr), [SHORT])[0] for attr in ANIMA_ENGINE_ATTRS}
        first, second = (hooked[attr] for attr in ANIMA_ENGINE_ATTRS)
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a.cond["c_negpip_mask"], b.cond["c_negpip_mask"]))
            self.assertTrue(torch.equal(a.cond["crossattn"], b.cond["crossattn"]))
        self.assertGreater(int((first[0].cond["c_negpip_mask"] < 0).sum()), 0, "(bad:-1) 행이 뒤집힌다")


class NoStackAcrossLinesTests(unittest.TestCase):
    """줄(스케줄 줄)을 torch.stack 하는 코드가 NegPiP 조건 경로에 다시 생기지 않는다 — 훅·공용 헬퍼·3.8B 대행 모두."""

    FUNCTIONS = (
        ("sam3ext/negpip/anima.py", "_hook_get_learned_conditioning"),
        ("sam3ext/negpip/mask.py", "negpip_line_conds"),
        ("sam3ext/anima38/runtime.py", "_apply_negpip"),
    )

    def test_no_torch_stack(self):
        for rel, name in self.FUNCTIONS:
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
            func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = {n.func.attr for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            with self.subTest(file=rel, function=name):
                self.assertNotIn("stack", calls)
                self.assertNotIn("cat", calls)

    def test_both_paths_use_the_shared_helper(self):
        for rel, name in self.FUNCTIONS[::2]:
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
            func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = {n.func.id for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            with self.subTest(file=rel):
                self.assertIn("negpip_line_conds", calls)

    def test_helper_returns_one_dict_per_line(self):
        engine = types.SimpleNamespace()
        conds = [torch.ones(1, 3, 2), torch.ones(1, 5, 2) * 2]

        def build(engine_, line, rows, device, dtype):
            mask = torch.ones(rows, dtype=dtype)
            mask[0] = -1.0
            return mask

        lines, count = negpip_mask.negpip_line_conds(engine, ["a", "b"], conds, build_mask=build)
        self.assertEqual(count, 2)
        self.assertEqual([tuple(x["crossattn"].shape) for x in lines], [(3, 2), (5, 2)])
        self.assertEqual([tuple(x["c_negpip_mask"].shape) for x in lines], [(3, 1), (5, 1)])
        self.assertEqual(lines[1]["crossattn"][:, 0].tolist(), [-2.0, 2.0, 2.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
