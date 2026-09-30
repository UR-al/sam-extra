"""내장 NegPiP 의 SD1/SDXL 음수 항 조건 행 — Forge 옛·새 CLIP 텍스트 엔진에서 같은 [단어…, EOS] 행을 고르는가.

상류 _cond_dealer 는 "(단어:w)" 조건에서 cond[1 : token_len + 2] 를 자른다. 옛 엔진(classic_engine.ClassicTextProcessingEngine,
Forge ad88b6b4 까지)은 tokenize 가 add_special_tokens=False 라 이것이 [단어…, EOS] 다. 새 엔진(sd_engine.ClipEngine, 21886f41~)은
조각마다 BOS/EOS 가 붙어 같은 자르기가 [BOS, 단어…, EOS, EOS] 가 되고 NegPiP 가 BOS(어텐션 싱크) 행의 V 를 뒤집는다.
내장은 엔진 자신에게 물어(빈 글자에 특수 토큰이 나오는가) 새 엔진에서만 옛 엔진과 같은 행을 고른다(sam3ext/negpip/utils.py).

- 순수 로직: clip_fragment_specials·clip_word_rows (가짜 청크).
- 실제 SD1.5 CLIP 토크나이저 + 실제 Forge 엔진 코드(새: 작업 트리 sd_engine, 옛: 작업 트리에 없으면 git show ad88b6b4)
  + 가짜 인코더(행 = 토큰 id)로 스크립트의 process_batch → _cond_dealer 를 끝까지 돌린다. 없으면 건너뛴다.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import logging
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext.negpip import utils as negpip_utils  # noqa: E402
import sam3ext.negpip as negpip_pkg  # noqa: E402

FORGE = ROOT.parents[1]
SCRIPT = ROOT / "scripts" / "negpip.py"
SD15_TOKENIZER = FORGE / "backend" / "huggingface" / "runwayml" / "stable-diffusion-v1-5" / "tokenizer"
OLD_FORGE = "ad88b6b4"   # classic_engine 이 있던 마지막 Forge
OLD_ENGINE_FILES = ("classic_engine.py", "emphasis.py", "parsing.py")

BOS, EOS = 49406, 49407


def _chunk(tokens):
    return types.SimpleNamespace(tokens=list(tokens))


def _pad(tokens, width=77):
    return tokens + [EOS] * (width - len(tokens))


class WordRowTests(unittest.TestCase):
    """가짜 청크로 행 번호 규칙 — 옛 엔진 청크에선 상류 자르기와 같고, 새 엔진 청크에선 특수 토큰을 건너뛴다."""

    SPECIALS = frozenset({BOS, EOS})

    def test_old_engine_chunk_gives_the_upstream_slice(self):
        for words in ([10], [10, 11], list(range(10, 20))):
            with self.subTest(words=len(words)):
                chunk = _chunk(_pad([BOS, *words]))
                token_len = len(words)
                self.assertEqual(negpip_utils.clip_word_rows([chunk], self.SPECIALS), list(range(1, token_len + 2)))

    def test_new_engine_chunk_skips_the_fragment_bos_and_keeps_the_eos_after_the_words(self):
        chunk = _chunk(_pad([BOS, BOS, 10, 11, EOS]))
        self.assertEqual(negpip_utils.clip_word_rows([chunk], self.SPECIALS), [2, 3, 4])

    def test_several_fragments_drop_the_inner_special_rows(self):
        # "a [b]" 처럼 가중치가 다른 조각 — 새 엔진은 조각마다 BOS/EOS 를 붙인다
        chunk = _chunk(_pad([BOS, BOS, 10, EOS, BOS, 11, 12, EOS]))
        self.assertEqual(negpip_utils.clip_word_rows([chunk], self.SPECIALS), [2, 5, 6, 7])

    def test_embedding_placeholders_are_word_rows(self):
        # 텍스처 인버전 자리는 토큰 0 — 특수 토큰이 아니다
        chunk = _chunk(_pad([BOS, BOS, 0, 0, 10, EOS]))
        self.assertEqual(negpip_utils.clip_word_rows([chunk], self.SPECIALS), [2, 3, 4, 5])

    def test_rows_continue_across_chunks(self):
        first = _chunk([BOS, BOS, *range(100, 174), EOS])   # 77 행: 시작 BOS·조각 BOS·단어 74·끝 EOS
        self.assertEqual(len(first.tokens), 77)
        second = _chunk(_pad([BOS, 200, 201, EOS]))
        rows = negpip_utils.clip_word_rows([first, second], self.SPECIALS)
        self.assertEqual(rows[:2], [2, 3])
        self.assertEqual(rows[-3:], [78, 79, 80])
        self.assertNotIn(76, rows)   # 첫 청크 끝 EOS
        self.assertNotIn(77, rows)   # 둘째 청크 BOS

    def test_no_words_gives_the_eos_after_the_start_tokens(self):
        self.assertEqual(negpip_utils.clip_word_rows([_chunk(_pad([BOS]))], self.SPECIALS), [1])        # 옛 엔진 cond[1:2]
        self.assertEqual(negpip_utils.clip_word_rows([_chunk(_pad([BOS, BOS, EOS]))], self.SPECIALS), [2])

    def test_specials_come_from_the_engine(self):
        class New:
            id_start, id_end = BOS, EOS

            def tokenize(self, texts):
                return [[BOS, EOS] for _ in texts]

        class Old(New):
            def tokenize(self, texts):
                return [[] for _ in texts]

        class Broken(New):
            def tokenize(self, texts):
                raise RuntimeError("no tokenizer")

        self.assertEqual(negpip_utils.clip_fragment_specials(New()), frozenset({BOS, EOS}))
        self.assertIsNone(negpip_utils.clip_fragment_specials(Old()))
        self.assertIsNone(negpip_utils.clip_fragment_specials(Broken()))
        self.assertIsNone(negpip_utils.clip_fragment_specials(object()))
        self.assertIsNone(negpip_utils.clip_fragment_specials(None))


# ---- 실제 Forge 엔진 코드 + 실제 SD1.5 토크나이저 ----

@contextlib.contextmanager
def _forge_text_modules(backend_root: Path):
    """backend.text_processing 을 backend_root 의 실제 파일로, 나머지(memory_management·args·opts·TI·_comfy)만 대역으로 잠시 올린다."""
    names = [n for n in sys.modules if n in ("backend", "modules") or n.startswith(("backend.", "modules."))]
    saved = {n: sys.modules.pop(n) for n in names}
    backend = types.ModuleType("backend")
    backend.__path__ = [str(backend_root)]
    mm = types.ModuleType("backend.memory_management")
    mm.text_encoder_device = lambda: torch.device("cpu")
    mm.intermediate_device = lambda: torch.device("cpu")
    mm.logger = logging.getLogger("negpip-clip-test")
    args = types.ModuleType("backend.args")
    args.dynamic_args = types.SimpleNamespace(last_extra_generation_params={})
    tp = types.ModuleType("backend.text_processing")
    tp.__path__ = [str(backend_root / "text_processing")]
    ti = types.ModuleType("backend.text_processing.textual_inversion")
    ti.EmbeddingDatabase = object
    comfy = types.ModuleType("backend.text_processing._comfy")
    comfy.EMBEDDINGS = list
    modules = types.ModuleType("modules")
    modules.__path__ = []
    shared = types.ModuleType("modules.shared")
    shared.opts = types.SimpleNamespace(emphasis="Original", comma_padding_backtrack=20)
    modules.shared = shared
    sys.modules.update({
        "backend": backend, "backend.memory_management": mm, "backend.args": args, "backend.text_processing": tp,
        "backend.text_processing.textual_inversion": ti, "backend.text_processing._comfy": comfy,
        "modules": modules, "modules.shared": shared,
    })
    try:
        yield shared.opts
    finally:
        for name in [n for n in sys.modules if n in ("backend", "modules") or n.startswith(("backend.", "modules."))]:
            del sys.modules[name]
        sys.modules.update(saved)


def _fake_encode(tokens: torch.Tensor) -> torch.Tensor:
    """가짜 CLIP — 행 i = [토큰 id, i, 1]. emphasis 가 행마다 같은 배수를 곱하므로 id = 행[0] / 행[2]."""
    batch, width = tokens.shape
    position = torch.arange(width, dtype=torch.float32).expand(batch, width)
    return torch.stack([tokens.to(torch.float32), position, torch.ones(batch, width)], dim=-1)


def _make_engine(module_name: str, class_name: str, backend_root: Path, tokenizer):
    with _forge_text_modules(backend_root) as opts:
        emphasis = importlib.import_module("backend.text_processing.emphasis")
        module = importlib.import_module(f"backend.text_processing.{module_name}")
        engine = object.__new__(getattr(module, class_name))
    # __init__ 은 실제 CLIP 인코더를 감싸므로 건너뛰고, 토큰화에 쓰는 속성만 __init__ 과 같게 둔다
    engine.tokenizer = tokenizer
    engine.chunk_length = 75
    engine.id_start, engine.id_end, engine.id_pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    engine.comma_token = tokenizer.get_vocab()[",</w>"]
    engine.embeddings = types.SimpleNamespace(find_embedding_at_position=lambda tokens, position: (None, None), fixes=None)
    engine.embedding_key = "clip_l"
    engine.return_pooled = False
    engine.encode_with_transformers = _fake_encode
    if "emphasis" not in type(engine).__dict__:   # 옛 엔진은 인스턴스 속성(__init__·__call__ 이 opts 로 맞춘다)
        engine.emphasis = emphasis.get_current_option(opts.emphasis)()
    return engine


def _old_engine_root(tmp: Path) -> Path | None:
    """옛 classic_engine 이 든 backend 폴더 — 작업 트리가 옛 Forge 면 그대로, 아니면 git show ad88b6b4 로 꺼낸다."""
    if (FORGE / "backend" / "text_processing" / "classic_engine.py").is_file():
        return FORGE / "backend"
    git = shutil.which("git")
    if git is None or not (FORGE / ".git").exists():
        return None
    target = tmp / "backend" / "text_processing"
    target.mkdir(parents=True)
    for name in OLD_ENGINE_FILES:
        result = subprocess.run(
            [git, "-C", str(FORGE), "show", f"{OLD_FORGE}:backend/text_processing/{name}"],
            capture_output=True, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            return None
        (target / name).write_text(result.stdout, encoding="utf-8")
    return tmp / "backend"


class _Script:
    def __init__(self):
        pass


def _load_script(engine):
    """스크립트를 Forge 없이 불러온다 — get_learned_conditioning 은 실제 엔진(가짜 인코더)을 부른다."""
    prompt_parser = types.ModuleType("modules.prompt_parser")
    prompt_parser.SdConditioning = lambda prompts, width=None, height=None: list(prompts)
    prompt_parser.get_learned_conditioning_prompt_schedules = lambda prompts, steps: [[[steps, x]] for x in prompts]

    def get_learned_conditioning(model, prompts, steps):
        cond = model.text_processing_engine(list(prompts))[0]
        if model.is_sdxl:
            cond = {"crossattn": cond}
        return [[types.SimpleNamespace(end_at_step=steps, cond=cond)]]

    prompt_parser.get_learned_conditioning = get_learned_conditioning
    modules = types.ModuleType("modules")
    modules.__path__ = []
    modules.scripts = types.SimpleNamespace(Script=_Script, AlwaysVisible=object())
    modules.shared = types.SimpleNamespace(opts=types.SimpleNamespace(emphasis="Original"))
    callbacks = types.ModuleType("modules.script_callbacks")
    callbacks.CFGDenoiserParams = object
    callbacks.on_cfg_denoiser = lambda callback: None
    anima = types.ModuleType("sam3ext.negpip.anima")
    anima.patch_anima_negpip = lambda cls, *, unpatch=False: None
    sd = types.ModuleType("sam3ext.negpip.sd")
    sd.patch_sd_negpip = lambda instance, cls, *, unpatch=False: None
    stubs = {
        "modules": modules, "modules.scripts": modules.scripts, "modules.shared": modules.shared,
        "modules.prompt_parser": prompt_parser, "modules.script_callbacks": callbacks,
        "sam3ext.negpip.anima": anima, "sam3ext.negpip.sd": sd,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("_test_builtin_negpip_clip_rows", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    module.IS_NEO = True   # Forge Neo (이 PC) — 테스트 프로세스엔 backend 가 없어 패키지는 False 로 읽는다
    return module


class RealClipEngineRowTests(unittest.TestCase):
    """실제 SD1.5 토크나이저 + 실제 Forge 엔진 코드로 process_batch → _cond_dealer 가 고른 행의 토큰 id."""

    TARGETS = ("aqua hair", "bad hands, extra fingers", "red [blue] eyes", "smile")

    @classmethod
    def setUpClass(cls):
        if not SD15_TOKENIZER.is_dir():
            raise unittest.SkipTest(f"SD1.5 토크나이저 없음: {SD15_TOKENIZER}")
        try:
            from transformers import CLIPTokenizer
        except ImportError as exc:   # pragma: no cover - Forge venv 밖
            raise unittest.SkipTest(f"transformers 없음: {exc}")
        cls.tok = CLIPTokenizer.from_pretrained(str(SD15_TOKENIZER))
        cls._tmp = tempfile.TemporaryDirectory()
        cls.engines = {}
        new_root = FORGE / "backend"
        if (new_root / "text_processing" / "sd_engine.py").is_file():
            cls.engines["new"] = _make_engine("sd_engine", "ClipEngine", new_root, cls.tok)
        old_root = _old_engine_root(Path(cls._tmp.name))
        if old_root is not None:
            cls.engines["old"] = _make_engine("classic_engine", "ClassicTextProcessingEngine", old_root, cls.tok)
        if not cls.engines:
            cls._tmp.cleanup()
            raise unittest.SkipTest("Forge CLIP 텍스트 엔진 소스 없음")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self._saved_patched = list(negpip_pkg.PATCHED)
        self.addCleanup(lambda: negpip_pkg.PATCHED.__setitem__(slice(None), self._saved_patched))

    def _engine(self, name):
        if name not in self.engines:
            self.skipTest(f"{name} 엔진 소스 없음")
        return self.engines[name]

    def _words(self, target):
        """옛 엔진이 고르던 행의 토큰: 파싱한 조각들의 단어 토큰(특수 토큰 없이) + EOS 하나."""
        text = target.replace("[", "").replace("]", "")
        return self.tok(text, add_special_tokens=False)["input_ids"] + [EOS]

    def _run(self, engine, target, *, xl=False):
        module = _load_script(engine)
        script = module.NegPiP()
        model = types.SimpleNamespace(
            is_webui_legacy_model=lambda: True, is_sdxl=xl,
            text_processing_engine=engine, text_processing_engine_l=engine,
        )
        p = types.SimpleNamespace(
            sd_model=model, scripts=types.SimpleNamespace(scripts=[], alwayson_scripts=[script]), script_args=[],
            prompts=[f"1girl, ({target}:-1.0)"], negative_prompts=[""], steps=20, batch_size=1, sampler_name="Euler a",
            width=512, height=512, extra_generation_params={}, cached_c=[None] * 3, cached_uc=[None] * 3,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            script.process_batch(p)
        self.assertTrue(script.active)
        conds, rows = script.conds_all[0][0][1][0]
        self.assertEqual(conds.shape[1], rows)
        ids = torch.round(conds[0, :, 0] / conds[0, :, 2]).to(torch.int64).tolist()
        return script, ids

    def test_new_engine_wraps_fragments_and_the_builtin_picks_the_old_rows(self):
        engine = self._engine("new")
        self.assertEqual(engine.tokenize([""])[0], [BOS, EOS], "새 ClipEngine 이 여전히 조각마다 특수 토큰을 붙인다")
        for xl in (False, True):
            for target in self.TARGETS:
                with self.subTest(target=target, xl=xl):
                    script, ids = self._run(engine, target, xl=xl)
                    self.assertEqual(script.clip_specials, frozenset({BOS, EOS}))
                    self.assertEqual(ids, self._words(target))
                    self.assertNotIn(BOS, ids, "BOS(어텐션 싱크) 행의 V 를 뒤집지 않는다")

    def test_upstream_slice_on_the_new_engine_takes_bos_and_two_eos(self):
        # 고치기 전(상류 그대로)의 자르기가 새 엔진에서 무엇을 잡는지 — 이 테스트가 뜻 있는 비교임을 보인다
        engine = self._engine("new")
        _, token_len = engine.tokenize_line("aqua hair")
        chunks, _ = engine.tokenize_line("(aqua hair:1.0)")
        upstream = chunks[0].tokens[1 : token_len + 2]
        words = self.tok("aqua hair", add_special_tokens=False)["input_ids"]
        self.assertEqual(upstream, [BOS, *words, EOS, EOS])

    def test_old_engine_keeps_the_upstream_slice(self):
        engine = self._engine("old")
        self.assertEqual(engine.tokenize([""])[0], [])
        for xl in (False, True):
            for target in self.TARGETS:
                with self.subTest(target=target, xl=xl):
                    script, ids = self._run(engine, target, xl=xl)
                    self.assertIsNone(script.clip_specials, "옛 엔진은 상류 자르기 그대로")
                    self.assertEqual(ids, self._words(target))
                    # 같은 청크에 새 규칙을 대도 상류 자르기와 같은 행 — 두 경로가 같은 뜻이다
                    chunks, token_len = engine.tokenize_line(target)
                    self.assertEqual(negpip_utils.clip_word_rows(chunks, frozenset({BOS, EOS})), list(range(1, token_len + 2)))

    def test_both_engines_give_the_same_rows(self):
        self._engine("new")
        self._engine("old")
        for target in self.TARGETS:
            with self.subTest(target=target):
                self.assertEqual(self._run(self.engines["new"], target)[1], self._run(self.engines["old"], target)[1])


if __name__ == "__main__":
    unittest.main()
