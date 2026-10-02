"""내장 NegPiP 의 SD1/SDXL 어텐션 훅 — 배치의 행마다 cond/uncond 와 항목을 Forge 가 넘기는 표시로 가려 제 음수 항 전부를 붙이는가.

상류 0585496(과 편입 첫 판)에서 확인된 세 가지(모두 상류부터 있던 것):

- 항이 둘 이상이면 첫 항만 어텐션에 들어갔다 — 조건 목록의 [0] 만 붙여서. 나머지 항은 프롬프트에서 지워지고 어디에도 없었다.
  부정 쪽은 break 가 안쪽 반복에 있어 `[(x:-1):5]` 가 0 스텝부터 걸렸다.
- 배치 항목 0 의 스케줄만 골라 batch_size 로 반복했다 — sd-dynamic-prompts 처럼 항목마다 프롬프트가 다르면 뒤 항목은 항목 0 의
  음수 항을 받거나 제 음수 항을 잃었다.
- cond/uncond 를 샘플러 이름·2*batch_size·모듈마다의 호출 수·문맥 길이로 추정했다 — Forge 가 메모리 부족·문맥 길이 lcm 비 > 4 로
  U·C 를 따로 돌리거나 CFG 1 로 C 만 돌리면 8 스텝씩 켜졌다 꺼지거나 반대쪽에 붙었다. DDIM/PLMS/UniPC 는 묶음에서도 반대 절반.

실제 Forge 소스: backend/sampling/sampling_function.py(calc_cond_uncond_batch — 조각 묶기·나누기·표시), backend/sampling/
condition.py(lcm 반복), backend/nn/unet.py(CrossAttention), modules/prompt_parser.py, backend/text_processing/parsing.py 와 SD1.5
CLIP 토크나이저. 대역: 어텐션 함수(SDPA — Forge attention_pytorch 와 같은 모양), 메모리 조회, 텍스트 인코더(토큰 id → 무작위 행).
기대값은 훅과 따로 계산한다: 행마다 [원래 문맥 + 그 항목의 음수 항 행] 에 음수 항 행의 V 만 뒤집은 어텐션, 항이 없으면 원래 어텐션.
"""
from __future__ import annotations

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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sam3ext.negpip as negpip_pkg  # noqa: E402

FORGE = ROOT.parents[1]
SD_HOOK = ROOT / "sam3ext" / "negpip" / "sd.py"
SCRIPT = ROOT / "scripts" / "negpip.py"
SAMPLING = FORGE / "backend" / "sampling" / "sampling_function.py"
UNET = FORGE / "backend" / "nn" / "unet.py"
PROMPT_PARSER = FORGE / "modules" / "prompt_parser.py"
PARSING = FORGE / "backend" / "text_processing" / "parsing.py"
SD15_TOKENIZER = FORGE / "backend" / "huggingface" / "runwayml" / "stable-diffusion-v1-5" / "tokenizer"

COND, UNCOND = 0, 1
DIM, CDIM, HEADS, TOKENS = 16, 8, 2, 3   # 질의 폭, 문맥 폭, 헤드, 질의 토큰 수 (작게 — CPU)


def attention_function(q, k, v, heads, mask=None, **kwargs):
    """Forge attention_pytorch 와 같은 모양: [B, L, H*d] → 헤드로 나눠 SDPA → [B, L, H*d]."""
    b = q.shape[0]
    dh = q.shape[-1] // heads
    q, k, v = (t.reshape(b, -1, heads, dh).transpose(1, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).reshape(b, -1, heads * dh)


class _Memory:
    free = 1 << 40


@contextlib.contextmanager
def _forge_stubs():
    """sampling_function·condition·unet·prompt_parser 는 실제 파일, 그들이 import 하는 무거운 모듈(메모리·인자·utils·어텐션)만 대역."""
    names = [n for n in sys.modules if n in ("backend", "modules") or n.startswith(("backend.", "modules."))]
    saved = {n: sys.modules.pop(n) for n in names}

    def module(name, path=None, **attrs):
        m = types.ModuleType(name)
        if path is not None:
            m.__path__ = path
        m.__dict__.update(attrs)
        return m

    backend = module("backend", [str(FORGE / "backend")])
    mm = module(
        "backend.memory_management", signal_empty_cache=False, soft_empty_cache=lambda *a, **k: None,
        get_free_memory=lambda *a, **k: _Memory.free, logger=logging.getLogger("negpip-sd-batching"),
    )
    args = module("backend.args", args=types.SimpleNamespace(disable_gpu_warning=True),
                  dynamic_args=types.SimpleNamespace(context_handler=None))

    def join_dicts(base, update):
        result = dict(base or {})
        result.update(update or {})
        return result

    utils = module("backend.utils", join_dicts=join_dicts)
    attention = module("backend.attention", attention_function=attention_function)
    shared_backend = module("backend.shared")
    modules = module("modules", [str(FORGE / "modules")])
    shared = module("modules.shared", opts=types.SimpleNamespace(emphasis="Original"), sd_model=None)
    backend.memory_management, backend.args, backend.utils, backend.attention = mm, args, utils, attention
    backend.shared = shared_backend
    modules.shared = shared
    sys.modules.update({m.__name__: m for m in (backend, mm, args, utils, attention, shared_backend, modules, shared)})
    try:
        yield shared
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


class _Harness:
    """Forge 소스와 내장 훅을 한 번 불러 둔다 (클래스마다)."""

    @classmethod
    def setUpClass(cls):
        for path in (SAMPLING, UNET, SD_HOOK):
            if not path.is_file():
                raise unittest.SkipTest(f"Forge 소스 없음: {path}")
        try:
            import einops  # noqa: F401  (unet.py 가 import)
        except ImportError as exc:   # pragma: no cover - Forge venv 밖
            raise unittest.SkipTest(f"einops 없음: {exc}")
        cls._stubs = _forge_stubs()
        cls.shared = cls._stubs.__enter__()
        saved_neo = negpip_pkg.IS_NEO
        negpip_pkg.IS_NEO = True   # 테스트 프로세스엔 진짜 backend 가 없어 패키지는 False 로 읽었다 — 훅은 Forge Neo 경로로
        try:
            cls.sf = importlib.import_module("backend.sampling.sampling_function")
            cls.unet = importlib.import_module("backend.nn.unet")
            cls.sd = _load("_test_negpip_sd_batching_hook", SD_HOOK)
        except BaseException:
            cls._stubs.__exit__(*sys.exc_info())
            raise
        finally:
            negpip_pkg.IS_NEO = saved_neo

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("_test_negpip_sd_batching_hook", None)
        cls._stubs.__exit__(None, None, None)

    def setUp(self):
        self._saved_patched = list(negpip_pkg.PATCHED)
        self.addCleanup(lambda: negpip_pkg.PATCHED.__setitem__(slice(None), self._saved_patched))
        self.sd._warned.clear()

    # ---- 가짜 UNet: 실제 CrossAttention 을 attn2 로 여럿, apply_model 은 Forge 처럼 transformer_options 를 넘긴다 ----

    def _diffusion(self, count=4, seed=0):
        torch.manual_seed(seed)
        root = torch.nn.Module()
        root.blocks = torch.nn.ModuleList()
        for _ in range(count):
            block = torch.nn.Module()
            block.attn2 = self.unet.CrossAttention(DIM, CDIM, heads=HEADS, dim_head=DIM // HEADS)
            root.blocks.append(block)
        return root

    def _patch(self, owner, diffusion):
        self.shared.sd_model = types.SimpleNamespace(
            forge_objects=types.SimpleNamespace(unet=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=diffusion)))
        )
        self.sd.patch_sd_negpip(owner, owner)
        self.addCleanup(lambda: self.sd.patch_sd_negpip(owner, owner, unpatch=True))

    class _Model:
        """calc_cond_uncond_batch 가 부르는 모델 — attn2 를 BasicTransformerBlock 처럼 부르고 입출력을 적는다."""

        def __init__(self, diffusion, split=False):
            self.diffusion, self.split, self.calls = diffusion, split, []

        def memory_required(self, shape):
            return (shape[0] * 60) if self.split else 1

        def apply_model(self, x, t, c_crossattn=None, transformer_options=None, **kwargs):
            labels = list(transformer_options["cond_or_uncond"])
            gen = torch.Generator().manual_seed(1234 + len(self.calls))
            h = torch.randn(x.shape[0], TOKENS, DIM, generator=gen)
            for block in self.diffusion.blocks:
                out = block.attn2(h, context=c_crossattn, transformer_options=dict(transformer_options))
                self.calls.append((labels, block.attn2, h, c_crossattn, out))
            return x

    # ---- 기대값 (훅과 따로) ----

    @staticmethod
    def _reference(attn, h, native, extra):
        """한 행: 원래 문맥(native [L, D]) + 음수 항 행(extra [n, D] | None), 붙인 행의 V 만 부호를 뒤집는다."""
        with torch.no_grad():
            return _Harness._reference_rows(attn, h, native, extra)

    @staticmethod
    def _reference_rows(attn, h, native, extra):
        ctx = native if extra is None else torch.cat([native, extra], 0)
        q = attn.to_q(h.unsqueeze(0))
        k = attn.to_k(ctx.unsqueeze(0))
        v = attn.to_v(ctx.unsqueeze(0))
        if extra is not None:
            v = torch.cat([v[:, : native.shape[0]], -v[:, native.shape[0] :]], 1)
        return attn.to_out(attention_function(q, k, v, attn.heads))[0]

    def _check_calls(self, calls, expected, native_len, *, atol=1e-5):
        """모든 attn2 호출의 모든 행이 기대값과 같다. expected[label][item] = 행 | None, native_len[label] = 원래 문맥 길이."""
        seen = set()
        for labels, attn, h, ctx, out in calls:
            per_chunk = h.shape[0] // len(labels)
            for chunk, label in enumerate(labels):
                for item in range(per_chunk):
                    row = chunk * per_chunk + item
                    native = ctx[row, : native_len[label]]
                    want = self._reference(attn, h[row], native, expected[label][item])
                    with self.subTest(labels=labels, row=row):
                        torch.testing.assert_close(out[row], want, atol=atol, rtol=1e-4)
                    seen.add((label, item))
        return seen


def _owner(**attrs):
    """스크립트 인스턴스 자리 — 훅이 읽는 속성만 (denoiser_callback 이 채우는 모양)."""
    base = dict(_patched=negpip_pkg.PATCHED, is_xl=False, batch_size=1, rev=True, active=True, c_len=1, uc_len=1,
                conds=[], c_tokens=[], unconds=[], uc_tokens=[], c_native=0, uc_native=0)
    base.update(attrs)
    return types.SimpleNamespace(**base)


class HookBatchLayoutTests(_Harness, unittest.TestCase):
    """실제 calc_cond_uncond_batch 가 만드는 배치 모양마다: C 조각의 행은 제 항목의 긍정 쪽 음수 항, U 조각의 행은 부정 쪽."""

    B = 2
    STEPS = 20   # 상류 Counter 는 모듈마다 16(SD1) 호출에서 뒤집혔다 — 16 을 넘겨 돌린다

    def _items(self, seed):
        gen = torch.Generator().manual_seed(seed)
        pos = [torch.randn(3, CDIM, generator=gen), None]            # 항목 0: 두 항(3 행), 항목 1: 없음
        neg = [None, torch.randn(2, CDIM, generator=gen)]            # 항목 0: 없음, 항목 1: 한 항(2 행)
        return pos, neg

    def _run(self, *, c_chunks=1, uc_chunks=1, split=False, cfg1=False, rev=True, xl=False, and_parts=1, seed=0):
        pos, neg = self._items(seed)
        owner = _owner(is_xl=xl, batch_size=self.B, rev=rev, conds=pos, c_tokens=[0 if r is None else r.shape[0] for r in pos],
                       unconds=neg, uc_tokens=[0 if r is None else r.shape[0] for r in neg],
                       c_native=77 * c_chunks, uc_native=77 * uc_chunks, c_len=c_chunks, uc_len=uc_chunks)
        diffusion = self._diffusion(seed=seed)
        self._patch(owner, diffusion)
        gen = torch.Generator().manual_seed(seed + 7)
        c = torch.randn(self.B * and_parts, 77 * c_chunks, CDIM, generator=gen)
        uc = torch.randn(self.B, 77 * uc_chunks, CDIM, generator=gen)
        if and_parts > 1:
            weights = [[(b * and_parts + j, 1.0) for j in range(and_parts)] for b in range(self.B)]
            cond = self.sf.compile_weighted_conditions(c, weights)
        else:
            cond = self.sf.compile_conditions(c)
        uncond = None if cfg1 else self.sf.compile_conditions(uc)
        model = self._Model(diffusion, split=split)
        _Memory.free = 100 if split else 1 << 40
        try:
            x = torch.zeros(self.B, 4, 8, 8)
            for _ in range(self.STEPS):
                self.sf.calc_cond_uncond_batch(model, cond, uncond, x, torch.tensor([1.0]), {})
        finally:
            _Memory.free = 1 << 40
            self.sd.patch_sd_negpip(owner, owner, unpatch=True)   # 한 테스트에서 여러 번 돌린다 — 다음 _run 이 새로 건다
        return model.calls, {COND: pos, UNCOND: neg}, {COND: 77 * c_chunks, UNCOND: 77 * uc_chunks}

    def _orders(self, calls):
        orders = []
        for labels, *_ in calls:
            if not orders or orders[-1] != labels:
                orders.append(labels)
        return orders

    def test_combined_batch(self):
        for rev in (True, False):   # 상류는 DDIM/PLMS/UniPC(rev=False)에서 [U, C] 묶음의 반대 절반에 붙였다
            with self.subTest(rev=rev):
                calls, expected, native = self._run(rev=rev)
                self.assertEqual(self._orders(calls)[0], [UNCOND, COND], "Forge 는 샘플러와 무관하게 [U, C] 로 묶는다")
                seen = self._check_calls(calls, expected, native)
                self.assertEqual(seen, {(COND, 0), (COND, 1), (UNCOND, 0), (UNCOND, 1)})

    def test_memory_split_runs_u_then_c(self):
        calls, expected, native = self._run(split=True)
        self.assertEqual(self._orders(calls)[:2], [[UNCOND], [COND]])
        self._check_calls(calls, expected, native)
        self.assertEqual(len(calls), self.STEPS * 2 * 4)

    def test_shape_split_runs_c_then_u(self):
        # 1 청크 vs 5 청크 — lcm 비 5 > 4 라 ConditionCrossAttn.can_concat 이 거짓: 메모리와 무관하게 C, U 따로
        calls, expected, native = self._run(c_chunks=1, uc_chunks=5)
        self.assertEqual(self._orders(calls)[:2], [[COND], [UNCOND]])
        self._check_calls(calls, expected, native)

    def test_cfg1_runs_cond_only(self):
        calls, expected, native = self._run(cfg1=True)
        self.assertEqual({tuple(labels) for labels, *_ in calls}, {(COND,)})
        self.assertEqual(self._check_calls(calls, expected, native), {(COND, 0), (COND, 1)})

    def test_and_prompt_has_several_cond_chunks(self):
        calls, expected, native = self._run(and_parts=2)
        self.assertEqual(self._orders(calls)[0], [UNCOND, COND, COND])
        self._check_calls(calls, expected, native)

    def test_sdxl_counter_limit_does_not_matter(self):
        calls, expected, native = self._run(xl=True, split=True)
        self._check_calls(calls, expected, native)

    def test_lcm_repeated_context_keeps_the_native_strength(self):
        # 1 청크 vs 2 청크: Forge 가 cond 문맥을 2 번 반복해 [U, C] 로 묶는다. 붙인 행을 한 번만 붙이면(상류) softmax 몫이 반으로 준다
        calls, expected, native = self._run(c_chunks=1, uc_chunks=2)
        self.assertEqual(self._orders(calls)[0], [UNCOND, COND])
        self.assertEqual(calls[0][3].shape[1], 154)
        self._check_calls(calls, expected, native)
        labels, attn, h, ctx, out = calls[0]
        row = self.B   # C 조각의 항목 0 (항이 있음)
        once = self._reference(attn, h[row], ctx[row], expected[COND][0])   # 반복한 문맥 뒤에 한 번만 붙이면
        self.assertGreater(float((once - out[row]).abs().max()), 1e-3, "비교가 뜻이 있다 — 반복 문맥 뒤 한 번 붙이기는 다르다")

    def test_rows_without_terms_equal_the_unhooked_forward(self):
        # 항이 없는 항목은 NegPiP 가 없을 때와 똑같다 — 원래 forward 를 부른다
        calls, _, _ = self._run()
        # 같은 계산을 행 부분집합으로 돌릴 뿐이라 차이는 배치 크기에 따른 부동소수 반올림뿐이다(Forge 가 메모리로 U·C 를
        # 나눌 때와 같은 정도) — CPU 에서 3e-8 수준
        for labels, attn, h, ctx, out in calls[:8]:
            plain = attn(h, context=ctx)   # _run 이 훅을 풀었다 — 원래 forward, 배치 전체
            per_chunk = h.shape[0] // len(labels)
            for chunk, label in enumerate(labels):
                row = chunk * per_chunk + (1 if label == COND else 0)   # C 항목 1·U 항목 0 은 항이 없다
                torch.testing.assert_close(out[row], plain[row], atol=1e-6, rtol=0)

    def test_no_terms_anywhere_is_a_plain_call(self):
        owner = _owner(batch_size=2, conds=[None, None], c_tokens=[0, 0], unconds=[None, None], uc_tokens=[0, 0])
        diffusion = self._diffusion(count=1)
        attn = diffusion.blocks[0].attn2
        self._patch(owner, diffusion)
        h, ctx = torch.randn(4, TOKENS, DIM), torch.randn(4, 77, CDIM)
        options = {"cond_or_uncond": [UNCOND, COND]}
        self.assertTrue(torch.equal(attn(h, context=ctx, transformer_options=options), attn.orig_forward(h, context=ctx)))

    def test_same_terms_are_one_group(self):
        # 같은 텐서를 붙이는 행은 한 번에 계산한다 — 흔한 경우(모든 항목이 같은 프롬프트)에 호출이 늘지 않는다
        rows = torch.randn(2, CDIM)
        owner = _owner(batch_size=2, conds=[rows, rows], c_tokens=[2, 2], unconds=[None, None], uc_tokens=[0, 0])
        diffusion = self._diffusion(count=1)
        attn = diffusion.blocks[0].attn2
        self._patch(owner, diffusion)
        seen = []
        original = self.sd._main_forward
        self.sd._main_forward = lambda *a: seen.append(a[2].shape[0]) or original(*a)
        self.addCleanup(setattr, self.sd, "_main_forward", original)
        attn(torch.randn(4, TOKENS, DIM), context=torch.randn(4, 77, CDIM), transformer_options={"cond_or_uncond": [UNCOND, COND]})
        self.assertEqual(seen, [2])

    def test_value_passed_separately_gets_the_rows_too(self):
        # attn2_patch 가 있으면 BasicTransformerBlock 이 value 를 따로 넘긴다 — K 만 길어지면 모양이 어긋난다(상류)
        rows = torch.randn(2, CDIM)
        owner = _owner(batch_size=1, conds=[rows], c_tokens=[2], unconds=[None], uc_tokens=[0], c_native=77)
        diffusion = self._diffusion(count=1)
        attn = diffusion.blocks[0].attn2
        self._patch(owner, diffusion)
        h, ctx = torch.randn(1, TOKENS, DIM), torch.randn(1, 77, CDIM)
        out = attn(h, context=ctx, value=ctx.clone(), transformer_options={"cond_or_uncond": [COND]})
        torch.testing.assert_close(out[0], self._reference(attn, h[0], ctx[0], rows), atol=1e-5, rtol=1e-4)

    def test_without_labels_falls_back_to_the_upstream_guess_with_one_warning(self):
        # transformer_options 가 없는 호출자 — 상류 추정(2*batch_size 면 rev 로 [U, C])을 쓰되 항목마다 제 행, 경고는 한 번
        pos = [torch.randn(2, CDIM), torch.randn(3, CDIM)]
        owner = _owner(batch_size=2, rev=True, conds=pos, c_tokens=[2, 3], unconds=[None, None], uc_tokens=[0, 0], c_native=77)
        diffusion = self._diffusion(count=1)
        attn = diffusion.blocks[0].attn2
        self._patch(owner, diffusion)
        h, ctx = torch.randn(4, TOKENS, DIM), torch.randn(4, 77, CDIM)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            first = attn(h, context=ctx)
            attn(h, context=ctx)
        self.assertEqual(out.getvalue().count("transformer_options"), 1)
        for item in range(2):
            torch.testing.assert_close(first[2 + item], self._reference(attn, h[2 + item], ctx[2 + item], pos[item]),
                                       atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(first[item], self._reference(attn, h[item], ctx[item], None), atol=1e-5, rtol=1e-4)

    def test_chunk_rows_pair_with_prompts_like_forge_repeat(self):
        # 조각의 행 수가 항목 수와 다른 호출자(타일을 배치로 쌓는 확장 등) — Forge 의 repeat_to_batch_size 처럼 행 r 은 항목 r % n
        pos = [torch.randn(2, CDIM), torch.randn(3, CDIM)]
        owner = _owner(batch_size=2, conds=pos, c_tokens=[2, 3], unconds=[None, None], uc_tokens=[0, 0], c_native=77)
        diffusion = self._diffusion(count=1)
        attn = diffusion.blocks[0].attn2
        self._patch(owner, diffusion)
        h, ctx = torch.randn(4, TOKENS, DIM), torch.randn(4, 77, CDIM)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            got = attn(h, context=ctx, transformer_options={"cond_or_uncond": [COND]})
        self.assertIn("repeat_to_batch_size", log.getvalue())
        condition = sys.modules["backend.sampling.condition"]
        self.assertEqual(condition.repeat_to_batch_size(torch.arange(2), 4).tolist(), [0, 1, 0, 1], "Forge 의 규칙")
        for row in range(4):
            torch.testing.assert_close(got[row], self._reference(attn, h[row], ctx[row], pos[row % 2]), atol=1e-5, rtol=1e-4)

    def test_attention_mask_skips_negpip_with_one_warning(self):
        owner = _owner(batch_size=1, conds=[torch.randn(2, CDIM)], c_tokens=[2], unconds=[None], uc_tokens=[0])
        diffusion = self._diffusion(count=1)
        attn = diffusion.blocks[0].attn2
        self._patch(owner, diffusion)
        h, ctx = torch.randn(1, TOKENS, DIM), torch.randn(1, 77, CDIM)
        mask = torch.zeros(1, TOKENS, 77)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            got = attn(h, context=ctx, mask=mask, transformer_options={"cond_or_uncond": [COND]})
        self.assertIn("attention mask", out.getvalue())
        self.assertTrue(torch.equal(got, attn.orig_forward(h, context=ctx, mask=mask)))


# ---- 스크립트까지: 실제 prompt_parser·Forge 파서·CLIP 토크나이저로 process_batch → denoiser_callback → 실제 배치 → 훅 ----


class _Chunk:
    def __init__(self, tokens):
        self.tokens = tokens


class ContextRepeatTests(_Harness, unittest.TestCase):
    """_context_repeat 의 두 가드 — lcm 반복으로 볼 수 있을 때만 배수를 돌려준다(아니면 붙일 행을 반복하지 않는다)."""

    def _ctx(self, *rows):
        return torch.cat([r.unsqueeze(0) for r in rows], 1)

    def test_true_k_fold_repeat_returns_k(self):
        base = torch.randn(1, 5, CDIM)
        self.assertEqual(self.sd._context_repeat(base.repeat(1, 3, 1), 5), 3)

    def test_multiple_of_native_that_is_not_a_repeat_returns_1(self):
        # attn2_patch(IP-Adapter 등)가 문맥을 native 의 배수 길이로 늘렸지만 같은 행의 반복은 아닌 경우
        torch.manual_seed(1)
        self.assertEqual(self.sd._context_repeat(torch.randn(1, 10, CDIM), 5), 1)

    def test_length_not_a_multiple_of_native_returns_1(self):
        base = torch.randn(1, 5, CDIM)
        context = torch.cat([base, base, base[:, :2]], 1)   # 12 행 — 5 의 배수가 아님(가드가 없으면 reshape 에서 실패)
        self.assertEqual(self.sd._context_repeat(context, 5), 1)

    def test_native_length_or_shorter_returns_1(self):
        self.assertEqual(self.sd._context_repeat(torch.randn(1, 5, CDIM), 5), 1)
        self.assertEqual(self.sd._context_repeat(torch.randn(1, 4, CDIM), 5), 1)
        self.assertEqual(self.sd._context_repeat(torch.randn(1, 10, CDIM), 0), 1)


class ScriptEndToEndTests(_Harness, unittest.TestCase):
    """옛 엔진 모양 토큰화(실제 Forge 파서 + 실제 SD1.5 토크나이저), 인코더는 토큰 id → 고정 무작위 행."""

    @classmethod
    def setUpClass(cls):
        if not SD15_TOKENIZER.is_dir() or not PROMPT_PARSER.is_file() or not PARSING.is_file():
            raise unittest.SkipTest("SD1.5 토크나이저·Forge prompt_parser 없음")
        try:
            import lark  # noqa: F401
            from transformers import CLIPTokenizer
        except ImportError as exc:   # pragma: no cover - Forge venv 밖
            raise unittest.SkipTest(f"transformers/lark 없음: {exc}")
        super().setUpClass()
        try:
            cls.tok = CLIPTokenizer.from_pretrained(str(SD15_TOKENIZER))
            cls.pp = _load("modules.prompt_parser", PROMPT_PARSER)
            cls.parsing = _load("_test_negpip_sd_batching_parsing", PARSING)
            cls.table = torch.randn(cls.tok.vocab_size + 2, CDIM, generator=torch.Generator().manual_seed(5))
            cls.script = cls._load_script()
        except BaseException:
            super().tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("_test_negpip_sd_batching_parsing", None)
        sys.modules.pop("_test_negpip_sd_batching_script", None)
        super().tearDownClass()

    @classmethod
    def _load_script(cls):
        class _Script:
            def __init__(self):
                pass

        modules = sys.modules["modules"]
        modules.scripts = types.SimpleNamespace(Script=_Script, AlwaysVisible=object())
        callbacks = types.ModuleType("modules.script_callbacks")
        callbacks.CFGDenoiserParams = object
        callbacks.on_cfg_denoiser = lambda callback: None
        modules.script_callbacks = callbacks
        anima = types.ModuleType("sam3ext.negpip.anima")
        anima.patch_anima_negpip = lambda cls_, *, unpatch=False: None
        stubs = {"modules.scripts": modules.scripts, "modules.script_callbacks": callbacks,
                 "sam3ext.negpip.anima": anima, "sam3ext.negpip.sd": cls.sd}
        saved = {name: sys.modules.get(name) for name in stubs}
        sys.modules.update(stubs)
        try:
            module = _load("_test_negpip_sd_batching_script", SCRIPT)
        finally:
            for name, value in saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
        module.IS_NEO = True
        return module

    # ---- 옛 엔진 모양 엔진·인코더 ----

    def _engine(self):
        tok, parsing, table = self.tok, self.parsing, self.table
        bos, eos = tok.bos_token_id, tok.eos_token_id

        class Engine:
            id_start, id_end = bos, eos

            def tokenize_line(self, line):
                parsed = parsing.parse_prompt_attention(line)
                ids = [i for text, _ in parsed if text != "BREAK" for i in tok(text, add_special_tokens=False, verbose=False)["input_ids"]]
                # 75 토큰마다 한 청크(옛 엔진처럼) — 긴 프롬프트는 77 행 청크 여럿
                parts = [ids[i : i + 75] for i in range(0, len(ids), 75)] or [[]]
                return [_Chunk([bos] + part + [eos] * (76 - len(part))) for part in parts], len(ids)

            def encode(self, text):
                chunks, _ = self.tokenize_line(text)
                return table[torch.tensor([t for c in chunks for t in c.tokens])]

        return Engine()

    def _ids(self, rows):
        return [int((self.table - row).abs().sum(1).argmin()) for row in rows]

    def _words(self, *terms):
        """기대 행의 토큰: 항마다 [단어…, EOS] 를 이은 것."""
        out = []
        for term in terms:
            out += self.tok(term, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]
        return out

    def _p(self, prompts, negatives, *, hr_prompts=None, hr_negatives=None, steps=20):
        engine = self._engine()
        diffusion = self._diffusion(count=2)
        model = types.SimpleNamespace(
            is_sdxl=False, text_processing_engine=engine,
            forge_objects=types.SimpleNamespace(unet=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=diffusion))),
            is_webui_legacy_model=lambda: True,
            get_learned_conditioning=lambda texts: torch.stack([engine.encode(t) for t in texts]),
        )
        self.shared.sd_model = model
        return types.SimpleNamespace(
            sd_model=model, scripts=types.SimpleNamespace(scripts=[], alwayson_scripts=[]), script_args=[],
            prompts=list(prompts), negative_prompts=list(negatives), steps=steps, batch_size=len(prompts),
            sampler_name="Euler a", width=512, height=512, extra_generation_params={},
            cached_c=[None] * 3, cached_uc=[None] * 3,
            hr_prompts=list(hr_prompts) if hr_prompts else None,
            hr_negative_prompts=list(hr_negatives) if hr_negatives else None,
        )

    def _process(self, p):
        script = self.script.NegPiP()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            script.process_batch(p)
        self.addCleanup(script.reset)
        return script, out.getvalue()

    def _callback(self, script, p, step, *, hr=False):
        engine = p.sd_model.text_processing_engine
        c = torch.stack([engine.encode(t) for t in (p.hr_prompts if hr else p.prompts)])
        uc = torch.stack([engine.encode(t) for t in (p.hr_negative_prompts if hr and p.hr_negative_prompts else p.negative_prompts)])
        script.denoiser_callback(types.SimpleNamespace(sampling_step=step, text_cond=c, text_uncond=uc))
        return c, uc

    def _item_ids(self, rows):
        return [None if r is None else self._ids(r) for r in rows]

    def test_every_term_of_a_prompt_reaches_attention(self):
        p = self._p(["1girl, (cat:-1), (dog:-1)"], ["lowres, (blurry:-1), (jpeg:-0.5)"])
        script, log = self._process(p)
        self.assertTrue(script.active)
        self.assertEqual(p.prompts, ["1girl, , "])
        self.assertEqual(p.negative_prompts, ["lowres, , "])
        c, uc = self._callback(script, p, 0)
        self.assertEqual(self._item_ids(script.conds), [self._words("cat", "dog")])
        self.assertEqual(self._item_ids(script.unconds), [self._words("blurry", "jpeg")])
        self.assertIn(f"NegPiP Enable (Positive: {len(self._words('cat', 'dog'))})", log)
        self.assertIn(f"NegPiP Enable (Negative: {len(self._words('blurry', 'jpeg'))})", log)
        # 실제 배치 → 훅: C 행에 cat·dog 4 행, U 행에 blurry·jpeg 4 행, V 는 그 행 모두 뒤집힘
        model = self._Model(p.sd_model.forge_objects.unet.model.diffusion_model)
        self.sf.calc_cond_uncond_batch(model, self.sf.compile_conditions(c), self.sf.compile_conditions(uc),
                                       torch.zeros(1, 4, 8, 8), torch.tensor([1.0]), {})
        expected = {COND: script.conds, UNCOND: script.unconds}
        self.assertEqual(self._check_calls(model.calls, expected, {COND: 77, UNCOND: 77}), {(COND, 0), (UNCOND, 0)})

    def test_each_batch_item_gets_its_own_terms(self):
        p = self._p(["1girl, (cat:-1)", "1girl, (dog:-1)", "1girl"], ["", "(blurry:-1)", ""])
        script, log = self._process(p)
        self.assertEqual(p.prompts, ["1girl, ", "1girl, ", "1girl"])
        c, uc = self._callback(script, p, 0)
        self.assertEqual(self._item_ids(script.conds), [self._words("cat"), self._words("dog"), None])
        self.assertEqual(self._item_ids(script.unconds), [None, self._words("blurry"), None])
        model = self._Model(p.sd_model.forge_objects.unet.model.diffusion_model)
        self.sf.calc_cond_uncond_batch(model, self.sf.compile_conditions(c), self.sf.compile_conditions(uc),
                                       torch.zeros(3, 4, 8, 8), torch.tensor([1.0]), {})
        seen = self._check_calls(model.calls, {COND: script.conds, UNCOND: script.unconds}, {COND: 77, UNCOND: 77})
        self.assertEqual(len(seen), 6)

    def test_first_item_without_terms_does_not_hide_the_others(self):
        # 상류: 항목 0 에 항이 없으면 뒤 항목의 항이 프롬프트에서만 지워지고 어디에도 붙지 않았다(로그도 없음)
        p = self._p(["1girl", "1girl, (dog:-1)"], ["", ""])
        script, log = self._process(p)
        self._callback(script, p, 0)
        self.assertEqual(self._item_ids(script.conds), [None, self._words("dog")])
        self.assertIn("NegPiP Enable (Positive: 2)", log)

    def test_identical_items_share_one_tensor(self):
        p = self._p(["1girl, (cat:-1)", "1girl, (cat:-1)"], ["", ""])
        script, _ = self._process(p)
        self._callback(script, p, 0)
        self.assertIs(script.conds[0], script.conds[1])

    def test_scheduled_term_waits_for_its_step_on_both_sides(self):
        # 상류의 부정 쪽 break 는 안쪽 반복에 있어 [(x:-1):5] 가 0 스텝부터 걸렸다. 두 쪽이 같은 문턱을 쓴다
        p = self._p(["1girl, [(cat:-1):5]"], ["lowres, [(blurry:-1):5]"])
        script, _ = self._process(p)
        active = {}
        for step in range(19):
            self._callback(script, p, step)
            active[step] = (script.conds[0] is not None, script.unconds[0] is not None)
        first = min(s for s, (pos, neg) in active.items() if pos)
        self.assertEqual(first, min(s for s, (pos, neg) in active.items() if neg))
        self.assertTrue(all(pos == neg for pos, neg in active.values()))
        self.assertEqual(first, 4, "상류 긍정 쪽과 같은 문턱: 끝 스텝 20 줄은 step >= sampling_step + 2")
        self.assertEqual(self._item_ids(script.unconds), [self._words("blurry")])

    def test_hires_prompts_per_item(self):
        p = self._p(["1girl, (cat:-1)", "1girl, (dog:-1)"], ["", ""],
                    hr_prompts=["1girl, (red:-1)", "1girl, (blue:-1)"], hr_negatives=["(x:-1)", ""])
        script, _ = self._process(p)
        script.before_hr()
        self._callback(script, p, 0, hr=True)
        self.assertEqual(self._item_ids(script.conds), [self._words("red"), self._words("blue")])
        self.assertEqual(self._item_ids(script.unconds), [self._words("x"), None])

    def _long(self, words=45):
        """두 청크(77 행 × 2)로 인코딩되는 긴 프롬프트."""
        text = ", ".join(["masterpiece"] * words)
        self.assertEqual(len(self._engine().tokenize_line(text)[0]), 2)
        return text

    def test_lcm_repeated_context_keeps_the_native_strength_end_to_end(self):
        # 긍정·부정 청크 수가 다르면 Forge 가 짧은 쪽 문맥을 lcm 길이로 반복해 [U, C] 로 묶는다(ConditionCrossAttn.concat).
        # denoiser_callback 이 text_cond/text_uncond 에서 읽은 원래 길이(context_rows)로 훅이 붙일 행도 같은 배수로 반복해야
        # 음수 항이 원래 세기 — 원래 길이를 못 읽거나 반대쪽 것을 읽으면 반복 문맥 뒤에 한 번만 붙어 softmax 몫이 1/2 로 준다.
        cases = {
            "cond_short": (["1girl, (cat:-1)"], [self._long() + ", (blurry:-1)"], {COND: 77, UNCOND: 154}),
            "uncond_short": ([self._long() + ", (cat:-1)"], ["lowres, (blurry:-1)"], {COND: 154, UNCOND: 77}),
        }
        for name, (prompts, negatives, native_len) in cases.items():
            with self.subTest(name):
                p = self._p(prompts, negatives)
                script, _ = self._process(p)
                c, uc = self._callback(script, p, 0)
                self.assertEqual((c.shape[1], uc.shape[1]), (native_len[COND], native_len[UNCOND]))
                self.assertEqual((script.c_native, script.uc_native), (native_len[COND], native_len[UNCOND]))
                self.assertEqual(self._item_ids(script.conds), [self._words("cat")])
                self.assertEqual(self._item_ids(script.unconds), [self._words("blurry")])
                model = self._Model(p.sd_model.forge_objects.unet.model.diffusion_model)
                self.sf.calc_cond_uncond_batch(model, self.sf.compile_conditions(c), self.sf.compile_conditions(uc),
                                               torch.zeros(1, 4, 8, 8), torch.tensor([1.0]), {})
                self.assertEqual(model.calls[0][0], [UNCOND, COND], "lcm 비 2 — 한 배치로 묶인다")
                self.assertEqual(model.calls[0][3].shape[1], 154, "짧은 쪽 문맥이 2 번 반복됐다")
                expected = {COND: script.conds, UNCOND: script.unconds}
                self.assertEqual(self._check_calls(model.calls, expected, native_len), {(COND, 0), (UNCOND, 0)})
                # 비교가 뜻이 있다 — 반복된 쪽에 한 번만 붙이면(상류) 기대값과 다르다
                labels, attn, h, ctx, out = model.calls[0]
                row = labels.index(COND if native_len[COND] == 77 else UNCOND)
                side = labels[row]
                once = self._reference(attn, h[row], ctx[row], expected[side][0])
                self.assertGreater(float((once - out[row]).abs().max()), 1e-3)

    def test_sdxl_dict_condition_gives_the_native_length(self):
        # SDXL: CFGDenoiserParams.text_cond/text_uncond 는 prompt_parser.DictWithShape({"crossattn", "vector"}) — 문맥 길이는 crossattn
        p = self._p(["1girl, (cat:-1)"], [self._long() + ", (blurry:-1)"])
        script, _ = self._process(p)
        engine = p.sd_model.text_processing_engine
        c = torch.stack([engine.encode(t) for t in p.prompts])
        uc = torch.stack([engine.encode(t) for t in p.negative_prompts])
        vector = torch.zeros(1, 2816)
        for wrap in (self.pp.DictWithShape, dict):
            with self.subTest(wrap=wrap.__name__):
                text_cond = wrap({"crossattn": c, "vector": vector})
                text_uncond = wrap({"crossattn": uc, "vector": vector})
                script.denoiser_callback(types.SimpleNamespace(sampling_step=0, text_cond=text_cond, text_uncond=text_uncond))
                self.assertEqual((script.c_native, script.uc_native), (77, 154))
                model = self._Model(p.sd_model.forge_objects.unet.model.diffusion_model)
                self.sf.calc_cond_uncond_batch(model, self.sf.compile_conditions(c), self.sf.compile_conditions(uc),
                                               torch.zeros(1, 4, 8, 8), torch.tensor([1.0]), {})
                self._check_calls(model.calls, {COND: script.conds, UNCOND: script.unconds}, {COND: 77, UNCOND: 154})

    def test_row_selection_is_unchanged(self):
        # 행 고르기(F1)는 그대로 — 한 청크 항은 상류 자르기 cond[1 : token_len + 2] 와 같은 행
        p = self._p(["1girl, (red eyes:-1.5)"], [""])
        script, _ = self._process(p)
        rows, count = script.conds_all[0][0][1]
        engine = p.sd_model.text_processing_engine
        full = engine.encode("(red eyes:1.5)")
        _, token_len = engine.tokenize_line("red eyes")
        self.assertEqual(count, token_len + 1)
        self.assertTrue(torch.equal(rows, full[1 : token_len + 2]))


if __name__ == "__main__":
    unittest.main()
