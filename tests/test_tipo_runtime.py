"""TIPO 런타임 — 파일 확인·받기, 장치 선택, 생성(가짜 모델·토크나이저, GPU 없이)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext.tipo import runtime as tr  # noqa: E402

GIB = 1024 ** 3


class FilesTests(unittest.TestCase):
    def test_missing_and_download_only_what_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "TIPO" / "TIPO-v2.1-1B-A200M"
            self.assertEqual(tr.missing_files(target), list(tr.FILES))
            target.mkdir(parents=True)
            (target / "config.json").write_text("{}", encoding="utf-8")
            calls = []

            def downloader(**kwargs):
                calls.append(kwargs)
                (Path(kwargs["local_dir"]) / kwargs["filename"]).write_text("x", encoding="utf-8")

            tr.download(target, downloader=downloader)
            self.assertEqual([c["filename"] for c in calls], ["tokenizer.json", "tokenizer_config.json", "model.safetensors"])
            self.assertTrue(all(c["repo_id"] == tr.REPO_ID and c["revision"] == tr.REVISION for c in calls),
                            "고정한 revision 에서만 받는다")
            self.assertEqual(tr.missing_files(target), [])

    def test_a_second_download_while_one_is_running_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = tr.TipoRuntime(directory=tmp)
            nested = []

            def downloader(**kwargs):
                nested.append(runtime.download(downloader=lambda **kw: None))
                (Path(kwargs["local_dir"]) / kwargs["filename"]).write_text("x", encoding="utf-8")

            self.assertTrue(runtime.download(downloader=downloader))
            self.assertEqual(nested, [False] * len(tr.FILES), "받는 중에 또 누르면(다른 탭·새로 고침) 바로 거절")
            self.assertEqual(runtime.missing_files(), [])
            self.assertTrue(runtime.download(downloader=lambda **kw: self.fail("이미 다 있으면 받지 않는다")))

    def test_model_dir_lives_under_forge_models(self):
        self.assertEqual(tr.model_dir(Path("C:/forge/models")), Path("C:/forge/models/TIPO/TIPO-v2.1-1B-A200M"))


class DeviceTests(unittest.TestCase):
    def test_choice(self):
        self.assertEqual(tr.choose_device("GPU", True, 8 * GIB), ("cuda", None))
        device, note = tr.choose_device("GPU", True, 2 * GIB)
        self.assertEqual(device, "cpu")
        self.assertIn("2.0 GB", note)
        self.assertEqual(tr.choose_device("GPU", False, 0)[0], "cpu")
        self.assertEqual(tr.choose_device("CPU", True, 30 * GIB), ("cpu", None))


class _Encoded:
    def __init__(self, ids):
        self.input_ids = ids


class _FakeTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 64000, 64001, 64002

    def __init__(self, with_bos=False):
        self.with_bos = with_bos
        self.decoded = None

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [64000, 5, 6] if self.with_bos else [5, 6]
        return _Encoded(torch.tensor([ids]))

    def decode(self, ids, skip_special_tokens=False):
        self.decoded = (ids.tolist(), skip_special_tokens)
        return ", outdoors\nlong: A girl."


class _FakeModel:
    def __init__(self, fail=False, tail=(7, 8)):
        self.moves = []
        self.fail = fail
        self.tail = list(tail)
        self.generate_kwargs = None
        self.input_ids = None
        self.seed_seen = None

    def generate(self, input_ids, **kwargs):
        if self.fail:
            raise RuntimeError("boom")
        self.input_ids = input_ids.tolist()
        self.generate_kwargs = kwargs
        self.seed_seen = torch.initial_seed()
        return torch.cat([input_ids, torch.tensor([self.tail])], dim=1)


class _Runtime(tr.TipoRuntime):
    """CUDA 를 실제로 건드리는 곳만 막는다(장치 이동은 기록만)."""

    emptied = 0
    fail_on = None

    def _place(self, model, device, dtype):
        model.moves.append((device, dtype))
        if device == self.fail_on:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    def _cpu_params(self, model):
        return "cpu-params"

    def _restore_cpu(self, model, cpu_params):
        model.moves.append(("restore", cpu_params))

    def _move_inputs(self, ids, device):
        return ids

    def _rng_devices(self, device):
        return []

    def _empty_cache(self):
        self.emptied += 1


class GenerateTests(unittest.TestCase):
    def _runtime(self, model=None, tokenizer=None, free=8 * GIB):
        self.loads = 0
        model = model or _FakeModel()
        tokenizer = tokenizer or _FakeTokenizer()

        def loader(directory):
            self.loads += 1
            return model, tokenizer

        return _Runtime(loader=loader, directory=Path("C:/nope"), cuda_info=lambda: (True, free)), model, tokenizer

    def test_gpu_run_moves_the_model_back_to_cpu(self):
        runtime, model, tokenizer = self._runtime()
        result = runtime.generate("tag: 1girl", requested_device="GPU", max_new_tokens=256, seed=42)
        self.assertEqual(model.moves, [("cpu", torch.float16), ("cuda", torch.float16), ("restore", "cpu-params")],
                         "Forge 메모리 관리 밖이라 GPU 에 상주시키지 않는다 — 쥐어 둔 CPU 텐서로 되돌린다(GPU→CPU 복사 없음)")
        self.assertEqual(runtime.emptied, 1)
        self.assertEqual((result.device, result.note, result.seed), ("cuda", None, 42))
        self.assertEqual(result.text, ", outdoors\nlong: A girl.")
        self.assertEqual(tokenizer.decoded, ([7, 8], True), "새 토큰만, 특수 토큰은 빼고")
        self.assertFalse(result.finished, "EOS 없이 끝났으면 토큰 한도에서 잘린 것")

    def test_finished_means_the_model_emitted_eos(self):
        runtime, _, _ = self._runtime(model=_FakeModel(tail=(7, tr.EOS_ID)))
        self.assertTrue(runtime.generate("tag:", requested_device="CPU", max_new_tokens=8, seed=1).finished)

    def test_low_vram_falls_back_to_cpu(self):
        runtime, model, _ = self._runtime(free=1 * GIB)
        result = runtime.generate("tag:", requested_device="GPU", max_new_tokens=128, seed=1)
        self.assertEqual(result.device, "cpu")
        self.assertIn("GB", result.note)
        self.assertEqual(model.moves, [("cpu", torch.float32)],
                         "CPU 는 fp32 로 돌리고 그대로 둔다 — 클릭마다 fp16 으로 되돌리지 않는다")
        self.assertEqual(runtime.emptied, 0, "CPU 회차는 CUDA 캐시를 건드리지 않는다")

    def test_sampling_follows_the_model_card(self):
        runtime, model, _ = self._runtime()
        runtime.generate("tag:", requested_device="CPU", max_new_tokens=384, seed=3)
        kwargs = model.generate_kwargs
        self.assertEqual((kwargs["do_sample"], kwargs["temperature"], kwargs["min_p"]), (True, 1.0, 0.1))
        self.assertEqual((kwargs["top_k"], kwargs["top_p"]), (0, 1.0), "transformers 기본 top_k=50 을 끈다")
        self.assertEqual((kwargs["max_new_tokens"], kwargs["eos_token_id"], kwargs["pad_token_id"]), (384, 64001, 64002))

    def test_seed_is_used_and_the_global_rng_is_restored(self):
        runtime, model, _ = self._runtime()
        before = torch.initial_seed()
        result = runtime.generate("tag:", requested_device="CPU", max_new_tokens=8, seed=1234)
        self.assertEqual(model.seed_seen, 1234)
        self.assertEqual(torch.initial_seed(), before, "Forge 의 전역 난수 상태를 건드리지 않는다")
        self.assertEqual(result.seed, 1234)
        random_result = runtime.generate("tag:", requested_device="CPU", max_new_tokens=8, seed=-1)
        self.assertGreaterEqual(random_result.seed, 0)

    def test_bos_is_added_once(self):
        runtime, model, _ = self._runtime()
        runtime.generate("tag:", requested_device="CPU", max_new_tokens=8, seed=1)
        self.assertEqual(model.input_ids, [[64000, 5, 6]])
        runtime, model, _ = self._runtime(tokenizer=_FakeTokenizer(with_bos=True))
        runtime.generate("tag:", requested_device="CPU", max_new_tokens=8, seed=1)
        self.assertEqual(model.input_ids, [[64000, 5, 6]])

    def test_model_loads_once(self):
        runtime, _, _ = self._runtime()
        runtime.generate("tag:", requested_device="CPU", max_new_tokens=8, seed=1)
        runtime.generate("tag:", requested_device="CPU", max_new_tokens=8, seed=2)
        self.assertEqual(self.loads, 1)

    def test_failure_still_moves_the_model_off_the_gpu(self):
        runtime, model, _ = self._runtime(model=_FakeModel(fail=True))
        with self.assertRaises(RuntimeError):
            runtime.generate("tag:", requested_device="GPU", max_new_tokens=8, seed=1)
        self.assertEqual(model.moves[-1], ("restore", "cpu-params"))

    def test_a_failed_move_to_the_gpu_is_undone(self):
        runtime, model, _ = self._runtime()
        runtime.fail_on = "cuda"
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            runtime.generate("tag:", requested_device="GPU", max_new_tokens=8, seed=1)
        self.assertEqual(model.moves, [("cpu", torch.float16), ("cuda", torch.float16), ("restore", "cpu-params")],
                         "일부만 GPU 로 옮겨진 채 남지 않게")
        self.assertEqual(runtime.emptied, 1)


class PlaceTests(unittest.TestCase):
    """실제 _place — 파라미터만 dtype 을 바꾸고 fp32 버퍼(RoPE inv_freq)는 그대로 둔다. CPU 만 쓴다."""

    def test_parameters_change_dtype_buffers_keep_fp32(self):
        module = torch.nn.Linear(4, 4).to(torch.float16)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 8, 2).float() / 8))
        module.register_buffer("inv_freq", inv_freq.clone(), persistent=False)
        tr.TipoRuntime._place(module, "cpu", torch.float32)
        self.assertEqual(module.weight.dtype, torch.float32)
        self.assertIsInstance(module.weight, torch.nn.Parameter)
        tr.TipoRuntime._place(module, "cpu", torch.float16)
        self.assertEqual(module.weight.dtype, torch.float16)
        self.assertEqual(module.inv_freq.dtype, torch.float32)
        self.assertTrue(torch.equal(module.inv_freq, inv_freq), "fp16 로 반올림되지 않는다")


_TINY_WORDS = ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>", "tag:", "target:", "1girl", "smile", "outdoors", ","]


def _save_tiny_model(directory, words=_TINY_WORDS, hidden=32, seed=0):
    """실제 파일 형식(config·tokenizer·safetensors, fp16)으로 아주 작은 무작위 Kohaku 모델을 저장한다."""
    import json

    from tokenizers import Tokenizer, models, pre_tokenizers

    from sam3ext.tipo.kohaku import KohakuConfig, KohakuForCausalLM

    tokenizer = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(words)}, unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(Path(directory) / "tokenizer.json"))
    (Path(directory) / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<|bos|>", "eos_token": "<|eos|>",
        "pad_token": "<|pad|>", "unk_token": "<|unk|>",
    }), encoding="utf-8")
    config = KohakuConfig(
        vocab_size=len(words), hidden_size=hidden, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=hidden // 4, intermediate_size=2 * hidden, moe_intermediate_size=hidden // 2, n_routed_experts=4,
        num_experts_per_tok=2, first_k_dense=1, bos_token_id=0, eos_token_id=1, pad_token_id=2,
    )
    torch.manual_seed(seed)
    KohakuForCausalLM(config).to(torch.float16).save_pretrained(directory)


class RealLoaderTests(unittest.TestCase):
    """실제 로드 경로(_load_pretrained) 끝까지 — 아주 작은 무작위 모델을 같은 파일 형식으로 저장해 CPU 로 생성."""

    def test_saved_tiny_model_loads_and_generates_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            _save_tiny_model(tmp)
            self.assertEqual(tr.missing_files(tmp), [])
            runtime = tr.TipoRuntime(directory=tmp, cuda_info=lambda: (False, 0))
            result = runtime.generate("target: tag: 1girl , smile", requested_device="GPU", max_new_tokens=4, seed=7)
        self.assertEqual(result.device, "cpu", "CUDA 가 없으면 CPU")
        self.assertIsInstance(result.text, str)
        self.assertEqual(runtime._model.dtype, torch.float32, "CPU 로 돌린 뒤에는 fp32 로 둔다 — 다음 CPU 클릭에 캐스트 없음")
        self.assertEqual(runtime._model.model.rotary_emb.inv_freq.dtype, torch.float32)


class _OldRuntime(tr.TipoRuntime):
    """supp-05 이전 구현(참조) — 클릭마다 GPU 로 올렸다 GPU→CPU 로 복사해 내리고, CPU 는 fp16→fp32→fp16 왕복 캐스트."""

    @staticmethod
    def _place(model, device, dtype):
        model.to(device=device)
        for param in model.parameters():
            if param.dtype != dtype:
                param.data = param.data.to(dtype)

    def generate(self, prompt_text, *, requested_device, max_new_tokens, seed):
        import random
        import time

        model, tokenizer = self._ensure_loaded()
        cuda_available, free_bytes = self._cuda_info()
        device, note = tr.choose_device(requested_device, cuda_available, free_bytes)
        seed = int(seed)
        if seed < 0:
            seed = random.randrange(2 ** 31)
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tr.EOS_ID
        start = time.perf_counter()
        try:
            self._place(model, device, torch.float16 if device == "cuda" else torch.float32)
            ids = self._move_inputs(self._encode(tokenizer, prompt_text), device)
            with torch.random.fork_rng(devices=self._rng_devices(device)):
                torch.manual_seed(seed)
                with torch.inference_mode():
                    output = model.generate(
                        ids, attention_mask=torch.ones_like(ids), max_new_tokens=int(max_new_tokens), do_sample=True,
                        temperature=1.0, min_p=0.1, top_k=0, top_p=1.0, eos_token_id=eos_id,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tr.PAD_ID,
                    )
            new_ids = output[0, ids.shape[1]:].cpu()
            finished = bool((new_ids == eos_id).any())
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
        finally:
            self._place(model, "cpu", torch.float16)
            if device == "cuda":
                self._empty_cache()
        return tr.GenerationResult(text, seed, device, time.perf_counter() - start, note, finished)


def _simulated_gpu(base):
    """'GPU' 회차를 CPU 에서 fp16 으로 흉내 낸다 — 옮기고 되돌리는 경로(쥐어 둔 CPU 텐서, dtype)를 GPU 없이 그대로 탄다."""

    class Sim(base):
        empties = 0

        def _place(self, model, device, dtype):
            base._place(model, "cpu" if device == "cuda" else device, dtype)

        def _move_inputs(self, ids, device):
            return ids

        def _rng_devices(self, device):
            return []

        def _empty_cache(self):
            self.empties += 1

    return Sim


_WIDE_WORDS = ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>", "tag:", "target:", ","] + [f"w{i}" for i in range(57)]


class BitIdenticalTests(unittest.TestCase):
    """supp-05 — 옛 구현(_OldRuntime)과 같은 시드에서 생성 토큰이 비트 단위로 같고, 쉬는 가중치 값도 같다. CPU 만 쓴다."""

    @classmethod
    def setUpClass(cls):
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(4)
        cls._tmp = tempfile.TemporaryDirectory()
        _save_tiny_model(cls._tmp.name, words=_WIDE_WORDS, hidden=64, seed=3)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        torch.set_num_threads(cls._threads)

    def _pair(self, gpu):
        info = (True, 8 * GIB) if gpu else (False, 0)
        classes = (_simulated_gpu(_OldRuntime), _simulated_gpu(tr.TipoRuntime)) if gpu else (_OldRuntime, tr.TipoRuntime)
        runtimes = [cls(directory=self._tmp.name, cuda_info=lambda: info) for cls in classes]
        decoded = []
        self._logits = []
        for runtime in runtimes:
            model, tokenizer = runtime._ensure_loaded()
            log = []
            original = tokenizer.decode

            def decode(ids, *args, _log=log, _original=original, **kwargs):
                _log.append(ids.tolist())
                return _original(ids, *args, **kwargs)

            tokenizer.decode = decode
            decoded.append(log)
            # 토큰만 비교하면 작은 무작위 모델의 샘플링이 작은 수치 차이를 가린다 — 스텝마다 logits 를 비트 단위로 비교한다.
            logits = []
            model.lm_head.register_forward_hook(lambda _m, _i, out, _l=logits: _l.append(out.detach().clone()))
            self._logits.append(logits)
        return runtimes, decoded

    def _run(self, runtimes, decoded, schedule):
        prompts = ["target: tag: w1 , w2", "tag: w5 w9 , w13 target:"]
        for step, (device, seed) in enumerate(schedule):
            for logits in self._logits:
                logits.clear()
            results = [r.generate(prompts[step % 2], requested_device=device, max_new_tokens=24, seed=seed) for r in runtimes]
            self.assertEqual(decoded[0][-1], decoded[1][-1], f"{step}번째({device}, 시드 {seed}) 생성 토큰이 옛 구현과 같다")
            self.assertGreater(len(decoded[0][-1]), 0)
            old_logits, new_logits = self._logits
            self.assertEqual(len(old_logits), len(new_logits))
            self.assertGreater(len(old_logits), 0)
            for i, (a, b) in enumerate(zip(old_logits, new_logits)):
                self.assertEqual(a.dtype, b.dtype, f"{step}번째 {i} 스텝 logits dtype")
                self.assertTrue(torch.equal(a, b), f"{step}번째({device}, 시드 {seed}) {i} 스텝 logits 가 비트 단위로 같다")
            self.assertEqual(results[0].text, results[1].text)
            self.assertEqual((results[0].device, results[0].finished), (results[1].device, results[1].finished))

    def _assert_same_weights(self, old, new):
        new_params = dict(new._model.named_parameters())
        for name, param in old._model.named_parameters():
            self.assertTrue(torch.equal(param.detach().float(), new_params[name].detach().float()), name)
        new_buffers = dict(new._model.named_buffers())
        for name, buf in old._model.named_buffers():
            self.assertTrue(torch.equal(buf, new_buffers[name]) and buf.dtype == new_buffers[name].dtype, name)

    def test_cpu_clicks_match_the_old_round_trip(self):
        runtimes, decoded = self._pair(gpu=False)
        self._run(runtimes, decoded, [("CPU", 7), ("CPU", 7), ("CPU", 11), ("CPU", 12345)])
        self._assert_same_weights(*runtimes)
        self.assertEqual(runtimes[0]._model.dtype, torch.float16, "옛 구현은 클릭마다 fp16 으로 되돌렸다")
        self.assertEqual(runtimes[1]._model.dtype, torch.float32)

    def test_repeated_cpu_clicks_do_not_cast_again(self):
        runtime = tr.TipoRuntime(directory=self._tmp.name, cuda_info=lambda: (False, 0))
        runtime.generate("tag: w1", requested_device="CPU", max_new_tokens=4, seed=1)
        pointers = [p.data_ptr() for p in runtime._model.parameters()]
        runtime.generate("tag: w1", requested_device="CPU", max_new_tokens=4, seed=2)
        self.assertEqual([p.data_ptr() for p in runtime._model.parameters()], pointers,
                         "두 번째 CPU 클릭은 새 텐서를 만들지 않는다")

    def test_simulated_gpu_clicks_match_and_reuse_the_cpu_tensors(self):
        runtimes, decoded = self._pair(gpu=True)
        new = runtimes[1]
        loaded = {name: p.detach().clone() for name, p in new._model.named_parameters()}
        pointers = [p.data_ptr() for p in new._model.parameters()]
        self._run(runtimes, decoded, [("GPU", 7), ("GPU", 7), ("GPU", 99)])
        self.assertEqual([p.data_ptr() for p in new._model.parameters()], pointers,
                         "내릴 때 새 CPU 텐서를 만들지 않고 쥐어 둔 것을 되돌린다(GPU→CPU 복사 없음)")
        for name, param in new._model.named_parameters():
            self.assertTrue(torch.equal(param, loaded[name]) and param.dtype == torch.float16, name)
        self.assertEqual(new.empties, 3)
        self._assert_same_weights(*runtimes)

    def test_mixed_gpu_and_cpu_clicks_match(self):
        runtimes, decoded = self._pair(gpu=True)
        new = runtimes[1]
        self._run(runtimes, decoded, [("GPU", 5), ("CPU", 5), ("CPU", 6)])
        self.assertEqual(new._model.dtype, torch.float32, "CPU 로 돈 뒤에는 fp32 로 쉰다")
        self._run(runtimes, decoded, [("GPU", 6)])
        self.assertEqual(new._model.dtype, torch.float16, "GPU 로 바꾸면 RAM 도 fp16 으로 한 번만 바꾼다")
        pointers = [p.data_ptr() for p in new._model.parameters()]
        self._run(runtimes, decoded, [("GPU", 5)])
        self.assertEqual([p.data_ptr() for p in new._model.parameters()], pointers, "GPU 를 이어 누르면 캐스트 없음")
        self._run(runtimes, decoded, [("CPU", 5)])
        self._assert_same_weights(*runtimes)

    def test_failed_simulated_gpu_move_restores_every_cpu_tensor(self):
        (_, new), _ = self._pair(gpu=True)
        new.generate("tag: w1", requested_device="CPU", max_new_tokens=2, seed=1)   # fp32 로 쉬는 상태에서
        before = [(p.data_ptr(), p.dtype) for p in new._model.parameters()]
        calls = []

        real_place = new._place

        def failing_place(model, device, dtype):
            calls.append(device)
            if device != "cuda":
                return real_place(model, device, dtype)
            params = list(model.parameters())
            params[0].data = params[0].data.to(torch.float32)   # 일부만 바뀐 채 실패
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        new._place = failing_place
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            new.generate("tag: w1", requested_device="GPU", max_new_tokens=2, seed=1)
        self.assertEqual(calls, ["cpu", "cuda"], "RAM 에서 fp16 으로 한 번 바꾼 뒤 올리다 실패")
        after = [(p.data_ptr(), p.dtype) for p in new._model.parameters()]
        self.assertTrue(all(dtype == torch.float16 for _, dtype in after), "실패해도 전부 쥐어 둔 CPU fp16 텐서로 되돌린다")
        self.assertNotEqual(after, before)
        self.assertEqual(new.empties, 1)
        (old, _), _ = self._pair(gpu=False)
        self._assert_same_weights(old, new)

    def test_forge_managed_gpu_clicks_match_with_evictions_in_between(self):
        """GPU 에 남겨 Forge 에 맡겨도(중간에 Forge 가 퇴출해도) 같은 시드의 토큰·logits 가 옛 구현과 비트 단위로 같다."""
        runtimes, decoded = self._pair(gpu=True)
        new = runtimes[1]
        memory = _FakeMemory()
        new._forge_memory = lambda: (memory, _FakeModelPatcher)
        plain = new.generate
        new.generate = lambda *args, **kwargs: plain(*args, keep_on_gpu=True, **kwargs)
        loaded = {name: p.detach().clone() for name, p in new._model.named_parameters()}
        self._run(runtimes, decoded, [("GPU", 7), ("GPU", 7)])
        self.assertTrue(new.resident)
        self.assertEqual(len(memory.entries(new)), 1)
        memory.free_memory(1e30, "cuda")   # Forge 생성 시작 — 자리가 필요하다
        self.assertFalse(new.resident)
        self._run(runtimes, decoded, [("GPU", 99), ("CPU", 99)])
        self.assertEqual(new._model.dtype, torch.float32)
        # Forge 는 샘플링 문맥(inference_mode) 안에서 올리고 내릴 수 있다 — 그래도 가중치는 일반 텐서로 둔다
        with torch.inference_mode():
            memory.load_models_gpu([new._patcher], force_full_load=True)
        self.assertTrue(new.resident)
        self.assertFalse(any(p.data.is_inference() for p in new._model.parameters()), "inference_mode 밖에서 옮긴다")
        self._run(runtimes, decoded, [("GPU", 5), ("GPU", 6)])
        with torch.inference_mode():
            memory.free_memory(1e30, "cuda")
        self.assertFalse(new.resident)
        for name, param in new._model.named_parameters():
            self.assertTrue(torch.equal(param, loaded[name]) and param.dtype == torch.float16, name)
            self.assertFalse(param.data.is_inference(), name)
        self.assertFalse(any(buf.is_inference() for buf in new._model.buffers()))
        self._assert_same_weights(*runtimes)


class PlaceEquivalenceTests(unittest.TestCase):
    """새 _place(파라미터별 장치+dtype) 가 옛 _place(통째로 옮긴 뒤 캐스트)와 같은 값을 만든다."""

    def _module(self):
        torch.manual_seed(0)
        module = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8)).to(torch.float16)
        module.register_buffer("inv_freq", torch.rand(4), persistent=False)
        module.register_buffer("bias16", torch.rand(4).half())
        return module

    def test_same_values_both_directions(self):
        for start in (torch.float16, torch.float32):
            for dtype in (torch.float32, torch.float16):
                old, new = self._module().to(start), self._module().to(start)
                _OldRuntime._place(old, "cpu", dtype)
                tr.TipoRuntime._place(new, "cpu", dtype)
                a_state, b_state = old.state_dict(), new.state_dict()
                self.assertEqual(list(a_state), list(b_state))
                for name in a_state:
                    a, b = a_state[name], b_state[name]
                    self.assertTrue(torch.equal(a, b) and a.dtype == b.dtype, f"{start}->{dtype} {name}")
                self.assertTrue(all(isinstance(p, torch.nn.Parameter) for p in new.parameters()))

    def test_no_copy_when_already_there(self):
        module = self._module()
        pointers = [p.data_ptr() for p in module.parameters()]
        tr.TipoRuntime._place(module, "cpu", torch.float16)
        self.assertEqual([p.data_ptr() for p in module.parameters()], pointers)

    def test_fp16_fp32_round_trip_is_exact(self):
        values = torch.randn(1 << 16).half()
        special = torch.tensor([0.0, -0.0, 6.1e-5, 5.96e-8, 65504.0, float("inf"), -float("inf")]).half()
        every_bit = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.float16)
        finite = every_bit[~torch.isnan(every_bit)]
        for x in (values, special, finite):
            self.assertTrue(torch.equal(x.float().half().view(torch.int16), x.view(torch.int16)),
                            "fp16→fp32→fp16 은 부호 있는 0·비정규수까지 같은 비트")


class _FakeCudaTensor:
    """``.to(device=, dtype=)`` 만 흉내 내는 가짜 텐서 — 장치 사이 복사를 ``log`` 에 (출발, 도착, dtype) 으로 적는다."""

    def __init__(self, device, dtype, log, numel=1024):
        self.device, self.dtype, self.log, self._numel = device, dtype, log, numel

    def to(self, device=None, dtype=None):
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        if (device, dtype) == (self.device, self.dtype):
            return self   # torch 와 같다 — 이미 거기 있으면 같은 텐서
        if device != self.device:
            self.log.append((self.device, device, dtype))
        return _FakeCudaTensor(device, dtype, self.log, self._numel)

    def numel(self):
        return self._numel

    def element_size(self):
        return 2 if self.dtype == torch.float16 else 4


class _FakeParam:
    def __init__(self, data):
        self.data = data


class _FakeCudaModel:
    """파라미터는 가짜 텐서, generate 는 진짜 텐서를 돌려준다. generate 때 모든 파라미터가 어디 있었는지 적는다."""

    def __init__(self, count=3, dtype=torch.float16, fail=False):
        self.log = []
        self.params = [_FakeParam(_FakeCudaTensor("cpu", dtype, self.log)) for _ in range(count)]
        self.buffer_moves = []
        self.seen = []
        self.fail = fail

    def parameters(self):
        return iter(self.params)

    def to(self, device=None):
        self.buffer_moves.append(device)
        return self

    def generate(self, input_ids, **kwargs):
        self.seen.append({(p.data.device, p.data.dtype) for p in self.params})
        if self.fail:
            raise RuntimeError("boom")
        return torch.cat([input_ids, torch.tensor([[7, tr.EOS_ID]])], dim=1)

    def h2d(self):
        return [entry for entry in self.log if entry[:2] == ("cpu", "cuda")]

    def d2h(self):
        return [entry for entry in self.log if entry[:2] == ("cuda", "cpu")]


class _FakeModelPatcher:
    """backend.patcher.base.ModelPatcher 대역 — 생성자 인자와 LoadedModel 이 부르는 것만. 올리고 내리는 메서드는 TIPO 가 덮어쓴다."""

    def __init__(self, model, load_device, offload_device, size=0, *, current_device=None, weight_inplace_update=False):
        self.model, self.load_device, self.offload_device, self.size = model, load_device, offload_device, size
        self.current_device = current_device or offload_device
        for name, value in (("model_loaded_weight_memory", 0), ("model_lowvram", False)):
            if not hasattr(model, name):
                setattr(model, name, value)

    def model_size(self):
        return self.size

    def loaded_size(self):
        return self.model.model_loaded_weight_memory

    def model_patches_to(self, device):
        pass

    def model_dtype(self):
        return None

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        raise AssertionError("TIPO 패처가 덮어써야 한다")

    partially_unload = detach = partially_load


class _OtherModel:
    """Forge 가 관리하는 다른 모델(UNet 등) — 부르면 통째로 올리고 통째로 내린다."""

    def __init__(self, size):
        self.model = self
        self.load_device, self.offload_device, self.current_device = "cuda", "cpu", "cpu"
        self.size = size
        self.model_loaded_weight_memory = 0

    def model_size(self):
        return self.size

    def loaded_size(self):
        return self.model_loaded_weight_memory

    def model_patches_to(self, device):
        pass

    def model_dtype(self):
        return None

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        gained, self.model_loaded_weight_memory, self.current_device = self.size - self.loaded_size(), self.size, device_to
        return gained

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        return 0

    def detach(self, unpatch_all=True):
        if unpatch_all:
            self.model_loaded_weight_memory, self.current_device = 0, self.offload_device
        return self


class _FakeLoadedModel:
    """backend.memory_management.LoadedModel 대역 — model_load·model_unload 가 패처를 부르는 순서와 인자가 같다."""

    def __init__(self, patcher):
        self.model, self.device, self.model_finalizer = patcher, patcher.load_device, None

    def model_memory(self):
        return self.model.model_size()

    def model_offloaded_memory(self):
        return self.model.model_size() - self.model.loaded_size()

    def model_memory_required(self, device):
        return self.model_offloaded_memory() if device == self.model.current_device else self.model_memory()

    def model_load(self):
        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())
        self.model.partially_load(self.device, 1e32)   # model_use_more_vram(lowvram_model_memory=0 → 1e32)
        self.model_finalizer = mock.Mock()

    def model_unload(self, memory_to_free=None, unpatch_weights=True):
        if memory_to_free is not None and memory_to_free < self.model.loaded_size():
            if self.model.partially_unload(self.model.offload_device, memory_to_free) >= memory_to_free:
                return False
        self.model.detach(unpatch_weights)
        self.model_finalizer.detach()
        self.model_finalizer = None
        return True


class _FakeMemory:
    """backend.memory_management 대역 — load_models_gpu·free_memory 의 순서를 따르고 VRAM 을 바이트로 센다."""

    def __init__(self, total=24 * GIB, inference=GIB):
        self.total, self.inference = total, inference
        self.current_loaded_models = []
        self.loads = []

    def get_torch_device(self):
        return "cuda"

    def get_free_memory(self, device=None):
        return self.total - sum(entry.model.loaded_size() for entry in self.current_loaded_models)

    def free_memory(self, memory_required, device, keep_loaded=()):
        # Forge 와 같이 덜 올라간 것부터, 그다음 작은 것부터 — 여유가 생기면 멈춘다
        order = sorted(self.current_loaded_models, key=lambda e: (-e.model_offloaded_memory(), e.model_memory()))
        unloaded = []
        for entry in order:
            if entry in keep_loaded:
                continue
            free = self.get_free_memory(device)
            if free > memory_required:
                break
            if entry.model_unload(memory_required - free):
                unloaded.append(entry)
        for entry in unloaded:
            self.current_loaded_models.remove(entry)
        return unloaded

    def load_models_gpu(self, models, memory_required=0, force_patch_weights=False, minimum_memory_required=None,
                        force_full_load=False):
        self.loads.append((tuple(models), force_full_load))
        current = self.current_loaded_models
        to_load = [next((e for e in current if e.model is m), None) or _FakeLoadedModel(m) for m in models]
        for entry in to_load:   # Forge 는 is_clone 으로 이미 올라간 자신도 목록에서 뺐다가 다시 넣는다
            for i in reversed(range(len(current))):
                if entry.model.model is current[i].model.model:
                    current.pop(i).model.detach(unpatch_all=False)
        need = sum(e.model_memory_required(e.device) for e in to_load)
        self.free_memory(need * 1.1 + max(self.inference, memory_required), "cuda")
        for entry in to_load:
            entry.model_load()
            current.insert(0, entry)

    def load(self, size):
        other = _OtherModel(size)
        self.load_models_gpu([other])
        return other

    def entries(self, runtime):
        return [e for e in self.current_loaded_models if e.model is runtime._patcher]


class _CudaRuntime(tr.TipoRuntime):
    """진짜 generate·_place·_cpu_params·_restore_cpu 를 device="cuda" 로 탄다. torch.cuda 를 부르는 세 곳만 막는다."""

    def __init__(self, model, free=8 * GIB, memory=True):
        self.emptied = 0
        self.inputs_to = []
        self.free = free
        self.memory = _FakeMemory() if memory is True else memory
        self.memory_calls = 0
        super().__init__(loader=lambda d: (model, _FakeTokenizer()), directory=Path("C:/nope"),
                         cuda_info=lambda: (True, self.free), forge_memory=self._memory)

    def _memory(self):
        self.memory_calls += 1
        return None if self.memory is None else (self.memory, _FakeModelPatcher)

    def _move_inputs(self, ids, device):
        self.inputs_to.append(device)
        return ids

    def _rng_devices(self, device):
        return []

    def _empty_cache(self):
        self.emptied += 1


class CudaPathTests(unittest.TestCase):
    """GPU 회차의 실제 코드 경로 — 가짜 CUDA 텐서로 올림(H2D)·내림(D2H) 횟수와 dtype 을 센다. GPU 는 쓰지 않는다."""

    def _click(self, runtime, device="GPU", keep=False, seed=3):
        return runtime.generate("tag: 1girl", requested_device=device, max_new_tokens=8, seed=seed, keep_on_gpu=keep)

    @staticmethod
    def _where(model):
        return {(p.data.device, p.data.dtype) for p in model.params}

    def test_default_gpu_click_uploads_fp16_once_and_never_copies_back(self):
        model = _FakeCudaModel()
        originals = [p.data for p in model.params]
        runtime = _CudaRuntime(model)
        for n in (1, 2):
            result = self._click(runtime)
            self.assertEqual(result.device, "cuda")
            self.assertFalse(result.resident)
            self.assertEqual(model.seen[-1], {("cuda", torch.float16)}, "생성 때는 모든 파라미터가 GPU fp16")
            self.assertEqual(runtime.inputs_to[-1], "cuda")
            self.assertEqual(len(model.h2d()), 3 * n, "클릭마다 파라미터마다 한 번 올린다(기본 동작 그대로)")
            self.assertEqual(model.d2h(), [], "내릴 때 GPU→CPU 복사는 없다")
            self.assertEqual([p.data for p in model.params], originals, "쥐어 둔 CPU 텐서 그대로 되돌린다")
            self.assertEqual(runtime.emptied, n)
        self.assertEqual(model.buffer_moves, ["cpu", "cuda", "cpu"] * 2, "RAM 에서 fp16 확인 → 올림 → 내림(버퍼)")
        self.assertEqual((runtime.memory_calls, runtime.memory.loads), (0, []), "끄기는 Forge 메모리 관리를 건드리지 않는다")
        self.assertIsNone(runtime._patcher)

    def test_fp32_resting_model_is_cast_in_ram_before_the_upload(self):
        for keep in (False, True):
            model = _FakeCudaModel(dtype=torch.float32)
            runtime = _CudaRuntime(model)
            self._click(runtime, keep=keep)
            self.assertEqual(model.h2d(), [("cpu", "cuda", torch.float16)] * 3,
                             "CPU 에서 fp16 으로 바꾼 뒤 올린다 — fp32 4 GB 를 올리지 않는다")
            self.assertEqual(self._where(model), {("cuda" if keep else "cpu", torch.float16)})

    def test_keep_hands_the_model_to_forge_and_skips_the_next_upload(self):
        model = _FakeCudaModel()
        originals = [p.data for p in model.params]
        runtime = _CudaRuntime(model)
        memory = runtime.memory
        first = self._click(runtime, keep=True)
        self.assertTrue(first.resident)
        self.assertTrue(runtime.resident)
        self.assertEqual(self._where(model), {("cuda", torch.float16)}, "끝나도 GPU 에 둔다")
        self.assertEqual(model.seen[-1], {("cuda", torch.float16)})
        self.assertEqual(runtime.inputs_to[-1], "cuda")
        self.assertEqual(runtime.emptied, 0)
        [entry] = memory.entries(runtime)
        self.assertEqual(entry.model.load_device, "cuda")
        self.assertEqual(entry.model.offload_device, torch.device("cpu"))
        self.assertEqual(entry.model.model_size(), 3 * 1024 * 2, "Forge 에 알리는 크기 = fp16 가중치")
        self.assertEqual(entry.model.loaded_size(), entry.model.model_size(), "통째로 올라갔다고 알린다")
        self.assertEqual(memory.loads[-1][1], True, "force_full_load — 반쯤 올린 TIPO 는 쓸 수 없다")
        second = self._click(runtime, keep=True, seed=4)
        self.assertTrue(second.resident)
        self.assertEqual(len(model.h2d()), 3, "두 번째 클릭은 올리지 않는다(H2D 0)")
        self.assertEqual(len(memory.entries(runtime)), 1, "Forge 목록에 한 번만")
        self.assertEqual(len(memory.loads), 2, "클릭마다 load_models_gpu — 자리 확보·순서는 Forge 가")
        self.assertEqual(model.seen, [{("cuda", torch.float16)}] * 2)
        memory.free_memory(1e30, "cuda")   # 생성 시작 등으로 Forge 가 자리가 필요할 때
        self.assertFalse(runtime.resident)
        self.assertEqual(memory.entries(runtime), [])
        self.assertEqual([p.data for p in model.params], originals, "Forge 가 내려도 쥐어 둔 CPU 텐서로 — 복사 없음")
        self.assertEqual(model.buffer_moves[-1], "cpu")
        self.assertEqual(model.d2h(), [])
        self.assertEqual(entry.model.loaded_size(), 0)
        self.assertEqual(entry.model.current_device, torch.device("cpu"))
        self.assertTrue(self._click(runtime, keep=True, seed=5).resident)
        self.assertEqual(len(model.h2d()), 6, "퇴출된 뒤의 클릭은 다시 올린다(RAM 사본에서 H2D 만)")
        self.assertIs(memory.entries(runtime)[0].model, entry.model, "패처는 다시 만들지 않는다")

    def test_forge_evicts_tipo_only_when_room_is_needed(self):
        model = _FakeCudaModel(count=2)
        for param in model.params:
            param.data._numel = GIB // 2   # fp16 로 1 GiB 씩 = 2 GiB
        runtime = _CudaRuntime(model)
        memory = runtime.memory
        self._click(runtime, keep=True)
        unet = memory.load(8 * GIB)   # 24 GiB 중 2 + 8 — 여유가 넉넉하면 둘 다 남는다
        self.assertTrue(runtime.resident, "여유가 있으면 GPU 에 남는다")
        self.assertEqual(unet.loaded_size(), 8 * GIB)
        big = memory.load(12 * GIB)   # 12 × 1.1 + 1 GiB 가 필요 — 2 GiB 인 TIPO 부터 내린다
        self.assertFalse(runtime.resident, "자리가 필요하면 Forge 가 내린다")
        self.assertEqual(memory.entries(runtime), [])
        self.assertEqual(self._where(model), {("cpu", torch.float16)}, "부분 퇴출 없이 통째로 CPU 로")
        self.assertEqual((unet.loaded_size(), big.loaded_size()), (8 * GIB, 12 * GIB), "TIPO 만 내려 충분했다")
        self.assertEqual(model.d2h(), [])

    def test_loading_tipo_lets_forge_make_room(self):
        model = _FakeCudaModel(count=2)
        for param in model.params:
            param.data._numel = GIB // 2
        runtime = _CudaRuntime(model)
        memory = runtime.memory
        unet = memory.load(20 * GIB)   # 여유 4 GiB — TIPO 2 GiB × 1.1 + 1 GiB 는 들어간다
        self._click(runtime, keep=True)
        self.assertEqual(unet.loaded_size(), 20 * GIB)
        runtime.release()
        unet2 = memory.load(1 * GIB)   # 여유 3 GiB — 이번엔 Forge 가 먼저 다른 모델을 내려 TIPO 자리를 만든다
        self._click(runtime, keep=True)
        self.assertTrue(runtime.resident)
        self.assertEqual(unet2.loaded_size(), 0, "작은 것부터 내렸다")
        self.assertEqual(unet.loaded_size(), 20 * GIB)

    def test_failed_generation_on_the_gpu_restores_the_cpu_tensors(self):
        for keep in (False, True):
            model = _FakeCudaModel(fail=True)
            originals = [p.data for p in model.params]
            runtime = _CudaRuntime(model)
            with self.assertRaises(RuntimeError):
                self._click(runtime, keep=keep)
            self.assertEqual([p.data for p in model.params], originals)
            self.assertFalse(runtime.resident, "실패하면 남기기를 골랐어도 내린다")
            self.assertEqual(runtime.memory.entries(runtime), [], "Forge 목록에서도 뺀다")
            self.assertEqual((model.d2h(), runtime.emptied), ([], 1))

    def test_failed_upload_inside_forge_restores_every_cpu_tensor(self):
        model = _FakeCudaModel(dtype=torch.float32)
        runtime = _CudaRuntime(model)
        real_place = tr.TipoRuntime._place

        def failing_place(model_, device, dtype):
            if device == "cuda":
                model_.params[0].data = model_.params[0].data.to(device="cuda", dtype=dtype)   # 일부만 올라간 채 실패
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return real_place(model_, device, dtype)

        runtime._place = failing_place
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            self._click(runtime, keep=True)
        self.assertEqual(self._where(model), {("cpu", torch.float16)}, "RAM 에서 바꾼 fp16 CPU 텐서로 전부 되돌린다")
        self.assertFalse(runtime.resident)
        self.assertEqual(runtime.memory.current_loaded_models, [], "Forge 목록에 들어가지 않았다")
        self.assertEqual(runtime.emptied, 1)

    def test_plain_gpu_click_after_a_kept_one_releases_first(self):
        model = _FakeCudaModel()
        originals = [p.data for p in model.params]
        runtime = _CudaRuntime(model)
        self._click(runtime, keep=True)
        result = self._click(runtime, keep=False)
        self.assertFalse(result.resident)
        self.assertEqual(runtime.memory.entries(runtime), [], "끄면 Forge 목록에서도 뺀다")
        self.assertEqual(len(model.h2d()), 6, "내린 뒤 예전처럼 올렸다가 내린다")
        self.assertEqual([p.data for p in model.params], originals)
        self.assertFalse(runtime.resident)
        self.assertEqual((model.d2h(), runtime.emptied), ([], 2))

    def test_cpu_click_while_resident_releases_first(self):
        model = _FakeCudaModel()
        runtime = _CudaRuntime(model)
        self._click(runtime, keep=True)
        result = self._click(runtime, device="CPU", keep=True)
        self.assertEqual(result.device, "cpu")
        self.assertFalse(result.resident)
        self.assertEqual(model.seen[-1], {("cpu", torch.float32)})
        self.assertEqual(model.d2h(), [], "GPU 사본은 버리고 CPU 텐서에서 fp32 로")
        self.assertFalse(runtime.resident)
        self.assertEqual(runtime.memory.entries(runtime), [])
        self.assertEqual(runtime.emptied, 1)

    def test_resident_weights_count_as_free_vram(self):
        model = _FakeCudaModel(count=2)
        for param in model.params:
            param.data._numel = GIB // 2   # fp16 로 1 GiB 씩 = 2 GiB
        runtime = _CudaRuntime(model, free=4 * GIB)
        self._click(runtime, keep=True)
        runtime.free = int(1.5 * GIB)   # 모델 2 GiB 가 이미 올라가 있으니 3.5 GiB 로 본다
        self.assertEqual(self._click(runtime, keep=True).device, "cuda")
        runtime.free = int(0.5 * GIB)
        result = self._click(runtime, keep=True)
        self.assertEqual(result.device, "cpu", "올라간 것까지 쳐도 3 GB 가 안 되면 CPU")
        self.assertIn("GB", result.note)
        self.assertFalse(runtime.resident, "CPU 로 돌기 전에 GPU 사본을 내린다")
        self.assertEqual(model.d2h(), [])

    def test_keep_outside_forge_does_not_stay_on_the_gpu(self):
        model = _FakeCudaModel()
        runtime = _CudaRuntime(model, memory=None)
        result = self._click(runtime, keep=True)
        self.assertEqual(result.device, "cuda")
        self.assertFalse(result.resident)
        self.assertFalse(runtime.resident)
        self.assertEqual(result.note, tr.KEEP_UNAVAILABLE_NOTE, "Forge 메모리 관리가 없으면 남기지 않는다")
        self.assertEqual(self._where(model), {("cpu", torch.float16)})
        self.assertEqual(runtime.emptied, 1)

    def test_forge_eviction_from_another_thread_waits_for_a_running_click(self):
        import threading

        model = _FakeCudaModel()
        runtime = _CudaRuntime(model)
        threads = []
        real_generate = model.generate

        def generate(input_ids, **kwargs):
            thread = threading.Thread(target=runtime.memory.free_memory, args=(1e30, "cuda"))
            thread.start()
            thread.join(0.2)
            threads.append(thread.is_alive())
            self.assertEqual(self._where(model), {("cuda", torch.float16)}, "생성 도중에는 파라미터를 빼 가지 않는다")
            threads.append(thread)
            return real_generate(input_ids, **kwargs)

        model.generate = generate
        self._click(runtime, keep=True)
        threads[1].join(5)
        self.assertTrue(threads[0], "다른 스레드의 퇴출은 클릭이 끝날 때까지 기다린다")
        self.assertFalse(threads[1].is_alive())
        self.assertFalse(runtime.resident, "기다렸다가 내린다")
        self.assertEqual(model.d2h(), [])

    def test_release_is_idempotent(self):
        model = _FakeCudaModel()
        runtime = _CudaRuntime(model)
        self.assertFalse(runtime.release())
        self._click(runtime, keep=True)
        self.assertTrue(runtime.release())
        self.assertEqual(runtime.memory.entries(runtime), [])
        self.assertFalse(runtime.release(), "두 번 내려도 아무것도 하지 않는다")
        self.assertEqual(runtime.emptied, 1)

    def test_patcher_only_unloads_whole_and_ignores_forge_reinsert_detach(self):
        model = _FakeCudaModel(dtype=torch.float32)
        runtime = _CudaRuntime(model)
        self._click(runtime, keep=True)
        patcher = runtime._patcher
        self.assertIsInstance(patcher, _FakeModelPatcher)
        self.assertIs(tr._patcher_class(_FakeModelPatcher), type(patcher), "기반 클래스마다 한 번만 만든다")
        self.assertEqual(patcher.partially_unload("cpu", 1), 0, "부분 퇴출은 하지 않는다 — Forge 가 detach 로 통째로")
        patcher.detach(unpatch_all=False)
        self.assertTrue(runtime.resident, "load_models_gpu 가 목록에서 뺐다 넣을 때의 detach 는 무시")
        self.assertEqual(patcher.partially_load("cuda", 1e32), 0, "이미 올라가 있으면 아무것도 옮기지 않는다")
        self.assertEqual(len(model.h2d()), 3)
        patcher.detach()
        self.assertFalse(runtime.resident)


class ForgeMemoryTests(unittest.TestCase):
    def test_outside_forge_is_none(self):
        with mock.patch.dict(sys.modules, {"backend": None}):
            self.assertIsNone(tr._forge_memory())

    def test_inside_forge_gives_memory_management_and_model_patcher(self):
        import types

        mm = types.ModuleType("backend.memory_management")
        base = types.ModuleType("backend.patcher.base")
        base.ModelPatcher = _FakeModelPatcher
        backend = types.ModuleType("backend")
        backend.memory_management = mm
        backend.__path__ = []
        patcher_pkg = types.ModuleType("backend.patcher")
        patcher_pkg.__path__ = []
        patcher_pkg.base = base
        with mock.patch.dict(sys.modules, {"backend": backend, "backend.memory_management": mm,
                                           "backend.patcher": patcher_pkg, "backend.patcher.base": base}):
            self.assertEqual(tr._forge_memory(), (mm, _FakeModelPatcher))

    def test_real_cuda_helpers_with_torch_cuda_mocked(self):
        runtime = tr.TipoRuntime(directory=Path("C:/nope"))
        with mock.patch("torch.cuda.current_device", return_value=1):
            self.assertEqual(runtime._rng_devices("cuda"), [1], "fork_rng 가 그 GPU 의 난수 상태도 되돌린다")
            self.assertEqual(runtime._rng_devices("cpu"), [])
        ids = mock.Mock()
        runtime._move_inputs(ids, "cuda")
        ids.to.assert_called_once_with("cuda")
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch("torch.cuda.empty_cache") as empty:
            runtime._empty_cache()
        empty.assert_called_once_with()
        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch("torch.cuda.empty_cache") as empty:
            runtime._empty_cache()
        empty.assert_not_called()


class SharedRuntimeTests(unittest.TestCase):
    def test_one_per_process(self):
        self.assertIs(tr.shared_runtime(), tr.shared_runtime())


if __name__ == "__main__":
    unittest.main()
