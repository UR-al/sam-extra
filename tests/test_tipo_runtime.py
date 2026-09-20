"""TIPO 런타임 — 파일 확인·받기, 장치 선택, 생성(가짜 모델·토크나이저, GPU 없이)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

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
        self.assertEqual(model.moves, [("cuda", torch.float16), ("cpu", torch.float16)],
                         "Forge 메모리 관리 밖이라 GPU 에 상주시키지 않는다")
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
        self.assertEqual(model.moves, [("cpu", torch.float32), ("cpu", torch.float16)],
                         "CPU 는 fp32 로 돌리고 끝나면 RAM 에 fp16 으로 둔다")

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
        self.assertEqual(model.moves[-1], ("cpu", torch.float16))

    def test_a_failed_move_to_the_gpu_is_undone(self):
        runtime, model, _ = self._runtime()
        runtime.fail_on = "cuda"
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            runtime.generate("tag:", requested_device="GPU", max_new_tokens=8, seed=1)
        self.assertEqual(model.moves, [("cuda", torch.float16), ("cpu", torch.float16)],
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


class RealLoaderTests(unittest.TestCase):
    """실제 로드 경로(_load_pretrained) 끝까지 — 아주 작은 무작위 모델을 같은 파일 형식으로 저장해 CPU 로 생성."""

    def test_saved_tiny_model_loads_and_generates_on_cpu(self):
        import json

        from tokenizers import Tokenizer, models, pre_tokenizers

        from sam3ext.tipo.kohaku import KohakuConfig, KohakuForCausalLM

        with tempfile.TemporaryDirectory() as tmp:
            words = ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>", "tag:", "target:", "1girl", "smile", "outdoors", ","]
            tokenizer = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(words)}, unk_token="<|unk|>"))
            tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            tokenizer.save(str(Path(tmp) / "tokenizer.json"))
            (Path(tmp) / "tokenizer_config.json").write_text(json.dumps({
                "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<|bos|>", "eos_token": "<|eos|>",
                "pad_token": "<|pad|>", "unk_token": "<|unk|>",
            }), encoding="utf-8")
            config = KohakuConfig(
                vocab_size=len(words), hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, intermediate_size=64, moe_intermediate_size=16, n_routed_experts=4, num_experts_per_tok=2,
                first_k_dense=1, bos_token_id=0, eos_token_id=1, pad_token_id=2,
            )
            torch.manual_seed(0)
            KohakuForCausalLM(config).to(torch.float16).save_pretrained(tmp)
            self.assertEqual(tr.missing_files(tmp), [])
            runtime = tr.TipoRuntime(directory=tmp, cuda_info=lambda: (False, 0))
            result = runtime.generate("target: tag: 1girl , smile", requested_device="GPU", max_new_tokens=4, seed=7)
        self.assertEqual(result.device, "cpu", "CUDA 가 없으면 CPU")
        self.assertIsInstance(result.text, str)
        self.assertEqual(runtime._model.dtype, torch.float16, "돌린 뒤에는 RAM 에 fp16 으로")
        self.assertEqual(runtime._model.model.rotary_emb.inv_freq.dtype, torch.float32)


class SharedRuntimeTests(unittest.TestCase):
    def test_one_per_process(self):
        self.assertIs(tr.shared_runtime(), tr.shared_runtime())


if __name__ == "__main__":
    unittest.main()
