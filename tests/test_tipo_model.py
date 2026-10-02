"""TIPO 모델 코드 번들 — HF 리포 원문 그대로인지, venv 의 transformers 로 도는지(아주 작은 무작위 모델, CPU)."""
from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KOHAKU_DIR = ROOT / "sam3ext" / "tipo" / "kohaku"
# KBlueLeaf/TIPO-v2.1-1B-A200M @ f5a318524a4ab30cdbbf51816cf406170f454e65 (KohakUwULLM, Apache-2.0)
PINNED_SHA256 = {
    "configuration_kohaku.py": "629eced6167c8a583874fed3cb8e424b345c96e2fda407d2fdb20c8e001ccfc8",
    "modeling_kohaku.py": "49db88b1e09f53d55bbf49ef50d47e09034bb4e5f1e1e92e154a599a14d8901d",
}


class BundledCodeTests(unittest.TestCase):
    def test_files_are_the_upstream_originals(self):
        for name, digest in PINNED_SHA256.items():
            with self.subTest(file=name):
                data = (KOHAKU_DIR / name).read_bytes().replace(b"\r\n", b"\n")   # git autocrlf 체크아웃에도
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest)


class TinyModelTests(unittest.TestCase):
    """가중치 없이 무작위로 만든 아주 작은 설정 — 번들 코드가 이 venv 의 transformers 로 도는지만 본다."""

    def setUp(self):
        from sam3ext.tipo.kohaku import KohakuConfig, KohakuForCausalLM

        config = KohakuConfig(
            vocab_size=128, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, intermediate_size=64, moe_intermediate_size=16, n_routed_experts=4, num_experts_per_tok=2,
            first_k_dense=1, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
        torch.manual_seed(0)
        self.model = KohakuForCausalLM(config).eval()

    def test_forward_logits(self):
        ids = torch.tensor([[1, 5, 6, 7]])
        with torch.inference_mode():
            logits = self.model(input_ids=ids, use_cache=False).logits
        self.assertEqual(tuple(logits.shape), (1, 4, 128))

    def test_generate_with_the_model_card_sampling(self):
        ids = torch.tensor([[1, 5, 6, 7]])
        with torch.inference_mode():
            out = self.model.generate(
                ids, attention_mask=torch.ones_like(ids), max_new_tokens=5, do_sample=True,
                temperature=1.0, min_p=0.1, eos_token_id=2, pad_token_id=0,
            )
        self.assertEqual(out.shape[0], 1)
        self.assertGreater(out.shape[1], 4)
        self.assertLessEqual(out.shape[1], 9)


if __name__ == "__main__":
    unittest.main()
