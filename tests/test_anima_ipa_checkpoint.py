"""어댑터 safetensors 의 키·shape·메타데이터만 보고 구조를 알아낸다."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext.anima_ipa.checkpoint import (  # noqa: E402
    UNSUPPORTED_INDEPENDENT_Q,
    UNSUPPORTED_INJECT_BEFORE_MLP,
    inspect_adapter,
)


def _per_block(blocks=4, embed=768, inner=2048):
    keys, shapes = [], {}
    for i in range(blocks):
        for name in ("ip_k_proj", "ip_v_proj"):
            keys.append(f"blocks.{i}.{name}.weight")
            shapes[f"blocks.{i}.{name}.weight"] = (inner, embed)
            keys.append(f"blocks.{i}.{name}.bias")
            shapes[f"blocks.{i}.{name}.bias"] = (inner,)
        keys.append(f"blocks.{i}.adaln_ip.1.weight")
        shapes[f"blocks.{i}.adaln_ip.1.weight"] = (inner, inner)
    return keys, shapes


class PerBlockTests(unittest.TestCase):
    def test_block_count_and_dims_come_from_the_keys(self):
        keys, shapes = _per_block(blocks=28)
        spec = inspect_adapter(keys, shapes, {})
        self.assertEqual(spec.num_blocks, 28)
        self.assertEqual(spec.embed_dim, 768)
        self.assertEqual(spec.inner_dim, 2048)
        self.assertFalse(spec.shared_projection)
        self.assertTrue(spec.supported)

    def test_optional_pieces_default_to_absent(self):
        keys, shapes = _per_block()
        spec = inspect_adapter(keys, shapes, {})
        self.assertIsNone(spec.compressor)
        self.assertFalse(spec.has_self_attn)
        self.assertFalse(spec.has_siglip_norm)
        self.assertFalse(spec.has_null_tokens)
        self.assertEqual(spec.lora_blocks, ())
        self.assertEqual(spec.lora_rank, 0)
        self.assertFalse(spec.norm_keys)


class SharedProjectionTests(unittest.TestCase):
    def test_shared_projection_dims_come_from_the_expand_layer(self):
        keys = [
            "shared_ip_k_proj.expand.weight",
            "shared_ip_v_proj.expand.weight",
            "blocks.0.adaln_ip.1.weight",
            "blocks.7.adaln_ip.1.weight",
        ]
        shapes = {
            "shared_ip_k_proj.expand.weight": (1536, 768),
            "shared_ip_v_proj.expand.weight": (1536, 768),
            "blocks.0.adaln_ip.1.weight": (1536, 1536),
            "blocks.7.adaln_ip.1.weight": (1536, 1536),
        }
        spec = inspect_adapter(keys, shapes, {})
        self.assertTrue(spec.shared_projection)
        self.assertEqual(spec.num_blocks, 8)
        self.assertEqual(spec.embed_dim, 768)
        self.assertEqual(spec.inner_dim, 1536)


class ExtraModuleTests(unittest.TestCase):
    def test_compressor_shape_gives_queries_and_layers(self):
        keys, shapes = _per_block()
        keys += [
            "siglip_compressor.queries",
            "siglip_compressor.layers.0.norm_q.weight",
            "siglip_compressor.layers.1.norm_q.weight",
        ]
        shapes["siglip_compressor.queries"] = (64, 768)
        spec = inspect_adapter(keys, shapes, {})
        self.assertEqual(spec.compressor, (64, 2))

    def test_self_attn_norm_and_null_tokens_are_detected(self):
        keys, shapes = _per_block()
        keys += ["ip_self_attn.qkv.weight", "siglip_norm.weight", "null_tokens"]
        spec = inspect_adapter(keys, shapes, {})
        self.assertTrue(spec.has_self_attn)
        self.assertTrue(spec.has_siglip_norm)
        self.assertTrue(spec.has_null_tokens)

    def test_lora_blocks_and_rank_are_read_from_the_peft_keys(self):
        keys, shapes = _per_block()
        for block in (0, 3):
            for name in ("q_proj", "k_proj"):
                down = (
                    f"lora.base_model.model.blocks.{block}.cross_attn."
                    f"{name}.lora_A.default.weight"
                )
                up = (
                    f"lora.base_model.model.blocks.{block}.cross_attn."
                    f"{name}.lora_B.default.weight"
                )
                keys += [down, up]
                shapes[down] = (16, 2048)
                shapes[up] = (2048, 16)
        spec = inspect_adapter(keys, shapes, {})
        self.assertEqual(spec.lora_blocks, (0, 3))
        self.assertEqual(spec.lora_rank, 16)

    def test_metadata_flag_turns_on_key_normalisation(self):
        keys, shapes = _per_block()
        spec = inspect_adapter(keys, shapes, {"ip_norm_keys": "True"})
        self.assertTrue(spec.norm_keys)


class RejectionTests(unittest.TestCase):
    """상류가 만들어만 두고 쓰지 않는 두 경로. 조용히 틀리게 도느니 거부한다."""

    def test_inject_before_mlp_is_rejected(self):
        keys, shapes = _per_block()
        spec = inspect_adapter(keys, shapes, {"ip_inject_before_mlp": "true"})
        self.assertFalse(spec.supported)
        self.assertIn(UNSUPPORTED_INJECT_BEFORE_MLP, spec.unsupported)

    def test_independent_ip_q_is_rejected(self):
        keys, shapes = _per_block()
        keys.append("shared_ip_q_proj.weight")
        shapes["shared_ip_q_proj.weight"] = (2048, 2048)
        spec = inspect_adapter(keys, shapes, {})
        self.assertFalse(spec.supported)
        self.assertIn(UNSUPPORTED_INDEPENDENT_Q, spec.unsupported)


if __name__ == "__main__":
    unittest.main()
