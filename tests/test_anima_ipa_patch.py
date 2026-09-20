"""Anima DiT 주입 — 가짜 DiT 로 설치·주입·복원을 검증한다.

진짜 Anima 를 띄우지 않는다. 블록 API(cross_attn 의 q_proj/q_norm/n_heads/head_dim 과
forward 시그니처)만 흉내 내면 충분하다 — backend/nn/anima.py:258-363 참고.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext.anima_ipa import patch as patch_module  # noqa: E402
from sam3ext.anima_ipa.checkpoint import AdapterSpec  # noqa: E402

EMBED, INNER, HEADS = 768, 64, 4
TOKENS = 5


def _spec(**overrides):
    values = dict(
        num_blocks=2, embed_dim=EMBED, inner_dim=INNER, shared_projection=False,
        compressor=None, has_self_attn=False, has_siglip_norm=False,
        has_null_tokens=False, lora_blocks=(), lora_rank=0, norm_keys=False,
        unsupported=(),
    )
    values.update(overrides)
    return AdapterSpec(**values)


class FakeCrossAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads, self.head_dim = HEADS, INNER // HEADS
        self.q_proj = nn.Linear(INNER, INNER, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_proj = nn.Linear(INNER, INNER, bias=False)

    def forward(self, x, context, rope_emb=None, transformer_options=None):
        return x * 0.5


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_dim = INNER
        self.cross_attn = FakeCrossAttn()

    def forward(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb, rope_emb_L_1_1_D=None,
                adaln_lora_B_T_3D=None, extra_per_block_pos_emb=None,
                transformer_options=None):
        # 진짜 블록처럼 cross_attn 을 부른다 — 전방 훅이 Q 입력을 잡는 자리다.
        return x_B_T_H_W_D + self.cross_attn(
            x_B_T_H_W_D, crossattn_emb, transformer_options=transformer_options
        )


class FakeDiT(nn.Module):
    def __init__(self, blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock() for _ in range(blocks)])


def _weights(spec):
    out = {}
    for i in range(spec.num_blocks):
        out[f"blocks.{i}.ip_k_proj.weight"] = torch.randn(INNER, EMBED) * 0.02
        out[f"blocks.{i}.ip_k_proj.bias"] = torch.zeros(INNER)
        out[f"blocks.{i}.ip_v_proj.weight"] = torch.randn(INNER, EMBED) * 0.02
        out[f"blocks.{i}.ip_v_proj.bias"] = torch.zeros(INNER)
        out[f"blocks.{i}.adaln_ip.1.weight"] = torch.zeros(INNER, INNER)
        out[f"blocks.{i}.adaln_ip.1.bias"] = torch.full((INNER,), 0.1)
    return out


def _inputs(batch=1):
    x = torch.randn(batch, 1, 2, 2, INNER)
    emb = torch.randn(batch, 1, INNER)
    context = torch.randn(batch, 3, INNER)
    return x, emb, context


def _run(dit, tokens=None, inputs=None):
    x, emb, context = inputs or _inputs()
    options = {} if tokens is None else {"sam3_ip_tokens": tokens}
    out = x
    for block in dit.blocks:
        out = block(out, emb, context, transformer_options=options)
    return out


class InstallTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.spec = _spec()
        self.weights = _weights(self.spec)

    def test_installing_changes_the_output_when_tokens_are_present(self):
        dit = FakeDiT()
        inputs = _inputs()
        before = _run(dit, inputs=inputs)
        patch_module.install(dit, self.spec, self.weights)
        after = _run(dit, tokens=torch.randn(1, TOKENS, EMBED), inputs=inputs)
        self.assertFalse(torch.allclose(before, after))

    def test_without_tokens_the_patched_block_is_a_no_op(self):
        dit = FakeDiT()
        inputs = _inputs()
        before = _run(dit, inputs=inputs)
        patch_module.install(dit, self.spec, self.weights)
        self.assertTrue(torch.equal(before, _run(dit, inputs=inputs)))

    def test_zero_gate_scale_means_no_contribution(self):
        dit = FakeDiT()
        inputs = _inputs()
        before = _run(dit, inputs=inputs)
        patch_module.install(dit, self.spec, self.weights, gate_scale=0.0)
        after = _run(dit, tokens=torch.randn(1, TOKENS, EMBED), inputs=inputs)
        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_installing_twice_patches_nothing_the_second_time(self):
        dit = FakeDiT()
        self.assertEqual(patch_module.install(dit, self.spec, self.weights), 2)
        self.assertEqual(patch_module.install(dit, self.spec, self.weights), 0)

    def test_the_captured_activation_is_dropped_after_the_forward(self):
        """블록마다 활성값 한 벌을 붙잡아 두면 고해상도에서 VRAM 을 잡아먹는다."""
        dit = FakeDiT()
        patch_module.install(dit, self.spec, self.weights)
        _run(dit, tokens=torch.randn(1, TOKENS, EMBED))
        for block in dit.blocks:
            self.assertIsNone(getattr(block, "_sam3_ip_x", None))

    def test_all_zero_tokens_contribute_nothing(self):
        dit = FakeDiT()
        inputs = _inputs()
        before = _run(dit, inputs=inputs)
        patch_module.install(dit, self.spec, self.weights)
        after = _run(dit, tokens=torch.zeros(1, TOKENS, EMBED), inputs=inputs)
        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_a_missing_block_weight_stops_instead_of_installing_noise(self):
        """빠진 가중치를 건너뛰면 무동작이 아니라 랜덤 투영이 잔차에 더해진다."""
        weights = _weights(_spec(num_blocks=2))
        for key in [k for k in weights if k.startswith("blocks.1.")]:
            del weights[key]
        with self.assertRaises(RuntimeError) as caught:
            patch_module.install(FakeDiT(blocks=2), _spec(num_blocks=2), weights)
        self.assertIn("1", str(caught.exception))


class BlockLineageTests(unittest.TestCase):
    """28블록 어댑터를 40/52블록 모델에 얹는다 — Forge 가 Edit LoRA 에 쓰는 계보표 그대로.

    표는 ``sam3ext/anima_lora_blocks.py::BLOCK_MAPPINGS`` 이고, 그 값은 Forge 본체
    ``extensions-builtin/sd_forge_lora/networks.py:60-64`` 와 같다.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.spec = _spec(num_blocks=28)
        self.weights = _weights(self.spec)

    def test_the_same_depth_is_the_identity_plan(self):
        self.assertEqual(patch_module._block_plan(28, 28), tuple(range(28)))
        self.assertEqual(patch_module._block_plan(52, 52), tuple(range(52)))

    def test_the_known_upward_pairs_use_every_adapter_block(self):
        for adapter, model in ((28, 40), (28, 52), (40, 52)):
            plan = patch_module._block_plan(adapter, model)
            self.assertEqual(len(plan), model, (adapter, model))
            self.assertEqual(set(plan), set(range(adapter)), (adapter, model))

    def test_downward_pairs_are_refused_even_though_the_table_has_them(self):
        """BLOCK_MAPPINGS 에는 (52,28) 같은 축소 키도 있다 — 그건 어댑터 블록을 버린다."""
        for adapter, model in ((52, 28), (52, 40), (40, 28)):
            self.assertIsNone(patch_module._block_plan(adapter, model), (adapter, model))

    def test_unknown_depth_pairs_are_refused(self):
        self.assertIsNone(patch_module._block_plan(2, 4))
        self.assertIsNone(patch_module._block_plan(27, 28))

    def test_a_deeper_model_is_patched_through_the_whole_lineage(self):
        from sam3ext.anima_lora_blocks import BLOCK_MAPPINGS

        dit = FakeDiT(blocks=52)
        self.assertEqual(patch_module.install(dit, self.spec, self.weights), 52)
        plan = BLOCK_MAPPINGS[(28, 52)]
        for index in (0, 1, 3, 4, 25, 51):
            source = plan[index]
            self.assertTrue(
                torch.equal(
                    dit.blocks[index].ip_k_proj.weight,
                    self.weights[f"blocks.{source}.ip_k_proj.weight"],
                ),
                f"block {index} should carry adapter block {source}",
            )

    def test_no_block_of_a_deeper_model_is_left_without_injection(self):
        dit = FakeDiT(blocks=52)
        patch_module.install(dit, self.spec, self.weights)
        for index, block in enumerate(dit.blocks):
            self.assertTrue(getattr(block, "_sam3_ip_patched", False), index)

    def test_the_lora_of_a_mapped_block_follows_the_same_source(self):
        spec = _spec(num_blocks=28, lora_blocks=(1,), lora_rank=2)
        weights = _weights(spec)
        base = "lora.base_model.model.blocks.1.cross_attn.q_proj"
        weights[f"{base}.lora_A.default.weight"] = torch.randn(2, INNER) * 0.1
        weights[f"{base}.lora_B.default.weight"] = torch.randn(INNER, 2) * 0.1
        dit = FakeDiT(blocks=52)
        patch_module.install(dit, spec, weights, use_lora=True)
        # plan[1] == plan[2] == plan[3] == 1 이므로 그 세 블록이 LoRA 를 받는다.
        for index in (1, 2, 3):
            self.assertIsInstance(
                dit.blocks[index].cross_attn.q_proj, patch_module._LoRALinear, index
            )
        self.assertNotIsInstance(
            dit.blocks[0].cross_attn.q_proj, patch_module._LoRALinear
        )


class CompatibilityTests(unittest.TestCase):
    def test_more_adapter_blocks_than_the_model_has_is_refused(self):
        with self.assertRaises(RuntimeError) as caught:
            patch_module.ensure_compatible(_spec(num_blocks=36), FakeDiT(blocks=28))
        self.assertIn("36", str(caught.exception))
        self.assertIn("28", str(caught.exception))

    def test_a_known_deeper_model_passes(self):
        patch_module.ensure_compatible(_spec(num_blocks=28), FakeDiT(blocks=40))
        patch_module.ensure_compatible(_spec(num_blocks=28), FakeDiT(blocks=52))
        patch_module.ensure_compatible(_spec(num_blocks=40), FakeDiT(blocks=52))

    def test_a_deeper_adapter_is_refused_with_both_numbers(self):
        """계보표에 (52,28) 이 있지만 그건 어댑터를 버리는 축소다 — Forge 본체도 거부한다."""
        for adapter, model in ((52, 28), (52, 40), (40, 28)):
            with self.assertRaises(RuntimeError) as caught:
                patch_module.ensure_compatible(_spec(num_blocks=adapter), FakeDiT(blocks=model))
            self.assertIn(str(adapter), str(caught.exception))
            self.assertIn(str(model), str(caught.exception))

    def test_an_unknown_depth_pair_is_refused_with_both_numbers(self):
        with self.assertRaises(RuntimeError) as caught:
            patch_module.ensure_compatible(_spec(num_blocks=2), FakeDiT(blocks=4))
        self.assertIn("2", str(caught.exception))
        self.assertIn("4", str(caught.exception))

    def test_a_dimension_mismatch_is_refused_with_both_numbers(self):
        with self.assertRaises(RuntimeError) as caught:
            patch_module.ensure_compatible(_spec(inner_dim=2560), FakeDiT())
        self.assertIn("2560", str(caught.exception))
        self.assertIn(str(INNER), str(caught.exception))

    def test_an_unsupported_adapter_is_refused_by_name(self):
        spec = _spec(unsupported=("독립 IP Q 투영(shared_ip_q_proj)",))
        with self.assertRaises(RuntimeError) as caught:
            patch_module.ensure_compatible(spec, FakeDiT())
        self.assertIn("shared_ip_q_proj", str(caught.exception))

    def test_a_model_without_blocks_is_refused(self):
        with self.assertRaises(RuntimeError):
            patch_module.ensure_compatible(_spec(), nn.Linear(2, 2))

    def test_a_matching_pair_passes(self):
        patch_module.ensure_compatible(_spec(), FakeDiT())


class CfgTests(unittest.TestCase):
    def test_correction_matches_the_documented_formula(self):
        denoised = torch.zeros(2, 4)
        with_ip = torch.full((2, 4), 3.0)
        without_ip = torch.full((2, 4), 1.0)
        out = patch_module.cfg_correction(
            denoised, with_ip, without_ip, cond_scale=7.0, ip_cfg_scale=4.0
        )
        # (4.0 - 1.0 - 7.0) * (3 - 1) = -8
        self.assertTrue(torch.allclose(out, torch.full((2, 4), -8.0)))

    def test_uncond_rows_get_the_null_tokens(self):
        tokens = torch.ones(1, TOKENS, EMBED)
        null = torch.zeros(1, TOKENS, EMBED)
        handler = patch_module.InjectionHandler(
            patch_module.Injection(tokens=tokens, null_tokens=null)
        )
        seen = {}

        def apply_model(x, t, **c):
            seen["tokens"] = c["transformer_options"]["sam3_ip_tokens"]
            return x

        handler(apply_model, {
            "input": torch.zeros(2, 1, 2, 2, INNER),
            "timestep": torch.zeros(2),
            "c": {"transformer_options": {}},
            "cond_or_uncond": [0, 1],
        })
        batch = seen["tokens"]
        self.assertEqual(batch.shape[0], 2)
        self.assertTrue(torch.allclose(batch[0], tokens[0]))
        self.assertTrue(torch.allclose(batch[1], null[0]))

    def test_the_existing_transformer_options_are_not_mutated(self):
        original = {"other": 1}
        handler = patch_module.InjectionHandler(
            patch_module.Injection(
                tokens=torch.ones(1, TOKENS, EMBED),
                null_tokens=torch.zeros(1, TOKENS, EMBED),
            )
        )
        handler(lambda x, t, **c: x, {
            "input": torch.zeros(1, 1, 2, 2, INNER),
            "timestep": torch.zeros(1),
            "c": {"transformer_options": original},
            "cond_or_uncond": [0],
        })
        self.assertEqual(original, {"other": 1})

    def test_one_pass_runs_the_model_once(self):
        handler = patch_module.InjectionHandler(
            patch_module.Injection(
                tokens=torch.ones(1, TOKENS, EMBED),
                null_tokens=torch.zeros(1, TOKENS, EMBED),
            )
        )
        calls = []
        handler(lambda x, t, **c: calls.append(c) or x, {
            "input": torch.zeros(2, 1, 2, 2, INNER),
            "timestep": torch.zeros(2),
            "c": {"transformer_options": {}},
            "cond_or_uncond": [0, 1],
        })
        self.assertEqual(len(calls), 1)
        self.assertEqual(handler.post_cfg({"denoised": torch.ones(1)}).tolist(), [1.0])

    def test_two_pass_runs_the_model_twice_and_corrects_afterwards(self):
        handler = patch_module.InjectionHandler(
            patch_module.Injection(
                tokens=torch.ones(1, TOKENS, EMBED),
                null_tokens=torch.zeros(1, TOKENS, EMBED),
                separate_cfg=True,
                cfg_scale=4.0,
            )
        )
        calls = []

        def apply_model(x, t, **c):
            calls.append(c["transformer_options"]["sam3_ip_tokens"])
            return torch.full((2, 4), float(len(calls)))

        handler(apply_model, {
            "input": torch.zeros(2, 4),
            "timestep": torch.zeros(2),
            "c": {"transformer_options": {}},
            "cond_or_uncond": [0, 1],
        })
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.allclose(calls[1], torch.zeros_like(calls[1])))
        corrected = handler.post_cfg({
            "denoised": torch.zeros(1, 4),
            "cond_denoised": torch.full((1, 4), 3.0),
            "cond_scale": 7.0,
        })
        # cond_without_ip = 2.0 (두 번째 호출의 cond 조각), (4-1-7) * (3-2) = -4
        self.assertTrue(torch.allclose(corrected, torch.full((1, 4), -4.0)))


class _FakePatcher:
    """UnetPatcher 대역 — clone/add_object_patch/래퍼 등록만."""

    def __init__(self, dit):
        self.model = types.SimpleNamespace(diffusion_model=dit)
        self.object_patches = {}
        self.wrapper = None
        self.post_cfg = None

    def clone(self):
        return _FakePatcher(self.model.diffusion_model)

    def get_model_object(self, name):
        return getattr(self.model, name)

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def set_model_unet_function_wrapper(self, fn):
        self.wrapper = fn

    def set_model_sampler_post_cfg_function(self, fn):
        self.post_cfg = fn


class _FakeSdModel:
    def __init__(self, dit):
        self.forge_objects_after_applying_lora = types.SimpleNamespace(
            unet=_FakePatcher(dit)
        )


class PatchedUnetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.spec = _spec()
        self.weights = _weights(self.spec)
        self.injection = patch_module.Injection(
            tokens=torch.randn(1, TOKENS, EMBED),
            null_tokens=torch.zeros(1, TOKENS, EMBED),
        )

    def test_the_original_patcher_comes_back_and_the_original_dit_is_untouched(self):
        dit = FakeDiT()
        model = _FakeSdModel(dit)
        original = model.forge_objects_after_applying_lora.unet
        inputs = _inputs()
        before = _run(dit, inputs=inputs)

        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            swapped = model.forge_objects_after_applying_lora.unet
            self.assertIsNot(swapped, original)
            self.assertIn("diffusion_model", swapped.object_patches)
            self.assertIsNotNone(swapped.wrapper)

        self.assertIs(model.forge_objects_after_applying_lora.unet, original)
        # 원본 DiT 는 껍데기 복제본만 패치되었으므로 결과가 그대로여야 한다.
        self.assertTrue(torch.equal(before, _run(dit, inputs=inputs)))
        self.assertTrue(torch.equal(before, _run(dit, tokens=self.injection.tokens, inputs=inputs)))

    def test_the_patched_copy_shares_weights_but_not_modules(self):
        dit = FakeDiT()
        model = _FakeSdModel(dit)
        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            patched = model.forge_objects_after_applying_lora.unet.object_patches[
                "diffusion_model"
            ]
            self.assertIsNot(patched, dit)
            self.assertIsNot(patched.blocks[0], dit.blocks[0])
            self.assertIs(
                patched.blocks[0].cross_attn.q_proj.weight,
                dit.blocks[0].cross_attn.q_proj.weight,
            )
            self.assertTrue(getattr(patched.blocks[0], "_sam3_ip_patched", False))
            self.assertFalse(getattr(dit.blocks[0], "_sam3_ip_patched", False))

    def test_an_exception_inside_the_block_still_restores(self):
        dit = FakeDiT()
        model = _FakeSdModel(dit)
        original = model.forge_objects_after_applying_lora.unet
        with self.assertRaises(ValueError):
            with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
                raise ValueError("boom")
        self.assertIs(model.forge_objects_after_applying_lora.unet, original)

    def test_an_incompatible_adapter_never_swaps_the_patcher(self):
        dit = FakeDiT()
        model = _FakeSdModel(dit)
        original = model.forge_objects_after_applying_lora.unet
        with self.assertRaises(RuntimeError):
            with patch_module.patched_unet(
                model, _spec(inner_dim=2560), self.weights, self.injection
            ):
                pass
        self.assertIs(model.forge_objects_after_applying_lora.unet, original)


class LoraTests(unittest.TestCase):
    def _lora_weights(self, spec):
        weights = _weights(spec)
        base = "lora.base_model.model.blocks.0.cross_attn.q_proj"
        weights[f"{base}.lora_A.default.weight"] = torch.randn(2, INNER) * 0.1
        weights[f"{base}.lora_B.default.weight"] = torch.randn(INNER, 2) * 0.1
        return weights

    def test_lora_wraps_the_cross_attention_linear_and_changes_its_output(self):
        torch.manual_seed(0)
        spec = _spec(lora_blocks=(0,), lora_rank=2)
        weights = self._lora_weights(spec)
        dit = FakeDiT()
        sample = torch.randn(1, INNER)
        plain = dit.blocks[0].cross_attn.q_proj(sample).clone()
        patch_module.install(dit, spec, weights, use_lora=True)
        self.assertFalse(
            torch.allclose(plain, dit.blocks[0].cross_attn.q_proj(sample))
        )

    def test_use_lora_false_leaves_the_linear_alone(self):
        torch.manual_seed(0)
        spec = _spec(lora_blocks=(0,), lora_rank=2)
        weights = self._lora_weights(spec)
        dit = FakeDiT()
        sample = torch.randn(1, INNER)
        plain = dit.blocks[0].cross_attn.q_proj(sample).clone()
        patch_module.install(dit, spec, weights, use_lora=False)
        self.assertTrue(torch.allclose(plain, dit.blocks[0].cross_attn.q_proj(sample)))


if __name__ == "__main__":
    unittest.main()
