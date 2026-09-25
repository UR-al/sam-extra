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

    def forward(self, x, emb, context, transformer_options=None):
        # 진짜 Anima.forward 처럼 self.blocks 를 돈다 — DiT 수준 래퍼 테스트가 여기에 닿는다.
        for block in self.blocks:
            x = block(x, emb, context, transformer_options=transformer_options)
        return x


class SkippingBlock(FakeBlock):
    """cross_attn 을 부르지 않는 블록 — 껍데기 훅이 못 잡는 배선 오류를 흉내 낸다."""

    def forward(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb, **kwargs):
        return x_B_T_H_W_D


def _foreign_block_wrapper(block, calls):
    """Safe PAG 식 인스턴스 forward — 원본 블록에 바인딩된 메서드를 닫아 둔다."""
    bound = block.forward

    def wrapped(*args, **kwargs):
        calls.append("block")
        return bound(*args, **kwargs)

    return wrapped


def _foreign_dit_wrapper(dit, calls):
    """Anima 3.8B 런타임 식 인스턴스 forward — 항상 원본에 바인딩된 클래스 forward 를 부른다."""
    native = type(dit).forward.__get__(dit, type(dit))

    def patched_forward(x, emb, context, *args, **kwargs):
        calls.append("dit")
        return native(x, emb, context, *args, **kwargs)

    return patched_forward


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

    def test_a_forward_that_never_reaches_the_cross_attn_hook_raises_with_tokens(self):
        """훅이 안 불리면 조용히 주입을 건너뛰지 않고 멈춘다 — 토큰이 없을 때는 그대로 통과."""
        dit = FakeDiT()
        dit.blocks = nn.ModuleList([SkippingBlock(), SkippingBlock()])
        patch_module.install(dit, self.spec, self.weights)
        _run(dit)   # 토큰 없음: 주입 요청이 없으므로 오류도 없다
        with self.assertRaises(RuntimeError) as caught:
            _run(dit, tokens=torch.randn(1, TOKENS, EMBED))
        self.assertIn("IP 주입", str(caught.exception))

    def test_a_block_records_that_it_injected(self):
        dit = FakeDiT()
        patch_module.install(dit, self.spec, self.weights)
        self.assertFalse(any(getattr(b, "_sam3_ip_applied", False) for b in dit.blocks))
        _run(dit)
        self.assertFalse(any(getattr(b, "_sam3_ip_applied", False) for b in dit.blocks))
        _run(dit, tokens=torch.randn(1, TOKENS, EMBED))
        self.assertTrue(all(getattr(b, "_sam3_ip_applied", False) for b in dit.blocks))


class ShellTests(unittest.TestCase):
    def test_an_instance_forward_of_the_original_is_not_carried_into_the_shell(self):
        """다른 확장이 원본에 얹은 인스턴스 forward 는 원본에 바인딩돼 있어 껍데기에서 쓸모가 없다."""
        block = FakeBlock()
        block.forward = _foreign_block_wrapper(block, [])
        copied = patch_module.shell(block)
        self.assertNotIn("forward", copied.__dict__)
        self.assertIn("forward", block.__dict__)          # 원본은 그대로
        self.assertIs(copied.cross_attn, block.cross_attn)  # 나머지 공유는 그대로


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
        self.assertEqual(
            patch_module.install(dit, self.spec, self.weights, duplicate_policy="all"), 52
        )
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

    def test_no_block_of_a_deeper_model_is_left_without_injection_under_all(self):
        dit = FakeDiT(blocks=52)
        patch_module.install(dit, self.spec, self.weights, duplicate_policy="all")
        for index, block in enumerate(dit.blocks):
            self.assertTrue(getattr(block, "_sam3_ip_patched", False), index)

    def test_the_lora_of_a_mapped_block_follows_the_same_source(self):
        spec = _spec(num_blocks=28, lora_blocks=(1,), lora_rank=2)
        weights = _weights(spec)
        base = "lora.base_model.model.blocks.1.cross_attn.q_proj"
        weights[f"{base}.lora_A.default.weight"] = torch.randn(2, INNER) * 0.1
        weights[f"{base}.lora_B.default.weight"] = torch.randn(INNER, 2) * 0.1
        dit = FakeDiT(blocks=52)
        patch_module.install(dit, spec, weights, use_lora=True, duplicate_policy="all")
        # plan[1] == plan[2] == plan[3] == 1 이므로 그 세 블록이 LoRA 를 받는다.
        for index in (1, 2, 3):
            self.assertIsInstance(
                dit.blocks[index].cross_attn.q_proj, patch_module._LoRALinear, index
            )
        self.assertNotIsInstance(
            dit.blocks[0].cross_attn.q_proj, patch_module._LoRALinear
        )


class DuplicatePolicyTests(unittest.TestCase):
    """28블록 어댑터를 더 깊은 모델에 얹을 때 끼워 넣은 복제 블록을 어떻게 다룰지 (lineage/all/split).

    all 은 3.8B 에서 강도 0.5 에도, 2.9B 에서 1.0 에 격자 무늬로 깨졌고 lineage 는 둘 다 멀쩡했다
    (2026-09-23 GPU 확인). 그래서 기본값은 lineage 다.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.spec = _spec(num_blocks=28)
        self.weights = _weights(self.spec)

    def _first_occurrences(self, plan):
        seen, first = set(), []
        for index, source in enumerate(plan):
            if source not in seen:
                first.append(index)
                seen.add(source)
        return first

    def test_the_default_is_lineage(self):
        from sam3ext.anima_ipa import options

        self.assertEqual(options.DEFAULT_DUPLICATE_POLICY, "lineage")
        self.assertEqual(patch_module.DEFAULT_DUPLICATE_POLICY, "lineage")
        self.assertEqual(set(options.DUPLICATE_POLICIES), {"lineage", "all", "split"})

    def test_lineage_gates_only_the_first_block_of_each_adapter_block(self):
        for model in (40, 52):
            plan = patch_module._block_plan(28, model)
            gates = patch_module._duplicate_gates(plan, "lineage")
            first = self._first_occurrences(plan)
            self.assertEqual(len(first), 28, model)
            self.assertEqual([i for i, g in enumerate(gates) if g is not None], first, model)
            self.assertTrue(all(gates[i] == 1.0 for i in first), model)
            # 계보 블록은 어댑터 블록을 0..27 순서대로 한 번씩 쓴다
            self.assertEqual([plan[i] for i in first], list(range(28)), model)

    def test_split_gates_sum_to_one_per_adapter_block(self):
        for model in (40, 52):
            plan = patch_module._block_plan(28, model)
            gates = patch_module._duplicate_gates(plan, "split")
            totals: dict[int, float] = {}
            for source, gate in zip(plan, gates):
                totals[source] = totals.get(source, 0.0) + gate
            for source, total in totals.items():
                self.assertAlmostEqual(total, 1.0, msg=(model, source))

    def test_all_gates_every_block_at_full_strength(self):
        plan = patch_module._block_plan(28, 52)
        self.assertEqual(patch_module._duplicate_gates(plan, "all"), [1.0] * 52)

    def test_the_same_depth_is_the_same_under_every_policy(self):
        plan = patch_module._block_plan(28, 28)
        for policy in ("lineage", "all", "split"):
            self.assertEqual(patch_module._duplicate_gates(plan, policy), [1.0] * 28, policy)

    def test_lineage_install_leaves_inserted_blocks_untouched(self):
        dit = FakeDiT(blocks=52)
        plan = patch_module._block_plan(28, 52)
        first = set(self._first_occurrences(plan))
        self.assertEqual(
            patch_module.install(dit, self.spec, self.weights, duplicate_policy="lineage"), 28
        )
        for index, block in enumerate(dit.blocks):
            patched = getattr(block, "_sam3_ip_patched", False)
            self.assertEqual(patched, index in first, index)
            if index not in first:
                self.assertFalse(hasattr(block, "adaln_ip"), index)
                self.assertFalse(hasattr(block, "ip_k_proj"), index)

    def test_lineage_is_what_install_does_by_default(self):
        dit = FakeDiT(blocks=52)
        self.assertEqual(patch_module.install(dit, self.spec, self.weights), 28)

    def test_lineage_blocks_carry_their_own_adapter_block(self):
        dit = FakeDiT(blocks=52)
        plan = patch_module._block_plan(28, 52)
        patch_module.install(dit, self.spec, self.weights, duplicate_policy="lineage")
        for index in self._first_occurrences(plan):
            source = plan[index]
            self.assertTrue(
                torch.equal(
                    dit.blocks[index].ip_k_proj.weight,
                    self.weights[f"blocks.{source}.ip_k_proj.weight"],
                ),
                index,
            )

    def test_lineage_gives_the_adapter_lora_to_the_lineage_block_only(self):
        spec = _spec(num_blocks=28, lora_blocks=(1,), lora_rank=2)
        weights = _weights(spec)
        base = "lora.base_model.model.blocks.1.cross_attn.q_proj"
        weights[f"{base}.lora_A.default.weight"] = torch.randn(2, INNER) * 0.1
        weights[f"{base}.lora_B.default.weight"] = torch.randn(INNER, 2) * 0.1
        dit = FakeDiT(blocks=52)
        patch_module.install(dit, spec, weights, use_lora=True, duplicate_policy="lineage")
        plan = patch_module._block_plan(28, 52)
        holders = [i for i in range(52)
                   if isinstance(dit.blocks[i].cross_attn.q_proj, patch_module._LoRALinear)]
        self.assertEqual(holders, [plan.index(1)])

    def test_split_scales_the_gate_of_every_copy(self):
        dit = FakeDiT(blocks=52)
        plan = patch_module._block_plan(28, 52)
        patch_module.install(dit, self.spec, self.weights, gate_scale=0.8, duplicate_policy="split")
        for index, block in enumerate(dit.blocks):
            copies = plan.count(plan[index])
            self.assertAlmostEqual(block.sam3_ip_gate_scale, 0.8 / copies, msg=index)

    def test_an_unknown_policy_falls_back_to_the_default(self):
        dit = FakeDiT(blocks=52)
        self.assertEqual(
            patch_module.install(dit, self.spec, self.weights, duplicate_policy="nope"), 28
        )

    def test_lineage_changes_the_output_of_a_deeper_model(self):
        """주입이 실제로 줄었는지 — 같은 가중치로 all 과 lineage 의 출력이 달라야 한다."""
        spec = _spec(num_blocks=28)
        weights = _weights(spec)
        for key in list(weights):
            if key.endswith("adaln_ip.1.bias"):
                weights[key] = torch.full((INNER,), 0.5)
        inputs = _inputs()
        tokens = torch.randn(1, TOKENS, EMBED)
        outs = {}
        for policy in ("all", "lineage"):
            torch.manual_seed(1)
            dit = FakeDiT(blocks=52)
            patch_module.install(dit, spec, weights, duplicate_policy=policy)
            outs[policy] = _run(dit, tokens=tokens, inputs=inputs)
        self.assertFalse(torch.allclose(outs["all"], outs["lineage"]))


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


class _SharedModelPatcher:
    """Forge ModelPatcher 처럼 복제끼리 모델 객체와 object_patches_backup 을 공유하고, patch_model 이 객체
    패치를 모델에 실제로 적용하며, 다른 복제로 바꿀 때 detach(unpatch_all=False) 는 되돌리지 않는다."""

    def __init__(self, model, backup=None):
        self.model = model
        self.object_patches = {}
        self.object_patches_backup = {} if backup is None else backup
        self.wrapper = None
        self.post_cfg = None

    def clone(self):
        return _SharedModelPatcher(self.model, self.object_patches_backup)

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        if name in self.object_patches_backup:
            return self.object_patches_backup[name]
        return getattr(self.model, name)

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def set_model_unet_function_wrapper(self, fn):
        self.wrapper = fn

    def set_model_sampler_post_cfg_function(self, fn):
        self.post_cfg = fn

    def patch_model(self):  # load_models_gpu 가 샘플링 때 부르는 부분
        for key, obj in self.object_patches.items():
            old = getattr(self.model, key)
            setattr(self.model, key, obj)
            self.object_patches_backup.setdefault(key, old)

    def detach(self, unpatch_all=True):  # 다른 복제를 올릴 때 Forge 가 부르는 방식(unpatch_all=False)
        return self.model


class PatchedUnetObjectPatchLeakTests(unittest.TestCase):
    """2026-09-23 GPU 에서 확인한 누수: IPA 뒤 KModel.diffusion_model 이 껍데기로 남아 다음 txt2img 의 LoRA 가 버려짐."""

    def setUp(self):
        torch.manual_seed(0)
        self.spec = _spec()
        self.weights = _weights(self.spec)
        self.injection = patch_module.Injection(
            tokens=torch.randn(1, TOKENS, EMBED),
            null_tokens=torch.zeros(1, TOKENS, EMBED),
        )

    def test_shell_dit_does_not_stay_on_the_shared_model_after_the_session(self):
        dit = FakeDiT()
        kmodel = types.SimpleNamespace(diffusion_model=dit)
        original = _SharedModelPatcher(kmodel)
        model = types.SimpleNamespace(forge_objects_after_applying_lora=types.SimpleNamespace(unet=original))
        inputs = _inputs()
        before = _run(dit, inputs=inputs)

        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            swapped = model.forge_objects_after_applying_lora.unet
            swapped.patch_model()   # 샘플링 동안 Forge 가 껍데기를 공유 모델에 건다
            self.assertIsNot(kmodel.diffusion_model, dit)
        swapped.detach(unpatch_all=False)   # 다음 생성이 원본 패처를 올릴 때 Forge 가 하는 일
        original.patch_model()               # 원본 패처엔 객체 패치가 없다

        self.assertIs(kmodel.diffusion_model, dit, "껍데기 DiT 가 원본 모델에 남으면 다음 LoRA 가 버려진다")
        self.assertEqual(original.object_patches_backup, {})
        self.assertTrue(torch.equal(before, _run(kmodel.diffusion_model, inputs=inputs)))

    def test_nothing_happens_when_the_session_never_loaded(self):
        dit = FakeDiT()
        kmodel = types.SimpleNamespace(diffusion_model=dit)
        model = types.SimpleNamespace(forge_objects_after_applying_lora=types.SimpleNamespace(unet=_SharedModelPatcher(kmodel)))
        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            pass
        self.assertIs(kmodel.diffusion_model, dit)


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

    def _patched_dit(self, model):
        return model.forge_objects_after_applying_lora.unet.object_patches["diffusion_model"]

    def test_injection_survives_a_foreign_instance_forward_on_the_blocks(self):
        """Safe PAG 가 원본 블록에 남긴 인스턴스 forward 는 원본 블록을 돌린다 — 껍데기 훅이 안 불려
        주입이 조용히 빠지던 경우(감사 H2)."""
        dit = FakeDiT()
        calls = []
        for block in dit.blocks:
            block.forward = _foreign_block_wrapper(block, calls)
        model = _FakeSdModel(dit)
        inputs = _inputs()
        before = _run(dit, inputs=inputs)
        self.assertEqual(calls, ["block"] * len(dit.blocks))   # 원본 블록은 래퍼를 거친다
        calls.clear()                                          # 세션 밖 호출이 단언을 채우지 않도록
        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            after = _run(self._patched_dit(model), tokens=self.injection.tokens, inputs=inputs)
        self.assertFalse(torch.allclose(before, after))
        # 껍데기 블록은 클래스 forward 를 쓰므로 블록 수준 래퍼는 IPA 패스에서 적용되지 않는다.
        self.assertEqual(calls, [])
        # 원본 블록의 래퍼는 그대로 남아 있고 원본은 여전히 주입 없이 돈다.
        self.assertTrue(all("forward" in b.__dict__ for b in dit.blocks))
        self.assertTrue(torch.equal(before, _run(dit, tokens=self.injection.tokens, inputs=inputs)))

    def test_injection_survives_a_foreign_instance_forward_on_the_dit(self):
        """Anima 3.8B 런타임이 원본 DiT 에 남긴 forward 래퍼는 원본에 바인딩된 클래스 forward 를 부른다.
        껍데기는 그 체인을 살리되(래퍼가 하는 조건 확장을 잃지 않도록) 그 동안만 껍데기 블록을 돌린다."""
        dit = FakeDiT()
        calls = []
        dit.forward = _foreign_dit_wrapper(dit, calls)
        original_blocks = dit.blocks
        model = _FakeSdModel(dit)
        x, emb, context = _inputs()
        before = dit(x, emb, context, transformer_options={})
        self.assertEqual(calls, ["dit"])
        calls.clear()                                     # 세션 밖 호출이 단언을 채우지 않도록
        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            patched = self._patched_dit(model)
            after = patched(
                x, emb, context, transformer_options={patch_module.TOKENS_KEY: self.injection.tokens}
            )
        self.assertFalse(torch.allclose(before, after))
        self.assertEqual(calls, ["dit"])                  # 껍데기 호출이 다른 확장의 래퍼를 정확히 한 번 거쳤다
        self.assertIs(dit.blocks, original_blocks)        # 호출이 끝나면 원본 blocks 가 돌아온다
        self.assertFalse(any(getattr(b, "_sam3_ip_patched", False) for b in dit.blocks))
        self.assertTrue(torch.equal(before, dit(x, emb, context, transformer_options={})))

    def test_a_foreign_dit_forward_that_raises_still_restores_the_original_blocks(self):
        dit = FakeDiT()
        original_blocks = dit.blocks

        def boom(*args, **kwargs):
            raise ValueError("boom")

        dit.forward = boom
        model = _FakeSdModel(dit)
        x, emb, context = _inputs()
        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            with self.assertRaises(ValueError):
                self._patched_dit(model)(x, emb, context, transformer_options={})
        self.assertIs(dit.blocks, original_blocks)

    def test_a_foreign_dit_forward_removed_mid_session_is_not_called(self):
        """세션 도중 외부 래퍼(NegPiP 등)가 풀리면 낡은 클로저를 부르지 않고 껍데기 클래스 forward 로 돈다."""
        dit = FakeDiT()
        calls = []
        dit.forward = _foreign_dit_wrapper(dit, calls)
        original_blocks = dit.blocks
        model = _FakeSdModel(dit)
        x, emb, context = _inputs()
        before = dit(x, emb, context, transformer_options={})
        calls.clear()
        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            del dit.forward                                   # 외부 확장이 래퍼를 해제했다
            after = self._patched_dit(model)(
                x, emb, context, transformer_options={patch_module.TOKENS_KEY: self.injection.tokens}
            )
        self.assertEqual(calls, [])                           # 해제된 래퍼는 불리지 않는다
        self.assertFalse(torch.allclose(before, after))       # 주입은 그대로 걸린다
        self.assertIs(dit.blocks, original_blocks)

    def test_a_foreign_dit_forward_replaced_mid_session_uses_the_current_one(self):
        dit = FakeDiT()
        stale, current = [], []
        dit.forward = _foreign_dit_wrapper(dit, stale)
        model = _FakeSdModel(dit)
        x, emb, context = _inputs()
        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            dit.forward = _foreign_dit_wrapper(dit, current)  # 다른 래퍼로 갈아 끼웠다
            self._patched_dit(model)(
                x, emb, context, transformer_options={patch_module.TOKENS_KEY: self.injection.tokens}
            )
        self.assertEqual(stale, [])
        self.assertEqual(current, ["dit"])

    def test_a_foreign_dit_forward_added_mid_session_is_followed(self):
        """시작 때 없던 래퍼가 도중에 생겨도 호출 시점의 원본 forward 를 따른다."""
        dit = FakeDiT()
        calls = []
        original_blocks = dit.blocks
        model = _FakeSdModel(dit)
        x, emb, context = _inputs()
        before = dit(x, emb, context, transformer_options={})
        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            dit.forward = _foreign_dit_wrapper(dit, calls)
            after = self._patched_dit(model)(
                x, emb, context, transformer_options={patch_module.TOKENS_KEY: self.injection.tokens}
            )
        self.assertEqual(calls, ["dit"])
        self.assertFalse(torch.allclose(before, after))
        self.assertIs(dit.blocks, original_blocks)

    def test_without_a_foreign_forward_the_shell_output_matches_its_class_forward(self):
        """외부 래퍼가 없으면 결과는 껍데기의 클래스 forward 와 비트 단위로 같다(기본 경로 불변)."""
        dit = FakeDiT()
        model = _FakeSdModel(dit)
        x, emb, context = _inputs()
        options = {patch_module.TOKENS_KEY: self.injection.tokens}
        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            patched = self._patched_dit(model)
            via_call = patched(x, emb, context, transformer_options=options)
            direct = FakeDiT.forward(patched, x, emb, context, transformer_options=options)
        self.assertTrue(torch.equal(via_call, direct))

    def test_the_handler_hint_says_the_shell_blocks_never_ran(self):
        dit = FakeDiT()
        model = _FakeSdModel(dit)
        x, emb, context = _inputs()
        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            swapped = model.forge_objects_after_applying_lora.unet
            args = {
                "input": x, "timestep": torch.zeros(1),
                "c": {"transformer_options": {}}, "cond_or_uncond": [0],
            }
            with self.assertRaises(RuntimeError) as caught:
                swapped.wrapper(lambda xx, t, **c: dit(xx, emb, context, **c), args)
        message = str(caught.exception)
        self.assertIn("한 번도 돌지 않았습니다", message)
        self.assertNotIn("transformer_options", message)

    def test_the_handler_hint_says_the_tokens_never_arrived(self):
        """껍데기 블록은 돌았는데 토큰이 빠진 경우 — 원인은 배선이 아니라 transformer_options 전달이다."""
        dit = FakeDiT()
        model = _FakeSdModel(dit)
        x, emb, context = _inputs()
        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            swapped = model.forge_objects_after_applying_lora.unet
            patched = self._patched_dit(model)
            args = {
                "input": x, "timestep": torch.zeros(1),
                "c": {"transformer_options": {}}, "cond_or_uncond": [0],
            }
            # transformer_options 를 새 딕셔너리로 바꿔 넘기는 중간 래퍼를 흉내 낸다.
            with self.assertRaises(RuntimeError) as caught:
                swapped.wrapper(
                    lambda xx, t, **c: patched(xx, emb, context, transformer_options={}), args
                )
        message = str(caught.exception)
        self.assertIn("transformer_options", message)
        self.assertIn(patch_module.TOKENS_KEY, message)
        self.assertNotIn("한 번도 돌지 않았습니다", message)

    def test_the_handler_stops_when_no_shell_block_injected(self):
        """DiT forward 가 껍데기 블록을 거치지 않으면(주입 0건) 첫 호출에서 바로 멈춘다."""
        dit = FakeDiT()
        model = _FakeSdModel(dit)
        x, emb, context = _inputs()
        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            swapped = model.forge_objects_after_applying_lora.unet
            args = {
                "input": x, "timestep": torch.zeros(1),
                "c": {"transformer_options": {}}, "cond_or_uncond": [0],
            }
            # 껍데기 블록을 돌지 않는 apply_model — 원본 DiT 를 부르는 낡은 래퍼와 같은 상황.
            with self.assertRaises(RuntimeError) as caught:
                swapped.wrapper(lambda xx, t, **c: dit(xx, emb, context, transformer_options={}), args)
            self.assertIn("IP 주입", str(caught.exception))
            # 정상 배선이면 통과한다(검사는 성공할 때까지 매 호출 반복되고 성공하면 멈춘다).
            patched = self._patched_dit(model)
            out = swapped.wrapper(
                lambda xx, t, **c: patched(xx, emb, context, transformer_options=c["transformer_options"]),
                args,
            )
            self.assertEqual(tuple(out.shape), tuple(x.shape))


class Anima38ConnectorWrapperTests(unittest.TestCase):
    """실제 Anima 3.8B 런타임의 DiT forward 래퍼(``Anima3BRuntime._wrap_forward``)가 원본 DiT 에 걸린 채로
    IP-Adapter 세션을 열면, 한 번의 forward 에서 커넥터의 조건 확장과 IP 주입이 **둘 다** 실행된다
    (감사 M10 + H2 — 캐릭터 레퍼런스 IP-Adapter 를 3.8B v2 커넥터 토글과 함께 쓰는 경우).

    런타임은 test_anima38 과 같은 방식으로 Forge 의존만 스텁해 실제 코드를 불러온다. 조건 확장
    (``_expand_v2_context``)만 결정적인 가짜로 바꾼다 — Qwen3.5·커넥터 가중치는 CPU 테스트 밖이다.
    """

    def setUp(self):
        from sam3ext.anima38 import marker
        from test_anima38 import _load_lifecycle_runtime

        torch.manual_seed(0)
        self.spec = _spec()
        self.weights = _weights(self.spec)
        self.injection = patch_module.Injection(
            tokens=torch.randn(1, TOKENS, EMBED),
            null_tokens=torch.zeros(1, TOKENS, EMBED),
        )
        self.runtime = _load_lifecycle_runtime().Anima3BRuntime()
        self.expanded = []

        def expand(rows, timesteps, run_ids, ids=None):
            self.expanded.append(ids)
            return rows + 1.0          # 커넥터가 자리표시 행을 의미 조건으로 바꾸는 것을 흉내 낸다

        self.runtime._expand_v2_context = expand
        self.runtime._v2_models = object()
        self.runtime._v2_active = True     # install() 뒤의 상태 — restore() 는 이 플래그만 끈다
        self.dit = FakeDiT()
        self.runtime._wrap_forward(self.dit)
        self.x, self.emb, context = _inputs()
        self.context = marker.stamp_run_id(context, 0)   # 긍정 프롬프트 줄 = run 0 마커가 새겨진 조건

    def _session(self):
        return patch_module.patched_unet(
            _FakeSdModel(self.dit), self.spec, self.weights, self.injection
        )

    def test_the_connector_wrapper_is_on_the_original_dit(self):
        self.assertIn("forward", self.dit.__dict__)
        self.assertTrue(self.runtime._patched(self.dit.forward))

    def test_one_forward_runs_both_the_connector_and_the_injection(self):
        original_blocks = self.dit.blocks
        connector_only = self.dit(self.x, self.emb, self.context, transformer_options={})
        self.assertEqual(self.expanded, [[0]])
        self.expanded.clear()
        options = {patch_module.TOKENS_KEY: self.injection.tokens}
        model = _FakeSdModel(self.dit)
        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            patched = model.forge_objects_after_applying_lora.unet.object_patches["diffusion_model"]
            both = patched(self.x, self.emb, self.context, transformer_options=options)
            # 기대값: 커넥터가 바꾼 조건(context+1)으로 껍데기(주입된) 블록을 돈 결과와 비트 단위로 같다.
            expected = FakeDiT.forward(
                patched, self.x, self.emb, self.context + 1.0, transformer_options=options
            )
            injected = [bool(getattr(b, "_sam3_ip_applied", False)) for b in patched.blocks]
        self.assertEqual(self.expanded, [[0]], "커넥터의 조건 확장이 IP 세션 안에서도 정확히 한 번 돌았다")
        self.assertEqual(injected, [True] * len(injected), "모든 껍데기 블록이 주입했다")
        self.assertTrue(torch.equal(both, expected))
        self.assertFalse(torch.allclose(both, connector_only), "주입이 결과를 바꾼다")
        self.assertIs(self.dit.blocks, original_blocks)

    def test_the_injection_handler_passes_its_first_call_check(self):
        """샘플러 경로(unet 래퍼 → apply_model → 껍데기 DiT → 커넥터 래퍼 → 껍데기 블록)에서
        InjectionHandler 의 '주입이 실제로 실행됐는가' 검사가 통과한다."""
        model = _FakeSdModel(self.dit)
        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            swapped = model.forge_objects_after_applying_lora.unet
            patched = swapped.object_patches["diffusion_model"]
            args = {
                "input": self.x, "timestep": torch.zeros(1),
                "c": {"transformer_options": {}}, "cond_or_uncond": [0],
            }
            out = swapped.wrapper(
                lambda xx, t, **c: patched(xx, self.emb, self.context, **c), args
            )
        self.assertEqual(tuple(out.shape), tuple(self.x.shape))
        self.assertEqual(self.expanded, [[0]])

    def test_after_restore_the_left_wrapper_is_transparent_and_the_injection_still_runs(self):
        """restore() 는 래퍼를 떼지 않고 플래그만 끈다 — 토글을 끈 다음 IP 잡은 커넥터 없이 주입만 된다."""
        self.runtime._v2_active = False
        options = {patch_module.TOKENS_KEY: self.injection.tokens}
        model = _FakeSdModel(self.dit)
        with patch_module.patched_unet(model, self.spec, self.weights, self.injection):
            patched = model.forge_objects_after_applying_lora.unet.object_patches["diffusion_model"]
            out = patched(self.x, self.emb, self.context, transformer_options=options)
            expected = FakeDiT.forward(
                patched, self.x, self.emb, self.context, transformer_options=options
            )
        self.assertEqual(self.expanded, [])
        self.assertTrue(torch.equal(out, expected))


class _ForgeLikePatcher(_FakePatcher):
    """Forge ModelPatcher 의 객체 패치 수명만 흉내 낸다(backend/patcher/base.py): ``object_patches_backup`` 은
    클론끼리 공유하고, 적재(partially_load) 때 먼저 unpatch_model 로 backup 을 되돌린 뒤 자기 패치를 건다.
    다른 클론으로 바뀔 때의 detach(unpatch_all=False) 는 되돌리지 않으므로 여기서도 아무것도 하지 않는다."""

    def __init__(self, model, backup):
        self.model = model
        self.object_patches = {}
        self.object_patches_backup = backup
        self.wrapper = None
        self.post_cfg = None

    def clone(self):
        n = _ForgeLikePatcher(self.model, self.object_patches_backup)
        n.object_patches = dict(self.object_patches)
        return n

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        if name in self.object_patches_backup:
            return self.object_patches_backup[name]
        return getattr(self.model, name)

    def load(self):
        for key, value in list(self.object_patches_backup.items()):
            setattr(self.model, key, value)
        self.object_patches_backup.clear()
        for key, value in self.object_patches.items():
            old = getattr(self.model, key)
            setattr(self.model, key, value)
            self.object_patches_backup.setdefault(key, old)


class Anima38AfterPreviousIpaJobTests(unittest.TestCase):
    """새로 불러온 3.8B 에서 IP 잡(토글 OFF)을 한 번 돌린 뒤 토글 ON 으로 IP 잡을 돌린다 — 검토 반례.
    지난 잡의 껍데기가 ``model.diffusion_model`` 에 남아 있어도 런타임이 진짜 DiT 를 감싸 커넥터 확장이 돈다."""

    def setUp(self):
        from sam3ext.anima38 import marker
        from test_anima38 import _load_lifecycle_runtime

        torch.manual_seed(0)
        self.spec = _spec()
        self.weights = _weights(self.spec)
        self.injection = patch_module.Injection(
            tokens=torch.randn(1, TOKENS, EMBED),
            null_tokens=torch.zeros(1, TOKENS, EMBED),
        )
        self.dit = FakeDiT()
        self.kmodel = types.SimpleNamespace(diffusion_model=self.dit)
        root = _ForgeLikePatcher(self.kmodel, {})
        self.sd = types.SimpleNamespace(
            forge_objects=types.SimpleNamespace(unet=root.clone()),
            forge_objects_after_applying_lora=types.SimpleNamespace(unet=root.clone()),
            forge_objects_original=types.SimpleNamespace(unet=root),
        )
        self.x, self.emb, context = _inputs()
        self.context = marker.stamp_run_id(context, 0)
        self.runtime = _load_lifecycle_runtime().Anima3BRuntime()
        self.expanded = []

        def expand(rows, timesteps, run_ids, ids=None):
            self.expanded.append(ids)
            return rows + 1.0

        self.runtime._expand_v2_context = expand
        self.runtime._v2_models = object()

    def _ipa_job(self):
        options = {patch_module.TOKENS_KEY: self.injection.tokens}
        with patch_module.patched_unet(self.sd, self.spec, self.weights, self.injection):
            swapped = self.sd.forge_objects_after_applying_lora.unet
            swapped.load()   # sample(): 적재 → model.diffusion_model 이 이번 잡의 껍데기
            shell = self.kmodel.diffusion_model
            out = shell(self.x, self.emb, self.context, transformer_options=options)   # apply_model 이 부르는 것
            injected = [bool(getattr(b, "_sam3_ip_applied", False)) for b in shell.blocks]
        return out, shell, injected

    def test_the_connector_runs_in_the_ipa_job_after_a_plain_one(self):
        _, first_shell, _ = self._ipa_job()
        # 잡이 끝나면 patched_unet 이 자기 객체 패치를 되돌린다 — 껍데기가 원본 KModel 에 남지 않는다
        # (남으면 다음 txt2img 의 LoRA 가 [LORA] Mismatch 로 버려졌다, 2026-09-23 GPU 확인).
        self.assertIs(self.kmodel.diffusion_model, self.dit)
        self.assertEqual(self.expanded, [])

        # 토글 ON: _install_v2 / ensure_attached 가 감쌀 대상
        target = self.runtime._real_dit(self.sd)
        self.assertIs(target, self.dit)
        self.runtime._v2_active = True
        self.runtime._wrap_forward(target)

        out, shell, injected = self._ipa_job()
        self.assertIsNot(shell, first_shell)
        self.assertEqual(self.expanded, [[0]], "커넥터 조건 확장이 정확히 한 번")
        self.assertEqual(injected, [True] * len(injected))
        options = {patch_module.TOKENS_KEY: self.injection.tokens}
        expected = FakeDiT.forward(shell, self.x, self.emb, self.context + 1.0, transformer_options=options)
        self.assertTrue(torch.equal(out, expected))

    def test_without_a_leftover_shell_the_target_is_the_model_attribute(self):
        self.assertIs(self.runtime._real_dit(self.sd), self.dit)
        self.sd.forge_objects.unet.load()   # 객체 패치 없는 적재(txt2img) — backup 은 비어 있다
        self.assertIs(self.runtime._real_dit(self.sd), self.dit)


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
