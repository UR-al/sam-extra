from __future__ import annotations

import logging
import re
import sys
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

from sam3ext import anima_lora_blocks as alb
from sam3ext.anima_lora_blocks import (
    ANIMA_29B_BLOCKS,
    ANIMA_38B_BLOCKS,
    ANIMA_BASE_BLOCKS,
    BLOCK_MAPPINGS,
    DUPLICATE_ADDITIVE,
    DUPLICATE_KEEP,
    DUPLICATE_SKIP,
    DUPLICATE_WEAK,
    anima_lora_block_indices,
    detect_anima_lora_layout,
    install_forge_lora_block_hook,
    remap_anima_lora_for_model,
    uninstall_forge_lora_block_hook,
)


class _Value:
    def __init__(self, lineage: int, clone_number: int = 0):
        self.lineage = lineage
        self.clone_number = clone_number

    def clone(self):
        return _Value(self.lineage, self.clone_number + 1)


def _lora(blocks: int, prefix: str = "lora_unet_blocks_"):
    separator = "_" if prefix.endswith("_") else "."
    return {
        **{
            f"{prefix}{index}{separator}self_attn_q_proj.lora_up.weight": _Value(index)
            for index in range(blocks)
        },
        "metadata": "preserved",
    }


def _sparse_lora(indices, prefix: str = "lora_unet_blocks_"):
    """블록 인덱스 집합만 주어진 부분(sparse) LoRA."""
    separator = "_" if prefix.endswith("_") else "."
    return {
        f"{prefix}{index}{separator}self_attn_q_proj.lora_up.weight": _Value(index)
        for index in indices
    }


def _block_keys(lora, prefix: str = "lora_unet_blocks_"):
    return {key: value for key, value in lora.items() if key.startswith(prefix)}


def _lineages(lora, prefix: str = "lora_unet_blocks_"):
    values = {}
    for key, value in lora.items():
        if not key.startswith(prefix):
            continue
        match = re.match(r"\d+", key[len(prefix) :])
        if match is None:
            continue
        index = int(match.group())
        values[index] = value.lineage
    return [values[index] for index in sorted(values)]


class AnimaLoraBlockMappingTests(unittest.TestCase):
    def test_base_to_29b_matches_forge_builtin_mapping(self):
        expected = (
            0, 1, 1, 2, 3, 3, 4, 5, 5, 6,
            7, 7, 8, 9, 9, 10, 11, 11, 12, 13,
            14, 14, 15, 16, 16, 17, 18, 18, 19, 20,
            20, 21, 22, 22, 23, 24, 24, 25, 26, 27,
        )
        self.assertEqual(
            BLOCK_MAPPINGS[(ANIMA_BASE_BLOCKS, ANIMA_29B_BLOCKS)],
            expected,
        )

    def test_every_directed_layout_pair_has_a_valid_mapping(self):
        layouts = (ANIMA_BASE_BLOCKS, ANIMA_29B_BLOCKS, ANIMA_38B_BLOCKS)
        expected_pairs = {
            (source, target)
            for source in layouts
            for target in layouts
            if source != target
        }
        self.assertEqual(set(BLOCK_MAPPINGS), expected_pairs)
        for (source, target), mapping in BLOCK_MAPPINGS.items():
            self.assertEqual(len(mapping), target)
            self.assertTrue(all(0 <= index < source for index in mapping))

    def test_40_to_52_uses_checkpoint_metadata_insertion_positions(self):
        mapping = BLOCK_MAPPINGS[(ANIMA_29B_BLOCKS, ANIMA_38B_BLOCKS)]
        inserted_to_source = {
            3: 2,
            7: 5,
            11: 8,
            15: 11,
            19: 14,
            23: 17,
            27: 20,
            31: 23,
            35: 26,
            39: 29,
            43: 32,
            47: 35,
        }
        self.assertEqual(
            {target: mapping[target] for target in inserted_to_source},
            inserted_to_source,
        )

    def test_each_expansion_then_contraction_preserves_source_lineage(self):
        for source, expanded in (
            (ANIMA_BASE_BLOCKS, ANIMA_29B_BLOCKS),
            (ANIMA_29B_BLOCKS, ANIMA_38B_BLOCKS),
            (ANIMA_BASE_BLOCKS, ANIMA_38B_BLOCKS),
        ):
            lora = _lora(source)
            up = remap_anima_lora_for_model(lora, expanded)
            down = remap_anima_lora_for_model(lora, source)
            self.assertTrue(up.remapped)
            self.assertEqual(up.duplicated_blocks, expanded - source)
            self.assertTrue(down.remapped)
            self.assertEqual(down.dropped_blocks, expanded - source)
            self.assertEqual(_lineages(lora), list(range(source)))

    def test_base_to_38b_composes_both_generations(self):
        lora = _lora(ANIMA_BASE_BLOCKS)
        result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS)
        self.assertTrue(result.supported)
        self.assertTrue(result.remapped)
        self.assertEqual(result.duplicated_blocks, 24)
        self.assertEqual(anima_lora_block_indices(lora), set(range(52)))
        self.assertEqual(_lineages(lora), list(BLOCK_MAPPINGS[(28, 52)]))
        self.assertEqual(lora["metadata"], "preserved")

    def test_native_diffusion_keys_follow_the_same_mapping(self):
        prefix = "diffusion_model.blocks."
        lora = _lora(ANIMA_29B_BLOCKS, prefix)
        result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS)
        self.assertTrue(result.supported)
        self.assertTrue(result.remapped)
        self.assertEqual(anima_lora_block_indices(lora), set(range(52)))
        self.assertEqual(
            _lineages(lora, prefix),
            list(BLOCK_MAPPINGS[(ANIMA_29B_BLOCKS, ANIMA_38B_BLOCKS)]),
        )

    def test_expansion_clones_duplicate_values_instead_of_aliasing(self):
        lora = _lora(ANIMA_29B_BLOCKS)
        original = lora["lora_unet_blocks_2_self_attn_q_proj.lora_up.weight"]
        remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS)
        first = lora["lora_unet_blocks_2_self_attn_q_proj.lora_up.weight"]
        inserted = lora["lora_unet_blocks_3_self_attn_q_proj.lora_up.weight"]
        self.assertIs(first, original)
        self.assertIsNot(inserted, original)
        self.assertEqual(inserted.lineage, original.lineage)

    def test_sparse_blocks_stay_while_llm_namespace_is_migrated(self):
        lora = _lora(ANIMA_BASE_BLOCKS)
        del lora["lora_unet_blocks_12_self_attn_q_proj.lora_up.weight"]
        lora["lora_unet_anima_v2_connector_test.weight"] = _Value(500)
        lora["lora_unet_llm_adapter_blocks_0_x"] = _Value(501)
        before_blocks = {
            key: value
            for key, value in lora.items()
            if key.startswith("lora_unet_blocks_")
        }
        adapter_value = lora["lora_unet_llm_adapter_blocks_0_x"]
        result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS)
        self.assertFalse(result.supported)
        self.assertFalse(result.remapped)
        self.assertEqual(result.moved_adapter_keys, 1)
        self.assertEqual(
            {
                key: value
                for key, value in lora.items()
                if key.startswith("lora_unet_blocks_")
            },
            before_blocks,
        )
        self.assertNotIn("lora_unet_llm_adapter_blocks_0_x", lora)
        self.assertIs(lora["lora_te_llm_adapter_blocks_0_x"], adapter_value)
        self.assertIn("lora_unet_anima_v2_connector_test.weight", lora)
        self.assertIsNone(detect_anima_lora_layout(lora))

    def test_connector_only_projection_is_rejected_without_emptying_lora(self):
        key = "lora_unet_anima_v2_connector_quality_anchor.weight"
        lora = {key: _Value(500)}
        before = dict(lora)
        result = remap_anima_lora_for_model(lora, ANIMA_BASE_BLOCKS)
        self.assertFalse(result.supported)
        self.assertFalse(result.remapped)
        self.assertEqual(result.dropped_auxiliary_keys, 0)
        self.assertEqual(lora, before)
        self.assertGreater(len(lora), 0)

    def test_connector_keys_are_dropped_if_compatible_keys_remain(self):
        connector = "lora_unet_anima_v2_connector_quality_anchor.weight"
        text_encoder = "lora_te_text_model_encoder_layers_0_x.weight"
        lora = {connector: _Value(500), text_encoder: _Value(501)}
        result = remap_anima_lora_for_model(lora, ANIMA_BASE_BLOCKS)
        self.assertTrue(result.supported)
        self.assertFalse(result.remapped)
        self.assertEqual(result.dropped_auxiliary_keys, 1)
        self.assertNotIn(connector, lora)
        self.assertIn(text_encoder, lora)

    def test_38b_only_keys_are_dropped_when_projecting_down(self):
        lora = _lora(ANIMA_38B_BLOCKS)
        lora["lora_unet_anima_v2_connector_quality_anchor.weight"] = _Value(500)
        lora["lora_te_qwen35_4b_blocks_0.weight"] = _Value(501)
        result = remap_anima_lora_for_model(lora, ANIMA_BASE_BLOCKS)
        self.assertTrue(result.remapped)
        self.assertEqual(result.dropped_auxiliary_keys, 1)
        self.assertFalse(any("anima_v2_connector" in key for key in lora))
        self.assertIn("lora_te_qwen35_4b_blocks_0.weight", lora)
        self.assertEqual(anima_lora_block_indices(lora), set(range(28)))

    def test_38b_only_keys_are_preserved_on_38b(self):
        lora = _lora(ANIMA_38B_BLOCKS)
        key = "lora_unet_anima_v2_connector_quality_anchor.weight"
        lora[key] = _Value(500)
        result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS)
        self.assertFalse(result.remapped)
        self.assertEqual(result.dropped_auxiliary_keys, 0)
        self.assertIn(key, lora)

    def test_same_layout_is_a_noop_and_preserves_tensor_objects(self):
        lora = _lora(ANIMA_29B_BLOCKS)
        before = dict(lora)
        result = remap_anima_lora_for_model(lora, ANIMA_29B_BLOCKS)
        self.assertTrue(result.supported)
        self.assertFalse(result.remapped)
        self.assertEqual(lora, before)
        for key, value in before.items():
            self.assertIs(lora[key], value)

    def test_llm_adapter_namespace_is_migrated(self):
        lora = _lora(ANIMA_BASE_BLOCKS)
        lora["lora_unet_llm_adapter_blocks_0_x"] = _Value(100)
        lora["diffusion_model.llm_adapter.out_proj.weight"] = _Value(101)
        result = remap_anima_lora_for_model(lora, ANIMA_BASE_BLOCKS)
        self.assertEqual(result.moved_adapter_keys, 2)
        self.assertIn("lora_te_llm_adapter_blocks_0_x", lora)
        self.assertIn("text_encoders.qwen3_06b.llm_adapter.out_proj.weight", lora)


# 3.8B 메타데이터의 끼워 넣은 위치(anima_lora_blocks._ANIMA_38B_INSERTED_TO_29B_SOURCE)
_INSERTED_38 = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
_DORA_MODULES = ("self_attn_output_proj", "mlp_layer2")


class AnimaLoraSparseLayoutTests(unittest.TestCase):
    """부분(sparse) LoRA: 순정 Forge 판정(가장 큰 인덱스+1 이 들어가는 가장 작은 레이아웃)과 같으면 그대로 통과."""

    def test_infer_forge_layout_matches_vanilla_rule(self):
        self.assertEqual(alb.infer_forge_lora_layout({0, 1, 20}), ANIMA_BASE_BLOCKS)
        self.assertEqual(alb.infer_forge_lora_layout({27}), ANIMA_BASE_BLOCKS)
        self.assertEqual(alb.infer_forge_lora_layout({28}), ANIMA_29B_BLOCKS)
        self.assertEqual(alb.infer_forge_lora_layout({0, 39}), ANIMA_29B_BLOCKS)
        self.assertEqual(alb.infer_forge_lora_layout({40}), ANIMA_38B_BLOCKS)
        self.assertEqual(alb.infer_forge_lora_layout({51}), ANIMA_38B_BLOCKS)
        self.assertIsNone(alb.infer_forge_lora_layout({52}))
        self.assertIsNone(alb.infer_forge_lora_layout(set()))

    def test_sparse_lora_with_matching_block_count_passes_through(self):
        cases = (
            ("2.9B minus block 12", [i for i in range(40) if i != 12], ANIMA_29B_BLOCKS),
            ("3.8B inherited blocks only", [i for i in range(52) if i not in _INSERTED_38], ANIMA_38B_BLOCKS),
            ("Base with block 5 pruned", [i for i in range(28) if i != 5], ANIMA_BASE_BLOCKS),
            ("prefix 0..20 on Base", list(range(21)), ANIMA_BASE_BLOCKS),
            ("native keys, 2.9B minus block 12", [i for i in range(40) if i != 12], ANIMA_29B_BLOCKS, "diffusion_model.blocks."),
        )
        for name, indices, target, *rest in cases:
            prefix = rest[0] if rest else "lora_unet_blocks_"
            with self.subTest(case=name):
                lora = _sparse_lora(indices, prefix)
                lora["lora_unet_llm_adapter_blocks_0_x"] = _Value(501)
                before = _block_keys(lora, prefix)
                result = remap_anima_lora_for_model(lora, target)
                self.assertTrue(result.supported)
                self.assertFalse(result.remapped)
                self.assertIsNone(result.source_blocks)
                self.assertEqual(result.inferred_blocks, target)
                self.assertEqual(result.moved_adapter_keys, 1)
                self.assertEqual(_block_keys(lora, prefix), before)
                for key, value in before.items():
                    self.assertIs(lora[key], value)
                self.assertIn("lora_te_llm_adapter_blocks_0_x", lora)

    def test_prefix_lora_is_neither_passed_through_nor_guessed_for_a_larger_model(self):
        # 순정은 앞 21블록만 있는 LoRA 를 28블록으로 보고 40블록으로 '추측 변환'한다. 같은 블록 수 케이스와 달리
        # 그대로 올리면 순정과도 다른 의미가 되므로, 변환도 통과도 하지 않는다.
        lora = _sparse_lora(range(21))
        lora["metadata"] = "preserved"
        before = dict(lora)
        result = remap_anima_lora_for_model(lora, ANIMA_29B_BLOCKS)
        self.assertFalse(result.supported)
        self.assertFalse(result.remapped)
        self.assertEqual(result.inferred_blocks, ANIMA_BASE_BLOCKS)
        self.assertEqual(lora, before)

    def test_sparse_lora_inferred_larger_than_model_is_refused(self):
        lora = _sparse_lora(i for i in range(52) if i not in _INSERTED_38)
        before = dict(lora)
        result = remap_anima_lora_for_model(lora, ANIMA_29B_BLOCKS)
        self.assertFalse(result.supported)
        self.assertEqual(result.inferred_blocks, ANIMA_38B_BLOCKS)
        self.assertEqual(lora, before)
        beyond = _sparse_lora([0, 60])
        result = remap_anima_lora_for_model(beyond, ANIMA_38B_BLOCKS)
        self.assertFalse(result.supported)
        self.assertIsNone(result.inferred_blocks)

    def test_sparse_pass_through_handles_connector_keys_like_a_complete_layout(self):
        connector = "lora_unet_anima_v2_connector_quality_anchor.weight"
        lora = _sparse_lora(i for i in range(40) if i != 12)
        lora[connector] = _Value(500)
        result = remap_anima_lora_for_model(lora, ANIMA_29B_BLOCKS)
        self.assertTrue(result.supported)
        self.assertEqual(result.dropped_auxiliary_keys, 1)
        self.assertNotIn(connector, lora)
        lora = _sparse_lora(i for i in range(52) if i not in _INSERTED_38)
        lora[connector] = _Value(500)
        result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS)
        self.assertTrue(result.supported)
        self.assertEqual(result.dropped_auxiliary_keys, 0)
        self.assertIn(connector, lora)

    def test_complete_layouts_still_report_their_source_blocks(self):
        lora = _lora(ANIMA_29B_BLOCKS)
        result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS)
        self.assertEqual((result.source_blocks, result.inferred_blocks), (ANIMA_29B_BLOCKS, ANIMA_29B_BLOCKS))
        self.assertTrue(result.remapped)
        self.assertFalse(result.guessed)

    def test_sparse_kind_splits_prefix_gap_and_unknown(self):
        self.assertEqual(alb.sparse_lora_kind(set(range(21))), alb.SPARSE_PREFIX)
        self.assertEqual(alb.sparse_lora_kind({i for i in range(40) if i != 12}), alb.SPARSE_GAP)
        self.assertEqual(alb.sparse_lora_kind({3, 4, 5}), alb.SPARSE_GAP)  # 0 부터가 아니면 접두가 아니다
        self.assertEqual(alb.sparse_lora_kind({0, 60}), alb.SPARSE_UNKNOWN)
        self.assertEqual(alb.sparse_lora_kind(set(range(40))), alb.SPARSE_PREFIX)  # 호출자는 완전한 레이아웃을 먼저 거른다


def _targets_from(mapping, present):
    """순정 Forge 의 역매핑(source → targets)과 같은 결과: 원본이 있는 타깃 번호와 그 원본."""
    return {target: source for target, source in enumerate(mapping) if source in present}


class AnimaLoraSparseForgeGuessTests(unittest.TestCase):
    """토글(sparse_forge_guess=True): 순정 판정 레이아웃 → 현재 모델로 이 확장의 대응표로 추측 변환."""

    def tearDown(self):
        alb.set_duplicate_policy(DUPLICATE_KEEP)
        alb.set_weak_copy(alb.DEFAULT_WEAK_STRENGTH, alb.WEAK_SCOPE_ATTN)

    def test_default_is_off_and_keeps_refusing(self):
        lora = _sparse_lora(range(21))
        before = dict(lora)
        result = remap_anima_lora_for_model(lora, ANIMA_29B_BLOCKS)
        self.assertFalse(result.supported)
        self.assertFalse(result.guessed)
        self.assertEqual(lora, before)

    def test_prefix_is_expanded_like_vanilla_forge(self):
        # 순정: 앞 21블록 LoRA 를 28블록으로 보고 MAPPING_2_TO_29 로 40블록에 복제한다(_ANIMA_28_TO_40 과 같은 표).
        lora = _sparse_lora(range(21))
        result = remap_anima_lora_for_model(lora, ANIMA_29B_BLOCKS, sparse_forge_guess=True)
        self.assertTrue(result.supported)
        self.assertTrue(result.remapped)
        self.assertTrue(result.guessed)
        self.assertEqual((result.source_blocks, result.inferred_blocks), (ANIMA_BASE_BLOCKS, ANIMA_BASE_BLOCKS))
        expected = _targets_from(BLOCK_MAPPINGS[(ANIMA_BASE_BLOCKS, ANIMA_29B_BLOCKS)], set(range(21)))
        self.assertEqual(anima_lora_block_indices(lora), set(expected))
        self.assertEqual(anima_lora_block_indices(lora), set(range(31)))
        for target, source in expected.items():
            self.assertEqual(lora[f"lora_unet_blocks_{target}_self_attn_q_proj.lora_up.weight"].lineage, source)

    def test_gap_layout_expands_without_inventing_missing_sources(self):
        present = {i for i in range(40) if i != 12}
        lora = _sparse_lora(present, "diffusion_model.blocks.")
        result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS, sparse_forge_guess=True)
        self.assertTrue(result.guessed)
        self.assertEqual(result.source_blocks, ANIMA_29B_BLOCKS)
        expected = _targets_from(BLOCK_MAPPINGS[(ANIMA_29B_BLOCKS, ANIMA_38B_BLOCKS)], present)
        self.assertEqual(anima_lora_block_indices(lora), set(expected))
        self.assertEqual(len(expected), 51)  # 원본 12 의 자리(3.8B 16번) 하나만 빈다 — 12 는 복제 원본이 아니다
        self.assertNotIn(16, expected)

    def test_downward_guess_uses_the_contraction_table(self):
        # 순정은 큰 레이아웃 → 작은 모델을 거부하지만, 이 확장의 대응표는 하향도 안다. 3.8B 에서 계보 블록만 담은
        # LoRA 는 52→40 축소가 정확히 2.9B 40블록이 된다.
        present = {i for i in range(52) if i not in _INSERTED_38}
        lora = _sparse_lora(present)
        result = remap_anima_lora_for_model(lora, ANIMA_29B_BLOCKS, sparse_forge_guess=True)
        self.assertTrue(result.guessed)
        self.assertEqual((result.source_blocks, result.target_blocks), (ANIMA_38B_BLOCKS, ANIMA_29B_BLOCKS))
        self.assertEqual(result.dropped_blocks, 12)
        self.assertEqual(anima_lora_block_indices(lora), set(range(40)))
        self.assertEqual(_lineages(lora), list(BLOCK_MAPPINGS[(ANIMA_38B_BLOCKS, ANIMA_29B_BLOCKS)]))

    def test_inserted_block_policy_and_weak_copy_still_apply(self):
        lora = _sparse_lora(range(21))
        result = remap_anima_lora_for_model(lora, ANIMA_29B_BLOCKS, DUPLICATE_SKIP, sparse_forge_guess=True)
        self.assertTrue(result.guessed)
        self.assertEqual(result.duplicate_policy, DUPLICATE_SKIP)
        mapping = BLOCK_MAPPINGS[(ANIMA_BASE_BLOCKS, ANIMA_29B_BLOCKS)]
        lineage_targets = {mapping.index(source) for source in range(21)}
        self.assertEqual(anima_lora_block_indices(lora), lineage_targets)
        self.assertEqual(result.skipped_duplicate_keys, 10)  # 0~20 안의 끼워 넣은 자리 10개

        import torch

        alb.set_weak_copy(0.1, alb.WEAK_SCOPE_ATTN)
        lora = _tensor_lora(21, {"self_attn_q_proj": ("lora_up.weight", "lora_down.weight")})
        source = {index: _delta(lora, index, "self_attn_q_proj") for index in range(21)}
        result = remap_anima_lora_for_model(lora, ANIMA_29B_BLOCKS, DUPLICATE_WEAK, sparse_forge_guess=True)
        self.assertTrue(result.guessed)
        self.assertEqual(result.weak_scaled_modules, 10)
        self.assertEqual(result.weak_strength, 0.1)
        for target, source_index in _targets_from(mapping, set(range(21))).items():
            scale = 1.0 if target in lineage_targets else 0.1
            torch.testing.assert_close(_delta(lora, target, "self_attn_q_proj"), source[source_index] * scale)

    def test_matching_unknown_and_complete_layouts_ignore_the_toggle(self):
        lora = _sparse_lora(range(21))
        before = dict(lora)
        result = remap_anima_lora_for_model(lora, ANIMA_BASE_BLOCKS, sparse_forge_guess=True)
        self.assertTrue(result.supported)
        self.assertFalse(result.remapped)
        self.assertFalse(result.guessed)
        self.assertEqual(lora, before)

        beyond = _sparse_lora([0, 60])
        before = dict(beyond)
        result = remap_anima_lora_for_model(beyond, ANIMA_38B_BLOCKS, sparse_forge_guess=True)
        self.assertFalse(result.supported)
        self.assertFalse(result.guessed)
        self.assertEqual(beyond, before)

        complete = _lora(ANIMA_BASE_BLOCKS)
        result = remap_anima_lora_for_model(complete, ANIMA_29B_BLOCKS, sparse_forge_guess=True)
        self.assertTrue(result.remapped)
        self.assertFalse(result.guessed)

    def test_unknown_future_target_is_not_guessed(self):
        lora = _sparse_lora(range(21))
        result = remap_anima_lora_for_model(lora, 64, sparse_forge_guess=True)
        self.assertFalse(result.supported)
        self.assertFalse(result.guessed)

    def test_guessed_projection_below_52_drops_connector_keys(self):
        connector = "lora_unet_anima_v2_connector_quality_anchor.weight"
        lora = _sparse_lora(i for i in range(52) if i not in _INSERTED_38)
        lora[connector] = _Value(500)
        result = remap_anima_lora_for_model(lora, ANIMA_BASE_BLOCKS, sparse_forge_guess=True)
        self.assertTrue(result.guessed)
        self.assertEqual(result.dropped_auxiliary_keys, 1)
        self.assertNotIn(connector, lora)
        self.assertEqual(anima_lora_block_indices(lora), set(range(28)))


def _dora_lora(blocks: int):
    """LoKr + DoRA 파일 모양: 모듈마다 lokr_w1 / lokr_w2 / alpha / dora_scale."""
    lora = {}
    for index in range(blocks):
        for module in _DORA_MODULES:
            for param in ("lokr_w1", "lokr_w2", "alpha", "dora_scale"):
                lora[f"lora_unet_blocks_{index}_{module}.{param}"] = _Value(index)
    return lora


def _params_at(lora, index):
    prefix = f"lora_unet_blocks_{index}_"
    return sorted(key[len(prefix):] for key in lora if key.startswith(prefix))


class AnimaLoraInsertedBlockPolicyTests(unittest.TestCase):
    def tearDown(self):
        alb.set_duplicate_policy(DUPLICATE_KEEP)

    def test_keep_is_the_default_and_duplicates_everything(self):
        self.assertEqual(alb.current_duplicate_policy(), DUPLICATE_KEEP)
        lora = _dora_lora(ANIMA_29B_BLOCKS)
        result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS)
        self.assertEqual(result.duplicate_policy, DUPLICATE_KEEP)
        self.assertEqual((result.dropped_dora_keys, result.skipped_duplicate_keys), (0, 0))
        self.assertEqual(_params_at(lora, 3), _params_at(lora, 2))

    def test_additive_drops_only_dora_scale_on_inserted_blocks(self):
        lora = _dora_lora(ANIMA_29B_BLOCKS)
        result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS, DUPLICATE_ADDITIVE)
        self.assertTrue(result.remapped)
        self.assertEqual(result.duplicate_policy, DUPLICATE_ADDITIVE)
        self.assertEqual(result.dropped_dora_keys, len(_INSERTED_38) * len(_DORA_MODULES))
        self.assertEqual(anima_lora_block_indices(lora), set(range(ANIMA_38B_BLOCKS)))
        for index in range(ANIMA_38B_BLOCKS):
            params = _params_at(lora, index)
            with self.subTest(index=index):
                has_dora = any(p.endswith(".dora_scale") for p in params)
                self.assertEqual(has_dora, index not in _INSERTED_38)
                self.assertEqual(sum(p.endswith(".lokr_w2") for p in params), len(_DORA_MODULES))
        # 끼워 넣은 블록의 나머지 인자는 여전히 앞 원본 블록의 복제본이다.
        self.assertEqual(lora["lora_unet_blocks_3_mlp_layer2.lokr_w2"].lineage, 2)
        self.assertIsNot(lora["lora_unet_blocks_3_mlp_layer2.lokr_w2"], lora["lora_unet_blocks_2_mlp_layer2.lokr_w2"])

    def test_skip_leaves_inserted_blocks_without_lora(self):
        lora = _dora_lora(ANIMA_29B_BLOCKS)
        per_block = len(_DORA_MODULES) * 4
        result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS, DUPLICATE_SKIP)
        self.assertEqual(result.skipped_duplicate_keys, len(_INSERTED_38) * per_block)
        self.assertEqual(anima_lora_block_indices(lora), set(range(ANIMA_38B_BLOCKS)) - set(_INSERTED_38))
        self.assertEqual(lora["lora_unet_blocks_4_mlp_layer2.lokr_w2"].lineage, 3)  # 원래 블록은 계보대로

    def test_base_to_38b_skip_covers_both_generations_of_inserts(self):
        lora = _lora(ANIMA_BASE_BLOCKS)
        result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS, DUPLICATE_SKIP)
        self.assertEqual(result.skipped_duplicate_keys, 24)
        self.assertEqual(len(anima_lora_block_indices(lora)), ANIMA_BASE_BLOCKS)

    def test_global_policy_applies_when_none_is_passed(self):
        self.assertTrue(alb.set_duplicate_policy("additive"))
        self.assertFalse(alb.set_duplicate_policy(DUPLICATE_ADDITIVE))
        lora = _dora_lora(ANIMA_29B_BLOCKS)
        result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS)
        self.assertEqual(result.duplicate_policy, DUPLICATE_ADDITIVE)
        self.assertNotIn("lora_unet_blocks_3_mlp_layer2.dora_scale", lora)
        with self.assertRaises(ValueError):
            alb.set_duplicate_policy("nope")

    def test_policy_does_not_touch_downward_or_same_layouts(self):
        alb.set_duplicate_policy(DUPLICATE_SKIP)
        lora = _dora_lora(ANIMA_38B_BLOCKS)
        down = remap_anima_lora_for_model(lora, ANIMA_29B_BLOCKS)
        self.assertEqual((down.dropped_dora_keys, down.skipped_duplicate_keys), (0, 0))
        self.assertEqual(anima_lora_block_indices(lora), set(range(ANIMA_29B_BLOCKS)))
        same = _dora_lora(ANIMA_29B_BLOCKS)
        result = remap_anima_lora_for_model(same, ANIMA_29B_BLOCKS)
        self.assertFalse(result.remapped)
        self.assertEqual(len(same), ANIMA_29B_BLOCKS * len(_DORA_MODULES) * 4)

    def test_normalize_accepts_keys_and_korean_labels(self):
        cases = {
            "keep": DUPLICATE_KEEP, "additive": DUPLICATE_ADDITIVE, "skip": DUPLICATE_SKIP,
            "Inserted: additive": DUPLICATE_ADDITIVE,
            "덧셈형으로 (DoRA 크기 보정 끔)": DUPLICATE_ADDITIVE,
            "덧셈형 (끼워 넣은 블록만 DoRA 크기 보정 끔)": DUPLICATE_ADDITIVE,
            "그대로 복제 (Forge 기본)": DUPLICATE_KEEP,
            "그대로 복제 (순정 · Forge 기본)": DUPLICATE_KEEP,
            "순정": DUPLICATE_KEEP,
            "넣지 않음 (원래 블록에만 · 모든 LoRA)": DUPLICATE_SKIP,
        }
        for value, policy in cases.items():
            with self.subTest(value=value):
                self.assertEqual(alb.normalize_duplicate_policy(value), policy)
        for value in (None, "", "maybe"):
            self.assertIsNone(alb.normalize_duplicate_policy(value))

    def test_remap_log_names_the_policy(self):
        lora = _dora_lora(ANIMA_29B_BLOCKS)
        additive = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS, DUPLICATE_ADDITIVE)
        self.assertIn("24 DoRA magnitudes dropped", alb._format_remap_message(additive))
        lora = _dora_lora(ANIMA_29B_BLOCKS)
        skip = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS, DUPLICATE_SKIP)
        self.assertIn("without LoRA", alb._format_remap_message(skip))


def _tensor_lora(blocks: int, modules: dict[str, tuple[str, ...]]):
    """모듈마다 주어진 인자를 블록 번호로 구분되는 무작위 텐서로 채운다(모양은 인자 종류별로 곱이 맞게)."""
    import torch

    shapes = {
        "lora_up.weight": (8, 2), "lora_down.weight": (2, 6), "lora_B.weight": (8, 2), "lora_A.weight": (2, 6),
        "lokr_w1": (2, 2), "lokr_w2": (4, 3),
        "hada_w1_a": (8, 2), "hada_w1_b": (2, 6), "hada_w2_a": (8, 2), "hada_w2_b": (2, 6),
        "diff": (8, 6), "diff_b": (8,), "alpha": (), "dora_scale": (8, 1),
    }
    generator = torch.Generator().manual_seed(1234)
    lora = {}
    for index in range(blocks):
        for module, params in modules.items():
            for param in params:
                lora[f"lora_unet_blocks_{index}_{module}.{param}"] = (
                    torch.randn(shapes[param], generator=generator) + index
                )
    return lora


def _delta(lora, index, module):
    """kohya 키 모듈 하나의 ΔW(alpha·dora_scale 제외)."""
    import torch

    key = f"lora_unet_blocks_{index}_{module}."
    get = lambda name: lora.get(key + name)  # noqa: E731
    if get("lora_up.weight") is not None:
        return get("lora_up.weight") @ get("lora_down.weight")
    if get("lora_B.weight") is not None:
        return get("lora_B.weight") @ get("lora_A.weight")
    if get("lokr_w1") is not None:
        return torch.kron(get("lokr_w1"), get("lokr_w2"))
    if get("hada_w1_a") is not None:
        return (get("hada_w1_a") @ get("hada_w1_b")) * (get("hada_w2_a") @ get("hada_w2_b"))
    return get("diff")


class AnimaLoraWeakCopyTests(unittest.TestCase):
    """브리지식 약한 복사: 끼워 넣은 블록에 ΔW 를 strength 배로(덧셈형), 범위 안 모듈에만."""

    def setUp(self):
        alb.set_duplicate_policy(DUPLICATE_WEAK)
        alb.set_weak_copy(0.12, "attn")

    def tearDown(self):
        alb.set_duplicate_policy(DUPLICATE_KEEP)
        alb.set_weak_copy(alb.DEFAULT_WEAK_STRENGTH, alb.WEAK_SCOPE_ATTN)

    def test_attn_scope_scales_attention_delta_and_leaves_mlp_out(self):
        import torch

        params = ("lokr_w1", "lokr_w2", "alpha", "dora_scale")
        lora = _tensor_lora(ANIMA_29B_BLOCKS, {"self_attn_output_proj": params, "mlp_layer2": params})
        source = {k: v.clone() for k, v in lora.items()}
        result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS)
        self.assertEqual(result.duplicate_policy, DUPLICATE_WEAK)
        self.assertEqual((result.weak_strength, result.weak_scope), (0.12, "attn"))
        self.assertEqual(result.weak_scaled_modules, len(_INSERTED_38))
        self.assertEqual(result.dropped_dora_keys, len(_INSERTED_38))  # 어텐션 모듈의 dora_scale 만
        self.assertEqual(result.skipped_duplicate_keys, len(_INSERTED_38) * 4)  # mlp 모듈 4개 키
        mapping = BLOCK_MAPPINGS[(ANIMA_29B_BLOCKS, ANIMA_38B_BLOCKS)]
        for target in range(ANIMA_38B_BLOCKS):
            src = mapping[target]
            params_here = _params_at(lora, target)
            with self.subTest(target=target):
                if target in _INSERTED_38:
                    self.assertEqual(params_here, ["self_attn_output_proj.alpha", "self_attn_output_proj.lokr_w1", "self_attn_output_proj.lokr_w2"])
                    torch.testing.assert_close(_delta(lora, target, "self_attn_output_proj"), 0.12 * _delta(source, src, "self_attn_output_proj"))
                    self.assertTrue(torch.equal(lora[f"lora_unet_blocks_{target}_self_attn_output_proj.alpha"], source[f"lora_unet_blocks_{src}_self_attn_output_proj.alpha"]))
                else:
                    self.assertEqual(len(params_here), 8)
                    for module in ("self_attn_output_proj", "mlp_layer2"):
                        self.assertTrue(torch.equal(_delta(lora, target, module), _delta(source, src, module)))
                        self.assertTrue(torch.equal(lora[f"lora_unet_blocks_{target}_{module}.dora_scale"], source[f"lora_unet_blocks_{src}_{module}.dora_scale"]))

    def test_each_adapter_kind_scales_delta_exactly_once(self):
        import torch

        kinds = {
            "self_attn_q_proj": ("lora_up.weight", "lora_down.weight", "alpha"),
            "self_attn_k_proj": ("lora_B.weight", "lora_A.weight"),
            "cross_attn_v_proj": ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "alpha"),
            "cross_attn_output_proj": ("diff", "diff_b"),
            "self_attn_v_proj": ("lokr_w1", "lokr_w2", "alpha", "dora_scale"),
        }
        alb.set_weak_copy(0.3, "attn")
        lora = _tensor_lora(ANIMA_29B_BLOCKS, kinds)
        source = {k: v.clone() for k, v in lora.items()}
        result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS)
        self.assertEqual(result.weak_scaled_modules, len(_INSERTED_38) * len(kinds))
        self.assertEqual(result.skipped_duplicate_keys, 0)
        mapping = BLOCK_MAPPINGS[(ANIMA_29B_BLOCKS, ANIMA_38B_BLOCKS)]
        for target in _INSERTED_38:
            for module in kinds:
                with self.subTest(target=target, module=module):
                    torch.testing.assert_close(_delta(lora, target, module), 0.3 * _delta(source, mapping[target], module))
            torch.testing.assert_close(
                lora[f"lora_unet_blocks_{target}_cross_attn_output_proj.diff_b"],
                0.3 * source[f"lora_unet_blocks_{mapping[target]}_cross_attn_output_proj.diff_b"],
            )
        # 원본 텐서는 건드리지 않는다(복사본만 줄인다).
        self.assertTrue(torch.equal(lora["lora_unet_blocks_2_self_attn_q_proj.lora_up.weight"], source["lora_unet_blocks_2_self_attn_q_proj.lora_up.weight"]))

    def test_scopes_include_mlp_and_modulation_only_when_asked(self):
        modules = {
            "self_attn_q_proj": ("lora_up.weight", "lora_down.weight"),
            "mlp_layer1": ("lora_up.weight", "lora_down.weight"),
            "adaln_modulation_self_attn_1": ("lora_up.weight", "lora_down.weight"),
        }
        expected = {"attn": {"self_attn_q_proj"}, "attn_mlp": {"self_attn_q_proj", "mlp_layer1"}, "all": set(modules)}
        for scope, present in expected.items():
            with self.subTest(scope=scope):
                alb.set_weak_copy(0.12, scope)
                lora = _tensor_lora(ANIMA_29B_BLOCKS, modules)
                result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS)
                self.assertEqual(result.weak_scaled_modules, len(_INSERTED_38) * len(present))
                got = {p.split(".")[0] for p in _params_at(lora, 3)}
                self.assertEqual(got, present)

    def test_base_to_38b_scales_every_insert_once_not_per_generation(self):
        import torch

        lora = _tensor_lora(ANIMA_BASE_BLOCKS, {"self_attn_q_proj": ("lora_up.weight", "lora_down.weight")})
        source = {k: v.clone() for k, v in lora.items()}
        result = remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS)
        self.assertEqual(result.weak_scaled_modules, 24)
        mapping = BLOCK_MAPPINGS[(ANIMA_BASE_BLOCKS, ANIMA_38B_BLOCKS)]
        seen = set()
        for target, src in enumerate(mapping):
            factor = 0.12 if src in seen else 1.0
            seen.add(src)
            with self.subTest(target=target):
                torch.testing.assert_close(_delta(lora, target, "self_attn_q_proj"), factor * _delta(source, src, "self_attn_q_proj"))

    def test_zero_strength_equals_skip_and_one_all_equals_additive(self):
        import torch

        params = ("lokr_w1", "lokr_w2", "alpha", "dora_scale")
        modules = {"self_attn_output_proj": params, "mlp_layer2": params}
        additive = _tensor_lora(ANIMA_29B_BLOCKS, modules)
        remap_anima_lora_for_model(additive, ANIMA_38B_BLOCKS, DUPLICATE_ADDITIVE)
        alb.set_weak_copy(1.0, "all")
        weak = _tensor_lora(ANIMA_29B_BLOCKS, modules)
        remap_anima_lora_for_model(weak, ANIMA_38B_BLOCKS)
        self.assertEqual(sorted(weak), sorted(additive))
        for key in weak:
            self.assertTrue(torch.equal(weak[key], additive[key]), key)
        alb.set_weak_copy(0.0, "attn")
        zero = _tensor_lora(ANIMA_29B_BLOCKS, modules)
        remap_anima_lora_for_model(zero, ANIMA_38B_BLOCKS)
        for target in _INSERTED_38:
            self.assertFalse(bool(_delta(zero, target, "self_attn_output_proj").abs().sum()))

    def test_state_key_parse_and_normalize(self):
        self.assertTrue(alb.set_weak_copy("0.18", "어텐션+MLP"))
        self.assertFalse(alb.set_weak_copy(0.18, "attn_mlp"))
        self.assertEqual(alb.current_weak_copy(), (0.18, "attn_mlp"))
        self.assertEqual(alb.policy_state_key(), "weak 0.18 attn_mlp")
        self.assertEqual(alb.policy_state_key(DUPLICATE_ADDITIVE), DUPLICATE_ADDITIVE)
        self.assertEqual(alb.parse_weak_key("weak 0.18 attn_mlp"), (0.18, "attn_mlp"))
        self.assertEqual(alb.parse_weak_key("weak"), (None, None))
        alb.set_weak_copy(7, "nope")
        self.assertEqual(alb.current_weak_copy(), (2.0, "attn"))  # 0~2 로 자르고 모르는 범위는 어텐션만
        alb.set_weak_copy("x", "전체 (모듈레이션·노름 포함)")
        self.assertEqual(alb.current_weak_copy(), (alb.DEFAULT_WEAK_STRENGTH, "all"))
        for value, policy in {"weak": DUPLICATE_WEAK, "weak 0.12 attn": DUPLICATE_WEAK, "약한 복사 (브리지식 · 강도·범위 조절)": DUPLICATE_WEAK}.items():
            self.assertEqual(alb.normalize_duplicate_policy(value), policy)

    def test_log_names_strength_and_scope(self):
        lora = _tensor_lora(ANIMA_29B_BLOCKS, {"self_attn_q_proj": ("lora_up.weight", "lora_down.weight")})
        message = alb._format_remap_message(remap_anima_lora_for_model(lora, ANIMA_38B_BLOCKS))
        self.assertIn("weak copy x0.12 (attn)", message)
        self.assertIn("12 modules", message)


class AnimaLoraForgeHookTests(unittest.TestCase):
    def setUp(self):
        # 거부 안내(gr.Warning)는 이벤트 밖에서 파이썬 warnings 로 떨어지므로 기록만 한다.
        patcher = mock.patch.object(alb, "_notify_ui")
        self.notify = patcher.start()
        self.addCleanup(patcher.stop)
        # 설정 토글은 기본(끔)으로 고정한다 — 다른 테스트의 modules 스텁에 흔들리지 않게.
        guess = mock.patch.object(alb, "sparse_forge_guess_enabled", return_value=False)
        self.guess = guess.start()
        self.addCleanup(guess.stop)

    def tearDown(self):
        uninstall_forge_lora_block_hook()

    @staticmethod
    def _fake_networks():
        calls = []
        module = types.ModuleType("networks")
        module.__file__ = "C:/forge/extensions-builtin/sd_forge_lora/networks.py"
        module.logger = logging.getLogger("test-anima-lora-blocks")

        def original(lora, blocks):
            calls.append((lora, blocks))

        module.process_anima = original
        module.load_lora_for_models = lambda *args, **kwargs: None
        return module, original, calls

    def test_hook_handles_52_blocks_without_calling_forge_fallback(self):
        module, original, calls = self._fake_networks()
        self.assertTrue(install_forge_lora_block_hook(module))
        self.assertFalse(install_forge_lora_block_hook(module))
        lora = _lora(ANIMA_BASE_BLOCKS)
        result = module.process_anima(lora, ANIMA_38B_BLOCKS)
        self.assertIs(result, True)
        self.assertEqual(calls, [])
        self.assertEqual(anima_lora_block_indices(lora), set(range(52)))
        self.assertTrue(uninstall_forge_lora_block_hook())
        self.assertIs(module.process_anima, original)

    def test_hook_does_not_delegate_sparse_layout_for_known_target(self):
        module, _, calls = self._fake_networks()
        install_forge_lora_block_hook(module)
        lora = _lora(17)
        result = module.process_anima(lora, ANIMA_38B_BLOCKS)
        self.assertIs(result, False)
        self.assertEqual(calls, [])
        self.assertEqual(anima_lora_block_indices(lora), set(range(17)))
        self.assertEqual(self.notify.call_count, 1)

    def test_hook_passes_matching_sparse_layout_to_forge_loader(self):
        module, _, calls = self._fake_networks()
        install_forge_lora_block_hook(module)
        cases = (
            ([i for i in range(40) if i != 12], ANIMA_29B_BLOCKS),
            ([i for i in range(52) if i not in _INSERTED_38], ANIMA_38B_BLOCKS),
            (list(range(21)), ANIMA_BASE_BLOCKS),
        )
        for indices, target in cases:
            with self.subTest(target=target):
                lora = _sparse_lora(indices)
                before = dict(lora)
                result = module.process_anima(lora, target)
                self.assertIs(result, True)
                self.assertEqual(calls, [])
                self.assertEqual(lora, before)
                self.notify.assert_not_called()

    def test_hook_refusals_tell_the_ui_why(self):
        module, _, calls = self._fake_networks()
        install_forge_lora_block_hook(module)

        def load_lora_for_models(lora, blocks, filename="default"):  # noqa: ARG001 — 순정처럼 filename 인자를 가진 호출자
            return module.process_anima(lora, blocks)

        self.assertIs(load_lora_for_models(_sparse_lora(range(21)), ANIMA_29B_BLOCKS, filename="C:\\loras\\prefix21.safetensors"), False)
        message = self.notify.call_args.args[0]
        self.assertIn("'prefix21.safetensors'", message)
        self.assertNotIn("loras", message)
        self.assertIn("ANIMA Base 1.0(28블록)", message)
        self.assertIn("ANIMA 2.9B(40블록)", message)
        self.assertIn("0~20", message)
        # 접두 LoRA 는 2.9B·3.8B 에서 학습한 것일 수도 있어 판정 모델(Base)을 쓰라고 권하면 안 된다.
        self.assertNotIn("그 모델", message)
        self.assertIn("확정할 수 없", message)
        # 접두 갈래: 앞 N블록만 있는 LoRA 라는 것과, 토글로 순정처럼 로드할 수 있다는 안내.
        self.assertIn("앞 21블록", message)
        self.assertIn("접두", message)
        self.assertIn(alb.SPARSE_GUESS_HINT, message)

        self.assertIs(load_lora_for_models(_sparse_lora(i for i in range(52) if i not in _INSERTED_38), ANIMA_29B_BLOCKS, "big.safetensors"), False)
        message = self.notify.call_args.args[0]
        self.assertIn("ANIMA 3.8B(52블록)", message)
        self.assertIn("'big.safetensors'", message)
        # 중간이 빈 갈래: 접두가 아니라 빈 블록이 문제라는 것 + 같은 토글 안내.
        self.assertIn("중간", message)
        self.assertNotIn("접두", message)
        self.assertIn("확정할 수 없", message)
        self.assertIn(alb.SPARSE_GUESS_HINT, message)

        self.assertIs(load_lora_for_models(_sparse_lora([0, 60]), ANIMA_38B_BLOCKS, "odd.safetensors"), False)
        message = self.notify.call_args.args[0]
        # 판정 불가 갈래: 토글로도 로드할 수 없으므로 토글을 권하지 않는다.
        self.assertIn("판정할 수 없", message)
        self.assertIn("60", message)
        self.assertNotIn(alb.SPARSE_GUESS_HINT, message)

        self.assertIs(module.process_anima({"lora_unet_anima_v2_connector_x.weight": _Value(1)}, ANIMA_BASE_BLOCKS), False)
        message = self.notify.call_args.args[0]
        self.assertIn("Semantic Connector", message)
        self.assertTrue(message.startswith("Anima LoRA 건너뜀"), message)  # 호출자에 filename 이 없으면 이름 생략
        self.assertEqual(self.notify.call_count, 4)
        self.assertEqual(calls, [])

    def test_hook_guess_toggle_loads_prefix_like_vanilla_and_informs(self):
        module, _, calls = self._fake_networks()
        install_forge_lora_block_hook(module)
        self.guess.return_value = True

        def load_lora_for_models(lora, blocks, filename="default"):  # noqa: ARG001
            return module.process_anima(lora, blocks)

        lora = _sparse_lora(range(21))
        with self.assertLogs(module.logger, level="INFO") as logs:
            result = load_lora_for_models(lora, ANIMA_29B_BLOCKS, filename="C:/loras/prefix21.safetensors")
        self.assertIs(result, True)
        self.assertEqual(calls, [])
        self.assertEqual(anima_lora_block_indices(lora), set(range(31)))
        warnings_logged = [record for record in logs.records if record.levelno >= logging.WARNING]
        self.assertEqual(len(warnings_logged), 1, logs.output)  # 콘솔 경고는 한 줄
        line = warnings_logged[0].getMessage()
        self.assertNotIn("\n", line)
        self.assertIn("guess", line.lower())
        self.assertIn("28", line)
        self.assertIn("40", line)
        self.assertEqual(self.notify.call_count, 1)
        message = self.notify.call_args.args[0]
        self.assertEqual(self.notify.call_args.kwargs.get("info"), True)  # 정보성 토스트
        self.assertIn("'prefix21.safetensors'", message)
        self.assertIn("추측 변환", message)
        self.assertIn("틀릴 수", message)
        self.assertIn("ANIMA Base 1.0(28블록)", message)
        self.assertIn("ANIMA 2.9B(40블록)", message)

    def test_hook_guess_toggle_still_refuses_undeterminable_layouts(self):
        module, _, calls = self._fake_networks()
        install_forge_lora_block_hook(module)
        self.guess.return_value = True
        lora = _sparse_lora([0, 60])
        before = dict(lora)
        self.assertIs(module.process_anima(lora, ANIMA_38B_BLOCKS), False)
        self.assertEqual(lora, before)
        self.assertEqual(calls, [])
        self.assertIn("판정할 수 없", self.notify.call_args.args[0])

    def test_hook_records_guesses_per_lora_load_for_infotext(self):
        module, _, _ = self._fake_networks()
        install_forge_lora_block_hook(module)
        model = types.SimpleNamespace(current_lora_hash="H1")
        stub = types.ModuleType("modules")
        stub.shared = types.SimpleNamespace(sd_model=model)

        def load_lora_for_models(lora, blocks, filename="default"):  # noqa: ARG001
            return module.process_anima(lora, blocks)

        with mock.patch.dict(sys.modules, {"modules": stub, "modules.shared": stub.shared}):
            self.guess.return_value = True
            load_lora_for_models(_sparse_lora(range(21)), ANIMA_29B_BLOCKS, filename="C:/l/prefix21.safetensors")
            load_lora_for_models(_lora(ANIMA_BASE_BLOCKS), ANIMA_29B_BLOCKS, filename="C:/l/full.safetensors")
            self.assertEqual(alb.sparse_guess_record(model), {"prefix21.safetensors": "28->40"})
            self.assertFalse(alb.sparse_cache_stale(model, True))
            self.assertTrue(alb.sparse_cache_stale(model, False))

            # 같은 LoRA 목록(같은 해시)을 토글을 끄고 다시 합치면 그 이름은 기록에서 빠진다.
            self.guess.return_value = False
            load_lora_for_models(_sparse_lora(range(21)), ANIMA_29B_BLOCKS, filename="C:/l/prefix21.safetensors")
            self.assertEqual(alb.sparse_guess_record(model), {})
            self.assertTrue(alb.sparse_cache_stale(model, True))

            # 다른 LoRA 목록이 합쳐지면(해시가 바뀌면) 옛 기록은 쓰지 않는다.
            model.current_lora_hash = "H2"
            self.assertEqual(alb.sparse_guess_record(model), {})
            self.assertFalse(alb.sparse_cache_stale(model, True))

    def test_matching_and_complete_loras_leave_no_sparse_record(self):
        module, _, _ = self._fake_networks()
        install_forge_lora_block_hook(module)
        model = types.SimpleNamespace(current_lora_hash="H1")
        stub = types.ModuleType("modules")
        stub.shared = types.SimpleNamespace(sd_model=model)
        with mock.patch.dict(sys.modules, {"modules": stub, "modules.shared": stub.shared}):
            module.process_anima(_sparse_lora(range(21)), ANIMA_BASE_BLOCKS)
            module.process_anima(_lora(ANIMA_29B_BLOCKS), ANIMA_38B_BLOCKS)
        self.assertFalse(hasattr(model, alb.SPARSE_STATE_ATTR))

    def test_hook_rollback_of_future_target_tells_the_ui(self):
        module, _, _ = self._fake_networks()

        def unsafe_original(lora, blocks):
            raise IndexError(blocks)

        module.process_anima = unsafe_original
        install_forge_lora_block_hook(module)
        self.assertIs(module.process_anima(_lora(ANIMA_BASE_BLOCKS), 64), False)
        self.assertIn("64블록", self.notify.call_args.args[0])

    def test_hook_keeps_connector_only_projection_nonempty(self):
        module, _, calls = self._fake_networks()
        install_forge_lora_block_hook(module)
        key = "lora_unet_anima_v2_connector_quality_anchor.weight"
        lora = {key: _Value(500)}
        result = module.process_anima(lora, ANIMA_BASE_BLOCKS)
        self.assertIs(result, False)
        self.assertEqual(calls, [])
        self.assertEqual(set(lora), {key})

    def test_hook_delegates_future_target_to_forge(self):
        module, _, calls = self._fake_networks()
        install_forge_lora_block_hook(module)
        lora = _lora(ANIMA_BASE_BLOCKS)
        module.process_anima(lora, 64)
        self.assertEqual(calls, [(lora, 64)])

    def test_failing_future_forge_fallback_is_rolled_back(self):
        module, _, _ = self._fake_networks()

        def unsafe_original(lora, blocks):
            lora["partially_written"] = _Value(999)
            raise IndexError(blocks)

        module.process_anima = unsafe_original
        install_forge_lora_block_hook(module)
        lora = _lora(ANIMA_BASE_BLOCKS)
        before = dict(lora)
        module.process_anima(lora, 64)
        self.assertEqual(lora, before)

    def test_loader_is_ordered_after_forge_lora_extension(self):
        root = Path(__file__).resolve().parents[1]
        metadata = (root / "metadata.ini").read_text(encoding="utf-8")
        self.assertIn("[scripts/anima_lora_blocks.py]", metadata)
        self.assertIn("After = sd_forge_lora", metadata)

    def test_retry_does_not_nest_above_an_external_wrapper(self):
        module, _, _ = self._fake_networks()
        self.assertTrue(install_forge_lora_block_hook(module))
        sam3_wrapper = module.process_anima

        def external_wrapper(lora, blocks):
            return sam3_wrapper(lora, blocks)

        module.process_anima = external_wrapper
        self.assertFalse(install_forge_lora_block_hook(module))
        self.assertIs(module.process_anima, external_wrapper)

        # Restore the top-level SAM3 callable so tearDown can detach it.
        module.process_anima = sam3_wrapper

    def test_known_module_survives_temporary_sys_modules_hiding(self):
        module, original, _ = self._fake_networks()
        self.assertTrue(install_forge_lora_block_hook(module))
        self.assertTrue(uninstall_forge_lora_block_hook())
        self.assertIs(module.process_anima, original)

        previous = __import__("sys").modules.pop("networks", None)
        try:
            self.assertTrue(install_forge_lora_block_hook())
            self.assertIsNot(module.process_anima, original)
        finally:
            if previous is not None:
                __import__("sys").modules["networks"] = previous


class AnimaLoraCallerFilenameTests(unittest.TestCase):
    """안내문의 LoRA 이름: 프레임 수가 아니라 filename 인자를 가진 load_lora_for_models 프레임으로 찾는다."""

    def setUp(self):
        patcher = mock.patch.object(alb, "_notify_ui")
        self.notify = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(uninstall_forge_lora_block_hook)

    def test_found_through_many_wrapper_frames(self):
        module, _, _ = AnimaLoraForgeHookTests._fake_networks()
        install_forge_lora_block_hook(module)
        hook = module.process_anima

        def wrap(inner):
            def wrapper(lora, blocks):
                return inner(lora, blocks)
            return wrapper

        wrapped = hook
        for _ in range(8):  # 예전 휴리스틱(3~6번 프레임만)으로는 닿지 않는 깊이
            wrapped = wrap(wrapped)

        def load_lora_for_models(lora, blocks, filename="default"):  # noqa: ARG001
            return wrapped(lora, blocks)

        self.assertIs(load_lora_for_models(_sparse_lora(range(21)), ANIMA_29B_BLOCKS, filename="D:/loras/deep.safetensors"), False)
        self.assertIn("'deep.safetensors'", self.notify.call_args.args[0])

    def test_unrelated_filename_locals_are_ignored(self):
        def other_extension_wrapper(filename="unrelated_config.json"):  # noqa: ARG001 — 이름만 같은 다른 값
            return alb._caller_lora_filename()

        def load_lora_for_models(lora, filename="default"):  # noqa: ARG001
            return other_extension_wrapper()

        self.assertEqual(load_lora_for_models({}, filename="C:/loras/real.safetensors"), "C:/loras/real.safetensors")
        # load_lora_for_models 가 스택에 없으면 filename 지역 변수가 있어도 이름을 추측하지 않는다.
        self.assertEqual(other_extension_wrapper(), "")

    def test_load_lora_frame_without_filename_argument_is_skipped(self):
        def load_lora_for_models(lora):  # noqa: ARG001 — filename 이 인자가 아니라 지역 변수일 뿐
            filename = "local_only.safetensors"  # noqa: F841
            return alb._caller_lora_filename()

        self.assertEqual(load_lora_for_models({}), "")

        def outer_load(lora, filename="default"):  # noqa: ARG001
            return load_lora_for_models(lora)

        outer_load.__code__ = outer_load.__code__.replace(co_name="load_lora_for_models")
        self.assertEqual(outer_load({}, filename="outer.safetensors"), "outer.safetensors")

    def test_empty_or_non_string_filename_gives_empty_name(self):
        def load_lora_for_models(lora, filename="default"):  # noqa: ARG001
            return alb._caller_lora_filename()

        self.assertEqual(load_lora_for_models({}, filename=""), "")
        self.assertEqual(load_lora_for_models({}, filename=None), "")
        self.assertEqual(alb._caller_lora_filename(), "")

    def test_frame_inspection_failure_gives_empty_name(self):
        with mock.patch.object(alb.sys, "_getframe", side_effect=RuntimeError("no frames")):
            self.assertEqual(alb._caller_lora_filename(), "")
        self.assertTrue(alb._skip_notice("사유").startswith("Anima LoRA 건너뜀: 사유"))


class AnimaLoraUiNoticeTests(unittest.TestCase):
    # 실제 gradio 를 import 하지 않는다: 설치 여부·버전·첫 import 경고에 테스트가 흔들리지 않도록 모의 모듈로 바꾼다.

    def test_notify_ui_raises_a_gradio_warning_with_the_message(self):
        fake = types.ModuleType("gradio")
        fake.Warning = mock.Mock()
        with mock.patch.dict(sys.modules, {"gradio": fake}):
            alb._notify_ui("sparse notice")
        fake.Warning.assert_called_once_with("sparse notice")

    def test_notify_ui_swallows_a_failing_gradio_warning(self):
        fake = types.ModuleType("gradio")
        fake.Warning = mock.Mock(side_effect=RuntimeError("no event"))
        with mock.patch.dict(sys.modules, {"gradio": fake}):
            with self.assertLogs(alb.LOGGER, level="DEBUG") as logs:
                alb._notify_ui("sparse notice")
        fake.Warning.assert_called_once_with("sparse notice")
        self.assertTrue(any("could not raise a UI warning" in line for line in logs.output))

    def test_notify_ui_without_gradio_only_logs(self):
        # sys.modules 의 None 항목은 import 를 ImportError 로 막는다(gradio 미설치와 같다).
        with mock.patch.dict(sys.modules, {"gradio": None}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with self.assertLogs(alb.LOGGER, level="DEBUG") as logs:
                    alb._notify_ui("sparse notice")
        self.assertEqual(caught, [])
        self.assertTrue(any("could not raise a UI warning" in line for line in logs.output))

    def test_notify_ui_info_uses_an_info_toast(self):
        fake = types.ModuleType("gradio")
        fake.Warning = mock.Mock()
        fake.Info = mock.Mock()
        with mock.patch.dict(sys.modules, {"gradio": fake}):
            alb._notify_ui("guessed", info=True)
        fake.Info.assert_called_once_with("guessed")
        fake.Warning.assert_not_called()

    def test_describe_indices(self):
        self.assertEqual(alb._describe_indices([]), "없음")
        self.assertEqual(alb._describe_indices(list(range(21))), "0~20")
        self.assertEqual(alb._describe_indices([i for i in range(40) if i != 12]), "0~39 중 39개")


class AnimaLoraSparseGuessOptionTests(unittest.TestCase):
    """설정값 읽기: Forge opts 의 진짜 bool True 일 때만 켠다(스텁·MagicMock·미등록은 끔)."""

    def _read(self, opts):
        stub = types.ModuleType("modules")
        stub.shared = types.SimpleNamespace(opts=opts)
        with mock.patch.dict(sys.modules, {"modules": stub, "modules.shared": stub.shared}):
            return alb.sparse_forge_guess_enabled()

    def test_reads_only_a_real_true(self):
        key = alb.OPT_SPARSE_FORGE_GUESS
        self.assertEqual(key, "sam3_anima_sparse_lora_forge_guess")
        self.assertIs(self._read(types.SimpleNamespace(**{key: True})), True)
        self.assertIs(self._read(types.SimpleNamespace(**{key: False})), False)
        self.assertIs(self._read(types.SimpleNamespace()), False)
        self.assertIs(self._read(mock.MagicMock()), False)

    def test_without_forge_is_off(self):
        with mock.patch.dict(sys.modules, {"modules": None}):
            self.assertIs(alb.sparse_forge_guess_enabled(), False)


def _load_blocks_script():
    """scripts/anima_lora_blocks.py 를 Forge 스텁으로 불러온다(실제 networks·gradio 없이)."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    modules_stub = types.ModuleType("modules")

    class Script:
        pass

    class OptionInfo:
        def __init__(self, default=None, label="", component=None, component_args=None, onchange=None,
                     section=None, refresh=None, comment_before="", comment_after="", infotext=None, **kwargs):
            self.default, self.label, self.component = default, label, component
            self.section, self.infotext, self.onchange, self.comment = section, infotext, onchange, ""

        def info(self, text):
            self.comment = text
            return self

    callbacks = {"before_ui": [], "unloaded": [], "ui_settings": []}
    added = {}
    modules_stub.scripts = types.SimpleNamespace(Script=Script, AlwaysVisible=object())
    modules_stub.script_callbacks = types.SimpleNamespace(
        on_before_ui=callbacks["before_ui"].append,
        on_script_unloaded=callbacks["unloaded"].append,
        on_ui_settings=callbacks["ui_settings"].append,
    )
    modules_stub.shared = types.SimpleNamespace(
        OptionInfo=OptionInfo,
        opts=types.SimpleNamespace(add_option=lambda key, info: added.__setitem__(key, info)),
        sd_model=None,
    )
    gradio_stub = types.ModuleType("gradio")
    gradio_stub.Checkbox = object()
    with mock.patch.object(alb, "install_forge_lora_block_hook", return_value=False):
        with mock.patch.dict(sys.modules, {
            "modules": modules_stub,
            "modules.scripts": modules_stub.scripts,
            "modules.script_callbacks": modules_stub.script_callbacks,
            "modules.shared": modules_stub.shared,
            "gradio": gradio_stub,
        }):
            spec = importlib.util.spec_from_file_location("_test_anima_lora_blocks_script", root / "scripts" / "anima_lora_blocks.py")
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
    return module, modules_stub, gradio_stub, callbacks, added


class AnimaLoraSparseGuessScriptTests(unittest.TestCase):
    """설정 등록·토글 변경 시 LoRA 캐시 무효화·infotext 기록(scripts/anima_lora_blocks.py)."""

    @classmethod
    def setUpClass(cls):
        cls.mod, cls.stub, cls.gradio, cls.callbacks, cls.added = _load_blocks_script()

    def _ctx(self, model, enabled):
        self.stub.shared.sd_model = model
        self.stub.shared.opts.sam3_anima_sparse_lora_forge_guess = enabled
        return mock.patch.dict(sys.modules, {
            "modules": self.stub,
            "modules.shared": self.stub.shared,
            "gradio": self.gradio,
        })

    @staticmethod
    def _p(model):
        return types.SimpleNamespace(sd_model=model, extra_generation_params={})

    def test_registers_the_toggle_in_the_sam_extra_lora_section(self):
        self.assertEqual(len(self.callbacks["ui_settings"]), 1)
        with self._ctx(None, False):
            self.callbacks["ui_settings"][0]()
        info = self.added[alb.OPT_SPARSE_FORGE_GUESS]
        self.assertIs(info.default, False)
        self.assertEqual(info.section, ("sam3_lora", "SAM Extra LoRA"))
        self.assertIs(info.component, self.gradio.Checkbox)
        self.assertEqual(info.infotext, alb.INFOTEXT_SPARSE_GUESS_KEY)
        self.assertIn("틀릴 수", info.label)
        self.assertIn("틀릴 수", info.comment)
        self.assertIn("부분 LoRA 순정 추측 변환", info.label)  # 거부 안내문이 가리키는 이름
        self.assertIn("부분 LoRA 순정 추측 변환", alb.SPARSE_GUESS_HINT)

    def test_script_is_always_visible_without_controls(self):
        script = self.mod.AnimaSparseLoraGuess()
        self.assertIs(script.show(False), self.stub.scripts.AlwaysVisible)
        self.assertEqual(script.ui(False), [])

    def test_process_invalidates_only_when_the_toggle_changed_since_that_load(self):
        model = types.SimpleNamespace(current_lora_hash="H1")
        setattr(model, alb.SPARSE_STATE_ATTR, {"hash": "H1", "enabled": False, "guessed": {}})
        p = self._p(model)
        p.extra_generation_params[alb.INFOTEXT_SPARSE_GUESS_KEY] = "stale XYZ cell"
        with self._ctx(model, False):
            self.mod.AnimaSparseLoraGuess().process(p)
        self.assertEqual(model.current_lora_hash, "H1")
        self.assertNotIn(alb.INFOTEXT_SPARSE_GUESS_KEY, p.extra_generation_params)

        with self._ctx(model, True):
            self.mod.AnimaSparseLoraGuess().process(p)
        self.assertIsNone(model.current_lora_hash)
        self.assertFalse(hasattr(model, alb.SPARSE_STATE_ATTR))

    def test_process_does_not_touch_an_unrelated_lora_set(self):
        model = types.SimpleNamespace(current_lora_hash="H2")
        setattr(model, alb.SPARSE_STATE_ATTR, {"hash": "H1", "enabled": False, "guessed": {}})
        with self._ctx(model, True):
            self.mod.AnimaSparseLoraGuess().process(self._p(model))
        self.assertEqual(model.current_lora_hash, "H2")

    def test_batch_hooks_record_guesses_including_a_hires_lora_set(self):
        model = types.SimpleNamespace(current_lora_hash="H1")
        setattr(model, alb.SPARSE_STATE_ATTR, {"hash": "H1", "enabled": True, "guessed": {"a.safetensors": "28->40"}})
        p = self._p(model)
        script = self.mod.AnimaSparseLoraGuess()
        with self._ctx(model, True):
            script.process(p)
            script.process_batch(p, batch_number=0, prompts=[], seeds=[], subseeds=[])
            self.assertEqual(p.extra_generation_params[alb.INFOTEXT_SPARSE_GUESS_KEY], "Forge guess (a.safetensors 28->40)")
            model.current_lora_hash = "H2"  # hires 의 다른 LoRA 목록
            setattr(model, alb.SPARSE_STATE_ATTR, {"hash": "H2", "enabled": True, "guessed": {"b.safetensors": "40->52"}})
            script.postprocess_batch(p, [], batch_number=0)
        self.assertEqual(
            p.extra_generation_params[alb.INFOTEXT_SPARSE_GUESS_KEY],
            "Forge guess (a.safetensors 28->40, b.safetensors 40->52)",
        )

    def test_no_guess_means_no_infotext(self):
        model = types.SimpleNamespace(current_lora_hash="H1")
        p = self._p(model)
        script = self.mod.AnimaSparseLoraGuess()
        with self._ctx(model, True):
            script.process(p)
            script.process_batch(p, batch_number=0, prompts=[], seeds=[], subseeds=[])
            script.postprocess_batch(p, [], batch_number=0)
        self.assertEqual(p.extra_generation_params, {})


if __name__ == "__main__":
    unittest.main()
