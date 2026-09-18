from __future__ import annotations

import logging
import re
import types
import unittest
from pathlib import Path

from sam3ext.anima_lora_blocks import (
    ANIMA_29B_BLOCKS,
    ANIMA_38B_BLOCKS,
    ANIMA_BASE_BLOCKS,
    BLOCK_MAPPINGS,
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


class AnimaLoraForgeHookTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
