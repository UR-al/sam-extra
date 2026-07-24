"""Focused tests for the current Forge Neo Anima attention-op hook."""

from __future__ import annotations

import unittest

import torch

from test_anima_safe_pag import _load_pag_module


class AnimaAttentionPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _load_pag_module()

    def setUp(self):
        self.pag._STATE.update(
            active=1,
            attn_b0=2,
            attn_b1=3,
            attn_method="pag",
            strength=1.0,
            legacy_attn=False,
            seg_sigma=1.0,
            head_spec="",
            attn_spatial_shape=(1, 2, 2),
            attn_hook_hits=0,
            attn_shape_warned=False,
            any_b0=2,
            attn_targets={0},
            slg_targets=set(),
            slg_b0=None,
            slg_b1=None,
        )

    @staticmethod
    def _inputs(batch=3, seq=4, heads=2, dim=3):
        q = torch.zeros(batch, seq, heads, dim)
        k = torch.zeros_like(q)
        value = torch.arange(
            batch * seq * heads * dim, dtype=torch.float32
        ).reshape(batch, seq, heads, dim)
        return q, k, value

    def test_ensure_patch_preserves_staticmethod_binding(self):
        pag = self.pag

        class DummyAttention:
            @staticmethod
            def torch_attention_op(query, key, value):
                return torch.zeros(
                    value.shape[0],
                    value.shape[1],
                    value.shape[2] * value.shape[3],
                )

            def forward(self, x):
                return x

        class DummyBlock:
            def __init__(self):
                self.self_attn = DummyAttention()

            def forward(self, x):
                return x

        class DummyModel:
            blocks = [DummyBlock()]

        self.assertEqual(pag._ensure_patched(DummyModel()), 1)
        q, k, value = self._inputs()
        out = DummyAttention().torch_attention_op(q, k, value)

        self.assertEqual(tuple(out.shape), (3, 4, 6))
        self.assertEqual(pag._STATE["attn_hook_hits"], 1)

    def test_inactive_fast_path_is_bitwise_unchanged(self):
        pag = self.pag
        q, k, value = self._inputs()
        baseline = value.reshape(3, 4, 6) + 7
        pag._ORIG_ANIMA_ATTN_OP = lambda *_args, **_kwargs: baseline.clone()
        pag._STATE["active"] = 0

        out = pag._patched_anima_attention_op(q, k, value)

        self.assertTrue(torch.equal(out, baseline))
        self.assertEqual(pag._STATE["attn_hook_hits"], 0)

    def test_pag_changes_only_appended_weak_row(self):
        pag = self.pag
        q, k, value = self._inputs()
        baseline = torch.full((3, 4, 6), -1.0)
        pag._ORIG_ANIMA_ATTN_OP = lambda *_args, **_kwargs: baseline.clone()

        out = pag._patched_anima_attention_op(q, k, value)

        self.assertTrue(torch.equal(out[:2], baseline[:2]))
        torch.testing.assert_close(out[2:3], value.reshape(3, 4, 6)[2:3])
        self.assertEqual(pag._STATE["attn_hook_hits"], 1)

    def test_shape_mismatch_falls_back_without_mutation(self):
        pag = self.pag
        q, k, value = self._inputs()
        baseline = torch.randn(3, 4, 7)
        pag._ORIG_ANIMA_ATTN_OP = lambda *_args, **_kwargs: baseline.clone()

        out = pag._patched_anima_attention_op(q, k, value)

        self.assertTrue(torch.equal(out, baseline))
        self.assertEqual(pag._STATE["attn_hook_hits"], 0)

    def test_teardown_restores_all_global_patches(self):
        pag = self.pag
        pag._teardown_global_patches()  # start from a clean base (test-order safe)

        class DummyAttention:
            @staticmethod
            def torch_attention_op(query, key, value):
                return torch.zeros(
                    value.shape[0], value.shape[1], value.shape[2] * value.shape[3]
                )

            def forward(self, x):
                return x

        class DummyBlock:
            def __init__(self):
                self.self_attn = DummyAttention()

            def forward(self, x):
                return x

        block = DummyBlock()
        model = type("DummyModel", (), {"blocks": [block]})()
        orig_attn_op = DummyAttention.torch_attention_op

        self.assertEqual(pag._ensure_patched(model), 1)
        # Patched: class attention op swapped, forwards carry our owner marker.
        self.assertIsNot(DummyAttention.torch_attention_op, orig_attn_op)
        self.assertIs(getattr(block.forward, "_anima_pag_owner", None), pag._PATCH_OWNER)
        self.assertIs(
            getattr(block.self_attn.forward, "_anima_pag_owner", None), pag._PATCH_OWNER
        )

        # A fake CNS target proves that restoration path too, without needing a
        # real k-diffusion install. The stored original is a callable, as the
        # real patch records (teardown only restores callables).
        def _orig_factory(x):
            return None

        fake_sampling = type("FakeSampling", (), {})()
        fake_sampling.default_noise_sampler = "PATCHED"
        fake_sampling._sam_extra_cns_orig_default_noise_sampler = _orig_factory
        pag._PATCHED_CNS_TARGETS.append((fake_sampling, None))

        pag._teardown_global_patches()

        # Attention op restored to the exact original; markers cleared.
        self.assertIs(DummyAttention.torch_attention_op, orig_attn_op)
        self.assertFalse(hasattr(DummyAttention, "_pag_orig_torch_attention_op"))
        # Forwards unwrapped (no owner marker) and behave like the original.
        self.assertIsNone(getattr(block.forward, "_anima_pag_owner", None))
        self.assertEqual(block.forward(5), 5)
        self.assertIsNone(getattr(block.self_attn.forward, "_anima_pag_owner", None))
        # CNS module restored and tracking cleared.
        self.assertIs(fake_sampling.default_noise_sampler, _orig_factory)
        self.assertFalse(
            hasattr(fake_sampling, "_sam_extra_cns_orig_default_noise_sampler")
        )
        self.assertEqual(len(pag._PATCHED_CNS_TARGETS), 0)

        # Re-patching from the cleaned base works (install → teardown idempotent).
        self.assertEqual(pag._ensure_patched(model), 1)
        self.assertIsNot(DummyAttention.torch_attention_op, orig_attn_op)
        pag._teardown_global_patches()
        self.assertIs(DummyAttention.torch_attention_op, orig_attn_op)


if __name__ == "__main__":
    unittest.main()
