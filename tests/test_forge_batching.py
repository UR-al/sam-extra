"""sam3ext/forge_batching.py — Forge's "Batch Cond/Uncond", read the same way on Forge Neo 2.29.2 and 2.30.

* The reader: 2.30's property over opts (``modules/shared_items.py``), 2.29.2's constant ``True``, and anything
  unreadable counts as on.
* What the docs say each Forge does, against Forge's own ``backend/sampling/sampling_function.py`` pinned with
  ``git show`` (read-only; skipped without a Forge git repository, as on GitHub CI): 2.29.2 has no setting, batches
  cond and uncond only *when memory allows* (a short free memory still splits them) and always reserves 2 × batch
  rows; 2.30 reads the setting in ``can_concat_cond`` (off: one call each, cond first) and ``sampling_prepare``
  (off: 1 × batch rows). ``unet_rows_per_image`` must match the rows ``sampling_prepare`` reserves.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import forge_batching  # noqa: E402
from tests._extra_samplers_fixtures import FORGE_2_29_2, FORGE_2_30, forge_git_source  # noqa: E402


class _Shared230:
    """modules/shared_items.py Shared.batch_cond_uncond of Forge 2.30: a property over opts."""

    def __init__(self, opts):
        self.opts = opts

    @property
    def batch_cond_uncond(self):
        return self.opts.batch_cond_uncond


class _SharedNotLoaded:
    @property
    def batch_cond_uncond(self):
        raise AttributeError("batch_cond_uncond")      # opts before the settings load


class ReaderTests(unittest.TestCase):
    def test_both_forge_versions_and_unreadable_values(self):
        cases = (
            ("2.30, on (default)", _Shared230(types.SimpleNamespace(batch_cond_uncond=True)), True, 2),
            ("2.30, off", _Shared230(types.SimpleNamespace(batch_cond_uncond=False)), False, 1),
            ("2.29.2 constant", types.SimpleNamespace(batch_cond_uncond=True), True, 2),
            ("no such attribute", types.SimpleNamespace(), True, 2),
            ("settings not loaded", _SharedNotLoaded(), True, 2),
        )
        for name, shared, batched, rows in cases:
            with self.subTest(name):
                self.assertIs(forge_batching.batch_cond_uncond(shared), batched)
                self.assertEqual(forge_batching.unet_rows_per_image(shared), rows)


def _pinned_sampler(rev, batch_cond_uncond):
    """Forge's sampling_function.py (and condition.py) as of ``rev``, with only host services stubbed."""
    conditions = types.ModuleType("_forge_batching_conditions")
    exec(compile(forge_git_source(rev, "backend/sampling/condition.py"), f"{rev}:backend/sampling/condition.py",
                 "exec"), conditions.__dict__)
    memory = types.SimpleNamespace(signal_empty_cache=False, get_free_memory=lambda _device: 100,
                                   load_models_gpu=lambda **kwargs: None)
    backend = types.ModuleType("backend")
    backend.memory_management = memory
    backend.utils = types.SimpleNamespace()
    fake_args = types.ModuleType("backend.args")
    fake_args.args = types.SimpleNamespace(disable_gpu_warning=True)
    fake_args.dynamic_args = types.SimpleNamespace(context_handler=None)
    shared = types.ModuleType("modules.shared")
    shared.batch_cond_uncond = bool(batch_cond_uncond)
    modules = types.ModuleType("modules")
    modules.__path__ = []
    modules.shared = shared
    sampler = types.ModuleType("_forge_batching_sampler")
    with mock.patch.dict(sys.modules, {
        "backend": backend, "backend.args": fake_args, "backend.sampling.condition": conditions,
        "modules": modules, "modules.shared": shared,
    }):
        exec(compile(forge_git_source(rev, "backend/sampling/sampling_function.py"),
                     f"{rev}:backend/sampling/sampling_function.py", "exec"), sampler.__dict__)
    return sampler, conditions.Condition, memory, shared


class _Model:
    def __init__(self):
        self.calls = []        # (rows, biases) of each UNet call: cond is 1, uncond 2

    @staticmethod
    def memory_required(shape):
        return shape[0]

    def apply_model(self, x, timestep, *, bias, transformer_options):
        self.calls.append((int(x.shape[0]), [round(float(b)) for b in bias.flatten()]))
        return bias.expand_as(x).clone()


class _Unet:
    """The parts of UnetPatcher sampling_prepare touches; records the shape it reserves memory for."""

    extra_preserved_memory_during_sampling = 0
    controlnet_linked_list = None

    def __init__(self):
        self.reserved = None
        self.extra_model_patchers_during_sampling = []
        self.model = types.SimpleNamespace(predictor=types.SimpleNamespace(percent_to_sigma=lambda p: p))

    def memory_required(self, shape):
        self.reserved = list(shape)
        return 0

    @staticmethod
    def has_online_lora():
        return False

    @staticmethod
    def list_controlnets():
        return []


BATCHED = [(2, [2, 1])]                    # one call, uncond rows first
LOW_VRAM_SPLIT = [(1, [2]), (1, [1])]       # free memory too short: uncond, then cond
SETTING_OFF = [(1, [1]), (1, [2])]          # 2.30 Batch Cond/Uncond off: cond, then uncond, whatever the memory


class PinnedForgeTests(unittest.TestCase):
    def _calls(self, rev, batch_cond_uncond, free_memory):
        sampler, Condition, memory, _shared = _pinned_sampler(rev, batch_cond_uncond)
        memory.get_free_memory = lambda _device: free_memory
        model = _Model()
        cond = [{"model_conds": {"bias": Condition(torch.full((1, 1, 1, 1), 1.0))}}]
        uncond = [{"model_conds": {"bias": Condition(torch.full((1, 1, 1, 1), 2.0))}}]
        with mock.patch.dict(sys.modules, {"backend.sampling.sampling_function": sampler}):
            sampler.sampling_function_inner(model, torch.zeros(1, 1, 2, 2), torch.ones(1), uncond, cond, 4.0, {})
        return model.calls

    def _reserved_rows(self, rev, batch_cond_uncond, batch=3):
        sampler, _Condition, _memory, _shared = _pinned_sampler(rev, batch_cond_uncond)
        unet = _Unet()
        sampler.sampling_prepare(unet, torch.zeros(batch, 4, 8, 8))
        self.assertEqual(unet.reserved[1:], [4, 8, 8])
        return unet.reserved[0] // batch

    def test_forge_2_29_2_batches_only_when_memory_allows_and_reserves_two_rows(self):
        """No setting to read: the flag stub changes nothing; a short free memory still splits the call."""
        for flag in (True, False):
            with self.subTest(stub_flag=flag):
                self.assertEqual(self._calls(FORGE_2_29_2, flag, free_memory=100), BATCHED)
                self.assertEqual(self._calls(FORGE_2_29_2, flag, free_memory=0.5), LOW_VRAM_SPLIT)
                self.assertEqual(self._reserved_rows(FORGE_2_29_2, flag), 2)
        self.assertEqual(forge_batching.unet_rows_per_image(types.SimpleNamespace(batch_cond_uncond=True)), 2)

    def test_forge_2_30_reads_the_setting(self):
        for flag, rows in ((True, 2), (False, 1)):
            with self.subTest(batch_cond_uncond=flag):
                shared = _Shared230(types.SimpleNamespace(batch_cond_uncond=flag))
                self.assertEqual(self._reserved_rows(FORGE_2_30, flag), rows)
                self.assertEqual(forge_batching.unet_rows_per_image(shared), rows)
        self.assertEqual(self._calls(FORGE_2_30, True, free_memory=100), BATCHED)
        self.assertEqual(self._calls(FORGE_2_30, True, free_memory=0.5), LOW_VRAM_SPLIT)
        for free_memory in (100, 0.5):
            self.assertEqual(self._calls(FORGE_2_30, False, free_memory), SETTING_OFF)

    def test_docs_do_not_say_forge_2_29_2_always_batches(self):
        """The docs once said 2.29.2 'always' batches; it batches when memory allows (the test above)."""
        texts = {
            "sam3ext/forge_batching.py": forge_batching.__doc__,
            "docs/GUIDANCE.md": (ROOT / "docs" / "GUIDANCE.md").read_text(encoding="utf-8"),
            "CHANGELOG.md": (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
        }
        for name, text in texts.items():
            with self.subTest(name):
                flat = " ".join(text.split())
                for wrong in ("always batches", "Forge 2.29.2는 늘 한 batch로 호출합니다", "2.29.2 는 늘 묶습니다"):
                    self.assertNotIn(wrong, flat)
        self.assertIn("whenever memory allows", " ".join(forge_batching.__doc__.split()))
        self.assertIn("메모리가 허락하면 늘 한 batch로", " ".join(texts["docs/GUIDANCE.md"].split()))


if __name__ == "__main__":
    unittest.main()
