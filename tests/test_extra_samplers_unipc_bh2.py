"""``UniPC bh2`` (``sam3ext.extra_samplers.unipc``): Forge's own UniPC with ``variant="bh2"``.

The oracle is Forge Neo's ``modules/sd_samplers_extra.py`` ``sample_unipc`` on Forge's
``modules/uni_pc/uni_pc.py``, executed read-only from Forge's files (``fx.forge_sd_samplers_extra()``).
Registration (the entry is skipped when Forge's ``sample_unipc`` is missing or has no ``variant`` parameter)
is tested in ``test_extra_samplers_registry.py``; the run through Forge's ``KDiffusionSampler.sample`` in
``test_extra_samplers_forge_path.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

import torch


def _fixtures():
    name = "_extra_samplers_fixtures"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


fx = _fixtures()

from sam3ext.extra_samplers import registry, unipc  # noqa: E402

STEPS = (3, 10, 28)
# Forge's UniPC builds its coefficients as float32 tensors (uni_pc.py ``torch.tensor(rks, device=x.device)``),
# so a float64 latent fails inside Forge's own function — for Forge's "UniPC" entry too. Forge samples in
# float32; that is what is compared here.
DTYPES = (torch.float32,)


class UniPcBh2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = fx.forge_k_sampling()
        cls.forge = fx.forge_sd_samplers_extra()

    def setUp(self):
        self.enterContext(fx.installed_k_sampling(self.ks))
        self.enterContext(fx.installed_sd_samplers_extra(self.forge))

    def _inputs(self, flow, steps, dtype, seed=0):
        sigmas = fx.flow_sigmas(steps, dtype=dtype) if flow else fx.eps_sigmas(steps, dtype=dtype)
        x = fx.seeded((2, 4, 5, 6), seed, dtype=dtype)
        return (x if flow else x * float(sigmas[0])), sigmas

    def test_it_is_forges_sample_unipc_with_variant_bh2(self):
        for flow in (False, True):
            for dtype in DTYPES:
                for steps in STEPS:
                    x, sigmas = self._inputs(flow, steps, dtype, seed=steps)
                    sampling = fx.FlowSampling if flow else fx.EpsSampling
                    with self.subTest(flow=flow, dtype=dtype, steps=steps):
                        seen, forge_seen = [], []
                        ours = unipc.sample_unipc_bh2(fx.ToyModel(sampling()), x.clone(), sigmas, disable=True,
                                                      callback=lambda d: seen.append((d["i"], d["x"])))
                        forge = self.forge.sample_unipc(fx.ToyModel(sampling()), x.clone(), sigmas, extra_args={},
                                                        disable=True, variant="bh2",
                                                        callback=lambda d: forge_seen.append((d["i"], d["x"])))
                        self.assertTrue(torch.equal(ours, forge))
                        self.assertTrue(torch.isfinite(ours).all())
                        self.assertEqual([i for i, _x in seen], [i for i, _x in forge_seen])
                        self.assertTrue(all(torch.equal(a, b) for (_i, a), (_j, b) in zip(seen, forge_seen)))

    def test_the_variant_reaches_unipc(self):
        """bh2 is not Forge's default bh1 on the same inputs (and the default really is bh1)."""
        self.assertEqual(inspect.signature(self.forge.sample_unipc).parameters["variant"].default, "bh1")
        for flow in (False, True):
            x, sigmas = self._inputs(flow, 10, torch.float32, seed=1)
            sampling = fx.FlowSampling if flow else fx.EpsSampling
            with self.subTest(flow=flow):
                ours = unipc.sample_unipc_bh2(fx.ToyModel(sampling()), x.clone(), sigmas, disable=True)
                bh1 = self.forge.sample_unipc(fx.ToyModel(sampling()), x.clone(), sigmas, extra_args={}, disable=True)
                self.assertGreater(float((ours - bh1).abs().max()), 1e-6)

    def test_extra_args_reach_the_model(self):
        x, sigmas = self._inputs(True, 6, torch.float32)
        model = fx.ToyModel(fx.FlowSampling(), cond_scale=1.5)
        unipc.sample_unipc_bh2(model, x, sigmas, extra_args={"model_options": {"transformer_options": {"k": 1}}},
                               disable=True)
        self.assertTrue(model.calls)
        self.assertTrue(all(call.model_options == {"transformer_options": {"k": 1}} for call in model.calls))

    def test_forges_module_is_looked_up_when_the_sampler_runs(self):
        calls = []
        stand_in = type(sys)("modules.sd_samplers_extra")
        stand_in.sample_unipc = lambda *args, **kwargs: calls.append(kwargs) or "ran"
        with fx.installed_sd_samplers_extra(stand_in):
            self.assertEqual(unipc.sample_unipc_bh2(None, None, None, extra_args={"a": 1}, callback=print), "ran")
        self.assertEqual(calls, [{"extra_args": {"a": 1}, "callback": print, "disable": False, "variant": "bh2"}])
        with fx.installed_sd_samplers_extra(stand_in):
            unipc.sample_unipc_bh2(None, None, None)
        self.assertEqual(calls[-1]["extra_args"], {})   # Forge's sample_unipc unpacks it into every model call
        self.assertEqual((unipc.UNIPC_MODULE, unipc.UNIPC_VARIANT), ("modules.sd_samplers_extra", "bh2"))

    def test_the_entry_has_forges_unipc_options(self):
        tree = ast.parse(fx.forge_source("modules/sd_samplers_kdiffusion.py"))
        table = next(n.value for n in tree.body
                     if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "samplers_k_diffusion")
        forge_unipc = next(entry for entry in table.elts
                           if isinstance(entry.elts[0], ast.Constant) and entry.elts[0].value == "UniPC")
        spec = next(spec for spec in registry.SPECS if spec.label == registry.LABEL_UNIPC_BH2)
        self.assertEqual(spec.options, ast.literal_eval(forge_unipc.elts[3]))
        self.assertEqual(spec.func, unipc.sample_unipc_bh2)
        self.assertEqual(spec.aliases, ("uni_pc_bh2",))


if __name__ == "__main__":
    unittest.main()
