"""Forge's CFG denoiser step counter around the Euler (SMEA) Dy CFG++ sub-steps.

Forge's ``CFGDenoiser.forward`` (modules/sd_samplers_cfg_denoiser.py, executed unchanged from Forge's
file) counts its calls in ``self.step`` and reads the count for

* prompt editing — ``prompt_parser.reconstruct_multicond_batch`` / ``reconstruct_cond_batch`` (Forge's
  real modules/prompt_parser.py, the schedules from its own ``get_learned_conditioning_prompt_schedules``);
* skip-early-CFG — ``0 < self.step / self.total_steps <= opts.skip_early_cond`` → ``cond_scale = 1``;
* the refiner's switch-by-steps — it hands itself to ``sd_samplers_common.apply_refiner``, which compares
  ``cfg_denoiser.step / cfg_denoiser.total_steps`` (the stand-in records that count).

A sub-step must be evaluated as the step it belongs to and must leave the count where it was, so none
of them shift (upstream's sub-steps advanced it by one call each). Without Forge next to the extension
(or ``$SAM3_FORGE_ROOT``) these tests skip.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
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

from sam3ext.extra_samplers import common, euler_dy  # noqa: E402

_DENOISER = "modules/sd_samplers_cfg_denoiser.py"
_CALLBACKS = "modules/script_callbacks.py"
_PROMPT_PARSER = "modules/prompt_parser.py"
_COMMON = "modules/sd_samplers_common.py"

STEPS = 8
CFG = 1.8
FULL = (8, 8)
SALT = {"a cat photo": 1.0, "a dog photo": 2.0}


def _class_source(rel_path: str, name: str) -> str:
    source = fx.forge_source(rel_path)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
    return ast.get_source_segment(source, node)


class _Interrupted(BaseException):
    """``sd_samplers_common.InterruptedException`` stand-in (a BaseException, like Forge's)."""


class _Host:
    """Forge's ``CFGDenoiser`` in a namespace with its module's names: torch, the real prompt_parser and
    callback parameter classes; stand-ins for the CFG itself (``sampling_function``: a toy CFG that runs
    the post-CFG list), ``state``, ``opts`` and ``sd_samplers_common``."""

    def __init__(self, prompt_parser, *, skip_early_cond: float = 0.0):
        self.records = []        # per forward: (latent H/W, active prompt, cond_scale, sub-step marker)
        self.refiner_seen = []   # per forward: (cfg_denoiser.step, cfg_denoiser.total_steps)
        self.interrupt_after = None
        self.state = types.SimpleNamespace(interrupted=False, skipped=False, sampling_step=0, sampling_steps=0)
        namespace = {
            "torch": torch,
            "prompt_parser": prompt_parser,
            "opts": types.SimpleNamespace(skip_early_cond=skip_early_cond, s_min_uncond_all=False),
            "state": self.state,
            "sd_samplers_common": types.SimpleNamespace(
                apply_refiner=self._apply_refiner, store_latent=lambda latent: None,
                InterruptedException=_Interrupted,
            ),
            "sampling_function": self._sampling_function,
            "cfg_denoiser_callback": lambda params: None,
            "cfg_after_cfg_callback": self._after_cfg,
        }
        for name in ("CFGDenoiserParams", "AfterCFGCallbackParams"):
            exec(compile(_class_source(_CALLBACKS, name), _CALLBACKS, "exec"), namespace)
        exec(compile(_class_source(_DENOISER, "CFGDenoiser"), _DENOISER, "exec"), namespace)
        self.CFGDenoiser = namespace["CFGDenoiser"]

    def _apply_refiner(self, cfg_denoiser, x, sigma) -> bool:
        self.refiner_seen.append((cfg_denoiser.step, cfg_denoiser.total_steps))
        return False

    def _sampling_function(self, denoiser, denoiser_params, cond_scale, cond_composition, extra_model_options=None):
        x, sigma = denoiser_params.x, denoiser_params.sigma
        prompt = next(text for text, salt in SALT.items() if float(denoiser_params.text_cond.mean()) == salt)
        cond = fx.toy_x0(x, sigma, SALT[prompt])
        uncond = fx.toy_x0(x, sigma, float(denoiser_params.text_uncond.mean()))
        denoised = uncond + (cond - uncond) * cond_scale
        options = extra_model_options or {}
        for fn in options.get("sampler_post_cfg_function", []):
            denoised = fn({
                "denoised": denoised, "cond": [], "uncond": [], "cond_scale": cond_scale, "model": None,
                "uncond_denoised": uncond, "cond_denoised": cond, "sigma": sigma, "model_options": options,
                "input": x,
            })
        marker = (options.get("transformer_options") or {}).get(common.SUBSTEP_MARKER)
        self.records.append((tuple(x.shape[-2:]), prompt, float(cond_scale), marker))
        return denoised, cond, uncond

    def _after_cfg(self, params) -> None:
        if self.interrupt_after is not None and len(self.records) == self.interrupt_after:
            self.state.interrupted = True   # Interrupt pressed during this evaluation

    def denoiser(self, p):
        class Denoiser(self.CFGDenoiser):   # Forge's CFGDenoiserKDiffusion: inner_model is the schedule linker
            @property
            def inner_model(inner):
                return inner._linker

        model = Denoiser(sampler=None)
        model._linker = types.SimpleNamespace(predictor=fx.FlowSampling(), inner_model=None)
        model.p = p
        model.total_steps = STEPS   # Sampler.launch_sampling: config.total_steps(steps)
        model.step = 0              # Sampler.initialize
        return model


class StepCounterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = fx.forge_k_sampling()
        try:
            cls.prompt_parser = fx.forge_module_from_source(_PROMPT_PARSER, "modules.prompt_parser")
        except ImportError as exc:   # lark: Forge's own requirement
            raise unittest.SkipTest(f"Forge's prompt_parser cannot be loaded here: {exc}")

    def setUp(self):
        self.enterContext(fx.installed_k_sampling(self.ks))

    def _extra_args(self, prompt: str) -> dict:
        pp = self.prompt_parser
        (schedule,) = pp.get_learned_conditioning_prompt_schedules([prompt], STEPS)
        scheduled = [pp.ScheduledPromptConditioning(end_at, torch.full((1, 4), SALT[text])) for end_at, text in schedule]
        cond = pp.MulticondLearnedConditioning(shape=(1,), batch=[[pp.ComposableScheduledPromptConditioning(scheduled)]])
        uncond = [[pp.ScheduledPromptConditioning(STEPS, torch.full((1, 4), -1.0))]]
        return {"cond": cond, "uncond": uncond, "cond_scale": CFG, "s_min_uncond": 0.0, "image_cond": None}

    def _run(self, *, smea: bool, substeps: bool, prompt: str, skip_early_cond: float = 0.0, interrupt_after=None):
        host = _Host(self.prompt_parser, skip_early_cond=skip_early_cond)
        host.interrupt_after = interrupt_after
        p = types.SimpleNamespace(is_hr_pass=False, extra_generation_params={}, scripts=None)
        model = host.denoiser(p)
        x = fx.seeded((1, 4, *FULL), 3, dtype=torch.float32)
        sampler = euler_dy.sample_euler_smea_dy_cfg_pp if smea else euler_dy.sample_euler_dy_cfg_pp
        out = sampler(model, x, fx.flow_sigmas(STEPS, dtype=torch.float32), extra_args=self._extra_args(prompt),
                      disable=True, noise_sampler=fx.NoiseSequence(x), substeps=substeps)
        return out, host, model

    @staticmethod
    def _main(host):
        return [record for record in host.records if record[0] == FULL and record[3] is None]

    @staticmethod
    def _step_of_call(smea: bool) -> list:
        """The step every model call belongs to: Euler Dy evaluates a sub-step after the own evaluation
        of steps 2 and 3, Euler SMEA Dy after those of steps 0 and 1."""
        substep_steps = (0, 1) if smea else (2, 3)
        calls = []
        for step in range(STEPS):
            calls.append(step)
            if step in substep_steps:
                calls.append(step)
        return calls

    def test_forges_refiner_switch_reads_the_counter(self):
        self.assertIn("cfg_denoiser.step / cfg_denoiser.total_steps", fx.forge_definition(_COMMON, "apply_refiner"))
        forward = fx.forge_definition(_DENOISER, "forward", cls="CFGDenoiser")
        for use in ("reconstruct_multicond_batch(cond, self.step)", "self.step / self.total_steps", "self.step += 1"):
            self.assertIn(use, forward)

    def test_a_substep_counts_as_its_step_and_the_count_does_not_drift(self):
        for smea in (False, True):
            with self.subTest(smea=smea):
                _, host, model = self._run(smea=smea, substeps=True, prompt="a [cat:dog:2] photo")
                # what apply_refiner's switch-by-steps sees: the sub-step has its step's count
                self.assertEqual([seen for seen, _total in host.refiner_seen], self._step_of_call(smea))
                self.assertTrue(all(total == STEPS for _seen, total in host.refiner_seen))
                self.assertEqual(model.step, STEPS)       # one count per step, as Forge's total_steps assumes
                markers = [record[3] for record in host.records]
                self.assertEqual(sum(1 for marker in markers if marker is not None), 2)

    def test_prompt_editing_is_not_shifted(self):
        for smea, prompt, edge in ((False, "a [cat:dog:2] photo", 2), (True, "a [cat:dog:1] photo", 1)):
            with self.subTest(smea=smea, prompt=prompt):
                _, plain, _ = self._run(smea=smea, substeps=False, prompt=prompt)
                _, host, _ = self._run(smea=smea, substeps=True, prompt=prompt)
                per_step = [record[1] for record in plain.records]
                self.assertEqual(len(per_step), STEPS)
                # the edit falls between the edge step (which has a sub-step) and the next one
                self.assertEqual((per_step[edge], per_step[edge + 1]), ("a cat photo", "a dog photo"))
                self.assertEqual([record[1] for record in self._main(host)], per_step)
                self.assertEqual([record[1] for record in host.records],
                                 [per_step[step] for step in self._step_of_call(smea)])

    def test_skip_early_cfg_is_not_shifted(self):
        # skip_early_cond 0.3 of 8 steps → steps 1-2 run at CFG 1 (0 < step/8 <= 0.3); 0.15 → step 1 only
        for smea, fraction in ((False, 0.3), (True, 0.15)):
            with self.subTest(smea=smea):
                _, plain, _ = self._run(smea=smea, substeps=False, prompt="a [cat:dog:2] photo",
                                        skip_early_cond=fraction)
                _, host, _ = self._run(smea=smea, substeps=True, prompt="a [cat:dog:2] photo",
                                       skip_early_cond=fraction)
                per_step = [record[2] for record in plain.records]
                self.assertEqual(per_step[:4], [CFG, 1.0, 1.0, CFG] if not smea else [CFG, 1.0, CFG, CFG])
                self.assertEqual([record[2] for record in self._main(host)], per_step)
                self.assertEqual([record[2] for record in host.records],
                                 [per_step[step] for step in self._step_of_call(smea)])

    def test_an_interrupt_in_a_substep_leaves_the_count_of_the_finished_steps(self):
        # Interrupt pressed during step 2's own evaluation (the third call): Forge's forward raises at the
        # entry of the sub-step that follows, before its own "self.step += 1".
        host = _Host(self.prompt_parser)
        host.interrupt_after = 3
        model = host.denoiser(types.SimpleNamespace(is_hr_pass=False, extra_generation_params={}, scripts=None))
        x = fx.seeded((1, 4, *FULL), 3, dtype=torch.float32)
        with self.assertRaises(_Interrupted):
            euler_dy.sample_euler_dy_cfg_pp(model, x, fx.flow_sigmas(STEPS, dtype=torch.float32),
                                            extra_args=self._extra_args("a [cat:dog:2] photo"), disable=True,
                                            noise_sampler=fx.NoiseSequence(x))
        self.assertEqual(len(host.records), 3)   # steps 0, 1, 2 evaluated; the sub-step never ran
        self.assertEqual(model.step, 3)


if __name__ == "__main__":
    unittest.main()
