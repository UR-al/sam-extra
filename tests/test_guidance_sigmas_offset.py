"""``forge_sampling_offset`` moved into sam3ext/guidance/sigmas.py — Detail Daemon and DAVE unchanged.

Detail Daemon (scripts/anima_detail_daemon.py) and the DAVE gate (scripts/anima_safe_pag.py) each had
an identical ``_forge_sampling_offset`` / ``_is_img2img_request``; Colorcraft needs the same rule. The
shared copy is the old body, and both scripts now import it under their old names. These tests pin
that: the scripts' names are the shared function, the shared function answers exactly like the old
code (copied below from forge_sam3_extension@991c45b) over Forge's own ``setup_img2img_steps``, and
neither script defines its own copy any more. The scripts' behaviour tests
(test_detail_daemon_origin.py ForgeWalkedListTests, test_dave_origin.py ForgeSamplingOffsetTests)
run unchanged against the shared function.
"""

from __future__ import annotations

import ast
import importlib.util
import itertools
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.guidance import sigmas  # noqa: E402


def _load(name, relpath):
    spec_ = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = sys.modules.get(name)
    if module is None:
        module = importlib.util.module_from_spec(spec_)
        sys.modules[name] = module
        spec_.loader.exec_module(module)
    return module


def _support():
    return _load("_colorcraft_support", "tests/_colorcraft_support.py")


def _load_script(name, relpath):
    """A sam-extra script without the WebUI (``modules`` stubbed only while it is imported)."""
    modules = types.ModuleType("modules")

    class Script:
        pass

    modules.scripts = types.SimpleNamespace(Script=Script, AlwaysVisible=object(), scripts_data=[])
    modules.script_callbacks = types.SimpleNamespace(on_cfg_denoiser=lambda fn: None, on_before_ui=lambda fn: None)
    with mock.patch.dict(sys.modules, {"modules": modules}):
        spec_ = importlib.util.spec_from_file_location(name, ROOT / relpath)
        module = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(module)
    return module


# --- the code as it was (forge_sam3_extension@991c45b) ------------------------------------------
# origin: forge_sam3_extension@991c45b:scripts/anima_detail_daemon.py:236-273 (identical bodies in
# scripts/anima_safe_pag.py:1201-1236; only the docstrings differed)


def _old_is_img2img_request(p) -> bool:
    return any(
        cls.__name__ == "StableDiffusionProcessingImg2Img"
        for cls in type(p).__mro__
    )


def _old_forge_sampling_offset(p):
    if getattr(p, "is_hr_pass", False):
        requested = getattr(p, "hr_second_pass_steps", 0) or getattr(p, "steps", None)
    elif _old_is_img2img_request(p):
        requested = None
    else:
        return 0
    try:
        from modules import sd_samplers_common

        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, requested)
        return int(steps) - int(t_enc) - 1
    except Exception:
        return None


# origin: forge_sampling_offset@991c45b:scripts/anima_safe_pag.py:1239-1247 (_sampler_publishes_sigmas)
def _old_sampler_publishes_sigmas(p) -> bool:
    denoiser = getattr(getattr(p, "sampler", None), "model_wrap_cfg", None)
    return not bool(getattr(denoiser, "classic_ddim_eps_estimation", False))


class SharedOffsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dd = _load_script("_offset_detail_daemon", "scripts/anima_detail_daemon.py")
        cls.pag = _load_script("_offset_safe_pag", "scripts/anima_safe_pag.py")

    def test_scripts_use_the_shared_functions(self):
        self.assertIs(self.dd._forge_sampling_offset, sigmas.forge_sampling_offset)
        self.assertIs(self.dd._is_img2img_request, sigmas.is_img2img_request)
        self.assertIs(self.pag._forge_sampling_offset, sigmas.forge_sampling_offset)
        self.assertIs(self.pag._is_img2img_request, sigmas.is_img2img_request)

    def test_neither_script_keeps_its_own_copy(self):
        for relpath in ("scripts/anima_detail_daemon.py", "scripts/anima_safe_pag.py"):
            tree = ast.parse((ROOT / relpath).read_text(encoding="utf-8"))
            defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            with self.subTest(script=relpath):
                self.assertNotIn("_forge_sampling_offset", defined)
                self.assertNotIn("_is_img2img_request", defined)
        colorcraft = (ROOT / "sam3ext" / "colorcraft" / "hook.py").read_text(encoding="utf-8")
        self.assertIn("from sam3ext.guidance.sigmas import forge_sampling_offset", colorcraft)

    def test_same_answers_as_the_old_code(self):
        support = _support()
        if support.FORGE is None:
            self.skipTest("the Forge checkout is not next to this extension")

        class Img2Img(support.StableDiffusionProcessingImg2Img):
            pass

        requests = []
        for steps, denoise in ((28, 0.5), (20, 0.75), (30, 0.35), (24, 1.0), (10, 0.05)):
            requests.append(types.SimpleNamespace(steps=steps, denoising_strength=denoise, is_hr_pass=False))
            requests.append(support.StableDiffusionProcessingImg2Img(steps=steps, denoising_strength=denoise,
                                                                     is_hr_pass=False))
            requests.append(Img2Img(steps=steps, denoising_strength=denoise, is_hr_pass=False))
            for hr_steps in (0, 10, 40):
                requests.append(types.SimpleNamespace(steps=steps, hr_second_pass_steps=hr_steps,
                                                      denoising_strength=denoise, is_hr_pass=True))
        requests.append(types.SimpleNamespace(is_hr_pass=True))                           # no steps at all
        stubs = [support.forge_modules(False), support.forge_modules(True), types.ModuleType("modules")]
        checked = 0
        for request, stub in itertools.product(requests, stubs):
            with mock.patch.dict(sys.modules, {"modules": stub}):
                old = _old_forge_sampling_offset(request)
                new = sigmas.forge_sampling_offset(request)
            with self.subTest(request=request, stub=getattr(stub, "sd_samplers_common", None) is not None):
                self.assertEqual(new, old)
                self.assertEqual(sigmas.is_img2img_request(request), _old_is_img2img_request(request))
            checked += 1
        self.assertEqual(checked, 31 * 3)   # 31 requests x (Forge, Forge with img2img_fix_steps, no Forge)

    def test_sampler_publishes_sigmas_matches_the_guidance_rule(self):
        cases = [
            types.SimpleNamespace(),
            types.SimpleNamespace(sampler=None),
            types.SimpleNamespace(sampler=types.SimpleNamespace(model_wrap_cfg=types.SimpleNamespace())),
            types.SimpleNamespace(sampler=types.SimpleNamespace(
                model_wrap_cfg=types.SimpleNamespace(classic_ddim_eps_estimation=True))),
            types.SimpleNamespace(sampler=types.SimpleNamespace(
                model_wrap_cfg=types.SimpleNamespace(classic_ddim_eps_estimation=False))),
        ]
        for p in cases:
            self.assertEqual(sigmas.sampler_publishes_sigmas(p), _old_sampler_publishes_sigmas(p))
            self.assertEqual(sigmas.sampler_publishes_sigmas(p), self.pag._sampler_publishes_sigmas(p))


if __name__ == "__main__":
    unittest.main()
