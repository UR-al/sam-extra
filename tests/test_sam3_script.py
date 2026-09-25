"""scripts/!sam3.py — process()/postprocess_image() 를 Forge 없이 돌리는 테스트 (tests/test_args.py 에서 옮김)."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext.args import Sam3Args  # noqa: E402


# ---------------------------------------------------------------------------
# scripts/!sam3.py — process()/postprocess_image() 를 Forge 없이 돌린다.
# ---------------------------------------------------------------------------

_SCRIPT_MODULE = None


def _sam3_script():
    """``scripts/!sam3.py`` 를 Forge 스텁으로 한 번만 불러온다 (tests/test_anima_section.py:_load_script 와 같은 방식).

    스크립트는 ``from modules import shared`` 를 모듈 수준에 묶어 두므로 여기서 넣은 스텁을 계속 쓴다.
    """
    global _SCRIPT_MODULE
    if _SCRIPT_MODULE is not None:
        return _SCRIPT_MODULE

    package = types.ModuleType("modules")
    package.__path__ = []

    class Script:
        is_img2img = False

    scripts = types.ModuleType("modules.scripts")
    scripts.Script = Script
    scripts.AlwaysVisible = object()
    scripts.scripts_data = []
    script_callbacks = types.ModuleType("modules.script_callbacks")
    script_callbacks.callback_map = {}
    for name in ("on_before_ui", "on_app_started", "on_after_component", "on_ui_settings", "on_script_unloaded"):
        setattr(script_callbacks, name, lambda fn, **kw: None)
    shared = types.ModuleType("modules.shared")
    shared.opts = types.SimpleNamespace()
    shared.cmd_opts = types.SimpleNamespace()
    shared.state = types.SimpleNamespace(job="", job_count=0, interrupted=False, skipped=False, textinfo=None)
    processing = types.ModuleType("modules.processing")

    class StableDiffusionProcessingImg2Img:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    processing.StableDiffusionProcessingImg2Img = StableDiffusionProcessingImg2Img
    processing.process_images = lambda p: None
    sd_samplers = types.ModuleType("modules.sd_samplers")
    sd_samplers.all_samplers = []
    sd_schedulers = types.ModuleType("modules.sd_schedulers")
    sd_schedulers.schedulers = []
    stubs = {"modules": package}
    for name, module in (
        ("scripts", scripts),
        ("script_callbacks", script_callbacks),
        ("shared", shared),
        ("processing", processing),
        ("sd_samplers", sd_samplers),
        ("sd_schedulers", sd_schedulers),
    ):
        setattr(package, name, module)
        stubs[f"modules.{name}"] = module

    saved = {key: sys.modules.get(key) for key in stubs}
    sys.modules.update(stubs)
    try:
        path = ROOT / "scripts" / "!sam3.py"
        spec = importlib.util.spec_from_file_location("_t_sam3_script", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for key, original in saved.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original
    _SCRIPT_MODULE = module
    return module


def _processing(**attrs):
    base = dict(extra_generation_params={}, prompt="girl", negative_prompt="")
    base.update(attrs)
    return types.SimpleNamespace(**base)


class ProcessFallbackTests(unittest.TestCase):
    """감사 M4/M5: process() 의 폴백 기본값은 Sam3Args 기본값과 같아야 하고, 검증 실패는 로그+infotext 로 알린다."""

    def setUp(self):
        self.module = _sam3_script()
        self.script = self.module.Sam3MaskScript()

    def test_missing_keys_fall_back_to_sam3args_defaults(self):
        p = _processing()
        self.script.process(p, True, {})
        self.assertEqual(p._sam3_args, {"enabled": True, **Sam3Args().dict()},
                         "API 로 키를 생략하면 UI/Sam3Args 기본값(original, unload=True)을 받아야 한다")
        self.assertIs(p.extra_generation_params["SAM3 Enable"], True)
        self.assertNotIn("SAM3 Inpainting Fill", p.extra_generation_params)
        self.assertNotIn("SAM3 Unload After", p.extra_generation_params)
        self.assertNotIn("SAM3 Error", p.extra_generation_params)

    def test_given_values_still_win_over_the_defaults(self):
        p = _processing()
        self.script.process(p, True, {"sam3_inpainting_fill": "latent noise", "sam3_unload_after": False})
        self.assertEqual(p._sam3_args["sam3_inpainting_fill"], "latent noise")
        self.assertIs(p._sam3_args["sam3_unload_after"], False)
        self.assertEqual(p.extra_generation_params["SAM3 Inpainting Fill"], "latent noise")

    def test_xyz_out_of_range_values_keep_sam3_on(self):
        p = _processing(_sam3_xyz={"sam3_threshold": 1.5, "sam3_mask_blur": -1, "sam3_inpainting_fill": "Original"})
        self.script.process(p, True, {"sam3_prompt": "face"})
        self.assertIs(p._sam3_args["enabled"], True, "XYZ 격자 전체가 SAM3 없이 생성되면 안 된다")
        self.assertEqual(p._sam3_args["sam3_threshold"], 1.0)
        self.assertEqual(p._sam3_args["sam3_mask_blur"], 0)
        self.assertEqual(p._sam3_args["sam3_inpainting_fill"], "original")

    def test_validation_failure_is_logged_and_recorded_in_infotext(self):
        p = _processing()
        stderr = io.StringIO()
        with mock.patch.object(self.module, "Sam3Args", side_effect=ValueError("bogus_key: extra fields not permitted")), \
                contextlib.redirect_stderr(stderr):
            self.script.process(p, True, {"sam3_prompt": "face"})
        self.assertEqual(p._sam3_args, {"enabled": False})
        self.assertIn("SAM3 disabled for this generation", stderr.getvalue())
        self.assertIn("bogus_key", stderr.getvalue())
        self.assertNotIn("SAM3 Enable", p.extra_generation_params)
        self.assertIn("bogus_key", p.extra_generation_params["SAM3 Error"])

    def test_validation_failure_while_disabled_stays_quiet(self):
        p = _processing(_sam3_xyz={"sam3_threshold": 0.5})
        stderr = io.StringIO()
        with mock.patch.object(self.module, "Sam3Args", side_effect=ValueError("boom")), \
                contextlib.redirect_stderr(stderr):
            self.script.process(p, False, {"sam3_prompt": "face"})
        self.assertEqual(p._sam3_args, {"enabled": False})
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(p.extra_generation_params, {})


class PostprocessFailureTests(unittest.TestCase):
    """감사 M6: 검출·인페인트가 예외로 끝나면 'SAM3 Enable' 대신 'SAM3 Error' 를 남기고 SAM3 번들을 내린다."""

    def _run(self, exc, *, unload_after):
        module = _sam3_script()
        script = module.Sam3MaskScript()
        p = _processing(
            extra_generation_params={"SAM3 Enable": True, "SAM3 Prompt": "face"},
            _sam3_args={
                "enabled": True,
                "sam3_prompt": "face",
                "sam3_threshold": 0.4,
                "sam3_checkpoint": "sam3.pt",
                "sam3_device": "cpu",
                "sam3_unload_after": unload_after,
            },
        )
        pp = types.SimpleNamespace(image=Image.new("RGB", (8, 8)))
        unloads: list = []
        stderr = io.StringIO()
        with mock.patch.object(module, "run_sam3_on_pil", side_effect=exc), \
                mock.patch.object(module, "unload_sam3", lambda: unloads.append(1)), \
                contextlib.redirect_stderr(stderr):
            with self.assertRaises(type(exc)):
                # Forge 의 ScriptRunner.postprocess_image 가 예외를 잡아 콘솔에 보고한다 — 그 경로는 그대로 둔다.
                script.postprocess_image(p, pp)
        return p, unloads, stderr.getvalue()

    def test_error_replaces_the_enable_flag_in_infotext(self):
        p, unloads, log = self._run(FileNotFoundError("SAM3 checkpoint not found: sam3.pt"), unload_after=False)
        self.assertNotIn("SAM3 Enable", p.extra_generation_params, "인페인트 없는 이미지가 Enable: True 로 저장되면 안 된다")
        self.assertEqual(p.extra_generation_params["SAM3 Prompt"], "face")
        self.assertIn("checkpoint not found", p.extra_generation_params["SAM3 Error"])
        self.assertIn("SAM3 checkpoint not found", log)
        self.assertEqual(unloads, [], "unload_after 가 꺼져 있고 OOM 도 아니면 내리지 않는다")
        self.assertIs(p._sam3_mask_found, False, "🎯 빠른 버튼이 결과 없음으로 판단하게")

    def test_unload_after_still_unloads_on_failure(self):
        _, unloads, _ = self._run(RuntimeError("cv2 error"), unload_after=True)
        self.assertEqual(unloads, [1])

    def test_oom_always_unloads(self):
        _, unloads, _ = self._run(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"), unload_after=False)
        self.assertEqual(unloads, [1], "3.5GB 번들이 VRAM 에 남으면 안 된다")


class InnerPassTests(unittest.TestCase):
    """감사 M1: 생성 중 내부 인페인트 패스(p2)는 SAM3 만 돈다 — 🎯 버튼이 아니어도 ADetailer 가 또 돌지 않게."""

    def test_inner_pass_disables_adetailer_without_the_quick_button(self):
        _sam3_script()   # modules 스텁을 깔고 sam3ext.inpaint_core 를 불러온다
        from sam3ext import inpaint_core

        class _P2:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        p = _processing(
            width=64, height=64, steps=20, cfg_scale=7.0, sampler_name="Euler a", scheduler="Simple", seed=1,
            sd_model=object(), outpath_samples="", outpath_grids="", scripts=None, script_args=[],
        )
        with mock.patch.object(inpaint_core, "StableDiffusionProcessingImg2Img", _P2):
            p2 = inpaint_core.build_i2i(p, Image.new("RGB", (64, 64)), Sam3Args().dict())
        self.assertFalse(getattr(p, "_sam3_quick", False))
        self.assertIs(p2._ad_disabled, True)
        self.assertIs(p2._sam3_inner, True)

    def _p2_with_seed_script(self):
        """script_args 대입 때 Forge Seed 스크립트 setup 처럼 시드를 칸 값(-1)으로 덮어쓰는 가짜 p2."""

        class _P2:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            @property
            def script_args(self):
                return self._script_args

            @script_args.setter
            def script_args(self, value):
                self._script_args = value
                # modules/processing_scripts/seed.py ScriptSeed.setup — API 호출이면 칸 기본값
                self.seed, self.subseed, self.subseed_strength = -1, -1, 0
                self.seed_resize_from_w = self.seed_resize_from_h = 0

        return _P2

    def test_inner_pass_keeps_the_outer_seed_after_the_seed_script_setup(self):
        _sam3_script()
        from sam3ext import inpaint_core

        p = _processing(
            width=64, height=64, steps=20, cfg_scale=7.0, sampler_name="Euler a", scheduler="Simple", seed=4140358882,
            subseed=77, subseed_strength=0.25, seed_resize_from_h=512, seed_resize_from_w=768,
            sd_model=object(), outpath_samples="", outpath_grids="", scripts=None, script_args=[],
        )
        with mock.patch.object(inpaint_core, "StableDiffusionProcessingImg2Img", self._p2_with_seed_script()):
            p2 = inpaint_core.build_i2i(p, Image.new("RGB", (64, 64)), Sam3Args().dict())
            self.assertEqual(p2.seed, 4140358882, "바깥 생성의 시드 — 예전엔 Seed 스크립트 칸의 -1 로 매번 무작위였다")
            self.assertEqual((p2.subseed, p2.subseed_strength), (77, 0.25))
            self.assertEqual((p2.seed_resize_from_h, p2.seed_resize_from_w), (512, 768))
            fixed = inpaint_core.build_i2i(p, Image.new("RGB", (64, 64)), dict(Sam3Args().dict(), sam3_use_seed=True, sam3_seed=123))
            self.assertEqual(fixed.seed, 123, "SAM3 시드를 따로 지정하면 그 값")

    def test_standalone_refine_keeps_its_seed_after_the_seed_script_setup(self):
        _sam3_script()
        from sam3ext import inpaint_core

        with mock.patch.object(inpaint_core, "StableDiffusionProcessingImg2Img", self._p2_with_seed_script()),                 mock.patch.object(inpaint_core, "shared", mock.MagicMock()):
            p2 = inpaint_core.build_standalone_i2i(
                Image.new("RGB", (64, 64)), dict(Sam3Args().dict(), sam3_seed=321),
                sd_model=object(), outpath_samples="", outpath_grids="", scripts_runner=object(), script_args=[],
            )
        self.assertEqual(p2.seed, 321)


# ---------------------------------------------------------------------------
# 후속 과제(2026-09-23): 배치 infotext, 실패 로그 문구, 'Unload after' RAM 보관 설정, Device 축 cost
# ---------------------------------------------------------------------------


def _enabled_args(**changes):
    args = {
        "enabled": True,
        "sam3_prompt": "face",
        "sam3_threshold": 0.4,
        "sam3_checkpoint": "sam3.pt",
        "sam3_device": "cpu",
        "sam3_unload_after": False,
        "sam3_save_artifacts": False,
        "sam3_preview_overlay": False,
        "sam3_mode": "Inpaint",
    }
    args.update(changes)
    return args


def _empty_result():
    blank = Image.new("L", (8, 8), 0)
    return types.SimpleNamespace(mask=blank, masks=[blank], overlay=Image.new("RGB", (8, 8)))


def _pp():
    return types.SimpleNamespace(image=Image.new("RGB", (8, 8)))


class BatchInfotextTests(unittest.TestCase):
    """배치(n_iter>1)에서 한 장이 실패해도 그 뒤 성공한 장의 infotext 는 'SAM3 Enable' 을 달고 'SAM3 Error' 가 없어야 한다."""

    def test_success_after_a_failed_image_restores_the_infotext(self):
        module = _sam3_script()
        script = module.Sam3MaskScript()
        params = {"Steps": 20, "SAM3 Enable": True, "SAM3 Prompt": "face", "SAM3 Version": "x"}
        p = _processing(extra_generation_params=params, _sam3_args=_enabled_args())
        expected = list(params.items())
        outcomes = [RuntimeError("cv2 error"), _empty_result()]

        def fake_run(**kwargs):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with mock.patch.object(module, "run_sam3_on_pil", fake_run), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(RuntimeError):
                script.postprocess_image(p, _pp())
            self.assertIn("SAM3 Error", p.extra_generation_params)
            self.assertNotIn("SAM3 Enable", p.extra_generation_params)
            script.postprocess_image(p, _pp())
        self.assertEqual(list(p.extra_generation_params.items()), expected,
                         "성공한 장은 실패가 없던 배치와 같은 infotext(순서 포함)여야 한다")
        self.assertIs(p._sam3_mask_found, False)

    def test_process_time_error_is_not_touched(self):
        # process() 의 검증 실패는 SAM3 가 꺼진 채로 남는다 — postprocess_image 는 아무것도 되돌리지 않는다.
        module = _sam3_script()
        p = _processing(extra_generation_params={"SAM3 Error": "bad"}, _sam3_args={"enabled": False})
        module.Sam3MaskScript().postprocess_image(p, _pp())
        self.assertEqual(p.extra_generation_params, {"SAM3 Error": "bad"})


class UnloadLogTests(unittest.TestCase):
    """'Unload after' 로그가 실제 동작(CPU RAM 보관 / 완전 해제)을 말해야 한다 — 예전 'unloaded from VRAM' 문구 금지."""

    def _fail(self, kept):
        module = _sam3_script()
        p = _processing(extra_generation_params={"SAM3 Enable": True}, _sam3_args=_enabled_args(sam3_unload_after=True))
        stderr = io.StringIO()
        with mock.patch.object(module, "run_sam3_on_pil", side_effect=RuntimeError("boom")), \
                mock.patch.object(module, "unload_sam3", lambda: kept), \
                contextlib.redirect_stderr(stderr):
            with self.assertRaises(RuntimeError):
                module.Sam3MaskScript().postprocess_image(p, _pp())
        return stderr.getvalue()

    def test_failure_log_when_the_bundle_is_kept_in_ram(self):
        log = self._fail(True)
        self.assertIn("[-] SAM3: model moved from VRAM to CPU RAM after the failure (moves back on next detection).", log)
        self.assertNotIn("unloaded from VRAM", log)

    def test_failure_log_when_the_bundle_is_released(self):
        log = self._fail(False)
        self.assertIn("[-] SAM3: model released from VRAM and RAM after the failure (reloads on next detection).", log)
        self.assertNotIn("unloaded from VRAM", log)

    def test_success_log_follows_the_setting(self):
        module = _sam3_script()
        for kept, text in (
            (True, "[-] SAM3: model moved from VRAM to CPU RAM (moves back on next detection)."),
            (False, "[-] SAM3: model released from VRAM and RAM (reloads on next detection)."),
        ):
            p = _processing(_sam3_args=_enabled_args(sam3_unload_after=True))
            stderr = io.StringIO()
            with mock.patch.object(module, "run_sam3_on_pil", lambda **kw: _empty_result()), \
                    mock.patch.object(module, "unload_sam3", lambda: kept), \
                    contextlib.redirect_stderr(stderr):
                module.Sam3MaskScript().postprocess_image(p, _pp())
            self.assertIn(text, stderr.getvalue())


class KeepInRamSettingTests(unittest.TestCase):
    """'Unload after' 뒤 번들을 RAM 에 둘지 끌 수 있는 Forge 설정 — 기본값은 지금 동작(보관)."""

    def _register(self):
        module = _sam3_script()
        added: dict = {}

        class OptionInfo:
            def __init__(self, default=None, label="", component=None, component_args=None, onchange=None,
                         section=None, **kwargs):
                self.default, self.label, self.onchange, self.section = default, label, onchange, section
                self.component, self.component_args = component, component_args

            def info(self, text):
                self.info_text = text
                return self

        opts = types.SimpleNamespace(add_option=lambda key, info: added.__setitem__(key, info))
        with mock.patch.object(module.shared, "OptionInfo", OptionInfo, create=True), \
                mock.patch.object(module.shared, "opts", opts):
            module.on_ui_settings()
        return module, added

    def test_setting_is_registered_with_the_current_behaviour_as_default(self):
        module, added = self._register()
        from sam3ext import core

        self.assertIn(core.OPT_UNLOAD_KEEP_IN_RAM, added)
        info = added[core.OPT_UNLOAD_KEEP_IN_RAM]
        self.assertIs(info.default, True, "기본값은 지금처럼 RAM 보관")
        self.assertEqual(info.section[0], "sam3_mask")
        # 키는 그대로 두고 표시 이름만 다른 설정 섹션(SAM Extra Anima 3.8B 등)과 같은 머리말로 맞춘다.
        self.assertEqual(info.section[1], "SAM Extra SAM3")
        self.assertIn("RAM", info.label)
        source = (ROOT / "scripts" / "!sam3.py").read_text(encoding="utf-8")
        self.assertIn("script_callbacks.on_ui_settings(on_ui_settings", source)

    def test_the_ipa_duplicate_policy_setting_defaults_to_lineage(self):
        module, added = self._register()
        from sam3ext.anima_ipa import options

        self.assertIn(options.OPT_DUPLICATE_POLICY, added)
        info = added[options.OPT_DUPLICATE_POLICY]
        self.assertEqual(info.default, "lineage")
        self.assertEqual(info.section, ("sam3_reference", "SAM Extra Character Reference"))
        values = [value for _label, value in info.component_args["choices"]]
        self.assertEqual(sorted(values), sorted(options.DUPLICATE_POLICIES))
        self.assertIn("SAM3 IPA Duplicates", info.info_text)

    def test_turning_the_setting_off_drops_a_bundle_parked_in_ram(self):
        module, added = self._register()
        from sam3ext import core

        onchange = added[core.OPT_UNLOAD_KEEP_IN_RAM].onchange
        drops: list = []
        with mock.patch.object(module, "drop_offloaded_sam3", lambda: drops.append(1)):
            with mock.patch.object(module.shared, "opts", types.SimpleNamespace(**{core.OPT_UNLOAD_KEEP_IN_RAM: True})):
                onchange()
            self.assertEqual(drops, [], "켤 때는 아무것도 버리지 않는다")
            with mock.patch.object(module.shared, "opts", types.SimpleNamespace(**{core.OPT_UNLOAD_KEEP_IN_RAM: False})):
                onchange()
        self.assertEqual(drops, [1])


class DeviceAxisCostTests(unittest.TestCase):
    """'[SAM3] Device' 가 바뀌어도 번들을 버리고 다시 빌드한다 — Checkpoint 축과 같은 cost 로 바깥 루프에."""

    def test_device_axis_has_the_checkpoint_cost(self):
        module = _sam3_script()

        class AxisOption:
            def __init__(self, label, type, apply, **kwargs):
                self.label, self.kwargs = label, kwargs

        xyz = types.SimpleNamespace(AxisOption=AxisOption, axis_options=[], format_value=lambda *a: "")
        script_class = type("XYZ", (), {"__module__": "xyz_grid.py"})
        with mock.patch.object(module.scripts, "scripts_data",
                               [types.SimpleNamespace(script_class=script_class, module=xyz)]):
            module.make_axis_on_xyz_grid()
        by_label = {axis.label: axis for axis in xyz.axis_options}
        self.assertEqual(by_label["[SAM3] Device"].kwargs.get("cost"), 0.9)
        self.assertEqual(by_label["[SAM3] Device"].kwargs.get("cost"), by_label["[SAM3] Checkpoint"].kwargs.get("cost"))


if __name__ == "__main__":
    unittest.main()
