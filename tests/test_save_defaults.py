"""sam3ext/save_defaults.py — '기본값 저장' 버튼이 활성 UI Preset 의 Forge 설정 22개에 쓰는 것.

가짜 opts(modules/options.py Options 의 data / data_labels / set / save / getattr 동작) 위에서:

* 22칸이 forge_main_entry 의 output_targets 순서대로 그 프리셋의 키에 들어가고, 다른 프리셋은 그대로다;
* 모듈은 modules_change 처럼 전체 경로로 바꿔 정렬하고, 숫자는 정수·실수로, 범위는 그 설정의 슬라이더 범위로 본다;
* 검사 하나라도 실패하면 아무것도 쓰지 않고, 쓰는 도중 실패하면 메모리 값도 되돌린다;
* Forge 에 없는 키(Distilled CFG·Shift 가 없는 프리셋의 dcfg)는 건너뛴다;
* 스텝·크기·CFG 10칸은 프리셋 값이 0 이하(= on_preset_change 가 gr.skip — 화면 값을 덮어쓰지 않음)면 그대로 둔다.

Forge 체크아웃이 있으면(없으면 건너뜀 — CI) Forge 의 진짜 코드로 확인한다: presets.register 가 만드는 설정(모든
프리셋), forge_main_entry 의 컴포넌트·출력 순서, 그리고 저장한 값이 on_preset_change 로 다시 읽혀 같은 UI 값이 되는지
(modules_change 의 저장 모양과도 같은지), 그리고 0 으로 둔 칸이 저장 뒤에도 Forge 에서 건너뛰어지는지(칸마다).
"""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no version check over the network

import ast  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest import mock  # noqa: E402

# imported before any sys.modules patch: presets.register imports gradio, and a patch.dict would drop it again
import gradio  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import save_defaults as sd  # noqa: E402
from tests._forge_checkout import require_forge_file  # noqa: E402


class FakeInfo:
    """modules/options.py OptionInfo 의 쓰는 부분."""

    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None,
                 refresh=None, comment_before="", comment_after="", infotext=None, restrict_api=False, category_id=None):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = section
        self.category_id = category_id
        self.do_not_save = False


class FakeRow(FakeInfo):
    def __init__(self):
        super().__init__("", label="")
        self.do_not_save = True


def options_section(section_identifier, options_dict):
    """modules/options.py options_section 그대로."""
    for v in options_dict.values():
        if len(section_identifier) == 2:
            v.section = section_identifier
        elif len(section_identifier) == 3:
            v.section = section_identifier[0:2]
            v.category_id = section_identifier[2]
    return options_dict


class FakeOpts:
    """modules/options.py Options 의 data / data_labels / set / save / __getattr__ 동작."""

    def __init__(self, labels, data=None):
        object.__setattr__(self, "data_labels", labels)
        object.__setattr__(self, "data", dict(data or {}))
        object.__setattr__(self, "saved", [])
        object.__setattr__(self, "fail_save", None)
        object.__setattr__(self, "refuse", set())

    def set(self, key, value, is_api=False, run_callbacks=True):
        oldval = self.data.get(key, None)
        if oldval == value:
            return False
        option = self.data_labels[key]
        if option.do_not_save:
            return False
        if key in self.refuse:          # Options.__setattr__ 의 RuntimeError(제한된 설정) → set 은 False
            return False
        self.data[key] = value
        return True

    def save(self, filename):
        if self.fail_save is not None:
            raise self.fail_save
        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file, indent=4, ensure_ascii=False)
        self.saved.append(filename)

    def __getattr__(self, item):
        data = object.__getattribute__(self, "data")
        if item in data:
            return data[item]
        labels = object.__getattribute__(self, "data_labels")
        if item in labels:
            return labels[item].default
        raise AttributeError(item)


SAMPLERS = ["Euler", "Euler a", "ER SDE", "DPM++ 2M", "LCM", "Res Multistep", "ER SDE (Tunable)"]
SCHEDULERS = ["Automatic", "Simple", "Normal", "Beta", "Karras", "Flow Cosmos rho7"]
DTYPES = ["Automatic", "Automatic (fp16 LoRA)", "float8-e4m3fn", "float8-e4m3fn (fp16 LoRA)",
          "float8-e5m2", "float8-e5m2 (fp16 LoRA)"]
MODULES = {
    "qwen_image_vae.safetensors": r"C:\forge\models\VAE\qwen_image_vae.safetensors",
    "qwen_3_06b_base.safetensors": r"C:\forge\models\text_encoder\qwen_3_06b_base.safetensors",
    "ae.safetensors": r"C:\forge\models\VAE\ae.safetensors",
}
CHECKPOINTS = {"anima-preview2.safetensors [1234abcd]", "animaPreview2_v20.safetensors", "sdxl\\base.safetensors"}


def mini_register(names=("sd", "xl", "flux", "anima", "wan")):
    """modules_forge/presets.py register 의 모양(키·기본값·슬라이더 범위) — CI 용. Forge 가 있으면 진짜와 비교한다."""
    shift = {"xl": -9.0, "anima": 3.0, "wan": 5.0}
    distill = {"flux": 3.0}
    labels = {"forge_preset": FakeInfo("sd")}
    for name in names:
        labels.update({
            f"forge_checkpoint_{name}": FakeInfo(None),
            f"forge_additional_modules_{name}": FakeInfo([]),
            f"forge_unet_storage_dtype_{name}": FakeInfo("Automatic"),
            f"{name}_t2i_ss1": FakeRow(),
            f"{name}_t2i_sampler": FakeInfo("Euler", "txt2img Sampler", None, lambda: {"choices": SAMPLERS}),
            f"{name}_t2i_scheduler": FakeInfo("Simple", "txt2img Scheduler", None, lambda: {"choices": SCHEDULERS}),
            f"{name}_i2i_sampler": FakeInfo("Euler", "img2img Sampler", None, lambda: {"choices": SAMPLERS}),
            f"{name}_i2i_scheduler": FakeInfo("Simple", "img2img Scheduler", None, lambda: {"choices": SCHEDULERS}),
        })
        for key in ("t2i_step", "t2i_hr_step", "i2i_step"):
            labels[f"{name}_{key}"] = FakeInfo(20, key, None, {"minimum": 0, "maximum": 150, "step": 1})
        for key in ("t2i_cfg", "t2i_hr_cfg", "i2i_cfg"):
            labels[f"{name}_{key}"] = FakeInfo(4.0, key, None, {"minimum": 0, "maximum": 24, "step": 0.5})
        if name in shift or name in distill:
            value = abs(shift.get(name, distill.get(name, 3.0)))
            for key in ("t2i_dcfg", "t2i_hr_dcfg", "i2i_dcfg"):
                labels[f"{name}_{key}"] = FakeInfo(value, key, None, {"minimum": 1, "maximum": 24, "step": 0.5})
        frames = 16 if name == "wan" else 1
        batch = {"minimum": 1, "maximum": frames * 15 + 1, "step": frames} if frames > 1 else \
            {"minimum": 1, "maximum": 8, "step": 1}
        for key in ("t2i_batch_size", "i2i_batch_size"):
            labels[f"{name}_{key}"] = FakeInfo(1, key, None, dict(batch))
        for key in ("t2i_width", "i2i_width", "t2i_height", "i2i_height"):
            labels[f"{name}_{key}"] = FakeInfo(0, key, None, {"minimum": 0, "maximum": 2048, "step": 64})
    return labels


def make_host(labels=None, data=None, presets=("sd", "xl", "flux", "anima", "wan"), tmp=None, **overrides):
    opts = FakeOpts(labels if labels is not None else mini_register(presets), data)
    values = dict(
        opts=opts,
        config_filename=str(Path(tmp or tempfile.gettempdir()) / "config.json"),
        presets=list(presets),
        checkpoint_known=lambda name: name in CHECKPOINTS,
        module_paths=dict(MODULES),
        dtypes=list(DTYPES),
        samplers=list(SAMPLERS),
        schedulers=list(SCHEDULERS),
        frozen=False,
    )
    values.update(overrides)
    return sd.Host(**values)


def ui_values(**changes):
    """22 UI 값 — FIELDS 순서(체크포인트, VAE/TE, Low Bits, 스텝 3, 샘플러 2, 스케줄러 2, 크기 4, CFG 3, dcfg 3, 배치 2)."""
    values = {
        "checkpoint": "anima-preview2.safetensors [1234abcd]",
        "modules": ["qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors"],
        "dtype": "Automatic",
        "t2i_step": 32, "t2i_hr_step": 0, "i2i_step": 28,
        "t2i_sampler": "ER SDE (Tunable)", "i2i_sampler": "Euler a",
        "t2i_scheduler": "Flow Cosmos rho7", "i2i_scheduler": "Beta",
        "t2i_width": 832, "i2i_width": 1024, "t2i_height": 1216, "i2i_height": 1024,
        "t2i_cfg": 4.5, "t2i_hr_cfg": 4.0, "i2i_cfg": 5.0,
        "t2i_dcfg": 3.0, "t2i_hr_dcfg": 2.5, "i2i_dcfg": 3.5,
        "t2i_batch_size": 2, "i2i_batch_size": 1,
    }
    values.update(changes)
    return [values[name] for name in (
        "checkpoint", "modules", "dtype", "t2i_step", "t2i_hr_step", "i2i_step", "t2i_sampler", "i2i_sampler",
        "t2i_scheduler", "i2i_scheduler", "t2i_width", "i2i_width", "t2i_height", "i2i_height", "t2i_cfg",
        "t2i_hr_cfg", "i2i_cfg", "t2i_dcfg", "t2i_hr_dcfg", "i2i_dcfg", "t2i_batch_size", "i2i_batch_size")]


def sizes(preset="anima", value=512):
    """너비·높이 넷을 0 보다 크게 — presets.register 의 기본값 0(= 덮어쓰지 않음)이면 버튼이 그대로 두는 칸들."""
    return {f"{preset}_{key}": value for key in ("t2i_width", "i2i_width", "t2i_height", "i2i_height")}


SKIP_TEMPLATES = ("{p}_t2i_step", "{p}_t2i_hr_step", "{p}_i2i_step", "{p}_t2i_width", "{p}_i2i_width",
                  "{p}_t2i_height", "{p}_i2i_height", "{p}_t2i_cfg", "{p}_t2i_hr_cfg", "{p}_i2i_cfg")


class FieldTableTests(unittest.TestCase):
    def test_twenty_two_fields_and_nineteen_lookups(self):
        self.assertEqual(len(sd.FIELDS), 22)
        self.assertEqual(len(sd.UI_COMPONENTS), 19)
        keys = [template.format(p="anima") for template, _kind, _label in sd.FIELDS]
        self.assertEqual(len(set(keys)), 22)
        self.assertEqual(keys[:3], ["forge_checkpoint_anima", "forge_additional_modules_anima",
                                    "forge_unet_storage_dtype_anima"])
        self.assertEqual(keys[-1], "anima_i2i_batch_size")
        # every lookup's tab matches the key's t2i/i2i half (the three model fields come from the quick settings row)
        for (template, _kind, label), (tab, _name) in zip(sd.FIELDS[3:], sd.UI_COMPONENTS):
            self.assertEqual(label.split(" ", 1)[0], tab, template)
            self.assertIn("_t2i_" if tab == "txt2img" else "_i2i_", template)


class SavePresetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def host(self, **kw):
        return make_host(tmp=self.tmp.name, **kw)

    def test_writes_the_active_preset_only_in_the_stored_shapes(self):
        host = self.host(data={"sd_t2i_step": 30, "xl_t2i_sampler": "DPM++ 2M", "anima_t2i_step": 28, **sizes()})
        def others(data):
            return {k: v for k, v in data.items() if not (k.startswith("anima_") or k.endswith("_anima"))}

        before_other = others(host.opts.data)
        payload = sd.save_preset("req-1", "anima", ui_values(), host)
        self.assertTrue(payload["ok"], payload)
        data = host.opts.data
        self.assertEqual(data["forge_checkpoint_anima"], "anima-preview2.safetensors [1234abcd]")
        self.assertEqual(data["forge_additional_modules_anima"], sorted([
            MODULES["qwen_image_vae.safetensors"], MODULES["qwen_3_06b_base.safetensors"]]))
        self.assertEqual(data["forge_unet_storage_dtype_anima"], "Automatic")
        self.assertEqual((data["anima_t2i_step"], data["anima_t2i_hr_step"], data["anima_i2i_step"]), (32, 0, 28))
        for key in ("anima_t2i_step", "anima_t2i_width", "anima_t2i_batch_size"):
            self.assertIs(type(data[key]), int, key)
        for key in ("anima_t2i_cfg", "anima_t2i_dcfg", "anima_i2i_dcfg"):
            self.assertIs(type(data[key]), float, key)
        self.assertEqual(data["anima_t2i_sampler"], "ER SDE (Tunable)")
        self.assertEqual(data["anima_t2i_scheduler"], "Flow Cosmos rho7")
        self.assertEqual((data["anima_t2i_width"], data["anima_t2i_height"]), (832, 1216))
        self.assertEqual((data["anima_t2i_hr_dcfg"], data["anima_i2i_batch_size"]), (2.5, 1))
        self.assertEqual(others(data), before_other)
        self.assertEqual(host.opts.saved, [host.config_filename])
        with open(host.config_filename, encoding="utf-8") as file:
            self.assertEqual(json.load(file), data)
        self.assertEqual(payload["request"], "req-1")
        self.assertEqual(payload["preset"], "anima")
        self.assertEqual(payload["written"], 22)
        self.assertEqual(payload["skipped"], [])
        self.assertEqual(payload["kept"], [])

    def test_counts_only_the_values_that_changed(self):
        host = self.host(data=sizes())
        first = sd.save_preset("a", "anima", ui_values(), host)
        # four of the 22 UI values equal the option defaults (Low Bits Automatic, Hires CFG 4.0, Shift 3.0, i2i batch 1)
        unchanged = {c["key"] for c in first["changes"]} ^ {t.format(p="anima") for t, _k, _l in sd.FIELDS}
        self.assertEqual(unchanged, {"forge_unet_storage_dtype_anima", "anima_t2i_hr_cfg", "anima_t2i_dcfg",
                                     "anima_i2i_batch_size"})
        self.assertEqual((first["changed"], first["written"]), (18, 22))
        again = sd.save_preset("b", "anima", ui_values(), host)
        self.assertTrue(again["ok"])
        self.assertEqual((again["changed"], again["changes"]), (0, []))
        third = sd.save_preset("c", "anima", ui_values(t2i_step=40, t2i_cfg=5.5), host)
        self.assertEqual(third["changed"], 2)
        self.assertEqual([(c["label"], c["old"], c["new"]) for c in third["changes"]],
                         [("txt2img Steps", 32, 40), ("txt2img CFG Scale", 4.5, 5.5)])
        json.loads(sd.to_json(third))

    def test_keys_forge_does_not_register_are_skipped(self):
        host = self.host(data=sizes("sd"))
        payload = sd.save_preset("r", "sd", ui_values(), host)    # sd: no Distilled CFG / Shift options
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["skipped"], ["sd_t2i_dcfg", "sd_t2i_hr_dcfg", "sd_i2i_dcfg"])
        self.assertEqual(payload["written"], 19)
        self.assertNotIn("sd_t2i_dcfg", host.opts.data)

    def test_restricted_options_are_skipped_not_forced(self):
        labels = mini_register()
        labels["anima_t2i_cfg"].do_not_save = True
        labels["anima_i2i_cfg"].component_args = {"visible": False}
        host = self.host(labels=labels)
        payload = sd.save_preset("r", "anima", ui_values(), host)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["skipped"], ["anima_t2i_cfg", "anima_i2i_cfg"])
        self.assertNotIn("anima_t2i_cfg", host.opts.data)

    def test_int_and_float_coercion(self):
        host = self.host(data=sizes())
        payload = sd.save_preset("r", "anima", ui_values(t2i_step=30.0, t2i_cfg=5, t2i_width=896.0), host)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual((host.opts.data["anima_t2i_step"], host.opts.data["anima_t2i_width"]), (30, 896))
        self.assertIs(type(host.opts.data["anima_t2i_step"]), int)
        self.assertEqual(host.opts.data["anima_t2i_cfg"], 5.0)
        self.assertIs(type(host.opts.data["anima_t2i_cfg"]), float)

    def test_bad_values_write_nothing(self):
        cases = {
            "t2i_step": [30.5, "30", True, None, math.nan, math.inf, 151, -1, 10 ** 400],
            "t2i_cfg": ["4.5", math.nan, -math.inf, 24.5, -0.5, False],
            "t2i_dcfg": [0.0, 25],                     # anima Shift slider: 1..24
            "t2i_batch_size": [0, 9, 1.5],
            "t2i_width": [2049, 64.5],
            "t2i_sampler": ["No such sampler", "", None, 3, ["Euler"]],
            "i2i_scheduler": ["karras", "", None],      # labels are case-sensitive in the dropdown
            "dtype": ["float16", None, ""],
            "checkpoint": ["missing.safetensors", "", None, 7],
            "modules": [["unknown.safetensors"], "qwen_image_vae.safetensors", [1], {"a": 1}],
        }
        for field_name, bad_values in cases.items():
            for bad in bad_values:
                with self.subTest(field=field_name, value=bad):
                    host = self.host(data={"anima_t2i_step": 28, **sizes()})
                    before = dict(host.opts.data)
                    payload = sd.save_preset("r", "anima", ui_values(**{field_name: bad}), host)
                    self.assertFalse(payload["ok"])
                    self.assertTrue(payload["error"])
                    self.assertEqual(host.opts.data, before)
                    self.assertEqual(host.opts.saved, [])
                    self.assertFalse(os.path.exists(host.config_filename))
                    json.loads(sd.to_json(payload))

    def test_error_names_the_field_and_counts_the_rest(self):
        host = self.host()
        payload = sd.save_preset("r", "anima", ui_values(t2i_step=200, i2i_cfg=99, t2i_sampler="nope"), host)
        self.assertFalse(payload["ok"])
        self.assertIn("txt2img Steps", payload["error"])
        self.assertIn("0~150", payload["error"])
        self.assertIn("(외 2개)", payload["error"])

    def test_video_preset_batch_uses_its_own_frame_range(self):
        host = self.host()
        self.assertTrue(sd.save_preset("r", "wan", ui_values(t2i_batch_size=33, i2i_batch_size=241), host)["ok"])
        self.assertEqual(host.opts.data["wan_t2i_batch_size"], 33)
        self.assertFalse(sd.save_preset("r", "wan", ui_values(t2i_batch_size=242), host)["ok"])
        self.assertFalse(sd.save_preset("r", "anima", ui_values(t2i_batch_size=33), host)["ok"])

    def test_unknown_preset_or_value_count(self):
        host = self.host()
        for preset in ("", None, "ANIMA", "anima ", "krea", 3):
            with self.subTest(preset=preset):
                payload = sd.save_preset("r", preset, ui_values(), host)
                self.assertFalse(payload["ok"])
                self.assertIn("UI Preset", payload["error"])
        for values in (ui_values()[:-1], ui_values() + [1], []):
            with self.subTest(count=len(values)):
                payload = sd.save_preset("r", "anima", values, host)
                self.assertFalse(payload["ok"])
                self.assertIn("22개여야", payload["error"])
        self.assertEqual(host.opts.data, {})

    def test_empty_or_missing_module_list_means_none(self):
        for value in ([], None):
            with self.subTest(value=value):
                host = self.host(data={"forge_additional_modules_anima": [MODULES["ae.safetensors"]]})
                self.assertTrue(sd.save_preset("r", "anima", ui_values(modules=value), host)["ok"])
                self.assertEqual(host.opts.data["forge_additional_modules_anima"], [])

    def test_module_paths_like_modules_change(self):
        host = self.host()
        names = ["qwen_3_06b_base.safetensors", r"C:\elsewhere\qwen_image_vae.safetensors", "ae.safetensors"]
        self.assertTrue(sd.save_preset("r", "anima", ui_values(modules=names), host)["ok"])
        self.assertEqual(host.opts.data["forge_additional_modules_anima"], sorted(MODULES.values()))

    def test_frozen_settings_write_nothing(self):
        host = self.host(frozen=True)
        payload = sd.save_preset("r", "anima", ui_values(), host)
        self.assertFalse(payload["ok"])
        self.assertIn("--freeze-settings", payload["error"])
        self.assertEqual((host.opts.data, host.opts.saved), ({}, []))

    def test_a_failed_save_rolls_the_memory_back(self):
        for failure in (PermissionError("locked"), AssertionError("changing settings is disabled")):
            with self.subTest(failure=type(failure).__name__):
                host = self.host(data={"anima_t2i_step": 28, "anima_t2i_sampler": "Euler"})
                before = dict(host.opts.data)
                host.opts.fail_save = failure
                payload = sd.save_preset("r", "anima", ui_values(), host)
                self.assertFalse(payload["ok"])
                self.assertIn(type(failure).__name__, payload["error"])
                self.assertEqual(host.opts.data, before)

    def test_an_option_that_refuses_the_value_rolls_back(self):
        host = self.host(data={"anima_t2i_step": 28})
        host.opts.refuse.add("anima_i2i_cfg")      # e.g. --hide-ui-dir-config / a frozen section in Forge
        payload = sd.save_preset("r", "anima", ui_values(), host)
        self.assertFalse(payload["ok"])
        self.assertIn("img2img CFG Scale", payload["error"])
        self.assertEqual(host.opts.data, {"anima_t2i_step": 28})
        self.assertEqual(host.opts.saved, [])

    def test_request_id_is_echoed_only_when_well_formed(self):
        host = self.host()
        for request, echoed in (("abc_123-XYZ", "abc_123-XYZ"), ("", ""), (None, ""), ("a b", ""),
                                ("x" * 65, ""), ("<script>", ""), (12, "")):
            with self.subTest(request=request):
                self.assertEqual(sd.save_preset(request, "anima", ui_values(), host)["request"], echoed)
                self.assertEqual(sd.save_preset(request, "nope", ui_values(), host)["request"], echoed)

    def test_old_values_that_are_not_json_still_report(self):
        # dcfg: on_preset_change always sets it (no "> 0" test); t2i_cfg: a NaN there is one Forge skips (nan > 0 is
        # False), so the button keeps it — both are reported without a bare NaN in the JSON
        host = self.host(data={"anima_t2i_dcfg": math.nan, "anima_t2i_cfg": math.nan})
        payload = sd.save_preset("r", "anima", ui_values(), host)
        self.assertTrue(payload["ok"])
        text = sd.to_json(payload)
        self.assertNotIn("NaN", text.replace('"nan"', ""))
        self.assertEqual([c["old"] for c in json.loads(text)["changes"] if c["key"] == "anima_t2i_dcfg"], ["nan"])
        self.assertEqual([k["value"] for k in json.loads(text)["kept"] if k["key"] == "anima_t2i_cfg"], ["nan"])
        self.assertTrue(math.isnan(host.opts.data["anima_t2i_cfg"]))

    # --- preset 0 = "do not override the screen value" (on_preset_change: v > 0 else gr.skip()) -----------------

    def test_skip_capable_fields_are_steps_sizes_and_cfg(self):
        self.assertEqual(sd.SKIP_IF_NOT_POSITIVE, frozenset(SKIP_TEMPLATES))
        self.assertLessEqual(sd.SKIP_IF_NOT_POSITIVE, {template for template, _kind, _label in sd.FIELDS})

    def test_forge_skips_is_on_preset_changes_test(self):
        for value in (0, 0.0, -0.0, -1, -0.5, -math.inf, math.nan, False):
            with self.subTest(value=value):
                self.assertTrue(sd.forge_skips(value))
        for value in (1, 0.5, 1e-9, 64, math.inf, True):
            with self.subTest(value=value):
                self.assertFalse(sd.forge_skips(value))
        for value in ("0", "", None, [], [0], {}):      # Forge's "v > 0" raises on these — not a skip
            with self.subTest(value=value):
                self.assertFalse(sd.forge_skips(value))

    def test_a_preset_zero_is_kept_not_replaced_by_the_screen_value(self):
        """The live test's finding: anima width/height 0 (Forge's default = keep the screen size) became 1024."""
        host = self.host(data={"anima_t2i_hr_step": 0, "anima_i2i_step": -5, "anima_i2i_cfg": 0.0,
                               "anima_t2i_step": 28, "anima_t2i_cfg": 4.0})
        before = dict(host.opts.data)
        payload = sd.save_preset("z", "anima", ui_values(t2i_hr_step=25), host)
        self.assertTrue(payload["ok"], payload)
        kept = ["anima_t2i_hr_step", "anima_i2i_step", "anima_t2i_width", "anima_i2i_width", "anima_t2i_height",
                "anima_i2i_height", "anima_i2i_cfg"]               # FIELDS order; the four sizes from the default 0
        self.assertEqual([k["key"] for k in payload["kept"]], kept)
        self.assertEqual([k["label"] for k in payload["kept"]],
                         ["txt2img Hires steps", "img2img Steps", "txt2img Width", "img2img Width", "txt2img Height",
                          "img2img Height", "img2img CFG Scale"])
        self.assertEqual([k["value"] for k in payload["kept"]], [0, -5, 0, 0, 0, 0, 0.0])
        for key in kept:
            with self.subTest(key=key):
                self.assertEqual(host.opts.data.get(key, "missing"), before.get(key, "missing"))
                self.assertNotIn(key, {c["key"] for c in payload["changes"]})
        self.assertNotIn("anima_t2i_width", host.opts.data)     # still Forge's default 0, not 832
        self.assertEqual(payload["written"], 22 - len(kept))
        # the positive ones are written as before (t2i steps 28 -> 32, t2i CFG 4.0 -> 4.5)
        self.assertEqual((host.opts.data["anima_t2i_step"], host.opts.data["anima_t2i_cfg"]), (32, 4.5))
        self.assertEqual(payload["changed"], 11)    # 15 written; Low Bits, Hires CFG, Shift, img2img batch = defaults
        self.assertEqual(host.opts.saved, [host.config_filename])
        # nothing to change next time either: a kept field never counts
        self.assertEqual(sd.save_preset("z2", "anima", ui_values(t2i_hr_step=30), host)["changed"], 0)

    def test_each_skip_field_alone(self):
        for template in SKIP_TEMPLATES:
            key = template.format(p="anima")
            for stored in (0, -1, math.nan):
                with self.subTest(key=key, stored=stored):
                    host = self.host(data={**sizes(), key: stored})
                    payload = sd.save_preset("one", "anima", ui_values(), host)
                    self.assertTrue(payload["ok"], payload)
                    self.assertEqual([k["key"] for k in payload["kept"]], [key])
                    self.assertEqual(payload["written"], 21)
                    stays = host.opts.data[key]
                    self.assertTrue(stays == stored or (math.isnan(stays) and math.isnan(stored)))
            with self.subTest(key=key, stored="positive"):
                host = self.host(data={**sizes(), key: 1})
                payload = sd.save_preset("one", "anima", ui_values(), host)
                self.assertEqual((payload["kept"], payload["written"]), ([], 22))
                self.assertIn(key, {c["key"] for c in payload["changes"]})

    def test_a_positive_preset_value_still_takes_the_screen_value_even_zero(self):
        host = self.host(data={"anima_t2i_hr_step": 20, **sizes()})
        payload = sd.save_preset("p", "anima", ui_values(t2i_hr_step=0, t2i_width=0), host)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual((host.opts.data["anima_t2i_hr_step"], host.opts.data["anima_t2i_width"]), (0, 0))
        self.assertEqual(payload["kept"], [])
        # from then on Forge keeps the screen value there, and so does the button
        again = sd.save_preset("p2", "anima", ui_values(t2i_hr_step=12, t2i_width=896), host)
        self.assertEqual([k["key"] for k in again["kept"]], ["anima_t2i_hr_step", "anima_t2i_width"])
        self.assertEqual((host.opts.data["anima_t2i_hr_step"], host.opts.data["anima_t2i_width"]), (0, 0))

    def test_other_fields_take_the_screen_value_even_over_zero(self):
        """Sampler / scheduler / dcfg / batch / checkpoint / modules / dtype: Forge always sets them, no 0 rule."""
        host = self.host(data={**sizes(), "anima_t2i_dcfg": 0.0, "anima_i2i_batch_size": 0, "anima_t2i_batch_size": -1,
                               "forge_checkpoint_anima": "", "forge_additional_modules_anima": [],
                               "forge_unet_storage_dtype_anima": "", "anima_t2i_sampler": "", "anima_i2i_scheduler": 0})
        payload = sd.save_preset("o", "anima", ui_values(i2i_batch_size=2), host)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["kept"], [])
        data = host.opts.data
        self.assertEqual((data["anima_t2i_dcfg"], data["anima_i2i_batch_size"], data["anima_t2i_batch_size"]),
                         (3.0, 2, 2))
        self.assertEqual((data["forge_checkpoint_anima"], data["forge_unet_storage_dtype_anima"]),
                         ("anima-preview2.safetensors [1234abcd]", "Automatic"))
        self.assertEqual((data["anima_t2i_sampler"], data["anima_i2i_scheduler"]), ("ER SDE (Tunable)", "Beta"))
        self.assertEqual(len(data["forge_additional_modules_anima"]), 2)

    def test_a_kept_field_is_not_range_checked(self):
        """It goes to ui-config only (Defaults Apply), so the preset slider's range has no say."""
        for screen in (2304, 64.5, None, "832"):
            with self.subTest(screen=screen):
                host = self.host()                              # widths: Forge's default 0
                payload = sd.save_preset("k", "anima", ui_values(t2i_width=screen), host)
                self.assertTrue(payload["ok"], payload)
                self.assertIn("anima_t2i_width", [k["key"] for k in payload["kept"]])
                self.assertNotIn("anima_t2i_width", host.opts.data)
        host = self.host(data=sizes())                          # a positive width is still checked
        self.assertIn("0~2048", sd.save_preset("k", "anima", ui_values(t2i_width=2304), host)["error"])

    def test_a_stored_value_forge_cannot_compare_is_replaced(self):
        """Forge's "v > 0" raises on a string / None there; the button writes the screen value as before (repairs it)."""
        host = self.host(data={**sizes(), "anima_t2i_step": "28", "anima_t2i_cfg": None})
        payload = sd.save_preset("s", "anima", ui_values(), host)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual((host.opts.data["anima_t2i_step"], host.opts.data["anima_t2i_cfg"]), (32, 4.5))
        self.assertEqual(payload["kept"], [])


# ---------------------------------------------------------------------------------------------------------
# Forge's real code (skipped without a Forge checkout)
# ---------------------------------------------------------------------------------------------------------

def _forge_presets_module():
    path = require_forge_file("modules_forge/presets.py")
    import importlib.util

    spec = importlib.util.spec_from_file_location("_forge_presets_for_save_defaults", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)       # imports only enum at module level; register() imports lazily
    return module


def forge_labels(presets_module):
    """presets.register(options_templates) with modules.options / modules.shared_items stubbed — the real keys."""
    options = types.ModuleType("modules.options")
    options.OptionInfo = FakeInfo
    options.OptionRow = FakeRow
    options.options_section = options_section
    shared_items = types.ModuleType("modules.shared_items")
    shared_items.list_samplers = lambda: [types.SimpleNamespace(name=name) for name in SAMPLERS]
    shared_items.list_schedulers = lambda: list(SCHEDULERS)
    package = types.ModuleType("modules")
    package.__path__ = []
    package.options = options
    package.shared_items = shared_items
    # shared_options.py keys main_entry's functions also set
    templates = {"forge_preset": FakeInfo("sd"), "sd_model_checkpoint": FakeInfo(None),
                 "forge_additional_modules": FakeInfo([]), "forge_unet_storage_dtype": FakeInfo("Automatic")}
    with mock.patch.dict(sys.modules, {"modules": package, "modules.options": options,
                                       "modules.shared_items": shared_items}):
        presets_module.register(templates)
    return templates


def _main_entry_tree():
    path = require_forge_file("modules_forge/main_entry.py")
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in modules_forge/main_entry.py")


def forge_functions(names, namespace):
    """main_entry.py 의 함수 정의를 그대로 실행(import 없이 — torch·backend 를 끌어오지 않는다)."""
    tree = _main_entry_tree()
    module = ast.Module(body=[_function(tree, name) for name in names], type_ignores=[])
    exec(compile(module, "modules_forge/main_entry.py", "exec"), namespace)
    return [namespace[name] for name in names]


class ForgeContractTests(unittest.TestCase):
    """The field table is main_entry's, in its order (Forge checkout needed)."""

    def test_lookups_are_forge_main_entrys_in_order(self):
        body = _function(_main_entry_tree(), "forge_main_entry")
        assigned = {}
        for node in body.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                    and getattr(node.value.func, "id", None) == "get_a1111_ui_component":
                assigned[node.targets[0].id] = tuple(arg.value for arg in node.value.args)
        targets = next(node for node in body.body if isinstance(node, ast.Assign)
                       and getattr(node.targets[0], "id", None) == "output_targets")
        names = [element.id for element in targets.value.elts]
        self.assertEqual(names[:3], ["ui_checkpoint", "ui_vae", "ui_forge_unet_dtype"])
        self.assertEqual(tuple(assigned[name] for name in names[3:]), sd.UI_COMPONENTS)

    def test_fields_are_on_preset_changes_keys_in_order(self):
        func = _function(_main_entry_tree(), "on_preset_change")
        returned = func.body[-1]
        self.assertIsInstance(returned, ast.Return)
        keys = []
        for element in returned.value.elts:
            for node in ast.walk(element):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "getattr" \
                        and isinstance(node.args[1], ast.JoinedStr):
                    keys.append("".join(part.value if isinstance(part, ast.Constant) else "{p}"
                                        for part in node.args[1].values))
                    break
        # the two batch updates read their keys in batch_args_t2i / batch_args_i2i above the return
        self.assertEqual(len(keys), 20)
        self.assertEqual(keys, [template for template, _kind, _label in sd.FIELDS[:20]])
        self.assertEqual([ast.unparse(element) for element in returned.value.elts[20:]],
                         ["gr.update(**batch_args_t2i)", "gr.update(**batch_args_i2i)"])
        read = {"".join(part.value if isinstance(part, ast.Constant) else "{p}" for part in node.args[1].values)
                for node in ast.walk(func) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "getattr"
                and len(node.args) > 1 and isinstance(node.args[1], ast.JoinedStr)}
        self.assertLessEqual({"{p}_t2i_batch_size", "{p}_i2i_batch_size"}, read)
        self.assertEqual([t for t, _k, _l in sd.FIELDS[20:]], ["{p}_t2i_batch_size", "{p}_i2i_batch_size"])

    def test_mini_registry_matches_forge_for_its_presets(self):
        real = forge_labels(_forge_presets_module())
        mini = mini_register()
        for preset in ("sd", "xl", "flux", "anima", "wan"):
            for template, kind, _label in sd.FIELDS:
                key = template.format(p=preset)
                with self.subTest(key=key):
                    self.assertEqual(key in mini, key in real)
                    if key in real and kind in ("int", "float"):
                        self.assertEqual(sd._slider_range(mini[key]), sd._slider_range(real[key]))


class ForgeRoundTripTests(unittest.TestCase):
    """Saved values come back through Forge's own on_preset_change as the same UI values (every preset)."""

    @classmethod
    def setUpClass(cls):
        cls.presets_module = _forge_presets_module()
        cls.labels = forge_labels(cls.presets_module)
        cls.preset_names = cls.presets_module.PresetArch.choices()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _forge(self, opts):
        import gradio as gr

        module_list = dict(MODULES)
        shared = types.SimpleNamespace(opts=opts, config_filename=str(Path(self.tmp.name) / "config.json"))
        namespace = {"os": os, "gr": gr, "shared": shared, "module_list": module_list,
                     "use_shift": self.presets_module.use_shift, "use_distill": self.presets_module.use_distill,
                     "is_video": self.presets_module.is_video,
                     "refresh_model_loading_parameters": lambda **kwargs: None}
        return forge_functions(["on_preset_change", "modules_change"], namespace)

    def test_every_preset_round_trips_through_on_preset_change(self):
        for preset in self.preset_names:
            with self.subTest(preset=preset):
                host = make_host(labels=self.labels, presets=self.preset_names, tmp=self.tmp.name, data=sizes(preset))
                on_preset_change, _modules_change = self._forge(host.opts)
                frames = self.presets_module.is_video(preset)
                values = ui_values(t2i_batch_size=1 + frames if frames > 1 else 3,
                                   i2i_batch_size=1 + 2 * frames if frames > 1 else 2)
                payload = sd.save_preset("rt", preset, values, host)
                self.assertTrue(payload["ok"], payload)
                updates = on_preset_change(preset)
                self.assertEqual(len(updates), 22)
                self.assertEqual(host.opts.data["forge_preset"], preset)
                has_dcfg = f"{preset}_t2i_dcfg" in self.labels
                for index, ((template, kind, label), value, update) in enumerate(zip(sd.FIELDS, values, updates)):
                    with self.subTest(field=label):
                        if kind == "modules":
                            self.assertEqual(sorted(update["value"]), sorted(value))
                        elif template == "{p}_t2i_hr_step" and value == 0:
                            self.assertNotIn("value", update)          # 0 = "keep the UI value" (gr.skip)
                        elif "dcfg" in template and not has_dcfg:
                            self.assertEqual(update["value"], 3.0)     # not saved: Forge's fallback, slider hidden
                            self.assertIs(update["visible"], False)
                        else:
                            self.assertEqual(update["value"], value)
                            if kind == "int":
                                self.assertIs(type(update["value"]), int)
                            if kind == "float":
                                self.assertIs(type(update["value"]), float)

    def test_a_fresh_preset_keeps_its_zero_sizes_and_forge_keeps_the_screen_size(self):
        """presets.register's width/height default is 0 for every preset: after a save Forge still skips them."""
        for preset in self.preset_names:
            with self.subTest(preset=preset):
                host = make_host(labels=self.labels, presets=self.preset_names, tmp=self.tmp.name)
                on_preset_change, _modules_change = self._forge(host.opts)
                frames = self.presets_module.is_video(preset)
                values = ui_values(t2i_batch_size=1 + frames if frames > 1 else 3,
                                   i2i_batch_size=1 + 2 * frames if frames > 1 else 2)
                payload = sd.save_preset("fresh", preset, values, host)
                self.assertTrue(payload["ok"], payload)
                size_keys = [f"{preset}_{k}" for k in ("t2i_width", "i2i_width", "t2i_height", "i2i_height")]
                self.assertEqual([k["key"] for k in payload["kept"]], size_keys)
                updates = on_preset_change(preset)
                for index, (template, _kind, label) in enumerate(sd.FIELDS):
                    if template.format(p=preset) in size_keys:
                        with self.subTest(field=label):
                            self.assertNotIn("value", updates[index])     # gr.skip(): the screen size stays

    def test_module_lists_are_stored_like_modules_change(self):
        for names in (["qwen_image_vae.safetensors", "qwen_3_06b_base.safetensors"], ["ae.safetensors"], []):
            with self.subTest(names=names):
                forge_opts = FakeOpts(self.labels, {"forge_additional_modules": ["x"]})
                _on_preset_change, modules_change = self._forge(forge_opts)
                modules_change(list(names), "anima", save=False, refresh=False)
                host = make_host(labels=self.labels, presets=self.preset_names, tmp=self.tmp.name)
                self.assertTrue(sd.save_preset("m", "anima", ui_values(modules=list(names)), host)["ok"])
                self.assertEqual(host.opts.data["forge_additional_modules_anima"],
                                 forge_opts.data["forge_additional_modules_anima"])

    def test_other_presets_are_untouched_and_forge_preset_is_not_written(self):
        host = make_host(labels=self.labels, presets=self.preset_names, tmp=self.tmp.name,
                         data={"forge_preset": "xl", "xl_t2i_step": 24, "sd_t2i_cfg": 6.0})
        self.assertTrue(sd.save_preset("o", "anima", ui_values(), host)["ok"])
        self.assertEqual(host.opts.data["forge_preset"], "xl")
        for key, value in host.opts.data.items():
            if not (key.startswith("anima_") or key.endswith("_anima")):
                self.assertIn(key, {"forge_preset", "xl_t2i_step", "sd_t2i_cfg"}, key)
        self.assertEqual((host.opts.data["xl_t2i_step"], host.opts.data["sd_t2i_cfg"]), (24, 6.0))


class ForgeSkipFieldTests(unittest.TestCase):
    """Each of the ten fields against Forge's real on_preset_change: the button keeps exactly what Forge skips."""

    @classmethod
    def setUpClass(cls):
        cls.presets_module = _forge_presets_module()
        cls.labels = forge_labels(cls.presets_module)
        cls.preset_names = cls.presets_module.PresetArch.choices()

    def setUp(self):
        import gradio as gr

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.shared = types.SimpleNamespace(opts=None, config_filename=str(Path(self.tmp.name) / "config.json"))
        namespace = {"os": os, "gr": gr, "shared": self.shared, "use_shift": self.presets_module.use_shift,
                     "use_distill": self.presets_module.use_distill, "is_video": self.presets_module.is_video}
        (self.forge_on_preset_change,) = forge_functions(["on_preset_change"], namespace)

    def _on_preset_change(self, opts):
        self.shared.opts = opts                 # Forge's function reads shared.opts at call time
        return self.forge_on_preset_change

    def test_the_ten_fields_are_the_ones_on_preset_change_skips_at_zero(self):
        returned = _function(_main_entry_tree(), "on_preset_change").body[-1]
        skipping = []
        for index, element in enumerate(returned.value.elts):
            if not isinstance(element, ast.IfExp):
                continue
            self.assertEqual(ast.unparse(element.orelse), "gr.skip()")
            test = element.test        # (v := getattr(shared.opts, f"{preset}_...", default)) > 0
            self.assertIsInstance(test, ast.Compare)
            self.assertEqual([type(op) for op in test.ops], [ast.Gt])
            self.assertEqual(ast.unparse(test.comparators[0]), "0")
            self.assertEqual(ast.unparse(element.body), "gr.update(value=v)")
            call = test.left.value
            key = "".join(p.value if isinstance(p, ast.Constant) else "{p}" for p in call.args[1].values)
            self.assertEqual(key, sd.FIELDS[index][0])
            skipping.append(key)
        self.assertEqual(frozenset(skipping), sd.SKIP_IF_NOT_POSITIVE)
        self.assertEqual(len(skipping), 10)

    def test_forge_skips_matches_forge_for_each_field_and_value(self):
        indexes = {template: i for i, (template, _k, _l) in enumerate(sd.FIELDS)}
        for template in SKIP_TEMPLATES:
            key = template.format(p="anima")
            for stored in (0, 0.0, -1, -0.5, math.nan, -math.inf, False, 1, 0.5, 20, 1024, math.inf, True):
                with self.subTest(key=key, stored=stored):
                    opts = FakeOpts(self.labels, {key: stored})
                    update = self._on_preset_change(opts)("anima")[indexes[template]]
                    self.assertEqual("value" not in update, sd.forge_skips(stored))
            for stored in ("0", None):
                with self.subTest(key=key, stored=stored):
                    with self.assertRaises(TypeError):          # Forge cannot read it at all; not a skip for us
                        self._on_preset_change(FakeOpts(self.labels, {key: stored}))("anima")
                    self.assertFalse(sd.forge_skips(stored))

    def test_after_a_save_forge_still_keeps_the_screen_value_where_the_preset_said_zero(self):
        indexes = {template: i for i, (template, _k, _l) in enumerate(sd.FIELDS)}
        screen = dict(zip((t for t, _k, _l in sd.FIELDS), ui_values(t2i_hr_step=25)))
        for preset in self.preset_names:
            frames = self.presets_module.is_video(preset)
            batch = {"t2i_batch_size": 1 + frames, "i2i_batch_size": 1 + frames} if frames > 1 else {}
            values = ui_values(t2i_hr_step=25, **batch)
            for template in SKIP_TEMPLATES:
                key = template.format(p=preset)
                for stored in (0, 64):
                    with self.subTest(preset=preset, key=key, stored=stored):
                        host = make_host(labels=self.labels, presets=self.preset_names, tmp=self.tmp.name,
                                         data={**sizes(preset), key: stored})
                        self.assertTrue(sd.save_preset("rt", preset, values, host)["ok"])
                        update = self._on_preset_change(host.opts)(preset)[indexes[template]]
                        if stored == 0:
                            self.assertEqual(host.opts.data[key], 0)
                            self.assertNotIn("value", update)            # the screen (ui-config) value stays
                        else:
                            self.assertEqual(update["value"], screen[template])


if __name__ == "__main__":
    unittest.main()
