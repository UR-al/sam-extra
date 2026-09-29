"""Anima VAE DeGrid 스크립트 — 실행 순서(모든 postprocess_image 뒤), 내부 패스 건너뛰기, 모델이 없을 때 건너뛰기,
infotext 기록·붙여 넣기, API 인자, Extras 스크립트, 설정 등록.

순서는 Forge 코드로 확인한다(확장 옆에 Forge 가 있을 때만 — CI 에서는 건너뜀):
- ``modules/processing.py`` 이미지 루프에서 ``postprocess_image_after_composite`` 가 모든 ``postprocess_image`` ·색 보정·
  ``apply_overlay`` 뒤, 저장·``infotext(i)`` 앞이고, 조건은 ``p.scripts is not None`` 하나뿐(모든 이미지).
- ``modules/scripts.py`` 는 메서드를 덮어쓴 스크립트만 그 콜백 목록에 넣는다 — DeGrid 는 postprocess_image 목록에 없다.
- ``metadata.ini`` 의 콜백 순서를 Forge 의 ``sort_callbacks``·``ExtensionMetadata``·``topological_sort``(AST 로 꺼냄)로
  돌려, 메인 탭에 켠 Extras Upscale 보다 DeGrid 가 앞인지 본다.
"""
from __future__ import annotations

import ast
import configparser
import dataclasses
import importlib.util
import os
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import ui_vae_degrid as uvd  # noqa: E402
from sam3ext import vae_degrid as vd  # noqa: E402
from sam3ext import vae_degrid_models as vdm  # noqa: E402
from sam3ext import vae_degrid_runtime as vdr  # noqa: E402

FORGE_ROOT = ROOT.parents[1]
FORGE_PROCESSING = FORGE_ROOT / "modules" / "processing.py"
FORGE_SCRIPTS = FORGE_ROOT / "modules" / "scripts.py"
FORGE_CALLBACKS = FORGE_ROOT / "modules" / "script_callbacks.py"
FORGE_UTIL = FORGE_ROOT / "modules" / "util.py"
FORGE_EXTENSIONS = FORGE_ROOT / "modules" / "extensions.py"
EXTENSION_NAME = "forge_sam3_extension"
UPSCALE_IN_MAIN_UI = (
    "base/postprocessing_upscale.py/script_postprocess_image_after_composite/ScriptPostprocessingForMainUI"
)


class _SysModules:
    """테스트 동안만 sys.modules 항목을 바꾼다."""

    def __init__(self, modules: dict):
        self.modules = modules
        self.saved: dict = {}

    def __enter__(self):
        for name, value in self.modules.items():
            self.saved[name] = sys.modules.get(name)
            sys.modules[name] = value
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _Script:
    def elem_id(self, item_id):
        return f"script_txt2img_degrid_{item_id}"

    def postprocess_image(self, p, pp, *args):
        pass

    def postprocess_image_after_composite(self, p, pp, *args):
        pass


class _ScriptPostprocessing:
    name = None
    order = 1000

    def process(self, pp, **args):
        pass


class _PostprocessedImage:
    def __init__(self, image):
        self.image = image
        self.info = {}


def _load_scripts():
    settings = []
    modules_stub = types.ModuleType("modules")
    modules_stub.scripts = types.SimpleNamespace(Script=_Script, AlwaysVisible=object())
    modules_stub.script_callbacks = types.SimpleNamespace(on_ui_settings=settings.append)
    modules_stub.scripts_postprocessing = types.SimpleNamespace(
        ScriptPostprocessing=_ScriptPostprocessing, PostprocessedImage=_PostprocessedImage)
    loaded = {}
    with _SysModules({"modules": modules_stub}):
        for stem in ("anima_vae_degrid", "anima_vae_degrid_extras"):
            spec = importlib.util.spec_from_file_location(f"_test_{stem}", ROOT / "scripts" / f"{stem}.py")
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            loaded[stem] = module
    return loaded["anima_vae_degrid"], loaded["anima_vae_degrid_extras"], settings


MOD, EXTRAS, SETTINGS = _load_scripts()
ENTRY = vdm.ModelEntry("qwenVAEDegridNafnet_v11", "/m/ESRGAN/qwenVAEDegridNafnet_v11.safetensors")


class _FakeRuntime:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)
        self.released = 0

    def run(self, image, entry, *, mode, strength, tile):
        self.calls.append((entry.name, mode, strength, tile))
        if len(self.calls) in self.fail_on:
            raise RuntimeError("CUDA out of memory")
        out = Image.new("RGB", image.size, (1, 2, 3))
        return vdr.DegridOutcome(out, entry.name, mode, strength, tile, tile, "cpu", "fp32", 0.01)

    def release(self):
        self.released += 1


def _p(**kw):
    p = types.SimpleNamespace(extra_generation_params={})
    for key, value in kw.items():
        setattr(p, key, value)
    return p


class OrderingTests(unittest.TestCase):
    def test_runs_in_after_composite_not_in_postprocess_image(self):
        cls = MOD.AnimaVaeDegrid
        # Forge create_ordered_callbacks_list 는 덮어쓰지 않은 메서드의 스크립트를 그 목록에서 뺀다
        self.assertIs(cls.postprocess_image, _Script.postprocess_image)
        self.assertIsNot(cls.postprocess_image_after_composite, _Script.postprocess_image_after_composite)

    def test_metadata_section_names_this_callback(self):
        config = configparser.ConfigParser()
        config.read(ROOT / "metadata.ini", encoding="utf-8")
        section = (f"callbacks/{EXTENSION_NAME}/{Path(MOD.__file__).name}/script_postprocess_image_after_composite/"
                   f"{MOD.AnimaVaeDegrid.__name__}")
        self.assertIn(section, config.sections())
        self.assertIn(UPSCALE_IN_MAIN_UI, config.get(section, "Before").split())

    @unittest.skipUnless(FORGE_PROCESSING.is_file(), "Forge 의 modules/processing.py 가 확장 옆에 없음")
    def test_forge_calls_after_composite_after_every_postprocess_image_for_every_image(self):
        tree = ast.parse(FORGE_PROCESSING.read_text(encoding="utf-8"))
        inner = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process_images_inner")
        loop = next(
            n for n in ast.walk(inner)
            if isinstance(n, ast.For) and isinstance(n.iter, ast.Call) and getattr(n.iter.func, "id", "") == "enumerate"
            and "x_samples_ddim" in ast.unparse(n.iter)
        )
        calls = []   # (이름, 줄, 감싼 if 조건들, 소스)

        def walk(statements, ifs):
            for statement in statements:
                if isinstance(statement, ast.If):
                    walk(statement.body, ifs + [ast.unparse(statement.test)])
                    walk(statement.orelse, ifs + ["not " + ast.unparse(statement.test)])
                elif isinstance(getattr(statement, "body", None), list):   # for/with/try — 조건이 아니다
                    walk(statement.body, ifs)
                    walk(getattr(statement, "orelse", []), ifs)
                    walk(getattr(statement, "finalbody", []), ifs)
                    for handler in getattr(statement, "handlers", []):
                        walk(handler.body, ifs)
                else:
                    for node in ast.walk(statement):
                        if isinstance(node, ast.Call):
                            func = node.func
                            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                            calls.append((name, node.lineno, tuple(ifs), ast.unparse(node)))

        walk(loop.body, [])
        after = [c for c in calls if c[0] == "postprocess_image_after_composite"]
        self.assertEqual(len(after), 1)
        _, after_line, after_ifs, _ = after[0]
        self.assertEqual(after_ifs, ("p.scripts is not None",))            # 모든 이미지(조건은 스크립트 유무뿐)
        before = [c for c in calls if c[0] in ("postprocess_image", "postprocess_maskoverlay", "apply_color_correction",
                                               "apply_overlay")]
        self.assertEqual({c[0] for c in before},
                         {"postprocess_image", "postprocess_maskoverlay", "apply_color_correction", "apply_overlay"})
        for name, line, _ifs, _src in before:
            with self.subTest(call=name):
                self.assertLess(line, after_line)
        saves = [c for c in calls if c[0] == "save_image" and "outpath_samples" in c[3] and "suffix" not in c[3]]
        self.assertTrue(saves and all(line > after_line for _n, line, _i, _s in saves))
        self.assertTrue(any(c[0] == "infotext" and c[1] > after_line for c in calls))

    @unittest.skipUnless(FORGE_SCRIPTS.is_file(), "Forge 의 modules/scripts.py 가 확장 옆에 없음")
    def test_forge_runner_uses_its_own_ordered_list_and_skips_non_overriders(self):
        source = FORGE_SCRIPTS.read_text(encoding="utf-8")
        tree = ast.parse(source)
        runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScriptRunner")
        method = next(n for n in runner.body if isinstance(n, ast.FunctionDef)
                      and n.name == "postprocess_image_after_composite")
        loop = next(n for n in ast.walk(method) if isinstance(n, ast.For))
        self.assertEqual(ast.unparse(loop.iter), "self.ordered_scripts('postprocess_image_after_composite')")
        self.assertIn("getattr(script.__class__, method_name, None) == getattr(Script, method_name, None)", source)

    @unittest.skipUnless(
        FORGE_CALLBACKS.is_file() and FORGE_UTIL.is_file() and FORGE_EXTENSIONS.is_file(),
        "Forge 의 modules/script_callbacks.py·util.py·extensions.py 가 확장 옆에 없음",
    )
    def test_forge_sort_callbacks_puts_degrid_before_main_ui_upscale(self):
        def extract(path, names, namespace):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
            self.assertEqual({n.name for n in nodes}, set(names))
            exec(compile(ast.Module(nodes, []), str(path), "exec"), namespace)
            return namespace

        reports = []
        ext_ns = extract(FORGE_EXTENSIONS, ("CallbackOrderInfo", "ExtensionMetadata"), {
            "configparser": configparser, "dataclasses": dataclasses, "os": os, "re": re,
            "errors": types.SimpleNamespace(report=lambda *a, **k: reports.append(a)), "loaded_extensions": {},
        })
        util_ns = extract(FORGE_UTIL, ("topological_sort",), {})
        metadata = ext_ns["ExtensionMetadata"](str(ROOT), EXTENSION_NAME)
        sc_ns = extract(FORGE_CALLBACKS, ("sort_callbacks",), {
            "extensions": types.SimpleNamespace(extensions=[types.SimpleNamespace(metadata=metadata)]),
            "util": types.SimpleNamespace(topological_sort=util_ns["topological_sort"]),
            "shared": types.SimpleNamespace(opts=types.SimpleNamespace()),
        })
        ours = (f"{EXTENSION_NAME}/{Path(MOD.__file__).name}/script_postprocess_image_after_composite/"
                f"{MOD.AnimaVaeDegrid.__name__}")
        other = "some-extension/other.py/script_postprocess_image_after_composite/Other"
        category = "script_postprocess_image_after_composite"
        # always-on 목록 순서: 메인 탭 Extras(맨 앞) → 다른 확장 → 우리
        callbacks = [types.SimpleNamespace(name=n) for n in (UPSCALE_IN_MAIN_UI, other, ours)]
        order = [c.name for c in sc_ns["sort_callbacks"](category, callbacks)]
        self.assertLess(order.index(ours), order.index(UPSCALE_IN_MAIN_UI))
        self.assertEqual(sorted(order), sorted(c.name for c in callbacks))
        # Upscale 을 켜지 않았으면 순서를 건드리지 않는다
        callbacks = [types.SimpleNamespace(name=n) for n in (other, ours)]
        self.assertEqual([c.name for c in sc_ns["sort_callbacks"](category, callbacks)], [other, ours])
        self.assertEqual(reports, [])


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.script = MOD.AnimaVaeDegrid()
        self.logs = []
        self.patches = [
            mock.patch.object(vdm, "discover", return_value=[ENTRY]),
            mock.patch.object(vdr, "log", side_effect=self.logs.append),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()

    def _run(self, p, images, runtime, args=(True, "", "Full (전체)", 1.0, 512)):
        self.script.process(p, *args)
        outs = []
        with mock.patch.object(vdr, "shared_runtime", return_value=runtime):
            for image in images:
                pp = types.SimpleNamespace(image=image)
                self.script.postprocess_image_after_composite(p, pp, *args)
                outs.append((pp.image, dict(p.extra_generation_params)))
        return outs

    def test_disabled_does_nothing(self):
        p = _p()
        runtime = _FakeRuntime()
        image = Image.new("RGB", (8, 8))
        [(out, params)] = self._run(p, [image], runtime, args=(False, "", "Full", 1.0, 512))
        self.assertIs(out, image)
        self.assertEqual(params, {})
        self.assertEqual(runtime.calls, [])

    def test_inner_passes_are_skipped(self):
        for flag in ("_sam3_inner", "_ad_inner"):
            with self.subTest(flag=flag):
                p = _p(**{flag: True})
                runtime = _FakeRuntime()
                image = Image.new("RGB", (8, 8))
                [(out, params)] = self._run(p, [image], runtime)
                self.assertIs(out, image)
                self.assertEqual(runtime.calls, [])
                self.assertEqual(params, {})

    def test_success_replaces_the_image_and_records_infotext(self):
        p = _p()
        runtime = _FakeRuntime()
        [(out, params)] = self._run(p, [Image.new("RGB", (8, 8))], runtime,
                                    args=(True, "qwenVAEDegridNafnet_v11", "Dark Pixels Mainly", 0.75, 256))
        self.assertEqual(out.getpixel((0, 0)), (1, 2, 3))
        self.assertEqual(runtime.calls, [("qwenVAEDegridNafnet_v11", vd.MODE_DARK, 0.75, 256)])
        self.assertEqual(params, {
            uvd.KEY_MODEL: "qwenVAEDegridNafnet_v11", uvd.KEY_MODE: "Dark Pixels Mainly",
            uvd.KEY_STRENGTH: "0.75", uvd.KEY_TILE: 256,
        })

    def test_missing_model_is_skipped_with_a_log_and_an_error_key(self):
        p = _p()
        runtime = _FakeRuntime()
        image = Image.new("RGB", (8, 8))
        with mock.patch.object(vdm, "discover", return_value=[]):
            [(out, params)] = self._run(p, [image], runtime)
        self.assertIs(out, image)
        self.assertEqual(runtime.calls, [])
        self.assertEqual(params, {uvd.KEY_ERROR: "model not found: auto"})
        self.assertTrue(any("찾지 못해" in line for line in self.logs))

    def test_unknown_model_name_is_skipped(self):
        p = _p()
        [(_, params)] = self._run(p, [Image.new("RGB", (8, 8))], _FakeRuntime(),
                                  args=(True, "4x-UltraSharp", "Full", 1.0, 512))
        self.assertEqual(params, {uvd.KEY_ERROR: "model not found: 4x-UltraSharp"})

    def test_failure_keeps_the_image_and_batch_infotext_follows_each_image(self):
        p = _p()
        runtime = _FakeRuntime(fail_on={2})
        images = [Image.new("RGB", (8, 8), (9, 9, 9)) for _ in range(3)]
        outs = self._run(p, images, runtime)
        (first, params1), (second, params2), (third, params3) = outs
        self.assertEqual(first.getpixel((0, 0)), (1, 2, 3))
        self.assertIs(second, images[1])                 # 실패한 장은 원본 그대로 저장
        self.assertIn(uvd.KEY_MODEL, params1)
        self.assertNotIn(uvd.KEY_ERROR, params1)
        self.assertEqual(set(params2), {uvd.KEY_ERROR})
        self.assertIn("CUDA out of memory", params2[uvd.KEY_ERROR])
        self.assertIn(uvd.KEY_MODEL, params3)
        self.assertNotIn(uvd.KEY_ERROR, params3)
        self.assertEqual(runtime.released, 1)

    def test_api_positional_and_dict_args(self):
        for args in ((True,), [{"enabled": True}], [{"enabled": "true", "mode": "bright", "strength": 2, "tile": 0}]):
            with self.subTest(args=args):
                p = _p()
                self.script.process(p, *args)
                cfg, entry = p._sam3_degrid
                self.assertTrue(cfg.enabled)
                self.assertIs(entry, ENTRY)
        p = _p()
        self.script.process(p, *[{"enabled": True, "mode": "bright", "strength": 2, "tile": 0}])
        cfg, _ = p._sam3_degrid
        self.assertEqual((cfg.mode, cfg.strength, cfg.tile), (vd.MODE_BRIGHT, 1.5, 0))


class ArgsAndInfotextTests(unittest.TestCase):
    def test_arg_names_are_fixed(self):
        self.assertEqual(uvd.ARG_NAMES, ("enabled", "model", "mode", "strength", "tile"))

    def test_coerce_args(self):
        self.assertEqual(uvd.coerce_args([]), uvd.DegridArgs())
        self.assertEqual(uvd.coerce_args([True, "None", "x", "abc", "64"]),
                         uvd.DegridArgs(True, "", vd.MODE_FULL, 1.0, 128))
        self.assertEqual(uvd.coerce_args([{"enabled": 1, "model": " m ", "mode": "Dark Pixels Mainly (어두운 점 위주)"}]),
                         uvd.DegridArgs(True, "m", vd.MODE_DARK, 1.0, 512))

    def test_infotext_round_trip(self):
        items = uvd.infotext_items("qwenVAEDegridNafnet_v11", vd.MODE_BRIGHT, 1.25, 0)
        params = {key: str(value) for key, value in items.items()}   # Forge infotext 는 문자열로 돌아온다
        choices = ["NAFNet-QwenVAE-DeGrid", "qwenVAEDegridNafnet_v11"]
        paste_model = uvd.make_paste_model(lambda: choices)
        self.assertTrue(uvd.paste_enabled(params))
        self.assertEqual(paste_model(params), "qwenVAEDegridNafnet_v11")
        self.assertEqual(uvd.paste_mode(params), "Bright Pixels Mainly (밝은 점 위주)")
        self.assertEqual(uvd.paste_strength(params), 1.25)
        self.assertEqual(uvd.paste_tile(params), 0)
        # 다른 PC 에 그 모델이 없으면 드롭다운은 건드리지 않는다
        self.assertIsNone(uvd.make_paste_model(lambda: ["other"])(params))
        # DeGrid 기록이 없는 이미지를 붙여 넣으면 끈다(다른 칸은 건드리지 않음)
        none = {"Steps": "20"}
        self.assertFalse(uvd.paste_enabled(none))
        for fn in (paste_model, uvd.paste_mode, uvd.paste_strength, uvd.paste_tile):
            self.assertIsNone(fn(none))

    def test_every_mode_label_is_a_ui_choice(self):
        for mode in vd.MODES:
            self.assertEqual(vd.normalize_mode(uvd.mode_choice(mode)), mode)
            self.assertEqual(vd.normalize_mode(vd.MODE_LABELS[mode]), mode)


class UiTests(unittest.TestCase):
    def test_ui_returns_all_arguments_in_order_with_defaults(self):
        import gradio as gr

        script = MOD.AnimaVaeDegrid()
        with mock.patch.object(vdm, "discover", return_value=[ENTRY]), gr.Blocks():
            components = script.ui(False)
        self.assertEqual(len(components), len(uvd.ARG_NAMES))
        self.assertEqual([c for c, _ in script.infotext_fields], components)
        enabled, model, mode, strength, tile = components
        self.assertIs(enabled.value, False)
        self.assertEqual(model.value, "qwenVAEDegridNafnet_v11")
        self.assertEqual(mode.value, uvd.MODE_CHOICES[0])
        self.assertEqual(strength.value, 1.0)
        self.assertEqual(tile.value, 512)
        self.assertEqual(script.title(), "Anima VAE DeGrid (NAFNet)")

    def test_empty_model_folder_shows_none(self):
        import gradio as gr

        with mock.patch.object(vdm, "discover", return_value=[]), gr.Blocks():
            _enabled, model, *_ = MOD.AnimaVaeDegrid().ui(False)
        self.assertEqual(model.value, vdm.NONE_NAME)

    def test_settings_are_registered(self):
        self.assertIn(MOD._on_ui_settings, SETTINGS)
        added = {}

        class OptionInfo:
            def __init__(self, default, label, component=None, component_args=None, section=None, **kw):
                self.default, self.label, self.section = default, label, section
                self.component_args = component_args

            def info(self, text):
                self.comment = text
                return self

        shared = types.SimpleNamespace(
            OptionInfo=OptionInfo,
            opts=types.SimpleNamespace(add_option=lambda key, info: added.__setitem__(key, info)),
        )
        modules_stub = types.ModuleType("modules")
        modules_stub.shared = shared
        with _SysModules({"modules": modules_stub}):
            MOD._on_ui_settings()
        self.assertEqual(set(added), {vdr.OPT_DEVICE, vdr.OPT_PRECISION, vdr.OPT_KEEP_LOADED})
        self.assertEqual(added[vdr.OPT_DEVICE].default, "auto")
        self.assertEqual(added[vdr.OPT_PRECISION].default, "fp16")
        self.assertIs(added[vdr.OPT_KEEP_LOADED].default, False)
        for info in added.values():
            self.assertEqual(info.section, ("sam3_degrid", "SAM Extra VAE DeGrid"))
        self.assertEqual([v for _l, v in added[vdr.OPT_DEVICE].component_args["choices"]], ["auto", "cpu"])


class ExtrasTests(unittest.TestCase):
    def setUp(self):
        self.script = EXTRAS.ScriptPostprocessingVaeDegrid()
        self.discover = mock.patch.object(vdm, "discover", return_value=[ENTRY])
        self.log = mock.patch.object(vdr, "log")
        self.discover.start()
        self.log.start()

    def tearDown(self):
        self.discover.stop()
        self.log.stop()

    def test_runs_before_upscale(self):
        self.assertLess(EXTRAS.ScriptPostprocessingVaeDegrid.order, 1000)
        self.assertEqual(EXTRAS.ScriptPostprocessingVaeDegrid.name, uvd.TITLE)

    def test_ui_keys_match_process_arguments(self):
        import gradio as gr

        with gr.Blocks():
            controls = self.script.ui()
        self.assertEqual(list(controls), list(uvd.ARG_NAMES))

    def test_process(self):
        runtime = _FakeRuntime()
        image = Image.new("RGB", (8, 8))
        with mock.patch.object(vdr, "shared_runtime", return_value=runtime):
            off = _PostprocessedImage(image)
            self.script.process(off, enabled=False, model="", mode="Full", strength=1.0, tile=512)
            self.assertIs(off.image, image)
            on = _PostprocessedImage(image)
            self.script.process(on, enabled=True, model="qwenVAEDegridNafnet_v11", mode="Full (전체)", strength=1.0,
                                tile=512)
        self.assertEqual(on.image.getpixel((0, 0)), (1, 2, 3))
        self.assertEqual(on.info[uvd.KEY_MODEL], "qwenVAEDegridNafnet_v11")
        self.assertEqual(on.info[uvd.KEY_MODE], "Full")

    def test_api_none_args_mean_off(self):
        pp = _PostprocessedImage(Image.new("RGB", (8, 8)))
        self.script.process(pp, enabled=None, model=None, mode=None, strength=None, tile=None)
        self.assertEqual(pp.info, {})

    def test_missing_model_and_failure(self):
        image = Image.new("RGB", (8, 8))
        with mock.patch.object(vdm, "discover", return_value=[]):
            pp = _PostprocessedImage(image)
            self.script.process(pp, enabled=True)
        self.assertIs(pp.image, image)
        self.assertEqual(pp.info, {uvd.KEY_ERROR: "model not found: auto"})
        runtime = _FakeRuntime(fail_on={1})
        with mock.patch.object(vdr, "shared_runtime", return_value=runtime):
            pp = _PostprocessedImage(image)
            self.script.process(pp, enabled=True)
        self.assertIs(pp.image, image)
        self.assertIn("CUDA out of memory", pp.info[uvd.KEY_ERROR])
        self.assertEqual(runtime.released, 1)


if __name__ == "__main__":
    unittest.main()
