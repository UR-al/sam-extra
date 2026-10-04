"""sam3ext/ui_save_defaults.py + scripts/layout_save_defaults.py — '기본값 저장' 숨은 버튼의 Gradio 연결(진짜 Gradio 4.40).

Forge 의 화면 짜임을 흉내 낸다(modules/ui.py create_ui): txt2img·img2img 는 각자의 gr.Blocks 에서 만들어져 demo 의
탭에 render 되고, UI Preset·Checkpoint·VAE / Text Encoder·Low Bits 는 demo 의 빠른 설정 줄에서 만들어지며, 그 뒤
demo 에 footer(gr.HTML elem_id "footer")가 생긴다. webui 처럼 생성자 직후 after_component 를 부르면
(modules/gradio_extensions.py) footer 때 숨은 버튼이 만들어져 main_entry 가 쓰는 23개 컴포넌트에 연결되는지,
그 이벤트를 Gradio 의 process_api 로 돌리면(페이지의 js 가 보내는 것처럼 출력값 하나가 더 붙어도) 활성 프리셋이
저장되는지 확인한다. Forge 체크아웃이 있으면 create_ui 의 순서(빠른 설정 → loadsave.setup_ui → footer →
forge_main_entry)가 아직 그대로인지 AST 로 본다(없으면 건너뜀 — CI).
"""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no version check over the network (Blocks construction)

import ast  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest import mock  # noqa: E402

import gradio as gr  # noqa: E402
import gradio.blocks  # noqa: E402
from gradio.state_holder import SessionState  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import save_defaults as sd  # noqa: E402
from sam3ext import ui_save_defaults as usd  # noqa: E402
from tests import test_save_defaults as base  # noqa: E402  (helpers only — no TestCase is imported by name)
from tests._forge_checkout import require_forge_file  # noqa: E402

PRESETS = ["sd", "xl", "flux", "anima", "wan"]


@contextlib.contextmanager
def _forge_after_component(*callbacks):
    """modules/gradio_extensions.py IOComponent_init / BlockContext_init: callbacks run right after __init__."""
    original_block = gradio.blocks.BlockContext.__init__
    original_component = gr.components.Component.__init__

    def block_init(self, *args, **kwargs):
        result = original_block(self, *args, **kwargs)
        for callback in callbacks:
            callback(self, **kwargs)
        return result

    def component_init(self, *args, **kwargs):
        result = original_component(self, *args, **kwargs)
        for callback in callbacks:
            callback(self, **kwargs)
        return result

    gradio.blocks.BlockContext.__init__ = block_init
    gr.components.Component.__init__ = component_init
    try:
        yield
    finally:
        gradio.blocks.BlockContext.__init__ = original_block
        gr.components.Component.__init__ = original_component


@contextlib.contextmanager
def _modules(stubs):
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _paste(component, target, api=None):
    """modules/infotext_utils.py PasteField: label is the target when it is a string."""
    return types.SimpleNamespace(component=component, label=target if isinstance(target, str) else None, api=api)


def _main_entry_stub():
    """modules_forge.main_entry: the four quick-settings globals and get_a1111_ui_component (same body as Forge's)."""
    module = types.ModuleType("modules_forge.main_entry")
    module.paste_fields = {}

    def get_a1111_ui_component(tab, label):
        fields = module.paste_fields[tab]["fields"]
        for f in fields:
            if f.label == label or f.api == label:
                return f.component

    module.get_a1111_ui_component = get_a1111_ui_component
    module.ui_forge_preset = module.ui_checkpoint = module.ui_vae = module.ui_forge_unet_dtype = None
    package = types.ModuleType("modules_forge")
    package.__path__ = []
    package.main_entry = module
    return module, {"modules_forge": package, "modules_forge.main_entry": module}


def _tab(tab, main_entry):
    """The parts of modules/ui.py + processing_scripts/sampler.py that main_entry looks up, with Forge's ids/ranges."""
    with gr.Blocks() as interface:
        sampler = gr.Dropdown(label="Sampling Method", elem_id=f"{tab}_sampling", choices=base.SAMPLERS,
                              value=base.SAMPLERS[0])
        scheduler = gr.Dropdown(label="Schedule Type", elem_id=f"{tab}_scheduler", choices=base.SCHEDULERS,
                                value=base.SCHEDULERS[0])
        steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tab}_steps", label="Sampling Steps", value=20)
        width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=1024, elem_id=f"{tab}_width")
        height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=1024, elem_id=f"{tab}_height")
        batch = gr.Slider(minimum=1, maximum=8, step=1, label="Batch Size", value=1, elem_id=f"{tab}_batch_size")
        dcfg = gr.Slider(minimum=1.0 if tab == "txt2img" else 0.0, maximum=24.0, step=0.5, label="Distilled CFG Scale",
                         value=3.0, elem_id=f"{tab}_distilled_cfg_scale")
        cfg = gr.Slider(minimum=1.0, maximum=24.0, step=0.5, label="CFG Scale", value=6.0, elem_id=f"{tab}_cfg_scale")
        fields = [
            _paste(steps, "Steps", api="steps"), _paste(sampler, object, api="sampler_name"),
            _paste(scheduler, object, api="scheduler"), _paste(cfg, "CFG scale", api="cfg_scale"),
            _paste(dcfg, "Distilled CFG Scale", api="distilled_cfg_scale"), _paste(width, "Size-1", api="width"),
            _paste(height, "Size-2", api="height"), _paste(batch, "Batch size", api="batch_size"),
        ]
        if tab == "txt2img":
            hr_steps = gr.Slider(minimum=0, maximum=150, step=1, label="Hires steps", value=0,
                                 elem_id="txt2img_hires_steps")
            hr_dcfg = gr.Slider(minimum=1.0, maximum=24.0, step=0.5, label="Hires Distilled CFG Scale", value=3.0,
                                elem_id="txt2img_hr_distilled_cfg")
            hr_cfg = gr.Slider(minimum=1.0, maximum=24.0, step=0.5, label="Hires CFG Scale", value=6.0,
                               elem_id="txt2img_hr_cfg")
            fields += [_paste(hr_steps, "Hires steps", api="hr_second_pass_steps"),
                       _paste(hr_cfg, "Hires CFG Scale", api="hr_cfg"),
                       _paste(hr_dcfg, "Hires Distilled CFG Scale", api="hr_distilled_cfg")]
    main_entry.paste_fields[tab] = {"fields": fields}
    return interface


def build_ui(main_entry, *, footers=1, skip=(), throwaway_footer=False):
    """modules/ui.py create_ui order: tab interfaces → demo(quick settings, tabs rendered, footer).

    ``throwaway_footer``: first a footer made with render=False inside demo (what Gradio builds while a handler
    returns gr.update() — never registered), with every main_entry component already in place."""
    txt2img = _tab("txt2img", main_entry)
    img2img = _tab("img2img", main_entry)
    with gr.Blocks() as demo:
        main_entry.ui_forge_preset = gr.Dropdown(label="UI Preset", value="anima", choices=PRESETS,
                                                 elem_id="forge_ui_preset")
        main_entry.ui_checkpoint = gr.Dropdown(label="Checkpoint", value=None, choices=sorted(base.CHECKPOINTS),
                                               elem_id="setting_sd_model_checkpoint", elem_classes=["model_selection"])
        main_entry.ui_vae = gr.Dropdown(label="VAE / Text Encoder", value=None, choices=sorted(base.MODULES),
                                        multiselect=True, elem_id="setting_sd_modules")
        main_entry.ui_forge_unet_dtype = gr.Dropdown(label="Diffusion in Low Bits", value=None, choices=base.DTYPES,
                                                     elem_id="forge_ui_dtype")
        for name in skip:
            setattr(main_entry, name, None)
        with gr.Tabs(elem_id="tabs"):
            with gr.TabItem("txt2img", id="txt2img", elem_id="tab_txt2img"):
                txt2img.render()
            with gr.TabItem("img2img", id="img2img", elem_id="tab_img2img"):
                img2img.render()
        if throwaway_footer:
            gr.HTML("<div>footer</div>", elem_id="footer", render=False)
        for _ in range(footers):
            gr.HTML("<div>footer</div>", elem_id="footer")
    return demo


def by_elem_id(demo, elem_id):
    found = [block for block in demo.blocks.values() if getattr(block, "elem_id", None) == elem_id]
    return found


def click_dependency(demo, button):
    matches = [fn for fn in demo.fns.values() if (button._id, "click") in [tuple(t) for t in fn.targets]]
    assert len(matches) == 1, matches
    return matches[0]


class WiringTests(unittest.TestCase):
    def setUp(self):
        usd._built_root = None
        self.main_entry, stubs = _main_entry_stub()
        self.stubs = _modules(stubs)
        self.stubs.__enter__()
        self.addCleanup(self.stubs.__exit__, None, None, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def build(self, **kwargs):
        with _forge_after_component(usd.on_after_component):
            return build_ui(self.main_entry, **kwargs)

    def test_the_hidden_button_takes_main_entrys_components_in_order(self):
        demo = self.build()
        (run,) = by_elem_id(demo, usd.RUN_ELEM_ID)
        (request,) = by_elem_id(demo, usd.REQUEST_ELEM_ID)
        (result,) = by_elem_id(demo, usd.RESULT_ELEM_ID)
        (bridge,) = by_elem_id(demo, usd.BRIDGE_ELEM_ID)
        self.assertIs(bridge.visible, False)
        dep = click_dependency(demo, run)
        me = self.main_entry
        expected = [request, me.ui_forge_preset, me.ui_checkpoint, me.ui_vae, me.ui_forge_unet_dtype]
        expected += [me.get_a1111_ui_component(tab, label) for tab, label in sd.UI_COMPONENTS]
        self.assertEqual([c._id for c in dep.inputs], [c._id for c in expected])
        self.assertEqual(len(dep.inputs), 2 + len(sd.FIELDS))
        self.assertEqual([c.elem_id for c in dep.inputs[5:]], [
            "txt2img_steps", "txt2img_hires_steps", "img2img_steps", "txt2img_sampling", "img2img_sampling",
            "txt2img_scheduler", "img2img_scheduler", "txt2img_width", "img2img_width", "txt2img_height",
            "img2img_height", "txt2img_cfg_scale", "txt2img_hr_cfg", "img2img_cfg_scale",
            "txt2img_distilled_cfg_scale", "txt2img_hr_distilled_cfg", "img2img_distilled_cfg_scale",
            "txt2img_batch_size", "img2img_batch_size"])
        self.assertEqual([c._id for c in dep.outputs], [result._id])
        self.assertEqual(dep.js, usd.ARGS_JS)
        self.assertFalse(dep.queue)
        config = demo.get_config_file()
        (entry,) = [d for d in config["dependencies"] if d["js"] == usd.ARGS_JS]
        self.assertEqual(entry["inputs"], [c._id for c in expected])
        self.assertEqual(entry["outputs"], [result._id])
        # the bridge sits after the footer, in demo itself (not inside a tab)
        ids = [child.elem_id for child in demo.children if getattr(child, "elem_id", None)]
        self.assertEqual(ids[-2:], ["footer", usd.BRIDGE_ELEM_ID])

    def _click(self, demo, values, *, extra_output=True, request_id="req-7"):
        (run,) = by_elem_id(demo, usd.RUN_ELEM_ID)
        dep = click_dependency(demo, run)
        inputs = [request_id, *values]
        if extra_output:
            inputs.append("")       # Gradio's frontend hands the js the outputs too, and sends back what it returns
        state = SessionState(demo)
        return json.loads(asyncio.run(demo.process_api(block_fn=dep, inputs=inputs, state=state))["data"][0])

    def test_a_click_through_gradio_saves_the_active_preset(self):
        demo = self.build()
        host = base.make_host(tmp=self.tmp.name, presets=PRESETS, data=base.sizes())
        values = base.ui_values()
        with mock.patch.object(usd, "forge_host", lambda: host), contextlib.redirect_stdout(io.StringIO()) as out:
            payload = self._click(demo, ["anima", *values])
        self.assertTrue(payload["ok"], payload)
        self.assertEqual((payload["request"], payload["preset"], payload["written"]), ("req-7", "anima", 22))
        self.assertEqual(payload["kept"], [])
        self.assertEqual(host.opts.data["anima_t2i_sampler"], "ER SDE (Tunable)")
        self.assertEqual(host.opts.data["anima_t2i_width"], 832)
        self.assertEqual(host.opts.saved, [host.config_filename])
        self.assertIn("wrote 22 options, 18 changed\n", out.getvalue())
        # without the trailing output value too
        with mock.patch.object(usd, "forge_host", lambda: host), contextlib.redirect_stdout(io.StringIO()):
            again = self._click(demo, ["anima", *values], extra_output=False, request_id="req-8")
        self.assertEqual((again["ok"], again["request"], again["changed"]), (True, "req-8", 0))

    def test_a_click_keeps_the_preset_sizes_forge_skips(self):
        """presets.register's default width/height 0 = Forge keeps the screen size: the click leaves them at 0."""
        demo = self.build()
        host = base.make_host(tmp=self.tmp.name, presets=PRESETS)
        with mock.patch.object(usd, "forge_host", lambda: host), contextlib.redirect_stdout(io.StringIO()) as out:
            payload = self._click(demo, ["anima", *base.ui_values()])
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["written"], 18)
        self.assertEqual([(k["key"], k["label"], k["value"]) for k in payload["kept"]],
                         [("anima_t2i_width", "txt2img Width", 0), ("anima_i2i_width", "img2img Width", 0),
                          ("anima_t2i_height", "txt2img Height", 0), ("anima_i2i_height", "img2img Height", 0)])
        for key in ("anima_t2i_width", "anima_i2i_width", "anima_t2i_height", "anima_i2i_height"):
            self.assertNotIn(key, host.opts.data)
        self.assertIn("wrote 18 options, 14 changed, kept 4 at 0 (Forge keeps the screen value)", out.getvalue())

    def test_a_refused_save_reaches_the_page_as_json(self):
        # The refusal must come from the save handler, not from Gradio's own input checks: Gradio 5 (the CI venv)
        # rejects an out-of-range slider value (e.g. steps 151 > 150) before the handler runs, Gradio 4.40 (Forge)
        # does not. A sampler the screen offers but this Forge does not list (the live test's 'Beta57 (RES4LYF)'
        # case) passes Gradio's checks in both and is refused by the handler.
        demo = self.build()
        samplers = [name for name in base.SAMPLERS if name != "ER SDE (Tunable)"]
        host = base.make_host(tmp=self.tmp.name, presets=PRESETS, samplers=samplers)
        with mock.patch.object(usd, "forge_host", lambda: host), contextlib.redirect_stderr(io.StringIO()) as err:
            payload = self._click(demo, ["anima", *base.ui_values(t2i_sampler="ER SDE (Tunable)")])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["request"], "req-7")
        self.assertIn("txt2img Sampler", payload["error"])
        self.assertIn("failed", err.getvalue())
        self.assertEqual(host.opts.data, {})

    def test_the_handler_never_raises(self):
        def broken():
            raise RuntimeError("Forge is not ready")

        with contextlib.redirect_stderr(io.StringIO()) as err:
            text = usd.handle_save("abc", "anima", *base.ui_values(), host_factory=broken)
        payload = json.loads(text)
        self.assertEqual(payload, {"ok": False, "request": "abc", "preset": "anima",
                                   "error": "RuntimeError: Forge is not ready"})
        self.assertIn("Traceback", err.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(json.loads(usd.handle_save(None, None, host_factory=broken))["request"], "")

    def test_built_once_per_screen_and_again_for_a_new_screen(self):
        demo = self.build(footers=2)
        self.assertEqual(len(by_elem_id(demo, usd.RUN_ELEM_ID)), 1)
        reloaded = self.build()                    # Reload UI: a new demo
        self.assertEqual(len(by_elem_id(reloaded, usd.RUN_ELEM_ID)), 1)
        self.assertIs(usd._built_root(), reloaded)

    def test_a_missing_component_builds_nothing_and_says_which(self):
        for name, label in (("ui_checkpoint", "Checkpoint"), ("ui_forge_preset", "UI Preset")):
            with self.subTest(name=name):
                usd._built_root = None
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    demo = self.build(skip=(name,))
                self.assertEqual(by_elem_id(demo, usd.RUN_ELEM_ID), [])
                self.assertIn(label, err.getvalue())
                self.assertIsNone(usd._built_root)
        self.main_entry.paste_fields.clear()
        usd._built_root = None
        with contextlib.redirect_stderr(io.StringIO()) as err:
            with _forge_after_component(usd.on_after_component):
                with gr.Blocks() as demo:
                    gr.HTML("x", elem_id="footer")
        self.assertEqual(by_elem_id(demo, usd.RUN_ELEM_ID), [])
        self.assertIn("the button is not wired", err.getvalue())

    def test_a_throwaway_footer_on_a_full_screen_builds_nothing(self):
        # every component main_entry uses is there and registered: only the footer's own registration stops it
        demo = self.build(footers=0, throwaway_footer=True)
        self.assertEqual(by_elem_id(demo, usd.RUN_ELEM_ID), [])
        self.assertIsNone(usd._built_root)
        demo = self.build(footers=1, throwaway_footer=True)       # the real footer after it still builds it, once
        self.assertEqual(len(by_elem_id(demo, usd.RUN_ELEM_ID)), 1)
        self.assertIs(usd._built_root(), demo)

    def test_components_of_another_screen_build_nothing_and_say_so(self):
        # main_entry still points at the components of a screen built before (its footer never came): a footer in a
        # new screen must not wire a button to them
        self.build(footers=0)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            with _forge_after_component(usd.on_after_component):
                with gr.Blocks() as other:
                    gr.HTML("x", elem_id="footer")
        self.assertEqual(by_elem_id(other, usd.RUN_ELEM_ID), [])
        self.assertIsNone(usd._built_root)
        self.assertIn(f"컴포넌트 {4 + len(sd.UI_COMPONENTS)}개가 이 화면에 등록되어 있지 않습니다", err.getvalue())
        self.assertIn("the button is not wired", err.getvalue())

    def test_other_components_and_throwaway_footers_are_ignored(self):
        with _forge_after_component(usd.on_after_component):
            with gr.Blocks() as demo:
                gr.HTML("not the footer", elem_id="footer_like")
                gr.HTML("x", elem_id="footer", render=False)   # Gradio's request-time rebuild: never registered
        self.assertEqual(by_elem_id(demo, usd.RUN_ELEM_ID), [])
        self.assertIsNone(usd._built_root)
        usd.on_after_component(types.SimpleNamespace(_id=-1), elem_id="footer")   # outside any Blocks
        self.assertIsNone(usd._built_root)


class ForgeHostTests(unittest.TestCase):
    """forge_host(): what every real click reads from Forge (modules/shared.py, shared_items.py, sd_models.py,
    modules_forge/main_entry.py, presets.py), on stand-ins with Forge's shapes."""

    def stubs(self, cmd_opts):
        modules = types.ModuleType("modules")
        modules.__path__ = []
        sd_models = types.ModuleType("modules.sd_models")
        known = {"anima-preview2.safetensors [1234abcd]": object()}
        sd_models.get_closet_checkpoint_match = known.get
        shared = types.ModuleType("modules.shared")
        shared.opts = self.opts
        shared.config_filename = "C:/forge/config.json"
        shared.cmd_opts = cmd_opts
        shared_items = types.ModuleType("modules.shared_items")
        shared_items.list_samplers = lambda: [types.SimpleNamespace(name="Euler", aliases=["k_euler"]),
                                              types.SimpleNamespace(name="ER SDE (Tunable)", aliases=[])]
        shared_items.list_schedulers = lambda: ["Automatic", "Flow Cosmos rho7"]      # labels (all_schedulers)
        for name, module in (("sd_models", sd_models), ("shared", shared), ("shared_items", shared_items)):
            setattr(modules, name, module)
        forge = types.ModuleType("modules_forge")
        forge.__path__ = []
        main_entry = types.ModuleType("modules_forge.main_entry")
        main_entry.module_list = self.module_list
        main_entry.forge_unet_storage_dtype_options = {"Automatic": (None, False), "bnb-nf4": ("nf4", False),
                                                       "float8-e4m3fn": ("fp8", False)}
        presets = types.ModuleType("modules_forge.presets")

        class PresetArch:
            @staticmethod
            def choices():
                return ["sd", "xl", "anima"]

        presets.PresetArch = PresetArch
        forge.main_entry, forge.presets = main_entry, presets
        return {"modules": modules, "modules.sd_models": sd_models, "modules.shared": shared,
                "modules.shared_items": shared_items, "modules_forge": forge, "modules_forge.main_entry": main_entry,
                "modules_forge.presets": presets}

    def setUp(self):
        self.opts = types.SimpleNamespace(data={})
        self.module_list = {"qwen_image_vae.safetensors": "C:/forge/models/VAE/qwen_image_vae.safetensors"}

    def test_every_field_comes_from_forge(self):
        with _modules(self.stubs(types.SimpleNamespace(freeze_settings=True))):
            host = usd.forge_host()
        self.assertIs(host.opts, self.opts)
        self.assertEqual(host.config_filename, "C:/forge/config.json")
        self.assertEqual(list(host.presets), ["sd", "xl", "anima"])
        self.assertTrue(host.checkpoint_known("anima-preview2.safetensors [1234abcd]"))
        self.assertFalse(host.checkpoint_known("missing.safetensors"))
        self.assertEqual(dict(host.module_paths), self.module_list)
        self.assertIsNot(host.module_paths, self.module_list)        # a copy: Forge refreshes its own dict
        self.assertEqual(list(host.dtypes), ["Automatic", "bnb-nf4", "float8-e4m3fn"])
        self.assertEqual(list(host.samplers), ["Euler", "ER SDE (Tunable)"])
        self.assertEqual(list(host.schedulers), ["Automatic", "Flow Cosmos rho7"])
        self.assertIs(host.frozen, True)

    def test_frozen_only_with_freeze_settings(self):
        for cmd_opts in (types.SimpleNamespace(freeze_settings=False), types.SimpleNamespace()):
            with self.subTest(cmd_opts=cmd_opts), _modules(self.stubs(cmd_opts)):
                self.assertIs(usd.forge_host().frozen, False)


class ScriptFileTests(unittest.TestCase):
    def test_the_script_registers_one_after_component_callback(self):
        registered = []
        package = types.ModuleType("modules")
        package.__path__ = []
        callbacks = types.ModuleType("modules.script_callbacks")
        callbacks.on_after_component = lambda fn, **kw: registered.append(("on_after_component", fn))
        package.script_callbacks = callbacks
        with _modules({"modules": package, "modules.script_callbacks": callbacks}):
            spec = importlib.util.spec_from_file_location("_test_layout_save_defaults",
                                                          ROOT / "scripts" / "layout_save_defaults.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.assertEqual(registered, [("on_after_component", usd.on_after_component)])
        source = (ROOT / "scripts" / "layout_save_defaults.py").read_text(encoding="utf-8")
        self.assertNotIn("class ", source.split('"""', 2)[2], "no Script class: nothing in the scripts column")

    def test_negpip_stays_the_last_script(self):
        names = sorted(name for name in os.listdir(ROOT / "scripts") if name.endswith(".py"))
        self.assertLess(names.index("layout_save_defaults.py"), names.index("negpip.py"))
        self.assertEqual(names[-1], "negpip.py")


class ForgeUiOrderTests(unittest.TestCase):
    """modules/ui.py create_ui still builds quick settings, then setup_ui, then the footer, then forge_main_entry."""

    def test_footer_comes_after_every_component_and_before_forge_main_entry(self):
        path = require_forge_file("modules/ui.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        create_ui = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_ui")
        lines = {}
        for node in ast.walk(create_ui):
            if not isinstance(node, ast.Call):
                continue
            text = ast.unparse(node.func)
            if text in ("settings.add_quicksettings", "loadsave.setup_ui", "main_entry.forge_main_entry",
                        "settings.add_functionality"):
                lines.setdefault(text, []).append(node.lineno)
            if text == "gr.HTML" and any(k.arg == "elem_id" and getattr(k.value, "value", None) == "footer"
                                         for k in node.keywords):
                lines.setdefault("footer", []).append(node.lineno)
            if text.endswith(".render") and isinstance(node.func, ast.Attribute) \
                    and getattr(node.func.value, "id", None) == "interface":
                lines.setdefault("interface.render", []).append(node.lineno)
        for key in ("settings.add_quicksettings", "interface.render", "loadsave.setup_ui", "footer",
                    "main_entry.forge_main_entry"):
            self.assertEqual(len(lines.get(key, [])), 1, (key, lines.get(key)))
        order = [lines[key][0] for key in ("settings.add_quicksettings", "interface.render", "loadsave.setup_ui",
                                           "footer", "main_entry.forge_main_entry")]
        self.assertEqual(order, sorted(order))
        # the quick settings row builds the four main_entry components
        settings = require_forge_file("modules/ui_settings.py").read_text(encoding="utf-8")
        self.assertIn("main_entry.make_checkpoint_manager_ui()", settings)


if __name__ == "__main__":
    unittest.main()
