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


def _settings_stubs(forge_package, settings_components, *, opts=None, get_value=None, presets=PRESETS):
    """modules/shared.py settings_components (Forge's key → Settings field table, set by ui_settings.create_ui) and
    opts, modules/ui_settings.py get_value_for_setting, modules_forge/presets.py PresetArch.choices()."""
    modules = types.ModuleType("modules")
    modules.__path__ = []
    shared = types.ModuleType("modules.shared")
    shared.settings_components = settings_components
    shared.opts = opts
    ui_settings = types.ModuleType("modules.ui_settings")
    if get_value is not None:
        ui_settings.get_value_for_setting = get_value
    modules.shared, modules.ui_settings = shared, ui_settings
    presets_module = types.ModuleType("modules_forge.presets")

    class PresetArch:
        @staticmethod
        def choices():
            return list(presets)

    presets_module.PresetArch = PresetArch
    forge_package.presets = presets_module
    return {"modules": modules, "modules.shared": shared, "modules.ui_settings": ui_settings,
            "modules_forge.presets": presets_module}


def forge_get_value_for_setting(opts):
    """modules/ui_settings.py get_value_for_setting, the same body, on these opts (what Forge's demo.load puts in every
    Settings field when the page opens). ForgeSettingsTests runs Forge's own."""
    def get_value_for_setting(key):
        value = getattr(opts, key)
        info = opts.data_labels[key]
        args = info.component_args() if callable(info.component_args) else info.component_args or {}
        args = {k: v for k, v in args.items() if k not in {"precision"}}
        return gr.update(value=value, **args)

    return get_value_for_setting


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


def build_ui(main_entry, *, footers=1, skip=(), throwaway_footer=False, settings=None):
    """modules/ui.py create_ui order: tab interfaces → demo(quick settings, tabs rendered, footer).

    ``throwaway_footer``: first a footer made with render=False inside demo (what Gradio builds while a handler
    returns gr.update() — never registered), with every main_entry component already in place.
    ``settings``: the Settings interface (built before demo, like settings.create_ui) rendered as a tab of demo."""
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
            if settings is not None:
                with gr.TabItem("Settings", id="settings", elem_id="tab_settings"):
                    settings.render()
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


def chained_after(demo, dependency):
    """The events Gradio's page runs when ``dependency`` finishes (.then / .success)."""
    return [fn for fn in demo.fns.values() if fn.trigger_after == dependency._id]


def page_values(demo):
    """The page's value of every component as built (Forge's demo.load puts the same setting values in the
    Settings fields when the page opens)."""
    return {block._id: getattr(block, "value", None) for block in demo.blocks.values()
            if isinstance(block, gr.components.Component)}


def run_on_page(demo, dependency, inputs, page):
    """What Gradio's page does for one click: run the event, put its outputs into the page's values, then run every
    event chained after it with its inputs read from the page. Returns the raw outputs, event by event."""
    state = SessionState(demo)
    outputs = []
    pending = [(dependency, inputs)]
    while pending:
        fn, args = pending.pop(0)
        if args is None:
            args = [page[component._id] for component in fn.inputs]
        data = asyncio.run(demo.process_api(block_fn=fn, inputs=args, state=state))["data"]
        outputs.append((fn, data))
        for component, item in zip(fn.outputs, data):
            if isinstance(item, dict) and item.get("__type__") == "update":
                if "value" in item:
                    page[component._id] = item["value"]
            else:
                page[component._id] = item
        pending += [(after, None) for after in chained_after(demo, fn)]
    return outputs


def settings_tab(opts, keys, *, state_keys=()):
    """modules/ui_settings.py UiSettings.create_ui: the Settings fields are built in their own gr.Blocks (rendered as a
    tab of demo later) with elem_id "setting_<key>", Forge's dropdown / slider for these options, the page's value
    the setting's; Forge keeps them in shared.settings_components (key → field). ``state_keys`` get a gr.State, as
    create_setting_component gives FormRow / gr.State options. One setting that is not a preset's is there too."""
    components = {}
    with gr.Blocks() as interface:
        gr.Button("Apply settings", elem_id="settings_submit")
        for key in keys:
            if key in state_keys:
                components[key] = gr.State(None)
                continue
            info = opts.data_labels[key]
            args = info.component_args() if callable(info.component_args) else dict(info.component_args or {})
            maker = gr.Dropdown if "choices" in args else gr.Slider
            components[key] = maker(label=info.label or key, value=getattr(opts, key), elem_id=f"setting_{key}", **args)
        components["sd_vae_checkpoint_cache"] = gr.Slider(minimum=0, maximum=10, step=1, value=0, label="VAE cache",
                                                          elem_id="setting_sd_vae_checkpoint_cache")
    return interface, components


class WiringTests(unittest.TestCase):
    def setUp(self):
        usd._built_root = None
        self.main_entry, stubs = _main_entry_stub()
        # no Settings fields: the save button alone (SettingsRefreshTests has them)
        stubs.update(_settings_stubs(stubs["modules_forge"], {}))
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
        self.assertEqual(chained_after(demo, dep), [], "no Settings fields on this screen: nothing to refresh")
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


class SettingsRefreshTests(unittest.TestCase):
    """The live v0.34.0 check: after 기본값 저장 wrote anima_t2i_sampler, Settings → Apply settings in the same page put the
    old value back — the Settings fields still held what Forge's demo.load put in when the page opened, and Forge's
    run_settings writes every field. The save click now refreshes the Settings fields of exactly the keys it wrote.

    Gradio 4.40's TabItem always mounts its children (``display: none`` while another tab is selected) and the page
    keeps every component's value, so a field of a Settings tab that was never opened gets the new value too — what
    the page then sends on Apply is the value the server last put there, which these tests follow (``run_on_page``)."""

    def setUp(self):
        usd._built_root = None
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.host = base.make_host(tmp=self.tmp.name, presets=PRESETS, data=base.sizes())
        self.opts = self.host.opts
        self.main_entry, stubs = _main_entry_stub()
        stubs.update(_settings_stubs(stubs["modules_forge"], {}, opts=self.opts,
                                     get_value=forge_get_value_for_setting(self.opts)))
        self.shared = stubs["modules.shared"]
        self.ui_settings = stubs["modules.ui_settings"]
        self.stubs = _modules(stubs)
        self.stubs.__enter__()
        self.addCleanup(self.stubs.__exit__, None, None, None)
        # the preset options that have a Settings field (Forge hides checkpoint / modules / Low Bits), every preset
        self.keys = [key for key in (template.format(p=preset) for preset in PRESETS for template, _k, _l in sd.FIELDS)
                     if key in self.opts.data_labels and not key.startswith("forge_")]

    def build(self, *, keys=None, state_keys=(), rendered=True):
        interface, components = settings_tab(self.opts, self.keys if keys is None else keys, state_keys=state_keys)
        self.shared.settings_components = components
        with _forge_after_component(usd.on_after_component):
            demo = build_ui(self.main_entry, settings=interface if rendered else None)
        (run,) = by_elem_id(demo, usd.RUN_ELEM_ID)
        return demo, click_dependency(demo, run), components

    def save(self, demo, dep, page, values=None, request_id="req-1"):
        """The page's click: the event's js hands Python [request id, inputs…, the result textbox's value]."""
        values = base.ui_values() if values is None else values
        (result,) = by_elem_id(demo, usd.RESULT_ELEM_ID)
        with mock.patch.object(usd, "forge_host", lambda: self.host), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            outputs = run_on_page(demo, dep, [request_id, "anima", *values, page[result._id]], page)
        return json.loads(outputs[0][1][0]), outputs

    def apply_settings(self, components, page):
        """modules/ui_settings.py run_settings: every Settings field's page value goes through opts.set
        (ForgeSettingsTests runs Forge's own)."""
        for key, component in components.items():
            if key in self.opts.data_labels and not getattr(component, "stateful", False):
                self.opts.set(key, page[component._id])

    def test_apply_settings_after_a_save_keeps_what_the_save_wrote(self):
        demo, dep, components = self.build()
        page = page_values(demo)                                  # the page as Forge's demo.load left it
        self.assertEqual(page[components["anima_t2i_sampler"]._id], "Euler")
        payload, _outputs = self.save(demo, dep, page)
        self.assertTrue(payload["ok"], payload)
        saved = {key: getattr(self.opts, key) for key in self.keys}
        self.assertEqual(saved["anima_t2i_sampler"], "ER SDE (Tunable)")
        self.apply_settings(components, page)                     # Settings → Apply settings, same page
        for key, value in saved.items():
            with self.subTest(key=key):
                self.assertEqual(getattr(self.opts, key), value, "Apply settings put the page's old value back")

    def test_the_click_refreshes_every_presets_fields_after_the_save(self):
        demo, dep, components = self.build(state_keys=("xl_t2i_hr_step",))
        (after,) = chained_after(demo, dep)
        (result,) = by_elem_id(demo, usd.RESULT_ELEM_ID)
        self.assertEqual([c._id for c in after.inputs], [result._id])
        expected = [components[key]._id for key in self.keys if key != "xl_t2i_hr_step"]
        self.assertEqual([c._id for c in after.outputs], expected, "every preset's fields, in key order")
        self.assertGreater(len(expected), 4 * 15)
        self.assertNotIn(components["sd_vae_checkpoint_cache"]._id, expected, "a setting the button never writes")
        self.assertNotIn(components["xl_t2i_hr_step"]._id, expected, "a gr.State has no value on the page")
        self.assertIsNone(after.js)
        self.assertFalse(after.queue)
        self.assertEqual(after.show_progress, "hidden")
        self.assertFalse(after.trigger_only_on_success)         # .then: runs after a refused save too (changes nothing)

    def test_only_the_written_keys_get_new_values(self):
        self.host = base.make_host(tmp=self.tmp.name, presets=PRESETS)   # sizes 0: Forge keeps the screen size
        self.opts = self.host.opts
        self.shared.opts = self.opts
        self.ui_settings.get_value_for_setting = forge_get_value_for_setting(self.opts)
        demo, dep, components = self.build()
        page = page_values(demo)
        before = dict(page)
        payload, outputs = self.save(demo, dep, page)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(len(payload["kept"]), 4)
        written = set(payload["written_keys"])
        changed = {key for key, component in components.items() if page[component._id] != before[component._id]}
        self.assertTrue(changed)
        self.assertLessEqual(changed, written)
        for key, component in components.items():
            with self.subTest(key=key):
                if key in written:
                    self.assertEqual(page[component._id], self.opts.data[key])
                else:                                            # other presets, the kept sizes, other settings
                    self.assertEqual(page[component._id], before[component._id])
        self.assertEqual(len(outputs), 2)
        by_key = dict(zip(self.keys, outputs[1][1]))
        self.assertEqual(by_key["anima_t2i_sampler"]["value"], "ER SDE (Tunable)")
        self.assertEqual([list(choice) for choice in by_key["anima_t2i_sampler"]["choices"]],
                         [[name, name] for name in base.SAMPLERS],
                         "Forge's get_value_for_setting: the choices come along as on page load")
        self.assertEqual(by_key["anima_t2i_width"], {"__type__": "update"}, "kept at 0: not written, not touched")
        self.assertEqual(by_key["xl_t2i_sampler"], {"__type__": "update"}, "another preset")

    def test_a_refused_save_changes_no_field(self):
        # refused by the handler, not by Gradio's own input checks (see WiringTests.test_a_refused_save_reaches_the_page_as_json)
        demo, dep, components = self.build()
        page = page_values(demo)
        before = dict(page)
        self.host.samplers = [name for name in base.SAMPLERS if name != "ER SDE (Tunable)"]
        payload, outputs = self.save(demo, dep, page)
        self.assertFalse(payload["ok"])
        self.assertIn("txt2img Sampler", payload["error"])
        self.assertEqual(len(outputs), 2, "the refresh still runs after a refused save")
        self.assertTrue(all(item == {"__type__": "update"} for item in outputs[1][1]))
        self.assertEqual({k: page[c._id] for k, c in components.items()}, {k: before[c._id] for k, c in components.items()})

    def test_a_settings_tab_that_is_not_on_the_screen_leaves_only_the_save(self):
        for name, kwargs in (("Settings tab hidden (hidden_tabs)", {"rendered": False}),
                             ("no field for any preset key", {"keys": []})):
            with self.subTest(name), contextlib.redirect_stderr(io.StringIO()) as err:
                usd._built_root = None
                demo, dep, _components = self.build(**kwargs)
                self.assertEqual(chained_after(demo, dep), [])
                page = page_values(demo)
                payload, outputs = self.save(demo, dep, page)
                self.assertTrue(payload["ok"], payload)
                self.assertEqual(len(outputs), 1)
                self.assertNotIn(usd.LOG_PREFIX, err.getvalue(), "nothing to report (library warnings aside)")

    def test_one_field_gets_its_value_not_a_list(self):
        """With one output Gradio reads the handler's return value as that field's value."""
        demo, dep, components = self.build(keys=["anima_t2i_sampler"])
        (after,) = chained_after(demo, dep)
        self.assertEqual(len(after.outputs), 1)
        page = page_values(demo)
        payload, _outputs = self.save(demo, dep, page)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(page[components["anima_t2i_sampler"]._id], "ER SDE (Tunable)")

    def test_a_field_whose_value_cannot_be_read_is_left_alone(self):
        real = self.ui_settings.get_value_for_setting

        def get_value(key):
            if key == "anima_t2i_cfg":
                raise KeyError(key)
            return real(key)

        self.ui_settings.get_value_for_setting = get_value
        demo, dep, components = self.build()
        page = page_values(demo)
        before = page[components["anima_t2i_cfg"]._id]
        with contextlib.redirect_stderr(io.StringIO()) as err:
            text = usd.settings_updates(sd.to_json({"ok": True, "written_keys": ["anima_t2i_cfg", "anima_t2i_step"]}),
                                        ["anima_t2i_cfg", "anima_t2i_step", "xl_t2i_step"])
        self.assertIn("anima_t2i_cfg not refreshed", err.getvalue())
        self.assertEqual(text[0], {"__type__": "update"})
        self.assertEqual(text[1]["value"], self.opts.anima_t2i_step)
        self.assertEqual(text[2], {"__type__": "update"})
        payload, _outputs = self.save(demo, dep, page)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(page[components["anima_t2i_cfg"]._id], before)
        self.assertEqual(page[components["anima_t2i_sampler"]._id], "ER SDE (Tunable)")

    def test_a_failed_lookup_still_wires_the_save_button(self):
        def broken():
            raise RuntimeError("no presets here")

        sys.modules["modules_forge.presets"].PresetArch.choices = staticmethod(broken)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            demo, dep, _components = self.build()
        self.assertIn("Settings fields are not refreshed after a save", err.getvalue())
        self.assertNotIn("the button is not wired", err.getvalue())
        self.assertEqual(chained_after(demo, dep), [])
        payload, _outputs = self.save(demo, dep, page_values(demo))
        self.assertTrue(payload["ok"], payload)

    def test_settings_value_falls_back_to_the_option_value(self):
        """A Forge without ui_settings.get_value_for_setting: the value alone."""
        del self.ui_settings.get_value_for_setting
        self.assertEqual(usd.settings_value("anima_t2i_step"), {"__type__": "update", "value": self.opts.anima_t2i_step})


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


def _forge_ui_settings(namespace):
    """modules/ui_settings.py get_value_for_setting · create_setting_component and UiSettings.run_settings, run as they
    are written (the module itself imports Forge's whole UI)."""
    tree = ast.parse(require_forge_file("modules/ui_settings.py").read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    settings_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UiSettings")
    methods = {node.name: node for node in settings_class.body if isinstance(node, ast.FunctionDef)}
    body = [functions["get_value_for_setting"], functions["create_setting_component"], methods["run_settings"]]
    exec(compile(ast.Module(body=body, type_ignores=[]), "modules/ui_settings.py", "exec"), namespace)
    return namespace["get_value_for_setting"], namespace["create_setting_component"], namespace["run_settings"]


def _forge_opts(labels, data):
    """FakeOpts with modules/options.py Options.same_type and typemap (Forge's own) and dumpjson — what run_settings
    calls."""
    tree = ast.parse(require_forge_file("modules/options.py").read_text(encoding="utf-8"))
    options = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Options")
    body = [node for node in options.body
            if (isinstance(node, ast.FunctionDef) and node.name == "same_type")
            or (isinstance(node, ast.Assign) and [ast.unparse(t) for t in node.targets] == ["typemap"])]
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), "modules/options.py", "exec"), namespace)
    same_type = namespace["same_type"]

    class ForgeOpts(base.FakeOpts):
        def same_type(self, x, y):
            return same_type(self, x, y)

        def dumpjson(self):
            return json.dumps(self.data, default=str)

    opts = ForgeOpts(labels, data)
    object.__setattr__(opts, "typemap", namespace["typemap"])
    return opts


class ForgeSettingsTests(unittest.TestCase):
    """Forge's own Settings code with the real preset options (modules_forge/presets.py register): the fields
    create_setting_component builds, the values get_value_for_setting puts in (on page load and now after a save),
    and run_settings — the Apply settings that undid the save in the live v0.34.0 check. Skipped without a Forge checkout."""

    def test_forge_keeps_the_settings_fields_where_the_button_looks(self):
        tree = ast.parse(require_forge_file("modules/ui_settings.py").read_text(encoding="utf-8"))
        self.assertIn("get_value_for_setting", {node.name for node in tree.body if isinstance(node, ast.FunctionDef)})
        settings_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UiSettings")
        methods = {node.name: ast.unparse(node) for node in settings_class.body if isinstance(node, ast.FunctionDef)}
        self.assertIn("shared.settings_components = self.component_dict", methods["create_ui"])
        self.assertIn("self.component_dict[k] = component", methods["add_quicksettings"])
        # Apply settings writes every field's page value — why a field must hold what the save wrote
        self.assertIn("zip(opts.data_labels.keys(), args, self.components)", methods["run_settings"])
        self.assertIn("opts.set(key, value)", methods["run_settings"])
        self.assertIn("inputs=self.components", methods["add_functionality"])
        # the Settings fields exist before the footer (where the bridge is built): create_ui order in modules/ui.py
        ui = ast.parse(require_forge_file("modules/ui.py").read_text(encoding="utf-8"))
        create_ui = next(node for node in ui.body if isinstance(node, ast.FunctionDef) and node.name == "create_ui")
        lines = {}
        for node in ast.walk(create_ui):
            if isinstance(node, ast.Call):
                text = ast.unparse(node.func)
                if text in ("settings.create_ui", "settings.add_quicksettings"):
                    lines.setdefault(text, []).append(node.lineno)
                if text == "gr.HTML" and any(k.arg == "elem_id" and getattr(k.value, "value", None) == "footer"
                                             for k in node.keywords):
                    lines.setdefault("footer", []).append(node.lineno)
        self.assertEqual(sorted(lines), ["footer", "settings.add_quicksettings", "settings.create_ui"])
        self.assertLess(max(lines["settings.create_ui"] + lines["settings.add_quicksettings"]), min(lines["footer"]))

    def test_apply_settings_after_a_save_keeps_the_saved_values(self):
        presets_module = base._forge_presets_module()
        labels = base.forge_labels(presets_module)
        presets = presets_module.PresetArch.choices()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        opts = _forge_opts(labels, base.sizes())
        host = base.make_host(labels=labels, presets=presets, tmp=tmp.name, opts=opts)
        namespace = {"opts": opts, "gr": gr, "FormRow": gr.Row, "CURRENT_ROW": None,
                     "ui_common": types.SimpleNamespace(create_refresh_button=lambda *args, **kwargs: None),
                     "shared": types.SimpleNamespace(config_filename=host.config_filename)}
        get_value, create_setting_component, run_settings = _forge_ui_settings(namespace)
        for info in labels.values():
            if not hasattr(info, "refresh"):
                info.refresh = None                               # OptionInfo's; FakeInfo keeps none
            if isinstance(info, base.FakeRow):
                info.component = gr.Row                           # modules/options.py OptionRow: component=FormRow
        # UiSettings.create_ui: a field per option with a shown section, the dummy for the rest (in label order)
        dummy = gr.Textbox(visible=False, render=False)
        component_dict, components = {}, []
        with gr.Blocks() as interface:
            for key, info in labels.items():
                if info.section is None or info.section[0] is None:
                    components.append(dummy)
                    continue
                component_dict[key] = create_setting_component(key)
                components.append(component_dict[key])
        self.assertIn("anima_t2i_sampler", component_dict)
        self.assertNotIn("forge_checkpoint_anima", component_dict, "Forge Hidden Options")

        usd._built_root = None
        main_entry, stubs = _main_entry_stub()
        stubs.update(_settings_stubs(stubs["modules_forge"], component_dict, opts=opts, get_value=get_value,
                                     presets=presets))
        with _modules(stubs):
            with _forge_after_component(usd.on_after_component):
                demo = build_ui(main_entry, settings=interface)
            (run,) = by_elem_id(demo, usd.RUN_ELEM_ID)
            dep = click_dependency(demo, run)
            # the page as UiSettings.add_functionality's demo.load leaves it
            page = page_values(demo)
            for key, component in component_dict.items():
                if not component.stateful:
                    page[component._id] = get_value(key)["value"]
            self.assertEqual(page[component_dict["anima_t2i_sampler"]._id], opts.anima_t2i_sampler)
            (result,) = by_elem_id(demo, usd.RESULT_ELEM_ID)
            with mock.patch.object(usd, "forge_host", lambda: host), contextlib.redirect_stdout(io.StringIO()):
                outputs = run_on_page(demo, dep, ["req-9", "anima", *base.ui_values(), page[result._id]], page)
        payload = json.loads(outputs[0][1][0])
        self.assertTrue(payload["ok"], payload)
        saved = {key: getattr(opts, key) for key in labels}
        self.assertEqual(saved["anima_t2i_sampler"], "ER SDE (Tunable)")
        # Settings → Apply settings in the same page (Forge's run_settings with every field's page value)
        fake_self = types.SimpleNamespace(components=components, dummy_component=dummy)
        args = [None if component is dummy else page[component._id] for component in components]
        _json, message = run_settings(fake_self, *args)
        self.assertEqual(opts.anima_t2i_sampler, "ER SDE (Tunable)", f"Apply settings put the old value back ({message})")
        self.assertEqual({key: getattr(opts, key) for key in labels}, saved, message)


if __name__ == "__main__":
    unittest.main()
