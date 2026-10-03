"""scripts/composition_camera.py — 구도 · 카메라 칸의 자리(after_component)와 설정.

스크립트는 Forge 스텁(modules.script_callbacks · modules.shared) 위에서 진짜 Gradio 4.40 으로 돈다(tests/
test_mcp_settings_script.py 와 같은 방식). 자리는 Forge 가 하듯 생성자 안에서 콜백을 불러 확인한다 — webui 는
Component.__init__·BlockContext.__init__ 이 돌아온 직후(``with`` 로 들어가기 전) after_component 를 부른다
(modules/gradio_extensions.py). 기본 배치와 Compact 배치는 modules/ui_toprow.py 의 컨테이너 중첩을 그대로 흉내 내고,
Forge 소스가 옆에 있으면 그 중첩이 아직 그대로인지 AST 로 확인한다(없으면 건너뜀 — CI).
"""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no version check over the network (Blocks construction)

import ast  # noqa: E402
import configparser  # noqa: E402
import contextlib  # noqa: E402
import importlib.util  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import gradio as gr  # noqa: E402
import gradio.blocks  # noqa: E402
from gradio.context import Context  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FORGE = ROOT.parents[1]
SCRIPT = ROOT / "scripts" / "composition_camera.py"


class _OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None,
                 refresh=None, comment_before="", comment_after="", infotext=None, restrict_api=False, category_id=None):
        self.default = default
        self.label = label
        self.component = component
        self.section = section
        self.comment = None

    def info(self, text):
        self.comment = text
        return self


class _Opts:
    def __init__(self):
        self.added = {}

    def add_option(self, key, info):
        self.added[key] = info


def _load():
    registered: list = []
    package = types.ModuleType("modules")
    package.__path__ = []
    callbacks = types.ModuleType("modules.script_callbacks")
    callbacks.on_ui_settings = lambda fn, **kw: registered.append(("on_ui_settings", fn))
    callbacks.on_after_component = lambda fn, **kw: registered.append(("on_after_component", fn))
    shared = types.ModuleType("modules.shared")
    shared.OptionInfo = _OptionInfo
    shared.opts = _Opts()
    package.script_callbacks = callbacks
    package.shared = shared
    stubs = {"modules": package, "modules.script_callbacks": callbacks, "modules.shared": shared}
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("_test_composition_camera", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module, registered


@contextlib.contextmanager
def _forge_after_component(*callbacks):
    """modules/gradio_extensions.py IOComponent_init / BlockContext_init: each callback runs, in order, right after the
    original __init__ returned — before a ``with`` enters a container."""
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


def _tipo_like(component, **kwargs):
    """!sam3.py's on_after_component for the same row: the TIPO accordion (sam3ext/ui_tipo.py build_tipo_panel)."""
    if Context.root_block is None or component._id not in Context.root_block.default_config.blocks:
        return
    if kwargs.get("elem_id") == "txt2img_styles_row":
        with gr.Accordion("TIPO 프롬프트 확장", open=False, elem_id="sam3_tipo_accordion"):
            gr.Radio(["a", "b"], elem_id="sam3_tipo_mode")


def _styles_ui(tab):
    """modules/ui_prompt_styles.py UiPromptStyles.__init__: the styles row, then the styles dialog group."""
    with gr.Row(elem_id=f"{tab}_styles_row"):
        gr.Dropdown(label="Styles", show_label=False, elem_id=f"{tab}_styles", choices=[], value=[], multiselect=True)
        gr.Button("🖌️", elem_id=f"{tab}_styles_edit_button")
    with gr.Group(elem_id=f"{tab}_styles_dialog", elem_classes="popup-dialog"):
        gr.Textbox(label="Prompt", elem_id=f"{tab}_edit_style_prompt")


def _classic(tab):
    """modules/ui_toprow.py Toprow.create_classic_toprow (Default · Scrollable · Accordion prompt layouts)."""
    with gr.Blocks() as demo:
        with gr.Row(elem_id=f"{tab}_toprow", variant="compact"):
            with gr.Column(elem_id=f"{tab}_prompt_container", scale=6):
                with gr.Row(elem_id=f"{tab}_prompt_row"):
                    gr.Textbox(label="Prompt", elem_id=f"{tab}_prompt", show_label=False, lines=3)
            with gr.Column(scale=1, elem_id=f"{tab}_actions_column") as column:
                with gr.Row(elem_id=f"{tab}_generate_box"):
                    gr.Button("Generate", elem_id=f"{tab}_generate")
                with gr.Row(elem_id=f"{tab}_tools"):
                    gr.Button("📋", elem_id=f"{tab}_style_apply")
                _styles_ui(tab)
    return demo, column


def _compact(tab):
    """modules/ui_toprow.py with is_compact: create_submit_box (render=False), then create_inline_toprow_prompts inside
    the settings column (modules/ui.py, category "prompt"), and the generate box rendered in the results column."""
    with gr.Blocks() as demo:
        with gr.Row(elem_id=f"{tab}_generate_box", render=False) as submit_box:
            gr.Button("Generate", elem_id=f"{tab}_generate")
        with gr.Column(elem_id=f"{tab}_settings"):
            with gr.Column(elem_id=f"{tab}_prompt_container", elem_classes="prompt-container-compact", scale=6):
                with gr.Row(elem_id=f"{tab}_prompt_row"):
                    gr.Textbox(label="Prompt", elem_id=f"{tab}_prompt", show_label=False, lines=3)
            with gr.Row(elem_classes=["toprow-compact-stylerow"]) as stylerow:
                with gr.Column(elem_classes=["toprow-compact-tools"]):
                    with gr.Row(elem_id=f"{tab}_tools"):
                        gr.Button("📋", elem_id=f"{tab}_style_apply")
                with gr.Column() as column:
                    _styles_ui(tab)
        with gr.Column(elem_id=f"{tab}_results"):
            submit_box.render()
    return demo, column, stylerow


def _ids(container):
    return [getattr(child, "elem_id", None) for child in container.children]


def _mounts(demo):
    return [block for block in demo.blocks.values() if isinstance(block, gr.HTML)
            and str(block.elem_id or "").startswith("sam3_composition_")]


class SettingsTests(unittest.TestCase):
    def test_registers_one_settings_and_one_after_component_callback(self):
        module, registered = _load()
        self.assertEqual(registered, [("on_ui_settings", module.on_ui_settings),
                                      ("on_after_component", module.on_after_component)])
        self.assertEqual(module.shared.opts.added, {}, "nothing is added before Forge asks")

    def test_the_switch_is_on_by_default_in_the_appearance_section(self):
        module, _ = _load()
        module.on_ui_settings()
        self.assertEqual(list(module.shared.opts.added), ["sam3_composition_panel"])
        info = module.shared.opts.added["sam3_composition_panel"]
        self.assertIs(info.default, True)
        self.assertIs(info.component, gr.Checkbox)
        self.assertEqual(info.section, ("sam3_appearance", "SAM Extra Appearance"))
        self.assertIn("구도 · 카메라", info.label)
        self.assertIn("Reload UI", info.label, "the label says when it applies")
        self.assertIn("재시작", info.label)
        self.assertIn("3D 카메라 제어가 아닙니다", info.comment)

    def test_section_is_the_existing_appearance_section(self):
        source = (ROOT / "scripts" / "appearance_theme.py").read_text(encoding="utf-8")
        self.assertIn('section = ("sam3_appearance", "SAM Extra Appearance")', source)
        module, _ = _load()
        self.assertEqual(module.SECTION, ("sam3_appearance", "SAM Extra Appearance"))

    def test_an_unregistered_option_reads_as_on(self):
        module, _ = _load()
        self.assertTrue(module.panel_enabled(), "shared.opts without the key (first start before ui_settings)")
        module.shared.opts.sam3_composition_panel = False
        self.assertFalse(module.panel_enabled())


class MountTests(unittest.TestCase):
    def setUp(self):
        self.module, _ = _load()

    def test_each_tab_gets_one_placeholder_right_below_its_styles_row(self):
        for tab in ("txt2img", "img2img"):
            with self.subTest(tab=tab), _forge_after_component(self.module.on_after_component):
                demo, column = _classic(tab)
                self.assertEqual(_ids(column), [f"{tab}_generate_box", f"{tab}_tools", f"{tab}_styles_row",
                                                f"sam3_composition_{tab}", f"{tab}_styles_dialog"])
                html = column.children[3]
                self.assertIsInstance(html, gr.HTML)
                self.assertEqual(html.value, f'<div class="sam3-composition-mount" data-sam3-composition-tab="{tab}"></div>')
                self.assertIn("sam3-composition-host", html.elem_classes)
                self.assertEqual(len(_mounts(demo)), 1)

    def test_in_txt2img_it_comes_after_the_tipo_panel(self):
        with _forge_after_component(_tipo_like, self.module.on_after_component):
            _, column = _classic("txt2img")
        self.assertEqual(_ids(column), ["txt2img_generate_box", "txt2img_tools", "txt2img_styles_row",
                                        "sam3_tipo_accordion", "sam3_composition_txt2img", "txt2img_styles_dialog"])

    def test_compact_prompt_layout_still_gets_the_placeholder(self):
        # Compact: the styles row sits in the second column of the toprow-compact-stylerow row inside the settings
        # column, next to the tools column — the panel lands there, under the styles row (and TIPO), at full width
        # of that column. The render=False generate box does not matter: the styles row itself is rendered.
        for tab in ("txt2img", "img2img"):
            with self.subTest(tab=tab), _forge_after_component(_tipo_like, self.module.on_after_component):
                demo, column, stylerow = _compact(tab)
                expected = [f"{tab}_styles_row"] + (["sam3_tipo_accordion"] if tab == "txt2img" else []) \
                    + [f"sam3_composition_{tab}", f"{tab}_styles_dialog"]
                self.assertEqual(_ids(column), expected)
                self.assertIs(stylerow.children[1], column)
                self.assertEqual(len(_mounts(demo)), 1)

    def test_the_placeholder_reaches_the_frontend_config(self):
        with _forge_after_component(self.module.on_after_component):
            demo, _ = _classic("img2img")
        config = demo.get_config_file()
        found = [c for c in config["components"] if c["props"].get("elem_id") == "sam3_composition_img2img"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["type"], "html")
        self.assertIn("sam3-composition-host", found[0]["props"]["elem_classes"])
        self.assertIn('data-sam3-composition-tab="img2img"', found[0]["props"]["value"])

    def test_other_components_are_ignored(self):
        with gr.Blocks() as demo:
            with gr.Column() as column:
                for elem_id in (None, "txt2img_styles", "txt2img_prompt", "extras_styles_row", "txt2img_styles_row_x"):
                    row = gr.Row(elem_id=elem_id)
                    self.module.on_after_component(row, elem_id=elem_id)
        self.assertEqual(_mounts(demo), [])
        self.assertEqual(len(column.children), 5)

    def test_outside_a_blocks_build_nothing_is_created(self):
        self.assertIsNone(Context.root_block)
        before = Context.id
        row = gr.Row(elem_id="txt2img_styles_row", render=False)
        self.module.on_after_component(row, elem_id="txt2img_styles_row")
        self.assertEqual(Context.id, before + 1, "only the row itself was constructed")

    def test_a_request_time_rebuild_never_adds_a_second_misplaced_mount(self):
        # Gradio's postprocess rebuilds a component with render=False (blocks.py postprocess_data); during a Reload UI
        # build Context.root_block is the NEW ui, so a root_block check alone would put an HTML into whatever column
        # that build is in. The registration check stops it.
        with gr.Blocks() as new_ui:
            with gr.Column() as column:
                echo = gr.Row(elem_id="txt2img_styles_row", render=False)
                self.module.on_after_component(echo, elem_id="txt2img_styles_row")
        self.assertEqual(column.children, [])
        self.assertEqual(_mounts(new_ui), [])

    def test_a_registered_row_inside_a_render_false_container_still_counts(self):
        with gr.Blocks() as demo:
            with gr.Column(render=False) as deferred:
                row = gr.Row(elem_id="img2img_styles_row")
                self.module.on_after_component(row, elem_id="img2img_styles_row")
            deferred.render()
        self.assertEqual(_ids(deferred), ["img2img_styles_row", "sam3_composition_img2img"])
        self.assertEqual(len(_mounts(demo)), 1)

    def test_switched_off_no_placeholder(self):
        self.module.shared.opts.sam3_composition_panel = False
        with _forge_after_component(_tipo_like, self.module.on_after_component):
            demo, column = _classic("txt2img")
        self.assertEqual(_mounts(demo), [])
        self.assertNotIn("sam3_composition_txt2img", _ids(column))

    def test_once_per_ui_build_and_again_after_reload_ui(self):
        with _forge_after_component(self.module.on_after_component):
            with gr.Blocks() as first:
                with gr.Column():
                    _styles_ui("txt2img")
                    _styles_ui("txt2img")       # a second row with the same id in one UI: no duplicate elem_id
            self.assertEqual(len(_mounts(first)), 1)
            demo, _ = _classic("txt2img")       # Reload UI builds new Blocks
            self.assertEqual(len(_mounts(demo)), 1)


def _forge_source(rel):
    path = FORGE / rel
    if not path.is_file():
        raise unittest.SkipTest(f"Forge source not found: {path}")
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _with_path_to_call(body, call_text, trail=()):
    """The ``with`` context expressions (unparsed) that enclose the first statement calling ``call_text``."""
    for statement in body:
        if isinstance(statement, ast.Expr) and ast.unparse(statement.value) == call_text:
            return list(trail)
        if isinstance(statement, ast.With):
            found = _with_path_to_call(statement.body, call_text,
                                       trail + tuple(ast.unparse(item.context_expr) for item in statement.items))
            if found is not None:
                return found
    return None


class ForgeLayoutSourceTests(unittest.TestCase):
    """The layouts simulated above are still Forge's (skipped without the Forge checkout next to the extension)."""

    def test_styles_row_is_the_first_container_of_the_styles_ui(self):
        init = _function(_forge_source("modules/ui_prompt_styles.py"), "__init__")
        withs = [node for node in init.body if isinstance(node, ast.With)]
        self.assertEqual(ast.unparse(withs[0].items[0].context_expr), "gr.Row(elem_id=f'{tabname}_styles_row')")
        self.assertTrue(ast.unparse(withs[1].items[0].context_expr).startswith("gr.Group(elem_id=f'{tabname}_styles_dialog'"))

    def test_classic_styles_ui_sits_in_the_actions_column(self):
        tree = _forge_source("modules/ui_toprow.py")
        path = _with_path_to_call(_function(tree, "create_classic_toprow").body, "self.create_styles_ui()")
        self.assertEqual(path, ["gr.Column(scale=1, elem_id=f'{self.id_part}_actions_column')"])

    def test_compact_styles_ui_sits_in_the_second_column_of_the_style_row(self):
        tree = _forge_source("modules/ui_toprow.py")
        inline = _function(tree, "create_inline_toprow_prompts")
        path = _with_path_to_call(inline.body, "self.create_styles_ui()")
        self.assertEqual(path, ["gr.Row(elem_classes=['toprow-compact-stylerow'])", "gr.Column()"])
        row = next(node for node in inline.body if isinstance(node, ast.With))
        columns = [ast.unparse(node.items[0].context_expr) for node in row.body if isinstance(node, ast.With)]
        self.assertEqual(columns, ["gr.Column(elem_classes=['toprow-compact-tools'])", "gr.Column()"])

    def test_after_component_fires_after_init_returns(self):
        tree = _forge_source("modules/gradio_extensions.py")
        for name, original in (("BlockContext_init", "original_BlockContext_init"),
                               ("IOComponent_init", "original_IOComponent_init")):
            lines = [ast.unparse(node) for node in _function(tree, name).body]
            init = next(i for i, line in enumerate(lines) if f"= {original}(" in line)
            callback = next(i for i, line in enumerate(lines) if "script_callbacks.after_component_callback(self" in line)
            self.assertLess(init, callback, name)


class LoadOrderTests(unittest.TestCase):
    def test_after_sam3_and_before_negpip_in_forge_file_order(self):
        # Forge: Extension.list_files sorts os.listdir; callbacks are registered while the files load, in that order.
        names = sorted(name for name in os.listdir(ROOT / "scripts") if name.endswith(".py"))
        self.assertLess(names.index("!sam3.py"), names.index("composition_camera.py"), "TIPO first, then this panel")
        self.assertEqual(names[-1], "negpip.py", "tests/test_integration_load_order.py keeps negpip.py last")

    def test_sam3_builds_tipo_on_the_same_row(self):
        source = (ROOT / "scripts" / "!sam3.py").read_text(encoding="utf-8")
        self.assertRegex(source, r'elif elem_id == "txt2img_styles_row":\s*\n\s*tipo_panel = _build_tipo_panel\(\)')

    def test_metadata_ini_does_not_reorder_after_component_callbacks(self):
        config = configparser.ConfigParser()
        self.assertTrue(config.read(ROOT / "metadata.ini", encoding="utf-8"))
        for section in config.sections():
            text = section + " " + " ".join(config[section].values())
            self.assertNotIn("after_component", text, section)
            self.assertNotIn("composition_camera", text, section)

    def test_not_an_always_on_script(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        self.assertEqual([node.name for node in tree.body if isinstance(node, ast.ClassDef)], [],
                         "no Script class: nothing for layout_lanes to place")


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, _ = _load()
        cls.ui = (ROOT / "javascript" / "composition_ui.js").read_text(encoding="utf-8")
        css = (ROOT / "style.css").read_text(encoding="utf-8")
        cls.css_text = css
        begin, end = "/* sam3-composition:begin */", "/* sam3-composition:end */"
        cls.block = css[css.index(begin) + len(begin):css.index(end)] if begin in css and end in css else ""

    def test_javascript_looks_for_the_same_mount(self):
        self.assertIn(f'var HOST_ID_PREFIX = "{self.module.MOUNT_ELEM_ID.format(tab="")}";', self.ui)
        self.assertIn(f'var MOUNT_CLASS = "{self.module.MOUNT_CLASS}";', self.ui)
        self.assertIn(f'var TABS = {list(self.module.TRIGGERS.values())!r};'.replace("'", '"'), self.ui)
        self.assertIn('app.querySelector("#" + tab + "_prompt")', self.ui)

    def test_new_files_are_utf8_without_bom(self):
        # Written with LF; not asserted here because a core.autocrlf=true checkout (this repo's Windows clones) turns
        # every text file into CRLF in the working tree — git stores LF either way.
        for rel in ("scripts/composition_camera.py", "javascript/composition_prompt.js", "javascript/composition_ui.js",
                    "tests/test_composition_camera.py", "tests/js/composition_prompt.test.mjs",
                    "tests/js/composition_prompt_origin.test.mjs", "tests/js/composition_ui.test.mjs",
                    "tests/_origin_composition_prompt/compositionPrompt.ts",
                    "tests/_origin_composition_prompt/package.json"):
            data = (ROOT / rel).read_bytes()
            with self.subTest(rel=rel):
                self.assertFalse(data.startswith(b"\xef\xbb\xbf"))
                data.decode("utf-8")

    def test_style_block_exists_once(self):
        self.assertEqual(self.css_text.count("/* sam3-composition:begin */"), 1)
        self.assertEqual(self.css_text.count("/* sam3-composition:end */"), 1)
        self.assertTrue(self.block.strip())

    def _rules(self):
        body = re.sub(r"/\*.*?\*/", "", self.block, flags=re.S)
        return [(selector.strip(), declarations) for selector, declarations in re.findall(r"([^{}]+)\{([^{}]*)\}", body)]

    @staticmethod
    def _selectors(selector):
        """Top-level comma split (not inside :is(...))."""
        parts, depth, current = [], 0, ""
        for char in selector:
            depth += char == "("
            depth -= char == ")"
            if char == "," and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += char
        return [part.strip() for part in parts + [current]]

    def test_every_selector_is_scoped_and_outranks_gradio_prose(self):
        # .gradio-container-4-40-0 .prose p/ul/li/button/input are (0,2,1); ours need three class-level parts.
        rules = self._rules()
        self.assertGreater(len(rules), 25)
        for selector, _ in rules:
            for part in self._selectors(selector):
                with self.subTest(selector=part):
                    # only the warning colour is theme-scoped (Gradio's .dark, sam-extra's [data-sam3-theme])
                    scoped = re.sub(r"^(\.dark|\[data-sam3-theme\]) (?=\.sam3-composition-mount )", "", part)
                    self.assertTrue(scoped.startswith(".sam3-composition-mount .sam3-composition"), part)
                    plain = re.sub(r":is\([^)]*\)", ":is", part)
                    weight = len(re.findall(r"\.[\w-]+|\[[^\]]+\]|(?<!:):(?!:)[\w-]+", plain))
                    self.assertGreaterEqual(weight, 3, part)

    def test_touch_targets_focus_rings_and_no_motion(self):
        coarse = self.block[self.block.index("@media (pointer: coarse)"):]
        for target in (".sam3-composition__summary", ".sam3-composition__button", ".sam3-composition__range input"):
            self.assertIn(target, coarse)
        self.assertIn("min-height: 44px", coarse)
        self.assertRegex(self.block, r":focus-visible \{\s*outline: 2px solid var\(--color-accent")
        lowered = re.sub(r"/\*.*?\*/", "", self.block, flags=re.S).lower()
        for banned in ("transition", "animation", "@keyframes", "z-index", "linear-gradient", "rgb(", "hsl(",
                       "box-shadow", "filter"):
            self.assertNotIn(banned, lowered, banned)
        self.assertEqual(re.findall(r"#[0-9a-f]{3,8}\b", lowered), [], "colours come from theme variables")
        self.assertEqual(lowered.count("!important"), 1, "only [hidden] { display: none !important }")


if __name__ == "__main__":
    unittest.main()
