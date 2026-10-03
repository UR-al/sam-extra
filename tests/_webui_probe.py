"""Load every script in ``scripts/`` the way Forge does and report what it registers (not a test module).

Run by ``tests/test_integration_load_order.py`` in a fresh interpreter (``python -B tests/_webui_probe.py``,
JSON on stdout): several scripts install hooks into Forge's modules when they are imported, so they
are never imported into the test process itself.

One ``modules`` stand-in for all scripts, with Forge's own pieces where the outcome depends on them:
``sd_samplers.set_samplers`` / ``add_sampler`` and ``sd_schedulers.Scheduler`` are executed from the
checkout, ``k_diffusion.sampling`` is Forge's file, the XYZ grid starts with the axis labels of Forge's
``scripts/xyz_grid.py``. The probe

1. executes the script files in Forge's order (sorted file names in one extension) and records every
   ``script_callbacks.on_*`` registration;
2. runs the ``on_ui_settings`` callbacks (option keys, sections), the ``on_before_ui`` callbacks
   against the XYZ grid (axis labels), the ``on_app_started`` callbacks on a FastAPI app (routes);
   instantiates every always-visible ``Script`` (title, ``section`` on txt2img);
3. simulates **Reload UI**: the script files are executed again (new module objects; ``sam3ext`` and
   Forge's modules stay imported), the callbacks registered anew run again — before_ui on a reloaded
   XYZ grid and on the one that was already there, app start on the same FastAPI app — and the sampler
   and scheduler lists are read once more.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import functools
import importlib.util
import io
import json
import os
import re
import sys
import types
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
FORGE = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402,F401

from tests import _extra_samplers_fixtures as esf  # noqa: E402


def forge_xyz_labels() -> list:
    """The labels of ``axis_options`` in Forge's scripts/xyz_grid.py (string literals, in order)."""
    tree = ast.parse((FORGE / "scripts" / "xyz_grid.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "axis_options" for t in node.targets):
            labels = []
            for item in node.value.elts:
                if isinstance(item, ast.Call) and item.args and isinstance(item.args[0], ast.Constant):
                    labels.append(item.args[0].value)
            return labels
    raise LookupError("axis_options not found in scripts/xyz_grid.py")


class AxisOption:
    """scripts/xyz_grid.py ``AxisOption`` (same signature)."""

    def __init__(self, label, type, apply, format_value=None, confirm=None, cost=0.0, choices=None, prepare=None):
        self.label, self.type, self.apply = label, type, apply
        self.format_value, self.confirm, self.cost = format_value, confirm, cost
        self.choices, self.prepare = choices, prepare


def new_xyz_grid():
    module = types.ModuleType("xyz_grid.py")
    module.AxisOption = AxisOption
    module.AxisOptionImg2Img = AxisOption
    module.AxisOptionTxt2Img = AxisOption
    module.axis_options = [AxisOption(label, str, None) for label in forge_xyz_labels()]
    return module


class Callbacks(types.ModuleType):
    """``modules.script_callbacks``: every ``on_*`` records ``(name, function)``."""

    def __init__(self):
        super().__init__("modules.script_callbacks")
        self.registered = []
        self.callback_map = {}

        class CFGDenoiserParams:
            pass

        self.CFGDenoiserParams = CFGDenoiserParams

    def __getattr__(self, item):
        if item.startswith("on_"):
            def register(callback=None, *args, **kwargs):
                self.registered.append((item, callback))
            return register
        raise AttributeError(item)


class OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None,
                 refresh=None, comment_before="", comment_after="", infotext=None, restrict_api=False, category_id=None):
        self.default, self.label, self.section, self.infotext = default, label, section, infotext

    def info(self, *args, **kwargs):
        return self

    def link(self, *args, **kwargs):
        return self

    def needs_reload_ui(self):
        return self

    def needs_restart(self):
        return self

    def html(self, *args, **kwargs):
        return self


class Options(types.SimpleNamespace):
    def __init__(self, added):
        super().__init__(hide_schedulers=[], hide_samplers=[], sam3_layout_sections=True)
        self._added = added

    def add_option(self, key, info):
        self._added.append((key, getattr(info, "section", None), getattr(info, "label", "")))

    def onchange(self, *args, **kwargs):
        pass


def forge_samplers_module(shared):
    """``modules.sd_samplers`` with Forge's own ``set_samplers`` / ``add_sampler`` and a few built-ins."""
    common = types.ModuleType("modules.sd_samplers_common")
    common.SamplerData = collections.namedtuple("SamplerData", ["name", "constructor", "aliases", "options"])
    samplers = types.ModuleType("modules.sd_samplers")
    samplers.shared = shared
    samplers.all_samplers = [common.SamplerData(name, None, [alias], {}) for name, alias in (
        ("Euler", "k_euler"), ("Euler a", "k_euler_a"), ("ER SDE", "er_sde"), ("DPM++ 2M", "k_dpmpp_2m"))]
    samplers.all_samplers_map = {x.name: x for x in samplers.all_samplers}
    samplers.samplers, samplers.samplers_for_img2img, samplers.samplers_map, samplers.samplers_hidden = [], [], {}, {}
    for name in ("set_samplers", "add_sampler"):
        exec(compile(esf.forge_definition("modules/sd_samplers.py", name), "modules/sd_samplers.py", "exec"),  # noqa: S102
             samplers.__dict__)
    samplers.get_sampler_and_scheduler = functools.cache(lambda sampler, scheduler: (sampler, scheduler))
    samplers.set_samplers()
    kdiffusion = types.ModuleType("modules.sd_samplers_kdiffusion")

    class KDiffusionSampler:
        def __init__(self, funcname, sd_model):
            self.func = funcname

    kdiffusion.KDiffusionSampler = KDiffusionSampler
    kdiffusion.k_diffusion_scheduler = {}
    return common, samplers, kdiffusion


def forge_schedulers_module():
    """``modules.sd_schedulers`` with Forge's ``Scheduler`` dataclass and three named built-ins."""
    module = types.ModuleType("modules.sd_schedulers")
    source = (FORGE / "modules" / "sd_schedulers.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    import dataclasses
    from typing import Callable

    namespace = {"dataclasses": dataclasses, "Callable": Callable}
    decorators = "".join(f"@{ast.get_source_segment(source, d)}\n" for d in node.decorator_list)
    code = compile(decorators + ast.get_source_segment(source, node), "modules/sd_schedulers.py", "exec")
    exec(code, namespace)  # noqa: S102
    Scheduler = namespace["Scheduler"]
    module.Scheduler = Scheduler
    module.all_schedulers = [Scheduler("automatic", "Automatic", None),
                             Scheduler("karras", "Karras", lambda *a, **k: None, default_rho=7.0),
                             Scheduler("exponential", "Exponential", lambda *a, **k: None)]
    module.schedulers = list(module.all_schedulers)
    module.schedulers_map = {**{x.name: x for x in module.schedulers}, **{x.label: x for x in module.schedulers}}
    return module


class Webui:
    """The ``modules`` stand-in; Forge's sampler/scheduler modules survive a reload, the rest is new."""

    def __init__(self):
        self.options = []
        self.persistent = {}
        shared = types.ModuleType("modules.shared")
        shared.opts = Options(self.options)
        common, samplers, kdiffusion = forge_samplers_module(shared)
        self.persistent.update({
            "modules.shared": shared, "modules.sd_samplers_common": common, "modules.sd_samplers": samplers,
            "modules.sd_samplers_kdiffusion": kdiffusion, "modules.sd_schedulers": forge_schedulers_module(),
        })
        shared.OptionInfo = OptionInfo
        shared.state = types.SimpleNamespace(sampling_step=0, sampling_steps=0, job="", job_count=0,
                                             interrupted=False, skipped=False, textinfo=None)
        shared.cmd_opts = types.SimpleNamespace()
        shared.sd_model = None
        self.k_sampling = esf.forge_k_sampling()

    def stubs(self, xyz):
        modules = types.ModuleType("modules")
        modules.__path__ = []

        class Script:
            is_img2img = False

            def title(self):
                return ""

            def elem_id(self, item_id):
                title = re.sub(r"[^a-z_0-9]", "", re.sub(r"\s", "_", self.title().lower()))
                return f"script_{'img2img' if self.is_img2img else 'txt2img'}_{title}_{item_id}"

        scripts = types.ModuleType("modules.scripts")
        scripts.Script = Script
        scripts.AlwaysVisible = object()
        scripts.scripts_data = [types.SimpleNamespace(script_class=type("Script", (), {"__module__": "xyz_grid.py"}),
                                                      module=xyz)]
        callbacks = Callbacks()
        processing = types.ModuleType("modules.processing")

        class Processing:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        for name in ("StableDiffusionProcessing", "StableDiffusionProcessingTxt2Img", "StableDiffusionProcessingImg2Img"):
            setattr(processing, name, type(name, (Processing,), {}))
        processing.process_images = lambda p: None
        postprocessing = types.ModuleType("modules.scripts_postprocessing")
        postprocessing.ScriptPostprocessing = type("ScriptPostprocessing", (), {})
        postprocessing.PostprocessedImage = object
        prompt_parser = types.ModuleType("modules.prompt_parser")
        prompt_parser.SdConditioning = list
        prompt_parser.get_learned_conditioning = None
        prompt_parser.get_learned_conditioning_prompt_schedules = None
        ui_components = types.ModuleType("modules.ui_components")
        ui_components.FormColorPicker = gr.ColorPicker
        ui_components.InputAccordion = None
        k_package = types.ModuleType("k_diffusion")
        k_package.__path__ = []
        k_package.sampling = self.k_sampling
        # sam3ext.negpip.anima / .sd patch Forge's text engines on import: stand-ins (tests/test_negpip_script.py)
        negpip_anima = types.ModuleType("sam3ext.negpip.anima")
        negpip_anima.patch_anima_negpip = lambda *a, **k: None
        negpip_sd = types.ModuleType("sam3ext.negpip.sd")
        negpip_sd.patch_sd_negpip = lambda *a, **k: None
        stubs = {
            "modules": modules, "modules.scripts": scripts, "modules.script_callbacks": callbacks,
            "modules.processing": processing, "modules.scripts_postprocessing": postprocessing,
            "modules.prompt_parser": prompt_parser, "modules.ui_components": ui_components,
            "k_diffusion": k_package, "k_diffusion.sampling": self.k_sampling,
            "sam3ext.negpip.anima": negpip_anima, "sam3ext.negpip.sd": negpip_sd,
            **self.persistent,
        }
        for name, module in stubs.items():
            if name.startswith("modules."):
                setattr(modules, name.split(".", 1)[1], module)
        return stubs, scripts, callbacks


@contextlib.contextmanager
def installed(stubs):
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


def script_files():
    """Forge's order inside one extension: ``Extension.list_files`` sorts ``os.listdir``."""
    return [ROOT / "scripts" / name for name in sorted(os.listdir(ROOT / "scripts")) if name.endswith(".py")]


def load_all(webui, xyz, generation):
    stubs, scripts_module, callbacks = webui.stubs(xyz)
    loaded, failed = [], {}
    with installed(stubs), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for path in script_files():
            name = f"_probe_{generation}_" + re.sub(r"\W", "_", path.stem)
            try:
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                loaded.append((path.name, module))
            except BaseException as exc:  # reported, never fatal
                failed[path.name] = f"{type(exc).__name__}: {exc}"
    return stubs, scripts_module, callbacks, loaded, failed


def run_callbacks(stubs, callbacks, kind, *args):
    errors = []
    with installed(stubs), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for name, callback in list(callbacks.registered):
            if name != kind or not callable(callback):
                continue
            try:
                callback(*args)
            except BaseException as exc:
                errors.append(f"{getattr(callback, '__module__', '?')}.{getattr(callback, '__name__', '?')}: "
                              f"{type(exc).__name__}: {exc}")
    return errors


def scripts_info(stubs, scripts_module, loaded):
    from sam3ext import layout_lanes

    info = []
    with installed(stubs), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for filename, module in loaded:
            for value in vars(module).values():
                if not (isinstance(value, type) and issubclass(value, scripts_module.Script)
                        and value is not scripts_module.Script and value.__module__ == module.__name__):
                    continue
                try:
                    script = value()
                    script.is_img2img = False
                    visible = script.show(False)
                    section = getattr(script, "section", None)
                    title = script.title()
                except BaseException as exc:
                    info.append({"file": filename, "class": value.__name__, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                info.append({
                    "file": filename, "class": value.__name__, "title": title,
                    "always": visible is scripts_module.AlwaysVisible, "section": section,
                    "key": layout_lanes.slot_key(filename),
                    "lane": layout_lanes.lane_for(layout_lanes.slot_key(filename)),
                    "registered": layout_lanes.slot_key(filename) in layout_lanes.REGISTRY,
                })
    return info


def routes(app):
    out = []
    for route in app.routes:
        methods = sorted(getattr(route, "methods", None) or [])
        out.append([getattr(route, "path", None), methods])
    return out


def main():
    from fastapi import FastAPI

    webui = Webui()
    app = FastAPI()
    demo = types.SimpleNamespace(blocks={}, dependencies=[], fns=[])
    report = {"order": [path.name for path in script_files()]}

    xyz = new_xyz_grid()
    report["forge_xyz"] = [option.label for option in xyz.axis_options]
    stubs, scripts_module, callbacks, loaded, failed = load_all(webui, xyz, 1)
    report["load_failed"] = failed
    report["callbacks"] = sorted({name for name, _ in callbacks.registered})
    report["settings_errors"] = run_callbacks(stubs, callbacks, "on_ui_settings")
    report["options"] = list(webui.options)
    report["before_ui_errors"] = run_callbacks(stubs, callbacks, "on_before_ui")
    report["xyz"] = [option.label for option in xyz.axis_options]
    report["app_errors"] = run_callbacks(stubs, callbacks, "on_app_started", demo, app)
    report["routes"] = routes(app)
    report["scripts"] = scripts_info(stubs, scripts_module, loaded)
    samplers = webui.persistent["modules.sd_samplers"]
    schedulers = webui.persistent["modules.sd_schedulers"]
    report["samplers"] = [x.name for x in samplers.all_samplers]
    report["schedulers"] = [x.label for x in schedulers.all_schedulers]
    report["visible_schedulers"] = [x.label for x in schedulers.schedulers]

    # Reload UI: the script files run again; Forge's modules and sam3ext stay; callbacks are new.
    del webui.options[:]
    reloaded_xyz = new_xyz_grid()
    stubs, scripts_module, callbacks, loaded, failed = load_all(webui, reloaded_xyz, 2)
    report["reload_load_failed"] = failed
    report["reload_settings_errors"] = run_callbacks(stubs, callbacks, "on_ui_settings")
    report["reload_options"] = list(webui.options)
    report["reload_before_ui_errors"] = run_callbacks(stubs, callbacks, "on_before_ui")
    report["reload_xyz"] = [option.label for option in reloaded_xyz.axis_options]
    # the XYZ module from before the reload, if a host kept it
    stubs_same, _, callbacks_same, _, _ = load_all(webui, xyz, 3)
    run_callbacks(stubs_same, callbacks_same, "on_before_ui")
    report["kept_xyz"] = [option.label for option in xyz.axis_options]
    report["reload_app_errors"] = run_callbacks(stubs, callbacks, "on_app_started", demo, app)
    report["reload_routes"] = routes(app)
    report["reload_samplers"] = [x.name for x in samplers.all_samplers]
    report["reload_samplers_map"] = sorted(samplers.samplers_map)
    report["reload_schedulers"] = [x.label for x in schedulers.all_schedulers]
    report["reload_visible_schedulers"] = [x.label for x in schedulers.schedulers]
    report["reload_scheduler_map"] = sorted(schedulers.schedulers_map)
    sys.stdout.write(json.dumps(report))


if __name__ == "__main__":
    main()
