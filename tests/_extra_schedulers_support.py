"""Helpers shared by the tests/test_extra_schedulers_*.py files (not a test module itself).

* ``forge_root()`` — the Forge checkout to read Forge's own code from: two levels above the
  extension (its normal place, ``extensions/<ext>``), else ``$SAM3_FORGE_ROOT``, else the
  development PC's ``C:\\sd-webui-forge-classic``. Tests that need it skip when none exists.
* ``forge_function(rel, name)`` — one function (or ``Class.method``) cut out of a Forge source file
  with ``ast`` and executed in a namespace the test provides: Forge's real code runs, nothing of
  Forge is imported, and nothing is copied into this repository.
* ``stub_modules(...)`` — temporarily replace ``sys.modules`` entries (and restore them).
* ``load_script(...)`` — load ``scripts/anima_extra_schedulers.py`` against a stub ``modules`` package.
"""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import functools
import importlib.util
import io
import os
import re
import sys
import types
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_DEV_FORGE = Path(r"C:\sd-webui-forge-classic")


def forge_root() -> Path | None:
    candidates = [ROOT.parents[1]]
    env = os.environ.get("SAM3_FORGE_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.append(_DEV_FORGE)
    for candidate in candidates:
        if (candidate / "modules" / "sd_schedulers.py").is_file() and (
            candidate / "modules_forge" / "packages" / "k_diffusion" / "sampling.py"
        ).is_file():
            return candidate
    return None


def forge_source(rel: str) -> str:
    root = forge_root()
    if root is None:
        raise FileNotFoundError("Forge checkout not found")
    return (root / rel).read_text(encoding="utf-8")


def _find(tree: ast.Module, name: str):
    parts = name.split(".")
    body = tree.body
    node = None
    for part in parts:
        node = next(
            (n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.name == part),
            None,
        )
        if node is None:
            raise LookupError(f"{name} not found")
        body = getattr(node, "body", [])
    return node


def forge_function_source(rel: str, name: str) -> str:
    """Source of a top-level function or ``Class.method`` (decorators included, dedented)."""
    source = forge_source(rel)
    node = _find(ast.parse(source), name)
    lines = ast.get_source_segment(source, node).split("\n")
    # A method's segment starts at ``def`` but the following lines keep the class indentation.
    if node.col_offset:
        lines = [lines[0]] + [line[node.col_offset:] if line.strip() else line for line in lines[1:]]
    decorators = [ast.get_source_segment(source, d) for d in getattr(node, "decorator_list", [])]
    return "\n".join([*(f"@{d}" for d in decorators), *lines]) + "\n"


def forge_function(rel: str, name: str, namespace: dict) -> Callable:
    """Execute Forge's ``name`` from ``rel`` in ``namespace`` and return it."""
    code = forge_function_source(rel, name)
    exec(compile(code, f"<forge:{rel}:{name}>", "exec"), namespace)
    return namespace[name.split(".")[-1]]


def forge_constant(rel: str, name: str):
    """The literal value of a top-level ``name = <literal>`` in a Forge source file."""
    tree = ast.parse(forge_source(rel))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise LookupError(f"{name} not found in {rel}")


@contextlib.contextmanager
def stub_modules(stubs: dict):
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


# ── a stand-in for Forge's modules.sd_schedulers ──
@dataclasses.dataclass
class StubScheduler:
    """Same fields as Forge's ``modules.sd_schedulers.Scheduler`` (sd_schedulers.py:22-30)."""

    name: str
    label: str
    function: Callable

    default_rho: float = -1.0
    need_inner_model: bool = False
    aliases: list = None


def stub_sd_schedulers(entries=(("automatic", "Automatic"), ("karras", "Karras"), ("exponential", "Exponential")),
                       hidden=()) -> types.ModuleType:
    module = types.ModuleType("modules.sd_schedulers")
    module.Scheduler = StubScheduler
    module.all_schedulers = [StubScheduler(name, label, (lambda *a, **k: None) if name != "automatic" else None)
                             for name, label in entries]
    module.schedulers = [s for s in module.all_schedulers if s.label not in hidden]
    module.schedulers_map = {**{x.name: x for x in module.schedulers}, **{x.label: x for x in module.schedulers}}
    return module


def load_script(*, sd_schedulers=None, scripts_data=None, sections_enabled=True, hidden_schedulers=()):
    """Load scripts/anima_extra_schedulers.py with a stub ``modules`` package (no WebUI).

    ``hidden_schedulers`` is Settings → Hide Schedulers (``opts.hide_schedulers``) at load time."""
    modules_stub = types.ModuleType("modules")
    modules_stub.__path__ = []

    class Script:
        is_img2img = False

        def elem_id(self, item_id):
            # Forge's Script.elem_id (modules/scripts.py) for a script shown on both tabs — the id comes
            # from the title (test_extra_schedulers_script.ForgeScriptIdentityTests runs Forge's own).
            title = re.sub(r"[^a-z_0-9]", "", re.sub(r"\s", "_", self.title().lower()))
            return f"script_{'img2img' if self.is_img2img else 'txt2img'}_{title}_{item_id}"

    callbacks = []
    modules_stub.scripts = types.SimpleNamespace(
        Script=Script, AlwaysVisible=object(), scripts_data=list(scripts_data or [])
    )
    modules_stub.script_callbacks = types.SimpleNamespace(
        on_before_ui=lambda fn: callbacks.append(("before_ui", fn)),
        on_ui_settings=lambda fn: callbacks.append(("ui_settings", fn)),
    )
    modules_stub.shared = types.SimpleNamespace(
        opts=types.SimpleNamespace(hide_schedulers=list(hidden_schedulers), sam3_layout_sections=sections_enabled)
    )
    if sd_schedulers is None:
        sd_schedulers = stub_sd_schedulers()
    modules_stub.sd_schedulers = sd_schedulers
    modules_stub.sd_samplers = types.SimpleNamespace(
        get_sampler_and_scheduler=functools.cache(lambda sampler, scheduler: (sampler, scheduler))
    )
    modules_stub.sd_samplers_kdiffusion = types.SimpleNamespace(k_diffusion_scheduler={})
    stubs = {"modules": modules_stub, "modules.sd_schedulers": sd_schedulers, "modules.shared": modules_stub.shared,
             "modules.sd_samplers": modules_stub.sd_samplers,
             "modules.sd_samplers_kdiffusion": modules_stub.sd_samplers_kdiffusion}
    stderr = io.StringIO()
    with stub_modules(stubs), contextlib.redirect_stderr(stderr):
        spec = importlib.util.spec_from_file_location("_test_extra_schedulers_script", ROOT / "scripts" / "anima_extra_schedulers.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    module._test_callbacks = callbacks
    module._test_modules_stub = modules_stub
    module._test_stderr = stderr.getvalue()
    return module
