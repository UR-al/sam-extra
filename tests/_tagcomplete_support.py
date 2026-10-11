"""Shared pieces of the built-in Tag Autocomplete tests (not a test module).

* ``TAC_KEYS`` — the 57 ``tac_*`` settings upstream registers, in registration order. They are exactly the 57 ``tac_*``
  keys of the development PC's ``config.json`` (read once on 2026-10-11 to write this list; no test reads that file).
* ``upstream_helper()`` — the installed sd-webui-tagcomplete-neo's ``scripts/tag_autocomplete_helper.py`` if this PC has
  it (``$SAM3_TAGCOMPLETE_UPSTREAM``, ``<forge>/extensions/sd-webui-tagcomplete-neo`` or, once moved aside at install time,
  ``<forge>/_removed_extensions/sd-webui-tagcomplete-neo``), else None.
* ``settings_from_source()`` — the settings an ``on_ui_settings`` of a helper source registers, evaluated from the
  source's AST (``sort_criteria`` and ``on_ui_settings`` only, nothing else of the module runs) against recording
  stand-ins and the real Gradio component classes.
* ``sandboxed()`` — stand-ins for Forge's modules and every folder the helper reads or writes inside a temporary
  directory, so ``sam3ext.tagcomplete.boot.start_helper()`` (or the entry script) can run in a test.
"""

from __future__ import annotations

import ast
import contextlib
import os
import shutil
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests._extra_schedulers_support import forge_root  # noqa: E402

UPSTREAM_COMMIT = "8e0d81fd0e25fd2cf98fb4d34d097df9f71a082b"
HELPER = ROOT / "sam3ext" / "tagcomplete" / "tag_autocomplete_helper.py"
SECTION = ["tac", "Tag Autocomplete"]

TAC_KEYS = (
    "tac_tagFile", "tac_active", "tac_activeIn.txt2img", "tac_activeIn.img2img", "tac_activeIn.negativePrompts",
    "tac_activeIn.thirdParty", "tac_activeIn.modelList", "tac_activeIn.modelListMode", "tac_slidingPopup",
    "tac_maxResults", "tac_showAllResults", "tac_resultStepLength", "tac_delayTime", "tac_useIndexedSearch",
    "tac_useWildcards", "tac_sortWildcardResults", "tac_wildcardExclusionList", "tac_skipWildcardRefresh",
    "tac_useEmbeddings", "tac_forceRefreshEmbeddings", "tac_includeEmbeddingsInNormalResults", "tac_useLoras",
    "tac_useLycos", "tac_useLoraPrefixForLycos", "tac_showWikiLinks", "tac_showExtraNetworkPreviews",
    "tac_modelSortOrder", "tac_useStyleVars", "tac_frequencySort", "tac_frequencyFunction", "tac_frequencyMinCount",
    "tac_frequencyMaxAge", "tac_frequencyRecommendCap", "tac_frequencyIncludeAlias", "tac_replaceUnderscores",
    "tac_undersocreReplacementExclusionList", "tac_escapeParentheses", "tac_appendComma", "tac_appendSpace",
    "tac_alwaysSpaceAtEnd", "tac_modelKeywordCompletion", "tac_modelKeywordLocation", "tac_modelKeywordCivitai",
    "tac_civitaiApiKey", "tac_wildcardCompletionMode", "tac_alias.searchByAlias", "tac_alias.onlyShowAlias",
    "tac_translation.translationFile", "tac_translation.oldFormat", "tac_translation.searchByTranslation",
    "tac_translation.liveTranslation", "tac_extra.extraFile", "tac_extra.addMode", "tac_chantFile", "tac_keymap",
    "tac_colormap", "tac_refreshTempFiles",
)

# (path, method) of upstream api_tac, in registration order
TACAPI_ROUTES = (
    ("/tacapi/v1/refresh-temp-files", "POST"), ("/tacapi/v1/refresh-embeddings", "POST"),
    ("/tacapi/v1/lora-info/{lora_name}", "GET"), ("/tacapi/v1/lyco-info/{lyco_name}", "GET"),
    ("/tacapi/v1/civitai-trigger-words/{lora_name}", "GET"), ("/tacapi/v1/lora-cached-hash/{lora_name}", "GET"),
    ("/tacapi/v1/thumb-preview/{filename}", "GET"), ("/tacapi/v1/thumb-preview-blob/{filename}", "GET"),
    ("/tacapi/v1/wildcard-contents", "GET"), ("/tacapi/v1/refresh-styles-if-changed", "GET"),
    ("/tacapi/v1/increase-use-count", "POST"), ("/tacapi/v1/get-use-count", "GET"),
    ("/tacapi/v1/get-use-count-list", "POST"), ("/tacapi/v1/reset-use-count", "PUT"),
    ("/tacapi/v1/get-all-use-counts", "GET"),
)

# the 15 upstream JS files in Forge's order (Extension.list_files: sorted(os.listdir))
JS_ORDER = (
    "__globals.js", "_baseParser.js", "_caretPosition.js", "_result.js", "_textAreas.js", "_utils.js",
    "ext_chants.js", "ext_embeddings.js", "ext_loras.js", "ext_lycos.js", "ext_modelKeyword.js", "ext_styles.js",
    "ext_umi.js", "ext_wildcards.js", "tagAutocomplete.js",
)

# what a file-list dropdown offers in both evaluations (the helper fills these from tags/ at import time)
FIXED_CSV_CHOICES = ["None", "danbooru.csv", "extra-quality-tags.csv"]
FIXED_JSON_CHOICES = ["None", "demo-chants.json"]

# helper-related modules a sandbox import puts into sys.modules (removed again afterwards)
_SAM3_MODULES = ("sam3ext.tagcomplete.tag_autocomplete_helper", "sam3ext.tagcomplete.shared_paths",
                 "sam3ext.tagcomplete.model_keyword_support", "sam3ext.tagcomplete.tag_frequency_db")


def upstream_helper() -> Path | None:
    candidates = []
    env = os.environ.get("SAM3_TAGCOMPLETE_UPSTREAM")
    if env:
        candidates.append(Path(env))
    forge = forge_root()
    if forge is not None:
        candidates += [forge / "extensions" / "sd-webui-tagcomplete-neo",
                       forge / "_removed_extensions" / "sd-webui-tagcomplete-neo"]
    for folder in candidates:
        helper = folder / "scripts" / "tag_autocomplete_helper.py"
        if helper.is_file():
            return helper
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Forge settings stand-ins
# ---------------------------------------------------------------------------------------------------------------------

class OptionInfo:
    """modules/options.py ``OptionInfo`` (same signature); records what its builder methods add."""

    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None,
                 refresh=None, comment_before="", comment_after="", infotext=None, restrict_api=False, category_id=None):
        self.default, self.label, self.component, self.component_args = default, label, component, component_args
        self.onchange, self.section, self.refresh, self.infotext = onchange, section, refresh, infotext
        self.comment_before, self.comment_after = comment_before, comment_after
        self.flags = []
        self.do_not_save = False

    def info(self, info):
        self.flags.append(("info", info))
        return self

    def html(self, html):
        self.flags.append(("html", html))
        return self

    def link(self, label, url):
        self.flags.append(("link", label, url))
        return self

    def needs_reload_ui(self):
        self.flags.append(("needs_reload_ui",))
        return self

    def needs_restart(self):
        self.flags.append(("needs_restart",))
        return self


class OptionHTML(OptionInfo):
    def __init__(self, text):
        super().__init__(str(text).strip(), label="")
        self.do_not_save = True


class Opts(types.SimpleNamespace):
    """``shared.opts``: attribute reads for settings, ``add_option`` records ``(key, OptionInfo)``."""

    def __init__(self, **values):
        super().__init__(**values)
        self.added = []

    def add_option(self, key, info):
        self.added.append((key, info))

    def onchange(self, *args, **kwargs):
        pass


def record(key, info) -> dict:
    """What a registration means for the Settings page and config.json — comparable across evaluations."""
    args = info.component_args() if callable(info.component_args) else info.component_args
    return {
        "key": key, "default": info.default, "label": info.label,
        "component": getattr(info.component, "__name__", None) if info.component is not None else None,
        "component_args": args, "section": list(info.section) if info.section else None,
        "refresh": getattr(info.refresh, "__name__", None), "onchange": getattr(info.onchange, "__name__", None),
        "infotext": info.infotext, "flags": list(info.flags), "do_not_save": info.do_not_save,
    }


def _named(name):
    def function(*args, **kwargs):
        return None
    function.__name__ = name
    return function


def settings_from_source(source: str) -> list[dict]:
    """Evaluate ``sort_criteria`` and ``on_ui_settings`` of a helper source (AST) and return the registrations."""
    import gradio as gr

    tree = ast.parse(source)
    wanted = [node for node in tree.body
              if (isinstance(node, ast.FunctionDef) and node.name == "on_ui_settings")
              or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "sort_criteria"
                                                       for t in node.targets))]
    if len(wanted) != 2:
        raise LookupError("sort_criteria / on_ui_settings not found")
    module = ast.Module(body=wanted, type_ignores=[])
    opts = Opts()
    shared = types.SimpleNamespace(opts=opts, OptionInfo=OptionInfo, OptionHTML=OptionHTML)
    namespace = {
        "shared": shared, "gr": gr,
        "csv_files_withnone": list(FIXED_CSV_CHOICES), "json_files_withnone": list(FIXED_JSON_CHOICES),
        "update_tag_files": _named("update_tag_files"), "update_json_files": _named("update_json_files"),
        "refresh_temp_files": _named("refresh_temp_files"),
    }
    exec(compile(module, "<tag_autocomplete_helper settings>", "exec"), namespace)  # noqa: S102
    namespace["on_ui_settings"]()
    return [record(key, info) for key, info in opts.added]


# ---------------------------------------------------------------------------------------------------------------------
# Importing the vendored helper against stand-ins
# ---------------------------------------------------------------------------------------------------------------------

class Callbacks(types.ModuleType):
    """``modules.script_callbacks``: every ``on_*`` records ``(name, function)``."""

    def __init__(self):
        super().__init__("modules.script_callbacks")
        self.registered = []

    def __getattr__(self, item):
        if item.startswith("on_"):
            def register(callback=None, *args, **kwargs):
                self.registered.append((item, callback))
            return register
        raise AttributeError(item)

    def named(self, kind):
        return [callback for name, callback in self.registered if name == kind]


@dataclass
class Sandbox:
    root: Path
    forge: Path
    tags: Path
    shared: types.ModuleType
    callbacks: Callbacks
    extensions: types.ModuleType

    @property
    def helper(self):
        return sys.modules.get("sam3ext.tagcomplete.tag_autocomplete_helper")


def forge_stubs(sandbox_root: Path, opts: Opts | None = None, active=()):
    """Stand-ins for the Forge modules the entry script and the helper import."""
    forge = sandbox_root / "forge"
    for name in ("embeddings", "models/Lora", "extensions", "extensions-builtin", "tmp"):
        (forge / name).mkdir(parents=True, exist_ok=True)
    modules = types.ModuleType("modules")
    modules.__path__ = []
    shared = types.ModuleType("modules.shared")
    shared.opts = opts if opts is not None else Opts()
    shared.OptionInfo = OptionInfo
    shared.OptionHTML = OptionHTML
    shared.cmd_opts = types.SimpleNamespace(embeddings_dir=str(forge / "embeddings"),
                                            lora_dir=str(forge / "models" / "Lora"))
    shared.prompt_styles = None
    callbacks = Callbacks()
    paths = types.ModuleType("modules.paths")
    paths.script_path = str(forge)
    paths.extensions_dir = str(forge / "extensions")
    extensions = types.ModuleType("modules.extensions")
    extensions.active = lambda: list(active)
    extensions.extensions = list(active)   # Forge: every extension folder, enabled or not (tests add disabled ones)
    extensions.extensions_builtin_dir = str(forge / "extensions-builtin")
    hashes = types.ModuleType("modules.hashes")
    hashes.sha256_from_cache = lambda *a, **k: None
    sd_models = types.ModuleType("modules.sd_models")
    errors = types.ModuleType("modules.errors")
    errors.reported = []
    errors.report = lambda message, exc_info=False: errors.reported.append(message)
    stubs = {"modules": modules, "modules.shared": shared, "modules.script_callbacks": callbacks,
             "modules.paths": paths, "modules.extensions": extensions, "modules.hashes": hashes,
             "modules.sd_models": sd_models, "modules.errors": errors}
    for name, module in stubs.items():
        if name != "modules":
            setattr(modules, name.split(".", 1)[1], module)
    return stubs, forge, shared, callbacks, extensions


@contextlib.contextmanager
def sandboxed(opts: Opts | None = None, active=(), tags_files=()):
    """Forge stand-ins + ``sam3ext.tagcomplete.paths.TAGS_DIR`` in a temporary folder; restores everything after.

    ``tags_files`` are copied from the extension's tags/ into the sandbox's tags/ first.
    """
    from sam3ext.tagcomplete import boot, paths

    root = Path(tempfile.mkdtemp(prefix="sam3_tac_"))
    stubs, forge, shared, callbacks, extensions = forge_stubs(root, opts, active)
    tags = root / "tags"
    tags.mkdir()
    for name in tags_files:
        shutil.copyfile(ROOT / "tags" / name, tags / name)
    package = sys.modules["sam3ext.tagcomplete"]
    saved_modules = {name: sys.modules.get(name) for name in (*stubs, *_SAM3_MODULES)}
    # `from sam3ext.tagcomplete import tag_frequency_db` takes the package attribute when there is one
    saved_attributes = {name: vars(package).get(name.rsplit(".", 1)[1]) for name in _SAM3_MODULES}
    saved_paths = (paths.TAGS_DIR, paths.TAGS_TEMP_DIR, paths.TAG_FREQUENCY_DB)
    saved_state = dict(boot.STATE)
    for name in _SAM3_MODULES:
        sys.modules.pop(name, None)
        vars(package).pop(name.rsplit(".", 1)[1], None)
    sys.modules.update(stubs)
    paths.TAGS_DIR, paths.TAGS_TEMP_DIR, paths.TAG_FREQUENCY_DB = tags, tags / "temp", tags / "tag_frequency.db"
    try:
        yield Sandbox(root=root, forge=forge, tags=tags, shared=shared, callbacks=callbacks, extensions=extensions)
    finally:
        paths.TAGS_DIR, paths.TAGS_TEMP_DIR, paths.TAG_FREQUENCY_DB = saved_paths
        boot.STATE.clear()
        boot.STATE.update(saved_state)
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for name, module in saved_attributes.items():
            attribute = name.rsplit(".", 1)[1]
            if module is None:
                vars(package).pop(attribute, None)
            else:
                setattr(package, attribute, module)
        shutil.rmtree(root, ignore_errors=True)


def run_quietly(function, *args, **kwargs):
    """Call ``function`` with stdout captured; returns ``(result, printed text)``."""
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()
