"""Where the Forge installation that ships this server keeps its settings, outputs and models.

The server runs from ``<data>/extensions/<sam-extra>/mcp_server``. Forge keeps every extension
in ``<data>/extensions`` (``modules/paths_internal.py``: ``extensions_dir = data_path/extensions``,
with or without ``--data-dir``), so the extension's own location names Forge's data folder and,
through it, the defaults of everything else:

- the settings file Forge reads and writes: ``<data>/config.json`` (``--ui-settings-file``
  default, ``modules/cmd_args.py``) - where Settings -> SAM Extra MCP stores the permissions;
- the models folder: ``<data>/models`` (``--model-ref`` moves it);
- the output folders: Forge stores them relative to its working folder
  (``modules/util.truncate_path``), which is the program folder holding ``webui.py`` when Forge
  is started from its own batch file. Without ``--data-dir`` that is the data folder itself.

Same-PC use therefore needs no ``FORGE_PATH_MAP``. For setups the location cannot describe,
environment variables override each piece:

``SAM_EXTRA_MCP_FORGE_CONFIG``   the settings file (Forge started with ``--ui-settings-file``)
``SAM_EXTRA_MCP_FORGE_ROOT``     the program folder (``--data-dir`` setups, relative outdirs)
``SAM_EXTRA_MCP_MODELS_DIR``     the models folder (``--model-ref``)
``SAM_EXTRA_MCP_FALLBACK_DIR``   where images the server cannot find on disk are saved
``SAM_EXTRA_MCP_CACHE_DIR``      where confirmed prompt dialects are remembered
                                 (``FORGENEO_CACHE_DIR`` is read too, as upstream)

Nothing here talks to Forge; this module only looks at the file system.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

ENV_SETTINGS_FILE = "SAM_EXTRA_MCP_FORGE_CONFIG"
ENV_FORGE_ROOT = "SAM_EXTRA_MCP_FORGE_ROOT"
ENV_MODELS_DIR = "SAM_EXTRA_MCP_MODELS_DIR"
ENV_FALLBACK_DIR = "SAM_EXTRA_MCP_FALLBACK_DIR"
ENV_CACHE_DIR = "SAM_EXTRA_MCP_CACHE_DIR"
ENV_UPSTREAM_CACHE_DIR = "FORGENEO_CACHE_DIR"

PROJECT_FOLDER = "mcp_server"
PACKAGE_FOLDER = "sam_extra_mcp"
SETTINGS_FILENAME = "config.json"
# Per-installation data of sam-extra lives here (the Notebook uses <data>/sam-extra too).
SAM_EXTRA_DATA = os.path.join("sam-extra", "mcp")
HOME_FALLBACK = ".sam-extra-mcp"


@dataclass(frozen=True)
class ForgePaths:
    extension_root: str | None
    data_dir: str | None
    forge_root: str | None
    settings_file: str | None
    settings_source: str
    models_dir: str | None
    fallback_dir: str
    cache_dir: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def settings_explicit(self) -> bool:
        """The settings file was named by the operator (SAM_EXTRA_MCP_FORGE_CONFIG), not derived."""
        return self.settings_source.startswith("environment")

    def output_candidates(self, outdir: str | None) -> tuple[str, ...]:
        """Absolute folders an outdir option may mean, most likely first."""
        value = (outdir or "").strip()
        if not value:
            return ()
        if _is_absolute(value):
            return (os.path.normpath(value),)
        parts = value.replace("\\", "/").split("/")
        found: list[str] = []
        for root in (self.forge_root, self.data_dir):
            if root:
                candidate = os.path.normpath(os.path.join(root, *parts))
                if candidate not in found:
                    found.append(candidate)
        return tuple(found)

    def resolve_output_dir(self, outdir: str | None) -> str | None:
        """The folder an outdir option points at, if it exists or Forge will create it.

        Forge creates output folders on first use (os.makedirs in images.save_image),
        so a folder that does not exist yet still counts when its parent exists (the
        img2img folder of an install that only ever ran txt2img) or - for a relative
        outdir - when the Forge folder it is relative to exists (a fresh install,
        where even "output" appears only with the first image).
        """
        candidates = self.output_candidates(outdir)
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        for candidate in candidates:
            parent = os.path.dirname(candidate.rstrip("/\\"))
            if parent and os.path.isdir(parent):
                return candidate
        if candidates and not _is_absolute((outdir or "").strip()):
            for root, candidate in zip((root for root in (self.forge_root, self.data_dir) if root), candidates):
                if os.path.isdir(root):
                    return candidate
        return None

    def as_dict(self) -> dict:
        return {
            "extension_root": self.extension_root,
            "data_dir": self.data_dir,
            "forge_root": self.forge_root,
            "settings_file": self.settings_file,
            "settings_source": self.settings_source,
            "models_dir": self.models_dir,
            "fallback_dir": self.fallback_dir,
            "cache_dir": self.cache_dir,
            "notes": list(self.notes),
        }


def _is_absolute(path: str) -> bool:
    normalised = path.replace("\\", "/")
    return os.path.isabs(path) or normalised.startswith("/") or (len(normalised) > 1 and normalised[1] == ":")


def extension_root_for(module_file: str) -> str | None:
    """``<ext>`` for ``<ext>/mcp_server/sam_extra_mcp/<module>.py``, else None.

    The unresolved path is tried first: an extension folder that is a junction
    or symlink inside Forge's extensions folder still names that Forge.
    """
    for candidate in (os.path.abspath(module_file), os.path.realpath(module_file)):
        package = os.path.dirname(candidate)
        project = os.path.dirname(package)
        if os.path.basename(package) != PACKAGE_FOLDER or os.path.basename(project) != PROJECT_FOLDER:
            continue
        if os.path.isfile(os.path.join(project, "pyproject.toml")):
            return os.path.dirname(project)
    return None


def data_dir_for(extension_root: str | None) -> str | None:
    """Forge's data folder: the parent of the ``extensions`` folder holding the extension."""
    if not extension_root:
        return None
    parent = os.path.dirname(os.path.normpath(extension_root))
    if os.path.basename(parent).lower() != "extensions":
        return None
    return os.path.dirname(parent)


def looks_like_forge_program(path: str | None) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, "webui.py")) and os.path.isdir(os.path.join(path, "modules"))


def discover(env: Mapping[str, str] | None = None, module_file: str | None = None) -> ForgePaths:
    env = os.environ if env is None else env
    notes: list[str] = []
    extension_root = extension_root_for(module_file or __file__)
    data_dir = data_dir_for(extension_root)
    if extension_root is None:
        notes.append("this server is not running from <extension>/mcp_server, so the Forge installation is unknown")
    elif data_dir is None:
        notes.append(
            f"the extension at {extension_root} is not inside a Forge 'extensions' folder, so the Forge "
            f"installation is unknown; set {ENV_SETTINGS_FILE} (and {ENV_FORGE_ROOT}) to point at it"
        )

    forge_root = (env.get(ENV_FORGE_ROOT) or "").strip() or None
    if forge_root and not os.path.isdir(forge_root):
        notes.append(f"{ENV_FORGE_ROOT} does not name a folder: {forge_root}")
        forge_root = None
    if forge_root is None and looks_like_forge_program(data_dir):
        forge_root = data_dir
    elif forge_root is None and data_dir:
        notes.append(
            f"{data_dir} holds no webui.py (Forge started with --data-dir?): relative output folders "
            f"are tried against it; set {ENV_FORGE_ROOT} to the program folder if they are not found"
        )

    explicit = (env.get(ENV_SETTINGS_FILE) or "").strip()
    if explicit:
        settings_file: str | None = os.path.abspath(explicit)
        settings_source = f"environment ({ENV_SETTINGS_FILE})"
    elif data_dir:
        settings_file = os.path.join(data_dir, SETTINGS_FILENAME)
        settings_source = "extension location (<Forge data>/config.json)"
    else:
        settings_file = None
        settings_source = "unknown"

    models_dir = (env.get(ENV_MODELS_DIR) or "").strip() or None
    if models_dir is None and data_dir:
        models_dir = os.path.join(data_dir, "models")

    own_data = os.path.join(data_dir, SAM_EXTRA_DATA) if data_dir else os.path.join(os.path.expanduser("~"), HOME_FALLBACK)
    fallback_dir = (env.get(ENV_FALLBACK_DIR) or "").strip() or os.path.join(own_data, "outputs")
    cache_dir = (
        (env.get(ENV_CACHE_DIR) or "").strip()
        or (env.get(ENV_UPSTREAM_CACHE_DIR) or "").strip()
        or own_data
    )

    return ForgePaths(
        extension_root=extension_root,
        data_dir=data_dir,
        forge_root=forge_root,
        settings_file=settings_file,
        settings_source=settings_source,
        models_dir=models_dir,
        fallback_dir=os.path.abspath(fallback_dir),
        cache_dir=os.path.abspath(cache_dir),
        notes=tuple(notes),
    )
