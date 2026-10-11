# Tag Autocomplete — vendored into sam-extra (forge_sam3_extension) as sam3ext/tagcomplete.
# Upstream: https://github.com/eduardoabreu81/sd-webui-tagcomplete-neo @ 8e0d81f (scripts/shared_paths.py), the Forge Neo fork
# (2026, Eduardo Abreu) of https://github.com/DominikDoom/a1111-sd-webui-tagcomplete.
# Copyright (c) 2022 Dominik Reh. MIT License — full text: sam3ext/tagcomplete/LICENSE (the fork's changes are
# released under the same licence). The rest of sam-extra is GPL-3.0-only; this file keeps the MIT terms.
# MODIFIED by sam-extra, 2026-10-11 (all other lines as upstream):
# - imports `modules.shared` only (upstream also imported `modules.scripts`, which only TAGS_PATH used);
# - TAGS_PATH is sam-extra's own tags/ folder (sam3ext/tagcomplete/paths.py TAGS_DIR) instead of
#   `scripts.basedir()/tags`: basedir() names the extension only while Forge runs a file from scripts/, and
#   this module lives outside scripts/ (tests import it without a Forge script load).

from pathlib import Path

from modules import shared
from modules.paths import extensions_dir, script_path

from sam3ext.tagcomplete import paths as _sam_paths

# Webui root path
FILE_DIR = Path(script_path).absolute()

# The extension base path
EXT_PATH = Path(extensions_dir).absolute()

# Tags base path
TAGS_PATH = Path(_sam_paths.TAGS_DIR).absolute()

# Embeddings directory (Forge Neo: models/embeddings/)
EMB_PATH = Path(shared.cmd_opts.embeddings_dir).absolute()

# Hypernetworks are removed in Forge Neo — always None
HYP_PATH = None

# LoRA and LyCORIS share one directory in Forge Neo
try:
    LORA_PATH = Path(shared.cmd_opts.lora_dir).absolute()
except (AttributeError, TypeError):
    LORA_PATH = None

# LyCORIS is unified with LoRA in Forge Neo
LYCO_PATH = LORA_PATH

# Wildcards directory (Forge Neo: scripts/wildcards/ under webui root)
WILDCARD_PATH = FILE_DIR.joinpath("scripts/wildcards").absolute()


def find_ext_wildcard_paths():
    """Returns paths to wildcard folders registered by other extensions."""
    found = list(EXT_PATH.glob("*/wildcards/"))

    # Append custom wildcard path from sd-dynamic-prompts if present
    try:
        from modules.shared import opts as _opts
        custom = getattr(_opts, "wildcard_dir", None)
    except ImportError:
        custom = None

    if custom is not None:
        p = Path(custom).absolute()
        if p.exists():
            found.append(p)

    return found


# The path to the extension wildcards folder
WILDCARD_EXT_PATHS = find_ext_wildcard_paths()

# Temporary file paths
STATIC_TEMP_PATH = FILE_DIR.joinpath("tmp").absolute()
TEMP_PATH = TAGS_PATH.joinpath("temp").absolute()

# Make sure these folders exist.
# Use parents=True + exist_ok=True so the extension keeps working after a Forge
# update or reinstall that might briefly remove the tags/temp subdirectory (#302).
TEMP_PATH.mkdir(parents=True, exist_ok=True)
STATIC_TEMP_PATH.mkdir(parents=True, exist_ok=True)
