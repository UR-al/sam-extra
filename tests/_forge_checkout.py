"""Forge's own source files for the tests that execute them (not a test module).

Some origin tests run Forge's real code — ``modules/ui_loadsave.py``, ``modules/sd_schedulers.py``,
``backend/sampling/condition.py`` … — straight from the Forge checkout, found the way
``_extra_schedulers_support.forge_root()`` finds it: two levels above the extension (its normal place,
``extensions/<ext>``), else ``$SAM3_FORGE_ROOT``, else the development PC's install.

GitHub CI checks out only this extension, so there is no Forge to read: ``require_forge_file()`` then
raises ``unittest.SkipTest`` and the test (or ``setUpClass``) that needed the file is skipped with that
reason. With a Forge checkout nothing is skipped — a file missing from it still fails the test, since
that means Forge moved the code the test follows.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests._extra_schedulers_support import forge_root  # noqa: E402

SKIP_REASON = (
    "Forge checkout not found (CI checks out only this extension; run the tests from "
    "<forge>/extensions/<ext> or set $SAM3_FORGE_ROOT)"
)


def require_forge_file(relpath: str) -> Path:
    """``<forge>/<relpath>``; ``unittest.SkipTest`` when there is no Forge checkout at all."""
    root = forge_root()
    if root is None:
        raise unittest.SkipTest(f"{SKIP_REASON}: needs {relpath}")
    return root / relpath
