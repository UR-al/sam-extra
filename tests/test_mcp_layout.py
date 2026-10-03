"""sam-extra MCP server: layout, packaging and licence notices.

- Forge puts every extension root on sys.path (modules/scripts.py load_scripts), so nothing at the
  extension root may be importable as ``mcp`` (it would shadow the SDK for every extension) and
  ``mcp_server`` must not be a package.
- The server runs from its own uv project: pyproject must name the package, the pinned
  dependencies and the console entry point.
- Every file vendored from or derived from forgeneo-mcp keeps the MIT copyright and permission
  notice and lists sam-extra's changes.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

from _mcp_support import MCP_PROJECT, ROOT

PACKAGE = MCP_PROJECT / "sam_extra_mcp"
VENDORED = PACKAGE / "forgeneo"
UPSTREAM_MODULES = (
    "__init__.py", "capabilities.py", "civitai.py", "client.py", "config.py", "dialects.py", "downloads.py",
    "fetcher.py", "generate.py", "history.py", "identity.py", "infotext.py", "loras.py", "modules.py",
    "presets.py", "profile.py",
)
DERIVED = (PACKAGE / "server.py", PACKAGE / "service.py", ROOT / "tests" / "test_mcp_forgeneo_upstream.py")
MIT_LINES = (
    "Copyright (c) 2026 Eduardo Abreu",
    "Permission is hereby granted, free of charge, to any person obtaining a copy",
    "The above copyright notice and this permission notice shall be included in all",
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR',
)


def _new_files() -> list[Path]:
    files = [path for path in MCP_PROJECT.rglob("*") if path.is_file() and "__pycache__" not in path.parts
             and ".venv" not in path.parts]
    files.append(ROOT / "scripts" / "mcp_settings.py")
    files.extend(sorted((ROOT / "tests").glob("test_mcp_*.py")))
    files.append(ROOT / "tests" / "_mcp_support.py")
    return files


class LayoutTests(unittest.TestCase):
    def test_nothing_at_the_extension_root_shadows_the_sdk(self):
        self.assertFalse((ROOT / "mcp").exists())
        self.assertFalse((ROOT / "mcp.py").exists())
        self.assertFalse((MCP_PROJECT / "__init__.py").exists(), "mcp_server must not be a package")
        self.assertFalse((MCP_PROJECT / "mcp").exists())

    def test_vendored_modules_are_all_there(self):
        self.assertEqual(sorted(path.name for path in VENDORED.glob("*.py")), sorted(UPSTREAM_MODULES))

    def test_every_vendored_or_derived_file_keeps_the_mit_notice(self):
        for path in [VENDORED / name for name in UPSTREAM_MODULES] + list(DERIVED):
            head = path.read_text(encoding="utf-8").split('"""', 1)[0]
            with self.subTest(file=path.name):
                for line in MIT_LINES:
                    self.assertIn(line, head)
                self.assertIn("a103dc5", head)
                self.assertIn("Changes in sam-extra", head)

    def test_files_are_utf8_without_bom(self):
        for path in _new_files():
            data = path.read_bytes()
            with self.subTest(file=str(path.relative_to(ROOT))):
                self.assertFalse(data.startswith(b"\xef\xbb\xbf"))
                data.decode("utf-8")

    def test_pyproject(self):
        project = tomllib.loads((MCP_PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(project["project"]["name"], "sam-extra-mcp")
        # The floor is the SDK release the server was verified against (2.2.0); a higher floor than
        # any published release would make the first `uv run` fail to resolve.
        self.assertEqual(project["project"]["dependencies"], ["mcp>=2.2,<3", "httpx>=0.27"])
        self.assertEqual(project["project"]["scripts"], {"sam-extra-mcp": "sam_extra_mcp.server:main"})
        self.assertEqual(project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"], ["sam_extra_mcp"])
        self.assertIn("MIT", project["project"]["license"])
        self.assertIn("GPL-3.0-only", project["project"]["license"])
        from sam_extra_mcp import __version__

        self.assertEqual(project["project"]["version"], __version__)

    def test_entry_point_exists(self):
        tree = ast.parse((PACKAGE / "server.py").read_text(encoding="utf-8"))
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertIn("main", functions)
        self.assertTrue((PACKAGE / "__main__.py").is_file())

    def test_only_server_imports_the_sdk(self):
        for path in PACKAGE.rglob("*.py"):
            if path.name == "server.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                for name in names:
                    with self.subTest(file=path.name, module=name):
                        self.assertFalse(name == "mcp" or name.startswith("mcp."))

    def test_core_imports_without_the_sdk_or_httpx(self):
        """In a clean interpreter where neither the MCP SDK nor httpx can be imported."""
        code = (
            "import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('mcp', 'httpx'):\n"
            "            raise ImportError('blocked: ' + name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            f"sys.path.insert(0, {str(MCP_PROJECT)!r})\n"
            "import sam_extra_mcp.policy, sam_extra_mcp.forge_paths, sam_extra_mcp.tool_hints\n"
            "import sam_extra_mcp.forgeneo.profile, sam_extra_mcp.forgeneo.history, sam_extra_mcp.forgeneo.dialects\n"
            "import sam_extra_mcp.forgeneo.loras, sam_extra_mcp.forgeneo.generate, sam_extra_mcp.forgeneo.capabilities\n"
            "import sam_extra_mcp.forgeneo.client, sam_extra_mcp.forgeneo.fetcher, sam_extra_mcp.service\n"
            "assert 'mcp' not in sys.modules and 'httpx' not in sys.modules\n"
            "try:\n"
            "    import sam_extra_mcp.server\n"
            "except ImportError as exc:\n"
            "    assert 'mcp' in str(exc), exc\n"
            "else:\n"
            "    raise SystemExit('server imported without the SDK')\n"
            "print('ok')\n"
        )
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                                encoding="utf-8", env=env, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_cmd_flags_appears_only_in_the_deny_list(self):
        for path in PACKAGE.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = node.body
                    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                        docstrings.add(id(body[0].value))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and "cmd-flags" in node.value:
                    if id(node) in docstrings:
                        continue
                    with self.subTest(file=path.name, line=node.lineno):
                        self.assertEqual(path.name, "client.py")
                        self.assertEqual(node.value, "/sdapi/v1/cmd-flags")
        from sam_extra_mcp.forgeneo.client import ALLOWED_ROUTES, NEVER_REQUESTED

        self.assertIn("/sdapi/v1/cmd-flags", NEVER_REQUESTED)
        self.assertFalse(any("cmd-flags" in route for _method, route in ALLOWED_ROUTES))

    def test_ignored_runtime_files(self):
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("mcp_server/uv.lock", gitignore)
        self.assertIn(".venv/", gitignore)


if __name__ == "__main__":
    unittest.main()
