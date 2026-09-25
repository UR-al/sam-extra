"""첫 설치 경로(감사 M25·M27·M28·M29·M26·L43) — install.py·preload.py·requirements·metadata.ini·README·CI.

install.py 는 Forge 가 ``python install.py`` 로 따로 실행한다. 여기서는 다른 이름으로 불러와 모듈 수준 부작용
(clone·pip)이 없는지 보고, 함수만 가짜 설치기·가짜 git 으로 돌린다. 네트워크·GPU 는 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import ast
import configparser
import contextlib
import importlib.util
import io
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CANONICAL_FOLDER = "forge_sam3_extension"


def _load_install():
    """install.py 를 부작용 없이 불러온다 — 불러오는 동안 clone·pip 이 한 번이라도 불리면 실패."""
    def _forbidden(*args, **kwargs):
        raise AssertionError(f"install.py 가 import 중에 외부 명령을 실행했다: {args!r}")

    spec = importlib.util.spec_from_file_location("_t_sam3_install", ROOT / "install.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with mock.patch.object(subprocess, "check_call", _forbidden), \
            mock.patch.object(subprocess, "run", _forbidden), \
            mock.patch.object(subprocess, "call", _forbidden):
        spec.loader.exec_module(module)
    return module


install = _load_install()


def _capture(fn, *args, **kwargs):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        result = fn(*args, **kwargs)
    return result, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# requirements.txt — torch 없음, install.py 가 이 파일을 그대로 읽는다(M25)
# ---------------------------------------------------------------------------


class RequirementsFileTests(unittest.TestCase):
    def test_requirements_never_list_the_cuda_torch_stack(self):
        names = {install._canonical(req.name) for req in install.read_requirements()}
        self.assertTrue(names)
        for forbidden in ("torch", "torchvision", "torchaudio", "xformers"):
            self.assertNotIn(forbidden, names, "pip install -r 이 Forge 의 CUDA torch 를 바꿔 끼우면 안 된다")

    def test_requirements_cover_what_sam3_needs(self):
        names = {install._canonical(req.name) for req in install.read_requirements()}
        for needed in ("sam3", "safetensors", "opencv-python", "timm", "einops", "huggingface-hub", "iopath"):
            self.assertIn(needed, names)

    def test_the_tile_repair_vendor_pure_python_dep_is_auto_installed(self):
        """Anima vendor(kohya sd-scripts)가 불러올 때 import 하는 imagesize — 순수 파이썬이라 requirements 로
        자동 설치한다. 판은 벤더 requirements.txt 와 같은 1.4.1(2.x 는 EXIF 회전을 반영한 크기를 준다)."""
        reqs = {install._canonical(req.name): req for req in install.read_requirements()}
        self.assertIn("imagesize", reqs)
        self.assertEqual(str(reqs["imagesize"].specifier), "==1.4.1")
        self.assertIn("imagesize", install._ANIMA_DEPS, "자동 설치를 끈 실행에서도 Tile-Repair 안내에 남는다")

    def test_every_line_parses(self):
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                Requirement(line)  # InvalidRequirement 이면 실패

    def test_parser_skips_comments_options_and_foreign_markers(self):
        path = Path(self._tmp()) / "requirements.txt"
        path.write_text(
            "# 주석\n\nsam3\n-r other.txt\n--index-url https://example.invalid\n"
            "Pillow>=11.1.0  # 끝 주석\nfoo; sys_platform == 'nonexistent-os'\nnot a valid ! line\n",
            encoding="utf-8",
        )
        reqs, out, _ = _capture(install.read_requirements, path)
        self.assertEqual([str(r) for r in reqs], ["sam3", "Pillow>=11.1.0"])
        self.assertIn("not a valid ! line", out, "읽지 못한 줄은 stdout 에 알린다")

    def test_missing_file_is_empty_not_a_crash(self):
        reqs, out, _ = _capture(install.read_requirements, Path(self._tmp()) / "nope.txt")
        self.assertEqual(reqs, [])
        self.assertIn("requirements.txt", out)

    def _tmp(self):
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return tmp.name


# ---------------------------------------------------------------------------
# check_environment — 빠진 것만 설치, torch 는 절대 설치하지 않음, 버전 불일치는 알리기만, 진단은 stdout
# ---------------------------------------------------------------------------


class CheckEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.installed: list[str] = []

    def _installer(self, spec: str) -> None:
        self.installed.append(spec)

    def _status(self, table):
        return lambda req: table.get(install._canonical(req.name), "ok")

    def _run(self, reqs, table, **kwargs):
        with mock.patch.object(install, "requirement_status", self._status(table)):
            return _capture(install.check_environment, [Requirement(r) for r in reqs],
                            installer=self._installer, **kwargs)

    def test_only_missing_requirements_are_installed_with_their_specifier(self):
        result, out, err = self._run(
            ["sam3", "timm", "Pillow>=11.1.0", "gradio<6.0,>=4.0"],
            {"sam3": "missing", "pillow": "missing"},
            auto_install=True,
        )
        self.assertEqual(self.installed, ["sam3", "Pillow>=11.1.0"])
        self.assertEqual(result["installed"], ["sam3", "Pillow>=11.1.0"])
        self.assertEqual(err, "", "진단은 stdout 으로(Forge 는 install.py 의 stdout 만 보여 준다)")
        self.assertIn("sam3", out)

    def test_nothing_missing_is_silent(self):
        result, out, err = self._run(["sam3", "timm"], {}, auto_install=True)
        self.assertEqual(self.installed, [])
        self.assertEqual((out, err), ("", ""))
        self.assertEqual(result["missing"], [])

    def test_torch_is_never_auto_installed_even_if_listed(self):
        result, out, _ = self._run(["torch", "torchvision", "sam3"], {"torch": "missing", "torchvision": "missing",
                                                                      "sam3": "missing"}, auto_install=True)
        self.assertEqual(self.installed, ["sam3"])
        self.assertIn("torch", out)
        self.assertIn("CUDA", out)
        self.assertEqual(sorted(result["manual"]), ["torch", "torchvision"])

    def test_version_mismatch_is_reported_but_never_changed(self):
        result, out, _ = self._run(["Pillow>=11.1.0"], {"pillow": "mismatch"}, auto_install=True)
        self.assertEqual(self.installed, [], "Forge 가 깐 패키지를 올리거나 내리지 않는다")
        self.assertEqual(result["mismatch"], ["Pillow>=11.1.0"])
        self.assertIn("Pillow>=11.1.0", out)

    def test_auto_install_off_only_prints_the_pip_command(self):
        result, out, _ = self._run(["sam3", "iopath"], {"sam3": "missing", "iopath": "missing"}, auto_install=False)
        self.assertEqual(self.installed, [])
        self.assertIn('pip install "sam3" "iopath"', out)
        self.assertIn(install.AUTO_INSTALL_FLAG, out, "어떻게 켜고 끄는지 알려 준다")
        self.assertEqual(result["installed"], [])

    def test_failed_install_is_reported_and_the_rest_continue(self):
        def installer(spec):
            self.installed.append(spec)
            if spec == "sam3":
                raise RuntimeError("pip exploded")

        with mock.patch.object(install, "requirement_status", self._status({"sam3": "missing", "iopath": "missing"})):
            result, out, err = _capture(install.check_environment, [Requirement("sam3"), Requirement("iopath")],
                                        installer=installer, auto_install=True)
        self.assertEqual(self.installed, ["sam3", "iopath"])
        self.assertEqual(result["failed"], ["sam3"])
        self.assertEqual(result["installed"], ["iopath"])
        self.assertIn("pip exploded", out)
        self.assertIn('pip install "sam3"', out)
        self.assertEqual(err, "")

    def test_auto_install_defaults_to_the_toggle(self):
        with mock.patch.object(install, "auto_install_enabled", lambda: False):
            self._run(["sam3"], {"sam3": "missing"})
        self.assertEqual(self.installed, [])
        with mock.patch.object(install, "auto_install_enabled", lambda: True):
            self._run(["sam3"], {"sam3": "missing"})
        self.assertEqual(self.installed, ["sam3"])


class RequirementStatusTests(unittest.TestCase):
    def test_installed_distribution_in_range_is_ok(self):
        self.assertEqual(install.requirement_status(Requirement("packaging")), "ok")

    def test_installed_distribution_out_of_range_is_mismatch(self):
        self.assertEqual(install.requirement_status(Requirement("packaging<0.1")), "mismatch")

    def test_absent_distribution_is_missing(self):
        self.assertEqual(install.requirement_status(Requirement("sam3-ext-surely-not-installed-pkg")), "missing")

    def test_import_name_fallback_counts_as_installed(self):
        # 배포 이름이 다른 빌드(예: opencv-python 대신 opencv-contrib/headless)는 import 이름으로 확인한다.
        with mock.patch.dict(install.import_name, {"sam3-ext-fake-dist": "json"}):
            self.assertEqual(install.requirement_status(Requirement("sam3-ext-fake-dist>=1")), "ok")


class AutoInstallToggleTests(unittest.TestCase):
    def test_default_is_on(self):
        self.assertTrue(install.auto_install_enabled(argv=[], env={}))

    def test_env_var_turns_it_off(self):
        for value in ("1", "true", "YES", "on"):
            self.assertFalse(install.auto_install_enabled(argv=[], env={install.AUTO_INSTALL_ENV: value}), value)
        self.assertTrue(install.auto_install_enabled(argv=[], env={install.AUTO_INSTALL_ENV: "0"}))

    def test_commandline_args_flag_turns_it_off(self):
        env = {"COMMANDLINE_ARGS": f"--api --theme dark {install.AUTO_INSTALL_FLAG} --flash"}
        self.assertFalse(install.auto_install_enabled(argv=[], env=env))
        self.assertFalse(install.auto_install_enabled(argv=["launch.py", install.AUTO_INSTALL_FLAG], env={}))

    def test_unbalanced_quotes_in_commandline_args_do_not_crash(self):
        env = {"COMMANDLINE_ARGS": f'--ckpt-dir "C:\\models {install.AUTO_INSTALL_FLAG}'}
        self.assertFalse(install.auto_install_enabled(argv=[], env=env))


class DefaultInstallerTests(unittest.TestCase):
    def test_uses_forge_launch_run_pip_with_a_quoted_specifier(self):
        calls = []
        fake_launch = types.ModuleType("launch")
        fake_launch.run_pip = lambda command, desc=None, **kw: calls.append((command, desc))
        with mock.patch.dict(sys.modules, {"launch": fake_launch}):
            install.pip_install("gradio<6.0,>=4.0")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 'install "gradio<6.0,>=4.0"', "cmd 에서 < > 가 리다이렉트로 먹히지 않게 따옴표")

    def test_falls_back_to_pip_subprocess_outside_forge(self):
        calls = []
        with mock.patch.dict(sys.modules, {"launch": None}), \
                mock.patch.object(install.subprocess, "check_call", lambda cmd, **kw: calls.append(cmd)):
            install.pip_install("sam3")
        self.assertEqual(calls, [[sys.executable, "-m", "pip", "install", "sam3"]])


class StdoutDiagnosticsTests(unittest.TestCase):
    def test_install_py_never_prints_to_stderr(self):
        # Forge 는 install.py 의 stdout 만 콘솔에 옮긴다(launch_utils.run 이 PIPE 로 받음) — stderr 는 사라진다(M27).
        source = (ROOT / "install.py").read_text(encoding="utf-8")
        self.assertNotIn("sys.stderr", source)

    def test_main_runs_every_step_once(self):
        order = []
        with mock.patch.object(install, "check_environment", lambda: order.append("env")), \
                mock.patch.object(install, "ensure_anima_vendor", lambda: order.append("anima") or True), \
                mock.patch.object(install, "check_anima_environment", lambda: order.append("anima_env")), \
                mock.patch.object(install, "ensure_lora_manager_vendor", lambda: order.append("lm") or True), \
                mock.patch.object(install, "check_lora_manager_version", lambda: order.append("lm_ver")), \
                mock.patch.object(install, "check_lora_manager_environment", lambda: order.append("lm_env")):
            install.main()
        self.assertEqual(order, ["env", "anima", "anima_env", "lm", "lm_ver", "lm_env"])

    def test_script_entrypoint_calls_main(self):
        tree = ast.parse((ROOT / "install.py").read_text(encoding="utf-8"))
        guards = [node for node in tree.body if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)]
        self.assertEqual(len(guards), 1)
        self.assertIn("main()", ast.unparse(guards[0]))


# ---------------------------------------------------------------------------
# vendor 핀(M29)
# ---------------------------------------------------------------------------


class VendorPinTests(unittest.TestCase):
    def test_lora_manager_pin_is_the_verified_commit(self):
        self.assertRegex(install._LM_PIN_COMMIT or "", r"^[0-9a-f]{40}$")
        self.assertEqual(install._LM_EXPECTED_VERSION, "1.2.0")

    def test_checkout_pinned_commit_fetches_and_detaches(self):
        calls = []

        def fake_check_call(cmd, **kwargs):
            calls.append(cmd)

        with mock.patch.object(install.subprocess, "check_call", fake_check_call), \
                mock.patch.object(install, "_git_head", lambda root: "0" * 40):
            ok, out, err = _capture(install._checkout_pinned_commit, Path("X"), "a" * 40, "LoRA Manager vendor")
        self.assertTrue(ok)
        self.assertEqual(calls[0][:3], ["git", "-C", "X"])
        self.assertIn("fetch", calls[0])
        self.assertIn("a" * 40, calls[0])
        self.assertIn("checkout", calls[1])
        self.assertIn("a" * 40, calls[1])
        self.assertEqual(err, "")

    def test_already_on_the_pin_does_nothing(self):
        with mock.patch.object(install.subprocess, "check_call", lambda *a, **k: self.fail("git 을 부르면 안 된다")), \
                mock.patch.object(install, "_git_head", lambda root: "b" * 40):
            ok, out, _ = _capture(install._checkout_pinned_commit, Path("X"), "b" * 40, "vendor")
        self.assertTrue(ok)
        self.assertEqual(out, "")

    def test_failed_pin_keeps_the_branch_tip_and_warns_on_stdout(self):
        def boom(cmd, **kwargs):
            raise subprocess.CalledProcessError(128, cmd)

        with mock.patch.object(install.subprocess, "check_call", boom), \
                mock.patch.object(install, "_git_head", lambda root: "0" * 40):
            ok, out, err = _capture(install._checkout_pinned_commit, Path("X"), "c" * 40, "LoRA Manager vendor")
        self.assertFalse(ok)
        self.assertIn("c" * 12, out)
        self.assertEqual(err, "")

    def test_lora_manager_version_check_warns_only_on_drift(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "standalone.py").write_text("", encoding="utf-8")
            (root / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "1.2.0"\n', encoding="utf-8")
            with mock.patch.object(install, "_LM_ROOT", root), mock.patch.object(install, "_LM_SENTINEL",
                                                                                 root / "standalone.py"):
                _, out, _ = _capture(install.check_lora_manager_version)
                self.assertEqual(out, "")
                (root / "pyproject.toml").write_text('[project]\nversion = "1.3.0"\n', encoding="utf-8")
                _, out, _ = _capture(install.check_lora_manager_version)
                self.assertIn("1.3.0", out)
                self.assertIn("1.2.0", out)


# ---------------------------------------------------------------------------
# preload.py — Forge 인자(토글)
# ---------------------------------------------------------------------------


class PreloadTests(unittest.TestCase):
    def _parser(self):
        spec = importlib.util.spec_from_file_location("_t_sam3_preload", ROOT / "preload.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parser = argparse.ArgumentParser()
        module.preload(parser)
        return parser

    def test_flags_default_off(self):
        args = self._parser().parse_args([])
        self.assertFalse(args.sam3_no_huggingface)
        self.assertFalse(args.sam3_no_auto_install)

    def test_auto_install_flag_is_accepted(self):
        # Forge 본 파서는 모르는 인자를 거부한다 — install.py 가 읽는 토글도 여기 등록돼 있어야 한다.
        args = self._parser().parse_args([install.AUTO_INSTALL_FLAG, "--sam3-no-huggingface"])
        self.assertTrue(args.sam3_no_auto_install)
        self.assertTrue(args.sam3_no_huggingface)


# ---------------------------------------------------------------------------
# 폴더명(M28) — metadata.ini 콜백 순서는 폴더명에 묶인다
# ---------------------------------------------------------------------------


def _folder_warning_fn():
    """scripts/!sam3.py 의 폴더명 경고 함수만 꺼내 온다(스크립트 전체를 Forge 없이 불러오지 않으려고)."""
    tree = ast.parse((ROOT / "scripts" / "!sam3.py").read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))
             and (getattr(n, "name", None) == "_extension_folder_warning"
                  or any(getattr(t, "id", None) == "SAM3_EXTENSION_FOLDER" for t in getattr(n, "targets", [])))]
    namespace: dict = {"Path": Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "!sam3.py", "exec"), namespace)
    return namespace, tree


class FolderNameTests(unittest.TestCase):
    def test_metadata_sections_reference_the_canonical_folder(self):
        config = configparser.ConfigParser()
        config.read(ROOT / "metadata.ini", encoding="utf-8")
        sections = [s for s in config.sections() if s.startswith("callbacks/")]
        self.assertTrue(sections)
        for section in sections:
            self.assertTrue(section[len("callbacks/"):].startswith(CANONICAL_FOLDER + "/"), section)

    def test_readme_clone_command_names_the_folder(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        clones = [line for line in readme.splitlines() if line.strip().startswith("git clone") and "sam-extra" in line]
        self.assertTrue(clones)
        for line in clones:
            self.assertTrue(line.rstrip().endswith(" " + CANONICAL_FOLDER), line)

    def test_script_warns_when_the_folder_was_renamed(self):
        namespace, _ = _folder_warning_fn()
        self.assertEqual(namespace["SAM3_EXTENSION_FOLDER"], CANONICAL_FOLDER)
        warn = namespace["_extension_folder_warning"]
        self.assertIsNone(warn(Path("C:/forge/extensions/forge_sam3_extension/scripts/!sam3.py")))
        self.assertIsNone(warn(Path("/x/extensions/Forge_SAM3_Extension/scripts/!sam3.py")), "Forge 는 소문자로 비교")
        message = warn(Path("/x/extensions/sam-extra/scripts/!sam3.py"))
        self.assertIn("sam-extra", message)
        self.assertIn(CANONICAL_FOLDER, message)
        self.assertIn("metadata.ini", message)

    def test_script_calls_the_warning_at_load(self):
        _, tree = _folder_warning_fn()
        calls = [n for n in tree.body if "_extension_folder_warning(" in ast.unparse(n)
                 and not isinstance(n, ast.FunctionDef)]
        self.assertTrue(calls, "시작 로그에 남도록 모듈 수준에서 부른다")

    def test_this_checkout_uses_the_canonical_folder(self):
        namespace, _ = _folder_warning_fn()
        if ROOT.name.lower() != CANONICAL_FOLDER:
            self.skipTest(f"이 작업 트리는 {ROOT.name} 폴더(CI 등)")
        self.assertIsNone(namespace["_extension_folder_warning"](ROOT / "scripts" / "!sam3.py"))


# ---------------------------------------------------------------------------
# 개발 의존성·CI(M26·L43)
# ---------------------------------------------------------------------------


class DevRequirementsTests(unittest.TestCase):
    def _dev_names(self):
        text = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
        names = set()
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if line and not line.startswith("-"):
                names.add(install._canonical(Requirement(line).name))
        return names, text

    def test_dev_requirements_cover_the_test_suite_imports(self):
        names, _ = self._dev_names()
        for needed in ("pytest", "safetensors", "transformers", "gradio", "pydantic", "numpy", "pillow",
                       "fastapi", "packaging", "opencv-python-headless"):
            self.assertIn(needed, names)

    def test_dev_requirements_leave_torch_to_the_cpu_index(self):
        names, _ = self._dev_names()
        self.assertNotIn("torch", names, "CI 는 CPU torch 를 따로 받는다(기본 PyPI torch 는 CUDA 휠 2 GB+)")

    def test_ci_uses_python_313_and_the_dev_requirements(self):
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn('python-version: "3.13"', ci)
        self.assertIn("requirements-dev.txt", ci)
        self.assertIn("download.pytorch.org/whl/cpu", ci)
        self.assertIn('unittest discover -s tests -p "test_*.py"', ci)


if __name__ == "__main__":
    unittest.main()
