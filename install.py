"""Forge 가 확장을 불러올 때 ``python install.py`` 로 따로 실행하는 설치 스크립트.

Forge(modules/launch_utils.py 의 run_extension_installer)는 이 스크립트의 stdout 만 콘솔에 옮긴다 — stderr 는
PIPE 로 받아 버리므로 진단은 전부 stdout(``_log``)으로 낸다(감사 M27). import 만 하면 아무 일도 하지 않고,
``python install.py`` 로 실행될 때만 ``main()`` 이 돈다(테스트가 함수만 불러 쓰게).
"""
from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import parse


_EXT_ROOT = Path(__file__).resolve().parent
_REQUIREMENTS = _EXT_ROOT / "requirements.txt"

# pip 이름 → import 이름(배포 이름으로 못 찾을 때 find_spec 으로 한 번 더 본다).
import_name = {
    "opencv-python": "cv2",
    "py-cpuinfo": "cpuinfo",
    "protobuf": "google.protobuf",
    "Pillow": "PIL",
}

# requirements.txt 에 있더라도 절대 자동 설치하지 않는다 — Forge 가 CUDA 빌드에 맞춰 깔아 둔 것을 PyPI 휠로
# 바꿔 끼우면 Forge 가 깨진다(감사 M25).
_NEVER_AUTO_INSTALL = frozenset({"torch", "torchvision", "torchaudio", "xformers"})

# 자동 설치 토글. 기본은 켜짐(빠진 패키지만 설치). 끄면 빠진 것을 알리고 pip 명령만 보여 준다.
# Forge 인자 --sam3-no-auto-install(preload.py 에 등록 — webui-user.bat 의 COMMANDLINE_ARGS 에 넣는다) 또는
# 환경 변수 SAM3_NO_AUTO_INSTALL=1. install.py 는 Forge 가 인자 없이 따로 실행하므로 COMMANDLINE_ARGS 도 읽는다.
# LoRA Manager 의 경량 deps 자동 설치도 같은 토글을 따른다.
AUTO_INSTALL_FLAG = "--sam3-no-auto-install"
AUTO_INSTALL_ENV = "SAM3_NO_AUTO_INSTALL"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _log(message: str) -> None:
    print(f"[forge_sam3_extension] {message}", flush=True)


def _canonical(name: str) -> str:
    return canonicalize_name(name)


def is_installed(package: str, min_version: str | None = None, max_version: str | None = None):
    name = import_name.get(package, package)
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return False

    if spec is None:
        return False

    if not min_version and not max_version:
        return True

    if not min_version:
        min_version = "0.0.0"
    if not max_version:
        max_version = "99999999.99999999.99999999"

    try:
        pkg_version = version(package)
        return parse(min_version) <= parse(pkg_version) <= parse(max_version)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# requirements.txt — 단일 출처. 빠진 것만 설치하고, 버전이 범위 밖인 것은 알리기만 한다(감사 M25).
# ---------------------------------------------------------------------------


def read_requirements(path: Path = _REQUIREMENTS) -> list[Requirement]:
    """``requirements.txt`` 를 읽어 이 환경에 해당하는 요구 사항만 돌려준다.

    주석·빈 줄·pip 옵션(``-r``, ``--index-url`` …)은 건너뛰고, 마커가 이 환경과 맞지 않는 줄도 뺀다.
    읽지 못한 줄은 stdout 에 알리고 건너뛴다.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        _log(f"could not read requirements.txt ({path}): {exc}")
        return []
    result: list[Requirement] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            req = Requirement(line)
        except InvalidRequirement as exc:
            _log(f"skipping unreadable requirements.txt line {raw.strip()!r}: {exc}")
            continue
        if req.marker is not None and not req.marker.evaluate():
            continue
        result.append(req)
    return result


def requirement_status(req: Requirement) -> str:
    """'ok' · 'missing' · 'mismatch'(깔려 있지만 버전이 범위 밖).

    배포 이름으로 찾지 못하면 import 이름으로 한 번 더 본다 — 다른 배포(opencv-contrib/headless 등)나
    소스 설치도 '있음' 으로 친다(버전은 모르므로 건드리지 않는다).
    """
    try:
        installed = version(req.name)
    except PackageNotFoundError:
        module = import_name.get(req.name) or req.name.replace("-", "_")
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        return "ok" if found else "missing"
    if req.specifier and not req.specifier.contains(installed, prereleases=True):
        return "mismatch"
    return "ok"


def auto_install_enabled(argv: list[str] | None = None, env=None) -> bool:
    """자동 설치 토글 — ``SAM3_NO_AUTO_INSTALL`` 가 참이거나 인자에 ``--sam3-no-auto-install`` 이 있으면 끔."""
    env = os.environ if env is None else env
    if str(env.get(AUTO_INSTALL_ENV, "")).strip().lower() in _TRUTHY:
        return False
    args = list(sys.argv[1:] if argv is None else argv)
    commandline = env.get("COMMANDLINE_ARGS", "") or ""
    try:
        args += shlex.split(commandline)
    except ValueError:  # 따옴표가 짝이 안 맞는 COMMANDLINE_ARGS — 공백으로만 나눠 본다
        args += commandline.split()
    return AUTO_INSTALL_FLAG not in args


def pip_install(spec: str) -> None:
    """요구 사항 하나를 Forge venv 에 설치한다 — Forge 의 ``launch.run_pip``(``--skip-install``·``--uv``·INDEX_URL
    을 따름)을 쓰고, Forge 밖(단독 실행)에서는 ``python -m pip`` 으로. 실패하면 예외."""
    try:
        import launch  # type: ignore  # Forge 가 PYTHONPATH 에 webui 루트를 넣어 준다
    except Exception:
        launch = None
    run_pip = getattr(launch, "run_pip", None) if launch is not None else None
    if run_pip is not None:
        # run_pip 은 shell=True 로 실행한다 — '<' '>' 가 cmd 리다이렉트로 먹히지 않게 따옴표로 감싼다.
        run_pip(f'install "{spec}"', desc=f"forge_sam3_extension requirement: {spec}")
        return
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", spec],
        env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )


def _pip_command(specs: list[str]) -> str:
    return "pip install " + " ".join(f'"{s}"' for s in specs)


def check_environment(requirements: list[Requirement] | None = None, *, auto_install: bool | None = None,
                      installer=None) -> dict:
    """requirements.txt 를 확인해 빠진 것만 설치한다(토글이 꺼져 있으면 알리기만).

    이미 깔린 패키지는 버전이 범위 밖이어도 올리거나 내리지 않는다 — Forge 가 쓰는 판을 바꾸면 Forge 가
    깨질 수 있다. torch 계열은 목록에 있어도 설치하지 않는다. 문제가 없으면 아무것도 출력하지 않는다.
    """
    reqs = read_requirements() if requirements is None else list(requirements)
    if auto_install is None:
        auto_install = auto_install_enabled()
    installer = pip_install if installer is None else installer

    missing: list[Requirement] = []
    mismatch: list[Requirement] = []
    for req in reqs:
        status = requirement_status(req)
        if status == "missing":
            missing.append(req)
        elif status == "mismatch":
            mismatch.append(req)

    report = {"missing": [str(r) for r in missing], "mismatch": [str(r) for r in mismatch],
              "manual": [], "installed": [], "failed": []}

    for req in mismatch:
        try:
            installed = version(req.name)
        except Exception:
            installed = "?"
        _log(
            f"{req.name} {installed} is outside requirements.txt ({req}); left as is so Forge's own "
            "packages are never up/downgraded. Change it manually if SAM3 misbehaves."
        )

    if not missing:
        return report

    manual = [r for r in missing if _canonical(r.name) in _NEVER_AUTO_INSTALL]
    installable = [r for r in missing if _canonical(r.name) not in _NEVER_AUTO_INSTALL]
    report["manual"] = [str(r) for r in manual]
    if manual:
        _log(
            f"missing {', '.join(str(r) for r in manual)} — not installed automatically. Install the build that "
            "matches Forge's CUDA torch (see Forge's TORCH_COMMAND), never the default PyPI wheel."
        )
    if not installable:
        return report

    specs = [str(r) for r in installable]
    if not auto_install:
        _log(
            f"missing dependencies: {', '.join(specs)}. Auto-install is off "
            f"({AUTO_INSTALL_FLAG} / {AUTO_INSTALL_ENV}=1); install them in the Forge venv:\n"
            f"   {_pip_command(specs)}"
        )
        return report

    _log(f"installing missing dependencies into the Forge venv: {', '.join(specs)} "
         f"(turn off with {AUTO_INSTALL_FLAG})")
    for spec in specs:
        try:
            installer(spec)
        except Exception as exc:  # noqa: BLE001 — 하나 실패해도 나머지는 설치하고 끝에 알린다
            report["failed"].append(spec)
            _log(f"could not install {spec}: {exc}")
        else:
            report["installed"].append(spec)
    if report["failed"]:
        _log(f"install these manually in the Forge venv:\n   {_pip_command(report['failed'])}")
    return report


# ---------------------------------------------------------------------------
# v0.8.0 Anima vendor bootstrap — shallow-clone kohya-ss/sd-scripts once.
# ---------------------------------------------------------------------------
# Forge runs install.py as a subprocess at extension load (launch_utils.py),
# so this happens before scripts/!sam3.py ever imports. Failure is non-fatal:
# the Anima panel just hides itself, the rest of the SAM3 extension still
# works.

_ANIMA_REPO = "https://github.com/kohya-ss/sd-scripts.git"
_ANIMA_BRANCH = "main"
# Optional pin: set to a verified upstream tag/branch to stop the vendor tree
# floating with upstream main (which can silently break the source-patch
# anchors in lora_manager_core.py and the Anima import surface). None tracks
# _ANIMA_BRANCH — the historical behaviour. A shallow clone can pin a tag or
# branch name; a commit SHA goes in _ANIMA_PIN_COMMIT instead.
_ANIMA_PIN: str | None = None
# 커밋 핀(감사 M29): 검증된 sd-scripts 커밋 SHA 를 적으면 clone 뒤 그 커밋으로 고정한다(_checkout_pinned_commit).
# 아직 None — 이 PC 의 anima_vendor 는 .git 없이 복사돼 있어 어느 커밋인지 확인할 수 없다(감사 L68). 검증한
# 커밋을 알게 되면 여기에 적는다. None 이면 예전처럼 _ANIMA_BRANCH 끝을 쓴다.
_ANIMA_PIN_COMMIT: str | None = None
_ANIMA_ROOT = Path(__file__).resolve().parent / "anima_vendor"
# Sentinel = the actual entrypoint upstream ships. If the clone was
# interrupted partway through, this file will be missing and the next run
# re-attempts cleanly.
_ANIMA_SENTINEL = _ANIMA_ROOT / "anima_minimal_inference.py"


def _have_git() -> bool:
    try:
        subprocess.check_call(
            ["git", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
        ).strip()
    except Exception:
        return None


def _checkout_pinned_commit(root: Path, commit: str, label: str) -> bool:
    """shallow clone 한 ``root`` 를 ``commit`` 으로 고정한다(GitHub 는 SHA 로 fetch 를 허용).

    이미 그 커밋이면 아무것도 하지 않는다. 실패하면 브랜치 끝을 그대로 두고 stdout 에 알린 뒤 False.
    """
    if _git_head(root) == commit:
        return True
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        subprocess.check_call(["git", "-C", str(root), "fetch", "--depth", "1", "origin", commit], env=env)
        subprocess.check_call(["git", "-C", str(root), "checkout", "--detach", "--quiet", commit], env=env)
    except Exception as e:  # noqa: BLE001 — 핀 실패는 치명적이지 않다(브랜치 끝으로 계속)
        _log(
            f"could not pin {label} to commit {commit[:12]} ({e}); keeping the branch tip, which may not match "
            "what this extension was tested with."
        )
        return False
    return True


def _reset_partial_clone(root: Path, label: str) -> None:
    """Remove a non-empty vendor dir left by an interrupted clone.

    ``git clone`` refuses to write into an existing non-empty directory, so a
    clone that died after creating ``root`` but before the sentinel landed
    would wedge every subsequent bootstrap. The sentinel is already known to
    be missing by the time this is called, so anything present is partial.
    """
    if not root.exists():
        return
    _log(f"removing partial {label} clone at {root} before retrying")
    try:
        shutil.rmtree(root)
    except Exception as e:  # noqa: BLE001 — best-effort; clone will error clearly
        _log(f"could not remove {root}: {e}")


def ensure_anima_vendor() -> bool:
    """Clone kohya-ss/sd-scripts into ``anima_vendor/`` if the sentinel file
    is missing. Idempotent.

    Returns True when the vendor tree is usable after this call, False when
    bootstrap failed (the Anima panel reads ``anima_available()`` to decide
    whether to render).
    """
    if _ANIMA_SENTINEL.exists():
        return True

    if not _have_git():
        _log(
            "Anima panel disabled: 'git' not on PATH. "
            "Install git, or clone kohya-ss/sd-scripts manually into "
            f"{_ANIMA_ROOT}"
        )
        return False

    _reset_partial_clone(_ANIMA_ROOT, "Anima vendor")
    _ANIMA_ROOT.parent.mkdir(parents=True, exist_ok=True)
    _log(f"cloning sd-scripts → {_ANIMA_ROOT} (first-run, ~30s)")
    try:
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--branch",
                _ANIMA_PIN or _ANIMA_BRANCH,
                _ANIMA_REPO,
                str(_ANIMA_ROOT),
            ],
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.CalledProcessError as e:
        _log(f"Anima vendor clone failed (exit {e.returncode}); the Anima panel will be disabled.")
        return False

    if _ANIMA_PIN_COMMIT:
        _checkout_pinned_commit(_ANIMA_ROOT, _ANIMA_PIN_COMMIT, "Anima vendor")
    return _ANIMA_SENTINEL.exists()


# Anima vendor pulls a few packages that aren't part of the SAM3 core deps.
# We don't auto-pip these here (torchvision in particular needs to match the
# installed torch+cuda ABI, and forcing a wrong wheel breaks Forge itself).
# Just emit a one-line breadcrumb so the user knows what to install before
# clicking ▶ Anima Tile-Repair. The pure-Python one (imagesize) is also listed in
# requirements.txt, so check_environment() — which runs first — installs it when it is
# missing; it stays in this list so a run with auto-install off still names it here.
_ANIMA_DEPS = (
    "torchvision",  # match torch CUDA build manually
    "imagesize",
    "accelerate",
    "transformers",
    "diffusers",
    "einops",
    "huggingface_hub",
    "safetensors",
)


def check_anima_environment():
    if not _ANIMA_SENTINEL.exists():
        return  # vendor missing — panel won't render anyway
    missing = [pkg for pkg in _ANIMA_DEPS if not is_installed(pkg)]
    if missing:
        joined = " ".join(missing)
        _log(
            "Anima panel: missing deps "
            f"({', '.join(missing)}). Install in the Forge venv before "
            f"clicking ▶ Anima Tile-Repair, e.g.\n"
            f"   pip install {joined}\n"
            f"(For torchvision, match your installed torch's CUDA build.)"
        )


# ---------------------------------------------------------------------------
# v0.9.0 LoRA Manager vendor bootstrap — shallow-clone willmiao/
# ComfyUI-Lora-Manager + auto-install its (lightweight, pure-python) deps.
# ---------------------------------------------------------------------------
# Unlike the Anima vendor, the LoRA Manager's extra deps are all small
# pure-python packages (aiohttp-socks, piexif, olefile, natsort, aiosqlite,
# beautifulsoup4) with no torch/cuda ABI coupling, so auto-installing the
# missing ones is safe and was explicitly opted into by the user.

_LM_REPO = "https://github.com/willmiao/ComfyUI-Lora-Manager.git"
_LM_BRANCH = "main"
# Optional pin (see _ANIMA_PIN): a tag/branch name for the shallow clone.
_LM_PIN: str | None = None
# 커밋 핀(감사 M29): lora_manager_core.py 의 소스 패치 앵커(update_routes.py·base.html·PresetTags.js·
# ModelModal.js)를 확인한 판 — pyproject version 1.2.0, 커밋 303cca0(2026-09-22 감사 때 이 PC 의 vendor HEAD).
# 새로 clone 할 때 브랜치 끝 대신 이 커밋으로 고정한다. 판을 올리려면 앵커를 다시 확인한 뒤 두 값을 함께 바꾼다.
_LM_PIN_COMMIT: str | None = "303cca0d8553e1d94ba1b80a78f2b0a65e1ad8c3"
_LM_EXPECTED_VERSION = "1.2.0"
_LM_ROOT = Path(__file__).resolve().parent / "lora_manager_vendor"
_LM_SENTINEL = _LM_ROOT / "standalone.py"

# pip name -> import name (for is_installed's find_spec check)
_LM_DEP_IMPORT = {
    "aiohttp": "aiohttp",
    "aiohttp-socks": "aiohttp_socks",
    "jinja2": "jinja2",
    "safetensors": "safetensors",
    "piexif": "piexif",
    "Pillow": "PIL",
    "olefile": "olefile",
    "toml": "toml",
    "numpy": "numpy",
    "natsort": "natsort",
    "GitPython": "git",
    "aiosqlite": "aiosqlite",
    "beautifulsoup4": "bs4",
    "platformdirs": "platformdirs",
    "pyyaml": "yaml",
    "brotli": "brotli",
}


def ensure_lora_manager_vendor() -> bool:
    """Clone willmiao/ComfyUI-Lora-Manager into ``lora_manager_vendor/`` if
    the sentinel (standalone.py) is missing. Idempotent.

    Returns True when the vendor tree is usable after this call.
    """
    if _LM_SENTINEL.exists():
        return True

    if not _have_git():
        _log(
            "LoRA Manager disabled: 'git' not on PATH. "
            "Install git, or clone willmiao/ComfyUI-Lora-Manager manually "
            f"into {_LM_ROOT}"
        )
        return False

    _reset_partial_clone(_LM_ROOT, "LoRA Manager vendor")
    _LM_ROOT.parent.mkdir(parents=True, exist_ok=True)
    _log(f"cloning ComfyUI-Lora-Manager → {_LM_ROOT} (first-run, ~20s)")
    try:
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--branch",
                _LM_PIN or _LM_BRANCH,
                _LM_REPO,
                str(_LM_ROOT),
            ],
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.CalledProcessError as e:
        _log(f"LoRA Manager clone failed (exit {e.returncode}); the Manage tab will be disabled.")
        return False

    if _LM_PIN_COMMIT:
        _checkout_pinned_commit(_LM_ROOT, _LM_PIN_COMMIT, "LoRA Manager vendor")
    return _LM_SENTINEL.exists()


def _read_pyproject_version(path: Path) -> str | None:
    try:
        import tomllib

        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        return None
    value = (data.get("project") or {}).get("version")
    return str(value) if value else None


def check_lora_manager_version() -> None:
    """vendor 판이 소스 패치 앵커를 확인한 판(_LM_EXPECTED_VERSION)과 다르면 stdout 에 한 줄 알린다."""
    if not _LM_SENTINEL.exists():
        return
    found = _read_pyproject_version(_LM_ROOT / "pyproject.toml")
    if found and found != _LM_EXPECTED_VERSION:
        _log(
            f"LoRA Manager vendor is {found}, but this extension's patches were checked against "
            f"{_LM_EXPECTED_VERSION} (commit {(_LM_PIN_COMMIT or '?')[:12]}). Some Forge patches may be skipped; "
            f"delete {_LM_ROOT} to re-clone the pinned version."
        )


def check_lora_manager_environment():
    """Auto-install the LoRA Manager's missing pure-python deps into the
    Forge venv (user opted into auto-install). No version pins — we only add
    packages that are entirely absent so existing Forge packages are never
    downgraded. The auto-install toggle (AUTO_INSTALL_FLAG) turns this into
    a notice with the pip command."""
    if not _LM_SENTINEL.exists():
        return

    # Seed import_name so is_installed resolves the non-obvious ones.
    for pip_name, imp in _LM_DEP_IMPORT.items():
        import_name.setdefault(pip_name, imp)

    missing = [pip for pip in _LM_DEP_IMPORT if not is_installed(pip)]
    if not missing:
        return

    joined = " ".join(missing)
    if not auto_install_enabled():
        _log(
            f"LoRA Manager: missing deps ({', '.join(missing)}). Auto-install is off ({AUTO_INSTALL_FLAG}); "
            f"install manually:\n   pip install {joined}"
        )
        return
    _log(f"LoRA Manager: installing missing deps ({', '.join(missing)}) into the Forge venv...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing],
            env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
        )
    except subprocess.CalledProcessError as e:
        _log(f"LoRA Manager dep install failed (exit {e.returncode}). Install manually:\n   pip install {joined}")


def main() -> None:
    check_environment()
    ensure_anima_vendor()
    check_anima_environment()
    ensure_lora_manager_vendor()
    check_lora_manager_version()
    check_lora_manager_environment()


if __name__ == "__main__":
    main()
