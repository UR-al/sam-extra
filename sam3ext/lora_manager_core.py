"""LoRA Manager integration core — lazy-spawns willmiao/ComfyUI-Lora-Manager's
standalone aiohttp server and points Forge's UI at it via an iframe.

Design (v0.9.0)
---------------
- The manager is vendored (shallow clone) at ``lora_manager_vendor/`` by
  install.py. We never modify the vendor tree.
- We write ``lora_manager_vendor/settings.json`` with ``use_portable_settings:
  true`` so the manager loads its config from there (deterministic, no
  platformdirs surprises) and point its ``folder_paths`` at Forge's actual
  model folders (LoRA / checkpoints / embeddings / VAE).
- The server is spawned lazily — only the first time the user opens the
  "Manage" tab — as a child process. We poll its ``/loras`` endpoint until it
  answers, then hand the URL back to the JS that injected the tab.
- ``atexit`` + a tracked Popen handle make sure the child dies with Forge.

This module is framework-agnostic (no Gradio import) so it can be unit-probed
standalone, mirroring anima_core / inpaint_core.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

EXTENSION_ROOT = Path(__file__).resolve().parent.parent
LM_VENDOR = EXTENSION_ROOT / "lora_manager_vendor"
LM_STANDALONE = LM_VENDOR / "standalone.py"
LM_SETTINGS = LM_VENDOR / "settings.json"
LM_LOG = LM_VENDOR / "forge_standalone.log"

DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"

# Spawn coordination — one server per Forge process.
_proc: subprocess.Popen | None = None
_proc_port: int | None = None
_spawn_lock = threading.Lock()


def lora_manager_available() -> bool:
    """Cheap probe used by the UI bridge — vendor cloned and entrypoint present."""
    return LM_STANDALONE.exists()


# ---------------------------------------------------------------------------
# settings.json — point the manager at Forge's model folders
# ---------------------------------------------------------------------------


def _forge_folder_paths() -> dict[str, list[str]]:
    """Collect Forge's LoRA / checkpoint / embeddings / VAE / DiT folders as
    the ``folder_paths`` dict the manager expects. Every value is a list and
    only existing directories are kept."""
    folders: dict[str, list[str]] = {
        "loras": [],
        "checkpoints": [],
        "embeddings": [],
        "unet": [],
    }
    try:
        from modules import paths, sd_models, shared

        co = shared.cmd_opts
        models_root = Path(paths.models_path)

        # LoRA: sd_forge_lora preload adds --lora-dir (single, default
        # models/Lora); --lora-dirs (list) is the core multi-dir flag.
        lora_dirs: list[str] = []
        single = getattr(co, "lora_dir", None)
        if single:
            lora_dirs.append(str(single))
        lora_dirs.extend(str(d) for d in (getattr(co, "lora_dirs", None) or []))
        if not lora_dirs:
            lora_dirs.append(str(models_root / "Lora"))
        folders["loras"] = lora_dirs

        # Checkpoints: sd_models.model_path (== models/Stable-diffusion) +
        # --ckpt-dirs.
        ckpt_dirs = [str(getattr(sd_models, "model_path", models_root / "Stable-diffusion"))]
        ckpt_dirs.extend(str(d) for d in (getattr(co, "ckpt_dirs", None) or []))
        folders["checkpoints"] = ckpt_dirs

        # Embeddings (textual inversion).
        emb = getattr(co, "embeddings_dir", None)
        folders["embeddings"] = [str(emb)] if emb else [str(models_root / "embeddings")]

        # DiT / diffusion models — Forge keeps these in Stable-diffusion too;
        # the manager merges unet+diffusers into "diffusion_models". Point it
        # at text_encoder for Qwen DiT/TE setups if present, else skip.
        te = models_root / "text_encoder"
        if te.is_dir():
            folders["unet"] = [str(te)]

        # VAE → there's no dedicated manager key beyond what it scans, but we
        # add it under a custom key the manager ignores gracefully if unknown.
        try:
            from modules import sd_vae

            vae_dirs = [str(getattr(sd_vae, "vae_path", models_root / "VAE"))]
            vae_dirs.extend(str(d) for d in (getattr(co, "vae_dirs", None) or []))
            folders["vae"] = vae_dirs
        except Exception:
            pass
    except Exception:
        # Forge modules not importable (shouldn't happen at runtime) — fall
        # back to conventional layout so the server at least starts.
        root = EXTENSION_ROOT.parent.parent / "models"
        folders["loras"] = [str(root / "Lora")]
        folders["checkpoints"] = [str(root / "Stable-diffusion")]
        folders["embeddings"] = [str(root / "embeddings")]

    # Drop non-existent dirs but keep at least the declared LoRA path so the
    # manager doesn't error on an empty list.
    cleaned: dict[str, list[str]] = {}
    for key, dirs in folders.items():
        existing = [d for d in dirs if d and os.path.isdir(d)]
        cleaned[key] = existing or dirs
    return cleaned


def ensure_settings_json() -> None:
    """Write/refresh ``lora_manager_vendor/settings.json`` in portable mode so
    the manager loads Forge's folder paths. Preserves a user-set
    ``civitai_api_key`` across rewrites (we only manage folder_paths)."""
    if not LM_VENDOR.is_dir():
        return

    existing: dict[str, Any] = {}
    if LM_SETTINGS.exists():
        try:
            existing = json.loads(LM_SETTINGS.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    payload = dict(existing)
    payload["use_portable_settings"] = True
    payload["folder_paths"] = _forge_folder_paths()
    # Keep an existing key; only seed a placeholder when absent.
    payload.setdefault("civitai_api_key", "")
    payload.setdefault("auto_organize_exclusions", [])

    try:
        LM_SETTINGS.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        print(f"[-] LoRA Manager: failed to write settings.json: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Health check + lazy spawn
# ---------------------------------------------------------------------------


def _health_ok(port: int, timeout: float = 1.0) -> bool:
    """True when the manager answers on ``/loras``. The root ``/`` 302-
    redirects to ``/loras`` so we hit ``/loras`` directly and accept any 2xx."""
    import urllib.request
    import urllib.error

    url = f"http://{DEFAULT_HOST}:{port}/loras"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except urllib.error.HTTPError as e:
        # Any HTTP response means the server is up (even a 4xx/5xx page).
        return True if e.code else False
    except Exception:
        return False


def is_running(port: int | None = None) -> bool:
    p = port if port is not None else (_proc_port or DEFAULT_PORT)
    return _health_ok(p)


def manager_url(port: int) -> str:
    return f"http://{DEFAULT_HOST}:{port}/loras"


def get_or_spawn(port: int = DEFAULT_PORT, wait_seconds: float = 40.0) -> dict[str, Any]:
    """Lazy entry point called by the UI bridge. Returns
    ``{"url": ..., "port": ..., "status": "running"|"spawned"|"error",
       "message": ...}``.

    - If a server already answers on ``port`` (ours or an external one), reuse.
    - Otherwise spawn standalone.py as a child and poll until healthy.
    """
    global _proc, _proc_port

    if not lora_manager_available():
        return {
            "url": "",
            "port": port,
            "status": "error",
            "message": "LoRA Manager vendor missing — re-run install.py.",
        }

    # Fast path: something is already serving on this port.
    if _health_ok(port):
        _proc_port = port
        return {
            "url": manager_url(port),
            "port": port,
            "status": "running",
            "message": "already running",
        }

    with _spawn_lock:
        # Re-check inside the lock (another thread may have spawned it).
        if _health_ok(port):
            _proc_port = port
            return {
                "url": manager_url(port),
                "port": port,
                "status": "running",
                "message": "already running",
            }

        # If our tracked process died, clear it.
        if _proc is not None and _proc.poll() is not None:
            _proc = None

        try:
            ensure_settings_json()
        except Exception:
            pass

        # Spawn. CREATE_NO_WINDOW on Windows so no extra console pops up; logs
        # go to forge_standalone.log for debugging.
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            log_fh = open(LM_LOG, "w", encoding="utf-8")
        except Exception:
            log_fh = subprocess.DEVNULL  # type: ignore[assignment]

        try:
            _proc = subprocess.Popen(
                [
                    sys.executable,
                    str(LM_STANDALONE),
                    "--host",
                    DEFAULT_HOST,
                    "--port",
                    str(port),
                ],
                cwd=str(LM_VENDOR),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            _proc_port = port
        except Exception as e:
            return {
                "url": "",
                "port": port,
                "status": "error",
                "message": f"spawn failed: {e}",
            }

    # Poll for readiness OUTSIDE the lock so other callers can short-circuit
    # on _health_ok once we're up.
    import time

    deadline_steps = int(max(1, wait_seconds / 0.5))
    for _ in range(deadline_steps):
        if _proc is not None and _proc.poll() is not None:
            # Process exited before becoming healthy — surface the log tail.
            tail = _read_log_tail()
            return {
                "url": "",
                "port": port,
                "status": "error",
                "message": f"server exited early. Log tail:\n{tail}",
            }
        if _health_ok(port):
            return {
                "url": manager_url(port),
                "port": port,
                "status": "spawned",
                "message": "started",
            }
        time.sleep(0.5)

    return {
        "url": manager_url(port),
        "port": port,
        "status": "error",
        "message": f"timed out after {wait_seconds}s waiting for the server. "
        f"Check {LM_LOG.name}.",
    }


def _read_log_tail(n: int = 20) -> str:
    try:
        lines = LM_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return "(no log)"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def shutdown() -> None:
    """Terminate the child server. Registered with atexit so the manager dies
    with Forge."""
    global _proc
    if _proc is None:
        return
    if _proc.poll() is None:
        try:
            _proc.terminate()
            try:
                _proc.wait(timeout=5)
            except Exception:
                _proc.kill()
        except Exception:
            pass
    _proc = None


atexit.register(shutdown)
