# Derived from https://github.com/eduardoabreu81/forgeneo-mcp at commit a103dc5
# (forgeneo_mcp/server.py: the bodies of the tool functions) for sam-extra
# (mcp_server/sam_extra_mcp/service.py).
#
# Copyright (c) 2026 Eduardo Abreu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Changes in sam-extra (2026-10-03):
# - The tool bodies moved out of the MCP registration into a Service class that does not import
#   the MCP SDK, so they run (and are tested) in an environment without it. server.py only
#   registers them. No module-level client/state is created at import any more.
# - Permission policy (policy.py, Settings -> SAM Extra MCP), checked before anything else,
#   whatever the arguments: generate needs sam3_mcp_allow_generate; models action=load needs
#   sam3_mcp_allow_model_switch; progress interrupt/skip need sam3_mcp_allow_interrupt;
#   module_download needs sam3_mcp_allow_download for confirm=True and for the size probe
#   (without it nothing leaves the machine). A denial returns "denied_by_policy".
#   models action=refresh stays open on purpose: a checkpoint-list rescan changes no selection.
# - Paths come from forge_paths (the extension's own location): the output folder for results
#   (saved_by="api") and for the history (saved_by="ui"), the models folder for downloads, the
#   fallback folder for images saved from the API response, and the dialect cache folder.
#   The history folder follows the live options instead of being resolved once.
# - generate: passes the fallback folder (upstream passed none, so it answered ok:false after
#   Forge had rendered), marks the history stale after a success, and clamps batch_size (1-8),
#   steps (1-150) and width/height (64-4096), reporting any adjustment. A call whose payload
#   would render more than config.max_pixels in total (width x height, hires size included,
#   x batch_size x n_iter; SAM_EXTRA_MCP_MAX_PIXELS) is refused before Forge is asked.
# - Refresh: capabilities, model_profile and loras take refresh=True to rebuild the indexes.
# - models: action=list caps `limit` once (upstream reported a "returned" count that could
#   differ from the list); unknown progress actions are errors (upstream answered with the
#   status, which reads like success for "stop").
# - module_download writes into <Forge data>/models/<VAE|text_encoder> from forge_paths (the
#   extension's location, or SAM_EXTRA_MCP_MODELS_DIR), never into a root guessed from a module
#   path (upstream cut Forge's module paths at "/models/" and rebuilt them through FORGE_PATH_MAP).
# - loras / lora_info carry UNTRUSTED_NOTICE: the LoRA text they return was written by the
#   LoRA's author or a download tool, not by Forge or the operator.

"""The tools of the sam-extra MCP server, without the MCP SDK.

Design rule (upstream's): the tools are faithful, not clever. `generate` sends
exactly the prompt it is given and never injects a LoRA on its own — discovery
lives in `loras`, and the decision to use one belongs to the agent that called it.

sam-extra's rule on top: what may change in Forge is the operator's decision,
made in Forge -> Settings -> SAM Extra MCP and re-read before every such call.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from typing import Any

from . import __version__, forge_paths
from .forge_paths import ForgePaths
from .forgeneo import UPSTREAM_COMMIT, dialects, downloads, fetcher, identity, modules
from .forgeneo.capabilities import probe
from .forgeneo.client import ForgeClient
from .forgeneo.config import Config
from .forgeneo.generate import (
    build_payload,
    encode_init_image,
    requested_pixels,
    resolve_output_dir,
    run_generation,
)
from .forgeneo.history import HistoryIndex
from .forgeneo.loras import UNTRUSTED_NOTICE, LoraIndex
from .forgeneo.profile import (
    _checkpoint_identity,
    _module_health,
    build_profile,
    resolve_dialect,
    switch_checkpoint,
)
from .policy import (
    ALLOW_DOWNLOAD,
    ALLOW_GENERATE,
    ALLOW_INTERRUPT,
    ALLOW_MODEL_SWITCH,
    Policy,
    PolicyReader,
    denial,
)

MAX_BATCH_SIZE = 8
MAX_STEPS = 150
MIN_SIDE = 64
MAX_SIDE = 4096
PROGRESS_ACTIONS = ("status", "interrupt", "skip")
MODEL_ACTIONS = ("list", "load", "refresh")


class Service:
    """Everything the MCP tools do, against one Forge instance."""

    def __init__(
        self,
        config: Config,
        paths: ForgePaths,
        *,
        client: Any = None,
        transport: Any = None,
        policy_reader: PolicyReader | None = None,
        download_transport: Any = None,
    ) -> None:
        self.config = config
        self.paths = paths
        self.client = client if client is not None else ForgeClient(config, transport=transport)
        self.policy_reader = policy_reader or PolicyReader(
            paths.settings_file,
            same_machine=config.same_machine,
            explicit=paths.settings_explicit,
        )
        self.history = HistoryIndex(
            config.output_dir, limit=config.history_limit, max_age=config.history_max_age
        )
        self.lora_index = LoraIndex(self.client, self.history, max_age=config.history_max_age)
        self._download_transport = download_transport
        self._history_lock = threading.Lock()
        identity.configure(paths.cache_dir)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        transport: Any = None,
        module_file: str | None = None,
    ) -> "Service":
        return cls(Config.from_env(env), forge_paths.discover(env, module_file=module_file), transport=transport)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    # -- policy ---------------------------------------------------------------

    def policy(self) -> Policy:
        return self.policy_reader.read()

    def _refused(self, key: str) -> dict | None:
        """None when `key` is allowed right now, else the denial to return."""
        current = self.policy()
        if current.allows(key):
            return None
        return denial(current, key)

    # -- shared plumbing --------------------------------------------------------

    def _options(self) -> tuple[dict | None, str | None]:
        result = self.client.options()
        if not result.ok:
            return None, result.error
        data = result.value if isinstance(result.value, dict) else {}
        self._sync_history_dir(data)
        return data, None

    def _sync_history_dir(self, options: dict) -> None:
        """Point the history at the folder the operator's own generations go to."""
        folder = resolve_output_dir(self.client, options, mode="txt2img", paths=self.paths, saved_by="ui")
        with self._history_lock:
            if folder and folder != self.history.output_dir:
                self.history.set_output_dir(folder)

    def _api_output_dir(self, options: dict, mode: str) -> str | None:
        return resolve_output_dir(self.client, options, mode=mode, paths=self.paths, saved_by="api")

    def _server_section(self) -> dict:
        return {
            "name": "sam-extra-mcp",
            "version": __version__,
            "vendored": f"forgeneo-mcp {UPSTREAM_COMMIT}",
            "config": self.config.describe(),
        }

    # -- tools ------------------------------------------------------------------

    def capabilities(self, refresh: bool = False) -> dict:
        data, _ = self._options()
        output_dir = self._api_output_dir(data, "txt2img") if data is not None else None
        extra = {
            "policy": self.policy().as_dict(),
            "forge_paths": self.paths.as_dict(),
            "server": self._server_section(),
        }
        return probe(self.client, self.history, self.lora_index, output_dir, refresh=refresh, extra=extra).as_dict()

    def model_profile(self, refresh: bool = False) -> dict:
        self._options()
        if refresh:
            self.history.refresh()
        result = build_profile(self.client, self.history)
        if isinstance(result, str):
            return {"ok": False, "error": result}
        return {"ok": True, **result.as_dict()}

    def prompt_dialect(self, confirm: str = "") -> dict:
        data, error = self._options()
        if data is None:
            return {"ok": False, "error": error}

        checkpoint = data.get("sd_model_checkpoint")
        preset = data.get("forge_preset")

        if confirm:
            key = confirm.strip().lower()
            if key not in dialects.BY_KEY:
                return {
                    "ok": False,
                    "error": f"unknown dialect '{confirm}'",
                    "choices": sorted(dialects.BY_KEY),
                }
            sha, _ = _checkpoint_identity(self.client, checkpoint)
            stored = identity.remember(sha or (checkpoint or ""), key)
            return {
                "ok": True,
                "confirmed": key,
                "cached": stored,
                **dialects.BY_KEY[key].as_dict(),
            }

        resolution = resolve_dialect(self.client, self.history, checkpoint, preset, self.lora_index.all())
        return {"ok": True, "checkpoint": checkpoint, "architecture": preset, **resolution.as_dict()}

    def loras(
        self,
        query: str = "",
        base_model: str = "",
        kind: str = "",
        limit: int = 20,
        verbose: bool = False,
        refresh: bool = False,
    ) -> dict:
        self._options()
        if refresh:
            self.lora_index.refresh()
        limit = max(1, min(int(limit), 50))
        matches = self.lora_index.search(query=query, base_model=base_model or None, kind=kind or None, limit=limit)
        return {
            "ok": self.lora_index.error is None,
            "error": self.lora_index.error,
            "query": query,
            "returned": len(matches),
            "untrusted_text": UNTRUSTED_NOTICE,
            "summary": self.lora_index.summary(),
            "results": [entry.as_dict(verbose=verbose) for entry in matches],
        }

    def lora_info(self, name: str) -> dict:
        self._options()
        entry = self.lora_index.get(name)
        if not entry:
            return {"ok": False, "error": f"no LoRA named '{name}'"}
        return {"ok": True, "untrusted_text": UNTRUSTED_NOTICE, **entry.as_dict(verbose=True)}

    def models(self, action: str = "list", name: str = "", preset: str = "", query: str = "", limit: int = 30) -> dict:
        action = (action or "list").lower().strip()
        if action == "refresh":
            # Deliberately not behind a permission: Forge re-reads its checkpoint folders
            # (sd_models.list_models, after any running job - it takes the queue lock) and the list
            # gains or loses entries; the loaded model, the selection and every file stay as they are.
            result = self.client.refresh_checkpoints()
            return {"ok": result.ok, "error": result.error}

        if action == "load":
            refused = self._refused(ALLOW_MODEL_SWITCH)
            if refused:
                return refused
            if not name:
                return {"ok": False, "error": "name is required to load a checkpoint"}
            return switch_checkpoint(self.client, name, preset or None)

        if action != "list":
            return {"ok": False, "error": f"unknown action '{action}'", "choices": list(MODEL_ACTIONS)}

        result = self.client.checkpoints()
        if not result.ok:
            return {"ok": False, "error": result.error}

        entries = result.value or []
        needle = query.lower().strip()
        filtered = [
            {"title": item.get("title"), "model_name": item.get("model_name")}
            for item in entries
            if not needle or needle in str(item.get("title", "")).lower()
        ]
        cap = max(1, min(int(limit), 100))
        return {
            "ok": True,
            "total": len(entries),
            "matching": len(filtered),
            "returned": len(filtered[:cap]),
            "results": filtered[:cap],
        }

    def module_check(self, preset: str = "") -> dict:
        data, error = self._options()
        if data is None:
            return {"ok": False, "error": error}

        arch = (preset or data.get("forge_preset") or "").strip()
        selected = data.get(f"forge_additional_modules_{arch}") or []

        listing = self.client.modules()
        if not listing.ok:
            return {"ok": False, "error": listing.error}

        report = modules.audit(arch, list(selected), list(listing.value or []))
        if not report.get("known"):
            return {
                "ok": True,
                "architecture": arch,
                "known": False,
                "detail": "no module requirements recorded for this architecture",
                "currently_selected": [str(item).replace(chr(92), "/").split("/")[-1] for item in selected],
            }
        return {"ok": True, **report}

    def module_download(self, preset: str = "", label: str = "", confirm: bool = False) -> dict:
        data, error = self._options()
        if data is None:
            return {"ok": False, "error": error}

        arch = (preset or data.get("forge_preset") or "").strip()
        available = downloads.for_architecture(arch)
        if not available:
            return {"ok": True, "architecture": arch, "available": [], "detail": "no catalogue entry"}

        current = self.policy()
        allowed = current.allows(ALLOW_DOWNLOAD)
        models_dir = self._models_dir()

        if not label:
            health = _module_health(self.client, data, arch)
            return {
                "ok": True,
                "architecture": arch,
                "current_state": "healthy" if health.get("healthy") else "incomplete",
                "problems": list(health.get("problems", [])),
                "available": [entry.as_dict() for entry in available],
                "models_dir": models_dir,
                "downloads_allowed": allowed,
                "how_to_download": (
                    "ask the operator which build they want, then call again with that label and confirm=True"
                    if allowed
                    else "downloads are switched off in Forge -> Settings -> SAM Extra MCP; give the operator "
                    "the page links instead, or ask them to allow downloads there"
                ),
            }

        chosen = next((entry for entry in available if entry.label.lower() == label.strip().lower()), None)
        if chosen is None:
            return {
                "ok": False,
                "error": f"no entry called '{label}' for {arch}",
                "choices": [entry.label for entry in available],
            }

        if confirm and not allowed:
            return denial(current, ALLOW_DOWNLOAD)

        if not models_dir:
            return {
                "ok": False,
                "error": (
                    "cannot locate the Forge models folder from here. Set SAM_EXTRA_MCP_MODELS_DIR, or "
                    f"download {chosen.direct_url} into {chosen.target_folder} manually"
                ),
            }

        if chosen.is_directory:
            return {
                "ok": True,
                "fetchable": False,
                "entry": chosen.as_dict(),
                "detail": (
                    "this entry links a folder of builds rather than one file, so it cannot be "
                    "fetched automatically. Open the page, choose the build that suits the hardware, "
                    f"and save it into {chosen.target_folder}"
                ),
            }

        folder = os.path.join(models_dir, "VAE" if chosen.kind == modules.VAE else "text_encoder")
        if not allowed:
            intent = fetcher.plan(chosen.direct_url, folder, chosen.filename, probe=False)
            return {
                "ok": True,
                "would_download": chosen.as_dict(),
                "plan": intent.as_dict(),
                "confirmed": False,
                "downloads_allowed": False,
                "detail": (
                    "downloads are switched off in Forge -> Settings -> SAM Extra MCP, so nothing was "
                    "requested from the network and nothing can be downloaded from here"
                ),
            }

        intent = fetcher.plan(chosen.direct_url, folder, chosen.filename, transport=self._download_transport)
        if not confirm:
            return {
                "ok": True,
                "would_download": chosen.as_dict(),
                "plan": intent.as_dict(),
                "confirmed": False,
                "downloads_allowed": True,
                "detail": "nothing was downloaded; pass confirm=True once the operator agrees",
            }

        result = fetcher.fetch(
            chosen.direct_url, folder, chosen.filename, permitted=True, transport=self._download_transport
        )
        return {"architecture": arch, "label": chosen.label, **result}

    def _models_dir(self) -> str | None:
        """Forge's models folder: <Forge data>/models from the extension's own location, or
        SAM_EXTRA_MCP_MODELS_DIR. Never inferred from a module path Forge reports: a folder
        named "vae" or "text_encoder" higher up such a path would send multi-GB files elsewhere."""
        folder = self.paths.models_dir
        return os.path.normpath(folder) if folder and os.path.isdir(folder) else None

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int | None = None,
        cfg_scale: float | None = None,
        sampler_name: str = "",
        scheduler: str = "",
        shift: float | None = None,
        width: int = 0,
        height: int = 0,
        seed: int = -1,
        batch_size: int = 1,
        init_image: str = "",
        denoising_strength: float = 0.7,
        use_profile_defaults: bool = True,
    ) -> dict:
        refused = self._refused(ALLOW_GENERATE)
        if refused:
            return refused
        if not prompt or not prompt.strip():
            return {"ok": False, "error": "prompt is empty"}

        mode = "img2img" if init_image else "txt2img"
        extra: dict[str, Any] = {}
        if init_image:
            encoded, error = encode_init_image(init_image)
            if error:
                return {"ok": False, "error": error}
            extra["init_images"] = [encoded]
            extra["denoising_strength"] = max(0.0, min(float(denoising_strength), 1.0))

        adjusted: dict[str, Any] = {}

        def clamp(name: str, value: int | None, low: int, high: int) -> int | None:
            if value is None:
                return None
            bounded = max(low, min(int(value), high))
            if bounded != value:
                adjusted[name] = {"asked": value, "used": bounded}
            return bounded

        steps = clamp("steps", steps, 1, MAX_STEPS)
        width = clamp("width", width, MIN_SIDE, MAX_SIDE) if width else 0
        height = clamp("height", height, MIN_SIDE, MAX_SIDE) if height else 0
        batch = clamp("batch_size", batch_size, 1, MAX_BATCH_SIZE) or 1

        resolved: dict[str, Any] = {
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler_name": sampler_name or None,
            "scheduler": scheduler or None,
            "distilled_cfg_scale": shift,
            "width": width or None,
            "height": height or None,
        }
        applied_from_profile: list[str] = []

        data, error = self._options()
        if use_profile_defaults and any(value is None for value in resolved.values()):
            profile = build_profile(self.client, self.history)
            if not isinstance(profile, str):
                for key, value in (
                    ("steps", int(profile.steps) if profile.steps is not None else None),
                    ("cfg_scale", profile.cfg),
                    ("sampler_name", profile.sampler),
                    ("scheduler", profile.scheduler),
                    ("distilled_cfg_scale", profile.shift),
                    ("width", profile.width),
                    ("height", profile.height),
                ):
                    if resolved[key] is None and value is not None:
                        resolved[key] = value
                        applied_from_profile.append(key)

        payload = build_payload(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=resolved["steps"],
            cfg_scale=resolved["cfg_scale"],
            sampler_name=resolved["sampler_name"],
            scheduler=resolved["scheduler"],
            distilled_cfg_scale=resolved["distilled_cfg_scale"],
            width=resolved["width"] or 1024,
            height=resolved["height"] or 1024,
            seed=seed,
            batch_size=batch,
            extra=extra or None,
        )

        # The pixel budget is checked on the final payload, so sizes taken from the profile count too.
        pixels = requested_pixels(payload)
        if pixels > self.config.max_pixels:
            return {
                "ok": False,
                "error": (
                    f"too large for one call: {payload['width']}x{payload['height']} x {payload['batch_size']} "
                    f"image(s) = {pixels / 1e6:.1f} megapixels, and the limit is {self.config.max_pixels / 1e6:.1f} MP "
                    "(all images of the call, hires pass included). Use a smaller size or batch_size, or several "
                    "calls; the operator can change the limit with SAM_EXTRA_MCP_MAX_PIXELS."
                ),
                "pixels": pixels,
                "max_pixels": self.config.max_pixels,
                "mode": mode,
                **({"defaults_applied": applied_from_profile} if applied_from_profile else {}),
            }

        output_dir = self._api_output_dir(data, mode) if data is not None else None
        result = run_generation(self.client, payload, output_dir, mode=mode, fallback_dir=self.paths.fallback_dir)
        if result.ok:
            self.history.mark_stale()
        response = result.as_dict()
        response["mode"] = mode
        if applied_from_profile:
            response["defaults_applied"] = applied_from_profile
        if adjusted:
            response["adjusted"] = adjusted
        if data is None and error:
            response.setdefault("warnings", []).append(f"could not read Forge's options: {error}")
        return response

    def progress(self, action: str = "status") -> dict:
        action = (action or "status").lower().strip()
        if action in ("interrupt", "skip"):
            refused = self._refused(ALLOW_INTERRUPT)
            if refused:
                return refused
            result = self.client.interrupt() if action == "interrupt" else self.client.skip()
            return {"ok": result.ok, "action": action, "error": result.error}
        if action != "status":
            return {"ok": False, "error": f"unknown action '{action}'", "choices": list(PROGRESS_ACTIONS)}

        result = self.client.progress()
        if not result.ok:
            return {"ok": False, "error": result.error}
        data = result.value or {}
        state = data.get("state") or {}
        return {
            "ok": True,
            "progress": round(float(data.get("progress") or 0.0), 3),
            "eta_seconds": round(float(data.get("eta_relative") or 0.0), 1),
            "job": state.get("job"),
            "step": state.get("sampling_step"),
            "total_steps": state.get("sampling_steps"),
            "interrupt_allowed": self.policy().allows(ALLOW_INTERRUPT),
        }
