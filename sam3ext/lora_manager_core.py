"""LoRA Manager integration core — lazy-spawns willmiao/ComfyUI-Lora-Manager's
standalone aiohttp server and points Forge's UI at it via an iframe.

Design (introduced in v0.9.0)
-----------------------------
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

# Marker so our appended CSS override is applied exactly once per vendor copy
# (and re-applied automatically if the vendor tree is re-cloned, since the
# fresh file won't contain the marker).
_CSS_MARKER = "/* === forge_sam3 css override (auto-applied) === */"

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
# CSS overrides (small UX patches to the vendored UI)
# ---------------------------------------------------------------------------
# The manager runs as a separate cross-origin server, so Forge can't inject
# CSS into the iframe. Instead we append our overrides directly to vendor CSS
# files on disk (idempotent, marker-guarded). aiohttp serves static files from
# disk per request, so a browser hard-refresh picks up the change without
# restarting the server. Re-clone of the vendor wipes our edits → re-applied.

_CSS_OVERRIDES = {
    # Fetch-all progress overlay: the status line is a single string
    #   "Processing (n/total) <LONG LORA NAME> | ❌ N failed | ⏭️ N skipped"
    # and .loading-status is white-space:nowrap + overflow:hidden + ellipsis,
    # so a long filename eats the line and clips the "failed/skipped" counters.
    # Allow wrapping so the counters drop to the next line instead of clipping.
    "static/css/components/loading.css": (
        ".loading-status {\n"
        "    white-space: normal !important;\n"
        "    overflow: visible !important;\n"
        "    text-overflow: clip !important;\n"
        "    word-break: break-word !important;\n"
        "    line-height: 1.4 !important;\n"
        "    max-width: min(90vw, 680px) !important;\n"
        "}\n"
    ),
    # Donation / support UI removal. GPL-3.0 permits feature removal; we only
    # hide DOM nodes via CSS — LICENSE, copyright notices, and author
    # attribution (README/source headers) are untouched, so §5(c) "Appropriate
    # Legal Notices" stay intact. This file is @import-ed by style.css so the
    # rules apply globally (header + banners), not just inside the modal.
    "static/css/components/modal/support-modal.css": (
        "/* Header triggers that open the support/donation modal */\n"
        "#supportToggleBtn { display: none !important; }\n"
        "#hamburgerDropdown .dropdown-item[data-action=\"support\"] { display: none !important; }\n"
        "#hamburgerDropdown .dropdown-divider { display: none !important; }\n"
        "/* Entire 'Support This Project' modal (Ko-fi, Patreon, WeChat QR,\n"
        "   supporters list, social links all live inside it) */\n"
        "#supportModal { display: none !important; }\n"
        "/* Defensive per-element rules (durable if the modal is refactored) */\n"
        "#supportModal a.kofi-button,\n"
        "#supportModal a.patreon-button,\n"
        "#supportModal #toggleQRCode,\n"
        "#supportModal #qrCodeContainer,\n"
        "#supportModal .support-right,\n"
        "#specialThanksGrid,\n"
        "#supportersGrid { display: none !important; }\n"
        "/* JS-injected live community-support donation banner */\n"
        "#banner-container .banner-item[data-banner-id=\"community-support\"] { display: none !important; }\n"
        "/* Donation links replayed in the banner-history panel */\n"
        "#bannerHistoryList a.banner-history-action[href*=\"ko-fi.com\"],\n"
        "#bannerHistoryList a.banner-history-action[href*=\"afdian.com\"] { display: none !important; }\n"
    ),
    # Notes field real-placeholder (Bug B). The source patch (apply_modal_bug_
    # patches) makes the notes div start EMPTY with a data-placeholder attr;
    # this CSS renders that placeholder via :empty::before so it clears on
    # focus/typing in every locale (the old code injected the localized text as
    # real content and only cleared the hardcoded English string). Appended to
    # style.css (always loaded, not previously patched).
    "static/css/style.css": (
        ".notes-content:empty::before {\n"
        "    content: attr(data-placeholder);\n"
        "    opacity: 0.45;\n"
        "    pointer-events: none;\n"
        "}\n"
    ),
}


def apply_css_overrides() -> None:
    """Append our marker-guarded CSS overrides to the vendor's CSS files.

    Idempotent: skips files that already contain ``_CSS_MARKER``. Safe to call
    on every spawn. Best-effort — failures are logged, never fatal.
    """
    if not LM_VENDOR.is_dir():
        return
    for rel_path, css in _CSS_OVERRIDES.items():
        target = LM_VENDOR / rel_path
        try:
            if not target.is_file():
                continue
            existing = target.read_text(encoding="utf-8", errors="replace")
            if _CSS_MARKER in existing:
                continue
            with target.open("a", encoding="utf-8") as fh:
                fh.write(f"\n\n{_CSS_MARKER}\n{css}")
            print(
                f"[-] LoRA Manager: applied CSS override to {rel_path}",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"[-] LoRA Manager: failed to apply CSS override to {rel_path}: {e}",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Update-check disable (stop the manager nagging about willmiao upstream)
# ---------------------------------------------------------------------------
# The notification center's only feature that actually follows the ORIGINAL
# project is the backend update check (py/routes/update_routes.py), which polls
# api.github.com/repos/willmiao/ComfyUI-Lora-Manager releases and surfaces an
# "update available" dot. There is no settings.json flag for it, and the
# frontend gate is per-browser localStorage — so we can't toggle it at spawn.
# We vendor a pinned copy (install.py controls the version), so the user can't
# act on upstream releases anyway; the nag is pure noise. We short-circuit
# check_updates() to always report "no update" via a marker-guarded source
# patch (same idempotent, re-applied-on-reclone pattern as the CSS overrides).
#
# Repointing the check to UR-al/sam-extra is deliberately NOT done: the
# manager compares against its own pyproject version (1.1.4), so our SAM3
# repo's tags would yield nonsensical results, and its self-update would
# overwrite the pinned vendor tree.

_UPDATE_PATCH_MARKER = "# === forge_sam3 update-check disabled (auto-applied) ==="
_UPDATE_ROUTE_REL = "py/routes/update_routes.py"
# Unique line (verified) at the top of check_updates()'s try-block; `nightly`
# and `web`/`UpdateRoutes` are all in scope right after it.
_UPDATE_ANCHOR = (
    "            nightly = request.query.get('nightly', 'false').lower() == 'true'\n"
)
_UPDATE_INJECT = (
    _UPDATE_ANCHOR
    + "            " + _UPDATE_PATCH_MARKER + "\n"
    + "            # Forge: vendor pinned by install.py; user can't act on\n"
    + "            # willmiao upstream releases — suppress the update nag.\n"
    + "            return web.json_response({\n"
    + "                'success': True,\n"
    + "                'current_version': UpdateRoutes._get_local_version(),\n"
    + "                'latest_version': UpdateRoutes._get_local_version(),\n"
    + "                'update_available': False,\n"
    + "                'changelog': '',\n"
    + "                'nightly': nightly,\n"
    + "            })\n"
)


def apply_update_check_patch() -> None:
    """Disable the vendored update check so the LoRA Manager stops nagging
    about willmiao upstream releases. Idempotent + marker-guarded; re-applied
    automatically when the vendor tree is re-cloned. Best-effort."""
    if not LM_VENDOR.is_dir():
        return
    target = LM_VENDOR / _UPDATE_ROUTE_REL
    try:
        if not target.is_file():
            return
        src = target.read_text(encoding="utf-8", errors="replace")
        if _UPDATE_PATCH_MARKER in src:
            return
        if _UPDATE_ANCHOR not in src:
            print(
                "[-] LoRA Manager: update-check anchor not found; "
                "skipping update-disable patch (upstream layout changed).",
                file=sys.stderr,
            )
            return
        patched = src.replace(_UPDATE_ANCHOR, _UPDATE_INJECT, 1)
        target.write_text(patched, encoding="utf-8")
        print(
            f"[-] LoRA Manager: disabled update check in {_UPDATE_ROUTE_REL}",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"[-] LoRA Manager: failed to apply update-check patch: {e}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Forge Neo rewire — convert ComfyUI interactions to Forge Neo
# ---------------------------------------------------------------------------
# 1. In-iframe bridge (forge_bridge.js): intercept the LoRA card "send to
#    ComfyUI" action (the .fa-paper-plane icon), build the <lora:name:weight>
#    syntax from the card's dataset (verified: ModelCard.js uses
#    card.dataset.file_name/folder/usage_tips; handleSendToWorkflow at :166;
#    syntax format uiHelpers.js buildLoraSyntax), and postMessage it to the
#    parent Forge window — where javascript/lora_manager.js inserts it into the
#    active tab's positive prompt (cross-origin: postMessage is the only path).
#    Also rebrands "ComfyUI" → "Forge Neo" labels as a DOM fallback.
# 2. base.html gets a <script> tag for the bridge (served via /loras_static).
# 3. locale JSON: "ComfyUI" → "Forge Neo", send-action labels → "Add LoRA".
# All marker-guarded / value-compared + re-applied on re-clone.

_BRIDGE_REL = "static/forge_bridge.js"
_BRIDGE_MARKER = "/* forge_bridge.js (forge_sam3) */"
_BRIDGE_JS = r'''/* forge_bridge.js (forge_sam3) */
/* Runs INSIDE the vendored LoRA Manager iframe (cross-origin from Forge).
 * Converts the "send to ComfyUI" card action into "Add LoRA" → posts the
 * <lora:...> syntax to the parent Forge window, which inserts it into the
 * active tab's positive prompt. Also rebrands ComfyUI -> Forge Neo. */
(function () {
    "use strict";
    function isLoras() {
        var p = document.body && document.body.getAttribute("data-page");
        return p === "loras";
    }
    function syntaxFormat() {
        try {
            var s = window.state && window.state.global && window.state.global.settings;
            return (s && s.lora_syntax_format) || "standard";
        } catch (e) { return "standard"; }
    }
    function num(v) {
        if (v === null || v === undefined || v === "") return null;
        var n = typeof v === "number" ? v : parseFloat(v);
        return isNaN(n) ? null : n;
    }
    function buildSyntax(fileName, tips) {
        tips = tips || {};
        var strength = num(tips.strength); if (strength === null) strength = 1;
        var clipSrc = (tips.clip_strength !== undefined && tips.clip_strength !== null)
            ? tips.clip_strength : tips.clipStrength;
        var clip = num(clipSrc);
        var name = (syntaxFormat() === "legacy") ? fileName.split("/").pop() : fileName;
        if (clip !== null) return "<lora:" + name + ":" + strength + ":" + clip + ">";
        return "<lora:" + name + ":" + strength + ">";
    }
    function syntaxFromCard(card) {
        var tips = {};
        try { tips = JSON.parse(card.dataset.usage_tips || "{}"); } catch (e) { tips = {}; }
        var folder = card.dataset.folder || "";
        var file = card.dataset.file_name || "";
        if (!file) return null;
        var loraName = folder ? (folder + "/" + file) : file;
        return buildSyntax(loraName, tips);
    }
    /* Capture-phase intercept — runs BEFORE the vendor's bubble-phase
     * delegation on #modelGrid, so stopImmediatePropagation blocks the
     * backend send (which fails in standalone) and we postMessage instead. */
    document.addEventListener("click", function (e) {
        var icon = e.target && e.target.closest && e.target.closest(".fa-paper-plane");
        if (!icon) return;
        if (!isLoras()) return;
        var card = icon.closest && icon.closest(".model-card");
        if (!card) return;
        var syntax = syntaxFromCard(card);
        if (!syntax) return;
        e.preventDefault();
        e.stopImmediatePropagation();
        try { window.parent.postMessage({ type: "sam3-add-lora", text: syntax }, "*"); } catch (err) {}
        try {
            icon.style.transition = "transform .15s";
            icon.style.transform = "scale(1.4)";
            setTimeout(function () { icon.style.transform = ""; }, 180);
        } catch (e2) {}
    }, true);

    /* Track the card a single-card context menu was opened on, so the
     * "send to workflow" menu item knows its target. */
    var _lastCard = null;
    document.addEventListener("contextmenu", function (e) {
        try {
            _lastCard = (e.target && e.target.closest)
                ? e.target.closest(".model-card") : null;
        } catch (err) { _lastCard = null; }
    }, true);

    function postLoras(syntaxes, replace) {
        var texts = [];
        for (var i = 0; i < syntaxes.length; i++) {
            if (syntaxes[i]) texts.push(syntaxes[i]);
        }
        if (!texts.length) return;
        try {
            window.parent.postMessage(
                { type: "sam3-add-lora", text: texts.join(", "), replace: !!replace },
                "*"
            );
        } catch (err) {}
    }

    /* Context-menu "send to workflow" — SINGLE (#loraContextMenu:
     * data-action sendappend/sendreplace) AND BULK multi-select
     * (#bulkContextMenu: send-to-workflow-append/replace over every
     * .model-card.selected). The vendor would send these to ComfyUI (a no-op
     * in standalone); intercept in capture phase, build the syntax for the
     * target card(s), and post to Forge instead. */
    document.addEventListener("click", function (e) {
        var item = e.target && e.target.closest
            && e.target.closest(".context-menu-item[data-action]");
        if (!item || !isLoras()) return;
        var action = (item.getAttribute("data-action") || "").toLowerCase();
        if (action.indexOf("send") === -1) return; /* skip copy-all etc. */
        var bulk = !!(item.closest && item.closest("#bulkContextMenu"));
        var replace = action.indexOf("replace") !== -1;
        var cards = [];
        if (bulk) {
            cards = Array.prototype.slice.call(
                document.querySelectorAll(".model-card.selected")
            );
        } else if (_lastCard) {
            cards = [_lastCard];
        }
        if (!cards.length) return;
        var syntaxes = [];
        for (var i = 0; i < cards.length; i++) {
            var s = syntaxFromCard(cards[i]);
            if (s) syntaxes.push(s);
        }
        if (!syntaxes.length) return;
        e.preventDefault();
        e.stopImmediatePropagation();
        postLoras(syntaxes, replace);
    }, true);

    /* Label rebrand fallback (locale patch is primary). */
    function relabel(root) {
        if (!root || !root.querySelectorAll) return;
        try {
            root.querySelectorAll("i.fa-paper-plane[title]").forEach(function (el) {
                var t = el.getAttribute("title") || "";
                if (/comfy|workflow|send/i.test(t)) el.setAttribute("title", "Add LoRA");
            });
            root.querySelectorAll("[title]").forEach(function (el) {
                var t = el.getAttribute("title");
                if (t && t.indexOf("ComfyUI") !== -1) el.setAttribute("title", t.replace(/ComfyUI/g, "Forge Neo"));
            });
        } catch (e) {}
    }
    function walkTextOnce() {
        try {
            var w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
            var n, hits = [];
            while ((n = w.nextNode())) { if (n.nodeValue && n.nodeValue.indexOf("ComfyUI") !== -1) hits.push(n); }
            hits.forEach(function (t) { t.nodeValue = t.nodeValue.replace(/ComfyUI/g, "Forge Neo"); });
        } catch (e) {}
    }
    function init() {
        relabel(document); walkTextOnce();
        try {
            var obs = new MutationObserver(function (muts) {
                muts.forEach(function (m) {
                    if (m.addedNodes) m.addedNodes.forEach(function (nd) { if (nd.nodeType === 1) relabel(nd); });
                });
            });
            obs.observe(document.body, { childList: true, subtree: true });
        } catch (e) {}
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
})();
'''


def write_forge_bridge() -> None:
    """Write the in-iframe bridge into vendor static (idempotent by content)."""
    if not LM_VENDOR.is_dir():
        return
    target = LM_VENDOR / _BRIDGE_REL
    try:
        if target.is_file():
            existing = target.read_text(encoding="utf-8", errors="replace")
            if existing.strip() == _BRIDGE_JS.strip():
                return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_BRIDGE_JS, encoding="utf-8")
        print(f"[-] LoRA Manager: wrote {_BRIDGE_REL}", file=sys.stderr)
    except Exception as e:
        print(f"[-] LoRA Manager: failed to write bridge: {e}", file=sys.stderr)


_TEMPLATE_REL = "templates/base.html"
_TPL_MARKER = "<!-- forge_sam3 bridge (auto-applied) -->"
_TPL_ANCHOR = "    {% block head_scripts %}{% endblock %}\n"
_TPL_INJECT = (
    "    " + _TPL_MARKER + "\n"
    "    <script src=\"/loras_static/forge_bridge.js?v={{ version }}\"></script>\n"
    + _TPL_ANCHOR
)


def inject_bridge_script_tag() -> None:
    """Splice the bridge <script> just before {% block head_scripts %} in
    base.html (classic script so it runs in the iframe immediately)."""
    if not LM_VENDOR.is_dir():
        return
    target = LM_VENDOR / _TEMPLATE_REL
    try:
        if not target.is_file():
            return
        src = target.read_text(encoding="utf-8", errors="replace")
        if _TPL_MARKER in src:
            return
        if _TPL_ANCHOR not in src:
            print(
                "[-] LoRA Manager: base.html anchor not found; skipping bridge "
                "<script> injection (upstream layout changed).",
                file=sys.stderr,
            )
            return
        target.write_text(src.replace(_TPL_ANCHOR, _TPL_INJECT, 1), encoding="utf-8")
        print(
            f"[-] LoRA Manager: injected bridge <script> into {_TEMPLATE_REL}",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"[-] LoRA Manager: failed to inject bridge script: {e}",
            file=sys.stderr,
        )


# Locale rebrand: send-action labels → "Add LoRA"; blanket ComfyUI → Forge Neo.
_LOCALE_ADD_LORA_KEYS = (
    "modelCard.actions.sendToWorkflow",
    "modelCard.actions.sendCheckpointToWorkflow",
    "modelCard.actions.sendEmbeddingToWorkflow",
    "modals.model.actions.sendToWorkflow",
    "modals.model.actions.sendToWorkflowText",
    "loras.contextMenu.sendToWorkflowAppend",
    "loras.contextMenu.sendToWorkflowReplace",
    # Bulk multi-select submenu (missing keys are ignored by _set_json_path).
    "loras.bulkOperations.sendToWorkflow",
    "loras.bulkOperations.sendToWorkflowAppend",
    "loras.bulkOperations.sendToWorkflowReplace",
)


def _set_json_path(obj: dict, dotted: str, value: str) -> bool:
    cur = obj
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return False
        cur = cur[p]
    last = parts[-1]
    if isinstance(cur, dict) and last in cur and cur[last] != value:
        cur[last] = value
        return True
    return False


def apply_locale_forge_neo() -> None:
    """Rebrand ComfyUI→Forge Neo and send-action labels→Add LoRA across every
    locale file. Idempotent by value comparison; re-applied on re-clone."""
    if not LM_VENDOR.is_dir():
        return
    loc_dir = LM_VENDOR / "locales"
    if not loc_dir.is_dir():
        return
    for jf in loc_dir.glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        changed = False

        for k in _LOCALE_ADD_LORA_KEYS:
            if _set_json_path(data, k, "Add LoRA"):
                changed = True

        def walk(node):
            nonlocal changed
            if isinstance(node, dict):
                for kk, vv in list(node.items()):
                    if isinstance(vv, str):
                        if "ComfyUI" in vv:
                            node[kk] = vv.replace("ComfyUI", "Forge Neo")
                            changed = True
                    else:
                        walk(vv)
            elif isinstance(node, list):
                for i, vv in enumerate(node):
                    if isinstance(vv, str):
                        if "ComfyUI" in vv:
                            node[i] = vv.replace("ComfyUI", "Forge Neo")
                            changed = True
                    else:
                        walk(vv)

        walk(data)
        if changed:
            try:
                jf.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"[-] LoRA Manager: rebranded locale {jf.name}", file=sys.stderr)
            except Exception as e:
                print(f"[-] LoRA Manager: failed locale {jf.name}: {e}", file=sys.stderr)


# Bug A (Usage Tips X) + Bug B (Notes placeholder) — minimal source patches.
_BUGA_FILE = "static/js/components/shared/PresetTags.js"
_BUGA_OLD = "    const currentPresets = parsePresets(loraCard.dataset.usage_tips);\n"
_BUGA_NEW = (
    "    /* forge_sam3: loraCard is null when the card is virtual-scrolled out;\n"
    "       fall back to the modal's stored usage_tips so removal still works. */\n"
    "    const currentPresets = parsePresets(\n"
    "        (loraCard && loraCard.dataset.usage_tips) ||\n"
    "        (document.getElementById('modelModal') && document.getElementById('modelModal').dataset.usageTips) ||\n"
    "        '{}');\n"
)

_BUGB_FILE = "static/js/components/shared/ModelModal.js"
# Render: don't inject the localized placeholder as real text; use data-placeholder.
_BUGB_OLD_RENDER = (
    "<div class=\"notes-content\" contenteditable=\"true\" spellcheck=\"false\">"
    "${modelWithFullData.notes || translate('modals.model.metadata.addNotesPlaceholder', {}, 'Add your notes here...')}</div>"
)
_BUGB_NEW_RENDER = (
    "<div class=\"notes-content\" contenteditable=\"true\" spellcheck=\"false\" "
    "data-placeholder=\"${translate('modals.model.metadata.addNotesPlaceholder', {}, 'Add your notes here...')}\">"
    "${modelWithFullData.notes || ''}</div>"
)
# Blur: keep the field truly empty (CSS :empty::before shows the placeholder)
# instead of refilling the hardcoded English string.
_BUGB_OLD_BLUR = "                    this.textContent = 'Add your notes here...';\n"
_BUGB_NEW_BLUR = "                    this.textContent = '';  /* forge_sam3: real placeholder via CSS */\n"


def apply_modal_bug_patches() -> None:
    """Source-patch the two LoRA-modal bugs (idempotent by content compare)."""
    if not LM_VENDOR.is_dir():
        return
    # Bug A
    a = LM_VENDOR / _BUGA_FILE
    try:
        if a.is_file():
            src = a.read_text(encoding="utf-8", errors="replace")
            if "forge_sam3: loraCard is null" not in src and _BUGA_OLD in src:
                a.write_text(src.replace(_BUGA_OLD, _BUGA_NEW, 1), encoding="utf-8")
                print(f"[-] LoRA Manager: patched Bug A in {_BUGA_FILE}", file=sys.stderr)
    except Exception as e:
        print(f"[-] LoRA Manager: Bug A patch failed: {e}", file=sys.stderr)
    # Bug B (render + blur)
    b = LM_VENDOR / _BUGB_FILE
    try:
        if b.is_file():
            src = b.read_text(encoding="utf-8", errors="replace")
            new = src
            if _BUGB_OLD_RENDER in new:
                new = new.replace(_BUGB_OLD_RENDER, _BUGB_NEW_RENDER, 1)
            if _BUGB_OLD_BLUR in new:
                new = new.replace(_BUGB_OLD_BLUR, _BUGB_NEW_BLUR, 1)
            if new != src:
                b.write_text(new, encoding="utf-8")
                print(f"[-] LoRA Manager: patched Bug B in {_BUGB_FILE}", file=sys.stderr)
    except Exception as e:
        print(f"[-] LoRA Manager: Bug B patch failed: {e}", file=sys.stderr)


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
    except urllib.error.HTTPError:
        # A 4xx/5xx page means *something* is listening, but our healthy
        # manager serves 200 at /loras. Treating any error page as "up" made
        # an unrelated service on this port read as the LoRA Manager and point
        # the iframe at a foreign server. Require a real success response.
        return False
    except Exception:
        return False


def is_running(port: int | None = None) -> bool:
    p = port if port is not None else (_proc_port or DEFAULT_PORT)
    return _health_ok(p)


def manager_url(port: int) -> str:
    return f"http://{DEFAULT_HOST}:{port}/loras"


def get_or_spawn(port: int = DEFAULT_PORT, wait_seconds: float = 3.0) -> dict[str, Any]:
    """Lazy entry point called by the UI bridge. Returns
    ``{"url": ..., "port": ..., "status": "running"|"spawned"|"starting"|"error",
       "message": ...}``.

    - ``running``  — a server already answered on ``port`` (reused).
    - ``spawned``  — we started it and it became healthy within ``wait_seconds``.
    - ``starting`` — we started it but it's still booting (the JS side then
      polls the URL until it answers).
    - ``error``    — vendor missing or the process died on launch.

    NON-BLOCKING by design: the first run of the manager scans + hashes the
    whole LoRA library (observed ~266 s for 1487 models) and aiohttp does not
    open the port until that on_startup scan finishes. We therefore do NOT
    block the Gradio event waiting for health — we kick the process off,
    give it a short grace poll, then hand "starting" back so the browser can
    poll the URL itself and show scan progress.
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

        # Already-booting guard: a live process that simply isn't answering
        # health yet is the *expected* state during the multi-minute first-run
        # library scan (aiohttp doesn't open the port until the scan finishes).
        # Without this, a second Manage-tab open would Popen another server on
        # the same port — the duplicate fails to bind and exits, and its early
        # exit can then be misreported as an error over the healthy first one.
        if _proc is not None and _proc.poll() is None:
            _proc_port = port
            return {
                "url": manager_url(port),
                "port": port,
                "status": "starting",
                "message": "server is still booting (first-run library scan) — "
                "reusing the process already starting on this port",
            }

        try:
            ensure_settings_json()
        except Exception:
            pass
        try:
            apply_css_overrides()
        except Exception:
            pass
        try:
            apply_update_check_patch()
        except Exception:
            pass
        for _patch in (
            write_forge_bridge,
            inject_bridge_script_tag,
            apply_locale_forge_neo,
            apply_modal_bug_patches,
        ):
            try:
                _patch()
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
        finally:
            # The child holds its own dup of the fd; the parent never writes to
            # it, so release our copy immediately (both on success and on the
            # failure path) instead of leaking a handle every crash/respawn.
            if log_fh is not subprocess.DEVNULL:
                try:
                    log_fh.close()
                except Exception:
                    pass

    # Short grace poll OUTSIDE the lock — only to catch the fast case where
    # the cache is already serialized and the server comes up in a second or
    # two. If it's still scanning, return "starting" immediately; the JS side
    # polls the URL until it answers (handles arbitrarily long first-run scans).
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

    # Still booting (most likely a first-run library scan). Hand back the URL
    # with a "starting" status; the browser polls it.
    return {
        "url": manager_url(port),
        "port": port,
        "status": "starting",
        "message": "server is booting (first run scans the LoRA library — this "
        "can take several minutes for large collections)",
    }


def _read_log_tail(n: int = 20) -> str:
    try:
        lines = LM_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return "(no log)"


# ---------------------------------------------------------------------------
# HTTP config/spawn endpoints
# ---------------------------------------------------------------------------
# The Live Workspace shell is a lightweight page with no Gradio doc, so it
# cannot click the hidden bridge buttons the normal txt2img UI uses. These
# same-origin JSON routes let the shell (or any page) query config and lazily
# spawn the server directly. They wrap the exact same get_or_spawn() lifecycle.

LORA_CONFIG_PATH = "/sam3-lora/config"
LORA_SPAWN_PATH = "/sam3-lora/spawn"
_OPT_TAB_MODE = "sam3_lora_manager_tab_mode"
_OPT_PORT = "sam3_lora_manager_port"
_TAB_MODE_REPLACE = "Replace LoRA tab"


def _read_lora_opts() -> tuple[bool, int]:
    """(replace_mode, port) from Settings — defensive so it works headless."""
    replace = False
    port = DEFAULT_PORT
    try:
        from modules import shared

        mode = getattr(shared.opts, _OPT_TAB_MODE, None)
        replace = str(mode) == _TAB_MODE_REPLACE
        port = int(getattr(shared.opts, _OPT_PORT, DEFAULT_PORT) or DEFAULT_PORT)
    except Exception:
        pass
    return replace, port


def lora_config_data() -> dict[str, Any]:
    replace, port = _read_lora_opts()
    return {
        "available": lora_manager_available(),
        "replace": replace,
        "port": port,
    }


def lora_spawn_data() -> dict[str, Any]:
    _replace, port = _read_lora_opts()
    try:
        return get_or_spawn(port)
    except Exception as e:  # pragma: no cover - defensive
        import traceback

        traceback.print_exc(file=sys.stderr)
        return {"url": "", "port": port, "status": "error", "message": str(e)}


def register_lora_routes(app: Any) -> bool:
    """Register the config/spawn JSON routes once. Returns True if newly added
    (idempotent — Forge re-fires app-start after Reload UI)."""
    existing = {getattr(r, "path", None) for r in getattr(app, "routes", ())}
    if LORA_CONFIG_PATH in existing and LORA_SPAWN_PATH in existing:
        return False

    from fastapi.responses import JSONResponse

    _no_store = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}

    async def _config_route() -> JSONResponse:
        return JSONResponse(lora_config_data(), headers=_no_store)

    async def _spawn_route() -> JSONResponse:
        return JSONResponse(lora_spawn_data(), headers=_no_store)

    if LORA_CONFIG_PATH not in existing:
        app.add_api_route(
            LORA_CONFIG_PATH, _config_route, methods=["GET"],
            include_in_schema=False, name="sam3-lora-config",
        )
    if LORA_SPAWN_PATH not in existing:
        app.add_api_route(
            LORA_SPAWN_PATH, _spawn_route, methods=["GET"],
            include_in_schema=False, name="sam3-lora-spawn",
        )
    return True


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
