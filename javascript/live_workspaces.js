/* SAM3 Live Workspaces — three persistent WebUI documents in one browser tab. */
(function () {
    "use strict";

    if (window.__sam3LiveWorkspacesLoaded) return;
    window.__sam3LiveWorkspacesLoaded = true;

    var query = new URLSearchParams(window.location.search);
    var frameSlot = query.get("__sam3_live_workspace");
    var liveDisabled = query.get("sam3_live") === "off";
    var SLOT_IDS = ["1", "2", "3"];
    var SHELL_KEY = "sam-extra.live-workspaces.shell.v1";

    function app() {
        return typeof gradioApp === "function" ? gradioApp() : document;
    }

    function waitFor(predicate, timeout) {
        return new Promise(function (resolve) {
            var deadline = Date.now() + (timeout || 15000);
            var timer = setInterval(function () {
                var value = null;
                try { value = predicate(); } catch (e) {}
                if (value || Date.now() >= deadline) {
                    clearInterval(timer);
                    resolve(value);
                }
            }, 100);
        });
    }

    function readShellState() {
        var fallback = { active: "1", names: { "1": "1", "2": "2", "3": "3" } };
        try {
            var parsed = JSON.parse(window.localStorage.getItem(SHELL_KEY) || "null");
            if (!parsed || SLOT_IDS.indexOf(String(parsed.active)) < 0) return fallback;
            SLOT_IDS.forEach(function (slot) {
                var name = parsed.names && String(parsed.names[slot] || "").trim();
                fallback.names[slot] = name ? name.slice(0, 40) : slot;
            });
            fallback.active = String(parsed.active);
        } catch (e) {}
        return fallback;
    }

    function writeShellState(state) {
        try { window.localStorage.setItem(SHELL_KEY, JSON.stringify(state)); } catch (e) {}
    }

    function childUrl(slot) {
        var url = new URL(window.location.href);
        url.searchParams.delete("sam3_live");
        url.searchParams.set("__sam3_live_workspace", slot);
        url.hash = "";
        return url.toString();
    }

    function monitorServerRestart() {
        var consecutiveFailures = 0;
        var reloading = false;
        setInterval(function () {
            var ping = new URL("/sdapi/v1/progress", window.location.origin);
            ping.searchParams.set("skip_current_image", "true");
            ping.searchParams.set("_sam3_live_ping", String(Date.now()));
            window.fetch(ping.toString(), { cache: "no-store" }).then(function (response) {
                if (!response.ok) throw new Error("WebUI health check " + response.status);
                if (consecutiveFailures >= 2 && !reloading) {
                    // All three child documents still hold the old Gradio
                    // component ids after a server restart. Reload the shell
                    // once on recovery so they reconnect cleanly and their
                    // startup path clears the comparison galleries.
                    reloading = true;
                    window.location.reload();
                    return;
                }
                consecutiveFailures = 0;
            }).catch(function () {
                consecutiveFailures = Math.min(3, consecutiveFailures + 1);
            });
        }, 5000);
    }

    function mountShell() {
        if (document.querySelector("#sam3_live_workspace_shell")) return;
        var state = readShellState();
        var shell = document.createElement("section");
        shell.id = "sam3_live_workspace_shell";
        shell.innerHTML = [
            '<header class="sam3-live-header">',
            '  <strong>Live Workspaces</strong>',
            '  <nav class="sam3-live-tabs" aria-label="Live workspaces"></nav>',
            '  <span class="sam3-live-note">각 화면은 독립 상태 · Generate는 현재 화면만 실행</span>',
            '  <button type="button" data-sam3-live-legacy title="기존 값 복원 Workspace UI로 전환">기본 UI</button>',
            '</header>',
            '<div class="sam3-live-frames"></div>'
        ].join("");
        document.body.appendChild(shell);

        var tabs = shell.querySelector(".sam3-live-tabs");
        var frames = shell.querySelector(".sam3-live-frames");

        function activate(slot) {
            state.active = slot;
            writeShellState(state);
            Array.prototype.forEach.call(tabs.querySelectorAll("button[data-slot]"), function (button) {
                var active = button.getAttribute("data-slot") === slot;
                button.classList.toggle("active", active);
                button.setAttribute("aria-selected", active ? "true" : "false");
            });
            Array.prototype.forEach.call(frames.querySelectorAll("iframe[data-slot]"), function (iframe) {
                var active = iframe.getAttribute("data-slot") === slot;
                if (active && !iframe.getAttribute("src")) {
                    iframe.src = iframe.getAttribute("data-src");
                }
                iframe.hidden = !active;
                iframe.toggleAttribute("inert", !active);
                iframe.style.pointerEvents = active ? "auto" : "none";
            });
        }

        SLOT_IDS.forEach(function (slot) {
            var button = document.createElement("button");
            button.type = "button";
            button.setAttribute("data-slot", slot);
            button.setAttribute("role", "tab");
            button.textContent = state.names[slot];
            button.title = "클릭: 전환 · 더블클릭: 이름 변경";
            button.addEventListener("click", function () { activate(slot); });
            button.addEventListener("dblclick", function () {
                var next = window.prompt("Workspace 이름", state.names[slot]);
                if (next === null) return;
                next = String(next).replace(/\s+/g, " ").trim().slice(0, 40);
                if (!next) return;
                state.names[slot] = next;
                button.textContent = next;
                writeShellState(state);
            });
            tabs.appendChild(button);

            var iframe = document.createElement("iframe");
            iframe.setAttribute("data-slot", slot);
            iframe.setAttribute("title", "Workspace " + state.names[slot]);
            iframe.setAttribute("loading", "eager");
            iframe.setAttribute("data-src", childUrl(slot));
            frames.appendChild(iframe);
        });

        shell.querySelector("[data-sam3-live-legacy]").addEventListener("click", function () {
            var url = new URL(window.location.href);
            url.searchParams.set("sam3_live", "off");
            window.location.href = url.toString();
        });
        activate(state.active);
        // Loading three full Gradio documents at exactly the same instant can
        // freeze the browser main thread. Warm inactive workspaces in sequence;
        // clicking one loads it immediately if its turn has not arrived yet.
        SLOT_IDS.filter(function (slot) { return slot !== state.active; })
            .forEach(function (slot, index) {
                setTimeout(function () {
                    var iframe = frames.querySelector('iframe[data-slot="' + slot + '"]');
                    if (iframe && !iframe.getAttribute("src")) iframe.src = iframe.getAttribute("data-src");
                }, 3500 * (index + 1));
            });
        document.documentElement.classList.add("sam3-live-shell-active");
        monitorServerRestart();
    }

    function smallestSection(element, boundary, reject) {
        var candidate = element;
        for (var node = element; node && node !== boundary; node = node.parentElement) {
            if (reject && node.contains(reject)) break;
            if (node.classList && (
                node.classList.contains("gradio-group")
                || node.classList.contains("gradio-column")
                || node.classList.contains("form")
            )) candidate = node;
        }
        return candidate;
    }

    function directChildContaining(container, element) {
        var node = element;
        while (node && node.parentElement && node.parentElement !== container) {
            node = node.parentElement;
        }
        return node && node.parentElement === container ? node : null;
    }

    function isBuiltInScriptPanel(group) {
        if (!group || !group.querySelector) return false;
        // Forge's built-in selectable Scripts. Everything else registered by
        // an extension stays under Parameters, even when it appears after the
        // Script selector in Gradio's shared script container.
        return [
            '[id^="script_txt2img_prompt_matrix_"]',
            '[id^="script_txt2img_prompts_from_file_or_textbox_"]',
            '[id^="script_txt2img_xyz_plot_"]'
        ].some(function (selector) {
            return !!group.querySelector(selector);
        });
    }

    function splitScriptContainer(container, scriptList) {
        var content = directChildContaining(container, scriptList);
        // Forge wraps every always-on extension and the selectable Script UI
        // in one .styler. Work with that actual content node, not the first
        // duplicate #script_list in the document (which belongs to img2img).
        if (!content || content === scriptList) {
            content = container.firstElementChild || container;
        }
        var selectorGroup = directChildContaining(content, scriptList);
        if (!selectorGroup) return scriptList;

        var scriptCore = document.createElement("div");
        scriptCore.className = "sam3-live-script-core";
        var node = selectorGroup;
        while (node) {
            var next = node.nextElementSibling;
            // The middle column is intentionally limited to Forge's Script
            // selector and built-in panels. Selectable extension panels such
            // as Vectorscope HDR, SAM3 utilities, and LoRA Block Weight remain
            // in the original container under Parameters.
            if (node === selectorGroup || isBuiltInScriptPanel(node)) {
                scriptCore.appendChild(node);
            }
            node = next;
        }
        return scriptCore;
    }

    function isolateLayoutBranch(host, stop) {
        for (var node = host; node && node !== stop; node = node.parentElement) {
            var parent = node.parentElement;
            if (!parent) break;
            Array.prototype.forEach.call(parent.children, function (sibling) {
                if (sibling === node) return;
                // Toasts are useful while generating and do not occupy layout.
                if (sibling.classList && sibling.classList.contains("toast-wrap")) return;
                if (sibling.classList) sibling.classList.add("sam3-live-original-branch");
            });
        }
    }

    async function mountChildLayout() {
        document.documentElement.classList.add("sam3-live-frame-active");
        var ready = await waitFor(function () {
            var txt2imgScripts = app().querySelector("#txt2img_script_container");
            return app().querySelector("#txt2img_toprow")
                && app().querySelector("#txt2img_settings")
                && txt2imgScripts
                && txt2imgScripts.querySelector("#script_list")
                && app().querySelector("#txt2img_gallery");
        }, 20000);
        if (!ready || document.querySelector("#sam3_live_child_layout")) return;

        var top = app().querySelector("#txt2img_toprow");
        var settings = app().querySelector("#txt2img_settings");
        var scriptContainer = app().querySelector("#txt2img_script_container");
        var scripts = scriptContainer.querySelector("#script_list");
        var gallery = app().querySelector("#txt2img_gallery");
        var boundary = app().querySelector("#txt2img_extra_tabs") || settings.parentElement;
        var scriptSection = splitScriptContainer(scriptContainer, scripts);
        var gallerySection = smallestSection(gallery, boundary, settings);

        var appRoot = app();
        var sourceRoot = null;
        if (appRoot && appRoot.classList
                && appRoot.classList.contains("gradio-container")) {
            sourceRoot = appRoot;
        } else if (appRoot && appRoot.querySelector) {
            sourceRoot = appRoot.querySelector(".gradio-container");
        }
        if (!sourceRoot) sourceRoot = document.querySelector(".gradio-container");

        var layout = document.createElement("main");
        layout.id = "sam3_live_child_layout";
        layout.innerHTML = [
            '<section class="sam3-live-prompt"></section>',
            '<section class="sam3-live-columns">',
            ' <div class="sam3-live-column" data-column="parameters"><h2>Parameters</h2><div></div></div>',
            ' <div class="sam3-live-column" data-column="scripts"><h2>Scripts</h2><div></div></div>',
            ' <div class="sam3-live-column" data-column="gallery"><h2>Gallery</h2><div></div></div>',
            '</section>'
        ].join("");
        // Keep the controls inside the original txt2img tab. Forge and several
        // extensions scope their layout CSS to #tab_txt2img; moving controls
        // outside that branch makes prompt helpers, dropdowns, and sliders
        // collapse or overlap.
        var layoutHost = app().querySelector("#tab_txt2img");
        if (sourceRoot && layoutHost && layoutHost.appendChild) {
            sourceRoot.classList.add("sam3-live-source-root");
            Array.prototype.forEach.call(layoutHost.children, function (child) {
                if (child.classList) child.classList.add("sam3-live-original-root");
            });
            isolateLayoutBranch(layoutHost, sourceRoot);
            layoutHost.appendChild(layout);
        } else if (sourceRoot && sourceRoot.appendChild) {
            sourceRoot.classList.add("sam3-live-source-root");
            Array.prototype.forEach.call(sourceRoot.children, function (child) {
                if (child.classList) child.classList.add("sam3-live-original-root");
            });
            sourceRoot.appendChild(layout);
        } else {
            document.body.appendChild(layout);
        }
        layout.querySelector(".sam3-live-prompt").appendChild(top);

        var parameterTarget = layout.querySelector('[data-column="parameters"] > div');
        parameterTarget.appendChild(settings);
        // Always-on script extensions retain their original order directly
        // below Parameters. Only the Script selector and its selected panel
        // (for example X/Y/Z plot) live in the middle column.
        parameterTarget.appendChild(scriptContainer);
        layout.querySelector('[data-column="scripts"] > div').appendChild(scriptSection);
        layout.querySelector('[data-column="gallery"] > div').appendChild(gallerySection);
    }

    function start() {
        if (frameSlot) mountChildLayout();
        else if (!liveDisabled) mountShell();
    }

    if (typeof onUiLoaded === "function") onUiLoaded(function () { setTimeout(start, 300); });
    else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
    else start();
})();
