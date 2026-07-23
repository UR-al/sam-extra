/* SAM3 Live Workspaces — dynamic persistent WebUI documents in one browser tab. */
(function () {
    "use strict";

    if (window.__sam3LiveWorkspacesLoaded) return;
    window.__sam3LiveWorkspacesLoaded = true;

    var query = new URLSearchParams(window.location.search);
    var frameSlot = query.get("__sam3_live_workspace");
    var liveDisabled = query.get("sam3_live") === "off";
    var standaloneShell = document.documentElement.hasAttribute(
        "data-sam3-standalone-live-shell"
    );
    var DEFAULT_SLOT_IDS = ["1", "2", "3"];
    var SHELL_KEY = "sam-extra.live-workspaces.shell.v1";
    var READY_MESSAGE = "sam3-live-workspace-ready-v1";
    var CONTROL_MESSAGE = "sam3-live-workspace-control-v1";
    var CONTROL_REPLY = "sam3-live-workspace-control-reply-v1";
    var MAX_IMPORT_BYTES = 4 * 1024 * 1024;

    var standaloneRedirect = null;
    if (!frameSlot && !liveDisabled && !standaloneShell
            && window.location.pathname !== "/sam3-live") {
        // The Forge root is a complete Gradio document. It used to finish
        // building that unused UI and then create another complete document
        // for Workspace 1. Move to the extension-owned lightweight parent as
        // soon as this script is evaluated; child workspaces still load "/".
        // Verify the Python route first so a browser-only refresh against an
        // older, not-yet-restarted WebUI falls back to the original shell
        // instead of landing on a temporary 404.
        var shellUrl = new URL("/sam3-live", window.location.origin);
        shellUrl.search = window.location.search;
        standaloneRedirect = window.fetch(shellUrl.toString(), {
            method: "GET",
            credentials: "same-origin",
            cache: "no-store"
        }).then(function (response) {
            if (!response.ok) return false;
            window.location.replace(shellUrl.toString());
            return true;
        }).catch(function () {
            return false;
        });
    }

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

    function readLegacyShellState() {
        var fallback = {
            active: "1",
            names: { "1": "1", "2": "2", "3": "3" }
        };
        try {
            var parsed = JSON.parse(window.localStorage.getItem(SHELL_KEY) || "null");
            if (!parsed) return fallback;
            DEFAULT_SLOT_IDS.forEach(function (slot) {
                var name = parsed.names && String(parsed.names[slot] || "").trim();
                fallback.names[slot] = name ? name.slice(0, 40) : slot;
            });
            if (parsed.active) fallback.active = String(parsed.active);
        } catch (e) {}
        return fallback;
    }

    function writeShellState(state) {
        try {
            window.localStorage.setItem(SHELL_KEY, JSON.stringify({
                active: state.active,
                names: state.names
            }));
        } catch (e) {}
    }

    function childUrl(slot) {
        var url = standaloneShell
            ? new URL("/", window.location.origin)
            : new URL(window.location.href);
        if (standaloneShell) url.search = window.location.search;
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
                    // Loaded child documents still hold the old Gradio
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

    async function mountShell() {
        if (document.querySelector("#sam3_live_workspace_shell")) return;
        var manager = await waitFor(function () {
            var candidate = window.__sam3WorkspaceManager;
            return candidate && candidate.storage ? candidate : null;
        }, 10000);
        if (!manager) {
            console.error("[SAM3 Live Workspaces] storage manager unavailable");
            return;
        }
        var storage = manager.storage;
        var listing = storage.list();
        var legacy = readLegacyShellState();

        // v0.11.0 stored Live-only names separately. Migrate those names once
        // when the shared Workspace store still contains the numeric fallback.
        DEFAULT_SLOT_IDS.forEach(function (slot) {
            if (listing.ids.indexOf(slot) === -1) return;
            var legacyName = legacy.names[slot];
            if (legacyName && legacyName !== slot && listing.names[slot] === slot) {
                storage.rename(slot, legacyName);
            }
        });
        listing = storage.list();
        var slotIds = listing.ids.slice();
        var state = {
            active: slotIds.indexOf(legacy.active) !== -1
                ? legacy.active : (slotIds[0] || "1"),
            names: listing.names
        };
        writeShellState(state);

        var shell = document.createElement("section");
        var shellStartedAt = Date.now();
        shell.id = "sam3_live_workspace_shell";
        shell.setAttribute("data-sam3-live-started-at", String(shellStartedAt));
        shell.innerHTML = [
            '<header class="sam3-live-header">',
            '  <button type="button" class="sam3-live-brand" data-sam3-live-add ',
            '    aria-label="Workspace 추가" title="현재 설정을 복사한 Workspace 추가">',
            '    <strong>Live Workspaces</strong><span aria-hidden="true">＋</span>',
            '  </button>',
            '  <nav class="sam3-live-tabs" aria-label="Live workspaces"></nav>',
            '  <span class="sam3-live-status" data-sam3-live-status aria-live="polite"></span>',
            '  <span class="sam3-live-note">현재 화면 우선 · 나머지는 순차 준비 · 각 화면은 독립 상태</span>',
            '  <details class="sam3-live-menu" data-sam3-live-menu>',
            '    <summary aria-label="Workspace 메뉴" title="Workspace 메뉴">⋯</summary>',
            '    <div class="sam3-live-menu-panel">',
            '      <label><span>현재 Workspace 이름</span>',
            '        <input type="text" maxlength="40" data-sam3-live-name></label>',
            '      <button type="button" data-sam3-live-rename>이름 저장</button>',
            '      <button type="button" class="sam3-live-delete" ',
            '        data-sam3-live-delete>현재 Workspace 삭제</button>',
            '      <button type="button" data-sam3-live-export>내보내기</button>',
            '      <label class="sam3-live-import">불러오기',
            '        <input type="file" accept="application/json,.json" data-sam3-live-import></label>',
            '    </div>',
            '  </details>',
            '  <button type="button" data-sam3-live-legacy title="기존 값 복원 Workspace UI로 전환">기본 UI</button>',
            '</header>',
            '<div class="sam3-live-frames">',
            '  <div class="sam3-live-loading" role="status" aria-live="polite">',
            '    <span class="sam3-live-spinner" aria-hidden="true"></span>',
            '    <span data-sam3-live-loading-text>Workspace 준비 중…</span>',
            '  </div>',
            '</div>'
        ].join("");
        document.body.appendChild(shell);

        var tabs = shell.querySelector(".sam3-live-tabs");
        var frames = shell.querySelector(".sam3-live-frames");
        var loading = shell.querySelector(".sam3-live-loading");
        var loadingText = shell.querySelector("[data-sam3-live-loading-text]");
        var status = shell.querySelector("[data-sam3-live-status]");
        var menu = shell.querySelector("[data-sam3-live-menu]");
        var nameInput = shell.querySelector("[data-sam3-live-name]");
        var loadQueue = [];
        var loadingSlot = null;
        var readySlots = Object.create(null);
        var pendingRequests = Object.create(null);
        var requestCounter = 0;

        function setShellStatus(message, tone) {
            status.textContent = message || "";
            status.setAttribute("data-tone", tone || "normal");
        }

        function frameFor(slot) {
            return frames.querySelector('iframe[data-slot="' + slot + '"]');
        }

        function buttonFor(slot) {
            return tabs.querySelector('button[data-slot="' + slot + '"]');
        }

        function syncNameEditor() {
            nameInput.value = state.names[state.active] || state.active;
        }

        function refreshLoading() {
            if (loadingSlot && readySlots[loadingSlot]) loadingSlot = null;
            var slot = state.active;
            var iframe = frameFor(slot);
            var ready = !!readySlots[slot];
            loading.hidden = ready;
            if (ready) return;
            var waiting = loadingSlot && loadingSlot !== slot;
            loadingText.textContent = waiting
                ? (state.names[slot] || slot) + " 대기 중 · "
                    + (state.names[loadingSlot] || loadingSlot) + " 준비가 끝나면 시작합니다"
                : (state.names[slot] || slot) + " 준비 중…";
            if (iframe) iframe.setAttribute("aria-busy", "true");
        }

        function loadNext() {
            if (loadingSlot && readySlots[loadingSlot]) loadingSlot = null;
            if (loadingSlot || !loadQueue.length) {
                refreshLoading();
                return;
            }
            var slot = loadQueue.shift();
            var iframe = frameFor(slot);
            if (!iframe || readySlots[slot]) {
                loadNext();
                return;
            }
            loadingSlot = slot;
            iframe.setAttribute("data-load-state", "loading");
            var button = buttonFor(slot);
            if (button) button.setAttribute("data-load-state", "loading");
            refreshLoading();
            if (!iframe.getAttribute("src")) {
                iframe.src = iframe.getAttribute("data-src");
            }
        }

        function queueLoad(slot, urgent) {
            var iframe = frameFor(slot);
            if (!iframe || readySlots[slot] || loadingSlot === slot) {
                refreshLoading();
                return;
            }
            loadQueue = loadQueue.filter(function (queued) { return queued !== slot; });
            if (urgent) loadQueue.unshift(slot);
            else loadQueue.push(slot);
            loadNext();
        }

        function queueDefaultWorkspaces() {
            DEFAULT_SLOT_IDS.forEach(function (slot) {
                if (slotIds.indexOf(slot) === -1 || slot === state.active) return;
                queueLoad(slot, false);
            });
        }

        function hasPendingLoads() {
            if (loadingSlot && !readySlots[loadingSlot]) return true;
            return loadQueue.length > 0;
        }

        function markReady(slot, detail) {
            var iframe = frameFor(slot);
            if (!iframe) return;
            readySlots[slot] = true;
            iframe.setAttribute(
                "data-ready-ms",
                String(Math.max(0, Date.now() - shellStartedAt))
            );
            iframe.setAttribute("data-load-state", detail && detail.degraded ? "degraded" : "ready");
            iframe.removeAttribute("aria-busy");
            var button = buttonFor(slot);
            if (button) {
                button.setAttribute("data-load-state", "ready");
                button.title = "클릭: 전환 · 더블클릭: 이름 변경 · 준비됨";
            }
            if (loadingSlot === slot) loadingSlot = null;
            refreshLoading();
            // Build one full Forge document at a time. This avoids three-way
            // CPU/DOM/API contention while still preparing every default
            // Workspace automatically in the background.
            setTimeout(loadNext, 100);
        }

        function activate(slot) {
            if (slotIds.indexOf(slot) === -1) return;
            state.active = slot;
            writeShellState(state);
            Array.prototype.forEach.call(tabs.querySelectorAll("button[data-slot]"), function (button) {
                var active = button.getAttribute("data-slot") === slot;
                button.classList.toggle("active", active);
                button.setAttribute("aria-selected", active ? "true" : "false");
            });
            Array.prototype.forEach.call(frames.querySelectorAll("iframe[data-slot]"), function (iframe) {
                var active = iframe.getAttribute("data-slot") === slot;
                iframe.hidden = !active;
                iframe.toggleAttribute("inert", !active);
                iframe.style.pointerEvents = active ? "auto" : "none";
            });
            syncNameEditor();
            queueLoad(slot, true);
            refreshLoading();
        }

        function renameSlot(slot, requestedName) {
            var result = storage.rename(slot, requestedName);
            if (!result.ok) {
                setShellStatus(result.error, "error");
                return false;
            }
            state.names[slot] = result.name;
            var button = buttonFor(slot);
            var iframe = frameFor(slot);
            if (button) button.textContent = result.name;
            if (iframe) iframe.title = "Workspace " + result.name;
            writeShellState(state);
            syncNameEditor();
            setShellStatus(result.name + " 이름 저장됨", "saved");
            return true;
        }

        function addSlot(slot, name) {
            if (buttonFor(slot)) return;
            if (slotIds.indexOf(slot) === -1) slotIds.push(slot);
            state.names[slot] = name || slot;

            var button = document.createElement("button");
            button.type = "button";
            button.setAttribute("data-slot", slot);
            button.setAttribute("role", "tab");
            button.textContent = state.names[slot];
            button.title = "클릭: 전환 · 더블클릭: 이름 변경";
            button.addEventListener("click", function () { activate(slot); });
            button.addEventListener("dblclick", function () {
                var next = window.prompt("Workspace 이름", state.names[slot]);
                if (next !== null) renameSlot(slot, next);
            });
            tabs.appendChild(button);

            var iframe = document.createElement("iframe");
            iframe.setAttribute("data-slot", slot);
            iframe.setAttribute("title", "Workspace " + state.names[slot]);
            iframe.setAttribute("loading", "eager");
            iframe.setAttribute("data-src", childUrl(slot));
            iframe.hidden = true;
            iframe.toggleAttribute("inert", true);
            iframe.style.pointerEvents = "none";
            frames.appendChild(iframe);
        }

        function requestChild(slot, action) {
            var iframe = frameFor(slot);
            if (!iframe || !readySlots[slot] || !iframe.contentWindow) {
                return Promise.resolve({ ok: false, error: "Workspace가 아직 준비 중입니다" });
            }
            return new Promise(function (resolve) {
                var requestId = "live-" + Date.now().toString(36) + "-" + (++requestCounter);
                var timer = setTimeout(function () {
                    delete pendingRequests[requestId];
                    resolve({ ok: false, error: "Workspace 응답 시간이 초과됐습니다" });
                }, 65000);
                pendingRequests[requestId] = {
                    slot: slot,
                    resolve: resolve,
                    timer: timer
                };
                iframe.contentWindow.postMessage({
                    type: CONTROL_MESSAGE,
                    requestId: requestId,
                    action: action
                }, window.location.origin);
            });
        }

        async function flushLoadedChildren() {
            for (var i = 0; i < slotIds.length; i++) {
                var slot = slotIds[i];
                if (!readySlots[slot]) continue;
                var result = await requestChild(slot, "flush");
                if (!result.ok) throw new Error(result.error || "Workspace 저장 실패");
            }
        }

        async function createWorkspace() {
            if (!readySlots[state.active]) {
                setShellStatus("현재 Workspace 준비가 끝난 뒤 추가해 주세요", "warning");
                return;
            }
            setShellStatus("현재 설정 저장 중…", "pending");
            var flushed = await requestChild(state.active, "flush");
            if (!flushed.ok) {
                setShellStatus(flushed.error || "현재 Workspace 저장 실패", "error");
                return;
            }
            var created = storage.createFrom(state.active);
            if (!created.ok) {
                setShellStatus(created.error, "error");
                return;
            }
            var refreshed = storage.list();
            state.names = refreshed.names;
            addSlot(created.slot, created.name);
            writeShellState(state);
            activate(created.slot);
            setShellStatus(created.name + " 생성됨 · 현재 설정 복사", "saved");
        }

        async function deleteWorkspace() {
            if (!readySlots[state.active]) {
                setShellStatus("현재 Workspace 준비가 끝난 뒤 삭제해 주세요", "warning");
                return;
            }
            var listing = storage.list();
            if (listing.ids.length <= 1) {
                setShellStatus("마지막 Workspace는 삭제할 수 없습니다", "warning");
                return;
            }

            var deletedSlot = state.active;
            var deletedName = state.names[deletedSlot] || deletedSlot;
            if (!window.confirm(
                "'" + deletedName
                    + "' Workspace의 설정과 로컬 txt2img 갤러리를 삭제할까요?"
            )) return;

            setShellStatus(deletedName + " 삭제 준비 중…", "pending");
            var prepared = await requestChild(deletedSlot, "prepare-delete");
            if (!prepared.ok) {
                setShellStatus(prepared.error || "Workspace 삭제 준비 실패", "error");
                return;
            }

            var result;
            try {
                result = await storage.remove(deletedSlot);
            } catch (e) {
                result = {
                    ok: false,
                    error: String(e && e.message ? e.message : e)
                };
            }
            if (!result.ok) {
                await requestChild(deletedSlot, "cancel-import");
                setShellStatus(result.error || "Workspace 삭제 실패", "error");
                return;
            }

            loadQueue = loadQueue.filter(function (slot) { return slot !== deletedSlot; });
            if (loadingSlot === deletedSlot) loadingSlot = null;
            delete readySlots[deletedSlot];
            var deletedButton = buttonFor(deletedSlot);
            var deletedFrame = frameFor(deletedSlot);
            if (deletedButton) deletedButton.remove();
            if (deletedFrame) deletedFrame.remove();

            slotIds = result.ids.slice();
            state.names = result.names;
            state.active = result.activeSlot;
            writeShellState(state);
            menu.open = false;
            activate(state.active);
            setShellStatus(
                "'" + result.deletedName + "' 삭제됨 · "
                    + (state.names[state.active] || state.active) + "로 전환",
                "saved"
            );
        }

        async function exportWorkspaces() {
            try {
                setShellStatus("모든 Workspace 저장 중…", "pending");
                await flushLoadedChildren();
                var payload = storage.exportPayload();
                var blob = new Blob([JSON.stringify(payload, null, 2)], {
                    type: "application/json"
                });
                var url = URL.createObjectURL(blob);
                var anchor = document.createElement("a");
                anchor.href = url;
                anchor.download = "sam-extra-workspaces-"
                    + new Date().toISOString().replace(/[:.]/g, "-") + ".json";
                document.body.appendChild(anchor);
                anchor.click();
                anchor.remove();
                setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
                setShellStatus("Workspace를 내보냈습니다", "saved");
                menu.open = false;
            } catch (e) {
                console.warn("[SAM3 Live Workspaces] export failed:", e);
                setShellStatus("내보내기에 실패했습니다", "error");
            }
        }

        async function prepareLoadedChildrenForImport() {
            for (var i = 0; i < slotIds.length; i++) {
                var slot = slotIds[i];
                if (!readySlots[slot]) continue;
                var result = await requestChild(slot, "prepare-import");
                if (!result.ok) throw new Error(result.error || "Workspace 가져오기 준비 실패");
            }
        }

        async function cancelLoadedChildrenImport() {
            for (var i = 0; i < slotIds.length; i++) {
                var slot = slotIds[i];
                if (!readySlots[slot]) continue;
                try { await requestChild(slot, "cancel-import"); } catch (e) {}
            }
        }

        function importWorkspaceFile(file) {
            if (!file) return;
            if (file.size > MAX_IMPORT_BYTES) {
                setShellStatus("가져올 파일이 너무 큽니다", "error");
                return;
            }
            if (hasPendingLoads()) {
                setShellStatus("Workspace 준비가 끝난 뒤 불러와 주세요", "warning");
                return;
            }
            if (loadingSlot) loadingSlot = null;
            var reader = new FileReader();
            reader.onload = async function () {
                var childrenPaused = false;
                try {
                    var parsed = JSON.parse(String(reader.result || ""));
                    var preview = storage.previewImport(parsed);
                    if (!window.confirm(
                        "현재 Workspace를 가져온 파일의 " + preview.ids.length
                            + "개 Workspace로 교체할까요? 로컬 txt2img 갤러리 기록도 비워집니다."
                    )) return;
                    setShellStatus("Workspace 불러오는 중…", "pending");
                    // Mark this before the sequential handshake: if a later
                    // child times out, earlier children may already be paused.
                    childrenPaused = true;
                    await prepareLoadedChildrenForImport();
                    var result = await storage.importStore(parsed, state.active);
                    writeShellState({
                        active: result.activeSlot,
                        names: result.names
                    });
                    window.location.reload();
                } catch (e) {
                    if (childrenPaused) await cancelLoadedChildrenImport();
                    console.warn("[SAM3 Live Workspaces] import failed:", e);
                    setShellStatus("올바른 Workspace 파일이 아닙니다", "error");
                }
            };
            reader.readAsText(file);
        }

        slotIds.forEach(function (slot) {
            addSlot(slot, state.names[slot] || slot);
        });

        window.addEventListener("message", function (event) {
            var detail = event && event.data;
            if (event.origin !== window.location.origin || !detail) return;
            if (detail.type === READY_MESSAGE) {
                var readySlot = String(detail.slot || "");
                if (slotIds.indexOf(readySlot) < 0) return;
                var readyFrame = frameFor(readySlot);
                if (!readyFrame || readyFrame.contentWindow !== event.source) return;
                markReady(readySlot, detail);
                return;
            }
            if (detail.type !== CONTROL_REPLY) return;
            var pending = pendingRequests[String(detail.requestId || "")];
            if (!pending) return;
            var replyFrame = frameFor(pending.slot);
            if (!replyFrame || replyFrame.contentWindow !== event.source) return;
            clearTimeout(pending.timer);
            delete pendingRequests[detail.requestId];
            pending.resolve(detail.result || { ok: false, error: "빈 Workspace 응답" });
        });

        shell.querySelector("[data-sam3-live-add]").addEventListener("click", createWorkspace);
        shell.querySelector("[data-sam3-live-delete]").addEventListener("click", deleteWorkspace);
        shell.querySelector("[data-sam3-live-rename]").addEventListener("click", function () {
            if (renameSlot(state.active, nameInput.value)) menu.open = false;
        });
        nameInput.addEventListener("keydown", function (event) {
            if (event.key === "Enter") {
                event.preventDefault();
                if (renameSlot(state.active, nameInput.value)) menu.open = false;
            } else if (event.key === "Escape") {
                event.preventDefault();
                syncNameEditor();
                menu.open = false;
            }
        });
        menu.addEventListener("toggle", function () {
            if (menu.open) syncNameEditor();
        });
        shell.querySelector("[data-sam3-live-export]").addEventListener("click", exportWorkspaces);
        shell.querySelector("[data-sam3-live-import]").addEventListener("change", function (event) {
            var input = event.target;
            importWorkspaceFile(input.files && input.files[0]);
            input.value = "";
        });
        shell.querySelector("[data-sam3-live-legacy]").addEventListener("click", function () {
            var url = standaloneShell
                ? new URL("/", window.location.origin)
                : new URL(window.location.href);
            if (standaloneShell) url.search = window.location.search;
            url.searchParams.delete("__sam3_live_workspace");
            url.searchParams.set("sam3_live", "off");
            window.location.href = url.toString();
        });

        syncNameEditor();
        activate(state.active);
        queueDefaultWorkspaces();
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
        if (!ready) return false;
        if (document.querySelector("#sam3_live_child_layout")) return true;

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
        return true;
    }

    async function mountChildFrame() {
        var layoutReady = false;
        try {
            layoutReady = await mountChildLayout();
        } finally {
            // workspace_manager waits for this handshake before it restores
            // Script/XYZ values into nodes moved by the Live layout.
            window.__sam3LiveChildLayoutSettled = true;
        }
        var managerReady = await waitFor(function () {
            var manager = window.__sam3WorkspaceManager;
            if (!manager || typeof manager.diagnostics !== "function") return null;
            var diagnostics = manager.diagnostics();
            return diagnostics.initialized
                && !diagnostics.bootstrapping
                && !diagnostics.restoring
                && !diagnostics.switching;
        }, 60000);
        try {
            window.parent.postMessage({
                type: READY_MESSAGE,
                slot: frameSlot,
                degraded: !layoutReady || !managerReady
            }, window.location.origin);
        } catch (e) {
            console.warn("[SAM3 Live Workspaces] ready signal failed:", e);
        }
    }

    function installChildControlBridge() {
        if (!frameSlot || window.__sam3LiveControlBridgeInstalled) return;
        window.__sam3LiveControlBridgeInstalled = true;
        window.addEventListener("message", async function (event) {
            var detail = event && event.data;
            if (event.origin !== window.location.origin
                    || event.source !== window.parent
                    || !detail || detail.type !== CONTROL_MESSAGE) return;
            var result;
            try {
                var manager = await waitFor(function () {
                    return window.__sam3WorkspaceManager || null;
                }, 60000);
                if (!manager) throw new Error("Workspace manager unavailable");
                if (detail.action === "flush") {
                    result = await manager.flushForLiveShell();
                } else if (detail.action === "prepare-import"
                        || detail.action === "prepare-delete") {
                    result = manager.prepareForLiveImport();
                } else if (detail.action === "cancel-import") {
                    result = manager.cancelLiveImport();
                } else {
                    result = { ok: false, error: "지원하지 않는 Workspace 명령" };
                }
            } catch (e) {
                result = { ok: false, error: String(e && e.message ? e.message : e) };
            }
            window.parent.postMessage({
                type: CONTROL_REPLY,
                requestId: detail.requestId,
                slot: frameSlot,
                result: result
            }, window.location.origin);
        });
    }

    function start() {
        if (frameSlot) {
            installChildControlBridge();
            mountChildFrame();
        }
        else if (!liveDisabled) {
            if (standaloneRedirect) {
                standaloneRedirect.then(function (redirected) {
                    if (!redirected) mountShell();
                });
            } else {
                mountShell();
            }
        }
    }

    if (typeof onUiLoaded === "function") onUiLoaded(function () { setTimeout(start, 300); });
    else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
    else start();
})();
