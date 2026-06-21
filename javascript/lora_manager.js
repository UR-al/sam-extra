/*
 * SAM3 extension — LoRA Manager tab injection (v0.9.1)
 *
 * Injects a "Manage" (or "LoRA" in replace mode) tab into Forge's
 * extra-networks tab strip (#txt2img_extra_tabs / #img2img_extra_tabs),
 * hosting an <iframe> onto the vendored ComfyUI-Lora-Manager standalone
 * server (lazily spawned on first open).
 *
 * Verified DOM (Forge + Gradio 4.40, with i18n):
 *   #${tab}_extra_tabs
 *     > div.tab-nav            → <button> per tab (text may be LOCALIZED, e.g. "로라")
 *     > div.tabitem            → one pane per tab, POSITIONALLY aligned with the
 *                                nav buttons. Some panes have stable elem_ids
 *                                (#txt2img_lora, #txt2img_checkpoints, ...), the
 *                                first "Generation" pane has a generated id.
 *   ⇒ select panes with ':scope > .tabitem' (NOT '[id^=tab_]').
 *   ⇒ replace-mode targets the LoRA tab by pane id '#${tab}_lora' (stable,
 *      i18n-proof), never by button text.
 *
 * Bridge (scripts/lora_manager.py):
 *   #sam3_lm_config_btn → #sam3_lm_config_out : {available, replace, port}
 *   #sam3_lm_spawn_btn  → #sam3_lm_spawn_out  : {url, status, message}
 *
 * Robustness lessons baked in:
 *   - Injection does NOT wait on the Python config bridge (which can be slow /
 *     time out during the congested first render). The tab is injected as soon
 *     as the DOM exists (MutationObserver + interval), defaulting to ADD mode;
 *     replace-mode / availability is applied later when config arrives.
 *   - The first server run hashes the whole LoRA library and aiohttp doesn't
 *     open the port until that finishes (observed ~266 s for 1487 LoRAs). So
 *     spawn is non-blocking and the iframe is loaded only after we poll the URL
 *     to reachability, showing scan progress meanwhile.
 */

(function () {
    "use strict";

    var CONFIG = null;
    var SPAWNED = false;
    var SPAWN_RESULT = null;
    var injected = { txt2img: false, img2img: false };
    var refs = {}; // tab -> {container, navBtns, panes, myBtn, myPane, myIndex, loraIdx, controlsDiv}

    function app() {
        return (typeof gradioApp === "function") ? gradioApp() : document;
    }

    // ---- Gradio hidden-bridge call (robust against startup congestion) ----
    function bridgeCall(btnId, outId, timeoutMs) {
        return new Promise(function (resolve, reject) {
            var root = app();
            var btn = root.querySelector("#" + btnId);
            if (!btn) { reject("bridge button missing: " + btnId); return; }

            function readOut() {
                var wrap = root.querySelector("#" + outId);
                if (!wrap) return null;
                var el = wrap.querySelector("textarea, input");
                return el ? el.value : null;
            }
            // clear stale value so a fresh non-empty result is detectable
            var w0 = root.querySelector("#" + outId);
            if (w0) { var e0 = w0.querySelector("textarea, input"); if (e0) e0.value = ""; }

            var t0 = Date.now();
            var clicks = 0;
            try { btn.click(); clicks = 1; } catch (e) {}
            var iv = setInterval(function () {
                var cur = readOut();
                if (cur) { clearInterval(iv); resolve(cur); return; }
                var elapsed = Date.now() - t0;
                if (elapsed > (timeoutMs || 120000)) {
                    clearInterval(iv);
                    reject("bridge timeout: " + outId);
                    return;
                }
                // Re-click every ~5s — the first click can be dropped while the
                // Gradio event queue is saturated during initial render.
                if (elapsed > clicks * 5000) {
                    clicks++;
                    var b = root.querySelector("#" + btnId);
                    if (b) { try { b.click(); } catch (e) {} }
                }
            }, 500);
        });
    }

    // ---- injection -------------------------------------------------------

    function injectForTab(tab) {
        if (injected[tab]) return true;
        var container = app().querySelector("#" + tab + "_extra_tabs");
        if (!container) return false;
        var nav = container.querySelector(":scope > div.tab-nav");
        if (!nav) return false;
        var navBtns = Array.prototype.slice.call(nav.querySelectorAll(":scope > button"));
        var panes = Array.prototype.slice.call(container.querySelectorAll(":scope > .tabitem"));
        if (navBtns.length === 0 || panes.length === 0) return false;

        injected[tab] = true;
        container.setAttribute("data-sam3-lm", "1");

        // Manage nav button (inherit gradio button styling from an existing tab).
        var myBtn = document.createElement("button");
        myBtn.textContent = "Manage";
        myBtn.className = navBtns[0].className;
        myBtn.setAttribute("data-sam3-lm-btn", "1");
        myBtn.style.cursor = "pointer";
        var controlsDiv = nav.querySelector(":scope > .extra-networks-controls-div");
        if (controlsDiv) nav.insertBefore(myBtn, controlsDiv);
        else nav.appendChild(myBtn);

        // Manage pane (iframe host) — must carry the .tabitem class so it sits
        // in the same content area as the other panes.
        var myPane = document.createElement("div");
        myPane.id = tab + "_loramanager";
        myPane.className = "tabitem sam3-lm-pane";
        myPane.style.display = "none";
        myPane.style.padding = "0";
        var statusEl = document.createElement("div");
        statusEl.className = "sam3-lm-status";
        statusEl.style.padding = "8px";
        statusEl.style.opacity = "0.85";
        var frame = document.createElement("iframe");
        frame.className = "sam3-lm-frame";
        frame.style.width = "100%";
        frame.style.height = "80vh";
        frame.style.border = "0";
        frame.style.display = "none";
        myPane.appendChild(statusEl);
        myPane.appendChild(frame);
        // place right after the last real .tabitem
        var lastPane = panes[panes.length - 1];
        if (lastPane && lastPane.parentNode === container) {
            container.insertBefore(myPane, lastPane.nextSibling);
        } else {
            container.appendChild(myPane);
        }

        var allBtns = navBtns.concat([myBtn]);
        var allPanes = panes.concat([myPane]);
        var myIndex = allBtns.length - 1;

        // LoRA tab index (by stable pane id, i18n-proof) — for replace mode.
        var loraIdx = -1;
        for (var i = 0; i < panes.length; i++) {
            if (panes[i].id === tab + "_lora") { loraIdx = i; break; }
        }

        function selectTab(idx) {
            for (var k = 0; k < allPanes.length; k++) {
                allPanes[k].style.display = (k === idx) ? "block" : "none";
                if (allBtns[k]) {
                    if (k === idx) allBtns[k].classList.add("selected");
                    else allBtns[k].classList.remove("selected");
                }
            }
            if (controlsDiv) controlsDiv.style.display = (idx === myIndex) ? "none" : "";
            if (idx === myIndex) ensureServer(frame, statusEl);
        }

        allBtns.forEach(function (btn, idx) {
            btn.addEventListener("click", function () {
                // defer so Gradio's own handler settles first, then enforce.
                setTimeout(function () { selectTab(idx); }, 0);
            });
        });

        refs[tab] = {
            container: container, navBtns: navBtns, panes: panes,
            myBtn: myBtn, myPane: myPane, myIndex: myIndex,
            loraIdx: loraIdx, selectTab: selectTab
        };

        // Apply config immediately if already known (replace mode / availability).
        if (CONFIG) applyConfigToTab(tab);
        return true;
    }

    function applyConfigToTab(tab) {
        var r = refs[tab];
        if (!r || !CONFIG) return;
        if (CONFIG.available === false) {
            r.myBtn.style.display = "none";
            r.myPane.style.display = "none";
            return;
        }
        if (CONFIG.replace && r.loraIdx >= 0) {
            r.myBtn.textContent = "LoRA";
            if (r.navBtns[r.loraIdx]) r.navBtns[r.loraIdx].style.display = "none";
            if (r.panes[r.loraIdx]) r.panes[r.loraIdx].style.display = "none";
            r.selectTab(r.myIndex);
        }
    }

    // ---- server spawn + iframe load -------------------------------------

    function loadFrame(url, frame, statusEl) {
        if (frame.getAttribute("data-loaded") === "1") return;
        frame.src = url;
        frame.style.display = "block";
        frame.setAttribute("data-loaded", "1");
        statusEl.textContent = "";
    }

    function pollUntilUp(url, frame, statusEl) {
        var tries = 0, maxTries = 360; // 12 min ceiling
        var iv = setInterval(function () {
            tries++;
            fetch(url, { mode: "no-cors", cache: "no-store" })
                .then(function () { clearInterval(iv); loadFrame(url, frame, statusEl); })
                .catch(function () {
                    if (tries >= maxTries) {
                        clearInterval(iv);
                        statusEl.innerHTML = "<span style='color:#c33'>LoRA Manager 시작 시간 초과 — " +
                            "lora_manager_vendor/forge_standalone.log 확인.</span>";
                        SPAWNED = false;
                    } else {
                        statusEl.textContent = "LoRA 모델 스캔 중... (최초 1회, 라이브러리가 크면 수 분 소요) " +
                            (tries * 2) + "s";
                    }
                });
        }, 2000);
    }

    function ensureServer(frame, statusEl) {
        if (frame.getAttribute("data-loaded") === "1") return;
        if (SPAWNED) {
            if (SPAWN_RESULT && SPAWN_RESULT.url) {
                if (SPAWN_RESULT.status === "running" || SPAWN_RESULT.status === "spawned") {
                    loadFrame(SPAWN_RESULT.url, frame, statusEl);
                } else {
                    pollUntilUp(SPAWN_RESULT.url, frame, statusEl);
                }
            }
            return;
        }
        SPAWNED = true;
        statusEl.textContent = "LoRA Manager 서버 시작 중...";
        bridgeCall("sam3_lm_spawn_btn", "sam3_lm_spawn_out", 60000)
            .then(function (raw) {
                var res;
                try { res = JSON.parse(raw); } catch (e) { res = { status: "error", message: raw }; }
                SPAWN_RESULT = res;
                if (!res.url) {
                    statusEl.innerHTML = "<span style='color:#c33'>LoRA Manager 시작 실패: " +
                        (res.message || "unknown") + "</span>";
                    SPAWNED = false;
                    return;
                }
                if (res.status === "running" || res.status === "spawned") {
                    loadFrame(res.url, frame, statusEl);
                } else {
                    statusEl.textContent = "LoRA 모델 스캔 중... (최초 1회, 라이브러리가 크면 수 분 소요)";
                    pollUntilUp(res.url, frame, statusEl);
                }
            })
            .catch(function (err) {
                statusEl.innerHTML = "<span style='color:#c33'>LoRA Manager 브리지 오류: " + err + "</span>";
                SPAWNED = false;
            });
    }

    // ---- bootstrap ------------------------------------------------------

    function tryAll() {
        var a = injectForTab("txt2img");
        var b = injectForTab("img2img");
        return a && b;
    }

    function loadConfig() {
        bridgeCall("sam3_lm_config_btn", "sam3_lm_config_out", 120000)
            .then(function (raw) {
                try { CONFIG = JSON.parse(raw); } catch (e) { CONFIG = { available: true, replace: false }; }
                Object.keys(refs).forEach(applyConfigToTab);
            })
            .catch(function (err) {
                console.log("[SAM3] LoRA Manager config bridge failed (defaulting to add-mode):", err);
            });
    }

    function start() {
        tryAll();
        // Keep trying as the DOM finishes building (heavy first render).
        var obs = new MutationObserver(function () { tryAll(); });
        try { obs.observe(document.documentElement, { childList: true, subtree: true }); } catch (e) {}
        var iv = setInterval(tryAll, 800);
        setTimeout(function () { try { obs.disconnect(); } catch (e) {} clearInterval(iv); }, 300000);

        // config affects only replace-mode/availability — fetch independently.
        loadConfig();
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(function () { setTimeout(start, 500); });
    } else {
        document.addEventListener("DOMContentLoaded", function () { setTimeout(start, 1500); });
    }
})();
