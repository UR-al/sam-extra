/*
 * SAM3 extension — LoRA Manager tab injection (v0.9.0)
 *
 * Injects a "Manage" (or "LoRA" in replace mode) tab into Forge's
 * extra-networks tab strip (#txt2img_extra_tabs / #img2img_extra_tabs). The
 * tab hosts an <iframe> pointing at the vendored ComfyUI-Lora-Manager
 * standalone server, which is lazily spawned the first time the tab is opened.
 *
 * Python bridge (scripts/lora_manager.py):
 *   #sam3_lm_config_btn  → #sam3_lm_config_out : JSON {available, replace, port}
 *   #sam3_lm_spawn_btn   → #sam3_lm_spawn_out  : JSON {url, status, message}
 *
 * DOM facts (verified against modules/ui_extra_networks.py + extraNetworks.js):
 *   - tab strip:  #${tab}_extra_tabs
 *   - nav bar:    #${tab}_extra_tabs > div.tab-nav   (contains <button> per tab)
 *   - tab panes:  #${tab}_extra_tabs > [id^='${tab}_']  (one div per tab,
 *                 e.g. #txt2img_lora, in the same order as the nav buttons)
 */

(function () {
    "use strict";

    var CONFIG = null;          // {available, replace, port}
    var SPAWNED = false;        // spawn attempted/succeeded once
    var SPAWN_RESULT = null;    // cached JSON from the spawn bridge

    function app() {
        return (typeof gradioApp === "function") ? gradioApp() : document;
    }

    // ---- Gradio hidden-bridge call -------------------------------------
    // Click a hidden Gradio button, wait for its paired output textbox to
    // receive a (non-empty, changed) value, resolve with that string.
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

            // Clear current value so we can detect the fresh result.
            var wrap0 = root.querySelector("#" + outId);
            if (wrap0) {
                var el0 = wrap0.querySelector("textarea, input");
                if (el0) el0.value = "";
            }

            var t0 = Date.now();
            var iv = setInterval(function () {
                var cur = readOut();
                if (cur) { clearInterval(iv); resolve(cur); return; }
                if (Date.now() - t0 > (timeoutMs || 60000)) {
                    clearInterval(iv);
                    reject("bridge timeout: " + outId);
                }
            }, 200);

            btn.click();
        });
    }

    // ---- Tab strip injection -------------------------------------------

    function navButtons(container) {
        var nav = container.querySelector(":scope > div.tab-nav");
        if (!nav) return { nav: null, buttons: [] };
        var btns = Array.prototype.slice.call(
            nav.querySelectorAll(":scope > button")
        );
        return { nav: nav, buttons: btns };
    }

    function tabPanes(container, tab) {
        return Array.prototype.slice.call(
            container.querySelectorAll(":scope > [id^='" + tab + "_']")
        );
    }

    function injectForTab(tab) {
        var container = app().querySelector("#" + tab + "_extra_tabs");
        if (!container) return false;
        if (container.getAttribute("data-sam3-lm") === "1") return true;

        var navInfo = navButtons(container);
        if (!navInfo.nav || navInfo.buttons.length === 0) return false; // not ready

        var panes = tabPanes(container, tab);
        if (panes.length === 0) return false;

        container.setAttribute("data-sam3-lm", "1");

        var replace = !!(CONFIG && CONFIG.replace);
        var label = replace ? "LoRA" : "Manage";

        // Build the Manage nav button (mimic Gradio nav button styling).
        var myBtn = document.createElement("button");
        myBtn.textContent = label;
        myBtn.className = navInfo.buttons[0].className; // inherit gradio styling
        myBtn.setAttribute("data-sam3-lm-btn", "1");
        myBtn.style.cursor = "pointer";

        // Insert the button after the last real tab button (before the
        // extra-networks controls div that extraNetworks.js appends).
        var controlsDiv = navInfo.nav.querySelector(":scope > .extra-networks-controls-div");
        if (controlsDiv) navInfo.nav.insertBefore(myBtn, controlsDiv);
        else navInfo.nav.appendChild(myBtn);

        // Build the Manage pane (iframe host).
        var myPane = document.createElement("div");
        myPane.id = tab + "_loramanager";
        myPane.className = "tabitem sam3-lm-pane";
        myPane.style.display = "none";
        myPane.style.padding = "0";

        var status = document.createElement("div");
        status.className = "sam3-lm-status";
        status.style.padding = "8px";
        status.style.opacity = "0.8";
        status.textContent = "";

        var frame = document.createElement("iframe");
        frame.className = "sam3-lm-frame";
        frame.style.width = "100%";
        frame.style.height = "78vh";
        frame.style.border = "0";
        frame.style.display = "none";
        frame.setAttribute("loading", "lazy");

        myPane.appendChild(status);
        myPane.appendChild(frame);
        container.appendChild(myPane);

        // Build unified switching arrays (gradio tabs + ours). Order matters:
        // nav buttons and panes are positionally aligned in Gradio's output.
        var loraIndex = -1;
        for (var i = 0; i < navInfo.buttons.length; i++) {
            var t = (navInfo.buttons[i].textContent || "").trim().toLowerCase();
            if (t === "lora") { loraIndex = i; break; }
        }

        var allButtons = navInfo.buttons.concat([myBtn]);
        var allPanes = panes.concat([myPane]);
        var myIndex = allButtons.length - 1;

        function selectTab(idx) {
            for (var k = 0; k < allPanes.length; k++) {
                allPanes[k].style.display = (k === idx) ? "block" : "none";
                if (allButtons[k]) {
                    if (k === idx) allButtons[k].classList.add("selected");
                    else allButtons[k].classList.remove("selected");
                }
            }
            // Hide the search/sort/refresh controls when our tab is active —
            // they belong to the card pages, not the iframe.
            if (controlsDiv) controlsDiv.style.display = (idx === myIndex) ? "none" : "";
            if (idx === myIndex) ensureServer(frame, status);
        }

        // Wire every button to deterministic switching (covers Gradio's
        // already-selected no-op case).
        allButtons.forEach(function (btn, idx) {
            btn.addEventListener("click", function () {
                // Defer one tick so Gradio's own handler runs first, then we
                // enforce the correct final visibility.
                setTimeout(function () { selectTab(idx); }, 0);
            });
        });

        if (replace && loraIndex >= 0) {
            // Hide the original LoRA tab entirely; our tab takes its place.
            navInfo.buttons[loraIndex].style.display = "none";
            if (panes[loraIndex]) panes[loraIndex].style.display = "none";
            selectTab(myIndex);
        }

        return true;
    }

    function ensureServer(frame, status) {
        if (SPAWNED) {
            if (SPAWN_RESULT && SPAWN_RESULT.url && frame.style.display === "none") {
                frame.src = SPAWN_RESULT.url;
                frame.style.display = "block";
                status.textContent = "";
            }
            return;
        }
        SPAWNED = true;
        status.textContent = "LoRA Manager 서버 시작 중... (최초 1회, ~10초)";
        bridgeCall("sam3_lm_spawn_btn", "sam3_lm_spawn_out", 60000)
            .then(function (raw) {
                var res;
                try { res = JSON.parse(raw); } catch (e) { res = { status: "error", message: raw }; }
                SPAWN_RESULT = res;
                if (res.url && (res.status === "running" || res.status === "spawned")) {
                    frame.src = res.url;
                    frame.style.display = "block";
                    status.textContent = "";
                } else {
                    status.innerHTML = "<span style='color:#c33'>LoRA Manager 시작 실패: " +
                        (res.message || "unknown") + "</span>";
                    SPAWNED = false; // allow retry on next open
                }
            })
            .catch(function (err) {
                status.innerHTML = "<span style='color:#c33'>LoRA Manager 브리지 오류: " + err + "</span>";
                SPAWNED = false;
            });
    }

    // ---- Bootstrap ------------------------------------------------------

    function tryInjectAll() {
        var okT = injectForTab("txt2img");
        var okI = injectForTab("img2img");
        return okT && okI;
    }

    function start() {
        // 1) fetch config (tab mode / availability) once.
        bridgeCall("sam3_lm_config_btn", "sam3_lm_config_out", 20000)
            .then(function (raw) {
                try { CONFIG = JSON.parse(raw); } catch (e) { CONFIG = { available: false }; }
                if (!CONFIG.available) {
                    console.log("[SAM3] LoRA Manager vendor unavailable — Manage tab skipped.");
                    return;
                }
                // 2) inject; retry until both strips exist.
                if (!tryInjectAll()) {
                    var tries = 0;
                    var iv = setInterval(function () {
                        tries++;
                        if (tryInjectAll() || tries > 60) clearInterval(iv);
                    }, 500);
                }
            })
            .catch(function (err) {
                console.log("[SAM3] LoRA Manager config bridge failed:", err);
            });
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(function () { setTimeout(start, 800); });
    } else {
        document.addEventListener("DOMContentLoaded", function () {
            setTimeout(start, 1500);
        });
    }
})();
