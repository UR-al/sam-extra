// sam-extra 자체 코드(GPL-3.0-only) — 상류 tagcomplete 코드가 아니다.
//
// 내장 Tag Autocomplete 로더. sd-webui-tagcomplete-neo(@ 8e0d81f, MIT)의 JS 15개는 바꾸지 않고
// sam3ext/tagcomplete/javascript/ 에 있다. Forge 는 확장의 javascript/ 맨 위 파일만 <head> 에 넣으므로
// (modules/extensions.py list_files — 하위 폴더는 읽지 않음) 그 파일들은 자동으로 실리지 않고, 이 로더가 붙인다:
//
// 1. Forge 의 설정(opts)이 페이지에 올라오기를 기다린다(script.js onOptionsAvailable). 켜졌는지·이유·파일 목록은 숨은 설정
//    sam3_tac_builtin_boot(sam3ext/tagcomplete/coexist.py — 화면에 없고 저장되지 않음, 값은 이 Forge 프로세스가 스크립트를
//    불러올 때 정한 것)에 들어 있다. 설정은 페이지와 함께 Gradio 가 보내므로 늦게 붙는 경로(app_started)를 기다리지 않는다 —
//    상류도 opts 가 차야 초기화를 시작한다(tagAutocomplete.js 의 onUiUpdate). 예전처럼 GET 경로를 물으면 Forge 가 막 떠서
//    app_started 가 끝나기 전에 열린 페이지(자동으로 여는 브라우저, Reload UI 뒤 새로 고침)에서 내장이 시작되지 않았다.
// 2. 그때 다른 tagcomplete 가 이미 실렸는지 본다 — 상류의 전역 TAC_CFG(var) · addAutocompleteToArea(function). 있으면 한 줄
//    남기고 멈춘다. 같은 const/class 파일을 두 번 실으면 "Identifier has already been declared" 로 반만 초기화되므로 둘이 같은
//    페이지에서 돌면 안 된다. 꺼졌으면(설정 스위치, 따로 설치된 사본, 시작 실패) 이유를 한 번 남기고 멈춘다.
// 2b. 시작 정보에 tagFileFallback({key: "tac_tagFile", missing, use})이 있으면 — 서버가 본 결과 고른 태그 파일이 tags/ 에
//    없고 기본 목록은 있음 — 페이지의 opts 에서만 opts.tac_tagFile 이 missing 일 때 use 로 바꾼다(상류는 없는 파일이면 추천
//    목록을 통째로 비움). 상류 JS 를 붙이기 전이라 상류가 처음 읽는 값부터 바뀐 값이다. Forge 는 opts 를 페이지로 보내기만 하고
//    되읽지 않으므로 config.json 과 Settings 화면의 값은 그대로다. Apply settings 뒤 Forge 가 opts 를 새로 채우면
//    onOptionsChanged 에서 다시 바꾼다(상류는 누른 뒤 500 ms 에 다시 읽음) — 다른 파일을 고르면 그 값은 건드리지 않는다.
// 3. 받은 순서(Forge 가 상류 확장 폴더에서 쓰던 이름순 그대로)대로 <script async=false> 를 <head> 에 붙인다 — 붙인
//    차례대로 실행된다. 하나가 실패해도 나머지는 붙이고 콘솔에 남긴다.
// 4. 다 실린 뒤 UI 가 이미 다 그려져 있으면(#txt2img_prompt) gradioApp() 에 빈 주석 노드를 넣었다 빼, Forge 의
//    MutationObserver(script.js)가 onUiUpdate 콜백을 한 번 더 돌게 한다 — 상류의 초기화는 onUiUpdate 에서만 시작한다.
//
// 전역 이름을 만들지 않는다. 상태는 <html data-sam3-tagcomplete="loading|started|skipped|failed"> 로만 남긴다.
(function () {
    "use strict";

    var BOOT_OPTION = "sam3_tac_builtin_boot";
    var TAG_FILE_SETTING = "tac_tagFile";
    var POLL_MS = 200;
    var PREFIX = "[sam-extra] Tag Autocomplete: ";
    var root = document.documentElement;

    if (root.hasAttribute("data-sam3-tagcomplete")) return;   // this file ran already on this page
    root.setAttribute("data-sam3-tagcomplete", "loading");

    function setState(state) {
        root.setAttribute("data-sam3-tagcomplete", state);
    }

    // Top-level var/function declarations of another tagcomplete copy (any a1111-sd-webui-tagcomplete fork).
    function anotherCopyLoaded() {
        try {
            if (typeof TAC_CFG !== "undefined") return true;   // eslint-disable-line no-undef
            return typeof addAutocompleteToArea === "function"; // eslint-disable-line no-undef
        } catch (_error) {
            return true;   // declared but not initialised yet: someone else is loading it
        }
    }

    // Forge's settings (script.js `let opts`, a global lexical binding — not a window property)
    function currentOptions() {
        try {
            return (typeof opts === "object" && opts !== null) ? opts : null;   // eslint-disable-line no-undef
        } catch (_error) {
            return null;
        }
    }

    function whenOptionsAvailable(callback) {
        if (typeof onOptionsAvailable === "function") {   // eslint-disable-line no-undef
            onOptionsAvailable(callback);                  // eslint-disable-line no-undef
            return;
        }
        (function poll() {   // a webui without onOptionsAvailable: wait for opts to fill
            var options = currentOptions();
            if (options && Object.keys(options).length > 0) callback();
            else setTimeout(poll, POLL_MS);
        })();
    }

    // {enabled, reason, upstream, scripts} or null (with a warning)
    function readBoot() {
        var options = currentOptions();
        var raw = options ? options[BOOT_OPTION] : undefined;
        if (raw === undefined || raw === null || raw === "") {
            console.warn(PREFIX + "Forge's settings have no " + BOOT_OPTION
                + " (sam-extra's scripts/builtin_tagcomplete.py did not run - see the server console); built-in copy not started");
            return null;
        }
        if (typeof raw === "string") {
            try {
                raw = JSON.parse(raw);
            } catch (_error) {
                raw = null;
            }
        }
        if (!raw || typeof raw !== "object") {
            console.warn(PREFIX + BOOT_OPTION + " is not a JSON object; built-in copy not started");
            return null;
        }
        return raw;
    }

    // {key: "tac_tagFile", missing, use} from the boot information, or null — the loader changes no other setting
    function readFallback(raw) {
        if (!raw || typeof raw !== "object") return null;
        if (raw.key !== TAG_FILE_SETTING || typeof raw.missing !== "string" || typeof raw.use !== "string") return null;
        if (!raw.missing || !raw.use || raw.missing === raw.use) return null;
        return { key: raw.key, missing: raw.missing, use: raw.use };
    }

    // Read the default list instead of a tag file that is not in tags/ — on this page's opts only (step 2b above).
    function useTagFileFallback(raw) {
        var fallback = readFallback(raw);
        if (!fallback) return;
        var told = false;
        function apply() {
            var options = currentOptions();
            if (!options || options[fallback.key] !== fallback.missing) return;   // another file was picked: leave it
            options[fallback.key] = fallback.use;
            if (told) return;
            told = true;
            console.warn(PREFIX + fallback.key + " = '" + fallback.missing + "' is not in the tags folder; suggestions use"
                + " the default list '" + fallback.use + "' on this page (the setting is not changed - pick another file in"
                + " Settings -> Tag Autocomplete)");
        }
        apply();
        if (typeof onOptionsChanged === "function") onOptionsChanged(apply);   // eslint-disable-line no-undef
    }

    function nudgeUiUpdate() {
        if (typeof gradioApp !== "function") return;
        var app = gradioApp();
        if (!app || !app.querySelector("#txt2img_prompt")) return;   // later UI mutations run onUiUpdate anyway
        var marker = document.createComment("sam3-tagcomplete");
        app.appendChild(marker);
        app.removeChild(marker);
    }

    function addScripts(urls) {
        return new Promise(function (resolve) {
            var pending = urls.length;
            var failed = 0;
            if (!pending) {
                resolve(0);
                return;
            }
            urls.forEach(function (url) {
                var script = document.createElement("script");
                script.type = "text/javascript";
                script.async = false;   // dynamic scripts with async=false run in the order they were added
                script.src = url;
                script.addEventListener("load", function () {
                    if (--pending === 0) resolve(failed);
                });
                script.addEventListener("error", function () {
                    failed++;
                    console.error(PREFIX + "could not load " + url);
                    if (--pending === 0) resolve(failed);
                });
                document.head.appendChild(script);
            });
        });
    }

    async function start() {
        if (anotherCopyLoaded()) {
            setState("skipped");
            console.log(PREFIX + "another tagcomplete is loaded on this page; built-in copy not started");
            return;
        }
        var boot = readBoot();
        if (!boot) {
            setState("failed");
            return;
        }
        if (!boot.enabled) {
            setState("skipped");
            console.log(PREFIX + "built-in copy not started: " + (boot.reason || "turned off"));
            return;
        }
        useTagFileFallback(boot.tagFileFallback);   // before upstream's first syncOptions reads opts
        var urls = Array.isArray(boot.scripts) ? boot.scripts.filter(function (u) { return typeof u === "string"; }) : [];
        var failed = await addScripts(urls);
        setState(failed ? "failed" : "started");
        nudgeUiUpdate();
    }

    var started = false;
    function run() {
        if (started) return;   // Forge runs onOptionsAvailable callbacks once; this guards the polling fallback too
        started = true;
        start().catch(function (error) {
            setState("failed");
            console.error(PREFIX + "loader error", error);
        });
    }

    function waitForOptions() {
        whenOptionsAvailable(run);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", waitForOptions, { once: true });
    } else {
        waitForOptions();
    }
})();
