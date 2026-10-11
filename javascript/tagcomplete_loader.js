// sam-extra 자체 코드(GPL-3.0-only) — 상류 tagcomplete 코드가 아니다.
//
// 내장 Tag Autocomplete 로더. sd-webui-tagcomplete-neo(@ 8e0d81f, MIT)의 JS 15개는 바꾸지 않고
// sam3ext/tagcomplete/javascript/ 에 있다. Forge 는 확장의 javascript/ 맨 위 파일만 <head> 에 넣으므로
// (modules/extensions.py list_files — 하위 폴더는 읽지 않음) 그 파일들은 자동으로 실리지 않고, 이 로더가 붙인다:
//
// 1. 페이지의 <head> 스크립트가 모두 돈 뒤(DOMContentLoaded) 다른 tagcomplete 가 이미 실렸는지 본다 — 상류의 전역
//    TAC_CFG(var) · addAutocompleteToArea(function). 있으면 한 줄 남기고 멈춘다. 같은 const/class 파일을 두 번 실으면
//    "Identifier has already been declared" 로 반만 초기화되므로 둘이 같은 페이지에서 돌면 안 된다.
// 2. GET sam3-tagcomplete/boot(sam3ext/tagcomplete/boot.py)로 켜졌는지와 파일 목록을 받는다. 서버가 막 떠서 경로가 아직
//    없으면(404·네트워크 오류) 500 ms 간격으로 5번 더 묻는다. 꺼졌으면(설정 스위치, 따로 설치된 사본, 시작 실패) 이유를
//    한 번 남기고 멈춘다.
// 3. 받은 순서(Forge 가 상류 확장 폴더에서 쓰던 이름순 그대로)대로 <script async=false> 를 <head> 에 붙인다 — 붙인
//    차례대로 실행된다. 하나가 실패해도 나머지는 붙이고 콘솔에 남긴다.
// 4. 다 실린 뒤 UI 가 이미 다 그려져 있으면(#txt2img_prompt) gradioApp() 에 빈 주석 노드를 넣었다 빼, Forge 의
//    MutationObserver(script.js)가 onUiUpdate 콜백을 한 번 더 돌게 한다 — 상류의 초기화는 onUiUpdate 에서만 시작한다.
//
// 전역 이름을 만들지 않는다. 상태는 <html data-sam3-tagcomplete="loading|started|skipped|failed"> 로만 남긴다.
(function () {
    "use strict";

    var BOOT_URL = "sam3-tagcomplete/boot";
    var RETRIES = 5;
    var RETRY_MS = 500;
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

    function sleep(ms) {
        return new Promise(function (resolve) { setTimeout(resolve, ms); });
    }

    async function fetchBoot() {
        for (var attempt = 0; attempt <= RETRIES; attempt++) {
            if (attempt > 0) await sleep(RETRY_MS);
            var response;
            try {
                response = await fetch(BOOT_URL, {
                    cache: "no-store",
                    credentials: "same-origin",
                    headers: { "X-SAM3-Notebook": "1" },
                });
            } catch (_error) {
                continue;   // server not reachable yet
            }
            if (response.status === 404) continue;   // app_started has not registered the route yet
            if (!response.ok) {
                console.warn(PREFIX + "boot request failed (HTTP " + response.status + "); built-in copy not started");
                return null;
            }
            try {
                return await response.json();
            } catch (_error) {
                console.warn(PREFIX + "boot answer is not JSON; built-in copy not started");
                return null;
            }
        }
        console.warn(PREFIX + "no answer from " + BOOT_URL + "; built-in copy not started");
        return null;
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
        var boot = await fetchBoot();
        if (!boot) {
            setState("failed");
            return;
        }
        if (!boot.enabled) {
            setState("skipped");
            console.log(PREFIX + "built-in copy not started: " + (boot.reason || "turned off"));
            return;
        }
        if (anotherCopyLoaded()) {   // loaded while we waited for the server
            setState("skipped");
            console.log(PREFIX + "another tagcomplete is loaded on this page; built-in copy not started");
            return;
        }
        var urls = Array.isArray(boot.scripts) ? boot.scripts.filter(function (u) { return typeof u === "string"; }) : [];
        var failed = await addScripts(urls);
        setState(failed ? "failed" : "started");
        nudgeUiUpdate();
    }

    function run() {
        start().catch(function (error) {
            setState("failed");
            console.error(PREFIX + "loader error", error);
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", run, { once: true });
    } else {
        run();
    }
})();
