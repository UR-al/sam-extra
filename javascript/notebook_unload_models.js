/* SAM Extra '모델 내리기' — notebook 레이아웃 탭 줄에서 '기본값 저장'(javascript/notebook_save_defaults.js) 바로 오른쪽의 버튼.
 *
 * 한 번 누르면 확인 없이 숨은 Gradio 버튼 #sam3_unload_models_run 을 누르고(sam3ext/ui_unload_models.py) 탭 줄 아래에 결과 한
 * 줄을 보여 준다(예: "VRAM 사용 19.6 GB → 3.1 GB (Forge 모델 · SAM3) · 이 프로세스 예약 15.2 → 0.0 GB", 생성 중이면
 * "생성 중에는 내릴 수 없습니다"). 결과 글에 마우스를 올리면 숫자의 뜻과 내린 것 · 쥔 것이 없던 것 · 실패한 것이 보인다.
 * 무엇을 어떤 순서로 내리는지는 sam3ext/unload_models.py, 결과 글도 거기서 만든다 — 이 스크립트는 그대로 보여 주기만 한다.
 *
 * 요청 id 는 숨은 버튼 이벤트의 js 가 window.sam3UnloadModelsArgs 로 첫 입력 칸에 넣고, 결과 JSON(#sam3_unload_models_result)은
 * 요청 id 가 같은 것만 받는다(지난 결과 · 다른 탭의 결과를 잘못 읽지 않게).
 *
 * 자리: 버튼은 '기본값 저장' 과 같은 모양(같은 클래스)이고 **그 묶음(#sam3_save_defaults) 안, 저장 버튼 바로 뒤**에 들어간다 —
 * 탭 줄의 flex 항목이 하나라 좁은 창에서 줄이 넘어가도 두 버튼이 함께 움직인다(따로 두면 이 버튼만 다음 줄 왼쪽 끝으로 갔다).
 * 결과 글(#sam3_unload_models_status)은 버튼 묶음 밖, 탭 줄의 따로 된 항목이다: style.css 가 한 줄을 통째로 주어(flex-basis
 * 100%) 버튼 아래 오른쪽에 둔다 — 결과가 길어도 버튼의 자리는 그대로다. 기본값 저장이 아직 없으면 줄 끝에 혼자 붙고, 생기면
 * 다음 확인 때 그 안으로 옮긴다. 버튼 · 결과 글은 한 번 만들어 다시 쓴다 — Gradio 가 탭 줄을 다시 그리면 같은 것을 새 줄에
 * 다시 붙인다(진행 중인 결과 글이 사라지지 않게). notebook 레이아웃이 켜져 있을 때만 생긴다.
 */
(function () {
    "use strict";

    if (window.__sam3UnloadModelsLoaded) return;
    window.__sam3UnloadModelsLoaded = true;

    var WRAP_ID = "sam3_unload_models";
    var STATUS_ID = "sam3_unload_models_status";
    var SAVE_WRAP_ID = "sam3_save_defaults";
    var RUN_ID = "sam3_unload_models_run";
    var RESULT_ID = "sam3_unload_models_result";
    var LABEL = "모델 내리기";
    var TITLE = "확인 없이 바로 내립니다: Forge 의 체크포인트(Settings 의 'Unload All Models' 와 같은 동작)와 "
        + "sam-extra 가 메모리에 쥔 모델(SAM3 · Anima 3.8B Qwen3.5/커넥터 · IP-Adapter · DeGrid · TIPO · "
        + "Modulation CLIP-L · VAE 2x 디코더 …), Forge 가 따로 쥔 업스케일러 · TAESD · 얼굴 복원 모델을 모두 내리고 "
        + "GPU 캐시를 비웁니다. 다음 생성(✨ 고해상도 포함) 때 고른 체크포인트를 다시 읽습니다. "
        + "생성 중에는 동작하지 않습니다.";
    var config = {
        mountPollMs: 1000,
        pollMs: 100,
        timeoutMs: 180000,
        doneVisibleMs: 15000,
        errorVisibleMs: 20000
    };
    var busy = false;
    var pendingRequestId = "";
    var statusTimer = null;
    var mountTimer = null;
    var requestCounter = 0;
    var nodes = null;      // { wrap, button, status } — 한 번 만들어 다시 쓴다

    function app() {
        return typeof gradioApp === "function" ? gradioApp() : document;
    }

    function find(selector) {
        var root = app();
        return (root && root.querySelector && root.querySelector(selector))
            || document.querySelector(selector);
    }

    // 숨은 Gradio 버튼의 js 가 부른다: Gradio 가 넘긴 [요청 id, (출력값)] 의 첫 칸만 바꾼다.
    window.sam3UnloadModelsArgs = function (args) {
        var out = Array.prototype.slice.call(args || []);
        out[0] = pendingRequestId || "";
        return out;
    };

    function newRequestId() {
        requestCounter += 1;
        return "um" + Date.now().toString(36) + requestCounter.toString(36)
            + Math.random().toString(36).slice(2, 8);
    }

    function message(error) {
        return String(error && error.message || error || "알 수 없는 오류");
    }

    function readResult() {
        var wrap = find("#" + RESULT_ID);
        var field = wrap && (wrap.matches("textarea, input") ? wrap : wrap.querySelector("textarea, input"));
        return field ? field.value : "";
    }

    function failure(text) {
        return { ok: false, text: text, detail: "" };
    }

    function requestUnload() {
        var run = find("#" + RUN_ID);
        if (!run) {
            return Promise.resolve(failure(
                "모델 내리기 연결이 없습니다 (Forge 콘솔의 [sam-extra] unload models 기록을 확인하세요)"));
        }
        var requestId = newRequestId();
        pendingRequestId = requestId;
        return new Promise(function (resolve) {
            var done = false;
            var poll = null;
            var timer = null;

            function finish(result) {
                if (done) return;
                done = true;
                window.clearInterval(poll);
                window.clearTimeout(timer);
                if (pendingRequestId === requestId) pendingRequestId = "";
                resolve(result);
            }

            function check() {
                var text = readResult();
                if (!text) return;
                var payload;
                try { payload = JSON.parse(text); } catch (error) { return; }
                if (!payload || payload.request !== requestId) return;   // 지난 결과
                finish({
                    ok: payload.ok === true,
                    text: String(payload.text || (payload.ok ? "모델을 내렸습니다" : "모델 내리기 실패")),
                    detail: String(payload.detail || "")
                });
            }

            poll = window.setInterval(check, config.pollMs);
            timer = window.setTimeout(function () {
                finish(failure("Forge 가 모델 내리기에 응답하지 않았습니다"));
            }, config.timeoutMs);
            try {
                run.click();
            } catch (error) {
                finish(failure("모델 내리기 실패: " + message(error)));
            }
        });
    }

    function setStatus(text, tone, detail) {
        var status = nodes && nodes.status;
        if (!status) return;
        window.clearTimeout(statusTimer);
        status.textContent = text || "";
        status.setAttribute("data-tone", tone || "normal");
        if (detail) status.title = detail;
        else status.removeAttribute("title");
        var visibleMs = tone === "saved" ? config.doneVisibleMs : tone === "error" ? config.errorVisibleMs : 0;
        if (visibleMs > 0) {
            statusTimer = window.setTimeout(function () {
                status.textContent = "";
                status.setAttribute("data-tone", "normal");
                status.removeAttribute("title");
            }, visibleMs);
        }
    }

    // '기본값 저장' 과 같다: disabled 로 바꾸면 키보드 포커스를 잃으므로 aria-disabled 로만 알리고, 두 번 눌림은 busy 가 막는다.
    function setButtonBusy(button, on) {
        if (!button) return;
        if (on) {
            button.setAttribute("aria-disabled", "true");
            button.setAttribute("aria-busy", "true");
        } else {
            button.removeAttribute("aria-disabled");
            button.removeAttribute("aria-busy");
        }
    }

    async function unloadModels() {
        if (busy) return null;
        busy = true;
        setButtonBusy(nodes && nodes.button, true);
        setStatus("내리는 중…", "busy");
        var result;
        try {
            result = await requestUnload().catch(function (error) {
                return failure("모델 내리기 실패: " + message(error));
            });
        } finally {
            busy = false;
            setButtonBusy(nodes && nodes.button, false);
        }
        var summary = { text: result.text, tone: result.ok ? "saved" : "error", detail: result.detail };
        setStatus(summary.text, summary.tone, summary.detail);
        return summary;
    }

    // ---- 버튼 ----------------------------------------------------------------------------------------

    function tabRow() {
        var layout = find("#sam3_notebook_layout");
        if (!layout) return null;
        var tabs = layout.querySelector("#txt2img_extra_tabs");
        return tabs ? tabs.querySelector(":scope > .tab-nav") : null;
    }

    function build() {
        var wrap = document.createElement("div");
        wrap.id = WRAP_ID;
        // 버튼만 담는다(결과 글은 따로). 자리는 style.css 의 .sam3-unload-models — 기본값 저장 묶음 안이든 줄에 혼자든
        wrap.className = "sam3-unload-models";
        var button = document.createElement("button");
        button.type = "button";
        button.className = "sam3-save-defaults-button sam3-unload-models-button";
        button.textContent = LABEL;
        button.title = TITLE;
        button.setAttribute("aria-describedby", STATUS_ID);
        button.addEventListener("click", function () { unloadModels(); });
        var status = document.createElement("span");
        status.id = STATUS_ID;
        status.className = "sam3-save-defaults-status sam3-unload-models-status";
        status.setAttribute("role", "status");
        status.setAttribute("aria-live", "polite");
        status.setAttribute("data-tone", "normal");
        wrap.appendChild(button);
        return { wrap: wrap, button: button, status: status };
    }

    function ensureButton() {
        var row = tabRow();
        if (!row) return false;
        if (!nodes) nodes = build();
        var wrap = nodes.wrap;
        var status = nodes.status;
        var save = row.querySelector(":scope > #" + SAVE_WRAP_ID);
        var saveButton = save && save.querySelector(":scope > button");
        if (saveButton) {
            // 기본값 저장 묶음 안, 저장 버튼 바로 뒤(탭 순서도 그 다음) — 저장 결과 글은 묶음 밖의 줄 항목이다
            if (wrap.previousElementSibling !== saveButton) saveButton.after(wrap);
        } else if (wrap.parentElement !== row) {
            row.appendChild(wrap);   // 기본값 저장이 아직 없으면 줄 끝 — 생기면 다음 확인 때 그 안으로 옮긴다
        }
        // 결과 글: 버튼 묶음 바로 뒤의 줄 항목(style.css 가 버튼 아래 한 줄로 둔다). 기본값 저장의 결과 글은 이 뒤에 붙는다
        // (notebook_save_defaults.js ensureButton)
        var anchor = wrap.parentElement === row ? wrap : save;
        if (status.previousElementSibling !== anchor) anchor.after(status);
        return true;
    }

    function boot() {
        ensureButton();
        if (!mountTimer) mountTimer = window.setInterval(ensureButton, config.mountPollMs);
    }

    // 기본값 저장의 스크립트가 먼저 읽혀(파일 이름 순) 같은 이벤트에서 먼저 붙인다 — 그 뒤에 자리를 맞춘다
    window.addEventListener("sam3:notebook-mounted", function () { ensureButton(); });

    // 테스트 전용 입구. 실제 페이지는 이 객체를 만들지 않는다.
    if (window.__sam3UnloadModelsTestHooks
            && typeof window.__sam3UnloadModelsTestHooks === "object") {
        var hooks = window.__sam3UnloadModelsTestHooks;
        hooks.config = config;
        hooks.ensureButton = ensureButton;
        hooks.unloadModels = unloadModels;
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(function () { window.setTimeout(boot, 300); });
    } else {
        document.addEventListener("DOMContentLoaded", function () {
            window.setTimeout(boot, 1300);
        });
    }
})();
