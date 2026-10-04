/* SAM Extra '기본값 저장' — notebook 레이아웃 탭 줄(Generation · 임베딩 · 체크포인트 · 로라 · Manage)의 오른쪽 끝 버튼 하나.
 *
 * 한 번 누르면 확인 없이 두 가지를 하고, 버튼 옆에 짧은 결과를 보여 준다
 * (예: "기본값 + anima 프리셋 저장됨 · 바뀐 항목: 기본값 3개, 프리셋 2개", 또는 오류 — 두 파일의 개수를 따로 적는다:
 * 한 칸이 두 파일에 다 들어가므로 더하면 한 번 바꾼 것이 두 번 세어진다):
 *
 * 1. Forge 의 Settings → Defaults 'Apply' — Forge 가 만든 #ui_defaults_apply 를 그대로 누른다
 *    (modules/ui_loadsave.py ui_apply: 화면의 모든 추적 컴포넌트 값을 ui-config 파일에 쓴다). 결과는 Forge 가 그 아래
 *    HTML 칸에 그리는 글("Wrote N changes." / "No changes.")에서 읽는다. 그 칸은 elem_id 가 없어서 Apply 버튼 뒤의
 *    첫 .prose 로 찾고, Gradio 가 출력 칸에 붙이는 .pending(처리 중)이 붙었다 떨어지는 것으로 끝을 안다.
 * 2. 활성 UI Preset 의 설정 22개 — 숨은 Gradio 버튼 #sam3_save_defaults_run 을 누른다(sam3ext/ui_save_defaults.py).
 *    그 이벤트는 main_entry 가 쓰는 컴포넌트들의 지금 값을 Gradio 가 직접 넘기고, 이벤트의 js 가
 *    window.sam3SaveDefaultsArgs 로 첫 입력 칸에 이번 요청 id 를 넣는다. 결과 JSON 은 숨은 Textbox
 *    #sam3_save_defaults_result 에 오고, 요청 id 가 같은 것만 받는다(지난 결과·다른 탭의 결과를 잘못 읽지 않게).
 *    프리셋 값이 0(on_preset_change 가 화면 값을 덮어쓰지 않음)인 스텝·크기·CFG 칸은 0 그대로 두고(결과의 kept — 바뀐
 *    항목으로 세지 않음) 결과 글의 툴팁에 그 칸 이름을 적는다. 그 칸의 화면 값은 1 의 ui-config 로 남는다.
 *
 * Forge 는 페이지를 열 때와 프리셋을 바꿀 때마다 2 의 칸들을 프리셋 값으로 덮어쓰므로(main_entry.on_preset_change),
 * 1 만으로는 체크포인트·스텝·샘플러·크기·CFG 같은 기본값이 남지 않는다 — 그래서 한 버튼이 둘 다 한다.
 * 버튼은 notebook 레이아웃이 켜져 있을 때만 생긴다. Gradio 가 탭 줄을 다시 그려도 주기적으로 다시 붙인다.
 */
(function () {
    "use strict";

    if (window.__sam3SaveDefaultsLoaded) return;
    window.__sam3SaveDefaultsLoaded = true;

    var WRAP_ID = "sam3_save_defaults";
    var STATUS_ID = "sam3_save_defaults_status";
    var RUN_ID = "sam3_save_defaults_run";
    var RESULT_ID = "sam3_save_defaults_result";
    var LABEL = "기본값 저장";
    var TITLE = "확인 없이 바로 저장합니다: Settings → Defaults 의 Apply(지금 화면 값을 ui-config 에) + "
        + "지금 UI Preset 의 체크포인트·VAE/TE·Low Bits·스텝·샘플러·스케줄러·크기·CFG·Shift·배치 "
        + "(Forge 가 페이지를 열 때 프리셋 값으로 덮어쓰는 칸들). 프리셋 값이 0(화면 값을 덮어쓰지 않음)인 "
        + "스텝·크기·CFG 는 0 그대로 둡니다. 다른 프리셋은 그대로입니다.";
    var config = {
        mountPollMs: 1000,
        pollMs: 100,
        defaultsTimeoutMs: 30000,
        presetTimeoutMs: 30000,
        savedVisibleMs: 8000,
        errorVisibleMs: 20000
    };
    var busy = false;
    var pendingRequestId = "";
    var statusTimer = null;
    var mountTimer = null;
    var requestCounter = 0;

    function app() {
        return typeof gradioApp === "function" ? gradioApp() : document;
    }

    function find(selector) {
        var root = app();
        return (root && root.querySelector && root.querySelector(selector))
            || document.querySelector(selector);
    }

    // 숨은 Gradio 버튼의 js 가 부른다: Gradio 가 넘긴 [요청 id, 프리셋, 값…, (출력값)] 의 첫 칸만 바꾼다.
    window.sam3SaveDefaultsArgs = function (args) {
        var out = Array.prototype.slice.call(args || []);
        out[0] = pendingRequestId || "";
        return out;
    };

    function newRequestId() {
        requestCounter += 1;
        return "sd" + Date.now().toString(36) + requestCounter.toString(36)
            + Math.random().toString(36).slice(2, 8);
    }

    function message(error) {
        return String(error && error.message || error || "알 수 없는 오류");
    }

    // ---- 1. Forge Settings → Defaults Apply -----------------------------------------------------------

    function defaultsReview(applyButton) {
        var scope = applyButton.closest("#settings_tab_defaults")
            || applyButton.closest(".tabitem") || applyButton.parentElement;
        if (!scope) return null;
        var proses = scope.querySelectorAll(".prose");
        for (var i = 0; i < proses.length; i++) {
            // Apply 버튼 뒤에 오는 첫 HTML 칸 (앞의 것은 설명 글)
            if (applyButton.compareDocumentPosition(proses[i]) & Node.DOCUMENT_POSITION_FOLLOWING) {
                var block = proses[i].closest(".block");
                return {
                    prose: proses[i],
                    block: block && scope.contains(block) ? block : proses[i].parentElement.parentElement
                        || proses[i].parentElement
                };
            }
        }
        return null;
    }

    function parseDefaultsResult(text) {
        var trimmed = String(text || "").replace(/\s+/g, " ").trim();
        var wrote = /Wrote (\d+) changes?\./.exec(trimmed);
        if (wrote) return { ok: true, changed: Number(wrote[1]), text: trimmed };
        if (/^No changes\.?$/.test(trimmed)) return { ok: true, changed: 0, text: trimmed };
        var numbers = trimmed.match(/\d+/g);
        if (numbers && numbers.length === 1) {
            return { ok: true, changed: Number(numbers[0]), text: trimmed };   // 번역된 문구
        }
        if (trimmed) return { ok: true, changed: null, text: trimmed };
        return { ok: false, error: "Forge 의 Defaults 결과 글이 비어 있습니다" };
    }

    function reviewFailed(review) {
        var errors = review.block.querySelectorAll(".error");
        for (var i = 0; i < errors.length; i++) {
            var wrap = errors[i].closest(".wrap");
            if (!wrap || !wrap.classList.contains("hide")) return true;
        }
        return false;
    }

    function applyForgeDefaults() {
        var applyButton = find("#ui_defaults_apply");
        if (!applyButton) {
            return Promise.resolve({
                ok: false,
                error: "Settings → Defaults 의 Apply 버튼이 없습니다 (Settings 탭을 숨겼나요?)"
            });
        }
        var review = defaultsReview(applyButton);
        if (!review) {
            return Promise.resolve({ ok: false, error: "Settings → Defaults 의 결과 칸을 찾지 못했습니다" });
        }
        return new Promise(function (resolve) {
            var startText = review.prose.textContent;
            var startFailed = reviewFailed(review);     // 지난번 Apply 의 오류 표시가 아직 남아 있을 수 있다
            var sawPending = false;
            var done = false;
            var observer = null;
            var poll = null;
            var timer = null;

            function finish(result) {
                if (done) return;
                done = true;
                if (observer) observer.disconnect();
                window.clearInterval(poll);
                window.clearTimeout(timer);
                resolve(result);
            }

            function check() {
                if (done) return;
                if (review.block.querySelector(".pending")) {
                    sawPending = true;
                    return;
                }
                var failed = reviewFailed(review);
                // 아직 시작 전: 처리 중 표시도, 새 글도, 새 오류도 없다
                if (!sawPending && review.prose.textContent === startText && !(failed && !startFailed)) return;
                if (failed) {
                    finish({ ok: false, error: "Forge 의 Defaults Apply 가 오류로 끝났습니다 (Forge 콘솔을 확인하세요)" });
                    return;
                }
                finish(parseDefaultsResult(review.prose.textContent));
            }

            if (typeof MutationObserver === "function") {
                observer = new MutationObserver(check);
                observer.observe(review.block, {
                    subtree: true, childList: true, attributes: true, characterData: true
                });
            }
            poll = window.setInterval(check, config.pollMs);
            timer = window.setTimeout(function () {
                finish({ ok: false, error: "Forge 가 Defaults Apply 에 응답하지 않았습니다" });
            }, config.defaultsTimeoutMs);
            try {
                applyButton.click();
            } catch (error) {
                finish({ ok: false, error: message(error) });
            }
        });
    }

    // ---- 2. 활성 UI Preset 의 설정 22개 ----------------------------------------------------------------

    function readResult() {
        var wrap = find("#" + RESULT_ID);
        var field = wrap && (wrap.matches("textarea, input") ? wrap : wrap.querySelector("textarea, input"));
        return field ? field.value : "";
    }

    function saveActivePreset() {
        var run = find("#" + RUN_ID);
        if (!run) {
            return Promise.resolve({
                ok: false,
                error: "프리셋 저장 연결이 없습니다 (Forge 콘솔의 [sam-extra] save defaults 기록을 확인하세요)"
            });
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
                finish(payload.ok ? payload : {
                    ok: false, preset: payload.preset, error: String(payload.error || "알 수 없는 오류")
                });
            }

            poll = window.setInterval(check, config.pollMs);
            timer = window.setTimeout(function () {
                finish({ ok: false, error: "Forge 가 프리셋 저장에 응답하지 않았습니다" });
            }, config.presetTimeoutMs);
            try {
                run.click();
            } catch (error) {
                finish({ ok: false, error: message(error) });
            }
        });
    }

    // ---- 결과 ----------------------------------------------------------------------------------------

    function changeLine(change) {
        function show(value) {
            if (Array.isArray(value)) {
                return value.map(function (item) {
                    return String(item).replace(/^.*[\\/]/, "");
                }).join(", ") || "없음";
            }
            return value === null || value === undefined ? "없음" : String(value);
        }
        return change.label + ": " + show(change.old) + " → " + show(change.new);
    }

    function summarize(defaults, preset) {
        var name = preset && preset.preset ? preset.preset : "UI";
        var defaultsCount = defaults.ok && typeof defaults.changed === "number" ? defaults.changed : 0;
        var presetCount = preset.ok ? Number(preset.changed) || 0 : 0;
        var detail = [];
        if (defaults.ok) detail.push("Settings → Defaults: " + (defaults.text || "저장됨"));
        else detail.push("Settings → Defaults: " + defaults.error);
        if (preset.ok) {
            detail.push(name + " 프리셋: 설정 " + preset.written + "개 저장, 바뀐 항목 " + presetCount + "개");
            (preset.changes || []).forEach(function (change) { detail.push("  " + changeLine(change)); });
            var kept = Array.isArray(preset.kept) ? preset.kept : [];
            if (kept.length) {
                // 0 = Forge 가 그 칸의 화면 값을 두는 값: 바꿔 쓰지 않았고 개수에도 넣지 않는다(화면 값은 ui-config 에)
                detail.push("  프리셋 값 0(화면 값을 덮어쓰지 않음)이라 그대로 둔 칸 " + kept.length + "개: "
                    + kept.map(function (item) { return item && item.label ? item.label : String(item); }).join(", "));
            }
        } else {
            detail.push("프리셋: " + preset.error);
        }
        var text;
        var tone = "error";
        if (defaults.ok && preset.ok) {
            tone = "saved";
            // ui-config 와 config.json 의 개수를 따로: 같은 칸(너비·CFG 등)이 두 파일에 다 쓰이므로 더하지 않는다
            text = "기본값 + " + name + " 프리셋 저장됨 · 바뀐 항목: "
                + (defaults.changed === null ? "" : "기본값 " + defaultsCount + "개, ")
                + "프리셋 " + presetCount + "개";
            if (defaults.changed === null) text += " (Defaults: " + defaults.text + ")";
        } else if (preset.ok) {
            text = name + " 프리셋 저장됨 (바뀐 항목 " + presetCount + "개) · 기본값 저장 실패: " + defaults.error;
        } else if (defaults.ok) {
            text = "기본값 저장됨 (바뀐 항목 " + defaultsCount + "개) · 프리셋 저장 실패: " + preset.error;
        } else {
            text = "저장 실패: " + defaults.error + " · " + preset.error;
        }
        return { text: text, tone: tone, detail: detail.join("\n") };
    }

    function setStatus(text, tone, detail) {
        var status = find("#" + STATUS_ID);
        if (!status) return;
        window.clearTimeout(statusTimer);
        status.textContent = text || "";
        status.setAttribute("data-tone", tone || "normal");
        if (detail) status.title = detail;
        else status.removeAttribute("title");
        var visibleMs = tone === "saved" ? config.savedVisibleMs : tone === "error" ? config.errorVisibleMs : 0;
        if (visibleMs > 0) {
            statusTimer = window.setTimeout(function () {
                status.textContent = "";
                status.setAttribute("data-tone", "normal");
                status.removeAttribute("title");
            }, visibleMs);
        }
    }

    // 저장하는 동안에도 버튼은 포커스를 지킨다: disabled 로 바꾸면 브라우저가 포커스를 <body> 로 보내 키보드 사용자가
    // 자리를 잃는다(다음 Tab 이 페이지 맨 위부터). 그래서 aria-disabled 로만 알리고, 두 번 눌림은 busy 가 막는다.
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

    async function saveDefaults() {
        if (busy) return null;
        busy = true;
        setButtonBusy(find("#" + WRAP_ID + " button"), true);
        setStatus("저장 중…", "busy");
        var summary;
        try {
            var defaults = await applyForgeDefaults().catch(function (error) {
                return { ok: false, error: message(error) };
            });
            var preset = await saveActivePreset().catch(function (error) {
                return { ok: false, error: message(error) };
            });
            summary = summarize(defaults, preset);
        } catch (error) {
            summary = { text: "저장 실패: " + message(error), tone: "error", detail: "" };
        } finally {
            busy = false;
            setButtonBusy(find("#" + WRAP_ID + " button"), false);
        }
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
        wrap.className = "sam3-save-defaults";
        var button = document.createElement("button");
        button.type = "button";
        button.className = "sam3-save-defaults-button";
        button.textContent = LABEL;
        button.title = TITLE;
        button.setAttribute("aria-describedby", STATUS_ID);
        button.addEventListener("click", function () { saveDefaults(); });
        var status = document.createElement("span");
        status.id = STATUS_ID;
        status.className = "sam3-save-defaults-status";
        status.setAttribute("role", "status");
        status.setAttribute("aria-live", "polite");
        status.setAttribute("data-tone", "normal");
        wrap.appendChild(button);
        wrap.appendChild(status);
        return wrap;
    }

    function ensureButton() {
        var row = tabRow();
        if (!row) return false;
        var wrap = find("#" + WRAP_ID);
        if (wrap && wrap.parentElement === row) return true;
        if (!wrap) wrap = build();
        row.appendChild(wrap);       // CSS 의 order 가 Forge 의 검색·정렬 칸보다 뒤(오른쪽 끝)에 둔다
        return true;
    }

    function boot() {
        ensureButton();
        if (!mountTimer) mountTimer = window.setInterval(ensureButton, config.mountPollMs);
    }

    window.addEventListener("sam3:notebook-mounted", function () { ensureButton(); });

    // 테스트 전용 입구. 실제 페이지는 이 객체를 만들지 않는다.
    if (window.__sam3SaveDefaultsTestHooks
            && typeof window.__sam3SaveDefaultsTestHooks === "object") {
        var hooks = window.__sam3SaveDefaultsTestHooks;
        hooks.config = config;
        hooks.ensureButton = ensureButton;
        hooks.saveDefaults = saveDefaults;
        hooks.parseDefaultsResult = parseDefaultsResult;
        hooks.summarize = summarize;
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(function () { window.setTimeout(boot, 300); });
    } else {
        document.addEventListener("DOMContentLoaded", function () {
            window.setTimeout(boot, 1300);
        });
    }
})();
