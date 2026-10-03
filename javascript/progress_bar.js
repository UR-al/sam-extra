/* SAM Extra 진행 막대 — txt2img·img2img 생성 진행률을 부드럽게 그린다 (Settings → SAM Extra Progress Bar).
 *
 * diamfang/sd-webui-smooth-progress@7fe58101ae315af1b5e6eda6080e72ac45bfe846 (코드는 2b966fc 와 같다)
 * javascript/smooth-progress.js 를 옮긴 것이다. 서버 쪽은 sam3ext/progress_api.py, 설정은 scripts/appearance_progress_bar.py.
 * 원본 라이선스:
 *
 *   MIT License
 *
 *   Copyright (c) 2026 diamfang
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 *
 * 이 확장에서 바꾼 것 (sam-extra, 2026-10-03):
 * - 시작과 끝: 상류는 자기 경로를 100ms 마다 영원히 묻고, 'generate' 처럼 보이는 클릭으로 시작을 짐작했다(그래서
 *   생성이 아닌 클릭 뒤 0/0 에 멈췄다). 여기서는 Forge 의 requestProgress(javascript/progressbar.js:77)를 감싼다.
 *   인자 여섯 개를 그대로 넘기고 원래 atEnd·onProgress 도 그대로 부르며, 그 페이지의 그 탭이 시작한 작업만
 *   시작부터 끝까지 GET /sam-extra/progress?id_task= 를 묻는다. 페이지가 숨어 있으면 묻지 않고, 다시 보이면
 *   Smooth > Accurate 막대를 서버 진행률로 바로 옮긴다(상류는 숨은 동안 멈춘 자리에서 ETA 끝까지 천천히 따라잡았다).
 * - 전체 작업: 막대와 % 는 Forge 기본 막대처럼 job_no/job_count 를 합친 작업 전체 진행률, ETA 는 작업 전체의 남은
 *   시간이다. Steps 는 지금 패스의 스텝이고 패스가 여럿이면 [2/4] 를 앞에 붙인다. 대기 중에는 Forge 의 대기열 글자
 *   (res.textinfo)를 보여 준다.
 * - 글자: ETA 는 상류의 초 단위('123s') 대신 Forge 기본 막대와 같은 초 / 분:초 / 시:분:초('02:03'). 중단 글자는
 *   상류의 'Interrupted'(글자 없음 형식이면 '❌') 대신 어느 형식에서나 '중단됨'. 스텝 수는 이 작업의 첫 스텝을 본
 *   뒤에야 쓴다(그 전 값은 지난 작업의 것일 수 있다 — 상류는 '0/0' 을 보였다).
 * - 중단: 돌고 있는 것을 본 작업의 Interrupt·Esc 는 바로 멈춘 자리에 '중단됨'. 시작 전의 클릭은 Forge 의 State.begin 이
 *   지우므로 중단으로 치지 않고, 시작 뒤에 닿은 중단은 서버의 interrupted 표시로 안다. Skip 은 중단이 아니다.
 * - 설정: ⚙️ 팝오버와 localStorage 대신 Forge Settings(sam3_progress_*), 저장하면 바로 반영한다. 높이는 10~50px.
 * - 모양: 색은 테마 변수(tokens.css·Gradio). 그라데이션·움직이는 색·빛 번짐(glow, drop-shadow) 없음. 막대 위 글자는
 *   두 겹(바탕용·채움용)으로 그려 어느 쪽에서도 읽힌다. 각 겹은 처음 만든 텍스트 노드의 .data 만 바꾼다 — 자식을
 *   넣고 빼지 않으므로 Forge 의 childList 관찰자(onUiUpdate)가 깨지 않는다. prefers-reduced-motion 이면 보간
 *   애니메이션 없이 실제 값으로만 움직인다.
 * - Forge 기본 막대(.progressDiv)는 지우지 않고 CSS 로 숨긴다 — sd-webui-api-payload-display 가 그 삽입을 기다린다.
 *   이 막대가 맡아 그리는 작업을 Forge 의 requestProgress 가 아직 붙들고 있는 동안만 숨기므로(data-native) 설정을
 *   켜기 전에 시작한 작업은 Forge 막대가 그대로 보이고, 사라지기 방식이면 Forge 가 작업을 놓을 때까지 끝 상태를 보인다.
 * - 상류 버그: 중단 표시 슬라이더가 4가지 중 3가지만 골랐던 것, 생성 아닌 클릭 뒤 0/0 에 멈추던 것, 높이 미제한,
 *   보호 없는 localStorage, ETA 를 모를 때(첫 스텝 전) Smooth > Accurate 가 99% 로 내달리던 것.
 * - sd-webui-smooth-progress 가 함께 설치돼 있으면(#spb-dynamic-css) 이 막대는 물러난다.
 */
(function () {
    "use strict";

    if (window.__sam3ProgressBarLoaded) return;
    window.__sam3ProgressBarLoaded = true;

    var API_PATH = "/sam-extra/progress";
    var ROOT_ATTR = "data-sam3-progress";
    var UPSTREAM_STYLE_ID = "spb-dynamic-css";
    var TABS = ["txt2img", "img2img"];
    var POLL_MS = 200;                 // 상류는 100ms 로 늘. 여기서는 작업 하나가 도는 동안만
    var RETRY_MS = 1000;
    var MAX_FAILURES = 5;              // 이만큼 잇달아 실패하면 그 작업은 Forge 응답(onProgress)만으로 그린다
    var DEFAULT_INACTIVITY_S = 40;     // requestProgress 의 inactivityTimeout 기본값
    var CAP_PCT = 99.2;                // 상류: 끝 신호 전에는 99.2% 에서 멈춘다
    var COMPLETE_MS = 300;             // 상류 .spb-smooth-complete: width 0.3s ease-out
    var DONE_HOLD_MS = 750;            // 상류 completeAnimation: 300 + 300 + 150ms 뒤 끝난 뒤 동작(빛 번짐 펄스는 뺐다)
    var INTERRUPT_HOLD_MS = 400;       // 상류 triggerInterruptedState: 400ms 뒤
    var STOPPED_TEXT = "중단됨";
    var WAITING_TEXT = "Waiting...";   // Forge progressapi 가 아직 모르는 작업에 쓰는 글자

    // Settings → SAM Extra Progress Bar. 기본값·선택지는 sam3ext/progress_api.py 와 같아야 한다
    // (tests/test_progress_api.py 가 두 파일을 대조한다).
    var DEFAULTS = Object.freeze({
        sam3_progress_enabled: false,
        sam3_progress_smoothness: "smooth_gt_acc",
        sam3_progress_text_format: "steps_pct_eta",
        sam3_progress_text_align: 50,
        sam3_progress_after_finish: "fade",
        sam3_progress_fade_seconds: 0.4,
        sam3_progress_interrupt_style: "red_text_red_bar",
        sam3_progress_height: 20,
        sam3_progress_color: "accent",
        sam3_progress_custom_color: "#ef256c",
    });
    var CHOICES = Object.freeze({
        sam3_progress_smoothness: ["smooth_gt_acc", "smooth_eq_acc", "smooth_lt_acc"],
        sam3_progress_text_format: ["steps_pct_eta", "steps_eta", "steps_pct", "pct_eta", "eta_only", "steps_only", "pct_only", "none"],
        sam3_progress_after_finish: ["fade", "fade_text_only", "keep"],
        sam3_progress_interrupt_style: ["text", "red_text", "text_red_bar", "red_text_red_bar"],
        sam3_progress_color: ["accent", "success", "blue", "green", "red", "dandelion", "custom"],
    });
    var RANGES = Object.freeze({
        sam3_progress_text_align: [0, 100],
        sam3_progress_fade_seconds: [0.1, 4.0],
        sam3_progress_height: [10, 50],
    });
    var PRESET_COLORS = Object.freeze({
        blue: "#2563eb",
        green: "#059669",
        red: "#dc2626",
        dandelion: "#FEDF08",
    });

    var config = null;
    var configKey = "";
    var original = null;               // 감싸기 전 requestProgress
    var wrapper = null;
    var listening = false;
    var steppedAside = false;
    var uiReady = false;
    var bars = Object.create(null);    // 탭 → 막대
    var frame = 0;
    var routeMissing = false;          // 404: 경로가 없는 Forge(확장 Python 이 못 올라옴) — Forge 응답만으로 그린다
    var reduced = false;
    var motionQuery = null;

    // ---- 작은 도우미 -----------------------------------------------------------

    function now() {
        return window.performance && typeof window.performance.now === "function"
            ? window.performance.now() : Date.now();
    }

    function app() {
        try {
            if (typeof gradioApp === "function") return gradioApp() || document;
        } catch (_error) {
            // gradioApp 는 Gradio 가 붙기 전에는 document 를 돌려준다. 그 밖의 실패도 document 로.
        }
        return document;
    }

    function clamp(value, low, high) {
        return Math.min(high, Math.max(low, value));
    }

    function finite(value) {
        var number = typeof value === "number" ? value : parseFloat(value);
        return isFinite(number) ? number : null;
    }

    function int(value) {
        var number = finite(value);
        return number === null ? 0 : Math.trunc(number);
    }

    function singleLine(text) {
        return typeof text === "string" && text && text.indexOf("\n") === -1 ? text : "";
    }

    function pad2(value) {
        return value < 10 ? "0" + value : String(value);
    }

    // Forge 기본 막대의 시간 표기(초 / 분:초 / 시:분:초)와 같은 모양.
    function formatTime(seconds) {
        if (seconds > 3600) {
            return pad2(Math.floor(seconds / 3600)) + ":" + pad2(Math.floor(seconds / 60) % 60) + ":"
                + pad2(Math.floor(seconds) % 60);
        }
        if (seconds > 60) return pad2(Math.floor(seconds / 60)) + ":" + pad2(Math.floor(seconds) % 60);
        return Math.floor(seconds) + "s";
    }

    // ---- 설정 ------------------------------------------------------------------

    function optionSource() {
        try {
            if (typeof opts !== "undefined" && opts && typeof opts === "object") return opts;
        } catch (_error) {
            // opts 는 script.js 의 전역 let — 아주 이른 시점에는 없을 수 있다.
        }
        return {};
    }

    function pick(source, key) {
        var value = source[key];
        return CHOICES[key].indexOf(value) === -1 ? DEFAULTS[key] : value;
    }

    function number(source, key, round) {
        var value = finite(source[key]);
        if (value === null) value = DEFAULTS[key];
        value = clamp(value, RANGES[key][0], RANGES[key][1]);
        return round ? Math.round(value) : value;
    }

    function hexColor(value) {
        if (typeof value !== "string") return null;
        var text = value.trim();
        var short = /^#([0-9a-f])([0-9a-f])([0-9a-f])$/i.exec(text);
        if (short) return ("#" + short[1] + short[1] + short[2] + short[2] + short[3] + short[3]).toLowerCase();
        return /^#[0-9a-f]{6}$/i.test(text) ? text.toLowerCase() : null;
    }

    function readConfig() {
        var source = optionSource();
        return {
            enabled: source.sam3_progress_enabled === true,
            smoothness: pick(source, "sam3_progress_smoothness"),
            format: pick(source, "sam3_progress_text_format"),
            align: number(source, "sam3_progress_text_align", true),
            afterFinish: pick(source, "sam3_progress_after_finish"),
            fadeSeconds: Math.round(number(source, "sam3_progress_fade_seconds", false) * 100) / 100,
            interruptStyle: pick(source, "sam3_progress_interrupt_style"),
            height: number(source, "sam3_progress_height", true),
            color: pick(source, "sam3_progress_color"),
            customColor: hexColor(source.sam3_progress_custom_color)
                || DEFAULTS.sam3_progress_custom_color,
        };
    }

    // WCAG 상대 휘도. 밝은 막대에는 어두운 글자, 어두운 막대에는 밝은 글자.
    function isLight(hex) {
        var channels = [1, 3, 5].map(function (offset) {
            var c = parseInt(hex.slice(offset, offset + 2), 16) / 255;
            return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
        });
        var luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
        return luminance > 0.179;
    }

    function colorVars(cfg) {
        if (cfg.color === "accent") return null;   // style.css 의 기본값: 테마 강조색
        if (cfg.color === "success") {
            return { color: "var(--sam3-color-success)", ink: "var(--sam3-color-accent-ink)" };
        }
        var hex = cfg.color === "custom" ? cfg.customColor : hexColor(PRESET_COLORS[cfg.color]);
        return {
            color: hex,
            ink: isLight(hex) ? "var(--sam3-color-accent-ink)" : "var(--sam3-color-ink)",
        };
    }

    function interruptAfterCurrent() {
        return optionSource().interrupt_after_current === true;
    }

    // ---- 글자 (상류 updateOverlayTextFormatted) -----------------------------------

    // input: { format, phase: waiting|running|done|interrupted, pct, eta(초|null), step, steps, jobNo, jobCount,
    //          waitText, textinfo }
    function formatText(input) {
        var format = input.format;
        var phase = input.phase;
        if (phase === "waiting") return format === "none" ? "" : (input.waitText || "");
        var interrupted = phase === "interrupted";
        var done = phase === "done";
        if (format === "none" && !interrupted) return "";

        var pct = done ? 100 : clamp(Math.round(finite(input.pct) || 0), 0, 100);
        var pctText = pct + "%";
        var steps = "";
        if (input.steps > 0) {
            steps = clamp(input.step, 0, input.steps) + "/" + input.steps;
            if (input.jobCount > 1) {
                steps = "[" + clamp(input.jobNo + 1, 1, input.jobCount) + "/" + input.jobCount + "] " + steps;
            }
        }
        var etaText = input.eta !== null && input.eta > 0 ? formatTime(input.eta) : "?";
        var parts;
        if (interrupted) {
            switch (format) {
                case "steps_pct_eta":
                case "steps_pct":
                    parts = [steps, pctText, STOPPED_TEXT];
                    break;
                case "steps_eta":
                case "steps_only":
                    parts = [steps, STOPPED_TEXT];
                    break;
                case "pct_eta":
                case "pct_only":
                    parts = [pctText, STOPPED_TEXT];
                    break;
                default:   // eta_only, none — 색이 아닌 신호(글자)는 늘 남긴다
                    parts = [STOPPED_TEXT];
            }
            return parts.filter(Boolean).join(" • ");
        }
        var hasEta = pct < 100 && !done;
        switch (format) {
            case "steps_pct_eta":
                parts = hasEta ? [steps, pctText, etaText] : [steps, pctText];
                break;
            case "steps_eta":
                parts = hasEta ? [steps, etaText] : [steps];
                break;
            case "steps_pct":
                parts = [steps, pctText];
                break;
            case "pct_eta":
                parts = hasEta ? [pctText, etaText] : [pctText];
                break;
            case "eta_only":
                parts = [hasEta ? etaText : "0s"];
                break;
            case "steps_only":
                parts = [steps];
                break;
            default:
                parts = [pctText];
        }
        if (!done && input.textinfo) parts.unshift(input.textinfo);
        return parts.filter(Boolean).join(" • ");
    }

    // ---- 막대 DOM ----------------------------------------------------------------

    function makeLabel() {
        var el = document.createElement("span");
        el.className = "sam3-progress-label";
        el.setAttribute("aria-hidden", "true");
        var text = document.createTextNode("");
        el.appendChild(text);
        return { el: el, text: text };
    }

    function buildBar(tab) {
        var root = document.createElement("div");
        root.id = "sam3_progress_" + tab;
        root.className = "sam3-progress";
        root.setAttribute("role", "progressbar");
        root.setAttribute("aria-label", "생성 진행률");
        root.setAttribute("aria-valuemin", "0");
        root.setAttribute("aria-valuemax", "100");
        root.setAttribute("aria-valuenow", "0");
        root.setAttribute("data-tab", tab);
        root.setAttribute("data-state", "idle");
        root.setAttribute("data-fill", "0");
        root.setAttribute("data-text", "0");
        root.setAttribute("data-native", "show");  // style.css: "hide" 일 때만 Forge 기본 막대를 숨긴다(syncNative)
        var under = makeLabel();                   // 바탕 위 글자
        var fill = document.createElement("div");  // 채움(clip-path 로 잘린다)과 그 위 글자
        fill.className = "sam3-progress-fill";
        fill.setAttribute("aria-hidden", "true");
        var over = makeLabel();
        fill.appendChild(over.el);
        root.appendChild(under.el);
        root.appendChild(fill);
        return {
            tab: tab,
            root: root,
            fill: fill,
            texts: [under.text, over.text],
            job: null,                 // 보이는 작업
            backlog: [],               // 같은 탭에서 겹친 다른 작업(드묾)
            finish: null,              // null | "done" | "interrupted"
            awaitingForge: false,      // 사라지기 방식: 끝 상태를 보이며 Forge 가 이 작업을 놓기를 기다린다
            visual: 0,
            speed: 0,
            lastDesired: null,
            lastFrame: 0,
            completing: null,
            timers: [],
            text: null,
            value: null,
            ariaNow: "0",
        };
    }

    function anchorFor(tab) {
        var root = app();
        return root.querySelector("#" + tab + "_results_panel")
            || root.querySelector("#" + tab + "_gallery_container");
    }

    function ensureBar(tab) {
        var bar = bars[tab];
        if (bar && bar.root.isConnected) return bar;
        var anchor = anchorFor(tab);
        if (!anchor || !anchor.parentNode) return null;
        if (!bar) {
            var stale = document.getElementById("sam3_progress_" + tab);
            if (stale && stale.parentNode) stale.parentNode.removeChild(stale);
            bar = buildBar(tab);
            bars[tab] = bar;
        }
        // 마운트(또는 Gradio 가 떼어 낸 뒤 다시 붙이기) 때 한 번만 넣는다. 그 뒤로는 자식을 넣고 빼지 않는다.
        anchor.parentNode.insertBefore(bar.root, anchor);
        styleBar(bar);
        if (!bar.job) showIdle(bar);
        return bar;
    }

    function setAttr(bar, name, value) {
        if (bar.root.getAttribute(name) !== value) bar.root.setAttribute(name, value);
    }

    function setState(bar, state) {
        setAttr(bar, "data-state", state);
    }

    // Forge 기본 막대를 숨길지: 이 막대가 맡은 작업(그리는 것과 그 뒤에 기다리는 것) 가운데 Forge 의 requestProgress 가
    // 아직 끝나지 않은 것이 있을 때만. 맡지 않은 작업(설정을 켜기 전에 시작한 작업 등)은 Forge 막대가 그대로 보인다.
    function syncNative(bar) {
        var alive = function (job) { return !!job && !job.nativeEnded; };
        setAttr(bar, "data-native", alive(bar.job) || bar.backlog.some(alive) ? "hide" : "show");
    }

    function setVisible(bar, fill, text) {
        setAttr(bar, "data-fill", fill ? "1" : "0");
        setAttr(bar, "data-text", text ? "1" : "0");
    }

    function setValue(bar, pct) {
        var value = Math.round(clamp(pct, 0, 100) * 100) / 100;
        if (bar.value !== value) {
            bar.root.style.setProperty("--sam3-progress-value", String(value));
            bar.value = value;
        }
        var aria = String(Math.round(value));
        if (bar.ariaNow !== aria) {
            bar.root.setAttribute("aria-valuenow", aria);
            bar.ariaNow = aria;
        }
    }

    function setText(bar, text) {
        if (bar.text === text) return;
        bar.texts.forEach(function (node) { node.data = text; });
        bar.text = text;
        if (text) bar.root.setAttribute("aria-valuetext", text);
        else bar.root.removeAttribute("aria-valuetext");
    }

    // 상류는 transition 을 끄고 reflow 로 0% 를 바로 적용했다. 같은 방식으로 새 작업은 즉시 보인다.
    function instant(bar, change) {
        bar.root.setAttribute("data-instant", "");
        change();
        void bar.root.offsetWidth;
        bar.root.removeAttribute("data-instant");
    }

    function styleBar(bar) {
        var style = bar.root.style;
        style.setProperty("--sam3-progress-height", config.height + "px");
        style.setProperty("--sam3-progress-align", String(config.align));
        style.setProperty("--sam3-progress-fade", config.fadeSeconds + "s");
        var colors = colorVars(config);
        if (colors) {
            style.setProperty("--sam3-progress-color", colors.color);
            style.setProperty("--sam3-progress-color-ink", colors.ink);
        } else {
            style.removeProperty("--sam3-progress-color");
            style.removeProperty("--sam3-progress-color-ink");
        }
        setAttr(bar, "data-interrupt", config.interruptStyle);
    }

    function clearTimers(bar) {
        bar.timers.forEach(function (id) { window.clearTimeout(id); });
        bar.timers = [];
    }

    function later(bar, ms, fn) {
        var id = window.setTimeout(function () {
            bar.timers = bar.timers.filter(function (other) { return other !== id; });
            fn();
        }, ms);
        bar.timers.push(id);
    }

    // 아무 작업도 없을 때. 상류 restoreLastState: '그대로 둠'·'글자만 숨김' 은 처음부터 채워진 막대를 둔다.
    function showIdle(bar) {
        bar.job = null;
        bar.finish = null;
        bar.completing = null;
        bar.awaitingForge = false;
        syncNative(bar);
        var full = config.afterFinish !== "fade";
        instant(bar, function () {
            setState(bar, "idle");
            setVisible(bar, full, false);
            bar.visual = full ? 100 : 0;
            setValue(bar, bar.visual);
        });
        setText(bar, "");
    }

    function render(bar, eta, pct) {
        var job = bar.job;
        if (!job) {
            setText(bar, "");
            return;
        }
        var phase = bar.finish || (job.phase === "running" ? "running" : "waiting");
        // State.begin 은 sampling_steps 를 지우지 않는다: 첫 패스의 첫 스텝 전에는 지난 작업의 스텝 수일 수 있다.
        // 끝났을 때도 마찬가지 — 첫 스텝을 본 적이 없으면(아주 빠른 작업) 그 수는 지난 작업의 것일 수 있다.
        var stepsKnown = job.steps > 0 && job.stepsSeen;
        setText(bar, formatText({
            format: config.format,
            phase: phase,
            pct: pct === undefined ? bar.visual : pct,
            eta: eta === undefined ? job.eta : eta,
            step: job.step,
            steps: stepsKnown ? job.steps : 0,
            jobNo: job.jobNo,
            jobCount: job.jobCount,
            waitText: job.waitText,
            textinfo: job.textinfo,
        }));
    }

    // ---- 작업 -------------------------------------------------------------------

    function createJob(bar, id, timeout) {
        var seconds = finite(timeout);
        return {
            id: id,
            bar: bar,
            phase: "waiting",
            ended: false,
            nativeEnded: false,
            begunAt: now(),
            timeout: seconds === null ? DEFAULT_INACTIVITY_S : seconds,
            everActive: false,
            completedSeen: false,
            queued: null,
            waitText: "",
            textinfo: "",
            progress: 0,
            step: 0,
            steps: 0,
            stepsSeen: false,       // 첫 스텝(또는 다음 패스)을 본 뒤에야 steps 가 이 작업의 것이다
            jobNo: 0,
            jobCount: 0,
            perImage: 1,
            eta: null,
            passEta: null,
            updatedAt: 0,
            interrupt: null,        // null | "stop"(이 이미지까지) | "hard"
            stopSeen: false,
            pollTimer: 0,
            inFlight: false,        // 묻는 중 — 묻기는 한 줄로만 이어진다
            pausedHidden: false,
            resync: false,          // 페이지가 다시 보였다 — 다음 응답에서 막대를 서버 값에 맞춘다(resync)
            failures: 0,
            useNative: false,
        };
    }

    function shown(job) {
        return job.bar.job === job && bars[job.bar.tab] === job.bar;
    }

    function showJob(bar, job) {
        clearTimers(bar);
        bar.backlog = bar.backlog.filter(function (other) { return other !== job; });
        bar.job = job;
        bar.finish = null;
        bar.completing = null;
        bar.awaitingForge = false;
        syncNative(bar);           // Forge 가 막대를 넣기 전에(감싸기가 원래 함수를 부르기 전) 숨김이 걸린다
        bar.visual = 0;
        bar.speed = 0;
        bar.lastDesired = null;
        bar.lastFrame = 0;
        instant(bar, function () {
            setState(bar, job.phase === "running" ? "running" : "waiting");
            setVisible(bar, true, true);
            setValue(bar, 0);
        });
        render(bar);
        if (job.phase === "running") onJobData(job);
        schedulePoll(job, 0);
    }

    function beginJob(id, container, timeout) {
        if (!config || !config.enabled || steppedAside) return null;
        if (upstreamPresent()) {
            stepAside();
            return null;
        }
        var match = container && container.nodeType === 1
            ? /^(txt2img|img2img)_gallery_container$/.exec(container.id || "") : null;
        if (!match) return null;
        var bar = ensureBar(match[1]);
        if (!bar) return null;
        var job = createJob(bar, String(id), timeout);
        if (bar.job && !bar.job.ended) {
            bar.backlog.push(job);
            syncNative(bar);
        } else {
            showJob(bar, job);
        }
        return job;
    }

    function cancelPoll(job) {
        if (job.pollTimer) {
            window.clearTimeout(job.pollTimer);
            job.pollTimer = 0;
        }
    }

    function routeUsable(job) {
        return !job.useNative && !routeMissing;
    }

    function schedulePoll(job, delay) {
        if (job.ended || !routeUsable(job) || job.pollTimer || job.inFlight || !shown(job)) return;
        job.pollTimer = window.setTimeout(function () {
            job.pollTimer = 0;
            poll(job);
        }, delay);
    }

    function request(id) {
        var url = API_PATH + (id ? "?id_task=" + encodeURIComponent(id) : "");
        var pending;
        try {
            pending = window.fetch(url, {
                method: "GET",
                credentials: "same-origin",
                cache: "no-store",
                headers: { "X-SAM3-Notebook": "1" },
            });
        } catch (error) {
            return Promise.reject(error);
        }
        return Promise.resolve(pending).then(function (response) {
            if (response.status === 404 || response.status === 405) {
                var missing = new Error("progress route missing");
                missing.missing = true;
                throw missing;
            }
            if (!response.ok) throw new Error("HTTP " + response.status);
            return response.json();
        });
    }

    function poll(job) {
        if (job.ended || !routeUsable(job) || job.inFlight || !shown(job)) return;
        if (document.hidden) {
            job.pausedHidden = true;   // visibilitychange 가 다시 시작한다
            return;
        }
        job.inFlight = true;
        request(job.id).then(function (data) {
            job.inFlight = false;
            job.failures = 0;
            try {
                applySnapshot(job, data);
            } catch (error) {
                console.error("[SAM Extra] progress bar", error);
            }
            schedulePoll(job, POLL_MS);
        }, function (error) {
            job.inFlight = false;
            if (job.ended) return;
            if (error && error.missing) {
                if (!routeMissing) {
                    console.warn("[SAM Extra] " + API_PATH + " 가 없습니다 — Forge 기본 진행 응답만으로 그립니다.");
                }
                routeMissing = true;
            } else {
                job.failures += 1;
                if (job.failures >= MAX_FAILURES) job.useNative = true;
            }
            if (!routeUsable(job)) {
                if (job.nativeEnded) endWithoutRoute(job);
                return;
            }
            schedulePoll(job, RETRY_MS);
        });
    }

    function queueText(position, size) {
        var at = int(position);
        var of = int(size);
        return at > 0 && of > 0 ? "In queue: " + at + "/" + of : WAITING_TEXT;
    }

    function applySnapshot(job, data) {
        if (job.ended || !data || typeof data !== "object") return;
        if (data.active) {
            job.everActive = true;
            job.queued = false;
            job.phase = "running";
            job.progress = clamp(finite(data.progress) || 0, 0, 1);
            job.step = int(data.step);
            job.steps = int(data.steps);
            job.jobNo = int(data.job_no);
            if (job.step > 0 || job.jobNo > 0) job.stepsSeen = true;
            job.jobCount = int(data.job_count);
            job.perImage = Math.max(1, int(data.passes_per_image));
            job.eta = finite(data.eta);
            job.passEta = finite(data.pass_eta);
            job.textinfo = singleLine(data.textinfo);
            job.updatedAt = now();
            if (data.interrupted) {
                // 상류처럼 플래그를 보는 즉시 멈춘 자리에 '중단됨' — 작업이 마무리(디코드·저장)되기를 기다리지 않는다.
                job.interrupt = "hard";
                finishJob(job);
                return;
            }
            if (data.stopping) {
                job.stopSeen = true;
                if (job.interrupt !== "hard") job.interrupt = "stop";
            }
            if (job.resync) resync(job);
            onJobData(job);
            return;
        }
        if (data.completed) {
            job.completedSeen = true;
            finishJob(job);
            return;
        }
        if (job.everActive) {   // 돌던 작업이 더는 돌지도, 기다리지도 않는다(Forge 기본 막대와 같은 판단)
            finishJob(job);
            return;
        }
        if (data.queued !== true && job.nativeEnded) {
            abandonJob(job);    // 한 번도 돌지 않았고 Forge 도 기다리기를 그만뒀다(시간 초과·요청 실패)
            return;
        }
        job.phase = "waiting";
        job.queued = data.queued === true;
        job.waitText = job.queued ? queueText(data.queue_position, data.queue_size) : (job.waitText || WAITING_TEXT);
        if (shown(job)) render(job.bar);
    }

    // Forge 의 /internal/progress 응답(requestProgress 의 onProgress). 대기열 글자는 늘 여기서 받는다.
    function nativeProgress(job, res) {
        if (job.ended || !res || typeof res !== "object") return;
        var bar = job.bar;
        if (res.active) {
            job.everActive = true;
            if (!shown(job) && bars[bar.tab] === bar && (!bar.job || bar.job.phase !== "running")) {
                if (bar.job && !bar.job.ended) bar.backlog.unshift(bar.job);
                showJob(bar, job);   // 기다리던 작업 대신 실제로 도는 작업을 보인다
            }
            if (job.useNative || routeMissing) {
                job.queued = false;
                job.phase = "running";
                job.progress = clamp(finite(res.progress) || 0, 0, 1);
                var eta = finite(res.eta);
                job.eta = eta !== null && eta > 0 ? eta : null;
                job.passEta = null;
                job.textinfo = singleLine(res.textinfo);
                job.updatedAt = now();
                if (job.resync) resync(job);
                onJobData(job);
            }
            return;
        }
        if (job.phase !== "running") {
            job.queued = res.queued === true;
            if (typeof res.textinfo === "string" && res.textinfo) job.waitText = res.textinfo;
            if (shown(job)) render(bar);
        }
    }

    // Forge 의 Interrupt 와 같은 판단(modules/ui_toprow.py interrupt_function): 여러 장이고 "Don't Interrupt in
    // the middle" 이 켜져 있으면 첫 클릭은 이 이미지까지만, 그다음은 즉시.
    function noteInterrupt(job) {
        if (job.ended || job.queued === true) return;   // 대기 중이면 그 클릭은 지금 도는 다른 작업을 멈춘다
        // 돌고 있는 것을 아직 못 본 작업: 시작 전이면 Forge 의 State.begin 이 그 중단을 지운다(작업은 끝까지 돈다).
        // 시작 직후에 닿은 중단이면 서버의 interrupted·stopping 표시로 다음 응답에서 안다.
        if (!job.everActive) return;
        if (!job.stopSeen && job.jobCount > 1 && interruptAfterCurrent()) {
            job.stopSeen = true;                         // 이 이미지까지 — 끝날 때 정한다
            if (job.interrupt !== "hard") job.interrupt = "stop";
            return;
        }
        job.interrupt = "hard";
        if (shown(job)) finishJob(job);                  // 멈춘 자리에 바로 표시한다
    }

    function finalKind(job) {
        if (job.interrupt === "hard") return "interrupted";
        if (job.interrupt === "stop") {
            // 이 이미지를 끝내고 멈춘다 — 마지막 이미지였으면 다 만든 것이다.
            if (job.jobCount > 0 && job.jobNo >= job.jobCount - job.perImage) return "done";
            return "interrupted";
        }
        return "done";
    }

    function nextFromBacklog(bar) {
        while (bar.backlog.length) {
            var next = bar.backlog.shift();
            if (!next.ended) {
                showJob(bar, next);
                return true;
            }
        }
        return false;
    }

    function finishJob(job) {
        if (job.ended) return;
        job.ended = true;
        cancelPoll(job);
        var bar = job.bar;
        if (!shown(job)) {
            bar.backlog = bar.backlog.filter(function (other) { return other !== job; });
            syncNative(bar);
            return;
        }
        if (nextFromBacklog(bar)) return;
        var kind = finalKind(job);
        bar.finish = kind;
        clearTimers(bar);
        if (kind === "done") {
            if (job.steps > 0) job.step = job.steps;                       // 상류: currentStep = totalSteps
            if (job.jobCount > 0) job.jobNo = job.jobCount - 1;
            job.progress = 1;
            setState(bar, "done");
            setVisible(bar, true, true);
            render(bar, null, 100);
            if (reduced || document.hidden) {
                bar.visual = 100;
                setValue(bar, 100);
            } else {
                bar.completing = { from: bar.visual, start: now() };
                ensureFrames();
            }
            later(bar, DONE_HOLD_MS, function () { afterFinish(bar); });
        } else {
            bar.completing = null;
            setState(bar, "interrupted");
            setVisible(bar, true, true);
            render(bar, null);
            later(bar, INTERRUPT_HOLD_MS, function () { afterFinish(bar); });
        }
    }

    // 한 번도 돌지 않은 작업이 끝났다: 서버가 완료 목록에 넣었으면(아주 빠른 작업) 끝난 것, 아니면 버린다.
    function abandonJob(job) {
        if (job.ended) return;
        job.ended = true;
        cancelPoll(job);
        var bar = job.bar;
        if (!shown(job)) {
            bar.backlog = bar.backlog.filter(function (other) { return other !== job; });
            syncNative(bar);
            return;
        }
        clearTimers(bar);
        if (!nextFromBacklog(bar)) showIdle(bar);
    }

    // requestProgress 의 atEnd: Forge 가 이 작업을 끝냈다 — 완료, 시작 전 시간 초과, 또는 Forge 쪽 요청 실패.
    // 어느 것인지는 서버에 한 번 더 물어 정한다(applySnapshot): 완료 목록에 있으면 끝(아주 빠른 작업 포함), 아직
    // 돌면 끝날 때까지 계속 그리고, 돈 적이 없고 기다리지도 않으면 버린다.
    function nativeEnd(job) {
        // Forge 의 requestProgress 가 이 작업을 놓았다 — 이 막대가 이미 끝낸 작업이어도 숨김과 사라지기를 다시 정한다.
        job.nativeEnded = true;
        var bar = job.bar;
        if (bars[bar.tab] === bar) {
            syncNative(bar);
            if (bar.job === job && bar.finish && bar.awaitingForge) afterFinish(bar);
        }
        if (job.ended) return;
        if (job.completedSeen) {
            finishJob(job);
            return;
        }
        if (!routeUsable(job) || !shown(job)) {
            endWithoutRoute(job);
            return;
        }
        cancelPoll(job);
        poll(job);   // 묻는 중이면 그 답이 정한다
    }

    // 서버에 물을 수 없을 때: 돈 적이 있으면 끝, Forge 의 대기 시간 초과로 끝났으면 버림, 아니면 끝.
    function endWithoutRoute(job) {
        if (job.ended) return;
        if (job.everActive) {
            finishJob(job);
            return;
        }
        var waited = (now() - job.begunAt) / 1000;
        if (job.timeout > 0 && waited >= job.timeout && job.queued !== true) abandonJob(job);
        else finishJob(job);
    }

    function afterFinish(bar) {
        bar.awaitingForge = false;
        if (!bar.finish) return;
        var mode = config.afterFinish;
        if (mode === "keep") {
            setVisible(bar, true, true);
            return;
        }
        if (mode === "fade_text_only") {
            setVisible(bar, true, false);
            return;
        }
        if (bar.job && !bar.job.nativeEnded) {
            // Forge 가 아직 이 작업을 붙들고 있다(중단 뒤 마무리 등). 지금 숨기면 그동안 Forge 막대가 다시 나오므로 끝 상태를
            // 그대로 그리다가, Forge 의 requestProgress 가 끝나면(nativeEnd) 숨긴다.
            bar.awaitingForge = true;
            setVisible(bar, true, true);
            return;
        }
        setVisible(bar, false, false);
        later(bar, config.fadeSeconds * 1000, function () {
            if (bar.finish) showIdle(bar);
        });
    }

    // ---- 움직임 (상류 animate) -------------------------------------------------------

    function onJobData(job) {
        if (!shown(job)) return;
        var bar = job.bar;
        if (bar.finish) return;
        setState(bar, "running");
        if (reduced) {
            // 움직임 줄이기: 보간 없이 서버 값 그대로(묻기마다 한 번). 앞으로만 간다.
            bar.visual = Math.max(bar.visual, Math.min(job.progress * 100, CAP_PCT));
            setValue(bar, bar.visual);
            render(bar, job.eta, bar.visual);
            return;
        }
        ensureFrames();
    }

    function ensureFrames() {
        if (frame || reduced) return;
        frame = window.requestAnimationFrame(tick);
    }

    function tick(time) {
        frame = 0;
        var again = false;
        TABS.forEach(function (tab) {
            var bar = bars[tab];
            if (bar && stepBar(bar, time)) again = true;
        });
        if (again) frame = window.requestAnimationFrame(tick);
    }

    function remainingSeconds(job, at) {
        if (job.eta === null) return null;
        return Math.max(0, job.eta - (at - job.updatedAt) / 1000);
    }

    function stepBar(bar, time) {
        if (bar.completing) {
            var t = Math.min(1, (now() - bar.completing.start) / COMPLETE_MS);
            var eased = 1 - Math.pow(1 - t, 3);
            bar.visual = bar.completing.from + (100 - bar.completing.from) * eased;
            setValue(bar, bar.visual);
            if (t >= 1) {
                bar.completing = null;
                return false;
            }
            return true;
        }
        var job = bar.job;
        if (!job || job.ended || bar.finish || job.phase !== "running") {
            bar.lastFrame = 0;
            return false;
        }
        if (!bar.lastFrame) bar.lastFrame = time;
        var dt = Math.min(time - bar.lastFrame, 100);
        bar.lastFrame = time;
        var at = now();
        var target = job.progress * 100;
        var remaining = remainingSeconds(job, at);
        var mode = config.smoothness;
        var textPct = null;

        if (mode === "smooth_gt_acc" && remaining !== null) {
            // 남은 거리를 남은 시간에 맞춰 일정한 속도로 간다.
            var targetSpeed = Math.max(0, 100 - bar.visual) / (Math.max(0.1, remaining) * 1000);
            bar.speed = bar.speed === 0 ? targetSpeed : bar.speed + (targetSpeed - bar.speed) * 0.05;
            bar.visual = Math.min(bar.visual + bar.speed * dt, CAP_PCT);
        } else if (mode === "smooth_lt_acc") {
            var actual = target;
            var predicted = actual;
            if (job.steps > 0 && job.passEta !== null && job.passEta > 0) {
                var stepMs = (job.passEta * 1000) / Math.max(1, job.steps - job.step);
                var perStep = 100 / (Math.max(1, job.jobCount) * job.steps);
                var inStep = Math.min(1, (at - job.updatedAt) / Math.max(100, stepMs));
                predicted = actual + perStep * inStep * 0.8;
            }
            var desired = Math.min(Math.max(actual, predicted), CAP_PCT);
            // 작업 안에서는 뒤로 가지 않는다(상류 lastDesiredPctM3).
            if (bar.lastDesired !== null && desired < bar.lastDesired) desired = bar.lastDesired;
            bar.lastDesired = desired;
            var lag = desired - bar.visual;
            var factor = lag > 15 ? 0.25 : (lag < 0 ? 0.03 : 0.08);
            bar.visual += lag * (1 - Math.exp(-factor * (dt / 16.66)));
            bar.visual = clamp(bar.visual, 0, CAP_PCT);
            // 상류: 글자는 막대와 실제 값 중 큰 쪽. 끝 신호 전에는 막대처럼 99.2% 를 넘지 않는다.
            textPct = Math.min(Math.max(bar.visual, actual), CAP_PCT);
        } else {
            // Smooth ~ Accurate — 그리고 ETA 를 아직 모를 때의 Smooth > Accurate(상류는 여기서 99% 로 내달렸다).
            var ratio = dt / 16.666;
            var distance = target - bar.visual;
            if (distance > 0) bar.visual += distance * (1 - Math.pow(1 - 0.035, ratio));
            else bar.visual += 0.008 * ratio;
            bar.visual = Math.min(bar.visual, CAP_PCT);
        }

        setValue(bar, bar.visual);
        render(bar, remaining === null ? null : Math.ceil(remaining), textPct === null ? bar.visual : textPct);
        return true;
    }

    // ---- 설치 / 물러나기 ------------------------------------------------------------

    function upstreamPresent() {
        return !!document.getElementById(UPSTREAM_STYLE_ID);
    }

    function wrap(fn) {
        var wrapped = function (id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout) {
            var args = Array.prototype.slice.call(arguments);
            var job = null;
            try {
                job = beginJob(id_task, progressbarContainer, inactivityTimeout);
            } catch (error) {
                console.error("[SAM Extra] progress bar", error);
            }
            if (job) {
                args[3] = function () {
                    try {
                        nativeEnd(job);
                    } catch (error) {
                        console.error("[SAM Extra] progress bar", error);
                    }
                    if (typeof atEnd === "function") return atEnd.apply(this, arguments);
                };
                args[4] = function (res) {
                    try {
                        nativeProgress(job, res);
                    } catch (error) {
                        console.error("[SAM Extra] progress bar", error);
                    }
                    if (typeof onProgress === "function") return onProgress.apply(this, arguments);
                };
            }
            return fn.apply(this, args);
        };
        wrapped.__sam3ProgressOriginal = fn;
        return wrapped;
    }

    function onClick(event) {
        if (!config || !config.enabled || steppedAside) return;
        var path = typeof event.composedPath === "function" ? event.composedPath() : [event.target];
        for (var i = 0; i < path.length; i += 1) {
            var node = path[i];
            if (!node || node.nodeType !== 1) continue;
            var match = /^(txt2img|img2img)_interrupt(?:ing)?$/.exec(node.id || "");
            if (!match) continue;
            var bar = bars[match[1]];
            if (bar && bar.job && !bar.job.ended) noteInterrupt(bar.job);
            return;
        }
    }

    function onVisibility() {
        if (document.hidden) return;
        TABS.forEach(function (tab) {
            var bar = bars[tab];
            if (!bar) return;
            if (bar.completing) ensureFrames();   // 숨어 있는 동안 멈춘 100% 채우기를 마저
            var job = bar.job;
            if (!job || job.ended) return;
            job.resync = true;
            if (job.pausedHidden) {
                job.pausedHidden = false;
                schedulePoll(job, 0);
            }
            if (job.phase === "running") ensureFrames();
        });
    }

    // 숨어 있던 동안 애니메이션 프레임이 멈춰 막대도 멈춰 있었다. Smooth > Accurate 는 남은 거리를 남은 시간으로 나눠
    // 가므로 그대로 두면 끝날 때까지 실제보다 한참 뒤처진다(상류도 같았다) — 다시 보인 뒤 첫 응답에서 서버의 진행률로
    // 바로 옮긴다. 앞으로만 간다. 다른 두 방식은 원래 서버 값을 쫓아가므로 그대로 둔다.
    function resync(job) {
        job.resync = false;
        var bar = job.bar;
        if (!shown(job) || bar.finish || config.smoothness !== "smooth_gt_acc") return;
        var target = Math.min(job.progress * 100, CAP_PCT);
        if (target > bar.visual) {
            bar.visual = target;
            setValue(bar, target);
        }
        bar.speed = 0;
    }

    function onMotionChange() {
        reduced = !!(motionQuery && motionQuery.matches);
        if (reduced && frame) {
            window.cancelAnimationFrame(frame);
            frame = 0;
        }
        TABS.forEach(function (tab) {
            var bar = bars[tab];
            if (!bar) return;
            if (reduced && bar.completing) {
                bar.completing = null;
                bar.visual = 100;
                setValue(bar, 100);
            }
            if (bar.job && !bar.job.ended && bar.job.phase === "running") onJobData(bar.job);
        });
    }

    function installHooks() {
        if (!wrapper && typeof window.requestProgress === "function") {
            original = window.requestProgress;
            wrapper = wrap(original);
            window.requestProgress = wrapper;
        }
        if (listening) return;
        listening = true;
        document.addEventListener("click", onClick, true);
        document.addEventListener("visibilitychange", onVisibility);
        if (typeof window.matchMedia === "function") {
            motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
            if (typeof motionQuery.addEventListener === "function") {
                motionQuery.addEventListener("change", onMotionChange);
            } else if (typeof motionQuery.addListener === "function") {
                motionQuery.addListener(onMotionChange);
            }
            reduced = !!motionQuery.matches;
        }
    }

    // 막대와 진행 중 작업을 모두 거둔다. 감싼 requestProgress 는 남지만 그대로 넘기기만 한다.
    function teardown() {
        if (frame) {
            window.cancelAnimationFrame(frame);
            frame = 0;
        }
        TABS.forEach(function (tab) {
            var bar = bars[tab];
            if (!bar) return;
            clearTimers(bar);
            [bar.job].concat(bar.backlog).forEach(function (job) {
                if (!job) return;
                job.ended = true;
                cancelPoll(job);
            });
            if (bar.root.parentNode) bar.root.parentNode.removeChild(bar.root);
            delete bars[tab];
        });
        document.documentElement.removeAttribute(ROOT_ATTR);
    }

    function stepAside() {
        if (steppedAside) return;
        steppedAside = true;
        teardown();
        console.info("[SAM Extra] sd-webui-smooth-progress 가 설치돼 있어 내장 진행 막대는 켜지 않습니다.");
    }

    function applyFinishMode(bar) {
        if (bar.job && !bar.job.ended) return;
        if (!bar.finish) {
            showIdle(bar);
            return;
        }
        clearTimers(bar);
        if (bar.completing) {
            bar.completing = null;
            bar.visual = 100;
            setValue(bar, 100);
        }
        afterFinish(bar);
    }

    function applyOptions() {
        var next = readConfig();
        var key = JSON.stringify(next);
        var finishChanged = !config || config.afterFinish !== next.afterFinish;
        config = next;
        if (!next.enabled || steppedAside) {
            teardown();
            configKey = key;
            return;
        }
        if (upstreamPresent()) {
            stepAside();
            return;
        }
        installHooks();
        if (document.documentElement.getAttribute(ROOT_ATTR) !== "on") {
            document.documentElement.setAttribute(ROOT_ATTR, "on");
        }
        if (uiReady) TABS.forEach(ensureBar);
        if (key === configKey) return;
        configKey = key;
        TABS.forEach(function (tab) {
            var bar = bars[tab];
            if (!bar) return;
            styleBar(bar);
            if (finishChanged) applyFinishMode(bar);
            render(bar);
        });
    }

    function boot() {
        // Forge 의 onUiLoaded 콜백들이 다 돈 뒤에 붙는다 — 그래야 sd-webui-smooth-progress 의 #spb-dynamic-css 가 이미 있다.
        window.setTimeout(function () {
            uiReady = true;
            applyOptions();
        }, 0);
    }

    if (typeof onOptionsAvailable === "function") onOptionsAvailable(applyOptions);
    if (typeof onOptionsChanged === "function") onOptionsChanged(applyOptions);
    if (typeof onUiLoaded === "function") onUiLoaded(boot);

    // 테스트 전용 입구. 실제 페이지는 이 객체를 만들지 않는다.
    if (window.__sam3ProgressTestHooks && typeof window.__sam3ProgressTestHooks === "object") {
        var hooks = window.__sam3ProgressTestHooks;
        hooks.boot = function () {
            uiReady = true;
            applyOptions();
        };
        hooks.applyOptions = applyOptions;
        hooks.formatText = formatText;
        hooks.formatTime = formatTime;
        hooks.readConfig = readConfig;
        hooks.isLight = isLight;
        hooks.bars = function () { return bars; };
        hooks.config = function () { return config; };
        hooks.steppedAside = function () { return steppedAside; };
        hooks.original = function () { return original; };
        hooks.defaults = DEFAULTS;
        hooks.choices = CHOICES;
        hooks.constants = {
            API_PATH: API_PATH, POLL_MS: POLL_MS, CAP_PCT: CAP_PCT, COMPLETE_MS: COMPLETE_MS,
            DONE_HOLD_MS: DONE_HOLD_MS, INTERRUPT_HOLD_MS: INTERRUPT_HOLD_MS, STOPPED_TEXT: STOPPED_TEXT,
        };
    }
})();
