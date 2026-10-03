// txt2img 의 always-on 아코디언을 묶음별 섹션으로 정리한다.
//
// 항목은 Python(sam3ext/layout_lanes.py)이 붙인 sam3-slot--<키> 클래스로만 알아본다. 라벨 글자는 ko_KR 이 바꾸고
// component-N id 는 실행마다 달라진다. 상태는 data-* 속성과 인라인 order 로만 쓴다 — Gradio 노드에 class 를
// 추가하면 Svelte 가 통째로 덮어쓰고, 자식을 넣고 빼면 Forge 의 onUiUpdate 가 매번 깨어난다(ko_KR 루프의 원인).
(function () {
    var SECTION_TITLES = {
        pinned: "고정",
        anima: "ANIMA 튜닝",
        det: "디테일러",
        script: "스크립트",
        lora: "로라·제어",
        etc: "도구·실험"
    };
    // sam-extra 항목의 짧은 이름. 나머지는 헤더 글자에서 가져온다(보여 주기 전용).
    var SHORT = {
        "anima-3-8b": "Anima 3.8B",
        "anima-detail-daemon": "Detail Daemon",
        "anima-skimmed-cfg": "Skimmed CFG",
        "anima-safe-pag": "Guidance",
        "anima-cfg-optimal-scale": "Optimal Scale",
        "colorcraft": "Colorcraft",
        "anima-vae-2x": "VAE 2x",
        "anima-speed": "SPEED",
        "sam3": "SAM3 Mask",
        "anima-vae-degrid": "VAE DeGrid",
        "anima-ref-poc": "Reference PoC"
    };
    // CSS order 값. 열 본문은 flex column 이라 order 만으로 자리가 정해진다.
    var ORDER = {
        pinnedHead: 100, pinnedHint: 101, pinnedFirst: 110, animaHead: 200, anima: 210,
        detHead: 100, det: 110, scriptHead: 200, more: 300, loraHead: 305, lora: 310,
        etcHead: 345, etc: 350
    };

    var state = { slots: [], mounted: false, timer: null, frame: null };

    function app() { return (window.gradioApp && window.gradioApp()) || document; }

    function layoutRoot() { return app().querySelector("#sam3_notebook_layout"); }

    function keyOf(el) {
        var match = /(?:^|\s)sam3-slot--([\w-]+)/.exec(el.className || "");
        return match ? match[1] : null;
    }

    function laneOf(el) {
        var match = /(?:^|\s)sam3-lane--([\w-]+)/.exec(el.className || "");
        return match ? match[1] : "etc";
    }

    function headLabel(labelWrap) {
        if (!labelWrap) return "";
        var span = labelWrap.querySelector("span");
        var text = (span && span.textContent) || "";
        return text.replace(/\s*Integrated\s*$/, "").replace(/\s*\([^)]*\)\s*$/, "").trim();
    }

    function collectSlots(layout) {
        var slots = [];
        Array.prototype.forEach.call(layout.querySelectorAll(".sam3-slot"), function (el) {
            var key = keyOf(el);
            var lane = laneOf(el);
            if (!key || lane === "hidden") return;
            var head = el.querySelector(".sam3-head");
            var labelWrap = head && head.querySelector(":scope > .label-wrap");
            slots.push({
                key: key,
                lane: lane,
                el: el,
                head: head,
                labelWrap: labelWrap,
                exp: /(?:^|\s)sam3-exp(?:\s|$)/.test(el.className || ""),
                guess: /(?:^|\s)sam3-on-guess(?:\s|$)/.test(el.className || ""),
                moved: false,
                pinned: false,
                label: SHORT[key] || headLabel(labelWrap) || key,
                home: el.parentElement,
                homeNext: el.nextElementSibling
            });
        });
        return slots;
    }

    function sectionHead(lane, order) {
        var head = document.createElement("h3");
        head.className = "sam3-lane-head";
        head.dataset.lane = lane;
        head.textContent = SECTION_TITLES[lane];
        head.style.order = order;
        return head;
    }

    function writeAttr(node, name, value) {
        if (!node) return;
        var current = node.getAttribute(name);
        if (value) {
            if (current !== value) node.setAttribute(name, value);
        } else if (current !== null) {
            node.removeAttribute(name);
        }
    }

    // ── 켜짐 판정 ────────────────────────────────────────────────────────────────────────────
    // 레거시 CFG 기반 모드 라디오의 값 → 실제로 켜지는 기능(anima_safe_pag.py 의 적용 규칙과 같다).
    var RADIO_FEATURES = { "APG": ["apg"], "CWM": ["cwm"], "SMC": ["smc"], "SMC + CWM": ["smc", "cwm"] };

    function featuresOf(input, index) {
        var owner = input.closest(".sam3-on") || input;
        var names = (owner.className.match(/sam3-on--([\w-]+)/g) || []).map(function (token) {
            return token.replace("sam3-on--", "");
        });
        return names.length ? names : ["_" + index];   // 이름이 없으면 컨트롤 하나가 기능 하나
    }

    function ownParts(slot, selector) {
        // 이 칸의 것만 센다 — 어떤 이유로든 다른 칸이 안쪽에 들어와 있어도 그 켜짐이 번지지 않게.
        return Array.prototype.filter.call(slot.el.querySelectorAll(selector), function (input) {
            return input.closest(".sam3-slot") === slot.el;
        });
    }

    function readState(slot) {
        var boxes = ownParts(slot, ".sam3-on input[type=checkbox]");
        var radios = ownParts(slot, ".sam3-on-radio input[type=radio]");
        var all = {};
        var on = {};
        Array.prototype.forEach.call(boxes, function (input, index) {
            featuresOf(input, index).forEach(function (name) {
                all[name] = true;
                if (input.checked) on[name] = true;
            });
        });
        Array.prototype.forEach.call(radios, function (input) {
            (RADIO_FEATURES[input.value] || []).forEach(function (name) {
                all[name] = true;
                if (input.checked) on[name] = true;
            });
        });
        var total = Object.keys(all).length;
        var live = Object.keys(on).length;
        return {
            total: total,
            on: live,
            pill: live === 0 ? "" : (total <= 1 ? "켜짐" : live + "/" + total + " 켜짐")
        };
    }

    // ── 켜진 기능 칩 줄 ──────────────────────────────────────────────────────────────────────
    var CORE_CHIPS = [
        { key: "core:hires", label: "고해상도 보정", box: "#txt2img_hr-checkbox input", head: "#txt2img_hr" },
        { key: "core:refiner", label: "리파이너", box: "#txt2img_enable-checkbox input", head: "#txt2img_enable" }
    ];

    function chipElement(key, label, onClick) {
        var chip = document.createElement("button");
        chip.type = "button";
        chip.className = "sam3-chip";
        chip.dataset.key = key;
        chip.hidden = true;
        var dot = document.createElement("span");
        dot.className = "sam3-chip-dot";
        chip.appendChild(dot);
        chip.appendChild(document.createTextNode(label));
        chip.addEventListener("click", onClick);
        return chip;
    }

    function buildChips(layout) {
        var bar = layout.querySelector("#sam3_onbar");
        if (!bar) return;
        bar.className = "sam3-onbar";
        bar.setAttribute("role", "group");
        bar.setAttribute("aria-label", "켜진 기능");
        state.title = document.createElement("span");
        state.title.className = "sam3-onbar-title";
        state.title.textContent = "켜진 기능";
        bar.appendChild(state.title);
        state.core = [];
        CORE_CHIPS.forEach(function (entry) {            // Forge 자체 기능 — 핀 없음, 맨 앞
            var input = app().querySelector(entry.box);
            if (!input) return;                          // 리파이너는 설정에 따라 없을 수 있다
            var chip = chipElement(entry.key, entry.label, function () {
                var head = app().querySelector(entry.head);
                if (head && typeof head.scrollIntoView === "function") head.scrollIntoView({ block: "nearest" });
            });
            bar.appendChild(chip);
            state.core.push({ chip: chip, input: input });
        });
        state.slots.forEach(function (slot) {
            if (readState(slot).total === 0) return;     // 켜짐 개념이 없는 칸은 칩을 만들지 않는다
            slot.chip = chipElement(slot.key, slot.label, function () { reveal(slot); });
            bar.appendChild(slot.chip);
        });
        state.empty = document.createElement("span");
        state.empty.className = "sam3-onbar-empty";
        state.empty.textContent = "없음 — 기본 설정으로 생성";
        state.empty.hidden = true;
        bar.appendChild(state.empty);
    }

    // ── 저장(이 브라우저에만) ────────────────────────────────────────────────────────────────
    var STORE_KEY = "sam-extra.layout.v1";
    var memory = { v: 1, pinned: [], moreOpen: false };

    function loadStore() {
        try {
            var raw = window.localStorage.getItem(STORE_KEY);
            if (raw) {
                var parsed = JSON.parse(raw);
                if (parsed && parsed.v === 1) {
                    memory.pinned = (parsed.pinned || []).slice(0, 64);
                    memory.moreOpen = parsed.moreOpen === true;
                }
            }
        } catch (error) { /* 시크릿 창 등 — 메모리에만 둔다 */ }
        return memory;
    }

    function saveStore() {
        try {
            window.localStorage.setItem(STORE_KEY, JSON.stringify(memory));
        } catch (error) { /* 저장하지 못해도 이번 화면에서는 그대로 동작한다 */ }
    }

    // ── 핀 ──────────────────────────────────────────────────────────────────────────────────
    var PIN_SVG = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor"'
        + ' stroke-width="1.5" stroke-linejoin="round" aria-hidden="true">'
        + '<path d="M9.8 1.8 14.2 6.2 11.6 7.4 8.9 10.1 9.3 13 3 6.7 5.9 7.1 8.6 4.4Z"></path>'
        + '<path d="M5.6 10.4 2 14"></path></svg>';

    function pinButton(slot) {
        var button = document.createElement("button");
        button.type = "button";
        button.className = "sam3-pin";
        button.innerHTML = PIN_SVG;
        button.addEventListener("click", function (event) {
            event.preventDefault();
            event.stopPropagation();      // InputAccordion 의 "열면 켜짐" 연결을 건드리지 않는다
            togglePin(slot);
        });
        return button;
    }

    function attachPins() {
        state.slots.forEach(function (slot) {
            if (!slot.head || !slot.labelWrap) return;
            slot.pin = pinButton(slot);
            // label-wrap(버튼)의 형제로 넣는다 — 안에 넣으면 헤더 클릭과 섞인다.
            slot.head.insertBefore(slot.pin, slot.labelWrap.nextSibling);
        });
    }

    function applyPinState(slot) {
        var pinned = memory.pinned.indexOf(slot.key) >= 0;
        slot.pinned = pinned;
        writeAttr(slot.el, "data-sam3-pinned", pinned ? "1" : "");
        if (slot.pin) {
            slot.pin.setAttribute("aria-pressed", pinned ? "true" : "false");
            slot.pin.setAttribute("aria-label", (pinned ? "고정 해제: " : "위에 고정: ") + slot.label);
        }
        writeAttr(slot.labelWrap, "data-sam3-tag",
            [pinned ? SECTION_TITLES[slot.lane] : "", slot.exp ? "실험" : ""].filter(Boolean).join(" · "));
        var column1 = layoutRoot().querySelector('[data-column="parameters"] > div');
        if (pinned) {
            // 1열 안에 이미 있으면(=ANIMA 칸) 옮기지 않는다. 부모가 #txt2img_settings 일 수 있어 contains 로 본다.
            if (column1 && !column1.contains(slot.el)) {
                column1.appendChild(slot.el);
                slot.moved = true;
            }
            slot.el.style.order = ORDER.pinnedFirst + memory.pinned.indexOf(slot.key);
        } else {
            if (slot.moved && slot.home) {
                var anchor = slot.homeNext && slot.homeNext.parentElement === slot.home ? slot.homeNext : null;
                slot.home.insertBefore(slot.el, anchor);
                slot.moved = false;
            }
            slot.el.style.order = ORDER[slot.lane];
        }
    }

    function togglePin(slot) {
        var index = memory.pinned.indexOf(slot.key);
        if (index >= 0) memory.pinned.splice(index, 1);
        else memory.pinned.push(slot.key);
        saveStore();
        state.slots.forEach(applyPinState);
        updateMore();
        flush();
    }

    // ── 더 보기 ─────────────────────────────────────────────────────────────────────────────
    function insideMore(slot) {
        return !slot.pinned && (slot.lane === "lora" || slot.lane === "etc");
    }

    function buildMore(column2) {
        var button = document.createElement("button");
        button.type = "button";
        button.className = "sam3-more";
        button.setAttribute("aria-expanded", "false");
        button.style.order = ORDER.more;
        var title = document.createElement("span");
        title.className = "sam3-more-title";
        title.textContent = "로라·제어, 도구·실험";
        button.appendChild(title);
        state.moreCount = document.createElement("span");
        state.moreCount.className = "sam3-more-count";
        button.appendChild(state.moreCount);
        state.moreOn = document.createElement("span");
        state.moreOn.className = "sam3-more-on";
        button.appendChild(state.moreOn);
        state.moreNames = state.slots
            .filter(function (slot) { return slot.lane === "lora" || slot.lane === "etc"; })
            .map(function (slot) {
                var span = document.createElement("span");
                span.className = "sam3-more-name";
                span.dataset.key = slot.key;
                span.textContent = slot.label;
                span.hidden = true;
                button.appendChild(span);
                return span;
            });
        button.addEventListener("click", function () {
            setMore(layoutRoot().dataset.sam3More === "closed");
        });
        column2.appendChild(button);
        state.more = button;
    }

    function setMore(open) {
        memory.moreOpen = open;
        saveStore();
        layoutRoot().dataset.sam3More = open ? "open" : "closed";
        if (state.more) state.more.setAttribute("aria-expanded", open ? "true" : "false");
        flush();
    }

    function updateMore() {
        if (!state.more) return;
        var inside = state.slots.filter(insideMore);
        writeAttr(state.moreCount, "data-count", String(inside.length));
        var on = inside.filter(function (slot) { return slot.el.dataset.sam3On === "on"; });
        writeAttr(state.moreOn, "data-count", String(on.length));
        state.moreNames.forEach(function (span) {
            var visible = on.some(function (slot) { return slot.key === span.dataset.key; });
            if (span.hidden === visible) span.hidden = !visible;
        });
    }

    function reveal(slot) {
        if (insideMore(slot) && layoutRoot().dataset.sam3More === "closed") setMore(true);
        if (slot.el && typeof slot.el.scrollIntoView === "function") {
            slot.el.scrollIntoView({ block: "nearest" });
        }
        if (slot.labelWrap && !slot.labelWrap.classList.contains("open")) slot.labelWrap.click();
        if (slot.labelWrap) slot.labelWrap.focus({ preventScroll: true });
    }

    // ── 검사(값이 바뀐 것만 쓴다) ────────────────────────────────────────────────────────────
    function scan() {
        if (!state.mounted) return;
        var layout = layoutRoot();
        var count = 0;
        state.slots.forEach(function (slot) {
            var status = readState(slot);
            writeAttr(slot.el, "data-sam3-on", status.on ? "on" : "");
            writeAttr(slot.labelWrap, "data-sam3-pill", status.pill);
            if (!slot.chip) return;
            var visible = status.on > 0;
            if (slot.chip.hidden === visible) slot.chip.hidden = !visible;
            writeAttr(slot.chip, "data-in-more",
                visible && insideMore(slot) && layout && layout.dataset.sam3More === "closed" ? "1" : "");
            if (visible) count += 1;
        });
        (state.core || []).forEach(function (entry) {
            var on = !!entry.input.checked;
            if (entry.chip.hidden === on) entry.chip.hidden = !on;
            if (on) count += 1;
        });
        if (state.empty && state.empty.hidden === (count === 0)) state.empty.hidden = count !== 0;
        writeAttr(state.title, "data-count", String(count));
        if (layout) writeAttr(layout, "data-sam3-count", String(count));
        updateMore();
    }

    function schedule() {
        if (state.frame !== null) return;
        var request = window.requestAnimationFrame || function (fn) { return window.setTimeout(fn, 16); };
        state.frame = request(function () {
            state.frame = null;
            scan();
        });
    }

    function flush() {                  // 예약된 검사를 기다리지 않고 바로 한 번(테스트·즉시 반영용)
        if (state.frame !== null) {
            (window.cancelAnimationFrame || window.clearTimeout)(state.frame);
            state.frame = null;
        }
        scan();
    }

    function poll() {                   // 1초 타이머가 부르는 것과 같은 함수
        if (document.visibilityState && document.visibilityState !== "visible") return;
        var tab = app().querySelector("#tab_txt2img");
        if (tab && tab.style.display === "none") return;
        scan();
    }

    function mount() {
        var layout = layoutRoot();
        if (!layout || state.mounted) return false;
        if (/[?&]sam3_lanes=off\b/.test(window.location.search)) return false;
        if (!layout.querySelector(".sam3-slot")) return false;
        var column1 = layout.querySelector('[data-column="parameters"] > div');
        var column2 = layout.querySelector('[data-column="scripts"] > div');
        if (!column1 || !column2) return false;

        state.slots = collectSlots(layout);
        column1.appendChild(sectionHead("pinned", ORDER.pinnedHead));
        var hint = document.createElement("p");
        hint.className = "sam3-pin-hint";
        hint.textContent = "헤더의 핀을 누르면 여기 고정됩니다";
        hint.style.order = ORDER.pinnedHint;
        column1.appendChild(hint);
        column1.appendChild(sectionHead("anima", ORDER.animaHead));
        ["det", "script", "lora", "etc"].forEach(function (lane) {
            column2.appendChild(sectionHead(lane, ORDER[lane + "Head"]));
        });
        state.slots.forEach(function (slot) {
            slot.el.style.order = ORDER[slot.lane];
            if (slot.guess) writeAttr(slot.el, "data-sam3-guess", "1");   // 추측으로 찾은 칸은 점선 알약
        });
        layout.dataset.sam3Lanes = "on";
        loadStore();
        layout.dataset.sam3More = memory.moreOpen ? "open" : "closed";    // 저장된 선택을 되살린다
        state.mounted = true;
        attachPins();
        buildMore(column2);
        if (state.more) state.more.setAttribute("aria-expanded", memory.moreOpen ? "true" : "false");
        state.slots.forEach(applyPinState);
        buildChips(layout);
        // 사용자 클릭, InputAccordion 의 프로그램 갱신, dynthres 의 버블 안 되는 change 까지 capture 로 받는다.
        layout.addEventListener("input", schedule, true);
        layout.addEventListener("change", schedule, true);
        if (typeof onAfterUiUpdate === "function") onAfterUiUpdate(schedule);
        // 인포텍스트 붙여넣기·프리셋 적용처럼 이벤트 없이 .checked 가 바뀌는 경우를 위한 느린 확인.
        state.timer = window.setInterval(poll, 1000);
        scan();
        return true;
    }

    function restore() {
        var layout = layoutRoot();
        if (!layout) return;
        Array.prototype.forEach.call(
            layout.querySelectorAll(".sam3-lane-head, .sam3-pin-hint, .sam3-pin, .sam3-more, .sam3-chip,"
                + " .sam3-onbar-title, .sam3-onbar-empty"),
            function (node) { node.remove(); }
        );
        state.slots.forEach(function (slot) {
            if (slot.moved && slot.home) {                     // 고정하며 옮긴 칸을 원래 자리로
                var anchor = slot.homeNext && slot.homeNext.parentElement === slot.home ? slot.homeNext : null;
                slot.home.insertBefore(slot.el, anchor);
                slot.moved = false;
            }
            slot.el.style.order = "";
            ["sam3On", "sam3Pinned", "sam3Guess"].forEach(function (name) { delete slot.el.dataset[name]; });
            if (slot.labelWrap) {
                ["sam3Pill", "sam3Tag"].forEach(function (name) { delete slot.labelWrap.dataset[name]; });
            }
        });
        if (state.timer) { window.clearInterval(state.timer); state.timer = null; }
        delete layout.dataset.sam3Lanes;
        delete layout.dataset.sam3More;
        state.mounted = false;
    }

    function boot() {
        try {
            mount();
        } catch (error) {
            console.error("[SAM3] lanes mount failed", error);
            try { restore(); } catch (ignored) { /* 원상복구도 실패하면 그냥 둔다 */ }
        }
    }

    window.addEventListener("sam3:notebook-mounted", boot);
    if (typeof onUiLoaded === "function") onUiLoaded(boot);

    // 테스트 훅. 실제 화면에서는 위의 두 경로로만 들어온다.
    window.__sam3LanesTestHooks = {
        mount: boot, scan: scan, flush: flush, poll: poll, restore: restore, state: state
    };
})();
