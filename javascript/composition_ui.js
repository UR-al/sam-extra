/* 구도 · 카메라 — txt2img·img2img 스타일 줄 아래(Generate 옆 열)의 접힌 칸 "구도 · 카메라 (프롬프트로 시점 잡기)".
 *
 * 사용자 앱 UR_IV 의 frontend/src/components/CompositionControl.vue(앱 커밋 d2fcc70798e61da137e87a8ff600d6fa73250fe6)를
 * Vue 없이 옮긴 것이다. 계산은 모두 javascript/composition_prompt.js(앱 compositionPrompt.ts 의 이식, 전역
 * sam3CompositionPrompt)가 한다. 화면 글자와 태그는 앱과 같다.
 *
 * 자리: scripts/composition_camera.py 가 Forge 의 스타일 줄(<탭>_styles_row) 바로 아래 — txt2img 는 TIPO 칸 다음 — 에 둔
 * 빈 gr.HTML(#sam3_composition_<탭> 안의 div.sam3-composition-mount)을 이 파일이 처음 한 번 채운다. 그 뒤로는 속성·aria·
 * 입력값·텍스트 노드 .data 만 바꾸고 자식을 넣고 빼지 않는다 — Forge 의 childList 관찰자(onUiUpdate)가 깨지 않게
 * (design.md "Runtime DOM rules"). 충돌 목록의 <li> 도 미리 만들어 두고 hidden 만 바꾼다.
 *
 * 앱과 같은 것: 프리셋 5개(aria-pressed), 궤도 그림 끌기(포인터 캡처, 취소되면 시작 상태로 되돌림), 방향키 5° · Shift 15° ·
 * Home 정면, 슬라이더 5개와 값, 태그 미리보기, 이미 있는 반대 태그 경고(CONTRAST_GROUPS), 추가 버튼 글자 세 가지
 * ("메인 태그에 추가" / "이미 포함된 구도" / "기존 구도 유지하고 추가"), 초기화, 알림 줄, 조작값 저장(프리셋·슬라이더 놓을 때·
 * 끌기 끝·키보드·초기화 때. 끌기 중간과 취소는 저장하지 않음).
 * Forge 에 맞춘 것:
 * - 대상은 그 탭의 메인 프롬프트 #<탭>_prompt textarea. 앱이 함께 보던 다른 칸(캐릭터·작가·접두·접미)은 Forge 에 없어
 *   otherSections 는 빈 문자열이다(부정 프롬프트는 넣지 않는다).
 * - 추가: 적어 둔 글자는 다시 쓰지 않고 붙일 부분만 끝에 넣는다. 누른 순간의 프롬프트로 다시 계산하고, 커서를 끝에 두고
 *   document.execCommand('insertText') 로 넣어 브라우저 Ctrl+Z 한 번에 되돌아간다. 브라우저가 내는 타이핑 모양의 input
 *   이벤트는 막고(태그 자동완성이 추천 목록을 띄우지 않게) Forge 의 updateInput 으로 Gradio 에 알린다 — insertAtEnd 참고.
 *   아무것도 안 바뀌면(false — 지원하지 않는 브라우저 등) 값을 쓰고 같은 updateInput 으로 알린다.
 *   넣은 뒤 포커스는 프롬프트 칸 끝에 남는다(바로 Ctrl+Z 할 수 있게).
 * - 미리보기·충돌은 프롬프트 입력(input 이벤트)마다 다시 계산한다. Gradio 가 값을 바꾸는 경우(↙️ 붙여넣기, 스타일 적용,
 *   TIPO 등)는 이벤트가 없으므로 칸을 열 때, 칸에 포인터·포커스가 들어올 때 다시 계산한다. 누를 때는 늘 새로 계산한다.
 * - 저장: 앱의 save_ui_prefs 대신 이 브라우저의 localStorage(탭마다 sam-extra.composition.<탭>.v1). 모든 접근을
 *   try/catch 로 감싸고(막힌 저장소에서도 패널은 돈다), 깨진 값은 앱과 같은 normalizeComposition 으로 고친다.
 * - 끄기: Settings → SAM Extra Appearance → sam3_composition_panel(Reload UI 또는 재시작 뒤). 끄면 자리가 없어 여기서 할 일도 없다.
 */
(function () {
    "use strict";

    if (window.__sam3CompositionUiLoaded) return;
    window.__sam3CompositionUiLoaded = true;

    var TABS = ["txt2img", "img2img"];
    // scripts/composition_camera.py 의 MOUNT_ELEM_ID · MOUNT_CLASS 와 같아야 한다(tests/test_composition_camera.py 가 대조).
    var HOST_ID_PREFIX = "sam3_composition_";
    var MOUNT_CLASS = "sam3-composition-mount";
    var STORAGE_PREFIX = "sam-extra.composition.";
    var STORAGE_VERSION = ".v1";
    var KEY_SHORTCUTS = "ArrowLeft ArrowRight ArrowUp ArrowDown Home";

    // 앱 CompositionControl.vue 의 글자 그대로.
    var TEXT = Object.freeze({
        title: "구도 · 카메라",
        subtitle: "프롬프트로 시점 잡기",
        intro: "태그와 구도 문구로 생성을 유도합니다. 실제 3D 카메라 제어가 아니며 모델에 따라 결과가 달라집니다. "
            + "조작만으로 프롬프트가 바뀌지 않습니다.",
        presets: "구도 프리셋",
        orbit: "카메라 구도 조작",
        front: "정면",
        back: "후면",
        orbitHelp: "드래그 또는 방향키로 시점 조작 · Shift + 방향키: 크게 이동 · Home: 정면 복원. "
            + "아래 슬라이더로도 모두 조절할 수 있습니다.",
        rangeHelp: "거리 0: 근접 / 100: 원경 · 화면 위치 −: 왼쪽 / +: 오른쪽. 작은 수치 차이는 같은 문구로 표현될 수 있습니다.",
        preview: "추가할 태그 · 구도 문구",
        conflict: "기존 구도와 충돌할 수 있습니다. 기존 태그는 삭제하지 않습니다.",
        append: "메인 태그에 추가",
        appendDone: "이미 포함된 구도",
        appendKeep: "기존 구도 유지하고 추가",
        reset: "초기화",
    });

    function readout(state) {
        return "방향 " + state.azimuth + "° · 높이 " + state.elevation + "°";
    }

    function appended(count) {
        return "메인 태그에 " + count + "개를 추가했습니다. 기존 프롬프트는 유지했습니다.";
    }

    var panels = Object.create(null);
    var watching = false;
    var finished = false;

    function logic() {
        var api = window.sam3CompositionPrompt;
        return api && typeof api.planCompositionAppend === "function" ? api : null;
    }

    function appRoot() {
        try {
            if (typeof gradioApp === "function") return gradioApp();
        } catch (_error) {
            // gradioApp() looks for <gradio-app>; the document is the same tree in Forge's light DOM.
        }
        return document;
    }

    function storageKey(tab) {
        return STORAGE_PREFIX + tab + STORAGE_VERSION;
    }

    function loadState(api, tab) {
        var raw = null;
        try {
            raw = window.localStorage.getItem(storageKey(tab));
        } catch (_error) {
            raw = null;   // storage blocked (privacy mode, policy): start from the defaults
        }
        var parsed = null;
        try {
            parsed = raw === null ? null : JSON.parse(raw);
        } catch (_error) {
            parsed = null;   // malformed old value: must not stop the panel
        }
        return api.normalizeComposition(parsed);
    }

    function saveState(tab, state) {
        try {
            window.localStorage.setItem(storageKey(tab), JSON.stringify(state));
        } catch (_error) {
            // storage blocked or full: the panel keeps working for this page
        }
    }

    function escapeHtml(value) {
        return String(value).replace(/[&<>"']/g, function (ch) {
            return { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[ch];
        });
    }

    function setText(node, value) {
        if (node.data !== value) node.data = value;
    }

    function setAttr(element, name, value) {
        if (value === null) {
            if (element.hasAttribute(name)) element.removeAttribute(name);
        } else if (element.getAttribute(name) !== value) {
            element.setAttribute(name, value);
        }
    }

    // 한 번만 만드는 텍스트 노드 — 이후에는 .data 만 바꾼다.
    function textSlot(element) {
        var node = element.ownerDocument.createTextNode("");
        element.appendChild(node);
        return node;
    }

    var DIAGRAM = [
        '<svg viewBox="0 0 300 170" aria-hidden="true" focusable="false">',
        '<ellipse class="sam3-composition__orbit-line" cx="150" cy="88" rx="90" ry="32"></ellipse>',
        '<path class="sam3-composition__orbit-line" d="M150 38V134 M50 88H250"></path>',
        '<text class="sam3-composition__diagram-label" x="150" y="158" text-anchor="middle">' + TEXT.front + "</text>",
        '<text class="sam3-composition__diagram-label" x="150" y="21" text-anchor="middle">' + TEXT.back + "</text>",
        '<line class="sam3-composition__camera-ray" data-ref="ray" x2="150" y2="80"></line>',
        '<g class="sam3-composition__subject">',
        '<circle cx="150" cy="64" r="9"></circle>',
        '<path d="M140 82Q150 75 160 82L163 103H137Z M143 105L140 124 M157 105L160 124"></path>',
        "</g>",
        '<g class="sam3-composition__camera" data-ref="camera">',
        '<rect x="-11" y="-7" width="18" height="14" rx="3"></rect>',
        '<path d="M7 -3L14 -6V6L7 3Z"></path>',
        "</g>",
        "</svg>",
    ].join("");

    function markup(api, tab, id) {
        var presets = api.COMPOSITION_PRESETS.map(function (preset, index) {
            return '<button type="button" class="sam3-composition__button" data-preset="' + index
                + '" aria-pressed="false">' + escapeHtml(preset.name) + "</button>";
        }).join("");
        var ranges = api.COMPOSITION_CONTROLS.map(function (control) {
            var inputId = id + "-" + control.key;
            return '<label class="sam3-composition__range" for="' + inputId + '">'
                + "<span>" + escapeHtml(control.label) + ' <output for="' + inputId + '"></output></span>'
                + '<input id="' + inputId + '" type="range" min="' + control.min + '" max="' + control.max
                + '" step="1" data-key="' + control.key + '">'
                + "</label>";
        }).join("");
        var conflictSlots = api.CONTRAST_GROUPS.map(function () { return "<li hidden></li>"; }).join("");
        return [
            '<details class="sam3-composition" data-tab="' + tab + '">',
            '<summary class="sam3-composition__summary">' + TEXT.title + " <span>" + TEXT.subtitle + "</span></summary>",
            '<div class="sam3-composition__body">',
            '<p class="sam3-composition__help">' + TEXT.intro + "</p>",
            '<div class="sam3-composition__presets" role="group" aria-label="' + TEXT.presets + '">' + presets + "</div>",
            '<div class="sam3-composition__orbit" tabindex="0" role="group" aria-label="' + TEXT.orbit
                + '" aria-describedby="' + id + '-help" aria-keyshortcuts="' + KEY_SHORTCUTS + '">',
            DIAGRAM,
            '<span class="sam3-composition__readout" data-ref="readout"></span>',
            "</div>",
            '<p id="' + id + '-help" class="sam3-composition__help">' + TEXT.orbitHelp + "</p>",
            '<div class="sam3-composition__ranges">' + ranges + "</div>",
            '<p class="sam3-composition__help">' + TEXT.rangeHelp + "</p>",
            '<div class="sam3-composition__preview" id="' + id + '-preview">',
            "<strong>" + TEXT.preview + '</strong><p data-ref="preview"></p>',
            "</div>",
            '<div class="sam3-composition__warning" role="status" hidden>' + TEXT.conflict
                + '<ul data-ref="conflicts">' + conflictSlots + "</ul></div>",
            '<div class="sam3-composition__actions">',
            '<button type="button" class="sam3-composition__button sam3-composition__append" data-ref="append"'
                + ' aria-describedby="' + id + '-preview"></button>',
            '<button type="button" class="sam3-composition__button" data-ref="reset">' + TEXT.reset + "</button>",
            "</div>",
            '<span class="sam3-composition__feedback" data-ref="feedback" role="status" aria-live="polite"></span>',
            "</div>",
            "</details>",
        ].join("");
    }

    // Gradio 에 값이 바뀌었다고 알린다 — Forge 의 updateInput(javascript/ui.js)과 같은 input 이벤트.
    function notifyGradio(field) {
        if (typeof window.updateInput === "function") {
            window.updateInput(field);
            return;
        }
        var view = field.ownerDocument.defaultView || window;
        field.dispatchEvent(new view.Event("input", { bubbles: true }));
    }

    // 붙일 부분(text)만 프롬프트 끝에 넣는다. 돌려주는 값: "insertText"(브라우저의 편집 — Ctrl+Z 한 번) · "value"(대체 경로).
    //
    // 브라우저가 insertText 로 내는 input 이벤트(inputType "insertText")는 타이핑과 똑같아 보여서, 태그 자동완성
    // (sd-webui-tagcomplete — inputType 이 없는 updateInput 이벤트는 일부러 무시한다)이 추천 목록을 띄운다. 그 목록이 떠 있을 때
    // Enter/Tab 을 누르면 방금 붙인 마지막 단어가 추천어로 바뀐다(실제 Forge 에서 확인). 그래서 그 이벤트는 창의 capture
    // 단계에서 막고, Forge 의 붙여넣기 버튼처럼 updateInput 으로 Gradio 에 알린다. 되돌리기 기록은 execCommand 가 이미
    // 남겼으므로 Ctrl+Z 한 번은 그대로다.
    function insertAtEnd(field, text) {
        var before = field.value;
        var doc = field.ownerDocument;
        var view = doc.defaultView || window;
        var swallow = function (event) {
            if (event.target === field) event.stopImmediatePropagation();
        };
        view.addEventListener("input", swallow, true);
        try {
            field.focus({ preventScroll: true });
            field.setSelectionRange(before.length, before.length);
            if (typeof doc.execCommand === "function") doc.execCommand("insertText", false, text);
        } catch (_error) {
            // execCommand may throw where editing commands are unsupported; fall through to the value path
        } finally {
            view.removeEventListener("input", swallow, true);
        }
        if (field.value !== before) {   // the browser inserted it (whatever the command reported)
            notifyGradio(field);
            return "insertText";
        }
        field.value = before + text;
        notifyGradio(field);
        return "value";
    }

    function createPanel(api, tab, mount, promptHost) {
        var id = "sam3-composition-" + tab;
        mount.innerHTML = markup(api, tab, id);
        var root = mount.querySelector(".sam3-composition");
        var ref = function (name) { return root.querySelector('[data-ref="' + name + '"]'); };
        var presetButtons = Array.prototype.slice.call(root.querySelectorAll("[data-preset]"));
        var orbit = root.querySelector(".sam3-composition__orbit");
        var ray = ref("ray");
        var camera = ref("camera");
        var readoutText = textSlot(ref("readout"));
        var ranges = api.COMPOSITION_CONTROLS.map(function (control) {
            var input = root.querySelector('input[data-key="' + control.key + '"]');
            return { control: control, input: input, output: textSlot(input.parentNode.querySelector("output")) };
        });
        var previewText = textSlot(ref("preview"));
        var warning = root.querySelector(".sam3-composition__warning");
        var conflictItems = Array.prototype.map.call(ref("conflicts").children, function (li) {
            return { li: li, text: textSlot(li) };
        });
        var appendButton = ref("append");
        var appendLabel = textSlot(appendButton);
        var resetButton = ref("reset");
        var feedbackText = textSlot(ref("feedback"));

        var state = loadState(api, tab);
        var feedback = "";
        var drag = null;

        function promptField() {
            return promptHost.querySelector("textarea");
        }

        function sameState(value) {
            return api.COMPOSITION_CONTROLS.every(function (control) {
                return state[control.key] === value[control.key];
            });
        }

        function render() {
            var tags = api.compositionTags(state);
            var field = promptField();
            var current = api.planCompositionAppend(field ? field.value : "", tags);
            var point = api.compositionCameraPoint(state);
            presetButtons.forEach(function (button, index) {
                setAttr(button, "aria-pressed", String(sameState(api.COMPOSITION_PRESETS[index].state)));
            });
            setAttr(ray, "x1", String(point.x));
            setAttr(ray, "y1", String(point.y));
            setAttr(ray, "stroke-dasharray", point.behind ? "4 4" : null);
            setAttr(camera, "transform", "translate(" + point.x + " " + point.y + ") rotate(" + state.roll + ")");
            setText(readoutText, readout(state));
            ranges.forEach(function (range) {
                var value = String(state[range.control.key]);
                if (range.input.value !== value) range.input.value = value;
                setText(range.output, value + range.control.unit);
            });
            setText(previewText, tags.join(", "));
            if (warning.hidden !== !current.conflicts.length) warning.hidden = !current.conflicts.length;
            conflictItems.forEach(function (item, index) {
                var conflict = current.conflicts[index];
                if (item.li.hidden !== (conflict === undefined)) item.li.hidden = conflict === undefined;
                setText(item.text, conflict === undefined ? "" : conflict);
            });
            if (appendButton.disabled !== !current.additions.length) appendButton.disabled = !current.additions.length;
            setText(appendLabel, !current.additions.length ? TEXT.appendDone
                : current.conflicts.length ? TEXT.appendKeep : TEXT.append);
            setText(feedbackText, feedback);
        }

        function update(value) {
            state = api.normalizeComposition(value);
            feedback = "";
            render();
        }

        function persist() {
            saveState(tab, state);
        }

        function applyPreset(value) {
            update({ ...value });
            persist();
        }

        function setControl(key, event) {
            update({ ...state, [key]: Number(event.target.value) });
        }

        function startDrag(event) {
            if (!event.isPrimary || event.button !== 0) return;
            var target = event.currentTarget;
            try {
                target.focus({ preventScroll: true });
            } catch (_error) {
                // focus is a convenience for the arrow keys
            }
            try {
                target.setPointerCapture(event.pointerId);
            } catch (_error) {
                // a pointer that is already gone: the drag still follows while it stays over the diagram
            }
            drag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, state: { ...state } };
        }

        function moveDrag(event) {
            if (!drag || drag.pointerId !== event.pointerId) return;
            update(api.dragComposition(drag.state, event.clientX - drag.x, event.clientY - drag.y));
        }

        function endDrag(event) {
            if (!drag || drag.pointerId !== event.pointerId) return;
            moveDrag(event);
            drag = null;
            var target = event.currentTarget;
            try {
                if (target.hasPointerCapture(event.pointerId)) target.releasePointerCapture(event.pointerId);
            } catch (_error) {
                // nothing to release
            }
            persist();
        }

        function cancelDrag(event) {
            if (!drag || drag.pointerId !== event.pointerId) return;
            // A canceled touch restores the starting control state, never a half gesture.
            state = drag.state;
            drag = null;
            render();
        }

        function onOrbitKey(event) {
            if (event.ctrlKey || event.altKey || event.metaKey) return;
            var step = event.shiftKey ? 15 : 5;
            var moves = {
                ArrowLeft: { azimuth: state.azimuth - step }, ArrowRight: { azimuth: state.azimuth + step },
                ArrowUp: { elevation: state.elevation + step }, ArrowDown: { elevation: state.elevation - step },
                Home: { azimuth: 0, elevation: 0 },
            };
            if (!Object.prototype.hasOwnProperty.call(moves, event.key)) return;
            event.preventDefault();
            update({ ...state, ...moves[event.key] });
            persist();
        }

        function appendToPrompt() {
            var field = promptField();
            if (!field) return;
            var before = field.value;
            var next = api.planCompositionAppend(before, api.compositionTags(state));
            if (!next.additions.length) {
                render();
                return;
            }
            // planCompositionAppend never rewrites the prompt: its text is the old prompt plus the appended part.
            insertAtEnd(field, next.text.slice(before.length));
            feedback = appended(next.additions.length);
            render();
        }

        presetButtons.forEach(function (button, index) {
            button.addEventListener("click", function () { applyPreset(api.COMPOSITION_PRESETS[index].state); });
        });
        orbit.addEventListener("pointerdown", startDrag);
        orbit.addEventListener("pointermove", moveDrag);
        orbit.addEventListener("pointerup", endDrag);
        orbit.addEventListener("pointercancel", cancelDrag);
        orbit.addEventListener("lostpointercapture", cancelDrag);
        orbit.addEventListener("keydown", onOrbitKey);
        ranges.forEach(function (range) {
            range.input.addEventListener("input", function (event) { setControl(range.control.key, event); });
            range.input.addEventListener("change", persist);
        });
        appendButton.addEventListener("click", appendToPrompt);
        resetButton.addEventListener("click", function () { applyPreset(api.DEFAULT_COMPOSITION); });
        // 프롬프트를 칠 때마다, 그리고 Gradio 가 이벤트 없이 바꾼 값을 따라잡게 칸을 열거나 다가올 때.
        promptHost.addEventListener("input", render);
        root.addEventListener("toggle", render);
        root.addEventListener("focusin", render);
        root.addEventListener("pointerenter", render);
        render();

        return {
            root: root,
            render: render,
            state: function () { return { ...state }; },
        };
    }

    function mountTab(api, tab) {
        if (panels[tab]) return true;
        var app = appRoot();
        var host = app.querySelector("#" + HOST_ID_PREFIX + tab);
        var mount = host && host.querySelector("." + MOUNT_CLASS);
        var promptHost = app.querySelector("#" + tab + "_prompt");
        if (!mount || !promptHost || !promptHost.querySelector("textarea")) return false;
        panels[tab] = createPanel(api, tab, mount, promptHost);
        return true;
    }

    function mountAll() {
        var api = logic();
        if (!api) return false;
        var done = true;
        TABS.forEach(function (tab) {
            if (!mountTab(api, tab)) done = false;
        });
        return done;
    }

    // Gradio 는 컴포넌트를 나눠 그린다. 처음에 못 찾은 탭은 화면이 바뀐 뒤 다시 찾는다(설정을 꺼서 자리가 없는 탭도
    // 같은 값싼 조회 두 번으로 끝난다). 둘 다 붙으면 더는 찾지 않는다.
    function start() {
        finished = mountAll();
        if (finished || watching || typeof onAfterUiUpdate !== "function") return;
        watching = true;
        onAfterUiUpdate(function () {
            if (!finished) finished = mountAll();
        });
    }

    if (typeof onUiLoaded === "function") onUiLoaded(start);

    if (window.__sam3CompositionTestHooks && typeof window.__sam3CompositionTestHooks === "object") {
        var hooks = window.__sam3CompositionTestHooks;
        hooks.TEXT = TEXT;
        hooks.panels = panels;
        hooks.mountAll = mountAll;
        hooks.insertAtEnd = insertAtEnd;
        hooks.storageKey = storageKey;
    }
})();
