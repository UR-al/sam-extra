// Colorcraft 공유 편집기(javascript/colorcraft_editor.js)의 접근성 — jsdom 에 Gradio 4.40 과 같은 모양의 패널(두 탭)을 두고
// Forge 처럼 onUiLoaded 뒤에 처리기를 부른다(Gradio 가 값을 적용한 뒤의 일은 0ms 타이머 → 한 프레임 뒤에 본다).
//   - 선택 줄(수정자 · 마스크 · 조합)과 편집기의 Pass: Radio 의 <fieldset> 에 제목 이름 · 설명, 라디오마다 위치/개수
//   - 편집기 칸: 지금 편집하는 항목("편집 중: 수정자 III (Chroma)" / "편집 중: 조합 C2")을 설명으로
//   - 알림(role="status"): 다른 항목으로 옮김 · Reset · 붙여넣기 뒤에 한 번. 자기 쓰기의 메아리와 Type 바꾸기는 말하지 않는다
//   - 마운트 뒤로는 노드를 넣고 빼지 않는다(design.md "Runtime DOM rules"), 리스너도 붙이지 않는다
//   - 표시 쪽(적기 · 예약)이 실패해도 처리기의 값은 그대로 나간다(콘솔에 "[Colorcraft] a11y:" 하나)
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { JSDOM } from "jsdom";
import { JS_DIR, loadEditor } from "./colorcraft_bridge.mjs";

const SCHEMA = loadEditor().schema;
const SCHEMA_JS = readFileSync(path.join(JS_DIR, "colorcraft_schema.js"), "utf8");
const EDITOR_JS = readFileSync(path.join(JS_DIR, "colorcraft_editor.js"), "utf8");
const NOTEBOOK_JS = readFileSync(path.join(JS_DIR, "notebook.js"), "utf8");
const MOD_MARKS = "● 켬 · ○ 값을 바꿨지만 꺼 둠";             // sam3ext/colorcraft/ui.py (tests/test_colorcraft_script.py 가 대조)
const MASK_MARKS = "● 켠 수정자가 씀 · ○ 값만 바꿈";
const prefix = (tab) => `script_${tab}_colorcraft_samextra_`;
const P = prefix("txt2img");
const N_MOD = SCHEMA.modifierFields.length;
const modIndex = (name) => SCHEMA.modifierFields.findIndex((f) => f.name === name);
const afterNudge = (window) => new Promise((r) => window.setTimeout(() => window.requestAnimationFrame(() => r()), 0));

function defaults(mod = "I") {
    const i = SCHEMA.modifierTags.indexOf(mod);
    return [...SCHEMA.modifierFields.map((f) => (f.name === "active" ? SCHEMA.activeDefaults[i] : f.default)),
        ...SCHEMA.leafFields.map((f) => f.default), ...SCHEMA.comboFields.map((f) => f.default)];
}

// ---------------------------------------------------------------- Gradio 4.40 모양의 패널

let radioName = 0;
function radios(id, title, info, choices, value) {
    // Radio: Block(<fieldset>) > StatusTracker · BlockTitle(<span data-testid="block-info">) · Info(<div>) · div.wrap > label > input
    return `<fieldset id="${id}" class="block padded hide-container"><div class="wrap default hide"></div>`
        + `<span data-testid="block-info"${info ? ' class="has-info"' : ""}>${title}</span>${info ? `<div>${info}</div>` : ""}`
        + `<div class="wrap">${choices.map((c) => `<label data-testid="${c}-radio-label">`
            + `<input type="radio" name="radio-${++radioName}" value="${c}" aria-checked="${c === value}"${c === value ? " checked" : ""}>`
            + `<span class="ml-2">${c}</span></label>`).join("")}</div></fieldset>`;
}

function control(id, f, extra = "") {
    if (f.name === "pass") return radios(id, "Pass", null, ["Base", "Hires", "Both"], f.default);
    if (typeof f.default === "boolean") {
        return `<div id="${id}" class="block gradio-checkbox"><label class="checkbox-container">`
            + `<input type="checkbox" name="test" data-testid="checkbox"><span class="ml-2">${f.name}</span></label></div>`;
    }
    if (typeof f.default === "number") {
        return `<div id="${id}" class="block gradio-slider"><div class="wrap"><div class="head"><label for="range_${id}">`
            + `<span data-testid="block-info">${f.name}</span></label><input aria-label="number input for ${f.name}" `
            + `data-testid="number-input" type="number" value="${f.default}"${extra}></div></div>`
            + `<input type="range" id="range_${id}" aria-label="range slider for ${f.name}" value="${f.default}"></div>`;
    }
    return `<div id="${id}" class="block gradio-dropdown"><div class="container"><span data-testid="block-info">${f.name}</span>`
        + `<input role="listbox" aria-controls="dropdown-options" aria-expanded="false" aria-label="${f.name}" `
        + `value="${f.default}"></div></div>`;
}

function panel(tab) {
    const pre = prefix(tab);
    const hint = `${pre}strength_hint`;
    return `<div id="${pre}accordion" class="block gradio-accordion"><button class="label-wrap open"><span>Colorcraft</span></button><div>`
        + `<div id="${pre}summary" class="block hide-container"><div class="wrap center hide"></div>`
        + '<div><div class="prose"><div class="samextra-cc-summary">꺼짐 — 켠 수정자: I Advanced</div></div></div></div>'
        + radios(`${pre}modifier_select`, "수정자", MOD_MARKS, SCHEMA.modifierTags, "I")
        + SCHEMA.modifierFields.map((f) => control(`${pre}mod_${f.name}`, f,
            f.name === "strength" ? ` aria-describedby="${hint}"` : "")).join("")
        + `<span id="${hint}" hidden>세기</span>`
        + radios(`${pre}mask_select`, "마스크 · 조합", MASK_MARKS, [...SCHEMA.maskTags, ...SCHEMA.comboTags], "M1")
        + SCHEMA.leafFields.map((f) => control(`${pre}leaf_${f.name}`, f)).join("")
        + SCHEMA.comboFields.map((f) => control(`${pre}combo_${f.name}`, f)).join("")
        + "</div></div>";
}

// 패널 두 탭 · 스키마 · 편집기를 올리고 Forge 처럼 onUiLoaded 를 부른다. 그 뒤 붙는 이벤트 리스너는 listeners 에 남는다.
function page({ notebook = false } = {}) {
    const dom = new JSDOM(`<!doctype html><html><body><div class="gradio-container">${panel("txt2img")}${panel("img2img")}`
        + "</div></body></html>", { url: "http://127.0.0.1:7860/", runScripts: "outside-only", pretendToBeVisual: true });
    const { window } = dom;
    window.gradioApp = () => window.document;
    if (notebook) {
        window.onUiLoaded = () => {};                                 // notebook.js 의 자동 설치는 돌리지 않는다
        window.opts = {};
        window.gradio_config = { components: [
            { id: 1, props: { elem_id: `${P}mod_kind`, choices: Object.keys(SCHEMA.kindIds).map((k) => [k, k]) } },
            { id: 2, props: { elem_id: `${P}combo_mask_a`, choices: SCHEMA.comboRefChoices[4].map((c) => [c, c]) } },
        ] };
        window.CSS = window.CSS || {};
        window.CSS.escape = window.CSS.escape || ((value) => String(value));
        window.Element.prototype.scrollIntoView = function () {};
        window.__sam3NotebookTestHooks = {};
        window.eval(NOTEBOOK_JS);
    }
    const loaded = [];
    window.onUiLoaded = (fn) => loaded.push(fn);
    window.document.querySelector("body");               // jsdom's selector engine adds its own mouse listeners on first use
    const listeners = [];
    const add = window.EventTarget.prototype.addEventListener;
    window.EventTarget.prototype.addEventListener = function (type, ...rest) {
        listeners.push(type);
        return add.call(this, type, ...rest);
    };
    window.eval(SCHEMA_JS);
    window.eval(EDITOR_JS);
    loaded.forEach((fn) => fn());
    const doc = window.document;
    const status = (tab = "txt2img") => doc.getElementById(`${prefix(tab)}a11y_status`);
    const describes = (tab, side) => doc.getElementById(`${prefix(tab)}a11y_${side}`).textContent;
    return { dom, window, doc, cc: window.samextraColorcraft, listeners, status, describes,
        restore: () => { window.EventTarget.prototype.addEventListener = add; } };
}

// 그 속성이 가리키는 노드들의 글자(accname 의 aria-labelledby/aria-describedby 처럼)
const refText = (doc, el, attr) => (el.getAttribute(attr) || "").split(/\s+/).filter(Boolean)
    .map((id) => doc.getElementById(id).textContent).join(" ");

// ---------------------------------------------------------------- 테스트

test("mount: one hidden node per tab inside its summary block; the summary itself is untouched", (t) => {
    const p = page();
    t.after(() => p.window.close());
    assert.equal(p.doc.querySelectorAll("[data-samextra-cc-a11y]").length, 2);
    for (const tab of ["txt2img", "img2img"]) {
        const pre = prefix(tab);
        const box = p.doc.querySelector(`#${pre}summary > [data-samextra-cc-a11y]`);
        assert.ok(box, tab);
        assert.equal(box.className, "samextra-cc-sr");                       // style.css: visually hidden, no layout
        assert.equal(box.getAttribute("data-samextra-cc-a11y"), tab);
        const status = p.status(tab);
        assert.equal(status.parentElement, box);
        assert.deepEqual(["role", "aria-live", "aria-atomic"].map((a) => status.getAttribute(a)), ["status", "polite", "true"]);
        assert.equal(status.textContent, "");
        for (const side of ["mod", "mask"]) {
            const desc = p.doc.getElementById(`${pre}a11y_${side}`);
            assert.equal(desc.parentElement, box);
            assert.equal(desc.hidden, true);                                  // read only through aria-describedby
        }
        assert.equal(p.describes(tab, "mod"), "편집 중: 수정자 I (Advanced)");
        assert.equal(p.describes(tab, "mask"), "편집 중: 마스크 M1");
        assert.equal(p.doc.querySelector(`#${pre}summary .prose`).textContent, "꺼짐 — 켠 수정자: I Advanced");
    }
});

test("the selectors and Pass are named radio groups with positions; keyboard semantics stay native", (t) => {
    const p = page();
    t.after(() => p.window.close());
    const group = (name) => p.doc.getElementById(`${P}${name}`);
    assert.equal(refText(p.doc, group("modifier_select"), "aria-labelledby"), "수정자");
    assert.equal(refText(p.doc, group("modifier_select"), "aria-describedby"), MOD_MARKS);
    assert.equal(refText(p.doc, group("mask_select"), "aria-labelledby"), "마스크 · 조합");
    assert.equal(refText(p.doc, group("mask_select"), "aria-describedby"), MASK_MARKS);
    assert.equal(refText(p.doc, group("mod_pass"), "aria-labelledby"), "Pass");
    assert.equal(group("mod_pass").hasAttribute("aria-describedby"), false);  // no info line under Pass
    for (const [name, size] of [["modifier_select", 10], ["mask_select", 15], ["mod_pass", 3]]) {
        const inputs = [...group(name).querySelectorAll("input[type='radio']")];
        assert.equal(inputs.length, size, name);
        assert.deepEqual(inputs.map((r) => [r.getAttribute("aria-posinset"), r.getAttribute("aria-setsize")]),
            inputs.map((_, i) => [String(i + 1), String(size)]), name);
    }
    const all = [...p.doc.querySelectorAll("input[type='radio']")];
    assert.ok(all.every((r) => !r.hasAttribute("tabindex") && !r.hasAttribute("role")), "no roving tabindex, no role");
    assert.equal(new Set(all.map((r) => r.name)).size, all.length);      // Gradio's own names, untouched
    // 선택 줄의 라디오는 편집기 설명을 달지 않는다(고르는 칸이지 편집하는 칸이 아니다)
    assert.ok([...group("modifier_select").querySelectorAll("input")].every((r) => !r.hasAttribute("aria-describedby")));
});

test("every editor control is described by the node it edits; descriptions it already had are kept", (t) => {
    const p = page();
    t.after(() => p.window.close());
    for (const tab of ["txt2img", "img2img"]) {
        const pre = prefix(tab);
        const groups = [["mod", SCHEMA.modifierFields, "mod"], ["leaf", SCHEMA.leafFields, "mask"],
            ["combo", SCHEMA.comboFields, "mask"]];
        for (const [kind, fields, side] of groups) {
            for (const f of fields) {
                const controls = [...p.doc.querySelectorAll(`#${pre}${kind}_${f.name} input, #${pre}${kind}_${f.name} button`)];
                assert.ok(controls.length > 0, `${kind}_${f.name}`);
                for (const el of controls) {
                    const ids = el.getAttribute("aria-describedby").split(" ");
                    assert.ok(ids.includes(`${pre}a11y_${side}`), `${tab} ${kind}_${f.name}`);
                    assert.ok(!ids.includes(`${pre}a11y_${side === "mod" ? "mask" : "mod"}`), `${tab} ${kind}_${f.name}`);
                }
            }
        }
        const strength = p.doc.querySelector(`#${pre}mod_strength input[type='number']`);
        assert.equal(strength.getAttribute("aria-describedby"), `${pre}strength_hint ${pre}a11y_mod`);
        assert.equal(refText(p.doc, strength, "aria-describedby"), "세기 편집 중: 수정자 I (Advanced)");
    }
});

test("a switch announces the node the editor now shows, once; the descriptions follow it", async (t) => {
    const p = page();
    t.after(() => p.window.close());
    const state = JSON.stringify({ v: 1, rev: "r0", "III.kind": "Chroma" });
    const out = p.cc.sync("txt2img", ["III", "M1", true, false, state, "I|M1|r0", ...defaults()]);
    assert.equal(out[2 + modIndex("kind")], "Chroma");
    assert.equal(p.status().textContent, "");                             // not before Gradio has rendered the update
    await afterNudge(p.window);
    assert.equal(p.status().textContent, "편집 중: 수정자 III (Chroma)");
    assert.equal(p.describes("txt2img", "mod"), "편집 중: 수정자 III (Chroma)");
    assert.equal(p.describes("txt2img", "mask"), "편집 중: 마스크 M1");

    // state.change: the echo of this write — nothing to load, nothing to say (not even the repeat marker)
    const shown = defaults("III");
    shown[modIndex("kind")] = "Chroma";
    assert.equal(p.cc.sync("txt2img", ["III", "M1", true, false, out[0], out[1], ...shown]), undefined);
    await afterNudge(p.window);
    assert.equal(p.status().textContent, "편집 중: 수정자 III (Chroma)");

    // the mask side: only the side that changed is announced
    const out2 = p.cc.sync("txt2img", ["III", "C2", true, true, out[0], out[1], ...shown]);
    assert.equal(out2.length, 2 + N_MOD + SCHEMA.leafFields.length + SCHEMA.comboFields.length + 7);
    await afterNudge(p.window);
    assert.equal(p.status().textContent, "편집 중: 조합 C2");
    assert.equal(p.describes("txt2img", "mask"), "편집 중: 조합 C2");
    assert.equal(p.describes("txt2img", "mod"), "편집 중: 수정자 III (Chroma)");
    // img2img is another panel: untouched
    assert.equal(p.status("img2img").textContent, "");
    assert.equal(p.describes("img2img", "mod"), "편집 중: 수정자 I (Advanced)");
});

test("Reset announces the node it reset, and a second press is announced again", async (t) => {
    const p = page();
    t.after(() => p.window.close());
    const state = JSON.stringify({ v: 1, rev: "r0", "III.kind": "Chroma", "III.exposure": 0.4 });
    const ed = defaults("III");
    ed[modIndex("kind")] = "Chroma";
    ed[modIndex("exposure")] = 0.4;
    const out = p.cc.reset("txt2img", "mod", [true, false, state, "III|M1|r0", ...ed]);
    assert.equal(out[2 + modIndex("kind")], "Advanced");
    await afterNudge(p.window);
    assert.equal(p.status().textContent, "초기화: 수정자 III (Advanced)");
    assert.equal(p.describes("txt2img", "mod"), "편집 중: 수정자 III (Advanced)");
    const first = p.status().textContent;
    p.cc.reset("txt2img", "mod", [true, false, out[0], out[1], ...defaults("III")]);
    await afterNudge(p.window);
    assert.notEqual(p.status().textContent, first);                     // changed text: a live region speaks again
    assert.equal(p.status().textContent.trim(), first);
    p.cc.reset("txt2img", "mask", [true, false, out[0], out[1], ...defaults("III")]);
    await afterNudge(p.window);
    assert.equal(p.status().textContent, "초기화: 마스크 M1");
});

test("a paste (a state rev this page never wrote) is announced once, naming both nodes", async (t) => {
    const p = page();
    t.after(() => p.window.close());
    const changes = [];
    new p.window.MutationObserver((records) => changes.push(...records))
        .observe(p.status(), { characterData: true, subtree: true, childList: true });
    // Forge's paste wrote state, ref, selectors and the editors in one response; Gradio then fires state.change and
    // both selectors' .change — three calls with the same inputs
    const state = JSON.stringify({ v: 1, rev: "pasted01", "II.active": true, "II.kind": "Luma", "M2.blur": 3 });
    const ed = defaults("II");
    ed[modIndex("active")] = true;
    ed[modIndex("kind")] = "Luma";
    for (let i = 0; i < 3; i++) assert.equal(p.cc.sync("txt2img", ["II", "M2", true, true, state, "II|M2|pasted01", ...ed]), undefined);
    await afterNudge(p.window);
    assert.equal(p.status().textContent, "붙여넣은 설정 불러옴 · 편집 중: 수정자 II (Luma) · 마스크 M2");
    assert.equal(changes.length, 1);
    assert.equal(p.describes("txt2img", "mod"), "편집 중: 수정자 II (Luma)");
    assert.equal(p.describes("txt2img", "mask"), "편집 중: 마스크 M2");
});

test("a Type change updates the description and says nothing", async (t) => {
    const p = page();
    t.after(() => p.window.close());
    assert.equal(p.cc.kind("txt2img", ["Punch"]).length, 10);
    await afterNudge(p.window);
    assert.equal(p.describes("txt2img", "mod"), "편집 중: 수정자 I (Punch)");
    assert.equal(p.status().textContent, "");
    p.cc.kind("txt2img", ["no such type"]);                             // shown as Advanced's groups; text unchanged
    await afterNudge(p.window);
    assert.equal(p.describes("txt2img", "mod"), "편집 중: 수정자 I (Punch)");
});

test("after mount the editor adds and removes no nodes and attaches no listeners", async (t) => {
    const p = page();
    t.after(() => p.window.close());
    const records = [];
    new p.window.MutationObserver((r) => records.push(...r)).observe(p.doc.body, { childList: true, subtree: true });
    const state = JSON.stringify({ v: 1, rev: "r0", "III.kind": "Chroma" });
    const out = p.cc.sync("txt2img", ["III", "C1", true, true, state, "I|M1|r0", ...defaults()]);
    p.cc.sync("txt2img", ["III", "C1", true, true, out[0], out[1], ...defaults("III")]);
    p.cc.reset("txt2img", "mask", [true, true, out[0], out[1], ...defaults("III")]);
    p.cc.kind("txt2img", ["Luma"]);
    p.cc.labels("txt2img", [true, true, out[0], out[1], ...defaults("III")]);
    p.cc.sync("img2img", ["II", "M1", true, false, JSON.stringify({ v: 1, rev: "p9" }), "II|M1|p9", ...defaults("II")]);
    await afterNudge(p.window);
    await afterNudge(p.window);
    assert.equal(p.status().textContent, "초기화: 조합 C1");
    assert.match(p.status("img2img").textContent, /^붙여넣은 설정 불러옴/);
    assert.deepEqual(records.map((r) => r.type), []);
    p.restore();
    assert.deepEqual(p.listeners, []);
});

test("a failure on the display side is logged and leaves every handler's values as they are", () => {
    const errors = [];
    const boom = () => { throw new Error("no timers on this page"); };
    const broken = loadEditor({ document: {}, setTimeout: boom, requestAnimationFrame: boom,
        console: { error: (...args) => errors.push(args.map(String).join(" ")) } });
    const plain = loadEditor();                                           // no document: nothing scheduled at all
    const sameRevs = (out) => JSON.parse(JSON.stringify(out === undefined ? null : out).replace(/\b[0-9a-f]{12}\b/g, "REV"));
    const state = JSON.stringify({ v: 1, rev: "r0", "III.kind": "Chroma", "C2.mask_a": "M3" });
    const calls = [
        ["sync", "txt2img", ["III", "C2", true, true, state, "I|M1|r0", ...defaults()]],   // switch
        ["sync", "txt2img", ["I", "M1", true, false, state, "I|M1|r0", ...defaults()]],    // echo: nothing to load
        ["reset", "txt2img", "mod", [true, false, state, "III|M1|r0", ...defaults("III")]],
        ["kind", "txt2img", ["Punch"]],
    ];
    for (const [name, ...args] of calls) {
        const got = broken.cc[name](...args);
        assert.deepEqual(sameRevs(got), sameRevs(plain.cc[name](...args)), name);
        if (name !== "sync" || args[1][0] === "III") assert.notEqual(got, undefined, name);
    }
    assert.equal(errors.length, calls.length);
    assert.ok(errors.every((e) => e.startsWith("[Colorcraft] a11y:")), errors.join("\n"));
});

test("real notebook.js: a fast-dropdown proxy installed after mount carries its node's description", async (t) => {
    const p = page({ notebook: true });
    t.after(() => p.window.close());                                     // notebook.js keeps intervals running
    p.restore();
    const hooks = p.window.__sam3NotebookTestHooks;
    assert.equal(hooks.installFastDropdown(`${P}mod_kind`, "Type"), true);
    assert.equal(hooks.installFastDropdown(`${P}combo_mask_a`, "Mask A"), true);
    const trigger = (name) => p.doc.querySelector(`#${P}${name} .sam3-fast-dropdown-trigger`);
    // at install, before any editor event: notebook.js copies the hidden input's description to its trigger
    assert.equal(trigger("mod_kind").getAttribute("aria-describedby"), `${P}a11y_mod`);
    assert.equal(trigger("combo_mask_a").getAttribute("aria-describedby"), `${P}a11y_mask`);
    assert.equal(trigger("mod_kind").getAttribute("aria-label"), "Type");
    const state = JSON.stringify({ v: 1, rev: "r0", "III.kind": "Chroma" });
    p.cc.sync("txt2img", ["III", "M1", true, false, state, "I|M1|r0", ...defaults()]);
    p.doc.querySelector(`#${P}mod_kind input[role='listbox']`).value = "Chroma";    // Gradio's flush
    await afterNudge(p.window);
    assert.equal(p.doc.querySelector(`#${P}mod_kind .sam3-fast-dropdown-value`).textContent, "Chroma");
    assert.equal(trigger("mod_kind").getAttribute("aria-describedby"), `${P}a11y_mod`);   // not duplicated
    assert.equal(refText(p.doc, trigger("mod_kind"), "aria-describedby"), "편집 중: 수정자 III (Chroma)");
});
