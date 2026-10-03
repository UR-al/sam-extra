// Colorcraft 공유 편집기(javascript/colorcraft_editor.js) — DOM 없는 vm 에서 Gradio 4.40 의 frontend_fn 처럼 부른다
// (inputs 뒤에 outputs 의 현재 값). 끝의 jsdom 부분은 실제 colorcraft_sliders.js 와 함께 올려 빠른 드롭다운 맞추기(0ms 타이머 →
// 한 프레임 뒤) · 숫자 칸 범위 풀기 · Ctrl+Enter 순서를 본다. 맞추기가 Gradio 가 그린 값과 같은 화면 갱신에 드는지는
// colorcraft_editor_timing.test.mjs, 접근성(선택 줄 이름, 편집기 칸 설명, 알림)은 colorcraft_editor_a11y.test.mjs 가 본다.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { JSDOM } from "jsdom";
import { loadEditor, callJs, JS_DIR } from "./colorcraft_bridge.mjs";

const env = loadEditor();
const { cc, schema } = env;
const N_MOD = schema.modifierFields.length;
const N_LEAF = schema.leafFields.length;
const N_COMBO = schema.comboFields.length;
const N_ED = N_MOD + N_LEAF + N_COMBO;
const SYNC_OUT = 2 + N_ED + 2 + 2 + 3;
const RESET_OUT = 2 + N_ED + 3;
const js = (name, ...extra) => `(...a) => window.samextraColorcraft ? window.samextraColorcraft.${name}(${["'txt2img'", ...extra.map((e) => `'${e}'`), "a"].join(", ")}) : undefined`;

function defaults(mod = "I", mask = "M1") {
    const i = schema.modifierTags.indexOf(mod);
    const m = schema.modifierFields.map((f) => (f.name === "active" ? schema.activeDefaults[i] : f.default));
    const l = schema.leafFields.map((f) => f.default);
    const c = schema.comboFields.map((f) => f.default);
    return [...m, ...l, ...c];
}
const modIndex = (name) => schema.modifierFields.findIndex((f) => f.name === name);
const leafIndex = (name) => N_MOD + schema.leafFields.findIndex((f) => f.name === name);
const isSkip = (v) => v && typeof v === "object" && v.__type__ === "update" && Object.keys(v).length === 1;
// 편집기가 Gradio 가 그린 뒤 하는 일(0ms 타이머 → 한 프레임)이 끝난 뒤: 같은 순서로 하나 더 걸면 그 뒤에 돈다.
const afterNudge = (window) => new Promise((r) => window.setTimeout(() => window.requestAnimationFrame(() => r()), 0));

test("sync: nothing to do when the editors already show the targets of this state", async () => {
    const out = await callJs(env, js("sync"), ["I", "M1", true, false, "", "I|M1|0", ...defaults()], SYNC_OUT);
    assert.deepEqual(out, []);
});

test("sync: switching tab commits the shown tab and loads the new one in one update", async () => {
    const ed = defaults();
    ed[modIndex("exposure")] = 0.5;
    ed[leafIndex("mask_hardness")] = 3.3;
    const out = await callJs(env, js("sync"), ["II", "M1", true, false, "", "I|M1|0", ...ed], SYNC_OUT);
    assert.equal(out.length, SYNC_OUT);
    const state = JSON.parse(out[0]);
    assert.equal(state["I.exposure"], 0.5);
    assert.equal(state["M1.mask_hardness"], 3.3);
    assert.equal(state.v, 1);
    assert.equal(out[1], `II|M1|${state.rev}`);
    assert.notEqual(state.rev, "0");
    assert.equal(out[2 + modIndex("exposure")], 0);                       // II's own value loaded
    assert.equal(out[2 + modIndex("active")], false);
    assert.ok(out.slice(2 + N_MOD, 2 + N_ED).every(isSkip));              // mask editors untouched
    const labels = out[SYNC_OUT - 3].choices.map((c) => c[0]);
    assert.deepEqual(labels.slice(0, 2), ["I ●", "II"]);
});

test("sync: a state the editors do not belong to (rev mismatch) is loaded, never overwritten", async () => {
    const pasted = JSON.stringify({ v: 1, rev: "p1", "II.active": true, "II.exposure": -0.4 });
    const ed = defaults();
    ed[modIndex("exposure")] = 2.0;                                      // stale editor value
    const out = await callJs(env, js("sync"), ["II", "M1", true, false, pasted, "I|M1|0", ...ed], SYNC_OUT);
    assert.ok(isSkip(out[0]));                                            // state kept as pasted
    assert.equal(out[1], "II|M1|p1");
    assert.equal(out[2 + modIndex("exposure")], -0.4);
    assert.equal(out[2 + N_MOD + schema.leafFields.findIndex((f) => f.name === "mask_axis")], schema.leafFields[0].default);
});

test("sync: a combo gets its own reference list", async () => {
    const out = await callJs(env, js("sync"), ["I", "C2", true, true, "", "I|M1|0", ...defaults()], SYNC_OUT);
    const maskA = out[2 + N_MOD + N_LEAF];
    assert.deepEqual(maskA.choices.map((c) => c[1]), ["none", ...schema.maskTags, "C1"]);
    assert.equal(maskA.value, "none");
    assert.deepEqual([out[2 + N_ED].visible, out[2 + N_ED + 1].visible], [false, true]);
});

test("commit: defaults and emptied number fields leave the state", () => {
    const st = cc._pure.parseState(JSON.stringify({ v: 1, rev: "r", "I.exposure": 0.3, "I.tint": 1 }));
    const ed = defaults();
    ed[modIndex("exposure")] = 0;                                         // back to the default
    ed[modIndex("tint")] = null;                                          // emptied field: the server reads the default
    ed[modIndex("contrast")] = 7.5;                                       // typed past the slider range: kept
    cc._pure.commit(st.map, { mod: "I", mask: "M1", rev: "r" }, ed);
    assert.deepEqual(JSON.parse(JSON.stringify(st.map)), { "I.contrast": 7.5 });
});

test("reset: keeps Active, reloads only that editor", async () => {
    const state = JSON.stringify({ v: 1, rev: "r", "II.active": true, "II.exposure": 0.7, "M1.blur": 3 });
    const ed = defaults("II");
    ed[modIndex("active")] = true;
    ed[modIndex("exposure")] = 0.9;
    ed[leafIndex("blur")] = 4;
    const out = await callJs(env, js("reset", "mod"), [true, false, state, "II|M1|r", ...ed], RESET_OUT);
    const next = JSON.parse(out[0]);
    assert.equal(next["II.active"], true);
    assert.equal(next["II.exposure"], undefined);
    assert.equal(next["M1.blur"], 4);                                     // the mask editor's edit was committed
    assert.equal(out[2 + modIndex("exposure")], 0);
    assert.ok(out.slice(2 + N_MOD, 2 + N_ED).every(isSkip));
});

test("labels: ● active, ○ edited but off, masks reached through combos", async () => {
    const state = JSON.stringify({ v: 1, rev: "r", "III.exposure": 0.2, "II.active": true, "II.kind": "Chroma",
        "II.mask": "C1", "C1.mask_a": "M2", "C1.mask_b": "M5" });
    const out = await callJs(env, js("labels"), [true, true, state, "I|M1|r", ...defaults()], 3);
    const mods = out[0].choices.map((c) => c[0]);
    const masks = out[1].choices.map((c) => c[0]);
    assert.deepEqual(mods.slice(0, 3), ["I ●", "II ●", "III ○"]);
    assert.deepEqual(masks.filter((m) => m.endsWith("●")), ["M2 ●", "M5 ●", "C1 ●"]);
    assert.match(out[2], /켠 수정자: I Advanced, II Chroma ➜ C1 · 쓰는 마스크: M2, M5, C1/);
});

test("labels: an incomplete combo marks nothing in use and the summary says (미완성)", async () => {
    // C1 has only A; C2 names C1, so neither resolves (spec.build_mask_specs) and II runs unmasked
    const state = JSON.stringify({ v: 1, rev: "r", "II.active": true, "II.kind": "Chroma", "II.mask": "C2",
        "C1.mask_a": "M3", "C2.mask_a": "C1", "C2.mask_b": "M4" });
    const out = await callJs(env, js("labels"), [true, true, state, "I|M1|r", ...defaults()], 3);
    const masks = out[1].choices.map((c) => c[0]);
    assert.deepEqual(masks.filter((m) => m.endsWith("●")), []);
    assert.deepEqual(masks.filter((m) => m.endsWith("○")), ["C1 ○", "C2 ○"]);
    assert.match(out[2], /켠 수정자: I Advanced, II Chroma ➜ C2\(미완성\) · 쓰는 마스크: 없음</);
    assert.deepEqual([...cc._pure.resolvedCombos(cc._pure.parseState(state).map)], []);
});

test("labels: editors that do not belong to the state (rev mismatch) are not counted", async () => {
    const state = JSON.stringify({ v: 1, rev: "p1", "III.exposure": 0.2 });
    const ed = defaults("II");
    ed[modIndex("active")] = true;                                       // a stale II editor, never committed
    ed[modIndex("exposure")] = 0.4;
    const out = await callJs(env, js("labels"), [true, false, state, "II|M1|old", ...ed], 3);
    assert.deepEqual(out[0].choices.map((c) => c[0]).slice(0, 3), ["I ●", "II", "III ○"]);
});

test("reset: a state the editors do not belong to reloads both editors and commits nothing", async () => {
    const pasted = JSON.stringify({ v: 1, rev: "p1", "I.exposure": -0.4, "M1.blur": 3 });
    const ed = defaults();
    ed[modIndex("exposure")] = 2.0;                                       // stale editor values
    ed[leafIndex("blur")] = 9;
    for (const section of ["mod", "mask"]) {
        const out = await callJs(env, js("reset", section), [true, false, pasted, "I|M1|old", ...ed], RESET_OUT);
        const next = JSON.parse(out[0]);
        assert.equal(next["I.exposure"], section === "mod" ? undefined : -0.4, section);
        assert.equal(next["M1.blur"], section === "mask" ? undefined : 3, section);
        assert.equal(out[1], `I|M1|${next.rev}`);
        assert.equal(out[2 + modIndex("exposure")], section === "mod" ? 0 : -0.4, section);  // loaded, not SKIP
        assert.equal(out[2 + leafIndex("blur")], section === "mask" ? 0 : 3, section);
    }
});

test("parseState: only a writer's rev is a rev (as panel_state._rev)", () => {
    const rev = (r) => cc._pure.parseState(JSON.stringify({ v: 1, rev: r, "I.exposure": 0.3 })).rev;
    for (const bad of [7, 1.0, 0, "0", "", "a|b", null, true, ["r"]]) assert.equal(rev(bad), null, String(bad));
    assert.equal(rev("3fa9c1d2e4b0"), "3fa9c1d2e4b0");
    assert.equal(cc._pure.parseState("").rev, "0");
    assert.equal(cc._pure.parseState('{"v": true}'), null);
    assert.equal(cc._pure.parseState('{"v": 1.0, "rev": "r"}').rev, "r");
});

test("sync: a non-empty state carrying rev 0 is loaded and given a rev, never committed over", async () => {
    const outside = JSON.stringify({ v: 1, rev: "0", "I.exposure": 0.3 });
    const ed = defaults();
    ed[modIndex("exposure")] = 1.5;                                       // the default editors of ref I|M1|0
    const out = await callJs(env, js("sync"), ["II", "M1", true, false, outside, "I|M1|0", ...ed], SYNC_OUT);
    const next = JSON.parse(out[0]);
    assert.equal(next["I.exposure"], 0.3);
    assert.notEqual(next.rev, "0");
    assert.equal(out[1], `II|M1|${next.rev}`);
});

test("kind: each Type shows only its node's groups", async () => {
    for (const [label, id] of Object.entries(schema.kindIds)) {
        const out = await callJs(env, js("kind"), [label], 10);
        const shown = schema.groupOrder.filter((g, i) => out[i].visible);
        assert.deepEqual(shown, schema.groupOrder.filter((g) => schema.kindGroups[id].includes(g)), label);
        const gates = schema.kindGates[id] || [];
        assert.deepEqual([out[7].visible, out[8].visible], [gates.includes("more_colors"), gates.includes("color_shift")], label);
        assert.equal(out[9].visible, schema.kindTakesMask[id], label);
    }
});

test("unreadable state: no handler writes anything", async () => {
    assert.deepEqual(await callJs(env, js("sync"), ["II", "M1", true, false, "{bad", "I|M1|0", ...defaults()], SYNC_OUT), []);
    assert.deepEqual(await callJs(env, js("labels"), [true, false, '{"v":2}', "I|M1|0", ...defaults()], 3), []);
});

test("commit: a combo's Mask A/B outside its own reference list is the default (as Python coerce)", () => {
    const st = cc._pure.parseState("");
    const ed = defaults();
    const a = N_MOD + N_LEAF + schema.comboFields.findIndex((f) => f.name === "mask_a");
    ed[a] = "C1";                                                      // C1 cannot reference itself
    cc._pure.commit(st.map, { mod: "I", mask: "C1", rev: "0" }, ed);
    assert.equal(JSON.parse(JSON.stringify(st.map))["C1.mask_a"], undefined);
    cc._pure.commit(st.map, { mod: "I", mask: "C2", rev: "0" }, ed);  // C2 may reference C1
    assert.equal(JSON.parse(JSON.stringify(st.map))["C2.mask_a"], "C1");
});

// jsdom: sam-extra 의 notebook.js 빠른 드롭다운(가짜)이 불러온 값을 Gradio 가 그린 뒤(0ms 타이머 → 한 프레임)에 듣는다
test("sync nudges the fast-dropdown proxies after Gradio's flush (values and the combo's list)", async (t) => {
    const P = "script_txt2img_colorcraft_samextra_";
    const dd = (name, value) => `<div class="gradio-dropdown" id="${P}${name}"><input role="listbox" value="${value}"></div>`;
    const dom = new JSDOM(`<!doctype html><body><div id="${P}accordion">${dd("mod_kind", "Chroma")}${dd("combo_mask_a", "none")}${dd("combo_mask_b", "M3")}</div></body>`,
        { runScripts: "outside-only", pretendToBeVisual: true });
    const { window } = dom;
    t.after(() => window.close());                                    // also when an assertion fails
    window.eval(readFileSync(path.join(JS_DIR, "colorcraft_schema.js"), "utf8"));
    window.eval(readFileSync(path.join(JS_DIR, "colorcraft_editor.js"), "utf8"));
    const synced = [], choices = [];
    for (const name of ["mod_kind", "combo_mask_a", "combo_mask_b"]) {
        const w = window.document.getElementById(P + name);
        w.__sam3FastDropdownSync = (v) => synced.push([name, v]);
        w.__sam3FastDropdownSetChoices = (c) => choices.push([name, [...c]]);
    }
    const out = window.samextraColorcraft.sync("txt2img", ["I", "C2", true, true, "", "I|M1|0", ...defaults()]);
    assert.equal(out.length, SYNC_OUT);
    assert.deepEqual(synced, []);                                     // not before Gradio has applied the update
    await new Promise((r) => window.setTimeout(r, 0));
    assert.deepEqual(synced, []);                                     // nor in the 0 ms timer: only in the frame after it
    await afterNudge(window);
    assert.deepEqual(synced.map((x) => x[0]).sort(), ["combo_mask_a", "combo_mask_b", "mod_kind"]);
    assert.deepEqual(synced.find((x) => x[0] === "mod_kind")[1], "Chroma");
    assert.deepEqual(choices.map((x) => x[0]).sort(), ["combo_mask_a", "combo_mask_b"]);
    assert.deepEqual(choices[0][1], ["none", ...schema.maskTags, "C1"]);
});

test("guard: a handler that throws logs one console.error and returns its failure value", async () => {
    const errors = [];
    const guarded = loadEditor({ console: { error: (...args) => errors.push(args.map(String).join(" ")) } });
    // null: the page's js string then shows the fallback (tests/test_colorcraft_editor_parity.py runs it)
    assert.equal(guarded.cc.sync("txt2img", null), null);                // a = null: the destructuring throws
    assert.equal(errors.length, 1);
    assert.match(errors[0], /^\[Colorcraft\] sync:/);
    assert.equal(guarded.cc.labels("txt2img", null), null);
    assert.equal(guarded.cc.reset("txt2img", "mod", null), null);
    assert.equal(errors.length, 3);
    assert.match(errors[1], /^\[Colorcraft\] labels:/);
    assert.equal(guarded.cc.kind("txt2img", null), undefined);           // kind: no update
    assert.equal(errors.length, 4);
});

// jsdom: 스키마 · 편집기 · 실제 colorcraft_sliders.js 를 Forge 와 같은 id 의 페이지에 올린다.
const PRE = "script_txt2img_colorcraft_samextra_";
const SLIDERS = readFileSync(path.join(JS_DIR, "colorcraft_sliders.js"), "utf8");

function forgePage() {
    const field = `<div class="gradio-slider" id="${PRE}mod_exposure"><input type="number" min="-2" max="2" step="0.01" value="0.3"></div>`;
    // img2img 아코디언도 둔다: 없으면 colorcraft_sliders.js 가 페이지 전체를 지켜보다 window.close() 뒤에 오류를 낸다
    const i2i = '<div id="script_img2img_colorcraft_samextra_accordion"></div>';
    const dom = new JSDOM(`<!doctype html><html><body><div id="${PRE}accordion">${field}</div>${i2i}</body></html>`,
        { url: "http://127.0.0.1:7860/", runScripts: "outside-only", pretendToBeVisual: true });
    const { window } = dom;
    window.gradioApp = () => window.document;
    window.onUiLoaded = (fn) => fn();
    const added = [];                                  // 편집기를 올리는 동안 붙은 이벤트 리스너
    const original = window.EventTarget.prototype.addEventListener;
    window.EventTarget.prototype.addEventListener = function (type, ...rest) {
        added.push(type);
        return original.call(this, type, ...rest);
    };
    window.eval(readFileSync(path.join(JS_DIR, "colorcraft_schema.js"), "utf8"));
    window.eval(readFileSync(path.join(JS_DIR, "colorcraft_editor.js"), "utf8"));
    const editorListeners = added.slice();
    window.EventTarget.prototype.addEventListener = original;
    window.eval(SLIDERS);
    const input = window.document.querySelector(`#${PRE}mod_exposure input[type=number]`);
    return { dom, window, input, editorListeners };
}

test("jsdom (a): the editor exports its four handlers and adds no listeners", (t) => {
    const { window, editorListeners } = forgePage();
    t.after(() => window.close());
    for (const name of ["sync", "reset", "labels", "kind"]) {
        assert.equal(typeof window.samextraColorcraft[name], "function", name);
    }
    assert.equal(typeof window.samextraColorcraftSchema, "object");
    assert.deepEqual(editorListeners, []);
});

test("jsdom (b): the editor's number fields are unclamped by colorcraft_sliders.js", (t) => {
    const { window, input } = forgePage();
    t.after(() => window.close());
    assert.equal(input.hasAttribute("min"), false);
    assert.equal(input.hasAttribute("max"), false);
});

test("jsdom (c): Ctrl+Enter on a field holding text writes 0 (input event) before Forge's bubbling shortcut runs", (t) => {
    const { window, input } = forgePage();
    t.after(() => window.close());
    const order = [];
    input.addEventListener("input", () => order.push(`input:${input.value}`));
    window.document.addEventListener("keydown", (e) => {               // Forge script.js: Ctrl+Enter → Generate
        if (e.key === "Enter" && e.ctrlKey) order.push(`generate:${input.value}`);
    });
    input.focus();
    input.value = "abc";
    input.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter", ctrlKey: true, bubbles: true }));
    assert.deepEqual(order, ["input:0", "generate:0"]);
});

// 실제 notebook.js(테스트 이음매 window.__sam3NotebookTestHooks)로 Colorcraft 드롭다운에 빠른 드롭다운을 설치하고, 편집기의
// nudge 가 그 표시(.sam3-fast-dropdown-value)와 조합 목록을 바꾸는지 본다 — 두 훅 이름이 양쪽에서 맞는지 지키는 테스트.
const NOTEBOOK = readFileSync(path.join(JS_DIR, "notebook.js"), "utf8");

test("real notebook.js: the nudge updates the installed proxies' value and the combo's list", async (t) => {
    const dd = (name, value) => `<div id="${PRE}${name}" class="gradio-dropdown"><div class="container">`
        + `<input role="listbox" value="${value}"></div></div>`;
    const dom = new JSDOM(`<!doctype html><html><body><div class="gradio-container"><div id="${PRE}accordion">`
        + `${dd("mod_kind", "Advanced")}${dd("combo_mask_a", "none")}</div></div></body></html>`,
    { url: "http://127.0.0.1:7860/", runScripts: "outside-only", pretendToBeVisual: true });
    const { window } = dom;
    t.after(() => window.close());                                    // notebook.js keeps intervals running
    window.gradioApp = () => window.document;
    window.onUiLoaded = () => {};
    window.opts = {};
    window.gradio_config = { components: [
        { id: 1, props: { elem_id: `${PRE}mod_kind`, choices: Object.keys(schema.kindIds).map((k) => [k, k]) } },
        { id: 2, props: { elem_id: `${PRE}combo_mask_a`, choices: schema.comboRefChoices[4].map((c) => [c, c]) } },
    ] };
    window.CSS = window.CSS || {};
    window.CSS.escape = window.CSS.escape || ((value) => String(value));
    window.Element.prototype.scrollIntoView = function () {};           // jsdom 에 없다(팝오버가 고른 항목으로 스크롤)
    window.__sam3NotebookTestHooks = {};
    window.eval(NOTEBOOK);
    window.eval(readFileSync(path.join(JS_DIR, "colorcraft_schema.js"), "utf8"));
    window.eval(readFileSync(path.join(JS_DIR, "colorcraft_editor.js"), "utf8"));
    const hooks = window.__sam3NotebookTestHooks;
    assert.equal(hooks.installFastDropdown(`${PRE}mod_kind`, "Type"), true);
    assert.equal(hooks.installFastDropdown(`${PRE}combo_mask_a`, "Mask A"), true);
    const doc = window.document;
    const shown = (name) => doc.querySelector(`#${PRE}${name} .sam3-fast-dropdown-value`).textContent;
    const offered = (name) => {
        doc.querySelector(`#${PRE}${name} .sam3-fast-dropdown-trigger`).click();
        const pop = doc.getElementById(`sam3-fast-dropdown-${PRE}${name}`);
        const list = [...pop.querySelectorAll(".sam3-fast-dropdown-option")].map((b) => b.textContent);
        doc.querySelector(`#${PRE}${name} .sam3-fast-dropdown-trigger`).click();
        return list;
    };
    assert.equal(shown("mod_kind"), "Advanced");
    assert.ok(offered("combo_mask_a").includes("C4"));                   // the config's full list

    // sync: II (Type Chroma in the state) and combo C2 are loaded; Gradio then writes the hidden inputs
    const state = JSON.stringify({ v: 1, rev: "r", "II.active": true, "II.kind": "Chroma" });
    const out = window.samextraColorcraft.sync("txt2img", ["II", "C2", true, true, state, "I|M1|r", ...defaults()]);
    assert.equal(out.length, SYNC_OUT);
    assert.equal(out[2 + modIndex("kind")], "Chroma");
    doc.querySelector(`#${PRE}mod_kind input[role='listbox']`).value = "Chroma";
    assert.equal(shown("mod_kind"), "Advanced");                         // nothing before Gradio's flush
    await afterNudge(window);
    assert.equal(shown("mod_kind"), "Chroma");
    assert.deepEqual(offered("combo_mask_a"), ["none", ...schema.maskTags, "C1"]);
});
