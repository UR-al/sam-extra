// composition_ui.js — 구도 · 카메라 칸. 실제 javascript/composition_prompt.js 와 composition_ui.js 를 jsdom 에 올린다.
//
// 자리는 scripts/composition_camera.py 가 만드는 Gradio 4.40 모양 그대로다: gr.HTML 은 div#sam3_composition_<탭>.block
// > div > div.prose > (값) div.sam3-composition-mount, 프롬프트는 div#<탭>_prompt > label > textarea. Gradio 의 Textbox 는
// textarea 의 input 이벤트로 값을 받으므로(Svelte bind:value) 그 리스너를 흉내 내 'Gradio 값' 을 적는다. Forge 의
// updateInput 은 javascript/ui.js:415 를 그대로 옮겼다. jsdom 에는 execCommand·PointerEvent·포인터 캡처가 없어 테스트가
// 만든다 — 가짜 execCommand('insertText')는 Chrome 처럼 포커스된 textarea 의 선택 자리에 넣고 inputType insertText 인
// input 이벤트를 낸다.
// 앞쪽 테스트들은 앱의 frontend/src/components/CompositionControl.test.ts 사례를 Forge 쪽 모양으로 옮긴 것이다.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM, VirtualConsole } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const LOGIC = readFileSync(path.join(ROOT, "javascript", "composition_prompt.js"), "utf8");
const UI = readFileSync(path.join(ROOT, "javascript", "composition_ui.js"), "utf8");
// Forge javascript/ui.js:415 (verbatim)
const FORGE_UPDATE_INPUT = `
function updateInput(target) {
    const e = new Event("input", { bubbles: true });
    Object.defineProperty(e, "target", { value: target });
    target.dispatchEvent(e);
}`;

const DEFAULT = { azimuth: 0, elevation: 0, distance: 45, roll: 0, framing: 0 };
const KEY = (tab) => `sam-extra.composition.${tab}.v1`;

const promptBlock = (tab) =>
  `<div id="${tab}_prompt" class="block gradio-textbox"><label><textarea data-testid="textbox"></textarea></label></div>`;
const mountBlock = (tab) =>
  `<div id="sam3_composition_${tab}" class="block gradio-html sam3-composition-host"><div>`
  + `<div class="prose gradio-html sam3-composition-host">`
  + `<div class="sam3-composition-mount" data-sam3-composition-tab="${tab}"></div>`
  + `</div></div></div>`;

// exec: "insert" (Chrome) · "false" (명령이 false) · "noop" (true 인데 아무것도 안 함) · "throw" · "missing" (함수 없음)
function load(t, { mounts = ["txt2img", "img2img"], prompts = {}, storage = {}, exec = "insert", updateInput = true,
  storageBlocked = false, logic = true } = {}) {
  const errors = [];
  const virtualConsole = new VirtualConsole();
  virtualConsole.on("jsdomError", (error) => errors.push(String(error && (error.stack || error.message || error))));
  virtualConsole.on("error", (...args) => errors.push(args.join(" ")));
  const body = ["txt2img", "img2img"].map((tab) =>
    `<div id="tab_${tab}" class="tabitem">${promptBlock(tab)}${mounts.includes(tab) ? mountBlock(tab) : ""}</div>`).join("");
  const dom = new JSDOM(`<!doctype html><html><body>${body}</body></html>`, {
    url: "http://127.0.0.1:7860/", runScripts: "outside-only", pretendToBeVisual: true, virtualConsole,
  });
  t.after(() => dom.window.close());
  const { window } = dom;
  const doc = window.document;
  for (const [key, value] of Object.entries(storage)) window.localStorage.setItem(key, value);
  if (storageBlocked) {
    window.Storage.prototype.getItem = () => { throw new window.DOMException("blocked", "SecurityError"); };
    window.Storage.prototype.setItem = () => { throw new window.DOMException("blocked", "SecurityError"); };
  }

  // Gradio's Textbox listener (registered when Gradio mounts, before any extension script)
  const gradio = {};
  for (const tab of ["txt2img", "img2img"]) {
    const area = doc.querySelector(`#${tab}_prompt textarea`);
    area.value = prompts[tab] || "";
    gradio[tab] = { value: area.value, inputTypes: [] };
    area.addEventListener("input", (event) => {
      gradio[tab].value = area.value;
      gradio[tab].inputTypes.push(event.inputType);
    });
  }

  const execCalls = [];
  if (exec !== "missing") {
    doc.execCommand = (command, showUi, value) => {
      const active = doc.activeElement;
      execCalls.push({
        command, value, active, start: active && active.selectionStart, end: active && active.selectionEnd,
        length: active && active.value !== undefined ? active.value.length : null,
      });
      if (exec === "throw") throw new window.DOMException("not supported", "NotSupportedError");
      if (exec === "false") return false;
      if (exec === "noop") return true;
      if (command !== "insertText" || !active || active.tagName !== "TEXTAREA") return false;
      const { selectionStart: start, selectionEnd: end } = active;
      active.value = active.value.slice(0, start) + value + active.value.slice(end);
      active.selectionStart = active.selectionEnd = start + value.length;
      active.dispatchEvent(new window.InputEvent("input", { bubbles: true, inputType: "insertText", data: value }));
      return true;
    };
  }

  const captures = new Map();
  const pointerLog = [];
  window.Element.prototype.setPointerCapture = function (id) { captures.set(id, this); pointerLog.push(["set", id]); };
  window.Element.prototype.hasPointerCapture = function (id) { return captures.get(id) === this; };
  window.Element.prototype.releasePointerCapture = function (id) { captures.delete(id); pointerLog.push(["release", id]); };

  const loaded = [];
  const afterUpdates = [];
  window.gradioApp = () => doc;
  window.onUiLoaded = (fn) => loaded.push(fn);
  window.onAfterUiUpdate = (fn) => afterUpdates.push(fn);
  const updateCalls = [];
  if (updateInput) {
    window.eval(FORGE_UPDATE_INPUT);
    const forge = window.updateInput;
    window.updateInput = (target) => { updateCalls.push(target); forge(target); };
  }
  window.__sam3CompositionTestHooks = {};
  if (logic) window.eval(LOGIC);
  window.eval(UI);
  loaded.forEach((fn) => fn());

  const q = (selector) => doc.querySelector(selector);
  const panel = (tab = "txt2img") => q(`#sam3_composition_${tab} .sam3-composition`);
  const env = {
    window, doc, errors, gradio, execCalls, updateCalls, pointerLog, captures, afterUpdates,
    hooks: window.__sam3CompositionTestHooks,
    panel,
    area: (tab = "txt2img") => q(`#${tab}_prompt textarea`),
    button: (label, tab = "txt2img") => [...panel(tab).querySelectorAll("button")].find((b) => b.textContent.includes(label)),
    range: (key, tab = "txt2img") => panel(tab).querySelector(`input[type="range"][data-key="${key}"]`),
    output: (key, tab = "txt2img") => panel(tab).querySelector(`output[for="sam3-composition-${tab}-${key}"]`),
    orbit: (tab = "txt2img") => panel(tab).querySelector('[aria-label="카메라 구도 조작"]'),
    append: (tab = "txt2img") => panel(tab).querySelector(".sam3-composition__append"),
    preview: (tab = "txt2img") => panel(tab).querySelector(".sam3-composition__preview p").textContent,
    warning: (tab = "txt2img") => panel(tab).querySelector(".sam3-composition__warning"),
    conflicts: (tab = "txt2img") => [...panel(tab).querySelectorAll(".sam3-composition__warning li")]
      .filter((li) => !li.hidden).map((li) => li.textContent),
    feedback: (tab = "txt2img") => panel(tab).querySelector(".sam3-composition__feedback").textContent,
    stored: (tab = "txt2img") => JSON.parse(window.localStorage.getItem(KEY(tab))),
    values: (tab = "txt2img") => Object.fromEntries(Object.keys(DEFAULT).map((key) => [key, Number(env.range(key, tab).value)])),
    type(text, tab = "txt2img") {
      const area = env.area(tab);
      area.value = text;
      area.dispatchEvent(new window.InputEvent("input", { bubbles: true, inputType: "insertText" }));
    },
    slide(key, value, tab = "txt2img", { release = false } = {}) {
      const input = env.range(key, tab);
      input.value = String(value);
      input.dispatchEvent(new window.Event("input", { bubbles: true }));
      if (release) input.dispatchEvent(new window.Event("change", { bubbles: true }));
    },
    key(key, init = {}, tab = "txt2img") {
      const event = new window.KeyboardEvent("keydown", { key, bubbles: true, cancelable: true, ...init });
      env.orbit(tab).dispatchEvent(event);
      return event;
    },
    pointer(type, init = {}, tab = "txt2img") {
      const event = new window.MouseEvent(type, {
        bubbles: true, cancelable: true, clientX: init.clientX || 0, clientY: init.clientY || 0, button: init.button || 0,
      });
      Object.defineProperty(event, "pointerId", { value: init.pointerId === undefined ? 1 : init.pointerId });
      Object.defineProperty(event, "isPrimary", { value: init.isPrimary === undefined ? true : init.isPrimary });
      env.orbit(tab).dispatchEvent(event);
      return event;
    },
  };
  return env;
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 5));

// ---- 앱 CompositionControl.test.ts 사례 ----------------------------------------------------

test("renders labeled native controls and never changes the prompt on preset or range changes", (t) => {
  const env = load(t, { prompts: { txt2img: "forest" } });
  const panel = env.panel();
  assert.equal(panel.querySelectorAll('input[type="range"]').length, 5);
  assert.equal(env.orbit().getAttribute("tabindex"), "0");
  assert.match(panel.textContent, /실제 3D 카메라 제어가 아니며/);
  env.button("낮은 시점").click();
  env.slide("elevation", 60);
  assert.equal(env.area().value, "forest");
  assert.equal(env.gradio.txt2img.value, "forest");
  assert.deepEqual([env.execCalls.length, env.updateCalls.length], [0, 0]);
  assert.match(env.preview(), /from above/);
  assert.equal(env.stored().elevation, -30, "the preset is saved; a slider is saved when released");
  env.slide("elevation", 60, "txt2img", { release: true });
  assert.equal(env.stored().elevation, 60);
  assert.deepEqual(env.errors, []);
});

test("appends explicitly through main tags and repeated clicks cannot duplicate tags", (t) => {
  // the app passed the character/prefix/suffix sections as otherPrompts; Forge has one prompt, so the tag is in it
  const env = load(t, { prompts: { txt2img: "forest, upper_body" } });
  env.append().click();
  assert.equal(env.area().value, "forest, upper_body, facing viewer, centered composition");
  assert.equal(env.append().disabled, true);
  assert.match(env.append().textContent, /이미 포함된/);
  env.append().click();                       // a disabled button: nothing
  env.append().disabled = false;              // and even when the click lands, the click re-plans from the prompt
  env.append().click();
  assert.equal(env.execCalls.length, 1);
  assert.equal(env.area().value, "forest, upper_body, facing viewer, centered composition");
  assert.match(env.feedback(), /2개를 추가했습니다/);
  assert.equal(env.feedback(), "메인 태그에 2개를 추가했습니다. 기존 프롬프트는 유지했습니다.");
  assert.deepEqual(env.errors, []);
});

test("exposes conflict-aware append without deleting the previous composition", (t) => {
  const env = load(t, { prompts: { txt2img: "from below, full body" } });
  env.button("위에서").click();
  assert.equal(env.warning().hidden, false);
  assert.match(env.warning().textContent, /기존 태그는 삭제하지 않습니다/);
  assert.deepEqual(env.conflicts(), ["from below ↔ from above", "full body ↔ close-up"]);
  env.button("기존 구도 유지하고 추가").click();
  assert.ok(env.area().value.startsWith("from below, full body, "), env.area().value);
  assert.equal(env.area().value, "from below, full body, three-quarter view from the subject's right, from above, close-up, "
    + "centered composition");
});

test("restores the saved state per tab and repairs corrupt or obsolete values like the app", (t) => {
  const env = load(t, {
    storage: {
      [KEY("txt2img")]: JSON.stringify({ ...DEFAULT, azimuth: -90 }),
      [KEY("img2img")]: '{"azimuth": 999, "elevation": "12", "distance": 33.4, "roll": null, "framing": -1e9}',
    },
  });
  assert.equal(env.range("azimuth").value, "-90");
  assert.deepEqual(env.values("img2img"), { azimuth: 180, elevation: 0, distance: 33, roll: 0, framing: -100 });
  env.slide("azimuth", 45, "txt2img", { release: true });
  assert.equal(env.range("azimuth").value, "45");
  assert.equal(env.stored("txt2img").azimuth, 45);
  assert.equal(env.stored("img2img").azimuth, 999, "the other tab's entry is untouched until that tab changes");
  for (const raw of ["{", "null", "[1,2]", "\"azimuth\"", "", "1e999"]) {
    const other = load(t, { storage: { [KEY("txt2img")]: raw } });
    assert.deepEqual(other.values(), DEFAULT, JSON.stringify(raw));
    assert.deepEqual(other.errors, []);
  }
});

test("handles keyboard, touch cancellation and repeated drags without leftover pointer state", (t) => {
  const env = load(t, { prompts: { txt2img: "forest" } });
  const shifted = env.key("ArrowRight", { shiftKey: true });
  assert.equal(env.range("azimuth").value, "15");
  assert.equal(shifted.defaultPrevented, true);
  env.pointer("pointerdown", { pointerId: 7, clientX: 0 });
  assert.deepEqual(env.pointerLog, [["set", 7]]);
  assert.equal(env.doc.activeElement, env.orbit(), "the diagram takes focus for the arrow keys");
  env.pointer("pointermove", { pointerId: 7, clientX: 100 });
  assert.equal(env.range("azimuth").value, "85");
  env.pointer("pointercancel", { pointerId: 7 });
  assert.equal(env.range("azimuth").value, "15", "a canceled gesture restores the starting state");
  assert.equal(env.stored().azimuth, 15, "and is never saved");
  env.pointer("pointermove", { pointerId: 7, clientX: 300 });
  assert.equal(env.range("azimuth").value, "15");
  env.pointer("pointerdown", { pointerId: 7, clientX: 0 });
  env.pointer("pointerup", { pointerId: 7, clientX: 10 });
  env.pointer("lostpointercapture", { pointerId: 7 });
  assert.equal(env.range("azimuth").value, "22");
  assert.deepEqual(env.pointerLog, [["set", 7], ["set", 7], ["release", 7]]);
  assert.equal(env.stored().azimuth, 22, "a finished drag is saved");
  assert.equal(env.area().value, "forest");
  assert.equal(env.execCalls.length, 0);
});

// The app's global history shortcut moved on the same ↑/↓; Forge has no page-wide arrow handler, but the keys must
// still be the diagram's: default prevented (no page scroll), only the elevation changes.
test("orbit ↑/↓ change the elevation only and are default-prevented", (t) => {
  const env = load(t);
  assert.equal(env.key("ArrowUp").defaultPrevented, true);
  assert.equal(env.range("elevation").value, "5");
  env.key("ArrowDown");
  env.key("ArrowDown");
  assert.deepEqual(env.values(), { ...DEFAULT, elevation: -5 });
});

test("targets each tab's own main prompt", (t) => {
  const env = load(t, { prompts: { txt2img: "forest", img2img: "from below" } });
  env.button("위에서", "img2img").click();
  assert.deepEqual(env.conflicts("img2img"), ["from below ↔ from above"]);
  assert.deepEqual(env.conflicts("txt2img"), []);
  env.append("txt2img").click();
  assert.equal(env.area("txt2img").value, "forest, facing viewer, upper body, centered composition");
  assert.equal(env.area("img2img").value, "from below");
  env.append("img2img").click();
  assert.ok(env.area("img2img").value.startsWith("from below, "));
  assert.equal(env.area("txt2img").value, "forest, facing viewer, upper body, centered composition");
  assert.equal(env.gradio.img2img.value, env.area("img2img").value);
});

// ---- 프롬프트에 넣기 ---------------------------------------------------------------------------

test("append inserts only the missing part at the end with one insertText — one native undo step", (t) => {
  const env = load(t, { prompts: { txt2img: "forest" } });
  const area = env.area();
  const pageSaw = [];                           // e.g. sd-webui-tagcomplete, which opens its list on typed input
  env.doc.addEventListener("input", (event) => pageSaw.push(event.inputType));
  area.setSelectionRange(2, 4);                 // the user's caret and selection were elsewhere
  env.button("측면").focus();
  env.append().click();
  assert.equal(env.execCalls.length, 1);
  const call = env.execCalls[0];
  assert.equal(call.command, "insertText");
  assert.equal(call.value, ", facing viewer, upper body, centered composition", "only the appended part");
  assert.equal(call.active, area, "the prompt has focus when the command runs");
  assert.deepEqual([call.start, call.end], [call.length, call.length], "a collapsed selection at the very end");
  assert.equal(area.value, "forest" + call.value);
  assert.equal(env.gradio.txt2img.value, area.value, "Gradio saw the change");
  // The browser's own input event (inputType insertText) looks like typing: tag autocomplete would open its list
  // and Enter/Tab would then replace the last appended word (seen in a real Forge). It is stopped at the window
  // in the capture phase; Gradio hears Forge's updateInput instead, which autocomplete ignores (no inputType).
  assert.deepEqual(env.gradio.txt2img.inputTypes, [undefined], "only Forge's updateInput event reaches the prompt");
  assert.deepEqual(pageSaw, [undefined], "nothing on the page sees an insertText input event");
  assert.deepEqual(env.updateCalls, [area]);
  assert.equal(env.doc.activeElement, area, "focus stays at the end of the prompt, ready for Ctrl+Z");
  assert.deepEqual(env.errors, []);
});

test("the input-event guard is gone after the append: typing reaches the page as usual", (t) => {
  const env = load(t, { prompts: { txt2img: "forest" } });
  env.append().click();
  const area = env.area();
  area.dispatchEvent(new env.window.InputEvent("input", { bubbles: true, inputType: "insertText", data: "x" }));
  assert.deepEqual(env.gradio.txt2img.inputTypes, [undefined, "insertText"]);
});

test("the inserted part is exactly the plan's tail: separators follow the app", (t) => {
  for (const [prompt, inserted] of [
    ["forest", ", facing viewer, upper body, centered composition"],
    ["forest,", " facing viewer, upper body, centered composition"],
    ["forest, ", "facing viewer, upper body, centered composition"],
    ["forest,\n", "facing viewer, upper body, centered composition"],
    ["", "facing viewer, upper body, centered composition"],
    ["  ", "facing viewer, upper body, centered composition"],
    ["(Facing_Viewer:1.2), centered composition", ", upper body"],
  ]) {
    const env = load(t, { prompts: { txt2img: prompt } });
    env.append().click();
    assert.deepEqual(env.execCalls.map((call) => call.value), [inserted], JSON.stringify(prompt));
    assert.equal(env.area().value, prompt + inserted);
  }
});

for (const exec of ["false", "noop", "throw", "missing"]) {
  test(`falls back to the value and Forge's updateInput when insertText ${exec === "missing" ? "is missing" : `returns ${exec}`}`, (t) => {
    const env = load(t, { prompts: { txt2img: "forest" }, exec });
    env.append().click();
    assert.equal(env.execCalls.length, exec === "missing" ? 0 : 1);
    assert.equal(env.area().value, "forest, facing viewer, upper body, centered composition");
    assert.deepEqual(env.updateCalls, [env.area()]);
    assert.equal(env.gradio.txt2img.value, env.area().value, "Gradio saw the change");
    assert.deepEqual(env.gradio.txt2img.inputTypes, [undefined], "Forge's updateInput event (no inputType)");
    assert.equal(env.append().disabled, true);
    assert.equal(env.feedback(), "메인 태그에 3개를 추가했습니다. 기존 프롬프트는 유지했습니다.");
    assert.deepEqual(env.errors, []);
  });
}

test("without Forge's updateInput the fallback still sends Gradio an input event", (t) => {
  const env = load(t, { prompts: { txt2img: "forest" }, exec: "false", updateInput: false });
  env.append().click();
  assert.equal(env.area().value, "forest, facing viewer, upper body, centered composition");
  assert.equal(env.gradio.txt2img.value, env.area().value);
});

// ---- 미리보기 · 충돌을 프롬프트에 맞춘다 ---------------------------------------------------------

test("preview, conflicts and the button label follow typing in the prompt", (t) => {
  const env = load(t);
  env.button("위에서 · 근접").click();
  assert.equal(env.append().textContent, "메인 태그에 추가");
  env.type("from below");
  assert.deepEqual(env.conflicts(), ["from below ↔ from above"]);
  assert.equal(env.append().textContent, "기존 구도 유지하고 추가");
  env.type("from below, wide_shot");
  assert.deepEqual(env.conflicts(), ["from below ↔ from above", "wide shot ↔ close-up"]);
  env.type("three-quarter view from the subject's right, (from_above:1.1), close-up, centered composition");
  assert.deepEqual(env.conflicts(), []);
  assert.equal(env.warning().hidden, true);
  assert.equal(env.append().disabled, true);
  assert.equal(env.append().textContent, "이미 포함된 구도");
  env.type("");
  assert.equal(env.append().disabled, false);
  assert.equal(env.append().textContent, "메인 태그에 추가");
});

test("values Gradio writes without an event are picked up on open, focus or pointer entry, and always on click", async (t) => {
  const env = load(t);
  env.button("위에서 · 근접").click();
  env.area().value = "from below";              // e.g. ↙️ paste, style apply, TIPO: no input event
  assert.deepEqual(env.conflicts(), []);
  env.panel().open = true;                       // the toggle event is queued
  await tick();
  assert.deepEqual(env.conflicts(), ["from below ↔ from above"]);
  env.area().value = "wide shot";
  env.panel().dispatchEvent(new env.window.MouseEvent("pointerenter"));
  assert.deepEqual(env.conflicts(), ["wide shot ↔ close-up"]);
  env.area().value = "portrait";
  env.button("초기화").dispatchEvent(new env.window.FocusEvent("focusin", { bubbles: true }));
  assert.deepEqual(env.conflicts(), ["portrait ↔ close-up"]);
  env.area().value = "three-quarter view from the subject's right, from above, close-up, centered composition";
  assert.equal(env.append().disabled, false, "still the old plan on screen");
  env.append().click();
  assert.equal(env.execCalls.length, 0, "the click plans from the prompt as it is now");
  assert.equal(env.append().disabled, true);
});

test("the feedback line clears on the next control change, not on typing", (t) => {
  const env = load(t, { prompts: { txt2img: "forest" } });
  env.append().click();
  assert.match(env.feedback(), /3개를 추가했습니다/);
  env.type(env.area().value + ", night");
  assert.match(env.feedback(), /3개를 추가했습니다/);
  env.slide("roll", 10);
  assert.equal(env.feedback(), "");
});

// ---- 조작 ------------------------------------------------------------------------------------

test("arrow keys move 5°, Shift 15°, Home returns to the front, and other keys pass through", (t) => {
  const env = load(t, { storage: { [KEY("txt2img")]: JSON.stringify({ ...DEFAULT, distance: 80, roll: 9, framing: -40 }) } });
  env.key("ArrowLeft");
  env.key("ArrowUp", { shiftKey: true });
  assert.deepEqual(env.values(), { azimuth: -5, elevation: 15, distance: 80, roll: 9, framing: -40 });
  assert.equal(env.stored().elevation, 15, "each key press is saved");
  for (const init of [{ ctrlKey: true }, { altKey: true }, { metaKey: true }]) {
    assert.equal(env.key("ArrowLeft", init).defaultPrevented, false, JSON.stringify(init));
  }
  assert.equal(env.key("a").defaultPrevented, false);
  assert.equal(env.key("Enter").defaultPrevented, false);
  assert.equal(env.key("toString").defaultPrevented, false, "only the five keys, not Object.prototype names");
  assert.deepEqual(env.values(), { azimuth: -5, elevation: 15, distance: 80, roll: 9, framing: -40 });
  assert.equal(env.key("Home").defaultPrevented, true);
  assert.deepEqual(env.values(), { azimuth: 0, elevation: 0, distance: 80, roll: 9, framing: -40 });
  for (let i = 0; i < 20; i += 1) env.key("ArrowRight", { shiftKey: true });
  assert.equal(env.range("azimuth").value, "180", "clamped like the app");
});

test("only the primary button of the primary pointer starts a drag, and only its own pointer moves it", (t) => {
  const env = load(t);
  env.pointer("pointerdown", { pointerId: 3, button: 2 });
  env.pointer("pointerdown", { pointerId: 4, isPrimary: false });
  env.pointer("pointermove", { pointerId: 3, clientX: 100 });
  assert.equal(env.range("azimuth").value, "0");
  env.pointer("pointerdown", { pointerId: 5, clientX: 10, clientY: 10 });
  env.pointer("pointermove", { pointerId: 6, clientX: 200 });
  assert.equal(env.range("azimuth").value, "0", "another pointer");
  env.pointer("pointermove", { pointerId: 5, clientX: 30, clientY: -40 });
  assert.deepEqual([env.values().azimuth, env.values().elevation], [14, 30]);
  env.pointer("pointerup", { pointerId: 5, clientX: 30, clientY: -40 });
  assert.deepEqual(env.pointerLog, [["set", 5], ["release", 5]]);
});

test("the reset button returns to the defaults, presses the first preset and saves", (t) => {
  const env = load(t, { storage: { [KEY("txt2img")]: JSON.stringify({ azimuth: 90, elevation: 0, distance: 55, roll: 0, framing: -55 }) } });
  const pressed = () => [...env.panel().querySelectorAll("[data-preset]")].map((b) => b.getAttribute("aria-pressed"));
  assert.deepEqual(pressed(), ["false", "false", "false", "true", "false"]);
  env.slide("roll", 7);
  assert.deepEqual(pressed(), ["false", "false", "false", "false", "false"]);
  env.button("초기화").click();
  assert.deepEqual(env.values(), DEFAULT);
  assert.deepEqual(pressed(), ["true", "false", "false", "false", "false"]);
  assert.deepEqual(env.stored(), DEFAULT);
});

test("the diagram, the readout and the slider values follow the state", (t) => {
  const env = load(t);
  const { compositionCameraPoint } = env.window.sam3CompositionPrompt;
  const ray = env.panel().querySelector(".sam3-composition__camera-ray");
  const camera = env.panel().querySelector(".sam3-composition__camera");
  const readout = env.panel().querySelector(".sam3-composition__readout");
  for (const state of [DEFAULT, { azimuth: 180, elevation: 15, distance: 100, roll: -12, framing: 40 },
    { azimuth: -35, elevation: -30, distance: 75, roll: 30, framing: 0 }]) {
    for (const [key, value] of Object.entries(state)) env.slide(key, value);
    const point = compositionCameraPoint(state);
    assert.equal(ray.getAttribute("x1"), String(point.x));
    assert.equal(ray.getAttribute("y1"), String(point.y));
    assert.equal(ray.getAttribute("x2"), "150");
    assert.equal(ray.getAttribute("y2"), "80");
    assert.equal(ray.getAttribute("stroke-dasharray"), point.behind ? "4 4" : null);
    assert.equal(camera.getAttribute("transform"), `translate(${point.x} ${point.y}) rotate(${state.roll})`);
    assert.equal(readout.textContent, `방향 ${state.azimuth}° · 높이 ${state.elevation}°`);
    assert.deepEqual(["azimuth", "elevation", "distance", "roll", "framing"].map((key) => env.output(key).textContent),
      [`${state.azimuth}°`, `${state.elevation}°`, `${state.distance}%`, `${state.roll}°`, `${state.framing}%`]);
  }
});

// ---- 마운트 · DOM 규칙 · 글자 ---------------------------------------------------------------

test("fills each placeholder once, collapsed, with the app's text, labels and accessible names", (t) => {
  const env = load(t);
  for (const tab of ["txt2img", "img2img"]) {
    const panel = env.panel(tab);
    assert.equal(panel.tagName, "DETAILS");
    assert.equal(panel.open, false, "collapsed by default, like the app");
    assert.equal(env.doc.querySelectorAll(`#sam3_composition_${tab} .sam3-composition`).length, 1);
    assert.equal(panel.querySelector("summary").textContent, "구도 · 카메라 프롬프트로 시점 잡기");
    const presets = [...panel.querySelectorAll("[data-preset]")];
    assert.deepEqual(presets.map((b) => b.textContent), ["정면 상반신", "낮은 시점 · 전신", "위에서 · 근접", "측면 · 여백", "후면 · 원경"]);
    assert.deepEqual(presets.map((b) => b.getAttribute("aria-pressed")), ["true", "false", "false", "false", "false"]);
    assert.equal(panel.querySelector(".sam3-composition__presets").getAttribute("aria-label"), "구도 프리셋");
    const orbit = env.orbit(tab);
    assert.equal(orbit.getAttribute("role"), "group");
    assert.equal(orbit.getAttribute("aria-keyshortcuts"), "ArrowLeft ArrowRight ArrowUp ArrowDown Home");
    assert.equal(env.doc.getElementById(orbit.getAttribute("aria-describedby")).textContent,
      "드래그 또는 방향키로 시점 조작 · Shift + 방향키: 크게 이동 · Home: 정면 복원. 아래 슬라이더로도 모두 조절할 수 있습니다.");
    assert.equal(orbit.querySelector("svg").getAttribute("aria-hidden"), "true");
    assert.deepEqual([...orbit.querySelectorAll("text")].map((node) => node.textContent), ["정면", "후면"]);
    const labels = [...panel.querySelectorAll("label")].map((label) => {
      const input = env.doc.getElementById(label.getAttribute("for"));
      return [label.firstElementChild.firstChild.data.trim(), input.min, input.max, input.step];
    });
    assert.deepEqual(labels, [["방향", "-180", "180", "1"], ["높이", "-75", "75", "1"], ["거리 · 크롭", "0", "100", "1"],
      ["기울기", "-30", "30", "1"], ["화면 내 인물 위치", "-100", "100", "1"]]);
    const help = [...panel.querySelectorAll(".sam3-composition__help")].map((p) => p.textContent);
    assert.deepEqual(help, [
      "태그와 구도 문구로 생성을 유도합니다. 실제 3D 카메라 제어가 아니며 모델에 따라 결과가 달라집니다. 조작만으로 프롬프트가 바뀌지 않습니다.",
      "드래그 또는 방향키로 시점 조작 · Shift + 방향키: 크게 이동 · Home: 정면 복원. 아래 슬라이더로도 모두 조절할 수 있습니다.",
      "거리 0: 근접 / 100: 원경 · 화면 위치 −: 왼쪽 / +: 오른쪽. 작은 수치 차이는 같은 문구로 표현될 수 있습니다.",
    ]);
    assert.equal(panel.querySelector(".sam3-composition__preview strong").textContent, "추가할 태그 · 구도 문구");
    assert.equal(env.preview(tab), "facing viewer, upper body, centered composition");
    assert.equal(env.warning(tab).getAttribute("role"), "status");
    assert.equal(env.warning(tab).firstChild.data, "기존 구도와 충돌할 수 있습니다. 기존 태그는 삭제하지 않습니다.");
    assert.equal(env.append(tab).getAttribute("aria-describedby"), panel.querySelector(".sam3-composition__preview").id);
    const feedback = panel.querySelector(".sam3-composition__feedback");
    assert.deepEqual([feedback.getAttribute("role"), feedback.getAttribute("aria-live")], ["status", "polite"]);
    assert.ok(env.button("초기화", tab));
    for (const button of panel.querySelectorAll("button")) assert.equal(button.type, "button", "never submits a form");
  }
  const ids = [...env.doc.querySelectorAll("[id]")].map((node) => node.id);
  assert.equal(new Set(ids).size, ids.length, "ids stay unique across the two tabs");
  assert.deepEqual(env.errors, []);
});

test("after mount, nothing adds or removes DOM nodes — only attributes, values and text data change", async (t) => {
  const env = load(t, { prompts: { txt2img: "forest" } });
  const records = [];
  const observer = new env.window.MutationObserver((list) => records.push(...list.filter((r) => r.type === "childList")));
  observer.observe(env.doc.body, { childList: true, subtree: true });
  env.panel().open = true;
  env.button("위에서 · 근접").click();
  env.type("from below, wide shot");                 // conflicts appear
  env.type("from below");                            // and shrink
  env.slide("roll", -20, "txt2img", { release: true });
  env.key("ArrowLeft", { shiftKey: true });
  env.pointer("pointerdown", { pointerId: 2 });
  env.pointer("pointermove", { pointerId: 2, clientX: 250, clientY: 80 });
  env.pointer("pointerup", { pointerId: 2, clientX: 250, clientY: 80 });
  env.append().click();
  env.type("");
  env.button("초기화").click();
  env.panel().open = false;
  await tick();
  observer.disconnect();
  assert.deepEqual(records.map((r) => r.target.className || r.target.nodeName), []);
});

test("waits for placeholders that render later, and stays idle where the setting removed them", (t) => {
  const env = load(t, { mounts: [] });
  assert.deepEqual(Object.keys(env.hooks.panels), []);
  assert.equal(env.afterUpdates.length, 1, "one retry hook");
  env.afterUpdates[0]();
  assert.deepEqual(Object.keys(env.hooks.panels), []);
  env.doc.querySelector("#tab_txt2img").insertAdjacentHTML("beforeend", mountBlock("txt2img"));
  env.afterUpdates[0]();
  assert.deepEqual(Object.keys(env.hooks.panels), ["txt2img"]);
  env.doc.querySelector("#tab_img2img").insertAdjacentHTML("beforeend", mountBlock("img2img"));
  env.afterUpdates[0]();
  assert.deepEqual(Object.keys(env.hooks.panels).sort(), ["img2img", "txt2img"]);
  let lookups = 0;
  const original = env.doc.querySelector.bind(env.doc);
  env.doc.querySelector = (...args) => { lookups += 1; return original(...args); };
  env.afterUpdates[0]();
  assert.equal(lookups, 0, "both mounted: the retry hook does nothing");
  assert.equal(env.doc.querySelectorAll(".sam3-composition").length, 2);
  assert.deepEqual(env.errors, []);
});

test("without the logic script nothing mounts and nothing throws", (t) => {
  const env = load(t, { logic: false });
  assert.equal(env.doc.querySelector(".sam3-composition"), null);
  assert.deepEqual(env.errors, []);
});

test("blocked storage: the panel starts from the defaults and keeps working", (t) => {
  const env = load(t, { storageBlocked: true, prompts: { txt2img: "forest" } });
  assert.deepEqual(env.values(), DEFAULT);
  env.button("후면").click();
  env.key("ArrowLeft");
  env.slide("distance", 10, "txt2img", { release: true });
  assert.deepEqual(env.values(), { azimuth: 175, elevation: 15, distance: 10, roll: 0, framing: 40 });
  env.append().click();
  assert.match(env.area().value, /^forest, from behind, /);
  assert.deepEqual(env.errors, []);
});
