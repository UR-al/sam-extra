// 내장 Tag Autocomplete — sam3ext/tagcomplete/javascript/ 의 상류 파일 15개(sd-webui-tagcomplete-neo 8e0d81f, 바꾸지 않음)를
// 로더가 붙이는 순서대로 한 페이지(jsdom)에 올린다.
//
// * 다른 확장이 쓰는 전역(prompt-all-in-one: addAutocompleteToArea · TAC_CFG · hideResults)과 파서 8개, onUiUpdate 초기화가
//   그대로 생긴다.
// * 같은 파일을 한 번 더 실으면 const 선언에서 실패한다 — 그래서 로더는 다른 사본이 있으면 아무것도 붙이지 않는다.
// * 구도 · 카메라 칸(javascript/composition_ui.js)이 프롬프트 끝에 붙인 글자는 실제 tagcomplete 의 입력 리스너가 무시하고
//   (추천 목록이 뜨지 않음), 사람이 친 글자에는 반응한다 — 따로 설치하던 때와 같은 상호작용이 내장본에서도 지켜진다.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM, VirtualConsole } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const JS_DIR = path.join(ROOT, "sam3ext", "tagcomplete", "javascript");
// Forge's Extension.list_files order (Python sorted(os.listdir)) — the order boot.manifest() gives the loader
const FILES = readdirSync(JS_DIR).filter((name) => name.endsWith(".js")).sort();
const SOURCES = FILES.map((name) => [name, readFileSync(path.join(JS_DIR, name), "utf8")]);
const LOGIC = readFileSync(path.join(ROOT, "javascript", "composition_prompt.js"), "utf8");
const UI = readFileSync(path.join(ROOT, "javascript", "composition_ui.js"), "utf8");
// Forge javascript/ui.js:415 (verbatim)
const FORGE_UPDATE_INPUT = `
function updateInput(target) {
    const e = new Event("input", { bubbles: true });
    Object.defineProperty(e, "target", { value: target });
    target.dispatchEvent(e);
}`;

const promptBlock = (tab) =>
  `<div id="${tab}_prompt" class="block gradio-textbox"><label><textarea data-testid="textbox"></textarea></label></div>`;
const mountBlock = (tab) =>
  `<div id="sam3_composition_${tab}" class="block gradio-html sam3-composition-host"><div>`
  + `<div class="prose gradio-html sam3-composition-host">`
  + `<div class="sam3-composition-mount" data-sam3-composition-tab="${tab}"></div>`
  + `</div></div></div>`;

function page(t) {
  const errors = [];
  const virtualConsole = new VirtualConsole();
  virtualConsole.on("jsdomError", (error) => errors.push(String(error && (error.stack || error.message || error))));
  virtualConsole.on("error", (...args) => errors.push(args.join(" ")));
  const body = ["txt2img", "img2img"].map((tab) => `<div id="tab_${tab}" class="tabitem">${promptBlock(tab)}${mountBlock(tab)}</div>`).join("");
  const dom = new JSDOM(`<!doctype html><html><head></head><body>${body}</body></html>`, {
    // "dangerously": the files run as classic <script> elements, which share one global scope for their top-level
    // const / let / class (window.eval would give each file its own)
    url: "http://127.0.0.1:7860/", runScripts: "dangerously", pretendToBeVisual: true, virtualConsole,
  });
  t.after(() => dom.window.close());
  const { window } = dom;
  const uiUpdates = [];
  const uiLoaded = [];
  window.gradioApp = () => window.document;
  window.onUiUpdate = (fn) => uiUpdates.push(fn);
  window.onUiLoaded = (fn) => uiLoaded.push(fn);
  window.onAfterUiUpdate = () => {};
  window.opts = {};
  return { window, doc: window.document, errors, uiUpdates, uiLoaded };
}

// like the loader's <script async=false> elements, in order (an inline script runs when it is inserted)
function loadTagcomplete(env, sources = SOURCES) {
  for (const [name, source] of sources) {
    const script = env.doc.createElement("script");
    script.setAttribute("data-file", name);
    script.textContent = source;
    env.doc.head.appendChild(script);
  }
}

test("the boot order is Forge's order of the upstream folder", () => {
  assert.deepEqual(FILES, [
    "__globals.js", "_baseParser.js", "_caretPosition.js", "_result.js", "_textAreas.js", "_utils.js", "ext_chants.js",
    "ext_embeddings.js", "ext_loras.js", "ext_lycos.js", "ext_modelKeyword.js", "ext_styles.js", "ext_umi.js",
    "ext_wildcards.js", "tagAutocomplete.js",
  ]);
});

test("the 15 files load on one page and define what other extensions use", (t) => {
  const env = page(t);
  loadTagcomplete(env);
  assert.deepEqual(env.errors, []);
  const { window } = env;
  // prompt-all-in-one-neo (phystonPrompt.vue, common.js) calls these
  assert.equal(typeof window.addAutocompleteToArea, "function");
  assert.equal(typeof window.hideResults, "function");
  assert.ok("TAC_CFG" in window);
  assert.equal(window.TAC_CFG, null, "set by the onUiUpdate init");
  assert.equal(window.eval("PARSERS.length"), 8);
  assert.equal(window.eval("QUEUE_FILE_LOAD.length"), 8);
  assert.equal(env.uiUpdates.length, 1, "upstream starts from one onUiUpdate callback");
});

test("a second copy on the same page fails half-way — why the loader never adds one", (t) => {
  const env = page(t);
  loadTagcomplete(env);
  assert.deepEqual(env.errors, []);
  loadTagcomplete(env, SOURCES.slice(0, 1));
  assert.equal(env.errors.length, 1);
  assert.match(env.errors[0], /already been declared/);
});

test("composition append is ignored by tagcomplete's input listener, typing is not", async (t) => {
  const env = page(t);
  const { window, doc } = env;
  // Chrome's execCommand("insertText"): inserts at the selection and fires an input event with inputType insertText
  doc.execCommand = (command, _showUi, value) => {
    const active = doc.activeElement;
    if (command !== "insertText" || !active || active.tagName !== "TEXTAREA") return false;
    const { selectionStart: start, selectionEnd: end } = active;
    active.value = active.value.slice(0, start) + value + active.value.slice(end);
    active.selectionStart = active.selectionEnd = start + value.length;
    active.dispatchEvent(new window.InputEvent("input", { bubbles: true, inputType: "insertText", data: value }));
    return true;
  };
  window.eval(FORGE_UPDATE_INPUT);
  window.eval(LOGIC);
  window.eval(UI);
  env.uiLoaded.forEach((fn) => fn());
  loadTagcomplete(env);
  assert.deepEqual(env.errors, []);

  // what tagcomplete's setup() does for the prompt (TAC_CFG as syncOptions would leave it, as far as this needs)
  window.eval("TAC_CFG = { activeIn: { txt2img: true, img2img: true, negativePrompts: true, thirdParty: true }, delayTime: 1, maxResults: 5 };");
  const area = doc.querySelector("#txt2img_prompt textarea");
  area.value = "forest";
  window.addAutocompleteToArea(area);
  assert.ok(area.classList.contains("autocomplete"), "the real listener is attached");
  const calls = [];
  window.autocomplete = async (textArea, prompt) => { calls.push([textArea, prompt]); };
  window.updateRuby = () => {};
  window.checkKeywordInsertionUndo = () => {};

  const append = doc.querySelector("#sam3_composition_txt2img .sam3-composition__append");
  assert.ok(append, "the composition panel is mounted");
  [...doc.querySelectorAll("#sam3_composition_txt2img button")].find((b) => b.textContent.includes("측면")).click();
  append.click();
  assert.notEqual(area.value, "forest", "the panel appended its words");
  await new Promise((resolve) => setTimeout(resolve, 60));
  assert.deepEqual(calls, [], "no autocomplete list for the appended words");

  area.dispatchEvent(new window.InputEvent("input", { bubbles: true, inputType: "insertText", data: "x" }));
  await new Promise((resolve) => setTimeout(resolve, 60));
  assert.equal(calls.length, 1, "typing still opens the list");
  assert.equal(calls[0][0], area);
  assert.deepEqual(env.errors, []);
});
