// 내장 Tag Autocomplete 와 notebook 탭 줄의 '기본값 저장' · '모델 내리기' 버튼 — 한 페이지에서 서로 건드리지 않는지(v0.34.0 통합).
//
// Forge 처럼 이 확장의 javascript/*.js 전부를 파일 이름 순서로 classic <script> 로 싣는다(한 전역 범위 — 최상위 const · let ·
// class 이름이 겹치면 뒤 파일이 통째로 실패). 그다음 로더(javascript/tagcomplete_loader.js)가 설정을 받고 붙이는 상류 15개를
// 같은 페이지에서 돌리고(jsdom 은 <script src> 를 받지 않으므로 테스트가 원문을 돌리고 load 를 낸다 — tagcomplete_loader.test.mjs
// 와 같은 방법), 상류의 진짜 onUiUpdate 초기화(태그 목록 읽기 · 프롬프트에 붙이기 · 상태 점 · extra networks 새로 고침 버튼의
// 클릭 리스너)를 끝까지 돌린다. Forge 의 script.js 처럼 gradioApp() 의 MutationObserver 가 onUiUpdate 를 부른다. fetch 는 가짜다
// (네트워크 없음). 숨은 #sam3_unload_models_run 의 응답은 이 테스트가 흉내 낸다(notebook_unload_models.test.mjs 와 같음).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM, VirtualConsole } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const PAGE_DIR = path.join(ROOT, "javascript");
const TAC_DIR = path.join(ROOT, "sam3ext", "tagcomplete", "javascript");
// Forge's Extension.list_files order (Python sorted(os.listdir)) for the extension's own scripts and the upstream folder
const PAGE_FILES = readdirSync(PAGE_DIR).filter((name) => name.endsWith(".js")).sort();
const TAC_FILES = readdirSync(TAC_DIR).filter((name) => name.endsWith(".js")).sort();
const read = (dir, name) => readFileSync(path.join(dir, name), "utf8");
const BRIDGE_PY = readFileSync(path.join(ROOT, "sam3ext", "ui_unload_models.py"), "utf8");
const ARGS_JS = JSON.parse(/^ARGS_JS = (".*")$/m.exec(BRIDGE_PY)[1]);
const TAGS = "extensions/forge_sam3_extension/tags";
// the keymap / colormap defaults of the helper's on_ui_settings (Settings shows these when nothing was saved)
const HELPER_PY = readFileSync(path.join(ROOT, "sam3ext", "tagcomplete", "tag_autocomplete_helper.py"), "utf8");
function pyDefault(name) {   // keymapDefault = """\ {...}\ """ — the JSON between the braces
  const start = HELPER_PY.indexOf(`${name} = """`);
  assert.ok(start >= 0, name);
  const open = HELPER_PY.indexOf("{", start);
  const body = HELPER_PY.slice(open, HELPER_PY.indexOf('"""', open)).replace(/\\\s*$/, "");
  return JSON.stringify(JSON.parse(body));
}
const URLS = TAC_FILES.map((name, i) => `file=extensions/forge_sam3_extension/sam3ext/tagcomplete/javascript/${name}?${1000 + i}`);
const BOOT = { enabled: true, reason: null, upstream: "eduardoabreu81/sd-webui-tagcomplete-neo@8e0d81f", scripts: URLS };
const SETTINGS = {
  sam3_tac_builtin_boot: JSON.stringify(BOOT), sam3_layout_sections: true,
  tac_active: true, "tac_activeIn.txt2img": true, "tac_activeIn.img2img": true, "tac_activeIn.negativePrompts": true,
  tac_maxResults: 5, tac_resultStepLength: 100, tac_useIndexedSearch: true, tac_tagFile: "danbooru.csv", tac_chantFile: "None",
  "tac_translation.translationFile": "None", "tac_translation.liveTranslation": false, "tac_extra.extraFile": "None",
  tac_keymap: pyDefault("keymapDefault"), tac_colormap: pyDefault("colorDefault"), tac_delayTime: 1,
};

// Forge script.js / ui.js stand-in: `let opts`, onOptionsAvailable / onOptionsChanged, and the onUiUpdate / onUiLoaded
// callbacks — one MutationObserver on gradioApp() (childList + subtree) runs the onUiUpdate ones, as script.js does.
const FORGE = `
let opts = {};
const optionsAvailableCallbacks = [];
const optionsChangedCallbacks = [];
const uiUpdateCallbacks = [];
const uiLoadedCallbacks = [];
function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("error running callback", callback, ":", e);
        }
    }
}
function onOptionsChanged(callback) { optionsChangedCallbacks.push(callback); }
function onOptionsAvailable(callback) {
    if (Object.keys(opts).length > 0) { callback(); return; }
    optionsAvailableCallbacks.push(callback);
}
function onUiUpdate(callback) { uiUpdateCallbacks.push(callback); }
function onAfterUiUpdate(callback) {}
function onUiLoaded(callback) { uiLoadedCallbacks.push(callback); }
function onUiTabChange(callback) {}
function forgeLoadSettings(json) {
    opts = JSON.parse(json);
    executeCallbacks(optionsAvailableCallbacks);
    executeCallbacks(optionsChangedCallbacks);
}
new MutationObserver((m) => executeCallbacks(uiUpdateCallbacks, m)).observe(gradioApp(), { childList: true, subtree: true });
`;

const TAB_ROW = `
  <div class="tab-nav scroll-hide" role="tablist">
    <button role="tab" class="selected">Generation</button><button role="tab">임베딩</button>
    <button role="tab">체크포인트</button><button role="tab">로라</button>
    <div class="extra-networks-controls-div">
      <button id="txt2img_extra_refresh" class="extra-network-control--refresh" title="Refresh">⟳</button>
    </div>
  </div>`;

const PAGE = `
  <gradio-app><div class="gradio-container">
    <main id="sam3_notebook_layout"><section class="sam3-notebook-prompt">
      <div id="txt2img_prompt" class="block gradio-textbox"><label><textarea data-testid="textbox"></textarea></label></div>
      <div id="txt2img_neg_prompt" class="block gradio-textbox"><label><textarea data-testid="textbox"></textarea></label></div>
      <div id="txt2img_tools" class="form"></div>
      <div id="txt2img_extra_tabs" class="tabs gradio-tabs extra-networks sam3-notebook-extra-tabs">
        ${TAB_ROW}
        <div class="tabitem"></div>
      </div>
    </section></main>
    <footer id="footer"></footer>
    <div id="sam3_unload_models_bridge" class="hidden">
      <div id="sam3_unload_models_request"><textarea></textarea></div>
      <button id="sam3_unload_models_run">sam3_unload_models</button>
      <div id="sam3_unload_models_result"><label><textarea></textarea></label></div>
    </div>
  </div></gradio-app>`;

const CSV = "1girl,0,4114588,\nsolo,0,3426446,\nsmile,0,1700000,\nsmall_breasts,0,500000,\n";

function page(t) {
  const errors = [];
  const virtualConsole = new VirtualConsole();
  virtualConsole.on("jsdomError", (error) => errors.push(String(error && (error.stack || error.message || error))));
  virtualConsole.on("error", (...args) => errors.push(args.map(String).join(" ")));
  const dom = new JSDOM(`<!doctype html><html><head></head><body>${PAGE}</body></html>`, {
    // "dangerously": the files run as classic <script> elements sharing one global scope for top-level const / let / class
    url: "http://127.0.0.1:7860/", runScripts: "dangerously", pretendToBeVisual: true, virtualConsole,
  });
  t.after(() => dom.window.close());
  const { window } = dom;
  const doc = window.document;
  window.gradioApp = () => doc.querySelector("gradio-app");
  window.__sam3SaveDefaultsTestHooks = {};
  window.__sam3UnloadModelsTestHooks = {};
  const fetches = [];
  window.fetch = async (url) => {
    const name = String(url);
    fetches.push(name);
    const body = name.startsWith("file=tmp/tagAutocompletePath.txt") ? TAGS
      : name.startsWith(`file=${TAGS}/danbooru.csv`) ? CSV : "";
    return { status: 200, ok: true, text: async () => body, json: async () => [], blob: async () => null };
  };
  const run = (source, file) => {   // a classic <script> in <head>, like Forge's head scripts (runs when inserted)
    const script = doc.createElement("script");
    if (file) script.setAttribute("data-file", file);
    script.textContent = source;
    doc.head.appendChild(script);
    return script;
  };
  run(FORGE, "forge");
  for (const name of PAGE_FILES) run(read(PAGE_DIR, name), name);
  return { window, doc, errors, fetches, run };
}

async function until(check, label, ms = 4000) {
  const start = Date.now();
  while (!check()) {
    if (Date.now() - start > ms) assert.fail(`timed out waiting for ${label}`);
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

const tick = (ms = 20) => new Promise((resolve) => setTimeout(resolve, ms));

// what Forge does after the page is built: settings arrive (the loader adds the upstream <script src> elements, which
// this test runs and then fires load on), onUiLoaded runs (the notebook buttons mount 300 ms later)
async function start(env) {
  const { window, doc } = env;
  window.forgeLoadSettings(JSON.stringify(SETTINGS));
  const added = () => [...doc.head.querySelectorAll("script[src]")];
  await until(() => added().length === TAC_FILES.length, "the loader's 15 scripts");
  TAC_FILES.forEach((name) => env.run(read(TAC_DIR, name), name));
  added().forEach((script) => script.dispatchEvent(new window.Event("load")));
  await until(() => doc.documentElement.getAttribute("data-sam3-tagcomplete") === "started", "the loader");
  window.eval("executeCallbacks(uiLoadedCallbacks)");
  await until(() => window.eval("typeof TAC_CFG === 'object' && TAC_CFG !== null && tagsLoaded === true"), "tagcomplete's setup");
  await until(() => doc.querySelector("#sam3_unload_models") && doc.querySelector("#sam3_save_defaults"), "the two buttons");
}

function fakeUnloadBridge(env) {
  const run = env.doc.querySelector("#sam3_unload_models_run");
  const result = env.doc.querySelector("#sam3_unload_models_result textarea");
  const argsJs = env.window.eval(`(${ARGS_JS})`);
  const sent = [];
  run.addEventListener("click", () => {
    const args = argsJs("", result.value);
    sent.push(args);
    setTimeout(() => {
      result.value = JSON.stringify({
        ok: true, request: args[0], refused: false, text: "VRAM 사용 19.6 GB → 3.1 GB (Forge 모델 · SAM3)", detail: "",
        released: ["forge", "sam3"], idle: [], errors: [], before: null, after: null,
      });
    }, 10);
  });
  return sent;
}

const row = (doc) => doc.querySelector("#txt2img_extra_tabs > .tab-nav");
const refreshes = (env) => env.fetches.filter((url) => url.includes("tacapi/v1/refresh-temp-files")).length;

test("every page script and the 15 upstream files load together; both start: tagcomplete on the prompts, the buttons in the tab row", async (t) => {
  const env = page(t);
  const { window, doc } = env;
  await start(env);
  assert.deepEqual(env.errors, [], "no script failed (no global name declared twice)");
  // tagcomplete: on both prompts, its status dot in the toolbar below Generate — nothing of it in the tab row
  assert.equal(doc.querySelectorAll(".autocompleteParent").length, 2);
  assert.ok(doc.querySelector("#txt2img_tools .tac-status-dot"));
  assert.equal(row(doc).querySelector(".autocompleteParent, .autocompleteResults, .tac-status-dot"), null);
  // the buttons: '모델 내리기' right after '기본값 저장' inside its group, the result line after the group
  const save = doc.querySelector("#sam3_save_defaults");
  const unload = doc.querySelector("#sam3_unload_models");
  assert.equal(save.parentElement, row(doc));
  assert.equal(unload.parentElement, save);
  assert.equal(unload.previousElementSibling, save.querySelector(":scope > button"));
  assert.equal(doc.querySelector("#sam3_unload_models_status").previousElementSibling, save);
  assert.equal(window.eval("allTags.length"), 4, "upstream read the tag list");
});

test("the unload button and tagcomplete's refresh listener on the same row do not take each other's clicks", async (t) => {
  const env = page(t);
  const { window, doc } = env;
  await start(env);
  const sent = fakeUnloadBridge(env);
  const before = refreshes(env);
  doc.querySelector("#sam3_unload_models button").click();
  await until(() => /VRAM 사용 19.6 GB/.test(doc.querySelector("#sam3_unload_models_status").textContent), "the result line");
  assert.equal(sent.length, 1);
  assert.equal(refreshes(env), before, "pressing 모델 내리기 does not refresh tagcomplete's lists");
  doc.querySelector("#txt2img_extra_refresh").click();
  await until(() => refreshes(env) > before, "tagcomplete's refresh");
  await tick(30);
  assert.equal(sent.length, 1, "the extra networks refresh does not press 모델 내리기");
  assert.deepEqual(env.errors, []);
});

test("Gradio redraws the tab row: onUiUpdate runs again, tagcomplete stays set up once, the buttons come back", async (t) => {
  const env = page(t);
  const { window, doc } = env;
  await start(env);
  const cfg = window.eval("TAC_CFG");
  const unload = doc.querySelector("#sam3_unload_models");
  const tabs = doc.querySelector("#txt2img_extra_tabs");
  const fresh = doc.createElement("div");
  fresh.innerHTML = TAB_ROW;
  tabs.replaceChild(fresh.firstElementChild, row(doc));   // a new row without our nodes (MutationObserver → onUiUpdate)
  await tick(30);
  window.__sam3SaveDefaultsTestHooks.ensureButton();
  window.__sam3UnloadModelsTestHooks.ensureButton();
  assert.equal(row(doc).querySelector("#sam3_unload_models"), unload, "the same button, on the new row");
  assert.equal(unload.parentElement, row(doc).querySelector("#sam3_save_defaults"));
  assert.equal(doc.querySelectorAll("#sam3_unload_models, #sam3_unload_models_status").length, 2);
  assert.equal(window.eval("TAC_CFG"), cfg, "tagcomplete did not start again");
  assert.equal(doc.querySelectorAll(".autocompleteParent").length, 2);
  assert.equal(doc.querySelectorAll(".tac-status-dot").length, 1);
  assert.deepEqual(env.errors, []);
});

test("while 모델 내리기 waits for its result, typing in the prompt still opens tagcomplete's list", async (t) => {
  const env = page(t);
  const { window, doc } = env;
  await start(env);
  fakeUnloadBridge(env);
  const button = doc.querySelector("#sam3_unload_models button");
  button.focus();
  button.click();
  assert.equal(button.getAttribute("aria-disabled"), "true", "busy");
  const area = doc.querySelector("#txt2img_prompt textarea");
  area.focus();
  area.value = "sm";
  area.selectionStart = area.selectionEnd = 2;
  area.dispatchEvent(new window.InputEvent("input", { bubbles: true, inputType: "insertText", data: "m" }));
  const results = () => [...doc.querySelectorAll(".autocompleteResults.p li")].map((li) => li.textContent);
  await until(() => results().length > 0, "tagcomplete's list");
  assert.ok(results().some((text) => text.includes("smile")), results().join(" | "));
  await until(() => button.getAttribute("aria-disabled") !== "true", "the unload result");
  assert.match(doc.querySelector("#sam3_unload_models_status").textContent, /VRAM 사용 19.6 GB/);
  assert.equal(area.value, "sm", "the button wrote nothing into the prompt");
  assert.deepEqual(env.errors, []);
});
