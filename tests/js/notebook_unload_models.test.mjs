// '모델 내리기' 버튼 (javascript/notebook_unload_models.js) — jsdom 위의 실제 스크립트(기본값 저장 스크립트와 함께).
//
// Gradio 대신 이 테스트가 숨은 #sam3_unload_models_run 의 응답을 흉내 낸다: sam3ext/ui_unload_models.py 의 ARGS_JS 를
// 파이썬 파일에서 그대로 꺼내 실행해 입력 배열을 만들고, 그 첫 칸(요청 id)을 결과 JSON 에 되돌려 숨은 Textbox 에 쓴다.
// 결과 글은 파이썬(sam3ext/unload_models.py)이 만든다 — 페이지는 text · detail 을 그대로 보여 준다.
// 창은 t.after 로 닫는다(스크립트가 인터벌을 건다).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const SCRIPT = readFileSync(path.join(ROOT, "javascript", "notebook_unload_models.js"), "utf8");
const SAVE_SCRIPT = readFileSync(path.join(ROOT, "javascript", "notebook_save_defaults.js"), "utf8");
const BRIDGE_PY = readFileSync(path.join(ROOT, "sam3ext", "ui_unload_models.py"), "utf8");
const ARGS_JS = JSON.parse(/^ARGS_JS = (".*")$/m.exec(BRIDGE_PY)[1]);
const CSS = readFileSync(path.join(ROOT, "style.css"), "utf8");

const LAYOUT = `
  <main id="sam3_notebook_layout"><section class="sam3-notebook-prompt">
    <div id="txt2img_extra_tabs" class="tabs gradio-tabs extra-networks sam3-notebook-extra-tabs">
      <div class="tab-nav scroll-hide" role="tablist">
        <button role="tab" class="selected">Generation</button><button role="tab">임베딩</button>
        <button role="tab">체크포인트</button><button role="tab">로라</button>
        <button data-sam3-lm-btn="1">Manage</button>
        <div class="extra-networks-controls-div"></div>
      </div>
      <div class="tabitem"></div>
    </div>
  </section></main>`;

const STOCK = `
  <div id="tab_txt2img"><div id="txt2img_extra_tabs" class="tabs gradio-tabs extra-networks">
    <div class="tab-nav scroll-hide" role="tablist">
      <button role="tab" class="selected">Generation</button><button role="tab">Checkpoints</button>
      <div class="extra-networks-controls-div"></div>
    </div>
    <div class="tabitem"></div>
  </div></div>`;

const BRIDGE = `
  <div id="sam3_unload_models_bridge" class="hidden">
    <div id="sam3_unload_models_request"><textarea></textarea></div>
    <button id="sam3_unload_models_run">sam3_unload_models</button>
    <div id="sam3_unload_models_result"><label><textarea></textarea></label></div>
  </div>`;

function build(t, { layout = true, stock = false, bridge = true, save = true } = {}) {
  const dom = new JSDOM(`<!doctype html><html><body><div class="gradio-container">
      ${layout ? LAYOUT : ""}${stock ? STOCK : ""}<footer id="footer"></footer>${bridge ? BRIDGE : ""}
    </div></body></html>`, { runScripts: "outside-only", pretendToBeVisual: true, url: "http://127.0.0.1:7860/" });
  t.after(() => dom.window.close());
  const { window } = dom;
  window.gradioApp = () => window.document;
  window.__sam3SaveDefaultsTestHooks = {};
  window.__sam3UnloadModelsTestHooks = {};
  if (save) window.eval(SAVE_SCRIPT);       // Forge loads the extension's scripts in file-name order
  window.eval(SCRIPT);
  const hooks = window.__sam3UnloadModelsTestHooks;
  Object.assign(hooks.config, { pollMs: 5, timeoutMs: 400, doneVisibleMs: 0, errorVisibleMs: 0 });
  return { dom, window, doc: window.document, hooks, saveHooks: window.__sam3SaveDefaultsTestHooks };
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const DONE = (args, extra = {}) => ({
  ok: true, request: args[0], refused: false,
  text: "VRAM 사용 19.6 GB → 3.1 GB (Forge 모델 · SAM3) · 이 프로세스 예약 15.2 → 0.0 GB",
  detail: "GPU 전체 사용: 19.6 GB → 3.1 GB (24.0 GB 중 — 같은 GPU 의 다른 프로그램 몫 포함)\n내림: Forge 모델, SAM3",
  released: ["forge", "sam3"], idle: [], errors: [], before: null, after: null, ...extra,
});

// The hidden Gradio button: Gradio calls the event's js with [inputs…, outputs…] and sends back what it returns.
function fakeBridge(env, respond, delayMs = 10) {
  const run = env.doc.querySelector("#sam3_unload_models_run");
  const result = env.doc.querySelector("#sam3_unload_models_result textarea");
  const argsJs = env.window.eval(`(${ARGS_JS})`);
  const sent = [];
  run.addEventListener("click", () => {
    const args = argsJs("", result.value);
    sent.push(args);
    setTimeout(() => {
      const payload = respond(args);
      if (payload !== undefined) result.value = typeof payload === "string" ? payload : JSON.stringify(payload);
    }, delayMs);
  });
  return sent;
}

function ui(env) {
  return {
    wrap: env.doc.querySelector("#sam3_unload_models"),
    button: env.doc.querySelector("#sam3_unload_models button"),
    status: env.doc.querySelector("#sam3_unload_models_status"),
    save: env.doc.querySelector("#sam3_save_defaults"),
  };
}

function mount(env) {
  if (env.saveHooks.ensureButton) env.saveHooks.ensureButton();
  return env.hooks.ensureButton();
}

test("right after '기본값 저장' in the notebook tab row, with the save button's look and its own status", (t) => {
  const env = build(t);
  assert.equal(mount(env), true);
  assert.equal(mount(env), true);
  const { wrap, button, status, save } = ui(env);
  const row = env.doc.querySelector("#txt2img_extra_tabs > .tab-nav");
  assert.equal(env.doc.querySelectorAll("#sam3_unload_models").length, 1);
  assert.equal(wrap.parentElement, row);
  assert.equal(wrap.previousElementSibling, save, "directly after the save group");
  assert.equal(row.lastElementChild, wrap);
  const saveButton = save.querySelector("button");
  // the same classes as the save button (one more each, for its own place in the row)
  assert.ok(wrap.classList.contains("sam3-save-defaults") && wrap.classList.contains("sam3-unload-models"));
  for (const name of saveButton.classList) assert.ok(button.classList.contains(name), name);
  assert.ok(button.classList.contains("sam3-unload-models-button"));
  assert.ok(status.classList.contains("sam3-save-defaults-status"));
  assert.equal(button.textContent, "모델 내리기");
  assert.equal(button.type, "button");
  assert.match(button.title, /Unload All Models/);
  assert.match(button.title, /SAM3/);
  assert.match(button.title, /생성 중에는 동작하지 않습니다/);
  assert.equal(button.getAttribute("aria-describedby"), "sam3_unload_models_status");
  assert.equal(status.getAttribute("role"), "status");
  assert.equal(status.getAttribute("aria-live"), "polite");
  assert.equal(status.textContent, "");
  assert.equal(wrap.querySelectorAll("input, select, [type=checkbox]").length, 0, "no extra toggles");
});

test("mounted before the save button: moved behind it on the next check", (t) => {
  const env = build(t);
  assert.equal(env.hooks.ensureButton(), true);              // the save group is not there yet
  const row = env.doc.querySelector("#txt2img_extra_tabs > .tab-nav");
  assert.equal(row.lastElementChild, ui(env).wrap);
  env.saveHooks.ensureButton();                              // the save script appends its group at the end
  assert.equal(row.lastElementChild, ui(env).save);
  env.hooks.ensureButton();
  assert.equal(ui(env).wrap.previousElementSibling, ui(env).save);
  assert.equal(row.lastElementChild, ui(env).wrap);
});

test("no save script at all: still at the row's end", (t) => {
  const env = build(t, { save: false });
  assert.equal(env.hooks.ensureButton(), true);
  assert.equal(env.doc.querySelector("#txt2img_extra_tabs > .tab-nav").lastElementChild, ui(env).wrap);
});

test("Forge's own layout (notebook layout off) gets no button, though its tab row is there", (t) => {
  const env = build(t, { layout: false, stock: true });
  assert.equal(env.hooks.ensureButton(), false);
  env.window.dispatchEvent(new env.window.CustomEvent("sam3:notebook-mounted"));
  assert.equal(env.doc.querySelector("#sam3_unload_models"), null);
});

test("re-attached after the save group when Gradio redraws the row (notebook-mounted event)", (t) => {
  const env = build(t, { layout: false });
  assert.equal(env.hooks.ensureButton(), false);
  const host = env.doc.createElement("div");
  host.innerHTML = LAYOUT;
  env.doc.querySelector(".gradio-container").appendChild(host.firstElementChild);
  env.window.dispatchEvent(new env.window.CustomEvent("sam3:notebook-mounted"));
  const row = env.doc.querySelector("#txt2img_extra_tabs > .tab-nav");
  assert.equal(ui(env).wrap.parentElement, row);
  assert.equal(ui(env).wrap.previousElementSibling, ui(env).save);
  const fresh = row.cloneNode(false);
  row.replaceWith(fresh);
  assert.equal(mount(env), true);
  assert.equal(ui(env).wrap.parentElement, fresh);
  assert.equal(ui(env).wrap.previousElementSibling, ui(env).save);
  assert.equal(env.doc.querySelectorAll("#sam3_unload_models").length, 1);
});

test("one click: the hidden button with this request id, then Python's one line next to the button", async (t) => {
  const env = build(t);
  mount(env);
  const sent = fakeBridge(env, (args) => DONE(args));
  ui(env).button.click();
  assert.equal(ui(env).button.getAttribute("aria-disabled"), "true");
  assert.equal(ui(env).button.getAttribute("aria-busy"), "true");
  assert.equal(ui(env).status.textContent, "내리는 중…");
  ui(env).button.click();                                  // a second click while busy does nothing
  assert.ok(await waitFor(() => ui(env).status.getAttribute("data-tone") !== "busy"));
  assert.equal(sent.length, 1);
  assert.match(sent[0][0], /^[A-Za-z0-9_-]{1,64}$/);
  assert.equal(sent[0].length, 2, "the request id and Gradio's trailing output value");
  assert.equal(ui(env).status.textContent,
    "VRAM 사용 19.6 GB → 3.1 GB (Forge 모델 · SAM3) · 이 프로세스 예약 15.2 → 0.0 GB");
  assert.equal(ui(env).status.getAttribute("data-tone"), "saved");
  assert.match(ui(env).status.title, /GPU 전체 사용: 19\.6 GB → 3\.1 GB/);
  assert.equal(ui(env).button.hasAttribute("aria-disabled"), false);
  assert.equal(ui(env).button.hasAttribute("aria-busy"), false);
  // the save button's own status is untouched
  assert.equal(env.doc.querySelector("#sam3_save_defaults_status").textContent, "");
});

test("a refusal while generating is shown as an error", async (t) => {
  const env = build(t);
  mount(env);
  fakeBridge(env, (args) => ({ ok: false, request: args[0], refused: true, text: "생성 중에는 내릴 수 없습니다",
                                detail: "생성 중에는 내릴 수 없습니다 — 생성이 끝난 뒤 다시 누르세요." }));
  const summary = await env.hooks.unloadModels();
  assert.equal(summary.tone, "error");
  assert.equal(ui(env).status.textContent, "생성 중에는 내릴 수 없습니다");
  assert.equal(ui(env).status.getAttribute("data-tone"), "error");
});

test("a partial failure keeps Python's text (what was released and what failed)", async (t) => {
  const env = build(t);
  mount(env);
  const text = "VRAM 사용 19.6 GB → 5.0 GB (Forge 모델 · SAM3) · 이 프로세스 예약 15.2 → 1.9 GB · 실패: TIPO";
  fakeBridge(env, (args) => DONE(args, { ok: false, text, detail: "실패 — TIPO: RuntimeError: boom" }));
  const summary = await env.hooks.unloadModels();
  assert.deepEqual({ ...summary }, { text, tone: "error", detail: "실패 — TIPO: RuntimeError: boom" });
});

test("an old result in the hidden textbox is not mistaken for this click's", async (t) => {
  const env = build(t);
  mount(env);
  env.doc.querySelector("#sam3_unload_models_result textarea").value =
    JSON.stringify(DONE(["old"], { text: "VRAM 사용 9.9 GB → 9.9 GB (내릴 모델 없음)" }));
  fakeBridge(env, (args) => DONE(args));
  const summary = await env.hooks.unloadModels();
  assert.match(summary.text, /^VRAM 사용 19\.6 GB/);
});

test("a click already running is not started twice (also from code)", async (t) => {
  const env = build(t);
  mount(env);
  const sent = fakeBridge(env, (args) => DONE(args), 30);
  const first = env.hooks.unloadModels();
  assert.equal(await env.hooks.unloadModels(), null);
  assert.equal((await first).tone, "saved");
  assert.equal(sent.length, 1);
});

test("the keyboard focus stays on the button while it unloads (it is never disabled)", async (t) => {
  const env = build(t);
  mount(env);
  fakeBridge(env, (args) => DONE(args), 30);
  const { button } = ui(env);
  button.focus();
  button.click();
  assert.equal(button.disabled, false);
  assert.equal(env.doc.activeElement, button);
  assert.ok(await waitFor(() => ui(env).status.getAttribute("data-tone") === "saved"));
  assert.equal(env.doc.activeElement, button);
});

test("a bridge that never answers with this request id times out as an error", async (t) => {
  const env = build(t);
  mount(env);
  fakeBridge(env, () => DONE(["someone-else"]));
  const summary = await env.hooks.unloadModels();
  assert.equal(summary.tone, "error");
  assert.equal(summary.text, "Forge 가 모델 내리기에 응답하지 않았습니다");
});

test("no bridge (wiring failed at startup): the failure says where to look", async (t) => {
  const env = build(t, { bridge: false });
  mount(env);
  const summary = await env.hooks.unloadModels();
  assert.equal(summary.tone, "error");
  assert.match(summary.text, /^모델 내리기 연결이 없습니다/);
  assert.match(summary.text, /\[sam-extra\] unload models/);
});

test("the CSS puts it right after the save group in the same look", () => {
  const block = (selector) => {
    const start = CSS.indexOf(selector + " {");
    assert.notEqual(start, -1, selector);
    return CSS.slice(start, CSS.indexOf("}", start));
  };
  const own = block("#sam3_notebook_layout .sam3-notebook-extra-tabs > .tab-nav > .sam3-save-defaults.sam3-unload-models");
  assert.match(own, /order: 2;/);
  assert.match(own, /margin-left: 0;/);
  // the save group keeps the row's right end (order 1, auto margin); the button look is the shared class
  assert.match(block("#sam3_notebook_layout .sam3-notebook-extra-tabs > .tab-nav > .sam3-save-defaults"), /order: 1;/);
  assert.match(block(".sam3-save-defaults-button"), /var\(--button-secondary-background-fill/);
  assert.doesNotMatch(CSS, /\.sam3-unload-models-button\s*\{/, "no separate button style");
});

async function waitFor(predicate, timeoutMs = 2000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (predicate()) return true;
    await delay(5);
  }
  return false;
}
