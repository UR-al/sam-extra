// '기본값 저장' 버튼 (javascript/notebook_save_defaults.js) — jsdom 위의 실제 스크립트.
//
// Gradio 대신 이 테스트가 두 버튼의 응답을 흉내 낸다:
//  - Forge 의 #ui_defaults_apply: 결과 HTML 칸의 안쪽 div 에 .pending 을 붙였다가 떼고 "Wrote N changes." 를 쓴다
//    (gradio html/Index.svelte 의 class:pending, modules/ui_loadsave.py ui_apply 의 문구).
//  - 숨은 #sam3_save_defaults_run: sam3ext/ui_save_defaults.py 의 ARGS_JS 를 파이썬 파일에서 그대로 꺼내 실행해
//    입력 배열을 만들고, 그 첫 칸(요청 id)을 결과 JSON 에 되돌려 숨은 Textbox 에 쓴다.
// 창은 t.after 로 닫는다(스크립트가 인터벌을 건다).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const SCRIPT = readFileSync(path.join(ROOT, "javascript", "notebook_save_defaults.js"), "utf8");
const BRIDGE_PY = readFileSync(path.join(ROOT, "sam3ext", "ui_save_defaults.py"), "utf8");
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

const SETTINGS = `
  <div id="tab_settings"><div id="settings_tab_defaults" class="tabitem">
    <div class="block gradio-html"><div class="wrap center hide"></div>
      <div><div class="prose gradio-html">This page allows you to change default values …</div></div></div>
    <div class="row"><button id="ui_defaults_view">View changes</button><button id="ui_defaults_apply">Apply</button></div>
    <div class="block gradio-html" id="review"><div class="wrap center full hide"></div>
      <div class="review-inner"><div class="prose gradio-html"></div></div></div>
  </div></div>`;

// Forge's own two-column txt2img (notebook layout off): the same tab row, outside #sam3_notebook_layout.
const STOCK = `
  <div id="tab_txt2img"><div id="txt2img_extra_tabs" class="tabs gradio-tabs extra-networks">
    <div class="tab-nav scroll-hide" role="tablist">
      <button role="tab" class="selected">Generation</button><button role="tab">Checkpoints</button>
      <div class="extra-networks-controls-div"></div>
    </div>
    <div class="tabitem"></div>
  </div></div>`;

const BRIDGE = `
  <div id="sam3_save_defaults_bridge" class="hidden">
    <div id="sam3_save_defaults_request"><textarea></textarea></div>
    <button id="sam3_save_defaults_run">sam3_save_defaults</button>
    <div id="sam3_save_defaults_result"><label><textarea></textarea></label></div>
  </div>`;

function build(t, { layout = true, stock = false, settings = true, bridge = true } = {}) {
  const dom = new JSDOM(`<!doctype html><html><body><div class="gradio-container">
      ${layout ? LAYOUT : ""}${stock ? STOCK : ""}${settings ? SETTINGS : ""}<footer id="footer"></footer>${bridge ? BRIDGE : ""}
    </div></body></html>`, { runScripts: "outside-only", pretendToBeVisual: true, url: "http://127.0.0.1:7860/" });
  t.after(() => dom.window.close());
  const { window } = dom;
  window.gradioApp = () => window.document;
  window.__sam3SaveDefaultsTestHooks = {};
  window.eval(SCRIPT);
  const hooks = window.__sam3SaveDefaultsTestHooks;
  Object.assign(hooks.config, { pollMs: 5, defaultsTimeoutMs: 400, presetTimeoutMs: 400, savedVisibleMs: 0,
                                errorVisibleMs: 0 });
  return { dom, window, doc: window.document, hooks };
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Forge's Defaults Apply as Gradio renders it: pending, then the new text (or an error status).
// startDelayMs: how long Gradio takes to show the pending state after the click.
function fakeForgeApply(env, { text = "Wrote 3 changes.", error = false, delayMs = 15, startDelayMs = 1 } = {}) {
  const apply = env.doc.querySelector("#ui_defaults_apply");
  const inner = env.doc.querySelector("#review .review-inner");
  const prose = env.doc.querySelector("#review .prose");
  const wrap = env.doc.querySelector("#review .wrap");
  const calls = [];
  apply.addEventListener("click", () => {
    calls.push(Date.now());
    setTimeout(() => {
      inner.classList.add("pending");
      wrap.classList.remove("hide");
      setTimeout(() => {
        inner.classList.remove("pending");
        if (error) {
          wrap.innerHTML = '<span class="error">Error</span>';
        } else {
          wrap.classList.add("hide");
          prose.innerHTML = `<p>${text}</p>`;
        }
      }, delayMs);
    }, startDelayMs);
  });
  return calls;
}

const UI_VALUES = ["anima-preview2.safetensors [1234abcd]", ["qwen_image_vae.safetensors"], "Automatic",
  32, 0, 28, "ER SDE (Tunable)", "Euler a", "Flow Cosmos rho7", "Beta", 832, 1024, 1216, 1024,
  4.5, 4.0, 5.0, 3.0, 2.5, 3.5, 2, 1];

// The hidden Gradio button: Gradio calls the event's js with [inputs…, outputs…] and sends back what it returns.
function fakeBridge(env, respond) {
  const run = env.doc.querySelector("#sam3_save_defaults_run");
  const result = env.doc.querySelector("#sam3_save_defaults_result textarea");
  const argsJs = env.window.eval(`(${ARGS_JS})`);
  const sent = [];
  run.addEventListener("click", () => {
    const args = argsJs("", "anima", ...UI_VALUES, result.value);
    sent.push(args);
    setTimeout(() => {
      const payload = respond(args);
      if (payload !== undefined) result.value = typeof payload === "string" ? payload : JSON.stringify(payload);
    }, 10);
  });
  return sent;
}

const OK = (args, extra = {}) => ({
  ok: true, request: args[0], preset: "anima", written: 22, changed: 2,
  changes: [
    { key: "anima_t2i_step", label: "txt2img Steps", old: 28, new: 32 },
    { key: "forge_additional_modules_anima", label: "VAE / Text Encoder", old: [],
      new: ["C:\\forge\\models\\VAE\\qwen_image_vae.safetensors"] },
  ],
  skipped: [], ...extra,
});

function ui(env) {
  return {
    wrap: env.doc.querySelector("#sam3_save_defaults"),
    button: env.doc.querySelector("#sam3_save_defaults button"),
    status: env.doc.querySelector("#sam3_save_defaults_status"),
  };
}

test("one button at the right end of the notebook tab row, with a live status", (t) => {
  const env = build(t);
  assert.equal(env.hooks.ensureButton(), true);
  assert.equal(env.hooks.ensureButton(), true);
  const { wrap, button, status } = ui(env);
  const row = env.doc.querySelector("#txt2img_extra_tabs > .tab-nav");
  assert.equal(env.doc.querySelectorAll("#sam3_save_defaults").length, 1);
  assert.equal(wrap.parentElement, row);
  assert.equal(row.lastElementChild, wrap, "after Generation · 임베딩 · 체크포인트 · 로라 · Manage and the controls");
  assert.equal(wrap.className, "sam3-save-defaults");
  assert.equal(button.textContent, "기본값 저장");
  assert.equal(button.type, "button");
  assert.match(button.title, /Settings → Defaults/);
  assert.equal(button.getAttribute("aria-describedby"), "sam3_save_defaults_status");
  assert.equal(status.getAttribute("role"), "status");
  assert.equal(status.getAttribute("aria-live"), "polite");
  assert.equal(status.textContent, "");
  assert.equal(wrap.querySelectorAll("input, select, [type=checkbox]").length, 0, "no extra toggles");
});

test("Forge's own layout (notebook layout off) gets no button, though its tab row is there", (t) => {
  const env = build(t, { layout: false, stock: true });
  assert.ok(env.doc.querySelector("#txt2img_extra_tabs > .tab-nav"));
  assert.equal(env.hooks.ensureButton(), false);
  env.window.dispatchEvent(new env.window.CustomEvent("sam3:notebook-mounted"));
  assert.equal(env.doc.querySelector("#sam3_save_defaults"), null);
});

test("no notebook layout, no button; it is re-attached when Gradio redraws the row", (t) => {
  const env = build(t, { layout: false });
  assert.equal(env.hooks.ensureButton(), false);
  assert.equal(env.doc.querySelector("#sam3_save_defaults"), null);
  const host = env.doc.createElement("div");
  host.innerHTML = LAYOUT;
  env.doc.querySelector(".gradio-container").appendChild(host.firstElementChild);
  env.window.dispatchEvent(new env.window.CustomEvent("sam3:notebook-mounted"));
  const row = env.doc.querySelector("#txt2img_extra_tabs > .tab-nav");
  assert.equal(ui(env).wrap.parentElement, row);
  const fresh = row.cloneNode(false);
  row.replaceWith(fresh);
  assert.equal(env.hooks.ensureButton(), true);
  assert.equal(ui(env).wrap.parentElement, fresh);
  assert.equal(env.doc.querySelectorAll("#sam3_save_defaults").length, 1);
});

test("one click: Forge's Defaults Apply, then the active preset — one combined status", async (t) => {
  const env = build(t);
  env.hooks.ensureButton();
  const applied = fakeForgeApply(env, { text: "Wrote 3 changes." });
  const sent = fakeBridge(env, (args) => OK(args));
  ui(env).button.click();
  assert.equal(ui(env).button.getAttribute("aria-disabled"), "true");
  assert.equal(ui(env).button.getAttribute("aria-busy"), "true");
  assert.equal(ui(env).status.textContent, "저장 중…");
  ui(env).button.click();                                  // a second click while busy does nothing
  const summary = await waitFor(env, () => ui(env).status.getAttribute("data-tone") !== "busy");
  assert.ok(summary);
  assert.equal(applied.length, 1);
  assert.equal(sent.length, 1);
  // the two files' counts apart: Forge's ui-config (Wrote 3 changes.) and the preset's (changed 2)
  assert.equal(ui(env).status.textContent, "기본값 + anima 프리셋 저장됨 · 바뀐 항목: 기본값 3개, 프리셋 2개");
  assert.equal(ui(env).status.getAttribute("data-tone"), "saved");
  assert.match(ui(env).status.title, /Settings → Defaults: Wrote 3 changes\./);
  assert.match(ui(env).status.title, /txt2img Steps: 28 → 32/);
  assert.match(ui(env).status.title, /VAE \/ Text Encoder: 없음 → qwen_image_vae\.safetensors/);
  assert.equal(ui(env).button.hasAttribute("aria-disabled"), false);
  assert.equal(ui(env).button.hasAttribute("aria-busy"), false);
  // the request id went in the first input; the other inputs (and Gradio's trailing output value) are untouched
  const [args] = sent;
  assert.match(args[0], /^[A-Za-z0-9_-]{1,64}$/);
  assert.deepEqual(JSON.parse(JSON.stringify(args.slice(1, 24))), ["anima", ...UI_VALUES]);   // jsdom realm arrays
  assert.equal(args.length, 25);
});

test("Forge's apply comes first, the preset bridge after it finished", async (t) => {
  const env = build(t);
  env.hooks.ensureButton();
  const order = [];
  env.doc.querySelector("#ui_defaults_apply").addEventListener("click", () => order.push("apply"));
  fakeForgeApply(env, { text: "No changes.", delayMs: 30 });
  fakeBridge(env, (args) => { order.push("preset"); return OK(args, { changed: 0, changes: [] }); });
  env.doc.querySelector("#sam3_save_defaults_run").addEventListener("click", () => order.push("run"));
  const summary = await env.hooks.saveDefaults();
  assert.deepEqual(order, ["apply", "run", "preset"]);
  assert.equal(summary.text, "기본값 + anima 프리셋 저장됨 · 바뀐 항목: 기본값 0개, 프리셋 0개");
  assert.equal(summary.tone, "saved");
});

test("a save already running is not started twice (also from code, not only the busy button)", async (t) => {
  const env = build(t);
  env.hooks.ensureButton();
  const applied = fakeForgeApply(env);
  const sent = fakeBridge(env, (args) => OK(args));
  const first = env.hooks.saveDefaults();
  assert.equal(await env.hooks.saveDefaults(), null);
  assert.equal((await first).tone, "saved");
  assert.equal(applied.length, 1);
  assert.equal(sent.length, 1);
});

test("an old result in the hidden textbox is not mistaken for this click's", async (t) => {
  const env = build(t);
  env.hooks.ensureButton();
  fakeForgeApply(env);
  env.doc.querySelector("#sam3_save_defaults_result textarea").value =
    JSON.stringify({ ok: true, request: "old", preset: "xl", written: 22, changed: 9, changes: [] });
  fakeBridge(env, (args) => OK(args, { changed: 1, changes: [] }));
  const summary = await env.hooks.saveDefaults();
  assert.equal(summary.text, "기본값 + anima 프리셋 저장됨 · 바뀐 항목: 기본값 3개, 프리셋 1개");
});

test("the keyboard focus stays on the button while it saves and after (it is never disabled)", async (t) => {
  // a disabled button loses the focus to <body> in a browser, and the next Tab starts again at the top of the page
  const env = build(t);
  env.hooks.ensureButton();
  fakeForgeApply(env, { delayMs: 30 });
  fakeBridge(env, (args) => OK(args));
  const { button } = ui(env);
  button.focus();
  assert.equal(env.doc.activeElement, button);
  button.click();
  assert.equal(button.disabled, false);
  assert.equal(button.getAttribute("aria-disabled"), "true");
  assert.equal(env.doc.activeElement, button);
  await delay(20);
  assert.equal(ui(env).status.textContent, "저장 중…");
  assert.equal(env.doc.activeElement, button);
  assert.ok(await waitFor(env, () => ui(env).status.getAttribute("data-tone") === "saved"));
  assert.equal(env.doc.activeElement, button);
  assert.equal(button.disabled, false);
  assert.equal(button.hasAttribute("aria-disabled"), false);
  assert.match(CSS, /\.sam3-save-defaults-button\[aria-disabled="true"\] \{[^}]*cursor: progress;/);
  assert.doesNotMatch(CSS, /\.sam3-save-defaults-button:disabled/);
});

test("the two files' counts are shown apart, never added (one edit is written to both files)", (t) => {
  const env = build(t);
  const preset = { ok: true, preset: "anima", written: 22, changed: 4, changes: [] };
  const both = env.hooks.summarize({ ok: true, changed: 3, text: "Wrote 3 changes." }, preset);
  assert.equal(both.text, "기본값 + anima 프리셋 저장됨 · 바뀐 항목: 기본값 3개, 프리셋 4개");
  assert.doesNotMatch(both.text, /7개/);
  assert.equal(both.tone, "saved");
  // a translated Defaults text without one number: only the preset's count, the text as Forge wrote it
  const translated = env.hooks.summarize({ ok: true, changed: null, text: "변경 없음" }, preset);
  assert.equal(translated.text, "기본값 + anima 프리셋 저장됨 · 바뀐 항목: 프리셋 4개 (Defaults: 변경 없음)");
});

test("preset fields kept at 0 are named in the tooltip and never counted", (t) => {
  const env = build(t);
  const kept = [
    { key: "anima_t2i_width", label: "txt2img Width", value: 0 },
    { key: "anima_t2i_height", label: "txt2img Height", value: 0 },
  ];
  const preset = { ok: true, preset: "anima", written: 20, changed: 1, kept,
                   changes: [{ key: "anima_t2i_step", label: "txt2img Steps", old: 28, new: 32 }] };
  const summary = env.hooks.summarize({ ok: true, changed: 2, text: "Wrote 2 changes." }, preset);
  assert.equal(summary.text, "기본값 + anima 프리셋 저장됨 · 바뀐 항목: 기본값 2개, 프리셋 1개");
  assert.equal(summary.detail, [
    "Settings → Defaults: Wrote 2 changes.",
    "anima 프리셋: 설정 20개 저장, 바뀐 항목 1개",
    "  txt2img Steps: 28 → 32",
    "  프리셋 값 0(화면 값을 덮어쓰지 않음)이라 그대로 둔 칸 2개: txt2img Width, txt2img Height",
  ].join("\n"));
  // an older Python side without the field, or nothing kept: no line
  for (const extra of [{}, { kept: [] }]) {
    const plain = env.hooks.summarize({ ok: true, changed: 0, text: "No changes." },
      { ok: true, preset: "anima", written: 22, changed: 0, changes: [], ...extra });
    assert.doesNotMatch(plain.detail, /그대로 둔 칸/);
  }
  // a failed preset save says nothing about kept fields
  const failed = env.hooks.summarize({ ok: true, changed: 0, text: "No changes." },
    { ok: false, preset: "anima", error: "txt2img Steps: 151 은(는) Forge 설정 범위 밖입니다", kept });
  assert.doesNotMatch(failed.detail, /그대로 둔 칸/);
});

test("the button's tooltip says a preset 0 stays", (t) => {
  const env = build(t);
  assert.equal(env.hooks.ensureButton(), true);
  assert.match(env.doc.querySelector("#sam3_save_defaults button").title,
    /프리셋 값이 0\(화면 값을 덮어쓰지 않음\)인 스텝·크기·CFG 는 0 그대로 둡니다/);
});

test("a bridge that never answers with this request id times out as an error", async (t) => {
  const env = build(t);
  env.hooks.ensureButton();
  fakeForgeApply(env);
  fakeBridge(env, () => ({ ok: true, request: "someone-else", preset: "anima", written: 22, changed: 0 }));
  const summary = await env.hooks.saveDefaults();
  assert.equal(summary.tone, "error");
  assert.equal(summary.text, "기본값 저장됨 (바뀐 항목 3개) · 프리셋 저장 실패: Forge 가 프리셋 저장에 응답하지 않았습니다");
});

test("a refused preset save shows Forge's reason", async (t) => {
  const env = build(t);
  env.hooks.ensureButton();
  fakeForgeApply(env, { text: "No changes." });
  fakeBridge(env, (args) => ({ ok: false, request: args[0], preset: "anima",
                               error: "txt2img Steps: 151 은(는) Forge 설정 범위 0~150 밖입니다" }));
  await env.hooks.saveDefaults();
  assert.equal(ui(env).status.textContent,
    "기본값 저장됨 (바뀐 항목 0개) · 프리셋 저장 실패: txt2img Steps: 151 은(는) Forge 설정 범위 0~150 밖입니다");
  assert.equal(ui(env).status.getAttribute("data-tone"), "error");
});

test("no Settings tab: the preset is still saved and the Defaults failure is named", async (t) => {
  const env = build(t, { settings: false });
  env.hooks.ensureButton();
  fakeBridge(env, (args) => OK(args));
  const summary = await env.hooks.saveDefaults();
  assert.equal(summary.tone, "error");
  assert.match(summary.text, /^anima 프리셋 저장됨 \(바뀐 항목 2개\) · 기본값 저장 실패: Settings → Defaults 의 Apply 버튼이 없습니다/);
});

test("an error still shown from the previous Apply does not end this one early", async (t) => {
  // Gradio shows the pending state a moment after the click; until then the old error status is on screen
  const env = build(t);
  env.hooks.ensureButton();
  const wrap = env.doc.querySelector("#review .wrap");
  wrap.classList.remove("hide");
  wrap.innerHTML = '<span class="error">Error</span>';
  fakeForgeApply(env, { text: "Wrote 2 changes.", startDelayMs: 40 });
  fakeBridge(env, (args) => OK(args));
  const summary = await env.hooks.saveDefaults();
  assert.equal(summary.tone, "saved");
  assert.equal(summary.text, "기본값 + anima 프리셋 저장됨 · 바뀐 항목: 기본값 2개, 프리셋 2개");
});

test("Forge's apply ending in an error status is a failure, not a stale success", async (t) => {
  const env = build(t);
  env.hooks.ensureButton();
  env.doc.querySelector("#review .prose").innerHTML = "<p>Wrote 7 changes.</p>";   // from an earlier click
  fakeForgeApply(env, { error: true });
  fakeBridge(env, (args) => OK(args));
  const summary = await env.hooks.saveDefaults();
  assert.equal(summary.tone, "error");
  assert.match(summary.text, /기본값 저장 실패: Forge 의 Defaults Apply 가 오류로 끝났습니다/);
});

test("a repeated 'No changes.' is still read as finished (the pending cycle, not the text)", async (t) => {
  const env = build(t);
  env.hooks.ensureButton();
  env.doc.querySelector("#review .prose").innerHTML = "<p>No changes.</p>";
  fakeForgeApply(env, { text: "No changes." });
  fakeBridge(env, (args) => OK(args, { changed: 0, changes: [] }));
  const started = Date.now();
  const summary = await env.hooks.saveDefaults();
  assert.equal(summary.text, "기본값 + anima 프리셋 저장됨 · 바뀐 항목: 기본값 0개, 프리셋 0개");
  assert.ok(Date.now() - started < 350, "did not wait for the timeout");
});

test("no bridge (wiring failed at startup): Defaults still applied, the preset failure says where to look", async (t) => {
  const env = build(t, { bridge: false });
  env.hooks.ensureButton();
  fakeForgeApply(env, { text: "Wrote 1 change." });
  const summary = await env.hooks.saveDefaults();
  assert.equal(summary.tone, "error");
  assert.match(summary.text, /^기본값 저장됨 \(바뀐 항목 1개\) · 프리셋 저장 실패: 프리셋 저장 연결이 없습니다/);
  assert.match(summary.text, /\[sam-extra\] save defaults/);
});

test("result text parsing", (t) => {
  const env = build(t);
  const parse = env.hooks.parseDefaultsResult;
  assert.deepEqual({ ...parse("Wrote 12 changes.") }, { ok: true, changed: 12, text: "Wrote 12 changes." });
  assert.deepEqual({ ...parse(" No changes. ") }, { ok: true, changed: 0, text: "No changes." });
  assert.equal(parse("변경 4건 기록").changed, 4);              // a translated phrase with one number
  assert.equal(parse("변경 없음").changed, null);
  assert.equal(parse("").ok, false);
});

test("the CSS keeps the button at the row's right end in the existing look", () => {
  const block = (selector) => {
    const start = CSS.indexOf(selector + " {");
    assert.notEqual(start, -1, selector);
    return CSS.slice(start, CSS.indexOf("}", start));
  };
  const wrap = block("#sam3_notebook_layout .sam3-notebook-extra-tabs > .tab-nav > .sam3-save-defaults");
  assert.match(wrap, /order: 1;/);
  assert.match(wrap, /margin-left: auto;/);
  assert.match(block("#sam3_notebook_layout .sam3-notebook-extra-tabs > .tab-nav:has(> .extra-networks-controls-div) > .sam3-save-defaults"),
    /margin-left: 0;/);
  assert.match(block(".sam3-save-defaults-button"), /var\(--button-secondary-background-fill/);
  assert.match(block(".sam3-save-defaults-status[data-tone=\"error\"]"), /--error-text-color/);
  assert.match(CSS, /\.sam3-save-defaults-button:focus-visible \{/);
});

async function waitFor(env, predicate, timeoutMs = 2000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (predicate()) return true;
    await delay(5);
  }
  return false;
}
