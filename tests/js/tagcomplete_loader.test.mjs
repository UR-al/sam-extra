// tagcomplete_loader.js — 내장 Tag Autocomplete 의 JS 를 붙이는 로더. 실제 javascript/tagcomplete_loader.js 를 jsdom 에 올린다.
//
// 시작 정보는 Forge 설정(opts)의 숨은 키 sam3_tac_builtin_boot 다(sam3ext/tagcomplete/coexist.py). Forge 의 script.js 처럼
// `let opts` · onOptionsAvailable · executeCallbacks 를 가진 작은 대역을 classic <script> 로 먼저 돌리고(let 은 window 속성이
// 아니라 같은 전역 렉시컬 범위라 runScripts "dangerously"), ui.js 의 load_webui_settings 처럼 설정 JSON 을 opts 에 넣으며
// 콜백을 부른다. jsdom 은 동적으로 붙인 <script src> 를 받지 않으므로(resources 없음) 테스트가 load / error 이벤트를 낸다.
// Forge 의 MutationObserver(script.js — gradioApp() 의 childList+subtree)를 흉내 내, 로더가 마지막에 넣었다 빼는 주석 노드가
// onUiUpdate 를 한 번 더 돌게 하는지 본다. 로더는 네트워크를 쓰지 않는다 — fetch 는 부르면 기록만 하는 가짜다.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM, VirtualConsole } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const LOADER = readFileSync(path.join(ROOT, "javascript", "tagcomplete_loader.js"), "utf8");
const BOOT = "sam3_tac_builtin_boot";
const FILES = [
  "__globals.js", "_baseParser.js", "_caretPosition.js", "_result.js", "_textAreas.js", "_utils.js", "ext_chants.js",
  "ext_embeddings.js", "ext_loras.js", "ext_lycos.js", "ext_modelKeyword.js", "ext_styles.js", "ext_umi.js",
  "ext_wildcards.js", "tagAutocomplete.js",
];
const URLS = FILES.map((name, i) => `file=extensions/forge_sam3_extension/sam3ext/tagcomplete/javascript/${name}?${1000 + i}`);
const ENABLED = { enabled: true, reason: null, upstream: "eduardoabreu81/sd-webui-tagcomplete-neo@8e0d81f", scripts: URLS };

// Forge script.js / ui.js stand-in (same behaviour: `let opts`, callbacks run at once when opts are already filled)
const FORGE = `
let opts = {};
const optionsAvailableCallbacks = [];
function onOptionsAvailable(callback) {
    if (Object.keys(opts).length > 0) {
        callback();
        return;
    }
    optionsAvailableCallbacks.push(callback);
}
function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("error running callback", callback, ":", e);
        }
    }
}
function forgeLoadSettings(json) {
    opts = JSON.parse(json);
    executeCallbacks(optionsAvailableCallbacks);
}
`;
// a webui without onOptionsAvailable: the loader waits for opts to fill by itself
const OLD_WEBUI = `
let opts = {};
function forgeLoadSettings(json) {
    opts = JSON.parse(json);
}
`;

function settingsWith(boot) {
  return { tac_active: true, [BOOT]: typeof boot === "string" ? boot : JSON.stringify(boot) };
}

function load(t, { prompt = true, before = "", webui = FORGE, fastTimers = false } = {}) {
  const logs = { log: [], warn: [], error: [] };
  const virtualConsole = new VirtualConsole();
  for (const kind of Object.keys(logs)) virtualConsole.on(kind, (...args) => logs[kind].push(args.join(" ")));
  virtualConsole.on("jsdomError", (error) => logs.error.push(String(error && (error.stack || error.message))));
  const dom = new JSDOM(
    `<!doctype html><html><head></head><body><gradio-app>${prompt ? '<div id="txt2img_prompt"><textarea></textarea></div>' : ""}`
    + "</gradio-app></body></html>",
    { url: "http://127.0.0.1:7860/", runScripts: "dangerously", virtualConsole },
  );
  t.after(() => dom.window.close());
  const { window } = dom;
  const doc = window.document;
  window.gradioApp = () => doc.querySelector("gradio-app");
  if (fastTimers) {
    const real = window.setTimeout.bind(window);
    window.setTimeout = (fn) => real(fn, 0);
  }
  const fetches = [];
  window.fetch = async (url) => {
    fetches.push(url);
    return { status: 404, ok: false, json: async () => null };
  };
  // Forge script.js: one MutationObserver on gradioApp() runs the onUiUpdate callbacks
  const uiUpdates = [];
  new window.MutationObserver((mutations) => uiUpdates.push(mutations)).observe(window.gradioApp(), { childList: true, subtree: true });
  const run = (source) => {   // a classic <script> in <head>, like Forge's head scripts (runs when inserted)
    const script = doc.createElement("script");
    script.textContent = source;
    doc.head.appendChild(script);
    script.remove();   // keeps head's <script> list to what the loader adds
  };
  if (webui) run(webui);
  if (before) run(before);
  run(LOADER);
  const scripts = () => [...doc.head.querySelectorAll("script")];
  return {
    window, doc, logs, fetches, uiUpdates, scripts, run,
    settings: (values) => window.forgeLoadSettings(JSON.stringify(values)),
    state: () => doc.documentElement.getAttribute("data-sam3-tagcomplete"),
  };
}

async function until(check, label, ms = 3000) {
  const start = Date.now();
  while (!check()) {
    if (Date.now() - start > ms) assert.fail(`timed out waiting for ${label}`);
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

const tick = (ms = 20) => new Promise((resolve) => setTimeout(resolve, ms));

function fire(window, script, type = "load") {
  script.dispatchEvent(new window.Event(type));
}

test("adds the 15 upstream files in the boot order once Forge's settings are there, then lets onUiUpdate run once more", async (t) => {
  const env = load(t);
  await tick();
  assert.equal(env.scripts().length, 0, "nothing before the settings arrive");
  assert.equal(env.state(), "loading");
  env.settings(settingsWith(ENABLED));
  await until(() => env.scripts().length === 15, "15 scripts");
  const scripts = env.scripts();
  assert.deepEqual(scripts.map((s) => s.getAttribute("src")), URLS);
  for (const script of scripts) assert.equal(script.async, false, "async=false keeps the insertion order");
  assert.equal(env.state(), "loading");
  const before = env.uiUpdates.length;
  scripts.forEach((script) => fire(env.window, script));
  await until(() => env.state() === "started", "started");
  assert.ok(env.uiUpdates.length > before, "Forge's observer saw the nudge");
  const nudged = env.uiUpdates.slice(before).flat();
  assert.ok(nudged.some((m) => [...m.addedNodes].some((n) => n.nodeType === 8 && n.data === "sam3-tagcomplete")));
  assert.equal(env.window.gradioApp().childNodes.length, 1, "the comment node is gone again");
  assert.deepEqual(env.fetches, [], "no request: nothing waits for a route");
  assert.deepEqual(env.logs.error, []);
  assert.deepEqual(env.logs.warn, []);
});

test("settings that arrive 3.5 s after the page loaded still start it (regression: the boot route gave up after 2.5 s)", async (t) => {
  // review probe: Forge registers app_started routes 3.4-7 s after a page opened during its start-up tail loads; the
  // old loader asked GET sam3-tagcomplete/boot 6 times in 2.5 s, then marked the page failed for good
  const env = load(t);
  await tick(3500);
  assert.equal(env.state(), "loading", "still waiting, not failed");
  assert.deepEqual(env.logs.warn, []);
  env.settings(settingsWith(ENABLED));
  await until(() => env.scripts().length === 15, "15 scripts");
  env.scripts().forEach((script) => fire(env.window, script));
  await until(() => env.state() === "started", "started");
  assert.deepEqual(env.fetches, []);
});

test("settings already loaded when the loader runs: it starts at once", async (t) => {
  const env = load(t, { webui: `${FORGE}\nforgeLoadSettings(${JSON.stringify(JSON.stringify(settingsWith(ENABLED)))});` });
  await until(() => env.scripts().length === 15, "15 scripts");
});

test("a webui without onOptionsAvailable: waits for opts to fill", async (t) => {
  const env = load(t, { webui: OLD_WEBUI, fastTimers: true });
  await tick(50);
  assert.equal(env.state(), "loading");
  env.settings(settingsWith(ENABLED));
  await until(() => env.scripts().length === 15, "15 scripts");
  await tick(50);
  assert.equal(env.scripts().length, 15, "started once");
});

test("the boot information may also be an object", async (t) => {
  const env = load(t);
  env.settings({ tac_active: true, [BOOT]: ENABLED });
  await until(() => env.scripts().length === 15, "15 scripts");
});

for (const [label, before] of [
  ["TAC_CFG", "var TAC_CFG = null;"],
  ["addAutocompleteToArea", "function addAutocompleteToArea(area) {}"],
]) {
  test(`another tagcomplete on the page (${label}): nothing is added`, async (t) => {
    const env = load(t, { before });
    env.settings(settingsWith(ENABLED));
    await until(() => env.state() === "skipped", "skipped");
    assert.equal(env.scripts().length, 0);
    assert.deepEqual(env.logs.log, ["[sam-extra] Tag Autocomplete: another tagcomplete is loaded on this page; built-in copy not started"]);
  });
}

test("the server says it is off: the reason is logged once, nothing is added", async (t) => {
  const reason = "another tagcomplete extension is enabled (sd-webui-tagcomplete-neo: C:\\forge\\extensions\\sd-webui-tagcomplete-neo)";
  const env = load(t);
  env.settings(settingsWith({ enabled: false, reason, upstream: "x", scripts: [] }));
  await until(() => env.state() === "skipped", "skipped");
  assert.equal(env.scripts().length, 0);
  assert.deepEqual(env.logs.log, [`[sam-extra] Tag Autocomplete: built-in copy not started: ${reason}`]);
});

test("no boot information in the settings (the entry script did not run): one warning, nothing added", async (t) => {
  const env = load(t);
  env.settings({ tac_active: true });
  await until(() => env.state() === "failed", "failed");
  assert.equal(env.scripts().length, 0);
  assert.equal(env.logs.warn.length, 1);
  assert.match(env.logs.warn[0], /have no sam3_tac_builtin_boot .*scripts\/builtin_tagcomplete\.py did not run/);
});

test("boot information that is not JSON: one warning, nothing added", async (t) => {
  const env = load(t);
  env.settings(settingsWith("{not json"));
  await until(() => env.state() === "failed", "failed");
  assert.equal(env.scripts().length, 0);
  assert.match(env.logs.warn[0], /sam3_tac_builtin_boot is not a JSON object/);
});

test("a file that fails to load is reported; the rest still run", async (t) => {
  const env = load(t);
  env.settings(settingsWith(ENABLED));
  await until(() => env.scripts().length === 15, "15 scripts");
  env.scripts().forEach((script, i) => fire(env.window, script, i === 3 ? "error" : "load"));
  await until(() => env.state() === "failed", "failed");
  assert.equal(env.logs.error.length, 1);
  assert.match(env.logs.error[0], /could not load file=.*_result\.js/);
});

test("no nudge before the UI is built (later mutations run onUiUpdate anyway)", async (t) => {
  const env = load(t, { prompt: false });
  env.settings(settingsWith(ENABLED));
  await until(() => env.scripts().length === 15, "15 scripts");
  env.scripts().forEach((script) => fire(env.window, script));
  await until(() => env.state() === "started", "started");
  await tick();
  assert.equal(env.uiUpdates.length, 0);
});

test("the loader runs once per page", async (t) => {
  const env = load(t);
  env.run(LOADER);
  env.settings(settingsWith(ENABLED));
  await until(() => env.scripts().length === 15, "15 scripts");
  await tick();
  assert.equal(env.scripts().length, 15);
});

test("defines no global names", async (t) => {
  // (no element ids on this page: jsdom exposes them as window properties too)
  const env = load(t, { prompt: false });
  env.settings(settingsWith({ enabled: false, reason: "off", scripts: [] }));
  await until(() => env.state() === "skipped", "skipped");
  const fresh = new JSDOM("<!doctype html><html></html>", { runScripts: "outside-only" });
  t.after(() => fresh.window.close());
  const baseline = new Set(Object.getOwnPropertyNames(fresh.window));
  const added = Object.getOwnPropertyNames(env.window).filter((name) => !baseline.has(name));
  assert.deepEqual(added.sort(), ["executeCallbacks", "fetch", "forgeLoadSettings", "gradioApp", "onOptionsAvailable"],
    "only the test's own stand-ins (Forge's let/const are not window properties)");
});
