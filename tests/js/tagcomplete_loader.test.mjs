// tagcomplete_loader.js — 내장 Tag Autocomplete 의 JS 를 붙이는 로더. 실제 javascript/tagcomplete_loader.js 를 jsdom 에 올린다.
//
// 서버(/sam3-tagcomplete/boot)는 가짜 fetch 다. jsdom 은 동적으로 붙인 <script src> 를 받지 않으므로(resources 없음) 테스트가
// load / error 이벤트를 낸다. Forge 의 MutationObserver(script.js — gradioApp() 의 childList+subtree)를 흉내 내, 로더가
// 마지막에 넣었다 빼는 주석 노드가 onUiUpdate 를 한 번 더 돌게 하는지 본다.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM, VirtualConsole } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const LOADER = readFileSync(path.join(ROOT, "javascript", "tagcomplete_loader.js"), "utf8");
const FILES = [
  "__globals.js", "_baseParser.js", "_caretPosition.js", "_result.js", "_textAreas.js", "_utils.js", "ext_chants.js",
  "ext_embeddings.js", "ext_loras.js", "ext_lycos.js", "ext_modelKeyword.js", "ext_styles.js", "ext_umi.js",
  "ext_wildcards.js", "tagAutocomplete.js",
];
const URLS = FILES.map((name, i) => `file=extensions/forge_sam3_extension/sam3ext/tagcomplete/javascript/${name}?${1000 + i}`);
const ENABLED = { enabled: true, reason: null, upstream: "eduardoabreu81/sd-webui-tagcomplete-neo@8e0d81f", scripts: URLS };

function json(status, body) {
  return { status, ok: status >= 200 && status < 300, json: async () => body };
}

// answers: list of fetch results in order (a function throws = network error); the last one repeats
function load(t, { answers = [json(200, ENABLED)], prompt = true, before = "", fastTimers = false } = {}) {
  const logs = { log: [], warn: [], error: [] };
  const virtualConsole = new VirtualConsole();
  for (const kind of Object.keys(logs)) virtualConsole.on(kind, (...args) => logs[kind].push(args.join(" ")));
  virtualConsole.on("jsdomError", (error) => logs.error.push(String(error && (error.stack || error.message))));
  const dom = new JSDOM(
    `<!doctype html><html><head></head><body><gradio-app>${prompt ? '<div id="txt2img_prompt"><textarea></textarea></div>' : ""}`
    + "</gradio-app></body></html>",
    { url: "http://127.0.0.1:7860/", runScripts: "outside-only", virtualConsole },
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
  window.fetch = async (url, init) => {
    fetches.push({ url, init });
    const answer = answers[Math.min(fetches.length - 1, answers.length - 1)];
    if (typeof answer === "function") return answer();
    return answer;
  };
  // Forge script.js: one MutationObserver on gradioApp() runs the onUiUpdate callbacks
  const uiUpdates = [];
  new window.MutationObserver((mutations) => uiUpdates.push(mutations)).observe(window.gradioApp(), { childList: true, subtree: true });
  if (before) window.eval(before);
  window.eval(LOADER);
  const scripts = () => [...doc.head.querySelectorAll("script")];
  return { window, doc, logs, fetches, uiUpdates, scripts, state: () => doc.documentElement.getAttribute("data-sam3-tagcomplete") };
}

async function until(check, label, ms = 3000) {
  const start = Date.now();
  while (!check()) {
    if (Date.now() - start > ms) assert.fail(`timed out waiting for ${label}`);
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

function fire(window, script, type = "load") {
  script.dispatchEvent(new window.Event(type));
}

test("adds the 15 upstream files in the boot route's order, then lets onUiUpdate run once more", async (t) => {
  const env = load(t);
  await until(() => env.scripts().length === 15, "15 scripts");
  assert.equal(env.fetches.length, 1);
  assert.equal(env.fetches[0].url, "sam3-tagcomplete/boot", "relative, like tagcomplete's own tacapi/... calls");
  assert.equal(env.fetches[0].init.headers["X-SAM3-Notebook"], "1");
  assert.equal(env.fetches[0].init.cache, "no-store");
  assert.equal(env.fetches[0].init.credentials, "same-origin");
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
  assert.deepEqual(env.logs.error, []);
  assert.deepEqual(env.logs.warn, []);
});

test("no nudge before the UI is built (later mutations run onUiUpdate anyway)", async (t) => {
  const env = load(t, { prompt: false });
  await until(() => env.scripts().length === 15, "15 scripts");
  env.scripts().forEach((script) => fire(env.window, script));
  await until(() => env.state() === "started", "started");
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(env.uiUpdates.length, 0);
});

for (const [label, before] of [
  ["TAC_CFG", "var TAC_CFG = null;"],
  ["addAutocompleteToArea", "function addAutocompleteToArea(area) {}"],
]) {
  test(`another tagcomplete on the page (${label}): nothing is fetched or added`, async (t) => {
    const env = load(t, { before });
    await until(() => env.state() === "skipped", "skipped");
    assert.equal(env.fetches.length, 0);
    assert.equal(env.scripts().length, 0);
    assert.deepEqual(env.logs.log, ["[sam-extra] Tag Autocomplete: another tagcomplete is loaded on this page; built-in copy not started"]);
  });
}

test("the server says it is off: the reason is logged once, nothing is added", async (t) => {
  const reason = "another tagcomplete extension is enabled (sd-webui-tagcomplete-neo: C:\\forge\\extensions\\sd-webui-tagcomplete-neo)";
  const env = load(t, { answers: [json(200, { enabled: false, reason, upstream: "x", scripts: [] })] });
  await until(() => env.state() === "skipped", "skipped");
  assert.equal(env.scripts().length, 0);
  assert.deepEqual(env.logs.log, [`[sam-extra] Tag Autocomplete: built-in copy not started: ${reason}`]);
});

test("404 while the route is not registered yet: asks again", async (t) => {
  const env = load(t, { answers: [json(404, null), json(404, null), json(200, ENABLED)], fastTimers: true });
  await until(() => env.scripts().length === 15, "15 scripts");
  assert.equal(env.fetches.length, 3);
});

test("network errors: gives up after 5 more tries with a warning", async (t) => {
  const env = load(t, { answers: [() => { throw new TypeError("Failed to fetch"); }], fastTimers: true });
  await until(() => env.state() === "failed", "failed");
  assert.equal(env.fetches.length, 6);
  assert.equal(env.scripts().length, 0);
  assert.equal(env.logs.warn.length, 1);
  assert.match(env.logs.warn[0], /no answer from sam3-tagcomplete\/boot/);
});

test("a refused request (login, missing header) is not retried", async (t) => {
  const env = load(t, { answers: [json(403, null)] });
  await until(() => env.state() === "failed", "failed");
  assert.equal(env.fetches.length, 1);
  assert.match(env.logs.warn[0], /HTTP 403/);
});

test("a file that fails to load is reported; the rest still run", async (t) => {
  const env = load(t);
  await until(() => env.scripts().length === 15, "15 scripts");
  env.scripts().forEach((script, i) => fire(env.window, script, i === 3 ? "error" : "load"));
  await until(() => env.state() === "failed", "failed");
  assert.equal(env.logs.error.length, 1);
  assert.match(env.logs.error[0], /could not load file=.*_result\.js/);
});

test("the loader runs once per page", async (t) => {
  const env = load(t);
  env.window.eval(LOADER);
  await until(() => env.scripts().length === 15, "15 scripts");
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(env.fetches.length, 1);
  assert.equal(env.scripts().length, 15);
});

test("defines no global names", async (t) => {
  // (no element ids on this page: jsdom exposes them as window properties too)
  const env = load(t, { answers: [json(200, { enabled: false, reason: "off", scripts: [] })], prompt: false });
  await until(() => env.state() === "skipped", "skipped");
  const fresh = new JSDOM("<!doctype html><html></html>", { runScripts: "outside-only" });
  t.after(() => fresh.window.close());
  const baseline = new Set(Object.getOwnPropertyNames(fresh.window));
  const added = Object.getOwnPropertyNames(env.window).filter((name) => !baseline.has(name));
  assert.deepEqual(added.sort(), ["fetch", "gradioApp"], "only the test's own stand-ins");
});
