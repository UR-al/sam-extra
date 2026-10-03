// progress_bar.js — SAM Extra 진행 막대(Settings → SAM Extra Progress Bar).
//
// Forge 의 requestProgress 는 가짜다. javascript/progressbar.js:77 의 계약을 그대로 흉내 낸다: progressbarContainer 앞에
// .progressDiv 를 넣고(인라인 display:block), 응답마다 onProgress(res), 끝나면 그 div 를 지우고 atEnd() 를 부른다.
// 서버는 sam3ext/progress_api.py 의 GET /sam-extra/progress 계약을 흉내 낸 가짜 fetch 다. 시계(setTimeout·
// requestAnimationFrame·performance.now)는 테스트가 돌린다 — 실제 시간을 기다리지 않는다. 창은 t.after 로 닫는다.
// 맨 끝 테스트 하나는 확장 옆 Forge 의 진짜 javascript/progressbar.js 를 읽어 돌린다(없으면 건너뜀 — CI·작업 트리).
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM, VirtualConsole } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const SCRIPT = readFileSync(path.join(ROOT, "javascript", "progress_bar.js"), "utf8");
const CSS = readFileSync(path.join(ROOT, "style.css"), "utf8").replace(/\r\n/g, "\n");
const BASE = "http://127.0.0.1:7860/";
// style.css 가 Forge 기본 막대(.progressDiv)를 숨기는 규칙의 선택자
const HIDE = (() => {
  const block = CSS.slice(CSS.indexOf("/* sam3-progress:begin */"), CSS.indexOf("/* sam3-progress:end */"));
  const match = /(html\[data-sam3-progress="on"\][^{]+)\{\s*display: none !important;\s*\}/.exec(block);
  return match ? match[1].trim() : "no-hide-rule-found";
})();
// <Forge>/extensions/forge_sam3_extension/tests/js/this file → <Forge>/javascript/progressbar.js
const FORGE_PROGRESSBAR = path.resolve(ROOT, "..", "..", "javascript", "progressbar.js");

const TAB = (tab) => `
  <div id="tab_${tab}" class="tabitem">
    <div id="${tab}_generate_box">
      <button id="${tab}_interrupt"><span>Interrupt</span></button>
      <button id="${tab}_interrupting">Interrupting...</button>
      <button id="${tab}_skip">Skip</button>
    </div>
    <div id="${tab}_results" class="gradio-column">
      <div id="${tab}_results_panel" class="gradio-column">
        <div id="${tab}_gallery_container" class="gradio-group"><div id="${tab}_gallery"></div></div>
      </div>
    </div>
  </div>`;
const EXTRAS = `
  <div id="tab_extras" class="tabitem">
    <div id="extras_results"><div id="extras_results_panel">
      <div id="extras_gallery_container"><div id="extras_gallery"></div></div>
    </div></div>
  </div>`;

// ---- 시계 -------------------------------------------------------------------------

function installClock(window) {
  let now = 1000;
  let seq = 0;
  const timers = new Map();
  const frames = new Map();
  window.setTimeout = (fn, ms = 0) => {
    seq += 1;
    timers.set(seq, { at: now + Math.max(0, Number(ms) || 0), fn });
    return seq;
  };
  window.clearTimeout = (id) => { timers.delete(id); };
  window.requestAnimationFrame = (fn) => {
    seq += 1;
    frames.set(seq, fn);
    return seq;
  };
  window.cancelAnimationFrame = (id) => { frames.delete(id); };
  Object.defineProperty(window.performance, "now", { configurable: true, value: () => now });
  // 대기 중인 마이크로태스크(가짜 fetch 의 promise 사슬)가 다 돌 때까지.
  const flush = () => new Promise((resolve) => setImmediate(resolve));
  async function runDue(limit) {
    for (;;) {
      let next = null;
      for (const [id, timer] of timers) {
        if (timer.at <= limit && (!next || timer.at < next[1].at)) next = [id, timer];
      }
      if (!next) return;
      timers.delete(next[0]);
      now = Math.max(now, next[1].at);
      next[1].fn();
      await flush();
    }
  }
  return {
    get now() { return now; },
    timers,
    frames,
    flush,
    // 시간을 16ms 프레임 단위로 앞으로 돌린다: 그사이 만기 타이머(순서대로), 프레임마다 rAF 콜백.
    async advance(ms) {
      const end = now + ms;
      await flush();
      await runDue(now);
      while (now < end) {
        const target = Math.min(end, now + 16);
        await runDue(target);
        now = target;
        if (frames.size) {
          const due = [...frames.values()];
          frames.clear();
          due.forEach((fn) => fn(now));
          await flush();
        }
      }
    },
  };
}

// ---- 가짜 서버 (sam3ext/progress_api.py 의 응답 모양) ------------------------------------

const JOB_FIELDS = ["time_start", "elapsed", "job_no", "job_count", "passes_per_image", "step", "steps", "progress",
  "pass_progress", "pass_eta", "eta", "interrupted", "skipped", "stopping", "textinfo"];

function snapshot(fields = {}) {
  const body = {
    version: 1, id_task: null, busy: false, active: false, queued: false, completed: false,
    queue_position: null, queue_size: 0, server_time: 0,
  };
  JOB_FIELDS.forEach((field) => { body[field] = null; });
  if (fields.active) {
    Object.assign(body, {
      busy: true, time_start: 1, elapsed: 0, job_no: 0, job_count: 1, passes_per_image: 1, step: 0, steps: 20,
      progress: 0, pass_progress: 0, pass_eta: null, eta: null, interrupted: false, skipped: false,
      stopping: false, textinfo: null,
    });
  }
  return Object.assign(body, fields);
}

function fakeServer() {
  const server = { calls: [], answers: new Map(), status: 200, gate: null, inFlight: 0, maxInFlight: 0 };
  const respond = (status, body) => ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => JSON.parse(JSON.stringify(body)),
  });
  // value: 응답 필드, 또는 부를 때마다 필드를 돌려주는 함수(시계에 따라 바뀌는 ETA 등)
  server.set = (id, value) => server.answers.set(id, value);
  server.fetch = async (url, init = {}) => {
    server.calls.push({ url, init });
    server.inFlight += 1;
    server.maxInFlight = Math.max(server.maxInFlight, server.inFlight);
    try {
      if (server.gate) await server.gate;
      const parsed = new URL(url, BASE);
      if (server.status === 404 || parsed.pathname !== "/sam-extra/progress") return respond(404, { detail: "Not Found" });
      if (!init.headers || init.headers["X-SAM3-Notebook"] !== "1") return respond(403, { detail: "header" });
      if (server.status !== 200) return respond(server.status, { detail: "error" });
      const id = parsed.searchParams.get("id_task");
      const answer = server.answers.get(id);
      const fields = typeof answer === "function" ? answer() : (answer || {});
      return respond(200, snapshot({ id_task: id, ...fields }));
    } finally {
      server.inFlight -= 1;
    }
  };
  server.polled = (id) => server.calls.filter((call) => new URL(call.url, BASE).searchParams.get("id_task") === id).length;
  return server;
}

// ---- 가짜 Forge requestProgress ----------------------------------------------------------

function installForge(window) {
  const forge = { calls: [] };
  forge.original = function requestProgress(id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout = 40) {
    const record = { args: [...arguments], length: arguments.length, id: id_task, timeout: inactivityTimeout, self: this };
    const div = window.document.createElement("div");
    div.className = "progressDiv";
    div.style.display = "block";                          // Forge: opts.show_progressbar ? "block" : "none"
    const inner = window.document.createElement("div");
    inner.className = "progress";
    div.appendChild(inner);
    progressbarContainer.parentNode.insertBefore(div, progressbarContainer);
    record.div = div;
    record.progress = (res) => (onProgress ? onProgress(res) : undefined);
    record.end = () => {                                   // removeProgressBar
      if (div.parentNode) div.parentNode.removeChild(div);
      return atEnd();
    };
    forge.calls.push(record);
  };
  window.requestProgress = forge.original;
  return forge;
}

// ---- 페이지 ------------------------------------------------------------------------------

// Forge 의 request()(progressbar.js:7)가 쓰는 XHR 대역. handler({ url, body }) 의 반환값이 응답 JSON 이다.
function installXhr(window, handler) {
  const xhr = { calls: [] };
  window.XMLHttpRequest = class {
    open(method, url) {
      this.method = method;
      this.url = url;
    }

    setRequestHeader() {}

    send(body) {
      const call = { method: this.method, url: this.url, body: JSON.parse(body) };
      xhr.calls.push(call);
      window.setTimeout(() => {
        this.readyState = 4;
        this.status = 200;
        this.responseText = JSON.stringify(handler(call));
        this.onreadystatechange();
      }, 5);
    }
  };
  return xhr;
}

function buildDom(t, { options = {}, upstream = false, reduced = false, head = "", forgeSource = null, xhr = null } = {}) {
  // 페이지 콘솔은 모은다: error 가 하나라도 찍히면(감싼 콜백 안의 예외 등) 그 테스트는 실패한다.
  const logs = { error: [], warn: [], info: [] };
  const virtualConsole = new VirtualConsole();
  ["error", "warn", "info"].forEach((level) => virtualConsole.on(level, (...args) => logs[level].push(args.join(" "))));
  virtualConsole.on("jsdomError", (error) => logs.error.push(String(error && error.stack || error)));
  const dom = new JSDOM(`<!doctype html><html><head>${upstream ? '<style id="spb-dynamic-css"></style>' : ""}${head}</head>
    <body>${TAB("txt2img")}${TAB("img2img")}${EXTRAS}</body></html>`, {
    runScripts: "outside-only", pretendToBeVisual: true, url: BASE, virtualConsole,
  });
  t.after(() => {
    dom.window.close();
    assert.deepEqual(logs.error, [], "no errors on the page console");
  });
  const { window } = dom;
  const clock = installClock(window);
  const server = fakeServer();
  window.fetch = server.fetch;
  window.gradioApp = () => window.document;
  const callbacks = { uiLoaded: [], optionsAvailable: [], optionsChanged: [] };
  window.onUiLoaded = (fn) => callbacks.uiLoaded.push(fn);
  window.onOptionsAvailable = (fn) => callbacks.optionsAvailable.push(fn);
  window.onOptionsChanged = (fn) => callbacks.optionsChanged.push(fn);
  window.opts = { sam3_progress_enabled: true, ...options };
  window.matchMedia = (query) => ({
    matches: reduced && query.includes("prefers-reduced-motion: reduce"), media: query,
    addEventListener() {}, removeEventListener() {},
  });
  let forge;
  if (forgeSource) {
    // Forge 가 하는 순서: Forge 의 javascript/*.js 가 먼저, 확장 스크립트가 나중(modules/ui_gradio_extensions.py).
    installXhr(window, xhr);
    window.eval(forgeSource);
    forge = { original: window.requestProgress, calls: [] };
  } else {
    forge = installForge(window);
  }
  window.__sam3ProgressTestHooks = {};
  window.eval(SCRIPT);
  const doc = window.document;
  const env = {
    dom, window, doc, clock, forge, server, callbacks, logs, hooks: window.__sam3ProgressTestHooks,
    bar: (tab) => doc.getElementById(`sam3_progress_${tab}`),
    text: (tab) => doc.getElementById(`sam3_progress_${tab}`).querySelector(".sam3-progress-label").firstChild.data,
    texts: (tab) => [...doc.getElementById(`sam3_progress_${tab}`).querySelectorAll(".sam3-progress-label")]
      .map((label) => label.firstChild.data),
    value: (tab) => parseFloat(doc.getElementById(`sam3_progress_${tab}`).style.getPropertyValue("--sam3-progress-value")),
    state: (tab) => doc.getElementById(`sam3_progress_${tab}`).getAttribute("data-state"),
    shown: (tab) => {
      const bar = doc.getElementById(`sam3_progress_${tab}`);
      return { fill: bar.getAttribute("data-fill"), text: bar.getAttribute("data-text") };
    },
    container: (tab) => doc.getElementById(`${tab}_gallery_container`),
    gallery: (tab) => doc.getElementById(`${tab}_gallery`),
    // Forge 의 submit(): requestProgress(id, gallery_container, gallery, atEnd) — 인자 네 개
    submit(tab, id, atEnd = () => {}) {
      window.requestProgress(id, env.container(tab), env.gallery(tab), atEnd);
      return forge.calls[forge.calls.length - 1];
    },
    setOptions(changes) {
      Object.assign(window.opts, changes);
      callbacks.optionsChanged.forEach((fn) => fn());
    },
    click(id) {
      doc.getElementById(id).firstChild.dispatchEvent(new window.MouseEvent("click", { bubbles: true, composed: true }));
    },
  };
  return env;
}

// Forge 가 페이지를 띄우는 순서: 옵션이 오고(load_webui_settings) → onUiLoaded 콜백들 → (그 뒤) 이 막대가 붙는다.
async function start(env) {
  env.callbacks.optionsAvailable.forEach((fn) => fn());
  env.callbacks.optionsChanged.forEach((fn) => fn());
  env.callbacks.uiLoaded.forEach((fn) => fn());
  await env.clock.advance(0);
}

async function started(t, options) {
  const env = buildDom(t, options);
  await start(env);
  return env;
}

// ---- 꺼져 있을 때 ---------------------------------------------------------------------------

test("disabled: no DOM, no fetches, Forge's requestProgress untouched", async (t) => {
  const env = await started(t, { options: { sam3_progress_enabled: false } });
  assert.equal(env.window.requestProgress, env.forge.original);
  assert.equal(env.doc.querySelector(".sam3-progress"), null);
  assert.equal(env.doc.documentElement.hasAttribute("data-sam3-progress"), false);
  const atEnd = () => "ended";
  const record = env.submit("txt2img", "task(off)", atEnd);
  assert.equal(record.args[3], atEnd, "callbacks reach Forge as they were");
  assert.equal(record.length, 4);
  env.setOptions({ sam3_progress_text_format: "pct_only", sam3_progress_height: 30 });
  await env.clock.advance(3000);
  assert.equal(env.server.calls.length, 0);
  assert.equal(env.doc.querySelector(".sam3-progress"), null);
  assert.equal(record.end(), "ended");
});

test("Settings enable it live and disable it again without leftovers", async (t) => {
  const env = await started(t, { options: { sam3_progress_enabled: false } });
  env.setOptions({ sam3_progress_enabled: true });
  assert.ok(env.bar("txt2img") && env.bar("img2img"));
  assert.equal(env.doc.documentElement.getAttribute("data-sam3-progress"), "on");
  assert.notEqual(env.window.requestProgress, env.forge.original);
  const ended = [];
  const record = env.submit("txt2img", "task(a)", () => ended.push("a"));
  env.server.set("task(a)", { active: true, progress: 0.2, step: 4 });
  await env.clock.advance(500);
  const polls = env.server.calls.length;
  assert.ok(polls > 0);
  env.setOptions({ sam3_progress_enabled: false });
  assert.equal(env.doc.querySelector(".sam3-progress"), null);
  assert.equal(env.doc.documentElement.hasAttribute("data-sam3-progress"), false);
  await env.clock.advance(3000);
  assert.equal(env.server.calls.length, polls, "no polling once disabled");
  record.end();
  assert.deepEqual(ended, ["a"], "Forge's own atEnd still runs");
  const after = env.submit("txt2img", "task(b)");
  await env.clock.advance(1000);
  assert.equal(env.server.polled("task(b)"), 0);
  assert.equal(typeof after.args[3], "function");
});

// ---- 감싸기 -------------------------------------------------------------------------------

test("the wrapper passes all six arguments through and still calls Forge's atEnd and onProgress", async (t) => {
  const env = await started(t);
  const seen = [];
  const atEnd = function () { seen.push(["end", [...arguments]]); return "end-result"; };
  const onProgress = function (res) { seen.push(["progress", res]); return "progress-result"; };
  const container = env.container("txt2img");
  const gallery = env.gallery("txt2img");
  env.window.requestProgress("task(six)", container, gallery, atEnd, onProgress, 7);
  const record = env.forge.calls[0];
  assert.equal(record.length, 6);
  assert.equal(record.args[0], "task(six)");
  assert.equal(record.args[1], container);
  assert.equal(record.args[2], gallery);
  assert.equal(record.args[5], 7);
  assert.equal(record.timeout, 7);
  assert.equal(typeof record.args[3], "function");
  assert.equal(typeof record.args[4], "function");
  const res = { active: false, queued: true, textinfo: "In queue: 1/2" };
  assert.equal(record.args[4](res), "progress-result");
  assert.equal(seen[0][1], res, "the same response object reaches the original onProgress");
  assert.equal(record.args[3](), "end-result");
  assert.deepEqual(seen[1], ["end", []]);

  // submit(): four arguments — Forge's own inactivityTimeout default (40) still applies.
  const four = env.submit("img2img", "task(four)", () => "x");
  assert.equal(four.length, 5);
  assert.equal(four.args[5], undefined);
  assert.equal(four.timeout, 40);
  assert.equal(four.args[4]({ active: false }), undefined, "no original onProgress: nothing to call");

  // restoreProgressTxt2img(): (…, atEnd, null, 0)
  env.window.requestProgress("task(restore)", container, gallery, () => "restored", null, 0);
  const restore = env.forge.calls[2];
  assert.equal(restore.length, 6);
  assert.equal(restore.timeout, 0);
  assert.equal(restore.args[4]({ active: true }), undefined);
  assert.equal(restore.args[3](), "restored");
  await env.clock.advance(100);                                // let the last confirmation poll settle
});

test("other callers (extras, model merger, extensions) pass through untouched", async (t) => {
  const env = await started(t);
  const atEnd = () => {};
  const onProgress = () => {};
  env.window.requestProgress("task(x)", env.doc.getElementById("extras_gallery_container"),
    env.doc.getElementById("extras_gallery"), atEnd, onProgress);
  const record = env.forge.calls[0];
  assert.equal(record.args[3], atEnd);
  assert.equal(record.args[4], onProgress);
  assert.equal(env.doc.querySelector("#tab_extras .sam3-progress"), null);
  await env.clock.advance(1000);
  assert.equal(env.server.calls.length, 0);
});

// ---- 탭마다 제 작업만 ---------------------------------------------------------------------

test("each tab shows only the job it started, and only between start and end", async (t) => {
  const env = await started(t);
  assert.equal(env.state("txt2img"), "idle");
  assert.equal(env.text("txt2img"), "");
  await env.clock.advance(2000);
  assert.equal(env.server.calls.length, 0, "no polling before a job starts (upstream polled every 100 ms forever)");

  env.server.set("task(t2i)", { active: true, job_count: 1, step: 5, steps: 20, progress: 0.25, pass_eta: 15, eta: 15 });
  env.server.set("task(i2i)", { queued: true, queue_position: 1, queue_size: 1 });
  const t2i = env.submit("txt2img", "task(t2i)");
  env.submit("img2img", "task(i2i)");
  await env.clock.advance(300);

  assert.equal(env.state("txt2img"), "running");
  assert.equal(env.state("img2img"), "waiting");
  assert.equal(env.text("img2img"), "In queue: 1/1");
  assert.match(env.text("txt2img"), /^5\/20 • \d+% • 1\ds$/);
  const urls = env.server.calls.map((call) => new URL(call.url, BASE));
  assert.ok(urls.every((url) => url.pathname === "/sam-extra/progress"));
  assert.deepEqual([...new Set(urls.map((url) => url.searchParams.get("id_task")))].sort(), ["task(i2i)", "task(t2i)"]);
  const headers = env.server.calls.map((call) => call.init);
  assert.ok(headers.every((init) => init.headers["X-SAM3-Notebook"] === "1" && init.credentials === "same-origin"
    && init.cache === "no-store"));

  env.server.set("task(t2i)", { completed: true });
  await env.clock.advance(300);
  t2i.end();
  const polls = env.server.polled("task(t2i)");
  await env.clock.advance(3000);
  assert.equal(env.server.polled("task(t2i)"), polls, "no polling after the job ended");
  assert.equal(env.state("img2img"), "waiting", "the other tab is unaffected");
});

test("while waiting it shows Forge's own queue text from onProgress", async (t) => {
  const env = await started(t);
  env.server.gate = new Promise(() => {});             // our route never answers
  const record = env.submit("txt2img", "task(q)");
  await env.clock.advance(50);
  record.progress({ active: false, queued: true, completed: false, textinfo: "In queue: 2/3" });
  assert.equal(env.text("txt2img"), "In queue: 2/3");
  record.progress({ active: false, queued: false, completed: false, textinfo: "Waiting..." });
  assert.equal(env.text("txt2img"), "Waiting...");
  env.setOptions({ sam3_progress_text_format: "none" });
  assert.equal(env.text("txt2img"), "");
});

test("whole-job progress: job counter, steps, percent and a counting-down ETA", async (t) => {
  const env = await started(t, { options: { sam3_progress_smoothness: "smooth_lt_acc" } });
  const t0 = env.clock.now;
  env.server.set("task(b)", () => ({
    active: true, job_no: 1, job_count: 4, step: 5, steps: 20, progress: 0.3125,
    pass_eta: 15, eta: 45 - (env.clock.now - t0) / 1000,
  }));
  env.submit("txt2img", "task(b)");
  await env.clock.advance(1500);
  const [step, pct, eta] = env.text("txt2img").split(" • ");
  assert.equal(step, "[2/4] 5/20");
  assert.ok(Number.parseInt(pct, 10) >= 31 && Number.parseInt(pct, 10) <= 33, pct);
  assert.equal(eta, "44s");
  await env.clock.advance(2000);
  assert.equal(env.text("txt2img").split(" • ")[2], "42s");
  assert.ok(env.value("txt2img") > 30 && env.value("txt2img") < 34);
  assert.equal(env.bar("txt2img").getAttribute("aria-valuenow"), String(Math.round(env.value("txt2img"))));
  assert.equal(env.bar("txt2img").getAttribute("aria-valuetext"), env.text("txt2img"));
  assert.deepEqual(new Set(env.texts("txt2img")).size, 1, "both label layers carry the same text");
});

test("stale step counts from the previous job are not shown before the first step", async (t) => {
  const env = await started(t);
  env.server.set("task(s)", { active: true, step: 0, steps: 28, progress: 0 });
  env.submit("txt2img", "task(s)");
  await env.clock.advance(300);
  assert.equal(env.text("txt2img"), "0% • ?");
  env.server.set("task(s)", { active: true, step: 1, steps: 20, progress: 0.05, eta: 19, pass_eta: 19 });
  await env.clock.advance(300);
  assert.match(env.text("txt2img"), /^1\/20 • \d% • 19s$/);
});

test("a job that ends before any look saw its first step does not finish with the previous job's step count", async (t) => {
  const env = await started(t);
  // Our only look at it was during setup: State.begin keeps the last job's sampling_steps (28).
  env.server.set("task(quick)", { active: true, step: 0, steps: 28, progress: 0 });
  const quick = env.submit("txt2img", "task(quick)");
  await env.clock.advance(150);
  assert.equal(env.text("txt2img"), "0% • ?");
  env.server.set("task(quick)", { completed: true });            // a few fast steps, done before the next look
  await env.clock.advance(250);
  assert.equal(env.state("txt2img"), "done");
  assert.equal(env.text("txt2img"), "100%", "no stale 28/28");
  quick.end();
  // Once a look saw a step, the job ends with its own count.
  env.server.set("task(seen)", { active: true, step: 1, steps: 4, progress: 0.25, eta: 3, pass_eta: 3 });
  const seen = env.submit("txt2img", "task(seen)");
  await env.clock.advance(250);
  env.server.set("task(seen)", { completed: true });
  await env.clock.advance(250);
  assert.equal(env.text("txt2img"), "4/4 • 100%");
  seen.end();
});

test("after the last pass Forge's nextjob zeroes the step: decoding and saving show the finished pass, not 0/20", async (t) => {
  // 실측: 마지막 스텝 뒤 디코드·저장 동안의 응답이 job_no 1/1, step 0/20 — 글자가 '0/20 • 99% • ?' 였다.
  const env = await started(t);
  env.server.set("task(z)", { active: true, job_no: 0, job_count: 1, step: 19, steps: 20, progress: 0.95, eta: 1,
    pass_eta: 1 });
  const single = env.submit("txt2img", "task(z)");
  await env.clock.advance(500);
  assert.match(env.text("txt2img"), /^19\/20 • /);
  env.server.set("task(z)", { active: true, job_no: 1, job_count: 1, step: 0, steps: 20, progress: 1 });
  await env.clock.advance(500);
  assert.match(env.text("txt2img"), /^20\/20 • \d+% • \?$/);
  env.server.set("task(z)", { completed: true });
  await env.clock.advance(300);
  single.end();
  // Between images the step really starts again at 0, of the next pass; after the last one it is finished.
  env.server.set("task(y)", { active: true, job_no: 1, job_count: 2, step: 0, steps: 20, progress: 0.5, eta: 9,
    pass_eta: 4 });
  env.submit("txt2img", "task(y)");
  await env.clock.advance(500);
  assert.match(env.text("txt2img"), /^\[2\/2\] 0\/20 • /);
  env.server.set("task(y)", { active: true, job_no: 2, job_count: 2, step: 0, steps: 20, progress: 1 });
  await env.clock.advance(500);
  assert.match(env.text("txt2img"), /^\[2\/2\] 20\/20 • /);
});

test("a job joined mid-run (Forge's Restore progress after a reload) starts where the job is", async (t) => {
  // 실측: 32/80 스텝(40%)에서 붙자 Smooth > Accurate 가 0% 에서 출발해 '32/80 • 1%', 실제 56% 일 때 '19%' 였다.
  const env = await started(t);
  const t0 = env.clock.now;
  const elapsed = () => (env.clock.now - t0) / 1000;
  env.server.set("task(r)", () => ({ active: true, step: 32 + Math.floor(elapsed() * 2.4), steps: 80,
    progress: Math.min(0.99, 0.4 + elapsed() * 0.03), eta: Math.max(0, 20 - elapsed()),
    pass_eta: Math.max(0, 20 - elapsed()) }));
  // ui.js restoreProgressTxt2img: requestProgress(id, gallery_container, gallery, atEnd, null, 0)
  env.window.requestProgress("task(r)", env.container("txt2img"), env.gallery("txt2img"), () => {}, null, 0);
  await env.clock.advance(500);
  assert.ok(env.value("txt2img") >= 40, `starts at the job's 40%: ${env.value("txt2img")}`);
  assert.match(env.text("txt2img"), /^3[23]\/80 • 4[0-2]% • /);
  await env.clock.advance(5000);
  const value = env.value("txt2img");
  assert.ok(Math.abs(value - 56.5) < 4, `keeps pace with the job (~56%): ${value}`);
});

// ---- 끝 ---------------------------------------------------------------------------------

test("a finished job fills to 100%, then fades bar and text (default)", async (t) => {
  const env = await started(t);
  env.server.set("task(d)", { active: true, step: 10, steps: 20, progress: 0.5, eta: 8, pass_eta: 8 });
  const record = env.submit("txt2img", "task(d)");
  await env.clock.advance(1000);
  assert.ok(env.value("txt2img") < 99.2);
  env.server.set("task(d)", { completed: true });
  await env.clock.advance(250);
  assert.equal(env.state("txt2img"), "done");
  assert.equal(env.text("txt2img"), "20/20 • 100%");
  await env.clock.advance(400);
  assert.equal(env.value("txt2img"), 100);
  assert.deepEqual(env.shown("txt2img"), { fill: "1", text: "1" });
  record.end();                                            // Forge's own end comes later: no effect
  await env.clock.advance(800);
  assert.deepEqual(env.shown("txt2img"), { fill: "0", text: "0" });
  await env.clock.advance(500);
  assert.equal(env.state("txt2img"), "idle");
  assert.equal(env.value("txt2img"), 0);
  assert.equal(env.text("txt2img"), "");
});

test("after-finish: keep everything, or fade only the text; both start full", async (t) => {
  for (const [mode, expected] of [["keep", { fill: "1", text: "1" }], ["fade_text_only", { fill: "1", text: "0" }]]) {
    const env = await started(t, { options: { sam3_progress_after_finish: mode } });
    assert.equal(env.value("txt2img"), 100, `${mode}: full bar before the first job (upstream 7fe5810)`);
    assert.deepEqual(env.shown("txt2img"), { fill: "1", text: "0" });
    env.server.set("task(k)", { active: true, step: 3, steps: 6, progress: 0.5, eta: 3, pass_eta: 3 });
    env.submit("txt2img", "task(k)");
    await env.clock.advance(200);
    assert.ok(env.value("txt2img") < 100, `${mode}: a new job starts from 0`);
    env.server.set("task(k)", { completed: true });
    await env.clock.advance(5000);
    assert.equal(env.state("txt2img"), "done", mode);
    assert.equal(env.value("txt2img"), 100, mode);
    assert.deepEqual(env.shown("txt2img"), expected, mode);
    assert.equal(env.text("txt2img"), "6/6 • 100%", mode);
  }
});

test("an interrupted job stops where it was the moment the flag is seen, and says so", async (t) => {
  const env = await started(t, { options: { sam3_progress_after_finish: "keep", sam3_progress_smoothness: "smooth_lt_acc" } });
  env.server.set("task(i)", { active: true, step: 8, steps: 20, progress: 0.4, eta: 6, pass_eta: 6 });
  const record = env.submit("txt2img", "task(i)");
  await env.clock.advance(1500);
  env.server.set("task(i)", { active: true, step: 8, steps: 20, progress: 0.4, eta: 6, pass_eta: 6, interrupted: true });
  await env.clock.advance(250);
  assert.equal(env.state("txt2img"), "interrupted", "no waiting for the job to wind down (upstream)");
  const stopped = env.value("txt2img");
  assert.ok(stopped > 35 && stopped < 45, `near 40%: ${stopped}`);
  const polls = env.server.calls.length;
  env.server.set("task(i)", { completed: true });              // decode and save, then the end
  await env.clock.advance(250);
  record.end();
  await env.clock.advance(2000);
  assert.equal(env.server.calls.length, polls, "no polling after the interruption");
  assert.equal(env.state("txt2img"), "interrupted");
  assert.equal(env.value("txt2img"), stopped);
  assert.match(env.text("txt2img"), /^8\/20 • 4\d% • 중단됨$/);
  assert.equal(env.bar("txt2img").getAttribute("data-interrupt"), "red_text_red_bar");
  assert.deepEqual(env.shown("txt2img"), { fill: "1", text: "1" });
});

test("an interruption fades like upstream: 400 ms, then the after-finish mode", async (t) => {
  const env = await started(t);
  env.server.set("task(j)", { active: true, step: 2, steps: 20, progress: 0.1, interrupted: true });
  const record = env.submit("txt2img", "task(j)");
  await env.clock.advance(100);
  assert.equal(env.state("txt2img"), "interrupted");
  assert.deepEqual(env.shown("txt2img"), { fill: "1", text: "1" });
  record.end();                                                // the job wound down at once: Forge let go of it
  await env.clock.advance(450);
  assert.deepEqual(env.shown("txt2img"), { fill: "0", text: "0" });
  await env.clock.advance(500);
  assert.equal(env.state("txt2img"), "idle");
});

test("an interrupted job keeps its bar until Forge lets go of it, so Forge's bar never shows through", async (t) => {
  const env = await started(t);                                // after-finish: fade (default)
  env.server.set("task(w)", { active: true, step: 5, steps: 20, progress: 0.25, eta: 15, pass_eta: 15 });
  const record = env.submit("txt2img", "task(w)");
  await env.clock.advance(400);
  env.click("txt2img_interrupt");
  assert.equal(env.state("txt2img"), "interrupted");
  await env.clock.advance(3000);                               // Forge winds it down: next step, decode, save
  assert.equal(env.state("txt2img"), "interrupted", "still drawn while Forge holds the job");
  assert.deepEqual(env.shown("txt2img"), { fill: "1", text: "1" });
  assert.equal(record.div.matches(HIDE), true, "Forge's bar stays hidden behind ours");
  record.end();                                                // Forge's requestProgress is over
  await env.clock.advance(50);
  assert.deepEqual(env.shown("txt2img"), { fill: "0", text: "0" });
  await env.clock.advance(500);
  assert.equal(env.state("txt2img"), "idle");
});

test("an Interrupt before the job started is not an interruption — Forge's State.begin discards it", async (t) => {
  const env = await started(t);
  const record = env.submit("txt2img", "task(e)");             // not on the server yet: Waiting...
  env.click("txt2img_interrupt");                              // before our first look came back
  await env.clock.advance(300);
  assert.equal(env.text("txt2img"), "Waiting...");
  env.click("txt2img_interrupt");                              // still waiting, not queued
  env.server.set("task(e)", { active: true, step: 2, steps: 4, progress: 0.5, eta: 1, pass_eta: 1 });
  await env.clock.advance(300);
  assert.equal(env.state("txt2img"), "running");
  env.server.set("task(e)", { completed: true });
  record.end();
  await env.clock.advance(100);
  assert.equal(env.state("txt2img"), "done");
  assert.equal(env.text("txt2img"), "4/4 • 100%");
  // One that reached the job after it started is Forge's own flag, and still counts.
  const late = env.submit("txt2img", "task(l)");
  await env.clock.advance(300);
  env.click("txt2img_interrupt");
  env.server.set("task(l)", { active: true, step: 1, steps: 4, progress: 0.25, interrupted: true });
  await env.clock.advance(300);
  assert.equal(env.state("txt2img"), "interrupted");
  late.end();
});

test("Interrupt (button or Esc) marks the job even if no poll saw the flag", async (t) => {
  const env = await started(t);
  env.server.set("task(c)", { active: true, step: 3, steps: 20, progress: 0.15, eta: 9, pass_eta: 9 });
  const record = env.submit("txt2img", "task(c)");
  await env.clock.advance(500);
  env.click("txt2img_interrupt");
  env.server.set("task(c)", { completed: true });            // ended before the next poll
  record.end();
  await env.clock.advance(100);
  assert.equal(env.state("txt2img"), "interrupted");
  assert.match(env.text("txt2img"), /중단됨$/);
  // The other tab's button does not touch this tab's job.
  env.server.set("task(o)", { active: true, step: 3, steps: 20, progress: 0.15 });
  const other = env.submit("txt2img", "task(o)");
  await env.clock.advance(300);
  env.click("img2img_interrupt");
  env.server.set("task(o)", { completed: true });
  other.end();
  await env.clock.advance(100);
  assert.equal(env.state("txt2img"), "done");
});

test("Interrupt while this tab's job only waits in the queue does not mark it", async (t) => {
  const env = await started(t);
  env.server.set("task(w)", { queued: true, queue_position: 1, queue_size: 1 });
  const record = env.submit("txt2img", "task(w)");
  await env.clock.advance(300);
  env.click("txt2img_interrupt");                              // stops the job that runs now (another tab's)
  env.server.set("task(w)", { active: true, step: 2, steps: 4, progress: 0.5, eta: 1, pass_eta: 1 });
  await env.clock.advance(300);
  env.server.set("task(w)", { completed: true });
  record.end();
  await env.clock.advance(100);
  assert.equal(env.state("txt2img"), "done");
});

test("\"Don't Interrupt in the middle\": stopping on the last image is done, earlier is interrupted", async (t) => {
  const env = await started(t, { options: { interrupt_after_current: true } });
  for (const [jobNo, expected] of [[1, "interrupted"], [3, "done"]]) {
    const id = `task(stop${jobNo})`;
    env.server.set(id, { active: true, job_no: jobNo, job_count: 4, step: 2, steps: 10, progress: (jobNo + 0.2) / 4,
      eta: 5, pass_eta: 5 });
    const record = env.submit("txt2img", id);
    await env.clock.advance(300);
    env.click("txt2img_interrupt");                            // first click: stop after this image
    env.server.set(id, { completed: true });
    record.end();
    await env.clock.advance(100);
    assert.equal(env.state("txt2img"), expected, `stop requested on image ${jobNo + 1} of 4`);
  }
  // A second click is an immediate interrupt, even on the last image.
  env.server.set("task(hard)", { active: true, job_no: 3, job_count: 4, step: 2, steps: 10, progress: 0.8 });
  const record = env.submit("txt2img", "task(hard)");
  await env.clock.advance(300);
  env.click("txt2img_interrupt");
  env.click("txt2img_interrupting");
  env.server.set("task(hard)", { completed: true });
  record.end();
  await env.clock.advance(100);
  assert.equal(env.state("txt2img"), "interrupted");
});

test("Skip moves on to the next batch item and is not an interruption", async (t) => {
  const env = await started(t);
  env.server.set("task(k)", { active: true, step: 3, steps: 20, progress: 0.15, skipped: true });
  const record = env.submit("txt2img", "task(k)");
  await env.clock.advance(300);
  env.click("txt2img_skip");
  env.server.set("task(k)", { completed: true });
  record.end();
  await env.clock.advance(100);
  assert.equal(env.state("txt2img"), "done");
});

test("an ultra-fast job that ended before any poll ends done, not stuck at 0/0", async (t) => {
  const env = await started(t);
  env.server.set("task(fast)", { completed: true });
  const record = env.submit("txt2img", "task(fast)");
  record.end();                                                // Forge's first poll already said completed
  await env.clock.advance(50);
  assert.equal(env.state("txt2img"), "done");
  assert.equal(env.text("txt2img"), "100%");
  await env.clock.advance(400);
  assert.equal(env.value("txt2img"), 100);

  const env2 = await started(t);
  env2.server.set("task(fast2)", { completed: true });         // our first poll is already too late
  env2.submit("txt2img", "task(fast2)");
  await env2.clock.advance(400);
  assert.equal(env2.state("txt2img"), "done");
  assert.equal(env2.value("txt2img"), 100);
});

test("a job that never started (Forge's inactivity timeout) goes back to idle, not done", async (t) => {
  const env = await started(t);
  const record = env.submit("txt2img", "task(lost)");
  await env.clock.advance(1000);
  assert.equal(env.state("txt2img"), "waiting");
  assert.equal(env.text("txt2img"), "Waiting...");
  record.end();
  await env.clock.advance(100);
  assert.equal(env.state("txt2img"), "idle");
  assert.equal(env.text("txt2img"), "");
  assert.notEqual(env.value("txt2img"), 100);
});

test("if Forge's own polling gives up while the job still runs, the bar keeps following it", async (t) => {
  const env = await started(t);
  env.server.set("task(r)", { active: true, step: 4, steps: 20, progress: 0.2, eta: 8, pass_eta: 8 });
  const record = env.submit("txt2img", "task(r)");
  await env.clock.advance(400);
  record.end();                                                // e.g. a failed /internal/progress request
  await env.clock.advance(400);
  assert.equal(env.state("txt2img"), "running");
  env.server.set("task(r)", { completed: true });
  await env.clock.advance(400);
  assert.equal(env.state("txt2img"), "done");
});

// ---- DOM 을 건드리지 않는다 ---------------------------------------------------------------

test("zero node insertions or removals while it animates — only text node data changes", async (t) => {
  const env = await started(t, { options: { sam3_progress_smoothness: "smooth_eq_acc" } });
  const bar = env.bar("txt2img");
  const nodes = [...bar.querySelectorAll(".sam3-progress-label")].map((label) => label.firstChild);
  const records = [];
  const observer = new env.window.MutationObserver((list) => records.push(...list));
  observer.observe(env.doc.body, { childList: true, subtree: true, characterData: true });
  const t0 = env.clock.now;
  env.server.set("task(m)", () => {
    const progress = Math.min(0.95, (env.clock.now - t0) / 4000);
    return { active: true, step: Math.floor(progress * 20), steps: 20, progress, eta: 4 - (env.clock.now - t0) / 1000,
      pass_eta: 4 - (env.clock.now - t0) / 1000 };
  });
  const record = env.submit("txt2img", "task(m)");
  await env.clock.advance(3000);
  env.server.set("task(m)", { completed: true });
  await env.clock.advance(300);
  record.end();
  await env.clock.advance(3000);
  await env.clock.flush();
  observer.disconnect();
  const structural = records.filter((rec) => rec.type === "childList");
  const foreign = structural.filter((rec) => ![...rec.addedNodes, ...rec.removedNodes]
    .every((node) => node.nodeType === 1 && node.classList.contains("progressDiv")));
  assert.deepEqual(foreign, [], "only Forge's own .progressDiv was inserted/removed");
  assert.equal(structural.length, 2, "Forge inserted and removed its bar once each");
  const texts = records.filter((rec) => rec.type === "characterData");
  assert.ok(texts.length > 10, `text updates: ${texts.length}`);
  assert.ok(texts.every((rec) => nodes.includes(rec.target)));
  assert.deepEqual([...bar.querySelectorAll(".sam3-progress-label")].map((label) => label.firstChild), nodes,
    "the same two text nodes throughout");
});

test("Forge's native bar is hidden by CSS next to ours, never removed by us", async (t) => {
  const rule = CSS.slice(CSS.indexOf("/* sam3-progress:begin */"), CSS.indexOf("/* sam3-progress:end */"));
  const hide = /(html\[data-sam3-progress="on"\][^{]+)\{\s*display: none !important;\s*\}/.exec(rule);
  assert.ok(hide, "the hide rule is in style.css");
  const env = await started(t);
  const record = env.submit("txt2img", "task(n)");
  const extras = env.doc.createElement("div");
  extras.className = "progressDiv";
  env.doc.getElementById("extras_gallery_container").before(extras);
  assert.ok(record.div.isConnected);
  assert.equal(record.div.matches(hide[1].trim()), true, "txt2img's native bar is hidden");
  assert.equal(extras.matches(hide[1].trim()), false, "a tab without our bar keeps Forge's");
  env.setOptions({ sam3_progress_enabled: false });
  assert.equal(record.div.matches(hide[1].trim()), false, "disabled: Forge's bar is back");
  assert.ok(record.div.isConnected, "still there for sd-webui-api-payload-display");
});

test("Forge's bar stays for a job our bar does not draw — the option turned on (or off and on) mid-generation", async (t) => {
  const env = await started(t, { options: { sam3_progress_enabled: false } });
  const before = env.submit("txt2img", "task(before)");        // started while the option was off: Forge's bar only
  env.setOptions({ sam3_progress_enabled: true });
  assert.equal(env.state("txt2img"), "idle");
  assert.equal(before.div.matches(HIDE), false, "a job we never took on keeps Forge's bar");
  await env.clock.advance(1000);
  assert.equal(env.server.polled("task(before)"), 0);
  before.end();
  env.server.set("task(ours)", { active: true, step: 2, steps: 8, progress: 0.25, eta: 6, pass_eta: 6 });
  const ours = env.submit("txt2img", "task(ours)");
  assert.equal(ours.div.matches(HIDE), true, "hidden from the moment Forge inserts it");
  await env.clock.advance(300);
  assert.equal(ours.div.matches(HIDE), true);
  env.setOptions({ sam3_progress_enabled: false });
  env.setOptions({ sam3_progress_enabled: true });             // nobody draws it any more
  assert.equal(env.state("txt2img"), "idle");
  assert.equal(ours.div.matches(HIDE), false, "Forge's bar is back for the rest of the job");
  ours.end();
});

// ---- 물러나기 ---------------------------------------------------------------------------

test("steps aside when sd-webui-smooth-progress is installed (#spb-dynamic-css)", async (t) => {
  const env = buildDom(t, { upstream: true });
  await start(env);
  assert.equal(env.window.requestProgress, env.forge.original);
  assert.equal(env.doc.querySelector(".sam3-progress"), null);
  assert.equal(env.doc.documentElement.hasAttribute("data-sam3-progress"), false);
  assert.equal(env.hooks.steppedAside(), true);
  assert.equal(env.logs.info.length, 1);
  assert.match(env.logs.info[0], /sd-webui-smooth-progress/);
  env.submit("txt2img", "task(u)");
  await env.clock.advance(2000);
  assert.equal(env.server.calls.length, 0);
});

test("steps aside when the upstream extension's style shows up after we mounted", async (t) => {
  const env = await started(t);
  assert.ok(env.bar("txt2img"));
  const style = env.doc.createElement("style");
  style.id = "spb-dynamic-css";
  env.doc.head.appendChild(style);
  const atEnd = () => "upstream-owned";
  env.window.requestProgress("task(u2)", env.container("txt2img"), env.gallery("txt2img"), atEnd);
  const record = env.forge.calls[0];
  assert.equal(record.args[3], atEnd, "passed through untouched");
  assert.equal(env.doc.querySelector(".sam3-progress"), null);
  assert.equal(env.doc.documentElement.hasAttribute("data-sam3-progress"), false);
  await env.clock.advance(2000);
  assert.equal(env.server.calls.length, 0);
  env.setOptions({ sam3_progress_height: 30 });
  assert.equal(env.doc.querySelector(".sam3-progress"), null, "stays aside");
});

// ---- 숨은 페이지, 한 줄 묻기 ------------------------------------------------------------------

test("polling pauses while the page is hidden and resumes when it is shown", async (t) => {
  const env = await started(t);
  let hidden = true;
  Object.defineProperty(env.doc, "hidden", { configurable: true, get: () => hidden });
  env.server.set("task(h)", { active: true, step: 2, steps: 20, progress: 0.1 });
  env.submit("txt2img", "task(h)");
  await env.clock.advance(3000);
  assert.equal(env.server.calls.length, 0);
  hidden = false;
  env.doc.dispatchEvent(new env.window.Event("visibilitychange"));
  await env.clock.advance(1000);
  const polls = env.server.calls.length;
  assert.ok(polls >= 4, `polls after showing: ${polls}`);
  hidden = true;
  env.doc.dispatchEvent(new env.window.Event("visibilitychange"));
  await env.clock.advance(2000);
  assert.ok(env.server.calls.length <= polls + 1);
});

test("one poll at a time, even when visibility flips while a request is pending", async (t) => {
  const env = await started(t);
  let release;
  env.server.gate = new Promise((resolve) => { release = resolve; });
  env.server.set("task(g)", { active: true, step: 2, steps: 20, progress: 0.1 });
  env.submit("txt2img", "task(g)");
  await env.clock.advance(100);
  for (let i = 0; i < 5; i += 1) {
    env.doc.dispatchEvent(new env.window.Event("visibilitychange"));
    await env.clock.advance(300);
  }
  assert.equal(env.server.calls.length, 1);
  env.server.gate = null;
  release();
  await env.clock.advance(2000);
  assert.equal(env.server.maxInFlight, 1);
  assert.ok(env.server.calls.length >= 8);
});

// ---- 설정 -------------------------------------------------------------------------------

test("settings apply live and out-of-range values are clamped", async (t) => {
  const env = await started(t, { options: { sam3_progress_height: 999, sam3_progress_text_align: -40,
    sam3_progress_fade_seconds: 9, sam3_progress_color: "custom", sam3_progress_custom_color: "#FFF",
    sam3_progress_interrupt_style: "red_text" } });
  const style = () => env.bar("txt2img").style;
  assert.equal(style().getPropertyValue("--sam3-progress-height"), "50px");
  assert.equal(style().getPropertyValue("--sam3-progress-align"), "0");
  assert.equal(style().getPropertyValue("--sam3-progress-fade"), "4s");
  assert.equal(style().getPropertyValue("--sam3-progress-color"), "#ffffff");
  assert.equal(style().getPropertyValue("--sam3-progress-color-ink"), "var(--sam3-color-accent-ink)");
  assert.equal(env.bar("txt2img").getAttribute("data-interrupt"), "red_text");
  env.setOptions({ sam3_progress_height: 3, sam3_progress_text_align: 37, sam3_progress_fade_seconds: "x",
    sam3_progress_color: "blue" });
  assert.equal(style().getPropertyValue("--sam3-progress-height"), "10px");
  assert.equal(style().getPropertyValue("--sam3-progress-align"), "37");
  assert.equal(style().getPropertyValue("--sam3-progress-fade"), "0.4s");
  assert.equal(style().getPropertyValue("--sam3-progress-color"), "#2563eb");
  assert.equal(style().getPropertyValue("--sam3-progress-color-ink"), "var(--sam3-color-ink)");
  env.setOptions({ sam3_progress_height: "tall", sam3_progress_color: "dandelion" });
  assert.equal(style().getPropertyValue("--sam3-progress-height"), "20px");
  assert.equal(style().getPropertyValue("--sam3-progress-color-ink"), "var(--sam3-color-accent-ink)");
  env.setOptions({ sam3_progress_color: "success" });
  assert.equal(style().getPropertyValue("--sam3-progress-color"), "var(--sam3-color-success)");
  env.setOptions({ sam3_progress_color: "accent", sam3_progress_custom_color: "not a colour" });
  assert.equal(style().getPropertyValue("--sam3-progress-color"), "");
  env.setOptions({ sam3_progress_color: "custom" });
  assert.equal(style().getPropertyValue("--sam3-progress-color"), "#ef256c", "invalid custom colour: default");
  env.setOptions({ sam3_progress_smoothness: "warp", sam3_progress_text_format: 3, sam3_progress_after_finish: null });
  const config = env.hooks.config();
  assert.deepEqual([config.smoothness, config.format, config.afterFinish], ["smooth_gt_acc", "steps_pct_eta", "fade"]);
  env.setOptions({ sam3_progress_enabled: "yes" });
  assert.equal(env.doc.querySelector(".sam3-progress"), null, "only a real true enables it");
});

test("changing the after-finish mode re-applies it to a finished bar", async (t) => {
  const env = await started(t, { options: { sam3_progress_after_finish: "keep" } });
  env.server.set("task(f)", { completed: true });
  const record = env.submit("txt2img", "task(f)");
  await env.clock.advance(2000);
  record.end();                                                // Forge's own poll saw it completed too
  assert.deepEqual(env.shown("txt2img"), { fill: "1", text: "1" });
  env.setOptions({ sam3_progress_after_finish: "fade_text_only" });
  assert.deepEqual(env.shown("txt2img"), { fill: "1", text: "0" });
  env.setOptions({ sam3_progress_after_finish: "fade" });
  assert.deepEqual(env.shown("txt2img"), { fill: "0", text: "0" });
  await env.clock.advance(1000);
  assert.equal(env.state("txt2img"), "idle");
});

// jsdom 은 사용자 정의 속성을 캐스케이드·상속하지 않는다: <html> 인라인의 테마 변수를 모든 요소가 물려받는 것처럼 읽게 한다.
function inheritRootVars(env) {
  const real = env.window.getComputedStyle.bind(env.window);
  env.window.getComputedStyle = (element, pseudo) => new Proxy(real(element, pseudo), {
    get(target, prop) {
      if (prop === "getPropertyValue") {
        return (name) => target.getPropertyValue(name)
          || (name.startsWith("--") ? env.doc.documentElement.style.getPropertyValue(name) : "");
      }
      const value = target[prop];
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}

test("colour lightness reads the theme notations: hex, rgb(), oklch()", async (t) => {
  const env = await started(t);
  const lum = env.hooks.colorLuminance;
  const near = (a, b) => Math.abs(a - b) < 1e-9;
  assert.ok(near(lum("#ffffff"), 1));
  assert.equal(lum("#000"), 0);
  assert.ok(near(lum("rgb(249, 115, 22)"), lum("#f97316")));
  assert.ok(near(lum("rgba(249 115 22 / 50%)"), lum("#F97316")));
  assert.ok(near(lum("oklch(86% .025 70)"), 0.86 ** 3));
  assert.ok(near(lum(" oklch(0.5 0.1 200) "), 0.125));
  for (const unread of ["", "red", "hsl(0 0% 50%)", "rgb(100%, 0%, 0%)", "oklch(.", "color-mix(in srgb, red, blue)", null]) {
    assert.equal(lum(unread), null, String(unread));
  }
  assert.equal(env.hooks.isLight("#f97316"), true);
  assert.equal(env.hooks.isLight("#1e3a8a"), false);
});

test("the theme-accent bar gets readable text in every theme, Forge Default included", async (t) => {
  // 실측(Forge Default 밝은 테마): 막대 #f97316 위 글자가 --button-primary-text-color #ea580c — 주황 위 주황(1.2:1).
  const env = buildDom(t);
  inheritRootVars(env);
  const root = env.doc.documentElement.style;
  root.setProperty("--color-accent", "#f97316");
  root.setProperty("--button-primary-text-color", "#ea580c");
  await start(env);
  const ink = (tab = "txt2img") => env.bar(tab).style.getPropertyValue("--sam3-progress-color-ink");
  const theme = () => env.doc.dispatchEvent(new env.window.CustomEvent("sam3:appearance-theme", { detail: {} }));
  assert.equal(env.bar("txt2img").style.getPropertyValue("--sam3-progress-color"), "", "the bar keeps the theme colour");
  assert.equal(ink(), "var(--sam3-color-accent-ink)", "light accent: dark text");
  assert.equal(ink("img2img"), "var(--sam3-color-accent-ink)");
  // A theme switch (appearance_theme.js announces it) picks the ink again.
  root.setProperty("--color-accent", "oklch(45% .15 265)");
  theme();
  assert.equal(ink(), "var(--sam3-color-ink)", "dark accent: light text");
  root.setProperty("--color-accent", "oklch(86% .025 70)");       // OLED Mono: the same ink its own buttons use
  theme();
  assert.equal(ink(), "var(--sam3-color-accent-ink)");
  // No --color-accent: style.css falls back to --sam3-color-accent, so does the ink.
  root.removeProperty("--color-accent");
  root.setProperty("--sam3-color-accent", "rgb(30, 58, 138)");
  theme();
  assert.equal(ink(), "var(--sam3-color-ink)");
  root.setProperty("--sam3-color-accent", "color-mix(in srgb, red, blue)");
  theme();
  assert.equal(ink(), "", "a colour it cannot read leaves style.css's ink");
  env.setOptions({ sam3_progress_color: "dandelion" });
  assert.equal(ink(), "var(--sam3-color-accent-ink)", "presets keep their own pair");
  env.setOptions({ sam3_progress_enabled: false });
  theme();                                                        // nothing to restyle, no error
  assert.equal(env.doc.querySelector(".sam3-progress"), null);
});

// ---- 움직임 -------------------------------------------------------------------------------

test("Smooth > Accurate waits for a known ETA instead of racing to 99% (upstream bug)", async (t) => {
  const env = await started(t, { options: { sam3_progress_smoothness: "smooth_gt_acc" } });
  env.server.set("task(e)", { active: true, step: 1, steps: 20, progress: 0.05, eta: null, pass_eta: null });
  env.submit("txt2img", "task(e)");
  await env.clock.advance(1500);
  assert.ok(env.value("txt2img") < 10, `unknown ETA: ${env.value("txt2img")}`);
  assert.match(env.text("txt2img"), / • \?$/);
  const t0 = env.clock.now;
  env.server.set("task(e)", () => ({ active: true, step: 2, steps: 20, progress: 0.1, eta: 10 - (env.clock.now - t0) / 1000,
    pass_eta: 10 - (env.clock.now - t0) / 1000 }));
  await env.clock.advance(5000);
  const value = env.value("txt2img");
  assert.ok(value > 40 && value < 70, `about half way through a 10 s estimate: ${value}`);
  await env.clock.advance(8000);
  assert.ok(env.value("txt2img") <= 99.2, "never past 99.2% before the end");
});

test("Smooth > Accurate holds still while the ETA is unknown — no creep through a long model load", async (t) => {
  // 실측: Forge 를 띄운 뒤 첫 생성은 첫 스텝 전 14초 동안 서버 진행률 0 — 막대와 글자가 7.7%('7% • ?')까지 올라갔다.
  const env = await started(t, { options: { sam3_progress_smoothness: "smooth_gt_acc" } });
  env.server.set("task(m)", { active: true, step: 0, steps: 20, progress: 0, eta: null, pass_eta: null });
  env.submit("txt2img", "task(m)");
  await env.clock.advance(60000);
  assert.equal(env.value("txt2img"), 0, "nothing reported, nothing shown");
  assert.equal(env.text("txt2img"), "0% • ?");
  env.server.set("task(m)", { active: true, step: 1, steps: 20, progress: 0.05, eta: null, pass_eta: null });
  await env.clock.advance(3000);
  const value = env.value("txt2img");
  assert.ok(value > 4 && value <= 5, `follows the server, not past it: ${value}`);
  assert.equal(env.text("txt2img"), "1/20 • 5% • ?");
});

test("Smooth ~ Accurate keeps upstream's coast while the server reports nothing", async (t) => {
  const env = await started(t, { options: { sam3_progress_smoothness: "smooth_eq_acc" } });
  env.server.set("task(q)", { active: true, step: 0, steps: 20, progress: 0, eta: null, pass_eta: null });
  env.submit("txt2img", "task(q)");
  await env.clock.advance(10000);
  const value = env.value("txt2img");
  assert.ok(value > 3 && value < 6, `upstream COAST_SPEED_PER_FRAME 0.008: ${value}`);
});

test("Smooth > Accurate snaps to the server's progress when the page is shown again", async (t) => {
  const env = await started(t, { options: { sam3_progress_smoothness: "smooth_gt_acc" } });
  const t0 = env.clock.now;
  const elapsed = () => (env.clock.now - t0) / 1000;
  // a 100 s job, linear, with an exact ETA
  env.server.set("task(v)", () => ({ active: true, step: Math.floor(elapsed() / 5), steps: 20,
    progress: Math.min(0.99, elapsed() / 100), eta: 100 - elapsed(), pass_eta: 100 - elapsed() }));
  const record = env.submit("txt2img", "task(v)");
  await env.clock.advance(10000);
  assert.ok(Math.abs(env.value("txt2img") - 10) < 2, String(env.value("txt2img")));
  // Hidden for a minute: the browser runs no animation frames and we ask nothing.
  let hidden = true;
  Object.defineProperty(env.doc, "hidden", { configurable: true, get: () => hidden });
  env.doc.dispatchEvent(new env.window.Event("visibilitychange"));
  const held = new Map(env.clock.frames);
  env.clock.frames.clear();
  const raf = env.window.requestAnimationFrame;
  env.window.requestAnimationFrame = (fn) => { held.set(-1 - held.size, fn); return -held.size; };
  const polls = env.server.calls.length;
  await env.clock.advance(60000);
  assert.ok(env.server.calls.length <= polls + 1, "no polling while hidden");
  hidden = false;                                              // shown again: the held frame runs
  env.window.requestAnimationFrame = raf;
  held.forEach((fn, id) => env.clock.frames.set(id, fn));
  env.doc.dispatchEvent(new env.window.Event("visibilitychange"));
  await env.clock.advance(300);
  const snapped = env.value("txt2img");
  assert.ok(snapped >= 69 && snapped <= 71.5, `at the server's ~70%, not still near 10%: ${snapped}`);
  await env.clock.advance(3000);
  const later = env.value("txt2img");
  assert.ok(later > snapped && later < 76, `keeps pace with the job (~73%): ${later}`);
  record.end();
});

test("reduced motion: no animation frames, the bar follows the server value", async (t) => {
  const env = buildDom(t, { reduced: true });
  await start(env);
  env.server.set("task(r)", { active: true, step: 10, steps: 20, progress: 0.5, eta: 7, pass_eta: 7 });
  env.submit("txt2img", "task(r)");
  await env.clock.advance(300);
  assert.equal(env.value("txt2img"), 50);
  assert.equal(env.text("txt2img"), "10/20 • 50% • 7s");
  assert.equal(env.clock.frames.size, 0);
  env.server.set("task(r)", { completed: true });
  await env.clock.advance(300);
  assert.equal(env.value("txt2img"), 100, "jumps to 100% without the fill animation");
  assert.equal(env.clock.frames.size, 0);
});

test("without the route (404) it draws from Forge's own responses", async (t) => {
  const env = await started(t, { options: { sam3_progress_smoothness: "smooth_eq_acc" } });
  env.server.status = 404;
  const record = env.submit("txt2img", "task(n)");
  await env.clock.advance(300);
  const calls = env.server.calls.length;
  record.progress({ active: true, queued: false, completed: false, progress: 0.3, eta: 12, textinfo: null });
  await env.clock.advance(2000);
  assert.equal(env.state("txt2img"), "running");
  assert.ok(env.value("txt2img") > 20 && env.value("txt2img") < 35, String(env.value("txt2img")));
  assert.match(env.text("txt2img"), /^\d+% • 1\ds$/);
  assert.equal(env.server.calls.length, calls, "stops asking a missing route");
  assert.equal(env.logs.warn.length, 1, "says so once");
  record.end();
  await env.clock.advance(100);
  assert.equal(env.state("txt2img"), "done");
});

// ---- 글자 ----------------------------------------------------------------------------------

test("text formats follow upstream's switch", async (t) => {
  const env = await started(t);
  const format = env.hooks.formatText;
  const base = { phase: "running", pct: 31.4, eta: 22, step: 5, steps: 20, jobNo: 0, jobCount: 1, waitText: "", textinfo: "" };
  const expect = {
    steps_pct_eta: "5/20 • 31% • 22s",
    steps_eta: "5/20 • 22s",
    steps_pct: "5/20 • 31%",
    pct_eta: "31% • 22s",
    eta_only: "22s",
    steps_only: "5/20",
    pct_only: "31%",
    none: "",
  };
  for (const [key, text] of Object.entries(expect)) {
    assert.equal(format({ ...base, format: key }), text, key);
  }
  const interrupted = {
    steps_pct_eta: "5/20 • 31% • 중단됨", steps_pct: "5/20 • 31% • 중단됨", steps_eta: "5/20 • 중단됨",
    steps_only: "5/20 • 중단됨", pct_eta: "31% • 중단됨", pct_only: "31% • 중단됨", eta_only: "중단됨", none: "중단됨",
  };
  for (const [key, text] of Object.entries(interrupted)) {
    assert.equal(format({ ...base, format: key, phase: "interrupted" }), text, `interrupted ${key}`);
  }
  assert.equal(format({ ...base, format: "steps_pct_eta", phase: "done", step: 20 }), "20/20 • 100%");
  assert.equal(format({ ...base, format: "eta_only", phase: "done" }), "0s");
  assert.equal(format({ ...base, format: "steps_pct_eta", eta: null }), "5/20 • 31% • ?");
  assert.equal(format({ ...base, format: "steps_pct_eta", jobNo: 1, jobCount: 4 }), "[2/4] 5/20 • 31% • 22s");
  assert.equal(format({ ...base, format: "steps_pct_eta", jobNo: 7, jobCount: 4 }), "[4/4] 5/20 • 31% • 22s");
  assert.equal(format({ ...base, format: "steps_pct_eta", steps: 0 }), "31% • 22s");
  assert.equal(format({ ...base, format: "steps_only", steps: 0 }), "");
  assert.equal(format({ ...base, format: "pct_only", textinfo: "Loading B" }), "Loading B • 31%");
  assert.equal(format({ ...base, format: "pct_eta", eta: 75 }), "31% • 01:15");
  assert.equal(format({ ...base, format: "pct_eta", eta: 3725 }), "31% • 01:02:05");
  assert.equal(format({ ...base, format: "pct_eta", phase: "waiting", waitText: "In queue: 1/2" }), "In queue: 1/2");
  assert.equal(format({ ...base, format: "none", phase: "waiting", waitText: "In queue: 1/2" }), "");
  assert.equal(env.hooks.formatTime(60), "60s");
  assert.equal(env.hooks.formatTime(61), "01:01");
  assert.equal(env.hooks.isLight("#fedf08"), true);
  assert.equal(env.hooks.isLight("#2563eb"), false);
});

// ---- Forge 의 진짜 requestProgress ---------------------------------------------------------

test("with Forge's real progressbar.js: its bar comes and goes, callbacks chain, ours follows the job",
  { skip: existsSync(FORGE_PROGRESSBAR) ? false : "Forge's javascript/progressbar.js is not next to the extension" },
  async (t) => {
    // origin: <Forge>/javascript/progressbar.js requestProgress (read from the Forge checkout, not copied)
    const native = { phase: "queued" };
    const env = buildDom(t, {
      forgeSource: readFileSync(FORGE_PROGRESSBAR, "utf8"),
      options: { show_progressbar: true, live_preview_refresh_period: 500, prevent_screen_sleep_during_generation: false,
        show_progress_in_title: true, live_previews_enable: false },
      xhr: ({ url, body }) => {
        assert.equal(url, "./internal/progress");
        assert.equal(body.id_task, "task(real)");
        if (native.phase === "queued") {
          return { active: false, queued: true, completed: false, id_live_preview: -1, textinfo: "In queue: 1/1" };
        }
        if (native.phase === "active") {
          return { active: true, queued: false, completed: false, progress: 0.5, eta: 4, id_live_preview: -1, textinfo: null };
        }
        return { active: false, queued: false, completed: true, id_live_preview: -1, textinfo: "Waiting..." };
      },
    });
    await start(env);
    assert.notEqual(env.window.requestProgress, env.forge.original, "Forge's function is wrapped");
    const ended = [];
    // Forge's submit(): requestProgress(id, txt2img_gallery_container, txt2img_gallery, atEnd)
    env.window.requestProgress("task(real)", env.container("txt2img"), env.gallery("txt2img"), () => ended.push("forge"));
    const nativeBar = env.doc.querySelector("#txt2img_results_panel > .progressDiv");
    assert.ok(nativeBar, "Forge inserted its own bar (sd-webui-api-payload-display waits for this)");
    assert.equal(nativeBar.nextElementSibling, env.container("txt2img"));
    const hide = /(html\[data-sam3-progress="on"\][^{]+)\{\s*display: none !important;/.exec(CSS);
    assert.equal(nativeBar.matches(hide[1].trim()), true);

    env.server.set("task(real)", { queued: true, queue_position: 1, queue_size: 1 });
    await env.clock.advance(600);
    assert.equal(env.state("txt2img"), "waiting");
    assert.equal(env.text("txt2img"), "In queue: 1/1");

    native.phase = "active";
    env.server.set("task(real)", { active: true, step: 5, steps: 10, progress: 0.5, eta: 4, pass_eta: 4 });
    await env.clock.advance(1200);
    assert.equal(env.state("txt2img"), "running");
    assert.match(env.text("txt2img"), /^5\/10 • \d+% • [1-4]s$/);
    assert.ok(nativeBar.isConnected, "Forge's bar is only hidden, never removed by us");

    native.phase = "completed";
    env.server.set("task(real)", { completed: true });
    await env.clock.advance(600);                              // one native poll (500 ms) + our polls (200 ms)
    assert.equal(env.state("txt2img"), "done");
    assert.equal(env.text("txt2img"), "10/10 • 100%");
    assert.equal(nativeBar.isConnected, false, "Forge removed its own bar at the end");
    assert.deepEqual(ended, ["forge"], "Forge's atEnd ran once, through our chain");
    await env.clock.advance(2000);
    assert.equal(env.state("txt2img"), "idle");
  });
