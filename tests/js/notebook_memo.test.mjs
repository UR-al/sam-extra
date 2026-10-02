// notebook_memo.js — Notebook 패널 안의 메모 탭.
//
// 서버는 sam3ext/notebook_memos.py 의 계약을 흉내 낸 가짜 fetch 다(GET 목록 / PUT upsert + base_updated_at 409 /
// DELETE 묘비). 패널 모양은 notebook.js 의 buildPanel() 을 본뜬다.
// 창은 t.after 로 닫는다 — 1초 마운트 폴링과 600ms 저장 타이머가 남아 있으면 node --test 가 끝나지 않는다.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const SCRIPT = readFileSync(path.join(ROOT, "javascript", "notebook_memo.js"), "utf8");
const API = "/sam3-notebook/memos";
const DRAFT_KEY = "sam-extra.notebook.memo.drafts.v2";
const LEGACY_DRAFT_KEY = "sam-extra.notebook.memo.drafts.v1";
const UI_KEY = "sam-extra.notebook.memo.ui.v1";
const ID_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$/;

const PANEL = `
  <details id="sam3_notebook_panel" open>
    <summary><strong>Notebook</strong><span class="sam3-notebook-status" data-notebook-status></span></summary>
    <div class="sam3-notebook-head-actions"><button type="button" data-notebook-add>＋</button></div>
    <div class="sam3-notebook-toolbar"><input type="search" data-notebook-search></div>
    <div class="sam3-notebook-presets" data-notebook-presets></div>
  </details>`;

function minutesAgo(minutes) {
  return new Date(Date.now() - minutes * 60000).toISOString();
}

function fakeServer(seed = []) {
  const memos = new Map(seed.map((memo) => [memo.id, { deleted: false, created_at: memo.updated_at, ...memo }]));
  // gate: Promise 면 요청을 기록한 뒤 그것이 풀릴 때까지 응답하지 않는다(요청이 '가는 중'인 상태).
  // failGet: GET 만 이 상태 코드로 실패한다(저장소 손상 500 등).
  const server = { calls: [], missing: false, revision: 0, gate: null, failGet: 0 };
  const respond = (status, body) => ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => JSON.parse(JSON.stringify(body)),
  });
  server.memo = (id) => memos.get(id);
  // 다른 곳(앱·다른 탭)에서 고친 것처럼 서버 쪽만 바꾼다.
  server.edit = (id, fields) => {
    const stored = memos.get(id);
    memos.set(id, { ...stored, ...fields, updated_at: new Date(Date.now() + 1000).toISOString() });
  };
  server.fetch = async (url, init = {}) => {
    const method = init.method || "GET";
    const body = init.body ? JSON.parse(init.body) : undefined;
    server.calls.push({ method, url, body, headers: init.headers });
    if (server.gate) await server.gate;
    if (server.missing) return respond(404, { detail: "Not Found" });
    if (!init.headers || init.headers["X-SAM3-Notebook"] !== "1") return respond(403, { detail: "header" });
    const { pathname, searchParams } = new URL(url, "http://127.0.0.1:7860");
    if (pathname === API && method === "GET" && server.failGet) {
      return respond(server.failGet, { detail: "Memo storage is damaged" });
    }
    if (pathname === API && method === "GET") {
      const live = [...memos.values()].filter((memo) => !memo.deleted)
        .sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at));
      return respond(200, { revision: server.revision, memos: live });
    }
    const id = decodeURIComponent(pathname.slice(API.length + 1));
    const stored = memos.get(id);
    if (method === "PUT") {
      if (stored && body.base_updated_at && Date.parse(stored.updated_at) > Date.parse(body.base_updated_at)) {
        return respond(409, { detail: "Memo changed elsewhere", memo: stored });
      }
      const floor = stored ? Date.parse(stored.updated_at) : 0;
      const updated_at = body.updated_at && Date.parse(body.updated_at) > floor
        ? body.updated_at : new Date(Math.max(Date.now(), floor + 1)).toISOString();
      const memo = {
        id, title: body.title, text: body.text, deleted: false, updated_at,
        created_at: stored ? stored.created_at : (body.created_at || updated_at),
      };
      memos.set(id, memo);
      server.revision += 1;
      return respond(200, { revision: server.revision, memo });
    }
    if (method === "DELETE") {
      if (!stored) return respond(404, { detail: "Unknown memo id" });
      const base = searchParams.get("base_updated_at");
      if (!stored.deleted && base && Date.parse(stored.updated_at) > Date.parse(base)) {
        return respond(409, { detail: "Memo changed elsewhere", memo: stored });
      }
      if (stored.deleted) return respond(200, { revision: server.revision, memo: stored });
      const tombstone = { ...stored, text: "", deleted: true, updated_at: new Date().toISOString() };
      memos.set(id, tombstone);
      server.revision += 1;
      return respond(200, { revision: server.revision, memo: tombstone });
    }
    return respond(405, { detail: "method" });
  };
  server.puts = () => server.calls.filter((call) => call.method === "PUT");
  return server;
}

function buildDom(t, { server, storage = {}, confirm = () => true } = {}) {
  const dom = new JSDOM(`<!doctype html><html><head></head><body><div id="tab_txt2img">${PANEL}</div></body></html>`, {
    runScripts: "outside-only", pretendToBeVisual: true, url: "http://127.0.0.1:7860/",
  });
  t.after(() => dom.window.close());
  const { window } = dom;
  window.gradioApp = () => window.document;
  window.onUiLoaded = () => {};                  // 테스트가 hooks.mount() 로 직접 붙인다
  window.fetch = server.fetch;
  window.confirm = confirm;
  for (const [key, value] of Object.entries(storage)) window.localStorage.setItem(key, value);
  window.__sam3NotebookMemoTestHooks = {};       // 이 객체가 있어야 테스트 훅이 붙는다
  window.eval(SCRIPT);
  const doc = window.document;
  return {
    dom, window, doc,
    hooks: window.__sam3NotebookMemoTestHooks,
    panel: () => doc.querySelector("#sam3_notebook_panel"),
    q: (selector) => doc.querySelector(selector),
    titles: () => [...doc.querySelectorAll(".sam3-memo-item-title")].map((node) => node.textContent),
    type(selector, value) {
      const field = doc.querySelector(selector);
      field.value = value;
      field.dispatchEvent(new window.Event("input", { bubbles: true }));
    },
  };
}

async function mounted(t, options) {
  const env = buildDom(t, options);
  assert.equal(env.hooks.mount(), true);
  await env.hooks.idle();
  return env;
}

test("mounts a tab bar under the summary and one memo section, keeping preset controls", async (t) => {
  const env = await mounted(t, { server: fakeServer() });
  const panel = env.panel();
  assert.equal(panel.children[0].tagName, "SUMMARY");
  assert.equal(panel.children[1].getAttribute("role"), "tablist");
  assert.ok(panel.lastElementChild.hasAttribute("data-sam3-memo"));
  assert.equal(panel.getAttribute("data-sam3-notebook-tab"), "presets");
  assert.ok(panel.querySelector("[data-notebook-add]"));
  assert.ok(env.doc.getElementById("sam3_notebook_memo_style"));

  env.hooks.mount();
  env.hooks.mount();
  assert.equal(env.doc.querySelectorAll("[data-sam3-memo]").length, 1);
  assert.equal(env.doc.querySelectorAll("[data-sam3-memo-tabs]").length, 1);

  env.q('[data-memo-tab="memo"]').click();
  assert.equal(panel.getAttribute("data-sam3-notebook-tab"), "memo");
  assert.equal(env.q('[data-memo-tab="memo"]').getAttribute("aria-selected"), "true");
  assert.equal(env.q('[data-memo-tab="presets"]').getAttribute("aria-selected"), "false");
  assert.equal(JSON.parse(env.window.localStorage.getItem(UI_KEY)).tab, "memo");
  // 탭 전환은 CSS 가 프리셋 요소를 숨긴다 — 규칙이 들어 있는지만 본다(jsdom 은 레이아웃을 하지 않는다).
  assert.match(
    env.doc.getElementById("sam3_notebook_memo_style").textContent,
    /\[data-sam3-notebook-tab="memo"\] > \.sam3-notebook-presets/,
  );
});

test("re-mounts into a panel that Gradio or notebook.js replaced", async (t) => {
  const server = fakeServer([{ id: "memo-a", title: "첫 메모", text: "a", updated_at: minutesAgo(1) }]);
  const env = await mounted(t, { server });
  env.panel().remove();
  env.q("#tab_txt2img").insertAdjacentHTML("beforeend", PANEL);
  assert.equal(env.hooks.mount(), true);
  assert.equal(env.doc.querySelectorAll("[data-sam3-memo]").length, 1);
  assert.deepEqual(env.titles(), ["첫 메모"]);
});

test("lists memos newest first and opens the one clicked", async (t) => {
  const server = fakeServer([
    { id: "memo-old", title: "", text: "\n  옛 메모 첫 줄\n둘째 줄", updated_at: minutesAgo(30) },
    { id: "memo-new", title: "새 메모", text: "본문", updated_at: minutesAgo(1) },
  ]);
  const env = await mounted(t, { server });
  assert.deepEqual(env.titles(), ["새 메모", "옛 메모 첫 줄"]);
  assert.equal(env.hooks.selectedId(), "memo-new");
  assert.equal(env.q("[data-memo-text]").value, "본문");

  env.q('[data-memo-id="memo-old"]').click();
  assert.equal(env.q("[data-memo-title]").value, "");
  assert.equal(env.q("[data-memo-text]").value, "\n  옛 메모 첫 줄\n둘째 줄");
  assert.equal(env.q('[data-memo-id="memo-old"]').getAttribute("aria-selected"), "true");
  assert.equal(server.calls[0].headers["X-SAM3-Notebook"], "1");
});

test("a new memo is stored at once and edits autosave once, with base_updated_at", async (t) => {
  const server = fakeServer();
  const env = await mounted(t, { server });

  env.q("[data-memo-new]").click();
  await env.hooks.idle();
  const [created] = server.puts();
  const id = decodeURIComponent(created.url.slice(API.length + 1));
  assert.match(id, ID_RE);
  assert.equal(created.body.base_updated_at, null);
  assert.equal(created.body.title, "");
  assert.equal(created.headers["Content-Type"], "application/json");
  const stored = server.memo(id);

  env.type("[data-memo-title]", "장보기");
  env.type("[data-memo-text]", "우유");
  env.type("[data-memo-text]", "우유, 달걀");
  assert.equal(server.puts().length, 1);           // 아직 디바운스 중
  assert.deepEqual(env.titles(), ["장보기"]);
  await new Promise((resolve) => setTimeout(resolve, 750));
  await env.hooks.idle();

  const puts = server.puts();
  assert.equal(puts.length, 2);
  assert.equal(puts[1].body.base_updated_at, stored.updated_at);
  assert.equal(puts[1].body.title, "장보기");
  assert.equal(puts[1].body.text, "우유, 달걀");
  assert.equal(server.memo(id).text, "우유, 달걀");
  assert.match(env.q("[data-memo-status]").textContent, /저장됨/);
  assert.equal(env.window.localStorage.getItem(DRAFT_KEY), null);

  // 다음 저장은 방금 받은 서버 시각을 base 로 쓴다.
  env.type("[data-memo-text]", "우유, 달걀, 빵");
  await env.hooks.flush();
  assert.equal(server.puts()[2].body.base_updated_at, server.puts()[1].body.updated_at);
});

test("a 409 keeps both: the server copy stays, my edit becomes a conflict copy", async (t) => {
  const server = fakeServer([{ id: "memo-1", title: "회의", text: "처음", updated_at: minutesAgo(5) }]);
  const env = await mounted(t, { server });
  server.edit("memo-1", { text: "앱에서 고친 내용" });

  env.type("[data-memo-text]", "브라우저에서 고친 내용");
  await env.hooks.flush();

  const puts = server.puts();
  assert.equal(puts.length, 2);
  assert.equal(puts[0].url, `${API}/memo-1`);
  const copyId = decodeURIComponent(puts[1].url.slice(API.length + 1));
  assert.notEqual(copyId, "memo-1");
  assert.match(copyId, ID_RE);
  assert.equal(puts[1].body.base_updated_at, null);
  assert.equal(puts[1].body.title, "회의 (충돌 사본)");
  assert.equal(puts[1].body.text, "브라우저에서 고친 내용");

  assert.equal(server.memo("memo-1").text, "앱에서 고친 내용");
  const byId = Object.fromEntries(env.hooks.memos().map((memo) => [memo.id, memo.text]));
  assert.deepEqual(byId, { "memo-1": "앱에서 고친 내용", [copyId]: "브라우저에서 고친 내용" });
  assert.equal(env.hooks.selectedId(), copyId);
  assert.equal(env.q("[data-memo-title]").value, "회의 (충돌 사본)");
  assert.equal(env.q("[data-memo-notice]").hidden, false);
  assert.match(env.q("[data-memo-notice-text]").textContent, /충돌 사본/);
});

test("an edit to a memo deleted elsewhere is saved as a copy", async (t) => {
  const server = fakeServer([{ id: "memo-1", title: "", text: "old", updated_at: minutesAgo(5) }]);
  const env = await mounted(t, { server });
  await server.fetch(`${API}/memo-1`, { method: "DELETE", headers: { "X-SAM3-Notebook": "1" } });

  env.type("[data-memo-text]", "지워진 줄 모르고 쓴 글");
  await env.hooks.flush();

  const memos = env.hooks.memos();
  assert.equal(memos.length, 1);
  assert.notEqual(memos[0].id, "memo-1");
  assert.equal(memos[0].title, "제목 없음 (충돌 사본)");
  assert.equal(server.memo(memos[0].id).text, "지워진 줄 모르고 쓴 글");
  assert.match(env.q("[data-memo-notice-text]").textContent, /삭제된 메모/);
});

test("delete asks first, then sends DELETE with the version it saw and selects the next memo", async (t) => {
  const seenA = minutesAgo(1);
  const server = fakeServer([
    { id: "memo-a", title: "A", text: "", updated_at: seenA },
    { id: "memo-b", title: "B", text: "", updated_at: minutesAgo(2) },
  ]);
  let answer = false;
  const env = await mounted(t, { server, confirm: () => answer });

  env.q("[data-memo-delete]").click();
  await env.hooks.idle();
  assert.equal(server.calls.filter((call) => call.method === "DELETE").length, 0);
  assert.deepEqual(env.titles(), ["A", "B"]);

  answer = true;
  env.q("[data-memo-delete]").click();
  await env.hooks.idle();
  const deletes = server.calls.filter((call) => call.method === "DELETE");
  assert.deepEqual(deletes.map((call) => call.url), [
    `${API}/memo-a?base_updated_at=${encodeURIComponent(seenA)}`,
  ]);
  assert.equal(server.memo("memo-a").deleted, true);
  assert.deepEqual(env.titles(), ["B"]);
  assert.equal(env.hooks.selectedId(), "memo-b");
});

test("focus refresh picks up outside changes without clobbering unsaved typing", async (t) => {
  const server = fakeServer([
    { id: "memo-a", title: "A", text: "a", updated_at: minutesAgo(1) },
    { id: "memo-b", title: "B", text: "b", updated_at: minutesAgo(2) },
  ]);
  const env = await mounted(t, { server });
  env.q('[data-memo-tab="memo"]').click();
  env.type("[data-memo-text]", "입력 중");        // memo-a, 아직 저장 전
  server.edit("memo-b", { title: "B (앱에서 고침)" });

  const realNow = env.window.Date.now;
  env.window.Date.now = () => realNow() + 60000;   // 포커스 새로 고침 간격을 넘긴다
  const before = server.calls.filter((call) => call.method === "GET").length;
  env.window.dispatchEvent(new env.window.Event("focus"));
  await env.hooks.idle();

  assert.equal(server.calls.filter((call) => call.method === "GET").length, before + 1);
  assert.ok(env.titles().includes("B (앱에서 고침)"));
  assert.equal(env.q("[data-memo-text]").value, "입력 중");
  assert.equal(env.hooks.memos().find((memo) => memo.id === "memo-a").text, "입력 중");
});

test("routes missing on an old Forge (404) disable new memos and say why", async (t) => {
  const server = fakeServer();
  server.missing = true;
  const env = await mounted(t, { server });
  assert.equal(env.q("[data-memo-new]").disabled, true);
  assert.match(env.q("[data-memo-status]").textContent, /다시 시작/);
  assert.match(env.q(".sam3-memo-empty").textContent, /메모 저장소가 없는/);
});

test("Forge's global Ctrl+Enter / Alt+Enter / Esc do not fire from the memo editor", async (t) => {
  const server = fakeServer([{ id: "memo-a", title: "A", text: "", updated_at: minutesAgo(1) }]);
  const env = await mounted(t, { server });
  const seen = [];
  env.doc.addEventListener("keydown", (event) => seen.push(event.key));
  const press = (selector, init) => {
    const event = new env.window.KeyboardEvent("keydown", { bubbles: true, cancelable: true, ...init });
    env.q(selector).dispatchEvent(event);
    return event;
  };

  press("[data-memo-text]", { key: "Enter", ctrlKey: true });
  press("[data-memo-text]", { key: "Enter", altKey: true });
  press("[data-memo-title]", { key: "Escape" });
  assert.deepEqual(seen, []);
  press("[data-memo-text]", { key: "a" });
  assert.deepEqual(seen, ["a"]);

  env.type("[data-memo-text]", "바로 저장");
  const save = press("[data-memo-text]", { key: "s", ctrlKey: true });
  assert.equal(save.defaultPrevented, true);
  await env.hooks.idle();
  assert.equal(server.memo("memo-a").text, "바로 저장");
});

// 탭별 초안 칸에서 메모 하나의 초안을 꺼낸다.
function draftsIn(raw) {
  const store = JSON.parse(raw || "{}");
  return Object.values(store).reduce((all, section) => ({ ...all, ...section.drafts }), {});
}


test("edits not yet saved when the page closed are re-sent from the draft", async (t) => {
  const server = fakeServer([{ id: "memo-a", title: "A", text: "저장된 글", updated_at: minutesAgo(3) }]);
  const first = await mounted(t, { server });
  first.type("[data-memo-text]", "저장 전에 닫은 글");
  assert.ok(first.window.localStorage.getItem(DRAFT_KEY));
  first.window.dispatchEvent(new first.window.Event("pagehide"));   // 브라우저가 닫기·새로 고침 때 보낸다
  const draft = first.window.localStorage.getItem(DRAFT_KEY);
  first.dom.window.close();                          // 600ms 저장 전에 닫는다
  assert.equal(Object.values(JSON.parse(draft))[0].closed, true);
  assert.equal(server.memo("memo-a").text, "저장된 글");

  const second = await mounted(t, { server, storage: { [DRAFT_KEY]: draft } });
  await second.hooks.idle();
  const last = server.puts().at(-1);
  assert.equal(last.body.text, "저장 전에 닫은 글");
  assert.equal(last.body.base_updated_at, draftsIn(draft)["memo-a"].base);
  assert.equal(server.memo("memo-a").text, "저장 전에 닫은 글");
  assert.equal(second.window.localStorage.getItem(DRAFT_KEY), null);
});

test("a new memo made before the first load succeeds keeps the previous session's draft", async (t) => {
  const server = fakeServer();
  server.failGet = 500;                              // 저장소 손상 등으로 첫 GET 이 실패
  const legacy = JSON.stringify({ "memo-prev": { title: "unsaved", text: "닫기 전에 쓴 글", base: null } });
  const env = await mounted(t, { server, storage: { [LEGACY_DRAFT_KEY]: legacy } });
  env.q("[data-memo-new]").click();
  await env.hooks.idle();
  const kept = draftsIn(env.window.localStorage.getItem(DRAFT_KEY));
  assert.equal(kept["memo-prev"].text, "닫기 전에 쓴 글");   // 새 메모의 초안 쓰기가 덮지 않았다
  assert.equal(env.window.localStorage.getItem(LEGACY_DRAFT_KEY), null);

  server.failGet = 0;
  await env.hooks.refresh();
  await env.hooks.idle();
  assert.equal(server.memo("memo-prev").text, "닫기 전에 쓴 글");
  assert.equal(env.hooks.memos().length, 2);
  assert.equal(env.window.localStorage.getItem(DRAFT_KEY), null);
});

test("another open tab's drafts are neither wiped nor taken over until that tab is gone", async (t) => {
  const server = fakeServer([{ id: "memo-a", title: "A", text: "저장된 글", updated_at: minutesAgo(3) }]);
  const otherTab = {
    "tab-other": {
      at: Date.now(), hidden: false, closed: false,
      drafts: { "memo-a": { title: "A", text: "다른 탭에서 입력 중", base: server.memo("memo-a").updated_at } },
    },
  };
  const env = await mounted(t, { server, storage: { [DRAFT_KEY]: JSON.stringify(otherTab) } });
  env.q("[data-memo-new]").click();                  // 이 탭의 초안 쓰기
  await env.hooks.idle();
  let store = JSON.parse(env.window.localStorage.getItem(DRAFT_KEY));
  assert.equal(store["tab-other"].drafts["memo-a"].text, "다른 탭에서 입력 중");
  assert.equal(server.memo("memo-a").text, "저장된 글");  // 가로채 저장하지 않았다

  // 그 탭이 소식 없이 멈춘 지 오래됐다(크래시) — 이제 이어받는다.
  const realNow = env.window.Date.now;
  env.window.Date.now = () => realNow() + 120000;
  await env.hooks.refresh();
  await env.hooks.idle();
  assert.equal(server.memo("memo-a").text, "다른 탭에서 입력 중");
  store = JSON.parse(env.window.localStorage.getItem(DRAFT_KEY) || "{}");
  assert.equal(store["tab-other"], undefined);
});

test("a 409 whose stored text already equals mine is not a conflict", async (t) => {
  const server = fakeServer([{ id: "memo-1", title: "회의", text: "처음", updated_at: minutesAgo(5) }]);
  const env = await mounted(t, { server });
  server.edit("memo-1", { text: "같은 글" });        // 다른 탭이 같은 초안을 먼저 올렸다
  env.type("[data-memo-text]", "같은 글");
  await env.hooks.flush();
  assert.equal(server.puts().length, 1);
  assert.deepEqual(Array.from(env.hooks.memos(), (memo) => memo.id), ["memo-1"]);
  assert.equal(env.q("[data-memo-notice]").hidden, true);
  // 다음 저장은 서버 시각을 딛는다.
  env.type("[data-memo-text]", "같은 글 + 더");
  await env.hooks.flush();
  assert.equal(server.memo("memo-1").text, "같은 글 + 더");
});

test("deleting a memo whose save then hits a 409 keeps the other side's newer edit", async (t) => {
  const server = fakeServer([{ id: "memo-1", title: "회의", text: "처음", updated_at: minutesAgo(5) }]);
  const env = await mounted(t, { server });
  server.edit("memo-1", { text: "앱에서 고친 내용" });
  env.type("[data-memo-text]", "지울 메모에 쓴 글");
  let release;
  server.gate = new Promise((resolve) => { release = resolve; });
  const flushing = env.hooks.flush();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(server.puts().length, 1);             // PUT 이 가는 중
  env.q("[data-memo-delete]").click();
  server.gate = null;
  release();
  await flushing;
  await env.hooks.idle();

  assert.equal(server.calls.filter((call) => call.method === "DELETE").length, 0);
  assert.equal(server.memo("memo-1").deleted, false);
  assert.equal(server.memo("memo-1").text, "앱에서 고친 내용");
  assert.equal(server.puts().length, 1);             // 지운 글로 충돌 사본을 만들지 않았다
  assert.deepEqual(Array.from(env.hooks.memos(), (memo) => [memo.id, memo.text]), [["memo-1", "앱에서 고친 내용"]]);
  assert.match(env.q("[data-memo-notice-text]").textContent, /지우지 않았습니다/);
});

test("a delete based on an old view gets a 409 and shows the newer edit instead", async (t) => {
  const server = fakeServer([{ id: "memo-1", title: "회의", text: "처음", updated_at: minutesAgo(5) }]);
  const env = await mounted(t, { server });
  server.edit("memo-1", { text: "앱에서 고친 내용" });   // 목록은 아직 옛 판이다
  env.q("[data-memo-delete]").click();
  await env.hooks.idle();

  assert.equal(server.memo("memo-1").deleted, false);
  assert.deepEqual(Array.from(env.hooks.memos(), (memo) => memo.text), ["앱에서 고친 내용"]);
  assert.match(env.q("[data-memo-notice-text]").textContent, /지우지 않았습니다/);
  // 이제 새 판을 보고 다시 지우면 지워진다.
  env.q("[data-memo-delete]").click();
  await env.hooks.idle();
  assert.equal(server.memo("memo-1").deleted, true);
});

test("a conflict copy title cut at the limit never splits an emoji", async (t) => {
  const title = "a".repeat(111) + "\u{1F600}" + "b";   // 113 코드 포인트, 114 UTF-16 단위
  const server = fakeServer([{ id: "memo-1", title, text: "처음", updated_at: minutesAgo(5) }]);
  const env = await mounted(t, { server });
  server.edit("memo-1", { text: "앱에서 고친 내용" });
  env.type("[data-memo-text]", "브라우저에서 고친 내용");
  await env.hooks.flush();

  const copyTitle = server.puts()[1].body.title;
  assert.equal(copyTitle.isWellFormed(), true);
  assert.equal(copyTitle, "a".repeat(111) + "\u{1F600}" + " (충돌 사본)");
  assert.equal([...copyTitle].length, 120);
});
