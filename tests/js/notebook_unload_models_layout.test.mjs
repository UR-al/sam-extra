// '모델 내리기' · '기본값 저장' 자리와 두 결과 글의 실제 렌더링 — headless Chrome(또는 Chromium · Edge)이 탭 줄을 그리고 두
// 버튼과 결과 글의 사각형을 잰다.
//
// jsdom 은 배치를 계산하지 않아서 tests/js/notebook_unload_models.test.mjs 는 DOM 자리와 CSS 글만 본다. 여기서는 진짜 브라우저가
// style.css · tokens.css 와 두 페이지 스크립트(기본값 저장 · 모델 내리기)를 그대로 읽어, 창 너비마다(1920 · 1280 · 1024) 진짜
// 클릭 흐름(숨은 Gradio 버튼 → 결과 JSON → 결과 한 줄)으로 결과 글을 채우기 전과 뒤를 잰다. 지키는 것:
// - '모델 내리기' 는 언제나 '기본값 저장' 과 같은 줄, 바로 오른쪽이다(좁은 창에서 줄이 넘어가도 둘이 함께).
// - 결과 글이 나와도 두 버튼은 그 자리다(결과는 버튼 아래 따로 된 한 줄, 오른쪽 정렬).
//   예전에는 1280px 로라 탭 · 1024px 에서 결과가 나오면 '모델 내리기' 가 다음 줄 왼쪽 끝으로 약 1150px 옮겨 갔다.
// - 기본값 저장의 결과 글도 같다(v0.34.0 실제 확인: 긴 저장 결과가 두 버튼을 왼쪽으로 약 485px 밀었다 — 결과 글이 버튼 묶음 안에
//   있었다). 두 결과가 함께 나와도 각자 한 줄이고 버튼은 그 자리다. 저장 쪽의 Forge Defaults Apply 와 숨은 버튼은 흉내 낸다
//   (tests/js/notebook_save_defaults.test.mjs 와 같은 방법).
// Gradio 4.40 탭 줄의 CSS(templates/frontend/assets/Tabs-*.css 의 .tab-nav)는 아래에 옮겨 적었다. Forge 의 style.css 는 확장 옆
// Forge 체크아웃(<Forge>/extensions/<ext>, 또는 $SAM3_FORGE_ROOT)에서 읽고, 없으면(CI) 탭 줄에 닿는 두 규칙만 옮겨 쓴다.
// 브라우저가 없으면 건너뛴다($SAM3_TEST_BROWSER 로 지정 가능).
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import os from "node:os";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const read = (relpath) => readFileSync(path.join(ROOT, relpath), "utf8");
const SCRIPT = read("javascript/notebook_unload_models.js");
const SAVE_SCRIPT = read("javascript/notebook_save_defaults.js");
const ARGS_JS = JSON.parse(/^ARGS_JS = (".*")$/m.exec(read("sam3ext/ui_unload_models.py"))[1]);
const SAVE_ARGS_JS = JSON.parse(/^ARGS_JS = (".*")$/m.exec(read("sam3ext/ui_save_defaults.py"))[1]);
const RESULT_TEXT = "VRAM 사용 19.6 GB → 3.1 GB (Forge 모델 · SAM3) · 이 프로세스 예약 15.2 → 0.0 GB";
// the long one from the v0.34.0 live check (Defaults written, the preset refused)
const SAVE_ERROR = "txt2img Sampler: 'EXP Heun 2 x0 SDE' 은(는) Forge 의 선택지에 없습니다";
const SAVE_TEXT = `기본값 저장됨 (바뀐 항목 1개) · 프리셋 저장 실패: ${SAVE_ERROR}`;

function findBrowser() {
  const named = process.env.SAM3_TEST_BROWSER || process.env.CHROME_PATH;
  if (named) return existsSync(named) ? named : null;
  const env = (name, fallback) => process.env[name] || fallback;
  const candidates = process.platform === "win32" ? [
    path.join(env("PROGRAMFILES", "C:\\Program Files"), "Google", "Chrome", "Application", "chrome.exe"),
    path.join(env("PROGRAMFILES(X86)", "C:\\Program Files (x86)"), "Google", "Chrome", "Application", "chrome.exe"),
    path.join(env("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
    path.join(env("PROGRAMFILES(X86)", "C:\\Program Files (x86)"), "Microsoft", "Edge", "Application", "msedge.exe"),
    path.join(env("PROGRAMFILES", "C:\\Program Files"), "Microsoft", "Edge", "Application", "msedge.exe"),
  ] : process.platform === "darwin" ? [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
  ] : ["/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser"];
  return candidates.find((candidate) => candidate && existsSync(candidate)) || null;
}

function forgeStyle() {
  for (const root of [path.resolve(ROOT, "..", ".."), process.env.SAM3_FORGE_ROOT]) {
    if (!root) continue;
    const file = path.join(root, "style.css");
    if (existsSync(file) && existsSync(path.join(root, "modules"))) return readFileSync(file, "utf8");
  }
  return null;
}

const BROWSER = findBrowser();
const FORGE_STYLE = forgeStyle();

// origin: gradio 4.40 templates/frontend/assets/Tabs-*.css (.tabs · .tab-nav · its buttons), without the svelte hash
const GRADIO_TABS = `
  :root { --size-1: 4px; --size-4: 16px; --border-color-primary: #ccc; --container-radius: 6px; }
  body { margin: 0; padding: 8px; font-family: sans-serif; font-size: 14px; }
  .tabs { position: relative; }
  .tab-nav { display: flex; position: relative; flex-wrap: wrap; border-bottom: 1px solid var(--border-color-primary); }
  .tab-nav > [role="tab"], .tab-nav > [data-sam3-lm-btn] { margin-bottom: -1px; border: 1px solid transparent;
    border-bottom: none; padding: var(--size-1) var(--size-4); font-size: 14px; background: none; }`;
// origin: <Forge>/style.css .extra-networks>div.tab-nav · .extra-networks-controls-div (used when no Forge checkout)
const FORGE_STAND_IN = `
  .extra-networks > div.tab-nav { min-height: 2.7rem; }
  .extra-networks-controls-div { align-self: center; margin-left: auto; }
  #txt2img_lora_controls { width: 28rem; height: 2rem; }`;

// clicks: which buttons the page presses after the first measurement, in this order ("save", "unload")
function page({ lora, clicks = ["unload"] }) {
  const controls = `<div class="extra-network-control" id="txt2img_lora_controls" style="display:${lora ? "flex" : "none"}">
      <div class="extra-network-control--search"><input class="extra-network-control--search-text" type="search"
        placeholder="Search"></div><small>Sort: </small>
      ${["default", "name", "date_created", "date_modified"].map((key) => `<div class="extra-network-control--sort"
        data-sortkey="${key}"><i class="extra-network-control--icon extra-network-control--sort-icon"></i></div>`).join("")}
      <small> </small><div class="extra-network-control--sort-dir"><i class="extra-network-control--icon"></i></div>
      <small> </small><div class="extra-network-control--tree-view"><i class="extra-network-control--icon"></i></div>
      <div class="extra-network-control--refresh"><i class="extra-network-control--icon"></i></div></div>`;
  const styles = [GRADIO_TABS, FORGE_STYLE ?? FORGE_STAND_IN, read("tokens.css"), read("style.css")];
  return `<!doctype html><html><head><meta charset="utf-8">
    ${styles.map((css) => `<style>${css}</style>`).join("\n")}</head><body>
    <div class="gradio-container"><div id="tab_txt2img"><main id="sam3_notebook_layout"><section class="sam3-notebook-prompt">
      <div id="txt2img_extra_tabs" class="tabs gradio-tabs extra-networks sam3-notebook-extra-tabs">
        <div class="tab-nav scroll-hide" role="tablist">
          <button role="tab" class="selected">Generation</button><button role="tab">임베딩</button>
          <button role="tab">체크포인트</button><button role="tab">로라</button><button data-sam3-lm-btn="1">Manage</button>
          <div class="extra-networks-controls-div">${controls}</div>
        </div>
        <div class="tabitem" style="height: 40px"></div>
      </div>
    </section></main></div></div>
    <div id="sam3_unload_models_bridge" style="display: none">
      <div id="sam3_unload_models_request"><textarea></textarea></div>
      <button id="sam3_unload_models_run">sam3_unload_models</button>
      <div id="sam3_unload_models_result"><textarea></textarea></div>
    </div>
    <div id="tab_settings" style="display: none"><div id="settings_tab_defaults" class="tabitem">
      <div class="row"><button id="ui_defaults_apply">Apply</button></div>
      <div class="block gradio-html" id="review"><div class="wrap center full hide"></div>
        <div class="review-inner"><div class="prose gradio-html"></div></div></div>
    </div></div>
    <div id="sam3_save_defaults_bridge" style="display: none">
      <div id="sam3_save_defaults_request"><textarea></textarea></div>
      <button id="sam3_save_defaults_run">sam3_save_defaults</button>
      <div id="sam3_save_defaults_result"><textarea></textarea></div>
    </div>
    <script>window.gradioApp = function () { return document; };
      window.__sam3SaveDefaultsTestHooks = {}; window.__sam3UnloadModelsTestHooks = {};</script>
    <script>${SAVE_SCRIPT}</script>
    <script>${SCRIPT}</script>
    <script>
    (async function () {
      var hooks = window.__sam3UnloadModelsTestHooks;
      var saveHooks = window.__sam3SaveDefaultsTestHooks;
      hooks.config.doneVisibleMs = 0;            // keep the result lines up while measuring
      saveHooks.config.savedVisibleMs = 0;
      saveHooks.config.errorVisibleMs = 0;
      saveHooks.ensureButton();
      hooks.ensureButton();
      // Forge's Settings → Defaults Apply as Gradio renders it: pending, then "Wrote N changes."
      document.querySelector("#ui_defaults_apply").addEventListener("click", function () {
        var inner = document.querySelector("#review .review-inner");
        setTimeout(function () {
          inner.classList.add("pending");
          setTimeout(function () {
            inner.classList.remove("pending");
            document.querySelector("#review .prose").innerHTML = "<p>Wrote 1 changes.</p>";
          }, 20);
        }, 5);
      });
      // the save's hidden Gradio button: the preset part refused (the live check's message)
      var saveArgsJs = ${SAVE_ARGS_JS};
      var saveResult = document.querySelector("#sam3_save_defaults_result textarea");
      document.querySelector("#sam3_save_defaults_run").addEventListener("click", function () {
        var args = saveArgsJs("", "anima", saveResult.value);
        setTimeout(function () {
          saveResult.value = JSON.stringify({ ok: false, request: args[0], preset: "anima", error: ${JSON.stringify(SAVE_ERROR)} });
        }, 30);
      });
      // the hidden Gradio button: its js builds the inputs, Python's JSON comes back with the same request id
      var argsJs = ${ARGS_JS};
      var result = document.querySelector("#sam3_unload_models_result textarea");
      document.querySelector("#sam3_unload_models_run").addEventListener("click", function () {
        var args = argsJs("", result.value);
        setTimeout(function () {
          result.value = JSON.stringify({ ok: true, request: args[0], refused: false, text: ${JSON.stringify(RESULT_TEXT)},
                                          detail: "GPU 전체 사용: 19.6 GB → 3.1 GB" });
        }, 30);
      });
      function box(rect) {
        return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom, width: rect.width, height: rect.height };
      }
      function text(status) {
        if (!status.textContent) return null;
        var range = document.createRange();
        range.selectNodeContents(status);
        return box(range.getBoundingClientRect());
      }
      function measure() {
        var status = document.getElementById("sam3_unload_models_status");
        var saveStatus = document.getElementById("sam3_save_defaults_status");
        return {
          row: box(document.querySelector("#txt2img_extra_tabs > .tab-nav").getBoundingClientRect()),
          save: box(document.querySelector("#sam3_save_defaults > button").getBoundingClientRect()),
          unload: box(document.querySelector("#sam3_unload_models button").getBoundingClientRect()),
          status: box(status.getBoundingClientRect()),
          statusText: text(status),
          saveStatus: box(saveStatus.getBoundingClientRect()),
          saveStatusText: text(saveStatus),
          saveText: saveStatus.textContent,
        };
      }
      var out = { before: measure() };
      var clicks = ${JSON.stringify(clicks)};
      for (var i = 0; i < clicks.length; i++) {
        if (clicks[i] === "save") await saveHooks.saveDefaults();
        else await hooks.unloadModels();
      }
      out.after = measure();
      var node = document.createElement("script");
      node.type = "application/json";
      node.id = "sam3-layout";
      node.textContent = JSON.stringify(out);
      document.body.appendChild(node);
    })();
    </script></body></html>`;
}

function render(t, width, options) {
  const dir = mkdtempSync(path.join(os.tmpdir(), "sam3-unload-layout-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const file = path.join(dir, "index.html");
  writeFileSync(file, page(options), "utf8");
  const args = ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--hide-scrollbars",
    `--user-data-dir=${path.join(dir, "profile")}`, `--window-size=${width},600`, "--virtual-time-budget=5000",
    "--dump-dom", pathToFileURL(file).href];
  if (process.platform === "linux") args.unshift("--no-sandbox");   // CI containers without user namespaces
  const html = execFileSync(BROWSER, args, { encoding: "utf8", timeout: 60000, stdio: ["ignore", "pipe", "pipe"] });
  const match = /<script type="application\/json" id="sam3-layout">([^<]*)<\/script>/.exec(html);
  assert.ok(match, "the page did not finish measuring");
  return JSON.parse(match[1]);
}

function assertTogether(layout, label) {
  const { save, unload } = layout;
  assert.ok(Math.abs(unload.top - save.top) <= 1, `${label}: on the save button's line (${save.top} vs ${unload.top})`);
  const gap = unload.left - save.right;
  assert.ok(gap >= 0 && gap <= 16, `${label}: right after the save button (gap ${gap}px)`);
}

const SKIP = BROWSER ? false : "no Chrome / Chromium / Edge found (set $SAM3_TEST_BROWSER)";

for (const [width, lora, where] of [[1920, false, "Generation"], [1280, true, "로라 (Forge's search/sort shown)"],
  [1280, false, "Generation"], [1024, false, "Generation"], [1024, true, "로라"]]) {
  test(`${width}px, ${where} tab: the two buttons stay together and the result never moves them`, { skip: SKIP }, (t) => {
    const { before, after } = render(t, width, { lora });
    assertTogether(before, "before the click");
    assertTogether(after, "with the result shown");
    for (const key of ["save", "unload"]) {
      assert.ok(Math.abs(after[key].left - before[key].left) <= 1,
        `${key}: moved ${after[key].left - before[key].left}px sideways when the result appeared`);
      assert.ok(Math.abs(after[key].top - before[key].top) <= 4,
        `${key}: moved ${after[key].top - before[key].top}px down when the result appeared`);
    }
    assert.equal(before.statusText, null);
    assert.ok(after.statusText && after.statusText.width > 0, "the result is shown");
    assert.ok(after.statusText.top >= Math.max(after.save.bottom, after.unload.bottom) - 1, "on a line under the buttons");
    assert.ok(after.row.right - after.statusText.right <= 12, "right-aligned in the row");
    assert.ok(after.status.bottom <= after.row.bottom + 1, "inside the tab row");
    if (width === 1920) {
      assert.ok(after.row.right - after.unload.right <= 12, "at the row's right end");
      assert.ok(Math.abs(after.statusText.right - after.unload.right) <= 12, "under the buttons");
    }
  });
}

function assertButtonsStay(before, after) {
  for (const key of ["save", "unload"]) {
    assert.ok(Math.abs(after[key].left - before[key].left) <= 1,
      `${key}: moved ${after[key].left - before[key].left}px sideways when the result appeared`);
    assert.ok(Math.abs(after[key].top - before[key].top) <= 4,
      `${key}: moved ${after[key].top - before[key].top}px down when the result appeared`);
  }
}

function assertOwnLine(after, textKey, label) {
  const shown = after[textKey];
  assert.ok(shown && shown.width > 0, `${label}: shown`);
  assert.ok(shown.top >= Math.max(after.save.bottom, after.unload.bottom) - 1, `${label}: on a line under the buttons`);
  assert.ok(after.row.right - shown.right <= 12, `${label}: right-aligned in the row`);
  assert.ok(shown.left >= after.row.left - 1, `${label}: inside the row on the left`);
}

for (const [width, lora, where] of [[1920, false, "Generation"], [1280, true, "로라 (Forge's search/sort shown)"],
  [1280, false, "Generation"], [1024, false, "Generation"], [1024, true, "로라"]]) {
  test(`${width}px, ${where} tab: a long 기본값 저장 result sits on its own line and never moves the buttons`, { skip: SKIP }, (t) => {
    const { before, after } = render(t, width, { lora, clicks: ["save"] });
    assertTogether(before, "before the click");
    assertTogether(after, "with the result shown");
    assertButtonsStay(before, after);
    assert.equal(before.saveStatusText, null);
    assert.equal(after.saveText, SAVE_TEXT);
    assertOwnLine(after, "saveStatusText", "the save result");
    assert.ok(after.saveStatus.bottom <= after.row.bottom + 1, "inside the tab row");
    if (width === 1920) {
      assert.ok(Math.abs(after.saveStatusText.right - after.unload.right) <= 12, "under the buttons");
    }
  });

  test(`${width}px, ${where} tab: both results at once — a line each, the buttons stay`, { skip: SKIP }, (t) => {
    const { before, after } = render(t, width, { lora, clicks: ["save", "unload"] });
    assertTogether(after, "with both results shown");
    assertButtonsStay(before, after);
    assertOwnLine(after, "saveStatusText", "the save result");
    assertOwnLine(after, "statusText", "the unload result");
    const [first, second] = [after.statusText, after.saveStatusText].sort((a, b) => a.top - b.top);
    assert.ok(second.top >= first.bottom - 1, "one result per line, not over each other");
    assert.ok(Math.max(after.status.bottom, after.saveStatus.bottom) <= after.row.bottom + 1, "inside the tab row");
  });
}
