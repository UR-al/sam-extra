// notebook.js 의 마운트 — 스크립트 컨테이너를 통째로 2열에 넣고, 칩 줄 자리를 만든다.
//
// 주의: 창을 t.after 로 반드시 닫는다. notebook.js 는 인터벌을 걸어 두므로 닫지 않으면 node --test 가 끝나지 않는다
// (tests/js/notebook_config_lookup.test.mjs:26-38 에 같은 이유가 적혀 있다).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const SCRIPT = readFileSync(path.join(ROOT, "javascript", "notebook.js"), "utf8");

function buildDom(t) {
  const dom = new JSDOM(`<!doctype html><html><body>
    <div id="tab_txt2img">
      <div id="txt2img_toprow"></div>
      <div id="txt2img_extra_tabs"></div>
      <div id="txt2img_settings"></div>
      <div id="txt2img_script_container"><div class="styler">
        <div id="component-164" class="gr-group sam3-slot sam3-slot--sam3 sam3-lane--det"></div>
        <div class="form"><div id="script_list"></div></div>
        <div id="component-1726" class="gr-group"></div>
      </div></div>
      <div id="txt2img_results"><div id="txt2img_gallery"></div></div>
    </div></body></html>`, {
      runScripts: "outside-only", pretendToBeVisual: true, url: "http://127.0.0.1:7860/"
    });
  t.after(() => dom.window.close());
  const { window } = dom;
  window.gradioApp = () => window.document;
  window.onUiLoaded = () => {};
  window.onAfterUiUpdate = () => {};
  window.__sam3NotebookTestHooks = {};          // 이 객체가 있어야 notebook.js 가 테스트 훅을 붙인다
  window.eval(SCRIPT);
  return dom;
}

test("the script container moves whole into the scripts column", (t) => {
  const dom = buildDom(t);
  const { window } = dom;
  const panel = window.document.createElement("details");
  let mounted = 0;
  window.addEventListener("sam3:notebook-mounted", () => { mounted += 1; });
  assert.equal(window.__sam3NotebookTestHooks.mountLayout(panel), true);
  const doc = window.document;
  const scriptsBody = doc.querySelector('[data-column="scripts"] > div');
  const container = doc.querySelector("#txt2img_script_container");
  assert.equal(container.parentElement, scriptsBody);
  assert.equal(doc.querySelector("#script_list").closest("#txt2img_script_container"), container);
  assert.equal(doc.querySelectorAll(".sam3-notebook-script-float").length, 0);
  assert.ok(doc.querySelector('[data-column="parameters"] > div > #txt2img_settings'));
  assert.equal(mounted, 1);
});

test("the chip bar sits between the prompt and the columns", (t) => {
  const dom = buildDom(t);
  const { window } = dom;
  window.__sam3NotebookTestHooks.mountLayout(window.document.createElement("details"));
  const layout = window.document.querySelector("#sam3_notebook_layout");
  const kids = [...layout.children].map((node) => node.id || node.className);
  assert.deepEqual(kids.slice(0, 3), ["sam3-notebook-prompt", "sam3_onbar", "sam3-notebook-columns"]);
});

test("mounting twice keeps one chip bar", (t) => {
  const dom = buildDom(t);
  const { window } = dom;
  const panel = window.document.createElement("details");
  window.__sam3NotebookTestHooks.mountLayout(panel);
  window.__sam3NotebookTestHooks.mountLayout(panel);
  assert.equal(window.document.querySelectorAll("#sam3_onbar").length, 1);
});
