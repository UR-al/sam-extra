// Behavioural coverage for long SAM Extra fast-dropdown lists.
//
// This loads the real notebook.js implementation into jsdom through its
// opt-in test seam. The regression was not search itself: the first 60 entries
// were the only DOM options, so a user who did not know a hidden XYZ axis name
// had no way to discover it. Reaching the list bottom must append every page.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const SCRIPT = readFileSync(path.join(ROOT, "javascript", "notebook.js"), "utf8");

function buildDropdown(choiceCount = 163, pageSize = 60) {
  const dom = new JSDOM(
    `<!doctype html><html><body>
      <div class="gradio-container">
        <div id="long_axis" class="gradio-dropdown">
          <div class="container"><input role="listbox" value="Axis 001"></div>
        </div>
      </div>
    </body></html>`,
    { runScripts: "outside-only", pretendToBeVisual: true }
  );
  const { window } = dom;
  const choices = Array.from(
    { length: choiceCount },
    (_, index) => `Axis ${String(index + 1).padStart(3, "0")}`
  );
  window.gradioApp = () => window.document;
  window.onUiLoaded = () => {};
  window.opts = { sam3_fast_dropdown_visible_choices: pageSize };
  window.gradio_config = {
    components: [{ id: 7, props: { elem_id: "long_axis", choices } }]
  };
  window.CSS = window.CSS || {};
  window.CSS.escape = window.CSS.escape || ((value) => String(value));
  window.__sam3NotebookTestHooks = {};
  window.eval(SCRIPT);
  assert.equal(
    window.__sam3NotebookTestHooks.installFastDropdown("long_axis", "Long axis"),
    true
  );
  return dom;
}

function dropdownState(dom) {
  const doc = dom.window.document;
  return {
    trigger: doc.querySelector("#long_axis .sam3-fast-dropdown-trigger"),
    list: doc.querySelector(".sam3-fast-dropdown-options"),
    status: doc.querySelector(".sam3-fast-dropdown-empty"),
  };
}

function optionCount(list) {
  return list.querySelectorAll(".sam3-fast-dropdown-option").length;
}

function scrollToBottom(dom, list) {
  list.scrollTop = Math.max(0, list.scrollHeight - list.clientHeight);
  list.dispatchEvent(new dom.window.Event("scroll", { bubbles: false }));
}

test("long dropdown appends every page when scrolled to the bottom", () => {
  const dom = buildDropdown();
  const state = dropdownState(dom);
  state.trigger.click();

  Object.defineProperty(state.list, "clientHeight", {
    configurable: true,
    value: 440,
  });
  Object.defineProperty(state.list, "scrollHeight", {
    configurable: true,
    get() { return optionCount(state.list) * 44; },
  });

  assert.equal(optionCount(state.list), 60);
  assert.match(state.status.textContent, /163개 중 60개 표시/);
  assert.match(state.status.textContent, /아래로 스크롤하면 더 표시/);

  scrollToBottom(dom, state.list);
  assert.equal(optionCount(state.list), 120);
  assert.match(state.status.textContent, /163개 중 120개 표시/);

  scrollToBottom(dom, state.list);
  assert.equal(optionCount(state.list), 163);
  assert.equal(state.status.hidden, true);
  assert.equal(
    state.list.lastElementChild.textContent,
    "Axis 163",
    "the formerly undiscoverable final choice must exist in the DOM"
  );
  dom.window.close();
});

test("typing a new query resets pagination to one configured page", () => {
  const dom = buildDropdown();
  const state = dropdownState(dom);
  state.trigger.click();
  Object.defineProperty(state.list, "clientHeight", { value: 440 });
  Object.defineProperty(state.list, "scrollHeight", {
    get() { return optionCount(state.list) * 44; },
  });

  scrollToBottom(dom, state.list);
  assert.equal(optionCount(state.list), 120);

  const search = dom.window.document.querySelector(".sam3-fast-dropdown-search");
  search.value = "Axis 1";
  search.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
  // Axis 100..163 provide 64 matches; a new query starts again at page size 60.
  assert.equal(optionCount(state.list), 60);
  assert.match(state.status.textContent, /64개 중 60개 표시/);
  dom.window.close();
});
