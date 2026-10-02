// Behavioural coverage for resolving a dropdown's Gradio config entry.
//
// Forge's page config holds ~5,600 components. The fast-dropdown scan runs
// after every UI update and used to walk that whole list once per dropdown per
// lookup (~4 ms of a ~6 ms pass measured on a real page). Lookups must stay
// correct when the config grows or is replaced, and must keep resolving a
// duplicated elem_id to its first component as the linear search did.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const SCRIPT = readFileSync(path.join(ROOT, "javascript", "notebook.js"), "utf8");

function dropdownHtml(ids) {
  return ids.map((id) =>
    `<div id="${id}" class="gradio-dropdown">
       <div class="container"><input role="listbox" value=""></div>
     </div>`
  ).join("");
}

// Installed fast dropdowns keep an interval running, so the window is always
// closed through the test context; otherwise a failing assertion would leave
// the timers alive and the test process would never exit.
function boot(t, ids, components) {
  const dom = new JSDOM(
    `<!doctype html><html><body>
      <div class="gradio-container">${dropdownHtml(ids)}</div>
    </body></html>`,
    { runScripts: "outside-only", pretendToBeVisual: true }
  );
  t.after(() => dom.window.close());
  const { window } = dom;
  window.gradioApp = () => window.document;
  window.onUiLoaded = () => {};
  window.opts = { sam3_fast_dropdown_visible_choices: 60 };
  window.gradio_config = { components };
  window.CSS = window.CSS || {};
  window.CSS.escape = window.CSS.escape || ((value) => String(value));
  window.__sam3NotebookTestHooks = {};
  window.eval(SCRIPT);
  return dom;
}

// Each installed dropdown appends its own popover to <body>, so options are
// read from the popover that belongs to this dropdown only.
function optionTexts(dom, id) {
  const doc = dom.window.document;
  doc.querySelector(`#${id} .sam3-fast-dropdown-trigger`).click();
  return Array.from(
    doc.querySelectorAll(`#sam3-fast-dropdown-${id} .sam3-fast-dropdown-option`)
  ).map((option) => option.textContent);
}

test("installing several dropdowns reads the config once, not per lookup", (t) => {
  const counter = { reads: 0 };
  const counted = (props) => {
    const component = {};
    Object.defineProperty(component, "props", {
      enumerable: true,
      get() { counter.reads++; return props; },
    });
    return component;
  };
  const ids = ["dd_0", "dd_1", "dd_2", "dd_3", "dd_4"];
  const components = [
    ...Array.from({ length: 3000 }, (_, i) => counted({ elem_id: `filler_${i}` })),
    ...ids.map((id) => counted({ elem_id: id, choices: ["A", "B"] })),
  ];
  const dom = boot(t, ids, components);
  counter.reads = 0;

  for (const id of ids) {
    assert.equal(dom.window.__sam3NotebookTestHooks.installFastDropdown(id, id), true);
  }

  // One pass over the 3,005 entries builds the index; a per-lookup scan would
  // need at least one full pass per installed dropdown (5 x 3,005).
  assert.ok(
    counter.reads < 2 * components.length,
    `config props were read ${counter.reads} times for ${components.length} components`
  );
});

test("a component appended to the config later is still found", (t) => {
  const components = [{ props: { elem_id: "early", choices: ["A"] } }];
  const dom = boot(t, ["early", "late"], components);
  const hooks = dom.window.__sam3NotebookTestHooks;
  assert.equal(hooks.installFastDropdown("early", "early"), true);
  assert.equal(hooks.installFastDropdown("late", "late"), false);

  components.push({ props: { elem_id: "late", choices: ["Late choice"] } });

  assert.equal(hooks.installFastDropdown("late", "late"), true);
  assert.deepEqual(optionTexts(dom, "late"), ["Late choice"]);
});

test("a replaced page config is used for later lookups", (t) => {
  const dom = boot(t, ["early", "axis"], [{ props: { elem_id: "early", choices: ["A"] } }]);
  const hooks = dom.window.__sam3NotebookTestHooks;
  assert.equal(hooks.installFastDropdown("early", "early"), true);

  dom.window.gradio_config = {
    components: [{ props: { elem_id: "axis", choices: ["New config choice"] } }],
  };

  assert.equal(hooks.installFastDropdown("axis", "axis"), true);
  assert.deepEqual(optionTexts(dom, "axis"), ["New config choice"]);
});

test("a duplicated elem_id resolves to its first component", (t) => {
  const dom = boot(t, ["dup"], [
    { props: { elem_id: "dup", choices: ["first"] } },
    { props: { elem_id: "dup", choices: ["second"] } },
  ]);
  assert.equal(dom.window.__sam3NotebookTestHooks.installFastDropdown("dup", "dup"), true);
  assert.deepEqual(optionTexts(dom, "dup"), ["first"]);
});
