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

function buildDropdown(choiceCount = 163, pageSize = 60, inputAttrs = "") {
  const dom = new JSDOM(
    `<!doctype html><html><body>
      <div class="gradio-container">
        <div id="long_axis" class="gradio-dropdown">
          <div class="container"><input role="listbox" value="Axis 001"${inputAttrs}></div>
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
  // jsdom has no layout: openPopover scrolls the selected choice into view
  window.HTMLElement.prototype.scrollIntoView = function () {};
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

// The proxy hides the Gradio input, so a description the page gave that input
// (Colorcraft's "편집 중: …" from colorcraft_editor.js) must reach the trigger
// that replaces it for keyboard and screen-reader users.
test("the trigger carries the hidden input's description, and none when it has none", () => {
  const described = buildDropdown(3, 60, ' aria-describedby="axis_note axis_hint"');
  assert.equal(
    dropdownState(described).trigger.getAttribute("aria-describedby"),
    "axis_note axis_hint"
  );
  assert.equal(dropdownState(described).trigger.getAttribute("aria-label"), "Long axis");
  described.window.close();

  const plain = buildDropdown(3);
  assert.equal(dropdownState(plain).trigger.hasAttribute("aria-describedby"), false);
  plain.window.close();
});

// ---------------------------------------------------------------------------
// One column (decided with the user): every fast dropdown — Sampler, Schedule
// type, Hires, the XYZ X/Y/Z type and value lists, any Gradio dropdown the
// global scan converts — shows ONE column of choices in list order and scrolls
// when it does not fit, however wide the screen is. Before, positionPopover()
// spread a long list over up to eight side-by-side columns on a wide monitor.
// ---------------------------------------------------------------------------

const ROW = 44;      // positionPopover's option row height
const CHROME = 64;   // search box + padding
const GUTTER = 8;
const GAP = 4;

// Opens the dropdown on a window of the given size with the trigger at `rect`
// (jsdom has no layout, so the trigger's box is stubbed).
function openAt(dom, { width, height, rect }) {
  const { window } = dom;
  Object.defineProperty(window, "innerWidth", { configurable: true, value: width });
  Object.defineProperty(window, "innerHeight", { configurable: true, value: height });
  const state = dropdownState(dom);
  const box = { left: 200, width: 320, height: 40, ...rect };
  box.right = box.left + box.width;
  if (box.bottom === undefined) box.bottom = box.top + box.height;
  state.trigger.getBoundingClientRect = () => box;
  state.trigger.click();
  return { ...state, box, popover: dom.window.document.querySelector(".sam3-fast-dropdown-popover") };
}

function px(value) {
  assert.match(value, /^\d+px$/);
  return Number.parseInt(value, 10);
}

function assertOneColumn(state, box) {
  assert.equal(state.list.style.gridTemplateColumns, "minmax(0, 1fr)", "one column of choices");
  assert.equal(px(state.popover.style.width), Math.max(box.width, 280), "as wide as the trigger, not N columns");
}

test("a long list on a wide screen is one column that scrolls, not side-by-side columns", (t) => {
  const dom = buildDropdown(163, 200);          // every choice rendered, no further page
  t.after(() => dom.window.close());
  const state = openAt(dom, { width: 3840, height: 1080, rect: { top: 100 } });
  assert.equal(optionCount(state.list), 163);
  assertOneColumn(state, state.box);
  assert.equal(state.list.getAttribute("data-has-more"), "false");
  assert.equal(state.list.getAttribute("data-scroll"), "true");
  assert.equal(state.list.style.overflowY, "auto");
  // Opens below the trigger and fills the room down to the gutter, whole rows only:
  // the panel is exactly the search box plus those rows.
  const top = px(state.popover.style.top);
  const panel = px(state.popover.style.maxHeight);
  assert.equal(top, state.box.bottom + GAP);
  const listHeight = px(state.list.style.maxHeight);
  assert.equal(listHeight % ROW, 0);
  assert.equal(listHeight, Math.floor((1080 - GUTTER - top - CHROME) / ROW) * ROW);
  assert.equal(panel, CHROME + listHeight);
  assert.ok(top + panel <= 1080 - GUTTER && top + panel > 1080 - GUTTER - ROW);
  assert.ok(listHeight < 163 * ROW);
});

test("paging keeps the single column, and the last page still fits in it", (t) => {
  const dom = buildDropdown(163, 60);
  t.after(() => dom.window.close());
  const state = openAt(dom, { width: 2560, height: 1440, rect: { top: 120, width: 300 } });
  Object.defineProperty(state.list, "clientHeight", { configurable: true, value: 440 });
  Object.defineProperty(state.list, "scrollHeight", {
    configurable: true,
    get() { return optionCount(state.list) * ROW; },
  });
  assert.equal(optionCount(state.list), 60);
  assert.equal(state.list.getAttribute("data-has-more"), "true");
  assert.equal(state.list.getAttribute("data-scroll"), "true", "a real scrollbar while more pages exist");
  assertOneColumn(state, state.box);
  scrollToBottom(dom, state.list);
  assert.equal(optionCount(state.list), 120);
  assertOneColumn(state, state.box);
  scrollToBottom(dom, state.list);
  assert.equal(optionCount(state.list), 163);
  assert.equal(state.list.getAttribute("data-has-more"), "false");
  assertOneColumn(state, state.box);
  assert.equal(state.list.getAttribute("data-scroll"), "true");
});

test("while more pages exist one rendered row stays below the fold even on a tall screen", (t) => {
  const dom = buildDropdown(40, 10);            // the smallest page size: 10 rows would fit
  t.after(() => dom.window.close());
  const state = openAt(dom, { width: 1920, height: 2160, rect: { top: 100 } });
  assert.equal(optionCount(state.list), 10);
  assert.equal(state.list.getAttribute("data-has-more"), "true");
  assert.equal(px(state.list.style.maxHeight), 9 * ROW);
  assert.equal(state.list.getAttribute("data-scroll"), "true");
  assertOneColumn(state, state.box);
});

test("a short list opens below at its natural height without a scrollbar", (t) => {
  const dom = buildDropdown(3);
  t.after(() => dom.window.close());
  const state = openAt(dom, { width: 1920, height: 1080, rect: { top: 100, width: 180 } });
  assertOneColumn(state, state.box);           // narrow trigger: 280px minimum
  assert.equal(px(state.popover.style.top), state.box.bottom + GAP);
  assert.equal(px(state.popover.style.maxHeight), CHROME + 3 * ROW);
  assert.equal(px(state.list.style.maxHeight), 3 * ROW);
  assert.equal(state.list.getAttribute("data-scroll"), "false");
  assert.equal(state.list.style.overflowY, "visible");
});

test("near the bottom of the screen the list opens above the trigger", (t) => {
  const short = buildDropdown(3);
  t.after(() => short.window.close());
  const a = openAt(short, { width: 1920, height: 1080, rect: { top: 980 } });
  assert.equal(px(a.popover.style.top) + px(a.popover.style.maxHeight), a.box.top - GAP);
  assert.equal(px(a.popover.style.maxHeight), CHROME + 3 * ROW);
  assert.equal(a.list.getAttribute("data-scroll"), "false");

  // Too long for either side: the roomier side (above) is filled with whole rows
  // and scrolls, and the panel still ends right above the trigger (no slack
  // between the last row and the trigger).
  const long = buildDropdown(163, 200);
  t.after(() => long.window.close());
  const b = openAt(long, { width: 1920, height: 1080, rect: { top: 900 } });
  assertOneColumn(b, b.box);
  const bTop = px(b.popover.style.top);
  const bPanel = px(b.popover.style.maxHeight);
  const bList = px(b.list.style.maxHeight);
  assert.equal(bList, Math.floor((b.box.top - GAP - GUTTER - CHROME) / ROW) * ROW);
  assert.equal(bPanel, CHROME + bList);
  assert.equal(bTop + bPanel, b.box.top - GAP);
  assert.ok(bTop >= GUTTER && bTop < GUTTER + ROW);
  assert.equal(b.list.getAttribute("data-scroll"), "true");
});

test("a phone-width window keeps one column inside the gutters", (t) => {
  const dom = buildDropdown(163, 60);
  t.after(() => dom.window.close());
  const state = openAt(dom, { width: 360, height: 740, rect: { top: 60, left: 40, width: 400 } });
  assert.equal(state.list.style.gridTemplateColumns, "minmax(0, 1fr)");
  assert.equal(px(state.popover.style.width), 360 - 2 * GUTTER);
  assert.equal(px(state.popover.style.left), GUTTER);
});

test("arrow keys walk the single column in list order", async (t) => {
  const dom = buildDropdown(12, 60);
  t.after(() => dom.window.close());
  const { window } = dom;
  const doc = window.document;
  const state = dropdownState(dom);
  const key = (target, name) => target.dispatchEvent(
    new window.KeyboardEvent("keydown", { key: name, bubbles: true, cancelable: true })
  );
  const tick = () => new Promise((resolve) => setTimeout(resolve, 5));
  key(state.trigger, "ArrowDown");             // opens and focuses the selected choice
  await tick();
  assert.equal(doc.activeElement.textContent, "Axis 001");
  key(doc.activeElement, "ArrowDown");
  assert.equal(doc.activeElement.textContent, "Axis 002");
  key(doc.activeElement, "End");
  assert.equal(doc.activeElement.textContent, "Axis 012");
  key(doc.activeElement, "ArrowDown");          // wraps to the top
  assert.equal(doc.activeElement.textContent, "Axis 001");
  key(doc.activeElement, "ArrowUp");            // and back to the bottom
  assert.equal(doc.activeElement.textContent, "Axis 012");
  key(doc.activeElement, "Home");
  assert.equal(doc.activeElement.textContent, "Axis 001");
  const options = [...state.list.querySelectorAll(".sam3-fast-dropdown-option")].map((o) => o.textContent);
  assert.deepEqual(options, Array.from({ length: 12 }, (_, i) => `Axis ${String(i + 1).padStart(3, "0")}`));
  assert.equal(state.list.getAttribute("role"), "listbox");
  assert.equal(state.trigger.getAttribute("aria-expanded"), "true");
  key(doc.activeElement, "Escape");
  assert.equal(doc.activeElement, state.trigger);
  assert.equal(state.trigger.getAttribute("aria-expanded"), "false");
});

// One column makes ArrowDown the normal way through a long list. Reaching the
// end of a page (a browser scrolls the focused row into view) appends the next
// page by rebuilding the list; the focused choice must survive that rebuild,
// or the focus falls to <body> and the keys stop working.
test("paging keeps the keyboard focus on its choice, and ArrowDown goes on into the next page", async (t) => {
  const dom = buildDropdown(163, 60);
  t.after(() => dom.window.close());
  const { window } = dom;
  const doc = window.document;
  const state = dropdownState(dom);
  const key = (target, name) => target.dispatchEvent(
    new window.KeyboardEvent("keydown", { key: name, bubbles: true, cancelable: true })
  );
  const tick = () => new Promise((resolve) => setTimeout(resolve, 5));
  key(state.trigger, "ArrowDown");
  await tick();
  Object.defineProperty(state.list, "clientHeight", { configurable: true, value: 440 });
  Object.defineProperty(state.list, "scrollHeight", {
    configurable: true,
    get() { return optionCount(state.list) * ROW; },
  });
  key(doc.activeElement, "End");                 // the last choice of the first page
  assert.equal(doc.activeElement.textContent, "Axis 060");
  scrollToBottom(dom, state.list);
  assert.equal(optionCount(state.list), 120);
  assert.equal(doc.activeElement.parentNode, state.list, "still a choice of the list, not <body>");
  assert.equal(doc.activeElement.textContent, "Axis 060");
  key(doc.activeElement, "ArrowDown");
  assert.equal(doc.activeElement.textContent, "Axis 061");
  key(doc.activeElement, "ArrowDown");
  assert.equal(doc.activeElement.textContent, "Axis 062");

  // A choices refresh rebuilds the list too: the same choice keeps the focus
  // where it is still listed, otherwise the row at the same place gets it.
  const wrapper = doc.querySelector("#long_axis");
  const choices = Array.from({ length: 163 }, (_, i) => `Axis ${String(i + 1).padStart(3, "0")}`);
  key(doc.activeElement, "Home");
  key(doc.activeElement, "ArrowDown");
  key(doc.activeElement, "ArrowDown");
  assert.equal(doc.activeElement.textContent, "Axis 003");
  wrapper.__sam3FastDropdownSetChoices(["Axis 000", ...choices]);
  assert.equal(doc.activeElement.parentNode, state.list);
  assert.equal(doc.activeElement.textContent, "Axis 003");
  wrapper.__sam3FastDropdownSetChoices(choices.filter((choice) => choice !== "Axis 003"));
  assert.equal(doc.activeElement.parentNode, state.list);
  assert.equal(doc.activeElement.textContent, "Axis 005");   // row 4 (Axis 000 is gone too)
});

test("a window too short for four rows on either side opens at the top gutter, as tall as it allows", (t) => {
  const dom = buildDropdown(163, 60);
  t.after(() => dom.window.close());
  // 800x420, the trigger at 196..240: 168px below, 184px above — neither holds
  // the search box and four rows (240px), so the old top-gutter fallback.
  const state = openAt(dom, { width: 800, height: 420, rect: { top: 196, height: 44 } });
  assert.equal(state.box.bottom, 240);
  assertOneColumn(state, state.box);
  assert.equal(px(state.popover.style.top), GUTTER);
  assert.equal(px(state.list.style.maxHeight), 7 * ROW);           // (420 - 2 * 8 - 64) / 44 whole rows
  assert.equal(px(state.popover.style.maxHeight), CHROME + 7 * ROW);
  assert.ok(GUTTER + px(state.popover.style.maxHeight) <= 420 - GUTTER);
  assert.equal(state.list.getAttribute("data-scroll"), "true");
});

test("a list that fits on both sides opens below, and so does a long one with the same room on both", (t) => {
  const short = buildDropdown(3);
  t.after(() => short.window.close());
  const a = openAt(short, { width: 1920, height: 1080, rect: { top: 500 } });
  assert.ok(a.box.top - GAP - GUTTER >= CHROME + 3 * ROW, "it would fit above too");
  assert.equal(px(a.popover.style.top), a.box.bottom + GAP);
  assert.equal(px(a.popover.style.maxHeight), CHROME + 3 * ROW);

  const long = buildDropdown(163, 200);
  t.after(() => long.window.close());
  const b = openAt(long, { width: 1920, height: 1080, rect: { top: (1080 - 40) / 2 } });
  const room = b.box.top - GAP - GUTTER;
  assert.equal(room, 1080 - GUTTER - (b.box.bottom + GAP), "the same room above and below");
  assert.equal(px(b.popover.style.top), b.box.bottom + GAP);
  assert.equal(px(b.list.style.maxHeight), Math.floor((room - CHROME) / ROW) * ROW);
  assert.equal(b.list.getAttribute("data-scroll"), "true");
});
