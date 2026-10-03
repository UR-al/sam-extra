// Colorcraft 슬라이더 숫자 칸 범위 풀기 — 실제 javascript/colorcraft_sliders.js 를 jsdom 에 올려 Gradio 4.40 슬라이더
// 모양(.gradio-slider 안의 input[type=number])으로 확인한다. 범위는 Colorcraft 아코디언 안에만.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const SCRIPT = readFileSync(path.join(ROOT, "javascript", "colorcraft_sliders.js"), "utf8");

const slider = (id) => `<div class="gradio-slider"><input type="number" id="${id}" min="-4" max="4" value="0.2"></div>`;
const PAGE = `
<div id="script_txt2img_colorcraft_samextra_accordion">${slider("ours")}</div>
<div id="script_img2img_colorcraft_samextra_accordion">${slider("ours_i2i")}</div>
<div id="script_txt2img_other_accordion">${slider("theirs")}</div>`;

function load(page = PAGE) {
  const dom = new JSDOM(`<!doctype html><html><body>${page}</body></html>`, {
    url: "http://127.0.0.1:7860/",
    runScripts: "outside-only",
  });
  const { window } = dom;
  window.gradioApp = () => window.document;
  window.onUiLoaded = (fn) => fn();
  const observed = [];                       // every node the script's MutationObservers watch
  const observe = window.MutationObserver.prototype.observe;
  window.MutationObserver.prototype.observe = function (target, options) {
    observed.push(target);
    return observe.call(this, target, options);
  };
  window.eval(SCRIPT);
  const el = (id) => window.document.getElementById(id);
  return { window, el, observed };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));   // MutationObserver callbacks

// Counts the DOM work the script does from here on (the test's own lookups happen before this is called).
function countWork(window) {
  const counts = { querySelectorAll: 0, getElementById: 0, matches: 0, removeAttribute: 0 };
  const wrap = (proto, name) => {
    const original = proto[name];
    proto[name] = function (...args) {
      counts[name] += 1;
      return original.apply(this, args);
    };
  };
  wrap(window.Element.prototype, "querySelectorAll");
  wrap(window.Document.prototype, "querySelectorAll");
  wrap(window.Document.prototype, "getElementById");
  wrap(window.Element.prototype, "matches");
  wrap(window.Element.prototype, "removeAttribute");
  return counts;
}

const ACCORDIONS = ["script_txt2img_colorcraft_samextra_accordion", "script_img2img_colorcraft_samextra_accordion"];

test("min/max are stripped inside the Colorcraft accordions only", () => {
  const { el } = load();
  for (const id of ["ours", "ours_i2i"]) {
    assert.equal(el(id).hasAttribute("min"), false, id);
    assert.equal(el(id).hasAttribute("max"), false, id);
  }
  assert.equal(el("theirs").getAttribute("min"), "-4");
  assert.equal(el("theirs").getAttribute("max"), "4");
});

test("sliders added later are unclamped too", async () => {
  const { window, el } = load();
  window.document.getElementById("script_txt2img_colorcraft_samextra_accordion")
    .insertAdjacentHTML("beforeend", slider("late"));
  await flush();
  assert.equal(el("late").hasAttribute("min"), false);
});

test("only the two Colorcraft accordions are observed", () => {
  const { observed } = load();
  assert.deepEqual(observed.map((node) => node.id), ACCORDIONS);
});

test("DOM changes outside the accordions cost nothing", async () => {
  const { window, el } = load();
  const doc = window.document;
  const other = el("script_txt2img_other_accordion");
  const theirs = el("theirs");
  const counts = countWork(window);
  for (let i = 0; i < 50; i += 1) {                      // progress text, live previews, other panels …
    doc.body.insertAdjacentHTML("beforeend", `<div class="progress">${i}%</div>`);
    other.insertAdjacentHTML("beforeend", slider(`elsewhere${i}`));
    doc.body.lastElementChild.remove();
  }
  theirs.setAttribute("min", "-1");
  await flush();
  assert.deepEqual(counts, { querySelectorAll: 0, getElementById: 0, matches: 0, removeAttribute: 0 });
  assert.equal(theirs.getAttribute("min"), "-1");
  assert.equal(el("elsewhere0").getAttribute("min"), "-4");
});

test("a change inside an accordion touches only what changed", async () => {
  const many = Array.from({ length: 400 }, (_, i) => slider(`s${i}`)).join("");
  const { window, el } = load(`<div id="${ACCORDIONS[0]}">${many}</div><div id="${ACCORDIONS[1]}">${many}</div>`);
  const scope = el(ACCORDIONS[0]);
  const counts = countWork(window);
  scope.insertAdjacentHTML("beforeend", slider("late"));
  await flush();
  // the added slider only — not the ~800 fields of both panels again (upstream's rescan: 1602 removals)
  assert.deepEqual(counts, { querySelectorAll: 1, getElementById: 0, matches: 1, removeAttribute: 2 });
  assert.equal(el("late").hasAttribute("min"), false);
  assert.equal(el("late").hasAttribute("max"), false);
});

test("a min/max set again inside an accordion is removed again", async () => {
  const { window, el } = load();
  const ours = el("ours");
  const counts = countWork(window);
  ours.setAttribute("max", "4");                          // e.g. Svelte re-applying the prop
  await flush();
  assert.equal(ours.hasAttribute("max"), false);
  assert.equal(counts.removeAttribute, 1);
  assert.equal(counts.querySelectorAll, 0);
});

test("an accordion that appears after load is picked up; then the page is no longer watched", async () => {
  const { window, el, observed } = load(`<div id="script_txt2img_other_accordion">${slider("theirs")}</div>`);
  const doc = window.document;
  assert.deepEqual(observed, [doc], "while an accordion is missing, the page is watched for it");
  doc.body.insertAdjacentHTML("beforeend", `<div id="${ACCORDIONS[0]}">${slider("ours")}</div>`);
  await flush();
  assert.equal(el("ours").hasAttribute("min"), false);
  doc.body.insertAdjacentHTML("beforeend", `<div id="${ACCORDIONS[1]}">${slider("ours_i2i")}</div>`);
  await flush();
  assert.equal(el("ours_i2i").hasAttribute("max"), false);
  assert.deepEqual(observed.slice(1).map((node) => node.id), ACCORDIONS);
  const counts = countWork(window);
  doc.body.insertAdjacentHTML("beforeend", "<div>elsewhere</div>");
  await flush();
  assert.deepEqual(counts, { querySelectorAll: 0, getElementById: 0, matches: 0, removeAttribute: 0 },
    "found both: the page-wide finder has stopped");
});

test("blur keeps a value beyond the range and repairs a broken one", () => {
  const { window, el } = load();
  const seen = [];
  const input = el("ours");
  input.addEventListener("blur", () => seen.push("gradio blur"));
  input.addEventListener("input", () => seen.push("input " + input.value));
  input.value = "7.5";
  input.dispatchEvent(new window.FocusEvent("blur"));
  assert.equal(input.value, "7.5");
  input.value = "1e";
  input.dispatchEvent(new window.FocusEvent("blur"));
  assert.equal(input.value, "0");
  assert.deepEqual(seen, ["input 0"], "Gradio's own blur (its clamp) never runs");
});

test("Enter is kept from Gradio, Ctrl+Enter passes through after the repair", () => {
  const { window, el } = load();
  const seen = [];
  const input = el("ours");
  input.addEventListener("keydown", (e) => seen.push(e.ctrlKey ? "ctrl+enter" : "enter"));
  input.value = "-12";
  input.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  assert.equal(input.value, "-12");
  input.value = "+";
  input.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter", ctrlKey: true, bubbles: true }));
  assert.equal(input.value, "0");
  assert.deepEqual(seen, ["ctrl+enter"]);
});

test("inputs outside the accordion are left to Gradio", () => {
  const { window, el } = load();
  const seen = [];
  const input = el("theirs");
  input.addEventListener("blur", () => seen.push("gradio blur"));
  input.value = "abc";                                       // a number input reads back "" for this
  input.dispatchEvent(new window.FocusEvent("blur"));
  assert.equal(input.value, "", "not repaired to 0 outside Colorcraft");
  assert.deepEqual(seen, ["gradio blur"]);
});
