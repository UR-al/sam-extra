// Notebook Apply/Undo queue and LoRA Manager message sender (2026-10-02 review, ui/02 and ui/03).
// JSDOM + simulated UI controls on the shipped scripts; no Forge server/network/GPU.
// Ported from docs/review_proposals_20261002/ui/verify_ui.mjs (which read the proposed copies).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const notebookSource = readFileSync(path.join(ROOT, "javascript/notebook.js"), "utf8");
const loraSource = readFileSync(path.join(ROOT, "javascript/lora_manager.js"), "utf8");
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
function deferred() {
  let resolve;
  const promise = new Promise(r => { resolve = r; });
  return { promise, resolve };
}
const preset = (name, value) => ({name, entries: [
  {target: "sampling_method", value}, {target: "width", value},
]});

function notebookHarness() {
  const dom = new JSDOM(`<!doctype html><details id="sam3_notebook_panel">
    <span data-notebook-status></span><input data-notebook-search>
    <button data-notebook-undo hidden>Undo</button><div data-notebook-presets></div>
  </details>`, {runScripts: "outside-only", url: "http://127.0.0.1:7860/"});
  const w = dom.window;
  w.gradioApp = () => w.document;
  w.onUiLoaded = () => {};
  w.console.error = () => {};
  w.__sam3NotebookTestHooks = {};
  const anchor = "        window.__sam3NotebookTestHooks.mountLayout = mountLayout;";
  assert.equal(notebookSource.split(anchor).length, 2);
  // Test-only, in-memory access to closures; never written to proposed source.
  const seam = `${anchor}
        window.__sam3NotebookTestHooks.applyPreset = applyPreset;
        window.__sam3NotebookTestHooks.applyUndo = applyUndo;
        window.__sam3NotebookTestHooks.bindPanel = bindPanelReferences;
        window.__sam3NotebookTestHooks.presetCard = presetCard;
        window.__sam3NotebookTestHooks.handlePanelClick = handlePanelClick;
        window.__sam3NotebookTestHooks.waitForQueue = function () { return applyChain; };
        window.__sam3NotebookTestHooks.pending = function () { return pendingApplyOperations; };
        window.__sam3NotebookTestHooks.setApplyStubs = function (stubs) {
            preflightEntries = stubs.preflight;
            captureApplyState = stubs.capture;
            applyEntry = stubs.apply;
            restoreApplyState = stubs.restore;
        };`;
  w.eval(notebookSource.replace(anchor, seam));
  const hooks = w.__sam3NotebookTestHooks;
  const panel = w.document.querySelector("details");
  hooks.bindPanel(panel);
  const state = {sampling_method: "old", width: "old"};
  const log = [];
  const stubs = {
    preflight() {},
    capture(entries) { return {entries: entries.map(e => ({target: e.target, value: state[e.target]}))}; },
    async apply(entry) { log.push("apply:" + entry.target + ":" + entry.value); state[entry.target] = entry.value; return false; },
    async restore(snapshot) { log.push("restore"); snapshot.entries.forEach(e => { state[e.target] = e.value; }); },
  };
  hooks.setApplyStubs(stubs);
  return {dom, hooks, state, log, stubs, panel};
}

test("Notebook serializes Apply and snapshots queued preset values", async () => {
  const h = notebookHarness();
  try {
    const gate = deferred();
    const originalApply = h.stubs.apply;
    h.stubs.apply = async entry => {
      await originalApply(entry);
      if (entry.target === "sampling_method" && entry.value === "A") await gate.promise;
      return false;
    };
    h.hooks.setApplyStubs(h.stubs);
    const a = preset("A", "A"), b = preset("B", "B");
    const pA = h.hooks.applyPreset(a), pB = h.hooks.applyPreset(b);
    b.entries[1].value = "edited while queued";
    await tick();
    assert.deepEqual(h.log, ["apply:sampling_method:A"]);
    assert.equal(h.hooks.pending(), 2);
    assert.equal(h.panel.querySelector("[data-notebook-undo]").disabled, true);
    assert.equal(h.panel.getAttribute("aria-busy"), "true");
    gate.resolve();
    await Promise.all([pA, pB]);
    assert.deepEqual(h.state, {sampling_method: "B", width: "B"});
    assert.equal(h.hooks.pending(), 0);
    assert.equal(h.panel.querySelector("[data-notebook-undo]").disabled, false);
    assert.equal(h.panel.getAttribute("aria-busy"), "false");
  } finally { h.dom.window.close(); }
});

test("Notebook queued Undo resolves the completed preceding Apply state", async () => {
  const h = notebookHarness();
  try {
    const gate = deferred();
    const originalApply = h.stubs.apply;
    h.stubs.apply = async entry => {
      await originalApply(entry);
      if (entry.target === "sampling_method") await gate.promise;
      return false;
    };
    h.hooks.setApplyStubs(h.stubs);
    const pA = h.hooks.applyPreset(preset("A", "A"));
    const pUndo = h.hooks.applyUndo();
    await tick();
    assert.equal(h.log.includes("restore"), false);
    gate.resolve();
    await Promise.all([pA, pUndo]);
    assert.deepEqual(h.state, {sampling_method: "old", width: "old"});
    assert.equal(h.panel.querySelector("[data-notebook-undo]").hidden, true);
  } finally { h.dom.window.close(); }
});

test("Notebook completes rollback before starting the next queued Apply", async () => {
  const h = notebookHarness();
  try {
    const restoreGate = deferred();
    const originalApply = h.stubs.apply, originalRestore = h.stubs.restore;
    h.stubs.apply = async entry => {
      if (entry.target === "width" && entry.value === "A") throw new Error("simulated control failure");
      return originalApply(entry);
    };
    h.stubs.restore = async snapshot => {
      h.log.push("rollback-start");
      await restoreGate.promise;
      await originalRestore(snapshot);
    };
    h.hooks.setApplyStubs(h.stubs);
    const pA = h.hooks.applyPreset(preset("A", "A")), pB = h.hooks.applyPreset(preset("B", "B"));
    await tick();
    assert.equal(h.log.includes("rollback-start"), true);
    assert.equal(h.log.some(item => item.endsWith(":B")), false);
    assert.equal(h.hooks.pending(), 2);
    restoreGate.resolve();
    await Promise.all([pA, pB]);
    assert.deepEqual(h.state, {sampling_method: "B", width: "B"});
    assert.equal(h.hooks.pending(), 0);
  } finally { h.dom.window.close(); }
});

test("Notebook actual Apply button and delegated Undo handler use the queue", async () => {
  const h = notebookHarness();
  try {
    const card = h.hooks.presetCard(preset("A", "A"));
    h.panel.querySelector("[data-notebook-presets]").appendChild(card);
    const apply = card.querySelector("[data-notebook-apply]");
    apply.click();
    assert.equal(apply.disabled, true);
    const undoButton = h.panel.querySelector("[data-notebook-undo]");
    const ev = {target: undoButton, preventDefault() {}, stopPropagation() {}};
    h.hooks.handlePanelClick(ev);
    assert.equal(h.hooks.pending(), 2);
    await h.hooks.waitForQueue();
    assert.deepEqual(h.state, {sampling_method: "old", width: "old"});
    assert.equal(apply.disabled, false);
  } finally { h.dom.window.close(); }
});

test("Notebook releases queue after apply and rollback both fail", async () => {
  const h = notebookHarness();
  try {
    const originalApply = h.stubs.apply;
    h.stubs.apply = async entry => {
      if (entry.value === "A") throw new Error("apply failure");
      return originalApply(entry);
    };
    h.stubs.restore = async () => { throw new Error("rollback failure"); };
    h.hooks.setApplyStubs(h.stubs);
    await Promise.all([h.hooks.applyPreset(preset("A", "A")), h.hooks.applyPreset(preset("B", "B"))]);
    assert.deepEqual(h.state, {sampling_method: "B", width: "B"});
    assert.equal(h.hooks.pending(), 0);
    assert.equal(h.panel.querySelector("[data-notebook-undo]").disabled, false);
  } finally { h.dom.window.close(); }
});

function strip(tab) {
  return `<div id="${tab}_extra_tabs"><div class="tab-nav"><button class="selected">Generation</button></div>
    <div class="tabitem" id="${tab}_generation">Generation</div></div>`;
}
async function loraHarness() {
  const dom = new JSDOM(`<!doctype html>${strip("txt2img")}${strip("img2img")}
    <div id="txt2img_prompt"><label><textarea>base</textarea></label></div>`,
    {runScripts: "outside-only", url: "http://127.0.0.1:7860/"});
  const w = dom.window;
  w.gradioApp = () => w.document;
  w.onUiLoaded = fn => fn();
  w.console.log = () => {};
  w.eval(loraSource);
  for (let i = 0; i < 30; i++) {
    if (w.document.querySelector("#txt2img_loramanager")) break;
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  const frame = w.document.querySelector("#txt2img_loramanager iframe");
  const otherFrame = w.document.querySelector("#img2img_loramanager iframe");
  assert.ok(frame);
  assert.ok(otherFrame);
  frame.src = "http://127.0.0.1:8911/loras";
  frame.setAttribute("data-loaded", "1");
  otherFrame.src = "http://127.0.0.1:9022/loras";
  otherFrame.setAttribute("data-loaded", "1");
  const textarea = w.document.querySelector("#txt2img_prompt textarea");
  function send({source = frame.contentWindow, origin = "http://127.0.0.1:8911", text = "<lora:test:1>", replace = false} = {}) {
    w.dispatchEvent(new w.MessageEvent("message", {source, origin, data: {type: "sam3-add-lora", text, replace}}));
  }
  return {dom, frame, otherFrame, textarea, send};
}

test("LoRA bridge accepts both actual embedded iframe sources with dynamic origins", async () => {
  const h = await loraHarness();
  try {
    h.send();
    assert.equal(h.textarea.value, "base, <lora:test:1>");
    h.send({source: h.otherFrame.contentWindow, origin: "http://127.0.0.1:9022", text: "<lora:second:0.5>"});
    assert.equal(h.textarea.value, "base, <lora:test:1>, <lora:second:0.5>");
  } finally { h.dom.window.close(); }
});

test("LoRA bridge rejects unrelated source, absent source, and wrong origin/port", async () => {
  const h = await loraHarness();
  try {
    h.send({source: h.dom.window});
    h.send({source: null});
    h.send({origin: "https://unrelated.invalid"});
    h.send({origin: "http://127.0.0.1:9022"});
    assert.equal(h.textarea.value, "base");
  } finally { h.dom.window.close(); }
});

test("LoRA bridge rejects disconnected/unloaded frames and invalid message text", async () => {
  const h = await loraHarness();
  try {
    h.frame.removeAttribute("data-loaded");
    h.send();
    h.frame.setAttribute("data-loaded", "1");
    h.send({text: "invalid"});
    const source = h.frame.contentWindow;
    h.frame.remove();
    h.send({source});
    assert.equal(h.textarea.value, "base");
  } finally { h.dom.window.close(); }
});

test("LoRA bridge tracks URL changes instead of hard-coding the default port", async () => {
  const h = await loraHarness();
  try {
    h.frame.src = "https://manager.example.invalid:9443/loras";
    h.send();
    assert.equal(h.textarea.value, "base");
    h.send({source: h.frame.contentWindow, origin: "https://manager.example.invalid:9443", replace: true});
    assert.equal(h.textarea.value, "base, <lora:test:1>");
  } finally { h.dom.window.close(); }
});
