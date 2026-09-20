// TIPO 장치(GPU/CPU) 기억 — 실제 javascript/tipo_device.js 를 jsdom 에 올리고 Gradio 4.40 의 라디오 모양으로 확인한다.
// Gradio 라디오는 <input type="radio" value="GPU"> 의 change 이벤트로 값을 바꾸므로, 되살릴 때 그 이벤트가 나야 한다.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const SCRIPT = readFileSync(path.join(ROOT, "javascript", "tipo_device.js"), "utf8");
// Forge 의 javascript/localStorage.js 와 같은 모양(CI 에는 Forge 가 없다).
const FORGE_STORAGE = `
function localSet(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
function localGet(k, def) { try { return localStorage.getItem(k); } catch (e) {} return def; }
`;

const RADIOS = `<div id="sam3_tipo_device"><div class="wrap">
  <label><input type="radio" name="radio-1" value="GPU" checked><span>GPU</span></label>
  <label><input type="radio" name="radio-1" value="CPU"><span>CPU</span></label>
</div></div>`;

function load({ saved = null, radiosLater = false } = {}) {
  const dom = new JSDOM(`<!doctype html><html><body>${radiosLater ? "" : RADIOS}</body></html>`, {
    url: "http://127.0.0.1:7860/",
    runScripts: "outside-only",
  });
  const { window } = dom;
  if (saved !== null) window.localStorage.setItem("sam3_tipo_device", saved);
  const updates = [];
  window.gradioApp = () => window.document;
  window.onUiLoaded = (fn) => fn();
  window.onAfterUiUpdate = (fn) => updates.push(fn);
  window.eval(FORGE_STORAGE);
  const changes = [];
  const watch = () =>
    window.document.querySelectorAll("#sam3_tipo_device input").forEach((input) =>
      input.addEventListener("change", () => changes.push(input.value)));
  if (!radiosLater) watch();
  window.eval(SCRIPT);
  const input = (value) => window.document.querySelector(`#sam3_tipo_device input[value="${value}"]`);
  return { window, updates, changes, input, watch };
}

test("nothing saved keeps the server default and remembers a new choice", () => {
  const { window, changes, input } = load();
  assert.equal(input("GPU").checked, true);
  assert.deepEqual(changes, []);
  input("CPU").click();
  assert.equal(window.localStorage.getItem("sam3_tipo_device"), "CPU");
});

test("a saved choice is restored through the change event Gradio listens to", () => {
  const { changes, input } = load({ saved: "CPU" });
  assert.equal(input("CPU").checked, true);
  assert.equal(input("GPU").checked, false);
  assert.deepEqual(changes, ["CPU"]);
});

test("an unknown saved value is ignored", () => {
  const { changes, input } = load({ saved: "TPU" });
  assert.equal(input("GPU").checked, true);
  assert.deepEqual(changes, []);
});

test("radios that render after load are picked up on the next UI update", () => {
  const { window, updates, changes, input, watch } = load({ saved: "CPU", radiosLater: true });
  assert.equal(updates.length, 1);
  window.document.body.innerHTML = RADIOS;
  watch();
  updates[0]();
  assert.equal(input("CPU").checked, true);
  assert.deepEqual(changes, ["CPU"]);
  updates[0]();
  assert.deepEqual(changes, ["CPU"], "한 번만 붙는다");
});
