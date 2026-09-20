// notebook_lanes.js — 묶음별 섹션, 켜짐 표시, 칩 줄, 핀, 더 보기.
//
// 고정 자료는 실제 DOM 을 본뜬 것이다(.agents/specs/assets/live_dom_2026-09-20.md):
// 묶음은 Python 이 붙인 sam3-slot--<키> / sam3-lane--<묶음> 클래스로만 알아보고, 헤더는 .gradio-accordion 안의
// button.label-wrap 이다. 창은 t.after 로 닫는다 — 1초 폴링이 돌아 닫지 않으면 테스트가 끝나지 않는다.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM } from "jsdom";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const SCRIPT = readFileSync(path.join(ROOT, "javascript", "notebook_lanes.js"), "utf8");

const slot = (key, lane, opts = {}) => `
  <div class="gr-group sam3-slot sam3-slot--${key} sam3-lane--${lane}${opts.exp ? " sam3-exp" : ""}${opts.guess ? " sam3-on-guess" : ""}">
    <div class="styler">
      <div class="gradio-accordion sam3-head">
        <button class="label-wrap"><span>${opts.label || key}</span><span class="icon">▼</span></button>
        ${opts.noOn ? "" : `<div class="gradio-checkbox sam3-on"><label><input type="checkbox"${opts.on ? " checked" : ""}></label></div>`}
      </div>
    </div>
  </div>`;

function buildDom(t, options = {}) {
  const dom = new JSDOM(`<!doctype html><html><body>
    <main id="sam3_notebook_layout">
      <section class="sam3-notebook-prompt"></section>
      <div id="sam3_onbar"></div>
      <section class="sam3-notebook-columns">
        <div class="sam3-notebook-column" data-column="parameters"><h2>Parameters</h2><div>
          <div id="txt2img_settings">
            <div id="component-71" class="gr-group"></div>
            <div id="txt2img_hr" class="gradio-accordion input-accordion">
              <button class="label-wrap"><span>고해상도 보정</span></button>
              <div id="txt2img_hr-checkbox"><label><input type="checkbox"></label></div>
            </div>
            ${slot("anima-3-8b", "anima", { on: true, label: "Anima 3.8B (Qwen3.5 / v2)" })}
            ${slot("anima-detail-daemon", "anima")}
            ${slot("anima-safe-pag", "anima")}
            ${slot("anima-vae-2x", "anima", { exp: true })}
          </div>
        </div></div>
        <div class="sam3-notebook-column" data-column="scripts"><h2>Scripts</h2><div>
          <div id="txt2img_script_container"><div class="styler">
            ${slot("sam3", "det")}
            ${slot("adetailer", "det")}
            ${slot("lora-block-weight", "lora", { on: true })}
            ${slot("controlnet", "lora")}
            ${slot("compile", "etc", { noOn: true })}
            ${slot("brand-new", "etc", { guess: true })}
            <div class="gr-group sam3-slot sam3-slot--negpip sam3-lane--hidden"></div>
            <div class="form"><div id="script_list"></div></div>
            <div id="component-1726" class="gr-group sam3-script-panel"></div>
          </div></div>
        </div></div>
        <div class="sam3-notebook-column" data-column="gallery"><h2>Gallery</h2><div></div></div>
      </section>
    </main></body></html>`, {
      runScripts: "outside-only", pretendToBeVisual: true, url: "http://127.0.0.1:7860/"
    });
  t.after(() => dom.window.close());
  const { window } = dom;
  window.gradioApp = () => window.document;
  window.onUiLoaded = (fn) => fn();
  window.onAfterUiUpdate = () => {};
  window.Element.prototype.scrollIntoView = function () {};   // jsdom 에는 없다
  if (options.storage) window.localStorage.setItem("sam-extra.layout.v1", options.storage);
  window.eval(SCRIPT);
  window.__sam3LanesTestHooks.mount();
  return dom;
}

test("sections are inserted into our own column bodies only", (t) => {
  const doc = buildDom(t).window.document;
  const col1 = doc.querySelector('[data-column="parameters"] > div');
  const col2 = doc.querySelector('[data-column="scripts"] > div');
  assert.deepEqual([...col1.querySelectorAll(".sam3-lane-head")].map((h) => h.dataset.lane),
    ["pinned", "anima"]);
  assert.deepEqual([...col2.querySelectorAll(".sam3-lane-head")].map((h) => h.dataset.lane),
    ["det", "script", "lora", "etc"]);
  assert.equal(doc.querySelector("#txt2img_settings .sam3-lane-head"), null, "Gradio 안에는 넣지 않는다");
  assert.equal(doc.querySelector("#txt2img_script_container .sam3-lane-head"), null);
  assert.equal(doc.querySelector("#sam3_notebook_layout").dataset.sam3Lanes, "on");
});

test("hidden lanes and the script selector keep their place", (t) => {
  const doc = buildDom(t).window.document;
  const styler = doc.querySelector("#txt2img_script_container > .styler");
  assert.equal(styler.querySelector("#script_list").parentElement.className, "form");
  assert.equal(doc.querySelector(".sam3-slot--negpip").closest(".styler"), styler);
});

test("pills show how many features are on", (t) => {
  const dom = buildDom(t);
  const doc = dom.window.document;
  const pill = (key) => doc.querySelector(`.sam3-slot--${key} .label-wrap`).dataset.sam3Pill;
  assert.equal(pill("anima-3-8b"), "켜짐");
  assert.equal(pill("anima-detail-daemon"), undefined);
  assert.equal(doc.querySelector(".sam3-slot--anima-3-8b").dataset.sam3On, "on");
  const box = doc.querySelector(".sam3-slot--anima-detail-daemon .sam3-on input");
  box.checked = true;
  box.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  dom.window.__sam3LanesTestHooks.flush();
  assert.equal(pill("anima-detail-daemon"), "켜짐");
});

test("a nested slot's checkbox never leaks into the slot around it", (t) => {
  const dom = buildDom(t);
  const doc = dom.window.document;
  const outer = doc.querySelector(".sam3-slot--sam3");
  // 다른 칸이 어떤 이유로든 이 칸 안쪽에 들어오더라도, 그 칸의 켜짐이 바깥 칸으로 번지면 안 된다.
  const nested = doc.createElement("div");
  nested.className = "gr-group sam3-slot sam3-slot--nested sam3-lane--etc";
  nested.innerHTML = '<div class="gradio-checkbox sam3-on"><label><input type="checkbox" checked></label></div>';
  outer.appendChild(nested);
  dom.window.__sam3LanesTestHooks.flush();
  assert.equal(outer.dataset.sam3On, undefined, "SAM3 Mask 는 꺼져 있어야 한다");
  assert.equal(doc.querySelector('#sam3_onbar .sam3-chip[data-key="sam3"]').hidden, true);
});

test("a slot found by guessing is marked", (t) => {
  const doc = buildDom(t).window.document;
  assert.equal(doc.querySelector(".sam3-slot--brand-new").dataset.sam3Guess, "1");
  assert.equal(doc.querySelector(".sam3-slot--anima-3-8b").dataset.sam3Guess, undefined);
});

test("the chip bar lists what is on", (t) => {
  const dom = buildDom(t);
  const doc = dom.window.document;
  const chips = () => [...doc.querySelectorAll("#sam3_onbar .sam3-chip")]
    .filter((chip) => !chip.hidden).map((chip) => chip.dataset.key);
  assert.deepEqual(chips().sort(), ["anima-3-8b", "lora-block-weight"]);
  const box = doc.querySelector(".sam3-slot--anima-3-8b .sam3-on input");
  box.checked = false;
  box.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  dom.window.__sam3LanesTestHooks.flush();          // jsdom 의 rAF 는 ~16ms 뒤라 직접 흘려보낸다
  assert.deepEqual(chips(), ["lora-block-weight"]);
  assert.equal(doc.querySelector("#sam3_onbar .sam3-onbar-title").dataset.count, "1");
});

test("Forge 자체 기능도 칩으로 보인다", (t) => {
  const dom = buildDom(t);
  const doc = dom.window.document;
  const hires = doc.querySelector("#txt2img_hr-checkbox input");
  hires.checked = true;
  hires.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  dom.window.__sam3LanesTestHooks.flush();
  const chip = doc.querySelector('#sam3_onbar .sam3-chip[data-key="core:hires"]');
  assert.equal(chip.hidden, false);
});

test("켜짐 개념이 없는 칸은 칩을 만들지 않는다", (t) => {
  const doc = buildDom(t).window.document;
  assert.equal(doc.querySelector('#sam3_onbar .sam3-chip[data-key="compile"]'), null);
});

test("켜진 게 없으면 안내 문구가 보인다", (t) => {
  const dom = buildDom(t);
  const doc = dom.window.document;
  doc.querySelectorAll(".sam3-on input").forEach((input) => { input.checked = false; });
  dom.window.__sam3LanesTestHooks.flush();
  assert.equal(doc.querySelector("#sam3_onbar .sam3-onbar-empty").hidden, false);
  assert.equal(doc.querySelector("#sam3_onbar .sam3-onbar-title").dataset.count, "0");
});

test("이벤트 없이 바뀐 체크박스는 폴링이 잡는다", (t) => {
  const dom = buildDom(t);
  const doc = dom.window.document;
  doc.querySelector(".sam3-slot--anima-detail-daemon .sam3-on input").checked = true;  // 이벤트 없음
  dom.window.__sam3LanesTestHooks.poll();          // 1초 타이머가 부르는 것과 같은 함수
  assert.equal(doc.querySelector(".sam3-slot--anima-detail-daemon").dataset.sam3On, "on");
});

test("scanning without changes writes nothing to the DOM", (t) => {
  const dom = buildDom(t);
  const doc = dom.window.document;
  // MutationObserver 콜백은 마이크로태스크라 같은 틱에서 세면 항상 0 이다 — takeRecords 로 동기 확인한다.
  const observer = new dom.window.MutationObserver(() => {});
  observer.observe(doc.body, { childList: true, subtree: true, attributes: true });
  for (let i = 0; i < 20; i += 1) dom.window.__sam3LanesTestHooks.scan();
  assert.equal(observer.takeRecords().length, 0);
  observer.disconnect();
});

test("pinning moves a column-2 item up and unpinning puts it back", (t) => {
  const dom = buildDom(t);
  const doc = dom.window.document;
  const slot = doc.querySelector(".sam3-slot--adetailer");
  const styler = slot.parentElement;
  const before = [...styler.children].indexOf(slot);
  slot.querySelector(".sam3-pin").click();
  assert.equal(slot.dataset.sam3Pinned, "1");
  assert.equal(slot.parentElement, doc.querySelector('[data-column="parameters"] > div'));
  slot.querySelector(".sam3-pin").click();
  assert.equal(slot.parentElement, styler);
  assert.equal([...styler.children].indexOf(slot), before);
  assert.equal(slot.dataset.sam3Pinned, undefined);
});

test("an anima pin never moves the node", (t) => {
  const doc = buildDom(t).window.document;
  const slot = doc.querySelector(".sam3-slot--anima-detail-daemon");
  const parent = slot.parentElement;
  slot.querySelector(".sam3-pin").click();
  assert.equal(slot.parentElement, parent);
  assert.equal(slot.dataset.sam3Pinned, "1");
  slot.querySelector(".sam3-pin").click();
  assert.equal(slot.parentElement, parent, "해제해도 옮기지 않는다");
});

test("pins survive a remount through localStorage", (t) => {
  const dom = buildDom(t);
  dom.window.document.querySelector(".sam3-slot--adetailer .sam3-pin").click();
  const stored = JSON.parse(dom.window.localStorage.getItem("sam-extra.layout.v1"));
  assert.deepEqual(stored.pinned, ["adetailer"]);
});

test("저장이 막혀 있어도 그 화면에서는 동작한다", (t) => {
  const dom = buildDom(t);
  const doc = dom.window.document;
  dom.window.localStorage.setItem = () => { throw new Error("blocked"); };
  const slot = doc.querySelector(".sam3-slot--adetailer");
  slot.querySelector(".sam3-pin").click();
  assert.equal(slot.dataset.sam3Pinned, "1");
});

test("a pin click never reaches the accordion header", (t) => {
  const doc = buildDom(t).window.document;
  const slot = doc.querySelector(".sam3-slot--adetailer");
  let headerClicks = 0;
  slot.querySelector(".label-wrap").addEventListener("click", () => { headerClicks += 1; });
  slot.querySelector(".sam3-pin").click();
  assert.equal(headerClicks, 0);
});

test("더 보기 is closed by default and counts what is inside", (t) => {
  const doc = buildDom(t).window.document;
  const more = doc.querySelector(".sam3-more");
  assert.equal(doc.querySelector("#sam3_notebook_layout").dataset.sam3More, "closed");
  assert.equal(more.getAttribute("aria-expanded"), "false");
  assert.equal(more.querySelector(".sam3-more-count").dataset.count, "4");  // lbw, controlnet, compile, brand-new
  assert.equal(more.querySelector(".sam3-more-on").dataset.count, "1");     // lbw 가 켜져 있다
});

test("펼침 상태는 저장돼 다음 마운트에서 되살아난다", (t) => {
  const first = buildDom(t);
  first.window.document.querySelector(".sam3-more").click();
  assert.equal(first.window.document.querySelector("#sam3_notebook_layout").dataset.sam3More, "open");
  const stored = first.window.localStorage.getItem("sam-extra.layout.v1");
  const second = buildDom(t, { storage: stored });
  assert.equal(second.window.document.querySelector("#sam3_notebook_layout").dataset.sam3More, "open");
});

test("a chip for a hidden item opens 더 보기 and the accordion", (t) => {
  const dom = buildDom(t);
  const doc = dom.window.document;
  const head = doc.querySelector(".sam3-slot--lora-block-weight .label-wrap");
  let opened = 0;
  head.addEventListener("click", () => { opened += 1; });
  assert.equal(doc.querySelector('#sam3_onbar .sam3-chip[data-key="lora-block-weight"]').dataset.inMore, "1");
  doc.querySelector('#sam3_onbar .sam3-chip[data-key="lora-block-weight"]').click();
  assert.equal(doc.querySelector("#sam3_notebook_layout").dataset.sam3More, "open");
  assert.equal(opened, 1);
  head.classList.add("open");
  doc.querySelector('#sam3_onbar .sam3-chip[data-key="lora-block-weight"]').click();
  assert.equal(opened, 1, "이미 열린 헤더는 다시 누르지 않는다");
});

test("restore leaves no trace, even after a pin", (t) => {
  const dom = buildDom(t);
  const doc = dom.window.document;
  const slot = doc.querySelector(".sam3-slot--adetailer");
  const styler = slot.parentElement;
  const before = [...styler.children].indexOf(slot);
  slot.querySelector(".sam3-pin").click();
  dom.window.__sam3LanesTestHooks.restore();
  assert.equal(doc.querySelectorAll(".sam3-pin, .sam3-chip, .sam3-lane-head, .sam3-more, .sam3-pin-hint").length, 0);
  assert.equal(doc.querySelector("#sam3_notebook_layout").dataset.sam3Lanes, undefined);
  assert.equal(doc.querySelector(".sam3-slot--anima-3-8b").dataset.sam3On, undefined);
  assert.equal(doc.querySelector(".sam3-slot--anima-3-8b .label-wrap").dataset.sam3Pill, undefined);
  assert.equal(slot.parentElement, styler);
  assert.equal([...styler.children].indexOf(slot), before);
});

test("without slots or with ?sam3_lanes=off nothing happens", (t) => {
  const dom = new JSDOM(`<!doctype html><html><body><main id="sam3_notebook_layout"></main></body></html>`,
    { runScripts: "outside-only", url: "http://127.0.0.1:7860/?sam3_lanes=off" });
  t.after(() => dom.window.close());
  dom.window.gradioApp = () => dom.window.document;
  dom.window.onUiLoaded = (fn) => fn();
  dom.window.onAfterUiUpdate = () => {};
  dom.window.eval(SCRIPT);
  dom.window.__sam3LanesTestHooks.mount();
  assert.equal(dom.window.document.querySelector("#sam3_notebook_layout").dataset.sam3Lanes, undefined);
});
