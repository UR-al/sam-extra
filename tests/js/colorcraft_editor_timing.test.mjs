// Colorcraft 공유 편집기(javascript/colorcraft_editor.js) — 빠른 드롭다운 맞추기(nudge)가 "언제" 도는지.
//
// 브라우저 이벤트 루프(태스크 · 프레임 · 콜백마다의 마이크로태스크 체크포인트)와 Gradio 4.40 의 갱신 순서를 결정적으로 흉내 낸
// 모형에 실제 편집기를 올린다(js/app/src/Blocks.svelte · init.ts, 실제 페이지에서 잰 순서와 같다):
//   - 이벤트는 requestAnimationFrame 안에서 처리기를 부른다("gradio" 리스너). flush 가 남아 있으면 그 flush 가 끝날 때 부른다
//     (wait_then_trigger_api_call).
//   - 처리기(frontend_fn)가 돌려준 값은 handle_update → update_value 가 requestAnimationFrame(flush) 로 다음 프레임에 적용한다.
//   - Svelte 는 flush 콜백 바로 뒤의 마이크로태스크에서 그리고, 값이 바뀐 칸의 change 이벤트가 다시 위로 돈다
//     (state → sync 메아리, Type → kind).
// 화면은 프레임이 끝날 때 그려진다: 프레임 안의 변경은 그 프레임에, 프레임 사이 태스크의 변경은 다음 프레임에 보인다.
// 빠른 드롭다운은 그 훅(__sam3FastDropdownSync)만 둔다 — 실제 notebook.js 와의 훅 이름은 colorcraft_editor.test.mjs 가 지킨다.
import test from "node:test";
import assert from "node:assert/strict";
import vm from "node:vm";
import { JSDOM } from "jsdom";
import { loadEditor } from "./colorcraft_bridge.mjs";

const P = "script_txt2img_colorcraft_samextra_";
const SCHEMA = loadEditor().schema;
const EDITORS = [
    ...SCHEMA.modifierFields.map((f) => `mod_${f.name}`),
    ...SCHEMA.leafFields.map((f) => `leaf_${f.name}`),
    ...SCHEMA.comboFields.map((f) => `combo_${f.name}`),
];

function eventLoop() {
    let tasks = [];
    let callbacks = [];
    let frame = null;                                   // 지금 도는 프레임 번호(프레임 밖이면 null)
    let done = 0;                                       // 끝난(그려진) 프레임 수
    const checkpoint = () => new Promise((resolve) => setImmediate(resolve));   // 마이크로태스크를 다 비운다
    const loop = {
        setTimeout: (fn) => { tasks.push(fn); },
        requestAnimationFrame: (fn) => { callbacks.push(fn); },
        now: () => ({ frame, done }),
        busy: () => tasks.length > 0 || callbacks.length > 0,
        async task(fn) {
            fn();
            await checkpoint();
        },
        async tasks() {
            while (tasks.length) await loop.task(tasks.shift());
        },
        async frame() {                                 // 이 프레임 전에 등록된 콜백만, 등록 순서대로
            const due = callbacks;
            callbacks = [];
            frame = done + 1;
            for (const cb of due) {
                cb();
                await checkpoint();
            }
            frame = null;
            done += 1;
        },
    };
    return loop;
}

// 그 변경이 화면에 보이는 프레임
const painted = (at) => (at.frame !== null ? at.frame : at.done + 1);

function gradio(loop, doc, env) {
    const values = {};
    const log = [];
    let pending = [];
    let scheduled = false;
    let waiting = [];
    const fn = (name) => {
        const page = vm.runInContext(`(...a) => window.samextraColorcraft ? window.samextraColorcraft.${name}('txt2img', a) : undefined`, env.context);
        return async (args) => page(...args);                       // frontend_fn: await 뒤에 handle_update
    };
    const deps = [
        { triggers: ["mod_select", "mask_select", "state"], fn: fn("sync"),
            inputs: ["mod_select", "mask_select", "enabled", "masking", "state", "ref", ...EDITORS],
            outputs: ["state", "ref", ...EDITORS, "leaf_group", "combo_group", "mod_reset", "mask_reset", "mod_select",
                "mask_select", "summary"] },
        { triggers: ["mod_kind"], fn: fn("kind"), inputs: ["mod_kind"], outputs: Array.from({ length: 10 }, (_, i) => `g${i}`) },
    ];

    function updateValue(updates) {
        pending.push(updates);
        if (!scheduled) {
            scheduled = true;
            loop.requestAnimationFrame(flush);
        }
    }
    function flush() {
        const changed = [];
        for (const u of pending.flat()) {
            if (u.prop !== "value") continue;
            if (JSON.stringify(values[u.id]) !== JSON.stringify(u.value)) changed.push(u.id);
            values[u.id] = u.value;
        }
        pending = [];
        Promise.resolve().then(() => render(changed));             // Svelte: 이 콜백 바로 뒤의 마이크로태스크
        scheduled = false;
        const run = waiting;
        waiting = [];
        run.forEach((go) => go());
    }
    function render(changed) {
        for (const id of changed) {
            const input = doc.querySelector(`#${P}${id} input[role='listbox']`);
            if (input) {
                input.value = values[id];
                log.push({ what: "gradio", v: values[id], ...loop.now() });
            }
        }
        changed.forEach(dispatchChange);
    }
    function dispatchChange(id) {                                     // Blocks 의 "gradio" 리스너
        for (const dep of deps) if (dep.triggers.includes(id)) loop.requestAnimationFrame(() => trigger(dep));
    }
    function trigger(dep) {
        if (scheduled) waiting.push(() => call(dep));
        else call(dep);
    }
    async function call(dep) {
        const args = await Promise.all([...dep.inputs, ...dep.outputs].map(async (id) => values[id]));
        const result = await dep.fn(args);
        handleUpdate(result === undefined ? [] : result, dep);
    }
    async function handleUpdate(data, dep) {
        updateValue(dep.outputs.map((id) => ({ id, prop: "value_is_output", value: true })));
        await Promise.resolve();                                      // tick()
        const updates = [];
        data.forEach((v, i) => {
            if (v && typeof v === "object" && v.__type__ === "update") {
                for (const [prop, x] of Object.entries(v)) if (prop !== "__type__") updates.push({ id: dep.outputs[i], prop, value: x });
            } else {
                updates.push({ id: dep.outputs[i], prop: "value", value: v });
            }
        });
        updateValue(updates);
    }
    return { values, log, updateValue, dispatchChange };
}

// III 의 Type 이 Chroma 인 state 에서 I 을 보고 있다가 III 을 누른다. schedule 은 편집기에 주는 타이머·프레임 함수(바꿔 끼워 옛
// 순서를 흉내 낸다). starve: 처리기가 돈 프레임 바로 뒤에 타이머보다 다음 프레임이 먼저 온다. pendingFlush: 누를 때 다른 갱신이
// 이미 flush 를 기다리고 있다.
async function switchToIII(schedule, { starve = false, pendingFlush = false } = {}) {
    const loop = eventLoop();
    const dom = new JSDOM(`<!doctype html><body><div id="${P}accordion"><div class="gradio-dropdown" id="${P}mod_kind">`
        + '<div class="container"><input role="listbox" value="Advanced"></div></div></div></body>');
    const doc = dom.window.document;
    const env = loadEditor({ document: doc, console, ...schedule(loop) });
    const page = gradio(loop, doc, env);
    const defaults = [
        ...SCHEMA.modifierFields.map((f) => (f.name === "active" ? SCHEMA.activeDefaults[0] : f.default)),
        ...SCHEMA.leafFields.map((f) => f.default),
        ...SCHEMA.comboFields.map((f) => f.default),
    ];
    EDITORS.forEach((id, i) => { page.values[id] = defaults[i]; });
    Object.assign(page.values, { mod_select: "I", mask_select: "M1", enabled: true, masking: false,
        state: JSON.stringify({ v: 1, rev: "r0", "III.kind": "Chroma" }), ref: "I|M1|r0" });

    let shown = "Advanced";                                           // 빠른 드롭다운이 보여 주는 글자
    doc.getElementById(`${P}mod_kind`).__sam3FastDropdownSync = (v) => {
        page.log.push({ what: "nudge", v, ...loop.now() });
        if (v !== shown) {
            shown = v;
            page.log.push({ what: "proxy", v, ...loop.now() });
        }
    };

    await loop.task(() => {                                           // 클릭: 라디오 값이 바뀌고 change
        page.values.mod_select = "III";
        page.dispatchChange("mod_select");
        if (pendingFlush) page.updateValue([{ id: "summary", prop: "loading_status", value: {} }]);
    });
    for (let i = 0; loop.busy(); i++) {
        assert.ok(i < 40, "the page never settles");
        await loop.frame();
        if (starve && i === 0) await loop.frame();                     // the frame after the handler's, before its timer
        await loop.tasks();
    }
    dom.window.close();
    const first = (what) => page.log.find((e) => e.what === what && e.v === "Chroma");
    return { shown, log: page.log, value: first("gradio"), proxy: first("proxy"), refs: page.values.ref };
}

const shipped = (loop) => ({ setTimeout: loop.setTimeout, requestAnimationFrame: loop.requestAnimationFrame });
// v0.32.0: requestAnimationFrame(() => requestAnimationFrame(nudge))
const twoFrames = (loop) => ({ setTimeout: (fn) => loop.requestAnimationFrame(fn), requestAnimationFrame: loop.requestAnimationFrame });
// 실제 Forge 점검이 페이지에만 넣어 본 변형: requestAnimationFrame(() => setTimeout(nudge, 0))
const frameThenTimer = (loop) => ({ setTimeout: (fn) => fn(),
    requestAnimationFrame: (fn) => loop.requestAnimationFrame(() => loop.setTimeout(fn)) });

test("the model: a switch loads III's Type (state echo and the Type change run through Gradio again)", async () => {
    const r = await switchToIII(shipped);
    assert.equal(r.shown, "Chroma");
    assert.ok(r.value, "Gradio wrote the loaded Type");
    assert.match(r.refs, /^III\|M1\|[0-9a-f]{12}$/);
    assert.ok(r.log.filter((e) => e.what === "nudge").length >= 2, "the state echo nudges again");
});

test("the proxy shows the loaded Type in the same screen update as Gradio's own field", async () => {
    for (const pendingFlush of [false, true]) {
        const r = await switchToIII(shipped, { pendingFlush });
        assert.equal(r.shown, "Chroma", `pendingFlush=${pendingFlush}`);
        assert.notEqual(r.proxy.frame, null, "in a frame callback, not in a task between frames");
        assert.equal(painted(r.proxy), painted(r.value), `pendingFlush=${pendingFlush}`);
        assert.ok(r.log.indexOf(r.value) < r.log.indexOf(r.proxy), "after Gradio has rendered the value");
    }
});

test("when the browser renders the next frame before the 0 ms timer, the nudge is one frame late but still after Gradio", async () => {
    const r = await switchToIII(shipped, { starve: true });
    assert.equal(r.shown, "Chroma");
    assert.ok(r.log.indexOf(r.value) < r.log.indexOf(r.proxy));
    assert.equal(painted(r.proxy), painted(r.value) + 1);              // v0.32.0 was this late every time
});

test("control: the earlier schedules show the proxy one screen update after the value (the model tells them apart)", async () => {
    for (const [name, schedule] of [["two frames (v0.32.0)", twoFrames], ["frame then 0 ms timer", frameThenTimer]]) {
        const r = await switchToIII(schedule);
        assert.equal(r.shown, "Chroma", name);
        assert.equal(painted(r.proxy), painted(r.value) + 1, name);
    }
    const variant = await switchToIII(frameThenTimer);
    assert.equal(variant.proxy.frame, null);                         // DOM right after the value's frame: "lag 0" in a DOM poll
    assert.equal(variant.proxy.done, variant.value.frame);
});
