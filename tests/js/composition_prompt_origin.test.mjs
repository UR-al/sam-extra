// composition_prompt.js 와 사용자 앱 원본 compositionPrompt.ts 의 대조.
//
// 원본은 tests/_origin_composition_prompt/compositionPrompt.ts — 머리 주석 뒤는 앱 파일 그대로이고(앱 커밋 d2fcc707,
// SHA-256 고정), Node 가 타입을 직접 지우므로(process.features.typescript — 여기 Node 24) 그 파일을 그대로 import 한다.
// 앱에서 모듈 안에만 있는 tagKey·CONTRAST_GROUPS 는 같은 원본 본문 끝에 export 한 줄만 붙인 임시 사본(.mts)으로 꺼낸다.
// 이식본은 이 realm 에서 실행한다(vm.runInThisContext) — 두 결과를 deepStrictEqual(-0·프로토타입까지)로 견준다.
//
// 격자: 프리셋 전부 · 각 조절값의 범위 전체와 그 바깥(정수·.5·.49 반올림 경계) · 태그 문턱값(방향 20/65/115/160,
// 높이 ±20/60, 거리 15/30/50/65/85, 기울기 ±6, 화면 위치 ±25) 바로 안팎의 곱 격자 · NaN·Infinity·문자열·객체 같은 깨진 값 ·
// 프롬프트 말뭉치(가중치, 이스케이프, 스케줄, 와일드카드, LoRA, 끝 쉼표, 빈 값)와 시드 고정 무작위 문자열.
// 타입 지우기가 없는 Node(CI 의 Node 20)에서는 해시 고정만 돌고 대조는 건너뛴다.
import { after, test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import vm from "node:vm";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const ORIGIN = path.join(ROOT, "tests", "_origin_composition_prompt", "compositionPrompt.ts");
const MARKER = "// ---- app frontend/src/utils/compositionPrompt.ts below (verbatim) ----\n";
// SHA-256 of the app's frontend/src/utils/compositionPrompt.ts at d2fcc70798e61da137e87a8ff600d6fa73250fe6
// (git blob d083758ad318339bdda996e038f6f924b1e949eb).
const ORIGIN_SHA256 = "69eb03cf56132fd64dfb3958dc848fc6b74b09fefce3c69817a03ff8a571ded7";
const STRIP = Boolean(process.features && process.features.typescript);
const SKIP = STRIP ? false : "this Node cannot import TypeScript (process.features.typescript is unset; Node 24 can)";

const PORT_SOURCE = path.join(ROOT, "javascript", "composition_prompt.js");
vm.runInThisContext(readFileSync(PORT_SOURCE, "utf8"), { filename: PORT_SOURCE });
const port = globalThis.sam3CompositionPrompt;

// A checkout with core.autocrlf=true writes CRLF; the pinned body is the app's LF bytes (as in progress_bar.test.mjs).
const originText = readFileSync(ORIGIN, "utf8").replace(/\r\n/g, "\n");
const originBody = originText.split(MARKER)[1];

let originModules = null;
let scratch = null;
async function origin() {
  if (!originModules) {
    const app = await import(pathToFileURL(ORIGIN).href);
    scratch = mkdtempSync(path.join(tmpdir(), "sam3-composition-origin-"));
    const exposed = path.join(scratch, "compositionPrompt.private.mts");
    writeFileSync(exposed, originBody + "\nexport { tagKey, CONTRAST_GROUPS };\n", "utf8");
    const internals = await import(pathToFileURL(exposed).href);
    originModules = { app, internals };
  }
  return originModules;
}
after(() => {
  if (scratch) rmSync(scratch, { recursive: true, force: true });
});

// ---- 도구 ------------------------------------------------------------------------------

function outcome(fn) {
  try {
    return { ok: true, value: fn() };
  } catch (error) {
    return { ok: false, name: error && error.constructor && error.constructor.name, message: String(error && error.message) };
  }
}

// 결과(또는 던진 오류의 종류·문구)가 같은지. 다르면 처음 몇 개를 모아 한 번에 보여 준다. label 은 어긋났을 때만
// 만든다(함수) — 수십만 번 JSON.stringify 하지 않게. 글자만 다루는 함수(split·tagKey·plan: 결과가 문자열과 문자열
// 배열뿐)는 strings: true 로 JSON 문자열끼리 견준다 — 그 값들에서는 deepStrictEqual 과 같은 판정이고 열 배쯤 빠르다.
// 수를 내는 함수는 deepStrictEqual(-0 과 0 을 가른다).
function collector(limit = 8) {
  const misses = [];
  let checked = 0;
  return {
    check(label, a, b, { strings = false } = {}) {
      checked += 1;
      if (misses.length >= limit) return;
      const left = outcome(a);
      const right = outcome(b);
      let same;
      if (strings) {
        same = JSON.stringify(left) === JSON.stringify(right);
      } else {
        try {
          assert.deepStrictEqual(left, right);
          same = true;
        } catch {
          same = false;
        }
      }
      if (!same) misses.push({ label: typeof label === "function" ? label() : label, port: left, origin: right });
    },
    done(minimum) {
      assert.deepStrictEqual(misses, [], `${misses.length} mismatches`);
      assert.ok(checked >= minimum, `only ${checked} comparisons`);
      return checked;
    },
  };
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomStrings(seed, count, alphabet, maxLength) {
  const random = mulberry32(seed);
  const out = [];
  for (let i = 0; i < count; i += 1) {
    const length = Math.floor(random() * (maxLength + 1));
    let text = "";
    for (let j = 0; j < length; j += 1) text += alphabet[Math.floor(random() * alphabet.length)];
    out.push(text);
  }
  return out;
}

const KEYS = ["azimuth", "elevation", "distance", "roll", "framing"];
const DEFAULT = { azimuth: 0, elevation: 0, distance: 45, roll: 0, framing: 0 };

// 숫자가 아닌 값·특수한 수. normalizeComposition 은 typeof 'number' 이고 유한한 수만 받는다.
const ODD_VALUES = [
  Number.NaN, Infinity, -Infinity, -0, 0, Number.MAX_VALUE, -Number.MAX_VALUE, Number.MIN_VALUE, -Number.MIN_VALUE,
  Number.EPSILON, 1e21, -1e21, 2 ** 53, -(2 ** 53), "24", "-24", "", " 5 ", "abc", "1e2", null, undefined, true, false,
  {}, [], [5], ["5"], new Number(5), Object(-30), 5n, Symbol("odd"), () => 5, new Date(0),
];

// 태그가 바뀌는 문턱값과 범위 끝, 그 바로 안팎(반올림 .5 와 .49 포함). 곱 격자용이라 문턱마다 양쪽 하나씩만 —
// 한 조절값의 모든 값은 아래 sweep 이 따로 훑는다.
const EDGES = {
  azimuth: [-181, -180, -160, -159.5, -115.5, -115, -65, -64.5, -20.5, -20, -0.5, -0, 0, 0.49, 20, 20.5, 64.5, 65, 115,
    115.5, 159.5, 160, 180, 181],
  elevation: [-76, -75, -20.5, -20, -19.5, 0, 19.5, 20, 59.5, 60, 75, 76],
  distance: [-1, 0, 15, 15.5, 30, 30.5, 50, 50.5, 65, 65.5, 85, 85.5, 100, 101],
  roll: [-31, -6, -5.5, 0, 5.49, 5.5, 6, 31],
  framing: [-101, -25, -24.5, 0, 24.49, 24.5, 25, 101],
};

function sweep(min, max) {
  const values = [];
  for (let v = min - 3; v <= max + 3; v += 1) values.push(v, v + 0.5, v - 0.5, v + 0.49, v - 0.49, v + 0.4999999999);
  return values;
}

// ---- 원본 파일 ---------------------------------------------------------------------------

test("the vendored origin is the app file verbatim after its header (SHA-256 pinned)", () => {
  assert.equal(originText.split(MARKER).length, 2, "the marker line appears exactly once");
  assert.equal(createHash("sha256").update(originBody, "utf8").digest("hex"), ORIGIN_SHA256);
  const header = originText.split(MARKER)[0];
  assert.match(header, /frontend\/src\/utils\/compositionPrompt\.ts/);
  assert.match(header, /d2fcc70798e61da137e87a8ff600d6fa73250fe6/);
  assert.notEqual(originText.charCodeAt(0), 0xfeff, "no BOM");
  for (const line of header.split("\n").filter(Boolean)) assert.match(line, /^\/\//, "the header is only line comments");
});

test("the port names the same app commit as the vendored origin", () => {
  const source = readFileSync(PORT_SOURCE, "utf8");
  assert.match(source, /frontend\/src\/utils\/compositionPrompt\.ts/);
  assert.match(source, /d2fcc70798e61da137e87a8ff600d6fa73250fe6/);
  assert.match(source, /d083758ad318339bdda996e038f6f924b1e949eb/);
});

// ---- 상수 ---------------------------------------------------------------------------------

test("constants, presets and contrast groups are the app's", { skip: SKIP }, async () => {
  const { app, internals } = await origin();
  assert.deepStrictEqual(port.DEFAULT_COMPOSITION, app.DEFAULT_COMPOSITION);
  assert.deepStrictEqual(port.COMPOSITION_CONTROLS, app.COMPOSITION_CONTROLS);
  assert.deepStrictEqual(port.COMPOSITION_PRESETS, app.COMPOSITION_PRESETS);
  assert.deepStrictEqual(port.CONTRAST_GROUPS, internals.CONTRAST_GROUPS);
  assert.deepStrictEqual(Object.keys(app).sort(), ["COMPOSITION_CONTROLS", "COMPOSITION_PRESETS", "DEFAULT_COMPOSITION",
    "compositionCameraPoint", "compositionTags", "dragComposition", "normalizeComposition", "planCompositionAppend",
    "splitCompositionPrompt"], "a new export in the app needs porting");
  for (const name of Object.keys(app)) assert.ok(name in port, name);
});

// ---- 수 계산: normalize · tags · camera point -----------------------------------------------

function compareState(c, app, label, value) {
  c.check(() => `normalize ${label()}`, () => port.normalizeComposition(value), () => app.normalizeComposition(value));
  c.check(() => `tags ${label()}`, () => port.compositionTags(value), () => app.compositionTags(value));
  c.check(() => `camera ${label()}`, () => port.compositionCameraPoint(value), () => app.compositionCameraPoint(value));
}

test("every preset and odd whole values give the same state, tags and camera point", { skip: SKIP }, async () => {
  const { app } = await origin();
  const c = collector();
  for (const preset of app.COMPOSITION_PRESETS) compareState(c, app, () => preset.name, preset.state);
  const inherited = Object.create({ azimuth: 50, elevation: 30, distance: 90 });
  const nullProto = Object.assign(Object.create(null), { azimuth: -100, roll: 12 });
  const getter = { get azimuth() { return 33.5; }, get roll() { return -7; } };
  const wholes = [null, undefined, 0, 1, -1, "", "azimuth", true, false, [], [1, 2, 3], {}, () => ({ azimuth: 90 }),
    inherited, nullProto, getter, JSON.parse('{"__proto__": {"azimuth": 50}, "distance": 99}'), { azimuth: 50, extra: 1 },
    Object.freeze({ ...DEFAULT }), new Map([["azimuth", 90]]), Symbol("whole"), 5n];
  wholes.forEach((value, index) => compareState(c, app, () => `whole #${index}`, value));
  c.done(3 * (app.COMPOSITION_PRESETS.length + wholes.length));
});

test("each control over its whole range and beyond, from every preset, matches", { skip: SKIP }, async () => {
  const { app } = await origin();
  const c = collector();
  const bases = [DEFAULT, ...app.COMPOSITION_PRESETS.map((preset) => preset.state)];
  for (const control of app.COMPOSITION_CONTROLS) {
    const values = [...sweep(control.min, control.max), ...ODD_VALUES];
    for (const base of bases) {
      for (const value of values) compareState(c, app, () => `${control.key}=${String(value)}`, { ...base, [control.key]: value });
    }
  }
  c.done(3 * bases.length * 5 * 300);
});

test("the product of all threshold edges matches tag for tag and point for point", { skip: SKIP }, async () => {
  const { app } = await origin();
  const misses = [];
  let checked = 0;
  for (const azimuth of EDGES.azimuth) {
    for (const elevation of EDGES.elevation) {
      for (const distance of EDGES.distance) {
        for (const roll of EDGES.roll) {
          for (const framing of EDGES.framing) {
            const value = { azimuth, elevation, distance, roll, framing };
            const a = port.normalizeComposition(value);
            const b = app.normalizeComposition(value);
            const ta = port.compositionTags(value).join("\u0000");
            const tb = app.compositionTags(value).join("\u0000");
            const pa = port.compositionCameraPoint(value);
            const pb = app.compositionCameraPoint(value);
            checked += 1;
            const same = KEYS.every((key) => Object.is(a[key], b[key])) && ta === tb
              && Object.is(pa.x, pb.x) && Object.is(pa.y, pb.y) && pa.behind === pb.behind;
            if (!same && misses.length < 8) misses.push({ value, port: [a, ta, pa], origin: [b, tb, pb] });
          }
        }
      }
    }
  }
  assert.deepStrictEqual(misses, []);
  assert.equal(checked, Object.values(EDGES).reduce((n, list) => n * list.length, 1));
});

test("dragComposition matches for every start, delta and odd input", { skip: SKIP }, async () => {
  const { app } = await origin();
  const c = collector();
  const starts = [DEFAULT, ...app.COMPOSITION_PRESETS.map((preset) => preset.state),
    { azimuth: 180, elevation: 75, distance: 100, roll: 30, framing: 100 },
    { azimuth: -180, elevation: -75, distance: 0, roll: -30, framing: -100 },
    { azimuth: 179, elevation: 74, distance: 45, roll: 0, framing: 0 },
    { azimuth: "10", elevation: null, distance: "x", roll: undefined, framing: NaN },
    { azimuth: 10.5, elevation: -0.5, distance: 45.5, roll: 5.5, framing: 24.5 },
    {}, Object.create({ azimuth: 40, distance: 90 }),
    JSON.parse('{"__proto__": {"distance": 99, "roll": 20}, "azimuth": 5}'),   // spread vs Object.assign would differ here
    { azimuth: Symbol("start") }];
  const deltas = [-10000, -1000, -257.15, -100, -50, -1.5, -0.7, -0.5, 0, -0, 0.5, 0.7, 1.5, 7, 50, 100, 257.15, 1000,
    10000, Number.NaN, Infinity, -Infinity, "5", "", "abc", null, undefined, true, [], {}, 5n];
  for (const start of starts) {
    for (const dx of deltas) {
      for (const dy of deltas) {
        c.check(() => `drag ${String(dx)},${String(dy)}`, () => port.dragComposition(start, dx, dy), () => app.dragComposition(start, dx, dy));
      }
    }
  }
  for (const start of [null, undefined, 5]) {
    c.check(`drag from ${start}`, () => port.dragComposition(start, 1, 1), () => app.dragComposition(start, 1, 1));
  }
  c.done(starts.length * deltas.length * deltas.length);
});

// ---- 프롬프트: split · tagKey · plan ------------------------------------------------------------

const PROMPTS = [
  "", " ", "  ", ",", ", ,", ",,,", " , ", "a", "a,", "a, ", "a,  ", "a ,", " a , b ", "a,b", "a,,b", "a, , b",
  "forest", "forest,", "forest, ", "forest,\n", "forest\n", "forest, upper_body", "cityscape", "city",
  "1girl, solo, (masterpiece:1.2), best quality",
  "rain, (from_above:1.15), (city, trees:1.2), \\(sign\\),  ",
  "from below, close-up", "from below, full body", "(from above, blue sky:1.1)",
  "facing viewer, upper body, centered composition", "forest, facing viewer, upper body, centered composition",
  "FACING_VIEWER, Upper_Body, (centered composition:1.3)",
  "((full_body)), wide_shot", "(((from above)))", "(from above:0.8), [from below]", "[from above:from below:0.5]",
  "[cat|dog], {red|blue,green}", "{a|b|c}, __wildcard__, __season/winter__, __a,b__",
  "<lora:detail_tweaker:0.6>, <lyco:style,x:1>", "<lora:a,b:1>, (c:1.1)", "<lora:from_above:1>",
  "a, (b, c:1.1), [d:e:0.5], {f|g,h}, <lora:a,b:1>, escaped\\,comma",
  "escaped\\(paren\\), \\[bracket\\], \\{brace\\}, back\\\\slash, end\\",
  "\\(from above\\), (from above\\)", "trailing backslash\\", "\\", "\\,", "\\\\,x",
  "unbalanced (open, still open", "unbalanced close), next", "mismatch (a], b)", "nested ([{<x>}]), y", ")(, a",
  "BREAK, from above BREAK close-up", "from above AND from below", "masterpiece, best quality, newest",
  "photo of a girl, from side, profile, view from the subject's right",
  "three-quarter view from the subject's left, cowboy shot", "dutch angle, frame tilted clockwise",
  "subject on the left side of the frame", "subject_on_the_right_side_of_the_frame", "bird's eye view, from above",
  "close-up portrait", "(close-up:1.4), portrait", "from behind, rear three-quarter view from the subject's left",
  "(upper body: 1.2), (cowboy shot:-0.5), (wide shot:+1), (full body:.7), (portrait:1.)",
  "(from above:1.2:0.5)", "(from above:abc)", "(:1.2)", "()", "(( ))", "( , )", "(a) (b)", "(a)(b), (c",
  "\t from above \t,\tclose-up", "from above\nclose-up", "from above, \n close-up \n", "from above,\u00a0close-up",
  "日本語, 한글 태그, from above", "emoji 🎥, from above", "\ud83c, lone surrogate",
  "a".repeat(300) + ", from above", "from above,", "from above, ", "from above ,", "from above,  ", "from above , ",
  "From Above, FROM_BELOW, Close_Up, Wide Shot", "centered_composition", "(centered composition)",
];
const ALPHABET = "()[]{}<>,,,  \\:|_aAbB.15-+\n\tfrom above";
const RANDOM_PROMPTS = randomStrings(20261003, 4000, ALPHABET, 40);

test("splitCompositionPrompt matches on the corpus and on 4000 random strings", { skip: SKIP }, async () => {
  const { app } = await origin();
  const c = collector();
  for (const text of [...PROMPTS, ...RANDOM_PROMPTS]) {
    c.check(() => `split ${JSON.stringify(text)}`, () => port.splitCompositionPrompt(text), () => app.splitCompositionPrompt(text),
      { strings: true });
  }
  for (const odd of [null, undefined, 5, ["a", ",", "b"], { length: 2, 0: ",", 1: "a" }]) {
    c.check(`split ${String(odd)}`, () => port.splitCompositionPrompt(odd), () => app.splitCompositionPrompt(odd));
  }
  c.done(PROMPTS.length + RANDOM_PROMPTS.length);
});

test("tagKey matches the app's module-private tagKey", { skip: SKIP }, async () => {
  const { internals } = await origin();
  const c = collector();
  const pieces = new Set([...PROMPTS, ...RANDOM_PROMPTS]);
  for (const text of [...PROMPTS, ...RANDOM_PROMPTS]) for (const piece of port.splitCompositionPrompt(text)) pieces.add(piece);
  for (const piece of pieces) {
    c.check(() => `tagKey ${JSON.stringify(piece)}`, () => port.tagKey(piece), () => internals.tagKey(piece), { strings: true });
  }
  c.done(pieces.size);
});

function tagListsFrom(app) {
  const lists = new Map();
  const add = (list) => lists.set(JSON.stringify(list), list);
  add([]);
  for (const preset of app.COMPOSITION_PRESETS) add(app.compositionTags(preset.state));
  for (const azimuth of [-180, -150, -90, -40, 0, 40, 90, 150, 180]) {
    for (const elevation of [-40, 0, 30, 70]) {
      for (const distance of [0, 20, 45, 60, 75, 100]) {
        for (const roll of [-12, 0, 12]) {
          for (const framing of [-50, 0, 50]) add(app.compositionTags({ azimuth, elevation, distance, roll, framing }));
        }
      }
    }
  }
  [["from above", "full body", "from above", "wide shot"], ["From_Above", "(from above:1.2)", "", "  ", "from  above"],
    ["city"], ["dutch angle"], ["a, b"], ["(x, y)"], ["\\(sign\\)"], [","], ["from above", "from below"],
    ["close-up", "portrait", "upper body", "cowboy shot", "full body", "wide shot"],
    ["centered composition", "subject on the left side of the frame", "subject on the right side of the frame"],
    ["facing viewer", "from behind"], ["FACING VIEWER"], ["(facing viewer)"], ["((facing_viewer:1.1))"]].forEach(add);
  return [...lists.values()];
}

const OTHER_SECTIONS = [undefined, "", " ", "upper_body", "((full_body)), wide_shot", "from_behind", "(from above, x)",
  "a,,b", "from above, close-up, centered composition", "facing viewer,"];

// 태그 목록이 달라질 때 갈리는 것은 이미 적힌 구도 태그와의 겹침·충돌이다 — 모든 태그 목록은 구도 태그가 든 프롬프트와
// 구분자 경우들(PLAN_FOCUS)에, 말뭉치 전체는 고른 태그 목록 × 모든 다른 섹션 값에 돌린다(둘 다 합치면 수십만 번이라 느리다).
const PLAN_FOCUS = ["", " ", ",", "forest", "forest,", "forest, ", "forest,\n", "from below, close-up",
  "from below, full body", "(from above, blue sky:1.1)", "facing viewer, upper body, centered composition",
  "FACING_VIEWER, Upper_Body, (centered composition:1.3)", "((full_body)), wide_shot", "(((from above)))",
  "(from above:0.8), [from below]", "photo of a girl, from side, profile, view from the subject's right",
  "three-quarter view from the subject's left, cowboy shot", "dutch angle, frame tilted clockwise",
  "subject on the left side of the frame", "subject_on_the_right_side_of_the_frame", "bird's eye view, from above",
  "(close-up:1.4), portrait", "from behind, rear three-quarter view from the subject's left",
  "(upper body: 1.2), (cowboy shot:-0.5), (wide shot:+1), (full body:.7), (portrait:1.)",
  "From Above, FROM_BELOW, Close_Up, Wide Shot", "centered_composition", "(centered composition)",
  "rain, (from_above:1.15), (city, trees:1.2), \\(sign\\),  "];

test("planCompositionAppend matches: every distinct tag list, and the corpus × every other section", { skip: SKIP }, async () => {
  const { app } = await origin();
  const c = collector();
  const lists = tagListsFrom(app);
  for (const text of PLAN_FOCUS) {
    for (const tags of lists) {
      c.check(() => `plan ${JSON.stringify(text)} ${JSON.stringify(tags)}`, () => port.planCompositionAppend(text, tags),
        () => app.planCompositionAppend(text, tags), { strings: true });
    }
  }
  const sampled = [...lists.slice(0, 1 + app.COMPOSITION_PRESETS.length), ...lists.filter((_, i) => i % 97 === 0),
    ...lists.slice(-15)];
  for (const text of PROMPTS) {
    for (const tags of sampled) {
      for (const other of OTHER_SECTIONS) {
        const run = (impl) => () => (other === undefined ? impl.planCompositionAppend(text, tags) : impl.planCompositionAppend(text, tags, other));
        c.check(() => `plan ${JSON.stringify(text)} ${JSON.stringify(tags)} ${JSON.stringify(other)}`, run(port), run(app),
          { strings: true });
      }
    }
  }
  assert.ok(lists.length > 1000, `${lists.length} distinct tag lists`);
  c.done(PLAN_FOCUS.length * lists.length + PROMPTS.length * sampled.length * OTHER_SECTIONS.length);
});

test("planCompositionAppend matches on random prompts, random tag picks and odd inputs", { skip: SKIP }, async () => {
  const { app } = await origin();
  const c = collector();
  const lists = tagListsFrom(app);
  const random = mulberry32(7);
  const tagPool = [...new Set(lists.flat()), ...RANDOM_PROMPTS.slice(0, 200)];
  for (const text of RANDOM_PROMPTS) {
    const picks = Array.from({ length: 1 + Math.floor(random() * 6) }, () => tagPool[Math.floor(random() * tagPool.length)]);
    const other = RANDOM_PROMPTS[Math.floor(random() * RANDOM_PROMPTS.length)];
    for (const tags of [picks, lists[Math.floor(random() * lists.length)]]) {
      c.check(() => `plan ${JSON.stringify(text)}`, () => port.planCompositionAppend(text, tags, other),
        () => app.planCompositionAppend(text, tags, other), { strings: true });
      c.check(() => `plan ${JSON.stringify(text)} (no other)`, () => port.planCompositionAppend(text, tags),
        () => app.planCompositionAppend(text, tags), { strings: true });
    }
  }
  for (const [current, tags, other] of [[null, [], ""], [undefined, ["a"]], ["a", null], ["a", "from above"],
    ["a", [null]], ["a", [5]], [5, ["a"]], ["a", ["b"], null], ["a", ["b"], 0], ["a", ["b"], ["c", "d"]]]) {
    c.check(`plan odd ${String(current)}`, () => port.planCompositionAppend(current, tags, other),
      () => app.planCompositionAppend(current, tags, other));
  }
  c.done(RANDOM_PROMPTS.length * 4);
});
