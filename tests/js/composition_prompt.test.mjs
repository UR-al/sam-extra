// composition_prompt.js — 구도 · 카메라의 태그 계산(사용자 앱 compositionPrompt.ts 이식).
//
// 앞부분은 앱의 frontend/src/utils/compositionPrompt.test.ts(앱 커밋 d2fcc707) 사례를 vitest 에서 node:test 로 옮긴
// 것이다 — 기대값은 그대로다. 뒤는 이식본에만 있는 것(앱에서 모듈 안에 숨어 있던 tagKey, 얼린 상수, 전역 이름).
// 원본 파일 자체와의 대조는 composition_prompt_origin.test.mjs 가 한다.
// 실제 javascript/composition_prompt.js 를 이 realm 에서 실행한다(vm.runInThisContext) — jsdom 의 다른 realm 에서 만든
// 배열은 deepStrictEqual 의 프로토타입 비교에 걸린다.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import vm from "node:vm";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const SOURCE = path.join(ROOT, "javascript", "composition_prompt.js");
vm.runInThisContext(readFileSync(SOURCE, "utf8"), { filename: SOURCE });
const {
  COMPOSITION_PRESETS, DEFAULT_COMPOSITION, compositionCameraPoint, compositionTags,
  dragComposition, normalizeComposition, planCompositionAppend, splitCompositionPrompt, tagKey,
} = globalThis.sam3CompositionPrompt;

// ---- 앱 compositionPrompt.test.ts ------------------------------------------------------

test("normalizes corrupt or obsolete preferences without accepting executable values", () => {
  assert.deepEqual(normalizeComposition(null), { ...DEFAULT_COMPOSITION });
  assert.deepEqual(normalizeComposition({ azimuth: Infinity, elevation: 100, distance: -4, roll: "24", framing: NaN }),
    { azimuth: 0, elevation: 75, distance: 0, roll: 0, framing: 0 });
});

test("uses neutral, unweighted descriptive defaults", () => {
  assert.deepEqual(compositionTags({ ...DEFAULT_COMPOSITION }), ["facing viewer", "upper body", "centered composition"]);
  for (const preset of COMPOSITION_PRESETS) assert.doesNotMatch(compositionTags(preset.state).join(","), /:\d|masterpiece|1girl/);
});

for (const [azimuth, expected] of [
  [-180, "from behind"], [-150, "rear three-quarter view from the subject's left"],
  [-90, "profile"], [-40, "three-quarter view from the subject's left"],
  [0, "facing viewer"], [40, "three-quarter view from the subject's right"], [180, "from behind"],
]) {
  test(`maps azimuth ${azimuth} to descriptive prompt ${expected}`, () => {
    assert.ok(compositionTags({ ...DEFAULT_COMPOSITION, azimuth }).includes(expected));
  });
}

test("keeps contradictory elevation and crop descriptions mutually exclusive", () => {
  const high = compositionTags({ ...DEFAULT_COMPOSITION, elevation: 70, distance: 100, roll: -12, framing: -50 });
  for (const tag of ["from above", "bird's eye view", "wide shot", "dutch angle", "frame tilted counterclockwise",
    "subject on the left side of the frame"]) {
    assert.ok(high.includes(tag), tag);
  }
  assert.ok(!high.includes("from below"));
  assert.ok(!high.includes("close-up"));
  assert.ok(compositionTags({ ...DEFAULT_COMPOSITION, elevation: -40, distance: 5 }).includes("from below"));
  assert.ok(!compositionTags({ ...DEFAULT_COMPOSITION, roll: 5 }).includes("dutch angle"));
});

for (const [distance, expected] of [[0, "close-up"], [20, "portrait"], [45, "upper body"], [60, "cowboy shot"],
  [75, "full body"], [100, "wide shot"]]) {
  test(`maps distance ${distance} to exactly one crop ${expected}`, () => {
    assert.ok(compositionTags({ ...DEFAULT_COMPOSITION, distance }).includes(expected));
  });
}

test("splits only top-level commas without changing schedules, weights or escaped literals", () => {
  assert.deepEqual(splitCompositionPrompt("a, (b, c:1.1), [d:e:0.5], {f|g,h}, <lora:a,b:1>, escaped\\,comma"),
    ["a", "(b, c:1.1)", "[d:e:0.5]", "{f|g,h}", "<lora:a,b:1>", "escaped\\,comma"]);
});

test("preserves existing text byte-for-byte and deduplicates normalized complete tags across sections", () => {
  const original = "rain, (from_above:1.15), (city, trees:1.2), \\(sign\\),  ";
  const result = planCompositionAppend(original, ["from above", "full body", "from above", "wide shot"], "((full_body)), wide_shot");
  assert.deepEqual(result.additions, []);
  assert.equal(result.text, original);
  assert.equal(planCompositionAppend("cityscape", ["city"]).text, "cityscape, city");
  assert.equal(planCompositionAppend(original, ["dutch angle"]).text, original + "dutch angle");
});

test("warns about opposed prior tags instead of silently removing or replacing them", () => {
  const result = planCompositionAppend("from below, close-up", ["from above", "full body"], "from_behind");
  assert.deepEqual(result.conflicts, ["from below ↔ from above", "close-up ↔ full body"]);
  assert.equal(result.text, "from below, close-up, from above, full body");
});

test("does not treat part of a weighted multi-tag group as a removable duplicate", () => {
  const result = planCompositionAppend("(from above, blue sky:1.1)", ["from above"]);
  assert.equal(result.text, "(from above, blue sky:1.1), from above");
});

test("is idempotent on repeated apply including a trailing comma", () => {
  const tags = compositionTags({ ...DEFAULT_COMPOSITION });
  const first = planCompositionAppend("forest,", tags);
  assert.equal(first.text, "forest, facing viewer, upper body, centered composition");
  assert.deepEqual(planCompositionAppend(first.text, tags).additions, []);
  assert.equal(planCompositionAppend("", tags).text.startsWith(","), false);
});

test("computes gestures from the starting state and clamps rapid out-of-bounds input", () => {
  const initial = { ...DEFAULT_COMPOSITION };
  const moved = dragComposition(initial, 100, -50);
  assert.deepEqual([moved.azimuth, moved.elevation], [70, 30]);
  assert.deepEqual(dragComposition(initial, 0, 0), initial);
  const clamped = dragComposition(initial, 9999, -9999);
  assert.deepEqual([clamped.azimuth, clamped.elevation], [180, 75]);
  assert.deepEqual(initial, { ...DEFAULT_COMPOSITION });
});

test("keeps camera diagram geometry in bounds across extreme controls", () => {
  for (const azimuth of [-180, -90, 0, 90, 180]) {
    for (const elevation of [-75, 0, 75]) {
      for (const distance of [0, 100]) {
        const point = compositionCameraPoint({ ...DEFAULT_COMPOSITION, azimuth, elevation, distance });
        assert.ok(point.x > 15 && point.x < 285, `x ${point.x}`);
        assert.ok(point.y > 15 && point.y < 155, `y ${point.y}`);
      }
    }
  }
});

// ---- 이식본에만 있는 것 ------------------------------------------------------------------

test("tagKey is the app's normalized top-level tag: case, underscores, complete weights only", () => {
  assert.equal(tagKey("  From_Above  "), "from above");
  assert.equal(tagKey("((From_Above:1.2))"), "from above");
  assert.equal(tagKey("(full body:.85)"), "full body");
  assert.equal(tagKey("(full body: -1.)"), "full body");
  assert.equal(tagKey("(full   body)"), "full body");
  assert.equal(tagKey("\\(sign\\)"), "\\(sign\\)", "escaped parentheses are literal text, never unwrapped");
  assert.equal(tagKey("(a, b:1.1)"), "(a, b:1.1)", "a weighted group of several tags stays one opaque key");
  // the app only asks whether the inside is one top-level piece — "a) (b" is — so this unwraps; kept as in the app
  assert.equal(tagKey("(a) (b)"), "a) (b");
  assert.equal(tagKey("[from above]"), "[from above]", "only round brackets are attention weights here");
  assert.equal(tagKey(""), "");
});

test("the API and its constants are frozen so other page scripts cannot change them", () => {
  const api = globalThis.sam3CompositionPrompt;
  assert.ok(Object.isFrozen(api));
  for (const value of [api.DEFAULT_COMPOSITION, api.COMPOSITION_CONTROLS, api.COMPOSITION_PRESETS, api.CONTRAST_GROUPS]) {
    assert.ok(Object.isFrozen(value));
  }
  for (const preset of api.COMPOSITION_PRESETS) assert.ok(Object.isFrozen(preset.state), preset.name);
  for (const group of api.CONTRAST_GROUPS) assert.ok(Object.isFrozen(group));
  const state = normalizeComposition(api.COMPOSITION_PRESETS[1].state);
  state.azimuth = 1;   // results are fresh, writable objects
  assert.equal(api.COMPOSITION_PRESETS[1].state.azimuth, -35);
});

test("the file is a plain browser script: one global, no module syntax", () => {
  const source = readFileSync(SOURCE, "utf8");
  assert.doesNotMatch(source, /^\s*(import|export)\s/m);
  assert.match(source, /root\.sam3CompositionPrompt = Object\.freeze\(\{/);
  assert.doesNotThrow(() => new vm.Script(source, { filename: SOURCE }));
});
