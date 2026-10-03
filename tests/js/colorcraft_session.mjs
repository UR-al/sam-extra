// Colorcraft 공유 편집기 세션 모델 — 무작위 사용자 편집 · 선택 줄 전환(sync) · Reset(reset)을 실제 js 처리기로 돌리고,
// Gradio 4.40 의 갱신 규칙(SKIP 은 값을 그대로, {choices, value} 는 값을 바꿈)으로 적용한다. 단계마다 Generate 가 보낼
// 67개 스크립트 인자와 사용자가 실제로 정한 값(경로 → 값)을 기록한다. tests/test_colorcraft_editor_parity.py 가 stdin 으로
// {seed, sessions, steps} 를 넘기고, 인자를 파이썬으로 읽어 그 값과 비교한다. (.test.mjs 가 아니라 테스트로는 돌지 않는다.)
import { readFileSync } from "node:fs";
import { loadEditor, callJs } from "./colorcraft_bridge.mjs";

const env = loadEditor();
const S = env.schema;
const NM = S.modifierFields.length, NL = S.leafFields.length, NC = S.comboFields.length, NE = NM + NL + NC;
const js = (name, ...extra) => `(...a) => window.samextraColorcraft.${name}(${["'txt2img'", ...extra.map((e) => `'${e}'`), "a"].join(", ")})`;

function rng(seed) {
    let s = seed >>> 0;
    return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

function defaultOf(tag, f) {
    const i = S.modifierTags.indexOf(tag);
    return i >= 0 && f.name === "active" ? S.activeDefaults[i] : f.default;
}

function randomValue(r, f, tag) {
    if (typeof f.default === "boolean") return r() < 0.5;
    if (typeof f.default === "number") {
        const x = r();
        if (x < 0.2) return f.default;
        if (x < 0.3) return 7.5;                     // typed past the range
        return Math.round((r() * 4 - 2) * 100) / 100;
    }
    if (f.name === "mask_a" || f.name === "mask_b") {
        const refs = S.comboRefChoices[S.comboTags.indexOf(tag)];
        return refs[Math.floor(r() * refs.length)];
    }
    // choice fields: pick from the values Python accepts
    const pools = { kind: Object.keys(S.kindIds), pass: ["Base", "Hires", "Both"],
        mask: ["none", ...S.maskTags, ...S.comboTags], mask_axis: ["exposure", "hue", "saturation"],
        mask_mode: ["highs", "lows", "split", "range", "protect range"], operation: ["and", "or", "subtract", "xor"],
        color_shift_mode: ["default", "legacy"], chroma_plane: ["temp_tint", "lab"] };
    const pool = pools[f.name] || [f.default];
    return pool[Math.floor(r() * pool.length)];
}

function apply(args, slots, out) {
    out.forEach((v, i) => {
        const slot = slots[i];
        if (slot === null || slot === undefined) return;
        if (v && typeof v === "object" && v.__type__ === "update") {
            if (Object.prototype.hasOwnProperty.call(v, "value")) args[slot] = v.value;
        } else {
            args[slot] = v;
        }
    });
}

async function session(seed, steps) {
    const r = rng(seed);
    // args layout: 0 enabled, 1 masking, 2 state, 3 debug, 4 debug_step, 5 ref, 6.. editors; plus selectors
    const args = [true, r() < 0.5, "", false, 5, "I|M1|0"];
    for (const f of S.modifierFields) args.push(defaultOf("I", f));
    for (const f of S.leafFields) args.push(f.default);
    for (const f of S.comboFields) args.push(f.default);
    const sel = { mod: "I", mask: "M1" };
    const truth = {};
    const shown = { mod: "I", mask: "M1" };       // what the editors show (from ref)
    const records = [];
    for (let step = 0; step < steps; step++) {
        const x = r();
        if (x < 0.55) {
            // user edits one control of the shown modifier or mask node
            const onMod = r() < 0.6;
            let tag, fields, offset;
            if (onMod) { tag = shown.mod; fields = S.modifierFields; offset = 6; }
            else if (S.maskTags.includes(shown.mask)) { tag = shown.mask; fields = S.leafFields; offset = 6 + NM; }
            else { tag = shown.mask; fields = S.comboFields; offset = 6 + NM + NL; }
            const j = Math.floor(r() * fields.length);
            const value = randomValue(r, fields[j], tag);
            args[offset + j] = value;
            truth[`${tag}.${fields[j].name}`] = value;
        } else if (x < 0.85) {
            // user picks another modifier or mask: selector change -> sync
            if (r() < 0.5) sel.mod = S.modifierTags[Math.floor(r() * S.modifierTags.length)];
            else sel.mask = [...S.maskTags, ...S.comboTags][Math.floor(r() * 15)];
            const out = await callJs(env, js("sync"), [sel.mod, sel.mask, args[0], args[1], args[2], args[5], ...args.slice(6)], 70);
            const slots = [2, 5, ...Array.from({ length: NE }, (_, i) => 6 + i)];
            apply(args, slots, out);
        } else {
            const section = r() < 0.6 ? "mod" : "mask";
            const out = await callJs(env, js("reset", section), [...args.slice(0, 2), args[2], args[5], ...args.slice(6)], 66);
            apply(args, [2, 5, ...Array.from({ length: NE }, (_, i) => 6 + i)], out);
            const tag = section === "mod" ? shown.mod : shown.mask;
            const fields = section === "mod" ? S.modifierFields : S.maskTags.includes(tag) ? S.leafFields : S.comboFields;
            for (const f of fields) if (!(section === "mod" && f.name === "active")) delete truth[`${tag}.${f.name}`];
        }
        const ref = String(args[5]).split("|");
        shown.mod = ref[0];
        shown.mask = ref[1];
        records.push({ args: args.slice(), truth: { ...truth } });
    }
    return records;
}

const req = JSON.parse(readFileSync(0, "utf8"));
const all = [];
for (let k = 0; k < req.sessions; k++) all.push(await session(req.seed + k, req.steps));
process.stdout.write(JSON.stringify(all));
