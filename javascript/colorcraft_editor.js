// Colorcraft (sam-extra) — 공유 편집기의 브라우저 쪽. 수정자 I~X · 마스크 M1~M10 · 조합 C1~C5 의 값은 숨은 Textbox
// (state, sam3ext/colorcraft/panel_state.py: "" 또는 {"v":1,"rev":…, 경로: 값} — 기본값은 뺀다)에 있고, 화면의 편집기
// 한 벌이 ref("<수정자>|<마스크>|<rev>")가 가리키는 두 항목을 보여 준다. 생성은 state 위에 편집기의 현재 값을 얹어 읽으므로
// (ref 의 rev 가 state 의 rev 와 같을 때만) 고치는 동안에는 state 를 쓰지 않는다. state 를 쓰는 것은 다른 항목으로
// 옮길 때(지금 편집기 값을 state 에 넣고 새 항목을 불러오기)와 Reset 뿐이고, state · ref · 편집기를 한 번의 갱신으로 함께
// 바꾼다. 붙여넣기는 파이썬(Forge 의 붙여넣기)이 state · ref · 편집기를 한 응답으로 쓴다.
//
// 모든 함수는 Gradio 4.40 의 js 전용 이벤트(fn=None)가 부른다 — 서버 왕복이 없다. Gradio 는 inputs 값 뒤에 outputs 의
// 현재 값까지 넘기므로 앞쪽만 읽는다. undefined 를 돌려주면 아무것도 바꾸지 않는다.
(function (root) {
    "use strict";

    const S = () => root.samextraColorcraftSchema;
    const SKIP = Object.freeze({ __type__: "update" });
    const update = (props) => Object.assign({ __type__: "update" }, props);
    const skips = (n) => Array.from({ length: n }, () => SKIP);
    const has = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);
    const MARK_ON = "●";
    const MARK_EDITED = "○";
    const INCOMPLETE = "(미완성)";

    const nMod = () => S().modifierFields.length;
    const nLeaf = () => S().leafFields.length;
    const nCombo = () => S().comboFields.length;
    const nEditors = () => nMod() + nLeaf() + nCombo();

    // spec.arg_names() 에서 전역 넷을 뺀 575 경로(수정자 → 잎 → 조합 순) — state 를 쓰는 순서
    let pathCache = null;
    function statePaths() {
        if (pathCache) return pathCache;
        const s = S();
        const out = [];
        for (const tag of s.modifierTags) for (const f of s.modifierFields) out.push(`${tag}.${f.name}`);
        for (const tag of s.maskTags) for (const f of s.leafFields) out.push(`${tag}.${f.name}`);
        for (const tag of s.comboTags) for (const f of s.comboFields) out.push(`${tag}.${f.name}`);
        pathCache = out;
        return out;
    }

    // ------------------------------------------------------------------ state · ref (순수 함수)

    // 쓰는 쪽(newRev · 파이썬 new_rev)의 rev 만 rev 다: 빈 문자열이 아니고 "|" 가 없는 문자열. 처음 rev "0" 은 빈 state
    // 의 것이라, 비어 있지 않은 state 의 "0" 은 rev 없음으로 본다(panel_state._rev 와 같음 — 편집기를 얹지 않는다).
    function writerRev(value) {
        return typeof value === "string" && value !== "" && !value.includes("|") && value !== S().initialRev
            ? value : null;
    }

    // {map, rev} — "" 는 기본 state(rev "0"). 읽을 수 없으면 null.
    function parseState(text) {
        if (text === null || text === undefined || text === "") return { map: {}, rev: S().initialRev };
        let data;
        try {
            data = JSON.parse(text);
        } catch (e) {
            return null;
        }
        if (!data || typeof data !== "object" || Array.isArray(data)) return null;
        if (has(data, "v") && data.v !== S().version) return null;
        const rev = writerRev(data.rev);
        const map = {};
        for (const path of statePaths()) if (has(data, path)) map[path] = data[path];
        return { map, rev };
    }

    function parseRef(text) {
        const parts = String(text || "").split("|");
        if (parts.length !== 3) return null;
        const [mod, mask, rev] = parts;
        const s = S();
        if (!s.modifierTags.includes(mod) || !(s.maskTags.includes(mask) || s.comboTags.includes(mask)) || !rev) {
            return null;
        }
        return { mod, mask, rev };
    }

    function serialize(map, rev) {
        const out = { v: S().version, rev };
        for (const path of statePaths()) if (has(map, path)) out[path] = map[path];
        return JSON.stringify(out);
    }

    function newRev() {
        const c = root.crypto;
        if (c && c.getRandomValues) {
            const bytes = c.getRandomValues(new Uint8Array(6));
            return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
        }
        return Math.random().toString(16).slice(2, 14).padEnd(12, "0");
    }

    function defaultOf(tag, f) {
        const s = S();
        const index = s.modifierTags.indexOf(tag);
        return index >= 0 && f.name === "active" ? s.activeDefaults[index] : f.default;
    }

    function fieldsOf(tag) {
        const s = S();
        if (s.modifierTags.includes(tag)) return s.modifierFields;
        if (s.maskTags.includes(tag)) return s.leafFields;
        return s.comboFields;
    }

    // 파이썬 coerce 가 기본값으로 읽을 값(빈 숫자 칸 등)과 기본값 자체는 state 에서 지운다.
    function unusable(value) {
        return value === null || value === undefined || (typeof value === "number" && !Number.isFinite(value));
    }

    // 조합의 Mask A/B 는 그 조합이 고를 수 있는 것만(spec.combo_ref_choices) — 아니면 파이썬 coerce 처럼 기본값.
    function allowed(tag, f, value) {
        const s = S();
        const ci = s.comboTags.indexOf(tag);
        return ci < 0 || (f.name !== "mask_a" && f.name !== "mask_b") || s.comboRefChoices[ci].includes(value);
    }

    function writeNode(map, tag, values) {
        fieldsOf(tag).forEach((f, j) => {
            const path = `${tag}.${f.name}`;
            const value = values[j];
            if (unusable(value) || value === defaultOf(tag, f) || !allowed(tag, f, value)) delete map[path];
            else map[path] = value;
        });
    }

    // 편집기 61칸(수정자 44 · 잎 10 · 조합 7)을 ref 가 가리키는 두 항목에 써 넣는다.
    function commit(map, ref, editors) {
        const s = S();
        writeNode(map, ref.mod, editors.slice(0, nMod()));
        if (s.maskTags.includes(ref.mask)) writeNode(map, ref.mask, editors.slice(nMod(), nMod() + nLeaf()));
        else writeNode(map, ref.mask, editors.slice(nMod() + nLeaf(), nEditors()));
        return map;
    }

    function valueOf(map, tag, f) {
        const path = `${tag}.${f.name}`;
        return has(map, path) && map[path] !== null ? map[path] : defaultOf(tag, f);
    }

    function field(tag, name) {
        return fieldsOf(tag).find((f) => f.name === name);
    }

    // ------------------------------------------------------------------ 편집기 값

    function modValues(map, tag) {
        return S().modifierFields.map((f) => valueOf(map, tag, f));
    }

    function leafValues(map, tag) {
        return S().leafFields.map((f) => valueOf(map, tag, f));
    }

    function comboValues(map, tag) {
        const s = S();
        const refs = s.comboRefChoices[s.comboTags.indexOf(tag)].map((c) => [c, c]);
        return s.comboFields.map((f) => {
            const value = valueOf(map, tag, f);
            return f.name === "mask_a" || f.name === "mask_b" ? update({ choices: refs, value }) : value;
        });
    }

    // [잎 칸 10, 조합 칸 7, 잎 그룹, 조합 그룹]
    function maskEditor(map, tag) {
        const s = S();
        if (s.maskTags.includes(tag)) {
            return [...leafValues(map, tag), ...skips(nCombo()), update({ visible: true }), update({ visible: false })];
        }
        return [...skips(nLeaf()), ...comboValues(map, tag), update({ visible: false }), update({ visible: true })];
    }

    // ------------------------------------------------------------------ 한눈에 (panel_state.overview 와 같은 글자)

    function kindId(label) {
        return S().kindIds[label] || "advanced";
    }

    function isActive(map, tag) {
        return valueOf(map, tag, field(tag, "active")) === true;
    }

    function edited(map, tag) {
        return fieldsOf(tag).some((f) => f.name !== "active" && valueOf(map, tag, f) !== defaultOf(tag, f));
    }

    // spec.build_mask_specs 가 풀어 내는 조합: A·B 가 둘 다 마스크이거나 앞에서 풀린 조합일 때만. 못 풀린 조합을 고른
    // 수정자는 마스크 없이 돈다(훅의 "incomplete combo, unmasked").
    function resolvedCombos(map) {
        const s = S();
        const done = new Set();
        const ok = (ref) => s.maskTags.includes(ref) || done.has(ref);
        for (const tag of s.comboTags) {
            const a = valueOf(map, tag, field(tag, "mask_a"));
            const b = valueOf(map, tag, field(tag, "mask_b"));
            if (ok(a) && ok(b)) done.add(tag);
        }
        return done;
    }

    // 켠 수정자의 Mask ➜ 에서 (풀린 조합의 A·B 를 따라) 실제로 쓰는 마스크·조합 — spec.active_masks 에서 못 풀린 조합을 뺀 것
    function reached(map, masking) {
        const s = S();
        const seen = new Set();
        if (masking !== true) return seen;
        const resolved = resolvedCombos(map);
        const pending = [];
        for (const tag of s.modifierTags) {
            const mask = valueOf(map, tag, field(tag, "mask"));
            if (isActive(map, tag) && s.kindTakesMask[kindId(valueOf(map, tag, field(tag, "kind")))]
                && mask !== s.maskNone) pending.push(mask);
        }
        while (pending.length) {
            const tag = pending.pop();
            if (tag === s.maskNone || seen.has(tag) || (s.comboTags.includes(tag) && !resolved.has(tag))) continue;
            seen.add(tag);
            if (s.comboTags.includes(tag)) {
                pending.push(valueOf(map, tag, field(tag, "mask_a")), valueOf(map, tag, field(tag, "mask_b")));
            }
        }
        return seen;
    }

    function escapeHtml(text) {
        return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;").replace(/'/g, "&#x27;");
    }

    function overview(map, enabled, masking) {
        const s = S();
        const used = reached(map, masking);
        const resolved = resolvedCombos(map);
        const active = [];
        const mods = s.modifierTags.map((tag) => {
            let mark = "";
            if (isActive(map, tag)) {
                mark = MARK_ON;
                const kind = valueOf(map, tag, field(tag, "kind"));
                const mask = valueOf(map, tag, field(tag, "mask"));
                const pass = valueOf(map, tag, field(tag, "pass"));
                let text = `${tag} ${kind}`;
                if (masking === true && s.kindTakesMask[kindId(kind)] && mask !== s.maskNone) {
                    text += ` ➜ ${mask}${s.comboTags.includes(mask) && !resolved.has(mask) ? INCOMPLETE : ""}`;
                }
                if (pass !== s.passBase) text += ` (${pass})`;
                active.push(text);
            } else if (edited(map, tag)) {
                mark = MARK_EDITED;
            }
            return [mark ? `${tag} ${mark}` : tag, tag];
        });
        const masks = s.maskTags.concat(s.comboTags).map((tag) => {
            const mark = used.has(tag) ? MARK_ON : edited(map, tag) ? MARK_EDITED : "";
            return [mark ? `${tag} ${mark}` : tag, tag];
        });
        let text = active.length ? `켠 수정자: ${active.join(", ")}` : "켠 수정자 없음";
        if (masking === true) {
            const order = s.maskTags.concat(s.comboTags).filter((t) => used.has(t));
            text += ` · 쓰는 마스크: ${order.length ? order.join(", ") : "없음"}`;
        }
        if (enabled !== true) text = `꺼짐 — ${text}`;
        return { mods, masks, html: `<div class="samextra-cc-summary">${escapeHtml(text)}</div>` };
    }

    function viewOutputs(map, enabled, masking) {
        const v = overview(map, enabled, masking);
        return [update({ choices: v.mods }), update({ choices: v.masks }), v.html];
    }

    // ------------------------------------------------------------------ 빠른 드롭다운(notebook.js) 맞추기

    // sam-extra 의 빠른 드롭다운(javascript/notebook.js)은 Gradio 드롭다운을 숨기고 그 값·선택지를 0.8초마다 다시 읽는다.
    // 편집기가 값을 불러오면 Gradio 가 반영한 뒤(두 프레임 뒤) 바로 맞춰 준다 — 고른 항목의 Type 이 늦게 바뀌어 보이거나 조합의
    // Mask A/B 목록이 잠깐 앞 조합의 것으로 남지 않게. 빠른 드롭다운이 없으면 아무것도 하지 않는다. 생성 값과는 무관하다.
    function nudgeProxies(tab, comboRefs) {
        const raf = root.requestAnimationFrame;
        const doc = root.document;
        if (typeof raf !== "function" || !doc || (tab !== "txt2img" && tab !== "img2img")) return;
        const pre = `script_${tab}_colorcraft_samextra_`;
        raf(() => raf(() => {
            try {
                if (comboRefs) {
                    for (const name of ["mask_a", "mask_b"]) {
                        const w = doc.getElementById(`${pre}combo_${name}`);
                        if (w && typeof w.__sam3FastDropdownSetChoices === "function") w.__sam3FastDropdownSetChoices(comboRefs);
                    }
                }
                const acc = doc.getElementById(`${pre}accordion`);
                if (!acc) return;
                acc.querySelectorAll(".gradio-dropdown").forEach((w) => {
                    const input = w.querySelector("input[role='listbox']");
                    if (input && typeof w.__sam3FastDropdownSync === "function") w.__sam3FastDropdownSync(input.value);
                });
            } catch (e) {
                if (root.console) root.console.error("[Colorcraft] nudge:", e);
            }
        }));
    }

    // ------------------------------------------------------------------ Gradio js 이벤트

    // 선택 줄(수정자·마스크)이 바뀌었거나 state 가 바뀌었을 때.
    // inputs  [mod_select, mask_select, enabled, masking, state, ref, ...편집기 61]
    // outputs [state, ref, ...편집기 61, 잎 그룹, 조합 그룹, mod_reset, mask_reset, mod_select, mask_select, summary]
    function sync(tab, a) {
        const s = S();
        const [modT, maskT, enabled, masking, stateText, refText] = a;
        const editors = a.slice(6, 6 + nEditors());
        if (!s.modifierTags.includes(modT) || !(s.maskTags.includes(maskT) || s.comboTags.includes(maskT))) {
            return undefined;
        }
        const st = parseState(stateText);
        if (st === null) return undefined;
        const ref = parseRef(refText);
        const fresh = ref !== null && st.rev !== null && ref.rev === st.rev;
        const sameMod = fresh && ref.mod === modT;
        const sameMask = fresh && ref.mask === maskT;
        const combo = s.comboTags.indexOf(maskT);
        nudgeProxies(tab, !sameMask && combo >= 0 ? s.comboRefChoices[combo] : null);
        if (sameMod && sameMask) return undefined;          // 이미 맞다(자기 쓰기의 메아리, 붙여넣기 직후)
        const map = st.map;
        let rev = st.rev;
        let stateOut = SKIP;
        if (fresh) {
            commit(map, ref, editors);                        // 떠나는 항목의 편집을 state 에
            rev = newRev();
            stateOut = serialize(map, rev);
        } else if (rev === null) {
            rev = newRev();                                   // rev 없는 state(바깥에서 온 것)는 rev 를 붙여 다시 쓴다
            stateOut = serialize(map, rev);
        }
        // fresh 가 아니면 편집기 값은 이 state 의 것이 아니다 — 둘 다 state 에서 다시 읽는다.
        return [stateOut, `${modT}|${maskT}|${rev}`,
            ...(sameMod ? skips(nMod()) : modValues(map, modT)),
            ...(sameMask ? skips(nLeaf() + nCombo() + 2) : maskEditor(map, maskT)),
            sameMod ? SKIP : update({ value: `Reset ${modT}` }),
            sameMask ? SKIP : update({ value: `Reset ${maskT}` }),
            ...viewOutputs(map, enabled, masking)];
    }

    // Reset 버튼. section = "mod"(Active 는 그대로 — v0.31.0 과 같음) 또는 "mask".
    // inputs  [enabled, masking, state, ref, ...편집기 61]
    // outputs [state, ref, ...편집기 61, mod_select, mask_select, summary]
    function reset(tab, section, a) {
        const s = S();
        const [enabled, masking, stateText, refText] = a;
        const editors = a.slice(4, 4 + nEditors());
        const st = parseState(stateText);
        const ref = parseRef(refText);
        if (st === null || ref === null) return undefined;
        const fresh = st.rev !== null && ref.rev === st.rev;
        const map = st.map;
        if (fresh) commit(map, ref, editors);
        const tag = section === "mod" ? ref.mod : ref.mask;
        for (const f of fieldsOf(tag)) {
            if (!(section === "mod" && f.name === "active")) delete map[`${tag}.${f.name}`];
        }
        const rev = newRev();
        nudgeProxies(tab, null);
        const reloadMod = section === "mod" || !fresh;
        const reloadMask = section === "mask" || !fresh;
        const mask = s.maskTags.includes(ref.mask)
            ? [...leafValues(map, ref.mask), ...skips(nCombo())]
            : [...skips(nLeaf()), ...comboValues(map, ref.mask)];
        return [serialize(map, rev), `${ref.mod}|${ref.mask}|${rev}`,
            ...(reloadMod ? modValues(map, ref.mod) : skips(nMod())),
            ...(reloadMask ? mask : skips(nLeaf() + nCombo())),
            ...viewOutputs(map, enabled, masking)];
    }

    // 편집기 값이 바뀔 때마다(.input), Enable·masking 이 바뀔 때: 선택 줄의 ●/○ 와 요약만.
    // inputs  [enabled, masking, state, ref, ...편집기 61]   outputs [mod_select, mask_select, summary]
    function labels(tab, a) {
        const [enabled, masking, stateText, refText] = a;
        const st = parseState(stateText);
        if (st === null) return undefined;
        const ref = parseRef(refText);
        if (ref !== null && st.rev !== null && ref.rev === st.rev) commit(st.map, ref, a.slice(4, 4 + nEditors()));
        return viewOutputs(st.map, enabled, masking);
    }

    // Type 이 바뀌면 그 노드의 칸만 보인다(spec.KIND_GROUPS · KIND_GATES · KIND_TAKES_MASK).
    // inputs [kind]   outputs [그룹 7, Apply Chroma Plus, Apply Color Shift, Mask ➜]
    function kind(tab, a) {
        const s = S();
        const id = kindId(a[0]);
        const shown = new Set(s.kindGroups[id]);
        const gates = new Set(s.kindGates[id] || []);
        return [...s.groupOrder.map((g) => update({ visible: shown.has(g) })),
            update({ visible: gates.has("more_colors") }), update({ visible: gates.has("color_shift") }),
            update({ visible: !!s.kindTakesMask[id] })];
    }

    // 처리 중 예외는 콘솔에 남기고 failed 를 돌려준다(Gradio 이벤트가 깨지지 않게). sync · reset · labels 는 null —
    // 페이지의 js 문자열(sam3ext/colorcraft/ui.py _js)이 그때 선택 줄을 ref 의 항목으로 되돌리고 요약 줄에 경고를 쓴다
    // (이 파일이 아예 없을 때와 같다). kind 는 undefined(아무것도 바꾸지 않음).
    function guard(name, fn, failed) {
        return function (...args) {
            try {
                return fn(...args);
            } catch (e) {
                if (root.console) root.console.error(`[Colorcraft] ${name}:`, e);
                return failed;
            }
        };
    }

    root.samextraColorcraft = {
        sync: guard("sync", sync, null), reset: guard("reset", reset, null), labels: guard("labels", labels, null),
        kind: guard("kind", kind, undefined),
        // 테스트용 순수 함수
        _pure: { parseState, parseRef, serialize, commit, overview, reached, resolvedCombos, modValues, leafValues,
            comboValues },
    };
})(typeof window !== "undefined" ? window : globalThis);
