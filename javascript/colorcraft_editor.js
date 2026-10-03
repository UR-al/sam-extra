// Colorcraft (sam-extra) — 공유 편집기의 브라우저 쪽. 수정자 I~X · 마스크 M1~M10 · 조합 C1~C5 의 값은 숨은 Textbox
// (state, sam3ext/colorcraft/panel_state.py: "" 또는 {"v":1,"rev":…, 경로: 값} — 기본값은 뺀다)에 있고, 화면의 편집기
// 한 벌이 ref("<수정자>|<마스크>|<rev>")가 가리키는 두 항목을 보여 준다. 생성은 state 위에 편집기의 현재 값을 얹어 읽으므로
// (ref 의 rev 가 state 의 rev 와 같을 때만) 고치는 동안에는 state 를 쓰지 않는다. state 를 쓰는 것은 다른 항목으로
// 옮길 때(지금 편집기 값을 state 에 넣고 새 항목을 불러오기)와 Reset 뿐이고, state · ref · 편집기를 한 번의 갱신으로 함께
// 바꾼다. 붙여넣기는 파이썬(Forge 의 붙여넣기)이 state · ref · 편집기를 한 응답으로 쓴다.
//
// 모든 함수는 Gradio 4.40 의 js 전용 이벤트(fn=None)가 부른다 — 서버 왕복이 없다. Gradio 는 inputs 값 뒤에 outputs 의
// 현재 값까지 넘기므로 앞쪽만 읽는다. undefined 를 돌려주면 아무것도 바꾸지 않는다.
//
// 화면 낭독기용(접근성): 선택 줄 두 개(Gradio Radio 의 fieldset)에 이름·설명을 잇고, 편집기 칸마다 지금 편집하는 항목을
// 설명(aria-describedby)으로 붙이며, 다른 항목으로 옮기거나 Reset · 붙여넣기 뒤에 알림(role="status")을 한 번 낸다. 그 글자를
// 담은 보이지 않는 노드 하나를 탭마다 요약 줄 블록 안에 마운트 때 한 번 넣고, 그 뒤로는 텍스트 노드의 .data 와 aria-*
// 속성만 바꾼다(design.md "Runtime DOM rules" — composition_ui.js 와 같은 방식). 이벤트 리스너는 붙이지 않는다.
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

    // ------------------------------------------------------------------ Gradio 가 그린 뒤 (빠른 드롭다운 · 접근성)

    const TABS = ["txt2img", "img2img"];
    const prefix = (tab) => `script_${tab}_colorcraft_samextra_`;

    function logError(what, e) {
        if (root.console) root.console.error(`[Colorcraft] ${what}:`, e);
    }

    // 처리기 안에서 표시만 하는 일(편집 중인 항목 적기 · 그린 뒤의 일 예약)을 돌린다. 여기서 무엇이 잘못돼도 처리기의 값은
    // 그대로 나간다 — guard 까지 올라가면 선택 줄이 되돌아가고 요약 줄에 경고가 뜨며, kind 는 칸을 숨기고 보이지 못한다.
    function display(what) {
        try {
            what();
        } catch (e) {
            logError("a11y", e);
        }
    }

    // Gradio 4.40 의 순서(js/app/src/Blocks.svelte · init.ts): 이벤트는 requestAnimationFrame 안에서 처리기를 부르고(프레임 F),
    // 처리기가 돌려준 값은 handle_update → update_value 가 requestAnimationFrame(flush) 로 다음 프레임(F+1)에 한꺼번에 적용하며,
    // Svelte 는 그 flush 콜백 바로 뒤의 마이크로태스크에서 그린다. 처리기는 그 rAF 가 등록되기 전(frontend_fn 안)에 불리므로,
    // 0ms 타이머로 이 이벤트의 마이크로태스크(handle_update 의 rAF 등록까지)가 다 끝나기를 기다린 뒤 프레임 콜백을 건다 — 그
    // 콜백은 F+1 에서 Gradio 의 flush 와 렌더 뒤에 돌아 값과 같은 화면 갱신에 그려진다. 두 프레임 뒤(v0.32.0)는 F+2 의 맨 앞,
    // 한 프레임 뒤의 0ms 타이머(실제 Forge 점검의 변형)는 F+1 과 F+2 사이에 돌아 DOM 은 값과 함께 바뀌어도 화면에는 F+2 가
    // 끝나야 그려진다 — F+2 는 F+1 렌더의 change 이벤트들이 다시 flush 하는 프레임이고, Forge 처럼 큰 페이지에서는 flush 마다
    // 레이아웃 전체를 다시 훑어 100ms 안팎이 걸린다. 타이머가 F+1 뒤로 밀리면(긴 프레임 직후 브라우저가 렌더를 먼저 할 때) 이
    // 콜백도 한 프레임 늦을 뿐 늘 Gradio 가 그린 뒤다(tests/js/colorcraft_editor_timing.test.mjs). 생성 값과는 무관하다.
    function afterRender(tab, job) {
        const later = root.setTimeout;
        const raf = root.requestAnimationFrame;
        const doc = root.document;
        if (typeof later !== "function" || typeof raf !== "function" || !doc || !TABS.includes(tab)) return;
        later(() => raf(() => {
            if (job.proxies) {
                try {
                    nudgeProxies(doc, tab, job.comboRefs);
                } catch (e) {
                    logError("nudge", e);
                }
            }
            try {
                refreshA11y(doc, tab, job.announce);
            } catch (e) {
                logError("a11y", e);
            }
        }), 0);
    }

    // sam-extra 의 빠른 드롭다운(javascript/notebook.js)은 Gradio 드롭다운을 숨기고 그 값·선택지를 0.8초마다 다시 읽는다.
    // 편집기가 값을 불러오면 Gradio 가 그린 직후에 맞춰 준다 — 고른 항목의 Type 이 늦게 바뀌어 보이거나 조합의 Mask A/B 목록이
    // 잠깐 앞 조합의 것으로 남지 않게. 빠른 드롭다운이 없으면 아무것도 하지 않는다.
    function nudgeProxies(doc, tab, comboRefs) {
        const pre = prefix(tab);
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
    }

    // ------------------------------------------------------------------ 접근성 (화면 낭독기)

    // 탭마다 편집기가 보여 주는 항목(알림·설명의 글자)과 이 페이지가 최근에 쓴 rev 들 — 그 밖의 rev 를 단 state 는 Forge 의
    // 붙여넣기가 쓴 것이다(파이썬만 state 를 따로 쓰고, rev 는 매번 새로 뽑는다).
    const pages = Object.create(null);
    const KEEP_REVS = 8;

    function page(tab) {
        if (!pages[tab]) {
            const s = S();
            const mod = s.modifierTags[0];
            pages[tab] = { mod, kind: defaultOf(mod, field(mod, "kind")), mask: s.maskTags[0], revs: [], a11y: null };
        }
        return pages[tab];
    }

    const wrote = (tab, rev) => page(tab).revs.includes(rev);

    function note(tab, mod, kindLabel, mask, rev) {
        const p = page(tab);
        Object.assign(p, { mod, kind: String(kindLabel), mask });
        if (!p.revs.includes(rev)) p.revs = [...p.revs, rev].slice(-KEEP_REVS);
    }

    const SR_CLASS = "samextra-cc-sr";     // style.css: 보이지 않고 자리도 차지하지 않는다
    const modNode = (p) => `수정자 ${p.mod} (${p.kind})`;
    const maskNode = (p) => `${S().comboTags.includes(p.mask) ? "조합" : "마스크"} ${p.mask}`;
    const editing = (nodes) => `편집 중: ${nodes.join(" · ")}`;
    const LOADED = "붙여넣은 설정 불러옴";
    const RESET = "초기화";

    // 알림 글자(편집기가 이제 보여 주는 항목). 다른 항목으로 옮김: 바뀐 쪽만 · Reset: 그 항목 · 붙여넣기(바깥 state): 둘 다.
    function announcement(p, what, sides) {
        if (what === "loaded") return `${LOADED} · ${editing([modNode(p), maskNode(p)])}`;
        if (what === "reset") return `${RESET}: ${sides === "mod" ? modNode(p) : maskNode(p)}`;
        return editing([...(sides.includes("mod") ? [modNode(p)] : []), ...(sides.includes("mask") ? [maskNode(p)] : [])]);
    }

    function setText(node, text) {
        if (node.data !== text) node.data = text;
    }

    function setAttr(el, name, value) {
        if (el.getAttribute(name) !== value) el.setAttribute(name, value);
    }

    function addToken(el, name, token) {
        const tokens = (el.getAttribute(name) || "").split(/\s+/).filter(Boolean);
        if (!tokens.includes(token)) el.setAttribute(name, [...tokens, token].join(" "));
    }

    // 같은 글자를 다시 알릴 때(같은 항목을 두 번 Reset)도 텍스트가 바뀌어야 낭독기가 읽는다 — 끝의 줄바꿈 없는 공백을 번갈아.
    function say(node, text) {
        node.data = node.data === text ? `${text}\u00a0` : text;
    }

    // 선택 줄(Gradio 4.40 Radio = <fieldset>, 이름은 <span data-testid="block-info">, 그 밑 info 는 <div>): fieldset 에 이름과
    // 설명을 잇는다(legend 가 없어 이름 없는 그룹이었다). 라디오마다 name 이 달라 브라우저가 "1/1" 로 읽으므로 위치·개수도 붙인다
    // (v0.31.0 탭 줄의 "n/10" 처럼). 키보드 동작(Tab 으로 이동 · Space 로 고르기)은 그대로다.
    function nameGroup(group, pre, name) {
        const title = group.querySelector(":scope > [data-testid='block-info']");
        if (title) {
            if (!title.id) title.id = `${pre}${name}_label`;
            setAttr(group, "aria-labelledby", title.id);
            const info = title.nextElementSibling;
            if (info && info.tagName === "DIV" && !info.querySelector("input") && info.textContent.trim()) {
                if (!info.id) info.id = `${pre}${name}_info`;
                setAttr(group, "aria-describedby", info.id);
            }
        }
        const radios = group.querySelectorAll("input[type='radio']");
        radios.forEach((r, i) => {
            setAttr(r, "aria-posinset", String(i + 1));
            setAttr(r, "aria-setsize", String(radios.length));
        });
    }

    // 편집기 칸(슬라이더의 숫자·막대, 체크박스, 드롭다운과 그 빠른 드롭다운 단추, Pass 라디오)에 "편집 중: …" 설명을 잇는다.
    function describeControls(wrapper, id) {
        if (!wrapper) return;
        wrapper.querySelectorAll("input, select, textarea, button").forEach((el) => addToken(el, "aria-describedby", id));
    }

    // Gradio 는 이 칸들을 다시 만들지 않지만(값·선택지만 바꿈), 빠른 드롭다운은 나중에(칸이 보일 때) 붙으므로 그릴 때마다 다시 본다.
    // 편집기의 Pass 도 같은 Radio 라 같은 방식으로 이름을 잇는다.
    function decorate(doc, tab, nodes) {
        const s = S();
        const pre = prefix(tab);
        for (const name of ["modifier_select", "mask_select", "mod_pass"]) {
            const group = doc.getElementById(`${pre}${name}`);
            if (group) nameGroup(group, pre, name);
        }
        for (const f of s.modifierFields) describeControls(doc.getElementById(`${pre}mod_${f.name}`), nodes.modId);
        for (const f of s.leafFields) describeControls(doc.getElementById(`${pre}leaf_${f.name}`), nodes.maskId);
        for (const f of s.comboFields) describeControls(doc.getElementById(`${pre}combo_${f.name}`), nodes.maskId);
    }

    function slot(doc, box, tag, id, attrs) {
        const el = doc.createElement(tag);
        el.id = id;
        for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
        const text = doc.createTextNode("");
        el.appendChild(text);
        box.appendChild(el);
        return text;
    }

    // 탭마다 한 번: 요약 줄 블록(#…_summary, Gradio 가 다시 그리는 것은 그 안의 .prose 뿐) 끝에 보이지 않는 노드 하나를 넣는다 —
    // 설명 둘(hidden: 낭독기가 따로 읽지 않고 aria-describedby 로만 쓰인다)과 알림(role="status"). 패널이 없으면 null.
    function mount(doc, tab) {
        const p = page(tab);
        if (p.a11y && p.a11y.box.isConnected) return p.a11y;
        const pre = prefix(tab);
        const host = doc.getElementById(`${pre}summary`);
        if (!host) return null;
        const modId = `${pre}a11y_mod`;
        const maskId = `${pre}a11y_mask`;
        const stale = doc.getElementById(modId);         // 이 파일을 다시 읽은 페이지: 앞 노드를 치우고 새로(아이디가 겹치지 않게)
        const old = stale && stale.closest("[data-samextra-cc-a11y]");
        if (old) old.remove();
        const box = doc.createElement("div");
        box.className = SR_CLASS;
        box.setAttribute("data-samextra-cc-a11y", tab);
        const mod = slot(doc, box, "span", modId, { hidden: "" });
        const mask = slot(doc, box, "span", maskId, { hidden: "" });
        const status = slot(doc, box, "div", `${pre}a11y_status`,
            { role: "status", "aria-live": "polite", "aria-atomic": "true" });
        mod.data = editing([modNode(p)]);
        mask.data = editing([maskNode(p)]);
        host.appendChild(box);                 // 이 뒤로는 넣고 빼지 않는다
        p.a11y = { box, mod, mask, status, modId, maskId };
        decorate(doc, tab, p.a11y);
        return p.a11y;
    }

    function refreshA11y(doc, tab, announce) {
        const nodes = mount(doc, tab);
        if (!nodes) return;
        decorate(doc, tab, nodes);
        const p = page(tab);
        setText(nodes.mod, editing([modNode(p)]));
        setText(nodes.mask, editing([maskNode(p)]));
        if (announce) say(nodes.status, announce);
    }

    // 페이지가 뜰 때(Forge 의 onUiLoaded) 두 탭에 마운트한다. 못 찾은 탭은 화면이 바뀐 뒤(onAfterUiUpdate) 다시 찾는다.
    let mounted = false;

    function mountAll() {
        const doc = root.document;
        if (!doc || !S()) return false;
        let done = true;
        for (const tab of TABS) if (!mount(doc, tab)) done = false;
        return done;
    }

    function tryMountAll() {
        try {
            mounted = mountAll();
        } catch (e) {
            logError("a11y", e);
        }
    }

    function start() {
        tryMountAll();
        if (mounted || typeof root.onAfterUiUpdate !== "function") return;
        root.onAfterUiUpdate(() => {
            if (!mounted) tryMountAll();
        });
    }

    if (typeof root.onUiLoaded === "function") root.onUiLoaded(start);

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
        const comboRefs = !sameMask && combo >= 0 ? s.comboRefChoices[combo] : null;
        const kindAt = s.modifierFields.findIndex((f) => f.name === "kind");
        if (sameMod && sameMask) {                            // 이미 맞다(자기 쓰기의 메아리, 붙여넣기 직후)
            display(() => {
                const pasted = !wrote(tab, ref.rev);          // 이 페이지가 쓰지 않은 rev: Forge 의 붙여넣기
                note(tab, modT, editors[kindAt], maskT, ref.rev);
                afterRender(tab, { proxies: true, comboRefs, announce: pasted ? announcement(page(tab), "loaded") : null });
            });
            return undefined;
        }
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
        display(() => {
            note(tab, modT, sameMod ? editors[kindAt] : valueOf(map, modT, field(modT, "kind")), maskT, rev);
            const changed = [...(sameMod ? [] : ["mod"]), ...(sameMask ? [] : ["mask"])];
            afterRender(tab, { proxies: true, comboRefs,
                announce: announcement(page(tab), fresh ? "switched" : "loaded", changed) });
        });
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
        display(() => {
            note(tab, ref.mod, valueOf(map, ref.mod, field(ref.mod, "kind")), ref.mask, rev);
            afterRender(tab, { proxies: true, comboRefs: null, announce: announcement(page(tab), "reset", section) });
        });
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

    // Type 이 바뀌면 그 노드의 칸만 보인다(spec.KIND_GROUPS · KIND_GATES · KIND_TAKES_MASK). 편집기 칸의 "편집 중: 수정자 …
    // (Type)" 설명도 새 Type 으로(알림은 내지 않는다).
    // inputs [kind]   outputs [그룹 7, Apply Chroma Plus, Apply Color Shift, Mask ➜]
    function kind(tab, a) {
        const s = S();
        const id = kindId(a[0]);
        const shown = new Set(s.kindGroups[id]);
        const gates = new Set(s.kindGates[id] || []);
        if (TABS.includes(tab) && typeof a[0] === "string" && has(s.kindIds, a[0])) {
            display(() => {
                page(tab).kind = a[0];
                afterRender(tab, { proxies: false, announce: null });
            });
        }
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
