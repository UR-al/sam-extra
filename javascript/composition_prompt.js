/* 구도 · 카메라 — 프롬프트 태그 계산 (패널은 javascript/composition_ui.js).
 *
 * 사용자 앱 UR_IV(AI Studio Pro)의 frontend/src/utils/compositionPrompt.ts 를 옮긴 것이다
 * (앱 커밋 d2fcc70798e61da137e87a8ff600d6fa73250fe6, git blob d083758ad318339bdda996e038f6f924b1e949eb).
 * 식·문턱값·문구·정규식은 원본 그대로이고, 바꾼 것은 문법뿐이다:
 * - 타입 표기를 지우고 export 대신 전역 sam3CompositionPrompt 하나로 내놓는다(Forge 는 javascript/*.js 를 모듈이 아닌
 *   <script> 로 넣는다).
 * - 원본에서 모듈 안에만 있던 tagKey·CONTRAST_GROUPS 도 내놓는다 — 테스트가 원본과 직접 대조한다.
 * - 상수는 Object.freeze 로 얼린다(전역이라 다른 스크립트가 바꾸지 못하게). 값과 모양은 같다.
 * 원본 사본과의 대조: tests/_origin_composition_prompt.ts(SHA-256 고정) · tests/js/composition_prompt_origin.test.mjs.
 * 3D 카메라가 아니라 프롬프트 유도다 — 결과 태그는 모델이 읽는 영어 그대로, 화면 글자는 앱과 같은 한국어.
 */
(function (root) {
    "use strict";

    /** Independent prompt steering, not a geometric camera or conditioning model. */
    const DEFAULT_COMPOSITION = Object.freeze({
        azimuth: 0, elevation: 0, distance: 45, roll: 0, framing: 0,
    });

    const COMPOSITION_CONTROLS = Object.freeze([
        Object.freeze({ key: "azimuth", label: "방향", min: -180, max: 180, unit: "°" }),
        Object.freeze({ key: "elevation", label: "높이", min: -75, max: 75, unit: "°" }),
        Object.freeze({ key: "distance", label: "거리 · 크롭", min: 0, max: 100, unit: "%" }),
        Object.freeze({ key: "roll", label: "기울기", min: -30, max: 30, unit: "°" }),
        Object.freeze({ key: "framing", label: "화면 내 인물 위치", min: -100, max: 100, unit: "%" }),
    ]);

    const COMPOSITION_PRESETS = Object.freeze([
        Object.freeze({ name: "정면 상반신", state: Object.freeze({ ...DEFAULT_COMPOSITION }) }),
        Object.freeze({ name: "낮은 시점 · 전신", state: Object.freeze({ azimuth: -35, elevation: -30, distance: 75, roll: 0, framing: 0 }) }),
        Object.freeze({ name: "위에서 · 근접", state: Object.freeze({ azimuth: 30, elevation: 40, distance: 15, roll: 0, framing: 0 }) }),
        Object.freeze({ name: "측면 · 여백", state: Object.freeze({ azimuth: 90, elevation: 0, distance: 55, roll: 0, framing: -55 }) }),
        Object.freeze({ name: "후면 · 원경", state: Object.freeze({ azimuth: 180, elevation: 15, distance: 100, roll: 0, framing: 40 }) }),
    ]);

    function normalizeComposition(value) {
        const source = value && typeof value === "object" ? value : {};
        const result = { ...DEFAULT_COMPOSITION };
        for (const control of COMPOSITION_CONTROLS) {
            const raw = source[control.key];
            const number = typeof raw === "number" ? raw : Number.NaN;
            result[control.key] = Number.isFinite(number)
                ? Math.round(Math.max(control.min, Math.min(control.max, number)))
                : DEFAULT_COMPOSITION[control.key];
        }
        return result;
    }

    function compositionTags(value) {
        const state = normalizeComposition(value);
        const angle = Math.abs(state.azimuth);
        const side = state.azimuth < 0 ? "left" : "right";
        const tags = angle <= 20 ? ["facing viewer"]
            : angle < 65 ? [`three-quarter view from the subject's ${side}`]
            : angle <= 115 ? ["from side", "profile", `view from the subject's ${side}`]
            : angle < 160 ? ["from behind", `rear three-quarter view from the subject's ${side}`]
            : ["from behind"];
        if (state.elevation >= 20) tags.push("from above");
        else if (state.elevation <= -20) tags.push("from below");
        if (state.elevation >= 60) tags.push("bird's eye view");
        tags.push(state.distance <= 15 ? "close-up" : state.distance <= 30 ? "portrait"
            : state.distance <= 50 ? "upper body" : state.distance <= 65 ? "cowboy shot"
            : state.distance <= 85 ? "full body" : "wide shot");
        if (Math.abs(state.roll) >= 6) {
            tags.push("dutch angle", `frame tilted ${state.roll < 0 ? "counterclockwise" : "clockwise"}`);
        }
        tags.push(Math.abs(state.framing) < 25 ? "centered composition"
            : `subject on the ${state.framing < 0 ? "left" : "right"} side of the frame`);
        return tags;
    }

    /** Split only top-level tags; preserve weighted groups, schedules and wildcards. */
    function splitCompositionPrompt(text) {
        const result = [];
        const closing = [];
        const pairs = { "(": ")", "[": "]", "{": "}", "<": ">" };
        let start = 0;
        for (let index = 0; index < text.length; index++) {
            const char = text[index];
            if (char === "\\") { index++; continue; }
            if (pairs[char]) closing.push(pairs[char]);
            else if (char === closing[closing.length - 1]) closing.pop();
            else if (char === "," && !closing.length) {
                result.push(text.slice(start, index).trim());
                start = index + 1;
            }
        }
        result.push(text.slice(start).trim());
        return result.filter(Boolean);
    }

    function tagKey(text) {
        let value = text.trim().toLowerCase().replace(/_/g, " ");
        // Only unwrap a complete attention expression, never an escaped literal.
        while (value.startsWith("(") && value.endsWith(")") && splitCompositionPrompt(value.slice(1, -1)).length === 1) {
            value = value.slice(1, -1).replace(/:\s*[+-]?(?:\d+\.?\d*|\.\d+)\s*$/, "").trim();
        }
        return value.replace(/\s+/g, " ");
    }

    const CONTRAST_GROUPS = Object.freeze([
        Object.freeze(["from above", "from below"]), Object.freeze(["facing viewer", "from behind"]),
        Object.freeze(["close-up", "portrait", "upper body", "cowboy shot", "full body", "wide shot"]),
        Object.freeze(["centered composition", "subject on the left side of the frame", "subject on the right side of the frame"]),
    ]);

    function planCompositionAppend(current, tags, otherSections = "") {
        const keys = new Set(splitCompositionPrompt(`${current}, ${otherSections}`).map(tagKey));
        const generated = new Set(tags.map(tagKey));
        const additions = tags.filter((tag) => {
            const key = tagKey(tag);
            if (!key || keys.has(key)) return false;
            keys.add(key);
            return true;
        });
        const conflicts = [];
        const original = new Set(splitCompositionPrompt(`${current}, ${otherSections}`).map(tagKey));
        for (const group of CONTRAST_GROUPS) {
            const incoming = group.filter((tag) => generated.has(tag));
            const previous = group.filter((tag) => original.has(tag) && !generated.has(tag));
            if (incoming.length && previous.length) conflicts.push(`${previous.join(" / ")} ↔ ${incoming.join(" / ")}`);
        }
        const separator = !current.trim() ? "" : /,\s*$/.test(current) ? (/\s$/.test(current) ? "" : " ") : ", ";
        return {
            additions, conflicts,
            // No reserialization: preserve the user's original prompt, syntax and spacing.
            text: additions.length ? current + separator + additions.join(", ") : current,
        };
    }

    /** Isometric diagram; coordinates remain inside the viewport at every setting. */
    function compositionCameraPoint(value) {
        const state = normalizeComposition(value);
        const angle = state.azimuth * Math.PI / 180;
        const elevation = state.elevation * Math.PI / 180;
        const radius = 52 + state.distance * 0.28;
        return {
            x: 150 + Math.sin(angle) * Math.cos(elevation) * radius,
            y: 88 + Math.cos(angle) * Math.cos(elevation) * radius * 0.33 - Math.sin(elevation) * 50,
            behind: Math.cos(angle) < 0,
        };
    }

    function dragComposition(start, deltaX, deltaY) {
        return normalizeComposition({ ...start, azimuth: start.azimuth + deltaX * 0.7, elevation: start.elevation - deltaY * 0.6 });
    }

    root.sam3CompositionPrompt = Object.freeze({
        DEFAULT_COMPOSITION,
        COMPOSITION_CONTROLS,
        COMPOSITION_PRESETS,
        CONTRAST_GROUPS,
        normalizeComposition,
        compositionTags,
        splitCompositionPrompt,
        tagKey,
        planCompositionAppend,
        compositionCameraPoint,
        dragComposition,
    });
})(globalThis);
