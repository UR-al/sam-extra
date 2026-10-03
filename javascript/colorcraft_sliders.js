// Colorcraft (sam-extra) — 슬라이더 숫자 칸의 범위 제한을 풉니다(원본 Forge 확장의 QoL). 드래그는 슬라이더 범위
// 안이지만, 숫자 칸에 직접 넣은 값은 범위 밖이어도 그대로 보냅니다(서버 쪽 sam3ext/colorcraft/spec.py 도 자르지 않음).
// 숫자가 아닌 글자는 0 으로 바꿉니다. Colorcraft 아코디언(txt2img·img2img) 안의 슬라이더에만 적용합니다.
//
// origin: muerrilla/ComfyUI-Colorcraft@d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf:javascript/colorcraft-sliders.js:1-69
// MIT License, Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Changes by sam-extra (2026-10-03): the scope selector is this extension's accordion
// (script title "Colorcraft (sam-extra)" → Forge elem_id script_<tab>_colorcraft_samextra_accordion), the
// commented-out select-on-click block is left out, and the comments are in Korean. The min/max stripping no
// longer rescans: upstream re-queried every slider number field of both panels (~800 here) on every DOM change
// anywhere in the page (progress text and live previews change it constantly during generation). Here each
// accordion is scanned once, then only what changes inside it is touched — added nodes, and a min/max that gets
// set again (upstream caught that only on the next page-wide rescan). Nothing outside the two accordions is
// observed once they exist. The blur/Enter handling (the unclamped number fields) is upstream's code unchanged.
(function () {
    const SCOPE_SEL = '#script_txt2img_colorcraft_samextra_accordion, #script_img2img_colorcraft_samextra_accordion';
    const INPUT_SEL = '.gradio-slider input[type=number]';

    // 완전한 float 표기만 받는다: 부호, 정수/소수(.5 포함), 지수. "+", "-", "e", "1e", "" 등은 거른다.
    const FLOAT_RE = /^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i;

    function inScope(el) {
        return el.matches && el.matches(INPUT_SEL) && el.closest(SCOPE_SEL) !== null;
    }

    function sanitize(el) {
        const v = el.value.trim();
        if (!FLOAT_RE.test(v)) {
            el.value = '0';
            // Gradio(Svelte) 는 .value 대입이 아니라 input 이벤트로 값을 읽는다.
            el.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }

    function intercept(e) {
        const el = e.target;
        if (inScope(el)) {
            e.stopImmediatePropagation();
            sanitize(el);
        }
    }

    document.addEventListener('blur', intercept, true);
    document.addEventListener('keydown', function (e) {
        if (e.key !== 'Enter') return;
        const el = e.target;
        if (!inScope(el)) return;

        sanitize(el); // 수식 키와 상관없이 잘못된 값부터 고친다

        if (!e.ctrlKey && !e.metaKey) {
            e.stopImmediatePropagation(); // 그냥 Enter: Gradio 의 범위 자르기를 막는다
        }
        // Ctrl/Cmd+Enter: 고친 뒤 생성 단축키로 그대로 넘긴다.
    }, true);

    // 범위 풀기는 Colorcraft 아코디언 안에서, 바뀐 부분에만 한다(원본은 페이지 어디서든 DOM 이 바뀌면 두 패널의
    // 숫자 칸을 전부 다시 훑었다 — 생성 중에는 진행 표시·미리보기 때문에 끊임없이).
    const SCOPE_IDS = SCOPE_SEL.split(',').map((sel) => sel.trim().slice(1));
    const PANEL_CHANGES = { childList: true, subtree: true, attributes: true, attributeFilter: ['min', 'max'] };

    function unclamp(el) {
        el.removeAttribute('min');
        el.removeAttribute('max');
    }

    // node 자신과 그 아래의 슬라이더 숫자 칸만.
    function unclampWithin(node) {
        if (node.nodeType !== 1) return;
        if (node.matches(INPUT_SEL)) unclamp(node);
        node.querySelectorAll(INPUT_SEL).forEach(unclamp);
    }

    function onPanelChanges(records) {
        for (const record of records) {
            if (record.type === 'childList') {
                record.addedNodes.forEach(unclampWithin);        // 새로 붙은 부분만
            } else if (record.target.hasAttribute(record.attributeName) && record.target.matches(INPUT_SEL)) {
                record.target.removeAttribute(record.attributeName);   // min/max 를 다시 단 경우
            }
        }
    }

    const panelObserver = new MutationObserver(onPanelChanges);
    const watched = new Set();

    // 아코디언마다 처음 한 번 전체를 풀고 그 안만 지켜본다. 아직 없는 아코디언이 있으면 true.
    function watchPanels() {
        let missing = false;
        for (const id of SCOPE_IDS) {
            const scope = document.getElementById(id);
            if (!scope) {
                missing = true;
            } else if (!watched.has(scope)) {
                watched.add(scope);
                unclampWithin(scope);
                panelObserver.observe(scope, PANEL_CHANGES);
            }
        }
        return missing;
    }

    onUiLoaded(function () {
        if (!watchPanels()) return;     // 보통은 여기서 끝: Gradio 는 처음에 모든 컴포넌트를 그린다
        // 아코디언이 늦게 생기는 경우에만, 나타날 때까지 id 두 개를 찾아본다(숫자 칸은 훑지 않는다). 다 찾으면 멈춘다.
        const finder = new MutationObserver(function () {
            if (!watchPanels()) finder.disconnect();
        });
        finder.observe(gradioApp(), { childList: true, subtree: true });
    });
})();
