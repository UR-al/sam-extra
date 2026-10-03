// Colorcraft 공유 편집기 테스트 다리 — 실제 javascript/colorcraft_schema.js · colorcraft_editor.js 를 DOM 없는 vm 에
// 올리고, Gradio 4.40 이 js= 의존성을 부르는 방식(process_frontend_fn: inputs 뒤에 outputs 의 현재 값, undefined → [],
// 배열이 아닌 하나짜리 결과는 감싼다)대로 부른다. tests/test_colorcraft_editor_parity.py 는 이 파일을 stdin 의 JSON 요청으로
// 돌리고, node --test 의 colorcraft_editor.test.mjs 는 loadEditor · callJs 를 가져다 쓴다. (.test.mjs 가 아니라 테스트로는
// 돌지 않는다.)
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import vm from "node:vm";
import { webcrypto } from "node:crypto";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const JS_DIR = path.resolve(HERE, "..", "..", "javascript");

// extra: window 에 더할 것. files: 올릴 javascript/ 파일(기본 둘 다 — 빼면 그 파일이 없는 페이지). globals: 컨텍스트 전역
// (페이지의 js 문자열은 window. 없이 console 을 부른다).
const FILES = ["colorcraft_schema.js", "colorcraft_editor.js"];

export function loadEditor(extra = {}, { files = FILES, globals = {} } = {}) {
    const window = Object.assign({ crypto: webcrypto }, extra);
    const context = vm.createContext(Object.assign({ window }, globals));
    for (const file of files) vm.runInContext(readFileSync(path.join(JS_DIR, file), "utf8"), context);
    return { cc: window.samextraColorcraft, schema: window.samextraColorcraftSchema, window, context };
}

export async function callJs(env, js, args, nOutputs) {
    const fn = vm.runInContext(`(${js})`, env.context);
    const result = await fn(...args);
    if (typeof result === "undefined") return [];
    // 이 realm 의 평범한 객체로(vm 안의 배열·객체는 프로토타입이 다르다)
    return JSON.parse(JSON.stringify(nOutputs === 1 && !Array.isArray(result) ? [result] : result));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const request = JSON.parse(readFileSync(0, "utf8"));
    const env = loadEditor();
    const { cc } = env;
    const out = [];
    for (const job of request.jobs) {
        if (job.op === "commit") {
            const st = cc._pure.parseState(job.state);
            const ref = cc._pure.parseRef(job.ref);
            cc._pure.commit(st.map, ref, job.editors);
            out.push(cc._pure.serialize(st.map, "x"));
        } else if (job.op === "overview") {
            const st = cc._pure.parseState(job.state);
            const ref = cc._pure.parseRef(job.ref);
            if (ref && st.rev === ref.rev) cc._pure.commit(st.map, ref, job.editors);
            out.push(cc._pure.overview(st.map, job.enabled, job.masking));
        } else if (job.op === "parse") {
            out.push(cc._pure.parseState(job.state));
        } else if (job.op === "call") {
            out.push(await callJs(env, job.js, job.args, job.nOutputs));
        } else if (job.op === "call_page") {
            // 다른 페이지: job.files 만 올리고 console.error 를 센다
            const errors = [];
            const console = { error: (...args) => errors.push(args.map(String).join(" ")) };
            const page = loadEditor({ console }, { files: job.files, globals: { console } });
            out.push({ out: await callJs(page, job.js, job.args, job.nOutputs), errors });
        }
    }
    process.stdout.write(JSON.stringify(out));
}
