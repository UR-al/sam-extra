// Settings → Tag Autocomplete 맨 위 안내의 실제 렌더링 — headless Chrome(또는 Chromium · Edge)이 Forge 의 .settings-info 상자를
// 그리고, 안의 글이 상자 밖으로 나가지 않는지 잰다.
//
// v0.34.0 실제 확인: 고른 태그 파일이 없을 때 안내 끝의 옮기기 명령(tools/migrate_tagcomplete.py 의 긴 절대 경로)이 끊을 자리가 없어
// 상자 오른쪽 밖으로 넘쳤다. 안내는 sam3ext/tagcomplete/coexist.py note_html 이 NOTE_CLASS 묶음 하나로 만들고, style.css 가 그
// 안에서 넘칠 때만 아무 데서나 줄을 바꾼다. 여기서는 그 모양의 안내(같은 묶음 · <code> · 화면 캡처와 같은 길이의 경로)를 창 너비마다
// 그려 모든 글의 사각형이 상자 안쪽에 있는지 본다. Forge 의 style.css(#settings .settings-info: max-width 48em, 점선 테두리,
// 1em 안쪽 여백)는 확장 옆 Forge 체크아웃(<Forge>/extensions/<ext>, 또는 $SAM3_FORGE_ROOT)에서 읽고, 없으면(CI) 그 규칙만 옮겨 쓴다.
// 브라우저가 없으면 건너뛴다($SAM3_TEST_BROWSER 로 지정 가능).
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import os from "node:os";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const read = (relpath) => readFileSync(path.join(ROOT, relpath), "utf8");
const NOTE_CLASS = "sam3-tac-note";

function findBrowser() {
  const named = process.env.SAM3_TEST_BROWSER || process.env.CHROME_PATH;
  if (named) return existsSync(named) ? named : null;
  const env = (name, fallback) => process.env[name] || fallback;
  const candidates = process.platform === "win32" ? [
    path.join(env("PROGRAMFILES", "C:\\Program Files"), "Google", "Chrome", "Application", "chrome.exe"),
    path.join(env("PROGRAMFILES(X86)", "C:\\Program Files (x86)"), "Google", "Chrome", "Application", "chrome.exe"),
    path.join(env("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
    path.join(env("PROGRAMFILES(X86)", "C:\\Program Files (x86)"), "Microsoft", "Edge", "Application", "msedge.exe"),
    path.join(env("PROGRAMFILES", "C:\\Program Files"), "Microsoft", "Edge", "Application", "msedge.exe"),
  ] : process.platform === "darwin" ? [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
  ] : ["/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser"];
  return candidates.find((candidate) => candidate && existsSync(candidate)) || null;
}

function forgeStyle() {
  for (const root of [path.resolve(ROOT, "..", ".."), process.env.SAM3_FORGE_ROOT]) {
    if (!root) continue;
    const file = path.join(root, "style.css");
    if (existsSync(file) && existsSync(path.join(root, "modules"))) return readFileSync(file, "utf8");
  }
  return null;
}

const BROWSER = findBrowser();
const FORGE_STYLE = forgeStyle();
// origin: <Forge>/style.css #settings .settings-info (used when no Forge checkout)
const FORGE_STAND_IN = `
  #settings .settings-info { max-width: 48em; border: 1px dotted #777; margin: 0; padding: 1em; }`;
// origin: gradio 4.40 templates/frontend/assets/index-*.css .prose (without the container class)
const GRADIO_PROSE = `
  body { margin: 0; padding: 8px; font-family: sans-serif; font-size: 14px; }
  .prose { font-size: 14px; } .prose p { margin-bottom: 4px; line-height: 1.5; }`;

// The shape coexist.note_html builds for a missing tag file (the live check's case), with a path as long as the
// live check's and no break opportunity in its last 100 characters.
const TAGS = "C:\\Users\\someone\\AppData\\Local\\Temp\\claude\\C--Users-someone-Desktop-Image-viewer\\"
  + "117acd50-ba13-492c-be8c-37a0630ffe93\\scratchpad\\v034\\live\\forge_data\\extensions\\forge_sam3_extension";
const COMMAND = `C:\\sd-webui-forge-classic\\venv\\Scripts\\python.exe -I ${TAGS}\\tools\\migrate_tagcomplete.py `
  + "--src &quot;&lt;old sd-webui-tagcomplete-neo folder&gt;&quot; --apply";
const NOTE = `<div class="${NOTE_CLASS}"><b>sam-extra 내장 Tag Autocomplete</b> — sd-webui-tagcomplete-neo(@8e0d81f, MIT)를
  sam-extra 에 합친 것입니다. 아래 설정은 따로 설치하던 확장과 같은 키(<code>tac_*</code>)라 값이 그대로 이어집니다.
  태그 파일은 <code>${TAGS}\\tags</code> 에서 읽습니다.<br>⚠ 고른 파일이 태그 폴더에 없습니다: <code>tac_tagFile</code> =
  <code>KR_danbooru_tags_with_description-20260907.csv</code> — 아래에서 있는 파일을 다시 고르세요. 그 전까지
  <code>tac_tagFile</code> 는 기본 목록 <code>danbooru_2026-10-01_pt20-ia-dd.csv</code> 로 추천합니다(설정 값은 바꾸지 않음).
  예전 tagcomplete 폴더에 있던 목록이면 Forge 를 끄고 먼저 옮기세요: <code>${COMMAND}</code></div>`;

function page() {
  const styles = [GRADIO_PROSE, FORGE_STYLE ?? FORGE_STAND_IN, read("tokens.css"), read("style.css")];
  return `<!doctype html><html><head><meta charset="utf-8">
    ${styles.map((css) => `<style>${css}</style>`).join("\n")}</head><body>
    <div class="gradio-container"><div id="tab_settings"><div id="settings" class="tabs">
      <div id="settings_tac" class="tabitem"><div id="column_settings_tac" class="column">
        <div id="setting_sam3_tac_builtin_note" class="block settings-info gradio-html">
          <div class="prose gradio-html">${NOTE}</div></div>
      </div></div>
    </div></div></div>
    <script>
      var box = document.getElementById("setting_sam3_tac_builtin_note");
      var inner = box.getBoundingClientRect();
      var style = getComputedStyle(box);
      var right = inner.right - parseFloat(style.borderRightWidth) - parseFloat(style.paddingRight);
      var left = inner.left + parseFloat(style.borderLeftWidth) + parseFloat(style.paddingLeft);
      var out = { box: { left: inner.left, right: inner.right, width: inner.width }, contentLeft: left, contentRight: right,
                  scroll: [box.scrollWidth, box.clientWidth], codes: [] };
      document.querySelectorAll("#setting_sam3_tac_builtin_note code").forEach(function (code) {
        var rects = Array.prototype.map.call(code.getClientRects(), function (r) { return { left: r.left, right: r.right }; });
        out.codes.push({ text: code.textContent.slice(-40), rects: rects });
      });
      var node = document.createElement("script");
      node.type = "application/json";
      node.id = "sam3-layout";
      node.textContent = JSON.stringify(out);
      document.body.appendChild(node);
    </script></body></html>`;
}

function render(t, width) {
  const dir = mkdtempSync(path.join(os.tmpdir(), "sam3-tac-note-layout-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const file = path.join(dir, "index.html");
  writeFileSync(file, page(), "utf8");
  const args = ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--hide-scrollbars",
    `--user-data-dir=${path.join(dir, "profile")}`, `--window-size=${width},900`, "--virtual-time-budget=2000",
    "--dump-dom", pathToFileURL(file).href];
  if (process.platform === "linux") args.unshift("--no-sandbox");   // CI containers without user namespaces
  const html = execFileSync(BROWSER, args, { encoding: "utf8", timeout: 60000, stdio: ["ignore", "pipe", "pipe"] });
  const match = /<script type="application\/json" id="sam3-layout">([^<]*)<\/script>/.exec(html);
  assert.ok(match, "the page did not finish measuring");
  return JSON.parse(match[1]);
}

const SKIP = BROWSER ? false : "no Chrome / Chromium / Edge found (set $SAM3_TEST_BROWSER)";

for (const width of [1600, 1024, 600]) {
  test(`${width}px: the Tag Autocomplete note's paths and migration command stay inside its box`, { skip: SKIP }, (t) => {
    const layout = render(t, width);
    assert.ok(layout.codes.length >= 6, "the note's code pieces were measured");
    for (const code of layout.codes) {
      for (const rect of code.rects) {
        assert.ok(rect.right <= layout.contentRight + 1,
          `'…${code.text}' runs ${Math.round(rect.right - layout.contentRight)}px past the box's right edge`);
        assert.ok(rect.left >= layout.contentLeft - 1, `'…${code.text}' starts left of the box`);
      }
    }
    assert.ok(layout.scroll[0] <= layout.scroll[1] + 1, `the box overflows sideways (${layout.scroll})`);
  });
}

test("the note's wrapper is the class sam3ext/tagcomplete/coexist.py builds", () => {
  const source = read("sam3ext/tagcomplete/coexist.py");
  assert.equal(/^NOTE_CLASS = "(.*)"$/m.exec(source)?.[1], NOTE_CLASS);
  assert.match(source, /return f'<div class="\{NOTE_CLASS\}">\{text\}<\/div>'/);
});
