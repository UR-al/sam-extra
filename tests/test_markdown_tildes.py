"""A single "~" in Markdown prose must be escaped ("\\~") — Gradio's Markdown (marked 12, gfm) and GitHub draw a pair of
single tildes as strikethrough: "σ̃ 0.002~80 … 0.006~240" showed "0.00280 … 0.006240" with the text between struck out
(v0.33.0 live check, Extra Schedulers / Extra Samplers help).

Checked here, without Gradio or Forge:

* every ``gr.Markdown(...)`` string of the extension (sam3ext/ and scripts/, read from the source with ``ast`` — module
  constants, f-string text, ``+`` / implicit concatenation, ``a if c else b``, ``"…".join([...])``);
* README.md and CHANGELOG.md;
* and the other way round: plain-text places (component ``label=`` / ``info=`` — Gradio 4.40 shows ``info`` as text —,
  ``gr.HTML``, Forge settings labels and ``.info()`` / ``.html()``, which Forge renders as HTML) carry no "\\~", which
  would show its backslash there.

Code spans (any backtick run length) and fenced code blocks are not prose: a "~" there is shown as is. The scanner treats
everything else as prose (indented code blocks included — none of the checked texts has one with a tilde).
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("sam3ext", "scripts")
DOCS = ("README.md", "CHANGELOG.md")
_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_FENCE_OPEN = re.compile(r"[ \t]*(`{3,}|~{3,})(.*)$")


# ---------------------------------------------------------------------------------------------------------------------
# Markdown: what is prose
# ---------------------------------------------------------------------------------------------------------------------

def _blank_fences(text: str) -> str:
    """Fenced code blocks (``` or ~~~, any indentation, to the closing fence or the end) -> spaces, line breaks kept."""
    out = []
    fence = None
    for line in text.split("\n"):
        if fence is None:
            match = _FENCE_OPEN.match(line)
            if match and not (match.group(1)[0] == "`" and "`" in match.group(2)):
                fence = match.group(1)
                out.append(" " * len(line))
                continue
            out.append(line)
            continue
        if re.match(r"[ \t]*" + re.escape(fence[0]) + "{%d,}[ \t]*$" % len(fence), line):
            fence = None
        out.append(" " * len(line))
    return "\n".join(out)


def _blank_code_spans(block: str) -> str:
    """CommonMark code spans in one block: a backtick run opens, the next run of the same length closes; an unclosed run
    is literal text. Backslash escapes outside spans make the next punctuation character literal."""
    out = list(block)
    i, n = 0, len(block)
    while i < n:
        char = block[i]
        if char == "\\" and i + 1 < n and block[i + 1] in _PUNCT:
            i += 2
            continue
        if char != "`":
            i += 1
            continue
        j = i
        while j < n and block[j] == "`":
            j += 1
        size, k, close = j - i, j, -1
        while k < n:
            if block[k] != "`":
                k += 1
                continue
            end = k
            while end < n and block[end] == "`":
                end += 1
            if end - k == size:
                close = end
                break
            k = end
        if close < 0:
            i = j
            continue
        for p in range(i, close):
            if out[p] != "\n":
                out[p] = " "
        i = close
    return "".join(out)


def prose(markdown: str) -> str:
    """The text with code blocks and code spans blanked out (same length and line breaks)."""
    text = _blank_fences(markdown.replace("\r\n", "\n"))
    parts = []
    last = 0
    for block in re.finditer(r"(?:[^\n]*\S[^\n]*(?:\n|$))+", text):    # blocks end at blank lines
        parts.append(text[last:block.start()])
        parts.append(_blank_code_spans(block.group(0)))
        last = block.end()
    parts.append(text[last:])
    return "".join(parts)


def unescaped_tildes(markdown: str) -> list[tuple[int, int, str]]:
    """(line, column, the line) of every "~" in prose that has no escaping backslash before it."""
    original = markdown.replace("\r\n", "\n")
    blank = prose(original)
    hits = []
    for match in re.finditer("~", blank):
        start = match.start()
        slashes = 0
        while start - slashes - 1 >= 0 and blank[start - slashes - 1] == "\\":
            slashes += 1
        if slashes % 2 == 0:
            line = blank.count("\n", 0, start) + 1
            column = start - (blank.rfind("\n", 0, start) + 1) + 1
            hits.append((line, column, original.split("\n")[line - 1]))
    return hits


# ---------------------------------------------------------------------------------------------------------------------
# The extension's source: Markdown strings and plain-text strings
# ---------------------------------------------------------------------------------------------------------------------

def _python_files():
    for top in SOURCE_DIRS:
        yield from sorted(p for p in (ROOT / top).rglob("*.py") if "__pycache__" not in p.parts)


class _Strings:
    """Possible string values of an expression, from the module's own source."""

    def __init__(self, tree: ast.AST):
        self.names: dict[str, list[ast.AST]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.names.setdefault(target.id, []).append(node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                self.names.setdefault(node.target.id, []).append(node.value)

    def values(self, node: ast.AST | None, depth: int = 0) -> list[str]:
        if node is None or depth > 8:
            return []
        if isinstance(node, ast.Constant):
            return [node.value] if isinstance(node.value, str) else []
        if isinstance(node, ast.JoinedStr):           # f-string: its literal text, "{…}" for each field
            return ["".join(v.value if isinstance(v, ast.Constant) else "{…}" for v in node.values)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.values(node.left, depth + 1) or ["{…}"]
            right = self.values(node.right, depth + 1) or ["{…}"]
            return [a + b for a in left for b in right]
        if isinstance(node, ast.IfExp):
            return self.values(node.body, depth + 1) + self.values(node.orelse, depth + 1)
        if isinstance(node, ast.Name):
            found = []
            for value in self.names.get(node.id, []):
                found += self.values(value, depth + 1)
            return found
        if isinstance(node, (ast.List, ast.Tuple)):
            return ["\n".join(x for element in node.elts for x in self.values(element, depth + 1))]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
            return [x for arg in node.args for x in self.values(arg, depth + 1)]
        return []


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _is_gr(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "gr"


def markdown_strings() -> list[tuple[str, int, str, str]]:
    """(file, line, source of the argument, value) for every gr.Markdown(...) value of the extension."""
    found = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        strings = _Strings(tree)
        rel = path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) == "Markdown":
                args = list(node.args[:1]) + [k.value for k in node.keywords if k.arg == "value"]
                for arg in args:
                    for value in strings.values(arg):
                        found.append((rel, node.lineno, ast.unparse(arg)[:60], value))
    return found


def plain_text_strings() -> list[tuple[str, int, str]]:
    """(file, line, value): component labels / info, gr.HTML, Forge settings labels and .info() / .html() — not Markdown."""
    found = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        strings = _Strings(tree)
        rel = path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            exprs = []
            if _is_gr(node) and name != "Markdown":
                exprs += [k.value for k in node.keywords if k.arg in ("label", "info", "placeholder")]
                if name == "HTML":
                    exprs += list(node.args[:1]) + [k.value for k in node.keywords if k.arg == "value"]
            if name == "OptionInfo":
                exprs += list(node.args[1:2]) + [k.value for k in node.keywords if k.arg == "label"]
            if name in ("info", "html") and isinstance(node.func, ast.Attribute) \
                    and isinstance(node.func.value, ast.Call) and _call_name(node.func.value) == "OptionInfo":
                exprs += list(node.args[:1])
            for expr in exprs:
                for value in strings.values(expr):
                    found.append((rel, node.lineno, value))
    return found


def _report(hits, limit=15):
    lines = [f"{where}:{line}:{column}: …{text[max(0, column - 30):column + 20]}…" for where, line, column, text in hits]
    more = f"\n… and {len(lines) - limit} more" if len(lines) > limit else ""
    return f"{len(hits)} unescaped '~' (write '\\~', or an en dash):\n" + "\n".join(lines[:limit]) + more


# ---------------------------------------------------------------------------------------------------------------------

class ScannerTests(unittest.TestCase):
    """The scanner itself, on the live finding's text and on code."""

    def test_the_live_findings_text_is_caught(self):
        old = ("기본값 ρ 7 · σ̃ 0.002~80 은 Anima 의 부모 모델(Cosmos-Predict2) 스케줄로 … σ̃ 범위 ×3(0.006~240)에 해당합니다\n"
               "- **잡음 구간**: 켜면 시작~끝(… σ 0.92~0.43)에서만 SDE 잡음을 넣고")
        hits = unescaped_tildes(old)
        self.assertEqual([line for line, _column, _text in hits], [1, 1, 2, 2])
        for _line, column, text in hits:
            self.assertEqual(text[column - 1], "~")
        (_l1, c1, t1), (_l2, c2, t2) = hits[:2]
        self.assertEqual((t1[c1 - 6:c1 + 2], t2[c2 - 6:c2 + 3]), ("0.002~80", "0.006~240"))
        self.assertEqual(unescaped_tildes(old.replace("~", "\\~")), [])
        self.assertEqual(len(unescaped_tildes("a~b c~d ~ e~~f")), 5)

    def test_code_is_not_prose(self):
        self.assertEqual(unescaped_tildes("`0.1~2` and ``a ` ~ b`` then ```x~y```"), [])
        self.assertEqual(unescaped_tildes("intro\n\n```text\n1~2 3~4\n```\n\n~~~\n5~6\n~~~\n"), [])
        self.assertEqual(unescaped_tildes("- item\n\n  ```\n  1~2\n  ```\n"), [])
        # a span may cross a line break inside one paragraph, never a blank line
        self.assertEqual(unescaped_tildes("`a\nb~c` d"), [])
        self.assertEqual(len(unescaped_tildes("`a\n\nb~c` d")), 1)

    def test_backticks_that_do_not_close_and_escaped_backslashes(self):
        self.assertEqual(len(unescaped_tildes("``a~b` c")), 1)        # run of 2 never closed: literal backticks
        self.assertEqual(len(unescaped_tildes("\\`a~b` c")), 1)       # escaped backtick opens nothing
        self.assertEqual(len(unescaped_tildes("a\\\\~b")), 1)         # "\\" is an escaped backslash: ~ is bare
        self.assertEqual(len(unescaped_tildes("a\\\\\\~b")), 0)
        self.assertEqual(unescaped_tildes("x\r\ny~z")[0][:2], (2, 2))


class GradioMarkdownTests(unittest.TestCase):
    def test_every_gradio_markdown_string_escapes_its_tildes(self):
        found = markdown_strings()
        hits = []
        for where, lineno, _arg, value in found:
            hits += [(where, lineno, column, text) for _line, column, text in unescaped_tildes(value)]
        if hits:
            self.fail(_report(hits))

    def test_the_collector_reaches_the_help_texts(self):
        found = markdown_strings()
        self.assertGreater(len(found), 60)
        seen = {(where, arg) for where, _line, arg, _value in found}
        for where, arg in (("sam3ext/ui_extra_schedulers.py", "HELP"), ("scripts/anima_extra_samplers.py", "_HELP"),
                           ("scripts/dora_infer_mode.py", None), ("scripts/anima_safe_pag.py", None),
                           ("sam3ext/ui_anima.py", None), ("sam3ext/colorcraft/ui.py", "INTRO")):
            with self.subTest(where=where):
                self.assertTrue(any(w == where and (arg is None or a == arg) for w, a in seen), (where, arg))
        values = {value for _w, _l, _a, value in found}
        # the ranges of the live finding are there, escaped (rendered "~")
        for text in ("σ̃ 0.002\\~80", "×3(0.006\\~240)", "시작\\~끝", "σ 0.92\\~0.43", "0.1\\~2 구간(flow σ 0.09\\~0.67)"):
            with self.subTest(text=text):
                self.assertTrue(any(text in value for value in values), text)

    def test_plain_text_places_carry_no_markdown_escape(self):
        found = plain_text_strings()
        self.assertGreater(len(found), 200)
        bad = [(where, line, value[:80]) for where, line, value in found if "\\~" in value]
        self.assertEqual(bad, [])
        # those places keep their plain "~" (shown as is): e.g. the Safe PAG / Anima 3.8B ranges
        self.assertTrue(any("0.05~0.10" in value for _w, _l, value in found))
        self.assertTrue(any("6~8 GB" in value for _w, _l, value in found))


class DocsTests(unittest.TestCase):
    def test_readme_and_changelog_escape_their_tildes(self):
        for name in DOCS:
            with self.subTest(doc=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                hits = [(name, line, column, src) for line, column, src in unescaped_tildes(text)]
                if hits:
                    self.fail(_report(hits))

    def test_the_escaped_ranges_are_still_there(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        for text, where, name in (("σ̃ 0.002\\~80", readme, "README.md"), ("σ̃ 0.002\\~80", changelog, "CHANGELOG.md"),
                                  ("σ̃ 0.9999\\~1 은 841 스텝부터", readme, "README.md")):
            with self.subTest(text=text, doc=name):
                self.assertTrue(text in where, (text, name))


if __name__ == "__main__":
    unittest.main()
