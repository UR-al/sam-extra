"""TIPO 프롬프트 규칙 — 사용자 프롬프트 → TIPO 입력, TIPO 출력 → Anima 프롬프트.

입력 형식은 KohakUwULLM 의 공식 TIPO 학습 렌더러와 모델 카드(TIPO-v2.1)를 따른다: 메타 줄(quality/rating/artist/
characters/copyrights/aspect ratio), ``target:`` 줄, ``tag:`` 줄. 태그를 이어 쓰게 하려면 태그 줄을 줄 중간에서 끝낸다.

사용자가 적은 텍스트는 한 글자도 바꾸지 않는다 — 새 태그와 설명만 뒤에 붙인다. Anima 규칙: 작가는 ``@``, 괄호는 A1111
강조 문법으로 읽히므로 ``\\(`` 로 이스케이프, 밑줄 대신 공백.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

MODE_TAGS = "태그만"
MODE_TAGS_NL = "태그+설명"
MODE_NL = "설명만"
MODES = (MODE_TAGS, MODE_TAGS_NL, MODE_NL)

LENGTH_SHORT = "짧게"
LENGTH_NORMAL = "보통"
LENGTH_LONG = "길게"
LENGTHS = (LENGTH_SHORT, LENGTH_NORMAL, LENGTH_LONG)
_LENGTH_TOKENS = {LENGTH_SHORT: "short", LENGTH_NORMAL: "long", LENGTH_LONG: "very_long"}

# TIPO-v2.1 품질 어휘(나쁜 → 좋은). Anima 도 같은 단어를 쓴다(단 great quality 는 Anima 에 없다).
TIPO_QUALITY = (
    "worst quality", "low quality", "normal quality", "good quality", "great quality", "best quality", "masterpiece",
)
_QUALITY_WORDS = set(TIPO_QUALITY) | {
    "high quality", "amazing quality", "very aesthetic", "aesthetic", "displeasing", "very displeasing",
}
_ERAS = ("newest", "recent", "mid", "early", "old")
# v2.1 모델 카드의 변환표(danbooru → TIPO). 여러 개 적혀 있으면 가장 센 것을 쓴다.
_RATINGS = {"safe": "safe", "general": "safe", "sensitive": "sensitive", "nsfw": "nsfw",
            "questionable": "nsfw", "explicit": "nsfw, explicit"}
_RATING_ORDER = ("safe", "sensitive", "nsfw", "nsfw, explicit")
_SCORE_RE = re.compile(r"^score_\d+(?:_up)?$")
_YEAR_RE = re.compile(r"^year (\d{4})$")
_PERSON_RE = re.compile(r"^(?:\d+\+?(?:girl|boy|other)s?|multiple (?:girls|boys|others)|no humans)$")
# 강조 괄호 한 겹 — (x) [x] (x:1.2) (x: 1.2 ). 가중치 모양은 Forge 의 prompt_parser 를 따른다.
_WEIGHT_RE = re.compile(r"^[\(\[]\s*(.+?)\s*(?::\s*[+-]?[.\d]+\s*)?[\)\]]$")
_SENTENCE_END_RE = re.compile(r"(?<!\\)[.!?][\"'”’]*$")
_SENTENCE_BREAK_RE = re.compile(r"(?<!\\)[.!?]\s+\S")
_CATEGORY_NAMES = {1: "artist", 3: "copyright", 4: "character", 5: "meta"}


def _balanced(text: str) -> bool:
    depth = 0
    for char in re.sub(r"\\.", "", text):   # 이스케이프한 괄호는 세지 않는다
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _unwrap(text: str) -> str:
    """강조 괄호와 가중치를 겹겹이 벗긴다 — ((x)), [x], (x: 1.2) → x. ``\\(x\\)`` 같은 이스케이프 괄호는 그대로."""
    text = text.strip()
    while True:
        match = _WEIGHT_RE.match(text)
        if not match or text.endswith(("\\)", "\\]")) or not _balanced(match.group(1)):
            return text
        text = match.group(1).strip()


def normalize(tag: str) -> str:
    """비교용 키 — 가중치·강조 괄호와 ``@`` 를 벗기고 이스케이프를 풀고 밑줄을 공백으로(4글자 이상만), 소문자."""
    text = tag.strip()
    if text.startswith("@"):
        text = text[1:]
    text = _unwrap(text)
    if text.startswith("@"):
        text = text[1:]
    text = text.replace("\\(", "(").replace("\\)", ")")
    if len(text) >= 4:   # ^_^ 같은 짧은 이모티콘 태그는 밑줄이 태그의 일부다(_format_tag 와 같은 규칙)
        text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


class TagIndex:
    """태그 → danbooru 카테고리 (tagcomplete CSV: 0 general, 1 artist, 3 copyright, 4 character, 5 meta)."""

    def __init__(self, mapping: dict[str, int]):
        self._mapping = {normalize(key): value for key, value in mapping.items()}

    def category(self, tag: str) -> str:
        return _CATEGORY_NAMES.get(self._mapping.get(normalize(tag), 0), "general")

    def knows(self, tag: str) -> bool:
        return normalize(tag) in self._mapping


def default_tag_csv_paths(extensions_dir: Path) -> list[Path]:
    tags = Path(extensions_dir) / "sd-webui-tagcomplete-neo" / "tags"
    return [tags / "danbooru_2025.csv", tags / "danbooru.csv"]


def load_tag_index(paths) -> TagIndex:
    """첫 번째로 있는 CSV(``tag,category,count,aliases``)를 읽는다. 없으면 빈 인덱스(작가는 ``@`` 로만 알아본다)."""
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        mapping: dict[str, int] = {}
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 2 or not row[1].strip().isdigit():
                    continue
                category = int(row[1])
                mapping.setdefault(normalize(row[0]), category)
                if len(row) > 3 and row[3]:
                    for alias in row[3].split(","):
                        if alias.strip():
                            mapping.setdefault(normalize(alias), category)
        return TagIndex(mapping)
    return TagIndex({})


@dataclass
class PromptParts:
    quality: str | None = None
    era: str | None = None
    rating: str | None = None
    artists: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    copyrights: list[str] = field(default_factory=list)
    meta: list[str] = field(default_factory=list)   # highres 등 — TIPO 는 태그 줄이 아니라 meta 줄로 배웠다
    tags: list[str] = field(default_factory=list)   # 인원수 태그 먼저
    known: set[str] = field(default_factory=set)
    has_person: bool = False


def _era_of_year(year: int) -> str:
    if year >= 2024:
        return "newest"
    if year >= 2020:
        return "recent"
    if year >= 2018:
        return "mid"
    if year >= 2015:
        return "early"
    return "old"


def _is_protected(item: str) -> bool:
    """TIPO 에 보내지 않는 항목 — 로라·와일드카드·{a|b}·괄호가 쉼표 너머로 이어지는 묶음."""
    if any(mark in item for mark in ("<", ">", "__", "{", "}", "|")):
        return True
    if item.count("(") - item.count("\\(") != item.count(")") - item.count("\\)"):
        return True
    return item.count("[") != item.count("]")


def _looks_like_clause(piece: str, index: TagIndex) -> bool:
    if not piece or index.knows(piece):
        return False
    words = piece.split()
    return len(words) >= 3 or (len(words) >= 2 and piece[0].isupper())


def _prose_flags(pieces: list[str], index: TagIndex) -> list[bool]:
    """한 줄의 쉼표 조각 중 문장에 속한 것 — 쉼표가 들어간 문장("A girl, standing in the rain, looks up.")도 통째로.

    5단어 이상이거나 조각 안에서 문장이 끝나면 문장. 조각이 문장 부호로 끝나면, 그 앞의 절처럼 보이는 조각(3단어 이상이거나
    대문자로 시작하는 2단어 이상, 태그 사전에 없는 것)까지 거슬러 올라가 같은 문장으로 묶는다. 그리고 문장이 시작된 조각부터
    그 문장이 끝나는 조각까지는 사이의 조각("singing")도 문장이다.
    """
    flags = [False] * len(pieces)
    ends = [i for i, piece in enumerate(pieces) if piece and _SENTENCE_END_RE.search(piece)]
    for i, piece in enumerate(pieces):
        if piece and (len(piece.split()) >= 5 or _SENTENCE_BREAK_RE.search(piece)):
            flags[i] = True
    for end in ends:
        flags[end] = True
        j = end - 1
        while j >= 0 and not flags[j] and _looks_like_clause(pieces[j], index):
            flags[j] = True
            j -= 1
    start = 0
    for end in ends:
        first = next((i for i in range(start, end) if flags[i]), None)
        if first is not None:
            for i in range(first, end):
                flags[i] = True
        start = end + 1
    return flags


def analyze_prompt(text: str, index: TagIndex) -> PromptParts:
    parts = PromptParts()
    people: list[str] = []
    general: list[str] = []
    quality_rank = -1
    items: list[tuple[str, bool]] = []
    for line in re.sub(r"\bBREAK\b", ",", text or "").split("\n"):
        pieces = [piece.strip() for piece in line.split(",")]
        items.extend(zip(pieces, _prose_flags(pieces, index)))
    for item, prose in items:
        if not item or _is_protected(item):
            continue
        if prose:   # 문장은 TIPO 에 보내지 않는다 — 태그와 같은 조각이면 중복 판정에만 쓴다
            key = normalize(_SENTENCE_END_RE.sub("", item))
            if key:
                parts.known.add(key)
            continue
        key = normalize(item)
        if not key or key in parts.known:
            continue
        parts.known.add(key)
        if _unwrap(item).startswith("@"):
            parts.artists.append(key)
        elif _SCORE_RE.match(key.replace(" ", "_")):
            continue
        elif key in _QUALITY_WORDS:
            if key in TIPO_QUALITY and TIPO_QUALITY.index(key) > quality_rank:
                quality_rank = TIPO_QUALITY.index(key)
                parts.quality = key
        elif key in _ERAS:
            parts.era = parts.era or key
        elif (year := _YEAR_RE.match(key)) is not None:
            parts.era = parts.era or _era_of_year(int(year.group(1)))
        elif key in _RATINGS:
            rating = _RATINGS[key]
            if parts.rating is None or _RATING_ORDER.index(rating) > _RATING_ORDER.index(parts.rating):
                parts.rating = rating
        elif _PERSON_RE.match(key):
            people.append(key)
            parts.has_person = True
        else:
            category = index.category(key)
            if category == "artist":
                parts.artists.append(key)
            elif category == "character":
                parts.characters.append(key)
            elif category == "copyright":
                parts.copyrights.append(key)
            elif category == "meta":
                parts.meta.append(key)
            else:
                general.append(key)
    parts.tags = people + general
    return parts


def build_tipo_prompt(parts: PromptParts, mode: str, length: str, aspect_ratio: float | None) -> str:
    lines = []
    quality = ", ".join(word for word in (parts.quality, parts.era) if word)
    if quality:
        lines.append(f"quality: {quality}")
    if parts.rating:
        lines.append(f"rating: {parts.rating}")
    for key, values in (("artist", parts.artists), ("characters", parts.characters), ("copyrights", parts.copyrights),
                        ("meta", parts.meta)):
        if values:
            lines.append(f"{key}: {', '.join(values)}")
    if aspect_ratio:
        lines.append(f"aspect ratio: {aspect_ratio:.1f}")
    token = _LENGTH_TOKENS.get(length, "long")
    lines.append(f"target: <|{token}|>" if mode == MODE_TAGS else f"target: <|{token}|> <|tag_to_long|>")
    tag_line = "tag:" + (f" {', '.join(parts.tags)}" if parts.tags else "")
    # 태그를 이어 쓰게 하려면 줄 중간에서 끝낸다. 설명만이면 줄을 닫아 바로 설명으로 넘어가게 한다.
    return "\n".join(lines + [tag_line]) + ("\n" if mode == MODE_NL else "")


def _split_tags(text: str) -> list[str]:
    return [tag.strip() for tag in text.split(",") if tag.strip()]


def _complete_sentences(text: str) -> str:
    ends = [match.end() for match in re.finditer(r"[.!?][\"'”’)]*(?=\s|$)", text)]
    return text[:ends[-1]].strip() if ends else ""


def parse_tipo_output(mode: str, text: str, finished: bool = True) -> tuple[list[str], str]:
    """생성된 부분만 받아 (새 태그, 설명) — ``long`` 이 없으면 ``short``.

    ``finished`` 가 False 면(EOS 전에 토큰 한도로 멈춤) 마지막 줄이 중간에 잘린 것이다: 잘린 태그는 버리고, 설명은 마지막으로
    끝난 문장까지만 쓴다.
    """
    text = text or ""
    cut = not finished and not text.endswith("\n")
    new_tags: list[str] = []
    if mode != MODE_NL:
        first, newline, text = text.partition("\n")
        tags = _split_tags(first)
        new_tags.extend(tags[:-1] if cut and not newline else tags)
    fields: dict[str, str] = {}
    current = None
    for line in text.split("\n"):
        match = re.match(r"^\s*([a-z][a-z ]*):\s*(.*)$", line)
        if match:
            current = match.group(1).strip()
            fields[current] = match.group(2).strip()
        elif current and line.strip():
            fields[current] = f"{fields[current]} {line.strip()}".strip()
    if cut and current in ("long", "short"):
        fields[current] = _complete_sentences(fields[current])
    elif cut and current == "tag":
        fields["tag"] = ", ".join(_split_tags(fields["tag"])[:-1])
    if "tag" in fields:
        new_tags.extend(_split_tags(fields["tag"]))
    return new_tags, fields.get("long") or fields.get("short") or ""


def _escape(text: str) -> str:
    """A1111 강조 문법으로 읽히지 않게 괄호·대괄호를 이스케이프(이미 된 것은 그대로)."""
    return re.sub(r"(?<!\\)([\(\)\[\]])", r"\\\1", text)


def _format_tag(tag: str) -> str:
    text = tag.strip()
    if len(text) >= 4:   # ^_^ 같은 짧은 이모티콘 태그는 밑줄이 태그의 일부다
        text = text.replace("_", " ")
    return _escape(text)


def assemble_prompt(user_text, parts, new_tags, description, mode, allow_new_names, index):
    """(새 프롬프트, 추가한 태그) — 사용자 텍스트 뒤에 새 태그(캐릭터→작품→@작가→일반)와 설명(새 줄)을 붙인다."""
    buckets: dict[str, list[str]] = {"character": [], "copyright": [], "artist": [], "general": []}
    seen = set(parts.known)
    if mode != MODE_NL:
        for raw in new_tags:
            key = normalize(raw)
            if not key or key in seen:
                continue
            seen.add(key)
            if (key in _QUALITY_WORDS or key in _ERAS or key in _RATINGS or _YEAR_RE.match(key)
                    or _SCORE_RE.match(key.replace(" ", "_"))):
                continue   # 품질·등급·시대는 사용자가 적은 것만 둔다
            if _PERSON_RE.match(key) and parts.has_person:
                continue   # 사용자가 정한 인원수와 부딪히지 않게
            category = index.category(key)
            if category == "meta":
                continue   # highres·commentary 같은 메타 태그는 그림이 아니라 파일을 설명한다
            if category in ("character", "copyright", "artist"):
                if not allow_new_names:
                    continue
                tag = _format_tag(raw)
                buckets[category].append(f"@{tag}" if category == "artist" else tag)
            else:
                buckets["general"].append(_format_tag(raw))
    added = buckets["character"] + buckets["copyright"] + buckets["artist"] + buckets["general"]
    text = (user_text or "").rstrip()
    while text.endswith(","):
        text = text[:-1].rstrip()
    if added:
        text = f"{text}, {', '.join(added)}" if text else ", ".join(added)
    caption = _escape(description.strip()) if mode != MODE_TAGS and description else ""
    if caption:
        text = f"{text}\n{caption}" if text else caption
    return text, added
