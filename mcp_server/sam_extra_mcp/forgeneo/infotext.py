# Vendored from https://github.com/eduardoabreu81/forgeneo-mcp at commit a103dc5
# (forgeneo_mcp/infotext.py) into sam-extra (mcp_server/sam_extra_mcp/forgeneo/infotext.py).
#
# Copyright (c) 2026 Eduardo Abreu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Changes in sam-extra (2026-10-03):
# - JPEG and WebP: the EXIF UserComment that Forge writes when enable_pnginfo is on
#   (modules/images.py save_image_with_geninfo, piexif "UNICODE" = UTF-16BE) is read too
#   (source "exif_user_comment"), before the .txt sidecar. Upstream listed this on its roadmap;
#   the generation matcher needs it to tell its own JPEG/WebP results from concurrent ones.
# - tEXt chunks are Latin-1 by the PNG spec (Pillow writes them that way): decoded as UTF-8 when
#   that is valid, else Latin-1, instead of UTF-8 with replacement characters. Compressed iTXt
#   and zTXt "parameters" chunks are read as well (size-capped decompression).
# - read_parameters_from_bytes(): the same readers for an image held in memory (the base64
#   images of an API response).
# - parse_infotext and the Infotext dataclass are unchanged.

"""Reading and parsing of A1111/Forge generation metadata ("infotext").

PNG parsing is done by hand rather than through Pillow: we only need the tEXt
chunk holding the "parameters" key, which sits near the start of the file. That
keeps the dependency list short and, more importantly, avoids reading whole
images over a network share when indexing thousands of outputs.

JPEG and WebP carry the same text in an EXIF UserComment, which is also read by
hand for the same reasons.
"""

from __future__ import annotations

import io
import os
import re
import struct
import zlib
from dataclasses import dataclass
from typing import BinaryIO

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PARAMETERS_KEY = b"parameters"
MAX_CHUNK_BYTES = 4 * 1024 * 1024
MAX_TEXT_BYTES = 4 * 1024 * 1024

JPEG_SOI = b"\xff\xd8"
EXIF_HEADER = b"Exif\x00\x00"
EXIF_IFD_POINTER = 0x8769
USER_COMMENT = 0x9286
MAX_EXIF_BYTES = 4 * 1024 * 1024
_TIFF_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}

LORA_PATTERN = re.compile(r"<lora:([^:>]+):([0-9]*\.?[0-9]+)", re.IGNORECASE)
# Key: value pairs on the trailing parameters line, tolerating quoted values.
PARAM_PATTERN = re.compile(r'\s*([\w \-/]+):\s*("(?:[^"]|\\")*"|[^,]*)(?:,|$)')


@dataclass(frozen=True)
class Infotext:
    prompt: str
    negative: str
    params: dict[str, str]
    loras: tuple[tuple[str, float], ...]

    def get_float(self, key: str) -> float | None:
        raw = self.params.get(key)
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    @property
    def checkpoint(self) -> str | None:
        return self.params.get("Model")

    @property
    def steps(self) -> float | None:
        return self.get_float("Steps")

    @property
    def cfg(self) -> float | None:
        return self.get_float("CFG scale")

    @property
    def sampler(self) -> str | None:
        return self.params.get("Sampler")

    @property
    def scheduler(self) -> str | None:
        return self.params.get("Schedule type")


def read_png_parameters(path: str) -> str | None:
    """Return the "parameters" text chunk of a PNG, or None if absent."""
    try:
        with open(path, "rb") as handle:
            return _png_parameters(handle)
    except OSError:
        return None


def _png_parameters(handle: BinaryIO) -> str | None:
    try:
        if handle.read(8) != PNG_SIGNATURE:
            return None
        while True:
            header = handle.read(8)
            if len(header) < 8:
                return None
            length, chunk_type = struct.unpack(">I4s", header)
            if chunk_type == b"IDAT" or chunk_type == b"IEND":
                return None  # pixel data reached; no text chunk present
            if length > MAX_CHUNK_BYTES:
                return None
            if chunk_type in (b"tEXt", b"iTXt", b"zTXt"):
                body = handle.read(length)
                text = _decode_text_chunk(chunk_type, body)
                if text is not None:
                    return text
            else:
                handle.seek(length, 1)
            handle.seek(4, 1)  # skip CRC
    except (OSError, struct.error):
        return None


def read_exif_parameters(path: str) -> str | None:
    """Return the EXIF UserComment of a JPEG or WebP file, or None if absent."""
    try:
        with open(path, "rb") as handle:
            return _exif_parameters(handle)
    except OSError:
        return None


def _exif_parameters(handle: BinaryIO) -> str | None:
    try:
        head = handle.read(12)
        handle.seek(0)
        if head[:2] == JPEG_SOI:
            tiff = _jpeg_exif(handle)
        elif head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            tiff = _webp_exif(handle)
        else:
            return None
    except (OSError, struct.error):
        return None
    if not tiff:
        return None
    return _user_comment(tiff)


def _jpeg_exif(handle: BinaryIO) -> bytes | None:
    handle.read(2)  # SOI
    while True:
        marker = handle.read(2)
        if len(marker) < 2 or marker[0] != 0xFF:
            return None
        kind = marker[1]
        if kind == 0xFF:  # fill byte; the marker type follows
            handle.seek(-1, 1)
            continue
        if kind in (0x01, 0xD8) or 0xD0 <= kind <= 0xD7:
            continue  # standalone markers carry no length
        if kind in (0xDA, 0xD9):  # start of scan / end of image: no metadata past here
            return None
        raw_length = handle.read(2)
        if len(raw_length) < 2:
            return None
        length = struct.unpack(">H", raw_length)[0]
        if length < 2:
            return None
        body = handle.read(length - 2)
        if kind == 0xE1 and body.startswith(EXIF_HEADER):
            return body[len(EXIF_HEADER):]


def _webp_exif(handle: BinaryIO) -> bytes | None:
    handle.seek(12)
    while True:
        header = handle.read(8)
        if len(header) < 8:
            return None
        fourcc, size = header[:4], struct.unpack("<I", header[4:])[0]
        if fourcc == b"EXIF":
            if size > MAX_EXIF_BYTES:
                return None
            body = handle.read(size)
            return body[len(EXIF_HEADER):] if body.startswith(EXIF_HEADER) else body
        handle.seek(size + (size & 1), 1)


def _user_comment(tiff: bytes) -> str | None:
    """UserComment from a TIFF-structured EXIF block, decoded per its 8-byte prefix."""
    if len(tiff) < 8:
        return None
    if tiff[:2] == b"II":
        order = "<"
    elif tiff[:2] == b"MM":
        order = ">"
    else:
        return None
    if struct.unpack(order + "H", tiff[2:4])[0] != 42:
        return None
    ifd0 = struct.unpack(order + "I", tiff[4:8])[0]
    pointer = _ifd_value(tiff, order, ifd0, EXIF_IFD_POINTER)
    if pointer is None:
        return None
    exif_ifd = struct.unpack(order + "I", pointer[:4])[0] if len(pointer) >= 4 else None
    if exif_ifd is None:
        return None
    raw = _ifd_value(tiff, order, exif_ifd, USER_COMMENT)
    if raw is None or len(raw) < 8:
        return None
    return _decode_user_comment(raw)


def _ifd_value(tiff: bytes, order: str, offset: int, wanted: int) -> bytes | None:
    """Raw value bytes of one tag in the IFD at `offset`, or None."""
    if offset + 2 > len(tiff):
        return None
    count = struct.unpack(order + "H", tiff[offset:offset + 2])[0]
    for index in range(count):
        entry = offset + 2 + index * 12
        if entry + 12 > len(tiff):
            return None
        tag, kind, items = struct.unpack(order + "HHI", tiff[entry:entry + 8])
        if tag != wanted:
            continue
        size = _TIFF_TYPE_SIZES.get(kind, 1) * items
        if size > MAX_EXIF_BYTES:
            return None
        if size <= 4:
            return tiff[entry + 8:entry + 8 + size]
        start = struct.unpack(order + "I", tiff[entry + 8:entry + 12])[0]
        if start + size > len(tiff):
            return None
        return tiff[start:start + size]
    return None


def _decode_user_comment(raw: bytes) -> str | None:
    prefix, body = raw[:8], raw[8:]
    if prefix == b"UNICODE\x00":
        if body[:2] in (b"\xfe\xff", b"\xff\xfe"):
            text = body.decode("utf-16", errors="replace")
        else:
            text = body.decode("utf-16-be", errors="replace")  # piexif's "unicode", as Forge writes it
    elif prefix == b"ASCII\x00\x00\x00":
        text = body.decode("latin-1")
    elif prefix == b"JIS\x00\x00\x00\x00\x00":
        text = body.decode("shift_jis", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    text = text.rstrip("\x00").strip()
    return text or None


def read_generation_metadata(path: str) -> tuple[str | None, str]:
    """Read an image's generation parameters from wherever Forge put them.

    Both carriers are optional and independent settings: `enable_pnginfo`
    embeds a tEXt chunk in the PNG (an EXIF UserComment in JPEG and WebP),
    `save_txt` writes a sibling .txt. Reading only the chunk loses the history
    of anyone who turned that off but kept the text files.

    Returns the text and which source supplied it.
    """
    lowered = path.lower()
    if lowered.endswith(".png"):
        embedded = read_png_parameters(path)
        if embedded:
            return embedded, "png_chunk"
    elif lowered.endswith((".jpg", ".jpeg", ".webp")):
        embedded = read_exif_parameters(path)
        if embedded:
            return embedded, "exif_user_comment"

    sidecar = os.path.splitext(path)[0] + ".txt"
    try:
        with open(sidecar, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read().strip()
    except OSError:
        return None, "none"
    return (text, "txt_sidecar") if text else (None, "none")


def read_parameters_from_bytes(blob: bytes) -> tuple[str | None, str]:
    """The same embedded-metadata readers for an image held in memory."""
    if blob.startswith(PNG_SIGNATURE):
        text = _png_parameters(io.BytesIO(blob))
        return (text, "png_chunk") if text else (None, "none")
    if blob[:2] == JPEG_SOI or (blob[:4] == b"RIFF" and blob[8:12] == b"WEBP"):
        text = _exif_parameters(io.BytesIO(blob))
        return (text, "exif_user_comment") if text else (None, "none")
    return None, "none"


def _decode_latin1_or_utf8(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _inflate(data: bytes) -> bytes | None:
    try:
        inflater = zlib.decompressobj()
        out = inflater.decompress(data, MAX_TEXT_BYTES)
    except zlib.error:
        return None
    return None if inflater.unconsumed_tail else out


def _decode_text_chunk(chunk_type: bytes, body: bytes) -> str | None:
    keyword, _, remainder = body.partition(b"\x00")
    if keyword != PARAMETERS_KEY:
        return None
    if chunk_type == b"tEXt":
        return _decode_latin1_or_utf8(remainder)
    if chunk_type == b"zTXt":
        if not remainder or remainder[0] != 0:
            return None
        inflated = _inflate(remainder[1:])
        return None if inflated is None else _decode_latin1_or_utf8(inflated)
    # iTXt: compression flag, compression method, language tag, translated key
    if len(remainder) < 2 or remainder[0] not in (0, 1):
        return None
    compressed = remainder[0] == 1
    rest = remainder[2:]
    _, _, rest = rest.partition(b"\x00")
    _, _, rest = rest.partition(b"\x00")
    if compressed:
        inflated = _inflate(rest)
        if inflated is None:
            return None
        rest = inflated
    return rest.decode("utf-8", errors="replace")


def parse_infotext(text: str) -> Infotext:
    """Split raw infotext into prompt, negative prompt and parameter pairs."""
    if not text:
        return Infotext("", "", {}, ())

    lines = text.split("\n")
    params_line = ""
    if len(lines) > 1 and _looks_like_params(lines[-1]):
        params_line = lines[-1]
        lines = lines[:-1]

    body = "\n".join(lines)
    prompt, separator, negative = body.partition("Negative prompt:")
    prompt = prompt.strip()
    negative = negative.strip() if separator else ""

    params = {
        key.strip(): value.strip().strip('"')
        for key, value in PARAM_PATTERN.findall(params_line)
        if key.strip()
    }
    loras = tuple(
        (name.strip(), float(weight))
        for name, weight in LORA_PATTERN.findall(prompt)
    )
    return Infotext(prompt=prompt, negative=negative, params=params, loras=loras)


def _looks_like_params(line: str) -> bool:
    return "Steps:" in line or "Sampler:" in line or "CFG scale:" in line
