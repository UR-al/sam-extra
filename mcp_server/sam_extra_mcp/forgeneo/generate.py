# Vendored from https://github.com/eduardoabreu81/forgeneo-mcp at commit a103dc5
# (forgeneo_mcp/generate.py) into sam-extra (mcp_server/sam_extra_mcp/forgeneo/generate.py).
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
# - resolve_output_dir() takes the server's ForgePaths: a relative outdir (Forge's default,
#   "output/txt2img-images") resolves against the Forge installation found from the extension's
#   own location, so same-PC use needs no FORGE_PATH_MAP. Upstream's module-path inference stays
#   as the next fallback. `saved_by="api"` skips the outdir_samples override, which Forge's API
#   ignores (modules/api/api.py sets outpath_samples = outdir_txt2img/img2img_samples); upstream
#   looked for API results in the override folder.
# - run_generation() only returns files that belong to this call. Upstream returned every media
#   file that appeared in the output folder while the request ran, so a generation started from
#   the web UI (or by another agent) at the same time came back as ours. A new file now counts
#   when its embedded infotext (PNG chunk, EXIF UserComment or .txt sidecar) equals one this call
#   produced, or carries one of its seeds with the same prompt, or - without any metadata - has
#   one of its seeds in its file name; and only when it was written between the request and the
#   response. Files with someone else's parameters are reported as ignored. Auxiliary saves of
#   the same image (-before-highres-fix, -mask, ADetailer's -ad-before ...) are listed with it,
#   not as results; grid files (grid-NNNN) are never results. Caveat: "related" is built from
#   seed-and-prompt matches, so a concurrent generation with the same seed and prompt is listed
#   there as well (see _match_files).
# - requested_pixels(): the total a payload asks Forge to render (sizes, hires, batch, n_iter),
#   for the server's pixel budget.
# - The API always returns the images it produced. Whatever this call cannot find on disk
#   (folder unknown, saving disabled, no metadata to tell it apart) is decoded from that response
#   - matched by its embedded infotext, else by position - and saved into the fallback folder
#   the server passes (sam-extra: <Forge data>/sam-extra/mcp/outputs). Upstream never passed
#   one, so generate answered ok:false after Forge had already rendered the picture.
# - Saved files get the extension of their real format (PNG, JPEG, WebP, AVIF, JXL) instead of
#   always ".png", and never overwrite an existing file. The grid image is not saved.
# - encode_init_image() accepts still images only (not .mp4/.mkv) and refuses files over
#   MAX_INIT_IMAGE_BYTES before reading them. It reads local files only (local_file_path):
#   a UNC share, \\?\ or \\.\ device path is refused before the file system is touched, and
#   the file is opened through its normalised absolute path (\??\UNC\... becomes local), so
#   an agent-chosen path cannot make Windows contact another host (SMB/WebDAV, NTLM, DNS).
#   The suffix is checked before the file's existence.

"""Generation requests and result collection.

The API always answers with base64 images. Returning those to an agent is a
context disaster — a single batch can be tens of megabytes — so when the output
directory is reachable we hand back file paths instead, and only fall back to
decoding base64 when it is not.

A file is only handed back when it can be shown to belong to this request:
the output folder is shared with everyone else using the instance.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .infotext import parse_infotext, read_generation_metadata, read_parameters_from_bytes

if TYPE_CHECKING:
    from ..forge_paths import ForgePaths
    from .client import ApiResult, ForgeClient

MODELS_MARKER = "/models/"
NEW_FILE_GRACE_SECONDS = 2.0
# Forge also drops a sidecar .txt with the infotext next to each image; the
# caller asked for artwork, not for the log.
MEDIA_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".avif", ".jxl", ".mp4", ".mkv")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".avif", ".jxl")
MAX_INIT_IMAGE_BYTES = 64 * 1024 * 1024
# Extra saves of the same image, written with the same infotext (modules/processing.py;
# "-ad-before" and "-ad-preview-N" are ADetailer's). They are reported next to the result,
# never as one.
AUXILIARY_SUFFIXES = (
    "-before-highres-fix",
    "-before-face-restoration",
    "-before-color-correction",
    "-mask",
    "-mask-composite",
    "-ad-before",
)
AUXILIARY_PATTERN = re.compile(r"-ad-preview(?:-\d+)?$")
# images.save_image names grids "<basename>-NNNN" with basename "grid"; a grid can carry the
# first image's infotext, so it must never stand in for a result.
GRID_PATTERN = re.compile(r"^grid-\d{4}")

MATCH_INFOTEXT = "infotext"
MATCH_SEED_PROMPT = "seed and prompt"
MATCH_SEED_FILENAME = "seed in file name"
SAVED_BY_INFOTEXT = "saved from the API response (matched by infotext)"
SAVED_BY_POSITION = "saved from the API response (matched by position)"
SAVED_UNATTRIBUTED = "saved from the API response (could not tell which seed)"
_STRENGTH = {MATCH_INFOTEXT: 3, MATCH_SEED_PROMPT: 2, MATCH_SEED_FILENAME: 1}


@dataclass(frozen=True)
class DeliveredImage:
    path: str
    index: int | None
    seed: int | None
    matched_by: str
    related: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        payload: dict[str, Any] = {"file": self.path, "seed": self.seed, "matched_by": self.matched_by}
        if self.related:
            payload["related_files"] = list(self.related)
        return payload


@dataclass(frozen=True)
class GenerationResult:
    ok: bool
    files: tuple[str, ...] = ()
    delivery: str = "none"
    info: dict | None = None
    error: str | None = None
    images: tuple[DeliveredImage, ...] = ()
    ignored: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        payload: dict[str, Any] = {"ok": self.ok, "delivery": self.delivery}
        if self.files:
            payload["files"] = list(self.files)
        if self.images:
            payload["images"] = [image.as_dict() for image in self.images]
        if self.info:
            payload["parameters"] = self.info
        if self.ignored:
            payload["ignored_new_files"] = {
                "count": len(self.ignored),
                "examples": [os.path.basename(path) for path in self.ignored[:5]],
                "why": (
                    "written while this request ran, but carrying another generation's parameters - "
                    "someone else's work (the web UI or another client), so not returned"
                ),
            }
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class ExpectedImage:
    """One image this request produced, as the API response describes it."""

    index: int
    seed: int | None
    prompt: str | None
    infotext: str | None


def resolve_output_dir(
    client: ForgeClient,
    options: dict,
    mode: str = "txt2img",
    paths: ForgePaths | None = None,
    saved_by: str = "ui",
) -> str | None:
    """Find a locally readable path for Forge's output folder.

    Explicit configuration wins. Otherwise a relative outdir option (Forge's
    default) is resolved against the installation this server ships in, then
    against a root inferred from any absolute module path the instance reports.

    `saved_by` matters when the operator set the "Output Directory" override
    (outdir_samples): the web UI saves there (modules/txt2img.py), but the API
    always saves to outdir_txt2img_samples / outdir_img2img_samples
    (modules/api/api.py). "ui" answers for the operator's own generations (the
    history), "api" for this server's.
    """
    if client.config.output_dir:
        return client.config.output_dir if os.path.isdir(client.config.output_dir) else None

    key = "outdir_img2img_samples" if mode == "img2img" else "outdir_txt2img_samples"
    override = "" if saved_by == "api" else (options.get("outdir_samples") or "")
    outdir = (override or options.get(key) or "").strip()
    if not outdir:
        return None

    normalised = outdir.replace("\\", "/")
    if _is_absolute(normalised):
        local = client.config.localise(normalised)
        return local if local and _usable_target(local) else None

    if paths is not None:
        found = paths.resolve_output_dir(outdir)
        if found:
            return found

    root = _installation_root(client, options)
    if not root:
        return None
    candidate = os.path.join(root, *normalised.split("/"))
    return candidate if _usable_target(candidate) else None


def _usable_target(path: str) -> bool:
    """Accept a folder Forge has not created yet.

    Output subfolders appear the first time a mode is used: on a machine that
    had only ever run txt2img, outdir_img2img_samples pointed at a directory
    that did not exist, and requiring it up front discarded the result of a
    generation that had already happened. An existing parent is enough to show
    the path mapping is right.
    """
    if os.path.isdir(path):
        return True
    parent = os.path.dirname(path.rstrip("/" + chr(92)))
    return bool(parent) and os.path.isdir(parent)


def _is_absolute(path: str) -> bool:
    return path.startswith("/") or (len(path) > 1 and path[1] == ":")


def _installation_root(client: ForgeClient, options: dict) -> str | None:
    """Infer the Forge root from any absolute path the instance reports."""
    candidates: list[str] = []
    for key, value in options.items():
        if not key.startswith("forge_additional_modules"):
            continue
        if isinstance(value, list):
            candidates.extend(str(item) for item in value)

    for raw in candidates:
        normalised = str(raw).replace("\\", "/")
        marker = normalised.lower().find(MODELS_MARKER)
        if marker == -1:
            continue
        local = client.config.localise(normalised[:marker])
        if local and os.path.isdir(local):
            return local
    return None


def local_file_path(path: str) -> str | None:
    """The absolute local path `path` names, or None for a network share or device path.

    Touching \\\\host\\share\\... (also //host/share, \\\\?\\UNC\\..., \\\\.\\...) makes
    Windows contact that host over SMB or WebDAV - the user's NTLM credentials and the
    host name (to DNS) leave the machine on a path the agent chose. \\??\\UNC\\... does the
    same without the leading pair, since the Win32 layer hands \\??\\ names to the NT
    namespace unchanged; the normalised absolute path turns it into a harmless local name
    (C:\\??\\...), so callers must use the returned path, not the one they were given.
    """
    try:
        absolute = os.path.abspath(path)
    except (OSError, ValueError):
        return None
    if absolute.replace(chr(92), "/").startswith("//"):
        return None
    return absolute


def encode_init_image(path: str) -> tuple[str | None, str | None]:
    """Read a local image into base64 for img2img. Returns (data, error)."""
    # The suffix is checked first: nothing touches the file system for a name that could not
    # be used anyway, so the error cannot serve as an existence check for arbitrary files.
    if not path.lower().endswith(IMAGE_SUFFIXES):
        return None, f"init image is not a recognised image file: {path}"
    local = local_file_path(path)
    if local is None:
        return None, (
            f"init image must be a file on this PC, not a network share or device path: {path} "
            "(copy it into a local folder first)"
        )
    if not os.path.isfile(local):
        return None, f"init image not found: {path}"
    try:
        size = os.path.getsize(local)
    except OSError as exc:
        return None, f"cannot read init image: {exc}"
    if size > MAX_INIT_IMAGE_BYTES:
        return None, f"init image is larger than {MAX_INIT_IMAGE_BYTES // (1024 * 1024)} MB: {path}"
    try:
        with open(local, "rb") as handle:
            blob = handle.read()
    except OSError as exc:
        return None, f"cannot read init image: {exc}"
    if not blob:
        return None, f"init image is empty: {path}"
    return base64.b64encode(blob).decode("ascii"), None


def build_payload(
    prompt: str,
    negative_prompt: str = "",
    steps: int | None = None,
    cfg_scale: float | None = None,
    sampler_name: str | None = None,
    scheduler: str | None = None,
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    batch_size: int = 1,
    distilled_cfg_scale: float | None = None,
    extra: dict | None = None,
) -> dict:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "batch_size": batch_size,
        "save_images": True,
        "send_images": True,
    }
    if steps is not None:
        payload["steps"] = steps
    if cfg_scale is not None:
        payload["cfg_scale"] = cfg_scale
    if distilled_cfg_scale is not None:
        # Forge feeds this into set_shift(); the UI labels it "Shift" or
        # "Distilled CFG Scale" per architecture. Omitting it does not mean
        # "use the architecture default" - it means the API's own 3.5.
        payload["distilled_cfg_scale"] = distilled_cfg_scale
    if sampler_name:
        payload["sampler_name"] = sampler_name
    if scheduler:
        payload["scheduler"] = scheduler
    if extra:
        payload.update(extra)
    return payload


def requested_pixels(payload: dict) -> int:
    """How many pixels a txt2img/img2img payload asks Forge to render, all images included.

    width x height per image - or, with the hires fix on, the larger of that and the
    hires size (hr_resize_x/hr_resize_y, a 0 side following the aspect ratio, else
    hr_scale, Forge's default 2.0) - times batch_size times n_iter.
    """

    def number(key: str, default: float = 0.0) -> float:
        try:
            value = float(payload.get(key) or default)
        except (TypeError, ValueError):
            return default
        return value if value == value and value not in (float("inf"), float("-inf")) else default

    width, height = max(0.0, number("width")), max(0.0, number("height"))
    per_image = width * height
    if payload.get("enable_hr"):
        resize_x, resize_y = max(0.0, number("hr_resize_x")), max(0.0, number("hr_resize_y"))
        if resize_x or resize_y:
            if not resize_x and height:
                resize_x = resize_y * width / height
            if not resize_y and width:
                resize_y = resize_x * height / width
            hires = resize_x * resize_y
        else:
            scale = number("hr_scale", 2.0)
            hires = (width * scale) * (height * scale)
        per_image = max(per_image, hires)
    batch = max(1, int(number("batch_size", 1.0)))
    iterations = max(1, int(number("n_iter", 1.0)))
    return int(per_image) * batch * iterations


def run_generation(
    client: ForgeClient,
    payload: dict,
    output_dir: str | None,
    mode: str = "txt2img",
    fallback_dir: str | None = None,
    clock: Callable[[], float] = time.time,
) -> GenerationResult:
    before = _snapshot(output_dir)
    started = clock()

    result: ApiResult = client.img2img(payload) if mode == "img2img" else client.txt2img(payload)
    finished = clock()
    if not result.ok:
        return GenerationResult(False, error=result.error)

    data = result.value or {}
    raw_info = _parse_info(data.get("info"))
    info = _trim_info(raw_info) if raw_info else None
    images_b64 = [item for item in (data.get("images") or []) if isinstance(item, str)]
    first = _as_int((raw_info or {}).get("index_of_first_image")) or 0
    expected = expected_images(raw_info, payload, len(images_b64))
    if not expected:
        return GenerationResult(False, error="Forge returned no images", info=info)

    warnings: list[str] = []
    delivered: dict[int, DeliveredImage] = {}
    ignored: tuple[str, ...] = ()
    if output_dir:
        candidates = _new_media(output_dir, before, started, finished)
        found, ignored, unattributed = _match_files(candidates, expected)
        delivered.update(found)
        if unattributed:
            warnings.append(
                f"{len(unattributed)} new file(s) carry no generation parameters and no seed of this "
                "request in their name, so they could not be told apart from other generations"
            )
    else:
        warnings.append("Forge's output folder is not readable from here")

    missing = [item for item in expected if item.index not in delivered]
    saved: list[DeliveredImage] = []
    if missing:
        if not fallback_dir:
            warnings.append("no fallback folder is configured for images that could not be found on disk")
        else:
            saved, save_warnings = _save_from_response(
                images_b64, first, expected, missing, fallback_dir, delivered_indices=set(delivered)
            )
            warnings.extend(save_warnings)

    from_disk = [delivered[index] for index in sorted(delivered)]
    images = tuple(from_disk + saved)
    if not images:
        seeds = ", ".join(str(item.seed) for item in expected if item.seed is not None) or "unknown"
        detail = "; ".join(warnings) or "nothing was found or saved"
        return GenerationResult(
            False,
            info=info,
            ignored=ignored,
            warnings=tuple(warnings),
            error=(
                f"Forge generated {len(expected)} image(s) (seeds {seeds}) but none could be located "
                f"or saved: {detail}"
            ),
        )

    still_missing = [item for item in expected if item.index not in {image.index for image in images}]
    if still_missing and not any(image.index is None for image in images):
        warnings.append(
            "not delivered: seed(s) " + ", ".join(str(item.seed) for item in still_missing)
        )
    if from_disk and saved:
        delivery = "filesystem + api response"
    elif from_disk:
        delivery = "filesystem"
    else:
        delivery = "api response"
    return GenerationResult(
        True,
        files=tuple(image.path for image in images),
        delivery=delivery,
        info=info,
        images=images,
        ignored=ignored,
        warnings=tuple(warnings),
    )


def expected_images(info: dict | None, payload: dict, image_count: int) -> list[ExpectedImage]:
    """The images a response describes, one per sample (the grid excluded).

    `infotexts[index_of_first_image:]`, `all_seeds` and `all_prompts` run in the
    same order as the saved samples (modules/processing.py). Without an info
    block nothing identifies an image on disk, but each returned image is still
    this request's own.
    """
    prompt = payload.get("prompt") if isinstance(payload.get("prompt"), str) else None
    if not isinstance(info, dict):
        return [ExpectedImage(index, None, prompt, None) for index in range(image_count)]
    first = _as_int(info.get("index_of_first_image")) or 0
    infotexts = [text if isinstance(text, str) else None for text in (info.get("infotexts") or [])][first:]
    seeds = [_as_int(seed) for seed in (info.get("all_seeds") or [])]
    prompts = [text if isinstance(text, str) else None for text in (info.get("all_prompts") or [])]
    count = len(infotexts) or len(seeds) or max(image_count - first, 0)
    return [
        ExpectedImage(
            index=index,
            seed=seeds[index] if index < len(seeds) else None,
            prompt=prompts[index] if index < len(prompts) else prompt,
            infotext=infotexts[index] if index < len(infotexts) else None,
        )
        for index in range(count)
    ]


def _normalise_text(text: str | None) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _match_text(text: str, expected: list[ExpectedImage]) -> tuple[int, str] | None:
    """Which expected image an infotext belongs to, if any."""
    normalised = _normalise_text(text)
    for item in expected:
        if item.infotext and _normalise_text(item.infotext) == normalised:
            return item.index, MATCH_INFOTEXT
    parsed = parse_infotext(normalised)
    seed = _as_int(parsed.params.get("Seed"))
    if seed is None:
        return None
    for item in expected:
        if item.seed != seed:
            continue
        if item.prompt is None or _normalise_text(item.prompt) == parsed.prompt:
            return item.index, MATCH_SEED_PROMPT
    return None


def _match_filename(path: str, expected: list[ExpectedImage]) -> tuple[int, str] | None:
    name = os.path.basename(path)
    hits = [
        item
        for item in expected
        if item.seed is not None and re.search(rf"(?<!\d){item.seed}(?!\d)", name)
    ]
    if len(hits) == 1:
        return hits[0].index, MATCH_SEED_FILENAME
    return None


def _is_auxiliary(path: str) -> bool:
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    return stem.endswith(AUXILIARY_SUFFIXES) or AUXILIARY_PATTERN.search(stem) is not None


def _is_grid(path: str) -> bool:
    return GRID_PATTERN.match(os.path.basename(path).lower()) is not None


def _match_files(
    candidates: list[tuple[float, str]], expected: list[ExpectedImage]
) -> tuple[dict[int, DeliveredImage], tuple[str, ...], tuple[str, ...]]:
    """Attribute new files to this request's images.

    Returns the best file per image (with any auxiliary saves of it), the files
    that carry someone else's parameters, and the files nothing could attribute.

    Known caveat, kept on purpose: the other files matched to an image become its
    `related` files, and a match on seed and prompt is enough for that - which is
    how auxiliary saves whose infotext differs (ADetailer's "-ad-before") are
    grouped. A concurrent generation with the same seed and the same prompt is
    therefore listed there too; it never becomes the result itself while this
    request's own file carries its exact infotext (that match ranks higher).
    """
    by_index: dict[int, list[tuple[int, bool, float, str, str]]] = {}
    foreign: list[str] = []
    unattributed: list[str] = []
    for mtime, path in candidates:
        if _is_grid(path):
            continue  # saved only when the grids share the samples folder; never a result
        text, _source = read_generation_metadata(path)
        hit = _match_text(text, expected) if text else _match_filename(path, expected)
        if hit is None:
            (foreign if text else unattributed).append(_normalise(path))
            continue
        index, how = hit
        by_index.setdefault(index, []).append((_STRENGTH[how], not _is_auxiliary(path), mtime, path, how))

    seeds = {item.index: item.seed for item in expected}
    chosen: dict[int, DeliveredImage] = {}
    for index, matches in by_index.items():
        matches.sort(reverse=True)
        _, _, _, path, how = matches[0]
        related = tuple(_normalise(other[3]) for other in matches[1:])
        chosen[index] = DeliveredImage(_normalise(path), index, seeds.get(index), how, related)
    return chosen, tuple(foreign), tuple(unattributed)


def _save_from_response(
    images_b64: list[str],
    first: int,
    expected: list[ExpectedImage],
    missing: list[ExpectedImage],
    target_dir: str,
    delivered_indices: set[int],
) -> tuple[list[DeliveredImage], list[str]]:
    """Save the response's own copies of the images that were not found on disk."""
    warnings: list[str] = []
    decoded: list[tuple[int, bytes]] = []
    for position, encoded in enumerate(images_b64):
        if position < first:
            continue  # the grid
        blob = _decode_image(encoded)
        if blob is None:
            warnings.append(f"returned image {position} could not be decoded")
            continue
        decoded.append((position, blob))

    wanted = {item.index: item for item in missing}
    assigned: dict[int, tuple[bytes, str]] = {}
    used: set[int] = set()
    identified: set[int] = set()
    for position, blob in decoded:
        text, _ = read_parameters_from_bytes(blob)
        if not text:
            continue
        hit = _match_text(text, expected)
        if hit is None:
            continue
        identified.add(position)
        index = hit[0]
        if index in wanted and index not in assigned:
            assigned[index] = (blob, SAVED_BY_INFOTEXT)
            used.add(position)

    aligned = len(images_b64) - first == len(expected)
    if aligned:
        by_position = dict(decoded)
        for index in wanted:
            position = first + index
            if index in assigned or position in identified or position not in by_position:
                continue
            assigned[index] = (by_position[position], SAVED_BY_POSITION)
            used.add(position)

    leftovers: list[bytes] = []
    if len(assigned) < len(wanted) and not aligned:
        # Cannot tell which unidentified image is which: every one of them is still
        # this request's own, so save them all rather than lose one.
        leftovers = [blob for position, blob in decoded if position not in used and position not in identified]
        if leftovers and delivered_indices:
            warnings.append(
                "some images saved from the API response could not be matched to a seed and may "
                "repeat a file already returned"
            )

    saved: list[DeliveredImage] = []
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as exc:
        return [], warnings + [f"cannot create the fallback folder {target_dir}: {exc}"]

    for index in sorted(assigned):
        blob, how = assigned[index]
        seed = wanted[index].seed
        path = _write_new_file(target_dir, f"sam-extra-mcp-{stamp}-{seed if seed is not None else 'x'}-{index}", blob)
        if path is None:
            warnings.append(f"could not save the image for seed {seed} into {target_dir}")
            continue
        saved.append(DeliveredImage(_normalise(path), index, seed, how))
    for number, blob in enumerate(leftovers):
        path = _write_new_file(target_dir, f"sam-extra-mcp-{stamp}-unmatched-{number}", blob)
        if path is None:
            warnings.append(f"could not save an image into {target_dir}")
            continue
        saved.append(DeliveredImage(_normalise(path), None, None, SAVED_UNATTRIBUTED))
    return saved, warnings


def _write_new_file(folder: str, stem: str, blob: bytes) -> str | None:
    """Write `blob` under a name that does not exist yet; never overwrite."""
    extension = image_extension(blob)
    if extension is None:
        return None
    for attempt in range(100):
        name = f"{stem}{extension}" if attempt == 0 else f"{stem}-{attempt}{extension}"
        path = os.path.join(folder, name)
        try:
            with open(path, "xb") as handle:
                handle.write(blob)
        except FileExistsError:
            continue
        except OSError:
            return None
        return path
    return None


def image_extension(blob: bytes) -> str | None:
    """File extension for an encoded image, from its signature."""
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if blob.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return ".webp"
    if blob[4:12] in (b"ftypavif", b"ftypavis"):
        return ".avif"
    if blob.startswith(b"\xff\x0a") or blob.startswith(b"\x00\x00\x00\x0cJXL \r\n\x87\n"):
        return ".jxl"
    return None


def _decode_image(encoded: str) -> bytes | None:
    payload = encoded.split(",", 1)[-1] if encoded.startswith("data:") else encoded
    try:
        blob = base64.b64decode(payload)
    except (binascii.Error, ValueError):
        return None
    return blob or None


def _snapshot(output_dir: str | None) -> set[str]:
    if not output_dir:
        return set()
    found: set[str] = set()
    for root, _, files in os.walk(output_dir):
        for name in files:
            found.add(os.path.join(root, name))
    return found


def _new_media(
    output_dir: str, before: set[str], started: float, finished: float | None = None
) -> list[tuple[float, str]]:
    """Media files that appeared while the request ran, oldest first."""
    fresh: list[tuple[float, str]] = []
    upper = None if finished is None else finished + NEW_FILE_GRACE_SECONDS
    for root, _, files in os.walk(output_dir):
        for name in files:
            if not name.lower().endswith(MEDIA_SUFFIXES):
                continue
            full = os.path.join(root, name)
            if full in before:
                continue
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            if mtime < started - NEW_FILE_GRACE_SECONDS:
                continue
            if upper is not None and mtime > upper:
                continue
            fresh.append((mtime, full))
    fresh.sort()
    return fresh


def _new_files(output_dir: str, before: set[str], started: float, finished: float | None = None) -> tuple[str, ...]:
    return tuple(_normalise(path) for _, path in _new_media(output_dir, before, started, finished))


def _normalise(path: str) -> str:
    """Present one separator style; UNC roots keep their leading pair."""
    unc = path.startswith("//") or path.startswith(chr(92) * 2)
    cleaned = path.replace(chr(92), "/")
    return "//" + cleaned.lstrip("/") if unc else cleaned


def _parse_info(raw: Any) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except ValueError:
            return None
        return loaded if isinstance(loaded, dict) else None
    return None


def _decode_info(raw: Any) -> dict | None:
    parsed = _parse_info(raw)
    return _trim_info(parsed) if parsed is not None else None


def _trim_info(info: dict) -> dict:
    """Keep the fields an agent needs; drop the rest to protect the context."""
    keys = (
        "prompt",
        "negative_prompt",
        "seed",
        "all_seeds",
        "steps",
        "cfg_scale",
        "sampler_name",
        "scheduler",
        "width",
        "height",
        "sd_model_name",
    )
    return {key: info[key] for key in keys if key in info}


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
