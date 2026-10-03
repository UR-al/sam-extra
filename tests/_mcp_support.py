"""Shared fixtures for the sam-extra MCP server tests (``mcp_server/``).

- puts ``mcp_server`` on sys.path so ``sam_extra_mcp`` imports without installing anything;
- builds a throw-away Forge tree (``<forge>/extensions/sam-extra/mcp_server/...``) so path
  discovery and the settings-file policy work exactly as in a real installation;
- ``FakeForge``: an ``httpx.MockTransport`` handler that answers like Forge's API and, on
  txt2img/img2img, writes real output files (PNG text chunk, EXIF UserComment or .txt sidecar)
  the way ``modules/processing.py`` does, so result matching is tested against real files;
- image builders with embedded generation parameters.

Nothing here touches the network, a GPU or a real Forge.
"""
from __future__ import annotations

import base64
import json
import os
import struct
import sys
import zlib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MCP_PROJECT = ROOT / "mcp_server"
if str(MCP_PROJECT) not in sys.path:
    sys.path.append(str(MCP_PROJECT))

try:
    import httpx
except ImportError:  # pragma: no cover - Forge's venv and CI (gradio) both ship httpx
    httpx = None

HAS_HTTPX = httpx is not None

try:  # the real SDK, not merely something importable as "mcp"
    from mcp.server import MCPServer  # noqa: F401
    from mcp.types import ToolAnnotations  # noqa: F401
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

NULL_JSON = {"content": b"null", "headers": {"content-type": "application/json"}}


# -- images with embedded parameters ------------------------------------------------------------

def _png_chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def png_bytes(text: str | None = None, *, force_itxt: bool = False) -> bytes:
    """A PNG whose "parameters" chunk is written the way Pillow writes it (tEXt, or iTXt when needed)."""
    out = bytearray(b"\x89PNG\r\n\x1a\n")
    out += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    if text is not None:
        try:
            if force_itxt:
                raise UnicodeEncodeError("latin-1", "", 0, 1, "forced")
            out += _png_chunk(b"tEXt", b"parameters\x00" + text.encode("latin-1"))
        except UnicodeEncodeError:
            out += _png_chunk(b"iTXt", b"parameters\x00\x00\x00\x00\x00" + text.encode("utf-8"))
    out += _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
    out += _png_chunk(b"IEND", b"")
    return bytes(out)


def exif_tiff(text: str, order: str = ">") -> bytes:
    """A TIFF/EXIF block with only a UserComment, encoded as piexif's "unicode" (UTF-16BE)."""
    comment = b"UNICODE\x00" + text.encode("utf-16-be")
    header = (b"MM" if order == ">" else b"II") + struct.pack(order + "HI", 42, 8)
    ifd0 = struct.pack(order + "H", 1) + struct.pack(order + "HHII", 0x8769, 4, 1, 26) + struct.pack(order + "I", 0)
    exif_ifd = struct.pack(order + "H", 1) + struct.pack(order + "HHII", 0x9286, 7, len(comment), 44)
    exif_ifd += struct.pack(order + "I", 0)
    return header + ifd0 + exif_ifd + comment


def jpeg_bytes(text: str | None = None, order: str = ">") -> bytes:
    out = bytearray(b"\xff\xd8")
    out += b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    if text is not None:
        exif = b"Exif\x00\x00" + exif_tiff(text, order)
        out += b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    out += b"\xff\xda" + struct.pack(">H", 2) + b"\x00\x00\xff\xd9"
    return bytes(out)


def webp_bytes(text: str | None = None, with_exif_header: bool = False) -> bytes:
    def chunk(fourcc: bytes, data: bytes) -> bytes:
        return fourcc + struct.pack("<I", len(data)) + data + (b"\x00" if len(data) % 2 else b"")

    body = b"WEBP" + chunk(b"VP8L", b"\x2f\x00\x00\x00\x00")
    if text is not None:
        tiff = exif_tiff(text, "<")
        body += chunk(b"EXIF", (b"Exif\x00\x00" + tiff) if with_exif_header else tiff)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def infotext(prompt: str, seed: int, *, negative: str = "worst quality", steps: int = 10, cfg: float = 1.5,
             model: str = "animeMix_v10", size: str = "832x1216", extra: str = "") -> str:
    return (
        f"{prompt}\nNegative prompt: {negative}\n"
        f"Steps: {steps}, Sampler: ER SDE, Schedule type: Beta, CFG scale: {cfg:g}, Seed: {seed}, "
        f"Size: {size}, Model: {model}{extra}, Version: f2.0"
    )


# -- a Forge installation on disk ------------------------------------------------------------------

def make_forge_tree(base: Path, *, settings: dict | None = None, program: bool = True,
                    extension_name: str = "sam-extra") -> dict:
    """``<base>/forge`` with an extension holding ``mcp_server``; returns the interesting paths."""
    forge = Path(base) / "forge"
    (forge / "modules").mkdir(parents=True, exist_ok=True)
    if program:
        (forge / "webui.py").write_text("# Forge\n", encoding="utf-8")
    extension = forge / "extensions" / extension_name
    package = extension / "mcp_server" / "sam_extra_mcp"
    package.mkdir(parents=True, exist_ok=True)
    (extension / "mcp_server" / "pyproject.toml").write_text('[project]\nname = "sam-extra-mcp"\n', encoding="utf-8")
    (forge / "models" / "VAE").mkdir(parents=True, exist_ok=True)
    (forge / "models" / "text_encoder").mkdir(parents=True, exist_ok=True)
    if settings is not None:
        write_settings(forge / "config.json", settings)
    return {
        "forge": forge,
        "extension": extension,
        "module_file": str(package / "forge_paths.py"),
        "settings": forge / "config.json",
        "output": forge / "output" / "txt2img-images",
    }


def write_settings(path: Path, settings: dict) -> None:
    """Write a settings file and make sure its stat changes even within one timer tick."""
    previous = path.stat() if path.exists() else None
    path.write_text(json.dumps(settings, indent=4), encoding="utf-8")
    if previous is not None:
        current = path.stat()
        if (current.st_mtime_ns, current.st_size) == (previous.st_mtime_ns, previous.st_size):
            os.utime(path, ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000))


# -- a fake Forge API -------------------------------------------------------------------------------

DEFAULT_OPTIONS = {
    "sd_model_checkpoint": "Anima/animeMix_v10.safetensors",
    "forge_preset": "anima",
    "outdir_samples": "",
    "outdir_txt2img_samples": "output/txt2img-images",
    "outdir_img2img_samples": "output/img2img-images",
    "enable_pnginfo": True,
    "save_txt": False,
    "anima_t2i_step": 32,
    "anima_t2i_cfg": 4.0,
    "anima_t2i_dcfg": 3.0,
    "anima_t2i_sampler": "ER SDE",
    "anima_t2i_scheduler": "Beta",
    "anima_t2i_width": 832,
    "anima_t2i_height": 1216,
    "forge_checkpoint_anima": "Anima/animeMix_v10.safetensors",
    "forge_checkpoint_krea": "Krea/kreaMix_v20.safetensors",
}


class FakeForge:
    """Answers like Forge's API; generation writes files like modules/processing.py does."""

    def __init__(self, forge_dir: Path, options: dict | None = None) -> None:
        self.forge = Path(forge_dir)
        self.options = dict(DEFAULT_OPTIONS if options is None else options)
        models = self.forge / "models"
        self.options.setdefault(
            "forge_additional_modules_anima",
            [str(models / "VAE" / "qwen_image_vae.safetensors"), str(models / "text_encoder" / "qwen_3_06b_base.safetensors")],
        )
        self.sd_models = [
            {"title": "Anima/animeMix_v10.safetensors [abc123]", "model_name": "animeMix_v10", "hash": "abc123",
             "sha256": "a" * 64, "filename": str(models / "Stable-diffusion" / "Anima" / "animeMix_v10.safetensors")},
            {"title": "Krea/kreaMix_v20.safetensors [def456]", "model_name": "kreaMix_v20", "hash": "def456",
             "sha256": "b" * 64, "filename": str(models / "Stable-diffusion" / "Krea" / "kreaMix_v20.safetensors")},
        ]
        self.sd_modules = [
            {"model_name": "qwen_image_vae.safetensors", "filename": str(models / "VAE" / "qwen_image_vae.safetensors")},
            {"model_name": "qwen_3_06b_base.safetensors", "filename": str(models / "text_encoder" / "qwen_3_06b_base.safetensors")},
        ]
        self.loras: list[dict] = []
        self.requests: list[tuple[str, str, object]] = []
        self.queries: list[tuple[str, dict]] = []
        self.authorizations: list[str | None] = []  # the Authorization header of each request
        self.fail_generation: tuple[int, str] | None = None  # (status, body) answered by txt2img/img2img
        # Generation behaviour
        self.save_files = True          # opts.samples_save
        self.samples_format = "png"     # png | jpg | webp
        self.embed = True               # opts.enable_pnginfo
        self.save_txt = False           # opts.save_txt
        self.return_grid = True
        self.next_seed = 1000
        self.counter = 0
        self.during_generation = None   # callable(fake) run before this call's files are written
        self.extra_saves: list[str] = []  # suffixes saved next to each image (e.g. "-before-highres-fix")

    # -- transport --

    def transport(self):
        return httpx.MockTransport(self.handle)

    def calls(self, method: str | None = None, path: str | None = None) -> list:
        return [entry for entry in self.requests
                if (method is None or entry[0] == method) and (path is None or entry[1] == path)]

    def handle(self, request):
        path = request.url.path
        method = request.method
        body = None
        if request.content:
            try:
                body = json.loads(request.content)
            except ValueError:
                body = request.content
        self.requests.append((method, path, body))
        self.queries.append((path, dict(request.url.params)))
        self.authorizations.append(request.headers.get("authorization"))
        route = (method, path)
        if self.fail_generation and route in (("POST", "/sdapi/v1/txt2img"), ("POST", "/sdapi/v1/img2img")):
            status, text = self.fail_generation
            return httpx.Response(status, text=text)
        if route == ("GET", "/sdapi/v1/options"):
            return httpx.Response(200, json=self.options)
        if route == ("POST", "/sdapi/v1/options"):
            self.options.update(body or {})
            return httpx.Response(200, **NULL_JSON)  # FastAPI's answer for a handler returning None
        if route == ("GET", "/sdapi/v1/sd-models"):
            return httpx.Response(200, json=self.sd_models)
        if route == ("GET", "/sdapi/v1/sd-modules"):
            return httpx.Response(200, json=self.sd_modules)
        if route == ("GET", "/sdapi/v1/loras"):
            return httpx.Response(200, json=self.loras)
        if route == ("GET", "/sdapi/v1/samplers"):
            return httpx.Response(200, json=[{"name": "ER SDE"}, {"name": "Euler"}])
        if route == ("GET", "/sdapi/v1/schedulers"):
            return httpx.Response(200, json=[{"name": "beta"}])
        if route == ("GET", "/sdapi/v1/progress"):
            return httpx.Response(200, json={"progress": 0.5, "eta_relative": 3.0,
                                             "state": {"job": "scripts_txt2img", "sampling_step": 5, "sampling_steps": 10},
                                             "current_image": None})
        if route == ("GET", "/openapi.json"):
            return httpx.Response(200, json={"paths": {p: {} for p in (
                "/sdapi/v1/txt2img", "/sdapi/v1/img2img", "/sdapi/v1/loras", "/sdapi/v1/options",
                "/sdapi/v1/cmd-flags", "/sam3-notebook/x")}})
        if route == ("POST", "/sdapi/v1/interrupt"):
            return httpx.Response(200, json={})
        if route in (("POST", "/sdapi/v1/skip"), ("POST", "/sdapi/v1/refresh-checkpoints")):
            return httpx.Response(200, **NULL_JSON)
        if route == ("POST", "/sdapi/v1/txt2img"):
            return httpx.Response(200, json=self.generate(body or {}, "txt2img"))
        if route == ("POST", "/sdapi/v1/img2img"):
            return httpx.Response(200, json=self.generate(body or {}, "img2img"))
        return httpx.Response(404, json={"detail": "Not Found"})

    # -- generation --

    def outdir(self, mode: str) -> Path:
        key = "outdir_img2img_samples" if mode == "img2img" else "outdir_txt2img_samples"
        value = self.options[key]
        folder = Path(value) if os.path.isabs(value) else self.forge / value
        return folder / date.today().isoformat()

    def encode(self, text: str | None) -> bytes:
        if self.samples_format == "jpg":
            return jpeg_bytes(text)
        if self.samples_format == "webp":
            return webp_bytes(text)
        return png_bytes(text)

    def write_output(self, folder: Path, seed: int, text: str, *, suffix: str = "") -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        self.counter += 1
        stem = f"{self.counter:05}-{seed}{suffix}"
        path = folder / f"{stem}.{self.samples_format}"
        path.write_bytes(self.encode(text if self.embed else None))
        if self.save_txt:
            (folder / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
        return path

    def generate(self, payload: dict, mode: str) -> dict:
        prompt = payload.get("prompt", "")
        negative = payload.get("negative_prompt", "")
        batch = int(payload.get("batch_size", 1))
        seed = int(payload.get("seed", -1))
        if seed == -1:
            seed = self.next_seed
            self.next_seed += 100
        seeds = [seed + offset for offset in range(batch)]
        size = f"{payload.get('width', 1024)}x{payload.get('height', 1024)}"
        texts = [infotext(prompt, s, negative=negative, steps=int(payload.get("steps", 10)), size=size) for s in seeds]
        if callable(self.during_generation):
            self.during_generation(self)
        if self.save_files:
            for s, text in zip(seeds, texts):
                for suffix in self.extra_saves:
                    self.write_output(self.outdir(mode), s, text, suffix=suffix)
                self.write_output(self.outdir(mode), s, text)
        images = [base64.b64encode(self.encode(text if self.embed else None)).decode("ascii") for text in texts]
        infotexts = list(texts)
        first = 0
        if batch > 1 and self.return_grid:
            grid_text = texts[0]
            images.insert(0, base64.b64encode(self.encode(grid_text if self.embed else None)).decode("ascii"))
            infotexts.insert(0, grid_text)
            first = 1
        info = {
            "prompt": prompt,
            "all_prompts": [prompt] * batch,
            "negative_prompt": negative,
            "seed": seeds[0],
            "all_seeds": seeds,
            "width": payload.get("width"),
            "height": payload.get("height"),
            "steps": payload.get("steps"),
            "sampler_name": payload.get("sampler_name"),
            "index_of_first_image": first,
            "infotexts": infotexts,
            "sd_model_name": "animeMix_v10",
            "job_timestamp": "20261003120000",
        }
        return {"images": images, "parameters": payload, "info": json.dumps(info)}


def make_service(base: Path, *, settings: dict | None = None, options: dict | None = None,
                 env: dict | None = None, download_transport=None, program: bool = True):
    """A Service wired to a FakeForge inside a throw-away Forge tree."""
    from sam_extra_mcp import forge_paths
    from sam_extra_mcp.forgeneo.config import Config
    from sam_extra_mcp.service import Service

    tree = make_forge_tree(base, settings=settings, program=program)
    fake = FakeForge(tree["forge"], options)
    environment = {"FORGE_URL": "http://127.0.0.1:7860", "SAM_EXTRA_MCP_CACHE_DIR": str(Path(base) / "cache")}
    environment.update(env or {})
    service = Service(
        Config.from_env(environment),
        forge_paths.discover(environment, module_file=tree["module_file"]),
        transport=fake.transport(),
        download_transport=download_transport,
    )
    return service, fake, tree
