"""Anima VAE DeGrid — 모델 파일 찾기(NAFNet 만)·불러오기.

찾는 곳: ``models/ESRGAN``(civitai·HF 안내대로 넣는 곳)과 ``models/DeGrid``(있으면). 하위 폴더는 보지 않는다.
ESRGAN 폴더에는 일반 업스케일러가 섞여 있으므로 **state dict 키가 NAFNet 인 파일만** 고른다 — spandrel NAFNet
아키텍처의 감지 키(``spandrel/architectures/NAFNet/__init__.py`` 15-34줄)를 모두 가진 파일. safetensors 는 헤더만 읽고
(텐서는 읽지 않음), .pth/.pt 는 **torch.load 를 부르지 않고** zip(새 형식)의 ``data.pkl`` 만 풀어 키 이름을 본다(텐서·저장소는
만들지 않는 대역으로 바꾸고, ``collections.OrderedDict`` 밖의 전역은 부르지 않음). zip 이 아닌 옛 형식 pickle(예: 4x-UltraSharp.pth)은
DeGrid 가 아니므로(DeGrid 는 safetensors·새 형식 pth 로 배포) 열지 않고 건너뛴다. 판정은 (경로, 크기, 수정 시각)으로 캐시한다.

Forge 는 ``torch.load``·``safetensors.torch.load_file`` 을 감싸서(``modules_forge/patch_basic.py`` ``build_loaded``) 실패하면
인자 중 **str 파일 경로를 모두 ``<파일>.corrupted`` 로 바꿔 버린다**(원래 로더는 ``torch.load_origin``·``load_file_origin``).
그래서 가중치는 ``torch_load``·``safetensors_load`` 로만 읽는다 — 원래 로더를 쓰고, 어느 로더에도 str 인자를 넘기지 않는다.

목록은 safetensors metadata 의 ``modelspec.version`` 이 높은 파일이 먼저다(v1.1 은 "1.1" 을 적어 두었다. 버전을 적지 않은
파일은 적은 파일 뒤, 같으면 폴더·이름 순). 모델을 비워 둔 API 호출(auto)과 드롭다운 기본값은 맨 앞 파일이다.

키만으로는 DeGrid(잔차를 냄)와 일반 복원 NAFNet(이미지를 냄 — SIDD width 32 노이즈 제거는 구성까지 같다)을 가를 수
없다. 그런 파일도 목록에 나오지만, 런타임이 출력으로 거른다(``vae_degrid.output_looks_like_image`` →
``NotResidualModelError``).

불러오기는 Forge 의 업스케일러와 같은 spandrel ``ModelLoader`` 를 쓴다(Forge venv 에 이미 있음). NAFNet 이 아니거나
배율·채널이 맞지 않으면 ValueError. 모델은 CPU fp32 로 한 번에 하나만 캐시한다(약 117 MB).
"""
from __future__ import annotations

import collections
import io
import json
import logging
import os
import pickle
import re
import struct
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path

MODEL_FOLDERS = ("ESRGAN", "DeGrid")
MODEL_EXTENSIONS = (".safetensors", ".pth", ".pt")
NONE_NAME = "None"

# spandrel NAFNetArch detect = KeyCondition.has_all(...)
NAFNET_KEYS = (
    "intro.weight",
    "ending.weight",
    "ups.0.0.weight",
    "downs.0.weight",
    "middle_blks.0.beta",
    "middle_blks.0.gamma",
    "middle_blks.0.conv1.weight",
    "middle_blks.0.conv2.weight",
    "middle_blks.0.conv3.weight",
    "middle_blks.0.sca.1.weight",
    "middle_blks.0.conv4.weight",
    "middle_blks.0.conv5.weight",
    "middle_blks.0.norm1.weight",
    "middle_blks.0.norm2.weight",
    "encoders.0.0.beta",
    "encoders.0.0.gamma",
    "decoders.0.0.beta",
    "decoders.0.0.gamma",
)

# spandrel canonicalize_state_dict 가 벗기는 감싸개(basicsr 체크포인트는 {"params": sd})와 공통 접두사.
_WRAPPER_KEYS = ("model_state_dict", "state_dict", "params_ema", "params-ema", "params", "model", "net")
_PREFIXES = ("module.", "netG.", "net_g.")

# zip .pth 의 data.pkl 은 키·모양·저장소 번호만 담아 작다(업스케일러 67 MB 파일에서 147 KB). 이보다 크면 읽지 않는다.
_PICKLE_READ_LIMIT = 64 << 20
_ZIP_MAGIC = b"PK\x03\x04"   # torch.serialization._is_zipfile 과 같은 검사(앞 4바이트)

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEntry:
    name: str
    path: str
    version: tuple = ()   # safetensors metadata 의 modelspec.version ("1.1" → (1, 1)), 없으면 ()


def forge_models_dir() -> Path:
    try:
        from modules import paths

        return Path(paths.models_path)
    except Exception:
        # <forge>/extensions/<ext>/sam3ext/vae_degrid_models.py
        return Path(__file__).resolve().parents[3] / "models"


def _esrgan_override() -> Path | None:
    """Forge ``--esrgan-models-path``(기본 models/ESRGAN)."""
    try:
        from modules import shared

        value = getattr(shared.cmd_opts, "esrgan_models_path", None)
    except Exception:
        return None
    return Path(value) if value else None


def model_dirs(models_dir=None) -> list[Path]:
    """찾을 폴더 — ``models/ESRGAN``(Forge 에서 ``--esrgan-models-path`` 를 바꿨으면 그 폴더)과 ``models/DeGrid``."""
    root = Path(models_dir) if models_dir is not None else forge_models_dir()
    dirs = [root / folder for folder in MODEL_FOLDERS]
    if models_dir is None:
        esrgan = _esrgan_override()
        if esrgan is not None:
            dirs[0] = esrgan
    return dirs


# ---------------------------------------------------------------------------
# 키 읽기
# ---------------------------------------------------------------------------


def read_safetensors_header(path) -> dict | None:
    """safetensors 헤더(8바이트 길이 + JSON)만 읽는다 — 텐서는 읽지 않는다. 읽을 수 없으면 None."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
            if len(head) != 8:
                return None
            (length,) = struct.unpack("<Q", head)
            if length <= 0 or length > 100 * 1024 * 1024:
                return None
            header = json.loads(handle.read(length).decode("utf-8"))
    except Exception:
        return None
    return header if isinstance(header, dict) else None


def read_safetensors_keys(path) -> list[str] | None:
    header = read_safetensors_header(path)
    if header is None:
        return None
    return [key for key in header if key != "__metadata__"]


def parse_version(text) -> tuple:
    """'1.1' → (1, 1). 숫자가 없으면 ()."""
    return tuple(int(part) for part in re.findall(r"\d+", str(text or ""))[:4])


def declared_version(path) -> tuple:
    """safetensors metadata 의 ``modelspec.version``. .pth·metadata 없음은 ()."""
    if Path(path).suffix.lower() != ".safetensors":
        return ()
    header = read_safetensors_header(path) or {}
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        return ()
    return parse_version(metadata.get("modelspec.version"))


def _unwrap(state) -> dict | None:
    if not isinstance(state, dict):
        return None
    for key in _WRAPPER_KEYS:
        inner = state.get(key)
        if isinstance(inner, dict):
            state = inner
            break
    if len(state) == 1:
        single = next(iter(state.values()))
        if isinstance(single, dict):
            state = single
    return state


def is_torch_zip(path) -> bool:
    """torch 새 형식(zip) 파일인지 — 앞 4바이트(``torch.serialization._is_zipfile`` 과 같음)와 zip 끝 레코드만 본다."""
    try:
        with open(path, "rb") as handle:
            if handle.read(4) != _ZIP_MAGIC:
                return False
        return zipfile.is_zipfile(path)
    except OSError:
        return False


class _Opaque:
    """data.pkl 안의 텐서·저장소·그 밖의 전역 대신 들어가는 대역. 받은 인자를 무시하고 아무 코드도 부르지 않는다."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def _encode_text(text="", encoding="latin1"):
    """pickle 프로토콜 2 가 bytes 를 적는 ``_codecs.encode(str, 'latin1')`` — str 만 받는다."""
    if isinstance(text, str) and str(encoding).lower().replace("_", "-") in ("latin1", "latin-1", "utf-8", "utf8", "ascii"):
        return text.encode(encoding)
    return _Opaque()


_PICKLE_GLOBALS = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("_codecs", "encode"): _encode_text,
}


class _KeysOnlyUnpickler(pickle.Unpickler):
    """state dict 의 모양(dict·키 이름)만 되살린다 — 텐서(persistent id)와 OrderedDict 밖의 전역은 ``_Opaque``."""

    def find_class(self, module, name):
        return _PICKLE_GLOBALS.get((module, name), _Opaque)

    def persistent_load(self, pid):
        return _Opaque()


def _zip_data_pkl(archive: zipfile.ZipFile):
    for info in archive.infolist():
        parts = info.filename.replace("\\", "/").split("/")
        if parts[-1] == "data.pkl" and len(parts) <= 2:
            return info
    return None


def read_torch_state_skeleton(path):
    """zip .pth/.pt 의 ``data.pkl`` 만 풀어 state dict 모양을 돌려준다(텐서 자리는 ``_Opaque``). torch 를 쓰지 않는다.
    zip 이 아니거나(옛 형식 pickle) 풀 수 없으면 None."""
    if not is_torch_zip(path):
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            info = _zip_data_pkl(archive)
            if info is None or info.file_size > _PICKLE_READ_LIMIT:
                return None
            data = archive.read(info)
        return _KeysOnlyUnpickler(io.BytesIO(data), encoding="utf-8", errors="replace").load()
    except Exception:
        return None


def read_torch_keys(path) -> list[str] | None:
    """.pth/.pt 의 텐서 이름(감싸개를 벗긴 뒤) — ``data.pkl`` 만 읽는다(``torch.load`` 를 부르지 않음).
    옛 형식(zip 이 아닌 pickle)은 열지 않고 None(디버그 로그 한 줄 — 판정 캐시 덕에 파일마다 한 번)."""
    if not is_torch_zip(path):
        _LOG.debug("VAE DeGrid: skipped %s (legacy torch pickle, not a DeGrid model)", Path(path).name)
        return None
    state = _unwrap(read_torch_state_skeleton(path))
    if state is None:
        return None
    return [str(key) for key in state]


def strip_prefixes(keys) -> list[str]:
    """모든 키가 같은 접두사(module. / netG. / net_g.)로 시작하면 벗긴다(spandrel remove_common_prefix 와 같은 뜻)."""
    keys = list(keys)
    for prefix in _PREFIXES:
        if keys and all(key.startswith(prefix) for key in keys):
            keys = [key[len(prefix):] for key in keys]
    return keys


def is_nafnet_keys(keys) -> bool:
    present = set(strip_prefixes(keys or []))
    return all(key in present for key in NAFNET_KEYS)


def read_keys(path) -> list[str] | None:
    suffix = Path(path).suffix.lower()
    if suffix == ".safetensors":
        return read_safetensors_keys(path)
    if suffix in (".pth", ".pt"):
        return read_torch_keys(path)
    return None


# ---------------------------------------------------------------------------
# 찾기
# ---------------------------------------------------------------------------

_CLASSIFY_CACHE: dict[tuple[str, int, int], tuple[bool, tuple]] = {}
_CACHE_LOCK = threading.Lock()


def _classify(path) -> tuple[bool, tuple]:
    """(NAFNet 인지, 선언된 버전) — (경로, 크기, 수정 시각)으로 캐시."""
    try:
        stat = os.stat(path)
    except OSError:
        return False, ()
    key = (os.path.abspath(str(path)), int(stat.st_size), int(stat.st_mtime_ns))
    with _CACHE_LOCK:
        cached = _CLASSIFY_CACHE.get(key)
    if cached is not None:
        return cached
    verdict = is_nafnet_keys(read_keys(path))
    result = (verdict, declared_version(path) if verdict else ())
    with _CACHE_LOCK:
        _CLASSIFY_CACHE[key] = result
    return result


def is_nafnet_file(path) -> bool:
    """NAFNet 가중치 파일인지(키만 본다)."""
    return _classify(path)[0]


def discover(models_dir=None, dirs=None) -> list[ModelEntry]:
    """NAFNet 파일 목록 — 선언된 버전이 높은 것 먼저. 이름은 파일 이름(확장자 뺌)이고, 폴더끼리 겹치면 ``폴더/이름``."""
    found: list[tuple[str, Path, tuple]] = []
    for folder in (dirs if dirs is not None else model_dirs(models_dir)):
        folder = Path(folder)
        try:
            names = sorted(os.listdir(folder), key=str.lower)
        except OSError:
            continue
        for filename in names:
            path = folder / filename
            if path.suffix.lower() not in MODEL_EXTENSIONS or not path.is_file():
                continue
            verdict, version = _classify(path)
            if verdict:
                found.append((folder.name, path, version))
    found.sort(key=lambda item: item[2], reverse=True)   # 안정 정렬 — 버전이 같으면 폴더·이름 순 그대로
    stems: dict[str, int] = {}
    for _folder, path, _version in found:
        stems[path.stem.lower()] = stems.get(path.stem.lower(), 0) + 1
    entries = []
    for folder_name, path, version in found:
        name = path.stem if stems[path.stem.lower()] == 1 else f"{folder_name}/{path.stem}"
        entries.append(ModelEntry(name=name, path=str(path), version=version))
    return entries


def resolve(name, entries) -> ModelEntry | None:
    """이름 → 항목. 비었거나 'None'·'auto' 면 첫 항목. 정확한 이름 → 대소문자 무시 → 파일 이름(확장자 포함) 순."""
    entries = list(entries or [])
    text = str(name or "").strip()
    if not entries:
        return None
    if not text or text.lower() in ("none", "auto"):
        return entries[0]
    for entry in entries:
        if entry.name == text:
            return entry
    lowered = text.lower()
    for entry in entries:
        if entry.name.lower() == lowered:
            return entry
    for entry in entries:
        path = Path(entry.path)
        if lowered in (path.name.lower(), path.stem.lower()):
            return entry
    return None


# ---------------------------------------------------------------------------
# 불러오기
# ---------------------------------------------------------------------------


def unpatched_loader(module, name: str):
    """Forge 가 감싸기 전의 로더 — ``build_loaded(module, name)`` 이 ``<name>_origin`` 에 둔다. Forge 밖이면 그 로더 그대로."""
    original = getattr(module, f"{name}_origin", None)
    return original if callable(original) else getattr(module, name)


def torch_load(path):
    """``.pth``/``.pt`` 를 CPU 로(weights_only). Forge 가 감싸기 전 ``torch.load`` 를 쓰고, 경로는 ``Path``·
    ``map_location`` 은 ``torch.device`` 로 넘긴다 — 감싼 로더는 실패하면 str 인자인 파일 이름을 바꾼다."""
    import torch

    loader = unpatched_loader(torch, "load")
    return loader(Path(path), map_location=torch.device("cpu"), weights_only=True)


def safetensors_load(path):
    """safetensors 를 CPU 로. ``torch_load`` 와 같은 이유로 감싸기 전 로더에 ``Path`` 만 넘긴다(device 는 기본값 cpu)."""
    import safetensors.torch

    loader = unpatched_loader(safetensors.torch, "load_file")
    return loader(Path(path))


def _read_state_dict(path):
    if Path(path).suffix.lower() == ".safetensors":
        return safetensors_load(path)
    return torch_load(path)


def load_nafnet(path):
    """spandrel 로 NAFNet 을 불러 CPU fp32 ``nn.Module`` 을 돌려준다(eval, requires_grad 끔)."""
    import spandrel

    state = _read_state_dict(path)
    descriptor = spandrel.ModelLoader(device="cpu").load_from_state_dict(state)
    arch = str(getattr(getattr(descriptor, "architecture", None), "id", "") or "")
    if arch.lower() != "nafnet":
        raise ValueError(f"{Path(path).name} is not a NAFNet model (spandrel: {arch or type(descriptor).__name__})")
    if getattr(descriptor, "scale", 1) != 1 or getattr(descriptor, "input_channels", 3) != 3 \
            or getattr(descriptor, "output_channels", 3) != 3:
        raise ValueError(f"{Path(path).name}: expected a 1x RGB NAFNet (scale={descriptor.scale})")
    model = descriptor.model
    model.eval()
    model.requires_grad_(False)
    return model
