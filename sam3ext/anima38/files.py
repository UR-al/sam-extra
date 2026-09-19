from __future__ import annotations

import os
from pathlib import Path

from safetensors import SafetensorError, safe_open

ARCHITECTURE = "anima_progressive_qwen35_cross_adapter_v1"
V2_ARCHITECTURE = "anima_qwen35_quality_anchored_semantic_connector_v2"
BUNDLE_ARCHITECTURE = "anima_3_8b_semantic_connector_v2_bundle"
BUNDLE_FORMAT = "1"
CONNECTOR_PREFIX = "net.anima_v2_connector."


def extension_root() -> Path:
    """이 파일은 <extension>/sam3ext/anima38/files.py 에 있다."""
    return Path(__file__).resolve().parents[2]


def forge_root() -> Path:
    """Forge 루트 — <forge>/extensions/<extension>/... 에서 두 단계 위."""
    return extension_root().parents[1]


def text_encoder_roots() -> list[Path]:
    roots = [forge_root() / "models" / "text_encoder"]
    try:
        # --data-dir 로 띄운 Forge(앱 관리형)는 models 가 데이터 폴더 아래에 있다.
        from modules import paths

        roots.append(Path(paths.models_path) / "text_encoder")
    except Exception:
        pass
    try:
        from modules_forge.main_entry import module_list

        roots.extend(Path(path).parent for path in module_list.values())
    except Exception:
        pass
    return list(dict.fromkeys(path.resolve() for path in roots if path.is_dir()))


QWEN35_MARKERS = ("qwen35_4b", "qwen3.5-4b", "qwen3_5_4b")


def qwen35_models() -> dict[str, str]:
    found: dict[str, str] = {}
    for root in text_encoder_roots():
        for path in root.rglob("*.safetensors"):
            if any(marker in path.name.lower() for marker in QWEN35_MARKERS):
                found.setdefault(path.name, str(path))
    return dict(sorted(found.items()))


def is_unused_qwen35_module(path: str | os.PathLike) -> bool:
    """VAE/Text Encoder 목록에 들어 있지만 Forge 로더가 읽기만 하고 버리는 Qwen3.5-4B 파일인가.

    Forge 의 replace_state_dict 는 텍스트 인코더를 ``model.`` 접두어 키로 알아보는데 이 파일엔 그런 키가
    없다. 3.8B 는 런타임이 파일을 따로 찾아 쓰므로 목록에 있어도 모델을 불러올 때마다 4.8 GB 를 읽는
    낭비일 뿐이다. ``model.`` 키가 있는 형식(Forge 가 나중에 지원할 수 있는)은 건드리지 않는다.
    """
    name = Path(path).name.lower()
    if not name.endswith(".safetensors") or not any(marker in name for marker in QWEN35_MARKERS):
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            return not any(key.startswith("model.") for key in checkpoint.keys())
    except (OSError, ValueError, SafetensorError):
        return False


def bundle_metadata(path: str | os.PathLike) -> dict[str, str] | None:
    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
    except (OSError, ValueError, SafetensorError):
        return None
    if (
        metadata.get("architecture") != BUNDLE_ARCHITECTURE
        or metadata.get("anima_v2_bundle_format") != BUNDLE_FORMAT
    ):
        return None
    return metadata


def adapters() -> dict[str, str]:
    found: dict[str, str] = {}
    for root in text_encoder_roots():
        for path in root.rglob("*.safetensors"):
            try:
                with safe_open(path, framework="pt", device="cpu") as checkpoint:
                    metadata = checkpoint.metadata() or {}
                    keys = set(checkpoint.keys())
                if metadata.get("architecture") != ARCHITECTURE:
                    continue
                if any(key.startswith(("timestep_gates.", "anchor_deviation")) for key in keys):
                    continue
            except (OSError, ValueError, SafetensorError):
                continue
            label = os.path.relpath(path, root).replace("\\", "/")
            found.setdefault(label, str(path))
    return dict(sorted(found.items()))


def tokenizer_dir() -> Path:
    bundled = extension_root() / "assets" / "qwen35_tokenizer"
    candidates = [bundled]
    candidates.extend(root / "qwen35_tokenizer" for root in text_encoder_roots())
    for candidate in candidates:
        if (candidate / "tokenizer.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Qwen3.5 tokenizer files are missing from the extension's "
        "assets/qwen35_tokenizer directory. Reinstall the extension."
    )
