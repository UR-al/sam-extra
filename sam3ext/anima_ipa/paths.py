"""가중치 경로와 받기. ``sam3ext/tipo/runtime.py`` 의 관용구를 그대로 따른다."""
from __future__ import annotations

from pathlib import Path

ADAPTER_REPO = "LuciferTC/Anima-IP-Adapter"
ADAPTER_REVISION = "99e9c351c9f00bdd188b5added0066a80d3b1de6"
ADAPTER_FILE = "ip_adapter-Character_Reference-10.safetensors"
ADAPTER_BYTES = 503_229_576

ENCODER_REPO = "google/siglip2-base-patch16-512"
ENCODER_REVISION = "a89f5c5093f902bf39d3cd4d81d2c09867f0724b"
ENCODER_SUBDIR = "siglip2-base-patch16-512"
# 이름 → 기대 크기(바이트). 크기가 다르면 받다 만 파일로 보고 다시 받는다.
ENCODER_FILES: dict[str, int] = {
    "config.json": 276,
    "preprocessor_config.json": 394,
    "model.safetensors": 1_503_344_520,
}

TOTAL_BYTES_LABEL = "2.0 GB"


def forge_models_dir() -> Path:
    try:
        from modules import paths

        return Path(paths.models_path)
    except Exception:
        # <forge>/extensions/<ext>/sam3ext/anima_ipa/paths.py
        return Path(__file__).resolve().parents[4] / "models"


def ipa_dir(models_dir=None) -> Path:
    return Path(models_dir or forge_models_dir()) / "anima_ipa"


def adapter_path(models_dir=None) -> Path:
    return ipa_dir(models_dir) / ADAPTER_FILE


def encoder_dir(models_dir=None) -> Path:
    return ipa_dir(models_dir) / ENCODER_SUBDIR


def _incomplete(path: Path, expected: int) -> bool:
    try:
        return not path.is_file() or path.stat().st_size != expected
    except OSError:
        return True


def missing_files(models_dir=None) -> list[str]:
    """없거나 크기가 다른 파일의 이름. 빈 리스트면 바로 쓸 수 있다."""
    missing: list[str] = []
    if _incomplete(adapter_path(models_dir), ADAPTER_BYTES):
        missing.append(ADAPTER_FILE)
    directory = encoder_dir(models_dir)
    for name, size in ENCODER_FILES.items():
        if _incomplete(directory / name, size):
            missing.append(name)
    return missing


def download(models_dir=None, downloader=None) -> None:
    """없는 파일만 받는다. ``downloader`` 는 테스트에서 주입한다."""
    if downloader is None:
        from huggingface_hub import hf_hub_download as downloader

    missing = set(missing_files(models_dir))
    if ADAPTER_FILE in missing:
        target = ipa_dir(models_dir)
        target.mkdir(parents=True, exist_ok=True)
        downloader(
            repo_id=ADAPTER_REPO,
            filename=ADAPTER_FILE,
            revision=ADAPTER_REVISION,
            local_dir=str(target),
        )
    encoder_missing = [name for name in ENCODER_FILES if name in missing]
    if encoder_missing:
        target = encoder_dir(models_dir)
        target.mkdir(parents=True, exist_ok=True)
        for name in encoder_missing:
            downloader(
                repo_id=ENCODER_REPO,
                filename=name,
                revision=ENCODER_REVISION,
                local_dir=str(target),
            )
