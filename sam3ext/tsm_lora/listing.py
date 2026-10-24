"""LoRA 폴더에서 TSM 파일 찾기 — 아코디언의 파일 목록.

Forge 는 LoRA 목록을 만들 때 safetensors 메타데이터를 이미 읽어 캐시해 둔다(``networks.available_networks`` 의
``NetworkOnDisk.metadata``). 메타데이터가 TSM 이라고 하면(``ss_saved_format=tsm-native`` 또는 algo ``tsm``) 바로
목록에 넣고, 메타데이터가 비었거나 algo 를 적지 않은 LyCORIS 파일만 헤더의 키 이름을 본다(텐서는 읽지 않는다).
키 확인 결과는 (경로, mtime, 크기) 로 캐시한다.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Iterable, Mapping

from .fileformat import TsmFormatError, is_tsm_state_dict, metadata_may_hide_tsm, metadata_says_tsm, read_header

_KEY_CACHE: dict[tuple[str, int, int], bool] = {}
_LOCK = threading.Lock()


def _stat_key(path: str) -> tuple[str, int, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return os.path.normcase(os.path.abspath(path)), int(stat.st_mtime_ns), int(stat.st_size)


def header_has_tsm_keys(path: str) -> bool:
    key = _stat_key(path)
    if key is None:
        return False
    with _LOCK:
        hit = _KEY_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        _meta, header = read_header(path)
        found = is_tsm_state_dict(header.keys())
    except TsmFormatError:
        found = False
    with _LOCK:
        _KEY_CACHE[key] = found
    return found


def is_tsm_candidate(path: str, metadata: Mapping[str, Any] | None) -> bool:
    if not str(path).lower().endswith(".safetensors"):
        return False
    if metadata_says_tsm(metadata):
        return True
    if metadata and not metadata_may_hide_tsm(metadata):
        return False
    return header_has_tsm_keys(path)


def find_tsm_files(entries: Iterable[tuple[str, str, Mapping[str, Any] | None]]) -> dict[str, str]:
    """(Forge 이름, 파일 경로, 메타데이터) 들 → {이름: 경로} (이름순). 메타데이터가 None 이면 헤더에서 읽는다."""
    found: dict[str, str] = {}
    for name, path, metadata in entries:
        if not path or not str(path).lower().endswith(".safetensors"):
            continue
        meta = metadata
        if meta is None:
            try:
                meta, _ = read_header(path)
            except TsmFormatError:
                continue
        if is_tsm_candidate(path, meta):
            found[str(name)] = str(path)
    return dict(sorted(found.items(), key=lambda item: item[0].lower()))


def walk_safetensors(directories: Iterable[str]) -> list[tuple[str, str, None]]:
    """Forge 목록을 못 쓸 때의 대안: 폴더를 돌며 (상대 경로 이름, 경로, None)."""
    rows: list[tuple[str, str, None]] = []
    seen: set[str] = set()
    for directory in directories:
        if not directory or not os.path.isdir(directory):
            continue
        for root, _dirs, files in os.walk(directory, followlinks=True):
            for filename in files:
                if not filename.lower().endswith(".safetensors"):
                    continue
                path = os.path.join(root, filename)
                norm = os.path.normcase(os.path.abspath(path))
                if norm in seen:
                    continue
                seen.add(norm)
                name = os.path.splitext(os.path.relpath(path, directory))[0].replace("\\", "/")
                rows.append((name, path, None))
    return rows
