"""Forge 모델 로더에서 쓰이지 않는 Qwen3.5-4B 텍스트 인코더를 빼는 훅.

3.8B 의 Qwen3.5 경로는 런타임이 ``models/text_encoder`` 에서 파일을 직접 찾는다 — VAE/Text Encoder
목록에 넣을 필요가 없다. 그런데 넣어 두면 Forge 는 체크포인트를 불러올 때마다(XYZ 의 체크포인트 축이면
칸마다) 4.8 GB 를 통째로 읽고 버린다. 1.0/2.9B/3.8B 를 같은 모듈 목록으로 비교할 수 있게 여기서 뺀다.
"""
from __future__ import annotations

import logging

from .files import is_unused_qwen35_module

logger = logging.getLogger(__name__)


def _log(message: str) -> None:
    """콘솔 코드페이지(cp949 등)에 없는 문자(일본어 경로 등)가 모델 로드를 죽이지 않게."""
    text = f"[Anima38] {message}"
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"))


def install(loader=None) -> None:
    if loader is None:
        from backend import loader
    original = loader.split_state_dict
    if getattr(original, "_anima38_filter", False):
        return

    def split_state_dict(path, additional_state_dicts=None):
        if isinstance(additional_state_dicts, list):
            try:
                skipped = [m for m in additional_state_dicts if is_unused_qwen35_module(m)]
            except Exception:   # 판별이 실패하면 Forge 순정 동작 그대로
                skipped = []
            if skipped:
                _log(
                    "VAE/Text Encoder 목록의 Qwen3.5 파일은 Forge 가 쓰지 않아 건너뜁니다 "
                    "(3.8B 는 자동으로 찾아 씁니다): " + ", ".join(str(m) for m in skipped)
                )
                additional_state_dicts = [m for m in additional_state_dicts if m not in skipped]
        return original(path, additional_state_dicts=additional_state_dicts)

    split_state_dict._anima38_filter = True
    loader.split_state_dict = split_state_dict
