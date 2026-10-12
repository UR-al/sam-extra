"""'모델 내리기' 버튼이 부르는 해제 훅 목록 — sam-extra 의 기능이 메모리에 쥔 모델·텐서를 놓는 함수들(순수 로직, torch·Forge 없음).

버튼(sam3ext/unload_models.py)은 Forge 의 모델을 내린 뒤 여기 등록된 훅을 **등록 순서대로 하나씩** 부른다. 훅 하나가 던져도
나머지는 그대로 불리고, 던진 것은 결과 글에 이름과 함께 남는다. 훅은 인자 없이 불리며 무언가를 놓았으면 True 를 돌려준다
(결과 글의 괄호에 그 ``label`` 이 나온다). 쥔 것이 없었으면 False(또는 None).

등록 방법은 두 가지다:

* ``BUILTIN_HOOKS`` — ``sam3ext`` 패키지 모듈의 함수. 모듈 이름과 함수 이름만 적고, 누를 때 ``sys.modules`` 에서 찾는다:
  아직 import 되지 않은 모듈은 아무것도 쥐고 있지 않으므로 건너뛴다(내리려고 무거운 모듈을 import 하지 않는다).
  Reload UI 는 ``sam3ext`` 모듈을 다시 읽지 않으므로(scripts/!sam3.py 머리의 설명) 확장을 업데이트한 뒤 Forge 를 다시
  시작하지 않아 함수가 없는 옛 모듈이면, 그 훅만 "Forge 를 다시 시작하세요" 오류로 끝난다.
* ``register(key, label, release)`` — 그 밖의 곳(``scripts/*.py`` 처럼 import 이름이 없는 스크립트 모듈, 다른 확장 기능).
  같은 ``key`` 로 다시 등록하면 바꿔 끼운다(Reload UI 가 스크립트를 다시 실행할 때).

새 기능이 모델을 쥐면 그 모듈에 ``release`` 함수를 두고 ``BUILTIN_HOOKS`` 에 한 줄을 더하거나(얼굴 검출기처럼) ``register``
로 등록한다. 같은 ``label`` 을 여러 훅이 써도 된다 — 결과 글에는 한 번만 나온다.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable

# (key, 결과 글의 이름, 모듈, 함수) — 이 순서대로 부른다. 함수는 인자 없이 불리고 놓은 것이 있으면 True.
BUILTIN_HOOKS: tuple[tuple[str, str, str, str], ...] = (
    # SAM3 번들(약 3.5 GB) — 'Unload after' 가 CPU RAM 에 보관한 사본까지 버린다(다음 검출은 파일에서 다시 빌드).
    ("sam3", "SAM3", "sam3ext.core", "release_sam3"),
    # SAM3 '얼굴 안에서만 찾기'의 YOLO 얼굴 검출기 — 캐시하지 않으므로 보통 없음, 놓은 뒤에도 살아 있는 검출기만 뗀다.
    ("face_detect", "얼굴 검출기", "sam3ext.face_detect", "release_face_detector"),
    # Anima 3.8B: Qwen3.5 4B · v1 어댑터 · v2 커넥터와 그 Forge 패처 · run 의 장치 사본 · 특징/TE 줄 캐시.
    ("anima38", "Anima 3.8B", "sam3ext.anima38.runtime", "unload_shared_runtime"),
    # 캐릭터 레퍼런스의 IP-Adapter 가중치와 SigLIP2 인코더(쉴 때 CPU RAM).
    ("anima_ipa", "IP-Adapter", "sam3ext.anima_ipa.runtime", "unload_shared_runtime"),
    ("vae_degrid", "DeGrid", "sam3ext.vae_degrid_runtime", "unload_shared_runtime"),
    ("tipo", "TIPO", "sam3ext.tipo.runtime", "unload_shared_runtime"),
    # Anima Guidance Suite 의 Modulation Guidance(보조 CLIP-L 텍스트 인코더 · 어댑터 · pooled 캐시).
    ("modulation", "Modulation CLIP-L", "sam3ext.guidance.modulation", "clear_modulation_caches"),
    # Colorcraft 기저 벡터(장치 사본 포함)와 앵커 latent.
    ("colorcraft", "Colorcraft", "sam3ext.colorcraft.hook", "release_caches"),
    # 생성마다 다시 만드는 작은 장치 행렬들(DCT 기저 · Butterworth 마스크).
    ("speed_dct", "샘플링 캐시", "sam3ext.speed.spectral", "clear_basis_cache"),
    ("hiflow_mask", "샘플링 캐시", "sam3ext.guidance.hiflow", "clear_mask_cache"),
    ("history_dct", "샘플링 캐시", "sam3ext.guidance.history", "clear_dct_cache"),
)


@dataclass(frozen=True)
class UnloadHook:
    key: str
    label: str
    release: Callable[[], Any]


class StaleModuleError(RuntimeError):
    """등록된 함수가 지금 메모리의 모듈에 없다 — 업데이트 뒤 Forge 를 다시 시작하지 않았다."""


def module_function(module: str, attr: str) -> Callable[[], Any]:
    """``module.attr()`` 을 누를 때 찾아 부르는 함수. 모듈이 import 되지 않았으면 False(쥔 것이 없다)."""

    def release() -> Any:
        loaded = sys.modules.get(module)
        if loaded is None:
            return False
        fn = getattr(loaded, attr, None)
        if not callable(fn):
            raise StaleModuleError(f"{module}.{attr} 이(가) 없습니다 — 확장을 업데이트했다면 Forge 를 다시 시작하세요")
        return fn()

    release.__name__ = attr
    release.__qualname__ = f"{module}.{attr}"
    return release


class UnloadRegistry:
    def __init__(self, builtin: tuple[tuple[str, str, str, str], ...] = ()) -> None:
        self._lock = threading.Lock()
        self._hooks: dict[str, UnloadHook] = {}
        for key, label, module, attr in builtin:
            self.register(key, label, module_function(module, attr))

    def register(self, key: str, label: str, release: Callable[[], Any]) -> UnloadHook:
        if not isinstance(key, str) or not key:
            raise ValueError("unload hook key must be a non-empty string")
        if not isinstance(label, str) or not label:
            raise ValueError("unload hook label must be a non-empty string")
        if not callable(release):
            raise TypeError(f"unload hook {key!r}: release must be callable")
        hook = UnloadHook(key, label, release)
        with self._lock:
            self._hooks[key] = hook        # 같은 key 는 제자리에서 바꾼다(순서 유지)
        return hook

    def unregister(self, key: str) -> bool:
        with self._lock:
            return self._hooks.pop(key, None) is not None

    def hooks(self) -> list[UnloadHook]:
        with self._lock:
            return list(self._hooks.values())


REGISTRY = UnloadRegistry(BUILTIN_HOOKS)


def register(key: str, label: str, release: Callable[[], Any]) -> UnloadHook:
    """프로세스 공용 목록에 훅을 더한다(같은 key 면 바꿔 끼움)."""
    return REGISTRY.register(key, label, release)


def unregister(key: str) -> bool:
    return REGISTRY.unregister(key)


def hooks() -> list[UnloadHook]:
    return REGISTRY.hooks()
