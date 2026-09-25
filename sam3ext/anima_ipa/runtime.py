"""어댑터 로드 캐시와 한 번의 생성용 세션.

쉴 때 가중치와 인코더는 CPU RAM 에 있다. 인코딩할 때만 GPU 로 올렸다가 내린다
(``encode.ReferenceEncoder`` 가 처리한다) — TIPO 와 같은 규칙이다.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from . import paths
from .checkpoint import inspect_adapter
from .options import IpaOptions
from .patch import Injection, patched_unet

GREY = (128, 128, 128)


@dataclass
class LoadedAdapter:
    spec: Any
    weights: dict
    encoder: Any


def _load_pretrained(models_dir) -> LoadedAdapter:
    """safetensors 와 SigLIP2 를 CPU 로 읽어 들인다."""
    import safetensors
    import safetensors.torch  # noqa: F401  (safe_open 을 쓰려면 필요하다)
    from torch import nn
    from transformers import SiglipVisionModel

    from .encode import ReferenceEncoder

    weights: dict[str, Any] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    with safetensors.safe_open(
        str(paths.adapter_path(models_dir)), framework="pt"
    ) as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            weights[key] = tensor
            shapes[key] = tuple(tensor.shape)
    spec = inspect_adapter(list(weights), shapes, metadata)

    vision = SiglipVisionModel.from_pretrained(
        str(paths.encoder_dir(models_dir)), local_files_only=True
    )
    vision.requires_grad_(False)
    vision.eval()

    siglip_norm = None
    if spec.has_siglip_norm:
        siglip_norm = nn.LayerNorm(spec.embed_dim, elementwise_affine=True)
        siglip_norm.load_state_dict(
            {
                key[len("siglip_norm."):]: value
                for key, value in weights.items()
                if key.startswith("siglip_norm.")
            }
        )
        siglip_norm.eval()

    encoder = ReferenceEncoder(vision, siglip_norm=siglip_norm)
    return LoadedAdapter(spec=spec, weights=weights, encoder=encoder)


class IpaRuntime:
    def __init__(self, models_dir=None, loader=None):
        self._models_dir = Path(models_dir) if models_dir is not None else None
        self._loader = loader or _load_pretrained
        self._loaded: LoadedAdapter | None = None
        self._download_lock = threading.Lock()

    @property
    def models_dir(self):
        return self._models_dir

    def missing_files(self) -> list[str]:
        return paths.missing_files(self._models_dir)

    def download(self, downloader=None) -> bool:
        """받았거나 이미 다 있으면 True. 다른 곳에서 받는 중이면 기다리지 않고 False."""
        if not self._download_lock.acquire(blocking=False):
            return False
        try:
            paths.download(self._models_dir, downloader=downloader)
        finally:
            self._download_lock.release()
        return True

    def load(self) -> LoadedAdapter:
        if self._loaded is None:
            self._loaded = self._loader(self._models_dir)
        return self._loaded

    @contextmanager
    def session(self, sd_model, image, options: IpaOptions) -> Iterator[int]:
        """참조를 인코딩하고, 샘플링 동안만 주입된 패처를 끼운다.

        어댑터 깊이를 내놓는다 — 모델 깊이와 다르면 블록 계보 매핑이 걸렸다는 뜻이고,
        부르는 쪽이 그 사실을 결과에 남긴다.
        """
        import torch
        from PIL import Image as PILImage

        options.validate()
        loaded = self.load()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokens = loaded.encoder.encode(
            image, size=options.ref_size, layer=options.siglip_layer, device=device
        )

        null_tokens = None
        if loaded.spec.has_null_tokens and not options.gray_null:
            stored = loaded.weights.get("null_tokens")
            if stored is not None:
                null_tokens = stored.to(dtype=tokens.dtype)
                if null_tokens.shape != tokens.shape:
                    null_tokens = null_tokens.expand_as(tokens)
        if null_tokens is None:
            # 0 벡터는 uncond 게이트를 꺼 버려 IP 기여가 텍스트 CFG 로 증폭된다.
            # 회색 이미지를 인코딩해 "신호 없음"을 SigLIP2 공간 안에서 표현한다.
            grey = PILImage.new("RGB", (options.ref_size, options.ref_size), GREY)
            null_tokens = loaded.encoder.encode(
                grey, size=options.ref_size, layer=options.siglip_layer, device=device
            )

        injection = Injection(
            tokens=tokens,
            null_tokens=null_tokens,
            gate_scale=float(options.strength),
            separate_cfg=bool(options.separate_cfg),
            cfg_scale=float(options.cfg_scale),
        )
        with patched_unet(
            sd_model, loaded.spec, loaded.weights, injection,
            use_lora=bool(options.use_lora),
            duplicate_policy=options.resolved_duplicate_policy(),
        ):
            yield int(loaded.spec.num_blocks)


_SHARED: IpaRuntime | None = None


def shared_runtime() -> IpaRuntime:
    global _SHARED
    if _SHARED is None:
        _SHARED = IpaRuntime()
    return _SHARED
