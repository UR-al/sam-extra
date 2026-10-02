"""참조 이미지를 SigLIP2 토큰으로 바꾼다.

torchvision 을 쓰지 않는다 — 정규화는 mean/std 0.5 한 줄이고, 의존을 하나 줄이는 편이 낫다.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def letterbox(image: Image.Image, target: int) -> Image.Image:
    """비율을 지키며 ``target`` 정사각형 한가운데에 놓고 남는 곳은 검정으로 채운다."""
    source = image.convert("RGB")
    width, height = source.size
    ratio = target / max(width, height)
    new_size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    resized = source.resize(new_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target, target), (0, 0, 0))
    canvas.paste(resized, ((target - new_size[0]) // 2, (target - new_size[1]) // 2))
    return canvas


def to_pixel_values(image: Image.Image, target: int) -> torch.Tensor:
    """(1, 3, target, target), mean/std 0.5 정규화, fp32."""
    array = np.asarray(letterbox(image, target), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return (tensor - 0.5) / 0.5


class ReferenceEncoder:
    """SigLIP2 비전 타워 + 어댑터에 딸린 선택적 모듈들.

    ``siglip_norm`` 은 중간 레이어를 뽑을 때만 쓴다 — 마지막 레이어는 이미 정규화돼 있다.
    """

    def __init__(self, vision_model, *, siglip_norm=None, compressor=None, self_attn=None):
        self.vision_model = vision_model
        self.siglip_norm = siglip_norm
        self.compressor = compressor
        self.self_attn = self_attn

    def encode(self, image, *, size: int, layer: int = -1, device="cpu") -> torch.Tensor:
        pixel_values = to_pixel_values(image, int(size)).to(device=device)
        self.vision_model.to(device)
        try:
            with torch.no_grad():
                if layer == -1:
                    tokens = self.vision_model(
                        pixel_values, interpolate_pos_encoding=True
                    ).last_hidden_state
                else:
                    output = self.vision_model(
                        pixel_values,
                        interpolate_pos_encoding=True,
                        output_hidden_states=True,
                    )
                    tokens = output.hidden_states[layer]
                    if self.siglip_norm is not None:
                        self._place(self.siglip_norm, device)
                        tokens = self.siglip_norm(tokens)
                if self.compressor is not None:
                    self._place(self.compressor, device)
                    tokens = self.compressor(tokens)
                if self.self_attn is not None:
                    self._place(self.self_attn, device)
                    tokens = self.self_attn(tokens)
        finally:
            # 인코더를 GPU 에 남기지 않는다 — Forge 메모리 관리 밖이라 Anima 를 밀어낸다.
            self.vision_model.to("cpu")
        return tokens.detach()

    @staticmethod
    def _place(module, device) -> None:
        mover = getattr(module, "to", None)
        if callable(mover):
            mover(device)
