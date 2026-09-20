"""IPA 모드에서 사용자가 만지는 값들.

torch 도 Forge 도 쓰지 않는다 — 요청 조립 테스트가 가벼워야 하기 때문이다.
"""
from __future__ import annotations

from dataclasses import dataclass

PATCH_SIZE = 16
MIN_REF_SIZE, MAX_REF_SIZE = 224, 1024


@dataclass(frozen=True)
class IpaOptions:
    strength: float = 1.0
    ref_size: int = 512
    separate_cfg: bool = False
    cfg_scale: float = 4.0
    siglip_layer: int = -1
    gray_null: bool = False
    use_lora: bool = True

    def validate(self) -> "IpaOptions":
        if not 0.0 <= float(self.strength) <= 2.0:
            raise ValueError("IP-Adapter 강도는 0 과 2 사이여야 합니다.")
        size = int(self.ref_size)
        if not MIN_REF_SIZE <= size <= MAX_REF_SIZE:
            raise ValueError(
                f"참조 해상도는 {MIN_REF_SIZE} 와 {MAX_REF_SIZE} 사이여야 합니다."
            )
        if size % PATCH_SIZE:
            raise ValueError(f"참조 해상도는 {PATCH_SIZE} 의 배수여야 합니다.")
        if not 1.0 <= float(self.cfg_scale) <= 10.0:
            raise ValueError("IP CFG 는 1 과 10 사이여야 합니다.")
        if not -1 <= int(self.siglip_layer) <= 24:
            raise ValueError("SigLIP 레이어는 -1 과 24 사이여야 합니다.")
        return self
