"""IPA 모드에서 사용자가 만지는 값들.

torch 도 Forge 도 쓰지 않는다 — 요청 조립 테스트가 가벼워야 하기 때문이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PATCH_SIZE = 16
MIN_REF_SIZE, MAX_REF_SIZE = 224, 1024

# 28블록 어댑터를 더 깊은 모델(2.9B 40 / 3.8B 52)에 얹을 때, 같은 어댑터 블록을 쓰게 되는 복제 블록들의 주입 방식.
#   lineage — 원래 계보 블록(각 어댑터 블록의 첫 번째 대응)에만 주입하고, 끼워 넣은 복제 블록은 IPA 를 걸지 않는다.
#             2.9B·3.8B 모두 격자 무늬 붕괴가 사라지는 것을 GPU 로 확인했다(2026-09-23). 기본값.
#   all     — 모든 복제 블록에 같은 강도로 주입(0.21.x 까지의 동작). 3.8B 는 강도 0.5 에서도, 2.9B 는 1.0 에서 깨진다.
#   split   — 같은 어댑터 블록을 쓰는 복제들이 강도를 나눠 갖는다(각 1/n). all 보다 낫지만 격자 무늬가 남는다.
# 블록 수가 같은 모델(28블록 베이스)에서는 복제가 없어 어느 값이든 결과가 같다.
DUPLICATE_POLICIES = ("lineage", "all", "split")
DEFAULT_DUPLICATE_POLICY = "lineage"
OPT_DUPLICATE_POLICY = "sam3_ipa_duplicate_policy"


def normalize_duplicate_policy(value: Any) -> str | None:
    """알려진 정책 이름이면 소문자로, 아니면 None."""
    text = str(value or "").strip().lower()
    return text if text in DUPLICATE_POLICIES else None


def duplicate_policy_from_opts(opts: Any = None) -> str:
    """Forge 설정의 정책(없거나 이상하면 기본값). ``opts`` 를 안 주면 ``modules.shared.opts`` 를 본다."""
    if opts is None:
        try:
            from modules import shared
        except Exception:  # Forge 밖(테스트)
            return DEFAULT_DUPLICATE_POLICY
        opts = getattr(shared, "opts", None)
    try:
        value = getattr(opts, OPT_DUPLICATE_POLICY, DEFAULT_DUPLICATE_POLICY)
    except Exception:
        return DEFAULT_DUPLICATE_POLICY
    return normalize_duplicate_policy(value) or DEFAULT_DUPLICATE_POLICY


@dataclass(frozen=True)
class IpaOptions:
    strength: float = 1.0
    ref_size: int = 512
    separate_cfg: bool = False
    cfg_scale: float = 4.0
    siglip_layer: int = -1
    gray_null: bool = False
    use_lora: bool = True
    # 3.8B v2 번들에서 Semantic Connector 를 설치할지. None 이면 Forge 설정
    # (sam3_anima38_reference_ipa, 기본 켬)을 따른다 — 결과가 바뀌는 토글이다.
    anima38_connector: bool | None = None
    # 복제 블록 정책(DUPLICATE_POLICIES). None 이면 Forge 설정(sam3_ipa_duplicate_policy, 기본 lineage)을 따른다.
    duplicate_policy: str | None = None

    def resolved_duplicate_policy(self, opts: Any = None) -> str:
        """요청이 정했으면 그 값, 아니면 Forge 설정."""
        if self.duplicate_policy is not None:
            return normalize_duplicate_policy(self.duplicate_policy) or DEFAULT_DUPLICATE_POLICY
        return duplicate_policy_from_opts(opts)

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
        if self.duplicate_policy is not None and normalize_duplicate_policy(self.duplicate_policy) is None:
            raise ValueError(
                "복제 블록 정책은 " + ", ".join(DUPLICATE_POLICIES) + " 중 하나여야 합니다."
            )
        return self
