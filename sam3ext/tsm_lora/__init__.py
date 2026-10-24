"""TSM LoRA 추론 — LyCORIS ``algo=tsm`` (시간 단계 전문가 LoRA) 파일을 Forge Anima DiT 에서 학습과 같은 식으로 돌린다.

TSM 파일(``ss_saved_format=tsm-native``)은 레이어마다 시간 구간별 LoRA 전문가 여러 개와 라우터를 담는다. Forge 의
LoRA 로더는 이 키를 모른다(``alpha`` 640개만 읽고 95% 가 남아 'LoRA mismatch' 로 파일을 통째로 건너뛴다). 전문가는
샘플링 스텝의 σ 로 고르므로 가중치에 한 번 합칠 수도 없다 — 여기서는 대상 Linear 마다 forward 훅으로 그 스텝의
Δ 를 더한다(LyCORIS bypass 와 같은 자리).

모듈(모두 Forge·Gradio 없이 import 된다 — CPU 단위 테스트용):

* :mod:`.fileformat` — safetensors 헤더 읽기, TSM 키 판별·분리, 검증(``TsmFormatError``), 파일 캐시.
* :mod:`.delta` — σ→t(학습과 같은 반올림), 전문가 고르기, 1·2단계 Δ(샘플별 라우터 풀링), 장치 캐시.
* :mod:`.blocks` — 모듈 이름 → Anima DiT 경로, 28/40/52 블록 변환(``anima_lora_blocks`` 의 표와 끼워 넣은 블록 정책).
* :mod:`.lokr_format` · :mod:`.lokr_delta` — TSM-LoKr(``algo=tsm-lokr``, 전문가가 LoKr + DoRA) 파일 형식과 병합 가중치·2단계 노름·장치 사본.
* :mod:`.runtime` — 생성 한 번의 세션, DiT·Linear 훅, ``networks.load_lora_for_models`` 가로채기(``<lora:…>`` 문법).

UI·infotext·XYZ 축은 ``scripts/tsm_lora.py``.
"""
from __future__ import annotations

from .fileformat import (  # noqa: F401
    TsmFile,
    TsmFormatError,
    TsmModule,
    is_tsm_state_dict,
    load_tsm_file,
    metadata_says_tsm,
    read_header,
)
