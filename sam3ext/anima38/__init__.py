"""Anima 3.8B — Qwen3.5-4B 조건부 경로(Semantic Connector v2)를 Forge Neo 에서 돌리는 런타임.

출처: https://github.com/GumGum10/forge-anima-3.8B (MIT, 2026 GumGum10 contributors) 의
``anima3b`` 패키지를 이 확장에 편입한 것. 원본 대비 바뀐 곳(경로 계산, runtime 의 패치 프로토콜·
캐시·레퍼런스 인계, 부분 로드 norm, 로더 필터 등)은 ``THIRD_PARTY_NOTICES.md`` 에 정리돼 있다.
``adapter.py``·``qwen35.py``·``semantic_v2.py``·``tokenizer.py`` 는 상류와 같다.

왜 편입했나: Anima-3.8B v1.1 체크포인트는 Semantic Connector v2 가중치를 **안에** 담고
있는데(safetensors metadata ``architecture = anima_3_8b_semantic_connector_v2_bundle``),
Forge 본체는 그 190개 텐서를 ``Anima Unexpected`` 로 버린다. 그 결과 어댑터·qwen35_4b 를
모듈에 넣어도 픽셀 하나 안 바뀐다(2026-09-03 A/B 실측). 이 런타임이 그 커넥터를 되살린다.
"""
