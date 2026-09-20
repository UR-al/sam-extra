"""TIPO 프롬프트 확장 — KBlueLeaf/TIPO-v2.1-1B-A200M 으로 txt2img 프롬프트를 확장한다.

- ``kohaku/`` : 모델 코드(HF 리포 원문 그대로, KohakUwULLM 의 Apache-2.0 파일과 같다) — trust_remote_code 없이 쓴다.
- ``prompt.py`` : TIPO 입력 만들기, 출력 파싱, Anima 프롬프트 조립(순수 함수).
- ``runtime.py`` : 모델 파일 확인·받기·로드·장치·생성.

가중치(Kohaku License 1.0)는 저장소에 넣지 않고, 사용자가 "모델 받기"를 누를 때 HF 에서 받는다.
"""
