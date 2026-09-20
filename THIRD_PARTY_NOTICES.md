# Third-party notices

## forge-anima-3.8B (MIT)

`sam3ext/anima38/` 는 [GumGum10/forge-anima-3.8B](https://github.com/GumGum10/forge-anima-3.8B)
(commit `59c27e5`, 2026-08-31) 의 `anima3b` 패키지를 편입한 것입니다. 원본 라이선스는 MIT
(Copyright (c) 2026 GumGum10 contributors) 이며, 이 확장에 편입한 사본에도 그 조건이 그대로
적용됩니다.

이 확장에서 바꾼 부분:

- `files.py` — 경로 계산(확장 안의 위치, `--data-dir` Forge), Forge 가 쓰지 않는 Qwen3.5 모듈 판별.
- `runtime.py` — NegPiP 와 함께 도는 run id 마커 프로토콜(`marker.py`, 신규), 플래그로 켜고 끄는
  멱등 패치, 공용 런타임(`shared_runtime`), inference_mode 밖에서 만드는 가중치, Anima 레퍼런스
  인계, 인코더 파일 사전 확인, 이전 모델을 붙잡지 않는 캐시, Qwen3.5 의미 특징 캐시, LoRA 복제본에
  따라가는 커넥터 패처, 커넥터 전용 `llm_adapter` 사본과 LoRA 패치 동기화.
- `layers.py` — 부분 로드 때 CPU 에 남은 RMSNorm 가중치를 계산 장치로 옮겨 씀.
- `loader_filter.py` (신규) — VAE/Text Encoder 목록의 Qwen3.5 파일을 Forge 로더에서 건너뜀.
- `scripts/anima_3_8b.py` — 이 확장에 맞게 다시 작성(API dict 인자, Bypass, 상태 기록·붙여 넣기).
- 상류의 `bundle_v2.py`(번들 제작 도구)와 `install.py` 는 포함하지 않았습니다.
- `adapter.py`, `qwen35.py`, `semantic_v2.py`, `tokenizer.py` 는 상류와 같습니다.

## TIPO-v2.1-1B-A200M model code (Apache-2.0) and weights (Kohaku License 1.0)

`sam3ext/tipo/kohaku/configuration_kohaku.py`, `modeling_kohaku.py` 는
[KBlueLeaf/TIPO-v2.1-1B-A200M](https://huggingface.co/KBlueLeaf/TIPO-v2.1-1B-A200M)
(revision `f5a318524a4ab30cdbbf51816cf406170f454e65`) 의 무수정 사본으로,
[KohakUwULLM](https://github.com/KohakuBlueleaf/KohakUwULLM) 의 `src/kohakuwullm/export/hf/` 파일과 같습니다
(Apache-2.0, Copyright Shih-Ying Yeh / KohakuBlueleaf). 모델 가중치는 **Kohaku License 1.0** 으로 공개돼 있으며 이 저장소에
포함하지 않습니다 — 사용자가 **모델 받기**를 누를 때 HF 에서 받습니다. 가중치를 재배포하려면 그 라이선스 사본과 고지가
필요합니다.

## Qwen3.5-4B tokenizer (Apache-2.0)

`assets/qwen35_tokenizer/tokenizer.json`, `tokenizer_config.json` 은
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) (revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) 의 무수정 사본입니다. 상류 저장소는 라이선스를
Apache-2.0 으로 명시합니다. 이 확장의 MIT 라이선스는 확장 코드에만 적용되며 이 두 파일의
상류 조건을 대체하지 않습니다.
