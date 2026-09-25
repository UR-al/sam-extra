# Third-party notices

이 확장 전체는 GPL-3.0-only(GNU GPL 3판만) 로 배포합니다(루트 [LICENSE](LICENSE)). 아래는 저장소에 함께 들어 있는 제3자 코드·자산과
그 원래 조건입니다. 편입한 파일에는 원래 고지가 그대로 남고, 확장 전체의 GPL-3.0-only 는 이 파일들의 상류 조건을
대체하지 않습니다(MIT·Apache-2.0 은 GPL-3.0 과 함께 배포할 수 있는 조건입니다). `install.py` 가 첫 실행 때 받는
vendor(`lora_manager_vendor/`, `anima_vendor/`)와 사용자가 따로 받는 모델 가중치는 저장소에 없으며 README 의
출처 / 크레딧 절에 적었습니다.

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
- `connector_fp32.py`, `connector_cache.py` (신규) — 커넥터 fp32 상주와 run 단위 계산 캐시(상류에 없는 이 확장의 코드).
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
Apache-2.0 으로 명시합니다. 이 확장의 GPL-3.0-only 라이선스는 이 두 파일의 상류 조건을 대체하지
않습니다.

## CLIP BPE vocabulary (MIT)

`assets/bpe_simple_vocab_16e6.txt.gz`(1.36 MB, SHA-256
`924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`)는 SAM3 텍스트 토크나이저가 읽는 BPE 병합
어휘로, OpenAI CLIP([openai/CLIP](https://github.com/openai/CLIP) 의 `clip/bpe_simple_vocab_16e6.txt.gz`)에서
나온 파일의 무수정 사본입니다. 확인한 근거:

- `sam3` 패키지(0.1.4)의 `sam3/model/tokenizer_ve.py` 머리 주석이 이 토크나이저를 "open_clip and openAI CLIP" 에서
  가져왔다고 밝히고, `sam3/model_builder.py` 가 같은 이름의 파일을 읽습니다.
- Forge 내장 `extensions-builtin/forge_legacy_preprocessors/annotator/oneformer/oneformer/data/` 의 같은 이름 파일
  (OneFormer, CLIP 토크나이저용)과 SHA-256 이 같습니다.
- 이 파일이 없을 때 `sam3ext/core.py` 는 Forge 내장 사본을 먼저 쓰고, 그것도 없으면
  [facebook/sam2](https://huggingface.co/facebook/sam2) 저장소의 `sam2/assets/bpe_simple_vocab_16e6.txt.gz` 를 받습니다.

라이선스는 MIT, Copyright (c) 2021 OpenAI 입니다. 근거는 위 OneFormer 사본 옆의 CLIP 토크나이저
(`oneformer/data/tokenizer.py`) 머리에 붙은 MIT 고지입니다 — openai/CLIP 저장소의 LICENSE 원문은 이 작업에서 네트워크로
다시 열어 보지 않았습니다. MIT 고지 전문은 이 문서 끝에 있습니다. 이 확장의 GPL-3.0-only 라이선스는 이 파일의 상류 조건을
대체하지 않습니다.

## Skimmed_CFG (Apache-2.0)

`sam3ext/guidance/skimmed_cfg.py` 의 `get_skimming_mask`, `skimmed_CFG` 는
[Extraltodeus/Skimmed_CFG](https://github.com/Extraltodeus/Skimmed_CFG)
(commit `d83005832ac42783adfd6f4ae96f6ef6406d1a74`) 의 `skimmed_CFG.py` 9-54줄을 바꾸지 않고 옮긴 것입니다. 같은 파일의
`skim_sigmas`, `skim_active`, `flip_filter_at`, `skim_pair` 는 상류 `CFG_Skimming_Single_Scale_Pre_CFG` 노드의
σ 계산과 `pre_cfg_patch` 본문(152-155줄, 160-197줄)을 함수로 나눈 것입니다. 조건, 순서, 부등호, 계산은 상류와 같고
함수 경계만 이 확장이 새로 나눴습니다(Apache-2.0 4(b)의 변경 표시는 파일 머리 주석에 있습니다). 상류 저장소의
`LICENSE` 는 Apache License 2.0 이며 NOTICE 파일은 없습니다. 이를 부르는 Forge 훅은 `scripts/anima_skimmed_cfg.py`
입니다. 이 확장의 GPL-3.0-only 라이선스는 이 코드의 상류 조건을 대체하지 않습니다.

## Detail Daemon (MIT)

`scripts/anima_detail_daemon.py` 의 `_make_schedule` 은
[muerrilla/sd-webui-detail-daemon](https://github.com/muerrilla/sd-webui-detail-daemon)
(commit `19479998340831d7804fca8efd3f262b54b6373f`) 의 `scripts/detail_daemon.py` 309-338줄 `make_schedule` 과,
그것을 옮긴 [Jonseed/ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon)
(commit `3394e44afea04ed0188fb37b21f0d9952469766b`) 의 `detail_daemon_node.py` 25-67줄
`make_detail_daemon_schedule` 을 옮긴 것입니다. 달라진 곳은 linspace 길이에 붙인 `max(0, …)` 가드 하나입니다. 같은
파일의 `get_dd_schedule` 은 ComfyUI-Detail-Daemon `detail_daemon_node.py` 226-262줄을 바꾸지 않고 옮겼고, 콜백의 σ 조정
(σ 목록·스케줄 텐서 구성, 범위 검사, `* 0.1 * cfg_scale`, `max(1e-06, …)`)은 같은 파일 282-296줄을 따릅니다. 대조
테스트 `tests/test_detail_daemon_origin.py` 는 두 상류의 해당 코드를 그대로 담고 있습니다. 원 라이선스는 둘 다 MIT
(Copyright (c) 2024 Sahand Ahmadian — sd-webui-detail-daemon, Copyright (c) 2024 Jonseed — ComfyUI-Detail-Daemon)
이며 MIT 고지 전문은 이 문서 끝에 있습니다. 이 확장의 GPL-3.0-only 라이선스는 이 코드의 상류 조건을 대체하지 않습니다.

## 코드를 편입하지 않은 재구현 (참고 출처)

아래 기능은 상류 코드를 파일째 가져오지 않고 이 확장 코드로 다시 작성했습니다. 전체 목록(DCW·CNS·DAVE·Modulation
Guidance 등)은 README 의 출처 / 크레딧 절과 [docs/GUIDANCE.md](docs/GUIDANCE.md) 의 크레딧 표에 있습니다. 라이선스를
확인하지 못한 것은 '미확인' 으로 적습니다.

| 기능 | 이 확장 파일 | 원저작 출처 | 원 라이선스 | 형태 |
|---|---|---|---|---|
| Anima Safe PAG | `scripts/anima_safe_pag.py` | [iljung1106/comfyui-anima-safe-pag](https://github.com/iljung1106/comfyui-anima-safe-pag) (ComfyUI 노드), PAG 논문 [arXiv:2403.17377](https://arxiv.org/abs/2403.17377) | 미확인 | Anima 배치 확장·블록 선택을 이식하고 Forge 훅으로 다시 작성 |

## MIT License 전문

`sam3ext/anima38/`(Copyright (c) 2026 GumGum10 contributors), `assets/bpe_simple_vocab_16e6.txt.gz`(Copyright (c) 2021
OpenAI), `scripts/anima_detail_daemon.py` 의 Detail Daemon 코드(Copyright (c) 2024 Sahand Ahmadian, Copyright (c) 2024
Jonseed)에 적용되는 조건입니다. 저작권 줄은 위 각 절의 것을 넣어 읽습니다.

```text
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Apache-2.0 전문(TIPO 모델 코드, Qwen3.5 토크나이저, Skimmed_CFG 코드)은 <https://www.apache.org/licenses/LICENSE-2.0> 에 있습니다.
