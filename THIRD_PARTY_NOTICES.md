# Third-party notices

이 확장의 코드는 GPL-3.0-only(GNU GPL 3판만) 로 배포합니다(루트 [LICENSE](LICENSE)). 예외로 편입한 NegPiP 파일은 AGPL-3.0-or-later
를 따르므로 저장소 전체는 SPDX `GPL-3.0-only AND AGPL-3.0-or-later` 입니다(아래 절). 아래는 저장소에 함께 들어 있는 제3자 코드·자산과
그 원래 조건입니다. 편입한 파일에는 원래 고지가 그대로 남고, 이 확장의 GPL-3.0-only 는 이 파일들의 상류 조건을
대체하지 않습니다(MIT·BSD-3-Clause·Apache-2.0 은 GPL-3.0 과 함께 배포할 수 있는 조건이고, AGPL-3.0 인 NegPiP 는 GPL-3.0 13조로 결합합니다 — 아래 절). `install.py` 가 첫 실행 때 받는
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
  따라가는 커넥터 패처, 커넥터 전용 `llm_adapter` 사본과 LoRA 패치 동기화, 새 Forge TE 엔진·엔진 속성 이름 대응, 아래에 깔린
  NegPiP 의 마스킹 대행(내장 `sam3ext/negpip/mask.py` 사용), v1/v2 경로의 `Emphasis` 생성 정보 기록.
- `native_engine.py` (신규) — Forge `21886f41` 의 `Qwen06Engine` 에서 커넥터 원본 입력을 뽑는 경로, 엔진별 `Emphasis` 기록
  규칙, Forge 판마다 다른 속성 이름(2.29.1 까지 `text_processing_engine_anima`, 2.29.2 부터 Flux2·Krea2·Qwen-Image·Z-Image 와
  같은 `text_processing_engine_qwen`)에서 Anima 엔진만 찾는 `anima_text_engine`(이 확장의 코드).
- `layers.py` — 부분 로드 때 CPU 에 남은 RMSNorm 가중치를 계산 장치로 옮겨 씀.
- `loader_filter.py` (신규) — VAE/Text Encoder 목록의 Qwen3.5 파일을 Forge 로더에서 건너뜀.
- `connector_fp32.py`, `connector_cache.py` (신규) — 커넥터 fp32 상주와 run 단위 계산 캐시(상류에 없는 이 확장의 코드).
- `scripts/anima_3_8b.py` — 이 확장에 맞게 다시 작성(API dict 인자, Bypass, 상태 기록·붙여 넣기).
- 상류의 `bundle_v2.py`(번들 제작 도구)와 `install.py` 는 포함하지 않았습니다.
- `adapter.py`, `qwen35.py`, `semantic_v2.py`, `tokenizer.py` 는 상류와 같습니다.

## forgeneo-mcp (MIT)

`mcp_server/sam_extra_mcp/forgeneo/` 는 [eduardoabreu81/forgeneo-mcp](https://github.com/eduardoabreu81/forgeneo-mcp)
(commit `a103dc5`) 의 `forgeneo_mcp` 패키지를 편입한 것이고, `mcp_server/sam_extra_mcp/server.py`·`service.py` 는 그 패키지의
`server.py` 에서, `tests/test_mcp_forgeneo_upstream.py` 는 상류 `tests/` 에서 나왔습니다. 원본 라이선스는
MIT(Copyright (c) 2026 Eduardo Abreu)이며, 각 파일 머리에 저작권·허가 고지와 이 확장에서 바꾼 곳 목록이 그대로 남아 있습니다.

이 확장에서 바꾼 부분:

- 패키지 이름 `forgeneo_mcp` → `sam_extra_mcp.forgeneo`. MCP SDK 는 `server.py` 만 불러오고(`mcp>=2.2,<3`, 1.x FastMCP 대체 경로
  제거) 도구 본문은 SDK 없이 도는 `service.py` 로 옮김. 권한 정책(`policy.py`)·경로 찾기(`forge_paths.py`)는 이 확장의 코드.
- `config.py` — Forge 가 같은 PC(`localhost` 나 루프백 IP 리터럴)면 FORGE_PATH_MAP 없이 Forge 경로를 그대로 씀, 자격 증명은
  repr·출력에서 숨김(FORGE_URL 에 적힌 `사용자:비밀번호@` 도 URL 에서 빼 인증으로 옮김), 기록 색인 갱신 주기(`FORGE_HISTORY_MAX_AGE`),
  한 번 생성의 화소 상한(`SAM_EXTRA_MCP_MAX_PIXELS`).
- `client.py` — 쓰는 경로만 허용(`/sdapi/v1/cmd-flags`·`/internal/sysinfo` 는 호출하지 않음 — cmd-flags 는 `--api-auth` 자격 증명을
  그대로 돌려줌), 리디렉션을 따라가지 않고 3xx 는 오류(상류는 따라감), 옵션 쓰기는 체크포인트 전환 키만(요청 경로에서 검사), 진행
  상황은 미리보기 이미지 없이, 테스트용 transport, httpx 지연 import.
- `generate.py` — 출력 폴더를 확장 위치에서 찾고 API 결과는 `outdir_samples` 무시(Forge API 와 같게), 결과 파일을 infotext·시드·
  요청 시간으로 이 요청 것만 고름(동시에 돌린 다른 생성 제외), 디스크에서 못 찾은 이미지는 API 응답에서 대체 폴더에 저장(상류는 렌더
  뒤 ok:false), 실제 형식 확장자·덮어쓰지 않음, 격자·보조 저장 구분, init 이미지는 이 PC 의 정지 이미지 64 MB 까지(네트워크 공유·
  장치 경로 거절, 정규화한 절대 경로로 엶), 요청이 그릴 화소 수 계산(`requested_pixels`: 크기·하이레스·배치·반복).
- `history.py`·`loras.py` — 색인 갱신(요청·생성 뒤·나이 제한), 강제 재구축 때 두 번 세던 문제, 바뀌지 않은 파일은 다시 읽지 않음, 잠금.
  LoRA 제작자가 쓴 글(제목·기반 모델·트리거·태그·설명)은 길이를 자르고 결과에서 `untrusted_tags`·`untrusted_description`·
  `untrusted_categories` 로 이름 붙임(상류는 `tags`·`description`·`categories`, 설명 600자 → 400자).
- `history.py` — `normalise_checkpoint` 는 Forge 체크포인트 제목 끝의 `[해시]`(`name.safetensors [0123456789]`, `sd_model_checkpoint`)를
  확장자보다 먼저 뗌(새 `drop_trailing_bracket`; 상류는 `.safetensors` 가 남아 불러온 체크포인트가 자기 생성 기록과 맞지 않았음),
  `_looks_danbooru` 는 `dialects._looks_tagged` 의 태그 판별도 셈(공백으로 쓴 Anima 태그를 산문으로 보던 것).
- `infotext.py` — JPEG·WebP 의 EXIF UserComment, tEXt 는 Latin-1(PNG 규격), 압축 iTXt·zTXt, 메모리 이미지 읽기.
- `fetcher.py` — httpx 로 교체, 크기와(Hugging Face 가 알려 주면) SHA256 이 맞을 때만 파일을 남기고 크기를 모르면 받지 않음,
  safetensors 머리 확인, 정책 결과(`permitted`)를 반드시 받음, 크기 확인 HEAD 가 GET 이 되지 않음, 덮어쓰지 않음(이름이 있으면 실패하는
  이동으로 자리에 놓음 — Windows 는 `os.rename`, 그 밖은 하드 링크).
- `modules.py` — sam-extra Anima 3.8B 모듈(Qwen3.5-4B, 확장 어댑터)을 인정하는 `extras`, Qwen2D VAE 계열.
- `profile.py`·`capabilities.py`·`identity.py` — 형 검사 전용 import, extras 전달, 정책·경로 보고, 응답 캐시 폴더 설정·원자적 쓰기,
  체크포인트 전환 때 넘긴 프리셋은 그 인스턴스에 있는 것만(상류는 그대로 씀), 체크포인트 사이드카 태그는 길이를 잘라
  `untrusted_checkpoint_tags` 로.
- `profile.py` — `preset_for_checkpoint` 는 체크포인트 이름을 `[해시]` 없이 비교(`_bare_name`; Forge 의 `forge_checkpoint_<프리셋>` 기록과
  `models` 에서 고른 이름이 해시만 달라도 상류는 프리셋 신호를 잃어 그 체크포인트의 VAE·텍스트 인코더 없이 불러올 수 있었음),
  `resolve_dialect` 설명을 아래 `identity.py` 에 맞춤.
- `identity.py` — `resolve` 에서 과거 프롬프트는 아키텍처가 뜻하는 방언을 뒤집지 않음: 모호한 아키텍처(xl)의 갈래를 고르거나 같은
  방언을 확인할 때만 씀(상류는 과거 프롬프트가 이겨, Anima 품질 태그 `masterpiece, best quality`·`score_*` 가 든 Anima 체크포인트를
  Illustrious·Pony 로 봄).
- `presets.py`·`dialects.py`·`downloads.py`·`civitai.py` 는 상류와 같습니다. 상류의 README·`glama.json`·배너 이미지는 넣지 않았습니다.
- `tests/test_mcp_forgeneo_upstream.py` 는 상류 pytest 130개 중 128개를 unittest 로 옮긴 것입니다.

MIT 고지 전문은 이 문서 끝에 있습니다. 이 확장의 GPL-3.0-only 라이선스는 이 코드의 상류 조건을 대체하지 않습니다.

## sd-forge-negpip — NegPiP (AGPL-3.0-or-later)

`sam3ext/negpip/`(`__init__.py`·`anima.py`·`sd.py`·`utils.py`·`mask.py`)와 `scripts/negpip.py` 는
[Haoming02/sd-forge-negpip](https://github.com/Haoming02/sd-forge-negpip) (branch `classic`, commit `0585496`, 2026-09-30)
의 `lib_negpip/` 패키지와 `scripts/negpip.py` 를 편입한 것입니다. 저자가 같은 날(2026-09-30) 저장소를 보관(archive)해 더 이상
갱신되지 않으므로 이 확장이 이어 받았습니다. 원 저작권은 Copyright (C) 2025 hako-mikan, Copyright (C) 2026 Haoming02 이고
라이선스는 **GNU Affero General Public License 3판 또는 그 이후 판(AGPL-3.0-or-later)** 입니다. 전문은 코드 옆
`sam3ext/negpip/LICENSE`(상류 `LICENSE` 무수정 사본, `tests/test_negpip_vendor.py` 가 SHA-256 을 고정)에 있습니다.

이 확장(GPL-3.0-only)과 AGPL-3.0 코드는 GPL-3.0 13조(AGPL-3.0 13조도 같은 허락)에 따라 한 작업으로 결합해 배포할 수 있습니다.
편입한 파일은 계속 AGPL-3.0-or-later 를 따르고(이 확장의 GPL-3.0-only 가 대체하지 않음), 이 확장의 나머지 파일은 GPL-3.0-only 그대로입니다.
다만 GPL-3.0 13조는 AGPL-3.0 13조의 네트워크 상호작용 요건이 "결합된 작업 그 자체(the combination as such)"에 적용된다고 정합니다 —
결합된 작업을 네트워크 너머 사용자가 원격으로 쓰게 하면, NegPiP 부분만이 아니라 결합된 작업 전체의 대응 소스를 받을 기회를 그 사용자에게
제공해야 합니다. 편입한 파일마다 머리에 원래 저작권·라이선스 고지와 수정 고지(AGPL-3.0 5조 a항,
`MODIFIED by sam-extra, 2026-09-30` — 그 뒤에 고친 파일은 고친 날짜의 줄을 더함: `anima.py`·`scripts/negpip.py` 2026-10-02)를 달았습니다.

이 확장에서 바꾼 부분:

- 이름: 패키지 `lib_negpip` → `sam3ext.negpip`. import 경로만 바꿨습니다. 스크립트 파일 이름 `scripts/negpip.py` 는 그대로입니다 —
  ADetailer 는 패스에 넣을 always-on 스크립트를 파일 이름(`ad_script_names` 기본값의 `negpip`)으로 고릅니다. Forge 는 확장 스크립트를
  `확장 이름/파일 이름` 으로 구별하고 경로로 불러오므로(`sys.modules` 에 넣지 않음) 따로 설치된 확장의 같은 이름 파일과 부딪히지 않습니다.
- `mask.py`(새 파일) — 상류 `anima.py` 의 `_build_negpip_mask` 를 Forge 를 import 하지 않는 모듈로 옮기고, Forge 의 두 Anima 텍스트
  엔진을 모두 지원합니다: 옛 `AnimaTextProcessingEngine`(Forge `ad88b6b4` 까지)은 `tokenize_line` 의 `t5_multipliers`(상류 `b3673ce`
  방식), 새 `Qwen06Engine`(Forge `21886f41` 부터)은 `t5_tokenizer.tokenize_with_weights`(상류 `0585496` 방식). 새 엔진의 emphasis
  `None`(가중치를 파싱하지 않아 토큰 행부터 다름)·`Ignore`(T5 가중치 1.0)에서는 음수 행을 만들지 않습니다 — 엔진이 뒤집지 않은 행을
  NegPiP 가 뒤집으면 뜻이 반대가 됩니다. `anima.py` 의 `_build_negpip_mask` 는 이것을 부릅니다.
- `__init__.py` — 패치 상태 `PATCHED` 를 패키지에 두어 Reload UI 로 스크립트 클래스가 새로 생겨도 유지합니다.
- `scripts/negpip.py` — `_patched` 로 `PATCHED` 를 쓰고, 따로 설치된 NegPiP 가 로드돼 있으면 쉬며(경고 한 번), Anima 에서
  emphasis 가 음수 가중치를 적용하지 않는 방식이면 켜지 않습니다.
- Anima 텍스트 엔진 찾기 — `scripts/negpip.py`(emphasis 판단)와 `anima.py`(조건 훅)는 엔진을 `sam3ext.anima38.native_engine` 의
  `anima_text_engine` 으로 찾습니다. 상류는 `sd_model.text_processing_engine_anima` 를 바로 읽는데, Forge 2.29.2(`46365871`)가 그 속성을
  Flux2·Krea2·Qwen-Image·Z-Image 와 같은 `text_processing_engine_qwen` 으로 옮겨 Anima 생성마다 속성 오류로 NegPiP 가 빠졌습니다. 헬퍼는
  두 이름을 다 찾고, 공용 이름에서는 T5 토크나이저를 가진 엔진(Anima 엔진만 가짐)만 돌려줍니다.
- **상류와 다른 동작 — SD1/SDXL `_cond_dealer`**: Forge `21886f41` 부터 2.29.1 까지 `sd_engine.ClipEngine.tokenize` 는 `add_special_tokens=False` 를
  넘기지 않아 프롬프트 조각마다 BOS/EOS 를 붙입니다. 상류의 자르기 `cond[1 : token_len + 2]` 는 옛 엔진(`classic_engine`, `ad88b6b4` 까지)
  에서 `[단어…, EOS]` 행이지만 그 엔진에서는 `[BOS, 단어…, EOS, EOS]` 가 되어 NegPiP 가 BOS(어텐션 싱크) 행의 V 까지 뒤집습니다. 편입본은
  `process_batch` 에서 엔진에 빈 글자를 토큰화해 특수 토큰이 나오는지 묻고(`utils.clip_fragment_specials`, 판 번호가 아님), 그렇다면
  `_cond_dealer` 가 청크의 특수 토큰이 아닌 행과 마지막 단어 바로 뒤 EOS 행(`utils.clip_word_rows`) — 옛 엔진과 같은 `[단어…, EOS]` — 을
  고릅니다. Forge 2.29.2(`0b1783c7`)는 `add_special_tokens=False` 를 되살려 행 배치가 옛 엔진과 같고, 편입본은 그 판에서 옛 엔진과 같은
  경로(아래 `id_start`·`id_end` 규칙)를 탑니다. 두 헬퍼는 `utils.py` 에 더했고(상류 함수는 그대로), `tests/test_negpip_clip_rows.py` 가 실제
  SD1.5 CLIP 토크나이저와 세 판 엔진 코드(옛 엔진·`21886f41`·`0b1783c7`)로 확인합니다.
- **상류와 다른 동작 — `_cond_dealer` 가 행을 세는 글자와 옛 엔진의 행 규칙**: 상류는 `"(글:-w)"` 를 인코딩하면서 행 수는 맨 글 `글`(가중치 1)을
  토큰화해 셉니다. Forge 의 `parse_prompt_attention` 은 `BREAK` 를 `["BREAK", -1]` 로 내고 묶음 가중치를 그 -1 에도 곱하며, 엔진은 가중치가
  정확히 -1 인 `BREAK` 에서만 청크를 나눕니다 — `(cat BREAK dog:2)` 는 한 청크(글자 `break`), 맨 글 `cat BREAK dog` 는 두 청크라 행 번호가 조건과
  어긋납니다(상류: 새 엔진에선 조건 밖을 잘라 채움 행 76개, 옛 엔진에선 채움 EOS 와 다음 청크 BOS 76~77 행. 위 새 엔진 행 규칙만 쓰면
  `IndexError` 로 NegPiP 가 조용히 꺼짐). 편입본은 인코딩한 바로 그 글자(첫 스케줄 줄)를 토큰화해 행을 세고, 옛 엔진·`IS_NEO` 가 아닌 경로도
  같은 행 규칙(엔진의 `id_start`·`id_end` 를 건너뛴 단어 행 + 뒤 EOS, `utils.clip_row_specials`·`encoded_clip_rows`)을 씁니다 — 한 청크 항은
  상류 자르기 `cond[1 : token_len + 2]` 와 한 행도 다르지 않고(Emphasis `None` 은 예외 — 괄호·가중치가 글자로 인코딩돼
  이제 그 글자 행 전부를 잡는다), 여러 청크 항(묶음 가중치 -1 의 `BREAK`, 75토큰 초과)은 청크 경계의 채움·BOS 행
  대신 단어 행만 잡아 두 엔진이 같은 행을 고릅니다. 엔진이 `id_start`·`id_end` 를 알리지 않으면 상류 자르기 그대로입니다. 방어로, 토큰화한
  청크의 행 수가 조건 행 수와 다르면 그 생성에서 물러나며(`NegPiP Disabled (condition rows: …)`) 지웠던 음수 항을 프롬프트에 되돌립니다
  (`utils.snapshot_prompts`·`restore_prompts`). `tests/test_negpip_clip_rows.py` 가 세 판 실제 엔진 코드로 확인합니다.
- **상류와 다른 동작 — Anima 조건 훅의 반환 모양**: 상류 `negpip_learned_conditioning` 은 스케줄 줄들을 `torch.stack` 해 dict 하나로 돌려줍니다.
  Anima 엔진(옛·새)은 줄마다 `max(512, T5 토큰 수)` 행이라, 프롬프트 편집 `[a:b:N]`·`[a|b]` 줄이 512 를 넘어 길이가 다르면
  `stack expects each tensor to be equal size` 로 조건 단계에서 죽습니다(Forge 순정은 스텝마다 한 줄을 골라 문제없음). 편입본은 Forge 의
  줄별 계약대로 줄마다 `{"crossattn": (L_i, D), "c_negpip_mask": (L_i, 1)}` dict 하나의 list 를 돌려줍니다(`mask.py` 의 `negpip_line_conds`,
  3.8B 런타임의 NegPiP 대행 `_apply_negpip` 도 같은 헬퍼). 스텝마다 줄 고르기와 배치 길이 맞추기는 Forge `prompt_parser` 가 순정과 같게
  하므로(`stack_conds` 끝 행 반복, 조건·마스크에 같게) 512 행 이하 줄은 예전과 같은 값입니다. `tests/test_negpip_anima_schedule.py` 가 실제
  `Qwen06Engine`·Anima 토크나이저·`prompt_parser` 로 확인합니다.
- **상류와 다른 동작 — SD1/SDXL 음수 항을 어텐션에 붙이는 방식 (`sd.py` `_hook_forward`, `scripts/negpip.py` `_cond_dealer`·
  `_calc_conds`·`denoiser_callback`·`process_batch` 의 로그, `utils.py` 의 `join_term_rows`·`active_rows`·`context_rows`)**. 상류
  `0585496` 에는 세 가지 결함이 있었습니다(모두 상류부터). (1) 한 프롬프트에 음수 항이 둘 이상이면 첫 항만 붙었습니다 — `554c122`
  재작성 뒤 `_calc_conds` 가 항마다 텐서 하나를 만들고 훅은 목록의 `[0]` 만 넘겼습니다(hako-mikan 원본은 한 영역의 항들을 `torch.cat`
  한 텐서 하나). 나머지 항은 프롬프트에서 지워진 채 어디에도 들어가지 않았습니다. 부정 쪽 `denoiser_callback` 은 `break` 가 안쪽
  반복에 있어 부정 프롬프트의 `[(x:-1):5]` 가 0 스텝부터 걸렸습니다(hako-mikan 원본부터). (2) 배치 항목 0 의 스케줄만 골라
  `batch_size` 로 반복했습니다 — 항목마다 프롬프트가 다르면(sd-dynamic-prompts 와일드카드 + Batch size ≥ 2) 뒤 항목은 항목 0 의 음수 항을
  받거나 제 음수 항을 잃었습니다. (3) cond/uncond 를 샘플러 이름(`rev`)·`x.shape[0] == 2*batch_size`·모듈마다의 호출 수(`Counter`, 한도
  16/70 은 원래 UNet 전체의 attn2 수라 모듈마다 두면 16/70 패스마다 뒤집힘)·문맥 길이로 추정했습니다 — Forge 의 `calc_cond_uncond_batch`
  는 샘플러와 무관하게 `[U, C]` 로 묶고(DDIM/PLMS/UniPC 는 반대 절반에 붙음), 메모리가 모자라면 U·C 를 따로, 문맥 청크 수의 lcm 비가
  4 를 넘으면 C·U 를 따로, CFG 1·Skip Early CFG·NGMS 면 C 만, AND 프롬프트면 C 조각을 여럿 돌립니다(이때 음수 항이 8 스텝씩 켜졌다
  꺼지거나 반대쪽에 붙음). 편입본은: `_cond_dealer` 가 반복하지 않은 `[행 수, D]` 를 돌려주고, `_calc_conds` 가 항목·스케줄 줄마다 그 줄의
  항 전부를 토큰 축으로 이은 `(행 | None, 행 수)` 하나를 만들며(같은 항·같은 줄은 한 번 인코딩해 같은 텐서), `denoiser_callback` 이 항목마다
  제 스케줄에서 긍정·부정 같은 문턱(상류 긍정 쪽의 `step >= sampling_step + 2`)으로 고르고 이 스텝 조건의 원래 문맥 길이를 적습니다.
  훅은 Forge 가 attn2 까지 넘기는 `transformer_options["cond_or_uncond"]`(Forge `ad88b6b4`·`ceff5168` 같음)로 조각마다 C/U 를 알고,
  조각의 행 r 에 항목 r(행 수가 항목 수와 다르면 Forge `repeat_to_batch_size` 처럼 r % 항목 수)의 행을 붙여 그 행들의 V 만 뒤집습니다.
  덧붙일 것이 같은 행끼리 한 번에 계산하고(흔한 경우 호출 수는 상류와 같음), 음수 항이 없는 행은 원래 `forward` 로 계산합니다(NegPiP 가
  없을 때와 같은 계산 — 차이는 배치 크기에 따른 부동소수 반올림뿐). Forge 가 cond·uncond 를 묶으며 문맥을 lcm 길이로 반복했으면 붙일 행도
  같은 배수로 반복해 음수 항의 softmax 몫을 원래 문맥 + 음수 항과 같게 둡니다(상류는 반복 문맥 뒤에 한 번만 붙여 반대쪽 프롬프트가 길면
  음수 항이 약해졌음). attn2_patch 가 `value` 를 따로 넘기면 거기에도 같은 행을 붙입니다(상류는 K 만 길어져 모양이 어긋남).
  `transformer_options` 에 표시가 없는 호출자(표시를 넘기지 않는 Forge 등)에서는 상류의 추정을 그대로 쓰고 콘솔에 경고를 한 번 남깁니다.
  어텐션 마스크를 받은 호출(Forge 의 attn2 는 넘기지 않음)은 그 호출만 NegPiP 없이 원래 `forward` 로 돌리고 경고를 한 번 남깁니다.
  `NegPiP Enable (Positive/Negative: N)` 의 N 은 어느 항목·줄이든 이은 행 수의 최대입니다(상류: 항목 0 의 첫 항, 항목 0 에 없으면 줄 없음).
  `Counter`·`patch_sd_negpip`·`_main_forward` 는 상류 그대로입니다. `tests/test_negpip_sd_batching.py` 가 실제 Forge
  `calc_cond_uncond_batch`·`condition.py`·`CrossAttention`·`prompt_parser`·파서와 SD1.5 CLIP 토크나이저(CPU)로 묶음·메모리 나눔(U, C)·
  모양 나눔(C, U)·CFG 1·AND·lcm 반복(긍정·부정 어느 쪽이 짧든 `denoiser_callback` 이 `text_cond`/`text_uncond` — SDXL 은
  `DictWithShape` 의 `crossattn` — 에서 원래 길이를 읽는 데까지)·항목별·스케줄·하이레스를 확인합니다.
- 로드 순서 — 확장 루트 `metadata.ini` 의 `[scripts/negpip.py] After = sd-dynamic-thresholding`: 따로 설치된 sd-forge-negpip 는 폴더
  이름순으로 Dynamic Thresholding 뒤였고, 상류 NegPiP `process_batch` 는 Dynamic Thresholding 이 바꾼 `p.sampler_name` 으로 cond/uncond
  절반을 골랐습니다. 편입본의 훅은 Forge 의 조각 표시를 읽고 샘플러 이름은 표시가 없는 호출자의 대체 경로에서만 쓰지만, 그 경로를 위해
  이 확장 폴더 자리에서도 그 순서를 지킵니다.
- 내장 NegPiP 스위치(2026-10-02) — `scripts/negpip.py` 에 Forge 설정 `sam3_builtin_negpip_enabled`(Settings → SAM Extra
  NegPiP)를 더했습니다.
  - 기본 켬이며, 그때는 예전과 같이 음수 가중치가 있으면 자동으로 켜집니다.
  - 끄면 `process_batch` 가 `reset()`(패치 해제) 뒤 바로 돌아가고 infotext 에 `SAM Extra NegPiP enabled: False` 를 남깁니다.
  - `NegPiP` infotext 키는 내장본이 쓴 것만 지웁니다.
  - `ui()` 는 그대로 `None`(스크립트 인수 0개)이고, 따로 설치한 sd-forge-negpip 는 이 스위치의 대상이 아닙니다.
  - 2026-10-02 검토 제안(`docs/review_proposals_20261002/generation/`)을 편입한 것입니다.
- `coexist.py` 는 이 확장이 새로 쓴 코드(GPL-3.0-only)로, 상류 코드가 아닙니다.
- 나머지 — SD1/SDXL 훅의 `Counter`·`patch_sd_negpip`·`_main_forward`, Anima 의 조건 외 훅, 프롬프트 파싱, `NEG_PATTERN`, 스크립트
  제목 `NegPiP`·스크립트 UI 없음 — 는 상류 `0585496` 과 같습니다.
  `NEG_PATTERN` 은 상류 `75b81b4` 에서 바뀐 것으로, 그 전 판(`b3673ce`)을 쓰던 설치와는 잡는 음수 항이 다릅니다
  (예: `(smile), (aqua hair:-1)` 에서 예전엔 전체, 이제 `(aqua hair:-1)` 만) — 상류의 수정이라 그대로 둡니다.
  `tests/test_negpip_vendor.py` 가 바꾸지 않은 함수 본문의 해시를 상류 `0585496` 과 대조합니다.
- 상류의 `README.md`·`img/`·`.gitignore` 는 포함하지 않았습니다.

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

## ComfyUI-Colorcraft (MIT)

`sam3ext/colorcraft/` 는 [muerrilla/ComfyUI-Colorcraft](https://github.com/muerrilla/ComfyUI-Colorcraft)
(commit `d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf`, 2026-09-01) 의 계산을 편입한 것입니다. 원 라이선스는 MIT
(Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla)) 이며 편입한 파일마다 머리에 원래 고지와 바꾼 내용을 적었습니다.

- 그대로 옮긴 것: `lib_colorcraft/vectors.py` → `vectors.py`, `masking.py` → `masking.py`, `debug.py` → `debug.py`,
  `schedule.py` 의 `make_schedule` 과 `nodes.py` 35-59줄 `sigma_to_value` → `schedule.py`, `color.py` 의 대비·색 이동·색
  기준점 함수 → `color.py`, `basis.py` 의 `load_basis`·`resolve_dev` 와 krea2·zimage 보정값 → `basis.py`,
  `vectors/colorcraft-krea2.safetensors`·`colorcraft-zimage.safetensors` → `data/`(바이트 그대로).
- 옮겨 쓴 것: `nodes.py` 의 post-CFG 본문(543-728줄) → `engine.py`(색 기준점을 생성 전에 만들어 넘기는 것, family 별 VAE 배율,
  미리 구한 스케줄 값 — 세 가지 치환만, 테스트가 구문 트리로 대조), Forge 스크립트의 Debug 패널(112-135·820-905·956-966줄) →
  `debug_panel.py`, 패널의 범위·스택 구조·infotext 형식 → `spec.py`·`ui.py`, `javascript/colorcraft-sliders.js` →
  `javascript/colorcraft_sliders.js`(범위 선택자를 이 확장의 아코디언으로, 페이지가 바뀔 때마다 숫자 칸 전체를 다시 훑던 감시를
  아코디언 두 개 안의 바뀐 노드만 보도록 — 범위 밖 값을 받는 blur·Enter 처리는 상류 그대로).

Forge Neo 구조와 Flux2 family 는 포크 [aoleg/ComfyUI-Colorcraft](https://github.com/aoleg/ComfyUI-Colorcraft)
(commit `f00066c63c9d8f96abc119cada3b51b36688fcac`, Oleg Afonin — 같은 MIT, LICENSE 에는 원본 고지
"Copyright (c) 2026 Sahand Ahmadiantehrani (Muerrilla)")를 따랐습니다: `lib_colorcraft/core.py` 의 `to_model_space`(latent 차원을 인자로)·
`family_for_latent_format`·flux2 보정값·VAE 배율 표, `vectors/colorcraft-flux2.safetensors`(바이트 그대로), UNet 복제본의
post-CFG 로 붙이는 `scripts/colorcraft_neo.py` 의 방식. flux2 보정값에는 원본이 뒤에 더한 `detail_scale`(4.0, Flux2 에서
재지 않음)을 붙였습니다.

이 확장에서 새로 쓴 것: Forge 훅(`hook.py` — 소유 표시가 붙은 붙이기·떼기, img2img·hires σ 구간, `vae.encode` +
`process_in` 색 기준점과 요청 사이 캐시, 오류 시 입력 그대로와 status, fp16/bf16, CompVis 샘플러의 원본 Forge 스텝 카운터), 노드
종류를 고르는 Type·Pass·infotext(이 확장의 키와 원본·포크 키 읽기), 공유 편집기(`ui.py` 의 편집기 한 벌과 선택 줄 — 상류는 노드마다
탭, `panel_state.py`, `javascript/colorcraft_editor.js` · 생성 파일 `javascript/colorcraft_schema.js`), `scripts/colorcraft.py`. 패널의
범위·스택 구조는 여전히 상류에서 온 것입니다(위 `spec.py`·`ui.py`).

`tests/_origin_colorcraft/`(원본 `nodes.py`·`lib_colorcraft/`·`scripts/colorcraft.py`·`LICENSE`)와
`tests/_origin_colorcraft_fork/`(포크 `core.py`·`params.py`·`spec.py`·`LICENSE`)는 대조 테스트용 사본입니다(머리 주석만
붙였고 테스트가 파일마다 SHA-256 을 고정). 확장 실행 코드는 이 사본을 import 하지 않습니다. MIT 고지 전문은 이 문서 끝에
있습니다. 이 확장의 GPL-3.0-only 라이선스는 이 코드의 상류 조건을 대체하지 않습니다.

## Anima Safe PAG — 적용 구간·번호 파싱 (MIT)

`sam3ext/guidance/sigma_window.py` 는 [iljung1106/comfyui-anima-safe-pag](https://github.com/iljung1106/comfyui-anima-safe-pag)
(commit `905b0107d1f924fc6acbcac3b6a879b566ff671c`) 의 `__init__.py` 9-34줄(`_sigma_to_float`·`_sigma_active`·
`_percent_range_to_sigmas`)을 옮긴 것입니다. 달라진 곳은 모델 대신 `percent_to_sigma` 함수를 받는 것과, torch 없이 도는
텐서 판별 하나입니다. `scripts/anima_safe_pag.py` 의 `_parse_blocks`·`_parse_attention_heads` 는 같은 파일 37-64줄처럼
역범위를 뒤집고, scale 범위 0~100 은 201줄을 따릅니다. 대조 테스트 `tests/test_anima_safe_pag_origin.py` 는 9-64줄을 그대로
담고 있습니다. 원 라이선스는 MIT(Copyright (c) 2026, 저장소 LICENSE 에 이름 없음)이며 MIT 고지 전문은 이 문서 끝에
있습니다. 나머지 PAG 구현(배치 확장·어텐션 패치·post-CFG)은 아래 표처럼 Forge 훅으로 다시 작성한 것입니다.

## ComfyUI-DCW — 대조 테스트용 원본 사본 (GPL-3.0)

`tests/_origin_comfyui_dcw.py` 는 [namemechan/ComfyUI-DCW](https://github.com/namemechan/ComfyUI-DCW)
(commit `66aaf9dddb03bad031c1e8443e255a811008e477`) 의 `dcw_node.py` 전체를 바꾸지 않고 옮긴 것입니다(머리 주석을 붙이고
줄끝을 CRLF 에서 LF 로 바꿨을 뿐이며, 테스트가 SHA-256 을 고정합니다). `tests/test_dcw_origin.py` 가 대조 오라클로만 불러
쓰고 확장 실행 코드는 이 파일을 import 하지 않습니다. 같은 테스트에는 ComfyUI(`comfyanonymous/ComfyUI@387f98aa`)
`comfy/samplers.py` 592-605줄의 `cfg_function` 이 그대로 들어 있습니다. 두 원본 모두 이 확장과 같은 GPL-3.0 입니다. 실행
코드의 DCW·RDC·CWM·SMC(`sam3ext/guidance/dcw.py`, `sam3ext/guidance/cwm_smc.py`)는 다시 작성한 구현이고, 기본값·범위·켜짐
규칙·CFG 1 동작·외부 CFG 함수가 있을 때의 처리는 원본 `INPUT_TYPES` 와 `patch()` 를 따릅니다.

## CNS Sampler Patch — 노이즈 재색칠 (GPL-3.0)

`sam3ext/guidance/cns.py` 의 `color_noise_wavelet` 은
[namemechan/comfyui-cns_sampler_patch](https://github.com/namemechan/comfyui-cns_sampler_patch)
(commit `42278b138284f7a8685ef174af0a50fe03246dd0`) 의 `cns_sampler_patch.py` 160-288줄(`_subband_energy`,
`color_noise_wavelet`)을 단계별로 옮긴 것입니다. 달라진 곳은 디버그 출력 블록을 뺀 것, Haar 도우미로 이 확장의
`sam3ext/guidance/haar.py`(같은 계산)를 쓰는 것, `x_t` 를 노이즈와 같은 장치로 옮기는 것입니다. `tests/_origin_cns_sampler_patch.py`
는 같은 파일 전체를 바꾸지 않고 옮긴 대조 오라클입니다(머리 주석을 붙이고 줄끝을 CRLF 에서 LF 로 바꿨을 뿐이며,
`tests/test_cns_origin.py` 가 SHA-256 을 고정합니다). 원본은 이 확장과 같은 GPL-3.0 입니다. Forge 샘플러에 거는 부분(k-diffusion
노이즈 원천 패치, `p.sampler.callback_state` 로 스텝 x 를 잡는 것)은 이 확장이 다시 작성한 것이고, 기본값·범위는 원본
`INPUT_TYPES`(391-439줄)를 따릅니다.

## Anima DAVE — 초반 스텝 게이트 (MIT)

`sam3ext/guidance/dave_gate.py` 는 [sorryhyun/ComfyUI-Anima-DAVE](https://github.com/sorryhyun/ComfyUI-Anima-DAVE)
(commit `83143e8d84768e25f72755ec00ea00ded07ee06e`) 의 `nodes.py` 91-106줄(`_current_step`)과 165-166줄·199-208줄(블록
켜짐 기준과 tau 게이트)을 옮긴 것입니다. 달라진 곳은 σ 스케줄과 현재 σ 를 `transformer_options` 대신 인자로 받는 것,
Forge 의 전체 σ 목록에서 실제로 도는 꼬리만 쓰는 것, σ 를 스케줄 dtype 으로 맞추는 것, 목록 입력용 파이썬 `isclose`,
forward 마다 한 번만 찾는 캐시입니다. 기본 블록 `8-18` 은 같은 저장소의 `dave_alpha.npz` 마스크입니다. 대조 테스트
`tests/test_dave_origin.py` 는 위 줄을 그대로 담고 있습니다. 원 라이선스는 MIT(Copyright (c) 2026 Seunghyun Ji)이며
MIT 고지 전문은 이 문서 끝에 있습니다. DC 감쇠 계산(`sam3ext/guidance/dave.py`)과 블록 래퍼는 이 확장이 다시 작성한
것입니다.

## Smooth Progress — 진행 막대 (MIT)

`sam3ext/progress_api.py`·`scripts/appearance_progress_bar.py`·`javascript/progress_bar.js` 와 `style.css` 의
`sam3-progress` 블록은 [diamfang/sd-webui-smooth-progress](https://github.com/diamfang/sd-webui-smooth-progress)
(commit `7fe58101ae315af1b5e6eda6080e72ac45bfe846`, 코드는 `2b966fc` 와 같음) 를 옮긴 것입니다:
`scripts/smooth-progress.py` 의 스텝별 ETA(`_step_eta`, 21-54줄)와 `/smooth-progress/api` 처리(57-112줄),
`javascript/smooth-progress.js` 의 부드러움 세 방식·글자 형식 여덟 가지·글자 위치·끝난 뒤/중단 표시·높이·사라지는
시간. 원 라이선스는 MIT(Copyright (c) 2026 diamfang)이고 세 파일 머리에 상류 고지 전문을 그대로 두었으며, MIT 고지
전문은 이 문서 끝에도 있습니다. 대조 테스트 사본 `tests/_origin_smooth_progress.py` 는 `scripts/smooth-progress.py`
전체를 바꾸지 않고 옮긴 것입니다(머리 주석만 붙였고 `tests/test_progress_origin.py` 가 SHA-256 을 고정합니다).

이 확장에서 바꾼 부분:

- 시작·끝: 'generate' 처럼 보이는 클릭으로 시작을 짐작하고 인증 없는 경로를 100ms 마다 계속 묻던 것을, Forge
  `requestProgress`(javascript/progressbar.js:77) 감싸기로 바꿨습니다. 인자 여섯 개와 원래 `atEnd`·`onProgress` 는
  그대로 넘기고, 그 탭이 시작한 작업만 시작부터 끝까지, 페이지가 보일 때만 묻습니다.
- 경로: `GET /smooth-progress/api` → `GET /sam-extra/progress?id_task=`. 다른 sam-extra 경로와 같은 로그인·
  `--api-auth` 의존성과 `X-SAM3-Notebook: 1` 헤더, `Cache-Control: no-store`, OpenAPI 스키마 제외. 작업 하나의
  실행·대기·완료와 대기열 위치를 답합니다.
- 진행률·ETA: 한 패스가 아니라 Forge 기본 막대와 같은 작업 전체(`job_no`/`job_count`). ETA 는 패스 종류(Hires
  첫 패스·hires 패스)별 스텝 평균과 끝난 패스 시간으로 잡은 작업 전체의 남은 시간이고, 새 패스는 스텝 0 을 못
  봐도 스텝 시계를 새로 시작합니다. 모르는 ETA 는 0.1 초 대신 `null`, Skip 은 중단으로 치지 않습니다.
- 설정: ⚙️ 팝오버와 localStorage 대신 Forge Settings(`sam3_progress_*`, SAM Extra Progress Bar), 기본 꺼짐.
  그라데이션·움직이는 색 프리셋과 애니메이션 속도는 빼고(디자인 규칙: 그라데이션·빛 번짐 없음) 단색 네 가지와
  테마 강조색·성공색·직접 지정 색만 남겼습니다. 높이는 10~50px 로 제한하고, 중단 표시 네 가지를 모두 고를 수
  있습니다(상류 슬라이더는 세 번째까지).
- 화면: 테마 변수 색, 빛 번짐(filter·drop-shadow) 없는 두 겹 글자, 지속 텍스트 노드의 `.data` 만 바꿈,
  prefers-reduced-motion 이면 보간 없음. Forge 기본 막대는 지우지 않고 CSS 로 숨깁니다. ETA 를 모를 때
  Smooth > Accurate 가 99% 로 내달리던 것을 고쳤습니다.
- sd-webui-smooth-progress 가 함께 설치돼 있으면(`#spb-dynamic-css`) 이 막대는 켜지지 않습니다.

## ComfyUI TSR — Temporal Score Rescaling (GPL-3.0)

`sam3ext/guidance/tsr.py` 의 식(`rescale_factors`·`apply_tsr`)은 ComfyUI(`comfyanonymous/ComfyUI`, commit
`174208df6ba033e5b4784c94d83d065c0d4165a4`) `comfy_extras/nodes_eps.py` 의 `TemporalScoreRescaling` 노드(PR #10351)를
옮긴 것입니다. 기본값 k 0.95·sigma 1.0 과 API 입력 범위 0.01–100 은 노드와 같습니다. 달라진 곳은 다음과 같습니다.

- Forge 에 `model_sampling` 이 없어 flow/eps 판별과 half-log-SNR 을 `sam3ext/guidance/sigmas.py` 에서 직접 계산합니다.
- 스칼라 σ 대신 행마다 r·α 를 구합니다.
- σ ≤ 0 행과 flow σ ≥ 1 행은 노드의 '보정 없음' 경우처럼 그대로 둡니다.
- UI 슬라이더 범위를 좁혔습니다.

원본은 이 확장과 같은 GPL-3.0 입니다.

## ComfyUI — Laplace 스케줄러 (GPL-3.0)

`sam3ext/extra_schedulers/schedulers.py` 의 `get_sigmas_laplace` 는 ComfyUI(`comfyanonymous/ComfyUI`, commit
`36c0b0a687e5e6d7b55e3e61ab24262ffc0f2508`) `comfy/k_diffusion/sampling.py` 52-59줄을 바꾸지 않고 옮긴 것입니다
(Hang et al., "Improved Noise Schedule for Diffusion Training", [arXiv:2407.03297](https://arxiv.org/abs/2407.03297) 의
Laplace 스케줄). Forge 의 다른 스케줄러처럼 마지막 0 을 붙이는 것만 이 확장이 더했습니다(노드는 `steps` 개만 내고
SamplerCustom 이 sigma_min 에서 멈춤). μ/β 의 기본값·범위는 같은 commit `comfy_extras/nodes_custom_sampler.py` 112-133줄
`LaplaceScheduler` 노드를 따릅니다. `tests/_origin_comfyui_laplace.py` 는 이 두 부분을 그대로 담은 대조 오라클이고(머리 주석과
구획 줄만 더함, `tests/test_extra_schedulers_origin.py` 가 구획마다 SHA-256 을 고정), 확장 실행 코드는 이 파일을 import 하지
않습니다. 원본은 이 확장과 같은 GPL-3.0 입니다. 같은 기능의 나머지 스케줄러(Cosine·CosineExponential blend·Phi·Karras
Dynamic·custom)는 아래 재구현 표에 있습니다.

## HiFlow — 흐름 정렬 (Apache-2.0)

`sam3ext/guidance/hiflow.py` 는 [Bujiazi/HiFlow](https://github.com/Bujiazi/HiFlow)의 코드를 옮겨 다시 쓴 것입니다.

- 정렬 식: `flux_pipeline_hiflow.py`, commit `31cc2b1c515195d8bfee002d3da58ac7c7773fef`
- Butterworth 저역 통과 마스크: `utils.py`, 같은 commit
- 기본값(α 1.0, β 0.5, cutoff 0.2): `run_hiflow.py`, commit `da351a8ef036384a42485744ba47bc9c1d882b94`

원 파이프라인은 Euler 속도를 고칩니다. 이 확장은 같은 식을 모델 호출마다 x0 를 바꾸는 꼴로 다시 쓰고(x_t 가 상쇄돼 같은
결과), Forge hires fix 와 1차 패스 기록(`sam3ext/guidance/trajectory.py`, 이 확장의 코드)에 연결했습니다. 바꾼 내용은
Apache-2.0 4(b) 에 따라 파일 머리 주석에 적었습니다. 상류 `LICENSE` 는 Apache License 2.0 이며 NOTICE 파일은 없습니다
(2026-10-02 저장소 목록 확인). 공식 실행의 모델 쪽 고해상도 기법(NTK RoPE, 비례 attention, swin padding)은 가져오지
않았습니다. 이 확장의 GPL-3.0-only 라이선스는 이 코드의 상류 조건을 대체하지 않습니다.

## ComfyUI-NAFNet-Residual — 대조 테스트용 잔차 함수 사본 (Apache-2.0)

`tests/test_vae_degrid.py` 의 `_apply_residual_mode` 는
[DraconicDragon/ComfyUI-NAFNet-Residual](https://github.com/DraconicDragon/ComfyUI-NAFNet-Residual)
(commit `e15460d3724c70d428e333518b58eb7ba8903d76`) 의 `patch.py` 48-62줄을 바꾸지 않고 옮긴 대조 오라클입니다. 확장 실행
코드는 이 사본을 import 하지 않습니다. 실행 코드의 잔차 적용(`sam3ext/vae_degrid.py` 의 `select_residual`·`apply_residual`·
`finalize`)은 같은 식(`nafnet_node.py` 132-149줄)을 다시 작성한 것이고, 강도 배율은 이 확장이 더했습니다. 상류
`LICENSE.txt` 는 Apache License 2.0 이며 저작권자 이름과 NOTICE 파일은 없습니다. 모델 가중치
([DraconicDragon/NAFNet-VAE-DeGrid](https://huggingface.co/DraconicDragon/NAFNet-VAE-DeGrid), Apache-2.0)는 저장소에 없고
사용자가 받습니다. NAFNet 구조는 Forge venv 의 spandrel(`spandrel/architectures/NAFNet`, megvii-research/NAFNet MIT)을
그대로 불러 쓰며 이 저장소에 넣지 않았습니다. 타일 합치기(`tiled_residual`)는 ComfyUI `comfy/utils.py` 의 `tiled_scale`
(GPL-3.0, Forge `backend/patcher/vae.py` 에 옮겨진 v0.3.64 판)과 같은 위치·feather 가중치로 다시 작성했고, 테스트는 Forge 가
확장 옆에 있을 때만 그 함수를 AST 로 꺼내 대조합니다(사본 없음). 이 확장의 GPL-3.0-only 라이선스는 위 코드·가중치의
상류 조건을 대체하지 않습니다.

## SPEED — Spectral Progressive Diffusion (MIT)

`sam3ext/speed/`(`spectral.py`·`schedule.py`·`runner.py`·`forge_host.py`)와 `scripts/anima_speed.py` 는 세 상류를 옮겨 다시
작성한 것입니다.

- [howardhx/speed](https://github.com/howardhx/speed) (공식, commit `ca7801c9bdffe681742e9592345bcf4885959be5`) 의 `utils.py`
  (`power_spectrum`·`activation_time`·`delta_optimal_transitions`·`kappa`·`align_timestep`·DCT/FFT/DWT 확장·`validate_scales`)와
  `comfyui/speed_sampler.py`(구간 나눔·전환 σ 패치·첫 DCT 축소) — Copyright (c) 2026 Howard Xiao.
- [aoleg/ComfyUI-SPEED](https://github.com/aoleg/ComfyUI-SPEED) (ruwwww/ComfyUI-SPEED 의 포크, commit
  `a8873591a27f2c1e086a2caf546f9b6aeec62b81`) 의 `speed_core.py`(프리셋 표·Adaptive delta·`sigma_divisor`/neo_shift·
  `_resolve_transitions`)·`spectral_utils.py`(`equivalent_delta`·`reference_coarse_fraction`)·`scripts/speed_forge.py`(Forge 스크립트·
  가드) — Copyright (c) 2026 A. Izzuddin Al Faruq (포크의 Forge/프리셋 작업은 Oleg Afonin).
- [sorryhyun/ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler) (commit
  `b46a364aec3b161b889c9cc26cd976a49eb537ae`) 의 `_vendor/networks/spd_core.py`(DCT 행렬 캐시·`dct_lowpass_init`·`spectral_expand`)와
  `spd.py`(respace 반복·`resolve_spd_schedule`) — Copyright (c) 2026 sorryhyun.

aoleg 포크의 README 는 공식 저장소를 BSD 3-Clause 로 적지만, 고정한 commit 의 공식 `LICENSE` 는 MIT 이고 이 확장은 그것을
따릅니다.

이 확장에서 바꾼 부분(파일 머리 주석에 같은 목록): scipy·numpy 변환·PyWavelets 대신 장치 위 float64 DCT 행렬(크기별 캐시)·`torch.fft`·
Haar 합성 · 공식 ca7801c9 와 같은 FFT 노이즈 수정(aoleg 판의 `/√2` 제거) · 이미지 시드별 확장·SDE·Brownian 노이즈(배치 = 단일 시드) ·
저해상도 스텝이 없으면 적용하지 않음, 마지막 전환이 없으면 건너뜀 · img2img·Hires 시작 latent 의 flow 형태 보정(설정, 끄면 원본) ·
두 방식(transition·respace)을 한 실행기로, respace 는 고른 샘플러를 구간마다 · Forge `sampling_sigmas` 갱신 · 가드(마스크·레퍼런스
latent(Anima·Flux Kontext·Flux.2 Klein·Qwen-Image-Edit·Krea 2)·Wan I2V·PiD·ControlNet·비 flow 모델·Spectrum Integrated·샘플러) ·
DWT 의 r ≠ 2 를 미리 거절 · 상태를 클래스가 아닌 요청(p)에 보관 · infotext·XYZ·설정.

대조 테스트용 원본 사본(확장 실행 코드는 import 하지 않음, `tests/test_speed_origin.py` 가 SHA-256 을 고정): `tests/_origin_speed_core.py`
(aoleg `speed_core.py` 무수정), `tests/_origin_speed_spectral_utils.py`(aoleg `spectral_utils.py` — FFT 한 줄만 공식 ca7801c9 처럼 고침,
머리 주석에 적음), `tests/_origin_speed_official_utils.py`(공식 `utils.py` 무수정), `tests/_origin_spd_core.py`·`tests/_origin_spd_sampler.py`
(sorryhyun `spd_core.py`·`spd.py` 무수정). `tests/test_anima_speed_script.py`·`test_speed_runner.py`·`test_speed_schedule.py` 는 aoleg
`tests/test_forge_script.py`·`tests/test_math.py` 의 경우를 옮겨 고친 것입니다. 원 라이선스는 모두 MIT 이며 MIT 고지 전문은 이 문서 끝에
있습니다. 이 확장의 GPL-3.0-only 라이선스는 이 코드의 상류 조건을 대체하지 않습니다.

## ComfyUI SamplerER_SDE — ER SDE 잡음 척도 (GPL-3.0)

`sam3ext/extra_samplers/er_sde.py` 의 `reverse_time_sde_noise_scaler`·`ode_noise_scaler`·`er_sde_kwargs` 는 ComfyUI
(`comfyanonymous/ComfyUI`, commit `36c0b0a687e5e6d7b55e3e61ab24262ffc0f2508`) `comfy_extras/nodes_custom_sampler.py` 585-633줄
`SamplerER_SDE` 의 두 잡음 척도(h(λ)=λ^(η+1), h(λ)=λ)와 ODE 규칙(`solver_type == "ODE" or eta == 0` → `s_noise = 0`),
`sample_er_sde` 에 넘기는 세 인자를 옮긴 것입니다. 풀이 자체는 Forge 내장 `sample_er_sde`(ComfyUI v0.3.75 사본)를 그대로 부릅니다.
달라진 곳은 η·stage 를 노드 입력 대신 Extra Samplers 아코디언에서 받고 노드 범위(stage 1–3, η 0–10)로 자르는 것과, 노드의
ER-SDE 선택지는 Forge 내장 ER SDE 이므로 다시 두지 않은 것입니다. `tests/_origin_comfyui_er_sde.py` 는 위 585-633줄과 같은
커밋 `comfy/k_diffusion/sampling.py` 78-88·152-176·1592-1656줄을 바꾸지 않고 담은 대조 오라클입니다(블록마다 SHA-256 고정,
첫 블록 앞의 테스트용 대역 코드는 상류 코드가 아님). 원본은 이 확장과 같은 GPL-3.0 입니다. 바탕 논문: Cui et al.,
"Elucidating the Solution Space of Extended Reverse-Time SDE for Diffusion Models", [arXiv:2309.06169](https://arxiv.org/abs/2309.06169).

## ComfyUI-Extra-Samplers — DPM++ 4M SDE (BSD-3-Clause)

`sam3ext/extra_samplers/dpmpp_4m_sde.py` 의 `sample_dpmpp_4m_sde` 는 [Clybius/ComfyUI-Extra-Samplers](https://github.com/Clybius/ComfyUI-Extra-Samplers)
(commit `52eac1b7c847d2727e0ca93ca26d9ffd77029daa`) `extra_samplers.py` 435-524줄 `sample_clyb_4m_sde_momentumized` 를
`clyb_4m_sde_momentumized` 항목이 쓰는 momentum 0 으로 옮긴 것입니다(Copyright (c) 2024, Clybius). 차수 선택, 비율, `d1`/`d2`,
φ 식은 원본과 같습니다. 바꾼 곳: Forge DPM++ 3M SDE 처럼 half-log-SNR 로 써서 flow 모델(α=1−σ)을 지원(`α_t` 곱, 첫 σ 보정),
momentum 항 제외(0 에서 항등), 잡음은 Forge 규칙(`eta > 0 and s_noise > 0`)과 Forge 의 시드별 Brownian 트리. 바뀐 내용은 파일
머리 주석에 적었고 BSD 고지 전문을 파일 머리에 그대로 두었습니다. `tests/_origin_clybius_extra_samplers.py` 는 같은 파일 전체를
바꾸지 않고 옮긴 대조 오라클입니다(머리 주석만 더함, SHA-256 고정). BSD 3-Clause 전문은 이 문서 끝에 있습니다. 바탕:
DPM-Solver++ (Lu et al., [arXiv:2211.01095](https://arxiv.org/abs/2211.01095)). 이 확장의 GPL-3.0-only 라이선스는 이 코드의 상류
조건을 대체하지 않습니다.

## Euler-Smea-Dyn-Sampler — Euler (SMEA) Dy 보조 스텝 (Apache-2.0)

`sam3ext/extra_samplers/euler_dy.py` 는 [Koishi-Star/Euler-Smea-Dyn-Sampler](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler)
(commit `d98a504c8419be5274068ad91ca6bbf2e13635a8`) `smea_sampling.py` 의 `dy_sampling_step`·`sample_euler_dy`·`smea_sampling_step`·
`sample_euler_smea_dy`·`_Rescaler`(WebUI 갈래)를 옮겨 바꾼 것입니다(상류 LICENSE 의 저작권 줄 "Copyright 2024 KBlueLeaf",
NOTICE 파일 없음). 보조 스텝의 위치·해상도·화소 선택·홀수 크기 처리·재잡음 식(`x − eps·sqrt(σ̂²−σ²)`, 매 스텝 잡음 추출)은
원본과 같습니다. Apache-2.0 4(b) 에 따라 파일 머리에 적은 변경 사항: 모든 Euler 갱신을 Forge 의 CFG++ 갱신으로 바꿈,
**churn 규칙을 원본의 `max(s_churn/N, √2−1)` 에서 k-diffusion 의 `min` 으로 바꿈**(기본 설정에서 재잡음 없음), flow 모델 churn 을
eps 등가 잡음 수준에서 계산, 5차원 latent, Forge 의 4·5차원 마스크·`image_cond`·Anima 레퍼런스 latent 크기 맞춤, 보조 평가 표시,
미리보기 latent 복구, 보조 스텝이 Forge 의 스텝 카운터(`CFGDenoiser.step`)를 늘리지 않음(원본은 보조 스텝마다 한 칸씩 밀림),
보조 스텝 없이 도는 모드(Spectrum Integrated·Wan 2.2 I2V `concat_latent`·PiD `lq_latent`·`extra_concat_condition` 요청).
`tests/_origin_koishi_smea_sampling.py` 는 같은 파일 전체를 바꾸지 않고 옮긴 대조 오라클입니다(머리 주석을 붙이고 줄끝을 CRLF 에서
LF 로 바꿨을 뿐, SHA-256 고정). CFG++ 갱신은 Forge 의 `sample_euler_ancestral_cfg_pp` 와 같고, 바탕 논문은 CFG++ (Chung et al.,
[arXiv:2406.08070](https://arxiv.org/abs/2406.08070))와 Karras et al. 2022 Algorithm 2 ([arXiv:2206.00364](https://arxiv.org/abs/2206.00364))입니다.
Apache License 2.0 전문은 이 문서 끝에 있습니다. 이 확장의 GPL-3.0-only 라이선스는 이 코드의 상류 조건을 대체하지 않습니다.

## 코드를 편입하지 않은 재구현 (참고 출처)

아래 기능은 상류 코드를 파일째 가져오지 않고 이 확장 코드로 다시 작성했습니다. 전체 목록(DCW·DAVE·Modulation
Guidance 등)은 README 의 출처 / 크레딧 절과 [docs/GUIDANCE.md](docs/GUIDANCE.md) 의 크레딧 표에 있습니다. 라이선스를
확인하지 못한 것은 '미확인' 으로 적습니다.

| 기능 | 이 확장 파일 | 원저작 출처 | 원 라이선스 | 형태 |
|---|---|---|---|---|
| Anima VAE DeGrid 잔차 모드·타일 | `sam3ext/vae_degrid.py` | [DraconicDragon/ComfyUI-NAFNet-Residual](https://github.com/DraconicDragon/ComfyUI-NAFNet-Residual) `nafnet_node.py` 87-149줄, ComfyUI `comfy/utils.py` `tiled_scale` | Apache-2.0 · GPL-3.0 | 식·타일 위치·OOM 재시도를 같게 다시 작성(대조 테스트는 위 절). 16 배수가 아닌 타일의 반사 패딩은 저자의 NAFNet-c `infer.py`(`F.pad(..., mode="reflect")`)와 같은 방식을 다시 작성(사본 없음) |
| Anima Safe PAG | `scripts/anima_safe_pag.py` | [iljung1106/comfyui-anima-safe-pag](https://github.com/iljung1106/comfyui-anima-safe-pag) (ComfyUI 노드), PAG 논문 [arXiv:2403.17377](https://arxiv.org/abs/2403.17377) | MIT | Anima 배치 확장·블록 선택을 이식하고 Forge 훅으로 다시 작성(적용 구간·파싱은 위 절처럼 편입) |
| Adaptive SMC | `sam3ext/guidance/cwm_smc.py` `apply_smc_adaptive` | [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora) `library/inference/corrections/smc_cfg.py` (같은 코드가 sorryhyun/ComfyUI-Spectrum-KSampler 에도 있음), CFG-Ctrl 논문 [arXiv:2603.03281](https://arxiv.org/abs/2603.03281) | MIT(Copyright (c) 2026 Seunghyun Ji) | 속도 공간 식과 기본값(α 0.2, λ 5)을 따르고 x0 공간에서 다시 작성(`σ_t/σ_prev` 보정, 원본 식 대조 테스트). 코드 사본 없음 |
| S²-Guidance | `sam3ext/guidance/s2.py` | [arXiv:2508.12880](https://arxiv.org/abs/2508.12880) 식 4·알고리즘 1 | 공식 코드 없음 | 논문 식 재구현 |
| Momentum Guidance | `sam3ext/guidance/history.py` | [arXiv:2602.20360](https://arxiv.org/abs/2602.20360) 식 12–13 | 공식 코드 없음 | 논문 식 재구현 |
| HiGS | `sam3ext/guidance/history.py` | [arXiv:2509.22300](https://arxiv.org/abs/2509.22300) 식 6·8, 알고리즘 2–3 | 논문 코드는 arXiv 라이선스(사용 안 함) | 논문 식 재구현 |
| Anima Optimal Scale | `scripts/anima_cfg_optimal_scale.py` | CFG-Zero* [arXiv:2503.18886](https://arxiv.org/abs/2503.18886) optimized-scale 식 | — | 식만 독립 구현(zero-init 제외). 2026-10-02 검토 제안(`docs/review_proposals_20261002/generation/`)을 편입 |
| PAG 강도 곡선 | `scripts/anima_safe_pag.py` `_pag_envelope_factor` | — | — | 이 확장의 자체 실험(논문 기법 아님) |
| Extra Schedulers — Cosine · CosineExponential blend · Phi | `sam3ext/extra_schedulers/schedulers.py` | 공개된 식(README 의 Extra Schedulers 표에 적음; Phi 는 황금비 φ) — 이름만 [aoleg/Neo_ExtraSchedulers](https://github.com/aoleg/Neo_ExtraSchedulers) README 에서. Phi 의 착상(황금비 지수)은 Extraltodeus 의 Golden Scheduler([sigmas_tools_and_the_golden_scheduler](https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler), 위 README 가 출처로 적음) | 상류 라이선스 없음(코드 사용 안 함) · Golden Scheduler 는 크레딧만(코드 열람·사용 안 함) | 식 재구현. CosineExponential 의 지수 부분은 k-diffusion `get_sigmas_exponential` 과 같은 식 |
| Extra Schedulers — Karras Dynamic | `sam3ext/extra_schedulers/schedulers.py` | Karras et al. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364) 램프 + 스텝마다 ρ + 2cos(2πi/n) — 이 변형의 출처는 미확인(위 README 가 "yoinked-h" 를 적음) | 미확인(코드 사용 안 함) | 식 재구현(오버플로 없는 같은 값의 꼴, 마지막 스텝은 정확히 σmin), ρ 는 Forge 의 rho 설정. 시그마가 도중에 올라가는 ρ(약 4 미만)는 오류로 멈춤 |
| Extra Schedulers — custom (식 · 시그마 목록) | `sam3ext/extra_schedulers/expression.py` · `sigma_list.py` | 변수 이름(m, M, n, s, x, phi)과 목록 규칙은 위 README 의 설명 | 상류 라이선스 없음(코드 사용 안 함) | 이 확장이 새로 쓴 AST 화이트리스트 계산기(eval/exec 없음). 목록 보간은 Forge 의 `sd_schedulers._loglinear_interp` 를 실행 중에 호출(복사 없음) |
| Extra Samplers — flow churn·CFG++ 결합 | `sam3ext/extra_samplers/euler_dy.py` | [EDM](https://arxiv.org/abs/2206.00364) Algorithm 2, [CFG++](https://arxiv.org/abs/2406.08070) | — | eps 등가 좌표 churn 과 Dy 보조 스텝의 CFG++ 결합은 이 확장의 식(코드 사본 없음) |

## 라이선스 없는 저장소 — 이름만 참고 (코드 미사용)

**라이선스 없는 저장소에서 가져온 코드는 없습니다.** [aoleg/Neo_ExtraSchedulers](https://github.com/aoleg/Neo_ExtraSchedulers)
(commit `ca55a59b1df7eda18333ae3a501d69f0fed6b27e`)와 그 원본
[DenOfEquity/webUI_ExtraSchedulers](https://github.com/DenOfEquity/webUI_ExtraSchedulers) 는 라이선스를 공개하지 않았습니다. 이
확장은 두 저장소의 코드를 열어 보지도 쓰지도 않았습니다 — aoleg 저장소의 README(사용자에게 보이는 이름·설명)만 읽었고,
DenOfEquity 저장소는 아무 파일도 열지 않았습니다. README 에서 가져온 것은 infotext 가 서로 붙도록 맞춘 이름과 규칙뿐입니다.

- Extra Schedulers: 스케줄 라벨(별칭 `cosine-exponential blend`·`karras dynamic` 포함), custom 식의 변수 이름(m, M, n, s, x,
  phi), 시그마 목록 규칙, 아코디언 이름. 구현은 공개된 식, 위 논문, ComfyUI(GPL-3.0)의 Laplace 함수로 다시 작성했습니다.
- Extra Samplers: 샘플러 이름(`ER SDE (Reverse-time)`·`ER SDE (ODE)`·`DPM++ 4M SDE`·`Euler Dy CFG++`·`Euler SMEA Dy CFG++`)과
  infotext 키 `ER SDE max stage` 의 규칙, 사용자에게 보이는 동작 설명. 구현은 위 절의 ComfyUI(GPL-3.0)·Clybius(BSD-3-Clause)·
  Koishi-Star(Apache-2.0) 코드와 Forge 의 `sample_er_sde`·CFG++ 갱신을 바탕으로 했습니다.

## MIT License 전문

`sam3ext/anima38/`(Copyright (c) 2026 GumGum10 contributors),
`mcp_server/sam_extra_mcp/forgeneo/`·`server.py`·`service.py` 와 `tests/test_mcp_forgeneo_upstream.py` 의 forgeneo-mcp 코드(Copyright (c) 2026 Eduardo Abreu),
`assets/bpe_simple_vocab_16e6.txt.gz`(Copyright (c) 2021 OpenAI),
`scripts/anima_detail_daemon.py` 의 Detail Daemon 코드(Copyright (c) 2024 Sahand Ahmadian, Copyright (c) 2024 Jonseed),
`sam3ext/colorcraft/`·`javascript/colorcraft_sliders.js` 와 Colorcraft 대조 테스트 사본의 ComfyUI-Colorcraft 코드(Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla); 포크 aoleg/ComfyUI-Colorcraft — Oleg Afonin),
`sam3ext/guidance/sigma_window.py` 와 PAG 대조 테스트의 Anima Safe PAG 코드(Copyright (c) 2026),
`sam3ext/guidance/dave_gate.py` 와 DAVE 대조 테스트의 Anima DAVE 코드(Copyright (c) 2026 Seunghyun Ji),
`sam3ext/progress_api.py`·`scripts/appearance_progress_bar.py`·`javascript/progress_bar.js`·`style.css` 의 진행 막대 블록과 대조 테스트 사본 `tests/_origin_smooth_progress.py` 의 Smooth Progress 코드(Copyright (c) 2026 diamfang),
`sam3ext/speed/`·`scripts/anima_speed.py` 와 SPEED 테스트(원본 사본과 옮겨 온 테스트)의 SPEED 코드(Copyright (c) 2026 Howard Xiao, Copyright (c) 2026 A. Izzuddin Al Faruq, Copyright (c) 2026 sorryhyun)에
적용되는 조건입니다. 저작권 줄은 위 각 절의 것을 넣어 읽습니다.

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

## BSD 3-Clause License 전문

`sam3ext/extra_samplers/dpmpp_4m_sde.py` 와 대조 테스트 `tests/_origin_clybius_extra_samplers.py` 의 ComfyUI-Extra-Samplers 코드에
적용되는 조건입니다(Copyright (c) 2024, Clybius).

```text
BSD 3-Clause License

Copyright (c) 2024, Clybius

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Apache License 2.0 전문

`sam3ext/extra_samplers/euler_dy.py` 와 대조 테스트 `tests/_origin_koishi_smea_sampling.py` 의 Euler-Smea-Dyn-Sampler 코드
(Copyright 2024 KBlueLeaf)에 적용되는 조건입니다. 같은 Apache-2.0 인 Skimmed_CFG 코드, HiFlow 코드, TIPO 모델 코드, Qwen3.5
토크나이저, ComfyUI-NAFNet-Residual 대조 테스트 사본에도 같은 조건이 적용됩니다(저작권 줄은 각 절의 것). 아래는
Euler-Smea-Dyn-Sampler 의 LICENSE 파일 그대로입니다(부록의 저작권 줄 포함). 같은 전문이
<https://www.apache.org/licenses/LICENSE-2.0> 에도 있습니다.

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2024 KBlueLeaf

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
