# Changelog

버전 태그는 GitHub Releases에도 발행됩니다. 아래는 요약이며, guidance/속도 기능의
상세는 [docs/GUIDANCE.md](docs/GUIDANCE.md)를 참고하세요.

## v0.30.0 — Anima 3.8B + 캐릭터 레퍼런스 IP-Adapter + 28/40/52블록 LoRA·DoRA

v0.21.2 이후 쌓인 큰 업데이트입니다. Anima 3.8B(Qwen3.5 / Semantic Connector v2) 런타임을 들여왔고, 캐릭터 레퍼런스 패널(이어붙이기 · IP-Adapter
방식)이 생겼습니다. Anima LoRA 는 Base 1.0(28)·2.9B(40)·3.8B(52) 사이를 자동으로 옮기고, DoRA 합치는 방식을 고를 수 있습니다. 그 밖에 TIPO 프롬프트
확장(🪄)·SAM3 빠른 버튼(🎯)·txt2img 섹션 정리가 추가됐고, SAM3·가이던스·3.8B 생성 시간을 줄였습니다 — 대부분은 결과가 픽셀 단위로 같고, PAG/SEG 의 두 가지 최적화만 잔
디테일이 달라집니다(설정으로 끌 수 있음). 가이던스(Detail Daemon·Safe PAG·Skimmed CFG·DCW(+a)·DAVE·CNS)와 Tile-Repair 는 가져온 원본 ComfyUI
노드·sd-scripts 와 같은 값·범위·적용 구간으로 맞춰 같은 설정에서도 결과가 달라지고, Tile-Repair HTTP API 와 Notebook 메모장이 생겼습니다. 이 확장 전체의 라이선스는
이제 **GPL-3.0-only** 입니다.

괄호 표시: (결과 변화) = v0.21.2 와 같은 설정·시드에서 이미지가 달라짐, (결과 같음) = 이미지 동일, (새 기능) = v0.21.2 에 없던 기능이라 비교 대상 없음, 토글 = 끌 수
있는 설정이 있음(뒤에 기본값), 토글 없음 = 끌 수 없음.

**업그레이드할 때**

- Forge 를 **재시작**하세요. Reload UI 로는 런타임 모듈이 바뀌지 않습니다.
- v0.21.2 와 같은 설정·시드에서 결과가 바뀔 수 있는 곳(자세한 내용은 각 절의 해당 항목):
  - PAG/SEG/SLG 앞쪽 블록 중복 계산 건너뛰기, SEG blur 1D 두 번 — 구도는 같고 잔 디테일만 다릅니다. 끄려면 Settings → **SAM Extra Guidance** 의
    두 설정을 끄세요. infotext 키 `Anima PAG prefix dedup`·`Anima SEG separable blur` 가 없는 예전 infotext 를 붙여 넣으면 현재
    설정(켬)으로 렌더되므로, 예전 이미지를 그대로 재현하려면 둘 다 끄세요.
  - 가이던스를 가져온 원본 ComfyUI 노드와 같게 맞췄습니다(토글 없음). Detail Daemon·Safe PAG·Skimmed CFG·DCW(+a)·DAVE·CNS 의 기본값·범위·적용
    구간(σ 기준)이 원본과 같아져, v0.21.2 는 물론 이전 v0.30 개발 빌드와도 같은 설정에서 이미지가 달라질 수 있습니다. Detail Daemon 은 강도가 ×0.1 이라
    v0.21.2 와 같은 강도는 amount × 10 이고, **Hires Pass** 를 켜지 않으면 hires 패스에는 걸리지 않습니다(v0.21.2 는 두 패스 모두). 기본값은 DCW λ
    0.10/0.02 → 0.05/0.01, CWM α 0.30/0.15 → 0, CNS gamma scale 3.0 → 2.0 이고, RDC 는 tau 기본 0(끔)입니다. API 에서 생략한
    위치 인자도 새 기본값을 씁니다(Custom SMC k 인자 27 은 0.20 → 0.10).
  - SMC/APG/CWM 은 CFG 1 에서 건너뛰고, Safe PAG 의 APG + rescale 자동 끄기도 CFG 1 에서는 적용하지 않습니다(토글 없음). Safe PAG 확장 배치가 OOM
    나면 그 생성의 남은 스텝·배치에서 PAG/SEG/SLG 를 끕니다.
  - SAM3: 인페인트 패스가 의도한 시드를 쓰고, SAM3 를 쓴 뒤의 다른 생성에서 fp32 행렬곱이 더 이상 TF32 로 돌지 않습니다(버그 수정, 토글 없음). API/XYZ 에서 생략한
    키의 기본값이 UI 와 같아졌습니다(`sam3_inpainting_fill`→ `original`, `sam3_unload_after`→True). 범위 밖 설정값은 SAM3 를 끄는 대신
    범위로 맞춰 실행합니다. SAM3 In-flight 인페인트의 내부 패스에서 ADetailer 가 더 돌지 않으므로(바깥 생성의 ADetailer 는 한 번 그대로) ADetailer 와 함께
    쓰던 생성은 결과가 달라질 수 있습니다.
  - 일부 블록만 담은(sparse) Anima LoRA 는 판정 블록 수가 현재 모델과 다르면 순정 Forge 처럼 추측 변환하지 않고 건너뜁니다. 순정처럼 하려면 Settings → **SAM
    Extra LoRA** → `sam3_anima_sparse_lora_forge_guess` 를 켜세요.
  - 3.8B v2 번들 체크포인트는 Qwen3.5 커넥터가 자동으로 켜집니다. 끄려면 `Anima 3.8B (Qwen3.5 / v2)` 아코디언의 Bypass.
  - Tile-Repair 는 sd-scripts 원본처럼 원본 비율을 지키는 Short Side 슬라이더·디코드(uint8 버림)·빈 네거티브를 쓰고, SAM3 ControlNet 에 Tile &
    Repair LLLite 를 고르면 preprocessor 가 늘 `None` 이 됩니다(토글 없음).
- 메모리: 3.8B 는 생성 사이 VRAM 에 최대 약 6~8 GB 를 남기고(`sam3_anima38_keep_resident`), SAM3 'Unload after' 는 모델을 RAM 에 약
  3.4 GB 로 보관합니다(`sam3_unload_keep_in_ram`). 3.8B 샘플링 중에는 커넥터 fp32 상주 (`sam3_anima38_connector_fp32`, VRAM 약
  +1.5 GB, 여유가 있을 때만)와 run 캐시(`sam3_anima38_connector_run_cache`, 최대 512 MB)도 씁니다. 넷 다 기본 켬이고 끌 수 있습니다.
- 업데이트 뒤 첫 시작 때 `ui-config.json` 을 한 번 옮깁니다(Guidance 아코디언 슬라이더, Skimmed CFG `Flip at`, Tile-Repair 네거티브). Forge 는 저장된 슬라이더 값·범위를
  라벨로 다시 적용해서, 라벨이 그대로인 슬라이더에 예전 기본값·범위가 되살아나기 때문입니다. 예전 기본값 그대로인 값만 원본 기본값으로 바꾸고(DCW λ low 0.10 → 0.05·high
  0.02 → 0.01, CWM α low 0.30·high 0.15 → 0, 예전 Enable RDC 가 꺼진 채 저장된 RDC tau 0.15 → 0), 예전 범위 저장값(PAG Attn
  Scale 최대 15, DCW λ high ±0.5, CWM α 최대 1, CNS strength step 0.01·gamma power 최소 0.05, Skimmed CFG `Flip at` step 0.05)은
  지웁니다. Tile-Repair 패널은 예전 Width·Height 슬라이더 키가 남은 설치만 그 키를 지우고, 네거티브가 예전 기본값 `blurry, low quality` 그대로면
  sd-scripts 기본값인 빈 칸으로 바꿉니다. 사용자가 바꾼 값은 남깁니다 —
  새 범위 밖이면 범위로 맞추고(DCW λ high ±0.3, CNS gamma power 0.1 이상), 예전 라벨로 저장된 CNS gamma scale 은 새 라벨로 옮기며, Enable RDC
  를 끈 채 바꿔 둔 RDC tau 도 남으므로 이제 Enable DCW 를 켜면 RDC 가 함께 돕니다(끄려면 tau 0). 바꾸기 전 파일은 같은 폴더에
  `ui-config.json.bak-anima-guidance-<날짜-시각>` 으로 복사하고, 새 파일은 임시 파일을 거쳐 한 번에 바꿔 씁니다. 바꾼 항목은 콘솔 `[AnimaSafePAG]
  ui-config.json migrated …` 아래 줄마다 나오고, 바꿀 것이 없으면 파일을 건드리지 않습니다. Detail Daemon Amount 는 라벨이 바뀌어 예전
  저장값이 적용되지 않으므로 대상이 아닙니다.
- Forge 를 `--api` 또는 `--nowebui` 와 `--api-auth` 로 띄우면 이 확장의 Tile &
  Repair(`/sam-extra/tile-repair…`)·Notebook·메모(`/sam3-notebook…`)·LoRA Manager(`/sam3-lora/*`) 경로에도 `/sdapi` 와
  같은 HTTP Basic 인증이 걸립니다. 이 경로를 부르는 스크립트·앱은 자격 증명을 보내야 하고(없거나 틀리면 401), `--gradio-auth` 로그인도 켜져
  있으면 둘 다 필요합니다.
- `/sam3-lora/config`·`/sam3-lora/spawn` 을 직접 부르는 외부 도구는 이제 다른 확장 경로처럼 헤더 `X-SAM3-Notebook: 1` 을 보내야
  하고(없으면 403), `--gradio-auth` 가 켜져 있으면 로그인 쿠키도 필요합니다(없으면 401). 페이지의 Manage 탭은 Gradio 버튼 브리지를 써서
  그대로 동작합니다.

### Anima 3.8B (Qwen3.5 / Semantic Connector v2)

- **3.8B 런타임 편입 (결과 변화, v2 번들은 자동 켬)**:
  [GumGum10/forge-anima-3.8B](https://github.com/GumGum10/forge-anima-3.8B) (MIT) 런타임을 `sam3ext/anima38/` 로 들여와
  `Anima 3.8B (Qwen3.5 / v2)` 아코디언 한 스크립트로 붙였습니다. v2 번들(safetensors metadata 로 판별)은 생성 때 자동으로 켜지고, **Bypass** 를
  켜면 순정 Anima(0.6B) 로 생성합니다. v1 은 아코디언에서 어댑터·강도·부정 프롬프트 사용을 고릅니다. `qwen35_4b` 가 없으면 설치 전에 확인해 경고하고 순정 Anima 로
  진행합니다. API 는 위치 인자와 SAM3 식 dict 둘 다 받습니다. Qwen3.5 토크나이저는 `assets/qwen35_tokenizer/` 에
  동봉했습니다(`THIRD_PARTY_NOTICES.md`). 아코디언에 상태 확인 버튼이 있습니다.
- **infotext·붙여 넣기**: `Anima38`(v2 bundle / v1 adapter / bypass / off: …)·`Anima38 encoder`·`Anima38 negative` 를
  남기고, PNG Info 붙여 넣기로 Bypass·부정·v1 설정을 되살립니다. infotext 키는 `Anima38 …` 형식입니다(Forge 가 `.` 이 든 키를 읽지 못하므로).
- **다른 확장과의 공존**: v2 조건을 네이티브 list 형식으로 넘기고 run id 는 텐서 마커로 전달해 NegPiP 등 `get_learned_conditioning` 래퍼와 충돌하지
  않습니다. 조건·forward 패치는 플래그로 켜고 끄는 멱등 패치라 다른 확장이 순서를 바꿔 되돌려도 낡은 래퍼가 되살아나지 않고, 샘플링 중 예외로 남은 패치는 다음 생성 시작 때 먼저
  원복합니다. NegPiP 이 아래에 깔린 순서에서는 그 마스킹을 대신 적용합니다. 격리 Forge 두 대(정·역방향 로드 순서)에서 6 케이스 (v2+NegPiP / bypass+NegPiP /
  v2 / bypass / 반복 / hires)가 픽셀 단위로 같았습니다.
- **LoRA 와 커넥터**: LoRA 의 `llm_adapter` 가중치를 v2 경로에도 적용합니다 — 커넥터가 번들 원본으로 만든 자기 `llm_adapter` 사본(약 +0.3 GiB RAM)에
  LoRA 패치를 받고 샘플링 직전마다 그 패스의 LoRA 세트에 맞춥니다. LoRA 세트가 바뀌어 Forge 가 UNet 을 새로 복제해도 커넥터가 Forge 메모리 관리 안에서 돕니다.
- **VRAM 상주 (결과 같음, 토글, 기본 켬)**: 같은 프롬프트 줄이면(시드만 바꾸는 XYZ, batch count, 캐릭터 레퍼런스 후보, ADetailer) 0.6B TE 결과와
  Qwen3.5 인코딩을 줄 캐시에서 꺼내 다시 인코딩하지 않고, batch count 사이에는 설치를 유지합니다. 커넥터(약 1.6 GB)는 생성이 끝나도 VRAM 에 남고, TE(약 1.75
  GB)·Qwen3.5(약 4.45 GiB)는 다음 샘플링에 쓸 여유가 있을 때만 남깁니다 — 합쳐 최대 약 6~8 GB. Forge 가 자리가 필요하면 퇴출하지만 Forge 밖의 VRAM
  사용(SAM3 검출, 같은 GPU 의 학습)은 모릅니다. 학습과 같이 쓸 때는 Settings → **SAM Extra Anima 3.8B**(`sam3_anima38`) → "Anima 3.8B:
  TE·Qwen3.5·커넥터를 생성 사이 VRAM 에 남기기 (최대 약 6~8 GB)"(`sam3_anima38_keep_resident`, 기본 켬)를 끄세요. 끄면 TE·Qwen3.5 는 인코딩
  직후, 커넥터는 생성이 끝날 때 내립니다(3.8B 가 설치된 모델의 Forge TE 도 함께 — 줄 캐시로 올리지 않은 TE 까지 내리므로 다음 생성에서 TE 를 다시 올릴 수 있음. 3.8B 를
  설치하지 않은 생성은 영향 없음). 번들에 커넥터 전용 `llm_adapter` 사본이 없는 폴백에서는 결과를 지키려고 늘 내립니다.
- **메모리 정리**: VRAM 이 모자라 Qwen3.5 가 부분 로드돼도 돕니다(RMSNorm 가중치 장치 불일치 수정). `qwen35_4b` 를 VAE/Text Encoder 목록에 넣어 두어도
  Forge 가 체크포인트마다 4.8 GB 를 읽고 버리지 않아, XYZ 체크포인트 축에서 1.0/2.9B/3.8B 를 같은 모듈 목록으로 비교할 수 있습니다. 체크포인트를 바꾸면 이전 3.8B
  모델(약 8 GiB)을 RAM 에서 놓고, 설치 도중 모델이 다시 로드되면(Hires 체크포인트·Refiner 등) 남은 샘플링 패처를 생성 끝에 내립니다. GPU 측정: Qwen3.5 가 커넥터에
  넘기는 층의 최대 |값| 35.5 — fp16 텍스트 인코더에서도 넘치지 않습니다.
- **다른 패스·탭과 섞일 때의 수정**: SAM3 In-flight 인페인트 패스 뒤의 ADetailer 패스와 배치의 다음 이미지가 0.6B 조건으로 떨어지지 않고, 다른 탭 생성이 도중에 죽어도
  Bypass 생성이 v2 로 돌지 않습니다. 커넥터·Qwen3.5·v1 어댑터 가중치는 inference_mode 밖에서 만들어, 캐릭터 레퍼런스가 3.8B 에서 `Inference tensors
  do not track version counter` 로 죽지 않습니다.

### 캐릭터 레퍼런스 (이어붙이기 · IP-Adapter)

- **방식 토글 (새 기능)**: 패널 맨 위에서 **이어붙이기**(split canvas + Anima Edit img2img)와 **IP-Adapter** 를 고릅니다. IP-Adapter 는
  참조 이미지를 SigLIP2 로 읽어 Anima DiT 블록에 K/V 를 주입하므로 캔버스가 없고 구도가 자유롭습니다. 가중치(어댑터 503 MB + 인코더 1.5 GB)는 전문가 설정의 **모델
  받기** 를 눌러야 커밋 해시로 고정해 받습니다. 주입은 샘플링 동안에만 살아 있고, 어댑터가 모델보다 깊거나 블록 계보 매핑이 없는 조합, 차원이 다른 어댑터, 빈 블록이 있는 어댑터(그대로 두면
  랜덤 투영이 잔차에 더해짐)는 숫자를 보여 주고 멈춥니다.
- **28블록 어댑터를 2.9B·3.8B 에 얹기 (새 기능, 토글, 기본 lineage)**: Edit LoRA 와 같은 블록 계보표로 40·52블록에 펼칩니다. 기본(`lineage`)은 각
  어댑터 블록의 원래 계보 자리에만 주입하고 끼워 넣은 블록(2.9B 12개·3.8B 24개)에는 IP 모듈도 어댑터 LoRA 도 걸지 않습니다. 설정: Settings → **SAM Extra
  Character Reference**(`sam3_reference`) → "캐릭터 레퍼런스 IP-Adapter: 28블록 어댑터를 2.9B·3.8B 에 얹을 때 끼워 넣은 블록 처리 (결과가
  바뀜)" (`sam3_ipa_duplicate_policy`: `lineage` 기본 / `all` 대응 블록 전부 / `split` 대응 블록끼리 강도 1/n). GPU 비교 (참조 1장·시드
  12345·강도 1.0): lineage 는 2.9B·3.8B(커넥터 켬·끔) 모두 깨지지 않았고, split 은 격자 무늬가 남았고, all 은 둘 다 깨졌습니다(2.9B 는 강도 1.0,
  3.8B 는 0.5 에서도). lineage 에서는 주입하는 블록이 줄어 같은 설정의 3.8B 한 장이 17.2 → 14.0 초로 줄었습니다. 매핑이 걸린 생성의 infotext 에
  `SAM3 IPA Blocks: 28→40`/`28→52`(어댑터→모델 블록 수)와 `SAM3 IPA Duplicates: <정책>` 이 남고, 상태 줄에 'IP-Adapter 28블록 → 52블록
  모델 (<정책 설명>, 강도를 다시 잡으세요)' 가 표시됩니다(lineage 면 '블록 계보 매핑·끼워 넣은 블록 제외'). 28블록 베이스에서는 어느 값이든 결과가
  같습니다.
- **IP-Adapter 에도 3.8B v2 커넥터 (토글, 기본 켬 — 끄면 0.6B 조건)**: Settings → **SAM Extra Anima 3.8B** → "Anima 3.8B: 캐릭터
  레퍼런스 IP-Adapter 방식에도 v2 커넥터 설치 (끄면 결과가 바뀜)"(`sam3_anima38_reference_ipa`, 기본 켬). v2 번들이면 txt2img 와 같은 설치·복원
  순서로 Qwen3.5 커넥터 조건을 만들고(강도 1.0, 부정 프롬프트 순정 경로) 그 위에 IP-Adapter 를 얹습니다 — 이어붙이기 방식과 조건이 같아집니다. 끄면 0.6B 조건으로
  샘플링합니다. infotext 에 `Reference Anima38: v2 bundle / not a 3.8B v2 bundle / missing encoder (…) / install failed
  (…) / check failed (…) / unavailable` 와 `Anima38 adapter/strength/architecture/bundle/negative/encoder` 키가 남고,
  상태 줄에 `3.8B 커넥터: <라벨>`(설치하지 못하면 '꺼짐' 과 이유)이 표시됩니다. v2 번들이 아닌 모델(2.9B 등)은 결과가 같습니다.
- **IP-Adapter 잡이 뒤 생성에 흔적을 남기지 않음**: 잡이 끝나면 이 확장이 건 Forge 객체 패치를 직접 되돌려, 뒤의 txt2img 에서 LoRA 가 `[LORA] Mismatch`
  로 통째로 빠지지 않습니다(GPU 확인: IPA 잡 4번 뒤의 txt2img 가 IPA 를 쓰기 전과 픽셀 단위로 같음). IPA 잡 바로 뒤의 첫 v2 설치(txt2img·이어붙이기)에서 커넥터
  조건이 조용히 빠지고 infotext 에만 Anima38 이 남지 않도록, v2 래퍼는 객체 패치 전의 원본 DiT 를 감쌉니다(버그 수정이라 토글 없음, 객체 패치가 없는 정상 흐름은 비트 단위로
  같음). IPA 잡은 샘플링 중 예외로 남은 다른 생성의 3.8B 설치를 먼저 내립니다.
- **IP-Adapter 와 다른 확장의 forward 패치**: Safe PAG 를 한 번 켜서 생성했거나 3.8B 래퍼가 남아 있어도 주입이 적용됩니다. 원본 DiT 의 외부 forward
  래퍼(3.8B v2 조건 확장, NegPiP 마스크 릴레이)는 호출할 때마다 현재 것을 찾아 통과시키고, 블록 수준 래퍼(Safe PAG SLG/DAVE)는 IPA 의 껍데기 블록에는 적용되지
  않습니다. 주입이 한 번도 일어나지 않는 배선이 감지되면 조용히 계속하지 않고 원인(껍데기 cross_attn 을 거치지 않음 / 껍데기 블록을 돌지 않음 / IP 토큰
  `sam3_ip_tokens` 가 실려 오지 않음)을 적은 RuntimeError 로 멈춰 상태 줄에 표시합니다.
- **이어붙이기 패널 (Feature 6)**: 메인 6개(캐릭터 이미지, 가져오기, 유지 범위, 프롬프트, 후보 수, 생성/상태) + 접힌 전문가 칸입니다. 원본 ReStyler v1.2 조건(빈
  칸 `#000000` 그대로·디노이즈 1.0, Extend 기본 꺼짐)으로 돌고 결과 크기는 txt2img 를 따르며, 결과는 txt2img 갤러리에 넣습니다. Forge Neo 의
  `anima_do_reference` 를 생성 중에만 켜고 `finally` 에서 복구하며, Forge 본체 파일이나 저장된 설정은 바꾸지 않습니다. LoRA 는 자동 감지하고(점 포함 이름 안전)
  Edit LoRA 가 없을 때만 막습니다. Anima 가 아닌 모델은 막고, Forge 재시작 직후에도 선택한 체크포인트를 먼저 불러와 판정합니다. 3.8B v2 번들은 레퍼런스 실행에도 커넥터를
  설치하고 레퍼런스 latent 를 DiT 에 넘깁니다. ⏹ Stop 뒤의 다음 실행도 정상으로 돌고, 투명 PNG 도 검게 나오지 않습니다. infotext 에 실제 생성 패널 크기·결과 크기·확대
  배율·masked content·모델 블록 수·3.8B 커넥터 상태(`Reference Anima38`)를 남깁니다(Forge 의 `Size` 는 캔버스 크기). 패널은 Gradio 워커 스레드에서
  돌고 작업이 끝나면 Generate 처럼 interrupted/skipped 플래그를 정리합니다.

### LoRA · DoRA

- **ANIMA 28/40/52블록 LoRA 양방향 호환**: Forge 의 LoRA 로드 지점을 확장 안에서 감싸 Base 1.0(28)·2.9B(40)·3.8B(52) 사이 여섯 방향을 자동으로
  옮깁니다. 기존 Base→2.9B 의미는 그대로이고, 3.8B 는 체크포인트 metadata 의 LLaMA-Pro 삽입 위치를 씁니다. 하향 변환은 상속 블록만 남기는 손실 투영입니다. 적용할 수
  없는 3.8B Semantic Connector 전용 키는 개수와 함께 경고하고 빼며, 별도 Qwen3.5 encoder 키는 블록 수만으로 지우지 않습니다. Forge 본체 파일은 고치지
  않습니다.
- **부분(sparse) LoRA (결과 변화, 토글, 기본 꺼짐)**: 일부 블록만 든 LoRA 는 순정 Forge 판정(가장 큰 블록 인덱스+1 이 들어가는 가장 작은 28·40·52)이 현재
  모델과 같으면 그대로 로드합니다(콘솔 `Loading sparse ANIMA LoRA as-is …`). 다르면(예: 앞 21블록만 → 2.9B) 기본으로는 건너뛰고 gr.Warning 토스트를
  띄웁니다 — 원래 학습한 모델을 확정할 수 없다는 안내이며 '앞 N블록만 담은 접두 LoRA' / '중간이 빈 LoRA' / '판정 불가' 로 나뉘고 앞의 둘에만 이 설정 안내가 붙습니다. 3.8B
  Semantic Connector 키만 든 LoRA 를 Base/2.9B 에 얹을 때와 블록 수를 모르는 모델에서도 같은 토스트가 뜨며, 토스트는 LoRA 조합이 바뀔 때 한 번 뜹니다. 콘솔 거부
  경고에는 `(Forge's own rule would call it <레이아웃>)` 이 붙습니다. 순정처럼 추측 변환하려면 Settings → **SAM Extra
  LoRA**(`sam3_lora`) → "Anima 부분 LoRA 순정 추측 변환 (일부 블록만 담은 LoRA — 블록 대응이 틀릴 수
  있음)"(`sam3_anima_sparse_lora_forge_guess`, 기본 꺼짐)을 켜세요. 켜면 판정 레이아웃에서 현재 모델로 이 확장의 대응표·끼워 넣은 블록 정책을 써서 변환하고,
  순정이 거부하는 하향(52→40 등)도 변환합니다(인덱스 52 이상은 건너뜀). 중간이 빈 LoRA 는 순정과 결과가 다를 수 있습니다. 추측 변환한 생성은 콘솔 경고·정보 토스트와 infotext
  `Anima sparse LoRA: "Forge guess (<파일> 28->40)"` 을 남기고, 토글을 바꾸면 다음 생성에서 다시 합칩니다. UI 없는 always-on 스크립트 `SAM
  Extra Anima sparse LoRA` 가 하나 늘었습니다.
- **DoRA 추론 방식 (토글, 기본 꺼짐 = 순정)**: `dora_scale` 이 든 LoRA·LoKr·LoHa 를 합치는 방식을 고르는 아코디언(더 보기)입니다.
  - **방식**: 순정(Forge/ComfyUI — 원본 가중치의 노름으로 나누고 대부분의 GPU 에서 fp16 이라 같은 파일이 학습 샘플과 다르게 나올 수 있었음)·**LyCORIS**(학습과
    같은 공식, 합친 가중치의 노름으로 나눔, fp32)·**Forge/Comfy 공식 · fp32**·**DoRA 끔**(크기 보정 없이 ΔW 만, 실험용).
  - **끼워 넣은 블록**: 작은 Anima LoRA 를 큰 모델에 얹을 때 복제되는 블록을 **그대로 복제**(순정, 기본)·**덧셈형** (끼워 넣은 블록만 DoRA 크기 보정을 뺌)·**넣지
    않음**·**약한 복사**(덧셈형 복제본의 ΔW 를 0~1 배로, 범위 어텐션만 / 어텐션+MLP / 전체, 기본 0.12·어텐션만)로 채웁니다. 3.8B 의 끼워 넣은 블록은 출력 가중치가
    작아 2.9B DoRA 를 그대로 복제하면 출력 투영이 레이어별로 자기 크기의 0.35~10.9배 바뀌지만, 렌더 비교(시드 3개)에서 덧셈형·넣지 않음이 순정보다 일관되게 낫지는 않아 기본은
    순정입니다.
  - **적용·기록**: 선택을 바꾸면 다음 생성에서 한 번 다시 합치고, 합친 상태를 모델에 표시해 Reload UI·체크포인트 전환 뒤에도 낡은 가중치를 쓰지 않습니다. infotext `DoRA
    mode`/`DoRA inserted`, 붙여 넣기, XYZ `[DoRA] Inference mode`·`[DoRA] Inserted blocks`·`[DoRA] Weak copy
    strength`·`[DoRA] Weak copy scope`(cost 0.8 — 바깥 루프로 가서 칸마다 다시 합치지 않음), API `alwayson_scripts["DoRA
    Inference Mode"]` 를 지원합니다.

### 가이던스 (PAG · SEG · APG · CFG · Skimmed CFG · Detail Daemon · DCW · RDC · DAVE · CNS)

- **PAG/SEG/SLG: 첫 target 블록 앞의 weak 행 중복 계산 건너뛰기 (결과 변화, 토글, 기본 켬)**: 첫 target 블록(기본 18) 이전 블록은 cond/uncond 행만
  돌리고 weak 행 자리에는 cond 행 출력을 복사합니다. GPU 에서 PAG 한 장이 56.2 → 51.4 초였고, 다시 돌려도 결과가 같으며, 끈 것과 구도는 같고 잔 디테일만 다릅니다. 끄는
  설정: Settings → **SAM Extra Guidance**(`sam3_guidance`) → "PAG/SEG/SLG: 첫 target 블록 이전의 weak 행 중복 계산 건너뛰기"
  (`sam3_guidance_pag_prefix_dedup`, 기본 켬). perturbation 을 켠 생성의 infotext 에 `Anima PAG prefix dedup: True/False`
  가 남고 붙여 넣기·`override_settings` 로 복원됩니다. 행을 자를 수 없는 블록(모르는 블록 인자, 행 수가 맞지 않는 텐서)부터는 예전 전체 배치 경로로 돌고 생성당 한 번 `앞쪽
  블록 중복 제거를 블록 … 부터 건너뜁니다` 경고를 남기며, 잘린 forward 가 OOM 이 아닌 예외를 내면 그 패스는 중복 제거를 끕니다. 로그: `attached ✅` 줄의
  `prefix_dedup=… seg_separable=…`, generation summary 의 `prefix_dedup_blocks=N prefix_dedup_fallbacks=M`.
- **SEG(공식) query blur 를 가로·세로 1D 두 번으로 (결과 변화, 토글, 기본 켬)**: 픽셀당 곱셈이 k² 에서 2k 로 줄어듭니다(1024² 면 3969 → 126). GPU
  에서 끈 것과 구도는 같고 잔 디테일만 다릅니다. 끄는 설정: Settings → **SAM Extra Guidance** → "SEG(공식): query Gaussian blur 를 가로·세로
  1D 두 번으로 계산"(`sam3_guidance_seg_separable_blur`, 기본 켬). 공식 SEG 이고 0<sigma≤9999 인 생성의 infotext 에 `Anima SEG
  separable blur: True/False` 가 남습니다. 두 Guidance 키는 XYZ 칸 사이 정리 목록에도 들어 있습니다. 예전 이미지를 그대로 재현하려면 두 설정을 모두 끄세요.
- **SMC / APG / CWM 은 CFG 1 에서 건너뜀 (결과 변화, 토글 없음)**: 레거시 CFG base mode·experimental stack 포함, CFG 1 이면 base 교체를
  건너뛰고 Forge 의 원래 결과를 둡니다. 예전에는 APG(eta=0)가 출력을 거의 0(검정/회색 이미지)으로 만들고 CWM/SMC 도 임의로 재가중했습니다. 원본 DCW(+a) 는
  SMC·CWM 을 CFG 1 에서도 돌리고 이 확장도 SMC·CWM 이 켜진 CFG 1 패스에는 원본처럼 `disable_cfg1_optimization` 을 걸지만, Forge 는 CFG 가 1
  이면 negative prompt 를 인코딩하지 않아 uncond 가 없으므로 결국 건너뜁니다. 콘솔에 생성당 한 번 `CFG base override (<켜진 것>) skipped:
  cond_scale=1` 경고, 진단의 CFG base 판정은 `NO-OP` 입니다. CFG=1 판정은 Forge 가 넘기는 `cond_scale` 로 하고, 없을 때만 uncond 를 봅니다.
- **Safe PAG: CFG 1 에서 rescale 자동 끄기를 적용하지 않음 (결과 변화, 토글 없음)**: CFG 1 이라 APG 가 돌지 않는 스텝에서는 PAG rescale 도 그대로
  적용합니다. 그래서 APG + rescale 자동 끄기를 켠 CFG 1 생성은 결과가 달라지고, 이제 APG 를 끈 생성과 비트 단위로 같습니다. CFG > 1 은 같습니다. 첫 CFG 1 경고에
  이 점이 적히고, `disable_cfg1_optimization` 이 켜져 있으면 uncond 패스는 돌았지만 결과가 cond 예측 그대로라고 안내합니다.
- **Safe PAG 확장 배치 OOM (결과 변화, 토글 없음)**: 실패한 forward 의 활성값을 놓은 뒤 폴백합니다. OOM 이면 캐시를 비우고 그 생성의 남은 스텝·hires 패스·남은
  배치에서 PAG/SEG/SLG 를 끕니다(예전에는 매 스텝 재시도). OOM 이 난 생성만 결과가 달라질 수 있고, 이때 hires·둘째 배치 이후 이미지의 infotext 에는 PAG 항목이 빠질
  수 있습니다.
- **Safe PAG 를 원본 노드와 같게 (결과 변화, 토글 없음)**:
  [iljung1106/comfyui-anima-safe-pag@905b0107](https://github.com/iljung1106/comfyui-anima-safe-pag) (MIT)
  기준입니다. PAG 의 Start/End 는 원본처럼 모델 스케줄의 σ 로 바꿔(`percent_to_sigma`, 양 끝 포함) 모델 호출마다 현재 σ 로 판정합니다. 예전에는 한 스텝 늦게
  오르는 Forge 스텝 비율을 써서, Anima(shift 3)의 기본 구간 0.0~0.7(σ 1.0~0.5625)이 simple 스케줄 20·28·30 스텝에서 원본의 15·20·22 스텝이
  아니라 14·19·21 스텝에 걸렸습니다. 이제 스텝 수·스케줄러·img2img denoise 가 달라도 원본과 같은 σ 구간이고 2차 샘플러의 중간 평가도 σ 로 판정합니다(predictor 가
  없는 모델만 예전 스텝 비율). SEG·SLG 는 원본 노드에 없는 기능이라 예전처럼 스텝 비율입니다. Attn Scale 상한을 15 → 100 으로 올렸고(슬라이더·API·XYZ),
  블록·head 번호의 역범위(`20-18`)는 원본처럼 18~20 으로 읽습니다(예전에는 그 부분을 버림, SLG·DAVE 블록 칸도 같음). ControlNet 이 붙은 호출에서
  PAG/SEG/SLG 를 쉬는 것은 원본에 없는 이 확장의 안전장치라, 실제로 막힌 패스의 infotext 에 `Anima Perturbation ControlNet guard` 를 남깁니다.
  `Anima Perturbation Guidance` infotext 에는 PAG σ 창(`pag_sigma_window=`)이 붙습니다.
- **Skimmed CFG 를 원본 노드와 같게 (결과 변화, 토글 없음)**:
  [Extraltodeus/Skimmed_CFG@d8300583](https://github.com/Extraltodeus/Skimmed_CFG) 의 수식 함수를
  `sam3ext/guidance/skimmed_cfg.py` 에 그대로 편입했습니다(Apache-2.0, 고지는 `THIRD_PARTY_NOTICES.md`). Start/End/Flip at 을
  원본처럼 모델의 `percent_to_sigma` 로 σ 로 바꿔 `end σ < σ < start σ`(경계 제외)인 스텝만 깎습니다 — 예전에는 한 스텝 늦은 스텝 비율이었습니다. 그래서
  flow 모델인 Anima 는 첫 스텝(σ 1)을 깎지 않고, start > end 면 예전처럼 바꿔 읽지 않고 아무 스텝도 깎지 않습니다. flip 규칙(flip 지점의 σ 보다 앞선 스텝에서
  뒤집음)·깎는 순서도 원본대로이고, 원본에 없던 `nan_to_num` 은 뺐습니다. RescaleCFG·Dynamic Thresholding 처럼 등록된 `sampler_cfg_function`
  은 버리지 않고 깎인 예측으로 다시 부르며, 없으면 Forge 처럼 `edit_strength` 를 반영한 선형 CFG 입니다. Flip at 슬라이더 step 은 0.05 → 0.01 이고,
  `ui-config.json` 에 저장된 예전 step 은 첫 시작 때 한 번 지웁니다.
- **DCW·CWM·SMC 를 원본 DCW(+a) 와 같게 (결과 변화, 토글 없음)**:
  [namemechan/ComfyUI-DCW@66aaf9dd](https://github.com/namemechan/ComfyUI-DCW) 기준으로 기본값·범위를 맞췄습니다: DCW λ low
  0.10 → 0.05(−0.5~0.5), λ high 0.02 → 0.01(범위 ±0.5 → ±0.3, step 0.001), CWM α low·high 0.30·0.15 → 0(범위 −1~1 →
  −1~2, 원본 권장 시작값 low 0.1~0.3·high 0.1~0.2), SMC Custom λ 0.5~30(API·XYZ 값도 이 범위로 맞춤). CWM 은 원본처럼 α 가 0 이 아닐 때만
  CFG 를 바꾸므로 새 기본값에서는 켜도 표준 CFG 와 같습니다. SMC/APG/CWM 의 CFG 배율은 원본 cfg 훅처럼 Forge 가 넘기는 `cond_scale`(다른 CFG 함수가 없으면
  Forge 처럼 `edit_strength` 를 곱함)이고, incoming 결과를 최소제곱으로 맞춘 값은 진단(`[VERIFY]` 의 `w_fit`)과 비선형 CFG 경고에만 씁니다.
  RescaleCFG·Dynamic Thresholding 처럼 다른 확장이 `sampler_cfg_function` 을 걸어 두면 원본처럼 SMC·CWM 만 비키고(경고 1회) 그 결과를 둡니다 —
  APG·PAG/SEG/SLG·DCW/RDC 는 그대로 적용합니다. Adaptive Guidance 가 uncond 를 건너뛴 스텝에도 원본 post-CFG 훅처럼 DCW/RDC 를 적용하고, 그
  스텝에서는 APG momentum 만 비우고 SMC 의 이전 오차는 유지합니다(예전에는 둘 다 비우고 DCW 도 건너뜀). infotext `Anima DCW` 는 DCW 가 실제로 돌 때(λ 가
  0 이 아니거나 RDC 가 켜짐) 남습니다.
- **DCW / CWM / SMC 명시적 ON/OFF**: Guidance 본문에 세 기능의 독립 체크박스를 두었습니다. SMC 는 고른 `Auto`/모델별/`Custom` 프리셋 값을 유지한 채
  master 체크박스로 바로 A/B 할 수 있습니다. CWM 은 원본처럼 α 가 0 이 아닐 때만 CFG 를 바꾸므로, 원본 기본값 0 에서는 켜도 결과가 같습니다. RDC 에는 따로 켜는
  체크박스가 없습니다(아래). 기존 script argument 와 XYZ 축 정수 인덱스는 그대로 두고 새 입력을 맨 뒤에 붙였습니다.
- **RDC 이식 (새 기능, 토글 tau, 기본 0 = 끔)**: [namemechan/ComfyUI-DCW](https://github.com/namemechan/ComfyUI-DCW) 의
  band-wise reverse drift compensation 을 Forge post-CFG 경로에 다시 작성했습니다. 원본처럼 따로 켜는 스위치가 없고 **Enable DCW 가 켜져 있고
  tau > 0** 일 때 DCW 의 Haar 변환 안에서 돕니다(DCW λ 를 둘 다 0 으로 두면 RDC 만). 이전 v0.30 개발 빌드의 Enable RDC 체크박스는 화면에서 뺐고, 그
  script argument 자리(58)는 남아 API 가 False 를 보내면 RDC 를 끄며 XYZ `[Anima RDC] Enable` 도 False 일 때만 끕니다.
  `tau`(0~0.5)·`alpha LL`(기본 0.03)·`alpha HH`(기본 0)를 UI/XYZ/infotext(`Anima RDC`)에 모두 노출하고, 생성마다 EMA 를 초기화하며
  해상도가 바뀌면 다시 시작합니다.
- **DAVE 를 원본 노드와 같게 (결과 변화, 토글 없음)**:
  [sorryhyun/ComfyUI-Anima-DAVE@83143e8d](https://github.com/sorryhyun/ComfyUI-Anima-DAVE) (MIT) 의 초반 스텝 게이트를
  `sam3ext/guidance/dave_gate.py` 로 옮겼습니다. 예전에는 한 스텝 늦은 Forge 스텝 비율이 tau 보다 작을 때 켰고, 이제 원본처럼 모델 호출의 σ 를 샘플러가 도는
  σ 스케줄에서 찾아 그 스텝 번호가 `k = max(1, min(n, round(tau × n)))` 보다 작을 때 켭니다(tau 0.10 이면 20·25 스텝에서 첫 2 스텝, 28·30 스텝에서
  첫 3 스텝). 스케줄에 없는 σ(2차 샘플러의 중간점)는 원본처럼 늘 켭니다. txt2img 는 σ 목록 전체, img2img·hires 는 Forge 와 같은 `steps − t_enc − 1`
  칸부터 세고, ADetailer 내부 img2img·img2img-hires-fix 처럼 이 스크립트가 준비하지 않은 실행은 `on_cfg_denoiser` 로 그 실행을 알아내 자기 스텝 수로
  끝에서 셉니다. σ 목록이 없는 DDIM·PLMS 는 Forge 스텝 위치로 판정합니다(한 스텝 늦음). 블록 칸을 비우면 원본 마스크와 같은 `8-18`, strength 0.001 이하는
  원본처럼 아무것도 하지 않습니다.
- **CNS 를 원본과 같게 (결과 변화, 토글 없음)**:
  [namemechan/comfyui-cns_sampler_patch@42278b13](https://github.com/namemechan/comfyui-cns_sampler_patch) 의
  `color_noise_wavelet` 을 `sam3ext/guidance/cns.py` 에 그대로 편입했습니다(GPL-3.0). Strength 가 1 보다 작으면 흰 노이즈와 `lerp` 로만
  섞고 표준편차를 다시 맞추지 않습니다(원본처럼 조금 낮아짐). 재색칠 기준 `x_t` 는 post-CFG 입력 대신 원본처럼 샘플러 스텝 callback 의 `x`(그 스텝의 시작 상태)를 쓰고,
  callback 을 감쌀 수 없는 샘플러에서만 예전처럼 post-CFG 입력을 씁니다(검증 로그 `x_t=callback|post_cfg`). 기본 Gamma scale 은 3.0 → 2.0,
  범위도 원본대로입니다(Strength step 0.05, Gamma power 최소 0.1, Gamma scale 0.1~25 · step 0.1 — 원본 README 의 Anima +
  euler_ancestral_cfg_pp 권장값 3.0 은 라벨에 적음). 이 스크립트가 붙은 패스의 샘플러가 도는 동안에만 색칠해, ADetailer 내부 img2img 같은 중첩 실행은 흰
  노이즈 그대로입니다.
- **Detail Daemon 을 ComfyUI-Detail-Daemon 과 같은 값으로 (결과 변화, 토글 없음)**:
  [Jonseed/ComfyUI-Detail-Daemon@3394e44](https://github.com/Jonseed/ComfyUI-Detail-Daemon) 의 값과 σ 조회를 따르고, 노드가
  다루지 않는 Forge 동작은 muerrilla 원본을 따릅니다. 강도는 `σ × max(1e-6, 1 − 스케줄 × 0.1 × CFG)` 이고 CFG 는 hires 패스에서도 늘
  `p.cfg_scale` 입니다. 포크 때 빠졌던 ×0.1 을 되살려 같은 amount 가 노드의 `detail_amount` 와 같은 결과를 냅니다 — **v0.21.2 와 같은 강도는
  amount × 10** 이고, 이를 위해 Amount 를 −1~1 → −5~5(기본 0.10)로 넓혔습니다. 원본에 없는 프리셋·Multiplier·CFG 결합 토글과 [0.05, 3] 클램프는
  없앴습니다(API 위치 인자 1·10·12 자리는 남기고 읽지 않음). 스케줄 위치는 노드처럼 모델 호출마다 그 σ 를 샘플러가 도는 σ 목록에서 찾아(가장 가까운 칸, 칸 사이는 선형 보간)
  정하므로 Forge 스텝 번호의 한 스텝 지연이 없고, 2차 샘플러의 중간 평가도 노드와 같은 곡선 값을 읽습니다. txt2img 는 σ 목록 전체, img2img·hires 는 Forge 와 같은
  `steps − t_enc − 1` 칸부터 세고(`DDIM` 스케줄이 σ 를 스텝 수 + 2 개 내놓아도 맞음), ADetailer 내부 img2img·img2img-hires-fix 처럼 이
  스크립트가 준비하지 않은 실행은 그 실행의 스텝 수로 끝에서 셉니다. σ 목록이 없는 DDIM·PLMS 는 모델 호출 수로 셉니다. muerrilla 원본처럼 **Hires Pass**(새
  체크박스, API 인자 13, 기본 끔)를 끄면 기본 패스에만, 켜면 hires 패스에만 걸리고(v0.21.2 는 두 패스 모두), DPM adaptive·HeunPP2 에서는 꺼집니다. σ 는
  원본처럼 제자리에서 바꿔 NGMS·soft inpainting 도 바뀐 σ 를 봅니다. infotext `Anima Detail Daemon` 에
  exponent·offset·fade·smooth·hires 까지 남깁니다. Amount 라벨이 바뀌어 `ui-config.json` 에 저장된 v0.21.2 의 Amount 값·범위는 적용되지
  않습니다(기본 0.10 으로 시작).

### SAM3

- **SAM3 를 쓴 뒤 다른 생성의 결과가 바뀌던 문제 (결과 변화, 버그 수정, 토글 없음)**: sam3 는 import 때 프로세스 전역 TF32 를 켜서, 한 번이라도 SAM3 를 쓰면 그
  뒤의 모든 생성에서 fp32 행렬곱(LyCORIS DoRA 합치기, 3.8B 커넥터 등)이 TF32 로 돌았습니다(3.8B 평균 픽셀 차이 38, 2.9B 11). 이제 import·빌드 동안 바뀐
  설정을 되돌리고 SAM3 검출 동안에만 TF32 를 켭니다(마스크는 같음). GPU 확인: SAM3 를 쓴 뒤의 3.8B·2.9B 생성이 쓰기 전과 픽셀 단위로 같습니다.
- **SAM3 인페인트 패스의 시드 (결과 변화, 버그 수정, 토글 없음)**: In-flight·Refine 인페인트 패스가 Forge Seed 스크립트에 시드를 덮어써 txt2img 시드 칸
  값(API 면 -1 = 매번 무작위)으로 돌던 문제입니다. 이제 바깥 생성의 시드 (또는 SAM3 Seed)를 씁니다. GPU 확인: 같은 설정의 SAM3 생성 두 번이 픽셀 단위로
  같습니다(3.8B·2.9B). UI 에서 시드 -1 로 만든 예전 이미지와는 SAM3 부분이 다를 수 있습니다.
- **API/XYZ 생략 키의 기본값을 UI 와 통일 (결과 변화)**: `sam3_inpainting_fill` 은 `latent noise`→`original`,
  `sam3_unload_after` 는 False→True 입니다. 예전 값을 원하면 두 키를 명시하세요.
- **범위 밖 설정값은 끄지 않고 맞춤 (결과 변화)**: XYZ `[SAM3] Threshold` 1.5, Mask Blur -1, API 의 잘못된 수치 등이 오면 SAM3 가 조용히 꺼지는 대신
  범위로 맞춰 실행합니다(threshold 1.5→1.0, blur -1→0, steps 0→1, inpaint width 0→64 등). Inpainting Fill / Mode / Mask
  Mode / CN Control Mode / CN Resize Mode 는 대소문자·공백을 무시하고, 모르는 값은 기본값(original / Inpaint / Individual / Balanced
  / Crop and Resize)으로 바꾸며 stderr 에 `[-] SAM3: unknown <필드> value '<값>', using the default '<기본값>'.` 을 남깁니다(값을
  생략하면 경고 없이 기본값). 그래도 검증이 실패하면 `[-] SAM3: invalid settings, SAM3 disabled for this generation: …` 과 infotext
  `SAM3 Error` 입니다.
- **실패를 infotext 에 기록하고 VRAM 정리**: 검출·인페인트가 예외로 끝나면 저장 이미지의 infotext 에서 `SAM3 Enable: True` 대신 `SAM3 Error:
  <예외형>: <메시지>` 가 들어가고 stderr 에 `[-] SAM3: failed, image saved without SAM3: …` 가 남습니다. 'Unload after' 가 켜져 있거나
  CUDA OOM 이면 모델을 VRAM 에서 내립니다(Refine 은 앞머리 `[-] SAM3 Refine:`). 예외 자체는 예전처럼 전파됩니다. 배치에서 앞 장이 실패해도 뒤에 성공한 장은
  `SAM3 Enable: True` 로 저장됩니다.
- **'Unload after' 가 모델을 CPU RAM 에 보관 (결과 같음, 토글, 기본 켬)**: 검출 뒤 VRAM 에서 내리는 것은 같지만 모델을 RAM 에 약 3.4 GB 로 남겨 다음
  검출은 GPU 로 옮기기만 합니다(재빌드·`sam3.pt` 재읽기 없음). 끄는 설정: Settings → **SAM Extra SAM3**(`sam3_mask`) → "'Unload after'
  뒤 SAM3 모델을 CPU RAM 에 보관 (약 3.4 GB)" (`sam3_unload_keep_in_ram`, 기본 켬). 끄면 VRAM·RAM 에서 모두 해제하고 다음 검출마다 다시 빌드해,
  RAM 약 3.4 GB 를 아끼는 대신 이미지당 약 2.5~5 초(추정)가 더 듭니다. 끄고 Apply 하면 보관 중인 모델도 곧바로 해제하고, RAM 의 모델은 Reload UI·확장 언로드 때도
  해제됩니다. 로그는 `model moved from VRAM to CPU RAM (moves back on next detection).` / `model released from VRAM and
  RAM (reloads on next detection).`(실패 경로는 `… after the failure …`)입니다. 체크포인트·장치를 바꾸면 옛 모델을 먼저 해제해 두 모델(약 7 GB)이
  GPU 에 함께 있던 순간이 없어지고, 같은 경로의 파일을 교체하면(mtime·크기) 새로 빌드합니다. 내부 API `unload_sam3()` 는 RAM 에 보관했는지(bool)를 돌려주고 선택
  인자 `keep_in_ram` 을 받습니다(인자 없는 호출은 그대로).
- **In-flight 내부 패스에서 ADetailer 제외 (결과 변화, 토글 없음)**: 생성 중 SAM3 인페인트의 내부 패스에서 ADetailer 가 돌지 않습니다(🎯 와 같음). 바깥 생성의
  ADetailer 는 그대로 한 번 돕니다.
- **SAM3 ControlNet 의 Anima LLLite 전처리 (결과 변화, 토글 없음)**: SAM3 인페인트 패스에 넣는 ControlNet 유닛이 Anima ControlNet-LLLite
  이면 원본(kohya sd-scripts·ComfyUI-Anima-LLLite)처럼 제어 이미지를 그대로 받게 합니다. 채널 수와 Tile & Repair 여부는 모델 파일의 safetensors
  헤더로 읽고, 못 읽으면 파일 이름으로 봅니다. Tile & Repair LLLite(3채널)는 preprocessor 를 늘 `None` 으로 바꾸고 — 기본 `inpaint_only` 가 고칠
  영역을 비워 복구할 내용을 지웠습니다 — lineart·canny·depth 같은 다른 3채널 Anima LLLite 는 고른 preprocessor 를 쓰되 `inpaint_*` 만 `None`
  으로, 4채널 인페인트 LLLite 는 예전처럼 `inpaint_*` 를 `None` 으로 바꿉니다. 바꿀 때마다 stderr 에 한 줄 남기고, Anima LLLite 가 아닌 모델(SDXL
  `kohya_controllllite_*` 포함)은 건드리지 않습니다.
- **SAM3 빠른 버튼(🎯)**: txt2img 갤러리 ✨(hires fix) 옆에서 선택한 이미지에 지금 SAM3 설정을 바로 돌립니다. ✨ 와 같은 규칙(지금 txt2img 설정 + 그 이미지의
  시드, 결과 배치는 Forge 의 hires button gallery insert 설정)이며 SAM3 아코디언이 꺼져 있어도 돌고 다른 후처리는 돌리지 않습니다. 결과는 `-sam3` 접미어로
  저장하고 infotext 에 SAM3 설정과 `SAM3 quick: True` 를 붙입니다.
- **XYZ 축 순서**: `[SAM3] Checkpoint`·`[SAM3] Device` 축(cost 0.9)이 시드 등 다른 축보다 바깥 루프로 가서 칸마다 SAM3 를 다시 빌드하지 않습니다.
  칸 결과는 같고 순서만 바뀝니다.

### TIPO 프롬프트 확장 (🪄)

- **프롬프트 확장 버튼**: txt2img 도구 줄의 🪄 가 TIPO-v2.1-1B-A200M 으로 지금 프롬프트를 확장해 프롬프트 칸에 다시 씁니다. 스타일 줄 아래에서 확장
  방식(태그만/태그+설명/설명만)·길이·새 작가·캐릭터·작품 허용·장치·시드를 고르고 ↩ 로 되돌립니다. 적어 둔 텍스트는 그대로 두고 새 태그(캐릭터→작품→@작가→일반, 괄호 이스케이프)와 설명만
  뒤에 붙입니다. 모델 코드는 확장 안에 넣었고(KohakUwULLM, Apache-2.0) 가중치(1.98 GB, Kohaku License 1.0)는 **모델 받기** 를 눌러야 고정 버전으로
  받습니다. 기다리는 동안 프롬프트를 고치면 결과를 넣지 않고, 토큰 한도에서 잘린 끝은 버리며, 토큰 수를 다시 셉니다.
- **장치 (생성 텍스트 같음)**: 라디오는 GPU(기본) / CPU 이고, 여유 VRAM 이 3 GB 보다 적으면 그 회차는 CPU 로 돕니다. 옆 체크박스 "GPU 에 남겨 두기"(기본 켬)는
  GPU 로 돈 TIPO(fp16 약 2 GB)를 Forge 메모리 관리에 맡겨, 여유가 있으면 VRAM 에 남아 다음 클릭에 다시 올리지 않고 이미지 생성 등으로 자리가 필요하면 통째로
  내립니다(학습 등 Forge 밖 VRAM 사용은 모름). 끄면 누를 때만 올렸다 내립니다. 상태 줄 장치는 'GPU(남김)' 이고, Forge 메모리 관리를 쓸 수 없으면 'Forge 메모리 관리를
  쓸 수 없어 GPU 에 남기지 않았습니다' 를 표시합니다. 쉴 때는 RAM 에 fp16 약 2 GB(CPU 로 돌린 뒤에는 fp32 약 4 GB) 사본을 두어, GPU 에서 내릴 때
  GPU→CPU 복사를 하지 않고 CPU 로 다시 돌릴 때 fp16↔fp32 캐스트를 반복하지 않습니다. 장치 라디오는 브라우저가 기억하고, 체크박스는 새로 고치면 기본값으로
  돌아갑니다.

### txt2img 화면 · 도구

- **txt2img 섹션 정리**: always-on 아코디언을 묶음별 섹션으로 나눕니다 — 1열(고정·ANIMA 튜닝), 2열(디테일러·스크립트·더 보기), 3열(갤러리 밑 **선택 이미지**
  탭: Refine·Tile-Repair·캐릭터 레퍼런스). 프롬프트 밑에 **켜진 기능** 칩 줄이 생기고 각 헤더에 `켜짐`/`k/n 켜짐` 표시와 핀이 붙습니다(핀은 브라우저에 기억). 항목은
  스크립트 파일 이름으로 알아보고 상태는 `data-*` 속성으로만 써서 예전 ko_KR 무한 루프가 재발할 수 없습니다. 가운데 열의 절대 위치·타이머 배치 엔진은 지웠습니다. Settings →
  **SAM Extra Appearance**(`sam3_appearance`) → "txt2img 섹션 정리(켜진 기능·고정·더 보기) — Forge 재시작 후
  적용"(`sam3_layout_sections`, 기본 켬) 또는 주소의 `?sam3_lanes=off` 로 끌 수 있습니다.
- **Refine · Tile-Repair · PiD 를 Forge 생성 큐 안에서 실행**: 세 패널의 실행이 txt2img Generate 와 같은 `queue_lock` 안에서 Forge 메인
  스레드로 돌아 서로 기다립니다(동시 실행 없음). 실행마다 중단 플래그를 초기화해, ⏹ Stop 뒤 다음 실행이 곧바로 `SAM3 Refine: no result (empty mask or
  interrupted)` 로 끝나던 문제가 없어졌습니다. 패널의 ⏹ Stop 은 그 패널 작업이 실제로 도는 중일 때만 중단합니다. txt2img 가 도는 동안 대기 중인 클릭은 Stop 으로
  취소되지 않고(txt2img 도 멈추지 않음) 잠금이 풀리면 그대로 실행됩니다. Anima Tile-Repair 의 ⏹ Stop 이 다음 디노이징 스텝에서 실제로 멈춥니다(모델 로드·텍스트
  인코딩·VAE 디코드 구간은 중단 불가). Refine 이 실패해도 진행 문구가 진행 표시줄에 남지 않습니다.
- **Tile-Repair 정리**: Text Encoder 자동 선택이 Qwen3 0.6B(예: `qwen_3_06b_base.safetensors`)만 고르고, 없으면 `Use Forge
  current` 가 기본입니다(예전에는 Qwen3-VL 등을 골라 크기 불일치로 죽었음). 제외 판정은 'vl'·'vlm'·'clip' 이 토큰일 때만입니다. 벤더에 전달된 적 없는 LLLite
  Strength / Start % / End % 슬라이더를 뺐고(Multiplier 만 남음 — 아래), infotext 의 LLLite 항목은 `LLLite: <model> (mult <x>)`
  로 짧아졌습니다.
- **Tile-Repair 를 sd-scripts 원본과 같게 (결과 변화, 토글 없음)**:
  [kohya-ss/sd-scripts@690ea7f9](https://github.com/kohya-ss/sd-scripts) 추론과
  [kohya-ss/ComfyUI-Anima-LLLite@b7495bd8](https://github.com/kohya-ss/ComfyUI-Anima-LLLite) 기준입니다. Width·Height
  슬라이더(기본 1024×1024) 대신 **Short Side**(기본 1024) 하나로 원본 비율을 지키고, 긴 변은 비율을 따라 두 변 모두 32 의 배수로 내립니다(최소 256). 디코드는
  sd-scripts 처럼 `(clamp(−1, 1) + 1) × 127.5` 를 uint8 로 잘라(예전에는 범위를 추정해 반올림) 픽셀 값이 1 씩 다를 수 있습니다. LLLite 목록에는
  safetensors 헤더로 가려낸 3채널(RGB) Anima LLLite 만 나오고(4채널 인페인트 LLLite·다른 ControlNet·`.safetensors` 가 아닌 파일은 빠짐), 기본은
  가장 새 Tile & Repair 파일(v20)입니다. Multiplier 는 ComfyUI-Anima-LLLite 의 strength 와 같은 −10~10(step 0.01, 기본 1.0)이고
  라벨이 바뀌어 저장된 예전 0~2 범위는 적용되지 않습니다. 네거티브 기본값은 sd-scripts 처럼 빈 칸이고, `ui-config.json` 에 예전 기본값(`blurry, low
  quality`)이 그대로 저장된 기존 설치는 첫 시작 때 한 번 빈 칸으로 옮깁니다(직접 적은 네거티브는 유지).
- **Tile & Repair HTTP API (새 기능)**: `POST /sam-extra/tile-repair` 가 JSON 하나로 패널의 Tile-Repair 모드와 같은 실행을 합니다.
  `image`(base64 PNG·JPEG·WebP, `data:` 접두어 가능)만 필수이고 나머지 키(`model`, `prompt`, `negative_prompt`, `steps`,
  `cfg_scale`, `flow_shift`, `multiplier`, `short_side`, `seed`, `dit`, `text_encoder`, `vae`,
  `unload_forge_before`)는 생략하면 패널 기본값(네거티브는 빈 칸)이며, 모르는 키는 400 으로 거절합니다. 결과는 base64 PNG(infotext 포함)·실제로 쓴
  시드·크기입니다. `GET /sam-extra/tile-repair/options` 는 패널의 선택지·기본값·범위를, `POST /sam-extra/tile-repair/stop` 은 이 경로의
  요청만 멈춥니다(패널 ⏹·txt2img 는 건드리지 않고, Forge 의 Interrupt 는 둘 다 멈춤). txt2img·패널과 같은 Forge 대기열에서 돌고, 입력은 디코드 64 MB·64
  MP, 출력도 64 MP 까지입니다(가는 원본에 short side 를 주어 수십억 픽셀이 되는 요청은 400). 인증은 Notebook 경로와 같습니다(`--gradio-auth` 로그인,
  Forge 가 API 를 띄울 때의 `--api-auth` HTTP Basic, 헤더 `X-SAM3-Notebook: 1`). LoRA 칸·PiD 모드·갤러리 삽입은 패널에만 있습니다. UR_IV
  데스크톱 앱처럼 같은 출처에서 부르는 클라이언트용입니다.
- **Anima VAE 2x 가 Anima 생성에서 실제로 돕니다**: 켜면 `TypeError: Cannot handle this data type` 로 생성이 죽던 문제를 고쳤습니다(순정과 같은
  5D 결과). decode 결과 모양이 맞지 않으면 `2x decode failed → stock decode: …` 를 남기고 순정 decode 로 폴백합니다. 12ch 디코더는 Forge 메모리
  관리로 decode 때만 올립니다. `--novram` 등으로 디코더가 GPU 에 못 올라가 실패하면 그 이유를 프로세스당 한 번만 안내하고 이후에는 조용히 순정 decode 로 넘어갑니다. 실제
  GPU 생성 확인은 아직입니다.
- **빠른 드롭다운**: UI 업데이트마다 도는 스캔이 Gradio config 전체(약 5,600개)를 선형 탐색하던 것을 elem_id 인덱스 조회로 바꿨습니다(스캔 1회 약 5.7 ms 중
  4.3 ms 가 이 탐색이었음). 설정값(기본 60개)은 이제 전체 상한이 아니라 한 번에 더하는 개수라, 목록 끝까지 스크롤하면 다음 묶음을 그려 XYZ 축처럼 긴 목록도 끝까지 볼 수 있습니다.
- **Notebook 메모장 (새 기능)**: Notebook 패널에 **프리셋 / 메모** 탭이 생겼습니다. 메모(제목 120자·본문 100,000자)는 입력을 멈추고 약 0.6 초 뒤 자동
  저장되며, `notebook.json` 옆의 `memos.json`(기본 Forge 데이터 경로의 `sam-extra/memos.json`)에 따로 저장해 프리셋 저장의 revision 과 섞이지
  않습니다. UR_IV 앱도 같은 메모를 `GET/PUT/DELETE /sam3-notebook/memos[/{id}]` 로 읽고 씁니다(Notebook 과 같은 인증·헤더). 저장·삭제는 메모마다
  마지막으로 본 `updated_at` 을 함께 보내, 그사이 다른 곳에서 바뀐 메모를 덮어쓰거나 지우지 않고 409 를 받습니다 — 화면은 서버 쪽 메모를 그대로 두고 내 편집을 '(충돌 사본)'
  새 메모로 남깁니다. 지운 메모는 동기화용 tombstone 으로 남고, 메모는 tombstone 포함 500개까지이며 넘치면 오래된 tombstone 부터 정리합니다. 저장 전 편집은 브라우저
  탭마다 localStorage 초안으로 남아, 닫힌 탭의 초안은 다른 탭이 이어받습니다. 경로가 없으면(업데이트 뒤 재시작 전) 메모 탭에 Forge 를 다시 시작하라고 표시합니다.
- **확장 HTTP 경로에 Forge `--api-auth` 적용**: Forge 가 `--api`·`--nowebui` 로 API 를 띄우고 `--api-auth` 가 있으면 Tile &
  Repair(`/sam-extra/tile-repair…`)·Notebook(`/sam3-notebook`)·메모 경로에도 `/sdapi` 와 같은 HTTP Basic 인증을 겁니다(없거나 틀리면
  401). `--gradio-auth` 로그인도 켜져 있으면 둘 다 필요합니다. `--api`·`--nowebui` 없이 준 `--api-auth` 는 Forge 에서처럼 아무것도 막지 않습니다.
- **LoRA Manager HTTP 경로도 같은 인증**: `/sam3-lora/config`·`/sam3-lora/spawn` 에 Notebook·메모·Tile & Repair 경로와 같은
  보호를 걸었습니다. 헤더 `X-SAM3-Notebook: 1` 이 없으면 403, `--gradio-auth` 로그인이나 (Forge 가 API 를 띄울 때) `--api-auth`
  HTTP Basic 이 없거나 틀리면 401 이고, 거절된 `/sam3-lora/spawn` 은 매니저 서버를 띄우지 않습니다. 페이지의 Manage 탭은 Gradio 버튼
  브리지를 써서 그대로 동작하고, 이 경로를 부르는 외부 도구는 헤더(와 자격 증명)를 보내야 합니다.
- **Forge 라이트박스 도구줄이 늘 보이던 문제 (버그 수정)**: 전역 테마(Settings → **SAM Extra Appearance**)를 켜면 테마 배경이 Forge 라이트박스의
  도구줄(`.modalControls`, 이것도 `.gradio-container`)까지 칠해, 마우스를 올리지 않아도 바가 배경색으로 늘 보였습니다. 도구줄을 테마 배경에서 뺐습니다.

### 성능 (결과 같음)

- **Anima 3.8B 커넥터 스텝 비용 (결과 같음, 토글 두 개, 기본 켬)**: Settings → **SAM Extra Anima 3.8B** 에 두 설정을 추가했습니다. 끄면 각각 예전
  경로 그대로이고, GPU 에서 둘 다 픽셀 단위로 같았습니다.
  - "Anima 3.8B: 커넥터를 샘플링 동안 fp32 로 상주 (VRAM 약 +1.5 GB, 장당 약 1~2초 빨라짐)" (`sam3_anima38_connector_fp32`): Forge 가
    다 올린 뒤 커넥터를 fp32 로 한 번 바꿔 두고 스텝마다의 캐스트를 건너뜁니다. Forge 가 남긴 여유 안에서만 바꾸고, 내리거나 다시 패치할 때는 먼저 원래 dtype 으로 되돌립니다.
    바뀌면 콘솔에 `[Anima38] connector fp32 resident: +… MB VRAM` 이 한 번 찍힙니다.
  - "Anima 3.8B: 커넥터의 timestep 무관 계산을 프롬프트 줄마다 한 번만 (VRAM 줄당 약 15~70 MB)" (`sam3_anima38_connector_run_cache`):
    의미 특징·source 의 K/V 와 첫 블록을 첫 스텝에 계산해 다시 씁니다(합계 최대 512 MB, 생성이 끝나면 버림). llm_adapter LoRA·번들이 바뀌면 다시 계산하고,
    커넥터가 TE 모듈을 같이 쓰는 폴백에서는 쓰지 않습니다.
  - GPU 시간(사용자 3.8B 설정, 부정 커넥터 켬, 두 번째 렌더): 둘 다 끔 44.3 초 / fp32 만 42.2 초 / 캐시만 43.3 초 / 둘 다 켬 41.7 초(참고: 커넥터
    Bypass 40.2 초).
- **이 릴리즈의 성능 작업 전체**: 사용자 3.8B 설정 한 장이 47.9 → 41.7 초가 됐고, 기준 이미지(DoRA LyCORIS, 고정 시드)를 다시 렌더하면 픽셀 단위로 같습니다. 커넥터
  fp32 상주·run 캐시와 VRAM 상주 외에 v2 run 텐서를 첫 스텝에만 올려 재사용하고 forward 당 GPU→CPU 동기화를 5~7회에서 1회로 줄였으며, Qwen3.5 의미 특징
  캐시를 bf16 그대로 저장해 RAM 을 절반으로 줄였습니다.
- **SAM3**: Detect/Exclude 토큰이 여러 개여도 이미지 백본은 이미지당 한 번만 돕니다. Mask Hull·Mask Outline Px (edge-aware)·작은 반경의 Mask
  Dilation 이 픽셀은 그대로 빨라졌습니다(성분이 많은 1536² Hull 1.1 초 → 약 0.006 초). 인페인트 패스마다 세 번 하던 `synchronize`+`empty_cache` 를
  한 번으로 줄였고, 아티팩트 overlay PNG 는 압축 레벨 2 로 저장합니다(픽셀 같음, 파일 약 9% 큼, 저장 약 0.27 → 0.11 초).
- **가이던스**: PAG/SEG 의 행·head 인덱스 텐서를 캐시합니다. Skimmed CFG 의 스텝당 GPU→CPU 동기화를 없앴던 변경은 원본 수식을 그대로 편입하면서 되돌렸습니다(원본과
  같은 불리언 마스크 인덱싱). 진단(guidance diagnostics)을 끄면 fit_error 와 PAG/SEG rel_delta 를 패스의 첫 스텝에서만 잽니다('비선형 CFG' 경고도 첫
  스텝 기준, `fit_error=?` 로 찍힐 수 있음). 켜면 예전처럼 매 스텝 잽니다.

### 설치 · 라이선스 · 개발

- **첫 설치**: `install.py` 가 `requirements.txt` 에서 **빠진 패키지만** Forge venv 에 설치합니다(예전에는 경고만). 이미 깔린 패키지는 범위 밖이어도
  알리기만 하고 torch 계열은 설치하지 않습니다(`requirements.txt` 에서 torch·py-cpuinfo·protobuf 를 뺌). Anima Tile-Repair 벤더가 불러올 때
  쓰는 `imagesize` 는 `requirements.txt` 에 넣어(벤더와 같은 `==1.4.1`) 없으면 함께 설치합니다(예전에는 시작 로그에 설치 안내만).
  `--sam3-no-auto-install`(COMMANDLINE_ARGS) 또는 `SAM3_NO_AUTO_INSTALL=1` 이면 빠진 패키지와 pip 명령만 출력합니다(LoRA Manager
  경량 deps 도 같음). 진단은 Forge 콘솔(`[forge_sam3_extension]`)에 보입니다. 로컬에 `sam3.pt` 가 없으면 Hugging Face `facebook/sam3`
  에서 받아 씁니다(예전에는 FileNotFoundError). `--sam3-no-huggingface` 면 넣을 위치와 플래그를 안내하는 FileNotFoundError 입니다. 확장 폴더명이
  `forge_sam3_extension` 이 아니면 시작 로그에 경고합니다(`metadata.ini` 콜백 순서가 폴더명에 묶임). LoRA Manager vendor 는 새로 clone 할 때
  커밋 `303cca0`(1.2.0)으로 고정하고 판이 다르면 알립니다.
- **라이선스를 GPL-3.0-only 로**: 확장 전체를 GNU GPL 3판만(SPDX `GPL-3.0-only`, 'or later' 없음)으로 배포합니다. 루트 `LICENSE` 를 추가하고
  README 라이선스 절·`THIRD_PARTY_NOTICES.md` 에 반영했으며, `package.json` 에 SPDX `GPL-3.0-only` 를 적고 README 의 '내부 사용' 문구를
  뺐습니다. 원본 동등성 작업으로 편입한 상류 코드 — Skimmed CFG 수식(Apache-2.0), Detail Daemon 스케줄·σ 조회와 Safe PAG 적용 구간·번호 파싱, DAVE
  초반 스텝 게이트(MIT), CNS 재색칠(GPL-3.0) — 와 대조 테스트용 ComfyUI-DCW 원본 사본(GPL-3.0)의 출처·커밋·고지를 `THIRD_PARTY_NOTICES.md` 에
  적었습니다.
- **새 설정 섹션**: **SAM Extra SAM3**(`sam3_mask`)·**SAM Extra Anima 3.8B**(`sam3_anima38`)·**SAM Extra LoRA**
  (`sam3_lora`)·**SAM Extra Guidance**(`sam3_guidance`)·**SAM Extra Character Reference**(`sam3_reference`) 가
  생겼습니다. README·docs 의 코드 불일치는 문서 쪽만 고쳤습니다.
- **개발**: `requirements-dev.txt` 를 추가했고 CI 는 Python 3.13 + `unittest discover` 로 바뀌었습니다(실제 GitHub Actions 실행은
  미확인). 테스트 사이에 가짜 모듈이 남아 실행 순서에 따라 결과가 흔들리던 문제를 고쳤습니다.
- **검증**: Python 단위 테스트 1434개 통과(skipped 2, CPU, `python -m unittest discover -s tests -t .`), jsdom 프런트엔드 테스트
  57개 통과(`npm test`). GPU(RTX 5090) 확인 — 픽셀 동일: 성능 작업 뒤 3.8B 기준 이미지 재렌더, 3.8B 커넥터 fp32·run 캐시, TF32 수정, SAM3
  인페인트 시드 수정, IP-Adapter 잡 뒤의 txt2img. 구도 같고 잔 디테일만 다름: PAG 앞쪽 블록 중복 제거(56.2 → 51.4 초), SEG separable blur.
  비교·동작 확인: IP-Adapter lineage 정책, IP-Adapter 의 3.8B 커넥터. 이 GPU 측정은 모두 원본 동등성 작업 전의 코드에서 잰 것입니다. 아직 GPU 로 확인하지
  않은 것: Safe PAG CFG 1 변경, VAE 2x, 부분 LoRA 추측 변환, 이어붙이기 권장 LoRA 화질 A/B, 그리고 원본 동등성 작업 전체(Detail Daemon·Safe
  PAG·Skimmed CFG·DCW(+a)·RDC·DAVE·CNS·Tile-Repair·SAM3 LLLite 전처리) — 이것은 원본 코드를 테스트 안에 그대로 두고 같은 입력의 결과를 비교하는
  CPU 단위 테스트(`tests/test_*_origin.py`, `test_skimmed_cfg.py`)로만 확인했습니다. Tile & Repair API·Notebook
  메모장·`--api-auth` 적용·LoRA Manager 경로 인증·ui-config 이전·라이트박스 수정도 단위 테스트(메모장은 jsdom 포함)로 확인했습니다. 그 밖에 GPU
  결과를 적지 않은 항목은 CPU 단위 테스트로만 확인했습니다.

## v0.21.2 — 레거시 콘솔 인코딩에서 로그가 생성을 죽이던 문제 + CI 연결

- **로그 한 줄이 샘플링을 중단시킬 수 있던 문제 수정**: `_log`가 em-dash나 `✅` 같은
  비-ASCII 문자를 그대로 `print`하는데, 레거시 코드 페이지(cp949, cp1252 등) 콘솔에서는
  `UnicodeEncodeError`가 발생합니다. 이 함수는 `_post_cfg`와 model wrapper **안에서**,
  심지어 그들의 `except` 핸들러에서도 호출되므로, 예외가 그대로 전파되어 처리된 폴백이
  하드 실패로 바뀌거나 생성이 죽을 수 있었습니다. `anima_skimmed_cfg.py`와
  `anima_safe_pag.py`의 `_log`가 인코딩 실패 시 ASCII 대체 표현으로 저하되고, 어떤 경우에도
  예외를 밖으로 내보내지 않도록 했습니다. 여러 릴리즈 전부터 있던 선재 결함입니다.
  `PYTHONIOENCODING=cp949`에서 전체 스위트가 통과하는 것과, cp949 stdout을 재현해 훅이
  incoming 결과를 그대로 반환하는지 확인하는 회귀 테스트를 추가했습니다.
- **CI에 프런트엔드 동작 테스트 연결**: v0.21.1에서 푸시 토큰의 `workflow` 스코프가 없어
  빠졌던 `.github/workflows/ci.yml` 변경(`npm ci` + `npm test`)이 이제 포함됩니다.
- **검증**: Python 149개(cp949 포함) + jsdom 6개 통과, `node --check` 전체 통과.

## v0.21.1 — 테마 라이트모드 수정 + 대비 실측 + 프런트엔드 동작 테스트

v0.21.0에서 미뤄둔 접근성·테스트 항목을 처리한 릴리즈. 기능 변경은 없습니다.

- **커스텀 테마가 Gradio `dark` 클래스도 설정**: 모든 팔레트가 어두운데 Forge/Gradio CSS
  상당량이 `document.body`의 `dark` 클래스에 걸려 있고, Gradio는 해석된 테마가 dark일 때만
  이를 붙입니다(기본값은 `__theme=system`). 라이트 모드 세션에서 흰 본문 글자가 라이트 전용
  표면 위에 얹히던 문제(토스트 배경, 코드 하이라이트, `ul.options li.selected`)를 없앴습니다.
  우리가 추가한 경우만 되돌리도록 추적해 Gradio 소유의 `dark`는 건드리지 않습니다.
- **대비를 실측해 정정**: 이전 헤더 주석은 `contrast: pass`로 단정했지만 실제로는 취소 버튼
  레이블이 3.98:1, 컨트롤 테두리가 1.36:1이었습니다. tokens.css의 OKLCH 값을 sRGB로 변환해
  계산한 결과로 교체했습니다.
  - `--sam3-color-error-ink` 96% → 99%, 두 팔레트의 `--sam3-color-error` 59% → 58%로
    취소 버튼 레이블을 **4.50~4.52:1**(WCAG 1.4.3 통과)로 올렸습니다.
  - fast 드롭다운의 오류 표시가 채움용 `--sam3-color-error`를 텍스트 색으로 쓰던 것을
    팔레트의 오류 텍스트 값 `--sam3-color-error-hover`(5.12~5.58:1)로 바로잡았습니다.
  - **알려진 미해결**: `--sam3-color-rule`은 표면 대비 1.36~1.58:1로 컨트롤 경계 기준
    3:1(WCAG 1.4.11)에 미달합니다. 입력 채움이 블록 배경과 1.03~1.07:1이라 테두리가 사실상
    유일한 식별 수단이므로 면제도 성립하지 않습니다. 해소에는 밝기 30% → 약 48% 상향이
    필요해 팔레트의 의도적 시각 변경에 해당하므로 이 릴리즈에서는 문서화만 했습니다.
  - 본문·muted·오류 텍스트·취소 레이블 대비를 **테스트에서 직접 계산**해 고정했습니다.
    주석이 아니라 계산이 지키므로 근거 없는 재인증이 불가능합니다.
- **jsdom 프런트엔드 동작 테스트 도입**: 문자열 검사는 "리스너가 엉뚱한 노드에 붙었다"나
  "패널이 안 닫힌다"를 잡지 못합니다. `package.json`(test 전용, 배포·번들 대상 아님)에
  jsdom을 추가하고 Forge extra-networks 탭 구조를 재현한 DOM에 실제 스크립트를 올려
  **클릭으로** 검증합니다 — Manage 주입, 활성화, native 탭 첫 클릭 닫힘, Gradio가 아무것도
  하지 않는 복귀 경로, 늦게 도착하는 선택 상태(옵저버), 탭 스트립 재생성 후에도 유지.
  위임 리스너를 끄면 6개 중 5개가 실패하는 것으로 실제 회귀 검출력을 확인했습니다.
  `node_modules/`는 제외하고 `package-lock.json`은 추적합니다.
  **CI 연결은 이 릴리즈에 포함되지 않았습니다** — `.github/workflows/ci.yml`에 `npm ci` +
  `npm test` 단계를 추가하는 변경은 푸시 토큰에 `workflow` 스코프가 없어 커밋하지 못했습니다.
  변경 내용은 작업 트리에 남아 있으며, `gh auth refresh -s workflow` 후 커밋하면 됩니다.
- **검증**: Python 148개 + jsdom 6개 통과, `node --check` 전체 통과.

## v0.21.0 — Live Workspaces 폐기 + txt2img Notebook

iframe 기반 Live Workspaces를 걷어내고 단일 Forge 문서 + 서버 저장 Notebook 프리셋으로
교체한 릴리즈. 선택형 다크 테마와 SMC upstream 프리셋도 포함합니다.

- **선택형 Forge 전역 다크 테마**: `Settings → SAM Extra Appearance`에
  `Forge Default / Graphite Ember / Obsidian Violet / Warm Espresso / OLED Mono`를
  추가했습니다. Forge 본체를 수정하지 않고 Gradio 색상 토큰을 확장 CSS로 연결하며,
  Settings 저장 직후 현재 페이지에 적용되고 Forge 옵션 저장소에 유지됩니다. 커스텀 테마는
  단색 표면·즉시 포커스 링·최소 모션·공통 상태 규칙을 공유하고, `Forge Default`는 전역
  덮어쓰기를 완전히 제거합니다.
- **단일 Forge 문서로 복귀**: `/sam3-live` 셸, iframe, 자동 리다이렉트, Workspace 스냅샷과
  관련 JavaScript/Python 라우트를 제거. 원래 Forge 루트 주소와 생성 큐·진행 미리보기를 그대로
  사용합니다.
- **Notebook 프리셋**: Gallery 아래에서 프리셋 추가·이름 변경·삭제·검색·내보내기/불러오기를
  지원. 프리셋 하나에 프롬프트, 네거티브, LoRA, XYZ X/Y/Z의 축 유형+값, 주요 생성 설정을
  여러 항목으로 묶어 현재 화면에 적용할 수 있습니다.
- **안전한 영구 저장**: Forge data 경로의 `sam-extra/notebook.json`에 revision 기반으로
  원자 저장하고 직전 세대 `.bak`, 저장 상태, 충돌 방지, 직전 적용 되돌리기를 추가했습니다.
  Forge/Gradio 로그인 가드와 same-origin 읽기·저장 헤더를 적용하고, 동기 파일 I/O는 작업
  스레드로 분리했습니다. 원본·백업이 모두 손상되면 빈 데이터로 덮어쓰지 않으며 저장소 I/O
  오류도 경로를 노출하지 않는 제한된 500 응답으로 처리합니다.
- **적용 안전성과 이전 데이터 이식**: 프리셋 전체를 사전 검사하고 도중 실패 시 이미 바뀐 값을
  원자적으로 복구합니다. Undo는 XYZ가 강제로 바꾸는 Script 선택·CSV 모드까지 되돌립니다.
  기존 `sam-extra.workspace-manager.v1` 데이터가 있으면 명시적 가져오기 버튼으로 지원 항목을
  Notebook 프리셋에 추가하며 원본 브라우저 데이터는 보존합니다. 프롬프트 안 LoRA 태그의
  위치를 유지하고 지원 밖 컨트롤·빈 Workspace·한도 초과분은 가져오기 전에 개수로 알립니다.
- **편한 3열 UI 유지**: Parameters / Scripts / Gallery 재배치는 유지하되 always-on 확장은
  Parameters 아래에 두고 Script 선택기와 내장 Script 패널만 가운데에 둡니다. Forge의
  Default/Compact 프롬프트 레이아웃을 모두 감지하며 Compact의 붙여넣기·지우기·스타일·토큰
  도구 행도 프롬프트와 함께 옮깁니다. 기존 레이아웃만 남은 재마운트에서도 Notebook 패널을
  Gallery 아래에 다시 연결합니다.
- **Forge 원본 탭 복구**: settings/gallery를 분리한 뒤 `#txt2img_extra_tabs`를
  Prompt/Negative 아래로 옮겨 Generation / Textual Inversion / Checkpoints / Lora 탭을
  계속 사용할 수 있게 했습니다. 재마운트 이중 이동과 숨은 탭 회귀도 테스트합니다.
- **LoRA Manager 단순화**: Workspace 자식 감지와 셸 메시지 중계를 제거하고 단일 Forge 페이지에
  Manage 탭과 프롬프트 삽입 브리지를 한 번 주입합니다.
- **SMC upstream 프리셋 이식**: ComfyUI-DCW의 `Off / Auto / 모델별 / Custom` 선택과
  모델 감지 규칙을 추가했습니다. Forge `Anima`는 `Cosmos / Wan`으로 자동 판별되어
  `lambda=6.0, k=0.20`이 적용되며, Custom 범위도 upstream과 맞췄습니다. 기존 SMC 체크박스,
  API/infotext 인자와 XYZ 축은 호환 영역에 유지하고 새 preset 인자는 맨 뒤에 추가했습니다.
  새 Custom UI 기본 `k=0.10`과 별개로, 구버전 호출이 기존 index 27을 생략한 경우에는 역사적
  폴백 `k=0.20`을 유지합니다.
- **Guidance 패널 순서**: 실행 수학은 upstream 호환인 `SMC → APG → CWM → DCW`를
  유지하면서 화면 순서는 `DCW → CWM → SMC`로 정리했습니다. XYZ 축 목록은 **정수 인덱스가
  저장되는 호환 표면**이므로 v0.20.0의 46개 순서를 그대로 유지하고 `[Anima SMC] Preset`만
  맨 뒤에 append했습니다. 축 순서를 고정하는 회귀 테스트를 추가했습니다.
- **드롭다운 백엔드 이중 실행 수정**: 값을 쓰면 Gradio가 이미 change+input을 발생시키므로
  (`handle_change`, `value_is_output`은 서버 출력에서만 참), 수동 dispatch를 값이 실제로
  바뀌지 않은 경우로 제한했습니다. 체크포인트 선택이 모델을 두 번 로드하던 문제가 사라집니다.
  multiselect는 검색 입력이 항상 비어 있어 `.token`에서 선택값을 읽습니다(VAE/Text Encoder).
- **fast 드롭다운 선택지 갱신**: 프록시가 실제 컨트롤을 숨기므로 설치 시점 목록에 고정되던
  문제를 수정했습니다. `window.gradio_config`가 실시간 갱신되는 점을 이용해 보관한 배열
  참조만 O(1)로 비교하고 달라졌을 때만 다시 매핑합니다. 빈 목록으로의 갱신도 반영합니다.
  외부 주입 목록(`/sdapi/v1` 모델 드롭다운)만 이 경로에서 제외됩니다.
- **LoRA Manager 탭이 닫히지 않던 문제 수정**: Gradio 4.40은 비선택 TabItem을 DOM에 두고
  인라인 `display`만 바꾸므로 합성 Manage 패널은 이 확장이 직접 숨겨야 합니다. 주입 시점
  노드를 캡처하는 방식을 버리고 컨테이너 위임 리스너 + 클릭 시점 노드 조회로 바꿨고, 다른
  탭 클릭 시 Gradio flush를 기다리지 않고 즉시 닫습니다. 순서를 통제할 수 없는 경우를 위해
  컨테이너 MutationObserver를 안전망으로 두었습니다.
- **스크립트 float에서 `z-index` 제거**: 정수 `z-index`는 stacking context를 만들어
  `position:fixed`인 Gradio 팝업까지 그 레벨에 갇히게 합니다. DOM 순서만으로 충분합니다.
- **`.gitignore`**: 어시스턴트 작업 디렉터리(`.codex/`, `.hallmark/`, `.agents/`)와
  `.claude/anima_*.json`을 제외해 배포 저장소에 섞이지 않게 했습니다.
- **검증**: 회귀 테스트 **146개** 통과, `node --check`·`py_compile`·제어문자 검사 통과,
  삭제된 모듈에 대한 dangling 참조 0개. 브라우저 실사용 확인(모델 로드 1회, Manage 탭 닫힘)은
  이 환경에서 완료하지 못했습니다 — 재시작 + 하드 리프레시 후 확인이 필요합니다.

## v0.20.0 — Skimmed/PAG 순서 복구 + Anima CLIP Modulation

- **Skimmed CFG가 PAG 계열을 지우던 실제 원인 수정**: Forge Neo의
  `ScriptRunner.process_before_every_sampling`이 두 번 정의돼 마지막 비정렬 구현이
  `sorting_priority`를 무시하는 문제를 확장 내부에서 우회. Skimmed post-CFG callback을
  owner tag로 dedupe한 뒤 목록 맨 앞에 prepend해 실제 순서를 항상
  `Skimmed → Safe PAG`로 고정했습니다. Forge 코어 파일은 수정하지 않습니다.
- **PAG scale 회귀 검증 강화**: 종전 `PAG → Skimmed` 순서는 scale 0/8 출력이 bitwise
  동일해지는 반면, 수정된 `Skimmed → PAG` 순서는 non-zero RMS 차이가 생기는 것을 실제
  post-CFG 체인 테스트로 고정. v0.19.1의 `cond_raw` 격리도 유지해 skim된 cond 오염을
  다시 막습니다.
- **Anima Modulation Guidance 추가**: 별도 768차원 CLIP-L의
  `base + w × (positive − negative)` 방향을 공개 Cosmos 어댑터 MLP/scales로 투영해
  선택한 Forge Anima block의 `adaln_lora_B_T_3D`에 가산합니다. 메인 Qwen conditioning은
  교체하지 않으며 기본 OFF입니다.
- **CLIP/어댑터 자산 경로**: `models/text_encoder`의 safetensors 헤더를 검사해 실제
  CLIP-L만 드롭다운에 표시하고 Anzhc Anime encoder를 우선합니다. 공식
  `yresearch/cosmos-pooled/checkpoint_4000.pt`는 첫 사용 시
  `models/anima_modulation_guidance/`로 내려받아 크기와 SHA-256을 검증합니다.
  local adapter는 `weights_only=True`로 필요한 tensor만 읽습니다.
- **Forge 전용 호환 계층**: ComfyUI Cosmos와 달리 Forge `Anima`에는
  `model_channels` 속성이 없으므로 실제 `Block.x_dim`에서 2048 폭을 추론합니다.
  CLIP/adapter 투영은 generation당 CPU에서 한 번만 수행하고 sampler 중에는 작은
  per-block vector만 model device/dtype으로 캐시합니다.
- **UI/XYZ/기록**: CLIP 모델, direction weight, block 범위, base/positive/negative prompt,
  adapter source를 UI에 추가하고 `[Anima Mod] Enable/Weight/Start/End` XYZ 축,
  PNG infotext와 opt-in verification block-hit 로그를 연결했습니다.
- **검증**: 회귀 테스트 **103개** 통과, Gradio 4.40에서 script argument 56개 UI 생성 확인.
  실제 로컬 권장 CLIP-L + 공식 어댑터로 `(28, 6144)` finite modulation 투영까지 확인했습니다.
  실제 checkpoint 이미지 품질 A/B는 아직 별도 검증 범위입니다.

## v0.19.1 — PAG cond 오염 수정 + Skimmed CFG XYZ

- **PAG/SEG 보정항 오염 수정**: `_apply_perturbation`이 weak 예측을 만들 때 사용한 원본
  cond(`_STATE["cond_raw"]`)와 차분하도록 변경. Skimmed CFG처럼 우리보다 먼저 도는
  post-CFG 훅이 Forge의 `cond_denoised`를 제자리에서 덮어쓰면, 덮어쓰인 cond와 차분하면서
  상수 오프셋이 섞여 scale·strength 반응이 둔해지던 문제를 제거. 캡처 텐서가 없거나 형상이
  다르면 기존 `args["cond_denoised"]` 경로로 폴백. 회귀 테스트 2개 추가.
- **Skimmed CFG XYZ 축 추가**: `[Anima Skim] Enable / Skimming CFG / Full Skim Negative /
  Disable Flipping Filter / Start / End / Flip At` 7종을 xyz_grid에 등록.

## v0.19.0 — LoRA 벌크/컨텍스트 전송을 Forge로 (ComfyUI 하드코딩 제거)

멀티 선택 후 우클릭 "한번에 넣기"가 ComfyUI로만 가서 못 쓰던 문제를 해결. 매니저의 모든
"send to workflow" 경로를 Forge 프롬프트 삽입으로 연동.

- **컨텍스트 메뉴 인터셉트**: 주입 브리지(`forge_bridge.js`)가 이제 카드 1개의 paper-plane뿐
  아니라 **단일 컨텍스트 메뉴**(`#loraContextMenu`의 `sendappend`/`sendreplace`)와
  **멀티 선택 벌크 서브메뉴**(`#bulkContextMenu`의 `send-to-workflow-append/replace`, 대상은
  `.model-card.selected` 전체)를 capture 단계에서 가로채 ComfyUI 전송을 막고 Forge로 postMessage.
  대상 카드의 `data-file_name`/`data-folder`/`data-usage_tips`로 `<lora:...>` 문법을 만들어
  전송(벌크는 콤마 결합).
- **Append/Replace 지원**: 메시지에 `replace` 플래그를 실어, Replace면 프롬프트의 기존
  `<lora:...>` 토큰을 제거한 뒤 새 세트를 넣음(Append는 기존대로 이어붙임). Live 셸도 이
  플래그를 활성 워크스페이스로 그대로 전달.
- **라벨 정리**: 벌크 서브메뉴 라벨(`loras.bulkOperations.*`)도 "Add LoRA"로 리브랜드 대상에 추가.
- **적용 시점**: 브리지는 서버 spawn 시 벤더 트리에 재기록(content 기준 idempotent)되므로,
  업데이트 후 다음 매니저 기동부터 자동 반영.
- **검증**: 회귀 테스트 88개 전부 통과(브리지 인터셉트·Forge측 replace 처리 자산 검사 포함).
  실제 멀티 선택 전송은 브라우저+Forge에서 확인 필요.

## v0.18.0 — LoRA Manager ↔ Live Workspace 연동

LoRA 매니저를 "iframe에 얹은 외부 앱"에서 **Live Workspace-인식 통합**으로 한 단계 끌어올린
릴리즈. 벤더 앱/서버 자체는 그대로 쓰되 연동을 실제 Forge Neo 흐름에 맞춤.

- **HTTP config/spawn 엔드포인트**: `/sam3-lora/config`, `/sam3-lora/spawn`(same-origin JSON)을
  추가해 경량 Live 셸이 히든 Gradio 브리지 버튼 없이 서버를 조회·기동. 기존 `get_or_spawn`
  라이프사이클을 그대로 래핑(일반 모드용 Gradio 브리지도 같은 payload 공유). 신규 테스트 2개.
- **Live 셸 공유 오버레이**: 셸 헤더의 `LoRA` 버튼이 **매니저 하나**를 오버레이로 엶(이전엔
  워크스페이스마다 중복 주입되어 `셸→워크스페이스→매니저` 3중 iframe이었음). 벤더 미설치 시
  버튼 자동 숨김.
- **활성 워크스페이스로 삽입 라우팅**: 매니저에서 Add LoRA → 셸이 그 메시지를 **현재 활성
  워크스페이스 iframe**의 프롬프트로 postMessage 전달(cross-origin은 source+shape로 검증).
- **중복 탭 억제**: Live 워크스페이스 자식 iframe에서는 `lora_manager.js`가 Manage 탭을 주입하지
  않음(네이티브 탭·일반 모드는 기존대로 자체 탭 유지). 삽입 브리지는 항상 리슨.
- **검증**: 회귀 테스트 85개 전부 통과(신규 route·shell 연동 자산 검사 포함). 실제 매니저
  기동·삽입 동작은 브라우저+Forge에서 확인 필요(이 환경 미검증).

## v0.17.0 — CI·개발 인프라 + 안정성 보강 + 정리

기능 추가 없이 안전망·정확성·정리에 집중한 릴리즈.

- **CI 도입**: `.github/workflows/ci.yml`이 push/PR마다 pytest(CPU torch+gradio) +
  `node --check`를 실행. 이전엔 회귀 테스트 83개가 자동으로 안 돌았음.
- **웹 세션 SessionStart 훅**: `.claude/hooks/session-start.sh`가 Claude Code on the web
  세션에서 테스트 의존성을 자동 설치(멱등·remote 전용). `.claude/settings.json`에 등록.
- **args 검증 강화**: `sam3_device`(auto/cpu/cuda/cuda:N, 그 외 auto로 폴백), seed 범위
  클램프, inpaint width/height 8의 배수 스냅, CN guidance start>end 자동 swap — 모두
  raise 대신 정규화(호출부가 검증 실패 시 SAM3를 꺼버리므로). 신규 테스트 5개.
- **Anima 전역 전략 복원**: `TokenizeStrategy`/`TextEncodingStrategy` 싱글턴을 Anima 패스
  전후로 `try/finally` 복원 — 이후 비-Anima 경로로의 상태 누수 방지.
- **guidance 패치 teardown 프레임워크**: Safe PAG의 attention/block/self_attn + k-diffusion
  noise 전역 monkey-patch에 clean uninstall 경로 추가. `on_script_unloaded`에 등록해 reload
  시 stale 패치 제거(런타임 경로는 그대로). install→teardown 테스트 추가.
- **정리**: `!sam3.py`의 동일 JS shim 4벌 → `_SELECTED_INDEX_JS` 하나로. install.py에
  벤더 pin 훅(`_ANIMA_PIN`/`_LM_PIN`, 기본 None=기존 동작) 추가. LoRA 모듈 버전 드리프트
  문구 정리.
- **실험 기능 진단 문서**: [docs/EXPERIMENTAL_STATUS.md](EXPERIMENTAL_STATUS.md) — Refine·Anima의
  전제 조건과 실제 Forge 실행으로만 확인 가능한 항목·캡처할 로그 정리.
- **검증**: 회귀 테스트 83개 전부 통과. 브라우저/GPU E2E는 이 환경에서 확인 불가.

## v0.16.0 — Live Workspace 기본화 + 모드 선택 + 탭 전환 부드럽게

기능 5를 Live Workspace 중심으로 재편하고, 인-페이지 툴바(비-Live UI)를 폐기하며,
Live 탭 전환 버벅임을 줄인 릴리즈. 코드 중복도 일부 정리.

- **모드 선택 설정**: 기존 on/off 토글(`sam3_workspaces_enable`)을 `Settings → SAM3 Workspaces`의
  라디오 `sam3_workspaces_mode`(`Live Workspace` 기본 / `기본 Forge UI`)로 교체. `Live Workspace`는
  `/`를 경량 `/sam3-live` 셸로 리다이렉트하고, `기본 Forge UI`는 리다이렉트 없이 순정 Forge를 유지.
  리다이렉트 결정은 `window.opts`가 로드되기 전이라 서버 프로브 `/sam3-live/enabled`(설정을 요청
  시점에 읽음)로 처리.
- **비-Live 인-페이지 툴바 폐기**: 이전 `?sam3_live=off` 경로의 워크스페이스 툴바(Mode D)를 제거.
  `createToolbar`와 셸의 `기본 UI` 전환 버튼 삭제, `mountToolbar`는 Live 자식 프레임만 처리.
  이후 호출자가 사라진 툴바 전용 헬퍼 함수 8개(`switchWorkspace`/`createWorkspace` 등, ~180줄)와
  `.sam3-workspace-*` 툴바 CSS(복원 상태 클래스 `.sam3-workspace-restoring` 제외)도 제거.
  워크스페이스 전환은 이제 Live 셸에서만 이뤄지며, 저장 로직·`실제 탭으로 열기`(네이티브 탭)는 유지.
- **탭 전환 부드럽게(버벅임 완화)**:
  - *숨겨진 워크스페이스 일시정지*: 셸→자식 `visibility` postMessage로 비활성 iframe의
    MutationObserver+800ms 폴링을 멈추고 활성 시 재개(필수 재-마운트 경로인 Forge `onAfterUiUpdate`는
    항상 유지). 세 개의 살아있는 Forge 문서가 계속 CPU를 태우던 문제 완화.
  - *inert 토글 최소화*: `activate()`가 모든 iframe이 아니라 바뀐 두 프레임(이전·새 활성)만
    inert/aria 갱신 → 전환마다 발생하던 style/a11y 리플로우 감소.
  - *인접 탭 선-빌드*: 배경 프리로드가 활성 탭의 가장 가까운 이웃부터 로드.
- **중복 코드 정리**: Refine·Anima의 `_as_float`/`_as_int`를 `sam3ext/coerce.py`로 통합.
- **검증**: 회귀 테스트 77개 전부 통과(신규 route 프로브·모드 게이트·전환 개선 자산 검사 포함).
  브라우저 E2E(실제 Live 셸/탭 전환)는 이 환경에서 확인하지 못했습니다.

## v0.15.0 — Workspace 토글·갤러리 타이밍, 충돌 정리 + 리뷰 버그 수정

코드 리뷰에서 나온 런타임 충돌·버그를 정리하고, txt2img Workspaces(기능 5)의 제어를
개선한 릴리즈.

- **아코디언 정렬 고정**: SAM3 계열 확장 아코디언을 연속된 음수 `sorting_priority` 블록으로
  묶어 SAM3 바로 밑에 차례대로 배치. `SAM3(-30) → Detail Daemon(-29) → Skimmed CFG(-28)
  → Safe PAG(-27) → VAE 2x(-26) → Reference PoC/로그 토글(-25)`. 기존엔 SAM3 mask에
  우선순위가 없어 guidance 계열이 우선순위 없는 타 확장 밑으로 밀려 맨 아래 렌더됐음.
- **Workspaces 설정 토글**: Settings → `SAM3 Workspaces`에 `txt2img Workspaces 활성화`
  옵션(`sam3_workspaces_enable`)을 추가. 끄면 `workspace_manager.js`가 `window.opts`를
  읽어 툴바/탭 마운트를 통째로 건너뜀(페이지 새로고침 후 적용).
- **갤러리 비움 타이밍 변경**: 생성 버튼을 누르는 즉시 이전 갤러리를 감추던 동작을 제거.
  이제 이전 결과를 그대로 두고 Forge live preview가 위에 겹쳐지며, **새 이미지가 완성될
  때** 최종 결과로 교체됨. 사용하지 않게 된 hide-on-generate 로직(JS·CSS)도 제거.
- **PAG 자동 감쇠(안전 브레이크) 제거**: PAG/SEG + SLG 병용 시 각 scale을 활성 항 수로
  나누던 `auto_decay` 토글을 삭제. perturbation은 항상 설정된 full scale로 적용됨. 스크립트
  인자 index는 inert placeholder로 보존해 append-only 계약 유지.
- **guidance 스택 충돌 점검**: PAG·DCW·CWM·SMC·Skimmed CFG 동시 사용이 서로를 무력화하지
  않음을 확인하고 회귀 테스트로 고정(Skimmed→Safe PAG 순서 불변식, 각 단계 기여 검증).
- **런타임 충돌 수정**: (1) unet `model_function_wrapper` 단일 슬롯을 두 스크립트가 덮어쓰던
  문제 — Safe PAG가 우선권을 갖고 경고 로그를 남기며, Ref PoC는 기존 wrapper가 있으면
  yield. (2) CNS 노이즈 패치의 `continue`가 `break`를 건너뛰어 두 k-diffusion 사본을 이중
  패치하던 버그 수정. (3) Anima VAE 2x wrapper 이중 wrap 방지(원본 VAE 재-wrap).
- **리뷰 버그 수정**: inpaint noise multiplier의 `0.0`→`1.0` falsy 강제 제거; Refine
  `inherit_main_neg_prompt` 폴백이 위젯 기본값과 반대로 뒤집히던 문제; Anima 랜덤 시드(-1)
  재현성(명시적 시드 선택); `write_artifacts`가 개별 마스크를 덮어쓰던 free-slot 탐색;
  `unload_sam3`의 명시적 CPU 이동; LoRA Manager 이중 spawn·health false-positive·로그 핸들
  누수; ControlNet `global_state` 등록 idempotent화; 중단된 vendor clone 자가복구.
- **검증**: 회귀 테스트 74개 전부 통과(guidance 조합 3개 신규). 실제 생성 E2E는 아직
  확인하지 않았습니다.

## v0.14.0 — 독립 CFG base 토글 + Skimmed CFG

상호배타였던 CFG base 라디오를 독립 토글로 분해해 SMC·APG·CWM을 자유롭게 조합할 수 있게
하고, anti-burn 기능인 Skimmed CFG를 별도 아코디언으로 추가한 릴리즈.

- **SMC·APG·CWM을 독립 토글로 분리**: 상호배타였던 `CFG base mode` 라디오를 대신해
  `Enable SMC` / `Enable CWM` 체크박스를 추가하고, 기존 `Enable APG`와 함께 원하는 조합을
  동시에 켤 수 있게 함. 켜진 것들은 항상 `SMC → APG → CWM` 순서로 적용되며, 셋 다 끄면
  incoming CFG를 그대로 보존. `Experimental stack` 없이도 APG+CWM, APG+SMC 조합 가능.
- **파라미터 재배치**: `CWM / SMC Advanced` 아코디언을 해체해 SMC lambda/k는 SMC 토글
  아래, CWM alpha low/high는 CWM 토글 아래로 이동.
- **하위 호환 유지**: 라디오와 `Experimental stack`은 `Legacy CFG base mode` 아코디언에
  남겨 새 토글과 OR로 합침. 스크립트 인자는 뒤에 append해 저장된 infotext·API 호출·
  기존 XYZ 그리드가 그대로 동작. XYZ에 `[Anima SMC] Enable`·`[Anima CWM] Enable` 추가.
- **Skimmed CFG 추가**: [Extraltodeus/Skimmed_CFG](https://github.com/Extraltodeus/Skimmed_CFG)의
  공개 수식을 Forge용으로 재작성한 anti-burn 기능을 `Anima Detail Daemon` 바로 아래
  독립 아코디언으로 추가. upstream은 ComfyUI pre-CFG 노드지만 Forge의 pre-CFG 계약이
  달라 post-CFG에서 동일 수식을 재구성. skim 결과를 Forge의 예측 tensor에 다시 써서
  **SMC/APG/CWM·PAG delta·DCW와 동시에 사용 가능**(ComfyUI에서 pre-CFG 노드를 물린 것과
  같은 조합). 정렬 우선순위로 Safe PAG보다 먼저 실행되도록 보장.
- **검증**: 회귀 테스트 71개 전부 통과(신규 Skimmed CFG 12개는 상단 수식 transcription과
  tensor 단위 일치 및 downstream 전파를 확인, 신규 CFG base 토글 5개 포함). 실제 생성
  E2E는 아직 확인하지 않았습니다.

## v0.13.0 — 실제 Workspace 탭 + Guidance 제어·UI 완성

Live Workspaces의 iframe 전환이 무거운 환경을 위해 같은 WebUI 포트의 실제 브라우저 탭으로
전환하는 경로를 추가하고, PAG/SEG 공식 구현의 전체 제어값과 안전 힌트를 UI에 노출한 릴리즈.

- **실제 브라우저 탭 모드**: Live 헤더의 `실제 탭으로 열기`가 로드된 Workspace를 먼저
  강제 저장한 뒤 현재 셸을 활성 Workspace로 바꾸고 나머지를 최상위 브라우저 탭으로 엶.
  탭마다 고정 slot·이름·자동 저장 상태를 표시하며 iframe은 0개가 되어 브라우저 기본 탭
  전환과 같은 경로를 사용. 팝업 차단은 감지해 열린 수와 허용 안내를 표시.
- **Live Workspace 안정성·UX**: 세 iframe을 모두 한 번 준비하되 비활성 화면은 표시만
  전환하고, 자동 저장 상태를 Live 헤더에 전달. Forge 상단 txt2img/img2img/PNG Info/
  Settings/Extensions 탭을 유지하고 확장 패널은 Parameters 아래, Forge 기본 Script/XYZ만
  중앙 Scripts에 배치.
- **Forge 갤러리 동작 보존**: Generate 시 이전 결과만 숨기고 Gallery 루트를 유지하여
  Waiting/Queue/진행률/중간 미리보기가 Forge 기본 경로로 표시된 뒤 이번 최종 결과만 남김.
- **공식 PAG/SEG 전체 제어**: Attn Scale, 공식 perturbation strength, block/head indices,
  Start/End, Rescale, full/partial rescale mode를 UI와 XYZ에 일치시킴. Legacy Soft PAG/
  SEG-Approx strength는 호환 아코디언으로 분리.
- **Guidance UI 정리**: DCW/DAVE/CNS를 중첩 탭 밖의 주 패널로 이동하고 SLG/APG/
  Adaptive/CWM/SMC를 포함한 각 수치 항목에 깨짐·과채도·구도 변화가 보일 때의 조절 방향을
  설명하는 맞춤 힌트를 추가.
- **검증**: Python 회귀 테스트 54개와 JavaScript 문법 검사 통과. 실제 `7860` Forge에서
  Workspace 1/2/3이 각각 별도 최상위 탭, 고정 slot, 개별 제목·저장 상태, iframe 0개로
  열리는 것을 확인. Forge 상단 탭 유지 및 새 `Traceback`/`KeyError` 0개 확인.

## v0.12.0 — 동적 Live Workspaces + RK/TDE Gradio 가드

Live Workspaces를 고정된 iframe 3개에서 동적으로 관리 가능한 작업공간 셸로 확장하고,
`--api` 시작 경로에서 타 샘플러 확장이 남기던 미등록 Gradio 컴포넌트 오류를 차단한 릴리즈.

- **경량 `/sam3-live` 셸**: 사용하지 않는 부모 Forge UI 한 벌을 먼저 만드는 구조를 제거.
  현재 선택한 Workspace를 우선 로드하고 나머지는 순차 백그라운드 준비. 같은 포트와 Forge
  서버를 그대로 사용하며 기존 루트 주소는 경량 셸로 자동 전환.
- **동적 Workspace 관리**: 기본 1/2/3과 기존 저장 데이터는 유지하면서 최대 20개까지 추가.
  현재 설정 복제, 이름 변경, 삭제, JSON 내보내기/가져오기를 Live 헤더에서 직접 수행.
- **시작 복원 경량화**: Gradio가 이미 파싱한 `window.gradio_config`를 재사용해 iframe마다
  약 5.5 MB `/config`를 중복 요청하지 않음. 갤러리 초기화는 설정 복원을 막지 않으며,
  Script/XYZ 의존성은 실제 값이 달라진 드라이버만 재실행하고 불일치가 있을 때만 검증 대기.
- **활성 화면 우선**: 로컬 Forge Neo 실측에서 활성 Workspace가 약 6초에 준비됐고, 나머지는
  활성 화면을 방해하지 않도록 순차 준비. 모든 Workspace는 계속 독립된 Gradio 문서이므로
  값 바꿔치기 없이 전환되고 Generate는 현재 화면 하나만 실행.
- **RK/TDE `KeyError` 가드**: Forge `--api`가 임시 `gr.Blocks`에서 script `ui()`를 재실행할 때
  RK Sampler/TDE Sampler의 모듈 리스트에 섞이는 throwaway 컴포넌트를 실제
  `modules.script_loading.loaded_scripts`와 callback globals에서 찾아 `demo.load` 등록 직전에
  in-place 제거. Forge 코어나 두 외부 확장 파일은 수정하지 않음.
- **UI 및 저장 안정성**: 확장 UI는 Parameters 아래에 유지하고 중앙 Scripts에는 Forge 기본
  Script/XYZ만 배치. 프롬프트·네거티브·XYZ 상태, 현재 세션 마지막 갤러리, 충돌 보호와
  서버 재시작 자동 복구를 동적 Workspace에서도 유지.
- **검증**: Python 회귀 테스트 46개, JavaScript 문법 검사, 실제 브라우저에서 1/2/3 전환,
  Prompt/Negative/XYZ 복원, 추가·이름 변경·삭제·내보내기 통과. 클린 Forge 부팅에서
  미등록 dependency 0개, `KeyError`/`Traceback` 0개 확인.

## v0.11.0 — Anima Guidance Suite 공식 모드 + Live Workspaces

Forge Neo 코어 파일을 수정하지 않고 Anima guidance 실행 경로와 단일 탭 다중 작업공간 UI를
확장 내부에서 완성한 릴리즈.

- **공식 PAG/SEG + SLG**: PAG는 appended weak row의 hard value-only attention, SEG는 실제
  Anima T/H/W 중 H/W query Gaussian blur로 동작. 기존 soft PAG / SEG-approx는
  `Legacy Soft/Approx` 호환 토글로 분리.
- **Guidance 오케스트레이터**: Preserve/APG/CWM/SMC/SMC+CWM CFG base, DCW, DAVE,
  CNS-inspired wavelet noise, Adaptive Guidance를 고정된 순서와 generation 단위 상태 정리로
  통합. 모든 기능은 기본 OFF이며 중립 설정은 기존 Forge 결과를 보존.
- **최신 Anima attention hook 복구**: `SelfCrossAttention.torch_attention_op`의 실제
  value/output 레이아웃과 staticmethod binding을 보존하면서 weak row에만 perturbation 적용.
  훅 미도달·shape 불일치는 원본 결과로 안전하게 폴백.
- **검증 로그**: 확장 하단의 debug/안전 아코디언에 opt-in Guidance verification summary를
  추가. PAG/SEG/SLG 적용 스텝, APG/Adaptive, DCW/DAVE/CNS 실행 여부를 생성 종료 시 요약.
- **Live Workspaces**: 같은 탭 안에 독립된 txt2img 문서 3개를 유지하여 값 바꿔치기 없이
  즉시 전환. prompt/negative, 생성 설정, Script/XYZ 상태, 마지막 생성 갤러리를 작업공간별
  보존하며 Generate는 현재 화면 하나에서만 실행.
- **Workspace UI 수정**: Prompt/Negative/Generate와 Parameters/Scripts/Gallery 3열 배치를
  원래 `#tab_txt2img` CSS 범위 안에 유지해 찌그러짐과 겹침을 제거. 모든 확장 UI는
  Parameters 아래에 두고 중앙 Scripts에는 Forge 기본 Script 선택기와 기본 패널만 표시.
- **갤러리 수명주기**: Generate 직전에 이전 결과를 비우고 이번 생성 결과만 유지. 페이지/WebUI
  재시작 시 갤러리를 비우며 서버 재시작 복구 시 세 iframe을 자동 재연결.
- **기타 안정성**: ForgeCanvas가 없는 Refine 배선에서 prompt 입력이 canvas 값으로 밀리던
  인덱스 오류 수정, `sam3ext.guidance` import가 SAM3 모델 의존성을 초기화하지 않도록
  package import를 지연 로딩으로 변경.
- **검증**: Forge Neo 2.27 + `anima_baseV10` 실제 경로 확인, Python 회귀 테스트 38개와
  JavaScript 문법·실제 브라우저 DOM/레이아웃 검증 통과.

## v0.10.0 — txt2img Workspaces (단일 탭 작업공간 3개)

여러 브라우저 탭을 다시 열고 설정을 복사하던 흐름을 대체하는 확장 전용 작업공간 관리자를 추가.
Forge Neo 기본 파일을 수정하지 않고 한 탭에서 Workspace 1/2/3을 전환합니다.

- positive/negative prompt, seed·steps·sampler·scheduler·크기 등 txt2img 생성 설정을 작업공간별 저장.
- 선택한 Script와 X/Y/Z Plot 축·값·옵션까지 함께 저장하고 복원.
- 입력 변경 시 브라우저 로컬 저장소에 자동 저장. 빈 작업공간의 첫 전환은 현재 설정을 복제.
- 같은 출처의 여러 탭이 동일 작업공간을 수정할 때 최신 저장본을 덮지 않는 충돌 보호.
- Workspaces 바를 Generation 리사이즈 행 위의 독립된 전체 너비 행으로 배치해 설정 UI 변형 방지.
- 이미지·파일·갤러리·생성 결과/output과 checkpoint/VAE 등 전역 Quicksettings는 저장 대상에서 제외.
- 저장소는 프로토콜·호스트·포트가 모두 같은 동일 출처에서만 공유. 다른 주소나 브라우저
  프로필로 옮길 수 있도록 내보내기/가져오기 제공.

## v0.9.18 — Anima PAG 검정 실루엣 붕괴 수정

`[Anima Pert] Enable=True`에서 이미지가 검게 붕괴하던 PAG 실행 경로를 Forge Neo의
실제 denoised(x0) 훅·상류 Safe-PAG 수식에 맞게 수정.

- **핵심 원인 1 — 잘못된 rescale**: 기존은 매 스텝 `CFG base + PAG correction`
  전체에 스케일 팩터를 곱해 밝기/에너지를 반복적으로 0 방향으로 빼앗음. 상류와
  동일하게 팩터는 **PAG correction에만** 적용하여 correction=0이면 CFG base가
  bit-identical로 보존됨.
- **핵심 원인 2 — 과도한 기본 블록**: 빈칸을 28블록 후반 전체 `14-27`로 해석하던
  로컬 동작을 상류 권장 단일 블록 `18`로 변경. UI/SLG/문서도 동일하게 맞춤.
- **x0 직접 보정**: Forge `model.apply_model` 결과는 이미 predictor 변환된 x0이므로,
  잘못된 raw-output/`c_out` 추정을 제거하고 `cond_x0 - weak_x0`를 직접 사용. 이로써
  Forge의 CFG=1 uncond 생략 경로에서도 PAG가 실제 적용됨.
- **XYZ 상태 누수 수정**: `p.extra_generation_params`를 재사용하는 True/False 셀 사이에
  이전 PAG/APG/AdaptiveG infotext가 남지 않도록 셀별 정리.
- **안전성**: XYZ/API에서 slider 범위를 우회한 NaN/Inf/과도한 scale·strength·range·
  rescale 입력을 유한값 + UI 범위로 정규화. 첫 유효 weak delta와 generation summary
  로그를 추가해 무효 훅을 즉시 구분할 수 있음.
- **검증**: PAG 회귀 테스트 8개 통과. `anima_baseV10` 실제 생성(20 steps, seed 동일)에서
  PAG False/True 변경 픽셀 98.58%, True 평균 RGB 103.89로 검정 붕괴 없이 weak/apply
  14/14 스텝 실행을 확인.

## v0.9.17 — Gradio `state_holder` KeyError 수정

생성/Refine 후 콘솔에 `KeyError: <숫자>` (gradio `state_holder.py` `__contains__`) 트레이스백이
찍히던 문제 수정.

- **원인**: 핸들러가 `gr.update()`를 반환하면 gradio가 **요청 시점에** 해당 컴포넌트를 다시
  만든다(`blocks.py` `postprocess_data`: `state[block._id] = block.__class__(**constructor_args)`,
  `render=False` 강제 주입). 이 `constructor_args`에는 **원본 elem_id가 그대로** 들어있고,
  webui는 `gradio.components.Component.__init__`를 패치해 두었기 때문에
  (`modules/gradio_extensions.py`) 이 일회용 인스턴스에 대해서도 `on_after_component`가
  발화한다. `render()`가 호출되지 않아 이 인스턴스의 `_id`는
  `demo.default_config.blocks`에 등록되지 않는데, `SessionState.blocks_config`는 그 dict의
  얕은 스냅샷이라 이후 해당 컴포넌트를 건드리는 이벤트에서 `KeyError`가 난다.
- **증상 경로**: `_refine_error_return` / `_anima_error_return`이
  `outputs=[gallery, status, html_info, generation_info]`로 배선돼 있어, Refine/Anima의
  early-return마다 `html_info_txt2img`·`generation_info_txt2img` 에코가 발생 → 캐시해 둔
  전역이 미등록 컴포넌트로 덮이고, `refine_panel` 센티넬까지 오염될 수 있었음.
- **수정**: `on_after_component`가 **실제로 등록된 컴포넌트에 대해서만** 동작하도록 가드 추가.
  `Context.root_block`은 ContextVar가 아닌 프로세스 전역이라 Reload UI 중 in-flight 요청
  에코가 새는 레이스가 남으므로, 등록 여부(`component._id in ...default_config.blocks`)를
  직접 확인한다. UI 빌드 중에는 완전한 no-op이라 기능 변화 없음
  (Compact 프롬프트 레이아웃의 `render=False` 컨테이너 자식도 즉시 등록되므로 안전).
- **부수 수정**: `modules/api/api.py`가 임시 `with gr.Blocks():` 안에서 모든 스크립트의
  `ui()`를 재실행하는데, 실제 빌드에서 `build_anima_panel()`이 실패했을 경우 그 패스가 죽은
  패널을 잡아 Tile-Repair가 프로세스 내내 비활성화될 수 있었음 → `anima_build_attempted`
  플래그로 차단.

## v0.9.16 — 경량화 패스 (기능 제거 없음)

전체 코드 감사 후 상시/반복 비용만 안전하게 트림. 모든 기능 유지.

- **매 생성 비용 ↓ (일반 non-Anima 포함)**: `Sam3MaskScript.process`가 SAM3 꺼짐 + XYZ
  없음이면 ~50필드 payload 조립 + `Sam3Args` pydantic 검증을 **건너뜀**(early-return).
- **매 스텝 비용 ↓**: `_post_cfg`가 합칠 게 없으면(예: Adaptive Guidance만 켜짐) `float()`
  왕복/latent 2회 할당 없이 즉시 반환.
- **어텐션 콜당 비용 ↓**: 영구 설치되는 SDPA 래퍼가 원본을 `_STATE.get()` 대신 모듈 전역
  `_ORIG_SDPA`로 참조 + 비활성 fast-path를 bool 체크 1회로 단축.
- **VRAM ↓**: (a) PAG가 스택한 latent 텐서(cond/uncond/attn/slg_raw + APG momentum)를
  `postprocess`에서 해제. (b) VAE 2x 디코더 캐시를 **최근 1개로 상한**(나머지 evict +
  `empty_cache`). (c) VAE 파일 목록 스캔을 memoize(탭별 재스캔 방지).

감사 결론: import-타임 디스크 스캔/불필요 무거운 import 없음, 기능 OFF 시 base Forge 대비
거의 무비용. 위 항목만 실제 개선 여지였음.

## v0.9.15 — Regional Style-Swap (RegionalSampler 워크플로 재현)

rouge-kasshoku의 "Anima Crossover Couple / RegionalSampler" 가이드(스타일 블리딩 해결)를
Forge Neo의 **기존 SAM3 Refine**로 재현하는 레시피 + 프리셋 버튼.

- **docs/REGIONAL_STYLE_SWAP.md** — 코드 없이 지금 바로 쓰는 단계별 레시피. 핵심 매핑:
  `denoise ≈ 1 − base_only_steps/steps`(예 steps33·B8 → 0.76), `overlap_factor ≈ mask blur`,
  region LoRA는 Replacement 프롬프트에만 넣어 **LoRA 격리** 달성, 동일 seed(🎯)+Euler.
- **🎭 Regional Swap preset 버튼**(Refine 패널): 한 번 클릭으로 가이드 기본값 세팅
  (Euler·CFG5·33steps·denoise0.76·mask blur16·inherit OFF·inpaint only masked·fill original).
  기존 위젯 값만 바꾸며 `REFINE_ARG_KEYS`/입력 배열은 건드리지 않음(저위험).
- 한계: SAM3 Refine는 image-레벨 인페인트(가이드 방법 #2)라 진짜 latent-레벨 RegionalSampler
  (#3)보다 seam이 약간 더 생길 수 있음 → mask blur + inpaint-only-masked로 완화.

## v0.9.14 — Anima VAE 2x (spacepxl decoder) [실험]

spacepxl **2x Wan-VAE 파인튜닝**을 디코더로 써서 speckle을 줄이고 skin/hair를 정리하는
독립 스크립트(`scripts/anima_vae_2x.py`). Qwen/Wan VAE가 latent 구조를 공유하므로 Anima
생성에도 적용됩니다(Forge Neo 로더가 `AutoencoderKLWan`/`AutoencoderKLQwenImage`를 같은
경로로 처리함을 확인).

- **동작**: 12채널 디코더(pixel-shuffle 2x)를 `WanVAE(conv_out_channels=12)`로 직접 빌드해
  `forge_objects.vae`의 decode만 대체(순정 로더는 채널을 하드코딩해 12ch를 못 실음).
  decode: latent(1프레임) → 12ch → `pixel_shuffle(2)`(12→3@2x) → (1x 모드면 downsample+
  약한 blur) → 3ch. 오류 시 순정 decode로 폴백.
- **감지**: state_dict `decoder.head.2.weight` shape[0]==12 (safetensors 헤더만 읽음).
- **UI**: Enable · VAE 파일 · 1x refined / 2x · (Advanced) blur sigma · latent renorm 토글.
- **검증됨**: 감지 로직·pixel-shuffle 채널 산술(12=3·2·2). **런타임 확인 필요**: Wan-2.1
  VAE config 정합(로드 diff 로그로 조정), Qwen↔Wan latent 정규화(색 틀어지면 renorm),
  1프레임 축 처리.

## v0.9.13 — Guidance 패널을 SAM3 바로 밑으로

`sorting_priority`를 98/97 → `0`으로 낮춰 Perturbation Guidance · Detail Daemon 아코디언이
SAM3 확장 블록 안에서 **SAM3 바로 밑**에 표시되도록 위치를 옮겼습니다(Forge는 낮은 값이
위쪽). 두 스크립트 모두 여전히 현재 `forge_objects.unet`에서 clone하므로 다른 unet 패치
스크립트와의 합성은 순서와 무관하게 유지됩니다.

## v0.9.12 — Anima Guidance & Speed Suite

SAM3와 **완전히 분리된 독립 기능 모음**을 추가했습니다. `sam3ext`를 import하지 않고 Forge
Neo 코어 파일/기본 동작을 건드리지 않으며, 전 구간 try/except로 어떤 오류에도 일반 생성으로
폴백합니다(켜 둬도 생성이 깨지지 않음). 두 독립 스크립트로 제공됩니다:
`scripts/anima_safe_pag.py`, `scripts/anima_detail_daemon.py`.

| 기능 | 효과 | 추가 forward | 대상 |
|---|---|---|---|
| **PAG / SEG / SLG** | 구조·디테일 강화 (perturbation guidance) | 있음(배치 접기) | Anima DiT |
| **APG** | 높은 CFG 과채도·번짐 억제 | 없음 | 모든 모델 |
| **Detail Daemon** | 질감·잔디테일↑, 배경 뽀샤시↓ | 없음 | 모든 모델 |
| **Adaptive Guidance** | 후반 uncond 생략 → 무손실 속도↑ (~−27%) | 음수(생략) | 모든 모델 |

**Forge Neo 연동 (코어 수정 없음)** — 실제 샘플링이 `sampler_calc_cond_batch_function`을
호출하지 않음을 소스에서 확인하고, 실제 호출되는 훅만 사용:
- `model_function_wrapper` — cond 행을 배치에 접어 perturbation 약한 예측을 *같은 forward*로
  계산(별도 호출 없음). Adaptive Guidance는 반대로 uncond 행을 제거.
- `post_cfg_function` — `c_out` 실측 복원으로 denoised(x0) 공간에서 정확히(eps/v/flow 무관)
  guidance 합성. 표준 CFG·APG·MaHiRo 위에도 안전하게 얹힘.
- 매 생성 `forge_objects.unet.clone()`에만 훅 → Forge 기본 동작·타 생성 무영향.

**설계 원칙** — 모든 자동동작은 토글(scale 자동감쇠, APG→rescale 자동 off, Detail Daemon
CFG couple). 값은 기본 쉽게(메인 슬라이더/프리셋) + 필요 시 깊게(Advanced 아코디언). 조합은
Perturbation(attn 택1·SLG 병용) + 크기보정(APG↔rescale) + Detail Daemon + Adaptive
Guidance가 서로 다른 지점이라 안전하게 병용됩니다.

**검증** — 전 스크립트 py_compile 통과. 수학 독립 검증: PAG `c_out` 복원(~3e-15), APG(eta=1→
표준 CFG 정확 환원·eta=0 직교·norm clamp), 다중항+auto_decay guidance, Detail Daemon 스케줄,
Adaptive Guidance 게이팅/재구성.

> ⚠️ **실험 기능** — 정적·수학 검증만 됐고, 실제 Anima 체크포인트로 end-to-end 확인이 1회
> 필요합니다(리포의 다른 Anima 기능과 동일 상태). 콘솔 `[AnimaSafePAG]` /
> `[AnimaDetailDaemon]` 로그로 훅 부착·동작 확인.

세부 버전 흐름: `v0.9.8` PAG 독립 스크립트 → `v0.9.9` APG → `v0.9.10` Detail Daemon 포크 →
`v0.9.11` SEG+SLG+scale 자동감쇠 토글 → `v0.9.12` Adaptive Guidance + docs 정리.

## v0.9.7 — Anima reference-latent shape logger (PoC)

`process_before_every_sampling → forge_objects.unet` 경로로 Anima UNet에 model-function
wrapper를 안전하게 붙일 수 있는지 확인하는 계측 PoC. 이후 guidance suite의 이식 토대가 됨.

## v0.9.6 — PiD Upscale 복원 모드

Forge Neo 네이티브 NVIDIA PiD(Pixel Diffusion Decoder) 초해상 복원을 Anima 복원 패널의
모드 옵션으로 추가.

## v0.9.5 — WF3 Tile-Repair 정적 blocker 수정 + TE/VAE 스마트 기본값

Anima Tile-Repair의 정적 버그 정리 및 Qwen3 TE / Qwen-Image VAE 자동 기본값.

## v0.9.4 — LoRA Manager: Forge Neo 연동 + 모달 버그 수정

vendored LoRA Manager의 "Send to ComfyUI" → "Add LoRA"(프롬프트 삽입), ComfyUI→Forge Neo
라벨 치환, 사용 팁 X 버튼/메모 placeholder 버그 수정.

## v0.9.3 — LoRA Manager: 후원 UI 제거 + 업데이트 알림 비활성화

기부 UI 숨김(GPL "Appropriate Legal Notices" 아님) + 상류 업데이트 폴링 short-circuit.
LICENSE·저작권·저자 귀속 미변경.

## v0.9.2 — LoRA Manager: fetch 진행 'failed' 잘림 수정

`.loading-status` 줄바꿈 허용으로 긴 LoRA 이름 뒤 카운터 잘림 수정.

## v0.9.1 — LoRA Manager: Manage 탭 빈 화면 수정

탭 pane selector 교정 + 첫 실행 스캔 논블로킹화(진행 표시 폴링).

## v0.9.0 — LoRA Manager 통합

[willmiao/ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager)를 lazy
spawn standalone 서버 + iframe으로 extra-networks strip의 Manage 탭에 임베드.

---

이전 버전(SAM3 검출/인페인트, Refine, ControlNet 통합, Anima Tile-Repair 등)의 상세는
[README.md](README.md)를 참고하세요.
