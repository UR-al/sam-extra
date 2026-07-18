# Changelog

버전 태그는 GitHub Releases에도 발행됩니다. 아래는 요약이며, guidance/속도 기능의
상세는 [docs/GUIDANCE.md](docs/GUIDANCE.md)를 참고하세요.

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
