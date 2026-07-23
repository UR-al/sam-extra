# Changelog

버전 태그는 GitHub Releases에도 발행됩니다. 아래는 요약이며, guidance/속도 기능의
상세는 [docs/GUIDANCE.md](docs/GUIDANCE.md)를 참고하세요.

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
