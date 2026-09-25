# Anima Guidance Suite

Anima/Cosmos/Predict2 계열 DiT의 guidance를 Forge Neo 코어 수정 없이 확장하는 독립 기능입니다.
SAM3 처리 모듈은 초기화하지 않고 `sam3ext.guidance`의 경량 수학 모듈만 사용합니다. 모든 기능은
기본 OFF이며, 전부 끄면 들어온 Forge 결과를 그대로 반환합니다.

> [!IMPORTANT]
> 2026-07-23, Forge Neo 2.27 + `anima_baseV10`에서 PAG·SEG·SLG·APG·Adaptive
> Guidance 분리 실행과 CWM+DCW+DAVE+CNS 최소 활성 조합의 실제 checkpoint 실행
> 경로를 확인했습니다.
> 이는 **훅과 수식이 실행된다는 검증**이며, 모든 sampler·attention backend에서 화질이
> 더 좋아진다는 보장은 아닙니다.
>
> 2026-07-24 추가된 Anima Modulation Guidance는 실제 권장 CLIP-L과 공식 170 MB
> 어댑터의 로드·투영 및 Forge block AdaLN 주입을 검증했습니다. 이 릴리즈에서는 실제
> checkpoint 이미지 A/B까지 완료했다는 뜻은 아닙니다.
>
> 2026-09-25~26 원본 동등성 작업(PAG σ 적용 구간, Skimmed CFG, DCW(+a)·RDC, DAVE, CNS, Detail Daemon을
> 원본 ComfyUI 노드의 값·범위·적용 구간과 같게)은 원본 코드와 결과를 비교하는 CPU 단위 테스트로만
> 확인했고, 아직 GPU 이미지로 다시 확인하지 않았습니다. 아래 2026-07-23 검증 로그는 그 전 코드의 것입니다.

## 구성

| 파일 | 역할 |
|---|---|
| `scripts/anima_safe_pag.py` | 단일 오케스트레이터, UI, Forge hook, XYZ 축 |
| `sam3ext/guidance/runtime.py` | generation/pass 단위 APG·SMC·RDC·CNS 상태 정리 |
| `sam3ext/guidance/haar.py` | 4D/5D·홀수 크기 공용 Haar DWT/IDWT |
| `sam3ext/guidance/cwm_smc.py` | CWM·SMC CFG base |
| `scripts/anima_skimmed_cfg.py` | Skimmed CFG anti-burn (독립 스크립트·아코디언) |
| `sam3ext/guidance/skimmed_cfg.py` | Skimmed CFG 수식·σ 게이트(원본 편입, Apache-2.0) |
| `sam3ext/guidance/sigma_window.py` | PAG σ 적용 구간(원본 Safe PAG 노드의 판정 편입) |
| `sam3ext/guidance/dcw.py` | post-CFG wavelet correction |
| `sam3ext/guidance/dave.py` | Anima block DC attenuation |
| `sam3ext/guidance/dave_gate.py` | DAVE 초반 스텝 게이트(원본 노드의 σ 스케줄 판정) |
| `sam3ext/guidance/cns.py` | 기존 sampler noise의 wavelet 재색칠(원본 `color_noise_wavelet` 편입) |
| `sam3ext/guidance/modulation.py` | 보조 CLIP-L·공식 어댑터 로드와 block AdaLN 투영 |
| `scripts/anima_detail_daemon.py` | 별도 Detail Daemon 기능 |
| `sam3ext/guidance/ui_config_migration.py` | 원본 기본값·범위로 바뀐 슬라이더의 `ui-config.json` 1회 이전 |

## 실제 처리 순서

```text
shared.state.sampling_step / sampling_steps (한 스텝 지연 — 아래 참고)
  → model wrapper: ADG cond-only 또는 PAG/SEG/SLG weak-row 확장
  → Anima block: CLIP modulation AdaLN 가산 → original forward → DAVE
                 → SLG weak-row restore
  → attention: weak row에만 hard PAG 또는 Gaussian-query SEG
  → post-CFG #1: Skimmed CFG (활성 시 항상 명시적으로 맨 앞)
  → post-CFG #2: Guidance Suite
      1. CNS x_t 폴백 저장(기본 출처는 sampler step callback)
      2. ADG skip이면 APG momentum만 비우고 3·4 건너뜀(SMC 상태 유지, incoming = cond)
      3. CFG base 토글(SMC → APG → CWM, 켜진 것만)
      4. PAG/SEG/SLG delta 가산
      5. DCW / RDC
  → ancestral/SDE noise sampler: CNS 재색칠
```

- `sampler_cfg_function` 슬롯은 사용하지 않습니다.
- `model_function_wrapper`와 `post_cfg_function`은 현재 `forge_objects.unet.clone()`에만 붙습니다.
- step 비율은 wrapper 호출 횟수가 아니라 Forge의 `shared.state.sampling_step`을 읽습니다. low-VRAM 분할,
  regional conditioning, 2차 sampler가 범위 계산을 오염시키지 않습니다. 다만 Forge는 이 값을 그 스텝의 모델 호출
  **뒤**에 올리므로, 이 값을 쓰는 SEG/SLG·Adaptive Guidance의 Start/End 구간은 **한 스텝 늦게** 판정됩니다
  (20 steps에서 5%만큼). PAG, Skimmed CFG, DAVE, Detail Daemon은 원본 ComfyUI 노드처럼 모델 호출의 σ로 구간이나
  스케줄 위치를 정해 이 지연이 없으므로, 같은 % 값이어도 시작 스텝이 하나 어긋날 수 있습니다.
- Perturbation Guidance(PAG/SEG/SLG)는 cond 행 사본(weak 행)을 배치에 덧붙여 한 forward로 돌리므로, 켜 둔
  구간에서는 샘플링 배치가 cond/uncond에 weak 행만큼 커져 활성 VRAM과 스텝 시간이 늘어납니다(PAG 하나면 대략 1.5배
  행 수, 첫 target 블록 앞은 `sam3_guidance_pag_prefix_dedup`이 줄여 줌).
- `model_function_wrapper`는 하나만 둘 수 있습니다. 다른 확장이 이미 wrapper를 달아 두었으면 Suite가 그 생성 동안
  자기 것으로 덮어쓰고 콘솔에 한 번 알립니다(`another extension already installed a unet model_function_wrapper …`).
  그 확장의 wrapper 기능은 그 생성에서 빠집니다.
- CFG base를 바꾸는 모드의 배율은 원본 DCW(+a) cfg 훅처럼 Forge가 넘기는 `cond_scale`이고, 다른 CFG 함수가 없으면
  Forge의 선형 CFG처럼 `edit_strength`를 곱하므로 `edit_strength`가 소실되지 않습니다. incoming 결과를 최소제곱으로
  맞춘 값은 진단용(`[VERIFY]`의 `w_fit`)이며, custom/nonlinear CFG의 fit 오차가 크면 경고합니다.

## 1. PAG / SEG / SLG

후반 블록의 약한 예측을 만들고 `scale × (cond − weak)`를 incoming CFG 결과에 더합니다.
Anima 엔진 전용이며 ControlNet이 전달된 호출에서는 충돌 방지를 위해 쉬어 갑니다. 이 ControlNet 가드는 원본
ComfyUI 노드에는 없는 이 확장의 안전장치이며, 실제로 막힌 패스의 infotext에 `Anima Perturbation ControlNet guard`가
남습니다.

| 방식 | 현재 기본 동작 | 상태 |
|---|---|---|
| PAG | 타깃 self-attention weak row를 value-only 경로로 보간 | 실제 Anima E2E 검증 |
| SEG | 타깃 weak query를 실제 T/H/W 중 H/W 축으로 Gaussian blur·보간 | 실제 Anima E2E 검증 |
| SLG | 타깃 block의 weak-row 출력을 block 입력으로 복원 | 실행 경로 확인됨(실제 Anima, 3 steps), 화질 A/B 미완 |

PAG와 SEG는 라디오에서 하나만 선택합니다. SLG는 둘 중 하나와 병용할 수 있습니다. 과거 결과를
재현할 때만 `Legacy Soft/Approx`를 켜세요. Legacy PAG는 출력의 value 경로로 보간하고 Legacy
SEG는 uniform-value 근사를 사용합니다. 공식 경로의 `Perturbation strength=1`은 전체
perturbation이며, 기본 `0.75`는 Anima Safe PAG의 부드러운 권장값입니다.

UI의 **Attn Scale**과 XYZ의 `[Anima Pert] Attn Scale`은 같은 값입니다. attention의
`QKᵀ` 점수 자체를 배율하는 값이 아니라, 최종 `scale × (cond − weak)` 보정량의 배율입니다.

| 필드 | 기본값 | 설명 |
|---|---:|---|
| Enable Perturbation Guidance | off | 전체 perturbation 토글 |
| Attention method | PAG | PAG / SEG / None |
| Attn Scale (PAG / SEG guidance scale) | 4.0 | `cond − weak` guidance 배율. 범위 0–100(원본 노드와 같음, XYZ·API도 같은 범위) |
| Perturbation strength | 0.75 | 공식 PAG→value / SEG→blurred query 보간, `1=전체` |
| SEG Gaussian sigma | 100 | `>9999`는 spatially uniform query |
| Legacy strength | 0.75 | Legacy 모드에서만 사용 |
| Attention blocks | `18` | 빈칸도 안전 기본 `18`. `20-18` 같은 역범위는 원본처럼 18–20으로 읽음(SLG·DAVE 블록 칸도 같음) |
| Attention heads | 빈칸 | 빈칸=전체, `0,2,4-7` 형식으로 일부 head만 선택. 역범위는 블록과 같이 뒤집어 읽음 |
| SLG enable / scale / blocks | off / 3.0 / `18` | layer-skip weak 예측 |
| Start / End | 0.0 / 0.7 | 공통 적용 구간. PAG는 σ 기준, SEG/SLG는 스텝 비율(아래) |
| Rescale | 0.20 | PAG 보정량만 std 보정 |
| Rescale mode | `full` | `full`=incoming CFG+guidance, `partial`=cond+guidance 기준 |

PAG의 Start/End는 원본 노드([iljung1106/comfyui-anima-safe-pag@905b0107](https://github.com/iljung1106/comfyui-anima-safe-pag),
MIT)처럼 붙일 때 모델의 `percent_to_sigma`로 σ 창을 한 번 만들고(0–1로 자르고, 뒤집혀 있으면 바꿈), 모델 호출마다
현재 σ가 그 사이(양 끝 포함)인지 봅니다(`sam3ext/guidance/sigma_window.py`). 스텝 수·스케줄러·img2img denoise가
달라도 같은 σ 구간이고 2차 sampler의 중간 평가도 σ로 판정합니다. 예를 들어 Anima(shift 3)의 기본 0.0–0.7은
σ 1.0–0.5625라 simple 스케줄 20·28·30 steps에서 원본과 같이 15·20·22 steps에 걸립니다(예전 스텝 비율 판정은
14·19·21). predictor가 없는 모델에서만 스텝 비율로 돌아갑니다. SEG와 SLG는 원본 PAG 노드에 없는 기능이라 계속
Forge 스텝 비율(한 스텝 늦음)로 잽니다. infotext `Anima Perturbation Guidance`에 PAG σ 창이
`pag_sigma_window=<상한>-<하한>`으로 남습니다.

PAG 자체를 A/B 할 때는 `Rescale=0`, SLG/APG/ADG off로 두어야 원인을 분리할 수 있습니다.

### 속도 설정 (Settings → SAM Extra Guidance)

둘 다 기본 켬이고, 결과가 아주 미세하게 달라질 수 있어 켜고 끈 값을 infotext에 남깁니다.
infotext를 붙여넣으면 같은 설정으로 돌아갑니다(override settings). API에서는
`override_settings`로 한 장씩 바꿀 수 있습니다.

| 설정 키 | infotext | 내용 |
|---|---|---|
| `sam3_guidance_pag_prefix_dedup` | `Anima PAG prefix dedup: True/False` | 첫 target 블록(PAG/SEG 블록과 SLG 블록 중 가장 앞, 기본 18) 이전 블록은 원래 cond/uncond 행만 돌리고 weak 행 자리에는 cond 행 출력을 복사합니다. 그 블록들에서 weak 행은 cond 행과 입력·연산이 같습니다. 블록 인자를 행 단위로 자를 수 없으면(모르는 kwargs, 행 수가 확장 배치도 1도 아닌 텐서, `transformer_options` 안의 모르는 확장-배치 텐서) 그 블록부터 예전 전체 배치 경로로 돌고, 잘린 배치에서 블록이 예외를 내면 그 패스는 끕니다. NegPiP 마스크(`negpip_mask`)는 행과 함께 자릅니다. |
| `sam3_guidance_seg_separable_blur` | `Anima SEG separable blur: True/False` | 공식 SEG query blur를 같은 1D Gaussian 커널의 가로·세로 depthwise conv 두 번으로 계산합니다(픽셀당 곱셈 k² → 2k). SEG 공식 모드이고 `0 < sigma ≤ 9999`일 때만 기록합니다. 끄면 예전 2D conv입니다. |

CPU 확인(실제 Forge Anima Block, 6블록 소형): 중복 제거 켬/끔 차이 fp32 최대 약 8e-7,
bf16·fp16은 비트 동일. separable/2D blur 차이 fp32 최대 약 2.4e-7, bf16 약 7.8e-3(bf16 1ulp
수준), fp16 약 9.8e-4. GPU(cuBLAS)는 배치 크기에 따라 커널이 달라 차이가 다를 수 있습니다.
콘솔 `generation summary: ... prefix_dedup_blocks=N prefix_dedup_fallbacks=M`에서 실제 적용
블록 호출 수와 폴백 수를 볼 수 있습니다.

## 2. CFG base 오케스트레이터

SMC·APG·CWM은 **서로 독립된 토글**입니다. 원하는 조합을 함께 켤 수 있고, 켜진 것들은
항상 `SMC → APG → CWM` 순서로 적용됩니다. 셋 다 끄면 incoming CFG를 그대로 보존하므로
MaHiRo/RescaleCFG/custom CFG를 쓰는 경우 먼저 전부 끈 상태로 비교하세요.

| 토글 | 동작 |
|---|---|
| (모두 off) | Forge 및 다른 CFG 확장의 결과를 그대로 유지 |
| Enable APG | guidance를 cond 평행/직교 성분으로 분해해 과포화 성분 억제 |
| Enable CWM | Haar 대역별 CFG 배율 적용 |
| Enable SMC | step 간 guidance error에 unit-L2 switching control 적용 |

적용 순서상 SMC는 error를 먼저 다듬고, APG는 그 error를 재투영하며, CWM은 마지막에
대역별 배율을 적용합니다. APG가 켜져 있으면 CFG 배율은 APG가 이미 반영하므로 CWM은
배율 1.0으로 그 결과 위에서 동작합니다.

SMC·CWM은 원본 DCW(+a) ([namemechan/ComfyUI-DCW@66aaf9dd](https://github.com/namemechan/ComfyUI-DCW))의 cfg 훅과
같게 동작합니다.

- CFG 배율은 Forge가 넘기는 `cond_scale`입니다. 다른 CFG 함수가 없으면 Forge의 선형 CFG처럼 `edit_strength`를
  곱합니다. incoming 결과를 `uncond + w × (cond − uncond)`로 최소제곱 맞춘 값은 진단용이라 `[VERIFY]`의
  `w_fit`·`fit`와 비선형 CFG 경고에만 쓰입니다.
- Enable CWM은 alpha low·high 중 하나라도 0이 아닐 때만 CFG를 바꿉니다(원본 `cwm_alpha_active`). 원본 기본값 0에서는
  켜도 표준 CFG와 같습니다.
- RescaleCFG·Dynamic Thresholding처럼 다른 확장이 `sampler_cfg_function`을 걸어 두면 원본처럼 SMC·CWM만 비키고
  (콘솔 경고 1회) 그 확장의 CFG 결과를 둡니다. APG는 그래도 결과를 교체하고, PAG/SEG/SLG·DCW/RDC도 그대로
  적용됩니다.

`Legacy CFG base mode` 아코디언의 라디오와 `Experimental stack` 체크박스는 구버전
호환용으로만 남아 있습니다. 저장된 infotext·API 호출·XYZ 그리드가 그대로 동작하도록
위 토글과 **OR**로 합쳐지며, `Experimental stack`은 세 토글을 모두 켜는 것과 같습니다.

### Skimmed CFG (별도 아코디언)

`Anima Detail Daemon` 바로 아래의 독립 스크립트입니다. 높은 CFG에서 과포화·번짐을 만드는
성분만 골라 낮은 CFG 값으로 되돌립니다(anti-burn). 추가 forward가 없습니다.

| 필드 | 기본값 | 주의 |
|---|---|---|
| Skimming CFG | 7.0 | 되돌릴 기준 스케일. `-1`이면 현재 CFG를 그대로 사용 |
| Full skim negative | off | 네거티브를 0까지 skim. `-1`과 조합하면 upstream의 Clean Skim |
| Disable flipping filter | off | 끄면 더 거칠어집니다 |
| Start / End (%) | 0.0 / 1.0 | 적용 구간. 원본처럼 모델의 `percent_to_sigma`로 σ를 구해 `end σ < σ < start σ`(경계 제외)인 스텝만 skim. flow 모델(Anima)은 첫 스텝(σ=1)을 깎지 않고, start > end면 아무 스텝도 깎지 않습니다 |
| Flip at (%) | 0.0 | 그 지점의 σ보다 큰(앞선) 스텝에서 필터를 뒤집습니다(0=사용 안 함) |

- **CFG > 1 전용**입니다. CFG=1이거나 uncond가 없는 경로에서는 자동으로 건너뜁니다.
- upstream([Extraltodeus/Skimmed_CFG](https://github.com/Extraltodeus/Skimmed_CFG), Apache-2.0)의 수식
  함수를 `sam3ext/guidance/skimmed_cfg.py`에 그대로 편입했고, σ 구간·flip 규칙·깎는 순서도 원본 노드와 같습니다.
- upstream은 ComfyUI **pre**-CFG 노드지만 Forge의 `sampler_pre_cfg_function`은 예측 이전의
  conditioning을 받는 다른 계약이라, post-CFG 목록 맨 앞에서 예측을 깎은 뒤 Forge의 CFG 단계를 깎인 예측으로
  다시 계산합니다. 등록된 `sampler_cfg_function`(RescaleCFG, Dynamic Thresholding 등)은 Forge 형식 인자로
  다시 부르고, 없으면 Forge의 `edit_strength`를 반영한 선형 결합을 씁니다.
- **SMC/APG/CWM과 동시에 사용할 수 있습니다.** upstream이 `conds_out`을 제자리에서 고쳐
  이후 전부가 skim된 예측을 보게 하는 것과 동일하게, 이 스크립트도 skim 결과를 Forge의
  예측 tensor에 다시 씁니다. Forge는 post-CFG 함수마다 args dict를 새로 만들지만 예측
  tensor는 같은 객체를 재사용하므로(`backend/sampling/sampling_function.py`), 뒤따르는
  Safe PAG의 CFG base·PAG/SEG/SLG delta·DCW가 모두 skim 위에서 동작합니다.
- 현재 Forge Neo에는 `process_before_every_sampling`의 정렬 버전 뒤에 비정렬 버전이 다시
  정의돼 있어 `sorting_priority`만으로 실행 순서를 보장할 수 없습니다. 따라서 Skimmed
  callback을 확장 내부에서 post-CFG 목록 **맨 앞에 dedupe+prepend**하여 실제 순서를
  `Skimmed → Safe PAG`로 고정합니다. Forge 코어 파일은 수정하지 않습니다.
- 적용 구간(start/end) 밖이거나 CFG=1이라 건너뛴 스텝에서는 예측을 건드리지 않습니다.

### APG

- `Enable APG` 체크박스는 이제 다른 토글과 무관하게 독립적으로 동작합니다.
- `eta=1`, `norm=0`, `momentum=0`이면 표준 선형 CFG로 환원됩니다.
- APG는 이 확장에서는 post-CFG denoised 공간 구현입니다. reference 구현과 픽셀 동일하지 않습니다.
- **CFG > 1 전용**입니다. CFG=1(Forge가 넘기는 `cond_scale`로 판정)이면 APG를 건너뛰고 콘솔에 한 번 경고하며,
  이때는 PAG rescale 자동 끄기도 적용하지 않습니다(APG를 끈 생성과 같은 결과).
- 원본 DCW(+a)는 SMC·CWM을 CFG 1에서도 돌리고, 이 확장도 SMC·CWM이 켜진 CFG 1 패스에는 원본처럼
  `disable_cfg1_optimization`을 겁니다. 하지만 Forge는 CFG가 1이면 negative prompt를 인코딩하지 않아 uncond가
  없으므로 그때는 SMC·CWM도 건너뜁니다(같은 경고 1회, 켜진 것만 이름이 적힘). CFG가 1이 아닌 패스에서 Forge의
  'Ignore Negative Prompt during Early Steps'·NGMS가 negative를 건너뛴 스텝도 그대로 negative 없이 둡니다.
- ADG가 uncond를 생략한 스텝에서는 APG momentum만 비웁니다. SMC의 이전 오차는 원본처럼 샘플링 한 번 동안
  이어지므로 지우지 않습니다.

### CWM / SMC

| 필드 | 기본값 | 주의 |
|---|---:|---|
| CWM alpha low | 0.0 | 초반 LL 대역 CFG 변화. 범위 −1–2, 원본 권장 시작 0.1–0.3 |
| CWM alpha high | 0.0 | 후반 HH 대역 CFG 변화. 범위 −1–2, 원본 권장 시작 0.1–0.2 |
| Enable SMC | off | 프리셋 값을 유지한 채 SMC master ON/OFF |
| SMC preset | `Auto` | master가 켜졌을 때 모델군을 감지해 upstream 값을 선택 |
| Custom lambda | 6.0 | `Custom`에서만 사용, 범위 0.5–30.0(API·XYZ 값도 이 범위로 맞춤) |
| Custom k | 0.10 | `Custom`에서만 사용, 범위 0–5.0. API 위치 인자가 짧아 이 칸(인덱스 27)이 빠져도 0.10 |

SMC 프리셋과 Auto 감지는
[namemechan/ComfyUI-DCW](https://github.com/namemechan/ComfyUI-DCW)의 공개 계약을
그대로 따릅니다.

| 프리셋 | lambda | k |
|---|---:|---:|
| SD1.5 / SD2 | 5.0 | 0.10 |
| SDXL | 5.0 | 0.10 |
| SD3 / SD3.5 | 6.0 | 0.10 |
| Flux | 6.0 | 0.70 |
| Qwen-Image | 6.0 | 0.10 |
| Cosmos / Wan | 6.0 | 0.20 |

Forge의 `Anima` 엔진은 Auto에서 `Cosmos / Wan`으로 판별됩니다. 감지하지 못한 모델은
upstream처럼 `SD1.5 / SD2`로 되돌아갑니다. 현재 master가 off이면 선택한 preset은
유지되지만 계산에는 들어가지 않습니다. 구버전의 `Enable SMC` 체크박스와
`[Anima SMC] Enable` XYZ는 호환용으로 유지되며, preset이 `Off`인 상태에서 이를 켜면
Custom lambda/k가 적용됩니다. 새 XYZ에는 `[Anima SMC] Preset`도 있습니다.

Anima 16채널 latent에서 `alpha high > +0.15`는 한 인물이 여러 인물로 갈라질 수 있습니다.
UI는 동적 경고만 표시하며 값을 강제로 자르지 않습니다.
SMC/CWM 입력의 NaN·양/음의 Inf는 reference 구현처럼 0으로 정리해 비정상 값이 latent 전체로
증폭되지 않게 합니다.

화면 패널은 찾기 쉽도록 `DCW → RDC → CWM → SMC` 순서입니다. 이는 수학적 실행 순서를
바꾸는 설정이 아니며 실제 처리는 원본 정의대로 `SMC → APG → CWM`, 그 뒤 DCW/RDC입니다.

## 3. DCW / RDC

CFG·perturbation 뒤 마지막에 live `x_t`와 denoised 예측의 Haar 대역 차이를 보정합니다.

```text
band_out = band_x0 + lambda_band(sigma) × channel_weight × (band_xt − band_x0)
```

기본은 off, `lambda low=0.05`(범위 −0.5–0.5, step 0.005), `lambda high=0.01`(범위 −0.3–0.3, step 0.001)입니다.
원본 DCW(+a) ([namemechan/ComfyUI-DCW@66aaf9dd](https://github.com/namemechan/ComfyUI-DCW))의 기본값·범위입니다.
둘 다 0이고 RDC도 꺼져 있으면 bitwise identity fast-path입니다. 4D/5D latent와 홀수 H/W를 지원하며 dtype을
보존합니다. Anima flow sigma는 `sigma/(sigma+1)` 최대치가 낮으므로 다른 EDM 예제와 수치 체감이 다를 수 있습니다.
원본 post-CFG 훅처럼 모든 모델 평가에 적용하므로, Adaptive Guidance가 uncond를 건너뛴 스텝(incoming이 cond 예측)에도
DCW/RDC가 걸립니다.

RDC는 같은 Haar 분해 결과의 각 대역에 generation-local EMA를 유지해 여러 step에 걸친
구도·포즈 drift를 되돌립니다. 원본처럼 따로 켜는 스위치가 없습니다. **Enable DCW가 켜져 있고 tau > 0**일 때만
DCW 보정 안에서 돌고, tau 기본값 0은 끔입니다. DCW lambda를 둘 다 0으로 두면 RDC만 쓸 수 있습니다. 예전
`Enable RDC` 체크박스는 화면에서 뺐지만 script argument 58 자리는 남아 있어, API가 `False`를 보내면 RDC를 끄고
`True`나 생략이면 위 규칙을 따릅니다. XYZ `[Anima RDC] Enable`도 `False`일 때만 끕니다.

```text
beta = 1 - exp(-abs(sigma_norm_prev - sigma_norm_now) / tau)
ema_new = (1 - beta) * ema_prev + beta * band_now
band_out = band_now - alpha * (band_now - ema_new)
```

| 필드 | 기본값 | 주의 |
|---|---:|---|
| RDC tau | 0.0 | 0 = RDC 끔(원본 기본). 범위 0–0.5. 0.05–0.10은 빠른 반응(짧은 기억), 0.2–0.3은 느리고 부드러운 보정, 클수록 초기 구도 고착 가능 |
| RDC alpha LL | 0.03 | 범위 0–0.3. 권장 시작 0.02–0.05. 포즈·구조가 굳으면 낮춤 |
| RDC alpha HH | 0.0 | 범위 0–0.1. 기본 0 권장. 필요해도 0.01 이하부터, 높으면 텍스처 흐림 |

첫 스텝과 해상도/device가 바뀐 첫 스텝은 EMA 기준만 seed하고 보정하지 않습니다. 새 sampling
pass가 시작될 때 상태를 비워 hires pass나 다음 생성으로 누출하지 않습니다.

## 4. DAVE

Anima block 출력의 token/spatial 평균(DC)을 초반에 약하게 감쇠해 지배적인 구조 성분을 줄입니다.

```text
out = x − strength × mean(x, token/spatial axes)
```

- 기본 off, `strength=0.30`, `tau=0.10`, blocks `8-18`. 블록 칸을 비우면 원본 마스크(`dave_alpha.npz`)와 같은
  `8-18`입니다. strength가 0.001 이하이면 원본처럼 아무것도 하지 않습니다.
- 켜지는 스텝은 원본 노드
  ([sorryhyun/ComfyUI-Anima-DAVE@83143e8d](https://github.com/sorryhyun/ComfyUI-Anima-DAVE) `nodes.py` 91-106,
  199-208줄, MIT)의 게이트를 그대로 따릅니다(`sam3ext/guidance/dave_gate.py`).
  - 모델 호출의 σ를 샘플러가 실제로 도는 σ 스케줄에서 찾습니다(`isclose`, rtol 1e-4, atol 1e-6, 처음 맞는 칸).
    Forge는 `sampling_sigmas`에 전체 목록을 둡니다. txt2img는 목록 전체를 돌고, img2img와 hires는 Forge와 같은
    `sigmas[steps - t_enc - 1:]`부터 돕니다. 이 시작 칸은 샘플링 직전에 Forge의 `setup_img2img_steps`로 구합니다.
    끝에서 `스텝 수 + 1`칸을 세는 방식은 쓰지 않습니다. `DDIM` 스케줄 타입은 24·28·30·32 steps에서 σ를
    `스텝 수 + 2`개 내놓아서, 끝에서 세면 첫 σ가 빠지고 켜지는 구간이 한 스텝 길어집니다.
  - DAVE는 `postprocess`까지 붙어 있어 ADetailer의 내부 img2img나 img2img-hires-fix처럼 이 스크립트가 준비하지
    않은 샘플링에도 걸립니다. 그런 실행은 `on_cfg_denoiser`로 지금 도는 요청을 알아내, 위 시작 칸 대신 그 실행의
    `스텝 수 + 1`칸을 끝에서 셉니다(원본 노드도 detailer가 넘긴 그 실행의 σ로 판정합니다).
  - `n`을 그 스텝 수로 두고 `k = max(1, min(n, round(tau × n)))`, 스텝 번호가 `k`보다 작을 때 켭니다.
    tau 0.10이면 20·25 steps는 0-1, 28·30 steps는 0-2, 50 steps는 0-4입니다(파이썬 `round`라 25 steps는 2.5→2).
  - 스케줄에 없는 σ(2차 sampler의 중간점, s_churn)는 원본처럼 0번 스텝으로 보고 **항상 켭니다**.
  - `tau=0`이거나 σ 목록이 없으면 모든 스텝에서 켭니다. 단, σ 목록을 내놓지 않는 Forge의 timestep sampler(DDIM,
    PLMS)는 원본에 대응이 없어 Forge 스텝 위치로 같은 `k`를 판정합니다(한 스텝 늦음).
- cond/uncond/PAG weak 행 모두 같은 선형 변환을 받습니다.
- forward hook을 사용하지 않고 기존 block wrapper 안에서 `original → DAVE → SLG restore` 순서를
  보장합니다.

실행 경로는 실제 Anima에서 확인했지만, “다양성 향상” 정도와 안전 block 범위는 고정 시드 여러
seed로 직접 비교해야 합니다.

## 5. Anima Modulation Guidance (보조 CLIP-L)

Anima의 메인 Qwen conditioning을 교체하지 않고, 별도 CLIP-L pooled embedding을 공개
Cosmos/Anima 어댑터로 투영해 선택한 block의 `adaln_lora_B_T_3D`에 더합니다.

```text
projected = MLP(CLIP(prompt))
mod = projected(base) + w × (projected(positive) − projected(negative))
block_adaln[i] = original_adaln + adapter.scales[i] ⊙ mod
```

- 기본 OFF이며 OFF일 때 기존 block 호출은 bitwise 그대로입니다.
- `w=0`도 **base modulation은 남습니다**. 완전한 비교 기준은 토글 OFF입니다.
- 권장 CLIP-L은 `models/text_encoder/Anzhc Noobai11 CLIP L Anime.safetensors`입니다.
  드롭다운은 safetensors 헤더를 읽어 768차원 CLIP-L만 표시하며 CLIP-G/Qwen은 제외합니다.
- 공식 어댑터가 없으면 첫 사용 시
  `models/anima_modulation_guidance/checkpoint_4000.pt`에 170,609,044 bytes를 내려받고
  SHA-256을 검증합니다. Local file 모드도 `torch.load(weights_only=True)`만 사용합니다.
- CLIP과 어댑터 계산은 CPU에서 generation당 한 번 수행합니다. sampler 중에는 최종
  block vector만 활성 model device/dtype으로 옮기므로 별도 CLIP forward를 매 step 반복하지
  않습니다.

| 필드 | 기본값 | 설명 |
|---|---:|---|
| Enable Anima Modulation Guidance | off | Anima 전용 전체 토글 |
| CLIP-L model | 감지된 Anzhc CLIP-L 우선 | `models/text_encoder`의 768차원 CLIP-L |
| Direction weight `w` | 3.0 | positive−negative 방향 배율 |
| Start / End block | 0 / -1 | 포함 범위, `-1`은 마지막 block |
| Base source | Main positive | 현재 프롬프트 또는 Custom |
| Positive direction | `masterpiece, best quality, highres` | 더하려는 CLIP 방향 |
| Negative source | Main negative | 현재 네거티브 또는 Custom |
| Adapter source | Auto-download official | 공식 자동 다운로드 또는 local `.pt` |

방향이 스타일 LoRA를 약화하거나 구도에 지나치게 개입하면 먼저 `w`를 낮추고, 그 다음
Start block을 뒤로 옮기거나 적용 block 범위를 줄이세요. 반드시 같은 seed 여러 장으로
토글 OFF와 비교하세요.

## 6. CNS-inspired Wavelet Noise

Euler a, ancestral, SDE처럼 sampler가 원본 noise sampler를 호출할 때만 동작합니다. 새 난수를
생성하지 않고 **기존 seeded/Brownian 출력**을 live `x_t` Haar 에너지에 맞춰 재색칠하므로 RNG
경로를 보존합니다. 재색칠 계산은 원본 [comfyui-cns_sampler_patch](https://github.com/namemechan/comfyui-cns_sampler_patch)
(`42278b13`) 의 `color_noise_wavelet` 그대로입니다. 색칠한 노이즈는 원본 noise의 표준편차로 맞춘 뒤, Strength가
1보다 작으면 흰 노이즈와 `lerp` 로만 섞고 다시 맞추지 않습니다(원본과 같이 표준편차가 조금 낮아집니다).

`x_t` 는 원본처럼 sampler의 처음 x에서 시작해 스텝마다 callback의 `x`(그 스텝의 시작 상태)로 바뀝니다. Forge에서는
`p.sampler.callback_state` 를 감싸 잡습니다. 감쌀 수 없는 sampler에서만 post-CFG의 `input` 을 대신 쓰는데, 이 값은
dpmpp_2s_ancestral·dpmpp_sde의 중간점이나 인페인트의 섞인 latent일 수 있습니다. 검증 로그의 `x_t=callback|post_cfg`
로 어느 쪽이었는지 보입니다.

색칠은 이 스크립트가 붙은 pass의 sampler가 샘플링하는 동안(`p.sampler.launch_sampling` 안)에만 합니다. 원본이
자기가 감싼 SAMPLER 하나에만 적용되는 것과 같아서, 이 스크립트를 빼고 만든 중첩 실행(ADetailer의 내부 img2img
등)은 흰 노이즈 그대로입니다.

| 필드 | 기본값 | 범위 · step (원본 INPUT_TYPES) |
|---|---:|---|
| Enable CNS-inspired Wavelet Noise | off | |
| Strength | 1.0 | 0~1 · 0.05 |
| Gamma power | 0.5 | 0.1~2 · 0.05 |
| Gamma scale | 2.0 | 0.1~25 · 0.1 (원본 README: Anima + euler_ancestral_cfg_pp 권장 3.0) |

결정론적 sampler가 noise sampler를 호출하지 않으면 자동 inert이며 검증 로그에
`INERT(no ancestral/SDE noise call)`이 표시됩니다. Adaptive Guidance와 병용할 때는
`Skip after >= 0.65`부터 시작하는 편이 안전합니다.

## 7. Adaptive Guidance (속도)

고정 `Skip after` 이후 cond/uncond가 한 batch로 합쳐진 호출에서 uncond 행을 생략합니다.
논문의 cosine-similarity 판정이 아닌 단순 threshold 구현입니다.

- 기본 off, `Skip after=0.5`
- low-VRAM이 cond/uncond를 따로 호출하면 생략할 수 없어 속도 차이가 없습니다.
- 생략 스텝에서는 perturbation과 CFG base(SMC·APG·CWM)도 쉬고 APG momentum만 비웁니다(SMC의 이전 오차는 유지).
  DCW/RDC는 원본 post-CFG 훅처럼 그 스텝에도 적용됩니다.
- `Keep every N`은 생략 구간에서도 N번째 스텝마다 uncond를 유지합니다.
- 특정 속도 향상률은 보장하지 않습니다. 검증 로그의 `SKIPPED-UNCOND`로 실제 생략을 확인하세요.

## 8. Detail Daemon

별도 `Anima Detail Daemon` 아코디언의 sigma schedule 기능입니다. Guidance Suite의 CFG base와는
별도이며, 추가 forward 없이 sampler σ만 바꿔 모든 모델에서 동작합니다. 값·범위·σ 조회는 ComfyUI 노드
[Jonseed/ComfyUI-Detail-Daemon@3394e44](https://github.com/Jonseed/ComfyUI-Detail-Daemon)와 같고, 노드가 다루지 않는
Forge 동작(hires 패스, 지원하지 않는 sampler)은 [muerrilla/sd-webui-detail-daemon](https://github.com/muerrilla/sd-webui-detail-daemon)을
따릅니다. 스케줄 함수는 두 원본(MIT)에서 그대로 가져왔습니다(고지는 `THIRD_PARTY_NOTICES.md`).

```text
sigma' = sigma × max(1e-6, 1 − schedule(sigma) × 0.1 × CFG)
```

- CFG는 hires 패스에서도 늘 `p.cfg_scale`입니다. 양수 amount는 σ를 낮춰 디테일을 늘리고 음수는 매끈하게 합니다.
  0이거나 끄면 아무것도 바꾸지 않습니다. 배율의 아래쪽은 1e-6에서 막고 위쪽 상한은 없습니다(원본과 같음).
- `schedule`은 샘플러가 도는 스텝 수만큼 만든 곡선입니다. 모델 호출마다 그 σ를 샘플러의 σ 목록에서 찾아(가장
  가까운 칸, 칸 사이는 선형 보간, 목록 범위 밖이면 적용 안 함) 값을 읽으므로 Forge 스텝 번호의 한 스텝 지연이 없고,
  2차 sampler의 중간 평가도 노드와 같은 값을 읽습니다.
- σ 목록은 Forge의 `sampling_sigmas`입니다. txt2img는 목록 전체, img2img와 hires는 Forge와 같은
  `sigmas[steps - t_enc - 1:]`부터 셉니다(DAVE와 같은 규칙이라 `DDIM` 스케줄 타입이 σ를 `스텝 수 + 2`개 내놓아도
  맞습니다). 콜백은 `postprocess`까지 켜져 있어 ADetailer 내부 img2img·img2img-hires-fix처럼 이 스크립트가 준비하지
  않은 실행에도 걸리고, 그런 실행은 그 실행의 `스텝 수 + 1`칸을 끝에서 셉니다. σ 목록을 내놓지 않는 DDIM·PLMS
  sampler는 muerrilla처럼 모델 호출 수로 위치를 셉니다.
- **Hires Pass**를 끄면(기본) 기본 패스에만, 켜면 hires 패스에만 적용합니다(muerrilla와 같음). DPM adaptive·HeunPP2
  sampler에서는 생성 전체에서 꺼집니다.
- σ는 muerrilla처럼 제자리에서 바꿔 Forge의 NGMS 판정·soft inpainting도 바뀐 σ를 봅니다.

| 필드 | 기본값 | 범위 · step |
|---|---:|---|
| Enable Detail Daemon | off | |
| Hires Pass | off | off = 기본 패스만, on = hires 패스만 |
| Detail amount | 0.10 | −5–5 · 0.01 (노드 `detail_amount`와 같은 값) |
| Start / End | 0.2 / 0.8 | 0–1 · 0.01 |
| Bias | 0.5 | 0–1 · 0.01 |
| Exponent | 1.0 | 0–10 · 0.05 |
| Start / End offset | 0.0 / 0.0 | −1–1 · 0.01 |
| Fade | 0.0 | 0–1 · 0.05 |
| Smooth | on | 코사인 스무딩 |

원본에 없는 프리셋·Multiplier·CFG 결합 토글은 없습니다. 예전 API 호출이 어긋나지 않도록 위치 인자 1(preset)·
10(multiplier)·12(cfg_couple) 자리는 숨긴 채 남겨 두고 읽지 않으며, Hires Pass는 인자 13입니다. XYZ는
`[Detail Daemon]` Enable·Amount·Start·End·Bias, infotext는 `Anima Detail Daemon`(amount·range·bias·exponent·offset·
fade·smooth·hires)입니다.

## 조합 원칙

- 처음에는 기능 하나씩, 같은 seed로 비교합니다.
- PAG/SEG는 택1이며 SLG는 병용 가능합니다. 자동 scale 감쇠는 제거됐으므로 각 scale을
  직접 조절합니다.
- CFG base 토글은 하나씩 켜서 효과를 익힌 뒤 조합하세요. 셋을 한 번에 켜면 서로 영향을 줍니다.
- DCW는 Suite 내부 마지막입니다. 다른 확장의 post-CFG callback과의 전역 순서는 보장할 수 없습니다.
- Modulation Guidance는 cond/uncond/PAG weak 행에 같은 block modulation을 적용합니다.
- CNS는 ancestral/SDE에서만 의미가 있습니다.
- TeaCache는 이 Suite에 포함하지 않습니다. ADG `keep_every`의 batch 크기 진동 및 stateful guidance와
  캐시가 충돌할 수 있습니다.

## XYZ Plot

기존 축에 새 Suite 축도 등록됩니다. `Enable=True,False`로 즉시 A/B할 수 있습니다.

- `[Anima Pert]`: Enable, Method, Attn Scale, 공식/Legacy strength, SEG sigma,
  block/head, SLG, Start/End, Rescale/Mode
- `[Anima APG]`, `[Anima AdaptiveG]`
- `[Anima CFG]`: Base Mode, Experimental Stack
- `[Anima CWM]`, `[Anima SMC]`
- `[Anima DCW]`, `[Anima RDC]`, `[Anima DAVE]`, `[Anima CNS]` — `[Anima RDC] Enable`은 예전 RDC 스위치 자리라
  `False`일 때만 RDC를 끄고, `True`면 `Enable DCW + tau > 0` 규칙을 따릅니다
- `[Anima Mod]`: Enable, Direction Weight, Start/End Block
- `[Anima Skim]`(Skimmed CFG, 7축): Enable, Skimming CFG, Full Skim Negative, Disable Flipping Filter, Start, End,
  Flip At
- `[Detail Daemon]`

WebUI의 Reload scripts 뒤에도 기존 label은 중복하지 않고 새 label만 추가합니다.

## 메타데이터와 검증 로그

활성 기능은 PNG infotext/API `info`에 다음 키를 기록합니다.

```text
Anima Perturbation Guidance
Anima APG
Anima Adaptive Guidance
Anima CFG Orchestrator
Anima DCW
Anima RDC
Anima DAVE
Anima CNS Wavelet Noise
Anima Modulation Guidance
Anima PAG prefix dedup      (PAG/SEG/SLG 켤 때)
Anima SEG separable blur    (공식 SEG blur 를 쓸 때)
Anima Perturbation ControlNet guard  (ControlNet 가드로 PAG/SEG/SLG 가 막힌 패스)
Anima Skimmed CFG           (별도 스크립트)
Anima Detail Daemon         (별도 스크립트)
```

`Anima DCW`는 DCW가 실제로 돌 때(lambda가 0이 아니거나 RDC가 켜짐), `Anima RDC`는 RDC가 켜졌을 때(Enable DCW +
tau > 0) 남습니다. `Anima Perturbation Guidance`에는 PAG σ 창(`pag_sigma_window=`)이 함께 적힙니다.

확장 목록 아래 `Anima Reference-Latent PoC (debug / 안전)`에서
`Log Guidance verification summary`를 잠시 켜면 다음을 확인할 수 있습니다.

```text
[AnimaSafePAG] patched SelfCrossAttention.torch_attention_op (staticmethod) ✅
[AnimaSafePAG] attention perturb active ✅ hits=... relative_raw_delta=...
[AnimaSafePAG] [VERIFY] verdict: perturb=..., APG=..., Adaptive=...
[AnimaSafePAG] [VERIFY] suite: attention=..., CFG=... (w_eff=..., w_fit=..., fit=...),
                               DCW=..., RDC=..., DAVE=..., CNS=..., Modulation=...
```

2026-07-23 실제 최소 검증(256×256, 3 steps):

- 공식 PAG block 18: 3/3 steps, 첫 weak `relative_raw_delta=2.326e-01`
- 공식 SEG sigma 1.0 block 18: 3/3 steps, 첫 weak `relative_raw_delta=2.879e-02`
- SLG block 18: 3/3 steps, 첫 `mean|cond-weak|=9.751e-02`
- APG: 3/3 evals, `w_eff=4`, `fit_error=0`
- Adaptive Guidance `skip_after=0`: combined batch의 uncond 3/3 steps 생략
- CWM+DCW+DAVE+CNS, Euler a: CFG 3 evals, DCW 3 evals, DAVE 3 block hits,
  CNS 2 noise calls, `w_eff=4`, `fit_error=0`

## 테스트

```bash
python -m unittest discover -s tests -v
```

검증 범위는 attention staticmethod binding/weak-row 한정 변경, official SEG 실제 H/W,
Haar 4D/5D·홀수 크기 round-trip, CWM/SMC/DCW/DAVE 중립값, RDC tau=0 identity와
step/해상도 state reset, APG 표준 CFG 환원,
SMC/CWM 비정상 수치 정리, ADG state flush, CNS 결정성·RNG 비소비,
Skimmed callback 실제 prepend 순서와 PAG scale 반응, CLIP adapter 수식·Forge Anima
shape 추론·block AdaLN 무변이 주입, pass 종료 tensor 해제와 Notebook 자산
구조를 포함합니다.

원본 대조 테스트(`tests/test_anima_safe_pag_origin.py`, `test_skimmed_cfg.py`, `test_dcw_origin.py`,
`test_dave_origin.py`, `test_cns_origin.py`, `test_detail_daemon_origin.py`)는 각 원본 노드의 해당 코드를 고정 커밋
그대로 테스트 안에 두고(DCW·CNS는 `tests/_origin_*.py`) 같은 입력에서 이 확장과 결과를 비교합니다. 모두 CPU
테스트이며 GPU 이미지 비교는 아닙니다. `test_ui_config_migration.py`는 아래 `ui-config.json` 이전 규칙을 검사합니다.

## 저장된 UI 값 1회 이전 (`ui-config.json`)

Forge는 UI를 만들 때 슬라이더의 값·최소·최대·step을 라벨 이름을 키로 `ui-config.json`에 저장해 두고 다음 시작에
다시 적용합니다. 원본 동등성 작업에서 기본값·범위만 바뀌고 라벨은 그대로인 슬라이더는 예전 값이 되살아나므로,
`on_before_ui`에서 UI를 만들기 전에 이 파일을 한 번 고칩니다(`sam3ext/guidance/ui_config_migration.py`, Guidance
아코디언과 Skimmed CFG의 txt2img·img2img 슬라이더, txt2img 탭 Tile-Repair 패널의 네거티브).

| 대상 | 바꾸는 것 |
|---|---|
| PAG Attn Scale | 저장된 최대가 예전 15이면 저장된 범위만 지움(값은 유지, 기본 4.0은 그대로) |
| RDC | 예전 `Enable RDC` 저장값이 있을 때만: 끈 채로 저장된 tau가 예전 기본 0.15면 0으로. 사용자가 바꾼 tau는 유지(이제 Enable DCW를 켜면 RDC도 돔 — 끄려면 tau 0). 예전 스위치 키는 지움 |
| DCW lambda | `lambda high`에 예전 범위(±0.5, step 0.005)가 저장돼 있을 때만: 그 범위를 지우고, 예전 기본값 그대로인 low 0.10 → 0.05, high 0.02 → 0.01. ±0.3 밖의 high는 ±0.3으로 |
| CWM alpha | 저장된 최대가 예전 1.0인 슬라이더만: 범위를 지우고, 예전 기본값 그대로인 low 0.30·high 0.15 → 0 |
| CNS | strength의 예전 step 0.01, gamma power의 예전 최소 0.05가 저장돼 있으면 범위를 지움(0.1 아래 gamma power는 0.1로). 라벨이 바뀐 gamma scale은 예전 라벨 키를 지우고, 예전 기본 3.0이 아닌 값만 새 라벨로 옮김(0.1–25로 맞춤) |
| Skimmed CFG Flip at | 저장된 step이 예전 0.05이면 저장된 범위를 지움(값은 유지, 새 step 0.01) |
| Tile-Repair 네거티브 | 예전 `SAM3 Anima Width`·`Height` 슬라이더 키가 남아 있을 때만: 그 키를 지우고, 네거티브가 예전 기본값 `blurry, low quality` 그대로면 sd-scripts 기본값 빈 칸으로. 직접 적은 네거티브는 유지 |

바꿀 것이 있으면 원본을 같은 폴더의 `ui-config.json.bak-anima-guidance-<날짜-시각>`으로 먼저 복사하고, 새 내용은
임시 파일(`ui-config.json.tmp-anima-guidance`)에 쓴 뒤 한 번에 바꿔 넣습니다. 바꾼 항목은 콘솔
`[AnimaSafePAG] ui-config.json migrated to the upstream PAG/DCW(+a)/CNS/Skimmed CFG/Tile-Repair defaults/ranges:`
아래에 줄마다 나옵니다.
모든 규칙은 이전이 스스로 지우는 예전 표지(저장된 예전 범위·예전 스위치 키)에 걸려 있어, 그 뒤에 사용자가 저장한 값은
다시 건드리지 않습니다. 파일이 없거나 읽을 수 없으면 그대로 둡니다.

Detail Daemon 슬라이더는 이 이전의 대상이 아닙니다. Amount는 라벨이 바뀌어 v0.21.2의 저장값이 적용되지 않습니다.

## 크레딧

| 기능 | 참고 프로젝트/논문 | 구현 형태 |
|---|---|---|
| PAG | [iljung1106/comfyui-anima-safe-pag](https://github.com/iljung1106/comfyui-anima-safe-pag) (MIT), [PAG 논문](https://arxiv.org/abs/2403.17377) | Forge 이식 + strength/head/rescale mode, σ 적용 구간·번호 파싱 편입(`sam3ext/guidance/sigma_window.py`, 고지는 THIRD_PARTY_NOTICES.md) |
| SEG | [SusungHong/SEG-SDXL](https://github.com/SusungHong/SEG-SDXL), [SEG 논문](https://arxiv.org/abs/2408.00760) | Anima H/W용 재구현 |
| SLG | Stability AI SD3.5 / Wan 커뮤니티 구현 | Forge block wrapper |
| APG | [MythicalChu/ComfyUI-APG_ImYourCFGNow](https://github.com/MythicalChu/ComfyUI-APG_ImYourCFGNow), [APG 논문](https://arxiv.org/abs/2410.02416) | post-CFG 재구현 |
| DCW/RDC/CWM/SMC | [namemechan/ComfyUI-DCW](https://github.com/namemechan/ComfyUI-DCW) (GPL-3.0) | 공개 수식 기반 Forge 재작성, vendor 아님 |
| Skimmed CFG | [Extraltodeus/Skimmed_CFG](https://github.com/Extraltodeus/Skimmed_CFG) (Apache-2.0) | 수식·σ 게이트 편입(`sam3ext/guidance/skimmed_cfg.py`), Forge post-CFG 훅 |
| DAVE | [daheekwon/DAVE](https://github.com/daheekwon/DAVE) (MIT), [ComfyUI-Anima-DAVE](https://github.com/sorryhyun/ComfyUI-Anima-DAVE) (MIT), [논문](https://arxiv.org/abs/2606.06813) | 초반 스텝 게이트 편입(`sam3ext/guidance/dave_gate.py`, 고지는 THIRD_PARTY_NOTICES.md), block 수식 재구현 |
| CNS | [namemechan/comfyui-cns_sampler_patch](https://github.com/namemechan/comfyui-cns_sampler_patch) (GPL-3.0), [논문](https://arxiv.org/abs/2605.30332) | `color_noise_wavelet` 편입(`sam3ext/guidance/cns.py`, 고지는 THIRD_PARTY_NOTICES.md), Forge noise 원천·callback 훅 |
| Anima Modulation Guidance | [Anzhc/Anima-Mod-Guidance-ComfyUI-Node](https://github.com/Anzhc/Anima-Mod-Guidance-ComfyUI-Node) (MIT 선언), [quickjkee/modulation-guidance](https://github.com/quickjkee/modulation-guidance) (MIT), [yresearch/cosmos-pooled adapter](https://huggingface.co/yresearch/cosmos-pooled) | 공개 수식·어댑터 형식 기반 Forge block 재작성, vendor 아님 |
| Detail Daemon | [muerrilla/sd-webui-detail-daemon](https://github.com/muerrilla/sd-webui-detail-daemon) (MIT), [Jonseed/ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon) (MIT) | schedule·σ 조회 함수 편입(고지는 THIRD_PARTY_NOTICES.md), Forge 훅 |

원본 저장소를 통째로 포함하지 않았으며, Forge 연결과 상태 관리는 이 확장에서 별도로 작성했습니다.
각 기법과 참조 코드의 저작권·라이선스는 원저자/원 저장소에 따릅니다.
