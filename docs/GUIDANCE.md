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

## 구성

| 파일 | 역할 |
|---|---|
| `scripts/anima_safe_pag.py` | 단일 오케스트레이터, UI, Forge hook, XYZ 축 |
| `sam3ext/guidance/runtime.py` | generation/pass 단위 APG·SMC·RDC·CNS 상태 정리 |
| `sam3ext/guidance/haar.py` | 4D/5D·홀수 크기 공용 Haar DWT/IDWT |
| `sam3ext/guidance/cwm_smc.py` | CWM·SMC CFG base |
| `scripts/anima_skimmed_cfg.py` | Skimmed CFG anti-burn (독립 스크립트·아코디언) |
| `sam3ext/guidance/dcw.py` | post-CFG wavelet correction |
| `sam3ext/guidance/dave.py` | Anima block DC attenuation |
| `sam3ext/guidance/cns.py` | 기존 sampler noise의 wavelet 재색칠 |
| `sam3ext/guidance/modulation.py` | 보조 CLIP-L·공식 어댑터 로드와 block AdaLN 투영 |
| `scripts/anima_detail_daemon.py` | 별도 Detail Daemon 기능 |

## 실제 처리 순서

```text
shared.state.sampling_step / sampling_steps
  → model wrapper: ADG cond-only 또는 PAG/SEG/SLG weak-row 확장
  → Anima block: CLIP modulation AdaLN 가산 → original forward → DAVE
                 → SLG weak-row restore
  → attention: weak row에만 hard PAG 또는 Gaussian-query SEG
  → post-CFG #1: Skimmed CFG (활성 시 항상 명시적으로 맨 앞)
  → post-CFG #2: Guidance Suite
      1. CNS용 live x_t 저장
      2. ADG skip이면 APG/SMC state reset 후 incoming 유지
      3. CFG base 토글(SMC → APG → CWM, 켜진 것만)
      4. PAG/SEG/SLG delta 가산
      5. DCW / RDC
  → ancestral/SDE noise sampler: CNS 재색칠
```

- `sampler_cfg_function` 슬롯은 사용하지 않습니다.
- `model_function_wrapper`와 `post_cfg_function`은 현재 `forge_objects.unet.clone()`에만 붙습니다.
- step 비율은 wrapper 호출 횟수가 아니라 Forge의 공식 sampling step을 읽습니다. low-VRAM 분할,
  regional conditioning, 2차 sampler가 범위 계산을 오염시키지 않습니다.
- CFG base를 바꾸는 모드는 Forge의 incoming 결과에서 `w_eff`를 최소제곱으로 복원하므로
  `edit_strength`가 소실되지 않습니다. custom/nonlinear CFG의 fit 오차가 크면 경고합니다.

## 1. PAG / SEG / SLG

후반 블록의 약한 예측을 만들고 `scale × (cond − weak)`를 incoming CFG 결과에 더합니다.
Anima 엔진 전용이며 ControlNet이 전달된 호출에서는 충돌 방지를 위해 쉬어 갑니다.

| 방식 | 현재 기본 동작 | 상태 |
|---|---|---|
| PAG | 타깃 self-attention weak row를 value-only 경로로 보간 | 실제 Anima E2E 검증 |
| SEG | 타깃 weak query를 실제 T/H/W 중 H/W 축으로 Gaussian blur·보간 | 실제 Anima E2E 검증 |
| SLG | 타깃 block의 weak-row 출력을 block 입력으로 복원 | 실제 Anima E2E 검증 |

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
| Attn Scale (PAG / SEG guidance scale) | 4.0 | `cond − weak` guidance 배율 |
| Perturbation strength | 0.75 | 공식 PAG→value / SEG→blurred query 보간, `1=전체` |
| SEG Gaussian sigma | 100 | `>9999`는 spatially uniform query |
| Legacy strength | 0.75 | Legacy 모드에서만 사용 |
| Attention blocks | `18` | 빈칸도 안전 기본 `18` |
| Attention heads | 빈칸 | 빈칸=전체, `0,2,4-7` 형식으로 일부 head만 선택 |
| SLG enable / scale / blocks | off / 3.0 / `18` | layer-skip weak 예측 |
| Start / End | 0.0 / 0.7 | 공통 적용 구간 |
| Rescale | 0.20 | PAG 보정량만 std 보정 |
| Rescale mode | `full` | `full`=incoming CFG+guidance, `partial`=cond+guidance 기준 |

PAG 자체를 A/B 할 때는 `Rescale=0`, SLG/APG/ADG off로 두어야 원인을 분리할 수 있습니다.

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
| Start / End (%) | 0.0 / 1.0 | 적용 스텝 구간 |
| Flip at (%) | 0.0 | 해당 지점 이전에서 필터를 뒤집습니다(0=사용 안 함) |

- **CFG > 1 전용**입니다. CFG=1이거나 uncond가 없는 경로에서는 자동으로 건너뜁니다.
- upstream은 ComfyUI **pre**-CFG 노드지만 Forge의 `sampler_pre_cfg_function`은 예측 이전의
  conditioning을 받는 다른 계약이라, 같은 수식을 **post-CFG**에서 재구성합니다. Forge가
  넘겨주는 `cond_denoised`/`uncond_denoised`/`input`/`cond_scale`이면 충분합니다.
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
- Forge의 CFG=1 positive-only 경로에서는 uncond가 없을 수 있으므로 **CFG > 1에서 사용**하세요.
- ADG가 uncond를 생략하는 순간 APG momentum과 SMC state를 즉시 비웁니다.

### CWM / SMC

| 필드 | 기본값 | 주의 |
|---|---:|---|
| CWM alpha low | 0.30 | 초반 LL 대역 CFG 변화 |
| CWM alpha high | 0.15 | 후반 HH 대역 CFG 변화 |
| Enable SMC | off | 프리셋 값을 유지한 채 SMC master ON/OFF |
| SMC preset | `Auto` | master가 켜졌을 때 모델군을 감지해 upstream 값을 선택 |
| Custom lambda | 6.0 | `Custom`에서만 사용, UI 범위 0.5–30.0 |
| Custom k | 0.10 | `Custom`에서만 사용, UI 범위 0–5.0 |

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

기본은 off, `lambda low=0.10`, `lambda high=0.02`입니다. 둘 다 0이면 bitwise identity
fast-path입니다. 4D/5D latent와 홀수 H/W를 지원하며 dtype을 보존합니다. Anima flow sigma는
`sigma/(sigma+1)` 최대치가 낮으므로 다른 EDM 예제와 수치 체감이 다를 수 있습니다.

RDC는 같은 Haar 분해 결과의 각 대역에 generation-local EMA를 유지해 여러 step에 걸친
구도·포즈 drift를 되돌립니다. 이 확장에서는 upstream의 `tau=0` 비활성 계약에 더해 명시적
`Enable RDC`를 제공하므로, 슬라이더 값을 유지한 채 A/B할 수 있고 DCW 순간 보정을 끈 채
RDC만 켤 수도 있습니다.

```text
beta = 1 - exp(-abs(sigma_norm_prev - sigma_norm_now) / tau)
ema_new = (1 - beta) * ema_prev + beta * band_now
band_out = band_now - alpha * (band_now - ema_new)
```

| 필드 | UI 시작값 | 주의 |
|---|---:|---|
| RDC tau | 0.15 | 0이면 수학적으로 no-op. 작을수록 짧은 기억, 클수록 초기 구도 고착 가능 |
| RDC alpha LL | 0.03 | 권장 시작 0.02–0.05. 포즈·구조가 굳으면 낮춤 |
| RDC alpha HH | 0.0 | 기본 0 권장. 필요해도 0.01 이하부터, 높으면 텍스처 흐림 |

첫 스텝과 해상도/device가 바뀐 첫 스텝은 EMA 기준만 seed하고 보정하지 않습니다. 새 sampling
pass가 시작될 때 상태를 비워 hires pass나 다음 생성으로 누출하지 않습니다.

## 4. DAVE

Anima block 출력의 token/spatial 평균(DC)을 초반에 약하게 감쇠해 지배적인 구조 성분을 줄입니다.

```text
out = x − strength × mean(x, token/spatial axes)
```

- 기본 off, `strength=0.30`, `tau=0.10`, blocks `8-18`
- `tau=0`은 전 구간, 양수 tau는 초반 비율까지만 적용
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
경로를 보존합니다. 최종 표준편차도 원본 noise와 맞춥니다.

| 필드 | 기본값 |
|---|---:|
| Enable CNS-inspired Wavelet Noise | off |
| Strength | 1.0 |
| Gamma power | 0.5 |
| Gamma scale | 3.0 |

결정론적 sampler가 noise sampler를 호출하지 않으면 자동 inert이며 검증 로그에
`INERT(no ancestral/SDE noise call)`이 표시됩니다. Adaptive Guidance와 병용할 때는
`Skip after >= 0.65`부터 시작하는 편이 안전합니다.

## 7. Adaptive Guidance (속도)

고정 `Skip after` 이후 cond/uncond가 한 batch로 합쳐진 호출에서 uncond 행을 생략합니다.
논문의 cosine-similarity 판정이 아닌 단순 threshold 구현입니다.

- 기본 off, `Skip after=0.5`
- low-VRAM이 cond/uncond를 따로 호출하면 생략할 수 없어 속도 차이가 없습니다.
- 생략 스텝에서는 perturbation도 쉬고 APG/SMC state를 비웁니다.
- `Keep every N`은 생략 구간에서도 N번째 스텝마다 uncond를 유지합니다.
- 특정 속도 향상률은 보장하지 않습니다. 검증 로그의 `SKIPPED-UNCOND`로 실제 생략을 확인하세요.

## 8. Detail Daemon

별도 `Anima Detail Daemon` 아코디언의 sigma schedule 기능입니다. Guidance Suite의 CFG base와는
별도이며, 모든 모델에서 동작합니다. 자세한 필드는 UI 설명을 따르세요.

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
- `[Anima DCW]`, `[Anima RDC]`, `[Anima DAVE]`, `[Anima CNS]`
- `[Anima Mod]`: Enable, Direction Weight, Start/End Block
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
```

확장 목록 아래 `Anima Reference-Latent PoC (debug / 안전)`에서
`Log Guidance verification summary`를 잠시 켜면 다음을 확인할 수 있습니다.

```text
[AnimaSafePAG] patched SelfCrossAttention.torch_attention_op (staticmethod) ✅
[AnimaSafePAG] attention perturb active ✅ hits=... relative_raw_delta=...
[AnimaSafePAG] [VERIFY] verdict: perturb=..., APG=..., Adaptive=...
[AnimaSafePAG] [VERIFY] suite: attention=..., CFG=... (w_eff=..., fit=...),
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
SMC/CWM 비정상 수치 정리, ADG state flush, CNS 결정성·RNG 비소비·표준편차 보존,
Skimmed callback 실제 prepend 순서와 PAG scale 반응, CLIP adapter 수식·Forge Anima
shape 추론·block AdaLN 무변이 주입, pass 종료 tensor 해제와 Notebook 자산
구조를 포함합니다.

## 크레딧

| 기능 | 참고 프로젝트/논문 | 구현 형태 |
|---|---|---|
| PAG | [iljung1106/comfyui-anima-safe-pag](https://github.com/iljung1106/comfyui-anima-safe-pag), [PAG 논문](https://arxiv.org/abs/2403.17377) | Forge 이식 + strength/head/rescale mode |
| SEG | [SusungHong/SEG-SDXL](https://github.com/SusungHong/SEG-SDXL), [SEG 논문](https://arxiv.org/abs/2408.00760) | Anima H/W용 재구현 |
| SLG | Stability AI SD3.5 / Wan 커뮤니티 구현 | Forge block wrapper |
| APG | [MythicalChu/ComfyUI-APG_ImYourCFGNow](https://github.com/MythicalChu/ComfyUI-APG_ImYourCFGNow), [APG 논문](https://arxiv.org/abs/2410.02416) | post-CFG 재구현 |
| DCW/RDC/CWM/SMC | [namemechan/ComfyUI-DCW](https://github.com/namemechan/ComfyUI-DCW) (GPL-3.0) | 공개 수식 기반 Forge 재작성, vendor 아님 |
| Skimmed CFG | [Extraltodeus/Skimmed_CFG](https://github.com/Extraltodeus/Skimmed_CFG) (LICENSE 파일 미공개) | 공개 수식 기반 Forge 재작성, vendor 아님 |
| DAVE | [daheekwon/DAVE](https://github.com/daheekwon/DAVE) (MIT), [ComfyUI-Anima-DAVE](https://github.com/sorryhyun/ComfyUI-Anima-DAVE) (MIT), [논문](https://arxiv.org/abs/2606.06813) | block 수식 재구현 |
| CNS | [namemechan/comfyui-cns_sampler_patch](https://github.com/namemechan/comfyui-cns_sampler_patch) (GPL-3.0), [논문](https://arxiv.org/abs/2605.30332) | CNS-inspired 재작성, vendor 아님 |
| Anima Modulation Guidance | [Anzhc/Anima-Mod-Guidance-ComfyUI-Node](https://github.com/Anzhc/Anima-Mod-Guidance-ComfyUI-Node) (MIT 선언), [quickjkee/modulation-guidance](https://github.com/quickjkee/modulation-guidance) (MIT), [yresearch/cosmos-pooled adapter](https://huggingface.co/yresearch/cosmos-pooled) | 공개 수식·어댑터 형식 기반 Forge block 재작성, vendor 아님 |
| Detail Daemon | [muerrilla/sd-webui-detail-daemon](https://github.com/muerrilla/sd-webui-detail-daemon) | Forge 재구현 |

원본 저장소를 통째로 포함하지 않았으며, Forge 연결과 상태 관리는 이 확장에서 별도로 작성했습니다.
각 기법과 참조 코드의 저작권·라이선스는 원저자/원 저장소에 따릅니다.
