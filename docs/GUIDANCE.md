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

## 구성

| 파일 | 역할 |
|---|---|
| `scripts/anima_safe_pag.py` | 단일 오케스트레이터, UI, Forge hook, XYZ 축 |
| `sam3ext/guidance/runtime.py` | generation/pass 단위 APG·SMC·CNS 상태 정리 |
| `sam3ext/guidance/haar.py` | 4D/5D·홀수 크기 공용 Haar DWT/IDWT |
| `sam3ext/guidance/cwm_smc.py` | CWM·SMC CFG base |
| `sam3ext/guidance/dcw.py` | post-CFG wavelet correction |
| `sam3ext/guidance/dave.py` | Anima block DC attenuation |
| `sam3ext/guidance/cns.py` | 기존 sampler noise의 wavelet 재색칠 |
| `scripts/anima_detail_daemon.py` | 별도 Detail Daemon 기능 |

## 실제 처리 순서

```text
shared.state.sampling_step / sampling_steps
  → model wrapper: ADG cond-only 또는 PAG/SEG/SLG weak-row 확장
  → Anima block: original forward → DAVE → SLG weak-row restore
  → attention: weak row에만 hard PAG 또는 Gaussian-query SEG
  → 단일 post-CFG:
      1. CNS용 live x_t 저장
      2. ADG skip이면 APG/SMC state reset 후 incoming 유지
      3. CFG base 라디오(Preserve/APG/CWM/SMC/SMC+CWM)
      4. PAG/SEG/SLG delta 가산
      5. DCW
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
| PAG | 타깃 self-attention weak row를 hard value-only 경로로 교체 | 실제 Anima E2E 검증 |
| SEG | 타깃 weak query를 실제 T/H/W 중 H/W 축으로 Gaussian blur | 실제 Anima E2E 검증 |
| SLG | 타깃 block의 weak-row 출력을 block 입력으로 복원 | 실제 Anima E2E 검증 |

PAG와 SEG는 라디오에서 하나만 선택합니다. SLG는 둘 중 하나와 병용할 수 있습니다. 과거 결과를
재현할 때만 `Legacy Soft/Approx`를 켜세요. Legacy PAG는 value 경로로 보간하고 Legacy SEG는
uniform-value 근사를 사용합니다.

| 필드 | 기본값 | 설명 |
|---|---:|---|
| Enable Perturbation Guidance | off | 전체 perturbation 토글 |
| Attention method | PAG | PAG / SEG / None |
| PAG / SEG scale | 4.0 | `cond − weak` guidance 배율 |
| SEG Gaussian sigma | 100 | `>9999`는 spatially uniform query |
| Legacy strength | 0.75 | Legacy 모드에서만 사용 |
| Attention blocks | `18` | 빈칸도 안전 기본 `18` |
| SLG enable / scale / blocks | off / 3.0 / `18` | layer-skip weak 예측 |
| Start / End | 0.0 / 0.7 | 공통 적용 구간 |
| Rescale | 0.20 | PAG 보정량만 std 보정 |
| 동시 사용 scale 자동 감쇠 | on | 활성 weak term 수로 각 scale 나눔 |

PAG 자체를 A/B 할 때는 `Rescale=0`, SLG/APG/ADG off로 두어야 원인을 분리할 수 있습니다.

## 2. CFG base 오케스트레이터

`Preserve incoming`이 기본입니다. 다른 모드는 incoming CFG를 의도적으로 교체하므로
MaHiRo/RescaleCFG/custom CFG를 쓰는 경우 먼저 Preserve로 비교하세요.

| 모드 | 동작 |
|---|---|
| Preserve incoming | Forge 및 다른 CFG 확장의 결과를 그대로 유지 |
| APG | guidance를 cond 평행/직교 성분으로 분해해 과포화 성분 억제 |
| CWM | Haar 대역별 CFG 배율 적용 |
| SMC | step 간 guidance error에 unit-L2 switching control 적용 |
| SMC + CWM | SMC로 error를 보정한 뒤 CWM 대역 배율 적용 |

고급 `Experimental stack`은 라디오와 별개로 `SMC → APG → CWM`을 명시적으로 실행합니다.
기본 사용에는 권장하지 않습니다.

### APG

- 빠른 `Enable APG` 체크박스는 CFG base가 Preserve일 때 APG로 바꾸는 호환 토글입니다.
- `eta=1`, `norm=0`, `momentum=0`이면 표준 선형 CFG로 환원됩니다.
- APG는 이 확장에서는 post-CFG denoised 공간 구현입니다. reference 구현과 픽셀 동일하지 않습니다.
- Forge의 CFG=1 positive-only 경로에서는 uncond가 없을 수 있으므로 **CFG > 1에서 사용**하세요.
- ADG가 uncond를 생략하는 순간 APG momentum과 SMC state를 즉시 비웁니다.

### CWM / SMC

| 필드 | 기본값 | 주의 |
|---|---:|---|
| CWM alpha low | 0.30 | 초반 LL 대역 CFG 변화 |
| CWM alpha high | 0.15 | 후반 HH 대역 CFG 변화 |
| SMC lambda | 6.0 | Anima/Cosmos 보수적 시작값, 벤치마크 최적값 아님 |
| SMC k | 0.20 | element-wise sign이 아닌 전체 unit-L2 방향 사용 |

Anima 16채널 latent에서 `alpha high > +0.15`는 한 인물이 여러 인물로 갈라질 수 있습니다.
UI는 동적 경고만 표시하며 값을 강제로 자르지 않습니다.
SMC/CWM 입력의 NaN·양/음의 Inf는 reference 구현처럼 0으로 정리해 비정상 값이 latent 전체로
증폭되지 않게 합니다.

## 3. DCW

CFG·perturbation 뒤 마지막에 live `x_t`와 denoised 예측의 Haar 대역 차이를 보정합니다.

```text
band_out = band_x0 + lambda_band(sigma) × channel_weight × (band_xt − band_x0)
```

기본은 off, `lambda low=0.10`, `lambda high=0.02`입니다. 둘 다 0이면 bitwise identity
fast-path입니다. 4D/5D latent와 홀수 H/W를 지원하며 dtype을 보존합니다. Anima flow sigma는
`sigma/(sigma+1)` 최대치가 낮으므로 다른 EDM 예제와 수치 체감이 다를 수 있습니다.

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

## 5. CNS-inspired Wavelet Noise

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

## 6. Adaptive Guidance (속도)

고정 `Skip after` 이후 cond/uncond가 한 batch로 합쳐진 호출에서 uncond 행을 생략합니다.
논문의 cosine-similarity 판정이 아닌 단순 threshold 구현입니다.

- 기본 off, `Skip after=0.5`
- low-VRAM이 cond/uncond를 따로 호출하면 생략할 수 없어 속도 차이가 없습니다.
- 생략 스텝에서는 perturbation도 쉬고 APG/SMC state를 비웁니다.
- `Keep every N`은 생략 구간에서도 N번째 스텝마다 uncond를 유지합니다.
- 특정 속도 향상률은 보장하지 않습니다. 검증 로그의 `SKIPPED-UNCOND`로 실제 생략을 확인하세요.

## 7. Detail Daemon

별도 `Anima Detail Daemon` 아코디언의 sigma schedule 기능입니다. Guidance Suite의 CFG base와는
별도이며, 모든 모델에서 동작합니다. 자세한 필드는 UI 설명을 따르세요.

## 조합 원칙

- 처음에는 기능 하나씩, 같은 seed로 비교합니다.
- PAG/SEG는 택1. SLG는 병용 가능하지만 scale 자동 감쇠를 유지합니다.
- CFG base는 라디오 하나만 사용합니다. Experimental stack은 별도 고급 실험입니다.
- DCW는 Suite 내부 마지막입니다. 다른 확장의 post-CFG callback과의 전역 순서는 보장할 수 없습니다.
- CNS는 ancestral/SDE에서만 의미가 있습니다.
- TeaCache는 이 Suite에 포함하지 않습니다. ADG `keep_every`의 batch 크기 진동 및 stateful guidance와
  캐시가 충돌할 수 있습니다.

## XYZ Plot

기존 축에 새 Suite 축도 등록됩니다. `Enable=True,False`로 즉시 A/B할 수 있습니다.

- `[Anima Pert]`: Enable, Method, Scale, Legacy, SEG sigma, Blocks, SLG, Start/End, Rescale
- `[Anima APG]`, `[Anima AdaptiveG]`
- `[Anima CFG]`: Base Mode, Experimental Stack
- `[Anima CWM]`, `[Anima SMC]`
- `[Anima DCW]`, `[Anima DAVE]`, `[Anima CNS]`
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
Anima DAVE
Anima CNS Wavelet Noise
```

확장 목록 아래 `Anima Reference-Latent PoC (debug / 안전)`에서
`Log Guidance verification summary`를 잠시 켜면 다음을 확인할 수 있습니다.

```text
[AnimaSafePAG] patched SelfCrossAttention.torch_attention_op (staticmethod) ✅
[AnimaSafePAG] attention perturb active ✅ hits=... relative_raw_delta=...
[AnimaSafePAG] [VERIFY] verdict: perturb=..., APG=..., Adaptive=...
[AnimaSafePAG] [VERIFY] suite: attention=..., CFG=... (w_eff=..., fit=...),
                               DCW=..., DAVE=..., CNS=...
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
Haar 4D/5D·홀수 크기 round-trip, CWM/SMC/DCW/DAVE 중립값, APG 표준 CFG 환원,
SMC/CWM 비정상 수치 정리, ADG state flush, CNS 결정성·RNG 비소비·표준편차 보존,
pass 종료 tensor 해제와 Live Workspaces 자산 구조를 포함합니다.

## 크레딧

| 기능 | 참고 프로젝트/논문 | 구현 형태 |
|---|---|---|
| PAG | [iljung1106/comfyui-anima-safe-pag](https://github.com/iljung1106/comfyui-anima-safe-pag), [PAG 논문](https://arxiv.org/abs/2403.17377) | Forge 이식 + 공식 hard mode |
| SEG | [SusungHong/SEG-SDXL](https://github.com/SusungHong/SEG-SDXL), [SEG 논문](https://arxiv.org/abs/2408.00760) | Anima H/W용 재구현 |
| SLG | Stability AI SD3.5 / Wan 커뮤니티 구현 | Forge block wrapper |
| APG | [MythicalChu/ComfyUI-APG_ImYourCFGNow](https://github.com/MythicalChu/ComfyUI-APG_ImYourCFGNow), [APG 논문](https://arxiv.org/abs/2410.02416) | post-CFG 재구현 |
| DCW/CWM/SMC | [namemechan/ComfyUI-DCW](https://github.com/namemechan/ComfyUI-DCW) (GPL-3.0) | 공개 수식 기반 Forge 재작성, vendor 아님 |
| DAVE | [daheekwon/DAVE](https://github.com/daheekwon/DAVE) (MIT), [ComfyUI-Anima-DAVE](https://github.com/sorryhyun/ComfyUI-Anima-DAVE) (MIT), [논문](https://arxiv.org/abs/2606.06813) | block 수식 재구현 |
| CNS | [namemechan/comfyui-cns_sampler_patch](https://github.com/namemechan/comfyui-cns_sampler_patch) (GPL-3.0), [논문](https://arxiv.org/abs/2605.30332) | CNS-inspired 재작성, vendor 아님 |
| Detail Daemon | [muerrilla/sd-webui-detail-daemon](https://github.com/muerrilla/sd-webui-detail-daemon) | Forge 재구현 |

원본 저장소를 통째로 포함하지 않았으며, Forge 연결과 상태 관리는 이 확장에서 별도로 작성했습니다.
각 기법과 참조 코드의 저작권·라이선스는 원저자/원 저장소에 따릅니다.
