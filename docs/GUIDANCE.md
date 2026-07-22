# Anima Guidance & Speed Suite

Anima(및 Cosmos/Predict2 계열 DiT) 생성 품질·속도를 높이는 **독립 기능 모음**입니다.
SAM3 워크플로와 완전히 분리돼 있고(`sam3ext`를 전혀 import하지 않음), **Forge Neo 코어
파일을 건드리지 않으며**, 전 구간 try/except로 어떤 오류에도 일반 생성으로 폴백합니다 —
켜 둬도 생성이 깨지지 않습니다.

> ⚠️ **실험 기능.** PAG는 Forge Neo 2.27 + `anima_baseV10`에서 True/False
> end-to-end 생성을 검증했습니다. 다른 파생 모델/정밀 조합은 webui 콘솔의
> `[AnimaSafePAG]` / `[AnimaDetailDaemon]` 로그로 훅 부착·동작을 확인하세요.

## 구성 파일

| 스크립트 | 담는 기능 |
|---|---|
| `scripts/anima_safe_pag.py` | Perturbation Guidance(PAG/SEG/SLG) · APG · Adaptive Guidance |
| `scripts/anima_detail_daemon.py` | Detail Daemon |

설치는 확장에 포함돼 별도 작업이 없습니다. t2i/img2img 패널에 **"Anima Perturbation
Guidance (PAG / SEG / SLG)"** 와 **"Anima Detail Daemon"** 아코디언이 보이면 정상입니다.

---

## 한눈 요약

| 기능 | 효과(느낌) | 추가 forward | 대상 모델 |
|---|---|---|---|
| **PAG** | 선·구조 또렷, 디테일↑ | 있음(배치 접기) | Anima DiT |
| **SEG** | PAG보다 부드럽게 구조 잡기 | 있음(배치 접기) | Anima DiT |
| **SLG** | 해부/구도 붕괴 방지 | 있음(배치 접기) | Anima DiT |
| **APG** | 높은 CFG의 과채도·번짐 억제 | **없음** | 모든 모델 |
| **Detail Daemon** | 질감·잔디테일↑, 배경 뽀샤시↓ | **없음** | 모든 모델 |
| **Adaptive Guidance** | 무손실에 가까운 **속도↑** | **음수(생략)** | 모든 모델 |

**설계 원칙**: 모든 자동동작은 **토글**로 끌 수 있고, 값은 기본은 쉽게(메인 슬라이더) +
필요 시 깊게(**Advanced 아코디언**) 조절합니다.

---

## Forge Neo 연동 원리 (코어 수정 없음)

Forge Neo 실제 샘플링 경로(`backend/sampling/sampling_function.py`)는 ComfyUI 노드가 쓰는
`sampler_calc_cond_batch_function`을 **호출하지 않으므로**(패처에 setter만 존재), 실제로
호출되는 훅만 사용합니다.

1. **`model_function_wrapper`** — `apply_model`이 보는 cond/uncond 배치를 감쌉니다.
   Perturbation은 여기서 **cond 행을 복제해 배치에 붙여** 약한 예측을 *같은 forward*로
   계산하고(별도 호출 없음), Adaptive Guidance는 반대로 **uncond 행을 제거**합니다.
2. **`post_cfg_function`** — Forge의 `model.apply_model` 결과는 이미 predictor가 변환한
   denoised(x0) 예측입니다. 따라서 `cond_x0 − weak_x0`를 직접 guidance로 더하며,
   eps/v/flow-matching 종류와 무관하게 표준 CFG·APG·MaHiRo 등의 결과 위에 안전하게
   얹힙니다. Rescale은 **CFG 전체가 아닌 PAG 보정량만** 조정합니다.

어텐션 perturbation은 `backend/nn/anima.py`의 모듈 전역 `scaled_dot_product_attention`을
감싸 **약한 예측 행에만** 적용합니다(value는 RoPE가 없어 rotary-safe, grouped-query로 head
수가 다르면 훼손 없이 자동 스킵). SLG는 `TransformerBlock.forward`를 감싸 지정 블록을 해당
행에서만 통째로 스킵합니다.

기능은 매 생성마다 `p.sd_model.forge_objects.unet.clone()`에만 훅을 답니다 → Forge 기본
동작·다른 생성에 영향 없음, 끄면 완전 no-op.

---

## 1. Perturbation Guidance (PAG / SEG / SLG)

후반 블록에 *약한 예측*을 만들어 CFG를 그 반대로 밀어 구조·디테일을 강화합니다.
**Anima 엔진에서만** 동작합니다. 여러 방식을 켜도 약한 예측을 **같은 배치에 접어** 단일
forward로 계산합니다(활성 수만큼 배치 행만 증가).

| 방식 | 느낌 | perturbation |
|---|---|---|
| **PAG** | 선/구조 또렷 | 어텐션 → value-only(identity). `lerp(정상, value, strength)` |
| **SEG** | 더 부드럽게 구조 잡기 | 어텐션 → uniform(seq 평균). `lerp(정상, mean(value), strength)` |
| **SLG** | 해부/구도 붕괴 방지 | 지정 블록을 통째로 스킵(출력=입력)한 약한 예측 |

- **PAG ↔ SEG** 는 성격이 겹쳐 **택1**(Attention perturbation 라디오).
- **SLG** 는 PAG/SEG와 **병용 가능**.
- SEG는 원논문의 가우시안 블러 대신 "uniform(∞-blur 극한=값 평균)으로 보간"하는
  shape-agnostic 근사입니다(H·W 재구성 불필요, 안전). SLG는 SD3.5/Wan에서 정평난 방식.

| 필드 | 기본값 | 설명 |
|---|---|---|
| Enable Perturbation Guidance | off | Anima 생성에만 적용(다른 모델 자동 스킵) |
| Attention perturbation | PAG | `PAG` / `SEG` / `None`(=SLG만) |
| Attention guidance scale | 4.0 | PAG/SEG guidance 세기 |
| Perturbation strength | 0.75 | PAG: →value / SEG: →uniform 블렌드 비율 |
| Attention block indices | `18` | 빈칸도 안전 기본 `18`로 처리. 필요 시 `18-20` 형식 |
| Enable SLG | off | 레이어 스킵 약한 예측 병용 |
| SLG guidance scale | 3.0 | SLG guidance 세기 |
| SLG skip block indices | `18` | 빈칸도 `18`로 처리. 스킵할 블록 |
| Start ~ End percent | 0.0 ~ 0.7 | 적용 샘플링 구간(나머지 스텝은 원가) |
| Rescale | 0.20 | 대비/채도 과다 억제 (APG 켜지면 자동 off) |
| **동시 사용 시 scale 자동 감쇠** *(토글)* | on | 활성 수로 각 scale ÷ (과대 guidance 방지). 끄면 원 scale |

---

## 2. APG (Adaptive Projected Guidance)

높은 CFG가 만드는 **과채도·번짐 성분(cond 평행)만 골라 억제**해, guidance를 세게 밀어도
자연스럽게 유지합니다(RescaleCFG의 상위호환). **추가 forward 없이** post-CFG에서 계산만
바꾸며 **모든 모델에서 동작**합니다. `eta=1·norm=0·momentum=0`이면 **표준 CFG로 정확히
환원**되어 안전합니다.

| 필드 | 기본값 | 설명 |
|---|---|---|
| Enable APG | off | 켜면 CFG 합성을 APG로 대체 |
| APG 켜지면 PAG rescale 자동 끄기 *(토글)* | on | 이중 크기보정 방지. 끄면 둘 다 적용 |
| eta *(Advanced)* | 0.0 | 평행(과채도) 성분 비중. 1.0=표준 CFG 환원, 0=최대 억제 |
| norm threshold *(Advanced)* | 15.0 | guidance L2 크기 상한(0=off) |
| momentum *(Advanced)* | 0.0 | 스텝 간 running-average(음수 권장, 0=off) |

guidance 세기는 메인 **CFG Scale** 슬라이더를 그대로 씁니다.

---

## 3. Adaptive Guidance (속도)

샘플링 **후반부에는 uncond(네거티브) 예측 기여가 미미**하므로, 지정 지점 이후 **uncond
forward를 생략**해 그 스텝 배치를 절반으로 줄입니다. 추가 계산이 아니라 **빼는** 쪽이라
무손실에 가까운 속도↑(예: 20스텝·skip 0.5 → forward 시간 약 **−27%**)이며 모든 모델에서
동작합니다. 생략 스텝에선 uncond=cond로 두어 CFG가 cond로 붕괴하고, perturbation도 함께
쉽니다(그 지점은 이득이 거의 없음).

| 필드 | 기본값 | 설명 |
|---|---|---|
| Enable Adaptive Guidance | off | 켜면 후반 uncond 생략 |
| Skip after | 0.5 | 이 지점(스텝 비율) 이후 생략 시작 |
| Keep every N *(Advanced)* | 0 | 생략 구간에서도 N스텝마다 uncond 유지(0=항상 생략, 보수적 품질용) |

---

## 4. Detail Daemon

매 스텝 **제거하는 노이즈량을 줄여** 디테일·질감을 늘립니다(배경 뽀샤시↓). 추가 forward
없이 sampler sigma만 조정하므로 **모든 모델(Anima RF 포함)에서 동작**합니다. muerrilla의
원본을 재구현해 직관적 UX로 재설계했습니다.

`sigma *= 1 − schedule[step] · amount · (cfg_scale if couple else 1)` — 양수 amount는 sigma를
낮춰(=노이즈 덜 제거) 디테일↑, 음수는 매끈, 0/끄면 완전 무효.

| 필드 | 기본값 | 설명 |
|---|---|---|
| Enable Detail Daemon | off | |
| Preset | Medium | Custom이면 아래 Amount 사용 (Subtle 0.05 / Medium 0.10 / Strong 0.25) |
| Detail amount | 0.10 | 음수=매끈, 양수=디테일↑ |
| start / end / bias *(Adv)* | 0.2 / 0.8 / 0.5 | 적용 구간과 피크 위치 |
| exponent / offsets / fade *(Adv)* | 1.0 / 0 / 0 | 곡률·구간밖 기본값·전체 감쇠 |
| smooth / multiplier *(Adv)* | on / 1.0 | 코사인 스무딩 · 스케줄 위 전역 강도 |
| couple to CFG scale *(Adv, 토글)* | on | 효과를 CFG scale에 비례(원본 both). 끄면 CFG 무관 |

---

## 조합 규칙 (중요)

기능을 **카테고리**로 나누면 충돌 여부가 명확합니다. **각 카테고리에서 하나씩** 고르면
서로 다른 지점이라 안전하게 겹칩니다.

| 카테고리 | 후보 | 규칙 |
|---|---|---|
| Perturbation | PAG / SEG / SLG | PAG↔SEG 택1, SLG 병용 가능 |
| 크기보정 | APG / Rescale | 하나만. **APG 켜면 rescale 자동 off**(토글로 해제) |
| 스케줄 | Detail Daemon | 독립 — 무엇과도 겹침 |
| 속도 | Adaptive Guidance | 독립 — 무엇과도 겹침 |
| (내장) 프롬프트 재믹스 | MaHiRo | post-CFG 위에 우리 guidance가 얹힘 → 병용 OK |

**권장 스택 예시**: PAG(구조) + APG(과채도 억제) + Detail Daemon(질감) + Adaptive
Guidance(속도).

⚠️ **피할 조합**: PAG+SEG 동시(중복·속도 2배 손해), APG+Rescale 이중 크기보정(밋밋).
여러 perturbation을 굳이 함께 쓸 땐 `scale 자동 감쇠` 토글을 켜 두세요.

> 참고: RescaleCFG/Epsilon Scaling은 v-pred/eps 모델용이라 RF인 Anima에선 효용이 적습니다.
> 그래서 Anima에선 rescale을 낮게 두거나 APG로 대체하는 것을 권장합니다.

---

## XYZ Plot 비교

축군 `[Anima Pert] …` · `[Anima APG] …` · `[Anima AdaptiveG] …` · `[Detail Daemon] …` 가
추가됩니다. **`… Enable`** 축을 `True, False`로 두면 **ON/OFF 비교 그리드**를 바로 뽑습니다
(UI 체크박스와 무관하게 축 값이 우선).

- `[Anima Pert]` — Enable · Attn Method · Attn Scale · Perturbation Strength · Attn Block
  Indices · SLG Enable · SLG Scale · SLG Block Indices · Start/End Percent · Rescale
- `[Anima APG]` — Enable · Eta · Norm Threshold · Momentum
- `[Anima AdaptiveG]` — Enable · Skip After · Keep Every
- `[Detail Daemon]` — Enable · Amount · Start · End · Bias

---

## 메타데이터 / 로그

각 기능은 결과 PNG의 `parameters`에 마커를 남깁니다: `Anima Perturbation Guidance: …`,
`Anima APG: …`, `Anima Adaptive Guidance: …`, `Anima Detail Daemon: …`.

콘솔 로그로 실제 부착·동작을 확인할 수 있습니다:

```
[AnimaSafePAG] patched backend.nn.anima.scaled_dot_product_attention ✅
[AnimaSafePAG] wrapped N self_attn + N block forward(s)
[AnimaSafePAG] attached ✅ engine=Anima pert=on (...) APG=on (...) AdaptiveG=on (...)
[AnimaDetailDaemon] active ✅ amount=... range=... couple=... cfg=...
```

`engine=... (not 'Anima')` 나 `head layout differ (grouped-query?)` 같은 로그가 보이면 해당
지점만 조정이 필요합니다.

---

## 크레딧

| 기능 | 원본 | 라이선스 |
|---|---|---|
| Perturbation Guidance (PAG) | [iljung1106/comfyui-anima-safe-pag](https://github.com/iljung1106/comfyui-anima-safe-pag) | 원 저장소 참고 |
| SEG | [SusungHong/SEG-SDXL](https://github.com/SusungHong/SEG-SDXL) (NeurIPS'24) | 원 저장소 참고 |
| SLG (Skip Layer Guidance) | SD3.5 / Wan 계열 커뮤니티 구현 | — |
| APG | [MythicalChu/ComfyUI-APG_ImYourCFGNow](https://github.com/MythicalChu/ComfyUI-APG_ImYourCFGNow) · [논문](https://huggingface.co/papers/2410.02416) | 원 저장소 참고 |
| Detail Daemon | [muerrilla/sd-webui-detail-daemon](https://github.com/muerrilla/sd-webui-detail-daemon) | 원 저장소 참고 |

각 알고리즘은 Forge Neo용으로 **이식/재구현**한 것이며(vendor 아님), 저작권은 각 원저자에게
있습니다.
