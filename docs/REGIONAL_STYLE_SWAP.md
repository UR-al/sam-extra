# Regional Style-Swap — RegionalSampler 워크플로를 SAM3 Refine로 재현

rouge-kasshoku의 "ANIMA Crossover Couple Generation using Regional Sampler" 가이드는
ComfyUI-Impact-Pack의 **`RegionalSampler`(latent-레벨, 2-pass, LoRA 격리)** 로 크로스오버
커플의 **스타일 블리딩**(예: Lina의 눈 스타일이 Nadia를 덮어씀)을 해결합니다.

Forge Neo에는 RegionalSampler가 없지만, **이미 있는 SAM3 Refine 패널(image-레벨 마스크
인페인트 + 수동마스크 + per-pass LoRA)** 로 **같은 결과를 근사**할 수 있습니다. 아래는 코드
없이 지금 바로 쓰는 레시피입니다.

> 정직한 한계: SAM3 Refine는 **image-레벨 인페인트**(가이드가 말한 방법 #2)라, 진짜
> latent-레벨 `RegionalSampler`(#3)보다 seam(경계)이 약간 더 생길 수 있습니다. mask blur +
> "inpaint only masked" + 동일 seed/sampler로 대부분 눌러집니다.

---

## 핵심 파라미터 매핑

| 가이드 (RegionalSampler) | SAM3 Refine 대응 | 근거 |
|---|---|---|
| `base_only_steps` | **Denoising Strength** = `1 − base_only_steps / steps` | B스텝 후 분기 = img2img가 (1−D)·T 지점부터 재생성 |
| `overlap_factor` (10→16→24) | **Mask Blur** (≈10→16→24) | 경계 블렌드 폭 |
| 마스크 작게 = 베이스 제약 강 | 동일 (작게 그림) | — |
| region LoRA weight (0.8~1.0) | Replacement에 `<lora:이름:0.8>` | LoRA를 이 패스에만 → **격리 달성** |
| 결정론적 샘플러 + 동일 seed | Sampler=**Euler**, 🎯 Pull from selected | 컴포지션 고정 |

**denoise 예시** (steps=33 기준): base_only_steps 6 → **0.82**, 8 → **0.76**, 10 → **0.70**,
12 → **0.64**, 16 → **0.52**. (베이스에 더 붙이려면 base_only_steps↑ = denoise↓.)

---

## 단계별 레시피

### 1) 베이스(레퍼런스) 이미지 t2i 생성
- 프롬프트 팁(가이드): 시리즈 태그를 **각 캐릭터마다 반복**(`kofune mio, summertime render, kofune ushio, summertime render`), **위치 태그**(`on the left/right`), **마침표로 개념 분리**, Booru 태그→자연어→배경 순.
- 프록시 전략: 타깃과 **구조가 비슷한 네이티브 캐릭터**로 뽑고, 베이스에서 미리 타깃 트레잇(`tanned skin`, `yellow eyes`)을 프롬프트해 두면 나중 스왑이 쉬워짐.
- 설정 예: `Euler`, `sgm_uniform`, CFG 5.0, 33 steps, 1216×832.
- **이 이미지의 seed를 기억**(나중에 동일 seed 사용).

### 2) Refine 패널에서 지역 스왑
갤러리에서 베이스 이미지 선택 → **SAM3 Refine** 아코디언:

1. **Manual Mask** 아코디언 열기 → **📋 Load selected to canvas** → 스왑할 영역(예: 오른쪽
   캐릭터의 눈/얼굴 상단)에 스크리블. *(Target 비워도 됨 — 스크리블 자체가 마스크가 됨.)*
   - **극단적 스타일 블리딩(눈)**: 얼굴 상단 안쪽만 **타이트하게** 칠하면 눈 스타일만 덮어쓰고
     나머지 두상은 베이스가 유지됨(가이드의 "mask less = more base constraint").
2. **Replacement**: region 프롬프트(트레잇+라이팅만, 프레이밍/앵글 X) + `<lora:타깃:0.8>`
   예) `masterpiece, best quality, anime screenshot, kana higa, long blonde hair, thick bangs, expressive yellow eyes, yellowish-brown skin, soft expression, subsurface scattering, film grain <lora:kanahiga:0.8>`
3. **Inherit main t2i prompt: OFF** (region 프롬프트만 사용 — 프레이밍은 베이스 이미지가 제공).
4. **Denoising Strength**: base_only_steps로 환산(기본 **0.76** ≈ B8). 각도/프레임이 자꾸
   틀어지면 **낮춰서**(0.64~0.52) 베이스에 더 고정.
5. **Mask Blur**: **16** (overlap_factor 16). 경계 색 튐 있으면 24로.
6. **Seed**: **🎯 Pull from selected** (베이스와 동일 seed).
7. **Sampler**: `Euler`(결정론적), **Scheduler**: `sgm_uniform`, **CFG**: 5.0, **Steps**: 33.
8. **Inpaint Only Masked**: ON(작은 영역·와이드샷일수록 유리), **Padding**: 32.
9. **Masked Content**: `original`(베이스 구조 유지) 권장.
10. **▶ Refine** → 결과가 갤러리에 추가됨. 마음에 안 들면 아래 순서로 조정:
    - Denoising(=base_only_steps) → LoRA weight → region 프롬프트 → 마스크 다시 그림.

### 3) 여러 캐릭터 스왑 (Case 2/3)
- 한 영역 스왑이 끝난 결과를 다시 선택 → 두 번째 캐릭터 영역에 반복(각자 LoRA).
- **LoRA 격리**: 각 패스의 Replacement에만 해당 LoRA를 넣으므로, 한 캐릭터 LoRA가 다른
  영역을 오염시키지 않음(가이드의 핵심 장점을 그대로 얻음).

---

## 가이드 기본값 → Refine 프리셋 (요약)

| 필드 | 값 |
|---|---|
| Sampler / Scheduler / CFG / Steps | Euler / sgm_uniform / 5.0 / 33 |
| Denoising Strength | 0.76 (= base_only_steps 8) |
| Mask Blur | 16 |
| Inherit main prompt / negative | OFF / OFF |
| Inpaint Only Masked / Padding | ON / 32 |
| Masked Content | original |
| Seed | 🎯 Pull from selected (동일 seed) |
| Replacement | region 트레잇+라이팅 + `<lora:타깃:0.8>` |

v0.9.15에서 이 프리셋을 **버튼 한 번**으로 적용하는 "🎭 Regional Swap preset"을 Refine 패널에
추가했습니다(아래 2번).
