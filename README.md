# sam-extra (Forge SAM3 Extension)

SAM3 / SAM3.1 마스크 + 인페인트 확장. 일곱 가지 워크플로 제공:

1. **In-flight** — t2i/img2img 생성 직후 자동으로 SAM3 마스킹 → 인페인트 (ADetailer 스타일)
2. **Refine 패널** (v0.4.0+) — ⚠️ **실험 기능 (아직 제대로 작동하지 않음)** — 갤러리에서 이미지 골라 즉시 SAM3+인페인트+CN으로 재손질, 결과를 갤러리에 누적
3. **Anima Tile-Repair** (v0.8.0+) — ⚠️ **실험 기능 (아직 제대로 작동하지 않음)** — [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)의 Anima ControlNet-LLLite 추론을 가져와 임베드 (Apache-2.0)
4. **LoRA Manager** (v0.9.0+) — [willmiao/ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager)를 그대로 가져와 extra-networks 탭에 임베드 (GPL-3.0)
5. **txt2img Notebook** — 한 Forge 화면에서 이름 붙인 프리셋을 만들고 프롬프트·네거티브·LoRA·XYZ Plot·생성 설정을 원하는 조합으로 즉시 적용
6. **Anima Character Reference / ReStyler** — 참조 이미지 옆에 생성 영역을 만들고 Anima Edit로 마스킹 인페인트한 뒤 원하는 해상도로 정확히 추출
7. **Anima 3.8B (Qwen3.5 / v2)** — Anima-3.8B v1.1 번들의 Semantic Connector v2 를 되살려 Qwen3.5-4B 의미 특징을 조건에 넣음 ([GumGum10/forge-anima-3.8B](https://github.com/GumGum10/forge-anima-3.8B) 편입, MIT)

ControlNet 통합 (LLLite 인페인트 모델 자동 호환 처리), 옷 교체용 Target/Replacement 워크플로, 시드 고정, VRAM 절약 옵션, XYZ plot 다축 등 지원.

또한 SAM3와 **완전히 분리된 독립 기능**으로 **Anima Guidance Suite**(v0.11.0+, PAG/SEG/SLG ·
APG/CWM/SMC · Skimmed CFG · DCW · DAVE · CNS · **보조 CLIP-L Modulation Guidance** ·
Adaptive Guidance · Detail Daemon)를 제공합니다.
기능마다 구현 방식과 검증 수준이 다르므로, 사용 전에 반드시 아래 **[구현·검증 상태](#구현검증-상태)**와
**[상세 가이드](docs/GUIDANCE.md)**를 확인하세요.

**TIPO 프롬프트 확장 (🪄)** — txt2img 프롬프트를 TIPO-v2.1 로 확장해 태그·설명을 덧붙입니다(아래 별도 기능 참고).

그리고 **Anima VAE 2x** (v0.9.14+, 실험) — spacepxl 2x Wan-VAE 파인튜닝을 디코더로 써서
speckle↓·skin/hair 정리(`scripts/anima_vae_2x.py`). Qwen/Wan VAE latent 공유로 Anima에도 적용.

> 워크플로 3·4는 외부 프로젝트를 vendored하여 통합한 것입니다. 워크플로 6은 공개
> ReStyler 워크플로의 동작을 Forge 네이티브 경로로 다시 구현했으며 외부 코드는
> vendoring하지 않습니다. 자세한 출처는 아래 [출처 / 크레딧](#출처--크레딧-credits) 참고.

---

## 설치

```bash
cd <sd-webui-forge-neo>/extensions
git clone https://github.com/UR-al/sam-extra.git
```

webui 재시작 후 t2i/img2img 패널에 **"SAM3 Mask"** 아코디언이 보이고, t2i 갤러리 아래 **"SAM3 Refine (post-generation)"** 아코디언이 보이면 정상.

ControlNet 통합은 `sd_forge_controlnet` 익스텐션이 함께 로드돼 있을 때만 활성. 없어도 SAM3 본체는 정상 동작.

## SAM3 체크포인트

| 모델 | 파일 | 출처 (공식) |
|---|---|---|
| SAM3 | `sam3.pt` (3.45 GB) | <https://huggingface.co/facebook/sam3> |
| SAM3 | `sam3.safetensors` (3.44 GB) | <https://huggingface.co/facebook/sam3> |
| SAM3.1 multiplex (fp16) | `sam3.1_multiplex_fp16.safetensors` (1.75 GB) | <https://huggingface.co/facebook/sam3> |

공식 가중치: Meta [facebook/sam3](https://huggingface.co/facebook/sam3) (GitHub: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)).

<!--
  내부 메모(공개 문서엔 안 띄움): 위 파일명/포맷별 실사용 미러 — 공식 repo에
  없는 repack(.safetensors / fp16 multiplex)은 아래에서 받았음.
  - sam3.pt / sam3.safetensors : https://huggingface.co/1038lab/sam3
  - sam3.1_multiplex_fp16       : https://huggingface.co/Comfy-Org/sam3.1/tree/main/checkpoints
-->

위 파일들을 `<sd-webui-forge-neo>/models/sam3/`에 그대로 넣으면 UI의 "SAM3 Checkpoint" 드롭다운에 **파일명만** 자동 노출 (v0.5.0+ 풀 경로 표시 제거). 폴더가 없으면 직접 생성.

체크포인트 하나도 없으면 Hugging Face의 `facebook/sam3`에서 자동 다운로드. 완전 오프라인 사용 시 `--sam3-no-huggingface` 옵션으로 자동 다운로드 차단.

---

## 워크플로 1: In-flight (자동 detailer)

t2i가 끝나면 SAM3가 마스킹 → 인페인트 → 결과가 원본 이미지를 대체.

**SAM3 Mask 패널**에서:
- Enable SAM3 ✔
- Detect Prompt: `face` (또는 `eyes, hair / hand`)
- **Exclude Prompt** (v0.7.3+): 두 번째 SAM3 detect로 보호 영역을 잡아 메인 마스크에서 차감. 예: Detect=`clothes`, Exclude=`face, eyes, hand`
- Inpaint Prompt: 비워두면 메인 t2i prompt 사용
- Inpaint 아코디언에서 denoising, mask blur, sampler/scheduler/seed/steps 등 별도 지정 가능 (각 "Use separate ..." 토글)
- ControlNet 아코디언에서 인페인트 패스에 CN 유닛 1개 주입 가능

**Detect Prompt 문법**:
- `,` — OR 머지 (한 마스크로 합침)
- `/` — 분리된 인페인트 패스 (예: `face / hand` → 얼굴 인페인트 후 손 인페인트)

### 🎯 빠른 버튼 — 이미 만든 이미지에 바로 돌리기

txt2img 갤러리 아래 ✨(hires fix) 버튼 오른쪽의 **🎯** 는 선택한 이미지에 지금 SAM3 설정을 바로 돌립니다.
규칙은 ✨ 와 같습니다: 지금 txt2img 화면 설정(프롬프트·샘플러 등)으로 돌리고 시드는 그 이미지의 시드
(Forge 설정 *txt2img upscale same seed*), 결과는 Forge 설정 *hires button gallery insert* 대로 원본 자리를
바꾸거나 원본 뒤에 끼웁니다.

- SAM3 아코디언의 Enable 이 꺼져 있어도 돕니다 — 버튼을 누른 것 자체가 요청입니다. 다른 후처리(ADetailer,
  하이레스)는 돌리지 않습니다.
- 결과는 outputs 에 `-sam3` 접미어로 저장되고(Forge 의 자동 저장 설정을 따름), infotext 는 원본 뒤에 SAM3 설정과
  `SAM3 quick: True` 가 붙습니다.
- 마스크를 못 찾거나 모드가 Mask only 면 갤러리는 그대로 두고 infotext 칸에 알립니다. 그리드·컨트롤 이미지는
  ✨ 처럼 거절합니다. txt2img 의 Override 설정(Clip skip 등)도 ✨ 처럼 인페인트에 적용됩니다.
- Forge 가 ✨ 연결 방식을 바꾸면 버튼이 연결되지 않고 콘솔에 경고만 남습니다(생성에는 영향 없음).

---

## 워크플로 2: Refine 패널 (post-generation)

> ⚠️ **실험 기능 — 아직 제대로 작동하지 않습니다.** 동작이 불안정할 수 있으며 추후 수정 예정입니다.

t2i 끝난 후 갤러리에서 이미지 골라 즉시 재손질. 결과는 갤러리에 누적 삽입.

```
t2i Generate → 갤러리 N장
  → 손볼 이미지 클릭
  → Refine 패널에서 Target/Replacement 입력
  → ▶ Refine
  → 선택 이미지 옆(또는 끝)에 결과 추가
  → 새 이미지 또 클릭해서 chain refine
```

**Refine 패널 구성**:

| 필드 | 역할 |
|---|---|
| Target (마스크/치환할 대상) | SAM3가 마스킹할 토큰 + 메인 prompt에서 제거할 토큰 |
| **Exclude (보호할 영역)** | **두 번째 SAM3 detect로 마스킹한 영역을 Target 마스크에서 빼냄. 예: Target=`clothes`, Exclude=`face, eyes, hand` → 옷만 인페인트, 얼굴·눈·손은 원본 유지** |
| Replacement (대체할 단어) | 마스크에 그릴 내용 + Target 자리에 한 번만 삽입 |
| Negative Prompt | 옵션 |
| Inherit main t2i prompt | (기본 ON) LoRA/스타일 유지하며 Target만 segment 단위로 제거 |
| Inherit main t2i negative | (기본 ON) 같은 규칙으로 메인 negative도 정리 |
| SAM3 Threshold / Mask Dilation / Mask Hull / Mask Blur / Mask Processing | 마스크 후처리 |
| Denoising / Inpaint only masked / Padding | i2i 파라미터 |
| Steps / CFG / Sampler / Scheduler / SAM3 Checkpoint | 샘플링 파라미터 (Refine 패널은 항상 override) |
| Seed (-1 = random) | 시드 고정 가능 |
| Unload SAM3 from VRAM after detection | 인페인트 동안 SAM3 VRAM 해제 (≤12GB GPU 권장) |
| ControlNet 아코디언 | CN 유닛 옵션 (모델/모듈/weight 등) |
| Insert result: After selected / At end | 결과 삽입 위치 |

### Target/Replacement 동작 (v0.5.x)

```
메인 t2i prompt:    1boy, solo, white shirt, black necktie, belt,
                    score_9, <lora:detailedAnatomy:0.8>
Target:             shirt, necktie, belt
Replacement:        nude

→ 실제 sampler prompt:
   1boy, solo, nude, score_9, <lora:detailedAnatomy:0.8>
   (3개 segment 모두 제거되고 nude 한 번만 삽입,
    LoRA·anatomy context 그대로 유지)
```

- **부분 매칭**: `shirt`만 적어도 `"white shirt"` segment 전체 제거 (orphan 토큰 안 남음)
- **여러 패턴, 한 replacement**: replacement는 첫 매치 자리에 1회만 (`nude, nude` 중복 안 됨)
- **검증 로그**: stderr에 `[-] SAM3 Refine prompt transform: ...` 출력 — 실제로 어떻게 변환됐는지 console로 확인 가능

---

## ControlNet 통합

SAM3 인페인트 패스에 ControlNet 유닛 1개 주입. preprocessor에 따라 의미가 달라짐:

| Preprocessor | 보존 | 시나리오 |
|---|---|---|
| `inpaint_only` / `inpaint_global_harmonious` | 마스크 주변 컨텍스트 | 얼굴 디테일러 (가장자리 자연스러움) |
| `tile_resample` | 저주파 (전반적 색·형태) | 디테일 강화 |
| `depth_*` | 신체 깊이 / 실루엣 | **옷 교체** (실루엣 유지, 텍스처 자유) |
| `openpose_*` | 포즈 | 포즈 잠금, 옷·외형 자유 |
| `lineart_*` / `canny` | 윤곽선 | 형태 잠금, 색·재질만 변경 |

### CN 모델 위치

CN Model 드롭다운은 기본 `models/ControlNet/` **+** `models/sam3/` 둘 다 스캔. SAM3 검출 체크포인트(`sam3*.*`)와 같은 폴더에 LLLite 인페인트 모델(`anima-lllite-inpainting-v2.safetensors` 등) 두면 자동으로 드롭다운 노출.

### LLLite anima 인페인트 자동 호환

`anima-lllite-inpainting-*` 모델은 4채널(RGB+mask) 입력이 필요. `inpaint_only` 같은 mask-stripping preprocessor와 조합하면 어설션 실패. **익스텐션이 자동 감지해서 preprocessor를 `None`으로 override** (stderr에 한 줄 로그). 사용자가 따로 신경 안 써도 됨.

### ⚠️ 옷 교체가 안 바뀌어 보일 때

`anima-lllite-inpainting-v2`는 *"주변 컨텍스트와 자연스럽게 섞기"* 가 목적이라 옷 교체를 적극 방해함. 옷을 **확실히 바꾸려면**:

- **CN 끄기** — 가장 효과적
- 또는 CN Weight 1.0 → 0.4~0.6
- 또는 `depth_*` CN으로 교체 (신체 실루엣만 유지, 옷은 자유)

---

## VRAM 절약 (≤12 GB GPU)

SAM3 체크포인트는 ~3.5 GB. 한번 로드되면 `lru_cache(maxsize=2)`에 잡혀서 인페인트 동안 VRAM 점유. Forge의 `reserve-vram` 경고가 뜨면:

1. **"Unload SAM3 from VRAM after detection"** 체크 (SAM3 패널 + Refine 패널 양쪽 모두 옵션 있음). 검출(~2초) 끝나면 캐시 비우고 `cuda.empty_cache()` → 인페인트 사이클이 풀 VRAM 활용. 다음 검출은 ~3~5초 재로딩 비용.
2. webui 실행 인자에 `--reserve-vram 2` 추가 — 모델 매니저가 헤드룸 2 GB 확보.

둘 같이 쓰면 가장 안정적.

---

## 마스크 후처리

머리카락·털·strand 등 가는 부분이 SAM3에 부분 누락되는 경우용:

| 옵션 | 효과 | 추천 시나리오 |
|---|---|---|
| **Mask Dilation (px)** (최대 256) | 마스크를 N 픽셀 바깥쪽 확장 | 강한 가장자리 (옷, 물체) |
| **Convex Hull** | 검출 영역을 최소 볼록 다각형으로 감쌈 (컴포넌트별 적용) | 머리·털 strand 사이 공간까지 자동 포함 |
| **Mask Blur** | 가장자리 부드럽게 | 인페인트 합성 자연스러움 |

적용 순서: `raw mask → hull → dilation → blur` (core.py에서 자동)

---

## 진행률 / 검증

매 Refine/in-flight 패스마다 stderr에:
- 마스크 커버리지 % (SAM3가 옷 전체 잡았는지 vs 일부만 잡았는지)
- 인페인트 knob 전체 (denoise, fill, sampler, scheduler, CN model/weight 등)
- ScriptSampler 슬롯 patch 결과 (사용자 설정이 정말 적용되는지)
- prompt 변환 결과 (Target/Replacement이 메인을 어떻게 바꿨는지)

추가로 webui 갤러리 사이드바에 **per-image infotext 갱신** — Refine으로 추가된 이미지 클릭 시 변환된 prompt가 즉시 보임 (v0.5.2+).

---

## XYZ Plot 축

기존 SAM3 항목 + ControlNet 통합 + 신규 v0.6.0 항목:

`Enable, Checkpoint, Mode, Mask Mode, Device, Detect Prompt, Exclude Prompt, Inpaint Prompt, Negative Prompt, Prompt S/R (2종), Threshold, Mask Dilation, Mask Hull, Mask Blur, Denoising, CFG, Steps, Inpaint Only Masked, Padding, Inpaint Width/Height, Sampler, Scheduler, Seed, Noise Multiplier, Restore Face, Unload After, CN Enable, CN Override, CN Model, CN Module, CN Weight, CN Guidance Start/End`

---

## Settings 저장

모든 위젯에 `elem_id` 부여 (v0.6.0). webui Settings → **"Save UI defaults"** 클릭 시 SAM3 패널 + Refine 패널 값 전부 저장됨. 다음 세션 시작 시 자동 복원.

### Forge UI 테마

`Settings → SAM Extra Appearance`의 **Forge UI 테마**에서 전체 Forge/Gradio 화면의
표면 팔레트를 바꿀 수 있습니다.

- `Forge Default`: 확장의 전역 색상 덮어쓰기를 완전히 해제
- `Graphite Ember`: 중립 흑연색 + Forge 계열 주황 강조
- `Obsidian Violet`: 보랏빛이 아주 약하게 섞인 흑요석색 + 보라 강조
- `Warm Espresso`: 따뜻한 갈흑색 + 호박색 강조
- `OLED Mono`: 순흑이 아닌 저채도 근흑색 + 미색 강조

Settings에서 선택 후 **Apply settings**를 누르면 현재 페이지에 즉시 적용되고 다음 실행에도
유지됩니다. 구현은 이 확장의 CSS 변수/JavaScript에만 있으며 Forge Neo 본체 파일과 Gradio
테마 설정은 변경하지 않습니다. 모든 커스텀 테마는 동일한 글꼴·간격·포커스·상태 규칙을
공유하고 팔레트만 전역으로 교체합니다.

---

## 의존성

`requirements.txt` — Forge launch 시 자동 설치. SAM3 본체는 `sam3` PyPI 패키지 필요.

ControlNet 통합은 `sd_forge_controlnet` 익스텐션에 lazy import 의존. 없으면 해당 UI/로직만 비활성화.

---

## 워크플로 3: 복원/업스케일 (Anima Tile-Repair + PiD) (v0.8.0+)

> ⚠️ **실험 기능 — 런타임 검증 미완.** 정적 버그는 정리했으나 실제 모델로 end-to-end 확인이 필요합니다.

갤러리 선택 이미지를 복원/업스케일하는 후처리 패널. **복원 모드** 두 가지를 옵션으로 제공합니다:

- **Anima Tile-Repair** (기본) — vendored kohya sd-scripts의 Anima ControlNet-LLLite tile 복원. **Qwen3 TE + Qwen-Image VAE 필수**(패널 드롭다운이 폴더의 후보 파일을 자동 선택; 없으면 명확한 안내). 마스크 기반 가능.
- **PiD Upscale** (v0.9.6+) — Forge Neo **네이티브** [NVIDIA PiD](https://huggingface.co/nvidia/PiD)(Pixel Diffusion Decoder) 초해상 복원. 파일명에 `PiD`가 포함된 체크포인트를 `models/Stable-diffusion/`에 넣으면 자동 활성화(`backend/loader.py`). img2img로 동작(`denoising_strength`→degrade σ로 재해석), **마스크 미사용·전체 이미지 업스케일**. vendor 불필요 — Forge가 모든 처리를 함.

아래 사용 흐름은 Anima Tile-Repair 기준입니다.

### 사용 흐름

```
t2i Generate → 갤러리 N장
  → 디테일/노이즈 복원하고 싶은 이미지 클릭
  → SAM3 Anima Tile-Repair 아코디언에서 LLLite 모델 선택
  → ▶ Anima Tile-Repair
  → 결과가 선택 이미지 옆에 삽입됨
```

### 의존성 자동 설치

확장 첫 로드 시 `install.py`가 `kohya-ss/sd-scripts` repo를 `extensions/forge_sam3_extension/anima_vendor/` 로 shallow clone합니다 (~30MB, ~30초). `git`이 PATH에 있어야 합니다. 실패하면 패널만 비활성화되고 나머지 SAM3 기능은 정상 작동.

### 필요 모델 (사용자 디스크 위치 기준)

| 종류 | 권장 경로 | 비고 |
|---|---|---|
| Anima DiT | `models/Stable-diffusion/ANIMA_*.safetensors` | "Use Forge current" 선택 시 현재 Forge sd_model 사용 |
| Qwen3 Text Encoder | `models/text_encoder/*_txt.safetensors` | 별도 지정 가능 |
| Qwen-Image VAE | `models/VAE/qwen_image_vae.safetensors` | 별도 지정 가능 |
| ControlNet-LLLite | `models/ControlNet/animaTileRepair_v10.safetensors` 등 | **필수** |

### VRAM 관리

기본 ON된 `Unload Forge SD before run` 옵션이 Anima 추론 전 `backend.memory_management.unload_all_models()`를 호출 → 현재 SD model을 VRAM에서 빼냅니다. **`sd_models.unload_model_weights()` (모델 nuke)와 다릅니다** — `forge_hash`가 보존돼서 다음 t2i가 idempotent reload로 살아남습니다.

### 한계 / 알려진 제약

- **단일 패스만 지원 (v0.8.0)**: 큰 이미지를 작은 tile로 split해서 추론하는 진짜 tiling 루프는 v0.8.1 작업. 현재는 source 이미지를 width/height 슬라이더 크기로 한 번에 추론.
- **Sampler 선택 불가**: Anima는 Flow Matching only. `flow_shift` + `infer_steps` 만 sampling을 결정.
- **Attention backend**: Windows 환경에서 `flash_attn` / `sageattention`은 빌드 어려움. vendor가 `torch` (SDPA) fallback으로 작동.

### infotext

결과 PNG의 `parameters` chunk에 `Anima Tile-Repair: on` 마커 + LLLite 설정 / steps / cfg / seed 가 적힙니다. 갤러리에서 결과를 클릭하면 사이드바 prompt가 변환된 prompt로 갱신.

---

## 워크플로 4: LoRA Manager 통합 (v0.9.0+, v0.9.1에서 정상 작동)

> **출처 명시:** 이 기능은 [**willmiao/ComfyUI-Lora-Manager**](https://github.com/willmiao/ComfyUI-Lora-Manager) (GPL-3.0) 프로젝트를 **그대로 가져와(vendored)** Forge에 임베드한 것입니다. LoRA 관리 UI/기능 전부는 원저자 willmiao의 저작물이며, 이 확장은 그 standalone 서버를 Forge UI 안에서 띄우는 통합 레이어만 추가합니다.

[willmiao/ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager)를 Forge에 통합. extra-networks 탭 strip(🎴 버튼으로 여는 Checkpoints/LoRA 카드 영역)에 **Manage 탭**을 추가해서 LoRA 관리(civitai 다운로드, 메타데이터/트리거워드 편집, recipe, preview)를 Forge 안에서 바로 합니다.

> **첫 실행 주의:** 서버가 처음 뜰 때 전체 LoRA 라이브러리를 스캔/해싱합니다 (예: 1487개 ≈ 4~5분). 이 동안 Manage 탭에 "LoRA 모델 스캔 중..." 진행 표시가 나오고, 끝나면 자동으로 UI가 로드됩니다. 두 번째 실행부터는 캐시 덕분에 즉시 뜹니다.

### 동작 방식

- standalone aiohttp 서버를 **lazy spawn** — Manage 탭을 처음 열 때만 백그라운드 프로세스로 실행 (최초 ~10초)
- Manage 탭 안에 `<iframe>`으로 manager UI 임베드
- Forge의 LoRA/checkpoint/embeddings 폴더 경로를 manager `settings.json`에 자동 동기화
- Forge 종료 시 서버 자동 종료 (atexit)
- **모든 "send to workflow" → Forge 프롬프트 삽입 (v0.19.0+):** 카드 paper-plane뿐 아니라
  단일 우클릭 메뉴와 **멀티 선택 후 우클릭 벌크 전송**(`.model-card.selected` 전체)도 가로채
  ComfyUI 대신 현재 프롬프트로 삽입합니다. Append는 이어붙이고, Replace는 기존 `<lora:...>`를
  교체합니다. (벤더의 ComfyUI 하드코딩 전송을 브리지가 capture 단계에서 차단)

Notebook은 Forge 문서 하나만 사용하므로 Manage 탭도 하나만 주입됩니다. 매니저에서 LoRA를
전송하면 현재 txt2img 프롬프트에 바로 추가하거나 기존 LoRA 토큰을 교체합니다.
`/sam3-lora/config`·`/sam3-lora/spawn` 엔드포인트는 같은 페이지의 매니저를 조회·기동하는 데
사용됩니다.

### 의존성 자동 설치

확장 첫 로드 시 `install.py`가:
1. `willmiao/ComfyUI-Lora-Manager`를 `lora_manager_vendor/`로 shallow clone (~20초)
2. 누락된 경량 deps(aiohttp-socks, piexif, olefile, natsort, aiosqlite, beautifulsoup4)를 Forge venv에 자동 `pip install`

### 설정 (Settings → SAM3 LoRA Manager)

| 옵션 | 기본값 | 설명 |
|---|---|---|
| Manage 탭 배치 | `Add Manage tab (keep LoRA)` | LoRA 탭 옆에 Manage 탭 추가 / `Replace LoRA tab`이면 LoRA 탭 자리를 대체 |
| 서버 포트 | `8765` | ComfyUI 기본 8188과 충돌 회피. 재시작 후 적용 |

txt2img + img2img 양쪽 extra-networks strip 모두에 주입됩니다.

### 포함 기능 / 빠지는 기능

standalone 웹 UI 기능(검색·다운로드·정리·프리뷰·메타데이터·트리거워드·레시피·통계 등)은 **원본과 100% 동일하게** 동작합니다. 빠지는 것은 ComfyUI 노드 그래프 전용 기능뿐입니다:

- **ComfyUI 커스텀 노드(Lora Loader, Trigger Words Toggle, Save Recipe 등) — N/A.** Forge엔 노드 그래프가 없습니다. LoRA 실제 적용은 Forge 본체가 담당 (프롬프트에 `<lora:이름:0.8>` 입력 / extra-networks 카드 클릭).
- **생성 직후 자동 레시피 캡처 제한** (라이브 생성 메타데이터 수집은 ComfyUI 실행 엔진에 의존 → standalone에서 mock). 단 이미지 파일 기반 수동 레시피 임포트는 정상.

### 한계

- iframe 임베드라 Forge Gradio 테마와 시각적으로 완전히 통합되지는 않음 (manager 자체 UI)
- `git` PATH 필요 (vendor clone)

---

## 별도 기능: ANIMA LoRA 블록 호환 변환

Forge 본체 파일을 수정하지 않고, LoRA가 로드되는 `networks.process_anima` seam에 작은
adapter를 설치합니다. 현재 선택한 ANIMA 체크포인트의 실제 DiT 블록 수를 읽어 다음 변환을
메모리에서 자동 적용합니다.

| LoRA 원본 | 현재 모델 | 동작 |
|---|---|---|
| Base 1.0 (28) | 2.9B (40) / 3.8B (52) | 새 블록에 앞선 계보의 LoRA delta를 복제 |
| 2.9B (40) | Base 1.0 (28) / 3.8B (52) | Base 계보만 선택하거나 3.8B 삽입 블록으로 확장 |
| 3.8B (52) | Base 1.0 (28) / 2.9B (40) | 상속 블록만 선택하고 3.8B 전용 블록을 제거 |

- 원본 `.safetensors` 파일은 변경하지 않으며 별도 버튼이나 체크포인트 변환 과정도 없습니다.
- Base→2.9B는 Forge의 기존 매핑을 그대로 유지합니다. 2.9B→3.8B는 3.8B 체크포인트
  metadata의 12개 삽입 위치(`3, 7, …, 47`)를 사용하고, Base↔3.8B는 두 세대 매핑을
  합성합니다.
- 상향 변환은 기존 Forge Base→2.9B와 같은 정책으로 삽입 블록에도 앞선 계보의 LoRA
  delta를 복제합니다. 상속 블록에만 재배치하는 별도
  [ComfyUI 브리지](https://github.com/Lakeside529/ComfyUI-Anima-3.8B-LoRA-Bridge)의
  기본 정책과는 다르며, 3.8B에서 어느 쪽이 더 좋은지는 LoRA·시드별 실제 이미지 A/B가
  필요합니다.
- 하향 변환은 2.9B/3.8B 전용 LoRA delta를 버리는 **손실 투영**입니다. 3.8B→Base는
  추가된 24블록을 제거하며 3.8B 전용 semantic-connector 키도 적용할 대상이 없어
  제외하고 콘솔에 경고합니다. 별도 text encoder인 Qwen3.5 키는 블록 수만으로 삭제하지
  않습니다.
- Forge가 인식하는 Kohya 키(`lora_unet_blocks_0…N`)와 native diffusion 키
  (`diffusion_model.blocks.0…N`)의 완전한 연속 레이아웃만 자동 변환합니다. 일부 블록만 든
  sparse LoRA는 원본 세대를 안전하게 판정할 수 없으므로 DiT 블록은 추측 변환하지 않습니다
  (Forge 공통 LLM-adapter 키 정규화만 그대로 수행).
- 단, 더 큰 모델에서 **선두 블록만 정확히 `0…27` 또는 `0…39`로 저장한 특수 partial
  LoRA**는 키만 보면 완전한 28/40블록 LoRA와 구별할 수 없습니다. 이런 파일은 자동 변환
  대상에 쓰지 않는 것이 안전합니다.
- 이는 **LoRA 인덱스 호환 변환**입니다. Base나 2.9B 체크포인트 자체를 학습된 3.8B
  체크포인트로 바꾸는 기능은 아닙니다.

---

## 워크플로 5: txt2img Notebook

Live Workspace의 다중 Gradio 문서·iframe·별도 라우트 대신, 원래 Forge 주소의 **문서 하나**만
사용합니다. txt2img는 Prompt/Negative/Generate 아래에 **Parameters / Scripts / Gallery** 3열로
정리되고, Notebook은 **Gallery 바로 아래**에 놓입니다. Forge의 Default/Compact 프롬프트 레이아웃
양쪽을 지원하며 좁은 화면에서는 한 열로 접힙니다.

- **Parameters**: sampler, 크기, CFG와 always-on 확장 기능을 원래 구성 그대로 표시
- **Scripts**: Forge의 Script 선택기와 Prompt Matrix / Prompts from file / X/Y/Z Plot 패널만 표시
- **Gallery**: Forge 기본 갤러리·진행 미리보기와 그 아래 Notebook 표시
- 네거티브 프롬프트 아래의 Forge 원본 **Generation / Textual Inversion / Checkpoints / Lora**
  탭 바도 함께 복구하므로 Checkpoint·LoRA 카드 브라우저를 계속 사용할 수 있음
- 커스텀 빠른 드롭다운은 처음에 설정된 개수(기본 60개)를 표시하고, 목록 끝까지 내리면
  다음 묶음을 자동으로 추가합니다. XYZ의 100개 이상 축 유형도 이름을 몰라도 끝까지
  스크롤해 고를 수 있으며, 묶음 크기는 `Settings → SAM Extra Appearance`에서 바꿀 수 있습니다.

Notebook을 펼친 뒤 머리의 `＋`를 누르면 `preset1`, `preset2` 순으로 프리셋이 생깁니다.
프리셋 이름을 클릭하면 편집 영역을 열고 닫으며, `✎`로 이름을 바꾸고 `…`에서 삭제합니다.
프리셋 안의 `＋ 항목 추가`로 여러 교체 항목을 한 프리셋에 묶을 수 있습니다.

지원 항목:

- 프롬프트, 네거티브 프롬프트, LoRA 토큰
- XYZ의 X/Y/Z 각각에 대한 **축 유형 + 축 값**
- Sampling Method, Schedule Type, Steps, Width/Height, Batch Count/Size
- Shift, CFG Scale, Rescale CFG, Seed

각 항목의 `현재값`은 현재 Forge UI 값을 프리셋으로 가져옵니다. 프리셋의 `적용`은 저장된 항목만
현재 화면에 덮어쓰며, 직전 적용은 Notebook 머리의 `↶`로 한 번 되돌릴 수 있습니다. XYZ 항목이
있으면 Script를 `X/Y/Z plot`으로 열고 CSV 입력 모드에서 축 유형을 먼저 적용한 뒤 값을 적용합니다.
적용 전 대상·숫자·선택지를 먼저 검사하고, 도중 오류가 나면 이미 바뀐 항목도 자동 복구합니다.
되돌리기에는 프리셋 값뿐 아니라 적용 전 Script 선택과 XYZ CSV 모드도 포함됩니다. 검색, JSON
내보내기/불러오기도 지원합니다.

### 저장 범위와 주의사항

- 변경은 약 0.65초 뒤 자동 저장되며 머리에 `저장 대기…`, `저장 중…`, `저장됨 HH:MM` 또는
  오류 상태가 표시됩니다.
- 저장 파일은 Forge 데이터 경로의 `sam-extra/notebook.json`입니다. 확장 업데이트로 덮어쓰지
  않으며 직전 정상 세대는 `notebook.json.bak`에 보관합니다.
- 원본과 백업이 모두 손상된 경우 빈 Notebook으로 간주해 덮어쓰지 않고 복구가 필요하다는 오류를
  표시합니다. 디스크 읽기·원자 저장은 Forge의 비동기 요청 처리를 막지 않도록 작업 스레드에서
  실행합니다.
- `/sam3-notebook`은 Forge/Gradio의 로그인 보호를 그대로 따르며, 읽기·저장 요청 모두 같은
  페이지에서 보내는 전용 헤더도 확인합니다.
- 다른 브라우저 탭에서 더 최신 revision을 저장했다면 오래된 탭은 덮어쓰지 않고 충돌을 표시합니다.
- Forge의 `Save UI defaults`는 기존처럼 전역 기본값을 담당합니다. Notebook은 기본값을 변경하지
  않고 사용자가 `적용`한 항목만 현재 txt2img 화면에 넣습니다.
- checkpoint/VAE 같은 전역 Quicksettings, 갤러리 이미지, img2img 상태는 프리셋에 저장하지
  않습니다.
- 같은 주소·브라우저에 기존 Live Workspace 데이터가 남아 있으면 Notebook 도구 모음에
  `이전 Workspaces 가져오기`가 나타납니다. 이를 누르면 각 Workspace의 지원 가능한
  Prompt/Negative/XYZ/생성 설정을 별도 프리셋으로 **추가**하며, 원본 localStorage 데이터는
  삭제하지 않습니다. LoRA 태그는 원래 프롬프트 안의 위치 그대로 보존하며, Notebook 지원
  대상 밖 컨트롤·빈 Workspace·프리셋 한도 초과분은 적용 전에 개수를 표시합니다.
- Python API와 새 JavaScript 자산을 등록하려면 업데이트 후 **WebUI를 완전히 재시작**하고
  브라우저에서 한 번 `Ctrl+Shift+R` 하세요. 접속 주소는 `/sam3-live`가 아닌 원래 Forge 루트
  주소(예: `http://127.0.0.1:7860/`)입니다.

---

## 워크플로 6: Anima Character Reference / ReStyler

캐릭터 이미지 **하나**만 넣으면 공개 [Anima ReStyler v1.2](https://civitai.com/models/2803070/anima-restyler)
워크플로와 같은 조건으로 그 캐릭터를 레퍼런스한 결과를 T2I 갤러리에 추가합니다. Forge Neo의
`anima_do_reference`를 작업 중에만 켜고, 끝나면 되돌립니다.

**사용법**
1. Anima 체크포인트를 로드합니다(3.8B v1.1 번들이면 3.8B 커넥터도 자동으로 켜집니다).
2. `models/Lora`에 **AnimeEditV2**를 둡니다(필수, 자동 감지). Extend LoRA는 선택이며 기본값은
   꺼짐입니다: [Extend Image - Image Edit (Anima Edit)](https://civitai.com/models/2752978?modelVersionId=3097523)
   v1.0을 civitai에 로그인한 상태로 받아(작성자가 로그인을 요구합니다) `Extend Image (Anima
   Edit) v1.safetensors`를 `models/Lora`에 두면(파일명에 "Extend Image"가 들어가면 자동 감지)
   전문가 설정에서 켤 수 있습니다.
3. 캐릭터 이미지를 넣거나, 비워 두고 T2I 갤러리에서 이미지를 선택합니다.
4. 유지 범위(정체성 / +의상 / +그림체)와 후보 수를 고르고 생성합니다.

결과 크기와 프롬프트는 txt2img 설정을 따릅니다(프롬프트 칸에 쓰면 그 문장을 그대로 사용).
상태 줄에 사용한 이미지, 모델 블록 수, 3.8B 커넥터, 찾은 LoRA, 생성 영역 크기와 확대 배율,
시드가 표시됩니다.

**전문가 설정**(접힘): 결과 크기 직접 지정, 캔버스 MP, Edit/Extend LoRA 강도, 앞머리, 샘플링
(또는 Forge Anima img2img 프리셋 따르기), 빈 칸 채우기(원본 단색 / 번진 이미지), 시드,
네거티브, 디버그 이미지 저장, 미리보기.

**기본값 = ReStyler v1.2**: 960×1088(수동 지정 시), 캔버스 1.4 MP, 빈 칸 `#000000` 그대로
(masked content `original`), 디노이즈 1.0, AnimeEditV2 0.72, Extend 0.4(기본 꺼짐),
Euler a / Simple / 30 / CFG 5, 앞머리 `(split screen, multiple views:1.2)`.

**불변 조건**: batch 1(레퍼런스 latent가 프로세스 전역), 전체 그림 인페인트(레퍼런스 패널 유지),
다른 스크립트 격리(와일드카드·negpip·LBW 문법은 처리되지 않으며 상태 줄에 경고).

> GPU로 원본 대비 정체성 유지 A/B는 아직 실행하지 않았습니다. 유지 범위(+의상/+그림체)의
> 내부 값은 임시값입니다.

---

## 별도 기능: Anima Guidance Suite

SAM3 처리와 분리된 opt-in 기능 모음입니다. Forge Neo 코어 파일은 수정하지 않으며, lightweight
`sam3ext.guidance` 수학 모듈만 불러옵니다. 모든 토글이 기본 OFF라 설치만으로 기존 seed 결과를
바꾸지 않습니다.

> [!IMPORTANT]
> 2026-07-23 Forge Neo 2.27 + `anima_baseV10`에서 공식 PAG와 공식 SEG가 실제
> `SelfCrossAttention.torch_attention_op`에 도달하고 non-zero weak delta를 만드는 것을 확인했습니다.
> CWM+DCW+DAVE+CNS 최소 활성 조합도 실제 Euler a 생성에서 실행 경로를 확인했습니다.
> 이는 **동작 검증**이며 모든 설정에서 화질 향상을 보장하는 벤치마크는 아닙니다.
> 2026-07-24 Modulation Guidance는 실제 권장 CLIP-L과 공식 어댑터 로드·투영 및
> Forge block 주입까지 검증했으며, 실제 checkpoint 이미지 A/B는 아직 별도 확인이 필요합니다.

### 구현·검증 상태

| 기능 | 이 확장에서 하는 일 | 현재 판정 / 제한 |
|---|---|---|
| **PAG** | appended weak row를 value-only attention으로 strength 보간 | 실제 Anima E2E 검증. Legacy Soft 토글 제공 |
| **SEG** | 실제 Anima T/H/W 중 H/W query에 Gaussian blur·보간 | 실제 Anima E2E 검증. Legacy uniform-value 근사 제공 |
| **SLG** | weak row의 선택 block을 no-op으로 복원 | 형상·순서 연결, 전용 화질 E2E 필요 |
| **APG** | post-CFG guidance를 cond 평행/직교 성분으로 투영 | 중립값 단위 검증. **CFG>1 권장**, reference와 픽셀 동일하지 않음 |
| **CWM / SMC** | Haar 대역별 CFG 배율 / step 간 unit-L2 switching control | 수학·중립값 검증, 실제 CWM 실행 확인 |
| **DCW** | live x_t와 x0의 wavelet 차이를 post-CFG 마지막에 보정 | 4D/5D·홀수 해상도·중립값 검증, 실제 실행 확인 |
| **RDC** | DCW Haar 대역의 step 간 EMA drift를 보정 | tau=0 비트 동일·상태/해상도 재초기화 단위 검증, 이미지 A/B 필요 |
| **DAVE** | Anima block 출력의 token/spatial DC 성분 감쇠 | 실제 block hit 확인, 다양성/권장 block은 추가 A/B 필요 |
| **CNS-inspired** | 기존 seeded/Brownian noise를 live x_t 에너지로 재색칠 | Euler a noise call 확인. deterministic sampler에서는 inert |
| **Anima Modulation Guidance** | 보조 CLIP-L 방향을 공개 어댑터로 block AdaLN에 가산 | 실제 CLIP/어댑터 로드·투영·주입 검증. 이미지 품질 A/B는 추가 필요 |
| **Adaptive Guidance** | combined batch의 후반 uncond row 생략 | low-VRAM 분리 호출에서는 생략/속도 이득 없음 |
| **Skimmed CFG** | 과포화를 만드는 성분만 낮은 CFG로 되돌림(anti-burn) | 상단 수식과 tensor 단위 일치 검증. **CFG>1 전용**, pre-CFG가 아닌 post-CFG 재구성 |
| **Detail Daemon** | sigma schedule로 디테일 강도 조절 | 별도 opt-in 기능 |

Modulation Guidance를 쓰려면 768차원 CLIP-L safetensors를
`models/text_encoder/`에 둡니다(상류 예제의 권장 파일명:
`Anzhc Noobai11 CLIP L Anime.safetensors`). CLIP 파일 자체는 자동 다운로드하지 않으며,
공식 `yresearch/cosmos-pooled` 어댑터만 첫 활성화 때
`models/anima_modulation_guidance/`로 자동 다운로드·무결성 검증합니다.

### 오케스트레이터 순서

`ADG/PAG 배치 → CLIP block modulation → DAVE → attention PAG/SEG → Skimmed CFG
→ CFG base(SMC → APG → CWM)
→ PAG/SEG/SLG delta → DCW/RDC → CNS sampler noise`

- CFG base의 **SMC·APG·CWM은 독립 토글**입니다. SMC는 프리셋과 별도의 master
  체크박스로 값을 유지한 채 A/B할 수 있습니다. 셋 다 끄면 MaHiRo/custom CFG 결과를
  그대로 유지하고, 켜진 것들은 항상 `SMC → APG → CWM` 순서로 적용됩니다.
- SMC는 ComfyUI-DCW와 같은 `Off / Auto / SD1.5·2 / SDXL / SD3·3.5 / Flux /
  Qwen-Image / Cosmos·Wan / Custom` 프리셋을 제공합니다. `Auto`에서 Forge `Anima`는
  `Cosmos / Wan (lambda 6.0, k 0.20)`으로 판별되며, 직접 수치는 `Custom`에서만 사용됩니다.
- CFG base가 켜지면 incoming에서 `w_eff`를 복원해 Forge의 `edit_strength`를 보존하고, 비선형 fit
  오차가 크면 경고합니다.
- Skimmed CFG는 별도 스크립트(별도 아코디언)이며 CFG base보다 **먼저** 실행되고, skim 결과를
  Forge의 예측 tensor에 다시 써서 이후 SMC/APG/CWM·PAG delta·DCW가 모두 그 위에서
  동작합니다. Forge의 중복 메서드 정의로 `sorting_priority`가 실제 실행에서 무시되는 문제를
  피하려고 callback을 목록 맨 앞에 직접 dedupe+prepend합니다.
- `Legacy CFG base mode` 아코디언의 라디오와 `Experimental stack`은 구버전 호환용이며
  위 토글과 OR로 합쳐집니다.
- CWM `alpha high > +0.15`는 Anima 16채널 latent에서 캐릭터 분리를 만들 수 있어 UI 경고가 뜹니다.
- DCW와 RDC도 각각 별도 토글입니다. RDC는 DCW의 Haar pass를 공유하는 step 간 EMA 보정이며,
  `tau=0.15`, `alpha LL=0.03`, `alpha HH=0`을 시작값으로 제공합니다. HH는 텍스처가
  뭉개질 수 있어 기본 0입니다.
- UI와 XYZ의 **Attn Scale**은 같은 값이며 attention 점수가 아니라
  `scale × (cond − weak)` 보정 배율입니다.
- PAG/SEG 공통 `Perturbation strength` 기본은 `0.75`, `1.0`이면 전체 perturbation입니다.
  block `18`, heads 빈칸(전체), Start/End `0.0/0.7`, Rescale `0.20`,
  Rescale mode `full`이 Anima Safe PAG 시작값입니다.
- CNS는 ancestral/SDE sampler에서만 의미가 있습니다. TeaCache는 포함하지 않습니다.
- Modulation Guidance는 메인 Qwen을 교체하지 않고 별도 768차원 CLIP-L을 사용합니다.
  기본 OFF이고 `w=0`에서도 base modulation은 남으므로 진짜 기준 이미지는 토글 OFF입니다.
- PAG/SEG 자체 A/B는 `Rescale=0`, SLG/APG/ADG off로 원인을 분리하세요.
- Guidance 본문의 관련 패널은 `DCW → RDC → CWM → SMC` 순으로 배치하고 DCW·DAVE·CNS도
  펼쳐 표시합니다. 이 화면 순서는 계산 순서(`SMC → APG → CWM`, 이후 DCW)와 구분됩니다.
  APG/Adaptive의 고급값만 접힌 세부 영역으로 둡니다.

확장 목록 아래 **Anima Reference-Latent PoC (debug / 안전)** 패널의
`Log Guidance verification summary`를 켜면 attention hit/raw delta, CFG `w_eff`/fit,
DCW/RDC eval, DAVE/Modulation block hit, CNS noise call, Adaptive 실제 생략 여부가 출력됩니다.

- 구현: `scripts/anima_safe_pag.py`, `sam3ext/guidance/`, `scripts/anima_detail_daemon.py`
- 테스트: `tests/test_anima_attention_patch.py`, `tests/test_anima_safe_pag.py`,
  `tests/test_guidance_suite.py`
- 상세 파라미터·처리 순서·XYZ·검증 로그·크레딧:
  **[docs/GUIDANCE.md](docs/GUIDANCE.md)**

---

## 워크플로 7: Anima 3.8B — Qwen3.5 / Semantic Connector v2 (편입)

Anima-3.8B v1.1 체크포인트는 Qwen3.5-4B 의 의미 특징을 매 디노이징 스텝에 주입하는
**Semantic Connector v2** 가중치를 파일 안에 담고 있습니다. Forge 본체는 그 190개 텐서를
`Anima Unexpected: anima_v2_connector…` 로 버리기 때문에, 확장 없이는 텍스트 인코더에
`Anima-3.8B-expanded_adapter` 나 `qwen35_4b` 를 넣어도 결과가 **한 픽셀도 바뀌지 않습니다**
(같은 시드 A/B 실측). [GumGum10/forge-anima-3.8B](https://github.com/GumGum10/forge-anima-3.8B)
(MIT) 의 런타임을 `sam3ext/anima38/` 로 편입해 이 확장 하나로 그 경로가 돕니다.

### 필요 파일
| 파일 | 위치 |
| --- | --- |
| `Anima-3.8B-v1.1.safetensors` (v2 번들) | `models/Stable-diffusion/` |
| `qwen35_4b.safetensors` (4.8 GB) | `models/text_encoder/` — 파일명에 `qwen35_4b` 가 들어가면 **자동으로 찾아 씁니다** |
| `qwen_3_06b_base.safetensors`, `qwen_image_vae.safetensors` | 순정 Anima 그대로 |

`qwen35_4b` 는 **VAE / Text Encoder 목록에서 고를 필요가 없습니다.** 목록에 넣어 두면 Forge 는
체크포인트를 불러올 때마다 4.8 GB 를 통째로 읽고 버렸는데(키 형식이 달라 쓰지 않음), 이제 확장이
그 파일을 로더에서 건너뜁니다. 그래서 XYZ Plot 의 체크포인트 축으로 Base 1.0 / 2.9B / 3.8B 를
**같은 모듈 목록**(`qwen_image_vae` + `qwen_3_06b_base`)으로 비교할 수 있습니다 — 3.8B 칸에서만
Qwen3.5 가 자동으로 붙습니다.

### 사용
1. 체크포인트로 v2 번들을 고르고, VAE/텍스트 인코더는 순정 Anima 처럼 `qwen_image_vae` +
   `qwen_3_06b_base` 만 선택합니다. (v1.1 번들에는 어댑터가 내장돼 있어 별도 어댑터 파일은
   필요 없습니다.)
2. 그냥 생성합니다 — 번들은 safetensors metadata 로 판별해 **아코디언이 접혀 있어도 자동**으로
   켜집니다. 콘솔에 `[Anima38] active — v2 bundle` 이 찍힙니다.
3. 아코디언의 **상태 확인** 버튼은 찾은 Qwen3.5 파일, 드롭다운에서 고른 체크포인트가 v2 번들인지, 이 탭의
   마지막 생성이 어떻게 돌았는지를 보여줍니다.
4. 선택: *Use adapter on negative prompt* 를 켜면 부정 프롬프트도 커넥터로 인코딩합니다.
   모델 카드의 공식 v1.1 ComfyUI 워크플로는 부정 프롬프트도 커넥터로 보냅니다. 이 확장의
   기본값은 기존과 같은 순정 인코더이며, 같은 시드 A/B 로 비교한 뒤 기본값을 정할 예정입니다.
5. 구형 v1(베이스 + 별도 `Anima-3.8B-expanded_adapter.safetensors`)은 아코디언을 켜고
   어댑터·강도를 고릅니다.

권장 시작점(모델 카드 v1.1): 약 1MP(예: 832×1216), CFG 4–7(공식 워크플로 6), 28–50 스텝(공식 40),
샘플러 `res_multistep` + 스케줄러 `Beta` — 모두 Forge 에 있습니다.

### 기록과 붙여 넣기
생성 파라미터(infotext)에 다음이 남습니다.

| 키 | 값 |
| --- | --- |
| `Anima38` | `v2 bundle` / `v1 adapter` / `bypass` / `off: <이유>` (예: `off: qwen35_4b.safetensors was not found …`) |
| `Anima38 encoder` | 실제로 쓴 Qwen3.5 파일 이름 |
| `Anima38 negative` | `native`(순정 인코더) / `connector`(커넥터) — v2 번들 |
| `Anima38 architecture` · `bundle` · `adapter` · `strength` · `negative strength` | 번들·어댑터 정보 |

PNG Info 나 Send to txt2img 로 붙여 넣으면 **Bypass·부정 프롬프트 설정·v1 어댑터와 강도**가 되살아나
같은 설정으로 다시 생성됩니다. 3.8B 기록이 없는 이미지는 이 칸들을 건드리지 않습니다. 키에 `.` 이 있으면
Forge 가 읽지 못해 이전 버전은 `Anima 3.8B …` 대신 `Anima38 …` 로 바꿨고, 예전 이미지(`Anima 3.8B …`)도
붙여 넣기에서는 그대로 읽습니다.

### API
`alwayson_scripts["Anima 3.8B (Qwen3.5 / v2)"]` 에 위치 인자 여섯 개
`[enabled, adapter, strength, negative, negative_strength, bypass]` 또는 SAM3 처럼 dict 하나
`{"args": [{"enabled": true, "negative": false}]}` 를 보낼 수 있습니다. v2 번들은 인자를
안 보내도 켜집니다. 끄려면 아코디언의 **Bypass** 체크박스 또는 `{"args": [{"bypass": true}]}`.
- **NegPiP 와 함께 쓸 수 있다 (설치 순서 무관).** v2 조건은 Forge 네이티브 계약(줄마다 텐서 하나)을 지키고 run id 는 마지막 토큰 행의 마커로 실어 보내므로 `get_learned_conditioning` 을 감싸는 확장이 있어도 죽지 않는다. NegPiP 이 우리 아래에 깔리는 순서에서는 NegPiP 의 가중치 마스킹을 우리가 대신 적용해 어느 순서에서도 결과가 같다. 패치는 생성이 끝나도 자리에 남되 꺼진 상태로 투명하게 위임하므로, 두 확장이 서로 다른 순서로 해제해도 사고가 없다.

### 동작·한계
- 두 인코더는 순차로 돌고 결과를 RAM 으로 옮긴 뒤 해제됩니다. 커넥터는 샘플링 모델의 일부로
  Forge 가 관리하며(오프로딩 가능) 생성이 끝나면 전부 원복됩니다. LoRA 세트가 바뀌어 Forge 가
  UNet 을 새로 복제해도(Feature 6, batch count 사이 LoRA 변경) 커넥터가 따라가 메모리 관리 안에 듭니다.
- 같은 프롬프트 줄의 Qwen3.5 특징은 최근 32줄까지 캐시합니다 — batch count·Feature 6 후보·
  ADetailer 가 같은 줄이면 Qwen3.5-4B 를 GPU 로 다시 올리지 않습니다. batch count 사이에는 설치를
  유지해 Forge 의 조건 캐시도 이어지고(Hires 체크포인트·Refiner 로 모델이 다시 로드되면 새로 설치, NegPiP 가
  앞 순서라 래퍼가 빠지면 다시 겁니다), 순정으로 가는 부정 프롬프트는 Forge 의 공용 캐시를 씁니다.
- `qwen35_4b.safetensors`(또는 동봉 토크나이저, v1 이면 어댑터 파일)가 없으면 생성 **전에** 알아채
  한 번 경고한 뒤 순정 Anima 로 진행합니다. infotext 에 `Anima 3.8B: off: …` 가 남습니다.
- VRAM 이 모자라 Forge 가 Qwen3.5 를 일부만 올려도 동작합니다(일부 가중치가 CPU 에 남아도 계산 장치로
  옮겨 씀). Qwen3.5 가 커넥터에 넘기는 층(7·15·23·31)은 여러 프롬프트에서 최대 |값| 35.5 로 측정돼
  fp16 텍스트 인코더에서도 넘침이 없습니다.
- 커넥터는 번들의 원본 `llm_adapter` 로 만든 **자기 사본**을 씁니다(텍스트 인코더와 같은 dtype, 약 +0.3 GiB
  RAM). 그래서 LoRA 에 든 `llm_adapter` 가중치도 v2 경로에 적용되고, 샘플링 직전마다 그 패스의 LoRA 세트에
  맞춥니다 — 텍스트 인코더와 같은 모듈을 둘이 제자리 패치하면
  LoRA 가 두 번 적용될 수 있어 사본을 둡니다.
- img2img 캔버스·ImageStitch 레퍼런스(Anima reference)를 순정과 똑같이 넘깁니다 — Feature 6 캐릭터
  레퍼런스가 3.8B 에서도 레퍼런스를 씁니다.
- SAM3 in-flight 인페인트 패스는 바깥 생성의 설치를 물려받아, 그 뒤에 도는 ADetailer 패스(같은 이미지나
  배치의 다음 이미지)도 v2 로 돕니다. 다른 탭의 생성이 도중에 죽어 남은 설치는 다음 생성 시작 때 내립니다.
- 체크포인트를 바꾸면 이전 모델을 붙잡지 않아 Forge 가 RAM 에서 비울 수 있습니다. Qwen3.5 는
  체크포인트와 무관해 계속 캐시합니다(XYZ 로 3.8B 를 오갈 때 다시 읽지 않게).
- LoRA·SAM3·ADetailer 와의 조합은 이 확장에서 함께 검증했습니다(ANIMA LoKr 3개 + SAM3
  인페인트).

## 별도 기능: TIPO 프롬프트 확장 (🪄)

txt2img 도구 줄(붙여넣기·지우기·스타일 적용 버튼)의 **🪄** 를 누르면
[TIPO-v2.1-1B-A200M](https://huggingface.co/KBlueLeaf/TIPO-v2.1-1B-A200M)(KBlueLeaf)이 지금 프롬프트를 읽고 어울리는
태그나 설명 문장을 덧붙여 프롬프트 칸에 다시 씁니다. 결과를 보고 고친 뒤 생성하면 됩니다. 설정은 스타일 줄 아래
**TIPO 프롬프트 확장** 칸에 있습니다.

### 처음 쓸 때
- 칸의 **모델 받기(1.98 GB)** 를 누르면 HF 에서 고정 버전(revision `f5a3185…`)을 `models/TIPO/TIPO-v2.1-1B-A200M/` 에
  받습니다. 누르기 전에는 아무것도 받지 않습니다.
- 새로 설치하는 패키지는 없습니다 — 모델 코드는 확장 안에 들어 있고(원작자 공개 코드 그대로) Forge 의 transformers 로 돕니다.

### 설정
| 칸 | 뜻 |
| --- | --- |
| 확장 방식 | 태그만 / 태그+설명(기본) / 설명만 |
| 길이 | 짧게 / 보통(기본) / 길게 |
| 새 작가·캐릭터·작품 허용 | 기본 꺼짐 — 켜면 TIPO 가 새 캐릭터·작품·작가(`@` 를 붙여서)도 넣을 수 있습니다 |
| 장치 | GPU(기본) / CPU — GPU 는 누를 때만 약 2 GB 를 올렸다가 끝나면 내립니다. 여유 VRAM 이 3 GB 보다 적으면 그 회차는 CPU 로 돕니다. 쉴 때는 RAM 에 약 2 GB 로 둡니다. 마지막으로 고른 장치는 브라우저가 기억합니다 |
| 시드 | -1 이면 누를 때마다 다른 결과, 숫자면 같은 결과 |
| ↩ 되돌리기 | 마지막 확장 전 프롬프트로 |

### 규칙
- **적어 둔 텍스트는 한 글자도 바꾸지 않습니다**(가중치·LoRA·순서 그대로). 새 태그는 그 뒤에 캐릭터 → 작품 → @작가 → 일반
  순서로 붙고, 설명 문장은 새 줄에 붙습니다.
- Anima 에 맞게 밑줄은 공백으로, 괄호는 `\(` `\)` 로 이스케이프합니다(그대로 두면 가중치로 읽힙니다).
- 품질·등급·시대·메타 태그는 적어 둔 것만 둡니다 — TIPO 가 낸 `great quality`·`newest`·`highres` 등은 버립니다. 이미 `1girl`
  처럼 인원수를 적었으면 TIPO 가 낸 다른 인원수 태그도 버립니다. `(smile:1.2)`·`((smile))`·`(@작가:0.8)` 처럼 가중치를 준 것도
  같은 태그로 보고 다시 붙이지 않습니다.
- 문장(쉼표가 들어간 문장 포함)은 TIPO 에 태그로 넘기지 않습니다.
- 토큰 한도에서 멈춰 끝이 잘렸으면 잘린 태그와 끝나지 않은 문장은 버리고, 상태 줄에 그렇게 알려 줍니다.
- 적어 둔 품질(`masterpiece` 등)·`year 2025`(→ 시대)·등급(`explicit` → `nsfw, explicit`, 여럿이면 가장 센 것)·`@작가`·캐릭터·
  작품·메타는 TIPO 에 조건으로 넘깁니다. 캐릭터·작품·메타 구분은 tagcomplete 확장의 danbooru CSV 를 씁니다(없으면 `@작가` 만
  알아봅니다).
- 확장은 Forge 생성 대기열 안에서 돕니다 — 생성 중에 누르면 그 생성이 끝난 뒤 돕니다. 기다리는 동안 프롬프트를 고쳤으면
  결과를 넣지 않고 상태 줄에 알려 줍니다(고친 내용을 덮어쓰지 않게).
- **모델 받기**는 한 번에 하나만 돕니다 — 받는 중에 다른 창에서 또 누르면 바로 알려 주고 다시 받지 않습니다. 새로 고친 뒤
  상태 줄이 예전 내용이면 🪄 나 **모델 받기**를 한 번 누르면 맞춰집니다.

## 출처 / 크레딧 (Credits)

이 확장은 아래 외부 프로젝트를 **그대로 가져와(vendored, shallow clone)** Forge에 통합합니다. 핵심 기능의 저작권은 각 원저자에게 있으며, 본 확장은 Forge 통합 레이어만 제공합니다. vendor 디렉터리는 저장소에 포함되지 않고 `install.py`가 첫 실행 시 자동으로 clone합니다.

| 기능 | 원본 프로젝트 | 라이선스 | vendor 위치 |
|---|---|---|---|
| **LoRA Manager** (워크플로 4) | [willmiao/ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager) | GPL-3.0 | `lora_manager_vendor/` |
| **Anima Tile-Repair** (워크플로 3) | [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) (`anima_minimal_inference*`) | Apache-2.0 | `anima_vendor/` |
| **Anima Character Reference / ReStyler** (워크플로 6) | [Anima ReStyler workflow](https://civitai.com/models/2803070/anima-restyler) · [원 아이디어 Reddit 게시물](https://www.reddit.com/r/StableDiffusion/s/0Az0DgoaKj) | 워크플로 페이지 조건 참고 | 동작을 Forge 네이티브 img2img/reference로 재구현 (vendor·코드 복사 없음) |
| **Anima 3.8B (Qwen3.5 / v2)** (워크플로 7) | [GumGum10/forge-anima-3.8B](https://github.com/GumGum10/forge-anima-3.8B) (commit `59c27e5`) | MIT | `sam3ext/anima38/` 에 편입 (저장소에 포함, 수정 사항은 `THIRD_PARTY_NOTICES.md`) |
| **TIPO 프롬프트 확장** (별도 기능) | [KBlueLeaf/TIPO-v2.1-1B-A200M](https://huggingface.co/KBlueLeaf/TIPO-v2.1-1B-A200M) · 모델 코드 [KohakUwULLM](https://github.com/KohakuBlueleaf/KohakUwULLM) | 가중치 Kohaku License 1.0 · 코드 Apache-2.0 | `sam3ext/tipo/kohaku/` 에 모델 코드만 편입, 가중치는 사용자가 받을 때 HF 에서 |
| **Anima Safe PAG** (별도 기능) | [iljung1106/comfyui-anima-safe-pag](https://github.com/iljung1106/comfyui-anima-safe-pag) | 원 저장소 라이선스 참고 | 이식 (vendor 아님) |
| **DCW / RDC / CWM / SMC** | [namemechan/ComfyUI-DCW](https://github.com/namemechan/ComfyUI-DCW) | GPL-3.0 | 수식 기반 Forge 재작성 (vendor 아님) |
| **DAVE** | [daheekwon/DAVE](https://github.com/daheekwon/DAVE) · [sorryhyun/ComfyUI-Anima-DAVE](https://github.com/sorryhyun/ComfyUI-Anima-DAVE) | MIT | Forge block 재구현 (vendor 아님) |
| **CNS-inspired Wavelet Noise** | [namemechan/comfyui-cns_sampler_patch](https://github.com/namemechan/comfyui-cns_sampler_patch) | GPL-3.0 | sampler-noise 재작성 (vendor 아님) |
| **Anima Modulation Guidance** | [Anzhc/Anima-Mod-Guidance-ComfyUI-Node](https://github.com/Anzhc/Anima-Mod-Guidance-ComfyUI-Node) · [quickjkee/modulation-guidance](https://github.com/quickjkee/modulation-guidance) · [yresearch/cosmos-pooled](https://huggingface.co/yresearch/cosmos-pooled) | MIT(코드 선언) / 자산 모델 카드 | Forge block 재작성 (vendor 아님) |
| **Skimmed CFG** (별도 기능) | [Extraltodeus/Skimmed_CFG](https://github.com/Extraltodeus/Skimmed_CFG) | **LICENSE 파일 미공개** | 공개 수식 기반 Forge 재작성 (vendor 아님) |
| **Detail Daemon** (별도 기능) | [muerrilla/sd-webui-detail-daemon](https://github.com/muerrilla/sd-webui-detail-daemon) | 원 저장소 라이선스 참고 | 이식/재구현 (vendor 아님) |
| SAM3 검출 | Meta [facebook/sam3](https://huggingface.co/facebook/sam3) ([facebookresearch/sam3](https://github.com/facebookresearch/sam3)) | Meta SAM 라이선스 | `sam3` PyPI 패키지 |

원저자분들께 감사드립니다. 각 프로젝트의 라이선스 전문은 vendor 디렉터리의 `LICENSE` 파일을 참고하세요.

### vendor 수정 사항 (GPL-3.0 §5 수정 고지)

LoRA Manager(GPL-3.0)는 vendor 그대로 실행하되, `lora_manager_core.py`가 spawn 시 아래 패치를 **marker-guarded·멱등**으로 적용합니다 (LICENSE·저작권 고지·저자 귀속은 일절 변경하지 않음 — DOM/동작만 조정):

- **fetch 진행 상태 줄바꿈** — `loading.css`의 `.loading-status` wrap 허용 (긴 LoRA 이름 뒤 카운터 잘림 수정)
- **후원/지원 UI 숨김** — Ko-fi/Patreon/WeChat/Afdian 등 기부 버튼·모달·배너를 `display:none`으로 숨김 (GPL이 보존을 요구하는 "Appropriate Legal Notices"가 아니므로 허용)
- **업데이트 알림 비활성화** — `update_routes.py`의 `check_updates`를 short-circuit해 willmiao 원본 릴리스 폴링/알림 점을 끔 (vendor는 install.py가 버전 고정하므로 사용자가 상류 업데이트에 조치 불가). 우리 repo로의 repoint은 버전 체계 불일치로 무의미하여 하지 않음.
- **Forge Neo 연동 (v0.9.4)** — ComfyUI 상호작용을 Forge Neo용으로 전환:
  - LoRA 카드의 "Send to ComfyUI"(✈️) → **"Add LoRA"**: 클릭 시 cross-origin iframe→부모 `postMessage`로 활성 탭 **Positive Prompt에 `<lora:이름:weight>` 삽입** (Forge 네이티브 LoRA 카드 클릭과 동일). 브릿지 스크립트 `static/forge_bridge.js`를 `base.html`에 주입.
  - locale 전 언어 + 런타임 DOM에서 **"ComfyUI" → "Forge Neo"** 라벨 치환 (willmiao 위키/repo URL은 링크·귀속 유지를 위해 변경 안 함).
  - **버그 수정**: 사용 팁 X 버튼(가상 스크롤 시 카드 null → 삭제 실패) 소스 한 줄 가드; 추가 메모 placeholder(비영어 locale에서 포커스 시 안 지워지던 것)를 `data-placeholder`+CSS `:empty::before` 진짜 placeholder로 전환.

이 패치들은 vendor 재clone 시 자동 재적용됩니다.

---

## 라이선스

본 확장(통합 레이어) 자체는 내부 사용. 단, 임베드한 LoRA Manager가 **GPL-3.0**이므로 재배포 시 GPL-3.0 조건을 따릅니다. (vendor 코드는 저장소에 포함되지 않으며 런타임에 clone됩니다.)
