# sam-extra (Forge SAM3 Extension)

SAM3 / SAM3.1 마스크 + 인페인트 확장. 네 가지 워크플로 제공:

1. **In-flight** — t2i/img2img 생성 직후 자동으로 SAM3 마스킹 → 인페인트 (ADetailer 스타일)
2. **Refine 패널** (v0.4.0+) — ⚠️ **실험 기능 (아직 제대로 작동하지 않음)** — 갤러리에서 이미지 골라 즉시 SAM3+인페인트+CN으로 재손질, 결과를 갤러리에 누적
3. **Anima Tile-Repair** (v0.8.0+) — ⚠️ **실험 기능 (아직 제대로 작동하지 않음)** — [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)의 Anima ControlNet-LLLite 추론을 가져와 임베드 (Apache-2.0)
4. **LoRA Manager** (v0.9.0+) — [willmiao/ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager)를 그대로 가져와 extra-networks 탭에 임베드 (GPL-3.0)

ControlNet 통합 (LLLite 인페인트 모델 자동 호환 처리), 옷 교체용 Target/Replacement 워크플로, 시드 고정, VRAM 절약 옵션, XYZ plot 다축 등 지원.

> 워크플로 3·4는 외부 프로젝트를 vendored하여 통합한 것입니다. 자세한 출처는 아래 [출처 / 크레딧](#출처--크레딧-credits) 참고.

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

## 별도 기능: Anima Safe PAG (v0.9.8+)

> SAM3 워크플로와 **완전히 분리된 독립 스크립트**입니다 (`scripts/anima_safe_pag.py`).
> PiD / Anima-Ref PoC처럼 `sam3ext`를 전혀 import하지 않고 Forge 코어 파일도 건드리지
> 않습니다. Anima 엔진이 로드된 경우에만 동작하고, 그 외 모델·오류 시 일반 생성으로
> 폴백하므로 켜 둬도 안전합니다.
>
> ⚠️ **실험 기능 — 런타임 검증 미완.** Forge Neo `neo` 브랜치 소스와 정적으로 일치하도록
> 작성했으나 실제 Anima 체크포인트로 end-to-end 확인이 1회 필요합니다. webui 콘솔의
> `[AnimaSafePAG]` 로그로 훅이 붙었는지 확인하세요.

[iljung1106/comfyui-anima-safe-pag](https://github.com/iljung1106/comfyui-anima-safe-pag)를
Forge Neo용으로 이식한 soft-PAG(Perturbed Attention Guidance)입니다. Anima/Cosmos/Predict2
계열 DiT의 후반 self-attention을 부드럽게 흐린 *약한 예측*을 만들고, CFG 결과를 그
예측에서 **멀어지는 방향**으로 보정해 구조 안정성·선 명료도·디테일을 개선합니다.

### 동작 원리 (Forge Neo, 코어 수정 없음)

Forge Neo 실제 샘플링 경로(`backend/sampling/sampling_function.py`)는 ComfyUI 노드가 쓰는
`sampler_calc_cond_batch_function`을 **호출하지 않으므로**(패처에 setter만 존재), 대신 실제로
호출되는 두 훅만 사용합니다:

1. **`model_function_wrapper`** — `apply_model` 배치에 **cond 행 하나를 복제해 붙여** 단일
   forward로 pag 예측까지 계산합니다. 별도 forward가 없어 커널 런치 오버헤드가 없고, 원본
   행은 그대로 반환하므로 CFG는 정상 진행됩니다. (PAG는 원리상 cond 예측을 한 번 더
   계산해야 하므로 활성 스텝에서 배치가 늘지만, 이를 같은 forward에 접어 최소화합니다.)
2. **`post_cfg_function`** — cond/uncond의 denoise 스케일 `c_out`을 실측 복원해 pag 예측을
   denoised(x0) 공간으로 **정확히**(eps/v/flow-matching 무관) 변환한 뒤
   `result = cfg + scale·(cond − pag)` + std 매칭 rescale을 적용합니다.

어텐션 perturbation은 `backend/nn/anima.py`의 모듈 전역 `scaled_dot_product_attention`을
감싸, 지정한 블록의 **pag 행에만** `lerp(정상, value, strength)`(value-only/identity 경로)를
적용합니다. value는 RoPE가 없어 rotary-safe하며, grouped-query로 head 수가 다르면 텐서를
훼손하지 않고 자동 스킵합니다.

### 파라미터

| 필드 | 기본값 | 설명 |
|---|---|---|
| Enable | off | 켜면 Anima 생성에만 적용 (다른 모델은 자동 스킵) |
| PAG scale | 4.0 | guidance 세기 |
| Perturbation strength | 0.75 | 정상 어텐션 ↔ value-only 블렌드 비율 |
| Block indices | 빈칸(=후반 절반 자동) | `18` / `14-27` / `14,16,18` (28블록은 >14 권장) |
| Start~End percent | 0.0~0.7 | 적용할 샘플링 구간 (나머지 스텝은 원가) |
| Rescale | 0.20 | 대비/채도 과다 억제 |

### APG (Adaptive Projected Guidance) — 같은 패널의 형제 기능 (v0.9.9+)

높은 CFG가 만드는 **과채도·번짐 성분만 골라 억제**해, guidance를 세게 밀어도 자연스럽게
유지합니다(RescaleCFG의 상위호환). **추가 forward 없이** CFG 합성 지점(post-CFG)에서 계산만
바꾸며, **Anima 외 모델에서도 동작**합니다. PAG와 독립이라 **같이 켜도 됩니다** — APG가
프롬프트/색이 안정된 베이스를 만들고 그 위에 PAG가 구조를 더합니다(우리 post-CFG는 `c_out`
실측 복원으로 베이스가 표준 CFG·APG·MaHiRo 무엇이든 그 위에 얹힘).

| 필드 | 기본값 | 설명 |
|---|---|---|
| Enable APG | off | 켜면 CFG 합성을 APG로 대체 |
| APG 켜지면 PAG rescale 자동 끄기 | on(토글) | 이중 크기보정 방지. 끄면 둘 다 적용 |
| eta *(Advanced)* | 0.0 | 평행(과채도) 성분 비중. **1.0 = 표준 CFG로 환원**, 0 = 최대 억제 |
| norm threshold *(Advanced)* | 15.0 | guidance L2 크기 상한(0=off) |
| momentum *(Advanced)* | 0.0 | 스텝 간 running-average(음수 권장, 0=off) |

guidance 세기는 메인 **CFG Scale** 슬라이더를 그대로 사용합니다. 세부값은 "APG Advanced"
아코디언에서 조절(기본은 쉽게, 필요 시 깊게).

> 조합 원칙: **Perturbation(PAG/SEG/SLG) 중 하나 + 크기보정(APG 또는 rescale) 중 하나**.
> APG를 켜면 PAG 내부 rescale은 자동으로 꺼집니다(위 토글로 해제 가능).

### XYZ Plot 비교

XYZ plot 축에 `[Anima PAG] …` 와 `[Anima APG] …` 항목이 추가됩니다. **`[Anima PAG] Enable`** 축을
`True, False`로 두면 **PAG ON/OFF 비교 그리드**를 바로 뽑을 수 있습니다(UI 체크박스
상태와 무관하게 축 값이 우선). 그 외 `Scale / Perturbation Strength / Block Indices /
Start·End Percent / Rescale` 축, 그리고 **`[Anima APG] Enable`**(ON/OFF) 및
`Eta / Norm Threshold / Momentum` 축도 제공됩니다. ON 셀은 PNG 메타데이터의
`Anima Safe PAG: …` / `Anima APG: …` 마커로 식별됩니다.

---

## 별도 기능: Detail Daemon (v0.9.10+)

> 또 하나의 **독립 스크립트**(`scripts/anima_detail_daemon.py`). muerrilla의
> [sd-webui-Detail-Daemon](https://github.com/muerrilla/sd-webui-detail-daemon)을
> 포크(스케줄/시그마 조정 math 재구현)해 **직관적 UX로 재설계**하고 우리 패널에
> 통합했습니다. SAM3·PAG/APG와 무관하며 Forge 코어도 안 건드립니다.

매 스텝 **제거하는 노이즈량을 줄여** 디테일·질감을 늘립니다(배경 뽀샤시↓). 추가 forward
없이 sampler sigma만 조정하므로 **모든 모델(Anima RF 포함)에서 동작**합니다.

`sigma *= 1 − schedule[step] · amount · (cfg_scale if couple else 1)` — 양수 amount는
sigma를 낮춰(=노이즈 덜 제거) 디테일↑, 음수는 매끈, 0/끄면 완전 무효.

### UX (쉽게 → 깊게)

- **메인**: `Enable` + **Preset**(Subtle/Medium/Strong/Custom) + **Detail amount** 슬라이더 하나.
- **Advanced 아코디언**: 원본 전체 곡선 파라미터(start/end/bias/exponent/offset/fade/smooth/
  multiplier) + **couple to CFG scale** 토글(원본 "both" 모드).

| 필드 | 기본값 | 설명 |
|---|---|---|
| Enable Detail Daemon | off | |
| Preset | Medium | Custom이면 아래 Amount 사용 (Subtle 0.05 / Medium 0.10 / Strong 0.25) |
| Detail amount | 0.10 | 음수=매끈, 양수=디테일↑ |
| start / end / bias *(Adv)* | 0.2 / 0.8 / 0.5 | 적용 구간과 피크 위치 |
| exponent / offsets / fade *(Adv)* | 1.0 / 0 / 0 | 곡률·구간밖 기본값·전체 감쇠 |
| smooth / multiplier *(Adv)* | on / 1.0 | 코사인 스무딩 · 스케줄 위 전역 강도 |
| couple to CFG scale *(Adv, 토글)* | on | 효과를 CFG scale에 비례(원본 both). 끄면 CFG 무관 |

XYZ: `[Detail Daemon] Enable/Amount/Start/End/Bias` 축 제공. PNG 메타 `Anima Detail Daemon: …`.

> 조합: Detail Daemon은 스케줄 계열이라 **PAG·APG·MaHiRo 무엇과도 겹칩니다**(서로 다른 지점).

---

## 출처 / 크레딧 (Credits)

이 확장은 아래 외부 프로젝트를 **그대로 가져와(vendored, shallow clone)** Forge에 통합합니다. 핵심 기능의 저작권은 각 원저자에게 있으며, 본 확장은 Forge 통합 레이어만 제공합니다. vendor 디렉터리는 저장소에 포함되지 않고 `install.py`가 첫 실행 시 자동으로 clone합니다.

| 기능 | 원본 프로젝트 | 라이선스 | vendor 위치 |
|---|---|---|---|
| **LoRA Manager** (워크플로 4) | [willmiao/ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager) | GPL-3.0 | `lora_manager_vendor/` |
| **Anima Tile-Repair** (워크플로 3) | [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) (`anima_minimal_inference*`) | Apache-2.0 | `anima_vendor/` |
| **Anima Safe PAG** (별도 기능) | [iljung1106/comfyui-anima-safe-pag](https://github.com/iljung1106/comfyui-anima-safe-pag) | 원 저장소 라이선스 참고 | 이식 (vendor 아님) |
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
