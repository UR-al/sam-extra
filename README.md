# sam-extra (Forge SAM3 Extension)

SAM3 / SAM3.1 마스크 + 인페인트 확장. 다섯 가지 워크플로 제공:

1. **In-flight** — t2i/img2img 생성 직후 자동으로 SAM3 마스킹 → 인페인트 (ADetailer 스타일)
2. **Refine 패널** (v0.4.0+) — ⚠️ **실험 기능 (아직 제대로 작동하지 않음)** — 갤러리에서 이미지 골라 즉시 SAM3+인페인트+CN으로 재손질, 결과를 갤러리에 누적
3. **Anima Tile-Repair** (v0.8.0+) — ⚠️ **실험 기능 (아직 제대로 작동하지 않음)** — [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)의 Anima ControlNet-LLLite 추론을 가져와 임베드 (Apache-2.0)
4. **LoRA Manager** (v0.9.0+) — [willmiao/ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager)를 그대로 가져와 extra-networks 탭에 임베드 (GPL-3.0)
5. **txt2img Workspaces** (v0.10.0+, v0.13.0 실제 탭 모드) — Live 셸 또는 실제 브라우저 탭에서 이름을 붙인 작업공간을 추가·삭제하고 프롬프트·생성 설정·스크립트·XYZ Plot 상태를 서로 독립적으로 유지

ControlNet 통합 (LLLite 인페인트 모델 자동 호환 처리), 옷 교체용 Target/Replacement 워크플로, 시드 고정, VRAM 절약 옵션, XYZ plot 다축 등 지원.

또한 SAM3와 **완전히 분리된 독립 기능**으로 **Anima Guidance Suite**(v0.11.0+, PAG/SEG/SLG ·
APG/CWM/SMC · Skimmed CFG · DCW · DAVE · CNS · Adaptive Guidance · Detail Daemon)를 제공합니다.
기능마다 구현 방식과 검증 수준이 다르므로, 사용 전에 반드시 아래 **[구현·검증 상태](#구현검증-상태)**와
**[상세 가이드](docs/GUIDANCE.md)**를 확인하세요.

그리고 **Anima VAE 2x** (v0.9.14+, 실험) — spacepxl 2x Wan-VAE 파인튜닝을 디코더로 써서
speckle↓·skin/hair 정리(`scripts/anima_vae_2x.py`). Qwen/Wan VAE latent 공유로 Anima에도 적용.

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

## 워크플로 5: txt2img Workspaces (Live 셸 / 실제 브라우저 탭) (v0.10.0+, v0.13.0 개선)

`Settings → SAM3 Workspaces`의 **작업공간 모드**로 `Live Workspace`(기본)와 `기본 Forge UI`
중 하나를 선택합니다(새로고침 후 적용). `기본 Forge UI`는 워크스페이스 없이 순정 Forge를 그대로
씁니다. 아래는 `Live Workspace` 모드 설명입니다.

현재 WebUI 포트(예: `7860` 또는 `8760`)에서 기본 3개(최대 20개)의 브라우저 화면을 하나의
**Live Workspaces** 안에 유지합니다. 접속 시 확장 전용 경량 셸인 `/sam3-live`로 자동 이동하지만
포트와 Forge 서버는 그대로입니다(리다이렉트 여부는 서버가 모드 설정을 읽어 결정). 각 Workspace는
별도 same-origin iframe의 Gradio 문서이므로 버튼을 누를 때 값을 갈아 끼우지 않고, 이미 설정된
화면의 표시만 전환합니다. 숨겨진 Workspace는 `inert` 상태라 클릭·단축키 입력을 받지 않고
백그라운드 재-마운트 감시(observer/폴링)를 일시정지해 전환 부담을 줄이며, **Generate는 현재 보고
있는 Workspace 하나에서만 실행**됩니다.

현재 선택한 Workspace를 먼저 준비해 바로 사용할 수 있게 하고 나머지는 한 번에 하나씩
백그라운드에서 로드합니다. `Live Workspaces ＋`를 누르면 현재 설정을 복사한 새 Workspace가
생기며, `⋯` 메뉴에서 이름 변경·삭제·JSON 내보내기/가져오기를 할 수 있습니다. 탭
더블클릭으로도 이름을 바꿀 수 있습니다. 헤더의 작은 상태 문구에는 `저장 대기…`,
`자동 저장됨`, 충돌/오류가 현재 Workspace 기준으로 표시됩니다.

iframe 전환이 무겁게 느껴지면 헤더의 **실제 탭으로 열기**를 누를 수 있습니다. 현재 화면들을 먼저
저장한 뒤 Live 셸 탭은 활성 Workspace로 바뀌고 나머지 Workspace는 같은 포트의 실제 브라우저
탭으로 열립니다. 이 모드에서는 iframe 표시 전환이 전혀 없어서 원래 같은 WebUI 주소를 탭 3개로
열어 사용하던 것과 같은 방식으로 전환합니다. 각 탭 제목과 상단 바에는 Workspace 이름이 표시되고,
자동 저장 상태도 개별 탭에 나타납니다. 여러 탭 열기가 차단되면 주소창의 팝업 허용 아이콘에서
현재 WebUI 주소를 허용한 뒤 버튼을 다시 누르면 됩니다. 상단의 `Live 관리로 돌아가기`는 해당 탭을
다시 `/sam3-live` 관리 셸로 바꿉니다.

각 화면은 위쪽 Prompt/Negative/Generate와 아래쪽 **Parameters / Scripts / Gallery** 3열로
정리됩니다. 좁은 화면에서는 한 열로 접힙니다. 순정 Forge로 돌아가려면 `Settings → SAM3 Workspaces`
에서 모드를 `기본 Forge UI`로 바꾸고 새로고침하세요(인-페이지 툴바 방식은 폐기됨). 각 iframe 안의 Forge 상단
**txt2img / img2img / PNG Info / Settings / Extensions** 탭은 그대로 유지되며, 확장 전용
3열 배치는 txt2img에만 적용됩니다. 생성 대기·큐·진행률·중간 미리보기는 Forge 기본 UI를
그대로 사용합니다.

기존 v0.10.0의 Workspace 1 / 2 / 3과 저장 데이터는 자동으로 유지됩니다. 각 작업공간에는 다음
상태가 독립적으로 보존됩니다.

- positive/negative prompt와 seed, steps, sampler, scheduler, 크기 등 txt2img 생성 설정
- 선택한 Script와 해당 Script의 옵션
- X/Y/Z Plot의 축 종류, 값, 관련 옵션
- `elem_id`가 없는 타 확장 컨트롤도 라벨과 상위 영역 조합이 유일하면 안전하게 포함
- 현재 세션의 txt2img **마지막 생성 결과**와 이미지별 infotext/generation info

각 iframe은 자기 고정 slot에 입력을 바꿀 때마다 **브라우저 로컬 저장소에 자동 저장**합니다.
Workspace 버튼 전환에는 값 복원이 전혀 없고, WebUI/페이지가 새로 시작할 때만 Forge 기본 UI가
준비된 뒤 해당 slot 저장본을 한 번 적용합니다. 즉 Forge의 `Save UI defaults`가 세 화면의 공통
기준이고, 각 Workspace 저장본은 그 위에 얹는 개별 프로필입니다.

갤러리는 큰 이미지 바이트를 복제하지 않고 Forge/Gradio의 파일 참조와 infotext만 브라우저
IndexedDB에 따로 저장합니다. **Generate를 누르면 그 Workspace의 이전 갤러리를 먼저 비우고 이번
생성 결과만 남깁니다.** 새 Workspace는 현재 설정만 복제하고 갤러리는 빈 상태로 시작하므로 공통
설정을 다시 입력하지 않고, Workspace를 오가며 서로 다른 LoRA·프롬프트의 마지막 결과를 비교할
수 있습니다. Live 셸은 WebUI 서버가 내려갔다가 돌아온 것을 감지하면 한 번 자동 새로고침하여
세 iframe의 오래된 Gradio 세션을 버리고 갤러리를 비웁니다. Forge Neo 기본 파일은 수정하지
않으며 이 확장의 프런트엔드에서만 동작합니다.

### 저장 범위와 주의사항

- txt2img 갤러리는 Workspace마다 **마지막 한 번의 생성 결과**만 현재 브라우저 세션 동안 보존합니다.
  페이지 새로고침이나 WebUI 재시작 시 모든 Workspace 갤러리를 비웁니다. 생성 한 번에 500장이
  넘는 경우 최신 500장까지만 남깁니다.
- 이미지 파일 자체를 이동·복제하지 않으므로 원본 출력/Gradio 임시 파일을 삭제하면 해당 썸네일은
  다시 열리지 않을 수 있습니다.
- 갤러리 기록은 로컬 전용이며 Workspace JSON **내보내기/가져오기에는 포함하지 않습니다**.
- 이미지·파일 입력과 img2img 갤러리는 저장하거나 복원하지 않습니다.
- 라벨과 상위 영역까지 같은 익명 컨트롤이 둘 이상이면 잘못된 복원을 막기 위해 해당 충돌
  컨트롤은 저장에서 제외합니다.
- checkpoint/VAE 등 Forge 전역 **Quicksettings**는 작업공간에 포함하지 않습니다. 모델 재로딩을
  유발하는 전역 상태가 작업공간 전환과 섞이지 않도록 하기 위함입니다. iframe이 여러 개여도 Forge
  서버·GPU·생성 큐와 현재 로드된 checkpoint는 하나이므로 이 항목들은 실제 브라우저 탭 여러 개처럼
  서로 공유됩니다.
- 데이터는 서버가 아닌 현재 브라우저 프로필의 로컬 저장소에 평문으로 남습니다. 민감한
  프롬프트를 공용 PC에 저장하지 마세요.
- 브라우저 저장소는 **동일 출처(same origin: 프로토콜 + 호스트 + 포트)** 단위입니다. 예를 들어
  `127.0.0.1:8760`과 `localhost:8760`, 또는 서로 다른 포트는 별도 저장소를 사용합니다.
- 같은 주소의 다른 탭이 동일한 Workspace를 먼저 수정하면 덮어쓰지 않고 충돌 경고를 표시합니다.
  이 경우 기존 탭을 닫거나 새로고침한 뒤 계속하세요.
- 다른 포트·호스트·브라우저 프로필로 옮길 때는 작업공간 메뉴의 **내보내기/가져오기**를
  사용하세요.
- Workspace 기능이 포함된 확장 업데이트 후에는 경량 셸의 Python 경로와 새 프런트엔드 자산을
  다시 등록해야 하므로 브라우저 새로고침만 하지 말고 **WebUI 프로세스를 한 번 완전히
  재시작**하세요. 순정 Forge로 돌아가려면 `Settings → SAM3 Workspaces`에서 모드를
  `기본 Forge UI`로 바꾸고 새로고침하면 됩니다.

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

### 구현·검증 상태

| 기능 | 이 확장에서 하는 일 | 현재 판정 / 제한 |
|---|---|---|
| **PAG** | appended weak row를 value-only attention으로 strength 보간 | 실제 Anima E2E 검증. Legacy Soft 토글 제공 |
| **SEG** | 실제 Anima T/H/W 중 H/W query에 Gaussian blur·보간 | 실제 Anima E2E 검증. Legacy uniform-value 근사 제공 |
| **SLG** | weak row의 선택 block을 no-op으로 복원 | 형상·순서 연결, 전용 화질 E2E 필요 |
| **APG** | post-CFG guidance를 cond 평행/직교 성분으로 투영 | 중립값 단위 검증. **CFG>1 권장**, reference와 픽셀 동일하지 않음 |
| **CWM / SMC** | Haar 대역별 CFG 배율 / step 간 unit-L2 switching control | 수학·중립값 검증, 실제 CWM 실행 확인 |
| **DCW** | live x_t와 x0의 wavelet 차이를 post-CFG 마지막에 보정 | 4D/5D·홀수 해상도·중립값 검증, 실제 실행 확인 |
| **DAVE** | Anima block 출력의 token/spatial DC 성분 감쇠 | 실제 block hit 확인, 다양성/권장 block은 추가 A/B 필요 |
| **CNS-inspired** | 기존 seeded/Brownian noise를 live x_t 에너지로 재색칠 | Euler a noise call 확인. deterministic sampler에서는 inert |
| **Adaptive Guidance** | combined batch의 후반 uncond row 생략 | low-VRAM 분리 호출에서는 생략/속도 이득 없음 |
| **Skimmed CFG** | 과포화를 만드는 성분만 낮은 CFG로 되돌림(anti-burn) | 상단 수식과 tensor 단위 일치 검증. **CFG>1 전용**, pre-CFG가 아닌 post-CFG 재구성 |
| **Detail Daemon** | sigma schedule로 디테일 강도 조절 | 별도 opt-in 기능 |

### 오케스트레이터 순서

`ADG/PAG 배치 → DAVE → attention PAG/SEG → Skimmed CFG → CFG base(SMC → APG → CWM)
→ PAG/SEG/SLG delta → DCW → CNS sampler noise`

- CFG base의 **SMC·APG·CWM은 독립 토글**입니다. 셋 다 끄면 MaHiRo/custom CFG 결과를
  그대로 유지하고, 켜진 것들은 항상 `SMC → APG → CWM` 순서로 적용됩니다.
- CFG base가 켜지면 incoming에서 `w_eff`를 복원해 Forge의 `edit_strength`를 보존하고, 비선형 fit
  오차가 크면 경고합니다.
- Skimmed CFG는 별도 스크립트(별도 아코디언)이며 CFG base보다 **먼저** 실행되고, skim 결과를
  Forge의 예측 tensor에 다시 써서 이후 SMC/APG/CWM·PAG delta·DCW가 모두 그 위에서
  동작합니다. ComfyUI에서 pre-CFG 노드를 물렸을 때와 같은 조합이 가능합니다.
- `Legacy CFG base mode` 아코디언의 라디오와 `Experimental stack`은 구버전 호환용이며
  위 토글과 OR로 합쳐집니다.
- CWM `alpha high > +0.15`는 Anima 16채널 latent에서 캐릭터 분리를 만들 수 있어 UI 경고가 뜹니다.
- UI와 XYZ의 **Attn Scale**은 같은 값이며 attention 점수가 아니라
  `scale × (cond − weak)` 보정 배율입니다.
- PAG/SEG 공통 `Perturbation strength` 기본은 `0.75`, `1.0`이면 전체 perturbation입니다.
  block `18`, heads 빈칸(전체), Start/End `0.0/0.7`, Rescale `0.20`,
  Rescale mode `full`이 Anima Safe PAG 시작값입니다.
- CNS는 ancestral/SDE sampler에서만 의미가 있습니다. TeaCache는 포함하지 않습니다.
- PAG/SEG 자체 A/B는 `Rescale=0`, SLG/APG/ADG off로 원인을 분리하세요.
- DCW·DAVE·CNS는 Guidance 본문에 펼쳐 표시하며, APG/Adaptive의 고급값만 접힌
  세부 영역으로 둡니다.

확장 목록 아래 **Anima Reference-Latent PoC (debug / 안전)** 패널의
`Log Guidance verification summary`를 켜면 attention hit/raw delta, CFG `w_eff`/fit,
DCW eval, DAVE block hit, CNS noise call, Adaptive 실제 생략 여부가 출력됩니다.

- 구현: `scripts/anima_safe_pag.py`, `sam3ext/guidance/`, `scripts/anima_detail_daemon.py`
- 테스트: `tests/test_anima_attention_patch.py`, `tests/test_anima_safe_pag.py`,
  `tests/test_guidance_suite.py`
- 상세 파라미터·처리 순서·XYZ·검증 로그·크레딧:
  **[docs/GUIDANCE.md](docs/GUIDANCE.md)**

---

## 출처 / 크레딧 (Credits)

이 확장은 아래 외부 프로젝트를 **그대로 가져와(vendored, shallow clone)** Forge에 통합합니다. 핵심 기능의 저작권은 각 원저자에게 있으며, 본 확장은 Forge 통합 레이어만 제공합니다. vendor 디렉터리는 저장소에 포함되지 않고 `install.py`가 첫 실행 시 자동으로 clone합니다.

| 기능 | 원본 프로젝트 | 라이선스 | vendor 위치 |
|---|---|---|---|
| **LoRA Manager** (워크플로 4) | [willmiao/ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager) | GPL-3.0 | `lora_manager_vendor/` |
| **Anima Tile-Repair** (워크플로 3) | [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) (`anima_minimal_inference*`) | Apache-2.0 | `anima_vendor/` |
| **Anima Safe PAG** (별도 기능) | [iljung1106/comfyui-anima-safe-pag](https://github.com/iljung1106/comfyui-anima-safe-pag) | 원 저장소 라이선스 참고 | 이식 (vendor 아님) |
| **DCW / CWM / SMC** | [namemechan/ComfyUI-DCW](https://github.com/namemechan/ComfyUI-DCW) | GPL-3.0 | 수식 기반 Forge 재작성 (vendor 아님) |
| **DAVE** | [daheekwon/DAVE](https://github.com/daheekwon/DAVE) · [sorryhyun/ComfyUI-Anima-DAVE](https://github.com/sorryhyun/ComfyUI-Anima-DAVE) | MIT | Forge block 재구현 (vendor 아님) |
| **CNS-inspired Wavelet Noise** | [namemechan/comfyui-cns_sampler_patch](https://github.com/namemechan/comfyui-cns_sampler_patch) | GPL-3.0 | sampler-noise 재작성 (vendor 아님) |
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
