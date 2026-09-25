# sam-extra (Forge SAM3 Extension)

SAM3 / SAM3.1 마스크 + 인페인트 확장. 일곱 가지 워크플로 제공:

1. **In-flight** — t2i/img2img 생성 직후 자동으로 SAM3 마스킹 → 인페인트 (ADetailer 스타일)
2. **Refine 패널** (v0.4.0+) — ⚠️ **실험 기능 (아직 제대로 작동하지 않음, [진단 체크리스트](docs/EXPERIMENTAL_STATUS.md))** — 갤러리에서 이미지 골라 즉시 SAM3+인페인트+CN으로 재손질, 결과를 갤러리에 누적
3. **Anima Tile-Repair** (v0.8.0+) — ⚠️ **실험 기능 (아직 제대로 작동하지 않음, [진단 체크리스트](docs/EXPERIMENTAL_STATUS.md))** — [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)의 Anima ControlNet-LLLite 추론을 가져와 임베드 (Apache-2.0)
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

## 목차

- 시작: [설치](#설치) · [SAM3 체크포인트](#sam3-체크포인트) · [의존성](#의존성) · [문서](#문서) · [라이선스](#라이선스)
- 워크플로: [1 In-flight](#워크플로-1-in-flight-자동-detailer) ·
  [2 Refine](#워크플로-2-refine-패널-post-generation) ·
  [3 Tile-Repair / PiD](#워크플로-3-복원업스케일-anima-tile-repair--pid-v080) ·
  [4 LoRA Manager](#워크플로-4-lora-manager-통합-v090-v091에서-정상-작동) ·
  [5 Notebook](#워크플로-5-txt2img-notebook) ·
  [6 Character Reference](#워크플로-6-anima-character-reference--restyler) ·
  [7 Anima 3.8B](#워크플로-7-anima-38b--qwen35--semantic-connector-v2-편입)
- SAM3 공통: [ControlNet 통합](#controlnet-통합) · [VRAM 절약](#vram-절약) · [마스크 후처리](#마스크-후처리) ·
  [진행률 / 검증](#진행률--검증) · [XYZ Plot 축](#xyz-plot-축) · [Settings 저장](#settings-저장)
- 별도 기능: [ANIMA LoRA 블록 호환 변환](#별도-기능-anima-lora-블록-호환-변환) ·
  [DoRA 추론 방식](#별도-기능-dora-추론-방식) · [Anima Guidance Suite](#별도-기능-anima-guidance-suite) ·
  [txt2img 화면 정리](#txt2img-화면-정리-) · [TIPO 프롬프트 확장](#별도-기능-tipo-프롬프트-확장-)
- 참고: [출처 / 크레딧](#출처--크레딧-credits)

---

## 설치

```bash
cd <sd-webui-forge-neo>/extensions
git clone https://github.com/UR-al/sam-extra.git forge_sam3_extension
```

**폴더명은 반드시 `forge_sam3_extension`** 으로 둡니다. Forge 는 확장 폴더명을 확장 이름으로 쓰는데
`metadata.ini` 의 콜백 순서 지정(`[callbacks/forge_sam3_extension/…]`)이 이 이름에 묶여 있어, 다른 이름(예: 기본
clone 폴더 `sam-extra`)이면 Forge 가 순서 지정을 버립니다. 이 경우 시작 로그에
`[-] SAM3: extension folder is '…', not 'forge_sam3_extension'` 경고가 나옵니다 — 폴더 이름을 바꾸고 재시작하세요.

첫 실행 때 `install.py` 가 `requirements.txt` 를 읽어 **Forge venv 에 없는 패키지만** 설치합니다(`sam3` 등).
이미 깔린 패키지는 버전이 달라도 올리거나 내리지 않고 콘솔에 알리기만 하며, `torch`/`torchvision` 은 목록에 없고
설치하지도 않습니다(Forge 가 CUDA 빌드에 맞춰 관리). `pip install -r requirements.txt` 를 직접 실행해도 torch 는
바뀌지 않습니다. 자동 설치를 끄려면 `webui-user.bat` 의 `COMMANDLINE_ARGS` 에 `--sam3-no-auto-install` 을
넣거나 환경 변수 `SAM3_NO_AUTO_INSTALL=1` 을 설정합니다 — 그러면 빠진 패키지와 설치 명령만 콘솔에 출력합니다.
설치·clone 진단은 모두 콘솔(stdout)에 `[forge_sam3_extension]` 머리말로 나옵니다.

webui 재시작 후 t2i/img2img 패널에 **"SAM3 Mask"** 아코디언이 보이고, txt2img 갤러리 아래 접힌 **"선택 이미지 도구"** 칸을
펼쳤을 때 **SAM3 Refine · Tile-Repair · 캐릭터 레퍼런스** 탭이 보이면 정상입니다(Tile-Repair 탭은 `anima_vendor/` 가
있을 때만 나옵니다). 화면 배치는 아래 [txt2img 화면 정리](#txt2img-화면-정리-) 참고.

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

체크포인트 하나도 없으면 드롭다운에 기본값 `sam3.pt` 만 보이고, 실행 때 로컬에 `sam3.pt` 가 없으면 Hugging Face의
`facebook/sam3`에서 `sam3.pt`(3.45 GB)를 자동 다운로드해 씁니다(HF 캐시에 한 번만 받음, 콘솔에 한 줄 안내).
완전 오프라인 사용 시 `--sam3-no-huggingface` 옵션으로 자동 다운로드를 끄면, 로컬 파일이 없을 때 넣어야 할 위치를
알려 주는 `FileNotFoundError` 로 멈춥니다. 다른 이름의 체크포인트(`sam3.safetensors` 등)를 골랐는데 파일이 없으면
다운로드하지 않고 오류를 냅니다.

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
> 진단 방법은 [docs/EXPERIMENTAL_STATUS.md](docs/EXPERIMENTAL_STATUS.md) 참고.

t2i 끝난 후 갤러리에서 이미지 골라 즉시 재손질. 결과는 갤러리에 누적 삽입.

```
t2i Generate → 갤러리 N장
  → 손볼 이미지 클릭
  → 갤러리 아래 "선택 이미지 도구" → SAM3 Refine 탭에서 Target/Replacement 입력
  → ▶ Refine
  → 선택 이미지 옆(또는 끝)에 결과 추가
  → 새 이미지 또 클릭해서 chain refine
```

Refine 은 Forge Generate 와 같은 대기열(`queue_lock`)에서 돌므로 txt2img 생성 중에 누르면 그 생성이 끝난 뒤
시작합니다. 도는 동안 ▶ Refine 자리에 **⏹ Stop** 이 나오고, 이 Stop 은 Refine 작업에만 걸립니다.

**Refine 패널 구성** (위에서 아래 순서):

| 필드 | 역할 |
|---|---|
| Manual Mask (접힘, 선택) | 캔버스에 대충 칠한 마스크로 SAM3 결과를 좁힘 — 아래 [Manual Mask](#manual-mask) 참고 |
| Target (마스크/치환할 대상) | SAM3가 마스킹할 토큰 + 메인 prompt에서 제거할 토큰. Manual Mask 를 칠했으면 비워도 됨 |
| **Exclude (보호할 영역)** | **두 번째 SAM3 detect로 마스킹한 영역을 Target 마스크에서 빼냄. 예: Target=`clothes`, Exclude=`face, eyes, hand` → 옷만 인페인트, 얼굴·눈·손은 원본 유지** |
| Replacement (대체할 단어) | 마스크에 그릴 내용 + Target 자리에 한 번만 삽입 |
| Negative | 옵션. (정리된) 상속 negative 뒤에 붙음 |
| Inherit main t2i prompt | (기본 ON) LoRA/스타일 유지하며 Target만 segment 단위로 제거 |
| Inherit main t2i negative | (기본 **OFF**) 켜면 메인 negative 도 가져오되 같은 규칙으로 Target 을 뺌. 기본이 꺼진 이유: t2i negative 의 단어(예: `nude`)가 Refine 요청과 정면으로 싸우기 때문 |
| Threshold / Mask Dilation / Mask Blur / Mask Processing | 검출 임계값·확장·가장자리 흐림, `Individual`(`/` 그룹마다 따로) / `Combined`(한 번에, 기본) |
| Convex Hull / Outline expand (edge-aware px) | 마스크 후처리 — 아래 [마스크 후처리](#마스크-후처리) 참고 |
| Unload SAM3 from VRAM after detection | (기본 OFF — In-flight 쪽은 ON) 인페인트 동안 SAM3 VRAM 해제 — CPU RAM 에 약 3.4 GB 로 남겨 두고 다음 검출은 GPU 로 옮기기만 함 (VRAM 16 GB 이하 권장) |
| Seed (-1 = random) · 🎲 · 🎯 Pull from selected | 🎲 는 -1, 🎯 는 선택한 갤러리 이미지의 시드를 가져옴. 갤러리가 바뀌면(새 생성·Refine 결과) 마지막 이미지의 시드를 자동으로 채움 |
| Resize mode | Just Resize / Crop and Resize / Resize and Fill (img2img 와 같은 뜻) |
| Mask mode: inpaint not-masked (invert) | 마스크 바깥을 인페인트 |
| Denoising / Masked Content / Inpaint Only Masked / Padding | i2i 파라미터. Masked Content 기본은 `latent noise`(In-flight 는 `original`) |
| Steps / CFG / Sampler / Scheduler / SAM3 Checkpoint | 샘플링 파라미터 — 이 패스에는 패널 값이 항상 쓰임 |
| SD Model Override | `Use current`(기본) 이 아니면 이 패스만 그 체크포인트로(Forge `override_settings`, 끝난 뒤 복원은 Forge 설정을 따름) |
| ControlNet 아코디언 | CN 유닛 옵션 (모델/모듈/weight 등). 화면의 CN 유닛은 끄고 이 설정만 씀 |
| 🎭 Regional Swap preset | RegionalSampler 가이드 기본값을 한 번에 채움 — [docs/REGIONAL_STYLE_SWAP.md](docs/REGIONAL_STYLE_SWAP.md) |
| Insert result: After selected / At end | 결과 삽입 위치 |
| ▶ Refine / ⏹ Stop | 실행 / 이 Refine 만 중단 |

- **다른 always-on 확장**: Refine 패스는 생성 버튼을 거치지 않으므로, SAM3 를 뺀 다른 always-on 스크립트(img2img
  쪽 목록)가 **txt2img 화면의 현재 값이 아니라 각 컴포넌트의 기본값**으로 돕니다(ControlNet 유닛은 위처럼 끔). 샘플링
  파라미터만 패널 값으로 확실히 덮어씁니다.

### Manual Mask

`Manual Mask (optional — narrow SAM3 to your scribble)` 아코디언(Forge 캔버스가 있는 빌드에서만 보임):

1. **📋 Load selected to canvas** 로 선택한 갤러리 이미지를 캔버스에 올립니다.
2. 고칠 영역을 대충 칠합니다.
3. ▶ Refine — SAM3 검출 결과(hull·outline·dilation 까지 적용한 것)와 칠한 영역의 **교집합**을 씁니다. 교집합이
   비었으면(SAM3 가 그 안에서 못 찾음, 또는 Target 이 비어 있음) 칠한 영역을 그대로 마스크로 씁니다. 그 뒤에
   Exclude 를 뺍니다. 이 경우 `/` 그룹은 하나로 합쳐집니다.

캔버스에 이미지가 올라가 있으면(칠하지 않았어도) 갤러리에서 새로 고른 이미지 대신 **캔버스 이미지로** 돕니다. 다른
이미지를 고치려면 다시 📋 를 누르거나 캔버스를 비우세요.

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

### Anima LLLite 자동 호환

Anima ControlNet-LLLite 는 원본(kohya sd-scripts · ComfyUI-Anima-LLLite)처럼 **사용자가 준 제어 이미지**를 그대로 받는다. 채널 수는 모델 파일의 safetensors 헤더(`lllite.cond_in_channels`, 없으면 `lllite_conditioning1.conv1` 입력 채널, 그것도 없으면 원본 기본값 3)로 읽고, 헤더를 못 읽으면 파일 이름으로 추정한다. 3채널이 표준(lineart·canny·depth 등 모든 제어 종류)이고 4채널이 인페인트다. SAM3 CN 유닛의 제어 이미지는 인페인트 입력 이미지라, lineart·canny·depth LLLite 에서는 고른 preprocessor 가 그 맵을 만든다.

- **Tile & Repair (3채널, 헤더 `modelspec.title` 에 `tile` — v1 `anima_tiled_lllite_v1`, v2 `anima_tile_multitask_v1`. 제목이 없으면 파일 이름 `animaTileRepair_*`)**: preprocessor 를 **항상 `None`** 으로 override. 기본값 `inpaint_only` 는 고칠 영역을 −1 로 비워(LLLite 입력으로는 −3) 복구할 내용을 지운다.
- **그 밖의 3채널 Anima LLLite (lineart·canny·depth 등)**: 고른 preprocessor 를 그대로 쓴다. `inpaint_*` 만 `None` 으로 override — 원본 3채널은 마스크를 쓰지 않는데 `inpaint_*` 는 마스크 영역을 제어 이미지에서 비운다.
- **4채널 인페인트 (`anima-lllite-inpainting-*`, RGB+mask)**: `inpaint_only` 같은 mask-stripping preprocessor와 조합하면 어설션 실패 → `inpaint_*` 를 `None` 으로 override.
- Anima LLLite 가 아닌 모델(SDXL `kohya_controllllite_*` 포함)은 건드리지 않는다.

override 할 때마다 stderr에 한 줄 로그. 사용자가 따로 신경 안 써도 됨. (Forge 내장 LLLite 는 3채널 cond 에 마스크를 곱해 마스크 밖 문맥을 검게 만든다 — 내장 코드의 호스트 차이라 그대로 둔다.)

### ⚠️ 옷 교체가 안 바뀌어 보일 때

`anima-lllite-inpainting-v2`는 *"주변 컨텍스트와 자연스럽게 섞기"* 가 목적이라 옷 교체를 적극 방해함. 옷을 **확실히 바꾸려면**:

- **CN 끄기** — 가장 효과적
- 또는 CN Weight 1.0 → 0.4~0.6
- 또는 `depth_*` CN으로 교체 (신체 실루엣만 유지, 옷은 자유)

---

## VRAM 절약

SAM3 체크포인트는 ~3.5 GB. 한번 로드되면 번들 하나가 캐시에 잡혀서(체크포인트·장치를 바꾸면 옛 번들을 먼저 해제하고 새로 빌드) 인페인트 동안 VRAM 점유. VRAM 16 GB 이하 GPU 이거나 Forge의 `reserve-vram` 경고가 뜨면:

1. **"Unload SAM3 from VRAM after detection"** 체크 (SAM3 패널 + Refine 패널 양쪽 모두 옵션 있음 — 기본값은 SAM3 패널(In-flight) 켜짐, Refine 패널 꺼짐). 검출(~2초) 끝나면 번들을 CPU RAM 으로 옮기고 `cuda.empty_cache()` → 인페인트 사이클이 풀 VRAM 활용. 다음 검출은 재빌드 없이 GPU 로 다시 옮기기만 함(추정 0.4~1.6초). 대신 RAM 에 약 3.4 GB 가 남으며, SAM3 체크포인트·장치 변경이나 Reload UI 때 해제됩니다. RAM 이 빠듯하면 Settings → **SAM Extra SAM3** → "'Unload after' 뒤 SAM3 모델을 CPU RAM 에 보관 (약 3.4 GB)"(`sam3_unload_keep_in_ram`, 기본 켬)을 끄세요 — 예전처럼 완전히 해제하고 다음 검출 때 새로 빌드합니다(이미지당 2.5~5 초 추가 추정). 끄고 Apply 하면 보관 중인 번들도 곧바로 해제되며, 결과 이미지는 같습니다.
2. webui 실행 인자에 `--reserve-vram 2` 추가 — 모델 매니저가 헤드룸 2 GB 확보.

둘 같이 쓰면 가장 안정적.

---

## 마스크 후처리

머리카락·털·strand 등 가는 부분이 SAM3에 부분 누락되는 경우용:

| 옵션 | 효과 | 추천 시나리오 |
|---|---|---|
| **Mask Dilation (px)** (최대 256) | 마스크를 N 픽셀 바깥쪽 확장 | 강한 가장자리 (옷, 물체) |
| **Convex Hull** | 검출 영역을 최소 볼록 다각형으로 감쌈 (컴포넌트별 적용) | 머리·털 strand 사이 공간까지 자동 포함 |
| **Outline expand (edge-aware px)** (최대 64) | 이미지의 Canny 가장자리를 벽으로 삼아 N 픽셀까지만 넓힘 — 옷 윤곽선에서 멈춤 | 옷 외곽선 잔여물 |
| **Mask Blur** | 가장자리 부드럽게 | 인페인트 합성 자연스러움 |

적용 순서(`sam3ext/core.py`): 검출 마스크(`/` 그룹마다) → Convex Hull → Outline expand → Dilation →
Manual Mask 교집합(Refine, 칠했을 때) → Exclude 영역 차감. Mask Blur 는 이 마스크를 넘겨받은 인페인트(img2img)
단계에서 Forge 가 적용합니다.

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

`Enable, Checkpoint, Mode, Mask Mode, Device, Detect Prompt, Exclude Prompt, Inpaint Prompt, Negative Prompt, Prompt S/R (2종), Threshold, Mask Dilation, Mask Hull, Mask Outline Expand, Mask Blur, Denoising, Inpainting Fill, CFG, Steps, Inpaint Only Masked, Padding, Inpaint Width/Height, Sampler, Scheduler, Seed, Noise Multiplier, Restore Face, Unload After, CN Enable, CN Override, CN Model, CN Module, CN Weight, CN Guidance Start/End` (모두 `[SAM3]` 접두어)

`[SAM3] Checkpoint`·`[SAM3] Device` 축은 값이 바뀔 때마다 SAM3 를 다시 빌드하므로(3~5 초) cost 를 줘서 바깥 루프로
돌립니다 — 칸 결과는 같고 격자를 도는 순서만 바뀝니다.

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

`requirements.txt` — Forge launch 시 `install.py` 가 빠진 것만 설치(위 [설치](#설치) 참고, `--sam3-no-auto-install`
로 끔). SAM3 본체는 `sam3` PyPI 패키지 필요. torch 는 Forge 가 관리하므로 목록에 없습니다.

개발·테스트: `pip install -r requirements-dev.txt`(CPU torch 는 따로:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`) 뒤 확장 폴더에서
`python -m unittest discover -s tests -p "test_*.py"`. CI(`.github/workflows/ci.yml`)도 Python 3.13 에서 같은 명령을 씁니다.

ControlNet 통합은 `sd_forge_controlnet` 익스텐션에 lazy import 의존. 없으면 해당 UI/로직만 비활성화.

---

## 워크플로 3: 복원/업스케일 (Anima Tile-Repair + PiD) (v0.8.0+)

> ⚠️ **실험 기능 — 런타임 검증 미완.** 정적 버그는 정리했으나 실제 모델로 end-to-end 확인이 필요합니다.
> 진단 방법은 [docs/EXPERIMENTAL_STATUS.md](docs/EXPERIMENTAL_STATUS.md) 참고.

갤러리 선택 이미지를 복원/업스케일하는 후처리 패널(갤러리 아래 **선택 이미지 도구 → Tile-Repair** 탭). **복원 모드** 두 가지를 옵션으로 제공합니다:

- **Anima Tile-Repair** (기본) — vendored kohya sd-scripts의 Anima ControlNet-LLLite tile 복원. **Qwen3 0.6B TE + Qwen-Image VAE 필수**(패널 드롭다운이 폴더의 Qwen3 0.6B 파일(`qwen_3_06b_*` 등)을 자동 선택 — Qwen3-VL·Qwen3.5·adapter·LLM-CLIP 파일은 고르지 않음; 없으면 `Use Forge current` 로 두고 실행 시 명확한 안내). **마스크는 쓰지 않고 이미지 전체를 한 번에 복원합니다**(이름의 'Tile' 은 LLLite 모델 이름에서 온 것이고 타일 분할은 하지 않음).
- **PiD Upscale** (v0.9.6+) — Forge Neo **네이티브** [NVIDIA PiD](https://huggingface.co/nvidia/PiD)(Pixel Diffusion Decoder) 초해상 복원. 파일명에 `PiD`가 포함된 체크포인트를 `models/Stable-diffusion/`에 넣으면 자동 활성화(`backend/loader.py`). img2img로 동작(`denoising_strength`→degrade σ로 재해석), **마스크 미사용·전체 이미지 업스케일**. vendor 불필요 — Forge가 모든 처리를 함.

아래 사용 흐름은 Anima Tile-Repair 기준입니다.

### 사용 흐름

```
t2i Generate → 갤러리 N장
  → 디테일/노이즈 복원하고 싶은 이미지 클릭
  → 선택 이미지 도구 → Tile-Repair 탭에서 LLLite 모델 선택
  → ▶ Anima Tile-Repair
  → 결과가 선택 이미지 옆에 삽입됨
```

Refine 처럼 Forge Generate 와 같은 대기열에서 돌고, 도는 동안 나오는 ⏹ Stop 은 이 작업에만 걸립니다.

### 의존성 자동 설치

확장 첫 로드 시 `install.py`가 `kohya-ss/sd-scripts` repo를 `extensions/forge_sam3_extension/anima_vendor/` 로 shallow clone합니다 (약 20MB, `main` 브랜치 — 커밋 고정은 아직 없음). `git`이 PATH에 있어야 합니다. 실패하면 Tile-Repair 탭만 빠지고 나머지 SAM3 기능은 정상 작동.

### 필요 모델 (사용자 디스크 위치 기준)

| 종류 | 권장 경로 | 비고 |
|---|---|---|
| Anima DiT | `models/Stable-diffusion/ANIMA_*.safetensors` | "Use Forge current" 선택 시 현재 Forge sd_model 사용 |
| Qwen3 0.6B Text Encoder | `models/text_encoder/qwen_3_06b_base.safetensors` 등 | **필수** — 드롭다운에서 골라야 함(`Use Forge current` 는 Anima용 TE 를 주지 못함). Qwen3-VL·Qwen3.5 파일은 크기가 달라 못 씀 |
| Qwen-Image VAE | `models/VAE/qwen_image_vae.safetensors` | **필수** — 드롭다운에서 고름(Forge 에 따로 불러 둔 VAE 가 없으면 `Use Forge current` 는 실패) |
| ControlNet-LLLite | `models/ControlNet/animaTileRepair_v20.safetensors` 등 | **필수** — 목록에는 safetensors 헤더로 가려낸 **3채널(RGB) Anima LLLite 파일만** 나옴(4채널 인페인트 LLLite·다른 ControlNet 은 이름과 상관없이 빠짐, 확장자는 `.safetensors` 만). 기본 선택은 가장 새 Tile & Repair 파일(v20) |
| LoRA Stack (선택, 4칸) | `models/Lora/*.safetensors` | `models/Lora` 바로 아래 파일만 보이고 하위 폴더는 목록에 안 나옴 |

### VRAM 관리

기본 ON된 `Unload Forge SD before run` 옵션이 Anima 추론 전 `backend.memory_management.unload_all_models()`를 호출 → 현재 SD model을 VRAM에서 빼냅니다. **`sd_models.unload_model_weights()` (모델 nuke)와 다릅니다** — `forge_hash`가 보존돼서 다음 t2i가 idempotent reload로 살아남습니다.

### 한계 / 알려진 제약

- **단일 패스만 지원**: 큰 이미지를 작은 tile로 나눠 추론하는 tiling 루프는 구현돼 있지 않습니다(계획된 버전 없음). source 이미지를 한 번에 추론합니다. 크기는 **SAM3 Anima Short Side** 슬라이더(기본 1024)가 짧은 변을 정하고, 긴 변은 source 비율을 따르며, 두 변 모두 32의 배수로 내림(최소 256)합니다.
- **LLLite 는 Multiplier 하나만**: 벤더 쪽에 strength·시작/끝 스텝 개념이 없어 Multiplier 슬라이더만 있습니다.
- **Sampler 선택 불가**: Anima는 Flow Matching only. `flow_shift` + `infer_steps` 만 sampling을 결정.
- **Attention backend**: Windows 환경에서 `flash_attn` / `sageattention`은 빌드 어려움. vendor가 `torch` (SDPA) fallback으로 작동.

### infotext

결과 PNG의 `parameters` chunk에 `Anima Tile-Repair: on` 마커 + steps / cfg / seed / size / flow shift / LLLite 모델·multiplier(`LLLite: <model> (mult <x>)`) 가 적힙니다. LoRA Stack 에서 쓴 LoRA 와 TE/VAE 선택은 적히지 않습니다. 갤러리에서 결과를 클릭하면 사이드바 prompt가 변환된 prompt로 갱신.

---

## 워크플로 4: LoRA Manager 통합 (v0.9.0+, v0.9.1에서 정상 작동)

> **출처 명시:** 이 기능은 [**willmiao/ComfyUI-Lora-Manager**](https://github.com/willmiao/ComfyUI-Lora-Manager) (GPL-3.0) 프로젝트를 **그대로 가져와(vendored)** Forge에 임베드한 것입니다. LoRA 관리 UI/기능 전부는 원저자 willmiao의 저작물이며, 이 확장은 그 standalone 서버를 Forge UI 안에서 띄우는 통합 레이어만 추가합니다.

[willmiao/ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager)를 Forge에 통합. extra-networks 탭 strip(🎴 버튼으로 여는 Checkpoints/LoRA 카드 영역)에 **Manage 탭**을 추가해서 LoRA 관리(civitai 다운로드, 메타데이터/트리거워드 편집, recipe, preview)를 Forge 안에서 바로 합니다.

> **첫 실행 주의:** 서버가 처음 뜰 때 전체 LoRA 라이브러리를 스캔/해싱합니다 (예: 1487개 ≈ 4~5분). 이 동안 Manage 탭에 "LoRA 모델 스캔 중..." 진행 표시가 나오고, 끝나면 자동으로 UI가 로드됩니다. 두 번째 실행부터는 캐시 덕분에 즉시 뜹니다.

### 동작 방식

- standalone aiohttp 서버를 **lazy spawn** — 기본(`Add Manage tab`)에서는 Manage 탭을 처음 열 때만 백그라운드
  프로세스로 실행 (최초 ~10초). **`Replace LoRA tab` 모드에서는 페이지를 열면 매니저 탭이 선택된 채로 나오므로 그때
  바로 서버가 뜹니다.** 같은 포트에 이미 매니저가 응답하고 있으면 새로 띄우지 않고 그 서버를 씁니다.
- Manage 탭 안에 `<iframe>`으로 manager UI 임베드
- Forge의 LoRA/checkpoint/embeddings 폴더 경로를 서버를 띄울 때마다 manager `settings.json`의 `folder_paths` 에
  씁니다. 다만 현재 벤더(1.2.0)는 첫 실행 뒤 자체 라이브러리(`libraries`) 설정을 우선하므로, Forge 쪽 폴더를 나중에
  바꾸면 매니저에 반영되지 않을 수 있습니다(그때는 매니저 설정에서 폴더를 직접 고치세요).
- Forge 종료 시 서버 자동 종료 (atexit)
- **모든 "send to workflow" → Forge 프롬프트 삽입 (v0.19.0+):** 카드 paper-plane뿐 아니라
  단일 우클릭 메뉴와 **멀티 선택 후 우클릭 벌크 전송**(`.model-card.selected` 전체)도 가로채
  ComfyUI 대신 현재 프롬프트로 삽입합니다. Append는 이어붙이고, Replace는 기존 `<lora:...>`를
  교체합니다. (벤더의 ComfyUI 하드코딩 전송을 브리지가 capture 단계에서 차단)

Manage 탭은 txt2img·img2img 의 extra-networks strip 에 하나씩 주입됩니다. 매니저에서 LoRA를
전송하면 지금 활성 탭의 프롬프트에 바로 추가하거나 기존 LoRA 토큰을 교체합니다.
페이지의 JS 는 숨은 Gradio 버튼 브리지로 설정 조회·서버 기동을 하며, 같은 내용을 HTTP 로도 여는
`/sam3-lora/config`·`/sam3-lora/spawn` 라우트는 JS 가 쓰지 않습니다(외부 도구용으로 남아 있음).
이 두 라우트는 Notebook·메모·Tile & Repair 경로와 같은 인증을 씁니다. 부르는 외부 도구는 헤더
`X-SAM3-Notebook: 1` 을 보내야 하고(없으면 403 — 거절된 `/sam3-lora/spawn` 은 서버를 띄우지 않음),
`--gradio-auth` 로그인이 켜져 있으면 로그인 쿠키를, Forge 를 `--api`·`--nowebui` 와 `--api-auth` 로
띄웠으면 그 HTTP Basic 자격 증명도 보내야 합니다(없거나 틀리면 401).

### 의존성 자동 설치

확장 첫 로드 시 `install.py`가:
1. `willmiao/ComfyUI-Lora-Manager`를 `lora_manager_vendor/`로 shallow clone (~20초) 한 뒤, 이 확장의 소스 패치를
   확인한 커밋 `303cca0`(pyproject 1.2.0)으로 고정합니다. 이미 있는 vendor 판이 1.2.0 이 아니면 콘솔에 알립니다.
2. 누락된 경량 deps(aiohttp-socks, piexif, olefile, natsort, aiosqlite, beautifulsoup4)를 Forge venv에 자동 `pip install`

### 설정 (Settings → SAM3 LoRA Manager)

| 옵션 | 기본값 | 설명 |
|---|---|---|
| Manage 탭 배치 | `Add Manage tab (keep LoRA)` | LoRA 탭 옆에 Manage 탭 추가 / `Replace LoRA tab`이면 LoRA 탭 자리를 대체(페이지를 열 때 매니저 탭이 선택되고 서버가 바로 뜸) |
| 서버 포트 | `8765` | ComfyUI 기본 8188과 충돌 회피. 재시작 후 적용. 이 포트에 이미 떠 있는 매니저는 그대로 재사용 |

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
  sparse LoRA는 원본 세대를 안전하게 판정할 수 없으므로 기본 설정에서는 DiT 블록을 추측 변환하지 않습니다
  (Forge 공통 LLM-adapter 키 정규화만 그대로 수행). 다만 순정 Forge 판정(가장 큰 블록 인덱스+1이
  들어가는 가장 작은 레이아웃)이 현재 모델과 같으면 변환 없이 그대로 로드합니다(예: 2.9B에서
  블록 12를 뺀 39블록 LoRA → 2.9B). 블록 수가 다르면 로드하지 않고 UI에 `Anima LoRA '<파일명>'
  건너뜀: …` 토스트로 알립니다 — 앞 N블록만 든 접두(prefix) LoRA를 더 큰 모델에 올리는 경우도 순정
  Forge(추측 변환)와 달리 거부합니다. Settings → **SAM Extra LoRA** → "Anima 부분 LoRA 순정 추측 변환"
  (`sam3_anima_sparse_lora_forge_guess`, 기본 꺼짐)을 켜면 순정처럼 판정 레이아웃에서 현재 모델로 변환해
  로드합니다(블록 대응이 틀릴 수 있음, 하향도 변환, 인덱스 52 이상은 건너뜀, 중간이 빈 LoRA 는 순정과 결과가
  다를 수 있음). 추측 변환한 생성은 infotext 에 `Anima sparse LoRA` 가 남습니다.
- 단, 더 큰 모델에서 **선두 블록만 정확히 `0…27` 또는 `0…39`로 저장한 특수 partial
  LoRA**는 키만 보면 완전한 28/40블록 LoRA와 구별할 수 없습니다. 이런 파일은 자동 변환
  대상에 쓰지 않는 것이 안전합니다.
- 이는 **LoRA 인덱스 호환 변환**입니다. Base나 2.9B 체크포인트 자체를 학습된 3.8B
  체크포인트로 바꾸는 기능은 아닙니다.

---

## 별도 기능: DoRA 추론 방식

`dora_scale` 이 든 LoRA·LoKr·LoHa(LyCORIS `dora_wd=true` 로 학습한 파일)를 합치는 방식과, 작은 Anima LoRA 를 큰
모델에 얹을 때 **끼워 넣은 블록**을 어떻게 채울지 고릅니다. txt2img·img2img 의 **DoRA 추론 방식** 아코디언을 켜면 적용되고,
꺼 두면 Forge/ComfyUI 순정입니다.

### 계산 방식

| 방식 | 출력 축 DoRA 의 노름 | 계산 정밀도 | 비고 |
|---|---|---|---|
| Forge/Comfy (순정) | 원본 가중치 `‖W₀‖` | `lora_compute_dtype` | ComfyUI 와 코드가 같음 |
| Forge/Comfy 공식 · fp32 | 원본 가중치 `‖W₀‖` | fp32 | 입력 축 파일에 알맞음 |
| LyCORIS (학습과 동일 · fp32) | 합친 가중치 `‖W₀+ΔW‖` | fp32 | 강도 1.0 에서 학습 때 가중치와 같음 |
| DoRA 끔 (크기 보정 없이 ΔW만) | 크기 보정 없음 — `W₀+ΔW` | `lora_compute_dtype` | `dora_scale` 을 지운 파일을 순정 Forge 에 넣은 것과 같음 (실험용) |

- 공식 차이는 **출력 축**(`wd_on_output=True`, LyCORIS 기본값, `dora_scale` 모양 `[out,1]`)에만 있습니다. 입력 축
  (`wd_on_output=False`)은 Forge 도 합친 가중치의 노름을 쓰므로 정밀도만 다릅니다.
- 순정 `lora_compute_dtype` 은 대부분의 GPU(RTX 20xx·Volta 이후, Windows 의 GTX 10xx, ROCm·DirectML·MPS)에서 fp16,
  CPU·`--force-fp32`·GTX 16xx 에서 fp32 입니다. DoRA 는 가중치 전체를 다시 스케일하므로 fp16 반올림과 fp16
  eps(9.8e-4)가 ΔW 에 비해 크게 불어나, 같은 파일도 fp16/fp32 에 따라 구도까지 달라질 수 있습니다.
- **Forge 공식 · fp32 가 늘 학습에 더 가까운 것은 아닙니다.** 출력 축에서 `‖W₀‖` 가 아주 작은 행은 순정의 큰 fp16
  eps 가 오히려 완충해 주는데, fp32 에서는 그대로 커집니다(실측: 2.9B 블록 39 q_proj). 출력 축 파일은 LyCORIS 를
  쓰세요.
- 실측(2.9B 에서 @enaa97-000010, CPU, 실제 Forge 어댑터 코드 대조): LyCORIS 방식은 15개 레이어 모두에서 학습
  가중치에 가장 가깝고, 최종 bf16 반올림으로 생기는 하한의 1~3% 이내입니다.
- 강도(`<lora:x:0.7>`)는 네 방식 모두 Forge 처럼 `W₀ + s·(W_dora − W₀)` 로 섞습니다(0 이면 LoRA 없음). LyCORIS
  라이브러리 자체의 multiplier 는 ΔW 를 늘 100% 넣고 크기 보정만 보간해서 강도 0 에서도 ΔW 가 남는데, 이 동작은
  따르지 않습니다.

### 끼워 넣은 블록 (상향 블록 변환)

위의 **ANIMA LoRA 블록 호환 변환**이 작은 LoRA 를 큰 모델에 얹을 때, 새로 끼워 넣은 블록에는 앞 원본 블록의 LoRA 를
복제합니다(Base→2.9B 12개, 2.9B→3.8B 12개, Base→3.8B 24개).

아래 수치는 2.9B 에서 학습한 @enaa97 을 3.8B 에 얹은 경우, 3.8B 가 끼워 넣은 12블록의 출력 투영
(`self_attn.output_proj`·`cross_attn.output_proj`·`mlp.layer2`, 36개 레이어)을 CPU 에서 잰 값입니다.

- 이 출력 투영은 원본 블록과 방향이 무관하고(cos≈0) 행이 훨씬 짧습니다(레이어별 행 길이 비 중앙값 0.085~0.78,
  전체 중앙값 0.4). DoRA 는 행 크기를 **2.9B 에서 학습한 절대값**(`dora_scale`)으로 맞추므로, 복제하면 행 배율
  `dora_scale/‖W₀‖` 이 레이어별 중앙값 1.3~11.8배(최대 1574배)가 되어 가중치가 바뀝니다.
- 레이어별 변화량 `‖W'−W₀‖/‖W₀‖`: 그대로 복제 10에포크 0.35~10.9배, 1에포크 0.34~9.3배. 덧셈형은 10에포크
  0.022~0.12배, 1에포크 0.008~0.04배.
- 부풀림은 학습량과 거의 무관합니다. 1에포크는 학습된 ΔW 가 `W₀` 의 0.6~0.8% 인데도, 출력 투영 종류별 변화량
  중앙값이 88~249% 로 10에포크(89~271%)와 거의 같습니다 — `dora_scale` 이 처음부터 2.9B 의 `‖W₀‖` 로 시작하기
  때문입니다. 원래 40블록에서는 DoRA·덧셈형·구운 버전이 1~2% 이내로 같습니다.
- Base→2.9B 에서 2.9B 가 끼워 넣은 블록도 원본과 방향은 무관하지만 행 길이는 0.8~1.5배로 비슷해서, 부풀림은 3.8B 보다
  훨씬 작을 것으로 봅니다(재지 않았습니다).

| 선택 | 끼워 넣은 블록 | 원래 블록 |
|---|---|---|
| 그대로 복제 (순정 · Forge 기본, 아코디언 기본값) | DoRA 그대로 복제 | DoRA 그대로 |
| 덧셈형 (끼워 넣은 블록만 DoRA 크기 보정 끔) | 복제본에서 `dora_scale` 만 빼고 `W₀+ΔW` | DoRA 그대로 |
| 넣지 않음 (원래 블록에만 · 모든 LoRA) | LoRA 없음 | 그대로 |
| 약한 복사 (브리지식 · 강도·범위 조절) | 범위 안 모듈만 `dora_scale` 을 빼고 `W₀+s·ΔW` (s = 강도) | 그대로 |

- 덧셈형은 DoRA 파일에만 영향이 있고, 넣지 않음은 DoRA 가 아닌 LoRA 도 끼워 넣은 블록에서 뺍니다.
- **약한 복사**는 Civitai 의 ComfyUI "Anima 2B LoRA Bridge"(2B LoRA 를 2.9B/3.8B 에 얹는 노드)가 권하는 방식입니다.
  끼워 넣은 블록의 복제본을 덧셈형으로 두고 ΔW 를 **강도** 배로 줄여, **범위** 안의 모듈에만 넣습니다. 범위는
  어텐션만(self_attn·cross_attn 투영, 브리지 기본) / 어텐션+MLP / 전체(모듈레이션·노름 포함)입니다. 브리지 권장값은
  강도 0.08~0.18(아코디언 기본 0.12), 어텐션만입니다. 아코디언 슬라이더는 0~1 이고, API·XYZ·PNG Info 붙여 넣기는
  0~2 까지 받습니다(넘으면 잘라냄). 강도는 모듈마다 인자 하나(LoRA up/B, LoKr w1, LoHa w1_a, 전체
  diff)에만 곱해서 ΔW 가 정확히 강도 배가 되고, 28→52 처럼 두 세대를 건너뛰어도 끼워 넣은 블록마다 한 번만 곱합니다.
  강도 0 은 넣지 않음과, 강도 1·전체는 덧셈형과 같습니다. 브리지의 블록 위치 대응은 이 확장과 같고, 차이는 이 약한
  복사뿐입니다. XYZ 축 `[DoRA] Weak copy strength`(숫자)·`[DoRA] Weak copy scope`(attn / attn_mlp / all)가 있고,
  이 두 축만 걸면(끼워 넣은 블록 축 없이) 약한 복사로 돕니다. 렌더 비교(3.8B, 마지막 에포크, LyCORIS, 시드 1개)에서
  약한 복사 0.08~0.18 은 모두 넣지 않음과 덧셈형 사이의 그림이었고 무너진 것은 없었습니다. 시드 하나라 어느 쪽이
  낫다고 말할 근거는 아직 없습니다.
- "덧셈형"의 "크기 보정 끔"은 **끼워 넣은 블록에서만** `dora_scale` 을 빼는 동작입니다. 모든 블록에서 끄려면 계산
  방식을 **DoRA 끔**으로 고르세요(이때 끼워 넣은 블록 선택은 그대로 복제·덧셈형이 같아지고, 넣지 않음만 다릅니다).
  원래 블록은 `W₀` 가 이미 학습한 길이 `m` 과 비슷해서 크기 보정을 끄면 가중치가 1~2%만 달라지지만, 학습한 채널별
  세기 조절은 사라집니다.
- 렌더 비교(@enaa97, 3.8B, 시드 3개 · 1/10/15/20 에포크, 원본 이미지 직접 확인 + 이름을 가린 판정자 8명):
  덧셈형·넣지 않음이 순정보다 **일관되게 낫지 않았습니다.** 심한 붕괴는 덧셈형에서 한 번(얼굴 중복) 나왔고,
  순정은 한 번도 없었습니다. 그래서 기본값은 순정이고, 두 선택은 특정 시드가 무너질 때 바꿔 보는 대안입니다.
  XYZ 축 `[DoRA] Inserted blocks`(keep / additive / skip)로 같은 시드를 비교할 수 있습니다.
- 하향 변환(큰 LoRA → 작은 모델)과 같은 블록 수에는 영향이 없습니다. 이 확장의 블록 변환 훅이 없으면 Forge 기본대로
  복제합니다.

### 공통

- 방식이나 끼워 넣은 블록 선택을 바꾸면 다음 생성에서 LoRA 를 한 번 다시 합칩니다(Forge 의 LoRA 캐시와 조건 캐시를
  비움). 같은 선택이면 합쳐 둔 가중치를 그대로 씁니다. 어떤 상태로 합쳤는지는 모델에 표시해 둡니다 — 새로 올라온
  모델은 그때의 상태로 표시하고, 표시가 없는 모델은 한 번 다시 합칩니다. 그래서 Reload UI·확장 끄기·체크포인트 전환
  (✨·hires 체크포인트·🎯) 뒤에도 낡은 가중치를 다시 쓰지 않습니다.
- SAM3 인페인트·Refine 의 내부 패스는 바깥 생성(또는 마지막 생성)의 선택을 그대로 쓰고, 그 선택을 infotext 에
  남깁니다. 캐릭터 레퍼런스(이어붙이기·IP-Adapter)는 스크립트 러너 없이 도는 별도 잡이라 이 스크립트를 거치지 않아,
  선택은 이어받지만 infotext 에는 남지 않습니다(같은 이유로 Safe PAG·Skimmed CFG 등 다른 always-on 스크립트도 그
  잡에서는 돌지 않습니다).
- fp32 계산은 레이어 하나씩 잠깐 fp32 사본을 만듭니다(8192×2048 레이어 기준 약 200 MB). 일반 경로에서는 결과를
  fp32 로 넘겨 Forge 가 저장 dtype 으로 한 번만 반올림합니다. fp32 계산이 OOM 이면 그 레이어만 순정으로 합칩니다.
- Low VRAM·온라인 LoRA(GGUF 등) 경로는 매 forward 마다 합치므로 fp32 방식의 비용이 스텝마다 들고, ΔW 는 이미
  bf16/fp16 으로 계산된 채 넘어와 노름·스케일만 fp32 입니다. 콘솔에 레이어 수와 합친 횟수가 따로 나옵니다.
- infotext 에 `DoRA mode: LyCORIS` / `Forge fp32` / `No magnitude`, `DoRA inserted: additive` / `skip` / `weak 0.12 attn` 을 남기고 PNG Info 붙여
  넣기로 되살립니다(두 키가 없는 이미지를 붙여 넣으면 순정으로 돌아갑니다). XYZ 축 `[DoRA] Inference mode`
  (Forge (stock) / Forge fp32 / LyCORIS fp32 / DoRA off (no magnitude))와 `[DoRA] Inserted blocks` 로 한 장에 비교할 수 있습니다. API 는
  `alwayson_scripts["DoRA Inference Mode"] = {"args": [true, "lycoris", "weak", 0.12, "attn"]}`(뒤 값을 빼면 keep ·
  0.12 · attn). 키 이름을 적은 dict 하나도 받습니다: `{"args": [{"enabled": true, "inserted": "weak", "weak_strength": 0.1}]}`.
- Forge 본체 파일은 고치지 않습니다. 어댑터 모듈마다 `from .base import weight_decompose` 로 복사된 함수를
  lora·lokr·loha·glora 모듈에서 바꿔 끼웁니다. OFT·BOFT 는 alpha 자리에 OFT 제약을, OFTv2 는 강도를 넘겨서
  제외합니다.

---

## 워크플로 5: txt2img Notebook

원래 Forge 주소의 **문서 하나**에서 동작하는 프리셋 도구입니다(예전 Live Workspace 의 다중 문서·iframe·별도
라우트는 없어졌습니다). Notebook 은 txt2img 3열의 갤러리 아래(생성 정보 다음)에 놓입니다. 열 구성과 칸 배치는
아래 [txt2img 화면 정리](#txt2img-화면-정리-) 절이 기준입니다.

- 네거티브 프롬프트 아래의 Forge 원본 **Generation / Textual Inversion / Checkpoints / Lora**
  탭 바는 그대로라 Checkpoint·LoRA 카드 브라우저를 계속 사용할 수 있음
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
  **이때 그 탭을 새로 고치면 저장되지 못한 편집은 조용히 버려집니다**(복구용 초안이 옛 revision 기준이라 쓰이지
  않음). 남기고 싶은 편집은 새로 고치기 전에 JSON 내보내기로 받아 두세요.
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
  브라우저에서 한 번 `Ctrl+Shift+R` 하세요. 접속 주소는 원래 Forge 루트 주소(예: `http://127.0.0.1:7860/`)입니다.

---

## 워크플로 6: Anima Character Reference / ReStyler

캐릭터 이미지 **하나**만 넣으면 공개 [Anima ReStyler v1.2](https://civitai.com/models/2803070/anima-restyler)
워크플로와 같은 조건으로 그 캐릭터를 레퍼런스한 결과를 T2I 갤러리에 추가합니다. Forge Neo의
`anima_do_reference`를 작업 중에만 켜고, 끝나면 되돌립니다.

패널은 txt2img 갤러리 아래 **선택 이미지 도구 → 캐릭터 레퍼런스** 탭에 있습니다.

**사용법**
1. Anima 체크포인트를 로드합니다(3.8B v1.1 번들이면 이어붙이기 방식에서는 3.8B 커넥터도 자동으로 켜집니다 —
   IP-Adapter 방식은 아래 참고).
2. `models/Lora`에 **AnimeEditV2**를 둡니다(필수, 자동 감지 — IP-Adapter 방식은 Edit LoRA 를 쓰지 않지만 지금은
   이 파일이 없으면 두 방식 모두 실행을 막습니다). Extend LoRA는 선택이며 기본값은
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

### 방식 두 가지 (패널 맨 위 `방식`)

- **이어붙이기**(기본) — 위에서 설명한 split-screen 캔버스 방식입니다. 추가 가중치가 필요 없고,
  의상·그림체까지 통째로 옮길 때 좋습니다.
- **IP-Adapter** — 참조 이미지를 SigLIP2로 읽어 Anima DiT 블록마다 K/V를 주입합니다. 캔버스가
  없어 **구도가 프롬프트대로 자유롭습니다**. 크기·샘플러·시드 설정은 그대로 쓰입니다.
  - 처음 쓸 때 전문가 설정의 **모델 받기 (2.0 GB)** 를 누르면 받습니다(저장소에는 넣지 않습니다).
    어댑터 [LuciferTC/Anima-IP-Adapter](https://huggingface.co/LuciferTC/Anima-IP-Adapter)
    (503 MB, 라이선스 `unknown`) + 인코더
    [google/siglip2-base-patch16-512](https://huggingface.co/google/siglip2-base-patch16-512)(1.5 GB).
    둘 다 커밋 해시로 고정해 받습니다.
  - 전용 설정: IP 강도(기본 1.0), 참조 해상도(기본 512·16의 배수), IP 강도 따로 조절(2-pass,
    샘플링이 2배 느립니다), SigLIP 레이어, 회색 null, 어댑터 LoRA 사용.
  - **2.9B(40블록)·3.8B(52블록)에서도 씁니다.** 세 모델은 차원이 모두 2048로 같고 블록 수만 다르므로,
    28블록 어댑터를 Edit LoRA와 **같은 블록 계보표**로 깊은 모델에 펼칩니다(표는
    `sam3ext/anima_lora_blocks.py`의 `BLOCK_MAPPINGS`이고, 값이 Forge 본체
    `extensions-builtin/sd_forge_lora/networks.py:60-64`와 같습니다). 28→52는 어댑터 28개를 전부
    쓰며 12개가 모델 블록 3개씩에 대응합니다.
    - **끼워 넣은 블록 처리** — Settings → **SAM Extra Character Reference** → "캐릭터 레퍼런스 IP-Adapter:
      28블록 어댑터를 2.9B·3.8B 에 얹을 때 끼워 넣은 블록 처리"(`sam3_ipa_duplicate_policy`):
      - **lineage**(기본): 각 어댑터 블록을 원래 계보 자리 한 곳에만 주입하고, 끼워 넣은 블록(2.9B 12개,
        3.8B 24개)에는 IP 모듈도 어댑터 LoRA 도 걸지 않습니다.
      - **all**(0.21까지의 동작): 대응하는 블록 전부에 같은 강도로 주입해 같은 참조가 2~3번 더해집니다.
        2.9B 는 강도 1.0, 3.8B 는 0.5 에서도 격자 무늬로 깨졌습니다.
      - **split**: 같은 어댑터 블록을 쓰는 블록들이 강도를 1/n 씩 나눕니다. all 보다 덜 깨지지만 격자 무늬가
        남았습니다.
      - 2026-09-23 GPU 비교(참조 1장·시드 1개·강도 1.0)에서 lineage 만 2.9B·3.8B 모두 깨지지 않았습니다.
        28블록 모델에서는 어느 값이든 결과가 같습니다.
    - 매핑이 걸리면 상태 줄과 infotext(`SAM3 IPA Blocks: 28→52`, `SAM3 IPA Duplicates: lineage`)에 남습니다.
      28블록에서 쓰던 강도가 그대로 맞는다는 보장은 없으니 한 번 다시 잡아 보세요.
  - 어댑터가 모델보다 **깊은** 조합(52블록 어댑터 + 28블록 모델 등)은 거부합니다 — 계보표에 축소
    항목이 있긴 하지만 그건 어댑터 블록을 버리는 것이라(52→28은 24개) 쓰지 않습니다. Forge 본체도
    하향은 거부합니다.
  - 차원이 다르거나, 계보표에 없는 깊이 조합이거나, 어댑터에 비어 있는 블록이 있으면 **어긋난 숫자를 보여 주고
    멈춥니다**. 어댑터 구조 중 거부하는 것은 두 가지(블록 MLP 앞 주입 `ip_inject_before_mlp`, 독립 IP Q 투영
    `shared_ip_q_proj`)뿐입니다. SigLIP compressor·self-attention·공유 투영 같은 다른 구조는 키 이름으로 알아보기만
    하고 모두 쓰이는지는 검사하지 않으므로, 지금 고정한 어댑터와 구조가 다른 파일은 멈추지 않고 다르게 돌 수 있습니다.
  - **3.8B v2 커넥터**: IP-Adapter 방식은 스크립트 러너 없이 도는 별도 잡이라 3.8B 아코디언이 붙지 않습니다. 대신
    체크포인트가 v2 번들이면 기본으로 이어붙이기 방식·txt2img 와 같은 Qwen3.5 커넥터 조건을 설치하고 그 위에
    IP-Adapter 를 얹습니다(부정 프롬프트는 순정 경로). Settings → **SAM Extra Anima 3.8B** → "Anima 3.8B: 캐릭터
    레퍼런스 IP-Adapter 방식에도 v2 커넥터 설치 (끄면 결과가 바뀜)"(`sam3_anima38_reference_ipa`, 기본 켬)를 끄면
    0.21까지처럼 0.6B 조건만으로 샘플링합니다. 켜져 있으면 infotext 의 `Reference Anima38`(설치 여부·이유)에
    남습니다. v2 번들이 아닌 모델(2.9B 등)에서는 아무것도 설치하지 않습니다.
  - 주입은 샘플링 동안에만 살아 있습니다. 잡이 끝나면 평소 생성에는 흔적이 남지 않습니다.
  - 다른 확장이 원본 DiT 블록이나 DiT 에 인스턴스 `forward` 패치를 남겨 두면 주입이 무효가 될 수 있습니다 — 주입이
    0건이면 오류로 멈추고 원인(껍데기 블록을 안 거침 / IP 토큰이 안 실려 옴)을 알려 줍니다.

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
| **SLG** | weak row의 선택 block을 no-op으로 복원 | 실행 경로 확인됨(실제 Anima, 3 steps), 화질 A/B 미완 |
| **APG** | post-CFG guidance를 cond 평행/직교 성분으로 투영 | 중립값 단위 검증. **CFG>1 전용**(CFG 1이면 건너뛰고 콘솔에 1회 경고), reference와 픽셀 동일하지 않음 |
| **CWM / SMC** | Haar 대역별 CFG 배율 / step 간 unit-L2 switching control | 수학·중립값 검증, 실제 CWM 실행 확인. **CFG>1 전용**(CFG 1이면 건너뜀) |
| **DCW** | live x_t와 x0의 wavelet 차이를 post-CFG 마지막에 보정 | 4D/5D·홀수 해상도·중립값 검증, 실제 실행 확인 |
| **RDC** | DCW Haar 대역의 step 간 EMA drift를 보정 | tau=0 비트 동일·상태/해상도 재초기화 단위 검증, 이미지 A/B 필요 |
| **DAVE** | Anima block 출력의 token/spatial DC 성분 감쇠 | 실제 block hit 확인, 다양성/권장 block은 추가 A/B 필요 |
| **CNS-inspired** | 기존 seeded/Brownian noise를 live x_t 에너지로 재색칠 | Euler a noise call 확인. deterministic sampler에서는 inert |
| **Anima Modulation Guidance** | 보조 CLIP-L 방향을 공개 어댑터로 block AdaLN에 가산 | 실제 CLIP/어댑터 로드·투영·주입 검증. 이미지 품질 A/B는 추가 필요 |
| **Adaptive Guidance** | combined batch의 후반 uncond row 생략 | low-VRAM 분리 호출에서는 생략/속도 이득 없음 |
| **Skimmed CFG** | 과포화를 만드는 성분만 낮은 CFG로 되돌림(anti-burn) | 원본 노드와 σ 구간·결과가 비트 단위로 같음(원본 수식 편입). **CFG>1 전용**, pre-CFG가 아닌 post-CFG 맨 앞에서 재구성 |
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
- CFG base의 배율은 원본 DCW(+a) cfg 훅처럼 Forge가 넘기는 `cond_scale`이고, 다른 CFG 함수가 없으면 Forge처럼
  `edit_strength`를 곱해 보존합니다. incoming을 최소제곱으로 맞춘 값은 진단(`w_fit`)과 비선형 fit 경고에만 씁니다.
  RescaleCFG처럼 다른 확장이 `sampler_cfg_function`을 걸어 두면 원본처럼 SMC·CWM만 비킵니다(경고 1회).
- Skimmed CFG는 별도 스크립트(별도 아코디언)이며 CFG base보다 **먼저** 실행되고, skim 결과를
  Forge의 예측 tensor에 다시 써서 이후 SMC/APG/CWM·PAG delta·DCW가 모두 그 위에서
  동작합니다. Forge의 중복 메서드 정의로 `sorting_priority`가 실제 실행에서 무시되는 문제를
  피하려고 callback을 목록 맨 앞에 직접 dedupe+prepend합니다.
- `Legacy CFG base mode` 아코디언의 라디오와 `Experimental stack`은 구버전 호환용이며
  위 토글과 OR로 합쳐집니다.
- CWM `alpha high > +0.15`는 Anima 16채널 latent에서 캐릭터 분리를 만들 수 있어 UI 경고가 뜹니다.
- RDC는 DCW의 Haar pass를 공유하는 step 간 EMA 보정입니다. 원본 ComfyUI-DCW처럼 따로 켜는 스위치가 없고
  **Enable DCW가 켜져 있고 `tau > 0`**일 때만 돕니다. 기본값은 `tau=0`(끔), `alpha LL=0.03`, `alpha HH=0`이며
  HH는 텍스처가 뭉개질 수 있어 기본 0입니다. DCW·CWM·CNS 기본값도 원본과 같습니다(DCW λ 0.05/0.01, CWM α 0,
  CNS gamma scale 2.0).
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
  `tests/test_guidance_suite.py`, `tests/test_guidance_composition.py`, `tests/test_skimmed_cfg.py`,
  `tests/test_modulation_guidance.py`, `tests/test_anima_detail_daemon.py`, `tests/test_detail_daemon_origin.py`
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
   함께 나오는 *Negative adapter strength* 슬라이더는 구형 v1 어댑터에만 쓰이고, v2 번들은 강도 1.0 고정이라
   슬라이더 값을 무시합니다(infotext 에는 1.0 으로 남음).
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
- 두 인코더는 순차로 돕니다. 같은 프롬프트 줄의 0.6B TE 결과는 CPU 에 최근 32줄까지 캐시해 TE 를 다시
  올리지 않습니다. 인코딩이 끝난 TE·Qwen3.5 는 여유 VRAM 이 다음 샘플링에 충분하면 Forge 관리 모델로 남기고
  (Forge 가 자리가 필요할 때 퇴출), 모자라면 곧바로 내립니다. 커넥터는 샘플링 모델의 일부로 Forge 가
  관리하며(오프로딩 가능) 생성이 끝나면 설치는 원복되지만 커넥터 자체는 다음 생성을 위해 VRAM 에 남습니다
  (Forge 퇴출·체크포인트 변경 때 내려감). 그래서 3.8B 생성 뒤 최대 약 6~8 GB 가 VRAM 에 남을 수 있고, Forge 밖의
  VRAM 사용(같은 GPU 의 학습 등)은 이 판단에 들어가지 않습니다. 학습과 함께 쓸 때는 Settings → **SAM Extra
  Anima 3.8B** → "Anima 3.8B: TE·Qwen3.5·커넥터를 생성 사이 VRAM 에 남기기 (최대 약 6~8 GB)"
  (`sam3_anima38_keep_resident`, 기본 켬)를 끄세요. 끄면 TE·Qwen3.5 는 인코딩 직후, 커넥터는 생성이 끝날 때 내리고
  (이때 3.8B 가 쓴 Forge TE 도 함께 내림), 결과 이미지는 같습니다. 재시작은 필요 없습니다. 생성이 끝날 때 도중 재로드로
  고아가 된 샘플링 패처도 VRAM 에서 내립니다. LoRA 세트가 바뀌어 Forge 가
  UNet 을 새로 복제해도(Feature 6, batch count 사이 LoRA 변경) 커넥터가 따라가 메모리 관리 안에 듭니다.
- 커넥터는 스텝마다 run(프롬프트 줄)마다 한 번씩 돕니다. 이 비용을 줄이는 두 설정이 Settings → **SAM Extra Anima
  3.8B** 에 있고 둘 다 기본 켬이며, 결과 이미지는 켜든 끄든 비트 단위로 같습니다. (1) "커넥터를 샘플링 동안 fp32 로
  상주"(`sam3_anima38_connector_fp32`): Forge 가 올린 bf16 가중치(LoRA 합친 값)를 fp32 로 한 번 바꿔 두어 스텝마다
  모듈 약 300개의 캐스트와 `--cuda-stream` 대기를 건너뜁니다. 커넥터가 VRAM 에서 약 1.6 GB → 3.1 GB 가 되며(Forge
  가 bf16 로 다 올리고 남긴 여유 안에서만 바꾸고 바꾼 뒤에는 Forge 메모리 계산에 포함, 자리가 모자라면 bf16 그대로 —
  Forge 의 적재·퇴출 판단은 끈 때와 같음), 위 상주 설정의 '최대 약 6~8 GB' 도 약 1.5 GB 늘어납니다.
  (2) "커넥터의 timestep 무관 계산을 프롬프트 줄마다 한 번만"(`sam3_anima38_connector_run_cache`): 의미 특징·0.6B
  source 의 K/V 와 첫 블록을 run 의 첫 스텝에 계산해 두고 다시 씁니다(줄당 약 15~70 MB, 합계 최대 512 MB, 생성이
  끝나면 버림). llm_adapter 에 LoRA 가 바뀌면 다시 계산하고, 커넥터가 텍스트 인코더 모듈을 같이 쓰는 폴백에서는
  쓰지 않습니다.
- 같은 프롬프트 줄의 Qwen3.5 특징은 최근 32줄까지 캐시합니다 — batch count·Feature 6 후보·
  ADetailer 가 같은 줄이면 Qwen3.5-4B 를 GPU 로 다시 올리지 않습니다. batch count 사이에는 설치를
  유지해 Forge 의 조건 캐시도 이어지고(다음 batch 를 시작할 때 모델이 다시 로드돼 있으면 새로 설치, NegPiP 가
  앞 순서라 래퍼가 빠지면 다시 겁니다), 순정으로 가는 부정 프롬프트는 Forge 의 공용 캐시를 씁니다.
- **한계**: 한 장 안에서 Hires 체크포인트가 다른 모델이면 하이레스 패스는 커넥터 없이 순정으로 돌 수 있고,
  infotext 에는 첫 패스 상태만 남습니다.
- `qwen35_4b.safetensors`(또는 동봉 토크나이저, v1 이면 어댑터 파일)가 없으면 생성 **전에** 알아채
  한 번 경고한 뒤 순정 Anima 로 진행합니다. infotext 에 `Anima38: off: …` 가 남습니다. 사전 확인은 파일이
  있는지만 봅니다 — 파일은 있는데 읽기·로드가 실패하면(깨진 파일 등) 인코딩 단계에서 생성이 오류로 멈춥니다.
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

## txt2img 화면 정리 (🧭)

프롬프트 밑에 **켜진 기능** 줄이 있습니다. 지금 켜져 있는 기능이 칩으로 보이고, 칩을 누르면 그 칸으로 갑니다
(숨어 있으면 펼쳐서 열어 줍니다). 아코디언은 묶음별로 나뉩니다.

| 자리 | 내용 |
| --- | --- |
| 1열 | Forge 기본 설정 → **고정**(핀 꽂은 것) → **ANIMA 튜닝**(3.8B·Detail Daemon·Skimmed CFG·Guidance·VAE 2x) |
| 2열 | **디테일러**(SAM3 Mask·A디테일러) → **스크립트**(드롭다운과 고른 패널) → **더 보기**(로라·제어, 도구·실험) |
| 3열 | 갤러리 → **선택 이미지** 탭(SAM3 Refine · Tile-Repair · 캐릭터 레퍼런스) → 생성 정보 → Notebook |

- **켜짐 표시**: 헤더 오른쪽에 `켜짐`, 기능이 여럿인 칸은 `2/11 켜짐` 처럼 나옵니다. 이름으로 켜기 컨트롤을
  추측한 확장은 점선 테두리로 구분합니다.
- **핀**: 헤더의 핀을 누르면 그 칸이 1열 **고정** 으로 올라갑니다(이 브라우저에 기억). 해제하면 원래 자리로 돌아갑니다.
- **더 보기**: 잘 안 쓰는 항목 묶음입니다. 접혀 있어도 안에 켜진 게 있으면 `켜짐 1 · 로라 블록 웨이트` 처럼 보여 줍니다.
- **선택 이미지 탭**: 갤러리에서 고른 이미지에 쓰는 도구 세 가지를 한곳에 모았습니다(예전에는 Refine 만 갤러리 밑에,
  나머지 둘은 왼쪽 열 중간에 있었습니다).
- **끄기**: 설정 → SAM Extra Appearance → "txt2img 섹션 정리" 를 끄고 Forge 를 재시작하면 Forge 기본 순서로 돌아갑니다.
  한 페이지에서만 끄려면 주소 끝에 `?sam3_lanes=off` 를 붙이세요.
- 바뀐 화면은 **Forge 재시작 + 강력 새로 고침(Ctrl+F5)** 후에 보입니다.

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
| 장치 | GPU(기본) / CPU — 옆의 **GPU 에 남겨 두기**(기본 켬)를 켜 두면 GPU 로 돈 TIPO(약 2 GB)를 Forge 메모리 관리에 맡겨, 여유가 있으면 VRAM 에 남아 다음 클릭에 다시 올리지 않고 이미지 생성 등으로 Forge 가 자리가 필요하면 통째로 내립니다(학습 등 Forge 밖 VRAM 사용은 모릅니다). 끄면 누를 때만 올렸다가 끝나면 내립니다. 이 체크박스는 새로 고치면 기본값으로 돌아갑니다. 여유 VRAM 이 3 GB 보다 적으면 그 회차는 CPU 로 돕니다. 쉴 때는 RAM 에 약 2 GB(CPU 로 돌린 뒤에는 fp32 약 4 GB)로 두며, 이 RAM 사본은 Forge 를 끌 때까지 내리는 방법이 없습니다. 마지막으로 고른 장치는 브라우저가 기억합니다 |
| 시드 | -1 이면 누를 때마다 다른 결과, 숫자면 같은 결과 |
| ↩ 되돌리기 | 마지막 확장 전 프롬프트로 |

### 규칙
- **적어 둔 태그·문장은 바꾸지 않습니다**(가중치·LoRA·순서 그대로). 예외로 프롬프트 끝의 쉼표·공백은 정리되며, 새로 붙일
  태그가 없어도 끝 쉼표만 지워진 결과가 '태그 0개 추가' 로 들어갈 수 있습니다. 새 태그는 그 뒤에 캐릭터 → 작품 → @작가 → 일반
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
| **Anima Safe PAG** (별도 기능) | [iljung1106/comfyui-anima-safe-pag](https://github.com/iljung1106/comfyui-anima-safe-pag) (commit `905b0107`) | MIT | Anima 배치·블록 선택을 이식, σ 적용 구간·번호 파싱 편입(`sam3ext/guidance/sigma_window.py`) |
| **DCW / RDC / CWM / SMC** | [namemechan/ComfyUI-DCW](https://github.com/namemechan/ComfyUI-DCW) | GPL-3.0 | 수식 기반 Forge 재작성 (vendor 아님) |
| **DAVE** | [daheekwon/DAVE](https://github.com/daheekwon/DAVE) · [sorryhyun/ComfyUI-Anima-DAVE](https://github.com/sorryhyun/ComfyUI-Anima-DAVE) (commit `83143e8d`) | MIT | 초반 스텝 게이트 편입(`sam3ext/guidance/dave_gate.py`), block 계산은 Forge 재구현 |
| **CNS-inspired Wavelet Noise** | [namemechan/comfyui-cns_sampler_patch](https://github.com/namemechan/comfyui-cns_sampler_patch) | GPL-3.0 | `color_noise_wavelet` 편입(`sam3ext/guidance/cns.py`), Forge 훅은 재작성 |
| **Anima Modulation Guidance** | [Anzhc/Anima-Mod-Guidance-ComfyUI-Node](https://github.com/Anzhc/Anima-Mod-Guidance-ComfyUI-Node) · [quickjkee/modulation-guidance](https://github.com/quickjkee/modulation-guidance) · [yresearch/cosmos-pooled](https://huggingface.co/yresearch/cosmos-pooled) | MIT(코드 선언) / 자산 모델 카드 | Forge block 재작성 (vendor 아님) |
| **Skimmed CFG** (별도 기능) | [Extraltodeus/Skimmed_CFG](https://github.com/Extraltodeus/Skimmed_CFG) | Apache-2.0 | 수식·σ 게이트 편입(`sam3ext/guidance/skimmed_cfg.py`), Forge 훅 |
| **Detail Daemon** (별도 기능) | [muerrilla/sd-webui-detail-daemon](https://github.com/muerrilla/sd-webui-detail-daemon) (commit `1947999`) · [Jonseed/ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon) (commit `3394e44`) | MIT | schedule·σ 조회 함수 편입(`scripts/anima_detail_daemon.py`), Forge 훅 |
| SAM3 검출 | Meta [facebook/sam3](https://huggingface.co/facebook/sam3) ([facebookresearch/sam3](https://github.com/facebookresearch/sam3)) | SAM License(Meta) | `sam3` PyPI 패키지 (저장소에 포함하지 않음) |

저장소에 함께 들어 있는 제3자 파일(`sam3ext/anima38/`, TIPO 모델 코드, `assets/` 의 Qwen3.5 토크나이저·CLIP BPE
어휘)과 편입한 상류 함수(Skimmed CFG, Detail Daemon, Safe PAG 적용 구간, DAVE 게이트, CNS 재색칠)의 출처와 라이선스는
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 에 있습니다.

원저자분들께 감사드립니다. 각 프로젝트의 라이선스 전문은 vendor 디렉터리의 `LICENSE` 파일을 참고하세요.

### vendor 수정 사항 (GPL-3.0 §5 수정 고지)

LoRA Manager(GPL-3.0)는 vendor 그대로 실행하되, `lora_manager_core.py`가 spawn 시 아래 패치를 **marker-guarded·멱등**으로 적용합니다 (LICENSE·저작권 고지·저자 귀속은 일절 변경하지 않음 — DOM/동작만 조정):

- **fetch 진행 상태 줄바꿈** — `loading.css`의 `.loading-status` wrap 허용 (긴 LoRA 이름 뒤 카운터 잘림 수정)
- **후원/지원 UI 숨김** — Ko-fi/Patreon/WeChat/Afdian 등 기부 버튼·모달·배너를 `display:none`으로 숨김 (GPL이 보존을 요구하는 "Appropriate Legal Notices"가 아니므로 허용)
- **업데이트 알림 비활성화** — `update_routes.py`의 `check_updates`를 short-circuit해 willmiao 원본 릴리스 폴링/알림 점을 끔 (vendor는 install.py가 새로 받을 때 커밋 `303cca0` 으로 고정하므로 사용자가 상류 업데이트에 조치 불가). 우리 repo로의 repoint은 버전 체계 불일치로 무의미하여 하지 않음.
- **Forge Neo 연동 (v0.9.4)** — ComfyUI 상호작용을 Forge Neo용으로 전환:
  - LoRA 카드의 "Send to ComfyUI"(✈️) → **"Add LoRA"**: 클릭 시 cross-origin iframe→부모 `postMessage`로 활성 탭 **Positive Prompt에 `<lora:이름:weight>` 삽입** (Forge 네이티브 LoRA 카드 클릭과 동일). 브릿지 스크립트 `static/forge_bridge.js`를 `base.html`에 주입.
  - locale 전 언어 + 런타임 DOM에서 **"ComfyUI" → "Forge Neo"** 라벨 치환 (willmiao 위키/repo URL은 링크·귀속 유지를 위해 변경 안 함).
  - **버그 수정**: 사용 팁 X 버튼(가상 스크롤 시 카드 null → 삭제 실패) 소스 한 줄 가드; 추가 메모 placeholder(비영어 locale에서 포커스 시 안 지워지던 것)를 `data-placeholder`+CSS `:empty::before` 진짜 placeholder로 전환.

이 패치들은 vendor 재clone 시 자동 재적용됩니다.

---

## 문서

| 문서 | 내용 |
|---|---|
| [CHANGELOG.md](CHANGELOG.md) | 버전별 변경 내역 |
| [docs/GUIDANCE.md](docs/GUIDANCE.md) | Anima Guidance Suite 상세(파라미터·처리 순서·XYZ·검증 로그·크레딧) |
| [docs/REGIONAL_STYLE_SWAP.md](docs/REGIONAL_STYLE_SWAP.md) | 🎭 Regional Swap — RegionalSampler 워크플로를 Refine 로 재현하는 레시피 |
| [docs/EXPERIMENTAL_STATUS.md](docs/EXPERIMENTAL_STATUS.md) | 실험 기능(Refine · Tile-Repair) 진단 체크리스트 |
| [design.md](design.md) | UI 디자인 토큰·규칙(개발용) |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | 저장소에 포함된 제3자 코드·자산의 출처와 라이선스 |
| [LICENSE](LICENSE) | 이 확장의 라이선스(GPL-3.0-only) 전문 |

## 라이선스

이 확장 전체는 **GNU General Public License 3판만(SPDX `GPL-3.0-only`)** 으로 배포합니다. 이후 판("or later")으로
바꿔 쓰는 것은 허용하지 않습니다. 전문은 [LICENSE](LICENSE) 에 있습니다.

- 저장소에 함께 들어 있는 제3자 코드·자산(`sam3ext/anima38/` 의 MIT 코드, TIPO 모델 코드(Apache-2.0),
  `assets/` 의 Qwen3.5 토크나이저(Apache-2.0)·CLIP BPE 어휘(MIT))는 원래 고지를 그대로 유지합니다 — 목록과 조건은
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 참고.
- `install.py` 가 첫 실행 때 받는 vendor(LoRA Manager: GPL-3.0, kohya-ss/sd-scripts: Apache-2.0)는 저장소에 포함되지
  않으며 각자의 라이선스를 따릅니다.
- 사용자가 따로 받는 모델 가중치(SAM3, TIPO, IP-Adapter, Modulation Guidance 어댑터 등)는 이 라이선스의 대상이 아니며
  각 모델 배포처의 조건을 따릅니다.
- Skimmed CFG 는 상류(Apache-2.0)의 수식 함수를, Detail Daemon 은 상류(MIT, muerrilla·Jonseed)의 schedule·σ 조회
  함수를, Safe PAG·DAVE 는 상류(MIT)의 적용 구간·번호 파싱과 초반 스텝 게이트를, CNS 는 상류(GPL-3.0)의
  `color_noise_wavelet` 을 그대로 편입했습니다(고지는 THIRD_PARTY_NOTICES.md).
