# sam-extra (Forge SAM3 Extension)

SAM3 / SAM3.1 마스크 + 인페인트 확장. ADetailer 스타일의 자동 검출 → 마스크 → 인페인트 파이프라인을 SAM3 텍스트 프롬프트 기반으로 수행합니다. **v0.4.0부터 ControlNet 통합과 t2i 갤러리 옆 Refine 패널이 추가됐습니다.**

## 설치

```
cd <sd-webui-forge-neo>/extensions
git clone https://github.com/UR-al/sam-extra.git
```

webui 재시작 후 txt2img / img2img 패널의 "SAM3 Mask" 아코디언이 표시되면 정상입니다. ControlNet 통합 기능은 `sd_forge_controlnet` 익스텐션이 함께 로드돼 있을 때만 활성화되며, 없어도 SAM3 확장 자체는 정상 동작합니다.

## 모델 다운로드

| 모델 | 파일 | 출처 |
| --- | --- | --- |
| SAM3 | `sam3.pt` (3.45 GB) | <https://huggingface.co/1038lab/sam3/tree/main> |
| SAM3 | `sam3.safetensors` (3.44 GB) | <https://huggingface.co/1038lab/sam3/tree/main> |
| SAM3.1 multiplex (fp16) | `sam3.1_multiplex_fp16.safetensors` (1.75 GB) | <https://huggingface.co/Comfy-Org/sam3.1/tree/main> (`/checkpoints` 폴더) |

## 설치 경로

다운로드한 파일은 아래 경로에 그대로 넣어 두면 됩니다 (파일명 그대로):

```
<sd-webui-forge-neo>/models/sam3/
├── sam3.pt
├── sam3.safetensors
└── sam3.1_multiplex_fp16.safetensors
```

`models/sam3/` 폴더가 없다면 직접 만들어 주세요. webui 실행 후 UI의 "SAM3 Checkpoint" 드롭다운에 자동 노출됩니다.

체크포인트가 하나도 없으면 Hugging Face의 `facebook/sam3` 에서 자동 다운로드됩니다. 완전한 오프라인 사용 시에는 `--sam3-no-huggingface` 옵션으로 자동 다운로드를 비활성화할 수 있습니다.

## 두 가지 워크플로

### 1) In-flight: t2i / i2i 생성 안에서 자동 인페인트

ADetailer와 같은 흐름. 이미지가 생성되면 즉시 SAM3가 마스킹 → 인페인트 → 결과가 원본 이미지를 대체합니다.

- "SAM3 Mask" 아코디언에서 Enable 체크
- Detect Prompt에 마스킹 대상(예: `face`)
- 옵션으로 Inpaint Prompt / Negative Prompt 입력 (비워두면 메인 프롬프트 사용)
- ControlNet 아코디언에서 인페인트 패스에 적용할 CN 유닛 지정 가능

### 2) Post-generation: 갤러리 옆 Refine 패널 (v0.4.0 신규)

t2i가 끝난 뒤 갤러리에서 원하는 이미지를 골라 다시 손볼 때 사용합니다. 셔츠 입은 남자 → 셔츠 클릭 → "빨간 가죽 자켓" 입력 → Refine → 갤러리 옆에 자켓 버전 추가 → 그 자켓 버전 또 클릭해서 "청바지" Refine 식으로 갤러리 안에서 변주가 쌓입니다.

흐름:

1. t2i Generate → 갤러리에 결과 표시
2. 갤러리에서 손볼 이미지 클릭 (선택 인덱스 자동 추적)
3. 갤러리 아래 "SAM3 Refine (post-generation)" 아코디언 펼치기
4. Detect / Inpaint / Negative 프롬프트 입력
5. 필요 시 ControlNet 토글 + 모델/모듈 선택
6. **▶ Refine** 클릭 → 결과가 선택 이미지 뒤(또는 끝)에 삽입됨

Refine 패널은 자체 inpaint·CN 파라미터를 가지며 t2i Generate를 거치지 않고도 즉시 실행됩니다. 외부 ControlNet 유닛(t2i 탭에 미리 켜둔 것)은 기본적으로 그대로 유지되고, "Override external CN units" 체크 시 비활성화됩니다.

**프롬프트 자동 폴백** (v0.4.2): Refine 패널의 Inpaint Prompt가 비어 있으면 메인 t2i 프롬프트를, Negative Prompt가 비어 있으면 메인 t2i 네거티브 프롬프트를 자동으로 사용합니다. 둘 다 비워두면 메인 t2i 프롬프트가 그대로 들어가서 "원본 스타일 유지하면서 마스크만 다시 디노이즈"하는 디테일러 워크플로가 한 줄 입력 없이 됩니다.

**메인 프롬프트 자동 병합** (v0.4.7): "Inherit main t2i prompt" 체크박스 (기본 ON). Refine prompt를 채워도 메인 t2i 프롬프트를 **앞쪽에 자동 prepend**합니다 — 메인의 `<lora:...>` 구문, 스타일 트리거(`score_9`, `masterpiece` 등)가 Refine 패스에서도 그대로 적용됩니다. 옷 교체처럼 새 prompt를 입력해도 원본의 화풍·LoRA가 살아남음. 깔끔히 override하고 싶으면 체크 해제.

> **옷 교체 시 CN 모델 주의**: `anima-lllite-inpainting-v2` 같은 **인페인트 전용 CN**은 "주변 컨텍스트와 매끄럽게 섞이게"가 목적이라 원본 옷의 색·실루엣을 적극적으로 보존합니다 — 교체를 적극적으로 방해함. 옷을 *바꾸려면* CN을 끄거나, `depth_*` 같은 *구조적* CN(신체 모양만 유지)으로 바꾸세요.

**LLLite anima 인페인트 모델**: `anima-lllite-inpainting-*` 같은 LLLite 인페인트 모델을 고르면 preprocessor를 `inpaint_only` 같은 걸로 두면 안 됩니다 (마스크 텐서가 소실되어 어설션 실패). v0.4.2부터는 자동 감지해서 `None`으로 override합니다 — LLLite 자체가 4채널(RGB+mask) 입력을 만들기 때문에 외부 전처리가 필요 없습니다.

## ControlNet 통합

SAM3 인페인트 패스에 ControlNet 유닛 1개를 주입합니다.

**모델 위치**: CN 모델 드롭다운은 기본 ControlNet 폴더(`models/ControlNet/`) **+** `models/sam3/` 둘 다 스캔합니다. SAM3 검출 체크포인트(`sam3*.{pt,safetensors}`)와 같은 폴더에 LLLite 인페인트 모델(`anima-lllite-inpainting-v2.safetensors` 등)을 두면 자동으로 드롭다운에 노출됩니다.

전처리기에 따라 의미가 완전히 달라집니다:

| Preprocessor | 보존하는 것 | 잘 맞는 시나리오 |
| --- | --- | --- |
| `inpaint_only`, `inpaint_global_harmonious` | 마스크 주변 컨텍스트 | 얼굴 디테일러, 작은 결함 수정 (가장자리 자연스러움 ↑) |
| `inpaint_only+lama` | 위 + LAMA 사전 채움 | 강한 변형 + 자연스러운 시작점 |
| `tile_resample` | 저주파(전반적 색·형태) | "같은 것을 더 디테일하게" |
| `depth_*` | 신체 깊이 / 실루엣 | **옷 교체** — 신체 형태 유지하며 텍스처/색만 변경 |
| `openpose_*` | 포즈(관절) | 포즈 잠그고 외형 자유 |
| `lineart_*`, `canny` | 윤곽선 | 형태 강하게 잠금, 색·재질만 변경 |

옷 갈아입히기 예시:

```
Detect prompt:    shirt
Inpaint prompt:   red leather jacket, detailed stitching
Denoising:        0.85   (높게 — 원본 옷이 비치지 않도록)
Mask Dilation:    8      (SAM3가 너무 타이트하게 잡을 때)

[ControlNet]
  Enable:         True
  Module:         depth_anything_v2   (또는 openpose)
  Model:          control_*_depth
  Weight:         0.7
  Guidance End:   0.8
  Pixel Perfect:  True
```

전체-프레임 변경 시(전신 옷 교체 등)는 "Inpaint only masked"를 끄는 게 보통 더 잘 됩니다 — 크롭된 영역만 보면 CN의 포즈/깊이 정보가 약해지기 때문.

## 마스크 후처리 (v0.4.3)

머리카락이나 모피처럼 가는 strand가 SAM3에 의해 부분적으로 누락되는 경우를 위한 두 가지 마스크 확장 옵션:

- **Mask Dilation (px)** — 마스크를 N 픽셀 바깥쪽으로 늘림. 작은 누락에 효과적. (최대 256까지 슬라이드 가능, v0.4.3에서 128→256으로 확장)
- **Convex Hull (wrap strands)** — 검출된 영역을 *최소 볼록다각형*으로 감쌈. SAM3가 잡지 못한 strand 사이 공간까지 자동으로 포함됨. 머리·털·안테나 등에 특히 효과적. 컴포넌트별로 적용되어 분리된 영역끼리 합쳐지지 않음.

적용 순서: hull → dilation → blur. 둘 다 SAM3 패널과 Refine 패널 모두에 노출되며, infotext / XYZ 축으로도 사용 가능.

## 주요 기능

- SAM3 / SAM3.1 (`.pt`, `.safetensors`) 체크포인트 지원
- 텍스트 프롬프트로 검출 (`face, eyes / hand` 처럼 `,` = OR-merge, `/` = 별도 인페인트 패스)
- Combined / Individual 마스크 모드
- **Mask Hull (convex hull) + dilation up to 256 px** — 머리·털 strand가 새는 경우용 (v0.4.3)
- 인페인트 옵션 (denoising, mask blur, only-masked padding, separate width/height, steps, CFG, sampler/scheduler, noise multiplier, restore face)
- **ControlNet 통합** (preprocessor / model / weight / guidance start·end / pixel-perfect / control mode / resize mode / processor res / threshold a·b / override external)
- **Post-generation Refine 패널** — 갤러리에서 이미지 선택 후 즉시 SAM3+인페인트(+CN) 적용, 결과를 갤러리에 누적 삽입, 체이닝 가능
- XYZ plot 축 다수 (기존 SAM3 항목 + CN Enable / Override / Model / Module / Weight / Guidance Start·End)

## 의존성

`requirements.txt` 에 정의되어 있으며 Forge launch 시 자동 설치됩니다.

ControlNet 통합은 `sd_forge_controlnet` 익스텐션에 의존하며 (런타임에 lazy import), 없으면 해당 기능만 비활성화됩니다.

## 라이선스

내부 사용.
