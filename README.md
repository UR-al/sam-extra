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
  [DoRA 추론 방식](#별도-기능-dora-추론-방식) · [Anima VAE DeGrid](#별도-기능-anima-vae-degrid-nafnet) ·
  [Extra Schedulers](#별도-기능-extra-schedulers) · [Extra Samplers](#별도-기능-extra-samplers-샘플러-5종) ·
  [Anima SPEED (저해상도 선행 샘플링)](#별도-기능-anima-speed-저해상도-선행-샘플링) ·
  [Colorcraft (latent 색 보정)](#별도-기능-colorcraft-latent-색-보정) ·
  [Anima Guidance Suite](#별도-기능-anima-guidance-suite) ·
  [txt2img 화면 정리](#txt2img-화면-정리-) · [진행 막대](#별도-기능-진행-막대-smooth-progress) ·
  [TIPO 프롬프트 확장](#별도-기능-tipo-프롬프트-확장-) · [구도 · 카메라](#별도-기능-구도--카메라-프롬프트로-시점-잡기) ·
  [NegPiP (내장)](#별도-기능-negpip-내장) · [MCP 서버 (에이전트 연결)](#별도-기능-mcp-서버--에이전트-연결)
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

### API: 이미 만든 이미지에 SAM3 만 — 원본 기준 (`sam3_source_image`)

`/sdapi/v1/img2img` 에 `denoising_strength: 0` 과 SAM3 Mask 를 실어 이미 만든 이미지에 SAM3 만 돌리는 호출자(UR_IV 의
단독 SAM3·Refine 등)는 SAM3 state 에 `"sam3_source_image": "init"` 을 넣으세요.

```json
"alwayson_scripts": {"SAM3 Mask": {"args": [{"sam3_enable": true, "sam3_prompt": "face", "sam3_source_image": "init"}]}}
```

- 이 키가 없으면 SAM3 는 부모 img2img 패스의 **출력**을 검출·인페인트합니다. denoise 0 이어도 그 출력은 VAE 인코드·디코드를
  거쳐 원본과 픽셀이 조금씩 달라서, 결과의 마스크 밖도 원본과 다릅니다(측정: 마스크 밖 픽셀의 약 90% 가 바뀜, 평균 차이 약
  2.6, 최대 139).
- `init` 이면 img2img init 이미지를 씁니다 — Forge 가 VAE 에 넣을 때처럼 투명한 부분을 `img2img_background_color` 로
  채운 것입니다. 마스크 밖은 원본 그대로이고, 마스크를 못 찾거나 Mask only 이거나 SAM3 가 실패해도 결과는 원본입니다.
- Forge 설정 **img2img 색 보정**(`img2img_color_correction`)이 켜져 있으면 Forge 는 스크립트 뒤에 이미지 전체를 init
  기준으로 LAB 히스토그램 맞춤합니다 — 원본에 해도 LAB 왕복으로 픽셀 대부분이 조금씩 바뀝니다. 원본으로 돌 때는 맞출 대상이
  곧 그 원본이라 이 보정을 끕니다(인페인트 패스는 자기 색 보정을 따로 합니다). 한 요청에 init 이미지가 여러 장이고 원본으로
  돌 수 없는 장(크기가 다르거나 읽을 수 없음)이 섞였으면, 그 장의 보정을 빼앗지 않도록 끄지 않고 예전처럼 출력을 씁니다
  (`output (color correction)`).
- 부모 패스가 이미지를 바꾸려던 경우에는 예전처럼 출력을 씁니다: init 이미지 없음(txt2img), 인페인트 마스크, denoise > 0,
  얼굴 복원, 크기가 다름(하이레스·VAE 2x·크기 조정).
- infotext `SAM3 Source` 가 어느 쪽으로 돌았는지 알립니다: `init image`, 또는 요청했지만 출력을 썼으면 `output (<이유>)`
  (`not img2img` / `inpaint mask` / `denoising > 0` / `face restoration` / `size WxH != WxH` /
  `init image unreadable` / `color correction`). 요청하지 않으면 이 키가 없습니다.
- `Sam3Args`(모르는 키는 검증 실패) 밖에서 state 로만 읽는 키라, 이 키를 모르는 예전 빌드도 오류 없이 무시합니다(그 결과에는
  `SAM3 Source` 가 없습니다). Forge UI·🎯 빠른 버튼·XYZ 는 이 키를 쓰지 않으므로 동작이 같습니다.

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

## 별도 기능: Anima VAE DeGrid (NAFNet)

Anima(Qwen·Wan VAE)로 만든 이미지에 생기는 **VAE 격자 무늬**를
[DraconicDragon/NAFNet-VAE-DeGrid](https://huggingface.co/DraconicDragon/NAFNet-VAE-DeGrid)(Apache-2.0, Civitai 의
"Qwen VAE DeGrid NAFNet 1x" 와 같은 파일) 모델로 지웁니다. txt2img·img2img 의 **Anima VAE DeGrid (NAFNet)** 아코디언을
켜면 이미지마다 **모든 후처리가 끝난 뒤, 저장 직전에** 한 번 적용하고, **Extras** 탭에서 이미 만든 이미지(한 장·배치·폴더)에도
쓸 수 있습니다. 기본은 꺼짐입니다.

### 준비

- 모델 파일을 `models/ESRGAN/` 또는 `models/DeGrid/`(새로 만들어도 됨)에 넣습니다. 권장은 v1.1
  (`VAE_DeGrid_NAFNet_small_v1.1.safetensors`, Civitai 이름 `qwenVAEDegridNafnet_v11.safetensors`, 117 MB, SHA-256
  `e6f59053acb3…ff470d4e`)입니다(같은 저장소의 v1.0 도 같은 NAFNet-small 구조라 목록에 나오지만, 실제로 돌려 확인한 것은
  v1.1). 목록에는 **state dict 가 NAFNet 인 파일만** 나옵니다 — 같은 폴더의 일반 업스케일러는 safetensors 헤더·새 형식(zip)
  `.pth` 의 `data.pkl`(둘 다 텐서는 읽지 않음, `torch.load` 도 부르지 않음)로 걸러 내고, 옛 형식 `.pth`(zip 이 아닌 pickle —
  예: `4x-UltraSharp.pth`)는 DeGrid 가 아니므로 열지 않고 건너뜁니다. 고른 모델은 Forge 가 감싸기 전 로더
  (`torch.load_origin`·`load_file_origin`)로 읽으므로, 읽다 실패해도 Forge 가 파일 이름을 `.corrupted` 로 바꾸지 않습니다.
  목록 맨 앞(드롭다운 기본값·API 에서 모델을 비웠을 때)은 metadata 의 `modelspec.version` 이 가장 높은 파일입니다(v1.1 은 `1.1` 을 적어 둠 — 버전을 적지 않은 파일은 그 뒤, 폴더·이름 순). 새 파일은 🔄 로 목록에
  들어옵니다.
- ⚠️ **DeGrid(잔차) NAFNet 만** 쓸 수 있습니다. 이미지를 내는 일반 복원 NAFNet(SIDD 노이즈 제거·GoPro 디블러 등)도 키가 같아
  목록에 나오지만(SIDD width 32 는 구성까지 같음), 고르면 출력이 잔차가 아니라 입력을 따라가는 이미지라서 적용하지 않고
  `Anima DeGrid error: not a DeGrid residual model: … output does not look like a residual …` 을 남깁니다(출력 |평균|
  2/255 초과이면서 입력과의 상관 0.9 초과, 또는 |평균| 25/255 초과이면서 상관 0.5 초과이거나 입력이 한 색, 또는 출력 밝기가
  입력 밝기를 따라감 — |출력 평균| 25/255 초과·한 부호로 쏠림(|평균| ≥ 0.5×평균|출력|)·채널별 평균의 투영 비 0.5 초과. 마지막
  것은 회색 입자 바탕·화면을 채운 스크린톤처럼 잔 결이 입력 분산의 대부분이라 흐림·median 같은 이미지 모델의 상관이 0.3 안팎으로
  낮아지는 경우를 거릅니다. 실제 v1.1·Anzhc 파인튜닝은 Anima 이미지에서 |평균| 0.2~0.5/255, 상관 -0.1~+0.45, 투영 비 ±0.04 이내).
- 화면을 채운 **잔 스크린톤(4px 망점)·1px 체커** 같은 무늬는 DeGrid 가 격자로 보고 누릅니다. 잔차가 31~57/255 로 커지지만
  입력과 반대로(상관 -0.5~-0.9) 움직이는 잔차라 거절하지 않고 적용합니다(ComfyUI 노드와 같음 — CPU 실측 512², 강도 1:
  4px 스크린톤 대비가 v1.1 ×0.68, Anzhc ×0.55). 스크린톤을 살리려면 그 이미지에서는 DeGrid 를 끄거나 강도를 낮추세요.
- 화면을 채운 **1px 줄무늬·3px 세로줄·저대비(118/138) 1px 줄무늬**처럼 학습에 없던 무늬에서는 DeGrid 잔차가 **폭주**합니다
  (|평균| 183~2143/255 — 그대로 더하면 PSNR 4~8 dB). 잔차 |평균| 이 **100/255** 를 넘으면 그 이미지는 DeGrid 없이 원본을 저장하고
  `Anima DeGrid error: output blew up (mean |residual| …/255 > 100/255 - …)` 을 남기며 콘솔에도 적습니다. 이 문턱은 **극단적인
  전체 폭주만** 거르는 어림값입니다 — 1px 체커·디더링(bayer) 이미지처럼 학습에 없던 무늬는 이 문턱 아래(60~94/255)에서도
  크게 망가진 채 적용될 수 있습니다(ComfyUI 노드도 같음). 그런 이미지에는 DeGrid 를 끄세요. 이미지 전체의 평균으로 보므로
  그런 무늬가 화면의 1/4 쯤이면 건너뛰지만, 더 작은 조각(512² 안의 128²)이면 그 부분만 망가진 채 적용될 수 있습니다.
  실제 Anima 이미지의 잔차는 0.2~3/255 입니다. 이미지 모델 검사도 아주 어두운 이미지(평균 25/255 아래)에서는 잘못 고른
  일반 복원 NAFNet 을 놓칠 수 있습니다.
- ⚠️ 이 모델은 이미지가 아니라 **잔차**를 냅니다. ESRGAN 폴더에 두면 Forge 의 Hires fix·Extras Upscale 업스케일러 목록에도
  보이지만, 거기서 고르면 잔차만 남은 거의 검은 이미지가 나옵니다(Forge 업스케일러는 출력을 [0,1] 로 자르고 BGR 로
  넣습니다). 이 기능으로만 쓰세요. 헷갈리면 `models/DeGrid/` 에 두세요.
- 설치·업데이트 뒤에는 Forge 를 **재시작**하세요(새 스크립트·설정 섹션·`metadata.ini` 콜백 순서).

### 사용

| 칸 | 뜻 |
|---|---|
| DeGrid 모델 | 찾은 NAFNet 파일(🔄 새로 고침). 없으면 `None` 이고 켜도 건너뜁니다 |
| 적용 방식 | **Full** 잔차 전체 — 어두운·밝은 격자 모두(기본) · **Dark Pixels Mainly** 양의 잔차만 — ComfyUI 기본 노드 경로(Load Upscale Model → Upscale Image → Image Blend)와 같고 Nyquist Notch 셰이더와 비슷 · **Bright Pixels Mainly** 음의 잔차만 |
| 강도 | 잔차에 곱하는 배율 0~1.5 (1 = 원본 노드) |
| 타일 크기 | 512 = 원본 노드(겹침 32, 가장자리 feather). 0 = 나누지 않음(VRAM 더 씀). 슬라이더는 0 과 128 단위(128~4096). NAFNet 의 채널 어텐션이 타일 평균을 써서 타일 크기에 따라 결과가 조금 다릅니다(512 와 나누지 않음의 차이: 8비트로 최대 2 단계, 픽셀 9% 가 1~2 단계) — 그래서 infotext 에는 **실제로 쓴** 타일(메모리 부족으로 줄였으면 줄인 값)이 남습니다 |

- 계산: `결과 = clamp(이미지 + 강도 × f(잔차), 0, 1)` — f 는 방식(전체 / 양수만 / 음수만). 잔차와 중간 합은 자르지 않고
  마지막 이미지만 자릅니다. 강도 1·같은 타일이면 식·타일 위치·feather 가중치가
  [ComfyUI-NAFNet-Residual](https://github.com/DraconicDragon/ComfyUI-NAFNet-Residual) 의 **NAFNet Restoration** 노드와
  같습니다(`tests/test_vae_degrid.py` 가 같은 입력 텐서로 노드의 잔차 함수·ComfyUI `tiled_scale` 과 비트 단위 대조). 전체
  파이프라인은 비트 단위로 같지 않습니다 — 노드는 입력을 channels-last 뷰(`image.movedim(-1,-3)`)로 넣어 float 결과가 1e-6
  정도(실제 v1.1 CPU 실측 최대 1.9e-6, 8비트로는 50만 값 중 많아야 수십 개) 다르고, 8비트 변환은 여기서 반올림, ComfyUI SaveImage 는
  버림이라 저장한 PNG 는 값의 약 절반이 1 단계 다릅니다. 입력은 RGB [0,1] fp32 입니다.
- 16 의 배수가 아닌 크기: 노드(spandrel NAFNet)는 안에서 모자란 칸을 **0** 으로 채워 오른쪽·아래 가장자리 잔차가 커집니다
  (실제 v1.1 CPU: Anima 250×190 조각 오른쪽 아래 56/255, 1000×1526 오른쪽 끝 45/255). 여기서는 타일마다 반사 패딩으로 16 의
  배수를 맞추고 잘라 냅니다(저자의 단독 추론 스크립트 NAFNet-c `infer.py` 와 같은 반사) — 같은 조각이 1.6/255 로 안쪽과
  비슷해집니다. 표준 Anima 크기(1216×1856 등 16 의 배수, 1.5 배 hires 포함)는 타일 512·256·128 의 모든 조각이 16 의 배수라
  모델을 그대로 불러 노드와 같습니다(1216×1856 은 고치기 전과 비트 단위로 같음). 주로 Extras 의 임의 크기 이미지와 8 의
  배수 img2img 크기에 해당합니다.
- 효과는 미세합니다. 개발 PC CPU 실측(1216×1856 Anima 이미지): 잔차 |평균| 약 0.5/255, 최대 약 25/255, Full 로 8비트 값이
  바뀐 픽셀 약 44%(대부분 1 단계, 선화 가장자리 주변이 가장 큼). VAE 를 거친 적 없는 합성 그림(512², 단색 면·선화)을
  Qwen-Image VAE 로 인코드·디코드한 뒤 돌리면 원본과의 PSNR 이 36.7 → 37.7 dB 로 올랐고(강도 1 이 0.5·1.5 보다 좋음,
  잔차를 반대 부호로 더하면 34.5 dB 로 나빠짐), VAE 를 거치지 않은 그림 자체는 거의 그대로 둡니다(PSNR 60 dB 이상).

### 언제 도는가

- 이미지마다 ADetailer·SAM3 in-flight 인페인트·img2img-hires-fix 등 **모든** always-on 스크립트의 `postprocess_image` 와
  img2img 색 보정·인페인트 합성이 끝난 뒤(`postprocess_image_after_composite`), 저장·infotext 직전에 돕니다. Forge
  `modules/processing.py` 가 이 순서로 부르므로 확장 설치 순서·폴더 이름과 무관합니다(테스트가 Forge 코드로 확인).
- SAM3 인페인트·ADetailer 의 내부 패스에서는 돌지 않고, 합친 최종 이미지에 한 번만 적용합니다.
- 콘솔에는 작업마다 `[AnimaDeGrid] 켬 …` 한 줄, 이미지마다 결과 요약 한 줄이 찍힙니다. ADetailer 는 배치의 마지막 장마다 모든
  스크립트의 `process` 를 복사본에 다시 부르는데(`p.scripts.process(copy(p))`), 그때는 `켬` 줄을 다시 찍지 않습니다(DeGrid 는
  원래도 이미지마다 한 번만 적용).
- Settings → Postprocessing 에서 Extras 의 Upscale 을 txt2img·img2img 탭에 켰다면 DeGrid 가 그보다 **먼저** 돕니다
  (`metadata.ini` 콜백 순서 — 확대하면 격자 간격이 달라짐). Extras 항목의 이름은 **Anima VAE DeGrid (NAFNet, Extras)** 로
  생성 탭 아코디언과 달라 API 이름이 겹치지 않지만, 그것을 메인 탭에도 켜면 두 번 적용되니 켜지 마세요.
- img2img 인페인트('Inpaint only masked' 포함)는 합성한 전체 이미지에 적용합니다 — 마스크 밖 원본도 지나가지만, 격자가
  없는 부분은 거의 바뀌지 않습니다.
- Anima Tile & Repair·캐릭터 레퍼런스 결과는 Forge 생성 파이프라인 밖의 별도 잡이라 이 아코디언이 붙지 않습니다 —
  Extras 탭의 DeGrid 로 처리하세요.

### 장치·메모리 (Settings → **SAM Extra VAE DeGrid**)

| 설정 | 기본 | 뜻 |
|---|---|---|
| `sam3_degrid_device` | auto | auto = Forge 가 쓰는 GPU, cpu = VRAM 을 쓰지 않음(학습과 같이 쓸 때 등, 1216×1856 한 장 약 6.5 초 — 개발 PC) |
| `sam3_degrid_gpu_precision` | fp32 | fp32 = ComfyUI 노드와 같은 계산. fp16 = fp32 가중치 + fp16 autocast(모델이 fp16 AMP 로 학습됨, 넘친 타일은 fp32 로 다시). CPU 는 늘 fp32 |
| `sam3_degrid_keep_loaded` | 끔 | 끄면 이미지마다 Forge `load_models_gpu` 로 올렸다 내리고 VRAM 캐시를 비웁니다. 켜면 Forge 메모리 관리에 맡겨 남깁니다(약 117 MB) |

- GPU 실측(RTX 5090, 1216×1856, 타일 512):
  - Forge 생성 탭의 DeGrid 단계는 한 장 0.77 초(같은 모델)~0.96 초(v1.1 ↔ Anzhc 로 바꿔 파일에서 다시 읽음)입니다 — 모델을
    올리고 내리는 시간 포함, 그때 기본이던 fp16 autocast 로 잰 값. Forge 를 켜고 처음 한 번은 모델 읽기·CUDA/cudnn 초기화로
    4.1 초. 요청 전체(약 41 초)에서는 요청 사이의 흔들림에 묻힙니다.
  - 계산만(동기화해 잼): v1.1 fp32 0.27~0.30 초, fp16 autocast 0.34~0.38 초(Anzhc 파인튜닝은 0.31~0.36 · 0.38~0.46 초). 이
    GPU 에서는 fp32 가 더 빨라서 기본이 fp32 입니다.
  - VRAM(torch): 가중치 111 MiB + 최대 활성 fp32 292 MiB · fp16 276 MiB, 합계 약 0.4 GB. fp16 autocast 는 VRAM 을 거의 줄이지
    않습니다 — VRAM 을 아끼려면 `sam3_degrid_device` 를 cpu 로. 모자라면(OOM) 타일을 반씩 줄여 128 까지 다시 합니다(노드와
    같음). `--novram` 처럼 Forge 가 모델을 일부만 올리면 직접 옮깁니다.
- fp16 autocast 와 fp32 의 차이: GPU(RTX 5090) 에서 잔차 최대 0.32~0.43/255 — 8비트 결과로는 픽셀 2~18% 가 1 단계이고 그보다
  큰 차이는 없습니다. CPU 로 재면 최대 0.27/255(평균 0.04/255)입니다. fp16 autocast 는 옛 GPU 에서 더 빠를 수 있어
  선택지로 남겼습니다(측정 안 함).
- TF32: fp32 는 합성곱의 TF32 를 torch 설정 그대로 둡니다 — `torch.backends.cudnn.allow_tf32` 는 torch 기본이 켬이고 Forge 도
  바꾸지 않습니다(ComfyUI 노드도 같은 기본으로 돎). 프로세스 전역 설정이라 DeGrid 만 따로 바꾸면 Forge 모델에도 번지고, 엄격
  fp32(TF32 끔)와는 잔차가 최대 0.11/255 다를 뿐 속도도 같아서(둘 다 0.275 초) 건드리지 않습니다.
- 설정 키가 `sam3_degrid_precision`(기본 fp16) 에서 `sam3_degrid_gpu_precision`(기본 fp32) 으로 바뀌었습니다. Forge 는 설정을
  저장할 때 기본값까지 `config.json` 에 적어 두어, 같은 키로는 새 기본이 이미 쓰던 설치에 먹지 않기 때문입니다. 예전 키는
  읽지 않으니 fp16 autocast 를 쓰려면 Settings 에서 다시 고르세요.
- bf16 은 쓰지 않습니다 — CPU 실측에서 잔차 오차(평균 0.36/255)가 잔차 크기와 비슷했습니다(fp16 은 0.04/255).

### 기록과 붙여 넣기

infotext 에 `Anima DeGrid model`, `Anima DeGrid mode`(Full / Dark Pixels Mainly / Bright Pixels Mainly),
`Anima DeGrid strength`, `Anima DeGrid tile`(**실제로 쓴** 타일 — 메모리 부족으로 512 → 256 → 128 로 줄였으면 줄인 값),
`Anima DeGrid precision`(`fp32` / `fp16-autocast` — 8비트 결과로 많아야 1 단계 다름, 설정이라 붙여 넣지는 않음)이 남습니다. 모델이
없거나, 이미지를 내는 모델이거나, 잔차가 폭주했거나, 실패하면 이미지는 DeGrid 없이 저장되고 `Anima DeGrid error` 만 남습니다
(앞의 셋은 `model not found: …` · `not a DeGrid residual model: …` · `output blew up (…)` 문구만, 그 밖의 오류는 예외 이름을 붙여). PNG Info 로 붙여 넣으면 켜짐·방식·강도·타일과
(그 PC 에 있으면) 모델이 되살아나고, 이 키가 없는 이미지를 붙여 넣으면 꺼집니다. Extras 결과는 PNG 의 `postprocessing` 항목에
같은 키가 남습니다.

### API

`alwayson_scripts["Anima VAE DeGrid (NAFNet)"] = {"args": [true, "qwenVAEDegridNafnet_v11", "full", 1.0, 512]}` —
위치 인자 `[enabled, model, mode, strength, tile]`(뒤는 빼면 기본값: 첫 NAFNet 파일 · full · 1.0 · 512). 키 이름을 적은
dict 하나도 받습니다: `{"args": [{"enabled": true, "mode": "dark"}]}`. `model` 이 비었거나 `"None"`/`"auto"` 면 찾은 첫
NAFNet 파일, `mode` 는 `full` / `dark` / `bright` 또는 `Full` / `Dark Pixels Mainly` / `Bright Pixels Mainly`, `strength` 0~1.5,
`tile` 0 또는 128~4096(1~127 은 128). 모델을 못 찾으면 로그를 남기고 건너뜁니다(`Anima DeGrid error: model not found: …`). Forge 의 Extras
API(`/sdapi/v1/extra-*`)는 내장 항목(Upscale 등) 인자만 넘기므로 Extras DeGrid 는 UI 전용입니다.

---

## 별도 기능: Extra Schedulers

Forge 의 **Schedule type**(과 Hires schedule type) 목록에 스케줄러 6개를 더합니다. 스케줄러는 스텝마다 쓸 노이즈
크기(시그마)를 정하는 것이라 샘플러와 따로 고릅니다. 아래 p 는 스텝 위치 i / (n − 1)(0 → 1)입니다.

| Schedule type | 곡선 |
|---|---|
| **Cosine** | σmin + ½(σmax − σmin)(1 − cos(π(1 − √p))) — 처음 내려가는 폭이 작음 |
| **CosineExponential blend** | (1 − p)·Cosine + p·Exponential — Cosine 으로 시작해 Exponential 의 긴 꼬리로 끝남 |
| **Phi** | σmin + (σmax − σmin)(1 − p)^(φ²), φ = 황금비(착상: Extraltodeus 의 [Golden Scheduler](https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler)) |
| **Laplace** | ComfyUI `get_sigmas_laplace`([arXiv:2407.03297](https://arxiv.org/abs/2407.03297)) — 시그마가 e^μ 근처에 모임. 아코디언의 μ/β. 노드의 clamp 가 같은 시그마를 되풀이하면 같은 곡선 중 sigma min~max 안의 구간에 스텝을 고르게 다시 놓음(되풀이가 없으면 노드와 비트 단위로 같음) |
| **Karras Dynamic** | Karras 램프에 스텝마다 지수 ρ + 2cos(2πi/n)(ρ 기본 7, Karras 와 같이 쓰는 Settings → Sampler Parameters → rho — `--adv-samplers` 일 때 보임, 붙여 넣은 `Schedule rho` 로도 바뀜). ρ 는 2 보다 커야 하고, 약 4 보다 작아 시그마가 도중에 올라가면 생성이 오류로 멈춤. 이 변형의 출처는 미확인 |
| **custom** | 아코디언에 적은 식 또는 시그마 목록 |

- **시그마가 그대로인 스텝은 쓰지 않습니다(6개 모두)**: 마지막 0 앞에서 같은 시그마가 두 번 이어지면 생성이
  `ExtraSchedulerError`(custom 은 `CustomSchedulerError`, Karras Dynamic 은 `KarrasDynamicError`)로 멈춥니다. 그런 스텝은 Euler
  에서는 버려지는 스텝이고, Res Multistep·DPM++ 2M 같은 multistep 샘플러는 길이 0 인 스텝으로 나눠 NaN(검은 이미지)을 냅니다.
  평평한 구간이 있는 custom 식(상수, sigma min 에 붙잡아 두는 `max(m, …)` 등), 같은 값이 이어지는 custom 목록, sigma min 과
  sigma max 를 같게 바꾼 설정도 여기에 걸립니다. 올라가는 custom 스케줄은 그대로 쓰고, Karras Dynamic 은 올라가는 스텝과
  제자리 스텝을 모두 거절합니다.
- **flow 모델에는 맞지 않는 스케줄러**: Cosine · CosineExponential blend · Phi · Karras Dynamic 과 M~m 사이를 보간하는 custom
  식(예: `m + (M - m) * (1 - x) ** 2`)은 Anima 3.8B(flow)에서 모두 물 빠진 듯 대비가 낮고 뿌연 이미지를 냈습니다. Forge 자체의
  Karras·Exponential 도 똑같습니다(평균 밝기 ≈217·표준편차 ≈46, Linear Quadratic 은 ≈182·≈90). 모델의 시간 shift 없이 sigma
  min~max 사이의 시그마 공간을 나누므로, 구도와 대비가 잡히는 σ = 1 근처를 한두 스텝 만에 지나가기 때문입니다. 버그가 아니라
  SD·SDXL 계열(eps/v) 모델용 스케줄러입니다. flow 모델에는 Simple · Beta · Linear Quadratic, Laplace(기본 μ 0) 또는 시간 shift 를
  넣은 custom 식을 쓰세요 — 예: `m + (M - m) * 3 * (1 - x) / (1 + 2 * (1 - x))`(shift 3 Simple 에 가까움 — Anima 에서 정상
  이미지, Res Multistep 에서는 잔 입자가 조금).

### Extra Schedulers 아코디언 (custom · Laplace 전용)

txt2img 의 ANIMA 튜닝 열(Anima 3.8B 바로 아래, img2img 는 스크립트 영역)에 접힌 **Extra Schedulers** 아코디언이 있습니다. 켜기
체크박스는 없고, 생성이 `custom` 이나 `Laplace` 를 쓸 때만 그 값이 쓰입니다.

| 칸 | 뜻 |
|---|---|
| custom: mode | Expression(식) / Sigma list(목록) |
| custom: expression | 스텝마다 계산할 식. 변수 `m`(sigma min) · `M`(sigma max) · `n`(스텝 수) · `s`(이번 스텝, 0부터) · `x`(= s / (n − 1)), 상수 `phi` · `pi` · `e`, 연산 `+ − * / **`, 함수 `abs sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil min max`. 기본 `M * (m / M) ** x`(Exponential 과 같음) |
| custom: sigma list | `[1.0, 0.6, 0.25, 0.1, 0.0]` 처럼 숫자 목록(쉼표·공백, 2~1000개). 1.0 으로 시작해 0.0 으로 끝나면 sigma max~min 으로 늘려 쓰고, 그 밖에는 시그마 값 그대로(끝의 0.0 은 마지막 0). 개수가 스텝 수와 다르면 Forge 의 로그-선형 보간(Align Your Steps 와 같음)으로 맞춤 |
| Laplace mu / beta | ComfyUI LaplaceScheduler 와 같은 값·범위(기본 0 / 0.5, μ −10~10, β 0~10) |

- 식의 값은 스텝마다 0 보다 커야 하고(마지막 0 은 자동), Anima·Flux 같은 flow 모델에서는 `M`(= 1)을 넘지 않게 쓰세요.
  끝에서 두 번째 시그마를 버리는 샘플러(DPM2·DPM++ 3M SDE·UniPC)는 Forge 가 스텝 + 1 개를 요청하므로 `n` 도 하나 큽니다.
- 식은 파이썬 코드로 실행하지 않습니다. AST 화이트리스트로 숫자·위 변수·연산·함수만 받고, 속성 접근·`__import__`·람다·
  컴프리헨션·문자열·비교 등은 계산 전에 거절합니다. 지수는 ±64, 식은 500자·200 노드·깊이 32 까지이고 모든 중간값이 유한해야
  합니다. PNG 를 붙여 넣어 들어온 식도 같은 계산기로만 읽습니다.
- 잘못된 식·목록이면 생성이 `Extra Schedulers: the custom scheduler's … cannot be used: …` 오류로 멈춥니다(다른 스케줄로 조용히
  바꾸지 않음).
- Laplace: ComfyUI 노드는 곡선을 x = 0~1 에서 고르게 찍어 [sigma min, sigma max] 로 자르므로(clamp), β 가 크거나 e^μ 가 sigma
  max/min 에 닿거나 그 밖이면 처음이나 끝에서 같은 시그마가 되풀이됩니다. flow 모델(sigma max 1)은 기본 μ 0(e^0 = 1)에서 앞쪽
  절반의 스텝이 모두 정확히 1 입니다. 이 확장은 그렇게 되풀이될 때 n 스텝을 같은 곡선 중 [sigma min, sigma max] 안에 드는 구간에
  고르게 다시 놓습니다 — 값은 모두 그 곡선 위에 있고, 되풀이가 없으면 노드 값과 비트 단위로 같습니다. β 가 0 이거나 곡선 전체가
  그 범위 밖이면 μ/β 를 적은 `ExtraSchedulerError` 로 생성이 멈춥니다.
  - Anima 3.8B(28 스텝, 시드 11, 832×1216): 노드처럼 자른 스케줄은 Res Multistep 에서 완전히 검은 이미지(NaN — 길이 0 인
    스텝으로 나눔), Euler 에서는 정상 이미지지만 28 스텝 중 14 스텝이 버려졌습니다. 다시 놓은 스케줄(1.0, 0.98, …, 0.19,
    0.0032)은 두 샘플러 모두 정상 이미지입니다.
  - Laplace 에는 Euler 계열 샘플러를 쓰세요. 위 Anima 설정에서 곡선의 마지막 스텝은 ≈0.19 에서 sigma min(≈0.003)으로 크게
    뛰는데, 2차 multistep 샘플러인 Res Multistep 은 이 스텝을 잘 다루지 못해 이미지는 멀쩡해도 잔 입자가 보입니다(Euler 는 깨끗함).
  - flow 모델에서도 μ 는 기본 0 그대로 두세요. 음수 μ(−1.5·−2 등)는 곡선이 한두 스텝 만에 1 에서 0.3~0.8 근처로 떨어져 Anima 에서
    물 빠진 이미지가 나왔습니다.

### 기록과 붙여 넣기 · XYZ · API

- infotext: Forge 가 `Schedule type`(·`Hires schedule type`, Karras Dynamic 에 rho 를 바꿨으면 `Schedule rho`)을 남기고, 이 확장이
  `Custom scheduler expression` 또는 `Custom scheduler sigmas`(custom 을 쓴 생성만), `Laplace mu`·`Laplace beta`(Laplace 를 쓴
  생성만)를 남깁니다. PNG Info 로 붙여 넣으면 되살아나고, 이 스케줄러를 쓰지 않은 이미지는 아코디언 값을 건드리지 않습니다.
- XYZ: `[Extra Schedulers (sam-extra)] Laplace mu` · `Laplace beta` · `Custom expression` · `Custom sigma list`(식·목록 축은 그
  방식으로 바꿈, 쉼표가 든 값은 큰따옴표로 감쌈 — 잘못된 식·목록과 슬라이더 범위 밖의 μ/β 는 그리드를 시작하기 전에 알려 줌).
  스케줄러 자체는 Forge 의 `Schedule type` 축.
- API: `"scheduler": "Laplace"`(이름 `laplace` 도 됨), `alwayson_scripts["Extra Schedulers (sam-extra)"] = {"args": ["expression",
  "M * (m / M) ** x", "[1.0, 0.6, 0.25, 0.1, 0.0]", 0.0, 0.5]}` — 위치 인자 `[custom_mode, custom_expression, custom_sigmas,
  laplace_mu, laplace_beta]` 또는 그 키의 dict 하나(빼면 기본값). 스크립트 이름은 아코디언 이름(`Extra Schedulers`)과 달리
  `(sam-extra)` 가 붙습니다 — API 는 스크립트를 이름으로 찾는데, aoleg/Neo_ExtraSchedulers 에 `Extra Schedulers` 아코디언이 있습니다.
- 이름이 같은 스케줄러가 이미 있으면(Forge 나 다른 확장, 예: aoleg/Neo_ExtraSchedulers 를 같이 설치) 그 라벨은 더하지 않고 콘솔에
  한 번 알립니다. Settings → Hide Schedulers 로 숨길 수 있습니다(재시작).
- 라벨은 라이선스 없는 [aoleg/Neo_ExtraSchedulers](https://github.com/aoleg/Neo_ExtraSchedulers) 의 infotext 와 맞췄지만 그 코드는
  쓰지 않은 재구현입니다. 그 README 에 적힌 이름(`cosine` · `cosine-exponential blend` · `phi` · `Laplace` · `Karras Dynamic` ·
  `custom`, 소문자 `laplace` · `karras dynamic` 도)으로 된 Schedule type 은 이 스케줄러로 되살아나지만, 식·Laplace 값의 infotext
  키는 다를 수 있어 그 값은 되살아나지 않을 수 있습니다.

---

## 별도 기능: Extra Samplers (샘플러 5종)

Forge 샘플러 목록(Sampler · Hires sampler 드롭다운, XYZ `Sampler` 축, API `sampler_name`)에 다섯 샘플러를 더합니다.
이름은 [aoleg/Neo_ExtraSchedulers](https://github.com/aoleg/Neo_ExtraSchedulers) 와 같아 infotext 가 서로 붙습니다
(그 저장소는 라이선스가 없어 코드는 쓰지 않았습니다). 같은 이름이 이미 등록돼 있으면 그쪽을 두고 이 확장은 건너뜁니다.

| 샘플러 | 무엇 | 쓰는 Forge 값 |
|---|---|---|
| `ER SDE (Reverse-time)` | Forge 내장 ER-SDE 풀이에 잡음 척도 h(λ)=λ^(η+1) (ComfyUI `SamplerER_SDE`). η=0 이면 ODE | Sigma noise |
| `ER SDE (ODE)` | 같은 풀이, h(λ)=λ — 잡음 없음(결정적) | — |
| `DPM++ 4M SDE` | DPM++ 3M SDE 에 이력 하나를 더한 4차 다단계 SDE (Clybius). flow 모델 지원 | Eta, Sigma noise |
| `Euler Dy CFG++` | 2·3번째 스텝에 반 해상도 보조 스텝(Koishi-Star Dy) + CFG++ | Sigma churn/tmin/tmax/noise |
| `Euler SMEA Dy CFG++` | 0번째 스텝 ×1.25 보조 스텝, 1번째 스텝 반 해상도 보조 스텝 + CFG++ | Sigma churn/tmin/tmax/noise |

- **Extra Samplers 아코디언**(txt2img 는 ANIMA 튜닝 열, img2img 는 스크립트 영역, 기본 접힘): `ER SDE max stage`(1–3, 기본 3 —
  1 = 지수 Euler, 2·3 = 다단계 보정)와 `ER SDE eta`(0–10, 기본 1 — Reverse-time 의 지수, 클수록 스텝마다 잡음↑)는 **ER SDE 두
  항목에만** 쓰입니다. Forge 내장 **ER SDE** 와 전역 Eta 설정은 이 값과 상관없습니다. "Forge 값" 은 Settings → Sampler
  Parameters(`--adv-samplers`)의 값입니다.
- **CFG++ 두 항목은 CFG 1~2**: 더 높으면 Forge 처럼 콘솔에 권장 경고가 남습니다. churn 은 k-diffusion 규칙(`min`)이라
  sigma churn 0(기본)에서는 다시 잡음을 넣지 않습니다(원본 Euler Dy 는 매 스텝 √2배로 다시 잡음을 넣는 `max` 규칙 — 이
  확장은 따르지 않음). flow 모델에서는 churn 을 eps 등가 잡음 수준 σ/(1−σ) 에서 계산해 σ 가 1 을 넘지 않습니다.
- **보조 스텝**: Dy 는 모델을 2번, SMEA Dy 는 2번(그중 하나는 1.25배 해상도 — VRAM 을 더 씀) 더 부릅니다. 인페인트
  마스크·인페인트 모델 조건·Anima 레퍼런스 latent 는 보조 스텝 해상도로 맞췄다가 되돌립니다. Hires 패스에서도 같습니다.
  보조 스텝은 자기 스텝으로 세어 Forge 의 스텝 카운터를 늘리지 않으므로 프롬프트 편집·Skip Early CFG·리파이너 스텝 전환이
  Euler CFG++ 와 같은 스텝에서 일어납니다.
- **보조 스텝을 쓰지 않는 경우**: Forge 의 **Spectrum Integrated** 가 켜져 있을 때(그 예측기가 해상도 변화를 따라가지 못해,
  보조 스텝을 그대로 돌리면 기본 설정에서 Euler Dy 가 오류로 멈춤), Wan 2.2 I2V(`concat_latent`)·PiD(`lq_latent`)·다른 확장의
  `extra_concat_condition` 이 있을 때. 이때 두 샘플러는 보조 스텝 없이 일반 CFG++ 스텝(= Euler CFG++)으로 돌고, 콘솔에 이유를
  한 번 남기며 infotext 에 `Extra Samplers status: dy sub-steps skipped (Spectrum)` 처럼 기록합니다.
- **가이던스와 함께**: 보조 평가(`transformer_options["sam_extra_substep"]` 표시)는 HiFlow 기록·정렬, Momentum·HiGS 이력, SMC 의
  이전 오차, APG 모멘텀, RDC 이동 평균을 바꾸지도 지우지도 않습니다. PAG·CFG base·DCW·TSR 은 보조 평가에도 그대로 걸립니다
  ([docs/GUIDANCE.md](docs/GUIDANCE.md) 조합 원칙).
- **Anima SPEED·Colorcraft 와 함께**: SPEED 는 샘플러를 해상도 구간마다 따로 부르면서 구간의 첫 스텝 번호(패스 기준)를
  `sam_extra_step_offset` 으로 넘기고, 두 샘플러는 그 번호로 보조 스텝 자리를 정합니다. 그래서 보조 스텝은 SPEED 없이 돌 때와 같은
  스텝(Dy 는 2·3, SMEA Dy 는 0·1)에서 한 번씩, 그때의 격자(저해상도 구간이면 저해상도 격자)에서 돌고, 구간마다 다시 돌거나 SMEA 의
  ×1.25 보조 스텝이 SPEED 의 확장 바로 뒤에 돌지 않습니다. Spectrum Integrated 가 켜져 있으면 SPEED 와 보조 스텝이 둘 다 물러나
  각자 status 에 이유를 남깁니다. Colorcraft 는 보조 평가를 보정하지 않고 넘겨 스텝마다 한 번만 보정합니다(Colorcraft status 에
  `Dy/SMEA sub-steps passed through xN`).
- **기록과 붙여 넣기**: `ER SDE max stage`(3 이 아닐 때)·`ER SDE eta`(1 이 아닐 때, 또는 Forge 의 Eta 설정이 1 이 아닐 때 —
  그때는 다른 패스의 샘플러가 남긴 Forge `Eta` 가 같은 infotext 에 있을 수 있어 1 도 적습니다). 둘 다 없는 infotext 를 붙이면
  기본값으로 돌아갑니다. aoleg 확장의 Reverse-time infotext 는 η 를 Forge 의 `Eta` 로 남기므로, `ER SDE eta` 가 없고 Sampler 가
  `ER SDE (Reverse-time)` 면 `Eta` 를 읽습니다.
- **XYZ**: `[Extra Samplers] ER SDE max stage`, `[Extra Samplers] ER SDE eta`.
- **API**: `alwayson_scripts: {"Extra Samplers": {"args": [max_stage, eta]}}` (생략 가능, 기본 3·1.0).
- **한계**: SMEA Dy 의 ×1.25 보조 스텝은 Hires 패스에서도 돌아 VRAM 을 더 씁니다. 이 확장이 남긴 `ER SDE eta` 는 aoleg 확장이
  읽지 않아 그쪽에서는 η 가 되살아나지 않습니다(반대 방향은 됨). `ER SDE eta` 가 크면 결과가 깨질 수 있습니다(ComfyUI 노드의
  툴팁과 같은 경고). DPM++ 4M SDE 의 momentum 과 Koishi-Star 의 다른 샘플러(Euler Negative 등)는 옮기지 않았습니다.
- GPU 는 Anima 3.8B(Linear Quadratic 28 스텝)에서만 확인했습니다. ER SDE (ODE)·Euler Dy CFG++·Euler SMEA Dy CFG++ 는 정상
  이미지이고 Dy·SMEA 는 Forge 의 Euler CFG++ 와 거의 같습니다. ER SDE (Reverse-time) 은 채도가 높은 다른 그림체, DPM++ 4M SDE 는
  얼굴이 단순해지고 가장자리에 세로선이 생기는데, Forge 자체 DPM++ 2M SDE·3M SDE 도 Anima 에서 회색조·가장자리 선이 나오므로
  Anima 와 SDE 잡음 주입의 궁합으로 봅니다. SDXL 은 확인하지 않았습니다.
- 출처: ComfyUI(GPL-3.0) `SamplerER_SDE`, [Clybius/ComfyUI-Extra-Samplers](https://github.com/Clybius/ComfyUI-Extra-Samplers)
  (BSD-3-Clause), [Koishi-Star/Euler-Smea-Dyn-Sampler](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler) (Apache-2.0).
  고지는 THIRD_PARTY_NOTICES.md.

---

## 별도 기능: Anima SPEED (저해상도 선행 샘플링)

[SPEED](https://arxiv.org/abs/2605.18736)(Spectral Progressive Diffusion)를 Forge 샘플러에 얹습니다. txt2img(ANIMA 튜닝 열)·img2img
의 **Anima SPEED (저해상도 선행 샘플링 · 실험)** 아코디언을 켜면, 노이즈가 지배적인 초반 스텝을 **DCT 로 줄인 저해상도 latent**(기본
가로·세로 0.5배 = 토큰 1/4)에서 돌리고, 전환 σ 에서 DCT 저주파 블록을 원래 크기 격자에 넣고 새 고주파를 σ 크기 노이즈로 채운 뒤
κ = r/(1+(r−1)σ) 로 보정해 **고른 샘플러로 이어서** 샘플링합니다. 기본은 꺼짐입니다.

- **빨라지지만 이미지가 달라집니다.** 손실 없는 가속이 아닙니다. 저해상도 구간은 구도만 잡고, 잔 디테일은 전환 뒤 원래 크기에서
  만들어집니다. 전환이 늦을수록(σ 가 낮을수록) 빠르고 디테일 위험이 큽니다.
- **flow 모델 전용**(Anima·Flux·Krea 2·Z-Image·Wan 등 σ ∈ (0, 1]). SDXL·SD1.x 는 건너뜁니다.
- Anima 3.8B 실측(832×1216, Res Multistep + Linear Quadratic 28 스텝, 시드 11): 기본값은 5/28 스텝이 저해상도로 약 1.13배
  빠르고(11.1 s → 9.8 s, 첫 SPEED 실행은 워밍업으로 이득 없음), manual σ 0.7 은 22/28 스텝으로 약 1.8배(6.1 s) 빠르지만 구도·머리색이
  크게 바뀝니다. 저해상도 스텝은 토큰이 1/4 이어도 약 3배만 빠릅니다(3.8B 는 가중치 읽기에 묶임). Linear Quadratic 은 앞 절반을
  σ 0.975 위에서 보내므로 기본값의 저해상도 구간이 짧습니다. 토큰 수 기준 어림(1024² 32 스텝 Beta: 기본 1.1~1.2배, σ 0.7
  1.6~1.8배)과 맞습니다.

### 방식과 전환 σ

| 칸 | 뜻 |
|---|---|
| Mode | **transition**(기본) = 공식 howardhx/speed·aoleg/ComfyUI-SPEED: 전환 스텝의 σ 만 정렬값 σ·κ 로 바꾸고 나머지 스케줄은 그대로, 저해상도 격자는 `round(s·H)`. **respace** = sorryhyun/ComfyUI-Spectrum-KSampler: 남은 σ 를 모두 `σ̃/σ` 배로 다시 배치하고 저해상도 격자를 짝수로 맞춤 — 데스크톱 앱 ComfyUI 팩(SpectrumSPDKSampler)과 같은 SPD 기하(기하만 같음, 홀수 latent 는 다름) |
| Transition sigma | **neo_shift**(기본) = 프리셋 σ* ÷ divisor(Forge 의 고정 shift 용, aoleg 측정 1.03) · **delta_optimal** = 프리셋 σ* 그대로 · **manual** = Manual sigma(s) |
| Spectrum preset | 잠재 공간 파워 스펙트럼 P(ω)=A·ω^−β. **anima**(기본, 1024² 기준 σ* 0.96, 스펙트럼 유도·미검증) · flux·flux2(측정) · krea-2·krea-2-raw·z-image·wan21 · custom(아래 A·β) |
| Scales | 1.0 으로 끝나는 해상도 비율(예: `0.5,1.0`, `0.25,0.5,1.0`) |
| Delta (δ) | 노이즈 지배 허용치(논문 식 9). 작을수록 일찍 전환 — 품질↑ 속도↓ |
| Sigma divisor | neo_shift 에서 σ* 를 나누는 값. 1.0 = delta_optimal 과 같음 |
| Manual sigma(s) | 전환마다 σ 하나(transition 모드는 내림차순). respace 모드에서 하나만 적으면 모든 전환에 씀(원본과 같음) |
| Adaptive delta | 1024 px 기준 σ* 와 스텝 비율에 고정 — 해상도가 커져도 저해상도 구간이 늘지 않음(aoleg) |
| Apply to Hires pass | 기본 꺼짐. 켜면 Hires 패스에도 적용(시작 σ 가 전환 σ 보다 높을 때만) |
| Transform (세부값) | dct(기본)·fft = 아무 비율, dwt = 단계마다 정확히 2배 |
| Spectral noise seed (세부값) | −1 = 이미지 시드. 고정하면 모든 이미지가 같은 확장 노이즈 |

데스크톱 앱 ComfyUI 팩의 SPEED(spd_scale 0.5, spd_sigma 0.7)와 같은 기하는 Mode `respace` · Transition sigma `manual` ·
Manual sigma `0.7` · Scales `0.5,1.0` · 샘플러 Euler 입니다. 앱 노드는 원래 크기 꼬리에 sorryhyun 의 Spectrum 캐싱을 함께 돌리므로
픽셀까지 같지는 않습니다(기하만 같음). 홀수 latent(예: 1000 px → 125)는 앱이 샘플링 전에 짝수로 채워 격자부터 다릅니다.

### 언제 쉬는가

`Anima SPEED status` 에 이유를 남기고 그 패스는 순정으로 돕니다: 마스크·인페인트(SAM3·ADetailer 인페인트 패스 포함), Anima 레퍼런스
latent(Anima Edit·img2img 참조 — `anima_do_reference` 를 끄면 씀), 다른 모델의 레퍼런스 latent(Flux Kontext·Flux.2 Klein·
Qwen-Image-Edit·Krea 2 — 원래 크기 그대로 토큰 뒤에 붙어 저해상도 격자와 위치가 맞지 않음), Wan 2.2 I2V 조건 latent(concat_latent),
PiD 의 저화질 입력 latent(lq_latent), ControlNet(Anima LLLite 포함), flow 가 아닌 모델, Forge 내장
**Spectrum Integrated**(멈춤 지점이 스텝 수에서 나와 크기가 바뀌는 순간 저해상도 텐서를 예측할 수 있음), Restart·UniPC·σ 목록이
없는 샘플러, 저해상도 스텝이 없는 패스(img2img·Hires 가 전환 σ 이하에서 시작 — 원본은 이때 시작 latent 의 고주파를 노이즈로 바꾸고
σ 를 올렸음), 마지막 전환이 스케줄 안에 없을 때(원본은 작은 latent 를 냄), Hires 패스(체크 꺼짐).

### 원본과 다른 점

- 배치: 이미지마다 자기 시드로 확장 노이즈(transition = numpy `default_rng(시드 + (k+1)·10000)`, respace = 장치의
  `torch.Generator(시드 + 10000)`)와 저해상도 구간의 SDE·ancestral 노이즈(Forge 의 이미지별 생성기, 난수 소스 설정 그대로),
  Brownian 노이즈(이미지별 BrownianTree)를 뽑습니다. 배치의 각 이미지 = 같은 시드를 혼자 만든 결과(CPU 테스트 기준 — GPU 에선
  새 기능을 다 꺼도 배치 크기에 따라 커널이 달라 조금 다르고, 구도가 저해상도 구간에서 정해지는 SPEED 는 그 차이를 키웁니다).
  배치 1 장은 원본과 같은 난수를 씁니다.
- FFT 고주파 노이즈의 `/√2` 를 뺐습니다(공식 ca7801c9 의 수정 — 빼지 않으면 새 고주파의 분산이 σ²/2).
- img2img·Hires: DCT 로 자른 시작 latent 는 이미지 성분이 √(HW/hw) 배 커지고 노이즈는 그대로입니다. 설정
  `sam3_speed_img2img_rescale`(기본 켬)이 이미지 성분을 원래 크기로 줄이고 모자란 노이즈를 시드별로 채워 (1−σ)·이미지 + σ·노이즈
  형태로 맞춥니다. κ 가 기대는 진폭 관계(원래 크기 격자에 넣은 저해상도 이미지는 r 배 약해짐)의 역이라, 끄면 저해상도 스텝과 κ 뒤의
  원래 크기 스텝이 r 배 센 이미지 성분으로 시작합니다. 끄면 원본과 같습니다. txt2img 는 어느 쪽이든 같습니다.
- DWT 는 단계 비율 r = 2 가 아니면 생성 전에 거절합니다(공식은 저해상도 스텝을 돈 뒤 전환에서 오류).
- 실행 동안 Forge 의 `sampling_sigmas` 를 바뀐 σ 목록으로 바꿔 Detail Daemon·DAVE·Momentum·HiGS·HiFlow·Colorcraft 가 전환 뒤 σ 를
  스케줄에서 찾고, 끝나면 되돌립니다. 바꾼 목록은 새 텐서라 원래 목록의 대신임을 표시합니다(`sam3ext/guidance/sigmas.py`
  `mark_republished`) — 목록 객체로 자기 샘플링 실행을 알아보는 Colorcraft 가 그 표시를 따라가 원래 크기 꼬리에서도 보정을 이어 갑니다.
- 샘플러가 `sam_extra_step_offset` 인자를 받으면 구간마다 그 구간의 첫 스텝 번호(패스 기준)를 넘깁니다. Euler (SMEA) Dy CFG++ 는
  이 번호로 보조 스텝을 SPEED 없이 돌 때와 같은 스텝(Dy 2·3, SMEA Dy 0·1)에, 그때의 격자에서 한 번씩 돌립니다 — 구간마다 다시 돌거나
  SMEA 의 ×1.25 보조 스텝이 확장 바로 뒤에 돌지 않습니다. 두 기능은 함께 돌고, Forge 의 Spectrum Integrated 가 켜져 있으면 둘 다
  물러나 각자 status(`Anima SPEED status`·`Extra Samplers status`)에 이유를 남깁니다.
- DCT 는 scipy 없이 장치 위 float64 행렬(크기별 캐시)로 계산합니다.
- 생성 도중 중단하면 저해상도 구간이었어도 원래 크기 latent 를 돌려줍니다. SPEED 자체가 실패하면 순정 샘플링으로 다시 돌고
  status 에 남깁니다.

### 기록·붙여 넣기·API

infotext `Anima SPEED`(설정 — 붙여 넣으면 되살아나고, 없는 이미지를 붙여 넣으면 꺼짐), `Anima SPEED status`(패스마다 적용 내용:
저해상도 스텝 수·격자·전환 σ, 또는 건너뛴 이유), `Anima SPEED img2img rescale`. 콘솔 `[AnimaSPEED]` 줄은 설정 `sam3_speed_log` 로 끕니다.
XYZ 축: `[Anima SPEED] Enable / Mode / Manual sigma / Delta / Sigma divisor / Scale`.
API: `alwayson_scripts["Anima SPEED"] = {"args": [true, "transition", "neo_shift", "anima", "0.5,1.0", 0.01, 1.03, "0.7", true, "dct", 203.615097, 1.915461, -1, false]}`
— 위치 인자 `[enabled, mode, threshold, preset, scales, delta, sigma_divisor, manual_sigmas, adaptive, transform, spectrum_A, spectrum_beta, seed, hires]`,
또는 dict 하나 `{"args": [{"enabled": true, "mode": "respace", "threshold": "manual", "manual_sigmas": "0.7"}]}`.

---

## 별도 기능: Colorcraft (latent 색 보정)

샘플링 중 **매 모델 호출의 CFG 결과(x0)** 를 latent 의 색 기저 벡터 방향으로 밀어 노출·대비·디테일·색을 조정합니다 —
후처리가 아니라 이미지가 만들어지는 동안 손대므로, 일찍 걸면 구도·밝기 같은 생성 방향이 바뀌고 늦게 걸면 색 보정처럼
동작합니다. [muerrilla/ComfyUI-Colorcraft](https://github.com/muerrilla/ComfyUI-Colorcraft)(`d28ac6a`)의 계산을 그대로
옮겼고 Forge 구조와 Flux2 벡터는 [aoleg 포크](https://github.com/aoleg/ComfyUI-Colorcraft)(`f00066c`)를 따릅니다. 추가
모델 호출은 없고 기본은 꺼짐입니다. txt2img·img2img 의 **Colorcraft (latent 색 보정 · ComfyUI-Colorcraft)** 아코디언.

### 지원 모델

| 벡터 | latent format(Forge) | 모델 |
|---|---|---|
| krea2 | Wan21 | Anima, Qwen-Image, Krea 2, Wan 2.1 |
| zimage | Flux | Flux, Chroma, Lumina 2, Z-Image |
| flux2 (포크) | Flux2 | Flux 2 Klein 4B/9B, ERNIE-Image |
| 없음 | SD15, SDXL 등 | **Contrast 와 Color Shift 만** — 나머지(노출·색·디테일·Chroma Plus·마스크)는 꺼지고 status 에 이유 |

### 사용

- **Enable Colorcraft** 를 켜고 탭 **I**(처음부터 Active)의 값을 움직입니다. 값은 작게(0.1~0.3) 두고 구간을 넓히는 편이
  안전합니다 — 원본 안내대로 너무 세거나 마지막 스텝까지 걸면 latent 가 깨질 수 있습니다.
- **CFG++ 샘플러에서는 세기를 절반쯤**: Euler CFG++·Euler (SMEA) Dy CFG++ 같은 CFG++ 샘플러에서는 같은 값이 더 세게 걸립니다.
  CFG++ 스텝은 보정된 x0 를 스텝의 일부만큼이 아니라 통째로 다음 latent 에 싣기 때문입니다. CFG++ 의 방식이지 sam-extra 의 버그가
  아니며, 원본 ComfyUI-Colorcraft 도 같은 post-CFG 지점에 겁니다. Anima 3.8B(시드 11, 28 스텝)에서 노출 +0.3 이 평균 밝기를 Forge 의
  Euler CFG++ 로 약 +27(186 → 214 — Euler Dy CFG++·Euler SMEA Dy CFG++ 도 같음), Res Multistep 으로 +14, Euler 로 +8 올렸습니다.
- **탭 I~X**: 켠(Active) 탭이 위에서부터 차례로 적용되는 수정자 스택입니다. **Type** 은 원본 노드(Advanced 전체 ·
  Basic 대비·색 이동 · Luma 노출·톤 · Chroma 색온도·틴트·바이브런스·채도·채도 대비 · Chroma Plus Lab·대각 축 · Punch
  대비·클래리티·샤프니스 · Shift 색 이동)이고 그 노드의 계산 순서와 칸만 보입니다. Advanced 의 **Apply Chroma Plus /
  Apply Color Shift** 는 노드의 게이트(기본 켬 — 원본 Forge 탭과 같음), **Dev** 는 모델별 보정값(recenter·max chroma·
  chroma plane) 덮어쓰기입니다.
- **스케줄**: Strength(−1~1)·Start·End(스텝 비율). **Advanced Schedule** 을 켜면 Exponent·Bias·Start/End Offset·Smooth.
  매 호출의 σ 로 스케줄 위치를 찾습니다(원본 노드와 같음). img2img·Hires 는 실제로 도는 σ 구간으로 스케줄을 만듭니다.
- **Pass**: Base(기본 패스·img2img) / Hires(Hires 패스만) / Both. SAM3·ADetailer 내부 패스에서는 돌지 않습니다.
- **Anima SPEED·Euler (SMEA) Dy CFG++ 와 함께**: SPEED 를 켜면 저해상도 구간의 x0 와 전환 뒤 원래 크기의 x0 를 모두 보정합니다.
  SPEED 가 전환마다 바꿔 다시 내놓는 σ 목록(같은 실행 표시가 붙음)을 따라가 그 목록에서 σ 를 찾으므로, 정렬된 전환 σ 와 다시 배치된
  꼬리에서도 스텝마다 제 스케줄 값을 읽습니다(Detail Daemon·DAVE·MG/HiGS·HiFlow 와 같은 방식). 저해상도 격자에서도 마스크 Blur 는
  이미지 픽셀 기준 크기를 지킵니다(Forge 가 넘기는 패스의 latent 격자로 비율을 잼). Euler (SMEA) Dy CFG++ 의 보조 스텝(반 해상도·
  ×1.25)은 보정하지 않고 넘겨 스텝마다 한 번만 보정합니다 — 보조 스텝까지 보정하면 그 스텝이 옮긴 화소에 같은 스텝의 보정이 두 번
  걸립니다.
- **Masking**: M1~M10 은 x0 에서 바로 읽는 축 마스크(exposure·hue·saturation·temperature·tint·대각·Lab·clarity·sharpness,
  highs/lows/split/range/protect range, Blur→Spread→Normalize→Contrast), C1~C5 는 퍼지 조합(and·or·subtract·xor, 앞 조합만
  참조). 탭의 **Mask ➜** 로 고릅니다(Basic 은 마스크 없음). A·B 가 다 차지 않은 조합을 고르면 마스크 없이 적용됩니다(원본과 같음).
- **Debug**: **Capture debug latent** 를 켜고 생성하면 **Debug Step** 의 편집 전 x0 를 잡아 두고, **Refresh Debug Images** 가
  고른 축 투영·마스크·조합을 아코디언 안 갤러리에 그립니다(지금 마스크 칸 값으로 — 다시 생성하지 않고 마스크를 다듬기).
  Composite 색을 고르면 VAE 로 디코드해 겹칩니다(GPU).
- 슬라이더 숫자 칸은 범위 밖 값도 받습니다(원본 Forge 확장과 같음, `javascript/colorcraft_sliders.js`).

### 기록과 붙여 넣기

- `SAM Extra Colorcraft`: 켠 탭과 그 탭 Type 의 기본값이 아닌 값(`v1;mods=I,II;I.exposure=0.3;…`). 붙여 넣으면 패널 전체가
  그 값으로 바뀌고, 이 키가 없는 이미지는 Colorcraft 를 끕니다. 원본·포크 확장이 남긴 `Colorcraft` 키도 읽습니다(포크의
  Chroma Center 는 지금 원본 눈금으로 바꾸고, 바이브런스는 값만 같고 지금 계산으로 렌더).
- `SAM Extra Colorcraft status`: 패스마다 벡터 family·latent format, 적용 탭, 호출 수·실제로 바꾼 호출 수와 알림(벡터 없음,
  색 기준점 실패, 덜 찬 조합, 넘긴 Dy/SMEA 보조 스텝 수 `Dy/SMEA sub-steps passed through xN`, 다른 샘플링 실행, 오류 → 입력
  그대로).
- `SAM Extra Colorcraft pre-DD sigma`: Detail Daemon 이 σ 를 바꾼 생성에만.

### Settings → **SAM Extra Colorcraft**

| 설정 | 기본 | 뜻 |
|---|---|---|
| `sam3_colorcraft_pre_dd_sigma` | 켬 | Detail Daemon 과 함께: 스케줄 위치를 Detail Daemon 이 바꾸기 전 σ 로 찾기. 끄면 ComfyUI 에서 두 노드를 이은 것과 같음 |
| `sam3_colorcraft_log` | 끔 | 패스마다 σ 목록과 탭별 스케줄 값을 콘솔에 |

### XYZ · API

- XYZ 축(탭 I 대상, 축을 쓰면 탭 I 이 켜짐): `[Colorcraft] Enable, Strength, Start, End, Exposure, Tone Compression,
  Contrast, Clarity, Sharpness, Temperature, Tint, Vibrance, Saturation, Chroma Contrast`.
- API: `alwayson_scripts["Colorcraft (sam-extra)"] = {"args": [...]}` — 위치 인자 579개(`enabled, masking`, 탭 I~X 각 44칸,
  M1~M10 각 10칸, C1~C5 각 7칸, `debug, debug_step`). 빠진 뒤쪽 인자는 기본값입니다. 첫 인자 하나로 줄여도 됩니다:
  infotext 값 그대로(`{"args": ["v1;mods=I;I.exposure=0.3"]}`) 또는 인자 경로 사전(`{"args": [{"I.exposure": 0.3}]}`,
  `enabled` 생략 = 켬).

### 한계

- GPU 는 Anima 3.8B 에서만 확인했습니다: 노출 +0.3 은 평균 밝기를, 색온도 +0.3 은 R−B 를, 채도 −0.5 는 채도를 맞는 방향으로
  움직였고, SPEED·Dy/SMEA 와 같이 돌 때도 보정 횟수가 단독과 같았습니다. VRAM 은 따로 재지 않았고 다른 모델의 화질은 확인하지
  않았습니다(원본 노드와의 비트 대조는 CPU). flux2 보정값은 포크가 예전 원본 보정
  기준으로 잰 값이고, clarity·sharpness 마스크 축의 flux2 정규화(4.0)는 원본 값을 그대로 썼습니다.
- 패널이 큽니다. 탭마다 스크립트 인자 579개, Gradio 블록 약 1,200개로 원본 Forge 패널과 비슷한 무게이고, 두 탭을 합치면
  페이지 설정(`/config`)이 약 0.96 MB 늘어납니다. 접힌 아코디언도 DOM 에는 그려집니다(Gradio 4.40).
- 원본 Forge 확장의 스케줄·마스크 그래프(캔버스)와 탭 색 표시는 옮기지 않았습니다.

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
- **NegPiP(이제 [내장](#별도-기능-negpip-내장))와 함께 쓸 수 있다 (설치 순서 무관).** v2 조건은 Forge 네이티브 계약(줄마다 텐서 하나)을 지키고 run id 는 마지막 토큰 행의 마커로 실어 보내므로 `get_learned_conditioning` 을 감싸는 확장이 있어도 죽지 않는다. NegPiP 이 우리 아래에 깔리는 순서에서는 NegPiP 의 가중치 마스킹을 우리가 대신 적용해 어느 순서에서도 결과가 같다(내장 NegPiP 기준 — 따로 설치한 sd-forge-negpip 가 아래에 깔리면 내장 마스크 규칙을 쓰므로 새 Forge 의 emphasis `None`·`Ignore` 에서는 그 판 단독과 다르다). 패치는 생성이 끝나도 자리에 남되 꺼진 상태로 투명하게 위임하므로, 두 확장이 서로 다른 순서로 해제해도 사고가 없다.

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
| 1열 | Forge 기본 설정 → **고정**(핀 꽂은 것) → **ANIMA 튜닝**(3.8B·Detail Daemon·Skimmed CFG·Guidance·Optimal Scale·Colorcraft·VAE 2x·SPEED·Extra Schedulers·Extra Samplers) |
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

## 별도 기능: 진행 막대 (Smooth Progress)

Settings → **SAM Extra Progress Bar** → **부드러운 진행 막대 사용** 을 켜고 **Apply settings** 를 누르면(기본 꺼짐)
txt2img·img2img 갤러리 위에 이 확장의 진행 막대가 생기고 Forge 기본 막대는 숨겨집니다.
[diamfang/sd-webui-smooth-progress](https://github.com/diamfang/sd-webui-smooth-progress)(MIT)를 옮긴 것입니다.

- **무엇을 보여 주나**: 막대와 % 는 작업 전체(배치·Hires 패스 포함, Forge 기본 막대와 같은 계산), ETA 는 작업
  전체의 남은 시간입니다. 스텝은 지금 패스 기준이고, 패스가 여럿이면 `[2/4] 12/28 • 45% • 01:10` 처럼 몇 번째
  패스인지 앞에 붙습니다. 대기 중에는 Forge 의 대기열 글자(`In queue: 1/2`)를 그대로 보여 줍니다.
- **그 탭의 작업만**: 그 페이지의 그 탭에서 시작한 작업만 그립니다. 다른 창이나 다른 탭에서 시작한 작업은 나오지
  않고, 서버에는 작업이 도는 동안에만(페이지가 보일 때) 0.2초마다 묻습니다.
- **설정**(저장하면 바로 반영): 부드러움(Smooth > / ~ / < Accurate), 글자 형식 8가지, 글자 위치(0~100%),
  끝난 뒤(막대와 글자 서서히 숨김 / 글자만 / 그대로), 숨기는 시간(0.1~4초), 중단했을 때(글자 / 빨간 글자 / 빨간
  막대 / 둘 다), 높이(10~50px), 색(테마 강조색·성공색·Blue·Green·Red·Dandelion·직접 지정).
- **중단**: Interrupt(또는 Esc)로 멈추면 멈춘 자리에 `중단됨` 이 남습니다. Skip 은 다음 배치로 넘어갈 뿐이라
  중단으로 치지 않고, "Don't Interrupt in the middle" 로 그 이미지까지만 멈췄는데 마지막 이미지였으면 완료로
  표시합니다.
- **움직임 줄이기**: OS 의 애니메이션 줄이기(prefers-reduced-motion)가 켜져 있으면 막대가 보간 없이 서버 값으로만
  움직입니다.
- **함께 쓰기**: sd-webui-smooth-progress 를 따로 설치해 두었으면 이 막대는 켜지지 않습니다. Forge 기본 막대는
  지우지 않고 이 막대가 그리는 작업 동안만 숨기므로(켜기 전에 시작한 작업은 Forge 막대로 보임) 그 막대를 기다리는
  sd-webui-api-payload-display 도 그대로 동작합니다.
- **API**: `GET /sam-extra/progress?id_task=task(...)` — 다른 sam-extra 경로처럼 헤더 `X-SAM3-Notebook: 1` 과
  Forge 로그인(`--gradio-auth`)·`--api-auth` 가 필요합니다. 응답 필드는 `sam3ext/progress_api.py` 머리에 있습니다.
- **한계**: 오류(OOM 등)로 끝난 작업도 완료(100%)처럼 끝납니다(두 진행 경로 모두 실패를 알리지 않음). ADetailer·SAM3 가
  작업 도중 패스를 더하는 동안은 99.2% 에 머물 수 있습니다. ETA 는 첫 스텝의 모델·조건 준비 시간이 섞여 몇 스텝 뒤에
  자리를 잡습니다(상류와 같음). Extras·모델 병합·확장 설치는 Forge 기본 막대 그대로이고, 켜 두는 동안 막대 자리는 늘
  차지합니다(화면이 밀리지 않게, 상류와 같음).

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

## 별도 기능: 구도 · 카메라 (프롬프트로 시점 잡기)

txt2img·img2img 의 스타일 줄 아래에 접힌 칸 **구도 · 카메라** 가 있습니다(기본 배치에서는 Generate 버튼 옆 열, txt2img 는
**TIPO 프롬프트 확장** 칸 다음 — Compact 배치에서는 프롬프트 아래 스타일 열). 방향·높이·거리·기울기·화면 내 위치를 고르면 그에
맞는 태그·구도 문구를 미리 보여 주고, **메인 태그에 추가** 를 누를 때만 메인 프롬프트 끝에 붙입니다. 실제 3D 카메라 제어가
아니라 프롬프트로 유도하는 것이라 모델에 따라 결과가 달라집니다. 사용자 앱(UR_IV)의 같은 칸을 옮긴 것으로 태그 계산과 화면
글자는 앱과 같습니다.

| 조절 | 범위 | 만드는 태그 |
| --- | --- | --- |
| 방향 | −180~180° | 20° 까지 `facing viewer` · 65° 미만 `three-quarter view from the subject's left/right` · 115° 까지 `from side`, `profile`, `view from the subject's left/right` · 160° 미만 `from behind`, `rear three-quarter view …` · 그 너머 `from behind` |
| 높이 | −75~75° | 20° 이상 `from above`(60° 이상이면 `bird's eye view` 도) · −20° 이하 `from below` |
| 거리 · 크롭 | 0~100% | 15 까지 `close-up` · 30 `portrait` · 50 `upper body` · 65 `cowboy shot` · 85 `full body` · 그 너머 `wide shot` |
| 기울기 | −30~30° | ±6° 이상이면 `dutch angle`, `frame tilted counterclockwise/clockwise` |
| 화면 내 인물 위치 | −100~100% | ±25 안이면 `centered composition`, 밖이면 `subject on the left/right side of the frame` |

- **조작**: 프리셋 다섯 개(정면 상반신 · 낮은 시점 · 전신 · 위에서 · 근접 · 측면 · 여백 · 후면 · 원경), 궤도 그림 끌기(터치 포함 —
  끌다가 취소되면 시작 상태로), 그림에 포커스를 두고 방향키(5°, Shift 15°, Home 은 정면), 슬라이더. 조작만으로는 프롬프트가
  바뀌지 않습니다.
- **메인 태그에 추가**: 프롬프트에 아직 없는 태그만 끝에 붙이고 적어 둔 글자는 그대로 둡니다. 같은 태그는 대소문자·밑줄·가중치를
  가리지 않고 알아봅니다(`(From_Above:1.2)` 가 있으면 `from above` 를 붙이지 않음). 붙인 뒤 커서는 프롬프트 끝에 있고 **Ctrl+Z
  한 번**에 되돌아갑니다. 다 들어 있으면 버튼이 **이미 포함된 구도** 로 꺼집니다.
- **태그 자동완성과 함께**: 붙인 글자를 타이핑으로 알리지 않아(Forge 의 붙여넣기 버튼처럼 `updateInput` 으로 Gradio 에 알림)
  tagcomplete 의 추천 목록이 뜨지 않습니다 — 목록이 떠 있으면 Enter/Tab 이 방금 붙인 마지막 단어를 추천어로 바꿀 수 있어서입니다.
- **충돌 경고**: 반대 태그(`from below` ↔ `from above`, `facing viewer` ↔ `from behind`, 크롭끼리, 화면 위치끼리)가 이미 있으면
  알려 주고 버튼이 **기존 구도 유지하고 추가** 로 바뀝니다. 기존 태그는 지우지 않습니다. 미리보기와 경고는 프롬프트를 칠 때마다
  바뀝니다.
- **기억**: 조작값은 탭마다 이 브라우저에 기억합니다(localStorage). 칸은 늘 접힌 채로 시작합니다.
- **끄기**: Settings → **SAM Extra Appearance** → **구도 · 카메라 칸 표시** 를 끄고 Reload UI(또는 Forge 재시작)하면 칸이 생기지
  않습니다.
- **한계**: ↙️ 붙여넣기·스타일 적용·TIPO 처럼 Gradio 가 바꾼 프롬프트는 칸을 열거나 칸에 마우스·포커스를 가져갈 때 다시 읽습니다
  (누르는 순간에는 늘 지금 프롬프트로 계산). 부정 프롬프트와 Styles 로 붙는 글자는 보지 않습니다. execCommand 를 지원하지 않는
  브라우저에서는 값을 직접 써서 넣으므로 Ctrl+Z 한 번으로 되돌아가지 않을 수 있습니다.

## 별도 기능: NegPiP (내장)

프롬프트 안에서 음수 가중치로 개념을 빼거나 강제합니다 — 긍정 프롬프트의 `(aqua hair:-1.0)` 은 그 개념을 **빼고**, 부정 프롬프트의
`(단어:-1.0)` 은 그 개념을 **강제**합니다. SD1·SDXL·Anima 에서 돕니다. [Haoming02/sd-forge-negpip](https://github.com/Haoming02/sd-forge-negpip)
(AGPL-3.0-or-later, 2026-09-30 보관)의 마지막 판 `0585496` 을 이 확장에 넣은 것이라 따로 설치할 필요가 없습니다.

- **켜고 끄기 없음**: 프롬프트에 음수 가중치가 있을 때만 돕니다(always-on, UI 없음). 켜지면 생성 정보에 `NegPiP: True` 가 남습니다.
  API 에서는 예전처럼 스크립트 제목 `NegPiP`(인자 0개)입니다.
- **sd-forge-negpip 는 지우세요**: 따로 설치한 `extensions/sd-forge-negpip` 가 함께 로드돼 있으면 NegPiP 가 두 번 걸리므로(Anima 는 서로
  상쇄) 내장 NegPiP 가 쉬고 콘솔에 `built-in NegPiP is standing down` 경고를 한 번 남깁니다. 폴더를 지우거나 Extensions 탭에서 끄고
  Forge 를 재시작하세요.
- **Forge 옛·새 텍스트 엔진 모두**: Forge `21886f41`(텍스트 엔진 재작성) 전후 어느 Forge 에서도 돕니다. Anima 마스크는 두 엔진 모두
  엔진이 조건에 곱한 가중치를 그대로 따릅니다. SD1/SDXL 은 새 엔진(`sd_engine.ClipEngine`)이 프롬프트 조각마다 BOS/EOS 를 붙여,
  상류 코드대로면 음수 항 조건에서 `[BOS, 단어…, EOS, EOS]` 행을 잡아 BOS 행까지 뒤집습니다 — 내장은 엔진에 물어 옛 엔진과 같은
  `[단어…, EOS]` 행만 고릅니다(상류와 다른 점). 행은 인코딩한 바로 그 `(단어:w)` 글자에서 세므로 `(red eyes BREAK blue hair:-1.5)`
  처럼 가중치 묶음 안의 `BREAK`(Forge 파서가 가중치 1 이 아닌 묶음에선 글자 `break` 로 남김)도 두 엔진에서 같은 행이고, 75토큰을
  넘는 항도 옛·새 엔진 모두 단어 행과 마지막 EOS 만 잡습니다(상류는 새 Forge 에서 IndexError 로 NegPiP 가 조용히 꺼지거나, 옛 Forge 에서
  채움 EOS·다음 청크 BOS 행을 잡았음). 새 Forge 는 조건 자체가 옛 Forge 와 달라 이미지까지 같지는 않습니다.
- **Anima 프롬프트 편집(`[a:b:N]`·`[a|b]`) + 긴 프롬프트**: 편집 줄들이 512 T5 토큰을 넘어 길이가 서로 다르면 상류 NegPiP 는 조건 단계에서
  `stack expects each tensor to be equal size` 로 생성이 죽었습니다. 내장은 줄마다 조건을 따로 돌려줘(Forge 순정과 같은 계약) 스텝마다
  순정처럼 그 줄의 길이를 씁니다.
- **예전 sd-forge-negpip(`b3673ce`)와 다른 점 — 음수 항 찾기**: 상류 `75b81b4` 에서 고친 패턴을 씁니다. 예전에는
  `(smile), (aqua hair:-1)` 을 통째로 한 음수 항으로 잡아 SD1/SDXL 에서 `(smile)` 까지 프롬프트에서 빠졌는데, 이제 `(aqua hair:-1)` 만
  잡습니다(앞의 escape 괄호 `\(` 부터 삼키던 경우도 고쳐짐). 같은 프롬프트라도 SD1/SDXL 결과가 달라질 수 있습니다. Anima 는 영향 없음.
- **SD1/SDXL 음수 항 (상류와 다른 점)**: 한 프롬프트의 음수 항 전부가 들어갑니다(상류는 첫 항만 — 나머지는 프롬프트에서 지워진 채 효과
  없음). 배치 안에서 항목마다 프롬프트가 다르면(sd-dynamic-prompts 등) 항목마다 제 음수 항이 붙습니다(상류는 항목 0 의 것을 모두에게).
  cond/uncond 는 Forge 가 어텐션까지 넘기는 표시로 가립니다 — VRAM 이 모자라 cond·uncond 를 따로 돌리거나, 긍정·부정 길이 차이로 따로
  돌리거나, CFG 1 이거나, DDIM/PLMS/UniPC 여도 긍정 쪽 음수 항은 cond 에, 부정 쪽은 uncond 에만 붙습니다(상류는 이런 경우 8 스텝씩
  켜졌다 꺼지거나 반대쪽에 붙음).
- **스크립트 순서**: 예전처럼 Dynamic Thresholding(`sd-dynamic-thresholding`) 뒤에 돕니다(`metadata.ini`). 상류 NegPiP 는 샘플러 이름으로
  cond/uncond 절반을 골라 Dynamic Thresholding 이 이름을 바꾼 뒤에 봐야 했습니다. 내장은 Forge 의 표시를 읽고, 샘플러 이름은 표시를
  넘기지 않는 호출자의 대체 경로(콘솔 경고 한 번)에서만 씁니다.
- **emphasis 설정 (Anima)**: 새 Forge 에서 Settings 의 emphasis 가 `None`(괄호를 글자로 읽음)이나 `Ignore`(가중치 무시)면 엔진이 음수
  가중치를 적용하지 않으므로 Anima NegPiP 는 켜지지 않고 콘솔에 `NegPiP Disabled (Emphasis: …)` 가 남습니다. `Original`·`No norm` 에서
  씁니다. 옛 Forge 는 `None` 에서만 꺼집니다. SD1/SDXL 에는 이 emphasis 판단이 없습니다(상류와 같음).
- **Forge Couple** 과는 함께 쓰지 않습니다(상류와 같이 Forge Couple 이 켜져 있으면 `NegPiP Disabled`).
- Anima 3.8B(워크플로 7)·Safe PAG·캐릭터 레퍼런스 IP-Adapter 와 함께 돕니다. 3.8B 가 NegPiP 보다 위에 설치되는 순서에서는 3.8B 런타임이
  같은 마스크를 대신 적용합니다. 아래에 깔린 것이 따로 설치한 sd-forge-negpip 여도 내장 규칙을 쓰므로, 새 Forge 의 emphasis
  `None`·`Ignore` 에서는 그 판(`0585496`) 단독과 결과가 다릅니다(내장은 뒤집는 행 없음).

## 별도 기능: MCP 서버 — 에이전트 연결

확장 안의 `mcp_server/` 는 같은 PC 의 MCP 클라이언트(Claude Code, Claude Desktop, Cursor 등)가 이 Forge 로 이미지를 만들게 하는
stdio MCP 서버입니다. [eduardoabreu81/forgeneo-mcp](https://github.com/eduardoabreu81/forgeneo-mcp)(commit `a103dc5`, MIT)를
편입했습니다. 불러온 체크포인트의 아키텍처, 과거 생성 기록, 프롬프트 방언(Anima·Illustrious·Pony 등)을 읽어 샘플링 값과 프롬프트
방식을 알려 주고, 생성한 파일의 경로를 돌려줍니다.

서버는 **Forge 안에서 돌지 않습니다.** MCP SDK 는 pydantic 2.11 이상이 필요한데 Forge 는 2.10.6 에 고정하고 실행할 때마다 다시
맞추므로, uv 가 만드는 별도 환경(`mcp_server/.venv`)에서 돕니다. Forge 는 `--api` 로 켜져 있어야 합니다.

### 등록 (한 번, 사용자 범위)

```
claude mcp add --scope user sam-extra -- uv run --project "<Forge>\extensions\forge_sam3_extension\mcp_server" sam-extra-mcp
```

- 경로는 이 확장의 `mcp_server` 폴더입니다. Settings → **SAM Extra MCP** 의 첫 항목 설명에 이 설치의 정확한 명령이 나옵니다.
- **첫 실행은 네트워크가 필요합니다** — uv 가 `mcp`·`httpx` 와 의존성을 받아 `.venv` 와 `uv.lock` 을 만듭니다(둘 다 git 이 무시).
  그 뒤로는 오프라인으로 뜹니다.
- 포트가 7860 이 아니면 `-e FORGE_URL=http://127.0.0.1:7861` 처럼 넘깁니다. `--api-auth` 를 쓰면 `-e FORGE_AUTH=사용자:비밀번호` 를
  더하되 **반드시 `--scope user`** 로 등록하세요 — 프로젝트의 `.mcp.json` 에 넣으면 저장소와 함께 퍼집니다.
- FORGE_PATH_MAP 은 필요 없습니다. 서버가 자기 위치(`<Forge>/extensions/<확장>/mcp_server`)에서 Forge 의 `config.json`·출력·models
  폴더를 찾습니다. `--ui-settings-file` 을 쓰면 `SAM_EXTRA_MCP_FORGE_CONFIG` 로 설정 파일을 알려 주세요.

### 에이전트가 바꿀 수 있는 것 (Settings → SAM Extra MCP)

| 설정 키 | 기본 | 허용하는 것 |
| --- | --- | --- |
| `sam3_mcp_allow_generate` | 켬 | `generate` — txt2img·img2img |
| `sam3_mcp_allow_model_switch` | 끔 | `models` 의 `load` — 체크포인트와 함께 프리셋·VAE·텍스트 인코더까지(인스턴스 전체) |
| `sam3_mcp_allow_interrupt` | 끔 | `progress` 의 `interrupt`·`skip` — 웹 UI 에서 시작한 생성도 멈춤 |
| `sam3_mcp_allow_download` | 끔 | `module_download` — VAE·텍스트 인코더를 models 폴더에 받음 |

서버는 도구를 부를 때마다 Forge 의 `config.json` 을 다시 읽으므로 **Apply settings 만 누르면 다음 호출부터 적용**됩니다(재시작 불필요).
꺼진 동작은 에이전트가 `confirm=True` 를 넘겨도 Forge 에 요청하기 전에 거절됩니다(`denied_by_policy`). 상태·목록·프로필·LoRA 검색과
`models` 의 `refresh`(체크포인트 목록 다시 읽기 — 불러온 모델·선택은 그대로)는 늘 허용됩니다.

### 동작

- 생성 결과는 **이 요청이 만든 파일만** 돌려줍니다. 파일에 든 생성 정보(PNG·JPEG/WebP EXIF·.txt)와 시드, 요청 시간으로 고르므로 같은
  시각 웹 UI 에서 만든 이미지가 섞이지 않습니다.
- 출력 폴더에서 찾지 못한 이미지(자동 저장을 껐을 때 등)는 Forge 응답에서 꺼내 `<Forge>/sam-extra/mcp/outputs` 에 저장합니다.
- 샘플링 값 추천의 근거인 과거 생성 기록은 생성 뒤와 5분마다 다시 읽습니다(`FORGE_HISTORY_MAX_AGE`, 즉시는 `refresh=true`).
- Anima 3.8B 의 `qwen35_4b`·`Anima-3.8B-expanded_adapter` 를 모듈 목록에 넣어 둬도 '다른 프리셋에서 남은 모듈'로 경고하지 않습니다.
- 내려받기는 받은 크기와(허깅페이스가 알려 주면) SHA256 이 맞을 때만 파일을 남기고, 있는 파일은 덮어쓰지 않습니다.
- `/sdapi/v1/cmd-flags`(실행 인자 — `--api-auth` 비밀번호 포함)는 부르지 않습니다.
- img2img 의 `init_image` 는 이 PC 의 파일만 받습니다. `\\서버\공유\...` 같은 네트워크 공유·장치 경로는 파일을 건드리기 전에
  거절합니다(Windows 가 그 서버에 접속하며 로그인 정보를 보내는 것을 막음) — 공유의 이미지는 로컬 폴더로 복사해서 넘기세요.
- 한 번의 `generate` 가 그리는 화소는 모두 합쳐 **2048×2048×4(약 16.8 MP)** 까지입니다(가로×세로, 하이레스면 확대 크기, × 배치 × 반복).
  넘으면 Forge 에 보내지 않고 거절합니다. 바꾸려면 등록할 때 `-e SAM_EXTRA_MCP_MAX_PIXELS=화소수` 를 더하세요.
- 내려받는 곳은 늘 `<Forge>/models/VAE`·`<Forge>/models/text_encoder` 입니다(`--models-dir` 류로 옮겼다면 `SAM_EXTRA_MCP_MODELS_DIR`).
- LoRA·체크포인트의 제목·태그·트리거·설명은 그 파일을 만든 사람이 쓴 글이라 길이를 잘라 `untrusted_*` 이름으로 넘기고, 에이전트에게
  지시로 따르지 말라고 알립니다. 체크포인트를 바꿀 때 넘긴 프리셋은 이 Forge 에 있는 것만 받습니다.

### 한계

- 권한은 MCP 서버가 지키는 것이지 샌드박스가 아닙니다. 셸·파일 쓰기 도구도 가진 에이전트는 `config.json` 을 고치거나 Forge API 를
  직접 부를 수 있습니다(Forge 2.29.2 의 `POST /sdapi/v1/options` 는 `restrict_api` 를 지키지 않습니다).
- 같은 PC 전용입니다. `FORGE_URL` 이 다른 컴퓨터를 가리키면 그 설정을 읽을 수 없어 모든 권한이 꺼집니다.
- 동영상(Wan) 결과는 아직 모으지 않습니다.
- 결과의 `related_files`(하이레스·ADetailer 중간 저장)는 시드와 프롬프트로 묶으므로, 같은 시각 같은 시드·프롬프트로 만든 다른 생성의
  파일도 거기 들어갈 수 있습니다(결과 파일 자체는 이 요청 것).

## 출처 / 크레딧 (Credits)

이 확장은 아래 외부 프로젝트를 **그대로 가져와(vendored, shallow clone)** Forge에 통합합니다. 핵심 기능의 저작권은 각 원저자에게 있으며, 본 확장은 Forge 통합 레이어만 제공합니다. vendor 디렉터리는 저장소에 포함되지 않고 `install.py`가 첫 실행 시 자동으로 clone합니다.

| 기능 | 원본 프로젝트 | 라이선스 | vendor 위치 |
|---|---|---|---|
| **LoRA Manager** (워크플로 4) | [willmiao/ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager) | GPL-3.0 | `lora_manager_vendor/` |
| **Anima Tile-Repair** (워크플로 3) | [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) (`anima_minimal_inference*`) | Apache-2.0 | `anima_vendor/` |
| **Anima Character Reference / ReStyler** (워크플로 6) | [Anima ReStyler workflow](https://civitai.com/models/2803070/anima-restyler) · [원 아이디어 Reddit 게시물](https://www.reddit.com/r/StableDiffusion/s/0Az0DgoaKj) | 워크플로 페이지 조건 참고 | 동작을 Forge 네이티브 img2img/reference로 재구현 (vendor·코드 복사 없음) |
| **NegPiP** (별도 기능) | [Haoming02/sd-forge-negpip](https://github.com/Haoming02/sd-forge-negpip) (commit `0585496`, 2026-09-30 보관) · 원작 hako-mikan | AGPL-3.0-or-later | `sam3ext/negpip/`·`scripts/negpip.py` 에 편입 (라이선스 전문 `sam3ext/negpip/LICENSE`, 수정 사항은 `THIRD_PARTY_NOTICES.md`) |
| **Anima 3.8B (Qwen3.5 / v2)** (워크플로 7) | [GumGum10/forge-anima-3.8B](https://github.com/GumGum10/forge-anima-3.8B) (commit `59c27e5`) | MIT | `sam3ext/anima38/` 에 편입 (저장소에 포함, 수정 사항은 `THIRD_PARTY_NOTICES.md`) |
| **TIPO 프롬프트 확장** (별도 기능) | [KBlueLeaf/TIPO-v2.1-1B-A200M](https://huggingface.co/KBlueLeaf/TIPO-v2.1-1B-A200M) · 모델 코드 [KohakUwULLM](https://github.com/KohakuBlueleaf/KohakUwULLM) | 가중치 Kohaku License 1.0 · 코드 Apache-2.0 | `sam3ext/tipo/kohaku/` 에 모델 코드만 편입, 가중치는 사용자가 받을 때 HF 에서 |
| **Anima Safe PAG** (별도 기능) | [iljung1106/comfyui-anima-safe-pag](https://github.com/iljung1106/comfyui-anima-safe-pag) (commit `905b0107`) | MIT | Anima 배치·블록 선택을 이식, σ 적용 구간·번호 파싱 편입(`sam3ext/guidance/sigma_window.py`) |
| **DCW / RDC / CWM / SMC** | [namemechan/ComfyUI-DCW](https://github.com/namemechan/ComfyUI-DCW) | GPL-3.0 | 수식 기반 Forge 재작성 (vendor 아님) |
| **DAVE** | [daheekwon/DAVE](https://github.com/daheekwon/DAVE) · [sorryhyun/ComfyUI-Anima-DAVE](https://github.com/sorryhyun/ComfyUI-Anima-DAVE) (commit `83143e8d`) | MIT | 초반 스텝 게이트 편입(`sam3ext/guidance/dave_gate.py`), block 계산은 Forge 재구현 |
| **CNS-inspired Wavelet Noise** | [namemechan/comfyui-cns_sampler_patch](https://github.com/namemechan/comfyui-cns_sampler_patch) | GPL-3.0 | `color_noise_wavelet` 편입(`sam3ext/guidance/cns.py`), Forge 훅은 재작성 |
| **Anima Modulation Guidance** | [Anzhc/Anima-Mod-Guidance-ComfyUI-Node](https://github.com/Anzhc/Anima-Mod-Guidance-ComfyUI-Node) · [quickjkee/modulation-guidance](https://github.com/quickjkee/modulation-guidance) · [yresearch/cosmos-pooled](https://huggingface.co/yresearch/cosmos-pooled) | MIT(코드 선언) / 자산 모델 카드 | Forge block 재작성 (vendor 아님) |
| **Skimmed CFG** (별도 기능) | [Extraltodeus/Skimmed_CFG](https://github.com/Extraltodeus/Skimmed_CFG) | Apache-2.0 | 수식·σ 게이트 편입(`sam3ext/guidance/skimmed_cfg.py`), Forge 훅 |
| **Anima VAE DeGrid** (별도 기능) | 모델 [DraconicDragon/NAFNet-VAE-DeGrid](https://huggingface.co/DraconicDragon/NAFNet-VAE-DeGrid) · 잔차 적용 [DraconicDragon/ComfyUI-NAFNet-Residual](https://github.com/DraconicDragon/ComfyUI-NAFNet-Residual) (commit `e15460d`) · 구조 [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) | 모델·노드 Apache-2.0 · NAFNet MIT | 잔차 모드를 다시 작성(대조 테스트에만 노드 함수 사본), 모델 구조는 Forge venv 의 spandrel, 가중치는 사용자가 받음 |
| **Detail Daemon** (별도 기능) | [muerrilla/sd-webui-detail-daemon](https://github.com/muerrilla/sd-webui-detail-daemon) (commit `1947999`) · [Jonseed/ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon) (commit `3394e44`) | MIT | schedule·σ 조회 함수 편입(`scripts/anima_detail_daemon.py`), Forge 훅 |
| **Extra Schedulers** (별도 기능) | Laplace: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) `get_sigmas_laplace` (commit `36c0b0a`) · [arXiv:2407.03297](https://arxiv.org/abs/2407.03297); Karras: [arXiv:2206.00364](https://arxiv.org/abs/2206.00364); Phi 착상: Extraltodeus [Golden Scheduler](https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler)(크레딧만); 이름: [aoleg/Neo_ExtraSchedulers](https://github.com/aoleg/Neo_ExtraSchedulers) README | Laplace GPL-3.0 · 나머지는 식 재구현 | `sam3ext/extra_schedulers/` (라이선스 없는 저장소의 코드는 쓰지 않음) |
| **Extra Samplers** (별도 기능) | [ComfyUI](https://github.com/comfyanonymous/ComfyUI) `SamplerER_SDE` (commit `36c0b0a6`) · [Clybius/ComfyUI-Extra-Samplers](https://github.com/Clybius/ComfyUI-Extra-Samplers) (commit `52eac1b7`) · [Koishi-Star/Euler-Smea-Dyn-Sampler](https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler) (commit `d98a504`) | GPL-3.0 · BSD-3-Clause · Apache-2.0 | `sam3ext/extra_samplers/` 에 잡음 척도·4M SDE·Dy/SMEA 편입(고지·수정 사항은 `THIRD_PARTY_NOTICES.md`), Forge 등록·CFG++·flow 처리는 이 확장의 코드 |
| **Anima SPEED** (별도 기능) | [howardhx/speed](https://github.com/howardhx/speed) (commit `ca7801c9`) · [aoleg/ComfyUI-SPEED](https://github.com/aoleg/ComfyUI-SPEED) (commit `a8873591`) · [sorryhyun/ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler) (commit `b46a364a`) · 논문 [arXiv:2605.18736](https://arxiv.org/abs/2605.18736) | MIT | 수식·프리셋·Forge 스크립트를 `sam3ext/speed/`·`scripts/anima_speed.py` 에 편입(torch 로 다시 작성, 대조 테스트에 원본 사본) |
| **Colorcraft** (별도 기능) | [muerrilla/ComfyUI-Colorcraft](https://github.com/muerrilla/ComfyUI-Colorcraft) (commit `d28ac6a`) · Forge Neo 구조·Flux2 벡터 [aoleg/ComfyUI-Colorcraft](https://github.com/aoleg/ComfyUI-Colorcraft) (commit `f00066c`) | MIT | 계산·벡터 편입(`sam3ext/colorcraft/`), Forge 훅·패널은 이 확장의 코드 (수정 사항은 `THIRD_PARTY_NOTICES.md`) |
| **진행 막대** (별도 기능) | [diamfang/sd-webui-smooth-progress](https://github.com/diamfang/sd-webui-smooth-progress) (commit `7fe5810`) | MIT | 스텝별 ETA·부드러움 방식·글자 형식 편입(`sam3ext/progress_api.py`·`javascript/progress_bar.js`), 시작·끝은 Forge `requestProgress` 감싸기로 다시 작성 |
| **MCP 서버** (별도 기능) | [eduardoabreu81/forgeneo-mcp](https://github.com/eduardoabreu81/forgeneo-mcp) (commit `a103dc5`) | MIT | `mcp_server/sam_extra_mcp/forgeneo/` 에 편입 (저장소에 포함, 수정 사항은 `THIRD_PARTY_NOTICES.md`). Forge 밖 uv 환경에서 실행 |
| SAM3 검출 | Meta [facebook/sam3](https://huggingface.co/facebook/sam3) ([facebookresearch/sam3](https://github.com/facebookresearch/sam3)) | SAM License(Meta) | `sam3` PyPI 패키지 (저장소에 포함하지 않음) |

저장소에 함께 들어 있는 제3자 파일(`sam3ext/anima38/`, `mcp_server/sam_extra_mcp/forgeneo/`, `sam3ext/colorcraft/` 의 계산·색 벡터,
TIPO 모델 코드, `assets/` 의 Qwen3.5 토크나이저·CLIP BPE 어휘)과 편입한 상류 함수(Skimmed CFG, Detail Daemon, Safe PAG 적용 구간,
DAVE 게이트, CNS 재색칠, 진행 막대(Smooth Progress), SPEED, Extra Schedulers 의 Laplace, Extra Samplers 의 ER SDE 잡음 척도·
DPM++ 4M SDE·Euler (SMEA) Dy 보조 스텝)의 출처와 라이선스는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 에 있습니다.

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
| [LICENSE](LICENSE) | 이 확장의 라이선스(GPL-3.0-only) 전문 — 편입한 NegPiP 파일은 [sam3ext/negpip/LICENSE](sam3ext/negpip/LICENSE)(AGPL-3.0-or-later) |

## 라이선스

이 확장의 코드는 **GNU General Public License 3판만(SPDX `GPL-3.0-only`)** 으로 배포합니다. 이후 판("or later")으로
바꿔 쓰는 것은 허용하지 않습니다. 전문은 [LICENSE](LICENSE) 에 있습니다. **예외는 편입한 NegPiP** 입니다 — 아래 둘째 항목과
같이 그 파일들은 AGPL-3.0-or-later 를 따르므로, 저장소 전체를 SPDX 로 적으면 `GPL-3.0-only AND AGPL-3.0-or-later` 입니다(`package.json`).

- 저장소에 함께 들어 있는 제3자 코드·자산(`sam3ext/anima38/` 의 MIT 코드, `mcp_server/sam_extra_mcp/forgeneo/` 의
  forgeneo-mcp 코드(MIT), `sam3ext/colorcraft/` 의 Colorcraft 코드·색 벡터(MIT), `sam3ext/speed/` 의 SPEED 코드(MIT),
  진행 막대의 Smooth Progress 코드(MIT), `sam3ext/extra_samplers/` 의 DPM++ 4M SDE 코드(BSD-3-Clause)·Euler (SMEA) Dy
  코드(Apache-2.0), TIPO 모델 코드(Apache-2.0), `assets/` 의 Qwen3.5 토크나이저(Apache-2.0)·CLIP BPE 어휘(MIT))는 원래 고지를
  그대로 유지합니다 — 목록과 조건은 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 참고.
- 편입한 NegPiP(`sam3ext/negpip/` 의 `__init__.py`·`anima.py`·`sd.py`·`utils.py`·`mask.py`, `scripts/negpip.py`)는
  **AGPL-3.0-or-later**(Copyright 2025 hako-mikan, 2026 Haoming02)를 그대로 따릅니다 — 이 확장의 GPL-3.0-only 로 바뀌지 않습니다.
  같은 폴더의 `coexist.py` 는 이 확장이 새로 쓴 코드(GPL-3.0-only)입니다. 전문은 `sam3ext/negpip/LICENSE` 에 있습니다.
  GPL-3.0 13조(AGPL-3.0 13조도 같은 뜻)에 따라 한 작업으로 결합해 배포합니다. 그 조항대로, 결합된 작업을 네트워크 너머 사용자가 원격으로
  쓰게 하면 AGPL-3.0 13조의 네트워크 상호작용 요건(그 사용자에게 대응 소스를 받을 기회 제공)이 NegPiP 부분만이 아니라 **결합된 작업
  그 자체**에 적용됩니다.
- `install.py` 가 첫 실행 때 받는 vendor(LoRA Manager: GPL-3.0, kohya-ss/sd-scripts: Apache-2.0)는 저장소에 포함되지
  않으며 각자의 라이선스를 따릅니다.
- 사용자가 따로 받는 모델 가중치(SAM3, TIPO, IP-Adapter, Modulation Guidance 어댑터 등)는 이 라이선스의 대상이 아니며
  각 모델 배포처의 조건을 따릅니다.
- Skimmed CFG 는 상류(Apache-2.0)의 수식 함수를, Detail Daemon 은 상류(MIT, muerrilla·Jonseed)의 schedule·σ 조회
  함수를, Safe PAG·DAVE 는 상류(MIT)의 적용 구간·번호 파싱과 초반 스텝 게이트를, CNS 는 상류(GPL-3.0)의
  `color_noise_wavelet` 을, Extra Schedulers 는 ComfyUI(GPL-3.0)의 `get_sigmas_laplace` 를, Extra Samplers 는 ComfyUI(GPL-3.0)
  `SamplerER_SDE` 의 ER SDE 잡음 척도를 그대로 편입했습니다(고지는 THIRD_PARTY_NOTICES.md).
