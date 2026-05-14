# sam-extra (Forge SAM3 Extension)

SAM3 / SAM3.1 마스크 + 인페인트 확장. ADetailer 스타일의 자동 검출 → 마스크 → 인페인트 파이프라인을 SAM3 텍스트 프롬프트 기반으로 수행합니다.

## 설치

```
cd <sd-webui-forge-neo>/extensions
git clone https://github.com/UR-al/sam-extra.git
```

webui 재시작 후 txt2img / img2img 패널의 "SAM3 Mask" 아코디언이 표시되면 정상입니다.

## 모델 다운로드

아래 Hugging Face 저장소에서 체크포인트를 받을 수 있습니다.

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

체크포인트가 하나도 없으면 Hugging Face의 `facebook/sam3` 에서 자동 다운로드됩니다.
완전한 오프라인 사용 시에는 `--sam3-no-huggingface` 옵션으로 자동 다운로드를 비활성화할 수 있습니다.

## 주요 기능

- SAM3 / SAM3.1 (`.pt`, `.safetensors`) 체크포인트 지원
- 텍스트 프롬프트로 검출 (`face, eyes / hand` 처럼 `,` = OR-merge, `/` = 별도 인페인트 패스)
- Combined / Individual 마스크 모드
- 인페인트 옵션 (denoising, mask blur, only-masked padding, separate width/height, steps, CFG, sampler/scheduler, noise multiplier, restore face)
- XYZ plot 축 다수: Enable, Checkpoint, Mode, Mask Mode, Device, Detect Prompt, Inpaint Prompt, Negative Prompt, Prompt S/R, Threshold, Mask Dilation, Mask Blur, Denoising Strength, CFG Scale, Steps, Inpaint Only Masked, Inpaint Padding, Inpaint Width/Height, Sampler, Scheduler, Noise Multiplier, Restore Face

## 의존성

`requirements.txt` 에 정의되어 있으며 Forge launch 시 자동 설치됩니다.

## 라이선스

내부 사용.
