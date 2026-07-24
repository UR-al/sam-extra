# 실험 기능 상태 · 진단 체크리스트 (Refine / Anima Tile-Repair)

README가 **⚠️ 실험 (아직 제대로 작동하지 않음)**으로 표시한 두 기능
(**Refine 패널**, **Anima Tile-Repair**)을 실제 Forge에서 진단할 때 쓰는 문서입니다.
코드 정적 리뷰로 잡을 수 있는 부분은 고쳤고, 나머지는 **GPU + Forge 실행이 있어야**
확인 가능합니다(테스트 환경엔 GPU/Forge가 없어 유닛·회귀 테스트까지만 검증됨).

## 이번 세션에서 고친 것 (두 기능에 직접 영향)

- **Anima VAE 2x 이중-wrap** → 원본 VAE 재-wrap 가드 (픽셀셔플 2회 = 출력 손상 방지).
- **Anima 랜덤 시드(-1) 재현성** → 명시적 시드를 직접 뽑아 infotext에 기록.
- **Refine `inherit_main_neg_prompt` 폴백 반전** → 위젯 기본값(off)과 일치.
- **unet `model_function_wrapper` 충돌** → guidance suite 우선권 + Ref PoC yield.
- **전역 tokenize/encode 전략 누수** → Anima 패스 후 `try/finally`로 복원.
- **guidance 전역 패치 teardown** → reload 시 stale 패치 제거 경로 추가.

이 중 특히 앞의 3개는 "출력이 이상하거나 재현이 안 되던" 증상을 직접 건드리므로,
재실행 시 이전과 다르게 동작할 수 있습니다.

## Refine 패널 — 전제 조건 & 확인 포인트

- 갤러리에서 이미지를 **선택**해야 함(선택 index는 `_SELECTED_INDEX_JS` shim이 DOM에서
  읽어 `args[1]`에 주입). 콘솔에서 `selected_gallery_index()`가 정의돼 있는지 확인.
- CN 통합은 `sd_forge_controlnet` 로드 시에만. 없으면 CN 없이 동작해야 함.
- 위젯 개수 정렬: `_refine_widget_count`가 canvas 유무(±2)와 extras(3)를 처리.
  ForgeCanvas 미설치 빌드에서도 슬라이스가 어긋나지 않는지 확인.
- **확인용 로그**: Refine 클릭 시 stderr에 나오는 SAM3 로그 + 브라우저 콘솔 오류
  (특히 Gradio file-cache/SelectData 경고, `KeyError <id>`).

## Anima Tile-Repair — 전제 조건 & 확인 포인트

필수 자산(하나라도 없으면 명확한 한국어 오류로 조기 실패하도록 되어 있음):

- `anima_vendor/` (kohya-ss/sd-scripts, install.py가 클론). 없으면 패널 자체가 안 뜸.
- **Qwen3 Text Encoder** (`models/text_encoder/`의 Qwen3 `.safetensors`) — 패널 드롭다운에서 선택.
  `Use Forge current`는 Anima용 TE를 못 줌.
- **Qwen-Image VAE** — 패널 드롭다운에서 명시적으로 선택. Forge 기본 VAE(SDXL)는 strict load 실패.
- **DiT 체크포인트** — 패널에서 선택하거나 Forge에 로드된 현재 DiT 사용.

- **확인용 로그**: `run_tile_repair` 실행 시 stderr의 `traceback` + "size mismatch"/
  "missing key" 여부. 이게 뜨면 잘못된(비-Anima) DiT/VAE/TE를 고른 것.

## 실제 실행으로만 확인 가능한 것 (다음 단계)

1. Refine: 실제 t2i 결과를 골라 Refine → 결과가 갤러리에 **추가**되고 infotext가
   갱신되는지. (JS shim·Gradio 이벤트 배선이 실 브라우저에서만 검증됨)
2. Anima: 위 자산을 갖춘 상태에서 Tile-Repair 1회 → 산출 이미지 품질 + 시드 재현성.
3. guidance teardown: `Reload UI`(또는 확장 업데이트) 후 attention/noise 패치가
   깨끗이 제거되는지(다른 샘플러/모델과 충돌 없는지).

> 위 1~3의 stderr 로그와 브라우저 콘솔 오류를 캡처해 주면, 그걸 바탕으로 다음 수정
> 지점을 정확히 짚을 수 있습니다.
