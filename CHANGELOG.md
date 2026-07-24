# Changelog

버전 태그는 GitHub Releases에도 발행됩니다. 아래는 요약이며, guidance/속도 기능의
상세는 [docs/GUIDANCE.md](docs/GUIDANCE.md)를 참고하세요.

## v0.17.0 — CI·개발 인프라 + 안정성 보강 + 정리

기능 추가 없이 안전망·정확성·정리에 집중한 릴리즈.

- **CI 도입**: `.github/workflows/ci.yml`이 push/PR마다 pytest(CPU torch+gradio) +
  `node --check`를 실행. 이전엔 회귀 테스트 83개가 자동으로 안 돌았음.
- **웹 세션 SessionStart 훅**: `.claude/hooks/session-start.sh`가 Claude Code on the web
  세션에서 테스트 의존성을 자동 설치(멱등·remote 전용). `.claude/settings.json`에 등록.
- **args 검증 강화**: `sam3_device`(auto/cpu/cuda/cuda:N, 그 외 auto로 폴백), seed 범위
  클램프, inpaint width/height 8의 배수 스냅, CN guidance start>end 자동 swap — 모두
  raise 대신 정규화(호출부가 검증 실패 시 SAM3를 꺼버리므로). 신규 테스트 5개.
- **Anima 전역 전략 복원**: `TokenizeStrategy`/`TextEncodingStrategy` 싱글턴을 Anima 패스
  전후로 `try/finally` 복원 — 이후 비-Anima 경로로의 상태 누수 방지.
- **guidance 패치 teardown 프레임워크**: Safe PAG의 attention/block/self_attn + k-diffusion
  noise 전역 monkey-patch에 clean uninstall 경로 추가. `on_script_unloaded`에 등록해 reload
  시 stale 패치 제거(런타임 경로는 그대로). install→teardown 테스트 추가.
- **정리**: `!sam3.py`의 동일 JS shim 4벌 → `_SELECTED_INDEX_JS` 하나로. install.py에
  벤더 pin 훅(`_ANIMA_PIN`/`_LM_PIN`, 기본 None=기존 동작) 추가. LoRA 모듈 버전 드리프트
  문구 정리.
- **실험 기능 진단 문서**: [docs/EXPERIMENTAL_STATUS.md](EXPERIMENTAL_STATUS.md) — Refine·Anima의
  전제 조건과 실제 Forge 실행으로만 확인 가능한 항목·캡처할 로그 정리.
- **검증**: 회귀 테스트 83개 전부 통과. 브라우저/GPU E2E는 이 환경에서 확인 불가.

## v0.16.0 — Live Workspace 기본화 + 모드 선택 + 탭 전환 부드럽게

기능 5를 Live Workspace 중심으로 재편하고, 인-페이지 툴바(비-Live UI)를 폐기하며,
Live 탭 전환 버벅임을 줄인 릴리즈. 코드 중복도 일부 정리.

- **모드 선택 설정**: 기존 on/off 토글(`sam3_workspaces_enable`)을 `Settings → SAM3 Workspaces`의
  라디오 `sam3_workspaces_mode`(`Live Workspace` 기본 / `기본 Forge UI`)로 교체. `Live Workspace`는
  `/`를 경량 `/sam3-live` 셸로 리다이렉트하고, `기본 Forge UI`는 리다이렉트 없이 순정 Forge를 유지.
  리다이렉트 결정은 `window.opts`가 로드되기 전이라 서버 프로브 `/sam3-live/enabled`(설정을 요청
  시점에 읽음)로 처리.
- **비-Live 인-페이지 툴바 폐기**: 이전 `?sam3_live=off` 경로의 워크스페이스 툴바(Mode D)를 제거.
  `createToolbar`와 셸의 `기본 UI` 전환 버튼 삭제, `mountToolbar`는 Live 자식 프레임만 처리.
  이후 호출자가 사라진 툴바 전용 헬퍼 함수 8개(`switchWorkspace`/`createWorkspace` 등, ~180줄)와
  `.sam3-workspace-*` 툴바 CSS(복원 상태 클래스 `.sam3-workspace-restoring` 제외)도 제거.
  워크스페이스 전환은 이제 Live 셸에서만 이뤄지며, 저장 로직·`실제 탭으로 열기`(네이티브 탭)는 유지.
- **탭 전환 부드럽게(버벅임 완화)**:
  - *숨겨진 워크스페이스 일시정지*: 셸→자식 `visibility` postMessage로 비활성 iframe의
    MutationObserver+800ms 폴링을 멈추고 활성 시 재개(필수 재-마운트 경로인 Forge `onAfterUiUpdate`는
    항상 유지). 세 개의 살아있는 Forge 문서가 계속 CPU를 태우던 문제 완화.
  - *inert 토글 최소화*: `activate()`가 모든 iframe이 아니라 바뀐 두 프레임(이전·새 활성)만
    inert/aria 갱신 → 전환마다 발생하던 style/a11y 리플로우 감소.
  - *인접 탭 선-빌드*: 배경 프리로드가 활성 탭의 가장 가까운 이웃부터 로드.
- **중복 코드 정리**: Refine·Anima의 `_as_float`/`_as_int`를 `sam3ext/coerce.py`로 통합.
- **검증**: 회귀 테스트 77개 전부 통과(신규 route 프로브·모드 게이트·전환 개선 자산 검사 포함).
  브라우저 E2E(실제 Live 셸/탭 전환)는 이 환경에서 확인하지 못했습니다.

## v0.15.0 — Workspace 토글·갤러리 타이밍, 충돌 정리 + 리뷰 버그 수정

코드 리뷰에서 나온 런타임 충돌·버그를 정리하고, txt2img Workspaces(기능 5)의 제어를
개선한 릴리즈.

- **아코디언 정렬 고정**: SAM3 계열 확장 아코디언을 연속된 음수 `sorting_priority` 블록으로
  묶어 SAM3 바로 밑에 차례대로 배치. `SAM3(-30) → Detail Daemon(-29) → Skimmed CFG(-28)
  → Safe PAG(-27) → VAE 2x(-26) → Reference PoC/로그 토글(-25)`. 기존엔 SAM3 mask에
  우선순위가 없어 guidance 계열이 우선순위 없는 타 확장 밑으로 밀려 맨 아래 렌더됐음.
- **Workspaces 설정 토글**: Settings → `SAM3 Workspaces`에 `txt2img Workspaces 활성화`
  옵션(`sam3_workspaces_enable`)을 추가. 끄면 `workspace_manager.js`가 `window.opts`를
  읽어 툴바/탭 마운트를 통째로 건너뜀(페이지 새로고침 후 적용).
- **갤러리 비움 타이밍 변경**: 생성 버튼을 누르는 즉시 이전 갤러리를 감추던 동작을 제거.
  이제 이전 결과를 그대로 두고 Forge live preview가 위에 겹쳐지며, **새 이미지가 완성될
  때** 최종 결과로 교체됨. 사용하지 않게 된 hide-on-generate 로직(JS·CSS)도 제거.
- **PAG 자동 감쇠(안전 브레이크) 제거**: PAG/SEG + SLG 병용 시 각 scale을 활성 항 수로
  나누던 `auto_decay` 토글을 삭제. perturbation은 항상 설정된 full scale로 적용됨. 스크립트
  인자 index는 inert placeholder로 보존해 append-only 계약 유지.
- **guidance 스택 충돌 점검**: PAG·DCW·CWM·SMC·Skimmed CFG 동시 사용이 서로를 무력화하지
  않음을 확인하고 회귀 테스트로 고정(Skimmed→Safe PAG 순서 불변식, 각 단계 기여 검증).
- **런타임 충돌 수정**: (1) unet `model_function_wrapper` 단일 슬롯을 두 스크립트가 덮어쓰던
  문제 — Safe PAG가 우선권을 갖고 경고 로그를 남기며, Ref PoC는 기존 wrapper가 있으면
  yield. (2) CNS 노이즈 패치의 `continue`가 `break`를 건너뛰어 두 k-diffusion 사본을 이중
  패치하던 버그 수정. (3) Anima VAE 2x wrapper 이중 wrap 방지(원본 VAE 재-wrap).
- **리뷰 버그 수정**: inpaint noise multiplier의 `0.0`→`1.0` falsy 강제 제거; Refine
  `inherit_main_neg_prompt` 폴백이 위젯 기본값과 반대로 뒤집히던 문제; Anima 랜덤 시드(-1)
  재현성(명시적 시드 선택); `write_artifacts`가 개별 마스크를 덮어쓰던 free-slot 탐색;
  `unload_sam3`의 명시적 CPU 이동; LoRA Manager 이중 spawn·health false-positive·로그 핸들
  누수; ControlNet `global_state` 등록 idempotent화; 중단된 vendor clone 자가복구.
- **검증**: 회귀 테스트 74개 전부 통과(guidance 조합 3개 신규). 실제 생성 E2E는 아직
  확인하지 않았습니다.

## v0.14.0 — 독립 CFG base 토글 + Skimmed CFG

상호배타였던 CFG base 라디오를 독립 토글로 분해해 SMC·APG·CWM을 자유롭게 조합할 수 있게
하고, anti-burn 기능인 Skimmed CFG를 별도 아코디언으로 추가한 릴리즈.

- **SMC·APG·CWM을 독립 토글로 분리**: 상호배타였던 `CFG base mode` 라디오를 대신해
  `Enable SMC` / `Enable CWM` 체크박스를 추가하고, 기존 `Enable APG`와 함께 원하는 조합을
  동시에 켤 수 있게 함. 켜진 것들은 항상 `SMC → APG → CWM` 순서로 적용되며, 셋 다 끄면
  incoming CFG를 그대로 보존. `Experimental stack` 없이도 APG+CWM, APG+SMC 조합 가능.
- **파라미터 재배치**: `CWM / SMC Advanced` 아코디언을 해체해 SMC lambda/k는 SMC 토글
  아래, CWM alpha low/high는 CWM 토글 아래로 이동.
- **하위 호환 유지**: 라디오와 `Experimental stack`은 `Legacy CFG base mode` 아코디언에
  남겨 새 토글과 OR로 합침. 스크립트 인자는 뒤에 append해 저장된 infotext·API 호출·
  기존 XYZ 그리드가 그대로 동작. XYZ에 `[Anima SMC] Enable`·`[Anima CWM] Enable` 추가.
- **Skimmed CFG 추가**: [Extraltodeus/Skimmed_CFG](https://github.com/Extraltodeus/Skimmed_CFG)의
  공개 수식을 Forge용으로 재작성한 anti-burn 기능을 `Anima Detail Daemon` 바로 아래
  독립 아코디언으로 추가. upstream은 ComfyUI pre-CFG 노드지만 Forge의 pre-CFG 계약이
  달라 post-CFG에서 동일 수식을 재구성. skim 결과를 Forge의 예측 tensor에 다시 써서
  **SMC/APG/CWM·PAG delta·DCW와 동시에 사용 가능**(ComfyUI에서 pre-CFG 노드를 물린 것과
  같은 조합). 정렬 우선순위로 Safe PAG보다 먼저 실행되도록 보장.
- **검증**: 회귀 테스트 71개 전부 통과(신규 Skimmed CFG 12개는 상단 수식 transcription과
  tensor 단위 일치 및 downstream 전파를 확인, 신규 CFG base 토글 5개 포함). 실제 생성
  E2E는 아직 확인하지 않았습니다.

## v0.13.0 — 실제 Workspace 탭 + Guidance 제어·UI 완성

Live Workspaces의 iframe 전환이 무거운 환경을 위해 같은 WebUI 포트의 실제 브라우저 탭으로
전환하는 경로를 추가하고, PAG/SEG 공식 구현의 전체 제어값과 안전 힌트를 UI에 노출한 릴리즈.

- **실제 브라우저 탭 모드**: Live 헤더의 `실제 탭으로 열기`가 로드된 Workspace를 먼저
  강제 저장한 뒤 현재 셸을 활성 Workspace로 바꾸고 나머지를 최상위 브라우저 탭으로 엶.
  탭마다 고정 slot·이름·자동 저장 상태를 표시하며 iframe은 0개가 되어 브라우저 기본 탭
  전환과 같은 경로를 사용. 팝업 차단은 감지해 열린 수와 허용 안내를 표시.
- **Live Workspace 안정성·UX**: 세 iframe을 모두 한 번 준비하되 비활성 화면은 표시만
  전환하고, 자동 저장 상태를 Live 헤더에 전달. Forge 상단 txt2img/img2img/PNG Info/
  Settings/Extensions 탭을 유지하고 확장 패널은 Parameters 아래, Forge 기본 Script/XYZ만
  중앙 Scripts에 배치.
- **Forge 갤러리 동작 보존**: Generate 시 이전 결과만 숨기고 Gallery 루트를 유지하여
  Waiting/Queue/진행률/중간 미리보기가 Forge 기본 경로로 표시된 뒤 이번 최종 결과만 남김.
- **공식 PAG/SEG 전체 제어**: Attn Scale, 공식 perturbation strength, block/head indices,
  Start/End, Rescale, full/partial rescale mode를 UI와 XYZ에 일치시킴. Legacy Soft PAG/
  SEG-Approx strength는 호환 아코디언으로 분리.
- **Guidance UI 정리**: DCW/DAVE/CNS를 중첩 탭 밖의 주 패널로 이동하고 SLG/APG/
  Adaptive/CWM/SMC를 포함한 각 수치 항목에 깨짐·과채도·구도 변화가 보일 때의 조절 방향을
  설명하는 맞춤 힌트를 추가.
- **검증**: Python 회귀 테스트 54개와 JavaScript 문법 검사 통과. 실제 `7860` Forge에서
  Workspace 1/2/3이 각각 별도 최상위 탭, 고정 slot, 개별 제목·저장 상태, iframe 0개로
  열리는 것을 확인. Forge 상단 탭 유지 및 새 `Traceback`/`KeyError` 0개 확인.

## v0.12.0 — 동적 Live Workspaces + RK/TDE Gradio 가드

Live Workspaces를 고정된 iframe 3개에서 동적으로 관리 가능한 작업공간 셸로 확장하고,
`--api` 시작 경로에서 타 샘플러 확장이 남기던 미등록 Gradio 컴포넌트 오류를 차단한 릴리즈.

- **경량 `/sam3-live` 셸**: 사용하지 않는 부모 Forge UI 한 벌을 먼저 만드는 구조를 제거.
  현재 선택한 Workspace를 우선 로드하고 나머지는 순차 백그라운드 준비. 같은 포트와 Forge
  서버를 그대로 사용하며 기존 루트 주소는 경량 셸로 자동 전환.
- **동적 Workspace 관리**: 기본 1/2/3과 기존 저장 데이터는 유지하면서 최대 20개까지 추가.
  현재 설정 복제, 이름 변경, 삭제, JSON 내보내기/가져오기를 Live 헤더에서 직접 수행.
- **시작 복원 경량화**: Gradio가 이미 파싱한 `window.gradio_config`를 재사용해 iframe마다
  약 5.5 MB `/config`를 중복 요청하지 않음. 갤러리 초기화는 설정 복원을 막지 않으며,
  Script/XYZ 의존성은 실제 값이 달라진 드라이버만 재실행하고 불일치가 있을 때만 검증 대기.
- **활성 화면 우선**: 로컬 Forge Neo 실측에서 활성 Workspace가 약 6초에 준비됐고, 나머지는
  활성 화면을 방해하지 않도록 순차 준비. 모든 Workspace는 계속 독립된 Gradio 문서이므로
  값 바꿔치기 없이 전환되고 Generate는 현재 화면 하나만 실행.
- **RK/TDE `KeyError` 가드**: Forge `--api`가 임시 `gr.Blocks`에서 script `ui()`를 재실행할 때
  RK Sampler/TDE Sampler의 모듈 리스트에 섞이는 throwaway 컴포넌트를 실제
  `modules.script_loading.loaded_scripts`와 callback globals에서 찾아 `demo.load` 등록 직전에
  in-place 제거. Forge 코어나 두 외부 확장 파일은 수정하지 않음.
- **UI 및 저장 안정성**: 확장 UI는 Parameters 아래에 유지하고 중앙 Scripts에는 Forge 기본
  Script/XYZ만 배치. 프롬프트·네거티브·XYZ 상태, 현재 세션 마지막 갤러리, 충돌 보호와
  서버 재시작 자동 복구를 동적 Workspace에서도 유지.
- **검증**: Python 회귀 테스트 46개, JavaScript 문법 검사, 실제 브라우저에서 1/2/3 전환,
  Prompt/Negative/XYZ 복원, 추가·이름 변경·삭제·내보내기 통과. 클린 Forge 부팅에서
  미등록 dependency 0개, `KeyError`/`Traceback` 0개 확인.

## v0.11.0 — Anima Guidance Suite 공식 모드 + Live Workspaces

Forge Neo 코어 파일을 수정하지 않고 Anima guidance 실행 경로와 단일 탭 다중 작업공간 UI를
확장 내부에서 완성한 릴리즈.

- **공식 PAG/SEG + SLG**: PAG는 appended weak row의 hard value-only attention, SEG는 실제
  Anima T/H/W 중 H/W query Gaussian blur로 동작. 기존 soft PAG / SEG-approx는
  `Legacy Soft/Approx` 호환 토글로 분리.
- **Guidance 오케스트레이터**: Preserve/APG/CWM/SMC/SMC+CWM CFG base, DCW, DAVE,
  CNS-inspired wavelet noise, Adaptive Guidance를 고정된 순서와 generation 단위 상태 정리로
  통합. 모든 기능은 기본 OFF이며 중립 설정은 기존 Forge 결과를 보존.
- **최신 Anima attention hook 복구**: `SelfCrossAttention.torch_attention_op`의 실제
  value/output 레이아웃과 staticmethod binding을 보존하면서 weak row에만 perturbation 적용.
  훅 미도달·shape 불일치는 원본 결과로 안전하게 폴백.
- **검증 로그**: 확장 하단의 debug/안전 아코디언에 opt-in Guidance verification summary를
  추가. PAG/SEG/SLG 적용 스텝, APG/Adaptive, DCW/DAVE/CNS 실행 여부를 생성 종료 시 요약.
- **Live Workspaces**: 같은 탭 안에 독립된 txt2img 문서 3개를 유지하여 값 바꿔치기 없이
  즉시 전환. prompt/negative, 생성 설정, Script/XYZ 상태, 마지막 생성 갤러리를 작업공간별
  보존하며 Generate는 현재 화면 하나에서만 실행.
- **Workspace UI 수정**: Prompt/Negative/Generate와 Parameters/Scripts/Gallery 3열 배치를
  원래 `#tab_txt2img` CSS 범위 안에 유지해 찌그러짐과 겹침을 제거. 모든 확장 UI는
  Parameters 아래에 두고 중앙 Scripts에는 Forge 기본 Script 선택기와 기본 패널만 표시.
- **갤러리 수명주기**: Generate 직전에 이전 결과를 비우고 이번 생성 결과만 유지. 페이지/WebUI
  재시작 시 갤러리를 비우며 서버 재시작 복구 시 세 iframe을 자동 재연결.
- **기타 안정성**: ForgeCanvas가 없는 Refine 배선에서 prompt 입력이 canvas 값으로 밀리던
  인덱스 오류 수정, `sam3ext.guidance` import가 SAM3 모델 의존성을 초기화하지 않도록
  package import를 지연 로딩으로 변경.
- **검증**: Forge Neo 2.27 + `anima_baseV10` 실제 경로 확인, Python 회귀 테스트 38개와
  JavaScript 문법·실제 브라우저 DOM/레이아웃 검증 통과.

## v0.10.0 — txt2img Workspaces (단일 탭 작업공간 3개)

여러 브라우저 탭을 다시 열고 설정을 복사하던 흐름을 대체하는 확장 전용 작업공간 관리자를 추가.
Forge Neo 기본 파일을 수정하지 않고 한 탭에서 Workspace 1/2/3을 전환합니다.

- positive/negative prompt, seed·steps·sampler·scheduler·크기 등 txt2img 생성 설정을 작업공간별 저장.
- 선택한 Script와 X/Y/Z Plot 축·값·옵션까지 함께 저장하고 복원.
- 입력 변경 시 브라우저 로컬 저장소에 자동 저장. 빈 작업공간의 첫 전환은 현재 설정을 복제.
- 같은 출처의 여러 탭이 동일 작업공간을 수정할 때 최신 저장본을 덮지 않는 충돌 보호.
- Workspaces 바를 Generation 리사이즈 행 위의 독립된 전체 너비 행으로 배치해 설정 UI 변형 방지.
- 이미지·파일·갤러리·생성 결과/output과 checkpoint/VAE 등 전역 Quicksettings는 저장 대상에서 제외.
- 저장소는 프로토콜·호스트·포트가 모두 같은 동일 출처에서만 공유. 다른 주소나 브라우저
  프로필로 옮길 수 있도록 내보내기/가져오기 제공.

## v0.9.18 — Anima PAG 검정 실루엣 붕괴 수정

`[Anima Pert] Enable=True`에서 이미지가 검게 붕괴하던 PAG 실행 경로를 Forge Neo의
실제 denoised(x0) 훅·상류 Safe-PAG 수식에 맞게 수정.

- **핵심 원인 1 — 잘못된 rescale**: 기존은 매 스텝 `CFG base + PAG correction`
  전체에 스케일 팩터를 곱해 밝기/에너지를 반복적으로 0 방향으로 빼앗음. 상류와
  동일하게 팩터는 **PAG correction에만** 적용하여 correction=0이면 CFG base가
  bit-identical로 보존됨.
- **핵심 원인 2 — 과도한 기본 블록**: 빈칸을 28블록 후반 전체 `14-27`로 해석하던
  로컬 동작을 상류 권장 단일 블록 `18`로 변경. UI/SLG/문서도 동일하게 맞춤.
- **x0 직접 보정**: Forge `model.apply_model` 결과는 이미 predictor 변환된 x0이므로,
  잘못된 raw-output/`c_out` 추정을 제거하고 `cond_x0 - weak_x0`를 직접 사용. 이로써
  Forge의 CFG=1 uncond 생략 경로에서도 PAG가 실제 적용됨.
- **XYZ 상태 누수 수정**: `p.extra_generation_params`를 재사용하는 True/False 셀 사이에
  이전 PAG/APG/AdaptiveG infotext가 남지 않도록 셀별 정리.
- **안전성**: XYZ/API에서 slider 범위를 우회한 NaN/Inf/과도한 scale·strength·range·
  rescale 입력을 유한값 + UI 범위로 정규화. 첫 유효 weak delta와 generation summary
  로그를 추가해 무효 훅을 즉시 구분할 수 있음.
- **검증**: PAG 회귀 테스트 8개 통과. `anima_baseV10` 실제 생성(20 steps, seed 동일)에서
  PAG False/True 변경 픽셀 98.58%, True 평균 RGB 103.89로 검정 붕괴 없이 weak/apply
  14/14 스텝 실행을 확인.

## v0.9.17 — Gradio `state_holder` KeyError 수정

생성/Refine 후 콘솔에 `KeyError: <숫자>` (gradio `state_holder.py` `__contains__`) 트레이스백이
찍히던 문제 수정.

- **원인**: 핸들러가 `gr.update()`를 반환하면 gradio가 **요청 시점에** 해당 컴포넌트를 다시
  만든다(`blocks.py` `postprocess_data`: `state[block._id] = block.__class__(**constructor_args)`,
  `render=False` 강제 주입). 이 `constructor_args`에는 **원본 elem_id가 그대로** 들어있고,
  webui는 `gradio.components.Component.__init__`를 패치해 두었기 때문에
  (`modules/gradio_extensions.py`) 이 일회용 인스턴스에 대해서도 `on_after_component`가
  발화한다. `render()`가 호출되지 않아 이 인스턴스의 `_id`는
  `demo.default_config.blocks`에 등록되지 않는데, `SessionState.blocks_config`는 그 dict의
  얕은 스냅샷이라 이후 해당 컴포넌트를 건드리는 이벤트에서 `KeyError`가 난다.
- **증상 경로**: `_refine_error_return` / `_anima_error_return`이
  `outputs=[gallery, status, html_info, generation_info]`로 배선돼 있어, Refine/Anima의
  early-return마다 `html_info_txt2img`·`generation_info_txt2img` 에코가 발생 → 캐시해 둔
  전역이 미등록 컴포넌트로 덮이고, `refine_panel` 센티넬까지 오염될 수 있었음.
- **수정**: `on_after_component`가 **실제로 등록된 컴포넌트에 대해서만** 동작하도록 가드 추가.
  `Context.root_block`은 ContextVar가 아닌 프로세스 전역이라 Reload UI 중 in-flight 요청
  에코가 새는 레이스가 남으므로, 등록 여부(`component._id in ...default_config.blocks`)를
  직접 확인한다. UI 빌드 중에는 완전한 no-op이라 기능 변화 없음
  (Compact 프롬프트 레이아웃의 `render=False` 컨테이너 자식도 즉시 등록되므로 안전).
- **부수 수정**: `modules/api/api.py`가 임시 `with gr.Blocks():` 안에서 모든 스크립트의
  `ui()`를 재실행하는데, 실제 빌드에서 `build_anima_panel()`이 실패했을 경우 그 패스가 죽은
  패널을 잡아 Tile-Repair가 프로세스 내내 비활성화될 수 있었음 → `anima_build_attempted`
  플래그로 차단.

## v0.9.16 — 경량화 패스 (기능 제거 없음)

전체 코드 감사 후 상시/반복 비용만 안전하게 트림. 모든 기능 유지.

- **매 생성 비용 ↓ (일반 non-Anima 포함)**: `Sam3MaskScript.process`가 SAM3 꺼짐 + XYZ
  없음이면 ~50필드 payload 조립 + `Sam3Args` pydantic 검증을 **건너뜀**(early-return).
- **매 스텝 비용 ↓**: `_post_cfg`가 합칠 게 없으면(예: Adaptive Guidance만 켜짐) `float()`
  왕복/latent 2회 할당 없이 즉시 반환.
- **어텐션 콜당 비용 ↓**: 영구 설치되는 SDPA 래퍼가 원본을 `_STATE.get()` 대신 모듈 전역
  `_ORIG_SDPA`로 참조 + 비활성 fast-path를 bool 체크 1회로 단축.
- **VRAM ↓**: (a) PAG가 스택한 latent 텐서(cond/uncond/attn/slg_raw + APG momentum)를
  `postprocess`에서 해제. (b) VAE 2x 디코더 캐시를 **최근 1개로 상한**(나머지 evict +
  `empty_cache`). (c) VAE 파일 목록 스캔을 memoize(탭별 재스캔 방지).

감사 결론: import-타임 디스크 스캔/불필요 무거운 import 없음, 기능 OFF 시 base Forge 대비
거의 무비용. 위 항목만 실제 개선 여지였음.

## v0.9.15 — Regional Style-Swap (RegionalSampler 워크플로 재현)

rouge-kasshoku의 "Anima Crossover Couple / RegionalSampler" 가이드(스타일 블리딩 해결)를
Forge Neo의 **기존 SAM3 Refine**로 재현하는 레시피 + 프리셋 버튼.

- **docs/REGIONAL_STYLE_SWAP.md** — 코드 없이 지금 바로 쓰는 단계별 레시피. 핵심 매핑:
  `denoise ≈ 1 − base_only_steps/steps`(예 steps33·B8 → 0.76), `overlap_factor ≈ mask blur`,
  region LoRA는 Replacement 프롬프트에만 넣어 **LoRA 격리** 달성, 동일 seed(🎯)+Euler.
- **🎭 Regional Swap preset 버튼**(Refine 패널): 한 번 클릭으로 가이드 기본값 세팅
  (Euler·CFG5·33steps·denoise0.76·mask blur16·inherit OFF·inpaint only masked·fill original).
  기존 위젯 값만 바꾸며 `REFINE_ARG_KEYS`/입력 배열은 건드리지 않음(저위험).
- 한계: SAM3 Refine는 image-레벨 인페인트(가이드 방법 #2)라 진짜 latent-레벨 RegionalSampler
  (#3)보다 seam이 약간 더 생길 수 있음 → mask blur + inpaint-only-masked로 완화.

## v0.9.14 — Anima VAE 2x (spacepxl decoder) [실험]

spacepxl **2x Wan-VAE 파인튜닝**을 디코더로 써서 speckle을 줄이고 skin/hair를 정리하는
독립 스크립트(`scripts/anima_vae_2x.py`). Qwen/Wan VAE가 latent 구조를 공유하므로 Anima
생성에도 적용됩니다(Forge Neo 로더가 `AutoencoderKLWan`/`AutoencoderKLQwenImage`를 같은
경로로 처리함을 확인).

- **동작**: 12채널 디코더(pixel-shuffle 2x)를 `WanVAE(conv_out_channels=12)`로 직접 빌드해
  `forge_objects.vae`의 decode만 대체(순정 로더는 채널을 하드코딩해 12ch를 못 실음).
  decode: latent(1프레임) → 12ch → `pixel_shuffle(2)`(12→3@2x) → (1x 모드면 downsample+
  약한 blur) → 3ch. 오류 시 순정 decode로 폴백.
- **감지**: state_dict `decoder.head.2.weight` shape[0]==12 (safetensors 헤더만 읽음).
- **UI**: Enable · VAE 파일 · 1x refined / 2x · (Advanced) blur sigma · latent renorm 토글.
- **검증됨**: 감지 로직·pixel-shuffle 채널 산술(12=3·2·2). **런타임 확인 필요**: Wan-2.1
  VAE config 정합(로드 diff 로그로 조정), Qwen↔Wan latent 정규화(색 틀어지면 renorm),
  1프레임 축 처리.

## v0.9.13 — Guidance 패널을 SAM3 바로 밑으로

`sorting_priority`를 98/97 → `0`으로 낮춰 Perturbation Guidance · Detail Daemon 아코디언이
SAM3 확장 블록 안에서 **SAM3 바로 밑**에 표시되도록 위치를 옮겼습니다(Forge는 낮은 값이
위쪽). 두 스크립트 모두 여전히 현재 `forge_objects.unet`에서 clone하므로 다른 unet 패치
스크립트와의 합성은 순서와 무관하게 유지됩니다.

## v0.9.12 — Anima Guidance & Speed Suite

SAM3와 **완전히 분리된 독립 기능 모음**을 추가했습니다. `sam3ext`를 import하지 않고 Forge
Neo 코어 파일/기본 동작을 건드리지 않으며, 전 구간 try/except로 어떤 오류에도 일반 생성으로
폴백합니다(켜 둬도 생성이 깨지지 않음). 두 독립 스크립트로 제공됩니다:
`scripts/anima_safe_pag.py`, `scripts/anima_detail_daemon.py`.

| 기능 | 효과 | 추가 forward | 대상 |
|---|---|---|---|
| **PAG / SEG / SLG** | 구조·디테일 강화 (perturbation guidance) | 있음(배치 접기) | Anima DiT |
| **APG** | 높은 CFG 과채도·번짐 억제 | 없음 | 모든 모델 |
| **Detail Daemon** | 질감·잔디테일↑, 배경 뽀샤시↓ | 없음 | 모든 모델 |
| **Adaptive Guidance** | 후반 uncond 생략 → 무손실 속도↑ (~−27%) | 음수(생략) | 모든 모델 |

**Forge Neo 연동 (코어 수정 없음)** — 실제 샘플링이 `sampler_calc_cond_batch_function`을
호출하지 않음을 소스에서 확인하고, 실제 호출되는 훅만 사용:
- `model_function_wrapper` — cond 행을 배치에 접어 perturbation 약한 예측을 *같은 forward*로
  계산(별도 호출 없음). Adaptive Guidance는 반대로 uncond 행을 제거.
- `post_cfg_function` — `c_out` 실측 복원으로 denoised(x0) 공간에서 정확히(eps/v/flow 무관)
  guidance 합성. 표준 CFG·APG·MaHiRo 위에도 안전하게 얹힘.
- 매 생성 `forge_objects.unet.clone()`에만 훅 → Forge 기본 동작·타 생성 무영향.

**설계 원칙** — 모든 자동동작은 토글(scale 자동감쇠, APG→rescale 자동 off, Detail Daemon
CFG couple). 값은 기본 쉽게(메인 슬라이더/프리셋) + 필요 시 깊게(Advanced 아코디언). 조합은
Perturbation(attn 택1·SLG 병용) + 크기보정(APG↔rescale) + Detail Daemon + Adaptive
Guidance가 서로 다른 지점이라 안전하게 병용됩니다.

**검증** — 전 스크립트 py_compile 통과. 수학 독립 검증: PAG `c_out` 복원(~3e-15), APG(eta=1→
표준 CFG 정확 환원·eta=0 직교·norm clamp), 다중항+auto_decay guidance, Detail Daemon 스케줄,
Adaptive Guidance 게이팅/재구성.

> ⚠️ **실험 기능** — 정적·수학 검증만 됐고, 실제 Anima 체크포인트로 end-to-end 확인이 1회
> 필요합니다(리포의 다른 Anima 기능과 동일 상태). 콘솔 `[AnimaSafePAG]` /
> `[AnimaDetailDaemon]` 로그로 훅 부착·동작 확인.

세부 버전 흐름: `v0.9.8` PAG 독립 스크립트 → `v0.9.9` APG → `v0.9.10` Detail Daemon 포크 →
`v0.9.11` SEG+SLG+scale 자동감쇠 토글 → `v0.9.12` Adaptive Guidance + docs 정리.

## v0.9.7 — Anima reference-latent shape logger (PoC)

`process_before_every_sampling → forge_objects.unet` 경로로 Anima UNet에 model-function
wrapper를 안전하게 붙일 수 있는지 확인하는 계측 PoC. 이후 guidance suite의 이식 토대가 됨.

## v0.9.6 — PiD Upscale 복원 모드

Forge Neo 네이티브 NVIDIA PiD(Pixel Diffusion Decoder) 초해상 복원을 Anima 복원 패널의
모드 옵션으로 추가.

## v0.9.5 — WF3 Tile-Repair 정적 blocker 수정 + TE/VAE 스마트 기본값

Anima Tile-Repair의 정적 버그 정리 및 Qwen3 TE / Qwen-Image VAE 자동 기본값.

## v0.9.4 — LoRA Manager: Forge Neo 연동 + 모달 버그 수정

vendored LoRA Manager의 "Send to ComfyUI" → "Add LoRA"(프롬프트 삽입), ComfyUI→Forge Neo
라벨 치환, 사용 팁 X 버튼/메모 placeholder 버그 수정.

## v0.9.3 — LoRA Manager: 후원 UI 제거 + 업데이트 알림 비활성화

기부 UI 숨김(GPL "Appropriate Legal Notices" 아님) + 상류 업데이트 폴링 short-circuit.
LICENSE·저작권·저자 귀속 미변경.

## v0.9.2 — LoRA Manager: fetch 진행 'failed' 잘림 수정

`.loading-status` 줄바꿈 허용으로 긴 LoRA 이름 뒤 카운터 잘림 수정.

## v0.9.1 — LoRA Manager: Manage 탭 빈 화면 수정

탭 pane selector 교정 + 첫 실행 스캔 논블로킹화(진행 표시 폴링).

## v0.9.0 — LoRA Manager 통합

[willmiao/ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager)를 lazy
spawn standalone 서버 + iframe으로 extra-networks strip의 Manage 탭에 임베드.

---

이전 버전(SAM3 검출/인페인트, Refine, ControlNet 통합, Anima Tile-Repair 등)의 상세는
[README.md](README.md)를 참고하세요.
