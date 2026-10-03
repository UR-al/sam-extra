# Changelog

버전 태그는 GitHub Releases에도 발행됩니다. 아래는 요약이며, guidance/속도 기능의
상세는 [docs/GUIDANCE.md](docs/GUIDANCE.md)를 참고하세요.

## v0.32.1 — Colorcraft 편집기 Type 표시 지연 · 화면 낭독기 접근성 · SPEED custom 스펙트럼 오류

v0.32.0 의 공유 편집기에서 남았던 표시 지연을 고치고, 편집기를 화면 낭독기로 쓸 수 있게 했습니다. 생성 결과 · 스크립트 인자(67개) ·
infotext 는 그대로이고(결과 같음), 새 설정은 없습니다. Forge 를 재시작하고 페이지를 새로 고치세요(바뀐 `javascript/` 파일).

### Colorcraft 편집기

- **Type 드롭다운 표시 지연 (고침)**: 항목을 바꾸면 Forge 의 빠른 드롭다운 버튼 글자가 Gradio 의 Type 값보다 한 화면 갱신 늦게(약
  0.1 초) 바뀌던 것을, 같은 화면 갱신에 그리도록 했습니다 — 편집기의 후속 갱신을 0 ms 타이머 뒤 애니메이션 프레임 하나(Gradio 가 다음
  프레임에 걸어 둔 갱신 뒤)로 옮김. 실제 Forge(7862 테스트 인스턴스, 헤드리스 Chrome)에서 I ↔ III 전환 11번씩: 두 탭 모두 11/11 이 같은
  프레임, 클릭부터 버튼 글자가 보이기까지 중앙값 335 ms → 227–246 ms(Gradio 자신의 갱신 시간과 같음).
- **화면 낭독기 접근성 (새 기능)**:
  - 선택 줄(수정자 · 마스크 · 조합)과 Pass 라디오 묶음이 제목(`수정자` · `마스크 · 조합` · `Pass`)을 이름으로, ●/○ 설명 줄을 설명으로
    갖고, 라디오마다 묶음 안 위치와 개수를 알립니다 — Gradio 는 라디오마다 이름을 따로 줘 그동안 낭독기가 모두 "1/1" 로 읽었습니다.
  - 편집기 칸(숫자 · 슬라이더 · 체크박스 · Type 버튼 · 조합 칸)마다 지금 고치는 항목을 설명으로 붙입니다 — 예: `편집 중: 수정자 III
    (Chroma)` · `편집 중: 조합 C2`. 칸에 원래 있던 설명은 그대로 둡니다.
  - 항목을 바꾸면(바뀐 쪽만) · Reset 하면(`초기화: …`) · 붙여 넣으면(`붙여넣은 설정 불러옴 · …`) 탭마다 하나 있는 상태 줄이 조용히
    (polite) 알립니다. 값 고치기 · Type 고르기 · Generate 는 알리지 않습니다.
  - 탭마다 숨은 노드 하나를 페이지가 뜰 때 요약 블록 안에 더하고, 그 뒤로는 글자와 `aria-*` 속성만 바뀝니다(리스너를 더하지 않음).
    낭독기용 표시 코드에서 오류가 나도 편집기 동작은 그대로이고 콘솔에 `[Colorcraft] a11y:` 한 줄만 남습니다.
  - `javascript/notebook.js` 의 빠른 드롭다운 버튼이 숨은 Gradio 입력의 `aria-describedby` 를 이어받습니다(설명이 없는 드롭다운은 그대로).
  - Chrome 접근성 트리로 두 탭 32/32 항목을 확인했고, 8번 가로챈 Generate 요청의 인자 67개는 같은 조작의 v0.32.0 과 같았습니다
    (rev 제외). NVDA 등 실제 낭독기로 들어 보지는 않았습니다.

### Anima SPEED

- **custom 스펙트럼 A · β 오류로 요청 전체가 실패하던 것 (고침)**: Spectrum preset `custom` 에서 A ≤ 0 이면 `ValueError: math domain
  error` · `ZeroDivisionError` 가 SPEED 밖으로 빠져 Forge 생성 요청 자체가 실패했고, A · β 가 NaN 이면 오류 없이 전환 σ NaN 인 계획으로
  돌았습니다. 이제 A 는 양수 · 유한, β 는 유한해야 하며, 아니면(또는 β 가 너무 커 전환 시간을 못 구하면) 그 패스는 순정으로 돌고
  `Anima SPEED status: invalid settings - spectrum A must be a positive number; …` 를 남깁니다. 다른 프리셋과 올바른 custom 값의
  결과는 그대로입니다.

테스트: 확장 2908 OK(skip 22), JS 219/219.

## v0.32.0 — Colorcraft 공유 편집기 (API 위치 인자 변경) · 진행 막대 · MCP 서버 수정 · CI

v0.31.0 에서 들인 기능을 다듬은 판입니다. Colorcraft 패널을 수정자·마스크마다 따로 있던 컨트롤 대신 편집기 한 벌로 바꿔 가볍게
했습니다 — 계산과 infotext 는 그대로이고, 그 대신 v0.31.0 의 위치 인자 579개 API 형식은 더 읽지 않습니다(옛 배열을 한 겹 감싸면
그대로 읽음). 진행 막대와 MCP 서버는 이번에 처음 실제 Forge 에서 돌려 보고 찾은 버그(진행 막대 4개, MCP 4개)를 고쳤고, GitHub CI 는
프런트엔드 검사를 Node 24 의 별도 잡으로 나누고 Python 잡의 오류 17개를 없앴습니다. 새 설정은 없습니다.

괄호 표시는 v0.31.0 과 같습니다: (결과 같음) = v0.31.0 과 같은 설정·시드에서 이미지 동일, (새 기능) = v0.31.0 에 없던 기능이라 비교
대상 없음. 이번에 더한 (바뀜) = v0.31.0 과 같은 요청·조작에서 동작이나 표시가 달라짐.

**업그레이드할 때**

- Forge 를 재시작하고 페이지를 새로 고치세요 — Colorcraft 패널의 스크립트 인자가 579 → 67개로 바뀌고, 새
  `javascript/colorcraft_editor.js` 가 있어야 편집기가 항목을 바꿉니다.
- API 로 Colorcraft 를 **위치 인자 579개**로 보내던 호출은 이제 적용되지 않습니다(이미지는 Colorcraft 없이, status 에 이유). 옛 배열을
  한 겹 더 감싸거나(`{"args": [[…579개…]]}`) 첫 인자 하나(infotext 값 · 인자 경로 사전)로 보내세요 — 아래 **API** 항목. 첫 인자 하나,
  `[true]` · `[true, true]` 로 보내던 호출은 그대로입니다.

### Colorcraft — 공유 편집기 (결과 같음, API 위치 인자는 바뀜)

- **패널을 편집기 한 벌로 (결과 같음 — CPU 비트 대조 · 실제 Forge GPU 대조)**: 수정자 I–X · 마스크 M1–M10 · 조합 C1–C5 마다
  따로 있던 컨트롤(탭마다 579개)을, 선택 줄에서 고른 항목을 보여 주는 편집기 하나(수정자 44칸 · 마스크 10칸 · 조합 7칸)로 바꿨습니다. 다른 항목의 값은 숨은 칸(JSON state)에 들고
  있다가, 생성할 때 그 위에 편집기의 지금 값을 얹어 읽습니다 — 슬라이더를 놓자마자 Generate(Ctrl+Enter)를 눌러도 편집기에 보이는
  값이 그대로 쓰이고, 어떤 이벤트가 먼저 끝나기를 기다리지 않습니다. 항목 고르기 · Reset(수정자 Reset 은 Active 를 그대로 둠) · 표시
  갱신은 브라우저(`javascript/colorcraft_editor.js`, 필드 표는 `spec.py` 에서 만든 `javascript/colorcraft_schema.js`)에서만 돌아 서버
  요청이 없습니다. Forge 처럼 큰 페이지에서는 항목을 바꾸는 데 0.2초 남짓 걸립니다(Gradio 4.40 이 갱신마다 레이아웃을 다시 그림).
- **한눈에 보기 (새 기능)**: 선택 줄의 ● 는 켠 수정자 · 켠 수정자가 실제로 쓰는 마스크, ○ 는 값을 바꿨지만 꺼 둔 수정자 · 값만 바꾼
  마스크입니다(선택 줄 이름 밑에 `● 켬 · ○ 값을 바꿨지만 꺼 둠` · `● 켠 수정자가 씀 · ○ 값만 바꿈`). Enable 아래 요약 한 줄이 켠
  수정자의 Type · 마스크 · 패스와 쓰는 마스크를 보여 줍니다 — 예: `켠 수정자: I Advanced, III Chroma ➜ C2 · 쓰는 마스크: M1, M2, M3,
  C1, C2`(Enable 이 꺼져 있으면 앞에 `꺼짐 — `, 켠 수정자가 없으면 `켠 수정자 없음`). A·B 가 다 차지 않은 조합은 ● 가 붙지 않고
  요약에 `➜ C1(미완성)` 으로 보입니다(그 수정자는 예전처럼 마스크 없이 적용).
- **편집기 스크립트가 없거나 오류가 나면**: 선택 줄이 편집기에 보이는 항목으로 돌아가고, 요약 줄에 `편집기 스크립트(javascript/
  colorcraft_editor.js)가 동작하지 않아 다른 항목을 고를 수 없습니다 — 페이지를 새로 고치세요. …` 가 뜹니다(브라우저 콘솔에 오류 한
  줄). Type 에 따라 칸을 숨기는 것도 멈추지만 생성은 편집기에 보이는 값 그대로입니다 — 다른 항목을 고른 것처럼 보이면서 보이지 않는
  항목에 편집이 들어가는 일은 없습니다.
- **가벼워짐**: 탭마다 스크립트 인자 579 → 67(enabled · masking · state · debug · debug_step · ref · 편집기 61칸), Gradio 블록 1,203 →
  154, 붙여 넣기 필드 579 → 65. 두 탭을 합쳐 이벤트 80 → 20(10개는 브라우저 전용), Generate 요청 하나의 Colorcraft 몫 약 2.9 KB →
  0.3 KB, 패널 만들기(`ui()`) 약 190 ms → 19 ms, 페이지 설정(`/config`, 실제 Gradio 4.40 의 두 탭 하네스) 943,946 B → 154,659 B(16.4 %).
  실제 Forge 전체 `/config` 는 v0.31.0 의 5,354,074 B → 4,521,310 B(−15.6 %, 컴포넌트 8,382 → 6,284)이고, 페이지를 열고
  Colorcraft 아코디언이 생기기까지가 5.06 초 → 3.65 초(8번 중앙값)로 줄었습니다.
- **그대로인 것**: 계산(엔진 · `spec.py` 해석), infotext(`SAM Extra Colorcraft` · `… status` · `… pre-DD sigma`), 붙여 넣기(이 확장 ·
  원본 · 포크 키, API 의 `infotext` 필드), XYZ 축(수정자 I), Debug 미리보기(편집기에 보이는 값까지 반영), Settings, 레이아웃 열의
  "켜짐" 표시(Enable 만 — 편집기의 Active 는 켜지 않음), 범위 밖 숫자 입력(`colorcraft_sliders.js`), txt2img · img2img 따로,
  ui-config.json 에 저장하지 않음(Reload UI · 새로 고침은 기본값).
- **붙여 넣기**: PNG Info · ↙ 로 붙여 넣으면 state · 선택 줄 · 요약 · 편집기가 Python 응답 하나로 함께 바뀌고, 편집기는 첫 켠 수정자
  (없으면 I)와 쓰는 첫 마스크 M(없으면 M1 — 조합 편집기에는 놓이지 않음)을 보여 줍니다.
- **API: 위치 인자 579개 형식을 더 읽지 않음 (바뀜 — 호환 깨짐)**: Forge 는 요청에서 스크립트 인자 수(이제 67)만큼만 잘라 넘기므로,
  v0.31.0 의 위치 인자 579개 요청은 Colorcraft 를 적용하지 않고 `SAM Extra Colorcraft status: not applied: v0.31.0 positional
  arguments (579 values) are no longer read - …` 와 콘솔 `[Colorcraft] not applied: …` 한 줄(생성마다, XYZ 는 칸마다)을 남깁니다.
  이미지는 Colorcraft 없이 만들어집니다(스크립트는 API 요청을 실패시킬 수 없음). 옛 형식은 세 번째 값(I.active)이 bool · 숫자이거나
  4–6번째 값이 옛 I.kind · I.pass · I.mask 값이면 알아보므로 I.active 가 null 이어도 거절되고, Enable(첫 값)이 false 면 예전처럼
  조용히 꺼져 있습니다.
  - **옮기는 법**: 옛 배열을 한 겹 더 감싸 보내면(`{"args": [[true, false, true, "Advanced", …579개…]]}`) v0.31.0 과 똑같이 읽습니다.
  - **짧은 형식 (그대로)**: 첫 인자 하나 — infotext 값 `{"args": ["v1;mods=I;I.exposure=0.3"]}` 또는 인자 경로 사전
    `{"args": [{"I.exposure": 0.3}]}`(`enabled` 생략 = 켬). `[true]`(기본값으로 켜기) · `[true, true]`(마스킹까지) · `[false]` 도 뜻이
    같습니다.
  - 패널의 67개 인자는 내부 형식입니다(script-info 에 보이지만 바뀔 수 있음). Python 에서 훅에 579개를 그대로 넘기는 호출은 그대로
    읽습니다.
- **고침**: 패널 소개 글의 "M1~M10 … C1~C5" 가 취소선으로 보이던 것(Gradio 4.40 Markdown 이 `~` 쌍을 취소선으로 그림 — README 의 같은
  줄도 GitHub 에서 그랬음, `–` 로 바꿈), Masking · Debug 아코디언이 안의 체크박스와 같은 elem_id 를 쓰던 것(`…_masking_panel` ·
  `…_debug_panel` 로 바꿔 겹치는 id 4개 → 0).
- **실제 Forge 대조 (Anima 3.8B, 1024², 50 스텝, ER SDE · Beta57, 시드 11)**: v0.31.0 패널로 만든 기준 4개 — Colorcraft 끔 ·
  I 노출 0.3 · M1 마스크로 I 노출 −0.4 · I 노출 0.3 + II 채도 −0.3 — 를 새 패널에서 같은 값으로 다시 만들면 픽셀 md5 와 PNG 파일 md5 가
  모두 같습니다(infotext 도 바이트 단위로 같음, 반복 생성도 같음). API 의 압축 첫 인자(`{"I.exposure": 0.3}` · 문자열)와 한 겹 감싼 579개
  배열은 UI 와 같은 이미지이고, 감싸지 않은 579개는 Colorcraft 없이(끈 것과 같은 이미지) `not applied: …` 상태를 남깁니다. XYZ
  `[Colorcraft] Exposure` 0 · 0.3 · −0.3 은 칸마다 값대로 따로 생성됐습니다. 실제 페이지 점검 94개 중 92개 통과 — 레이아웃 열 "켜짐"
  표시, 붙여 넣기 16개(↙ · PNG Info, 이 확장 · 원본 · 포크 키), Reload UI, img2img 독립. 나머지 2개는 항목을 바꿀 때 Type 드롭다운
  글자가 약 0.35~0.49 초 뒤에 바뀌는 표시 지연이고(값 자체는 그보다 먼저 바뀌며 생성은 늘 맞음), 고치지 않았습니다.
- **한계**: 붙여 넣기는 조합 편집기에 놓이지 않고, 항목을 고른 뒤 약 2 프레임 안의 편집은 로드에 덮입니다(손으로는 닿지 않음). 선택 줄은
  라디오라 v0.31.0 탭의 tablist 역할이 없고 편집기 칸 이름에 항목이 붙지 않습니다(어느 항목인지는 고른 라디오와 `Reset II` 같은 버튼
  이름). 브라우저 전용 갱신 · 숨은 칸 · 라디오 change 같은 Gradio 4.40 동작에 기대므로 Gradio · Forge Neo 를 올리면 다시 확인해야
  합니다. ui-config.json 에 남은 옛 키 6개(`…/Tabs@script_*_colorcraft_samextra_{modifier,mask,combo}_tabs/selected`)는 쓰이지 않습니다
  (해 없음).
- **검증**: Colorcraft 테스트 모듈(`tests/test_colorcraft*.py`) 135 → 184개 — 무작위 설정 400개 × 세 배치(새 state · 지난 state · rev 불일치)가 v0.31.0 위치
  인자와 같은 `Config` · infotext · 체인, 실제 훅이 krea2 · zimage · flux2 에서 v0.31.0 과 비트 단위로 같음(평가 1,920번 `torch.equal`,
  그중 1,884번은 x0 를 바꿈, CPU), API 형식 · 거절 · 감싼 배열, UI · API 붙여 넣기(이 확장 · 원본 · 포크), Python ↔ JS 대조(커밋 ·
  요약 각 무작위 설정 150개, 무작위 세션 60 × 40 스텝), 이벤트 배선(입력 · 출력 id 와 순서 고정). JS 21개(Gradio 처럼 부르는 vm 테스트와
  jsdom — 실제 `notebook.js` 빠른 드롭다운 · `colorcraft_sliders.js` 와 함께). 돌연변이 37/37 검출. 헤드리스 Chrome 으로 실제 패널을 띄운 프로세스 안 Gradio
  4.40 에서 점검 T1–T18 · T22(끌기 · 입력 직후 Generate 40/40, 빠른 드롭다운이 항목 전환을 44–58 ms 에 따라감), 편집기 스크립트 ·
  스키마가 없는 페이지, 실제 입력 무작위 480 동작(확인점 81개)과 검토 하네스 16,000 스텝 모두 실패 0.

### 진행 막대 — 실제 Forge 에서 찾은 버그 4개 (바뀜 — 표시만, 생성 결과 같음)

v0.31.0 에서 남겨 둔 실제 Forge 확인(설정을 켜고 헤드리스 Chrome 으로 테스트 Forge 에서 실제 생성)을 하며 찾았습니다.

- **테마 강조색 막대 위 글자**: Forge Default 테마에서 채운 부분 위 글자가 읽히지 않았습니다 — 밝은 테마는 주황 막대(`#f97316`) 위
  주황 글자(`#ea580c`, 대비 1.2:1), 어두운 테마는 흰 글자(2.9:1). 글자색을 `--button-primary-text-color` 대신 테마의 실제 강조색(hex ·
  `rgb()` · `oklch()`)의 밝기로 고르고(이제 두 테마 모두 어두운 글자), SAM Extra 테마를 바꾸면 다시 고릅니다. SAM Extra 테마의 모양은
  그대로입니다.
- **'빨간 글자' 중단 표시**: Forge Default 어두운 테마에서는 Gradio 의 `--error-text-color` 가 오류 바탕 위 글자색(`#fef2f2`)이라 거의
  흰색으로 나왔습니다. 그 테마에서만 오류 빨강(`--error-icon-color`, 실측 `rgb(239,68,68)`)을 씁니다(`style.css`). 밝은 테마
  (`rgb(185,28,28)`)와 SAM Extra 테마는 그대로입니다.
- **첫 스텝 전 미끄러짐**: 기본 방식 Smooth > Accurate 에서 서버 진행률이 0 인 첫 스텝 전(모델 불러오기 등)에도 막대 · 글자가 조금씩
  올라갔습니다(실측 14.3초 동안 7.7% — 막는 값이 없어 약 3분 반이면 99.2%). 이제 그동안 0% 에 머뭅니다. 상류에서 온 미끄러짐은
  Smooth ~ Accurate 에만 남겼습니다.
- **끝과 복원의 글자**: 마지막 패스 뒤 Forge 의 `nextjob` 이 스텝을 0 으로 돌려 디코드 · 저장하는 동안 `0/20 • 99% • ?` 로 보이던
  것을 끝난 패스(`20/20`)로 보입니다. 새로 고친 뒤 Forge 의 Restore progress 로 붙은 작업이 0 에서 출발해 한참 뒤처지던 것(32/80
  스텝에서 `1%`, 실제 56% 일 때 `19%`)은 처음 '돌고 있음' 을 본 응답에서 그 자리로 옮깁니다(`29/80 • 37%`).
- **검증**: 실제 Forge(테스트 Forge, 헤드리스 Chrome, 20 스텝)에서 — 막대는 뒤로 가지 않음(100 ms 표본 227개, 가장 큰 도약 2.16%),
  이 탭의 작업만 물음(요청 102개, 간격 중앙값 209 ms, 끝나면 멈춤), Forge 막대는 숨김, 중단(`10/20 • 41% • 중단됨` — Forge 가 작업을
  놓을 때까지 남았다가 사라짐), 연달은 생성, img2img, 다른 클라이언트의 API 작업에는 그리지 않음(뒤에 선 이 탭 작업은 `Waiting...` →
  `In queue: 1/1`), 막대를 끄면 Forge 막대, Settings 의 Apply 로 바로 반영, 설정 변형(부드러움 세 방식 · 글자 형식 · 높이 · 색 · 중단
  표시 · 움직임 줄이기). `GET /sam-extra/progress`: 헤더 없음 403 · `X-SAM3-Notebook: 1` 200 · 257자 `id_task` 400 · POST 405,
  `no-store`, `/openapi.json` 에 없음. Forge Default 밝은 · 어두운 테마는 페이지 안에서 테마를 바꿔 확인했습니다. JS 43개(새 6개 —
  5개는 고치기 전 코드에서 실패, 1개는 Smooth ~ Accurate 의 미끄러짐을 지킴), Python 62개(새 1개).
- **한계**: `--gradio-auth` · `--api-auth` 를 켠 실제 Forge(Gradio 4.40 앱 안에서만 확인: 쿠키 없음 401 · 로그인 200, 자격 없음 401 ·
  Basic 200), 배치 2 이상 · Hires, "Don't Interrupt in the middle", 숨은 탭에서 돌아온 뒤 맞추기는 실제 Forge 에서 보지 않았습니다.
  그대로 남은 것: 작업마다 마지막 1초쯤은 ETA 가 `?`(상류와 같음), 작업 시작 뒤 첫 응답까지 약 0.2초 글자가 빔, 페이지를 연 뒤 약
  3.4초는 Forge 가 시작할 때의 옵션을 내주므로 Forge 시작 뒤 꺼 둔 막대가 잠깐 보였다 사라짐.

### MCP 서버 — 실제 클라이언트로 찾은 버그 4개 (바뀜)

v0.31.0 에서 남겨 둔 실제 Forge 생성과 실제 클라이언트 연결을 확인하며 찾았습니다. 고친 상류 파일의 머리 주석과
`THIRD_PARTY_NOTICES.md` 의 수정 목록에 적었습니다.

- **불러온 체크포인트가 자기 생성 기록과 맞지 않던 것** (`forgeneo/history.py` `normalise_checkpoint`): Forge 는 해시를 알면 체크포인트
  이름 끝에 붙이는데(`name.safetensors [0123456789]`, `sd_model_checkpoint` 도 이 모양), 상류는 확장자를 먼저 떼려다 실패해
  `.safetensors` 가 남았습니다. 그래서 해시가 붙은 이름(보통의 Forge 설정)에서는 샘플링 값 추천이 기록 없이 기본값으로
  떨어졌습니다. 이제 `[해시]` 를 먼저 뗍니다. 실측(테스트 Forge, Anima 3.8B): 기록 0개 · 50 스텝(인스턴스 기본값) · turbo
  "unknown" → 기록 17개 · 20 스텝 · turbo "no".
- **체크포인트 전환의 프리셋 신호** (`forgeneo/profile.py` `preset_for_checkpoint`): Forge 가 프리셋마다 적어 둔 체크포인트
  (`forge_checkpoint_<프리셋>`)와 `models` 목록에서 고른 이름이 `[해시]` 만 달라도 신호를 잃어, 모델을 바꿀 때 그 체크포인트의 VAE ·
  텍스트 인코더 없이 불러올 수 있었습니다. 이제 `[해시]` 를 빼고 비교합니다(단위 테스트로만 확인 — 실제 모델 전환은 하지 않음).
- **프롬프트 방언** (`forgeneo/identity.py`): 위 수정으로 기록이 읽히자, Anima 의 품질 태그(`masterpiece, best quality`, `score_*`)가
  든 과거 프롬프트 때문에 `prompt_dialect` 가 Anima 체크포인트를 높은 확신으로 Illustrious(또는 Pony)라고 했습니다. 이제 과거
  프롬프트는 아키텍처가 정하지 못하는 경우(SDXL 계열)의 갈래를 고르거나 아키텍처의 방언을 확인할 뿐 뒤집지 않습니다. 실측: `anima`.
- **Anima 태그를 산문으로 본 것** (`forgeneo/history.py` `_looks_danbooru`): 역시 기록이 읽히자 드러났습니다. Anima 는 태그를 공백으로
  써서(`short hair, black hair, 1boy, solo`) "natural language (94% prose)" 로 보고했습니다. 이제 방언 모듈의 태그 판별도 셉니다.
  실측: "danbooru tags (100%)".
- **테스트**: `tests/test_mcp_layout.py` 가 `tomllib` 을 무조건 불러와, Python 3.10(서버 자신의 uv 환경 — MCP SDK 가 있어 등록 테스트가
  실제로 도는 곳)에서 MCP 테스트가 모두 깨지던 것을 그 테스트 하나만 건너뛰게 고쳤습니다.
- **검증**: MCP SDK(uv 환경: mcp 2.3.0 · Python 3.10.11)의 stdio 클라이언트로 테스트 Forge(7860)에 붙여 — 도구 10개와 읽기 전용 ·
  파괴 표시, 설정 파일에서 읽은 권한(생성만 켬), 불러온 체크포인트 · 프리셋 · 체크포인트 21개 · LoRA 385개, 읽기 도구 응답, 실제 생성
  한 장(768×768, 16 스텝, 7.4초 — 디스크의 파일을 생성 정보로 찾아 돌려줌, 대체 폴더는 만들지 않음), 거절(모델 불러오기 · 중단 ·
  건너뛰기 `denied_by_policy`, 내려받기가 꺼져 있으면 네트워크 요청 없음, 화소 상한 초과, 네트워크 공유 init 이미지, 빈 프롬프트,
  모르는 동작), 서버가 도는 중 설정 파일을 바꾸면 다음 호출부터 적용(문자열 `"true"` 는 끔, 깨진 JSON 은 모두 끔), 한국어 · 일본어
  인자 · 경로. **Claude Code**(`claude` 2.1.252, `-p`, 임시 MCP 설정 · `--strict-mcp-config`, 읽기 도구 5개만 허용)에서는 서버가
  연결되고(connected) 도구 5개가 올라오는 것까지 확인했습니다 — 안에서 띄운 CLI 가 로그인(OAuth 세션 갱신)에 실패해 Claude Code 를
  통한 도구 호출은 확인하지 못했습니다. Python 테스트 312 → 327개(새 `test_mcp_checkpoint_titles` 7 · `test_mcp_observed_prompts` 8) —
  Forge venv(SDK 없음, 5개 건너뜀)와 uv 환경(SDK 2.3.0, 1개 건너뜀) 모두 통과.
- **한계**: 생성 중 `progress`, img2img, 동시에 돈 다른 생성과의 분리, 스위치를 켠 모델 불러오기 · 중단 · 내려받기는 실제로 돌리지
  않았습니다. 그대로 남은 것: `capabilities` 는 Forge 가 API 출력 폴더를 처음 만들기 전까지 `readable:false`, `generate` 결과에 쓴
  스케줄러 · shift 값이 없음(Forge 응답에 없음), Forge 의 "Output Directory" 덮어쓰기를 켜면 서버가 만든 이미지는 생성 기록에 들어가지
  않음, Settings 에 보이는 등록 명령에는 `FORGE_URL` · `SAM_EXTRA_MCP_FORGE_CONFIG` 가 없음(README **등록** 절).

### CI — 프런트엔드 잡 따로 (Node 24) · Python 잡 복구 (결과 같음 — 테스트와 CI 만)

- **프런트엔드 잡 따로, Node 20 → 24**: 프런트엔드 검사(`node --check javascript/*.js` · `npm ci` · `npm test`)를 Python 과 따로 도는
  `frontend` 잡으로 나눴습니다. 한 잡이던 때는 Python 단계가 실패하면 JS 단계가 통째로 건너뛰어져, GitHub 에서는 적어도 2026-09-19
  부터 JS 검사가 돌지 않았습니다. Node 20 은 타입 지우기가 없어 앱 원본 `compositionPrompt.ts` 대조 11개 중 9개를 건너뛰었는데, 이제
  Node 24 로 모두 돌고, CI(`CI=true`)에서 타입 지우기가 없는 Node 로 돌면 건너뛰지 않고 실패합니다(로컬은 예전처럼 건너뜀). `test`
  잡 이름(`CI / test`)은 그대로입니다.
- **Python 잡 복구 (오류 17개 → 0)**: GitHub 의 Python 잡은 v0.31.0(`6adeb83`)에서 `errors=17` 로 실패했습니다(확인한 2026-09-19 이후
  실행도 모두 실패).
  - Forge 본체 파일을 실행하는 원본 대조 테스트(오류 10개 — `test_anima_safe_pag` · `test_cns_origin` · `test_dave_origin` ·
    `test_dcw_origin` · `test_detail_daemon_origin` · `test_ui_config_migration`)는 Forge 체크아웃이 없으면 그 파일이 필요한 테스트만
    이유를 남기고 skip 합니다(`tests/_forge_checkout.py` — 확장이 놓인 `<Forge>/extensions/` 위, 없으면 `SAM3_FORGE_ROOT`, 없으면 개발
    PC 의 설치에서 찾음). Forge 를 찾았는데 파일이 없으면 여전히 실패하고, Forge 안에서는 하나도 건너뛰지 않습니다. Forge 로더는 import
    때가 아니라 처음 쓸 때 불러, Forge 없이도 나머지 테스트는 돕니다.
  - `test_speed_origin` 4개(scipy 없음): `requirements-dev.txt` 에 SPEED 원본 대조용 `scipy` 와 `PyWavelets`(공식 SPEED 대조 · haar
    테스트가 CI 에서 조용히 건너뛰던 것)를 더했습니다. 확장 코드는 둘 다 import 하지 않고, 둘 다 Forge venv 에 있습니다.
  - `test_anima_vae_2x` 1개: gradio 5 의 Accordion 은 `gr.Blocks` 안에서만 만들어지므로 Forge(`modules/ui.py`)처럼 `gr.Blocks()` 안에서
    UI 를 만듭니다. gradio 는 묶지 않았습니다 — 4.40 은 `pillow<11` 이라 `Pillow>=11.1.0` 과 함께 설치되지 않습니다(Forge 는 gradio 를
    먼저 설치해 피함). CI 는 gradio 5.x, Forge 는 4.40 으로 돕니다.
  - `test_tipo_runtime` 2개: CI 가 transformers 5.18 을 받았습니다 → `transformers<5`(4.57.6 · huggingface-hub 0.36.2 · tokenizers
    0.22.2 — Forge Neo 와 같음).
  - Windows autocrlf 체크아웃에서는 `test_negpip_vendor` 가 `sam3ext/negpip/LICENSE` 의 원래 바이트로 해시를 재 실패했습니다 → 다른 원본
    고정 테스트처럼 줄 끝을 LF 로 맞춰 비교합니다(`.gitattributes` 는 Forge 가 `git clone` 으로 설치하는 방식까지 바꾸므로 넣지 않음).
- **transformers 를 `<5` 로 묶은 이유**: 상류 원문 그대로(SHA 고정)인 TIPO 모델 코드(`sam3ext/tipo/kohaku`)는 transformers 4 용입니다.
  5.x 에서는 `_tied_weights_keys` 가 목록이라 `save_pretrained` 가 실패하고(테스트 보조 함수만 씀 — 이제 파일을 직접 써서 4.57.6 에서
  가중치 파일이 바이트까지 같음), 더 큰 문제로 `from_pretrained` 가 RoPE 버퍼(`inv_freq`)를 채우지 않아 위치 인코딩이 0 이나 쓰레기
  값이 됩니다 — 런타임이 조용히 틀린 출력을 냅니다. Forge Neo 는 4.57.6 이라 지금은 영향이 없고, 불러온 `inv_freq` 가 새로 계산한 값과
  같은지 보는 검사를 `RealLoaderTests` 에 더했습니다(transformers 5.18 에서 실패함을 확인).
- **검증**: Forge 가 없는 새 Python 3.13 venv 에 `ci.yml` 과 같은 설치(CPU torch 2.14.1, gradio 5.50.0, transformers 4.57.6)로 — 고치기
  전 `errors=17`(GitHub 과 같은 17개) → `Ran 2698 tests … OK (skipped=342)`, 모든 파일을 CRLF 로 바꿔도 통과, Forge 를 찾게 하면
  skip 208. WSL · Docker 가 없어 Linux 가 아니라 Windows 에서 돌렸습니다. Colorcraft 공유 편집기를 넣은 트리로 다시 돌려 `Ran 2747 tests
  … OK (skipped=347)`, 프런트엔드 잡은 Node 24.19 로 CI 처럼(`CI=true`) 돌려 204개 중 203개 통과(1개는 Forge 의 `progressbar.js` 가
  옆에 있어야 도는 진행 막대 테스트라 건너뜀). GitHub Actions 위에서는 이 판을 올린 뒤 처음 돕니다.
- **한계**: GitHub 이 `actions/checkout@v4` · `setup-node@v4` · `setup-python@v5` 에 다는 "Node.js 20 is deprecated" 경고는 테스트가
  쓰는 Node 가 아니라 그 액션 자체의 실행 환경이라 그대로 두었습니다. CI 에는 spandrel 이 없어 그 테스트 8개는 건너뜁니다.

### 설정 · 라이선스 · 검증

- **설정**: 새 설정은 없습니다.
- **라이선스**: `THIRD_PARTY_NOTICES.md` — forgeneo-mcp 수정 목록에 `history.py`(`normalise_checkpoint` · `_looks_danbooru`) ·
  `profile.py`(`preset_for_checkpoint`) · `identity.py`(`resolve`)를 더하고, ComfyUI-Colorcraft 절에 패널이 이제 sam-extra 가 새로 쓴
  공유 편집기임을 적었습니다.
- **검증**: Python 2906개 통과(skip 22 — 그중 5개는 MCP SDK 가 있어야 도는 등록 테스트, Forge venv · CPU), JS 204개 통과(진행 막대 6 ·
  Colorcraft 편집기 21개를 더함), `node --check javascript/*.js` 이상 없음. 새 Python 테스트는 Colorcraft 49 · MCP 15 · 진행 막대 1개이고,
  CI 를 위해 고친 테스트는 Forge 안에서 하나도 건너뛰지 않습니다. 실제 Forge 에서는 진행 막대 · MCP 서버와 Colorcraft 공유 편집기
  (위 **실제 Forge 대조**)를 확인했습니다. GitHub CI 와 같은 깨끗한 Python 3.13 · gradio 5.50 venv(Forge 없음)에서 공유 편집기까지 넣고
  2747개 통과(skip 347), 프런트엔드 잡 204개 통과(1개는 Forge 의 `progressbar.js` 가 있어야 해 skip).

## v0.31.0 — Colorcraft · Anima SPEED · Extra Schedulers · Extra Samplers · 진행 막대 · MCP 서버 · 구도·카메라

새 기능 일곱 가지입니다. 샘플링 중 latent 색 보정(Colorcraft), 초반 스텝을 저해상도로 돌리는 Anima SPEED(실험), 스케줄러
6개(Extra Schedulers), 샘플러 5개(Extra Samplers), 부드러운 진행 막대, 같은 PC 의 MCP 클라이언트가 이 Forge 로 이미지를 만들게
하는 MCP 서버, 사용자 앱에서 옮긴 구도·카메라 칸(태그로 시점 잡기)입니다. Colorcraft·SPEED·진행 막대는 기본으로 꺼져 있습니다. Extra Schedulers·Extra Samplers 는 Schedule type·
Sampler 에서 골랐을 때만 쓰이고, MCP 서버는 Forge 밖에서 따로 실행합니다(기본은 생성만 허용). 편입한 상류 코드의 출처·커밋·
고지는 `THIRD_PARTY_NOTICES.md` 에 있습니다.

괄호 표시는 v0.30.0 과 같습니다: (결과 같음) = v0.30.1 과 같은 설정·시드에서 이미지 동일, (새 기능) = v0.30.1 에 없던 기능이라
비교 대상 없음, 토글 = 끌 수 있는 설정(뒤에 기본값).

### Colorcraft — 샘플링 중 latent 색 보정 (새 기능, 토글 — 기본 꺼짐)

- **Colorcraft (sam-extra)**: [muerrilla/ComfyUI-Colorcraft](https://github.com/muerrilla/ComfyUI-Colorcraft)(`d28ac6a`, MIT)의
  최신 계산을 Forge 로 옮겼습니다. 매 모델 호출의 CFG 결과(x0)를 latent 의 색 기저 방향으로 밀어 노출·톤 압축·대비·클래리티·
  샤프니스·색온도·틴트·바이브런스·채도·채도 대비·Lab 축·색 이동을 조정합니다. 추가 모델 호출은 없습니다. Forge 구조와 Flux2 벡터는
  [aoleg/ComfyUI-Colorcraft](https://github.com/aoleg/ComfyUI-Colorcraft)(`f00066c`, MIT)를 따릅니다. txt2img·img2img 의
  **Colorcraft (latent 색 보정 · ComfyUI-Colorcraft)** 아코디언에 있습니다(txt2img 는 ANIMA 튜닝 열의 Anima Optimal Scale 바로 아래
  — Optimal Scale 도 이 열에 등록해, v0.30.x 처럼 '도구·실험' 묶음으로 가서 더 보기를 닫으면 숨지 않습니다).
  - 모델: Anima·Qwen-Image·Krea 2·Wan(krea2 벡터), Flux·Chroma·Lumina 2·Z-Image(zimage), Flux 2 Klein·ERNIE-Image(flux2).
    SD 1.5·SDXL 은 Contrast·Color Shift 만 동작하고 이유를 `SAM Extra Colorcraft status` 에 남깁니다.
  - 수정자 스택 I~X: 탭마다 원본 노드 종류(Advanced·Basic·Luma·Chroma·Chroma Plus·Punch·Shift)와 자기 스케줄(Strength·
    Start·End·Advanced Schedule), 적용 패스(Base·Hires·Both), 마스크를 고릅니다. Masking 에 축 마스크 M1~M10(clarity·
    sharpness 디테일 축 포함, Blur·Spread·Normalize·Contrast)과 조합 C1~C5(and·or·subtract·xor), Debug 에 축 투영·마스크
    미리보기가 있습니다.
  - 원본 노드와 비트 단위로 같습니다 — krea2·zimage·flux2 에서 20 가지 설정을 스케줄 σ 와 그 사이(2차 샘플러) 마다
    대조(Forge 의 실제 latent format 클래스, CPU). img2img·hires 는 Forge 가 실제로 도는 σ 구간(`sampling_sigmas[offset:]`)
    으로 스케줄을 만듭니다.
  - 색 기준점은 생성 전에 `vae.encode` + `process_in` 으로 한 번 만듭니다(Anima 의 `encode_first_stage` 는 레퍼런스 latent 를
    덮어써서 쓰지 않음). 같은 VAE 면 다음 요청(XYZ 칸·API)도 다시 인코딩하지 않습니다(최대 32개, VAE 가 바뀌면 버림). 오류가
    나면 그 호출은 입력을 그대로 넘기고 status 에 남깁니다. fp16·bf16 은 fp32 로 계산해 원래 dtype 으로 돌려줍니다. sam-extra 의
    다른 post-CFG(Optimal Scale·가이던스) 뒤에 맨 마지막으로 돕니다.
  - Anima SPEED·Euler (SMEA) Dy CFG++ 와 함께: SPEED 의 전환 뒤에도 보정을 이어 갑니다 — SPEED 가 다시 내놓는 σ 목록(같은 실행
    표시)을 따라가 스텝마다 그 목록에서 σ 를 찾고(Detail Daemon·DAVE·MG/HiGS·HiFlow 와 같음), 저해상도 격자에서도 마스크 blur 가
    이미지 픽셀 크기를 지킵니다. Dy/SMEA 보조 스텝은 보정하지 않고 넘겨 스텝마다 한 번만 보정하고, status 에
    `Dy/SMEA sub-steps passed through xN` 으로 셉니다.
  - infotext: `SAM Extra Colorcraft`(붙여 넣으면 패널 복원), `SAM Extra Colorcraft status`. 원본·포크 확장이 남긴
    `Colorcraft` 키도 붙여 넣을 수 있습니다. XYZ 축 `[Colorcraft] …` 14개(탭 I). Settings → **SAM Extra Colorcraft**.
    API 는 위치 인자 579개 대신 첫 인자 하나(infotext 값 또는 인자 경로 사전)로도 받습니다.
- **한계**: GPU 는 Anima 3.8B 에서만 확인했습니다(노출·색온도·채도가 맞는 방향, SPEED·Dy/SMEA 와 함께 보정 횟수 같음 — VRAM 은
  재지 않음, 원본 노드와의 비트 대조는 CPU). CFG++ 샘플러에서는 더 세게 걸립니다 —
  Anima 3.8B(시드 11, 28 스텝)에서 노출 +0.3 이 평균 밝기를 Forge 의 Euler CFG++ 로 약 +27(186 → 214, Euler Dy CFG++·Euler SMEA Dy
  CFG++ 도 같음), Res Multistep 으로 +14, Euler 로 +8 올렸습니다. CFG++ 스텝은 보정된 x0 를 스텝의 일부만큼이 아니라 통째로 다음
  latent 에 싣기 때문으로, sam-extra 의 버그가 아니라 CFG++ 의 방식입니다(원본 ComfyUI-Colorcraft 도 같은 post-CFG 지점에 겁니다).
  CFG++ 샘플러에서는 세기를 절반쯤으로 두세요. flux2 보정값은 포크가 예전 원본 보정 기준으로 잰 값입니다. 패널이 큽니다 — 탭마다
  스크립트 인자 579개, Gradio 블록 약 1,200개, `/config` 약 479 KB 로 원본 Forge 패널(인자 495개, 블록 약 1,180개, 476 KB)과
  비슷합니다. XYZ 축은 탭 I 만 바꾸고, 원본 Forge 확장의 스케줄·마스크 그래프와 탭 색 표시는 옮기지 않았습니다.
- **Detail Daemon·Guidance Suite 내부 정리 (결과 같음)**: `scripts/anima_detail_daemon.py` 와 `scripts/anima_safe_pag.py`(DAVE
  게이트 등) 두 곳에 똑같이 있던 `_forge_sampling_offset`·`_is_img2img_request` 를 `sam3ext/guidance/sigmas.py` 로 옮겨 Colorcraft 와
  함께 씁니다. 기존 테스트와 옛 코드 대조 테스트가 같은 값을 확인합니다.
- **검증**: 새 Python 테스트 139개(원본·포크 대조, 훅, infotext, 스크립트, Debug, 벡터, σ 오프셋), JS 10개.

### Anima SPEED — 저해상도 선행 샘플링 (실험, 토글 — 기본 꺼짐)

- **새 아코디언 Anima SPEED (저해상도 선행 샘플링 · 실험)**(txt2img 는 ANIMA 튜닝 열, img2img 에도 있음): SPEED(Spectral
  Progressive Diffusion, arXiv 2605.18736)를 Forge 샘플러에 얹습니다. 초반의 노이즈가 지배적인 스텝을 DCT 로 줄인 저해상도
  latent 에서 돌리고, 전환 σ 에서 고주파를 σ 크기 노이즈로 채워 원래 크기로 넓힌 뒤(κ = r/(1+(r−1)σ) 로 보정) 고른 샘플러로
  이어서 샘플링합니다. 앞쪽 스텝의 토큰이 줄어 빨라지는 대신 이미지가 달라집니다. flow 모델 전용입니다(SDXL·SD1.x 는 건너뜀).
  Anima 3.8B 실측(832×1216, Res Multistep + Linear Quadratic 28 스텝): 기본값은 5/28 스텝 저해상도로 약 1.13배, manual σ 0.7 은
  22/28 스텝으로 약 1.8배 빠르고 구도가 크게 바뀝니다(저해상도 스텝은 토큰 1/4 에 약 3배 빠름).
- **두 방식**: `transition`(기본) = 공식 howardhx/speed·aoleg/ComfyUI-SPEED — 전환 스텝의 σ 만 정렬값으로 바꿈. `respace` =
  sorryhyun/ComfyUI-Spectrum-KSampler — 남은 σ 를 비율로 다시 배치하고 저해상도 격자를 짝수로 맞춤(데스크톱 앱 ComfyUI 팩이 부르는
  SpectrumSPDKSampler 와 같은 SPD 기하만 맞춤 — 원본의 Spectrum 캐싱은 들어 있지 않고, 홀수 latent 는 원본이 짝수로 채우므로 다름).
- **전환 σ**: 파워 스펙트럼 프리셋(`anima` 기본, flux·flux2·krea-2·z-image·wan21·custom)과 δ, 1024 px 기준에 고정하는 Adaptive
  delta, Forge 의 고정 shift 용 `neo_shift` divisor(1.03), 또는 직접 적은 σ. Anima 32스텝 Beta 기준 기본값은 약 6/32 스텝,
  manual σ 0.7 은 약 18/32 스텝이 저해상도입니다.
- **원본과 다른 점(버그 수정)**: FFT 고주파 노이즈의 `/√2` 제거(공식 ca7801c9 와 같음, aoleg 판에는 남아 있음) · img2img·Hires 가
  전환 σ 보다 낮은 σ 에서 시작하면(저해상도 스텝이 없으면) 아무것도 하지 않음(원본은 시작 latent 의 고주파를 노이즈로 바꾸고 σ 를
  올림) · 배치 이미지마다 자기 시드로 확장·SDE 노이즈를 뽑아 같은 시드를 혼자 만든 결과와 같음 · SDE 샘플러(DPM++ SDE 계열)의
  저해상도 구간에 시드별 Brownian 노이즈 · 마지막 전환이 스케줄 안에 없으면 작은 latent 를 내지 않고 건너뜀 · Detail Daemon·DAVE·
  Momentum·HiGS 가 바뀐 σ 를 찾도록 `sampling_sigmas` 를 실행 동안 갱신 · img2img·Hires 의 저해상도 시작 latent 를 flow 형태로 맞춤
  (설정 `sam3_speed_img2img_rescale`, 기본 켬 — κ 가 기대는 진폭 관계의 역이라 끄면 이미지 성분이 r 배 센 채로 시작, 끄면 원본 동작) ·
  DWT 는 단계 비율 r 이 2 가 아니면 미리 거절(공식은 저해상도 스텝을 돈 뒤 오류).
- **쉬는 경우**(이유는 infotext `Anima SPEED status`): 마스크·인페인트, 레퍼런스 latent(Anima·Flux Kontext·Flux.2 Klein·
  Qwen-Image-Edit·Krea 2), Wan 2.2 I2V 조건(concat_latent), PiD(lq_latent), ControlNet(LLLite 포함), flow 가
  아닌 모델(SDXL·SD1.x), Forge 내장 Spectrum Integrated, Restart·UniPC 와 σ 목록이 없는 샘플러, 저해상도 스텝이 없는 패스, Hires 패스
  (`Apply to Hires pass` 를 켜야 적용).
- **다른 새 기능과 함께**: 전환 뒤 다시 내놓는 σ 목록에 같은 실행 표시(`mark_republished`)를 붙여 Colorcraft 가 원래 크기 꼬리도
  바뀐 σ 로 계속 보정합니다. `sam_extra_step_offset` 을 받는 샘플러에는 구간마다 시작 스텝(패스 기준)을 넘겨, Euler (SMEA) Dy CFG++
  의 보조 스텝이 구간마다 다시 돌지 않고 SPEED 없이 돌 때와 같은 스텝에서 돕니다. 두 기능은 함께 돌고, Forge 의 Spectrum Integrated
  아래에서는 둘 다 물러나 각자 status 에 이유를 남깁니다.
- **기록**: infotext `Anima SPEED`(설정, 붙여 넣으면 되살아남)·`Anima SPEED status`·`Anima SPEED img2img rescale`. XYZ 축 6개
  (`[Anima SPEED] Enable/Mode/Manual sigma/Delta/Sigma divisor/Scale`). 설정 `sam3_speed_log`·`sam3_speed_img2img_rescale`.
- **한계**: 다단계 샘플러는 전환마다 이력을 새로 시작합니다(공식과 같음). img2img-hires-fix 확장이 따로 도는 패스에는 걸리지
  않습니다. 콘솔의 tqdm 막대는 구간마다 하나씩 나옵니다(웹 UI 진행률은 패스 전체를 셈). GPU 에서는 배치의 첫 이미지가 같은 시드를
  혼자 만든 것과 조금 다릅니다(CPU 테스트에서는 같음) — 배치 2 와 1 의 GPU 커널이 달라 새 기능을 모두 꺼도 생기는 차이이고(Anima
  3.8B·시드 11·28 스텝, 0–255 화소 평균 절대 차: 모두 끔 7 · Colorcraft 10 · Euler CFG++ 3), SPEED 는 구도가 저해상도 스텝에서 잡혀
  23–26 으로 키웁니다(이 시드에서는 머리색이 바뀜).
- **검증**: 상류 원본(공식 utils·aoleg core·sorryhyun spd_core/spd, SHA-256 고정)과 같은 입력으로 CPU 대조 — transition 모드는
  세그먼트 모양·패치된 σ 가 비트 단위로 같고 출력은 float 오차 안(원본 float32 scipy, 여기 float64 torch), respace 모드는 모델 호출마다
  입력 모양·σ 가 비트 단위로 같습니다. Forge 의 실제 k-diffusion 샘플러 6종으로 배치 = 단일 시드, Forge 실제 `sampling_function` +
  Anima DiT 로 PAG·DCW·TSR·Momentum·HiGS·HiFlow·Detail Daemon 이 크기 변화를 견디는지 확인. 새 Python 테스트 132개.

### Extra Schedulers — 스케줄러 6개 (새 기능)

- **스케줄러 6개 추가**: Schedule type(·Hires schedule type) 목록에 **Cosine · CosineExponential blend · Phi · Laplace ·
  Karras Dynamic · custom** 이 생겼습니다. Cosine 은 처음 내려가는 폭이 작고, CosineExponential blend 는 Cosine 으로 시작해
  Exponential 의 긴 꼬리로 끝나며, Phi 는 (1 − p)^(φ²) 곡선입니다(착상: Extraltodeus 의 Golden Scheduler). Laplace 는 ComfyUI
  `get_sigmas_laplace` 를 그대로 옮겨(arXiv:2407.03297) Forge 처럼 마지막 0 을 붙이고, 노드의 clamp 가 같은 시그마를 되풀이하면
  (flow 모델은 기본 μ 0 에서 앞쪽 절반이 정확히 1) n 스텝을 같은 곡선 중 sigma min~max 안의 구간에 고르게 다시 놓습니다 — 되풀이가
  없으면 노드와 비트 단위로 같고, β 가 0 이거나 곡선 전체가 그 범위 밖이면 μ/β 를 적은 오류로 멈춥니다(Anima 3.8B·28 스텝 실측:
  노드 그대로면 Res Multistep 은 검은 이미지(NaN), Euler 는 정상이지만 14 스텝이 버려짐 — 다시 놓으면 둘 다 정상). Karras Dynamic 은
  Karras 램프에 스텝마다 ρ + 2cos(2πi/n) 지수를 쓰며(ρ 기본 7, Karras 와 같이 쓰는 Settings 의 rho 로 바꿈 — 2 보다 커야 하고,
  약 4 보다 작아 시그마가 도중에 올라가면 그 스케줄을 쓰지 않고 생성을 오류로 멈춤), 이 변형의 출처는 확인되지 않았습니다.
  aoleg/Neo_ExtraSchedulers README 의 이름(`cosine-exponential blend`·`karras dynamic` 등)으로 적힌 Schedule type 도 이 스케줄러로
  읽습니다. 기존 스케줄러는 그대로라 결과가 같습니다(결과 같음).
- **custom 스케줄러**: txt2img·img2img 의 **Extra Schedulers** 아코디언(접힘, txt2img 는 ANIMA 튜닝 열의 Anima 3.8B 아래)에 적은
  식(`m` `M` `n` `s` `x` `phi` `pi` `e`, `+ − * / **`, `sqrt` `exp` `log` 등 함수)을 스텝마다 계산하거나, 시그마 목록(`[1.0, 0.6,
  0.25, 0.1, 0.0]` — 1.0 으로 시작해 0.0 으로 끝나면 sigma max~min 으로 늘림)을 Forge 의 로그-선형 보간으로 스텝 수에 맞춥니다.
  식은 파이썬으로 실행하지 않고 AST 화이트리스트 계산기로만 읽습니다 — 속성 접근·`__import__`·람다·컴프리헨션·문자열·64 를 넘는
  지수·500자 초과는 거절하고, 모든 계산은 유한한 float 이어야 합니다. 잘못된 식·목록은 생성을 오류로 멈춥니다(조용히 다른
  스케줄로 바꾸지 않음). 켜기 체크박스는 없고, 생성이 custom 이나 Laplace 를 쓸 때만 아코디언 값이 쓰입니다.
- **시그마가 그대로인 스텝은 쓰지 않음 (6개 모두)**: 마지막 0 앞에 같은 시그마가 두 번 이어지면 생성을 `ExtraSchedulerError`
  (custom 은 `CustomSchedulerError`, Karras Dynamic 은 `KarrasDynamicError`)로 멈춥니다. 그런 스텝은 Euler 에서는 버려지고, Res
  Multistep·DPM++ 2M 같은 multistep 샘플러는 0 으로 나눕니다(NaN, 검은 이미지). 평평한 구간이 있는 custom 식(상수, sigma min 을
  붙드는 `max(m, …)`)·같은 값이 이어지는 custom 목록·sigma min = sigma max 설정도 해당됩니다. 올라가는 custom 스케줄은 그대로 쓰고,
  Karras Dynamic 은 올라가는 스텝과 제자리 스텝을 모두 거절합니다.
- **infotext·붙여 넣기·XYZ·API**: `Custom scheduler expression`/`Custom scheduler sigmas`(custom 을 쓴 생성만), `Laplace mu`/`Laplace beta`
  (Laplace 를 쓴 생성만, 둘 다)를 남기고 PNG Info 로 되살립니다. XYZ 축 `[Extra Schedulers (sam-extra)] Laplace mu`·`Laplace beta`·
  `Custom expression`·`Custom sigma list` 가 생겼습니다(쉼표가 든 값은 큰따옴표로, μ/β 가 슬라이더 범위 밖이면 그리드 시작 전에 알림).
  API 키는 `alwayson_scripts["Extra Schedulers (sam-extra)"]` 입니다(같은 이름의 아코디언을 가진 aoleg/Neo_ExtraSchedulers 와 겹치지 않게).
- **한계**: Cosine · CosineExponential blend · Phi · Karras Dynamic 과 M~m 을 보간하는 custom 식(예: `m + (M - m) * (1 - x) ** 2`)은
  SD·SDXL 계열(eps/v) 모델용입니다. 모델의 시간 shift 없이 시그마 공간을 나눠 구도·대비가 잡히는 σ = 1 근처를 한두 스텝에
  지나가므로, Anima 3.8B(flow)에서는 물 빠진 듯 대비가 낮고 뿌연 이미지가 나옵니다. Forge 자체의 Karras·Exponential 도 같은
  모습이라(평균 밝기 ≈217·표준편차 ≈46, Linear Quadratic 은 ≈182·≈90) 버그가 아닙니다. flow 모델에는 Simple·Beta·Linear Quadratic,
  Laplace(기본 μ 0), 시간 shift 를 넣은 custom 식(예: `m + (M - m) * 3 * (1 - x) / (1 + 2 * (1 - x))` — shift 3 Simple 에 가까움)을
  쓰세요. flow 모델에서 Laplace 의 μ 를 음수(−1.5·−2)로 두면 곡선이 한두 스텝 만에 1 에서 0.3~0.8 로 떨어져 Anima 에서는 물 빠진
  이미지가 나왔습니다. Laplace 는 Euler 계열 샘플러로 쓰세요 — Anima 에서는 마지막 스텝이 ≈0.19 에서 sigma min(≈0.003)으로 크게 뛰어, 2차
  multistep 샘플러인 Res Multistep 은 잔 입자가 남습니다(시간 shift custom 식도 Res Multistep 에서는 잔 입자가 조금).
  aoleg/Neo_ExtraSchedulers 로 만든 이미지는 README 에 적힌 이름이면 Schedule type 이 되살아나지만, 그 확장의 식·Laplace 값 infotext
  키는 알 수 없어 그 값은 되살아나지 않습니다(그 확장의 실제 이미지로는 확인하지 않음).
- **라이선스**: 라벨은 라이선스 없는 aoleg/Neo_ExtraSchedulers 의 infotext 와 맞췄지만 그 코드는 읽지도 쓰지도 않았습니다
  (README 의 이름만). 편입한 코드는 ComfyUI(GPL-3.0)의 Laplace 함수뿐입니다(THIRD_PARTY_NOTICES).
- **검증**: 새 테스트 146개(파서 보안·식 대조·ComfyUI 원본 대조·Laplace 다시 놓기·제자리 스텝 거절·Forge 실제
  `sd_schedulers`/`get_sigmas`/infotext 조회로 등록 확인). GPU 이미지는 Anima 3.8B 에서만 확인했습니다(위 Laplace·flow 모델
  결과). SD·SDXL 에서의 화질은 확인하지 않았습니다.

### Extra Samplers — 샘플러 5개 (새 기능)

- **새 샘플러**: Forge 샘플러 목록(Sampler·Hires sampler, XYZ `Sampler` 축, API `sampler_name`)에 `ER SDE (Reverse-time)`,
  `ER SDE (ODE)`, `DPM++ 4M SDE`, `Euler Dy CFG++`, `Euler SMEA Dy CFG++` 가 생깁니다. 이름은 aoleg/Neo_ExtraSchedulers 와 같아서
  그 확장으로 만든 이미지의 infotext 를 붙여 넣어도 같은 샘플러가 잡힙니다(그 확장의 코드는 라이선스가 없어 쓰지 않았고 README 만
  참고했습니다). 같은 이름이 이미 목록에 있으면(그 확장이 설치된 경우) 이 확장은 그 이름을 건너뛰고 콘솔에 한 번 남깁니다.
- **ER SDE 두 항목**: Forge 내장 `sample_er_sde` 에 ComfyUI `SamplerER_SDE` 노드의 나머지 두 잡음 척도를 넣은 것입니다
  — Reverse-time h(λ)=λ^(η+1), ODE h(λ)=λ(잡음 없음). 내장 **ER SDE** 는 그대로입니다(노드의 ER-SDE η=1 과 같음을
  테스트로 확인). **Extra Samplers** 아코디언(txt2img 는 ANIMA 튜닝 열, img2img 는 스크립트 영역, 접힘)의 `ER SDE max stage`
  (1–3, 기본 3)·`ER SDE eta`(0–10, 기본 1)가 이 두 항목에만 쓰이고, 기본값이 아니면 infotext `ER SDE max stage`·`ER SDE eta` 로
  남습니다. XYZ 축 `[Extra Samplers] ER SDE max stage`·`[Extra Samplers] ER SDE eta`. API 는 `alwayson_scripts["Extra Samplers"]`
  (`[max_stage, eta]`, 생략 가능).
- **DPM++ 4M SDE**: Clybius/ComfyUI-Extra-Samplers(BSD-3-Clause)의 4차 다단계 SDE 를 Forge DPM++ 3M SDE 와 같은
  방식으로(half-log-SNR) 옮겨 Anima·Flux 같은 flow 모델에서도 돕니다. 3차까지의 스텝은 Forge 3M SDE 와 비트까지 같습니다.
  옵션·Eta·Sigma noise·스케줄러(Automatic = exponential, 끝에서 두 번째 σ 버림)는 3M SDE 와 같습니다.
- **Euler (SMEA) Dy CFG++**: Koishi-Star/Euler-Smea-Dyn-Sampler(Apache-2.0)의 Dy(2·3번째 스텝에 반 해상도 보조 스텝)·SMEA
  (0번째 스텝 ×1.25 보조 스텝)를 Forge 의 CFG++ 갱신으로 바꾼 것입니다. CFG 1~2 를 권장합니다. churn 은 원본의 `max`
  대신 k-diffusion 의 `min` 이라 기본 설정(sigma churn 0)에서는 다시 잡음을 넣지 않습니다(원본과 다른 점). Anima(5차원
  latent)·flow 모델·인페인트 마스크·Anima 레퍼런스 latent 를 지원합니다. 보조 스텝은 Forge 의 스텝 카운터를 늘리지 않아
  프롬프트 편집(`[a:b:N]`)·Skip Early CFG·리파이너 스텝 전환이 밀리지 않습니다(원본은 보조 스텝마다 한 칸씩 밀림).
- **보조 스텝을 건너뛰는 경우**: Forge 의 **Spectrum Integrated** 가 켜져 있거나(그 예측기가 해상도 변화를 따라가지 못해, 보조
  스텝을 그대로 돌리면 기본 설정에서 Euler Dy 가 오류로 멈춤) Wan 2.2 I2V(`concat_latent`)·PiD(`lq_latent`)·`extra_concat_condition`
  처럼 전체 해상도 입력을 쓰는 요청에서는 보조 스텝 없이 일반 CFG++ 스텝(= Euler CFG++)으로 돌고, 이유를 콘솔에 한 번·infotext
  `Extra Samplers status`(예: `dy sub-steps skipped (Spectrum)`)에 남깁니다.
- **가이던스와 함께 쓸 때**: Dy/SMEA 보조 평가는 HiFlow 기록·정렬, Momentum/HiGS 이력, SMC 의 이전 오차, APG 모멘텀, RDC
  이동 평균을 바꾸지도 지우지도 않습니다(표시가 없으면 해상도가 바뀌는 평가에서 HiFlow 의 1차 기록과 SMC·APG·RDC 상태가 지워짐).
  PAG·CFG base·DCW·TSR 은 보조 평가에도 걸립니다.
- **Anima SPEED·Colorcraft 와 함께**: SPEED 가 샘플러를 해상도 구간마다 따로 불러도 보조 스텝은 SPEED 가 넘기는 구간 시작 스텝
  (`sam_extra_step_offset`)으로 패스 안 스텝 번호대로 돌아, 구간마다 다시 돌지 않고 SMEA 의 ×1.25 보조 스텝이 SPEED 의 확장 바로
  뒤에 돌지도 않습니다. Spectrum Integrated 아래에서는 SPEED 와 보조 스텝이 둘 다 물러나 각자 status 에 남깁니다. Colorcraft 는 보조
  평가를 보정하지 않고 넘깁니다(스텝마다 한 번 보정).
- **한계**: SMEA Dy 의 ×1.25 보조 스텝은 Hires 패스에서도 돌아 VRAM 을 더 씁니다. 이 확장이 남긴 `ER SDE eta` 는 aoleg 확장이
  읽지 않아 그쪽에서는 η 가 되살아나지 않습니다. DPM++ 4M SDE 의 momentum 은 옮기지 않았습니다.
- **검증**: 새 테스트 123개. ComfyUI·Clybius·Koishi-Star 원본을 그대로 담은 대조 테스트: ER SDE 는 eps·flow 모두 비트 일치,
  4M SDE 는 Clybius 와 수치 일치(flow 는 eps 등가 좌표에서), Euler Dy 는 CFG++ 를 끄고 `max` 규칙이면 원본과 비트 일치. Forge 의
  실제 `KDiffusionSampler`·`CFGDenoiser`·`prompt_parser` 로 배치 = 단일 시드, 스텝 카운터, Spectrum·Wan I2V·PiD 가드를
  확인했습니다. GPU 는 Anima 3.8B 에서만: ER SDE (ODE)·Dy·SMEA 는 정상(Dy·SMEA ≈ Forge Euler CFG++), ER SDE (Reverse-time) 은
  채도 높은 다른 그림체, DPM++ 4M SDE 는 단순한 얼굴·가장자리 선 — Forge 자체 DPM++ 2M/3M SDE 도 Anima 에서 회색조·가장자리
  선이라 SDE 잡음 주입과 Anima 의 궁합으로 봅니다. SDXL 은 확인하지 않았습니다.

### 진행 막대 — sd-webui-smooth-progress 편입 (새 기능, 토글 — 기본 꺼짐)

- **부드러운 진행 막대**: Settings → **SAM Extra Progress Bar** 를 켜면 txt2img·img2img 갤러리 위에 부드럽게 채워지는
  막대가 생기고 Forge 기본 막대는 숨겨집니다. diamfang/sd-webui-smooth-progress(MIT, `7fe5810`)의 스텝별 ETA·부드러움
  세 방식·글자 형식 여덟 가지·끝난 뒤/중단 표시를 옮겼습니다. 결과 이미지와 생성에는 영향이 없습니다.
- **다시 만든 부분**: 상류는 'generate' 처럼 보이는 클릭으로 시작을 짐작하고 인증 없는 경로를 100ms 마다 계속
  물었습니다. 이제 Forge `requestProgress` 를 감싸 그 탭이 시작한 작업만, 작업 중에만(페이지가 보일 때)
  `GET /sam-extra/progress` 를 묻습니다. 이 경로는 다른 sam-extra 경로와 같은 인증·헤더(`X-SAM3-Notebook: 1`)·
  `no-store` 입니다.
- **작업 전체 진행률**: 배치·Hires 패스를 합친 Forge 기본 막대와 같은 진행률과, 패스 종류별 스텝 평균·끝난 패스
  시간으로 잡은 작업 전체 ETA. 대기 중에는 Forge 의 대기열 글자.
- **상류 버그 수정**: 중단 표시 네 가지 중 세 가지만 고를 수 있던 것, 생성이 아닌 클릭 뒤 `0/0` 에 멈추던 것,
  높이 미제한(이제 10~50px), 보호 없는 localStorage(설정은 Forge 옵션으로), ETA 를 모를 때 Smooth > Accurate 가
  99% 로 내달리던 것, 스텝 0 을 못 본 새 패스의 준비 시간이 스텝 시간에 섞이던 것.
- **디자인 규칙**: 테마 색, 그라데이션·빛 번짐 없음, 움직이는 동안 DOM 노드를 넣고 빼지 않음(텍스트 노드
  `.data` 만), prefers-reduced-motion 존중. sd-webui-smooth-progress 가 함께 설치돼 있으면 이 막대는 물러납니다.
- **한계**: 오류(OOM 등)로 끝난 작업도 완료(100%)처럼 끝납니다. ADetailer·SAM3 가 작업 도중 패스를 더하는 동안은 99.2% 에
  머물 수 있고, ETA 는 첫 스텝의 준비 시간이 섞여 몇 스텝 뒤에 자리를 잡습니다. Extras·모델 병합·확장 설치는 Forge 기본 막대
  그대로입니다(상류와 같음).
- **검증**: Python 61개(상류 `_step_eta`·경로 원본 대조, Forge `progressapi` 대조 포함), JS 37개(jsdom, Forge
  진짜 `progressbar.js` 와 함께 도는 1개 포함). 실제 Forge 브라우저에서의 확인은 아직입니다(Chromium 정적 하네스로만 확인).

### MCP 서버 — forgeneo-mcp 편입 (새 기능, 기본은 생성만 허용)

- **MCP 서버**: 확장 안의 `mcp_server/` 로 같은 PC 의 Claude Code 같은 MCP 클라이언트가 이 Forge 로 이미지를 만듭니다.
  [eduardoabreu81/forgeneo-mcp](https://github.com/eduardoabreu81/forgeneo-mcp)(`a103dc5`, MIT)를 편입했고, MCP SDK 와 Forge 의
  pydantic 고정이 맞지 않아 uv 의 별도 환경에서 돕니다(첫 실행만 네트워크). Forge 는 `--api` 로 켜져 있어야 합니다. 등록:
  `claude mcp add --scope user sam-extra -- uv run --project <확장>\mcp_server sam-extra-mcp`. 도구는 10개입니다(`capabilities`·
  `model_profile`·`prompt_dialect`·`loras`·`lora_info`·`models`·`module_check`·`module_download`·`generate`·`progress`).
- **권한**: Settings → **SAM Extra MCP** 의 네 스위치(`sam3_mcp_allow_generate` 켬, `sam3_mcp_allow_model_switch`·
  `sam3_mcp_allow_interrupt`·`sam3_mcp_allow_download` 끔)를 서버가 호출마다 Forge 의 `config.json` 에서 다시 읽어 지킵니다 —
  에이전트의 `confirm=True` 로도 넘을 수 없습니다. Forge 쪽에는 설정 네 개만 더해지며 생성 결과는 그대로입니다.
- **상류 대비 고친 것**: FORGE_PATH_MAP 없이도 같은 PC 의 출력 폴더를 찾고(상류는 렌더가 끝난 뒤 `ok:false`), 그래도 못 찾은
  이미지는 응답에서 저장합니다 · 같은 시각 다른 생성의 파일을 돌려주지 않습니다(infotext·시드·시간 대조) · 생성 기록과 LoRA 색인을
  다시 읽습니다(상류는 한 번만, 강제 재구축 때는 두 번 셈) · Anima 3.8B·Qwen3.5 모듈을 남은 모듈로 오인하지 않습니다 ·
  `/sdapi/v1/cmd-flags` 를 부르지 않고 리디렉션도 따라가지 않습니다 · 내려받기는 크기·SHA256 을 확인하고 덮어쓰지 않는 이동으로
  자리에 놓습니다 · 체크포인트 전환의 프리셋은 인스턴스에 있는 것만 받습니다.
- **더한 안전장치**: init 이미지는 이 PC 파일만, 한 번 생성의 화소 상한(`SAM_EXTRA_MCP_MAX_PIXELS`, 기본 16.8 MP), LoRA·체크포인트
  제작자 글은 `untrusted_*` 로 표시.
- **한계**: 권한은 MCP 서버가 지키는 것이지 샌드박스가 아닙니다 — 셸·파일 쓰기 도구도 가진 에이전트는 `config.json` 을 고치거나
  Forge API 를 직접 부를 수 있습니다(Forge 2.29.2 의 `POST /sdapi/v1/options` 는 `restrict_api` 를 지키지 않음). 같은 PC 전용이라
  `FORGE_URL` 이 다른 컴퓨터를 가리키면 모든 권한이 꺼집니다. 동영상(Wan) 결과는 모으지 않고, `uv.lock` 은 함께 배포하지 않습니다.
- **검증**: 새 Python 테스트 312개(상류 pytest 130개 중 128개를 unittest 로 옮긴 것 포함). MCP SDK 가 있어야 도는 등록 테스트 5개는
  Forge venv 에서 건너뛰고, mcp 2.2.0 · Python 3.11 환경에서는 312개 모두 통과했습니다. SDK 의 `ClientSession` 으로 stdio 에 붙여 도구
  10개 목록, 정책 읽기, 생성·중단 거절을 확인했습니다. 실제 Forge 생성과 Claude Code 등록으로는 아직 확인하지 않았습니다.

### 구도 · 카메라 — 프롬프트로 시점 잡기 (새 기능, 토글 — 기본 켬, 결과 같음)

- **구도 · 카메라 칸**: txt2img·img2img 스타일 줄 아래(기본 배치에서는 Generate 옆 열, txt2img 는 TIPO 칸 다음)에 접힌 칸이
  생깁니다. 방향(−180~180°)·높이(−75~75°)·거리·크롭(0~100%)·기울기(−30~30°)·화면 내 인물 위치(−100~100%)를 프리셋 다섯 개·끌 수
  있는 궤도 그림(방향키 5°, Shift 15°, Home 정면)·슬라이더로 정하면 `facing viewer`·`from above`·`cowboy shot`·`dutch angle`·
  `centered composition` 같은 태그·구도 문구를 미리 보여 주고, **메인 태그에 추가** 를 눌렀을 때만 메인 프롬프트 끝에 없는 태그만
  붙입니다. 3D 카메라가 아니라 프롬프트 유도입니다. 사용자 앱 UR_IV 의 `compositionPrompt.ts`·`CompositionControl.vue`(앱
  `d2fcc70`)를 옮겼고, 화면 글자와 태그는 앱과 같습니다.
- **적어 둔 프롬프트는 그대로**: 붙일 부분만 끝에 넣습니다(`document.execCommand('insertText')` — 브라우저 Ctrl+Z 한 번에
  되돌아감, 안 되면 값을 씀). 브라우저가 내는 타이핑 모양의 input 이벤트는 막고 Forge `updateInput` 으로 Gradio 에 알려, 태그
  자동완성(tagcomplete)이 추천 목록을 띄우지 않습니다 — 목록이 떠 있으면 Enter/Tab 이 방금 붙인 단어를 바꿀 수 있습니다. 같은
  태그는 대소문자·밑줄·가중치를 가리지 않고 알아보고, 반대 태그가 이미 있으면 경고만 하고 지우지 않습니다(버튼: 메인 태그에 추가 /
  이미 포함된 구도 / 기존 구도 유지하고 추가). 미리보기·경고는 입력할 때마다 바뀝니다.
- **설정·저장**: Settings → **SAM Extra Appearance** → `sam3_composition_panel`(기본 켬, Reload UI 또는 재시작 뒤 적용). 조작값은
  탭마다 이 브라우저의 localStorage 에 두고, 깨진 값은 앱과 같은 규칙으로 고칩니다. 누르기 전에는 프롬프트가 바뀌지 않고 생성·
  infotext 는 그대로입니다.
- **검증**: 앱 원본 파일을 `tests/_origin_composition_prompt/` 에 SHA-256 고정으로 두고 Node 가 그대로 불러와 이식본과 대조합니다 —
  프리셋, 조절값 전 범위와 그 바깥(.5·.49 반올림 경계), 태그 문턱값 안팎 곱 격자 약 26만 상태, NaN·Infinity·문자열 같은 깨진 값,
  가중치·이스케이프·스케줄·와일드카드·LoRA·끝 쉼표·빈 값 프롬프트 말뭉치와 무작위 문자열 4000개(모두 같음). JS 64개(로직 26 · 원본
  대조 11 · jsdom 동작 27), Python 27개(자리·설정·Forge 기본/Compact 배치). 타입 지우기가 없는 Node 20(그때 CI)에서는 원본 대조
  9개를 건너뛰었습니다 — v0.32.0 에서 CI 를 Node 24 로 올려 고쳤습니다. 실제 Forge(헤드리스 Chrome, 진짜 마우스 클릭·Ctrl+Z)에서 txt2img·img2img 모두 확인: 붙이면 Gradio 값도 바뀌고, Ctrl+Z
  한 번에 화면·Gradio 값이 함께 돌아오며, 자동완성 목록은 뜨지 않습니다(직접 타이핑하면 뜸).

### 설정 · 라이선스 · 검증

- **새 설정 섹션**: **SAM Extra Colorcraft**(`sam3_colorcraft`)·**SAM Extra SPEED**(`sam3_speed`)·**SAM Extra Progress Bar**
  (`sam3_progress`)·**SAM Extra MCP**(`sam3_mcp`). 구도·카메라는 기존 **SAM Extra Appearance** 에 `sam3_composition_panel`(기본 켬)을
  더합니다. Extra Schedulers·Extra Samplers 는 설정 키가 없습니다(Karras Dynamic 은 Forge 의 rho, 새 샘플러는 Forge 의 Eta·Sigma
  churn/tmin/tmax/noise 를 씀).
- **라이선스**: 편입한 상류 — Colorcraft(muerrilla·aoleg 포크, MIT), SPEED(howardhx·aoleg·sorryhyun, MIT), Smooth Progress
  (diamfang, MIT), forgeneo-mcp(Eduardo Abreu, MIT), ComfyUI `get_sigmas_laplace`·`SamplerER_SDE` 잡음 척도(GPL-3.0), Clybius
  DPM++ 4M SDE(BSD-3-Clause), Koishi-Star Euler (SMEA) Dy(Apache-2.0) — 의 커밋과 바꾼 곳을 `THIRD_PARTY_NOTICES.md` 에 적고, MIT
  고지 적용 목록에 저작권자를 더했으며 BSD 3-Clause·Apache 2.0 전문을 넣었습니다. Extra Schedulers 의 나머지 스케줄러는 식
  재구현이고, 두 기능이 이름을 맞춘 aoleg/Neo_ExtraSchedulers(라이선스 없음)의 코드는 쓰지 않았습니다(README 만 참고).
  `mcp_server/pyproject.toml` 의 라이선스는 `GPL-3.0-only AND MIT` 입니다. Colorcraft 색 벡터(`sam3ext/colorcraft/data/*.safetensors`,
  1.4~6.4 KB)는 `.gitignore` 예외로 저장소에 함께 들어갑니다.
- **검증**: Python 2841개 통과(skip 22 — 그중 5개는 MCP SDK 가 있어야 도는 등록 테스트, CPU). 새 테스트는 Colorcraft 139 ·
  Anima SPEED 132 · Extra Schedulers 146 · Extra Samplers 123 · 진행 막대 61 · MCP 312 · 구도·카메라 27 · 기능 조합 44 개입니다.
  JS 177개 통과(진행 막대 37개·Colorcraft 10개·구도·카메라 64개를 더함). 실제 Forge 에서는 위에 적은 Anima 3.8B GPU 실측과
  구도·카메라 칸의 브라우저 동작을 확인했고, 진행 막대의 브라우저 표시(설정을 켜야 함)와 실제 MCP 클라이언트 연결은 아직입니다.
- **검증 — 기능 조합**: 새 통합 테스트 44개(`tests/test_integration_*.py`, CPU 하네스 `tests/_integration_support.py`)가
  SPEED+Colorcraft, Dy+Colorcraft, SPEED+Dy, Forge 의 실제 `get_sigmas` 로 만든 스케줄러 스케줄을 SPEED 로 돌리기, post-CFG 순서,
  다섯 샘플러 모두 + 가이던스 전체, 스크립트 22개 전부를 Forge 순서로 불러오기를 확인합니다.

## v0.30.1 — 진단 로그의 Momentum·HiGS 적용 횟수가 생성마다 쌓이던 문제

- **`[VERIFY] detail`의 MG·HiGS 횟수 수정**: `Log Guidance verification summary`를 켜면 남는 `MG=APPLIED(n evals)`와
  `HiGS=APPLIED(n evals)`가 한 Forge 실행 안에서 생성마다 더해졌습니다. 같은 요청을 보내도 9, 18, 27 …로 늘었습니다.
  패스가 시작될 때 `GuidanceRuntime.reset_pass()`가 Momentum·HiGS 이력(EMA)만 비우고 횟수는 그대로 두었기 때문입니다.
  `HistoryState.reset()`은 생성 중간(ADG가 uncond를 건너뛴 스텝)에도 쓰이므로 횟수를 남기는 것이 맞습니다. 그래서 이제
  패스마다 새 이력으로 시작합니다. TSR처럼 패스마다 0부터 세고, Hires.fix를 켜면 hires 패스의 수가 남습니다.
- **이미지는 그대로입니다**: 진단 출력만 바뀝니다. Forge 2.29.2 · Anima 3.8B에서 Momentum과 HiGS 요청을 각각 두 번씩
  보냈습니다. 두 번 모두 MG 9회 · HiGS 24회로 같았고, 결과 md5는 고치기 전과 같았습니다.
- **검증**: Python 1857개 통과(skip 17). 같은 생성을 두 번 돌려 두 번째 횟수가 첫 번째와 같은지 보는 회귀 테스트를
  더했습니다. 고치기 전 코드에서는 이 테스트가 4회 · 2회로 실패합니다.

## v0.30.0 — Anima 3.8B + 캐릭터 레퍼런스 IP-Adapter + 28/40/52블록 LoRA·DoRA + 디테일 가이던스

v0.21.2 이후 쌓인 큰 업데이트입니다. Anima 3.8B(Qwen3.5 / Semantic Connector v2) 런타임을 들여왔고, 캐릭터 레퍼런스 패널(이어붙이기 · IP-Adapter
방식)이 생겼습니다. Anima LoRA 는 Base 1.0(28)·2.9B(40)·3.8B(52) 사이를 자동으로 옮기고, DoRA 합치는 방식을 고를 수 있습니다. 그 밖에 TIPO 프롬프트
확장(🪄)·SAM3 빠른 버튼(🎯)·txt2img 섹션 정리·VAE 격자 제거(Anima VAE DeGrid, NAFNet)가 추가됐고, SAM3·가이던스·3.8B 생성 시간을 줄였습니다 — 대부분은 결과가 픽셀 단위로 같고, PAG/SEG 의 두 가지 최적화만 잔
디테일이 달라집니다(설정으로 끌 수 있음). 가이던스(Detail Daemon·Safe PAG·Skimmed CFG·DCW(+a)·DAVE·CNS)와 Tile-Repair 는 가져온 원본 ComfyUI
노드·sd-scripts 와 같은 값·범위·적용 구간으로 맞춰 같은 설정에서도 결과가 달라지고, Tile-Repair HTTP API 와 Notebook 메모장이 생겼습니다. 가이던스에는
디테일 단계(S²-Guidance·Adaptive SMC·TSR·Momentum Guidance·HiGS·HiFlow)와 Anima Optimal Scale 이 더해졌고, 모두 기본으로 꺼져 있습니다. 이 확장의
라이선스는 이제 **GPL-3.0-only** 입니다(편입한 NegPiP 파일만 **AGPL-3.0-or-later** — **설치 · 라이선스 · 개발** 절).

괄호 표시: (결과 변화) = v0.21.2 와 같은 설정·시드에서 이미지가 달라짐, (결과 같음) = 이미지 동일, (새 기능) = v0.21.2 에 없던 기능이라 비교 대상 없음, 토글 = 끌 수
있는 설정이 있음(뒤에 기본값), 토글 없음 = 끌 수 없음.

**업그레이드할 때**

- Forge 를 **재시작**하세요. Reload UI 로는 런타임 모듈이 바뀌지 않습니다.
- v0.21.2 와 같은 설정·시드에서 결과가 바뀔 수 있는 곳(자세한 내용은 각 절의 해당 항목):
  - PAG/SEG/SLG 앞쪽 블록 중복 계산 건너뛰기, SEG blur 1D 두 번 — 구도는 같고 잔 디테일만 다릅니다. 끄려면 Settings → **SAM Extra Guidance** 의
    두 설정을 끄세요. infotext 키 `Anima PAG prefix dedup`·`Anima SEG separable blur` 가 없는 예전 infotext 를 붙여 넣으면 현재
    설정(켬)으로 렌더되므로, 예전 이미지를 그대로 재현하려면 둘 다 끄세요.
  - 가이던스를 가져온 원본 ComfyUI 노드와 같게 맞췄습니다(토글 없음). Detail Daemon·Safe PAG·Skimmed CFG·DCW(+a)·DAVE·CNS 의 기본값·범위·적용
    구간(σ 기준)이 원본과 같아져, v0.21.2 는 물론 이전 v0.30 개발 빌드와도 같은 설정에서 이미지가 달라질 수 있습니다. Detail Daemon 은 강도가 ×0.1 이라
    v0.21.2 와 같은 강도는 amount × 10 이고, **Hires Pass** 를 켜지 않으면 hires 패스에는 걸리지 않습니다(v0.21.2 는 두 패스 모두). 기본값은 DCW λ
    0.10/0.02 → 0.05/0.01, CWM α 0.30/0.15 → 0, CNS gamma scale 3.0 → 2.0 이고, RDC 는 tau 기본 0(끔)입니다. API 에서 생략한
    위치 인자도 새 기본값을 씁니다(Custom SMC k 인자 27 은 0.20 → 0.10).
  - SMC/APG/CWM 은 CFG 1 에서 건너뛰고, Safe PAG 의 APG + rescale 자동 끄기도 CFG 1 에서는 적용하지 않습니다(토글 없음). Safe PAG 확장 배치가 OOM
    나면 그 생성의 남은 스텝·배치에서 PAG/SEG/SLG 를 끕니다.
  - SAM3: 인페인트 패스가 의도한 시드를 쓰고, SAM3 를 쓴 뒤의 다른 생성에서 fp32 행렬곱이 더 이상 TF32 로 돌지 않습니다(버그 수정, 토글 없음). API/XYZ 에서 생략한
    키의 기본값이 UI 와 같아졌습니다(`sam3_inpainting_fill`→ `original`, `sam3_unload_after`→True). 범위 밖 설정값은 SAM3 를 끄는 대신
    범위로 맞춰 실행합니다. SAM3 In-flight 인페인트의 내부 패스에서 ADetailer 가 더 돌지 않으므로(바깥 생성의 ADetailer 는 한 번 그대로) ADetailer 와 함께
    쓰던 생성은 결과가 달라질 수 있습니다.
  - 일부 블록만 담은(sparse) Anima LoRA 는 판정 블록 수가 현재 모델과 다르면 순정 Forge 처럼 추측 변환하지 않고 건너뜁니다. 순정처럼 하려면 Settings → **SAM
    Extra LoRA** → `sam3_anima_sparse_lora_forge_guess` 를 켜세요.
  - 3.8B v2 번들 체크포인트는 Qwen3.5 커넥터가 자동으로 켜집니다. 끄려면 `Anima 3.8B (Qwen3.5 / v2)` 아코디언의 Bypass.
  - Tile-Repair 는 sd-scripts 원본처럼 원본 비율을 지키는 Short Side 슬라이더·디코드(uint8 버림)·빈 네거티브를 쓰고, SAM3 ControlNet 에 Tile &
    Repair LLLite 를 고르면 preprocessor 가 늘 `None` 이 됩니다(토글 없음).
- NegPiP 가 이 확장에 들어왔습니다. 따로 설치한 `extensions/sd-forge-negpip` 는 지우고(또는 끄고) 재시작하세요 — 남아 있으면 내장 NegPiP 가
  쉬고 경고를 한 번 남깁니다(두 번 적용 방지). 예전 판(`b3673ce`)과 음수 항을 찾는 규칙이 조금 다르고, 새 Forge 에서 emphasis 가
  `None`/`Ignore` 면 Anima NegPiP 는 켜지지 않습니다(**NegPiP 내장** 절).
- 메모리: 3.8B 는 생성 사이 VRAM 에 최대 약 6~8 GB 를 남기고(`sam3_anima38_keep_resident`), SAM3 'Unload after' 는 모델을 RAM 에 약
  3.4 GB 로 보관합니다(`sam3_unload_keep_in_ram`). 3.8B 샘플링 중에는 커넥터 fp32 상주 (`sam3_anima38_connector_fp32`, VRAM 약
  +1.5 GB, 여유가 있을 때만)와 run 캐시(`sam3_anima38_connector_run_cache`, 최대 512 MB)도 씁니다. 넷 다 기본 켬이고 끌 수 있습니다.
- 업데이트 뒤 첫 시작 때 `ui-config.json` 을 한 번 옮깁니다(Guidance 아코디언 슬라이더, Skimmed CFG `Flip at`, Tile-Repair 네거티브). Forge 는 저장된 슬라이더 값·범위를
  라벨로 다시 적용해서, 라벨이 그대로인 슬라이더에 예전 기본값·범위가 되살아나기 때문입니다. 예전 기본값 그대로인 값만 원본 기본값으로 바꾸고(DCW λ low 0.10 → 0.05·high
  0.02 → 0.01, CWM α low 0.30·high 0.15 → 0, 예전 Enable RDC 가 꺼진 채 저장된 RDC tau 0.15 → 0), 예전 범위 저장값(PAG Attn
  Scale 최대 15, DCW λ high ±0.5, CWM α 최대 1, CNS strength step 0.01·gamma power 최소 0.05, Skimmed CFG `Flip at` step 0.05)은
  지웁니다. Tile-Repair 패널은 예전 Width·Height 슬라이더 키가 남은 설치만 그 키를 지우고, 네거티브가 예전 기본값 `blurry, low quality` 그대로면
  sd-scripts 기본값인 빈 칸으로 바꿉니다. 사용자가 바꾼 값은 남깁니다 —
  새 범위 밖이면 범위로 맞추고(DCW λ high ±0.3, CNS gamma power 0.1 이상), 예전 라벨로 저장된 CNS gamma scale 은 새 라벨로 옮기며, Enable RDC
  를 끈 채 바꿔 둔 RDC tau 도 남으므로 이제 Enable DCW 를 켜면 RDC 가 함께 돕니다(끄려면 tau 0). 바꾸기 전 파일은 같은 폴더에
  `ui-config.json.bak-anima-guidance-<날짜-시각>` 으로 복사하고, 새 파일은 임시 파일을 거쳐 한 번에 바꿔 씁니다. 바꾼 항목은 콘솔 `[AnimaSafePAG]
  ui-config.json migrated …` 아래 줄마다 나오고, 바꿀 것이 없으면 파일을 건드리지 않습니다. Detail Daemon Amount 는 라벨이 바뀌어 예전
  저장값이 적용되지 않으므로 대상이 아닙니다.
- Forge 를 `--api` 또는 `--nowebui` 와 `--api-auth` 로 띄우면 이 확장의 Tile &
  Repair(`/sam-extra/tile-repair…`)·Notebook·메모(`/sam3-notebook…`)·LoRA Manager(`/sam3-lora/*`) 경로에도 `/sdapi` 와
  같은 HTTP Basic 인증이 걸립니다. 이 경로를 부르는 스크립트·앱은 자격 증명을 보내야 하고(없거나 틀리면 401), `--gradio-auth` 로그인도 켜져
  있으면 둘 다 필요합니다.
- `/sam3-lora/config`·`/sam3-lora/spawn` 을 직접 부르는 외부 도구는 이제 다른 확장 경로처럼 헤더 `X-SAM3-Notebook: 1` 을 보내야
  하고(없으면 403), `--gradio-auth` 가 켜져 있으면 로그인 쿠키도 필요합니다(없으면 401). 페이지의 Manage 탭은 Gradio 버튼 브리지를 써서
  그대로 동작합니다.

### Anima 3.8B (Qwen3.5 / Semantic Connector v2)

- **3.8B 런타임 편입 (결과 변화, v2 번들은 자동 켬)**:
  [GumGum10/forge-anima-3.8B](https://github.com/GumGum10/forge-anima-3.8B) (MIT) 런타임을 `sam3ext/anima38/` 로 들여와
  `Anima 3.8B (Qwen3.5 / v2)` 아코디언 한 스크립트로 붙였습니다. v2 번들(safetensors metadata 로 판별)은 생성 때 자동으로 켜지고, **Bypass** 를
  켜면 순정 Anima(0.6B) 로 생성합니다. v1 은 아코디언에서 어댑터·강도·부정 프롬프트 사용을 고릅니다. `qwen35_4b` 가 없으면 설치 전에 확인해 경고하고 순정 Anima 로
  진행합니다. API 는 위치 인자와 SAM3 식 dict 둘 다 받습니다. Qwen3.5 토크나이저는 `assets/qwen35_tokenizer/` 에
  동봉했습니다(`THIRD_PARTY_NOTICES.md`). 아코디언에 상태 확인 버튼이 있습니다.
- **infotext·붙여 넣기**: `Anima38`(v2 bundle / v1 adapter / bypass / off: …)·`Anima38 encoder`·`Anima38 negative` 를
  남기고, PNG Info 붙여 넣기로 Bypass·부정·v1 설정을 되살립니다. infotext 키는 `Anima38 …` 형식입니다(Forge 가 `.` 이 든 키를 읽지 못하므로).
- **다른 확장과의 공존**: v2 조건을 네이티브 list 형식으로 넘기고 run id 는 텐서 마커로 전달해 NegPiP 등 `get_learned_conditioning` 래퍼와 충돌하지
  않습니다. 조건·forward 패치는 플래그로 켜고 끄는 멱등 패치라 다른 확장이 순서를 바꿔 되돌려도 낡은 래퍼가 되살아나지 않고, 샘플링 중 예외로 남은 패치는 다음 생성 시작 때 먼저
  원복합니다. NegPiP 이 아래에 깔린 순서에서는 그 마스킹을 대신 적용합니다. 격리 Forge 두 대(정·역방향 로드 순서)에서 6 케이스 (v2+NegPiP / bypass+NegPiP /
  v2 / bypass / 반복 / hires)가 픽셀 단위로 같았습니다.
- **LoRA 와 커넥터**: LoRA 의 `llm_adapter` 가중치를 v2 경로에도 적용합니다 — 커넥터가 번들 원본으로 만든 자기 `llm_adapter` 사본(약 +0.3 GiB RAM)에
  LoRA 패치를 받고 샘플링 직전마다 그 패스의 LoRA 세트에 맞춥니다. LoRA 세트가 바뀌어 Forge 가 UNet 을 새로 복제해도 커넥터가 Forge 메모리 관리 안에서 돕니다.
- **VRAM 상주 (결과 같음, 토글, 기본 켬)**: 같은 프롬프트 줄이면(시드만 바꾸는 XYZ, batch count, 캐릭터 레퍼런스 후보, ADetailer) 0.6B TE 결과와
  Qwen3.5 인코딩을 줄 캐시에서 꺼내 다시 인코딩하지 않고, batch count 사이에는 설치를 유지합니다. 커넥터(약 1.6 GB)는 생성이 끝나도 VRAM 에 남고, TE(약 1.75
  GB)·Qwen3.5(약 4.45 GiB)는 다음 샘플링에 쓸 여유가 있을 때만 남깁니다 — 합쳐 최대 약 6~8 GB. Forge 가 자리가 필요하면 퇴출하지만 Forge 밖의 VRAM
  사용(SAM3 검출, 같은 GPU 의 학습)은 모릅니다. 학습과 같이 쓸 때는 Settings → **SAM Extra Anima 3.8B**(`sam3_anima38`) → "Anima 3.8B:
  TE·Qwen3.5·커넥터를 생성 사이 VRAM 에 남기기 (최대 약 6~8 GB)"(`sam3_anima38_keep_resident`, 기본 켬)를 끄세요. 끄면 TE·Qwen3.5 는 인코딩
  직후, 커넥터는 생성이 끝날 때 내립니다(3.8B 가 설치된 모델의 Forge TE 도 함께 — 줄 캐시로 올리지 않은 TE 까지 내리므로 다음 생성에서 TE 를 다시 올릴 수 있음. 3.8B 를
  설치하지 않은 생성은 영향 없음). 번들에 커넥터 전용 `llm_adapter` 사본이 없는 폴백에서는 결과를 지키려고 늘 내립니다.
- **메모리 정리**: VRAM 이 모자라 Qwen3.5 가 부분 로드돼도 돕니다(RMSNorm 가중치 장치 불일치 수정). `qwen35_4b` 를 VAE/Text Encoder 목록에 넣어 두어도
  Forge 가 체크포인트마다 4.8 GB 를 읽고 버리지 않아, XYZ 체크포인트 축에서 1.0/2.9B/3.8B 를 같은 모듈 목록으로 비교할 수 있습니다. 체크포인트를 바꾸면 이전 3.8B
  모델(약 8 GiB)을 RAM 에서 놓고, 설치 도중 모델이 다시 로드되면(Hires 체크포인트·Refiner 등) 남은 샘플링 패처를 생성 끝에 내립니다. GPU 측정: Qwen3.5 가 커넥터에
  넘기는 층의 최대 |값| 35.5 — fp16 텍스트 인코더에서도 넘치지 않습니다.
- **다른 패스·탭과 섞일 때의 수정**: SAM3 In-flight 인페인트 패스 뒤의 ADetailer 패스와 배치의 다음 이미지가 0.6B 조건으로 떨어지지 않고, 다른 탭 생성이 도중에 죽어도
  Bypass 생성이 v2 로 돌지 않습니다. 커넥터·Qwen3.5·v1 어댑터 가중치는 inference_mode 밖에서 만들어, 캐릭터 레퍼런스가 3.8B 에서 `Inference tensors
  do not track version counter` 로 죽지 않습니다.
- **새 Forge 텍스트 엔진 대응 (결과 같음, 토글 없음)**: Forge `21886f41`("Rewrite TextProcessingEngine")이 Anima 0.6B TE 엔진을
  `Qwen06Engine`(ComfyUI v0.36 `sd1_clip` 이식)으로 바꿔, 그 뒤 Forge 에서는 3.8B 생성이 전부 `'Qwen06Engine' object has no attribute
  'tokenize_line'` 으로 죽었습니다. 엔진에 옛 API 가 있으면 예전 경로를, 없으면 `Qwen06Engine.__call__` 의 한 줄 계산(emphasis `None` 은 가중치
  파싱 끔, `Ignore` 는 T5 가중치 1.0, qwen 가중치는 1.0 강제)을 그대로 따라 합니다 — 실제 Forge 엔진·토크나이저로 CPU 에서 네 emphasis 모두 같은 값을
  확인했습니다. NegPiP 는 이제 이 확장에 들어 있어(아래 **NegPiP 내장**) 옛·새 Forge 모두에서 v2 생성에도 적용됩니다(새 Forge 의 emphasis
  `None`/`Ignore` 는 엔진이 음수 가중치를 쓰지 않아 NegPiP 도 없음 — 아래) — 개발 빌드 한때처럼 새 Forge 에서 NegPiP 를 끄지 않아도 됩니다. v1/v2 경로는 엔진 `__call__` 을 건너뛰어 순정 엔진이 남기는 `Emphasis` 생성 정보가
  빠졌는데, 이제 엔진마다 같은 규칙으로 남깁니다(옛 엔진: 가중치 문법이 있으면 늘, 새 엔진: 방식이 `None`/`Ignore` 일 때만. 옛 엔진은 순정처럼
  방식을 설정에서 다시 읽어 파싱합니다).
- **Forge 2.29.2 의 엔진 속성 이름 (결과 같음, 토글 없음)**: Forge 2.29.2(`46365871`)가 Anima 텍스트 엔진을 `sd_model.text_processing_engine_anima`
  에서 Flux2(Klein)·Krea2·Qwen-Image·Z-Image 와 같은 `text_processing_engine_qwen` 으로 옮겼습니다. 옛 이름만 읽으면 그 Forge 에서 3.8B 설치가
  `Anima 3.8B requires a loaded Anima checkpoint.` 로 실패해, 콘솔에 `install failed … continuing with native Anima` 만 남기고 순정 0.6B 로
  조용히 물러납니다(Anima 의 NegPiP 는 아래 **NegPiP 내장**). 이제 두 이름을 다 찾되, 공용 이름에서는 T5 토크나이저를 가진 엔진(Anima 0.6B
  엔진 두 세대만 가짐)만 Anima 로 봐 Z-Image·Flux2 모델을 Anima 로 잘못 알아보지 않습니다. 설치된 Forge 소스에서 이 판별이 서는지(공용 이름에
  달리는 엔진 가운데 Anima 엔진만 `t5_tokenizer` 를 가짐)를 테스트가 지킵니다.

### NegPiP 내장 (sd-forge-negpip 편입)

- **NegPiP 를 확장에 넣었습니다 (대부분 결과 같음 — 아래 "결과 변화" 로 적은 경우만 달라짐, 토글 없음)**: [Haoming02/sd-forge-negpip](https://github.com/Haoming02/sd-forge-negpip)
  이 2026-09-30 보관(archive)돼, 마지막 커밋 `0585496` 을 `sam3ext/negpip/`·`scripts/negpip.py` 로 편입했습니다(AGPL-3.0-or-later, 고지는
  `THIRD_PARTY_NOTICES.md`, 전문은 `sam3ext/negpip/LICENSE`). 스크립트 제목 `NegPiP`·UI 없음·always-on 과 파일 이름 `negpip.py` 가 그대로라
  API·UR_IV·SAM3 안쪽 패스 복사와 ADetailer 패스(설정 `ad_script_names` 가 파일 이름 `negpip` 으로 고른다)가 예전처럼 동작합니다. 쓰는 법도 같습니다 — 긍정 프롬프트의 `(단어:-1.0)` 은 개념을 빼고, 부정 프롬프트의 음수 가중치는 개념을
  강제합니다(SD1·SDXL·Anima). 스크립트 순서도 예전 자리(`sd-dynamic-thresholding` 뒤)를 `metadata.ini` 로 지킵니다 — 상류는 Dynamic
  Thresholding 이 `p.sampler_name` 을 `…_dynthres<N>` 로 바꾼 뒤에 cond/uncond 절반을 골랐습니다. 내장은 Forge 가 넘기는 cond/uncond 표시를
  읽고(아래) 샘플러 이름은 표시가 없는 호출자의 대체 경로에서만 쓰지만, 그 경로를 위해 순서를 지킵니다.
- **SD1/SDXL: 두 번째 음수 항부터 아무 효과가 없던 문제 (결과 변화, 상류에도 있던 버그)**: `1girl, (hat:-1), (glasses:-0.8)` 처럼 음수 항이
  둘 이상이면 첫 항만 어텐션에 들어가고, 나머지 항은 프롬프트에서 지워진 채 어디에도 들어가지 않았습니다(빼지도 넣지도 않음). 콘솔의
  `NegPiP Enable (Positive: N)` 도 첫 항의 행 수만 보여 드러나지 않았습니다. 이제 한 프롬프트의 음수 항 전부가 들어가고 N 은 이은 행 수입니다.
  부정 프롬프트의 프롬프트 편집 `[(x:-1):5]` 가 0 스텝부터 걸리던 것도 긍정 쪽과 같은 스텝부터 걸리게 고쳤습니다.
- **SD1/SDXL: 배치 안에서 프롬프트가 다르면 항목 0 의 음수 항이 모두에게 가던 문제 (결과 변화, 상류에도 있던 버그)**: sd-dynamic-prompts
  와일드카드·`{a|b}` 에 Batch size 2 이상처럼 항목마다 프롬프트가 다를 때, 뒤 항목은 항목 0 의 음수 항을 받거나(항목 0 에 음수 항이 없으면)
  제 음수 항을 조용히 잃었습니다. 하이레스 프롬프트·부정 프롬프트도 같았습니다. 이제 항목마다 제 음수 항이 붙고, 음수 항이 없는 항목은
  NegPiP 가 없을 때와 같은 계산입니다.
- **SD1/SDXL: cond/uncond 를 Forge 의 실제 배치로 가림 (결과 변화, 상류에도 있던 버그)**: 상류는 cond/uncond 를 샘플러 이름·배치 크기·호출
  수·문맥 길이로 추정했습니다. Forge 가 VRAM 이 모자라(SDXL 고해상도·하이레스, GPU 를 다른 작업과 나눌 때) cond·uncond 를 따로 돌리거나,
  긍정·부정 프롬프트의 청크 수 비가 커서(예: 75 토큰 이하 긍정 + 300 토큰 넘는 부정) 따로 돌리거나, CFG 1(Lightning·DMD·Hyper)·Skip Early
  CFG·NGMS 로 cond 만 돌리면 음수 항이 8 스텝씩 켜졌다 꺼지거나 긍정 쪽 음수 항이 uncond 에 붙었습니다. DDIM/PLMS/UniPC 는 평소 배치에서도
  부정 쪽 음수 항이 cond 에 붙었습니다. 이제 Forge 가 어텐션까지 넘기는 cond/uncond 표시로 가려 어느 경우에나 cond 에는 긍정 쪽, uncond
  에는 부정 쪽 음수 항만 붙습니다(AND 프롬프트 포함). 긍정·부정 청크 수가 달라 Forge 가 짧은 쪽 문맥을 반복해 묶을 때 음수 항이 약해지던
  것도 원래 세기로 맞췄습니다. 표시를 넘기지 않는 호출자에서는 예전 추정을 쓰고 콘솔에 경고를 한 번 남깁니다.
- **음수 항 찾기 규칙 — 상류 수정 (결과 변화, SD1/SDXL)**: 대부분이 쓰던 예전 sd-forge-negpip(`b3673ce`)과 달리 상류 `75b81b4` 의 `NEG_PATTERN`
  을 씁니다. 예전 패턴은 괄호 안에 `:` 만 없으면 무엇이든 삼켜 `(smile), (aqua hair:-1)` 을 통째로 한 음수 항으로 잡았고(SD1/SDXL 에서 `(smile)` 까지
  프롬프트에서 빠져 음수 항에 들어감), 이제는 `(aqua hair:-1)` 만 잡습니다. 앞의 escape 괄호에서 시작해 삼키던 것도(`\(escaped\) text, (x:-1)` → 예전 `(escaped\) text, (x:-1)`) 이제
  `(x:-1)` 만 잡습니다. 상류 저자의 버그 수정이라 그대로 두었습니다. Anima 는 이 패턴을 켤지 판단에만 쓰고 마스크는 엔진 가중치로 만들어 영향이 없습니다.
- **새 Forge 의 SD1/SDXL — BOS 행을 뒤집지 않음 (결과 변화, 상류와 다름)**: Forge `21886f41` 부터 2.29.1 까지 `sd_engine.ClipEngine.tokenize` 가
  `add_special_tokens=False` 를 넘기지 않아 프롬프트 조각마다 BOS/EOS 가 붙습니다. 상류(두 판 모두)의 `_cond_dealer` 자르기 `cond[1:token_len+2]`
  는 그 엔진에서 `[단어…, EOS]` 대신 `[BOS, 단어…, EOS, EOS]` 행을 잡아 BOS(어텐션 싱크) 행의 V 까지 뒤집습니다. 내장은 엔진 자신에게 물어(빈 글자를
  토큰화하면 특수 토큰이 나오는가 — 판 번호가 아님) 그런 엔진이면 옛 엔진과 같은 `[단어…, EOS]` 행만 고릅니다. 그 판은 Forge 자체가 조건에 조각마다
  BOS/EOS 를 넣으므로 옛 Forge 와 이미지까지 같지는 않습니다. Forge 2.29.2(`0b1783c7`)는 `add_special_tokens=False` 를 되살려 행 배치가 옛 엔진과
  같아졌고, 내장은 그 판에서 옛 엔진과 같은 경로로 같은 행을 고릅니다(75토큰 이하·`BREAK` 없는 항은 상류 자르기와 같은 행). 실제 SD1.5 CLIP
  토크나이저와 세 판 엔진 코드(옛 엔진·`21886f41`·`0b1783c7`, CPU, 가짜 인코더)로 모두 같은 행을 고르는 것을 확인했습니다.
- **가중치 묶음 안의 `BREAK`·75토큰 넘는 음수 항 (결과 변화, 상류와 다름, SD1/SDXL)**: 상류는 `(글:-w)` 를 인코딩하면서 행 수는 맨 글(가중치 1)로
  셉니다. Forge 파서는 가중치가 1 이 아닌 묶음 안의 `BREAK` 를 청크 구분이 아닌 글자 `break` 로 남기므로 `(red eyes BREAK blue hair:-1.5)`·
  `(x BREAK y:-0.5)` 같은 항은 두 글의 청크 배치가 달랐습니다 — 새 Forge 에서는 `IndexError` 가 콘솔에 찍히고 NegPiP 가 조용히 꺼져 음수 항이
  프롬프트에서 그냥 사라진 채 생성됐고, 옛 Forge·상류에서는 채움 EOS 와 다음 청크 BOS 76~77 행을 뒤집었습니다(`-1` 이면 뒤 단어를 잃음).
  이제 인코딩한 바로 그 글자로 행을 세고, 옛 엔진도 새 엔진과 같은 행 규칙(엔진의 시작·끝 토큰을 건너뛴 단어 행 + 뒤 EOS)을 써 두 엔진이 같은
  행을 고릅니다 — 75토큰 이하·`BREAK` 없는 항은 예전(상류 자르기)과 한 행도 다르지 않고(Forge 설정 Emphasis 가 `None` 이면 예외:
  괄호·가중치가 글자로 인코딩되므로 이제 그 글자 행 전부를 잡습니다 — 상류는 맨 글 길이만큼 잘라 `(` 같은 앞 글자 행을 잡았습니다), 75토큰 넘는 항은 청크 경계의 채움·BOS 행 대신 단어
  행만 뒤집습니다. 만일 토큰화한 행 수가 조건과 다르면 틀린 행을 뒤집는 대신 그 생성에서 NegPiP 가 물러나고(`NegPiP Disabled (condition rows: …)`)
  지웠던 음수 항을 프롬프트에 되돌립니다.
- **Anima: 긴 프롬프트의 프롬프트 편집이 NegPiP 와 함께 죽던 문제 (상류에도 있던 버그)**: Anima 는 줄마다 `max(512, T5 토큰 수)` 행이라
  `[짧은:아주 긴:0.5]`·`[a|b]` 줄이 512 를 넘어 길이가 다르면, 음수 가중치가 있을 때 NegPiP 가 줄들을 한 텐서로 쌓다가
  `stack expects each tensor to be equal size` 로 생성이 조건 단계에서 죽었습니다(순정 Forge 는 됨). 이제 줄마다 조건을 따로 돌려주고 스텝마다
  줄 고르기·배치 길이 맞추기는 Forge 가 순정과 같게 합니다 — 512 행 이하 줄은 결과가 예전과 같습니다. 3.8B 런타임이 NegPiP 마스킹을 대신할 때도
  같습니다.
- **옛·새 Forge 의 Anima 마스크 (결과 같음)**: Anima 마스크가 Forge 의 옛 텍스트 엔진(`AnimaTextProcessingEngine`)과 새 엔진(`Qwen06Engine`, Forge
  `21886f41`)을 둘 다 압니다. 상류 마지막 판은 새 엔진만, 그 전 판은 옛 엔진만 알았습니다. 실제 Forge 엔진·토크나이저(CPU)로 두 세대 모두 마스크
  행이 엔진이 조건에 곱한 가중치와 한 칸도 어긋나지 않음을 확인했습니다.
- **Forge 2.29.2 의 Anima — NegPiP 가 빠지던 문제 (결과 같음)**: 2.29.2 가 Anima 텍스트 엔진을 `text_processing_engine_qwen` 으로 옮겨(위
  **Anima 3.8B**), 옛 이름을 바로 읽던 Anima 경로(상류 그대로)가 생성마다 `Error running process_batch: …negpip.py`(속성 오류)를 남기고 NegPiP
  없이 생성됐습니다. 이제 3.8B 와 같은 헬퍼로 두 이름을 다 찾습니다(emphasis 판단·조건 훅). SD1/SDXL 은 이 속성을 쓰지 않아 영향이 없었습니다.
- **새 Forge 의 emphasis `None`·`Ignore` 에서는 Anima NegPiP 가 켜지지 않습니다 (결과 변화)**: 새 엔진은 `None` 이면 괄호·`:-1` 을 글자로
  토큰화하고, `Ignore` 면 괄호는 먹되 가중치를 1.0 으로 둡니다 — 엔진이 음수 가중치를 적용하지 않습니다. 상류 `0585496` 은 여기서도 마스크를
  만들어 `None` 이면 행이 어긋나고 `Ignore` 면 K 만 뒤집혀 뜻이 반대가 됐습니다. 이제 콘솔에 `NegPiP Disabled (Emphasis: None)` 을 남기고
  건너뜁니다(Forge 가 `Emphasis` 생성 정보를 남깁니다). 옛 엔진은 `Ignore` 에서도 가중치를 곱하므로 예전처럼 켜집니다. SD1/SDXL 은 NegPiP 가
  음수 항을 프롬프트에서 빼 따로 인코딩하므로 이 판단이 없습니다(상류 그대로).
- **sd-forge-negpip 를 지우세요**: 따로 설치된 `sd-forge-negpip` 가 함께 로드돼 있으면 두 개가 같은 훅을 두 번 걸어(Anima 는 -1 마스크가 두 번
  곱해져 NegPiP 가 상쇄됨) 내장 쪽이 쉬고 콘솔에 경고를 한 번 남깁니다. 판별은 스크립트 러너의 `NegPiP` 스크립트 객체와 모델에 남은 NegPiP 패치
  흔적입니다(남아 있는 `lib_negpip` 모듈만으로는 쉬지 않음 — 확장을 꺼도 재시작 전까지 남기 때문). `extensions/sd-forge-negpip` 를 지우고(또는 끄고)
  Forge 를 재시작하면 내장이 대신합니다.
- **Anima 3.8B**: 3.8B 런타임이 NegPiP 마스킹을 대신할 때(3.8B 가 NegPiP 위에 있는 순서) 내장 마스크 규칙을 씁니다 — 아래에 깔린 것이 따로 설치된
  sd-forge-negpip 여도 그렇습니다. 옛 엔진에서는 `b3673ce` 와 같고, 새 엔진에서는 emphasis `Original`·`No norm` 이면 `0585496` 과 같으며
  `None`·`Ignore` 면 위 규칙대로 뒤집는 행이 없습니다(단독 `0585496` 과 다름). 개발 빌드 한때의 '새 엔진이면 경고 후 건너뛰기'는 없앴습니다 — 헬퍼가
  두 엔진을 다 알고, 헬퍼 오류를 삼키면 NegPiP 가 조용히 꺼지기 때문입니다.
- **내장 NegPiP 스위치 (결과 같음, 토글, 기본 켬)**: Settings → **SAM Extra NegPiP** → "내장 NegPiP 사용 (음수 가중치가 있으면 자동
  적용)"(`sam3_builtin_negpip_enabled`)이 생겼습니다.
  - 기본 켬은 지금까지와 같습니다.
  - 끄면 내장 NegPiP 만 건너뛰고 음수 가중치는 순정 Forge 가 처리하며, infotext 에 `SAM Extra NegPiP enabled: False` 가 남습니다. 붙여
    넣기·`override_settings` 로 복원됩니다.
  - `NegPiP` infotext 키는 내장본이 쓴 것만 지웁니다.
  - 스크립트 인수 0개(`ui()` 없음) 계약은 그대로이고, 따로 설치한 sd-forge-negpip 에는 적용되지 않습니다.
  - 2026-10-02 검토 제안을 편입했습니다. 실제 Forge 에서 Settings 항목(기본 켬)과, 끈 요청의 `SAM Extra NegPiP enabled: False`
    기록·NegPiP 미적용을 확인했습니다.

### 캐릭터 레퍼런스 (이어붙이기 · IP-Adapter)

- **방식 토글 (새 기능)**: 패널 맨 위에서 **이어붙이기**(split canvas + Anima Edit img2img)와 **IP-Adapter** 를 고릅니다. IP-Adapter 는
  참조 이미지를 SigLIP2 로 읽어 Anima DiT 블록에 K/V 를 주입하므로 캔버스가 없고 구도가 자유롭습니다. 가중치(어댑터 503 MB + 인코더 1.5 GB)는 전문가 설정의 **모델
  받기** 를 눌러야 커밋 해시로 고정해 받습니다. 주입은 샘플링 동안에만 살아 있고, 어댑터가 모델보다 깊거나 블록 계보 매핑이 없는 조합, 차원이 다른 어댑터, 빈 블록이 있는 어댑터(그대로 두면
  랜덤 투영이 잔차에 더해짐)는 숫자를 보여 주고 멈춥니다.
- **28블록 어댑터를 2.9B·3.8B 에 얹기 (새 기능, 토글, 기본 lineage)**: Edit LoRA 와 같은 블록 계보표로 40·52블록에 펼칩니다. 기본(`lineage`)은 각
  어댑터 블록의 원래 계보 자리에만 주입하고 끼워 넣은 블록(2.9B 12개·3.8B 24개)에는 IP 모듈도 어댑터 LoRA 도 걸지 않습니다. 설정: Settings → **SAM Extra
  Character Reference**(`sam3_reference`) → "캐릭터 레퍼런스 IP-Adapter: 28블록 어댑터를 2.9B·3.8B 에 얹을 때 끼워 넣은 블록 처리 (결과가
  바뀜)" (`sam3_ipa_duplicate_policy`: `lineage` 기본 / `all` 대응 블록 전부 / `split` 대응 블록끼리 강도 1/n). GPU 비교 (참조 1장·시드
  12345·강도 1.0): lineage 는 2.9B·3.8B(커넥터 켬·끔) 모두 깨지지 않았고, split 은 격자 무늬가 남았고, all 은 둘 다 깨졌습니다(2.9B 는 강도 1.0,
  3.8B 는 0.5 에서도). lineage 에서는 주입하는 블록이 줄어 같은 설정의 3.8B 한 장이 17.2 → 14.0 초로 줄었습니다. 매핑이 걸린 생성의 infotext 에
  `SAM3 IPA Blocks: 28→40`/`28→52`(어댑터→모델 블록 수)와 `SAM3 IPA Duplicates: <정책>` 이 남고, 상태 줄에 'IP-Adapter 28블록 → 52블록
  모델 (<정책 설명>, 강도를 다시 잡으세요)' 가 표시됩니다(lineage 면 '블록 계보 매핑·끼워 넣은 블록 제외'). 28블록 베이스에서는 어느 값이든 결과가
  같습니다.
- **IP-Adapter 에도 3.8B v2 커넥터 (토글, 기본 켬 — 끄면 0.6B 조건)**: Settings → **SAM Extra Anima 3.8B** → "Anima 3.8B: 캐릭터
  레퍼런스 IP-Adapter 방식에도 v2 커넥터 설치 (끄면 결과가 바뀜)"(`sam3_anima38_reference_ipa`, 기본 켬). v2 번들이면 txt2img 와 같은 설치·복원
  순서로 Qwen3.5 커넥터 조건을 만들고(강도 1.0, 부정 프롬프트 순정 경로) 그 위에 IP-Adapter 를 얹습니다 — 이어붙이기 방식과 조건이 같아집니다. 끄면 0.6B 조건으로
  샘플링합니다. infotext 에 `Reference Anima38: v2 bundle / not a 3.8B v2 bundle / missing encoder (…) / install failed
  (…) / check failed (…) / unavailable` 와 `Anima38 adapter/strength/architecture/bundle/negative/encoder` 키가 남고,
  상태 줄에 `3.8B 커넥터: <라벨>`(설치하지 못하면 '꺼짐' 과 이유)이 표시됩니다. v2 번들이 아닌 모델(2.9B 등)은 결과가 같습니다.
- **IP-Adapter 잡이 뒤 생성에 흔적을 남기지 않음**: 잡이 끝나면 이 확장이 건 Forge 객체 패치를 직접 되돌려, 뒤의 txt2img 에서 LoRA 가 `[LORA] Mismatch`
  로 통째로 빠지지 않습니다(GPU 확인: IPA 잡 4번 뒤의 txt2img 가 IPA 를 쓰기 전과 픽셀 단위로 같음). IPA 잡 바로 뒤의 첫 v2 설치(txt2img·이어붙이기)에서 커넥터
  조건이 조용히 빠지고 infotext 에만 Anima38 이 남지 않도록, v2 래퍼는 객체 패치 전의 원본 DiT 를 감쌉니다(버그 수정이라 토글 없음, 객체 패치가 없는 정상 흐름은 비트 단위로
  같음). IPA 잡은 샘플링 중 예외로 남은 다른 생성의 3.8B 설치를 먼저 내립니다.
- **IP-Adapter 와 다른 확장의 forward 패치**: Safe PAG 를 한 번 켜서 생성했거나 3.8B 래퍼가 남아 있어도 주입이 적용됩니다. 원본 DiT 의 외부 forward
  래퍼(3.8B v2 조건 확장, NegPiP 마스크 릴레이)는 호출할 때마다 현재 것을 찾아 통과시키고, 블록 수준 래퍼(Safe PAG SLG/DAVE)는 IPA 의 껍데기 블록에는 적용되지
  않습니다. 주입이 한 번도 일어나지 않는 배선이 감지되면 조용히 계속하지 않고 원인(껍데기 cross_attn 을 거치지 않음 / 껍데기 블록을 돌지 않음 / IP 토큰
  `sam3_ip_tokens` 가 실려 오지 않음)을 적은 RuntimeError 로 멈춰 상태 줄에 표시합니다.
- **이어붙이기 패널 (Feature 6)**: 메인 6개(캐릭터 이미지, 가져오기, 유지 범위, 프롬프트, 후보 수, 생성/상태) + 접힌 전문가 칸입니다. 원본 ReStyler v1.2 조건(빈
  칸 `#000000` 그대로·디노이즈 1.0, Extend 기본 꺼짐)으로 돌고 결과 크기는 txt2img 를 따르며, 결과는 txt2img 갤러리에 넣습니다. Forge Neo 의
  `anima_do_reference` 를 생성 중에만 켜고 `finally` 에서 복구하며, Forge 본체 파일이나 저장된 설정은 바꾸지 않습니다. LoRA 는 자동 감지하고(점 포함 이름 안전)
  Edit LoRA 가 없을 때만 막습니다. Anima 가 아닌 모델은 막고, Forge 재시작 직후에도 선택한 체크포인트를 먼저 불러와 판정합니다. 3.8B v2 번들은 레퍼런스 실행에도 커넥터를
  설치하고 레퍼런스 latent 를 DiT 에 넘깁니다. ⏹ Stop 뒤의 다음 실행도 정상으로 돌고, 투명 PNG 도 검게 나오지 않습니다. infotext 에 실제 생성 패널 크기·결과 크기·확대
  배율·masked content·모델 블록 수·3.8B 커넥터 상태(`Reference Anima38`)를 남깁니다(Forge 의 `Size` 는 캔버스 크기). 패널은 Gradio 워커 스레드에서
  돌고 작업이 끝나면 Generate 처럼 interrupted/skipped 플래그를 정리합니다.

### LoRA · DoRA

- **ANIMA 28/40/52블록 LoRA 양방향 호환**: Forge 의 LoRA 로드 지점을 확장 안에서 감싸 Base 1.0(28)·2.9B(40)·3.8B(52) 사이 여섯 방향을 자동으로
  옮깁니다. 기존 Base→2.9B 의미는 그대로이고, 3.8B 는 체크포인트 metadata 의 LLaMA-Pro 삽입 위치를 씁니다. 하향 변환은 상속 블록만 남기는 손실 투영입니다. 적용할 수
  없는 3.8B Semantic Connector 전용 키는 개수와 함께 경고하고 빼며, 별도 Qwen3.5 encoder 키는 블록 수만으로 지우지 않습니다. Forge 본체 파일은 고치지
  않습니다.
- **부분(sparse) LoRA (결과 변화, 토글, 기본 꺼짐)**: 일부 블록만 든 LoRA 는 순정 Forge 판정(가장 큰 블록 인덱스+1 이 들어가는 가장 작은 28·40·52)이 현재
  모델과 같으면 그대로 로드합니다(콘솔 `Loading sparse ANIMA LoRA as-is …`). 다르면(예: 앞 21블록만 → 2.9B) 기본으로는 건너뛰고 gr.Warning 토스트를
  띄웁니다 — 원래 학습한 모델을 확정할 수 없다는 안내이며 '앞 N블록만 담은 접두 LoRA' / '중간이 빈 LoRA' / '판정 불가' 로 나뉘고 앞의 둘에만 이 설정 안내가 붙습니다. 3.8B
  Semantic Connector 키만 든 LoRA 를 Base/2.9B 에 얹을 때와 블록 수를 모르는 모델에서도 같은 토스트가 뜨며, 토스트는 LoRA 조합이 바뀔 때 한 번 뜹니다. 콘솔 거부
  경고에는 `(Forge's own rule would call it <레이아웃>)` 이 붙습니다. 순정처럼 추측 변환하려면 Settings → **SAM Extra
  LoRA**(`sam3_lora`) → "Anima 부분 LoRA 순정 추측 변환 (일부 블록만 담은 LoRA — 블록 대응이 틀릴 수
  있음)"(`sam3_anima_sparse_lora_forge_guess`, 기본 꺼짐)을 켜세요. 켜면 판정 레이아웃에서 현재 모델로 이 확장의 대응표·끼워 넣은 블록 정책을 써서 변환하고,
  순정이 거부하는 하향(52→40 등)도 변환합니다(인덱스 52 이상은 건너뜀). 중간이 빈 LoRA 는 순정과 결과가 다를 수 있습니다. 추측 변환한 생성은 콘솔 경고·정보 토스트와 infotext
  `Anima sparse LoRA: "Forge guess (<파일> 28->40)"` 을 남기고, 토글을 바꾸면 다음 생성에서 다시 합칩니다. UI 없는 always-on 스크립트 `SAM
  Extra Anima sparse LoRA` 가 하나 늘었습니다.
- **DoRA 추론 방식 (토글, 기본 꺼짐 = 순정)**: `dora_scale` 이 든 LoRA·LoKr·LoHa 를 합치는 방식을 고르는 아코디언(더 보기)입니다.
  - **방식**: 순정(Forge/ComfyUI — 원본 가중치의 노름으로 나누고 대부분의 GPU 에서 fp16 이라 같은 파일이 학습 샘플과 다르게 나올 수 있었음)·**LyCORIS**(학습과
    같은 공식, 합친 가중치의 노름으로 나눔, fp32)·**Forge/Comfy 공식 · fp32**·**DoRA 끔**(크기 보정 없이 ΔW 만, 실험용).
  - **끼워 넣은 블록**: 작은 Anima LoRA 를 큰 모델에 얹을 때 복제되는 블록을 **그대로 복제**(순정, 기본)·**덧셈형** (끼워 넣은 블록만 DoRA 크기 보정을 뺌)·**넣지
    않음**·**약한 복사**(덧셈형 복제본의 ΔW 를 0~1 배로, 범위 어텐션만 / 어텐션+MLP / 전체, 기본 0.12·어텐션만)로 채웁니다. 3.8B 의 끼워 넣은 블록은 출력 가중치가
    작아 2.9B DoRA 를 그대로 복제하면 출력 투영이 레이어별로 자기 크기의 0.35~10.9배 바뀌지만, 렌더 비교(시드 3개)에서 덧셈형·넣지 않음이 순정보다 일관되게 낫지는 않아 기본은
    순정입니다.
  - **적용·기록**: 선택을 바꾸면 다음 생성에서 한 번 다시 합치고, 합친 상태를 모델에 표시해 Reload UI·체크포인트 전환 뒤에도 낡은 가중치를 쓰지 않습니다. infotext `DoRA
    mode`/`DoRA inserted`, 붙여 넣기, XYZ `[DoRA] Inference mode`·`[DoRA] Inserted blocks`·`[DoRA] Weak copy
    strength`·`[DoRA] Weak copy scope`(cost 0.8 — 바깥 루프로 가서 칸마다 다시 합치지 않음), API `alwayson_scripts["DoRA
    Inference Mode"]` 를 지원합니다.

### 가이던스 (PAG · SEG · APG · CFG · Skimmed CFG · Detail Daemon · DCW · RDC · DAVE · CNS)

- **PAG/SEG/SLG: 첫 target 블록 앞의 weak 행 중복 계산 건너뛰기 (결과 변화, 토글, 기본 켬)**: 첫 target 블록(기본 18) 이전 블록은 cond/uncond 행만
  돌리고 weak 행 자리에는 cond 행 출력을 복사합니다. GPU 에서 PAG 한 장이 56.2 → 51.4 초였고, 다시 돌려도 결과가 같으며, 끈 것과 구도는 같고 잔 디테일만 다릅니다. 끄는
  설정: Settings → **SAM Extra Guidance**(`sam3_guidance`) → "PAG/SEG/SLG: 첫 target 블록 이전의 weak 행 중복 계산 건너뛰기"
  (`sam3_guidance_pag_prefix_dedup`, 기본 켬). perturbation 을 켠 생성의 infotext 에 `Anima PAG prefix dedup: True/False`
  가 남고 붙여 넣기·`override_settings` 로 복원됩니다. 행을 자를 수 없는 블록(모르는 블록 인자, 행 수가 맞지 않는 텐서)부터는 예전 전체 배치 경로로 돌고 생성당 한 번 `앞쪽
  블록 중복 제거를 블록 … 부터 건너뜁니다` 경고를 남기며, 잘린 forward 가 OOM 이 아닌 예외를 내면 그 패스는 중복 제거를 끕니다. 로그: `attached ✅` 줄의
  `prefix_dedup=… seg_separable=…`, generation summary 의 `prefix_dedup_blocks=N prefix_dedup_fallbacks=M`.
- **SEG(공식) query blur 를 가로·세로 1D 두 번으로 (결과 변화, 토글, 기본 켬)**: 픽셀당 곱셈이 k² 에서 2k 로 줄어듭니다(1024² 면 3969 → 126). GPU
  에서 끈 것과 구도는 같고 잔 디테일만 다릅니다. 끄는 설정: Settings → **SAM Extra Guidance** → "SEG(공식): query Gaussian blur 를 가로·세로
  1D 두 번으로 계산"(`sam3_guidance_seg_separable_blur`, 기본 켬). 공식 SEG 이고 0<sigma≤9999 인 생성의 infotext 에 `Anima SEG
  separable blur: True/False` 가 남습니다. 두 Guidance 키는 XYZ 칸 사이 정리 목록에도 들어 있습니다. 예전 이미지를 그대로 재현하려면 두 설정을 모두 끄세요.
- **SMC / APG / CWM 은 CFG 1 에서 건너뜀 (결과 변화, 토글 없음)**: 레거시 CFG base mode·experimental stack 포함, CFG 1 이면 base 교체를
  건너뛰고 Forge 의 원래 결과를 둡니다. 예전에는 APG(eta=0)가 출력을 거의 0(검정/회색 이미지)으로 만들고 CWM/SMC 도 임의로 재가중했습니다. 원본 DCW(+a) 는
  SMC·CWM 을 CFG 1 에서도 돌리고 이 확장도 SMC·CWM 이 켜진 CFG 1 패스에는 원본처럼 `disable_cfg1_optimization` 을 걸지만, Forge 는 CFG 가 1
  이면 negative prompt 를 인코딩하지 않아 uncond 가 없으므로 결국 건너뜁니다. 콘솔에 생성당 한 번 `CFG base override (<켜진 것>) skipped:
  cond_scale=1` 경고, 진단의 CFG base 판정은 `NO-OP` 입니다. CFG=1 판정은 Forge 가 넘기는 `cond_scale` 로 하고, 없을 때만 uncond 를 봅니다.
- **Safe PAG: CFG 1 에서 rescale 자동 끄기를 적용하지 않음 (결과 변화, 토글 없음)**: CFG 1 이라 APG 가 돌지 않는 스텝에서는 PAG rescale 도 그대로
  적용합니다. 그래서 APG + rescale 자동 끄기를 켠 CFG 1 생성은 결과가 달라지고, 이제 APG 를 끈 생성과 비트 단위로 같습니다. CFG > 1 은 같습니다. 첫 CFG 1 경고에
  이 점이 적히고, `disable_cfg1_optimization` 이 켜져 있으면 uncond 패스는 돌았지만 결과가 cond 예측 그대로라고 안내합니다.
- **Safe PAG 확장 배치 OOM (결과 변화, 토글 없음)**: 실패한 forward 의 활성값을 놓은 뒤 폴백합니다. OOM 이면 캐시를 비우고 그 생성의 남은 스텝·hires 패스·남은
  배치에서 PAG/SEG/SLG 를 끕니다(예전에는 매 스텝 재시도). OOM 이 난 생성만 결과가 달라질 수 있고, 이때 hires·둘째 배치 이후 이미지의 infotext 에는 PAG 항목이 빠질
  수 있습니다.
- **Safe PAG 를 원본 노드와 같게 (결과 변화, 토글 없음)**:
  [iljung1106/comfyui-anima-safe-pag@905b0107](https://github.com/iljung1106/comfyui-anima-safe-pag) (MIT)
  기준입니다. PAG 의 Start/End 는 원본처럼 모델 스케줄의 σ 로 바꿔(`percent_to_sigma`, 양 끝 포함) 모델 호출마다 현재 σ 로 판정합니다. 예전에는 한 스텝 늦게
  오르는 Forge 스텝 비율을 써서, Anima(shift 3)의 기본 구간 0.0~0.7(σ 1.0~0.5625)이 simple 스케줄 20·28·30 스텝에서 원본의 15·20·22 스텝이
  아니라 14·19·21 스텝에 걸렸습니다. 이제 스텝 수·스케줄러·img2img denoise 가 달라도 원본과 같은 σ 구간이고 2차 샘플러의 중간 평가도 σ 로 판정합니다(predictor 가
  없는 모델만 예전 스텝 비율). SEG·SLG 는 원본 노드에 없는 기능이라 예전처럼 스텝 비율입니다. Attn Scale 상한을 15 → 100 으로 올렸고(슬라이더·API·XYZ),
  블록·head 번호의 역범위(`20-18`)는 원본처럼 18~20 으로 읽습니다(예전에는 그 부분을 버림, SLG·DAVE 블록 칸도 같음). ControlNet 이 붙은 호출에서
  PAG/SEG/SLG 를 쉬는 것은 원본에 없는 이 확장의 안전장치라, 실제로 막힌 패스의 infotext 에 `Anima Perturbation ControlNet guard` 를 남깁니다.
  `Anima Perturbation Guidance` infotext 에는 PAG σ 창(`pag_sigma_window=`)이 붙습니다.
- **Skimmed CFG 를 원본 노드와 같게 (결과 변화, 토글 없음)**:
  [Extraltodeus/Skimmed_CFG@d8300583](https://github.com/Extraltodeus/Skimmed_CFG) 의 수식 함수를
  `sam3ext/guidance/skimmed_cfg.py` 에 그대로 편입했습니다(Apache-2.0, 고지는 `THIRD_PARTY_NOTICES.md`). Start/End/Flip at 을
  원본처럼 모델의 `percent_to_sigma` 로 σ 로 바꿔 `end σ < σ < start σ`(경계 제외)인 스텝만 깎습니다 — 예전에는 한 스텝 늦은 스텝 비율이었습니다. 그래서
  flow 모델인 Anima 는 첫 스텝(σ 1)을 깎지 않고, start > end 면 예전처럼 바꿔 읽지 않고 아무 스텝도 깎지 않습니다. flip 규칙(flip 지점의 σ 보다 앞선 스텝에서
  뒤집음)·깎는 순서도 원본대로이고, 원본에 없던 `nan_to_num` 은 뺐습니다. RescaleCFG·Dynamic Thresholding 처럼 등록된 `sampler_cfg_function`
  은 버리지 않고 깎인 예측으로 다시 부르며, 없으면 Forge 처럼 `edit_strength` 를 반영한 선형 CFG 입니다. Flip at 슬라이더 step 은 0.05 → 0.01 이고,
  `ui-config.json` 에 저장된 예전 step 은 첫 시작 때 한 번 지웁니다.
- **DCW·CWM·SMC 를 원본 DCW(+a) 와 같게 (결과 변화, 토글 없음)**:
  [namemechan/ComfyUI-DCW@66aaf9dd](https://github.com/namemechan/ComfyUI-DCW) 기준으로 기본값·범위를 맞췄습니다: DCW λ low
  0.10 → 0.05(−0.5~0.5), λ high 0.02 → 0.01(범위 ±0.5 → ±0.3, step 0.001), CWM α low·high 0.30·0.15 → 0(범위 −1~1 →
  −1~2, 원본 권장 시작값 low 0.1~0.3·high 0.1~0.2), SMC Custom λ 0.5~30(API·XYZ 값도 이 범위로 맞춤). CWM 은 원본처럼 α 가 0 이 아닐 때만
  CFG 를 바꾸므로 새 기본값에서는 켜도 표준 CFG 와 같습니다. SMC/APG/CWM 의 CFG 배율은 원본 cfg 훅처럼 Forge 가 넘기는 `cond_scale`(다른 CFG 함수가 없으면
  Forge 처럼 `edit_strength` 를 곱함)이고, incoming 결과를 최소제곱으로 맞춘 값은 진단(`[VERIFY]` 의 `w_fit`)과 비선형 CFG 경고에만 씁니다.
  RescaleCFG·Dynamic Thresholding 처럼 다른 확장이 `sampler_cfg_function` 을 걸어 두면 원본처럼 SMC·CWM 만 비키고(경고 1회) 그 결과를 둡니다 —
  APG·PAG/SEG/SLG·DCW/RDC 는 그대로 적용합니다. Adaptive Guidance 가 uncond 를 건너뛴 스텝에도 원본 post-CFG 훅처럼 DCW/RDC 를 적용하고, 그
  스텝에서는 APG momentum 만 비우고 SMC 의 이전 오차는 유지합니다(예전에는 둘 다 비우고 DCW 도 건너뜀). infotext `Anima DCW` 는 DCW 가 실제로 돌 때(λ 가
  0 이 아니거나 RDC 가 켜짐) 남습니다.
- **DCW / CWM / SMC 명시적 ON/OFF**: Guidance 본문에 세 기능의 독립 체크박스를 두었습니다. SMC 는 고른 `Auto`/모델별/`Custom` 프리셋 값을 유지한 채
  master 체크박스로 바로 A/B 할 수 있습니다. CWM 은 원본처럼 α 가 0 이 아닐 때만 CFG 를 바꾸므로, 원본 기본값 0 에서는 켜도 결과가 같습니다. RDC 에는 따로 켜는
  체크박스가 없습니다(아래). 기존 script argument 와 XYZ 축 정수 인덱스는 그대로 두고 새 입력을 맨 뒤에 붙였습니다.
- **RDC 이식 (새 기능, 토글 tau, 기본 0 = 끔)**: [namemechan/ComfyUI-DCW](https://github.com/namemechan/ComfyUI-DCW) 의
  band-wise reverse drift compensation 을 Forge post-CFG 경로에 다시 작성했습니다. 원본처럼 따로 켜는 스위치가 없고 **Enable DCW 가 켜져 있고
  tau > 0** 일 때 DCW 의 Haar 변환 안에서 돕니다(DCW λ 를 둘 다 0 으로 두면 RDC 만). 이전 v0.30 개발 빌드의 Enable RDC 체크박스는 화면에서 뺐고, 그
  script argument 자리(58)는 남아 API 가 False 를 보내면 RDC 를 끄며 XYZ `[Anima RDC] Enable` 도 False 일 때만 끕니다.
  `tau`(0~0.5)·`alpha LL`(기본 0.03)·`alpha HH`(기본 0)를 UI/XYZ/infotext(`Anima RDC`)에 모두 노출하고, 생성마다 EMA 를 초기화하며
  해상도가 바뀌면 다시 시작합니다.
- **DAVE 를 원본 노드와 같게 (결과 변화, 토글 없음)**:
  [sorryhyun/ComfyUI-Anima-DAVE@83143e8d](https://github.com/sorryhyun/ComfyUI-Anima-DAVE) (MIT) 의 초반 스텝 게이트를
  `sam3ext/guidance/dave_gate.py` 로 옮겼습니다. 예전에는 한 스텝 늦은 Forge 스텝 비율이 tau 보다 작을 때 켰고, 이제 원본처럼 모델 호출의 σ 를 샘플러가 도는
  σ 스케줄에서 찾아 그 스텝 번호가 `k = max(1, min(n, round(tau × n)))` 보다 작을 때 켭니다(tau 0.10 이면 20·25 스텝에서 첫 2 스텝, 28·30 스텝에서
  첫 3 스텝). 스케줄에 없는 σ(2차 샘플러의 중간점)는 원본처럼 늘 켭니다. txt2img 는 σ 목록 전체, img2img·hires 는 Forge 와 같은 `steps − t_enc − 1`
  칸부터 세고, ADetailer 내부 img2img·img2img-hires-fix 처럼 이 스크립트가 준비하지 않은 실행은 `on_cfg_denoiser` 로 그 실행을 알아내 자기 스텝 수로
  끝에서 셉니다. σ 목록이 없는 DDIM·PLMS 는 Forge 스텝 위치로 판정합니다(한 스텝 늦음). 블록 칸을 비우면 원본 마스크와 같은 `8-18`, strength 0.001 이하는
  원본처럼 아무것도 하지 않습니다.
- **DAVE + Detail Daemon 우회 (결과 변화, 토글, 기본 켬)**: Detail Daemon 은 모델에 넘기는 σ 를 줄이는데, 줄어든 σ 는 스케줄에 없어 위
  원본 규칙대로면 '0번 스텝'으로 판정되고 DAVE 가 모든 스텝에 걸려 이미지가 무너집니다(GPU 확인: tau 1.0 과 같은 모양. 원본 ComfyUI 노드 둘을
  이어도 같습니다). 켜면 Detail Daemon 이 줄이기 전 σ 로 판정해 둘을 함께 써도 DAVE 가 tau 구간에만 걸립니다. Detail Daemon 을 끈 생성은 켜고
  끔에 관계없이 같습니다. 끄는 설정: Settings → **SAM Extra Guidance** → "DAVE + Detail Daemon: DAVE 적용 스텝을 Detail Daemon 이 바꾸기
  전 σ 로 판정(우회)"(`sam3_guidance_dave_pre_dd_sigma`). DAVE 를 쓴 생성의 infotext 에 `Anima DAVE pre-DD sigma: True/False` 가 남고
  붙여 넣기·`override_settings` 로 복원됩니다.
- **CNS 를 원본과 같게 (결과 변화, 토글 없음)**:
  [namemechan/comfyui-cns_sampler_patch@42278b13](https://github.com/namemechan/comfyui-cns_sampler_patch) 의
  `color_noise_wavelet` 을 `sam3ext/guidance/cns.py` 에 그대로 편입했습니다(GPL-3.0). Strength 가 1 보다 작으면 흰 노이즈와 `lerp` 로만
  섞고 표준편차를 다시 맞추지 않습니다(원본처럼 조금 낮아짐). 재색칠 기준 `x_t` 는 post-CFG 입력 대신 원본처럼 샘플러 스텝 callback 의 `x`(그 스텝의 시작 상태)를 쓰고,
  callback 을 감쌀 수 없는 샘플러에서만 예전처럼 post-CFG 입력을 씁니다(검증 로그 `x_t=callback|post_cfg`). 기본 Gamma scale 은 3.0 → 2.0,
  범위도 원본대로입니다(Strength step 0.05, Gamma power 최소 0.1, Gamma scale 0.1~25 · step 0.1 — 원본 README 의 Anima +
  euler_ancestral_cfg_pp 권장값 3.0 은 라벨에 적음). 이 스크립트가 붙은 패스의 샘플러가 도는 동안에만 색칠해, ADetailer 내부 img2img 같은 중첩 실행은 흰
  노이즈 그대로입니다.
- **Detail Daemon 을 ComfyUI-Detail-Daemon 과 같은 값으로 (결과 변화, 토글 없음)**:
  [Jonseed/ComfyUI-Detail-Daemon@3394e44](https://github.com/Jonseed/ComfyUI-Detail-Daemon) 의 값과 σ 조회를 따르고, 노드가
  다루지 않는 Forge 동작은 muerrilla 원본을 따릅니다. 강도는 `σ × max(1e-6, 1 − 스케줄 × 0.1 × CFG)` 이고 CFG 는 hires 패스에서도 늘
  `p.cfg_scale` 입니다. 포크 때 빠졌던 ×0.1 을 되살려 같은 amount 가 노드의 `detail_amount` 와 같은 결과를 냅니다 — **v0.21.2 와 같은 강도는
  amount × 10** 이고, 이를 위해 Amount 를 −1~1 → −5~5(기본 0.10)로 넓혔습니다. 원본에 없는 프리셋·Multiplier·CFG 결합 토글과 [0.05, 3] 클램프는
  없앴습니다(API 위치 인자 1·10·12 자리는 남기고 읽지 않음). 스케줄 위치는 노드처럼 모델 호출마다 그 σ 를 샘플러가 도는 σ 목록에서 찾아(가장 가까운 칸, 칸 사이는 선형 보간)
  정하므로 Forge 스텝 번호의 한 스텝 지연이 없고, 2차 샘플러의 중간 평가도 노드와 같은 곡선 값을 읽습니다. txt2img 는 σ 목록 전체, img2img·hires 는 Forge 와 같은
  `steps − t_enc − 1` 칸부터 세고(`DDIM` 스케줄이 σ 를 스텝 수 + 2 개 내놓아도 맞음), ADetailer 내부 img2img·img2img-hires-fix 처럼 이
  스크립트가 준비하지 않은 실행은 그 실행의 스텝 수로 끝에서 셉니다. σ 목록이 없는 DDIM·PLMS 는 모델 호출 수로 셉니다. muerrilla 원본처럼 **Hires Pass**(새
  체크박스, API 인자 13, 기본 끔)를 끄면 기본 패스에만, 켜면 hires 패스에만 걸리고(v0.21.2 는 두 패스 모두), DPM adaptive·HeunPP2 에서는 꺼집니다. σ 는
  원본처럼 제자리에서 바꿔 NGMS·soft inpainting 도 바뀐 σ 를 봅니다. infotext `Anima Detail Daemon` 에
  exponent·offset·fade·smooth·hires 까지 남깁니다. Amount 라벨이 바뀌어 `ui-config.json` 에 저장된 v0.21.2 의 Amount 값·범위는 적용되지
  않습니다(기본 0.10 으로 시작).
- **디테일 단계 — TSR · Momentum Guidance · HiGS · HiFlow (새 기능, 토글, 기본 끔)**: Guidance 아코디언의 CNS 아래에 넷을 더했습니다.
  PAG/SEG/SLG 항 뒤, DCW 앞에서 켜진 것만 `HiFlow → Momentum → HiGS → TSR` 순서로 돌고, 추가 모델 호출이 없습니다.
  - TSR: ComfyUI `nodes_eps.py` 의 Temporal Score Rescaling 을 옮겼습니다(GPL-3.0). k 0.95, sigma 1.0.
  - Momentum Guidance: arXiv 2602.20360 을 논문 식으로 다시 구현했습니다. α 0.5, β 0.6, 노이즈 수준 창 0.30–0.95.
  - HiGS: arXiv 2509.22300 을 다시 구현했습니다. w 1.75, η 0, α 0.75, R_c 0.05, t 0.40–1.00.
  - HiFlow: Bujiazi/HiFlow 공식 코드의 방향·가속도 정렬을 옮겼습니다(Apache-2.0). txt2img hires fix 전용이며 α 1.0, β 0.5, cutoff 0.2
    입니다. 1차 패스의 x0 를 σ 별로 기록했다가 hires 패스에서 씁니다.

  넷 다 끄면 예전과 같은 경로입니다. 다음이 새로 생겼습니다.
  - infotext: `Anima TSR`·`Anima Momentum Guidance`·`Anima HiGS`·`Anima HiFlow`
  - XYZ: `[Anima TSR]`·`[Anima MG]`·`[Anima HiGS]`·`[Anima HiFlow]`
  - 검증 로그: `[VERIFY] detail`

  CPU 단위 테스트(원본 식 대조·경계 조건·post-CFG 경로)와 실제 Forge(neo 2.29.2, Anima 3.8B) 한 시드 실행으로 넷 모두 실제로
  적용되는 것을 확인했습니다. 화질 비교는 아직입니다. **HiGS 기본 w 1.75 는 Res Multistep 샘플러에서 이미지를 무너뜨립니다** —
  멀티스텝 샘플러에서는 w 0.5 이하부터 쓰세요(docs/GUIDANCE.md 9절).
- **S²-Guidance (새 기능, SLG mode, 기본 `Fixed`)**: SLG mode 를 `Stochastic (S²)` 로 고르면, 모델 호출마다 블록을 무작위로 골라 건너뛴
  weak 예측으로 `ω·(cond − drop)` 을 더합니다(arXiv 2508.12880, 공식 코드가 없어 논문으로 다시 구현). 기본값은 다음과 같습니다.
  - ω 0.25, drop ratio 0.05(28·40·52 블록에서 1·2·3 블록), 블록 0 제외, 구간 0.10–0.90
  - 뽑기는 시드로 재현됩니다.
  - 비용은 고정 SLG 와 같습니다.
  - XYZ `[Anima Pert] SLG Mode`·`[Anima S2]` 를 더했습니다.
- **Adaptive SMC (새 기능, SMC controller, 기본 `Unit-L2`)**: SMC controller 를 `Adaptive sign` 으로 고르면, sorryhyun 의 Anima 판(anima_lora
  `smc_cfg.py`, MIT)처럼 원소별 sign 과 이득 `α·mean|e|` 를 씁니다. 기본값은 α 0.2, λ 5 입니다.
  - 속도 공간 식을 x0 공간에서 같게 계산합니다(`σ_t/σ_prev` 보정, 원본 식과 대조하는 테스트).
  - 기존 `Unit-L2`(원본 DCW(+a) 식)는 1 MP Anima 에서 원소당 보정이 약 4e-4 라 거의 효과가 없어 이 선택지를 더했습니다.
  - XYZ `[Anima SMC] Controller`·`Adaptive Alpha`·`Adaptive Lambda` 를 더했습니다.
- **PAG 강도 곡선 (자체 실험, 토글, 기본 끔)**: Settings → **SAM Extra Guidance** → "PAG 강도를 sigma 구간 양끝에서 부드럽게 줄이기
  (자체 실험)"(`sam3_guidance_pag_cosine_envelope`)입니다.
  - 켜면 PAG 항에만 PAG σ 창 안의 `sin²(π·u)` 를 곱합니다. 양끝은 0 이고, 그 호출에는 PAG weak 행도 만들지 않습니다.
  - 논문 기법이 아닌 이 확장의 실험이며, 꺼 두면 결과가 같습니다.
  - infotext `Anima PAG cosine envelope`·`Anima PAG envelope status` 가 남습니다.
  - 2026-10-02 검토 제안을 편입했습니다. 실제 Forge 에서 곡선 적용(PAG 적용 스텝 24 → 23)은 확인했고, 화질 비교는 하지
    않았습니다.
- **Anima Optimal Scale (새 기능, 실험, 기본 끔)**: 새 아코디언 `Anima Optimal Scale (실험 · CFG-Zero* optimized-scale)`
  (`scripts/anima_cfg_optimal_scale.py`)입니다.
  - CFG-Zero*(arXiv 2503.18886)의 optimized-scale 식만 구현했고, **zero-init 은 넣지 않았습니다**.
  - Anima · CFG > 1 · 선형 CFG 결과에서만 `blend`(기본 0.25)만큼 더합니다.
  - Skimmed CFG, 다른 CFG 함수, 앞선 post-CFG 보정이 있으면 건너뛰고, 그 이유를 `Anima Optimal Scale status` 에 남깁니다.
  - 자기 콜백만 붙이고 떼므로 다른 확장의 콜백·wrapper 는 그대로입니다.
  - 2026-10-02 검토 제안을 편입했습니다. 실제 Forge 에서 28번 모두 적용됐고(PAG 와 함께 켜도 Suite 보다 먼저 돌아 같음),
    화질 비교는 하지 않았습니다.
- **Safe PAG 스크립트 인수 62 → 91개**: 새 입력(S²·Adaptive SMC·디테일 단계)은 맨 뒤 62–90 자리에 붙였습니다. 기존 인덱스는 그대로이고,
  62개만 보내는 예전 API 호출은 새 자리를 기본값(전부 끔)으로 둡니다.

### SAM3

- **SAM3 를 쓴 뒤 다른 생성의 결과가 바뀌던 문제 (결과 변화, 버그 수정, 토글 없음)**: sam3 는 import 때 프로세스 전역 TF32 를 켜서, 한 번이라도 SAM3 를 쓰면 그
  뒤의 모든 생성에서 fp32 행렬곱(LyCORIS DoRA 합치기, 3.8B 커넥터 등)이 TF32 로 돌았습니다(3.8B 평균 픽셀 차이 38, 2.9B 11). 이제 import·빌드 동안 바뀐
  설정을 되돌리고 SAM3 검출 동안에만 TF32 를 켭니다(마스크는 같음). GPU 확인: SAM3 를 쓴 뒤의 3.8B·2.9B 생성이 쓰기 전과 픽셀 단위로 같습니다.
- **SAM3 인페인트 패스의 시드 (결과 변화, 버그 수정, 토글 없음)**: In-flight·Refine 인페인트 패스가 Forge Seed 스크립트에 시드를 덮어써 txt2img 시드 칸
  값(API 면 -1 = 매번 무작위)으로 돌던 문제입니다. 이제 바깥 생성의 시드 (또는 SAM3 Seed)를 씁니다. GPU 확인: 같은 설정의 SAM3 생성 두 번이 픽셀 단위로
  같습니다(3.8B·2.9B). UI 에서 시드 -1 로 만든 예전 이미지와는 SAM3 부분이 다를 수 있습니다.
- **API/XYZ 생략 키의 기본값을 UI 와 통일 (결과 변화)**: `sam3_inpainting_fill` 은 `latent noise`→`original`,
  `sam3_unload_after` 는 False→True 입니다. 예전 값을 원하면 두 키를 명시하세요.
- **범위 밖 설정값은 끄지 않고 맞춤 (결과 변화)**: XYZ `[SAM3] Threshold` 1.5, Mask Blur -1, API 의 잘못된 수치 등이 오면 SAM3 가 조용히 꺼지는 대신
  범위로 맞춰 실행합니다(threshold 1.5→1.0, blur -1→0, steps 0→1, inpaint width 0→64 등). Inpainting Fill / Mode / Mask
  Mode / CN Control Mode / CN Resize Mode 는 대소문자·공백을 무시하고, 모르는 값은 기본값(original / Inpaint / Individual / Balanced
  / Crop and Resize)으로 바꾸며 stderr 에 `[-] SAM3: unknown <필드> value '<값>', using the default '<기본값>'.` 을 남깁니다(값을
  생략하면 경고 없이 기본값). 그래도 검증이 실패하면 `[-] SAM3: invalid settings, SAM3 disabled for this generation: …` 과 infotext
  `SAM3 Error` 입니다.
- **실패를 infotext 에 기록하고 VRAM 정리**: 검출·인페인트가 예외로 끝나면 저장 이미지의 infotext 에서 `SAM3 Enable: True` 대신 `SAM3 Error:
  <예외형>: <메시지>` 가 들어가고 stderr 에 `[-] SAM3: failed, image saved without SAM3: …` 가 남습니다. 'Unload after' 가 켜져 있거나
  CUDA OOM 이면 모델을 VRAM 에서 내립니다(Refine 은 앞머리 `[-] SAM3 Refine:`). 예외 자체는 예전처럼 전파됩니다. 배치에서 앞 장이 실패해도 뒤에 성공한 장은
  `SAM3 Enable: True` 로 저장됩니다.
- **'Unload after' 가 모델을 CPU RAM 에 보관 (결과 같음, 토글, 기본 켬)**: 검출 뒤 VRAM 에서 내리는 것은 같지만 모델을 RAM 에 약 3.4 GB 로 남겨 다음
  검출은 GPU 로 옮기기만 합니다(재빌드·`sam3.pt` 재읽기 없음). 끄는 설정: Settings → **SAM Extra SAM3**(`sam3_mask`) → "'Unload after'
  뒤 SAM3 모델을 CPU RAM 에 보관 (약 3.4 GB)" (`sam3_unload_keep_in_ram`, 기본 켬). 끄면 VRAM·RAM 에서 모두 해제하고 다음 검출마다 다시 빌드해,
  RAM 약 3.4 GB 를 아끼는 대신 이미지당 약 2.5~5 초(추정)가 더 듭니다. 끄고 Apply 하면 보관 중인 모델도 곧바로 해제하고, RAM 의 모델은 Reload UI·확장 언로드 때도
  해제됩니다. 로그는 `model moved from VRAM to CPU RAM (moves back on next detection).` / `model released from VRAM and
  RAM (reloads on next detection).`(실패 경로는 `… after the failure …`)입니다. 체크포인트·장치를 바꾸면 옛 모델을 먼저 해제해 두 모델(약 7 GB)이
  GPU 에 함께 있던 순간이 없어지고, 같은 경로의 파일을 교체하면(mtime·크기) 새로 빌드합니다. 내부 API `unload_sam3()` 는 RAM 에 보관했는지(bool)를 돌려주고 선택
  인자 `keep_in_ram` 을 받습니다(인자 없는 호출은 그대로).
- **In-flight 내부 패스에서 ADetailer 제외 (결과 변화, 토글 없음)**: 생성 중 SAM3 인페인트의 내부 패스에서 ADetailer 가 돌지 않습니다(🎯 와 같음). 바깥 생성의
  ADetailer 는 그대로 한 번 돕니다.
- **SAM3 ControlNet 의 Anima LLLite 전처리 (결과 변화, 토글 없음)**: SAM3 인페인트 패스에 넣는 ControlNet 유닛이 Anima ControlNet-LLLite
  이면 원본(kohya sd-scripts·ComfyUI-Anima-LLLite)처럼 제어 이미지를 그대로 받게 합니다. 채널 수와 Tile & Repair 여부는 모델 파일의 safetensors
  헤더로 읽고, 못 읽으면 파일 이름으로 봅니다. Tile & Repair LLLite(3채널)는 preprocessor 를 늘 `None` 으로 바꾸고 — 기본 `inpaint_only` 가 고칠
  영역을 비워 복구할 내용을 지웠습니다 — lineart·canny·depth 같은 다른 3채널 Anima LLLite 는 고른 preprocessor 를 쓰되 `inpaint_*` 만 `None`
  으로, 4채널 인페인트 LLLite 는 예전처럼 `inpaint_*` 를 `None` 으로 바꿉니다. 바꿀 때마다 stderr 에 한 줄 남기고, Anima LLLite 가 아닌 모델(SDXL
  `kohya_controllllite_*` 포함)은 건드리지 않습니다.
- **SAM3 빠른 버튼(🎯)**: txt2img 갤러리 ✨(hires fix) 옆에서 선택한 이미지에 지금 SAM3 설정을 바로 돌립니다. ✨ 와 같은 규칙(지금 txt2img 설정 + 그 이미지의
  시드, 결과 배치는 Forge 의 hires button gallery insert 설정)이며 SAM3 아코디언이 꺼져 있어도 돌고 다른 후처리는 돌리지 않습니다. 결과는 `-sam3` 접미어로
  저장하고 infotext 에 SAM3 설정과 `SAM3 quick: True` 를 붙입니다.
- **XYZ 축 순서**: `[SAM3] Checkpoint`·`[SAM3] Device` 축(cost 0.9)이 시드 등 다른 축보다 바깥 루프로 가서 칸마다 SAM3 를 다시 빌드하지 않습니다.
  칸 결과는 같고 순서만 바뀝니다.
- **API: 원본 기준 SAM3 — `sam3_source_image` (새 기능, 요청할 때만)**: img2img(denoise 0)로 이미 만든 이미지에 SAM3 만 돌리는 API 호출자가
  SAM3 state 에 `"sam3_source_image": "init"` 을 넣으면, 부모 패스 출력(denoise 0 이어도 VAE 왕복으로 픽셀이 조금씩 바뀜) 대신 init 이미지(Forge 처럼
  `img2img_background_color` 로 flatten)로 검출·인페인트합니다. 마스크 밖이 원본 그대로이고, 마스크 없음·Mask only·실패도 원본을 돌려줍니다. init 이
  없거나 인페인트 마스크·denoise > 0·얼굴 복원·크기가 다르면 예전처럼 출력을 씁니다. 원본으로 돌 때는 Forge img2img 색 보정
  (`img2img_color_correction`, 스크립트 뒤에 이미지 전체를 LAB 왕복)을 끕니다 — 켜 두면 원본이어도 픽셀 대부분이 바뀝니다(원본으로 돌 수 없는 init 이
  섞인 여러 장 요청은 끄지 않고 `output (color correction)`). infotext 는 `SAM3 Source: init image` 또는
  `SAM3 Source: output (<이유>)` 입니다. 키가 없으면(Forge UI·🎯·XYZ·예전 호출자) 동작과 infotext 가 그대로입니다(결과 같음). Sam3Args 밖에서 state
  로만 읽어 예전 빌드도 이 키를 오류 없이 무시합니다. `scripts/!sam3.py` 만 바뀌어 Settings → Reload UI 로 적용됩니다. 자세한 내용은 README
  워크플로 1 의 'API: 이미 만든 이미지에 SAM3 만'.
- **수동 마스크만 있으면 SAM3 를 불러오지 않음 (드물게 결과 변화)**: Target·Exclude 가 비어 있고 그린 마스크가 있으면 다음을 모두 건너뛰고 그
  마스크를 씁니다.
  - torch import·장치 확인·체크포인트 찾기(Hugging Face 받기 포함)·모델 빌드
  - 빈 글자 검출

  마스크·overlay 는 예전 '검출과 겹치지 않으면 그린 마스크 그대로' 경로와 같습니다. 다만 다음 두 경우는 결과가 다를 수 있습니다.
  - 예전에는 빈 글자 검출이 그린 마스크와 겹치면 교집합으로 좁혔습니다. 이제는 좁히지 않습니다.
  - 빈 마스크는 이제 패스를 만들지 않습니다.

  콘솔에는 `[-] SAM3: manual mask only; text detection and checkpoint loading skipped` 가 남고, 결과의 장치·체크포인트는
  `manual`·`not used (manual mask)` 입니다.
- **Refine: PNG 기록과 결과 안내 (새 기능)**: Refine 결과 PNG 에 검출·마스크 설정과 마스크 해시를 남깁니다.
  - 검출: `SAM3 Refine Target`·`Exclude`·`Threshold`·`Checkpoint Requested/Used`·`Device`·`Pass`
  - 마스크 설정: `Mask Dilation/Hull/Outline/Blur/Invert`·`Masked Content` 등
  - blur·invert 전 마스크의 SHA-256: `SAM3 Refine Mask SHA256`·`Mask Dimensions`·`Mask Stage`
  - 그린 마스크를 썼으면 `SAM3 Refine Manual Mask SHA256`

  결과가 없을 때는 이유를 나눠 알립니다.
  - 검출된 마스크 없음 / 사용자 중단 / img2img 스크립트 준비 안 됨 / 이미지 미반환 / 시도 N개 중 M개 패스 오류
  - 일부만 성공하면 추가한 개수 옆에 오류·미반환·중단 수를 붙입니다.

  내부 결과는 list 와 호환되는 `RefineResults` 라 예전 호출자는 그대로 동작하며, 화면 문구에 예외 원문은 넣지 않습니다. 2026-10-02 검토
  제안(02+03 병합본)을 편입했습니다.

### TIPO 프롬프트 확장 (🪄)

- **프롬프트 확장 버튼**: txt2img 도구 줄의 🪄 가 TIPO-v2.1-1B-A200M 으로 지금 프롬프트를 확장해 프롬프트 칸에 다시 씁니다. 스타일 줄 아래에서 확장
  방식(태그만/태그+설명/설명만)·길이·새 작가·캐릭터·작품 허용·장치·시드를 고르고 ↩ 로 되돌립니다. 적어 둔 텍스트는 그대로 두고 새 태그(캐릭터→작품→@작가→일반, 괄호 이스케이프)와 설명만
  뒤에 붙입니다. 모델 코드는 확장 안에 넣었고(KohakUwULLM, Apache-2.0) 가중치(1.98 GB, Kohaku License 1.0)는 **모델 받기** 를 눌러야 고정 버전으로
  받습니다. 기다리는 동안 프롬프트를 고치면 결과를 넣지 않고, 토큰 한도에서 잘린 끝은 버리며, 토큰 수를 다시 셉니다.
- **장치 (생성 텍스트 같음)**: 라디오는 GPU(기본) / CPU 이고, 여유 VRAM 이 3 GB 보다 적으면 그 회차는 CPU 로 돕니다. 옆 체크박스 "GPU 에 남겨 두기"(기본 켬)는
  GPU 로 돈 TIPO(fp16 약 2 GB)를 Forge 메모리 관리에 맡겨, 여유가 있으면 VRAM 에 남아 다음 클릭에 다시 올리지 않고 이미지 생성 등으로 자리가 필요하면 통째로
  내립니다(학습 등 Forge 밖 VRAM 사용은 모름). 끄면 누를 때만 올렸다 내립니다. 상태 줄 장치는 'GPU(남김)' 이고, Forge 메모리 관리를 쓸 수 없으면 'Forge 메모리 관리를
  쓸 수 없어 GPU 에 남기지 않았습니다' 를 표시합니다. 쉴 때는 RAM 에 fp16 약 2 GB(CPU 로 돌린 뒤에는 fp32 약 4 GB) 사본을 두어, GPU 에서 내릴 때
  GPU→CPU 복사를 하지 않고 CPU 로 다시 돌릴 때 fp16↔fp32 캐스트를 반복하지 않습니다. 장치 라디오는 브라우저가 기억하고, 체크박스는 새로 고치면 기본값으로
  돌아갑니다.
- **↩ 되돌리기가 그 뒤의 편집을 덮어쓰지 않음 (버그 수정)**: 확장 전·후 프롬프트를 함께 기억해, 프롬프트가 확장 결과 그대로일 때만 되돌립니다.
  그 뒤에 프롬프트를 고쳤으면 되돌리지 않고 안내합니다(2026-10-02 검토 제안).

### Anima VAE DeGrid (NAFNet)

- **VAE 격자 제거 (새 기능, 기본 끔)**: txt2img·img2img 의 **Anima VAE DeGrid (NAFNet)** 아코디언과 Extras 탭 항목이 생겼습니다.
  [DraconicDragon/NAFNet-VAE-DeGrid](https://huggingface.co/DraconicDragon/NAFNet-VAE-DeGrid)(Apache-2.0, v1.1 권장,
  `models/ESRGAN` 또는 `models/DeGrid`)가 내는 **잔차**를 이미지에 더해 Qwen/Wan VAE 격자 무늬를 지웁니다. 방식은 Full / Dark Pixels
  Mainly / Bright Pixels Mainly, 강도 0~1.5, 타일 크기(기본 512, 슬라이더 128 단위). 식·타일 위치·feather 가중치는 ComfyUI-NAFNet-Residual
  의 NAFNet Restoration 노드와 같고(같은 입력 텐서로 비트 단위 대조 — 노드 전체와는 channels-last 입력 때문에 float 로 ~1e-6, 8비트는
  여기서 반올림·ComfyUI SaveImage 는 버림), 마지막 이미지만 [0,1] 로 자릅니다. 16 의 배수가 아닌 크기는 타일마다 반사 패딩해
  (노드·spandrel 은 0 으로 채워 오른쪽·아래 가장자리 잔차가 최대 56/255 까지 커짐) 가장자리도 안쪽과 비슷하고, 표준 Anima 크기는
  모든 타일이 16 의 배수라 노드와 같습니다. 목록에는 state dict 가 NAFNet 인 파일만 나오고(헤더만 읽음, 선언된 버전이 높은 v1.1 이
  기본값), 이미지를 내는 일반 복원 NAFNet 을 고르면 출력이 입력의 무늬(상관)나 밝기(채널별 평균)를 따라가는 것으로 알아채
  적용하지 않습니다(회색 입자 바탕·화면을 채운 스크린톤에서 흐림·median 모델의 상관이 낮아져도 밝기로 거름. 화면을 채운 잔
  스크린톤·1px 체커에서 DeGrid 잔차가 커지는 것은 입력과 반대로 움직이므로 거절하지 않음). 화면을 채운 1px 줄무늬 같은 무늬에서
  잔차가 극단적으로 폭주하면(|평균| 100/255 초과 — 실측 183~2143/255. 어림 문턱이라 1px 체커·디더링은 그 아래에서도 망가질 수 있음) 그 이미지는 원본 그대로 두고
  `Anima DeGrid error: output blew up (…)` 을 남깁니다. 모델은 Forge venv 의 spandrel 로 불러옵니다. 모델 파일이 없으면 로그를
  남기고 건너뜁니다.
- **`.pth` 목록·불러오기 안전**: 목록을 만들 때 `models/ESRGAN` 의 옛 형식 `.pth`(zip 이 아닌 pickle — 예: `4x-UltraSharp.pth`)를
  `torch.load(mmap=True)` 로 열어 Forge 시작 콘솔에 `mmap can only be used with files saved with …` 오류와 `patch_basic.py`
  트레이스백이 찍히던 문제를 고쳤습니다. 이제 `.pth`/`.pt` 는 `torch.load` 없이 zip 의 `data.pkl` 에서 키 이름만 읽고(텐서는 만들지
  않고 pickle 속 전역도 부르지 않음), 옛 형식은 DeGrid 가 아니므로 열지 않습니다(디버그 로그 한 줄). 고른 모델은 Forge 가 감싸기 전
  로더(`torch.load_origin`·`safetensors.torch.load_file_origin`)에 `Path` 로만 넘깁니다 — Forge 가 감싼 로더는 실패하면 str 인자인
  파일을 `.corrupted` 로 이름을 바꿔 버립니다(이번 오류에서는 `Path` 를 넘겨 이름이 바뀌지 않았음). 목록 만들기는 헤더·`data.pkl` 만
  읽어 파일 12개(.pth 3개·safetensors 6개)인 `models/ESRGAN` 에서 처음 25~49 ms(실측), 그다음은 캐시(1 ms 미만)입니다.
- **순서**: 이미지마다 모든 always-on 스크립트의 `postprocess_image`(ADetailer·SAM3 인페인트 등)와 색 보정·인페인트 합성이 끝난 뒤,
  저장 직전(`postprocess_image_after_composite`)에 한 번 돕니다 — 설치 순서와 무관합니다. 메인 탭에 켠 Extras Upscale 보다는
  앞이고(`metadata.ini` 콜백 순서), SAM3·ADetailer 내부 패스에서는 돌지 않습니다. Extras 탭에서는 Upscale 보다 먼저 돕니다(항목 이름
  **Anima VAE DeGrid (NAFNet, Extras)** — 메인 탭에 켜도 생성 탭 스크립트와 API 이름이 겹치지 않음). 콘솔의 `[AnimaDeGrid] 켬` 줄은
  작업마다 한 번입니다 — ADetailer 가 배치의 마지막 장마다 `p.scripts.process(copy(p))` 로 process 를 다시 불러 이미지마다 두 번
  찍히던 것을 고쳤습니다(적용은 원래도 한 번).
- **infotext·API**: `Anima DeGrid model` / `mode` / `strength` / `tile`(실제로 쓴 타일 — 메모리 부족으로 줄였으면 줄인 값) /
  `precision`(실패·모델 없음은 `Anima DeGrid error`)을 남기고 PNG Info 붙여 넣기로 되살립니다(정밀도는 설정이라 기록만). API 는 위치
  인자 `[enabled, model, mode, strength, tile]` 또는 dict 하나.
- **새 설정 섹션 SAM Extra VAE DeGrid**(`sam3_degrid`): 계산 장치(auto / cpu), GPU 정밀도(`sam3_degrid_gpu_precision` — fp32 기본,
  ComfyUI 노드와 같은 계산 / fp16 autocast — fp32 가중치, 넘친 타일만 fp32 로 다시), VRAM 에 남기기(기본 끔 — 이미지마다 Forge
  `load_models_gpu` 로 올렸다가 내리고 캐시를 비움). 개발 중에는 키가 `sam3_degrid_precision`(기본 fp16 autocast)이었습니다 — RTX 5090
  에서 fp32 가 더 빨라(계산만 0.275 초 대 0.362 초) 기본을 바꾸면서, Forge 가 config.json 에 적어 둔 예전 기본 fp16 이 남지 않게 키도
  바꿨습니다(예전 키는 읽지 않음). 합성곱 TF32 는 torch 기본(켬) 그대로 둡니다(엄격 fp32 와 잔차 최대 0.11/255, 속도 같음).
- **검증**: CPU 단위 테스트 143개(실제 가중치 14개와 실제 `models/ESRGAN` 목록(읽기만) 1개는 `SAM3_RUN_FORGE_INTEGRATION_TESTS=1`
  일 때만, 그중 실제 Anima 조각은 `SAM3_DEGRID_ANIMA_IMAGES` 에 PNG 를 줄 때만) — Forge `patch_basic.build_loaded` 로 감싼 로더가 찾기·
  불러오기에서 불리지 않고 파일 이름도 바뀌지 않는지, 같은 입력으로 노드의 잔차 함수·
  Forge/ComfyUI `tiled_scale` 과 비트 단위 대조, Forge `processing.py`·`sort_callbacks`·`metadata.ini` 로 실행 순서 확인, 생성 탭
  (Forge 가 `torch.inference_mode()` 안에서 부름)에서 처음 불러 장치로 옮긴 뒤에도 계산되는지(CPU 대역: dtype 왕복 — 모델을
  inference_mode 밖에서 만들지 않으면 'Inference tensors do not track version counter.' 로 그 뒤 모든 이미지가 실패), 16 배수가
  아닌 크기의 가장자리, 이미지를 내는 NAFNet 거절(실제 v1.1·Anzhc 가중치로 잔 스크린톤·1px 체커·회색 입자·Anima 조각은 적용,
  같은 망이 복원 이미지를 내거나 흐림·median 이미지 모델이면 입자·스크린톤에서도 거절), 잔차 폭주 건너뛰기(실제 가중치로 화면을 채운
  1px 줄무늬·3px 세로줄·저대비 1px 줄무늬 — 원본 유지, infotext 문구), GPU 에 올릴 때 Forge 가 자리를 내려고 자기 모델을 옮겨도 부른 문맥(생성 탭은 inference_mode)에서 옮기는지. 실제
  v1.1 가중치(CPU): Anima 1216×1856 에서 잔차 |평균| 약 0.5/255, 8비트 값이 바뀐 픽셀 약 44%(대부분 1 단계), CPU 로 약 6.5 초. VAE 를 거친 적 없는 합성 그림을 Qwen-Image VAE 로 왕복한 뒤 돌리면 PSNR
  36.7 → 37.7 dB. GPU(RTX 5090, Forge 생성 탭, 1216×1856 — 그때 기본이던 fp16 autocast): 결과가 같은 이미지에 오프라인으로
  돌린 DeGrid 와 비트 단위로 같고(8 쌍), SAM3 인페인트·ADetailer 가 끝난 이미지에 돌며(SAM3 내부 패스에서는 안 돎), DeGrid 단계는
  한 장 0.77~0.96 초(모델 올리기·내리기 포함, Forge 를 켜고 처음은 4.1 초), VRAM 약 0.4 GB(가중치 111 MiB + 최대 활성 276~292 MiB).
  계산만은 fp32 0.27~0.30 초·fp16 autocast 0.34~0.38 초이고, fp16 autocast 와 fp32 는 잔차가 최대 0.32~0.43/255(8비트로 픽셀
  2~18% 가 1 단계) 다릅니다. ADetailer 복사본 재호출·정밀도 기본값·예전 설정 키 무시는 단위 테스트로 확인했습니다(fp32 기본으로
  Forge 에서 다시 재지는 않음).

### txt2img 화면 · 도구

- **txt2img 섹션 정리**: always-on 아코디언을 묶음별 섹션으로 나눕니다 — 1열(고정·ANIMA 튜닝), 2열(디테일러·스크립트·더 보기), 3열(갤러리 밑 **선택 이미지**
  탭: Refine·Tile-Repair·캐릭터 레퍼런스). 프롬프트 밑에 **켜진 기능** 칩 줄이 생기고 각 헤더에 `켜짐`/`k/n 켜짐` 표시와 핀이 붙습니다(핀은 브라우저에 기억). 항목은
  스크립트 파일 이름으로 알아보고 상태는 `data-*` 속성으로만 써서 예전 ko_KR 무한 루프가 재발할 수 없습니다. 가운데 열의 절대 위치·타이머 배치 엔진은 지웠습니다. Settings →
  **SAM Extra Appearance**(`sam3_appearance`) → "txt2img 섹션 정리(켜진 기능·고정·더 보기) — Forge 재시작 후
  적용"(`sam3_layout_sections`, 기본 켬) 또는 주소의 `?sam3_lanes=off` 로 끌 수 있습니다.
- **Refine · Tile-Repair · PiD 를 Forge 생성 큐 안에서 실행**: 세 패널의 실행이 txt2img Generate 와 같은 `queue_lock` 안에서 Forge 메인
  스레드로 돌아 서로 기다립니다(동시 실행 없음). 실행마다 중단 플래그를 초기화해, ⏹ Stop 뒤 다음 실행이 곧바로 `SAM3 Refine: no result (empty mask or
  interrupted)` 로 끝나던 문제가 없어졌습니다. 패널의 ⏹ Stop 은 그 패널 작업이 실제로 도는 중일 때만 중단합니다. txt2img 가 도는 동안 대기 중인 클릭은 Stop 으로
  취소되지 않고(txt2img 도 멈추지 않음) 잠금이 풀리면 그대로 실행됩니다. Anima Tile-Repair 의 ⏹ Stop 이 다음 디노이징 스텝에서 실제로 멈춥니다(모델 로드·텍스트
  인코딩·VAE 디코드 구간은 중단 불가). Refine 이 실패해도 진행 문구가 진행 표시줄에 남지 않습니다.
- **Tile-Repair 정리**: Text Encoder 자동 선택이 Qwen3 0.6B(예: `qwen_3_06b_base.safetensors`)만 고르고, 없으면 `Use Forge
  current` 가 기본입니다(예전에는 Qwen3-VL 등을 골라 크기 불일치로 죽었음). 제외 판정은 'vl'·'vlm'·'clip' 이 토큰일 때만입니다. 벤더에 전달된 적 없는 LLLite
  Strength / Start % / End % 슬라이더를 뺐고(Multiplier 만 남음 — 아래), infotext 의 LLLite 항목은 `LLLite: <model> (mult <x>)`
  로 짧아졌습니다.
- **Tile-Repair 를 sd-scripts 원본과 같게 (결과 변화, 토글 없음)**:
  [kohya-ss/sd-scripts@690ea7f9](https://github.com/kohya-ss/sd-scripts) 추론과
  [kohya-ss/ComfyUI-Anima-LLLite@b7495bd8](https://github.com/kohya-ss/ComfyUI-Anima-LLLite) 기준입니다. Width·Height
  슬라이더(기본 1024×1024) 대신 **Short Side**(기본 1024) 하나로 원본 비율을 지키고, 긴 변은 비율을 따라 두 변 모두 32 의 배수로 내립니다(최소 256). 디코드는
  sd-scripts 처럼 `(clamp(−1, 1) + 1) × 127.5` 를 uint8 로 잘라(예전에는 범위를 추정해 반올림) 픽셀 값이 1 씩 다를 수 있습니다. LLLite 목록에는
  safetensors 헤더로 가려낸 3채널(RGB) Anima LLLite 만 나오고(4채널 인페인트 LLLite·다른 ControlNet·`.safetensors` 가 아닌 파일은 빠짐), 기본은
  가장 새 Tile & Repair 파일(v20)입니다. Multiplier 는 ComfyUI-Anima-LLLite 의 strength 와 같은 −10~10(step 0.01, 기본 1.0)이고
  라벨이 바뀌어 저장된 예전 0~2 범위는 적용되지 않습니다. 네거티브 기본값은 sd-scripts 처럼 빈 칸이고, `ui-config.json` 에 예전 기본값(`blurry, low
  quality`)이 그대로 저장된 기존 설치는 첫 시작 때 한 번 빈 칸으로 옮깁니다(직접 적은 네거티브는 유지).
- **Tile & Repair HTTP API (새 기능)**: `POST /sam-extra/tile-repair` 가 JSON 하나로 패널의 Tile-Repair 모드와 같은 실행을 합니다.
  `image`(base64 PNG·JPEG·WebP, `data:` 접두어 가능)만 필수이고 나머지 키(`model`, `prompt`, `negative_prompt`, `steps`,
  `cfg_scale`, `flow_shift`, `multiplier`, `short_side`, `seed`, `dit`, `text_encoder`, `vae`,
  `unload_forge_before`)는 생략하면 패널 기본값(네거티브는 빈 칸)이며, 모르는 키는 400 으로 거절합니다. 결과는 base64 PNG(infotext 포함)·실제로 쓴
  시드·크기입니다. `GET /sam-extra/tile-repair/options` 는 패널의 선택지·기본값·범위를, `POST /sam-extra/tile-repair/stop` 은 이 경로의
  요청만 멈춥니다(패널 ⏹·txt2img 는 건드리지 않고, Forge 의 Interrupt 는 둘 다 멈춤). txt2img·패널과 같은 Forge 대기열에서 돌고, 입력은 디코드 64 MB·64
  MP, 출력도 64 MP 까지입니다(가는 원본에 short side 를 주어 수십억 픽셀이 되는 요청은 400). 인증은 Notebook 경로와 같습니다(`--gradio-auth` 로그인,
  Forge 가 API 를 띄울 때의 `--api-auth` HTTP Basic, 헤더 `X-SAM3-Notebook: 1`). LoRA 칸·PiD 모드·갤러리 삽입은 패널에만 있습니다. UR_IV
  데스크톱 앱처럼 같은 출처에서 부르는 클라이언트용입니다.
- **Anima VAE 2x 가 Anima 생성에서 실제로 돕니다**: 켜면 `TypeError: Cannot handle this data type` 로 생성이 죽던 문제를 고쳤습니다(순정과 같은
  5D 결과). decode 결과 모양이 맞지 않으면 `2x decode failed → stock decode: …` 를 남기고 순정 decode 로 폴백합니다. 12ch 디코더는 Forge 메모리
  관리로 decode 때만 올립니다. `--novram` 등으로 디코더가 GPU 에 못 올라가 실패하면 그 이유를 프로세스당 한 번만 안내하고 이후에는 조용히 순정 decode 로 넘어갑니다. 실제
  GPU 생성 확인은 아직입니다.
- **빠른 드롭다운**: UI 업데이트마다 도는 스캔이 Gradio config 전체(약 5,600개)를 선형 탐색하던 것을 elem_id 인덱스 조회로 바꿨습니다(스캔 1회 약 5.7 ms 중
  4.3 ms 가 이 탐색이었음). 설정값(기본 60개)은 이제 전체 상한이 아니라 한 번에 더하는 개수라, 목록 끝까지 스크롤하면 다음 묶음을 그려 XYZ 축처럼 긴 목록도 끝까지 볼 수 있습니다.
- **Notebook 메모장 (새 기능)**: Notebook 패널에 **프리셋 / 메모** 탭이 생겼습니다. 메모(제목 120자·본문 100,000자)는 입력을 멈추고 약 0.6 초 뒤 자동
  저장되며, `notebook.json` 옆의 `memos.json`(기본 Forge 데이터 경로의 `sam-extra/memos.json`)에 따로 저장해 프리셋 저장의 revision 과 섞이지
  않습니다. UR_IV 앱도 같은 메모를 `GET/PUT/DELETE /sam3-notebook/memos[/{id}]` 로 읽고 씁니다(Notebook 과 같은 인증·헤더). 저장·삭제는 메모마다
  마지막으로 본 `updated_at` 을 함께 보내, 그사이 다른 곳에서 바뀐 메모를 덮어쓰거나 지우지 않고 409 를 받습니다 — 화면은 서버 쪽 메모를 그대로 두고 내 편집을 '(충돌 사본)'
  새 메모로 남깁니다. 지운 메모는 동기화용 tombstone 으로 남고, 메모는 tombstone 포함 500개까지이며 넘치면 오래된 tombstone 부터 정리합니다. 저장 전 편집은 브라우저
  탭마다 localStorage 초안으로 남아, 닫힌 탭의 초안은 다른 탭이 이어받습니다. 경로가 없으면(업데이트 뒤 재시작 전) 메모 탭에 Forge 를 다시 시작하라고 표시합니다.
- **확장 HTTP 경로에 Forge `--api-auth` 적용**: Forge 가 `--api`·`--nowebui` 로 API 를 띄우고 `--api-auth` 가 있으면 Tile &
  Repair(`/sam-extra/tile-repair…`)·Notebook(`/sam3-notebook`)·메모 경로에도 `/sdapi` 와 같은 HTTP Basic 인증을 겁니다(없거나 틀리면
  401). `--gradio-auth` 로그인도 켜져 있으면 둘 다 필요합니다. `--api`·`--nowebui` 없이 준 `--api-auth` 는 Forge 에서처럼 아무것도 막지 않습니다.
- **LoRA Manager HTTP 경로도 같은 인증**: `/sam3-lora/config`·`/sam3-lora/spawn` 에 Notebook·메모·Tile & Repair 경로와 같은
  보호를 걸었습니다. 헤더 `X-SAM3-Notebook: 1` 이 없으면 403, `--gradio-auth` 로그인이나 (Forge 가 API 를 띄울 때) `--api-auth`
  HTTP Basic 이 없거나 틀리면 401 이고, 거절된 `/sam3-lora/spawn` 은 매니저 서버를 띄우지 않습니다. 페이지의 Manage 탭은 Gradio 버튼
  브리지를 써서 그대로 동작하고, 이 경로를 부르는 외부 도구는 헤더(와 자격 증명)를 보내야 합니다.
- **Notebook Apply/Undo 를 한 줄로 (버그 수정)**: Apply 와 Undo 를 같은 대기열에서 차례로 처리합니다. 처리 중에는 버튼을 잠그므로, 빠르게
  연속으로 눌러도 두 레시피의 값이 섞여 들어가지 않습니다(2026-10-02 검토 제안, JSDOM 테스트).
- **LoRA Manager 메시지 보낸 쪽 확인 (보안)**: LoRA 를 프롬프트에 넣는 메시지는 이 페이지가 띄운 LoRA Manager iframe 이 매니저 주소에서 보낸 것만
  받습니다(`event.source`·origin 확인). 다른 창·frame 이나 다른 포트가 보낸 메시지는 무시합니다(2026-10-02 검토 제안, JSDOM 테스트).
- **Anima VAE 2x 실제 decode 결과 기록 (새 기능)**: `Anima VAE 2x` infotext 끝에 `decode=pending|applied|stock fallback` 을 붙이고, 패스마다
  `Anima VAE 2x main outcome`·`Anima VAE 2x hires outcome`(`applied_calls=…; stock_fallback_calls=…; last=…`)을 남깁니다.
  - 순정 decode 로 폴백한 이유는 `… last error` 한 줄(240자 이내)로 남깁니다.
  - 기록이 실패해도 decode 결과는 바뀌지 않습니다.
  - 2026-10-02 검토 제안을 편입했습니다. 실제 GPU 생성 확인은 아직입니다.
- **Forge 라이트박스 도구줄이 늘 보이던 문제 (버그 수정)**: 전역 테마(Settings → **SAM Extra Appearance**)를 켜면 테마 배경이 Forge 라이트박스의
  도구줄(`.modalControls`, 이것도 `.gradio-container`)까지 칠해, 마우스를 올리지 않아도 바가 배경색으로 늘 보였습니다. 도구줄을 테마 배경에서 뺐습니다.

### 성능 (결과 같음)

- **Anima 3.8B 커넥터 스텝 비용 (결과 같음, 토글 두 개, 기본 켬)**: Settings → **SAM Extra Anima 3.8B** 에 두 설정을 추가했습니다. 끄면 각각 예전
  경로 그대로이고, GPU 에서 둘 다 픽셀 단위로 같았습니다.
  - "Anima 3.8B: 커넥터를 샘플링 동안 fp32 로 상주 (VRAM 약 +1.5 GB, 장당 약 1~2초 빨라짐)" (`sam3_anima38_connector_fp32`): Forge 가
    다 올린 뒤 커넥터를 fp32 로 한 번 바꿔 두고 스텝마다의 캐스트를 건너뜁니다. Forge 가 남긴 여유 안에서만 바꾸고, 내리거나 다시 패치할 때는 먼저 원래 dtype 으로 되돌립니다.
    바뀌면 콘솔에 `[Anima38] connector fp32 resident: +… MB VRAM` 이 한 번 찍힙니다.
  - "Anima 3.8B: 커넥터의 timestep 무관 계산을 프롬프트 줄마다 한 번만 (VRAM 줄당 약 15~70 MB)" (`sam3_anima38_connector_run_cache`):
    의미 특징·source 의 K/V 와 첫 블록을 첫 스텝에 계산해 다시 씁니다(합계 최대 512 MB, 생성이 끝나면 버림). llm_adapter LoRA·번들이 바뀌면 다시 계산하고,
    커넥터가 TE 모듈을 같이 쓰는 폴백에서는 쓰지 않습니다.
  - GPU 시간(사용자 3.8B 설정, 부정 커넥터 켬, 두 번째 렌더): 둘 다 끔 44.3 초 / fp32 만 42.2 초 / 캐시만 43.3 초 / 둘 다 켬 41.7 초(참고: 커넥터
    Bypass 40.2 초).
- **이 릴리즈의 성능 작업 전체**: 사용자 3.8B 설정 한 장이 47.9 → 41.7 초가 됐고, 기준 이미지(DoRA LyCORIS, 고정 시드)를 다시 렌더하면 픽셀 단위로 같습니다. 커넥터
  fp32 상주·run 캐시와 VRAM 상주 외에 v2 run 텐서를 첫 스텝에만 올려 재사용하고 forward 당 GPU→CPU 동기화를 5~7회에서 1회로 줄였으며, Qwen3.5 의미 특징
  캐시를 bf16 그대로 저장해 RAM 을 절반으로 줄였습니다.
- **SAM3**: Detect/Exclude 토큰이 여러 개여도 이미지 백본은 이미지당 한 번만 돕니다. Mask Hull·Mask Outline Px (edge-aware)·작은 반경의 Mask
  Dilation 이 픽셀은 그대로 빨라졌습니다(성분이 많은 1536² Hull 1.1 초 → 약 0.006 초). 인페인트 패스마다 세 번 하던 `synchronize`+`empty_cache` 를
  한 번으로 줄였고, 아티팩트 overlay PNG 는 압축 레벨 2 로 저장합니다(픽셀 같음, 파일 약 9% 큼, 저장 약 0.27 → 0.11 초).
- **가이던스**: PAG/SEG 의 행·head 인덱스 텐서를 캐시합니다. Skimmed CFG 의 스텝당 GPU→CPU 동기화를 없앴던 변경은 원본 수식을 그대로 편입하면서 되돌렸습니다(원본과
  같은 불리언 마스크 인덱싱). 진단(guidance diagnostics)을 끄면 fit_error 와 PAG/SEG rel_delta 를 패스의 첫 스텝에서만 잽니다('비선형 CFG' 경고도 첫
  스텝 기준, `fit_error=?` 로 찍힐 수 있음). 켜면 예전처럼 매 스텝 잽니다.

### 설치 · 라이선스 · 개발

- **첫 설치**: `install.py` 가 `requirements.txt` 에서 **빠진 패키지만** Forge venv 에 설치합니다(예전에는 경고만). 이미 깔린 패키지는 범위 밖이어도
  알리기만 하고 torch 계열은 설치하지 않습니다(`requirements.txt` 에서 torch·py-cpuinfo·protobuf 를 뺌). Anima Tile-Repair 벤더가 불러올 때
  쓰는 `imagesize` 는 `requirements.txt` 에 넣어(벤더와 같은 `==1.4.1`) 없으면 함께 설치합니다(예전에는 시작 로그에 설치 안내만).
  `--sam3-no-auto-install`(COMMANDLINE_ARGS) 또는 `SAM3_NO_AUTO_INSTALL=1` 이면 빠진 패키지와 pip 명령만 출력합니다(LoRA Manager
  경량 deps 도 같음). 진단은 Forge 콘솔(`[forge_sam3_extension]`)에 보입니다. 로컬에 `sam3.pt` 가 없으면 Hugging Face `facebook/sam3`
  에서 받아 씁니다(예전에는 FileNotFoundError). `--sam3-no-huggingface` 면 넣을 위치와 플래그를 안내하는 FileNotFoundError 입니다. 확장 폴더명이
  `forge_sam3_extension` 이 아니면 시작 로그에 경고합니다(`metadata.ini` 콜백 순서가 폴더명에 묶임). LoRA Manager vendor 는 새로 clone 할 때
  커밋 `303cca0`(1.2.0)으로 고정하고 판이 다르면 알립니다.
- **라이선스를 GPL-3.0-only 로**: 이 확장의 코드를 GNU GPL 3판만(SPDX `GPL-3.0-only`, 'or later' 없음)으로 배포합니다. 루트 `LICENSE` 를 추가하고
  README 라이선스 절·`THIRD_PARTY_NOTICES.md` 에 반영했으며, `package.json` 에 SPDX `GPL-3.0-only AND AGPL-3.0-or-later`(편입한 NegPiP 몫)를 적고
  README 의 '내부 사용' 문구를 뺐습니다. 원본 동등성 작업으로 편입한 상류 코드 — Skimmed CFG 수식(Apache-2.0), Detail Daemon 스케줄·σ 조회와 Safe PAG 적용 구간·번호 파싱, DAVE
  초반 스텝 게이트(MIT), CNS 재색칠(GPL-3.0) — 와 대조 테스트용 ComfyUI-DCW 원본 사본(GPL-3.0)의 출처·커밋·고지를 `THIRD_PARTY_NOTICES.md` 에
  적었습니다. 편입한 NegPiP(`sam3ext/negpip/` 의 상류 파일, `scripts/negpip.py`)는 **AGPL-3.0-or-later** 를 그대로 따르며(확장 전체가 GPL-3.0-only
  인 것이 아님) 라이선스 전문을 코드 옆에 두었습니다. GPL-3.0 13조로 결합해 배포하므로, 결합된 작업을 네트워크 너머 사용자에게 서비스하면 AGPL-3.0
  13조의 네트워크 상호작용 요건(원격 사용자에게 대응 소스를 받을 기회 제공)이 NegPiP 부분만이 아니라 결합된 작업 그 자체에 적용됩니다.
- **새 설정 섹션**: **SAM Extra SAM3**(`sam3_mask`)·**SAM Extra Anima 3.8B**(`sam3_anima38`)·**SAM Extra LoRA**
  (`sam3_lora`)·**SAM Extra Guidance**(`sam3_guidance`)·**SAM Extra Character Reference**(`sam3_reference`)·
  **SAM Extra VAE DeGrid**(`sam3_degrid`) 가
  생겼습니다. README·docs 의 코드 불일치는 문서 쪽만 고쳤습니다.
- **개발**: `requirements-dev.txt` 를 추가했고 CI 는 Python 3.13 + `unittest discover` 로 바뀌었습니다(실제 GitHub Actions 실행은
  미확인). 테스트 사이에 가짜 모듈이 남아 실행 순서에 따라 결과가 흔들리던 문제를 고쳤습니다.
- **검증**: Python 단위 테스트 1437개 통과(skipped 2, CPU, `python -m unittest discover -s tests -t .`), jsdom 프런트엔드 테스트
  57개 통과(`npm test`). GPU(RTX 5090) 확인 — 픽셀 동일: 성능 작업 뒤 3.8B 기준 이미지 재렌더, 3.8B 커넥터 fp32·run 캐시, TF32 수정, SAM3
  인페인트 시드 수정, IP-Adapter 잡 뒤의 txt2img. 구도 같고 잔 디테일만 다름: PAG 앞쪽 블록 중복 제거(56.2 → 51.4 초), SEG separable blur.
  비교·동작 확인: IP-Adapter lineage 정책, IP-Adapter 의 3.8B 커넥터. 이 GPU 측정은 모두 원본 동등성 작업 전의 코드에서 잰 것입니다. 아직 GPU 로 확인하지
  않은 것: Safe PAG CFG 1 변경, VAE 2x, 부분 LoRA 추측 변환, 이어붙이기 권장 LoRA 화질 A/B, 그리고 원본 동등성 작업 전체(Detail Daemon·Safe
  PAG·Skimmed CFG·DCW(+a)·RDC·DAVE·CNS·Tile-Repair·SAM3 LLLite 전처리) — 이것은 원본 코드를 테스트 안에 그대로 두고 같은 입력의 결과를 비교하는
  CPU 단위 테스트(`tests/test_*_origin.py`, `test_skimmed_cfg.py`)로만 확인했습니다. Tile & Repair API·Notebook
  메모장·`--api-auth` 적용·LoRA Manager 경로 인증·ui-config 이전·라이트박스 수정도 단위 테스트(메모장은 jsdom 포함)로 확인했습니다. 그 밖에 GPU
  결과를 적지 않은 항목은 CPU 단위 테스트로만 확인했습니다.

## v0.21.2 — 레거시 콘솔 인코딩에서 로그가 생성을 죽이던 문제 + CI 연결

- **로그 한 줄이 샘플링을 중단시킬 수 있던 문제 수정**: `_log`가 em-dash나 `✅` 같은
  비-ASCII 문자를 그대로 `print`하는데, 레거시 코드 페이지(cp949, cp1252 등) 콘솔에서는
  `UnicodeEncodeError`가 발생합니다. 이 함수는 `_post_cfg`와 model wrapper **안에서**,
  심지어 그들의 `except` 핸들러에서도 호출되므로, 예외가 그대로 전파되어 처리된 폴백이
  하드 실패로 바뀌거나 생성이 죽을 수 있었습니다. `anima_skimmed_cfg.py`와
  `anima_safe_pag.py`의 `_log`가 인코딩 실패 시 ASCII 대체 표현으로 저하되고, 어떤 경우에도
  예외를 밖으로 내보내지 않도록 했습니다. 여러 릴리즈 전부터 있던 선재 결함입니다.
  `PYTHONIOENCODING=cp949`에서 전체 스위트가 통과하는 것과, cp949 stdout을 재현해 훅이
  incoming 결과를 그대로 반환하는지 확인하는 회귀 테스트를 추가했습니다.
- **CI에 프런트엔드 동작 테스트 연결**: v0.21.1에서 푸시 토큰의 `workflow` 스코프가 없어
  빠졌던 `.github/workflows/ci.yml` 변경(`npm ci` + `npm test`)이 이제 포함됩니다.
- **검증**: Python 149개(cp949 포함) + jsdom 6개 통과, `node --check` 전체 통과.

## v0.21.1 — 테마 라이트모드 수정 + 대비 실측 + 프런트엔드 동작 테스트

v0.21.0에서 미뤄둔 접근성·테스트 항목을 처리한 릴리즈. 기능 변경은 없습니다.

- **커스텀 테마가 Gradio `dark` 클래스도 설정**: 모든 팔레트가 어두운데 Forge/Gradio CSS
  상당량이 `document.body`의 `dark` 클래스에 걸려 있고, Gradio는 해석된 테마가 dark일 때만
  이를 붙입니다(기본값은 `__theme=system`). 라이트 모드 세션에서 흰 본문 글자가 라이트 전용
  표면 위에 얹히던 문제(토스트 배경, 코드 하이라이트, `ul.options li.selected`)를 없앴습니다.
  우리가 추가한 경우만 되돌리도록 추적해 Gradio 소유의 `dark`는 건드리지 않습니다.
- **대비를 실측해 정정**: 이전 헤더 주석은 `contrast: pass`로 단정했지만 실제로는 취소 버튼
  레이블이 3.98:1, 컨트롤 테두리가 1.36:1이었습니다. tokens.css의 OKLCH 값을 sRGB로 변환해
  계산한 결과로 교체했습니다.
  - `--sam3-color-error-ink` 96% → 99%, 두 팔레트의 `--sam3-color-error` 59% → 58%로
    취소 버튼 레이블을 **4.50~4.52:1**(WCAG 1.4.3 통과)로 올렸습니다.
  - fast 드롭다운의 오류 표시가 채움용 `--sam3-color-error`를 텍스트 색으로 쓰던 것을
    팔레트의 오류 텍스트 값 `--sam3-color-error-hover`(5.12~5.58:1)로 바로잡았습니다.
  - **알려진 미해결**: `--sam3-color-rule`은 표면 대비 1.36~1.58:1로 컨트롤 경계 기준
    3:1(WCAG 1.4.11)에 미달합니다. 입력 채움이 블록 배경과 1.03~1.07:1이라 테두리가 사실상
    유일한 식별 수단이므로 면제도 성립하지 않습니다. 해소에는 밝기 30% → 약 48% 상향이
    필요해 팔레트의 의도적 시각 변경에 해당하므로 이 릴리즈에서는 문서화만 했습니다.
  - 본문·muted·오류 텍스트·취소 레이블 대비를 **테스트에서 직접 계산**해 고정했습니다.
    주석이 아니라 계산이 지키므로 근거 없는 재인증이 불가능합니다.
- **jsdom 프런트엔드 동작 테스트 도입**: 문자열 검사는 "리스너가 엉뚱한 노드에 붙었다"나
  "패널이 안 닫힌다"를 잡지 못합니다. `package.json`(test 전용, 배포·번들 대상 아님)에
  jsdom을 추가하고 Forge extra-networks 탭 구조를 재현한 DOM에 실제 스크립트를 올려
  **클릭으로** 검증합니다 — Manage 주입, 활성화, native 탭 첫 클릭 닫힘, Gradio가 아무것도
  하지 않는 복귀 경로, 늦게 도착하는 선택 상태(옵저버), 탭 스트립 재생성 후에도 유지.
  위임 리스너를 끄면 6개 중 5개가 실패하는 것으로 실제 회귀 검출력을 확인했습니다.
  `node_modules/`는 제외하고 `package-lock.json`은 추적합니다.
  **CI 연결은 이 릴리즈에 포함되지 않았습니다** — `.github/workflows/ci.yml`에 `npm ci` +
  `npm test` 단계를 추가하는 변경은 푸시 토큰에 `workflow` 스코프가 없어 커밋하지 못했습니다.
  변경 내용은 작업 트리에 남아 있으며, `gh auth refresh -s workflow` 후 커밋하면 됩니다.
- **검증**: Python 148개 + jsdom 6개 통과, `node --check` 전체 통과.

## v0.21.0 — Live Workspaces 폐기 + txt2img Notebook

iframe 기반 Live Workspaces를 걷어내고 단일 Forge 문서 + 서버 저장 Notebook 프리셋으로
교체한 릴리즈. 선택형 다크 테마와 SMC upstream 프리셋도 포함합니다.

- **선택형 Forge 전역 다크 테마**: `Settings → SAM Extra Appearance`에
  `Forge Default / Graphite Ember / Obsidian Violet / Warm Espresso / OLED Mono`를
  추가했습니다. Forge 본체를 수정하지 않고 Gradio 색상 토큰을 확장 CSS로 연결하며,
  Settings 저장 직후 현재 페이지에 적용되고 Forge 옵션 저장소에 유지됩니다. 커스텀 테마는
  단색 표면·즉시 포커스 링·최소 모션·공통 상태 규칙을 공유하고, `Forge Default`는 전역
  덮어쓰기를 완전히 제거합니다.
- **단일 Forge 문서로 복귀**: `/sam3-live` 셸, iframe, 자동 리다이렉트, Workspace 스냅샷과
  관련 JavaScript/Python 라우트를 제거. 원래 Forge 루트 주소와 생성 큐·진행 미리보기를 그대로
  사용합니다.
- **Notebook 프리셋**: Gallery 아래에서 프리셋 추가·이름 변경·삭제·검색·내보내기/불러오기를
  지원. 프리셋 하나에 프롬프트, 네거티브, LoRA, XYZ X/Y/Z의 축 유형+값, 주요 생성 설정을
  여러 항목으로 묶어 현재 화면에 적용할 수 있습니다.
- **안전한 영구 저장**: Forge data 경로의 `sam-extra/notebook.json`에 revision 기반으로
  원자 저장하고 직전 세대 `.bak`, 저장 상태, 충돌 방지, 직전 적용 되돌리기를 추가했습니다.
  Forge/Gradio 로그인 가드와 same-origin 읽기·저장 헤더를 적용하고, 동기 파일 I/O는 작업
  스레드로 분리했습니다. 원본·백업이 모두 손상되면 빈 데이터로 덮어쓰지 않으며 저장소 I/O
  오류도 경로를 노출하지 않는 제한된 500 응답으로 처리합니다.
- **적용 안전성과 이전 데이터 이식**: 프리셋 전체를 사전 검사하고 도중 실패 시 이미 바뀐 값을
  원자적으로 복구합니다. Undo는 XYZ가 강제로 바꾸는 Script 선택·CSV 모드까지 되돌립니다.
  기존 `sam-extra.workspace-manager.v1` 데이터가 있으면 명시적 가져오기 버튼으로 지원 항목을
  Notebook 프리셋에 추가하며 원본 브라우저 데이터는 보존합니다. 프롬프트 안 LoRA 태그의
  위치를 유지하고 지원 밖 컨트롤·빈 Workspace·한도 초과분은 가져오기 전에 개수로 알립니다.
- **편한 3열 UI 유지**: Parameters / Scripts / Gallery 재배치는 유지하되 always-on 확장은
  Parameters 아래에 두고 Script 선택기와 내장 Script 패널만 가운데에 둡니다. Forge의
  Default/Compact 프롬프트 레이아웃을 모두 감지하며 Compact의 붙여넣기·지우기·스타일·토큰
  도구 행도 프롬프트와 함께 옮깁니다. 기존 레이아웃만 남은 재마운트에서도 Notebook 패널을
  Gallery 아래에 다시 연결합니다.
- **Forge 원본 탭 복구**: settings/gallery를 분리한 뒤 `#txt2img_extra_tabs`를
  Prompt/Negative 아래로 옮겨 Generation / Textual Inversion / Checkpoints / Lora 탭을
  계속 사용할 수 있게 했습니다. 재마운트 이중 이동과 숨은 탭 회귀도 테스트합니다.
- **LoRA Manager 단순화**: Workspace 자식 감지와 셸 메시지 중계를 제거하고 단일 Forge 페이지에
  Manage 탭과 프롬프트 삽입 브리지를 한 번 주입합니다.
- **SMC upstream 프리셋 이식**: ComfyUI-DCW의 `Off / Auto / 모델별 / Custom` 선택과
  모델 감지 규칙을 추가했습니다. Forge `Anima`는 `Cosmos / Wan`으로 자동 판별되어
  `lambda=6.0, k=0.20`이 적용되며, Custom 범위도 upstream과 맞췄습니다. 기존 SMC 체크박스,
  API/infotext 인자와 XYZ 축은 호환 영역에 유지하고 새 preset 인자는 맨 뒤에 추가했습니다.
  새 Custom UI 기본 `k=0.10`과 별개로, 구버전 호출이 기존 index 27을 생략한 경우에는 역사적
  폴백 `k=0.20`을 유지합니다.
- **Guidance 패널 순서**: 실행 수학은 upstream 호환인 `SMC → APG → CWM → DCW`를
  유지하면서 화면 순서는 `DCW → CWM → SMC`로 정리했습니다. XYZ 축 목록은 **정수 인덱스가
  저장되는 호환 표면**이므로 v0.20.0의 46개 순서를 그대로 유지하고 `[Anima SMC] Preset`만
  맨 뒤에 append했습니다. 축 순서를 고정하는 회귀 테스트를 추가했습니다.
- **드롭다운 백엔드 이중 실행 수정**: 값을 쓰면 Gradio가 이미 change+input을 발생시키므로
  (`handle_change`, `value_is_output`은 서버 출력에서만 참), 수동 dispatch를 값이 실제로
  바뀌지 않은 경우로 제한했습니다. 체크포인트 선택이 모델을 두 번 로드하던 문제가 사라집니다.
  multiselect는 검색 입력이 항상 비어 있어 `.token`에서 선택값을 읽습니다(VAE/Text Encoder).
- **fast 드롭다운 선택지 갱신**: 프록시가 실제 컨트롤을 숨기므로 설치 시점 목록에 고정되던
  문제를 수정했습니다. `window.gradio_config`가 실시간 갱신되는 점을 이용해 보관한 배열
  참조만 O(1)로 비교하고 달라졌을 때만 다시 매핑합니다. 빈 목록으로의 갱신도 반영합니다.
  외부 주입 목록(`/sdapi/v1` 모델 드롭다운)만 이 경로에서 제외됩니다.
- **LoRA Manager 탭이 닫히지 않던 문제 수정**: Gradio 4.40은 비선택 TabItem을 DOM에 두고
  인라인 `display`만 바꾸므로 합성 Manage 패널은 이 확장이 직접 숨겨야 합니다. 주입 시점
  노드를 캡처하는 방식을 버리고 컨테이너 위임 리스너 + 클릭 시점 노드 조회로 바꿨고, 다른
  탭 클릭 시 Gradio flush를 기다리지 않고 즉시 닫습니다. 순서를 통제할 수 없는 경우를 위해
  컨테이너 MutationObserver를 안전망으로 두었습니다.
- **스크립트 float에서 `z-index` 제거**: 정수 `z-index`는 stacking context를 만들어
  `position:fixed`인 Gradio 팝업까지 그 레벨에 갇히게 합니다. DOM 순서만으로 충분합니다.
- **`.gitignore`**: 어시스턴트 작업 디렉터리(`.codex/`, `.hallmark/`, `.agents/`)와
  `.claude/anima_*.json`을 제외해 배포 저장소에 섞이지 않게 했습니다.
- **검증**: 회귀 테스트 **146개** 통과, `node --check`·`py_compile`·제어문자 검사 통과,
  삭제된 모듈에 대한 dangling 참조 0개. 브라우저 실사용 확인(모델 로드 1회, Manage 탭 닫힘)은
  이 환경에서 완료하지 못했습니다 — 재시작 + 하드 리프레시 후 확인이 필요합니다.

## v0.20.0 — Skimmed/PAG 순서 복구 + Anima CLIP Modulation

- **Skimmed CFG가 PAG 계열을 지우던 실제 원인 수정**: Forge Neo의
  `ScriptRunner.process_before_every_sampling`이 두 번 정의돼 마지막 비정렬 구현이
  `sorting_priority`를 무시하는 문제를 확장 내부에서 우회. Skimmed post-CFG callback을
  owner tag로 dedupe한 뒤 목록 맨 앞에 prepend해 실제 순서를 항상
  `Skimmed → Safe PAG`로 고정했습니다. Forge 코어 파일은 수정하지 않습니다.
- **PAG scale 회귀 검증 강화**: 종전 `PAG → Skimmed` 순서는 scale 0/8 출력이 bitwise
  동일해지는 반면, 수정된 `Skimmed → PAG` 순서는 non-zero RMS 차이가 생기는 것을 실제
  post-CFG 체인 테스트로 고정. v0.19.1의 `cond_raw` 격리도 유지해 skim된 cond 오염을
  다시 막습니다.
- **Anima Modulation Guidance 추가**: 별도 768차원 CLIP-L의
  `base + w × (positive − negative)` 방향을 공개 Cosmos 어댑터 MLP/scales로 투영해
  선택한 Forge Anima block의 `adaln_lora_B_T_3D`에 가산합니다. 메인 Qwen conditioning은
  교체하지 않으며 기본 OFF입니다.
- **CLIP/어댑터 자산 경로**: `models/text_encoder`의 safetensors 헤더를 검사해 실제
  CLIP-L만 드롭다운에 표시하고 Anzhc Anime encoder를 우선합니다. 공식
  `yresearch/cosmos-pooled/checkpoint_4000.pt`는 첫 사용 시
  `models/anima_modulation_guidance/`로 내려받아 크기와 SHA-256을 검증합니다.
  local adapter는 `weights_only=True`로 필요한 tensor만 읽습니다.
- **Forge 전용 호환 계층**: ComfyUI Cosmos와 달리 Forge `Anima`에는
  `model_channels` 속성이 없으므로 실제 `Block.x_dim`에서 2048 폭을 추론합니다.
  CLIP/adapter 투영은 generation당 CPU에서 한 번만 수행하고 sampler 중에는 작은
  per-block vector만 model device/dtype으로 캐시합니다.
- **UI/XYZ/기록**: CLIP 모델, direction weight, block 범위, base/positive/negative prompt,
  adapter source를 UI에 추가하고 `[Anima Mod] Enable/Weight/Start/End` XYZ 축,
  PNG infotext와 opt-in verification block-hit 로그를 연결했습니다.
- **검증**: 회귀 테스트 **103개** 통과, Gradio 4.40에서 script argument 56개 UI 생성 확인.
  실제 로컬 권장 CLIP-L + 공식 어댑터로 `(28, 6144)` finite modulation 투영까지 확인했습니다.
  실제 checkpoint 이미지 품질 A/B는 아직 별도 검증 범위입니다.

## v0.19.1 — PAG cond 오염 수정 + Skimmed CFG XYZ

- **PAG/SEG 보정항 오염 수정**: `_apply_perturbation`이 weak 예측을 만들 때 사용한 원본
  cond(`_STATE["cond_raw"]`)와 차분하도록 변경. Skimmed CFG처럼 우리보다 먼저 도는
  post-CFG 훅이 Forge의 `cond_denoised`를 제자리에서 덮어쓰면, 덮어쓰인 cond와 차분하면서
  상수 오프셋이 섞여 scale·strength 반응이 둔해지던 문제를 제거. 캡처 텐서가 없거나 형상이
  다르면 기존 `args["cond_denoised"]` 경로로 폴백. 회귀 테스트 2개 추가.
- **Skimmed CFG XYZ 축 추가**: `[Anima Skim] Enable / Skimming CFG / Full Skim Negative /
  Disable Flipping Filter / Start / End / Flip At` 7종을 xyz_grid에 등록.

## v0.19.0 — LoRA 벌크/컨텍스트 전송을 Forge로 (ComfyUI 하드코딩 제거)

멀티 선택 후 우클릭 "한번에 넣기"가 ComfyUI로만 가서 못 쓰던 문제를 해결. 매니저의 모든
"send to workflow" 경로를 Forge 프롬프트 삽입으로 연동.

- **컨텍스트 메뉴 인터셉트**: 주입 브리지(`forge_bridge.js`)가 이제 카드 1개의 paper-plane뿐
  아니라 **단일 컨텍스트 메뉴**(`#loraContextMenu`의 `sendappend`/`sendreplace`)와
  **멀티 선택 벌크 서브메뉴**(`#bulkContextMenu`의 `send-to-workflow-append/replace`, 대상은
  `.model-card.selected` 전체)를 capture 단계에서 가로채 ComfyUI 전송을 막고 Forge로 postMessage.
  대상 카드의 `data-file_name`/`data-folder`/`data-usage_tips`로 `<lora:...>` 문법을 만들어
  전송(벌크는 콤마 결합).
- **Append/Replace 지원**: 메시지에 `replace` 플래그를 실어, Replace면 프롬프트의 기존
  `<lora:...>` 토큰을 제거한 뒤 새 세트를 넣음(Append는 기존대로 이어붙임). Live 셸도 이
  플래그를 활성 워크스페이스로 그대로 전달.
- **라벨 정리**: 벌크 서브메뉴 라벨(`loras.bulkOperations.*`)도 "Add LoRA"로 리브랜드 대상에 추가.
- **적용 시점**: 브리지는 서버 spawn 시 벤더 트리에 재기록(content 기준 idempotent)되므로,
  업데이트 후 다음 매니저 기동부터 자동 반영.
- **검증**: 회귀 테스트 88개 전부 통과(브리지 인터셉트·Forge측 replace 처리 자산 검사 포함).
  실제 멀티 선택 전송은 브라우저+Forge에서 확인 필요.

## v0.18.0 — LoRA Manager ↔ Live Workspace 연동

LoRA 매니저를 "iframe에 얹은 외부 앱"에서 **Live Workspace-인식 통합**으로 한 단계 끌어올린
릴리즈. 벤더 앱/서버 자체는 그대로 쓰되 연동을 실제 Forge Neo 흐름에 맞춤.

- **HTTP config/spawn 엔드포인트**: `/sam3-lora/config`, `/sam3-lora/spawn`(same-origin JSON)을
  추가해 경량 Live 셸이 히든 Gradio 브리지 버튼 없이 서버를 조회·기동. 기존 `get_or_spawn`
  라이프사이클을 그대로 래핑(일반 모드용 Gradio 브리지도 같은 payload 공유). 신규 테스트 2개.
- **Live 셸 공유 오버레이**: 셸 헤더의 `LoRA` 버튼이 **매니저 하나**를 오버레이로 엶(이전엔
  워크스페이스마다 중복 주입되어 `셸→워크스페이스→매니저` 3중 iframe이었음). 벤더 미설치 시
  버튼 자동 숨김.
- **활성 워크스페이스로 삽입 라우팅**: 매니저에서 Add LoRA → 셸이 그 메시지를 **현재 활성
  워크스페이스 iframe**의 프롬프트로 postMessage 전달(cross-origin은 source+shape로 검증).
- **중복 탭 억제**: Live 워크스페이스 자식 iframe에서는 `lora_manager.js`가 Manage 탭을 주입하지
  않음(네이티브 탭·일반 모드는 기존대로 자체 탭 유지). 삽입 브리지는 항상 리슨.
- **검증**: 회귀 테스트 85개 전부 통과(신규 route·shell 연동 자산 검사 포함). 실제 매니저
  기동·삽입 동작은 브라우저+Forge에서 확인 필요(이 환경 미검증).

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
