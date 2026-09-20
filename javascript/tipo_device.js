// TIPO 프롬프트 확장 — 장치(GPU/CPU) 선택을 이 브라우저에 기억한다.
//
// Gradio 는 새로 고치면 서버 시작 때의 기본값(GPU)으로 돌아간다. 마지막으로 고른 값을 Forge 의 localSet 으로 저장했다가
// 화면이 뜨면 그 라디오를 눌러 되살린다(누르면 Gradio 의 값도 함께 바뀐다). 설정 칸은 접혀 있어도 DOM 에는 있다.
(function () {
    const KEY = "sam3_tipo_device";
    let attached = false;

    function radios() {
        return Array.from(gradioApp().querySelectorAll('#sam3_tipo_device input[type="radio"]'));
    }

    function attach() {
        if (attached) return true;
        const inputs = radios();
        if (!inputs.length) return false;
        attached = true;
        inputs.forEach((input) => {
            input.addEventListener("change", () => {
                if (input.checked) localSet(KEY, input.value);
            });
        });
        const saved = localGet(KEY, null);
        const target = inputs.find((input) => input.value === saved);
        if (target && !target.checked) target.click();
        return true;
    }

    onUiLoaded(() => {
        if (!attach()) onAfterUiUpdate(attach);
    });
})();
