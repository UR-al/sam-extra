"""SAM3 빠른 버튼(🎯) — txt2img 갤러리의 ✨(hires fix) 옆에서, 선택한 이미지에 지금 SAM3 설정을 바로 돌린다.

✨(modules/txt2img.py 의 txt2img_upscale_function)와 같은 규칙을 따른다: 지금 txt2img 화면 설정으로 processing
을 만들고(txt2img_create_processing), 시드는 그 이미지의 infotext 에서(Forge 설정 txt2img_upscale_same_seed),
결과는 Forge 설정 hires_button_gallery_insert 대로 갤러리에 넣는다. SAM3 아코디언이 꺼져 있어도 돈다 — 버튼을
누른 것 자체가 요청이다. ADetailer·하이레스 같은 다른 후처리는 돌리지 않는다.

버튼은 Forge 가 ✨ 를 만들 때(on_after_component, elem_id ``txt2img_upscale``) 같은 줄에 넣고, ✨ 의 click 이
등록된 뒤 같은 입력으로 연결한다 — txt2img 입력 목록(스크립트 인자 포함)은 확장이 직접 받을 수 없어서다.

이 모듈에는 ``from __future__ import annotations`` 를 쓰지 않는다: Gradio 는 ``gr.Request`` 주석을 보고 요청
객체를 넣어 주는데, 주석이 문자열이 되면 그 판별이 흔들린다(Forge 의 txt2img.py 도 같은 이유로 쓰지 않는다).
"""
import json
from contextlib import closing

BUTTON_ICON = "🎯"
BUTTON_ELEM_ID = "txt2img_sam3_quick"
BUTTON_TOOLTIP = (
    "SAM3 설정으로 선택한 이미지를 바로 마스킹·인페인트합니다 "
    "(✨ 처럼 지금 txt2img 설정 + 그 이미지의 시드)."
)
SAM3_SCRIPT_STEM = "!sam3"
NO_MASK_MESSAGE = "SAM3: 선택한 이미지에서 마스크를 찾지 못했습니다 (감지 프롬프트·임계값을 확인하세요)."
MASK_ONLY_MESSAGE = "SAM3: 마스크는 찾았지만 모드가 Inpaint 가 아니라 이미지는 그대로입니다."
STOPPED_MESSAGE = "SAM3: 중단했습니다."


def validate_selection(gallery, index, geninfo):
    """✨ 와 같은 거절 규칙 — 빈 갤러리, 범위 밖, 그리드·컨트롤 이미지. 문제없으면 None."""
    if not gallery:
        return "No image to process with SAM3."
    if index < 0 or index >= len(gallery):
        return f"Bad image index: {index}"
    first = geninfo.get("index_of_first_image", 0)
    count = len(geninfo.get("infotexts") or [])
    if len(gallery) > 1 and (index < first or index >= count):
        return "Unable to run SAM3 on grid or control images."
    return None


def force_enabled(args):
    """SAM3 스크립트 인자에서 Enable 만 켠다 (UI 의 dict 는 건드리지 않는다)."""
    args = tuple(args)
    if args and isinstance(args[0], bool):
        return (True, *args[1:])
    if args and isinstance(args[0], dict):
        return ({**args[0], "sam3_enable": True}, *args[1:])
    return (True, *args)


def _int(value, default):
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def seeds_from_infotext(infotext, parse):
    params = parse(infotext or "", [])
    return _int(params.get("Seed"), -1), _int(params.get("Variation seed"), -1)


def _quote(value):
    """Forge infotext_utils.quote 와 같다 — 쉼표·줄바꿈·콜론이 있으면 JSON 문자열로."""
    text = str(value)
    if "," not in text and "\n" not in text and ":" not in text:
        return text
    return json.dumps(text, ensure_ascii=False)


def compose_infotext(source_infotext, extra_params):
    """원본 이미지의 infotext 뒤에 SAM3 기록을 붙인다 (파라미터 줄이 마지막 줄이다)."""
    fields = {key: value for key, value in extra_params.items() if str(key).startswith("SAM3")}
    fields["SAM3 quick"] = True
    suffix = ", ".join(f"{key}: {_quote(value)}" for key, value in fields.items())
    source = (source_infotext or "").rstrip()
    return f"{source}, {suffix}" if source else suffix


def _mark_saved(item):
    picture = item[0] if isinstance(item, (tuple, list)) else item
    filename = getattr(picture, "filename", None)
    if filename:
        try:
            picture.already_saved_as = filename.rsplit("?", 1)[0]
        except AttributeError:
            pass


def place_result(gallery, index, geninfo, image, infotext, *, insert):
    """✨ 와 같은 배치 — insert 면 원본 뒤에 끼우고, 아니면 원본 자리를 바꾼다. (갤러리, geninfo, 선택 번호)."""
    infotexts = list(geninfo.get("infotexts") or [])
    new_gallery = []
    new_infotexts = []
    for position, item in enumerate(gallery):
        if insert or position != index:
            _mark_saved(item)   # 다시 저장하지 않게
            new_gallery.append(item)
            new_infotexts.append(infotexts[position] if position < len(infotexts) else None)
        if position == index:
            new_gallery.append(image)
            new_infotexts.append(infotext)
    return new_gallery, {**geninfo, "infotexts": new_infotexts}, index + 1 if insert else index


def apply_quick_pass_settings(p, p2, seed):
    """build_i2i 가 빠른 버튼 패스(``p._sam3_quick``)에만 거는 설정. 빠른 버튼은 process_images(p) 를 거치지
    않으니 ✨ 와 같아지려면 여기서 채워야 한다:

    - 시드: p2.script_args 를 대입할 때 Forge 의 Seed 스크립트가 p2.seed 를 UI 시드로 덮는다 — 그 이미지의 시드로.
    - txt2img Override 설정(Clip skip·VAE/TE 등): process_images(p) 가 적용하던 것 — p2 의 process_images 가
      적용하고 끝나면 되돌린다.
    - ADetailer: 버튼은 SAM3 만 요청한다 — 안쪽 인페인트 패스에서 ADetailer 가 또 돌지 않게.
    """
    p2.seed = seed
    p2.override_settings = dict(getattr(p, "override_settings", None) or {})
    p2._ad_disabled = True


def find_click_dependency(blocks, button):
    """Blocks 에 등록된 이벤트 중 이 버튼의 click — ✨ 의 입력·출력 목록을 그대로 빌려 온다."""
    fns = getattr(blocks, "fns", None)
    if not fns:
        return None
    target = getattr(button, "_id", None)
    for dependency in fns.values() if isinstance(fns, dict) else fns:
        for block_id, event in getattr(dependency, "targets", None) or ():
            if block_id == target and event == "click":
                return dependency
    return None


def _find_sam3_script(p):
    runner = getattr(p, "scripts", None)
    for script in getattr(runner, "alwayson_scripts", None) or ():
        filename = str(getattr(script, "filename", "")).replace("\\", "/").rsplit("/", 1)[-1]
        if filename.rsplit(".", 1)[0].lower() == SAM3_SCRIPT_STEM:
            return script
    return None


def run_sam3_quick(id_task, request, gallery, gallery_index, generation_info, *args):
    """✨ 의 txt2img_upscale_function 과 같은 흐름으로 SAM3 만 돌린다. 출력: 갤러리, generation_info, infotext, 로그."""
    import gradio as gr
    from modules import images, infotext_utils, scripts, sd_models, shared
    from modules.txt2img import txt2img_create_processing
    from modules.ui import plaintext_to_html

    def unchanged(message):
        # 갤러리를 그대로 돌려주면 Forge 가 모든 항목을 temp 로 다시 저장하고 그쪽을 가리키게 된다 — no-op update
        return gr.update(), gr.update(), message, ""

    geninfo = json.loads(generation_info or "{}")
    index = int(gallery_index)
    error = validate_selection(gallery, index, geninfo)
    if error:
        return unchanged(error)

    sd_models.forge_model_reload()   # 드롭다운에서 바꾼 체크포인트 — process_images 를 거치지 않으니 먼저 불러온다
    p = txt2img_create_processing(id_task, request, *args)
    p.batch_size = 1
    p.n_iter = 1
    p._sam3_quick = True   # build_i2i → apply_quick_pass_settings
    infotexts = geninfo.get("infotexts") or []
    source_infotext = infotexts[index] if index < len(infotexts) else ""
    if getattr(shared.opts, "txt2img_upscale_same_seed", True):
        p.seed, p.subseed = seeds_from_infotext(source_infotext, infotext_utils.parse_generation_parameters)

    with closing(p):
        sam3 = _find_sam3_script(p)
        if sam3 is None:
            return unchanged("SAM3 script is not loaded in txt2img.")
        image = infotext_utils.image_from_url_text(gallery[index])
        p.width, p.height = image.size   # ✨ 처럼 선택한 이미지 크기 — 통째 인페인트가 UI 크기로 줄이지 않게
        sam3_args = force_enabled(p.script_args[sam3.args_from : sam3.args_to])
        sam3.process(p, *sam3_args)
        if not (getattr(p, "_sam3_args", None) or {}).get("enabled"):
            return unchanged("SAM3 settings are invalid — check the SAM3 accordion.")
        state = shared.state
        if state.job_count < 0:
            state.job_count = 0   # state.begin() 의 -1 — 인페인트 패스 수를 더할 때 진행률이 한 칸 밀리지 않게
        pp = scripts.PostprocessImageArgs(image, index)
        sam3.postprocess_image(p, pp, *sam3_args)

    shared.total_tqdm.clear()
    if state.interrupted or state.skipped:
        return unchanged(STOPPED_MESSAGE)
    if not getattr(p, "_sam3_mask_found", False):
        return unchanged(NO_MASK_MESSAGE)   # 미리보기 오버레이가 켜져 있어도 저장하지 않는다
    result = pp.image
    if result is image:
        return unchanged(MASK_ONLY_MESSAGE)

    infotext = compose_infotext(source_infotext, p.extra_generation_params)
    if getattr(shared.opts, "samples_save", True) and not getattr(p, "do_not_save_samples", False):
        images.save_image(
            result, p.outpath_samples, "", p.seed, p.prompt, shared.opts.samples_format,
            info=infotext, p=p, suffix="-sam3",
        )
    new_gallery, geninfo, selected = place_result(
        gallery, index, geninfo, result, infotext,
        insert=bool(getattr(shared.opts, "hires_button_gallery_insert", False)),
    )
    return gr.update(value=new_gallery, selected_index=selected), json.dumps(geninfo), plaintext_to_html(infotext), ""


def wire_quick_button(button, dependency):
    """✨ 의 click 과 같은 입력·출력·JS(submit_txt2img_upscale: 진행 표시 + 선택 번호)로 연결한다."""
    import gradio as gr
    from modules.call_queue import wrap_gradio_gpu_call

    def sam3_quick(id_task: str, request: gr.Request, gallery, gallery_index, generation_info, *args):
        from modules_forge import main_thread

        return main_thread.run_and_wait_result(
            run_sam3_quick, id_task, request, gallery, gallery_index, generation_info, *args
        )

    button.click(
        fn=wrap_gradio_gpu_call(sam3_quick, extra_outputs=[None, "", ""]),
        js="submit_txt2img_upscale",
        inputs=list(dependency.inputs),
        outputs=list(dependency.outputs),
        show_progress=False,
    )
