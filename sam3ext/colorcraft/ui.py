# Colorcraft (sam-extra) — the Gradio panel: one collapsed accordion with the modifier stack (tabs
# I-X), the masking sub-accordion (leaves M1-M10, combos C1-C5) and paste/reset wiring.
#
# Layout after muerrilla/ComfyUI-Colorcraft@d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf:scripts/colorcraft.py
# (ui(), :414-650 — tab stack, control grouping, labels, ranges, per-tab reset, the Debug widgets).
#
# MIT License
#
# Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla)
# (The fork aoleg/ComfyUI-Colorcraft — Oleg Afonin's Forge Neo port — is under the same MIT License;
#  its LICENSE carries the upstream notice with the earlier spelling "Sahand Ahmadiantehrani".)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# sam-extra code (2026-10-03). Differences from upstream's panel: a plain accordion with an Enable
# checkbox (sam-extra's ``sam3-on`` convention) instead of an InputAccordion; a Type selector per tab
# that shows only that node's controls; Pass (Base/Hires/Both) instead of the hires checkbox; the
# Advanced node's gates and dev overrides; static Mask choices (an incomplete combo runs unmasked, as
# upstream does) so a pasted selection is always a valid choice. Upstream's canvas plots and tab colouring
# (its JavaScript) are not part of this port; the Debug panel renders into its own gallery.
"""Build the Colorcraft controls in script-argument order (``spec.arg_names()``)."""

from __future__ import annotations

from functools import partial

from . import debug_panel, spec

ON_CLASSES = ["sam3-on", "sam3-on--colorcraft"]

INTRO = (
    "샘플링 중 **매 스텝의 CFG 결과(x0)** 를 latent 의 색 기저 방향으로 밀어 노출·색온도·채도·대비·디테일을 "
    "조정합니다 — [ComfyUI-Colorcraft](https://github.com/muerrilla/ComfyUI-Colorcraft) (muerrilla) 의 최신 계산 "
    "그대로입니다. 추가 모델 호출은 없습니다.\n\n"
    "- **지원 모델**: Anima · Qwen-Image · Krea 2 · Wan (krea2 벡터), Flux · Chroma · Lumina 2 · Z-Image "
    "(zimage 벡터), Flux 2 Klein · ERNIE-Image (flux2 벡터, aoleg 포크). **SD 1.5 · SDXL** 은 벡터가 없어 "
    "**Contrast 와 Color Shift 만** 동작합니다 — 나머지는 꺼지고 이유가 infotext `SAM Extra Colorcraft status` 에 "
    "남습니다.\n"
    "- **탭 I ~ X = 수정자 스택**: 켠(Active) 탭이 위에서부터 차례로 적용됩니다. **Type** 은 원본 노드 종류"
    "(Advanced · Basic · Luma · Chroma · Chroma Plus · Punch · Shift)이며 그 노드의 계산·순서를 그대로 씁니다. "
    "**Strength · Start · End** 는 세기와 적용 구간(스텝 비율), **Pass** 는 기본 패스 · Hires 패스 · 둘 다.\n"
    "- **Masking**: M1~M10 은 latent 자체에서 읽는 축 마스크, C1~C5 는 그 조합입니다. 탭의 **Mask ➜** 로 고릅니다.\n"
    "- 값은 작게(0.1~0.3) 두고 구간을 넓히는 편이 안전합니다. 일찍(Start 작게) 걸면 생성 방향을 바꾸고, 늦게 걸면 "
    "색 보정처럼 동작합니다. 화질은 GPU 에서 아직 검증되지 않았습니다."
)


def _component(gr, f, elem_id, value=None, choices=None, visible=True, elem_classes=None):
    value = f.default if value is None else value
    if f.kind == spec.FLOAT:
        comp = gr.Slider(minimum=f.minimum, maximum=f.maximum, step=f.step, value=value, label=f.label,
                         elem_id=elem_id, min_width=60, visible=visible)
    elif f.kind == spec.BOOL:
        comp = gr.Checkbox(value=value, label=f.label, elem_id=elem_id, min_width=60, visible=visible,
                           elem_classes=elem_classes)
    elif f.name == "pass":
        comp = gr.Radio(choices=list(f.choices), value=value, label=f.label, elem_id=elem_id, min_width=60,
                        visible=visible)
    else:
        comp = gr.Dropdown(choices=list(choices if choices is not None else f.choices), value=value, label=f.label,
                           elem_id=elem_id, min_width=60, visible=visible)
    # Per-image settings, not preferences: keep ~580 values (and their visibility) out of ui-config.json,
    # like upstream. Infotext is how a look comes back.
    comp.do_not_save_to_config = True
    return comp


def _kind_updates(gr, label):
    """Visibility of a tab's groups, gates and mask selector for Type ``label``."""
    kind = spec.KIND_IDS.get(label, "advanced")
    shown = set(spec.KIND_GROUPS[kind])
    gates = set(spec.KIND_GATES.get(kind, ()))
    updates = [gr.update(visible=group in shown) for group in _GROUP_ORDER]
    updates += [gr.update(visible=gate in gates) for gate in ("more_colors", "color_shift")]
    updates.append(gr.update(visible=spec.KIND_TAKES_MASK[kind]))
    return updates


_GROUP_ORDER = ("luma", "contrast", "detail", "chroma", "chroma_plus", "color_shift", "dev")


def build(elem_id, on_debug_refresh=None):
    """``(components, infotext_fields)`` — components in ``spec.arg_names()`` order.

    ``on_debug_refresh(axes, masks, combos, composite, overlay, style, *mask_values)`` renders the Debug
    gallery (``mask_values``: every leaf then every combo control, in ``spec.arg_names()`` order)."""
    import gradio as gr

    by_name = {}

    def make(path, f, **kwargs):
        comp = _component(gr, f, elem_id(path.replace(".", "_").replace("+", "p")), **kwargs)
        by_name[path] = comp
        return comp

    with gr.Accordion("Colorcraft (latent 색 보정 · ComfyUI-Colorcraft)", open=False,
                      elem_id=elem_id("accordion")):
        gr.Markdown(INTRO)
        make("enabled", spec.GLOBAL_FIELDS[0], elem_classes=list(ON_CLASSES))

        with gr.Tabs(elem_id=elem_id("modifier_tabs")):
            for index, tag in enumerate(spec.MODIFIER_TAGS):
                with gr.Tab(tag, id=f"colorcraft_mod_{index}"):
                    _modifier_tab(gr, make, index, tag)

        with gr.Accordion("Masking (latent 에서 읽는 마스크)", open=False, elem_id=elem_id("masking")):
            gr.Markdown(
                "**축 마스크**는 매 스텝의 x0 에서 바로 읽습니다(그린 영역이 아님). highs/lows = 한쪽, range = 띠, "
                "protect range = 띠 밖, split = 부호 있는 방향. **Hue** 의 Center 는 π 단위, 나머지 축은 ±1 정규화. "
                "Blur 는 이미지 픽셀 단위이며 Spread → Normalize → Contrast 순으로 다듬습니다. 조합(C)은 퍼지 집합 "
                "연산이며 앞쪽 조합만 참조할 수 있습니다. A·B 가 다 차지 않은 조합을 고른 탭은 마스크 없이 적용됩니다"
                "(원본과 같음)."
            )
            make("masking", spec.GLOBAL_FIELDS[1])
            with gr.Tabs(elem_id=elem_id("mask_tabs")):
                for index, tag in enumerate(spec.MASK_TAGS):
                    with gr.Tab(tag, id=f"colorcraft_mask_{index}"):
                        _leaf_tab(gr, make, tag)
            with gr.Tabs(elem_id=elem_id("combo_tabs")):
                for index, tag in enumerate(spec.COMBO_TAGS):
                    with gr.Tab(tag, id=f"colorcraft_combo_{index}"):
                        _combo_tab(gr, make, index, tag)

        with gr.Accordion("Debug (축·마스크 미리보기)", open=False, elem_id=elem_id("debug")):
            _debug_panel(gr, make, elem_id, by_name, on_debug_refresh)

    components = [by_name[path] for path in spec.arg_names()]
    fields = [(by_name[path], partial(spec.paste_value, path=path)) for path in spec.arg_names()]
    return components, fields


def _modifier_tab(gr, make, index, tag):
    f = spec.MODIFIER_BY_NAME
    comps = {}

    def add(name, **kwargs):
        comps[name] = make(f"{tag}.{name}", f[name], value=spec.modifier_default(index, name), **kwargs)
        return comps[name]

    with gr.Row():
        add("active")
        add("kind")
        add("pass")
        add("mask")
    with gr.Row():
        add("strength")
        add("start")
        add("end")
    with gr.Accordion("Advanced Schedule (곡선 모양)", open=False):
        add("advanced")
        with gr.Row():
            add("exponent")
            add("bias")
        with gr.Row():
            add("start_off")
            add("end_off")
            add("smooth")
    groups = {}
    with gr.Group() as groups["luma"]:
        with gr.Row():
            add("exposure")
            add("tone_compression")
    with gr.Group() as groups["contrast"]:
        add("contrast")
    with gr.Group() as groups["detail"]:
        with gr.Row():
            add("clarity")
            add("sharpness")
    with gr.Group() as groups["chroma"]:
        with gr.Row():
            add("vibrance")
            add("saturation")
        with gr.Row():
            add("temperature")
            add("tint")
        with gr.Row():
            add("chroma_contrast")
            add("chroma_center")
    with gr.Accordion("Chroma Plus", open=False) as groups["chroma_plus"]:
        add("more_colors")
        with gr.Row():
            add("temp_plus_tint")
            add("temp_minus_tint")
        with gr.Row():
            add("lab_a")
            add("lab_b")
        with gr.Row():
            add("lab_a_plus_b")
            add("lab_a_minus_b")
    with gr.Accordion("Color Shift", open=False) as groups["color_shift"]:
        add("color_shift")
        with gr.Row():
            add("color_shift_mode")
            add("color_shift_amount")
            add("color_shift_brightness")
        with gr.Row():
            add("color_shift_red")
            add("color_shift_green")
            add("color_shift_blue")
    with gr.Accordion("Dev (모델별 보정값 덮어쓰기 · Advanced)", open=False) as groups["dev"]:
        with gr.Row():
            add("recenter_override")
            add("recenter")
        with gr.Row():
            add("max_chroma_override")
            add("max_chroma")
        with gr.Row():
            add("chroma_plane_override")
            add("chroma_plane")
    reset = gr.Button(f"Reset {tag}", size="sm")

    outputs = [groups[name] for name in _GROUP_ORDER] + [comps["more_colors"], comps["color_shift"], comps["mask"]]
    comps["kind"].change(fn=partial(_kind_updates, gr), inputs=[comps["kind"]], outputs=outputs,
                         show_progress=False)
    reset_names = [field_.name for field_ in spec.MODIFIER_FIELDS if field_.name != "active"]
    defaults = tuple(spec.modifier_default(index, name) for name in reset_names)
    reset.click(fn=lambda values=defaults: values, inputs=[], outputs=[comps[n] for n in reset_names],
                show_progress=False)


def _leaf_tab(gr, make, tag):
    comps = {}
    with gr.Row():
        with gr.Column(min_width=60):
            for name in ("mask_axis", "mask_strength", "mask_center", "blur"):
                comps[name] = make(f"{tag}.{name}", spec.LEAF_BY_NAME[name])
        with gr.Column(min_width=60):
            for name in ("mask_mode", "mask_hardness", "mask_width", "spread"):
                comps[name] = make(f"{tag}.{name}", spec.LEAF_BY_NAME[name])
        with gr.Column(min_width=60):
            for name in ("normalize", "contrast"):
                comps[name] = make(f"{tag}.{name}", spec.LEAF_BY_NAME[name])
    reset = gr.Button(f"Reset {tag}", size="sm")
    names = [f.name for f in spec.LEAF_FIELDS]
    defaults = tuple(spec.LEAF_BY_NAME[n].default for n in names)
    reset.click(fn=lambda values=defaults: values, inputs=[], outputs=[comps[n] for n in names],
                show_progress=False)


def _combo_tab(gr, make, index, tag):
    comps = {}
    refs = spec.combo_ref_choices(index)
    with gr.Row():
        comps["mask_a"] = make(f"{tag}.mask_a", spec.COMBO_BY_NAME["mask_a"], choices=refs)
        comps["mask_b"] = make(f"{tag}.mask_b", spec.COMBO_BY_NAME["mask_b"], choices=refs)
        comps["operation"] = make(f"{tag}.operation", spec.COMBO_BY_NAME["operation"])
        comps["normalize"] = make(f"{tag}.normalize", spec.COMBO_BY_NAME["normalize"])
    with gr.Row():
        for name in ("blur", "spread", "contrast"):
            comps[name] = make(f"{tag}.{name}", spec.COMBO_BY_NAME[name])
    reset = gr.Button(f"Reset {tag}", size="sm")
    names = [f.name for f in spec.COMBO_FIELDS]
    defaults = tuple(spec.COMBO_BY_NAME[n].default for n in names)
    reset.click(fn=lambda values=defaults: values, inputs=[], outputs=[comps[n] for n in names],
                show_progress=False)


DEBUG_HELP = (
    "**Capture debug latent** 를 켜고 한 번 생성하면 **Debug Step** 스텝에서 편집 전 x0 를 잡아 둡니다(기본 패스·Hires "
    "패스 중 마지막에 잡힌 것, 같은 크기의 배치는 이어 붙임). 그 뒤 **Refresh Debug Images** 가 고른 축 투영과 "
    "마스크·조합을 아래 갤러리에 그립니다 — 지금 마스크 칸의 값으로 그리므로 다시 생성하지 않고 마스크를 다듬을 수 "
    "있습니다. Composite 색을 고르면 잡아 둔 latent 를 VAE 로 디코드해 그 위에 겹칩니다(GPU 사용). 벡터가 있는 모델에서만 "
    "동작합니다."
)


def _debug_panel(gr, make, elem_id, by_name, on_debug_refresh):
    gr.Markdown(DEBUG_HELP)
    with gr.Row():
        make("debug", spec.DEBUG_FIELDS[0])
        make("debug_step", spec.DEBUG_FIELDS[1])
    panel = []
    with gr.Row():
        axis_style = gr.Dropdown(choices=list(debug_panel.DEBUG_AXIS_STYLES), value="colormap",
                                 label="Axis Projection Style", elem_id=elem_id("debug_axis_style"))
        composite = gr.Dropdown(choices=["none"] + list(debug_panel.DEBUG_COMPOSITE_COLORS), value="none",
                                label="Composite Mask Color", elem_id=elem_id("debug_composite"))
        overlay = gr.Dropdown(choices=list(debug_panel.DEBUG_OVERLAY_COLORS), value="white", label="Overlay Color",
                              elem_id=elem_id("debug_overlay"))
    axes = gr.CheckboxGroup(choices=list(debug_panel.DEBUG_AXIS_OPTIONS), value=[], label="Axis Projections",
                            elem_id=elem_id("debug_axes"))
    masks = gr.CheckboxGroup(choices=list(spec.MASK_TAGS), value=[], label="Masks", elem_id=elem_id("debug_masks"))
    combos = gr.CheckboxGroup(choices=list(spec.COMBO_TAGS), value=[], label="Combos",
                              elem_id=elem_id("debug_combos"))
    refresh = gr.Button("Refresh Debug Images", elem_id=elem_id("debug_refresh"))
    gallery = gr.Gallery(label="Colorcraft debug", columns=4, height="auto", elem_id=elem_id("debug_gallery"))
    panel += [axis_style, composite, overlay, axes, masks, combos, gallery]
    for component in panel:
        component.do_not_save_to_config = True
    if on_debug_refresh is not None:
        mask_paths = [path for path in spec.arg_names() if path.split(".", 1)[0] in spec.MASK_TAGS + spec.COMBO_TAGS]
        refresh.click(fn=on_debug_refresh,
                      inputs=[axes, masks, combos, composite, overlay, axis_style] + [by_name[p] for p in mask_paths],
                      outputs=[gallery], show_progress=False)
    return panel
