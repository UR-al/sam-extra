# Colorcraft (sam-extra) — the Gradio panel: one collapsed accordion with the modifier stack (I-X, one
# shared editor), the masking sub-accordion (leaves M1-M10, combos C1-C5, one shared editor) and paste/reset
# wiring.
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
# checkbox (sam-extra's ``sam3-on`` convention) instead of an InputAccordion; a Type selector that shows
# only that node's controls; Pass (Base/Hires/Both) instead of the hires checkbox; the Advanced node's
# gates and dev overrides; static Mask choices (an incomplete combo runs unmasked, as upstream does) so a
# pasted selection is always a valid choice; one shared modifier editor and one leaf/combo editor picked
# by selector radios (upstream has a tab per node); the other nodes' values live in a hidden JSON state
# (panel_state.py) and the browser syncs them (javascript/colorcraft_editor.js). Upstream's canvas plots
# and tab colouring (its JavaScript) are not part of this port; the Debug panel renders into its own
# gallery.
"""Build the Colorcraft panel: 67 script arguments in ``panel_state.ARG_NAMES`` order.

ONE modifier editor (44 controls), ONE leaf editor (10) and ONE combo editor (7) show the modifier and the mask node
named by ``ref``; two selector radios pick them. Every editor control is also a script argument, and the
generation lays the editors over the hidden state (``panel_state.config_from_script_args``), so whatever the
editor shows when Generate fires is what runs - no event has to finish first.

Who writes what:
* the user: the editors (and the selectors, Enable, masking);
* js ``sync`` (selector .change, state .change): commit the editors into the state, load the new node, new rev -
  state, ref and editors in ONE update;
* js ``reset``: the node's defaults into state + editors, new rev, one update;
* js ``labels`` (every editor .input, enabled/masking .change): selector marks and the summary only;
* js ``kind`` (Type .change): visibility only;
* without colorcraft_editor.js (or when it throws), the inline fallback of ``_js``: selectors back on ``ref``'s
  nodes and ``JS_FAILED`` in the summary;
* Forge's paste (Python): enabled, masking, state, ref, selectors, summary and the modifier + leaf editors in ONE
  response;
* Python Debug Refresh: reads, writes only the gallery.
"""

from __future__ import annotations

import json
from functools import partial

from . import debug_panel, panel_state, spec

ON_CLASSES = ["sam3-on", "sam3-on--colorcraft"]

INTRO = (
    "샘플링 중 **매 스텝의 CFG 결과(x0)** 를 latent 의 색 기저 방향으로 밀어 노출·색온도·채도·대비·디테일을 "
    "조정합니다 — [ComfyUI-Colorcraft](https://github.com/muerrilla/ComfyUI-Colorcraft) (muerrilla) 의 최신 계산 "
    "그대로입니다. 추가 모델 호출은 없습니다.\n\n"
    "- **지원 모델**: Anima · Qwen-Image · Krea 2 · Wan (krea2 벡터), Flux · Chroma · Lumina 2 · Z-Image "
    "(zimage 벡터), Flux 2 Klein · ERNIE-Image (flux2 벡터, aoleg 포크). **SD 1.5 · SDXL** 은 벡터가 없어 "
    "**Contrast 와 Color Shift 만** 동작합니다 — 나머지는 꺼지고 이유가 infotext `SAM Extra Colorcraft status` 에 "
    "남습니다.\n"
    "- **수정자 I–X**: 아래 **수정자** 선택 줄에서 고른 수정자를 그 아래 칸이 편집합니다. 켠(Active) 수정자가 "
    "I 부터 차례로 적용됩니다. 선택 줄의 **●** 는 켠 수정자, **○** 는 값을 바꿨지만 꺼 둔 수정자입니다. **Type** 은 "
    "원본 노드 종류(Advanced · Basic · Luma · Chroma · Chroma Plus · Punch · Shift)이며 그 노드의 계산·순서를 그대로 씁니다. "
    "**Strength · Start · End** 는 세기와 적용 구간(스텝 비율), **Pass** 는 기본 패스 · Hires 패스 · 둘 다.\n"
    "- **Masking**: M1–M10 은 latent 자체에서 읽는 축 마스크, C1–C5 는 그 조합입니다. 수정자의 **Mask ➜** 로 "
    "고릅니다.\n"
    "- 값은 작게(0.1–0.3) 두고 구간을 넓히는 편이 안전합니다. 일찍(Start 작게) 걸면 생성 방향을 바꾸고, 늦게 걸면 "
    "색 보정처럼 동작합니다. 화질은 GPU 에서 아직 검증되지 않았습니다."
)

MASK_HELP = (
    "**축 마스크**는 매 스텝의 x0 에서 바로 읽습니다(그린 영역이 아님). highs/lows = 한쪽, range = 띠, "
    "protect range = 띠 밖, split = 부호 있는 방향. **Hue** 의 Center 는 π 단위, 나머지 축은 ±1 정규화. "
    "Blur 는 이미지 픽셀 단위이며 Spread → Normalize → Contrast 순으로 다듬습니다. 조합(C)은 퍼지 집합 "
    "연산이며 앞쪽 조합만 참조할 수 있습니다. A·B 가 다 차지 않은 조합을 고른 수정자는 마스크 없이 적용됩니다"
    "(원본과 같음). 선택 줄의 **●** 는 켠 수정자가 실제로 쓰는 마스크(A·B 가 다 찬 조합만), **○** 는 값만 바꾼 "
    "마스크입니다. 요약 줄의 **(미완성)** 은 그 조합이 다 차지 않아 마스크 없이 적용된다는 뜻입니다."
)

DEBUG_HELP = (
    "**Capture debug latent** 를 켜고 한 번 생성하면 **Debug Step** 스텝에서 편집 전 x0 를 잡아 둡니다(기본 패스·Hires "
    "패스 중 마지막에 잡힌 것, 같은 크기의 배치는 이어 붙임). 그 뒤 **Refresh Debug Images** 가 고른 축 투영과 "
    "마스크·조합을 아래 갤러리에 그립니다 — 지금 마스크 칸의 값으로 그리므로 다시 생성하지 않고 마스크를 다듬을 수 "
    "있습니다. Composite 색을 고르면 잡아 둔 latent 를 VAE 로 디코드해 그 위에 겹칩니다(GPU 사용). 벡터가 있는 모델에서만 "
    "동작합니다."
)

# Under each selector's label: what its marks mean (panel_state.overview writes them).
MOD_MARKS = "● 켬 · ○ 값을 바꿨지만 꺼 둠"
MASK_MARKS = "● 켠 수정자가 씀 · ○ 값만 바꿈"

GROUP_ORDER = panel_state.GROUP_ORDER


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
    # Per-image settings, not preferences: keep the panel's values (and their visibility) out of ui-config.json,
    # like upstream. Infotext is how a look comes back.
    comp.do_not_save_to_config = True
    return comp


def _tab(elem_id):
    """``txt2img`` or ``img2img``, from Forge's ``script_<tab>_…`` ids: colorcraft_editor.js finds this panel's
    elements by it (the fast-dropdown nudge, and the screen-reader names, descriptions and announcements)."""
    return "img2img" if str(elem_id("accordion")).startswith("script_img2img_") else "txt2img"


# Shown when javascript/colorcraft_editor.js is missing or one of its handlers throws (it then returns null): the
# selectors go back to the nodes the editors show (``ref``), so the page never claims to edit another node.
JS_FAILED = ("편집기 스크립트(javascript/colorcraft_editor.js)가 동작하지 않아 다른 항목을 고를 수 없습니다 — "
             "페이지를 새로 고치세요. 생성에는 지금 편집기에 보이는 값이 쓰입니다.")


def _js(tab, name, *extra, ref_at=None, n_out=0):
    """The js-only event: ``samextraColorcraft.<name>(tab, *extra, a)``. With ``ref_at`` (the input index of
    ``ref``) the event's last three outputs are ``[mod_select, mask_select, summary]`` and a missing or failed
    editor gets the inline fallback: every other output unchanged, the selectors on ``ref``'s nodes, ``JS_FAILED``
    in the summary."""
    args = ", ".join([repr(tab)] + [repr(e) for e in extra] + ["a"])
    call = f"window.samextraColorcraft.{name}({args})"
    if ref_at is None:
        return f"(...a) => window.samextraColorcraft ? {call} : undefined"
    note = json.dumps(f'<div class="samextra-cc-summary">{panel_state._escape(JS_FAILED)}</div>', ensure_ascii=False)
    return ("(...a) => { const r = window.samextraColorcraft ? " + call + " : null; if (r !== null) return r; "
            "console.error('[Colorcraft] javascript/colorcraft_editor.js is not loaded or failed'); "
            f"const u = {{__type__: 'update'}}, t = String(a[{ref_at}]).split('|'), "
            "v = (x) => (t.length === 3 ? {__type__: 'update', value: x} : u); "
            f"return [...Array({n_out - 3}).fill(u), v(t[0]), v(t[1]), {note}]; }}")


def build(elem_id, on_debug_refresh=None, parts=None):
    """``(components, infotext_fields)`` — components in ``panel_state.ARG_NAMES`` order.

    ``on_debug_refresh(axes, masks, combos, composite, overlay, style, *mask_values)`` renders the Debug
    gallery (``mask_values``: every leaf then every combo value of the effective config, v0.31.0 order).
    ``parts``, when a dict, receives every named component (tests)."""
    import gradio as gr

    tab = _tab(elem_id)
    mk = _component
    mod0, mask0 = spec.MODIFIER_TAGS[0], spec.MASK_TAGS[0]
    mod_choices0, mask_choices0, summary0 = panel_state.overview(spec.default_config())
    mod_ed, leaf_ed, combo_ed, groups = {}, {}, {}, {}

    with gr.Accordion("Colorcraft (latent 색 보정 · ComfyUI-Colorcraft)", open=False, elem_id=elem_id("accordion")):
        intro = gr.Markdown(INTRO)
        enabled = mk(gr, spec.GLOBAL_FIELDS[0], elem_id("enabled"), elem_classes=list(ON_CLASSES))
        summary = gr.HTML(summary0, elem_id=elem_id("summary"))
        mod_select = gr.Radio(choices=mod_choices0, value=mod0, label="수정자", info=MOD_MARKS,
                              elem_id=elem_id("modifier_select"))

        def add(name, **kw):
            mod_ed[name] = mk(gr, spec.MODIFIER_BY_NAME[name], elem_id(f"mod_{name}"),
                              value=spec.modifier_default(0, name), **kw)

        with gr.Row():
            for name in ("active", "kind", "pass", "mask"):
                add(name)
        with gr.Row():
            for name in ("strength", "start", "end"):
                add(name)
        with gr.Accordion("Advanced Schedule (곡선 모양)", open=False):
            add("advanced")
            with gr.Row():
                add("exponent")
                add("bias")
            with gr.Row():
                for name in ("start_off", "end_off", "smooth"):
                    add(name)
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
            for pair in (("vibrance", "saturation"), ("temperature", "tint"), ("chroma_contrast", "chroma_center")):
                with gr.Row():
                    for name in pair:
                        add(name)
        with gr.Accordion("Chroma Plus", open=False) as groups["chroma_plus"]:
            add("more_colors")
            for pair in (("temp_plus_tint", "temp_minus_tint"), ("lab_a", "lab_b"), ("lab_a_plus_b", "lab_a_minus_b")):
                with gr.Row():
                    for name in pair:
                        add(name)
        with gr.Accordion("Color Shift", open=False) as groups["color_shift"]:
            add("color_shift")
            with gr.Row():
                for name in ("color_shift_mode", "color_shift_amount", "color_shift_brightness"):
                    add(name)
            with gr.Row():
                for name in ("color_shift_red", "color_shift_green", "color_shift_blue"):
                    add(name)
        with gr.Accordion("Dev (모델별 보정값 덮어쓰기 · Advanced)", open=False) as groups["dev"]:
            for pair in (("recenter_override", "recenter"), ("max_chroma_override", "max_chroma"),
                         ("chroma_plane_override", "chroma_plane")):
                with gr.Row():
                    for name in pair:
                        add(name)
        mod_reset = gr.Button(f"Reset {mod0}", size="sm", elem_id=elem_id("mod_reset"))

        with gr.Accordion("Masking (latent 에서 읽는 마스크)", open=False, elem_id=elem_id("masking_panel")):
            mask_help = gr.Markdown(MASK_HELP)
            masking = mk(gr, spec.GLOBAL_FIELDS[1], elem_id("masking"))
            mask_select = gr.Radio(choices=mask_choices0, value=mask0, label="마스크 · 조합", info=MASK_MARKS,
                                   elem_id=elem_id("mask_select"))
            with gr.Group(visible=True, elem_id=elem_id("leaf_group")) as leaf_group:
                with gr.Row():
                    for column in (("mask_axis", "mask_strength", "mask_center", "blur"),
                                   ("mask_mode", "mask_hardness", "mask_width", "spread"), ("normalize", "contrast")):
                        with gr.Column(min_width=60):
                            for name in column:
                                leaf_ed[name] = mk(gr, spec.LEAF_BY_NAME[name], elem_id(f"leaf_{name}"))
            with gr.Group(visible=False, elem_id=elem_id("combo_group")) as combo_group:
                with gr.Row():
                    for name in ("mask_a", "mask_b"):
                        # every reference a combo can name; js narrows it to the shown combo's own list
                        combo_ed[name] = mk(gr, spec.COMBO_BY_NAME[name], elem_id(f"combo_{name}"),
                                            choices=spec.combo_ref_choices(spec.COMBO_COUNT))
                    for name in ("operation", "normalize"):
                        combo_ed[name] = mk(gr, spec.COMBO_BY_NAME[name], elem_id(f"combo_{name}"))
                with gr.Row():
                    for name in ("blur", "spread", "contrast"):
                        combo_ed[name] = mk(gr, spec.COMBO_BY_NAME[name], elem_id(f"combo_{name}"))
            mask_reset = gr.Button(f"Reset {mask0}", size="sm", elem_id=elem_id("mask_reset"))

        with gr.Accordion("Debug (축·마스크 미리보기)", open=False, elem_id=elem_id("debug_panel")):
            debug_help = gr.Markdown(DEBUG_HELP)
            with gr.Row():
                debug = mk(gr, spec.DEBUG_FIELDS[0], elem_id("debug"))
                debug_step = mk(gr, spec.DEBUG_FIELDS[1], elem_id("debug_step"))
            with gr.Row():
                axis_style = gr.Dropdown(choices=list(debug_panel.DEBUG_AXIS_STYLES), value="colormap",
                                         label="Axis Projection Style", elem_id=elem_id("debug_axis_style"))
                composite = gr.Dropdown(choices=["none"] + list(debug_panel.DEBUG_COMPOSITE_COLORS), value="none",
                                        label="Composite Mask Color", elem_id=elem_id("debug_composite"))
                overlay_color = gr.Dropdown(choices=list(debug_panel.DEBUG_OVERLAY_COLORS), value="white",
                                            label="Overlay Color", elem_id=elem_id("debug_overlay"))
            axes = gr.CheckboxGroup(choices=list(debug_panel.DEBUG_AXIS_OPTIONS), value=[], label="Axis Projections",
                                    elem_id=elem_id("debug_axes"))
            masks = gr.CheckboxGroup(choices=list(spec.MASK_TAGS), value=[], label="Masks",
                                     elem_id=elem_id("debug_masks"))
            combos = gr.CheckboxGroup(choices=list(spec.COMBO_TAGS), value=[], label="Combos",
                                      elem_id=elem_id("debug_combos"))
            refresh = gr.Button("Refresh Debug Images", elem_id=elem_id("debug_refresh"))
            gallery = gr.Gallery(label="Colorcraft debug", columns=4, height="auto", elem_id=elem_id("debug_gallery"))

        # Hidden carriers. Gradio 4.40 mounts visible=False blocks and sends their values as event inputs.
        state = gr.Textbox(value=panel_state.DEFAULT_STATE, visible=False, label="Colorcraft state",
                           elem_id=elem_id("state"))
        ref = gr.Textbox(value=panel_state.DEFAULT_REF, visible=False, label="Colorcraft editor", elem_id=elem_id("ref"))

    for comp in [intro, summary, mod_select, mod_reset, mask_help, mask_select, mask_reset, debug_help, axis_style,
                 composite, overlay_color, axes, masks, combos, refresh, gallery, state, ref]:
        comp.do_not_save_to_config = True

    editors = ([mod_ed[n] for n in panel_state.MOD_NAMES] + [leaf_ed[n] for n in panel_state.LEAF_NAMES]
               + [combo_ed[n] for n in panel_state.COMBO_NAMES])
    head = [enabled, masking, state, ref]
    view = [mod_select, mask_select, summary]

    # Every editor event runs in the browser (js-only, no server request); colorcraft_editor.js reads the inputs
    # and writes the outputs by position. Those ending in ``view`` get the visible fallback (``_js``).
    def on(triggers, name, inputs, outputs, *extra):
        ends_in_view = all(a is b for a, b in zip(outputs[-len(view):], view))
        ref_at = next(i for i, c in enumerate(inputs) if c is ref) if ends_in_view else None
        gr.on(triggers=triggers, fn=None, inputs=inputs, outputs=outputs,
              js=_js(tab, name, *extra, ref_at=ref_at, n_out=len(outputs)),
              queue=False, show_progress="hidden", show_api=False)

    on([mod_select.change, mask_select.change, state.change], "sync",
       [mod_select, mask_select] + head + editors,
       [state, ref] + editors + [leaf_group, combo_group, mod_reset, mask_reset] + view)
    on([mod_reset.click], "reset", head + editors, [state, ref] + editors + view, "mod")
    on([mask_reset.click], "reset", head + editors, [state, ref] + editors + view, "mask")
    on([c.input for c in editors] + [enabled.change, masking.change], "labels", head + editors, view)
    kind_outputs = [groups[g] for g in GROUP_ORDER] + [mod_ed["more_colors"], mod_ed["color_shift"], mod_ed["mask"]]
    on([mod_ed["kind"].change], "kind", [mod_ed["kind"]], kind_outputs)

    if on_debug_refresh is not None:
        def on_debug(axes_v, masks_v, combos_v, composite_v, overlay_v, style_v, en, msk, st, rf, *ed):
            # the effective config: the state with the visible editors laid over, as a generation reads it
            try:
                config = panel_state.config_from_script_args([en, msk, st, None, None, rf, *ed])
            except panel_state.RefusedArgs:
                config = spec.default_config()
            return on_debug_refresh(axes_v, masks_v, combos_v, composite_v, overlay_v, style_v,
                                    *panel_state.mask_values(config))

        refresh.click(fn=on_debug, inputs=[axes, masks, combos, composite, overlay_color, axis_style] + head + editors,
                      outputs=[gallery], show_progress=False)

    # Paste: every carrier in ONE response (Forge's paste_func returns all fields together), so the state, ref,
    # selectors and editors change at once; the selectors' .change then runs ``sync``, a no-op (ref matches).
    def paste_view(part, params):
        hit = panel_state.pasted(params)
        if hit is None:
            return None
        config, _, mod, mask, _ = hit
        mods, msks, html = panel_state.overview(config)
        if part == "mod_select":
            return gr.update(choices=mods, value=mod)
        if part == "mask_select":
            return gr.update(choices=msks, value=mask)
        if part == "summary":
            return html
        if part == "leaf_group":
            return gr.update(visible=True)
        if part == "combo_group":
            return gr.update(visible=False)
        if part == "mod_reset":
            return gr.update(value=f"Reset {mod}")
        if part == "mask_reset":
            return gr.update(value=f"Reset {mask}")
        return None

    fields = [(enabled, panel_state.paste_enabled), (masking, panel_state.paste_masking),
              (state, panel_state.paste_state), (ref, panel_state.paste_ref)]
    fields += [(c, partial(paste_view, name)) for c, name in (
        (mod_select, "mod_select"), (mask_select, "mask_select"), (summary, "summary"), (leaf_group, "leaf_group"),
        (combo_group, "combo_group"), (mod_reset, "mod_reset"), (mask_reset, "mask_reset"))]
    fields += [(mod_ed[n], partial(panel_state.paste_editor, name=f"mod.{n}")) for n in panel_state.MOD_NAMES]
    fields += [(leaf_ed[n], partial(panel_state.paste_editor, name=f"leaf.{n}")) for n in panel_state.LEAF_NAMES]

    components = [enabled, masking, state, debug, debug_step, ref] + editors
    assert len(components) == panel_state.ARG_COUNT
    if isinstance(parts, dict):
        parts.update(enabled=enabled, masking=masking, state=state, ref=ref, mod_select=mod_select,
                     mask_select=mask_select, summary=summary, mod=mod_ed, leaf=leaf_ed, combo=combo_ed, groups=groups,
                     leaf_group=leaf_group, combo_group=combo_group, mod_reset=mod_reset, mask_reset=mask_reset,
                     debug=debug, debug_step=debug_step, refresh=refresh, gallery=gallery)
    return components, fields
