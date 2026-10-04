"""Anima Tile-Repair UI panel — sits below the SAM3 in-flight accordion.

Mirrors the layout/wiring conventions of ``ui_refine.py`` so the gallery
splice + Stop-button swap + JS shim for selected gallery index behave the
same way the user is already used to.
"""
from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any

import gradio as gr

from .panel_container import ACCORDION, panel_container

from .coerce import as_float, as_int
from .forge_exclusive import run_exclusive, stop_if_job
from .anima_core import (
    AnimaTileRepairArgs,
    anima_available,
    list_dit_choices,
    list_lllite_choices,
    list_lora_choices,
    list_te_choices,
    list_vae_choices,
    list_pid_checkpoints,
    default_lllite_choice,
    default_te_choice,
    default_vae_choice,
    run_tile_repair,
    run_pid_upscale,
)
from .ui_refine import _coerce_gallery_item_to_pil, _plaintext_to_html


# ---------------------------------------------------------------------------
# Panel dataclass — what scripts/!sam3.py needs to wire the click chain.
# ---------------------------------------------------------------------------


@dataclass
class AnimaPanel:
    container: gr.Blocks   # 아코디언(단독) 또는 칼럼(선택 이미지 탭 안)
    selected_index_state: gr.Number
    # Models
    lllite_model: gr.Dropdown
    dit_override: gr.Dropdown
    te_override: gr.Dropdown
    vae_override: gr.Dropdown
    # Prompts
    positive: gr.Textbox
    negative: gr.Textbox
    # LoRA stack (4 fixed slots)
    lora_names: list = field(default_factory=list)
    lora_weights: list = field(default_factory=list)
    # Sampler
    steps: gr.Slider = None  # type: ignore[assignment]
    cfg: gr.Slider = None  # type: ignore[assignment]
    flow_shift: gr.Slider = None  # type: ignore[assignment]
    seed: gr.Number = None  # type: ignore[assignment]
    seed_random_button: gr.Button = None  # type: ignore[assignment]
    seed_pull_button: gr.Button = None  # type: ignore[assignment]
    # Output sizing — short side only; the long side follows the source
    # aspect ratio (anima_core.tile_repair_size)
    short_side: gr.Slider = None  # type: ignore[assignment]
    # LLLite — only the multiplier reaches the vendor (audit M14 removed the
    # Strength / Start % / End % sliders that were never passed anywhere)
    lllite_multiplier: gr.Slider = None  # type: ignore[assignment]
    # Housekeeping
    unload_forge_before: gr.Checkbox = None  # type: ignore[assignment]
    insert_mode: gr.Radio = None  # type: ignore[assignment]
    # Restoration mode + PiD (Pixel Diffusion Decoder) option
    restore_mode: gr.Radio = None  # type: ignore[assignment]
    pid_checkpoint: gr.Dropdown = None  # type: ignore[assignment]
    pid_scale: gr.Slider = None  # type: ignore[assignment]
    pid_steps: gr.Slider = None  # type: ignore[assignment]
    pid_degrade: gr.Slider = None  # type: ignore[assignment]
    # Run / Stop
    repair_button: gr.Button = None  # type: ignore[assignment]
    stop_button: gr.Button = None  # type: ignore[assignment]
    status: gr.HTML = None  # type: ignore[assignment]

    def all_widgets(self) -> list:
        """Ordered list of input widgets — must match ``ANIMA_ARG_KEYS``."""
        out = [
            self.lllite_model,
            self.dit_override,
            self.te_override,
            self.vae_override,
            self.positive,
            self.negative,
        ]
        for name, weight in zip(self.lora_names, self.lora_weights):
            out.append(name)
            out.append(weight)
        out.extend([
            self.steps,
            self.cfg,
            self.flow_shift,
            self.seed,
            self.short_side,
            self.lllite_multiplier,
            self.unload_forge_before,
            self.insert_mode,
            # PiD option (appended last to keep ANIMA_ARG_KEYS prefix stable)
            self.restore_mode,
            self.pid_checkpoint,
            self.pid_scale,
            self.pid_steps,
            self.pid_degrade,
        ])
        return out


ANIMA_ARG_KEYS: tuple[str, ...] = (
    "lllite_model",
    "dit_override",
    "te_override",
    "vae_override",
    "positive",
    "negative",
    "lora_name_0", "lora_weight_0",
    "lora_name_1", "lora_weight_1",
    "lora_name_2", "lora_weight_2",
    "lora_name_3", "lora_weight_3",
    "steps",
    "cfg",
    "flow_shift",
    "seed",
    "short_side",
    "lllite_multiplier",
    "unload_forge_before",
    "insert_mode",
    "restore_mode",
    "pid_checkpoint",
    "pid_scale",
    "pid_steps",
    "pid_degrade",
)


# Numeric coercion helpers — same defensive layer Refine uses (sam3ext.coerce).
_as_float = as_float
_as_int = as_int


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------


def build_anima_panel(*, container: str = ACCORDION) -> AnimaPanel:
    """Render the Anima Tile-Repair accordion. Must be called inside an open
    ``gr.Blocks`` context that is a sibling of ``txt2img_gallery``.

    Returns ``None`` would have been an option but the on_after_component
    wiring already guards with ``anima_available()``, so by the time we get
    here the vendor is present (or the panel renders anyway as scaffolding
    and just shows a no-op error banner on click).
    """
    lllite_choices = list_lllite_choices()
    dit_choices = list_dit_choices()
    te_choices = list_te_choices()
    vae_choices = list_vae_choices()
    lora_choices = list_lora_choices()

    with panel_container(
        container, "SAM3 — Anima Tile-Repair (post-generation)", "sam3_anima_panel"
    ) as acc:
        # Hidden Number — JS shim writes the gallery selection here.
        # gr.State would shift positional args, so we use Number(visible=False)
        # exactly like ui_refine.
        selected_index_state = gr.Number(
            value=-1, precision=0, visible=False, elem_id="sam3_anima_selected_index"
        )

        gr.Markdown(
            "Pick an image in the gallery above, then click **▶ Anima Tile-Repair**. "
            "The vendor's Anima DiT runs with the ControlNet-LLLite you select; the "
            "source image is fed as the LLLite control signal (same mechanism as the "
            "ComfyUI `AnimaLLLiteApply` node). Result inserts next to the selected "
            "image. **First Tile-Repair click can take \\~20-40 s** because the Anima DiT/VAE/TE "
            "weights load from disk."
        )

        # --- Models -----------------------------------------------------
        with gr.Row():
            lllite_model = gr.Dropdown(
                label="SAM3 Anima LLLite Model",
                choices=lllite_choices,
                # Only 3-channel LLLites are listed (header check); default is
                # the newest Tile & Repair file (v20 over v10).
                value=default_lllite_choice(lllite_choices),
                type="value",
                elem_id="sam3_anima_lllite",
            )
            dit_override = gr.Dropdown(
                label="SAM3 Anima DiT Override",
                choices=dit_choices,
                value="Use Forge current",
                type="value",
                elem_id="sam3_anima_dit",
            )
        with gr.Row():
            te_override = gr.Dropdown(
                label="SAM3 Anima Text Encoder Override",
                choices=te_choices,
                value=default_te_choice(te_choices),
                type="value",
                elem_id="sam3_anima_te",
            )
            vae_override = gr.Dropdown(
                label="SAM3 Anima VAE Override",
                choices=vae_choices,
                value=default_vae_choice(vae_choices),
                type="value",
                elem_id="sam3_anima_vae",
            )

        # --- Prompts ----------------------------------------------------
        with gr.Row():
            positive = gr.Textbox(
                label="SAM3 Anima Prompt",
                value="repair the low-quality anime image, reduce blur and "
                      "compression artifacts, preserve the original composition",
                lines=2,
                elem_id="sam3_anima_positive",
            )
        with gr.Row():
            # sd-scripts --negative_prompt default is "".
            negative = gr.Textbox(
                label="SAM3 Anima Negative",
                value="",
                lines=1,
                elem_id="sam3_anima_negative",
            )

        # --- LoRA stack -------------------------------------------------
        # 4 slots, default bypass tone (all "None" + 0.0). User flips
        # weight to a non-zero value to actually activate a LoRA.
        lora_names: list[gr.Dropdown] = []
        lora_weights: list[gr.Slider] = []
        with gr.Accordion(
            "SAM3 Anima LoRA Stack (4 slots — leave None to bypass)",
            open=False,
            elem_id="sam3_anima_lora_accordion",
        ):
            for i in range(4):
                with gr.Row():
                    lora_names.append(
                        gr.Dropdown(
                            label=f"SAM3 Anima LoRA #{i + 1}",
                            choices=lora_choices,
                            value="None",
                            type="value",
                            elem_id=f"sam3_anima_lora_name_{i}",
                        )
                    )
                    lora_weights.append(
                        gr.Slider(
                            label=f"SAM3 Anima LoRA Weight #{i + 1}",
                            minimum=-2.0,
                            maximum=2.0,
                            step=0.05,
                            value=0.0,
                            elem_id=f"sam3_anima_lora_weight_{i}",
                        )
                    )

        # --- Sampler ----------------------------------------------------
        with gr.Row():
            steps = gr.Slider(
                label="SAM3 Anima Steps",
                minimum=1,
                maximum=150,
                step=1,
                value=50,
                elem_id="sam3_anima_steps",
            )
            cfg = gr.Slider(
                label="SAM3 Anima CFG Scale",
                minimum=0.0,
                maximum=20.0,
                step=0.1,
                value=3.5,
                elem_id="sam3_anima_cfg",
            )
            flow_shift = gr.Slider(
                label="SAM3 Anima Flow Shift",
                minimum=0.0,
                maximum=30.0,
                step=0.1,
                value=5.0,
                elem_id="sam3_anima_flow_shift",
            )
        with gr.Row():
            seed = gr.Number(
                label="SAM3 Anima Seed (-1 = random)",
                value=-1,
                precision=0,
                scale=4,
                elem_id="sam3_anima_seed",
            )
            seed_random_button = gr.Button(
                "🎲", scale=0, min_width=40, elem_id="sam3_anima_seed_random"
            )
            seed_pull_button = gr.Button(
                "🎯 Pull from selected",
                scale=0,
                min_width=160,
                elem_id="sam3_anima_seed_pull",
            )

        # --- Output sizing ---------------------------------------------
        # The source keeps its aspect ratio: this sets the short side, the
        # long side follows, both rounded down to a multiple of 32 (min 256).
        with gr.Row():
            short_side = gr.Slider(
                label="SAM3 Anima Short Side (keeps source aspect ratio)",
                minimum=256,
                maximum=4096,
                step=32,
                value=1024,
                elem_id="sam3_anima_short_side",
            )

        # --- LLLite ------------------------------------------------------
        # Only the multiplier exists on the vendor side
        # (networks/control_net_lllite_anima.set_multiplier); there is no
        # strength or step schedule, so no sliders for them. Range is the
        # kohya ComfyUI-Anima-LLLite strength input (-10..10, step .01,
        # default 1.0 = sd-scripts --lllite_multiplier default). The label
        # is also the ui-config.json key; it changed with the range so the
        # old saved 0..2 / .05 bounds are not reapplied.
        with gr.Row():
            lllite_multiplier = gr.Slider(
                label="SAM3 Anima LLLite Multiplier (-10 ~ 10)",
                minimum=-10.0,
                maximum=10.0,
                step=0.01,
                value=1.0,
                elem_id="sam3_anima_lllite_multiplier",
            )

        # --- Housekeeping ----------------------------------------------
        with gr.Row():
            unload_forge_before = gr.Checkbox(
                label="SAM3 Anima Unload Forge SD before run "
                "(recommended for ≤16 GB GPU)",
                value=True,
                elem_id="sam3_anima_unload_forge",
            )
            insert_mode = gr.Radio(
                label="SAM3 Anima Insert Result",
                choices=["After selected", "At end"],
                value="After selected",
                elem_id="sam3_anima_insert_mode",
            )

        # --- Restoration mode: Anima tile-repair vs PiD upscale --------
        pid_choices = list_pid_checkpoints()
        with gr.Accordion(
            "Restoration Mode (Anima Tile-Repair / PiD Upscale)",
            open=False,
            elem_id="sam3_anima_restore_accordion",
        ):
            gr.Markdown(
                "기본은 **Anima Tile-Repair**(ControlNet-LLLite). "
                "**PiD Upscale**는 Forge Neo 네이티브 PiD(Pixel Diffusion Decoder) "
                "초해상 복원 — 파일명에 `PiD`가 포함된 체크포인트를 "
                "`models/Stable-diffusion/`에 넣어야 동작합니다 "
                "(없으면 드롭다운에 안내가 뜹니다). 마스크 미사용·전체 이미지 업스케일."
            )
            restore_mode = gr.Radio(
                label="SAM3 Anima Restoration Mode",
                choices=["Anima Tile-Repair", "PiD Upscale"],
                value="Anima Tile-Repair",
                elem_id="sam3_anima_restore_mode",
            )
            with gr.Row():
                pid_checkpoint = gr.Dropdown(
                    label="SAM3 Anima PiD Checkpoint",
                    choices=pid_choices,
                    value=pid_choices[0] if pid_choices else "(no PiD checkpoint found)",
                    type="value",
                    elem_id="sam3_anima_pid_ckpt",
                )
                pid_scale = gr.Slider(
                    label="SAM3 Anima PiD Resize (x)",
                    minimum=1.0,
                    maximum=4.0,
                    step=0.5,
                    value=4.0,
                    elem_id="sam3_anima_pid_scale",
                )
            with gr.Row():
                pid_steps = gr.Slider(
                    label="SAM3 Anima PiD Steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=8,
                    elem_id="sam3_anima_pid_steps",
                )
                pid_degrade = gr.Slider(
                    label="SAM3 Anima PiD Degrade σ (denoise 재해석)",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.4,
                    elem_id="sam3_anima_pid_degrade",
                )

        # --- Run / Stop ------------------------------------------------
        with gr.Row():
            repair_button = gr.Button(
                "▶ Run (Tile-Repair / PiD)",
                variant="primary",
                elem_id="sam3_anima_repair_button",
            )
            # Hidden until click; Stop swap mirrors Refine.
            stop_button = gr.Button(
                "⏹ Stop",
                variant="stop",
                visible=False,
                elem_id="sam3_anima_stop_button",
            )

        status = gr.HTML(value="", elem_id="sam3_anima_status")

    return AnimaPanel(
        container=acc,
        selected_index_state=selected_index_state,
        lllite_model=lllite_model,
        dit_override=dit_override,
        te_override=te_override,
        vae_override=vae_override,
        positive=positive,
        negative=negative,
        lora_names=lora_names,
        lora_weights=lora_weights,
        steps=steps,
        cfg=cfg,
        flow_shift=flow_shift,
        seed=seed,
        seed_random_button=seed_random_button,
        seed_pull_button=seed_pull_button,
        short_side=short_side,
        lllite_multiplier=lllite_multiplier,
        unload_forge_before=unload_forge_before,
        insert_mode=insert_mode,
        restore_mode=restore_mode,
        pid_checkpoint=pid_checkpoint,
        pid_scale=pid_scale,
        pid_steps=pid_steps,
        pid_degrade=pid_degrade,
        repair_button=repair_button,
        stop_button=stop_button,
        status=status,
    )


# ---------------------------------------------------------------------------
# Click handler — same 4-tuple return shape as ui_refine.handle_refine_click.
# ---------------------------------------------------------------------------


def _anima_error_return(gallery_value, message: str):
    return gallery_value, message, gr.update(), gr.update()


# Job names the two restoration modes run under (shared.state.begin). The
# Stop button only interrupts while one of them holds shared.state.job.
TILE_REPAIR_JOB = "sam3_tile_repair"
PID_UPSCALE_JOB = "sam3_pid_upscale"
_ANIMA_JOB_PREFIXES = (TILE_REPAIR_JOB, PID_UPSCALE_JOB)


def stop_anima():
    """⏹ handler for the Anima panel — same rule as ``ui_refine.stop_refine``:
    never interrupt a txt2img that holds the queue while our click waits."""
    stop_if_job(_ANIMA_JOB_PREFIXES)


def _map_widget_values(values: tuple) -> AnimaTileRepairArgs:
    keyed = dict(zip(ANIMA_ARG_KEYS, values))
    lora_slots: list[tuple[str, float]] = []
    for i in range(4):
        name = str(keyed.get(f"lora_name_{i}") or "None")
        weight = _as_float(keyed.get(f"lora_weight_{i}"), 0.0)
        lora_slots.append((name, weight))
    return AnimaTileRepairArgs(
        lllite_model=str(keyed.get("lllite_model") or "None"),
        dit_override=str(keyed.get("dit_override") or "Use Forge current"),
        te_override=str(keyed.get("te_override") or "Use Forge current"),
        vae_override=str(keyed.get("vae_override") or "Use Forge current"),
        positive=str(keyed.get("positive") or ""),
        negative=str(keyed.get("negative") or ""),
        lora_slots=lora_slots,
        steps=_as_int(keyed.get("steps"), 50),
        cfg=_as_float(keyed.get("cfg"), 3.5),
        flow_shift=_as_float(keyed.get("flow_shift"), 5.0),
        seed=_as_int(keyed.get("seed"), -1),
        short_side=_as_int(keyed.get("short_side"), 1024),
        lllite_multiplier=_as_float(keyed.get("lllite_multiplier"), 1.0),
        unload_forge_before=bool(keyed.get("unload_forge_before", True)),
        insert_mode=str(keyed.get("insert_mode") or "After selected"),
        restore_mode=str(keyed.get("restore_mode") or "Anima Tile-Repair"),
        pid_checkpoint=str(keyed.get("pid_checkpoint") or ""),
        pid_scale=_as_float(keyed.get("pid_scale"), 4.0),
        pid_steps=_as_int(keyed.get("pid_steps"), 8),
        pid_degrade=_as_float(keyed.get("pid_degrade"), 0.4),
    )


def handle_anima_click(
    gallery_value,
    selected_index,
    *all_values,
    progress=gr.Progress(track_tqdm=True),
):
    """Anima-button handler. Returns
    ``(updated_gallery, status_html, html_info, generation_info_json)`` —
    identical shape to ``ui_refine.handle_refine_click`` so the existing
    gallery sidebar wiring picks up the new entries.

    ``progress=gr.Progress(track_tqdm=True)`` makes the browser progress bar
    pick up the vendor's per-step tqdm — same trick as the Refine panel.
    """
    expected_widget_count = len(ANIMA_ARG_KEYS)
    if len(all_values) < expected_widget_count:
        return _anima_error_return(
            gallery_value,
            "<span style='color:#c33'>SAM3 Anima: missing widget values.</span>",
        )

    widget_values = all_values[:expected_widget_count]
    extras = all_values[expected_widget_count:]
    current_info_json = str(extras[0]) if len(extras) > 0 else ""

    repair = _map_widget_values(widget_values)
    is_pid = repair.restore_mode == "PiD Upscale"

    # Anima Tile-Repair needs the vendored sd-scripts + a prompt. PiD Upscale
    # uses Forge Neo's NATIVE pipeline (no vendor) and is condition-driven (no
    # prompt), so those checks only apply to the Anima path.
    if not is_pid:
        if not anima_available():
            return _anima_error_return(
                gallery_value,
                "<span style='color:#c33'>Anima vendor missing — install.py "
                "didn't clone kohya-ss/sd-scripts. Re-run Forge or clone "
                "manually into <code>anima_vendor/</code>.</span>",
            )
        if not repair.positive.strip():
            return _anima_error_return(
                gallery_value,
                "<span style='color:#c33'>SAM3 Anima: prompt is empty.</span>",
            )

    gallery_list = list(gallery_value or [])
    if not gallery_list:
        return _anima_error_return(
            gallery_value,
            "<span style='color:#c33'>SAM3 Anima: gallery is empty.</span>",
        )

    try:
        idx = int(selected_index) if selected_index is not None else -1
    except (TypeError, ValueError):
        idx = -1
    if idx < 0 or idx >= len(gallery_list):
        idx = len(gallery_list) - 1

    source = _coerce_gallery_item_to_pil(gallery_list[idx])
    if source is None:
        return _anima_error_return(
            gallery_value,
            f"<span style='color:#c33'>SAM3 Anima: could not load selected "
            f"image (index {idx}).</span>",
        )

    def _pid():
        return run_pid_upscale(
            source,
            pid_checkpoint=repair.pid_checkpoint,
            scale=repair.pid_scale,
            degrade_sigma=repair.pid_degrade,
            steps=repair.pid_steps,
        )

    def _tile():
        return run_tile_repair(source, repair)

    # Like Forge's own Generate (wrap_gradio_gpu_call): queue_lock so we never
    # unload/swap models under a running txt2img, a fresh shared.state so a
    # previous ⏹ Stop does not abort this run, and the work on Forge's main
    # thread. See forge_exclusive.run_exclusive.
    try:
        if is_pid:
            new_pairs = run_exclusive(PID_UPSCALE_JOB, _pid, on_main_thread=True)
        else:
            new_pairs = run_exclusive(TILE_REPAIR_JOB, _tile, on_main_thread=True)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return _anima_error_return(
            gallery_value,
            f"<pre style='color:#c33'>SAM3 {'PiD' if is_pid else 'Anima'} failed: "
            f"{type(exc).__name__}: {exc}</pre>",
        )

    if not new_pairs:
        return _anima_error_return(
            gallery_value,
            "<span style='color:#c80'>SAM3 Anima: no output produced "
            "(interrupted?).</span>",
        )

    new_images = [img for img, _ in new_pairs]
    new_infotexts = [info for _, info in new_pairs]

    if repair.insert_mode == "At end" or idx < 0:
        updated = gallery_list + new_images
    else:
        updated = gallery_list[: idx + 1] + new_images + gallery_list[idx + 1:]

    # Splice infotexts into generation_info JSON — same shape Refine uses.
    info_payload: dict[str, Any]
    try:
        info_payload = json.loads(current_info_json) if current_info_json else {}
    except Exception:
        info_payload = {}
    existing = list(info_payload.get("infotexts") or [""] * len(gallery_list))
    while len(existing) < len(gallery_list):
        existing.append("")
    if repair.insert_mode == "At end" or idx < 0:
        merged = existing + new_infotexts
    else:
        merged = existing[: idx + 1] + new_infotexts + existing[idx + 1:]
    info_payload["infotexts"] = merged
    new_info_json = json.dumps(info_payload, ensure_ascii=False)

    latest_html = _plaintext_to_html(new_infotexts[-1] if new_infotexts else "")
    return (
        updated,
        f"<span style='color:#383'>SAM3 Anima: added {len(new_images)} "
        f"image(s). Click the new thumbnail to recheck infotext.</span>",
        latest_html,
        new_info_json,
    )
