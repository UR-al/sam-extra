# Colorcraft (sam-extra) — the Debug panel: capture the pre-edit x0 at one step during generation,
# then render axis projections and mask/combo previews on demand.
#
# origin: muerrilla/ComfyUI-Colorcraft@d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf:scripts/colorcraft.py
#   * DEBUG_CAPTION_MARKER :41, DEBUG_AXIS_OPTIONS :57-61, the Debug widgets :614-626
#   * decode_debug_latent :112-135
#   * build_debug_gallery_images :820-905
#   * the capture in denoised_callback :956-966 (pre-edit x, a resolution change starts fresh, batches
#     of the same size accumulate) and the debug-step clamp in denoiser_callback :936-937
#
# MIT License
#
# Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla)
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
# Changes by sam-extra (2026-10-03):
#   * Capture only while the panel's "Capture debug latent" is on (upstream captures whenever
#     Colorcraft runs), once per step (upstream's step counter captures every model call of the debug
#     step), on the pass's own sampling run, on a model with a basis (as upstream).
#   * The step is found by sigma on the walked list (or upstream's step counter for samplers without
#     one), clamped to the run's last step like upstream.
#   * Images go to a gallery inside the Debug accordion instead of being spliced into Forge's output
#     gallery (no after_component wiring, nothing added to the generation's own results).
#   * Image size uses the family's own downscale factor (16 for Flux2; upstream's constant 8).
#   * ``decode`` and the basis are passed in, so the renderer runs without the WebUI.
"""Upstream's Debug panel on Forge: capture during sampling, render when asked."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import basis as basis_mod
from .debug import (
    DEBUG_AXIS_STYLES,
    DEBUG_COMPOSITE_COLORS,
    DEBUG_OVERLAY_COLORS,
    build_curve_infos,
    collect_leaves,
    composite_mask_images,
    compute_axis_projection,
    debug_tensor_to_images,
    downscale_latent_for_storage,
    render_hue_images,
)
from .masking import compute_hue_projection, compute_saturation_projection, resolve_mask_tensor
from .vectors import chroma_axes

# zero-width-space prefix, invisible in the UI (upstream :41)
DEBUG_CAPTION_MARKER = "​[colorcraft-debug]"

# Real stored basis vectors, plus saturation; "hue" is the raw angle the mask gate uses, "hue (weighted)"
# the same angle desaturated where chroma is small (upstream :57-61).
DEBUG_AXIS_OPTIONS = [
    "exposure", "temperature", "tint", "temp+tint", "temp-tint",
    "lab-a", "lab-b", "lab-a+b", "lab-a-b", "clarity", "sharpness", "saturation",
    "hue", "hue (weighted)",
]

# Upstream's Debug Step slider (:624): 0..50, step 1, default 5.
DEBUG_STEP_MIN, DEBUG_STEP_MAX, DEBUG_STEP_DEFAULT = 0, 50, 5

MANY_IMAGES = 20


@dataclass
class Capture:
    latent: torch.Tensor   # [B, C, H, W] on the CPU, the pre-edit x0 (5-D Wan latents squeezed)
    is_5d: bool
    family: str


def clamp_step(debug_step, n_steps):
    """Upstream's clamp (:936-937): the debug step is at most the run's last step."""
    try:
        step = int(round(float(debug_step)))
    except (TypeError, ValueError):
        step = DEBUG_STEP_DEFAULT
    return max(0, min(step, int(n_steps) - 1))


def capture(state, x, is_5d, family):
    """Keep ``x`` (pre-edit, 4-D) for the panel (upstream :949-967).

    A capture of another size or family (base pass → hires pass) starts fresh; same-size captures
    (n_iter batches of one request) accumulate."""
    captured = x.detach().float().cpu()
    previous = state.get("debug")
    if previous is not None and (previous.latent.shape[1:] != captured.shape[1:] or previous.family != family):
        previous = None
    latent = captured if previous is None else torch.cat([previous.latent, captured], dim=0)
    state["debug"] = Capture(latent, bool(is_5d), family)
    return state["debug"]


def decode_debug_latent(sd_model, latent, width, height, is_5d=False):
    """The compositing base images: decode a (downscaled) capture, one RGB image per batch item.

    origin: scripts/colorcraft.py:112-135 — ``sd_model`` is passed in (upstream reads shared.sd_model)."""
    from PIL import Image

    from modules import devices
    from modules.shared import device

    latent = latent.to(device=device, dtype=devices.dtype_vae)
    if is_5d:
        latent = latent.unsqueeze(2)
    with torch.no_grad():
        img = sd_model.decode_first_stage(latent)
    if img.dim() == 5:
        img = img.squeeze(1)
    img = (img / 2 + 0.5).clamp(0.0, 1.0)
    images = []
    for b in range(img.shape[0]):
        arr = (img[b].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
        pil = Image.fromarray(arr, mode="RGB").resize((width, height), Image.BILINEAR)
        images.append(pil)
    return images


def cpu_basis_for(family):
    from .hook import load_family_basis

    source = load_family_basis(family)
    # a plain CPU float32 copy: the cached vectors may be inference tensors (made during sampling)
    return None if source is None else {k: v.detach().to(device="cpu", dtype=torch.float32, copy=True)
                                        for k, v in source.items()}


def build_debug_gallery_images(debug_capture, debug_axes, debug_masks, debug_combos, debug_composite_color,
                               debug_overlay_color, debug_axis_style, leaf_specs, combo_specs, decode=None,
                               warn=print):
    """``[(image, caption)]`` for the selected axes, masks and combos (upstream :820-905).

    ``decode(latent, width, height, is_5d)`` makes the compositing base images; only called when a
    composite colour is chosen."""
    if debug_capture is None:
        warn("[Colorcraft] Debug: no debug latent has been captured yet -- turn on 'Capture debug latent' "
             "and generate once first.")
        return []
    cur_basis = cpu_basis_for(debug_capture.family) if debug_capture.family else None
    if cur_basis is None:
        warn("[Colorcraft] Debug: no basis for the captured model -- nothing to show.")
        return []
    dev = basis_mod.resolve_dev({}, debug_capture.family)
    downscale = basis_mod.vae_downscale_factor(debug_capture.family)

    x = debug_capture.latent
    p_height = x.shape[2] * downscale
    p_width = x.shape[3] * downscale
    composite_color = DEBUG_COMPOSITE_COLORS.get(debug_composite_color)
    base_images = None
    if composite_color is not None and decode is not None:
        downscaled = downscale_latent_for_storage(x, max_dim=64)
        try:
            base_images = decode(downscaled, p_width, p_height, debug_capture.is_5d)
        except Exception as exc:
            warn(f"[Colorcraft] Debug: could not decode the captured latent ({type(exc).__name__}: {exc}); "
                 "showing masks without compositing.")
            base_images = None

    results = []
    total = 0

    def emit(label, imgs):
        nonlocal total
        total += len(imgs)
        for img in imgs:
            results.append((img, f"{DEBUG_CAPTION_MARKER} {label}"))

    for axis in debug_axes or []:
        if axis in ("hue", "hue (weighted)"):
            axis1, axis2 = chroma_axes(cur_basis, dev["chroma_plane"])
            proj = compute_hue_projection(x, axis1, axis2, dev.get("hue_bias", 0.0))
            chroma_frac = None
            if axis == "hue (weighted)":
                sat_proj = compute_saturation_projection(x, axis1, axis2, dev["max_chroma"])
                chroma_frac = ((sat_proj + 1.0) / 2.0).clamp(0.0, 1.0)
            imgs = render_hue_images(proj, p_width, p_height, f"axis:{axis}", debug_overlay_color, None, chroma_frac)
            emit(f"axis:{axis}", imgs)
            continue
        proj = compute_axis_projection(axis, x, cur_basis, dev)
        if proj is None:
            continue
        imgs = debug_tensor_to_images(proj, p_width, p_height, True, False, f"axis:{axis}",
                                      debug_overlay_color, None, debug_axis_style)
        emit(f"axis:{axis}", imgs)

    for kind, tags, spec_map in (("mask", debug_masks, leaf_specs), ("combo", debug_combos, combo_specs)):
        for tag in tags or []:
            spec = spec_map.get(tag)
            if spec is None:
                warn(f"[Colorcraft] Debug: {kind} {tag} has an unresolved reference -- skipping.")
                continue
            label = f"{kind}:{tag}"
            mask_tensor = resolve_mask_tensor(spec, x, cur_basis, dev, downscale)
            curve_infos = build_curve_infos(spec, x, cur_basis, dev)
            # Checked against every leaf (not just curve_infos, which leaves hue out): a hue-axis split
            # leaf can still go negative.
            is_signed = any(leaf.get("mask_mode") == "split" for leaf in collect_leaves(spec))
            if composite_color is not None and base_images is not None:
                imgs = composite_mask_images(mask_tensor, composite_color, base_images, label, debug_overlay_color,
                                             curve_infos)
            else:
                imgs = debug_tensor_to_images(mask_tensor, p_width, p_height, False, is_signed, label,
                                              debug_overlay_color, curve_infos, debug_axis_style)
            emit(label, imgs)

    if total > MANY_IMAGES:
        warn(f"[Colorcraft] Debug: {total} images this refresh (batch size x selected axes/masks/combos) -- "
             "consider narrowing the selection.")
    return results


def render_for_panel(debug_capture, axes, masks, combos, composite, overlay, axis_style, mask_values,
                     sd_model=None, warn=print):
    """The Refresh button: mask/combo specs from the panel's current values, then the gallery images.

    ``mask_values`` are every leaf then every combo control in script-argument order. ``sd_model`` (default:
    the loaded ``shared.sd_model``) decodes the composite base images."""
    from . import spec

    values = list(mask_values)
    config = spec.default_config()
    for i in range(spec.MASK_COUNT):
        for j, f in enumerate(spec.LEAF_FIELDS):
            index = i * spec.N_LEAF + j
            config.leaves[i][f.name] = spec.coerce(f, values[index] if index < len(values) else f.default)
    base = spec.MASK_COUNT * spec.N_LEAF
    for i in range(spec.COMBO_COUNT):
        for j, f in enumerate(spec.COMBO_FIELDS):
            index = base + i * spec.N_COMBO + j
            choices = spec.combo_ref_choices(i) if f.name in ("mask_a", "mask_b") else None
            config.combos[i][f.name] = spec.coerce(f, values[index] if index < len(values) else f.default, choices)
    leaf_specs, combo_specs = spec.build_mask_specs(config)

    def decode(latent, width, height, is_5d):
        model = sd_model
        if model is None:
            from modules import shared

            model = shared.sd_model
        return decode_debug_latent(model, latent, width, height, is_5d)

    return build_debug_gallery_images(debug_capture, axes, masks, combos, composite, overlay, axis_style,
                                      leaf_specs, combo_specs, decode=decode, warn=warn)


__all__ = [
    "Capture", "DEBUG_AXIS_OPTIONS", "DEBUG_AXIS_STYLES", "DEBUG_CAPTION_MARKER", "DEBUG_COMPOSITE_COLORS",
    "DEBUG_OVERLAY_COLORS", "DEBUG_STEP_DEFAULT", "DEBUG_STEP_MAX", "DEBUG_STEP_MIN", "build_debug_gallery_images",
    "capture", "clamp_step", "decode_debug_latent", "render_for_panel",
]
