# Colorcraft (sam-extra) — the per-evaluation modifier chain (the body of the ComfyUI node's
# post-CFG function), framework-agnostic.
#
# origin: muerrilla/ComfyUI-Colorcraft@d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf:nodes.py
#   * build_entries    :550-560  (schedule per chain entry, inside ``wrapped_sampler_function``)
#   * needs_basis      :547-548, 562-567 (``ADVANCED_KINDS`` / ``any_advanced``)
#   * apply_chain      :575-583, 605-728 (``post_cfg_function``: 5-D squeeze, the modifier loop, unsqueeze)
# structure after aoleg/ComfyUI-Colorcraft@f00066c63c9d8f96abc119cada3b51b36688fcac:lib_colorcraft/engine.py
#   (the fork lifted the same body out of the node with injected host callables; its body is the
#    older upstream math — this file carries upstream d28ac6a's)
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
# Changes by sam-extra (2026-10-03) — the modifier loop is upstream's statement for statement
# (tests/test_colorcraft_origin.py compares the two syntax trees after exactly these substitutions):
#   * ``get_color_latent(colour, latent_format, device, dtype)`` → ``get_color_latent(colour, device,
#     dtype)``: Forge's anchors are VAE-encoded once before sampling (a ``vae.encode`` inside the
#     sampling loop would load the VAE and can evict the UNet), so the latent format is bound there.
#   * ``VAE_DOWNSCALE_FACTOR`` (upstream's constant 8) → the family's own factor (16 for Flux2).
#   * ``s = sigma_to_value(cur_sigma, sigmas, schedule)`` → ``s = values[i]``: the hook works the
#     values out first with the same ``sigma_to_value`` (or, for a sampler that publishes no sigma
#     list, upstream's Forge step counter) so it can pass a call through untouched when every
#     modifier is at 0.
#   * Upstream's debug-panel capture before the loop is not part of this port.
"""The modifier chain one model evaluation runs: contrast, colour shift and the basis-vector edits."""

from __future__ import annotations

from .basis import resolve_dev
from .color import apply_color_shift, apply_contrast
from .masking import apply_mask_gate
from .schedule import make_schedule, sigma_to_value
from .vectors import (
    apply_chroma_contrast,
    apply_tone_compression,
    apply_vector_offset,
    apply_vibrance,
    chroma_axes,
)

__all__ = [
    "ADVANCED_KINDS",
    "KINDS",
    "NEUTRAL_COLOR",
    "apply_chain",
    "build_entries",
    "needed_colors",
    "needs_basis",
    "schedule_values",
]

# The ComfyUI node classes, in the order the panel offers them (``kind`` is what each node's
# ``make`` writes into its chain entry, nodes.py ColorcraftBasic … ColorcraftShift).
KINDS = ("advanced", "basic", "luma", "chroma", "chroma_plus", "punch", "shift")

# Kinds whose controls depend on the resolved basis vectors.
ADVANCED_KINDS = {"advanced", "luma", "chroma", "chroma_plus", "punch"}

# The anchor contrast pivots on: a mid-grey image (``build_color_latent`` with no colour offset).
NEUTRAL_COLOR = (0.0, 0.0, 0.0, 0.0)

_SCHEDULE_KEYS = ("start", "end", "bias", "exponent", "start_off", "end_off", "smooth")


def build_entries(modifiers, num_steps):
    """``(schedule, kind, params, mask)`` per chain entry for a run of ``num_steps`` steps.

    origin: nodes.py:550-560. Basic/Advanced own their schedule widgets inline (in ``params``);
    Luma/Chroma/Punch get theirs from a separate schedule dict stashed on the entry."""
    built = []
    for entry in modifiers:
        p = entry["params"]
        # Basic/Advanced own their schedule widgets inline (in `p`);
        # Luma/Chroma/Punch get theirs from a required
        # COLORCRAFT_SCHEDULE input instead, stashed on the entry.
        sched = entry.get("schedule") or p
        schedule_kwargs = {k: sched[k] for k in _SCHEDULE_KEYS}
        schedule = make_schedule(num_steps, amount=sched["strength"], **schedule_kwargs)
        built.append((schedule, entry["kind"], p, entry.get("mask")))
    return built


def needs_basis(built):
    """Does anything in the chain need the basis vectors? (origin: nodes.py:562-567)

    Shift works on any model and isn't in ADVANCED_KINDS, but its optional masking input does need
    a basis -- so it counts only in that case. Accepts built entries or raw chain entries."""
    for item in built:
        if isinstance(item, dict):
            kind, mask_params = item["kind"], item.get("mask")
        else:
            _, kind, _, mask_params = item
        if kind in ADVANCED_KINDS or (kind == "shift" and mask_params):
            return True
    return False


def needed_colors(modifiers):
    """Every ``(red, green, blue, brightness)`` anchor the chain can ask ``get_color_latent`` for.

    Mirrors the branches of ``apply_chain``: contrast on Basic/Punch (any kind but Advanced whose
    params carry ``contrast``) and on Advanced; colour shift on Basic and Shift (gated on the amount)
    and on Advanced (gated on ``color_shift`` too). Taken from the raw chain so the anchors can be
    encoded before the step count is known."""
    colors = []
    for entry in modifiers:
        kind, p = entry["kind"], entry["params"]
        if "contrast" in p and p["contrast"] != 0:
            colors.append(NEUTRAL_COLOR)
        if kind in ("basic", "shift") and p.get("color_shift_amount", 0) != 0:
            colors.append((p["red"], p["green"], p["blue"], p["brightness"]))
        elif kind == "advanced" and p.get("color_shift") and p.get("color_shift_amount", 0) != 0:
            colors.append((p["red"], p["green"], p["blue"], p["brightness"]))
    seen, unique = set(), []
    for color in colors:
        if color not in seen:
            seen.add(color)
            unique.append(color)
    return unique


def schedule_values(built, cur_sigma, sigmas):
    """``sigma_to_value`` for every entry — the ``s`` each one runs at for this evaluation."""
    return [sigma_to_value(cur_sigma, sigmas, schedule) for schedule, _, _, _ in built]


def apply_chain(built, x0, values, family, cur_basis, get_color_latent, vae_downscale_factor):
    """Run the chain on the CFG-combined prediction ``x0`` (``denoised``) and return the result.

    ``values[i]`` is entry ``i``'s schedule value for this evaluation; ``cur_basis`` the family's
    vectors on ``x0``'s device and dtype (None: vector controls off); ``get_color_latent(colour,
    device, dtype)`` the pre-encoded model-space anchor. Out-of-place throughout: with every entry
    at 0 the result is ``x0`` itself (as a 5-D view when ``x0`` is 5-D, like upstream)."""
    is_5d = x0.dim() == 5
    if is_5d:
        x0 = x0.squeeze(2)

    out = x0
    for i, (schedule, kind, p, mask_params) in enumerate(built):
        s = values[i]
        if s == 0:
            continue

        pre = out
        if kind != "advanced" and "contrast" in p and p["contrast"] != 0:
            neutral_anchor = get_color_latent(
                (0.0, 0.0, 0.0, 0.0), out.device, out.dtype,
            )
            out = apply_contrast(out, s * p["contrast"], neutral_anchor)

        if kind == "basic":
            if p["color_shift_amount"] != 0:
                color_anchor = get_color_latent(
                    (p["red"], p["green"], p["blue"], p["brightness"]),
                    out.device, out.dtype,
                )
                out = apply_color_shift(out, s * p["color_shift_amount"], p["mode"], color_anchor)

        elif kind == "advanced":
            dev = resolve_dev(p, family) if cur_basis is not None else None
            if cur_basis is not None:
                axis1, axis2 = chroma_axes(cur_basis, dev["chroma_plane"])
                out = apply_vector_offset(out, cur_basis["exposure"], s * p["exposure"])
                out = apply_tone_compression(out, cur_basis["exposure"], s * p["tone_compression"])

            if p["contrast"] != 0:
                neutral_anchor = get_color_latent(
                    (0.0, 0.0, 0.0, 0.0), out.device, out.dtype,
                )
                out = apply_contrast(out, s * p["contrast"], neutral_anchor)

            if cur_basis is not None:
                out = apply_vibrance(out, axis1, axis2, s * p["vibrance"] * 2.0, k=dev["vibrance_k"], recenter=dev["recenter"], r_max=dev["max_chroma"])
                out = apply_vibrance(out, axis1, axis2, s * p["saturation"], k=0.0, r_max=0.0, recenter=dev["recenter"])
                out = apply_vector_offset(out, cur_basis["temperature"], s * p["temperature"])
                out = apply_vector_offset(out, cur_basis["tint"], s * p["tint"])
                if p["more_colors"]:
                    out = apply_vector_offset(out, cur_basis["temp+tint"], s * p["temp_plus_tint"])
                    out = apply_vector_offset(out, cur_basis["temp-tint"], s * p["temp_minus_tint"])
                    out = apply_vector_offset(out, cur_basis["lab-a"], s * p["lab_a"])
                    out = apply_vector_offset(out, cur_basis["lab-b"], s * p["lab_b"])
                    out = apply_vector_offset(out, cur_basis["lab-a+b"], s * p["lab_a_plus_b"])
                    out = apply_vector_offset(out, cur_basis["lab-a-b"], s * p["lab_a_minus_b"])
                out = apply_chroma_contrast(
                    out, axis1, axis2, s * p["chroma_contrast"],
                    r_max=dev["max_chroma"], chroma_center=p["chroma_center"], recenter=dev["recenter"],
                )

            if p["color_shift"] and p["color_shift_amount"] != 0:
                color_anchor = get_color_latent(
                    (p["red"], p["green"], p["blue"], p["brightness"]),
                    out.device, out.dtype,
                )
                out = apply_color_shift(out, s * p["color_shift_amount"], p["mode"], color_anchor)

            if cur_basis is not None:
                out = apply_vector_offset(out, cur_basis["clarity"], s * p["clarity"])
                out = apply_vector_offset(out, cur_basis["sharpness"], s * p["sharpness"])

            # Reuses the same `dev` already resolved above for its own
            # edits to gate an external mask -- one resolution, shared.
            if mask_params and cur_basis is not None:
                out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, vae_downscale_factor)

        elif kind == "luma":
            dev = resolve_dev(p, family) if cur_basis is not None else None
            if cur_basis is not None:
                out = apply_vector_offset(out, cur_basis["exposure"], s * p["exposure"])
                out = apply_tone_compression(out, cur_basis["exposure"], s * p["tone_compression"])
            if mask_params and cur_basis is not None:
                out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, vae_downscale_factor)

        elif kind == "chroma":
            dev = resolve_dev(p, family) if cur_basis is not None else None
            if cur_basis is not None:
                axis1, axis2 = chroma_axes(cur_basis, dev["chroma_plane"])
                out = apply_vibrance(out, axis1, axis2, s * p["vibrance"] * 2.0, k=dev["vibrance_k"], recenter=dev["recenter"], r_max=dev["max_chroma"])
                out = apply_vibrance(out, axis1, axis2, s * p["saturation"], k=0.0, r_max=0.0, recenter=dev["recenter"])
                out = apply_chroma_contrast(
                    out, axis1, axis2, s * p["chroma_contrast"],
                    r_max=dev["max_chroma"], chroma_center=p["chroma_center"], recenter=dev["recenter"],
                )
                out = apply_vector_offset(out, cur_basis["temperature"], s * p["temperature"])
                out = apply_vector_offset(out, cur_basis["tint"], s * p["tint"])
            if mask_params and cur_basis is not None:
                out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, vae_downscale_factor)

        elif kind == "chroma_plus":
            dev = resolve_dev(p, family) if cur_basis is not None else None
            if cur_basis is not None:
                out = apply_vector_offset(out, cur_basis["temp+tint"], s * p["temp_plus_tint"])
                out = apply_vector_offset(out, cur_basis["temp-tint"], s * p["temp_minus_tint"])
                out = apply_vector_offset(out, cur_basis["lab-a"], s * p["lab_a"])
                out = apply_vector_offset(out, cur_basis["lab-b"], s * p["lab_b"])
                out = apply_vector_offset(out, cur_basis["lab-a+b"], s * p["lab_a_plus_b"])
                out = apply_vector_offset(out, cur_basis["lab-a-b"], s * p["lab_a_minus_b"])
            if mask_params and cur_basis is not None:
                out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, vae_downscale_factor)

        elif kind == "punch":
            dev = resolve_dev(p, family) if cur_basis is not None else None
            if cur_basis is not None:
                out = apply_vector_offset(out, cur_basis["clarity"], s * p["clarity"])
                out = apply_vector_offset(out, cur_basis["sharpness"], s * p["sharpness"])
            if mask_params and cur_basis is not None:
                out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, vae_downscale_factor)

        elif kind == "shift":
            if p["color_shift_amount"] != 0:
                color_anchor = get_color_latent(
                    (p["red"], p["green"], p["blue"], p["brightness"]),
                    out.device, out.dtype,
                )
                out = apply_color_shift(out, s * p["color_shift_amount"], p["mode"], color_anchor)
            if mask_params and cur_basis is not None:
                dev = resolve_dev(p, family)
                out = apply_mask_gate(pre, out, mask_params, dev, cur_basis, vae_downscale_factor)

    if is_5d:
        out = out.unsqueeze(2)
    return out
