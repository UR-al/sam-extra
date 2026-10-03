# Colorcraft (sam-extra) — basis-vector families, per-family calibration and vector loading.
#
# origin: muerrilla/ComfyUI-Colorcraft@d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf:lib_colorcraft/basis.py:1-58
#   (krea2/zimage tables, load_basis, resolve_dev)
# origin: aoleg/ComfyUI-Colorcraft@f00066c63c9d8f96abc119cada3b51b36688fcac:lib_colorcraft/core.py
#   (the "flux2" family: :114-143 family/downscale tables, :177-181 its MODEL_DEV_DEFAULTS row,
#    :255-261 family_for_latent_format; vectors/colorcraft-flux2.safetensors)
#
# MIT License
#
# Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla)
# (The fork aoleg/ComfyUI-Colorcraft — Oleg Afonin's Forge Neo port with the Flux2 family — is under
#  the same MIT License; its LICENSE carries the upstream notice with the earlier spelling
#  "Sahand Ahmadiantehrani".)
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
#   * Three families: upstream's krea2/zimage rows unchanged, plus the fork's flux2 (Flux2 latent
#     format, 128 packed channels, 16x VAE). Upstream later added ``detail_scale`` (4.0 for both of
#     its families, the normaliser of the clarity/sharpness mask axes); the fork's flux2 row predates
#     it, so flux2 uses upstream's 4.0 — not measured on Flux2.
#   * The flux2 row is otherwise the fork's measurement. It was taken against the calibration
#     upstream had then (krea2 hue_bias 0, zimage vibrance_k 2.0); upstream has since moved krea2's
#     hue_bias to -0.1 and zimage's vibrance_k to 1.5 and changed the vibrance curve. Not re-measured.
#   * The VAE downscale factor is per family (the fork's table) because Flux2's VAE is 16x; upstream
#     has one constant 8. It converts the mask-blur radius from image pixels to latent pixels.
#   * ``VECTORS_DIR`` points at the vectors shipped inside this package (sam3ext/colorcraft/data);
#     callers pass it to ``load_basis`` (unchanged) like upstream's entry points do.
"""Which basis-vector family a model's latent format uses, and that family's calibration."""

import os

from safetensors.torch import load_file

BASIS_FAMILIES = ["krea2", "zimage", "flux2"]

# Both upstream VAE families downscale 8x -- used to convert
# ColorcraftMaskBlur's radius from decoded-image pixels (what the UI shows)
# to latent pixels (what gaussian_blur_mask operates on).
VAE_DOWNSCALE_FACTOR = 8

# Per family (fork): Flux2's VAE downscales 16x (Forge backend/patcher/vae.py ``is_flux2``), the Wan
# VAE's spatial factor is 8.
VAE_DOWNSCALE_FACTORS = {
    "krea2": 8,
    "zimage": 8,
    "flux2": 16,
}

# latent_format class name -> basis family. Krea2/QwenImage report "Wan21";
# Flux/Z-Image report "Flux". Comfy reads the class name off latent_format
# directly; Forge reads it off p.sd_model.model_config.latent_format.
# Forge Neo (modules_forge/packages/huggingface_guess/model_list.py): Anima, Wan 2.1, Qwen-Image and
# Krea 2 use Wan21; Flux, Chroma, Lumina 2 and Z-Image use Flux; Flux 2 Klein (4B/9B) and ERNIE-Image
# use Flux2. SD 1.5 / SDXL (and Mugen's unpacked SDXL_Flux2) have no family: only the vector-free
# controls (contrast, colour shift) work there.
LATENT_FORMAT_TO_FAMILY = {
    "Wan21": "krea2",
    "Flux": "zimage",
    "Flux2": "flux2",
}

# Per-model calibrated defaults for the chroma-plane math. vibrance_k/
# exposure_scale/color_scale/hue_bias are internal only, no UI override.
# recenter/max_chroma/chroma_plane are overridable on Advanced (see
# resolve_dev) as artistic controls. exposure_scale/color_scale normalize
# each axis's raw projection to roughly +-1 across models. hue_bias
# (radians) rotates zimage's hue angle to match krea2's calibration.
MODEL_DEV_DEFAULTS = {
    "krea2":  {"vibrance_k": 1.0, "max_chroma": 2.5, "recenter": 0.5, "chroma_plane": "temp_tint", "exposure_scale": 3.5, "color_scale": 3.0, "detail_scale": 4.0, "hue_bias": -0.1},
    "zimage": {"vibrance_k": 1.5, "max_chroma": 5.0, "recenter": 0.5, "chroma_plane": "temp_tint", "exposure_scale": 7.5, "color_scale": 6.0, "detail_scale": 4.0, "hue_bias": -0.4},
    # fork row (measured on a 24-image corpus, see the header) + upstream's detail_scale
    "flux2":  {"vibrance_k": 1.83, "max_chroma": 4.6, "recenter": 0.5, "chroma_plane": "temp_tint", "exposure_scale": 4.0, "color_scale": 3.9, "detail_scale": 4.0, "hue_bias": +0.12},
}

# Shipped with this package: colorcraft-krea2/zimage (upstream d28ac6a) and colorcraft-flux2 (fork
# f00066c). tests/test_colorcraft_vectors.py pins their SHA-256, keys and shapes.
VECTORS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_basis(family, vectors_dir):
    """Loads colorcraft-<family>.safetensors from vectors_dir. Returns
    dict[name -> 1D tensor] or None if missing. vectors_dir must be
    supplied by the caller, computed relative to its own entry-point
    file -- resolving it from this module's own __file__ would point at
    lib_colorcraft/ instead of the repo root."""
    path = os.path.join(vectors_dir, f"colorcraft-{family}.safetensors")
    if not os.path.isfile(path):
        return None
    return load_file(path)


def resolve_dev(params, family):
    """Resolves recenter/max_chroma/chroma_plane against
    MODEL_DEV_DEFAULTS, honoring per-field *_override flags in params
    when present (only Advanced exposes these)."""
    dev = MODEL_DEV_DEFAULTS[family]
    return {
        "vibrance_k": dev["vibrance_k"],
        "exposure_scale": dev["exposure_scale"],
        "color_scale": dev["color_scale"],
        "detail_scale": dev["detail_scale"],
        "hue_bias": dev["hue_bias"],
        "recenter": params["recenter"] if params.get("recenter_override") else dev["recenter"],
        "max_chroma": params["max_chroma"] if params.get("max_chroma_override") else dev["max_chroma"],
        "chroma_plane": params["chroma_plane"] if params.get("chroma_plane_override") else dev["chroma_plane"],
    }


# origin: aoleg/ComfyUI-Colorcraft@f00066c:lib_colorcraft/core.py:255-261 (unchanged)
def family_for_latent_format(latent_format):
    """`latent_format` may be a class or an instance; ComfyUI hands over an
    instance and so does Forge Neo (`model_list.py:66` instantiates it)."""
    if latent_format is None:
        return None
    name = latent_format.__name__ if isinstance(latent_format, type) else type(latent_format).__name__
    return LATENT_FORMAT_TO_FAMILY.get(name)


def vae_downscale_factor(family):
    """Image pixels per latent pixel for ``family`` (8 when unknown, upstream's constant)."""
    return VAE_DOWNSCALE_FACTORS.get(family, VAE_DOWNSCALE_FACTOR)
