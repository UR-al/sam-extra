# Colorcraft (sam-extra) — vector-free whole-latent operations (contrast, colour shift) and the
# colour anchors they pull toward.
#
# origin: muerrilla/ComfyUI-Colorcraft@d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf:lib_colorcraft/color.py:1-60
#   _color_adjust, _color_adjust_legacy, apply_contrast, apply_color_shift, build_color_latent — unchanged
# origin: aoleg/ComfyUI-Colorcraft@f00066c63c9d8f96abc119cada3b51b36688fcac:lib_colorcraft/core.py:597-604
#   to_model_space — the fork's signature (latent rank passed in instead of read off a ComfyUI VAE)
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
# Changes by sam-extra (2026-10-03):
#   * ``to_model_space(latent_format, anchor, dims)`` takes the latent's spatial rank instead of
#     reading ``vae.latent_dim``. Forge's ``VAE.clone()`` does not copy ``latent_dim`` and Forge's
#     ``Wan21`` latent format has no ``latent_dimensions``, so upstream's fallback rank 2 would
#     broadcast Wan21's 5-D mean/std into a [1, 16, 16, 1, 1] anchor. The hook passes the rank of the
#     latent ``vae.encode`` actually returned. The arithmetic is upstream's.
#   * Everything else is upstream's code unchanged (compared with tests/_origin_colorcraft/).
"""Contrast and colour shift — the two controls that need no basis vectors and work on any model."""

import torch

def _color_adjust(denoised, t, anchor):
    anchor = anchor.to(dtype=denoised.dtype, device=denoised.device)
    anchor = anchor.reshape(-1)
    anchor = anchor.view((1, anchor.shape[0]) + (1,) * (denoised.dim() - 2))
    x = denoised - anchor
    reduce_dims = tuple(range(2, denoised.dim()))
    mx = x.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
    xn = x / mx
    signs = torch.sign(xn)
    ax = xn.abs()
    if t > 0:
        y = 1 - (1 - ax).pow(1.0 / (t + 1))
    else:
        y = 1 - (1 - ax).pow(abs(t) + 1)
    return anchor + y * mx * signs


def _color_adjust_legacy(denoised, t, anchor):
    anchor = anchor.to(dtype=denoised.dtype, device=denoised.device)
    anchor = anchor.reshape(-1)
    anchor = anchor.view((1, anchor.shape[0]) + (1,) * (denoised.dim() - 2))
    return torch.lerp(denoised, anchor, t)


def apply_contrast(x, alpha, neutral_anchor):
    if alpha == 0:
        return x
    return _color_adjust(x, -alpha, neutral_anchor)


def apply_color_shift(x, alpha, mode, color_anchor):
    if alpha == 0:
        return x
    fn = _color_adjust_legacy if mode == "legacy" else _color_adjust
    return fn(x, alpha, color_anchor)


def build_color_latent(vae, device, red, green, blue, brightness):
    """Encodes a flat-color 512x512 image and averages spatial (and
    temporal, if present) dims to get one anchor value per channel.
    device is host-specific, supplied by the caller."""
    img = torch.full((1, 512, 512, 3), 0.5, device=device)
    img[..., 0] += red
    img[..., 1] += green
    img[..., 2] += blue
    img += brightness
    latent = vae.encode(img)
    return latent.mean(dim=tuple(range(2, latent.dim())))[0]


# origin: aoleg/ComfyUI-Colorcraft@f00066c:lib_colorcraft/core.py:597-604 (fork signature; upstream's
# lib_colorcraft/color.py:53-60 reads ``dims`` off the VAE — see the header).
def to_model_space(latent_format, anchor, dims=2):
    """Lifts a per-channel VAE-space anchor into the model space the sampler
    actually works in. `dims` is the latent's spatial rank — 2 for Flux-family
    VAEs, 3 for the Wan-family ones (which carry a temporal axis)."""
    if latent_format is None or not hasattr(latent_format, "process_in"):
        return anchor
    shaped = anchor.view((1, anchor.shape[0]) + (1,) * dims)
    return latent_format.process_in(shaped)[0].reshape(-1)
