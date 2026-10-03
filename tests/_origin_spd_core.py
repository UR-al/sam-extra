# Verbatim copy of sorryhyun's SPD spectral primitives for the respace-mode parity tests
# (tests/test_speed_origin.py).
#
# origin: sorryhyun/ComfyUI-Spectrum-KSampler@b46a364aec3b161b889c9cc26cd976a49eb537ae:
#         _vendor/networks/spd_core.py:1-124
# https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler - MIT License, Copyright (c) 2026 sorryhyun
# (notice below). Only this header was added; everything after the marker line is the upstream file
# unchanged (test_speed_origin.py pins its SHA-256). Test oracle only: the extension never imports it.
#
# MIT License
#
# Copyright (c) 2026 sorryhyun
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
# ---- upstream _vendor/networks/spd_core.py below (verbatim) ----
"""SPD spectral primitives — velocity-source-agnostic numerics core.

The 2D separable type-II DCT helpers and the SPD spectral-expansion geometry
(paper T_Φ + Eq. i–iii + Eq. 5–6, Xiao et al. arXiv:2605.18736). **Pure compute**
— torch only, no adapter / sampler / comfy imports — so this module is the single
source of truth shared verbatim between ``networks/spd.py`` (the CLI sampler +
fine-tune target construction) and the ComfyUI Spectrum "SPEED" node (vendored by
scripts/release/sync_vendor.py). The node previously hand-mirrored these and ran
without the matrix cache; vendoring this module gains it the cache (bit-identical).

DO NOT EDIT the vendored copy — regenerate it via scripts/release/sync_vendor.py.
"""

from __future__ import annotations

import math

import torch

# The type-II DCT basis is constant for a given (n, device, dtype), and both the
# SPD sampler and the fine-tune target construction only ever see a handful of
# bucket sizes — so build each matrix once and reuse it read-only instead of
# rebuilding dense (n×n) matrices on every dct2/idct2 call. Callers must NOT
# mutate the returned tensor (dct2/idct2 only matmul against it). Math is
# bit-identical to the un-cached construction.
_DCT_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _dct_matrix(n: int, device, dtype) -> torch.Tensor:
    key = (n, device, dtype)
    m = _DCT_CACHE.get(key)
    if m is None:
        nr = torch.arange(n, device=device, dtype=dtype)
        k = nr.unsqueeze(1)
        m = torch.cos(torch.pi * k * (2 * nr + 1) / (2 * n))
        m[0] *= 1.0 / math.sqrt(n)
        m[1:] *= math.sqrt(2.0 / n)
        _DCT_CACHE[key] = m
    return m


def dct2(x: torch.Tensor) -> torch.Tensor:
    """2D type-II DCT over the last two dims of a (B, C, H, W) tensor."""
    B, C, H, W = x.shape
    Dh = _dct_matrix(H, x.device, x.dtype)
    Dw = _dct_matrix(W, x.device, x.dtype)
    y = x.reshape(B * C, H, W)
    y = Dh @ y
    y = y @ Dw.T
    return y.reshape(B, C, H, W)


def idct2(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`dct2` (last two dims of a (B, C, H, W) tensor)."""
    B, C, H, W = x.shape
    Dh = _dct_matrix(H, x.device, x.dtype)
    Dw = _dct_matrix(W, x.device, x.dtype)
    y = x.reshape(B * C, H, W)
    y = Dh.T @ y
    y = y @ Dw
    return y.reshape(B, C, H, W)


def _snap(v: float, mult: int) -> int:
    """Round to nearest positive multiple of ``mult`` (DiT patch_spatial)."""
    return max(mult, int(round(v / mult)) * mult)


def dct_lowpass_init(x5: torch.Tensor, scale: float, patch: int) -> torch.Tensor:
    """DCT low-pass of a (B,C,1,H,W) latent down to a (B,C,1,h,w) grid (paper T_Φ)."""
    B, C, T, H, W = x5.shape
    x4 = x5.squeeze(2).float()
    xi = dct2(x4)
    h = min(_snap(H * scale, patch), H)
    w = min(_snap(W * scale, patch), W)
    x_low = idct2(xi[:, :, :h, :w])
    return x_low.unsqueeze(2).to(x5.dtype)


def spectral_expand(
    x5: torch.Tensor,
    sigma_val: float,
    scale_lo: float,
    scale_hi: float,
    H_full: int,
    W_full: int,
    patch: int,
    gen: torch.Generator,
    hf_scale: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Embed the current low-res DCT block into a larger grid, fill HF slots with
    σ-scaled noise, iDCT, scale by κ (Eq. iii) and align the timestep (Eq. 5–6).

    ``hf_scale`` attenuates the fresh HF noise fill (paper prescription = 1.0).
    It is the continuity↔detail knob the SPD∘Spectrum frontier probe sweeps:
    γ→0 injects no fresh HF (max LL-feature continuity across the seam, but an
    off-manifold under-detailed state); γ=1 is the on-manifold paper default.
    Default 1.0 ⇒ bit-for-bit identical to the sampler/training path.

    Returns (expanded (B,C,1,h_hi,w_hi) latent, sigma_aligned).
    """
    B, C, T, h_lo, w_lo = x5.shape
    x4 = x5.squeeze(2).float()
    xi = dct2(x4)

    h_hi = max(_snap(H_full * scale_hi, patch), h_lo)
    w_hi = max(_snap(W_full * scale_hi, patch), w_lo)

    r = scale_hi / scale_lo
    sigma_aligned = (r * sigma_val) / (1.0 + (r - 1.0) * sigma_val)
    kappa = r / (1.0 + (r - 1.0) * sigma_val)

    xi_new = torch.zeros(B, C, h_hi, w_hi, device=x5.device, dtype=torch.float32)
    xi_new[:, :, :h_lo, :w_lo] = xi
    noise = torch.randn(
        xi_new.shape, generator=gen, device=x5.device, dtype=torch.float32
    )
    mask = torch.zeros_like(xi_new)
    mask[:, :, h_lo:, :] = 1.0
    mask[:, :, :h_lo, w_lo:] = 1.0
    xi_new = xi_new + mask * sigma_val * noise * hf_scale

    x4_new = idct2(xi_new) * kappa
    return x4_new.unsqueeze(2).to(x5.dtype), float(sigma_aligned)
