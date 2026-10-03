# Derived from (MIT License, notices below):
#   howardhx/speed@ca7801c9bdffe681742e9592345bcf4885959be5 utils.py:80-249 (kappa, align_timestep,
#     _dct_expand_np, _dwt_expand_np, _fft_expand_np) and comfyui/speed_sampler.py:124-148
#     (_initial_dct_downscale) — Copyright (c) 2026 Howard Xiao
#   aoleg/ComfyUI-SPEED@a8873591a27f2c1e086a2caf546f9b6aeec62b81 spectral_utils.py:130-254,
#     speed_core.py:303-374 — Copyright (c) 2026 A. Izzuddin Al Faruq (fork work by Oleg Afonin)
#   sorryhyun/ComfyUI-Spectrum-KSampler@b46a364aec3b161b889c9cc26cd976a49eb537ae
#     _vendor/networks/spd_core.py:20-124 (_dct_matrix cache, dct2/idct2, _snap, dct_lowpass_init,
#     spectral_expand) — Copyright (c) 2026 sorryhyun
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
# OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Modified by sam-extra, 2026-10-03 (this file is part of a GPL-3.0-only extension; the MIT notices
# above stay with the derived code):
#   * scipy/numpy/PyWavelets are gone: the DCT is a cached orthonormal DCT-II matrix built on the
#     latent's device (sorryhyun's cache idea), the FFT is ``torch.fft`` and the one-level Haar
#     synthesis is written out (pywt ``waverec2(..., "haar", mode="periodization")`` arithmetic);
#   * the maths runs in float64 on the device (float32 on MPS) so TF32 matmuls or half-precision
#     latents cannot erode the kept low-frequency block, and is cast back by the caller;
#   * noise is an argument (the runner draws it per image seed), not drawn here;
#   * the FFT fill uses ``sigma * (n_r + i n_i)`` without the ``/ sqrt(2)`` — the fix upstream made
#     in ca7801c9 (the real part then has variance sigma**2, like the DCT fill); aoleg's fork still
#     divides;
#   * ``amplitude_resize`` (new) resizes an x0 estimate keeping its magnitude — used when a run is
#     interrupted on the coarse grid so Forge still returns a full-size latent.
"""Pure-torch spectral primitives for SPEED (no scipy/numpy at runtime).

Conventions follow the paper and its three implementations:

* ``dct2``/``idct2`` are the orthonormal 2-D DCT-II over the last two axes, i.e.
  ``scipy.fft.dctn(x, type=2, norm="ortho")`` and its inverse (tests compare against scipy).
* ``dct_downscale`` is the paper's T_Phi: keep the top-left ``h x w`` block of the spectrum and
  invert on the small grid. Smooth content grows by ``sqrt(H W / (h w))`` while white noise keeps
  unit variance — the reason the expansion rescales by kappa.
* ``expand_dct``/``expand_fft``/``expand_haar`` embed the coarse spectrum in the larger grid and fill
  the new coefficients with ``sigma * noise``. They return the *un-rescaled* expansion; the caller
  multiplies by kappa (official order).
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch

__all__ = [
    "align_timestep",
    "amplitude_resize",
    "clear_basis_cache",
    "dct2",
    "dct_basis",
    "dct_downscale",
    "expand_dct",
    "expand_fft",
    "expand_haar",
    "idct2",
    "kappa",
    "respace_alignment",
    "snap",
    "work_dtype",
]

# A handful of grid sizes per generation (coarse and full, height and width). Bounded so a long
# session with many resolutions cannot keep stale device matrices alive.
_BASIS_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_BASIS_CACHE_LIMIT = 16


def work_dtype(device) -> torch.dtype:
    """float64 everywhere except MPS, which has no float64 kernels."""
    dev = torch.device(device) if device is not None else torch.device("cpu")
    return torch.float32 if dev.type == "mps" else torch.float64


def dct_basis(n: int, device=None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Orthonormal DCT-II matrix ``D`` (``D @ v`` transforms a column vector), cached per device.

    Built in float64 on the CPU and then moved, so every device/dtype gets the same rounding.
    Callers must not modify the returned tensor (it is shared, like sorryhyun's ``_DCT_CACHE``).
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"DCT size must be positive, got {n}")
    dev = torch.device(device) if device is not None else torch.device("cpu")
    key = (n, str(dev), dtype)
    cached = _BASIS_CACHE.get(key)
    if cached is not None:
        _BASIS_CACHE.move_to_end(key)
        return cached
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    i = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    matrix = torch.cos(math.pi * (2.0 * i + 1.0) * k / (2.0 * n)) * math.sqrt(2.0 / n)
    matrix[0] = matrix[0] / math.sqrt(2.0)
    matrix = matrix.to(device=dev, dtype=dtype)
    _BASIS_CACHE[key] = matrix
    while len(_BASIS_CACHE) > _BASIS_CACHE_LIMIT:
        _BASIS_CACHE.popitem(last=False)
    return matrix


def clear_basis_cache() -> None:
    _BASIS_CACHE.clear()


def dct2(x: torch.Tensor) -> torch.Tensor:
    """2-D orthonormal DCT-II over the last two axes (any leading axes), in ``x``'s dtype."""
    d_h = dct_basis(x.shape[-2], x.device, x.dtype)
    d_w = dct_basis(x.shape[-1], x.device, x.dtype)
    return d_h @ x @ d_w.transpose(0, 1)


def idct2(c: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`dct2`."""
    d_h = dct_basis(c.shape[-2], c.device, c.dtype)
    d_w = dct_basis(c.shape[-1], c.device, c.dtype)
    return d_h.transpose(0, 1) @ c @ d_w


def snap(value: float, multiple: int) -> int:
    """sorryhyun ``_snap``: nearest positive multiple of ``multiple`` (Python's round-half-even)."""
    return max(multiple, int(round(value / multiple)) * multiple)


def _work(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=work_dtype(x.device))


def dct_downscale(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """T_Phi: keep the low ``size`` block of the spectrum, invert on the small grid.

    Returns the work dtype (float64 off MPS). ``size`` must not exceed the input grid.
    """
    h, w = int(size[0]), int(size[1])
    big_h, big_w = int(x.shape[-2]), int(x.shape[-1])
    if h > big_h or w > big_w or h <= 0 or w <= 0:
        raise ValueError(f"DCT downscale: cannot go from {(big_h, big_w)} to {(h, w)}")
    work = _work(x)
    if (h, w) == (big_h, big_w):
        return work
    return idct2(dct2(work)[..., :h, :w])


def amplitude_resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize an x0 estimate through the DCT keeping its amplitude (``sqrt(hw/HW)`` correction).

    Upscaling zero-pads the spectrum (a band-limited interpolation), downscaling truncates it.
    Returns the work dtype.
    """
    h, w = int(size[0]), int(size[1])
    src_h, src_w = int(x.shape[-2]), int(x.shape[-1])
    work = _work(x)
    if (h, w) == (src_h, src_w):
        return work
    coeffs = dct2(work)
    out = torch.zeros(*work.shape[:-2], h, w, dtype=work.dtype, device=work.device)
    keep_h, keep_w = min(h, src_h), min(w, src_w)
    out[..., :keep_h, :keep_w] = coeffs[..., :keep_h, :keep_w]
    return idct2(out) * math.sqrt((h * w) / (src_h * src_w))


def _check_target(x: torch.Tensor, size: tuple[int, int], name: str) -> tuple[int, int, int, int]:
    big_h, big_w = int(size[0]), int(size[1])
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if big_h < h or big_w < w:
        raise ValueError(
            f"{name} expand: cannot expand to target {(big_h, big_w)} smaller than source ({h}, {w})."
        )
    return h, w, big_h, big_w


def expand_dct(x: torch.Tensor, size: tuple[int, int], sigma: float, noise: torch.Tensor) -> torch.Tensor:
    """DCT spectral noise expansion (official ``_dct_expand_np``, sorryhyun ``spectral_expand``).

    ``noise`` has the full target shape ``(..., H, W)``; its top-left ``h x w`` block is replaced by
    the source spectrum (upstream draws that block too and overwrites it), every other coefficient
    becomes ``sigma * noise``. Returns the work dtype, not yet rescaled by kappa.
    """
    h, w, big_h, big_w = _check_target(x, size, "DCT")
    work = _work(x)
    if tuple(noise.shape[-2:]) != (big_h, big_w) or noise.shape[:-2] != work.shape[:-2]:
        raise ValueError(f"DCT expand: noise {tuple(noise.shape)} does not fit {tuple(work.shape[:-2])}+{(big_h, big_w)}")
    big = noise.to(device=work.device, dtype=work.dtype) * float(sigma)
    big[..., :h, :w] = dct2(work)
    return idct2(big)


def expand_fft(
    x: torch.Tensor, size: tuple[int, int], sigma: float, noise_real: torch.Tensor, noise_imag: torch.Tensor,
) -> torch.Tensor:
    """FFT spectral noise expansion (official ``_fft_expand_np`` after ca7801c9).

    The centred source spectrum ``fftshift(fft2(x, ortho))`` replaces the centre of a full grid of
    ``sigma * (n_r + i n_i)`` (itself fftshifted, as upstream does); the real part of the inverse is
    returned. Both noise tensors have the full target shape.
    """
    h, w, big_h, big_w = _check_target(x, size, "FFT")
    work = _work(x)
    pad_h, pad_w = (big_h - h) // 2, (big_w - w) // 2
    dims = (-2, -1)
    real = noise_real.to(device=work.device, dtype=work.dtype)
    imag = noise_imag.to(device=work.device, dtype=work.dtype)
    x_src = torch.fft.fftshift(torch.fft.fft2(work, dim=dims, norm="ortho"), dim=dims)
    x_big = torch.fft.fftshift(torch.complex(real, imag) * float(sigma), dim=dims)
    x_big[..., pad_h:pad_h + h, pad_w:pad_w + w] = x_src
    return torch.fft.ifft2(torch.fft.ifftshift(x_big, dim=dims), dim=dims, norm="ortho").real


def expand_haar(
    x: torch.Tensor, sigma: float, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor,
) -> torch.Tensor:
    """One-level Haar synthesis with ``x`` as LL and ``sigma``-scaled noise as LH/HL/HH.

    Same arithmetic as ``pywt.waverec2([LL, (LH, HL, HH)], "haar", mode="periodization")`` (official
    ``_dwt_expand_np``); the output is exactly twice the source grid.
    """
    work = _work(x)
    s = float(sigma)
    a = work
    h = lh.to(device=work.device, dtype=work.dtype) * s
    v = hl.to(device=work.device, dtype=work.dtype) * s
    d = hh.to(device=work.device, dtype=work.dtype) * s
    rows, cols = int(work.shape[-2]), int(work.shape[-1])
    out = torch.empty(*work.shape[:-2], 2 * rows, 2 * cols, dtype=work.dtype, device=work.device)
    out[..., 0::2, 0::2] = (a + h + v + d) * 0.5
    out[..., 0::2, 1::2] = (a + h - v - d) * 0.5
    out[..., 1::2, 0::2] = (a - h + v - d) * 0.5
    out[..., 1::2, 1::2] = (a - h - v + d) * 0.5
    return out


def kappa(t: float, r: float) -> float:
    """State rescaling after the expansion, paper Eq. 5 (official ``kappa``)."""
    return r / (1.0 + (r - 1.0) * t)


def align_timestep(t: float, r: float) -> float:
    """Aligned flow time after the expansion, paper Eq. 6 (official ``align_timestep``: ``t * kappa``)."""
    return t * kappa(t, r)


def respace_alignment(sigma: float, r: float) -> tuple[float, float]:
    """``(sigma_aligned, kappa)`` written the way sorryhyun's ``spectral_expand`` computes them.

    Mathematically the same as :func:`align_timestep`/:func:`kappa`; kept separately because the
    respace plan reproduces sorryhyun's float arithmetic bit for bit.
    """
    sigma_aligned = (r * sigma) / (1.0 + (r - 1.0) * sigma)
    factor = r / (1.0 + (r - 1.0) * sigma)
    return sigma_aligned, factor
