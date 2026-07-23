"""Small, dependency-free 2-D Haar helpers for 4-D/5-D latent tensors."""

from __future__ import annotations

import torch
import torch.nn.functional as F


_FP8_DTYPES = {
    dtype
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    )
    if (dtype := getattr(torch, name, None)) is not None
}


def safe_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.bfloat16 if dtype in _FP8_DTYPES else dtype


def pad_even(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad only H/W to even sizes and return the original spatial size."""
    if x.ndim < 4:
        raise ValueError(f"Haar expects a 4-D or 5-D latent, got {tuple(x.shape)}")
    height, width = int(x.shape[-2]), int(x.shape[-1])
    pad_h, pad_w = height % 2, width % 2
    if not (pad_h or pad_w):
        return x, (height, width)

    # Reflect padding on a 5-D tensor requires an explicit untouched T pair.
    pad = (0, pad_w, 0, pad_h) + (0, 0) * max(0, x.ndim - 4)
    mode = "reflect" if height > 1 and width > 1 else "replicate"
    try:
        return F.pad(x, pad, mode=mode), (height, width)
    except RuntimeError:
        return F.pad(x, pad, mode="constant", value=0.0), (height, width)


def haar_dwt2d(x: torch.Tensor):
    """Return LL/LH/HL/HH bands, preserving all leading dimensions."""
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("Haar DWT requires even H/W; call pad_even first")
    low_h = (x[..., 0::2, :] + x[..., 1::2, :]) * 0.5
    high_h = (x[..., 0::2, :] - x[..., 1::2, :]) * 0.5
    ll = (low_h[..., 0::2] + low_h[..., 1::2]) * 0.5
    lh = (low_h[..., 0::2] - low_h[..., 1::2]) * 0.5
    hl = (high_h[..., 0::2] + high_h[..., 1::2]) * 0.5
    hh = (high_h[..., 0::2] - high_h[..., 1::2]) * 0.5
    return ll, lh, hl, hh


def haar_idwt2d(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
) -> torch.Tensor:
    """Invert :func:`haar_dwt2d` for arbitrary leading dimensions."""
    if not (ll.shape == lh.shape == hl.shape == hh.shape):
        raise ValueError("all Haar subbands must have identical shapes")
    *leading, height, width = ll.shape
    low_h = torch.empty(
        *leading, height, width * 2, device=ll.device, dtype=ll.dtype
    )
    high_h = torch.empty_like(low_h)
    low_h[..., 0::2] = ll + lh
    low_h[..., 1::2] = ll - lh
    high_h[..., 0::2] = hl + hh
    high_h[..., 1::2] = hl - hh

    out = torch.empty(
        *leading, height * 2, width * 2, device=ll.device, dtype=ll.dtype
    )
    out[..., 0::2, :] = low_h + high_h
    out[..., 1::2, :] = low_h - high_h
    return out


def sigma_norm(sigma, like: torch.Tensor):
    """Source-compatible ``sigma / (sigma + 1)`` broadcast to ``like``."""
    if torch.is_tensor(sigma):
        value = sigma.float() / (sigma.float() + 1.0)
        if value.ndim == 1:
            value = value.view(-1, *([1] * (like.ndim - 1)))
        return value.to(device=like.device, dtype=safe_compute_dtype(like.dtype))
    value = float(sigma)
    return value / (value + 1.0)
