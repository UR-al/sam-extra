"""DAVE DC attenuation math for Anima block outputs."""

from __future__ import annotations

import torch


def apply_dave(output: torch.Tensor, attenuation: float) -> torch.Tensor:
    """Return ``output - attenuation * spatial/token mean(output)``."""
    if attenuation == 0.0:
        return output
    if output.ndim < 3:
        raise ValueError(f"DAVE expects token/spatial features, got {output.ndim}D")
    dims = tuple(range(1, output.ndim - 1)) or (1,)
    mean = output.float().mean(dim=dims, keepdim=True)
    return (output.float() - float(attenuation) * mean).to(output.dtype)
