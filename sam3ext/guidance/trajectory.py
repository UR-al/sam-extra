"""Per-step x0 trajectory of a sampling pass, looked up by sigma.

HiFlow aligns the hires pass with the trajectory of the low-resolution pass that made its init
image. The base pass records the guided x0 of every sampler step (one entry per step-start sigma);
the hires pass asks for the record at its own sigma. Hires runs its own schedule (``Hires steps``,
``Hires Shift``, denoise < 1), so its sigmas are generally not on the base schedule: ``at`` linearly
interpolates between the two neighbouring records in sigma (rectified flow is linear in t = σ), and
clamps outside the recorded range.

Records live on the CPU in float16 (a 1 MP Anima latent is ~0.5 MB per step and batch row) and the
store belongs to one request; the caller clears it when the request ends.
"""

from __future__ import annotations

import bisect

import torch

__all__ = ["Trajectory"]


class Trajectory:
    """Sigma-keyed x0 records of one pass. Re-recording a sigma replaces the old record."""

    __slots__ = ("_sigmas", "_records", "shape", "storage_dtype")

    def __init__(self, storage_dtype: torch.dtype = torch.float16):
        self._sigmas: list[float] = []   # ascending
        self._records: list[torch.Tensor] = []
        self.shape: tuple | None = None
        self.storage_dtype = storage_dtype

    def __len__(self) -> int:
        return len(self._sigmas)

    def clear(self) -> None:
        self._sigmas.clear()
        self._records.clear()
        self.shape = None

    @property
    def sigmas(self) -> tuple[float, ...]:
        return tuple(self._sigmas)

    def record(self, sigma: float, x0: torch.Tensor) -> None:
        """Keep ``x0`` (detached, CPU, ``storage_dtype``) for ``sigma``. A new shape restarts the store."""
        sigma = float(sigma)
        shape = tuple(x0.shape)
        if self.shape is not None and shape != self.shape:
            self.clear()
        self.shape = shape
        stored = x0.detach().to(device="cpu", dtype=self.storage_dtype).clone()
        index = bisect.bisect_left(self._sigmas, sigma)
        if index < len(self._sigmas) and abs(self._sigmas[index] - sigma) <= 1e-9 * max(1.0, abs(sigma)):
            self._records[index] = stored
            return
        self._sigmas.insert(index, sigma)
        self._records.insert(index, stored)

    def at(self, sigma: float, *, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
        """x0 at ``sigma``: linear in sigma between neighbours, clamped at the ends. None if empty."""
        if not self._sigmas:
            return None
        sigma = float(sigma)
        sigmas = self._sigmas
        if sigma <= sigmas[0]:
            out = self._records[0].to(dtype=dtype)
        elif sigma >= sigmas[-1]:
            out = self._records[-1].to(dtype=dtype)
        else:
            hi = bisect.bisect_left(sigmas, sigma)
            lo = hi - 1
            s0, s1 = sigmas[lo], sigmas[hi]
            weight = 0.0 if s1 == s0 else (sigma - s0) / (s1 - s0)
            a = self._records[lo].to(dtype=dtype)
            b = self._records[hi].to(dtype=dtype)
            out = torch.lerp(a, b, weight)
        if device is not None:
            out = out.to(device=device)
        return out
