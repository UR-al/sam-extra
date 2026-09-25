"""DAVE early-step gate, ported from the Anima DAVE ComfyUI node.

origin: sorryhyun/ComfyUI-Anima-DAVE@83143e8d84768e25f72755ec00ea00ded07ee06e:nodes.py:91-106, 159-208
(MIT License, Copyright (c) 2026 Seunghyun Ji — full notice in THIRD_PARTY_NOTICES.md)

The node decides once per model forward whether the pooled blocks get the DC
attenuation:

* ``tau <= 0`` runs on every step (:199-200, 207-208);
* otherwise the forward's sigma is looked up in the sampler's sigma schedule —
  the *first* entry ``isclose(rtol=1e-4, atol=1e-6)`` to it, or index 0 when
  none is (:101-105). A sigma that is not on the schedule (a second-order
  sampler's midpoint, s_churn) therefore counts as step 0 and is always on;
* with ``n = len(schedule) - 1`` it is on while ``step < k``,
  ``k = max(1, min(n, round(tau * n)))`` (:205-206) — Python ``round``, so
  tau 0.1 at 25 steps is ``round(2.5) = 2``;
* without a schedule (missing, or fewer than two entries) it is on (:98-100,
  202-203).

A block is pooled when ``clip(strength * w, 0, 1) > 1e-3`` (:165-166); with the
shipped mask (``dave_alpha.npz``: w = 1 on blocks 8-18 of 28, 0 elsewhere) that
is ``DEFAULT_BLOCKS`` and ``attenuation_active(strength)``.

Host seam: ComfyUI publishes the schedule it samples as
``transformer_options['sample_sigmas']``. Forge stores its whole list in
``transformer_options['sampling_sigmas']`` (modules/sd_samplers_kdiffusion.py
``sample``/``sample_img2img``, :192, :246). txt2img's first pass walks all of
it; img2img and the hires pass walk ``sigmas[steps - t_enc - 1:]`` (:148).
``sampled_schedule`` takes that same slice from the offset the caller worked
out with Forge's own ``setup_img2img_steps``, which gives the list ComfyUI
would publish for the same run. So the step index and ``n`` count the steps
that actually run. Counting back ``shared.state.sampling_steps + 1`` entries
from the end would be wrong. Some schedulers return more than ``steps + 1``
sigmas; Forge's ``ddim_scheduler`` returns ``steps + 2`` at 24, 28, 30 and 32
steps on Anima. Counting from the end would then drop the first sigma the
sampler visits.

Changes from the original, all at that seam: the schedule and sigma are passed
in instead of read from ``transformer_options``; the sigma is cast to the
schedule's dtype (a no-op when they match — the original raises on a
mismatch); a missing sigma counts as a missing schedule; a plain list is
matched with the same ``isclose`` formula in Python; and ``ForwardGateCache``
memoises the decision so every pooled block of one forward shares one lookup.
"""

from __future__ import annotations

__all__ = [
    "ACTIVE_ATTENUATION",
    "DEFAULT_BLOCKS",
    "ISCLOSE_ATOL",
    "ISCLOSE_RTOL",
    "ForwardGateCache",
    "attenuation_active",
    "current_step",
    "gate_active",
    "sampled_schedule",
    "step_gate",
    "window_steps",
]

# origin :101-103 — ComfyUI's own isclose(sample_sigmas, sigma_now) mapping.
ISCLOSE_RTOL = 1e-4
ISCLOSE_ATOL = 1e-6
# origin :166 — a block with atten <= 1e-3 is not pooled.
ACTIVE_ATTENUATION = 1e-3
# origin: dave_alpha.npz ``weight`` — 1.0 on flat blocks 8..18, 0.0 elsewhere.
DEFAULT_BLOCKS = "8-18"


def _is_tensor(value) -> bool:
    return hasattr(value, "flatten") and hasattr(value, "shape") and hasattr(value, "dtype")


def attenuation_active(strength) -> bool:
    """``clip(strength * 1.0, 0, 1) > 1e-3`` — a default-mask block is pooled (origin :165-166)."""
    atten = min(1.0, max(0.0, float(strength)))
    return atten > ACTIVE_ATTENUATION


def sampled_schedule(schedule, offset=None):
    """The part of Forge's ``sampling_sigmas`` the sampler walks: ``schedule[offset:]``.

    ``offset`` is ``steps - t_enc - 1`` for img2img and the hires pass, the
    same expression as ``sigma_sched = sigmas[steps - t_enc - 1:]``
    (modules/sd_samplers_kdiffusion.py:148). It is 0 for txt2img's first pass.
    ``None`` or 0 returns the schedule unchanged.
    """
    if schedule is None or offset is None:
        return schedule
    try:
        offset = int(offset)
    except (TypeError, ValueError):
        return schedule
    if offset == 0:
        return schedule
    return schedule[offset:]


def current_step(schedule, sigma):
    """``(step_index, n_steps)`` of ``sigma`` on ``schedule``, or ``(None, None)`` (origin :91-106).

    The first ``isclose`` entry, else 0; ``n_steps = len(schedule) - 1``.
    """
    if schedule is None or len(schedule) < 2 or sigma is None:
        return None, None
    n_steps = len(schedule) - 1
    if _is_tensor(schedule):
        import torch

        if torch.is_tensor(sigma):
            cur0 = sigma.flatten()[0].to(device=schedule.device, dtype=schedule.dtype)
        else:
            cur0 = torch.tensor(float(sigma), device=schedule.device, dtype=schedule.dtype)
        matches = torch.isclose(schedule, cur0, rtol=ISCLOSE_RTOL, atol=ISCLOSE_ATOL)
        nz = torch.nonzero(matches).flatten()
        step = int(nz[0].item()) if nz.numel() else 0
        return step, n_steps
    if _is_tensor(sigma):
        cur0 = float(sigma.flatten()[0].item())
    else:
        cur0 = float(sigma)
    for index, value in enumerate(schedule):
        # torch.isclose(input, other): |input - other| <= atol + rtol * |other|
        if abs(float(value) - cur0) <= ISCLOSE_ATOL + ISCLOSE_RTOL * abs(cur0):
            return index, n_steps
    return 0, n_steps


def window_steps(tau, n_steps) -> int:
    """``k = max(1, min(n, round(tau * n)))`` — how many leading steps run (origin :205)."""
    n_steps = int(n_steps)
    return max(1, min(n_steps, round(float(tau) * n_steps)))


def step_gate(step, n_steps, tau) -> bool:
    """``tau <= 0`` or ``step < k`` for a known step index (origin :199-208)."""
    if float(tau) <= 0.0:
        return True
    return int(step) < window_steps(tau, n_steps)


def gate_active(tau, schedule, sigma, offset=None) -> bool:
    """Is DAVE on for the forward at ``sigma``? (origin :199-208)

    ``offset`` picks the walked part of Forge's list (``sampled_schedule``).
    """
    if float(tau) <= 0.0:
        return True
    step, n_steps = current_step(sampled_schedule(schedule, offset), sigma)
    if step is None:  # no schedule published → every step (origin :202-203)
        return True
    return step < window_steps(tau, n_steps)


class ForwardGateCache:
    """One-entry memo of ``gate_active`` for the blocks of one forward.

    The key holds the schedule and sigma objects themselves (compared with
    ``is``), so their ids cannot be reused while cached; a new forward brings a
    new sigma tensor and recomputes. The original decides once per forward in
    its APPLY_MODEL wrapper — this keeps the per-block Forge wrapper at one
    lookup (and one device sync) per forward.
    """

    __slots__ = ("_key", "_value", "lookups")

    def __init__(self):
        self._key = None
        self._value = False
        self.lookups = 0

    def clear(self) -> None:
        self._key = None
        self._value = False

    def active(self, tau, schedule, sigma, offset=None) -> bool:
        key = self._key
        if (
            key is not None
            and key[0] is schedule
            and key[1] is sigma
            and key[2] == offset
            and key[3] == tau
        ):
            return self._value
        value = bool(gate_active(tau, schedule, sigma, offset))
        self.lookups += 1
        self._key = (schedule, sigma, offset, tau)
        self._value = value
        return value
