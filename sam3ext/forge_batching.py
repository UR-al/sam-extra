"""Forge's "Batch Cond/Uncond" option, read the same way on Forge Neo 2.29.2 and 2.30.

Forge 2.30 (45f41b81) added Settings → Optimizations → **Batch Cond/Uncond** (``opts.batch_cond_uncond``,
on by default) and made ``modules.shared.batch_cond_uncond`` a property over it
(``modules/shared_items.py``). ``backend/sampling/sampling_function.py`` reads it twice:

* ``can_concat_cond`` — off: cond and uncond never share a UNet call (one call each, cond first);
* ``sampling_prepare`` — the inference memory Forge reserves for ``(2 if on else 1) * batch`` rows.

Forge 2.29.2 (d70373eb) keeps the old constant ``shared.batch_cond_uncond = True`` and always batches
(memory for ``2 * batch``). ``getattr(shared, "batch_cond_uncond", True)`` gives the right answer on both;
anything unreadable (outside Forge, before the settings load, a stub) counts as on — Forge's default and
the only behaviour 2.29.2 has.
"""

from __future__ import annotations

__all__ = ["batch_cond_uncond", "unet_rows_per_image"]


def batch_cond_uncond(shared=None) -> bool:
    """Does Forge run cond and uncond in one UNet call? ``shared``: ``modules.shared`` (imported when None)."""
    try:
        if shared is None:
            from modules import shared
        return bool(getattr(shared, "batch_cond_uncond", True))
    except Exception:
        return True


def unet_rows_per_image(shared=None) -> int:
    """Rows per image Forge's ``sampling_prepare`` reserves inference memory for: 2 batched, 1 split."""
    return 2 if batch_cond_uncond(shared) else 1
