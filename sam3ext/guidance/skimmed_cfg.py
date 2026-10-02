"""Skimmed CFG maths and sigma gate, vendored from Extraltodeus/Skimmed_CFG.

Source: https://github.com/Extraltodeus/Skimmed_CFG
        commit d83005832ac42783adfd6f4ae96f6ef6406d1a74, file ``skimmed_CFG.py``
Licence: Apache License 2.0 (upstream ``LICENSE``; notice in this extension's
         ``THIRD_PARTY_NOTICES.md``, full text at
         https://www.apache.org/licenses/LICENSE-2.0).

What is copied, and what was changed (Apache-2.0 section 4(b)):

* ``get_skimming_mask`` and ``skimmed_CFG`` are upstream ``skimmed_CFG.py``
  lines 9-54, copied unmodified.
* ``skim_sigmas``, ``skim_active``, ``flip_filter_at`` and ``skim_pair`` are the
  body of upstream ``CFG_Skimming_Single_Scale_Pre_CFG.execute`` and its
  ``pre_cfg_patch`` closure (lines 152-155 and 160-197), split into functions so
  the Forge post-CFG hook in ``scripts/anima_skimmed_cfg.py`` can call them. The
  conditions, their order, the strict inequalities and the arithmetic are
  upstream's; only the function boundaries are new. Upstream's first check
  (``not torch.any(conds_out[1])``) is done by the caller.

Modified by the sam-extra authors, 2026-09-25.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Upstream skimmed_CFG.py:9-54 — copied unmodified (Apache-2.0).
# ---------------------------------------------------------------------------


@torch.no_grad()
def get_skimming_mask(
    x_orig,
    cond,
    uncond,
    cond_scale,
    return_denoised=False,
    disable_flipping_filter=False,
):
    denoised = x_orig - (
        (x_orig - uncond) + cond_scale * ((x_orig - cond) - (x_orig - uncond))
    )
    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    matching_diff_after = (
        cond.sign() == (cond * cond_scale - uncond * (cond_scale - 1)).sign()
    )

    if disable_flipping_filter:
        outer_influence = matching_pred_signs & matching_diff_after
    else:
        deviation_influence = denoised.sign() == (denoised - x_orig).sign()
        outer_influence = (
            matching_pred_signs & matching_diff_after & deviation_influence
        )

    if return_denoised:
        return outer_influence, denoised
    else:
        return outer_influence


@torch.no_grad()
def skimmed_CFG(
    x_orig, cond, uncond, cond_scale, skimming_scale, disable_flipping_filter=False
):
    outer_influence, denoised = get_skimming_mask(
        x_orig, cond, uncond, cond_scale, True, disable_flipping_filter
    )
    low_cfg_denoised_outer = x_orig - (
        (x_orig - uncond) + skimming_scale * ((x_orig - cond) - (x_orig - uncond))
    )
    low_cfg_denoised_outer_difference = denoised - low_cfg_denoised_outer
    cond[outer_influence] = cond[outer_influence] - (
        low_cfg_denoised_outer_difference[outer_influence] / cond_scale
    )
    return cond


# ---------------------------------------------------------------------------
# Upstream CFG_Skimming_Single_Scale_Pre_CFG (skimmed_CFG.py:152-197), split
# into functions. Same conditions, order and arithmetic.
# ---------------------------------------------------------------------------


def skim_sigmas(percent_to_sigma, start_at_percentage, end_at_percentage,
                flip_at_percentage):
    """``(start_sigma, end_sigma, flip_sigma)`` — upstream lines 152-155.

    ``percent_to_sigma`` is the model's own schedule mapping (ComfyUI
    ``model_sampling.percent_to_sigma``, Forge ``KModel.predictor``). The
    percentages are passed through unchanged, exactly as upstream does.
    """
    start_at_sigma = percent_to_sigma(start_at_percentage)
    end_at_sigma = percent_to_sigma(end_at_percentage)
    flip_at_sigma = percent_to_sigma(flip_at_percentage)
    return float(start_at_sigma), float(end_at_sigma), float(flip_at_sigma)


def skim_active(sigma, start_at_sigma, end_at_sigma):
    """Upstream lines 166-170 (sigma part): skim only strictly inside the window.

    ``sigma <= end_at_sigma or sigma >= start_at_sigma`` returns early, so the
    bounds themselves are excluded. On a flow model ``percent_to_sigma(0)`` is
    1.0 and so is the first sampled sigma, which is why upstream never skims
    the first step there.
    """
    return not (sigma <= end_at_sigma or sigma >= start_at_sigma)


def flip_filter_at(sigma, disable_flipping_filter, flip_at_percentage,
                   flip_at_sigma):
    """Upstream lines 174-176: the filter is flipped *before* the flip point."""
    flip_filter = disable_flipping_filter
    if flip_at_percentage > 0 and sigma > flip_at_sigma:
        flip_filter = not disable_flipping_filter
    return flip_filter


def skim_pair(x_orig, cond, uncond, cond_scale, skimming_cfg,
              full_skim_negative, flip_filter):
    """Upstream lines 172 and 178-196: skim the negative, then the positive.

    Like upstream this rewrites ``uncond`` and then ``cond`` in place (the
    positive is skimmed against the already-skimmed negative at
    ``cond_scale - 1``) and returns ``(cond, uncond)``.
    """
    practical_scale = cond_scale if skimming_cfg < 0 else skimming_cfg

    uncond = skimmed_CFG(
        x_orig,
        uncond,
        cond,
        cond_scale,
        practical_scale if not full_skim_negative else 0,
        flip_filter,
    )
    cond = skimmed_CFG(
        x_orig,
        cond,
        uncond,
        cond_scale - 1,
        practical_scale,
        flip_filter,
    )
    return cond, uncond
