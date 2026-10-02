"""CNS wavelet recoloring of an existing noise sample.

origin: namemechan/comfyui-cns_sampler_patch@42278b138284f7a8685ef174af0a50fe03246dd0
:cns_sampler_patch.py:160-288 (``_subband_energy`` / ``color_noise_wavelet``), GPL-3.0 — the
same license as this extension. Steps 1-8 below follow the upstream function line by line;
the Haar helpers are this extension's ``haar.py`` (the same arithmetic as upstream :77-156).
Differences: no DEBUG_LOGGING block, and ``x_t`` is moved to the noise device (Forge can keep
the live latent and the seeded noise on different devices; upstream never moves it).
``tests/test_cns_origin.py`` runs the upstream function next to this one.
"""

from __future__ import annotations

import torch

from .haar import haar_dwt2d, haar_idwt2d, pad_even


def _energy(band: torch.Tensor) -> torch.Tensor:
    # origin: cns_sampler_patch.py:160-166 (_subband_energy)
    dims = tuple(range(1, band.ndim))
    return band.float().pow(2).mean(dim=dims, keepdim=True).clamp(min=1e-8)


def color_noise_wavelet(
    noise: torch.Tensor,
    x_t: torch.Tensor,
    strength: float = 1.0,
    gamma_power: float = 0.5,
    gamma_scale: float = 2.0,
) -> torch.Tensor:
    """Recolor ``noise`` by the live ``x_t`` Haar band energies (upstream defaults).

    The colored noise gets the input's global std back (step 7); a ``strength`` below 1
    then mixes it with the white noise by plain ``lerp`` and does *not* renormalise
    again, exactly like upstream (so 0 < strength < 1 lowers the std slightly).
    """
    if strength == 0.0:
        return noise

    orig_dtype = noise.dtype
    noise_f = noise.float()
    x_f = x_t.to(device=noise_f.device).float()

    # Step 1: Haar DWT on x_t -> subband energies (origin :218-230)
    x_p, (height, width) = pad_even(x_f)
    ll_x, lh_x, hl_x, hh_x = haar_dwt2d(x_p)
    e_ll = _energy(ll_x)
    e_lh = _energy(lh_x)
    e_hl = _energy(hl_x)
    e_hh = _energy(hh_x)
    e_sum = e_ll + e_lh + e_hl + e_hh

    # Step 2: gamma proxy = energy fraction / gamma_scale (origin :232-237)
    g_ll = (e_ll / e_sum / gamma_scale).clamp(0.0, 1.0)
    g_lh = (e_lh / e_sum / gamma_scale).clamp(0.0, 1.0)
    g_hl = (e_hl / e_sum / gamma_scale).clamp(0.0, 1.0)
    g_hh = (e_hh / e_sum / gamma_scale).clamp(0.0, 1.0)

    # Step 3: deficit = 1 - gamma (origin :239-243)
    d_ll = (1.0 - g_ll).clamp(min=1e-8)
    d_lh = (1.0 - g_lh).clamp(min=1e-8)
    d_hl = (1.0 - g_hl).clamp(min=1e-8)
    d_hh = (1.0 - g_hh).clamp(min=1e-8)

    # Step 4: beta = deficit ** gamma_power (origin :245-249)
    b_ll = d_ll.pow(gamma_power)
    b_lh = d_lh.pow(gamma_power)
    b_hl = d_hl.pow(gamma_power)
    b_hh = d_hh.pow(gamma_power)

    # Step 5: RMS normalisation, mean(beta^2) = 1 (origin :251-260)
    rms = ((b_ll**2 + b_lh**2 + b_hl**2 + b_hh**2) / 4.0).sqrt().clamp(min=1e-8)
    b_ll = b_ll / rms
    b_lh = b_lh / rms
    b_hl = b_hl / rms
    b_hh = b_hh / rms

    # Step 6: beta on the noise subbands, IDWT, crop (origin :262-272)
    n_p, _ = pad_even(noise_f)
    n_ll, n_lh, n_hl, n_hh = haar_dwt2d(n_p)
    colored = haar_idwt2d(
        n_ll * b_ll, n_lh * b_lh, n_hl * b_hl, n_hh * b_hh
    )[..., :height, :width]

    # Step 7: crop-energy correction to the input std (origin :274-282)
    orig_std = noise_f.std().clamp(min=1e-8)
    colored = colored * (orig_std / colored.std().clamp(min=1e-8))

    # Step 8: strength lerp, no renormalisation afterwards (origin :284-288)
    if strength < 1.0:
        colored = torch.lerp(noise_f, colored, strength)

    return colored.to(dtype=orig_dtype)
