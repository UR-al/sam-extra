# BSD 3-Clause License
#
# Copyright (c) 2024, Clybius
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""DPM++ 4M SDE — Clybius' fourth-order multistep DPM-Solver++ SDE, with Forge's flow-model handling.

Origin: Clybius/ComfyUI-Extra-Samplers@52eac1b7c847d2727e0ca93ca26d9ffd77029daa ``extra_samplers.py``
``sample_clyb_4m_sde_momentumized`` (lines 435-524), registered by that pack as the KSampler entry
``clyb_4m_sde_momentumized`` through ``sample_clyb_4m_sde`` (lines 680-681: momentum 0.0 and a
``BrownianTreeNoiseSampler`` from ``get_noise_sampler``) with the penultimate sigma discarded
(lines 1312-1326). BSD-3-Clause, notice above; the full licence is also in THIRD_PARTY_NOTICES.md.
Background: DPM-Solver++ (Lu et al., arXiv:2211.01095) and the k-diffusion/ComfyUI DPM++ 2M/3M SDE
samplers it extends with one more history term.

Modified by sam-extra, 2026-10-03 (the BSD licence asks for nothing more; listed for the reader):

* **Flow models.** Written in half-log-SNR λ = log(α/σ) like Forge's ``sample_dpmpp_3m_sde``
  (``sigma_to_half_log_snr``, ``offset_first_sigma_for_snr``, ``alpha_t = σ_t·e^{λ_t}``): the step is
  ``x ← (σ_t/σ_s)·e^{−hη}·x + α_t·(1 − e^{−h(1+η)})·x0`` and every history correction is scaled by
  ``α_t``. On an epsilon/v model λ = −log σ and α_t = 1, which is Clybius' update; on a rectified
  flow model it is Clybius' update in epsilon-equivalent coordinates (x/α, σ/α) mapped back.
  The 1st-3rd order steps are Forge's ``sample_dpmpp_3m_sde`` term for term, so the first three
  steps of the two samplers are identical.
* **Momentum** is not ported. ``sample_clyb_4m_sde`` runs it at 0.0, where upstream's
  ``momentum_func`` returns its ``diff`` argument unchanged.
* **Noise** is added when ``eta > 0 and s_noise > 0`` (Forge's 3M SDE rule; upstream: ``if eta:``,
  which adds ``0·noise`` when ``s_noise`` is 0). In a Forge generation the sampler option
  ``brownian_noise`` hands in Forge's tree seeded per image (``Sampler.create_noise_sampler``), as
  for DPM++ 3M SDE. Called without one, a CPU ``BrownianTreeNoiseSampler`` seeded from
  ``extra_args["seed"]`` is built like Forge's 3M SDE. (Upstream's ``sample_clyb_4m_sde`` calls
  ``get_noise_sampler(x, sigmas, noise_sampler_type, noise_sampler, extra_args)`` positionally, so
  its ``extra_args`` slot receives ``noise_sampler`` (None) and the tree gets no seed.)
* Forge's ``trange`` and k-diffusion helpers; the unused ``time``/``sigma_min`` bookkeeping of the
  momentum term is dropped. The order-selection branches, ratios, ``d1``/``d2`` and φ expressions
  are upstream's.
"""

from __future__ import annotations

from functools import partial

import torch

from .common import k_sampling, model_sampling

__all__ = ["sample_dpmpp_4m_sde"]


@torch.no_grad()
def sample_dpmpp_4m_sde(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    eta=1.0, s_noise=1.0, noise_sampler=None,
):
    """DPM-Solver++(4M) SDE (Clybius, momentum 0) — epsilon, v and rectified-flow models."""
    if len(sigmas) <= 1:
        return x

    ks = k_sampling()
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    if noise_sampler is None:
        noise_sampler = ks.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True)
    s_in = x.new_ones([x.shape[0]])

    sampling = model_sampling(model)
    lambda_fn = partial(ks.sigma_to_half_log_snr, model_sampling=sampling)
    sigmas = ks.offset_first_sigma_for_snr(sigmas, sampling)

    denoised_1, denoised_2, denoised_3 = None, None, None
    h_1, h_2, h_3 = None, None, None
    for i in ks.trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1)

            alpha_t = sigmas[i + 1] * lambda_t.exp()

            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised

            if h_3 is not None:
                # DPM-Solver++(4M) SDE — Clybius' extrapolation over three history differences.
                r0 = h_3 / h_2
                r1 = h_2 / h
                r2 = h / h_1
                d1_0 = (denoised - denoised_1) / r2
                d1_1 = (denoised_1 - denoised_2) / r1
                d1_2 = (denoised_2 - denoised_3) / r0
                d1 = d1_0 + (d1_0 - d1_1) * r2 / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r2 + r1) * (r0 + r1))
                d2 = (d1_0 - d1_1) / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) / ((r2 + r1) * (r0 + r1))
                phi_3 = h_eta.neg().expm1() / h_eta + 1
                phi_4 = phi_3 / h_eta - 0.5
                x = x + (alpha_t * phi_3) * d1 - (alpha_t * phi_4) * d2
            elif h_2 is not None:
                # DPM-Solver++(3M) SDE
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2
            elif h_1 is not None:
                # DPM-Solver++(2M) SDE
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + (alpha_t * phi_2) * d

            if eta > 0 and s_noise > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

            denoised_1, denoised_2, denoised_3 = denoised, denoised_1, denoised_2
            h_1, h_2, h_3 = h, h_1, h_2
    return x
