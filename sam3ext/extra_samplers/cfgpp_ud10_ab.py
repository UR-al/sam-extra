"""CFG++ UD10 AB — ComfyUI's CFG++ Euler with a variable-step AB2 history and a σ = 0 extrapolation.

Ported from ComfyUI ``comfy/k_diffusion/sampling.py`` ``_sample_cfgpp_history`` and
``sample_cfgpp_ud10_ab`` (comfyanonymous/ComfyUI@3ac5d7941dfa2504555260512132d1cb5664648d, PR #15951
"Add cfgpp_ud10_ab sampler.", lines 419-484, unchanged through 387f98aa; GPL-3.0, like this extension).
The author tuned it on base Anima ("basic testing"). Each step is a CFG++ Euler step (Chung et al.,
arXiv:2406.08070: the guided x0 with the *unconditional* derivative), whose unconditional derivative is
extrapolated from the previous step's (``uncond_history_weight``), blended (``torch.lerp`` with
``history_weight``) with a second-order Adams–Bashforth step over the last two step derivatives
(``linear_multistep_coeff(2, …)``, scipy ``quad``); the last step to σ = 0 is pulled towards an
extrapolation of the x0 history to σ = 0 (first order: the line through the last two x0, ``zero_order``
2: the parabola through three). ``sample_cfgpp_ud10_ab`` fixes the three weights (module constants).

What is upstream's, expression for expression: the loop body, the history lists, the blend, the AB2
coefficients and both σ = 0 extrapolations, the post-CFG hook (installed without
``disable_cfg1_optimization``). Host changes (sam-extra, 2026-10-04):

1. The model sampling is Forge's ``model.inner_model.predictor`` (``common.model_sampling``) instead of
   ``model_patcher.get_model_object("model_sampling")``; ``sigma_to_half_log_snr`` (which tests
   ``prediction_type == "const"`` instead of ``isinstance(…, CONST)``), ``to_d``,
   ``linear_multistep_coeff``, ``trange`` and ``set_model_options_post_cfg_function`` are Forge's
   (``k_diffusion.sampling``, resolved at call time) — the same code as ComfyUI's.
2. **The unconditional prediction when CFG skipped it.** ComfyUI's post-CFG arguments carry
   ``uncond=None`` when its CFG-1 optimisation skipped the unconditional pass, and the sampler then uses
   ``cond_denoised``. Forge always passes the request's uncond conditioning there and leaves
   ``uncond_denoised`` at zeros when it skipped the pass (``backend/sampling/sampling_function.py``
   ``sampling_function_inner``). The hook therefore evaluates Forge's own skip predicate —
   ``math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False`` —
   and takes ``cond_denoised`` when it holds (or when ``uncond`` is None, as upstream). That covers CFG 1
   and Forge's *Skip Early CFG* / *NGMS* steps (both run the step at ``cond_scale = 1.0``) with ComfyUI's
   meaning.
3. ``@torch.no_grad()`` on the registered function (Forge's samplers carry it; upstream relies on the
   caller); results are unchanged.

``tests/test_extra_samplers_cfgpp_ud10_ab_origin.py`` runs upstream's verbatim code
(``tests/_origin_comfyui_cfgpp_history.py``) against this port, bit for bit, at CFG 1 / 1.5 / 2 / 4.5 on
ε and flow models. Forge's CFG++ warning ("CFG between 1.0 ~ 2.0 is recommended …") is logged above
CFG 2 (``registry``); ComfyUI recommends CFG 2 for this sampler.
"""

from __future__ import annotations

import math
from functools import partial

import torch

from .common import k_sampling, model_sampling

__all__ = [
    "UD10_HISTORY_WEIGHT",
    "UD10_UNCOND_HISTORY_WEIGHT",
    "UD10_ZERO_WEIGHT",
    "sample_cfgpp_history",
    "sample_cfgpp_ud10_ab",
]

# The three weights ``sample_cfgpp_ud10_ab`` passes (sampling.py:483-484 @ 3ac5d794).
UD10_HISTORY_WEIGHT = 0.25
UD10_ZERO_WEIGHT = 1.0
UD10_UNCOND_HISTORY_WEIGHT = 0.1


def _uncond_was_skipped(args: dict) -> bool:
    """Forge's CFG-1 skip predicate (``sampling_function_inner``): no unconditional pass ran for this call."""
    model_options = args.get("model_options") or {}
    return math.isclose(args["cond_scale"], 1.0) and model_options.get("disable_cfg1_optimization", False) == False  # noqa: E712 - Forge's comparison


def sample_cfgpp_history(model, x, sigmas, extra_args=None, callback=None, disable=None, history_weight=0.5,
                         zero_weight=None, zero_order=1, uncond_history_weight=0.0):
    """CFG++ Euler with variable-step AB2 history and optional sigma-zero extrapolation (upstream's
    ``_sample_cfgpp_history``; host changes in the module docstring)."""
    ks = k_sampling()
    to_d, linear_multistep_coeff = ks.to_d, ks.linear_multistep_coeff
    extra_args = {} if extra_args is None else extra_args
    sampling = model_sampling(model)
    lambda_fn = partial(ks.sigma_to_half_log_snr, model_sampling=sampling)
    s_in = x.new_ones([x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    derivatives = []
    denoised_history = []
    old_uncond_d = None
    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        skipped = _uncond_was_skipped(args) or args["uncond"] is None
        uncond_denoised = args["cond_denoised"] if skipped else args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ks.set_model_options_post_cfg_function(model_options, post_cfg_function)

    for i in ks.trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        alpha_s = sigmas[i] * lambda_fn(sigmas[i]).exp()
        alpha_t = sigmas[i + 1] * lambda_fn(sigmas[i + 1]).exp() if sigmas[i + 1] != 0 else sigmas[i + 1].new_ones([])
        current_uncond_d = to_d(x, sigmas[i], alpha_s * uncond_denoised)
        uncond_d = current_uncond_d
        dt = sigmas[i + 1] - sigmas[i]
        if i > 0 and uncond_history_weight:
            step_ratio = dt / (sigmas[i] - sigmas[i - 1])
            uncond_d = uncond_d + uncond_history_weight * step_ratio * (current_uncond_d - old_uncond_d)
        euler_step = alpha_t * denoised + sigmas[i + 1] * uncond_d - x
        d = euler_step / dt
        derivatives.append(d)
        if len(derivatives) > 2:
            derivatives.pop(0)

        if len(derivatives) == 1:
            step = euler_step
        else:
            coeffs = [linear_multistep_coeff(2, sigmas_cpu, i, j) for j in range(2)]
            history_step = sum(coeff * derivative for coeff, derivative in zip(coeffs, reversed(derivatives)))
            step = torch.lerp(euler_step, history_step, history_weight)
        x = x + step
        if sigmas[i + 1] == 0 and zero_weight is not None and denoised_history:
            if zero_order == 2 and len(denoised_history) > 1:
                sigma_0, sigma_1, sigma_2 = sigmas[i - 2], sigmas[i - 1], sigmas[i]
                weight_0 = sigma_1 * sigma_2 / ((sigma_0 - sigma_1) * (sigma_0 - sigma_2))
                weight_1 = sigma_0 * sigma_2 / ((sigma_1 - sigma_0) * (sigma_1 - sigma_2))
                weight_2 = sigma_0 * sigma_1 / ((sigma_2 - sigma_0) * (sigma_2 - sigma_1))
                zero_prediction = weight_0 * denoised_history[-2] + weight_1 * denoised_history[-1] + weight_2 * denoised
            else:
                denoised_slope = (denoised - denoised_history[-1]) / (sigmas[i] - sigmas[i - 1])
                zero_prediction = denoised - sigmas[i] * denoised_slope
            x = torch.lerp(x, zero_prediction, zero_weight)
        denoised_history.append(denoised)
        if len(denoised_history) > 2:
            denoised_history.pop(0)
        old_uncond_d = current_uncond_d
    return x


@torch.no_grad()
def sample_cfgpp_ud10_ab(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """``CFG++ UD10 AB``: history weight 0.25, σ = 0 extrapolation weight 1.0, unconditional history 0.1."""
    return sample_cfgpp_history(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                                history_weight=UD10_HISTORY_WEIGHT, zero_weight=UD10_ZERO_WEIGHT,
                                uncond_history_weight=UD10_UNCOND_HISTORY_WEIGHT)
