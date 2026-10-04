# Verbatim excerpts of ComfyUI's CFG++ history sampler (``cfgpp_ud10_ab``) and the helpers it calls, for the
# origin-parity tests of ``CFG++ UD10 AB`` (tests/test_extra_samplers_cfgpp_ud10_ab_origin.py).
#
# origin: comfyanonymous/ComfyUI@3ac5d7941dfa2504555260512132d1cb5664648d
#   ("Add cfgpp_ud10_ab sampler.", PR #15951, 2026-08-28; the sampler is unchanged through 387f98aa)
#   comfy/k_diffusion/utils.py     lines 21-29
#       (git blob a644df2f3cf82b32ac6e9bf2cb7bfc70c95e05f9, file SHA-256 fc30630d8a0e4520b2e864c8f84deea3231c9b04d3f1404fec93fc425f629272)
#   comfy/model_patcher.py         lines 114-118
#       (git blob 72942aa044823c262dee8c9eb4029c0e95a3e2f2, file SHA-256 ac214200ea7991b9903c24f28457e2924357b025dbb9e353f1d3d2d397a9bd17)
#   comfy/k_diffusion/sampling.py  lines 63-65, 152-176, 406-416, 419-484
#       (git blob 4a638008a3df3a2f167e9bf343a258a0f5655ede, file SHA-256 cfc4221ed0bc1f84a6f5689d3ec440cad3008c2541ab8f5e2599d1a73fac74ce)
# https://github.com/comfyanonymous/ComfyUI — GPL-3.0 (upstream LICENSE), the same license as this
# extension. Every "---- upstream <file>:<lines> below (verbatim) ----" block is those upstream lines
# unchanged up to its "---- end ----" line (the test pins the SHA-256 of each block, so an accidental
# edit fails). The preamble before the first block is NOT upstream code: it is the test host and
# supplies the names the excerpts use from the rest of their files - torch, ``functools.partial``,
# scipy's ``integrate``, ``comfy.utils.model_trange`` as ``trange``, ``comfy.model_sampling.CONST`` (the
# flow-model class the half-log-SNR helpers test with isinstance), and ``utils`` / ``comfy.model_patcher``
# namespaces whose attributes resolve, when called, to the verbatim ``append_dims`` /
# ``set_model_options_post_cfg_function`` blocks below. Test oracle only: the extension never imports it.

from functools import partial

import torch
from scipy import integrate
from tqdm.auto import trange as _tqdm_trange


class CONST:
    """comfy.model_sampling.CONST stand-in (rectified-flow model sampling)."""


class _Late:
    """Attributes resolved from this module's globals at call time (the verbatim blocks below define them)."""

    def __getattr__(self, name):
        return globals()[name]


comfy = type("comfy", (), {})()
comfy.model_sampling = type("model_sampling", (), {"CONST": CONST})()
comfy.model_patcher = _Late()
utils = _Late()


def trange(*args, **kwargs):
    """comfy.utils.model_trange stand-in."""
    return _tqdm_trange(*args, **kwargs)


# ---- upstream comfy/k_diffusion/utils.py:21-29 below (verbatim) ----
def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    expanded = x[(...,) + (None,) * dims_to_append]
    # MPS will get inf values if it tries to index into the new axes, but detaching fixes this.
    # https://github.com/pytorch/pytorch/issues/84364
    return expanded.detach().clone() if expanded.device.type == 'mps' else expanded
# ---- end ----


# ---- upstream comfy/model_patcher.py:114-118 below (verbatim) ----
def set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=False):
    model_options["sampler_post_cfg_function"] = model_options.get("sampler_post_cfg_function", []) + [post_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options
# ---- end ----


# ---- upstream comfy/k_diffusion/sampling.py:63-65 below (verbatim) ----
def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / utils.append_dims(sigma, x.ndim)
# ---- end ----


# ---- upstream comfy/k_diffusion/sampling.py:152-176 below (verbatim) ----
def sigma_to_half_log_snr(sigma, model_sampling):
    """Convert sigma to half-logSNR log(alpha_t / sigma_t)."""
    if isinstance(model_sampling, comfy.model_sampling.CONST):
        # log((1 - t) / t) = log((1 - sigma) / sigma)
        return sigma.logit().neg()
    return sigma.log().neg()


def half_log_snr_to_sigma(half_log_snr, model_sampling):
    """Convert half-logSNR log(alpha_t / sigma_t) to sigma."""
    if isinstance(model_sampling, comfy.model_sampling.CONST):
        # 1 / (1 + exp(half_log_snr))
        return half_log_snr.neg().sigmoid()
    return half_log_snr.neg().exp()


def offset_first_sigma_for_snr(sigmas, model_sampling, percent_offset=1e-4):
    """Adjust the first sigma to avoid invalid logSNR."""
    if len(sigmas) <= 1:
        return sigmas
    if isinstance(model_sampling, comfy.model_sampling.CONST):
        if sigmas[0] >= 1:
            sigmas = sigmas.clone()
            sigmas[0] = model_sampling.percent_to_sigma(percent_offset)
    return sigmas
# ---- end ----


# ---- upstream comfy/k_diffusion/sampling.py:406-416 below (verbatim) ----
def linear_multistep_coeff(order, t, i, j):
    if order - 1 > i:
        raise ValueError(f'Order {order} too high for step {i}')
    def fn(tau):
        prod = 1.
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod
    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]
# ---- end ----


# ---- upstream comfy/k_diffusion/sampling.py:419-484 below (verbatim) ----
def _sample_cfgpp_history(model, x, sigmas, extra_args=None, callback=None, disable=None, history_weight=0.5, zero_weight=None, zero_order=1, uncond_history_weight=0.0):
    """CFG++ Euler with variable-step AB2 history and optional sigma-zero extrapolation."""
    extra_args = {} if extra_args is None else extra_args
    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
    s_in = x.new_ones([x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    derivatives = []
    denoised_history = []
    old_uncond_d = None
    uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"] if args["uncond"] is not None else args["cond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function)

    for i in trange(len(sigmas) - 1, disable=disable):
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


def sample_cfgpp_ud10_ab(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return _sample_cfgpp_history(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, history_weight=0.25, zero_weight=1.0, uncond_history_weight=0.1)
# ---- end ----
