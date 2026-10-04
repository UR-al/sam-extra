# Verbatim excerpts of ComfyUI's iPNDM / iPNDM_v / DEIS samplers (zju-pi/diff-sampler under Apache 2, adapted to
# k-diffusion by ComfyUI) and of ComfyUI's ``deis.py`` (zju-pi's DEIS coefficients), for the origin-parity
# tests of ``IPNDM`` / ``IPNDM_V`` / ``DEIS`` (tests/test_extra_samplers_ipndm_deis_origin.py).
#
# origin: comfyanonymous/ComfyUI@387f98aa2822f684b8597959a52a467d88cc4806
#   comfy/k_diffusion/sampling.py  lines 1173-1330
#       (git blob 4a638008a3df3a2f167e9bf343a258a0f5655ede, file SHA-256 cfc4221ed0bc1f84a6f5689d3ec440cad3008c2541ab8f5e2599d1a73fac74ce)
#   comfy/k_diffusion/deis.py      lines 1-120 (the whole file)
#       (git blob a1167a4a36c42b22168ea8b3d80b1d83b2505db5, file SHA-256 8d4e5057a062b77ef2e33af8208a2d6a3c3fbb431315a4c37689a307d66bc807)
# https://github.com/comfyanonymous/ComfyUI — GPL-3.0 (upstream LICENSE). The samplers carry their own
# "#From https://github.com/zju-pi/diff-sampler/... under Apache 2 license" lines and deis.py its
# "#Taken from ... gits-main/solver_utils.py under Apache 2 license" lines (zju-pi/diff-sampler, Apache-2.0).
# Every "---- upstream <file>:<lines> below (verbatim) ----" block is those upstream lines unchanged up to
# its "---- end ----" line (the test pins the SHA-256 of each block, so an accidental edit fails). The
# preamble before the first block is NOT upstream code: it is the test host and supplies the names the
# excerpts use from the rest of their file - torch, ``comfy.utils.model_trange`` as ``trange`` and the
# ``deis`` module the samplers call (an object whose attributes resolve, when called, to the verbatim
# deis.py block below). Test oracle only: the extension never imports it.

import torch
from tqdm.auto import trange as _tqdm_trange


class _Late:
    """Attributes resolved from this module's globals at call time (the deis.py block below defines them)."""

    def __getattr__(self, name):
        return globals()[name]


deis = _Late()


def trange(*args, **kwargs):
    """comfy.utils.model_trange stand-in."""
    return _tqdm_trange(*args, **kwargs)


# ---- upstream comfy/k_diffusion/sampling.py:1173-1330 below (verbatim) ----
#From https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
#under Apache 2 license
def sample_ipndm(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=4):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    x_next = x

    buffer_model = []
    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]

        x_cur = x_next

        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        d_cur = (x_cur - denoised) / t_cur

        order = min(max_order, i+1)
        if t_next == 0:     # Denoising step
            x_next = denoised
        elif order == 1:    # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:    # Use one history point.
            x_next = x_cur + (t_next - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
        elif order == 3:    # Use two history points.
            x_next = x_cur + (t_next - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
        elif order == 4:    # Use three history points.
            x_next = x_cur + (t_next - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24

        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur
        else:
            buffer_model.append(d_cur)

    return x_next


#From https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
#under Apache 2 license
def sample_ipndm_v(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=4):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    x_next = x
    t_steps = sigmas

    buffer_model = []
    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]

        x_cur = x_next

        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        d_cur = (x_cur - denoised) / t_cur

        order = min(max_order, i+1)
        if t_next == 0:     # Denoising step
            x_next = denoised
        elif order == 1:    # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:    # Use one history point.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            coeff1 = (2 + (h_n / h_n_1)) / 2
            coeff2 = -(h_n / h_n_1) / 2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1])
        elif order == 3:    # Use two history points.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            h_n_2 = (t_steps[i-1] - t_steps[i-2])
            temp = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp
            coeff3 = temp * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2])
        elif order == 4:    # Use three history points.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            h_n_2 = (t_steps[i-1] - t_steps[i-2])
            h_n_3 = (t_steps[i-2] - t_steps[i-3])
            temp1 = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            temp2 = ((1 - h_n / (3 * (h_n + h_n_1))) / 2 + (1 - h_n / (2 * (h_n + h_n_1))) * h_n / (6 * (h_n + h_n_1 + h_n_2))) \
                   * (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3))
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp1 + temp2
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp1 - (1 + (h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3)))) * temp2
            coeff3 = temp1 * h_n_1 / h_n_2 + ((h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * (1 + h_n_2 / h_n_3)) * temp2
            coeff4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2] + coeff4 * buffer_model[-3])

        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())

    return x_next


#From https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/solvers.py
#under Apache 2 license
@torch.no_grad()
def sample_deis(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=3, deis_mode='tab'):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    x_next = x
    t_steps = sigmas

    coeff_list = deis.get_deis_coeff_list(t_steps, max_order, deis_mode=deis_mode)

    buffer_model = []
    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]

        x_cur = x_next

        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        d_cur = (x_cur - denoised) / t_cur

        order = min(max_order, i+1)
        if t_next <= 0:
            order = 1

        if order == 1:          # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:        # Use one history point.
            coeff_cur, coeff_prev1 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1]
        elif order == 3:        # Use two history points.
            coeff_cur, coeff_prev1, coeff_prev2 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2]
        elif order == 4:        # Use three history points.
            coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2] + coeff_prev3 * buffer_model[-3]

        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())

    return x_next
# ---- end ----


# ---- upstream comfy/k_diffusion/deis.py:1-120 below (verbatim) ----
#Taken from: https://github.com/zju-pi/diff-sampler/blob/main/gits-main/solver_utils.py
#under Apache 2 license
import torch
import numpy as np

# A pytorch reimplementation of DEIS (https://github.com/qsh-zh/deis).
#############################
### Utils for DEIS solver ###
#############################
#----------------------------------------------------------------------------
# Transfer from the input time (sigma) used in EDM to that (t) used in DEIS.

def edm2t(edm_steps, epsilon_s=1e-3, sigma_min=0.002, sigma_max=80):
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    vp_beta_d = 2 * (np.log(torch.tensor(sigma_min).cpu() ** 2 + 1) / epsilon_s - np.log(torch.tensor(sigma_max).cpu() ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(torch.tensor(sigma_max).cpu() ** 2 + 1) - 0.5 * vp_beta_d
    t_steps = vp_sigma_inv(vp_beta_d.clone().detach().cpu(), vp_beta_min.clone().detach().cpu())(edm_steps.clone().detach().cpu())
    return t_steps, vp_beta_min, vp_beta_d + vp_beta_min

#----------------------------------------------------------------------------

def cal_poly(prev_t, j, taus):
    poly = 1
    for k in range(prev_t.shape[0]):
        if k == j:
            continue
        poly *= (taus - prev_t[k]) / (prev_t[j] - prev_t[k])
    return poly

#----------------------------------------------------------------------------
# Transfer from t to alpha_t.

def t2alpha_fn(beta_0, beta_1, t):
    return torch.exp(-0.5 * t ** 2 * (beta_1 - beta_0) - t * beta_0)

#----------------------------------------------------------------------------

def cal_intergrand(beta_0, beta_1, taus):
    with torch.inference_mode(mode=False):
        taus = taus.clone()
        beta_0 = beta_0.clone()
        beta_1 = beta_1.clone()
        with torch.enable_grad():
            taus.requires_grad_(True)
            alpha = t2alpha_fn(beta_0, beta_1, taus)
            log_alpha = alpha.log()
            log_alpha.sum().backward()
            d_log_alpha_dtau = taus.grad
    integrand = -0.5 * d_log_alpha_dtau / torch.sqrt(alpha * (1 - alpha))
    return integrand

#----------------------------------------------------------------------------

def get_deis_coeff_list(t_steps, max_order, N=10000, deis_mode='tab'):
    """
    Get the coefficient list for DEIS sampling.

    Args:
        t_steps: A pytorch tensor. The time steps for sampling.
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 4
        N: A `int`. Use how many points to perform the numerical integration when deis_mode=='tab'.
        deis_mode: A `str`. Select between 'tab' and 'rhoab'. Type of DEIS.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """
    if deis_mode == 'tab':
        t_steps, beta_0, beta_1 = edm2t(t_steps)
        C = []
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            order = min(i+1, max_order)
            if order == 1:
                C.append([])
            else:
                taus = torch.linspace(t_cur, t_next, N)   # split the interval for integral appximation
                dtau = (t_next - t_cur) / N
                prev_t = t_steps[[i - k for k in range(order)]]
                coeff_temp = []
                integrand = cal_intergrand(beta_0, beta_1, taus)
                for j in range(order):
                    poly = cal_poly(prev_t, j, taus)
                    coeff_temp.append(torch.sum(integrand * poly) * dtau)
                C.append(coeff_temp)

    elif deis_mode == 'rhoab':
        # Analytical solution, second order
        def get_def_intergral_2(a, b, start, end, c):
            coeff = (end**3 - start**3) / 3 - (end**2 - start**2) * (a + b) / 2 + (end - start) * a * b
            return coeff / ((c - a) * (c - b))

        # Analytical solution, third order
        def get_def_intergral_3(a, b, c, start, end, d):
            coeff = (end**4 - start**4) / 4 - (end**3 - start**3) * (a + b + c) / 3 \
                    + (end**2 - start**2) * (a*b + a*c + b*c) / 2 - (end - start) * a * b * c
            return coeff / ((d - a) * (d - b) * (d - c))

        C = []
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            order = min(i, max_order)
            if order == 0:
                C.append([])
            else:
                prev_t = t_steps[[i - k for k in range(order+1)]]
                if order == 1:
                    coeff_cur = ((t_next - prev_t[1])**2 - (t_cur - prev_t[1])**2) / (2 * (t_cur - prev_t[1]))
                    coeff_prev1 = (t_next - t_cur)**2 / (2 * (prev_t[1] - t_cur))
                    coeff_temp = [coeff_cur, coeff_prev1]
                elif order == 2:
                    coeff_cur = get_def_intergral_2(prev_t[1], prev_t[2], t_cur, t_next, t_cur)
                    coeff_prev1 = get_def_intergral_2(t_cur, prev_t[2], t_cur, t_next, prev_t[1])
                    coeff_prev2 = get_def_intergral_2(t_cur, prev_t[1], t_cur, t_next, prev_t[2])
                    coeff_temp = [coeff_cur, coeff_prev1, coeff_prev2]
                elif order == 3:
                    coeff_cur = get_def_intergral_3(prev_t[1], prev_t[2], prev_t[3], t_cur, t_next, t_cur)
                    coeff_prev1 = get_def_intergral_3(t_cur, prev_t[2], prev_t[3], t_cur, t_next, prev_t[1])
                    coeff_prev2 = get_def_intergral_3(t_cur, prev_t[1], prev_t[3], t_cur, t_next, prev_t[2])
                    coeff_prev3 = get_def_intergral_3(t_cur, prev_t[1], prev_t[2], t_cur, t_next, prev_t[3])
                    coeff_temp = [coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3]
                C.append(coeff_temp)
    return C

# ---- end ----
