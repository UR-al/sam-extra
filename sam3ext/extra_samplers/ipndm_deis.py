# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The three samplers in this file are derived from zju-pi/diff-sampler
# (https://github.com/zju-pi/diff-sampler, commit 68d5ce427f261962b89ce3b0ee8f6b29f0577328,
# ``diff-solvers-main/solvers.py``: ``ipndm_sampler``, ``ipndm_v_sampler``, ``deis_sampler``; Apache License
# 2.0), through ComfyUI's adaptation to the k-diffusion interface (comfyanonymous/ComfyUI@387f98aa2822f684b8597959a52a467d88cc4806,
# ``comfy/k_diffusion/sampling.py`` lines 1173-1330, marked there "#From https://github.com/zju-pi/diff-sampler/
# blob/main/diff-solvers-main/solvers.py under Apache 2 license"; ComfyUI is GPL-3.0, like this extension).
# The attribution names the repository and the commit, as ComfyUI does; the DEIS coefficients are not
# copied — Forge's own vendored ``k_diffusion/deis.py`` (zju-pi's ``gits-main/solver_utils.py``, Apache-2.0)
# is called at run time.
# MODIFIED by sam-extra, 2026-10-04 and 2026-10-05 (Apache-2.0 section 4(b)) — every change is listed in the docstring
# below.
"""IPNDM, IPNDM_V and DEIS — zju-pi's multistep ODE samplers (one model call per step) on Forge.

All three integrate the probability-flow ODE ``dx/dσ = d`` with ``d = (x − D)/σ`` (the k-diffusion
derivative; on a flow model ``x = (1 − σ)·x0 + σ·n`` it is exactly the velocity ``n − x0``, so they solve
the flow ODE as they solve the ε one) by an Adams–Bashforth method in σ over the last ``max_order`` step
derivatives:

* ``IPNDM`` (``sample_ipndm``) — the fixed AB coefficients of orders 1-4 (iPNDM, Zhang & Chen,
  arXiv:2204.13902: the pseudo linear multistep method of Liu et al., arXiv:2202.09778, started with lower
  orders instead of Runge–Kutta steps). The fixed coefficients assume equal steps.
* ``IPNDM_V`` (``sample_ipndm_v``) — the variable-step coefficients from the step sizes ``h_n … h_{n−3}``
  (with change 4 below; on Anima's shift-3 grid about as accurate as IPNDM — exact Gaussian denoiser, within 5%).
* ``DEIS`` (``sample_deis``) — DEIS-AB ("tab", Zhang & Chen, arXiv:2204.13902): coefficients integrated
  over the Lagrange basis in DEIS' VP time (σ mapped with EDM's constants, ``edm2t``), from Forge's vendored
  ``k_diffusion.deis.get_deis_coeff_list`` (a monotone time variable, so consistent on flow schedules too).

Upstream (zju-pi ``solvers.py``, as adapted by ComfyUI 387f98aa — the result here equals ComfyUI's
``ipndm`` / ``deis`` bit for bit; IPNDM_V equals ComfyUI's ``ipndm_v`` when consecutive step ratios are equal
(``h_{n−1}/h_{n−2} = h_{n−2}/h_{n−3}``: equal steps, geometric lists), otherwise it carries the coeff4 fix
(change 4) — ``tests/test_extra_samplers_ipndm_deis_origin.py``): ``x_next = x + (σ_next − σ)·Σ c_k d_k`` per
step, a step to σ = 0 returns the x0 prediction (IPNDM, IPNDM_V) or is forced to order 1 (DEIS,
``t_next <= 0``), a history buffer of ``max_order − 1`` derivatives. Changes (sam-extra, 2026-10-04; change 4
on 2026-10-05, v0.33.1):

1. Host: Forge's ``trange`` (``k_diffusion.sampling``) and Forge's ``k_diffusion.deis``; every function
   carries ``@torch.no_grad()`` (upstream only on DEIS).
2. The callback gets the current latent ``x_cur`` (ComfyUI passes the initial ``x`` every step — a preview
   bug; results are unaffected).
3. ``max_order`` is clamped to 1..4 (the orders the coefficients exist for; upstream silently skips the
   update for a larger order) and ``max_order = 1`` keeps no history (upstream's buffer update indexes an
   empty list and raises ``IndexError``), so order 1 is plain Euler with the final denoising step.
4. IPNDM_V's order-4 weight ``coeff4`` ends in ``h_n_2 / h_n_3`` where upstream has ``h_n_1 / h_n_2`` — a typo
   (that token pair is the only difference from upstream's step code). ``temp2`` multiplies the third divided
   difference, whose weights are ``1, −(1 + h_{n−1}/h_{n−2} + q), h_{n−1}/h_{n−2} + q·(1 + h_{n−2}/h_{n−3}),
   −q·h_{n−2}/h_{n−3}`` with ``q = h_{n−1}(h_{n−1}+h_{n−2}) / (h_{n−2}(h_{n−2}+h_{n−3}))``; upstream's last entry
   ``−q·h_{n−1}/h_{n−2}`` makes the four step weights sum to ``1 + q·temp2·(h_{n−2}/h_{n−3} − h_{n−1}/h_{n−2})``
   instead of 1. On Anima's Linear Quadratic 28 list the sum is about −174 at step 15 (σ 0.968 → 0.952) and the
   image turns into green noise, on Forge and on ComfyUI alike; fixed, the weights sum to 1 on every list. Known
   upstream inaccuracies kept on purpose, so the result stays ComfyUI's whenever the step ratios are equal:
   ``temp1`` (order 3's ``temp`` too) closes its parenthesis and ``/ 2`` in the wrong place and ``temp2`` adds the
   ``(1 − h_n/(2(h_n+h_{n−1})))·h_n/(6(h_n+h_{n−1}+h_{n−2}))`` term the integral subtracts. Neither changes the
   weight sum (still 1); they make the order-3/4 weights differ from the exact variable-step Adams–Bashforth ones —
   at equal steps order 4 is (57, −65, 43, −11)/24 instead of AB4's (55, −59, 37, −9)/24 (IPNDM's), order 3 is
   AB3's.

No scheduler hint, no extra parameters (ComfyUI's KSampler defaults: order 4, 4 and 3, DEIS mode 'tab').
"""

from __future__ import annotations

import importlib

import torch

from .common import k_sampling

__all__ = [
    "DEIS_MAX_ORDER",
    "DEIS_MODE",
    "IPNDM_MAX_ORDER",
    "sample_deis",
    "sample_ipndm",
    "sample_ipndm_v",
]

IPNDM_MAX_ORDER = 4
DEIS_MAX_ORDER = 3
DEIS_MODE = "tab"
_HIGHEST_ORDER = 4   # the coefficient tables stop at order 4


def _order(max_order) -> int:
    return max(1, min(_HIGHEST_ORDER, int(max_order)))


def _keep(buffer_model: list, d_cur: torch.Tensor, max_order: int) -> None:
    """Upstream's history update: the last ``max_order − 1`` derivatives (none for order 1)."""
    if max_order < 2:
        return
    if len(buffer_model) == max_order - 1:
        for k in range(max_order - 2):
            buffer_model[k] = buffer_model[k + 1]
        buffer_model[-1] = d_cur
    else:
        buffer_model.append(d_cur)


@torch.no_grad()
def sample_ipndm(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=IPNDM_MAX_ORDER):
    """iPNDM: Adams–Bashforth in σ with fixed coefficients (orders 1-4)."""
    max_order = _order(max_order)
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    x_next = x

    buffer_model = []
    for i in k_sampling().trange(len(sigmas) - 1, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]

        x_cur = x_next

        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({'x': x_cur, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        d_cur = (x_cur - denoised) / t_cur

        order = min(max_order, i + 1)
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

        _keep(buffer_model, d_cur, max_order)

    return x_next


@torch.no_grad()
def sample_ipndm_v(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=IPNDM_MAX_ORDER):
    """iPNDM_v: Adams–Bashforth in σ with variable-step coefficients (orders 1-4)."""
    max_order = _order(max_order)
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    x_next = x
    t_steps = sigmas

    buffer_model = []
    for i in k_sampling().trange(len(sigmas) - 1, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]

        x_cur = x_next

        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({'x': x_cur, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        d_cur = (x_cur - denoised) / t_cur

        order = min(max_order, i + 1)
        if t_next == 0:     # Denoising step
            x_next = denoised
        elif order == 1:    # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:    # Use one history point.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i - 1])
            coeff1 = (2 + (h_n / h_n_1)) / 2
            coeff2 = -(h_n / h_n_1) / 2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1])
        elif order == 3:    # Use two history points.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i - 1])
            h_n_2 = (t_steps[i - 1] - t_steps[i - 2])
            temp = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp
            coeff3 = temp * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2])
        elif order == 4:    # Use three history points.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i - 1])
            h_n_2 = (t_steps[i - 1] - t_steps[i - 2])
            h_n_3 = (t_steps[i - 2] - t_steps[i - 3])
            temp1 = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            temp2 = ((1 - h_n / (3 * (h_n + h_n_1))) / 2 + (1 - h_n / (2 * (h_n + h_n_1))) * h_n / (6 * (h_n + h_n_1 + h_n_2))) \
                * (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3))
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp1 + temp2
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp1 - (1 + (h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3)))) * temp2
            coeff3 = temp1 * h_n_1 / h_n_2 + ((h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * (1 + h_n_2 / h_n_3)) * temp2
            # Upstream (zju-pi, copied by ComfyUI) ends this weight in ``* h_n_1 / h_n_2``, a typo: temp2 multiplies the
            # third divided difference, whose last entry is ``-q * h_n_2 / h_n_3`` (q = the bracket; coeff3 has the
            # matching ``q * (1 + h_n_2 / h_n_3)``). With the typo the weights sum to
            # ``1 + q * temp2 * (h_n_2 / h_n_3 - h_n_1 / h_n_2)``, not 1 — about -174 at step 15 of Anima's Linear Quadratic
            # 28 list (green noise, on ComfyUI too). Fixed, they sum to 1 on every list, and the result is still upstream's
            # whenever h_n_1 / h_n_2 == h_n_2 / h_n_3 (equal steps, geometric lists). Change 4 in the docstring.
            coeff4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_2 / h_n_3
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2] + coeff4 * buffer_model[-3])

        _keep(buffer_model, d_cur.detach(), max_order)

    return x_next


@torch.no_grad()
def sample_deis(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=DEIS_MAX_ORDER,
                deis_mode=DEIS_MODE):
    """DEIS-AB with the coefficients of Forge's vendored ``k_diffusion.deis.get_deis_coeff_list``."""
    max_order = _order(max_order)
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    x_next = x
    t_steps = sigmas

    coeff_list = importlib.import_module("k_diffusion.deis").get_deis_coeff_list(t_steps, max_order, deis_mode=deis_mode)

    buffer_model = []
    for i in k_sampling().trange(len(sigmas) - 1, disable=disable):
        t_cur = sigmas[i]
        t_next = sigmas[i + 1]

        x_cur = x_next

        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None:
            callback({'x': x_cur, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        d_cur = (x_cur - denoised) / t_cur

        order = min(max_order, i + 1)
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

        _keep(buffer_model, d_cur.detach(), max_order)

    return x_next
