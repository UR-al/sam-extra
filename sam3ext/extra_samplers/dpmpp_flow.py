"""DPM++ 2M SDE Heun and the three "flow ODE" DPM++ entries — Forge's own DPM-Solver++ functions.

Forge Neo's ``modules_forge/packages/k_diffusion/sampling.py`` (a copy of ComfyUI v0.3.75's
``comfy/k_diffusion/sampling.py``) already has ``sample_dpmpp_2m_sde`` (with k-diffusion's
``solver_type="heun"`` correction, crowsonkb/k-diffusion@4601bf085320592473f681a62808ed873d17fad5, MIT) and
``sample_dpmpp_3m_sde``. Both step in the half-log-SNR ``λ`` of the model's parameterisation
(``sigma_to_half_log_snr``: ``λ = log((1 − σ)/σ)`` on a flow model such as Anima, ``−log σ`` otherwise) and
carry ``α_t = σ_t·e^{λ_t}`` explicitly, so at η = 0 they are DPM-Solver++ (Lu et al., arXiv:2211.01095) for the
model's own probability-flow ODE. Forge registers the 2M/3M SDE variants only (with ``exponential`` as the
Automatic scheduler) and no Heun SDE entry. These wrappers are sam-extra's own code; nothing is copied:

* ``DPM++ 2M SDE Heun`` — ``sample_dpmpp_2m_sde(..., solver_type="heun")`` with Forge's Eta, ``Sigma noise``
  and per-image Brownian tree (option ``brownian_noise``), the entry ComfyUI registers as
  ``dpmpp_2m_sde_heun`` (comfyanonymous/ComfyUI@3aad339b63f03e17dc6ebae035b90afc2fefb627, PR #9542).
  ``solver_type`` is fixed here, not taken from the sampler options.
* ``DPM++ 2M (flow ODE)`` / ``DPM++ 2M Heun (flow ODE)`` / ``DPM++ 3M (flow ODE)`` — the same functions at
  ``eta = 0`` (and ``s_noise = 0``): no ``eta`` parameter, so Forge's ``Sampler.initialize`` neither passes its
  global Eta nor writes ``Eta`` to the infotext, and a noise sampler that must never be called
  (``_no_noise``), so no Brownian tree is built and nothing is drawn. ``DPM++ 3M (flow ODE)`` keeps Forge's
  3M option ``discard_next_to_last_sigma`` and is therefore exactly Forge's ``DPM++ 3M SDE`` at Eta 0.
  On ε/v models λ = −log σ: ``DPM++ 2M (flow ODE)`` is Forge's ``DPM++ 2M`` and ``DPM++ 2M Heun (flow ODE)``
  RES 2M, both up to the order of floating-point operations.

None of the four carries a scheduler hint: Forge's Automatic then picks the model's own schedule
(``Normal`` on flow models), not ``exponential``/``karras``, which ignore a flow model's shift.
"""

from __future__ import annotations

from .common import k_sampling

__all__ = [
    "sample_dpmpp_2m_flow_ode",
    "sample_dpmpp_2m_heun_flow_ode",
    "sample_dpmpp_2m_sde_heun",
    "sample_dpmpp_3m_flow_ode",
]


def _no_noise(sigma, sigma_next):
    """The flow ODE entries' noise sampler: at η = 0 Forge's functions never call it."""
    raise RuntimeError("a flow ODE sampler drew noise")


def sample_dpmpp_2m_sde_heun(model, x, sigmas, extra_args=None, callback=None, disable=None,
                             eta=1.0, s_noise=1.0, noise_sampler=None):
    """DPM-Solver++(2M) SDE with the Heun (φ₂) correction."""
    return k_sampling().sample_dpmpp_2m_sde(
        model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
        eta=eta, s_noise=s_noise, noise_sampler=noise_sampler, solver_type="heun",
    )


def sample_dpmpp_2m_flow_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M) for the model's probability-flow ODE (midpoint correction)."""
    return k_sampling().sample_dpmpp_2m_sde(
        model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
        eta=0.0, s_noise=0.0, noise_sampler=_no_noise, solver_type="midpoint",
    )


def sample_dpmpp_2m_heun_flow_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M) for the model's probability-flow ODE (Heun / φ₂ correction)."""
    return k_sampling().sample_dpmpp_2m_sde(
        model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
        eta=0.0, s_noise=0.0, noise_sampler=_no_noise, solver_type="heun",
    )


def sample_dpmpp_3m_flow_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(3M) for the model's probability-flow ODE."""
    return k_sampling().sample_dpmpp_3m_sde(
        model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
        eta=0.0, s_noise=0.0, noise_sampler=_no_noise,
    )
