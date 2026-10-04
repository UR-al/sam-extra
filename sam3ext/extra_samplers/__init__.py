"""Extra samplers for Forge Neo — ER SDE, Euler (SMEA) Dy (± CFG++), DPM++ in λ, multistep ODE, CFG++, Restart.

Eighteen entries for Forge's sampler dropdown, registered by ``scripts/anima_extra_samplers.py``
(``registry.register``). Where another WebUI already has the sampler, its label is used, so infotexts
paste across (aoleg/Neo_ExtraSchedulers for the ER SDE pair, DPM++ 4M SDE and the Dy CFG++ pair;
Koishi-Star/reForge for Euler Dy / Euler SMEA Dy; ComfyUI/A1111 for DPM++ 2M SDE Heun; lllyasviel's Forge /
reForge for IPNDM, IPNDM_V and DEIS). The script is
not named ``extra_samplers.py``: that file name is Panchovix/sd_forge_neo_extra_samplers', and Forge keys
ui-config, logs and the module name by it.

* **ER SDE** — ``ER SDE (Reverse-time)`` — ``er_sde.sample_er_sde_reverse_time``; ``ER SDE (ODE)`` —
  ``er_sde.sample_er_sde_ode``; ``ER SDE (Tunable)`` — ``er_sde.sample_er_sde_tunable`` (= Forge's built-in
  ``ER SDE`` at η 1): Forge's own ``sample_er_sde`` with the noise scalers of ComfyUI's ``SamplerER_SDE``
  node (GPL-3.0). Reverse-time and Tunable can restrict their noise to a σ window (``er_sde_window``,
  sam-extra's own code; the idea credited to pamparamm/ComfyUI-ppm, AGPL-3.0, not used).
* **Dy** — ``Euler Dy CFG++`` / ``Euler SMEA Dy CFG++`` (``euler_dy.sample_euler_dy_cfg_pp`` /
  ``sample_euler_smea_dy_cfg_pp``, Forge's CFG++ update) and ``Euler Dy`` / ``Euler SMEA Dy``
  (``euler_dy.sample_euler_dy`` / ``sample_euler_smea_dy``, plain Euler steps):
  Koishi-Star/Euler-Smea-Dyn-Sampler (Apache-2.0) with k-diffusion's ``min`` churn rule and a flow-correct
  churn. ``substep_guard`` turns their extra sub-steps off on requests that cannot follow a resolution
  change (Forge's Spectrum Integrated, Wan I2V, PiD, ``extra_concat_condition``).
* **DPM++ in λ** — ``DPM++ 4M SDE`` — ``dpmpp_4m_sde.sample_dpmpp_4m_sde``: Clybius/ComfyUI-Extra-Samplers
  (BSD-3-Clause) with Forge's flow-model handling. ``DPM++ 2M SDE Heun``, ``DPM++ 2M (flow ODE)``,
  ``DPM++ 2M Heun (flow ODE)``, ``DPM++ 3M (flow ODE)`` — ``dpmpp_flow``: Forge's own
  ``sample_dpmpp_2m_sde`` / ``sample_dpmpp_3m_sde`` (Heun correction; η = 0 for the ODE entries).
* **Multistep ODE** — ``UniPC bh2`` — ``unipc.sample_unipc_bh2``: Forge's own UniPC with ``variant="bh2"``.
  ``IPNDM`` / ``IPNDM_V`` / ``DEIS`` — ``ipndm_deis``: zju-pi/diff-sampler (Apache-2.0) as ComfyUI runs them,
  DEIS on Forge's vendored ``k_diffusion.deis`` coefficients.
* **CFG++** — ``CFG++ UD10 AB`` — ``cfgpp_ud10_ab.sample_cfgpp_ud10_ab``: ComfyUI's ``cfgpp_ud10_ab`` (GPL-3.0).
* **Restart** — ``Restart (flow)`` — ``restart_flow.sample_restart_flow``: Restart sampling (Xu et al.,
  arXiv:2306.14878) written from the paper with the flow forward kernel (α = 1 − σ).

None of the new entries has a scheduler hint (Forge's Automatic then uses the model's schedule).

aoleg/Neo_ExtraSchedulers and DenOfEquity/webUI_ExtraSchedulers publish no licence; only their
READMEs were read (for the labels and the user-facing behaviour). No code of theirs is used.

``params`` holds the per-request ``ER SDE max stage`` / ``ER SDE eta`` / noise-window values of the accordion.
Nothing here imports Forge at import time; Forge's modules are resolved when a sampler runs or
when ``register`` is called.
"""

from .params import (
    KEY_ETA,
    KEY_MAX_STAGE,
    KEY_NOISE_WINDOW,
    LABEL_ER_SDE_ODE,
    LABEL_ER_SDE_REVERSE_TIME,
    LABEL_ER_SDE_TUNABLE,
    ErSdeSettings,
)
from .registry import (
    LABEL_CFGPP_UD10_AB,
    LABEL_DPMPP_2M_FLOW_ODE,
    LABEL_DPMPP_2M_HEUN_FLOW_ODE,
    LABEL_DPMPP_2M_SDE_HEUN,
    LABEL_DPMPP_3M_FLOW_ODE,
    LABEL_DPMPP_4M_SDE,
    LABEL_DEIS,
    LABEL_EULER_DY,
    LABEL_EULER_DY_CFG_PP,
    LABEL_EULER_SMEA_DY,
    LABEL_EULER_SMEA_DY_CFG_PP,
    LABEL_IPNDM,
    LABEL_IPNDM_V,
    LABEL_RESTART_FLOW,
    LABEL_UNIPC_BH2,
    SPECS,
    register,
)

LABELS = tuple(spec.label for spec in SPECS)

__all__ = [
    "KEY_ETA",
    "KEY_MAX_STAGE",
    "KEY_NOISE_WINDOW",
    "LABELS",
    "LABEL_CFGPP_UD10_AB",
    "LABEL_DPMPP_2M_FLOW_ODE",
    "LABEL_DPMPP_2M_HEUN_FLOW_ODE",
    "LABEL_DPMPP_2M_SDE_HEUN",
    "LABEL_DPMPP_3M_FLOW_ODE",
    "LABEL_DPMPP_4M_SDE",
    "LABEL_DEIS",
    "LABEL_ER_SDE_ODE",
    "LABEL_ER_SDE_REVERSE_TIME",
    "LABEL_ER_SDE_TUNABLE",
    "LABEL_EULER_DY",
    "LABEL_EULER_DY_CFG_PP",
    "LABEL_EULER_SMEA_DY",
    "LABEL_EULER_SMEA_DY_CFG_PP",
    "LABEL_IPNDM",
    "LABEL_IPNDM_V",
    "LABEL_RESTART_FLOW",
    "LABEL_UNIPC_BH2",
    "SPECS",
    "ErSdeSettings",
    "register",
]
