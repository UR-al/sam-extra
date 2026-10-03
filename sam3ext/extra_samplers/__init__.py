"""Extra samplers for Forge Neo — ER SDE (Reverse-time / ODE), DPM++ 4M SDE, Euler (SMEA) Dy CFG++.

Five entries for Forge's sampler dropdown, registered by ``scripts/anima_extra_samplers.py``
(``registry.register``) with the labels aoleg/Neo_ExtraSchedulers uses, so infotexts from either
extension paste into the other (the script is not named ``extra_samplers.py``: that file name is
Panchovix/sd_forge_neo_extra_samplers', and Forge keys ui-config, logs and the module name by it):

* ``ER SDE (Reverse-time)`` — ``er_sde.sample_er_sde_reverse_time``; ``ER SDE (ODE)`` —
  ``er_sde.sample_er_sde_ode``: Forge's own ``sample_er_sde`` with the noise scalers of ComfyUI's
  ``SamplerER_SDE`` node (GPL-3.0).
* ``DPM++ 4M SDE`` — ``dpmpp_4m_sde.sample_dpmpp_4m_sde``: Clybius/ComfyUI-Extra-Samplers
  (BSD-3-Clause) with Forge's flow-model handling.
* ``Euler Dy CFG++`` — ``euler_dy.sample_euler_dy_cfg_pp``; ``Euler SMEA Dy CFG++`` —
  ``euler_dy.sample_euler_smea_dy_cfg_pp``: Koishi-Star/Euler-Smea-Dyn-Sampler (Apache-2.0) with
  Forge's CFG++ update. ``substep_guard`` turns their extra sub-steps off on requests that cannot
  follow a resolution change (Forge's Spectrum Integrated, Wan I2V, PiD, ``extra_concat_condition``).

aoleg/Neo_ExtraSchedulers and DenOfEquity/webUI_ExtraSchedulers publish no licence; only their
READMEs were read (for the labels and the user-facing behaviour). No code of theirs is used.

``params`` holds the per-request ``ER SDE max stage`` / ``ER SDE eta`` values of the accordion.
Nothing here imports Forge at import time; Forge's modules are resolved when a sampler runs or
when ``register`` is called.
"""

from .params import (
    KEY_ETA,
    KEY_MAX_STAGE,
    LABEL_ER_SDE_ODE,
    LABEL_ER_SDE_REVERSE_TIME,
    ErSdeSettings,
)
from .registry import (
    LABEL_DPMPP_4M_SDE,
    LABEL_EULER_DY_CFG_PP,
    LABEL_EULER_SMEA_DY_CFG_PP,
    SPECS,
    register,
)

LABELS = tuple(spec.label for spec in SPECS)

__all__ = [
    "KEY_ETA",
    "KEY_MAX_STAGE",
    "LABELS",
    "LABEL_DPMPP_4M_SDE",
    "LABEL_ER_SDE_ODE",
    "LABEL_ER_SDE_REVERSE_TIME",
    "LABEL_EULER_DY_CFG_PP",
    "LABEL_EULER_SMEA_DY_CFG_PP",
    "SPECS",
    "ErSdeSettings",
    "register",
]
