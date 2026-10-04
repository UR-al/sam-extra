"""UniPC bh2 — Forge's own UniPC with the ``bh2`` B(h) variant.

Forge Neo's ``UniPC`` entry is ``modules/sd_samplers_extra.py`` ``sample_unipc`` on
``modules/uni_pc/uni_pc.py`` (a copy of ComfyUI v0.3.64 ``comfy/extra_samplers/uni_pc.py``, GPL-3.0), called
with its default ``variant="bh1"``. ``sample_unipc`` already takes ``variant``; this entry passes ``"bh2"`` —
UniPC's other B(h) choice (Zhao et al., "UniPC: A Unified Predictor-Corrector Framework for Fast Sampling
of Diffusion Models", arXiv:2302.04867; ComfyUI's ``uni_pc_bh2``). sam-extra's own two lines; no code is
copied. Forge's function is looked up when the sampler runs (``modules.sd_samplers_extra``); the registry
registers the entry only when that function exists and has the ``variant`` parameter.

Same options as Forge's ``UniPC`` (``discard_next_to_last_sigma``) and no scheduler hint. On flow models it is
Forge's UniPC as it is (it works on the VE noise level ``σ`` through ``predict_eps_sigma`` and ends at
σ = 0.001); a flow-λ UniPC is not part of this entry.
"""

from __future__ import annotations

import importlib

__all__ = ["UNIPC_MODULE", "UNIPC_VARIANT", "sample_unipc_bh2"]

UNIPC_MODULE = "modules.sd_samplers_extra"
UNIPC_VARIANT = "bh2"


def sample_unipc_bh2(model, x, sigmas, extra_args=None, callback=None, disable=False):
    """Forge's ``sample_unipc`` with ``variant="bh2"``.

    ``extra_args=None`` becomes ``{}``: Forge's function unpacks it into every model call (``**model_kwargs``)
    and fails on None; Forge's own sampler always passes a dict, so nothing else changes."""
    sample_unipc = importlib.import_module(UNIPC_MODULE).sample_unipc
    return sample_unipc(model, x, sigmas, extra_args={} if extra_args is None else extra_args, callback=callback,
                        disable=disable, variant=UNIPC_VARIANT)
