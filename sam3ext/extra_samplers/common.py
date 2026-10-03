"""Shared plumbing of the extra samplers: Forge's k-diffusion module, the parameterisation, CFG++.

Forge Neo keeps its k-diffusion code (a copy of ComfyUI v0.3.75 ``comfy/k_diffusion/sampling.py``)
in ``modules_forge/packages/k_diffusion/sampling.py`` and imports it as ``k_diffusion.sampling``.
The samplers resolve that module when they run, not when they are imported:

* **Noise.** ``default_noise_sampler(x)`` calls ``torch.randn_like`` through that module's ``torch``
  name, which Forge's ``Sampler.initialize`` replaces with ``TorchHijack(p)`` — one generator per
  image seed, so a picture does not change with the batch size (``modules/sd_samplers_common.py``).
  The CNS stage of this extension patches the same attribute, so it recolours this noise exactly
  like Forge's own samplers'. DPM++ 4M SDE gets Forge's per-image ``BrownianTreeNoiseSampler``
  (option ``brownian_noise``) and builds one from the same module when called without it.
* **Parameterisation.** ``sigma_to_half_log_snr`` / ``half_log_snr_to_sigma`` /
  ``offset_first_sigma_for_snr`` switch on ``prediction_type == "const"`` (rectified flow: Anima,
  Flux, SD3: ``x = (1 − σ)·x0 + σ·n``); everything else uses the k-diffusion form ``x = x0 + σ·n``.
  These are the helpers Forge's DPM++ 2M/3M SDE and CFG++ samplers use.
* **CFG++.** Forge's ``sample_euler_ancestral_cfg_pp`` appends a post-CFG function that keeps
  ``args["uncond_denoised"]`` (``set_model_options_post_cfg_function`` with
  ``disable_cfg1_optimization=True``, so the unconditional prediction exists at CFG 1 too).
  ``install_uncond_capture`` does the same.
"""

from __future__ import annotations

import importlib

__all__ = [
    "SUBSTEP_MARKER",
    "UncondCapture",
    "install_uncond_capture",
    "is_const",
    "k_sampling",
    "model_sampling",
    "with_substep_marker",
]

# ``transformer_options`` key set on the extra model evaluations of Euler (SMEA) Dy — the
# half-resolution Dy step and the ×1.25 SMEA step — value "dy" or "smea". Forge merges the
# sampler's model_options into the UNet's (``sampling_function`` → ``join_dicts``), so every hook
# (post-CFG ``args["model_options"]["transformer_options"]``, block patches) can tell these
# evaluations from the step's own one.
SUBSTEP_MARKER = "sam_extra_substep"


def k_sampling():
    """Forge's ``k_diffusion.sampling`` module (resolved at call time; tests install their own)."""
    return importlib.import_module("k_diffusion.sampling")


def model_sampling(model):
    """The predictor Forge's k-diffusion samplers read: ``model.inner_model.predictor``."""
    return model.inner_model.predictor


def is_const(sampling) -> bool:
    """Forge's ``_is_const``: rectified-flow parameterisation."""
    return getattr(sampling, "prediction_type", None) == "const"


class UncondCapture:
    """Forge's CFG++ post-CFG hook: remember the unconditional x0, return the guided one unchanged.

    origin: Haoming02/sd-webui-forge-classic (Forge Neo 2.29.2)
    modules_forge/packages/k_diffusion/sampling.py ``sample_euler_ancestral_cfg_pp`` ``post_cfg_function``.
    """

    __slots__ = ("value",)

    def __init__(self):
        self.value = None

    def __call__(self, args):
        self.value = args["uncond_denoised"]
        return args["denoised"]


def install_uncond_capture(extra_args: dict, ks=None) -> UncondCapture:
    """Append an ``UncondCapture`` to ``extra_args["model_options"]`` exactly as Forge's CFG++ does."""
    ks = k_sampling() if ks is None else ks
    capture = UncondCapture()
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = ks.set_model_options_post_cfg_function(
        model_options, capture, disable_cfg1_optimization=True
    )
    return capture


def with_substep_marker(extra_args: dict, kind: str) -> dict:
    """A copy of ``extra_args`` whose ``model_options["transformer_options"]`` carries ``SUBSTEP_MARKER``.

    Only the dictionaries on the path are copied: the post-CFG list (and the CFG++ capture in it)
    is shared with the step's own evaluation."""
    args = dict(extra_args)
    options = dict(args.get("model_options") or {})
    transformer_options = dict(options.get("transformer_options") or {})
    transformer_options[SUBSTEP_MARKER] = kind
    options["transformer_options"] = transformer_options
    args["model_options"] = options
    return args
