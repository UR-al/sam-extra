from __future__ import annotations

from collections import UserList
from functools import partial
from typing import Any, Literal, NamedTuple

try:
    from pydantic.v1 import BaseModel, Extra, NonNegativeFloat, NonNegativeInt, PositiveInt, confloat
except ImportError:
    from pydantic import BaseModel, Extra, NonNegativeFloat, NonNegativeInt, PositiveInt, confloat


class Arg(NamedTuple):
    attr: str
    name: str


class ArgsList(UserList):
    @property
    def attrs(self) -> tuple[str, ...]:
        return tuple(attr for attr, _ in self)


class Sam3Args(BaseModel, extra=Extra.forbid):
    sam3_mode: Literal["Mask only", "Inpaint"] = "Inpaint"
    sam3_mask_mode: Literal["Combined", "Individual"] = "Individual"
    sam3_prompt: str = "face"
    sam3_inpaint_prompt: str = ""
    sam3_negative_prompt: str = ""
    sam3_threshold: confloat(ge=0.0, le=1.0) = 0.4
    sam3_mask_dilation: NonNegativeInt = 0
    sam3_checkpoint: str = "sam3.pt"
    sam3_device: str = "auto"
    sam3_mask_blur: NonNegativeInt = 4
    sam3_denoising_strength: confloat(ge=0.0, le=1.0) = 0.4
    sam3_inpaint_only_masked: bool = True
    sam3_inpaint_only_masked_padding: NonNegativeInt = 32
    sam3_use_inpaint_width_height: bool = False
    sam3_inpaint_width: PositiveInt = 512
    sam3_inpaint_height: PositiveInt = 512
    sam3_use_steps: bool = False
    sam3_steps: PositiveInt = 28
    sam3_use_cfg_scale: bool = False
    sam3_cfg_scale: NonNegativeFloat = 7.0
    sam3_use_sampler: bool = False
    sam3_sampler: str = "Use same sampler"
    sam3_scheduler: str = "Use same scheduler"
    sam3_use_noise_multiplier: bool = False
    sam3_noise_multiplier: confloat(ge=0.0, le=2.0) = 1.0
    sam3_restore_face: bool = False
    sam3_preview_overlay: bool = False
    sam3_save_artifacts: bool = True

    @staticmethod
    def ppop(p: dict[str, Any], key: str, pops: list[str] | None = None, cond: Any = None) -> None:
        if pops is None:
            pops = [key]
        if key not in p:
            return
        value = p[key]
        cond = (not bool(value)) if cond is None else value == cond
        if cond:
            for pop_key in pops:
                p.pop(pop_key, None)

    def extra_params(self) -> dict[str, Any]:
        params = {name: getattr(self, attr) for attr, name in ALL_ARGS}
        ppop = partial(self.ppop, params)
        ppop("SAM3 Inpaint Prompt")
        ppop("SAM3 Negative Prompt")
        ppop("SAM3 Mask Dilation", cond=0)
        ppop("SAM3 Mask Blur", cond=4)
        ppop("SAM3 Denoising Strength", cond=0.4)
        ppop("SAM3 Inpaint Only Masked", ["SAM3 Inpaint Only Masked", "SAM3 Inpaint Padding"], cond=True)
        ppop(
            "SAM3 Use Inpaint Width Height",
            ["SAM3 Use Inpaint Width Height", "SAM3 Inpaint Width", "SAM3 Inpaint Height"],
        )
        ppop("SAM3 Use Separate Steps", ["SAM3 Use Separate Steps", "SAM3 Steps"])
        ppop("SAM3 Use Separate CFG Scale", ["SAM3 Use Separate CFG Scale", "SAM3 CFG Scale"])
        ppop("SAM3 Use Separate Sampler", ["SAM3 Use Separate Sampler", "SAM3 Sampler", "SAM3 Scheduler"])
        ppop("SAM3 Use Noise Multiplier", ["SAM3 Use Noise Multiplier", "SAM3 Noise Multiplier"])
        ppop("SAM3 Restore Face")
        ppop("SAM3 Preview Overlay")
        ppop("SAM3 Save Artifacts", cond=True)
        return params


ALL_ARGS = ArgsList(
    [
        Arg("sam3_mode", "SAM3 Mode"),
        Arg("sam3_mask_mode", "SAM3 Mask Mode"),
        Arg("sam3_prompt", "SAM3 Prompt"),
        Arg("sam3_inpaint_prompt", "SAM3 Inpaint Prompt"),
        Arg("sam3_negative_prompt", "SAM3 Negative Prompt"),
        Arg("sam3_threshold", "SAM3 Threshold"),
        Arg("sam3_mask_dilation", "SAM3 Mask Dilation"),
        Arg("sam3_checkpoint", "SAM3 Checkpoint"),
        Arg("sam3_device", "SAM3 Device"),
        Arg("sam3_mask_blur", "SAM3 Mask Blur"),
        Arg("sam3_denoising_strength", "SAM3 Denoising Strength"),
        Arg("sam3_inpaint_only_masked", "SAM3 Inpaint Only Masked"),
        Arg("sam3_inpaint_only_masked_padding", "SAM3 Inpaint Padding"),
        Arg("sam3_use_inpaint_width_height", "SAM3 Use Inpaint Width Height"),
        Arg("sam3_inpaint_width", "SAM3 Inpaint Width"),
        Arg("sam3_inpaint_height", "SAM3 Inpaint Height"),
        Arg("sam3_use_steps", "SAM3 Use Separate Steps"),
        Arg("sam3_steps", "SAM3 Steps"),
        Arg("sam3_use_cfg_scale", "SAM3 Use Separate CFG Scale"),
        Arg("sam3_cfg_scale", "SAM3 CFG Scale"),
        Arg("sam3_use_sampler", "SAM3 Use Separate Sampler"),
        Arg("sam3_sampler", "SAM3 Sampler"),
        Arg("sam3_scheduler", "SAM3 Scheduler"),
        Arg("sam3_use_noise_multiplier", "SAM3 Use Noise Multiplier"),
        Arg("sam3_noise_multiplier", "SAM3 Noise Multiplier"),
        Arg("sam3_restore_face", "SAM3 Restore Face"),
        Arg("sam3_preview_overlay", "SAM3 Preview Overlay"),
        Arg("sam3_save_artifacts", "SAM3 Save Artifacts"),
    ]
)
