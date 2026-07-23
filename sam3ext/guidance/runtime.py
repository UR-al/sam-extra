"""Generation-scoped state shared by the guidance orchestrator.

The sampler is single-threaded for one WebUI generation, but one denoise step
may invoke the model wrapper multiple times. Keeping every mutable buffer under
one object makes pass-boundary cleanup explicit and prevents stale APG/SMC/CNS
tensors from surviving a hires pass or later generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, MutableMapping


@dataclass
class GuidanceRuntime:
    state: MutableMapping[str, Any]
    apg: MutableMapping[str, Any]
    adg: MutableMapping[str, Any]
    cfg: MutableMapping[str, Any] = field(default_factory=dict)
    dcw: MutableMapping[str, Any] = field(default_factory=dict)
    dave: MutableMapping[str, Any] = field(default_factory=dict)
    cns: MutableMapping[str, Any] = field(default_factory=dict)
    smc_prev: Any = None
    cns_x_t: Any = None
    cns_noise_calls: int = 0

    def reset_cfg_state(self) -> None:
        self.apg["avg"] = None
        self.apg["last_sigma"] = None
        self.smc_prev = None

    def reset_pass(self) -> None:
        self.reset_cfg_state()
        self.cns_x_t = None
        self.cns_noise_calls = 0
        self.state["active"] = 0
        self.state["step_open"] = False
        self.state["attn_raw"] = None
        self.state["slg_raw"] = None
        self.state["adg_skipped"] = False
        self.state["attn_spatial_shape"] = None

    def close_step(self) -> None:
        self.state["step_open"] = False
        self.state["attn_raw"] = None
        self.state["slg_raw"] = None
        self.state["adg_skipped"] = False
        self.state["attn_spatial_shape"] = None
