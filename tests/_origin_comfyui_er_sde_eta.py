# Verbatim excerpt of ComfyUI's SamplerER_SDE node with the eta-extended noise scalers, for the
# origin-parity tests of ``ER SDE (Tunable)`` (tests/test_extra_samplers_er_sde_eta_origin.py).
#
# origin: comfyanonymous/ComfyUI@40e46c711025947f126cccaa1a692a14937a3096
#   ("Extend ER-SDE noise scaler by scaling h(t)", PR #15428, 2026-08-09 — the commit that introduced
#   ``er_sde_noise_scaler`` with ``** eta``)
#   comfy_extras/nodes_custom_sampler.py  lines 585-633
#       (git blob d5aa730d22cd0bed99f8a995529d2fff33db3a45, file SHA-256 b0efb86763d1df01ddb4248e851524e0acf2cb050cc4f303900f9ee070ea45c0)
# https://github.com/comfyanonymous/ComfyUI — GPL-3.0 (upstream LICENSE), the same license as this
# extension. The "---- upstream <file>:<lines> below (verbatim) ----" block is those upstream lines
# unchanged up to its "---- end ----" line (the test pins its SHA-256, so an accidental edit fails). The
# lines are byte-identical at 36c0b0a6, the commit of tests/_origin_comfyui_er_sde.py; this copy pins
# the commit that introduced them. The preamble before the block is NOT upstream code: it is the test
# host and supplies the names the excerpt uses from the rest of its file - torch,
# ``comfy.samplers.ksampler`` (returns the sampler name and options) and the ``comfy_api.latest.io``
# schema classes the node is declared with. Test oracle only: the extension never imports it.

from types import SimpleNamespace

import torch


def _ksampler(sampler_name, extra_options=None, inpaint_options=None):
    return SimpleNamespace(sampler_name=sampler_name, extra_options=dict(extra_options or {}))


comfy = SimpleNamespace(samplers=SimpleNamespace(ksampler=_ksampler))


class _Field:
    def __init__(self, kind):
        self.kind = kind

    def Input(self, name, **kwargs):
        return SimpleNamespace(kind=self.kind, name=name, **kwargs)

    def Output(self, *args, **kwargs):
        return SimpleNamespace(kind=self.kind, args=args, **kwargs)


class _ComfyNode:
    pass


io = SimpleNamespace(
    ComfyNode=_ComfyNode,
    Schema=lambda **kwargs: SimpleNamespace(**kwargs),
    Combo=_Field("COMBO"),
    Int=_Field("INT"),
    Float=_Field("FLOAT"),
    Sampler=_Field("SAMPLER"),
    NodeOutput=lambda *args: SimpleNamespace(args=args),
)


# ---- upstream comfy_extras/nodes_custom_sampler.py:585-633 below (verbatim) ----
class SamplerER_SDE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerER_SDE",
            category="model/sampling/samplers",
            inputs=[
                io.Combo.Input("solver_type", options=["ER-SDE", "Reverse-time SDE", "ODE"]),
                io.Int.Input("max_stage", default=3, min=1, max=3, advanced=True),
                io.Float.Input("eta", default=1.0, min=0.0, max=10.0, step=0.01, round=False, tooltip="Stochastic strength of SDEs.\nWhen eta=0, they reduce to deterministic ODE.\nLarge eta may cause invalid outputs. If this occurs, try decreasing this value.", advanced=True),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
            ],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, solver_type, max_stage, eta, s_noise) -> io.NodeOutput:
        # Extend existing noise scalers phi(x) with eta-controlled noise scalers:
        #   psi(x) = x**(1-eta) * phi(x)**eta
        # where eta is constant and directly scales the h^2(t) contribution.

        def er_sde_noise_scaler(x: torch.Tensor) -> torch.Tensor:
            return x * ((x ** 0.3).exp() + 10.0) ** eta

        def reverse_time_sde_noise_scaler(x: torch.Tensor) -> torch.Tensor:
            return x ** (eta + 1)

        def ode_noise_scaler(x: torch.Tensor) -> torch.Tensor:
            return x

        solver_scalers = {
            "ER-SDE": er_sde_noise_scaler,
            "Reverse-time SDE": reverse_time_sde_noise_scaler,
            "ODE": ode_noise_scaler,
        }

        if solver_type == "ODE" or eta == 0:
            s_noise = 0.0
            solver_type = "ODE"
        noise_scaler = solver_scalers[solver_type]

        sampler_name = "er_sde"
        sampler = comfy.samplers.ksampler(
            sampler_name,
            {"s_noise": s_noise, "noise_scaler": noise_scaler, "max_stage": max_stage},
        )
        return io.NodeOutput(sampler)

    get_sampler = execute
# ---- end ----
