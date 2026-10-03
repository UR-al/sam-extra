# Verbatim excerpts of ComfyUI's ER-SDE code for the origin-parity tests
# (tests/test_extra_samplers_er_sde_origin.py).
#
# origin: comfyanonymous/ComfyUI@36c0b0a687e5e6d7b55e3e61ab24262ffc0f2508
#   comfy/k_diffusion/sampling.py         lines 78-88, 152-176, 1592-1656
#       (git blob 4a638008a3df3a2f167e9bf343a258a0f5655ede, file SHA-256 cfc4221ed0bc1f84a6f5689d3ec440cad3008c2541ab8f5e2599d1a73fac74ce)
#   comfy_extras/nodes_custom_sampler.py  lines 585-633
#       (git blob c73a8f6dcc0183e1b46a9525b1212590c45ecd77, file SHA-256 b0ba1521c72475e06fed15db004274ed7c8bb2bf73f1d58849faf6f9f9d264fe)
# https://github.com/comfyanonymous/ComfyUI — GPL-3.0 (upstream LICENSE), the same license as this
# extension. Every "---- upstream <file>:<lines> below (verbatim) ----" block is those upstream lines
# unchanged up to its "---- end ----" line (the test pins the SHA-256 of each block, so an accidental
# edit fails). The preamble before the first block is NOT upstream code: it is the test host and
# supplies the names the excerpts use from the rest of their files - torch,
# ``comfy.model_sampling.CONST`` (the flow-model class the half-log-SNR helpers test with isinstance),
# ``comfy.samplers.ksampler`` (returns the sampler name and options), ``comfy.utils.model_trange`` as
# ``trange`` and the ``comfy_api.latest.io`` schema classes the node is declared with. Test oracle
# only: the extension never imports it.

from types import SimpleNamespace

import torch
from tqdm.auto import trange as _tqdm_trange


class CONST:
    """comfy.model_sampling.CONST stand-in (rectified-flow model sampling)."""


def _ksampler(sampler_name, extra_options=None, inpaint_options=None):
    return SimpleNamespace(sampler_name=sampler_name, extra_options=dict(extra_options or {}))


comfy = SimpleNamespace(
    model_sampling=SimpleNamespace(CONST=CONST),
    samplers=SimpleNamespace(ksampler=_ksampler),
)


def trange(*args, **kwargs):
    """comfy.utils.model_trange stand-in."""
    return _tqdm_trange(*args, **kwargs)


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


# ---- upstream comfy/k_diffusion/sampling.py:78-88 below (verbatim) ----
def default_noise_sampler(x, seed=None):
    if seed is not None:
        if x.device == torch.device("cpu"):
            seed += 1

        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None

    return lambda sigma, sigma_next: torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)
# ---- end ----


# ---- upstream comfy/k_diffusion/sampling.py:152-176 below (verbatim) ----
def sigma_to_half_log_snr(sigma, model_sampling):
    """Convert sigma to half-logSNR log(alpha_t / sigma_t)."""
    if isinstance(model_sampling, comfy.model_sampling.CONST):
        # log((1 - t) / t) = log((1 - sigma) / sigma)
        return sigma.logit().neg()
    return sigma.log().neg()


def half_log_snr_to_sigma(half_log_snr, model_sampling):
    """Convert half-logSNR log(alpha_t / sigma_t) to sigma."""
    if isinstance(model_sampling, comfy.model_sampling.CONST):
        # 1 / (1 + exp(half_log_snr))
        return half_log_snr.neg().sigmoid()
    return half_log_snr.neg().exp()


def offset_first_sigma_for_snr(sigmas, model_sampling, percent_offset=1e-4):
    """Adjust the first sigma to avoid invalid logSNR."""
    if len(sigmas) <= 1:
        return sigmas
    if isinstance(model_sampling, comfy.model_sampling.CONST):
        if sigmas[0] >= 1:
            sigmas = sigmas.clone()
            sigmas[0] = model_sampling.percent_to_sigma(percent_offset)
    return sigmas
# ---- end ----


# ---- upstream comfy/k_diffusion/sampling.py:1592-1656 below (verbatim) ----
@torch.no_grad()
def sample_er_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0, noise_sampler=None, noise_scaler=None, max_stage=3):
    """Extended Reverse-Time SDE solver (VP ER-SDE-Solver-3). arXiv: https://arxiv.org/abs/2309.06169.
    Code reference: https://github.com/QinpengCui/ER-SDE-Solver/blob/main/er_sde_solver.py.
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_noise = s_noise * getattr(model.inner_model.model_patcher.get_model_object('model_sampling'), "noise_scale", 1.0)
    s_in = x.new_ones([x.shape[0]])

    def default_er_sde_noise_scaler(x):
        return x * ((x ** 0.3).exp() + 10.0)

    noise_scaler = default_er_sde_noise_scaler if noise_scaler is None else noise_scaler
    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
    half_log_snrs = sigma_to_half_log_snr(sigmas, model_sampling)
    er_lambdas = half_log_snrs.neg().exp()  # er_lambda_t = sigma_t / alpha_t

    old_denoised = None
    old_denoised_d = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

            # Stage 1 Euler
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised

            if stage_used >= 2:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)

                # Stage 2
                s = torch.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                if stage_used >= 3:
                    # Stage 3
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u
                old_denoised_d = denoised_d

            if s_noise > 0:
                x = x + alpha_t * noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
        old_denoised = denoised
    return x
# ---- end ----


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
