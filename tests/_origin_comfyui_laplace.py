# Verbatim excerpts of upstream ComfyUI for the Laplace origin-parity tests (tests/test_extra_schedulers_origin.py).
#
# origin: comfyanonymous/ComfyUI@36c0b0a687e5e6d7b55e3e61ab24262ffc0f2508
#   comfy/k_diffusion/sampling.py:52-59            get_sigmas_laplace
#     (git blob 4a638008a3df3a2f167e9bf343a258a0f5655ede, file SHA-256
#      cfc4221ed0bc1f84a6f5689d3ec440cad3008c2541ab8f5e2599d1a73fac74ce)
#   comfy_extras/nodes_custom_sampler.py:112-133   LaplaceScheduler (the node: inputs, defaults, ranges)
#     (git blob c73a8f6dcc0183e1b46a9525b1212590c45ecd77, file SHA-256
#      b0ba1521c72475e06fed15db004274ed7c8bb2bf73f1d58849faf6f9f9d264fe)
# https://github.com/comfyanonymous/ComfyUI — GPL-3.0 (upstream LICENSE), the same license as this extension.
#
# Every block below a "# ---- upstream <path>:<first>-<last> (verbatim) ----" line is those upstream lines
# unchanged (both files use LF line endings); tests/test_extra_schedulers_origin.py pins the SHA-256 of each
# block, so an accidental edit fails. Only this header and the marker lines were added. The node block refers
# to ``io`` (comfy_api.latest) when the class is created and to ``k_diffusion_sampling`` when it runs; the
# test puts stand-ins for both into the module before executing it. Test oracle only: the extension never
# imports this file.
import torch

# ---- upstream comfy/k_diffusion/sampling.py:52-59 (verbatim) ----
def get_sigmas_laplace(n, sigma_min, sigma_max, mu=0., beta=0.5, device='cpu'):
    """Constructs the noise schedule proposed by Tiankai et al. (2024). """
    epsilon = 1e-5 # avoid log(0)
    x = torch.linspace(0, 1, n, device=device)
    clamp = lambda x: torch.clamp(x, min=sigma_min, max=sigma_max)
    lmb = mu - beta * torch.sign(0.5-x) * torch.log(1 - 2 * torch.abs(0.5-x) + epsilon)
    sigmas = clamp(torch.exp(lmb))
    return sigmas


# ---- upstream comfy_extras/nodes_custom_sampler.py:112-133 (verbatim) ----
class LaplaceScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LaplaceScheduler",
            category="model/sampling/schedulers",
            inputs=[
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("sigma_max", default=14.614642, min=0.0, max=5000.0, step=0.01, round=False, advanced=True),
                io.Float.Input("sigma_min", default=0.0291675, min=0.0, max=5000.0, step=0.01, round=False, advanced=True),
                io.Float.Input("mu", default=0.0, min=-10.0, max=10.0, step=0.1, round=False, advanced=True),
                io.Float.Input("beta", default=0.5, min=0.0, max=10.0, step=0.1, round=False, advanced=True),
            ],
            outputs=[io.Sigmas.Output()]
        )

    @classmethod
    def execute(cls, steps, sigma_max, sigma_min, mu, beta) -> io.NodeOutput:
        sigmas = k_diffusion_sampling.get_sigmas_laplace(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, mu=mu, beta=beta)
        return io.NodeOutput(sigmas)

    get_sigmas = execute
