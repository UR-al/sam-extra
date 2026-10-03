"""Shared fixtures of the extra-samplers tests (``test_extra_samplers_*.py``); not a test module.

* Forge's real k-diffusion module. ``forge_k_sampling()`` executes Forge Neo's
  ``modules_forge/packages/k_diffusion/sampling.py`` unchanged; its two imports from the rest of
  Forge (``utils.append_dims`` and ``backend.patcher.base.set_model_options_post_cfg_function``)
  are Forge's own function bodies, taken from those files with ``ast`` — no other Forge code runs.
  Forge is looked for where the extension normally lives (``<forge>/extensions/<this>``), or at
  ``$SAM3_FORGE_ROOT``; without it these tests skip.
* The three verbatim origin copies (``tests/_origin_*.py``) with the stand-ins their imports need.
* Toy models: an elementwise x0 predictor, a CFG denoiser stand-in that runs the post-CFG list and
  the inpainting blend in Forge's order, deterministic noise sources, predictor stand-ins.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import os
import sys
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORGE_ENV = "SAM3_FORGE_ROOT"
_K_DIR = Path("modules_forge") / "packages" / "k_diffusion"


# ---------------------------------------------------------------------------
# Forge
# ---------------------------------------------------------------------------


def forge_root() -> Path | None:
    candidates = []
    if os.environ.get(FORGE_ENV):
        candidates.append(Path(os.environ[FORGE_ENV]))
    candidates.append(ROOT.parents[1])  # <forge>/extensions/<this extension>
    for candidate in candidates:
        if (candidate / _K_DIR / "sampling.py").is_file():
            return candidate
    return None


def require_forge() -> Path:
    root = forge_root()
    if root is None:
        raise unittest.SkipTest(f"Forge Neo not found next to the extension (or at ${FORGE_ENV})")
    return root


@contextlib.contextmanager
def stub_modules(stubs: dict):
    """Swap only the named ``sys.modules`` entries and put them back (see test_anima_safe_pag)."""
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def forge_source(rel_path: str) -> str:
    return (require_forge() / rel_path).read_text(encoding="utf-8")


def forge_definition(rel_path: str, name: str, *, cls: str | None = None) -> str:
    """Source of a top-level function (or of a method of class ``cls``) in a Forge file."""
    source = forge_source(rel_path)
    tree = ast.parse(source)
    scope = tree.body
    if cls is not None:
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
        scope = owner.body
    for node in scope:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(source, node)
            if cls is not None:
                # dedent a method so it can be exec'd at module level
                lines = segment.splitlines()
                indent = len(lines[0]) - len(lines[0].lstrip())
                segment = "\n".join(line[indent:] if line[:indent].isspace() else line for line in lines)
            decorators = "".join(f"@{ast.get_source_segment(source, d)}\n" for d in node.decorator_list)
            return decorators + segment
    raise LookupError(f"{name} not found in {rel_path}")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_CACHE: dict = {}


def forge_k_sampling():
    """Forge Neo's ``k_diffusion.sampling`` module, executed from Forge's file (cached)."""
    if "forge_k" in _CACHE:
        return _CACHE["forge_k"]
    root = require_forge()
    namespace: dict = {}
    exec(compile(forge_definition("backend/patcher/base.py", "set_model_options_post_cfg_function"),
                 "backend/patcher/base.py", "exec"), namespace)
    utils_ns: dict = {}
    exec(compile(forge_definition(str(_K_DIR / "utils.py"), "append_dims"), "k_diffusion/utils.py", "exec"), utils_ns)

    backend = types.ModuleType("backend")
    backend.__path__ = []
    patcher = types.ModuleType("backend.patcher")
    patcher.__path__ = []
    base = types.ModuleType("backend.patcher.base")
    base.set_model_options_post_cfg_function = namespace["set_model_options_post_cfg_function"]
    package_name = "_sam_extra_forge_k_diffusion"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root / _K_DIR)]
    utils = types.ModuleType(package_name + ".utils")
    utils.append_dims = utils_ns["append_dims"]
    package.utils = utils
    with stub_modules({
        "backend": backend, "backend.patcher": patcher, "backend.patcher.base": base,
        package_name: package, package_name + ".utils": utils,
    }):
        sampling = _load(package_name + ".sampling", root / _K_DIR / "sampling.py")
    _CACHE["forge_k"] = sampling
    return sampling


_SPECTRUM = Path("extensions-builtin") / "sd_forge_spectrum" / "lib_spectrum" / "forecaster.py"
# scripts/spectrum.py slider defaults: w, m, lam, window_size, flex_window, warmup_steps, stop_caching_step
SPECTRUM_DEFAULTS = (0.25, 6, 0.5, 2, 0.0, 6, 0.9)


def forge_module_from_source(rel_path: str, name: str):
    """A Forge module executed from its file's text under ``name`` (read-only: no bytecode is written)."""
    path = require_forge() / rel_path
    if not path.is_file():
        raise unittest.SkipTest(f"{rel_path} is not in this Forge")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def forge_spectrum_forecaster():
    """Forge's built-in Spectrum Integrated forecaster, under its import name ``lib_spectrum.forecaster``."""
    if "spectrum" not in _CACHE:
        _CACHE["spectrum"] = forge_module_from_source(str(_SPECTRUM), "lib_spectrum.forecaster")
    return _CACHE["spectrum"]


class UnetPatcherStandIn:
    """What ``SpectrumNode.patch`` calls on Forge's ``UnetPatcher``: ``clone`` and
    ``set_model_unet_function_wrapper`` (backend/patcher/unet.py), plus the fields the guard reads."""

    def __init__(self, model_options=None, model=None, extra_concat_condition=None):
        self.model_options = dict(model_options) if model_options else {"transformer_options": {}}
        self.model = model
        self.extra_concat_condition = extra_concat_condition

    def clone(self):
        return UnetPatcherStandIn(self.model_options, self.model, self.extra_concat_condition)

    def set_model_unet_function_wrapper(self, function):
        self.model_options["model_function_wrapper"] = function


@contextlib.contextmanager
def installed_k_sampling(module):
    """Make ``module`` the ``k_diffusion.sampling`` the extra samplers import at call time."""
    package = types.ModuleType("k_diffusion")
    package.__path__ = []
    package.sampling = module
    with stub_modules({"k_diffusion": package, "k_diffusion.sampling": module}):
        yield module


# ---------------------------------------------------------------------------
# Origin copies
# ---------------------------------------------------------------------------

ORIGIN_DIR = ROOT / "tests"


def origin_body(filename: str, marker: str) -> str:
    """Text after ``marker`` in an origin copy, read with universal newlines (CRLF checkouts)."""
    text = (ORIGIN_DIR / filename).read_text(encoding="utf-8")
    head, sep, body = text.partition(marker)
    assert sep, f"marker missing in {filename}"
    return body


def origin_blocks(filename: str) -> dict:
    """``{"<file>:<first>-<last>": text}`` of the verbatim blocks of an excerpt copy."""
    text = (ORIGIN_DIR / filename).read_text(encoding="utf-8")
    blocks = {}
    prefix, suffix = "# ---- upstream ", " below (verbatim) ----\n"
    for chunk in text.split(prefix)[1:]:
        key, sep, rest = chunk.partition(suffix)
        assert sep, chunk[:80]
        body, sep, _tail = rest.partition("# ---- end ----\n")
        assert sep, key
        blocks[key] = body
    return blocks


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_comfy_er_sde_origin():
    if "comfy_er_sde" not in _CACHE:
        _CACHE["comfy_er_sde"] = _load("_origin_comfyui_er_sde", ORIGIN_DIR / "_origin_comfyui_er_sde.py")
    return _CACHE["comfy_er_sde"]


def _unused(*args, **kwargs):  # names the origin imports but the tests never call
    raise AssertionError("stand-in called")


def load_clybius_origin(k_sampling_module):
    """The verbatim Clybius ``extra_samplers.py`` with its imports satisfied (Forge's k-diffusion
    supplies the shared helpers it names; the rest are stand-ins that are never called)."""
    key = ("clybius", id(k_sampling_module))
    if key in _CACHE:
        return _CACHE[key]
    pkg = "_origin_clybius_pkg"
    package = types.ModuleType(pkg)
    package.__path__ = []
    other = types.ModuleType(pkg + ".other_samplers")
    other.__path__ = []
    refined = types.ModuleType(pkg + ".other_samplers.refined_exp_solver")
    refined.sample_refined_exp_s = _unused
    refined._de_second_order = _unused
    ttm = types.ModuleType(pkg + ".other_samplers.sample_ttm")
    ttm.sample_ttm_jvp = _unused
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy_sample = types.ModuleType("comfy.sample")
    comfy_kd = types.ModuleType("comfy.k_diffusion")
    comfy_kd.__path__ = []
    comfy_kds = types.ModuleType("comfy.k_diffusion.sampling")
    for name in ("BrownianTreeNoiseSampler", "get_ancestral_step", "to_d", "default_noise_sampler", "sample_lcm"):
        setattr(comfy_kds, name, getattr(k_sampling_module, name))
    comfy_kds.PIDStepSizeController = _unused
    comfy_kds.DPMSolver = _unused
    kornia = types.ModuleType("kornia")
    with stub_modules({
        pkg: package, pkg + ".other_samplers": other,
        pkg + ".other_samplers.refined_exp_solver": refined, pkg + ".other_samplers.sample_ttm": ttm,
        "comfy": comfy, "comfy.sample": comfy_sample, "comfy.k_diffusion": comfy_kd,
        "comfy.k_diffusion.sampling": comfy_kds, "kornia": kornia,
    }):
        module = _load(pkg + ".extra_samplers", ORIGIN_DIR / "_origin_clybius_extra_samplers.py")
    _CACHE[key] = module
    return module


def load_koishi_origin(k_sampling_module):
    """The verbatim Koishi-Star ``smea_sampling.py`` on its WebUI backend (``k_diffusion.sampling``
    is ``k_sampling_module``; ``modules.sd_samplers_kdiffusion`` only has to exist)."""
    key = ("koishi", id(k_sampling_module))
    if key in _CACHE:
        return _CACHE[key]
    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []
    kdiffusion = types.ModuleType("modules.sd_samplers_kdiffusion")
    kd_pkg = types.ModuleType("k_diffusion")
    kd_pkg.__path__ = []
    with stub_modules({
        "modules": modules_pkg, "modules.sd_samplers_kdiffusion": kdiffusion,
        "k_diffusion": kd_pkg, "k_diffusion.sampling": k_sampling_module,
    }):
        module = _load("_origin_koishi_smea_sampling", ORIGIN_DIR / "_origin_koishi_smea_sampling.py")
    assert module.BACKEND == "WebUI" and module.sampling is k_sampling_module
    _CACHE[key] = module
    return module


class TorchWithNoise:
    """``torch`` for a module whose ``torch.randn_like`` must draw from ``noise`` (Forge's
    ``TorchHijack`` does the same for ``k_diffusion.sampling``)."""

    def __init__(self, noise):
        self._noise = noise

    def __getattr__(self, item):
        if item == "randn_like":
            return lambda x: self._noise(None, None)
        return getattr(torch, item)


@contextlib.contextmanager
def origin_torch(module, noise):
    saved = module.torch
    module.torch = TorchWithNoise(noise)
    try:
        yield
    finally:
        module.torch = saved


# ---------------------------------------------------------------------------
# Models, noise
# ---------------------------------------------------------------------------


def time_snr_shift(alpha, t):
    """Forge/ComfyUI ``time_snr_shift``."""
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


class EpsSampling:
    """Forge ``Prediction`` stand-in for an epsilon model (k-diffusion ``x = x0 + σ·n``)."""

    prediction_type = "epsilon"


class FlowSampling:
    """Forge ``PredictionDiscreteFlow`` stand-in (``prediction_type == "const"``, shifted percent)."""

    prediction_type = "const"

    def __init__(self, shift: float = 3.0):
        self.shift = shift

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return time_snr_shift(self.shift, 1.0 - percent)


def toy_x0(x: torch.Tensor, sigma: torch.Tensor, salt: float) -> torch.Tensor:
    """An elementwise, σ-dependent x0 prediction (exact under any strides; per frame on 5-D)."""
    s = sigma.reshape(-1, *([1] * (x.ndim - 1))).to(x.dtype)
    return (
        torch.tanh(x * (0.7 + 0.1 * salt) / (1.0 + s)) * (0.9 - 0.05 * salt)
        + 0.1 * torch.sin(1.3 * x + salt) * torch.exp(-s)
        + 0.03 * salt
    )


@dataclass
class Call:
    shape: tuple
    sigma: float
    marker: object
    init_latent: tuple | None
    mask: tuple | None
    image_cond: tuple | None
    model_options: dict = field(default_factory=dict)


class ToyModel:
    """``model(x, sigma, **extra_args)`` as Forge's k-diffusion samplers call the CFG denoiser.

    ``inner_model.predictor`` is what Forge's samplers read, ``inner_model.model_patcher
    .get_model_object("model_sampling")`` what ComfyUI's read. With ``cond_scale`` the call runs
    a CFG on two toy predictions, then the post-CFG functions in ``model_options`` with Forge's
    ``sampling_function_inner`` arguments, then the inpainting blend of ``CFGDenoiser.forward``."""

    def __init__(self, sampling, cond_scale: float | None = None, salt: float = 0.5):
        self.inner_model = types.SimpleNamespace(
            predictor=sampling,
            model_patcher=types.SimpleNamespace(get_model_object=lambda name: sampling),
        )
        self.cond_scale = cond_scale
        self.salt = salt
        self.init_latent = None
        self.mask = None
        self.nmask = None
        self.calls: list[Call] = []

    def __call__(self, x, sigma, **extra_args):
        options = extra_args.get("model_options") or {}
        image_cond = extra_args.get("image_cond")
        if self.cond_scale is None:
            denoised = toy_x0(x, sigma, self.salt)
        else:
            cond = toy_x0(x, sigma, 1.0)
            uncond = toy_x0(x, sigma, -1.0)
            if torch.is_tensor(image_cond) and tuple(image_cond.shape[-2:]) == tuple(x.shape[-2:]):
                cond = cond + 0.05 * image_cond[:, :1]
            denoised = uncond + (cond - uncond) * self.cond_scale
            for fn in options.get("sampler_post_cfg_function", []):
                denoised = fn({
                    "denoised": denoised, "cond": [], "uncond": [], "cond_scale": self.cond_scale,
                    "model": None, "uncond_denoised": uncond, "cond_denoised": cond, "sigma": sigma,
                    "model_options": options, "input": x,
                })
        if self.mask is not None:
            denoised = denoised * self.nmask + self.init_latent * self.mask
        transformer_options = options.get("transformer_options") or {}
        self.calls.append(Call(
            shape=tuple(x.shape),
            sigma=float(sigma.reshape(-1)[0]),
            marker=transformer_options.get("sam_extra_substep"),
            init_latent=None if self.init_latent is None else tuple(self.init_latent.shape),
            mask=None if self.mask is None else tuple(self.mask.shape),
            image_cond=None if not torch.is_tensor(image_cond) else tuple(image_cond.shape),
            model_options=options,
        ))
        return denoised


class NoiseSequence:
    """``noise_sampler(sigma, sigma_next)``: the next tensor of a seeded sequence shaped like ``like``."""

    def __init__(self, like: torch.Tensor, seed: int = 0):
        self.shape = tuple(like.shape)
        self.dtype = like.dtype
        self.generator = torch.Generator().manual_seed(seed)
        self.drawn: list[torch.Tensor] = []

    def __call__(self, sigma, sigma_next):
        noise = torch.randn(self.shape, generator=self.generator, dtype=self.dtype)
        self.drawn.append(noise)
        return noise

    @property
    def calls(self) -> int:
        return len(self.drawn)


class ReplayNoise:
    """Hands out given tensors in order (e.g. the frames of another run's noise)."""

    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.calls = 0

    def __call__(self, sigma, sigma_next):
        noise = self.tensors[self.calls]
        self.calls += 1
        return noise


def flow_sigmas(steps: int, shift: float = 3.0, dtype=torch.float64) -> torch.Tensor:
    """A shifted flow schedule from 1.0 to 0 (Forge's ``simple`` shape, without the model)."""
    t = torch.linspace(1.0, 0.0, steps + 1, dtype=dtype)
    return time_snr_shift(shift, t)


def eps_sigmas(steps: int, sigma_max: float = 14.6, sigma_min: float = 0.03, dtype=torch.float64) -> torch.Tensor:
    """A Karras schedule (rho 7) ending in 0."""
    ramp = torch.linspace(0, 1, steps, dtype=dtype)
    min_inv, max_inv = sigma_min ** (1 / 7), sigma_max ** (1 / 7)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** 7
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def seeded(shape, seed: int, dtype=torch.float64) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)
