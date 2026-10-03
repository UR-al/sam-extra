"""Shared fixtures for the Colorcraft tests (not a test module).

* ``load_origin()`` — the pinned upstream ComfyUI node package (tests/_origin_colorcraft, MIT) as the
  oracle, behind a minimal fake ``comfy``; ``load_fork()`` — the fork's spec/params/core.
* Forge's own code where the port depends on it, read from the Forge checkout this extension lives in
  (``ROOT.parents[1]``, like the other origin tests): the real latent format classes
  (modules_forge/packages/huggingface_guess/latent.py) and ``setup_img2img_steps``.
* Host stand-ins: a deterministic colour-dependent VAE, Forge's ModelPatcher option handling, ``p``.
* ``Scenario`` — one Colorcraft set-up expressed once and turned both into the node graph (oracle) and
  into this extension's script arguments, so the two runs can be compared bit for bit.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORIGIN_DIR = ROOT / "tests" / "_origin_colorcraft"
FORK_DIR = ROOT / "tests" / "_origin_colorcraft_fork"
UPSTREAM_COMMIT = "d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf"
FORK_COMMIT = "f00066c63c9d8f96abc119cada3b51b36688fcac"


def _forge_root():
    for candidate in (ROOT.parents[1],):
        if (candidate / "modules_forge" / "packages" / "huggingface_guess" / "latent.py").is_file():
            return candidate
    return None


FORGE = _forge_root()


def require_forge(test):
    if FORGE is None:
        test.skipTest("the Forge checkout is not next to this extension (extensions/<ext>)")


# ---------------------------------------------------------------------------
# Origin (oracle) loaders
# ---------------------------------------------------------------------------


def origin_body(path: Path) -> str:
    """The verbatim upstream text after the marker line (universal newlines, like the other pins)."""
    text = path.read_text(encoding="utf-8")
    marker = next(line for line in text.splitlines(keepends=True) if line.startswith("# ---- upstream "))
    return text.split(marker, 1)[1]


@contextlib.contextmanager
def fake_comfy():
    """``comfy.samplers`` / ``model_patcher`` / ``model_management`` just big enough for nodes.py."""
    comfy = types.ModuleType("comfy")
    samplers = types.ModuleType("comfy.samplers")
    model_patcher = types.ModuleType("comfy.model_patcher")
    model_management = types.ModuleType("comfy.model_management")

    class KSAMPLER:
        def __init__(self, sampler_function, extra_options=None, inpaint_options=None):
            self.sampler_function = sampler_function
            self.extra_options = extra_options or {}
            self.inpaint_options = inpaint_options or {}

    def set_model_options_post_cfg_function(model_options, fn, disable_cfg1_optimization=False):
        # origin: comfyanonymous/ComfyUI comfy/model_patcher.py (copy the dict, append the function)
        model_options = dict(model_options)
        model_options["sampler_post_cfg_function"] = list(model_options.get("sampler_post_cfg_function", [])) + [fn]
        if disable_cfg1_optimization:
            model_options["disable_cfg1_optimization"] = True
        return model_options

    samplers.KSAMPLER = KSAMPLER
    model_patcher.set_model_options_post_cfg_function = set_model_options_post_cfg_function
    model_management.get_torch_device = lambda: torch.device("cpu")
    comfy.samplers, comfy.model_patcher, comfy.model_management = samplers, model_patcher, model_management
    names = {"comfy": comfy, "comfy.samplers": samplers, "comfy.model_patcher": model_patcher,
             "comfy.model_management": model_management}
    with mock.patch.dict(sys.modules, names):
        yield


def _synthetic_package(name, path):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    package.__package__ = name
    sys.modules[name] = package
    return package


_ORIGIN = {}


def load_origin():
    """``(nodes, basis, color, masking, schedule, vectors)`` of upstream d28ac6a, loaded once."""
    if "nodes" in _ORIGIN:
        return _ORIGIN["modules"]
    name = "_cc_origin"
    # Imported inside the fake ``comfy`` (whose sys.modules patch also drops these modules again on
    # exit); the objects captured here are the ones nodes.py itself uses.
    with fake_comfy():
        _synthetic_package(name, ORIGIN_DIR)
        nodes = importlib.import_module(f"{name}.nodes")
        lib = {sub: sys.modules[f"{name}.lib_colorcraft.{sub}"]
               for sub in ("basis", "color", "masking", "schedule", "vectors", "debug")}
    assert nodes.resolve_dev.__globals__ is lib["basis"].__dict__
    modules = types.SimpleNamespace(nodes=nodes, **lib)
    _ORIGIN["nodes"] = nodes
    _ORIGIN["modules"] = modules
    return modules


_FORK = {}


def load_fork():
    """The fork's ``core``, ``params`` and ``spec`` (f00066c) — infotext writer and the flux2 tables."""
    if "spec" in _FORK:
        return _FORK["modules"]
    name = "_cc_fork"
    _synthetic_package(name, FORK_DIR)
    _synthetic_package(f"{name}.lib_colorcraft", FORK_DIR / "lib_colorcraft")
    loaded = {}
    for sub in ("core", "params", "spec"):
        spec_ = importlib.util.spec_from_file_location(f"{name}.lib_colorcraft.{sub}",
                                                       FORK_DIR / "lib_colorcraft" / f"{sub}.py")
        module = importlib.util.module_from_spec(spec_)
        sys.modules[spec_.name] = module
        spec_.loader.exec_module(module)
        setattr(sys.modules[f"{name}.lib_colorcraft"], sub, module)
        loaded[sub] = module
    modules = types.SimpleNamespace(**loaded)
    _FORK["spec"] = loaded["spec"]
    _FORK["modules"] = modules
    return modules


def upstream_forge_functions(*names):
    """Top-level functions/constants of upstream's Forge script, executed on their own (no WebUI)."""
    source = origin_body(ORIGIN_DIR / "scripts" / "colorcraft.py")
    tree = ast.parse(source)
    wanted = [node for node in tree.body
              if (isinstance(node, (ast.FunctionDef, ast.Assign))
                  and ((isinstance(node, ast.FunctionDef) and node.name in names)
                       or (isinstance(node, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))))]
    scope = {"re": __import__("re"), "print": lambda *a, **k: None}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "upstream scripts/colorcraft.py", "exec"), scope)  # noqa: S102
    return scope


# ---------------------------------------------------------------------------
# Forge pieces read from the checkout
# ---------------------------------------------------------------------------

_LATENT = {}


def latent_formats():
    """Forge's real latent format module (Wan21, Flux, Flux2, SDXL, SD15 …)."""
    if "module" not in _LATENT:
        path = FORGE / "modules_forge" / "packages" / "huggingface_guess" / "latent.py"
        spec_ = importlib.util.spec_from_file_location("_cc_forge_latent", path)
        module = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(module)
        _LATENT["module"] = module
    return _LATENT["module"]


def forge_function(relpath, name, namespace):
    source = (FORGE / relpath).read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            code = compile(ast.Module(body=[node], type_ignores=[]), str(FORGE / relpath), "exec")
            scope = dict(namespace)
            exec(code, scope)  # noqa: S102 - Forge's own function body
            return scope[name]
    raise AssertionError(f"{name} not found in {relpath}")


def setup_img2img_steps(fix_steps=False):
    # origin: Forge modules/sd_samplers_common.py setup_img2img_steps (read from the checkout)
    return forge_function("modules/sd_samplers_common.py", "setup_img2img_steps",
                          {"opts": types.SimpleNamespace(img2img_fix_steps=fix_steps)})


def forge_modules(fix_steps=False):
    stub = types.ModuleType("modules")
    stub.sd_samplers_common = types.SimpleNamespace(setup_img2img_steps=setup_img2img_steps(fix_steps))
    return stub


def flow_sigmas(steps, shift=3.0):
    """Anima's flow sigmas σ = s·t / (1 + (s − 1)·t), t = 1 → 0 (float32, like Forge)."""
    t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return (shift * t / (1 + (shift - 1) * t)).to(torch.float32)


# ---------------------------------------------------------------------------
# Host stand-ins
# ---------------------------------------------------------------------------


class FakeEncoderModule:
    """Stands in for the VAE's ``nn.Module`` — weakly referenceable like one (the anchor cache needs it)."""


class FakeVAE:
    """``encode`` takes [B,H,W,3] in 0..1 like Forge's and ComfyUI's VAE; deterministic and
    colour-dependent (with a spatial pattern, so the spatial mean matters). Counts its calls."""

    def __init__(self, channels=16, latent_dim=2, size=8):
        self.latent_channels = channels
        self.latent_dim = latent_dim
        self.size = size
        self.first_stage_model = FakeEncoderModule()
        self.calls = 0
        g = torch.Generator().manual_seed(channels * 7 + latent_dim)
        self.mix = torch.randn(channels, 3, generator=g) * 0.8
        self.bias = torch.randn(channels, generator=g) * 0.2
        self.pattern = torch.randn(channels, size, size, generator=g) * 0.05

    def encode(self, img):
        self.calls += 1
        mean = img.reshape(-1, img.shape[-1]).float().mean(0).cpu()
        per_channel = self.mix @ (mean * 2.0 - 1.0) + self.bias
        latent = per_channel[:, None, None] + self.pattern * (1.0 + mean.sum())
        latent = latent.unsqueeze(0)
        if self.latent_dim == 3:
            latent = latent.unsqueeze(2)
        return latent


class FakeUnet:
    """Forge ModelPatcher option handling: ``clone`` deep-copies dicts/lists of ``model_options``
    (backend/utils.py ``deepcopy_``) and keeps everything else by reference."""

    def __init__(self, model_options=None):
        self.model_options = model_options if model_options is not None else {"transformer_options": {}}
        self.clones = 0

    def clone(self):
        self.clones += 1
        return FakeUnet(_deepcopy_(self.model_options))

    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options["sampler_post_cfg_function"] = self.model_options.get("sampler_post_cfg_function", []) + [fn]


def _deepcopy_(obj):
    if isinstance(obj, dict):
        return {k: _deepcopy_(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deepcopy_(v) for v in obj]
    return obj


class StableDiffusionProcessingImg2Img(types.SimpleNamespace):
    """Stands in for modules.processing.StableDiffusionProcessingImg2Img (matched by class name)."""


FAMILIES = {
    # family: (latent format class name, channels, 5-D latents)
    "krea2": ("Wan21", 16, True),
    "zimage": ("Flux", 16, False),
    "flux2": ("Flux2", 128, False),
}


def make_p(family="krea2", latent_format=None, vae=None, unet=None, request_cls=types.SimpleNamespace, sampler=None,
           **fields):
    """A Forge request with ``sd_model.model_config.latent_format`` / ``forge_objects``."""
    if latent_format is None and family is not None:
        latent_format = getattr(latent_formats(), FAMILIES[family][0])()
    if vae is None:
        channels = FAMILIES[family][1] if family else 4
        vae = FakeVAE(channels=channels, latent_dim=3 if family and FAMILIES[family][2] else 2)
    objects = types.SimpleNamespace(unet=unet if unet is not None else FakeUnet(), vae=vae)
    sd_model = types.SimpleNamespace(model_config=types.SimpleNamespace(latent_format=latent_format),
                                     forge_objects=objects)
    values = dict(is_hr_pass=False, steps=10, extra_generation_params={}, sampler=sampler)
    values.update(fields)
    return request_cls(sd_model=sd_model, **values)


def latent(family, seed, batch=2, size=24, dtype=torch.float32):
    _, channels, five_d = FAMILIES[family]
    g = torch.Generator().manual_seed(seed)
    shape = (batch, channels, 1, size, size) if five_d else (batch, channels, size, size)
    return (torch.randn(shape, generator=g) * 1.2).to(dtype)


# ---------------------------------------------------------------------------
# Scenarios: one set-up, two hosts
# ---------------------------------------------------------------------------


@dataclass
class Mod:
    kind: str = "advanced"
    values: dict = field(default_factory=dict)      # panel field names (spec.MODIFIER_FIELDS)
    mask: str = "none"


@dataclass
class Scenario:
    label: str
    mods: list
    leaves: dict = field(default_factory=dict)      # "M1": {leaf field: value}
    combos: dict = field(default_factory=dict)      # "C1": {combo field: value}
    masking: bool = False


def scenario_config(scenario):
    """This extension's ``Config`` for the scenario (modifier i on tab i)."""
    from sam3ext.colorcraft import spec

    config = spec.default_config()
    config.enabled = True
    config.masking = scenario.masking
    for i, mod in enumerate(scenario.mods):
        modifier = config.modifiers[i]
        modifier["active"] = True
        modifier["kind"] = spec.KIND_LABELS[mod.kind]
        modifier["mask"] = mod.mask
        modifier.update(mod.values)
    for tag, values in scenario.leaves.items():
        config.leaves[spec.MASK_TAGS.index(tag)].update(values)
    for tag, values in scenario.combos.items():
        config.combos[spec.COMBO_TAGS.index(tag)].update(values)
    return config


def scenario_args(scenario):
    from sam3ext.colorcraft import spec

    return spec.config_to_args(scenario_config(scenario))


def oracle_chain(scenario, nodes):
    """The same set-up built with the ComfyUI node classes, the way a graph would wire it."""
    from sam3ext.colorcraft import spec

    config = scenario_config(scenario)

    def leaf_node(tag):
        leaf = config.leaves[spec.MASK_TAGS.index(tag)]
        (mask,) = nodes.ColorcraftMasking().make(
            mask_mode=leaf["mask_mode"], mask_axis=leaf["mask_axis"], mask_strength=leaf["mask_strength"],
            mask_width=leaf["mask_width"], mask_center=leaf["mask_center"], mask_hardness=leaf["mask_hardness"])
        return _blur_node(mask, leaf)

    def _blur_node(mask, values):
        if values["blur"] != 0 or values["spread"] != 0 or values["contrast"] != 0 or values["normalize"]:
            (mask,) = nodes.ColorcraftMaskBlur().make(mask, values["blur"], values["spread"], values["contrast"],
                                                       values["normalize"])
        return mask

    def mask_node(ref):
        if ref == "none":
            return None
        if ref.startswith("M"):
            return leaf_node(ref)
        combo = config.combos[spec.COMBO_TAGS.index(ref)]
        a, b = mask_node(combo["mask_a"]), mask_node(combo["mask_b"])
        if a is None or b is None:
            return None
        (mask,) = nodes.ColorcraftMaskCombine().make(a, b, combo["operation"])
        return _blur_node(mask, combo)

    chain = None
    for i, mod in enumerate(scenario.mods):
        m = config.modifiers[i]
        sched = {
            "strength": m["strength"], "start": m["start"], "end": m["end"], "advanced": True,
            "bias": m["bias"] if m["advanced"] else 0.5,
            "exponent": m["exponent"] if m["advanced"] else 0.0,
            "start_off": m["start_off"] if m["advanced"] else 0.0,
            "end_off": m["end_off"] if m["advanced"] else 0.0,
            "smooth": m["smooth"], "plot_steps": 8,
        }
        masking = mask_node(m["mask"]) if config.masking else None
        shift = dict(color_shift_amount=m["color_shift_amount"], mode=m["color_shift_mode"],
                     red=m["color_shift_red"], green=m["color_shift_green"], blue=m["color_shift_blue"],
                     brightness=m["color_shift_brightness"])
        if mod.kind == "basic":
            (chain,) = nodes.ColorcraftBasic().make(modifiers=chain, contrast=m["contrast"], **sched, **shift)
        elif mod.kind == "advanced":
            (chain,) = nodes.ColorcraftAdvanced().make(
                modifiers=chain, masking=masking, **sched,
                exposure=m["exposure"], tone_compression=m["tone_compression"], contrast=m["contrast"],
                clarity=m["clarity"], sharpness=m["sharpness"], temperature=m["temperature"], tint=m["tint"],
                vibrance=m["vibrance"], saturation=m["saturation"], chroma_contrast=m["chroma_contrast"],
                chroma_center=m["chroma_center"], more_colors=m["more_colors"],
                temp_plus_tint=m["temp_plus_tint"], temp_minus_tint=m["temp_minus_tint"], lab_a=m["lab_a"],
                lab_b=m["lab_b"], lab_a_plus_b=m["lab_a_plus_b"], lab_a_minus_b=m["lab_a_minus_b"],
                color_shift=m["color_shift"], **shift, dev=True,
                recenter_override=m["recenter_override"], recenter=m["recenter"],
                max_chroma_override=m["max_chroma_override"], max_chroma=m["max_chroma"],
                chroma_plane_override=m["chroma_plane_override"], chroma_plane=m["chroma_plane"])
        else:
            (schedule,) = nodes.ColorcraftSchedule().make(**sched)
            if mod.kind == "luma":
                (chain,) = nodes.ColorcraftLuma().make(schedule, modifiers=chain, masking=masking,
                                                       exposure=m["exposure"], tone_compression=m["tone_compression"])
            elif mod.kind == "chroma":
                (chain,) = nodes.ColorcraftChroma().make(
                    schedule, modifiers=chain, masking=masking, temperature=m["temperature"], tint=m["tint"],
                    vibrance=m["vibrance"], saturation=m["saturation"], chroma_contrast=m["chroma_contrast"],
                    chroma_center=m["chroma_center"])
            elif mod.kind == "chroma_plus":
                (chain,) = nodes.ColorcraftChromaPlus().make(
                    schedule, modifiers=chain, masking=masking, temp_plus_tint=m["temp_plus_tint"],
                    temp_minus_tint=m["temp_minus_tint"], lab_a=m["lab_a"], lab_b=m["lab_b"],
                    lab_a_plus_b=m["lab_a_plus_b"], lab_a_minus_b=m["lab_a_minus_b"])
            elif mod.kind == "punch":
                (chain,) = nodes.ColorcraftPunch().make(schedule, modifiers=chain, masking=masking,
                                                        contrast=m["contrast"], clarity=m["clarity"],
                                                        sharpness=m["sharpness"])
            elif mod.kind == "shift":
                (chain,) = nodes.ColorcraftShift().make(schedule, modifiers=chain, masking=masking, **shift)
    return chain


@contextlib.contextmanager
def oracle_family(origin, family, fork=None):
    """Point the oracle at our vector files; for flux2 extend its tables the way the fork does."""
    from sam3ext.colorcraft import basis as ours

    patches = [mock.patch.object(origin.nodes, "VECTORS_DIR", ours.VECTORS_DIR)]
    if family == "flux2":
        fork = fork or load_fork()
        row = dict(fork.core.MODEL_DEV_DEFAULTS["flux2"], detail_scale=4.0)
        patches += [
            mock.patch.object(origin.nodes, "BASIS_FAMILIES", origin.basis.BASIS_FAMILIES + ["flux2"]),
            mock.patch.object(origin.nodes, "LATENT_FORMAT_TO_FAMILY",
                              dict(origin.basis.LATENT_FORMAT_TO_FAMILY, Flux2="flux2")),
            mock.patch.object(origin.basis, "MODEL_DEV_DEFAULTS", dict(origin.basis.MODEL_DEV_DEFAULTS, flux2=row)),
            mock.patch.object(origin.nodes, "VAE_DOWNSCALE_FACTOR", fork.core.VAE_DOWNSCALE_FACTORS["flux2"]),
        ]
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        yield


class _ComfyModel:
    def __init__(self, latent_format):
        self.latent_format = latent_format


def oracle_callback(origin, chain, vae, walked, latent_format):
    """Upstream ``ColorcraftSampler.wrap`` driven far enough to hand back its post-CFG function."""
    captured = {}

    class BaseSampler:
        extra_options = {}
        inpaint_options = {}

        @staticmethod
        def sampler_function(model, x, sigmas, *args, extra_args=None, **kwargs):
            captured["fn"] = extra_args["model_options"]["sampler_post_cfg_function"][-1]
            return x

    with mock.patch("builtins.print"):
        (wrapped, _debug) = origin.nodes.ColorcraftSampler().wrap(BaseSampler, vae, chain)
        wrapped.sampler_function(None, torch.zeros(1), walked, extra_args={"model_options": {}})
    fn = captured["fn"]
    model = _ComfyModel(latent_format)

    def call(x0, sigma):
        with mock.patch("builtins.print"):
            return fn({"denoised": x0, "sigma": torch.tensor([float(sigma)]), "model": model})

    return call


def forge_args(x0, sigma, unet, batch=None):
    batch = x0.shape[0] if batch is None else batch
    return {
        "denoised": x0, "cond": [{}], "uncond": [{}], "cond_scale": 4.0, "model": types.SimpleNamespace(),
        "uncond_denoised": x0, "cond_denoised": x0, "sigma": torch.full((batch,), float(sigma)),
        "model_options": unet.model_options, "input": x0,
    }


def attach_ours(p, args, *, modules_stub=None):
    """``process_before_every_sampling`` of the hook; returns the attached callback (or None)."""
    from sam3ext.colorcraft import hook

    stub = modules_stub if modules_stub is not None else forge_modules()
    with mock.patch.dict(sys.modules, {"modules": stub}), mock.patch.object(hook, "_log"):
        hook.process(p, args)
    unet = p.sd_model.forge_objects.unet
    callbacks = (getattr(unet, "model_options", None) or {}).get("sampler_post_cfg_function", [])
    return callbacks[-1] if callbacks and hook.is_owned(callbacks[-1]) else None


def _sched(start=0.0, end=1.0, strength=1.0, **shaping):
    values = {"start": start, "end": end, "strength": strength}
    if shaping:
        values["advanced"] = True
        values.update(shaping)
    return values


def _leaf(axis, mode="highs", **values):
    return dict({"mask_axis": axis, "mask_mode": mode}, **values)


# Chosen so every branch of the node's post-CFG body runs: each node kind, both gates on and off, both
# colour-shift modes, the dev overrides, every mask axis family (linear, hue, saturation, the newer
# clarity/sharpness detail axes), every mask mode and combine operation, nested combos, blur → spread →
# normalize → contrast refinement, schedule shaping (including a negative exponent and offsets) and a
# negative strength. Amounts are within upstream's Forge slider ranges.
SCENARIOS = [
    Scenario("advanced-luma-contrast", [Mod("advanced", dict(_sched(), exposure=0.4, tone_compression=0.2, contrast=0.3))]),
    Scenario("advanced-chroma", [Mod("advanced", dict(_sched(), temperature=0.5, tint=-0.3, vibrance=0.6, saturation=0.4,
                                                       chroma_contrast=0.35, chroma_center=0.2))]),
    Scenario("advanced-detail", [Mod("advanced", dict(_sched(), clarity=0.5, sharpness=-0.4, tone_compression=0.3))]),
    Scenario("advanced-chroma-plus", [Mod("advanced", dict(_sched(), temp_plus_tint=0.3, temp_minus_tint=-0.2, lab_a=0.25,
                                                            lab_b=-0.15, lab_a_plus_b=0.1, lab_a_minus_b=-0.1))]),
    Scenario("advanced-chroma-plus-gated-off", [Mod("advanced", dict(_sched(), more_colors=False, temp_plus_tint=0.9,
                                                                      lab_a=0.9, exposure=0.2))]),
    Scenario("advanced-color-shift", [Mod("advanced", dict(_sched(), color_shift_amount=0.5, color_shift_red=0.4,
                                                            color_shift_green=-0.2, color_shift_blue=0.3,
                                                            color_shift_brightness=0.1))]),
    Scenario("advanced-color-shift-gated-off", [Mod("advanced", dict(_sched(), color_shift=False, color_shift_amount=0.5,
                                                                      color_shift_red=0.4, exposure=0.2))]),
    Scenario("advanced-color-shift-legacy", [Mod("advanced", dict(_sched(), color_shift_amount=0.5, color_shift_red=0.4,
                                                                   color_shift_mode="legacy", contrast=-0.25))]),
    Scenario("advanced-dev-overrides", [Mod("advanced", dict(_sched(), vibrance=0.5, saturation=-0.3, chroma_contrast=0.4,
                                                              recenter_override=True, recenter=0.9,
                                                              max_chroma_override=True, max_chroma=4.0))]),
    Scenario("advanced-dev-lab-plane", [Mod("advanced", dict(_sched(), temperature=0.5, vibrance=0.4, chroma_contrast=-0.3,
                                                              chroma_plane_override=True, chroma_plane="lab"))]),
    Scenario("basic", [Mod("basic", dict(_sched(0.2, 0.9), contrast=0.6, color_shift_amount=0.4, color_shift_red=-0.3,
                                          color_shift_blue=0.5, color_shift_brightness=-0.2))]),
    Scenario("luma-masked-exposure-highs", [Mod("luma", dict(_sched(), exposure=0.5, tone_compression=-0.4), mask="M1")],
             leaves={"M1": _leaf("exposure", "highs", mask_center=0.1, mask_hardness=2.0)}, masking=True),
    Scenario("chroma-masked-hue-range", [Mod("chroma", dict(_sched(), temperature=0.6, tint=0.2, vibrance=0.3,
                                                             saturation=0.2, chroma_contrast=0.5, chroma_center=-0.4),
                                             mask="M2")],
             leaves={"M2": _leaf("hue", "range", mask_center=0.3, mask_width=0.5, mask_hardness=3.0, mask_strength=0.8)},
             masking=True),
    Scenario("chroma-plus-masked-saturation-protect", [Mod("chroma_plus", dict(_sched(), lab_a=0.4, lab_b=-0.3,
                                                                                temp_plus_tint=0.2), mask="M3")],
             leaves={"M3": _leaf("saturation", "protect range", mask_center=0.2, mask_width=0.4)}, masking=True),
    Scenario("punch-masked-split", [Mod("punch", dict(_sched(), contrast=0.4, clarity=0.6, sharpness=0.3), mask="M4")],
             leaves={"M4": _leaf("temperature", "split", mask_hardness=1.5)}, masking=True),
    Scenario("shift-masked-combo-refined", [Mod("shift", dict(_sched(0.1, 0.8), color_shift_amount=0.7,
                                                               color_shift_red=0.5, color_shift_green=-0.5),
                                                mask="C1")],
             leaves={"M1": _leaf("exposure", "highs"), "M2": _leaf("tint", "lows", mask_center=0.1, mask_hardness=2.0)},
             combos={"C1": {"mask_a": "M1", "mask_b": "M2", "operation": "and", "blur": 24.0, "spread": 0.5,
                            "contrast": 1.5, "normalize": True}},
             masking=True),
    Scenario("detail-mask-axes-nested-combos", [Mod("advanced", dict(_sched(), exposure=0.45, temperature=0.3), mask="C2")],
             leaves={"M1": _leaf("clarity", "highs", mask_center=0.05, blur=8.0, normalize=True, contrast=2.0),
                     "M2": _leaf("sharpness", "lows", mask_hardness=4.0, spread=-0.5),
                     "M3": _leaf("lab-a+b", "range", mask_width=0.6, mask_strength=0.7)},
             combos={"C1": {"mask_a": "M1", "mask_b": "M2", "operation": "or"},
                     "C2": {"mask_a": "C1", "mask_b": "M3", "operation": "subtract", "contrast": -1.0}},
             masking=True),
    Scenario("xor-combo-and-unmasked-neighbour", [
        Mod("advanced", dict(_sched(0.0, 0.6), exposure=0.3), mask="C1"),
        Mod("advanced", dict(_sched(0.4, 1.0), tint=0.4)),
    ], leaves={"M5": _leaf("temp-tint", "highs"), "M6": _leaf("lab-b", "lows")},
        combos={"C1": {"mask_a": "M5", "mask_b": "M6", "operation": "xor", "blur": 3.0}}, masking=True),
    Scenario("stack-luma-chroma-punch-shaping", [
        Mod("luma", dict(_sched(0.0, 0.5, bias=0.3, exponent=1.5, start_off=0.1, end_off=-0.1, smooth=False),
                         exposure=0.3)),
        Mod("chroma", dict(_sched(0.3, 0.9, bias=0.7, exponent=-2.0), vibrance=0.5, temperature=-0.2)),
        Mod("punch", dict(_sched(0.5, 1.0, strength=0.8), clarity=0.4, contrast=0.2)),
    ]),
    Scenario("negative-strength-and-offsets", [Mod("advanced", dict(_sched(0.2, 0.7, strength=-1.0, start_off=0.2,
                                                                          end_off=0.3, bias=0.5),
                                                                    exposure=-0.5, contrast=0.4))]),
]


def midpoints(walked):
    """The schedule sigmas a first-order sampler visits, then the midpoints a second-order one adds."""
    return ([float(walked[i]) for i in range(len(walked) - 1)]
            + [float((walked[i] + walked[i + 1]) / 2) for i in range(len(walked) - 1)])


def parity_run(test, family, scenario, walked, full=None, evals=None, p_fields=None, modules_stub=None,
               request_cls=types.SimpleNamespace):
    """Our hook (published list ``full``, its own offset) against the node handed ``walked``.

    Asserts bit-identity per evaluation; returns ``(moved, total, p)``."""
    origin, fork = load_origin(), load_fork()
    p = make_p(family, request_cls=request_cls, **(p_fields or {}))
    vae = p.sd_model.forge_objects.vae
    latent_format = p.sd_model.model_config.latent_format
    callback = attach_ours(p, scenario_args(scenario), modules_stub=modules_stub)
    test.assertIsNotNone(callback, scenario.label)
    unet = p.sd_model.forge_objects.unet
    unet.model_options.setdefault("transformer_options", {})["sampling_sigmas"] = full if full is not None else walked
    evals = midpoints(walked) if evals is None else evals
    moved = 0
    with oracle_family(origin, family, fork):
        oracle = oracle_callback(origin, oracle_chain(scenario, origin.nodes),
                                 FakeVAE(vae.latent_channels, vae.latent_dim), walked, latent_format)
        for k, sigma in enumerate(evals):
            x0 = latent(family, 1000 + k)
            ours = callback(forge_args(x0.clone(), sigma, unet))
            theirs = oracle(x0.clone(), sigma)
            test.assertTrue(torch.equal(ours, theirs),
                            f"{family} {scenario.label} eval {k} sigma={sigma}: max |d| "
                            f"{(ours - theirs).abs().max().item()}")
            moved += not torch.equal(ours, x0)
    return moved, len(evals), p
