# Colorcraft (sam-extra) — the Forge hook: a post-CFG function on a cloned UNet, per sampling pass.
#
# Structure after aoleg/ComfyUI-Colorcraft@f00066c63c9d8f96abc119cada3b51b36688fcac:scripts/colorcraft_neo.py
# (Oleg Afonin's Forge Neo port: post-CFG function registered on a UNet clone in
# process_before_every_sampling, sigma list read lazily from transformer_options, latent format from
# p.sd_model.model_config, colour anchors VAE-encoded before sampling). The per-evaluation math is
# muerrilla/ComfyUI-Colorcraft@d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf (engine.py); color_key is upstream's
# anchor cache key (nodes.py get_color_latent) and the step counter upstream's Forge one (schedule.py).
#
# MIT License
#
# Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla)
# (The fork aoleg/ComfyUI-Colorcraft — Oleg Afonin's Forge Neo port — is under the same MIT License;
#  its LICENSE carries the upstream notice with the earlier spelling "Sahand Ahmadiantehrani".)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This file is sam-extra code (2026-10-03) on those structures. Differences from the fork's hook, all deliberate:
#   * owner-tagged attach/detach like scripts/anima_cfg_optimal_scale.py; appended last, so it runs
#     after every other sam-extra post-CFG function of the pass (Optimal Scale, the guidance suite);
#   * img2img and hires sample ``sampling_sigmas[offset:]`` — the schedule is built on that slice
#     (the list a ComfyUI sampler would be handed), offset from Forge's own setup_img2img_steps;
#   * anchors: ``vae.encode`` + ``latent_format.process_in``, never ``encode_first_stage`` (Anima's
#     override stores the encoded image as its reference latent); kept across requests for the same VAE
#     (bounded, dropped when the VAE goes away) so XYZ cells and API calls do not encode again;
#   * an exception passes the input through and is written to the status infotext key;
#   * float16/bfloat16 predictions are graded in float32 and returned in their own dtype;
#   * samplers that publish no sigma list (CompVis DDIM/PLMS) use upstream's Forge step counter;
#   * sampling runs other than the pass's own (an extension sampling again from postprocess while the
#     clone is still installed) are passed through and counted, like the node only grades its sampler;
#     the run's list re-published by Anima SPEED after a transition (tagged by
#     ``sam3ext.guidance.sigmas.mark_republished``) is the same run, looked up on its patched sigmas;
#   * the extra Dy/SMEA sub-step evaluations of Euler (SMEA) Dy CFG++ (``sam_extra_substep``) are
#     passed through and counted: one grade per sampler step (neither upstream has the other);
#   * on SPEED's coarse grid the mask blur keeps its image-pixel size (``grid_downscale``, from the
#     pass's latent grid Forge hands ``process_before_every_sampling`` as ``x``).
"""Attach Colorcraft to one Forge sampling pass and run the chain on every model evaluation."""

from __future__ import annotations

import sys
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from sam3ext.guidance.dave_gate import ISCLOSE_ATOL, ISCLOSE_RTOL, pre_dd_sigma
from sam3ext.guidance.sigmas import forge_sampling_offset, republished_from, sampler_publishes_sigmas

try:  # Colorcraft must keep working even if the extra-sampler package fails to import
    from sam3ext.extra_samplers.common import SUBSTEP_MARKER
except Exception:  # noqa: BLE001 - same literal as sam3ext/extra_samplers/common.py
    SUBSTEP_MARKER = "sam_extra_substep"

from . import basis as basis_mod
from . import debug_panel
from . import panel_state, spec
from .color import build_color_latent, to_model_space
from .engine import apply_chain, build_entries, needed_colors
from .schedule import forge_step_position, sigma_to_value

OWNER = "sam-extra/colorcraft"
OWNER_ATTR = "_sam_extra_colorcraft_owner"
STATE_ATTR = "_sam3_colorcraft"
REFUSED_ATTR = "_sam3_colorcraft_refused"
INFOTEXT_PRE_DD = "SAM Extra Colorcraft pre-DD sigma"
OPT_PRE_DD = "sam3_colorcraft_pre_dd_sigma"
OPT_LOG = "sam3_colorcraft_log"

_HALF = (torch.float16, torch.bfloat16)


def _log(message):
    print(f"[Colorcraft] {message}", file=sys.stderr)


def _option(name, default):
    try:
        from modules import shared

        return bool(getattr(shared.opts, name, default))
    except Exception:
        return default


def inner_pass(p):
    """SAM3 in-flight inpaint / Refine (``_sam3_inner``) and ADetailer (``_ad_inner``) inner img2img."""
    return bool(getattr(p, "_sam3_inner", False) or getattr(p, "_ad_inner", False))


# ---------------------------------------------------------------------------
# Owner-tagged attach / detach
# ---------------------------------------------------------------------------


def is_owned(fn):
    return getattr(fn, OWNER_ATTR, None) == OWNER


def attach(unet, callback):
    """Keep every foreign post-CFG function, drop a stale one of ours, append ``callback`` last."""
    options = dict(getattr(unet, "model_options", {}) or {})
    previous = options.get("sampler_post_cfg_function", [])
    if not isinstance(previous, (list, tuple)):
        raise TypeError("sampler_post_cfg_function is not a callback list")
    setattr(callback, OWNER_ATTR, OWNER)
    options["sampler_post_cfg_function"] = [fn for fn in previous if not is_owned(fn)] + [callback]
    unet.model_options = options


def detach_owned(p):
    """Remove only this script's callback when a host hands back an already patched UNet.

    Forge resets ``forge_objects`` before every pass, so normally there is nothing to do and nothing
    is cloned. Metadata is left alone: the base pass's record has to survive a hires pass."""
    model = getattr(p, "sd_model", None)
    objects = getattr(model, "forge_objects", None)
    unet = getattr(objects, "unet", None)
    options = getattr(unet, "model_options", {}) or {}
    callbacks = options.get("sampler_post_cfg_function", [])
    if isinstance(callbacks, (list, tuple)) and any(is_owned(fn) for fn in callbacks):
        candidate = unet.clone()
        copied = dict(candidate.model_options)
        copied["sampler_post_cfg_function"] = [fn for fn in callbacks if not is_owned(fn)]
        candidate.model_options = copied
        objects.unet = candidate
        return True
    return False


# ---------------------------------------------------------------------------
# Basis vectors and anchors
# ---------------------------------------------------------------------------

_BASIS_CACHE: dict = {}
_BASIS_DEVICE_CACHE: dict = {}


def load_family_basis(family):
    """The family's vectors (cloned out of the safetensors mapping, so the file stays replaceable)."""
    if family not in _BASIS_CACHE:
        loaded = basis_mod.load_basis(family, basis_mod.VECTORS_DIR)
        _BASIS_CACHE[family] = None if loaded is None else {k: v.detach().clone() for k, v in loaded.items()}
    return _BASIS_CACHE[family]


def basis_on(family, device, dtype):
    """Upstream ``get_basis``: the family's vectors on ``device`` in ``dtype`` (cached)."""
    key = (family, str(device), dtype)
    if key not in _BASIS_DEVICE_CACHE:
        source = load_family_basis(family)
        _BASIS_DEVICE_CACHE[key] = {k: v.to(device=device, dtype=dtype) for k, v in source.items()}
    return _BASIS_DEVICE_CACHE[key]


class _RankRecorder:
    """Wraps the VAE so the anchor's model-space lift uses the rank ``vae.encode`` really returned."""

    def __init__(self, vae):
        self.vae = vae
        self.rank = None

    def encode(self, img):
        latent = self.vae.encode(img)
        self.rank = latent.dim() - 2
        return latent


def _anchor_device():
    try:
        from modules import devices

        return devices.device
    except Exception:
        return torch.device("cpu")


def color_key(color):
    """Upstream's anchor cache key (nodes.py ``get_color_latent``): rounded to 4 decimals."""
    return tuple(round(float(c), 4) for c in color)


# Anchors outlive the request: XYZ cells, the API and repeated generations with the same VAE reuse them
# instead of loading the VAE for a 512x512 encode each time (upstream's ComfyUI node keeps its colour
# cache on the sampler object, which ComfyUI reuses while its inputs are unchanged). Entry: key ->
# (weak reference to the encoder module, anchor). The key holds everything the encode depends on — the
# encoder module's identity, its compute dtype and device, the weight patches on its ModelPatcher
# (``patches_uuid``, replaced whenever patches are added), the latent format (hence the family and
# ``process_in``), the device the colour image is built on, and upstream's rounded colour. The weak
# reference is the invalidation: Forge builds a new VAE module whenever the checkpoint or VAE selection
# changes (modules/sd_models.py ``forge_model_reload``); once the old one is gone its entries are dropped,
# and an entry never answers for a new module that happens to get the old one's ``id``. An encoder that
# cannot be weakly referenced is cached for the request only. Bounded, least recently used first out.
ANCHOR_CACHE_LIMIT = 32
_ANCHOR_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()


def _encoder(vae):
    """The module the anchors are encoded with (the VAE object itself when it has no first stage)."""
    module = getattr(vae, "first_stage_model", None)
    return vae if module is None else module


def _anchor_key(vae, module, latent_format, device, colour):
    patcher = getattr(vae, "patcher", None)
    return (
        id(module), str(getattr(vae, "vae_dtype", None)), str(getattr(vae, "device", None)),
        str(getattr(patcher, "patches_uuid", None)),
        type(latent_format).__name__ if latent_format is not None else "none",
        str(device), colour,
    )


def _cached_anchor(module, key):
    entry = _ANCHOR_CACHE.get(key)
    if entry is None:
        return None
    ref, anchor = entry
    if ref() is not module:          # its encoder is gone (and the id was reused): never answer for another
        del _ANCHOR_CACHE[key]
        return None
    _ANCHOR_CACHE.move_to_end(key)
    return anchor


def _remember_anchor(module, key, anchor):
    try:
        ref = weakref.ref(module)
    except TypeError:
        return
    for stale in [k for k, (r, _) in _ANCHOR_CACHE.items() if r() is None]:
        del _ANCHOR_CACHE[stale]     # anchors of VAEs that no longer exist (a model reload)
    _ANCHOR_CACHE[key] = (ref, anchor)
    _ANCHOR_CACHE.move_to_end(key)
    while len(_ANCHOR_CACHE) > ANCHOR_CACHE_LIMIT:
        _ANCHOR_CACHE.popitem(last=False)


def clear_anchor_cache():
    _ANCHOR_CACHE.clear()


def encode_anchors(vae, latent_format, colors, cache, device=None):
    """Model-space anchors for ``colors`` — ``vae.encode`` of a flat 512x512 image + ``process_in``.

    Looked up in ``cache`` (this request: the hires pass reuses the base pass's) and then in the
    cross-request cache above; only a colour neither holds is encoded. Runs before sampling, under
    inference mode. A cached anchor is the very tensor the encode produced, so it is bit-identical."""
    device = device if device is not None else _anchor_device()
    module = _encoder(vae)
    anchors = {}
    with torch.inference_mode():
        for color in colors:
            key = color_key(color)
            if key in anchors:
                continue
            cache_key = _anchor_key(vae, module, latent_format, device, key)
            anchor = cache.get(cache_key)
            if anchor is None:
                anchor = _cached_anchor(module, cache_key)
            if anchor is None:
                recorder = _RankRecorder(vae)
                raw = build_color_latent(recorder, device, *color)
                dims = recorder.rank if recorder.rank is not None else getattr(vae, "latent_dim", 2)
                anchor = to_model_space(latent_format, raw, dims).detach().float()
                _remember_anchor(module, cache_key, anchor)
            cache[cache_key] = anchor
            anchors[key] = anchor
    return anchors


# ---------------------------------------------------------------------------
# Status infotext
# ---------------------------------------------------------------------------


@dataclass
class PassStatus:
    name: str
    family: str = ""
    tags: list = field(default_factory=list)
    evals: int = 0
    edited: int = 0
    idle: int = 0
    foreign: int = 0
    substeps: int = 0
    errors: int = 0
    last_error: str = ""
    nonfinite: int = 0
    notes: list = field(default_factory=list)
    attached: bool = False

    def render(self):
        parts = [self.family] if self.family else []
        if self.tags:
            parts.append("mods " + "+".join(self.tags))
        if self.attached:
            parts.append(f"{self.evals} evals, {self.edited} edited" if self.evals else "pending model evaluation")
        if self.substeps:
            parts.append(f"Dy/SMEA sub-steps passed through x{self.substeps}")
        if self.foreign:
            parts.append(f"other sampling run passed through x{self.foreign}")
        if self.nonfinite:
            parts.append(f"non-finite result passed through x{self.nonfinite}")
        if self.errors:
            parts.append(f"error {self.last_error} x{self.errors} (input passed through)")
        parts += self.notes
        return f"{self.name}: " + "; ".join(parts)


def _request_state(p):
    state = getattr(p, STATE_ATTR, None)
    if not isinstance(state, dict):
        state = {"passes": {}, "anchors": {}, "warned": set()}
        setattr(p, STATE_ATTR, state)
    return state


def _params(p):
    params = getattr(p, "extra_generation_params", None)
    if not isinstance(params, dict):
        params = {}
        p.extra_generation_params = params
    return params


def write_status(p):
    state = _request_state(p)
    passes = state["passes"]
    text = " | ".join(passes[name].render() for name in ("base", "hires") if name in passes)
    if text:
        _params(p)[spec.STATUS_KEY] = text


def _refuse(p, exc):
    """Script arguments the reader will not guess at (``panel_state.RefusedArgs``, e.g. v0.31.0's 579 positional
    values cut to 67 by Forge's API): nothing is attached, the status says why, and the console gets one line
    per processing job (``REFUSED_ATTR`` keeps its hires pass quiet; an XYZ grid gives every cell its own copy of
    ``p``, so one line per cell, like the status). A script cannot fail an API request, so the image is made
    without Colorcraft."""
    reason = f"unreadable panel state ({exc})" if isinstance(exc, panel_state.UnreadableState) else str(exc)
    _params(p)[spec.STATUS_KEY] = f"not applied: {reason}"
    if not getattr(p, REFUSED_ATTR, False):
        setattr(p, REFUSED_ATTR, True)
        _log(f"not applied: {reason}")


# ---------------------------------------------------------------------------
# The per-pass run state and the post-CFG function
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """Everything one pass's callback needs. Built in ``process_before_every_sampling``."""
    entries: list
    tags: list
    family: str | None
    downscale: int
    anchors: dict
    offset: int | None
    publishes: bool
    stale_sigmas: object
    denoiser: object
    pre_dd: bool
    log: bool
    status: PassStatus
    p: object = None
    params: dict = None
    run_src: object = None          # the sampling_sigmas object of the pass's own run
    walked: object = None           # its walked part, on the CPU
    walked_src: object = None       # the list ``walked`` was read from (run_src, or SPEED's re-published copy)
    walk_offset: int = 0            # where ``walked`` starts in that list
    built: list = None
    counter_built: dict = field(default_factory=dict)
    device_anchors: dict = field(default_factory=dict)
    pre_dd_written: bool = False
    debug_step: float | None = None  # Debug panel: capture the pre-edit x0 at this step (None = off)
    state: dict = None               # the request's state (where the capture goes)
    step_index: int | None = None    # step of the current evaluation, worked out only for the capture
    n_steps: int = 0
    debug_done: bool = False
    full_hw: tuple | None = None     # the pass's latent grid (Forge's ``x``); SPEED samples a smaller one first


def latent_grid(x):
    """``(H, W)`` of a 4-D/5-D latent (Forge's ``process_before_every_sampling`` ``x``), else None."""
    if torch.is_tensor(x) and x.dim() >= 4:
        return int(x.shape[-2]), int(x.shape[-1])
    return None


def grid_downscale(run, x0):
    """Image pixels per latent pixel of the grid ``x0`` is on — the unit of the mask blur.

    Upstream divides the blur by the VAE factor, i.e. the panel's blur is in image pixels. Anima SPEED
    samples the first steps on a smaller grid of the same image, where one latent pixel covers
    ``full / coarse`` times more of it; scaling the factor keeps the blur's image-pixel size there (and
    a radius that fits the pass's grid fits the coarse one). The pass's own grid gets the family's
    factor unchanged."""
    full = run.full_hw
    h, w = int(x0.shape[-2]), int(x0.shape[-1])
    if full is None or (h, w) == tuple(full) or h <= 0 or w <= 0:
        return run.downscale
    return run.downscale * 0.5 * (full[0] / h + full[1] / w)


def lookup_sigma(args, use_pre_dd):
    """The sigma to look the schedule up by: upstream's ``sigma.max().item()``.

    With Detail Daemon on, the sigma handed to the model was scaled in ``on_cfg_denoiser``; when the
    pre-DD setting is on (default) the sampler's own sigma, noted by Detail Daemon, is used instead —
    the schedule follows the sampler's steps. Returns ``(value, scaled_by_dd)``."""
    sigma = args["sigma"]
    noted = pre_dd_sigma(sigma)
    scaled = noted is not None
    if use_pre_dd and scaled:
        sigma = noted
    if torch.is_tensor(sigma):
        return float(sigma.max().item()), scaled
    return float(sigma), scaled


def _start_index(full, sigma):
    """Index of the first entry of ``full`` isclose to ``sigma`` (DAVE's rule), else None."""
    values = full.flatten().tolist()
    for index, value in enumerate(values[:-1]):
        if abs(value - sigma) <= ISCLOSE_ATOL + ISCLOSE_RTOL * abs(sigma):
            return index
    return None


def _published_sigmas(args):
    """Forge's ``transformer_options['sampling_sigmas']`` for this call (the whole list), or None."""
    options = args.get("model_options")
    if not isinstance(options, dict):
        return None
    transformer = options.get("transformer_options")
    if not isinstance(transformer, dict):
        return None
    return transformer.get("sampling_sigmas")


def _values_by_sigma(run, args, sigmas_src):
    """Schedule values for this evaluation from the sigma list (the ComfyUI node's lookup)."""
    if run.run_src is None:
        run.run_src = sigmas_src
        run.walked_src = sigmas_src
        full = sigmas_src.detach().to("cpu", copy=True).flatten()
        offset = run.offset
        if offset is None:
            first, _ = lookup_sigma(args, run.pre_dd)
            offset = _start_index(full, first) or 0
            run.status.notes.append(f"offset from first sigma ({offset})")
        run.walk_offset = offset if 0 < offset < len(full) - 1 else 0
        full = full[run.walk_offset:]
        run.walked = full
        run.built = build_entries(run.entries, len(full) - 1)
        if run.log:
            _log(f"{run.status.name} pass: {len(full) - 1} steps, sigmas "
                 f"{[round(float(s), 4) for s in full]}")
            for tag, (schedule, kind, _, _) in zip(run.tags, run.built):
                _log(f"  {tag} ({kind}): {[round(float(v), 4) for v in schedule]}")
    elif sigmas_src is not run.walked_src:
        if sigmas_src is not run.run_src and republished_from(sigmas_src) is not run.run_src:
            return None
        # Anima SPEED re-published this run's list after a transition (the aligned sigma, or the
        # re-spaced tail) and puts the original back at the end: the step count is the same, and the
        # schedule is looked up on the sigmas the sampler walks now — every step reads its own value,
        # as Detail Daemon, DAVE, MG/HiGS and HiFlow look SPEED's patched list up too.
        full = sigmas_src.detach().to("cpu", copy=True).flatten()[run.walk_offset:]
        if len(full) != len(run.walked):
            return None
        run.walked_src = sigmas_src
        run.walked = full
        if run.log:
            _log(f"{run.status.name} pass: sigmas re-published mid-run "
                 f"{[round(float(s), 4) for s in full]}")
    cur_sigma, scaled = lookup_sigma(args, run.pre_dd)
    if scaled and not run.pre_dd_written and run.params is not None:
        run.params[INFOTEXT_PRE_DD] = str(bool(run.pre_dd))
        run.pre_dd_written = True
    if run.debug_step is not None and not run.debug_done:
        run.step_index = _start_index(run.walked, cur_sigma)
        run.n_steps = len(run.walked) - 1
    return [sigma_to_value(cur_sigma, run.walked, schedule) for schedule, _, _, _ in run.built]


def _values_by_counter(run):
    """Upstream's Forge step counter, for samplers that publish no sigma list."""
    denoiser = run.denoiser
    try:
        from modules import shared

        sampling_step = shared.state.sampling_step
        total_sampling_steps = shared.state.sampling_steps
    except Exception:
        sampling_step = total_sampling_steps = 0
    current, actual = forge_step_position(
        sampling_step, total_sampling_steps, getattr(denoiser, "step", 0),
        getattr(denoiser, "total_steps", 0), getattr(denoiser, "steps", 0),
    )
    if current is None:
        return None
    run.step_index, run.n_steps = current, actual
    built = run.counter_built.get(actual)
    if built is None:
        built = build_entries(run.entries, actual)
        run.counter_built[actual] = built
    run.built = built
    return [float(schedule[current]) for schedule, _, _, _ in built]


def _same_tensor(a, b):
    return (a.data_ptr() == b.data_ptr() and a.shape == b.shape and a.stride() == b.stride()
            and a.dtype == b.dtype)


def _extra_sampler_substep(args):
    """Is this the extra evaluation of Euler (SMEA) Dy CFG++ at another resolution?

    Those samplers mark it in ``transformer_options`` (``sam3ext.extra_samplers.common``); Forge
    merges the sampler's options into the post-CFG ``model_options``."""
    options = args.get("model_options")
    transformer = options.get("transformer_options") if isinstance(options, dict) else None
    return isinstance(transformer, dict) and bool(transformer.get(SUBSTEP_MARKER))


def run_evaluation(run, args):
    """One post-CFG call: the graded ``denoised``, or the incoming object itself when nothing applies."""
    x0 = args["denoised"]
    status = run.status
    if _extra_sampler_substep(args):
        # One grade per sampler step, at the step's own evaluation. A Dy/SMEA sub-step evaluates the
        # same step again at half or ×1.25 resolution and its update is added to (Dy) or replaces
        # (SMEA) the step's; grading it too would give the pixels it moves that step's edit twice
        # (a 2×2-periodic pattern for Dy). Passed through, like the guidance stack keeps its step
        # state out of it.
        status.substeps += 1
        return x0
    status.evals += 1
    values = None
    sigmas_src = _published_sigmas(args) if run.publishes else None
    if sigmas_src is not None and sigmas_src is run.stale_sigmas:
        sigmas_src = None                   # left over from an earlier run, not this sampler's
    if sigmas_src is not None and torch.is_tensor(sigmas_src) and sigmas_src.numel() >= 2:
        values = _values_by_sigma(run, args, sigmas_src)
        if values is None:
            status.foreign += 1
            return x0
    elif run.denoiser is not None and not run.publishes:
        values = _values_by_counter(run)
    if values is None:
        if "no sigma list: passed through" not in status.notes:
            status.notes.append("no sigma list: passed through")
        return x0
    if (run.debug_step is not None and not run.debug_done and run.family is not None
            and run.step_index is not None
            and run.step_index == debug_panel.clamp_step(run.debug_step, run.n_steps)):
        # upstream captures x as this call received it, before any modifier's edit
        debug_panel.capture(run.state, x0.squeeze(2) if x0.dim() == 5 else x0, x0.dim() == 5, run.family)
        run.debug_done = True
        status.notes.append(f"debug latent captured at step {run.step_index}")
    if all(v == 0 for v in values):
        status.idle += 1
        return x0

    work = x0.float() if x0.dtype in _HALF else x0
    cur_basis = basis_on(run.family, work.device, work.dtype) if run.family else None

    def get_color_latent(color, device, dtype):
        key = (color_key(color), str(device), dtype)
        anchor = run.device_anchors.get(key)
        if anchor is None:
            source = run.anchors.get(color_key(color))
            if source is None:
                raise KeyError(f"no anchor encoded for colour {color}")
            anchor = source.to(device=device, dtype=dtype)
            run.device_anchors[key] = anchor
        return anchor

    out = apply_chain(run.built, work, values, run.family, cur_basis, get_color_latent, grid_downscale(run, work))
    if _same_tensor(out, work):
        status.idle += 1
        return x0
    if not bool(torch.isfinite(out).all()):
        status.nonfinite += 1
        return x0
    if out.dtype != x0.dtype:
        out = out.to(x0.dtype)
    status.edited += 1
    return out


def make_callback(run):
    def colorcraft_post_cfg(args):
        incoming = args["denoised"]
        try:
            return run_evaluation(run, args)
        except Exception as exc:  # never break a generation: pass the prediction through
            run.status.errors += 1
            run.status.last_error = type(exc).__name__
            if run.status.errors == 1:
                _log(f"{run.status.name} pass: {type(exc).__name__}: {exc} — input passed through")
            return incoming
        finally:
            if run.p is not None:
                write_status(run.p)

    setattr(colorcraft_post_cfg, OWNER_ATTR, OWNER)
    return colorcraft_post_cfg


# ---------------------------------------------------------------------------
# process_before_every_sampling
# ---------------------------------------------------------------------------


def _family(p, entries_need_basis, state, status):
    latent_format = getattr(getattr(getattr(p, "sd_model", None), "model_config", None), "latent_format", None)
    fmt = type(latent_format).__name__ if latent_format is not None else "none"
    family = basis_mod.family_for_latent_format(latent_format)
    if family is not None and load_family_basis(family) is None:
        status.notes.append(f"colorcraft-{family}.safetensors missing: vector controls off")
        if "missing" not in state["warned"]:
            state["warned"].add("missing")
            _log(f"vectors/colorcraft-{family}.safetensors is missing; vector controls are off this run.")
        family = None
    status.family = f"{family} ({fmt})" if family else f"no basis ({fmt})"
    if family is None and entries_need_basis:
        status.notes.append("vector controls off on this model: only Contrast and Color Shift apply")
        if "nobasis" not in state["warned"]:
            state["warned"].add("nobasis")
            _log(f"no Colorcraft basis for latent format {fmt} (SD 1.5/SDXL and others): exposure, "
                 "chroma, detail, Chroma Plus and masking are off; Contrast and Color Shift still work.")
    return latent_format, family


def _vector_free(entry):
    """Can this entry change anything without basis vectors? (contrast or colour shift)."""
    p = entry["params"]
    if "contrast" in p and p["contrast"] != 0:
        return True
    if entry["kind"] in ("basic", "shift"):
        return p.get("color_shift_amount", 0) != 0
    if entry["kind"] == "advanced":
        return bool(p.get("color_shift")) and p.get("color_shift_amount", 0) != 0
    return False


def process(p, args, full_hw=None):
    """``process_before_every_sampling``: attach this pass's callback, or do nothing at all.

    Off (disabled, no active tab for this pass, every tab at 0, a SAM3/ADetailer inner pass) clones
    nothing and writes no metadata. With the Debug panel's capture on, a pass with nothing to apply
    still attaches (capture only, on a model with a basis) so masks can be previewed before any edit.
    ``full_hw`` is the pass's latent grid (``latent_grid`` of Forge's ``x``), for ``grid_downscale``."""
    detach_owned(p)
    if inner_pass(p):
        return None
    xyz = getattr(p, spec.XYZ_ATTR, None)
    if not panel_state.args_enabled(args) and not (xyz and "enabled" in xyz):
        return None                     # off: nothing past the first argument is read
    try:
        config = panel_state.config_from_script_args(args)
    except panel_state.RefusedArgs as exc:
        _refuse(p, exc)
        return None
    config = spec.apply_xyz(config, xyz)
    if not config.enabled:
        return None
    is_hires = bool(getattr(p, "is_hr_pass", False))
    chain = spec.build_chain(config, is_hires)
    debug_on = bool(config.debug)
    if not chain.entries and not debug_on:
        return None

    state = _request_state(p)
    if not is_hires:
        state["passes"].clear()   # a new first pass (next n_iter batch) starts a new record
    params = _params(p)
    pass_name = "hires" if is_hires else "base"
    status = PassStatus(pass_name, tags=list(chain.tags))
    state["passes"][pass_name] = status
    if chain.entries:
        params[spec.INFOTEXT_KEY] = spec.to_infotext(config)
    if chain.unresolved_masks:
        status.notes.append("incomplete combo, unmasked: " + ", ".join(chain.unresolved_masks))

    latent_format, family = _family(p, chain.needs_basis, state, status)
    pairs = list(zip(chain.tags, chain.entries))
    if family is None:
        pairs = [(tag, entry) for tag, entry in pairs if _vector_free(entry)]
    entries = [entry for _, entry in pairs]
    capture_only = not entries and debug_on and family is not None
    if not entries and not capture_only:
        status.notes.append("nothing to apply")
        if debug_on and family is None:
            status.notes.append("debug capture needs a model with a basis")
        write_status(p)
        return None
    if capture_only:
        status.notes.append("debug capture only")

    objects = getattr(getattr(p, "sd_model", None), "forge_objects", None)
    anchors = {}
    colors = needed_colors(entries)
    if colors:
        try:
            anchors = encode_anchors(getattr(objects, "vae", None), latent_format, colors, state["anchors"])
        except Exception as exc:
            status.notes.append(f"anchors failed ({type(exc).__name__}): Contrast/Color Shift off")
            _log(f"colour anchors could not be encoded ({type(exc).__name__}: {exc}); "
                 "Contrast and Color Shift are off for this pass.")
            pairs = _without_anchor_edits(pairs)
            if family is None:
                pairs = []
            entries = [entry for _, entry in pairs]
            if not entries and not (debug_on and family is not None):
                status.notes.append("nothing to apply")
                write_status(p)
                return None

    original = getattr(objects, "unet", None)
    if original is None:
        status.notes.append("no UNet")
        write_status(p)
        return None
    sampler = getattr(p, "sampler", None)
    run = Run(
        entries=entries,
        tags=[tag for tag, _ in pairs],
        family=family,
        downscale=basis_mod.vae_downscale_factor(family),
        anchors=anchors,
        offset=forge_sampling_offset(p),
        publishes=sampler_publishes_sigmas(p),
        stale_sigmas=None,
        denoiser=getattr(sampler, "model_wrap_cfg", None),
        pre_dd=_option(OPT_PRE_DD, True),
        log=_option(OPT_LOG, False),
        status=status,
        p=p,
        params=params,
        debug_step=config.debug_step if debug_on else None,
        state=state,
        full_hw=tuple(full_hw) if full_hw is not None else None,
    )
    try:
        candidate = original.clone()
        transformer = (candidate.model_options or {}).get("transformer_options") or {}
        run.stale_sigmas = transformer.get("sampling_sigmas") if isinstance(transformer, dict) else None
        attach(candidate, make_callback(run))
        objects.unet = candidate
    except Exception as exc:
        status.notes.append(f"not attached ({type(exc).__name__})")
        _log(f"not attached: {type(exc).__name__}: {exc}")
        write_status(p)
        return None
    status.attached = True
    write_status(p)
    kinds = ", ".join(f"{tag} {spec.KIND_LABELS[e['kind']]}" for tag, e in pairs) or "debug capture only"
    _log(f"{pass_name} pass: {status.family}; {kinds}")
    return run


def _without_anchor_edits(pairs):
    """``(tag, entry)`` pairs with contrast and colour shift zeroed (no anchors); empty ones dropped."""
    kept = []
    for tag, entry in pairs:
        p = dict(entry["params"])
        if "contrast" in p:
            p["contrast"] = 0.0
        if "color_shift_amount" in p:
            p["color_shift_amount"] = 0.0
        if not spec.params_are_noop(entry["kind"], p):
            kept.append((tag, dict(entry, params=p)))
    return kept


__all__ = [
    "ANCHOR_CACHE_LIMIT", "INFOTEXT_PRE_DD", "OPT_LOG", "OPT_PRE_DD", "OWNER", "OWNER_ATTR", "attach",
    "clear_anchor_cache", "detach_owned", "encode_anchors", "grid_downscale", "inner_pass", "latent_grid",
    "make_callback", "process", "run_evaluation",
]

