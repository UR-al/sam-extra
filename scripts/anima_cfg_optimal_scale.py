"""Anima Optimal Scale — the optimized-scale part of CFG-Zero* as an experimental post-CFG step.

Default OFF. Independent implementation of the optimized-scale equation of CFG-Zero*
(arXiv 2503.18886) for Anima (``prediction_type == "const"``), with a blend control and a
percent window; **zero-init (zeroing the first solver steps) is deliberately omitted** — its
meaning on short runs, img2img, hires and multi-evaluation samplers has to be checked first.
Anima image quality and GPU behaviour have not been validated.

Integrated from the 2026-10-02 review proposal
(docs/review_proposals_20261002/generation/anima_cfg_optimal_scale.py.proposal); the math and the
safety rules are the proposal's, the title, owner tag, infotext keys and element ids follow this
extension's conventions.

With Anima's ``x0 = x − σ·v``, the optimized-scale CFG
``v = s*·v_u + w·(v_c − s*·v_u)``, ``s* = ⟨v_c, v_u⟩/‖v_u‖²``, differs from plain CFG by
``(w − 1)·(s* − 1)·σ·v_u``, i.e. in x0 space by ``(w − 1)·(s* − 1)·r_u`` with the residuals
``r = x − x0`` (σ cancels in s*; the epsilon is scaled by σ² so it matches the v-space one without
dividing by a small σ). The correction is added to the incoming result times ``blend``.
"""
from __future__ import annotations

import math
import sys

import gradio as gr
import torch
from modules import scripts

from sam3ext import layout_lanes

OWNER = "sam-extra/anima-optimal-scale"
KEY = "Anima Optimal Scale"
STATUS_KEY = KEY + " status"
_SKIMMED_OWNER = "sam-extra/anima-skimmed-cfg"   # scripts/anima_skimmed_cfg.py _POST_CFG_OWNER


def _number(value, default):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _optimal_scale_residual(incoming, x, cond_x0, uncond_x0, scale, blend, sigma):
    """Return x0 with the optimized-scale residual added, keeping incoming corrections.

    For Anima's const prediction ``x0 = x − σ·v``, σ cancels in the optimal-scale quotient; the
    epsilon is scaled by σ² as well, so this equals the v-space epsilon without dividing by a
    small σ. With ordinary CFG, ``blend = 1`` is the optimized-scale formula; on top of other
    callbacks this is an experimental additive composition, not paper parity.
    """
    if blend <= 0.0:
        return incoming
    cond_residual = x.float() - cond_x0.float()
    uncond_residual = x.float() - uncond_x0.float()
    dimensions = tuple(range(1, x.ndim))
    product = (cond_residual * uncond_residual).sum(dim=dimensions, keepdim=True)
    norm = (uncond_residual * uncond_residual).sum(dim=dimensions, keepdim=True)
    sigma = sigma.reshape(-1, *([1] * (x.ndim - 1))).float()
    alpha = product / (norm + 1e-8 * sigma * sigma)
    correction = (scale - 1.0) * (alpha - 1.0) * uncond_residual * blend
    return incoming + correction.to(dtype=incoming.dtype)


def _attach(unet, callback):
    """Keep every foreign callback and remove only our own stale instance."""
    options = dict(getattr(unet, "model_options", {}) or {})
    previous = options.get("sampler_post_cfg_function", [])
    if not isinstance(previous, (list, tuple)):
        raise TypeError("sampler_post_cfg_function is not a callback list")
    callback._sam_extra_optimal_scale_owner = OWNER
    options["sampler_post_cfg_function"] = [
        fn for fn in previous
        if getattr(fn, "_sam_extra_optimal_scale_owner", None) != OWNER
    ] + [callback]
    unet.model_options = options


def _edit_strength(conditions):
    return sum(float(item.get("strength", 1.0)) for item in (conditions or []))


def _detach_owned(p):
    """OFF removes only this script's callback if a host reuses a patched UNet."""
    model = getattr(p, "sd_model", None)
    objects = getattr(model, "forge_objects", None)
    unet = getattr(objects, "unet", None)
    options = getattr(unet, "model_options", {}) or {}
    callbacks = options.get("sampler_post_cfg_function", [])
    if isinstance(callbacks, (list, tuple)) and any(
        getattr(fn, "_sam_extra_optimal_scale_owner", None) == OWNER for fn in callbacks
    ):
        candidate = unet.clone()
        copied = dict(candidate.model_options)
        copied["sampler_post_cfg_function"] = [
            fn for fn in callbacks
            if getattr(fn, "_sam_extra_optimal_scale_owner", None) != OWNER
        ]
        candidate.model_options = copied
        objects.unet = candidate
    params = getattr(p, "extra_generation_params", {}) or {}
    params.pop(KEY, None)
    params.pop(STATUS_KEY, None)


def _make_callback(params, blend, start, end):
    # One closure per sampling pass: no process-global active flag or tensor.
    counts = {"applied": 0, "skipped": 0}

    def update(reason=None):
        params[STATUS_KEY] = (
            f"applied={counts['applied']}; skipped={counts['skipped']}"
            + (f"; last_skip={reason}" if reason else "")
        )

    def skip(incoming, reason):
        counts["skipped"] += 1
        update(reason)
        return incoming

    def callback(args):
        incoming = args["denoised"]
        try:
            options = args.get("model_options") or {}
            if options.get("sampler_cfg_function") is not None:
                return skip(incoming, "custom CFG function")
            if any(
                getattr(fn, "_sam_extra_post_cfg_owner", None) == _SKIMMED_OWNER
                for fn in options.get("sampler_post_cfg_function", [])
            ):
                return skip(incoming, "Skimmed CFG modifies the prediction pair")
            cfg = float(args["cond_scale"])
            if not math.isfinite(cfg) or cfg <= 1.0 + 1e-6:
                return skip(incoming, "CFG <= 1 or skipped negative step")
            x = args.get("input")
            cond = args.get("cond_denoised")
            uncond = args.get("uncond_denoised")
            if not all(torch.is_tensor(t) for t in (incoming, x, cond, uncond)):
                return skip(incoming, "missing predictions")
            if x.ndim < 2 or not (x.shape == cond.shape == uncond.shape == incoming.shape):
                return skip(incoming, "prediction shape mismatch")
            if not args.get("uncond") or torch.equal(cond, uncond):
                return skip(incoming, "negative not evaluated or identical predictions")
            predictor = getattr(args.get("model"), "predictor", None)
            if getattr(predictor, "prediction_type", None) != "const":
                return skip(incoming, "non-const prediction")
            sigma = float(args["sigma"].reshape(-1)[0])
            if args["sigma"].reshape(-1).shape[0] not in (1, x.shape[0]):
                return skip(incoming, "sigma batch mismatch")
            sigma_start = float(predictor.percent_to_sigma(start))
            sigma_end = float(predictor.percent_to_sigma(end))
            if not math.isfinite(sigma) or sigma <= 0.0:
                return skip(incoming, "invalid sigma")
            if not sigma_end <= sigma <= sigma_start:
                return skip(incoming, "outside sigma window")
            strength = _edit_strength(args.get("cond"))
            if not math.isfinite(strength) or strength <= 0.0:
                return skip(incoming, "invalid edit strength")
            linear = uncond + (cond - uncond) * (cfg * strength)
            if not torch.allclose(incoming, linear, rtol=1e-5, atol=1e-6):
                return skip(incoming, "prior post-CFG correction (APG/PAG/DCW/etc.)")
            result = _optimal_scale_residual(
                incoming, x, cond, uncond, cfg * strength, blend, args["sigma"],
            )
            if not bool(torch.isfinite(result).all()):
                return skip(incoming, "non-finite result")
            counts["applied"] += 1
            update()
            return result
        except Exception as exc:
            # Preserve the incoming result on unsupported Forge contracts.
            return skip(incoming, type(exc).__name__)

    return callback


class AnimaOptimalScale(scripts.Script):
    sorting_priority = -25

    @property
    def section(self):
        return layout_lanes.anima_section(bool(getattr(self, "is_img2img", False)))

    def title(self):
        return "Anima Optimal Scale"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Anima Optimal Scale (실험 · CFG-Zero* optimized-scale)", open=False):
            gr.Markdown(
                "CFG-Zero*의 **optimized-scale 부분만** 구현한 실험 기능입니다 — 초기 스텝을 0으로 만드는 "
                "zero-init은 포함하지 않습니다. Anima·CFG > 1 전용, 기본 OFF. 추가 모델 호출 없이 CFG 보정을 "
                "더합니다. Skimmed CFG와 이미 결과를 바꾼 선행 APG/PAG/DCW 등의 post-CFG에서는 건너뛰고, "
                "그 이유를 infotext `Anima Optimal Scale status`에 남깁니다. 뒤에 실행되는 다른 보정과의 "
                "조합 화질과 Anima 화질 효과는 미검증입니다."
            )
            enabled = gr.Checkbox(
                label="Enable Anima Optimal Scale",
                value=False,
                elem_id="anima_optimal_scale_enable",
                elem_classes=["sam3-on", "sam3-on--optimal-scale"],
            )
            blend = gr.Slider(
                label="Optimal-scale blend",
                minimum=0, maximum=1, step=0.05, value=0.25,
                info="1이면 optimized-scale 식 그대로(표준 CFG일 때). 이 블렌드는 이 확장이 더한 실험 조절값입니다.",
                elem_id="anima_optimal_scale_blend",
            )
            start = gr.Slider(
                label="Optimal-scale start (%)",
                minimum=0, maximum=1, step=0.01, value=0.0,
                elem_id="anima_optimal_scale_start",
            )
            end = gr.Slider(
                label="Optimal-scale end (%)",
                minimum=0, maximum=1, step=0.01, value=1.0,
                elem_id="anima_optimal_scale_end",
            )

        # A single infotext key restores the enable state and the three values.
        def setting(params, key, default):
            fields = str(params.get(KEY, "")).split(";")
            values = dict(part.strip().split("=", 1) for part in fields if "=" in part)
            return _number(values.get(key), default)

        self.infotext_fields = [
            (enabled, lambda params: KEY in params),
            (blend, lambda params: setting(params, "blend", 0.25)),
            (start, lambda params: setting(params, "start", 0.0)),
            (end, lambda params: setting(params, "end", 1.0)),
        ]
        return [enabled, blend, start, end]

    def process_before_every_sampling(self, p, *args, **kwargs):
        _detach_owned(p)
        if not args or not bool(args[0]):
            return  # OFF: no new callback, metadata or math; clone only to remove an old own callback.
        blend = min(1.0, max(0.0, _number(args[1] if len(args) > 1 else 0.25, 0.25)))
        if blend <= 0.0:
            return  # Same no-op contract as OFF.
        start = _number(args[2] if len(args) > 2 else 0.0, 0.0)
        end = _number(args[3] if len(args) > 3 else 1.0, 1.0)
        if not 0.0 <= start < end <= 1.0:
            return
        model = getattr(p, "sd_model", None)
        if type(model).__name__ != "Anima":
            return
        original = getattr(getattr(model, "forge_objects", None), "unet", None)
        if original is None:
            return
        params = p.extra_generation_params
        try:
            candidate = original.clone()
            _attach(candidate, _make_callback(params, blend, start, end))
            model.forge_objects.unet = candidate
        except Exception as exc:
            print(f"[AnimaOptimalScale] not attached: {type(exc).__name__}", file=sys.stderr)
            return
        params[KEY] = f"blend={blend:g}; start={start:g}; end={end:g}; zero_init=omitted"
        params[STATUS_KEY] = "pending model evaluation"
