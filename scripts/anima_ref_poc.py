"""Anima Reference-Latent PoC — shape logger (v0.9.7).

A SAFE, self-contained probe to verify whether we can inject a Cosmos/Anima
reference latent into Forge Neo's Anima diffusion forward from an EXTENSION,
without editing Forge core. It does NOT touch the SAM3 / Refine machinery.

How it works
------------
Forge Neo dispatches an optional ``model_options['model_function_wrapper']``
around ``model.apply_model`` at sampling time:

    backend/sampling/sampling_function.py:271
    output = model_options["model_function_wrapper"](
        model.apply_model,
        {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": ...},
    )

We attach such a wrapper to the Anima UNet in ``process_before_every_sampling``
(which runs AFTER processing.py resets forge_objects from the original each
batch, so the wrapper survives into sampling). The wrapper:

  1. Logs the real ``input`` tensor shape / ndim and the conditioning keys.
  2. (optional) Tries the ComfyUI Cosmos-Reference mechanism — temporal-axis
     concat (dim=2) of a reference latent, ``apply_model``, then slice the
     output back to the original T length — purely to see if it runs without
     blowing up and whether the output shape is preserved.

Everything is wrapped in try/except and ALWAYS falls back to the normal
``apply_model`` call on any error, so enabling this can never break a
generation — worst case it just logs and renders normally.

Run it: enable the checkbox, do a normal Anima **img2img** generate, then paste
the ``[AnimaRefPoC]`` lines from the webui console. Those shapes tell us
whether the real reference-edit feature is implementable on this path.
"""
from __future__ import annotations

import sys

import gradio as gr

from modules import scripts

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _log(msg: str) -> None:
    print(f"[AnimaRefPoC] {msg}", file=sys.stderr)


class AnimaRefPoC(scripts.Script):
    # Run late so other scripts have set up the unet first.
    sorting_priority = 99

    def title(self):
        return "Anima Reference PoC (shape logger)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Anima Reference-Latent PoC (debug / 안전)", open=False):
            gr.Markdown(
                "Cosmos/Anima reference-latent를 Forge에서 코어 수정 없이 주입 가능한지 "
                "확인하는 **안전한 계측**입니다. 켜고 **Anima로 img2img 한 번** 생성한 뒤 "
                "webui 콘솔의 `[AnimaRefPoC]` 줄을 복사해 주세요. 오류 시 항상 일반 "
                "생성으로 폴백하므로 결과물에는 영향이 없습니다."
            )
            enabled = gr.Checkbox(
                label="Enable PoC (log apply_model input shape)",
                value=False,
                elem_id="anima_ref_poc_enable",
            )
            do_concat = gr.Checkbox(
                label="Also try reference temporal-concat (init_latent을 T축 concat→apply_model→slice)",
                value=True,
                elem_id="anima_ref_poc_concat",
            )
        return [enabled, do_concat]

    def process_before_every_sampling(self, p, *args, **kwargs):
        if torch is None:
            return
        try:
            enabled = bool(args[0]) if len(args) > 0 else False
            do_concat = bool(args[1]) if len(args) > 1 else False
        except Exception:
            return
        if not enabled:
            return

        sd_model = getattr(p, "sd_model", None)
        forge_objects = getattr(sd_model, "forge_objects", None)
        unet = getattr(forge_objects, "unet", None)
        if unet is None:
            _log("no forge_objects.unet — cannot attach wrapper")
            return

        engine = type(sd_model).__name__ if sd_model is not None else "?"
        is_wan = bool(getattr(sd_model, "is_wan", False))
        _log(f"engine={engine}, is_wan={is_wan}, do_concat={do_concat}")
        if engine != "Anima":
            _log("engine is not 'Anima' — wrapper still attached for logging, "
                 "but reference-concat only makes sense on Anima/Cosmos.")

        # Reference latent = the img2img source's encoded latent. Forge sets
        # this on the processing object during init (img2img only).
        ref = getattr(p, "init_latent", None)
        if ref is None:
            _log("p.init_latent is None (txt2img?) — will log input shape only, "
                 "no concat. Use img2img for the full probe.")
        else:
            try:
                _log(f"ref(init_latent) shape={tuple(ref.shape)} ndim={ref.ndim} dtype={ref.dtype}")
            except Exception:
                pass

        logged = {"input": False}

        def _wrapper(apply_model, w_args):
            x = w_args.get("input")
            ts = w_args.get("timestep")
            c = w_args.get("c") or {}
            # Log the real input shape once (avoid spamming every step).
            if not logged["input"]:
                logged["input"] = True
                try:
                    ckeys = list(c.keys()) if hasattr(c, "keys") else type(c).__name__
                    _log(f"apply_model input.shape={tuple(x.shape)} ndim={x.ndim} "
                         f"dtype={x.dtype} c_keys={ckeys}")
                except Exception as e:
                    _log(f"input-log failed: {e}")

            # Optional: attempt the temporal-concat reference injection.
            if do_concat and ref is not None and hasattr(x, "ndim") and x.ndim == 5:
                try:
                    r = ref
                    if r.ndim == 4:
                        r = r.unsqueeze(2)  # [B,C,H,W] -> [B,C,1,H,W]
                    r = r.to(device=x.device, dtype=x.dtype)
                    # Match batch (CFG doubles the batch).
                    if r.shape[0] != x.shape[0]:
                        if r.shape[0] == 1:
                            r = r.repeat(x.shape[0], 1, 1, 1, 1)
                        else:
                            r = r[:1].repeat(x.shape[0], 1, 1, 1, 1)
                    # Spatial must match.
                    if r.shape[-2:] != x.shape[-2:]:
                        _log(f"ref H/W {tuple(r.shape[-2:])} != input H/W "
                             f"{tuple(x.shape[-2:])} — skipping concat (need same size)")
                        return apply_model(x, ts, **c)
                    t0 = x.shape[2]
                    xcat = torch.cat([x, r], dim=2)
                    if not logged.get("concat"):
                        logged["concat"] = True
                        _log(f"concat input {tuple(x.shape)} + ref T={r.shape[2]} "
                             f"-> {tuple(xcat.shape)}")
                    out = apply_model(xcat, ts, **c)
                    if not logged.get("out"):
                        logged["out"] = True
                        _log(f"apply_model(concat) out={tuple(out.shape)}")
                    out = out[:, :, :t0]
                    if not logged.get("sliced"):
                        logged["sliced"] = True
                        _log(f"sliced out={tuple(out.shape)} (expected T={t0}) — "
                             f"CONCAT PATH OK ✅")
                    return out
                except Exception as e:
                    if not logged.get("concat_fail"):
                        logged["concat_fail"] = True
                        _log(f"concat path FAILED: {type(e).__name__}: {e} — "
                             f"falling back to normal apply_model")
            # Default / fallback path.
            return apply_model(x, ts, **c)

        try:
            unet = unet.clone()
            unet.set_model_unet_function_wrapper(_wrapper)
            p.sd_model.forge_objects.unet = unet
            _log("wrapper attached to unet ✅ (run a sampling step to see shapes)")
        except Exception as e:
            _log(f"failed to attach wrapper: {type(e).__name__}: {e}")
