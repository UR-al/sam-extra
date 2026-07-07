"""Anima VAE 2x — spacepxl 2x Wan-VAE decoder for Forge Neo (standalone).

A SELF-CONTAINED extension (independent of SAM3 and the guidance suite) that
uses spacepxl's 2x-upscale Wan2.1 VAE finetune as a *decoder* to reduce
speckle / clean up skin & hair on semi-realistic images. Touches no Forge core
file; on any error it falls back to Forge's normal decode, so enabling it can
never break a generation.

Why it works (verified against Forge Neo `neo` source)
------------------------------------------------------
- Forge Neo routes Qwen-Image VAE and Wan VAE through the SAME loader branch
  (`AutoencoderKLWan` / `AutoencoderKLQwenImage`), i.e. they share the latent
  structure → an Anima (Qwen) latent can be decoded by a Wan decoder.
- spacepxl's finetune only changes the decoder's final conv from 3→12 output
  channels; those 12 = 3·2·2 become a 3-channel 2x image via pixel shuffle:
  `F.pixel_shuffle(x, 2)`. Detection key = `decoder.head.2.weight` (shape[0]).
- Forge's `backend.nn.wan_vae.WanVAE(..., conv_out_channels=N, ...)` takes the
  decoder output channels as a constructor arg, and `WanVAE.decode(z)` accepts
  `[B, C, T, H, W]` (we feed a single-frame `T=1`).

Because Forge's loader HARDCODES conv_out_channels (it doesn't read it from the
state_dict), a 12-channel spacepxl VAE can't be loaded through the normal VAE
dropdown (shape mismatch). So this extension builds the decoder itself with
`conv_out_channels=12`, then swaps a thin decode-override wrapper into
`forge_objects.vae` for the generation.

⚠️ EXPERIMENTAL — runtime iteration expected. The parts that are unit-verified:
state-dict detection, the pixel-shuffle + downsample math. The parts that may
need one tuning pass on a real checkpoint (watch the `[AnimaVAE2x]` logs):
  * the Wan-2.1 VAE architecture config used to build the decoder (if
    `load_state_dict` reports shape/key mismatches, the logged diff pins it),
  * latent normalization between the Qwen and Wan VAE spaces (if colors shift,
    enable the renorm toggle),
  * the single-frame temporal axis handling.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import traceback

import gradio as gr

from modules import scripts

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore


def _log(msg: str) -> None:
    print(f"[AnimaVAE2x] {msg}", file=sys.stderr)


# Best-guess Wan-2.1 VAE architecture (used to build the 12ch decoder). If the
# real checkpoint mismatches, `load_state_dict(strict=False)` logs the diff and
# these can be corrected. conv_out_channels is overridden to the detected value.
_WAN21_VAE_CONFIG = dict(
    base_dim=96,
    z_dim=16,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    attn_scales=[],
    temporal_downsample=[False, True, True],
    dropout=0.0,
)

_DETECT_KEY_SUFFIX = "decoder.head.2.weight"  # spacepxl / Forge Decoder3d head

# Cache built decoders by absolute file path so we don't rebuild every gen.
_DECODER_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Detection — read the safetensors header only (no full tensor load)
# ---------------------------------------------------------------------------


def _read_safetensors_header(path: str) -> dict | None:
    try:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n).decode("utf-8"))
        return header
    except Exception as e:
        _log(f"header read failed for {os.path.basename(path)}: {type(e).__name__}: {e}")
        return None


def detect_output_channels(path: str) -> int | None:
    """Return the decoder head output-channel count (3 stock, 12 spacepxl), or
    None if the file has no recognizable Wan decoder head."""
    if not path or not os.path.isfile(path):
        return None
    header = _read_safetensors_header(path)
    if not header:
        return None
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        if key.endswith(_DETECT_KEY_SUFFIX):
            shape = meta.get("shape") if isinstance(meta, dict) else None
            if shape:
                return int(shape[0])
    return None


def is_spacepxl_2x(path: str) -> bool:
    return detect_output_channels(path) == 12


# ---------------------------------------------------------------------------
# Decoder construction (best-effort, cached, defensive)
# ---------------------------------------------------------------------------


def _strip_prefix(sd: dict) -> dict:
    """Drop a common leading prefix (e.g. 'vae.', 'first_stage_model.') so keys
    line up with the bare Wan VAE module names."""
    for prefix in ("first_stage_model.", "vae.", "model."):
        if any(k.startswith(prefix) for k in sd):
            keep = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            keep.update({k: v for k, v in sd.items() if not k.startswith(prefix)})
            return keep
    return sd


def _build_decoder(path: str, device, dtype):
    """Build a Wan VAE with conv_out_channels = detected (12), load weights.
    Returns the model in eval() on ``device``/``dtype`` or None on failure."""
    key = (os.path.abspath(path), str(device), str(dtype))
    if key in _DECODER_CACHE:
        return _DECODER_CACHE[key]

    out_ch = detect_output_channels(path)
    if out_ch not in (12,):
        _log(f"not a 12ch spacepxl VAE (out_channels={out_ch}) — skipping build.")
        _DECODER_CACHE[key] = None
        return None

    try:
        from safetensors.torch import load_file
    except Exception as e:
        _log(f"safetensors unavailable: {type(e).__name__}: {e}")
        _DECODER_CACHE[key] = None
        return None

    try:
        from backend.nn.wan_vae import WanVAE
    except Exception as e:
        _log(f"cannot import backend.nn.wan_vae.WanVAE: {type(e).__name__}: {e}")
        _DECODER_CACHE[key] = None
        return None

    try:
        cfg = dict(_WAN21_VAE_CONFIG)
        model = WanVAE(conv_out_channels=out_ch, **cfg)
        sd = _strip_prefix(load_file(path))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            _log(f"load_state_dict diff — missing={len(missing)} unexpected="
                 f"{len(unexpected)}. First missing: {list(missing)[:3]}. "
                 f"First unexpected: {list(unexpected)[:3]}. "
                 f"(If many, the _WAN21_VAE_CONFIG needs adjusting to match this "
                 f"checkpoint.)")
        model = model.to(device=device, dtype=dtype).eval()
        _log(f"built 12ch Wan decoder from {os.path.basename(path)} ✅")
        _DECODER_CACHE[key] = model
        return model
    except Exception as e:
        _log(f"decoder build failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        _DECODER_CACHE[key] = None
        return None


# ---------------------------------------------------------------------------
# The 12ch → 3ch 2x transform (unit-verifiable)
# ---------------------------------------------------------------------------


def _gaussian_blur(x, sigma: float):
    """Light separable Gaussian on [B,C,H,W]. Small fixed radius."""
    if sigma <= 0:
        return x
    radius = max(1, int(round(sigma * 2)))
    xs = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    k = (k / k.sum()).to(x.dtype)
    c = x.shape[1]
    kh = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    kw = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    x = F.conv2d(x, kh, padding=(radius, 0), groups=c)
    x = F.conv2d(x, kw, padding=(0, radius), groups=c)
    return x


def _transform(dec_out, refine_1x: bool, blur_sigma: float):
    """dec_out: raw decoder output [B, 12, T, H, W] (or [B,12,H,W]).
    Returns pixels in Forge's decode format: [B, H, W, 3], values in [0,1]."""
    x = dec_out
    if x.ndim == 5:
        x = x[:, :, 0]                 # single frame → [B,12,H,W]
    x = F.pixel_shuffle(x, 2)          # 12=3·2·2 → [B,3,2H,2W]
    if refine_1x:
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x = _gaussian_blur(x, blur_sigma)
    x = x.add(1.0).div(2.0).clamp(0.0, 1.0)   # [-1,1] → [0,1] (Forge process_output)
    return x.movedim(1, -1).contiguous()      # → [B,H,W,3]


# ---------------------------------------------------------------------------
# VAE wrapper — override decode, delegate everything else to the stock VAE
# ---------------------------------------------------------------------------


class _VAE2xWrapper:
    """Duck-types Forge's VAE object: overrides ``decode`` to use the 12ch
    spacepxl decoder + pixel shuffle, and delegates every other attribute /
    method (encode, device, dtype, ratios, …) to the original VAE."""

    def __init__(self, orig, decoder, refine_1x, blur_sigma, renorm):
        object.__setattr__(self, "_orig", orig)
        object.__setattr__(self, "_decoder", decoder)
        object.__setattr__(self, "_refine_1x", refine_1x)
        object.__setattr__(self, "_blur_sigma", blur_sigma)
        object.__setattr__(self, "_renorm", renorm)

    def decode(self, samples_in, *args, **kwargs):
        try:
            dev = next(self._decoder.parameters()).device
            dt = next(self._decoder.parameters()).dtype
            z = samples_in
            if self._renorm:
                z = (z - z.mean()) / (z.std() + 1e-6)
            if z.ndim == 4:
                z = z.unsqueeze(2)            # [B,C,H,W] → [B,C,1,H,W]
            z = z.to(device=dev, dtype=dt)
            with torch.no_grad():
                out = self._decoder.decode(z)
            px = _transform(out, self._refine_1x, float(self._blur_sigma))
            return px.to(samples_in.device)
        except Exception as e:
            _log(f"2x decode failed → stock decode: {type(e).__name__}: {e}")
            return self._orig.decode(samples_in, *args, **kwargs)

    def clone(self):
        return _VAE2xWrapper(
            self._orig.clone(), self._decoder, self._refine_1x,
            self._blur_sigma, self._renorm,
        )

    def __getattr__(self, name):
        # Anything we don't override → the real VAE.
        return getattr(object.__getattribute__(self, "_orig"), name)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _list_vae_files() -> list[str]:
    out = ["None"]
    try:
        from modules import sd_vae
        sd_vae.refresh_vae_list()
        out.extend(sorted(sd_vae.vae_dict.keys()))
    except Exception:
        pass
    return out


def _resolve_vae_path(name: str) -> str | None:
    if not name or name == "None":
        return None
    try:
        from modules import sd_vae
        return sd_vae.vae_dict.get(name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The extension script
# ---------------------------------------------------------------------------


class AnimaVAE2x(scripts.Script):
    sorting_priority = 1  # just under SAM3 / guidance block

    def title(self):
        return "Anima VAE 2x (spacepxl decoder)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Anima VAE 2x (spacepxl decoder)", open=False):
            gr.Markdown(
                "spacepxl **2x Wan-VAE 파인튜닝**을 디코더로 써서 speckle을 줄이고 "
                "skin/hair를 정리합니다. Qwen/Wan VAE는 latent 구조를 공유하므로 Anima "
                "생성에도 적용됩니다. **12채널 디코더(pixel-shuffle 2x)를 직접 빌드**해 "
                "decode만 대체하며, 오류 시 순정 decode로 폴백합니다.\n\n"
                "⚠️ 실험 기능 — 실제 spacepxl 체크포인트로 1회 검증 필요. 콘솔 "
                "`[AnimaVAE2x]` 로그 확인."
            )
            enabled = gr.Checkbox(
                label="Enable VAE 2x decode",
                value=False,
                elem_id="anima_vae2x_enable",
            )
            vae_file = gr.Dropdown(
                label="spacepxl 2x VAE (12ch decoder)",
                choices=_list_vae_files(),
                value="None",
                elem_id="anima_vae2x_file",
            )
            mode = gr.Radio(
                label="Output",
                choices=["1x refined (downsample)", "2x upscaled"],
                value="1x refined (downsample)",
                elem_id="anima_vae2x_mode",
            )
            with gr.Accordion("Advanced", open=False):
                blur_sigma = gr.Slider(
                    label="Refine blur sigma (1x 모드에서 downsample 후 약한 블러)",
                    minimum=0.0, maximum=2.0, step=0.05, value=0.5,
                    elem_id="anima_vae2x_blur",
                )
                renorm = gr.Checkbox(
                    label="Latent renorm (Qwen↔Wan 색 틀어지면 켜기)",
                    value=False,
                    elem_id="anima_vae2x_renorm",
                )
        return [enabled, vae_file, mode, blur_sigma, renorm]

    def process_before_every_sampling(self, p, *args, **kwargs):
        if torch is None:
            return
        try:
            enabled = bool(args[0]) if len(args) > 0 else False
            vae_name = str(args[1]) if len(args) > 1 else "None"
            mode = str(args[2]) if len(args) > 2 else "1x refined (downsample)"
            blur_sigma = float(args[3]) if len(args) > 3 else 0.5
            renorm = bool(args[4]) if len(args) > 4 else False
        except Exception as e:
            _log(f"bad args, disabling: {type(e).__name__}: {e}")
            return
        if not enabled:
            return

        path = _resolve_vae_path(vae_name)
        if not path:
            _log("no VAE selected — skipping.")
            return
        if not is_spacepxl_2x(path):
            _log(f"{vae_name} is not a 12ch spacepxl VAE — skipping (use a 2x "
                 "finetune whose decoder.head.2.weight has 12 out-channels).")
            return

        sd_model = getattr(p, "sd_model", None)
        forge_objects = getattr(sd_model, "forge_objects", None)
        orig_vae = getattr(forge_objects, "vae", None)
        if orig_vae is None:
            _log("no forge_objects.vae — cannot attach.")
            return

        try:
            device = next(orig_vae.first_stage_model.parameters()).device
            dtype = getattr(orig_vae, "vae_dtype", None) or torch.bfloat16
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16

        decoder = _build_decoder(path, device, dtype)
        if decoder is None:
            _log("decoder unavailable — leaving stock VAE.")
            return

        try:
            refine_1x = mode.startswith("1x")
            p.sd_model.forge_objects.vae = _VAE2xWrapper(
                orig_vae, decoder, refine_1x, blur_sigma, renorm
            )
            if not hasattr(p, "extra_generation_params"):
                p.extra_generation_params = {}
            p.extra_generation_params["Anima VAE 2x"] = (
                f"{vae_name}, {'1x-refined' if refine_1x else '2x'}, "
                f"blur={blur_sigma}, renorm={renorm}"
            )
            _log(f"attached ✅ vae={vae_name} mode={'1x' if refine_1x else '2x'} "
                 f"blur={blur_sigma} renorm={renorm}")
        except Exception as e:
            _log(f"failed to attach wrapper: {type(e).__name__}: {e}")
