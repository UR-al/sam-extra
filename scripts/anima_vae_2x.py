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

⚠️ EXPERIMENTAL — runtime iteration expected. The parts that are unit-verified
(tests/test_anima_vae_2x.py): state-dict detection, the pixel-shuffle +
downsample math, the 5D `[B,T,H,W,3]` output contract of the is_wan pipeline,
the stock-decode fallback and the load-device placement. The parts that may
need one tuning pass on a real checkpoint (watch the `[AnimaVAE2x]` logs):
  * the Wan-2.1 VAE architecture config used to build the decoder (if
    `load_state_dict` reports shape/key mismatches, the logged diff pins it),
  * latent normalization between the Qwen and Wan VAE spaces (if colors shift,
    enable the renorm toggle).
"""
from __future__ import annotations

import json
import os
import struct
import sys
import traceback

import gradio as gr

from modules import scripts

from sam3ext import layout_lanes

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
    ``device`` is the *load* device (``VAE.device``); the weights are built on
    the VAE offload device (CPU) and handed to Forge's ModelPatcher, which
    moves them to ``device`` for each decode (see ``_load_decoder``).
    Returns the ModelPatcher (or the bare eval() module if ModelPatcher is
    unavailable) or None on failure."""
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
        offload = _decoder_offload_device()
        model = model.to(device=offload, dtype=dtype).eval()
        model = _wrap_patcher(model, torch.device(device), offload)
        _log(f"built 12ch Wan decoder from {os.path.basename(path)} ✅ "
             f"(load_device={device} offload={offload} dtype={dtype})")
        # Keep only the most-recent decoder resident — evict others so trying
        # different 2x VAEs in a session doesn't accumulate decoders in VRAM.
        for old in [k for k in _DECODER_CACHE if k != key]:
            _DECODER_CACHE.pop(old, None)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
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
    Returns pixels in Forge's decode format, values in [0,1], with the SAME
    rank as the input: 5D → [B, T, H', W', 3] (what stock `VAE.decode` returns
    for an is_wan engine such as Anima — `decode_first_stage` then does
    `movedim(-1, 2)` and processing.py flattens the 5D batch), 4D → [B, H', W', 3].
    Dropping the T axis here would make the is_wan pipeline crash *outside*
    the wrapper (no fallback), so it is kept and the frames are folded into
    the batch dim for the 2D ops instead."""
    x = dec_out
    keep_t = x.ndim == 5
    if keep_t:
        b, c, t, h, w = x.shape
        x = x.movedim(2, 1).reshape(b * t, c, h, w)   # fold frames → [B·T,12,H,W]
    x = F.pixel_shuffle(x, 2)          # 12=3·2·2 → [B·T,3,2H,2W]
    if refine_1x:
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x = _gaussian_blur(x, blur_sigma)
    x = x.add(1.0).div(2.0).clamp(0.0, 1.0)   # [-1,1] → [0,1] (Forge process_output)
    x = x.movedim(1, -1)                      # → [B·T,H',W',3]
    if keep_t:
        x = x.reshape(b, t, *x.shape[1:])     # unfold → [B,T,H',W',3]
    return x.contiguous()


# ---------------------------------------------------------------------------
# Device placement — build on the offload device, let Forge load it for decode
# ---------------------------------------------------------------------------


def _decoder_load_device(orig_vae):
    """Forge's VAE keeps its *load* device in ``VAE.device`` (the GPU unless
    --cpu-vae); the weights themselves usually sit offloaded on CPU between
    decodes, so ``next(first_stage_model.parameters()).device`` is the wrong
    thing to build against (it would run the 3D Wan decode on CPU)."""
    dev = getattr(orig_vae, "device", None)
    if dev is not None:
        return torch.device(dev)
    try:
        from backend import memory_management
        return memory_management.get_torch_device()
    except Exception:
        return torch.device("cpu")


def _decoder_offload_device():
    try:
        from backend import memory_management
        return memory_management.vae_offload_device()
    except Exception:
        return torch.device("cpu")


def _wrap_patcher(model, load_device, offload_device):
    """Hand the decoder to Forge's memory management (same as the stock VAE:
    ``ModelPatcher`` + ``load_models_gpu`` in decode) so it is moved to the GPU
    only for the decode and evicted like any other model instead of sitting
    in VRAM for the rest of the process. Returns the bare module (placed on
    ``load_device``) if ModelPatcher is unavailable."""
    try:
        from backend.patcher.base import ModelPatcher
        return ModelPatcher(model, load_device=load_device, offload_device=offload_device)
    except Exception as e:
        _log(f"ModelPatcher unavailable ({type(e).__name__}: {e}) — decoder stays on {load_device}.")
        return model.to(device=load_device)


def _load_decoder(decoder, memory_required: float):
    """Make the decoder resident on its load device and return
    ``(module, device)``. A ModelPatcher goes through ``load_models_gpu``
    (frees other models if needed, registers it for eviction); a bare module
    is used where it already is."""
    if hasattr(decoder, "load_device") and hasattr(decoder, "model"):
        from backend import memory_management
        memory_management.load_models_gpu(
            [decoder], memory_required=memory_required, force_full_load=True
        )
        return decoder.model, torch.device(decoder.load_device)
    return decoder, next(decoder.parameters()).device


# --novram 등: load_models_gpu 가 NO_VRAM 이면 force_full_load 와 무관하게
# lowvram_model_memory=0.1 로 부분 로드한다. 수동 캐스트가 없는 모듈(Resample 의
# nn.Conv2d 등)은 CPU 에 남고 latent 는 load_device 로 가므로 디코드는 매번 장치
# 불일치로 실패해 순정 decode 로 폴백한다. 결과 경로는 그대로 두고 이유만 한 번 알린다.
_CPU_STRANDED_LOGGED = False


def _stranded_params(module, dev) -> list:
    """``dev`` 와 다른 장치에 남은 파라미터 이름 목록. Forge ops 모듈
    (``parameters_manual_cast``)은 그때그때 캐스트되므로 원인이 아니라 뺀다.
    진단용 — 무엇이든 실패하면 빈 목록."""
    if module is None or dev is None or torch is None:
        return []
    try:
        dev_type = torch.device(dev).type
        out = []
        for mname, m in module.named_modules():
            if getattr(m, "parameters_manual_cast", False):
                continue
            for pname, p in m.named_parameters(recurse=False):
                if p.device.type != dev_type:
                    out.append(f"{mname}.{pname}" if mname else pname)
        return out
    except Exception:
        return []


def _vram_state_name() -> str | None:
    try:
        from backend import memory_management
        return memory_management.vram_state.name
    except Exception:
        return None


def _log_decode_failure(e, decoder, dev) -> None:
    """2x 디코드 실패 로그. 디코더 가중치가 load_device 에 못 올라가 CPU 에 남은
    경우(--novram 등)는 설정이 바뀌지 않는 한 매번 같은 이유로 실패하므로 이유를
    담아 한 번만 알리고, 그 밖의 실패는 예전처럼 매번 알린다."""
    global _CPU_STRANDED_LOGGED
    stranded = _stranded_params(decoder, dev)
    if not stranded:
        _log(f"2x decode failed → stock decode: {type(e).__name__}: {e}")
        return
    if _CPU_STRANDED_LOGGED:
        return
    _CPU_STRANDED_LOGGED = True
    state = _vram_state_name()
    state_txt = f"VRAM State: {state}" if state else "VRAM State 알 수 없음"
    _log(
        f"2x decode 불가 → 순정 decode 로 폴백: 디코더 가중치 {len(stranded)}개"
        f"(예: {stranded[:3]})가 load_device={dev} 로 올라가지 않고 CPU 에 남았습니다 "
        f"({state_txt}). --novram(NO_VRAM) 에서는 Forge load_models_gpu 가 "
        f"force_full_load 를 무시하고 부분 로드해 수동 캐스트가 없는 모듈이 CPU 에 "
        f"남으므로, 이 실행 설정에서는 매번 순정 decode 로 폴백합니다 "
        f"(원래 오류: {type(e).__name__}: {e}). 이 안내는 한 번만 표시합니다."
    )


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
        decoder = dev = None   # 실패 진단(_log_decode_failure)용
        try:
            z = samples_in
            if self._renorm:
                z = (z - z.mean()) / (z.std() + 1e-6)
            was_4d = z.ndim == 4
            if was_4d:
                z = z.unsqueeze(2)            # [B,C,H,W] → [B,C,1,H,W]
            try:
                # Same estimate stock VAE.decode uses (is_wan formula on 5D).
                mem = float(self._orig.memory_used_decode(tuple(z.shape), self._orig.vae_dtype))
            except Exception:
                mem = 0.0
            decoder, dev = _load_decoder(self._decoder, mem)
            dt = next(decoder.parameters()).dtype
            z = z.to(device=dev, dtype=dt)
            with torch.no_grad():
                out = decoder.decode(z)       # [B,12,T,H,W]
            if was_4d:
                out = out[:, :, 0]            # back to the caller's 4D world
            px = _transform(out, self._refine_1x, float(self._blur_sigma))
            # Contract check: rank must match the latent (5D for is_wan, 4D
            # otherwise) and the channel axis must be last. Anything else
            # would blow up later in decode_first_stage/processing — outside
            # this try — so treat it as a failure and take the stock path.
            if px.ndim != samples_in.ndim or px.shape[-1] != 3:
                raise RuntimeError(
                    f"unexpected decode shape {tuple(px.shape)} for latent "
                    f"{tuple(samples_in.shape)}"
                )
            out_dev = getattr(self._orig, "output_device", None) or samples_in.device
            return px.to(device=out_dev, dtype=torch.float32)
        except Exception as e:
            _log_decode_failure(e, decoder, dev)
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


_VAE_FILE_CACHE: list[str] | None = None


def _list_vae_files() -> list[str]:
    # Cached: ui() runs once per tab at startup — avoid a second
    # refresh_vae_list() folder scan for the i2i tab.
    global _VAE_FILE_CACHE
    if _VAE_FILE_CACHE is not None:
        return _VAE_FILE_CACHE
    out = ["None"]
    try:
        from modules import sd_vae
        sd_vae.refresh_vae_list()
        out.extend(sorted(sd_vae.vae_dict.keys()))
    except Exception:
        pass
    _VAE_FILE_CACHE = out
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
    @property
    def section(self):
        # Forge 는 사용자 섹션을 설정값 칼럼(#txt2img_settings) 안에 만든다 → 1열 "ANIMA 튜닝" 자리.
        # 설정(sam3_layout_sections)을 끄거나 img2img 면 None 이라 예전과 똑같이 스크립트 컨테이너로 간다.
        return layout_lanes.anima_section(bool(getattr(self, "is_img2img", False)))

    sorting_priority = -26  # just under the guidance block, above the log toggles

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
                elem_classes=["sam3-on"],
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

        # Idempotency guard: if forge_objects.vae is already our wrapper (e.g. a
        # hires/second pass re-entered without Forge resetting forge_objects),
        # re-wrap the ORIGINAL stock VAE rather than the wrapper. Wrapping a
        # wrapper would apply the 2x pixel-shuffle twice and corrupt the decode.
        if isinstance(orig_vae, _VAE2xWrapper):
            orig_vae = getattr(orig_vae, "_orig", orig_vae)

        # Load device, not the weights' current device: Forge keeps the VAE
        # offloaded on CPU between decodes, so the latter would build (and
        # run) the 3D Wan decoder on the CPU.
        device = _decoder_load_device(orig_vae)
        try:
            dtype = getattr(orig_vae, "vae_dtype", None) or torch.bfloat16
        except Exception:
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
