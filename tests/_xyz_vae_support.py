"""Fixtures shared by the tests/test_xyz_vae*.py files (not a test module).

* ``tiny_3d`` / ``tiny_2d`` / ``tiny_kl`` — tiny VAE layouts with the tensor names of the families Forge builds
  (``backend/nn/wan_vae.py`` ``WanVAE``: 3D causal convs and RMS ``gamma``; ``backend/nn/wan_vae_2d.py`` ``Qwen2DVAE``:
  2D convs, ``quant_conv`` / ``post_quant_conv`` and diffusers-style block names; the LDM / diffusers AutoencoderKL of
  SD / Flux) - a few tensors each instead of ~190, shapes in the same pattern;
* ``USER_VAES`` — the development PC's five VAE files (two 3D Wan, three 2D Qwen2D) as tiny files of their family, each
  filled with its own value so a test can tell whose weights a model holds;
* ``write_vae`` — a valid ``.safetensors`` file (F32 or BF16) whose tensors all hold one value (standard library only).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

Z = 4      # latent channels of the tiny VAEs
DIM = 8    # their width


def tiny_3d(z: int = Z, dim: int = DIM) -> dict:
    """A tiny 3D Wan / Qwen-Image VAE (the names of Forge's ``WanVAE``)."""
    return {
        "conv1.weight": [2 * z, 2 * z, 1, 1, 1], "conv1.bias": [2 * z],
        "conv2.weight": [z, z, 1, 1, 1], "conv2.bias": [z],
        "encoder.conv1.weight": [dim, 3, 3, 3, 3], "encoder.conv1.bias": [dim],
        "encoder.middle.0.residual.0.gamma": [dim, 1, 1, 1],
        "encoder.middle.0.residual.2.weight": [dim, dim, 3, 3, 3], "encoder.middle.0.residual.2.bias": [dim],
        "encoder.head.0.gamma": [dim, 1, 1, 1],
        "encoder.head.2.weight": [2 * z, dim, 3, 3, 3], "encoder.head.2.bias": [2 * z],
        "decoder.conv1.weight": [dim, z, 3, 3, 3], "decoder.conv1.bias": [dim],
        "decoder.middle.0.residual.0.gamma": [dim, 1, 1, 1],
        "decoder.middle.0.residual.2.weight": [dim, dim, 3, 3, 3], "decoder.middle.0.residual.2.bias": [dim],
        "decoder.head.0.gamma": [dim, 1, 1, 1],
        "decoder.head.2.weight": [3, dim, 3, 3, 3], "decoder.head.2.bias": [3],
    }


def tiny_2d(z: int = Z, dim: int = DIM) -> dict:
    """A tiny 2D Qwen2D VAE (the names of Forge's ``Qwen2DVAE``)."""
    return {
        "quant_conv.weight": [2 * z, 2 * z, 1, 1], "quant_conv.bias": [2 * z],
        "post_quant_conv.weight": [z, z, 1, 1], "post_quant_conv.bias": [z],
        "encoder.conv_in.weight": [dim, 3, 3, 3], "encoder.conv_in.bias": [dim],
        "encoder.mid_block.resnets.0.norm1.gamma": [dim, 1, 1],
        "encoder.mid_block.resnets.0.conv1.weight": [dim, dim, 3, 3], "encoder.mid_block.resnets.0.conv1.bias": [dim],
        "encoder.norm_out.gamma": [dim, 1, 1],
        "encoder.conv_out.weight": [2 * z, dim, 3, 3], "encoder.conv_out.bias": [2 * z],
        "decoder.conv_in.weight": [dim, z, 3, 3], "decoder.conv_in.bias": [dim],
        "decoder.mid_block.resnets.0.norm1.gamma": [dim, 1, 1],
        "decoder.mid_block.resnets.0.conv1.weight": [dim, dim, 3, 3], "decoder.mid_block.resnets.0.conv1.bias": [dim],
        "decoder.norm_out.gamma": [dim, 1, 1],
        "decoder.conv_out.weight": [3, dim, 3, 3], "decoder.conv_out.bias": [3],
    }


def tiny_kl(z: int = 4, dim: int = DIM, *, diffusers: bool = False) -> dict:
    """A tiny AutoencoderKL VAE (SD / SDXL / Flux) in the LDM layout Forge's model has, or the diffusers layout
    ``replace_state_dict`` converts."""
    block = "decoder.up_blocks.0.resnets.0" if diffusers else "decoder.up.0.block.0"
    return {
        "quant_conv.weight": [2 * z, 2 * z, 1, 1], "post_quant_conv.weight": [z, z, 1, 1],
        "encoder.conv_in.weight": [dim, 3, 3, 3], "decoder.conv_in.weight": [dim, z, 3, 3],
        f"{block}.norm1.weight": [dim], f"{block}.norm1.bias": [dim],
        "decoder.conv_out.weight": [3, dim, 3, 3],
    }


# The development PC's VAE files: name -> (family, dtype, the value every tensor holds)
USER_VAES = {
    "qwen_image_vae.safetensors": ("3d", "BF16", 1.0),
    "qwen_image_vae_HDR_fp32.safetensors": ("3d", "F32", 3.0),
    "qwen_image_vae_2dHDR_fp32.safetensors": ("2d", "F32", 2.0),
    "Qwen2D-Anime-dense_epoch_1.safetensors": ("2d", "F32", 4.0),
    "Qwen2D-Anime-dense-440k_ema_epoch_1.safetensors": ("2d", "BF16", 5.0),
}
# The order of the user's grid (Y values of xyz_grid-0000-791263211: 3D, 2D, 3D, 2D, 2D)
USER_ORDER = ("qwen_image_vae.safetensors", "qwen_image_vae_2dHDR_fp32.safetensors",
              "qwen_image_vae_HDR_fp32.safetensors", "Qwen2D-Anime-dense_epoch_1.safetensors",
              "Qwen2D-Anime-dense-440k_ema_epoch_1.safetensors")
FAMILY_SHAPES = {"3d": tiny_3d, "2d": tiny_2d}


def _bf16(value: float) -> bytes:
    return struct.pack("<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16)


def write_vae(path, shapes: dict, value: float = 0.0, dtype: str = "F32", metadata: dict | None = None) -> Path:
    """A valid safetensors file whose tensors all hold ``value`` (F32 or BF16 - ``value`` exactly representable)."""
    path = Path(path)
    one = struct.pack("<f", value) if dtype == "F32" else _bf16(value)
    entries, offset, chunks = {}, 0, []
    for key, shape in shapes.items():
        count = 1
        for dim in shape:
            count *= dim
        data = one * count
        entries[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        chunks.append(data)
    if metadata is not None:
        entries["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blob = json.dumps(entries).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for chunk in chunks:
            handle.write(chunk)
    return path


def write_user_vaes(folder) -> dict:
    """The five ``USER_VAES`` as tiny files of their family in ``folder``; ``{name: path}``."""
    out = {}
    for name, (family, dtype, value) in USER_VAES.items():
        out[name] = str(write_vae(Path(folder) / name, FAMILY_SHAPES[family](), value, dtype))
    return out
