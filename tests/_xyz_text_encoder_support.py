"""Fixtures shared by the tests/test_xyz_text_encoder*.py files (not a test module).

* ``CASES`` — safetensors headers (key -> shape) for every branch of Forge's ``backend/loader.py``
  ``replace_state_dict`` and for the kinds of file found in the development PC's ``models/text_encoder`` (Qwen3.5,
  the Anima 3.8B adapter, a W/mean/std tap, a layers-50-63 tail, a two-slot 32B, t5gemma), with the slots
  ``sam3ext.xyz_text_encoder.te_slots`` must give (``RAISES`` where Forge itself fails);
* ``write_safetensors`` — a small but valid ``.safetensors`` file (zeros) for a header;
* ``header`` — a header dict from ``{key: shape}``.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

RAISES = "raises"

QWEN3_LAYER = "model.layers.0.post_attention_layernorm.weight"
Q_NORM = "model.layers.0.self_attn.q_norm.weight"


def header(shapes: dict, metadata: dict | None = None) -> dict:
    out = {key: {"dtype": "F32", "shape": list(shape), "data_offsets": [0, 0]} for key, shape in shapes.items()}
    if metadata is not None:
        out["__metadata__"] = dict(metadata)
    return out


def write_safetensors(path, shapes: dict, metadata: dict | None = None, *, fake_shapes: bool = False) -> Path:
    """A valid safetensors file whose tensors are zeros. ``fake_shapes``: write the shapes in the header but only one
    float of data per tensor (the axis reads headers only; keeps files with an 8B-wide shape small)."""
    path = Path(path)
    entries, offset = {}, 0
    for key, shape in shapes.items():
        count = 1
        for dim in shape:
            count *= dim
        size = 4 if fake_shapes else 4 * count
        entries[key] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    if metadata is not None:
        entries["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blob = json.dumps(entries).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        handle.write(b"\0" * offset)
    return path


# name -> (shapes, metadata, the slots Forge writes the file into | RAISES)
CASES = {
    # CLIP: the probe keys, shape[0] 768 (L) / 1280 (G)
    "clip_l_hf": ({"text_model.encoder.layers.0.layer_norm1.bias": [768]}, None, {"clip_l"}),
    "clip_l_sd1_prefixed": ({"cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.bias": [768]}, None,
                            {"clip_l"}),
    "clip_l_sdxl_prefixed": ({"conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.bias": [768]},
                             None, {"clip_l"}),
    "clip_l_flux_prefixed": ({"text_encoders.clip_l.transformer.text_model.encoder.layers.0.layer_norm1.bias": [768]},
                             None, {"clip_l"}),
    "clip_g_hf": ({"text_model.encoder.layers.0.layer_norm1.bias": [1280]}, None, {"clip_g"}),
    "clip_g_resblocks": ({"transformer.resblocks.0.ln_1.bias": [1280]}, None, {"clip_g"}),
    "clip_g_sdxl_resblocks": ({"conditioner.embedders.1.model.transformer.resblocks.0.ln_1.bias": [1280]}, None,
                              {"clip_g"}),
    "clip_l_resblocks": ({"transformer.resblocks.0.ln_1.bias": [768]}, None, {"clip_l"}),
    "clip_other_width": ({"text_model.encoder.layers.0.layer_norm1.bias": [1024]}, None, set()),
    "clip_l_and_g_in_one": ({"text_encoders.clip_l.transformer.text_model.encoder.layers.0.layer_norm1.bias": [768],
                             "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.bias": [1280]}, None,
                            {"clip_l", "clip_g"}),
    # T5 / UMT5
    "t5xxl": ({"encoder.block.0.layer.0.SelfAttention.k.weight": [4096, 4096], "shared.weight": [32128, 4096]}, None,
              {"t5xxl"}),
    "umt5xxl": ({"encoder.block.0.layer.0.SelfAttention.k.weight": [4096, 4096], "shared.weight": [256384, 4096]}, None,
                {"umt5xxl"}),
    "t5_without_shared": ({"encoder.block.0.layer.0.SelfAttention.k.weight": [4096, 4096]}, None, RAISES),
    "t5_qweight": ({"encoder.block.0.layer.0.SelfAttention.k.qweight": [4096, 512]}, None, {"t5xxl"}),
    # the elif chain
    "gemma2_2b": ({"model.layers.0.post_feedforward_layernorm.weight": [2304],
                   QWEN3_LAYER: [2304]}, None, {"gemma2_2b"}),
    "gemma2_with_q_norm": ({"model.layers.0.post_feedforward_layernorm.weight": [2304], Q_NORM: [256]}, None, RAISES),
    "qwen3vl_4b": ({"model.visual.deepstack_merger_list.0.norm.weight": [4096],
                    "model.visual.merger.linear_fc2.weight": [2560, 4096]}, None, {"qwen3vl_4b"}),
    "qwen3vl_8b": ({"model.visual.deepstack_merger_list.0.norm.weight": [4608],
                    "model.visual.merger.linear_fc2.weight": [4096, 4608]}, None, {"qwen3vl_8b"}),
    # qwen3-vl-4b-heretic: Qwen3-VL keys AND the Qwen3 keys — the elif takes Qwen3-VL first
    "qwen3vl_heretic_with_qwen3_keys": ({"model.visual.deepstack_merger_list.0.norm.weight": [4096],
                                         "model.visual.merger.linear_fc2.weight": [2560, 4096],
                                         QWEN3_LAYER: [2560], Q_NORM: [128]}, None, {"qwen3vl_4b"}),
    "qwen3vl_without_merger": ({"model.visual.deepstack_merger_list.0.norm.weight": [4096]}, None, RAISES),
    "qwen25_7b": ({"model.layers.0.self_attn.k_proj.bias": [512], QWEN3_LAYER: [3584]}, None, {"qwen25_7b"}),
    "qwen25_with_visual": ({"model.layers.0.self_attn.k_proj.bias": [512],
                            "visual.blocks.0.attn.proj.weight": [1280, 1280]}, None, {"qwen25_7b"}),
    "qwen25_wrong_bias": ({"model.layers.0.self_attn.k_proj.bias": [256]}, None, RAISES),
    "qwen3_06b": ({QWEN3_LAYER: [1024], Q_NORM: [128]}, None, {"qwen3_06b"}),
    "qwen3_4b": ({QWEN3_LAYER: [2560], Q_NORM: [128]}, None, {"qwen3_4b"}),
    "qwen3_8b": ({QWEN3_LAYER: [4096], Q_NORM: [128]}, None, {"qwen3_8b"}),
    "ministral3_3b": ({QWEN3_LAYER: [3072]}, None, {"ministral3_3b"}),
    "ministral3_wrong_width": ({QWEN3_LAYER: [2048]}, None, RAISES),
    "visual_only": ({"visual.blocks.0.attn.proj.weight": [1280, 1280]}, None, {"qwen25_7b"}),
    # the development PC's files (signatures only)
    "qwen32b_two_slots": ({QWEN3_LAYER: [5120], Q_NORM: [128], "visual.blocks.0.attn.proj.weight": [1152, 1152]}, None,
                          {"qwen3_8b", "qwen25_7b"}),
    "qwen32b_tail_layers_50_63": ({"model.layers.50.post_attention_layernorm.weight": [5120],
                                   "model.layers.50.self_attn.q_norm.weight": [128]}, None, set()),
    "tap_w_mean_std": ({"W": [4096, 4096], "mean": [4096], "std": [4096]}, None, set()),
    "t5gemma": ({"encoder.layers.0.self_attn.q_proj.weight": [768, 768], "encoder.embed_tokens.weight": [256000, 768]},
                None, set()),
    "qwen35_4b": ({"layers.0.linear_attn.in_proj_qkv.weight": [8192, 2560], "embed_tokens.weight": [248320, 2560]},
                  None, set()),
    "anima38_adapter": ({"blocks.0.cross_attn.q.weight": [2048, 2048]},
                        {"architecture": "anima_progressive_qwen35_cross_adapter_v1"}, set()),
    "vae_sd": ({"decoder.conv_in.weight": [512, 16, 3, 3]}, None, set()),
    "vae_wan": ({"decoder.middle.0.residual.0.gamma": [384]}, None, set()),
    "empty": ({}, None, set()),
}


class FakeOpts:
    """``shared.opts`` as far as the axis and Forge's settings code use it: values from ``data`` (else the default),
    ``set`` with Forge's semantics (``modules/options.py``: an equal value is not set), ``save`` recorded."""

    def __init__(self, data: dict, defaults: dict | None = None):
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "defaults", dict(defaults or {}))
        object.__setattr__(self, "saved", [])

    def __getattr__(self, key):
        data = object.__getattribute__(self, "data")
        if key in data:
            return data[key]
        defaults = object.__getattribute__(self, "defaults")
        if key in defaults:
            value = defaults[key]
            return list(value) if isinstance(value, list) else value
        raise AttributeError(key)

    def __setattr__(self, key, value):
        self.data[key] = value

    def set(self, key, value, is_api=False, run_callbacks=True):
        if self.data.get(key, None) == value:
            return False
        self.data[key] = value
        return True

    def get_default(self, key):
        return self.defaults.get(key)

    def save(self, *args, **kwargs):
        self.saved.append(args)


class CheckpointInfo:
    """``modules.sd_models.CheckpointInfo``: name, title, filename; ``repr`` is stable (it goes into the load hash)."""

    def __init__(self, name: str, shorthash: str):
        self.name = name
        self.title = f"{name} [{shorthash}]"
        self.filename = "C:/forge/models/Stable-diffusion/" + name
        self.ids = [name, self.title, shorthash]

    def __repr__(self):
        return f"<ckpt {self.name}>"


# Synthetic module files for the grid tests (real small safetensors in a temp models folder)
GRID_FILES = {
    ("VAE", "qwen_image_vae.safetensors"): CASES["vae_sd"][0],
    ("VAE", "qwen_image_vae_HDR_fp32.safetensors"): CASES["vae_sd"][0],
    ("text_encoder", "qwen_3_06b_base.safetensors"): CASES["qwen3_06b"][0],
    ("text_encoder", "qwen_3_06b_base_fp16_uncensored.safetensors"): CASES["qwen3_06b"][0],
    ("text_encoder", "qwen_3_06b_base_full_fp32.safetensors"): CASES["qwen3_06b"][0],
    ("text_encoder", "qwen3vl_4b_bf16.safetensors"): CASES["qwen3vl_4b"][0],
    ("text_encoder", "qwen35_4b.safetensors"): CASES["qwen35_4b"][0],
}


def write_models(models_path) -> tuple[dict, dict]:
    """Write ``GRID_FILES`` under ``models_path``; ``(module_list, module_map)`` the way Forge's ``refresh_models`` builds
    them (file name -> full path / path relative to its folder)."""
    module_list, module_map = {}, {}
    for (folder, name), shapes in GRID_FILES.items():
        path = write_safetensors(Path(models_path) / folder / name, shapes, fake_shapes=True)
        module_list[name] = str(path)
        module_map[name] = name
    return module_list, module_map
