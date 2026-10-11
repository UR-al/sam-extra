"""Fixtures shared by the tests/test_xyz_text_encoder*.py files (not a test module).

* ``CASES`` — safetensors headers (key -> shape) for every branch of Forge's ``backend/loader.py``
  ``replace_state_dict`` and for the kinds of file found in the development PC's ``models/text_encoder`` (Qwen3.5,
  the Anima 3.8B adapter, a W/mean/std tap, a layers-50-63 tail, a two-slot 32B, t5gemma), with the slots
  ``sam3ext.xyz_text_encoder.te_slots`` must give (``RAISES`` where Forge itself fails);
* ``write_safetensors`` — a small but valid ``.safetensors`` file (zeros) for a header;
* ``write_gguf`` — a GGUF file (version 2/3, either byte order, ``comfy.gguf.orig_shape`` arrays, other key/value
  types to skip), with zero tensor data or header only; ``GGUF_CASES`` — tensor tables of GGUF encoders as llama.cpp /
  city96 name them, with the slots Forge's ``gguf_remapping`` + ``replace_state_dict`` give them;
* ``CHECKPOINTS`` — checkpoint headers (SD1, SDXL, the SDXL refiner, Flux all-in-one / UNet-only, Anima) with the model
  type and the encoder slots they have built in;
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


# GGUF value types (gguf-py GGUFValueType)
GGUF_UINT32, GGUF_INT32, GGUF_FLOAT32, GGUF_STRING, GGUF_ARRAY, GGUF_UINT64 = 4, 5, 6, 8, 9, 10
_GGUF_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _gguf_value(order: str, value_type: int, value) -> bytes:
    if value_type == GGUF_STRING:
        data = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        return struct.pack(order + "Q", len(data)) + data
    if value_type == GGUF_ARRAY:
        item_type, items = value
        out = struct.pack(order + "IQ", item_type, len(items))
        return out + b"".join(_gguf_value(order, item_type, item) for item in items)
    return struct.pack(order + _GGUF_SCALAR[value_type], value)


def write_gguf(path, shapes: dict, *, orig_shapes: dict | None = None, kv=(), version: int = 3, order: str = "<",
               with_data: bool = False, magic: bytes = b"GGUF", raw_names: dict | None = None) -> Path:
    """A GGUF file whose tensors have the torch-order ``shapes`` (stored innermost first, as GGUF does) and F32 type.

    ``orig_shapes``: ``comfy.gguf.orig_shape.<name>`` INT32 arrays; ``kv``: more ``(key, type, value)`` entries (written
    first); ``with_data``: the aligned zero data section a reader that maps tensors needs (else header only);
    ``raw_names``: tensor name -> the bytes written for it (non-UTF-8 names)."""
    path = Path(path)
    entries = [(key, value_type, value) for key, value_type, value in kv]
    for name, shape in (orig_shapes or {}).items():
        entries.append((f"comfy.gguf.orig_shape.{name}", GGUF_ARRAY, (GGUF_INT32, list(shape))))
    out = bytearray(magic)
    out += struct.pack(order + "I", version)
    out += struct.pack(order + "QQ", len(shapes), len(entries))
    for key, value_type, value in entries:
        out += _gguf_value(order, GGUF_STRING, key) + struct.pack(order + "I", value_type)
        out += _gguf_value(order, value_type, value)
    offset = 0
    sizes = []
    for name, shape in shapes.items():
        dims = list(reversed(shape))
        name_bytes = (raw_names or {}).get(name, name.encode("utf-8"))
        out += _gguf_value(order, GGUF_STRING, name_bytes)
        out += struct.pack(order + "I", len(dims)) + b"".join(struct.pack(order + "Q", dim) for dim in dims)
        out += struct.pack(order + "IQ", 0, offset)
        count = 1
        for dim in dims:
            count *= dim
        size = 4 * count
        size += -size % 32
        sizes.append(size)
        offset += size
    if with_data:
        out += b"\0" * (-len(out) % 32)
        out += b"\0" * offset
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


# GGUF encoders: name -> (tensor shapes in torch order, the slots Forge writes the file into). Names as llama.cpp /
# city96's T5 conversion writes them; Forge's gguf_remapping turns them into the Hugging Face names te_slots reads.
GGUF_CASES = {
    "t5xxl_q8": ({"enc.blk.0.attn_k.weight": [4096, 4096], "token_embd.weight": [32128, 4096],
                  "enc.blk.0.attn_norm.weight": [4096], "enc.output_norm.weight": [4096]}, {"t5xxl"}),
    "umt5xxl_q8": ({"enc.blk.0.attn_k.weight": [4096, 4096], "token_embd.weight": [256384, 4096]}, {"umt5xxl"}),
    "qwen3_06b_q8": ({"blk.0.attn_norm.weight": [1024], "blk.0.attn_q_norm.weight": [128],
                      "blk.0.ffn_norm.weight": [1024], "blk.0.attn_q.weight": [2048, 1024],
                      "token_embd.weight": [151936, 1024]}, {"qwen3_06b"}),
    "qwen3_4b_q4": ({"blk.0.attn_norm.weight": [2560], "blk.0.attn_q_norm.weight": [128],
                     "blk.0.ffn_norm.weight": [2560], "token_embd.weight": [151936, 2560]}, {"qwen3_4b"}),
    "qwen25_7b_q4": ({"blk.0.attn_norm.weight": [3584], "blk.0.attn_k.bias": [512], "blk.0.ffn_norm.weight": [3584],
                      "token_embd.weight": [152064, 3584]}, {"qwen25_7b"}),
    "ministral3_q4": ({"blk.0.attn_norm.weight": [3072], "blk.0.ffn_norm.weight": [3072],
                       "token_embd.weight": [131072, 3072]}, {"ministral3_3b"}),
    "qwen25_vision_mmproj": ({"mm.0.weight": [5120, 5120], "v.blk.0.attn_out.weight": [1280, 1280],
                              "v.patch_embd.weight": [1280, 3, 14, 14]}, {"qwen25_7b"}),
    "llm_without_ffn_norm": ({"blk.0.attn_norm.weight": [1024], "token_embd.weight": [151936, 1024]}, set()),
    "flux_unet_q8": ({"double_blocks.0.img_attn.norm.key_norm.scale": [128]}, set()),
}


# Checkpoint headers: name -> (shapes, replace_state_dict's model type, the encoder slots built in)
SD1_LEGACY = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"
CHECKPOINTS = {
    "sd15": ({SD1_LEGACY: [320, 768],
              "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.bias": [768]}, "sd1", {"clip_l"}),
    "sdxl": ({SD1_LEGACY: [640, 2048],
              "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.bias": [768],
              "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.bias": [1280]}, "sdxl", {"clip_l", "clip_g"}),
    "sdxl_without_encoders": ({SD1_LEGACY: [640, 2048]}, "sdxl", set()),
    "sdxl_refiner": ({SD1_LEGACY: [768, 1280],
                      "conditioner.embedders.0.model.transformer.resblocks.0.ln_1.bias": [1280]}, "xlrf", {"clip_g"}),
    "flux_all_in_one": ({"model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale": [128],
                         "text_encoders.clip_l.transformer.text_model.encoder.layers.0.layer_norm1.bias": [768],
                         "text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.k.weight": [4096, 4096],
                         "vae.decoder.conv_in.weight": [512, 16, 3, 3]}, "flux", {"clip_l", "t5xxl"}),
    "flux_unet_only": ({"double_blocks.0.img_attn.norm.key_norm.scale": [128]}, "flux", set()),
    "anima_unet_only": ({"net.blocks.0.self_attn.q_proj.weight": [2048, 2048]}, "-", set()),
    "zimage_all_in_one": ({"model.diffusion_model.layers.0.attention.qkv.weight": [11520, 3840],
                           "text_encoders.qwen3_4b.transformer.model.layers.0.post_attention_layernorm.weight": [2560]},
                          "-", {"qwen3_4b"}),
}


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
    ``set`` with Forge's semantics (``modules/options.py``: an equal value is not set), ``save`` recorded - its
    arguments in ``saved``, what ``json.dump(self.data)`` would write in ``written``."""

    def __init__(self, data: dict, defaults: dict | None = None):
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "defaults", dict(defaults or {}))
        object.__setattr__(self, "saved", [])
        object.__setattr__(self, "written", [])

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
        self.written.append(json.loads(json.dumps(self.data)))


class CheckpointInfo:
    """``modules.sd_models.CheckpointInfo``: name, title, filename; ``repr`` is stable (it goes into the load hash)."""

    def __init__(self, name: str, shorthash: str, filename: str | None = None):
        self.name = name
        self.title = f"{name} [{shorthash}]"
        self.filename = filename or "C:/forge/models/Stable-diffusion/" + name
        self.ids = [name, self.title, shorthash]

    def __repr__(self):
        return f"<ckpt {self.name}>"


# Synthetic module files for the grid tests (real small safetensors / GGUF headers in a temp models folder)
GRID_FILES = {
    ("VAE", "qwen_image_vae.safetensors"): CASES["vae_sd"][0],
    ("VAE", "qwen_image_vae_HDR_fp32.safetensors"): CASES["vae_sd"][0],
    ("text_encoder", "qwen_3_06b_base.safetensors"): CASES["qwen3_06b"][0],
    ("text_encoder", "qwen_3_06b_base_fp16_uncensored.safetensors"): CASES["qwen3_06b"][0],
    ("text_encoder", "qwen_3_06b_base_full_fp32.safetensors"): CASES["qwen3_06b"][0],
    ("text_encoder", "qwen3vl_4b_bf16.safetensors"): CASES["qwen3vl_4b"][0],
    ("text_encoder", "qwen35_4b.safetensors"): CASES["qwen35_4b"][0],
    # Flux-style files
    ("VAE", "ae.safetensors"): CASES["vae_sd"][0],
    ("text_encoder", "clip_l.safetensors"): CASES["clip_l_hf"][0],
    ("text_encoder", "clip_l_other.safetensors"): CASES["clip_l_hf"][0],
    ("text_encoder", "t5xxl_fp16.safetensors"): CASES["t5xxl"][0],
    ("text_encoder", "t5xxl_fp8.safetensors"): CASES["t5xxl"][0],
    ("text_encoder", "t5-v1_1-xxl-encoder-Q8_0.gguf"): GGUF_CASES["t5xxl_q8"][0],
}


def write_models(models_path) -> tuple[dict, dict]:
    """Write ``GRID_FILES`` under ``models_path``; ``(module_list, module_map)`` the way Forge's ``refresh_models`` builds
    them (file name -> full path / path relative to its folder)."""
    module_list, module_map = {}, {}
    for (folder, name), shapes in GRID_FILES.items():
        target = Path(models_path) / folder / name
        if name.endswith(".gguf"):
            path = write_gguf(target, shapes)
        else:
            path = write_safetensors(target, shapes, fake_shapes=True)
        module_list[name] = str(path)
        module_map[name] = name
    return module_list, module_map


def write_checkpoint(folder, case: str, name: str | None = None) -> Path:
    """A checkpoint file (header + one float per tensor) for ``CHECKPOINTS[case]``."""
    return write_safetensors(Path(folder) / (name or f"{case}.safetensors"), CHECKPOINTS[case][0], fake_shapes=True)
