"""A GGUF file's tensor table — names and shapes — read from its header only (stdlib; no tensor data, no numpy).

Forge Neo opens ``.gguf`` modules and checkpoints with its vendored gguf-py ``GGUFReader`` (``backend/utils.py``
``load_torch_file``) and gives each tensor the shape ``comfy.gguf.orig_shape.<name>`` (an INT32 array; ComfyUI-GGUF
writes it for tensors it reshaped) or else the header's dimensions reversed (GGUF stores them innermost first). This
reads the same two things the same way, so a GGUF file can be told apart by its tensor names and shapes like a
safetensors header: magic, version 2 or 3 (gguf-py refuses others), the key/value table (skipped except for the
``orig_shape`` arrays), then the tensor table. What Forge cannot open gives ``None``: a bad magic or version, a
duplicated tensor name, a key or name that is not UTF-8, a truncated table, and a file written for the other byte order
(gguf-py takes a version whose low 16 bits are 0 for one, and then calls ``ndarray.newbyteorder``, which NumPy 2 - the
one Forge Neo runs on - no longer has). A duplicated key is kept once, its first value (gguf-py warns and ``get_field``
returns the first). Tensor data and the ggml types are not checked.
"""
from __future__ import annotations

import os
import struct

MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)
ORIG_SHAPE_PREFIX = "comfy.gguf.orig_shape."

# GGUFValueType
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
_SCALAR_SIZES = {UINT8: 1, INT8: 1, UINT16: 2, INT16: 2, UINT32: 4, INT32: 4, FLOAT32: 4, BOOL: 1, UINT64: 8, INT64: 8,
                 FLOAT64: 8}
_MAX_DIMS = 16
_MAX_DEPTH = 8


class _Bad(Exception):
    pass


class _Reader:
    """Little-endian reads that never run past the end of the file."""

    def __init__(self, handle, size: int):
        self.handle, self.size = handle, size

    def take(self, count: int) -> bytes:
        if count < 0 or self.handle.tell() + count > self.size:
            raise _Bad("truncated")
        data = self.handle.read(count)
        if len(data) != count:
            raise _Bad("truncated")
        return data

    def skip(self, count: int) -> None:
        if count < 0 or self.handle.tell() + count > self.size:
            raise _Bad("truncated")
        self.handle.seek(count, os.SEEK_CUR)

    def unpack(self, code: str):
        return struct.unpack("<" + code, self.take(struct.calcsize(code)))[0]

    def string(self) -> str:
        length = self.unpack("Q")
        try:
            return self.take(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _Bad("not utf-8") from exc

    def skip_string(self) -> None:
        self.skip(self.unpack("Q"))

    def skip_value(self, value_type: int, depth: int = 0) -> None:
        if value_type in _SCALAR_SIZES:
            self.skip(_SCALAR_SIZES[value_type])
        elif value_type == STRING:
            self.skip_string()
        elif value_type == ARRAY:
            if depth > _MAX_DEPTH:
                raise _Bad("nested too deep")
            item_type = self.unpack("I")
            count = self.unpack("Q")
            if item_type in _SCALAR_SIZES:
                self.skip(count * _SCALAR_SIZES[item_type])
            else:
                if count > self.size:
                    raise _Bad("truncated")
                for _ in range(count):
                    self.skip_value(item_type, depth + 1)
        else:
            raise _Bad(f"unknown value type {value_type}")


def read_gguf_shapes(path) -> dict | None:
    """``{tensor name: shape}`` in file order (shapes as Forge's tensors have them), or ``None`` (not a GGUF file
    Forge could open)."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            reader = _Reader(handle, size)
            if reader.take(4) != MAGIC:
                return None
            (version,) = struct.unpack("<I", reader.take(4))
            if version & 0xFFFF == 0 or version not in SUPPORTED_VERSIONS:
                return None
            tensor_count = reader.unpack("Q")
            kv_count = reader.unpack("Q")
            # every entry takes at least 8 bytes (a string length): a count beyond the file is a broken file
            if tensor_count > size or kv_count > size:
                return None
            keys, orig_shapes = set(), {}
            for _ in range(kv_count):
                key = reader.string()
                first = key not in keys
                keys.add(key)
                value_type = reader.unpack("I")
                if value_type == ARRAY and key.startswith(ORIG_SHAPE_PREFIX):
                    item_type = reader.unpack("I")
                    count = reader.unpack("Q")
                    if item_type == INT32 and 0 < count <= _MAX_DIMS:
                        shape = tuple(reader.unpack("i") for _ in range(count))
                        if first:
                            orig_shapes[key[len(ORIG_SHAPE_PREFIX):]] = shape
                    elif item_type in _SCALAR_SIZES:
                        reader.skip(count * _SCALAR_SIZES[item_type])
                    else:
                        if count > size:
                            return None
                        for _ in range(count):
                            reader.skip_value(item_type, 1)
                else:
                    reader.skip_value(value_type)
            shapes = {}
            for _ in range(tensor_count):
                name = reader.string()
                if name in shapes:
                    return None
                n_dims = reader.unpack("I")
                if n_dims > _MAX_DIMS:
                    return None
                dims = [reader.unpack("Q") for _ in range(n_dims)]
                reader.skip(4 + 8)   # ggml type, data offset
                shapes[name] = orig_shapes.get(name) or tuple(reversed(dims))
            return shapes
    except (OSError, _Bad, struct.error, OverflowError, ValueError):
        return None
