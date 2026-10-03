"""Helpers for the Colorcraft shared-editor tests: random looks, stale states, Forge's API argument fill, the
node bridges (tests/js/colorcraft_bridge.mjs, colorcraft_session.mjs) and a by-repr Config comparison."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.colorcraft import panel_state as P  # noqa: E402
from sam3ext.colorcraft import spec  # noqa: E402

JS_TESTS = ROOT / "tests" / "js"
NODE = shutil.which("node")


def elem_id(tab):
    """Forge's ``Script.elem_id`` for this script's title ("Colorcraft (sam-extra)")."""
    return lambda item_id: f"script_{tab}_colorcraft_samextra_{item_id}"


def same_config(test, a, b, label=""):
    """Every value equal by type and repr (``-0.0`` apart from ``0.0``, ``True`` apart from ``1.0``)."""
    va, vb = spec.config_to_args(a), spec.config_to_args(b)
    for name, x, y in zip(spec.arg_names(), va, vb):
        test.assertTrue(type(x) is type(y) and repr(x) == repr(y), f"{label} {name}: {x!r} != {y!r}")


def random_value(rng, f, choices=None):
    if f.kind == spec.FLOAT:
        r = rng.random()
        if r < 0.35:
            return f.default
        if r < 0.8:
            steps = round((f.maximum - f.minimum) / f.step)
            return round(f.minimum + f.step * rng.randint(0, steps), 6)
        return rng.choice([7.5, -12.25, 0.30000000000000004, 1e-7, 123.456789])     # typed, unclamped
    if f.kind == spec.BOOL:
        return rng.random() < 0.4 if f.default is False else rng.random() < 0.7
    pool = list(choices if choices is not None else f.choices)
    return f.default if rng.random() < 0.4 else rng.choice(pool)


def random_config(rng):
    """A random look, coerced the way v0.31.0 reads its 579 positional values (the reference)."""
    config = spec.default_config()
    config.enabled = rng.random() < 0.8
    config.masking = rng.random() < 0.5
    for m in config.modifiers:
        dense = rng.random() < 0.4
        for f in spec.MODIFIER_FIELDS:
            if dense or rng.random() < 0.15:
                m[f.name] = random_value(rng, f)
    for leaf in config.leaves:
        if rng.random() < 0.4:
            for f in spec.LEAF_FIELDS:
                leaf[f.name] = random_value(rng, f)
    for ci, combo in enumerate(config.combos):
        if rng.random() < 0.4:
            for f in spec.COMBO_FIELDS:
                choices = spec.combo_ref_choices(ci) if f.name in ("mask_a", "mask_b") else None
                combo[f.name] = random_value(rng, f, choices)
    return spec.config_from_args(spec.config_to_args(config))


def random_configs(seed, n):
    rng = random.Random(seed)
    return [random_config(rng) for _ in range(n)]


def stale(config, mod, mask):
    """The state as it was before the editors' nodes were edited: both shown nodes at their defaults."""
    out = config.copy()
    for tag in (mod, mask):
        values = P.node_values(out, tag)
        for f in P.node_fields(tag):
            if not (tag == mod and f.name == "active"):
                values[f.name] = P.node_default(tag, f)
    return out


def stale_args(config, mod, mask, rev="r1"):
    """The 67 arguments with the edits only in the visible editors (the state does not have them yet)."""
    return ([config.enabled, config.masking, P.dump_state(stale(config, mod, mask), rev), config.debug,
             config.debug_step, P.make_ref(mod, mask, rev)] + P.editor_values(config, mod, mask))


def forge_api_fill(request_args):
    """modules/api/api.py ``init_script_args`` for one always-on script: copy min(ours, request) values over the
    defaults (``init_default_script_args``)."""
    script_args = P.default_args()
    for i in range(min(len(script_args), len(request_args))):
        script_args[i] = request_args[i]
    return script_args


def node_json(script, payload):
    out = subprocess.run([NODE, str(JS_TESTS / script)], input=json.dumps(payload), capture_output=True, text=True,
                         encoding="utf-8", check=True)
    return json.loads(out.stdout)


def bridge(jobs):
    return node_json("colorcraft_bridge.mjs", {"jobs": jobs})
