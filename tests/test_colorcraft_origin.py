"""Colorcraft — origin parity with muerrilla/ComfyUI-Colorcraft@d28ac6a (the latest math) and the
aoleg fork @f00066c (Flux2 family).

The oracle is upstream's own ComfyUI node package, loaded verbatim from tests/_origin_colorcraft (MIT,
SHA-256 pinned below) behind a minimal fake ``comfy``: its ``ColorcraftSampler.wrap`` is driven far
enough to hand back its post-CFG function, fed the chain its node classes build. The Forge side runs
this extension's real attach path (``hook.process``) and its post-CFG callback with Forge's real latent
format classes (Wan21 / Flux / Flux2 from the Forge checkout) and a deterministic colour-dependent VAE.

What must hold:

* the pinned copies are unchanged;
* every library function this port claims to copy unchanged has the same syntax tree as upstream's,
  and the modifier loop is upstream's statement for statement after exactly the documented
  substitutions (anchors pre-encoded, per-family downscale, precomputed schedule values);
* through the Forge hook, every evaluation is **bit-identical** to the node on all three families
  (krea2 on 5-D Wan21 latents, zimage on Flux, flux2 on 128-channel Flux2) over 20 scenarios, at
  schedule sigmas and between them (second-order sampler midpoints) — and the latent actually moves;
* the panel's ranges and defaults are upstream's (Forge panel where it has the control, the node
  otherwise), with the documented host differences only.

For flux2 the oracle's family tables are extended exactly the way the fork extends them (its
``MODEL_DEV_DEFAULTS['flux2']`` row + upstream's ``detail_scale`` 4.0, 16x downscale).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import sys
import textwrap
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _support():
    spec_ = importlib.util.spec_from_file_location("_colorcraft_support", ROOT / "tests" / "_colorcraft_support.py")
    module = sys.modules.get(spec_.name)
    if module is None:
        module = importlib.util.module_from_spec(spec_)
        sys.modules[spec_.name] = module
        spec_.loader.exec_module(module)
    return module


S = _support()

from sam3ext.colorcraft import basis, color, engine, masking, schedule, spec, vectors  # noqa: E402

# SHA-256 of each pinned file's upstream text (after the header's marker line; LF line endings — the
# upstream files are LF already). Upstream: muerrilla/ComfyUI-Colorcraft@d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf,
# fork: aoleg/ComfyUI-Colorcraft@f00066c63c9d8f96abc119cada3b51b36688fcac. Git blob ids at those commits
# (``git hash-object``) were checked when the copies were made; the LICENSE files are byte copies.
ORIGIN_SHA256 = {
    "_origin_colorcraft/nodes.py": "2d1566f2fafdf1d27073feed8c5fd95e55c8fc00c3ce0cb0a9235d5303a88b70",
    "_origin_colorcraft/lib_colorcraft/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "_origin_colorcraft/lib_colorcraft/basis.py": "644642e9989ad0de5d7d2f7fac24addbe1634f33a352ba74433f1b88340cc284",
    "_origin_colorcraft/lib_colorcraft/color.py": "b7df301fcadc685ad0a6a78ee3a2a4c657e6e33fcf9639ef7a776b238307f37f",
    "_origin_colorcraft/lib_colorcraft/debug.py": "ee28d359d6d5a46df5e088b4475f24cd1bfea2349b13f8606f3402cbb8ba7fa7",
    "_origin_colorcraft/lib_colorcraft/masking.py": "ce584f9b0a6db5ab8e42560afbc8c1a092cb268553045bc345b36f318b9f2df8",
    "_origin_colorcraft/lib_colorcraft/schedule.py": "bc1b8b98079e861d50b7a1f4ad045fbd216523e9ee0ca3fb4b624f88c64c2534",
    "_origin_colorcraft/lib_colorcraft/vectors.py": "9f8a1dce3866a2c57cd912d89606d38a6e972b6d8a39009b18e0fd42f098666d",
    "_origin_colorcraft/scripts/colorcraft.py": "056adf79867285cb706000f03cb18c2ba46de4ce65b958d8b0b4c21676f2cec6",
    "_origin_colorcraft_fork/lib_colorcraft/core.py": "d578e3995f84fc8bef349eae319cf256bd4d4b7396d4b93b4da0b4a59abd0278",
    "_origin_colorcraft_fork/lib_colorcraft/params.py": "1b2b4df698e4abaa0d9e2bfebcb74bb4a43eed98d6207c534456f935bd7a334b",
    "_origin_colorcraft_fork/lib_colorcraft/spec.py": "28ce4a4294fb431f43f1b5d0ad712a83d6ae6860b06f5e0c2d5ff4dcf0818287",
}
LICENSE_SHA256 = {
    "_origin_colorcraft/LICENSE": "b1a3a92dd9b78c9e99969106bac1ba7d84fbf337b489352d74e322402f77160c",
    "_origin_colorcraft_fork/LICENSE": "d05d785c0378e5dd55a01ac7395049e08b6f9bd2762b113d749bdff41681f47b",
}


def _function_node(module_or_source, name):
    source = module_or_source if isinstance(module_or_source, str) else inspect.getsource(module_or_source)
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _dump(node):
    return ast.dump(node, include_attributes=False)


class PinnedCopyTests(unittest.TestCase):
    def test_vendored_upstream_files_are_unchanged(self):
        for rel, digest in ORIGIN_SHA256.items():
            with self.subTest(file=rel):
                body = S.origin_body(ROOT / "tests" / rel)
                self.assertEqual(hashlib.sha256(body.encode("utf-8")).hexdigest(), digest)

    def test_license_texts_ship_next_to_the_copies(self):
        for rel, digest in LICENSE_SHA256.items():
            with self.subTest(file=rel):
                text = (ROOT / "tests" / rel).read_text(encoding="utf-8")
                self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(), digest)
                self.assertIn("MIT License", text)
                self.assertIn("(Muerrilla)", text)

    def test_headers_name_the_pinned_commits(self):
        for rel in ORIGIN_SHA256:
            commit = S.FORK_COMMIT if rel.startswith("_origin_colorcraft_fork") else S.UPSTREAM_COMMIT
            with self.subTest(file=rel):
                head = (ROOT / "tests" / rel).read_text(encoding="utf-8").split("# ---- upstream", 1)[0]
                self.assertIn(commit, head)
                self.assertIn("MIT License", head)


class UnchangedFunctionTests(unittest.TestCase):
    """The functions the port says it copied unchanged have upstream's syntax tree."""

    @classmethod
    def setUpClass(cls):
        cls.origin = S.load_origin()
        cls.fork = S.load_fork()

    def _same(self, ours_module, theirs_module, name):
        self.assertEqual(_dump(_function_node(ours_module, name)), _dump(_function_node(theirs_module, name)), name)

    def test_vectors_masking_schedule(self):
        for ours, theirs in ((vectors, self.origin.vectors), (masking, self.origin.masking),
                             (schedule, self.origin.schedule)):
            names = [n.name for n in ast.parse(inspect.getsource(theirs)).body if isinstance(n, ast.FunctionDef)]
            self.assertTrue(names)
            for name in names:
                with self.subTest(module=ours.__name__, function=name):
                    self._same(ours, theirs, name)

    def test_upstream_module_constants(self):
        self.assertEqual(masking.MASK_AXIS_OPTIONS, self.origin.masking.MASK_AXIS_OPTIONS)
        self.assertIn("clarity", masking.MASK_AXIS_OPTIONS)
        self.assertIn("sharpness", masking.MASK_AXIS_OPTIONS)
        self.assertEqual(masking.HARDNESS_GAIN, self.origin.masking.HARDNESS_GAIN)

    def test_color_functions(self):
        for name in ("_color_adjust", "_color_adjust_legacy", "apply_contrast", "apply_color_shift",
                     "build_color_latent"):
            with self.subTest(function=name):
                self._same(color, self.origin.color, name)
        # to_model_space is the fork's signature (rank passed in), the arithmetic upstream's
        self._same(color, self.fork.core, "to_model_space")

    def test_sigma_to_value_is_the_nodes(self):
        self._same(schedule, self.origin.nodes, "sigma_to_value")

    def test_basis_functions_and_tables(self):
        for name in ("load_basis", "resolve_dev"):
            with self.subTest(function=name):
                self._same(basis, self.origin.basis, name)
        self._same(basis, self.fork.core, "family_for_latent_format")
        for family in ("krea2", "zimage"):
            self.assertEqual(basis.MODEL_DEV_DEFAULTS[family], self.origin.basis.MODEL_DEV_DEFAULTS[family])
        self.assertEqual(basis.MODEL_DEV_DEFAULTS["flux2"],
                         dict(self.fork.core.MODEL_DEV_DEFAULTS["flux2"], detail_scale=4.0))
        self.assertEqual(basis.LATENT_FORMAT_TO_FAMILY, self.fork.core.LATENT_FORMAT_TO_FAMILY)
        for key, value in self.origin.basis.LATENT_FORMAT_TO_FAMILY.items():
            self.assertEqual(basis.LATENT_FORMAT_TO_FAMILY[key], value)
        self.assertEqual(basis.VAE_DOWNSCALE_FACTORS, self.fork.core.VAE_DOWNSCALE_FACTORS)
        self.assertEqual(basis.VAE_DOWNSCALE_FACTOR, self.origin.basis.VAE_DOWNSCALE_FACTOR)
        self.assertEqual(basis.BASIS_FAMILIES, self.fork.core.BASIS_FAMILIES)


class _Normalise(ast.NodeTransformer):
    """Upstream's post-CFG loop with this port's three documented substitutions applied."""

    def visit_Assign(self, node):
        self.generic_visit(node)
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "sigma_to_value":
            node.value = ast.Subscript(value=ast.Name("values", ast.Load()), slice=ast.Name("i", ast.Load()),
                                       ctx=ast.Load())
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "get_color_latent":
            node.args = [a for a in node.args if not (isinstance(a, ast.Name) and a.id == "latent_format")]
        return node

    def visit_Name(self, node):
        if node.id == "VAE_DOWNSCALE_FACTOR":
            return ast.copy_location(ast.Name("vae_downscale_factor", node.ctx), node)
        return node


def _modifier_loop(function_node):
    for node in ast.walk(function_node):
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Tuple)
                and len(node.target.elts) == 2 and isinstance(node.target.elts[1], ast.Tuple)
                and [e.id for e in node.target.elts[1].elts] == ["schedule", "kind", "p", "mask_params"]):
            return node
    raise AssertionError("modifier loop not found")


class EngineStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.origin = S.load_origin()

    def test_modifier_loop_is_upstreams_after_the_documented_substitutions(self):
        upstream_post_cfg = _function_node(S.origin_body(S.ORIGIN_DIR / "nodes.py"), "post_cfg_function")
        theirs = _Normalise().visit(copy.deepcopy(_modifier_loop(upstream_post_cfg)))
        ours = _modifier_loop(_function_node(engine, "apply_chain"))
        self.assertEqual(_dump(ours), _dump(theirs))

    def test_schedule_per_entry_is_upstreams(self):
        upstream = _function_node(S.origin_body(S.ORIGIN_DIR / "nodes.py"), "wrapped_sampler_function")
        loop = next(n for n in ast.walk(upstream) if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == "entry")
        ours = next(n for n in ast.walk(_function_node(engine, "build_entries")) if isinstance(n, ast.For))
        # same statements; the key tuple is a module constant here
        theirs_src = ast.unparse(loop).replace(
            '(\'start\', \'end\', \'bias\', \'exponent\', \'start_off\', \'end_off\', \'smooth\')', "_SCHEDULE_KEYS")
        self.assertEqual(ast.unparse(ours), theirs_src)
        self.assertEqual(engine._SCHEDULE_KEYS, ("start", "end", "bias", "exponent", "start_off", "end_off", "smooth"))

    def test_kinds_are_the_node_classes(self):
        made = set()
        tree = ast.parse(S.origin_body(S.ORIGIN_DIR / "nodes.py"))
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        for name in ("ColorcraftBasic", "ColorcraftAdvanced", "ColorcraftLuma", "ColorcraftChroma",
                     "ColorcraftChromaPlus", "ColorcraftPunch", "ColorcraftShift"):
            for node in ast.walk(classes[name]):
                if isinstance(node, ast.Dict):
                    keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                    if "kind" in keys:
                        made.add(node.values[keys.index("kind")].value)
        self.assertEqual(made, set(engine.KINDS))
        self.assertEqual(engine.ADVANCED_KINDS, {"advanced", "luma", "chroma", "chroma_plus", "punch"})

    def test_each_kind_carries_its_nodes_parameters(self):
        """``spec.params_of`` builds exactly the keys each node's ``make`` puts into its entry."""
        drop = {"strength", "start", "end", "advanced", "bias", "exponent", "start_off", "end_off", "smooth",
                "plot_steps", "dev"}
        classes = {"basic": "ColorcraftBasic", "advanced": "ColorcraftAdvanced", "luma": "ColorcraftLuma",
                   "chroma": "ColorcraftChroma", "chroma_plus": "ColorcraftChromaPlus", "punch": "ColorcraftPunch",
                   "shift": "ColorcraftShift"}
        for kind, cls_name in classes.items():
            required = getattr(self.origin.nodes, cls_name).INPUT_TYPES()["required"]
            expected = set(required) - drop - {"schedule"}
            modifier = spec.default_config().modifiers[0]
            modifier["kind"] = spec.KIND_LABELS[kind]
            with self.subTest(kind=kind):
                self.assertEqual(set(spec.params_of(modifier)), expected)


class NumericParityTests(unittest.TestCase):
    """Library functions on random tensors (the same values the static checks promise)."""

    @classmethod
    def setUpClass(cls):
        cls.origin = S.load_origin()

    def test_vector_and_mask_functions(self):
        g = torch.Generator().manual_seed(3)
        x = torch.randn(2, 16, 12, 12, generator=g)
        a1, a2, b = (torch.randn(16, generator=g) for _ in range(3))
        o = self.origin
        cases = [
            ("apply_vector_offset", (x, b, 0.3)), ("apply_vector_scale", (x, b, -0.4)),
            ("apply_vibrance", (x, a1, a2, 0.7)), ("apply_tone_compression", (x, b, 0.5)),
        ]
        for name, args in cases:
            with self.subTest(function=name):
                self.assertTrue(torch.equal(getattr(vectors, name)(*args), getattr(o.vectors, name)(*args)))
        self.assertTrue(torch.equal(vectors.apply_vibrance(x, a1, a2, 0.6, k=1.5, recenter=0.5, r_max=2.5),
                                    o.vectors.apply_vibrance(x, a1, a2, 0.6, k=1.5, recenter=0.5, r_max=2.5)))
        self.assertTrue(torch.equal(vectors.apply_chroma_contrast(x, a1, a2, 0.8, r_max=2.5, chroma_center=-0.3),
                                    o.vectors.apply_chroma_contrast(x, a1, a2, 0.8, r_max=2.5, chroma_center=-0.3)))
        mask = torch.rand(2, 1, 12, 12, generator=g)
        for name, args in (("apply_mask_spread", (mask, 0.6)), ("apply_mask_normalize", (mask,)),
                           ("apply_mask_contrast", (mask, 2.5)), ("gaussian_blur_mask", (mask, 1.3))):
            with self.subTest(function=name):
                self.assertTrue(torch.equal(getattr(masking, name)(*args), getattr(o.masking, name)(*args)))

    def test_schedules(self):
        for args in ((20, 0.5, 0.75, 0.5, 1.0, 0.0, 0.0, 0.0, True), (9, 0.1, 0.9, 0.3, -0.8, 1.5, 0.2, -0.1, False),
                     (30, 0.0, 1.0, 0.7, 0.6, -2.0, 0.0, 0.3, True), (2, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, True)):
            with self.subTest(args=args):
                self.assertTrue((schedule.make_schedule(*args) == self.origin.schedule.make_schedule(*args)).all())
        sigmas = S.flow_sigmas(12)
        sched = schedule.make_schedule(12, 0.2, 0.8, 0.4, 0.9, 1.0, 0.1, 0.0, True)
        for sigma in [float(s) for s in sigmas] + [0.987, 0.5, 0.0004, 1.2]:
            self.assertEqual(schedule.sigma_to_value(sigma, sigmas, sched),
                             self.origin.nodes.sigma_to_value(sigma, sigmas, sched))


class NodeParityTests(unittest.TestCase):
    """Bit-identical to the ComfyUI node through the real Forge hook, on all three families."""

    @classmethod
    def setUpClass(cls):
        cls.origin = S.load_origin()
        cls.fork = S.load_fork()

    def setUp(self):
        S.require_forge(self)

    def _run(self, family, scenario, walked, full=None, evals=None, p_fields=None, modules_stub=None):
        p = S.make_p(family, **(p_fields or {}))
        vae = p.sd_model.forge_objects.vae
        latent_format = p.sd_model.model_config.latent_format
        callback = S.attach_ours(p, S.scenario_args(scenario), modules_stub=modules_stub)
        self.assertIsNotNone(callback, scenario.label)
        unet = p.sd_model.forge_objects.unet
        unet.model_options.setdefault("transformer_options", {})["sampling_sigmas"] = full if full is not None else walked
        if evals is None:
            evals = ([float(walked[i]) for i in range(len(walked) - 1)]
                     + [float((walked[i] + walked[i + 1]) / 2) for i in range(len(walked) - 1)])
        with S.oracle_family(self.origin, family, self.fork):
            oracle = S.oracle_callback(self.origin, S.oracle_chain(scenario, self.origin.nodes),
                                       S.FakeVAE(vae.latent_channels, vae.latent_dim), walked, latent_format)
            moved = 0
            for k, sigma in enumerate(evals):
                x0 = S.latent(family, 1000 + k)
                ours = callback(S.forge_args(x0.clone(), sigma, unet))
                theirs = oracle(x0.clone(), sigma)
                self.assertEqual(ours.shape, x0.shape)
                self.assertTrue(torch.equal(ours, theirs),
                                f"{family} {scenario.label} eval {k} σ={sigma}: "
                                f"max |Δ| {(ours - theirs).abs().max().item()}")
                moved += not torch.equal(ours, x0)
        status = p.extra_generation_params[spec.STATUS_KEY]
        self.assertNotIn("error", status)
        return moved, len(evals), p

    def test_every_scenario_on_every_family(self):
        walked = S.flow_sigmas(10)
        for family in ("krea2", "zimage", "flux2"):
            for scenario in S.SCENARIOS:
                with self.subTest(family=family, scenario=scenario.label):
                    moved, total, _ = self._run(family, scenario, walked)
                    # the latent actually moved (not two no-ops agreeing); the windows leave edges out
                    self.assertGreaterEqual(moved, total // 2, f"{family} {scenario.label} moved {moved}/{total}")

    def test_families_resolve_from_forges_latent_format_classes(self):
        formats = S.latent_formats()
        self.assertEqual(basis.family_for_latent_format(formats.Wan21()), "krea2")
        self.assertEqual(basis.family_for_latent_format(formats.Flux()), "zimage")
        self.assertEqual(basis.family_for_latent_format(formats.Flux2()), "flux2")
        for unsupported in ("SD15", "SDXL", "SDXL_Flux2", "RGB"):
            self.assertIsNone(basis.family_for_latent_format(getattr(formats, unsupported)()), unsupported)

    def test_forge_maps_anima_to_wan21(self):
        """Anima (and Qwen-Image, Krea 2, Wan 2.1) report Wan21 in the local Forge — the krea2 basis."""
        source = (S.FORGE / "modules_forge" / "packages" / "huggingface_guess" / "model_list.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        formats = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if (isinstance(stmt, ast.Assign) and any(getattr(t, "id", None) == "latent_format" for t in stmt.targets)
                            and isinstance(stmt.value, ast.Attribute)):
                        formats[node.name] = stmt.value.attr
        for model in ("Anima", "QwenImage", "Krea2", "WAN21_T2V"):
            self.assertEqual(formats[model], "Wan21", model)
        for model in ("Flux", "Lumina2"):
            self.assertEqual(formats[model], "Flux", model)
        for model in ("Flux2K4B", "Flux2K9B", "ErnieImage"):
            self.assertEqual(formats[model], "Flux2", model)
        self.assertEqual(formats["SDXL"], "SDXL")
        self.assertEqual(formats["SD15"], "SD15")


class RangeParityTests(unittest.TestCase):
    """Slider ranges/defaults: upstream's Forge panel where it has the control, the node otherwise."""

    @classmethod
    def setUpClass(cls):
        cls.origin = S.load_origin()
        source = S.origin_body(S.ORIGIN_DIR / "scripts" / "colorcraft.py")
        # Upstream's ui() builds each tab kind in its own ``for i in range(<COUNT>)`` loop with a fresh
        # ``w = {}``: collect the ``w["name"] = gr.<Widget>(...)`` calls per loop.
        cls.widgets = {}
        for loop in ast.walk(ast.parse(source)):
            if not (isinstance(loop, ast.For) and isinstance(loop.iter, ast.Call)
                    and getattr(loop.iter.func, "id", None) == "range" and loop.iter.args
                    and isinstance(loop.iter.args[0], ast.Name)):
                continue
            section = cls.widgets.setdefault(loop.iter.args[0].id, {})
            for node in ast.walk(loop):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Subscript)
                        and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == "w"
                        and isinstance(node.value, ast.Call)):
                    key = node.targets[0].slice.value
                    kwargs = {kw.arg: kw.value for kw in node.value.keywords}
                    widget = {}
                    for name in ("minimum", "maximum", "step", "value", "choices"):
                        if name in kwargs:
                            try:
                                widget[name] = ast.literal_eval(kwargs[name])
                            except ValueError:
                                widget[name] = ast.unparse(kwargs[name])
                    section[key] = widget

    def _upstream_widget(self, section, name):
        return self.widgets[section][name]

    def test_upstream_tabs_were_found(self):
        self.assertTrue({"MODIFIER_COUNT", "MASK_COUNT", "COMBO_COUNT"} <= set(self.widgets))
        self.assertIn("exposure", self.widgets["MODIFIER_COUNT"])
        self.assertIn("mask_hardness", self.widgets["MASK_COUNT"])
        self.assertIn("operation", self.widgets["COMBO_COUNT"])

    def test_modifier_sliders_match_upstreams_forge_panel(self):
        checked = 0
        for f in spec.MODIFIER_FIELDS:
            # active: modifier I starts on here (host difference); advanced: upstream's is an
            # InputAccordion(False, label="Advanced Schedule") rather than a gr.* widget.
            if f.origin != "forge" or f.name in ("active", "advanced"):
                continue
            with self.subTest(field=f.name):
                widget = self._upstream_widget("MODIFIER_COUNT", f.name)
                if f.kind == spec.FLOAT:
                    self.assertEqual((f.minimum, f.maximum, f.step, f.default),
                                     (widget["minimum"], widget["maximum"], widget["step"], widget["value"]))
                else:
                    self.assertEqual(f.default, widget["value"])
                checked += 1
        self.assertEqual(checked, 32)
        self.assertFalse(spec.MODIFIER_BY_NAME["advanced"].default)
        self.assertEqual(spec.MODIFIER_BY_NAME["advanced"].label, "Advanced Schedule")
        self.assertFalse(spec.MODIFIER_BY_NAME["active"].default)

    def test_mask_and_combo_sliders_match_upstreams_forge_panel(self):
        for section, fields in (("MASK_COUNT", spec.LEAF_FIELDS), ("COMBO_COUNT", spec.COMBO_FIELDS)):
            for f in fields:
                widget = self._upstream_widget(section, f.name)
                with self.subTest(section=section, field=f.name):
                    if f.kind == spec.FLOAT:
                        self.assertEqual((f.minimum, f.maximum, f.step, f.default),
                                         (widget["minimum"], widget["maximum"], widget["step"], widget["value"]))
                    elif f.name == "mask_axis":
                        self.assertEqual(widget["value"], "MASK_AXIS_OPTIONS[0]")
                        self.assertEqual(f.default, masking.MASK_AXIS_OPTIONS[0])
                    else:
                        self.assertEqual(f.default, widget["value"])

    def test_node_only_controls_use_the_nodes_ranges(self):
        required = self.origin.nodes.ColorcraftAdvanced.INPUT_TYPES()["required"]
        for f in spec.MODIFIER_FIELDS:
            if f.origin != "node":
                continue
            with self.subTest(field=f.name):
                node_type = required[f.name]
                if f.kind == spec.FLOAT:
                    opts = node_type[1]
                    self.assertEqual((f.minimum, f.maximum, f.step, f.default),
                                     (opts["min"], opts["max"], opts["step"], opts["default"]))
                elif f.kind == spec.CHOICE:
                    self.assertEqual(list(f.choices), node_type[0])
                elif f.name in ("more_colors", "color_shift"):
                    # Documented host difference: upstream's Forge tab always applies Chroma Plus and Color
                    # Shift, so the node's gates start ON here (the node's default is False).
                    self.assertEqual(node_type[1]["default"], False)
                    self.assertTrue(f.default)
                else:
                    self.assertEqual(f.default, node_type[1]["default"])

    def test_choice_lists_are_upstreams(self):
        scope = S.upstream_forge_functions("MASK_MODE_OPTIONS", "COMBO_OP_OPTIONS", "COLOR_SHIFT_MODE_OPTIONS",
                                           "MODIFIER_COUNT", "MASK_COUNT", "COMBO_COUNT", "ROMAN")
        self.assertEqual(spec.MASK_MODE_OPTIONS, scope["MASK_MODE_OPTIONS"])
        self.assertEqual(spec.COMBO_OP_OPTIONS, scope["COMBO_OP_OPTIONS"])
        self.assertEqual(spec.COLOR_SHIFT_MODE_OPTIONS, scope["COLOR_SHIFT_MODE_OPTIONS"])
        self.assertEqual((spec.MODIFIER_COUNT, spec.MASK_COUNT, spec.COMBO_COUNT),
                         (scope["MODIFIER_COUNT"], scope["MASK_COUNT"], scope["COMBO_COUNT"]))
        self.assertEqual(spec.ROMAN, scope["ROMAN"])
        self.assertEqual(spec.combo_ref_choices(2), ["none"] + spec.MASK_TAGS + ["C1", "C2"])

    def test_field_lists_follow_upstreams_order(self):
        scope = S.upstream_forge_functions("MODIFIER_FIELDS", "LEAF_FIELDS", "COMBO_FIELDS")
        self.assertEqual([f.name for f in spec.LEAF_FIELDS], scope["LEAF_FIELDS"])
        self.assertEqual([f.name for f in spec.COMBO_FIELDS], scope["COMBO_FIELDS"])
        self.assertEqual(spec.UPSTREAM_MODIFIER_FIELDS, scope["MODIFIER_FIELDS"])
        ours = [f.name for f in spec.MODIFIER_FIELDS if f.name in scope["MODIFIER_FIELDS"]]
        theirs = [name for name in scope["MODIFIER_FIELDS"] if name in ours]
        self.assertEqual(ours, theirs)
        self.assertEqual(sorted(set(scope["MODIFIER_FIELDS"]) - {f.name for f in spec.MODIFIER_FIELDS}), ["hires"])


if __name__ == "__main__":
    unittest.main()
