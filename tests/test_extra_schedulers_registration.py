"""Extra Schedulers — registration in Forge's modules.sd_schedulers.

First against a stand-in module (always runs), then against Forge's own files when a Forge
checkout is found: the real ``modules/sd_schedulers.py`` and ``modules/sd_samplers.py`` are loaded
with their heavy imports stubbed, and Forge's real ``KDiffusionSampler.get_sigmas`` (cut out of
``modules/sd_samplers_kdiffusion.py``) calls the registered functions the way Forge does.
"""
from __future__ import annotations

import ast
import contextlib
import functools
import importlib.util
import io
import math
import sys
import types
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import _extra_samplers_fixtures as fixtures  # noqa: E402
import _extra_schedulers_support as support  # noqa: E402
from sam3ext.extra_schedulers import registry  # noqa: E402
from sam3ext.extra_schedulers import schedulers as sch  # noqa: E402
from sam3ext.extra_schedulers import settings as st  # noqa: E402

EXPECTED = [
    ("cosine", "Cosine"),
    ("cosine_exponential", "CosineExponential blend"),
    ("phi", "Phi"),
    ("laplace", "Laplace"),
    ("karras_dynamic", "Karras Dynamic"),
    ("custom", "custom"),
    ("react_cosinusoidal_dynsf", "React Cosinusoidal DynSF"),
    ("flow_cosmos_rho7", "Flow Cosmos rho7"),
    ("flow_cosmos_dynamic", "Flow Cosmos Dynamic"),
]
# The two that ask Forge for its model (flow or eps/v).
NEED_MODEL = ("Flow Cosmos rho7", "Flow Cosmos Dynamic")


def _register(module, **kwargs):
    stderr = io.StringIO()
    registry._LOGGED.clear()
    with contextlib.redirect_stderr(stderr):
        report = registry.register_schedulers(module, **kwargs)
    return report, stderr.getvalue()


class StubRegistrationTests(unittest.TestCase):
    def test_all_nine_are_added_in_order_with_forges_fields(self):
        module = support.stub_sd_schedulers()
        report, _log = _register(module, hidden_labels=[])
        self.assertEqual(report.added, [label for _name, label in EXPECTED])
        self.assertIsNone(report.error)
        self.assertEqual(len(EXPECTED), 9)
        ours = module.all_schedulers[-9:]
        self.assertEqual([(s.name, s.label) for s in ours], EXPECTED)
        self.assertEqual(module.schedulers[-9:], ours)
        for scheduler in ours:
            with self.subTest(label=scheduler.label):
                self.assertIs(module.schedulers_map[scheduler.name], scheduler)
                self.assertIs(module.schedulers_map[scheduler.label], scheduler)
                # only the Flow Cosmos schedulers ask Forge for its model (flow or eps/v)
                self.assertIs(scheduler.need_inner_model, scheduler.label in NEED_MODEL)
                self.assertTrue(registry.is_ours(scheduler))
        rho = {s.label: s.default_rho for s in ours}
        self.assertEqual(rho.pop("Karras Dynamic"), 7.0)
        self.assertEqual(set(rho.values()), {-1.0})

    def test_the_map_is_updated_in_place_and_keeps_foreign_keys(self):
        module = support.stub_sd_schedulers()
        original_map = module.schedulers_map
        foreign = support.StubScheduler("foreign", "Foreign", lambda *a, **k: None)
        original_map["only-in-the-map"] = foreign
        _register(module, hidden_labels=[])
        self.assertIs(module.schedulers_map, original_map)
        self.assertIs(module.schedulers_map["only-in-the-map"], foreign)
        self.assertIs(module.schedulers_map["Karras"], module.all_schedulers[1])

    def test_hidden_labels_stay_out_of_the_dropdown_and_the_map(self):
        module = support.stub_sd_schedulers()
        report, _log = _register(module, hidden_labels=["Phi", "custom"])
        self.assertEqual(report.hidden, ["Phi", "custom"])
        labels = [s.label for s in module.schedulers]
        self.assertNotIn("Phi", labels)
        self.assertNotIn("custom", labels)
        self.assertIn("Phi", [s.label for s in module.all_schedulers])
        self.assertNotIn("Phi", module.schedulers_map)
        self.assertNotIn("phi", module.schedulers_map)
        self.assertIn("Laplace", module.schedulers_map)
        self.assertIn("hidden by Settings", registry.summary(report))

    def test_registering_twice_adds_nothing(self):
        module = support.stub_sd_schedulers()
        _register(module, hidden_labels=["Phi"])
        before = list(module.all_schedulers), list(module.schedulers)
        report, log = _register(module, hidden_labels=["Phi"])
        self.assertEqual(report.added, [])
        self.assertEqual(report.already, [label for _name, label in EXPECTED])
        self.assertEqual((list(module.all_schedulers), list(module.schedulers)), before)
        self.assertEqual(log, "")
        self.assertIsNone(registry.summary(report))

    def test_a_taken_label_is_skipped_and_logged_once(self):
        module = support.stub_sd_schedulers()
        foreign = support.StubScheduler("laplace_other", "Laplace", lambda *a, **k: None)
        module.all_schedulers.append(foreign)
        module.schedulers.append(foreign)
        module.schedulers_map.update({"laplace_other": foreign, "Laplace": foreign})
        report, log = _register(module, hidden_labels=[])
        self.assertEqual(report.skipped, ["Laplace"])
        self.assertEqual(len(report.added), 8)
        self.assertIs(module.schedulers_map["Laplace"], foreign)
        self.assertNotIn("laplace", module.schedulers_map)
        self.assertEqual(log.count("'Laplace' is not added"), 1)
        # A second registration does not log it again.
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            registry.register_schedulers(module, hidden_labels=[])
        self.assertEqual(stderr.getvalue(), "")

    def test_taken_names_and_aliases_count_case_insensitively(self):
        module = support.stub_sd_schedulers()
        module.all_schedulers.append(support.StubScheduler("COSINE", "Something else", lambda *a, **k: None))
        module.all_schedulers.append(support.StubScheduler("x", "phi", lambda *a, **k: None))
        module.all_schedulers.append(support.StubScheduler("y", "Y", lambda *a, **k: None, aliases=["Karras dynamic"]))
        report, _log = _register(module, hidden_labels=[])
        self.assertEqual(sorted(report.skipped), ["Cosine", "Karras Dynamic", "Phi"])
        self.assertEqual(sorted(report.added), ["CosineExponential blend", "Flow Cosmos Dynamic", "Flow Cosmos rho7",
                                                "Laplace", "React Cosinusoidal DynSF", "custom"])

    def test_forge_lookup_cache_is_cleared_and_the_kdiffusion_map_mirrored(self):
        calls = []

        @functools.cache
        def get_sampler_and_scheduler(sampler, scheduler):
            calls.append(scheduler)
            return sampler, scheduler

        get_sampler_and_scheduler("Euler", "Laplace")
        samplers = types.SimpleNamespace(get_sampler_and_scheduler=get_sampler_and_scheduler)
        kdiffusion = types.SimpleNamespace(k_diffusion_scheduler={"karras": object()})
        module = support.stub_sd_schedulers()
        _register(module, hidden_labels=["Phi"], sd_samplers=samplers, sd_samplers_kdiffusion=kdiffusion)
        self.assertEqual(get_sampler_and_scheduler.cache_info().currsize, 0)
        self.assertIn("karras", kdiffusion.k_diffusion_scheduler)
        self.assertIs(kdiffusion.k_diffusion_scheduler["laplace"], sch.laplace)
        self.assertIs(kdiffusion.k_diffusion_scheduler["react_cosinusoidal_dynsf"], sch.react_cosinusoidal_dynsf)
        self.assertIs(kdiffusion.k_diffusion_scheduler["flow_cosmos_rho7"], sch.flow_cosmos_rho7)
        self.assertIs(kdiffusion.k_diffusion_scheduler["flow_cosmos_dynamic"], sch.flow_cosmos_dynamic)
        self.assertNotIn("phi", kdiffusion.k_diffusion_scheduler)

    def test_the_fork_readme_labels_resolve_to_ours(self):
        module = support.stub_sd_schedulers()
        report, _log = _register(module, hidden_labels=[])
        ours = {s.label: s for s in module.all_schedulers if registry.is_ours(s)}
        # An alias equal to the scheduler's own name or label is left out: it is already a key.
        self.assertEqual({label: s.aliases for label, s in ours.items()}, {
            "Cosine": None, "CosineExponential blend": ["cosine-exponential blend"], "Phi": None, "Laplace": None,
            "Karras Dynamic": ["karras dynamic"], "custom": None, "React Cosinusoidal DynSF": None,
            "Flow Cosmos rho7": None, "Flow Cosmos Dynamic": None,
        })
        readme = ["cosine", "cosine-exponential blend", "phi", "Laplace", "Karras Dynamic", "custom"]   # as written there
        for written, label in zip(readme, registry.SCHEDULER_LABELS):
            with self.subTest(written=written):
                self.assertIs(module.schedulers_map[written], ours[label])
                self.assertIs(module.schedulers_map[written.lower()], ours[label])
        self.assertEqual(report.skipped_aliases, [])
        self.assertEqual(set(registry.README_LABELS.values()), {w.lower() for w in readme})

    def test_a_taken_alias_is_skipped_and_logged_once(self):
        module = support.stub_sd_schedulers()
        foreign = support.StubScheduler("other_blend", "Cosine-Exponential Blend", lambda *a, **k: None)
        module.all_schedulers.append(foreign)
        module.schedulers.append(foreign)
        module.schedulers_map.update({"other_blend": foreign, "Cosine-Exponential Blend": foreign})
        map_only = support.StubScheduler("map_only", "Map only", lambda *a, **k: None)
        module.schedulers_map["karras dynamic"] = map_only   # a key only in the map
        report, log = _register(module, hidden_labels=[])
        self.assertEqual(report.added, list(registry.SCHEDULER_LABELS), "the schedulers themselves are added")
        self.assertEqual(report.skipped_aliases, ["cosine-exponential blend", "karras dynamic"])
        self.assertIsNone(module.schedulers_map["CosineExponential blend"].aliases)
        self.assertNotIn("cosine-exponential blend", module.schedulers_map)   # compared case-insensitively, like names
        self.assertIs(module.schedulers_map["karras dynamic"], map_only)
        self.assertEqual(log.count("alias 'cosine-exponential blend' of 'CosineExponential blend' is not added"), 1)
        self.assertIn("aliases skipped (taken): cosine-exponential blend, karras dynamic", registry.summary(report))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            again = registry.register_schedulers(module, hidden_labels=[])   # Reload UI
        self.assertEqual((again.added, again.skipped_aliases, stderr.getvalue()), ([], [], ""))

    def test_a_hidden_schedulers_alias_stays_out_of_the_map(self):
        module = support.stub_sd_schedulers()
        _register(module, hidden_labels=["CosineExponential blend"])
        self.assertNotIn("cosine-exponential blend", module.schedulers_map)
        self.assertIn("karras dynamic", module.schedulers_map)

    def test_never_raises(self):
        report, log = _register(object(), hidden_labels=[])
        self.assertIsNotNone(report.error)
        self.assertIn("schedulers not registered", log)
        self.assertIn("error:", registry.summary(report))

    def test_functions_carry_the_owner_tag(self):
        for spec in registry.SCHEDULER_SPECS:
            self.assertEqual(spec.function._sam_extra_owner, registry.OWNER)
        self.assertEqual(registry.SCHEDULER_LABELS, tuple(label for _name, label in EXPECTED))

    def test_react_dynsf_follows_the_six_without_a_readme_alias(self):
        self.assertEqual(registry.SCHEDULER_SPECS[5].name, "custom")
        spec = registry.SCHEDULER_SPECS[6]
        self.assertEqual((spec.name, spec.label, spec.function), ("react_cosinusoidal_dynsf",
                                                                  registry.LABEL_REACT_DYNSF,
                                                                  sch.react_cosinusoidal_dynsf))
        self.assertEqual(registry.LABEL_REACT_DYNSF, "React Cosinusoidal DynSF")
        self.assertEqual((spec.aliases, spec.default_rho, spec.need_inner_model), ((), -1.0, False))
        self.assertNotIn("react_cosinusoidal_dynsf", registry.README_LABELS)
        # The desktop app pins the label and the factor by AST: module-level literals.
        for path, name, value in (("sam3ext/extra_schedulers/registry.py", "LABEL_REACT_DYNSF", "React Cosinusoidal DynSF"),
                                  ("sam3ext/extra_schedulers/schedulers.py", "REACT_DYNSF_FACTOR", 2.15)):
            tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
            literal = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                           and [getattr(t, "id", None) for t in node.targets] == [name])
            self.assertIsInstance(literal, ast.Constant)
            self.assertEqual(literal.value, value)

    def test_flow_cosmos_rho7_follows_react_and_asks_for_the_model(self):
        """KeithZ117/Comfyui-anima-sampler's option name, Forge-style label, no alias, no rho of its own (Settings →
        rho does not reach it: Forge passes rho only to a scheduler with a default_rho), and need_inner_model so
        that Forge passes the model whose predictor tells flow from eps/v."""
        spec = registry.SCHEDULER_SPECS[7]
        self.assertEqual(registry.SCHEDULER_SPECS[6].label, registry.LABEL_REACT_DYNSF)
        self.assertEqual((spec.name, spec.label, spec.function), ("flow_cosmos_rho7", registry.LABEL_FLOW_COSMOS_RHO7,
                                                                  sch.flow_cosmos_rho7))
        self.assertEqual(registry.LABEL_FLOW_COSMOS_RHO7, "Flow Cosmos rho7")
        self.assertEqual((spec.aliases, spec.default_rho, spec.need_inner_model), ((), -1.0, True))
        self.assertNotIn("flow_cosmos_rho7", registry.README_LABELS)
        tree = ast.parse((ROOT / "sam3ext/extra_schedulers/registry.py").read_text(encoding="utf-8"))
        literal = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                       and [getattr(t, "id", None) for t in node.targets] == ["LABEL_FLOW_COSMOS_RHO7"])
        self.assertIsInstance(literal, ast.Constant)
        self.assertEqual(literal.value, "Flow Cosmos rho7")

    def test_flow_cosmos_dynamic_is_appended_last_right_after_flow_cosmos_rho7(self):
        """D20: this extension's own name and Forge-style label, no alias, no rho of its own (it reads the accordion's
        Flow Cosmos rho; Settings → rho does not reach it), need_inner_model like Flow Cosmos rho7. The label is a
        module-level literal (the desktop app pins labels by AST); FLOW_COSMOS_LABELS names the two schedulers that use
        the three Flow Cosmos values."""
        spec = registry.SCHEDULER_SPECS[-1]
        self.assertEqual(registry.SCHEDULER_SPECS[-2].name, "flow_cosmos_rho7")
        self.assertEqual((spec.name, spec.label, spec.function),
                         ("flow_cosmos_dynamic", registry.LABEL_FLOW_COSMOS_DYNAMIC, sch.flow_cosmos_dynamic))
        self.assertEqual(registry.LABEL_FLOW_COSMOS_DYNAMIC, "Flow Cosmos Dynamic")
        self.assertEqual(sch.FLOW_COSMOS_DYNAMIC, registry.LABEL_FLOW_COSMOS_DYNAMIC, "the name in its messages")
        self.assertEqual((spec.aliases, spec.default_rho, spec.need_inner_model), ((), -1.0, True))
        self.assertNotIn("flow_cosmos_dynamic", registry.README_LABELS)
        self.assertEqual(registry.FLOW_COSMOS_LABELS, ("Flow Cosmos rho7", "Flow Cosmos Dynamic"))
        tree = ast.parse((ROOT / "sam3ext/extra_schedulers/registry.py").read_text(encoding="utf-8"))
        literal = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                       and [getattr(t, "id", None) for t in node.targets] == ["LABEL_FLOW_COSMOS_DYNAMIC"])
        self.assertIsInstance(literal, ast.Constant)
        self.assertEqual(literal.value, "Flow Cosmos Dynamic")


# ---------------------------------------------------------------------------
# Forge's own files
# ---------------------------------------------------------------------------
SamplerDataTuple = namedtuple("SamplerData", ["name", "constructor", "aliases", "options"])


class _SamplerData(SamplerDataTuple):
    def total_steps(self, steps):
        return steps * 2 if self.options.get("second_order", False) else steps


def _load_forge_modules(hidden_schedulers=()):
    """Forge's real modules/sd_schedulers.py and modules/sd_samplers.py, heavy imports stubbed."""
    root = support.forge_root()
    opts = types.SimpleNamespace(hide_schedulers=list(hidden_schedulers), hide_samplers=[])
    shared = types.SimpleNamespace(opts=opts)
    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []
    modules_pkg.shared = shared
    kd_sampling = types.SimpleNamespace(
        get_sigmas_karras=lambda *a, **k: None,
        get_sigmas_exponential=lambda *a, **k: None,
        get_sigmas_polyexponential=lambda *a, **k: None,
    )
    k_diffusion = types.ModuleType("k_diffusion")
    k_diffusion.sampling = kd_sampling
    stubs = {"modules": modules_pkg, "modules.shared": shared, "k_diffusion": k_diffusion,
             "k_diffusion.sampling": kd_sampling}
    with support.stub_modules(stubs):
        spec = importlib.util.spec_from_file_location("modules.sd_schedulers", root / "modules" / "sd_schedulers.py")
        sd_schedulers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sd_schedulers)
        modules_pkg.sd_schedulers = sd_schedulers

        def constructor(model):
            return None

        common = types.SimpleNamespace(SamplerData=_SamplerData, sample_to_image=None, samples_to_image_grid=None,
                                       sample_to_video=None)
        kdiffusion = types.SimpleNamespace(samplers_data_k_diffusion=[
            _SamplerData("Euler", constructor, ["k_euler"], {}),
            _SamplerData("DPM++ 2M", constructor, ["k_dpmpp_2m"], {"scheduler": "karras"}),
        ])
        modules_pkg.sd_samplers_common = common
        modules_pkg.sd_samplers_kdiffusion = kdiffusion
        modules_pkg.sd_samplers_timesteps = types.SimpleNamespace(samplers_data_timesteps=[])
        forge_pkg = types.ModuleType("modules_forge")
        forge_pkg.__path__ = []
        forge_pkg.forge_alter_samplers = types.SimpleNamespace(samplers_data_alter=[])
        more = {"modules.sd_samplers_common": common, "modules.sd_samplers_kdiffusion": kdiffusion,
                "modules.sd_schedulers": sd_schedulers, "modules_forge": forge_pkg,
                "modules_forge.forge_alter_samplers": forge_pkg.forge_alter_samplers}
        with support.stub_modules(more):
            spec = importlib.util.spec_from_file_location("modules.sd_samplers", root / "modules" / "sd_samplers.py")
            sd_samplers = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sd_samplers)
    return sd_schedulers, sd_samplers, opts


@unittest.skipUnless(support.forge_root(), "Forge checkout not found")
class ForgeRegistryTests(unittest.TestCase):
    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)
        self.sd_schedulers, self.sd_samplers, self.opts = _load_forge_modules(hidden_schedulers=["Phi"])

    def _register(self):
        report, _log = _register(self.sd_schedulers, hidden_labels=self.opts.hide_schedulers,
                                 sd_samplers=self.sd_samplers, sd_samplers_kdiffusion=types.SimpleNamespace())
        return report

    def test_registers_into_forges_real_lists(self):
        forge_labels = [s.label for s in self.sd_schedulers.all_schedulers]
        report = self._register()
        self.assertEqual(report.hidden, ["Phi"])
        self.assertEqual([s.label for s in self.sd_schedulers.all_schedulers],
                         forge_labels + [label for _name, label in EXPECTED])
        visible = [s.label for s in self.sd_schedulers.schedulers]
        self.assertNotIn("Phi", visible)
        for label in ("Cosine", "CosineExponential blend", "Laplace", "Karras Dynamic", "custom", "React Cosinusoidal DynSF",
                      "Flow Cosmos rho7", "Flow Cosmos Dynamic"):
            self.assertIn(label, visible)
            self.assertIsInstance(self.sd_schedulers.schedulers_map[label], self.sd_schedulers.Scheduler)
        # Forge's own formula for the map gives the same mapping, plus our two fork-README aliases.
        forge_formula = {**{x.name: x for x in self.sd_schedulers.schedulers},
                         **{x.label: x for x in self.sd_schedulers.schedulers}}
        aliases = {alias: x for x in self.sd_schedulers.schedulers if registry.is_ours(x) for alias in x.aliases or ()}
        self.assertEqual(set(aliases), {"cosine-exponential blend", "karras dynamic"})
        self.assertEqual(self.sd_schedulers.schedulers_map, {**forge_formula, **aliases})

    def test_infotext_lookup_finds_the_new_labels_after_the_cache_is_cleared(self):
        lookup = self.sd_samplers.get_sampler_and_scheduler
        self.assertEqual(lookup("Euler", "Laplace"), ("Euler", "Automatic"))   # cached before registration
        self._register()
        self.assertEqual(lookup("Euler", "Laplace"), ("Euler", "Laplace"))
        self.assertEqual(self.sd_samplers.get_scheduler_from_infotext({"Sampler": "Euler", "Schedule type": "custom"}),
                         "custom")
        self.assertEqual(lookup("DPM++ 2M", "Karras Dynamic"), ("DPM++ 2M", "Karras Dynamic"))
        self.assertEqual(lookup("Euler", "cosine_exponential"), ("Euler", "CosineExponential blend"))
        self.assertEqual(lookup("Euler", "Phi"), ("Euler", "Automatic"), "hidden schedulers are not looked up")
        hires = self.sd_samplers.get_hr_sampler_and_scheduler(
            {"Sampler": "Euler", "Schedule type": "Karras", "Hires schedule type": "Laplace"})
        self.assertEqual(hires, ("Use same sampler", "Laplace"))

    def test_forges_get_sigmas_calls_the_new_schedulers(self):
        self._register()
        get_sigmas, opts = self._forge_get_sigmas()
        model_sigmas = torch.linspace(0.0291675, 14.614642, 1000)
        sampler = types.SimpleNamespace(config=types.SimpleNamespace(options={}),
                                        model_wrap=types.SimpleNamespace(sigmas=model_sigmas))
        lo, hi = model_sigmas[0].item(), model_sigmas[-1].item()

        def run(label, steps=12, **sampler_options):
            sampler.config.options = sampler_options
            p = types.SimpleNamespace(scheduler=label, hr_scheduler=None, is_hr_pass=False, extra_generation_params={},
                                      sampler_noise_scheduler_override=None)
            return get_sigmas(sampler, p, steps), p

        st.set_active(st.ExtraSchedulerSettings(laplace_mu=0.7, laplace_beta=0.9, custom_expression="m + (M - m) * (1 - x) ** 2"))
        for label, function in (("Cosine", sch.cosine), ("CosineExponential blend", sch.cosine_exponential_blend),
                                ("Laplace", sch.laplace), ("Karras Dynamic", sch.karras_dynamic), ("custom", sch.custom),
                                ("React Cosinusoidal DynSF", sch.react_cosinusoidal_dynsf)):
            with self.subTest(label=label):
                sigmas, p = run(label)
                self.assertTrue(torch.equal(sigmas, function(n=12, sigma_min=lo, sigma_max=hi)))
                self.assertEqual(p.extra_generation_params["Schedule type"], label)
                self.assertNotIn("Schedule rho", p.extra_generation_params)

        opts.rho = 4.5   # Settings → rho reaches Karras Dynamic like Karras and is written to infotext
        sigmas, p = run("Karras Dynamic")
        self.assertTrue(torch.equal(sigmas, sch.karras_dynamic(12, lo, hi, rho=4.5)))
        self.assertEqual(p.extra_generation_params["Schedule rho"], 4.5)
        sigmas, p = run("Cosine")
        self.assertNotIn("Schedule rho", p.extra_generation_params, "rho only for schedulers that declare one")
        opts.rho = 0

        sigmas, _p = run("custom", discard_next_to_last_sigma=True)   # DPM2/3M SDE-style samplers
        full = sch.custom(n=13, sigma_min=lo, sigma_max=hi)
        self.assertTrue(torch.equal(sigmas, torch.cat([full[:-2], full[-1:]])))

    def test_reforge_infotexts_find_react_dynsf(self):
        """reForge writes "Schedule type: React Cosinusoidal DynSF"; Forge's own lookups find ours by label and name."""
        self._register()
        lookup = self.sd_samplers.get_sampler_and_scheduler
        for written in ("React Cosinusoidal DynSF", "react_cosinusoidal_dynsf"):
            with self.subTest(written=written):
                self.assertEqual(lookup("Euler", written), ("Euler", "React Cosinusoidal DynSF"))
        self.assertEqual(self.sd_samplers.get_scheduler_from_infotext(
            {"Sampler": "Euler", "Schedule type": "React Cosinusoidal DynSF"}), "React Cosinusoidal DynSF")

    def test_flow_cosmos_rho7_infotext_pastes_back_and_xyz_lists_it(self):
        """Forge writes "Schedule type: Flow Cosmos rho7"; its infotext parser and the API's lookup find ours by
        label and by name. The XYZ "Schedule type" axis offers the labels of ``sd_schedulers.schedulers``
        (scripts/xyz_grid.py) and sets ``p.scheduler`` to one, which get_sigmas looks up (test below)."""
        self._register()
        lookup = self.sd_samplers.get_sampler_and_scheduler
        for written in ("Flow Cosmos rho7", "flow_cosmos_rho7"):
            with self.subTest(written=written):
                self.assertEqual(lookup("Euler", written), ("Euler", "Flow Cosmos rho7"))
                self.assertIs(self.sd_schedulers.schedulers_map[written].function, sch.flow_cosmos_rho7)
        self.assertEqual(self.sd_samplers.get_scheduler_from_infotext(
            {"Sampler": "Euler", "Schedule type": "Flow Cosmos rho7"}), "Flow Cosmos rho7")
        self.assertEqual(self.sd_samplers.get_hr_sampler_and_scheduler(
            {"Sampler": "Euler", "Schedule type": "Karras", "Hires schedule type": "Flow Cosmos rho7"}),
            ("Use same sampler", "Flow Cosmos rho7"))
        self.assertIn("Flow Cosmos rho7", [x.label for x in self.sd_schedulers.schedulers])   # the XYZ axis choices

    def test_flow_cosmos_dynamic_infotext_pastes_back_and_xyz_lists_it(self):
        """As for Flow Cosmos rho7: "Schedule type: Flow Cosmos Dynamic" (or its name) finds ours through Forge's own
        lookups, also in the Hires field and in legacy "Sampler: <sampler> <scheduler>" infotext; the XYZ "Schedule
        type" axis offers it by label."""
        self._register()
        lookup = self.sd_samplers.get_sampler_and_scheduler
        for written in ("Flow Cosmos Dynamic", "flow_cosmos_dynamic"):
            with self.subTest(written=written):
                self.assertEqual(lookup("Euler", written), ("Euler", "Flow Cosmos Dynamic"))
                self.assertIs(self.sd_schedulers.schedulers_map[written].function, sch.flow_cosmos_dynamic)
        self.assertEqual(lookup("Euler Flow Cosmos Dynamic", None), ("Euler", "Flow Cosmos Dynamic"))
        self.assertEqual(lookup("Euler Flow Cosmos rho7", None), ("Euler", "Flow Cosmos rho7"))
        self.assertEqual(self.sd_samplers.get_scheduler_from_infotext(
            {"Sampler": "Euler", "Schedule type": "Flow Cosmos Dynamic"}), "Flow Cosmos Dynamic")
        self.assertEqual(self.sd_samplers.get_hr_sampler_and_scheduler(
            {"Sampler": "Euler", "Schedule type": "Flow Cosmos rho7", "Hires schedule type": "Flow Cosmos Dynamic"}),
            ("Use same sampler", "Flow Cosmos Dynamic"))
        labels = [x.label for x in self.sd_schedulers.schedulers]   # the XYZ axis choices
        self.assertEqual(labels.index("Flow Cosmos Dynamic"), labels.index("Flow Cosmos rho7") + 1)

    def test_forges_get_sigmas_runs_flow_cosmos_dynamic_with_the_models_predictor(self):
        """Forge's real get_sigmas and model_wrap around Forge's real predictors: the flow list on Anima (shift 3),
        Karras Dynamic with the accordion's ρ on SDXL eps; Settings → rho never reaches it (no default_rho), Settings →
        sigma min reaches the eps/v branch only; Forge writes no rho for it."""
        self._register()
        get_sigmas, opts = self._forge_get_sigmas()
        sampling_path = "modules_forge/packages/k_diffusion/sampling.py"
        append_zero = support.forge_function(sampling_path, "append_zero", {"torch": torch, "math": math})
        linker = support.forge_function("modules_forge/packages/k_diffusion/external.py", "ForgeScheduleLinker",
                                        {"torch": torch, "nn": torch.nn,
                                         "sampling": types.SimpleNamespace(append_zero=append_zero)})
        flow_wrap = linker(fixtures.forge_flow_predictor(3.0))
        eps_wrap = linker(fixtures.forge_eps_predictor())

        def run(model_wrap, steps=12):
            sampler = types.SimpleNamespace(config=types.SimpleNamespace(options={}), model_wrap=model_wrap)
            p = types.SimpleNamespace(scheduler="Flow Cosmos Dynamic", hr_scheduler=None, is_hr_pass=False,
                                      extra_generation_params={}, sampler_noise_scheduler_override=None)
            return get_sigmas(sampler, p, steps), p

        lo, hi = eps_wrap.sigmas[0].item(), eps_wrap.sigmas[-1].item()
        flow_list = sch._with_final_zero(sch._flow_cosmos_dynamic_times(12)[0], "cpu")
        stderr = io.StringIO()
        with mock.patch.object(sch, "_flow_cosmos_dynamic_not_flow_logged", False), contextlib.redirect_stderr(stderr):
            sigmas, p = run(flow_wrap)
            self.assertTrue(torch.equal(sigmas, flow_list))
            self.assertEqual(p.extra_generation_params, {"Schedule type": "Flow Cosmos Dynamic"})
            self.assertEqual(stderr.getvalue(), "")
            sigmas, p = run(eps_wrap)
            self.assertTrue(torch.equal(sigmas, sch.karras_dynamic(12, lo, hi, rho=7.0)))
            self.assertEqual(p.extra_generation_params, {"Schedule type": "Flow Cosmos Dynamic"})
            run(eps_wrap)
            self.assertEqual(stderr.getvalue().count("Flow Cosmos Dynamic: not a flow model"), 1)
            opts.rho = 4.5   # Settings → rho reaches Karras Dynamic, not Flow Cosmos Dynamic
            sigmas, p = run(eps_wrap)
            self.assertTrue(torch.equal(sigmas, sch.karras_dynamic(12, lo, hi, rho=7.0)))
            self.assertNotIn("Schedule rho", p.extra_generation_params)
            opts.rho = 0
            opts.sigma_min = 0.1   # Settings → sigma min override: the eps/v branch's range only
            sigmas, _p = run(eps_wrap)
            self.assertTrue(torch.equal(sigmas, sch.karras_dynamic(12, 0.1, hi)))
            sigmas, _p = run(flow_wrap)
            self.assertTrue(torch.equal(sigmas, flow_list), "the flow list does not use the model's range")
            opts.sigma_min = 0
            st.set_active(st.ExtraSchedulerSettings(flow_cosmos_rho=5.0, flow_cosmos_sigma_max=240.0,
                                                    flow_cosmos_sigma_min=0.006))
            sigmas, _p = run(flow_wrap)
            self.assertTrue(torch.equal(sigmas, sch._with_final_zero(
                sch._flow_cosmos_dynamic_times(12, 5.0, 0.006, 240.0)[0], "cpu")))
            sigmas, _p = run(eps_wrap)
            self.assertTrue(torch.equal(sigmas, sch.karras_dynamic(12, lo, hi, rho=5.0)))

    def test_forges_get_sigmas_runs_flow_cosmos_rho7_with_the_models_predictor(self):
        """Forge's real get_sigmas with its real model_wrap (``ForgeScheduleLinker``) around Forge's real predictors:
        the flow list on Anima (shift 3), Forge's Karras on SDXL eps; Settings → rho never reaches it, Settings →
        sigma min reaches the Karras branch as it reaches Forge's Karras and does not change the flow list."""
        self._register()
        get_sigmas, opts = self._forge_get_sigmas()
        # Forge's k_diffusion.sampling (executed from Forge's file), whose get_sigmas_karras the eps/v branch calls
        self.enterContext(fixtures.installed_k_sampling(fixtures.forge_k_sampling()))
        sampling_path = "modules_forge/packages/k_diffusion/sampling.py"
        namespace = {"torch": torch, "math": math}
        append_zero = support.forge_function(sampling_path, "append_zero", namespace)
        karras = support.forge_function(sampling_path, "get_sigmas_karras", namespace)
        linker = support.forge_function("modules_forge/packages/k_diffusion/external.py", "ForgeScheduleLinker",
                                        {"torch": torch, "nn": torch.nn,
                                         "sampling": types.SimpleNamespace(append_zero=append_zero)})
        flow_wrap = linker(fixtures.forge_flow_predictor(3.0))
        eps_wrap = linker(fixtures.forge_eps_predictor())
        real_is_flow_model = sch.is_flow_model
        seen = []

        def spy(model):
            seen.append(model)
            return real_is_flow_model(model)

        def run(model_wrap, steps=12):
            sampler = types.SimpleNamespace(config=types.SimpleNamespace(options={}), model_wrap=model_wrap)
            p = types.SimpleNamespace(scheduler="Flow Cosmos rho7", hr_scheduler=None, is_hr_pass=False,
                                      extra_generation_params={}, sampler_noise_scheduler_override=None)
            with mock.patch.object(sch, "is_flow_model", side_effect=spy):
                return get_sigmas(sampler, p, steps), p

        flow_list = sch.flow_cosmos_rho7(12, 0.5, 1.0, inner_model=flow_wrap)
        lo, hi = eps_wrap.sigmas[0].item(), eps_wrap.sigmas[-1].item()
        stderr = io.StringIO()
        with mock.patch.object(sch, "_flow_cosmos_not_flow_logged", False), contextlib.redirect_stderr(stderr):
            sigmas, p = run(flow_wrap)
            self.assertIs(seen[-1], flow_wrap, "Forge passes its model_wrap as inner_model")
            self.assertTrue(torch.equal(sigmas, flow_list))
            self.assertEqual(p.extra_generation_params, {"Schedule type": "Flow Cosmos rho7"})
            self.assertEqual(stderr.getvalue(), "")

            sigmas, p = run(eps_wrap)
            self.assertIs(seen[-1], eps_wrap)
            self.assertTrue(torch.equal(sigmas, karras(12, lo, hi)))
            self.assertEqual(p.extra_generation_params, {"Schedule type": "Flow Cosmos rho7"})
            run(eps_wrap)
            self.assertEqual(stderr.getvalue().count("not a flow model"), 1)

            opts.rho = 4.5   # Settings → rho: Karras takes it, Flow Cosmos rho7 does not (no default_rho)
            sigmas, p = run(eps_wrap)
            self.assertTrue(torch.equal(sigmas, karras(12, lo, hi, rho=7.0)))
            self.assertNotIn("Schedule rho", p.extra_generation_params)
            opts.rho = 0
            opts.sigma_min = 0.1   # Settings → sigma min override
            sigmas, p = run(eps_wrap)
            self.assertTrue(torch.equal(sigmas, karras(12, 0.1, hi)))
            self.assertEqual(p.extra_generation_params["Schedule min sigma"], 0.1)
            sigmas, _p = run(flow_wrap)
            self.assertTrue(torch.equal(sigmas, flow_list), "the flow list does not use the model's range")
            opts.sigma_min = 0

            # D19: the accordion's values (the script's process sets them before sampling) — rho on both, σ̃ on flow
            st.set_active(st.ExtraSchedulerSettings(flow_cosmos_rho=5.0, flow_cosmos_sigma_max=240.0,
                                                    flow_cosmos_sigma_min=0.006))
            sigmas, p = run(flow_wrap)
            self.assertTrue(torch.equal(sigmas, sch._with_final_zero(sch._flow_cosmos_times(12, 5.0, 0.006, 240.0),
                                                                     "cpu")))
            self.assertEqual(p.extra_generation_params, {"Schedule type": "Flow Cosmos rho7"},
                             "Forge writes no rho for it (the script writes the Flow Cosmos keys)")
            sigmas, p = run(eps_wrap)
            self.assertTrue(torch.equal(sigmas, karras(12, lo, hi, rho=5.0)))
            self.assertNotIn("Schedule rho", p.extra_generation_params)
            opts.rho = 4.5   # still not Settings → rho
            sigmas, _p = run(eps_wrap)
            self.assertTrue(torch.equal(sigmas, karras(12, lo, hi, rho=5.0)))
            opts.rho = 0

    def test_the_fork_readme_labels_resolve_through_forges_own_lookups(self):
        self._register()   # Phi is hidden in setUp
        lookup = self.sd_samplers.get_sampler_and_scheduler
        cases = {"cosine": "Cosine", "cosine-exponential blend": "CosineExponential blend", "Laplace": "Laplace",
                 "laplace": "Laplace", "Karras Dynamic": "Karras Dynamic", "karras dynamic": "Karras Dynamic",
                 "custom": "custom"}
        for written, label in cases.items():
            with self.subTest(written=written):
                self.assertEqual(lookup("Euler", written), ("Euler", label))
                self.assertEqual(
                    self.sd_samplers.get_scheduler_from_infotext({"Sampler": "Euler", "Schedule type": written}), label)
        self.assertEqual(lookup("Euler", "phi"), ("Euler", "Automatic"), "hidden schedulers are not looked up")
        # Legacy "Sampler: <sampler> <scheduler>" infotext is where Forge reads Scheduler.aliases.
        self.assertEqual(lookup("Euler cosine-exponential blend", None), ("Euler", "CosineExponential blend"))
        self.assertEqual(self.sd_samplers.get_hr_sampler_and_scheduler(
            {"Sampler": "Euler", "Schedule type": "Karras", "Hires schedule type": "karras dynamic"}),
            ("Use same sampler", "Karras Dynamic"))

    def test_forges_get_sigmas_with_an_alias_and_a_rising_karras_dynamic(self):
        self._register()
        get_sigmas, opts = self._forge_get_sigmas()
        model_sigmas = torch.linspace(0.0291675, 14.614642, 1000)
        sampler = types.SimpleNamespace(config=types.SimpleNamespace(options={}),
                                        model_wrap=types.SimpleNamespace(sigmas=model_sigmas))
        lo, hi = model_sigmas[0].item(), model_sigmas[-1].item()
        p = types.SimpleNamespace(scheduler="cosine-exponential blend", hr_scheduler=None, is_hr_pass=False,
                                  extra_generation_params={}, sampler_noise_scheduler_override=None)
        self.assertTrue(torch.equal(get_sigmas(sampler, p, 12), sch.cosine_exponential_blend(12, lo, hi)))
        self.assertEqual(p.extra_generation_params["Schedule type"], "CosineExponential blend")
        # Settings → rho 3 (shared with Karras) reaches Karras Dynamic, whose schedule then rises: the
        # generation stops with a readable error instead of sampling it.
        opts.rho = 3.0
        p = types.SimpleNamespace(scheduler="Karras Dynamic", hr_scheduler=None, is_hr_pass=False,
                                  extra_generation_params={}, sampler_noise_scheduler_override=None)
        with self.assertRaises(sch.KarrasDynamicError) as ctx:
            get_sigmas(sampler, p, 20)
        self.assertIsInstance(ctx.exception, ValueError)
        self.assertIn("Raise rho to about 4 or more", str(ctx.exception))
        opts.rho = 4.0
        sigmas = get_sigmas(sampler, p, 20)
        self.assertTrue(bool((sigmas[1:] <= sigmas[:-1]).all()))
        self.assertEqual(p.extra_generation_params["Schedule rho"], 4.0)

    def _forge_get_sigmas(self):
        opts = types.SimpleNamespace(always_discard_next_to_last_sigma=False, sigma_min=0, sigma_max=0, rho=0,
                                     beta_dist_alpha=0.6, beta_dist_beta=0.6)
        namespace = {"opts": opts, "sd_schedulers": self.sd_schedulers, "torch": torch,
                     "devices": types.SimpleNamespace(cpu=torch.device("cpu"))}
        return support.forge_function("modules/sd_samplers_kdiffusion.py", "KDiffusionSampler.get_sigmas", namespace), opts


if __name__ == "__main__":
    unittest.main()
