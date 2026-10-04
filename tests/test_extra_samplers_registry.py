"""Registration of the extra samplers in Forge's sampler list (``sam3ext.extra_samplers.registry``).

Two hosts:

* ``StubForgeTests`` — hand-written stand-ins of ``modules.sd_samplers`` / ``sd_samplers_common`` /
  ``sd_samplers_kdiffusion`` (always runs): the table, duplicate labels, reloads, missing helpers,
  the fallback without ``add_sampler``, the runtime keyword arguments and the CFG++ warning.
* ``ForgeCodeTests`` — the same flow on Forge Neo's own code (when Forge sits next to the
  extension): ``set_samplers``/``add_sampler`` (modules/sd_samplers.py), ``SamplerData`` and
  ``Sampler.__init__``/``Sampler.initialize`` (modules/sd_samplers_common.py),
  ``KDiffusionSampler.__init__`` and the ``sampler_extra_params`` table
  (modules/sd_samplers_kdiffusion.py), taken out of those files with ``ast`` and executed unchanged
  against stand-ins of what they import. This is where Forge's own ``initialize`` decides what the
  samplers receive (``s_noise``/``eta``/``s_churn``… and their ``Sigma …``/``Eta`` infotext).
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import inspect
import sys
import textwrap
import types
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

import torch


def _fixtures():
    name = "_extra_samplers_fixtures"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


fx = _fixtures()

from sam3ext.extra_samplers import params as sampler_params  # noqa: E402
from sam3ext.extra_samplers import registry  # noqa: E402

LABELS = [
    "ER SDE (Reverse-time)", "ER SDE (ODE)", "ER SDE (Tunable)", "DPM++ 4M SDE", "Euler Dy CFG++", "Euler SMEA Dy CFG++",
    "Euler Dy", "Euler SMEA Dy", "DPM++ 2M SDE Heun", "DPM++ 2M (flow ODE)", "DPM++ 2M Heun (flow ODE)",
    "DPM++ 3M (flow ODE)", "UniPC bh2", "CFG++ UD10 AB", "IPNDM", "IPNDM_V", "DEIS", "Restart (flow)",
]
BEFORE_0_33 = ("ER SDE (Reverse-time)", "ER SDE (ODE)", "DPM++ 4M SDE", "Euler Dy CFG++", "Euler SMEA Dy CFG++")
NEW_IN_0_33 = [label for label in LABELS if label not in BEFORE_0_33]
FLOW_ODE_LABELS = ("DPM++ 2M (flow ODE)", "DPM++ 2M Heun (flow ODE)", "DPM++ 3M (flow ODE)")
# v0.33.0 entries that take none of the ER SDE values and none of Forge's Eta
PLAIN_NEW = ("CFG++ UD10 AB", "IPNDM", "IPNDM_V", "DEIS", "Restart (flow)")


def _k_sampling_stub(missing=()):
    """Every name the table requires, minus ``missing``."""
    module = types.ModuleType("k_diffusion.sampling")
    for spec in registry.SPECS:
        for name in spec.requires:
            if name not in missing:
                setattr(module, name, object())
    return module


def _deis_stub(*, function=True):
    """``k_diffusion.deis`` stand-in (Forge's vendored DEIS coefficients; DEIS's forge_requires)."""
    module = types.ModuleType("k_diffusion.deis")
    if function:
        module.get_deis_coeff_list = lambda t_steps, max_order, N=10000, deis_mode="tab": [[] for _ in t_steps[:-1]]
    return module


def _sd_samplers_extra_stub(*, variant=True):
    """``modules.sd_samplers_extra`` stand-in: Forge's ``sample_unipc`` signature (with or without ``variant``)."""
    module = types.ModuleType("modules.sd_samplers_extra")
    if variant:
        def sample_unipc(model, x, sigmas, extra_args=None, callback=None, disable=False, variant="bh1"):
            return ("unipc", variant)
    else:
        def sample_unipc(model, x, sigmas, extra_args=None, callback=None, disable=False):
            return ("unipc", None)
    module.sample_unipc = sample_unipc
    return module


# ---------------------------------------------------------------------------
# Stand-in Forge
# ---------------------------------------------------------------------------

_SamplerDataTuple = namedtuple("SamplerData", ["name", "constructor", "aliases", "options"])


class _StubSamplerData(_SamplerDataTuple):
    pass


class _StubKDiffusionSampler:
    """Forge's constructor contract: ``(funcname, sd_model, options=None)`` and a name-keyed lookup."""

    sampler_extra_params = {"sample_euler": ["s_churn", "s_tmin", "s_tmax", "s_noise"]}

    def __init__(self, funcname, sd_model, options=None):
        self.funcname = funcname
        self.func = funcname
        self.sd_model = sd_model
        self.extra_params = self.sampler_extra_params.get(funcname, [])
        self.initialized_with = None

    def initialize(self, p):
        self.initialized_with = p
        return {"base": True}

    def sample(self, p, *args, **kwargs):
        return ("sample", args)

    def sample_img2img(self, p, *args, **kwargs):
        return ("sample_img2img", args)


def _stub_sd_samplers(with_add=True, preset=()):
    module = types.ModuleType("modules.sd_samplers")
    module.all_samplers = list(preset)
    module.all_samplers_map = {x.name: x for x in module.all_samplers}
    module.set_calls = 0

    def set_samplers():
        module.set_calls += 1
        module.samplers = module.all_samplers

    def add_sampler(sampler):
        module.add_calls.append(sampler.name)
        if sampler.name not in [x.name for x in module.all_samplers]:
            module.all_samplers.append(sampler)
            module.all_samplers_map = {x.name: x for x in module.all_samplers}
            set_samplers()

    @functools.cache
    def get_sampler_and_scheduler(sampler_name, scheduler_name):
        return sampler_name, scheduler_name

    module.set_samplers = set_samplers
    module.add_calls = []
    if with_add:
        module.add_sampler = add_sampler
    module.get_sampler_and_scheduler = get_sampler_and_scheduler
    return module


class _Host:
    """``modules`` + ``k_diffusion.sampling`` stand-ins installed for one test."""

    def __init__(self, *, with_add=True, preset=(), missing=(), base=_StubKDiffusionSampler, samplers_extra=True,
                 deis=True):
        self.sd_samplers = _stub_sd_samplers(with_add, preset)
        self.common = types.ModuleType("modules.sd_samplers_common")
        self.common.SamplerData = _StubSamplerData
        self.stored = []
        self.common.store_latent = self.stored.append
        self.kdiffusion = types.ModuleType("modules.sd_samplers_kdiffusion")
        self.kdiffusion.KDiffusionSampler = base
        self.modules = types.ModuleType("modules")
        self.modules.__path__ = []
        self.modules.sd_samplers = self.sd_samplers
        self.modules.sd_samplers_common = self.common
        self.modules.sd_samplers_kdiffusion = self.kdiffusion
        self.k_sampling = _k_sampling_stub(missing)
        # Forge's UniPC module (UniPC bh2's forge_requires): True = with ``variant``, "no-variant", or None
        self.samplers_extra = None
        if samplers_extra:
            self.samplers_extra = _sd_samplers_extra_stub(variant=samplers_extra is True)
            self.modules.sd_samplers_extra = self.samplers_extra
        # Forge's vendored k_diffusion.deis (DEIS's forge_requires): True, "no-function", or None (no module)
        self.deis = None if not deis else _deis_stub(function=deis is True)

    def install(self, case: unittest.TestCase):
        stubs = {
            "modules": self.modules,
            "modules.sd_samplers": self.sd_samplers,
            "modules.sd_samplers_common": self.common,
            "modules.sd_samplers_kdiffusion": self.kdiffusion,
        }
        # None in sys.modules makes the import fail (a Forge without the module)
        stubs["modules.sd_samplers_extra"] = self.samplers_extra
        case.enterContext(fx.stub_modules(stubs))
        case.enterContext(fx.installed_k_sampling(self.k_sampling))
        # None in sys.modules makes ``importlib.import_module("k_diffusion.deis")`` fail (a Forge without it)
        case.enterContext(fx.stub_modules({"k_diffusion.deis": self.deis}))
        if self.deis is not None:
            sys.modules["k_diffusion"].deis = self.deis
        return self


def _p(**kwargs):
    # Forge's StableDiffusionProcessing.__post_init__ defaults (processing.py: s_tmax ``or float("inf")``)
    defaults = dict(extra_generation_params={}, cfg_scale=4.0, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"),
                    s_noise=1.0, eta=None)
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


class _Reset(unittest.TestCase):
    def setUp(self):
        registry._LOGGED.clear()
        registry._CLASS_CACHE.clear()
        sampler_params.reset_active()
        self.addCleanup(registry._LOGGED.clear)
        self.addCleanup(registry._CLASS_CACHE.clear)
        self.addCleanup(sampler_params.reset_active)


class StubForgeTests(_Reset):
    def test_the_table(self):
        self.assertEqual([spec.label for spec in registry.SPECS], LABELS)
        table = {spec.label: spec for spec in registry.SPECS}
        self.assertEqual(table["ER SDE (Reverse-time)"].extra_params, ("s_noise",))
        self.assertEqual(table["ER SDE (ODE)"].extra_params, ())
        self.assertEqual(table["DPM++ 4M SDE"].extra_params, ("eta", "s_noise"))
        self.assertEqual(table["DPM++ 4M SDE"].options,
                         {"scheduler": "exponential", "discard_next_to_last_sigma": True, "brownian_noise": True})
        for label in ("Euler Dy CFG++", "Euler SMEA Dy CFG++"):
            self.assertEqual(table[label].extra_params, ("s_churn", "s_tmin", "s_tmax", "s_noise"))
            self.assertTrue(table[label].cfg_pp)
            self.assertEqual(table[label].options, {})
        self.assertEqual(table["ER SDE (Tunable)"].extra_params, ("s_noise",))
        self.assertEqual([spec.label for spec in registry.SPECS if spec.cfg_pp],
                         ["Euler Dy CFG++", "Euler SMEA Dy CFG++", "CFG++ UD10 AB"])
        for spec in registry.SPECS:
            with self.subTest(label=spec.label):
                parameters = inspect.signature(spec.func).parameters
                for name in spec.extra_params:
                    self.assertIn(name, parameters)   # Forge only passes names in the signature
                if spec.kind == "er_sde":
                    self.assertNotIn("eta", parameters)   # Forge would hand it its global ancestral eta
                self.assertEqual(len(set(spec.aliases)), len(spec.aliases))
        aliases = [alias for spec in registry.SPECS for alias in spec.aliases]
        self.assertEqual(len(aliases), len(set(aliases)))

    def test_the_v0_33_entries(self):
        table = {spec.label: spec for spec in registry.SPECS}
        expected = {
            # label: (aliases, options, extra_params, kind, cfg_pp, forge_requires)
            "Euler Dy": (("euler_dy", "k_euler_dy"), {}, ("s_churn", "s_tmin", "s_tmax", "s_noise"), "dy", False, ()),
            "Euler SMEA Dy": (("euler_smea_dy", "k_euler_smea_dy"), {}, ("s_churn", "s_tmin", "s_tmax", "s_noise"),
                              "dy", False, ()),
            "DPM++ 2M SDE Heun": (("dpmpp_2m_sde_heun", "k_dpmpp_2m_sde_heun"), {"brownian_noise": True},
                                  ("eta", "s_noise"), "", False, ()),
            "DPM++ 2M (flow ODE)": (("dpmpp_2m_flow_ode",), {}, (), "", False, ()),
            "DPM++ 2M Heun (flow ODE)": (("dpmpp_2m_heun_flow_ode",), {}, (), "", False, ()),
            "DPM++ 3M (flow ODE)": (("dpmpp_3m_flow_ode",), {"discard_next_to_last_sigma": True}, (), "", False, ()),
            "UniPC bh2": (("uni_pc_bh2",), {"discard_next_to_last_sigma": True}, (), "", False,
                          ("modules.sd_samplers_extra:sample_unipc(variant)",)),
            # batch 2
            "ER SDE (Tunable)": (("er_sde_tunable",), {}, ("s_noise",), "er_sde", False, ()),
            "CFG++ UD10 AB": (("cfgpp_ud10_ab",), {}, (), "", True, ()),
            "IPNDM": (("ipndm",), {}, (), "", False, ()),
            "IPNDM_V": (("ipndm_v",), {}, (), "", False, ()),
            "DEIS": (("deis",), {}, (), "", False, ("k_diffusion.deis:get_deis_coeff_list",)),
            "Restart (flow)": (("restart_flow",), {"second_order": True}, ("s_noise",), "", False, ()),
        }
        self.assertEqual(sorted(expected), sorted(NEW_IN_0_33))
        for label, (aliases, options, extra_params, kind, cfg_pp, forge_requires) in expected.items():
            with self.subTest(label=label):
                spec = table[label]
                self.assertEqual(spec.aliases, aliases)
                self.assertEqual(spec.options, options)
                self.assertEqual(spec.extra_params, extra_params)
                self.assertEqual(spec.kind, kind)
                self.assertIs(spec.cfg_pp, cfg_pp)
                self.assertEqual(spec.forge_requires, forge_requires)
                self.assertIn(label, [value for name, value in vars(registry).items() if name.startswith("LABEL_")])
        # Flow first: no new entry asks Forge's Automatic for Karras/Exponential (only the existing 4M SDE does).
        self.assertEqual([spec.label for spec in registry.SPECS if "scheduler" in spec.options], ["DPM++ 4M SDE"])
        # The flow ODE entries cannot receive Forge's global Eta (and so never write it to the infotext).
        for label in FLOW_ODE_LABELS:
            with self.subTest(label=label):
                parameters = inspect.signature(table[label].func).parameters
                self.assertNotIn("eta", parameters)
                self.assertNotIn("s_noise", parameters)
                self.assertNotIn("noise_sampler", parameters)
        heun = inspect.signature(table["DPM++ 2M SDE Heun"].func).parameters
        self.assertIn("eta", heun)
        self.assertIn("s_noise", heun)
        self.assertNotIn("solver_type", heun)   # fixed inside, never taken from Forge's options
        # none of the batch-2 entries can receive Forge's global Eta; the window is the ER SDE SDEs' only
        for label in ("ER SDE (Tunable)", *PLAIN_NEW):
            with self.subTest(label=label):
                self.assertNotIn("eta", inspect.signature(table[label].func).parameters)
        for label in LABELS:
            accepts = "er_sde_window" in inspect.signature(table[label].func).parameters
            self.assertEqual(accepts, label in ("ER SDE (Reverse-time)", "ER SDE (Tunable)"), label)
        self.assertNotIn("sam_extra_step_offset", inspect.signature(table["Restart (flow)"].func).parameters)

    def test_app_pinned_constants_are_module_level_literals(self):
        """The desktop app pins sam-extra constants by AST (a module-level ``NAME = <literal>``)."""
        expected = {
            "params.py": {
                "LABEL_ER_SDE_TUNABLE": "ER SDE (Tunable)",
                "ARG_NAMES": ("max_stage", "eta", "noise_window", "noise_start", "noise_end"),
                "DEFAULT_NOISE_WINDOW": False, "DEFAULT_NOISE_START": 0.2, "DEFAULT_NOISE_END": 0.8,
                "NOISE_PERCENT_MIN": 0.0, "NOISE_PERCENT_MAX": 1.0, "NOISE_PERCENT_STEP": 0.01,
                "KEY_NOISE_WINDOW": "ER SDE noise window", "NOISE_WINDOW_OFF": "off",
                "WINDOW_SKIPPED_STATUS": "noise window skipped (no percent_to_sigma)",
            },
            "cfgpp_ud10_ab.py": {"UD10_HISTORY_WEIGHT": 0.25, "UD10_ZERO_WEIGHT": 1.0, "UD10_UNCOND_HISTORY_WEIGHT": 0.1},
            "ipndm_deis.py": {"IPNDM_MAX_ORDER": 4, "DEIS_MAX_ORDER": 3, "DEIS_MODE": "tab"},
            "restart_flow.py": {"RESTART_S_MIN": 0.1, "RESTART_S_MAX": 2.0, "RESTART_MIN_STEPS": 20, "RESTART_TIMES": 1,
                                "RESTART_TIMES_LONG": 2, "RESTART_LONG_FROM": 36},
            "er_sde.py": {"ER_SDE": "ER-SDE"},
        }
        def literal(node):
            try:
                return ast.literal_eval(node)
            except ValueError:   # a module-level expression (not a literal): not pinnable
                return ValueError

        folder = Path(registry.__file__).parent
        for filename, constants in expected.items():
            tree = ast.parse((folder / filename).read_text(encoding="utf-8"))
            literals = {
                node.targets[0].id: literal(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            }
            for name, value in constants.items():
                with self.subTest(file=filename, name=name):
                    self.assertIn(name, literals)
                    self.assertEqual(literals[name], value)

    def test_label_literals_are_module_level_constants(self):
        """The desktop app pins the labels by AST: a module-level ``LABEL_* = "<literal>"`` in registry.py."""
        tree = ast.parse(Path(registry.__file__).read_text(encoding="utf-8"))
        literals = {
            node.targets[0].id: node.value.value for node in tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("LABEL_") and isinstance(node.value, ast.Constant)
        }
        self.assertEqual(sorted(literals.values()), sorted(label for label in LABELS if not label.startswith("ER SDE")))
        # … and the package exports every one of them
        package = sys.modules["sam3ext.extra_samplers"]
        for name, value in literals.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(package, name), value)
                self.assertEqual(getattr(registry, name), value)
                self.assertIn(name, package.__all__)
                self.assertIn(name, registry.__all__)
        self.assertEqual(package.LABELS, tuple(LABELS))

    def test_register_adds_every_entry(self):
        host = _Host().install(self)
        logged = []
        report = registry.register(log=logged.append)
        self.assertEqual(report.added, LABELS)
        self.assertEqual(host.sd_samplers.add_calls, LABELS)
        self.assertEqual([x.name for x in host.sd_samplers.all_samplers], LABELS)
        self.assertEqual(logged, [])
        for data, spec in zip(host.sd_samplers.all_samplers, registry.SPECS):
            self.assertIsInstance(data, _StubSamplerData)
            self.assertEqual(data.aliases, list(spec.aliases))
            self.assertEqual(data.options, spec.options)
            self.assertIsNot(data.options, spec.options)
            self.assertTrue(getattr(data.constructor, registry.OWNER_ATTR))
        self.assertIs(registry.last_report(), report)

    def test_constructed_sampler_uses_the_tables_extra_params(self):
        host = _Host().install(self)
        registry.register(log=lambda m: None)
        model = object()
        for data, spec in zip(host.sd_samplers.all_samplers, registry.SPECS):
            with self.subTest(label=spec.label):
                sampler = data.constructor(model)
                self.assertIsInstance(sampler, _StubKDiffusionSampler)
                self.assertIs(sampler.func, spec.func)
                self.assertIs(sampler.sd_model, model)
                # Forge's lookup by function name gives nothing for a function object …
                self.assertEqual(_StubKDiffusionSampler.sampler_extra_params.get(spec.func, []), [])
                # … so the subclass sets it.
                self.assertEqual(sampler.extra_params, list(spec.extra_params))

    def test_a_foreign_label_is_kept_and_logged_once(self):
        foreign = _StubSamplerData("Euler Dy CFG++", lambda model: "theirs", ["k_euler_dy_cfg_pp"], {})
        host = _Host(preset=[foreign]).install(self)
        logged = []
        first = registry.register(log=logged.append)
        second = registry.register(log=logged.append)
        self.assertEqual(first.skipped_foreign, ["Euler Dy CFG++"])
        self.assertEqual(second.skipped_foreign, ["Euler Dy CFG++"])
        self.assertEqual(len(logged), 1)
        self.assertIn("Euler Dy CFG++", logged[0])
        names = [x.name for x in host.sd_samplers.all_samplers]
        self.assertEqual(names.count("Euler Dy CFG++"), 1)
        self.assertIs(host.sd_samplers.all_samplers[0], foreign)
        self.assertEqual(sorted(second.replaced), sorted(label for label in LABELS if label != "Euler Dy CFG++"))

    def test_a_reload_replaces_our_own_entries_in_place(self):
        host = _Host().install(self)
        registry.register(log=lambda m: None)
        before = list(host.sd_samplers.all_samplers)
        report = registry.register(log=lambda m: None)
        after = host.sd_samplers.all_samplers
        self.assertEqual(report.replaced, LABELS)
        self.assertEqual(report.added, [])
        self.assertEqual([x.name for x in after], LABELS)
        for old, new in zip(before, after):
            self.assertIsNot(old, new)
        self.assertEqual(host.sd_samplers.all_samplers_map, {x.name: x for x in after})

    def test_missing_forge_helpers_skip_only_the_samplers_that_need_them(self):
        host = _Host(missing=("sample_er_sde",)).install(self)
        logged = []
        report = registry.register(log=logged.append)
        self.assertEqual(sorted(report.skipped_missing), ["ER SDE (ODE)", "ER SDE (Reverse-time)", "ER SDE (Tunable)"])
        self.assertEqual(report.added, [label for label in LABELS if not label.startswith("ER SDE")])
        self.assertEqual(len(logged), 3)
        self.assertTrue(all("sample_er_sde" in line for line in logged))

    def test_missing_dpmpp_sde_functions_skip_the_entries_built_on_them(self):
        host = _Host(missing=("sample_dpmpp_2m_sde",)).install(self)
        report = registry.register(log=lambda m: None)
        self.assertEqual(sorted(report.skipped_missing),
                         ["DPM++ 2M (flow ODE)", "DPM++ 2M Heun (flow ODE)", "DPM++ 2M SDE Heun"])
        self.assertIn("DPM++ 3M (flow ODE)", report.added)
        self.assertEqual(len(host.sd_samplers.all_samplers), len(LABELS) - 3)

    def test_unipc_bh2_needs_forges_sample_unipc_with_a_variant_parameter(self):
        for samplers_extra, reason in ((None, "modules.sd_samplers_extra.sample_unipc with the parameter 'variant'"),
                                       ("no-variant", "with the parameter 'variant'")):
            with self.subTest(samplers_extra=samplers_extra):
                registry._LOGGED.clear()
                _Host(samplers_extra=samplers_extra).install(self)   # the later host's stand-ins win
                logged = []
                first = registry.register(log=logged.append)
                second = registry.register(log=logged.append)
                self.assertEqual(first.skipped_missing,
                                 {"UniPC bh2": ["modules.sd_samplers_extra:sample_unipc(variant)"]})
                self.assertEqual(second.skipped_missing, first.skipped_missing)
                self.assertEqual(first.added, [label for label in LABELS if label != "UniPC bh2"])
                self.assertEqual(len(logged), 1)   # logged once
                self.assertIn('"UniPC bh2" not registered', logged[0])
                self.assertIn(reason, logged[0])

    def test_deis_needs_forges_vendored_deis_module(self):
        for deis, reason in ((None, "k_diffusion.deis.get_deis_coeff_list"),
                             ("no-function", "k_diffusion.deis.get_deis_coeff_list")):
            with self.subTest(deis=deis):
                registry._LOGGED.clear()
                _Host(deis=deis).install(self)   # the later host's stand-ins win
                logged = []
                first = registry.register(log=logged.append)
                second = registry.register(log=logged.append)
                self.assertEqual(first.skipped_missing, {"DEIS": ["k_diffusion.deis:get_deis_coeff_list"]})
                self.assertEqual(second.skipped_missing, first.skipped_missing)
                self.assertEqual(first.added, [label for label in LABELS if label != "DEIS"])
                self.assertEqual(len(logged), 1)
                self.assertIn('"DEIS" not registered', logged[0])
                self.assertIn(reason, logged[0])

    def test_missing_multistep_helpers_skip_the_new_entries_that_need_them(self):
        host = _Host(missing=("linear_multistep_coeff", "default_noise_sampler")).install(self)
        report = registry.register(log=lambda m: None)
        # UD10's AB2 coefficients; the Dy samplers and Restart (flow) draw from Forge's default noise sampler
        self.assertEqual(sorted(report.skipped_missing),
                         sorted(["CFG++ UD10 AB", "Restart (flow)", "Euler Dy CFG++", "Euler SMEA Dy CFG++",
                                 "Euler Dy", "Euler SMEA Dy"]))
        self.assertIn("IPNDM", report.added)
        self.assertEqual(len(host.sd_samplers.all_samplers), len(LABELS) - 6)

    def test_forge_requirement_strings(self):
        parse = registry._parse_forge_requirement
        self.assertEqual(parse("modules.sd_samplers_extra:sample_unipc(variant)"),
                         ("modules.sd_samplers_extra", "sample_unipc", ("variant",)))
        self.assertEqual(parse("k_diffusion.deis:get_deis_coeff_list"), ("k_diffusion.deis", "get_deis_coeff_list", ()))
        self.assertEqual(parse("m:f( a, b )"), ("m", "f", ("a", "b")))
        module = types.ModuleType("_sam_extra_requirement_probe")
        module.f = lambda a, b=1: None
        module.not_callable = 3
        with fx.stub_modules({"_sam_extra_requirement_probe": module}):
            self.assertTrue(registry._forge_requirement_met("_sam_extra_requirement_probe:f"))
            self.assertTrue(registry._forge_requirement_met("_sam_extra_requirement_probe:f(a, b)"))
            self.assertFalse(registry._forge_requirement_met("_sam_extra_requirement_probe:f(variant)"))
            self.assertFalse(registry._forge_requirement_met("_sam_extra_requirement_probe:g"))
            self.assertTrue(registry._forge_requirement_met("_sam_extra_requirement_probe:not_callable"))
            self.assertFalse(registry._forge_requirement_met("_sam_extra_requirement_probe:not_callable(x)"))
        self.assertFalse(registry._forge_requirement_met("_sam_extra_no_such_module_:f"))
        self.assertEqual(registry._describe_forge_requirement("m:f(a, b)"), "m.f with the parameters 'a', 'b'")
        self.assertEqual(registry._describe_forge_requirement("m:f"), "m.f")

    def test_forge_without_add_sampler_gets_the_same_steps(self):
        host = _Host(with_add=False).install(self)
        report = registry.register(log=lambda m: None)
        self.assertEqual(report.added, LABELS)
        self.assertEqual([x.name for x in host.sd_samplers.all_samplers], LABELS)
        self.assertEqual(host.sd_samplers.all_samplers_map.keys(), set(LABELS))
        self.assertGreaterEqual(host.sd_samplers.set_calls, 1)

    def test_the_sampler_lookup_cache_is_cleared(self):
        host = _Host().install(self)
        host.sd_samplers.get_sampler_and_scheduler("Euler Dy CFG++", "Automatic")
        self.assertEqual(host.sd_samplers.get_sampler_and_scheduler.cache_info().currsize, 1)
        registry.register(log=lambda m: None)
        self.assertEqual(host.sd_samplers.get_sampler_and_scheduler.cache_info().currsize, 0)

    def test_without_forge_nothing_is_registered(self):
        self.enterContext(fx.stub_modules({"modules": types.ModuleType("modules")}))
        logged = []
        report = registry.register(log=logged.append)
        self.assertIsNotNone(report.error)
        self.assertEqual(report.active, [])
        self.assertEqual(len(logged), 1)

    def test_runtime_keyword_arguments_and_infotext(self):
        host = _Host().install(self)
        registry.register(log=lambda m: None)
        samplers = {data.name: data.constructor(object()) for data in host.sd_samplers.all_samplers}

        p = _p()
        sampler_params.apply_to(p, sampler_params.ErSdeSettings(max_stage=2, eta=0.5))
        kwargs = samplers["ER SDE (Reverse-time)"].initialize(p)
        self.assertEqual(kwargs, {"base": True, "max_stage": 2, "er_sde_eta": 0.5})
        self.assertEqual(p.extra_generation_params, {"ER SDE max stage": 2, "ER SDE eta": 0.5})

        p_ode = _p()
        sampler_params.apply_to(p_ode, sampler_params.ErSdeSettings(max_stage=1, eta=0.5))
        self.assertEqual(samplers["ER SDE (ODE)"].initialize(p_ode), {"base": True, "max_stage": 1})
        self.assertEqual(p_ode.extra_generation_params, {"ER SDE max stage": 1})

        p_default = _p()
        sampler_params.apply_to(p_default, sampler_params.ErSdeSettings())
        samplers["ER SDE (Reverse-time)"].initialize(p_default)
        self.assertEqual(p_default.extra_generation_params, {})

        self.assertEqual(samplers["DPM++ 4M SDE"].initialize(_p()), {"base": True})
        for label in ("Euler Dy CFG++", "Euler SMEA Dy CFG++", "Euler Dy", "Euler SMEA Dy"):
            kwargs = samplers[label].initialize(_p())
            self.assertEqual(set(kwargs), {"base", "after_substep"})
            latent = torch.zeros(1)
            kwargs["after_substep"](latent)
            self.assertIs(host.stored[-1], latent)
        for label in ("DPM++ 2M SDE Heun", *FLOW_ODE_LABELS, "UniPC bh2", *PLAIN_NEW):
            with self.subTest(label=label):
                p = _p()
                sampler_params.apply_to(p, sampler_params.ErSdeSettings(max_stage=1, eta=0.5, noise_window=True))
                self.assertEqual(samplers[label].initialize(p), {"base": True})
                self.assertEqual(p.extra_generation_params, {})   # the ER SDE values are not theirs

        p_tunable = _p()
        sampler_params.apply_to(p_tunable, sampler_params.ErSdeSettings(max_stage=2, eta=0.5))
        self.assertEqual(samplers["ER SDE (Tunable)"].initialize(p_tunable),
                         {"base": True, "max_stage": 2, "er_sde_eta": 0.5})
        self.assertEqual(p_tunable.extra_generation_params, {"ER SDE max stage": 2, "ER SDE eta": 0.5})

    def test_the_noise_window_reaches_reverse_time_and_tunable(self):
        host = _Host().install(self)
        logged = []
        self.enterContext(mock.patch.object(registry, "_log", logged.append))
        registry.register(log=lambda m: None)
        samplers = {data.name: data.constructor(object()) for data in host.sd_samplers.all_samplers}
        for sampler in samplers.values():   # Forge's KDiffusionSampler has model_wrap.predictor
            sampler.model_wrap = types.SimpleNamespace(predictor=fx.FlowSampling(3.0))
        window = sampler_params.ErSdeSettings(max_stage=3, eta=1.0, noise_window=True, noise_start=0.1, noise_end=0.9)
        for label in ("ER SDE (Reverse-time)", "ER SDE (Tunable)"):
            with self.subTest(label=label):
                p = _p()
                sampler_params.apply_to(p, window)
                self.assertEqual(samplers[label].initialize(p),
                                 {"base": True, "max_stage": 3, "er_sde_eta": 1.0, "er_sde_window": (0.1, 0.9)})
                self.assertEqual(p.extra_generation_params, {"ER SDE noise window": "0.1-0.9"})
        p = _p()
        sampler_params.apply_to(p, window)
        self.assertEqual(samplers["ER SDE (ODE)"].initialize(p), {"base": True, "max_stage": 3})
        self.assertEqual(p.extra_generation_params, {})
        # η 0 is the ODE: no window, no key
        p = _p()
        sampler_params.apply_to(p, sampler_params.ErSdeSettings(3, 0.0, True, 0.1, 0.9))
        self.assertEqual(samplers["ER SDE (Tunable)"].initialize(p), {"base": True, "max_stage": 3, "er_sde_eta": 0.0})
        self.assertEqual(p.extra_generation_params, {"ER SDE eta": 0.0})
        # the box off: nothing
        p = _p()
        sampler_params.apply_to(p, sampler_params.ErSdeSettings(3, 1.0, False, 0.1, 0.9))
        self.assertEqual(samplers["ER SDE (Tunable)"].initialize(p), {"base": True, "max_stage": 3, "er_sde_eta": 1.0})
        self.assertEqual(p.extra_generation_params, {})
        self.assertEqual(logged, [])

    def test_a_window_the_model_cannot_place_is_dropped_with_a_status(self):
        host = _Host().install(self)
        logged = []
        self.enterContext(mock.patch.object(registry, "_log", logged.append))
        registry.register(log=lambda m: None)
        samplers = {data.name: data.constructor(object()) for data in host.sd_samplers.all_samplers}
        settings = sampler_params.ErSdeSettings(3, 1.5, True, 0.2, 0.8)
        for predictor in (None, fx.EpsSampling()):   # no model_wrap at all / a predictor without percent_to_sigma
            for label in ("ER SDE (Tunable)", "ER SDE (Reverse-time)"):
                with self.subTest(predictor=predictor, label=label):
                    if predictor is not None:
                        samplers[label].model_wrap = types.SimpleNamespace(predictor=predictor)
                    p = _p()
                    sampler_params.apply_to(p, settings)
                    self.assertEqual(samplers[label].initialize(p), {"base": True, "max_stage": 3, "er_sde_eta": 1.5})
                    self.assertEqual(p.extra_generation_params,
                                     {"ER SDE eta": 1.5, "Extra Samplers status": "noise window skipped (no percent_to_sigma)"})
        self.assertEqual(len(logged), 2)   # once per sampler
        self.assertTrue(all("percent_to_sigma" in line for line in logged))

    def test_the_status_keeps_dy_reasons_and_adds_the_window_part(self):
        """``Extra Samplers status`` parts joined by "; " in first-seen order; the Dy text is unchanged."""
        from sam3ext.extra_samplers import substep_guard

        p = _p()
        substep_guard.record_status(p, [substep_guard.REASON_SPECTRUM])
        self.assertEqual(p.extra_generation_params["Extra Samplers status"], "dy sub-steps skipped (Spectrum)")
        substep_guard.record_status_part(p, "noise_window", sampler_params.WINDOW_SKIPPED_STATUS)
        substep_guard.record_status(p, [substep_guard.REASON_PID])
        substep_guard.record_status_part(p, "noise_window", sampler_params.WINDOW_SKIPPED_STATUS)
        self.assertEqual(p.extra_generation_params["Extra Samplers status"],
                         "dy sub-steps skipped (Spectrum + PiD lq_latent); noise window skipped (no percent_to_sigma)")
        q = _p()
        substep_guard.record_status_part(q, "noise_window", sampler_params.WINDOW_SKIPPED_STATUS)
        substep_guard.record_status(q, [substep_guard.REASON_WAN])
        self.assertEqual(q.extra_generation_params["Extra Samplers status"],
                         "noise window skipped (no percent_to_sigma); dy sub-steps skipped (Wan I2V concat_latent)")

    def test_a_request_without_the_script_uses_the_last_requests_values(self):
        host = _Host().install(self)
        registry.register(log=lambda m: None)
        sampler = host.sd_samplers.all_samplers[0].constructor(object())
        self.assertEqual(sampler.initialize(_p())["max_stage"], 3)   # nothing seen yet: defaults
        sampler_params.apply_to(_p(), sampler_params.ErSdeSettings(max_stage=1, eta=2.0))
        inner = _p()   # e.g. ADetailer's img2img, which filters the always-on scripts
        kwargs = sampler.initialize(inner)
        self.assertEqual((kwargs["max_stage"], kwargs["er_sde_eta"]), (1, 2.0))
        self.assertEqual(inner.extra_generation_params, {"ER SDE max stage": 1, "ER SDE eta": 2.0})

    def test_cfg_pp_samplers_warn_above_cfg_two_like_forges(self):
        host = _Host().install(self)
        registry.register(log=lambda m: None)
        samplers = {data.name: data.constructor(object()) for data in host.sd_samplers.all_samplers}
        with self.assertLogs(level="WARNING") as logs:
            self.assertEqual(samplers["Euler Dy CFG++"].sample(_p(cfg_scale=4.0), 1), ("sample", (1,)))
            samplers["Euler SMEA Dy CFG++"].sample_img2img(_p(cfg_scale=2.5), 2)
        self.assertEqual(logs.output, ["WARNING:root:CFG between 1.0 ~ 2.0 is recommended when using CFG++ samplers"] * 2)
        with self.assertNoLogs(level="WARNING"):
            samplers["Euler Dy CFG++"].sample(_p(cfg_scale=2.0))
            samplers["DPM++ 4M SDE"].sample(_p(cfg_scale=7.0))
            samplers["CFG++ UD10 AB"].sample(_p(cfg_scale=2.0))
            for label in NEW_IN_0_33:   # none of them but CFG++ UD10 AB is a CFG++ sampler
                if label == "CFG++ UD10 AB":
                    continue
                samplers[label].sample(_p(cfg_scale=7.0))
                samplers[label].sample_img2img(_p(cfg_scale=7.0))
        with self.assertLogs(level="WARNING") as logs:   # ComfyUI recommends CFG 2 for it; Forge warns above
            samplers["CFG++ UD10 AB"].sample(_p(cfg_scale=4.5))
            samplers["CFG++ UD10 AB"].sample_img2img(_p(cfg_scale=2.1))
        self.assertEqual(len(logs.output), 2)


# ---------------------------------------------------------------------------
# Forge's own code
# ---------------------------------------------------------------------------

_SD_SAMPLERS = "modules/sd_samplers.py"
_COMMON = "modules/sd_samplers_common.py"
_KDIFFUSION = "modules/sd_samplers_kdiffusion.py"


def _forge_assignment(rel_path: str, name: str):
    tree = ast.parse(fx.forge_source(rel_path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise LookupError(name)


def _forge_class(name: str, bases: str, methods, namespace: dict, extra: str = ""):
    """``class name(bases)`` whose methods are Forge's source (so zero-argument ``super()`` works)."""
    body = [textwrap.indent(fx.forge_definition(path, method, cls=cls), "    ") for path, cls, method in methods]
    source = f"class {name}({bases}):\n" + "\n\n".join(body) + ("\n\n" + textwrap.indent(extra, "    ") if extra else "") + "\n"
    exec(compile(source, f"<forge {name}>", "exec"), namespace)
    return namespace[name]


class _CFGDenoiserKDiffusion:
    def __init__(self, sampler):
        self.sampler = sampler
        # Forge's ForgeScheduleLinker: the model's predictor (here Anima's flow parameterisation)
        self.inner_model = types.SimpleNamespace(sigmas=torch.tensor([0.03, 14.6]), predictor=fx.FlowSampling(3.0))


def _forge_host(opts):
    """Forge's sampler plumbing, executed from Forge's files against the given ``opts``."""
    fx.require_forge()
    namespace = {
        "namedtuple": namedtuple,
        "inspect": inspect,
        "opts": opts,
        "k_diffusion": types.SimpleNamespace(sampling=types.SimpleNamespace()),
        "TorchHijack": lambda p: ("hijack", p),
        "CFGDenoiserKDiffusion": _CFGDenoiserKDiffusion,
        "sampler_extra_params": ast.literal_eval(_forge_assignment(_KDIFFUSION, "sampler_extra_params")),
    }
    namespace["SamplerDataTuple"] = eval(ast.unparse(_forge_assignment(_COMMON, "SamplerDataTuple")), namespace)
    sampler_data = _forge_class("SamplerData", "SamplerDataTuple", [(_COMMON, "SamplerData", "total_steps")], namespace)
    sampler = _forge_class("Sampler", "object", [(_COMMON, "Sampler", "__init__"), (_COMMON, "Sampler", "initialize")],
                           namespace)
    kdiffusion = _forge_class(
        "KDiffusionSampler", "Sampler", [(_KDIFFUSION, "KDiffusionSampler", "__init__")], namespace,
        extra="def sample(self, p, *args, **kwargs):\n    return 'sample'\n",
    )

    sd_samplers = types.ModuleType("modules.sd_samplers")
    sd_samplers.__dict__.update({
        "shared": types.SimpleNamespace(opts=opts),
        "all_samplers": [], "all_samplers_map": {}, "samplers": [], "samplers_for_img2img": [],
        "samplers_map": {}, "samplers_hidden": {},
    })
    for function in ("set_samplers", "add_sampler", "visible_sampler_names"):
        exec(compile(fx.forge_definition(_SD_SAMPLERS, function), _SD_SAMPLERS, "exec"), sd_samplers.__dict__)
    common = types.ModuleType("modules.sd_samplers_common")
    common.SamplerData = sampler_data
    common.store_latent = lambda latent: None
    kd_module = types.ModuleType("modules.sd_samplers_kdiffusion")
    kd_module.KDiffusionSampler = kdiffusion
    kd_module.sampler_extra_params = namespace["sampler_extra_params"]
    modules = types.ModuleType("modules")
    modules.__path__ = []
    modules.sd_samplers, modules.sd_samplers_common, modules.sd_samplers_kdiffusion = sd_samplers, common, kd_module
    modules.sd_samplers_extra = fx.forge_sd_samplers_extra()   # Forge's own file (UniPC bh2's forge_requires)
    return modules


def _forge_sampler_table() -> dict:
    """Forge's ``samplers_k_diffusion`` (modules/sd_samplers_kdiffusion.py) as ``{label: (function, aliases,
    options)}``, read with ``ast`` (a label written as a conditional expression is taken from its else branch)."""
    table = {}
    for entry in _forge_assignment(_KDIFFUSION, "samplers_k_diffusion").elts:
        label_node, function_node, aliases_node, options_node = entry.elts
        if isinstance(label_node, ast.IfExp):
            label_node = label_node.orelse
        label = ast.literal_eval(label_node)
        function = ast.unparse(function_node) if not isinstance(function_node, ast.Constant) else function_node.value
        table[label] = (function, ast.literal_eval(aliases_node), ast.literal_eval(options_node))
    return table


class ForgeCodeTests(_Reset):
    def setUp(self):
        super().setUp()
        self.opts = types.SimpleNamespace(hide_samplers=["Euler SMEA Dy CFG++"], eta_ancestral=1.0,
                                          s_churn=0.0, s_tmin=0.0, s_tmax=0.0, s_noise=1.0)
        modules = _forge_host(self.opts)
        self.enterContext(fx.stub_modules({
            "modules": modules, "modules.sd_samplers": modules.sd_samplers,
            "modules.sd_samplers_common": modules.sd_samplers_common,
            "modules.sd_samplers_kdiffusion": modules.sd_samplers_kdiffusion,
            "modules.sd_samplers_extra": modules.sd_samplers_extra,
        }))
        self.enterContext(fx.installed_k_sampling(fx.forge_k_sampling()))
        self.enterContext(fx.installed_k_diffusion_deis(fx.forge_deis()))   # Forge's vendored DEIS (DEIS's forge_requires)
        self.sd_samplers = modules.sd_samplers
        self.kd_module = modules.sd_samplers_kdiffusion
        self.sd_samplers_extra = modules.sd_samplers_extra

    def _registered(self):
        registry.register(log=lambda m: None)
        return {data.name: data.constructor(object()) for data in self.sd_samplers.all_samplers}

    def test_forge_keys_extra_params_by_function_name(self):
        table = self.kd_module.sampler_extra_params
        self.assertTrue(table)
        self.assertTrue(all(isinstance(key, str) for key in table))
        init = fx.forge_definition(_KDIFFUSION, "__init__", cls="KDiffusionSampler")
        self.assertIn("self.extra_params = sampler_extra_params.get(funcname, [])", init)

    def test_forges_add_sampler_and_set_samplers_take_the_entries(self):
        registry.register(log=lambda m: None)
        names = [x.name for x in self.sd_samplers.all_samplers]
        self.assertEqual(names, LABELS)
        self.assertEqual(set(self.sd_samplers.all_samplers_map), set(LABELS))
        for spec in registry.SPECS:
            self.assertEqual(self.sd_samplers.samplers_map[spec.label.lower()], spec.label)
            for alias in spec.aliases:
                self.assertEqual(self.sd_samplers.samplers_map[alias.lower()], spec.label)
        self.assertNotIn("Euler SMEA Dy CFG++", self.sd_samplers.visible_sampler_names())   # Forge's hide_samplers
        self.assertEqual(self.sd_samplers.all_samplers[0].total_steps(10), 10)

    def test_forges_initialize_passes_what_each_sampler_takes(self):
        samplers = self._registered()

        p = _p()
        sampler_params.apply_to(p, sampler_params.ErSdeSettings(max_stage=2, eta=0.35))
        kwargs = samplers["ER SDE (Reverse-time)"].initialize(p)
        self.assertEqual(kwargs, {"s_noise": 1.0, "max_stage": 2, "er_sde_eta": 0.35})
        self.assertEqual(p.extra_generation_params, {"ER SDE max stage": 2, "ER SDE eta": 0.35})

        p = _p()
        sampler_params.apply_to(p, sampler_params.ErSdeSettings())
        self.assertEqual(samplers["ER SDE (ODE)"].initialize(p), {"max_stage": 3})
        self.assertEqual(p.extra_generation_params, {})

        p = _p(eta=0.6)
        self.assertEqual(samplers["DPM++ 4M SDE"].initialize(p), {"eta": 0.6, "s_noise": 1.0})
        self.assertEqual(p.extra_generation_params, {"Eta": 0.6})

        p = _p()
        kwargs = samplers["Euler Dy CFG++"].initialize(p)
        self.assertEqual({k: v for k, v in kwargs.items() if k != "after_substep"},
                         {"s_churn": 0.0, "s_tmin": 0.0, "s_tmax": float("inf"), "s_noise": 1.0})
        self.assertIn("after_substep", kwargs)

        for label in ("Euler Dy", "Euler SMEA Dy"):
            with self.subTest(label=label):
                p = _p()
                kwargs = samplers[label].initialize(p)
                self.assertEqual({k: v for k, v in kwargs.items() if k != "after_substep"},
                                 {"s_churn": 0.0, "s_tmin": 0.0, "s_tmax": float("inf"), "s_noise": 1.0})
                self.assertIn("after_substep", kwargs)
                self.assertEqual(p.extra_generation_params, {})

        p = _p(eta=0.6)   # Forge's global Eta reaches the Heun SDE entry like Forge's DPM++ 2M SDE …
        self.assertEqual(samplers["DPM++ 2M SDE Heun"].initialize(p), {"eta": 0.6, "s_noise": 1.0})
        self.assertEqual(p.extra_generation_params, {"Eta": 0.6})
        for label in (*FLOW_ODE_LABELS, "UniPC bh2", "CFG++ UD10 AB", "IPNDM", "IPNDM_V", "DEIS"):   # … no Eta key
            with self.subTest(label=label):
                p = _p(eta=0.6)
                self.assertEqual(samplers[label].initialize(p), {})
                self.assertEqual(p.extra_generation_params, {})

        p = _p(eta=0.6)   # Restart (flow): Forge's s_noise only
        self.assertEqual(samplers["Restart (flow)"].initialize(p), {"s_noise": 1.0})
        self.assertEqual(p.extra_generation_params, {})

        p = _p()   # ER SDE (Tunable): Forge's s_noise + the accordion's values, the window placed by the model's predictor
        sampler_params.apply_to(p, sampler_params.ErSdeSettings(2, 0.35, True, 0.2, 0.8))
        self.assertEqual(samplers["ER SDE (Tunable)"].initialize(p),
                         {"s_noise": 1.0, "max_stage": 2, "er_sde_eta": 0.35, "er_sde_window": (0.2, 0.8)})
        self.assertEqual(p.extra_generation_params,
                         {"ER SDE max stage": 2, "ER SDE eta": 0.35, "ER SDE noise window": "0.2-0.8"})
        self.assertIs(samplers["ER SDE (Tunable)"].model_wrap.predictor, samplers["ER SDE (Tunable)"].model_wrap_cfg.inner_model.predictor)

    def test_the_options_match_forges_own_entries(self):
        """Forge's table (modules/sd_samplers_kdiffusion.py): the 3M flow ODE keeps "DPM++ 3M SDE"'s discard flag,
        UniPC bh2 has "UniPC"'s options, the Heun SDE "DPM++ 2M SDE"'s without its scheduler hint; the 2M ODE
        entries discard nothing (like Forge's 2M SDE)."""
        forge = _forge_sampler_table()
        table = {spec.label: spec for spec in registry.SPECS}
        self.assertEqual(forge["DPM++ 3M SDE"][0], "sample_dpmpp_3m_sde")
        self.assertEqual(table["DPM++ 3M (flow ODE)"].options["discard_next_to_last_sigma"],
                         forge["DPM++ 3M SDE"][2]["discard_next_to_last_sigma"])
        self.assertEqual(forge["UniPC"][0], "sd_samplers_extra.sample_unipc")
        self.assertEqual(table["UniPC bh2"].options, forge["UniPC"][2])
        forge_2m_sde = dict(forge["DPM++ 2M SDE"][2])
        self.assertEqual(forge_2m_sde.pop("scheduler"), "exponential")
        self.assertEqual(table["DPM++ 2M SDE Heun"].options, forge_2m_sde)
        for label in ("DPM++ 2M (flow ODE)", "DPM++ 2M Heun (flow ODE)"):
            self.assertNotIn("discard_next_to_last_sigma", table[label].options)
            self.assertNotIn("discard_next_to_last_sigma", forge["DPM++ 2M SDE"][2])
        # Forge's own sample_unipc takes the variant UniPC bh2 passes
        self.assertIn("variant", inspect.signature(self.sd_samplers_extra.sample_unipc).parameters)
        # Restart (flow) runs Heun steps: Forge's "Heun" options, not Forge's "Restart" ones (its karras hint)
        self.assertEqual(table["Restart (flow)"].options, forge["Heun"][2])
        self.assertEqual(forge["Restart"][2].get("scheduler"), "karras")
        self.assertNotIn("scheduler", table["Restart (flow)"].options)
        # No new label or alias collides with Forge's built-in entries
        forge_names = {name.lower() for label, (_f, aliases, _o) in forge.items() for name in (label, *aliases)}
        for label in NEW_IN_0_33:
            spec = table[label]
            with self.subTest(label=label):
                self.assertFalse({name.lower() for name in (spec.label, *spec.aliases)} & forge_names)

    def test_our_eta_is_written_when_the_infotext_carries_forges_eta(self):
        """Forge writes its ancestral η as ``Eta`` for a pass whose sampler takes ``eta`` (here a DPM++ 4M
        SDE hires pass); ``paste_eta`` reads ``Eta`` as the Reverse-time η when ``ER SDE eta`` is missing
        (aoleg compatibility) — so the Reverse-time pass writes ``ER SDE eta`` even at its default 1.0."""
        samplers = self._registered()
        p = _p(eta=0.6)
        sampler_params.apply_to(p, sampler_params.ErSdeSettings())
        samplers["ER SDE (Reverse-time)"].initialize(p)
        samplers["DPM++ 4M SDE"].initialize(p)
        self.assertEqual(p.extra_generation_params, {"ER SDE eta": 1.0, "Eta": 0.6})
        pasted = {key: str(value) for key, value in p.extra_generation_params.items()}
        pasted.update({"Sampler": "ER SDE (Reverse-time)", "Hires sampler": "DPM++ 4M SDE"})
        self.assertEqual(sampler_params.paste_eta(pasted), 1.0)

        self.opts.eta_ancestral = 0.8   # Forge's setting instead of p.eta
        p = _p()
        sampler_params.apply_to(p, sampler_params.ErSdeSettings())
        samplers["ER SDE (Reverse-time)"].initialize(p)
        self.assertEqual(p.extra_generation_params, {"ER SDE eta": 1.0})

        p = _p(eta=0.6)                 # the ODE has no η
        sampler_params.apply_to(p, sampler_params.ErSdeSettings())
        samplers["ER SDE (ODE)"].initialize(p)
        self.assertEqual(p.extra_generation_params, {})

    def test_forges_sampler_parameter_settings_reach_the_samplers_and_the_infotext(self):
        self.opts.s_noise = 0.97
        self.opts.s_churn = 0.5
        self.opts.s_tmax = 10.0
        samplers = self._registered()

        for label in ("ER SDE (Tunable)", "Restart (flow)"):
            with self.subTest(label=label):
                p = _p()
                self.assertEqual(samplers[label].initialize(p)["s_noise"], 0.97)
                self.assertEqual(p.extra_generation_params, {"Sigma noise": 0.97})

        p = _p()
        self.assertEqual(samplers["ER SDE (Reverse-time)"].initialize(p)["s_noise"], 0.97)
        self.assertEqual(p.extra_generation_params, {"Sigma noise": 0.97})

        p = _p()
        kwargs = samplers["Euler SMEA Dy CFG++"].initialize(p)
        self.assertEqual((kwargs["s_churn"], kwargs["s_tmax"], kwargs["s_noise"]), (0.5, 10.0, 0.97))
        self.assertEqual(p.extra_generation_params, {"Sigma churn": 0.5, "Sigma tmax": 10.0, "Sigma noise": 0.97})

        p = _p()
        self.assertNotIn("s_noise", samplers["ER SDE (ODE)"].initialize(p))   # the ODE draws no noise
        self.assertEqual(p.extra_generation_params, {})


if __name__ == "__main__":
    unittest.main()
