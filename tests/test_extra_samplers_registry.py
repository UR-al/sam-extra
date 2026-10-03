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
    "ER SDE (Reverse-time)", "ER SDE (ODE)", "DPM++ 4M SDE", "Euler Dy CFG++", "Euler SMEA Dy CFG++",
]


def _k_sampling_stub(missing=()):
    """Every name the table requires, minus ``missing``."""
    module = types.ModuleType("k_diffusion.sampling")
    for spec in registry.SPECS:
        for name in spec.requires:
            if name not in missing:
                setattr(module, name, object())
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

    def __init__(self, *, with_add=True, preset=(), missing=(), base=_StubKDiffusionSampler):
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

    def install(self, case: unittest.TestCase):
        case.enterContext(fx.stub_modules({
            "modules": self.modules,
            "modules.sd_samplers": self.sd_samplers,
            "modules.sd_samplers_common": self.common,
            "modules.sd_samplers_kdiffusion": self.kdiffusion,
        }))
        case.enterContext(fx.installed_k_sampling(self.k_sampling))
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
        self.assertEqual(sorted(report.skipped_missing), ["ER SDE (ODE)", "ER SDE (Reverse-time)"])
        self.assertEqual(report.added, ["DPM++ 4M SDE", "Euler Dy CFG++", "Euler SMEA Dy CFG++"])
        self.assertEqual(len(logged), 2)
        self.assertTrue(all("sample_er_sde" in line for line in logged))

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
        for label in ("Euler Dy CFG++", "Euler SMEA Dy CFG++"):
            kwargs = samplers[label].initialize(_p())
            self.assertEqual(set(kwargs), {"base", "after_substep"})
            latent = torch.zeros(1)
            kwargs["after_substep"](latent)
            self.assertIs(host.stored[-1], latent)

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
        self.inner_model = types.SimpleNamespace(sigmas=torch.tensor([0.03, 14.6]))


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
    return modules


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
        }))
        self.enterContext(fx.installed_k_sampling(fx.forge_k_sampling()))
        self.sd_samplers = modules.sd_samplers
        self.kd_module = modules.sd_samplers_kdiffusion

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
