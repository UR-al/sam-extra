"""Family order of Forge's Sampling method / Schedule type lists (sam3ext/list_order.py, scripts/list_order.py).

* ``new_order`` — the user's order (decided 2026-10-04) on the lists of the development PC's Forge (Forge
  2.29.2 + this extension + sd-forge-res4lyf 3a2e48d + sd_forge_neo_extra_samplers), entries that are not
  installed, the prefix / "(RES4LYF)" / rest rules, the kept first entries, idempotence.
* Every label this extension registers and every Forge built-in (read from the checkout) is placed in the
  table on purpose, so a new sampler or scheduler cannot fall into "the rest" unnoticed.
* ``apply`` on Forge's own code (``set_samplers`` / ``add_sampler`` / ``get_sampler_and_scheduler`` …
  executed from the checkout, this extension's real ``register_schedulers``): what Forge reads afterwards —
  the XYZ "Sampler" / "Hires sampler" / "Schedule type" choice lambdas (scripts/xyz_grid.py), the
  txt2img/img2img dropdowns (modules/processing_scripts/sampler.py), the Hires dropdowns (modules/ui.py) and
  ``/sdapi/v1/samplers`` / ``/sdapi/v1/schedulers`` (modules/api/api.py) — follows the new order, and every
  map lookup, default and legacy "Sampler: <sampler> <scheduler>" parse is what it was.
* The script registers only an ``on_before_ui`` callback, and its file sorts before ``negpip.py``.

The whole load (all scripts, before_ui, Reload UI) is checked by tests/test_integration_load_order.py.
"""

from __future__ import annotations

import ast
import collections
import functools
import io
import os
import sys
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import list_order as lo  # noqa: E402
from tests import _extra_samplers_fixtures as esf  # noqa: E402

# The user's order (labels as Forge shows them; "(RES4LYF)" samplers as one block in RES4LYF's own order).
EXPECTED_SAMPLERS = (
    "DPM++ 2M", "DPM++ 2M SDE", "DPM++ 2M SDE Heun", "DPM++ 3M SDE", "DPM++ 4M SDE", "DPM++ SDE", "DPM++ 2s a RF",
    "DPM++ 2M (flow ODE)", "DPM++ 2M Heun (flow ODE)", "DPM++ 3M (flow ODE)", "DPM++ 2M CFG++", "DPM++ SDE CFG++",
    "DPM2",
    "Euler", "Euler a", "Euler A2", "Euler CFG++", "Euler a CFG++", "Euler Dy", "Euler SMEA Dy", "Euler Dy CFG++",
    "Euler SMEA Dy CFG++", "CFG++ UD10 AB",
    "ER SDE", "ER SDE (Reverse-time)", "ER SDE (ODE)", "ER SDE (Tunable)",
    "Res Multistep", "Res Multistep CFG++", "Res Multistep A", "Res Multistep A CFG++",
    "Heun", "EXP Heun 2 x0", "EXP Heun 2 x0 SDE",
    "LMS", "IPNDM", "IPNDM_V", "DEIS",
    "UniPC", "UniPC bh2",
    "Restart", "Restart (flow)",
    "LCM", "DDIM", "PLMS", "Kohaku LoNyu Yog", "Gradient Estimation", "Gradient Estimation CFG++", "SEEDS 2",
    "SEEDS 3", "SA Solver", "SA Solver PECE",
    "RES 2M (RES4LYF)", "RES 3M (RES4LYF)", "RES 2S (RES4LYF)", "RES 3S (RES4LYF)", "RES 5S (RES4LYF)",
    "RES 6S (RES4LYF)", "RES 2M ODE (RES4LYF)", "RES 3M ODE (RES4LYF)", "RES 2S ODE (RES4LYF)",
    "RES 3S ODE (RES4LYF)", "RES 5S ODE (RES4LYF)", "RES 6S ODE (RES4LYF)", "DEIS 2M (RES4LYF)",
    "DEIS 3M (RES4LYF)", "DEIS 2M ODE (RES4LYF)", "DEIS 3M ODE (RES4LYF)",
)
EXPECTED_SCHEDULERS = (
    "Automatic",
    "Simple", "Normal", "Uniform", "SGM Uniform", "DDIM",
    "Beta", "Beta57 (RES4LYF)",
    "Linear Quadratic", "KL Optimal",
    "Karras", "Karras Dynamic", "Flow Cosmos rho7", "Flow Cosmos Dynamic",
    "Exponential", "Polyexponential", "Cosine", "CosineExponential blend", "Phi",
    "Laplace", "React Cosinusoidal DynSF", "Tan (RES4LYF)",
    "Align Your Steps", "Turbo", "Bong Tangent", "FlowMatchEulerDiscrete", "Flux2",
    "custom",
)

# Registration order on the development PC (2026-10-04): Forge's lists (modules/sd_samplers_kdiffusion.py,
# sd_samplers_timesteps.py, modules_forge/forge_alter_samplers.py; modules/sd_schedulers.py), then the
# extensions in load order — this one (Extra Samplers SPECS / Extra Schedulers SCHEDULER_SPECS),
# sd-forge-res4lyf 3a2e48d (scripts/forge_res4lyf.py: _RES_SAMPLERS through add_sampler; Tan / Beta57 appended
# to sd_schedulers.schedulers only), sd_forge_neo_extra_samplers (scripts/extra_samplers.py _samplers_extra).
FORGE_SAMPLERS = (
    "DPM++ 2M", "DPM++ SDE", "DPM++ 2M SDE", "DPM++ 3M SDE", "DPM++ 2s a RF", "Euler a", "Euler", "ER SDE", "LCM",
    "LMS", "Heun", "DPM2", "Res Multistep", "Kohaku LoNyu Yog", "Restart", "UniPC", "DDIM", "PLMS",
    "DPM++ 2M CFG++", "Euler a CFG++", "Euler CFG++",
)
RES4LYF_SAMPLERS = tuple(f"{label} (RES4LYF)" for label in (
    "RES 2M", "RES 3M", "RES 2S", "RES 3S", "RES 5S", "RES 6S", "RES 2M ODE", "RES 3M ODE", "RES 2S ODE",
    "RES 3S ODE", "RES 5S ODE", "RES 6S ODE", "DEIS 2M", "DEIS 3M", "DEIS 2M ODE", "DEIS 3M ODE"))
RES4LYF_SCHEDULERS = (("tan", "Tan (RES4LYF)"), ("beta57", "Beta57 (RES4LYF)"))
NEO_EXTRA_SAMPLERS = (
    ("Gradient Estimation", "gradient_estimation"), ("Gradient Estimation CFG++", "gradient_estimation_cfg_pp"),
    ("SEEDS 2", "seeds_2"), ("SEEDS 3", "seeds_3"), ("SA Solver", "sa_solver"), ("SA Solver PECE", "sa_solver_pece"),
    ("DPM++ SDE CFG++", "dpmpp_sde_cfg_pp"), ("EXP Heun 2 x0", "exp_heun_2_x0"),
    ("EXP Heun 2 x0 SDE", "exp_heun_2_x0_sde"), ("Res Multistep CFG++", "res_multistep_cfg_pp"),
    ("Res Multistep A", "res_multistep_aa"), ("Res Multistep A CFG++", "res_multistep_a_cfg_pp"),
    ("Euler A2", "euler_a2"),
)


def _our_sampler_specs():
    from sam3ext.extra_samplers import registry
    return registry.SPECS


def _our_scheduler_specs():
    from sam3ext.extra_schedulers import registry
    return registry.SCHEDULER_SPECS


def _family_labels(sections) -> list[str]:
    return [label for section in sections if isinstance(section, lo.Family) for label in section.labels]


def _ordered(labels, sections, **kwargs) -> list[str]:
    order = lo.new_order(labels, sections, **kwargs)
    return [labels[i] for i in order]


def _dev_pc_samplers() -> list[str]:
    return [*FORGE_SAMPLERS, *(spec.label for spec in _our_sampler_specs()), *RES4LYF_SAMPLERS,
            *(label for label, _alias in NEO_EXTRA_SAMPLERS)]


class NewOrderTests(unittest.TestCase):
    def test_the_development_pcs_samplers_come_out_in_the_users_order(self):
        labels = _dev_pc_samplers()
        self.assertEqual(sorted(labels), sorted(EXPECTED_SAMPLERS))
        self.assertEqual(_ordered(labels, lo.SAMPLER_SECTIONS), list(EXPECTED_SAMPLERS))

    def test_the_development_pcs_schedulers_come_out_in_the_users_order(self):
        forge = ["Automatic", "Karras", "Exponential", "Polyexponential", "Normal", "Simple", "Uniform", "SGM Uniform",
                 "Linear Quadratic", "KL Optimal", "DDIM", "Align Your Steps", "Beta", "Turbo", "Bong Tangent",
                 "FlowMatchEulerDiscrete", "Flux2"]
        ours = [spec.label for spec in _our_scheduler_specs()]
        res4lyf = [label for _name, label in RES4LYF_SCHEDULERS]
        visible = forge + ours + res4lyf          # sd_schedulers.schedulers
        installed = [label for label in EXPECTED_SCHEDULERS if label in visible]
        self.assertEqual(sorted(visible), sorted(installed))
        self.assertEqual(_ordered(visible, lo.SCHEDULER_SECTIONS), installed)
        everything = forge + ours                  # sd_schedulers.all_schedulers (RES4LYF does not add there)
        self.assertEqual(_ordered(everything, lo.SCHEDULER_SECTIONS),
                         [label for label in installed if label in everything])

    def test_entries_that_are_not_installed_are_skipped(self):
        # Forge alone, Forge + this extension, without RES4LYF or the neo extra samplers, in any start order
        for labels in (list(FORGE_SAMPLERS), [*FORGE_SAMPLERS, *(s.label for s in _our_sampler_specs())],
                       [*FORGE_SAMPLERS, *RES4LYF_SAMPLERS],
                       [FORGE_SAMPLERS[0], *reversed(_dev_pc_samplers()[1:])]):
            with self.subTest(n=len(labels)):
                # the families in the table's order; the RES4LYF block in its current order
                self.assertEqual(_ordered(labels, lo.SAMPLER_SECTIONS),
                                 [label for label in EXPECTED_SAMPLERS if label in labels and label not in RES4LYF_SAMPLERS]
                                 + [label for label in labels if label in RES4LYF_SAMPLERS])

    def test_an_unlisted_label_joins_the_family_its_name_starts_with(self):
        labels = ["DPM++ 2M", "Euler", "DPM adaptive", "Euler Foo", "euler bar", "DPM++ 2M", "ER SDE",
                  "ER SDE Foo", "Res Multistep X", "Heun", "EXP Heun 3", "Heunpp2", "UniPC bh1", "Restart Z",
                  "LMS Foo", "SEEDS 4", "DPM fast", "DEIS 9", "DDIM CFG++"]
        got = _ordered(labels, lo.SAMPLER_SECTIONS)
        self.assertEqual(got, [
            "DPM++ 2M", "DPM++ 2M", "DPM adaptive", "DPM fast",          # DPM family; unlisted at its end, in order
            "Euler", "Euler Foo", "euler bar",                           # case-insensitive
            "ER SDE", "ER SDE Foo",
            "Res Multistep X",
            "Heun", "EXP Heun 3", "Heunpp2",
            "LMS Foo", "DEIS 9",
            "UniPC bh1",
            "Restart Z",
            "SEEDS 4", "DDIM CFG++",
        ])

    def test_the_longest_matching_start_wins(self):
        sections = (lo.Family(("A",), prefixes=("Foo",)), lo.Family(("B",), prefixes=("Foo Bar",)), lo.REST)
        self.assertEqual(_ordered(["A", "B", "Foo Bar 1", "Foo 2", "Foo Baz"], sections),
                         ["A", "Foo 2", "Foo Baz", "B", "Foo Bar 1"])

    def test_res4lyf_samplers_form_one_block_whatever_their_name_starts_with(self):
        labels = ["DPM++ 2M", "DEIS 2M (RES4LYF)", "DPM++ 2M Foo (RES4LYF)", "Euler (res4lyf)", "DEIS", "Zeta"]
        self.assertEqual(_ordered(labels, lo.SAMPLER_SECTIONS),
                         ["DPM++ 2M", "DEIS", "DEIS 2M (RES4LYF)", "DPM++ 2M Foo (RES4LYF)", "Euler (res4lyf)",
                          "Zeta"])

    def test_the_rest_keeps_its_order_at_the_end_and_custom_stays_last(self):
        samplers = ["DPM++ 2M", "Zeta", "Alpha", "RES 2M (RES4LYF)", "Euler", "Mu"]
        self.assertEqual(_ordered(samplers, lo.SAMPLER_SECTIONS),
                         ["DPM++ 2M", "Euler", "RES 2M (RES4LYF)", "Zeta", "Alpha", "Mu"])
        schedulers = ["Automatic", "custom", "Zeta", "Karras", "Alpha", "Foo (RES4LYF)", "Flow Cosmos rho5"]
        self.assertEqual(_ordered(schedulers, lo.SCHEDULER_SECTIONS),
                         ["Automatic", "Karras", "Flow Cosmos rho5", "Zeta", "Alpha", "Foo (RES4LYF)", "custom"])

    def test_flux_realistic_takes_the_slot_of_dpmpp_2s_a_rf(self):
        # forbidden_knowledge: Forge labels the same sample_dpmpp_2s_ancestral_RF entry "Flux Realistic"
        labels = [label if label != "DPM++ 2s a RF" else "Flux Realistic" for label in _dev_pc_samplers()]
        expected = [label if label != "DPM++ 2s a RF" else "Flux Realistic" for label in EXPECTED_SAMPLERS]
        self.assertEqual(_ordered(labels, lo.SAMPLER_SECTIONS), expected)

    def test_the_first_entry_stays_first(self):
        labels = ["Euler", "Euler a", "ER SDE", "DPM++ 2M", "custom"]
        self.assertEqual(_ordered(labels, lo.SAMPLER_SECTIONS), ["Euler", "DPM++ 2M", "Euler a", "ER SDE", "custom"])
        self.assertEqual(_ordered(["Karras", "Automatic", "Beta"], lo.SCHEDULER_SECTIONS),
                         ["Karras", "Automatic", "Beta"])
        self.assertEqual(_ordered(["custom", "Beta", "Automatic"], lo.SCHEDULER_SECTIONS),
                         ["custom", "Automatic", "Beta"])

    def test_kept_indices_stay_in_front_in_their_order(self):
        labels = ["Euler", "LMS", "DPM++ 2M", "Heun"]
        self.assertEqual(_ordered(labels, lo.SAMPLER_SECTIONS, keep=(0, 3)), ["Euler", "Heun", "DPM++ 2M", "LMS"])
        self.assertEqual(_ordered(labels, lo.SAMPLER_SECTIONS, keep=(3, 0, None, 9, -1)),
                         ["Euler", "Heun", "DPM++ 2M", "LMS"])
        self.assertEqual(_ordered(labels, lo.SAMPLER_SECTIONS, keep=()), ["DPM++ 2M", "Euler", "Heun", "LMS"])
        self.assertEqual(lo.new_order([], lo.SAMPLER_SECTIONS), [])

    def test_always_a_permutation_and_idempotent(self):
        import random

        pool = [*EXPECTED_SAMPLERS, "Flux Realistic", "Zeta", "DPM adaptive", "Euler Foo", "x (RES4LYF)", "Zeta"]
        rng = random.Random(20261004)
        for trial in range(200):
            labels = rng.sample(pool, rng.randint(0, len(pool)))
            order = lo.new_order(labels, lo.SAMPLER_SECTIONS)
            self.assertEqual(sorted(order), list(range(len(labels))), trial)
            once = [labels[i] for i in order]
            self.assertEqual(_ordered(once, lo.SAMPLER_SECTIONS), once, trial)
            if labels:
                self.assertEqual(once[0], labels[0], trial)

    def test_the_tables(self):
        for sections in (lo.SAMPLER_SECTIONS, lo.SCHEDULER_SECTIONS):
            labels = _family_labels(sections)
            self.assertEqual([k for k, c in collections.Counter(labels).items() if c > 1], [])
            self.assertEqual(sum(1 for s in sections if s is lo.REST), 1)
        self.assertEqual(_family_labels(lo.SAMPLER_SECTIONS)[0], "DPM++ 2M")
        self.assertEqual(_family_labels(lo.SCHEDULER_SECTIONS)[0], "Automatic")
        self.assertEqual(lo.SCHEDULER_SECTIONS[-1], lo.Family(("custom",)))
        self.assertEqual(lo.SAMPLER_SECTIONS[-2:], (lo.Suffix("(RES4LYF)"), lo.REST))
        self.assertEqual([label for label in _family_labels(lo.SAMPLER_SECTIONS) if label != "Flux Realistic"],
                         [label for label in EXPECTED_SAMPLERS if not label.endswith("(RES4LYF)")])
        self.assertEqual(_family_labels(lo.SCHEDULER_SECTIONS), list(EXPECTED_SCHEDULERS))


class EveryEntryIsPlacedTests(unittest.TestCase):
    """New samplers/schedulers must be placed in the table on purpose (else they would land in "the rest")."""

    def test_every_extra_sampler_and_scheduler_of_this_extension(self):
        samplers = set(_family_labels(lo.SAMPLER_SECTIONS))
        for spec in _our_sampler_specs():
            self.assertIn(spec.label, samplers)
        schedulers = set(_family_labels(lo.SCHEDULER_SECTIONS))
        for spec in _our_scheduler_specs():
            self.assertIn(spec.label, schedulers)

    def test_every_forge_built_in(self):
        samplers = set(_family_labels(lo.SAMPLER_SECTIONS))
        forge = forge_sampler_labels()
        self.assertEqual(sorted(label for label, _alt, _aliases in forge if label not in FORGE_SAMPLERS), [],
                         "Forge's sampler list changed: update FORGE_SAMPLERS and the table")
        for label, alternative, _aliases in forge:
            self.assertIn(label, samplers)
            if alternative is not None:
                self.assertIn(alternative, samplers)
        schedulers = set(_family_labels(lo.SCHEDULER_SECTIONS))
        for name, label, _kwargs in forge_scheduler_rows():
            self.assertIn(label, schedulers, name)

    def test_the_other_extensions_snapshot_matches_the_installed_files(self):
        root = esf.forge_root()
        res4lyf = None if root is None else root / "extensions" / "sd-forge-res4lyf" / "scripts" / "forge_res4lyf.py"
        neo = None if root is None else root / "extensions" / "sd_forge_neo_extra_samplers" / "scripts" / "extra_samplers.py"
        if res4lyf is None or not res4lyf.is_file() or not neo.is_file():
            self.skipTest("sd-forge-res4lyf / sd_forge_neo_extra_samplers are not installed in this Forge")
        tree = ast.parse(res4lyf.read_text(encoding="utf-8"))
        rows = _assigned_list(tree, "_RES_SAMPLERS")
        self.assertEqual([f"{ast.literal_eval(row.elts[0])} (RES4LYF)" for row in rows], list(RES4LYF_SAMPLERS))
        additions = _assigned_list(tree, "additions")
        self.assertEqual([(ast.literal_eval(c.args[0]), ast.literal_eval(c.args[1])) for c in additions],
                         list(RES4LYF_SCHEDULERS))
        rows = _assigned_list(ast.parse(neo.read_text(encoding="utf-8")), "_samplers_extra")
        self.assertEqual([(ast.literal_eval(r.elts[0]), ast.literal_eval(r.elts[2])[0]) for r in rows],
                         list(NEO_EXTRA_SAMPLERS))


# ---------------------------------------------------------------------------
# Forge's own code
# ---------------------------------------------------------------------------


def _assigned_list(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value.elts
    raise LookupError(name)


def forge_sampler_labels():
    """``(label, alternative label, aliases)`` of Forge's built-in samplers, in ``all_samplers`` order."""
    out = []
    tree = ast.parse(esf.forge_source("modules/sd_samplers_kdiffusion.py"))
    for row in _assigned_list(tree, "samplers_k_diffusion"):
        first = row.elts[0]
        if isinstance(first, ast.IfExp):     # "Flux Realistic" if opts.forbidden_knowledge else "DPM++ 2s a RF"
            out.append((ast.literal_eval(first.orelse), ast.literal_eval(first.body), ast.literal_eval(row.elts[2])))
        else:
            out.append((ast.literal_eval(first), None, ast.literal_eval(row.elts[2])))
    tree = ast.parse(esf.forge_source("modules/sd_samplers_timesteps.py"))
    for row in _assigned_list(tree, "samplers_timesteps"):
        out.append((ast.literal_eval(row.elts[0]), None, ast.literal_eval(row.elts[2])))
    tree = ast.parse(esf.forge_source("modules_forge/forge_alter_samplers.py"))
    for call in _assigned_list(tree, "samplers_data_alter"):
        out.append((ast.literal_eval(call.args[0]), None, [ast.literal_eval(call.args[1])]))
    return out


def forge_scheduler_rows():
    """``(name, label, keyword arguments)`` of Forge's ``all_schedulers`` (functions left out)."""
    rows = []
    for call in _assigned_list(ast.parse(esf.forge_source("modules/sd_schedulers.py")), "all_schedulers"):
        kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        rows.append((ast.literal_eval(call.args[0]), ast.literal_eval(call.args[1]), kwargs))
    return rows


def _forge_scheduler_class():
    source = esf.forge_source("modules/sd_schedulers.py")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    import dataclasses
    from typing import Callable

    namespace = {"dataclasses": dataclasses, "Callable": Callable}
    decorators = "".join(f"@{ast.get_source_segment(source, d)}\n" for d in node.decorator_list)
    exec(compile(decorators + ast.get_source_segment(source, node), "modules/sd_schedulers.py", "exec"), namespace)  # noqa: S102
    return namespace["Scheduler"]


SamplerData = collections.namedtuple("SamplerData", ["name", "constructor", "aliases", "options"])


def _constructor(model):   # never called here
    raise AssertionError


class DevPcForge:
    """``modules.sd_samplers`` / ``sd_schedulers`` / ``sd_samplers_kdiffusion`` with Forge's code, filled the way
    the development PC's Forge fills them at load."""

    def __init__(self, hide_samplers=(), hide_schedulers=()):
        opts = types.SimpleNamespace(hide_samplers=list(hide_samplers), hide_schedulers=list(hide_schedulers))
        self.opts = opts
        Scheduler = _forge_scheduler_class()
        sched = types.ModuleType("modules.sd_schedulers")
        sched.Scheduler = Scheduler
        sched.all_schedulers = [Scheduler(name, label, (lambda *a, **k: None) if name != "automatic" else None, **kw)
                                for name, label, kw in forge_scheduler_rows()]
        sched.schedulers = [s for s in sched.all_schedulers if s.label not in opts.hide_schedulers]
        sched.schedulers_map = {**{x.name: x for x in sched.schedulers}, **{x.label: x for x in sched.schedulers}}
        self.sd_schedulers = sched

        kd = types.ModuleType("modules.sd_samplers_kdiffusion")
        kd.k_diffusion_scheduler = {x.name: x.function for x in sched.schedulers}
        self.sd_samplers_kdiffusion = kd

        sam = types.ModuleType("modules.sd_samplers")
        sam.functools = functools
        sam.shared = types.SimpleNamespace(opts=opts)
        sam.sd_schedulers = sched
        sam.all_samplers = [SamplerData(label, _constructor, list(aliases), {})
                            for label, _alt, aliases in forge_sampler_labels()]
        sam.all_samplers_map = {x.name: x for x in sam.all_samplers}
        sam.samplers, sam.samplers_for_img2img, sam.samplers_map, sam.samplers_hidden = [], [], {}, {}
        for name in ("find_sampler_config", "set_samplers", "add_sampler", "visible_sampler_names", "visible_samplers",
                     "get_sampler_from_infotext", "get_scheduler_from_infotext", "get_hr_sampler_and_scheduler",
                     "get_sampler_and_scheduler"):
            exec(compile(esf.forge_definition("modules/sd_samplers.py", name), "modules/sd_samplers.py", "exec"),  # noqa: S102
                 sam.__dict__)
        sam.set_samplers()
        self.sd_samplers = sam

    def load_extensions(self):
        """This extension (Extra Samplers through add_sampler as ``register`` does, the real
        ``register_schedulers``), RES4LYF, sd_forge_neo_extra_samplers — in Forge's load order."""
        from sam3ext.extra_schedulers import registry as es_registry

        sam, sched = self.sd_samplers, self.sd_schedulers
        for spec in _our_sampler_specs():
            sam.add_sampler(SamplerData(spec.label, _constructor, list(spec.aliases), dict(spec.options)))
        with redirect_stderr(io.StringIO()):
            report = es_registry.register_schedulers(sd_schedulers=sched, sd_samplers=sam,
                                                     sd_samplers_kdiffusion=self.sd_samplers_kdiffusion,
                                                     hidden_labels=self.opts.hide_schedulers)
        assert report.error is None, report.error
        for label in RES4LYF_SAMPLERS:              # forge_res4lyf.py _register_samplers
            funcname = "sample_" + label.removesuffix(" (RES4LYF)").lower().replace(" ", "_")
            sam.add_sampler(SamplerData(label, _constructor, [funcname, f"k_{funcname}"], {"scheduler": "beta"}))
        existing = {s.name for s in sched.schedulers}   # forge_res4lyf.py _register_schedulers
        for name, label in RES4LYF_SCHEDULERS:
            if name in existing:
                continue
            scheduler = sched.Scheduler(name, label, lambda *a, **k: None, need_inner_model=name == "beta57")
            sched.schedulers.append(scheduler)
            sched.schedulers_map[scheduler.name] = scheduler
            sched.schedulers_map[scheduler.label] = scheduler
        extra = [SamplerData(label, _constructor, [alias], {}) for label, alias in NEO_EXTRA_SAMPLERS]
        names = {s.name for s in sam.all_samplers}      # extra_samplers.py registration
        sam.all_samplers.extend(s for s in extra if s.name not in names)
        sam.all_samplers_map = {x.name: x for x in sam.all_samplers}
        sam.set_samplers()
        return self

    def apply(self):
        logged = io.StringIO()
        report = lo.apply(self.sd_samplers, self.sd_schedulers, log=lambda message: logged.write(message + "\n"))
        return report, logged.getvalue()

    def reload_ui(self):
        """Reload UI (webui.py ``webui_worker`` → ``initialize.initialize_rest(reload_script_modules=True)``):
        Forge's modules stay loaded with the lists as they are, ``set_samplers()`` runs with the current
        Settings (Hide samplers needs only a Reload UI), every script loads again — the registrations above
        find their entries and add nothing — and then the before_ui callbacks run."""
        self.sd_samplers.set_samplers()
        self.load_extensions()
        return self.apply()

    def shown(self) -> dict:
        """What Forge shows and resolves, by name (comparable between two separate set-ups), maps in key order."""
        sam, sched = self.sd_samplers, self.sd_schedulers
        found = self.lookups()
        found.update(
            all_samplers_map=[(k, v.name) for k, v in sam.all_samplers_map.items()],
            samplers_map=list(sam.samplers_map.items()),
            schedulers_map=[(k, v.name, v.label) for k, v in sched.schedulers_map.items()],
            k_diffusion_scheduler=list(self.sd_samplers_kdiffusion.k_diffusion_scheduler),
            default_config=sam.find_sampler_config(None).name,
            all_samplers=[s.name for s in sam.all_samplers],
            visible_samplers=sam.visible_sampler_names(),
            all_schedulers=[(s.name, s.label) for s in sched.all_schedulers],
            schedulers=[(s.name, s.label) for s in sched.schedulers],
            dropdowns=self.dropdown_choices(),
            xyz=self.xyz_choices(),
            api=self.api_lists(),
        )
        return found

    # -- what Forge reads -------------------------------------------------------------------------------------

    def _namespace(self):
        return {"sd_samplers": self.sd_samplers, "sd_schedulers": self.sd_schedulers, "opts": self.opts}

    def xyz_choices(self) -> dict:
        """scripts/xyz_grid.py: the ``choices`` lambdas of the Sampler / Hires sampler / Schedule type axes."""
        tree = ast.parse(esf.forge_source("scripts/xyz_grid.py"))
        out = collections.defaultdict(list)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id.startswith("AxisOption") and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in ("Sampler", "Hires sampler", "Schedule type")):
                choices = next(k.value for k in node.keywords if k.arg == "choices")
                function = eval(compile(ast.Expression(choices), "scripts/xyz_grid.py", "eval"), self._namespace())  # noqa: S307
                out[f"{node.func.id}:{node.args[0].value}"].append(function())
        return dict(out)

    def dropdown_choices(self) -> dict:
        """modules/processing_scripts/sampler.py ``ui`` (txt2img/img2img dropdowns) and modules/ui.py's Hires
        dropdowns (elem_id hr_sampler / hr_scheduler): their ``choices`` expressions, evaluated now."""
        out = {}
        tree = ast.parse(esf.forge_source("modules/processing_scripts/sampler.py"))
        ui = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "ui")
        for node in ui.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                out[node.targets[0].id] = eval(compile(ast.Expression(node.value), "sampler.py", "eval"),  # noqa: S307
                                               self._namespace())
        tree = ast.parse(esf.forge_source("modules/ui.py"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                keywords = {k.arg: k.value for k in node.keywords}
                elem_id = keywords.get("elem_id")
                if isinstance(elem_id, ast.Constant) and elem_id.value in ("hr_sampler", "hr_scheduler"):
                    out[elem_id.value] = eval(compile(ast.Expression(keywords["choices"]), "ui.py", "eval"),  # noqa: S307
                                              self._namespace())
        return out

    def api_lists(self) -> tuple:
        """modules/api/api.py ``Api.get_samplers`` / ``Api.get_schedulers`` (``/sdapi/v1/samplers|schedulers``)."""
        namespace = self._namespace()
        for name in ("get_samplers", "get_schedulers"):
            exec(compile(esf.forge_definition("modules/api/api.py", name, cls="Api"), "api.py", "exec"), namespace)  # noqa: S102
        return namespace["get_samplers"](None), namespace["get_schedulers"](None)

    def lookups(self) -> dict:
        """Every name Forge resolves: the maps, the defaults and ``get_sampler_and_scheduler`` on every sampler ×
        scheduler option, separate and in the legacy "<sampler> <scheduler>" form."""
        sam, sched = self.sd_samplers, self.sd_schedulers
        sam.get_sampler_and_scheduler.cache_clear()
        options = sorted({o for s in sched.schedulers for o in (s.label, s.name, *(s.aliases or []))})
        names = [s.name for s in sam.all_samplers]
        parsed = {}
        for sampler in [None, "", "nonexistent", *names]:
            for scheduler in [None, "nonexistent", *options]:
                parsed[(sampler, scheduler)] = sam.get_sampler_and_scheduler(sampler, scheduler)
                parsed[(sampler, scheduler, "raw")] = sam.get_sampler_and_scheduler(sampler, scheduler,
                                                                                     convert_automatic=False)
            for option in options:
                combined = f"{sampler} {option}"
                parsed[(combined, None)] = sam.get_sampler_and_scheduler(combined, None)
        hr = {}
        for name in ["Use same sampler", *sorted(names)[::7]]:
            for option in ["Use same scheduler", *options[::5]]:
                d = {"Sampler": "Euler", "Schedule type": "Karras", "Hires sampler": name,
                     "Hires schedule type": option}
                hr[(name, option)] = sam.get_hr_sampler_and_scheduler(d)
        return {
            "all_samplers_map": {k: id(v) for k, v in sam.all_samplers_map.items()},
            "samplers_map": dict(sam.samplers_map),
            "schedulers_map": {k: id(v) for k, v in sched.schedulers_map.items()},
            "k_diffusion_scheduler": dict(self.sd_samplers_kdiffusion.k_diffusion_scheduler),
            "default_config": id(sam.find_sampler_config(None)),
            "default_pair": sam.get_sampler_and_scheduler(None, None),
            "first_visible": sam.visible_sampler_names()[:1],
            "parsed": parsed,
            "hr": hr,
            "entries": sorted((s.name, tuple(s.aliases), tuple(sorted(s.options))) for s in sam.all_samplers),
            "scheduler_entries": sorted((s.name, s.label, tuple(s.aliases or ()), s.default_rho, s.need_inner_model)
                                        for s in sched.all_schedulers + sched.schedulers),
        }


class ApplyOnForgeTests(unittest.TestCase):
    def setUp(self):
        esf.require_forge()
        self.forge = DevPcForge().load_extensions()

    def test_forge_and_the_extensions_register_in_load_order_first(self):
        # the starting point: registration order (this is what the lists look like without the reorder)
        self.assertEqual([s.name for s in self.forge.sd_samplers.all_samplers], _dev_pc_samplers())
        self.assertEqual(self.forge.sd_samplers.all_samplers[0].name, "DPM++ 2M")
        self.assertEqual(self.forge.sd_schedulers.schedulers[0].label, "Automatic")

    def test_the_lists_are_sorted_in_place(self):
        sam, sched = self.forge.sd_samplers, self.forge.sd_schedulers
        lists = (sam.all_samplers, sam.samplers, sam.samplers_for_img2img, sched.all_schedulers, sched.schedulers)
        report, logged = self.forge.apply()
        self.assertTrue(report.samplers_moved and report.schedulers_moved)
        self.assertEqual((report.errors, report.restored_keys), ([], []))
        self.assertEqual(logged, "Sampling method and Schedule type lists sorted by family.\n")
        for before, after in zip(lists, (sam.all_samplers, sam.samplers, sam.samplers_for_img2img,
                                         sched.all_schedulers, sched.schedulers)):
            self.assertIs(before, after)
        self.assertIs(sam.samplers, sam.all_samplers)          # Forge's set_samplers ran
        self.assertEqual([s.name for s in sam.all_samplers], list(EXPECTED_SAMPLERS))
        self.assertEqual([s.label for s in sched.schedulers],
                         [label for label in EXPECTED_SCHEDULERS if label in {s.label for s in sched.schedulers}])
        self.assertEqual([s.label for s in sched.all_schedulers],
                         [label for label in EXPECTED_SCHEDULERS
                          if label in {s.label for s in sched.all_schedulers}])
        self.assertNotIn("Tan (RES4LYF)", [s.label for s in sched.all_schedulers])   # RES4LYF's choice, kept

    def test_what_forge_shows_follows_the_new_order(self):
        self.forge.apply()
        sam, sched = self.forge.sd_samplers, self.forge.sd_schedulers
        samplers = list(EXPECTED_SAMPLERS)
        schedulers = [s.label for s in sched.schedulers]
        self.assertEqual(schedulers[:4], ["Automatic", "Simple", "Normal", "Uniform"])
        self.assertEqual(schedulers[-1], "custom")
        xyz = self.forge.xyz_choices()
        self.assertEqual(xyz, {"AxisOptionTxt2Img:Sampler": [samplers], "AxisOptionTxt2Img:Hires sampler": [samplers],
                               "AxisOptionImg2Img:Sampler": [samplers], "AxisOption:Schedule type": [schedulers]})
        dropdowns = self.forge.dropdown_choices()
        self.assertEqual(dropdowns["sampler_names"], samplers)
        self.assertEqual(dropdowns["scheduler_names"], schedulers)
        self.assertEqual(dropdowns["hr_sampler"], ["Use same sampler", *samplers])
        self.assertEqual(dropdowns["hr_scheduler"], ["Use same scheduler", *schedulers])
        api_samplers, api_schedulers = self.forge.api_lists()
        self.assertEqual([row["name"] for row in api_samplers], samplers)
        self.assertEqual([row["label"] for row in api_schedulers], schedulers)
        by_name = {s.name: s for s in sam.all_samplers}
        for row in api_samplers:
            self.assertEqual((row["aliases"], row["options"]), (by_name[row["name"]].aliases, by_name[row["name"]].options))

    def test_every_lookup_and_default_is_unchanged(self):
        before = self.forge.lookups()
        report, _logged = self.forge.apply()
        self.assertTrue(report.samplers_moved)
        after = self.forge.lookups()
        self.assertEqual(before["default_pair"], ("DPM++ 2M", "Automatic"))
        for key in before:
            self.assertEqual(after[key], before[key], key)
        self.assertGreater(len(before["parsed"]), 10000)

    def test_a_second_run_changes_nothing_and_logs_nothing(self):
        self.forge.apply()
        sam, sched = self.forge.sd_samplers, self.forge.sd_schedulers
        snapshot = ([id(s) for s in sam.all_samplers], [id(s) for s in sched.all_schedulers],
                    [id(s) for s in sched.schedulers], list(sched.schedulers_map), list(sam.all_samplers_map))
        report, logged = self.forge.apply()
        self.assertFalse(report.samplers_moved or report.schedulers_moved)
        self.assertEqual(logged, "")
        self.assertEqual(snapshot, ([id(s) for s in sam.all_samplers], [id(s) for s in sched.all_schedulers],
                                    [id(s) for s in sched.schedulers], list(sched.schedulers_map),
                                    list(sam.all_samplers_map)))

    def test_the_maps_are_rebuilt_in_the_new_order(self):
        self.forge.apply()
        sam, sched = self.forge.sd_samplers, self.forge.sd_schedulers
        self.assertEqual(list(sam.all_samplers_map), list(EXPECTED_SAMPLERS))
        visible = sched.schedulers
        formula = [*(x.name for x in visible), *(x.label for x in visible)]
        keys = list(sched.schedulers_map)
        self.assertEqual(keys[:len(dict.fromkeys(formula))], list(dict.fromkeys(formula)))
        self.assertIn("karras dynamic", keys[len(dict.fromkeys(formula)):])     # our aliases after them
        keys = list(sam.samplers_map)          # Forge's set_samplers ran on the new order
        self.assertEqual(keys[:2], ["dpm++ 2m", "k_dpmpp_2m"])
        self.assertLess(keys.index("euler"), keys.index("euler a"))
        self.assertLess(keys.index("dpm2"), keys.index("euler"))

    def test_the_lookup_cache_is_cleared(self):
        sam = self.forge.sd_samplers
        sam.get_sampler_and_scheduler("Euler", "Karras")
        self.assertEqual(sam.get_sampler_and_scheduler.cache_info().currsize, 1)
        self.forge.apply()
        self.assertEqual(sam.get_sampler_and_scheduler.cache_info().currsize, 0)

    def test_hidden_entries(self):
        # Settings → Hide samplers / Hide schedulers: the first visible sampler stays the dropdowns' first choice
        forge = DevPcForge(hide_samplers=["DPM++ 2M", "Euler"], hide_schedulers=["Automatic", "Simple"]).load_extensions()
        before = forge.lookups()
        first = forge.dropdown_choices()
        self.assertEqual(first["sampler_names"][0], "DPM++ SDE")
        self.assertEqual(first["scheduler_names"][0], "Karras")
        forge.apply()
        after = forge.lookups()
        for key in before:
            self.assertEqual(after[key], before[key], key)
        choices = forge.dropdown_choices()
        self.assertEqual(choices["sampler_names"][0], "DPM++ SDE")
        self.assertEqual(choices["scheduler_names"][0], "Karras")
        names = [s.name for s in forge.sd_samplers.all_samplers]
        self.assertEqual(names[:2], ["DPM++ 2M", "DPM++ SDE"])
        self.assertEqual(names[2:], [label for label in EXPECTED_SAMPLERS if label not in ("DPM++ 2M", "DPM++ SDE")])
        self.assertEqual(choices["sampler_names"],
                         [label for label in names if label not in ("DPM++ 2M", "Euler")])
        self.assertNotIn("Simple", choices["scheduler_names"])
        self.assertEqual([s.label for s in forge.sd_schedulers.all_schedulers][0], "Automatic")

    def test_a_shared_key_keeps_its_old_entry(self):
        # two entries sharing an alias (an extension that does not check): Forge's samplers_map keeps the later
        # one; the reorder moves "Zeta" (no family) behind Euler, and the key must not change hands
        sam = self.forge.sd_samplers
        sam.all_samplers.insert(1, SamplerData("Zeta", _constructor, ["k_euler"], {}))
        sam.all_samplers_map = {x.name: x for x in sam.all_samplers}
        sam.set_samplers()
        self.assertEqual(sam.samplers_map["k_euler"], "Euler")
        before = self.forge.lookups()
        report, logged = self.forge.apply()
        self.assertEqual([s.name for s in sam.all_samplers][-1], "Zeta")
        self.assertEqual(self.forge.lookups(), before)
        self.assertEqual(report.restored_keys, ["samplers_map['k_euler']"])
        self.assertIn("kept the previous entry for samplers_map['k_euler'] (two entries share that key)", logged)

    # Reload UI: Forge keeps its modules, so the lists come back already sorted, set_samplers() rebuilds
    # samplers_map over that order, and the scripts register again. Each run must equal a fresh start.

    def test_reload_ui_after_hiding_the_first_sampler_equals_a_fresh_start(self):
        # Settings → Hide samplers needs only a Reload UI (shared_options.py needs_reload_ui)
        fresh = DevPcForge(hide_samplers=["DPM++ 2M"]).load_extensions()
        self.assertEqual(fresh.dropdown_choices()["sampler_names"][0], "DPM++ SDE")      # Forge alone
        fresh.apply()
        self.forge.apply()
        self.forge.opts.hide_samplers = ["DPM++ 2M"]
        report, logged = self.forge.reload_ui()
        self.assertEqual(report.errors, [])
        self.assertEqual(self.forge.dropdown_choices()["sampler_names"][0], "DPM++ SDE")
        self.assertEqual(self.forge.sd_samplers.visible_sampler_names()[:3], ["DPM++ SDE", "DPM++ 2M SDE",
                                                                              "DPM++ 2M SDE Heun"])
        self.assertEqual(self.forge.shown(), fresh.shown())
        self.assertEqual(logged, "Sampling method list sorted by family.\n")       # the first visible one moved
        # and back: nothing hidden after another Reload UI is the first start again
        start = DevPcForge().load_extensions()
        start.apply()
        self.forge.opts.hide_samplers = []
        self.forge.reload_ui()
        self.assertEqual(self.forge.shown(), start.shown())

    def test_reload_ui_with_nothing_changed_changes_nothing_and_logs_nothing(self):
        self.forge.apply()
        shown = self.forge.shown()
        report, logged = self.forge.reload_ui()
        self.assertFalse(report.samplers_moved or report.schedulers_moved)
        self.assertEqual((logged, report.errors, report.restored_keys), ("", [], []))
        self.assertEqual(self.forge.shown(), shown)

    @staticmethod
    def _with_zeta(forge):
        # the shared alias of test_a_shared_key_keeps_its_old_entry, registered right after Forge's first sampler
        sam = forge.sd_samplers
        sam.all_samplers.insert(1, SamplerData("Zeta", _constructor, ["k_euler"], {}))
        sam.all_samplers_map = {x.name: x for x in sam.all_samplers}
        sam.set_samplers()
        return forge

    def test_a_shared_key_keeps_its_old_entry_after_reload_ui(self):
        forge = self._with_zeta(self.forge)
        before = forge.lookups()
        forge.apply()
        fresh = self._with_zeta(DevPcForge().load_extensions())
        fresh.apply()
        report, logged = forge.reload_ui()       # set_samplers() over the sorted list hands k_euler to Zeta
        self.assertEqual(forge.sd_samplers.samplers_map["k_euler"], "Euler")
        self.assertEqual(forge.lookups(), before)
        self.assertEqual(forge.shown(), fresh.shown())
        self.assertEqual(report.restored_keys, ["samplers_map['k_euler']"])
        self.assertIn("kept the previous entry for samplers_map['k_euler'] (two entries share that key)", logged)

    @staticmethod
    def _with_karras_zeta(forge):
        # an extension loaded last that checks labels but not names (as the RES4LYF schedulers check names only):
        # "turbo" is also Forge's Turbo; the label puts it in the Karras family, before Turbo once sorted, so
        # Forge's formula over the sorted list (this extension's register_schedulers writes it at every load)
        # would hand "turbo" to Forge's Turbo
        sched = forge.sd_schedulers
        if "Karras Zeta" not in {s.label for s in sched.schedulers}:
            scheduler = sched.Scheduler("turbo", "Karras Zeta", lambda *a, **k: None)
            sched.schedulers.append(scheduler)
            sched.schedulers_map[scheduler.name] = scheduler
            sched.schedulers_map[scheduler.label] = scheduler
        return forge

    def test_a_shared_scheduler_key_keeps_its_old_entry_after_reload_ui(self):
        forge = self._with_karras_zeta(self.forge)
        sched = forge.sd_schedulers
        theirs = sched.schedulers_map["turbo"]
        self.assertEqual(theirs.label, "Karras Zeta")
        report, _logged = forge.apply()
        self.assertEqual(report.restored_keys, ["schedulers_map['turbo']"])
        self.assertIs(sched.schedulers_map["turbo"], theirs)
        fresh = self._with_karras_zeta(DevPcForge().load_extensions())
        fresh.apply()
        forge.sd_samplers.set_samplers()
        forge.load_extensions()
        self._with_karras_zeta(forge)            # loads again too: finds its label, adds nothing
        self.assertIsNot(sched.schedulers_map["turbo"], theirs)     # register_schedulers over the sorted list
        report, _logged = forge.apply()
        self.assertIs(sched.schedulers_map["turbo"], theirs)
        self.assertEqual(report.restored_keys, ["schedulers_map['turbo']"])
        self.assertEqual(forge.shown(), fresh.shown())

    def test_failures_are_logged_and_never_raised(self):
        broken = types.SimpleNamespace(all_samplers=[object(), object()])
        sched = types.SimpleNamespace(schedulers=None, all_schedulers=[1, 2])
        logged = io.StringIO()
        report = lo.apply(broken, sched, log=lambda m: logged.write(m + "\n"))
        self.assertEqual(len(report.errors), 2)
        self.assertEqual(len(broken.all_samplers), 2)
        self.assertIn("family order not applied (samplers: AttributeError", logged.getvalue())
        self.assertEqual(sched.all_schedulers, [1, 2])

    def test_other_list_objects_are_sorted_too(self):
        sam = self.forge.sd_samplers
        sam.samplers = [x for x in sam.all_samplers if x.name != "LCM"]                # a host that filters
        sam.samplers_for_img2img = [x for x in sam.all_samplers if x.name != "PLMS"]
        sam.set_samplers = lambda: None                                                    # and keeps its own lists
        self.forge.apply()
        self.assertEqual([s.name for s in sam.all_samplers], list(EXPECTED_SAMPLERS))
        self.assertEqual([s.name for s in sam.samplers], [label for label in EXPECTED_SAMPLERS if label != "LCM"])
        self.assertEqual([s.name for s in sam.samplers_for_img2img],
                         [label for label in EXPECTED_SAMPLERS if label != "PLMS"])


class ScriptFileTests(unittest.TestCase):
    PATH = ROOT / "scripts" / "list_order.py"

    def test_it_only_registers_before_ui(self):
        registered = []
        callbacks = types.ModuleType("modules.script_callbacks")
        for name in ("on_before_ui", "on_ui_settings", "on_app_started", "on_script_unloaded", "on_ui_tabs"):
            setattr(callbacks, name, lambda fn, *a, _n=name, **k: registered.append((_n, fn)))
        modules = types.ModuleType("modules")
        modules.__path__ = []
        modules.script_callbacks = callbacks
        calls = []
        source = self.PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertFalse([n for n in tree.body if isinstance(n, ast.ClassDef)], "no Script class")
        with esf.stub_modules({"modules": modules, "modules.script_callbacks": callbacks}):
            namespace = {"__name__": "_test_list_order_script"}
            exec(compile(source, str(self.PATH), "exec"), namespace)  # noqa: S102
        self.assertEqual([name for name, _fn in registered], ["on_before_ui"])
        original = lo.apply
        try:
            lo.apply = lambda *a, **k: calls.append((a, k))
            registered[0][1]()
        finally:
            lo.apply = original
        self.assertEqual(calls, [((), {})])

    def test_it_sorts_before_negpip_which_stays_last(self):
        names = sorted(name for name in os.listdir(ROOT / "scripts") if name.endswith(".py"))
        self.assertLess(names.index("list_order.py"), names.index("negpip.py"))
        self.assertEqual(names[-1], "negpip.py")


if __name__ == "__main__":
    unittest.main()
