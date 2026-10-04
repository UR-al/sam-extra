"""All scripts together: load order, Reload UI, and what they register — with the six new features present
(Colorcraft, Anima SPEED, Extra Samplers, Extra Schedulers, the progress bar, the MCP settings).

``tests/_webui_probe.py`` loads every file in ``scripts/`` in Forge's order against one ``modules``
stand-in (Forge's ``set_samplers``/``add_sampler``, ``Scheduler`` and k-diffusion from the checkout),
runs the ``on_ui_settings`` / ``on_before_ui`` / ``on_app_started`` callbacks, then simulates Reload UI
(the files run again, ``sam3ext`` and Forge's modules stay). It runs in its own interpreter because
several scripts install hooks when imported. Checked here:

* every script loads and every callback runs, before and after the reload;
* ``negpip.py`` is still the extension's last script, Colorcraft loads after every script that
  prepares the UNet or the sampler before it;
* the extra samplers and schedulers are each registered once, also after a reload;
* ``scripts/list_order.py`` sorts both lists by family in ``on_before_ui`` (after every script has
  registered), Forge's first entries stay first, and a reload sorts nothing differently;
* XYZ axis labels are unique (Forge's built-in ones included) and a reload neither duplicates nor drops
  any — with a reloaded XYZ grid and with one a host kept;
* settings keys are unique, each section id has one title (and each title one id);
* the HTTP routes are registered once, also when app start fires again on the same app;
* every always-on script that puts itself in the ANIMA section has an ``anima`` lane in
  ``layout_lanes.REGISTRY``. Fixed here: Anima Optimal Scale was missing, so the notebook layout
  treated it as "도구·실험" — hidden under "더 보기" while closed, below the ANIMA items when open —
  instead of right above Colorcraft, whose sorting priority places it under Optimal Scale.
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no version check over the network

import collections
import configparser
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORGE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext import layout_lanes  # noqa: E402

NEW_SCRIPTS = ("colorcraft.py", "anima_speed.py", "anima_extra_samplers.py", "anima_extra_schedulers.py",
               "appearance_progress_bar.py", "mcp_settings.py")
OUR_SAMPLERS = (
    "ER SDE (Reverse-time)", "ER SDE (ODE)", "DPM++ 4M SDE", "Euler Dy CFG++", "Euler SMEA Dy CFG++",
    # v0.33.0 (UniPC bh2 and DEIS need Forge's modules.sd_samplers_extra / k_diffusion.deis: the probe has them)
    "ER SDE (Tunable)", "Euler Dy", "Euler SMEA Dy", "DPM++ 2M SDE Heun", "DPM++ 2M (flow ODE)",
    "DPM++ 2M Heun (flow ODE)", "DPM++ 3M (flow ODE)", "UniPC bh2", "CFG++ UD10 AB", "IPNDM", "IPNDM_V", "DEIS",
    "Restart (flow)",
)
OUR_SAMPLER_ALIASES = (
    "er_sde_reverse_time", "er_sde_ode", "dpmpp_4m_sde", "euler_dy_cfg_pp", "euler_smea_dy_cfg_pp",
    "er_sde_tunable", "euler_dy", "k_euler_dy", "euler_smea_dy", "k_euler_smea_dy", "dpmpp_2m_sde_heun",
    "k_dpmpp_2m_sde_heun", "dpmpp_2m_flow_ode", "dpmpp_2m_heun_flow_ode", "dpmpp_3m_flow_ode", "uni_pc_bh2",
    "cfgpp_ud10_ab", "ipndm", "ipndm_v", "deis", "restart_flow",
)
OUR_SCHEDULERS = ("Cosine", "CosineExponential blend", "Phi", "Laplace", "Karras Dynamic", "custom",
                  "React Cosinusoidal DynSF", "Flow Cosmos rho7", "Flow Cosmos Dynamic")
NEW_XYZ_PREFIXES = {"[Colorcraft]": 14, "[Anima SPEED]": 6, "[Extra Samplers]": 3, "[Extra Schedulers (sam-extra)]": 7}
# The probe's Forge stand-in (DPM++ 2M, Euler a, Euler, ER SDE / Automatic, Karras, Exponential) plus ours,
# after scripts/list_order.py (sam3ext/list_order.py: the families of the table, Forge's first entry first).
FAMILY_ORDER_SAMPLERS = (
    "DPM++ 2M", "DPM++ 2M SDE Heun", "DPM++ 4M SDE", "DPM++ 2M (flow ODE)", "DPM++ 2M Heun (flow ODE)",
    "DPM++ 3M (flow ODE)",
    "Euler", "Euler a", "Euler Dy", "Euler SMEA Dy", "Euler Dy CFG++", "Euler SMEA Dy CFG++", "CFG++ UD10 AB",
    "ER SDE", "ER SDE (Reverse-time)", "ER SDE (ODE)", "ER SDE (Tunable)",
    "IPNDM", "IPNDM_V", "DEIS",
    "UniPC bh2",
    "Restart (flow)",
)
FAMILY_ORDER_SCHEDULERS = ("Automatic", "Karras", "Karras Dynamic", "Flow Cosmos rho7", "Flow Cosmos Dynamic",
                           "Exponential", "Cosine", "CosineExponential blend", "Phi", "Laplace",
                           "React Cosinusoidal DynSF", "custom")


def _duplicates(items):
    return sorted(item for item, count in collections.Counter(items).items() if count > 1)


class LoadOrderTests(unittest.TestCase):
    report: dict = {}

    @classmethod
    def setUpClass(cls):
        required = ("scripts/xyz_grid.py", "modules/sd_samplers.py", "modules/sd_schedulers.py",
                    "modules_forge/packages/k_diffusion/sampling.py")
        if not all((FORGE / rel).is_file() for rel in required):
            raise unittest.SkipTest("the Forge checkout is not next to this extension")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", GRADIO_ANALYTICS_ENABLED="False")
        result = subprocess.run([sys.executable, "-B", str(ROOT / "tests" / "_webui_probe.py")], cwd=str(ROOT),
                                capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
                                timeout=600)
        if result.returncode != 0:
            raise AssertionError(f"the probe failed:\n{result.stderr[-4000:]}")
        cls.report = json.loads(result.stdout)

    def test_every_script_loads_and_every_callback_runs(self):
        r = self.report
        for key in ("load_failed", "reload_load_failed"):
            self.assertEqual(r[key], {}, key)
        for key in ("settings_errors", "before_ui_errors", "app_errors", "reload_settings_errors",
                    "reload_before_ui_errors", "reload_app_errors"):
            self.assertEqual(r[key], [], key)

    def test_negpip_is_still_the_last_script(self):
        order = self.report["order"]
        self.assertEqual(order[-1], "negpip.py")
        for name in NEW_SCRIPTS:
            self.assertIn(name, order)
            self.assertLess(order.index(name), order.index("negpip.py"), name)
        # Colorcraft clones the UNet after the scripts that patch it or the sampler before sampling
        for name in ("anima_cfg_optimal_scale.py", "anima_detail_daemon.py", "anima_safe_pag.py",
                     "anima_skimmed_cfg.py", "anima_speed.py"):
            self.assertLess(order.index(name), order.index("colorcraft.py"), name)
        # nothing in metadata.ini moves the new scripts (Forge's own list_scripts on the real folder:
        # tests/test_negpip_load_order.py)
        config = configparser.ConfigParser()
        config.read(ROOT / "metadata.ini", encoding="utf-8")
        self.assertIn("scripts/negpip.py", config.sections())
        for section in config.sections():
            for name in NEW_SCRIPTS:
                self.assertNotIn(name, section)
                self.assertNotIn(name, " ".join(config[section].values()))

    def test_samplers_and_schedulers_are_registered_once(self):
        r = self.report
        for key in ("samplers_at_load", "samplers", "reload_samplers"):
            self.assertEqual(_duplicates(r[key]), [], key)
            for label in OUR_SAMPLERS:
                self.assertEqual(r[key].count(label), 1, (key, label))
        # registered after Forge's own (the order before scripts/list_order.py's before_ui callback)
        self.assertEqual(r["samplers_at_load"][:4], ["DPM++ 2M", "Euler a", "Euler", "ER SDE"], "Forge's own first")
        self.assertEqual(sorted(r["samplers_at_load"][4:]), sorted(OUR_SAMPLERS))
        self.assertEqual(r["reload_samplers"], r["samplers"])
        for alias in OUR_SAMPLER_ALIASES:
            self.assertIn(alias, r["reload_samplers_map"])
        for key in ("schedulers", "visible_schedulers", "reload_schedulers", "reload_visible_schedulers"):
            self.assertEqual(_duplicates(r[key]), [], key)
            for label in OUR_SCHEDULERS:
                self.assertEqual(r[key].count(label), 1, (key, label))
        self.assertEqual(r["reload_schedulers"], r["schedulers"])
        for name in ("cosine", "karras_dynamic", "laplace", "custom", "karras dynamic", "react_cosinusoidal_dynsf",
                     "flow_cosmos_rho7", "flow_cosmos_dynamic"):
            self.assertIn(name, r["reload_scheduler_map"])

    def test_before_ui_sorts_the_lists_by_family(self):
        r = self.report
        self.assertIn("list_order.py", r["order"])
        self.assertLess(r["order"].index("list_order.py"), r["order"].index("negpip.py"))
        for key in ("samplers", "visible_samplers", "reload_samplers"):
            self.assertEqual(r[key], list(FAMILY_ORDER_SAMPLERS), key)
        for key in ("schedulers", "visible_schedulers", "reload_schedulers", "reload_visible_schedulers"):
            self.assertEqual(r[key], list(FAMILY_ORDER_SCHEDULERS), key)
        self.assertEqual(sorted(r["schedulers_at_load"]), sorted(FAMILY_ORDER_SCHEDULERS))
        self.assertNotEqual(r["schedulers_at_load"], list(FAMILY_ORDER_SCHEDULERS), "the probe would not see a sort")
        self.assertNotEqual(r["samplers_at_load"], list(FAMILY_ORDER_SAMPLERS), "the probe would not see a sort")
        # the maps keep their keys (also after the reload)
        for key in ("samplers_map", "reload_samplers_map"):
            self.assertEqual(r[key], r["samplers_map_at_load"], key)
        for key in ("scheduler_map", "reload_scheduler_map"):
            self.assertEqual(r[key], r["scheduler_map_at_load"], key)

    def test_xyz_axes_are_unique_across_reloads(self):
        r = self.report
        labels = r["xyz"]
        forge = r["forge_xyz"]
        added = labels[len(forge):]
        self.assertEqual(labels[:len(forge)], forge, "Forge's own axes are kept, in order")
        # (Forge itself has two "Sampler" axes, one per tab)
        self.assertEqual(_duplicates(added), [])
        self.assertEqual(sorted(set(added) & set(forge)), [])
        self.assertEqual(r["reload_xyz"], labels, "a reloaded XYZ grid gets the same axes")
        self.assertEqual(r["kept_xyz"], labels, "a kept XYZ grid gets nothing twice")
        for prefix, count in NEW_XYZ_PREFIXES.items():
            self.assertEqual(sum(1 for label in labels if label.startswith(prefix + " ")), count, prefix)
        ours = [label for label in labels if label.startswith("[")]
        prefixes = {label.split("]", 1)[0] + "]" for label in ours}
        for a in prefixes:
            for b in prefixes:
                if a != b:
                    self.assertFalse(b.startswith(a), (a, b))   # no prefix guard swallows another

    def test_settings_keys_and_sections_are_unique(self):
        r = self.report
        keys = [key for key, _section, _label in r["options"]]
        self.assertEqual(_duplicates(keys), [])
        self.assertEqual(sorted(map(tuple, (o[:2] for o in r["reload_options"]))),
                         sorted(map(tuple, (o[:2] for o in r["options"]))))
        titles, ids = collections.defaultdict(set), collections.defaultdict(set)
        for _key, section, _label in r["options"]:
            self.assertIsInstance(section, list, _key)
            ids[section[0]].add(section[1])
            titles[section[1]].add(section[0])
        for section_id, names in ids.items():
            self.assertEqual(len(names), 1, (section_id, names))
        for title, section_ids in titles.items():
            self.assertEqual(len(section_ids), 1, (title, section_ids))
        for section_id in ("sam3_colorcraft", "sam3_speed", "sam3_progress", "sam3_mcp"):
            self.assertIn(section_id, ids)

    def test_routes_are_registered_once(self):
        r = self.report
        pairs = [(path, method) for path, methods in r["routes"] for method in methods]
        self.assertEqual(_duplicates(pairs), [])
        self.assertEqual(r["reload_routes"], r["routes"], "app start fired again on the same app")
        self.assertEqual(sum(1 for path, _ in r["routes"] if path == "/sam-extra/progress"), 1)

    def test_every_anima_section_script_is_in_the_anima_lane(self):
        scripts = [s for s in self.report["scripts"] if "error" not in s]
        self.assertEqual([s for s in self.report["scripts"] if "error" in s], [])
        in_section = [s for s in scripts if s["always"] and s["section"] == layout_lanes.ANIMA_SECTION]
        self.assertGreaterEqual({s["file"] for s in in_section},
                                {"colorcraft.py", "anima_speed.py", "anima_extra_samplers.py", "anima_extra_schedulers.py",
                                 "anima_cfg_optimal_scale.py"})
        for script in in_section:
            with self.subTest(script=script["file"]):
                self.assertTrue(script["registered"], script)
                self.assertEqual(script["lane"], "anima", script)

    def test_lane_registry_and_short_names(self):
        keys = {layout_lanes.slot_key(name) for name in self.report["order"]}
        for key in ("colorcraft", "anima-speed", "anima-extra-samplers", "anima-extra-schedulers"):
            self.assertIn(key, keys)
            self.assertEqual(layout_lanes.REGISTRY[key].lane, "anima")
        self.assertTrue(layout_lanes.REGISTRY["anima-extra-samplers"].on.none)
        self.assertTrue(layout_lanes.REGISTRY["anima-extra-schedulers"].on.none)
        source = (ROOT / "javascript" / "notebook_lanes.js").read_text(encoding="utf-8")
        block = re.search(r"var SHORT = \{(.*?)\};", source, re.S).group(1)
        for key in re.findall(r'"([\w-]+)":', block):
            self.assertIn(key, layout_lanes.REGISTRY, key)


if __name__ == "__main__":
    unittest.main()
