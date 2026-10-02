"""One-time ui-config.json migration for the PAG scale and DCW(+a) sliders (sam3ext/guidance/ui_config_migration.py).

The DCW(+a) parity change moved five sliders to the upstream ranges/defaults
(origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:636-667 lambda_l/lambda_h, :675-704
alpha_l/alpha_h, :757-765 rdc_tau default 0.0) and hid the RDC switch, but kept the labels.
The PAG parity change raised the Attn Scale maximum 15 -> 100 under the same label
(origin: iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py:201).
Forge's real ``UiLoadsave`` (modules/ui_loadsave.py, loaded from the install below) reapplies
the saved value/minimum/maximum/step per label, so these tests feed it a ui-config holding the
pre-parity keys (the values an install saved before the change) and check what the UI and the
script then use.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr


ROOT = Path(__file__).resolve().parents[1]
FORGE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3ext.guidance import ui_config_migration as migration  # noqa: E402


def _load_base_tests():
    spec = importlib.util.spec_from_file_location(
        "_ui_config_migration_base_tests", ROOT / "tests" / "test_anima_safe_pag.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_forge_ui_loadsave():
    """Forge's ``modules/ui_loadsave.py`` with only ``errors``/``ui_components`` stubbed."""
    errors = types.ModuleType("modules.errors")

    def display(error, task):
        raise error

    errors.display = display
    components = types.ModuleType("modules.ui_components")
    components.InputAccordionImpl = type("InputAccordionImpl", (), {})
    components.ToolButton = type("ToolButton", (), {})
    package = types.ModuleType("modules")
    package.errors = errors
    package.ui_components = components
    with mock.patch.dict(sys.modules, {
        "modules": package, "modules.errors": errors, "modules.ui_components": components,
    }):
        spec = importlib.util.spec_from_file_location(
            "_forge_ui_loadsave", FORGE / "modules" / "ui_loadsave.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


_BASE = _load_base_tests()
_LOADSAVE = _load_forge_ui_loadsave()

PREFIX = "customscript/anima_safe_pag.py"
TAU = "RDC tau (EMA 기억 구간)"
CWM_LOW = "CWM alpha low (초반 저주파 CFG)"
CWM_HIGH = "CWM alpha high (후반 고주파 CFG)"
# Same label before and after the scale-100 change (forge_sam3_extension@861ac02:
# scripts/anima_safe_pag.py:2703 and now).
PAG_SCALE = "Attn Scale — PAG / SEG guidance scale (cond−weak 배율)"

# UI/script-argument index (scripts/anima_safe_pag.py ui()).
ARG = {
    "cwm_alpha_low": 24, "cwm_alpha_high": 25, "dcw_enabled": 28,
    "dcw_lambda_low": 29, "dcw_lambda_high": 30, "rdc_switch": 58, "rdc_tau": 59,
}


def _slider(tab, label, value, minimum, maximum, step):
    base = f"{PREFIX}/{tab}/{label}"
    return {
        f"{base}/visible": True, f"{base}/value": value,
        f"{base}/minimum": minimum, f"{base}/maximum": maximum, f"{base}/step": step,
    }


def _legacy_tab(tab, *, dcw_low, dcw_high, cwm_low, cwm_high, rdc_on=False, tau=0.15):
    """Keys the pre-parity script saved (forge_sam3_extension@3522928:scripts/
    anima_safe_pag.py:3050-3122): DCW ±0.5/.005, CWM max 1.0, Enable RDC + tau."""
    settings = {
        f"{PREFIX}/{tab}/Enable DCW/visible": True,
        f"{PREFIX}/{tab}/Enable DCW/value": False,
        f"{PREFIX}/{tab}/Enable RDC/visible": True,
        f"{PREFIX}/{tab}/Enable RDC/value": rdc_on,
    }
    settings.update(_slider(tab, "DCW lambda low", dcw_low, -0.5, 0.5, 0.005))
    settings.update(_slider(tab, "DCW lambda high", dcw_high, -0.5, 0.5, 0.005))
    settings.update(_slider(tab, CWM_LOW, cwm_low, -1.0, 1.0, 0.01))
    settings.update(_slider(tab, CWM_HIGH, cwm_high, -1.0, 1.0, 0.01))
    settings.update(_slider(tab, TAU, tau, 0.0, 0.5, 0.01))
    return settings


def _legacy_pag_scale(tab, value):
    """The PAG scale slider the pre-parity script saved (forge_sam3_extension@861ac02:
    scripts/anima_safe_pag.py:2702-2707): same label, 0..15 step .1."""
    return _slider(tab, PAG_SCALE, value, 0.0, 15.0, 0.1)


def _installed_like():
    """The shape found in this install's ui-config.json: txt2img has user-set DCW
    0.08/0.015, CWM low 0.2 and PAG scale 2, img2img still holds the old defaults;
    both PAG scale sliders still carry the old maximum 15."""
    return {
        **_legacy_tab("txt2img", dcw_low=0.08, dcw_high=0.015, cwm_low=0.2, cwm_high=0.15),
        **_legacy_pag_scale("txt2img", 2),
        **_legacy_tab("img2img", dcw_low=0.1, dcw_high=0.02, cwm_low=0.3, cwm_high=0.15),
        **_legacy_pag_scale("img2img", 4.0),
    }


class _Unet:
    """Forge ModelPatcher registration surface (backend/patcher/base.py)."""

    def __init__(self):
        self.model_options = {}

    def clone(self):
        return self

    def set_model_unet_function_wrapper(self, function):
        self.model_options["model_function_wrapper"] = function

    def set_model_sampler_pre_cfg_function(self, function, disable_cfg1_optimization=False):
        self.model_options.setdefault("sampler_pre_cfg_function", []).append(function)

    def set_model_sampler_post_cfg_function(self, function, disable_cfg1_optimization=False):
        self.model_options.setdefault("sampler_post_cfg_function", []).append(function)


class UiConfigMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pag = _BASE._load_pag_module()

    def setUp(self):
        _BASE.AnimaSafePagTests.setUp(self)
        if getattr(self, "_tmp", None) is not None:  # re-run inside a subTest loop
            self._tmp.cleanup()
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "ui-config.json"

    def tearDown(self):
        self._tmp.cleanup()

    # -- helpers ------------------------------------------------------------

    def _write(self, settings):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=4, ensure_ascii=False)

    def _read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _forge_ui(self, tab):
        """Build the script UI and let Forge's UiLoadsave apply the file (modules/ui.py:866-905)."""
        with gr.Blocks() as block:
            inputs = self.pag.AnimaSafePAG().ui(tab == "img2img")
        for component in inputs:  # modules/scripts.py:672-673
            component.custom_script_source = "anima_safe_pag.py"
        loadsave = _LOADSAVE.UiLoadsave(str(self.path))
        loadsave.add_block(block, tab)
        return inputs, loadsave

    def _attach(self, args):
        model = type("Anima", (), {})()
        model.forge_objects = types.SimpleNamespace(unet=_Unet())
        request = types.SimpleNamespace(
            sd_model=model, extra_generation_params={}, steps=20, cfg_scale=4.0,
        )
        with mock.patch.object(self.pag, "_log"):
            self.pag.AnimaSafePAG().process_before_every_sampling(request, *args)

    def _rdc_on_after_ticking_enable_dcw(self, tab):
        inputs, _loadsave = self._forge_ui(tab)
        args = [component.value for component in inputs]
        args[ARG["dcw_enabled"]] = True
        self._attach(args)
        return bool(self.pag._DCW["rdc_on"]), args[ARG["rdc_tau"]]

    # -- RDC ----------------------------------------------------------------

    def test_stale_tau_turns_rdc_on_with_enable_dcw_unless_migrated(self):
        for tab in migration.TABS:
            with self.subTest(tab=tab, migrated=False):
                self.setUp()
                self._write(_installed_like())
                self.assertEqual(self._rdc_on_after_ticking_enable_dcw(tab), (True, 0.15))
            with self.subTest(tab=tab, migrated=True):
                self.setUp()
                self._write(_installed_like())
                self.assertTrue(migration.migrate_ui_config_file(self.path))
                self.assertEqual(self._rdc_on_after_ticking_enable_dcw(tab), (False, 0.0))

    def test_rdc_saved_on_keeps_its_tau(self):
        self._write(_legacy_tab(
            "txt2img", dcw_low=0.1, dcw_high=0.02, cwm_low=0.3, cwm_high=0.15,
            rdc_on=True, tau=0.2,
        ))
        migration.migrate_ui_config_file(self.path)
        saved = self._read()
        self.assertEqual(saved[f"{PREFIX}/txt2img/{TAU}/value"], 0.2)
        self.assertNotIn(f"{PREFIX}/txt2img/Enable RDC/value", saved)
        self.assertNotIn(f"{PREFIX}/txt2img/Enable RDC/visible", saved)
        self.assertEqual(self._rdc_on_after_ticking_enable_dcw("txt2img"), (True, 0.2))

    def test_rdc_saved_off_keeps_a_user_set_tau(self):
        """Only the old default tau 0.15 means "never touched" and moves to upstream's 0
        (off). A tau the user set stays even with the old switch saved off (user decision:
        saved values stay); the log says RDC now follows Enable DCW + tau."""
        self._write(_legacy_tab(
            "txt2img", dcw_low=0.1, dcw_high=0.02, cwm_low=0.3, cwm_high=0.15,
            rdc_on=False, tau=0.27,
        ))
        changes = migration.migrate_ui_config_file(self.path)
        saved = self._read()
        self.assertEqual(saved[f"{PREFIX}/txt2img/{TAU}/value"], 0.27)
        self.assertNotIn(f"{PREFIX}/txt2img/Enable RDC/value", saved)
        self.assertNotIn(f"{PREFIX}/txt2img/Enable RDC/visible", saved)
        self.assertTrue(
            any(TAU in line and "0.27" in line and "Enable DCW" in line for line in changes),
            changes,
        )
        self.assertEqual(self._rdc_on_after_ticking_enable_dcw("txt2img"), (True, 0.27))

    # -- PAG scale ------------------------------------------------------------

    def _pag_scale(self, tab):
        inputs, _loadsave = self._forge_ui(tab)
        return next(item for item in inputs if getattr(item, "label", None) == PAG_SCALE)

    def test_pag_scale_reaches_the_upstream_maximum_and_keeps_the_saved_scale(self):
        """Origin range 0..100 (iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py:201).
        Forge reapplies the saved maximum 15 under the unchanged label unless it is dropped;
        the saved scale (user-set 2 / default 4) stays."""
        self._write(_installed_like())
        expected = {"txt2img": 2, "img2img": 4.0}
        for tab, value in expected.items():
            with self.subTest(tab=tab, migrated=False):
                slider = self._pag_scale(tab)
                self.assertEqual((slider.maximum, slider.value), (15.0, value))
        changes = migration.migrate_ui_config_file(self.path)
        self.assertTrue(any(PAG_SCALE in line for line in changes), changes)
        for tab, value in expected.items():
            with self.subTest(tab=tab, migrated=True):
                slider = self._pag_scale(tab)
                self.assertEqual(
                    (slider.minimum, slider.maximum, slider.step, slider.value),
                    (0.0, 100.0, 0.1, value),
                )

    def test_pag_scale_saved_with_the_new_maximum_is_untouched(self):
        settings = _slider("txt2img", PAG_SCALE, 12.5, 0.0, 100.0, 0.1)
        before = dict(settings)
        self.assertEqual(migration.migrate_ui_settings(settings), [])
        self.assertEqual(settings, before)

    # -- slider bounds / defaults ------------------------------------------

    def test_sliders_reach_the_upstream_bounds_and_old_defaults_move(self):
        self._write(_installed_like())
        migration.migrate_ui_config_file(self.path)
        with gr.Blocks():
            fresh = self.pag.AnimaSafePAG().ui(False)
        expected_values = {
            # txt2img: values the user set stay; CWM high was the old default 0.15.
            "txt2img": {"dcw_lambda_low": 0.08, "dcw_lambda_high": 0.015,
                        "cwm_alpha_low": 0.2, "cwm_alpha_high": 0.0},
            # img2img: all four were the old defaults -> upstream defaults.
            "img2img": {"dcw_lambda_low": 0.05, "dcw_lambda_high": 0.01,
                        "cwm_alpha_low": 0.0, "cwm_alpha_high": 0.0},
        }
        for tab, values in expected_values.items():
            inputs, _loadsave = self._forge_ui(tab)
            for name, value in values.items():
                index = ARG[name]
                with self.subTest(tab=tab, slider=name):
                    self.assertAlmostEqual(inputs[index].value, value, places=9)
                    for field in ("minimum", "maximum", "step"):
                        self.assertEqual(
                            getattr(inputs[index], field), getattr(fresh[index], field), field,
                        )
        # upstream bounds, spelled out: lambda_h ±0.3 step .001, alpha -1..2.
        self.assertEqual((fresh[30].minimum, fresh[30].maximum, fresh[30].step), (-0.3, 0.3, 0.001))
        self.assertEqual((fresh[24].minimum, fresh[24].maximum), (-1.0, 2.0))
        self.assertEqual((fresh[25].minimum, fresh[25].maximum), (-1.0, 2.0))

    def test_without_the_migration_the_old_bounds_win(self):
        self._write(_installed_like())
        inputs, _loadsave = self._forge_ui("img2img")
        self.assertEqual((inputs[30].minimum, inputs[30].maximum, inputs[30].step), (-0.5, 0.5, 0.005))
        self.assertEqual(inputs[24].maximum, 1.0)
        self.assertEqual((inputs[29].value, inputs[30].value), (0.1, 0.02))

    def test_lambda_high_beyond_the_new_range_is_clamped_like_the_server(self):
        settings = _legacy_tab("txt2img", dcw_low=0.1, dcw_high=-0.45, cwm_low=0.3, cwm_high=0.15)
        changes = migration.migrate_ui_settings(settings)
        self.assertEqual(settings[f"{PREFIX}/txt2img/DCW lambda high/value"], -0.3)
        self.assertTrue(any("-0.45 -> -0.3" in line for line in changes), changes)

    # -- one-time ------------------------------------------------------------

    def test_runs_once_and_never_touches_values_saved_afterwards(self):
        self._write(_installed_like())
        self.assertTrue(migration.migrate_ui_config_file(self.path))
        # Forge re-saves the new bounds after building the UI (modules/ui.py:928-929).
        for tab in migration.TABS:
            _inputs, loadsave = self._forge_ui(tab)
            loadsave.dump_defaults()
        self.assertEqual(migration.migrate_ui_config_file(self.path), [])
        saved = self._read()
        self.assertEqual(saved[f"{PREFIX}/img2img/DCW lambda high/step"], 0.001)
        self.assertEqual(saved[f"{PREFIX}/img2img/{CWM_LOW}/maximum"], 2.0)
        self.assertEqual(saved[f"{PREFIX}/img2img/{PAG_SCALE}/maximum"], 100.0)
        self.assertEqual(saved[f"{PREFIX}/txt2img/{PAG_SCALE}/value"], 2)

        # The user now saves the old defaults and tau on purpose: left alone.
        saved[f"{PREFIX}/img2img/DCW lambda low/value"] = 0.1
        saved[f"{PREFIX}/img2img/DCW lambda high/value"] = 0.02
        saved[f"{PREFIX}/img2img/{CWM_LOW}/value"] = 0.3
        saved[f"{PREFIX}/img2img/{TAU}/value"] = 0.15
        before = dict(saved)
        self.assertEqual(migration.migrate_ui_settings(saved), [])
        self.assertEqual(saved, before)

    def test_fresh_install_is_untouched(self):
        self._write({})
        for tab in migration.TABS:
            _inputs, loadsave = self._forge_ui(tab)
            loadsave.dump_defaults()
        before = self.path.read_bytes()
        self.assertEqual(migration.migrate_ui_config_file(self.path), [])
        self.assertEqual(self.path.read_bytes(), before)

    # -- file handling -------------------------------------------------------

    def test_backup_keeps_the_original_bytes_and_output_has_no_bom(self):
        self._write(_installed_like())
        original = self.path.read_bytes()
        changes = migration.migrate_ui_config_file(self.path)
        backups = list(self.path.parent.glob("ui-config.json.bak-anima-guidance-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        self.assertIn(f"backup: {backups[0]}", changes)
        self.assertFalse(self.path.read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertFalse(list(self.path.parent.glob("*.tmp-anima-guidance")))

    def test_missing_or_broken_file_is_left_alone(self):
        self.assertEqual(migration.migrate_ui_config_file(self.path), [])
        self.assertFalse(self.path.exists())
        self.path.write_text("{ not json", encoding="utf-8")
        self.assertEqual(migration.migrate_ui_config_file(self.path), [])
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{ not json")

    def test_before_ui_hook_migrates_the_configured_file(self):
        """on_before_ui runs before ui.create_ui() builds UiLoadsave (webui.py, modules/ui.py:866)."""
        self._write(_installed_like())
        shared = types.SimpleNamespace(cmd_opts=types.SimpleNamespace(ui_config_file=str(self.path)))
        with mock.patch.object(self.pag, "shared", shared), \
                mock.patch.object(self.pag, "_make_pag_xyz_axis"), \
                mock.patch.object(self.pag, "_log") as log:
            self.pag._pag_on_before_ui()
        self.assertEqual(self._read()[f"{PREFIX}/txt2img/{TAU}/value"], 0.0)
        self.assertIn("ui-config.json migrated", log.call_args_list[0].args[0])

    def test_before_ui_hook_survives_a_failing_migration(self):
        with mock.patch.object(self.pag, "migrate_ui_config_file", side_effect=OSError("locked")), \
                mock.patch.object(self.pag, "shared", types.SimpleNamespace(
                    cmd_opts=types.SimpleNamespace(ui_config_file=str(self.path)))), \
                mock.patch.object(self.pag, "_make_pag_xyz_axis") as axis, \
                mock.patch.object(self.pag, "_log") as log:
            self.pag._pag_on_before_ui()
        axis.assert_called_once()
        self.assertIn("ui-config migration failed", log.call_args_list[0].args[0])



EXTENSION_ROOT = Path(__file__).resolve().parents[1]
SKIM = f"customscript/{migration.SKIMMED_SCRIPT_FILE}"
FLIP = migration.SKIMMED_FLIP_LABEL
NEGATIVE = f"txt2img/{migration.TILE_REPAIR_NEGATIVE_LABEL}/value"


def _skimmed_legacy(tab):
    # forge_sam3_extension@4d0028e:scripts/anima_skimmed_cfg.py:399-403 as Forge saved it.
    return {f"{SKIM}/{tab}/{FLIP}/{field}": value for field, value in (
        ("visible", True), ("value", 0.3), ("minimum", 0.0), ("maximum", 1.0), ("step", 0.05))}


def _tile_repair_legacy(negative="blurry, low quality"):
    # forge_sam3_extension@4d0028e:sam3ext/ui_anima.py as Forge saved it on the txt2img tab.
    saved = {NEGATIVE: negative, "txt2img/SAM3 Anima Negative/visible": True}
    for label in ("SAM3 Anima Width", "SAM3 Anima Height"):
        for field, value in (("visible", True), ("value", 1024), ("minimum", 256),
                             ("maximum", 4096), ("step", 32)):
            saved[f"txt2img/{label}/{field}"] = value
    return saved


class OtherPanelMigrationTests(unittest.TestCase):
    """Skimmed CFG ``Flip at`` bounds and the Tile-Repair negative default (plain dict rules)."""

    def test_labels_and_new_values_are_the_extension_sources(self):
        skimmed = (EXTENSION_ROOT / "scripts" / "anima_skimmed_cfg.py").read_text(encoding="utf-8")
        flip = skimmed[skimmed.index(f'label="{FLIP}"'):][:200]
        self.assertIn("step=0.01", flip)
        panel = (EXTENSION_ROOT / "sam3ext" / "ui_anima.py").read_text(encoding="utf-8")
        negative = panel[panel.index(f'label="{migration.TILE_REPAIR_NEGATIVE_LABEL}"'):][:120]
        self.assertIn('value=""', negative)
        for label in ("SAM3 Anima Width", "SAM3 Anima Height"):
            self.assertNotIn(f'label="{label}"', panel)   # replaced: the old keys are never read

    def test_skimmed_legacy_step_drops_the_bounds_and_keeps_the_value(self):
        saved = {**_skimmed_legacy("txt2img"), **_skimmed_legacy("img2img")}
        changes = migration.migrate_ui_settings(saved)
        self.assertEqual(len(changes), 2)
        for tab in migration.TABS:
            with self.subTest(tab=tab):
                for field in ("minimum", "maximum", "step"):
                    self.assertNotIn(f"{SKIM}/{tab}/{FLIP}/{field}", saved)
                self.assertEqual(saved[f"{SKIM}/{tab}/{FLIP}/value"], 0.3)
        self.assertEqual(migration.migrate_ui_settings(saved), [])

    def test_skimmed_new_step_is_untouched(self):
        saved = _skimmed_legacy("txt2img")
        saved[f"{SKIM}/txt2img/{FLIP}/step"] = 0.01
        before = dict(saved)
        self.assertEqual(migration.migrate_ui_settings(saved), [])
        self.assertEqual(saved, before)

    def test_tile_repair_old_default_negative_becomes_empty_once(self):
        saved = _tile_repair_legacy()
        changes = migration.migrate_ui_settings(saved)
        self.assertEqual(saved[NEGATIVE], "")
        self.assertFalse([key for key in saved if "Anima Width" in key or "Anima Height" in key])
        self.assertEqual(len(changes), 2)
        # Saved on purpose afterwards: the marker is gone, so it stays.
        saved[NEGATIVE] = "blurry, low quality"
        self.assertEqual(migration.migrate_ui_settings(saved), [])
        self.assertEqual(saved[NEGATIVE], "blurry, low quality")

    def test_tile_repair_user_negative_stays(self):
        saved = _tile_repair_legacy("lowres, jpeg artifacts")
        changes = migration.migrate_ui_settings(saved)
        self.assertEqual(saved[NEGATIVE], "lowres, jpeg artifacts")
        self.assertEqual(len(changes), 1)   # only the replaced slider keys

    def test_tile_repair_without_the_legacy_sliders_is_untouched(self):
        saved = {NEGATIVE: "blurry, low quality"}
        self.assertEqual(migration.migrate_ui_settings(saved), [])
        self.assertEqual(saved[NEGATIVE], "blurry, low quality")

    def test_tile_repair_step_only_runs_for_the_txt2img_tab(self):
        saved = _tile_repair_legacy()
        self.assertEqual(migration.migrate_ui_settings(saved, tabs=iter(["img2img"])), [])
        self.assertEqual(migration.migrate_ui_settings(saved, tabs=iter(["txt2img"]))[-1:],
                         ["txt2img/SAM3 Anima Negative: old default 'blurry, low quality' -> sd-scripts default ''"])


if __name__ == "__main__":
    unittest.main()
