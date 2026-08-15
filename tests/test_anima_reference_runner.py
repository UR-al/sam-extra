from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from sam3ext.anima_reference_core import ReferenceCanvasConfig
from sam3ext.anima_reference_runner import (
    ReferenceGenerationRequest,
    build_processing_args,
    candidate_seed,
    native_anima_reference,
    run_anima_reference,
)

FORGE_ROOT = Path(__file__).resolve().parents[3]
_RUN_FORGE_INTEGRATION = (
    os.environ.get("SAM3_RUN_FORGE_INTEGRATION_TESTS") == "1"
)
if (
    _RUN_FORGE_INTEGRATION
    and (FORGE_ROOT / "modules" / "processing.py").is_file()
):
    sys.path.insert(0, str(FORGE_ROOT))
_HAS_FORGE_RUNTIME = (
    _RUN_FORGE_INTEGRATION
    and
    importlib.util.find_spec("modules") is not None
    and importlib.util.find_spec("modules.processing") is not None
)


class ReferenceRequestTests(unittest.TestCase):
    def test_candidate_seed_preserves_random_and_steps_fixed_seed(self):
        self.assertEqual(candidate_seed(-1, 5, 3), -1)
        self.assertEqual(candidate_seed(100, 5, 3), 115)

    def test_processing_mapping_keeps_reference_panel_visible(self):
        request = ReferenceGenerationRequest(
            reference_image=Image.new("RGB", (320, 480)),
            steps=41,
            cfg_scale=4.5,
            shift=3.25,
            denoising_strength=0.93,
            mask_blur=7,
            mask_invert=True,
            initial_noise_multiplier=0.8,
        )
        mapped = build_processing_args(request, seed=123)

        self.assertFalse(mapped["sam3_inpaint_only_masked"])
        self.assertFalse(mapped["sam3_use_inpaint_width_height"])
        self.assertEqual(mapped["sam3_steps"], 41)
        self.assertEqual(mapped["sam3_seed"], 123)
        self.assertEqual(mapped["sam3_mask_blur"], 7)
        self.assertTrue(mapped["sam3_mask_invert"])

    def test_invalid_sampler_values_fail_before_forge_sampling(self):
        request = ReferenceGenerationRequest(
            reference_image=Image.new("RGB", (64, 64)),
            candidate_count=0,
        )
        with self.assertRaisesRegex(ValueError, "Candidate count"):
            request.validate()

    def test_native_reference_context_restores_option_and_clears_latents(self):
        class Dynamic:
            ref_latents = ["stale"]
            is_referencing = True

        model = _Anima()
        options = {"anima_do_reference": False}
        with native_anima_reference(
            True,
            model,
            opts_data=options,
            dynamic_args_obj=Dynamic,
        ):
            self.assertTrue(options["anima_do_reference"])
            self.assertEqual(Dynamic.ref_latents, [])
            self.assertFalse(Dynamic.is_referencing)
            Dynamic.ref_latents.append("generated")

        self.assertFalse(options["anima_do_reference"])
        self.assertEqual(Dynamic.ref_latents, [])
        self.assertGreaterEqual(model.clear_count, 2)


class _Anima:
    text_processing_engine_anima = object()

    def __init__(self):
        self.clear_count = 0

    def clear_references(self):
        self.clear_count += 1


class _FakeProcessing:
    def __init__(self):
        self.prompt = ""
        self.negative_prompt = ""
        self.image_mask = None
        self.distilled_cfg_scale = 0.0
        self.mask_round = None
        self.inpainting_mask_weight = None
        self.eta = None
        self.s_min_uncond = None
        self.s_churn = None
        self.s_tmin = None
        self.s_tmax = None
        self.s_noise = None
        self.override_settings = {}
        self.do_not_save_samples = False
        self.do_not_save_grid = False
        self.extra_generation_params = {}
        self.all_seeds = [321]
        self.closed = False

    def close(self):
        self.closed = True


@unittest.skipUnless(
    _HAS_FORGE_RUNTIME,
    "requires a Forge Neo checkout around the extension",
)
class ReferenceForgeAdapterTests(unittest.TestCase):
    def test_native_option_is_scoped_and_result_is_exactly_cropped(self):
        # Forge parses ``sys.argv`` while importing shared_cmd_options.  The
        # unittest discovery arguments are not Forge flags, so hide them only
        # for this optional installed-runtime import.
        argv = list(sys.argv)
        try:
            sys.argv[:] = [sys.argv[0]]
            from modules import processing, shared
            from sam3ext import inpaint_core
        finally:
            sys.argv[:] = argv

        model = _Anima()
        fake_p = _FakeProcessing()
        captured = {}

        def fake_build(image, args, **kwargs):
            captured["canvas"] = image
            captured["args"] = args
            return fake_p

        def fake_process(p):
            self.assertTrue(shared.opts.data["anima_do_reference"])
            self.assertIs(p, fake_p)
            return type(
                "Processed",
                (),
                {
                    "images": [captured["canvas"].copy()],
                    "infotexts": ["Seed: 321"],
                    "info": "",
                },
            )()

        request = ReferenceGenerationRequest(
            reference_image=Image.new("RGB", (320, 640), "red"),
            canvas=ReferenceCanvasConfig(
                output_width=832,
                output_height=1216,
                composite_megapixels=0,
            ),
            edit_lora_enabled=False,
            extend_lora_enabled=False,
            save_target=False,
            seed=321,
        )
        previous = shared.opts.data.get("anima_do_reference")
        previous_model = shared.sd_model
        interrupted = shared.state.interrupted
        skipped = shared.state.skipped
        job_count = shared.state.job_count
        job = shared.state.job
        try:
            shared.sd_model = model
            shared.state.interrupted = False
            shared.state.skipped = False
            with (
                patch.object(
                    inpaint_core,
                    "build_standalone_i2i",
                    side_effect=fake_build,
                ),
                patch.object(processing, "process_images", side_effect=fake_process),
            ):
                result = run_anima_reference(request, sd_model=model)
        finally:
            shared.sd_model = previous_model
            shared.state.interrupted = interrupted
            shared.state.skipped = skipped
            shared.state.job_count = job_count
            shared.state.job = job

        self.assertEqual(shared.opts.data.get("anima_do_reference"), previous)
        self.assertTrue(fake_p.closed)
        self.assertFalse(captured["args"]["sam3_inpaint_only_masked"])
        self.assertEqual(result.outputs[0].target_image.size, (832, 1216))
        self.assertEqual(result.outputs[0].seed, 321)
        self.assertEqual(
            fake_p.extra_generation_params["SAM3 Feature"],
            "6 - Anima Character Reference",
        )


if __name__ == "__main__":
    unittest.main()
