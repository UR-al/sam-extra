from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

from PIL import Image

from sam3ext import anima_reference_runner as runner_module
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
    def test_the_default_mode_is_the_existing_split_screen_path(self):
        request = ReferenceGenerationRequest(reference_image=Image.new("RGB", (8, 8)))
        self.assertEqual(request.mode, "i2i")
        request.validate()

    def test_an_unknown_mode_is_refused(self):
        request = ReferenceGenerationRequest(
            reference_image=Image.new("RGB", (8, 8)), mode="magic"
        )
        with self.assertRaises(ValueError):
            request.validate()

    def test_ipa_options_are_validated_only_in_ipa_mode(self):
        from sam3ext.anima_ipa.options import IpaOptions

        bad = IpaOptions(ref_size=500)
        # i2i 모드에서는 IPA 값을 보지 않는다.
        ReferenceGenerationRequest(
            reference_image=Image.new("RGB", (8, 8)), ipa=bad
        ).validate()
        with self.assertRaises(ValueError):
            ReferenceGenerationRequest(
                reference_image=Image.new("RGB", (8, 8)), mode="ipa", ipa=bad
            ).validate()

    def test_ipa_mode_drops_the_split_screen_prefix_and_edit_lora(self):
        request = ReferenceGenerationRequest(
            reference_image=Image.new("RGB", (8, 8)),
            mode="ipa",
            prompt="1girl, standing",
            edit_lora_name="AnimeEditV2",
        )
        self.assertEqual(request.sampling_prompt, "1girl, standing")
        # 이어붙이기 쪽 프롬프트는 그대로 남아 있다(모드만 다르다).
        self.assertIn("split screen", request.internal_prompt)

    def test_i2i_mode_still_samples_the_composed_prompt(self):
        request = ReferenceGenerationRequest(
            reference_image=Image.new("RGB", (8, 8)), prompt="1girl"
        )
        self.assertEqual(request.sampling_prompt, request.internal_prompt)

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
            initial_noise_multiplier=0.8,
        )
        mapped = build_processing_args(request, seed=123)

        self.assertFalse(mapped["sam3_inpaint_only_masked"])
        self.assertFalse(mapped["sam3_use_inpaint_width_height"])
        self.assertEqual(mapped["sam3_steps"], 41)
        self.assertEqual(mapped["sam3_seed"], 123)
        self.assertEqual(mapped["sam3_mask_blur"], 7)
        self.assertFalse(mapped["sam3_mask_invert"])
        self.assertFalse(mapped["sam3_restore_face"])
        self.assertEqual(mapped["sam3_resize_mode"], "Just Resize")

    def test_default_request_runs_the_restyler_recipe(self):
        request = ReferenceGenerationRequest(reference_image=Image.new("RGB", (64, 64)))
        mapped = build_processing_args(request, seed=1)

        self.assertEqual(mapped["sam3_inpainting_fill"], "original")
        self.assertEqual(mapped["sam3_denoising_strength"], 1.0)
        self.assertEqual(request.canvas.target_color, "#000000")
        self.assertEqual(
            request.internal_prompt,
            "<lora:AnimeEditV2:0.72>, (split screen, multiple views:1.2)",
        )
        self.assertEqual(request.candidate_count, 2)

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


class _FakeState:
    def __init__(self):
        self.interrupted = False
        self.skipped = False
        self.job_count = 0
        self.job = ""
        self.textinfo = ""


class _ProcessingDouble:
    """Fields of Forge's img2img processing object that the runner touches."""

    UNSET = object()

    def __init__(self, sd_model, init_canvas, args):
        self.sd_model = sd_model
        self.init_canvas = init_canvas
        self.args = args
        self.prompt = ""
        self.negative_prompt = ""
        self.image_mask = None
        self.distilled_cfg_scale = 0.0
        self.eta = _ProcessingDouble.UNSET
        self.s_min_uncond = _ProcessingDouble.UNSET
        self.do_not_save_samples = False
        self.do_not_save_grid = False
        self.extra_generation_params = {}
        self.all_seeds = [321]
        self.closed = False

    def close(self):
        self.closed = True


class _T2IProcessingDouble:
    """IPA 경로가 만지는 txt2img processing 필드. ``sample()`` 은 Forge 가 부른다."""

    def __init__(self, sd_model, request, on_sample=None):
        self.sd_model = sd_model
        self.request = request
        self.on_sample = on_sample
        self.prompt = ""
        self.negative_prompt = ""
        self.distilled_cfg_scale = 0.0
        self.do_not_save_samples = True
        self.do_not_save_grid = True
        self.extra_generation_params = {}
        self.all_seeds = [321]
        self.sampled = 0
        self.session_active_during_sample = None

    def sample(self, *args, **kwargs):
        self.sampled += 1
        # 주입 컨텍스트 안에서 불리는지 여기서 확인한다(바깥에서 읽으면 이미 풀린 뒤다).
        if self.on_sample is not None:
            self.on_sample(self)
        return "latent"


class _FakeForge:
    """Just enough of Forge for run_anima_reference, swapped into sys.modules."""

    def __init__(self, model, process=None, selected=None):
        self.model = model
        self.selected = selected   # what forge_model_reload() loads (None: nothing new)
        self.reloads = 0
        self.built = []
        self.saved = []
        self.state = _FakeState()
        self.on_sample = None       # IPA 테스트가 sample() 안에서 확인할 때 쓴다
        self.stitch = type("ImageStitch", (), {"cached_parameters": [1, 2]})
        self._process = process or self.processed_for

    def processed_for(self, p):
        return types.SimpleNamespace(
            images=[p.init_canvas.copy()], infotexts=["Seed: 321"], info=""
        )

    def _modules(self):
        package = types.ModuleType("modules")
        package.__path__ = []
        shared = types.ModuleType("modules.shared")
        shared.opts = types.SimpleNamespace(
            data={},
            samples_format="png",
            outdir_txt2img_samples="out",
            outdir_txt2img_grids="grids",
        )
        shared.state = self.state
        shared.sd_model = self.model
        processing = types.ModuleType("modules.processing")
        processing.process_images = self._process
        images = types.ModuleType("modules.images")

        def save_image(image, path, basename, seed, prompt, fmt, **kwargs):
            self.saved.append(kwargs.get("suffix"))

        images.save_image = save_image
        scripts = types.ModuleType("modules.scripts")
        scripts.scripts_data = [types.SimpleNamespace(script_class=self.stitch)]
        sd_models = types.ModuleType("modules.sd_models")

        def forge_model_reload():
            self.reloads += 1
            if self.selected is not None:
                shared.sd_model = self.selected
            return shared.sd_model, self.selected is not None

        sd_models.forge_model_reload = forge_model_reload
        for name, module in (
            ("shared", shared),
            ("processing", processing),
            ("images", images),
            ("scripts", scripts),
            ("sd_models", sd_models),
        ):
            setattr(package, name, module)
        inpaint = types.ModuleType("sam3ext.inpaint_core")

        def build(image, args, **kwargs):
            p = _ProcessingDouble(kwargs["sd_model"], image, args)
            self.built.append(p)
            return p

        inpaint.build_standalone_i2i = build

        def build_t2i(request, **kwargs):
            p = _T2IProcessingDouble(
                kwargs["sd_model"], request, on_sample=self.on_sample
            )
            self.built.append(p)
            return p

        inpaint.build_standalone_t2i = build_t2i
        inpaint.pause_total_tqdm = contextlib.nullcontext
        return {
            "modules": package,
            "modules.shared": shared,
            "modules.processing": processing,
            "modules.images": images,
            "modules.scripts": scripts,
            "modules.sd_models": sd_models,
            "sam3ext.inpaint_core": inpaint,
        }

    def run(self, request, *, pass_model=True):
        with mock.patch.dict(sys.modules, self._modules()):
            if pass_model:
                return run_anima_reference(request, sd_model=self.model)
            return run_anima_reference(request)   # the UI handler's call


def _request(**overrides):
    values = dict(
        reference_image=Image.new("RGB", (832, 1216), "red"),
        canvas=ReferenceCanvasConfig(output_width=832, output_height=1216),
        candidate_count=1,
        save_target=False,
        seed=321,
    )
    values.update(overrides)
    return ReferenceGenerationRequest(**values)


class ReferenceRunnerBehaviourTests(unittest.TestCase):
    def setUp(self):
        # These tests don't exercise the Anima 3.8B connector.  Without this,
        # run_anima_reference calls the real _anima38_runtime(), which does a
        # real `import torch` inside forge.run()'s
        # mock.patch.dict(sys.modules, ...) block; that patch's __exit__
        # restores the *entire* sys.modules snapshot taken at __enter__,
        # evicting the newly-imported torch and corrupting its native state
        # for the next real import in the same process (intermittent
        # interpreter crash). Default-mock it so no test in this class does a
        # real import under the sys.modules patch unless it opts in, as
        # ReferenceAnima38Tests does below.
        patcher = mock.patch.object(
            runner_module, "_anima38_runtime", return_value=(None, "not tested")
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_sampler_extras_keep_forge_settings_unless_requested(self):
        forge = _FakeForge(_Anima())
        forge.run(_request())
        p = forge.built[0]
        self.assertIs(p.eta, _ProcessingDouble.UNSET)
        self.assertIs(p.s_min_uncond, _ProcessingDouble.UNSET)

        forge = _FakeForge(_Anima())
        forge.run(_request(eta=0.5, s_min_uncond=0.25))
        p = forge.built[0]
        self.assertEqual((p.eta, p.s_min_uncond), (0.5, 0.25))

    def test_lazily_loaded_checkpoint_is_loaded_before_the_anima_check(self):
        # Right after a Forge restart shared.sd_model is FakeInitialModel until
        # the first generation loads the selected checkpoint.
        forge = _FakeForge(object(), selected=_Anima())
        forge.run(_request(), pass_model=False)
        self.assertEqual(forge.reloads, 1)
        self.assertIs(forge.built[0].sd_model, forge.selected)

    def test_selected_checkpoint_wins_over_the_stale_loaded_one(self):
        # The dropdown now points at a non-Anima checkpoint; the old Anima is still in memory.
        forge = _FakeForge(_Anima(), selected=object())
        with self.assertRaisesRegex(RuntimeError, "Anima"):
            forge.run(_request(), pass_model=False)
        self.assertEqual(forge.built, [])

    def test_explicit_model_skips_the_reload(self):
        forge = _FakeForge(_Anima(), selected=object())
        forge.run(_request())
        self.assertEqual(forge.reloads, 0)

    def test_non_anima_model_is_rejected_before_sampling(self):
        forge = _FakeForge(object())
        with self.assertRaisesRegex(RuntimeError, "Anima"):
            forge.run(_request())
        self.assertEqual(forge.built, [])

    def test_interrupted_candidate_is_neither_saved_nor_returned(self):
        forge = None

        def interrupt(p):
            forge.state.interrupted = True
            return forge.processed_for(p)

        forge = _FakeForge(_Anima(), process=interrupt)
        result = forge.run(_request(save_target=True, candidate_count=2))

        self.assertEqual(result.outputs, ())
        self.assertEqual(forge.saved, [])
        self.assertEqual(len(forge.built), 1)
        self.assertTrue(result.diagnostics["interrupted"])

    def test_image_stitch_cache_is_invalidated_after_a_run(self):
        forge = _FakeForge(_Anima())
        forge.run(_request())
        self.assertIsNone(forge.stitch.cached_parameters)

    def test_diagnostics_and_infotext_describe_the_real_panel(self):
        # 832x1216 reference and output at 1.4 MP:
        # canvas height 1008, each panel 688 wide -> upscale 1216/1008 = 1.21
        forge = _FakeForge(_Anima())
        result = forge.run(_request(keep_scope="outfit", sampling_source="forge_preset"))

        self.assertEqual(result.diagnostics["target_panel"], (688, 1008))
        self.assertEqual(result.diagnostics["upscale"], 1.21)
        self.assertIsNone(result.diagnostics["model_blocks"])
        params = forge.built[0].extra_generation_params
        self.assertEqual(params["Reference masked content"], "original")
        self.assertEqual(params["Reference target panel"], "688x1008")
        self.assertEqual(params["Reference upscale"], "1.21")
        self.assertEqual(params["Reference keep"], "outfit")
        self.assertEqual(params["Reference sampling"], "forge_preset")
        self.assertEqual(params["Reference model blocks"], "unknown")

    def test_model_block_count_reads_the_diffusion_model(self):
        model = _Anima()
        model.forge_objects = types.SimpleNamespace(
            unet=types.SimpleNamespace(
                model=types.SimpleNamespace(
                    diffusion_model=types.SimpleNamespace(blocks=[object()] * 52)
                )
            )
        )
        forge = _FakeForge(model)
        result = forge.run(_request())
        self.assertEqual(result.diagnostics["model_blocks"], 52)
        self.assertEqual(
            forge.built[0].extra_generation_params["Reference model blocks"], "52"
        )


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


class _FakeAnima38:
    def __init__(self, v2=True, error=None, v2_error=None):
        self.v2 = v2
        self.error = error
        self.v2_error = v2_error
        self.calls = []

    def is_v2_bundle(self, model):
        if self.v2_error is not None:
            raise self.v2_error
        return self.v2

    def install(self, processing, adapter, strength, negative):
        self.calls.append(("install", processing, adapter, strength, negative))
        if self.error is not None:
            raise self.error

    def restore(self, processing):
        self.calls.append(("restore", processing))


class ReferenceAnima38Tests(unittest.TestCase):
    def _run(self, runtime, error=None, encoder_present=True):
        forge = None

        def process(p):
            if runtime is not None:
                runtime.calls.append(("process", p))
            return forge.processed_for(p)

        forge = _FakeForge(_Anima(), process=process)
        with (
            mock.patch.object(
                runner_module, "_anima38_runtime", return_value=(runtime, error)
            ),
            mock.patch.object(
                runner_module,
                "_anima38_encoder_present",
                return_value=encoder_present,
            ),
        ):
            result = forge.run(_request())
        return forge, result

    def test_v2_bundle_gets_the_connector_around_sampling(self):
        runtime = _FakeAnima38()
        forge, result = self._run(runtime)
        p = forge.built[0]
        self.assertEqual(
            runtime.calls,
            [
                ("install", p, "Anima-3.8B-expanded_adapter.safetensors", 1.0, None),
                ("process", p),
                ("restore", p),
            ],
        )
        self.assertEqual(result.diagnostics["anima38"], "v2 bundle")
        self.assertEqual(p.extra_generation_params["Reference Anima38"], "v2 bundle")

    def test_non_bundle_model_runs_natively(self):
        runtime = _FakeAnima38(v2=False)
        forge, result = self._run(runtime)
        self.assertEqual(runtime.calls, [("process", forge.built[0])])
        self.assertEqual(result.diagnostics["anima38"], "not a 3.8B v2 bundle")

    def test_install_failure_falls_back_and_restores(self):
        # F3: a failed install must restore before sampling, not just skip it.
        runtime = _FakeAnima38(error=FileNotFoundError("qwen35_4b not found"))
        forge, result = self._run(runtime)
        p = forge.built[0]
        self.assertEqual(
            [call[0] for call in runtime.calls], ["install", "restore", "process"]
        )
        self.assertEqual(result.diagnostics["anima38"], "missing encoder (qwen35_4b not found)")
        self.assertEqual(len(result.outputs), 1)

    def test_unavailable_runtime_is_reported(self):
        forge, result = self._run(None, error="ImportError: no torch")
        self.assertEqual(result.diagnostics["anima38"], "unavailable (ImportError: no torch)")
        self.assertEqual(len(result.outputs), 1)

    def test_missing_encoder_precheck_skips_install(self):
        # F2: without qwen35_4b, install() itself is never reached.
        runtime = _FakeAnima38()
        forge, result = self._run(runtime, encoder_present=False)
        self.assertEqual([call[0] for call in runtime.calls], ["process"])
        self.assertEqual(
            result.diagnostics["anima38"],
            "missing encoder (qwen35_4b.safetensors not found in models/text_encoder)",
        )
        self.assertEqual(len(result.outputs), 1)

    def test_check_failed_label_is_reported_and_logged(self):
        # F11: an is_v2_bundle exception must be labelled and logged, like install failures.
        runtime = _FakeAnima38(v2_error=RuntimeError("boom"))
        with mock.patch("sys.stderr", new_callable=io.StringIO) as fake_stderr:
            forge, result = self._run(runtime)
        self.assertEqual([call[0] for call in runtime.calls], ["process"])
        self.assertEqual(result.diagnostics["anima38"], "check failed (RuntimeError)")
        self.assertIn("Anima 3.8B check failed", fake_stderr.getvalue())

    def test_generic_install_failure_falls_back_and_restores(self):
        # F3 + F11: any install() exception (not just FileNotFoundError) restores.
        runtime = _FakeAnima38(error=RuntimeError("boom"))
        with mock.patch("sys.stderr", new_callable=io.StringIO) as fake_stderr:
            forge, result = self._run(runtime)
        self.assertEqual(
            [call[0] for call in runtime.calls], ["install", "restore", "process"]
        )
        self.assertEqual(result.diagnostics["anima38"], "install failed (RuntimeError: boom)")
        self.assertIn("Anima 3.8B install failed", fake_stderr.getvalue())
        self.assertEqual(len(result.outputs), 1)


class ReferenceIpaPathTests(unittest.TestCase):
    """IPA 모드는 캔버스를 만들지 않고 txt2img 잡을 돌린다.

    주의: ``mock.patch.dict(sys.modules, ...)`` 안에서 torch 를 처음 import 하면 인터프리터가
    깨진다(이 파일 위쪽 ReferenceRunnerBehaviourTests.setUp 의 기록). 그래서 주입 세션은
    통째로 가짜로 막는다 — 이 테스트는 배선만 본다.
    """

    def setUp(self):
        self.entered = {"active": False}
        self.sessions = []

        @contextlib.contextmanager
        def session(sd_model, image, options):
            self.sessions.append(
                {"sd_model": sd_model, "image": image, "options": options}
            )
            self.entered["active"] = True
            try:
                yield
            finally:
                self.entered["active"] = False

        self.session_factory = session

    def _record_during_sample(self, p):
        p.session_active_during_sample = self.entered["active"]

    def _process(self, p):
        # Forge 가 하는 일: sample() 을 부른다. 주입은 그 안에서만 살아 있어야 한다.
        p.sample()
        return types.SimpleNamespace(
            images=[Image.new("RGB", (64, 64), "blue")],
            infotexts=["Seed: 321"],
            info="",
        )

    def _run(self, model=None, **overrides):
        from sam3ext.anima_ipa.options import IpaOptions

        model = model or _Anima()
        forge = _FakeForge(model, process=self._process)
        forge.on_sample = self._record_during_sample
        values = dict(
            reference_image=Image.new("RGB", (512, 768), "red"),
            mode="ipa",
            ipa=IpaOptions(),
            candidate_count=1,
            save_target=False,
            seed=7,
        )
        values.update(overrides)
        request = ReferenceGenerationRequest(**values)
        with mock.patch.object(runner_module, "_ipa_session", self.session_factory):
            with mock.patch.dict(sys.modules, forge._modules()):
                result = run_anima_reference(request, sd_model=model)
        return forge, result

    def test_no_canvas_is_built_and_the_result_is_not_cropped(self):
        forge, result = self._run()
        self.assertEqual(len(result.outputs), 1)
        self.assertEqual(result.outputs[0].target_image.size, (64, 64))
        self.assertIsNone(result.prepared, "IPA 에는 캔버스도 마스크도 없다")

    def test_the_session_wraps_sampling(self):
        forge, _ = self._run()
        self.assertEqual(len(self.sessions), 1)
        self.assertEqual(self.sessions[0]["image"].size, (512, 768))
        self.assertTrue(forge.built[0].session_active_during_sample)
        self.assertFalse(self.entered["active"], "잡이 끝나면 주입이 풀려야 한다")

    def test_the_prompt_has_no_split_screen_prefix(self):
        forge, _ = self._run(prompt="1girl, standing")
        self.assertEqual(forge.built[0].prompt, "1girl, standing")

    def test_the_infotext_records_the_mode_and_strength(self):
        from sam3ext.anima_ipa.options import IpaOptions

        forge, result = self._run(ipa=IpaOptions(strength=0.75))
        params = forge.built[0].extra_generation_params
        self.assertEqual(params["SAM3 Feature"], 6)
        self.assertEqual(params["SAM3 Reference Mode"], "ipa")
        self.assertEqual(params["SAM3 IPA Strength"], 0.75)
        self.assertNotIn("SAM3 IPA CFG", params, "1-pass 에서는 IP CFG 가 의미 없다")
        self.assertEqual(result.diagnostics.get("mode"), "ipa")

    def test_two_pass_cfg_is_recorded_when_it_is_on(self):
        from sam3ext.anima_ipa.options import IpaOptions

        forge, _ = self._run(ipa=IpaOptions(separate_cfg=True, cfg_scale=5.5))
        self.assertEqual(forge.built[0].extra_generation_params["SAM3 IPA CFG"], 5.5)

    def test_candidates_get_stepped_seeds(self):
        forge, result = self._run(candidate_count=3, seed=100, seed_step=5)
        self.assertEqual(len(result.outputs), 3)
        self.assertEqual(len(self.sessions), 3)

    def test_a_non_anima_model_is_refused_before_any_job_runs(self):
        model = types.SimpleNamespace(is_anima=False)
        with self.assertRaises(RuntimeError) as caught:
            self._run(model=model)
        self.assertIn("Anima", str(caught.exception))

    def test_a_block_lineage_mapping_is_recorded_where_the_user_can_see_it(self):
        """28블록 어댑터가 52블록 모델에 얹히면 결과에 그 사실이 남아야 한다."""
        model = _Anima()
        model.forge_objects = types.SimpleNamespace(
            unet=types.SimpleNamespace(
                model=types.SimpleNamespace(
                    diffusion_model=types.SimpleNamespace(blocks=[None] * 52)
                )
            )
        )

        @contextlib.contextmanager
        def session(sd_model, image, options):
            yield 28        # 실제 런타임은 어댑터 깊이를 내놓는다

        forge = _FakeForge(model, process=self._process)
        forge.on_sample = self._record_during_sample
        request = ReferenceGenerationRequest(
            reference_image=Image.new("RGB", (64, 64)), mode="ipa", candidate_count=1,
            save_target=False,
        )
        with mock.patch.object(runner_module, "_ipa_session", session):
            with mock.patch.dict(sys.modules, forge._modules()):
                result = run_anima_reference(request, sd_model=model)

        self.assertEqual(forge.built[0].extra_generation_params["SAM3 IPA Blocks"], "28→52")
        self.assertEqual(result.diagnostics.get("ipa_adapter_blocks"), 28)
        self.assertEqual(result.diagnostics.get("model_blocks"), 52)

    def test_the_same_depth_leaves_no_mapping_note(self):
        forge, result = self._run()
        self.assertNotIn("SAM3 IPA Blocks", forge.built[0].extra_generation_params)
        self.assertNotIn("SAM3 IPA Duplicates", forge.built[0].extra_generation_params)

    def _run_deep(self, *, setting=None, **request_overrides):
        """28블록 어댑터 + 52블록 모델. 세션이 받은 옵션도 돌려준다."""
        model = _Anima()
        model.forge_objects = types.SimpleNamespace(
            unet=types.SimpleNamespace(
                model=types.SimpleNamespace(
                    diffusion_model=types.SimpleNamespace(blocks=[None] * 52)
                )
            )
        )
        received = []

        @contextlib.contextmanager
        def session(sd_model, image, options):
            received.append(options)
            yield 28

        forge = _FakeForge(model, process=self._process)
        forge.on_sample = self._record_during_sample
        values = dict(
            reference_image=Image.new("RGB", (64, 64)), mode="ipa", candidate_count=2,
            save_target=False,
        )
        values.update(request_overrides)
        request = ReferenceGenerationRequest(**values)
        modules = forge._modules()
        if setting is not None:
            from sam3ext.anima_ipa.options import OPT_DUPLICATE_POLICY

            setattr(modules["modules.shared"].opts, OPT_DUPLICATE_POLICY, setting)
        with mock.patch.object(runner_module, "_ipa_session", session):
            with mock.patch.dict(sys.modules, modules):
                result = run_anima_reference(request, sd_model=model)
        return forge, result, received

    def test_a_mapped_run_records_the_default_duplicate_policy(self):
        forge, result, received = self._run_deep()
        for built in forge.built:
            self.assertEqual(built.extra_generation_params["SAM3 IPA Duplicates"], "lineage")
        self.assertEqual(result.diagnostics.get("ipa_duplicate_policy"), "lineage")
        # 세션은 None 이 아니라 정해진 값을 받는다 — infotext 와 실제 주입이 같은 값을 쓴다.
        self.assertEqual([o.duplicate_policy for o in received], ["lineage", "lineage"])

    def test_the_forge_setting_picks_the_duplicate_policy(self):
        forge, result, received = self._run_deep(setting="split")
        self.assertEqual(forge.built[0].extra_generation_params["SAM3 IPA Duplicates"], "split")
        self.assertEqual(result.diagnostics.get("ipa_duplicate_policy"), "split")
        self.assertEqual({o.duplicate_policy for o in received}, {"split"})

    def test_the_request_policy_beats_the_forge_setting(self):
        from sam3ext.anima_ipa.options import IpaOptions

        forge, result, received = self._run_deep(
            setting="split", ipa=IpaOptions(duplicate_policy="all", strength=0.7)
        )
        self.assertEqual(forge.built[0].extra_generation_params["SAM3 IPA Duplicates"], "all")
        self.assertEqual({o.duplicate_policy for o in received}, {"all"})
        # 나머지 옵션은 그대로 간다
        self.assertEqual({o.strength for o in received}, {0.7})

    def test_an_interrupt_is_reported_in_the_diagnostics(self):
        model = _Anima()
        forge = _FakeForge(model, process=self._process)
        forge.state.interrupted = True
        request = ReferenceGenerationRequest(
            reference_image=Image.new("RGB", (64, 64)), mode="ipa", candidate_count=2
        )
        with mock.patch.object(runner_module, "_ipa_session", self.session_factory):
            with mock.patch.dict(sys.modules, forge._modules()):
                result = run_anima_reference(request, sd_model=model)
        self.assertTrue(result.diagnostics.get("interrupted"))
        self.assertEqual(result.outputs, ())


class _FakeAnima38WithRelease(_FakeAnima38):
    """설치 순서의 첫 단계(release_foreign_install)도 기록한다."""

    def release_foreign_install(self, processing, on_error=None):
        self.calls.append(("release", processing))
        return None


class ReferenceIpaAnima38Tests(unittest.TestCase):
    """IP-Adapter 방식의 3.8B v2 커넥터 토글(감사 M10) — 기본 켬(이어붙이기와 같은 설치), 끄면 예전 0.6B 조건.

    주의: ReferenceIpaPathTests 와 같은 이유로 주입 세션과 anima38 런타임은 가짜다(배선만 본다).
    """

    def _run(self, runtime, *, setting=None, anima38_connector=None, error=None,
             encoder_present=True, interrupt=False, shared_runtime=None):
        from sam3ext.anima_ipa.options import IpaOptions

        calls = runtime.calls if runtime is not None else []
        state = {"session": False}

        @contextlib.contextmanager
        def session(sd_model, image, options):
            calls.append(("session", sd_model))
            state["session"] = True
            try:
                yield
            finally:
                state["session"] = False

        forge = None

        def process(p):
            calls.append(("process", p))
            p.sample()
            if interrupt:
                forge.state.interrupted = True
            return types.SimpleNamespace(
                images=[Image.new("RGB", (64, 64), "blue")], infotexts=["Seed: 7"], info=""
            )

        model = _Anima()
        forge = _FakeForge(model, process=process)
        request = ReferenceGenerationRequest(
            reference_image=Image.new("RGB", (64, 64), "red"),
            mode="ipa",
            ipa=IpaOptions(anima38_connector=anima38_connector),
            candidate_count=1,
            save_target=False,
            seed=7,
        )
        modules = forge._modules()
        if setting is not None:
            setattr(modules["modules.shared"].opts, runner_module.OPT_IPA_ANIMA38, setting)
        if shared_runtime is not None:
            fake_module = types.ModuleType(runner_module._ANIMA38_RUNTIME_MODULE)
            fake_module._SHARED_RUNTIME = shared_runtime
            modules[runner_module._ANIMA38_RUNTIME_MODULE] = fake_module
        lookup = mock.Mock(return_value=(runtime, error))
        with (
            mock.patch.object(runner_module, "_ipa_session", session),
            mock.patch.object(runner_module, "_anima38_runtime", lookup),
            mock.patch.object(
                runner_module, "_anima38_encoder_present", return_value=encoder_present
            ),
            mock.patch.dict(sys.modules, modules),
        ):
            result = run_anima_reference(request, sd_model=model)
        return forge, result, lookup

    def test_the_default_installs_the_connector_like_the_canvas_mode(self):
        runtime = _FakeAnima38WithRelease()
        forge, result, lookup = self._run(runtime)
        lookup.assert_called_once()
        p = forge.built[0]
        self.assertEqual(
            [call[0] for call in runtime.calls],
            ["release", "install", "process", "session", "restore"],
        )
        self.assertEqual(p.extra_generation_params["Reference Anima38"], "v2 bundle")
        self.assertTrue(result.diagnostics["anima38_connector"])

    def test_the_toggle_off_leaves_the_infotext_as_before(self):
        runtime = _FakeAnima38WithRelease()
        forge, result, lookup = self._run(runtime, setting=False)
        lookup.assert_not_called()          # 꺼져 있으면 무거운 런타임을 불러오지도 않는다
        params = forge.built[0].extra_generation_params
        self.assertNotIn("Reference Anima38", params)
        self.assertFalse(any(key.startswith("Anima38") for key in params))
        self.assertEqual(
            [call[0] for call in runtime.calls], ["process", "session"],
            "설치·복원이 없다",
        )
        self.assertFalse(result.diagnostics["anima38_connector"])
        self.assertEqual(result.diagnostics["anima38"], runner_module.ANIMA38_IPA_OFF)

    def test_the_setting_installs_the_connector_around_the_ipa_sampling(self):
        runtime = _FakeAnima38WithRelease()
        forge, result, _lookup = self._run(runtime, setting=True)
        p = forge.built[0]
        self.assertEqual(
            runtime.calls,
            [
                ("release", p),
                ("install", p, "Anima-3.8B-expanded_adapter.safetensors", 1.0, None),
                ("process", p),
                ("session", p.sd_model),     # IP 주입은 커넥터가 설치된 채로 sample() 안에서 산다
                ("restore", p),
            ],
        )
        self.assertEqual(p.extra_generation_params["Reference Anima38"], "v2 bundle")
        self.assertTrue(result.diagnostics["anima38_connector"])
        self.assertEqual(result.diagnostics["anima38"], "v2 bundle")
        self.assertEqual(len(result.outputs), 1)

    def test_the_request_value_wins_over_the_setting(self):
        runtime = _FakeAnima38WithRelease()
        forge, result, _ = self._run(runtime, setting=False, anima38_connector=True)
        self.assertEqual(result.diagnostics["anima38"], "v2 bundle")
        runtime = _FakeAnima38WithRelease()
        forge, result, lookup = self._run(runtime, setting=True, anima38_connector=False)
        lookup.assert_not_called()
        self.assertNotIn("Reference Anima38", forge.built[0].extra_generation_params)

    def test_a_model_that_is_not_a_v2_bundle_samples_natively_and_says_so(self):
        runtime = _FakeAnima38WithRelease(v2=False)
        forge, result, _ = self._run(runtime, setting=True)
        self.assertEqual([call[0] for call in runtime.calls], ["release", "process", "session"])
        self.assertEqual(
            forge.built[0].extra_generation_params["Reference Anima38"], "not a 3.8B v2 bundle"
        )

    def test_a_failed_install_restores_before_sampling_and_the_job_continues(self):
        runtime = _FakeAnima38WithRelease(error=RuntimeError("boom"))
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            forge, result, _ = self._run(runtime, setting=True)
        self.assertEqual(
            [call[0] for call in runtime.calls],
            ["release", "install", "restore", "process", "session"],
        )
        self.assertEqual(result.diagnostics["anima38"], "install failed (RuntimeError: boom)")
        self.assertEqual(len(result.outputs), 1)

    def test_a_missing_encoder_skips_the_install(self):
        runtime = _FakeAnima38WithRelease()
        forge, result, _ = self._run(runtime, setting=True, encoder_present=False)
        self.assertEqual([call[0] for call in runtime.calls], ["release", "process", "session"])
        self.assertTrue(result.diagnostics["anima38"].startswith("missing encoder"))

    def test_an_unavailable_runtime_is_reported(self):
        forge, result, _ = self._run(None, setting=True, error="ImportError: no torch")
        self.assertEqual(result.diagnostics["anima38"], "unavailable (ImportError: no torch)")
        self.assertEqual(len(result.outputs), 1)

    def test_an_interrupted_candidate_still_restores(self):
        runtime = _FakeAnima38WithRelease()
        forge, result, _ = self._run(runtime, setting=True, interrupt=True)
        self.assertEqual(runtime.calls[-1], ("restore", forge.built[0]))
        self.assertEqual(result.outputs, ())

    def test_with_the_toggle_off_a_leftover_install_is_still_released(self):
        """txt2img 가 샘플링 중 예외로 끝나 남긴 v2 설치로 이 잡의 조건이 만들어지면 안 된다 —
        스크립트의 Bypass 와 같은 규칙(공용 release_foreign_install). 커넥터 설치는 하지 않는다."""
        shared_runtime = _FakeAnima38WithRelease()
        forge, result, lookup = self._run(None, setting=False, shared_runtime=shared_runtime)
        lookup.assert_not_called()
        self.assertEqual(shared_runtime.calls, [("release", forge.built[0])])
        self.assertEqual(result.diagnostics["anima38"], runner_module.ANIMA38_IPA_OFF)


class AnimaModelDetectionTests(unittest.TestCase):
    """_is_anima_engine — Forge 의 Anima 클래스는 이름으로, 그 밖의 모델(대역·래퍼)은 텍스트 엔진 속성으로 알아본다.
    Forge 2.29.2 는 그 속성을 text_processing_engine_qwen(Flux2·Krea2·Qwen-Image·Z-Image 공용)으로 바꿨다."""

    def test_new_forge_attribute_with_the_anima_engine(self):
        engine = types.SimpleNamespace(qwen_tokenizer=object(), t5_tokenizer=object())   # Qwen06Engine 처럼
        self.assertTrue(runner_module._is_anima_engine(types.SimpleNamespace(text_processing_engine_qwen=engine)))

    def test_other_models_on_the_shared_attribute_are_not_anima(self):
        engine = types.SimpleNamespace(tokenizer=object())   # Z-Image·Flux2 엔진처럼 T5 토크나이저가 없다
        self.assertFalse(runner_module._is_anima_engine(types.SimpleNamespace(text_processing_engine_qwen=engine)))

    def test_old_attribute_and_class_name_still_count(self):
        self.assertTrue(runner_module._is_anima_engine(types.SimpleNamespace(text_processing_engine_anima=object())))
        self.assertTrue(runner_module._is_anima_engine(type("Anima", (), {})()))
        self.assertFalse(runner_module._is_anima_engine(types.SimpleNamespace()))
        self.assertFalse(runner_module._is_anima_engine(None))


if __name__ == "__main__":
    unittest.main()
