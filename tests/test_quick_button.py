"""SAM3 빠른 버튼(🎯) — txt2img 갤러리 ✨ 옆에서 선택한 이미지에 SAM3 설정을 바로 돌린다.

✨(Forge txt2img_upscale_function)와 같은 규칙: 지금 txt2img 설정으로 processing 을 만들고 시드는 그 이미지
것, 결과는 Forge 의 hires_button_gallery_insert 설정대로 갤러리에 넣는다.
"""
from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext import quick_button as qb  # noqa: E402


class _Saved(types.SimpleNamespace):
    """Forge 갤러리 항목의 이미지 — 파일 경로(?쿼리 포함)를 가진다."""


def _gallery(*names):
    return [(_Saved(filename=f"C:/out/{name}.png?123", size=(64, 96)), "") for name in names]


class SelectionTests(unittest.TestCase):
    def test_empty_gallery_and_bad_index(self):
        self.assertIn("No image", qb.validate_selection([], 0, {}))
        self.assertIn("Bad image index", qb.validate_selection(_gallery("a"), 3, {"infotexts": ["a"]}))
        self.assertIn("Bad image index", qb.validate_selection(_gallery("a"), -1, {"infotexts": ["a"]}))

    def test_grid_and_control_images_are_refused_like_the_hires_button(self):
        geninfo = {"index_of_first_image": 1, "infotexts": ["grid", "a", "b"]}
        gallery = _gallery("grid", "a", "b", "control")
        self.assertIn("grid", qb.validate_selection(gallery, 0, geninfo))
        self.assertIn("grid", qb.validate_selection(gallery, 3, geninfo))
        self.assertIsNone(qb.validate_selection(gallery, 1, geninfo))

    def test_single_image_is_always_allowed(self):
        self.assertIsNone(qb.validate_selection(_gallery("a"), 0, {}))


class ArgumentTests(unittest.TestCase):
    def test_button_forces_sam3_on_in_both_argument_forms(self):
        self.assertEqual(qb.force_enabled((False, {"sam3_prompt": "face"}, 3)), (True, {"sam3_prompt": "face"}, 3))
        forced = qb.force_enabled(({"sam3_enable": False, "sam3_prompt": "face"},))
        self.assertEqual(forced, ({"sam3_enable": True, "sam3_prompt": "face"},))
        self.assertEqual(qb.force_enabled(()), (True,))

    def test_forcing_does_not_mutate_the_ui_state(self):
        state = {"sam3_enable": False}
        qb.force_enabled((state,))
        self.assertIs(state["sam3_enable"], False)

    def test_seed_comes_from_the_images_infotext(self):
        parse = lambda text, _skip: {"Seed": "123", "Variation seed": "7"}  # noqa: E731
        self.assertEqual(qb.seeds_from_infotext("…", parse), (123, 7))
        self.assertEqual(qb.seeds_from_infotext("…", lambda text, _skip: {}), (-1, -1))
        self.assertEqual(qb.seeds_from_infotext("…", lambda text, _skip: {"Seed": "nan?"}), (-1, -1))


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.gallery = _gallery("a", "b", "c")
        self.geninfo = {"infotexts": ["ia", "ib", "ic"], "seed": 1}
        self.result = Image.new("RGB", (64, 96))

    def test_replace_keeps_the_position(self):
        gallery, geninfo, selected = qb.place_result(self.gallery, 1, self.geninfo, self.result, "ir", insert=False)
        self.assertEqual([item[0] if isinstance(item, tuple) else item for item in gallery][1], self.result)
        self.assertEqual(len(gallery), 3)
        self.assertEqual(geninfo["infotexts"], ["ia", "ir", "ic"])
        self.assertEqual(geninfo["seed"], 1)
        self.assertEqual(selected, 1)
        self.assertEqual(gallery[0][0].already_saved_as, "C:/out/a.png", "다시 저장하지 않게 표시")

    def test_insert_puts_the_result_after_the_source(self):
        gallery, geninfo, selected = qb.place_result(self.gallery, 1, self.geninfo, self.result, "ir", insert=True)
        self.assertEqual(len(gallery), 4)
        self.assertIs(gallery[2], self.result)
        self.assertEqual(geninfo["infotexts"], ["ia", "ib", "ir", "ic"])
        self.assertEqual(selected, 2)

    def test_extra_gallery_items_without_infotext_stay_aligned(self):
        gallery, geninfo, _ = qb.place_result(_gallery("a", "map"), 0, {"infotexts": ["ia"]}, self.result, "ir", insert=False)
        self.assertEqual(geninfo["infotexts"], ["ir", None])
        self.assertEqual(len(gallery), 2)


class InfotextTests(unittest.TestCase):
    def test_sam3_fields_are_appended_to_the_source_infotext(self):
        params = {"SAM3 Enable": True, "SAM3 Prompt": "face, hands", "Hires upscale": 2, "SAM3 Version": "0.21.2"}
        text = qb.compose_infotext("girl\nSteps: 30, Seed: 123", params)
        self.assertTrue(text.startswith("girl\nSteps: 30, Seed: 123, "))
        self.assertIn('SAM3 Prompt: "face, hands"', text)
        self.assertIn("SAM3 Enable: True", text)
        self.assertNotIn("Hires upscale", text)
        self.assertIn("SAM3 quick: True", text)


class DependencyLookupTests(unittest.TestCase):
    def test_finds_the_hires_buttons_click_wiring(self):
        button = types.SimpleNamespace(_id=42)
        wanted = types.SimpleNamespace(targets=[(42, "click")], inputs=["i"], outputs=["o"])
        blocks = types.SimpleNamespace(fns={
            1: types.SimpleNamespace(targets=[(42, "change")], inputs=[], outputs=[]),
            2: wanted,
            3: types.SimpleNamespace(targets=[(7, "click")], inputs=[], outputs=[]),
        })
        self.assertIs(qb.find_click_dependency(blocks, button), wanted)
        self.assertIsNone(qb.find_click_dependency(blocks, types.SimpleNamespace(_id=99)))
        self.assertIsNone(qb.find_click_dependency(None, button))


class WiringTests(unittest.TestCase):
    def test_button_reuses_the_hires_buttons_inputs_outputs_and_js(self):
        import inspect

        gradio = types.ModuleType("gradio")

        class Request:
            pass

        gradio.Request = Request
        call_queue = types.ModuleType("modules.call_queue")
        call_queue.wrap_gradio_gpu_call = lambda fn, extra_outputs=None: ("wrapped", fn, extra_outputs)
        package = types.ModuleType("modules")
        package.__path__ = []
        package.call_queue = call_queue
        clicks = []
        button = types.SimpleNamespace(click=lambda **kw: clicks.append(kw))
        dependency = types.SimpleNamespace(inputs=("task", "gallery", "index", "info", "prompt"), outputs=("g", "i", "h", "l"))
        with mock.patch.dict(sys.modules, {"gradio": gradio, "modules": package, "modules.call_queue": call_queue}):
            qb.wire_quick_button(button, dependency)
        (kwargs,) = clicks
        self.assertEqual(kwargs["inputs"], list(dependency.inputs))
        self.assertEqual(kwargs["outputs"], list(dependency.outputs))
        self.assertEqual(kwargs["js"], "submit_txt2img_upscale", "진행 표시·선택 번호를 ✨ 와 같은 JS 로")
        self.assertIs(kwargs["show_progress"], False)
        marker, fn, extra = kwargs["fn"]
        self.assertEqual((marker, extra), ("wrapped", [None, "", ""]))
        self.assertIs(inspect.signature(fn).parameters["request"].annotation, Request,
                      "Gradio 가 요청 객체를 넣으려면 주석이 실제 gr.Request 여야 한다")


class QuickPassSettingsTests(unittest.TestCase):
    """build_i2i 가 빠른 버튼 패스(p._sam3_quick)에만 거는 설정. 빠른 버튼은 process_images(p) 를 거치지 않는다."""

    def _pair(self):
        p = types.SimpleNamespace(override_settings={"CLIP_stop_at_last_layers": 2})
        p2 = types.SimpleNamespace(seed=-1, override_settings={})   # script_args 대입 때 Seed 스크립트가 UI 시드로 덮었다
        return p, p2

    def test_seed_overrides_and_adetailer(self):
        p, p2 = self._pair()
        qb.apply_quick_pass_settings(p, p2, 123)
        self.assertEqual(p2.seed, 123, "그 이미지의 시드로 인페인트")
        self.assertEqual(p2.override_settings, {"CLIP_stop_at_last_layers": 2}, "txt2img Override 설정(Clip skip 등)")
        self.assertIsNot(p2.override_settings, p.override_settings)
        self.assertIs(p2._ad_disabled, True, "버튼은 SAM3 만 요청한다 — 안쪽 패스에서 ADetailer 를 돌리지 않는다")

    def test_missing_override_settings(self):
        p2 = types.SimpleNamespace(seed=-1)
        qb.apply_quick_pass_settings(types.SimpleNamespace(), p2, 5)
        self.assertEqual((p2.seed, p2.override_settings), (5, {}))


class _FakeSam3Script:
    filename = "C:/sd/extensions/forge_sam3_extension/scripts/!sam3.py"
    args_from, args_to = 1, 3

    def __init__(self, mask=True, inpaint=True, overlay=False):
        self.mask = mask
        self.inpaint = inpaint     # False = Mask only 모드
        self.overlay = overlay     # 미리보기 오버레이
        self.process_args = None
        self.seen = {}

    def process(self, p, *args):
        self.process_args = args
        enabled = args[0] is True
        p._sam3_args = {"enabled": enabled}
        if enabled:
            p.extra_generation_params["SAM3 Enable"] = True
            p.extra_generation_params["SAM3 Prompt"] = "face"

    def postprocess_image(self, p, pp, *args):
        assert p._sam3_args["enabled"], "버튼은 아코디언이 꺼져 있어도 SAM3 를 돌린다"
        import modules.shared as shared

        self.seen = {"size": (p.width, p.height), "job_count": shared.state.job_count, "quick": p._sam3_quick}
        p._sam3_mask_found = self.mask
        if self.overlay:
            pp.image = pp.image.copy()   # 실제 스크립트: 마스크가 없어도 오버레이 사본을 넣는다
        if self.mask and self.inpaint:
            pp.image = Image.new("RGB", pp.image.size, "white")


class _Forge:
    """run_sam3_quick 이 쓰는 Forge 모듈만."""

    def __init__(self, sam3=None, samples_save=True, insert=False):
        self.sam3 = sam3 if sam3 is not None else _FakeSam3Script()
        self.saved = []
        self.reloads = 0
        self.created = []
        self.state = types.SimpleNamespace(job_count=-1, interrupted=False, skipped=False)   # state.begin() 직후
        self.opts = types.SimpleNamespace(
            samples_save=samples_save,
            samples_format="png",
            hires_button_gallery_insert=insert,
            txt2img_upscale_same_seed=True,
        )

    def create_processing(self, id_task, request, *args):
        other = types.SimpleNamespace(filename="C:/sd/scripts/other.py", args_from=0, args_to=1)
        scripts = [other] + ([self.sam3] if self.sam3 is not False else [])
        p = types.SimpleNamespace(
            scripts=types.SimpleNamespace(alwayson_scripts=scripts),
            script_args=list(args),
            seed=-1, subseed=-1, prompt="girl", batch_size=4, n_iter=2, width=832, height=1216,
            outpath_samples="C:/out", do_not_save_samples=False,
            extra_generation_params={}, closed=False,
        )
        p.close = lambda: setattr(p, "closed", True)
        self.created.append(p)
        return p

    def modules(self):
        package = types.ModuleType("modules")
        package.__path__ = []
        txt2img = types.ModuleType("modules.txt2img")
        txt2img.txt2img_create_processing = self.create_processing
        infotext_utils = types.ModuleType("modules.infotext_utils")
        infotext_utils.image_from_url_text = lambda item: Image.new("RGB", item[0].size)
        infotext_utils.parse_generation_parameters = lambda text, skip: {"Seed": "123", "Variation seed": "5"}
        images = types.ModuleType("modules.images")
        images.save_image = lambda image, path, basename, *a, **kw: self.saved.append((path, kw.get("info"), kw.get("suffix")))
        scripts = types.ModuleType("modules.scripts")

        class PostprocessImageArgs:   # modules/scripts.py 와 같은 서명
            def __init__(self, image, index):
                self.image = image
                self.index = index

        scripts.PostprocessImageArgs = PostprocessImageArgs
        sd_models = types.ModuleType("modules.sd_models")
        sd_models.forge_model_reload = lambda: setattr(self, "reloads", self.reloads + 1)
        shared = types.ModuleType("modules.shared")
        shared.opts = self.opts
        shared.total_tqdm = types.SimpleNamespace(clear=lambda: None)
        shared.state = self.state
        ui = types.ModuleType("modules.ui")
        ui.plaintext_to_html = lambda text, classname=None: f"<p>{text}</p>"
        gradio = types.ModuleType("gradio")
        gradio.update = lambda **kw: kw
        mods = {"modules": package, "gradio": gradio}
        for name, module in (("txt2img", txt2img), ("infotext_utils", infotext_utils), ("images", images),
                             ("scripts", scripts), ("sd_models", sd_models), ("shared", shared), ("ui", ui)):
            setattr(package, name, module)
            mods[f"modules.{name}"] = module
        return mods

    def run(self, gallery, index, geninfo, args=("id", False, {"sam3_prompt": "face"}, 9)):
        with mock.patch.dict(sys.modules, self.modules()):
            return qb.run_sam3_quick("task", None, gallery, index, json.dumps(geninfo), *args)


class HandlerTests(unittest.TestCase):
    def setUp(self):
        self.gallery = _gallery("a", "b")
        self.geninfo = {"infotexts": ["girl\nSeed: 111", "girl\nSeed: 123"]}

    def test_selected_image_gets_sam3_with_its_own_seed(self):
        forge = _Forge()
        update, geninfo_json, html, _ = forge.run(self.gallery, 1, self.geninfo)
        p = forge.created[0]
        self.assertEqual(forge.sam3.seen["size"], (64, 96), "✨ 처럼 선택한 이미지 크기로 돈다")
        self.assertEqual(forge.sam3.seen["job_count"], 0, "state.begin() 의 -1 이 진행률을 한 칸 밀지 않게")
        self.assertIs(forge.sam3.seen["quick"], True, "build_i2i 가 빠른 버튼 패스를 알아볼 표시")
        self.assertEqual((p.seed, p.subseed, p.batch_size, p.n_iter), (123, 5, 1, 1))
        self.assertEqual(forge.sam3.process_args, (True, {"sam3_prompt": "face"}))
        self.assertEqual(forge.reloads, 1, "드롭다운에서 바꾼 체크포인트를 먼저 불러온다")
        self.assertTrue(p.closed)
        self.assertEqual(len(update["value"]), 2)
        self.assertEqual(update["value"][1].getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(update["selected_index"], 1)
        infotexts = json.loads(geninfo_json)["infotexts"]
        self.assertEqual(infotexts[0], "girl\nSeed: 111")
        self.assertIn("SAM3 Prompt: face", infotexts[1])
        self.assertIn("SAM3 quick: True", infotexts[1])
        self.assertEqual(len(forge.saved), 1)
        self.assertEqual(forge.saved[0][2], "-sam3")
        self.assertIn("SAM3 quick", html)

    def test_insert_setting_keeps_the_source_image(self):
        forge = _Forge(insert=True)
        update, _, _, _ = forge.run(self.gallery, 0, self.geninfo)
        self.assertEqual(len(update["value"]), 3)
        self.assertEqual(update["selected_index"], 1)

    def test_no_mask_leaves_the_gallery_alone(self):
        for sam3 in (_FakeSam3Script(mask=False), _FakeSam3Script(mask=False, overlay=True)):
            with self.subTest(overlay=sam3.overlay):
                forge = _Forge(sam3=sam3)
                gallery, geninfo, message, _ = forge.run(self.gallery, 1, self.geninfo)
                self.assertEqual((gallery, geninfo), ({}, {}), "그대로 두는 길은 no-op update — 갤러리를 다시 저장하지 않게")
                self.assertIn("마스크를 찾지 못했습니다", message)
                self.assertEqual(forge.saved, [])

    def test_mask_only_mode_is_not_reported_as_no_mask(self):
        forge = _Forge(sam3=_FakeSam3Script(mask=True, inpaint=False))
        gallery, _, message, _ = forge.run(self.gallery, 1, self.geninfo)
        self.assertEqual(gallery, {})
        self.assertIn("Inpaint", message)
        self.assertEqual(forge.saved, [])

    def test_skip_or_interrupt_keeps_the_gallery(self):
        forge = _Forge()
        forge.state.skipped = True
        gallery, _, message, _ = forge.run(self.gallery, 1, self.geninfo)
        self.assertEqual(gallery, {})
        self.assertIn("중단", message)
        self.assertEqual(forge.saved, [])

    def test_grid_selection_is_refused_before_anything_runs(self):
        forge = _Forge()
        geninfo = {"index_of_first_image": 1, "infotexts": ["grid", "a"]}
        gallery, _, message, _ = forge.run(_gallery("grid", "a"), 0, geninfo)
        self.assertEqual(gallery, {})
        self.assertIn("grid", message)
        self.assertEqual((forge.created, forge.reloads), ([], 0))

    def test_missing_sam3_script_is_reported(self):
        forge = _Forge(sam3=False)
        _, _, message, _ = forge.run(self.gallery, 0, self.geninfo)
        self.assertIn("SAM3", message)

    def test_saving_follows_forges_setting(self):
        forge = _Forge(samples_save=False)
        forge.run(self.gallery, 0, self.geninfo)
        self.assertEqual(forge.saved, [])


if __name__ == "__main__":
    unittest.main()
