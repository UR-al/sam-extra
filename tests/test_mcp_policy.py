"""sam-extra MCP server: the operator's permissions (Settings -> SAM Extra MCP) and their enforcement.

The server reads Forge's settings file before every state-changing call and refuses, before any
request is made, whatever the agent passes. These tests run the real service against a fake Forge
API and count the requests that reach it.
"""
from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _mcp_support import HAS_HTTPX, httpx, make_forge_tree, make_service, write_settings

from sam_extra_mcp import policy
from sam_extra_mcp.policy import (
    ALLOW_DOWNLOAD,
    ALLOW_GENERATE,
    ALLOW_INTERRUPT,
    ALLOW_MODEL_SWITCH,
    PolicyReader,
)


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class PermissionTableTests(unittest.TestCase):
    def test_the_four_permissions_and_their_defaults(self):
        self.assertEqual(
            {permission.key: permission.default for permission in policy.PERMISSIONS},
            {
                "sam3_mcp_allow_generate": True,
                "sam3_mcp_allow_model_switch": False,
                "sam3_mcp_allow_interrupt": False,
                "sam3_mcp_allow_download": False,
            },
        )
        self.assertEqual(policy.SETTINGS_SECTION, ("sam3_mcp", "SAM Extra MCP"))

    def test_every_key_is_sam3_prefixed(self):
        for permission in policy.PERMISSIONS:
            self.assertTrue(permission.key.startswith("sam3_mcp_"))


class PolicyReaderTests(_Tmp):
    def reader(self, **kwargs):
        return PolicyReader(str(self.base / "config.json"), sleep=lambda seconds: None, **kwargs)

    def test_no_settings_file_means_the_defaults(self):
        current = self.reader().read()
        self.assertEqual(current.state, policy.STATE_DEFAULTS)
        self.assertTrue(current.allows(ALLOW_GENERATE))
        for key in (ALLOW_MODEL_SWITCH, ALLOW_INTERRUPT, ALLOW_DOWNLOAD):
            self.assertFalse(current.allows(key))
        self.assertTrue(current.problems)

    def test_values_come_from_the_file(self):
        write_settings(self.base / "config.json", {
            ALLOW_GENERATE: False, ALLOW_MODEL_SWITCH: True, ALLOW_INTERRUPT: True, ALLOW_DOWNLOAD: True,
            "unrelated_option": 3,
        })
        current = self.reader().read()
        self.assertEqual(current.state, policy.STATE_SETTINGS)
        self.assertFalse(current.allows(ALLOW_GENERATE))
        self.assertTrue(current.allows(ALLOW_MODEL_SWITCH))
        self.assertTrue(current.allows(ALLOW_INTERRUPT))
        self.assertTrue(current.allows(ALLOW_DOWNLOAD))
        self.assertEqual(current.problems, ())

    def test_absent_keys_take_their_defaults(self):
        write_settings(self.base / "config.json", {"sd_model_checkpoint": "x"})
        current = self.reader().read()
        self.assertTrue(current.allows(ALLOW_GENERATE))
        self.assertFalse(current.allows(ALLOW_DOWNLOAD))

    def test_a_value_that_is_not_a_boolean_is_off(self):
        write_settings(self.base / "config.json", {ALLOW_GENERATE: "true", ALLOW_DOWNLOAD: 1})
        current = self.reader().read()
        self.assertFalse(current.allows(ALLOW_GENERATE))
        self.assertFalse(current.allows(ALLOW_DOWNLOAD))
        self.assertEqual(len(current.problems), 2)

    def test_a_corrupt_file_locks_everything(self):
        (self.base / "config.json").write_text("{ not json", encoding="utf-8")
        slept = []
        current = PolicyReader(str(self.base / "config.json"), sleep=slept.append, attempts=3).read()
        self.assertEqual(current.state, policy.STATE_LOCKED)
        for permission in policy.PERMISSIONS:
            self.assertFalse(current.allows(permission.key))
        self.assertEqual(len(slept), 2, "retried before giving up")

    def test_a_file_caught_mid_write_is_read_again(self):
        path = self.base / "config.json"
        path.write_text('{"sam3_mcp_allow_down', encoding="utf-8")

        def finish_writing(_seconds):
            path.write_text(json.dumps({ALLOW_DOWNLOAD: True}), encoding="utf-8")

        current = PolicyReader(str(path), sleep=finish_writing).read()
        self.assertEqual(current.state, policy.STATE_SETTINGS)
        self.assertTrue(current.allows(ALLOW_DOWNLOAD))

    def test_a_json_value_that_is_not_an_object_locks_everything(self):
        (self.base / "config.json").write_text("[1, 2]", encoding="utf-8")
        self.assertEqual(self.reader().read().state, policy.STATE_LOCKED)

    def test_nesting_too_deep_to_parse_locks_instead_of_raising(self):
        # json raises RecursionError here, not ValueError: uncaught, every tool call would fail.
        (self.base / "config.json").write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
        current = self.reader().read()
        self.assertEqual(current.state, policy.STATE_LOCKED)
        self.assertFalse(current.allows(ALLOW_GENERATE))

    def test_an_oversized_file_locks_everything_without_being_read(self):
        write_settings(self.base / "config.json", {ALLOW_GENERATE: True, "padding": "x" * 256})
        with mock.patch.object(policy, "MAX_SETTINGS_BYTES", 128), mock.patch("builtins.open", side_effect=AssertionError):
            current = self.reader().read()
        self.assertEqual(current.state, policy.STATE_LOCKED)
        self.assertFalse(current.allows(ALLOW_GENERATE))
        self.assertIn("bytes", current.problems[0])

    def test_unknown_location_locks_everything(self):
        current = PolicyReader(None).read()
        self.assertEqual(current.state, policy.STATE_LOCKED)
        self.assertFalse(current.allows(ALLOW_GENERATE))
        self.assertIn("SAM_EXTRA_MCP_FORGE_CONFIG", current.problems[0])

    def test_a_forge_on_another_machine_locks_everything_unless_named(self):
        write_settings(self.base / "config.json", {ALLOW_GENERATE: True})
        self.assertEqual(self.reader(same_machine=False).read().state, policy.STATE_LOCKED)
        self.assertEqual(self.reader(same_machine=False, explicit=True).read().state, policy.STATE_SETTINGS)

    def test_a_change_applies_on_the_next_read(self):
        path = self.base / "config.json"
        write_settings(path, {ALLOW_INTERRUPT: False})
        reader = self.reader()
        self.assertFalse(reader.read().allows(ALLOW_INTERRUPT))
        write_settings(path, {ALLOW_INTERRUPT: True})
        self.assertTrue(reader.read().allows(ALLOW_INTERRUPT))
        write_settings(path, {ALLOW_INTERRUPT: False})
        self.assertFalse(reader.read().allows(ALLOW_INTERRUPT))

    def test_unchanged_file_is_not_parsed_again(self):
        path = self.base / "config.json"
        write_settings(path, {ALLOW_GENERATE: True})
        reader = self.reader()
        first = reader.read()
        self.assertIs(reader.read(), first)

    def test_denial_names_the_setting_and_who_may_change_it(self):
        current = policy.from_settings({ALLOW_MODEL_SWITCH: False}, "config.json")
        result = policy.denial(current, ALLOW_MODEL_SWITCH)
        self.assertIs(result["ok"], False)
        self.assertEqual(result["denied_by_policy"], ALLOW_MODEL_SWITCH)
        self.assertIn("Settings -> SAM Extra MCP", result["error"])
        self.assertIn("only the operator", result["error"])
        self.assertEqual(result["policy"]["permissions"][ALLOW_MODEL_SWITCH], False)


@unittest.skipUnless(HAS_HTTPX, "httpx is not installed")
class EnforcementTests(_Tmp):
    """The service refuses before any request reaches Forge."""

    def service(self, settings=None, **kwargs):
        service, fake, tree = make_service(self.base, settings=settings, **kwargs)
        self.addCleanup(service.close)
        return service, fake, tree

    def test_generation_is_allowed_by_default(self):
        service, fake, _ = self.service(settings={})
        result = service.generate("a lighthouse at dusk", steps=4, width=512, height=512)
        self.assertIs(result["ok"], True, result)
        self.assertEqual(len(fake.calls("POST", "/sdapi/v1/txt2img")), 1)

    def test_generation_switched_off_never_reaches_forge(self):
        service, fake, _ = self.service(settings={ALLOW_GENERATE: False})
        result = service.generate("a lighthouse at dusk")
        self.assertEqual(result["denied_by_policy"], ALLOW_GENERATE)
        self.assertEqual(fake.requests, [], "nothing at all was requested")

    def test_img2img_is_generation_too(self):
        service, fake, _ = self.service(settings={ALLOW_GENERATE: False})
        image = self.base / "init.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        result = service.generate("repaint", init_image=str(image))
        self.assertEqual(result["denied_by_policy"], ALLOW_GENERATE)
        self.assertEqual(fake.calls("POST", "/sdapi/v1/img2img"), [])

    def test_model_switch_is_off_by_default(self):
        service, fake, _ = self.service(settings={})
        result = service.models(action="load", name="Krea/kreaMix_v20.safetensors", preset="krea")
        self.assertEqual(result["denied_by_policy"], ALLOW_MODEL_SWITCH)
        self.assertEqual(fake.calls("POST", "/sdapi/v1/options"), [])
        # Listing and refreshing stay available.
        self.assertIs(service.models(action="list")["ok"], True)
        self.assertIs(service.models(action="refresh")["ok"], True)

    def test_refresh_needs_no_switch_even_with_everything_off(self):
        """Documented choice: a checkpoint-list rescan changes nothing loaded or selected."""
        service, fake, _ = self.service(settings={key: False for key in policy.DEFAULTS})
        self.assertIs(service.models(action="refresh")["ok"], True)
        self.assertEqual(len(fake.calls("POST", "/sdapi/v1/refresh-checkpoints")), 1)
        self.assertEqual(service.models(action="load", name="Krea/kreaMix_v20.safetensors")["denied_by_policy"],
                         ALLOW_MODEL_SWITCH)
        self.assertEqual(fake.calls("POST", "/sdapi/v1/options"), [])

    def test_an_unknown_preset_is_refused_before_anything_is_written(self):
        service, fake, _ = self.service(settings={ALLOW_MODEL_SWITCH: True})
        for preset in ("sdxl-turbo", "anima xl", "../xl", "krea2", "xl"):  # xl: a Forge preset, not on this instance
            result = service.models(action="load", name="Krea/kreaMix_v20.safetensors", preset=preset)
            self.assertIs(result["ok"], False, preset)
            self.assertIn("unknown preset", result["error"])
            self.assertEqual(result["choices"], ["anima", "krea"])
        self.assertEqual(fake.calls("POST", "/sdapi/v1/options"), [])

    def test_a_known_preset_is_matched_without_regard_to_case(self):
        service, fake, _ = self.service(settings={ALLOW_MODEL_SWITCH: True})
        result = service.models(action="load", name="Krea/kreaMix_v20.safetensors", preset=" KREA ")
        self.assertIs(result["ok"], True, result)
        self.assertEqual(fake.calls("POST", "/sdapi/v1/options")[0][2]["forge_preset"], "krea")

    def test_model_switch_when_allowed_sends_only_switch_keys(self):
        service, fake, _ = self.service(settings={ALLOW_MODEL_SWITCH: True})
        result = service.models(action="load", name="Krea/kreaMix_v20.safetensors", preset="krea")
        self.assertIs(result["ok"], True, result)
        posted = fake.calls("POST", "/sdapi/v1/options")
        self.assertEqual(len(posted), 1)
        self.assertLessEqual(set(posted[0][2]), {"sd_model_checkpoint", "forge_preset", "forge_additional_modules"})

    def test_interrupt_and_skip_are_off_by_default(self):
        service, fake, _ = self.service(settings={})
        for action in ("interrupt", "skip", "INTERRUPT "):
            result = service.progress(action=action)
            self.assertEqual(result["denied_by_policy"], ALLOW_INTERRUPT, action)
        self.assertEqual(fake.calls("POST"), [])
        status = service.progress()
        self.assertIs(status["ok"], True)
        self.assertIs(status["interrupt_allowed"], False)

    def test_interrupt_when_allowed(self):
        service, fake, _ = self.service(settings={ALLOW_INTERRUPT: True})
        self.assertIs(service.progress(action="interrupt")["ok"], True)
        self.assertIs(service.progress(action="skip")["ok"], True)
        self.assertEqual(len(fake.calls("POST", "/sdapi/v1/interrupt")), 1)
        self.assertEqual(len(fake.calls("POST", "/sdapi/v1/skip")), 1)

    def test_download_confirm_cannot_bypass_the_policy(self):
        network = []

        def hugging_face(request):
            network.append((request.method, str(request.url)))
            return httpx.Response(500)

        service, fake, tree = self.service(settings={}, download_transport=httpx.MockTransport(hugging_face))
        result = service.module_download(label="Qwen3 0.6B base", confirm=True)
        self.assertEqual(result["denied_by_policy"], ALLOW_DOWNLOAD)
        self.assertEqual(network, [], "nothing was asked of the download host")
        self.assertEqual(list((tree["forge"] / "models" / "text_encoder").iterdir()), [])

    def test_download_planning_while_off_stays_on_this_machine(self):
        network = []

        def hugging_face(request):
            network.append(request.method)
            return httpx.Response(500)

        service, _, tree = self.service(settings={}, download_transport=httpx.MockTransport(hugging_face))
        listing = service.module_download()
        self.assertIs(listing["downloads_allowed"], False)
        self.assertIn("switched off", listing["how_to_download"])
        plan = service.module_download(label="Qwen3 0.6B base")
        self.assertIs(plan["ok"], True)
        self.assertIs(plan["downloads_allowed"], False)
        self.assertIs(plan["plan"]["probed"], False)
        self.assertEqual(plan["plan"]["destination"],
                         str(tree["forge"] / "models" / "text_encoder" / "qwen_3_06b_base.safetensors"))
        self.assertEqual(network, [])

    def test_a_settings_change_applies_to_the_next_call(self):
        service, fake, tree = self.service(settings={ALLOW_GENERATE: False})
        self.assertEqual(service.generate("first")["denied_by_policy"], ALLOW_GENERATE)
        write_settings(tree["settings"], {ALLOW_GENERATE: True})
        self.assertIs(service.generate("second", steps=4)["ok"], True)
        write_settings(tree["settings"], {ALLOW_GENERATE: False})
        self.assertEqual(service.generate("third")["denied_by_policy"], ALLOW_GENERATE)
        self.assertEqual(len(fake.calls("POST", "/sdapi/v1/txt2img")), 1)

    def test_a_corrupt_settings_file_stops_generation_too(self):
        service, fake, tree = self.service(settings={})
        tree["settings"].write_text("{ broken", encoding="utf-8")
        service.policy_reader._sleep = lambda seconds: None
        result = service.generate("anything")
        self.assertEqual(result["denied_by_policy"], ALLOW_GENERATE)
        self.assertEqual(result["policy"]["state"], policy.STATE_LOCKED)
        self.assertEqual(fake.calls("POST"), [])

    def test_policy_is_visible_in_capabilities(self):
        service, _, tree = self.service(settings={ALLOW_DOWNLOAD: True})
        report = service.capabilities()
        self.assertEqual(report["policy"]["permissions"], {
            ALLOW_GENERATE: True, ALLOW_MODEL_SWITCH: False, ALLOW_INTERRUPT: False, ALLOW_DOWNLOAD: True,
        })
        self.assertEqual(report["policy"]["source"], str(tree["settings"]))

    def test_policy_is_visible_while_forge_is_down(self):
        tree = make_forge_tree(self.base, settings={ALLOW_GENERATE: False})
        from sam_extra_mcp import forge_paths
        from sam_extra_mcp.forgeneo.config import Config
        from sam_extra_mcp.service import Service

        def down(request):
            raise httpx.ConnectError("refused", request=request)

        env = {"FORGE_URL": "http://127.0.0.1:7860"}
        service = Service(Config.from_env(env), forge_paths.discover(env, module_file=tree["module_file"]),
                          transport=httpx.MockTransport(down))
        self.addCleanup(service.close)
        report = service.capabilities()
        self.assertIs(report["reachable"], False)
        self.assertIs(report["policy"]["permissions"][ALLOW_GENERATE], False)


@unittest.skipUnless(HAS_HTTPX, "httpx is not installed")
class ClientGuardTests(_Tmp):
    """The client itself refuses what the server never needs."""

    def test_cmd_flags_and_sysinfo_are_never_requested(self):
        from sam_extra_mcp.forgeneo.client import ForgeClient
        from sam_extra_mcp.forgeneo.config import Config

        seen = []
        client = ForgeClient(Config(), transport=httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200, json={})))
        self.addCleanup(client.close)
        for path in ("/sdapi/v1/cmd-flags", "/sdapi/v1/cmd-flags?x=1", "/internal/sysinfo", "/internal/sysinfo-download",
                     "/sdapi/v1/server-kill", "/sdapi/v1/extensions"):
            result = client.get(path)
            self.assertIs(result.ok, False, path)
            self.assertIn("refused", result.error)
        self.assertIs(client.post("/sdapi/v1/server-restart", {}).ok, False)
        self.assertEqual(seen, [])

    def test_only_checkpoint_switch_options_can_be_written(self):
        from sam_extra_mcp.forgeneo.client import ForgeClient
        from sam_extra_mcp.forgeneo.config import Config

        seen = []
        client = ForgeClient(Config(), transport=httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200, json={})))
        self.addCleanup(client.close)
        result = client.set_options({"sd_model_checkpoint": "x", ALLOW_GENERATE: True})
        self.assertIs(result.ok, False)
        self.assertIn(ALLOW_GENERATE, result.error)
        self.assertIs(client.set_options({"samples_save": False}).ok, False)
        self.assertEqual(seen, [])
        self.assertIs(client.set_options({"sd_model_checkpoint": "x", "forge_preset": "anima"}).ok, True)
        self.assertEqual(len(seen), 1)

    def test_redirects_are_not_followed(self):
        """A 3xx is an error: the allow-list is checked on the path asked for, so a followed
        redirect could reach any route (cmd-flags included)."""
        from sam_extra_mcp.forgeneo.client import ForgeClient
        from sam_extra_mcp.forgeneo.config import Config

        seen = []

        def redirecting(request):
            seen.append((request.method, request.url.path))
            return httpx.Response(307, headers={"location": "/sdapi/v1/cmd-flags"})

        client = ForgeClient(Config(), transport=httpx.MockTransport(redirecting))
        self.addCleanup(client.close)
        result = client.options()
        self.assertIs(result.ok, False)
        self.assertIn("redirect", result.error)
        self.assertIs(client.txt2img({"prompt": "x"}).ok, False)
        self.assertEqual(seen, [("GET", "/sdapi/v1/options"), ("POST", "/sdapi/v1/txt2img")])

    def test_the_options_guard_holds_for_post_too(self):
        """Not only set_options(): any options write through the client is checked."""
        from sam_extra_mcp.forgeneo.client import ForgeClient
        from sam_extra_mcp.forgeneo.config import Config

        seen = []
        client = ForgeClient(Config(), transport=httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200, json={})))
        self.addCleanup(client.close)
        for path in ("/sdapi/v1/options", "/sdapi/v1/options?x=1"):
            result = client.post(path, {ALLOW_DOWNLOAD: True})
            self.assertIs(result.ok, False, path)
            self.assertIn(ALLOW_DOWNLOAD, result.error)
        self.assertEqual(seen, [])

    def test_credentials_in_forge_url_reach_forge_but_never_the_agent(self):
        service, fake, _ = make_service(self.base, settings={},
                                        env={"FORGE_URL": "http://alice:s3cret-pass@127.0.0.1:7860"})
        self.addCleanup(service.close)
        report = service.capabilities()
        self.assertIs(report["reachable"], True)
        self.assertNotIn("s3cret", json.dumps(report))
        expected = "Basic " + base64.b64encode(b"alice:s3cret-pass").decode("ascii")
        self.assertTrue(fake.authorizations)
        self.assertEqual(set(fake.authorizations), {expected})

        from sam_extra_mcp.forgeneo.client import ForgeClient

        def down(request):
            raise httpx.ConnectError("refused", request=request)

        client = ForgeClient(service.config, transport=httpx.MockTransport(down))
        self.addCleanup(client.close)
        self.assertNotIn("s3cret", client.options().error)

    def test_no_tool_ever_touches_cmd_flags(self):
        service, fake, tree = make_service(self.base, settings={
            ALLOW_GENERATE: True, ALLOW_MODEL_SWITCH: True, ALLOW_INTERRUPT: True, ALLOW_DOWNLOAD: False})
        self.addCleanup(service.close)
        service.capabilities(refresh=True)
        service.model_profile(refresh=True)
        service.prompt_dialect()
        service.loras(query="style", refresh=True)
        service.lora_info("none")
        service.models()
        service.models(action="load", name="Krea/kreaMix_v20.safetensors")
        service.module_check()
        service.module_download()
        service.generate("a cat", steps=4)
        service.progress()
        service.progress(action="interrupt")
        paths = {path for _method, path, _body in fake.requests}
        self.assertNotIn("/sdapi/v1/cmd-flags", paths)
        self.assertFalse(any(path.startswith("/internal") for path in paths))


if __name__ == "__main__":
    unittest.main()
