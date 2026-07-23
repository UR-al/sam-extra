from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LiveWorkspaceAssetTests(unittest.TestCase):
    def test_child_layout_stays_inside_gradio_container(self):
        script = (ROOT / "javascript" / "live_workspaces.js").read_text(
            encoding="utf-8"
        )
        css = (ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn("sourceRoot.appendChild(layout)", script)
        self.assertIn(
            'app().querySelector("#txt2img_script_container")',
            script,
        )
        self.assertIn(
            'scriptContainer.querySelector("#script_list")',
            script,
        )
        self.assertIn("splitScriptContainer(scriptContainer, scripts)", script)
        self.assertIn("isBuiltInScriptPanel", script)
        self.assertIn("script_txt2img_prompt_matrix_", script)
        self.assertIn("script_txt2img_prompts_from_file_or_textbox_", script)
        self.assertIn("script_txt2img_xyz_plot_", script)
        self.assertNotIn("isAlwaysOnAccordionGroup", script)
        self.assertIn('app().querySelector("#tab_txt2img")', script)
        self.assertIn("layoutHost.appendChild(layout)", script)
        self.assertIn("parameterTarget.appendChild(scriptContainer)", script)
        self.assertIn("consecutiveFailures >= 2", script)
        self.assertIn("window.location.reload()", script)
        self.assertIn('var READY_MESSAGE = "sam3-live-workspace-ready-v1"', script)
        self.assertIn("window.parent.postMessage", script)
        self.assertIn("readyFrame.contentWindow !== event.source", script)
        self.assertIn('new URL("/sam3-live", window.location.origin)', script)
        self.assertIn("data-sam3-standalone-live-shell", script)
        self.assertIn('new URL("/", window.location.origin)', script)
        self.assertIn("var standaloneRedirect = null", script)
        self.assertIn("if (!redirected) mountShell()", script)
        self.assertIn("Build one full Forge document at a time", script)
        self.assertIn("setTimeout(loadNext, 100)", script)
        self.assertNotIn("queueInactiveWarmup", script)
        self.assertNotIn("3500 * (index + 1)", script)
        self.assertIn(
            ".sam3-live-original-root",
            css,
        )
        self.assertIn(".sam3-live-original-branch", css)
        self.assertIn(".sam3-live-loading", css)
        self.assertIn("@keyframes sam3-live-spin", css)
        self.assertNotIn(
            'var scripts = app().querySelector("#script_list")',
            script,
        )
        self.assertNotIn(
            "html.sam3-live-frame-active .sam3-live-source-root {\n"
            "    display: none",
            css,
        )

    def test_child_frames_are_pinned_without_nested_workspace_toolbar(self):
        manager = (ROOT / "javascript" / "workspace_manager.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("var activeSlot = LIVE_FRAME_SLOT || readActiveSlot()", manager)
        self.assertIn("if (LIVE_FRAME_SLOT) return true", manager)
        self.assertIn('baseKey = type + "|semantic:"', manager)

    def test_live_shell_exposes_shared_workspace_management(self):
        live = (ROOT / "javascript" / "live_workspaces.js").read_text(
            encoding="utf-8"
        )
        manager = (ROOT / "javascript" / "workspace_manager.js").read_text(
            encoding="utf-8"
        )
        css = (ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn("data-sam3-live-add", live)
        self.assertIn('aria-label="Workspace 추가"', live)
        self.assertIn("data-sam3-live-delete", live)
        self.assertIn("data-sam3-live-export", live)
        self.assertIn("data-sam3-live-import", live)
        self.assertIn("storage.createFrom(state.active)", live)
        self.assertIn("await storage.remove(deletedSlot)", live)
        self.assertIn("storage.exportPayload()", live)
        self.assertIn("storage.importStore(parsed, state.active)", live)
        self.assertIn('var CONTROL_MESSAGE = "sam3-live-workspace-control-v1"', live)
        self.assertIn("await manager.flushForLiveShell()", live)
        self.assertIn("manager.prepareForLiveImport()", live)
        self.assertIn("manager.cancelLiveImport()", live)
        self.assertIn("window.__sam3LiveChildLayoutSettled = true", live)
        self.assertIn("function queueDefaultWorkspaces()", live)
        self.assertIn("activate(state.active);", live)
        self.assertIn("queueDefaultWorkspaces();", live)
        self.assertIn("현재 화면 우선 · 나머지는 순차 준비", live)
        self.assertIn("return loadQueue.length > 0", live)
        self.assertNotIn("startupLoadingSlots", live)
        self.assertNotIn('var SLOT_IDS = ["1", "2", "3"]', live)

        self.assertIn("var LIVE_SHELL_PAGE =", manager)
        self.assertIn("if (!LIVE_SHELL_PAGE)", manager)
        self.assertIn("window.gradio_config.components", manager)
        self.assertIn('cache: "default"', manager)
        self.assertIn('if (!LIVE_FRAME_SLOT) await delay(250)', manager)
        self.assertIn('reason === "startup"', manager)
        self.assertIn("stagedDriverKeys", manager)
        self.assertIn("cancelLiveImport: cancelLiveImport", manager)
        self.assertIn("window.__sam3LiveChildLayoutSettled === true", manager)
        self.assertIn("createFrom: createStoredWorkspace", manager)
        self.assertIn("remove: deleteStoredWorkspace", manager)
        self.assertIn("exportPayload: workspaceExportPayload", manager)
        self.assertIn("importStore: replaceWorkspaceStore", manager)
        self.assertIn(".sam3-live-menu-panel", css)
        self.assertIn(".sam3-live-brand", css)
        self.assertIn(".sam3-live-delete", css)
        self.assertIn(".sam3-live-status", css)

    def test_xyz_plot_drivers_and_values_are_captured_and_restored(self):
        manager = (ROOT / "javascript" / "workspace_manager.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('id === "script_list"', manager)
        self.assertIn(
            '/^script_txt2img_xyz_plot_[xyz]_type$/.test(id)',
            manager,
        )
        self.assertIn(
            '"dropdown|xyz:" + axis + "-values-choice"',
            manager,
        )
        self.assertIn("controls[adapter.key] = record", manager)
        self.assertIn(
            "record.active = logicallyVisibleInWorkspace(adapter.el)",
            manager,
        )
        self.assertIn("offsetParent becomes null for every control", manager)
        self.assertIn("stageDriverValue(driver, driverRecord)", manager)
        self.assertIn("triggerDriverDependency(dependencyDriver", manager)
        self.assertIn("waitForXyzChoiceOptions(snapshot, catalog)", manager)


if __name__ == "__main__":
    unittest.main()
