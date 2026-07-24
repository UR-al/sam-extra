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
        self.assertNotIn(".sam3-live-original-branch", css)
        self.assertNotIn("isolateLayoutBranch", script)
        self.assertIn('iframe.setAttribute("data-active"', script)
        self.assertNotIn("iframe.hidden =", script)
        self.assertIn('.sam3-live-frames iframe[data-active="true"]', css)
        self.assertNotIn(".sam3-live-frames iframe[hidden]", css)
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
        self.assertIn(
            'var STATUS_MESSAGE = "sam3-live-workspace-status-v1"',
            live,
        )
        self.assertIn(
            'var LIVE_STATUS_MESSAGE = "sam3-live-workspace-status-v1"',
            manager,
        )
        self.assertIn("childStatuses[statusSlot]", live)
        self.assertIn("window.parent.postMessage", manager)

    def test_live_shell_can_handoff_to_real_browser_tabs(self):
        live = (ROOT / "javascript" / "live_workspaces.js").read_text(
            encoding="utf-8"
        )
        manager = (ROOT / "javascript" / "workspace_manager.js").read_text(
            encoding="utf-8"
        )
        css = (ROOT / "style.css").read_text(encoding="utf-8")

        self.assertIn("data-sam3-live-native-tabs", live)
        self.assertIn("async function openNativeWorkspaceTabs()", live)
        self.assertIn("var flushPromise = flushLoadedChildren()", live)
        self.assertIn('var handle = window.open("", nativeTabTarget(slot))', live)
        self.assertIn("handle.location.replace(nativeTabUrl(entry.slot))", live)
        self.assertIn("window.location.assign(nativeTabUrl(state.active))", live)
        self.assertIn('url.searchParams.set("__sam3_native_tab", "1")', live)
        self.assertIn(
            "var nativeWorkspaceTab = !!frameSlot && window.parent === window",
            live,
        )
        self.assertIn('document.title = name + " · Forge Neo"', live)
        self.assertIn("data-sam3-native-status", live)
        self.assertIn("Live 관리로 돌아가기", live)
        self.assertIn(
            "var NATIVE_WORKSPACE_TAB = !!LIVE_FRAME_SLOT "
            "&& window.parent === window",
            manager,
        )
        self.assertIn(
            'document.querySelector("[data-sam3-native-status]")',
            manager,
        )
        self.assertIn(".sam3-native-workspace-bar", css)
        self.assertIn(".sam3-native-workspace-status", css)

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

    def test_generate_keeps_previous_gallery_until_completion(self):
        """On Generate the previous gallery stays visible until the new image
        actually completes; only non-generate (switch/reset) paths clear it."""
        manager = (ROOT / "javascript" / "workspace_manager.js").read_text(
            encoding="utf-8"
        )
        css = (ROOT / "style.css").read_text(encoding="utf-8")

        clear_start = manager.index("function clearVisibleWorkspaceOutputs")
        clear_end = manager.index("function dispatchGradioChange", clear_start)
        clear_body = manager[clear_start:clear_end]

        split = clear_body.index("// Non-generate paths")
        generate_branch = clear_body[:split]
        non_generate_branch = clear_body[split:]

        # Generate: no hide, no output mutation, early return — the previous
        # result is left on screen and the final image replaces it normally.
        self.assertIn('if (reason === "generate")', generate_branch)
        self.assertIn("return;", generate_branch)
        self.assertNotIn("dispatchOutputValue(", generate_branch)
        self.assertNotIn("emptyWorkspaceOutputs()", generate_branch)

        # Non-generate: clear gallery/info/html immediately.
        self.assertIn(
            "dispatchOutputValue(outputComponentIds.gallery, [])",
            non_generate_branch,
        )
        self.assertIn(
            'dispatchOutputValue(outputComponentIds.generationInfo, "")',
            non_generate_branch,
        )
        self.assertIn(
            'dispatchOutputValue(outputComponentIds.htmlInfo, "")',
            non_generate_branch,
        )

        # The old hide-on-generate machinery is fully removed.
        self.assertNotIn("GalleryGenerationPending", manager)
        self.assertNotIn("sam3-workspace-generation-pending", manager)
        self.assertNotIn("sam3-workspace-generation-pending", css)


if __name__ == "__main__":
    unittest.main()
