from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NotebookAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (ROOT / "javascript" / "notebook.js").read_text(
            encoding="utf-8"
        )
        cls.css = (ROOT / "style.css").read_text(encoding="utf-8")

    def test_live_workspace_runtime_assets_are_removed(self):
        self.assertFalse((ROOT / "javascript" / "live_workspaces.js").exists())
        self.assertFalse((ROOT / "javascript" / "workspace_manager.js").exists())
        self.assertFalse((ROOT / "sam3ext" / "live_workspace_route.py").exists())
        self.assertNotIn("/sam3-live", self.script)
        self.assertNotIn("__sam3_live_workspace", self.script)

    def test_notebook_has_requested_preset_controls(self):
        self.assertIn('details.id = "sam3_notebook_panel"', self.script)
        self.assertIn("function addPreset()", self.script)
        self.assertIn("function startRename(", self.script)
        self.assertIn("프리셋 삭제", self.script)
        self.assertIn("＋ 항목 추가", self.script)
        self.assertIn("function applyPreset(", self.script)
        self.assertIn("data-notebook-status", self.script)
        self.assertIn("function installDelegatedPanelEvents()", self.script)
        self.assertIn("function installPanelEvents(details)", self.script)
        self.assertIn("bindPanelReferences(existingPanel)", self.script)
        self.assertIn("var eventRoot = app()", self.script)
        self.assertIn('<div class="sam3-notebook-head-actions">', self.script)
        self.assertNotIn(
            'document.addEventListener("click", function (event)',
            self.script,
        )

    def test_notebook_supports_prompt_negative_and_xyz_type_values(self):
        for target in (
            '"prompt"',
            '"negative_prompt"',
            '"xyz_x"',
            '"xyz_y"',
            '"xyz_z"',
        ):
            self.assertIn(target, self.script)
        self.assertIn("entry.axis_type", self.script)
        self.assertIn('prefix + "_type"', self.script)
        self.assertIn('prefix + "_values"', self.script)
        self.assertIn('"X/Y/Z plot"', self.script)

    def test_default_and_compact_prompt_layouts_are_supported(self):
        self.assertIn("function promptLayoutNodes()", self.script)
        self.assertIn('"#txt2img_toprow"', self.script)
        self.assertIn('"#txt2img_prompt_container"', self.script)
        self.assertIn('"#txt2img_generate_box"', self.script)
        self.assertIn(
            '"#txt2img_settings .toprow-compact-stylerow"',
            self.script,
        )
        self.assertIn("var promptNodes = promptLayoutNodes()", self.script)
        self.assertIn("return promptLayoutNodes().length", self.script)
        self.assertIn(".sam3-notebook-prompt-compact", self.css)
        self.assertIn(
            ".sam3-notebook-prompt-compact > .toprow-compact-stylerow",
            self.css,
        )
        self.assertNotIn(
            'return app().querySelector("#txt2img_toprow")\n'
            '                && app().querySelector("#txt2img_settings")',
            self.script,
        )

    def test_preset_apply_is_preflighted_transactional_and_fully_undoable(self):
        self.assertIn("function preflightEntries(entries)", self.script)
        self.assertIn("function captureApplyState(entries)", self.script)
        self.assertIn("async function restoreApplyState(state, label)", self.script)
        self.assertIn("previousState = captureApplyState(preset.entries)", self.script)
        self.assertIn(
            "await restoreApplyState(previousState, null)",
            self.script,
        )
        self.assertIn(
            'script: readWrapperDisplay("script_list")',
            self.script,
        )
        self.assertIn(
            '"script_txt2img_xyz_plot_csv_mode"',
            self.script,
        )
        self.assertIn("var undoState = null", self.script)
        self.assertNotIn("var undoEntries =", self.script)

    def test_legacy_workspaces_have_explicit_non_destructive_import(self):
        self.assertIn(
            'var LEGACY_WORKSPACE_KEY = "sam-extra.workspace-manager.v1"',
            self.script,
        )
        self.assertIn("function legacyWorkspacePresets(raw)", self.script)
        self.assertIn("function importLegacyWorkspaces()", self.script)
        self.assertIn("data-notebook-legacy", self.script)
        self.assertIn("원본 Workspace 데이터는 삭제하지 않습니다", self.script)
        self.assertNotIn(
            "localStorage.removeItem(LEGACY_WORKSPACE_KEY)",
            self.script,
        )
        self.assertIn(
            "Importing the prompt verbatim preserves",
            self.script,
        )
        self.assertIn("function legacyWorkspaceConversion(raw)", self.script)
        self.assertIn("skippedControls", self.script)
        self.assertIn("emptySlots", self.script)
        self.assertIn("가져오지 않는 항목:", self.script)
        for elem_id in (
            "txt2img_prompt",
            "txt2img_neg_prompt",
            "script_txt2img_xyz_plot_",
        ):
            self.assertIn(elem_id, self.script)

    def test_existing_layout_reattaches_a_missing_notebook_panel(self):
        self.assertIn(
            'var existingLayout = app().querySelector("#sam3_notebook_layout")',
            self.script,
        )
        self.assertIn(
            "if (notebookPanel && !existingLayout.contains(notebookPanel))",
            self.script,
        )
        self.assertIn("existingGallery.appendChild(notebookPanel)", self.script)

    def test_notebook_get_and_put_use_the_same_origin_header(self):
        self.assertGreaterEqual(
            self.script.count('"X-SAM3-Notebook": "1"'),
            2,
        )

    def test_scripts_are_positioned_without_reparenting_gradio_components(self):
        self.assertIn("function scriptLayoutNodes(", self.script)
        self.assertIn("function positionScriptPanels(", self.script)
        self.assertNotIn("function isBuiltInScriptPanel(", self.script)
        self.assertIn(
            "result.push(node);",
            self.script,
        )
        self.assertIn(
            "positionScriptPanels(",
            self.script,
        )
        self.assertIn(
            "node === nodes[0] && scriptList",
            self.script,
        )
        self.assertNotIn(
            "parameterTarget.appendChild(scriptContainer)",
            self.script,
        )
        self.assertNotIn(
            "scriptsTarget.appendChild(scriptContainer)",
            self.script,
        )
        self.assertIn(
            ".sam3-notebook-script-float",
            self.css,
        )
        self.assertIn(
            "position: absolute !important",
            self.css,
        )

    def test_static_generation_dropdowns_use_themed_fast_popovers(self):
        self.assertIn("function installFastDropdown(", self.script)
        self.assertIn("function installFastDropdowns()", self.script)
        for elem_id in (
            "txt2img_sampling",
            "txt2img_scheduler",
            "script_list",
            "script_txt2img_xyz_plot_x_type",
            "script_txt2img_xyz_plot_y_type",
            "script_txt2img_xyz_plot_z_type",
        ):
            self.assertIn(elem_id, self.script)
        self.assertIn(
            "isMulti ? next : choice",
            self.script,
        )
        self.assertIn('popover.setAttribute("popover", "auto")', self.script)
        self.assertIn('list.setAttribute("role", "listbox")', self.script)
        self.assertIn('option.setAttribute("role", "option")', self.script)
        self.assertIn('trigger.removeAttribute("aria-invalid")', self.script)
        self.assertIn("var syncTimer = window.setInterval(function ()", self.script)
        self.assertIn("function choiceDisplayValue(meta, rawValue)", self.script)
        self.assertIn(
            "return translatedChoice(Array.isArray(choice) ? choice[0] : choice)",
            self.script,
        )
        self.assertIn(
            "translatedChoice(label) === String(displayValue)",
            self.script,
        )
        self.assertIn(
            'translated === undefined || translated === null || translated === ""',
            self.script,
        )
        self.assertIn(
            "choiceDisplayValue(meta, currentInput.value)",
            self.script,
        )
        self.assertIn("var triggerWasOpenOnPointerDown = false", self.script)
        self.assertIn("sam3-fast-dropdown-trigger", self.script)
        self.assertIn(".sam3-fast-dropdown-popover", self.css)
        self.assertIn(".sam3-fast-dropdown-trigger:hover", self.css)
        self.assertIn(".sam3-fast-dropdown-trigger:focus-visible", self.css)
        self.assertIn(".sam3-fast-dropdown-trigger:active", self.css)
        self.assertIn(".sam3-fast-dropdown-trigger:disabled", self.css)
        self.assertIn('[data-state="loading"]', self.css)
        self.assertIn('[data-state="error"]', self.css)
        self.assertIn('[data-state="success"]', self.css)
        self.assertNotIn("sam3-fast-static-select", self.script)
        self.assertNotIn("wrapper.remove()", self.script)

    def test_script_selector_choices_are_prebuilt_before_first_click(self):
        self.assertIn(
            'installFastDropdown(\n                "script_list", "Script"',
            self.script,
        )
        self.assertLess(
            self.script.index("setChoices(allChoices)"),
            self.script.index('trigger.addEventListener("click", function ()'),
        )

    def test_fast_dropdowns_expand_globally_without_eagerly_rendering_every_choice(self):
        self.assertIn("function installGlobalFastDropdowns()", self.script)
        self.assertIn(
            'querySelectorAll(".gradio-dropdown")',
            self.script,
        )
        self.assertIn(
            'wrapper.closest("#tab_settings, #settings")',
            self.script,
        )
        self.assertIn("new IntersectionObserver(", self.script)
        self.assertIn("MAX_RENDERED_FAST_CHOICES", self.script)
        self.assertIn("function renderOptions(query)", self.script)
        self.assertIn('"/sdapi/v1/sd-models"', self.script)
        self.assertIn('"/sdapi/v1/sd-modules"', self.script)
        self.assertNotIn("function captureNativeDropdownChoices(wrapper)", self.script)
        self.assertNotIn("input.click()", self.script)
        self.assertIn('wrapper.classList.contains("multiselect")', self.script)
        self.assertIn("selectedTokenValues()", self.script)
        self.assertIn("Array.isArray(displayValue)", self.script)
        self.assertIn('"#forge_refresh_checkpoint"', self.script)
        self.assertIn("meta.props.allow_custom_value", self.script)

    def test_global_dropdown_scans_are_coalesced_away_from_tab_clicks(self):
        self.assertIn("fastDropdownScanTimer", self.script)
        self.assertIn("function scheduleGlobalFastDropdownScan(", self.script)
        self.assertIn(
            "window.clearTimeout(fastDropdownScanTimer)",
            self.script,
        )
        self.assertIn("function queueGlobalFastDropdownPrepare(", self.script)
        self.assertIn("window.requestIdleCallback", self.script)
        self.assertNotIn(
            "window.setTimeout(installGlobalFastDropdowns, 80)",
            self.script,
        )

    def test_multiselect_fast_dropdown_keeps_gradio_style_value_chips(self):
        self.assertIn("sam3-fast-dropdown-multi-value", self.script)
        self.assertIn("sam3-fast-dropdown-chip", self.script)
        self.assertIn("selectedValues.forEach(function (choice)", self.script)
        self.assertIn(".sam3-fast-dropdown-multi-value", self.css)
        self.assertIn(".sam3-fast-dropdown-chip", self.css)

    def test_visible_choice_limit_is_configurable_and_short_lists_expand(self):
        self.assertIn(
            'FAST_DROPDOWN_LIMIT_OPTION = "sam3_fast_dropdown_visible_choices"',
            self.script,
        )
        self.assertIn("function fastDropdownVisibleChoiceLimit()", self.script)
        self.assertIn("function refreshFastDropdownVisibleLimits()", self.script)
        self.assertIn("gridTemplateColumns", self.script)
        self.assertIn("data-scroll", self.script)
        self.assertIn("var renderedChoiceLimit", self.script)
        self.assertIn('list.addEventListener("scroll"', self.script)
        self.assertIn("renderedChoiceLimit += fastDropdownVisibleChoiceLimit()", self.script)
        self.assertIn("아래로 스크롤하면 더 표시", self.script)
        self.assertNotIn("검색어를 더 입력하세요", self.script)

    def test_original_forge_extra_network_tabs_are_restored_after_extraction(self):
        gallery_move = self.script.index(
            "galleryTarget.appendChild(gallerySection)"
        )
        restore_call = self.script.index(
            "restoreForgeExtraNetworkTabs(layout)",
            gallery_move,
        )
        self.assertGreater(restore_call, gallery_move)
        self.assertIn(
            'app().querySelector("#txt2img_extra_tabs")',
            self.script,
        )
        self.assertIn("layout.contains(extraTabs)", self.script)
        self.assertIn(
            'layout.querySelector(".sam3-notebook-prompt")',
            self.script,
        )
        self.assertIn("host.appendChild(extraTabs)", self.script)
        self.assertIn(".sam3-notebook-extra-tabs", self.css)
        self.assertIn(
            ".sam3-notebook-extra-tabs > .tabitem:empty",
            self.css,
        )

    def test_notebook_is_below_gallery_in_three_column_layout(self):
        gallery = self.script.index(
            "galleryTarget.appendChild(gallerySection)"
        )
        notebook = self.script.index(
            "galleryTarget.appendChild(notebookPanel)"
        )
        self.assertGreater(notebook, gallery)
        for heading in ("Parameters", "Scripts", "Gallery"):
            self.assertIn(f"<h2>{heading}</h2>", self.script)
        self.assertIn(".sam3-notebook-columns", self.css)
        self.assertIn(
            '[data-column="gallery"] #txt2img_results',
            self.css,
        )
        self.assertIn("position: static !important", self.css)

    def test_value_change_is_not_dispatched_twice_to_the_backend(self):
        """Setting a value already makes Gradio fire change+input.

        The frontend compiles handle_change to gt(l,e,t){l("change",e),t||l("input")}
        and value_is_output is only true for server outputs, so a client-side
        value update already triggers every bound backend handler. Dispatching
        again would run each handler twice - picking a checkpoint would load the
        model twice - so the explicit dispatch must be conditional on the value
        NOT having changed.
        """
        self.assertIn(
            "var previousDisplay = readWrapperDisplay(elemId);", self.script
        )
        self.assertIn(
            "var settledDisplay = readWrapperDisplay(elemId);", self.script
        )
        guarded = self.script.index(
            "if (String(settledDisplay) === String(previousDisplay))"
        )
        dispatch = self.script.index(
            "dispatchGradioChange(componentId);", guarded
        )
        self.assertGreater(dispatch, guarded)

    def test_multiselect_display_is_read_from_tokens_not_the_search_box(self):
        """A multiselect's search input stays empty, so it must not be the
        change signal: the dispatch guard would see every VAE/Text Encoder
        edit as unchanged and double-fire the backend."""
        start = self.script.index("function readWrapperDisplay(elemId)")
        end = self.script.index("async function setWrapperValue(", start)
        body = self.script[start:end]
        self.assertIn('wrapper.classList.contains("multiselect")', body)
        self.assertIn('".token > span:first-child"', body)
        tokens = body.index('".token > span:first-child"')
        generic = body.index('"textarea, input[role=\'listbox\']')
        self.assertLess(tokens, generic, "token read must precede the input read")

    def test_fast_dropdown_rereads_choices_from_the_live_config(self):
        """The proxy hides the real control, so it must not freeze its options.

        Gradio keeps window.gradio_config live (its update queue assigns onto
        the same props objects), so re-reading the config picks up every
        server-side choices refresh without per-control endpoints.
        """
        self.assertIn("function refreshChoicesFromConfig()", self.script)
        self.assertIn("refreshChoicesFromConfig();", self.script)
        # Only externally injected lists opt out of the refresh.
        self.assertIn("var externalChoices = Array.isArray(choiceOverride);", self.script)
        self.assertIn("if (externalChoices || !meta || !meta.props) return;", self.script)
        # O(1) hot path: compare the retained array reference, do not re-map.
        self.assertIn("if (liveChoices === lastConfigChoices) return;", self.script)
        # An empty refresh must clear the list. Bailing on a zero-length result
        # would strand the old options: the new reference is already cached, so
        # the check never runs again for it.
        start = self.script.index("function refreshChoicesFromConfig()")
        end = self.script.index("var syncTimer", start)
        body = self.script[start:end]
        self.assertNotIn("if (!next.length) return;", body)
        self.assertIn("setChoices(next);", body)

    def test_global_dropdowns_are_not_installed_as_external_choices(self):
        """Passing the config list as choiceOverride would freeze it again.

        installFastDropdown already falls back to displayChoices, so the global
        scan must omit the third argument to stay refreshable.
        """
        start = self.script.index("function prepareGlobalFastDropdown(wrapper)")
        end = self.script.index("function refreshFastModelDropdowns()", start)
        body = self.script[start:end]
        self.assertIn(
            "installFastDropdown(wrapper.id, fastDropdownLabel(wrapper));",
            body,
        )
        self.assertNotIn("fastDropdownLabel(wrapper), choices", body)

    def test_script_floats_do_not_create_a_stacking_context(self):
        """An integer z-index would trap Gradio popups below sibling floats."""
        rule = self.css.index(".sam3-notebook-script-float {")
        end = self.css.index("}", rule)
        self.assertNotIn("z-index", self.css[rule:end])


if __name__ == "__main__":
    unittest.main()
