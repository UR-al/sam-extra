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
        self.assertIn(
            ".sam3-live-original-root",
            css,
        )
        self.assertIn(".sam3-live-original-branch", css)
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


if __name__ == "__main__":
    unittest.main()
