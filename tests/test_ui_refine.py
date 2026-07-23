from __future__ import annotations

import unittest

from sam3ext.ui_refine import _refine_widget_count


class RefineInputAlignmentTests(unittest.TestCase):
    def test_optional_canvas_keeps_three_txt2img_extras_aligned(self):
        self.assertEqual(_refine_widget_count(45), 42)  # canvas + 3 extras
        self.assertEqual(_refine_widget_count(43), 40)  # no canvas + 3 extras
        self.assertIsNone(_refine_widget_count(42))


if __name__ == "__main__":
    unittest.main()
