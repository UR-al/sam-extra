from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AppearanceThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.python = (ROOT / "scripts" / "appearance_theme.py").read_text(
            encoding="utf-8"
        )
        cls.javascript = (
            ROOT / "javascript" / "appearance_theme.js"
        ).read_text(encoding="utf-8")
        cls.css = (ROOT / "style.css").read_text(encoding="utf-8")
        cls.tokens = (ROOT / "tokens.css").read_text(encoding="utf-8")
        cls.design = (ROOT / "design.md").read_text(encoding="utf-8")

    def test_layout_sections_option_is_registered(self):
        """txt2img 섹션 정리를 끄는 스위치. 값(기본 True)은 tests/test_layout_lanes.py 가 함께 본다."""
        self.assertIn("OPT_LAYOUT_SECTIONS", self.python)
        self.assertIn("shared.opts.add_option(\n        OPT_LAYOUT_SECTIONS,", self.python)
        self.assertIn("txt2img 섹션 정리", self.python)
        self.assertIn("sam3_lanes=off", self.python)

    def test_settings_registers_all_requested_theme_choices(self):
        self.assertIn('OPT_APPEARANCE_THEME = "sam3_appearance_theme"', self.python)
        self.assertIn("script_callbacks.on_ui_settings(on_ui_settings)", self.python)
        for label in (
            "Forge Default",
            "Graphite Ember",
            "Obsidian Violet",
            "Warm Espresso",
            "OLED Mono",
        ):
            self.assertIn(f'"{label}"', self.python)
            self.assertIn(f'"{label}"', self.javascript)

    def test_settings_registers_fast_dropdown_visible_choice_limit(self):
        self.assertIn(
            'OPT_FAST_DROPDOWN_VISIBLE_CHOICES = '
            '"sam3_fast_dropdown_visible_choices"',
            self.python,
        )
        self.assertIn("DEFAULT_FAST_DROPDOWN_VISIBLE_CHOICES = 60", self.python)
        self.assertIn('"minimum": 10', self.python)
        self.assertIn('"maximum": 200', self.python)
        self.assertIn('"step": 5', self.python)

    def test_frontend_applies_options_and_cleans_up_for_forge_default(self):
        self.assertIn("onOptionsAvailable(applyConfiguredTheme)", self.javascript)
        self.assertIn("onOptionsChanged(applyConfiguredTheme)", self.javascript)
        self.assertIn("delete root.dataset.sam3Theme", self.javascript)
        self.assertIn("root.dataset.sam3Theme = currentSlug", self.javascript)
        self.assertIn("sam-extra.appearance-theme.v1", self.javascript)

    def test_style_imports_locked_tokens_and_has_hallmark_stamp(self):
        self.assertTrue(
            self.css.startswith("/* Hallmark · pre-emit critique:"),
            "style.css must keep the Hallmark app stamp as its first line",
        )
        self.assertIn('@import url("./tokens.css");', self.css)
        self.assertIn("design-system: design.md", self.css)
        self.assertIn("# Design — sam-extra Forge Appearance", self.design)

    def test_each_custom_palette_has_a_complete_semantic_contract(self):
        required = (
            "--sam3-color-paper:",
            "--sam3-color-paper-2:",
            "--sam3-color-paper-3:",
            "--sam3-color-paper-4:",
            "--sam3-color-ink:",
            "--sam3-color-muted:",
            "--sam3-color-rule:",
            "--sam3-color-accent:",
            "--sam3-color-accent-ink:",
            "--sam3-color-focus:",
            "--sam3-color-error:",
            "--sam3-color-success:",
        )
        for slug in (
            "graphite-ember",
            "obsidian-violet",
            "warm-espresso",
            "oled-mono",
        ):
            match = re.search(
                rf'data-sam3-theme="{re.escape(slug)}"\]\s*\{{(?P<body>.*?)\n\}}',
                self.tokens,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(match, slug)
            body = match.group("body")
            for token in required:
                self.assertIn(token, body, f"{slug}: {token}")

    def test_new_theme_layer_avoids_known_css_slop_patterns(self):
        lowered = self.tokens.lower()
        self.assertNotIn("#000", lowered)
        self.assertNotIn("#fff", lowered)
        self.assertNotIn("rgb(", lowered)
        self.assertNotIn("rgba(", lowered)
        self.assertNotIn("linear-gradient", lowered)
        self.assertNotIn("radial-gradient", lowered)
        self.assertNotIn("transition: all", lowered)
        self.assertNotIn("z-index: 9999", lowered)

    def test_global_control_states_and_reduced_motion_are_present(self):
        for marker in (
            ":focus-visible",
            ":active",
            ":disabled",
            '[aria-busy="true"]',
            '[aria-invalid="true"]',
            '[data-state="success"]',
            "@media (prefers-reduced-motion: reduce)",
        ):
            self.assertIn(marker, self.css)

    def test_forge_default_is_an_exact_opt_out(self):
        self.assertNotIn('data-sam3-theme="forge-default"', self.tokens)
        self.assertIn(
            'if (slug === DEFAULT_SLUG) localStorage.removeItem(STORAGE_KEY)',
            self.javascript,
        )

    def test_theme_background_skips_the_lightbox_toolbar(self):
        """Forge's lightbox toolbar is `<div class="modalControls gradio-container">`
        and is meant to stay transparent until hovered. The page-background rule
        for `.gradio-container` uses !important, so without an exclusion it
        painted the toolbar solid and the bar never faded out."""
        start = self.css.index("[data-sam3-theme] :where(.gradio-container")
        selector = self.css[start:self.css.index("{", start)]
        self.assertIn(":not(.modalControls)", selector)
        self.assertNotIn("[data-sam3-theme] :where(.gradio-container) {", self.css)

    def test_custom_palettes_also_set_gradios_dark_class(self):
        """Every palette is dark, but a lot of Forge/Gradio CSS is gated on the
        `dark` class that Gradio only puts on document.body when its resolved
        theme is dark (the default is __theme=system). Without it a light-mode
        session renders near-white body text over light-mode-only surfaces.
        Removal must be limited to the class we added ourselves."""
        self.assertIn("function syncDarkClass()", self.javascript)
        self.assertIn('body.classList.add("dark")', self.javascript)
        self.assertIn("addedDarkClass = true", self.javascript)
        self.assertIn('body.classList.remove("dark")', self.javascript)
        start = self.javascript.index("function syncDarkClass()")
        end = self.javascript.index("function syncThemeRoots()", start)
        body = self.javascript[start:end]
        # Only strip it back when we are the ones who added it.
        self.assertIn("if (addedDarkClass) {", body)
        # And it must run as part of the normal sync path.
        self.assertIn("syncDarkClass();", self.javascript[end:])

    def test_text_token_pairs_meet_wcag_and_the_known_gap_is_declared(self):
        """Measure the palettes instead of trusting the header comment.

        An earlier revision certified "contrast: pass" while the cancel-button
        label sat at 3.98:1 and every control border at 1.36:1, so this computes
        the ratios from tokens.css. Text pairs must clear WCAG 1.4.3 (4.5:1).
        The border gap is a deliberate visual trade-off, so it is asserted as a
        declared limitation rather than silently tolerated.
        """
        import math

        def srgb(lightness, chroma, hue):
            rad = math.radians(hue)
            a, b = chroma * math.cos(rad), chroma * math.sin(rad)
            l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
            m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
            s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
            l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
            rgb = (
                4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
            )

            def encode(value):
                value = max(0.0, min(1.0, value))
                if value <= 0.0031308:
                    return 12.92 * value
                return 1.055 * value ** (1 / 2.4) - 0.055

            return tuple(encode(v) for v in rgb)

        def luminance(rgb):
            def channel(c):
                return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

            r, g, b = (channel(c) for c in rgb)
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        def ratio(one, two):
            a, b = luminance(one), luminance(two)
            hi, lo = max(a, b), min(a, b)
            return (hi + 0.05) / (lo + 0.05)

        def token(block, name):
            found = re.search(
                r"--" + name + r"\s*:\s*oklch\(\s*([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s*\)",
                block,
            )
            self.assertIsNotNone(found, f"{name} missing from a palette")
            return srgb(
                float(found.group(1)) / 100,
                float(found.group(2)),
                float(found.group(3)),
            )

        palettes = re.findall(
            r'\[data-sam3-theme="([^"]+)"\]\s*\{([^}]*)\}', self.tokens
        )
        self.assertGreaterEqual(len(palettes), 4)
        for name, block in palettes:
            with self.subTest(palette=name):
                paper = token(block, "sam3-color-paper")
                paper2 = token(block, "sam3-color-paper-2")
                pairs = {
                    "body text": (token(block, "sam3-color-ink"), paper),
                    "muted text": (token(block, "sam3-color-muted"), paper2),
                    "error text": (token(block, "sam3-color-error-hover"), paper),
                    "cancel label": (
                        token(block, "sam3-color-error-ink"),
                        token(block, "sam3-color-error"),
                    ),
                }
                for label, (fg, bg) in pairs.items():
                    self.assertGreaterEqual(
                        round(ratio(fg, bg), 2), 4.5, f"{name}: {label}"
                    )

        # The border shortfall is real; keep it documented so nobody re-certifies
        # a blanket pass without measuring.
        self.assertIn("KNOWN GAP: --sam3-color-rule", self.css)


if __name__ == "__main__":
    unittest.main()
