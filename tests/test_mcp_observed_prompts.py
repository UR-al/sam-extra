"""sam-extra MCP server: what the prompts of past generations may decide.

Hidden upstream while a hashed checkpoint title never matched its history (see
test_mcp_checkpoint_titles.py); once it matched, on a live Forge 2.29.2 an Anima 3.8B checkpoint
whose 17 past prompts were Anima-style tags ("absurdres, highres, newest, 1boy, solo, short hair")
came back as:

- prompt_dialect "illustrious", confidence high: one prompt said "masterpiece, best quality", and
  observed vocabulary outranked the architecture. Anima's own quality prefix has those words (and
  score_7 / score_9, which made it read as Pony), and an Illustrious answer for an Anima checkpoint
  names another architecture's dialect. Past prompts now only choose among the lineages of an
  ambiguous architecture (xl) or confirm the one the architecture implies.
- prompt_style "natural language (94% of past generations were prose)": the history counted a
  prompt as tags only with "score_9", "masterpiece", "best quality" or underscores, and Anima asks
  for spaces. It now also uses the dialect module's tag test (subject-count and quality vocabulary,
  short comma-separated fragments).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _mcp_support import DEFAULT_OPTIONS, HAS_HTTPX, infotext, make_service, png_bytes

from sam_extra_mcp.forgeneo import dialects
from sam_extra_mcp.forgeneo.history import _looks_danbooru
from sam_extra_mcp.forgeneo.identity import resolve

ANIMA_TAGS = "absurdres, highres, newest, 1boy, solo, male focus, full body, short hair, black hair"
QUALITY_TAGS = "masterpiece, best quality, score_7, 1girl, solo, long hair, smile"
PONY_LADDER = "score_9, score_8_up, score_7_up, 1girl, solo"
PROSE = "A quiet harbour at dawn, fishing boats resting on still water while gulls circle overhead."


class ArchitectureOutranksVocabularyTests(unittest.TestCase):
    def test_an_anima_checkpoint_stays_anima_whatever_its_quality_tags(self):
        for prompts in ([ANIMA_TAGS] * 4 + [QUALITY_TAGS], [PONY_LADDER] * 5, [QUALITY_TAGS] * 6):
            result = resolve(identifier=None, architecture="anima", observed_prompts=prompts)
            self.assertIs(result.dialect, dialects.ANIMA, prompts[0])
            self.assertEqual(result.source, "architecture")

    def test_prompts_that_agree_with_the_architecture_still_count(self):
        result = resolve(identifier=None, architecture="anima", observed_prompts=[ANIMA_TAGS] * 5)
        self.assertIs(result.dialect, dialects.ANIMA)
        self.assertEqual(result.source, "observed prompts")
        self.assertEqual(result.confidence, "high")

    def test_a_natural_language_architecture_ignores_tagged_habits(self):
        result = resolve(identifier=None, architecture="flux", observed_prompts=[QUALITY_TAGS] * 6)
        self.assertIs(result.dialect, dialects.NATURAL)
        self.assertEqual(result.source, "architecture")

    def test_the_ambiguous_architecture_is_still_read_from_prompts(self):
        result = resolve(identifier=None, architecture="xl", observed_prompts=[QUALITY_TAGS] * 6)
        self.assertIs(result.dialect, dialects.ILLUSTRIOUS)
        self.assertEqual(result.source, "observed prompts")
        result = resolve(identifier=None, architecture="xl", observed_prompts=[PONY_LADDER] * 6)
        self.assertIs(result.dialect, dialects.PONY)

    def test_too_few_prompts_still_fall_back(self):
        result = resolve(identifier=None, architecture="xl", observed_prompts=[QUALITY_TAGS] * 2)
        self.assertIsNone(result.dialect)


class HistoryTagCountTests(unittest.TestCase):
    def test_anima_style_tags_with_spaces_count_as_tags(self):
        self.assertIs(_looks_danbooru(ANIMA_TAGS), True)
        self.assertIs(_looks_danbooru("short hair, black hair, red eyes, smile, upper body, white background"), True)

    def test_prose_is_still_prose(self):
        self.assertIs(_looks_danbooru(PROSE), False)
        self.assertIs(_looks_danbooru("a photograph of a mountain at sunrise"), False)


@unittest.skipUnless(HAS_HTTPX, "httpx is required for the fake Forge transport")
class LiveShapeTests(unittest.TestCase):
    """The live case: an Anima checkpoint under a hashed title, tag prompts in its history."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        options = dict(DEFAULT_OPTIONS, sd_model_checkpoint="Anima/animeMix_v10.safetensors [abc123]")
        self.service, _fake, tree = make_service(base, options=options)
        self.addCleanup(self.service.close)
        folder = tree["forge"] / "output" / "txt2img-images" / "2026-10-03"
        folder.mkdir(parents=True)
        prompts = [ANIMA_TAGS] * 16 + [QUALITY_TAGS]
        for seed, prompt in enumerate(prompts, start=1):
            text = infotext(prompt, seed, steps=20, cfg=5.0, model="animeMix_v10")
            (folder / f"{seed:05}-{seed}.png").write_bytes(png_bytes(text))

    def tearDown(self):
        self._tmp.cleanup()

    def test_prompt_dialect_and_style(self):
        dialect = self.service.prompt_dialect()
        self.assertEqual(dialect["dialect"], "anima", dialect)
        profile = self.service.model_profile()
        self.assertEqual(profile["samples_observed"], 17)
        self.assertTrue(profile["prompt_style"].startswith("danbooru tags"), profile["prompt_style"])


if __name__ == "__main__":
    unittest.main()
