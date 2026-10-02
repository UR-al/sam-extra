"""TIPO 프롬프트 규칙 — 사용자 프롬프트 → TIPO 입력, TIPO 출력 → Anima 프롬프트(원문은 그대로, 새 것만 뒤에)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sam3ext.tipo import prompt as tp  # noqa: E402

CSV_ROWS = """hatsune_miku,4,90000,"miku,初音ミク"
vocaloid,3,80000,
shimakaze_(kancolle),4,5000,
kantai_collection,3,70000,kancolle
wlop,1,3000,
ke-ta,1,2000,
highres,5,900000,
smile,0,500000,
"""


def _index():
    return tp.TagIndex({
        "hatsune miku": 4, "miku": 4, "vocaloid": 3, "shimakaze (kancolle)": 4,
        "kantai collection": 3, "kancolle": 3, "wlop": 1, "ke-ta": 1, "highres": 5, "smile": 0,
    })


class TagIndexTests(unittest.TestCase):
    def test_csv_loader_reads_categories_and_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "danbooru_2025.csv"
            path.write_bytes(("\ufeff" + CSV_ROWS).encode("utf-8"))
            index = tp.load_tag_index([Path(tmp) / "missing.csv", path])
        self.assertEqual(index.category("hatsune miku"), "character")
        self.assertEqual(index.category("Hatsune_Miku"), "character")
        self.assertEqual(index.category("miku"), "character", "별칭")
        self.assertEqual(index.category("kancolle"), "copyright")
        self.assertEqual(index.category("shimakaze \\(kancolle\\)"), "character")
        self.assertEqual(index.category("wlop"), "artist")
        self.assertEqual(index.category("highres"), "meta")
        self.assertEqual(index.category("smile"), "general")
        self.assertEqual(index.category("never heard of it"), "general")

    def test_no_csv_gives_an_empty_index(self):
        index = tp.load_tag_index([Path("C:/nope/a.csv")])
        self.assertEqual(index.category("hatsune miku"), "general")

    def test_default_paths_point_at_tagcomplete(self):
        paths = tp.default_tag_csv_paths(Path("C:/forge/extensions"))
        self.assertEqual(paths[0], Path("C:/forge/extensions/sd-webui-tagcomplete-neo/tags/danbooru_2025.csv"))


class AnalyzeTests(unittest.TestCase):
    TEXT = (
        "masterpiece, best quality, score_7, year 2025, safe, 1girl, hatsune miku, vocaloid, @ke-ta, "
        "(smile:1.2), <lora:style:0.8>, __colors__, {red|blue} hair, "
        "a girl standing in the rain under the stars, looking at viewer"
    )

    def test_prompt_is_split_into_tipo_fields(self):
        parts = tp.analyze_prompt(self.TEXT, _index())
        self.assertEqual(parts.quality, "masterpiece")
        self.assertEqual(parts.era, "newest")
        self.assertEqual(parts.rating, "safe")
        self.assertEqual(parts.artists, ["ke-ta"])
        self.assertEqual(parts.characters, ["hatsune miku"])
        self.assertEqual(parts.copyrights, ["vocaloid"])
        self.assertEqual(parts.tags, ["1girl", "smile", "looking at viewer"], "인원수 먼저, 로라·와일드카드·{a|b}·문장은 빼고")
        self.assertTrue(parts.has_person)
        for key in ("smile", "score_7", "masterpiece", "Hatsune_Miku", "@ke-ta"):
            self.assertIn(tp.normalize(key), parts.known, "중복 판정은 정규화한 키로")

    def test_year_and_rating_mapping(self):
        cases = [("year 2024", "newest"), ("year 2021", "recent"), ("year 2019", "mid"), ("year 2016", "early"),
                 ("year 2010", "old"), ("recent", "recent")]
        for text, era in cases:
            with self.subTest(text=text):
                self.assertEqual(tp.analyze_prompt(text, tp.TagIndex({})).era, era)
        self.assertEqual(tp.analyze_prompt("general", tp.TagIndex({})).rating, "safe")
        self.assertEqual(tp.analyze_prompt("questionable", tp.TagIndex({})).rating, "nsfw")

    def test_quality_keeps_the_best_tipo_word(self):
        parts = tp.analyze_prompt("good quality, best quality, high quality", tp.TagIndex({}))
        self.assertEqual(parts.quality, "best quality")
        self.assertIsNone(tp.analyze_prompt("high quality, score_9", tp.TagIndex({})).quality)

    def test_plain_artist_known_to_the_index_and_escaped_character(self):
        parts = tp.analyze_prompt("wlop, shimakaze \\(kancolle\\), kancolle, highres", _index())
        self.assertEqual(parts.artists, ["wlop"])
        self.assertEqual(parts.characters, ["shimakaze (kancolle)"])
        self.assertEqual(parts.copyrights, ["kancolle"])
        self.assertEqual((parts.meta, parts.tags), (["highres"], []), "메타 태그는 태그 줄이 아니라 meta 줄로")

    def test_explicit_uses_the_v21_form_and_the_most_severe_rating_wins(self):
        index = tp.TagIndex({})
        self.assertEqual(tp.analyze_prompt("explicit, 1girl", index).rating, "nsfw, explicit", "v2.1 변환표")
        self.assertEqual(tp.analyze_prompt("masterpiece, nsfw, explicit, 1girl", index).rating, "nsfw, explicit")
        self.assertEqual(tp.analyze_prompt("safe, sensitive", index).rating, "sensitive")
        self.assertEqual(tp.analyze_prompt("nsfw, general", index).rating, "nsfw")

    def test_sentences_with_commas_stay_out_of_the_tag_line(self):
        index = tp.TagIndex({})
        own_line = tp.analyze_prompt("1girl, solo, (smile:1.2)\nA girl, standing in the rain, looks up at the sky.",
                                     index)
        self.assertEqual(own_line.tags, ["1girl", "solo", "smile"])
        same_line = tp.analyze_prompt("1girl, solo, She is smiling, holding an umbrella, looks up.", index)
        self.assertEqual(same_line.tags, ["1girl", "solo"])
        tag_line_with_period = tp.analyze_prompt("1girl, long hair, solo.", index)
        self.assertEqual(tag_line_with_period.tags, ["1girl", "long hair"])
        self.assertIn("solo", tag_line_with_period.known, "문장 쪽으로 뺀 조각도 중복 판정에는 쓴다")
        known_tag = tp.analyze_prompt("1girl, looking at viewer, She smiles.", tp.TagIndex({"looking at viewer": 0}))
        self.assertEqual(known_tag.tags, ["1girl", "looking at viewer"], "사전에 있는 태그는 문장으로 묶지 않는다")
        middle = tp.analyze_prompt("1girl, She stands on a stage, singing, under bright lights.",
                                   tp.TagIndex({"singing": 0}))
        self.assertEqual(middle.tags, ["1girl"], "문장이 시작된 뒤 끝나기 전의 조각은 사전에 있어도 문장")
        self.assertEqual(tp.analyze_prompt("a girl standing in the rain under the stars, looking at viewer",
                                           index).tags, ["looking at viewer"], "문장 부호가 없으면 긴 조각만 문장")

    def test_weighted_and_nested_emphasis(self):
        parts = tp.analyze_prompt("1girl, (@wlop:0.8), ((smile)), (smile: 1.2), [blush]", tp.TagIndex({}))
        self.assertEqual(parts.artists, ["wlop"], "가중치를 준 @작가도 작가 줄로, @ 없이")
        self.assertEqual(parts.tags, ["1girl", "smile", "blush"], "같은 태그는 한 번만")

    def test_break_and_unbalanced_groups(self):
        parts = tp.analyze_prompt("1girl BREAK smile, (red dress, blue sky:1.1), solo", tp.TagIndex({}))
        self.assertEqual(parts.tags, ["1girl", "smile", "solo"])


class NormalizeTests(unittest.TestCase):
    def test_emphasis_weights_and_underscores(self):
        cases = [
            ("(smile: 1.2)", "smile"), ("(smile:1.2 )", "smile"), ("( smile )", "smile"), ("((smile))", "smile"),
            ("[smile]", "smile"), ("(smile:.5)", "smile"), ("(@wlop:0.8)", "wlop"), ("@wlop", "wlop"),
            ("^_^", "^_^"), ("long_hair", "long hair"), ("Hatsune_Miku", "hatsune miku"),
            ("hatsune miku \\(cosplay\\)", "hatsune miku (cosplay)"),
            ("(hatsune miku \\(cosplay\\):1.1)", "hatsune miku (cosplay)"),
            ("(a) b (c)", "(a) b (c)"),
        ]
        for raw, key in cases:
            with self.subTest(raw=raw):
                self.assertEqual(tp.normalize(raw), key)


class BuildPromptTests(unittest.TestCase):
    def setUp(self):
        self.parts = tp.analyze_prompt(
            "masterpiece, year 2025, safe, 1girl, hatsune miku, vocaloid, @ke-ta, smile", _index()
        )

    def test_tags_and_description(self):
        self.assertEqual(
            tp.build_tipo_prompt(self.parts, tp.MODE_TAGS_NL, tp.LENGTH_NORMAL, 832 / 1216),
            "quality: masterpiece, newest\n"
            "rating: safe\n"
            "artist: ke-ta\n"
            "characters: hatsune miku\n"
            "copyrights: vocaloid\n"
            "aspect ratio: 0.7\n"
            "target: <|long|> <|tag_to_long|>\n"
            "tag: 1girl, smile",
        )

    def test_tags_only_and_description_only(self):
        tags_only = tp.build_tipo_prompt(self.parts, tp.MODE_TAGS, tp.LENGTH_SHORT, None)
        self.assertTrue(tags_only.endswith("target: <|short|>\ntag: 1girl, smile"))
        self.assertNotIn("aspect ratio", tags_only)
        nl_only = tp.build_tipo_prompt(self.parts, tp.MODE_NL, tp.LENGTH_LONG, None)
        self.assertTrue(nl_only.endswith("target: <|very_long|> <|tag_to_long|>\ntag: 1girl, smile\n"),
                        "설명만: 태그 줄을 닫아 태그를 더 잇지 않게")

    def test_meta_and_explicit_lines(self):
        parts = tp.analyze_prompt("explicit, 1girl, highres, smile", _index())
        text = tp.build_tipo_prompt(parts, tp.MODE_TAGS, tp.LENGTH_NORMAL, None)
        self.assertEqual(text, "rating: nsfw, explicit\nmeta: highres\ntarget: <|long|>\ntag: 1girl, smile")

    def test_empty_fields_are_left_out(self):
        text = tp.build_tipo_prompt(tp.analyze_prompt("", tp.TagIndex({})), tp.MODE_TAGS, tp.LENGTH_NORMAL, None)
        self.assertEqual(text, "target: <|long|>\ntag:")


class ParseOutputTests(unittest.TestCase):
    def test_tag_continuation_then_long_caption(self):
        tags, description = tp.parse_tipo_output(
            tp.MODE_TAGS_NL, ", outdoors, cherry blossoms, petals\nlong: A girl smiles under the cherry blossoms.\n"
        )
        self.assertEqual(tags, ["outdoors", "cherry blossoms", "petals"])
        self.assertEqual(description, "A girl smiles under the cherry blossoms.")

    def test_tags_only_and_description_only(self):
        self.assertEqual(tp.parse_tipo_output(tp.MODE_TAGS, " outdoors, sky\n"), (["outdoors", "sky"], ""))
        self.assertEqual(tp.parse_tipo_output(tp.MODE_NL, "long: A quiet street.\n"), ([], "A quiet street."))

    def test_short_fallback_and_wrapped_caption(self):
        self.assertEqual(tp.parse_tipo_output(tp.MODE_NL, "short: A girl.\n"), ([], "A girl."))
        tags, description = tp.parse_tipo_output(tp.MODE_TAGS_NL, ", sky\nlong: First line\nsecond line.\n")
        self.assertEqual((tags, description), (["sky"], "First line second line."))

    def test_cut_off_output_drops_the_incomplete_tail(self):
        self.assertEqual(tp.parse_tipo_output(tp.MODE_TAGS, ", long hair, rain, wet cl", finished=False),
                         (["long hair", "rain"], ""))
        self.assertEqual(tp.parse_tipo_output(tp.MODE_TAGS, ", long hair, rain, wet cl"),
                         (["long hair", "rain", "wet cl"], ""), "끝까지 생성했으면(EOS) 그대로")
        self.assertEqual(
            tp.parse_tipo_output(tp.MODE_TAGS_NL, ", rain, wet clothes\nlong: A girl stands in the rain. Her dre",
                                 finished=False),
            (["rain", "wet clothes"], "A girl stands in the rain."))
        self.assertEqual(tp.parse_tipo_output(tp.MODE_TAGS_NL, ", rain\nlong: A girl stan", finished=False),
                         (["rain"], ""), "끝난 문장이 없으면 설명은 버린다")
        self.assertEqual(tp.parse_tipo_output(tp.MODE_TAGS_NL, ", rain, wet cl", finished=False), (["rain"], ""))
        self.assertEqual(tp.parse_tipo_output(tp.MODE_TAGS, ", rain, wet clothes\n", finished=False),
                         (["rain", "wet clothes"], ""), "줄이 끝난 뒤 잘렸으면 마지막 태그도 온전하다")
        self.assertEqual(tp.parse_tipo_output(tp.MODE_NL, "long: One! Two? Thr", finished=False), ([], "One! Two?"))


class AssembleTests(unittest.TestCase):
    USER = "masterpiece, 1girl, hatsune miku, (smile:1.2), <lora:style:0.8>, "

    def _assemble(self, new_tags, description="", mode=None, allow=False, user=None):
        user = self.USER if user is None else user
        parts = tp.analyze_prompt(user, _index())
        return tp.assemble_prompt(user, parts, new_tags, description, mode or tp.MODE_TAGS_NL, allow, _index())

    def test_user_text_is_kept_verbatim_and_new_things_follow(self):
        text, added = self._assemble(["outdoors", "cherry blossoms"], "A girl (smiling) under petals.")
        self.assertTrue(text.startswith("masterpiece, 1girl, hatsune miku, (smile:1.2), <lora:style:0.8>, outdoors"))
        self.assertEqual(text, "masterpiece, 1girl, hatsune miku, (smile:1.2), <lora:style:0.8>, outdoors, cherry blossoms\n"
                               "A girl \\(smiling\\) under petals.")
        self.assertEqual(added, ["outdoors", "cherry blossoms"])

    def test_duplicates_quality_rating_era_and_person_tags_are_dropped(self):
        _, added = self._assemble(["Smile", "smile", "masterpiece", "great quality", "score_8", "newest", "safe",
                                   "2girls", "solo", "outdoors", "outdoors"])
        self.assertEqual(added, ["solo", "outdoors"])

    def test_weighted_user_tags_and_meta_tags_are_not_added_again(self):
        user = "1girl, (smile: 1.2), ((blush)), (@wlop:0.8)"
        parts = tp.analyze_prompt(user, _index())
        _, added = tp.assemble_prompt(user, parts, ["smile", "blush", "wlop", "highres", "outdoors"], "",
                                      tp.MODE_TAGS, True, _index())
        self.assertEqual(added, ["outdoors"], "가중치 붙은 태그·@작가와 같은 것, 메타 태그(highres)는 붙이지 않는다")

    def test_brackets_and_underscores(self):
        _, added = self._assemble(["long_hair", "looking_at_viewer", "^_^", "shimakaze (kancolle)"], allow=True)
        self.assertEqual(added, ["shimakaze \\(kancolle\\)", "long hair", "looking at viewer", "^_^"])

    def test_new_names_need_the_checkbox(self):
        new = ["kancolle", "wlop", "shimakaze (kancolle)", "outdoors"]
        self.assertEqual(self._assemble(new, allow=False)[1], ["outdoors"])
        self.assertEqual(self._assemble(new, allow=True)[1],
                         ["shimakaze \\(kancolle\\)", "kancolle", "@wlop", "outdoors"],
                         "캐릭터 → 작품 → @작가 → 일반")

    def test_modes(self):
        text, added = self._assemble(["outdoors"], "A girl.", mode=tp.MODE_TAGS)
        self.assertNotIn("A girl", text)
        self.assertEqual(added, ["outdoors"])
        text, added = self._assemble(["outdoors"], "A girl.", mode=tp.MODE_NL)
        self.assertEqual(added, [])
        self.assertTrue(text.endswith("<lora:style:0.8>\nA girl."))

    def test_empty_user_text(self):
        text, _ = self._assemble(["1girl", "smile"], "", user="")
        self.assertEqual(text, "1girl, smile")


if __name__ == "__main__":
    unittest.main()
