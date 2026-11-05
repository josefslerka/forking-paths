from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from apertus.garden.api.post_routes import POST_ROUTES
from apertus.garden.domains.tokenization import TokenizationService
from apertus.garden.server import GardenHandler, ModelState


class CodePointTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    all_special_ids = [1, 2]
    vocab_size = 0x110000 + 100

    @staticmethod
    def _token_id(character: str) -> int:
        return ord(character) + 100

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        ids = [self._token_id(character) for character in text]
        return [self.bos_token_id, *ids] if add_special_tokens else ids

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return "".join(
            chr(token_id - 100)
            for token_id in token_ids
            if token_id not in self.all_special_ids
        )

    def convert_ids_to_tokens(self, token_id: int) -> str:
        if token_id == self.bos_token_id:
            return "<s>"
        if token_id == self.eos_token_id:
            return "</s>"
        return f"U+{token_id - 100:04X}"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list]:
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": self.encode(text, add_special_tokens=False),
            "offset_mapping": [
                (index, index + 1)
                for index in range(len(text))
            ],
        }


class ByteSplitTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    all_special_ids = [1, 2]
    vocab_size = 131_072
    text = "ma\u0301m"
    ids = [1831, 1204, 86954]

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        assert text == self.text
        return [self.bos_token_id, *self.ids] if add_special_tokens else self.ids

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        ordinary = [
            token_id
            for token_id in token_ids
            if token_id not in self.all_special_ids
        ]
        if ordinary == self.ids:
            return self.text
        return {
            (1831,): "ma",
            (1204,): "\ufffd",
            (86954,): "\ufffdm",
            (1831, 1204): "ma\ufffd",
        }.get(tuple(ordinary), self.text)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return {
            1: "<s>",
            2: "</s>",
            1831: "ma",
            1204: "Ì",
            86954: "ģm",
        }[token_id]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list]:
        assert text == self.text
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": self.ids,
            "offset_mapping": [(0, 2), (2, 3), (2, 4)],
        }


class TokenizationAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ModelState(model_path=Path("."), mock=False)
        self.state.status = "ready"
        self.state.active_model_id = "bf16"
        self.state.model_generation = 7
        self.state.tokenizer = CodePointTokenizer()

    def test_exact_offsets_utf8_ranges_roundtrip_and_boundaries(self) -> None:
        result = self.state.tokenization(" Aé\t")
        trace = result["a"]

        self.assertEqual(result["analysis_kind"], "tokenization")
        self.assertEqual(result["execution"]["forward_passes"], 0)
        self.assertFalse(result["execution"]["model_weights_used"])
        self.assertEqual(trace["token_count"], 4)
        self.assertEqual(trace["unicode"]["code_point_count"], 4)
        self.assertEqual(trace["unicode"]["utf8_byte_count"], 5)
        self.assertEqual(
            trace["encoded_with_special_token_ids"],
            [1, 132, 165, 333, 109],
        )
        self.assertTrue(trace["roundtrip_exact"])
        self.assertEqual(trace["detokenized_text"], " Aé\t")
        self.assertEqual(trace["offset_coverage"]["ratio"], 1.0)

        accented = trace["tokens"][2]
        self.assertEqual(
            (
                accented["character_start"],
                accented["character_end"],
            ),
            (2, 3),
        )
        self.assertEqual(
            (
                accented["utf8_byte_start"],
                accented["utf8_byte_end"],
                accented["utf8_byte_count"],
            ),
            (2, 4, 2),
        )
        self.assertEqual(accented["utf8_hex"], "C3 A9")
        self.assertEqual(accented["code_points"], ["U+00E9"])

        boundary = trace["boundaries"][2]
        self.assertEqual(boundary["prefix_text"], " A")
        self.assertEqual(boundary["prefix_token_ids"], [1, 132, 165])
        self.assertEqual(boundary["next_token"]["source_text"], "é")
        self.assertFalse(boundary["at_end"])
        self.assertEqual(len(trace["boundaries"]), 5)
        self.assertTrue(trace["boundaries"][-1]["at_end"])

    def test_model_state_facade_delegates_to_explicit_domain_service(
        self,
    ) -> None:
        service = TokenizationService(
            tokenizer=self.state.tokenizer,
            mock=False,
            mock_words=(),
        )

        self.assertEqual(
            self.state._text_surprisal_tokenization("Aé"),
            service.text_tokenization("Aé"),
        )
        self.assertEqual(
            self.state._tokenization_text_payload("Aé", label="Text A"),
            service.text_payload("Aé", label="Text A"),
        )

    def test_canonical_equivalence_can_hide_token_divergence(self) -> None:
        result = self.state.tokenization("mám", "ma\u0301m")
        comparison = result["comparison"]

        self.assertFalse(comparison["same_text"])
        self.assertFalse(comparison["same_utf8_bytes"])
        self.assertTrue(comparison["canonically_equivalent_nfc"])
        self.assertFalse(comparison["identical_token_ids"])
        self.assertEqual(comparison["token_count_a"], 3)
        self.assertEqual(comparison["token_count_b"], 4)
        self.assertEqual(comparison["common_token_prefix_count"], 1)
        self.assertEqual(
            comparison["first_token_divergence"]["index"],
            1,
        )
        self.assertEqual(
            comparison["first_character_divergence"],
            {
                "index": 1,
                "a": "U+00E1",
                "b": "U+0061",
            },
        )

    def test_leading_space_is_an_observable_token_difference(self) -> None:
        result = self.state.tokenization("Praha", " Praha")
        comparison = result["comparison"]

        self.assertFalse(comparison["same_text"])
        self.assertFalse(comparison["identical_token_ids"])
        self.assertEqual(comparison["common_token_prefix_count"], 0)
        self.assertEqual(
            comparison["first_token_divergence"]["a"]["source_text"],
            "P",
        )
        self.assertEqual(
            comparison["first_token_divergence"]["b"]["source_text"],
            " ",
        )

    def test_whole_words_are_server_defined_unicode_spans(self) -> None:
        text = "Žluťoučký, e-mail O’Neill 123 ma\u0301m"
        trace = self.state.tokenization(text)["a"]

        self.assertEqual(
            [span["text"] for span in trace["word_spans"]],
            ["Žluťoučký", "e-mail", "O’Neill", "123", "ma\u0301m"],
        )
        self.assertEqual(
            trace["word_span_definition"]["kind"],
            "maximal_unicode_orthographic_span_v1",
        )
        self.assertEqual(
            trace["word_span_definition"]["linguistic_claim"],
            "observable_text_span_not_lemma_or_morphological_word",
        )
        self.assertEqual(
            trace["word_spans"][-1]["token_count"],
            len("ma\u0301m"),
        )

    def test_repeated_words_and_astral_prefix_keep_exact_ranges(self) -> None:
        text = "🙂 zámek, zámek"
        trace = self.state.tokenization(text)["a"]
        spans = trace["word_spans"]

        self.assertEqual([span["text"] for span in spans], ["zámek", "zámek"])
        self.assertEqual(
            [
                (span["character_start"], span["character_end"])
                for span in spans
            ],
            [(2, 7), (9, 14)],
        )
        self.assertEqual(
            [
                (span["utf8_byte_start"], span["utf8_byte_end"])
                for span in spans
            ],
            [(5, 11), (13, 19)],
        )

    def test_empty_optional_side_is_rejected_only_when_supplied(self) -> None:
        self.assertIsNone(self.state.tokenization("A")["b"])
        with self.assertRaisesRegex(ValueError, "Text B"):
            self.state.tokenization("A", "")

    def test_mock_tokenization_does_not_require_real_tokenizer_object(
        self,
    ) -> None:
        state = ModelState(model_path=Path("."), mock=True)
        state.status = "ready"

        result = state.tokenization("Praha", "Brno")

        self.assertEqual(result["analysis_kind"], "tokenization")
        self.assertIsNone(state.tokenizer)
        self.assertEqual(
            len(result["a"]["encoded_with_special_token_ids"]),
            len(result["b"]["encoded_with_special_token_ids"]),
        )

    def test_byte_level_ranges_remain_exact_when_token_splits_code_point(
        self,
    ) -> None:
        self.state.tokenizer = ByteSplitTokenizer()
        result = self.state.tokenization(ByteSplitTokenizer.text)
        trace = result["a"]

        self.assertEqual(
            [
                (token["utf8_byte_start"], token["utf8_byte_end"])
                for token in trace["tokens"]
            ],
            [(0, 2), (2, 3), (3, 5)],
        )
        self.assertTrue(trace["tokens"][1]["splits_unicode_code_point"])
        self.assertEqual(trace["tokens"][1]["utf8_hex"], "CC")
        self.assertEqual(
            (
                trace["tokens"][1]["covering_character_start"],
                trace["tokens"][1]["covering_character_end"],
            ),
            (2, 3),
        )
        self.assertEqual(trace["offset_coverage"]["utf8_byte_ratio"], 1.0)
        self.assertEqual(trace["offset_coverage"]["character_ratio"], 1 / 3)
        self.assertEqual(trace["boundaries"][2]["utf8_byte_offset"], 3)
        self.assertIsNone(trace["boundaries"][2]["character_offset"])
        self.assertFalse(trace["boundaries"][2]["prefix_roundtrip_exact"])
        self.assertEqual(trace["boundaries"][-1]["prefix_text"], self.state.tokenizer.text)
        self.assertTrue(trace["boundaries"][-1]["prefix_roundtrip_exact"])
        self.assertEqual(len(trace["word_spans"]), 1)
        self.assertEqual(trace["word_spans"][0]["text"], self.state.tokenizer.text)
        self.assertEqual(trace["word_spans"][0]["token_indices"], [0, 1, 2])
        self.assertEqual(trace["word_spans"][0]["token_ids"], self.state.tokenizer.ids)
        self.assertEqual(
            (
                trace["word_spans"][0]["character_start"],
                trace["word_spans"][0]["character_end"],
                trace["word_spans"][0]["utf8_byte_start"],
                trace["word_spans"][0]["utf8_byte_end"],
            ),
            (0, 4, 0, 5),
        )
        self.assertIn(
            "tokens_split_utf8_code_point",
            {warning["code"] for warning in result["warnings"]},
        )


class TokenizationRouteAndPageTest(unittest.TestCase):
    def test_page_and_api_routes_are_registered(self) -> None:
        get_source = inspect.getsource(GardenHandler.do_GET)
        route = POST_ROUTES["/api/tokenization"]
        controller_source = inspect.getsource(
            getattr(GardenHandler, route.controller)
        )

        self.assertIn('"/tokenizace"', get_source)
        self.assertIn("TOKENIZATION_FILE", get_source)
        self.assertEqual(route.controller, "_post_tokenization")
        self.assertIn("self.state.tokenization", controller_source)

    def test_static_page_exposes_comparison_boundaries_and_handoffs(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "static"
            / "tokenization.html"
        ).read_text(encoding="utf-8")

        self.assertIn('data-workbench-page="tokenization"', source)
        self.assertIn('id="text-a"', source)
        self.assertIn('id="text-b"', source)
        self.assertIn('id="boundary-slider"', source)
        self.assertIn('id="open-garden"', source)
        self.assertIn('id="open-surprise"', source)
        self.assertIn('id="open-embedding"', source)
        self.assertIn("/api/tokenization", source)
        self.assertIn("apertus.garden-handoff", source)
        self.assertIn("apertus.embedding-handoff", source)


if __name__ == "__main__":
    unittest.main()
