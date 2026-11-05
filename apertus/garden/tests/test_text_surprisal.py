from __future__ import annotations

import json
import math
import tempfile
import threading
import time
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from apertus.garden.api.post_routes import POST_ROUTES
from apertus.garden.server import (
    MAX_TEXT_SURPRISAL_CHARACTERS,
    MAX_TEXT_SURPRISAL_LABEL_CHARACTERS,
    MAX_TEXT_SURPRISAL_TOKENS,
    GardenHandler,
    ModelChangedError,
    ModelState,
)


class CharacterTokenizer:
    bos_token_id = 0
    eos_token_id = 5
    all_special_ids = [0, 5]
    pieces = {
        0: "",
        1: " ",
        2: "A",
        3: "B",
        4: "C",
        5: "",
    }
    ids = {value: key for key, value in pieces.items() if value}

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        token_ids = [self.ids[character] for character in text]
        return [self.bos_token_id, *token_ids] if add_special_tokens else token_ids

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return "".join(
            self.pieces[token_id]
            for token_id in token_ids
            if not skip_special_tokens or token_id not in self.all_special_ids
        )

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return {
            0: "<s>",
            1: "Ġ",
            2: "A",
            3: "B",
            4: "C",
            5: "</s>",
        }[token_id]

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
                (index, index + 1) for index in range(len(text))
            ],
        }


class ByteFragmentTokenizer:
    bos_token_id = 0
    eos_token_id = None
    all_special_ids = [0]
    byte_by_id = {
        1: 0xF0,
        2: 0x9F,
        3: 0x98,
        4: 0x80,
    }

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        if text != "😀":
            raise ValueError("Testovací tokenizer zná pouze emoji.")
        token_ids = [1, 2, 3, 4]
        return [self.bos_token_id, *token_ids] if add_special_tokens else token_ids

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        payload = bytes(
            self.byte_by_id[token_id]
            for token_id in token_ids
            if token_id in self.byte_by_id
        )
        return payload.decode("utf-8", errors="replace")

    def convert_ids_to_tokens(self, token_id: int) -> str:
        if token_id == self.bos_token_id:
            return "<s>"
        return f"<0x{self.byte_by_id[token_id]:02X}>"


def deterministic_probabilities() -> np.ndarray:
    return np.array(
        [
            [0.10, 0.40, 0.20, 0.20, 0.05, 0.05],
            [0.05, 0.15, 0.10, 0.60, 0.05, 0.05],
            [0.25, 0.25, 0.25, 0.125, 0.075, 0.05],
        ],
        dtype=np.float64,
    )


class TextSurprisalMathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ModelState(model_path=Path("."), mock=False)
        self.state.status = "ready"
        self.state.active_model_id = "bf16"
        self.state.model_generation = 7
        self.state.transition_id = 3
        self.state.tokenizer = CharacterTokenizer()

    def run_analysis(self) -> tuple[dict, object]:
        logits = np.log(deterministic_probabilities())
        patcher = patch.object(
            self.state,
            "_text_surprisal_forward_logits",
            return_value=logits,
        )
        forward = patcher.start()
        self.addCleanup(patcher.stop)
        result = self.state.text_surprisal(
            " AB  ",
            3,
            "bf16",
            7,
            self.state.server_instance_id,
        )
        return result, forward

    def test_hybrid_provenance_names_quantizer_and_exact_bf16_io(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory)
            (model_path / "config.json").write_text(
                json.dumps(
                    {
                        "quantization": {
                            "bits": 8,
                            "group_size": 64,
                            "mode": "affine",
                        }
                    }
                ),
                encoding="utf-8",
            )
            state = ModelState(
                model_path=model_path,
                model_paths={
                    "hybrid": model_path,
                    "bf16": model_path,
                },
                active_model_id="hybrid",
                mock=False,
            )
            with patch(
                "apertus.garden.server.metadata_fingerprint",
                return_value=("f" * 64, []),
            ):
                composition = state._model_provenance_snapshot_unlocked()[
                    "runtime_composition"
                ]

        self.assertEqual(
            composition["kind"],
            "quantized_body_with_bf16_input_output",
        )
        self.assertEqual(
            composition["quantized_body"]["quantization"],
            {"bits": 8, "group_size": 64, "mode": "affine"},
        )
        self.assertEqual(
            composition["bf16_input_output"]["weights"],
            ["model.embed_tokens.weight", "lm_head.weight"],
        )

    def test_bos_alignment_exact_math_and_finite_json(self) -> None:
        result, forward = self.run_analysis()

        forward.assert_called_once_with(
            " AB",
            [0, 1, 2],
            [1, 2, 3],
        )
        self.assertEqual(result["analysis_kind"], "text_surprisal")
        self.assertEqual(result["original_text"], " AB  ")
        self.assertEqual(result["normalized_text"], " AB")
        self.assertEqual(result["text"], " AB")
        self.assertEqual(result["normalization"]["trimmed_trailing_characters"], 2)
        self.assertEqual(result["token_count"], 3)
        self.assertEqual(result["character_count"], 3)
        self.assertEqual(result["utf8_byte_count"], 3)
        self.assertEqual(
            [token["token_id"] for token in result["tokens"]],
            [1, 2, 3],
        )
        self.assertEqual(
            "".join(token["piece"] for token in result["tokens"]),
            result["normalized_text"],
        )
        self.assertNotIn(0, [token["token_id"] for token in result["tokens"]])

        expected_probabilities = [0.4, 0.1, 0.125]
        expected_ranks = [1, 3, 4]
        for token, probability, rank in zip(
            result["tokens"],
            expected_probabilities,
            expected_ranks,
            strict=True,
        ):
            self.assertAlmostEqual(token["probability"], probability)
            self.assertAlmostEqual(token["logprob"], math.log(probability))
            self.assertAlmostEqual(
                token["surprisal_bits"],
                -math.log2(probability),
            )
            self.assertEqual(token["rank"], rank)
            self.assertAlmostEqual(
                token["effective_tokens"],
                2 ** token["entropy_bits"],
            )
            self.assertEqual(
                [alternative["rank"] for alternative in token["alternatives"]],
                sorted(
                    alternative["rank"]
                    for alternative in token["alternatives"]
                ),
            )

        self.assertAlmostEqual(
            result["tokens"][1]["top1_gap_bits"],
            math.log2(0.6 / 0.1),
        )
        self.assertEqual(
            [
                alternative["token_id"]
                for alternative in result["tokens"][2]["alternatives"]
            ],
            [0, 1, 2],
        )
        expected_entropies = [
            -sum(
                probability * math.log2(probability)
                for probability in row
            )
            for row in deterministic_probabilities()
        ]
        for token, entropy in zip(
            result["tokens"],
            expected_entropies,
            strict=True,
        ):
            self.assertAlmostEqual(token["entropy_bits"], entropy)
        expected_total = sum(-math.log2(value) for value in expected_probabilities)
        summary = result["summary"]
        self.assertAlmostEqual(summary["total_surprisal_bits"], expected_total)
        self.assertAlmostEqual(
            summary["mean_surprisal_bits"],
            expected_total / 3,
        )
        self.assertAlmostEqual(
            summary["perplexity"],
            2 ** (expected_total / 3),
        )
        self.assertAlmostEqual(
            summary["bits_per_character"],
            expected_total / 3,
        )
        self.assertAlmostEqual(
            summary["bits_per_utf8_byte"],
            expected_total / 3,
        )
        self.assertAlmostEqual(
            summary["mean_entropy_bits"],
            sum(expected_entropies) / 3,
        )
        self.assertEqual(summary["max_surprisal_token"]["index"], 1)
        self.assertEqual(result["execution"]["forward_passes"], 1)
        self.assertFalse(result["execution"]["generation"])
        self.assertFalse(result["execution"]["steering"])
        measured_phases = sum(
            result["timing_ms"][key]
            for key in (
                "tokenization",
                "forward",
                "summary",
                "aggregation",
            )
        )
        self.assertGreaterEqual(
            result["timing_ms"]["total"] + 1e-9,
            measured_phases,
        )
        self.assertEqual(
            result["aggregation_method"]["version"],
            "1.2.0",
        )
        self.assertIn(
            "nikoli o metriku strukturovanosti",
            result["aggregation_method"]["statement_cs"],
        )

        whitespace_aggregation = result["aggregations"][
            "whitespace_span"
        ]
        orthographic_words = result["aggregations"][
            "orthographic_word"
        ]
        self.assertEqual(len(orthographic_words["spans"]), 1)
        self.assertEqual(orthographic_words["spans"][0]["text"], "AB")
        self.assertTrue(orthographic_words["spans"][0]["complete"])
        self.assertAlmostEqual(
            orthographic_words["spans"][0]["word_surprisal_bits"],
            -math.log2(0.1) - math.log2(0.125),
        )
        self.assertAlmostEqual(
            orthographic_words["spans"][0]["pre_word_entropy_bits"],
            expected_entropies[1],
        )
        self.assertEqual(len(whitespace_aggregation["spans"]), 1)
        word = whitespace_aggregation["spans"][0]
        self.assertEqual(word["text"], "AB")
        self.assertEqual(word["character_start"], 1)
        self.assertEqual(word["character_end"], 3)
        self.assertEqual(word["token_indices"], [1, 2])
        self.assertEqual(word["token_ids"], [2, 3])
        self.assertEqual(word["token_count"], 2)
        self.assertAlmostEqual(
            word["total_surprisal_bits"],
            -math.log2(0.1) - math.log2(0.125),
        )
        self.assertAlmostEqual(
            word["surprisal_bits_per_token"],
            word["total_surprisal_bits"] / 2,
        )
        self.assertAlmostEqual(
            word["surprisal_bits_per_character"],
            word["total_surprisal_bits"] / 2,
        )
        self.assertEqual(word["character_count"], 2)
        self.assertEqual(word["non_whitespace_character_count"], 2)
        self.assertEqual(
            word["covered_non_whitespace_character_count"],
            2,
        )
        self.assertEqual(word["character_coverage_ratio"], 1.0)
        self.assertTrue(word["complete"])
        self.assertEqual(word["ambiguous_token_indices"], [])

        word_summary = whitespace_aggregation["summary"]
        self.assertEqual(word_summary["count"], 1)
        self.assertAlmostEqual(
            word_summary["mean"],
            word["surprisal_bits_per_character"],
        )
        self.assertAlmostEqual(
            word_summary["median"],
            word["surprisal_bits_per_character"],
        )
        self.assertAlmostEqual(
            word_summary["q1"],
            word["surprisal_bits_per_character"],
        )
        self.assertAlmostEqual(
            word_summary["q3"],
            word["surprisal_bits_per_character"],
        )
        self.assertEqual(word_summary["iqr"], 0.0)
        self.assertEqual(word_summary["raw_mad"], 0.0)
        self.assertEqual(word_summary["assigned_token_count"], 2)
        self.assertEqual(word_summary["ambiguous_token_count"], 0)
        self.assertEqual(word_summary["unassigned_token_count"], 1)
        self.assertAlmostEqual(word_summary["coverage_ratio"], 2 / 3)

        line = result["aggregations"]["line"]["spans"][0]
        self.assertEqual(line["text"], " AB")
        self.assertEqual(line["character_count"], 3)
        self.assertEqual(line["non_whitespace_character_count"], 2)
        self.assertEqual(line["token_indices"], [0, 1, 2])
        self.assertAlmostEqual(
            line["total_surprisal_bits"],
            expected_total,
        )
        self.assertAlmostEqual(
            line["surprisal_bits_per_character"],
            expected_total / 2,
        )
        json.dumps(result, allow_nan=False)

    def test_quantiles_and_raw_mad_use_explicit_linear_method(self) -> None:
        summary = ModelState._text_surprisal_descriptive_statistics(
            [1.0, 2.0, 4.0, 8.0]
        )
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean"], 3.75)
        self.assertEqual(summary["median"], 3.0)
        self.assertEqual(summary["q1"], 1.75)
        self.assertEqual(summary["q3"], 5.0)
        self.assertEqual(summary["iqr"], 3.25)
        self.assertEqual(summary["raw_mad"], 1.5)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 8.0)
        self.assertEqual(
            ModelState._text_surprisal_descriptive_statistics([]),
            {
                "count": 0,
                "mean": None,
                "median": None,
                "q1": None,
                "q3": None,
                "iqr": None,
                "raw_mad": None,
                "min": None,
                "max": None,
            },
        )

    @staticmethod
    def aggregation_token(
        index: int,
        token_id: int,
        start: int | None,
        end: int | None,
        surprisal_bits: float,
        *,
        lossless: bool = True,
    ) -> dict:
        return {
            "index": index,
            "token_id": token_id,
            "character_start": start,
            "character_end": end,
            "piece_is_lossless": lossless,
            "surprisal_bits": surprisal_bits,
        }

    def test_leading_space_token_maps_to_one_whitespace_span(self) -> None:
        text = " alpha beta"
        tokens = [
            self.aggregation_token(0, 10, 0, 6, 4.0),
            self.aggregation_token(1, 11, 6, 11, 6.0),
        ]
        result = ModelState._text_surprisal_aggregation(
            text,
            tokens,
            "whitespace_span",
        )
        self.assertEqual(
            [span["text"] for span in result["spans"]],
            ["alpha", "beta"],
        )
        self.assertEqual(
            [span["token_indices"] for span in result["spans"]],
            [[0], [1]],
        )
        self.assertTrue(all(span["complete"] for span in result["spans"]))
        self.assertEqual(
            [
                span["surprisal_bits_per_character"]
                for span in result["spans"]
            ],
            [4 / 5, 6 / 4],
        )
        self.assertEqual(result["summary"]["assigned_token_count"], 2)
        self.assertEqual(result["summary"]["coverage_ratio"], 1.0)

    def test_empty_lines_are_ignored_and_newlines_not_normalized(self) -> None:
        text = " A \n\n \t \nB"
        tokens = [
            self.aggregation_token(0, 10, 0, 3, 3.0),
            self.aggregation_token(1, 11, 3, 8, 5.0),
            self.aggregation_token(2, 12, 8, 9, 2.0),
            self.aggregation_token(3, 13, 9, 10, 4.0),
        ]
        result = ModelState._text_surprisal_aggregation(
            text,
            tokens,
            "line",
        )
        self.assertEqual(
            [span["text"] for span in result["spans"]],
            [" A ", "B"],
        )
        self.assertEqual(
            [
                span["non_whitespace_character_count"]
                for span in result["spans"]
            ],
            [1, 1],
        )
        self.assertEqual(
            [span["token_indices"] for span in result["spans"]],
            [[0], [3]],
        )
        self.assertEqual(result["summary"]["assigned_token_count"], 2)
        self.assertEqual(result["summary"]["unassigned_token_count"], 2)

    def test_token_crossing_span_boundary_is_ambiguous(self) -> None:
        text = "A B"
        tokens = [
            self.aggregation_token(0, 10, 0, 3, 7.0),
        ]
        whitespace = ModelState._text_surprisal_aggregation(
            text,
            tokens,
            "whitespace_span",
        )
        self.assertEqual(whitespace["summary"]["assigned_token_count"], 0)
        self.assertEqual(
            whitespace["summary"]["ambiguous_token_count"],
            1,
        )
        self.assertEqual(whitespace["summary"]["coverage_ratio"], 0.0)
        self.assertEqual(whitespace["summary"]["count"], 0)
        self.assertTrue(
            all(not span["complete"] for span in whitespace["spans"])
        )
        self.assertEqual(
            [
                span["ambiguous_token_indices"]
                for span in whitespace["spans"]
            ],
            [[0], [0]],
        )

        line = ModelState._text_surprisal_aggregation(
            text,
            tokens,
            "line",
        )
        self.assertEqual(line["summary"]["assigned_token_count"], 1)
        self.assertEqual(line["summary"]["ambiguous_token_count"], 0)
        self.assertTrue(line["spans"][0]["complete"])

    def test_sentence_like_rules_are_line_bounded_and_transparent(self) -> None:
        text = "První! Druhá？\nTřetí bez tečky"
        self.assertEqual(
            [
                text[start:end]
                for start, end in (
                    ModelState._text_surprisal_span_definitions(
                        text,
                        "sentence_like",
                    )
                )
            ],
            ["První!", "Druhá？", "Třetí bez tečky"],
        )

    def test_pair_comparison_uses_only_normalized_metrics(self) -> None:
        probability_row = np.array(
            [0.05, 0.05, 0.50, 0.25, 0.10, 0.05],
            dtype=np.float64,
        )

        def forward(_text, context_token_ids, _target_token_ids):
            return np.tile(
                np.log(probability_row),
                (len(context_token_ids), 1),
            )

        expected_metric_keys = {
            "mean_surprisal_bits",
            "bits_per_non_whitespace_character",
            (
                "median_whitespace_span_bits_per_"
                "non_whitespace_character"
            ),
            (
                "whitespace_span_iqr_bits_per_"
                "non_whitespace_character"
            ),
            (
                "whitespace_span_raw_mad_bits_per_"
                "non_whitespace_character"
            ),
        }
        with patch.object(
            self.state,
            "_text_surprisal_forward_logits",
            side_effect=forward,
        ) as patched_forward:
            result = self.state._text_surprisal_comparison_local(
                "A",
                "B",
                "První",
                "Druhý",
                3,
            )
            swapped = self.state._text_surprisal_comparison_local(
                "B",
                "A",
                "Druhý",
                "První",
                3,
            )
            identical = self.state._text_surprisal_comparison_local(
                "A",
                "A",
                "První",
                "První",
                3,
            )

        self.assertEqual(patched_forward.call_count, 6)
        self.assertEqual(
            result["analysis_kind"],
            "text_surprisal_comparison",
        )
        self.assertEqual(result["a"]["normalized_text"], "A")
        self.assertEqual(result["b"]["normalized_text"], "B")
        self.assertEqual(
            result["labels"],
            {"a": "První", "b": "Druhý"},
        )
        self.assertEqual(
            result["comparison"]["delta_definition"],
            "B - A",
        )
        metrics = result["comparison"]["metrics"]
        self.assertEqual(set(metrics), expected_metric_keys)
        self.assertNotIn("total_surprisal_bits", metrics)
        self.assertEqual(
            metrics["mean_surprisal_bits"],
            {
                "a": 1.0,
                "b": 2.0,
                "delta_b_minus_a": 1.0,
                "unit": "bits_per_token",
            },
        )
        self.assertEqual(
            metrics["bits_per_non_whitespace_character"],
            {
                "a": 1.0,
                "b": 2.0,
                "delta_b_minus_a": 1.0,
                "unit": "bits_per_non_whitespace_character",
            },
        )
        self.assertEqual(
            result["comparison"]["diagnostics"]["token_count"],
            {"a": 1, "b": 1},
        )
        self.assertFalse(
            result["comparison"]["method"]["token_alignment"]
        )
        self.assertEqual(
            {
                key: result["comparison"]["method"][key]
                for key in (
                    "id",
                    "version",
                    "status",
                    "delta_definition",
                    "same_model_and_settings",
                    "token_alignment",
                )
            },
            {
                "id": "paired_descriptive_text_surprisal",
                "version": "1.0.0",
                "status": "descriptive",
                "delta_definition": "B - A",
                "same_model_and_settings": True,
                "token_alignment": False,
            },
        )
        self.assertIn(
            "Není to klasifikace",
            result["comparison"]["method"]["statement_cs"],
        )
        self.assertEqual(result["execution"]["forward_passes"], 2)
        self.assertFalse(result["execution"]["generation"])
        self.assertFalse(result["execution"]["token_alignment"])
        self.assertFalse(result["execution"]["partial_success"])
        self.assertGreaterEqual(
            result["timing_ms"]["total"] + 1e-9,
            result["timing_ms"]["a"] + result["timing_ms"]["b"],
        )
        for metric_name in expected_metric_keys:
            forward_delta = metrics[metric_name]["delta_b_minus_a"]
            reverse_delta = swapped["comparison"]["metrics"][
                metric_name
            ]["delta_b_minus_a"]
            identical_delta = identical["comparison"]["metrics"][
                metric_name
            ]["delta_b_minus_a"]
            self.assertEqual(reverse_delta, -forward_delta)
            self.assertEqual(identical_delta, 0.0)
        json.dumps(result, allow_nan=False)

    def test_pair_metric_rejects_nonfinite_values_and_delta(self) -> None:
        for value_a, value_b in (
            (math.inf, 1.0),
            (1.0, math.nan),
            (-1e308, 1e308),
        ):
            with self.subTest(value_a=value_a, value_b=value_b):
                with self.assertRaises(RuntimeError):
                    ModelState._text_surprisal_comparison_metric(
                        value_a,
                        value_b,
                        "bits_per_token",
                    )

    def test_pair_comparison_has_no_partial_success(self) -> None:
        with patch.object(
            self.state,
            "_text_surprisal_local",
            side_effect=[{"unused": True}, ValueError("B selhalo")],
        ) as local:
            with self.assertRaisesRegex(ValueError, "B selhalo"):
                self.state._text_surprisal_comparison_local(
                    "A",
                    "B",
                    "A",
                    "B",
                    3,
                )
        self.assertEqual(local.call_count, 2)

    def test_pair_token_limit_applies_independently_to_both_texts(
        self,
    ) -> None:
        oversized = "A" * (MAX_TEXT_SURPRISAL_TOKENS + 1)

        def zero_logits(
            _text: str,
            context_token_ids: list[int],
            _target_token_ids: list[int],
        ) -> np.ndarray:
            return np.zeros((len(context_token_ids), 6), dtype=np.float64)

        with patch.object(
            self.state,
            "_text_surprisal_forward_logits",
            side_effect=zero_logits,
        ) as forward:
            with self.assertRaisesRegex(ValueError, "limit"):
                self.state._text_surprisal_comparison_local(
                    oversized,
                    "A",
                    "A",
                    "B",
                    3,
                )
            forward.assert_not_called()

            with self.assertRaisesRegex(ValueError, "limit"):
                self.state._text_surprisal_comparison_local(
                    "A",
                    oversized,
                    "A",
                    "B",
                    3,
                )
            self.assertEqual(forward.call_count, 1)

    def test_pair_identity_failure_happens_before_either_forward(
        self,
    ) -> None:
        with patch.object(
            self.state,
            "_text_surprisal_forward_logits",
        ) as forward:
            for model_id, generation, server_id in (
                ("4bit", 7, self.state.server_instance_id),
                ("bf16", 8, self.state.server_instance_id),
                ("bf16", 7, "another-server"),
            ):
                with self.subTest(
                    model_id=model_id,
                    generation=generation,
                    server_id=server_id,
                ):
                    with self.assertRaises(ModelChangedError):
                        self.state.text_surprisal_comparison(
                            "A",
                            "B",
                            "A",
                            "B",
                            3,
                            model_id,
                            generation,
                            server_id,
                        )
            forward.assert_not_called()

    def test_analysis_does_not_mutate_any_global_model_cache(self) -> None:
        cached_logits = np.array([1.0, 2.0], dtype=np.float32)
        active_cache = object()
        self.state.logits_cache = OrderedDict(
            [((1, 2), cached_logits.copy())]
        )
        self.state.lens_cache = OrderedDict(
            [((7, "lens", 3), {"kept": True})]
        )
        self.state.tuned_lens_cache = OrderedDict(
            [((7, "a", "b", "c", "d", 3), {"kept": True})]
        )
        self.state.model_provenance_cache = {
            "kept": {"metadata_fingerprint": "unchanged"}
        }
        self.state.model_content_fingerprint_cache = {"kept": "unchanged"}
        self.state.active_ids = [1, 2]
        self.state.active_cache = active_cache
        self.state.active_logits = cached_logits.copy()

        self.run_analysis()

        self.assertEqual(list(self.state.logits_cache), [(1, 2)])
        np.testing.assert_array_equal(
            self.state.logits_cache[(1, 2)],
            cached_logits,
        )
        self.assertEqual(
            self.state.lens_cache,
            OrderedDict([((7, "lens", 3), {"kept": True})]),
        )
        self.assertEqual(
            self.state.tuned_lens_cache,
            OrderedDict(
                [((7, "a", "b", "c", "d", 3), {"kept": True})]
            ),
        )
        self.assertEqual(
            self.state.model_provenance_cache,
            {"kept": {"metadata_fingerprint": "unchanged"}},
        )
        self.assertEqual(
            self.state.model_content_fingerprint_cache,
            {"kept": "unchanged"},
        )
        self.assertEqual(self.state.active_ids, [1, 2])
        self.assertIs(self.state.active_cache, active_cache)
        np.testing.assert_array_equal(self.state.active_logits, cached_logits)

    def test_identity_and_transition_checks_happen_before_forward(self) -> None:
        with patch.object(
            self.state,
            "_text_surprisal_forward_logits",
        ) as forward:
            for model_id, generation, server_id in (
                ("4bit", 7, self.state.server_instance_id),
                ("bf16", 8, self.state.server_instance_id),
                ("bf16", 7, "another-server"),
            ):
                with self.subTest(
                    model_id=model_id,
                    generation=generation,
                    server_id=server_id,
                ):
                    with self.assertRaises(ModelChangedError):
                        self.state.text_surprisal(
                            "AB",
                            3,
                            model_id,
                            generation,
                            server_id,
                        )
            forward.assert_not_called()

            self.state.status = "switching"
            with self.assertRaisesRegex(RuntimeError, "připraven"):
                self.state.text_surprisal(
                    "AB",
                    3,
                    "bf16",
                    7,
                    self.state.server_instance_id,
                )
            forward.assert_not_called()

    def test_byte_fragment_tokens_use_byte_safe_display(self) -> None:
        self.state.tokenizer = ByteFragmentTokenizer()
        logits = np.zeros((4, 5), dtype=np.float64)
        with patch.object(
            self.state,
            "_text_surprisal_forward_logits",
            return_value=logits,
        ):
            result = self.state.text_surprisal(
                "😀",
                2,
                "bf16",
                7,
                self.state.server_instance_id,
            )

        self.assertEqual(result["normalized_text"], "😀")
        self.assertEqual(result["token_count"], 4)
        self.assertTrue(
            all(
                token["piece_is_lossless"] is False
                for token in result["tokens"]
            )
        )
        self.assertEqual(
            [token["display"] for token in result["tokens"]],
            ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"],
        )
        self.assertNotIn(
            "\ufffd",
            "".join(token["display"] for token in result["tokens"]),
        )
        for unit in (
            "orthographic_word",
            "punctuation",
            "whitespace_span",
            "line",
            "sentence_like",
        ):
            aggregation = result["aggregations"][unit]
            self.assertEqual(
                aggregation["summary"]["ambiguous_token_count"],
                4,
            )
            self.assertEqual(
                aggregation["summary"]["assigned_token_count"],
                0,
            )
            self.assertEqual(
                aggregation["summary"]["coverage_ratio"],
                0.0,
            )
            self.assertEqual(aggregation["summary"]["count"], 0)
            if unit in {"orthographic_word", "punctuation"}:
                self.assertEqual(aggregation["spans"], [])
            else:
                self.assertFalse(aggregation["spans"][0]["complete"])
        json.dumps(result, allow_nan=False)


class TextSurprisalWorkerTest(unittest.TestCase):
    def test_running_app_uses_one_serial_worker_job_and_one_forward(self) -> None:
        state = ModelState(model_path=Path("."), mock=True)
        state.start()
        deadline = time.monotonic() + 2.0
        while state.status != "ready" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(state.status, "ready")

        calls: list[str] = []
        original_forward = state._text_surprisal_forward_logits

        def observed_forward(*args):
            calls.append(threading.current_thread().name)
            return original_forward(*args)

        try:
            with patch.object(
                state,
                "_text_surprisal_forward_logits",
                side_effect=observed_forward,
            ) as forward:
                result = state.text_surprisal(
                    "Jedna hotová věta.",
                    4,
                    "mock",
                    state.model_generation,
                    state.server_instance_id,
                )
            self.assertEqual(forward.call_count, 1)
            self.assertEqual(calls, ["apertus-model-worker"])
            self.assertEqual(
                result["execution"]["worker"],
                "single_serial_worker_job",
            )
            self.assertEqual(result["execution"]["forward_passes"], 1)
        finally:
            state.stop()
            if state.worker_thread is not None:
                state.worker_thread.join(timeout=2)

    def test_pair_uses_one_worker_job_two_forwards_and_one_identity_check(
        self,
    ) -> None:
        state = ModelState(model_path=Path("."), mock=True)
        state.start()
        deadline = time.monotonic() + 2.0
        while state.status != "ready" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(state.status, "ready")

        cached_logits = np.array([1.0, 2.0], dtype=np.float32)
        active_cache = object()
        state.logits_cache = OrderedDict(
            [((1, 2), cached_logits.copy())]
        )
        state.active_ids = [1, 2]
        state.active_cache = active_cache
        state.active_logits = cached_logits.copy()
        calls: list[str] = []
        original_forward = state._text_surprisal_forward_logits
        original_identity = state._validate_request_identity_unlocked

        def observed_forward(*args):
            calls.append(threading.current_thread().name)
            return original_forward(*args)

        try:
            with (
                patch.object(
                    state.work_queue,
                    "put_nowait",
                    wraps=state.work_queue.put_nowait,
                ) as dispatch,
                patch.object(
                    state,
                    "_text_surprisal_forward_logits",
                    side_effect=observed_forward,
                ) as forward,
                patch.object(
                    state,
                    "_validate_request_identity_unlocked",
                    wraps=original_identity,
                ) as identity,
            ):
                result = state.text_surprisal_comparison(
                    "První hotový text.",
                    "Druhý hotový text.",
                    "A",
                    "B",
                    4,
                    "mock",
                    state.model_generation,
                    state.server_instance_id,
                )

            self.assertEqual(dispatch.call_count, 1)
            submitted_job = dispatch.call_args.args[0]
            self.assertEqual(
                submitted_job[0],
                "text_surprisal_comparison",
            )
            self.assertEqual(identity.call_count, 1)
            self.assertEqual(forward.call_count, 2)
            self.assertEqual(
                calls,
                ["apertus-model-worker", "apertus-model-worker"],
            )
            self.assertEqual(
                result["execution"]["worker"],
                "single_serial_worker_job",
            )
            self.assertEqual(result["execution"]["forward_passes"], 2)
            self.assertEqual(list(state.logits_cache), [(1, 2)])
            np.testing.assert_array_equal(
                state.logits_cache[(1, 2)],
                cached_logits,
            )
            self.assertEqual(state.active_ids, [1, 2])
            self.assertIs(state.active_cache, active_cache)
            np.testing.assert_array_equal(
                state.active_logits,
                cached_logits,
            )
        finally:
            state.stop()
            if state.worker_thread is not None:
                state.worker_thread.join(timeout=2)


class TextSurprisalValidationAndHttpContractTest(unittest.TestCase):
    def test_request_fields_are_required_and_lossless(self) -> None:
        request = {
            "text": "  AB  ",
            "top_k": 5,
            "model_id": "bf16",
            "model_generation": 7,
            "server_instance_id": "server",
        }
        self.assertEqual(
            GardenHandler._text_surprisal_request_fields(request),
            ("  AB  ", 5, "bf16", 7, "server"),
        )
        for key in (
            "text",
            "top_k",
            "model_id",
            "model_generation",
            "server_instance_id",
        ):
            invalid = dict(request)
            invalid.pop(key)
            with self.subTest(missing=key):
                with self.assertRaises(ValueError):
                    GardenHandler._text_surprisal_request_fields(invalid)

    def test_conservative_limits_and_scalar_types_are_enforced(self) -> None:
        state = ModelState(model_path=Path("."), mock=False)
        state.status = "ready"
        state.model_generation = 1
        state.tokenizer = CharacterTokenizer()
        for text, top_k in (
            ("   ", 3),
            ("AB", True),
            ("AB", 0),
            ("AB", 13),
            ("A" * (MAX_TEXT_SURPRISAL_CHARACTERS + 1), 3),
            ("A" * (MAX_TEXT_SURPRISAL_TOKENS + 1), 3),
        ):
            with self.subTest(length=len(text), top_k=top_k):
                with self.assertRaises(ValueError):
                    state.text_surprisal(
                        text,
                        top_k,
                        "bf16",
                        1,
                        state.server_instance_id,
                    )

    def test_pair_request_fields_validate_both_texts_and_labels(self) -> None:
        request = {
            "text_a": "  A  ",
            "text_b": " B ",
            "label_a": " První ",
            "label_b": "",
            "top_k": 5,
            "model_id": "bf16",
            "model_generation": 7,
            "server_instance_id": "server",
        }
        self.assertEqual(
            GardenHandler._text_surprisal_comparison_request_fields(
                request
            ),
            (
                "  A  ",
                " B ",
                "První",
                "Text B",
                5,
                "bf16",
                7,
                "server",
            ),
        )
        without_labels = dict(request)
        without_labels.pop("label_a")
        without_labels.pop("label_b")
        self.assertEqual(
            GardenHandler._text_surprisal_comparison_request_fields(
                without_labels
            )[2:4],
            ("Text A", "Text B"),
        )
        for key in (
            "text_a",
            "text_b",
            "top_k",
            "model_id",
            "model_generation",
            "server_instance_id",
        ):
            invalid = dict(request)
            invalid.pop(key)
            with self.subTest(missing=key):
                with self.assertRaises(ValueError):
                    GardenHandler._text_surprisal_comparison_request_fields(
                        invalid
                    )
        for field_name in ("text_a", "text_b"):
            invalid = dict(request)
            invalid[field_name] = "A" * (
                MAX_TEXT_SURPRISAL_CHARACTERS + 1
            )
            with self.subTest(too_long=field_name):
                with self.assertRaises(ValueError):
                    GardenHandler._text_surprisal_comparison_request_fields(
                        invalid
                    )
        invalid_label = dict(request)
        invalid_label["label_a"] = 42
        with self.assertRaises(ValueError):
            GardenHandler._text_surprisal_comparison_request_fields(
                invalid_label
            )
        oversized_label = dict(request)
        oversized_label["label_b"] = "L" * (
            MAX_TEXT_SURPRISAL_LABEL_CHARACTERS + 1
        )
        with self.assertRaises(ValueError):
            GardenHandler._text_surprisal_comparison_request_fields(
                oversized_label
            )

    def test_route_dispatch_is_present(self) -> None:
        import inspect

        text_route = POST_ROUTES["/api/text-surprisal"]
        comparison_route = POST_ROUTES[
            "/api/text-surprisal-comparison"
        ]
        text_source = inspect.getsource(
            getattr(GardenHandler, text_route.controller)
        )
        comparison_source = inspect.getsource(
            getattr(GardenHandler, comparison_route.controller)
        )
        self.assertIn("self.state.text_surprisal(", text_source)
        self.assertIn(
            "self.state.text_surprisal_comparison(",
            comparison_source,
        )
        get_source = inspect.getsource(GardenHandler.do_GET)
        self.assertIn('"/prekvapeni"', get_source)
        self.assertIn("SURPRISE_FILE", get_source)


if __name__ == "__main__":
    unittest.main()
