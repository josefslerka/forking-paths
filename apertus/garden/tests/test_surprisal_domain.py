from __future__ import annotations

from apertus.garden.domains.surprisal import (
    aggregation,
    comparison_metric,
    descriptive_statistics,
    span_definitions,
)
from apertus.garden.server import ModelState


def test_model_state_surprisal_facade_matches_domain() -> None:
    values = [1.0, 2.0, 4.0, 8.0]
    assert ModelState._text_surprisal_descriptive_statistics(
        values
    ) == descriptive_statistics(values)
    assert ModelState._text_surprisal_span_definitions(
        "A. B!",
        "sentence_like",
    ) == span_definitions("A. B!", "sentence_like")
    assert ModelState._text_surprisal_comparison_metric(
        1.0,
        3.0,
        "bits",
    ) == comparison_metric(1.0, 3.0, "bits")


def test_surprisal_aggregation_keeps_coverage_contract() -> None:
    tokens = [
        {
            "index": 0,
            "token_id": 7,
            "surprisal_bits": 2.0,
            "piece_is_lossless": True,
            "character_start": 0,
            "character_end": 1,
        },
        {
            "index": 1,
            "token_id": 8,
            "surprisal_bits": 1.0,
            "piece_is_lossless": True,
            "character_start": 2,
            "character_end": 3,
        },
    ]
    result = aggregation("A B", tokens, "whitespace_span")

    assert result["summary"]["coverage_ratio"] == 1.0
    assert result["summary"]["complete_span_count"] == 2
    assert [
        span["surprisal_bits_per_character"]
        for span in result["spans"]
    ] == [2.0, 1.0]


def test_orthographic_words_sum_subwords_and_keep_boundary_entropy() -> None:
    tokens = [
        {
            "index": 0,
            "token_id": 10,
            "surprisal_bits": 2.0,
            "entropy_bits": 7.5,
            "piece_is_lossless": True,
            "character_start": 0,
            "character_end": 2,
        },
        {
            "index": 1,
            "token_id": 11,
            "surprisal_bits": 3.0,
            "entropy_bits": 4.0,
            "piece_is_lossless": True,
            "character_start": 2,
            "character_end": 5,
        },
        {
            "index": 2,
            "token_id": 12,
            "surprisal_bits": 4.0,
            "entropy_bits": 6.25,
            "piece_is_lossless": True,
            "character_start": 5,
            "character_end": 11,
        },
    ]

    result = aggregation("zámek stojí", tokens, "orthographic_word")

    assert [span["text"] for span in result["spans"]] == [
        "zámek",
        "stojí",
    ]
    assert result["spans"][0]["token_indices"] == [0, 1]
    assert result["spans"][0]["word_surprisal_bits"] == 5.0
    assert result["spans"][0]["pre_word_entropy_bits"] == 7.5
    assert (
        result["spans"][0]["pre_word_next_token_entropy_bits"]
        == 7.5
    )
    assert result["spans"][1]["token_indices"] == [2]
    assert result["spans"][1]["word_surprisal_bits"] == 4.0
    assert result["spans"][1]["pre_word_entropy_bits"] == 6.25
    assert result["summary"]["mean_word_surprisal_bits"] == 4.5
    assert result["summary"]["mean_pre_word_entropy_bits"] == 6.875
    assert (
        result["summary"]["mean_pre_word_next_token_entropy_bits"]
        == 6.875
    )


def test_orthographic_word_omits_metrics_for_boundary_contamination() -> None:
    result = aggregation(
        ".Slovo",
        [
            {
                "index": 0,
                "token_id": 20,
                "surprisal_bits": 3.0,
                "entropy_bits": 5.0,
                "piece_is_lossless": True,
                "character_start": 0,
                "character_end": 6,
            }
        ],
        "orthographic_word",
    )

    word = result["spans"][0]
    assert word["text"] == "Slovo"
    assert word["complete"] is False
    assert word["word_surprisal_bits"] is None
    assert word["pre_word_entropy_bits"] is None
    assert word["boundary_contaminated_token_indices"] == [0]
    assert result["summary"]["complete_span_count"] == 0


def test_punctuation_is_a_separate_scored_unit() -> None:
    result = aggregation(
        "Slovo?!",
        [
            {
                "index": 0,
                "token_id": 20,
                "surprisal_bits": 3.0,
                "entropy_bits": 5.0,
                "piece_is_lossless": True,
                "character_start": 0,
                "character_end": 5,
            },
            {
                "index": 1,
                "token_id": 21,
                "surprisal_bits": 1.25,
                "entropy_bits": 2.5,
                "piece_is_lossless": True,
                "character_start": 5,
                "character_end": 7,
            },
        ],
        "punctuation",
    )

    assert len(result["spans"]) == 1
    punctuation = result["spans"][0]
    assert punctuation["text"] == "?!"
    assert punctuation["span_kind"] == "punctuation"
    assert punctuation["complete"] is True
    assert punctuation["punctuation_surprisal_bits"] == 1.25
    assert punctuation["pre_unit_next_token_entropy_bits"] == 2.5
    assert punctuation["pre_punctuation_next_token_entropy_bits"] == 2.5
    assert result["summary"]["mean_punctuation_surprisal_bits"] == 1.25


def test_internal_word_joiner_is_not_a_punctuation_unit() -> None:
    result = aggregation("e-mail — ano.", [], "punctuation")

    assert [span["text"] for span in result["spans"]] == ["—", "."]
