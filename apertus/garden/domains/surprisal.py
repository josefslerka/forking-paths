"""Pure descriptive analytics over already-scored text tokens."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

from .tokenization import TokenizationService


def punctuation_ranges(text: str) -> list[tuple[int, int]]:
    """Return punctuation clusters that are not internal to a word span."""
    word_positions = {
        position
        for start, end in TokenizationService.orthographic_word_ranges(text)
        for position in range(start, end)
    }
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        is_punctuation = (
            cursor not in word_positions
            and unicodedata.category(text[cursor]).startswith("P")
        )
        if not is_punctuation:
            cursor += 1
            continue
        start = cursor
        cursor += 1
        while (
            cursor < len(text)
            and cursor not in word_positions
            and unicodedata.category(text[cursor]).startswith("P")
        ):
            cursor += 1
        ranges.append((start, cursor))
    return ranges


def span_definitions(
    text: str,
    unit: str,
) -> list[tuple[int, int]]:
    """Return transparent observable spans over Python characters."""
    if unit == "orthographic_word":
        return TokenizationService.orthographic_word_ranges(text)

    if unit == "punctuation":
        return punctuation_ranges(text)

    if unit == "whitespace_span":
        return [
            (match.start(), match.end())
            for match in re.finditer(r"\S+", text)
        ]

    line_spans: list[tuple[int, int]] = []
    line_break_characters = frozenset(
        "\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"
    )
    cursor = 0
    for line_with_ending in text.splitlines(keepends=True):
        content_end = cursor + len(line_with_ending)
        while (
            content_end > cursor
            and text[content_end - 1] in line_break_characters
        ):
            content_end -= 1
        if text[cursor:content_end].strip():
            line_spans.append((cursor, content_end))
        cursor += len(line_with_ending)
    if cursor < len(text) and text[cursor:].strip():
        line_spans.append((cursor, len(text)))
    if unit == "line":
        return line_spans

    if unit != "sentence_like":
        raise ValueError(f"Neznámá agregační jednotka: {unit}.")

    terminal_characters = frozenset(".!?…。｡！？‼⁇⁈⁉؟۔։।॥")
    trailing_closers = frozenset("\"'”’»›)]}）］｝")
    sentence_spans: list[tuple[int, int]] = []
    for line_start, line_end in line_spans:
        candidate_start = line_start
        position = line_start
        while position < line_end:
            if text[position] not in terminal_characters:
                position += 1
                continue
            candidate_end = position + 1
            while (
                candidate_end < line_end
                and text[candidate_end] in terminal_characters
            ):
                candidate_end += 1
            while (
                candidate_end < line_end
                and text[candidate_end] in trailing_closers
            ):
                candidate_end += 1
            start = candidate_start
            end = candidate_end
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end and text[start:end].strip():
                sentence_spans.append((start, end))
            candidate_start = candidate_end
            position = candidate_end

        start = candidate_start
        end = line_end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end and text[start:end].strip():
            sentence_spans.append((start, end))
    return sentence_spans


def linear_quantile(
    values: list[float],
    quantile: float,
) -> float | None:
    """Use explicit (n - 1) * q linear interpolation."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def descriptive_statistics(
    values: list[float],
) -> dict[str, float | int | None]:
    """Summarize complete span values without robustness rescaling."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "raw_mad": None,
            "min": None,
            "max": None,
        }
    checked = [float(value) for value in values]
    if not all(math.isfinite(value) for value in checked):
        raise RuntimeError(
            "Agregace překvapení obsahuje nekonečnou hodnotu."
        )
    median = linear_quantile(checked, 0.5)
    q1 = linear_quantile(checked, 0.25)
    q3 = linear_quantile(checked, 0.75)
    assert median is not None
    assert q1 is not None
    assert q3 is not None
    raw_mad = linear_quantile(
        [abs(value - median) for value in checked],
        0.5,
    )
    return {
        "count": len(checked),
        "mean": math.fsum(checked) / len(checked),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "raw_mad": raw_mad,
        "min": min(checked),
        "max": max(checked),
    }


def aggregation(
    text: str,
    tokens: list[dict[str, Any]],
    unit: str,
) -> dict[str, Any]:
    """Aggregate already-computed token surprise into heuristic spans."""
    definitions = span_definitions(text, unit)
    token_indices_by_span: list[list[int]] = [[] for _ in definitions]
    ambiguous_indices_by_span: list[list[int]] = [[] for _ in definitions]
    assigned_token_count = 0
    ambiguous_token_count = 0
    unassigned_token_count = 0

    for token in tokens:
        token_index = int(token["index"])
        start = token.get("character_start")
        end = token.get("character_end")
        reliable_offset = (
            token.get("piece_is_lossless") is True
            and type(start) is int
            and type(end) is int
            and 0 <= start < end <= len(text)
        )
        if not reliable_offset:
            ambiguous_token_count += 1
            continue
        overlapping_spans = [
            span_index
            for span_index, (span_start, span_end) in enumerate(definitions)
            if max(start, span_start) < min(end, span_end)
        ]
        if len(overlapping_spans) == 1:
            token_indices_by_span[overlapping_spans[0]].append(token_index)
            assigned_token_count += 1
        elif len(overlapping_spans) > 1:
            ambiguous_token_count += 1
            for span_index in overlapping_spans:
                ambiguous_indices_by_span[span_index].append(token_index)
        else:
            unassigned_token_count += 1

    token_by_index = {int(token["index"]): token for token in tokens}
    spans: list[dict[str, Any]] = []
    complete_values: list[float] = []
    complete_total_values: list[float] = []
    complete_pre_entropy_values: list[float] = []
    for span_index, (span_start, span_end) in enumerate(definitions):
        token_indices = token_indices_by_span[span_index]
        span_tokens = [
            token_by_index[token_index] for token_index in token_indices
        ]
        span_text = text[span_start:span_end]
        character_count = len(span_text)
        non_whitespace_positions = {
            position
            for position in range(span_start, span_end)
            if not text[position].isspace()
        }
        covered_non_whitespace_positions: set[int] = set()
        boundary_contaminated_token_indices: list[int] = []
        for token in span_tokens:
            token_start = int(token["character_start"])
            token_end = int(token["character_end"])
            covered_non_whitespace_positions.update(
                position
                for position in range(
                    max(span_start, token_start),
                    min(span_end, token_end),
                )
                if not text[position].isspace()
            )
            if unit in {"orthographic_word", "punctuation"} and any(
                not text[position].isspace()
                for position in range(token_start, token_end)
                if not span_start <= position < span_end
            ):
                boundary_contaminated_token_indices.append(
                    int(token["index"])
                )
        non_whitespace_character_count = len(non_whitespace_positions)
        covered_count = len(covered_non_whitespace_positions)
        coverage = (
            covered_count / non_whitespace_character_count
            if non_whitespace_character_count
            else 0.0
        )
        complete = (
            bool(span_tokens)
            and not ambiguous_indices_by_span[span_index]
            and not boundary_contaminated_token_indices
            and coverage == 1.0
        )
        pre_span_entropy_bits = None
        if span_tokens:
            candidate_entropy = span_tokens[0].get("entropy_bits")
            if candidate_entropy is not None:
                candidate_entropy = float(candidate_entropy)
                if math.isfinite(candidate_entropy):
                    pre_span_entropy_bits = candidate_entropy
        if span_tokens:
            total_bits = math.fsum(
                float(token["surprisal_bits"]) for token in span_tokens
            )
            bits_per_token = total_bits / len(span_tokens)
            bits_per_character = total_bits / non_whitespace_character_count
        else:
            total_bits = None
            bits_per_token = None
            bits_per_character = None
        if complete and bits_per_character is not None:
            complete_values.append(bits_per_character)
        if complete and total_bits is not None:
            complete_total_values.append(total_bits)
        if complete and pre_span_entropy_bits is not None:
            complete_pre_entropy_values.append(pre_span_entropy_bits)
        spans.append(
            {
                "index": span_index,
                "span_kind": (
                    "word"
                    if unit == "orthographic_word"
                    else "punctuation"
                    if unit == "punctuation"
                    else "span"
                ),
                "text": span_text,
                "character_start": span_start,
                "character_end": span_end,
                "token_indices": token_indices,
                "token_ids": [
                    int(token["token_id"]) for token in span_tokens
                ],
                "token_count": len(span_tokens),
                "total_surprisal_bits": total_bits,
                "surprisal_bits_per_token": bits_per_token,
                "surprisal_bits_per_character": bits_per_character,
                "character_count": character_count,
                "non_whitespace_character_count": (
                    non_whitespace_character_count
                ),
                "covered_non_whitespace_character_count": covered_count,
                "character_coverage_ratio": coverage,
                "complete": complete,
                "first_token_index": (
                    token_indices[0] if token_indices else None
                ),
                "last_token_index": (
                    token_indices[-1] if token_indices else None
                ),
                "pre_span_entropy_bits": pre_span_entropy_bits,
                "pre_unit_next_token_entropy_bits": (
                    pre_span_entropy_bits if complete else None
                ),
                "word_surprisal_bits": (
                    total_bits
                    if unit == "orthographic_word" and complete
                    else None
                ),
                "pre_word_next_token_entropy_bits": (
                    pre_span_entropy_bits
                    if unit == "orthographic_word" and complete
                    else None
                ),
                "pre_word_entropy_bits": (
                    pre_span_entropy_bits
                    if unit == "orthographic_word" and complete
                    else None
                ),
                "punctuation_surprisal_bits": (
                    total_bits
                    if unit == "punctuation" and complete
                    else None
                ),
                "pre_punctuation_next_token_entropy_bits": (
                    pre_span_entropy_bits
                    if unit == "punctuation" and complete
                    else None
                ),
                "boundary_contaminated_token_count": len(
                    boundary_contaminated_token_indices
                ),
                "boundary_contaminated_token_indices": (
                    boundary_contaminated_token_indices
                ),
                "ambiguous_token_count": len(
                    ambiguous_indices_by_span[span_index]
                ),
                "ambiguous_token_indices": (
                    ambiguous_indices_by_span[span_index]
                ),
            }
        )

    summary = descriptive_statistics(complete_values)
    total_summary = descriptive_statistics(complete_total_values)
    pre_entropy_summary = descriptive_statistics(
        complete_pre_entropy_values
    )
    summary.update(
        {
            "assigned_token_count": assigned_token_count,
            "ambiguous_token_count": ambiguous_token_count,
            "unassigned_token_count": unassigned_token_count,
            "total_token_count": len(tokens),
            "coverage_ratio": (
                assigned_token_count / len(tokens) if tokens else 0.0
            ),
            "span_count": len(spans),
            "complete_span_count": sum(
                bool(span["complete"]) for span in spans
            ),
            "incomplete_span_count": sum(
                not bool(span["complete"]) for span in spans
            ),
            "boundary_contaminated_token_count": sum(
                int(span["boundary_contaminated_token_count"])
                for span in spans
            ),
        }
    )
    if unit == "orthographic_word":
        summary.update(
            {
                "word_surprisal_bits": total_summary,
                "pre_word_next_token_entropy_bits": pre_entropy_summary,
                "pre_word_entropy_bits": pre_entropy_summary,
                "mean_word_surprisal_bits": total_summary["mean"],
                "mean_pre_word_next_token_entropy_bits": (
                    pre_entropy_summary["mean"]
                ),
                "median_word_surprisal_bits": total_summary["median"],
                "mean_pre_word_entropy_bits": pre_entropy_summary["mean"],
            }
        )
    elif unit == "punctuation":
        summary.update(
            {
                "punctuation_surprisal_bits": total_summary,
                "pre_punctuation_next_token_entropy_bits": (
                    pre_entropy_summary
                ),
                "mean_punctuation_surprisal_bits": total_summary["mean"],
                "mean_pre_punctuation_next_token_entropy_bits": (
                    pre_entropy_summary["mean"]
                ),
            }
        )
    return {
        "unit": unit,
        "label_cs": {
            "orthographic_word": "ortografické slovo",
            "punctuation": "interpunkční jednotka",
            "whitespace_span": "úsek oddělený mezerami",
            "line": "neprázdný řádek",
            "sentence_like": "větně podobný úsek",
        }[unit],
        "spans": spans,
        "summary": summary,
    }


def aggregations(
    text: str,
    tokens: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        unit: aggregation(text, tokens, unit)
        for unit in (
            "orthographic_word",
            "punctuation",
            "whitespace_span",
            "line",
            "sentence_like",
        )
    }


def comparison_metric(
    value_a: float | int | None,
    value_b: float | int | None,
    unit: str,
) -> dict[str, float | str | None]:
    checked_a = float(value_a) if value_a is not None else None
    checked_b = float(value_b) if value_b is not None else None
    for value in (checked_a, checked_b):
        if value is not None and not math.isfinite(value):
            raise RuntimeError(
                "Párové srovnání překvapení obsahuje nekonečnou hodnotu."
            )
    delta = (
        checked_b - checked_a
        if checked_a is not None and checked_b is not None
        else None
    )
    if delta is not None and not math.isfinite(delta):
        raise RuntimeError(
            "Rozdíl párového srovnání překvapení překročil konečný "
            "číselný rozsah."
        )
    return {
        "a": checked_a,
        "b": checked_b,
        "delta_b_minus_a": delta,
        "unit": unit,
    }


def comparison_metrics(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
) -> dict[str, dict[str, float | str | None]]:
    def bits_per_non_whitespace_character(
        result: dict[str, Any],
    ) -> float | None:
        character_count = sum(
            not character.isspace()
            for character in str(result["normalized_text"])
        )
        if character_count == 0:
            return None
        return (
            float(result["summary"]["total_surprisal_bits"])
            / character_count
        )

    summary_a = result_a["summary"]
    summary_b = result_b["summary"]
    spans_a = result_a["aggregations"]["whitespace_span"]["summary"]
    spans_b = result_b["aggregations"]["whitespace_span"]["summary"]
    per_token_unit = "bits_per_token"
    per_character_unit = "bits_per_non_whitespace_character"
    return {
        "mean_surprisal_bits": comparison_metric(
            summary_a["mean_surprisal_bits"],
            summary_b["mean_surprisal_bits"],
            per_token_unit,
        ),
        "bits_per_non_whitespace_character": comparison_metric(
            bits_per_non_whitespace_character(result_a),
            bits_per_non_whitespace_character(result_b),
            per_character_unit,
        ),
        "median_whitespace_span_bits_per_non_whitespace_character": (
            comparison_metric(
                spans_a["median"],
                spans_b["median"],
                per_character_unit,
            )
        ),
        "whitespace_span_iqr_bits_per_non_whitespace_character": (
            comparison_metric(
                spans_a["iqr"],
                spans_b["iqr"],
                per_character_unit,
            )
        ),
        "whitespace_span_raw_mad_bits_per_non_whitespace_character": (
            comparison_metric(
                spans_a["raw_mad"],
                spans_b["raw_mad"],
                per_character_unit,
            )
        ),
    }


__all__ = [
    "aggregation",
    "aggregations",
    "comparison_metric",
    "comparison_metrics",
    "descriptive_statistics",
    "linear_quantile",
    "span_definitions",
]
