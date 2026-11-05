"""Pure probability-distribution mathematics for the workbench.

The functions in this module have no model, cache, HTTP, or persistence
dependencies.  They intentionally keep NumPy arrays as their public boundary
because callers already hold full-vocabulary logits and probabilities.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


__all__ = [
    "distribution_distance",
    "entropy_bits_from_probabilities",
    "probabilities_at_temperature",
    "probability_views",
    "raw_distribution_metrics",
    "summarize_probabilities",
]


def probabilities_at_temperature(
    logits: np.ndarray, temperature: float
) -> np.ndarray:
    if not np.all(np.isfinite(logits)):
        raise ValueError("Logity obsahují neplatné hodnoty.")
    scaled = logits.astype(np.float64, copy=False) / temperature
    scaled -= float(np.max(scaled))
    probabilities = np.exp(scaled)
    probabilities /= float(np.sum(probabilities))
    return probabilities


def entropy_bits_from_probabilities(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    return max(
        0.0,
        float(-np.sum(positive * np.log2(positive))),
    )


def distribution_distance(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[float, float]:
    """Return Jensen-Shannon divergence in bits and total variation."""
    if left.shape != right.shape:
        raise ValueError("Porovnávané distribuce mají rozdílný tvar.")
    midpoint = 0.5 * (left + right)

    def kl_to_midpoint(probabilities: np.ndarray) -> float:
        positive = probabilities > 0
        return float(
            np.sum(
                probabilities[positive]
                * np.log2(probabilities[positive] / midpoint[positive])
            )
        )

    js_divergence = min(
        1.0,
        max(
            0.0,
            0.5 * (kl_to_midpoint(left) + kl_to_midpoint(right)),
        ),
    )
    total_variation = min(
        1.0,
        max(0.0, float(0.5 * np.sum(np.abs(right - left)))),
    )
    return js_divergence, total_variation


def summarize_probabilities(
    probabilities: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    count = min(top_k, probabilities.size)
    cutoff_index = probabilities.size - count
    cutoff_probability = float(
        np.partition(probabilities, cutoff_index)[cutoff_index]
    )
    higher_ids = np.flatnonzero(probabilities > cutoff_probability)
    tied_ids = np.flatnonzero(probabilities == cutoff_probability)
    remaining = count - higher_ids.size
    indices = np.concatenate((higher_ids, tied_ids[:remaining]))
    indices = indices[
        np.lexsort((indices, -probabilities[indices]))
    ]
    values = probabilities[indices]
    other = max(0.0, 1.0 - float(np.sum(values)))
    positive = probabilities[probabilities > 0]
    entropy_bits = max(0.0, float(-np.sum(positive * np.log2(positive))))
    effective_tokens = float(2**entropy_bits)
    normalized_entropy = (
        entropy_bits / math.log2(probabilities.size)
        if probabilities.size > 1
        else 0.0
    )
    return (
        indices,
        values,
        other,
        entropy_bits,
        effective_tokens,
        normalized_entropy,
    )


def probability_views(
    logits: np.ndarray, temperature: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return adjusted and raw T=1 probabilities without duplicate softmax work."""
    raw_probabilities = probabilities_at_temperature(logits, 1.0)
    adjusted_probabilities = (
        raw_probabilities
        if math.isclose(temperature, 1.0)
        else probabilities_at_temperature(logits, temperature)
    )
    return adjusted_probabilities, raw_probabilities


def raw_distribution_metrics(
    logits: np.ndarray, raw_probabilities: np.ndarray
) -> dict[str, Any]:
    """Summarize the full-vocabulary model distribution at temperature 1."""
    vocab_size = raw_probabilities.size
    if vocab_size == 0:
        raise ValueError("Distribuce modelu je prázdná.")

    positive = raw_probabilities[raw_probabilities > 0]
    entropy_bits = max(
        0.0,
        float(-np.sum(positive * np.log2(positive))),
    )

    highest_logit = float(np.max(logits))
    highest_ids = np.flatnonzero(logits == highest_logit)
    top1_token_id = int(highest_ids[0])

    top2_token_id: int | None = None
    top2_probability: float | None = None
    top1_top2_gap_bits: float | None = None
    if vocab_size >= 2:
        if highest_ids.size >= 2:
            top2_token_id = int(highest_ids[1])
        else:
            second_logit = float(np.partition(logits, -2)[-2])
            top2_token_id = int(np.flatnonzero(logits == second_logit)[0])
        top2_probability = float(raw_probabilities[top2_token_id])
        top1_top2_gap_bits = max(
            0.0,
            float(
                (float(logits[top1_token_id]) - float(logits[top2_token_id]))
                / math.log(2.0)
            ),
        )

    tail_top_k = 10
    visible_count = min(tail_top_k, vocab_size)
    if visible_count == vocab_size:
        tail_probability = 0.0
    else:
        top_mass = float(
            np.sum(
                np.partition(raw_probabilities, vocab_size - visible_count)[
                    vocab_size - visible_count :
                ]
            )
        )
        tail_probability = max(0.0, 1.0 - top_mass)

    return {
        "temperature": 1.0,
        "entropy_bits": entropy_bits,
        "effective_tokens": float(2**entropy_bits),
        "normalized_entropy": (
            entropy_bits / math.log2(vocab_size)
            if vocab_size > 1
            else 0.0
        ),
        "tail_top_k": tail_top_k,
        "tail_probability": tail_probability,
        "top1_token_id": top1_token_id,
        "top1_probability": float(raw_probabilities[top1_token_id]),
        "top2_token_id": top2_token_id,
        "top2_probability": top2_probability,
        "top1_top2_gap_bits": top1_top2_gap_bits,
    }
