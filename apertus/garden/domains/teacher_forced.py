"""Shared numerical contracts for teacher-forced analyses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


NATURAL_LOG_OF_TWO = math.log(2.0)


@dataclass(frozen=True, slots=True)
class TokenLikelihood:
    """Likelihood of one vocabulary item under one causal logit row."""

    token_id: int
    logprob: float
    probability: float
    log2_probability: float
    surprisal_bits: float
    rank: int


def validate_logit_row(
    logits: np.ndarray,
    *,
    error_message: str = "Teacher-forced čtečka vrátila neplatné logity.",
) -> np.ndarray:
    values = np.asarray(logits).astype(np.float64, copy=False)
    if (
        values.ndim != 1
        or values.size < 1
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(error_message)
    return values


def validate_logit_matrix(
    logits: np.ndarray,
    *,
    expected_rows: int,
    error_message: str = (
        "Teacher-forced forward vrátil neplatnou matici pozice × slovník."
    ),
) -> np.ndarray:
    values = np.asarray(logits)
    if (
        values.ndim != 2
        or values.shape[0] != expected_rows
        or values.shape[1] < 1
        or not np.all(np.isfinite(values))
    ):
        raise RuntimeError(error_message)
    return values


def log_probabilities(logits: np.ndarray) -> np.ndarray:
    values = validate_logit_row(
        logits,
        error_message="Čtečka vrátila neplatný vektor logitů.",
    )
    maximum = float(np.max(values))
    shifted = values - maximum
    return shifted - math.log(float(np.sum(np.exp(shifted))))


def token_likelihood(
    logits: np.ndarray,
    token_id: int,
    *,
    out_of_range_label: str = "Cílový token",
) -> TokenLikelihood:
    values = validate_logit_row(logits)
    if not 0 <= token_id < values.size:
        raise RuntimeError(
            f"{out_of_range_label} {token_id} leží mimo slovník "
            f"o velikosti {values.size}."
        )
    normalized = log_probabilities(values)
    logprob = float(normalized[token_id])
    probability = math.exp(logprob)
    log2_probability = logprob / NATURAL_LOG_OF_TWO
    return TokenLikelihood(
        token_id=token_id,
        logprob=logprob,
        probability=probability,
        log2_probability=log2_probability,
        surprisal_bits=-log2_probability,
        rank=1 + int(np.count_nonzero(values > values[token_id])),
    )


def score_target_rows(
    logits: np.ndarray,
    target_token_ids: list[int],
) -> list[TokenLikelihood]:
    values = validate_logit_matrix(
        logits,
        expected_rows=len(target_token_ids),
    )
    return [
        token_likelihood(
            values[index],
            int(token_id),
            out_of_range_label="Token textu",
        )
        for index, token_id in enumerate(target_token_ids)
    ]


def probability_ratio_from_log2_advantage(
    information_advantage_bits: float,
) -> tuple[float | None, str]:
    """Convert a log2 ratio without hiding overflow or underflow."""
    log_ratio = float(information_advantage_bits) * NATURAL_LOG_OF_TWO
    max_log = math.log(float(np.finfo(np.float64).max))
    min_log = math.log(float(np.nextafter(0.0, 1.0)))
    if log_ratio > max_log:
        return None, "overflow"
    if log_ratio < min_log:
        return 0.0, "underflow"
    return math.exp(log_ratio), "finite"


def persistent_half_life(
    values: list[float],
) -> int | None:
    """Return the first step after which the series stays below half root."""
    if not values or values[0] <= 0.0:
        return None
    threshold = float(values[0]) / 2.0
    for index in range(1, len(values)):
        if all(float(value) <= threshold for value in values[index:]):
            return index
    return None


def trapezoid_area(values: list[float]) -> float:
    return math.fsum(
        0.5 * (float(left) + float(right))
        for left, right in zip(values, values[1:])
    )


__all__ = [
    "NATURAL_LOG_OF_TWO",
    "TokenLikelihood",
    "log_probabilities",
    "persistent_half_life",
    "probability_ratio_from_log2_advantage",
    "score_target_rows",
    "token_likelihood",
    "trapezoid_area",
    "validate_logit_matrix",
    "validate_logit_row",
]
