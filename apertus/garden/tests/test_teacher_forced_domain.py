from __future__ import annotations

import math

import numpy as np
import pytest

from apertus.garden.domains.teacher_forced import (
    log_probabilities,
    persistent_half_life,
    probability_ratio_from_log2_advantage,
    score_target_rows,
    token_likelihood,
    trapezoid_area,
    validate_logit_matrix,
)


def test_shared_token_likelihood_contract() -> None:
    logits = np.asarray([0.0, 2.0, 1.0], dtype=np.float32)
    score = token_likelihood(logits, 2)
    normalized = log_probabilities(logits)

    assert score.token_id == 2
    assert score.rank == 2
    assert score.logprob == pytest.approx(float(normalized[2]))
    assert score.probability == pytest.approx(math.exp(normalized[2]))
    assert score.surprisal_bits == pytest.approx(
        -score.logprob / math.log(2.0)
    )


def test_matrix_scoring_validates_alignment_and_vocabulary() -> None:
    logits = np.asarray(
        [[0.0, 1.0, 2.0], [4.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    scores = score_target_rows(logits, [2, 0])

    assert [score.rank for score in scores] == [1, 1]
    assert validate_logit_matrix(logits, expected_rows=2) is logits
    with pytest.raises(RuntimeError, match="pozice × slovník"):
        validate_logit_matrix(logits, expected_rows=3)
    with pytest.raises(RuntimeError, match="mimo slovník"):
        score_target_rows(logits, [2, 9])


def test_context_persistence_helpers_are_explicit_at_edges() -> None:
    assert probability_ratio_from_log2_advantage(2.0) == (4.0, "finite")
    assert persistent_half_life([1.0, 0.6, 0.49, 0.4]) == 2
    assert persistent_half_life([1.0, 0.4, 0.6]) is None
    assert persistent_half_life([0.0, 0.0]) is None
    assert trapezoid_area([1.0, 0.5, 0.0]) == pytest.approx(1.0)
