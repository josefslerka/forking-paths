from __future__ import annotations

import math

import pytest

from apertus.garden.core.performance import (
    LIVE_HYBRID_MAX_PEAK_RSS_BYTES,
    STUB_PROCESS_BUDGET,
    STUB_SCENARIO_BUDGETS,
    ScenarioBudget,
    evaluate_scenario,
    percentile,
    process_peak_rss_bytes,
    summarize_scenario,
)


def test_percentile_is_interpolated_and_rejects_invalid_samples() -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0], 0.95) == pytest.approx(2.9)
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile([math.inf], 0.5)
    with pytest.raises(ValueError):
        percentile([1.0], 1.1)


def test_scenario_summary_and_evaluation_name_every_violation() -> None:
    summary = summarize_scenario(
        [10.0, 20.0, 30.0],
        [100, 200, 300],
    )
    assert summary == {
        "samples": 3,
        "median_ms": 20.0,
        "p95_ms": pytest.approx(29.0),
        "max_ms": 30.0,
        "max_response_bytes": 300,
    }
    evaluation = evaluate_scenario(
        summary,
        ScenarioBudget(
            median_ms=19.0,
            p95_ms=28.0,
            max_response_bytes=299,
        ),
    )
    assert evaluation["passed"] is False
    assert len(evaluation["violations"]) == 3
    assert all(
        metric in " ".join(evaluation["violations"])
        for metric in ("median_ms", "p95_ms", "max_response_bytes")
    )


def test_release_budgets_cover_every_stable_computation_class() -> None:
    assert set(STUB_SCENARIO_BUDGETS) == {
        "health",
        "tokenization",
        "next_token_uncached",
        "next_token_cached",
        "text_surprisal",
        "context_contrast",
    }
    assert STUB_PROCESS_BUDGET.bounded_session_requests >= 64
    assert STUB_PROCESS_BUDGET.max_rss_growth_bytes > 0
    assert LIVE_HYBRID_MAX_PEAK_RSS_BYTES >= 20 * 1024**3
    assert process_peak_rss_bytes() > 0
