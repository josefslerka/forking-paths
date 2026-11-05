"""Release performance budgets and deterministic evaluation helpers.

The budgets in this module deliberately separate a repeatable mock-runtime
regression gate from hardware-dependent live Apertus smoke tests.  They are
not a promise that every supported machine has identical inference latency.
"""

from __future__ import annotations

import math
import resource
import sys
from dataclasses import asdict, dataclass
from typing import Iterable


BUDGET_VERSION = "0.1.0"
MIB = 1024 * 1024


@dataclass(frozen=True)
class ScenarioBudget:
    """Upper bounds for one repeated HTTP scenario."""

    median_ms: float
    p95_ms: float
    max_response_bytes: int


@dataclass(frozen=True)
class ProcessBudget:
    """Coarse process bounds intended to catch order-of-magnitude regressions."""

    max_rss_bytes: int
    max_rss_growth_bytes: int
    bounded_session_requests: int
    bounded_session_ms: float


# These bounds are intentionally wider than a normal local run.  Their job is
# to catch blocking I/O, accidental real-model work in --mock, unbounded JSON
# growth, and cache/memory regressions without turning scheduler jitter into a
# release failure.
STUB_SCENARIO_BUDGETS: dict[str, ScenarioBudget] = {
    "health": ScenarioBudget(
        median_ms=40.0,
        p95_ms=120.0,
        max_response_bytes=256_000,
    ),
    "tokenization": ScenarioBudget(
        median_ms=100.0,
        p95_ms=300.0,
        max_response_bytes=1_000_000,
    ),
    "next_token_uncached": ScenarioBudget(
        median_ms=120.0,
        p95_ms=400.0,
        max_response_bytes=2_000_000,
    ),
    "next_token_cached": ScenarioBudget(
        median_ms=60.0,
        p95_ms=200.0,
        max_response_bytes=2_000_000,
    ),
    "text_surprisal": ScenarioBudget(
        median_ms=180.0,
        p95_ms=600.0,
        max_response_bytes=4_000_000,
    ),
    "context_contrast": ScenarioBudget(
        median_ms=300.0,
        p95_ms=900.0,
        max_response_bytes=8_000_000,
    ),
}

STUB_PROCESS_BUDGET = ProcessBudget(
    max_rss_bytes=768 * MIB,
    max_rss_growth_bytes=192 * MIB,
    bounded_session_requests=96,
    bounded_session_ms=8_000.0,
)

# A safety ceiling for the supported local Hybrid runtime.  This is asserted
# by the live browser gate, while request timeouts remain workload-specific.
# It is deliberately not presented as a cross-hardware latency SLA.
LIVE_HYBRID_MAX_PEAK_RSS_BYTES = 32 * 1024 * MIB


def percentile(values: Iterable[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for finite observations."""

    observations = sorted(float(value) for value in values)
    if not observations:
        raise ValueError("Percentil vyžaduje alespoň jedno pozorování.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("Kvantile musí být mezi 0 a 1.")
    if any(not math.isfinite(value) or value < 0.0 for value in observations):
        raise ValueError(
            "Výkonnostní pozorování musí být konečná nezáporná čísla."
        )
    if len(observations) == 1:
        return observations[0]
    position = (len(observations) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return observations[lower_index]
    fraction = position - lower_index
    return (
        observations[lower_index] * (1.0 - fraction)
        + observations[upper_index] * fraction
    )


def summarize_scenario(
    latency_ms: Iterable[float],
    response_bytes: Iterable[int],
) -> dict[str, float | int]:
    """Summarize one scenario without depending on the HTTP runner."""

    latencies = [float(value) for value in latency_ms]
    sizes = [int(value) for value in response_bytes]
    if not latencies or len(latencies) != len(sizes):
        raise ValueError(
            "Latence a velikosti odpovědí musí mít stejný neprázdný počet."
        )
    if any(size < 0 for size in sizes):
        raise ValueError("Velikost odpovědi nesmí být záporná.")
    return {
        "samples": len(latencies),
        "median_ms": percentile(latencies, 0.5),
        "p95_ms": percentile(latencies, 0.95),
        "max_ms": max(latencies),
        "max_response_bytes": max(sizes),
    }


def evaluate_scenario(
    summary: dict[str, float | int],
    budget: ScenarioBudget,
) -> dict[str, object]:
    """Compare a scenario summary with its release bounds."""

    violations: list[str] = []
    for metric, limit in (
        ("median_ms", budget.median_ms),
        ("p95_ms", budget.p95_ms),
        ("max_response_bytes", budget.max_response_bytes),
    ):
        observed = float(summary[metric])
        if observed > float(limit):
            violations.append(
                f"{metric}: {observed:.3f} > {float(limit):.3f}"
            )
    return {
        "passed": not violations,
        "violations": violations,
        "budget": asdict(budget),
    }


def process_peak_rss_bytes() -> int:
    """Return peak RSS with the platform-specific ru_maxrss unit normalized."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the other supported Unix test runners
    # report KiB.
    return value if sys.platform == "darwin" else value * 1024


__all__ = [
    "BUDGET_VERSION",
    "LIVE_HYBRID_MAX_PEAK_RSS_BYTES",
    "MIB",
    "ProcessBudget",
    "STUB_PROCESS_BUDGET",
    "STUB_SCENARIO_BUDGETS",
    "ScenarioBudget",
    "evaluate_scenario",
    "percentile",
    "process_peak_rss_bytes",
    "summarize_scenario",
]
