"""Pure numerical helpers for residual-stream activation patching.

The server owns model execution and token semantics.  This module keeps the
counterfactual accounting small, testable and independent of MLX.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


class ActivationPatchingError(ValueError):
    """Raised when a patching contrast cannot be summarized safely."""


CONTRAST_DENOMINATOR_EPSILON = 1e-6
JS_DENOMINATOR_EPSILON = 1e-12


def _finite_logits(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ActivationPatchingError(
            f"{name} musí být jednorozměrný vektor alespoň dvou logitů."
        )
    if not np.all(np.isfinite(array)):
        raise ActivationPatchingError(f"{name} obsahuje neplatná čísla.")
    return array


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    """Return a stable float64 softmax."""

    values = _finite_logits("Logity", logits)
    shifted = values - float(np.max(values))
    exponentials = np.exp(shifted)
    denominator = float(np.sum(exponentials, dtype=np.float64))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ActivationPatchingError(
            "Softmax logitů nemá platný normalizační součet."
        )
    return exponentials / denominator


def entropy_bits(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ActivationPatchingError(
            "Entropie vyžaduje konečný vektor pravděpodobností."
        )
    positive = values > 0.0
    return float(
        -np.sum(
            values[positive] * np.log2(values[positive]),
            dtype=np.float64,
        )
    )


def jensen_shannon_bits(
    left_probabilities: np.ndarray,
    right_probabilities: np.ndarray,
) -> float:
    """Symmetric Jensen–Shannon divergence in bits."""

    left = np.asarray(left_probabilities, dtype=np.float64)
    right = np.asarray(right_probabilities, dtype=np.float64)
    if (
        left.ndim != 1
        or right.ndim != 1
        or left.shape != right.shape
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or np.any(left < 0.0)
        or np.any(right < 0.0)
    ):
        raise ActivationPatchingError(
            "JS divergence vyžaduje dva konečné pravděpodobnostní vektory "
            "stejného tvaru."
        )
    middle = 0.5 * (left + right)

    def kl_bits(source: np.ndarray) -> float:
        positive = source > 0.0
        return float(
            np.sum(
                source[positive]
                * np.log2(source[positive] / middle[positive]),
                dtype=np.float64,
            )
        )

    # Numerical roundoff can produce tiny negative dust.
    return max(0.0, 0.5 * (kl_bits(left) + kl_bits(right)))


def token_rank(logits: np.ndarray, token_id: int) -> int:
    values = _finite_logits("Logity", logits)
    if not 0 <= token_id < values.size:
        raise ActivationPatchingError(
            f"Token #{token_id} leží mimo vektor logitů."
        )
    return 1 + int(np.count_nonzero(values > values[token_id]))


@dataclass(frozen=True)
class PatchingReference:
    """Immutable A/B baseline used to summarize every patched cell."""

    source_logits: np.ndarray
    target_logits: np.ndarray
    source_probabilities: np.ndarray
    target_probabilities: np.ndarray
    token_a: int
    token_b: int
    source_contrast: float
    target_contrast: float
    contrast_denominator: float
    baseline_js_bits: float


def build_reference(
    source_logits: np.ndarray,
    target_logits: np.ndarray,
    token_a: int,
    token_b: int,
) -> PatchingReference:
    source = _finite_logits("Logity reference A", source_logits)
    target = _finite_logits("Logity příjemce B", target_logits)
    if source.shape != target.shape:
        raise ActivationPatchingError(
            "A a B mají rozdílnou velikost výstupního slovníku."
        )
    if token_a == token_b:
        raise ActivationPatchingError(
            "Kontrast vyžaduje dva různé výstupní tokeny."
        )
    for token_id in (token_a, token_b):
        if not 0 <= token_id < source.size:
            raise ActivationPatchingError(
                f"Token #{token_id} leží mimo výstupní slovník."
            )

    source_probabilities = probabilities_from_logits(source)
    target_probabilities = probabilities_from_logits(target)
    source_contrast = float(source[token_a] - source[token_b])
    target_contrast = float(target[token_a] - target[token_b])
    denominator = source_contrast - target_contrast
    if abs(denominator) <= CONTRAST_DENOMINATOR_EPSILON:
        raise ActivationPatchingError(
            "Výstupní kontrast A a B je příliš malý pro stabilní recovery. "
            "Zvol reference, které se v kontrastu skutečně liší."
        )
    return PatchingReference(
        source_logits=source,
        target_logits=target,
        source_probabilities=source_probabilities,
        target_probabilities=target_probabilities,
        token_a=token_a,
        token_b=token_b,
        source_contrast=source_contrast,
        target_contrast=target_contrast,
        contrast_denominator=denominator,
        baseline_js_bits=jensen_shannon_bits(
            source_probabilities,
            target_probabilities,
        ),
    )


def summarize_patch(
    reference: PatchingReference,
    patched_logits: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Summarize one patched distribution against A and B baselines."""

    patched = _finite_logits("Logity po patchi", patched_logits)
    if patched.shape != reference.source_logits.shape:
        raise ActivationPatchingError(
            "Patch vrátil jinou velikost výstupního slovníku."
        )
    probabilities = probabilities_from_logits(patched)
    contrast = float(
        patched[reference.token_a] - patched[reference.token_b]
    )
    delta_contrast = contrast - reference.target_contrast
    contrast_recovery = (
        delta_contrast / reference.contrast_denominator
    )
    js_to_source = jensen_shannon_bits(
        probabilities,
        reference.source_probabilities,
    )
    js_to_target = jensen_shannon_bits(
        probabilities,
        reference.target_probabilities,
    )
    distribution_recovery = (
        1.0 - js_to_source / reference.baseline_js_bits
        if reference.baseline_js_bits > JS_DENOMINATOR_EPSILON
        else None
    )
    top1_token_id = int(
        np.flatnonzero(patched == np.max(patched))[0]
    )
    return (
        {
            "contrast": contrast,
            "delta_contrast_from_target": delta_contrast,
            "contrast_recovery": float(contrast_recovery),
            "probability_a": float(probabilities[reference.token_a]),
            "probability_b": float(probabilities[reference.token_b]),
            "rank_a": token_rank(patched, reference.token_a),
            "rank_b": token_rank(patched, reference.token_b),
            "entropy_bits": entropy_bits(probabilities),
            "js_to_source_bits": js_to_source,
            "js_to_target_bits": js_to_target,
            "distribution_recovery": (
                float(distribution_recovery)
                if distribution_recovery is not None
                else None
            ),
            "top1_token_id": top1_token_id,
            "top1_probability": float(probabilities[top1_token_id]),
        },
        probabilities,
    )


def identity_control(
    reference: PatchingReference,
    recomputed_target_logits: np.ndarray,
) -> dict[str, float]:
    """Measure numerical closure of an explicit B→B no-op recomputation."""

    diagnostics = recomputation_control(
        reference.target_logits,
        recomputed_target_logits,
        reference.token_a,
        reference.token_b,
    )
    return {
        "max_abs_logit_error": diagnostics["max_abs_logit_error"],
        "contrast_delta": diagnostics["contrast_delta"],
        "js_to_target_bits": diagnostics["js_to_expected_bits"],
    }


def recomputation_control(
    expected_logits: np.ndarray,
    recomputed_logits: np.ndarray,
    token_a: int,
    token_b: int,
) -> dict[str, float]:
    """Compare a recomputed control row with its expected baseline."""

    expected = _finite_logits(
        "Očekávané logity kontroly",
        expected_logits,
    )
    recomputed = _finite_logits(
        "Přepočtené logity kontroly",
        recomputed_logits,
    )
    if recomputed.shape != expected.shape:
        raise ActivationPatchingError(
            "Kontrolní přepočet vrátil jinou velikost slovníku."
        )
    for token_id in (token_a, token_b):
        if not 0 <= token_id < expected.size:
            raise ActivationPatchingError(
                f"Token #{token_id} leží mimo výstupní slovník."
            )
    expected_probabilities = probabilities_from_logits(expected)
    recomputed_probabilities = probabilities_from_logits(recomputed)
    expected_contrast = float(expected[token_a] - expected[token_b])
    recomputed_contrast = float(recomputed[token_a] - recomputed[token_b])
    return {
        "max_abs_logit_error": float(
            np.max(np.abs(recomputed - expected))
        ),
        "contrast_delta": recomputed_contrast - expected_contrast,
        "js_to_expected_bits": jensen_shannon_bits(
            recomputed_probabilities,
            expected_probabilities,
        ),
    }
