"""Exact residual-stream accounting for a two-token logit contrast.

The decoder residual stream is additive, while Apertus' final RMSNorm is a
single scalar-and-diagonal map once the final residual is fixed.  This module
uses that fixed map to assign the final ``logit(A) - logit(B)`` contrast to the
embedding, every attention update, every MLP update, and explicitly reported
finite-precision remainders.

This is descriptive accounting for one realised forward pass.  It is not a
causal ablation and it does not identify where a concept is represented.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class ComponentAttributionError(ValueError):
    """Raised when a residual decomposition cannot be computed safely."""


def _finite_vector(
    name: str,
    value: np.ndarray | list[float],
    *,
    width: int | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ComponentAttributionError(f"{name} musí být jednorozměrný vektor.")
    if width is not None and array.shape != (width,):
        raise ComponentAttributionError(
            f"{name} má tvar {array.shape}; očekává se ({width},)."
        )
    if not np.all(np.isfinite(array)):
        raise ComponentAttributionError(f"{name} obsahuje neplatná čísla.")
    return array


def _finite_matrix(
    name: str,
    value: np.ndarray | list[list[float]],
    *,
    rows: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ComponentAttributionError(f"{name} musí být matice.")
    if rows is not None and array.shape[0] != rows:
        raise ComponentAttributionError(
            f"{name} má {array.shape[0]} řádků; očekává se {rows}."
        )
    if width is not None and array.shape[1] != width:
        raise ComponentAttributionError(
            f"{name} má šířku {array.shape[1]}; očekává se {width}."
        )
    if not np.all(np.isfinite(array)):
        raise ComponentAttributionError(f"{name} obsahuje neplatná čísla.")
    return array


def _rms_readout(
    state: np.ndarray,
    *,
    norm_weight: np.ndarray,
    row_a: np.ndarray,
    row_b: np.ndarray,
    epsilon: float,
) -> tuple[float, float, float, float]:
    denominator = math.sqrt(float(np.mean(state * state)) + epsilon)
    normalized = state * norm_weight / denominator
    logit_a = float(normalized @ row_a)
    logit_b = float(normalized @ row_b)
    return logit_a, logit_b, logit_a - logit_b, denominator


def _component_payload(
    *,
    kind: str,
    label: str,
    vector: np.ndarray,
    final_linear_scale: np.ndarray,
    row_a: np.ndarray,
    row_b: np.ndarray,
    layer: int | None = None,
) -> dict[str, Any]:
    normalized_component = vector * final_linear_scale
    logit_a = float(normalized_component @ row_a)
    logit_b = float(normalized_component @ row_b)
    return {
        "kind": kind,
        "label": label,
        "layer": layer,
        "logit_a_contribution": logit_a,
        "logit_b_contribution": logit_b,
        "contrast_contribution": logit_a - logit_b,
        "residual_norm": float(np.linalg.norm(vector)),
    }


def decompose_residual_contrast(
    *,
    embedding: np.ndarray | list[float],
    attention_updates: np.ndarray | list[list[float]],
    mlp_updates: np.ndarray | list[list[float]],
    final_residual: np.ndarray | list[float],
    norm_weight: np.ndarray | list[float],
    lm_head_row_a: np.ndarray | list[float],
    lm_head_row_b: np.ndarray | list[float],
    epsilon: float,
    after_attention_states: np.ndarray | list[list[float]] | None = None,
    after_layer_states: np.ndarray | list[list[float]] | None = None,
    actual_logit_a: float | None = None,
    actual_logit_b: float | None = None,
) -> dict[str, Any]:
    """Decompose one realised next-token contrast into decoder components.

    ``embedding`` and all update matrices refer to the final prefix position.
    The same final RMSNorm denominator and learned diagonal are used for every
    additive component.  A residual-rounding vector closes the residual sum;
    an additional scalar numerical closure makes finite-precision model logits
    explicit instead of silently attributing them to a decoder block.
    """

    if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon):
        raise ComponentAttributionError("epsilon musí být konečné číslo.")
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ComponentAttributionError("epsilon musí být kladné.")

    embedding_array = _finite_vector("embedding", embedding)
    width = int(embedding_array.shape[0])
    if width == 0:
        raise ComponentAttributionError("Reziduální prostor nesmí být prázdný.")
    attention_array = _finite_matrix(
        "attention_updates",
        attention_updates,
        width=width,
    )
    layer_count = int(attention_array.shape[0])
    if layer_count == 0:
        raise ComponentAttributionError("Je nutná alespoň jedna vrstva.")
    mlp_array = _finite_matrix(
        "mlp_updates",
        mlp_updates,
        rows=layer_count,
        width=width,
    )
    final_array = _finite_vector(
        "final_residual",
        final_residual,
        width=width,
    )
    norm_array = _finite_vector(
        "norm_weight",
        norm_weight,
        width=width,
    )
    row_a = _finite_vector("lm_head_row_a", lm_head_row_a, width=width)
    row_b = _finite_vector("lm_head_row_b", lm_head_row_b, width=width)

    after_attention_array = (
        None
        if after_attention_states is None
        else _finite_matrix(
            "after_attention_states",
            after_attention_states,
            rows=layer_count,
            width=width,
        )
    )
    after_layer_array = (
        None
        if after_layer_states is None
        else _finite_matrix(
            "after_layer_states",
            after_layer_states,
            rows=layer_count,
            width=width,
        )
    )

    if (actual_logit_a is None) != (actual_logit_b is None):
        raise ComponentAttributionError(
            "actual_logit_a a actual_logit_b musejí být uvedeny společně."
        )
    if actual_logit_a is not None:
        actual_logit_a = float(actual_logit_a)
        actual_logit_b = float(actual_logit_b)
        if not math.isfinite(actual_logit_a) or not math.isfinite(actual_logit_b):
            raise ComponentAttributionError(
                "Skutečné modelové logity musejí být konečné."
            )

    residual_sum_without_remainder = (
        embedding_array
        + np.sum(attention_array, axis=0)
        + np.sum(mlp_array, axis=0)
    )
    residual_rounding = final_array - residual_sum_without_remainder
    final_denominator = math.sqrt(
        float(np.mean(final_array * final_array)) + epsilon
    )
    final_linear_scale = norm_array / final_denominator

    embedding_payload = _component_payload(
        kind="embedding",
        label="Vstupní embedding",
        vector=embedding_array,
        final_linear_scale=final_linear_scale,
        row_a=row_a,
        row_b=row_b,
    )
    components: list[dict[str, Any]] = [embedding_payload]
    layers: list[dict[str, Any]] = []
    cumulative_a = float(embedding_payload["logit_a_contribution"])
    cumulative_b = float(embedding_payload["logit_b_contribution"])

    for index in range(layer_count):
        layer_number = index + 1
        attention_payload = _component_payload(
            kind="attention",
            label=f"L{layer_number} · attention",
            layer=layer_number,
            vector=attention_array[index],
            final_linear_scale=final_linear_scale,
            row_a=row_a,
            row_b=row_b,
        )
        components.append(attention_payload)
        cumulative_a += float(attention_payload["logit_a_contribution"])
        cumulative_b += float(attention_payload["logit_b_contribution"])
        cumulative_after_attention = cumulative_a - cumulative_b

        mlp_payload = _component_payload(
            kind="mlp",
            label=f"L{layer_number} · MLP",
            layer=layer_number,
            vector=mlp_array[index],
            final_linear_scale=final_linear_scale,
            row_a=row_a,
            row_b=row_b,
        )
        components.append(mlp_payload)
        cumulative_a += float(mlp_payload["logit_a_contribution"])
        cumulative_b += float(mlp_payload["logit_b_contribution"])

        layer_payload: dict[str, Any] = {
            "layer": layer_number,
            "attention": attention_payload,
            "mlp": mlp_payload,
            "layer_contrast_contribution": (
                float(attention_payload["contrast_contribution"])
                + float(mlp_payload["contrast_contribution"])
            ),
            "cumulative_after_attention": cumulative_after_attention,
            "cumulative_after_layer": cumulative_a - cumulative_b,
        }
        if after_attention_array is not None:
            (
                local_a,
                local_b,
                local_gap,
                local_denominator,
            ) = _rms_readout(
                after_attention_array[index],
                norm_weight=norm_array,
                row_a=row_a,
                row_b=row_b,
                epsilon=epsilon,
            )
            layer_payload["local_reader_after_attention"] = {
                "logit_a": local_a,
                "logit_b": local_b,
                "contrast": local_gap,
                "rms_denominator": local_denominator,
            }
        if after_layer_array is not None:
            (
                local_a,
                local_b,
                local_gap,
                local_denominator,
            ) = _rms_readout(
                after_layer_array[index],
                norm_weight=norm_array,
                row_a=row_a,
                row_b=row_b,
                epsilon=epsilon,
            )
            layer_payload["local_reader_after_layer"] = {
                "logit_a": local_a,
                "logit_b": local_b,
                "contrast": local_gap,
                "rms_denominator": local_denominator,
            }
        layers.append(layer_payload)

    residual_rounding_payload = _component_payload(
        kind="residual_rounding",
        label="Reziduální zaokrouhlení",
        vector=residual_rounding,
        final_linear_scale=final_linear_scale,
        row_a=row_a,
        row_b=row_b,
    )
    components.append(residual_rounding_payload)
    cumulative_a += float(residual_rounding_payload["logit_a_contribution"])
    cumulative_b += float(residual_rounding_payload["logit_b_contribution"])

    formula_logit_a, formula_logit_b, formula_gap, _ = _rms_readout(
        final_array,
        norm_weight=norm_array,
        row_a=row_a,
        row_b=row_b,
        epsilon=epsilon,
    )
    target_logit_a = formula_logit_a if actual_logit_a is None else actual_logit_a
    target_logit_b = formula_logit_b if actual_logit_b is None else actual_logit_b
    numerical_closure_payload = {
        "kind": "numerical_closure",
        "label": "Numerická uzávěrka projekce",
        "layer": None,
        "logit_a_contribution": target_logit_a - cumulative_a,
        "logit_b_contribution": target_logit_b - cumulative_b,
        "contrast_contribution": (
            (target_logit_a - cumulative_a)
            - (target_logit_b - cumulative_b)
        ),
        "residual_norm": None,
    }
    components.append(numerical_closure_payload)

    reconstructed_a = cumulative_a + float(
        numerical_closure_payload["logit_a_contribution"]
    )
    reconstructed_b = cumulative_b + float(
        numerical_closure_payload["logit_b_contribution"]
    )
    target_gap = target_logit_a - target_logit_b
    reconstructed_gap = reconstructed_a - reconstructed_b
    final_residual_norm = float(np.linalg.norm(final_array))
    residual_rounding_norm = float(np.linalg.norm(residual_rounding))

    ranked_components = sorted(
        (
            component
            for component in components
            if component["kind"] in {"attention", "mlp"}
        ),
        key=lambda component: abs(
            float(component["contrast_contribution"])
        ),
        reverse=True,
    )

    return {
        "method": {
            "id": "fixed_final_rmsnorm_residual_accounting",
            "version": "1.0.0",
            "contrast_definition": "logit(A) - logit(B)",
            "status": "descriptive_not_causal",
            "statement_cs": (
                "Skutečné výstupy attention a MLP z jednoho dopředného "
                "průchodu se promítají společnou finální RMSNorm do "
                "kontrastu logit(A) − logit(B). Jde o aditivní účetnictví "
                "realizovaného průchodu, nikoli o ablaci ani kauzální důkaz."
            ),
        },
        "layer_count": layer_count,
        "embedding": embedding_payload,
        "layers": layers,
        "residual_rounding": residual_rounding_payload,
        "numerical_closure": numerical_closure_payload,
        "components": components,
        "strongest_components": ranked_components[:12],
        "normalization": {
            "kind": "fixed_final_rmsnorm",
            "epsilon": epsilon,
            "final_rms_denominator": final_denominator,
        },
        "diagnostics": {
            "final_residual_norm": final_residual_norm,
            "residual_rounding_norm": residual_rounding_norm,
            "residual_rounding_relative_norm": (
                residual_rounding_norm / max(final_residual_norm, 1e-30)
            ),
            "formula_logit_a": formula_logit_a,
            "formula_logit_b": formula_logit_b,
            "formula_contrast": formula_gap,
            "pre_closure_logit_a_error": target_logit_a - cumulative_a,
            "pre_closure_logit_b_error": target_logit_b - cumulative_b,
            "pre_closure_contrast_error": (
                target_gap - (cumulative_a - cumulative_b)
            ),
            "post_closure_contrast_error": target_gap - reconstructed_gap,
        },
        "final": {
            "logit_a": target_logit_a,
            "logit_b": target_logit_b,
            "contrast": target_gap,
            "winner": "A" if target_gap > 0 else "B" if target_gap < 0 else "tie",
            "reconstructed_logit_a": reconstructed_a,
            "reconstructed_logit_b": reconstructed_b,
            "reconstructed_contrast": reconstructed_gap,
            "reconstruction_error": target_gap - reconstructed_gap,
        },
    }
