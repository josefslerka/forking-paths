"""Cache-free traces through the model's own MLX decoder blocks.

States retain their native dtype until readout. No decoder, normalization,
attention mask or MLP is reimplemented here. The global garden KV cache is
never read or modified.
"""
from __future__ import annotations

from typing import Any

from .steering import decoder_body


def project_context_states(trajectories: list[Any]) -> dict[str, Any]:
    """One fixed PCA frame for all contexts and layers, not one fit per panel.

    Unit normalization matches the directional comparison used by cosine
    distance. The two displayed coordinates are illustrative; measurements
    continue to use the original full-dimensional states.
    """
    import numpy as np

    arrays = [np.asarray(states, dtype=np.float64) for states in trajectories]
    if not arrays or any(states.ndim != 2 or not states.size for states in arrays):
        raise ValueError("Context trajectories must be nonempty matrices")
    if any(states.shape != arrays[0].shape for states in arrays):
        raise ValueError("Context trajectories must share layers and dimensions")
    rows = np.vstack(arrays)
    if not np.isfinite(rows).all():
        raise ValueError("Context states must be finite")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    unit_rows = np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)
    centered = unit_rows - unit_rows.mean(axis=0, keepdims=True)
    # The small Gram matrix avoids a full SVD over thousands of coordinates.
    eigenvalues, eigenvectors = np.linalg.eigh(centered @ centered.T)
    order = np.argsort(eigenvalues)[::-1]
    positive = np.maximum(eigenvalues[order], 0.0)
    total_variance = float(positive.sum())
    coordinates = np.zeros((len(rows), 2), dtype=np.float64)
    explained = [0.0, 0.0]
    for component, index in enumerate(order[:2]):
        value = float(eigenvalues[index])
        if value <= max(1e-14, total_variance * 1e-12):
            continue
        axis = centered.T @ eigenvectors[:, index] / np.sqrt(value)
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        coordinates[:, component] = centered @ axis
        explained[component] = value / total_variance
    layer_count = arrays[0].shape[0]
    return {
        "method": "shared_pca_unit_context_states_all_layers",
        "normalization": "unit_l2_before_shared_pca",
        "dimension": int(rows.shape[1]),
        "explained_variance_ratio": explained,
        "contexts": [
            {
                "context_index": context_index,
                "points": [
                    {"layer": layer, "x": float(point[0]), "y": float(point[1])}
                    for layer, point in enumerate(
                        coordinates[context_index * layer_count:(context_index + 1) * layer_count]
                    )
                ],
            }
            for context_index in range(len(arrays))
        ],
        "scope_warning": (
            "One shared projection and scale for all contexts and layers; "
            "the panels are not fitted independently. Projected distances "
            "are illustrative, not the full-dimensional cosine distance."
        ),
    }


def trace_last_position(model: Any, token_ids: list[int]) -> tuple[Any, Any, Any]:
    """Return body, native L0..LN states and the final full-prefix residual."""
    body, states, hidden = trace_positions(model, token_ids, [len(token_ids) - 1])
    return body, states[:, 0, :], hidden


def trace_positions(model: Any, token_ids: list[int], positions: list[int]) -> tuple[Any, Any, Any]:
    """Trace selected causal positions, retaining native precision and L0."""
    if not token_ids or not positions or any(type(p) is not int or p < 0 or p >= len(token_ids) for p in positions):
        raise ValueError("Selected positions must be valid indices in a nonempty prefix")
    if positions != sorted(set(positions)):
        raise ValueError("Selected positions must be unique and ordered")
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    body = decoder_body(model)
    hidden = body.embed_tokens(mx.array([token_ids]))
    mask = create_attention_mask(hidden, None)
    states = [hidden[0, positions, :]]
    for layer in body.layers:
        hidden = layer(hidden, mask=mask, cache=None)
        states.append(hidden[0, positions, :])
    stacked = mx.stack(states, axis=0)
    mx.eval(stacked)
    return body, stacked, hidden


def readout(model: Any, body: Any, states: Any) -> Any:
    """Apply the model's own final norm and tied or untied output head."""
    normalized = body.norm(states)
    head = getattr(model, "lm_head", None)
    if callable(head):
        return head(normalized)
    return body.embed_tokens.as_linear(normalized)
