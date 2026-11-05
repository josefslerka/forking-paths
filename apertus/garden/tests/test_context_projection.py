from __future__ import annotations

import numpy as np
import pytest

from apertus.garden.domains.representations import project_context_states


def test_shared_projection_preserves_equal_states_and_uses_all_layers():
    left = np.array([[1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    right = np.array([[1., 0., 0.], [1., -1., 0.], [0., -1., 0.]])
    result = project_context_states([left, right])
    assert len(result["contexts"]) == 2
    assert sum(result["explained_variance_ratio"]) == pytest.approx(1.)
    projected = np.array([[p["x"], p["y"]] for c in result["contexts"] for p in c["points"]])
    np.testing.assert_allclose(projected[0], projected[3], atol=1e-12)
    rows = np.vstack([left, right])
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    # This example has rank two, so the shared projection retains all distances.
    for a in range(len(rows)):
        for b in range(len(rows)):
            assert np.linalg.norm(projected[a] - projected[b]) == pytest.approx(np.linalg.norm(rows[a] - rows[b]), abs=1e-12)


@pytest.mark.parametrize("constant", [0., 1.])
def test_constant_states_produce_finite_zero_coordinates(constant):
    result = project_context_states([np.full((4, 8), constant)] * 2)
    assert result["explained_variance_ratio"] == [0., 0.]
    assert all(p["x"] == 0. and p["y"] == 0. for c in result["contexts"] for p in c["points"])


def test_unit_normalization_does_not_encode_norm_as_a_direction_change():
    left = np.array([[1., 0.], [1., 1.]])
    result = project_context_states([left, left * 100.])
    for a, b in zip(result["contexts"][0]["points"], result["contexts"][1]["points"], strict=True):
        assert a["x"] == pytest.approx(b["x"], abs=1e-12)
        assert a["y"] == pytest.approx(b["y"], abs=1e-12)


def test_invalid_state_arrays_are_rejected():
    for values in ([], [np.zeros((0, 3))], [np.ones((3, 4)), np.ones((2, 4))], [np.array([[np.nan]])]):
        with pytest.raises(ValueError):
            project_context_states(values)
