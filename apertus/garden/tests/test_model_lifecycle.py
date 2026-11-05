from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apertus.garden.experiments import RECORD_SCHEMA, ExperimentStore
from apertus.garden.server import (
    MODEL_BOUND_CACHE_FIELDS,
    ModelChangedError,
    ModelState,
)


pytestmark = pytest.mark.contract


class CacheCatalogSpy:
    def __init__(self) -> None:
        self.clear_count = 0

    def clear_runtime_cache(self) -> None:
        self.clear_count += 1


def lifecycle_state() -> ModelState:
    state = ModelState(Path("model-a"), mock=True)
    state.model_paths = {
        "model-a": Path("model-a"),
        "model-b": Path("model-b"),
    }
    state.active_model_id = "model-a"
    state.model_path = Path("model-a")
    state.status = "ready"
    state.phase = None
    state.model_generation = 3
    return state


def populate_model_bound_state(state: ModelState) -> None:
    for field_name in MODEL_BOUND_CACHE_FIELDS:
        getattr(state, field_name)["sentinel"] = {"model": "old"}
    state.active_ids = [1, 2, 3]
    state.active_cache = object()
    state.active_logits = np.array([1.0], dtype=np.float64)
    state.vocab_catalog = [(1, "old", "old")]
    state.tokenizer_identity_cache = {"kind": "old"}
    state.tokenizer = object()
    state.model = object()


def test_release_clears_every_declared_model_bound_value() -> None:
    state = lifecycle_state()
    tuned_catalog = CacheCatalogSpy()
    jacobian_catalog = CacheCatalogSpy()
    state.tuned_lens_catalog = tuned_catalog  # type: ignore[assignment]
    state.jacobian_lens_catalog = jacobian_catalog  # type: ignore[assignment]
    populate_model_bound_state(state)

    state._release_model_local()

    assert all(
        not getattr(state, field_name)
        for field_name in MODEL_BOUND_CACHE_FIELDS
    )
    assert state.active_ids is None
    assert state.active_cache is None
    assert state.active_logits is None
    assert state.vocab_catalog is None
    assert state.tokenizer_identity_cache is None
    assert state.tokenizer is None
    assert state.model is None
    assert tuned_catalog.clear_count == 1
    assert jacobian_catalog.clear_count == 1


def test_switch_increments_generation_and_recovers_after_failed_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = lifecycle_state()
    populate_model_bound_state(state)
    model_b = object()
    tokenizer_b = object()
    monkeypatch.setattr(
        state,
        "_load_model_components",
        lambda model_id, model_path: (model_b, tokenizer_b),
    )

    state._switch_model_local("model-b")

    assert state.status == "ready"
    assert state.active_model_id == "model-b"
    assert state.model_path == Path("model-b")
    assert state.model is model_b
    assert state.tokenizer is tokenizer_b
    assert state.model_generation == 4
    assert all(
        not getattr(state, field_name)
        for field_name in MODEL_BOUND_CACHE_FIELDS
    )

    def fail_load(model_id: str, model_path: Path) -> tuple[Any, Any]:
        raise RuntimeError(f"broken {model_id} at {model_path}")

    monkeypatch.setattr(state, "_load_model_components", fail_load)
    state._switch_model_local("model-a")

    assert state.status == "error"
    assert state.phase is None
    assert state.target_model_id is None
    assert state.model is None
    assert state.tokenizer is None
    assert state.model_generation == 4
    assert state.error == "RuntimeError: broken model-a at model-a"

    model_a = object()
    tokenizer_a = object()
    monkeypatch.setattr(
        state,
        "_load_model_components",
        lambda model_id, model_path: (model_a, tokenizer_a),
    )
    state._switch_model_local("model-a")

    assert state.status == "ready"
    assert state.active_model_id == "model-a"
    assert state.model is model_a
    assert state.tokenizer is tokenizer_a
    assert state.model_generation == 5
    assert state.error is None


def test_switch_noop_and_stale_runtime_identity_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = lifecycle_state()
    monkeypatch.setattr(
        state,
        "_model_available_unlocked",
        lambda model_id, model_path: True,
    )

    response = state.request_model_switch("model-a")

    assert response["accepted"] is False
    assert state.model_generation == 3
    assert state.work_queue.empty()

    state._validate_request_identity_unlocked(
        3,
        state.server_instance_id,
        "model-a",
    )
    with pytest.raises(
        ModelChangedError,
        match="Server byl od vytvoření této zahrady restartován",
    ):
        state._validate_request_identity_unlocked(
            3,
            "old-server",
            "model-a",
        )
    with pytest.raises(
        ModelChangedError,
        match="Model se od vytvoření této zahrady změnil",
    ):
        state._validate_request_identity_unlocked(
            2,
            state.server_instance_id,
            "model-a",
        )
    with pytest.raises(
        ModelChangedError,
        match="jiný než právě aktivní model",
    ):
        state._validate_request_identity_unlocked(
            3,
            state.server_instance_id,
            "model-b",
        )


def test_archive_identity_and_records_survive_store_restart(
    tmp_path: Path,
) -> None:
    database = tmp_path / "experiments.sqlite3"
    record_id = str(uuid.uuid4())
    record = {
        "schema": RECORD_SCHEMA,
        "schema_version": 1,
        "id": record_id,
        "kind": "lifecycle_restart_contract",
        "captured_at": "2026-08-06T12:00:00.000Z",
        "annotation": {
            "title": "Restart",
            "notes": "",
            "tags": ["contract"],
        },
        "capture": {"status": "complete"},
        "provenance": {
            "model": {
                "model_id": "mock",
                "model_generation": 3,
                "server_instance_id": "before-restart",
            }
        },
        "payload": {"value": "persistent"},
        "diagnostics": {},
    }

    first_store = ExperimentStore(database)
    archive_id = first_store.status()["archive_id"]
    first_store.save(record)

    restarted_store = ExperimentStore(database)

    assert restarted_store.status()["archive_id"] == archive_id
    assert restarted_store.status()["record_count"] == 1
    assert restarted_store.get(record_id)["payload"] == {
        "value": "persistent"
    }
