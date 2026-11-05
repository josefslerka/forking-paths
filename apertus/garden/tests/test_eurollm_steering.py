from __future__ import annotations

import csv
import io
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apertus.garden.domains.steering import decoder_body
from apertus.garden.model_comparison import new_job, run_comparison, validate_request
from apertus.garden.model_registry import model_descriptor
from apertus.garden.steering_comparison import path_contrasts, resolve_layer
from apertus.garden.tests.test_model_comparison import ready_state, wait_job
from apertus.garden.tests.test_steering import contrast_pairs


def request() -> dict:
    return {"model_ids": ["mock", "mock-eurollm9"], "title": "Steering obou modelů",
            "steering": {"contrast_pairs": contrast_pairs(), "relative_depth": .5,
                         "alpha_magnitude": 1., "sham_samples": 2, "seed": 20261008},
            "items": [{"id": "novy", "test_text": "Jeho reakce byla",
                       "negative_text": " velmi krutá.", "positive_text": " velmi laskavá."}]}


def test_layer_depth_and_capability_follow_the_actual_model(tmp_path: Path) -> None:
    config = {"model_type": "llama", "num_hidden_layers": 42}
    (tmp_path / "config.json").write_text(json.dumps(config))
    descriptor = model_descriptor("eurollm9", tmp_path, "Euro", backend="mlx", mock=False)
    assert descriptor["layers"] == 42 and "steering" in descriptor["capabilities"]
    assert "patching" not in descriptor["capabilities"] and "detector" not in descriptor["capabilities"]
    assert "steering" not in model_descriptor("eurollm9", tmp_path, "Euro", backend="torch", mock=False)["capabilities"]
    config["sliding_window"] = 128
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert "steering" not in model_descriptor("eurollm9", tmp_path, "Euro", backend="mlx", mock=False)["capabilities"]
    assert resolve_layer(.5, 32) == 16
    assert resolve_layer(.5, 42) == 21
    assert resolve_layer(.75, 42) == 32


def test_decoder_guard_rejects_an_unreproduced_attention_path() -> None:
    body = SimpleNamespace(embed_tokens=lambda x: x, norm=lambda x: x,
                           layers=[SimpleNamespace(use_sliding=False)] * 42)
    model = SimpleNamespace(model_type="llama", model=body, lm_head=lambda x: x,
                            args=SimpleNamespace(num_hidden_layers=42))
    assert decoder_body(model) is body
    body.layers[0] = SimpleNamespace(use_sliding=True)
    with pytest.raises(RuntimeError, match="attention"):
        decoder_body(model)
    body.layers[0] = SimpleNamespace(use_sliding=False)
    model.args.num_hidden_layers = 32
    with pytest.raises(RuntimeError, match="konfigurací"):
        decoder_body(model)


def test_eurollm_accepts_layer_42_and_zero_is_identity_without_global_cache_changes(tmp_path: Path) -> None:
    state = ready_state(tmp_path)
    try:
        state._switch_model_local("mock-eurollm9")
        ids = [1, 2]
        cached = np.array([.1, .2])
        state.active_ids = ids
        state.active_logits = cached
        generation = state.model_generation
        result = state.steering(contrast_pairs(), "Nový test v jiném rámci", 42, 0., 1., 5, 6, 10,
                                state.active_model_id, generation, state.server_instance_id)
        assert result["total_variation"] == 0 and result["js_divergence_bits"] == 0
        assert state.active_ids is ids and state.active_logits is cached
        assert state.model_generation == generation
        with pytest.raises(ValueError, match="42"):
            state.steering(contrast_pairs(), "Nový test", 43, 0., 1., 5, 6, 10,
                           state.active_model_id, generation, state.server_instance_id)
        state._switch_model_local("mock")
        with pytest.raises(ValueError, match="32"):
            state.steering(contrast_pairs(), "Nový test", 33, 0., 1., 5, 6, 10,
                           state.active_model_id, state.model_generation, state.server_instance_id)
    finally:
        state.stop()
        if state.worker_thread is not None:
            state.worker_thread.join(timeout=2)


def test_cross_model_unit_is_whole_path_not_the_native_token_mean() -> None:
    def score(tokens: int) -> dict:
        return {"per_pair": [{
            "positive": {"token_count": tokens, "total_logprobs": {"negative": -8., "baseline": -6., "positive": -4.},
                         "mean_logprobs": {"negative": -8./tokens, "baseline": -6./tokens, "positive": -4./tokens}},
            "negative": {"token_count": 1, "total_logprobs": {key: -5. for key in ("negative", "baseline", "positive")}}
        }]}
    left, right = path_contrasts(score(1)), path_contrasts(score(20))
    assert left == right
    assert left["baseline"] == pytest.approx(-1 / np.log(2))
    bad = score(1)
    bad["per_pair"][0]["positive"]["total_logprobs"]["positive"] = float("nan")
    with pytest.raises(ValueError, match="konečné"):
        path_contrasts(bad)


@pytest.mark.parametrize("mutation", [
    lambda r: r["steering"].update(relative_depth=True),
    lambda r: r["steering"].update(relative_depth=0),
    lambda r: r["steering"].update(alpha_magnitude=float("inf")),
    lambda r: r["steering"].update(sham_samples=True),
    lambda r: r["steering"].update(seed=-1),
    lambda r: r["steering"].update(contrast_pairs=contrast_pairs()[:2]),
    lambda r: r["steering"].update(contrast_pairs=contrast_pairs()+[contrast_pairs()[0]]),
    lambda r: r["items"][0].update(text="Smíšený protokol"),
    lambda r: r["items"][0].update(positive_text=r["items"][0]["negative_text"]),
    lambda r: r["items"][0].update(test_text=contrast_pairs()[0]["positive_text"]),
])
def test_invalid_design_is_rejected_before_switching(mutation) -> None:
    value = request()
    mutation(value)
    with pytest.raises(ValueError):
        validate_request(value)


def test_shared_steering_job_retains_controls_own_layers_and_roundtrips(tmp_path: Path) -> None:
    state = ready_state(tmp_path)
    state.start()
    try:
        value = request()
        before = deepcopy(value)
        job = state.start_model_comparison(value)
        result = wait_job(state, job["id"])
        assert result["status"] == "completed" and result["saved"], result["errors"]
        assert state.active_model_id == "mock" and state.status == "ready"
        assert value == before
        record = result["record"]
        assert record["provenance"]["protocol"]["id"] == "steering-comparison-v1"
        row = record["payload"]["comparisons"][0]
        assert (row["a"]["layer"], row["b"]["layer"]) == (16, 21)
        for side in ("a", "b"):
            assert row[side]["sham_sample_count"] == 2
            assert row[side]["all_effective_norm_matches_passed"]
            assert row[side]["sham_reference_only"]
        exported = json.loads(state.comparison_store.export_json(record_id=job["id"]))
        assert exported["records"][0]["payload"] == record["payload"]
        csv_rows = list(csv.DictReader(io.StringIO(state.comparison_store.export_csv(record_id=job["id"]).decode())))
        assert csv_rows[0]["comparison_item_kind"] == "steering"
        assert float(csv_rows[0]["comparison_a_bits"].lstrip("'")) == row["a"]["effect_bits"]
        assert csv_rows[0]["steering_b_layer"] == "21"
        assert csv_rows[0]["steering_b_direction_sha256"] == row["b"]["direction_sha256"]
    finally:
        state.stop()
        if state.worker_thread is not None:
            state.worker_thread.join(timeout=2)


def test_partial_failure_preserves_measurement_and_restores_original_model(tmp_path: Path, monkeypatch) -> None:
    state = ready_state(tmp_path)
    try:
        original = state._steering_sham_control_local
        def fail_second_model(*args):
            if state.active_model_id == "mock-eurollm9":
                raise RuntimeError("Testovací chyba druhého modelu")
            return original(*args)
        monkeypatch.setattr(state, "_steering_sham_control_local", fail_second_model)
        job = new_job(validate_request(request()), state.server_instance_id)
        run_comparison(state, job)
        assert job["status"] == "failed"
        assert state.active_model_id == "mock" and state.status == "ready"
        assert job["completed_measurements"] == 1
        assert "novy" in job["measurements"]["mock"]
        assert job["record"]["payload"]["comparisons"] == []
        assert job["record"]["provenance"]["protocol"]["id"] == "steering-comparison-v1"
    finally:
        state.stop()
        if state.worker_thread is not None:
            state.worker_thread.join(timeout=2)
