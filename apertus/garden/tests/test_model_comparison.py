from __future__ import annotations

import csv
import io
import json
import threading
import time
from pathlib import Path

import pytest

from apertus.garden.experiments import ExperimentStore
from apertus.garden.model_comparison import aligned_units, new_job, run_comparison, validate_request
from apertus.garden.model_registry import model_descriptor
from apertus.garden.server import ModelState


def request() -> dict:
    return {"model_ids": ["mock", "mock-eurollm9"], "title": "Párové měření",
            "items": [{"id": "text", "text": "Ahoj 300 Ťok."},
                      {"id": "context", "prefix_a": "Lékař řekl, že", "prefix_b": "Pacient řekl, že",
                       "continuation": " operace proběhla úspěšně."}]}


def ready_state(tmp_path: Path) -> ModelState:
    state = ModelState(tmp_path, mock=True)
    state.load()
    state.comparison_store = ExperimentStore(tmp_path / "archive.sqlite3")
    return state


def wait_job(state: ModelState, job_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = state.model_comparison_status(job_id)
        if result["phase"] == "done":
            return result
        time.sleep(.01)
    raise AssertionError("Comparison did not finish")


def test_standalone_space_belongs_to_following_canonical_path() -> None:
    response = {"normalized_text": "A 300 Ťok.", "tokens": [
        {"character_start": 0, "character_end": 1, "surprisal_bits": 1},
        {"character_start": 1, "character_end": 2, "surprisal_bits": 2},
        {"character_start": 2, "character_end": 5, "surprisal_bits": 3},
        {"character_start": 5, "character_end": 10, "surprisal_bits": 4},
    ]}
    result = aligned_units(response, mock=True)
    assert [u["surprisal_bits"] for u in result["units"]] == [1, 5, 4]
    assert result["assigned_token_count"] == result["token_count"] == 4
    assert result["unassigned_token_indices"] == []


def test_token_crossing_shared_units_is_excluded_without_double_counting() -> None:
    response = {"normalized_text": "A B", "tokens": [{"character_start": 0, "character_end": 3, "surprisal_bits": 5}]}
    result = aligned_units(response, mock=True)
    assert result["assigned_token_count"] == 0
    assert result["unassigned_token_indices"] == [0]
    assert all(not u["complete"] and u["surprisal_bits"] is None for u in result["units"])


def test_unknown_unicode_offset_does_not_fabricate_a_word_score() -> None:
    result = aligned_units({"normalized_text": "Ťok", "tokens": [{"character_start": None, "character_end": None, "surprisal_bits": 5}]}, mock=True)
    assert result["units"][0]["surprisal_bits"] is None
    assert "unreliable_token_offsets" in result["units"][0]["reasons"]


@pytest.mark.parametrize("raw_tokens", [["Na", "ĠÅ", "¤ok", "."], ["▁Na", "▁", "<0xC5>", "<0xA4>", "ok", "."]])
def test_split_utf8_character_is_recovered_from_exact_bytes(raw_tokens) -> None:
    class Tokenizer:
        def encode(self, text, **kwargs):
            return {"": [0], "Na": [0, 1], "Na Ťok.": [0, *range(1, len(raw_tokens) + 1)]}[text]
    tokens = [{"tokenizer_token": raw, "character_start": None, "character_end": None, "surprisal_bits": i + 1}
              for i, raw in enumerate(raw_tokens)]
    result = aligned_units({"normalized_text": "Na Ťok.", "tokens": tokens}, Tokenizer())
    assert all(unit["complete"] for unit in result["units"])
    assert result["assigned_token_count"] == len(tokens)
    assert sum(unit["surprisal_bits"] for unit in result["units"]) == sum(t["surprisal_bits"] for t in tokens)
    assert "exact_bytes" in result["token_offset_source"]


def test_prefix_retokenization_is_not_reported_as_a_complete_conditional_path() -> None:
    class Tokenizer:
        def encode(self, text: str, **kwargs: object) -> list[int]:
            return {"A B": [0, 1, 2], "": [0], "A": [0, 99]}[text]
    response = {"normalized_text": "A B", "tokens": [
        {"character_start": 0, "character_end": 1, "surprisal_bits": 1},
        {"character_start": 1, "character_end": 3, "surprisal_bits": 2},
    ]}
    result = aligned_units(response, Tokenizer())
    assert all(not u["complete"] for u in result["units"])


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(model_ids=["mock", "mock"]),
    lambda r: r.update(items=r["items"] * 17),
    lambda r: r["items"][1].update(id="text"),
    lambda r: r["items"][0].update(prefix_a="x"),
    lambda r: r["items"][1].update(prefix_a="text "),
    lambda r: r["items"][1].update(continuation=" "),
])
def test_invalid_protocol_is_rejected_before_model_switch(mutation: object) -> None:
    value = request()
    mutation(value)
    with pytest.raises(ValueError):
        validate_request(value)


def test_whole_batch_is_paired_saved_and_original_model_restored(tmp_path: Path) -> None:
    state = ready_state(tmp_path)
    job = new_job(validate_request(request()), state.server_instance_id)
    run_comparison(state, job)
    assert job["status"] == "completed" and job["saved"]
    assert state.active_model_id == "mock" and state.status == "ready"
    assert job["completed_measurements"] == 4
    assert job["record"]["payload"]["summary"]["paired_items"] == 2
    assert job["record"]["provenance"]["models"]["mock-eurollm9"]["family"] == "llama"
    a = job["measurements"]["mock"]["text"]["response"]
    b = job["measurements"]["mock-eurollm9"]["text"]["response"]
    assert a["summary"]["total_surprisal_bits"] != b["summary"]["total_surprisal_bits"]
    exported = json.loads(state.comparison_store.export_json(record_id=job["id"]))
    copied = ExperimentStore(tmp_path / "copy.sqlite3")
    copied.import_payload(exported)
    assert copied.get(job["id"])["payload"] == job["record"]["payload"]
    csv_rows = list(csv.DictReader(io.StringIO(state.comparison_store.export_csv(record_id=job["id"]).decode())))
    assert any(row["comparison_item_kind"] == "context" for row in csv_rows)
    assert any(row["comparison_unit_text"] == "300" for row in csv_rows)


def test_cancel_reserves_worker_and_retains_partial_measurement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = ready_state(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = state._text_surprisal_local
    def slow(*args: object) -> dict:
        entered.set()
        assert release.wait(3)
        return original(*args)
    monkeypatch.setattr(state, "_text_surprisal_local", slow)
    state.start()
    try:
        job = state.start_model_comparison(request())
        assert entered.wait(3)
        with pytest.raises(RuntimeError):
            state.request_model_switch("mock-eurollm9")
        with pytest.raises(RuntimeError):
            state.start_model_comparison(request())
        state.model_comparison_status(job["id"], cancel=True)
        release.set()
        result = wait_job(state, job["id"])
        assert result["status"] == "cancelled" and result["saved"]
        assert result["completed_measurements"] == 1
        assert result["record"]["capture"]["status"] == "cancelled"
        assert state.active_model_id == "mock"
    finally:
        release.set()
        state.stop()
        state.worker_thread.join(3)


def test_failed_second_model_load_keeps_first_results_and_recovers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = ready_state(tmp_path)
    original = state._load_model_components
    def load(model_id: str, path: Path) -> tuple:
        if model_id == "mock-eurollm9":
            raise RuntimeError("broken weights")
        return original(model_id, path)
    monkeypatch.setattr(state, "_load_model_components", load)
    job = new_job(validate_request(request()), state.server_instance_id)
    run_comparison(state, job)
    assert job["status"] == "failed" and job["saved"]
    assert job["completed_measurements"] == 2
    assert "broken weights" in job["errors"][0]
    assert state.active_model_id == "mock" and state.status == "ready"


def test_capabilities_reject_apertus_calibrations_for_eurollm(tmp_path: Path) -> None:
    state = ready_state(tmp_path)
    state._switch_model_local("mock-eurollm9")
    with pytest.raises(ValueError):
        state.require_capability("detector")
    with pytest.raises(ValueError):
        state.require_capability("tuned")
    state.require_capability("raw")
    state.require_capability("embeddings")
    state.require_capability("surprise")


def test_model_registry_reads_architecture_instead_of_assuming_32_layers(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "num_hidden_layers": 42, "quantization": {"bits": 6}}))
    descriptor = model_descriptor("eurollm9", tmp_path, "EuroLLM 9B", backend="mlx", mock=False)
    assert descriptor["layers"] == 42 and descriptor["quantization"]["bits"] == 6
    assert {"raw", "embeddings"} <= set(descriptor["capabilities"])


def test_examples_are_valid_unique_exploratory_items() -> None:
    data = json.loads((Path(__file__).resolve().parents[1] / "model_comparison_examples.json").read_text())
    value = validate_request({"model_ids": ["a", "b"], "items": data["items"]})
    assert len(value["items"]) == 30
    assert data["status"] == "exploratory_examples_not_preregistered"
