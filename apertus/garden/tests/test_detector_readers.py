from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from apertus.garden import detector_readers
from apertus.garden.detector_readers import JOB_TTL_SECONDS, new_job, pair_window, run_comparison
from apertus.garden.experiments import ExperimentStore
from apertus.garden.server import ModelState


def response_for_spans(text, spans, *, surprise=1.0):
    return {"analysis_kind": "text_surprisal", "normalized_text": text,
            "token_ids": list(range(1, len(spans) + 1)),
            "tokens": [{"character_start": start, "character_end": end,
                        "surprisal_bits": surprise, "entropy_bits": .5}
                       for start, end in spans], "timing_ms": {"forward": 1}}


def sample(start=128, end=256):
    return {"index": 1, "position": "early", "context_start": 0,
            "target_start": start, "target_end": end, "target_token_count": end - start}


def test_identical_characters_with_different_native_token_counts():
    text = "p" * 128 + "t" * 128
    a = response_for_spans(text, [(i, i + 1) for i in range(256)])
    b = response_for_spans(text, [(i, i + 2) for i in range(0, 256, 2)], surprise=3)
    window, rows_a, rows_b = pair_window(sample(), a, b, None, None, None, None, mock=True)
    assert window["status"] == "paired"
    assert window["target_non_whitespace_character_count"] == 128
    assert len(rows_a) == 128 and len(rows_b) == 64
    assert window["measurements"]["apertus"]["features"]["surprisal_bits_per_non_whitespace_character"] == 1
    assert window["measurements"]["eurollm"]["features"]["surprisal_bits_per_non_whitespace_character"] == 1.5
    assert window["delta_bits_per_character"] == .5
    assert "p" * 100 not in json.dumps(window)


def test_boundary_trim_depends_on_token_boundaries_and_not_surprisal():
    text = "p" * 127 + "q" + "t" * 128
    a = response_for_spans(text, [(i, i + 1) for i in range(256)])
    b = response_for_spans(text, [(0, 127), (127, 129), *[(i, i + 1) for i in range(129, 256)]])
    first, _, _ = pair_window(sample(), a, b, None, None, None, None, mock=True)
    b["tokens"][2]["surprisal_bits"] = 1000
    second, _, _ = pair_window(sample(), a, b, None, None, None, None, mock=True)
    assert first["target_utf8_byte_start"] == second["target_utf8_byte_start"] == 129
    assert first["trimmed_apertus_target_tokens"] == 1


def test_target_start_inside_unicode_is_trimmed_to_a_literal_shared_boundary():
    text = "p" * 128 + "Ť" + "t" * 128
    raw_a = [*"p" * 128, "Å", "¤", *"t" * 128]
    raw_b = [*"p" * 128, "Ť", *"t" * 128]
    def raw_response(raw):
        return {"analysis_kind": "text_surprisal", "normalized_text": text,
                "token_ids": list(range(1, len(raw) + 1)),
                "tokens": [{"tokenizer_token": token, "surprisal_bits": 1, "entropy_bits": .5} for token in raw]}
    class PrefixTokenizer:
        def __init__(self, split_unicode):
            self.split_unicode = split_unicode
        def encode(self, prefix, **_):
            count = len(prefix.encode()) if self.split_unicode else len(prefix)
            return list(range(1, count + 1))
    window, a, b = pair_window(sample(129, 258), raw_response(raw_a), raw_response(raw_b),
                               PrefixTokenizer(True), PrefixTokenizer(False), "byte_level_bpe", "sentencepiece_bpe")
    assert window["status"] == "paired"
    assert window["trimmed_apertus_target_tokens"] == 1
    assert window["context_character_count"] == 129
    assert window["target_sha256"] == hashlib.sha256(("t" * 128).encode()).hexdigest()
    assert len(a) == len(b) == 128


def test_canonical_prefix_is_verified_even_with_exact_byte_offsets():
    text = "p" * 128 + "t" * 128
    a = {"analysis_kind": "text_surprisal", "normalized_text": text,
         "token_ids": list(range(1, 257)),
         "tokens": [{"tokenizer_token": char, "surprisal_bits": 1, "entropy_bits": .5} for char in text]}
    class Retokenizing:
        def encode(self, prefix, **_):
            return [999] if len(prefix) < 256 else list(range(1, 257))
    window, rows_a, rows_b = pair_window(sample(), a, a, Retokenizing(), Retokenizing(), "byte_level_bpe", "byte_level_bpe")
    assert window["status"] == "excluded" and window["reason"] == "no_shared_canonical_boundary"
    assert rows_a == rows_b == []


@pytest.mark.parametrize("trim", [16, 17])
def test_trim_limit_is_enforced_before_aggregation(trim):
    text = "p" * 128 + "t" * 128
    a = response_for_spans(text, [(i, i + 1) for i in range(256)])
    b = response_for_spans(text, [(0, 127), (127, 128 + trim), *[(i, i + 1) for i in range(128 + trim, 256)]])
    window, _, _ = pair_window(sample(), a, b, None, None, None, None, mock=True)
    assert (window["status"] == "paired") == (trim == 16)


def test_unknown_offsets_and_nonfinite_values_are_never_fabricated():
    text = "p" * 128 + "t" * 128
    a = response_for_spans(text, [(i, i + 1) for i in range(256)])
    broken = copy.deepcopy(a)
    broken["tokens"][0]["character_start"] = None
    window, _, _ = pair_window(sample(), a, broken, None, None, None, None, mock=True)
    assert window["reason"] == "unreliable_eurollm_offsets"
    broken = copy.deepcopy(a)
    broken["tokens"][128]["surprisal_bits"] = float("nan")
    with pytest.raises(ValueError, match="neplatné"):
        pair_window(sample(), a, broken, None, None, None, None, mock=True)


def fake_detector(state, monkeypatch):
    """Only replace the calibrated extractor; exercise the real worker/scorers."""
    state.predictive_detector_engine = SimpleNamespace(tokenizer=None)
    def analyze(text, *, capture_window=None):
        windows = []
        for index, position in enumerate(("early", "middle_1", "late"), 1):
            window_text = f"okno{index} " + "slovo " * 254 + "konec"
            response = state._text_surprisal_local(window_text, 1)
            if capture_window is not None:
                capture_window(window_text, response)
            windows.append({**sample(), "index": index, "position": position})
        return {"analysis_kind": "predictive_stylometry", "schema_version": 2,
                "analysis_id": "calibrated-fixture", "verdict": "no_ai_signal",
                "primary_decision": {"model": "apertus", "challenger_changes_verdict": False},
                "challenger": {"decision_authority": False},
                "scores": {"primary": {"models": {"apertus": {"score": .314159, "strict_threshold": .479}}}},
                "reader": {"model_id": state.active_model_id, "model_generation": state.model_generation,
                           "server_instance_id": state.server_instance_id},
                "apertus_features": {"calibrated_original": 123},
                "sampling": {"apertus": {"windows": windows}},
                "input": {"sha256": hashlib.sha256(text.encode()).hexdigest()}}
    monkeypatch.setattr(state, "_predictive_stylometry_local", analyze)
    return analyze


def identity(state):
    return {"model_id": state.active_model_id, "model_generation": state.model_generation,
            "server_instance_id": state.server_instance_id}


def ready_state(tmp_path):
    state = ModelState(tmp_path, mock=True)
    state.load()
    state.comparison_store = ExperimentStore(tmp_path / "archive.sqlite3")
    return state


def test_parallel_measurements_preserve_calibrated_verdict_restore_and_do_not_archive(tmp_path, monkeypatch):
    state = ready_state(tmp_path)
    analyze = fake_detector(state, monkeypatch)
    text = "Dokument, který se nesmí ocitnout v archivu."
    baseline = analyze(text)
    job = new_job(text, identity(state), "mock-eurollm9")
    run_comparison(state, job)
    assert job["phase"] == "done" and job["status"] == "completed"
    assert job["completed_measurements"] == 6
    measured = job["result"]
    assert {key: value for key, value in measured.items() if key != "reader_comparison"} == baseline
    comparison = measured["reader_comparison"]
    assert comparison["decision_authority"] is False and comparison["research_only"] is True
    assert comparison["implementation_sha256"] == hashlib.sha256(Path(detector_readers.__file__).read_bytes()).hexdigest()
    assert comparison["summary"]["paired_windows"] == 3
    assert state.active_model_id == "mock" and state.status == "ready"
    assert job["restored_identity"]["model_generation"] == baseline["reader"]["model_generation"] + 2
    assert "text" not in job
    assert text not in json.dumps(measured, ensure_ascii=False)
    assert state.comparison_store.list_summaries()["total"] == 0


@pytest.mark.parametrize("failure", ["scoring", "loading"])
def test_euro_failure_preserves_apertus_result_and_releases_document(tmp_path, monkeypatch, failure):
    state = ready_state(tmp_path)
    fake_detector(state, monkeypatch)
    original = state._text_surprisal_local
    switch = state._switch_model_local
    def score(*args):
        if state.active_model_id == "mock-eurollm9":
            raise RuntimeError("Euro scoring failure")
        return original(*args)
    def failed_load(model):
        if model == "mock-eurollm9":
            state.status, state.error = "error", "Euro loading failure"
        else:
            switch(model)
    monkeypatch.setattr(state, "_text_surprisal_local", score)
    if failure == "loading":
        monkeypatch.setattr(state, "_switch_model_local", failed_load)
    job = new_job("private document", identity(state), "mock-eurollm9")
    run_comparison(state, job)
    assert job["status"] == "failed" and job["phase"] == "done"
    assert job["result"]["scores"]["primary"]["models"]["apertus"]["score"] == .314159
    assert job["result"]["reader_comparison"]["summary"]["paired_windows"] == 0
    assert state.active_model_id == "mock" and state.status == "ready"
    assert "text" not in job


def test_cancel_retains_only_completed_pairs_and_restores_original(tmp_path, monkeypatch):
    state = ready_state(tmp_path)
    fake_detector(state, monkeypatch)
    job = new_job("private document", identity(state), "mock-eurollm9")
    original = state._text_surprisal_local
    def score(*args):
        response = original(*args)
        if state.active_model_id == "mock-eurollm9":
            job["cancel"].set()
        return response
    monkeypatch.setattr(state, "_text_surprisal_local", score)
    run_comparison(state, job)
    assert job["status"] == "cancelled" and job["completed_measurements"] == 4
    assert job["result"]["reader_comparison"]["summary"]["paired_windows"] == 1
    assert state.active_model_id == "mock" and "text" not in job


def test_worker_reservation_stale_identity_cancel_and_expiry(tmp_path, monkeypatch):
    state = ready_state(tmp_path)
    fake_detector(state, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = state._text_surprisal_local
    def paused(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)
    monkeypatch.setattr(state, "_text_surprisal_local", paused)
    state.start()
    try:
        with pytest.raises(Exception, match="generace|změnil|identity|model",):
            state.start_detector_reader_comparison("private", "mock", 999, state.server_instance_id)
        job = state.start_detector_reader_comparison("private", **{
            "expected_model_id": "mock", "expected_generation": state.model_generation,
            "expected_server_instance_id": state.server_instance_id})
        assert entered.wait(3)
        with pytest.raises(RuntimeError):
            state.request_model_switch("mock-eurollm9")
        with pytest.raises(RuntimeError):
            state.start_model_comparison({"model_ids": ["mock", "mock-eurollm9"], "items": [{"text": "Ahoj"}]})
        state.detector_reader_status(job["id"], cancel=True)
        release.set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = state.detector_reader_status(job["id"])
            if result["phase"] == "done":
                break
            time.sleep(.01)
        assert result["status"] == "cancelled" and result["result"] is None
        assert "text" not in state.detector_reader_jobs[job["id"]]
        assert state.active_model_id == "mock"
        state.detector_reader_jobs[job["id"]]["created_monotonic"] -= JOB_TTL_SECONDS + 1
        with pytest.raises(ValueError, match="vypršelo"):
            state.detector_reader_status(job["id"])
    finally:
        release.set()
        state.stop()
        state.worker_thread.join(3)


def test_restore_failure_is_reported_and_sensitive_input_is_released(tmp_path, monkeypatch):
    state = ready_state(tmp_path)
    fake_detector(state, monkeypatch)
    switch = state._switch_model_local
    def fail_restore(model):
        if model == "mock":
            state.status, state.error = "error", "Restore failure"
        else:
            switch(model)
    monkeypatch.setattr(state, "_switch_model_local", fail_restore)
    job = new_job("sensitive input", identity(state), "mock-eurollm9")
    run_comparison(state, job)
    assert job["phase"] == "done" and job["status"] == "failed"
    assert job["restored_identity"] is None and job["restoration_error"] == "Restore failure"
    assert job["result"]["reader_comparison"]["decision_authority"] is False
    assert "text" not in job
