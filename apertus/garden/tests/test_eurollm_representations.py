from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apertus.garden.model_registry import model_descriptor
from apertus.garden.tests.test_model_comparison import ready_state


@pytest.fixture
def state(tmp_path: Path):
    value = ready_state(tmp_path)
    value._switch_model_local("mock-eurollm9")
    yield value
    value.stop()
    if value.worker_thread is not None:
        value.worker_thread.join(timeout=2)


def test_supported_runtime_exposes_raw_and_embeddings_but_no_foreign_calibration(tmp_path: Path) -> None:
    config = {"model_type": "llama", "num_hidden_layers": 42}
    (tmp_path / "config.json").write_text(json.dumps(config))
    caps = model_descriptor("eurollm9", tmp_path, "Euro", backend="mlx", mock=False)["capabilities"]
    assert {"raw", "embeddings", "steering"} <= set(caps)
    assert not {"tuned", "jlens", "attribution", "patching", "detector"} & set(caps)
    for backend, sliding in (("torch", None), ("mlx", 128)):
        config["sliding_window"] = sliding
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert not {"raw", "embeddings"} & set(model_descriptor(
            "eurollm9", tmp_path, "Euro", backend=backend, mock=False)["capabilities"])


def test_raw_has_42_layers_and_final_self_distance_zero(state) -> None:
    result = state.lens("Hlavním městem České republiky je", 8)
    assert result["layer_count"] == len(result["layers"]) == 42
    assert result["layers"][-1]["layer"] == 42
    assert result["layers"][-1]["relative_depth"] == 1.
    assert result["layers"][-1]["js_divergence_to_final_bits"] == 0.
    assert result["layers"][-1]["final_top1_rank"] == 1
    assert result["stable_from_relative_depth"] == result["stable_from_layer"] / 42
    assert result["protocol"] == "raw-layers-v2"
    assert state.lens(result["text"], 8)["cached"] is True
    state._switch_model_local("mock")
    again = state.lens(result["text"], 8)
    assert again["cached"] is False and again["layer_count"] == 32


def test_summary_rejects_wrong_depth_instead_of_silently_truncating(state) -> None:
    with pytest.raises(ValueError, match="42"):
        state._summarize_lens_logits(np.zeros((32, 128)), 8)


def test_context_trajectory_reaches_42_and_does_not_read_the_suffix(state) -> None:
    contexts = [{"label": "Kopec", "text": "Na kopci stojí zámek.", "token_index": 3},
                {"label": "Dveře", "text": "Na dveřích je rozbitý zámek.", "token_index": 4}]
    result = state.embeddings(contexts, 8)
    assert result["layer_count"] == 42
    assert len(result["contexts"][0]["layers"]) == 43
    assert len(result["context_projection"]["contexts"][0]["points"]) == 43
    assert result["contexts"][0]["layers"][-1]["state_summary"]["dimension"] == 64
    assert result["pairs"][0]["points"][-1]["relative_depth"] == 1.
    assert result["pairs"][0]["input_cosine_distance"] == pytest.approx(0., abs=1e-12)
    assert result["pairs"][0]["max_distance"] > 0.
    assert result["execution"]["future_suffix_used"] is False
    with_suffix = [{**c, "text": c["text"] + " Budoucí slova se nesmějí použít."} for c in contexts]
    changed = state.embeddings(with_suffix, 8)
    assert result["pairs"] == changed["pairs"]
    assert result["contexts"][0]["layers"] == changed["contexts"][0]["layers"]
    state._switch_model_local("mock")
    assert state.embeddings(contexts, 8)["layer_count"] == 32


def test_word_span_labels_the_actual_depth(state) -> None:
    text = "Na kopci stojí zámek."
    context = {"label": "Kopec", "text": text, "token_index": 3,
               "span_start_token_index": 3, "span_end_token_index": 3,
               "span_character_start": 15, "span_character_end": 20, "span_text": "zámek"}
    result = state.embeddings([context], 8, analysis_mode="word_span")
    assert "L1_to_L42" in result["word_span"]["representation_contract"]
    assert "L1–L42" in result["interpretation"]["context"]
    assert result["contexts"][0]["layers"][-1]["layer"] == 42
