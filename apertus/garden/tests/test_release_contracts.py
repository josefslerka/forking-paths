from __future__ import annotations

import csv
import io
import json
import math
import uuid
from pathlib import Path

import numpy as np
import pytest

from apertus.garden import server as server_module
from apertus.garden.api import contracts as api_contracts
from apertus.garden.component_attribution import decompose_residual_contrast
from apertus.garden.core.distributions import (
    distribution_distance,
    entropy_bits_from_probabilities,
    probabilities_at_temperature,
    probability_views,
    raw_distribution_metrics,
    summarize_probabilities,
)
from apertus.garden.experiments import RECORD_SCHEMA, ExperimentStore
from apertus.garden.server import (
    GardenHandler,
    ModelChangedError,
    ModelState,
)


pytestmark = pytest.mark.contract


class ExactCharacterTokenizer:
    """Small reversible tokenizer for contracts that must not load a model."""

    bos_token_id = 1
    eos_token_id = 2
    all_special_ids = [1, 2]
    vocab_size = 0x110000 + 100

    @staticmethod
    def _token_id(character: str) -> int:
        return ord(character) + 100

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        token_ids = [self._token_id(character) for character in text]
        return [self.bos_token_id, *token_ids] if add_special_tokens else token_ids

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return "".join(
            chr(token_id - 100)
            for token_id in token_ids
            if token_id not in self.all_special_ids
        )

    def convert_ids_to_tokens(self, token_id: int) -> str:
        if token_id == self.bos_token_id:
            return "<s>"
        if token_id == self.eos_token_id:
            return "</s>"
        return f"U+{token_id - 100:04X}"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list]:
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": self.encode(text, add_special_tokens=False),
            "offset_mapping": [
                (index, index + 1)
                for index in range(len(text))
            ],
        }


def ready_mock_state() -> ModelState:
    state = ModelState(Path("."), mock=True)
    state.status = "ready"
    state.model_generation = 7
    return state


def steering_pairs() -> list[dict[str, str]]:
    return [
        {
            "negative_text": "Tón je chladný v tomto rámci",
            "positive_text": "Tón je vřelý v tomto rámci",
        },
        {
            "negative_text": "Odpověď je odmítavá v tomto rámci",
            "positive_text": "Odpověď je vstřícná v tomto rámci",
        },
        {
            "negative_text": "Vyznění je odtažité v tomto rámci",
            "positive_text": "Vyznění je přátelské v tomto rámci",
        },
    ]


def experiment_record(kind: str = "contract_snapshot") -> dict:
    return {
        "schema": RECORD_SCHEMA,
        "schema_version": 1,
        "id": str(uuid.uuid4()),
        "kind": kind,
        "captured_at": "2026-08-06T08:00:00.000Z",
        "annotation": {
            "title": "Release contract",
            "notes": "",
            "tags": ["contract"],
        },
        "capture": {"status": "complete"},
        "provenance": {
            "model": {
                "model_id": "mock",
                "model_generation": 7,
                "server_instance_id": "contract-server",
            }
        },
        "payload": {"value": kind},
        "diagnostics": {},
    }


def test_softmax_is_finite_normalized_and_keeps_logit_order() -> None:
    logits = np.array([-1_000.0, -3.0, 0.0, 7.0], dtype=np.float64)

    reexports = {
        "distribution_distance": distribution_distance,
        "entropy_bits_from_probabilities": entropy_bits_from_probabilities,
        "probabilities_at_temperature": probabilities_at_temperature,
        "probability_views": probability_views,
        "raw_distribution_metrics": raw_distribution_metrics,
        "summarize_probabilities": summarize_probabilities,
    }
    assert all(
        getattr(server_module, name) is function
        for name, function in reexports.items()
    )
    for temperature in (0.2, 0.5, 1.0, 2.0):
        probabilities = probabilities_at_temperature(logits, temperature)

        assert np.all(np.isfinite(probabilities))
        assert np.all(probabilities >= 0.0)
        assert math.fsum(
            float(value) for value in probabilities
        ) == pytest.approx(
            1.0,
            abs=1e-12,
        )
        assert int(np.argmax(probabilities)) == int(np.argmax(logits))


def test_entropy_effective_support_and_visible_mass_are_consistent() -> None:
    probabilities = np.full(4, 0.25, dtype=np.float64)

    indices, values, other, entropy, effective, normalized = (
        summarize_probabilities(probabilities, 2)
    )

    assert entropy_bits_from_probabilities(probabilities) == pytest.approx(2.0)
    assert entropy == pytest.approx(2.0)
    assert effective == pytest.approx(4.0)
    assert normalized == pytest.approx(1.0)
    assert len(indices) == len(values) == 2
    assert float(np.sum(values)) + other == pytest.approx(1.0)


def test_js_and_total_variation_obey_metric_contracts() -> None:
    left = np.array([0.75, 0.25, 0.0], dtype=np.float64)
    right = np.array([0.0, 0.25, 0.75], dtype=np.float64)

    forward = distribution_distance(left, right)
    reverse = distribution_distance(right, left)
    identity = distribution_distance(left, left)
    disjoint = distribution_distance(
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    )

    assert forward == pytest.approx(reverse)
    assert identity == pytest.approx((0.0, 0.0))
    assert 0.0 <= forward[0] <= 1.0
    assert 0.0 <= forward[1] <= 1.0
    assert disjoint == pytest.approx((1.0, 1.0))


def test_tokenization_roundtrips_exact_text_and_exposes_unicode_divergence() -> None:
    state = ModelState(Path("."), mock=False)
    state.status = "ready"
    state.active_model_id = "bf16"
    state.model_generation = 7
    state.tokenizer = ExactCharacterTokenizer()

    result = state.tokenization(" Mám\tPrahu.\n", " Ma\u0301m\tPrahu.\n")

    assert result["execution"]["forward_passes"] == 0
    assert result["a"]["roundtrip_exact"] is True
    assert result["b"]["roundtrip_exact"] is True
    assert result["a"]["detokenized_text"] == " Mám\tPrahu.\n"
    assert result["a"]["offset_coverage"]["ratio"] == pytest.approx(1.0)
    assert result["comparison"]["canonically_equivalent_nfc"] is True
    assert result["comparison"]["identical_token_ids"] is False


def test_mock_next_token_response_is_deterministic_for_the_same_input() -> None:
    state = ready_mock_state()
    identity = {
        "expected_generation": state.model_generation,
        "expected_server_instance_id": state.server_instance_id,
    }

    first = state.next_tokens("Stejný prefix", 1.0, 5, **identity)
    second = state.next_tokens("Stejný prefix", 1.0, 5, **identity)

    assert first["candidates"] == second["candidates"]
    assert first["raw_metrics"] == second["raw_metrics"]
    assert first["entropy_bits"] == second["entropy_bits"]
    assert first["top_k_mass"] + first["other_probability"] == pytest.approx(
        1.0
    )


def test_stale_model_identity_is_rejected_before_inference() -> None:
    state = ready_mock_state()

    with pytest.raises(ModelChangedError):
        state.next_tokens(
            "Prefix",
            1.0,
            5,
            expected_generation=state.model_generation - 1,
            expected_server_instance_id=state.server_instance_id,
        )


def test_alpha_zero_is_an_exact_steering_identity() -> None:
    state = ready_mock_state()

    result = state.steering(
        steering_pairs(),
        "Nový text v nezávislém rámci",
        16,
        0.0,
        1.0,
        5,
        6,
        10,
        "mock",
        state.model_generation,
        state.server_instance_id,
    )

    assert result["js_divergence_bits"] == 0.0
    assert result["total_variation"] == 0.0
    assert result["delta_entropy_bits"] == 0.0
    assert result["left"]["top_tokens"] == result["right"]["top_tokens"]
    assert result["steering"]["effective_shift_norm"] == 0.0


def test_component_accounting_closes_to_the_final_logit_contrast() -> None:
    rng = np.random.default_rng(20260806)
    embedding = rng.normal(size=12)
    attention = rng.normal(scale=0.2, size=(4, 12))
    mlp = rng.normal(scale=0.3, size=(4, 12))
    final_residual = embedding + attention.sum(axis=0) + mlp.sum(axis=0)

    result = decompose_residual_contrast(
        embedding=embedding,
        attention_updates=attention,
        mlp_updates=mlp,
        final_residual=final_residual,
        norm_weight=rng.normal(1.0, 0.05, size=12),
        lm_head_row_a=rng.normal(size=12),
        lm_head_row_b=rng.normal(size=12),
        epsilon=1e-5,
    )

    assert result["final"]["reconstruction_error"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert result["final"]["reconstructed_contrast"] == pytest.approx(
        result["final"]["contrast"],
        abs=1e-12,
    )


def test_archive_can_export_exactly_one_selected_record(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    selected = experiment_record()
    other = experiment_record("other_contract_snapshot")
    store.save(selected)
    store.save(other)

    exported = json.loads(store.export_json(record_id=selected["id"]))
    rows = list(
        csv.DictReader(
            io.StringIO(
                store.export_csv(record_id=selected["id"]).decode("utf-8")
            )
        )
    )

    assert exported["record_count"] == 1
    assert exported["record_id"] == selected["id"]
    assert [record["id"] for record in exported["records"]] == [selected["id"]]
    assert rows
    assert {row["record_id"] for row in rows} == {selected["id"]}


def test_http_validators_preserve_text_and_require_model_identity() -> None:
    identity = {
        "model_id": "mock",
        "model_generation": 7,
        "server_instance_id": "server",
    }
    continuation_request = {
        "prefix": " Prefix ",
        "continuation": " pokračování ",
        "top_k": 5,
        **identity,
    }
    valid_contracts = (
        (
            api_contracts.tokenization_request_fields,
            GardenHandler._tokenization_request_fields,
            {"text_a": "  Praha\t", "text_b": None},
        ),
        (
            api_contracts.embeddings_request_fields,
            GardenHandler._embeddings_request_fields,
            {
                "contexts": [
                    {
                        "text": "Praha",
                        "label": "Město",
                        "token_index": 0,
                    }
                ],
                "neighbor_count": 12,
                "analysis_mode": "token",
                **identity,
            },
        ),
        (
            api_contracts.embedding_neighborhood_request_fields,
            GardenHandler._embedding_neighborhood_request_fields,
            {"token_id": 48_053, "neighbor_count": 6, **identity},
        ),
        (
            api_contracts.text_surprisal_request_fields,
            GardenHandler._text_surprisal_request_fields,
            {"text": "Text.", "top_k": 5, **identity},
        ),
        (
            api_contracts.continuation_surprisal_request_fields,
            GardenHandler._continuation_surprisal_request_fields,
            continuation_request,
        ),
        (
            api_contracts.text_surprisal_comparison_request_fields,
            GardenHandler._text_surprisal_comparison_request_fields,
            {
                "text_a": "Text A.",
                "text_b": "Text B.",
                "label_a": "A",
                "label_b": "B",
                "top_k": 5,
                **identity,
            },
        ),
        (
            api_contracts.context_contrast_request_fields,
            GardenHandler._context_contrast_request_fields,
            {
                "prefix_a": "Kontext A",
                "prefix_b": "Kontext B",
                "continuation": " pokračuje.",
                "top_k": 5,
                **identity,
            },
        ),
    )

    for direct_validator, handler_validator, request in valid_contracts:
        assert handler_validator(request) == direct_validator(request)

    assert api_contracts.tokenization_request_fields(
        {"text_a": "  Praha\t", "text_b": None}
    ) == ("  Praha\t", None)
    assert api_contracts.continuation_surprisal_request_fields(
        continuation_request
    ) == (
        " Prefix ",
        " pokračování ",
        5,
        "mock",
        7,
        "server",
    )

    for missing in (
        "model_id",
        "model_generation",
        "server_instance_id",
    ):
        invalid = dict(continuation_request)
        invalid.pop(missing)
        with pytest.raises(ValueError) as direct_error:
            api_contracts.continuation_surprisal_request_fields(invalid)
        with pytest.raises(ValueError) as handler_error:
            GardenHandler._continuation_surprisal_request_fields(invalid)
        assert str(handler_error.value) == str(direct_error.value)

    for constant_name in (
        name for name in api_contracts.__all__ if name.isupper()
    ):
        assert getattr(server_module, constant_name) == getattr(
            api_contracts,
            constant_name,
        )
