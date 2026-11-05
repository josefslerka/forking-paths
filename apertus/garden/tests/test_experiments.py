from __future__ import annotations

import copy
import csv
import io
import json
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path

from apertus.garden.experiments import (
    EXPORT_FORMAT,
    RECORD_SCHEMA,
    ConflictError,
    ExperimentStore,
    SchemaVersionError,
    ValidationError,
)


CAPTURED_AT = "2026-07-28T10:11:12.000Z"


def record(
    *,
    record_id: str | None = None,
    kind: str = "distribution_comparison",
    payload: dict | None = None,
) -> dict:
    return {
        "schema": RECORD_SCHEMA,
        "schema_version": 1,
        "id": record_id or str(uuid.uuid4()),
        "kind": kind,
        "captured_at": CAPTURED_AT,
        "annotation": {"title": "Pilot", "notes": "", "tags": ["test"]},
        "capture": {
            "status": "complete",
        },
        "provenance": {
            "model": {
                "model_id": "mock",
                "model_generation": 2,
                "server_instance_id": "server-1",
            },
        },
        "payload": payload
        or {
            "snapshot": {
                "left": {"prefix": "A"},
                "right": {"prefix": "B"},
                "temperature": 1.0,
            },
            "response": {
                "left": {"entropy_bits": 1.0, "effective_tokens": 2.0},
                "right": {"entropy_bits": 2.0, "effective_tokens": 4.0},
                "delta_entropy_bits": 1.0,
                "effective_support_ratio": 2.0,
                "js_divergence_bits": 0.1,
                "total_variation": 0.2,
                "top_k_overlap": {"k": 10, "count": 7, "fraction": 0.7},
                "gains": [
                    {
                        "token_id": 42,
                        "display": "=SUM(A1:A2)",
                        "is_special": False,
                        "probability_left": 0.1,
                        "probability_right": 0.3,
                        "delta": 0.2,
                        "rank_left": 5,
                        "rank_right": 1,
                    }
                ],
                "losses": [],
            }
        },
        "diagnostics": {"saved_after_context_changed": False},
    }


class ExperimentStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "experiments.sqlite3"
        self.store = ExperimentStore(self.database)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_initializes_secure_status_and_sqlite_pragmas(self) -> None:
        status = self.store.status()
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["path_label"], "experiments.sqlite3")
        self.assertNotIn(str(self.database.parent), json.dumps(status))
        self.assertEqual(status["schema_version"], 1)
        self.assertEqual(status["count"], 0)
        self.assertEqual(status["journal_mode"], "wal")
        self.assertRegex(
            status["archive_id"],
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        )
        self.assertEqual(status["archive_id"], self.store.status()["archive_id"])
        second_store = ExperimentStore(
            Path(self.temporary.name) / "replacement.sqlite3"
        )
        self.assertNotEqual(status["archive_id"], second_store.status()["archive_id"])
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")

    def test_save_is_idempotent_but_uuid_content_is_immutable(self) -> None:
        original = record()
        stored, created = self.store.save(original)
        self.assertTrue(created)
        self.assertIn("saved_at", stored)
        self.assertNotIn("saved_at", original)

        reordered = json.loads(json.dumps(original, sort_keys=True))
        same, created = self.store.save(reordered)
        self.assertFalse(created)
        self.assertEqual(same, stored)

        changed = copy.deepcopy(original)
        changed["annotation"]["notes"] = "different"
        with self.assertRaises(ConflictError) as caught:
            self.store.save(changed)
        self.assertEqual(caught.exception.record_id, original["id"])

    def test_direct_save_validates_canonical_envelope(self) -> None:
        invalid = record()
        invalid["saved_at"] = "2020-01-01T00:00:00Z"
        with self.assertRaises(ValidationError):
            self.store.save(invalid)
        invalid = record(kind="contains spaces")
        with self.assertRaises(ValidationError):
            self.store.save(invalid)
        invalid = record()
        invalid["annotation"] = None
        with self.assertRaises(ValidationError):
            self.store.save(invalid)

    def test_list_filter_pagination_get_and_delete(self) -> None:
        comparison = record()
        garden = record(kind="garden_snapshot")
        self.store.save(comparison)
        self.store.save(garden)

        page = self.store.list_summaries(limit=1, offset=1)
        self.assertEqual(page["total"], 2)
        self.assertEqual(len(page["records"]), 1)
        self.assertNotIn("payload", page["records"][0])
        filtered = self.store.list_summaries("garden_snapshot")
        self.assertEqual(filtered["total"], 1)
        self.assertEqual(filtered["records"][0]["id"], garden["id"])
        self.assertEqual(self.store.get(comparison["id"])["payload"], comparison["payload"])

        self.assertTrue(self.store.delete(comparison["id"]))
        self.assertFalse(self.store.delete(comparison["id"]))
        self.assertIsNone(self.store.get(comparison["id"]))

    def test_import_reports_each_record_and_export_wrapper_preserves_saved_at(self) -> None:
        first = record()
        stored, _ = self.store.save(first)
        exported = json.loads(self.store.export_json())
        self.assertEqual(exported["format"], EXPORT_FORMAT)
        self.assertEqual(exported["records"][0], stored)

        second_database = Path(self.temporary.name) / "restored.sqlite3"
        restored = ExperimentStore(second_database)
        summary = restored.import_payload(exported)
        self.assertEqual(summary["created"], 1)
        self.assertEqual(summary["imported"], 1)
        self.assertEqual(summary["rejected"], 0)
        self.assertEqual(restored.get(first["id"])["saved_at"], stored["saved_at"])

        conflicting = copy.deepcopy(first)
        conflicting["payload"]["response"]["total_variation"] = 0.9
        invalid = {"id": "not-a-uuid"}
        mixed = restored.import_payload([first, conflicting, invalid])
        self.assertEqual(mixed["existing"], 1)
        self.assertEqual(mixed["conflicts"], 1)
        self.assertEqual(mixed["rejected"], 1)
        self.assertEqual([item["status"] for item in mixed["results"]], [
            "existing",
            "conflict",
            "rejected",
        ])

    def test_import_accepts_legacy_record_discriminator_and_annotation(self) -> None:
        legacy = record()
        legacy["format"] = legacy.pop("schema")
        legacy["annotation"] = "legacy note"
        summary = self.store.import_payload(legacy)
        self.assertEqual(summary["created"], 1)
        restored = self.store.get(legacy["id"])
        self.assertEqual(restored["schema"], RECORD_SCHEMA)
        self.assertEqual(restored["annotation"]["notes"], "legacy note")

    def test_json_export_is_lossless_and_kind_filtered(self) -> None:
        comparison = record()
        self.store.save(comparison)
        self.store.save(record(kind="garden_snapshot"))
        wrapper = json.loads(self.store.export_json("distribution_comparison"))
        self.assertEqual(wrapper["record_count"], 1)
        self.assertEqual(wrapper["kind"], "distribution_comparison")
        self.assertEqual(wrapper["records"][0], self.store.get(comparison["id"]))

    def test_json_and_csv_export_can_target_exactly_one_record(self) -> None:
        selected = record()
        other = record(kind="garden_snapshot")
        self.store.save(selected)
        self.store.save(other)

        wrapper = json.loads(
            self.store.export_json(record_id=selected["id"])
        )
        self.assertEqual(wrapper["record_count"], 1)
        self.assertEqual(wrapper["record_id"], selected["id"])
        self.assertEqual(
            [item["id"] for item in wrapper["records"]],
            [selected["id"]],
        )

        rows = list(
            csv.DictReader(
                io.StringIO(
                    self.store.export_csv(
                        record_id=selected["id"]
                    ).decode("utf-8")
                )
            )
        )
        self.assertTrue(rows)
        self.assertEqual(
            {row["record_id"] for row in rows},
            {selected["id"]},
        )
        with self.assertRaises(ValidationError):
            self.store.export_json(
                "distribution_comparison",
                record_id=selected["id"],
            )

    def test_csv_flattens_comparison_garden_and_lens_and_defuses_formulas(self) -> None:
        comparison = record()
        garden = record(
            kind="garden_snapshot",
            payload={
                "controls": {"temperature": 0.75, "top_k": 12},
                "columns": [
                    {
                        "session_column_id": 1,
                        "prompt": "@danger",
                        "nodes": [
                            {
                                "session_node_id": 1,
                                "session_parent_id": None,
                                "prefix": "+prefix",
                                "incoming": {
                                    "token_id": 7,
                                    "probability": 0.25,
                                },
                                "distribution": {
                                    "entropy_bits": 2.5,
                                    "candidates": [
                                        {
                                            "token_id": 8,
                                            "display": "-candidate",
                                            "probability": 0.4,
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ]
            },
        )
        lens = record(
            kind="lens_trajectory",
            payload={
                "request": {"temperature": 1.0},
                "result": {
                    "model_id": "mock",
                    "model_generation": 3,
                    "server_instance_id": "server-2",
                    "layers": [
                        {
                            "layer": 1,
                            "entropy_bits": 3.0,
                            "candidates": [
                                {
                                    "token_id": 9,
                                    "display": "\tformula",
                                    "probability": 0.6,
                                }
                            ],
                        }
                    ],
                },
            },
        )
        for item in (comparison, garden, lens):
            self.store.save(item)

        rows = list(
            csv.DictReader(io.StringIO(self.store.export_csv().decode("utf-8")))
        )
        scopes = {row["export_scope"] for row in rows}
        self.assertIn("displayed_movers", scopes)
        self.assertIn("captured_nodes", scopes)
        self.assertIn("captured_layers", scopes)
        self.assertIn("displayed_candidates", scopes)
        comparison_row = next(row for row in rows if row["mover_direction"])
        self.assertEqual(comparison_row["token"], "'=SUM(A1:A2)")
        self.assertEqual(comparison_row["a_prefix"], "A")
        self.assertEqual(comparison_row["b_prefix"], "B")
        self.assertEqual(comparison_row["model_id"], "mock")
        garden_row = next(row for row in rows if row["candidate_token_id"] == "8")
        self.assertEqual(garden_row["candidate_token"], "'-candidate")
        self.assertEqual(garden_row["node_id"], "1")
        self.assertEqual(garden_row["incoming_token_id"], "7")
        self.assertEqual(garden_row["temperature"], "0.75")
        lens_row = next(row for row in rows if row["lens_candidate_token_id"] == "9")
        self.assertEqual(lens_row["lens_candidate_token"], "'\tformula")
        self.assertEqual(lens_row["layer"], "1")

    def test_csv_flattens_raw_vs_tuned_lens_without_breaking_json(self) -> None:
        tuned = record(
            kind="lens_trajectory",
            payload={
                "reader": {
                    "kind": "raw_vs_tuned_lens",
                    "artifact_id": "pilot-64",
                },
                "result": {
                    "calibration": {
                        "artifact_id": "pilot-64",
                        "canonical_manifest_sha256": "a" * 64,
                        "weights": {"sha256": "b" * 64},
                        "metrics": {"sha256": "c" * 64},
                        "validation": {"status": "pilot"},
                    },
                    "layers": [
                        {
                            "layer": 16,
                            "raw": {
                                "kl_divergence_to_final_bits": 0.4,
                                "js_divergence_to_final_bits": 0.1,
                                "candidates": [
                                    {
                                        "token_id": 10,
                                        "display": "RAW",
                                        "probability": 0.2,
                                    }
                                ],
                            },
                            "tuned": {
                                "kl_divergence_to_final_bits": 0.2,
                                "js_divergence_to_final_bits": 0.05,
                                "candidates": [
                                    {
                                        "token_id": 11,
                                        "display": "TUNED",
                                        "probability": 0.3,
                                    }
                                ],
                            },
                            "comparison": {
                                "kl_improvement_bits": 0.2,
                            },
                        }
                    ],
                },
            },
        )
        self.store.save(tuned)
        exported_json = json.loads(self.store.export_json())
        self.assertEqual(
            exported_json["records"][0]["payload"],
            tuned["payload"],
        )
        rows = list(
            csv.DictReader(
                io.StringIO(self.store.export_csv().decode("utf-8"))
            )
        )
        candidate_rows = [
            row
            for row in rows
            if row["export_scope"] == "displayed_candidates"
        ]
        self.assertEqual(
            {row["lens_variant"] for row in candidate_rows},
            {"raw", "tuned"},
        )
        tuned_row = next(
            row for row in candidate_rows if row["lens_variant"] == "tuned"
        )
        self.assertEqual(tuned_row["artifact_id"], "pilot-64")
        self.assertEqual(
            tuned_row["artifact_manifest_sha256"],
            "a" * 64,
        )
        self.assertEqual(
            tuned_row["artifact_weights_sha256"],
            "b" * 64,
        )
        self.assertEqual(
            tuned_row["artifact_metrics_sha256"],
            "c" * 64,
        )
        self.assertEqual(
            tuned_row["kl_divergence_to_final_bits"],
            "0.2",
        )
        self.assertEqual(tuned_row["lens_candidate_token"], "TUNED")

    def test_csv_flattens_raw_vs_jacobian_lens_with_artifact_provenance(
        self,
    ) -> None:
        jacobian = record(
            kind="lens_trajectory",
            payload={
                "reader": {
                    "kind": "raw_vs_jacobian_lens",
                    "artifact_id": "jlens-bf16-v1",
                },
                "result": {
                    "artifact": {
                        "format": "apertus.jacobian-lens-artifact",
                        "schema_version": 1,
                        "artifact_id": "jlens-bf16-v1",
                        "artifact_scope": "cs_en_balanced_wikipedia_pilot",
                        "canonical_manifest_sha256": "a" * 64,
                        "reader": {
                            "source_layers": [4, 16],
                            "target_layer": 32,
                        },
                        "model": {"mode": "bf16"},
                        "weights": {"sha256": "b" * 64},
                        "metrics": {"sha256": "c" * 64},
                        "compatibility": {
                            "allowed_model_modes": ["bf16"],
                        },
                    },
                    "transfer_probe": {
                        "status": "passed",
                        "prompt_count": 3,
                    },
                    "layers": [
                        {
                            "layer": 16,
                            "raw": {
                                "kl_divergence_to_final_bits": 0.4,
                                "js_divergence_to_final_bits": 0.1,
                                "candidates": [
                                    {
                                        "token_id": 10,
                                        "display": "RAW",
                                        "probability": 0.2,
                                    }
                                ],
                            },
                            "jacobian": {
                                "kl_divergence_to_final_bits": 0.9,
                                "js_divergence_to_final_bits": 0.3,
                                "candidates": [
                                    {
                                        "token_id": 12,
                                        "display": "J-LENS",
                                        "probability": 0.15,
                                    }
                                ],
                            },
                            "comparison": {
                                "kl_jacobian_minus_raw_bits": 0.5,
                            },
                        }
                    ],
                },
            },
        )
        self.store.save(jacobian)
        exported_json = json.loads(self.store.export_json())
        self.assertEqual(
            exported_json["records"][0]["payload"],
            jacobian["payload"],
        )
        rows = list(
            csv.DictReader(
                io.StringIO(
                    self.store.export_csv("lens_trajectory").decode("utf-8")
                )
            )
        )
        candidate_rows = [
            row
            for row in rows
            if row["export_scope"] == "displayed_candidates"
        ]
        self.assertEqual(
            {row["lens_variant"] for row in candidate_rows},
            {"raw", "jacobian"},
        )
        jlens_row = next(
            row
            for row in candidate_rows
            if row["lens_variant"] == "jacobian"
        )
        self.assertEqual(jlens_row["reader_kind"], "raw_vs_jacobian_lens")
        self.assertEqual(jlens_row["artifact_id"], "jlens-bf16-v1")
        self.assertEqual(
            jlens_row["artifact_manifest_sha256"],
            "a" * 64,
        )
        self.assertEqual(
            jlens_row["artifact_weights_sha256"],
            "b" * 64,
        )
        self.assertEqual(
            jlens_row["artifact_metrics_sha256"],
            "c" * 64,
        )
        self.assertEqual(
            jlens_row["artifact_format"],
            "apertus.jacobian-lens-artifact",
        )
        self.assertEqual(jlens_row["artifact_model_mode"], "bf16")
        self.assertEqual(jlens_row["artifact_source_layers_json"], "[4,16]")
        self.assertEqual(jlens_row["artifact_target_layer"], "32")
        self.assertEqual(jlens_row["transfer_probe_status"], "passed")
        self.assertEqual(
            json.loads(jlens_row["transfer_probe_json"]),
            {"prompt_count": 3, "status": "passed"},
        )
        self.assertEqual(jlens_row["lens_candidate_token"], "J-LENS")

    def test_text_surprisal_round_trips_and_csv_flattens_safe_rows(self) -> None:
        captured = record(
            kind="text_surprisal",
            payload={
                "request": {
                    "text": "=2+2  ",
                    "top_k": 2,
                    "model_id": "mock",
                    "model_generation": 2,
                    "server_instance_id": "server-1",
                },
                "response": {
                    "analysis_kind": "text_surprisal",
                    "original_text": "=2+2  ",
                    "normalized_text": "=2+2",
                    "text": "=2+2",
                    "normalization": {
                        "policy": "trim_trailing_whitespace_only",
                        "changed": True,
                        "trimmed_trailing_characters": 2,
                        "leading_whitespace_preserved": True,
                    },
                    "top_k": 2,
                    "temperature": 1.0,
                    "probability_kind": "raw_model",
                    "token_count": 1,
                    "character_count": 5,
                    "utf8_byte_count": 5,
                    "aggregation_method": {
                        "id": "descriptive_text_span_aggregation",
                        "version": "1.0.0",
                        "status": "heuristic",
                        "summary_population": "complete_spans_only",
                        "quantile_method": "linear_interpolation",
                        "mad_definition": "raw_median_absolute_deviation",
                    },
                    "aggregations": {
                        "whitespace_span": {
                            "unit": "whitespace_span",
                            "label_cs": "Úseky podle bílých znaků",
                            "spans": [
                                {
                                    "index": 0,
                                    "text": "=2+2",
                                    "character_start": 0,
                                    "character_end": 4,
                                    "token_indices": [0],
                                    "token_ids": [17],
                                    "token_count": 1,
                                    "total_surprisal_bits": 3.0,
                                    "surprisal_bits_per_token": 3.0,
                                    "surprisal_bits_per_character": 0.75,
                                    "character_count": 4,
                                    "non_whitespace_character_count": 4,
                                    "covered_non_whitespace_character_count": 4,
                                    "character_coverage_ratio": 1.0,
                                    "complete": True,
                                    "ambiguous_token_indices": [],
                                }
                            ],
                            "summary": {
                                "count": 1,
                                "mean": 0.75,
                                "median": 0.75,
                                "q1": 0.75,
                                "q3": 0.75,
                                "iqr": 0.0,
                                "raw_mad": 0.0,
                                "min": 0.75,
                                "max": 0.75,
                                "assigned_token_count": 1,
                                "ambiguous_token_count": 0,
                                "unassigned_token_count": 0,
                                "coverage_ratio": 1.0,
                                "span_count": 1,
                                "complete_span_count": 1,
                                "incomplete_span_count": 0,
                            },
                        },
                        "line": {
                            "unit": "line",
                            "label_cs": "Řádky",
                            "spans": [],
                            "summary": {
                                "count": 0,
                                "mean": None,
                                "median": None,
                                "q1": None,
                                "q3": None,
                                "iqr": None,
                                "raw_mad": None,
                                "min": None,
                                "max": None,
                                "assigned_token_count": 0,
                                "ambiguous_token_count": 1,
                                "unassigned_token_count": 0,
                                "coverage_ratio": 0.0,
                                "span_count": 0,
                                "complete_span_count": 0,
                                "incomplete_span_count": 0,
                            },
                        },
                        "sentence_like": {
                            "unit": "sentence_like",
                            "label_cs": "Větám podobné úseky",
                            "spans": [
                                {
                                    "index": 0,
                                    "text": "+formula",
                                    "character_start": 0,
                                    "character_end": 4,
                                    "token_indices": [0],
                                    "token_ids": [17],
                                    "token_count": 1,
                                    "total_surprisal_bits": 3.0,
                                    "surprisal_bits_per_token": 3.0,
                                    "surprisal_bits_per_character": 0.75,
                                    "character_count": 4,
                                    "non_whitespace_character_count": 4,
                                    "covered_non_whitespace_character_count": 4,
                                    "character_coverage_ratio": 1.0,
                                    "complete": False,
                                    "ambiguous_token_indices": [0],
                                }
                            ],
                            "summary": {
                                "count": 0,
                                "mean": None,
                                "median": None,
                                "q1": None,
                                "q3": None,
                                "iqr": None,
                                "raw_mad": None,
                                "min": None,
                                "max": None,
                                "assigned_token_count": 0,
                                "ambiguous_token_count": 1,
                                "unassigned_token_count": 0,
                                "coverage_ratio": 0.0,
                                "span_count": 1,
                                "complete_span_count": 0,
                                "incomplete_span_count": 1,
                            },
                        },
                    },
                    "tokens": [
                        {
                            "index": 0,
                            "token_id": 17,
                            "text": "+observed",
                            "piece": "+observed",
                            "display": "\tobserved",
                            "tokenizer_token": "-tokenizer",
                            "character_start": 0,
                            "character_end": 5,
                            "probability": 0.125,
                            "logprob": -2.0794415416798357,
                            "surprisal_bits": 3.0,
                            "rank": 4,
                            "entropy_bits": 2.5,
                            "effective_tokens": 5.656854249,
                            "top1_gap_bits": 1.25,
                            "word_start_hint": True,
                            "alternatives": [
                                {
                                    "token_id": 1,
                                    "text": "@alternative",
                                    "display": "=ALT()",
                                    "is_special": False,
                                    "probability": 0.5,
                                    "rank": 1,
                                },
                                {
                                    "token_id": 2,
                                    "text": "-second",
                                    "display": "+SECOND",
                                    "is_special": True,
                                    "probability": 0.25,
                                    "rank": 2,
                                },
                            ],
                        }
                    ],
                    "summary": {
                        "total_surprisal_bits": 3.0,
                        "mean_surprisal_bits": 3.0,
                        "perplexity": 8.0,
                        "bits_per_character": 0.6,
                        "bits_per_utf8_byte": 0.6,
                        "mean_entropy_bits": 2.5,
                        "max_surprisal_token": {
                            "index": 0,
                            "token_id": 17,
                            "display": "\tobserved",
                            "surprisal_bits": 3.0,
                        },
                    },
                    "transition_id": 9,
                    "execution": {
                        "mode": "teacher_forced",
                        "forward_passes": 1,
                    },
                    "timing_ms": {"total": 12.5},
                    "latency_ms": 12.5,
                },
            },
        )
        stored, created = self.store.save(captured)
        self.assertTrue(created)
        self.assertEqual(
            self.store.list_summaries("text_surprisal")["records"][0]["id"],
            captured["id"],
        )

        exported = json.loads(self.store.export_json("text_surprisal"))
        self.assertEqual(exported["records"], [stored])
        restored = ExperimentStore(
            Path(self.temporary.name) / "text-surprisal-restored.sqlite3"
        )
        imported = restored.import_payload(exported)
        self.assertEqual(imported["created"], 1)
        self.assertEqual(restored.get(captured["id"]), stored)

        rows = list(
            csv.DictReader(
                io.StringIO(
                    self.store.export_csv("text_surprisal").decode("utf-8")
                )
            )
        )
        self.assertEqual(
            [row["export_scope"] for row in rows],
            [
                "text_surprisal_summary",
                "text_surprisal_aggregation_summary",
                "text_surprisal_aggregation_spans",
                "text_surprisal_aggregation_summary",
                "text_surprisal_aggregation_summary",
                "text_surprisal_aggregation_spans",
                "text_surprisal_tokens",
                "text_surprisal_alternatives",
                "text_surprisal_alternatives",
            ],
        )
        (
            summary_row,
            whitespace_summary_row,
            whitespace_span_row,
            line_summary_row,
            sentence_summary_row,
            sentence_span_row,
            token_row,
            first_alternative,
            second_alternative,
        ) = rows
        self.assertEqual(summary_row["analysis_kind"], "text_surprisal")
        self.assertEqual(summary_row["original_text"], "'=2+2  ")
        self.assertEqual(summary_row["normalized_text"], "'=2+2")
        self.assertEqual(summary_row["text"], "'=2+2")
        self.assertEqual(
            json.loads(summary_row["normalization"]),
            {
                "changed": True,
                "leading_whitespace_preserved": True,
                "policy": "trim_trailing_whitespace_only",
                "trimmed_trailing_characters": 2,
            },
        )
        self.assertEqual(summary_row["token_count"], "1")
        self.assertEqual(summary_row["total_surprisal_bits"], "3.0")
        self.assertEqual(summary_row["perplexity"], "8.0")
        self.assertEqual(
            json.loads(summary_row["aggregation_method_json"])["status"],
            "heuristic",
        )
        self.assertEqual(
            set(json.loads(summary_row["aggregations_json"])),
            {"whitespace_span", "line", "sentence_like"},
        )

        self.assertEqual(
            whitespace_summary_row["aggregation_type"],
            "whitespace_span",
        )
        self.assertEqual(
            whitespace_summary_row["aggregation_label"],
            "Úseky podle bílých znaků",
        )
        self.assertEqual(
            whitespace_summary_row["aggregation_span_count"],
            "1",
        )
        self.assertEqual(
            whitespace_summary_row["aggregation_complete_span_count"],
            "1",
        )
        self.assertEqual(
            whitespace_summary_row["aggregation_coverage_ratio"],
            "1.0",
        )
        self.assertEqual(
            whitespace_summary_row["aggregation_total_token_count"],
            "1",
        )
        self.assertEqual(whitespace_summary_row["aggregation_mean"], "0.75")
        self.assertEqual(whitespace_summary_row["aggregation_median"], "0.75")
        self.assertEqual(whitespace_summary_row["aggregation_q1"], "0.75")
        self.assertEqual(whitespace_summary_row["aggregation_q3"], "0.75")
        self.assertEqual(whitespace_summary_row["aggregation_iqr"], "0.0")
        self.assertEqual(whitespace_summary_row["aggregation_raw_mad"], "0.0")
        self.assertEqual(whitespace_summary_row["aggregation_min"], "0.75")
        self.assertEqual(whitespace_summary_row["aggregation_max"], "0.75")

        self.assertEqual(
            whitespace_span_row["aggregation_span_index"],
            "0",
        )
        self.assertEqual(whitespace_span_row["aggregation_character_start"], "0")
        self.assertEqual(whitespace_span_row["aggregation_character_end"], "4")
        self.assertEqual(whitespace_span_row["aggregation_text"], "'=2+2")
        self.assertEqual(
            json.loads(whitespace_span_row["aggregation_token_indices_json"]),
            [0],
        )
        self.assertEqual(
            json.loads(whitespace_span_row["aggregation_token_ids_json"]),
            [17],
        )
        self.assertEqual(whitespace_span_row["aggregation_token_count"], "1")
        self.assertEqual(
            whitespace_span_row["aggregation_total_surprisal_bits"],
            "3.0",
        )
        self.assertEqual(
            whitespace_span_row["aggregation_surprisal_bits_per_token"],
            "3.0",
        )
        self.assertEqual(
            whitespace_span_row[
                "aggregation_surprisal_bits_per_non_whitespace_character"
            ],
            "0.75",
        )
        self.assertEqual(
            whitespace_span_row["aggregation_character_coverage_ratio"],
            "1.0",
        )
        self.assertEqual(whitespace_span_row["aggregation_complete"], "true")
        self.assertEqual(
            whitespace_span_row["aggregation_span_ambiguous_token_count"],
            "0",
        )
        self.assertEqual(
            json.loads(
                whitespace_span_row[
                    "aggregation_ambiguous_token_indices_json"
                ]
            ),
            [],
        )
        self.assertEqual(line_summary_row["aggregation_type"], "line")
        self.assertEqual(line_summary_row["aggregation_mean"], "")
        self.assertEqual(
            sentence_summary_row["aggregation_type"],
            "sentence_like",
        )
        self.assertEqual(sentence_span_row["aggregation_text"], "'+formula")
        self.assertEqual(sentence_span_row["aggregation_complete"], "false")
        self.assertEqual(
            sentence_span_row["aggregation_span_ambiguous_token_count"],
            "1",
        )

        self.assertEqual(token_row["text_token_index"], "0")
        self.assertEqual(token_row["text_token_id"], "17")
        self.assertEqual(token_row["text_token"], "'+observed")
        self.assertEqual(token_row["text_token_display"], "'\tobserved")
        self.assertEqual(token_row["tokenizer_token"], "'-tokenizer")
        self.assertEqual(token_row["token_character_start"], "0")
        self.assertEqual(token_row["token_character_end"], "5")
        self.assertEqual(token_row["token_surprisal_bits"], "3.0")
        self.assertEqual(token_row["token_word_start_hint"], "true")

        self.assertEqual(first_alternative["alternative_token"], "'@alternative")
        self.assertEqual(first_alternative["alternative_display"], "'=ALT()")
        self.assertEqual(first_alternative["alternative_is_special"], "false")
        self.assertEqual(second_alternative["alternative_token"], "'-second")
        self.assertEqual(second_alternative["alternative_display"], "'+SECOND")
        self.assertEqual(second_alternative["alternative_is_special"], "true")

    def test_text_surprisal_comparison_is_lossless_and_flattens_paired_csv(
        self,
    ) -> None:
        def comparison_side(
            text: str,
            token_id: int,
            mean_bits: float,
        ) -> dict:
            aggregation_types = (
                "whitespace_span",
                "line",
                "sentence_like",
            )
            aggregations = {}
            for aggregation_type in aggregation_types:
                aggregations[aggregation_type] = {
                    "unit": aggregation_type,
                    "label_cs": f"={aggregation_type}",
                    "spans": [
                        {
                            "index": 0,
                            "text": text,
                            "character_start": 0,
                            "character_end": len(text),
                            "token_indices": [0],
                            "token_ids": [token_id],
                            "token_count": 1,
                            "total_surprisal_bits": mean_bits,
                            "surprisal_bits_per_token": mean_bits,
                            "surprisal_bits_per_character": (
                                mean_bits / len(text)
                            ),
                            "character_count": len(text),
                            "non_whitespace_character_count": len(text),
                            "covered_non_whitespace_character_count": len(text),
                            "character_coverage_ratio": 1.0,
                            "complete": True,
                            "ambiguous_token_indices": [],
                        }
                    ],
                    "summary": {
                        "count": 1,
                        "mean": mean_bits / len(text),
                        "median": mean_bits / len(text),
                        "q1": mean_bits / len(text),
                        "q3": mean_bits / len(text),
                        "iqr": 0.0,
                        "raw_mad": 0.0,
                        "min": mean_bits / len(text),
                        "max": mean_bits / len(text),
                        "assigned_token_count": 1,
                        "ambiguous_token_count": 0,
                        "unassigned_token_count": 0,
                        "coverage_ratio": 1.0,
                        "span_count": 1,
                        "complete_span_count": 1,
                        "incomplete_span_count": 0,
                    },
                }
            return {
                "analysis_kind": "text_surprisal",
                "original_text": text,
                "normalized_text": text,
                "text": text,
                "normalization": {"policy": "none"},
                "top_k": 1,
                "temperature": 1.0,
                "probability_kind": "raw_model",
                "token_count": 1,
                "character_count": len(text),
                "utf8_byte_count": len(text.encode("utf-8")),
                "aggregation_method": {
                    "id": "descriptive_text_span_aggregation",
                    "version": "1.0.0",
                    "status": "heuristic",
                },
                "aggregations": aggregations,
                "tokens": [
                    {
                        "index": 0,
                        "token_id": token_id,
                        "text": text,
                        "piece": text,
                        "display": text,
                        "character_start": 0,
                        "character_end": len(text),
                        "probability": 2 ** (-mean_bits),
                        "logprob": -mean_bits * 0.6931471805599453,
                        "surprisal_bits": mean_bits,
                        "rank": 1,
                        "entropy_bits": mean_bits,
                        "effective_tokens": 2**mean_bits,
                        "top1_gap_bits": 0.0,
                        "word_start_hint": True,
                        "alternatives": [],
                    }
                ],
                "summary": {
                    "total_surprisal_bits": mean_bits,
                    "mean_surprisal_bits": mean_bits,
                    "perplexity": 2**mean_bits,
                    "bits_per_character": mean_bits / len(text),
                    "bits_per_utf8_byte": mean_bits / len(text.encode("utf-8")),
                    "mean_entropy_bits": mean_bits,
                },
            }

        metric_values = {
            "mean_surprisal_bits": (2.0, 3.0, 1.0, "bits_per_token"),
            "bits_per_non_whitespace_character": (
                0.5,
                0.75,
                0.25,
                "bits_per_non_whitespace_character",
            ),
            "median_whitespace_span_bits_per_non_whitespace_character": (
                0.4,
                0.7,
                0.3,
                "bits_per_non_whitespace_character",
            ),
            "whitespace_span_iqr_bits_per_non_whitespace_character": (
                0.1,
                0.2,
                0.1,
                "bits_per_non_whitespace_character",
            ),
            "whitespace_span_raw_mad_bits_per_non_whitespace_character": (
                0.05,
                0.08,
                0.03,
                "bits_per_non_whitespace_character",
            ),
        }
        comparison_metrics = {
            name: {
                "a": values[0],
                "b": values[1],
                "delta_b_minus_a": values[2],
                "unit": values[3],
            }
            for name, values in metric_values.items()
        }
        captured = record(
            kind="text_surprisal_comparison",
            payload={
                "labels": {
                    "a": "=Varianta A",
                    "b": "+Varianta B",
                },
                "a": comparison_side("=Ahoj", 10, 2.0),
                "b": comparison_side("+Nazdar", 20, 3.0),
                "comparison": {
                    "delta_definition": "B - A",
                    "method": {
                        "id": "paired_descriptive_text_surprisal",
                        "version": "1.0.0",
                        "status": "descriptive",
                        "delta": "B - A",
                        "same_model": True,
                        "token_alignment": "none",
                    },
                    "provenance": {
                        "source": "@shared",
                        "model_identity": "same_run",
                    },
                    "metrics": comparison_metrics,
                },
            },
        )
        stored, created = self.store.save(captured)
        self.assertTrue(created)
        self.assertEqual(
            self.store.list_summaries(
                "text_surprisal_comparison"
            )["records"][0]["id"],
            captured["id"],
        )
        exported = json.loads(
            self.store.export_json("text_surprisal_comparison")
        )
        self.assertEqual(exported["records"], [stored])

        rows = list(
            csv.DictReader(
                io.StringIO(
                    self.store.export_csv(
                        "text_surprisal_comparison"
                    ).decode("utf-8")
                )
            )
        )
        side_summaries = [
            row
            for row in rows
            if row["export_scope"]
            == "text_surprisal_comparison_side_summary"
        ]
        side_tokens = [
            row
            for row in rows
            if row["export_scope"]
            == "text_surprisal_comparison_side_tokens"
        ]
        aggregation_summaries = [
            row
            for row in rows
            if row["export_scope"]
            == "text_surprisal_comparison_side_aggregation_summary"
        ]
        aggregation_spans = [
            row
            for row in rows
            if row["export_scope"]
            == "text_surprisal_comparison_side_aggregation_spans"
        ]
        metric_rows = [
            row
            for row in rows
            if row["export_scope"]
            == "text_surprisal_comparison_metric"
        ]

        self.assertEqual(len(rows), 21)
        self.assertEqual(len(side_summaries), 2)
        self.assertEqual(len(side_tokens), 2)
        self.assertEqual(len(aggregation_summaries), 6)
        self.assertEqual(len(aggregation_spans), 6)
        self.assertEqual(len(metric_rows), 5)
        self.assertEqual(
            {row["comparison_side"] for row in side_tokens},
            {"A", "B"},
        )
        self.assertEqual(
            {row["aggregation_type"] for row in aggregation_summaries},
            {"whitespace_span", "line", "sentence_like"},
        )
        self.assertEqual(
            {row["comparison_side"] for row in aggregation_spans},
            {"A", "B"},
        )
        side_a = next(
            row for row in side_summaries if row["comparison_side"] == "A"
        )
        side_b = next(
            row for row in side_summaries if row["comparison_side"] == "B"
        )
        self.assertEqual(side_a["comparison_side_label"], "'=Varianta A")
        self.assertEqual(side_b["comparison_side_label"], "'+Varianta B")
        self.assertEqual(side_a["original_text"], "'=Ahoj")
        self.assertEqual(side_b["original_text"], "'+Nazdar")
        self.assertEqual(
            json.loads(side_a["payload_json"]),
            stored["payload"],
        )
        self.assertEqual(
            json.loads(side_a["comparison_side_json"]),
            stored["payload"]["a"],
        )

        mean_row = next(
            row
            for row in metric_rows
            if row["comparison_metric"] == "mean_surprisal_bits"
        )
        self.assertEqual(mean_row["comparison_metric_a"], "2.0")
        self.assertEqual(mean_row["comparison_metric_b"], "3.0")
        self.assertEqual(
            mean_row["comparison_metric_delta_b_minus_a"],
            "1.0",
        )
        self.assertEqual(
            mean_row["comparison_metric_unit"],
            "bits_per_token",
        )
        self.assertEqual(
            mean_row["comparison_metric_method"],
            "paired_descriptive_text_surprisal",
        )
        self.assertEqual(
            json.loads(mean_row["comparison_method_json"])["token_alignment"],
            "none",
        )
        self.assertEqual(
            mean_row["comparison_delta_definition"],
            "B - A",
        )
        self.assertEqual(
            json.loads(mean_row["comparison_metric_provenance_json"]),
            {
                "model_identity": "same_run",
                "source": "@shared",
            },
        )
        self.assertEqual(
            json.loads(mean_row["provenance_json"]),
            stored["provenance"],
        )

    def test_refuses_future_database_version(self) -> None:
        database = Path(self.temporary.name) / "future.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute("PRAGMA user_version = 2")
        with self.assertRaises(SchemaVersionError):
            ExperimentStore(database)


if __name__ == "__main__":
    unittest.main()
