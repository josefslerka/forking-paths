from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from apertus.garden.server import ModelState
from apertus.garden.tuned_lens import (
    ARTIFACT_FORMAT,
    ArtifactValidationError,
    TunedLensCatalog,
    model_content_fingerprint,
    sha256_file,
    validate_manifest,
)


def artifact_manifest(
    directory: Path,
    *,
    artifact_id: str = "pilot-1",
    model_id: str = "hybrid",
    metadata_fingerprint: str = "a" * 64,
    validation_status: str = "pilot",
    accepted: bool = True,
) -> dict:
    weights = directory / "translators.safetensors"
    weights.write_bytes(b"not-real-mlx-weights")
    metrics = directory / "metrics.json"
    metrics.write_text("{}\n", encoding="utf-8")
    selected_layers = [4, 16]
    data_floor = {
        "train": {
            language: {"documents": 32, "positions": 4096}
            for language in ("cs", "en")
        },
        "validation": {
            language: {"documents": 8, "positions": 384}
            for language in ("cs", "en")
        },
        "test": {
            language: {"documents": 8, "positions": 384}
            for language in ("cs", "en")
        },
    }
    held_out = [
        {
            "layer": layer,
            "by_language": {
                language: {
                    "kl_improvement_bits": 0.25,
                    "kl_improvement_ci95_document_bootstrap": [0.1, 0.4],
                }
                for language in ("cs", "en")
            },
        }
        for layer in selected_layers
    ]
    return {
        "format": ARTIFACT_FORMAT,
        "schema_version": 1,
        "artifact_id": artifact_id,
        "created_at": "2026-07-28T12:00:00Z",
        "reader": {
            "kind": "tuned_lens_style_probe",
            "translator_kind": "low_rank_residual_v1",
            "rank": 2,
            "selected_layers": selected_layers,
            "reference_layer": 32,
            "hook": "post_decoder_block_residual",
            "layer_numbering": "one_based",
            "objective": "full_vocab_kl_final_to_reader",
            "temperature": 1.0,
            "hidden_size": 4096,
            "readout": "lm_head(final_rms_norm(translated_residual))",
        },
        "model": {
            "model_id": model_id,
            "metadata_fingerprint": metadata_fingerprint,
            "content_fingerprint": "b" * 64,
            "quantization_mode": model_id,
        },
        "dataset": {
            "dataset_id": "balanced-cs-en",
            "languages": ["cs", "en"],
            "document_manifest_sha256": "c" * 64,
            "source_file_sha256": "d" * 64,
            "state_cache_index_sha256": "e" * 64,
            "split_seed": 1,
            "sample_seed": 2,
            "max_positions_per_document": 48,
            "document_counts": {
                "train": {"cs": 32, "en": 32},
                "validation": {"cs": 8, "en": 8},
                "test": {"cs": 8, "en": 8},
            },
        },
        "validation": {
            "status": validation_status,
            "acceptance": {
                "passed": accepted,
                "criterion": "test",
                "by_layer": {
                    str(layer): accepted for layer in selected_layers
                },
                "data_floor": data_floor,
                "data_floor_passed": accepted,
            },
            "held_out": {"split": "test", "by_layer": held_out},
        },
        "weights": {
            "file": weights.name,
            "bytes": weights.stat().st_size,
            "sha256": sha256_file(weights),
        },
        "metrics": {
            "file": metrics.name,
            "bytes": metrics.stat().st_size,
            "sha256": sha256_file(metrics),
        },
        "limitations": ["pilot"],
    }


class TunedLensArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_artifact(self, **overrides) -> dict:
        directory = self.root / overrides.get("artifact_id", "pilot-1")
        directory.mkdir()
        manifest = artifact_manifest(directory, **overrides)
        (directory / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        return manifest

    def test_manifest_and_catalog_require_exact_model_metadata(self) -> None:
        manifest = self.write_artifact()
        validate_manifest(
            manifest,
            artifact_dir=self.root / "pilot-1",
            verify_weights=True,
        )
        catalog = TunedLensCatalog(self.root)
        ready = catalog.status(
            model_id="hybrid",
            metadata_fingerprint="a" * 64,
        )
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(
            ready["calibration"]["artifact_id"],
            "pilot-1",
        )
        self.assertEqual(ready["content_verification"], "pending")
        incompatible = catalog.status(
            model_id="hybrid",
            metadata_fingerprint="different",
        )
        self.assertEqual(incompatible["status"], "incompatible")
        missing = catalog.status(
            model_id="bf16",
            metadata_fingerprint="a" * 64,
        )
        self.assertEqual(missing["status"], "missing")

    def test_unvalidated_artifact_is_not_exposed_as_ready(self) -> None:
        self.write_artifact(
            validation_status="unvalidated",
            accepted=False,
        )
        status = TunedLensCatalog(self.root).status(
            model_id="hybrid",
            metadata_fingerprint="a" * 64,
        )
        self.assertEqual(status["status"], "unvalidated")

    def test_reader_contract_and_held_out_gate_are_enforced(self) -> None:
        directory = self.root / "pilot-1"
        directory.mkdir()
        manifest = artifact_manifest(directory)
        manifest["reader"]["temperature"] = 2.0
        with self.assertRaises(ArtifactValidationError):
            validate_manifest(manifest, artifact_dir=directory)

        manifest = artifact_manifest(directory)
        manifest["artifact_id"] = "../outside"
        with self.assertRaises(ArtifactValidationError):
            validate_manifest(manifest, artifact_dir=directory)

        manifest = artifact_manifest(directory)
        manifest["validation"]["held_out"]["by_layer"][0][
            "by_language"
        ]["cs"]["kl_improvement_bits"] = float("nan")
        with self.assertRaises(ArtifactValidationError):
            validate_manifest(manifest, artifact_dir=directory)

        manifest = artifact_manifest(directory)
        manifest["validation"]["held_out"]["by_layer"][0][
            "by_language"
        ]["cs"]["kl_improvement_ci95_document_bootstrap"] = [-0.1, 0.4]
        with self.assertRaises(ArtifactValidationError):
            validate_manifest(manifest, artifact_dir=directory)

    def test_newer_unvalidated_artifact_does_not_hide_ready_one(self) -> None:
        self.write_artifact(artifact_id="pilot-ready")
        newer = self.write_artifact(
            artifact_id="pilot-failed",
            validation_status="unvalidated",
            accepted=False,
        )
        newer["created_at"] = "2026-07-29T12:00:00Z"
        (self.root / "pilot-failed" / "manifest.json").write_text(
            json.dumps(newer),
            encoding="utf-8",
        )
        status = TunedLensCatalog(self.root).status(
            model_id="hybrid",
            metadata_fingerprint="a" * 64,
        )
        self.assertEqual(status["status"], "ready")
        self.assertEqual(
            status["calibration"]["artifact_id"],
            "pilot-ready",
        )

    def test_invalid_tensor_shapes_latch_artifact_as_invalid(self) -> None:
        self.write_artifact()
        fake_core = types.ModuleType("mlx.core")
        fake_core.load = lambda _path: {
            "unexpected": types.SimpleNamespace(shape=(1,))
        }
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.core = fake_core
        catalog = TunedLensCatalog(self.root)
        with patch.dict(
            sys.modules,
            {"mlx": fake_mlx, "mlx.core": fake_core},
        ):
            with self.assertRaises(ArtifactValidationError):
                catalog.load(
                    model_id="hybrid",
                    metadata_fingerprint="a" * 64,
                    content_fingerprint="b" * 64,
                )
        status = catalog.status(
            model_id="hybrid",
            metadata_fingerprint="a" * 64,
        )
        self.assertEqual(status["status"], "invalid")
        self.assertEqual(status["content_verification"], "failed")

    def test_fingerprint_cache_is_bound_to_resolved_source(self) -> None:
        first = self.root / "model-a"
        second = self.root / "model-b"
        first.mkdir()
        second.mkdir()
        for directory, content in (
            (first, b"AAAA"),
            (second, b"BBBB"),
        ):
            (directory / "config.json").write_text(
                '{"model_type":"apertus"}',
                encoding="utf-8",
            )
            (directory / "model.safetensors").write_bytes(content)
        timestamp = 1_700_000_000_000_000_000
        for path in (
            first / "config.json",
            second / "config.json",
            first / "model.safetensors",
            second / "model.safetensors",
        ):
            os.utime(path, ns=(timestamp, timestamp))
        cache = self.root / "fingerprint.json"
        first_digest, _ = model_content_fingerprint(
            "8bit",
            [("quantized_body", first)],
            cache_path=cache,
        )
        second_digest, _ = model_content_fingerprint(
            "8bit",
            [("quantized_body", second)],
            cache_path=cache,
        )
        self.assertNotEqual(first_digest, second_digest)

    def test_mock_reader_returns_paired_full_distribution_metrics(self) -> None:
        state = ModelState(
            model_path=Path("."),
            mock=True,
        )
        state.status = "ready"
        state.model_generation = 1
        result = state.tuned_lens(
            "V zahradě po dešti roste",
            5,
            "mock-low-rank-pilot-v1",
            "mock",
            1,
            state.server_instance_id,
        )
        self.assertEqual(result["lens_kind"], "raw_tuned_comparison")
        self.assertEqual(result["reference_layer"], 32)
        self.assertEqual(len(result["layers"]), 7)
        self.assertTrue(
            all(
                layer["raw"]["kl_divergence_to_final_bits"] >= 0
                and layer["tuned"]["kl_divergence_to_final_bits"] >= 0
                and layer["comparison"]["kl_improvement_bits"] > 0
                for layer in result["layers"]
            )
        )
        self.assertEqual(
            result["summary"]["tuned_closer_by_kl_count"],
            len(result["layers"]),
        )
        cached = state.tuned_lens(
            "V zahradě po dešti roste",
            5,
            "mock-low-rank-pilot-v1",
            "mock",
            1,
            state.server_instance_id,
        )
        self.assertTrue(cached["cached"])


if __name__ == "__main__":
    unittest.main()
