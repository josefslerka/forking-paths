from __future__ import annotations

import hashlib
import json
import sys
import types
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from apertus.garden.install_jacobian_lens import install_runtime_zip
from apertus.garden.jacobian_lens import (
    ARTIFACT_FORMAT,
    ArtifactValidationError,
    JacobianLensCatalog,
    JacobianLensIncompatibleError,
    TransferProbeError,
    apply_jacobian,
    runtime_probe_report,
    static_transport_probe_report,
    validate_bf16_runtime_contract,
    validate_manifest,
    validate_weight_shapes,
)


def digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


class ArrayWithDtype:
    def __init__(self, value: np.ndarray, dtype_name: str):
        self.value = np.asarray(value)
        self.dtype = dtype_name
        self.shape = self.value.shape

    def __array__(self, dtype=None, copy=None):
        del copy
        return np.asarray(self.value, dtype=dtype)

    def astype(self, dtype):
        return self.value.astype(dtype)


def fixture() -> tuple[dict, dict[str, bytes], dict[str, object]]:
    hidden_size = 4
    jacobian = np.array(
        [
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 3.0, 0.0],
            [0.0, 0.0, 1.0, 4.0],
            [5.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float16,
    )
    residual_value = np.array([1.0, -2.0, 3.0, 0.5], dtype=np.float32)
    transported = residual_value @ jacobian.astype(np.float32).T
    probe_json = {
        "format": "apertus.jacobian-lens-transfer-probe",
        "schema_version": 1,
        "model_id": "swiss-ai/Apertus-8B-2509",
        "model_revision": "3162c99675aa588097cecd4a24b9aa1f712af477",
        "tokenizer_sha256": "b" * 64,
        "records": [
            {
                "prompt_index": 0,
                "prompt": "test",
                "input_ids": [1, 4417],
                "last_token_id": 4417,
                "final_top20": [],
                "layers": [
                    {
                        "layer": 1,
                        "raw_top20": [],
                        "jlens_top20": [],
                    }
                ],
            }
        ],
    }
    contents = {
        "jacobians.safetensors": b"weights",
        "metrics.json": json.dumps(
            {
                "format": "apertus.jacobian-lens-metrics",
                "schema_version": 1,
            }
        ).encode(),
        "transfer_probe.safetensors": b"probe tensors",
        "transfer_probe.json": json.dumps(probe_json).encode(),
        "corpus.manifest.json": b"{}",
        "run.manifest.json": b"{}",
        "split_summaries.json": b"{}",
        "first_prompt_benchmark.json": b"{}",
    }
    files = {
        "canonical_master": {
            "name": "jacobian_lens.merged.pt",
            "sha256": "c" * 64,
            "location": "not included",
        },
        "weights": {
            "name": "jacobians.safetensors",
            "sha256": digest(contents["jacobians.safetensors"]),
            "bytes": len(contents["jacobians.safetensors"]),
            "dtype": "float16",
            "keys": ["layer_01.jacobian"],
        },
        "metrics": {
            "name": "metrics.json",
            "sha256": digest(contents["metrics.json"]),
        },
        "transfer_probe_tensors": {
            "name": "transfer_probe.safetensors",
            "sha256": digest(contents["transfer_probe.safetensors"]),
            "residual_dtype": "bfloat16",
            "transported_dtype": "float32",
        },
        "transfer_probe_json": {
            "name": "transfer_probe.json",
            "sha256": digest(contents["transfer_probe.json"]),
        },
        "corpus_manifest": {
            "name": "corpus.manifest.json",
            "sha256": digest(contents["corpus.manifest.json"]),
        },
        "run_manifest": {
            "name": "run.manifest.json",
            "sha256": digest(contents["run.manifest.json"]),
        },
        "split_summaries": {
            "name": "split_summaries.json",
            "sha256": digest(contents["split_summaries.json"]),
        },
        "first_prompt_benchmark": {
            "name": "first_prompt_benchmark.json",
            "sha256": digest(contents["first_prompt_benchmark.json"]),
        },
    }
    manifest = {
        "format": ARTIFACT_FORMAT,
        "schema_version": 1,
        "artifact_id": "tiny-jlens-v1",
        "artifact_scope": "unit_test",
        "created_at_utc": "2026-08-01T00:00:00+00:00",
        "compatibility": {
            "allowed_model_modes": ["bf16"],
            "hybrid_transfer": "not_validated",
            "mlx_enablement_requires_transfer_probe_match": True,
        },
        "reader": {
            "kind": "average_jacobian_lens",
            "estimator": "current_and_future_targets",
            "hook": "post_decoder_block_residual",
            "layer_numbering": "one_based",
            "target_layer": 32,
            "transport": "residual @ jacobian.T",
            "readout": "lm_head(final_rms_norm(transported_residual))",
            "hidden_size": hidden_size,
            "vocab_size": 8,
            "source_layers": [1],
            "n_prompts": 2,
            "max_seq_len": 8,
            "skip_first": 1,
        },
        "model": {
            "hf_repo": "swiss-ai/Apertus-8B-2509",
            "hf_revision": "3162c99675aa588097cecd4a24b9aa1f712af477",
            "mode": "bf16",
            "architectures": ["ApertusForCausalLM"],
            "tie_word_embeddings": False,
            "config_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
        },
        "corpus": {
            "languages": ["cs", "en"],
            "corpus_sha256": "d" * 64,
        },
        "files": files,
        "software": {"transformers": "5.14.1"},
        "fit": {"jlens_commit": "e" * 40},
        "limitations": ["Unit-test fixture."],
    }
    tensors = {
        "weights": {"layer_01.jacobian": jacobian},
        "probe": {
            "prompt_00.input_ids": np.array([1, 4417], dtype=np.int64),
            "prompt_00.layer_01.residual": ArrayWithDtype(
                residual_value,
                "bfloat16",
            ),
            "prompt_00.layer_01.transported": transported.astype(np.float32),
        },
        "probe_json": probe_json,
    }
    return manifest, contents, tensors


def write_artifact(
    root: Path,
    manifest: dict,
    contents: dict[str, bytes],
) -> Path:
    directory = root / manifest["artifact_id"]
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    for name, value in contents.items():
        (directory / name).write_bytes(value)
    return directory


class ManifestTests(unittest.TestCase):
    def test_valid_manifest_and_file_hashes(self):
        manifest, contents, _ = fixture()
        with TemporaryDirectory() as temporary:
            directory = write_artifact(Path(temporary), manifest, contents)
            result = validate_manifest(
                manifest,
                artifact_dir=directory,
                verify_files=True,
                expected_hidden_size=4,
                expected_vocab_size=8,
            )
        self.assertEqual(result["artifact_id"], "tiny-jlens-v1")

    def test_accepts_paired_penultimate_target_and_rejects_bad_pairing(self):
        manifest, _, _ = fixture()
        manifest["reader"]["target_layer"] = 31
        manifest["reader"]["paper_primary_target_variant"] = True
        result = validate_manifest(
            manifest,
            expected_hidden_size=4,
            expected_vocab_size=8,
        )
        self.assertEqual(result["reader"]["target_layer"], 31)

        manifest["reader"]["source_layers"] = [31]
        manifest["files"]["weights"]["keys"] = ["layer_31.jacobian"]
        with self.assertRaisesRegex(ArtifactValidationError, "před targetem"):
            validate_manifest(
                manifest,
                expected_hidden_size=4,
                expected_vocab_size=8,
            )

        manifest["reader"]["source_layers"] = [1]
        manifest["files"]["weights"]["keys"] = ["layer_01.jacobian"]
        manifest["reader"]["paper_primary_target_variant"] = False
        with self.assertRaisesRegex(
            ArtifactValidationError,
            "paper_primary_target_variant",
        ):
            validate_manifest(
                manifest,
                expected_hidden_size=4,
                expected_vocab_size=8,
            )

    def test_rejects_hybrid_and_bad_hash(self):
        manifest, contents, _ = fixture()
        manifest["compatibility"]["allowed_model_modes"] = ["hybrid"]
        with self.assertRaises(ArtifactValidationError):
            validate_manifest(
                manifest,
                expected_hidden_size=4,
                expected_vocab_size=8,
            )
        manifest["compatibility"]["allowed_model_modes"] = ["bf16"]
        with TemporaryDirectory() as temporary:
            directory = write_artifact(Path(temporary), manifest, contents)
            (directory / "metrics.json").write_bytes(b"changed")
            with self.assertRaises(ArtifactValidationError):
                validate_manifest(
                    manifest,
                    artifact_dir=directory,
                    verify_files=True,
                    expected_hidden_size=4,
                    expected_vocab_size=8,
                )

    def test_rejects_bad_weight_shape_dtype_and_nonfinite_value(self):
        manifest, _, tensors = fixture()
        for value in (
            np.zeros((4, 3), dtype=np.float16),
            np.zeros((4, 4), dtype=np.float32),
            np.full((4, 4), np.nan, dtype=np.float16),
        ):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises(ArtifactValidationError):
                    validate_weight_shapes(
                        manifest,
                        {"layer_01.jacobian": value},
                    )


class ProbeTests(unittest.TestCase):
    def test_static_transport_catches_transpose(self):
        manifest, _, tensors = fixture()
        report = static_transport_probe_report(
            manifest,
            tensors["weights"],
            tensors["probe"],
            tensors["probe_json"],
        )
        self.assertTrue(report["passed"])

        def wrong_transport(residual, matrix):
            return np.asarray(residual, dtype=np.float32) @ np.asarray(
                matrix,
                dtype=np.float32,
            )

        report = static_transport_probe_report(
            manifest,
            tensors["weights"],
            tensors["probe"],
            tensors["probe_json"],
            transport=wrong_transport,
        )
        self.assertFalse(report["passed"])

    def test_runtime_probe_reports_token_and_residual_failures(self):
        manifest, _, tensors = fixture()
        good = runtime_probe_report(
            manifest,
            tensors["probe_json"],
            tensors["probe"],
            actual_input_ids={0: [1, 4417]},
            actual_residuals={
                (0, 1): np.array([1.0, -2.0, 3.0, 0.5]),
            },
        )
        self.assertTrue(good.passed)
        bad = runtime_probe_report(
            manifest,
            tensors["probe_json"],
            tensors["probe"],
            actual_input_ids={0: [4417]},
            actual_residuals={(0, 1): np.zeros(4)},
        )
        self.assertFalse(bad.passed)

    def test_apply_jacobian_orientation_with_fake_mlx(self):
        manifest, _, tensors = fixture()
        fake_core = types.ModuleType("mlx.core")
        fake_core.float32 = np.float32
        fake_core.transpose = np.transpose
        fake_core.matmul = np.matmul
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.core = fake_core
        previous_mlx = sys.modules.get("mlx")
        previous_core = sys.modules.get("mlx.core")
        sys.modules["mlx"] = fake_mlx
        sys.modules["mlx.core"] = fake_core
        try:
            state = np.array([1.0, -2.0, 3.0, 0.5], dtype=np.float16)
            actual = apply_jacobian(
                state,
                layer=1,
                manifest=manifest,
                weights=tensors["weights"],
            )
        finally:
            if previous_mlx is None:
                sys.modules.pop("mlx", None)
            else:
                sys.modules["mlx"] = previous_mlx
            if previous_core is None:
                sys.modules.pop("mlx.core", None)
            else:
                sys.modules["mlx.core"] = previous_core
        expected = (
            state.astype(np.float32)
            @ tensors["weights"]["layer_01.jacobian"].astype(np.float32).T
        )
        np.testing.assert_allclose(actual, expected)

    def test_bf16_runtime_structure_gate(self):
        manifest, _, _ = fixture()
        bf16_weight = ArrayWithDtype(np.zeros((1, 1)), "bfloat16")
        body = types.SimpleNamespace(
            layers=[object() for _ in range(32)],
            norm=object(),
            embed_tokens=types.SimpleNamespace(weight=bf16_weight),
        )
        model = types.SimpleNamespace(
            model_type="apertus",
            model=body,
            lm_head=types.SimpleNamespace(weight=bf16_weight),
            args=types.SimpleNamespace(hidden_size=4, vocab_size=8),
        )
        validate_bf16_runtime_contract(
            model,
            model_id="bf16",
            manifest=manifest,
        )
        with self.assertRaises(JacobianLensIncompatibleError):
            validate_bf16_runtime_contract(
                model,
                model_id="hybrid",
                manifest=manifest,
            )


class CatalogTests(unittest.TestCase):
    def test_status_lists_and_selects_multiple_target_artifacts(self):
        final_manifest, final_contents, _ = fixture()
        final_manifest["artifact_id"] = "final-target"
        penultimate_manifest = json.loads(json.dumps(final_manifest))
        penultimate_manifest["artifact_id"] = "penultimate-target"
        penultimate_manifest["created_at_utc"] = "2026-08-02T00:00:00+00:00"
        penultimate_manifest["reader"]["target_layer"] = 31
        penultimate_manifest["reader"]["paper_primary_target_variant"] = True

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root, final_manifest, final_contents)
            write_artifact(root, penultimate_manifest, final_contents)
            catalog = JacobianLensCatalog(
                root,
                tensor_loader=lambda _path: {},
                expected_hidden_size=4,
                expected_vocab_size=8,
            )

            default_status = catalog.status(model_id="bf16")
            self.assertEqual(
                default_status["artifact_id"],
                "final-target",
            )
            self.assertEqual(len(default_status["artifacts"]), 2)
            selected = catalog.status(
                model_id="bf16",
                expected_artifact_id="final-target",
            )
            self.assertEqual(selected["artifact_id"], "final-target")
            self.assertEqual(selected["target_layer"], 32)
            missing = catalog.status(
                model_id="bf16",
                expected_artifact_id="not-installed",
            )
            self.assertEqual(missing["status"], "missing")
            self.assertEqual(len(missing["artifacts"]), 2)

            hybrid_default = catalog.status(model_id="hybrid")
            self.assertEqual(hybrid_default["status"], "incompatible")
            self.assertEqual(
                hybrid_default["artifact_id"],
                "final-target",
            )
            self.assertEqual(hybrid_default["target_layer"], 32)
            self.assertEqual(hybrid_default["source_layers"], [1])

            hybrid_penultimate = catalog.status(
                model_id="hybrid",
                expected_artifact_id="penultimate-target",
            )
            self.assertEqual(
                hybrid_penultimate["artifact_id"],
                "penultimate-target",
            )
            self.assertEqual(hybrid_penultimate["target_layer"], 31)
            self.assertEqual(len(hybrid_penultimate["artifacts"]), 2)

    def test_load_attest_and_invalidate_by_model_fingerprint(self):
        manifest, contents, tensors = fixture()

        def loader(path: Path):
            return (
                tensors["weights"]
                if path.name == "jacobians.safetensors"
                else tensors["probe"]
            )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root, manifest, contents)
            catalog = JacobianLensCatalog(
                root,
                tensor_loader=loader,
                expected_hidden_size=4,
                expected_vocab_size=8,
            )
            self.assertEqual(
                catalog.status(model_id="bf16")["status"],
                "pending_probe",
            )
            loaded, _, _, _ = catalog.load(model_id="bf16")
            fingerprint = "f" * 64
            report = runtime_probe_report(
                manifest,
                tensors["probe_json"],
                tensors["probe"],
                actual_input_ids={0: [1, 4417]},
                actual_residuals={
                    (0, 1): np.array([1.0, -2.0, 3.0, 0.5]),
                },
            ).to_dict()
            report.update(
                {
                    "jacobian_output_checks": [
                        {
                            "passed": True,
                            "top1_matches": True,
                            "top20_overlap": 1.0,
                        }
                    ],
                    "output_checks": [
                        {
                            "passed": True,
                            "top20_overlap": 1.0,
                        }
                    ],
                }
            )
            valid_report = json.loads(json.dumps(report))
            catalog.mark_runtime_probe(
                artifact_id=loaded["artifact_id"],
                weights_sha256=loaded["files"]["weights"]["sha256"],
                model_fingerprint=fingerprint,
                report=report,
            )
            self.assertEqual(
                catalog.status(
                    model_id="bf16",
                    model_fingerprint=fingerprint,
                )["status"],
                "ready",
            )
            self.assertEqual(
                catalog.status(
                    model_id="bf16",
                    model_fingerprint="0" * 64,
                )["status"],
                "pending_probe",
            )
            with self.assertRaises(JacobianLensIncompatibleError):
                catalog.load(model_id="hybrid")
            with self.assertRaises(TransferProbeError):
                catalog.mark_runtime_probe(
                    artifact_id=loaded["artifact_id"],
                    weights_sha256=loaded["files"]["weights"]["sha256"],
                    model_fingerprint=fingerprint,
                    report={
                        "kind": "pytorch_bf16_to_mlx_runtime_probe",
                        "passed": False,
                    },
                )

            invalid_reports = []
            low_single_overlap = json.loads(json.dumps(valid_report))
            low_single_overlap["jacobian_output_checks"][0][
                "top20_overlap"
            ] = 0.59
            invalid_reports.append(low_single_overlap)

            low_mean_overlap = json.loads(json.dumps(valid_report))
            low_mean_overlap["jacobian_output_checks"] = [
                {
                    "passed": True,
                    "top1_matches": True,
                    "top20_overlap": value,
                }
                for value in (0.79, 0.80, 0.80)
            ]
            low_mean_overlap["output_checks"] = [
                {"passed": True, "top20_overlap": 0.80}
                for _ in range(3)
            ]
            invalid_reports.append(low_mean_overlap)

            noninferior_failure = json.loads(json.dumps(valid_report))
            noninferior_failure["jacobian_output_checks"] = [
                {
                    "passed": True,
                    "top1_matches": True,
                    "top20_overlap": 0.84,
                }
                for _ in range(3)
            ]
            noninferior_failure["output_checks"] = [
                {"passed": True, "top20_overlap": 0.90}
                for _ in range(3)
            ]
            invalid_reports.append(noninferior_failure)

            for invalid_report in invalid_reports:
                with self.subTest(
                    jacobian_checks=invalid_report[
                        "jacobian_output_checks"
                    ],
                    final_checks=invalid_report["output_checks"],
                ), self.assertRaises(TransferProbeError):
                    catalog.mark_runtime_probe(
                        artifact_id=loaded["artifact_id"],
                        weights_sha256=loaded["files"]["weights"][
                            "sha256"
                        ],
                        model_fingerprint=fingerprint,
                        report=invalid_report,
                    )


class InstallerTests(unittest.TestCase):
    def make_zip(
        self,
        path: Path,
        manifest: dict,
        contents: dict[str, bytes],
        *,
        extra: tuple[str, bytes] | None = None,
    ) -> None:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            for name, value in contents.items():
                archive.writestr(name, value)
            if extra is not None:
                archive.writestr(*extra)

    def test_installs_atomically_and_is_idempotent(self):
        manifest, contents, _ = fixture()
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            archive_path = temporary_path / "artifact.zip"
            root = temporary_path / "installed"
            self.make_zip(archive_path, manifest, contents)
            installed = install_runtime_zip(
                archive_path,
                root,
                expected_hidden_size=4,
                expected_vocab_size=8,
            )
            self.assertTrue((installed / "manifest.json").is_file())
            self.assertEqual(
                install_runtime_zip(
                    archive_path,
                    root,
                    expected_hidden_size=4,
                    expected_vocab_size=8,
                ),
                installed,
            )

    def test_rejects_traversal_and_rolls_back(self):
        manifest, contents, _ = fixture()
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            archive_path = temporary_path / "bad.zip"
            root = temporary_path / "installed"
            self.make_zip(
                archive_path,
                manifest,
                contents,
                extra=("../escape", b"no"),
            )
            with self.assertRaises(ArtifactValidationError):
                install_runtime_zip(
                    archive_path,
                    root,
                    expected_hidden_size=4,
                    expected_vocab_size=8,
                )
            self.assertFalse((root / manifest["artifact_id"]).exists())
            self.assertFalse((temporary_path / "escape").exists())

    def test_rejects_undeclared_member_and_bad_member_hash(self):
        manifest, contents, _ = fixture()
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            root = temporary_path / "installed"
            extra_zip = temporary_path / "extra.zip"
            self.make_zip(
                extra_zip,
                manifest,
                contents,
                extra=("undeclared.txt", b"no"),
            )
            with self.assertRaises(ArtifactValidationError):
                install_runtime_zip(
                    extra_zip,
                    root,
                    expected_hidden_size=4,
                    expected_vocab_size=8,
                )
            bad_zip = temporary_path / "bad-hash.zip"
            bad_contents = dict(contents)
            bad_contents["metrics.json"] = b"changed"
            self.make_zip(bad_zip, manifest, bad_contents)
            with self.assertRaises(ArtifactValidationError):
                install_runtime_zip(
                    bad_zip,
                    root,
                    expected_hidden_size=4,
                    expected_vocab_size=8,
                )
            self.assertFalse((root / manifest["artifact_id"]).exists())


if __name__ == "__main__":
    unittest.main()
