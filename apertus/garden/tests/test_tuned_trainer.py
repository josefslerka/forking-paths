from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apertus.garden.train_tuned_lens import (
    Document,
    load_documents,
    validate_state_cache,
)
from apertus.garden.tuned_lens import sha256_file


class TunedLensStateCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.cache = Path(self.temporary.name)
        self.state_file = self.cache / "00001-state.safetensors"
        self.state_file.write_bytes(b"deterministic-state")
        self.document = Document(
            doc_id="cs-1",
            pair_id="topic-1",
            language="cs",
            source="test",
            revision="1",
            license="test",
            text="Dostatečně dlouhý testovací dokument.",
            split="train",
        )
        self.index = {
            "format": "apertus.tuned-lens-state-cache",
            "schema_version": 3,
            "storage_dtype": "bfloat16",
            "layers": [4],
            "sample_seed": 7,
            "max_positions_per_document": 8,
            "provenance": {
                "model_id": "hybrid",
                "metadata_fingerprint": "a" * 64,
                "content_fingerprint": "b" * 64,
                "document_manifest_sha256": "c" * 64,
                "source_file_sha256": "d" * 64,
            },
            "documents": [
                {
                    "doc_id": "cs-1",
                    "pair_id": "topic-1",
                    "split": "train",
                    "language": "cs",
                }
            ],
            "files": [
                {
                    "file": self.state_file.name,
                    "doc_id": "cs-1",
                    "pair_id": "topic-1",
                    "split": "train",
                    "language": "cs",
                    "positions": 8,
                    "bytes": self.state_file.stat().st_size,
                    "file_sha256": sha256_file(self.state_file),
                }
            ],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def validate(self) -> None:
        validate_state_cache(
            self.cache,
            self.index,
            documents=[self.document],
            layers=[4],
            sample_seed=7,
            max_positions_per_document=8,
            model_id="hybrid",
            metadata_fingerprint="a" * 64,
            content_fingerprint="b" * 64,
            document_manifest_sha256="c" * 64,
            source_file_sha256="d" * 64,
        )

    def test_exact_provenance_and_file_digest_pass(self) -> None:
        self.validate()

    def test_tampered_file_or_split_is_rejected(self) -> None:
        self.state_file.write_bytes(b"tampered-state")
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_missing_positions_is_rejected_before_training(self) -> None:
        del self.index["files"][0]["positions"]
        with self.assertRaisesRegex(RuntimeError, "počet pozic"):
            self.validate()

    def test_trainer_rejects_reserved_exhibit_even_without_builder(self) -> None:
        corpus = self.cache / "corpus.jsonl"
        corpus.write_text(
            json.dumps(
                {
                    "doc_id": "cs-exhibit",
                    "language": "cs",
                    "source": "test",
                    "revision": "1",
                    "license": "test",
                    "text": (
                        "Toto je delší dokument. Hlavním městem České "
                        "republiky je fráze vyhrazená pro živý experiment."
                    ),
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "rezervovaný exhibit"):
            load_documents(
                corpus,
                split_seed=1,
                max_documents_per_language=None,
            )
        self.state_file.write_bytes(b"deterministic-state")
        self.index["documents"][0]["split"] = "test"
        with self.assertRaises(RuntimeError):
            self.validate()


if __name__ == "__main__":
    unittest.main()
