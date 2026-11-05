#!/usr/bin/env python3
"""Offline calibration for the Apertus RAW × TUNED pilot.

The script intentionally does not train from the interactive prompt archive.
It expects document-level JSONL, fixes the split before tokenization, extracts
post-block residual states once, then trains one translator at a time against
the frozen final L32 distribution over the complete 131,072-token vocabulary.

Example JSONL row:

    {"doc_id":"cs-001","pair_id":"topic-001","language":"cs",
     "source":"Wikimedia","revision":"123","license":"CC BY-SA 4.0",
     "text":"..."}

Translated or parallel variants must share ``pair_id`` so they remain in the
same train/validation/test split.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

try:
    from .calibration_policy import EXCLUDED_EXHIBIT_FRAGMENTS
    from .server import (
        DEFAULT_8BIT_MODEL,
        DEFAULT_MODEL,
        DEFAULT_QUANTIZED_MODEL,
        HYBRID_MODEL_ID,
        MODEL_LABELS,
        ModelState,
        encode_with_bos,
        metadata_fingerprint,
    )
    from .tuned_lens import (
        ARTIFACT_FORMAT,
        ARTIFACT_SCHEMA_VERSION,
        DEFAULT_PILOT_LAYERS,
        REFERENCE_LAYER,
        SAFE_ID_PATTERN,
        canonical_json_sha256,
        model_content_fingerprint,
        sha256_file,
        validate_manifest,
    )
except ImportError:
    from calibration_policy import EXCLUDED_EXHIBIT_FRAGMENTS
    from server import (
        DEFAULT_8BIT_MODEL,
        DEFAULT_MODEL,
        DEFAULT_QUANTIZED_MODEL,
        HYBRID_MODEL_ID,
        MODEL_LABELS,
        ModelState,
        encode_with_bos,
        metadata_fingerprint,
    )
    from tuned_lens import (
        ARTIFACT_FORMAT,
        ARTIFACT_SCHEMA_VERSION,
        DEFAULT_PILOT_LAYERS,
        REFERENCE_LAYER,
        SAFE_ID_PATTERN,
        canonical_json_sha256,
        model_content_fingerprint,
        sha256_file,
        validate_manifest,
    )


SPLITS = ("train", "validation", "test")
LANGUAGES = ("cs", "en")
NORMALIZE_SPACE = re.compile(r"\s+")
TRAINER_VERSION = 1


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def installed_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def code_provenance() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("tuned_lens.py"),
        Path(__file__).resolve().with_name("server.py"),
        Path(__file__).resolve().with_name("calibration_policy.py"),
    ]
    result: dict[str, Any] = {
        "repository_root_name": repo_root.name,
        "captured_at": utc_now(),
        "capture_phase": "training_start",
        "source_files": {
            str(path.relative_to(repo_root)): sha256_file(path)
            for path in source_paths
        },
    }
    try:
        result["git_commit"] = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *(str(path.relative_to(repo_root)) for path in source_paths),
            ],
            check=True,
            capture_output=True,
        ).stdout
        diff = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "diff",
                "--binary",
                "--",
                *(str(path.relative_to(repo_root)) for path in source_paths),
            ],
            check=True,
            capture_output=True,
        ).stdout
        result["dirty"] = bool(status.strip())
        result["status_sha256"] = hashlib.sha256(status).hexdigest()
        result["tracked_diff_sha256"] = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        result.update(
            {
                "git_commit": None,
                "dirty": None,
                "status_sha256": None,
                "tracked_diff_sha256": None,
            }
        )
    return result


def normalize_document(text: str) -> str:
    return NORMALIZE_SPACE.sub(" ", text).strip()


@dataclass(frozen=True)
class Document:
    doc_id: str
    pair_id: str
    language: str
    source: str
    revision: str
    license: str
    text: str
    split: str

    def public_record(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "pair_id": self.pair_id,
            "language": self.language,
            "source": self.source,
            "revision": self.revision,
            "license": self.license,
            "split": self.split,
            "text_sha256": hashlib.sha256(
                self.text.encode("utf-8")
            ).hexdigest(),
            "characters": len(self.text),
        }


def stable_split(pair_id: str, seed: int) -> str:
    digest = hashlib.sha256(
        f"apertus-tuned-split-v1\0{seed}\0{pair_id}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < 0.80:
        return "train"
    if value < 0.90:
        return "validation"
    return "test"


def load_documents(
    path: Path,
    *,
    split_seed: int,
    max_documents_per_language: int | None,
) -> list[Document]:
    documents: list[Document] = []
    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}
    pair_splits: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: neplatný JSON."
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(
                    f"{path}:{line_number}: řádek musí být JSON objekt."
                )
            required = (
                "doc_id",
                "language",
                "source",
                "revision",
                "license",
                "text",
            )
            missing = [
                field
                for field in required
                if not isinstance(raw.get(field), str)
                or not raw[field].strip()
            ]
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: chybí {', '.join(missing)}."
                )
            doc_id = raw["doc_id"].strip()
            if doc_id in seen_ids:
                raise ValueError(f"Duplicitní doc_id: {doc_id}")
            seen_ids.add(doc_id)
            language = raw["language"].strip().casefold()
            if language not in LANGUAGES:
                raise ValueError(
                    f"Dokument {doc_id}: podporované jazyky jsou cs a en."
                )
            text = normalize_document(raw["text"])
            if len(text) < 40:
                raise ValueError(
                    f"Dokument {doc_id}: po normalizaci je příliš krátký."
                )
            folded_text = text.casefold()
            matched_exhibits = [
                fragment
                for fragment in EXCLUDED_EXHIBIT_FRAGMENTS
                if fragment in folded_text
            ]
            if matched_exhibits:
                raise ValueError(
                    f"Dokument {doc_id}: obsahuje rezervovaný exhibit "
                    f"{matched_exhibits[0]!r}."
                )
            text_hash = hashlib.sha256(
                folded_text.encode("utf-8")
            ).hexdigest()
            if text_hash in seen_text:
                raise ValueError(
                    f"Shodný normalizovaný text: {doc_id} a "
                    f"{seen_text[text_hash]}."
                )
            seen_text[text_hash] = doc_id
            pair_id = str(raw.get("pair_id") or doc_id).strip()
            declared_split = raw.get("split")
            if declared_split is not None:
                if declared_split not in SPLITS:
                    raise ValueError(
                        f"Dokument {doc_id}: split musí být train, "
                        "validation nebo test."
                    )
                split = declared_split
            else:
                split = stable_split(pair_id, split_seed)
            previous = pair_splits.setdefault(pair_id, split)
            if previous != split:
                raise ValueError(
                    f"Paralelní skupina {pair_id} zasahuje do více splitů."
                )
            documents.append(
                Document(
                    doc_id=doc_id,
                    pair_id=pair_id,
                    language=language,
                    source=raw["source"].strip(),
                    revision=raw["revision"].strip(),
                    license=raw["license"].strip(),
                    text=text,
                    split=split,
                )
            )

    documents.sort(key=lambda item: (item.split, item.language, item.doc_id))
    if max_documents_per_language is not None:
        limited: list[Document] = []
        counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        for document in documents:
            key = (document.split, document.language)
            if counts[key] >= max_documents_per_language:
                continue
            limited.append(document)
            counts[key] += 1
        documents = limited
    for split in SPLITS:
        for language in LANGUAGES:
            if not any(
                document.split == split and document.language == language
                for document in documents
            ):
                raise ValueError(
                    f"Po dělení chybí dokumenty: {split}/{language}."
                )
    return documents


def dataset_manifest(
    documents: list[Document],
    *,
    dataset_id: str,
    source_path: Path,
    split_seed: int,
) -> dict[str, Any]:
    public = [document.public_record() for document in documents]
    pair_languages: defaultdict[str, set[str]] = defaultdict(set)
    for document in documents:
        pair_languages[document.pair_id].add(document.language)
    shared_pair_groups = sum(
        languages == set(LANGUAGES)
        for languages in pair_languages.values()
    )
    if shared_pair_groups == len(pair_languages):
        pairing_status = "complete"
    elif shared_pair_groups:
        pairing_status = "partial"
    else:
        pairing_status = "unavailable"
    counts = {
        split: {
            language: sum(
                document.split == split and document.language == language
                for document in documents
            )
            for language in LANGUAGES
        }
        for split in SPLITS
    }
    return {
        "dataset_id": dataset_id,
        "format": "document_jsonl",
        "source_file_sha256": sha256_file(source_path),
        "document_manifest_sha256": canonical_json_sha256(public),
        "split_seed": split_seed,
        "split_unit": "pair_id_before_tokenization",
        "cross_language_pairing": {
            "status": pairing_status,
            "shared_pair_groups": shared_pair_groups,
            "total_pair_groups": len(pair_languages),
            "parallel_topic_leakage_ruled_out": (
                pairing_status == "complete"
            ),
        },
        "languages": list(LANGUAGES),
        "document_counts": counts,
        "documents": public,
        "normalization": "collapse_unicode_whitespace",
        "deduplication": "exact_normalized_sha256",
    }


def deterministic_positions(
    token_count: int,
    *,
    document_id: str,
    seed: int,
    maximum: int,
) -> list[int]:
    # Position t reads p(. | x_<=t).  Excluding t=0 avoids treating the bare
    # BOS state as calibration data; excluding the last token preserves a
    # true next-token target for secondary NLL evaluation.
    candidates = list(range(1, max(1, token_count - 1)))
    local_seed = int.from_bytes(
        hashlib.sha256(
            f"{seed}\0{document_id}".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    rng = random.Random(local_seed)
    rng.shuffle(candidates)
    return sorted(candidates[:maximum])


def extract_states(
    *,
    model: Any,
    tokenizer: Any,
    documents: list[Document],
    layers: list[int],
    output_dir: Path,
    sample_seed: int,
    max_positions_per_document: int,
    model_id: str,
    metadata_fingerprint: str,
    content_fingerprint: str,
    document_manifest_sha256: str,
    source_file_sha256: str,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    body = model.model
    if len(body.layers) != REFERENCE_LAYER:
        raise RuntimeError("Apertus musí mít právě 32 decoderových vrstev.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise RuntimeError(
            f"State cache už existuje a nebude přepsána: {output_dir}"
        )
    working_dir = output_dir.parent / (
        f".{output_dir.name}.{uuid.uuid4().hex}.partial"
    )
    working_dir.mkdir()
    index: list[dict[str, Any]] = []
    position_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    character_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    token_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    started = time.perf_counter()
    for number, document in enumerate(documents, start=1):
        token_ids = encode_with_bos(tokenizer, document.text)
        positions = deterministic_positions(
            len(token_ids),
            document_id=document.doc_id,
            seed=sample_seed,
            maximum=max_positions_per_document,
        )
        if not positions:
            continue
        ids = mx.array([token_ids])
        hidden = body.embed_tokens(ids)
        causal_mask = create_attention_mask(hidden, None)
        selected: dict[int, Any] = {}
        for layer_number, layer in enumerate(body.layers, start=1):
            hidden = layer(hidden, mask=causal_mask, cache=None)
            if layer_number in layers:
                selected[layer_number] = hidden[0, positions, :]
        teacher = hidden[0, positions, :]
        arrays = {
            f"layer_{layer:02d}": selected[layer].astype(mx.bfloat16)
            for layer in layers
        }
        arrays["teacher"] = teacher.astype(mx.bfloat16)
        arrays["target_ids"] = mx.array(
            [token_ids[position + 1] for position in positions]
        )
        mx.eval(*arrays.values())
        file_name = (
            f"{number:05d}-"
            f"{hashlib.sha256(document.doc_id.encode()).hexdigest()[:12]}"
            ".safetensors"
        )
        file_path = working_dir / file_name
        mx.save_safetensors(str(file_path), arrays)
        key = (document.split, document.language)
        position_counts[key] += len(positions)
        character_counts[key] += len(document.text)
        token_counts[key] += len(token_ids)
        index.append(
            {
                "file": file_name,
                "doc_id": document.doc_id,
                "pair_id": document.pair_id,
                "split": document.split,
                "language": document.language,
                "positions": len(positions),
                "token_count": len(token_ids),
                "characters": len(document.text),
                "positions_sha256": canonical_json_sha256(positions),
                "bytes": file_path.stat().st_size,
                "file_sha256": sha256_file(file_path),
            }
        )
        print(
            f"[{number:04d}/{len(documents):04d}] "
            f"{document.split:10} {document.language} "
            f"{len(positions):4d} pozic · {document.doc_id}",
            flush=True,
        )
        del ids, hidden, teacher, selected, arrays
        gc.collect()
        mx.clear_cache()

    diagnostics = {
        split: {
            language: {
                "positions": position_counts[(split, language)],
                "characters": character_counts[(split, language)],
                "tokens": token_counts[(split, language)],
                "characters_per_token": (
                    character_counts[(split, language)]
                    / token_counts[(split, language)]
                    if token_counts[(split, language)]
                    else None
                ),
            }
            for language in LANGUAGES
        }
        for split in SPLITS
    }
    payload = {
        "format": "apertus.tuned-lens-state-cache",
        "schema_version": 3,
        "storage_dtype": "bfloat16",
        "layers": layers,
        "sample_seed": sample_seed,
        "max_positions_per_document": max_positions_per_document,
        "provenance": {
            "model_id": model_id,
            "metadata_fingerprint": metadata_fingerprint,
            "content_fingerprint": content_fingerprint,
            "document_manifest_sha256": document_manifest_sha256,
            "source_file_sha256": source_file_sha256,
        },
        "created_at": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "diagnostics": diagnostics,
        "documents": [
            {
                "doc_id": document.doc_id,
                "pair_id": document.pair_id,
                "split": document.split,
                "language": document.language,
            }
            for document in documents
        ],
        "files": index,
    }
    (working_dir / "index.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    working_dir.rename(output_dir)
    return payload


def validate_state_cache(
    cache_dir: Path,
    cache_index: dict[str, Any],
    *,
    documents: list[Document],
    layers: list[int],
    sample_seed: int,
    max_positions_per_document: int,
    model_id: str,
    metadata_fingerprint: str,
    content_fingerprint: str,
    document_manifest_sha256: str,
    source_file_sha256: str,
) -> None:
    expected_provenance = {
        "model_id": model_id,
        "metadata_fingerprint": metadata_fingerprint,
        "content_fingerprint": content_fingerprint,
        "document_manifest_sha256": document_manifest_sha256,
        "source_file_sha256": source_file_sha256,
    }
    if (
        cache_index.get("format") != "apertus.tuned-lens-state-cache"
        or cache_index.get("schema_version") != 3
        or cache_index.get("storage_dtype") != "bfloat16"
        or cache_index.get("layers") != layers
        or cache_index.get("sample_seed") != sample_seed
        or cache_index.get("max_positions_per_document")
        != max_positions_per_document
        or cache_index.get("provenance") != expected_provenance
    ):
        raise RuntimeError(
            "State cache neodpovídá modelu, korpusu, splitu nebo samplingu."
        )
    files = cache_index.get("files")
    if not isinstance(files, list):
        raise RuntimeError("State cache nemá platný seznam dokumentů.")
    expected_documents = {
        document.doc_id: {
            "pair_id": document.pair_id,
            "split": document.split,
            "language": document.language,
        }
        for document in documents
    }
    indexed_documents = cache_index.get("documents")
    if not isinstance(indexed_documents, list):
        raise RuntimeError("State cache nemá dokumentový manifest.")
    actual_documents: dict[str, dict[str, str]] = {}
    for item in indexed_documents:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("doc_id"), str)
            or item["doc_id"] in actual_documents
        ):
            raise RuntimeError(
                "State cache obsahuje neplatný dokumentový manifest."
            )
        actual_documents[item["doc_id"]] = {
            "pair_id": item.get("pair_id"),
            "split": item.get("split"),
            "language": item.get("language"),
        }
    if actual_documents != expected_documents:
        raise RuntimeError(
            "State cache neodpovídá přesně dokumentům a jejich splitům."
        )

    cached_file_documents: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError("State cache obsahuje neplatný záznam.")
        doc_id = item.get("doc_id")
        file_name = item.get("file")
        if (
            not isinstance(doc_id, str)
            or doc_id in cached_file_documents
            or not isinstance(file_name, str)
            or Path(file_name).name != file_name
        ):
            raise RuntimeError(
                "State cache obsahuje duplicitní nebo nebezpečný záznam."
            )
        cached_file_documents.add(doc_id)
        positions = item.get("positions")
        if (
            type(positions) is not int
            or positions <= 0
            or positions > max_positions_per_document
        ):
            raise RuntimeError(
                "State cache obsahuje neplatný počet pozic dokumentu."
            )
        if (
            doc_id not in expected_documents
            or expected_documents[doc_id]
            != {
                "pair_id": item.get("pair_id"),
                "split": item.get("split"),
                "language": item.get("language"),
            }
        ):
            raise RuntimeError(
                "State cache váže soubor k jinému dokumentu nebo splitu."
            )
        path = cache_dir / file_name
        expected_bytes = item.get("bytes")
        expected_sha = item.get("file_sha256")
        if (
            not path.is_file()
            or type(expected_bytes) is not int
            or path.stat().st_size != expected_bytes
            or not isinstance(expected_sha, str)
            or sha256_file(path) != expected_sha
        ):
            raise RuntimeError(
                f"State cache selhala při ověření souboru {file_name}."
            )


class ResidualTranslator:
    """Factory wrapper that constructs a small MLX module lazily."""

    @staticmethod
    def create(
        kind: str,
        hidden_size: int,
        rank: int,
        seed: int,
    ) -> Any:
        import mlx.core as mx
        import mlx.nn as nn

        mx.random.seed(seed)

        class Translator(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = mx.zeros((hidden_size,), dtype=mx.float32)
                if kind == "low_rank_residual_v1":
                    scale = 1.0 / math.sqrt(hidden_size)
                    self.down = (
                        mx.random.normal((rank, hidden_size))
                        * scale
                    ).astype(mx.float32)
                    self.up = mx.zeros(
                        (hidden_size, rank),
                        dtype=mx.float32,
                    )
                    self.diag = mx.zeros(
                        (hidden_size,),
                        dtype=mx.float32,
                    )
                elif kind == "full_affine_residual_v1":
                    self.weight = mx.zeros(
                        (hidden_size, hidden_size),
                        dtype=mx.float32,
                    )
                elif kind != "bias_only_residual_v1":
                    raise ValueError(f"Neznámý translator: {kind}")

            def __call__(self, hidden: Any) -> Any:
                if kind == "low_rank_residual_v1":
                    low_rank = (
                        hidden @ mx.transpose(self.down)
                    ) @ mx.transpose(self.up)
                    return hidden + hidden * self.diag + low_rank + self.bias
                if kind == "full_affine_residual_v1":
                    return hidden + hidden @ mx.transpose(self.weight) + self.bias
                return hidden + self.bias

            def flat_weights(self) -> dict[str, Any]:
                result = {"bias": self.bias}
                if kind == "low_rank_residual_v1":
                    result.update(
                        {
                            "diag": self.diag,
                            "down": self.down,
                            "up": self.up,
                        }
                    )
                elif kind == "full_affine_residual_v1":
                    result["weight"] = self.weight
                return result

            def restore(
                self,
                weights: dict[str, Any],
                *,
                preserve_dtype: bool = False,
            ) -> None:
                for name, value in weights.items():
                    restored = mx.array(value)
                    if not preserve_dtype:
                        restored = restored.astype(mx.float32)
                    setattr(self, name, restored)

        return Translator()


def cached_documents(
    cache_dir: Path,
    cache_index: dict[str, Any],
    *,
    split: str,
    language: str | None = None,
) -> list[dict[str, Any]]:
    result = []
    for item in cache_index["files"]:
        if item["split"] != split:
            continue
        if language is not None and item["language"] != language:
            continue
        result.append({**item, "path": cache_dir / item["file"]})
    return result


def balanced_batches(
    documents: list[dict[str, Any]],
    *,
    layer: int,
    batch_size: int,
    epoch_seed: int,
    runtime_precision: bool = False,
) -> Iterator[tuple[Any, Any, Any, list[str], list[str]]]:
    import mlx.core as mx

    by_language = {
        language: [
            document
            for document in documents
            if document["language"] == language
        ]
        for language in LANGUAGES
    }
    rng = random.Random(epoch_seed)
    for language in LANGUAGES:
        rng.shuffle(by_language[language])
    ordered = []
    for index in range(max(map(len, by_language.values()))):
        for language in LANGUAGES:
            if index < len(by_language[language]):
                ordered.append(by_language[language][index])
    state_parts: list[Any] = []
    teacher_parts: list[Any] = []
    target_parts: list[Any] = []
    doc_ids: list[str] = []
    languages: list[str] = []
    for document in ordered:
        arrays = mx.load(str(document["path"]))
        states = arrays[f"layer_{layer:02d}"]
        teacher = arrays["teacher"]
        if not runtime_precision:
            states = states.astype(mx.float32)
            teacher = teacher.astype(mx.float32)
        targets = arrays["target_ids"]
        del arrays
        order = list(range(int(states.shape[0])))
        rng.shuffle(order)
        for position in order:
            state_parts.append(states[position])
            teacher_parts.append(teacher[position])
            target_parts.append(targets[position])
            doc_ids.append(document["doc_id"])
            languages.append(document["language"])
            if len(state_parts) == batch_size:
                yield (
                    mx.stack(state_parts),
                    mx.stack(teacher_parts),
                    mx.stack(target_parts),
                    doc_ids,
                    languages,
                )
                state_parts, teacher_parts, target_parts = [], [], []
                doc_ids, languages = [], []
    if state_parts:
        yield (
            mx.stack(state_parts),
            mx.stack(teacher_parts),
            mx.stack(target_parts),
            doc_ids,
            languages,
        )


def log_softmax(logits: Any) -> Any:
    import mlx.core as mx

    values = logits.astype(mx.float32)
    return values - mx.logsumexp(values, axis=-1, keepdims=True)


def reader_logits(norm: Any, head: Any, states: Any) -> Any:
    import mlx.core as mx

    return head(norm(states)).astype(mx.float32)


def bootstrap_interval(
    values: list[float],
    *,
    seed: int,
    draws: int = 2000,
) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        means[index] = float(
            np.mean(rng.choice(array, size=array.size, replace=True))
        )
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def evaluate_translator(
    *,
    translator: Any,
    norm: Any,
    head: Any,
    documents: list[dict[str, Any]],
    layer: int,
    batch_size: int,
    seed: int,
    runtime_precision: bool = False,
) -> dict[str, Any]:
    import mlx.core as mx

    per_document: defaultdict[str, list[dict[str, float]]] = defaultdict(list)
    document_languages: dict[str, str] = {}
    for (
        states,
        teacher_states,
        targets,
        doc_ids,
        languages,
    ) in balanced_batches(
        documents,
        layer=layer,
        batch_size=batch_size,
        epoch_seed=seed,
        runtime_precision=runtime_precision,
    ):
        raw_logp = log_softmax(reader_logits(norm, head, states))
        tuned_logp = log_softmax(
            reader_logits(norm, head, translator(states))
        )
        final_logp = mx.stop_gradient(
            log_softmax(reader_logits(norm, head, teacher_states))
        )
        final_p = mx.exp(final_logp)
        raw_p = mx.exp(raw_logp)
        tuned_p = mx.exp(tuned_logp)
        raw_kl = mx.sum(
            final_p * (final_logp - raw_logp),
            axis=-1,
        ) / math.log(2.0)
        tuned_kl = mx.sum(
            final_p * (final_logp - tuned_logp),
            axis=-1,
        ) / math.log(2.0)

        raw_mid_logp = mx.logaddexp(raw_logp, final_logp) - math.log(2.0)
        tuned_mid_logp = (
            mx.logaddexp(tuned_logp, final_logp) - math.log(2.0)
        )
        raw_js = 0.5 * (
            mx.sum(raw_p * (raw_logp - raw_mid_logp), axis=-1)
            + mx.sum(final_p * (final_logp - raw_mid_logp), axis=-1)
        ) / math.log(2.0)
        tuned_js = 0.5 * (
            mx.sum(tuned_p * (tuned_logp - tuned_mid_logp), axis=-1)
            + mx.sum(final_p * (final_logp - tuned_mid_logp), axis=-1)
        ) / math.log(2.0)
        final_top1 = mx.argmax(final_logp, axis=-1)
        raw_top1 = mx.argmax(raw_logp, axis=-1)
        tuned_top1 = mx.argmax(tuned_logp, axis=-1)
        raw_rank = 1 + mx.sum(
            raw_logp
            > mx.take_along_axis(
                raw_logp,
                final_top1[:, None],
                axis=-1,
            ),
            axis=-1,
        )
        tuned_rank = 1 + mx.sum(
            tuned_logp
            > mx.take_along_axis(
                tuned_logp,
                final_top1[:, None],
                axis=-1,
            ),
            axis=-1,
        )
        raw_entropy = -mx.sum(
            raw_p * raw_logp,
            axis=-1,
        ) / math.log(2.0)
        tuned_entropy = -mx.sum(
            tuned_p * tuned_logp,
            axis=-1,
        ) / math.log(2.0)
        final_entropy = -mx.sum(
            final_p * final_logp,
            axis=-1,
        ) / math.log(2.0)
        target_indices = targets[:, None]
        raw_nll = -mx.take_along_axis(
            raw_logp,
            target_indices,
            axis=-1,
        )[:, 0] / math.log(2.0)
        tuned_nll = -mx.take_along_axis(
            tuned_logp,
            target_indices,
            axis=-1,
        )[:, 0] / math.log(2.0)
        tensors = (
            raw_kl,
            tuned_kl,
            raw_js,
            tuned_js,
            raw_top1,
            tuned_top1,
            final_top1,
            raw_rank,
            tuned_rank,
            raw_entropy,
            tuned_entropy,
            final_entropy,
            raw_nll,
            tuned_nll,
        )
        mx.eval(*tensors)
        arrays = [np.asarray(tensor) for tensor in tensors]
        for index, (doc_id, language) in enumerate(
            zip(doc_ids, languages)
        ):
            document_languages[doc_id] = language
            per_document[doc_id].append(
                {
                    "raw_kl_bits": float(arrays[0][index]),
                    "tuned_kl_bits": float(arrays[1][index]),
                    "raw_js_bits": float(arrays[2][index]),
                    "tuned_js_bits": float(arrays[3][index]),
                    "raw_top1_agreement": float(
                        arrays[4][index] == arrays[6][index]
                    ),
                    "tuned_top1_agreement": float(
                        arrays[5][index] == arrays[6][index]
                    ),
                    "raw_final_top1_rank": float(arrays[7][index]),
                    "tuned_final_top1_rank": float(arrays[8][index]),
                    "raw_entropy_bias_bits": float(
                        arrays[9][index] - arrays[11][index]
                    ),
                    "tuned_entropy_bias_bits": float(
                        arrays[10][index] - arrays[11][index]
                    ),
                    "raw_true_token_nll_bits": float(arrays[12][index]),
                    "tuned_true_token_nll_bits": float(arrays[13][index]),
                }
            )

    document_means: dict[str, dict[str, float]] = {}
    for doc_id, rows in per_document.items():
        document_means[doc_id] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        document_means[doc_id]["positions"] = float(len(rows))

    def aggregate(language: str | None) -> dict[str, Any]:
        selected = [
            values
            for doc_id, values in document_means.items()
            if language is None or document_languages[doc_id] == language
        ]
        if not selected:
            return {"documents": 0, "positions": 0}
        result: dict[str, Any] = {
            "documents": len(selected),
            "positions": int(sum(item["positions"] for item in selected)),
        }
        for key in (
            "raw_kl_bits",
            "tuned_kl_bits",
            "raw_js_bits",
            "tuned_js_bits",
            "raw_top1_agreement",
            "tuned_top1_agreement",
            "raw_final_top1_rank",
            "tuned_final_top1_rank",
            "raw_entropy_bias_bits",
            "tuned_entropy_bias_bits",
            "raw_true_token_nll_bits",
            "tuned_true_token_nll_bits",
        ):
            result[key] = float(np.mean([item[key] for item in selected]))
        improvements = [
            item["raw_kl_bits"] - item["tuned_kl_bits"]
            for item in selected
        ]
        lower, upper = bootstrap_interval(
            improvements,
            seed=seed + (0 if language is None else 1 + LANGUAGES.index(language)),
        )
        result.update(
            {
                "raw_median_final_top1_rank": float(
                    np.median(
                        [item["raw_final_top1_rank"] for item in selected]
                    )
                ),
                "tuned_median_final_top1_rank": float(
                    np.median(
                        [item["tuned_final_top1_rank"] for item in selected]
                    )
                ),
                "kl_improvement_bits": float(np.mean(improvements)),
                "kl_relative_reduction": (
                    1.0
                    - result["tuned_kl_bits"] / result["raw_kl_bits"]
                    if result["raw_kl_bits"] > 0
                    else None
                ),
                "kl_improvement_ci95_document_bootstrap": [lower, upper],
            }
        )
        return result

    return {
        "pooled": aggregate(None),
        "by_language": {
            language: aggregate(language)
            for language in LANGUAGES
        },
    }


def train_one_layer(
    *,
    kind: str,
    hidden_size: int,
    rank: int,
    layer: int,
    norm: Any,
    head: Any,
    train_documents: list[dict[str, Any]],
    validation_documents: list[dict[str, Any]],
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    translator = ResidualTranslator.create(
        kind,
        hidden_size,
        rank,
        seed + layer,
    )
    optimizer = optim.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        bias_correction=True,
    )

    def loss_function(states: Any, teacher_states: Any) -> Any:
        tuned_logp = log_softmax(
            reader_logits(norm, head, translator(states))
        )
        teacher_logp = mx.stop_gradient(
            log_softmax(reader_logits(norm, head, teacher_states))
        )
        return nn.losses.kl_div_loss(
            tuned_logp,
            teacher_logp,
            axis=-1,
            reduction="mean",
        )

    loss_and_grad = nn.value_and_grad(translator, loss_function)
    best_loss = math.inf
    best_weights: dict[str, np.ndarray] | None = None
    history = []
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        losses = []
        for states, teacher_states, _targets, _doc_ids, _languages in (
            balanced_batches(
                train_documents,
                layer=layer,
                batch_size=batch_size,
                epoch_seed=seed + layer * 1000 + epoch,
            )
        ):
            loss, gradients = loss_and_grad(states, teacher_states)
            gradients, gradient_norm = optim.clip_grad_norm(
                gradients,
                max_norm=1.0,
            )
            optimizer.update(translator, gradients)
            mx.eval(
                loss,
                gradient_norm,
                translator.parameters(),
                optimizer.state,
            )
            losses.append(float(loss.item()))

        validation = evaluate_translator(
            translator=translator,
            norm=norm,
            head=head,
            documents=validation_documents,
            layer=layer,
            batch_size=batch_size,
            seed=seed + epoch,
        )
        validation_loss = float(
            validation["pooled"].get("tuned_kl_bits", math.inf)
        )
        history.append(
            {
                "epoch": epoch,
                "train_kl_nats": float(np.mean(losses)) if losses else None,
                "validation": validation,
            }
        )
        print(
            f"L{layer:02d} epoch {epoch:02d}: "
            f"train KL={np.mean(losses):.6f} nat · "
            f"val tuned KL={validation_loss:.6f} bit",
            flush=True,
        )
        if validation_loss + 1e-7 < best_loss:
            best_loss = validation_loss
            best_weights = {
                name: np.asarray(value).copy()
                for name, value in translator.flat_weights().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_weights is None:
        raise RuntimeError(f"L{layer}: trénink nevytvořil checkpoint.")
    translator.restore(best_weights)
    mx.eval(translator.parameters())
    return translator, {
        "best_validation_kl_bits": best_loss,
        "epochs_completed": len(history),
        "history": history,
    }


def parse_layers(value: str) -> list[int]:
    try:
        layers = sorted({int(item.strip()) for item in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Vrstvy musí být čárkou oddělená celá čísla."
        ) from exc
    if not layers or any(not 1 <= layer < REFERENCE_LAYER for layer in layers):
        raise argparse.ArgumentTypeError("Vrstvy musí ležet mezi L1 a L31.")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Dokumentový JSONL korpus.")
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Stabilní identifikátor korpusu a jeho revize.",
    )
    parser.add_argument(
        "--model-mode",
        choices=["bf16", HYBRID_MODEL_ID, "8bit", "4bit"],
        default=HYBRID_MODEL_ID,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-8bit", type=Path, default=DEFAULT_8BIT_MODEL)
    parser.add_argument(
        "--model-4bit",
        type=Path,
        default=DEFAULT_QUANTIZED_MODEL,
    )
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default=list(DEFAULT_PILOT_LAYERS),
        help="Například 4,16,17,22,23,24,31.",
    )
    parser.add_argument(
        "--translator",
        choices=("low-rank", "bias-only", "full-affine"),
        default="low-rank",
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--split-seed", type=int, default=20260728)
    parser.add_argument("--sample-seed", type=int, default=20260729)
    parser.add_argument("--training-seed", type=int, default=20260730)
    parser.add_argument("--max-positions-per-document", type=int, default=128)
    parser.add_argument("--max-documents-per-language", type=int)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "tuned_lens",
    )
    parser.add_argument("--artifact-id")
    parser.add_argument(
        "--state-cache",
        type=Path,
        help="Volitelný existující nebo nově vytvořený cache adresář.",
    )
    parser.add_argument(
        "--reuse-state-cache",
        action="store_true",
        help="Nevytahovat stavy znovu; ověřit a použít index.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_code_provenance = code_provenance()
    if not SAFE_ID_PATTERN.fullmatch(args.dataset_id):
        raise SystemExit(
            "--dataset-id smí obsahovat jen písmena, číslice, tečku, "
            "podtržítko a pomlčku."
        )
    if args.artifact_id and not SAFE_ID_PATTERN.fullmatch(args.artifact_id):
        raise SystemExit(
            "--artifact-id smí obsahovat jen písmena, číslice, tečku, "
            "podtržítko a pomlčku."
        )
    if args.rank <= 0:
        raise SystemExit("--rank musí být kladný.")
    if args.batch_size <= 0 or args.epochs <= 0 or args.patience <= 0:
        raise SystemExit("Batch, epochy a patience musí být kladné.")
    corpus_path = args.corpus.expanduser().resolve()
    documents = load_documents(
        corpus_path,
        split_seed=args.split_seed,
        max_documents_per_language=args.max_documents_per_language,
    )
    dataset = dataset_manifest(
        documents,
        dataset_id=args.dataset_id,
        source_path=corpus_path,
        split_seed=args.split_seed,
    )
    model_paths = {
        "bf16": args.model.expanduser().resolve(),
        HYBRID_MODEL_ID: args.model_8bit.expanduser().resolve(),
        "8bit": args.model_8bit.expanduser().resolve(),
        "4bit": args.model_4bit.expanduser().resolve(),
    }
    state = ModelState(
        model_path=model_paths[args.model_mode],
        model_paths=model_paths,
        active_model_id=args.model_mode,
        tuned_lens_root=args.artifact_root.expanduser().resolve(),
    )
    sources = state._model_sources_unlocked()
    metadata_digest, metadata_inventory = metadata_fingerprint(
        args.model_mode,
        sources,
    )
    fingerprint_cache = (
        args.artifact_root.expanduser().resolve()
        / "_model_fingerprints"
        / f"{args.model_mode}.json"
    )
    print("Ověřuji úplný fingerprint použitých modelových souborů…", flush=True)
    content_digest, content_inventory = model_content_fingerprint(
        args.model_mode,
        sources,
        cache_path=fingerprint_cache,
    )
    print(f"Načítám {MODEL_LABELS[args.model_mode]}…", flush=True)
    model, tokenizer = state._load_model_components(
        args.model_mode,
        model_paths[args.model_mode],
    )
    hidden_size = int(model.args.hidden_size)
    if hidden_size != 4096:
        raise RuntimeError(
            f"Pilot očekává hidden_size 4096, model má {hidden_size}."
        )
    cache_dir = (
        args.state_cache.expanduser().resolve()
        if args.state_cache
        else (
            args.artifact_root.expanduser().resolve()
            / "_state_cache"
            / (
                f"{args.dataset_id}-{args.model_mode}-"
                f"{dataset['document_manifest_sha256'][:12]}-"
                f"{content_digest[:12]}-bf16-v3"
            )
        )
    )
    cache_index_path = cache_dir / "index.json"
    if args.reuse_state_cache:
        cache_index = json.loads(cache_index_path.read_text(encoding="utf-8"))
        validate_state_cache(
            cache_dir,
            cache_index,
            documents=documents,
            layers=args.layers,
            sample_seed=args.sample_seed,
            max_positions_per_document=args.max_positions_per_document,
            model_id=args.model_mode,
            metadata_fingerprint=metadata_digest,
            content_fingerprint=content_digest,
            document_manifest_sha256=dataset[
                "document_manifest_sha256"
            ],
            source_file_sha256=dataset["source_file_sha256"],
        )
    else:
        cache_index = extract_states(
            model=model,
            tokenizer=tokenizer,
            documents=documents,
            layers=args.layers,
            output_dir=cache_dir,
            sample_seed=args.sample_seed,
            max_positions_per_document=args.max_positions_per_document,
            model_id=args.model_mode,
            metadata_fingerprint=metadata_digest,
            content_fingerprint=content_digest,
            document_manifest_sha256=dataset[
                "document_manifest_sha256"
            ],
            source_file_sha256=dataset["source_file_sha256"],
        )
    dataset["tokenization_diagnostics"] = cache_index["diagnostics"]
    dataset["state_cache_index_sha256"] = sha256_file(cache_index_path)
    dataset["sample_seed"] = args.sample_seed
    dataset["max_positions_per_document"] = (
        args.max_positions_per_document
    )

    # Keep only the exact frozen readout.  Releasing the decoder body before
    # optimization prevents a 24GB M2 from holding model, activations and Adam
    # state for multiple full translators at once.
    norm = model.model.norm
    head = model.lm_head
    del tokenizer
    model.model.layers = []
    del model
    gc.collect()
    import mlx.core as mx

    mx.clear_cache()
    mx.eval(norm.parameters(), head.parameters())

    kind = {
        "low-rank": "low_rank_residual_v1",
        "bias-only": "bias_only_residual_v1",
        "full-affine": "full_affine_residual_v1",
    }[args.translator]
    train_documents = cached_documents(
        cache_dir,
        cache_index,
        split="train",
    )
    validation_documents = cached_documents(
        cache_dir,
        cache_index,
        split="validation",
    )
    test_documents = cached_documents(
        cache_dir,
        cache_index,
        split="test",
    )
    all_weights: dict[str, Any] = {}
    training_by_layer = []
    validation_by_layer = []
    test_by_layer = []
    for layer in args.layers:
        print(f"\n=== Trénuji L{layer:02d} ===", flush=True)
        translator, training = train_one_layer(
            kind=kind,
            hidden_size=hidden_size,
            rank=args.rank,
            layer=layer,
            norm=norm,
            head=head,
            train_documents=train_documents,
            validation_documents=validation_documents,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.training_seed,
        )
        prefix = f"layer_{layer:02d}"
        converted_weights = {
            f"{prefix}.{name}": value.astype(mx.bfloat16)
            for name, value in translator.flat_weights().items()
        }
        mx.eval(*converted_weights.values())
        # Runtime storage is BF16; metrics above were computed before this
        # downcast. A publication artifact must re-evaluate the reloaded copy.
        all_weights.update(converted_weights)
        training_by_layer.append({"layer": layer, **training})
        del translator
        gc.collect()
        mx.clear_cache()

    def layer_passes(item: dict[str, Any]) -> bool:
        for language in LANGUAGES:
            metrics = item["by_language"][language]
            interval = metrics.get(
                "kl_improvement_ci95_document_bootstrap",
                [None, None],
            )
            if (
                metrics.get("kl_improvement_bits", 0.0) <= 0.0
                or interval[0] is None
                or interval[0] <= 0.0
            ):
                return False
        return True

    artifact_id = args.artifact_id or (
        f"apertus-{args.model_mode}-{args.translator}-"
        f"{args.dataset_id}-{uuid.uuid4().hex[:8]}"
    )
    artifact_dir = (
        args.artifact_root.expanduser().resolve() / artifact_id
    )
    if artifact_dir.exists():
        raise RuntimeError(
            f"Artefakt už existuje a nebude přepsán: {artifact_dir}"
        )
    artifact_dir.mkdir(parents=True)
    weight_path = artifact_dir / "translators.safetensors"
    mx.save_safetensors(str(weight_path), all_weights)
    mx.eval(*all_weights.values())
    # Acceptance is based on the exact downcast file that the server will
    # reload, not on the higher-precision in-memory checkpoint.
    reloaded_weights = mx.load(str(weight_path))
    validation_by_layer = []
    test_by_layer = []
    for layer in args.layers:
        translator = ResidualTranslator.create(
            kind,
            hidden_size,
            args.rank,
            args.training_seed + layer,
        )
        prefix = f"layer_{layer:02d}"
        translator.restore(
            {
                key.removeprefix(prefix + "."): value
                for key, value in reloaded_weights.items()
                if key.startswith(prefix + ".")
            },
            preserve_dtype=True,
        )
        mx.eval(translator.parameters())
        validation_metrics = evaluate_translator(
            translator=translator,
            norm=norm,
            head=head,
            documents=validation_documents,
            layer=layer,
            batch_size=args.batch_size,
            seed=args.training_seed + 300 + layer,
            runtime_precision=True,
        )
        test_metrics = evaluate_translator(
            translator=translator,
            norm=norm,
            head=head,
            documents=test_documents,
            layer=layer,
            batch_size=args.batch_size,
            seed=args.training_seed + 400 + layer,
            runtime_precision=True,
        )
        validation_by_layer.append(
            {
                "layer": layer,
                **validation_metrics["pooled"],
                "pooled": validation_metrics["pooled"],
                "by_language": validation_metrics["by_language"],
            }
        )
        test_by_layer.append(
            {
                "layer": layer,
                **test_metrics["pooled"],
                "pooled": test_metrics["pooled"],
                "by_language": test_metrics["by_language"],
            }
        )
        del translator
        gc.collect()
        mx.clear_cache()
    acceptance_by_layer = {
        str(item["layer"]): layer_passes(item)
        for item in test_by_layer
    }
    cache_files = cache_index["files"]
    data_floor = {
        split: {
            language: {
                "documents": sum(
                    item["split"] == split
                    and item["language"] == language
                    for item in cache_files
                ),
                "positions": sum(
                    int(item["positions"])
                    for item in cache_files
                    if item["split"] == split
                    and item["language"] == language
                ),
            }
            for language in LANGUAGES
        }
        for split in SPLITS
    }
    data_floor_passed = all(
        data_floor["train"][language]["documents"] >= 32
        and data_floor["train"][language]["positions"] >= 4096
        and data_floor["validation"][language]["documents"] >= 8
        and data_floor["test"][language]["documents"] >= 8
        for language in LANGUAGES
    )
    acceptance_passed = (
        bool(acceptance_by_layer)
        and all(acceptance_by_layer.values())
        and data_floor_passed
    )
    validation_status = "pilot" if acceptance_passed else "unvalidated"
    metrics = {
        "format": "apertus.tuned-lens-metrics",
        "schema_version": 1,
        "created_at": utc_now(),
        "training_by_layer": training_by_layer,
        "validation_by_layer": validation_by_layer,
        "test_by_layer": test_by_layer,
        "acceptance_by_layer": acceptance_by_layer,
        "data_floor": data_floor,
        "data_floor_passed": data_floor_passed,
        "acceptance_passed": acceptance_passed,
    }
    metrics_path = artifact_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "format": ARTIFACT_FORMAT,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "created_at": utc_now(),
        "reader": {
            "kind": (
                "canonical_tuned_lens"
                if kind == "full_affine_residual_v1"
                else "tuned_lens_style_probe"
            ),
            "translator_kind": kind,
            **({"rank": args.rank} if kind == "low_rank_residual_v1" else {}),
            "selected_layers": args.layers,
            "reference_layer": REFERENCE_LAYER,
            "hook": "post_decoder_block_residual",
            "layer_numbering": "one_based",
            "objective": "full_vocab_kl_final_to_reader",
            "temperature": 1.0,
            "hidden_size": hidden_size,
            "readout": "lm_head(final_rms_norm(translated_residual))",
        },
        "model": {
            "model_id": args.model_mode,
            "model_label": MODEL_LABELS[args.model_mode],
            "metadata_fingerprint": metadata_digest,
            "metadata_inventory": metadata_inventory,
            "content_fingerprint": content_digest,
            "content_inventory": content_inventory,
            "quantization_mode": args.model_mode,
        },
        "dataset": {
            key: value
            for key, value in dataset.items()
            if key != "documents"
        },
        "training": {
            "trainer_version": TRAINER_VERSION,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "training_seed": args.training_seed,
            "initialization": (
                "identity residual; random down and zero up"
                if kind == "low_rank_residual_v1"
                else "identity residual; zero correction"
            ),
            "storage_dtype": "bfloat16",
            "python_version": sys.version.split()[0],
            "mlx_version": installed_version("mlx"),
            "mlx_lm_version": installed_version("mlx-lm"),
            "code": run_code_provenance,
        },
        "validation": {
            "status": validation_status,
            "acceptance": {
                "passed": acceptance_passed,
                "criterion": (
                    "document-bootstrap 95% CI of RAW−TUNED full-vocab KL "
                    "above zero in both cs and en for every selected layer; "
                    "engineering floor 32/8/8 documents per language and "
                    "4096 train positions per language"
                ),
                "by_layer": acceptance_by_layer,
                "data_floor": data_floor,
                "data_floor_passed": data_floor_passed,
            },
            "held_out": {
                "split": "test",
                "by_layer": test_by_layer,
            },
            "selection_split": {
                "split": "validation",
                "by_layer": validation_by_layer,
            },
            "bias_only_control": "not_run"
            if kind != "bias_only_residual_v1"
            else "this_artifact",
            "shuffled_teacher_control": "not_run",
            "bf16_replication": args.model_mode == "bf16",
        },
        "weights": {
            "file": weight_path.name,
            "bytes": weight_path.stat().st_size,
            "sha256": sha256_file(weight_path),
            "storage_dtype": "bfloat16",
        },
        "metrics": {
            "file": metrics_path.name,
            "bytes": metrics_path.stat().st_size,
            "sha256": sha256_file(metrics_path),
        },
        "limitations": [
            (
                "Low-rank tuned-lens-style probe; nejde o neomezený afinní "
                "translator z původní metody."
                if kind == "low_rank_residual_v1"
                else (
                    "Bias-only kontrola, nikoli čtečka změny báze."
                    if kind == "bias_only_residual_v1"
                    else "Plný afinní tuned lens."
                )
            ),
            "Naučená čtečka aproximuje finální distribuci; není kamerou do významu.",
            "Výsledek neměří pravdu ani mínění.",
            (
                "Před silným tvrzením je nutná samostatná BF16 replikace."
                if args.model_mode != "bf16"
                else "Tento artefakt je modelově specifický pro BF16."
            ),
        ],
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    validate_manifest(
        manifest,
        artifact_dir=artifact_dir,
        verify_weights=True,
    )
    print(
        f"\nArtefakt: {artifact_dir}\n"
        f"Stav: {validation_status}\n"
        f"Held-out acceptance: {acceptance_passed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
