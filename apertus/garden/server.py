#!/usr/bin/env python3
"""Local next-token probability garden for Apertus and MLX-LM."""

from __future__ import annotations

import argparse
import filecmp
import gc
import hashlib
import ipaddress
import json
import math
import platform
import queue
import re
import subprocess
import threading
import time
import unicodedata
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

try:
    from .model_admin import AddonError, AdminError, ModelAdmin
except ImportError:
    from model_admin import AddonError, AdminError, ModelAdmin

try:
    from .registry_config import ADAPTERS, ModelRegistry, load_registry, resolve_weight_file, validate_model_files
    from .settings import cache_directory, data_directory, registry_path
except ImportError:
    from registry_config import ADAPTERS, ModelRegistry, load_registry, resolve_weight_file, validate_model_files
    from settings import cache_directory, data_directory, registry_path

try:
    from .backends import create_backend
    from .domains.steering import decoder_body as steering_decoder_body
    from .domains.representations import project_context_states, readout as representation_readout, trace_last_position
    from .model_registry import EUROLLM_ID, model_descriptor
    from .model_comparison import new_job, run_comparison, validate_request as validate_model_comparison
    from .detector_readers import JOB_TTL_SECONDS, new_job as new_detector_reader_job, run_comparison as run_detector_reader_comparison
except ImportError:
    from backends import create_backend
    from domains.steering import decoder_body as steering_decoder_body
    from domains.representations import project_context_states, readout as representation_readout, trace_last_position
    from model_registry import EUROLLM_ID, model_descriptor
    from model_comparison import new_job, run_comparison, validate_request as validate_model_comparison
    from detector_readers import JOB_TTL_SECONDS, new_job as new_detector_reader_job, run_comparison as run_detector_reader_comparison

try:
    from .activation_patching import (
        ActivationPatchingError,
        build_reference as build_patching_reference,
        entropy_bits as patching_entropy_bits,
        identity_control as summarize_patching_identity,
        recomputation_control as summarize_patching_recomputation,
        summarize_patch,
    )
    from .api.contracts import (
        MAX_CONTEXT_CONTRAST_CHARACTERS,
        MAX_CONTEXT_CONTRAST_CONTINUATION_TOKENS,
        MAX_CONTEXT_CONTRAST_PREFIX_TOKENS,
        MAX_CONTEXT_CONTRAST_TOP_K,
        MAX_CONTEXT_CONTRAST_UTF8_BYTES,
        MAX_EMBEDDING_CHARACTERS,
        MAX_EMBEDDING_CONTEXTS,
        MAX_EMBEDDING_GRAPH_NEIGHBORS,
        MAX_EMBEDDING_NEIGHBORS,
        MAX_EMBEDDING_TOKENS,
        MAX_EMBEDDING_UTF8_BYTES,
        MAX_TEXT_SURPRISAL_CHARACTERS,
        MAX_TEXT_SURPRISAL_LABEL_CHARACTERS,
        MAX_TEXT_SURPRISAL_TOKENS,
        MAX_TEXT_SURPRISAL_TOP_K,
        MAX_TEXT_SURPRISAL_UTF8_BYTES,
        MAX_TOKENIZATION_CHARACTERS,
        MAX_TOKENIZATION_TOKENS,
        MAX_TOKENIZATION_UTF8_BYTES,
        MIN_EMBEDDING_GRAPH_NEIGHBORS,
        MIN_EMBEDDING_NEIGHBORS,
        context_contrast_request_fields,
        continuation_surprisal_request_fields,
        embedding_neighborhood_request_fields,
        embeddings_request_fields,
        text_surprisal_comparison_request_fields,
        text_surprisal_request_fields,
        tokenization_request_fields,
    )
    from .api.post_routes import POST_ROUTES
    from .component_attribution import decompose_residual_contrast
    from .core.distributions import (
        distribution_distance,
        entropy_bits_from_probabilities,
        probabilities_at_temperature,
        probability_views,
        raw_distribution_metrics,
        summarize_probabilities,
    )
    from .core.performance import process_peak_rss_bytes
    from .domains.tokenization import (
        TokenizationService,
        build_tokenizer_identity,
        byte_level_token_bytes,
        encode_with_bos,
        exact_byte_offsets,
        safe_decode,
        visible_token,
    )
    from .domains.surprisal import (
        aggregation as surprisal_aggregation,
        aggregations as surprisal_aggregations,
        comparison_metric as surprisal_comparison_metric,
        comparison_metrics as surprisal_comparison_metrics,
        descriptive_statistics as surprisal_descriptive_statistics,
        linear_quantile as surprisal_linear_quantile,
        span_definitions as surprisal_span_definitions,
    )
    from .domains.teacher_forced import (
        log_probabilities as teacher_forced_log_probabilities,
        persistent_half_life,
        probability_ratio_from_log2_advantage,
        score_target_rows,
        token_likelihood,
        trapezoid_area,
        validate_logit_matrix,
    )
    from .experiments import (
        ConflictError as ExperimentConflictError,
        ExperimentError,
        ExperimentStore,
        SchemaVersionError as ExperimentSchemaVersionError,
        ValidationError as ExperimentValidationError,
    )
    from .jacobian_lens import (
        JacobianLensCatalog,
        JacobianLensChangedError,
        JacobianLensError,
        JacobianLensIncompatibleError,
        JacobianLensUnavailableError,
        RUNTIME_FINAL_TOP20_OVERLAP_MIN,
        RUNTIME_JACOBIAN_MEAN_NONINFERIORITY_MARGIN,
        RUNTIME_JACOBIAN_TOP20_OVERLAP_MEAN_MIN,
        RUNTIME_JACOBIAN_TOP20_OVERLAP_MIN,
        TransferProbeError,
        apply_jacobian,
        canonical_json_sha256 as canonical_jacobian_json_sha256,
        compare_vectors,
        runtime_probe_report,
        validate_bf16_runtime_contract,
    )
    from .tuned_lens import (
        CalibrationChangedError,
        CalibrationIncompatibleError,
        TunedLensCatalog,
        TunedLensError,
        TunedLensUnavailableError,
        apply_translator,
        canonical_json_sha256,
        model_content_fingerprint,
    )
except ImportError:  # Direct execution via `python3 server.py`.
    from activation_patching import (
        ActivationPatchingError,
        build_reference as build_patching_reference,
        entropy_bits as patching_entropy_bits,
        identity_control as summarize_patching_identity,
        recomputation_control as summarize_patching_recomputation,
        summarize_patch,
    )
    from api.contracts import (
        MAX_CONTEXT_CONTRAST_CHARACTERS,
        MAX_CONTEXT_CONTRAST_CONTINUATION_TOKENS,
        MAX_CONTEXT_CONTRAST_PREFIX_TOKENS,
        MAX_CONTEXT_CONTRAST_TOP_K,
        MAX_CONTEXT_CONTRAST_UTF8_BYTES,
        MAX_EMBEDDING_CHARACTERS,
        MAX_EMBEDDING_CONTEXTS,
        MAX_EMBEDDING_GRAPH_NEIGHBORS,
        MAX_EMBEDDING_NEIGHBORS,
        MAX_EMBEDDING_TOKENS,
        MAX_EMBEDDING_UTF8_BYTES,
        MAX_TEXT_SURPRISAL_CHARACTERS,
        MAX_TEXT_SURPRISAL_LABEL_CHARACTERS,
        MAX_TEXT_SURPRISAL_TOKENS,
        MAX_TEXT_SURPRISAL_TOP_K,
        MAX_TEXT_SURPRISAL_UTF8_BYTES,
        MAX_TOKENIZATION_CHARACTERS,
        MAX_TOKENIZATION_TOKENS,
        MAX_TOKENIZATION_UTF8_BYTES,
        MIN_EMBEDDING_GRAPH_NEIGHBORS,
        MIN_EMBEDDING_NEIGHBORS,
        context_contrast_request_fields,
        continuation_surprisal_request_fields,
        embedding_neighborhood_request_fields,
        embeddings_request_fields,
        text_surprisal_comparison_request_fields,
        text_surprisal_request_fields,
        tokenization_request_fields,
    )
    from api.post_routes import POST_ROUTES
    from component_attribution import decompose_residual_contrast
    from core.distributions import (
        distribution_distance,
        entropy_bits_from_probabilities,
        probabilities_at_temperature,
        probability_views,
        raw_distribution_metrics,
        summarize_probabilities,
    )
    from core.performance import process_peak_rss_bytes
    from domains.tokenization import (
        TokenizationService,
        build_tokenizer_identity,
        byte_level_token_bytes,
        encode_with_bos,
        exact_byte_offsets,
        safe_decode,
        visible_token,
    )
    from domains.surprisal import (
        aggregation as surprisal_aggregation,
        aggregations as surprisal_aggregations,
        comparison_metric as surprisal_comparison_metric,
        comparison_metrics as surprisal_comparison_metrics,
        descriptive_statistics as surprisal_descriptive_statistics,
        linear_quantile as surprisal_linear_quantile,
        span_definitions as surprisal_span_definitions,
    )
    from domains.teacher_forced import (
        log_probabilities as teacher_forced_log_probabilities,
        persistent_half_life,
        probability_ratio_from_log2_advantage,
        score_target_rows,
        token_likelihood,
        trapezoid_area,
        validate_logit_matrix,
    )
    from experiments import (
        ConflictError as ExperimentConflictError,
        ExperimentError,
        ExperimentStore,
        SchemaVersionError as ExperimentSchemaVersionError,
        ValidationError as ExperimentValidationError,
    )
    from jacobian_lens import (
        JacobianLensCatalog,
        JacobianLensChangedError,
        JacobianLensError,
        JacobianLensIncompatibleError,
        JacobianLensUnavailableError,
        RUNTIME_FINAL_TOP20_OVERLAP_MIN,
        RUNTIME_JACOBIAN_MEAN_NONINFERIORITY_MARGIN,
        RUNTIME_JACOBIAN_TOP20_OVERLAP_MEAN_MIN,
        RUNTIME_JACOBIAN_TOP20_OVERLAP_MIN,
        TransferProbeError,
        apply_jacobian,
        canonical_json_sha256 as canonical_jacobian_json_sha256,
        compare_vectors,
        runtime_probe_report,
        validate_bf16_runtime_contract,
    )
    from tuned_lens import (
        CalibrationChangedError,
        CalibrationIncompatibleError,
        TunedLensCatalog,
        TunedLensError,
        TunedLensUnavailableError,
        apply_translator,
        canonical_json_sha256,
        model_content_fingerprint,
    )

try:
    from .predictive_detector import (
        MAX_INPUT_CHARACTERS as MAX_PREDICTIVE_INPUT_CHARACTERS,
        MAX_INPUT_UTF8_BYTES as MAX_PREDICTIVE_INPUT_UTF8_BYTES,
        PredictiveDetector,
    )
except ImportError:  # Direct execution via `python3 server.py`.
    from predictive_detector import (
        MAX_INPUT_CHARACTERS as MAX_PREDICTIVE_INPUT_CHARACTERS,
        MAX_INPUT_UTF8_BYTES as MAX_PREDICTIVE_INPUT_UTF8_BYTES,
        PredictiveDetector,
    )


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = cache_directory() / "models/bf16"
DEFAULT_QUANTIZED_MODEL = cache_directory() / "models/4bit"
DEFAULT_8BIT_MODEL = cache_directory() / "models/8bit"
DEFAULT_DATABASE = data_directory() / "experiments.sqlite3"
DEFAULT_TUNED_LENS_ROOT = data_directory() / "tuned_lens"
DEFAULT_JACOBIAN_LENS_ROOT = data_directory() / "jacobian_lens"
INDEX_FILE = APP_DIR / "static" / "index.html"
GUIDE_FILE = APP_DIR / "static" / "guide.html"
LENS_FILE = APP_DIR / "static" / "lens.html"
TUNED_LENS_FILE = APP_DIR / "static" / "tuned.html"
JACOBIAN_LENS_FILE = APP_DIR / "static" / "jlens.html"
COMPONENT_ATTRIBUTION_FILE = APP_DIR / "static" / "attribution.html"
ACTIVATION_PATCHING_FILE = APP_DIR / "static" / "patching.html"
CONTEXT_CONTRAST_FILE = APP_DIR / "static" / "contrast.html"
TOKENIZATION_FILE = APP_DIR / "static" / "tokenization.html"
EMBEDDINGS_FILE = APP_DIR / "static" / "embedding.html"
STEERING_FILE = APP_DIR / "static" / "steering.html"
SURPRISE_FILE = APP_DIR / "static" / "surprise.html"
PREDICTIVE_DETECTOR_FILE = APP_DIR / "static" / "detector.html"
ARCHIVE_FILE = APP_DIR / "static" / "archive.html"
MODEL_COMPARISON_FILE = APP_DIR / "static" / "model-comparison.html"
MODEL_COMPARISON_JS_FILE = APP_DIR / "static" / "model-comparison.js"
STEERING_COMPARISON_JS_FILE = APP_DIR / "static" / "steering-comparison.js"
DEFAULT_EUROLLM_MODEL = cache_directory() / "models/eurollm9"
WORKBENCH_SHELL_CSS_FILE = APP_DIR / "static" / "workbench-shell.css"
WORKBENCH_SHELL_JS_FILE = APP_DIR / "static" / "workbench-shell.js"
API_CLIENT_JS_FILE = APP_DIR / "static" / "api-client.js"
I18N_CATALOG_JS_FILE = APP_DIR / "static" / "i18n-catalog.generated.js"
I18N_JS_FILE = APP_DIR / "static" / "i18n.js"
MAX_REQUEST_BYTES = 1_000_000
MAX_EXPERIMENT_REQUEST_BYTES = 20_000_000
MAX_IMPORT_REQUEST_BYTES = 25_000_000
MAX_INPUT_TOKENS = 2_048
MAX_LENS_INPUT_TOKENS = 256
EMBEDDING_SEARCH_CHUNK_SIZE = 2_048
LOGIT_CACHE_SIZE = 64
LENS_CACHE_SIZE = 8
TUNED_LENS_CACHE_SIZE = 8
JACOBIAN_LENS_CACHE_SIZE = 8
COMPONENT_ATTRIBUTION_CACHE_SIZE = 8
ACTIVATION_PATCHING_CACHE_SIZE = 2
CONTEXT_CONTRAST_CACHE_SIZE = 8
EMBEDDING_CACHE_SIZE = 8
EMBEDDING_NEIGHBORHOOD_CACHE_SIZE = 64
MODEL_BOUND_CACHE_FIELDS = (
    "logits_cache",
    "lens_cache",
    "tuned_lens_cache",
    "jacobian_lens_cache",
    "component_attribution_cache",
    "activation_patching_cache",
    "context_contrast_cache",
    "embedding_cache",
    "embedding_neighborhood_cache",
    "jacobian_probe_report_cache",
    "model_content_fingerprint_cache",
)
MODEL_WORK_QUEUE_SIZE = 8
DEFAULT_MODEL_JOB_TIMEOUT_SECONDS = 300
STEERING_CONTROL_JOB_TIMEOUT_SECONDS = 3_600
SEARCH_RESULT_LIMIT = 24
MAX_PINNED_TOKENS = 64
LOOKUP_TOKEN_LIMIT = 32
COMPARE_MOVER_LIMIT = 24
COMPARE_OVERLAP_LIMIT = 100
MIN_STEERING_PAIRS = 3
MAX_STEERING_PAIRS = 8
MAX_STEERING_INPUT_TOKENS = 256
MAX_STEERING_TOTAL_TOKENS = 2_048
MAX_STEERING_TRAJECTORY_TOKENS = 64
MAX_STEERING_PHRASE_PAIRS = 12
MIN_STEERING_SHAM_SAMPLES = 2
MAX_STEERING_SHAM_SAMPLES = 99
MIN_STEERING_SWEEP_LAYERS = 2
MAX_STEERING_SWEEP_LAYERS = 12
MAX_ACTIVATION_PATCHING_TOKENS = 64
MAX_ACTIVATION_PATCHING_POSITIONS = 32
MAX_ACTIVATION_PATCHING_CELLS = 1_024
ACTIVATION_PATCHING_BATCH_SIZE = 4
ACTIVATION_PATCHING_SOURCE_CONTROL_INDEX = ACTIVATION_PATCHING_BATCH_SIZE
ACTIVATION_PATCHING_TARGET_CONTROL_INDEX = (
    ACTIVATION_PATCHING_BATCH_SIZE + 1
)
ACTIVATION_PATCHING_FIXED_BATCH_SIZE = (
    ACTIVATION_PATCHING_BATCH_SIZE + 2
)
ACTIVATION_PATCHING_IDENTITY_LOGIT_TOLERANCE = 5e-3
# Mean log-probabilities are reported in natural-log units per token.
# A smaller epsilon would overstate numerical dust as a directional effect,
# especially for quantized variants.
STEERING_MONOTONICITY_TOLERANCE = 1e-4
# BF16 has discrete norm plateaus: post-cast iterative rescaling cannot
# reliably reach 1e-5 in 4096 dimensions. This still limits residual
# norm mismatch to one hundredth of one percent.
STEERING_EFFECTIVE_NORM_RELATIVE_TOLERANCE = 1e-4
HYBRID_MODEL_ID = "hybrid"
SERVER_VERSION = "ApertusGarden/1.12"
MODEL_METADATA_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
)
HYBRID_BF16_WEIGHTS = (
    "model.embed_tokens.weight",
    "lm_head.weight",
)
MODEL_LABELS = {
    "bf16": "BF16 · reference",
    HYBRID_MODEL_ID: "Hybrid · 8bit/BF16",
    "8bit": "8bit · vyvážený",
    "4bit": "4bit · rychlý",
    "mock": "Vývojová simulace",
    "mock-eurollm9": "EuroLLM · vývojová simulace",
    EUROLLM_ID: "EuroLLM 9B",
}

MOCK_WORDS = [
    " invaze",
    " válka",
    " konflikt",
    " operace",
    " která",
    " Ruska",
    " na",
    " proti",
    " pokračuje",
    " začala",
    " znamená",
    " podle",
    ".",
    ",",
    "\n",
    " v",
    " a",
    " že",
    " se",
    " je",
    " byla",
    " jsou",
    " tento",
    " však",
    " také",
    " vojenská",
    " rozsáhlá",
    " území",
    " Ukrajině",
    " útok",
    " agrese",
    " krize",
]


def installed_package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def source_control_provenance() -> dict[str, Any]:
    """Capture the repository revision once, without exposing a local path."""
    repository_root = APP_DIR.parents[1]
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("invalid git revision")
        return {
            "available": True,
            "vcs": "git",
            "commit_sha": revision,
            "dirty": bool(status.strip()),
            "repository_label": repository_root.name,
            "captured_at_server_start": True,
        }
    except (
        FileNotFoundError,
        subprocess.SubprocessError,
        ValueError,
    ):
        return {
            "available": False,
            "vcs": "git",
            "commit_sha": None,
            "dirty": None,
            "repository_label": repository_root.name,
            "captured_at_server_start": True,
        }


RUNTIME_PROVENANCE = {
    "server_version": SERVER_VERSION,
    "python_version": platform.python_version(),
    "numpy_version": np.__version__,
    "mlx_version": installed_package_version("mlx"),
    "mlx_lm_version": installed_package_version("mlx-lm"),
    "torch_version": installed_package_version("torch"),
    "transformers_version": installed_package_version("transformers"),
    "analysis_source_sha256": {
        name: hashlib.sha256((APP_DIR / name).read_bytes()).hexdigest()
        for name in ("server.py", "backends.py", "model_registry.py", "registry_config.py", "model_admin.py", "model_setup.py", "experimental_addons.py", "predictive_detector.py", "settings.py", "model_comparison.py", "steering_comparison.py", "detector_readers.py", "domains/steering.py", "domains/tokenization.py", "domains/teacher_forced.py", "domains/representations.py")
    },
    "source_control": source_control_provenance(),
}


def metadata_fingerprint(
    model_id: str,
    sources: list[tuple[str, Path]],
) -> tuple[str, list[dict[str, Any]]]:
    """Fingerprint model metadata without reading multi-gigabyte weight shards."""
    digest = hashlib.sha256()
    digest.update(b"apertus-garden-model-metadata-v1\0")
    digest.update(model_id.encode("utf-8"))
    manifest: list[dict[str, Any]] = []
    for role, model_path in sources:
        digest.update(b"\0role\0")
        digest.update(role.encode("utf-8"))
        digest.update(b"\0directory\0")
        digest.update(model_path.name.encode("utf-8"))
        source_manifest: dict[str, Any] = {
            "role": role,
            "directory": model_path.name,
            "metadata_files": [],
        }
        for filename in MODEL_METADATA_FILES:
            path = model_path / filename
            if not path.is_file():
                continue
            try:
                contents = path.read_bytes()
            except OSError:
                continue
            file_digest = hashlib.sha256(contents).hexdigest()
            digest.update(b"\0file\0")
            digest.update(filename.encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_digest.encode("ascii"))
            source_manifest["metadata_files"].append(
                {
                    "name": filename,
                    "bytes": len(contents),
                    "sha256": file_digest,
                }
            )

            if filename != "model.safetensors.index.json":
                continue
            try:
                index = json.loads(contents)
                shard_names = sorted(
                    {
                        str(shard_name)
                        for shard_name in index.get("weight_map", {}).values()
                    }
                )
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                shard_names = []
            shards = []
            for shard_name in shard_names:
                shard_path = model_path / shard_name
                try:
                    shard_size = shard_path.stat().st_size
                except OSError:
                    continue
                shards.append({"name": shard_name, "bytes": shard_size})
                digest.update(b"\0shard\0")
                digest.update(shard_name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(shard_size).encode("ascii"))
            if shards:
                source_manifest["weight_shards"] = shards
        manifest.append(source_manifest)
    return digest.hexdigest(), manifest


class LogitsNotCachedError(RuntimeError):
    """Raised when a read-only distribution lookup targets an evicted node."""

    def __init__(self, message: str, *, side: str | None = None):
        super().__init__(message)
        self.side = side


class ModelChangedError(RuntimeError):
    """Raised when a request targets distributions from another model generation."""


def extract_logits(output: Any) -> Any:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def normalized_token_for_search(token: str) -> str:
    return (
        token.replace("▁", " ")
        .replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace("␠", " ")
        .casefold()
    )


def aggregate_steering_deltas(
    deltas: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a robust direction from pair deltas without magnitude dominance."""
    matrix = np.asarray(deltas, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(
            "Kontrastní rozdíly musí tvořit neprázdnou matici páry × dimenze."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Kontrastní rozdíly obsahují neplatné hodnoty.")

    pair_norms = np.linalg.norm(matrix, axis=1)
    if np.any(pair_norms <= np.finfo(np.float64).eps):
        raise ValueError(
            "Alespoň jeden kontrastní pár má nulový reprezentační rozdíl."
        )

    unit_directions = matrix / pair_norms[:, None]
    mean_unit_direction = np.mean(unit_directions, axis=0)
    resultant_length = float(np.linalg.norm(mean_unit_direction))
    if resultant_length <= 1e-8:
        raise ValueError(
            "Kontrastní páry se navzájem téměř přesně ruší; "
            "společný steeringový směr nelze určit."
        )

    unit_direction = mean_unit_direction / resultant_length
    native_scale = float(np.median(pair_norms))
    pair_cosines = unit_directions @ unit_direction
    return unit_direction.astype(np.float64, copy=False), {
        "aggregation": "normalized_pair_mean_with_median_native_scale",
        "native_scale": native_scale,
        "pair_norms": [float(value) for value in pair_norms],
        "pair_cosines_to_direction": [
            float(value) for value in pair_cosines
        ],
        "resultant_length": resultant_length,
        "cancellation_fraction": max(0.0, 1.0 - resultant_length),
    }


@dataclass
class ModelState:
    model_path: Path
    mock: bool = False
    model_paths: dict[str, Path] = field(default_factory=dict)
    backend_name: str = "mlx"
    device: str = "auto"
    registry: ModelRegistry | None = None
    backend: Any = field(default=None, init=False)
    model_descriptors: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    comparison_jobs: OrderedDict[str, dict[str, Any]] = field(default_factory=OrderedDict, init=False)
    detector_reader_jobs: OrderedDict[str, dict[str, Any]] = field(default_factory=OrderedDict, init=False)
    comparison_store: Any = field(default=None, init=False)
    tuned_lens_root: Path = DEFAULT_TUNED_LENS_ROOT
    jacobian_lens_root: Path = DEFAULT_JACOBIAN_LENS_ROOT
    active_model_id: str = "bf16"
    status: str = "loading"
    error: str | None = None
    model: Any = None
    tokenizer: Any = None
    cache_lock: threading.Lock = field(default_factory=threading.Lock)
    logits_cache: OrderedDict[tuple[int, ...], np.ndarray] = field(
        default_factory=OrderedDict
    )
    lens_cache: OrderedDict[tuple[int, str, int], dict[str, Any]] = field(
        default_factory=OrderedDict
    )
    tuned_lens_cache: OrderedDict[
        tuple[int, str, str, str, str, int], dict[str, Any]
    ] = field(default_factory=OrderedDict)
    jacobian_lens_cache: OrderedDict[
        tuple[int, str, str, str, str, int, tuple[int, ...]], dict[str, Any]
    ] = field(default_factory=OrderedDict)
    component_attribution_cache: OrderedDict[
        tuple[int, str, int, int], dict[str, Any]
    ] = field(default_factory=OrderedDict)
    activation_patching_cache: OrderedDict[
        tuple[Any, ...], dict[str, Any]
    ] = field(default_factory=OrderedDict)
    context_contrast_cache: OrderedDict[
        tuple[int, str, str, str, str, int], dict[str, Any]
    ] = field(default_factory=OrderedDict)
    embedding_cache: OrderedDict[
        tuple[Any, ...],
        dict[str, Any],
    ] = field(default_factory=OrderedDict)
    embedding_neighborhood_cache: OrderedDict[
        tuple[int, str, int, int],
        dict[str, Any],
    ] = field(default_factory=OrderedDict)
    jacobian_probe_report_cache: dict[
        tuple[str, str, str], dict[str, Any]
    ] = field(default_factory=dict)
    work_queue: queue.Queue[Any] = field(
        default_factory=lambda: queue.Queue(maxsize=MODEL_WORK_QUEUE_SIZE)
    )
    worker_thread: threading.Thread | None = None
    state_lock: threading.RLock = field(default_factory=threading.RLock)
    active_ids: list[int] | None = None
    active_cache: Any = None
    active_logits: np.ndarray | None = None
    vocab_catalog: list[tuple[int, str, str]] | None = None
    phase: str | None = "loading"
    target_model_id: str | None = None
    model_generation: int = 0
    transition_id: int = 0
    server_instance_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    hybrid_weight_paths: dict[str, Path] = field(default_factory=dict)
    model_provenance_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    tuned_lens_catalog: TunedLensCatalog | None = None
    jacobian_lens_catalog: JacobianLensCatalog | None = None
    model_content_fingerprint_cache: dict[str, str] = field(
        default_factory=dict
    )
    tokenizer_identity_cache: dict[str, Any] | None = None
    active_operation: str | None = None
    active_operation_started_monotonic: float | None = None
    outstanding_operation: str | None = None
    outstanding_operation_started_monotonic: float | None = None
    predictive_detector_engine: PredictiveDetector | None = None
    load_initial_model: bool = field(default=True, init=False)
    experimental_addons: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.backend = create_backend(self.backend_name, self.device)
        if self.mock:
            self.active_model_id = "mock"
            self.model_paths = {"mock": self.model_path, "mock-eurollm9": self.model_path}
        elif not self.model_paths:
            self.model_paths = {self.active_model_id: self.model_path}
        if not self.mock:
            self.hybrid_weight_paths = self._resolve_hybrid_weight_paths()
        self.tuned_lens_catalog = TunedLensCatalog(
            self.tuned_lens_root.expanduser().resolve()
        )
        self.jacobian_lens_catalog = JacobianLensCatalog(
            self.jacobian_lens_root.expanduser().resolve()
        )

    def model_descriptor(self, model_id: str) -> dict[str, Any]:
        if model_id not in self.model_descriptors:
            self.model_descriptors[model_id] = model_descriptor(
                model_id, self.model_paths[model_id], self.model_label(model_id, model_id),
                backend=self.backend_name, mock=self.mock,
                registration=self.registry.models.get(model_id) if self.registry else None,
                enable_research_artifacts=(self.registry.enable_research_artifacts if self.registry else True) and self.experimental_addons is None,
            )
            if self.registry:
                self.model_descriptors[model_id]["checkpoint"] = self.registry.source_identity(model_id)
        descriptor = self.model_descriptors[model_id]
        if self.experimental_addons is not None:
            availability = self.experimental_addons.availability(model_id)
            descriptor = {**descriptor, "capabilities": [item for item in descriptor["capabilities"] if item not in availability] +
                          [kind for kind, item in availability.items() if item["available"]], "experimental_addons": availability}
        return descriptor

    def refresh_experimental_addons(self, *, initial=False) -> None:
        """Bind local artifact directories while idle/on the numerical worker."""
        with self.state_lock:
            manager = self.experimental_addons
            self.tuned_lens_catalog = TunedLensCatalog(manager.root("tuned") or self.tuned_lens_root)
            self.jacobian_lens_catalog = JacobianLensCatalog(manager.root("jlens") or self.jacobian_lens_root)
            self.predictive_detector_engine = None
            self.model_descriptors.clear()
            self.model_provenance_cache.clear()
            with self.cache_lock:
                self.tuned_lens_cache.clear()
                self.jacobian_lens_cache.clear()
                self.jacobian_probe_report_cache.clear()
            if not initial:
                self.model_generation += 1

    def model_label(self, model_id: str, fallback: str | None = None) -> str:
        if self.registry and model_id in self.registry.models:
            return self.registry.models[model_id].label
        return MODEL_LABELS.get(model_id, fallback or model_id)

    def is_quantized(self) -> bool:
        return not self.mock and (self.active_model_id == HYBRID_MODEL_ID or bool(self.model_descriptor(self.active_model_id).get("quantization")))

    def apply_registry(self, registry: ModelRegistry) -> None:
        """Apply validated metadata only while idle; never reload weights on HTTP threads."""
        with self.state_lock:
            if self.outstanding_operation is not None or self.status in {"loading", "switching", "maintenance"}:
                raise RuntimeError("The model worker is busy.")
            if self.model is not None and self.registry:
                before = self.registry.models[self.active_model_id]
                after = registry.models.get(self.active_model_id)
                if registry.backend != self.backend_name or after is None or not after.enabled or (before.path,before.adapter,before.kind) != (after.path,after.adapter,after.kind):
                    raise ValueError("The loaded model cannot be replaced by a registry edit.")
            self.registry = registry
            if self.backend_name != registry.backend:
                self.backend_name = registry.backend
                self.backend = create_backend(self.backend_name, self.device)
            self.model_paths = registry.model_paths()
            if self.active_model_id not in self.model_paths or self.model is None:
                self.active_model_id = registry.default_model
            self.model_path = self.model_paths[self.active_model_id]
            self.model_descriptors.clear()
            self.model_provenance_cache.clear()
            self.model_content_fingerprint_cache.clear()
            self.hybrid_weight_paths = self._resolve_hybrid_weight_paths()
            with self.cache_lock:
                for name in MODEL_BOUND_CACHE_FIELDS:
                    getattr(self, name).clear()
            self.active_ids = self.active_cache = self.active_logits = self.tokenizer_identity_cache = None
            self.model_generation += 1
            if self.model is None:
                self.status, self.phase, self.error = "setup_required", None, None

    def enqueue_model_admin(self, callback: Any) -> None:
        with self.state_lock:
            if self.outstanding_operation is not None or self.status in {"loading", "switching", "maintenance"}:
                raise RuntimeError("The model worker is busy.")
            previous = (self.status, self.phase)
            self.status, self.phase = "maintenance", "queued"
            self.outstanding_operation = "model_admin"
            self.outstanding_operation_started_monotonic = time.monotonic()
            try:
                self.work_queue.put_nowait(("model_admin", (callback,), None, None))
            except queue.Full:
                self.status, self.phase = previous
                self.outstanding_operation = self.outstanding_operation_started_monotonic = None
                raise RuntimeError("The model worker queue is full.")

    def require_capability(self, capability: str) -> None:
        with self.state_lock:
            if capability not in self.model_descriptor(self.active_model_id)["capabilities"]:
                raise ValueError(f"Aktivní model nepodporuje tento pohled ({capability}).")

    def start_model_comparison(self, request: dict[str, Any]) -> dict[str, Any]:
        checked = validate_model_comparison(request)
        with self.state_lock:
            if self.status != "ready" or self.outstanding_operation is not None:
                raise RuntimeError("Výpočetní část právě pracuje; počkej na dokončení.")
            for model_id in checked["model_ids"]:
                if model_id not in self.model_paths or not self._model_available_unlocked(model_id, self.model_paths[model_id]):
                    raise ValueError(f"Model není dostupný: {model_id}")
                if "steering" in checked and "steering" not in self.model_descriptor(model_id)["capabilities"]:
                    raise ValueError(f"Model nepodporuje steering: {model_id}")
            # Keep bounded progress state. Canonical results are retained in SQLite.
            while len(self.comparison_jobs) >= 8:
                self.comparison_jobs.popitem(last=False)
            job = new_job(checked, self.server_instance_id)
            self.comparison_jobs[job["id"]] = job
            self.outstanding_operation = "model_comparison"
            self.outstanding_operation_started_monotonic = time.monotonic()
            try:
                self.work_queue.put_nowait(("model_comparison", (job,), None, None))
            except queue.Full:
                self.outstanding_operation = None
                self.outstanding_operation_started_monotonic = None
                del self.comparison_jobs[job["id"]]
                raise RuntimeError("Výpočetní fronta je plná.")
            return self.model_comparison_status(job["id"])

    def model_comparison_status(self, job_id: str, *, cancel: bool = False) -> dict[str, Any]:
        with self.state_lock:
            if job_id not in self.comparison_jobs:
                raise ValueError("Srovnávací běh nebyl nalezen; uložené výsledky najdeš v archivu.")
            job = self.comparison_jobs[job_id]
            if cancel and job["phase"] != "done":
                job["cancel"].set()
            keys = ("id", "status", "phase", "current_model_id", "completed_measurements", "total_measurements",
                    "errors", "saved", "storage_error", "restoration_error", "record")
            # Detach data before the HTTP thread serializes it.
            return json.loads(json.dumps({key: job[key] for key in keys if key in job}))

    def _prune_detector_reader_jobs_unlocked(self) -> None:
        now = time.monotonic()
        for job_id, job in list(self.detector_reader_jobs.items()):
            if job["phase"] == "done" and now - job["created_monotonic"] > JOB_TTL_SECONDS:
                del self.detector_reader_jobs[job_id]

    def start_detector_reader_comparison(
        self, text: str, expected_model_id: str, expected_generation: int,
        expected_server_instance_id: str,
    ) -> dict[str, Any]:
        with self.state_lock:
            if self.status != "ready" or self.outstanding_operation is not None:
                raise RuntimeError("Výpočetní část právě pracuje; počkej na dokončení.")
            self._validate_request_identity_unlocked(expected_generation, expected_server_instance_id, expected_model_id)
            required_model = "mock" if self.mock else HYBRID_MODEL_ID
            euro_model = "mock-eurollm9" if self.mock else EUROLLM_ID
            if self.active_model_id != required_model:
                raise ValueError("AI detektor vyžaduje model Hybrid; EuroLLM slouží jako paralelní měření.")
            if euro_model not in self.model_paths or not self._model_available_unlocked(euro_model, self.model_paths[euro_model]):
                raise ValueError("EuroLLM 9B není dostupný. Spusť analýzu bez jeho měření.")
            self._prune_detector_reader_jobs_unlocked()
            while len(self.detector_reader_jobs) >= 8:
                self.detector_reader_jobs.popitem(last=False)
            job = new_detector_reader_job(text, {
                "model_id": expected_model_id, "model_generation": expected_generation,
                "server_instance_id": expected_server_instance_id,
            }, euro_model)
            self.detector_reader_jobs[job["id"]] = job
            self.outstanding_operation = "detector_readers"
            self.outstanding_operation_started_monotonic = time.monotonic()
            try:
                self.work_queue.put_nowait(("detector_readers", (job,), None, None))
            except queue.Full:
                self.outstanding_operation = None
                self.outstanding_operation_started_monotonic = None
                del self.detector_reader_jobs[job["id"]]
                raise RuntimeError("Výpočetní fronta je plná.")
            return self.detector_reader_status(job["id"])

    def detector_reader_status(self, job_id: str, *, cancel: bool = False) -> dict[str, Any]:
        with self.state_lock:
            self._prune_detector_reader_jobs_unlocked()
            if job_id not in self.detector_reader_jobs:
                raise ValueError("Měření detektoru nebylo nalezeno nebo vypršelo. Výsledky se neukládají.")
            job = self.detector_reader_jobs[job_id]
            if cancel and job["phase"] != "done":
                job["cancel"].set()
            keys = ("id", "status", "phase", "request_identity", "input_sha256",
                    "completed_measurements", "total_measurements", "errors", "result",
                    "restored_identity", "restoration_error")
            return json.loads(json.dumps({key: job[key] for key in keys if key in job}))

    @property
    def model_name(self) -> str:
        if self.mock:
            return "Apertus 8B (mock)"
        if self.active_model_id == HYBRID_MODEL_ID:
            return "Apertus 8B (8bit/BF16 hybrid)"
        if self.registry:
            return self.model_label(self.active_model_id)
        return self.model_path.name

    def _model_provenance_unlocked(self) -> dict[str, Any]:
        cached = self.model_provenance_cache.get(self.active_model_id)
        if cached is not None:
            return cached
        provenance = self._model_provenance_snapshot_unlocked()
        self.model_provenance_cache[self.active_model_id] = provenance
        return provenance

    def _model_provenance_snapshot_unlocked(self) -> dict[str, Any]:
        """Build provenance without changing any request-global cache."""
        sources = self._model_sources_unlocked()
        fingerprint, manifest = metadata_fingerprint(
            self.active_model_id,
            sources,
        )
        provenance = {
            "descriptor": self.model_descriptor(self.active_model_id),
            "model_id": self.active_model_id,
            "model_label": self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            "metadata_fingerprint": fingerprint,
            "fingerprint_scope": (
                "mock_runtime"
                if self.mock
                else "metadata_and_weight_manifest"
            ),
            "sources": manifest,
            "backend": "mock" if self.mock else self.backend_name,
        }
        if self.registry:
            provenance["registry_sha256"] = self.registry.sha256
        if not self.mock and self.model is not None:
            if self.backend_name == "torch":
                provenance["execution_precision"] = {"floating_dtype": str(next(self.model.parameters()).dtype), "device": str(self.model.device)}
            else:
                norm = getattr(getattr(self.model, "model", None), "norm", None)
                provenance["execution_precision"] = {"floating_dtype": str(getattr(getattr(norm, "weight", None), "dtype", "unknown")), "device": "mlx_default_device"}
        if self.mock:
            provenance["runtime_composition"] = {"kind": "mock"}
            return provenance

        quantized_path = (
            self.model_paths.get(HYBRID_MODEL_ID)
            if self.active_model_id == HYBRID_MODEL_ID
            else self.model_paths.get(self.active_model_id)
        )
        quantization: dict[str, Any] | None = None
        if quantized_path is not None:
            try:
                config = json.loads(
                    (quantized_path / "config.json").read_text()
                )
                candidate = config.get("quantization_config")
                if candidate is None:
                    candidate = config.get("quantization")
                if isinstance(candidate, dict):
                    quantization = {
                        key: candidate[key]
                        for key in ("bits", "group_size", "mode")
                        if key in candidate
                    }
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                quantization = None

        if self.active_model_id == HYBRID_MODEL_ID:
            provenance["runtime_composition"] = {
                "kind": "quantized_body_with_bf16_input_output",
                "quantized_body": {
                    "source_role": "quantized_body",
                    "quantization": quantization,
                },
                "bf16_input_output": {
                    "source_role": "bf16_io",
                    "weights": list(HYBRID_BF16_WEIGHTS),
                },
                "statement": (
                    "Transformer body from the quantized source; exact BF16 "
                    "input embedding and output head from the BF16 source."
                ),
            }
        elif quantization is not None:
            provenance["runtime_composition"] = {
                "kind": "quantized_model",
                "quantization": quantization,
            }
        else:
            provenance["runtime_composition"] = {
                "kind": "unquantized_model"
            }
        return provenance

    def _model_sources_unlocked(self) -> list[tuple[str, Path]]:
        if self.mock:
            return []
        if self.active_model_id == HYBRID_MODEL_ID:
            return [
                ("quantized_body", self.model_paths[HYBRID_MODEL_ID]),
                ("bf16_io", self.registry.models["bf16"].path if self.registry else self.model_paths["bf16"]),
            ]
        return [("model", self.model_paths[self.active_model_id])]

    def _tuned_lens_status_unlocked(self) -> dict[str, Any]:
        if self.tuned_lens_catalog is None:
            return {
                "status": "missing",
                "message": "Tuned-lens katalog není inicializován.",
                "content_verification": "not_started",
            }
        provenance = self._model_provenance_unlocked()
        return self.tuned_lens_catalog.status(
            model_id=self.active_model_id,
            metadata_fingerprint=provenance["metadata_fingerprint"],
            mock=self.mock,
        )

    def tuned_lens_status(self) -> dict[str, Any]:
        with self.state_lock:
            response = self._tuned_lens_status_unlocked()
            response.update(
                {
                    "model_id": self.active_model_id,
                    "model_generation": self.model_generation,
                    "server_instance_id": self.server_instance_id,
                }
            )
            return response

    def _jacobian_lens_status_unlocked(
        self,
        expected_artifact_id: str | None = None,
    ) -> dict[str, Any]:
        if self.mock:
            artifact = {
                "artifact_id": "mock-average-jacobian-v1",
                "artifact_scope": "synthetic_ui_only",
                "canonical_manifest_sha256": "d" * 64,
                "compatibility": {
                    "allowed_model_modes": ["bf16"],
                    "hybrid_transfer": "not_validated",
                    "mlx_enablement_requires_transfer_probe_match": True,
                },
                "reader": {
                    "kind": "average_jacobian_lens",
                    "source_layers": [4, 16, 17, 22, 23, 24, 31],
                    "target_layer": 32,
                    "transport": "residual @ jacobian.T",
                    "readout": (
                        "lm_head(final_rms_norm(transported_residual))"
                    ),
                },
                "model": {"mode": "bf16", "hf_repo": "mock"},
                "weights": {
                    "sha256": "0" * 64,
                    "bytes": 1,
                    "dtype": "float16",
                },
                "limitations": [
                    "Pouze syntetická simulace uživatelského rozhraní."
                ],
            }
            if (
                expected_artifact_id is not None
                and expected_artifact_id != artifact["artifact_id"]
            ):
                return {
                    "status": "missing",
                    "message": "Požadovaný J-Lens artefakt není nainstalován.",
                    "content_verification": "not_started",
                    "artifact_id": expected_artifact_id,
                    "artifacts": [artifact],
                }
            return {
                "status": "ready",
                "message": "Vývojová simulace J-Lens je připravena.",
                "content_verification": "mock",
                "artifact_id": artifact["artifact_id"],
                "source_layers": artifact["reader"]["source_layers"],
                "target_layer": artifact["reader"]["target_layer"],
                "artifact": artifact,
                "calibration": artifact,
                "artifacts": [artifact],
            }
        if self.jacobian_lens_catalog is None:
            return {
                "status": "missing",
                "message": "J-Lens katalog není inicializován.",
                "content_verification": "not_started",
            }
        model_fingerprint = self.model_content_fingerprint_cache.get(
            f"{self.active_model_id}:"
            f"{self._model_provenance_unlocked()['metadata_fingerprint']}"
        )
        response = self.jacobian_lens_catalog.status(
            model_id=self.active_model_id,
            mock=self.mock,
            model_fingerprint=model_fingerprint,
            expected_artifact_id=expected_artifact_id,
        )
        artifact = response.get("calibration")
        if isinstance(artifact, dict) and model_fingerprint is not None:
            key = (
                str(artifact.get("artifact_id", "")),
                str(artifact.get("weights", {}).get("sha256", "")),
                model_fingerprint,
            )
            report = self.jacobian_probe_report_cache.get(key)
            if report is not None:
                response["transfer_probe"] = report
        return response

    def jacobian_lens_status(
        self,
        expected_artifact_id: str | None = None,
    ) -> dict[str, Any]:
        with self.state_lock:
            response = self._jacobian_lens_status_unlocked(
                expected_artifact_id
            )
            response.update(
                {
                    "model_id": self.active_model_id,
                    "model_generation": self.model_generation,
                    "server_instance_id": self.server_instance_id,
                }
            )
            return response

    def _resolve_hybrid_weight_paths(self) -> dict[str, Path]:
        bf16_path = self.registry.models["bf16"].path if self.registry and "bf16" in self.registry.models else self.model_paths.get("bf16")
        quantized_path = self.registry.models["8bit"].path if self.registry and "8bit" in self.registry.models else self.model_paths.get("8bit")
        if bf16_path is None or quantized_path is None:
            return {}
        bf16_path = bf16_path.resolve()
        quantized_path = quantized_path.resolve()
        if not self._hybrid_sources_compatible(bf16_path, quantized_path):
            return {}
        index_path = bf16_path / "model.safetensors.index.json"
        if not index_path.exists():
            return {}
        try:
            index = json.loads(index_path.read_text())
            weight_map = index["weight_map"]
            resolved: dict[str, Path] = {}
            for weight_name in HYBRID_BF16_WEIGHTS:
                shard_path = resolve_weight_file(bf16_path, weight_map[weight_name])
                if not shard_path.exists():
                    return {}
                resolved[weight_name] = shard_path
            return resolved
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _hybrid_sources_compatible(
        bf16_path: Path,
        quantized_path: Path,
    ) -> bool:
        try:
            bf16_config = json.loads((bf16_path / "config.json").read_text())
            quantized_config = json.loads(
                (quantized_path / "config.json").read_text()
            )
            for key in ("quantization", "quantization_config"):
                bf16_config.pop(key, None)
                quantized_config.pop(key, None)
            if bf16_config != quantized_config:
                return False
            return all(
                filecmp.cmp(
                    bf16_path / filename,
                    quantized_path / filename,
                    shallow=False,
                )
                for filename in ("tokenizer.json", "tokenizer_config.json")
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def _model_available_unlocked(self, model_id: str, model_path: Path) -> bool:
        if self.mock:
            return True
        if not model_path.exists():
            return False
        if self.registry is not None:
            if validate_model_files(model_path):
                return False
            if self.model_descriptor(model_id).get("adapter_error"):
                return False
        if self.backend_name == "torch":
            if model_id == HYBRID_MODEL_ID:
                return False
            if self.model_descriptor(model_id).get("quantization"):
                return False
        if model_id == HYBRID_MODEL_ID:
            return all(
                (weight_path := self.hybrid_weight_paths.get(weight_name))
                is not None
                and weight_path.exists()
                for weight_name in HYBRID_BF16_WEIGHTS
            )
        return True

    def _load_bf16_weight(self, weight_name: str) -> Any:
        import mlx.core as mx

        shard_path = self.hybrid_weight_paths.get(weight_name)
        if shard_path is None:
            raise FileNotFoundError(
                f"Hybridní BF16 váha není dostupná: {weight_name}"
            )
        shard = mx.load(str(shard_path))
        if weight_name not in shard:
            raise KeyError(f"V souboru chybí váha {weight_name}.")
        weight = shard[weight_name]
        del shard
        return weight

    def _install_hybrid_io(self, model: Any) -> None:
        import mlx.core as mx
        import mlx.nn as nn

        hidden_size = int(model.args.hidden_size)
        vocab_size = int(model.args.vocab_size)
        embedding_weight = self._load_bf16_weight(
            "model.embed_tokens.weight"
        )
        head_weight = self._load_bf16_weight("lm_head.weight")
        expected_shape = (vocab_size, hidden_size)
        for weight_name, weight in (
            ("model.embed_tokens.weight", embedding_weight),
            ("lm_head.weight", head_weight),
        ):
            if tuple(weight.shape) != expected_shape:
                raise ValueError(
                    f"Váha {weight_name} má tvar {tuple(weight.shape)}, "
                    f"očekává se {expected_shape}."
                )
            if weight.dtype != mx.bfloat16:
                raise ValueError(
                    f"Váha {weight_name} není BF16, ale {weight.dtype}."
                )

        embedding = nn.Embedding(vocab_size, hidden_size)
        embedding.weight = embedding_weight
        head = nn.Linear(hidden_size, vocab_size, bias=False)
        head.weight = head_weight
        model.model.embed_tokens = embedding
        model.lm_head = head
        model.eval()
        del embedding_weight, head_weight
        gc.collect()

    def _load_model_components(
        self,
        model_id: str,
        model_path: Path,
    ) -> tuple[Any, Any]:
        if self.mock:
            return None, None
        if self.registry is not None:
            problems = validate_model_files(model_path)
            if problems:
                raise ValueError("; ".join(problems))
            registration = self.registry.models[model_id]
            config = json.loads((model_path / "config.json").read_text())
            ADAPTERS[registration.adapter].validate(config)
        if model_id == HYBRID_MODEL_ID and self.backend_name != "mlx":
            raise ValueError("Hybridní složení Apertusu vyžaduje MLX.")
        model, tokenizer = self.backend.load(str(model_path), lazy=model_id == HYBRID_MODEL_ID)
        if model_id == HYBRID_MODEL_ID:
            self._install_hybrid_io(model)
            import mlx.core as mx
            mx.eval(model.parameters())
            self.backend.synchronize()
        return model, tokenizer

    def load(self) -> None:
        if self.mock:
            with self.state_lock:
                self.status = "ready"
                self.phase = None
                self.model_generation = 1
            return
        try:
            model, tokenizer = self._load_model_components(
                self.active_model_id,
                self.model_path,
            )
            with self.state_lock:
                self.model = model
                self.tokenizer = tokenizer
                self.status = "ready"
                self.phase = None
                self.error = None
                self.model_generation += 1
        except Exception as exc:  # pragma: no cover - depends on local Metal runtime
            try:
                self._release_model_local()
            except Exception:
                pass
            with self.state_lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.status = "error"
                self.phase = None

    def start(self, *, load_initial: bool = True) -> None:
        if self.worker_thread is not None:
            return
        self.load_initial_model = load_initial
        if not load_initial:
            self.status, self.phase = "setup_required", None
        self.worker_thread = threading.Thread(
            target=self._worker_loop, name="apertus-model-worker", daemon=True
        )
        self.worker_thread.start()

    def _worker_loop(self) -> None:
        if self.load_initial_model:
            self.load()
        while True:
            job = self.work_queue.get()
            if job is None:
                return
            operation, arguments, done, result = job
            with self.state_lock:
                self.active_operation = operation
                self.active_operation_started_monotonic = time.monotonic()
                if self.outstanding_operation is None:
                    self.outstanding_operation = operation
                    self.outstanding_operation_started_monotonic = (
                        self.active_operation_started_monotonic
                    )
            try:
                if operation == "next":
                    result["value"] = self._next_tokens_local(*arguments)
                elif operation == "search":
                    result["value"] = self._search_tokens_local(*arguments)
                elif operation == "lookup":
                    result["value"] = self._lookup_tokens_local(*arguments)
                elif operation == "compare":
                    result["value"] = self._compare_distributions_local(*arguments)
                elif operation == "steering":
                    result["value"] = self._steering_local(*arguments)
                elif operation == "steering_trajectory":
                    result["value"] = self._steering_trajectory_local(
                        *arguments
                    )
                elif operation == "steering_aligned_path":
                    result["value"] = self._steering_aligned_path_local(
                        *arguments
                    )
                elif operation == "steering_phrase_score":
                    result["value"] = self._steering_phrase_score_local(
                        *arguments
                    )
                elif operation == "steering_sham_control":
                    result["value"] = self._steering_sham_control_local(
                        *arguments
                    )
                elif operation == "steering_layer_sweep":
                    result["value"] = self._steering_layer_sweep_local(
                        *arguments
                    )
                elif operation == "text_surprisal":
                    result["value"] = self._text_surprisal_local(*arguments)
                elif operation == "predictive_stylometry":
                    result["value"] = self._predictive_stylometry_local(
                        *arguments
                    )
                elif operation == "continuation_surprisal":
                    result["value"] = (
                        self._continuation_surprisal_local(*arguments)
                    )
                elif operation == "text_surprisal_comparison":
                    result["value"] = (
                        self._text_surprisal_comparison_local(*arguments)
                    )
                elif operation == "context_contrast":
                    result["value"] = self._context_contrast_local(
                        *arguments
                    )
                elif operation == "embeddings":
                    result["value"] = self._embeddings_local(*arguments)
                elif operation == "embedding_neighborhood":
                    result["value"] = (
                        self._embedding_neighborhood_local(*arguments)
                    )
                elif operation == "lens":
                    result["value"] = self._lens_local(*arguments)
                elif operation == "tuned_lens":
                    result["value"] = self._tuned_lens_local(*arguments)
                elif operation == "jacobian_lens":
                    result["value"] = self._jacobian_lens_local(*arguments)
                elif operation == "component_attribution":
                    result["value"] = self._component_attribution_local(
                        *arguments
                    )
                elif operation == "activation_patching":
                    result["value"] = self._activation_patching_local(
                        *arguments
                    )
                elif operation == "model_comparison":
                    run_comparison(self, *arguments)
                elif operation == "detector_readers":
                    run_detector_reader_comparison(self, *arguments)
                elif operation == "model_admin":
                    arguments[0]()
                elif operation == "switch":
                    self._switch_model_local(*arguments)
                else:
                    raise ValueError(f"Neznámá operace: {operation}")
            except Exception as exc:
                if result is not None:
                    result["error"] = exc
            finally:
                with self.state_lock:
                    self.active_operation = None
                    self.active_operation_started_monotonic = None
                    self.outstanding_operation = None
                    self.outstanding_operation_started_monotonic = None
                if done is not None:
                    done.set()

    def stop(self) -> None:
        if self.worker_thread is not None:
            self.work_queue.put(None)

    def _health_unlocked(self) -> dict[str, Any]:
        worker_operation = (
            self.active_operation or self.outstanding_operation
        )
        worker_started = (
            self.outstanding_operation_started_monotonic
            if self.outstanding_operation_started_monotonic is not None
            else self.active_operation_started_monotonic
        )
        worker_elapsed_ms = (
            (time.monotonic() - worker_started) * 1_000
            if worker_started is not None
            else None
        )
        active_model = (
            {
                **self.model_descriptor(self.active_model_id),
                "id": self.active_model_id,
                "label": self.model_descriptor(self.active_model_id)["label"],
            }
            if self.mock or self.model is not None
            else None
        )
        return {
            "status": self.status,
            "phase": self.phase,
            "model": self.model_name,
            "active_model": active_model,
            "target_model": (
                {
                    "id": self.target_model_id,
                    "label": self.model_label(
                        self.target_model_id, self.target_model_id
                    ),
                }
                if self.target_model_id is not None
                else None
            ),
            "available_models": [
                {
                    **self.model_descriptor(model_id),
                    "id": model_id,
                    "label": self.model_descriptor(model_id)["label"],
                    "available": self._model_available_unlocked(
                        model_id,
                        model_path,
                    ),
                }
                for model_id, model_path in self.model_paths.items()
            ],
            "model_generation": self.model_generation,
            "transition_id": self.transition_id,
            "server_instance_id": self.server_instance_id,
            "mock": self.mock,
            "backend": "mock" if self.mock else self.backend_name,
            "capabilities": (["guide", "archive", "model-admin"] if self.status == "setup_required" else self.model_descriptor(self.active_model_id)["capabilities"]),
            "experimental_addons": self.experimental_addons.availability(self.active_model_id) if self.experimental_addons is not None else {},
            "error": self.error,
            "cache_entries": len(self.logits_cache),
            "performance": {
                "process_peak_rss_bytes": process_peak_rss_bytes(),
                "measurement": "ru_maxrss_normalized",
            },
            "worker": {
                "busy": worker_operation is not None,
                "operation": worker_operation,
                "phase": (
                    "running"
                    if self.active_operation is not None
                    else "queued"
                    if self.outstanding_operation is not None
                    else None
                ),
                "elapsed_ms": worker_elapsed_ms,
                "queue_depth": self.work_queue.qsize(),
            },
            "tuned_lens": self._tuned_lens_status_unlocked(),
            "jacobian_lens": self._jacobian_lens_status_unlocked(),
            "provenance": {
                "runtime": dict(RUNTIME_PROVENANCE),
                "model": self._model_provenance_unlocked(),
            },
        }

    def health(self) -> dict[str, Any]:
        with self.state_lock:
            return self._health_unlocked()

    def _release_model_local(self) -> None:
        had_model = self.model is not None
        with self.cache_lock:
            for field_name in MODEL_BOUND_CACHE_FIELDS:
                getattr(self, field_name).clear()
        if self.tuned_lens_catalog is not None:
            self.tuned_lens_catalog.clear_runtime_cache()
        if self.jacobian_lens_catalog is not None:
            self.jacobian_lens_catalog.clear_runtime_cache()
        if not self.mock and had_model:
            self.backend.synchronize()
        self.active_ids = None
        self.active_cache = None
        self.active_logits = None
        self.vocab_catalog = None
        self.tokenizer_identity_cache = None
        self.tokenizer = None
        self.model = None
        if self.mock or not had_model:
            return
        gc.collect()
        self.backend.clear_cache()
        gc.collect()

    def _switch_model_local(self, target_model_id: str) -> None:
        target_path = self.model_paths[target_model_id]
        try:
            with self.state_lock:
                self.phase = "unloading"
            self._release_model_local()
            with self.state_lock:
                self.phase = "loading"
            model, tokenizer = self._load_model_components(
                target_model_id,
                target_path,
            )
            with self.state_lock:
                self.model = model
                self.tokenizer = tokenizer
                self.model_path = target_path
                self.active_model_id = target_model_id
                self.target_model_id = None
                self.status = "ready"
                self.phase = None
                self.error = None
                self.model_generation += 1
        except Exception as exc:  # pragma: no cover - depends on local Metal runtime
            try:
                self._release_model_local()
            except Exception:
                pass
            with self.state_lock:
                self.target_model_id = None
                self.status = "error"
                self.phase = None
                self.error = f"{type(exc).__name__}: {exc}"

    def request_model_switch(self, target_model_id: str) -> dict[str, Any]:
        with self.state_lock:
            if target_model_id not in self.model_paths:
                raise ValueError(f"Neznámý model: {target_model_id}")
            if not self._model_available_unlocked(
                target_model_id,
                self.model_paths[target_model_id],
            ):
                raise ValueError(
                    f"Model {self.model_label(target_model_id, target_model_id)} "
                    "není na disku."
                )
            if self.status in {"loading", "switching"}:
                raise RuntimeError("Model se právě načítá nebo přepíná.")
            if self.status == "ready" and target_model_id == self.active_model_id:
                response = self._health_unlocked()
                response["accepted"] = False
                return response
            if self.outstanding_operation is not None:
                raise RuntimeError(
                    "Model právě dokončuje předchozí výpočet; přepnutí "
                    "zatím nebylo zařazeno."
                )
            self.status = "switching"
            self.phase = "queued"
            self.target_model_id = target_model_id
            self.error = None
            self.transition_id += 1
            self.outstanding_operation = "switch"
            self.outstanding_operation_started_monotonic = time.monotonic()
            self.work_queue.put(("switch", (target_model_id,), None, None))
            response = self._health_unlocked()
            response["accepted"] = True
            return response

    def _cached_logits(
        self, text: str, provided_token_ids: list[int] | None
    ) -> tuple[np.ndarray, list[int], bool]:
        token_ids = (
            list(provided_token_ids)
            if provided_token_ids is not None
            else encode_with_bos(self.tokenizer, text)
        )
        if len(token_ids) > MAX_INPUT_TOKENS:
            raise ValueError(
                f"Text má {len(token_ids)} tokenů; limit aplikace je {MAX_INPUT_TOKENS}."
            )
        if not token_ids:
            raise ValueError("Tokenizer nevrátil žádné tokeny.")
        key = tuple(token_ids)

        if (
            self.active_ids == token_ids
            and self.active_logits is not None
            and self.active_cache is not None
        ):
            return self.active_logits, token_ids, True

        with self.cache_lock:
            cached = self.logits_cache.get(key)
            if cached is not None:
                self.logits_cache.move_to_end(key)
                return cached, token_ids, True

        extends_active_path = (
            self.active_ids is not None and self.active_cache is not None
            and len(token_ids) == len(self.active_ids) + 1 and token_ids[:-1] == self.active_ids
        )
        cache = self.active_cache if extends_active_path else self.backend.make_cache(self.model)
        ids = [token_ids[-1]] if extends_active_path else token_ids
        logits, cache = self.backend.last_logits(self.model, ids, cache)
        self.active_ids = token_ids
        self.active_cache = cache
        self.active_logits = logits

        with self.cache_lock:
            self.logits_cache[key] = logits
            self.logits_cache.move_to_end(key)
            while len(self.logits_cache) > LOGIT_CACHE_SIZE:
                self.logits_cache.popitem(last=False)
        return logits, token_ids, False

    def _mock_logits(
        self, text: str, provided_token_ids: list[int] | None
    ) -> tuple[np.ndarray, list[int], bool]:
        # Deterministic development mode with the same API shape as the real model.
        vocab_size = 32
        seed = sum((index + 1) * ord(char) for index, char in enumerate(text))
        rng = np.random.default_rng(seed % (2**32))
        logits = rng.normal(0, 0.9, size=vocab_size).astype(np.float32)
        logits[:8] += np.array([3.8, 3.1, 2.6, 2.1, 1.8, 1.5, 1.2, 0.9])
        token_ids = (
            list(provided_token_ids)
            if provided_token_ids is not None
            else list(range(max(1, len(text.split()))))
        )
        return logits, token_ids, False

    def _get_cached_logits_only(
        self, provided_token_ids: list[int] | None
    ) -> tuple[np.ndarray, list[int], bool]:
        if provided_token_ids is None:
            raise ValueError("Pro hledání je nutné poslat token_ids uzlu.")
        token_ids = list(provided_token_ids)
        if not token_ids:
            raise ValueError("token_ids nesmí být prázdné.")
        key = tuple(token_ids)

        if self.active_ids == token_ids and self.active_logits is not None:
            return self.active_logits, token_ids, True

        with self.cache_lock:
            cached = self.logits_cache.get(key)
            if cached is not None:
                self.logits_cache.move_to_end(key)
                return cached, token_ids, True

        raise LogitsNotCachedError(
            "Úplná distribuce tohoto uzlu už není v cache."
        )

    def _next_tokens_local(
        self,
        text: str,
        temperature: float,
        top_k: int,
        provided_token_ids: list[int] | None,
        pinned_token_ids: list[int] | None = None,
        cache_only: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()

        if self.mock:
            logits, token_ids, cached = self._mock_logits(text, provided_token_ids)
        elif cache_only:
            logits, token_ids, cached = self._get_cached_logits_only(
                provided_token_ids
            )
        else:
            logits, token_ids, cached = self._cached_logits(text, provided_token_ids)

        adjusted_probabilities, raw_probabilities = probability_views(
            logits,
            temperature,
        )
        (
            indices,
            probabilities,
            other,
            entropy_bits,
            effective_tokens,
            normalized_entropy,
        ) = summarize_probabilities(
            adjusted_probabilities,
            top_k=top_k,
        )
        candidates = [
            self._candidate_payload(
                int(token_id),
                float(probability),
                float(raw_probabilities[token_id]),
                text,
                token_ids,
            )
            for token_id, probability in zip(indices, probabilities)
        ]
        pinned_candidates: list[dict[str, Any]] = []
        if pinned_token_ids:
            for token_id in dict.fromkeys(pinned_token_ids):
                if 0 <= token_id < adjusted_probabilities.size:
                    pinned_candidates.append(
                        self._candidate_payload(
                            token_id,
                            float(adjusted_probabilities[token_id]),
                            float(raw_probabilities[token_id]),
                            text,
                            token_ids,
                        )
                    )

        elapsed_ms = (time.perf_counter() - started) * 1_000
        return {
            "text": text,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "temperature": temperature,
            "top_k": top_k,
            "candidates": candidates,
            "pinned_candidates": pinned_candidates,
            "other_probability": other,
            "top_k_mass": 1.0 - other,
            "entropy_bits": entropy_bits,
            "effective_tokens": effective_tokens,
            "normalized_entropy": normalized_entropy,
            "raw_metrics": raw_distribution_metrics(
                logits,
                raw_probabilities,
            ),
            "latency_ms": elapsed_ms,
            "cached": cached,
            "model": self.model_name,
            "model_id": self.active_model_id,
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
            "probability_kind": (
                "raw_model" if math.isclose(temperature, 1.0) else "temperature_adjusted"
            ),
        }

    def _candidate_payload(
        self,
        token_id: int,
        probability: float,
        raw_probability: float,
        text: str,
        token_ids: list[int],
    ) -> dict[str, Any]:
        special_ids = (
            set(getattr(self.tokenizer, "all_special_ids", [])) if not self.mock else set()
        )
        is_special = token_id in special_ids
        if self.mock:
            piece = MOCK_WORDS[token_id % len(MOCK_WORDS)]
            next_text = text + piece
        else:
            next_ids = token_ids + [token_id]
            next_text = safe_decode(
                self.tokenizer, next_ids, skip_special=not is_special
            )
            decoded_token = safe_decode(
                self.tokenizer, [token_id], skip_special=False
            )
            if next_text.startswith(text):
                piece = next_text[len(text) :]
            else:
                piece = decoded_token
            if not piece:
                try:
                    piece = str(self.tokenizer.convert_ids_to_tokens(token_id))
                except Exception:
                    piece = f"token {token_id}"
        return {
            "token_id": token_id,
            "text": piece,
            "display": visible_token(piece, f"token {token_id}"),
            "next_text": next_text,
            "next_token_ids": token_ids + [token_id],
            "probability": probability,
            "raw_probability": raw_probability,
            "is_special": is_special,
        }

    def _build_vocab_catalog(self) -> list[tuple[int, str, str]]:
        if self.vocab_catalog is not None:
            return self.vocab_catalog
        if self.mock:
            self.vocab_catalog = [
                (
                    token_id,
                    normalized_token_for_search(piece),
                    normalized_token_for_search(piece).strip(),
                )
                for token_id, piece in enumerate(MOCK_WORDS)
            ]
            return self.vocab_catalog

        get_vocab = getattr(self.tokenizer, "get_vocab", None)
        if not callable(get_vocab):
            inner = getattr(self.tokenizer, "_tokenizer", None)
            get_vocab = getattr(inner, "get_vocab", None)
        if callable(get_vocab):
            vocabulary = get_vocab()
            raw_by_id = {
                int(token_id): str(piece)
                for piece, token_id in vocabulary.items()
            }
            vocab_size = max(raw_by_id, default=-1) + 1
        else:
            vocab_size = int(getattr(self.model.args, "vocab_size", 0))
            if vocab_size <= 0:
                raise RuntimeError("Tokenizer neposkytuje prohledávatelný slovník.")
            raw_by_id = {}

        self.vocab_catalog = []
        batch_decode = getattr(self.tokenizer, "batch_decode", None)
        batch_size = 4_096
        for start in range(0, vocab_size, batch_size):
            token_ids = list(range(start, min(start + batch_size, vocab_size)))
            decoded: list[str]
            if callable(batch_decode):
                try:
                    decoded = list(
                        batch_decode(
                            [[token_id] for token_id in token_ids],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    )
                except TypeError:
                    decoded = list(
                        batch_decode(
                            [[token_id] for token_id in token_ids],
                            skip_special_tokens=False,
                        )
                    )
            else:
                decoded = [
                    safe_decode(
                        self.tokenizer, [token_id], skip_special=False
                    )
                    for token_id in token_ids
                ]

            for token_id, decoded_piece in zip(token_ids, decoded):
                raw_piece = raw_by_id.get(token_id, "")
                decoded_key = normalized_token_for_search(str(decoded_piece))
                raw_key = normalized_token_for_search(raw_piece)
                searchable = f"{decoded_key}\0{raw_key}"
                exact = decoded_key.strip()
                self.vocab_catalog.append((token_id, searchable, exact))
        return self.vocab_catalog

    def _search_tokens_local(
        self,
        text: str,
        query: str,
        temperature: float,
        provided_token_ids: list[int] | None,
        limit: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if self.mock:
            logits, token_ids, cached = self._mock_logits(text, provided_token_ids)
        else:
            logits, token_ids, cached = self._get_cached_logits_only(
                provided_token_ids
            )
        probabilities, raw_probabilities = probability_views(logits, temperature)
        query_text = query.strip()
        needle = normalized_token_for_search(query_text).strip()
        if not needle:
            raise ValueError("Hledaný token nesmí být prázdný.")

        if query_text.startswith("#") and query_text[1:].isdigit():
            requested_id = int(query_text[1:])
            matching_ids = (
                [requested_id]
                if 0 <= requested_id < probabilities.size
                else []
            )
            match_kinds = {token_id: "id" for token_id in matching_ids}
        else:
            matches = [
                (token_id, exact == needle)
                for token_id, searchable, exact in self._build_vocab_catalog()
                if needle in searchable and 0 <= token_id < probabilities.size
            ]
            matches.sort(
                key=lambda item: (
                    not item[1],
                    -float(probabilities[item[0]]),
                )
            )
            matching_ids = [token_id for token_id, _ in matches]
            match_kinds = {
                token_id: "exact" if is_exact else "substring"
                for token_id, is_exact in matches
            }
        if query_text.startswith("#") and query_text[1:].isdigit():
            matching_ids.sort(
                key=lambda token_id: probabilities[token_id], reverse=True
            )
        results: list[dict[str, Any]] = []
        for token_id in matching_ids[:limit]:
            candidate = self._candidate_payload(
                token_id,
                float(probabilities[token_id]),
                float(raw_probabilities[token_id]),
                text,
                token_ids,
            )
            candidate["rank"] = 1 + int(np.count_nonzero(logits > logits[token_id]))
            candidate["match_kind"] = match_kinds[token_id]
            results.append(candidate)

        return {
            "text": text,
            "token_ids": token_ids,
            "query": query,
            "temperature": temperature,
            "matches_total": len(matching_ids),
            "results": results,
            "latency_ms": (time.perf_counter() - started) * 1_000,
            "cached": cached,
            "model_id": self.active_model_id,
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
        }

    def _lookup_tokens_local(
        self,
        text: str,
        requested_token_ids: list[int],
        temperature: float,
        provided_token_ids: list[int] | None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if self.mock:
            logits, token_ids, cached = self._mock_logits(text, provided_token_ids)
        else:
            logits, token_ids, cached = self._get_cached_logits_only(
                provided_token_ids
            )
        probabilities, raw_probabilities = probability_views(logits, temperature)
        results: list[dict[str, Any]] = []
        for token_id in dict.fromkeys(requested_token_ids):
            if not 0 <= token_id < probabilities.size:
                continue
            candidate = self._candidate_payload(
                token_id,
                float(probabilities[token_id]),
                float(raw_probabilities[token_id]),
                text,
                token_ids,
            )
            candidate["rank"] = 1 + int(np.count_nonzero(logits > logits[token_id]))
            results.append(candidate)
        return {
            "text": text,
            "token_ids": token_ids,
            "requested_token_ids": requested_token_ids,
            "temperature": temperature,
            "results": results,
            "latency_ms": (time.perf_counter() - started) * 1_000,
            "cached": cached,
            "model_id": self.active_model_id,
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
        }

    def _comparison_token_display(self, token_id: int) -> str:
        if self.mock:
            piece = MOCK_WORDS[token_id % len(MOCK_WORDS)]
        else:
            piece = safe_decode(
                self.tokenizer,
                [token_id],
                skip_special=False,
            )
            if not piece:
                try:
                    piece = str(self.tokenizer.convert_ids_to_tokens(token_id))
                except Exception:
                    piece = f"token {token_id}"
        return visible_token(piece, f"token {token_id}")

    def _compare_distributions_local(
        self,
        left_text: str,
        left_token_ids: list[int],
        right_text: str,
        right_token_ids: list[int],
        temperature: float,
        mover_count: int,
        overlap_k: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if self.mock:
            left_logits, left_ids, _ = self._mock_logits(
                left_text,
                left_token_ids,
            )
            right_logits, right_ids, _ = self._mock_logits(
                right_text,
                right_token_ids,
            )
        else:
            try:
                left_logits, left_ids, _ = self._get_cached_logits_only(
                    left_token_ids
                )
            except LogitsNotCachedError as exc:
                raise LogitsNotCachedError(
                    "Úplná distribuce uzlu A už není v cache.",
                    side="left",
                ) from exc
            try:
                right_logits, right_ids, _ = self._get_cached_logits_only(
                    right_token_ids
                )
            except LogitsNotCachedError as exc:
                raise LogitsNotCachedError(
                    "Úplná distribuce uzlu B už není v cache.",
                    side="right",
                ) from exc

        if left_logits.ndim != 1 or right_logits.ndim != 1:
            raise ValueError("Porovnávané logity musí být jednorozměrné.")
        if left_logits.shape != right_logits.shape:
            raise ValueError(
                "Porovnávané distribuce mají rozdílnou velikost slovníku."
            )
        vocabulary_size = int(left_logits.size)
        if overlap_k > vocabulary_size:
            raise ValueError(
                f"overlap_k nesmí překročit velikost slovníku "
                f"({vocabulary_size})."
            )

        left_probabilities = probabilities_at_temperature(
            left_logits,
            temperature,
        )
        right_probabilities = probabilities_at_temperature(
            right_logits,
            temperature,
        )

        def entropy_and_support(
            probabilities: np.ndarray,
        ) -> tuple[float, float]:
            positive = probabilities > 0
            entropy_bits = max(
                0.0,
                float(
                    -np.sum(
                        probabilities[positive]
                        * np.log2(probabilities[positive])
                    )
                ),
            )
            return entropy_bits, float(2**entropy_bits)

        left_entropy, left_effective = entropy_and_support(
            left_probabilities
        )
        right_entropy, right_effective = entropy_and_support(
            right_probabilities
        )

        midpoint = 0.5 * (left_probabilities + right_probabilities)

        def kl_to_midpoint(probabilities: np.ndarray) -> float:
            positive = probabilities > 0
            return float(
                np.sum(
                    probabilities[positive]
                    * np.log2(
                        probabilities[positive] / midpoint[positive]
                    )
                )
            )

        js_divergence = min(
            1.0,
            max(
                0.0,
                0.5
                * (
                    kl_to_midpoint(left_probabilities)
                    + kl_to_midpoint(right_probabilities)
                ),
            ),
        )
        total_variation = min(
            1.0,
            max(
                0.0,
                float(
                    0.5
                    * np.sum(
                        np.abs(
                            right_probabilities - left_probabilities
                        )
                    )
                ),
            ),
        )

        left_top = np.argsort(
            -left_probabilities,
            kind="stable",
        )[:overlap_k]
        right_top = np.argsort(
            -right_probabilities,
            kind="stable",
        )[:overlap_k]
        overlap_count = len(
            set(int(token_id) for token_id in left_top)
            & set(int(token_id) for token_id in right_top)
        )

        deltas = right_probabilities - left_probabilities
        all_token_ids = np.arange(vocabulary_size)
        gain_ids = all_token_ids[deltas > 0]
        loss_ids = all_token_ids[deltas < 0]
        if gain_ids.size:
            gain_ids = gain_ids[
                np.lexsort((gain_ids, -deltas[gain_ids]))
            ][:mover_count]
        if loss_ids.size:
            loss_ids = loss_ids[
                np.lexsort((loss_ids, deltas[loss_ids]))
            ][:mover_count]

        comparison_special_ids = (
            set(getattr(self.tokenizer, "all_special_ids", []))
            if not self.mock
            else set()
        )

        def mover_payload(token_id_value: Any) -> dict[str, Any]:
            token_id = int(token_id_value)
            return {
                "token_id": token_id,
                "display": self._comparison_token_display(token_id),
                "is_special": token_id in comparison_special_ids,
                "probability_left": float(left_probabilities[token_id]),
                "probability_right": float(right_probabilities[token_id]),
                "delta": float(deltas[token_id]),
                "rank_left": 1
                + int(
                    np.count_nonzero(
                        left_logits > left_logits[token_id]
                    )
                ),
                "rank_right": 1
                + int(
                    np.count_nonzero(
                        right_logits > right_logits[token_id]
                    )
                ),
            }

        return {
            "left": {
                "text": left_text,
                "token_ids": left_ids,
                "entropy_bits": left_entropy,
                "effective_tokens": left_effective,
            },
            "right": {
                "text": right_text,
                "token_ids": right_ids,
                "entropy_bits": right_entropy,
                "effective_tokens": right_effective,
            },
            "delta_entropy_bits": right_entropy - left_entropy,
            "effective_support_ratio": right_effective / left_effective,
            "js_divergence_bits": js_divergence,
            "total_variation": total_variation,
            "top_k_overlap": {
                "k": overlap_k,
                "count": overlap_count,
                "fraction": overlap_count / overlap_k,
            },
            "gains": [mover_payload(token_id) for token_id in gain_ids],
            "losses": [mover_payload(token_id) for token_id in loss_ids],
            "temperature": temperature,
            "probability_kind": (
                "raw_model"
                if math.isclose(temperature, 1.0)
                else "temperature_adjusted"
            ),
            "vocab_size": vocabulary_size,
            "model_id": self.active_model_id,
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
            "latency_ms": (time.perf_counter() - started) * 1_000,
        }

    def steering_layer_count(self) -> int:
        """Use loaded-model metadata, with exact development-simulation depths."""
        count = self.model_descriptor(self.active_model_id).get("layers")
        if type(count) is not int or count < 1:
            raise ValueError("Model nemá známý počet steeringových vrstev.")
        return count

    def _validate_steering_layer(self, layer: int) -> None:
        count = self.steering_layer_count()
        if type(layer) is not int or not 1 <= layer <= count:
            raise ValueError(f"layer musí být celé číslo mezi 1 a {count}.")

    def _steering_body(self) -> Any:
        if self.backend_name != "mlx":
            raise RuntimeError("Steering aktivací vyžaduje podporovaný MLX backend.")
        return steering_decoder_body(self.model)

    def _encode_steering_text(self, text: str) -> list[int]:
        if self.mock:
            pieces = text.split()
            token_ids = [1]
            for piece in pieces:
                digest = hashlib.sha256(piece.encode("utf-8")).digest()
                token_ids.append(
                    2 + int.from_bytes(digest[:4], "big") % 1_000_000
                )
        else:
            token_ids = encode_with_bos(self.tokenizer, text)
        if not token_ids:
            raise ValueError("Tokenizér vrátil prázdnou posloupnost.")
        if len(token_ids) > MAX_STEERING_INPUT_TOKENS:
            raise ValueError(
                f"Steeringový text má {len(token_ids)} tokenů; limit je "
                f"{MAX_STEERING_INPUT_TOKENS}."
            )
        return token_ids

    def _prepare_steering_inputs(
        self,
        contrast_pairs: list[dict[str, Any]],
        test_text: str,
    ) -> tuple[
        list[tuple[list[int], list[int]]],
        list[dict[str, Any]],
        str,
        list[int],
        int,
    ]:
        """Validate and encode shared steering inputs once per request."""
        if not MIN_STEERING_PAIRS <= len(contrast_pairs) <= MAX_STEERING_PAIRS:
            raise ValueError(
                f"Steering vyžaduje {MIN_STEERING_PAIRS} až "
                f"{MAX_STEERING_PAIRS} kontrastních párů."
            )

        test_text = test_text.rstrip()
        if not test_text.strip():
            raise ValueError("Testovací prompt musí být neprázdný.")
        encoded_pairs: list[tuple[list[int], list[int]]] = []
        prepared_pairs: list[dict[str, Any]] = []
        contrast_texts: set[str] = set()
        seen_pairs: set[tuple[str, str]] = set()
        total_tokens = 0
        for pair_index, pair in enumerate(contrast_pairs, start=1):
            if not isinstance(pair, dict):
                raise ValueError(
                    f"Kontrastní pár {pair_index} musí být JSON objekt."
                )
            negative_text = pair.get("negative_text")
            positive_text = pair.get("positive_text")
            for name, value in (
                ("negative_text", negative_text),
                ("positive_text", positive_text),
            ):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Pár {pair_index}: {name} musí být neprázdný "
                        "řetězec."
                    )
            assert isinstance(negative_text, str)
            assert isinstance(positive_text, str)
            negative_text = negative_text.rstrip()
            positive_text = positive_text.rstrip()
            if negative_text == positive_text:
                raise ValueError(
                    f"Pár {pair_index} obsahuje dvě shodné varianty."
                )
            pair_key = tuple(sorted((negative_text, positive_text)))
            if pair_key in seen_pairs:
                raise ValueError(
                    f"Kontrastní pár {pair_index} je duplicitní nebo "
                    "obrací již použitý pár; "
                    "steering vyžaduje nejméně tři různé páry."
                )
            seen_pairs.add(pair_key)
            contrast_texts.update((negative_text, positive_text))
            negative_ids = self._encode_steering_text(negative_text)
            positive_ids = self._encode_steering_text(positive_text)
            total_tokens += len(negative_ids) + len(positive_ids)
            if negative_ids[-1] != positive_ids[-1]:
                raise ValueError(
                    f"Pár {pair_index} nekončí stejným token_id "
                    f"({negative_ids[-1]} ≠ {positive_ids[-1]}). "
                    "Použij společný koncový boundary token."
                )
            encoded_pairs.append((negative_ids, positive_ids))
            prepared_pairs.append(
                {
                    "negative_text": negative_text,
                    "positive_text": positive_text,
                    "negative_token_ids": negative_ids,
                    "positive_token_ids": positive_ids,
                    "boundary_token_id": negative_ids[-1],
                    "boundary_token": self._comparison_token_display(
                        negative_ids[-1]
                    ),
                }
            )

        if test_text in contrast_texts:
            raise ValueError(
                "Testovací prompt musí být nezávislý: nesmí být shodný "
                "s žádnou variantou kontrastních párů."
            )
        test_token_ids = self._encode_steering_text(test_text)
        total_tokens += len(test_token_ids)
        if total_tokens > MAX_STEERING_TOTAL_TOKENS:
            raise ValueError(
                f"Steeringové texty mají dohromady {total_tokens} tokenů; "
                f"souhrnný limit je {MAX_STEERING_TOTAL_TOKENS}."
            )
        return (
            encoded_pairs,
            prepared_pairs,
            test_text,
            test_token_ids,
            total_tokens,
        )

    @staticmethod
    def _mock_steering_state(text: str, layer: int) -> np.ndarray:
        digest = hashlib.sha256(
            f"steering-state-v1\0{layer}\0{text}".encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 1.0, size=64).astype(np.float64)

    def _mlx_residual_at_layer(
        self,
        token_ids: list[int],
        layer: int,
    ) -> tuple[np.ndarray, float]:
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        body = self._steering_body()
        started = time.perf_counter()
        ids = mx.array([token_ids])
        hidden = body.embed_tokens(ids)
        causal_mask = create_attention_mask(hidden, None)
        for decoder_layer in body.layers[:layer]:
            hidden = decoder_layer(
                hidden,
                mask=causal_mask,
                cache=None,
            )
        final_position = hidden[0, -1, :].astype(mx.float32)
        mx.eval(final_position)
        state = np.asarray(final_position).astype(np.float64, copy=False)
        return state, (time.perf_counter() - started) * 1_000

    def _mlx_steering_direction(
        self,
        encoded_pairs: list[tuple[list[int], list[int]]],
        layer: int,
    ) -> tuple[np.ndarray, dict[str, Any], float]:
        """Compute one normalized contrastive direction for a request."""
        self._steering_body()
        self._validate_steering_layer(layer)

        contrast_forward_ms = 0.0
        deltas: list[np.ndarray] = []
        for negative_ids, positive_ids in encoded_pairs:
            negative_state, elapsed = self._mlx_residual_at_layer(
                negative_ids,
                layer,
            )
            contrast_forward_ms += elapsed
            positive_state, elapsed = self._mlx_residual_at_layer(
                positive_ids,
                layer,
            )
            contrast_forward_ms += elapsed
            deltas.append(positive_state - negative_state)
        unit_direction, diagnostics = aggregate_steering_deltas(
            np.stack(deltas, axis=0)
        )
        return unit_direction, diagnostics, contrast_forward_ms

    def _mock_steering_direction(
        self,
        contrast_pairs: list[dict[str, Any]],
        layer: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        deltas = np.stack(
            [
                self._mock_steering_state(pair["positive_text"], layer)
                - self._mock_steering_state(pair["negative_text"], layer)
                for pair in contrast_pairs
            ],
            axis=0,
        )
        return aggregate_steering_deltas(deltas)

    def _mlx_steering_logits(
        self,
        encoded_pairs: list[tuple[list[int], list[int]]],
        test_token_ids: list[int],
        layer: int,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, float]]:
        """Run cache-free contrast forwards and one batched intervention."""
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        body = self._steering_body()
        layers = getattr(body, "layers", None)
        (
            unit_direction,
            diagnostics,
            contrast_forward_ms,
        ) = self._mlx_steering_direction(
            encoded_pairs,
            layer,
        )
        shift = (
            alpha
            * float(diagnostics["native_scale"])
            * unit_direction
        )

        test_forward_started = time.perf_counter()
        ids = mx.array([test_token_ids])
        hidden = body.embed_tokens(ids)
        causal_mask = create_attention_mask(hidden, None)
        for decoder_layer in layers[:layer]:
            hidden = decoder_layer(
                hidden,
                mask=causal_mask,
                cache=None,
            )

        mlx_shift = mx.array(
            shift.astype(np.float32, copy=False)
        ).astype(hidden.dtype)
        evaluated_shift = mlx_shift.astype(mx.float32)
        mx.eval(evaluated_shift)
        diagnostics["effective_shift_norm"] = float(
            np.linalg.norm(np.asarray(evaluated_shift))
        )
        if math.isclose(alpha, 0.0, abs_tol=0.0):
            steered_hidden = hidden
        else:
            steered_last = hidden[:, -1:, :] + mlx_shift[None, None, :]
            steered_hidden = mx.concatenate(
                (hidden[:, :-1, :], steered_last),
                axis=1,
            )
        batched_hidden = mx.concatenate(
            (hidden, steered_hidden),
            axis=0,
        )
        batched_mask = create_attention_mask(batched_hidden, None)
        for decoder_layer in layers[layer:]:
            batched_hidden = decoder_layer(
                batched_hidden,
                mask=batched_mask,
                cache=None,
            )
        final_states = batched_hidden[:, -1, :]
        mx.eval(final_states)
        test_forward_ms = (
            time.perf_counter() - test_forward_started
        ) * 1_000

        projection_started = time.perf_counter()
        normalized_states = body.norm(final_states)
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is not None:
            projected = lm_head(normalized_states)
        else:
            as_linear = getattr(body.embed_tokens, "as_linear", None)
            if not callable(as_linear):
                raise RuntimeError(
                    "Model nemá lm_head ani vázanou výstupní projekci."
                )
            projected = as_linear(normalized_states)
        projected = projected.astype(mx.float32)
        mx.eval(projected)
        logits = np.asarray(projected)
        projection_ms = (
            time.perf_counter() - projection_started
        ) * 1_000
        left_logits = np.asarray(logits[0]).astype(np.float32, copy=True)
        right_logits = np.asarray(logits[1]).astype(np.float32, copy=True)
        if math.isclose(alpha, 0.0, abs_tol=0.0):
            # Alpha zero is the explicit numerical identity control.
            right_logits = left_logits.copy()
        return left_logits, right_logits, diagnostics, {
            "contrast_forward_ms": contrast_forward_ms,
            "test_batched_forward_ms": test_forward_ms,
            "projection_ms": projection_ms,
        }

    def _mock_steering_logits(
        self,
        contrast_pairs: list[dict[str, Any]],
        test_text: str,
        layer: int,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, float]]:
        unit_direction, diagnostics = self._mock_steering_direction(
            contrast_pairs,
            layer,
        )
        left_logits, _, _ = self._mock_logits(test_text, None)
        projection_seed = int.from_bytes(
            hashlib.sha256(
                f"mock-steering-projection-v1\0{layer}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(projection_seed)
        projection = rng.normal(
            0.0,
            0.18,
            size=(unit_direction.size, left_logits.size),
        )
        shift = (
            alpha
            * float(diagnostics["native_scale"])
            * unit_direction
        )
        diagnostics["effective_shift_norm"] = float(
            np.linalg.norm(shift.astype(np.float32))
        )
        right_logits = (
            left_logits.astype(np.float64, copy=False)
            + shift @ projection
        ).astype(np.float32)
        if math.isclose(alpha, 0.0, abs_tol=0.0):
            right_logits = left_logits.copy()
        return left_logits, right_logits, diagnostics, {
            "contrast_forward_ms": 0.0,
            "test_batched_forward_ms": 0.0,
            "projection_ms": 0.0,
        }

    def _steering_top_tokens(
        self,
        logits: np.ndarray,
        probabilities: np.ndarray,
        top_k: int,
    ) -> list[dict[str, Any]]:
        token_ids = np.argsort(
            -probabilities,
            kind="stable",
        )[:top_k]
        result: list[dict[str, Any]] = []
        for rank, token_id_value in enumerate(token_ids, start=1):
            token_id = int(token_id_value)
            candidate = self._lens_token_payload(token_id)
            candidate.update(
                {
                    "rank": rank,
                    "probability": float(probabilities[token_id]),
                    "logit": float(logits[token_id]),
                }
            )
            result.append(candidate)
        return result

    def _steering_comparison_payload(
        self,
        left_logits: np.ndarray,
        right_logits: np.ndarray,
        test_text: str,
        test_token_ids: list[int],
        temperature: float,
        top_k: int,
        mover_count: int,
        overlap_k: int,
    ) -> dict[str, Any]:
        if left_logits.ndim != 1 or right_logits.ndim != 1:
            raise ValueError("Steeringové logity musí být jednorozměrné.")
        if left_logits.shape != right_logits.shape:
            raise ValueError(
                "Baseline a steering mají rozdílnou velikost slovníku."
            )
        vocabulary_size = int(left_logits.size)
        if overlap_k > vocabulary_size:
            raise ValueError(
                f"overlap_k nesmí překročit velikost slovníku "
                f"({vocabulary_size})."
            )

        left_probabilities = probabilities_at_temperature(
            left_logits,
            temperature,
        )
        right_probabilities = probabilities_at_temperature(
            right_logits,
            temperature,
        )

        def entropy_and_support(
            probabilities: np.ndarray,
        ) -> tuple[float, float]:
            positive = probabilities > 0
            entropy = max(
                0.0,
                float(
                    -np.sum(
                        probabilities[positive]
                        * np.log2(probabilities[positive])
                    )
                ),
            )
            return entropy, float(2**entropy)

        left_entropy, left_effective = entropy_and_support(
            left_probabilities
        )
        right_entropy, right_effective = entropy_and_support(
            right_probabilities
        )
        midpoint = 0.5 * (left_probabilities + right_probabilities)

        def kl_to_midpoint(probabilities: np.ndarray) -> float:
            positive = probabilities > 0
            return float(
                np.sum(
                    probabilities[positive]
                    * np.log2(
                        probabilities[positive] / midpoint[positive]
                    )
                )
            )

        js_divergence = min(
            1.0,
            max(
                0.0,
                0.5
                * (
                    kl_to_midpoint(left_probabilities)
                    + kl_to_midpoint(right_probabilities)
                ),
            ),
        )
        total_variation = min(
            1.0,
            max(
                0.0,
                float(
                    0.5
                    * np.sum(
                        np.abs(
                            right_probabilities - left_probabilities
                        )
                    )
                ),
            ),
        )
        left_top = np.argsort(
            -left_probabilities,
            kind="stable",
        )[:overlap_k]
        right_top = np.argsort(
            -right_probabilities,
            kind="stable",
        )[:overlap_k]
        overlap_count = len(
            set(int(token_id) for token_id in left_top)
            & set(int(token_id) for token_id in right_top)
        )

        deltas = right_probabilities - left_probabilities
        all_token_ids = np.arange(vocabulary_size)
        gain_ids = all_token_ids[deltas > 0]
        loss_ids = all_token_ids[deltas < 0]
        if gain_ids.size:
            gain_ids = gain_ids[
                np.lexsort((gain_ids, -deltas[gain_ids]))
            ][:mover_count]
        if loss_ids.size:
            loss_ids = loss_ids[
                np.lexsort((loss_ids, deltas[loss_ids]))
            ][:mover_count]
        special_ids = (
            set(getattr(self.tokenizer, "all_special_ids", []))
            if not self.mock
            else set()
        )

        def mover_payload(token_id_value: Any) -> dict[str, Any]:
            token_id = int(token_id_value)
            token = self._lens_token_payload(token_id)
            return {
                **token,
                "is_special": token_id in special_ids,
                "probability_left": float(left_probabilities[token_id]),
                "probability_right": float(right_probabilities[token_id]),
                "delta": float(deltas[token_id]),
                "rank_left": 1
                + int(np.count_nonzero(left_logits > left_logits[token_id])),
                "rank_right": 1
                + int(
                    np.count_nonzero(right_logits > right_logits[token_id])
                ),
            }

        gains = [mover_payload(token_id) for token_id in gain_ids]
        losses = [mover_payload(token_id) for token_id in loss_ids]
        return {
            "left": {
                "label": "baseline",
                "text": test_text,
                "token_ids": list(test_token_ids),
                "entropy_bits": left_entropy,
                "effective_tokens": left_effective,
                "top_tokens": self._steering_top_tokens(
                    left_logits,
                    left_probabilities,
                    top_k,
                ),
            },
            "right": {
                "label": "steered",
                "text": test_text,
                "token_ids": list(test_token_ids),
                "entropy_bits": right_entropy,
                "effective_tokens": right_effective,
                "top_tokens": self._steering_top_tokens(
                    right_logits,
                    right_probabilities,
                    top_k,
                ),
            },
            "delta_entropy_bits": right_entropy - left_entropy,
            "effective_support_ratio": right_effective / left_effective,
            "js_divergence_bits": js_divergence,
            "total_variation": total_variation,
            "top_k_overlap": {
                "k": overlap_k,
                "count": overlap_count,
                "fraction": overlap_count / overlap_k,
            },
            "gains": gains,
            "losses": losses,
            "movers": {
                "gains": gains,
                "losses": losses,
            },
            "temperature": temperature,
            "probability_kind": (
                "raw_model"
                if math.isclose(temperature, 1.0)
                else "temperature_adjusted"
            ),
            "vocab_size": vocabulary_size,
        }

    def _steering_local(
        self,
        contrast_pairs: list[dict[str, Any]],
        test_text: str,
        layer: int,
        alpha: float,
        temperature: float,
        top_k: int,
        mover_count: int,
        overlap_k: int,
    ) -> dict[str, Any]:
        self._validate_steering_layer(layer)
        started = time.perf_counter()
        (
            encoded_pairs,
            prepared_pairs,
            test_text,
            test_token_ids,
            total_tokens,
        ) = self._prepare_steering_inputs(contrast_pairs, test_text)

        if self.mock:
            (
                left_logits,
                right_logits,
                direction_diagnostics,
                timings,
            ) = self._mock_steering_logits(
                prepared_pairs,
                test_text,
                layer,
                alpha,
            )
        else:
            (
                left_logits,
                right_logits,
                direction_diagnostics,
                timings,
            ) = self._mlx_steering_logits(
                encoded_pairs,
                test_token_ids,
                layer,
                alpha,
            )

        response = self._steering_comparison_payload(
            left_logits,
            right_logits,
            test_text,
            test_token_ids,
            temperature,
            top_k,
            mover_count,
            overlap_k,
        )
        pair_diagnostics: list[dict[str, Any]] = []
        for pair, norm, cosine in zip(
            prepared_pairs,
            direction_diagnostics["pair_norms"],
            direction_diagnostics["pair_cosines_to_direction"],
            strict=True,
        ):
            pair_diagnostics.append(
                {
                    **pair,
                    "delta_norm": norm,
                    "cosine_to_direction": cosine,
                }
            )

        warnings: list[dict[str, str]] = []
        if abs(alpha) > 1.0:
            warnings.append(
                {
                    "code": "extrapolation",
                    "message": (
                        "|α| > 1 překračuje typickou velikost "
                        "kontrastního rozdílu."
                    ),
                }
            )
        if math.isclose(alpha, 0.0, abs_tol=0.0):
            warnings.append(
                {
                    "code": "identity_control",
                    "message": (
                        "α = 0 je identitní kontrola; obě distribuce jsou "
                        "záměrně totožné."
                    ),
                }
            )
        elif direction_diagnostics["effective_shift_norm"] == 0.0:
            warnings.append(
                {
                    "code": "ineffective_shift",
                    "message": (
                        "Nenulové α po převodu do výpočetního dtype zaniklo; "
                        "zásah je numericky totožný s identitní kontrolou."
                    ),
                }
            )
        if direction_diagnostics["resultant_length"] < 0.25:
            warnings.append(
                {
                    "code": "weak_pair_coherence",
                    "message": (
                        "Kontrastní páry mají slabý společný směr; "
                        "interpretuj zásah opatrně."
                    ),
                }
            )
        if any(
            cosine < 0
            for cosine in direction_diagnostics[
                "pair_cosines_to_direction"
            ]
        ):
            warnings.append(
                {
                    "code": "opposed_pair",
                    "message": (
                        "Alespoň jeden pár míří proti agregovanému směru."
                    ),
                }
            )
        if self.is_quantized():
            warnings.append(
                {
                    "code": "quantized_model",
                    "message": (
                        "Výsledek pochází z kvantovaného nebo hybridního "
                        "modelu; těsné změny mohou být numericky citlivé."
                    ),
                }
            )

        response.update(
            {
                "steering": {
                    "kind": "contrastive_activation_addition",
                    "hook": "post_decoder_block_residual_last_position",
                    "layer": layer,
                    "layer_numbering": "one_based",
                    "alpha": alpha,
                    "pair_count": len(prepared_pairs),
                    "total_input_tokens": total_tokens,
                    "aggregation": direction_diagnostics["aggregation"],
                    "native_scale": direction_diagnostics["native_scale"],
                    "applied_shift_norm": (
                        abs(alpha)
                        * float(direction_diagnostics["native_scale"])
                    ),
                    "effective_shift_norm": direction_diagnostics[
                        "effective_shift_norm"
                    ],
                    "resultant_length": direction_diagnostics[
                        "resultant_length"
                    ],
                    "cancellation_fraction": direction_diagnostics[
                        "cancellation_fraction"
                    ],
                    "pair_diagnostics": pair_diagnostics,
                    "boundary_policy": (
                        "same_final_token_id_within_each_pair"
                    ),
                    "execution": "single_serial_worker_job",
                    "cache_policy": "cache_free_no_state_mutation",
                },
                "warnings": warnings,
                "test_text": test_text,
                "test_token_count": len(test_token_ids),
                "top_k": top_k,
                "cached": False,
                "model_id": self.active_model_id,
                "model_label": self.model_label(
                    self.active_model_id,
                    self.active_model_id,
                ),
                "model_generation": self.model_generation,
                "server_instance_id": self.server_instance_id,
                "timings": timings,
                "latency_ms": (
                    time.perf_counter() - started
                )
                * 1_000,
            }
        )
        return response

    def _trajectory_token_piece(
        self,
        token_ids: list[int],
        token_id: int,
    ) -> str:
        if self.mock:
            return MOCK_WORDS[token_id % len(MOCK_WORDS)]
        previous_text = safe_decode(
            self.tokenizer,
            token_ids,
            skip_special=True,
        )
        next_text = safe_decode(
            self.tokenizer,
            token_ids + [token_id],
            skip_special=True,
        )
        if next_text.startswith(previous_text):
            return next_text[len(previous_text) :]
        return safe_decode(
            self.tokenizer,
            [token_id],
            skip_special=False,
        )

    def _trajectory_eos_token_ids(self) -> set[int]:
        if self.mock:
            return set()
        value = getattr(self.tokenizer, "eos_token_id", None)
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set)):
            return {int(token_id) for token_id in value}
        return {int(value)}

    @staticmethod
    def _trajectory_stop_reason(
        token_id: int,
        token_text: str,
        eos_token_ids: set[int],
    ) -> str | None:
        if token_id in eos_token_ids:
            return "eos"
        if any(mark in token_text for mark in (".", "!", "?")):
            return "sentence_end"
        if "\n" in token_text or "\r" in token_text:
            return "newline"
        return None

    def _run_steering_trajectory_generation(
        self,
        *,
        test_text: str,
        test_token_ids: list[int],
        alpha_magnitude: float,
        temperature: float,
        max_new_tokens: int,
        generation_mode: str,
        seed: int,
        stepper: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
        """Decode three branch-isolated trajectories from one batched stepper."""
        branch_specs = (
            ("negative", -alpha_magnitude),
            ("baseline", 0.0),
            ("positive", alpha_magnitude),
        )
        baseline_index = 1
        branch_token_ids = [list(test_token_ids) for _ in branch_specs]
        completions = ["" for _ in branch_specs]
        token_records: list[list[dict[str, Any]]] = [
            [] for _ in branch_specs
        ]
        cumulative_logprobs = [0.0 for _ in branch_specs]
        stop_reasons: list[str | None] = [None for _ in branch_specs]
        contexts_aligned = [True for _ in branch_specs]
        branch_first_divergence: list[int | None] = [
            None for _ in branch_specs
        ]
        first_event: dict[str, Any] | None = None
        eos_token_ids = self._trajectory_eos_token_ids()
        special_ids = (
            set(getattr(self.tokenizer, "all_special_ids", []))
            if not self.mock
            else set()
        )
        rng = np.random.default_rng(seed)
        generation_started = time.perf_counter()
        next_input_ids = [list(test_token_ids) for _ in branch_specs]

        for step_index in range(max_new_tokens):
            step_started = time.perf_counter()
            logits = np.asarray(stepper(next_input_ids))
            if logits.ndim != 2 or logits.shape[0] != len(branch_specs):
                raise RuntimeError(
                    "Trajektoriový forward musí vrátit logity tří větví."
                )
            probabilities = [
                probabilities_at_temperature(row, temperature)
                for row in logits
            ]
            entropies = [
                entropy_bits_from_probabilities(row)
                for row in probabilities
            ]
            baseline_probabilities = probabilities[baseline_index]
            common_quantile = (
                float(rng.random())
                if generation_mode == "sample"
                else None
            )

            selected_ids: list[int] = []
            for branch_index, branch_probabilities in enumerate(
                probabilities
            ):
                if stop_reasons[branch_index] is not None:
                    filler = (
                        min(eos_token_ids)
                        if eos_token_ids
                        else int(np.argmax(branch_probabilities))
                    )
                    selected_ids.append(filler)
                    continue
                if generation_mode == "greedy":
                    selected_ids.append(int(np.argmax(branch_probabilities)))
                    continue
                assert common_quantile is not None
                cumulative = np.cumsum(branch_probabilities)
                sampled = int(
                    np.searchsorted(
                        cumulative,
                        common_quantile,
                        side="right",
                    )
                )
                selected_ids.append(
                    min(sampled, branch_probabilities.size - 1)
                )

            pieces = [
                self._trajectory_token_piece(
                    branch_token_ids[index],
                    selected_ids[index],
                )
                for index in range(len(branch_specs))
            ]
            step_latency_ms = (
                time.perf_counter() - step_started
            ) * 1_000

            for branch_index, (key, alpha) in enumerate(branch_specs):
                if stop_reasons[branch_index] is not None:
                    continue
                token_id = selected_ids[branch_index]
                probability = float(
                    probabilities[branch_index][token_id]
                )
                safe_probability = max(
                    probability,
                    float(np.finfo(np.float64).tiny),
                )
                logprob = math.log(safe_probability)
                cumulative_logprobs[branch_index] += logprob
                if branch_index == baseline_index:
                    divergence = {
                        "context_aligned": True,
                        "js_divergence_bits": 0.0,
                        "total_variation": 0.0,
                    }
                elif contexts_aligned[branch_index]:
                    js_bits, total_variation = distribution_distance(
                        baseline_probabilities,
                        probabilities[branch_index],
                    )
                    divergence = {
                        "context_aligned": True,
                        "js_divergence_bits": js_bits,
                        "total_variation": total_variation,
                    }
                else:
                    divergence = {
                        "context_aligned": False,
                        "js_divergence_bits": None,
                        "total_variation": None,
                    }

                token_piece = pieces[branch_index]
                stop_reason = self._trajectory_stop_reason(
                    token_id,
                    token_piece,
                    eos_token_ids,
                )
                rank = 1 + int(
                    np.count_nonzero(
                        logits[branch_index] > logits[branch_index][token_id]
                    )
                )
                if self.mock:
                    prefix_before_text = (
                        test_text + completions[branch_index]
                    )
                    prefix_after_text = (
                        prefix_before_text
                        + (
                            ""
                            if token_id in eos_token_ids
                            else token_piece
                        )
                    )
                else:
                    prefix_before_text = safe_decode(
                        self.tokenizer,
                        branch_token_ids[branch_index],
                        skip_special=True,
                    )
                    prefix_after_text = safe_decode(
                        self.tokenizer,
                        branch_token_ids[branch_index] + [token_id],
                        skip_special=True,
                    )
                record = {
                    "step": step_index + 1,
                    "step_index": step_index,
                    "token_id": token_id,
                    "text": token_piece,
                    "display": visible_token(
                        token_piece,
                        f"token {token_id}",
                    ),
                    "p_conditional": probability,
                    "logprob": logprob,
                    "cumulative_logprob": cumulative_logprobs[
                        branch_index
                    ],
                    "surprisal_bits": -math.log2(safe_probability),
                    "entropy_bits": entropies[branch_index],
                    "rank": rank,
                    "is_special": token_id in special_ids,
                    "stop_reason": stop_reason,
                    "prefix_before_text": prefix_before_text,
                    "prefix_after_text": prefix_after_text,
                    "divergence_from_baseline": divergence,
                    "latency_ms": step_latency_ms,
                }
                token_records[branch_index].append(record)
                branch_token_ids[branch_index].append(token_id)
                if token_id not in eos_token_ids:
                    completions[branch_index] += token_piece
                if stop_reason is not None:
                    stop_reasons[branch_index] = stop_reason

            baseline_token_id = selected_ids[baseline_index]
            newly_diverged: list[int] = []
            for branch_index in (0, 2):
                if (
                    contexts_aligned[branch_index]
                    and selected_ids[branch_index] != baseline_token_id
                ):
                    contexts_aligned[branch_index] = False
                    branch_first_divergence[branch_index] = step_index + 1
                    newly_diverged.append(branch_index)
            if first_event is None and newly_diverged:
                first_event = {
                    "step": step_index + 1,
                    "step_index": step_index,
                    "all_equal_before": True,
                    "diverged_branches": [
                        branch_specs[index][0]
                        for index in newly_diverged
                    ],
                    "branch_token_ids": {
                        branch_specs[index][0]: selected_ids[index]
                        for index in range(len(branch_specs))
                    },
                    "branch_token_texts": {
                        branch_specs[index][0]: pieces[index]
                        for index in range(len(branch_specs))
                    },
                }

            next_input_ids = [[token_id] for token_id in selected_ids]
            if all(reason is not None for reason in stop_reasons):
                break

        for branch_index, reason in enumerate(stop_reasons):
            if reason is None:
                stop_reasons[branch_index] = "max_new_tokens"

        branches: list[dict[str, Any]] = []
        for branch_index, (key, alpha) in enumerate(branch_specs):
            branch_tokens = token_records[branch_index]
            branches.append(
                {
                    "key": key,
                    "label": key,
                    "alpha": alpha,
                    "full_text": test_text + completions[branch_index],
                    "completion": completions[branch_index],
                    "stop_reason": stop_reasons[branch_index],
                    "generated_token_ids": [
                        token["token_id"] for token in branch_tokens
                    ],
                    "generated_token_count": len(branch_tokens),
                    "cumulative_logprob": cumulative_logprobs[
                        branch_index
                    ],
                    "first_divergence_from_baseline_step": (
                        None
                        if branch_index == baseline_index
                        else branch_first_divergence[branch_index]
                    ),
                    "tokens": branch_tokens,
                }
            )

        negative_step = branch_first_divergence[0]
        positive_step = branch_first_divergence[2]
        non_null_steps = [
            step
            for step in (negative_step, positive_step)
            if step is not None
        ]
        first_divergence = {
            "any_step": min(non_null_steps) if non_null_steps else None,
            "negative_vs_baseline_step": negative_step,
            "positive_vs_baseline_step": positive_step,
            "event": first_event,
        }
        generation_ms = (
            time.perf_counter() - generation_started
        ) * 1_000
        return branches, first_divergence, generation_ms

    def _mlx_trajectory_stepper(
        self,
        *,
        unit_direction: np.ndarray,
        native_scale: float,
        layer: int,
        alpha_magnitude: float,
    ) -> tuple[Any, dict[str, Any]]:
        """Create a request-local three-branch KV-cached MLX stepper."""
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.cache import make_prompt_cache

        body = self._steering_body()
        layers = body.layers
        cache = make_prompt_cache(self.model)
        branch_alphas = np.array(
            [-alpha_magnitude, 0.0, alpha_magnitude],
            dtype=np.float32,
        )
        branch_shifts = (
            branch_alphas[:, None]
            * float(native_scale)
            * unit_direction.astype(np.float32, copy=False)[None, :]
        )
        mlx_shifts = mx.array(branch_shifts)
        runtime: dict[str, Any] = {
            "effective_shift_norm": None,
            "effective_shift_vector": None,
            "baseline_residual_norm": None,
            "shift_dtype_name": None,
        }

        def stepper(next_input_ids: list[list[int]]) -> np.ndarray:
            ids = mx.array(next_input_ids)
            hidden = body.embed_tokens(ids)
            mask = create_attention_mask(hidden, cache[0])
            for decoder_layer, layer_cache in zip(
                layers[:layer],
                cache[:layer],
                strict=True,
            ):
                hidden = decoder_layer(
                    hidden,
                    mask=mask,
                    cache=layer_cache,
                )

            typed_shifts = mlx_shifts.astype(hidden.dtype)
            if runtime["shift_dtype_name"] is None:
                runtime["shift_dtype_name"] = str(hidden.dtype)
            if runtime["baseline_residual_norm"] is None:
                baseline_residual = hidden[1, -1, :].astype(mx.float32)
                mx.eval(baseline_residual)
                runtime["baseline_residual_norm"] = float(
                    np.linalg.norm(np.asarray(baseline_residual))
                )
            if runtime["effective_shift_norm"] is None:
                evaluated_shift = typed_shifts[2].astype(mx.float32)
                mx.eval(evaluated_shift)
                evaluated_array = np.asarray(evaluated_shift).astype(
                    np.float64,
                    copy=True,
                )
                runtime["effective_shift_norm"] = float(
                    np.linalg.norm(evaluated_array)
                )
                runtime["effective_shift_vector"] = evaluated_array
            shifted_last = hidden[:, -1:, :] + typed_shifts[:, None, :]
            if hidden.shape[1] == 1:
                hidden = shifted_last
            else:
                hidden = mx.concatenate(
                    (hidden[:, :-1, :], shifted_last),
                    axis=1,
                )

            for decoder_layer, layer_cache in zip(
                layers[layer:],
                cache[layer:],
                strict=True,
            ):
                hidden = decoder_layer(
                    hidden,
                    mask=mask,
                    cache=layer_cache,
                )
            normalized = body.norm(hidden[:, -1, :])
            lm_head = getattr(self.model, "lm_head", None)
            if lm_head is not None:
                projected = lm_head(normalized)
            else:
                as_linear = getattr(body.embed_tokens, "as_linear", None)
                if not callable(as_linear):
                    raise RuntimeError(
                        "Model nemá lm_head ani vázanou výstupní projekci."
                    )
                projected = as_linear(normalized)
            projected = projected.astype(mx.float32)
            mx.eval(projected, [item.state for item in cache])
            return np.asarray(projected).astype(
                np.float32,
                copy=True,
            )

        return stepper, runtime

    def _mock_trajectory_stepper(
        self,
        *,
        unit_direction: np.ndarray,
        native_scale: float,
        layer: int,
        alpha_magnitude: float,
        test_text: str,
    ) -> tuple[Any, dict[str, Any]]:
        projection_seed = int.from_bytes(
            hashlib.sha256(
                f"mock-steering-projection-v1\0{layer}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(projection_seed)
        projection = rng.normal(
            0.0,
            0.18,
            size=(unit_direction.size, len(MOCK_WORDS)),
        )
        direction_logits = (
            float(native_scale) * unit_direction @ projection
        )
        branch_alphas = (-alpha_magnitude, 0.0, alpha_magnitude)
        branch_texts = [test_text, test_text, test_text]

        def stepper(next_input_ids: list[list[int]]) -> np.ndarray:
            del next_input_ids
            rows: list[np.ndarray] = []
            for branch_index, alpha in enumerate(branch_alphas):
                logits, _, _ = self._mock_logits(
                    branch_texts[branch_index],
                    None,
                )
                rows.append(
                    (
                        logits.astype(np.float64, copy=False)
                        + alpha * direction_logits
                    ).astype(np.float32)
                )
            return np.stack(rows, axis=0)

        # The generic decoder owns the evolving text. Keep the mock stepper
        # synchronized by wrapping its token-piece method below.
        original_piece = self._trajectory_token_piece

        def synchronized_stepper(
            next_input_ids: list[list[int]],
        ) -> np.ndarray:
            for index, token_ids in enumerate(next_input_ids):
                if len(token_ids) == 1 and token_ids[0] < len(MOCK_WORDS):
                    branch_texts[index] += original_piece([], token_ids[0])
            return stepper(next_input_ids)

        effective_shift_vector = (
            alpha_magnitude * float(native_scale) * unit_direction
        ).astype(np.float32).astype(np.float64)
        return synchronized_stepper, {
            "effective_shift_norm": float(
                np.linalg.norm(effective_shift_vector)
            ),
            "effective_shift_vector": effective_shift_vector,
            "baseline_residual_norm": float(
                np.linalg.norm(self._mock_steering_state(test_text, layer))
            ),
            "shift_dtype_name": "float32",
        }

    def _steering_trajectory_local(
        self,
        contrast_pairs: list[dict[str, Any]],
        test_text: str,
        layer: int,
        alpha_magnitude: float,
        temperature: float,
        max_new_tokens: int,
        generation_mode: str,
        seed: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        self._validate_steering_layer(layer)
        if not 0.0 < alpha_magnitude <= 2.0:
            raise ValueError("alpha_magnitude musí být větší než 0 a nejvýše 2.")
        if not math.isfinite(temperature) or not 0.2 <= temperature <= 2.0:
            raise ValueError("Teplota musí být mezi 0,2 a 2,0.")
        if not 1 <= max_new_tokens <= MAX_STEERING_TRAJECTORY_TOKENS:
            raise ValueError(
                "max_new_tokens musí být mezi 1 a "
                f"{MAX_STEERING_TRAJECTORY_TOKENS}."
            )
        if generation_mode not in {"greedy", "sample"}:
            raise ValueError("generation_mode musí být greedy nebo sample.")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed musí být celé číslo mezi 0 a 2^32−1.")

        (
            encoded_pairs,
            prepared_pairs,
            test_text,
            test_token_ids,
            total_tokens,
        ) = self._prepare_steering_inputs(contrast_pairs, test_text)
        direction_started = time.perf_counter()
        if self.mock:
            unit_direction, direction_diagnostics = (
                self._mock_steering_direction(
                    prepared_pairs,
                    layer,
                )
            )
            contrast_forward_ms = 0.0
            stepper, shift_runtime = self._mock_trajectory_stepper(
                unit_direction=unit_direction,
                native_scale=float(direction_diagnostics["native_scale"]),
                layer=layer,
                alpha_magnitude=alpha_magnitude,
                test_text=test_text,
            )
        else:
            (
                unit_direction,
                direction_diagnostics,
                contrast_forward_ms,
            ) = self._mlx_steering_direction(encoded_pairs, layer)
            stepper, shift_runtime = (
                self._mlx_trajectory_stepper(
                    unit_direction=unit_direction,
                    native_scale=float(
                        direction_diagnostics["native_scale"]
                    ),
                    layer=layer,
                    alpha_magnitude=alpha_magnitude,
                )
            )
        direction_ms = (
            time.perf_counter() - direction_started
        ) * 1_000

        branches, first_divergence, generation_ms = (
            self._run_steering_trajectory_generation(
                test_text=test_text,
                test_token_ids=test_token_ids,
                alpha_magnitude=alpha_magnitude,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                generation_mode=generation_mode,
                seed=seed,
                stepper=stepper,
            )
        )
        effective_shift_norm_value = shift_runtime[
            "effective_shift_norm"
        ]
        if effective_shift_norm_value is None:
            raise RuntimeError(
                "Trajektorie nevyhodnotila efektivní normu zásahu."
            )
        effective_shift_norm = float(effective_shift_norm_value)
        direction_diagnostics["effective_shift_norm"] = (
            effective_shift_norm
        )

        pair_diagnostics = [
            {
                **pair,
                "delta_norm": norm,
                "cosine_to_direction": cosine,
            }
            for pair, norm, cosine in zip(
                prepared_pairs,
                direction_diagnostics["pair_norms"],
                direction_diagnostics["pair_cosines_to_direction"],
                strict=True,
            )
        ]
        warnings: list[dict[str, str]] = [
            {
                "code": "autoregressive_context_divergence",
                "message": (
                    "Po prvním rozdílném tokenu větve používají různé "
                    "kontexty; JS a TV se proto dále nevykazují."
                ),
            }
        ]
        if alpha_magnitude > 1.0:
            warnings.append(
                {
                    "code": "extrapolation",
                    "message": (
                        "α > 1 překračuje typickou velikost "
                        "kontrastního rozdílu."
                    ),
                }
            )
        if direction_diagnostics["resultant_length"] < 0.25:
            warnings.append(
                {
                    "code": "weak_pair_coherence",
                    "message": (
                        "Kontrastní páry mají slabý společný směr; "
                        "interpretuj zásah opatrně."
                    ),
                }
            )
        if self.is_quantized():
            warnings.append(
                {
                    "code": "quantized_model",
                    "message": (
                        "Výsledek pochází z kvantovaného nebo hybridního "
                        "modelu; těsné změny mohou být numericky citlivé."
                    ),
                }
            )

        total_ms = (time.perf_counter() - started) * 1_000
        return {
            "test_text": test_text,
            "test_token_ids": test_token_ids,
            "test_token_count": len(test_token_ids),
            "alpha_magnitude": alpha_magnitude,
            "layer": layer,
            "layer_numbering": "one_based",
            "temperature": temperature,
            "probability_kind": (
                "raw_model"
                if math.isclose(temperature, 1.0)
                else "temperature_adjusted"
            ),
            "generation_mode": generation_mode,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "branches": branches,
            "first_divergence": first_divergence,
            "steering": {
                "kind": "contrastive_activation_addition",
                "hook": "post_decoder_block_residual_last_position",
                "layer": layer,
                "layer_numbering": "one_based",
                "alpha_magnitude": alpha_magnitude,
                "branch_alphas": {
                    "negative": -alpha_magnitude,
                    "baseline": 0.0,
                    "positive": alpha_magnitude,
                },
                "steering_applied_every_step": True,
                "direction_computations": 1,
                "pair_count": len(prepared_pairs),
                "total_input_tokens": total_tokens,
                "aggregation": direction_diagnostics["aggregation"],
                "native_scale": direction_diagnostics["native_scale"],
                "applied_shift_norm": (
                    alpha_magnitude
                    * float(direction_diagnostics["native_scale"])
                ),
                "effective_shift_norm": effective_shift_norm,
                "resultant_length": direction_diagnostics[
                    "resultant_length"
                ],
                "cancellation_fraction": direction_diagnostics[
                    "cancellation_fraction"
                ],
                "pair_diagnostics": pair_diagnostics,
                "boundary_policy": "same_final_token_id_within_each_pair",
                "execution": "single_serial_worker_job",
                "batch_strategy": "three_branches_one_batched_forward",
                "cache_policy": (
                    "request_local_batched_branch_isolated_kv_cache_"
                    "no_global_mutation"
                ),
            },
            "warnings": warnings,
            "cached": False,
            "model_id": self.active_model_id,
            "model_label": self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
            "timing_ms": {
                "contrast_forward": contrast_forward_ms,
                "direction_total": direction_ms,
                "generation": generation_ms,
                "total": total_ms,
            },
            "latency_ms": total_ms,
        }

    def _prepare_teacher_forced_continuation(
        self,
        test_text: str,
        test_token_ids: list[int],
        continuation_text: str,
    ) -> tuple[str, list[int], list[str], list[int]]:
        """Tokenize an exact continuation without silently moving its boundary."""
        if not isinstance(continuation_text, str) or not continuation_text.strip():
            raise ValueError(
                "continuation_text musí být neprázdný řetězec."
            )

        full_text = test_text + continuation_text
        if self.mock:
            pieces = re.findall(r"\s+|[^\s]+", continuation_text)
            continuation_token_ids = [
                int.from_bytes(
                    hashlib.sha256(piece.encode("utf-8")).digest()[:4],
                    "big",
                )
                % len(MOCK_WORDS)
                for piece in pieces
            ]
            full_token_ids = list(test_token_ids) + continuation_token_ids
        else:
            full_token_ids = self._encode_steering_text(full_text)
            prefix_length = len(test_token_ids)
            if full_token_ids[:prefix_length] != test_token_ids:
                raise ValueError(
                    "Tokenizace na hranici promptu a pokračování změnila "
                    "poslední token promptu. Přidej na začátek pokračování "
                    "mezeru nebo jinou explicitní hranici."
                )
            continuation_token_ids = full_token_ids[prefix_length:]
            pieces = []
            running_ids = list(test_token_ids)
            for token_id in continuation_token_ids:
                pieces.append(
                    self._trajectory_token_piece(running_ids, token_id)
                )
                running_ids.append(token_id)

        if not continuation_token_ids:
            raise ValueError(
                "Pokračování po tokenizaci neobsahuje žádný nový token."
            )
        if len(continuation_token_ids) > MAX_STEERING_TRAJECTORY_TOKENS:
            raise ValueError(
                f"Pokračování má {len(continuation_token_ids)} tokenů; "
                f"limit je {MAX_STEERING_TRAJECTORY_TOKENS}."
            )
        return (
            full_text,
            continuation_token_ids,
            pieces,
            full_token_ids,
        )

    def _score_teacher_forced_sequence(
        self,
        *,
        test_text: str,
        test_token_ids: list[int],
        continuation_text: str,
        continuation_token_ids: list[int],
        continuation_pieces: list[str],
        alpha_magnitude: float,
        temperature: float,
        stepper: Any,
    ) -> tuple[list[dict[str, Any]], float]:
        """Score one fixed path under −α, 0 and +α with aligned contexts."""
        branch_specs = (
            ("negative", -alpha_magnitude),
            ("baseline", 0.0),
            ("positive", alpha_magnitude),
        )
        baseline_index = 1
        token_records: list[list[dict[str, Any]]] = [
            [] for _ in branch_specs
        ]
        cumulative_logprobs = [0.0 for _ in branch_specs]
        special_ids = (
            set(getattr(self.tokenizer, "all_special_ids", []))
            if not self.mock
            else set()
        )
        scoring_started = time.perf_counter()
        next_input_ids = [list(test_token_ids) for _ in branch_specs]
        shared_prefix_ids = list(test_token_ids)
        shared_prefix_text = test_text

        for token_index, (token_id, token_piece) in enumerate(
            zip(
                continuation_token_ids,
                continuation_pieces,
                strict=True,
            )
        ):
            step_started = time.perf_counter()
            logits = np.asarray(stepper(next_input_ids))
            if logits.ndim != 2 or logits.shape[0] != len(branch_specs):
                raise RuntimeError(
                    "Teacher-forced forward musí vrátit logity tří větví."
                )
            if not 0 <= token_id < logits.shape[1]:
                raise RuntimeError(
                    f"Token pokračování {token_id} leží mimo slovník "
                    f"o velikosti {logits.shape[1]}."
                )
            probabilities = [
                probabilities_at_temperature(row, temperature)
                for row in logits
            ]
            entropies = [
                entropy_bits_from_probabilities(row)
                for row in probabilities
            ]
            baseline_probabilities = probabilities[baseline_index]
            step_latency_ms = (
                time.perf_counter() - step_started
            ) * 1_000
            if self.mock:
                prefix_before_text = shared_prefix_text
                prefix_after_text = shared_prefix_text + token_piece
            else:
                prefix_before_text = safe_decode(
                    self.tokenizer,
                    shared_prefix_ids,
                    skip_special=True,
                )
                prefix_after_text = safe_decode(
                    self.tokenizer,
                    shared_prefix_ids + [token_id],
                    skip_special=True,
                )

            for branch_index, (key, alpha) in enumerate(branch_specs):
                branch_probabilities = probabilities[branch_index]
                probability = float(branch_probabilities[token_id])
                safe_probability = max(
                    probability,
                    float(np.finfo(np.float64).tiny),
                )
                logprob = math.log(safe_probability)
                cumulative_logprobs[branch_index] += logprob
                if branch_index == baseline_index:
                    js_bits = 0.0
                    total_variation = 0.0
                else:
                    js_bits, total_variation = distribution_distance(
                        baseline_probabilities,
                        branch_probabilities,
                    )
                rank = 1 + int(
                    np.count_nonzero(
                        logits[branch_index]
                        > logits[branch_index][token_id]
                    )
                )
                token_count = token_index + 1
                token_records[branch_index].append(
                    {
                        "step": token_count,
                        "step_index": token_index,
                        "token_id": token_id,
                        "text": token_piece,
                        "display": visible_token(
                            token_piece,
                            f"token {token_id}",
                        ),
                        "p_conditional": probability,
                        "logprob": logprob,
                        "cumulative_logprob": cumulative_logprobs[
                            branch_index
                        ],
                        "mean_logprob": (
                            cumulative_logprobs[branch_index] / token_count
                        ),
                        "surprisal_bits": -math.log2(safe_probability),
                        "entropy_bits": entropies[branch_index],
                        "rank": rank,
                        "is_special": token_id in special_ids,
                        "prefix_before_text": prefix_before_text,
                        "prefix_after_text": prefix_after_text,
                        "divergence_from_baseline": {
                            "context_aligned": True,
                            "js_divergence_bits": js_bits,
                            "total_variation": total_variation,
                        },
                        "latency_ms": step_latency_ms,
                    }
                )

            next_input_ids = [[token_id] for _ in branch_specs]
            shared_prefix_ids.append(token_id)
            shared_prefix_text = prefix_after_text

        branches: list[dict[str, Any]] = []
        token_count = len(continuation_token_ids)
        for branch_index, (key, alpha) in enumerate(branch_specs):
            cumulative_logprob = cumulative_logprobs[branch_index]
            mean_logprob = cumulative_logprob / token_count
            branches.append(
                {
                    "key": key,
                    "label": key,
                    "alpha": alpha,
                    "test_text": test_text,
                    "continuation_text": continuation_text,
                    "full_text": test_text + continuation_text,
                    "continuation_token_ids": list(
                        continuation_token_ids
                    ),
                    "token_count": token_count,
                    "cumulative_logprob": cumulative_logprob,
                    "total_logprob": cumulative_logprob,
                    "mean_logprob": mean_logprob,
                    "length_normalized_logprob": mean_logprob,
                    "geometric_mean_probability": math.exp(mean_logprob),
                    "perplexity": math.exp(-mean_logprob),
                    "tokens": token_records[branch_index],
                }
            )
        scoring_ms = (
            time.perf_counter() - scoring_started
        ) * 1_000
        return branches, scoring_ms

    @staticmethod
    def _steering_monotonicity(
        values: list[float],
        *,
        tolerance: float = STEERING_MONOTONICITY_TOLERANCE,
    ) -> dict[str, Any]:
        if len(values) != 3:
            raise ValueError(
                "Diagnostika monotónnosti očekává právě tři hodnoty."
            )
        first_delta = float(values[1] - values[0])
        second_delta = float(values[2] - values[1])
        return {
            "order": ["negative", "baseline", "positive"],
            "tolerance": tolerance,
            "strictly_increasing": (
                first_delta > tolerance and second_delta > tolerance
            ),
            "nondecreasing": (
                first_delta >= -tolerance and second_delta >= -tolerance
            ),
            "negative_to_baseline_delta": first_delta,
            "baseline_to_positive_delta": second_delta,
        }

    def _teacher_forced_direction(
        self,
        *,
        encoded_pairs: list[tuple[list[int], list[int]]],
        prepared_pairs: list[dict[str, Any]],
        test_text: str,
        layer: int,
        alpha_magnitude: float,
    ) -> tuple[
        np.ndarray,
        dict[str, Any],
        float,
        Any,
    ]:
        """Compute one direction and return a fresh-stepper factory."""
        if self.mock:
            unit_direction, diagnostics = self._mock_steering_direction(
                prepared_pairs,
                layer,
            )
            contrast_forward_ms = 0.0

            def stepper_factory() -> tuple[Any, dict[str, Any]]:
                return self._mock_trajectory_stepper(
                    unit_direction=unit_direction,
                    native_scale=float(diagnostics["native_scale"]),
                    layer=layer,
                    alpha_magnitude=alpha_magnitude,
                    test_text=test_text,
                )

        else:
            (
                unit_direction,
                diagnostics,
                contrast_forward_ms,
            ) = self._mlx_steering_direction(encoded_pairs, layer)

            def stepper_factory() -> tuple[Any, dict[str, Any]]:
                return self._mlx_trajectory_stepper(
                    unit_direction=unit_direction,
                    native_scale=float(diagnostics["native_scale"]),
                    layer=layer,
                    alpha_magnitude=alpha_magnitude,
                )

        return (
            unit_direction,
            diagnostics,
            contrast_forward_ms,
            stepper_factory,
        )

    def _steering_teacher_forced_warnings(
        self,
        direction_diagnostics: dict[str, Any],
        alpha_magnitude: float,
    ) -> list[dict[str, str]]:
        warnings: list[dict[str, str]] = [
            {
                "code": "teacher_forced_shared_context",
                "message": (
                    "Všechny větve skórují stejné tokeny ve stejném "
                    "kontextu; nejde o volnou generaci."
                ),
            }
        ]
        if alpha_magnitude > 1.0:
            warnings.append(
                {
                    "code": "extrapolation",
                    "message": (
                        "α > 1 překračuje typickou velikost "
                        "kontrastního rozdílu."
                    ),
                }
            )
        if direction_diagnostics["resultant_length"] < 0.25:
            warnings.append(
                {
                    "code": "weak_pair_coherence",
                    "message": (
                        "Kontrastní páry mají slabý společný směr; "
                        "interpretuj zásah opatrně."
                    ),
                }
            )
        if any(
            cosine < 0
            for cosine in direction_diagnostics[
                "pair_cosines_to_direction"
            ]
        ):
            warnings.append(
                {
                    "code": "opposed_pair",
                    "message": (
                        "Alespoň jeden pár míří proti agregovanému směru."
                    ),
                }
            )
        if self.is_quantized():
            warnings.append(
                {
                    "code": "quantized_model",
                    "message": (
                        "Výsledek pochází z kvantovaného nebo hybridního "
                        "modelu; těsné změny mohou být numericky citlivé."
                    ),
                }
            )
        return warnings

    def _teacher_forced_steering_payload(
        self,
        *,
        layer: int,
        alpha_magnitude: float,
        total_input_tokens: int,
        prepared_pairs: list[dict[str, Any]],
        direction_diagnostics: dict[str, Any],
        effective_shift_norm: float,
        cache_policy: str,
    ) -> dict[str, Any]:
        pair_diagnostics = [
            {
                **pair,
                "delta_norm": norm,
                "cosine_to_direction": cosine,
            }
            for pair, norm, cosine in zip(
                prepared_pairs,
                direction_diagnostics["pair_norms"],
                direction_diagnostics["pair_cosines_to_direction"],
                strict=True,
            )
        ]
        return {
            "kind": "contrastive_activation_addition",
            "hook": "post_decoder_block_residual_last_position",
            "layer": layer,
            "layer_numbering": "one_based",
            "alpha_magnitude": alpha_magnitude,
            "branch_alphas": {
                "negative": -alpha_magnitude,
                "baseline": 0.0,
                "positive": alpha_magnitude,
            },
            "steering_applied_every_step": True,
            "teacher_forced": True,
            "shared_context_at_every_step": True,
            "direction_computations": 1,
            "pair_count": len(prepared_pairs),
            "total_input_tokens": total_input_tokens,
            "aggregation": direction_diagnostics["aggregation"],
            "native_scale": direction_diagnostics["native_scale"],
            "applied_shift_norm": (
                alpha_magnitude
                * float(direction_diagnostics["native_scale"])
            ),
            "effective_shift_norm": effective_shift_norm,
            "resultant_length": direction_diagnostics[
                "resultant_length"
            ],
            "cancellation_fraction": direction_diagnostics[
                "cancellation_fraction"
            ],
            "pair_diagnostics": pair_diagnostics,
            "boundary_policy": "same_final_token_id_within_each_pair",
            "continuation_boundary_policy": (
                "prefix_token_ids_must_remain_exact_prefix_of_full_sequence"
            ),
            "execution": "single_serial_worker_job",
            "batch_strategy": "three_alphas_one_batched_forward",
            "cache_policy": cache_policy,
        }

    def _steering_aligned_path_local(
        self,
        contrast_pairs: list[dict[str, Any]],
        test_text: str,
        continuation_text: str,
        layer: int,
        alpha_magnitude: float,
        temperature: float,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        self._validate_steering_layer(layer)
        if (
            not math.isfinite(alpha_magnitude)
            or not 0.0 < alpha_magnitude <= 2.0
        ):
            raise ValueError(
                "alpha_magnitude musí být větší než 0 a nejvýše 2."
            )
        if (
            not math.isfinite(temperature)
            or not 0.2 <= temperature <= 2.0
        ):
            raise ValueError("Teplota musí být mezi 0,2 a 2,0.")

        (
            encoded_pairs,
            prepared_pairs,
            test_text,
            test_token_ids,
            total_tokens,
        ) = self._prepare_steering_inputs(contrast_pairs, test_text)
        (
            full_text,
            continuation_token_ids,
            continuation_pieces,
            full_token_ids,
        ) = self._prepare_teacher_forced_continuation(
            test_text,
            test_token_ids,
            continuation_text,
        )
        total_tokens += len(continuation_token_ids)
        if total_tokens > MAX_STEERING_TOTAL_TOKENS:
            raise ValueError(
                f"Steeringové texty mají dohromady {total_tokens} tokenů; "
                f"souhrnný limit je {MAX_STEERING_TOTAL_TOKENS}."
            )

        direction_started = time.perf_counter()
        (
            _unit_direction,
            direction_diagnostics,
            contrast_forward_ms,
            stepper_factory,
        ) = self._teacher_forced_direction(
            encoded_pairs=encoded_pairs,
            prepared_pairs=prepared_pairs,
            test_text=test_text,
            layer=layer,
            alpha_magnitude=alpha_magnitude,
        )
        direction_ms = (
            time.perf_counter() - direction_started
        ) * 1_000
        stepper, shift_runtime = stepper_factory()
        branches, scoring_ms = self._score_teacher_forced_sequence(
            test_text=test_text,
            test_token_ids=test_token_ids,
            continuation_text=continuation_text,
            continuation_token_ids=continuation_token_ids,
            continuation_pieces=continuation_pieces,
            alpha_magnitude=alpha_magnitude,
            temperature=temperature,
            stepper=stepper,
        )
        effective_shift_norm_value = shift_runtime[
            "effective_shift_norm"
        ]
        if effective_shift_norm_value is None:
            raise RuntimeError(
                "Teacher-forced cesta nevyhodnotila efektivní normu zásahu."
            )
        effective_shift_norm = float(effective_shift_norm_value)
        direction_diagnostics["effective_shift_norm"] = (
            effective_shift_norm
        )
        total_ms = (time.perf_counter() - started) * 1_000
        return {
            "analysis_kind": "steering_aligned_path",
            "test_text": test_text,
            "test_token_ids": test_token_ids,
            "continuation_text": continuation_text,
            "continuation_token_ids": continuation_token_ids,
            "continuation_token_count": len(continuation_token_ids),
            "full_text": full_text,
            "full_token_ids": full_token_ids,
            "temperature": temperature,
            "probability_kind": (
                "raw_model"
                if math.isclose(temperature, 1.0)
                else "temperature_adjusted"
            ),
            "branches": branches,
            "steering": self._teacher_forced_steering_payload(
                layer=layer,
                alpha_magnitude=alpha_magnitude,
                total_input_tokens=total_tokens,
                prepared_pairs=prepared_pairs,
                direction_diagnostics=direction_diagnostics,
                effective_shift_norm=effective_shift_norm,
                cache_policy=(
                    "request_local_batched_aligned_kv_cache_"
                    "no_global_mutation"
                ),
            ),
            "warnings": self._steering_teacher_forced_warnings(
                direction_diagnostics,
                alpha_magnitude,
            ),
            "cached": False,
            "model_id": self.active_model_id,
            "model_label": self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
            "timing_ms": {
                "contrast_forward": contrast_forward_ms,
                "direction_total": direction_ms,
                "scoring": scoring_ms,
                "total": total_ms,
            },
            "latency_ms": total_ms,
        }

    def _prepare_phrase_score_pairs(
        self,
        *,
        test_text: str,
        test_token_ids: list[int],
        continuation_pairs: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        if not isinstance(continuation_pairs, list) or not (
            1 <= len(continuation_pairs) <= MAX_STEERING_PHRASE_PAIRS
        ):
            raise ValueError(
                "continuation_pairs musí obsahovat 1 až "
                f"{MAX_STEERING_PHRASE_PAIRS} párů."
            )
        prepared: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        continuation_token_total = 0
        for pair_index, pair in enumerate(continuation_pairs, start=1):
            if not isinstance(pair, dict):
                raise ValueError(
                    f"Pokračovací pár {pair_index} musí být JSON objekt."
                )
            negative_text = pair.get("negative_text")
            positive_text = pair.get("positive_text")
            label = pair.get("label")
            if label is not None and (
                not isinstance(label, str) or len(label) > 120
            ):
                raise ValueError(
                    f"Pár {pair_index}: label musí být řetězec "
                    "o délce nejvýše 120 znaků."
                )
            for name, value in (
                ("negative_text", negative_text),
                ("positive_text", positive_text),
            ):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Pár {pair_index}: {name} musí být neprázdný "
                        "řetězec."
                    )
            assert isinstance(negative_text, str)
            assert isinstance(positive_text, str)
            if negative_text == positive_text:
                raise ValueError(
                    f"Pokračovací pár {pair_index} obsahuje dvě "
                    "shodné varianty."
                )
            pair_key = tuple(sorted((negative_text, positive_text)))
            if pair_key in seen_pairs:
                raise ValueError(
                    f"Pokračovací pár {pair_index} je duplicitní "
                    "nebo obrací již použitý pár."
                )
            seen_pairs.add(pair_key)

            negative = self._prepare_teacher_forced_continuation(
                test_text,
                test_token_ids,
                negative_text,
            )
            positive = self._prepare_teacher_forced_continuation(
                test_text,
                test_token_ids,
                positive_text,
            )
            continuation_token_total += (
                len(negative[1]) + len(positive[1])
            )
            prepared.append(
                {
                    "index": pair_index - 1,
                    "label": label.strip() if isinstance(label, str) else None,
                    "negative": {
                        "text": negative_text,
                        "full_text": negative[0],
                        "token_ids": negative[1],
                        "pieces": negative[2],
                        "full_token_ids": negative[3],
                    },
                    "positive": {
                        "text": positive_text,
                        "full_text": positive[0],
                        "token_ids": positive[1],
                        "pieces": positive[2],
                        "full_token_ids": positive[3],
                    },
                }
            )
        return prepared, continuation_token_total

    def _steering_phrase_score_local(
        self,
        contrast_pairs: list[dict[str, Any]],
        test_text: str,
        continuation_pairs: list[dict[str, Any]],
        layer: int,
        alpha_magnitude: float,
        temperature: float,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        self._validate_steering_layer(layer)
        if (
            not math.isfinite(alpha_magnitude)
            or not 0.0 < alpha_magnitude <= 2.0
        ):
            raise ValueError(
                "alpha_magnitude musí být větší než 0 a nejvýše 2."
            )
        if (
            not math.isfinite(temperature)
            or not 0.2 <= temperature <= 2.0
        ):
            raise ValueError("Teplota musí být mezi 0,2 a 2,0.")

        (
            encoded_pairs,
            prepared_pairs,
            test_text,
            test_token_ids,
            total_tokens,
        ) = self._prepare_steering_inputs(contrast_pairs, test_text)
        prepared_continuations, continuation_token_total = (
            self._prepare_phrase_score_pairs(
                test_text=test_text,
                test_token_ids=test_token_ids,
                continuation_pairs=continuation_pairs,
            )
        )
        total_tokens += continuation_token_total
        if total_tokens > MAX_STEERING_TOTAL_TOKENS:
            raise ValueError(
                f"Steeringové texty mají dohromady {total_tokens} tokenů; "
                f"souhrnný limit je {MAX_STEERING_TOTAL_TOKENS}."
            )

        direction_started = time.perf_counter()
        (
            _unit_direction,
            direction_diagnostics,
            contrast_forward_ms,
            stepper_factory,
        ) = self._teacher_forced_direction(
            encoded_pairs=encoded_pairs,
            prepared_pairs=prepared_pairs,
            test_text=test_text,
            layer=layer,
            alpha_magnitude=alpha_magnitude,
        )
        direction_ms = (
            time.perf_counter() - direction_started
        ) * 1_000
        scoring_started = time.perf_counter()
        items: list[dict[str, Any]] = []
        effective_shift_norm: float | None = None
        alpha_order = ("negative", "baseline", "positive")
        alpha_values = (
            -alpha_magnitude,
            0.0,
            alpha_magnitude,
        )

        for prepared_pair in prepared_continuations:
            sequence_payloads: dict[str, dict[str, Any]] = {}
            for pole in ("negative", "positive"):
                sequence = prepared_pair[pole]
                stepper, shift_runtime = stepper_factory()
                branches, _sequence_ms = (
                    self._score_teacher_forced_sequence(
                        test_text=test_text,
                        test_token_ids=test_token_ids,
                        continuation_text=sequence["text"],
                        continuation_token_ids=sequence["token_ids"],
                        continuation_pieces=sequence["pieces"],
                        alpha_magnitude=alpha_magnitude,
                        temperature=temperature,
                        stepper=stepper,
                    )
                )
                runtime_norm = shift_runtime["effective_shift_norm"]
                if runtime_norm is None:
                    raise RuntimeError(
                        "Skórování frází nevyhodnotilo efektivní "
                        "normu zásahu."
                    )
                if effective_shift_norm is None:
                    effective_shift_norm = float(runtime_norm)
                scores = [
                    {
                        "key": branch["key"],
                        "alpha": branch["alpha"],
                        "token_count": branch["token_count"],
                        "cumulative_logprob": branch[
                            "cumulative_logprob"
                        ],
                        "total_logprob": branch["total_logprob"],
                        "mean_logprob": branch["mean_logprob"],
                        "length_normalized_logprob": branch[
                            "length_normalized_logprob"
                        ],
                        "geometric_mean_probability": branch[
                            "geometric_mean_probability"
                        ],
                        "perplexity": branch["perplexity"],
                    }
                    for branch in branches
                ]
                sequence_payloads[pole] = {
                    "text": sequence["text"],
                    "full_text": sequence["full_text"],
                    "token_ids": sequence["token_ids"],
                    "token_count": len(sequence["token_ids"]),
                    "scores": scores,
                }

            negative_scores = sequence_payloads["negative"]["scores"]
            positive_scores = sequence_payloads["positive"]["scores"]
            contrasts: list[dict[str, Any]] = []
            contrast_values: list[float] = []
            for branch_index, (key, alpha) in enumerate(
                zip(alpha_order, alpha_values, strict=True)
            ):
                negative_score = negative_scores[branch_index]
                positive_score = positive_scores[branch_index]
                normalized_contrast = float(
                    positive_score["mean_logprob"]
                    - negative_score["mean_logprob"]
                )
                contrast_values.append(normalized_contrast)
                contrasts.append(
                    {
                        "key": key,
                        "alpha": alpha,
                        "positive_minus_negative_mean_logprob": (
                            normalized_contrast
                        ),
                        "positive_minus_negative_total_logprob": float(
                            positive_score["cumulative_logprob"]
                            - negative_score["cumulative_logprob"]
                        ),
                    }
                )
            items.append(
                {
                    "index": prepared_pair["index"],
                    "label": prepared_pair["label"],
                    "negative": sequence_payloads["negative"],
                    "positive": sequence_payloads["positive"],
                    "contrasts": contrasts,
                    "monotonicity": self._steering_monotonicity(
                        contrast_values
                    ),
                }
            )

        assert effective_shift_norm is not None
        direction_diagnostics["effective_shift_norm"] = (
            effective_shift_norm
        )
        mean_contrasts = [
            float(
                np.mean(
                    [
                        item["contrasts"][branch_index][
                            "positive_minus_negative_mean_logprob"
                        ]
                        for item in items
                    ]
                )
            )
            for branch_index in range(3)
        ]
        aggregate_monotonicity = self._steering_monotonicity(
            mean_contrasts
        )
        aggregate_monotonicity.update(
            {
                "pairs_strictly_increasing": sum(
                    bool(item["monotonicity"]["strictly_increasing"])
                    for item in items
                ),
                "pairs_nondecreasing": sum(
                    bool(item["monotonicity"]["nondecreasing"])
                    for item in items
                ),
                "pair_count": len(items),
            }
        )
        scoring_ms = (
            time.perf_counter() - scoring_started
        ) * 1_000
        total_ms = (time.perf_counter() - started) * 1_000
        return {
            "analysis_kind": "steering_phrase_score",
            "test_text": test_text,
            "test_token_ids": test_token_ids,
            "pair_count": len(items),
            "temperature": temperature,
            "probability_kind": (
                "raw_model"
                if math.isclose(temperature, 1.0)
                else "temperature_adjusted"
            ),
            "items": items,
            "summary": {
                "contrast_definition": (
                    "positive_mean_logprob_minus_negative_mean_logprob"
                ),
                "alpha_order": list(alpha_order),
                "alpha_values": list(alpha_values),
                "mean_contrasts": [
                    {
                        "key": key,
                        "alpha": alpha,
                        "positive_minus_negative_mean_logprob": value,
                    }
                    for key, alpha, value in zip(
                        alpha_order,
                        alpha_values,
                        mean_contrasts,
                        strict=True,
                    )
                ],
                "monotonicity": aggregate_monotonicity,
            },
            "steering": self._teacher_forced_steering_payload(
                layer=layer,
                alpha_magnitude=alpha_magnitude,
                total_input_tokens=total_tokens,
                prepared_pairs=prepared_pairs,
                direction_diagnostics=direction_diagnostics,
                effective_shift_norm=effective_shift_norm,
                cache_policy=(
                    "request_local_per_sequence_three_alpha_batched_kv_"
                    "cache_no_global_mutation"
                ),
            ),
            "warnings": self._steering_teacher_forced_warnings(
                direction_diagnostics,
                alpha_magnitude,
            ),
            "cached": False,
            "model_id": self.active_model_id,
            "model_label": self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
            "timing_ms": {
                "contrast_forward": contrast_forward_ms,
                "direction_total": direction_ms,
                "scoring": scoring_ms,
                "total": total_ms,
            },
            "latency_ms": total_ms,
        }

    def _teacher_forced_stepper_factory_for_direction(
        self,
        *,
        unit_direction: np.ndarray,
        native_scale: float,
        layer: int,
        alpha_magnitude: float,
        test_text: str,
    ) -> Any:
        """Build fresh request-local steppers for one already-derived direction."""
        if self.mock:

            def stepper_factory() -> tuple[Any, dict[str, Any]]:
                return self._mock_trajectory_stepper(
                    unit_direction=unit_direction,
                    native_scale=native_scale,
                    layer=layer,
                    alpha_magnitude=alpha_magnitude,
                    test_text=test_text,
                )

        else:

            def stepper_factory() -> tuple[Any, dict[str, Any]]:
                return self._mlx_trajectory_stepper(
                    unit_direction=unit_direction,
                    native_scale=native_scale,
                    layer=layer,
                    alpha_magnitude=alpha_magnitude,
                )

        return stepper_factory

    def _score_matched_continuation_effect(
        self,
        *,
        test_text: str,
        test_token_ids: list[int],
        prepared_continuations: list[dict[str, Any]],
        alpha_magnitude: float,
        temperature: float,
        stepper_factory: Any,
    ) -> dict[str, Any]:
        """Score a direction with the symmetric matched-pair estimand.

        For pair i, C_i(alpha) is the length-normalized log-probability
        contrast positive minus negative. The pair effect is
        0.5 * (C_i(+alpha) - C_i(-alpha)); the direction effect is the mean
        across held-out continuation pairs.
        """
        started = time.perf_counter()
        alpha_keys = ("negative", "baseline", "positive")
        alpha_values = (-alpha_magnitude, 0.0, alpha_magnitude)
        per_pair: list[dict[str, Any]] = []
        effective_shift_norm: float | None = None
        effective_shift_vector: np.ndarray | None = None
        baseline_prefix_residual_norm: float | None = None
        shift_dtype_name: str | None = None
        next_token: dict[str, float] | None = None

        for prepared_pair in prepared_continuations:
            sequence_payloads: dict[str, dict[str, Any]] = {}
            for pole in ("negative", "positive"):
                sequence = prepared_pair[pole]
                stepper, runtime = stepper_factory()
                branches, _ = self._score_teacher_forced_sequence(
                    test_text=test_text,
                    test_token_ids=test_token_ids,
                    continuation_text=sequence["text"],
                    continuation_token_ids=sequence["token_ids"],
                    continuation_pieces=sequence["pieces"],
                    alpha_magnitude=alpha_magnitude,
                    temperature=temperature,
                    stepper=stepper,
                )
                runtime_shift = runtime.get("effective_shift_norm")
                runtime_shift_vector = runtime.get(
                    "effective_shift_vector"
                )
                runtime_residual = runtime.get("baseline_residual_norm")
                runtime_dtype = runtime.get("shift_dtype_name")
                if (
                    runtime_shift is None
                    or not isinstance(runtime_shift_vector, np.ndarray)
                    or runtime_residual is None
                    or not isinstance(runtime_dtype, str)
                ):
                    raise RuntimeError(
                        "Kontrolní skórování nevyhodnotilo normu zásahu "
                        "nebo baseline rezidua či dtype."
                    )
                if effective_shift_norm is None:
                    effective_shift_norm = float(runtime_shift)
                    effective_shift_vector = np.asarray(
                        runtime_shift_vector,
                        dtype=np.float64,
                    ).copy()
                if baseline_prefix_residual_norm is None:
                    baseline_prefix_residual_norm = float(runtime_residual)
                if shift_dtype_name is None:
                    shift_dtype_name = runtime_dtype
                elif shift_dtype_name != runtime_dtype:
                    raise RuntimeError(
                        "Dtype steeringového posunu se v jednom requestu "
                        "neočekávaně změnil."
                    )

                mean_logprobs = {
                    branch["key"]: float(branch["mean_logprob"])
                    for branch in branches
                }
                sequence_payloads[pole] = {
                    "text": sequence["text"],
                    "token_count": len(sequence["token_ids"]),
                    "mean_logprobs": mean_logprobs,
                    "total_logprobs": {branch["key"]: float(branch["cumulative_logprob"]) for branch in branches},
                }
                if next_token is None:
                    baseline_first = branches[1]["tokens"][0]
                    negative_first = branches[0]["tokens"][0]
                    positive_first = branches[2]["tokens"][0]
                    next_token = {
                        "baseline_entropy_bits": float(
                            baseline_first["entropy_bits"]
                        ),
                        "negative_entropy_bits": float(
                            negative_first["entropy_bits"]
                        ),
                        "positive_entropy_bits": float(
                            positive_first["entropy_bits"]
                        ),
                        "negative_minus_baseline_entropy_bits": float(
                            negative_first["entropy_bits"]
                            - baseline_first["entropy_bits"]
                        ),
                        "positive_minus_baseline_entropy_bits": float(
                            positive_first["entropy_bits"]
                            - baseline_first["entropy_bits"]
                        ),
                        "negative_vs_baseline_js_bits": float(
                            negative_first["divergence_from_baseline"][
                                "js_divergence_bits"
                            ]
                        ),
                        "positive_vs_baseline_js_bits": float(
                            positive_first["divergence_from_baseline"][
                                "js_divergence_bits"
                            ]
                        ),
                    }

            contrasts = {
                key: float(
                    sequence_payloads["positive"]["mean_logprobs"][key]
                    - sequence_payloads["negative"]["mean_logprobs"][key]
                )
                for key in alpha_keys
            }
            symmetric_effect = 0.5 * (
                contrasts["positive"] - contrasts["negative"]
            )
            per_pair.append(
                {
                    "index": prepared_pair["index"],
                    "label": prepared_pair["label"],
                    "negative": sequence_payloads["negative"],
                    "positive": sequence_payloads["positive"],
                    "contrasts_nats_per_token": contrasts,
                    "contrast_points": [
                        {
                            "key": key,
                            "alpha": alpha,
                            "positive_minus_negative_mean_logprob": (
                                contrasts[key]
                            ),
                        }
                        for key, alpha in zip(
                            alpha_keys,
                            alpha_values,
                            strict=True,
                        )
                    ],
                    "symmetric_effect_nats_per_token": symmetric_effect,
                    "symmetric_effect_bits_per_token": (
                        symmetric_effect / math.log(2.0)
                    ),
                }
            )

        assert effective_shift_norm is not None
        assert effective_shift_vector is not None
        assert baseline_prefix_residual_norm is not None
        assert shift_dtype_name is not None
        assert next_token is not None
        pair_effects = [
            float(item["symmetric_effect_nats_per_token"]) for item in per_pair
        ]
        effect_nats = float(np.mean(pair_effects))
        return {
            "estimand": (
                "mean_i(0.5*(C_i(+alpha)-C_i(-alpha))); "
                "C_i(alpha)=positive_mean_logprob-"
                "negative_mean_logprob"
            ),
            "effect_nats_per_token": effect_nats,
            "effect_bits_per_token": effect_nats / math.log(2.0),
            "per_pair": per_pair,
            "next_token": next_token,
            "effective_shift_norm": effective_shift_norm,
            "_effective_shift_vector": effective_shift_vector,
            "baseline_prefix_residual_norm": (baseline_prefix_residual_norm),
            "prefix_relative_shift_norm": (
                effective_shift_norm / baseline_prefix_residual_norm
                if baseline_prefix_residual_norm > 0.0
                else None
            ),
            "shift_dtype_name": shift_dtype_name,
            "scoring_ms": (time.perf_counter() - started) * 1_000,
        }

    @staticmethod
    def _steering_direction_sha256(direction: np.ndarray) -> str:
        """Hash one direction in a stable little-endian float64 encoding."""
        canonical = np.ascontiguousarray(np.asarray(direction, dtype="<f8"))
        digest = hashlib.sha256()
        digest.update(b"apertus-steering-direction-v1\0")
        digest.update(str(tuple(canonical.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical.tobytes(order="C"))
        return digest.hexdigest()

    def _cast_steering_shift(
        self,
        *,
        unit_direction: np.ndarray,
        native_scale: float,
        alpha_magnitude: float,
        dtype_name: str,
    ) -> np.ndarray:
        """Return the actually representable positive shift as float64."""
        shift = (
            np.float32(alpha_magnitude)
            * float(native_scale)
            * np.asarray(unit_direction, dtype=np.float32)
        )
        short_dtype_name = dtype_name.rsplit(".", 1)[-1]
        if self.mock:
            if short_dtype_name != "float32":
                raise RuntimeError(
                    f"Mock očekává float32 shift, nikoli {dtype_name}."
                )
            return shift.astype(np.float32).astype(np.float64)

        import mlx.core as mx

        mlx_dtype = getattr(mx, short_dtype_name, None)
        if mlx_dtype is None:
            raise RuntimeError(
                f"Neznámý MLX dtype steeringového posunu: {dtype_name}."
            )
        typed = mx.array(shift.astype(np.float32)).astype(mlx_dtype)
        evaluated = typed.astype(mx.float32)
        mx.eval(evaluated)
        return np.asarray(evaluated).astype(np.float64, copy=True)

    def _post_cast_norm_match(
        self,
        *,
        target_effective_shift: np.ndarray,
        sham_direction: np.ndarray,
        native_scale: float,
        alpha_magnitude: float,
        dtype_name: str,
        relative_tolerance: float,
    ) -> tuple[float, np.ndarray, float]:
        """Rescale a sham until its post-cast norm matches the target."""
        target_norm = float(np.linalg.norm(target_effective_shift))
        if not math.isfinite(target_norm) or target_norm <= 0.0:
            raise RuntimeError(
                "Target effective shift nemá platnou nenulovou normu."
            )
        correction_factor = 1.0
        effective_shift: np.ndarray | None = None
        relative_error = math.inf
        for _attempt in range(12):
            effective_shift = self._cast_steering_shift(
                unit_direction=sham_direction,
                native_scale=native_scale * correction_factor,
                alpha_magnitude=alpha_magnitude,
                dtype_name=dtype_name,
            )
            effective_norm = float(np.linalg.norm(effective_shift))
            if not math.isfinite(effective_norm) or effective_norm <= 0.0:
                raise RuntimeError(
                    "Sham effective shift po castu nemá platnou normu."
                )
            relative_error = abs(effective_norm - target_norm) / target_norm
            if relative_error <= relative_tolerance:
                return correction_factor, effective_shift, relative_error
            correction_factor *= target_norm / effective_norm
        assert effective_shift is not None
        raise RuntimeError(
            "Sham zásah se nepodařilo post-cast norm-matchnout: "
            f"relativní chyba {relative_error:.3g} > "
            f"{relative_tolerance:.3g}."
        )

    def _steering_control_provenance(
        self,
        *,
        test_text: str,
        test_token_ids: list[int],
        prepared_pairs: list[dict[str, Any]],
        prepared_continuations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.mock:
            tokenizer_payload: dict[str, Any] = {
                "kind": "deterministic_mock_tokenizer",
                "bos_token_id": 1,
            }
        else:
            tokenizer = self.tokenizer
            tokenizer_payload = {
                "kind": (
                    f"{type(tokenizer).__module__}."
                    f"{type(tokenizer).__qualname__}"
                ),
                "bos_token_id": getattr(tokenizer, "bos_token_id", None),
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                "vocab_size": getattr(tokenizer, "vocab_size", None),
            }
        source_control = RUNTIME_PROVENANCE.get("source_control", {})
        reproducibility_limitations = [
            (
                "Model fingerprint covers metadata and the weight "
                "manifest, not a byte hash of every weight shard."
            )
        ]
        if not source_control.get("available"):
            reproducibility_limitations.append(
                "Source-control revision was unavailable at server start."
            )
        elif source_control.get("dirty"):
            reproducibility_limitations.append(
                "The source-control worktree was dirty at server start; "
                "the commit SHA alone does not reconstruct local edits."
            )
        return {
            "schema": "steering-controls-provenance-v1",
            "model": self._model_provenance_unlocked(),
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
            "runtime": dict(RUNTIME_PROVENANCE),
            "tokenizer": tokenizer_payload,
            "encoding_policy": (
                "encode_with_bos; exact prefix boundary required for "
                "continuations"
            ),
            "test": {
                "text": test_text,
                "token_ids": list(test_token_ids),
            },
            "contrast_pairs": [
                {
                    "negative_text": pair["negative_text"],
                    "positive_text": pair["positive_text"],
                    "negative_token_ids": list(pair["negative_token_ids"]),
                    "positive_token_ids": list(pair["positive_token_ids"]),
                    "boundary_token_id": pair["boundary_token_id"],
                }
                for pair in prepared_pairs
            ],
            "continuation_pairs": [
                {
                    "index": pair["index"],
                    "label": pair["label"],
                    "negative_text": pair["negative"]["text"],
                    "positive_text": pair["positive"]["text"],
                    "negative_token_ids": list(pair["negative"]["token_ids"]),
                    "positive_token_ids": list(pair["positive"]["token_ids"]),
                }
                for pair in prepared_continuations
            ],
            "direction_hash": {
                "algorithm": "sha256",
                "canonical_encoding": (
                    "apertus-steering-direction-v1 + shape + "
                    "little_endian_float64_bytes"
                ),
            },
            "reproducibility_scope": "diagnostic_not_bit_exact",
            "reproducibility_limitations": reproducibility_limitations,
        }

    @staticmethod
    def _orthogonal_gaussian_direction(
        target_direction: np.ndarray,
        seed: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Draw one isotropic Gaussian direction and remove target projection."""
        target = np.asarray(target_direction, dtype=np.float64)
        target_norm = float(np.linalg.norm(target))
        if not math.isfinite(target_norm) or target_norm <= 0.0:
            raise RuntimeError("Cílový steeringový směr nemá platnou normu.")
        target = target / target_norm
        rng = np.random.Generator(np.random.PCG64(seed))
        for _attempt in range(8):
            raw = rng.normal(0.0, 1.0, size=target.shape).astype(
                np.float64,
                copy=False,
            )
            projection = float(np.dot(raw, target))
            projected = raw - projection * target
            projected_norm = float(np.linalg.norm(projected))
            if math.isfinite(projected_norm) and projected_norm > 1e-12:
                unit = projected / projected_norm
                return unit, {
                    "raw_norm": float(np.linalg.norm(raw)),
                    "removed_target_projection": projection,
                    "projected_norm_before_normalization": projected_norm,
                    "unit_norm": float(np.linalg.norm(unit)),
                    "cosine_to_target": float(np.dot(unit, target)),
                }
        raise RuntimeError(
            "Nepodařilo se vytvořit náhodný směr kolmý k targetu."
        )

    @staticmethod
    def _sham_distribution_summary(
        target_effect: float,
        sham_effects: list[float],
    ) -> dict[str, Any]:
        values = np.asarray(sham_effects, dtype=np.float64)
        if values.ndim != 1 or values.size < MIN_STEERING_SHAM_SAMPLES:
            raise ValueError(
                "Sham distribuce musí obsahovat alespoň dva body."
            )
        if not math.isfinite(target_effect) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Target i všechny sham efekty musí být konečná čísla."
            )
        mean = float(np.mean(values))
        sample_sd = float(np.std(values, ddof=1))
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        exceedance_count = int(np.count_nonzero(values >= target_effect))
        less_count = int(np.count_nonzero(values < target_effect))
        equal_count = int(np.count_nonzero(values == target_effect))
        z_score = (
            float((target_effect - mean) / sample_sd)
            if sample_sd > float(np.finfo(np.float64).eps)
            else None
        )
        return {
            "sham_effect_nats_per_token": {
                "mean": mean,
                "sample_sd": sample_sd,
                "median": median,
                "mad": mad,
                "mad_definition": "unscaled_median_absolute_deviation",
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            },
            "target_z_score": z_score,
            "z_definition": "(target-sham_mean)/sham_sample_sd",
            "target_exceedance_count": exceedance_count,
            "monte_carlo_reference_tail_fraction": (
                (1 + exceedance_count) / (values.size + 1)
            ),
            "monte_carlo_reference_tail_fraction_definition": (
                "(1 + count(sham_effect >= target_effect)) / "
                "(sample_count + 1); descriptive orthogonal-reference "
                "tail, not an exact randomization p"
            ),
            "target_percentile_midrank_fraction": (
                (less_count + 0.5 * equal_count) / values.size
            ),
            "target_percentile_midrank_fraction_definition": (
                "(count(sham_effect < target_effect) + "
                "0.5*count(sham_effect == target_effect)) / sample_count"
            ),
        }

    def _steering_sham_control_local(
        self,
        contrast_pairs: list[dict[str, Any]],
        test_text: str,
        continuation_pairs: list[dict[str, Any]],
        layer: int,
        alpha_magnitude: float,
        temperature: float,
        sample_count: int,
        seed: int,
    ) -> dict[str, Any]:
        """Compare the target direction with isotropic norm-matched shams."""
        started = time.perf_counter()
        self._validate_steering_layer(layer)
        if (
            not math.isfinite(alpha_magnitude)
            or not 0.0 < alpha_magnitude <= 2.0
        ):
            raise ValueError(
                "alpha_magnitude musí být větší než 0 a nejvýše 2."
            )
        if not math.isfinite(temperature) or not 0.2 <= temperature <= 2.0:
            raise ValueError("Teplota musí být mezi 0,2 a 2,0.")
        if type(sample_count) is not int or not (
            MIN_STEERING_SHAM_SAMPLES
            <= sample_count
            <= MAX_STEERING_SHAM_SAMPLES
        ):
            raise ValueError(
                "sample_count musí být celé číslo mezi "
                f"{MIN_STEERING_SHAM_SAMPLES} a "
                f"{MAX_STEERING_SHAM_SAMPLES}."
            )
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("seed musí být uint32.")

        (
            encoded_pairs,
            prepared_pairs,
            test_text,
            test_token_ids,
            total_tokens,
        ) = self._prepare_steering_inputs(contrast_pairs, test_text)
        prepared_continuations, continuation_token_total = (
            self._prepare_phrase_score_pairs(
                test_text=test_text,
                test_token_ids=test_token_ids,
                continuation_pairs=continuation_pairs,
            )
        )
        total_tokens += continuation_token_total
        if total_tokens > MAX_STEERING_TOTAL_TOKENS:
            raise ValueError(
                f"Steeringové texty mají dohromady {total_tokens} tokenů; "
                f"souhrnný limit je {MAX_STEERING_TOTAL_TOKENS}."
            )

        direction_started = time.perf_counter()
        if self.mock:
            target_direction, direction_diagnostics = (
                self._mock_steering_direction(prepared_pairs, layer)
            )
            contrast_forward_ms = 0.0
        else:
            (
                target_direction,
                direction_diagnostics,
                contrast_forward_ms,
            ) = self._mlx_steering_direction(encoded_pairs, layer)
        direction_ms = (time.perf_counter() - direction_started) * 1_000
        native_scale = float(direction_diagnostics["native_scale"])

        target_factory = self._teacher_forced_stepper_factory_for_direction(
            unit_direction=target_direction,
            native_scale=native_scale,
            layer=layer,
            alpha_magnitude=alpha_magnitude,
            test_text=test_text,
        )
        target_score = self._score_matched_continuation_effect(
            test_text=test_text,
            test_token_ids=test_token_ids,
            prepared_continuations=prepared_continuations,
            alpha_magnitude=alpha_magnitude,
            temperature=temperature,
            stepper_factory=target_factory,
        )
        target_effective_shift = np.asarray(
            target_score.pop("_effective_shift_vector"),
            dtype=np.float64,
        )
        target_effective_norm = float(
            np.linalg.norm(target_effective_shift)
        )
        runtime_target_effective_norm = float(
            target_score["effective_shift_norm"]
        )
        if not math.isclose(
            target_effective_norm,
            runtime_target_effective_norm,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Samostatná post-cast kontrola target normy nesouhlasí "
                "s normou změřenou ve forwardu "
                f"({target_effective_norm:.9g} vs. "
                f"{runtime_target_effective_norm:.9g}; "
                f"dtype {target_score['shift_dtype_name']})."
            )
        target = {
            "kind": "target_contrastive_direction",
            **target_score,
            "native_scale": native_scale,
            "nominal_native_scale": native_scale,
            "corrected_native_scale": native_scale,
            "applied_shift_norm": alpha_magnitude * native_scale,
            "direction": {
                "sha256": self._steering_direction_sha256(target_direction),
                "unit_norm": float(np.linalg.norm(target_direction)),
                "aggregation": direction_diagnostics["aggregation"],
                "resultant_length": float(
                    direction_diagnostics["resultant_length"]
                ),
                "cancellation_fraction": float(
                    direction_diagnostics["cancellation_fraction"]
                ),
                "pair_norms": [
                    float(value)
                    for value in direction_diagnostics["pair_norms"]
                ],
                "pair_cosines_to_direction": [
                    float(value)
                    for value in direction_diagnostics[
                        "pair_cosines_to_direction"
                    ]
                ],
                "effective_cosine_to_target_after_cast": 1.0,
            },
        }

        seed_sequence = np.random.SeedSequence(seed)
        child_sequences = seed_sequence.spawn(sample_count)
        shams: list[dict[str, Any]] = []
        effective_norm_match_tolerance = (
            STEERING_EFFECTIVE_NORM_RELATIVE_TOLERANCE
        )
        for sample_index, child_sequence in enumerate(child_sequences):
            child_seed = int(
                child_sequence.generate_state(1, dtype=np.uint32)[0]
            )
            sham_direction, draw = self._orthogonal_gaussian_direction(
                target_direction,
                child_seed,
            )
            (
                correction_factor,
                expected_sham_effective_shift,
                expected_norm_relative_error,
            ) = self._post_cast_norm_match(
                target_effective_shift=target_effective_shift,
                sham_direction=sham_direction,
                native_scale=native_scale,
                alpha_magnitude=alpha_magnitude,
                dtype_name=target_score["shift_dtype_name"],
                relative_tolerance=effective_norm_match_tolerance,
            )
            corrected_native_scale = native_scale * correction_factor
            sham_factory = self._teacher_forced_stepper_factory_for_direction(
                unit_direction=sham_direction,
                native_scale=corrected_native_scale,
                layer=layer,
                alpha_magnitude=alpha_magnitude,
                test_text=test_text,
            )
            sham_score = self._score_matched_continuation_effect(
                test_text=test_text,
                test_token_ids=test_token_ids,
                prepared_continuations=prepared_continuations,
                alpha_magnitude=alpha_magnitude,
                temperature=temperature,
                stepper_factory=sham_factory,
            )
            runtime_sham_effective_shift = np.asarray(
                sham_score.pop("_effective_shift_vector"),
                dtype=np.float64,
            )
            sham_effective_norm = float(sham_score["effective_shift_norm"])
            effective_norm_relative_error = (
                abs(sham_effective_norm - target_effective_norm)
                / target_effective_norm
                if target_effective_norm > 0.0
                else math.inf
            )
            if (
                not math.isfinite(effective_norm_relative_error)
                or effective_norm_relative_error
                > effective_norm_match_tolerance
            ):
                raise RuntimeError(
                    "Sham zásah po castu nesplnil effective norm-match: "
                    f"relativní chyba {effective_norm_relative_error:.3g} "
                    f"> {effective_norm_match_tolerance:.3g}."
                )
            expected_forward_relative_error = (
                abs(
                    sham_effective_norm
                    - float(np.linalg.norm(expected_sham_effective_shift))
                )
                / target_effective_norm
            )
            effective_cosine = float(
                np.dot(
                    runtime_sham_effective_shift,
                    target_effective_shift,
                )
                / (
                    np.linalg.norm(runtime_sham_effective_shift)
                    * target_effective_norm
                )
            )
            shams.append(
                {
                    "kind": "isotropic_norm_matched_sham",
                    "sample_index": sample_index,
                    "seed": child_seed,
                    **sham_score,
                    "native_scale": native_scale,
                    "nominal_native_scale": native_scale,
                    "corrected_native_scale": corrected_native_scale,
                    "nominal_applied_shift_norm": (
                        alpha_magnitude * native_scale
                    ),
                    "pre_cast_correction_factor": correction_factor,
                    "applied_shift_norm": (
                        alpha_magnitude * corrected_native_scale
                    ),
                    "direction": {
                        "sha256": self._steering_direction_sha256(
                            sham_direction
                        ),
                        **draw,
                        "effective_cosine_to_target_after_cast": (
                            effective_cosine
                        ),
                    },
                    "effective_norm_difference_from_target": float(
                        sham_effective_norm - target_effective_norm
                    ),
                    "effective_norm_relative_error": (
                        effective_norm_relative_error
                    ),
                    "effective_norm_match_tolerance": (
                        effective_norm_match_tolerance
                    ),
                    "effective_norm_match_passed": True,
                    "pre_forward_expected_norm_relative_error": (
                        expected_norm_relative_error
                    ),
                    "precomputed_vs_forward_norm_relative_error": (
                        expected_forward_relative_error
                    ),
                }
            )

        summary = self._sham_distribution_summary(
            float(target["effect_nats_per_token"]),
            [float(sham["effect_nats_per_token"]) for sham in shams],
        )
        summary["sample_count"] = sample_count
        summary["target_effect_nats_per_token"] = float(
            target["effect_nats_per_token"]
        )
        summary["target_effect_bits_per_token"] = float(
            target["effect_bits_per_token"]
        )
        summary["inference"] = {
            "z_score_is_descriptive": True,
            "tail_fraction_is_one_sided": True,
            "exchangeable_null": False,
            "exact_randomization_p": False,
            "minimum_attainable_empirical_p": 1.0 / (sample_count + 1),
            "minimum_attainable_empirical_p_formula": ("1/(sample_count+1)"),
        }
        warnings = self._steering_teacher_forced_warnings(
            direction_diagnostics,
            alpha_magnitude,
        )
        warnings.extend(
            [
                {
                    "code": "isotropic_sham_not_semantic_null",
                    "message": (
                        "Izotropní norm-matched sham testuje specificitu "
                        "vůči náhodnému směru stejné velikosti; není to "
                        "sémantická nulová kontrola."
                    ),
                },
                {
                    "code": "finite_randomization_resolution",
                    "message": (
                        "Nejmenší dosažitelný jednostranný Monte Carlo "
                        "podíl v ocasu je "
                        f"1/{sample_count + 1}."
                    ),
                },
                {
                    "code": "descriptive_z_score",
                    "message": (
                        "Z-skóre je popisná poloha targetu vůči konečnému "
                        "sham vzorku."
                    ),
                },
                {
                    "code": "not_exchangeable_randomization_test",
                    "message": (
                        "Target je fixní a shamy jsou losované podmíněně "
                        "v jeho ortogonálním doplňku, takže body nejsou "
                        "exchangeable. Korigovaný podíl v ocasu popisuje "
                        "zvolenou referenci; není to exaktní randomizační "
                        "p ani kontrola chyby I. druhu."
                    ),
                },
            ]
        )
        provenance = self._steering_control_provenance(
            test_text=test_text,
            test_token_ids=test_token_ids,
            prepared_pairs=prepared_pairs,
            prepared_continuations=prepared_continuations,
        )
        total_ms = (time.perf_counter() - started) * 1_000
        return {
            "analysis_kind": "steering_sham_control",
            "test_text": test_text,
            "test_token_ids": test_token_ids,
            "temperature": temperature,
            "probability_kind": (
                "raw_model"
                if math.isclose(temperature, 1.0)
                else "temperature_adjusted"
            ),
            "layer": layer,
            "alpha_magnitude": alpha_magnitude,
            "pair_count": len(prepared_continuations),
            "target": target,
            "shams": shams,
            "summary": summary,
            "randomization": {
                "kind": "isotropic_norm_matched_sham",
                "raw_draw_distribution": (
                    "standard_gaussian_in_residual_space"
                ),
                "direction_distribution": (
                    "uniform_unit_sphere_in_target_orthogonal_complement_"
                    "via_projected_gaussian"
                ),
                "bit_generator": "explicit_numpy_PCG64",
                "seed_derivation": "numpy_SeedSequence_spawn",
                "orthogonalization": (
                    "project_target_direction_out_then_l2_normalize_"
                    "in_float64; post_cast_cosine_reported_separately"
                ),
                "norm_matching": (
                    "same_nominal_native_scale_and_alpha_then_per_sham_"
                    "post_cast_rescaling_to_target_effective_norm"
                ),
                "effective_norm_relative_tolerance": (
                    effective_norm_match_tolerance
                ),
                "all_effective_norm_matches_passed": True,
                "root_seed": seed,
                "child_seeds": [int(sham["seed"]) for sham in shams],
                "sample_count": sample_count,
                "residual_dimension": int(target_direction.size),
                "exchangeable_null": False,
                "exact_randomization_test": False,
            },
            "steering": {
                "kind": "contrastive_activation_addition",
                "hook": "post_decoder_block_residual_last_position",
                "layer": layer,
                "layer_numbering": "one_based",
                "direction_policy": "target_plus_orthogonal_isotropic_shams",
                "alpha_magnitude": alpha_magnitude,
                "teacher_forced": True,
                "shared_context_at_every_step": True,
                "pair_count": len(prepared_pairs),
                "held_out_continuation_pair_count": len(
                    prepared_continuations
                ),
                "total_input_tokens": total_tokens,
                "execution": "single_serial_worker_job",
                "cache_policy": (
                    "request_local_per_sequence_three_alpha_batched_kv_"
                    "cache_no_global_mutation"
                ),
            },
            "warnings": warnings,
            "provenance": provenance,
            "cached": False,
            "model_id": self.active_model_id,
            "model_label": self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
            "timing_ms": {
                "contrast_forward": contrast_forward_ms,
                "direction_total": direction_ms,
                "target_scoring": target_score["scoring_ms"],
                "sham_scoring": float(
                    sum(float(sham["scoring_ms"]) for sham in shams)
                ),
                "total": total_ms,
            },
            "latency_ms": total_ms,
        }

    def _steering_layer_sweep_local(
        self,
        contrast_pairs: list[dict[str, Any]],
        test_text: str,
        continuation_pairs: list[dict[str, Any]],
        layers: list[int],
        reference_layer: int,
        alpha_magnitude: float,
        temperature: float,
    ) -> dict[str, Any]:
        """Re-derive and score the target contrastive direction per layer."""
        started = time.perf_counter()
        if not isinstance(layers, list) or not (
            MIN_STEERING_SWEEP_LAYERS
            <= len(layers)
            <= MAX_STEERING_SWEEP_LAYERS
        ):
            raise ValueError(
                "layers musí obsahovat "
                f"{MIN_STEERING_SWEEP_LAYERS} až "
                f"{MAX_STEERING_SWEEP_LAYERS} vrstev."
            )
        if any(
            type(layer) is not int or not 1 <= layer <= self.steering_layer_count() for layer in layers
        ):
            raise ValueError(
                f"Každá položka layers musí být celé číslo mezi 1 a {self.steering_layer_count()}."
            )
        if len(set(layers)) != len(layers):
            raise ValueError("layers nesmí obsahovat duplicity.")
        if type(reference_layer) is not int or reference_layer not in layers:
            raise ValueError("reference_layer musí být jednou z layers.")
        if (
            not math.isfinite(alpha_magnitude)
            or not 0.0 < alpha_magnitude <= 2.0
        ):
            raise ValueError(
                "alpha_magnitude musí být větší než 0 a nejvýše 2."
            )
        if not math.isfinite(temperature) or not 0.2 <= temperature <= 2.0:
            raise ValueError("Teplota musí být mezi 0,2 a 2,0.")

        (
            encoded_pairs,
            prepared_pairs,
            test_text,
            test_token_ids,
            total_tokens,
        ) = self._prepare_steering_inputs(contrast_pairs, test_text)
        prepared_continuations, continuation_token_total = (
            self._prepare_phrase_score_pairs(
                test_text=test_text,
                test_token_ids=test_token_ids,
                continuation_pairs=continuation_pairs,
            )
        )
        total_tokens += continuation_token_total
        if total_tokens > MAX_STEERING_TOTAL_TOKENS:
            raise ValueError(
                f"Steeringové texty mají dohromady {total_tokens} tokenů; "
                f"souhrnný limit je {MAX_STEERING_TOTAL_TOKENS}."
            )

        derived: list[dict[str, Any]] = []
        contrast_forward_total = 0.0
        for layer in layers:
            direction_started = time.perf_counter()
            if self.mock:
                direction, diagnostics = self._mock_steering_direction(
                    prepared_pairs,
                    layer,
                )
                contrast_forward_ms = 0.0
            else:
                (
                    direction,
                    diagnostics,
                    contrast_forward_ms,
                ) = self._mlx_steering_direction(
                    encoded_pairs,
                    layer,
                )
            contrast_forward_total += contrast_forward_ms
            derived.append(
                {
                    "layer": layer,
                    "direction": direction,
                    "diagnostics": diagnostics,
                    "contrast_forward_ms": contrast_forward_ms,
                    "direction_total_ms": (
                        time.perf_counter() - direction_started
                    )
                    * 1_000,
                }
            )

        reference_direction = next(
            item["direction"]
            for item in derived
            if item["layer"] == reference_layer
        )
        points: list[dict[str, Any]] = []
        for item in derived:
            layer_started = time.perf_counter()
            layer = int(item["layer"])
            direction = item["direction"]
            diagnostics = item["diagnostics"]
            native_scale = float(diagnostics["native_scale"])
            stepper_factory = (
                self._teacher_forced_stepper_factory_for_direction(
                    unit_direction=direction,
                    native_scale=native_scale,
                    layer=layer,
                    alpha_magnitude=alpha_magnitude,
                    test_text=test_text,
                )
            )
            score = self._score_matched_continuation_effect(
                test_text=test_text,
                test_token_ids=test_token_ids,
                prepared_continuations=prepared_continuations,
                alpha_magnitude=alpha_magnitude,
                temperature=temperature,
                stepper_factory=stepper_factory,
            )
            effective_shift_norm = float(score["effective_shift_norm"])
            baseline_prefix_residual_norm = float(
                score["baseline_prefix_residual_norm"]
            )
            points.append(
                {
                    "layer": layer,
                    "remaining_decoder_blocks": 32 - layer,
                    "effect_nats_per_token": float(
                        score["effect_nats_per_token"]
                    ),
                    "effect_bits_per_token": float(
                        score["effect_bits_per_token"]
                    ),
                    "estimand": score["estimand"],
                    "per_pair": score["per_pair"],
                    "next_token": score["next_token"],
                    "native_scale": native_scale,
                    "applied_shift_norm": alpha_magnitude * native_scale,
                    "effective_shift_norm": effective_shift_norm,
                    "baseline_prefix_residual_norm": (
                        baseline_prefix_residual_norm
                    ),
                    "prefix_relative_shift_norm": (
                        effective_shift_norm / baseline_prefix_residual_norm
                        if baseline_prefix_residual_norm > 0.0
                        else None
                    ),
                    "direction_sha256": (
                        self._steering_direction_sha256(direction)
                    ),
                    "direction_coherence": {
                        "resultant_length": float(
                            diagnostics["resultant_length"]
                        ),
                        "cancellation_fraction": float(
                            diagnostics["cancellation_fraction"]
                        ),
                        "pair_norms": [
                            float(value)
                            for value in diagnostics["pair_norms"]
                        ],
                        "pair_cosines_to_direction": [
                            float(value)
                            for value in diagnostics[
                                "pair_cosines_to_direction"
                            ]
                        ],
                    },
                    "cosine_to_reference_direction": float(
                        np.dot(direction, reference_direction)
                    ),
                    "timing_ms": {
                        "contrast_forward": float(item["contrast_forward_ms"]),
                        "direction_total": float(item["direction_total_ms"]),
                        "scoring": float(score["scoring_ms"]),
                        "total": (time.perf_counter() - layer_started) * 1_000
                        + float(item["direction_total_ms"]),
                    },
                }
            )

        max_point = max(
            points,
            key=lambda point: float(point["effect_nats_per_token"]),
        )
        min_point = min(
            points,
            key=lambda point: float(point["effect_nats_per_token"]),
        )
        total_ms = (time.perf_counter() - started) * 1_000
        warnings = [
            {
                "code": "locally_rederived_directions",
                "message": (
                    "Směr je odvozen samostatně v každé vrstvě; sweep "
                    "měří lokální steerovatelnost, nikoli přenos směru "
                    "z referenční vrstvy."
                ),
            },
            {
                "code": "unequal_remaining_depth",
                "message": (
                    "Pozdější zásahy mají méně zbývajících decoderových "
                    "bloků; rozdíl efektu proto není čistou vlastností "
                    "reprezentace v dané vrstvě."
                ),
            },
            {
                "code": "convergent_not_independent",
                "message": (
                    "Shoda s dřívějším lens nálezem je konvergentní "
                    "evidence, nikoli nezávislá replikace."
                ),
            },
            {
                "code": "descriptive_profile_no_peak_test",
                "message": (
                    "Vrstva s největším pozorovaným efektem je popisné "
                    "maximum testovaných bodů; sweep neobsahuje statistický "
                    "test existence ani polohy vrcholu."
                ),
            },
        ]
        if self.is_quantized():
            warnings.append(
                {
                    "code": "quantized_model",
                    "message": (
                        "Výsledek pochází z kvantovaného nebo hybridního "
                        "modelu; těsné rozdíly vrstev mohou být numericky "
                        "citlivé."
                    ),
                }
            )
        provenance = self._steering_control_provenance(
            test_text=test_text,
            test_token_ids=test_token_ids,
            prepared_pairs=prepared_pairs,
            prepared_continuations=prepared_continuations,
        )
        return {
            "analysis_kind": "steering_layer_sweep",
            "test_text": test_text,
            "test_token_ids": test_token_ids,
            "temperature": temperature,
            "probability_kind": (
                "raw_model"
                if math.isclose(temperature, 1.0)
                else "temperature_adjusted"
            ),
            "alpha_magnitude": alpha_magnitude,
            "layers": list(layers),
            "reference_layer": reference_layer,
            "direction_policy": "locally_rederived_per_layer",
            "pair_count": len(prepared_continuations),
            "points": points,
            "summary": {
                "estimand": points[0]["estimand"],
                "inference_policy": (
                    "descriptive_profile_no_peak_significance_claim"
                ),
                "peak_claim": False,
                "observed_max_effect_layer": int(max_point["layer"]),
                "observed_max_effect_nats_per_token": float(
                    max_point["effect_nats_per_token"]
                ),
                "observed_max_effect_bits_per_token": float(
                    max_point["effect_bits_per_token"]
                ),
                "observed_min_effect_layer": int(min_point["layer"]),
                "observed_min_effect_nats_per_token": float(
                    min_point["effect_nats_per_token"]
                ),
                "effect_profile": [
                    {
                        "layer": int(point["layer"]),
                        "effect_nats_per_token": float(
                            point["effect_nats_per_token"]
                        ),
                        "effect_bits_per_token": float(
                            point["effect_bits_per_token"]
                        ),
                    }
                    for point in points
                ],
            },
            "steering": {
                "kind": "contrastive_activation_addition",
                "hook": "post_decoder_block_residual_last_position",
                "layer_numbering": "one_based",
                "direction_policy": "locally_rederived_per_layer",
                "direction_computations": len(layers),
                "alpha_magnitude": alpha_magnitude,
                "teacher_forced": True,
                "shared_context_at_every_step": True,
                "contrast_pair_count": len(prepared_pairs),
                "held_out_continuation_pair_count": len(
                    prepared_continuations
                ),
                "total_input_tokens": total_tokens,
                "execution": "single_serial_worker_job",
                "cache_policy": (
                    "request_local_per_layer_per_sequence_three_alpha_"
                    "batched_kv_cache_no_global_mutation"
                ),
            },
            "warnings": warnings,
            "provenance": provenance,
            "cached": False,
            "model_id": self.active_model_id,
            "model_label": self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
            "timing_ms": {
                "contrast_forward": contrast_forward_total,
                "direction_total": float(
                    sum(item["direction_total_ms"] for item in derived)
                ),
                "scoring": float(
                    sum(point["timing_ms"]["scoring"] for point in points)
                ),
                "total": total_ms,
            },
            "latency_ms": total_ms,
        }

    def _tokenization_service(self) -> TokenizationService:
        return TokenizationService(
            tokenizer=self.tokenizer,
            mock=self.mock,
            mock_words=tuple(MOCK_WORDS),
        )

    def _text_surprisal_tokenization(
        self,
        text: str,
    ) -> tuple[
        list[int],
        list[int],
        list[str],
        list[str | None],
        list[tuple[int, int] | None],
    ]:
        """Return ordinary targets and BOS-aligned contexts for fixed text."""
        return self._tokenization_service().text_tokenization(text)

        if self.mock:
            pieces = re.findall(r"\s*\S+", text)
            token_ids = [
                int.from_bytes(
                    hashlib.sha256(piece.encode("utf-8")).digest()[:4],
                    "big",
                )
                % len(MOCK_WORDS)
                for piece in pieces
            ]
            offsets: list[tuple[int, int] | None] = []
            cursor = 0
            for piece in pieces:
                end = cursor + len(piece)
                offsets.append((cursor, end))
                cursor = end
            return (
                token_ids,
                [0, *token_ids[:-1]],
                pieces,
                [None for _ in token_ids],
                offsets,
            )

        try:
            token_ids = list(
                self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            )
        except TypeError:
            token_ids = encode_with_bos(self.tokenizer, text)
            bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_token_id is not None and token_ids[:1] == [bos_token_id]:
                token_ids = token_ids[1:]
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None and token_ids[-1:] == [eos_token_id]:
                token_ids = token_ids[:-1]

        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if type(bos_token_id) is not int or bos_token_id < 0:
            raise RuntimeError(
                "Tokenizer nemá platný BOS token; první textový token "
                "nelze skórovat z definovaného kontextu."
            )

        raw_tokens: list[str | None] = []
        for token_id in token_ids:
            try:
                raw_token = self.tokenizer.convert_ids_to_tokens(token_id)
            except Exception:
                raw_token = None
            raw_tokens.append(
                str(raw_token) if raw_token is not None else None
            )

        pieces: list[str] | None = None
        offsets: list[tuple[int, int] | None] | None = None
        tokenizer_call = getattr(self.tokenizer, "__call__", None)
        if callable(tokenizer_call):
            try:
                encoded = tokenizer_call(
                    text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                offset_ids = list(encoded["input_ids"])
                offset_values = list(encoded["offset_mapping"])
                candidate_offsets = [
                    (int(start), int(end))
                    for start, end in offset_values
                ]
                candidate_pieces = [
                    text[start:end] for start, end in candidate_offsets
                ]
                contiguous = (
                    bool(candidate_offsets)
                    and candidate_offsets[0][0] == 0
                    and candidate_offsets[-1][1] == len(text)
                    and all(
                        left[1] == right[0]
                        for left, right in zip(
                            candidate_offsets,
                            candidate_offsets[1:],
                        )
                    )
                )
                if (
                    offset_ids == token_ids
                    and len(candidate_offsets) == len(token_ids)
                    and contiguous
                    and "".join(candidate_pieces) == text
                ):
                    pieces = candidate_pieces
                    offsets = list(candidate_offsets)
            except (
                KeyError,
                TypeError,
                ValueError,
                AttributeError,
            ):
                pass

        if pieces is None or offsets is None:
            decoded_pieces: list[str] = []
            running_ids: list[int] = []
            decoded_prefix = ""
            for token_id in token_ids:
                running_ids.append(token_id)
                decoded = safe_decode(
                    self.tokenizer,
                    running_ids,
                    skip_special=True,
                )
                if decoded.startswith(decoded_prefix):
                    piece = decoded[len(decoded_prefix) :]
                else:
                    piece = safe_decode(
                        self.tokenizer,
                        [token_id],
                        skip_special=False,
                    )
                decoded_pieces.append(piece)
                decoded_prefix = decoded
            if "".join(decoded_pieces) == text:
                pieces = decoded_pieces
                offsets = []
                cursor = 0
                for piece in pieces:
                    end = cursor + len(piece)
                    offsets.append((cursor, end))
                    cursor = end
            else:
                pieces = [
                    safe_decode(
                        self.tokenizer,
                        [token_id],
                        skip_special=False,
                    )
                    for token_id in token_ids
                ]
                offsets = [None for _ in token_ids]

        return (
            token_ids,
            [bos_token_id, *token_ids[:-1]],
            pieces,
            raw_tokens,
            offsets,
        )

    @staticmethod
    def _unicode_text_payload(text: str) -> dict[str, Any]:
        return TokenizationService.unicode_text_payload(text)

        normalized = {
            form: unicodedata.normalize(form, text)
            for form in ("NFC", "NFD", "NFKC", "NFKD")
        }
        return {
            "code_point_count": len(text),
            "utf8_byte_count": len(text.encode("utf-8")),
            "utf8_hex": " ".join(
                f"{value:02X}" for value in text.encode("utf-8")
            ),
            "normalization": {
                form: {
                    "is_normalized": value == text,
                    "changed": value != text,
                }
                for form, value in normalized.items()
            },
        }

    def _tokenizer_identity_unlocked(self) -> dict[str, Any]:
        if self.tokenizer_identity_cache is not None:
            return dict(self.tokenizer_identity_cache)

        payload = build_tokenizer_identity(
            tokenizer=self.tokenizer,
            mock=self.mock,
            mock_vocabulary_size=len(MOCK_WORDS),
            model_paths=self.model_paths,
            active_model_id=self.active_model_id,
        )
        self.tokenizer_identity_cache = payload
        return dict(payload)

        if self.mock:
            payload = {
                "kind": "deterministic_mock_tokenizer",
                "algorithm": "development_mock",
                "vocab_size": len(MOCK_WORDS),
                "bos_token_id": 0,
                "eos_token_id": None,
                "shared_across_model_modes": True,
                "verification": "single_mock_runtime",
                "tokenizer_json_sha256": None,
                "model_files": [],
            }
            self.tokenizer_identity_cache = payload
            return dict(payload)

        entries: list[dict[str, Any]] = []
        for model_id, model_path in self.model_paths.items():
            tokenizer_path = model_path / "tokenizer.json"
            if not tokenizer_path.is_file():
                continue
            try:
                contents = tokenizer_path.read_bytes()
            except OSError:
                continue
            entries.append(
                {
                    "model_id": model_id,
                    "directory": model_path.name,
                    "bytes": len(contents),
                    "sha256": hashlib.sha256(contents).hexdigest(),
                }
            )
        fingerprints = {
            str(entry["sha256"])
            for entry in entries
        }
        shared = bool(entries) and len(fingerprints) == 1
        tokenizer = self.tokenizer
        payload = {
            "kind": (
                f"{type(tokenizer).__module__}."
                f"{type(tokenizer).__qualname__}"
            ),
            "algorithm": "byte_level_bpe",
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "shared_across_model_modes": shared,
            "verification": (
                "exact_tokenizer_json_sha256"
                if shared
                else "tokenizer_files_differ_or_are_unavailable"
            ),
            "tokenizer_json_sha256": (
                next(iter(fingerprints))
                if len(fingerprints) == 1
                else None
            ),
            "model_files": entries,
        }
        self.tokenizer_identity_cache = payload
        return dict(payload)

    @staticmethod
    def _orthographic_word_ranges(
        text: str,
    ) -> list[tuple[int, int]]:
        """Return maximal observable word-like spans in Unicode text.

        This deliberately defines an operational orthographic unit, not a
        lemma or a morphologically analysed word. A span starts with a
        Unicode letter or number, may continue with combining marks, and may
        contain an internal apostrophe or hyphen when another letter/number
        follows it.
        """
        return TokenizationService.orthographic_word_ranges(text)

        def starts_word(character: str) -> bool:
            return unicodedata.category(character)[:1] in {"L", "N"}

        def continues_word(character: str) -> bool:
            return unicodedata.category(character)[:1] in {"L", "M", "N"}

        joiners = {"'", "’", "-", "‑"}
        ranges: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(text):
            if not starts_word(text[cursor]):
                cursor += 1
                continue
            start = cursor
            cursor += 1
            while cursor < len(text):
                if continues_word(text[cursor]):
                    cursor += 1
                    continue
                if (
                    text[cursor] in joiners
                    and cursor + 1 < len(text)
                    and starts_word(text[cursor + 1])
                ):
                    cursor += 1
                    continue
                break
            ranges.append((start, cursor))
        return ranges

    @classmethod
    def _orthographic_word_spans(
        cls,
        text: str,
        tokens: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Map orthographic spans to the exact tokenizer units they touch."""
        return TokenizationService.orthographic_word_spans(text, tokens)

        spans: list[dict[str, Any]] = []
        for character_start, character_end in cls._orthographic_word_ranges(
            text
        ):
            byte_start = len(text[:character_start].encode("utf-8"))
            byte_end = len(text[:character_end].encode("utf-8"))
            token_indices: list[int] = []
            for token in tokens:
                token_byte_start = token.get("utf8_byte_start")
                token_byte_end = token.get("utf8_byte_end")
                if (
                    type(token_byte_start) is int
                    and type(token_byte_end) is int
                    and max(byte_start, token_byte_start)
                    < min(byte_end, token_byte_end)
                ):
                    token_indices.append(int(token["index"]))
            contiguous = all(
                index == 0
                or token_index == token_indices[index - 1] + 1
                for index, token_index in enumerate(token_indices)
            )
            if not token_indices or not contiguous:
                continue
            spans.append(
                {
                    "text": text[character_start:character_end],
                    "character_start": character_start,
                    "character_end": character_end,
                    "utf8_byte_start": byte_start,
                    "utf8_byte_end": byte_end,
                    "token_indices": token_indices,
                    "token_start_index": token_indices[0],
                    "token_end_index": token_indices[-1],
                    "token_ids": [
                        int(tokens[index]["token_id"])
                        for index in token_indices
                    ],
                    "token_count": len(token_indices),
                }
            )
        return spans

    def _tokenization_text_payload(
        self,
        text: str,
        *,
        label: str,
    ) -> dict[str, Any]:
        return self._tokenization_service().text_payload(
            text,
            label=label,
        )

        (
            token_ids,
            _context_token_ids,
            pieces,
            raw_tokens,
            offsets,
        ) = self._text_surprisal_tokenization(text)
        if not token_ids:
            raise ValueError(
                f"{label} po tokenizaci neobsahuje žádný textový token."
            )
        if len(token_ids) > MAX_TOKENIZATION_TOKENS:
            raise ValueError(
                f"{label} má {len(token_ids)} tokenů; limit je "
                f"{MAX_TOKENIZATION_TOKENS}."
            )

        if self.mock:
            encoded_with_special = [0, *token_ids]
            decoded = "".join(pieces)
            special_ids = {0}
        else:
            encoded_with_special = encode_with_bos(self.tokenizer, text)
            decoded = safe_decode(
                self.tokenizer,
                token_ids,
                skip_special=True,
            )
            special_ids = set(
                getattr(self.tokenizer, "all_special_ids", [])
            )

        text_utf8 = text.encode("utf-8")
        raw_byte_pieces = [
            byte_level_token_bytes(raw_token)
            for raw_token in raw_tokens
        ]
        exact_byte_offsets: list[tuple[int, int]] | None = None
        if (
            raw_byte_pieces
            and all(piece is not None for piece in raw_byte_pieces)
            and b"".join(
                piece
                for piece in raw_byte_pieces
                if piece is not None
            ) == text_utf8
        ):
            exact_byte_offsets = []
            byte_cursor = 0
            for byte_piece in raw_byte_pieces:
                assert byte_piece is not None
                byte_end = byte_cursor + len(byte_piece)
                exact_byte_offsets.append((byte_cursor, byte_end))
                byte_cursor = byte_end

        byte_boundary_to_character = {0: 0}
        character_byte_ranges: list[tuple[int, int]] = []
        byte_cursor = 0
        for character_index, character in enumerate(text):
            byte_end = byte_cursor + len(character.encode("utf-8"))
            character_byte_ranges.append((byte_cursor, byte_end))
            byte_boundary_to_character[byte_end] = character_index + 1
            byte_cursor = byte_end

        tokens: list[dict[str, Any]] = []
        for index, (
            token_id_value,
            piece,
            raw_token,
            offset,
        ) in enumerate(
            zip(
                token_ids,
                pieces,
                raw_tokens,
                offsets,
                strict=True,
            )
        ):
            token_id = int(token_id_value)
            decoded_single = (
                piece
                if self.mock
                else safe_decode(
                    self.tokenizer,
                    [token_id],
                    skip_special=False,
                )
            )
            character_start: int | None = None
            character_end: int | None = None
            utf8_byte_start: int | None = None
            utf8_byte_end: int | None = None
            source_piece = piece
            if offset is not None:
                character_start, character_end = offset
                source_piece = text[character_start:character_end]
                utf8_byte_start = len(
                    text[:character_start].encode("utf-8")
                )
                utf8_byte_end = len(
                    text[:character_end].encode("utf-8")
                )
            character_range_exact = (
                character_start is not None
                and character_end is not None
            )
            byte_range_exact = character_range_exact
            covering_character_start: int | None = None
            covering_character_end: int | None = None
            if exact_byte_offsets is not None:
                utf8_byte_start, utf8_byte_end = exact_byte_offsets[index]
                byte_range_exact = True
                start_character = byte_boundary_to_character.get(
                    utf8_byte_start
                )
                end_character = byte_boundary_to_character.get(
                    utf8_byte_end
                )
                if (
                    start_character is not None
                    and end_character is not None
                ):
                    character_start = start_character
                    character_end = end_character
                    source_piece = text[character_start:character_end]
                    character_range_exact = True
                else:
                    character_start = None
                    character_end = None
                    character_range_exact = False
                    overlapping_characters = [
                        character_index
                        for character_index, (
                            character_byte_start,
                            character_byte_end,
                        ) in enumerate(character_byte_ranges)
                        if (
                            character_byte_end > utf8_byte_start
                            and character_byte_start < utf8_byte_end
                        )
                    ]
                    if overlapping_characters:
                        covering_character_start = min(
                            overlapping_characters
                        )
                        covering_character_end = (
                            max(overlapping_characters) + 1
                        )
                    source_piece = ""
            source_bytes = (
                text_utf8[utf8_byte_start:utf8_byte_end]
                if (
                    byte_range_exact
                    and utf8_byte_start is not None
                    and utf8_byte_end is not None
                )
                else source_piece.encode("utf-8")
            )
            splits_unicode_code_point = (
                byte_range_exact and not character_range_exact
            )
            display_piece = (
                f"⟦{'·'.join(f'{value:02X}' for value in source_bytes)}⟧"
                if splits_unicode_code_point
                else raw_token
                if (
                    offset is None
                    and raw_token
                    and ("\ufffd" in decoded_single or not decoded_single)
                )
                else source_piece
            )
            tokens.append(
                {
                    "index": index,
                    "token_id": token_id,
                    "display": visible_token(
                        display_piece,
                        f"token {token_id}",
                    ),
                    "source_text": (
                        source_piece if character_range_exact else None
                    ),
                    "source_text_cover": (
                        text[
                            covering_character_start:
                            covering_character_end
                        ]
                        if (
                            covering_character_start is not None
                            and covering_character_end is not None
                        )
                        else None
                    ),
                    "decoded_text": decoded_single,
                    "tokenizer_token": raw_token,
                    "is_special": token_id in special_ids,
                    "piece_is_lossless": character_range_exact,
                    "character_range_exact": character_range_exact,
                    "utf8_byte_range_exact": byte_range_exact,
                    "splits_unicode_code_point": (
                        splits_unicode_code_point
                    ),
                    "character_start": character_start,
                    "character_end": character_end,
                    "covering_character_start": (
                        covering_character_start
                    ),
                    "covering_character_end": covering_character_end,
                    "character_count": (
                        character_end - character_start
                        if character_start is not None
                        and character_end is not None
                        else None
                    ),
                    "utf8_byte_start": utf8_byte_start,
                    "utf8_byte_end": utf8_byte_end,
                    "utf8_byte_count": len(source_bytes),
                    "utf8_hex": " ".join(
                        f"{value:02X}" for value in source_bytes
                    ),
                    "code_points": [
                        f"U+{ord(character):04X}"
                        for character in (
                            source_piece
                            if character_range_exact
                            else (
                                text[
                                    covering_character_start:
                                    covering_character_end
                                ]
                                if (
                                    covering_character_start is not None
                                    and covering_character_end is not None
                                )
                                else ""
                            )
                        )
                    ],
                }
            )

        bos_token_id = (
            0
            if self.mock
            else getattr(self.tokenizer, "bos_token_id", None)
        )
        special_prefix: list[dict[str, Any]] = []
        ordinary_start = 0
        if (
            type(bos_token_id) is int
            and encoded_with_special[:1] == [bos_token_id]
        ):
            raw_bos: str | None = None
            decoded_bos = ""
            if not self.mock:
                try:
                    converted = self.tokenizer.convert_ids_to_tokens(
                        bos_token_id
                    )
                    raw_bos = (
                        str(converted) if converted is not None else None
                    )
                except Exception:
                    raw_bos = None
                decoded_bos = safe_decode(
                    self.tokenizer,
                    [bos_token_id],
                    skip_special=False,
                )
            special_prefix.append(
                {
                    "index": -1,
                    "token_id": bos_token_id,
                    "display": raw_bos or "<s>",
                    "source_text": "",
                    "decoded_text": decoded_bos,
                    "tokenizer_token": raw_bos or "<s>",
                    "is_special": True,
                    "role": "bos",
                    "piece_is_lossless": True,
                    "character_start": 0,
                    "character_end": 0,
                    "character_count": 0,
                    "utf8_byte_start": 0,
                    "utf8_byte_end": 0,
                    "utf8_byte_count": 0,
                    "utf8_hex": "",
                    "code_points": [],
                }
            )
            ordinary_start = 1

        boundaries: list[dict[str, Any]] = []
        for boundary_index in range(len(token_ids) + 1):
            if boundary_index == 0:
                prefix_text = ""
                character_offset: int | None = 0
                byte_offset: int | None = 0
            else:
                previous = tokens[boundary_index - 1]
                byte_offset = previous["utf8_byte_end"]
                character_offset = (
                    byte_boundary_to_character.get(byte_offset)
                    if type(byte_offset) is int
                    else None
                )
                if character_offset is not None:
                    prefix_text = text[:character_offset]
                else:
                    prefix_ids = token_ids[:boundary_index]
                    prefix_text = (
                        "".join(pieces[:boundary_index])
                        if self.mock
                        else safe_decode(
                            self.tokenizer,
                            prefix_ids,
                            skip_special=True,
                        )
                    )
            prefix_ids = list(
                encoded_with_special[
                    : ordinary_start + boundary_index
                ]
            )
            next_token = (
                dict(tokens[boundary_index])
                if boundary_index < len(tokens)
                else None
            )
            boundaries.append(
                {
                    "index": boundary_index,
                    "prefix_text": prefix_text,
                    "prefix_token_ids": prefix_ids,
                    "text_token_count": boundary_index,
                    "character_offset": character_offset,
                    "utf8_byte_offset": (
                        byte_offset
                        if type(byte_offset) is int
                        else (
                            len(prefix_text.encode("utf-8"))
                            if character_offset is not None
                            else None
                        )
                    ),
                    "next_token": next_token,
                    "at_end": boundary_index == len(tokens),
                    "prefix_roundtrip_exact": (
                        character_offset is not None
                        and text[:character_offset] == prefix_text
                    ),
                }
            )

        word_spans = self._orthographic_word_spans(text, tokens)
        exact_character_ranges = sum(
            token["character_range_exact"]
            for token in tokens
        )
        exact_byte_ranges = sum(
            token["utf8_byte_range_exact"]
            for token in tokens
        )
        unicode_payload = self._unicode_text_payload(text)
        return {
            "label": label,
            "text": text,
            "unicode": unicode_payload,
            "token_count": len(tokens),
            "token_ids": [int(token_id) for token_id in token_ids],
            "encoded_with_special_token_ids": [
                int(token_id) for token_id in encoded_with_special
            ],
            "special_prefix_tokens": special_prefix,
            "tokens": tokens,
            "word_spans": word_spans,
            "word_span_definition": {
                "kind": "maximal_unicode_orthographic_span_v1",
                "starts_with": "Unicode letter or number",
                "continues_with": "Unicode letter, mark, or number",
                "internal_joiners": ["'", "’", "-", "‑"],
                "linguistic_claim": (
                    "observable_text_span_not_lemma_or_morphological_word"
                ),
            },
            "boundaries": boundaries,
            "detokenized_text": decoded,
            "roundtrip_exact": decoded == text,
            "offset_coverage": {
                "exact_token_count": exact_character_ranges,
                "total_token_count": len(tokens),
                "ratio": exact_character_ranges / len(tokens),
                "exact_character_token_count": exact_character_ranges,
                "character_ratio": exact_character_ranges / len(tokens),
                "exact_utf8_byte_token_count": exact_byte_ranges,
                "utf8_byte_ratio": exact_byte_ranges / len(tokens),
            },
        }

    @staticmethod
    def _tokenization_comparison(
        a: dict[str, Any],
        b: dict[str, Any],
    ) -> dict[str, Any]:
        return TokenizationService.comparison(a, b)

        ids_a = list(a["token_ids"])
        ids_b = list(b["token_ids"])
        common_prefix = 0
        for left, right in zip(ids_a, ids_b):
            if left != right:
                break
            common_prefix += 1

        max_suffix = min(len(ids_a), len(ids_b)) - common_prefix
        common_suffix = 0
        while (
            common_suffix < max_suffix
            and ids_a[-1 - common_suffix] == ids_b[-1 - common_suffix]
        ):
            common_suffix += 1

        first_token_divergence: dict[str, Any] | None = None
        if ids_a != ids_b:
            first_token_divergence = {
                "index": common_prefix,
                "a": (
                    dict(a["tokens"][common_prefix])
                    if common_prefix < len(ids_a)
                    else None
                ),
                "b": (
                    dict(b["tokens"][common_prefix])
                    if common_prefix < len(ids_b)
                    else None
                ),
            }

        text_a = str(a["text"])
        text_b = str(b["text"])
        common_character_prefix = 0
        for left, right in zip(text_a, text_b):
            if left != right:
                break
            common_character_prefix += 1
        first_character_divergence = (
            None
            if text_a == text_b
            else {
                "index": common_character_prefix,
                "a": (
                    f"U+{ord(text_a[common_character_prefix]):04X}"
                    if common_character_prefix < len(text_a)
                    else None
                ),
                "b": (
                    f"U+{ord(text_b[common_character_prefix]):04X}"
                    if common_character_prefix < len(text_b)
                    else None
                ),
            }
        )

        return {
            "same_text": text_a == text_b,
            "same_utf8_bytes": text_a.encode("utf-8") == text_b.encode("utf-8"),
            "canonically_equivalent_nfc": (
                unicodedata.normalize("NFC", text_a)
                == unicodedata.normalize("NFC", text_b)
            ),
            "compatibility_equivalent_nfkc": (
                unicodedata.normalize("NFKC", text_a)
                == unicodedata.normalize("NFKC", text_b)
            ),
            "identical_token_ids": ids_a == ids_b,
            "token_count_a": len(ids_a),
            "token_count_b": len(ids_b),
            "token_count_delta_b_minus_a": len(ids_b) - len(ids_a),
            "common_token_prefix_count": common_prefix,
            "common_token_suffix_count": common_suffix,
            "first_token_divergence": first_token_divergence,
            "common_character_prefix_count": common_character_prefix,
            "first_character_divergence": first_character_divergence,
        }

    def _tokenization_local(
        self,
        text_a: str,
        text_b: str | None,
    ) -> dict[str, Any]:
        return self._tokenization_service().analyze(
            text_a,
            text_b,
            tokenizer_payload=self._tokenizer_identity_unlocked(),
            model_id=self.active_model_id,
            model_label=self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            model_generation=self.model_generation,
            server_instance_id=self.server_instance_id,
        )

        started = time.perf_counter()
        for label, text in (("Text A", text_a), ("Text B", text_b)):
            if text is None:
                continue
            if not isinstance(text, str) or text == "":
                raise ValueError(f"{label} musí být neprázdný řetězec.")
            if len(text) > MAX_TOKENIZATION_CHARACTERS:
                raise ValueError(
                    f"{label} může mít nejvýše "
                    f"{MAX_TOKENIZATION_CHARACTERS} znaků."
                )
            if len(text.encode("utf-8")) > MAX_TOKENIZATION_UTF8_BYTES:
                raise ValueError(
                    f"{label} může mít nejvýše "
                    f"{MAX_TOKENIZATION_UTF8_BYTES} UTF-8 bajtů."
                )

        a = self._tokenization_text_payload(text_a, label="Text A")
        b = (
            self._tokenization_text_payload(text_b, label="Text B")
            if text_b is not None
            else None
        )
        tokenizer_payload = self._tokenizer_identity_unlocked()
        total_ms = (time.perf_counter() - started) * 1_000
        warnings: list[dict[str, str]] = []
        for side in (a, b):
            if side is None:
                continue
            if not side["roundtrip_exact"]:
                warnings.append(
                    {
                        "code": "roundtrip_not_exact",
                        "message": (
                            f"{side['label']}: detokenizace se přesně "
                            "nevrátila k původnímu textu."
                        ),
                    }
                )
            if side["offset_coverage"]["utf8_byte_ratio"] < 1.0:
                warnings.append(
                    {
                        "code": "offsets_incomplete",
                        "message": (
                            f"{side['label']}: některé tokeny nemají "
                            "spolehlivý bajtový rozsah."
                        ),
                    }
                )
            split_count = sum(
                bool(token["splits_unicode_code_point"])
                for token in side["tokens"]
            )
            if split_count:
                warnings.append(
                    {
                        "code": "tokens_split_utf8_code_point",
                        "message": (
                            f"{side['label']}: {split_count} tokenů dělí "
                            "UTF‑8 uvnitř Unicode kódového bodu; bajtové "
                            "rozsahy jsou přesné, samostatný znakový úsek "
                            "pro ně neexistuje."
                        ),
                    }
                )

        return {
            "analysis_kind": "tokenization",
            "a": a,
            "b": b,
            "comparison": (
                self._tokenization_comparison(a, b)
                if b is not None
                else None
            ),
            "tokenizer": tokenizer_payload,
            "execution": {
                "mode": "deterministic_tokenization_only",
                "forward_passes": 0,
                "generation": False,
                "probabilities": False,
                "model_weights_used": False,
                "normalization_applied_to_text": False,
            },
            "warnings": warnings,
            "model_id": self.active_model_id,
            "model_label": self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
            "timing_ms": {
                "total": total_ms,
            },
            "latency_ms": total_ms,
        }

    @staticmethod
    def _embedding_cosine(
        left: np.ndarray,
        right: np.ndarray,
    ) -> float:
        left64 = np.asarray(left, dtype=np.float64)
        right64 = np.asarray(right, dtype=np.float64)
        denominator = float(
            np.linalg.norm(left64) * np.linalg.norm(right64)
        )
        if denominator <= 1e-12:
            return 0.0
        return float(
            np.clip(
                np.dot(left64, right64) / denominator,
                -1.0,
                1.0,
            )
        )

    @staticmethod
    def _mock_embedding_rows(token_ids: list[int]) -> np.ndarray:
        """Return a stable synthetic lookup table for mock-mode contracts."""
        width = 64
        rows: list[np.ndarray] = []
        for token_id in token_ids:
            digest = hashlib.sha256(
                f"apertus-mock-input-embedding-v1\0{int(token_id)}".encode(
                    "utf-8"
                )
            ).digest()
            rng = np.random.default_rng(
                int.from_bytes(digest[:8], "big")
            )
            row = rng.normal(0.0, 0.45, size=width)
            # A weak shared structure gives the mock vocabulary meaningful
            # but non-degenerate neighbours without pretending to model
            # Apertus itself.
            row[int(token_id) % 8] += 1.2
            row[8 + (int(token_id) // 4) % 8] += 0.65
            rows.append(row.astype(np.float32))
        return np.stack(rows, axis=0)

    def _embedding_rows_local(
        self,
        token_ids: list[int],
    ) -> np.ndarray:
        if not token_ids:
            raise ValueError("Je třeba vyžádat alespoň jeden embedding.")
        if self.mock:
            return self._mock_embedding_rows(token_ids)

        import mlx.core as mx

        body = self._steering_body()
        embedding = getattr(body, "embed_tokens", None)
        if not callable(embedding):
            raise RuntimeError(
                "Aktivní model nemá použitelnou vstupní embeddingovou "
                "tabulku."
            )
        ids = mx.array([[int(token_id) for token_id in token_ids]])
        rows = embedding(ids)[0].astype(mx.float32)
        mx.eval(rows)
        return np.asarray(rows)

    def _embedding_neighbors_local(
        self,
        token_id: int,
        query: np.ndarray,
        neighbor_count: int,
    ) -> tuple[list[dict[str, Any]], np.ndarray, float]:
        started = time.perf_counter()
        vocabulary_size = (
            len(MOCK_WORDS)
            if self.mock
            else int(
                getattr(
                    getattr(self.model, "args", None),
                    "vocab_size",
                    getattr(self.tokenizer, "vocab_size", 0),
                )
            )
        )
        if vocabulary_size <= 0:
            raise RuntimeError("Velikost slovníku není dostupná.")
        if not 0 <= token_id < vocabulary_size:
            raise ValueError(
                f"Token #{token_id} leží mimo slovník 0–"
                f"{vocabulary_size - 1}."
            )

        query64 = np.asarray(query, dtype=np.float64)
        query_norm = float(np.linalg.norm(query64))
        if query_norm <= 1e-12:
            raise RuntimeError(
                "Vybraný embeddingový řádek má nulovou normu; "
                "kosinové sousedy nelze určit."
            )

        candidates: list[tuple[float, int, float, float]] = []
        if self.mock:
            all_ids = list(range(vocabulary_size))
            rows = self._mock_embedding_rows(all_ids).astype(np.float64)
            norms = np.linalg.norm(rows, axis=1)
            dots = rows @ query64
            cosines = dots / np.maximum(norms * query_norm, 1e-12)
            for candidate_id, cosine, norm, dot in zip(
                all_ids,
                cosines,
                norms,
                dots,
                strict=True,
            ):
                if candidate_id != token_id and math.isfinite(float(cosine)):
                    candidates.append(
                        (
                            float(cosine),
                            int(candidate_id),
                            float(norm),
                            float(dot),
                        )
                    )
        else:
            import mlx.core as mx

            embedding = self.model.model.embed_tokens
            query_mx = mx.array(query.astype(np.float32))
            for start in range(
                0,
                vocabulary_size,
                EMBEDDING_SEARCH_CHUNK_SIZE,
            ):
                stop = min(
                    vocabulary_size,
                    start + EMBEDDING_SEARCH_CHUNK_SIZE,
                )
                chunk_ids = list(range(start, stop))
                rows = embedding(mx.array([chunk_ids]))[0].astype(mx.float32)
                dots_mx = rows @ query_mx
                norms_mx = mx.sqrt(mx.sum(rows * rows, axis=-1))
                cosines_mx = dots_mx / (
                    norms_mx * float(query_norm) + 1e-12
                )
                mx.eval(cosines_mx, norms_mx, dots_mx)
                cosines = np.asarray(cosines_mx)
                norms = np.asarray(norms_mx)
                dots = np.asarray(dots_mx)
                local_count = min(
                    neighbor_count + 1,
                    len(chunk_ids),
                )
                if local_count == len(chunk_ids):
                    local_indices = np.arange(len(chunk_ids))
                else:
                    local_indices = np.argpartition(
                        cosines,
                        -local_count,
                    )[-local_count:]
                for local_index in local_indices:
                    candidate_id = start + int(local_index)
                    cosine = float(cosines[local_index])
                    if (
                        candidate_id != token_id
                        and math.isfinite(cosine)
                    ):
                        candidates.append(
                            (
                                cosine,
                                candidate_id,
                                float(norms[local_index]),
                                float(dots[local_index]),
                            )
                        )

        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected = candidates[:neighbor_count]
        neighbor_ids = [candidate_id for _, candidate_id, _, _ in selected]
        neighbor_rows = self._embedding_rows_local(neighbor_ids)
        neighbors: list[dict[str, Any]] = []
        for rank, (
            cosine,
            candidate_id,
            norm,
            dot,
        ) in enumerate(selected, start=1):
            angle = math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))
            neighbors.append(
                {
                    **self._lens_token_payload(candidate_id),
                    "rank": rank,
                    "cosine_similarity": cosine,
                    "cosine_distance": 1.0 - cosine,
                    "angle_degrees": float(angle),
                    "norm": norm,
                    "dot_product": dot,
                }
            )
        return (
            neighbors,
            neighbor_rows,
            (time.perf_counter() - started) * 1_000,
        )

    def _embedding_neighborhood_local(
        self,
        token_id: int,
        neighbor_count: int,
    ) -> dict[str, Any]:
        """Return one on-demand kNN star over input-embedding rows."""
        total_started = time.perf_counter()
        if type(token_id) is not int:
            raise ValueError("token_id musí být celé číslo.")
        if type(neighbor_count) is not int or not (
            MIN_EMBEDDING_GRAPH_NEIGHBORS
            <= neighbor_count
            <= MAX_EMBEDDING_GRAPH_NEIGHBORS
        ):
            raise ValueError(
                "neighbor_count musí být celé číslo mezi "
                f"{MIN_EMBEDDING_GRAPH_NEIGHBORS} a "
                f"{MAX_EMBEDDING_GRAPH_NEIGHBORS}."
            )

        cache_key = (
            self.model_generation,
            self.active_model_id,
            token_id,
            neighbor_count,
        )
        with self.cache_lock:
            cached = self.embedding_neighborhood_cache.get(cache_key)
            if cached is not None:
                self.embedding_neighborhood_cache.move_to_end(cache_key)
                response = dict(cached)
                response["cached"] = True
                response["latency_ms"] = 0.0
                response["timing_ms"] = dict(response["timing_ms"])
                response["timing_ms"]["total"] = 0.0
                return response

        vocabulary_size = (
            len(MOCK_WORDS)
            if self.mock
            else int(
                getattr(
                    getattr(self.model, "args", None),
                    "vocab_size",
                    getattr(self.tokenizer, "vocab_size", 0),
                )
            )
        )
        if vocabulary_size <= 0:
            raise RuntimeError("Velikost slovníku není dostupná.")
        if not 0 <= token_id < vocabulary_size:
            raise ValueError(
                f"Token #{token_id} leží mimo slovník 0–"
                f"{vocabulary_size - 1}."
            )

        row_started = time.perf_counter()
        row = self._embedding_rows_local([token_id])[0]
        row_ms = (time.perf_counter() - row_started) * 1_000
        neighbors, _, neighbor_ms = self._embedding_neighbors_local(
            token_id,
            row,
            neighbor_count,
        )
        total_ms = (time.perf_counter() - total_started) * 1_000
        response = {
            "analysis_kind": "input_embedding_token_neighborhood",
            "root": {
                **self._lens_token_payload(token_id),
                "embedding": self._embedding_row_summary(row),
            },
            "neighbors": neighbors,
            "graph_contract": {
                "node_unit": "tokenizer_vocabulary_row",
                "edge_metric": "cosine_similarity_of_input_embedding_rows",
                "layout": (
                    "client_side_schematic_topology_not_embedding_geometry"
                ),
                "claim": (
                    "token_rows_not_words_synonyms_meanings_or_causal_links"
                ),
                "full_embedding_vector_returned": False,
            },
            "execution": {
                "generation": False,
                "forward_passes": 0,
                "vocabulary_scan": True,
                "probabilities": False,
            },
            "model_id": self.active_model_id,
            "model_label": self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
            "cached": False,
            "timing_ms": {
                "embedding_row": row_ms,
                "neighbor_scan": neighbor_ms,
                "total": total_ms,
            },
            "latency_ms": total_ms,
        }
        with self.cache_lock:
            self.embedding_neighborhood_cache[cache_key] = response
            self.embedding_neighborhood_cache.move_to_end(cache_key)
            while (
                len(self.embedding_neighborhood_cache)
                > EMBEDDING_NEIGHBORHOOD_CACHE_SIZE
            ):
                self.embedding_neighborhood_cache.popitem(last=False)
        return response

    @staticmethod
    def _embedding_local_projection(
        token_payloads: list[dict[str, Any]],
        rows: np.ndarray,
    ) -> dict[str, Any]:
        matrix = np.asarray(rows, dtype=np.float64)
        centered = matrix - np.mean(matrix, axis=0, keepdims=True)
        if len(matrix) < 2 or not np.any(centered):
            coordinates = np.zeros((len(matrix), 2), dtype=np.float64)
            explained = [0.0, 0.0]
        else:
            u, singular_values, _ = np.linalg.svd(
                centered,
                full_matrices=False,
            )
            dimensions = min(2, len(singular_values))
            coordinates = np.zeros((len(matrix), 2), dtype=np.float64)
            coordinates[:, :dimensions] = (
                u[:, :dimensions] * singular_values[:dimensions]
            )
            variances = singular_values**2
            variance_total = float(np.sum(variances))
            explained = [
                (
                    float(variances[index] / variance_total)
                    if variance_total > 0 and index < len(variances)
                    else 0.0
                )
                for index in range(2)
            ]
            for axis in range(2):
                pivot = int(np.argmax(np.abs(coordinates[:, axis])))
                if coordinates[pivot, axis] < 0:
                    coordinates[:, axis] *= -1

        points = []
        for index, (token, coordinate) in enumerate(
            zip(token_payloads, coordinates, strict=True)
        ):
            points.append(
                {
                    **token,
                    "x": float(coordinate[0]),
                    "y": float(coordinate[1]),
                    "selected": index == 0,
                }
            )
        return {
            "method": "local_pca_selected_row_and_displayed_neighbors",
            "points": points,
            "explained_variance_ratio": explained,
            "scope_warning": (
                "Souřadnice jsou lokální PCA pouze nad zobrazenými "
                "řádky; nejsou globální mapou slovníku."
            ),
        }

    @staticmethod
    def _embedding_row_summary(row: np.ndarray) -> dict[str, Any]:
        """Return bounded, inspectable statistics without exposing the row."""
        row_float32 = np.asarray(row, dtype="<f4")
        return {
            "dimension": int(row_float32.size),
            "runtime_dtype": "float32_measurement_copy",
            "norm": float(
                np.linalg.norm(
                    np.asarray(row_float32, dtype=np.float64)
                )
            ),
            "mean": float(np.mean(row_float32)),
            "standard_deviation": float(np.std(row_float32)),
            "mean_absolute_value": float(
                np.mean(np.abs(row_float32))
            ),
            "minimum": float(np.min(row_float32)),
            "maximum": float(np.max(row_float32)),
            "float32_sha256": hashlib.sha256(
                row_float32.tobytes()
            ).hexdigest(),
            "coordinate_preview": [
                {
                    "index": index,
                    "value": float(value),
                }
                for index, value in enumerate(row_float32[:16])
            ],
            "coordinate_preview_warning": (
                "Prvních 16 souřadnic je kontrolní náhled, nikoli "
                "sémantické vysvětlení os."
            ),
        }

    def _embedding_context_trajectory(
        self,
        prefix_token_ids: list[int],
        *,
        context_key: str,
    ) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        if self.mock:
            selected_id = int(prefix_token_ids[-1])
            base = self._mock_embedding_rows([selected_id])[0].astype(
                np.float64
            )
            digest = hashlib.sha256(
                (
                    "apertus-mock-context-trajectory-v1\0"
                    + context_key
                    + "\0"
                    + ",".join(str(value) for value in prefix_token_ids)
                ).encode("utf-8")
            ).digest()
            rng = np.random.default_rng(
                int.from_bytes(digest[:8], "big")
            )
            direction = rng.normal(0.0, 1.0, size=base.size)
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            companion = rng.normal(0.0, 1.0, size=base.size)
            companion -= direction * float(np.dot(companion, direction))
            companion /= max(float(np.linalg.norm(companion)), 1e-12)
            scale = max(float(np.linalg.norm(base)), 1.0)
            states = [base.astype(np.float32)]
            layer_count = self.representation_layer_count()
            for layer in range(1, layer_count + 1):
                progress = layer / layer_count
                state = (
                    base
                    + scale
                    * (
                        0.95 * progress * direction
                        + 0.18
                        * math.sin(progress * math.pi)
                        * companion
                    )
                )
                states.append(state.astype(np.float32))
            return (
                np.stack(states, axis=0),
                (time.perf_counter() - started) * 1_000,
            )

        import mlx.core as mx
        self._steering_body()
        _body, states, _final_hidden = trace_last_position(self.model, prefix_token_ids)
        stacked = states.astype(mx.float32)
        mx.eval(stacked)
        return (
            np.asarray(stacked),
            (time.perf_counter() - started) * 1_000,
        )

    def _embeddings_local(
        self,
        contexts: list[dict[str, Any]],
        neighbor_count: int,
        analysis_mode: str = "token",
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        layer_count = self.representation_layer_count()
        if analysis_mode not in {"token", "word_span"}:
            raise ValueError(
                "analysis_mode musí být token nebo word_span."
            )
        if not 1 <= len(contexts) <= MAX_EMBEDDING_CONTEXTS:
            raise ValueError(
                "contexts musí obsahovat 1 až "
                f"{MAX_EMBEDDING_CONTEXTS} kontexty."
            )
        if not (
            MIN_EMBEDDING_NEIGHBORS
            <= neighbor_count
            <= MAX_EMBEDDING_NEIGHBORS
        ):
            raise ValueError(
                "neighbor_count musí být mezi "
                f"{MIN_EMBEDDING_NEIGHBORS} a "
                f"{MAX_EMBEDDING_NEIGHBORS}."
            )

        normalized_contexts: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        span_token_sequences: list[tuple[int, ...]] = []
        span_texts: list[str] = []
        for position, context in enumerate(contexts):
            if not isinstance(context, dict):
                raise ValueError("Každý kontext musí být JSON objekt.")
            text = context.get("text")
            label_value = context.get("label", f"Kontext {position + 1}")
            token_index = context.get("token_index")
            if not isinstance(text, str) or not text:
                raise ValueError("Každý kontext musí mít neprázdný text.")
            if len(text) > MAX_EMBEDDING_CHARACTERS:
                raise ValueError(
                    f"Kontext může mít nejvýše "
                    f"{MAX_EMBEDDING_CHARACTERS} znaků."
                )
            if len(text.encode("utf-8")) > MAX_EMBEDDING_UTF8_BYTES:
                raise ValueError(
                    f"Kontext může mít nejvýše "
                    f"{MAX_EMBEDDING_UTF8_BYTES} UTF-8 bajtů."
                )
            if not isinstance(label_value, str):
                raise ValueError("Popisek kontextu musí být řetězec.")
            label = label_value.strip() or f"Kontext {position + 1}"
            if len(label) > 120:
                raise ValueError(
                    "Popisek kontextu může mít nejvýše 120 znaků."
                )
            if type(token_index) is not int or token_index < 0:
                raise ValueError(
                    "token_index musí být nezáporné celé číslo."
                )
            trace = self._tokenization_text_payload(text, label=label)
            if trace["token_count"] > MAX_EMBEDDING_TOKENS:
                raise ValueError(
                    f"{label} má {trace['token_count']} tokenů; limit je "
                    f"{MAX_EMBEDDING_TOKENS}."
                )
            if token_index >= trace["token_count"]:
                raise ValueError(
                    f"{label}: token_index {token_index} leží mimo "
                    f"0–{trace['token_count'] - 1}."
                )

            if analysis_mode == "word_span":
                span_start = context.get("span_start_token_index")
                span_end = context.get("span_end_token_index")
                character_start = context.get("span_character_start")
                character_end = context.get("span_character_end")
                requested_span_text = context.get("span_text")
                for field_name, value in (
                    ("span_start_token_index", span_start),
                    ("span_end_token_index", span_end),
                    ("span_character_start", character_start),
                    ("span_character_end", character_end),
                ):
                    if type(value) is not int or value < 0:
                        raise ValueError(
                            f"{label}: {field_name} musí být nezáporné "
                            "celé číslo."
                        )
                assert type(span_start) is int
                assert type(span_end) is int
                assert type(character_start) is int
                assert type(character_end) is int
                if not (
                    0 <= span_start <= span_end < trace["token_count"]
                ):
                    raise ValueError(
                        f"{label}: tokenové rozpětí leží mimo 0–"
                        f"{trace['token_count'] - 1}."
                    )
                if token_index != span_end:
                    raise ValueError(
                        f"{label}: token_index musí být poslední token "
                        "slovního rozpětí."
                    )
                if not (
                    0 <= character_start < character_end <= len(text)
                ):
                    raise ValueError(
                        f"{label}: znakové rozpětí slova leží mimo text."
                    )
                if (
                    not isinstance(requested_span_text, str)
                    or not requested_span_text
                ):
                    raise ValueError(
                        f"{label}: span_text musí být neprázdný řetězec."
                    )
                actual_span_text = text[character_start:character_end]
                if requested_span_text != actual_span_text:
                    raise ValueError(
                        f"{label}: span_text neodpovídá přesnému úseku "
                        "vstupního textu."
                    )
                authoritative_spans = trace.get("word_spans")
                if not isinstance(authoritative_spans, list):
                    authoritative_spans = self._orthographic_word_spans(
                        text,
                        trace["tokens"],
                    )
                authoritative_span = next(
                    (
                        candidate
                        for candidate in authoritative_spans
                        if (
                            candidate.get("character_start")
                            == character_start
                            and candidate.get("character_end")
                            == character_end
                        )
                    ),
                    None,
                )
                if authoritative_span is None:
                    raise ValueError(
                        f"{label}: zvolené rozpětí není celé maximální "
                        "ortografické slovo podle serverové definice."
                    )
                if (
                    authoritative_span.get("text")
                    != actual_span_text
                ):
                    raise ValueError(
                        f"{label}: serverová podoba ortografického slova "
                        "neodpovídá zvolenému textu."
                    )

                span_byte_start = len(
                    text[:character_start].encode("utf-8")
                )
                span_byte_end = len(
                    text[:character_end].encode("utf-8")
                )
                overlapping_indices: list[int] = []
                for token in trace["tokens"]:
                    token_byte_start = token.get("utf8_byte_start")
                    token_byte_end = token.get("utf8_byte_end")
                    if (
                        type(token_byte_start) is not int
                        or type(token_byte_end) is not int
                    ):
                        raise ValueError(
                            f"{label}: tokenizer neposkytl přesné bajtové "
                            "rozsahy potřebné pro slovní rozpětí."
                        )
                    if (
                        max(token_byte_start, span_byte_start)
                        < min(token_byte_end, span_byte_end)
                    ):
                        overlapping_indices.append(int(token["index"]))
                expected_indices = list(range(span_start, span_end + 1))
                if overlapping_indices != expected_indices:
                    raise ValueError(
                        f"{label}: deklarované tokenové rozpětí "
                        "neodpovídá tokenům, které překrývají zvolené "
                        "slovo."
                    )
                if authoritative_span.get("token_indices") != expected_indices:
                    raise ValueError(
                        f"{label}: deklarované tokenové rozpětí "
                        "neodpovídá serverové mapě celého slova."
                    )
                span_ids = tuple(
                    int(trace["tokens"][index]["token_id"])
                    for index in expected_indices
                )
                span_token_sequences.append(span_ids)
                span_texts.append(actual_span_text)
            else:
                span_start = token_index
                span_end = token_index
                character_start = None
                character_end = None
                actual_span_text = ""
                span_token_sequences.append(
                    (
                        int(
                            trace["tokens"][token_index]["token_id"]
                        ),
                    )
                )

            normalized_contexts.append(
                {
                    "label": label,
                    "text": text,
                    "token_index": token_index,
                    "span_start_token_index": span_start,
                    "span_end_token_index": span_end,
                    "span_character_start": character_start,
                    "span_character_end": character_end,
                    "span_text": actual_span_text,
                }
            )
            traces.append(trace)

        selected_ids = [
            int(trace["tokens"][context["token_index"]]["token_id"])
            for trace, context in zip(
                traces,
                normalized_contexts,
                strict=True,
            )
        ]
        if analysis_mode == "token" and len(set(selected_ids)) != 1:
            detail = ", ".join(
                f"{context['label']}: #{token_id}"
                for context, token_id in zip(
                    normalized_contexts,
                    selected_ids,
                    strict=True,
                )
            )
            raise ValueError(
                "Všechny kontexty musí označovat stejné tokenové ID "
                f"({detail})."
            )
        if analysis_mode == "word_span":
            if len(set(span_texts)) != 1:
                raise ValueError(
                    "Všechna slovní rozpětí musí mít přesně stejný "
                    "viditelný text; Unicode se nenormalizuje."
                )
            if len(set(span_token_sequences)) != 1:
                detail = ", ".join(
                    (
                        f"{context['label']}: "
                        + " ".join(f"#{value}" for value in sequence)
                    )
                    for context, sequence in zip(
                        normalized_contexts,
                        span_token_sequences,
                        strict=True,
                    )
                )
                raise ValueError(
                    "Stejné slovo musí v každém kontextu tvořit shodnou "
                    f"posloupnost tokenových ID ({detail})."
                )
            if len(set(selected_ids)) != 1:
                raise ValueError(
                    "Shodná posloupnost celého slova musí končit shodným "
                    "tokenovým ID."
                )
        selected_id = selected_ids[0]
        cache_key = (
            self.model_generation,
            self.active_model_id,
            analysis_mode,
            tuple(
                (
                    context["label"],
                    context["text"],
                    context["token_index"],
                    context["span_start_token_index"],
                    context["span_end_token_index"],
                    context["span_character_start"],
                    context["span_character_end"],
                    context["span_text"],
                )
                for context in normalized_contexts
            ),
            neighbor_count,
        )
        with self.cache_lock:
            cached = self.embedding_cache.get(cache_key)
            if cached is not None:
                self.embedding_cache.move_to_end(cache_key)
                response = dict(cached)
                response["cached"] = True
                response["latency_ms"] = 0.0
                response["timing_ms"] = dict(response["timing_ms"])
                response["timing_ms"]["total"] = 0.0
                return response

        row_started = time.perf_counter()
        fragment_ids = list(span_token_sequences[0])
        fragment_rows = self._embedding_rows_local(fragment_ids)
        selected_row = fragment_rows[-1]
        row_ms = (time.perf_counter() - row_started) * 1_000
        selected_payload = self._lens_token_payload(selected_id)
        neighbors, neighbor_rows, neighbor_ms = (
            self._embedding_neighbors_local(
                selected_id,
                selected_row,
                neighbor_count,
            )
        )
        projection_payloads = [
            {
                **selected_payload,
                "rank": 0,
                "point_role": (
                    "selected_token"
                    if analysis_mode == "token"
                    else "closing_fragment"
                ),
                "fragment_position": (
                    len(fragment_ids) - 1
                    if analysis_mode == "word_span"
                    else None
                ),
            }
        ]
        projection_rows = [selected_row]
        if analysis_mode == "word_span":
            for position, (token_id, row) in enumerate(
                zip(fragment_ids[:-1], fragment_rows[:-1], strict=True)
            ):
                projection_payloads.append(
                    {
                        **self._lens_token_payload(token_id),
                        "rank": None,
                        "point_role": "word_fragment",
                        "fragment_position": position,
                    }
                )
                projection_rows.append(row)
        projection_payloads.extend(
            {
                **{
                    key: neighbor[key]
                    for key in (
                        "token_id",
                        "text",
                        "display",
                        "tokenizer_token",
                        "is_special",
                        "rank",
                    )
                },
                "point_role": "vocabulary_neighbor",
                "fragment_position": None,
            }
            for neighbor in neighbors
        )
        projection_rows.extend(neighbor_rows)
        projection: dict[str, Any] | None = (
            self._embedding_local_projection(
                projection_payloads,
                np.vstack(projection_rows),
            )
        )
        projection["method"] = (
            "local_pca_selected_row_and_displayed_neighbors"
            if analysis_mode == "token"
            else (
                "local_pca_word_fragment_rows_closing_fragment_neighbors"
            )
        )

        fragment_payloads = [
            {
                "position": position,
                "is_closing_fragment": position == len(fragment_ids) - 1,
                "token": self._lens_token_payload(token_id),
                "embedding": self._embedding_row_summary(row),
            }
            for position, (token_id, row) in enumerate(
                zip(fragment_ids, fragment_rows, strict=True)
            )
        ]
        selected_embedding = {
            "source": "model.model.embed_tokens",
            "lookup_contract": (
                "same_token_id_selects_same_row_in_active_model"
            ),
            "active_weight_mode": self.active_model_id,
            "role": (
                "selected_token_input_row"
                if analysis_mode == "token"
                else "closing_subtoken_input_row_not_whole_word"
            ),
            **self._embedding_row_summary(selected_row),
        }

        context_payloads: list[dict[str, Any]] = []
        trajectories: list[np.ndarray] = []
        forward_times: list[float] = []
        for trace, context in zip(
            traces,
            normalized_contexts,
            strict=True,
        ):
            label = str(context["label"])
            text = str(context["text"])
            token_index = int(context["token_index"])
            span_start = int(context["span_start_token_index"])
            span_end = int(context["span_end_token_index"])
            boundary = trace["boundaries"][token_index + 1]
            prefix_token_ids = [
                int(value) for value in boundary["prefix_token_ids"]
            ]
            states, forward_ms = self._embedding_context_trajectory(
                prefix_token_ids,
                context_key=str(boundary["prefix_text"]),
            )
            trajectories.append(states)
            forward_times.append(forward_ms)
            base_state = states[0]
            layers = []
            for layer, state in enumerate(states):
                cosine_to_input = self._embedding_cosine(
                    base_state,
                    state,
                )
                layers.append(
                    {
                        "layer": layer,
                        "relative_depth": layer / layer_count,
                        "state_summary": self._embedding_row_summary(state),
                        "norm": float(
                            np.linalg.norm(
                                np.asarray(state, dtype=np.float64)
                            )
                        ),
                        "cosine_to_input": cosine_to_input,
                        "cosine_distance_to_input": (
                            1.0 - cosine_to_input
                        ),
                    }
                )
            end_character = trace["tokens"][token_index][
                "character_end"
            ]
            context_payloads.append(
                {
                    "label": label,
                    "text": text,
                    "selected_token_index": token_index,
                    "selected_token": dict(trace["tokens"][token_index]),
                    "selected_span": (
                        {
                            "text": context["span_text"],
                            "token_start_index": span_start,
                            "token_end_index": span_end,
                            "token_indices": list(
                                range(span_start, span_end + 1)
                            ),
                            "token_ids": list(
                                span_token_sequences[
                                    len(context_payloads)
                                ]
                            ),
                            "token_count": span_end - span_start + 1,
                            "character_start": context[
                                "span_character_start"
                            ],
                            "character_end": context[
                                "span_character_end"
                            ],
                            "closing_token": dict(
                                trace["tokens"][token_index]
                            ),
                        }
                        if analysis_mode == "word_span"
                        else None
                    ),
                    "prefix_text": boundary["prefix_text"],
                    "prefix_token_ids": prefix_token_ids,
                    "prefix_token_count": len(prefix_token_ids),
                    "suffix_text": text[
                        (
                            end_character
                            if type(end_character) is int
                            else len(boundary["prefix_text"])
                        ):
                    ],
                    "tokens": [
                        {
                            "index": token["index"],
                            "token_id": token["token_id"],
                            "display": token["display"],
                            "source_text": token["source_text"],
                            "selected": (
                                span_start
                                <= int(token["index"])
                                <= span_end
                            ),
                            "closes_selected_span": (
                                int(token["index"]) == token_index
                            ),
                            "seen_by_selected_state": (
                                int(token["index"]) <= token_index
                            ),
                        }
                        for token in trace["tokens"]
                    ],
                    "layers": layers,
                    "forward_ms": forward_ms,
                }
            )

        pairs: list[dict[str, Any]] = []
        for left_index in range(len(trajectories)):
            for right_index in range(left_index + 1, len(trajectories)):
                points = []
                for layer, (left, right) in enumerate(
                    zip(
                        trajectories[left_index],
                        trajectories[right_index],
                        strict=True,
                    )
                ):
                    cosine = self._embedding_cosine(left, right)
                    points.append(
                        {
                            "layer": layer,
                            "relative_depth": layer / layer_count,
                            "cosine_similarity": cosine,
                            "cosine_distance": 1.0 - cosine,
                            "normalized_euclidean_distance": math.sqrt(
                                max(0.0, 2.0 - 2.0 * cosine)
                            ),
                        }
                    )
                max_point = max(
                    points,
                    key=lambda point: point["cosine_distance"],
                )
                pairs.append(
                    {
                        "left_index": left_index,
                        "right_index": right_index,
                        "label": (
                            f"{context_payloads[left_index]['label']} × "
                            f"{context_payloads[right_index]['label']}"
                        ),
                        "points": points,
                        "input_cosine_similarity": points[0][
                            "cosine_similarity"
                        ],
                        "input_cosine_distance": points[0][
                            "cosine_distance"
                        ],
                        "final_cosine_similarity": points[-1][
                            "cosine_similarity"
                        ],
                        "final_cosine_distance": points[-1][
                            "cosine_distance"
                        ],
                        "max_distance": max_point["cosine_distance"],
                        "max_distance_layer": max_point["layer"],
                    }
                )

        context_projection = project_context_states(trajectories)
        total_ms = (time.perf_counter() - total_started) * 1_000
        response = {
            "protocol": "context-trajectories-v2",
            "layer_count": layer_count,
            "analysis_kind": (
                "input_embedding_context_trajectories"
                if analysis_mode == "token"
                else "word_span_closing_position_context_trajectories"
            ),
            "analysis_mode": analysis_mode,
            "selected_token": selected_payload,
            "word_span": (
                {
                    "text": span_texts[0],
                    "token_ids": fragment_ids,
                    "token_count": len(fragment_ids),
                    "closing_token": selected_payload,
                    "fragments": fragment_payloads,
                    "representation_contract": (
                        "no_single_word_embedding_row; L0_is_the_closing_"
                        f"subtoken_row; L1_to_L{layer_count}_are_states_at_the_"
                        "word_closing_position"
                    ),
                }
                if analysis_mode == "word_span"
                else None
            ),
            "embedding": selected_embedding,
            "embedding_fragments": (
                fragment_payloads
                if analysis_mode == "word_span"
                else []
            ),
            "neighbors": neighbors,
            "projection": projection,
            "context_projection": context_projection,
            "contexts": context_payloads,
            "pairs": pairs,
            "tokenizer": self._tokenizer_identity_unlocked(),
            "interpretation": {
                "input_row": (
                    "Stejné tokenové ID vybralo tentýž výchozí řádek."
                    if analysis_mode == "token"
                    else (
                        "Celé slovo nemá jeden vstupní řádek. L0 zde "
                        "znamená pouze embedding posledního tokenového "
                        "střepu."
                    )
                ),
                "context": (
                    "Po L0 jsou stavy podmíněné pouze prefixem končícím "
                    "zvoleným tokenem."
                    if analysis_mode == "token"
                    else (
                        f"L1–L{layer_count} jsou stavy na pozici posledního střepu. "
                        "Díky kauzální attention už tato pozice může "
                        "čerpat ze všech předchozích střepů slova i z "
                        "levého kontextu."
                    )
                ),
                "word_span": (
                    "Slovní rozpětí je pozorovatelný znakový úsek a "
                    "shodná tokenová sekvence, nikoli nový vektor "
                    "vložený do modelu."
                    if analysis_mode == "word_span"
                    else None
                ),
                "geometry_warning": (
                    "Kosinová blízkost popisuje geometrii reprezentací; "
                    "sama nedokazuje totožnost významu ani kauzalitu."
                ),
                "future_tokens_used": False,
            },
            "execution": {
                "mode": (
                    "lookup_neighbors_and_cache_free_context_forwards"
                    if analysis_mode == "token"
                    else (
                        "word_span_fragment_lookup_closing_fragment_"
                        "neighbors_and_cache_free_context_forwards"
                    )
                ),
                "forward_passes": len(contexts),
                "generation": False,
                "probabilities": False,
                "future_suffix_used": False,
                "full_embedding_vector_returned": False,
                "vocabulary_scan": True,
            },
            "provenance": {
                "runtime": dict(RUNTIME_PROVENANCE),
                "model": self._model_provenance_snapshot_unlocked(),
            },
            "model_id": self.active_model_id,
            "model_label": self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
            "cached": False,
            "timing_ms": {
                "embedding_row": row_ms,
                "neighbor_scan": neighbor_ms,
                "context_forwards": forward_times,
                "total": total_ms,
            },
            "latency_ms": total_ms,
        }
        with self.cache_lock:
            self.embedding_cache[cache_key] = response
            self.embedding_cache.move_to_end(cache_key)
            while len(self.embedding_cache) > EMBEDDING_CACHE_SIZE:
                self.embedding_cache.popitem(last=False)
        return response

    def _text_surprisal_forward_logits(
        self,
        text: str,
        context_token_ids: list[int],
        target_token_ids: list[int],
    ) -> np.ndarray:
        """Run one cache-free forward over BOS plus prior ordinary tokens."""
        if self.mock:
            digest = hashlib.sha256(
                f"text-surprisal-v1\0{text}{self.active_model_id if self.active_model_id == 'mock-eurollm9' else ''}".encode("utf-8")
            ).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            logits = rng.normal(
                0.0,
                1.0,
                size=(len(context_token_ids), len(MOCK_WORDS)),
            )
            for index, token_id in enumerate(target_token_ids):
                logits[index, (token_id + index + 1) % len(MOCK_WORDS)] += 2.5
                logits[index, token_id] += 1.0
            return logits.astype(np.float32)

        return self.backend.full_logits(self.model, context_token_ids)

    def _text_surprisal_word_start_hint(
        self,
        text: str,
        piece: str,
        raw_token: str | None,
        offset: tuple[int, int] | None,
    ) -> bool | None:
        if (
            raw_token
            and len(raw_token) > 1
            and raw_token.startswith(("▁", "Ġ"))
        ):
            return True
        if piece and piece[0].isspace() and piece.strip():
            return True
        if offset is None:
            return None
        start, _end = offset
        visible_piece = piece.lstrip()
        if not visible_piece or not visible_piece[0].isalnum():
            return None
        if start == 0:
            return True
        if text[start - 1].isspace():
            return True
        if text[start - 1].isalnum():
            return False
        return None

    @staticmethod
    def _text_surprisal_span_definitions(
        text: str,
        unit: str,
    ) -> list[tuple[int, int]]:
        """Return transparent, non-linguistic spans over Python characters."""
        return surprisal_span_definitions(text, unit)

        if unit == "whitespace_span":
            return [
                (match.start(), match.end())
                for match in re.finditer(r"\S+", text)
            ]

        line_spans: list[tuple[int, int]] = []
        line_break_characters = frozenset(
            "\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"
        )
        cursor = 0
        for line_with_ending in text.splitlines(keepends=True):
            content_end = cursor + len(line_with_ending)
            while (
                content_end > cursor
                and text[content_end - 1] in line_break_characters
            ):
                content_end -= 1
            if text[cursor:content_end].strip():
                line_spans.append((cursor, content_end))
            cursor += len(line_with_ending)
        if cursor < len(text) and text[cursor:].strip():
            line_spans.append((cursor, len(text)))
        if unit == "line":
            return line_spans

        if unit != "sentence_like":
            raise ValueError(f"Neznámá agregační jednotka: {unit}.")

        terminal_characters = frozenset(
            ".!?…。｡！？‼⁇⁈⁉؟۔։।॥"
        )
        trailing_closers = frozenset(
            "\"'”’»›)]}）］｝"
        )
        sentence_spans: list[tuple[int, int]] = []
        for line_start, line_end in line_spans:
            candidate_start = line_start
            position = line_start
            while position < line_end:
                if text[position] not in terminal_characters:
                    position += 1
                    continue
                candidate_end = position + 1
                while (
                    candidate_end < line_end
                    and text[candidate_end] in terminal_characters
                ):
                    candidate_end += 1
                while (
                    candidate_end < line_end
                    and text[candidate_end] in trailing_closers
                ):
                    candidate_end += 1
                start = candidate_start
                end = candidate_end
                while start < end and text[start].isspace():
                    start += 1
                while end > start and text[end - 1].isspace():
                    end -= 1
                if start < end and text[start:end].strip():
                    sentence_spans.append((start, end))
                candidate_start = candidate_end
                position = candidate_end

            start = candidate_start
            end = line_end
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end and text[start:end].strip():
                sentence_spans.append((start, end))
        return sentence_spans

    @staticmethod
    def _text_surprisal_linear_quantile(
        values: list[float],
        quantile: float,
    ) -> float | None:
        """Use explicit (n - 1) * q linear interpolation."""
        return surprisal_linear_quantile(values, quantile)

        if not values:
            return None
        ordered = sorted(float(value) for value in values)
        position = (len(ordered) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return (
            ordered[lower] * (1.0 - weight)
            + ordered[upper] * weight
        )

    @classmethod
    def _text_surprisal_descriptive_statistics(
        cls,
        values: list[float],
    ) -> dict[str, float | int | None]:
        """Summarize complete span values without robustness rescaling."""
        return surprisal_descriptive_statistics(values)

        if not values:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "q1": None,
                "q3": None,
                "iqr": None,
                "raw_mad": None,
                "min": None,
                "max": None,
            }
        checked = [float(value) for value in values]
        if not all(math.isfinite(value) for value in checked):
            raise RuntimeError(
                "Agregace překvapení obsahuje nekonečnou hodnotu."
            )
        median = cls._text_surprisal_linear_quantile(checked, 0.5)
        q1 = cls._text_surprisal_linear_quantile(checked, 0.25)
        q3 = cls._text_surprisal_linear_quantile(checked, 0.75)
        assert median is not None
        assert q1 is not None
        assert q3 is not None
        raw_mad = cls._text_surprisal_linear_quantile(
            [abs(value - median) for value in checked],
            0.5,
        )
        return {
            "count": len(checked),
            "mean": math.fsum(checked) / len(checked),
            "median": median,
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "raw_mad": raw_mad,
            "min": min(checked),
            "max": max(checked),
        }

    @classmethod
    def _text_surprisal_aggregation(
        cls,
        text: str,
        tokens: list[dict[str, Any]],
        unit: str,
    ) -> dict[str, Any]:
        """Aggregate already-computed token surprise into heuristic spans."""
        return surprisal_aggregation(text, tokens, unit)

        span_definitions = cls._text_surprisal_span_definitions(text, unit)
        token_indices_by_span: list[list[int]] = [
            [] for _ in span_definitions
        ]
        ambiguous_indices_by_span: list[list[int]] = [
            [] for _ in span_definitions
        ]
        assigned_token_count = 0
        ambiguous_token_count = 0
        unassigned_token_count = 0

        for token in tokens:
            token_index = int(token["index"])
            start = token.get("character_start")
            end = token.get("character_end")
            reliable_offset = (
                token.get("piece_is_lossless") is True
                and type(start) is int
                and type(end) is int
                and 0 <= start < end <= len(text)
            )
            if not reliable_offset:
                ambiguous_token_count += 1
                continue
            overlapping_spans = [
                span_index
                for span_index, (span_start, span_end) in enumerate(
                    span_definitions
                )
                if max(start, span_start) < min(end, span_end)
            ]
            if len(overlapping_spans) == 1:
                token_indices_by_span[overlapping_spans[0]].append(
                    token_index
                )
                assigned_token_count += 1
            elif len(overlapping_spans) > 1:
                ambiguous_token_count += 1
                for span_index in overlapping_spans:
                    ambiguous_indices_by_span[span_index].append(
                        token_index
                    )
            else:
                unassigned_token_count += 1

        token_by_index = {
            int(token["index"]): token for token in tokens
        }
        spans: list[dict[str, Any]] = []
        complete_values: list[float] = []
        for span_index, (span_start, span_end) in enumerate(
            span_definitions
        ):
            token_indices = token_indices_by_span[span_index]
            span_tokens = [
                token_by_index[token_index]
                for token_index in token_indices
            ]
            span_text = text[span_start:span_end]
            character_count = len(span_text)
            non_whitespace_positions = {
                position
                for position in range(span_start, span_end)
                if not text[position].isspace()
            }
            covered_non_whitespace_positions: set[int] = set()
            for token in span_tokens:
                token_start = int(token["character_start"])
                token_end = int(token["character_end"])
                covered_non_whitespace_positions.update(
                    position
                    for position in range(
                        max(span_start, token_start),
                        min(span_end, token_end),
                    )
                    if not text[position].isspace()
                )
            non_whitespace_character_count = len(
                non_whitespace_positions
            )
            covered_non_whitespace_character_count = len(
                covered_non_whitespace_positions
            )
            character_coverage_ratio = (
                covered_non_whitespace_character_count
                / non_whitespace_character_count
                if non_whitespace_character_count
                else 0.0
            )
            complete = (
                bool(span_tokens)
                and not ambiguous_indices_by_span[span_index]
                and character_coverage_ratio == 1.0
            )
            if span_tokens:
                total_surprisal_bits = math.fsum(
                    float(token["surprisal_bits"])
                    for token in span_tokens
                )
                surprisal_bits_per_token = (
                    total_surprisal_bits / len(span_tokens)
                )
                surprisal_bits_per_character = (
                    total_surprisal_bits
                    / non_whitespace_character_count
                )
            else:
                total_surprisal_bits = None
                surprisal_bits_per_token = None
                surprisal_bits_per_character = None
            if complete and surprisal_bits_per_character is not None:
                complete_values.append(surprisal_bits_per_character)
            spans.append(
                {
                    "index": span_index,
                    "text": span_text,
                    "character_start": span_start,
                    "character_end": span_end,
                    "token_indices": token_indices,
                    "token_ids": [
                        int(token["token_id"]) for token in span_tokens
                    ],
                    "token_count": len(span_tokens),
                    "total_surprisal_bits": total_surprisal_bits,
                    "surprisal_bits_per_token": (
                        surprisal_bits_per_token
                    ),
                    "surprisal_bits_per_character": (
                        surprisal_bits_per_character
                    ),
                    "character_count": character_count,
                    "non_whitespace_character_count": (
                        non_whitespace_character_count
                    ),
                    "covered_non_whitespace_character_count": (
                        covered_non_whitespace_character_count
                    ),
                    "character_coverage_ratio": (
                        character_coverage_ratio
                    ),
                    "complete": complete,
                    "ambiguous_token_count": len(
                        ambiguous_indices_by_span[span_index]
                    ),
                    "ambiguous_token_indices": (
                        ambiguous_indices_by_span[span_index]
                    ),
                }
            )

        summary = cls._text_surprisal_descriptive_statistics(
            complete_values
        )
        summary.update(
            {
                "assigned_token_count": assigned_token_count,
                "ambiguous_token_count": ambiguous_token_count,
                "unassigned_token_count": unassigned_token_count,
                "total_token_count": len(tokens),
                "coverage_ratio": (
                    assigned_token_count / len(tokens)
                    if tokens
                    else 0.0
                ),
                "span_count": len(spans),
                "complete_span_count": sum(
                    bool(span["complete"]) for span in spans
                ),
                "incomplete_span_count": sum(
                    not bool(span["complete"]) for span in spans
                ),
            }
        )
        return {
            "unit": unit,
            "label_cs": {
                "whitespace_span": "úsek oddělený mezerami",
                "line": "neprázdný řádek",
                "sentence_like": "větně podobný úsek",
            }[unit],
            "spans": spans,
            "summary": summary,
        }

    @classmethod
    def _text_surprisal_aggregations(
        cls,
        text: str,
        tokens: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        return surprisal_aggregations(text, tokens)

        return {
            unit: cls._text_surprisal_aggregation(text, tokens, unit)
            for unit in (
                "whitespace_span",
                "line",
                "sentence_like",
            )
        }

    def _text_surprisal_local(
        self,
        original_text: str,
        top_k: int,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        if not isinstance(original_text, str):
            raise ValueError("Pole text musí být řetězec.")
        if len(original_text) > MAX_TEXT_SURPRISAL_CHARACTERS:
            raise ValueError(
                f"Text může mít nejvýše "
                f"{MAX_TEXT_SURPRISAL_CHARACTERS} znaků."
            )
        original_utf8_bytes = len(original_text.encode("utf-8"))
        if original_utf8_bytes > MAX_TEXT_SURPRISAL_UTF8_BYTES:
            raise ValueError(
                f"Text může mít nejvýše "
                f"{MAX_TEXT_SURPRISAL_UTF8_BYTES} UTF-8 bajtů."
            )
        if type(top_k) is not int or not 1 <= top_k <= MAX_TEXT_SURPRISAL_TOP_K:
            raise ValueError(
                f"top_k musí být celé číslo mezi 1 a "
                f"{MAX_TEXT_SURPRISAL_TOP_K}."
            )

        text = original_text.rstrip()
        if not text.strip():
            raise ValueError("Pole text musí obsahovat neprázdný text.")

        tokenization_started = time.perf_counter()
        (
            target_token_ids,
            context_token_ids,
            pieces,
            raw_tokens,
            offsets,
        ) = self._text_surprisal_tokenization(text)
        if not target_token_ids:
            raise ValueError(
                "Text po tokenizaci neobsahuje žádné běžné textové tokeny."
            )
        if len(target_token_ids) > MAX_TEXT_SURPRISAL_TOKENS:
            raise ValueError(
                f"Text má {len(target_token_ids)} textových tokenů; "
                f"limit je {MAX_TEXT_SURPRISAL_TOKENS}."
            )
        tokenization_ms = (
            time.perf_counter() - tokenization_started
        ) * 1_000

        forward_started = time.perf_counter()
        logits = np.asarray(
            self._text_surprisal_forward_logits(
                text,
                context_token_ids,
                target_token_ids,
            )
        )
        forward_ms = (time.perf_counter() - forward_started) * 1_000
        logits = validate_logit_matrix(
            logits,
            expected_rows=len(target_token_ids),
        )
        target_likelihoods = score_target_rows(
            logits,
            target_token_ids,
        )

        summary_started = time.perf_counter()
        tokens: list[dict[str, Any]] = []
        natural_log_of_two = math.log(2.0)
        special_ids = (
            set(getattr(self.tokenizer, "all_special_ids", []))
            if not self.mock
            else set()
        )
        for index, (
            token_id_value,
            piece,
            raw_token,
            offset,
            likelihood,
        ) in enumerate(
            zip(
                target_token_ids,
                pieces,
                raw_tokens,
                offsets,
                target_likelihoods,
                strict=True,
            )
        ):
            token_id = int(token_id_value)
            if not 0 <= token_id < logits.shape[1]:
                raise RuntimeError(
                    f"Token textu {token_id} leží mimo slovník "
                    f"o velikosti {logits.shape[1]}."
                )
            row = np.asarray(logits[index])
            log_probabilities = self._log_probabilities(row)
            if not np.all(np.isfinite(log_probabilities)):
                raise RuntimeError(
                    "Teacher-forced normalizace vrátila neplatné "
                    "log-pravděpodobnosti."
                )
            probabilities = np.exp(log_probabilities)
            logprob = likelihood.logprob
            probability = likelihood.probability
            surprisal_bits = likelihood.surprisal_bits
            entropy_bits = entropy_bits_from_probabilities(probabilities)
            (
                alternative_ids,
                alternative_probabilities,
                _other_probability,
                _entropy_bits,
                _effective_tokens,
                _normalized_entropy,
            ) = summarize_probabilities(probabilities, top_k)
            alternatives: list[dict[str, Any]] = []
            for alternative_id_value, alternative_probability in zip(
                alternative_ids,
                alternative_probabilities,
                strict=True,
            ):
                alternative_id = int(alternative_id_value)
                alternative = self._lens_token_payload(alternative_id)
                alternative["probability"] = float(
                    alternative_probability
                )
                alternative["rank"] = 1 + int(
                    np.count_nonzero(
                        row > row[alternative_id]
                    )
                )
                alternatives.append(alternative)

            piece_is_lossless = offset is not None
            display_piece = (
                raw_token
                if (
                    not piece_is_lossless
                    and raw_token
                    and ("\ufffd" in piece or not piece)
                )
                else piece
            )
            token: dict[str, Any] = {
                "index": index,
                "token_id": token_id,
                "text": piece,
                "piece": piece,
                "decoded_text": piece,
                "display": visible_token(
                    display_piece,
                    f"token {token_id}",
                ),
                "tokenizer_token": raw_token,
                "piece_is_lossless": piece_is_lossless,
                "is_special": token_id in special_ids,
                "probability": probability,
                "logprob": logprob,
                "surprisal_bits": surprisal_bits,
                "rank": likelihood.rank,
                "entropy_bits": entropy_bits,
                "effective_tokens": float(2**entropy_bits),
                "top1_gap_bits": max(
                    0.0,
                    float(
                        (
                            float(np.max(row))
                            - float(row[token_id])
                        )
                        / natural_log_of_two
                    ),
                ),
                "word_start_hint": self._text_surprisal_word_start_hint(
                    text,
                    piece,
                    raw_token,
                    offset,
                ),
                "alternatives": alternatives,
            }
            if offset is not None:
                token["character_start"] = offset[0]
                token["character_end"] = offset[1]
            else:
                token["character_start"] = None
                token["character_end"] = None
            tokens.append(token)

        total_surprisal_bits = math.fsum(
            token["surprisal_bits"] for token in tokens
        )
        mean_surprisal_bits = total_surprisal_bits / len(tokens)
        perplexity_exponent = mean_surprisal_bits * natural_log_of_two
        perplexity = (
            math.exp(perplexity_exponent)
            if perplexity_exponent
            <= math.log(float(np.finfo(np.float64).max))
            else None
        )
        mean_entropy_bits = math.fsum(
            token["entropy_bits"] for token in tokens
        ) / len(tokens)
        character_count = len(text)
        utf8_byte_count = len(text.encode("utf-8"))
        max_surprisal_token = max(
            tokens,
            key=lambda token: (
                token["surprisal_bits"],
                -token["index"],
            ),
        )
        finite_summary_values = (
            total_surprisal_bits,
            mean_surprisal_bits,
            mean_entropy_bits,
        )
        if not all(math.isfinite(value) for value in finite_summary_values):
            raise RuntimeError(
                "Teacher-forced souhrn překročil konečný číselný rozsah."
            )
        summary_ms = (time.perf_counter() - summary_started) * 1_000
        aggregation_started = time.perf_counter()
        aggregations = self._text_surprisal_aggregations(text, tokens)
        aggregation_ms = (
            time.perf_counter() - aggregation_started
        ) * 1_000
        total_ms = (time.perf_counter() - total_started) * 1_000

        return {
            "analysis_kind": "text_surprisal",
            "aggregation_method": {
                "id": "descriptive_text_span_aggregation",
                "version": "1.2.0",
                "status": "heuristic",
                "statement_cs": (
                    "Jde o popisnou agregaci již naměřeného "
                    "tokenového překvapení, nikoli o metriku "
                    "strukturovanosti, poetičnosti ani významu."
                ),
                "assignment_policy": (
                    "Token se spolehlivým znakovým rozsahem se přiřadí "
                    "právě tehdy, když protíná právě jeden neprázdný "
                    "úsek. Token přes více úseků je nejednoznačný; "
                    "token bez spolehlivého rozsahu se vynechá jako "
                    "nejednoznačný."
                ),
                "coverage_ratio": (
                    "assigned_token_count divided by all text tokens"
                ),
                "length_normalization": (
                    "Unicode code points excluding whitespace"
                ),
                "summary_population": (
                    "complete_spans_only_over_"
                    "surprisal_bits_per_character"
                ),
                "quantile_method": (
                    "linear_interpolation_at_(n_minus_1)_times_q"
                ),
                "mad_definition": (
                    "raw_median_absolute_deviation_without_1.4826_scaling"
                ),
                "units": {
                    "orthographic_word": (
                        "maximal Unicode word-like span starting with a "
                        "letter or number; continuing marks, apostrophes, "
                        "and hyphens are retained when internal. Word "
                        "surprisal is the sum of its token surprisals; "
                        "pre-word next-token entropy is the token-vocabulary "
                        "entropy before its first assigned token. Values are "
                        "omitted when token boundaries contaminate the word "
                        "span"
                    ),
                    "punctuation": (
                        "maximal cluster of Unicode punctuation outside an "
                        "orthographic word. It is displayed and exported as "
                        "a separate observed unit but excluded from word "
                        "summary statistics. Surprisal is the sum of its "
                        "assigned token surprisals; entropy is next-token "
                        "entropy immediately before its first token"
                    ),
                    "whitespace_span": (
                        "maximal non-empty regex span matched by \\S+; "
                        "punctuation remains attached"
                    ),
                    "line": (
                        "non-empty physical line; newline excluded from "
                        "the span and normalized character denominator"
                    ),
                    "sentence_like": (
                        "rule-based segment at .!? and common Unicode "
                        "terminal variants or a line boundary; not an "
                        "authoritative linguistic sentence"
                    ),
                },
            },
            "original_text": original_text,
            "normalized_text": text,
            "text": text,
            "normalization": {
                "policy": "trim_trailing_whitespace_only",
                "changed": text != original_text,
                "trimmed_trailing_characters": (
                    len(original_text) - len(text)
                ),
                "leading_whitespace_preserved": True,
            },
            "top_k": top_k,
            "temperature": 1.0,
            "probability_kind": "raw_model",
            "token_count": len(tokens),
            "token_ids": list(target_token_ids),
            "character_count": character_count,
            "utf8_byte_count": utf8_byte_count,
            "tokens": tokens,
            "aggregations": aggregations,
            "summary": {
                "total_surprisal_bits": total_surprisal_bits,
                "mean_surprisal_bits": mean_surprisal_bits,
                "perplexity": perplexity,
                "perplexity_overflowed": perplexity is None,
                "bits_per_character": (
                    total_surprisal_bits / character_count
                ),
                "bits_per_utf8_byte": (
                    total_surprisal_bits / utf8_byte_count
                ),
                "mean_entropy_bits": mean_entropy_bits,
                "max_surprisal_token": dict(max_surprisal_token),
            },
            "execution": {
                "mode": "teacher_forced",
                "forward_passes": 1,
                "generation": False,
                "steering": False,
                "worker": "single_serial_worker_job",
                "cache_policy": "request_local_cache_free_no_global_mutation",
                "alignment": (
                    "ordinary_token_i_scored_by_logits_at_"
                    "bos_plus_tokens_before_i"
                ),
            },
            "model": self.model_name,
            "model_id": self.active_model_id,
            "model_label": self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            "model_generation": self.model_generation,
            "transition_id": self.transition_id,
            "server_instance_id": self.server_instance_id,
            "provenance": {
                "server": {
                    "runtime": dict(RUNTIME_PROVENANCE),
                    "model": self._model_provenance_snapshot_unlocked(),
                }
            },
            "cached": False,
            "timing_ms": {
                "tokenization": tokenization_ms,
                "forward": forward_ms,
                "aggregation": aggregation_ms,
                "summary": summary_ms,
                "total": total_ms,
            },
            "latency_ms": total_ms,
        }

    def _continuation_surprisal_local(
        self,
        prefix_original: str,
        continuation_original: str,
        top_k: int,
    ) -> dict[str, Any]:
        """Score only a fixed continuation with a request-local causal KV cache.

        Unlike the full-text matrix forward, this path never places future
        continuation tokens in the input tensor.  It therefore also avoids
        sequence-length-dependent quantized kernels changing the already
        consumed prefix.
        """
        total_started = time.perf_counter()
        if not isinstance(prefix_original, str) or not prefix_original.strip():
            raise ValueError("prefix musí být neprázdný řetězec.")
        if (
            not isinstance(continuation_original, str)
            or not continuation_original.strip()
        ):
            raise ValueError("continuation musí být neprázdný řetězec.")
        if prefix_original != prefix_original.rstrip():
            raise ValueError(
                "prefix nesmí mít koncovou mezeru; explicitní hranice patří "
                "na začátek continuation."
            )
        if continuation_original != continuation_original.rstrip():
            raise ValueError("continuation nesmí mít koncovou mezeru.")
        full_text = prefix_original + continuation_original
        if len(full_text) > MAX_TEXT_SURPRISAL_CHARACTERS:
            raise ValueError(
                f"Prefix a pokračování mohou mít dohromady nejvýše "
                f"{MAX_TEXT_SURPRISAL_CHARACTERS} znaků."
            )
        if len(full_text.encode("utf-8")) > MAX_TEXT_SURPRISAL_UTF8_BYTES:
            raise ValueError(
                f"Prefix a pokračování mohou mít dohromady nejvýše "
                f"{MAX_TEXT_SURPRISAL_UTF8_BYTES} UTF-8 bajtů."
            )
        if type(top_k) is not int or not 1 <= top_k <= MAX_TEXT_SURPRISAL_TOP_K:
            raise ValueError(
                f"top_k musí být celé číslo mezi 1 a "
                f"{MAX_TEXT_SURPRISAL_TOP_K}."
            )

        tokenization_started = time.perf_counter()
        (
            prefix_token_ids,
            _prefix_context_ids,
            _prefix_pieces,
            _prefix_raw_tokens,
            _prefix_offsets,
        ) = self._text_surprisal_tokenization(prefix_original)
        (
            full_token_ids,
            _full_context_ids,
            full_pieces,
            full_raw_tokens,
            full_offsets,
        ) = self._text_surprisal_tokenization(full_text)
        if not prefix_token_ids:
            raise ValueError("Prefix po tokenizaci neobsahuje textový token.")
        if full_token_ids[: len(prefix_token_ids)] != prefix_token_ids:
            raise ValueError(
                "Tokenizace na hranici prefixu změnila jeho poslední token. "
                "Pokračování musí začínat explicitní mezerou."
            )
        continuation_token_ids = full_token_ids[len(prefix_token_ids) :]
        continuation_pieces = full_pieces[len(prefix_token_ids) :]
        continuation_raw_tokens = full_raw_tokens[len(prefix_token_ids) :]
        continuation_offsets = full_offsets[len(prefix_token_ids) :]
        if not continuation_token_ids:
            raise ValueError(
                "Pokračování po tokenizaci neobsahuje žádný nový token."
            )
        if len(full_token_ids) > MAX_TEXT_SURPRISAL_TOKENS:
            raise ValueError(
                f"Prefix a pokračování mají {len(full_token_ids)} tokenů; "
                f"limit je {MAX_TEXT_SURPRISAL_TOKENS}."
            )
        boundary = len(prefix_original)
        byte_alignment = None
        if not self.mock:
            byte_alignment = exact_byte_offsets(
                full_text, [{"tokenizer_token": raw} for raw in full_raw_tokens],
                self._tokenizer_identity_unlocked().get("algorithm"),
            )
        if byte_alignment:
            offset_source, full_byte_ranges = byte_alignment
            continuation_byte_ranges = full_byte_ranges[len(prefix_token_ids):]
            byte_boundary = len(prefix_original.encode("utf-8"))
            if (continuation_byte_ranges[0]["utf8_byte_start"] != byte_boundary
                    or any(span["utf8_byte_start"] < byte_boundary for span in continuation_byte_ranges)):
                raise ValueError("Pokračování nemá spolehlivou samostatnou tokenovou hranici.")
            continuation_offsets = [(span["character_start"], span["character_end"])
                                    for span in continuation_byte_ranges]
        else:
            offset_source = "response_character_offsets"
            continuation_byte_ranges = [None] * len(continuation_token_ids)
            for offset in continuation_offsets:
                if offset is None or offset[0] < boundary:
                    raise ValueError(
                        "Pokračování nemá spolehlivou samostatnou tokenovou hranici."
                    )
            if "".join(continuation_pieces) != continuation_original:
                raise RuntimeError(
                    "Tokenové kusy neskládají přesné zadané pokračování."
                )
        bos_token_id = (
            0
            if self.mock
            else getattr(self.tokenizer, "bos_token_id", None)
        )
        if type(bos_token_id) is not int or bos_token_id < 0:
            raise RuntimeError("Tokenizer nemá platný BOS token.")
        prompt_ids = [bos_token_id, *prefix_token_ids]
        tokenization_ms = (
            time.perf_counter() - tokenization_started
        ) * 1_000

        if self.mock:
            cache: Any = None
            consumed_ids = list(prompt_ids)

            def next_logits(input_ids: list[int]) -> np.ndarray:
                nonlocal consumed_ids
                if input_ids != prompt_ids:
                    consumed_ids.extend(input_ids)
                digest = hashlib.sha256(
                    (
                        "continuation-surprisal-v1\0"
                        + (self.active_model_id if self.active_model_id == "mock-eurollm9" else "")
                        + ",".join(str(value) for value in consumed_ids)
                    ).encode("utf-8")
                ).digest()
                rng = np.random.default_rng(
                    int.from_bytes(digest[:8], "big")
                )
                return rng.normal(
                    0.0,
                    1.0,
                    size=len(MOCK_WORDS),
                ).astype(np.float32)

        else:
            cache = self.backend.make_cache(self.model)

            def next_logits(input_ids: list[int]) -> np.ndarray:
                nonlocal cache
                last, cache = self.backend.last_logits(self.model, input_ids, cache)
                return last

        scoring_started = time.perf_counter()
        logits = next_logits(prompt_ids)
        if logits.ndim != 1 or not np.all(np.isfinite(logits)):
            raise RuntimeError("Inkrementální prompt vrátil neplatné logity.")
        prompt_logits_sha256 = hashlib.sha256(
            np.asarray(logits, dtype=np.float32).tobytes()
        ).hexdigest()
        natural_log_of_two = math.log(2.0)
        special_ids = (
            set(getattr(self.tokenizer, "all_special_ids", []))
            if not self.mock
            else set()
        )
        tokens: list[dict[str, Any]] = []
        cumulative_surprisal = 0.0
        for index, (
            token_id_value,
            piece,
            raw_token,
            offset,
        ) in enumerate(
            zip(
                continuation_token_ids,
                continuation_pieces,
                continuation_raw_tokens,
                continuation_offsets,
                strict=True,
            )
        ):
            token_id = int(token_id_value)
            if not 0 <= token_id < logits.size:
                raise RuntimeError(
                    f"Token pokračování {token_id} leží mimo slovník."
                )
            log_probabilities = self._log_probabilities(logits)
            probabilities = np.exp(log_probabilities)
            likelihood = token_likelihood(
                logits,
                token_id,
                out_of_range_label="Token pokračování",
            )
            logprob = likelihood.logprob
            probability = likelihood.probability
            surprisal_bits = likelihood.surprisal_bits
            cumulative_surprisal += surprisal_bits
            (
                alternative_ids,
                alternative_probabilities,
                _other_probability,
                entropy_bits,
                effective_tokens,
                _normalized_entropy,
            ) = summarize_probabilities(probabilities, top_k)
            alternatives: list[dict[str, Any]] = []
            for alternative_id_value, alternative_probability in zip(
                alternative_ids,
                alternative_probabilities,
                strict=True,
            ):
                alternative_id = int(alternative_id_value)
                alternative = self._lens_token_payload(alternative_id)
                alternative["probability"] = float(
                    alternative_probability
                )
                alternative["rank"] = 1 + int(
                    np.count_nonzero(logits > logits[alternative_id])
                )
                alternatives.append(alternative)
            token = {
                "index": index,
                "token_id": token_id,
                "text": piece,
                "piece": piece,
                "display": visible_token(
                    raw_token if raw_token and not piece else piece,
                    f"token {token_id}",
                ),
                "tokenizer_token": raw_token,
                "probability": probability,
                "logprob": logprob,
                "surprisal_bits": surprisal_bits,
                "cumulative_surprisal_bits": cumulative_surprisal,
                "rank": likelihood.rank,
                "entropy_bits": entropy_bits,
                "effective_tokens": effective_tokens,
                "is_special": token_id in special_ids,
                "alternatives": alternatives,
                "character_start": offset[0] if offset is not None else None,
                "character_end": offset[1] if offset is not None else None,
            }
            span = continuation_byte_ranges[index]
            if span is not None:
                token["utf8_byte_start"] = span["utf8_byte_start"]
                token["utf8_byte_end"] = span["utf8_byte_end"]
            tokens.append(token)
            if index + 1 < len(continuation_token_ids):
                logits = next_logits([token_id])
                if logits.ndim != 1 or not np.all(np.isfinite(logits)):
                    raise RuntimeError(
                        "Inkrementální krok vrátil neplatné logity."
                    )
        scoring_ms = (time.perf_counter() - scoring_started) * 1_000
        total_surprisal_bits = math.fsum(
            float(token["surprisal_bits"]) for token in tokens
        )
        total_ms = (time.perf_counter() - total_started) * 1_000
        return {
            "analysis_kind": "continuation_surprisal",
            "prefix": prefix_original,
            "continuation": continuation_original,
            "full_text": full_text,
            "top_k": top_k,
            "temperature": 1.0,
            "probability_kind": "raw_model",
            "prefix_token_ids": list(prefix_token_ids),
            "continuation_token_ids": list(continuation_token_ids),
            "continuation_token_count": len(continuation_token_ids),
            "prompt_logits_sha256": prompt_logits_sha256,
            "tokens": tokens,
            "summary": {
                "total_continuation_surprisal_bits": total_surprisal_bits,
                "mean_continuation_surprisal_bits": (
                    total_surprisal_bits / len(tokens)
                ),
            },
            "execution": {
                "mode": "teacher_forced_incremental_kv",
                "token_offset_source": offset_source,
                "generation": False,
                "steering": False,
                "future_continuation_tokens_in_forward": False,
                "prompt_forward_count": 1,
                "incremental_token_forward_count": max(0, len(tokens) - 1),
                "worker": "single_serial_worker_job",
                "cache_policy": (
                    "request_local_causal_kv_no_global_mutation"
                ),
            },
            "model": self.model_name,
            "model_id": self.active_model_id,
            "model_label": self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            "model_generation": self.model_generation,
            "transition_id": self.transition_id,
            "server_instance_id": self.server_instance_id,
            "provenance": {
                "server": {
                    "runtime": dict(RUNTIME_PROVENANCE),
                    "model": self._model_provenance_snapshot_unlocked(),
                }
            },
            "cached": False,
            "timing_ms": {
                "tokenization": tokenization_ms,
                "scoring": scoring_ms,
                "total": total_ms,
            },
            "latency_ms": total_ms,
        }

    @staticmethod
    def _text_surprisal_comparison_metric(
        value_a: float | int | None,
        value_b: float | int | None,
        unit: str,
    ) -> dict[str, float | str | None]:
        return surprisal_comparison_metric(value_a, value_b, unit)

        checked_a = float(value_a) if value_a is not None else None
        checked_b = float(value_b) if value_b is not None else None
        for value in (checked_a, checked_b):
            if value is not None and not math.isfinite(value):
                raise RuntimeError(
                    "Párové srovnání překvapení obsahuje nekonečnou "
                    "hodnotu."
                )
        delta = (
            checked_b - checked_a
            if checked_a is not None and checked_b is not None
            else None
        )
        if delta is not None and not math.isfinite(delta):
            raise RuntimeError(
                "Rozdíl párového srovnání překvapení překročil "
                "konečný číselný rozsah."
            )
        return {
            "a": checked_a,
            "b": checked_b,
            "delta_b_minus_a": delta,
            "unit": unit,
        }

    @classmethod
    def _text_surprisal_comparison_metrics(
        cls,
        result_a: dict[str, Any],
        result_b: dict[str, Any],
    ) -> dict[str, dict[str, float | str | None]]:
        return surprisal_comparison_metrics(result_a, result_b)

        def non_whitespace_character_count(
            result: dict[str, Any],
        ) -> int:
            return sum(
                not character.isspace()
                for character in str(result["normalized_text"])
            )

        def bits_per_non_whitespace_character(
            result: dict[str, Any],
        ) -> float | None:
            character_count = non_whitespace_character_count(result)
            if character_count == 0:
                return None
            return (
                float(result["summary"]["total_surprisal_bits"])
                / character_count
            )

        def whitespace_span_summary(
            result: dict[str, Any],
        ) -> dict[str, Any]:
            return result["aggregations"]["whitespace_span"]["summary"]

        summary_a = result_a["summary"]
        summary_b = result_b["summary"]
        spans_a = whitespace_span_summary(result_a)
        spans_b = whitespace_span_summary(result_b)
        per_token_unit = "bits_per_token"
        per_character_unit = "bits_per_non_whitespace_character"
        return {
            "mean_surprisal_bits": cls._text_surprisal_comparison_metric(
                summary_a["mean_surprisal_bits"],
                summary_b["mean_surprisal_bits"],
                per_token_unit,
            ),
            "bits_per_non_whitespace_character": (
                cls._text_surprisal_comparison_metric(
                    bits_per_non_whitespace_character(result_a),
                    bits_per_non_whitespace_character(result_b),
                    per_character_unit,
                )
            ),
            "median_whitespace_span_bits_per_non_whitespace_character": (
                cls._text_surprisal_comparison_metric(
                    spans_a["median"],
                    spans_b["median"],
                    per_character_unit,
                )
            ),
            "whitespace_span_iqr_bits_per_non_whitespace_character": (
                cls._text_surprisal_comparison_metric(
                    spans_a["iqr"],
                    spans_b["iqr"],
                    per_character_unit,
                )
            ),
            "whitespace_span_raw_mad_bits_per_non_whitespace_character": (
                cls._text_surprisal_comparison_metric(
                    spans_a["raw_mad"],
                    spans_b["raw_mad"],
                    per_character_unit,
                )
            ),
        }

    def _text_surprisal_comparison_local(
        self,
        text_a: str,
        text_b: str,
        label_a: str,
        label_b: str,
        top_k: int,
    ) -> dict[str, Any]:
        """Score two independent fixed texts serially without alignment."""
        total_started = time.perf_counter()
        result_a = self._text_surprisal_local(text_a, top_k)
        result_b = self._text_surprisal_local(text_b, top_k)
        metrics = self._text_surprisal_comparison_metrics(
            result_a,
            result_b,
        )
        whitespace_a = result_a["aggregations"]["whitespace_span"][
            "summary"
        ]
        whitespace_b = result_b["aggregations"]["whitespace_span"][
            "summary"
        ]
        labels = {"a": label_a, "b": label_b}
        total_ms = (time.perf_counter() - total_started) * 1_000
        return {
            "analysis_kind": "text_surprisal_comparison",
            "a": result_a,
            "b": result_b,
            "labels": labels,
            "comparison": {
                "delta_definition": "B - A",
                "labels": labels,
                "metrics": metrics,
                "diagnostics": {
                    "token_count": {
                        "a": int(result_a["token_count"]),
                        "b": int(result_b["token_count"]),
                    },
                    "whitespace_span_coverage_ratio": {
                        "a": float(whitespace_a["coverage_ratio"]),
                        "b": float(whitespace_b["coverage_ratio"]),
                    },
                },
                "method": {
                    "id": "paired_descriptive_text_surprisal",
                    "version": "1.0.0",
                    "status": "descriptive",
                    "delta_definition": "B - A",
                    "same_model_and_settings": True,
                    "token_alignment": False,
                    "statement_cs": (
                        "Jde o popisné srovnání normalizovaných veličin "
                        "dvou nezávislých textů, nikoli o úsudek, zda jsou "
                        "texty srovnatelné. Není to klasifikace žánru, "
                        "poetičnosti, strukturovanosti ani kvality."
                    ),
                },
            },
            "model": self.model_name,
            "model_id": self.active_model_id,
            "model_label": self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            "model_generation": self.model_generation,
            "transition_id": self.transition_id,
            "server_instance_id": self.server_instance_id,
            "provenance": result_a["provenance"],
            "execution": {
                "mode": "independent_teacher_forced_pair",
                "worker": "single_serial_worker_job",
                "forward_passes": 2,
                "generation": False,
                "steering": False,
                "token_alignment": False,
                "partial_success": False,
                "order": "A_then_B",
                "cache_policy": (
                    "two_independent_request_local_cache_free_forwards_"
                    "no_global_mutation"
                ),
            },
            "cached": False,
            "timing_ms": {
                "a": float(result_a["timing_ms"]["total"]),
                "b": float(result_b["timing_ms"]["total"]),
                "total": total_ms,
            },
            "latency_ms": total_ms,
        }

    def _prepare_context_contrast_inputs(
        self,
        prefix_a_original: str,
        prefix_b_original: str,
        continuation_original: str,
    ) -> dict[str, Any]:
        """Prepare two exact prefixes with one boundary-stable continuation."""
        for field_name, value in (
            ("prefix_a", prefix_a_original),
            ("prefix_b", prefix_b_original),
            ("continuation", continuation_original),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Pole {field_name} musí být neprázdný řetězec."
                )
            if len(value) > MAX_CONTEXT_CONTRAST_CHARACTERS:
                raise ValueError(
                    f"Pole {field_name} může mít nejvýše "
                    f"{MAX_CONTEXT_CONTRAST_CHARACTERS} znaků."
                )
            if len(value.encode("utf-8")) > MAX_CONTEXT_CONTRAST_UTF8_BYTES:
                raise ValueError(
                    f"Pole {field_name} může mít nejvýše "
                    f"{MAX_CONTEXT_CONTRAST_UTF8_BYTES} UTF-8 bajtů."
                )

        prefix_a = prefix_a_original.rstrip()
        prefix_b = prefix_b_original.rstrip()
        continuation = continuation_original.rstrip()
        if not prefix_a.strip() or not prefix_b.strip():
            raise ValueError("Oba prefixy musí po oříznutí zůstat neprázdné.")
        if not continuation.strip():
            raise ValueError(
                "Pokračování musí po oříznutí zůstat neprázdné."
            )

        if self.mock:
            def mock_pieces(text: str) -> list[str]:
                return re.findall(r"\s*\S+", text)

            def mock_ids(pieces: list[str]) -> list[int]:
                return [
                    int.from_bytes(
                        hashlib.sha256(piece.encode("utf-8")).digest()[:4],
                        "big",
                    )
                    % len(MOCK_WORDS)
                    for piece in pieces
                ]

            prefix_pieces_a = mock_pieces(prefix_a)
            prefix_pieces_b = mock_pieces(prefix_b)
            continuation_pieces = mock_pieces(continuation)
            prefix_ids_a = [0, *mock_ids(prefix_pieces_a)]
            prefix_ids_b = [0, *mock_ids(prefix_pieces_b)]
            continuation_ids = mock_ids(continuation_pieces)
            full_ids_a = [*prefix_ids_a, *continuation_ids]
            full_ids_b = [*prefix_ids_b, *continuation_ids]
        else:
            prefix_ids_a = encode_with_bos(self.tokenizer, prefix_a)
            prefix_ids_b = encode_with_bos(self.tokenizer, prefix_b)
            full_ids_a = encode_with_bos(
                self.tokenizer,
                prefix_a + continuation,
            )
            full_ids_b = encode_with_bos(
                self.tokenizer,
                prefix_b + continuation,
            )
            if full_ids_a[: len(prefix_ids_a)] != prefix_ids_a:
                raise ValueError(
                    "Tokenizace na hranici prefixu A a pokračování změnila "
                    "poslední token prefixu. Přidej na začátek pokračování "
                    "mezeru nebo jinou explicitní hranici."
                )
            if full_ids_b[: len(prefix_ids_b)] != prefix_ids_b:
                raise ValueError(
                    "Tokenizace na hranici prefixu B a pokračování změnila "
                    "poslední token prefixu. Přidej na začátek pokračování "
                    "mezeru nebo jinou explicitní hranici."
                )
            continuation_ids_a = full_ids_a[len(prefix_ids_a) :]
            continuation_ids_b = full_ids_b[len(prefix_ids_b) :]
            if continuation_ids_a != continuation_ids_b:
                raise ValueError(
                    "Pokračování nemá po obou prefixech stejné tokeny. "
                    "Zvol explicitnější hranici; bez shodných token_id by "
                    "trajektorie nebyly přímo porovnatelné."
                )
            continuation_ids = continuation_ids_a

            def pieces_for_ids(
                initial_ids: list[int],
                token_ids: list[int],
            ) -> list[str]:
                running_ids = list(initial_ids)
                pieces: list[str] = []
                for token_id in token_ids:
                    pieces.append(
                        self._trajectory_token_piece(running_ids, token_id)
                    )
                    running_ids.append(token_id)
                return pieces

            prefix_pieces_a = pieces_for_ids(
                prefix_ids_a[:1],
                prefix_ids_a[1:],
            )
            prefix_pieces_b = pieces_for_ids(
                prefix_ids_b[:1],
                prefix_ids_b[1:],
            )
            continuation_pieces = pieces_for_ids(
                prefix_ids_a,
                continuation_ids,
            )

        prefix_count_a = len(prefix_ids_a) - 1
        prefix_count_b = len(prefix_ids_b) - 1
        if prefix_count_a > MAX_CONTEXT_CONTRAST_PREFIX_TOKENS:
            raise ValueError(
                f"Prefix A má {prefix_count_a} textových tokenů; limit je "
                f"{MAX_CONTEXT_CONTRAST_PREFIX_TOKENS}."
            )
        if prefix_count_b > MAX_CONTEXT_CONTRAST_PREFIX_TOKENS:
            raise ValueError(
                f"Prefix B má {prefix_count_b} textových tokenů; limit je "
                f"{MAX_CONTEXT_CONTRAST_PREFIX_TOKENS}."
            )
        if not continuation_ids:
            raise ValueError(
                "Pokračování po tokenizaci neobsahuje žádný nový token."
            )
        if (
            len(continuation_ids)
            > MAX_CONTEXT_CONTRAST_CONTINUATION_TOKENS
        ):
            raise ValueError(
                f"Pokračování má {len(continuation_ids)} tokenů; limit je "
                f"{MAX_CONTEXT_CONTRAST_CONTINUATION_TOKENS}."
            )
        if not (
            len(prefix_pieces_a) == prefix_count_a
            and len(prefix_pieces_b) == prefix_count_b
            and len(continuation_pieces) == len(continuation_ids)
        ):
            raise RuntimeError(
                "Tokenové texty kontextového kontrastu nejsou zarovnané."
            )

        return {
            "prefix_a_original": prefix_a_original,
            "prefix_b_original": prefix_b_original,
            "continuation_original": continuation_original,
            "prefix_a": prefix_a,
            "prefix_b": prefix_b,
            "continuation": continuation,
            "prefix_ids_a": prefix_ids_a,
            "prefix_ids_b": prefix_ids_b,
            "full_ids_a": full_ids_a,
            "full_ids_b": full_ids_b,
            "continuation_ids": continuation_ids,
            "prefix_pieces_a": prefix_pieces_a,
            "prefix_pieces_b": prefix_pieces_b,
            "continuation_pieces": continuation_pieces,
            "normalization": {
                "policy": "trim_trailing_whitespace_only",
                "prefix_a_changed": prefix_a != prefix_a_original,
                "prefix_b_changed": prefix_b != prefix_b_original,
                "continuation_changed": (
                    continuation != continuation_original
                ),
                "leading_continuation_whitespace_preserved": True,
            },
        }

    def _context_contrast_forward_logits(
        self,
        text: str,
        full_token_ids: list[int],
    ) -> np.ndarray:
        """Return a causal logit row after every consumed token, including last."""
        logits = np.asarray(
            self._text_surprisal_forward_logits(
                text,
                full_token_ids,
                full_token_ids,
            )
        )
        return validate_logit_matrix(
            logits,
            expected_rows=len(full_token_ids),
            error_message=(
                "Kontextový forward vrátil neplatnou matici "
                "pozice × slovník."
            ),
        )

    @staticmethod
    def _context_contrast_token_logprob(
        logits: np.ndarray,
        token_id: int,
    ) -> tuple[float, float, int]:
        likelihood = token_likelihood(
            logits,
            token_id,
            out_of_range_label="Cílový token",
        )
        return (
            likelihood.logprob,
            likelihood.probability,
            likelihood.rank,
        )

    def _context_contrast_top_tokens(
        self,
        logits: np.ndarray,
        probabilities: np.ndarray,
        top_k: int,
    ) -> list[dict[str, Any]]:
        token_ids = np.argsort(-probabilities, kind="stable")[:top_k]
        result: list[dict[str, Any]] = []
        for rank, token_id_value in enumerate(token_ids, start=1):
            token_id = int(token_id_value)
            result.append(
                {
                    **self._lens_token_payload(token_id),
                    "probability": float(probabilities[token_id]),
                    "rank": rank,
                    "logit": float(logits[token_id]),
                }
            )
        return result

    def _context_contrast_access_curve(
        self,
        *,
        logits: np.ndarray,
        prefix_ids: list[int],
        prefix_pieces: list[str],
        target_token_id: int,
    ) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        natural_log_of_two = math.log(2.0)
        for consumed_count in range(len(prefix_ids)):
            logprob, probability, rank = (
                self._context_contrast_token_logprob(
                    np.asarray(logits[consumed_count]),
                    target_token_id,
                )
            )
            if consumed_count == 0:
                consumed_token = {
                    "token_id": int(prefix_ids[0]),
                    "text": "",
                    "display": "⟨BOS⟩",
                    "is_bos": True,
                }
                context_text = ""
            else:
                token_id = int(prefix_ids[consumed_count])
                piece = prefix_pieces[consumed_count - 1]
                consumed_token = {
                    "token_id": token_id,
                    "text": piece,
                    "display": visible_token(
                        piece,
                        f"token {token_id}",
                    ),
                    "is_bos": False,
                }
                if self.mock:
                    context_text = "".join(
                        prefix_pieces[:consumed_count]
                    )
                else:
                    context_text = safe_decode(
                        self.tokenizer,
                        prefix_ids[: consumed_count + 1],
                        skip_special=True,
                    )
            points.append(
                {
                    "consumed_prefix_token_count": consumed_count,
                    "context_text": context_text,
                    "last_consumed_token": consumed_token,
                    "target_token_id": target_token_id,
                    "probability": probability,
                    "log2_probability": logprob / natural_log_of_two,
                    "surprisal_bits": -logprob / natural_log_of_two,
                    "rank": rank,
                }
            )
        return points

    def _context_contrast_local(
        self,
        prefix_a_original: str,
        prefix_b_original: str,
        continuation_original: str,
        top_k: int,
    ) -> dict[str, Any]:
        """Measure one fixed continuation under two exact causal contexts."""
        total_started = time.perf_counter()
        if type(top_k) is not int or not (
            3 <= top_k <= MAX_CONTEXT_CONTRAST_TOP_K
        ):
            raise ValueError(
                "top_k musí být celé číslo mezi 3 a "
                f"{MAX_CONTEXT_CONTRAST_TOP_K}."
            )
        prepared = self._prepare_context_contrast_inputs(
            prefix_a_original,
            prefix_b_original,
            continuation_original,
        )
        cache_key = (
            self.model_generation,
            self.active_model_id,
            prepared["prefix_a"],
            prepared["prefix_b"],
            prepared["continuation"],
            top_k,
        )
        with self.cache_lock:
            cached_response = self.context_contrast_cache.get(cache_key)
            if cached_response is not None:
                self.context_contrast_cache.move_to_end(cache_key)
                response = json.loads(
                    json.dumps(
                        cached_response,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
                elapsed_ms = (
                    time.perf_counter() - total_started
                ) * 1_000
                response["cached"] = True
                response["latency_ms"] = elapsed_ms
                response["timing_ms"]["cache_lookup"] = elapsed_ms
                return response

        forward_a_started = time.perf_counter()
        logits_a = self._context_contrast_forward_logits(
            prepared["prefix_a"] + prepared["continuation"],
            prepared["full_ids_a"],
        )
        forward_a_ms = (
            time.perf_counter() - forward_a_started
        ) * 1_000
        forward_b_started = time.perf_counter()
        logits_b = self._context_contrast_forward_logits(
            prepared["prefix_b"] + prepared["continuation"],
            prepared["full_ids_b"],
        )
        forward_b_ms = (
            time.perf_counter() - forward_b_started
        ) * 1_000
        if logits_a.shape[1] != logits_b.shape[1]:
            raise RuntimeError(
                "Oba forwardy vrátily rozdílnou velikost slovníku."
            )

        continuation_ids = list(prepared["continuation_ids"])
        continuation_pieces = list(prepared["continuation_pieces"])
        prefix_ids_a = list(prepared["prefix_ids_a"])
        prefix_ids_b = list(prepared["prefix_ids_b"])
        prefix_count_a = len(prefix_ids_a) - 1
        prefix_count_b = len(prefix_ids_b) - 1
        target_token_id = int(continuation_ids[0])
        natural_log_of_two = math.log(2.0)

        access_a = self._context_contrast_access_curve(
            logits=logits_a,
            prefix_ids=prefix_ids_a,
            prefix_pieces=list(prepared["prefix_pieces_a"]),
            target_token_id=target_token_id,
        )
        access_b = self._context_contrast_access_curve(
            logits=logits_b,
            prefix_ids=prefix_ids_b,
            prefix_pieces=list(prepared["prefix_pieces_b"]),
            target_token_id=target_token_id,
        )

        persistence: list[dict[str, Any]] = []
        cumulative_information_advantage = 0.0
        sequence_log2_probability_a = 0.0
        sequence_log2_probability_b = 0.0
        root_probabilities_a: np.ndarray | None = None
        root_probabilities_b: np.ndarray | None = None
        root_logits_a: np.ndarray | None = None
        root_logits_b: np.ndarray | None = None
        for consumed_count in range(len(continuation_ids) + 1):
            row_a = np.asarray(logits_a[prefix_count_a + consumed_count])
            row_b = np.asarray(logits_b[prefix_count_b + consumed_count])
            probabilities_a = probabilities_at_temperature(row_a, 1.0)
            probabilities_b = probabilities_at_temperature(row_b, 1.0)
            js_bits, total_variation = distribution_distance(
                probabilities_a,
                probabilities_b,
            )
            state: dict[str, Any] = {
                "consumed_continuation_token_count": consumed_count,
                "consumed_text": "".join(
                    continuation_pieces[:consumed_count]
                ),
                "js_divergence_bits": js_bits,
                "total_variation": total_variation,
                "entropy_bits_a": entropy_bits_from_probabilities(
                    probabilities_a
                ),
                "entropy_bits_b": entropy_bits_from_probabilities(
                    probabilities_b
                ),
                "cumulative_information_advantage_a_over_b_bits": (
                    cumulative_information_advantage
                ),
            }
            if consumed_count == 0:
                root_probabilities_a = probabilities_a
                root_probabilities_b = probabilities_b
                root_logits_a = row_a
                root_logits_b = row_b
            if consumed_count < len(continuation_ids):
                token_id = int(continuation_ids[consumed_count])
                token_piece = continuation_pieces[consumed_count]
                logprob_a, probability_a, rank_a = (
                    self._context_contrast_token_logprob(row_a, token_id)
                )
                logprob_b, probability_b, rank_b = (
                    self._context_contrast_token_logprob(row_b, token_id)
                )
                log2_probability_a = logprob_a / natural_log_of_two
                log2_probability_b = logprob_b / natural_log_of_two
                token_advantage = (
                    log2_probability_a - log2_probability_b
                )
                cumulative_information_advantage += token_advantage
                sequence_log2_probability_a += log2_probability_a
                sequence_log2_probability_b += log2_probability_b
                state["next_token"] = {
                    "token_id": token_id,
                    "text": token_piece,
                    "display": visible_token(
                        token_piece,
                        f"token {token_id}",
                    ),
                    "probability_a": probability_a,
                    "probability_b": probability_b,
                    "log2_probability_a": log2_probability_a,
                    "log2_probability_b": log2_probability_b,
                    "surprisal_bits_a": -log2_probability_a,
                    "surprisal_bits_b": -log2_probability_b,
                    "rank_a": rank_a,
                    "rank_b": rank_b,
                    "information_advantage_a_over_b_bits": (
                        token_advantage
                    ),
                }
                state[
                    "cumulative_information_advantage_after_token_bits"
                ] = cumulative_information_advantage
            else:
                state["next_token"] = None
                state[
                    "cumulative_information_advantage_after_token_bits"
                ] = cumulative_information_advantage
            persistence.append(state)

        if (
            root_probabilities_a is None
            or root_probabilities_b is None
            or root_logits_a is None
            or root_logits_b is None
        ):
            raise RuntimeError("Kořenové distribuce nebyly vypočteny.")

        information_advantage = (
            sequence_log2_probability_a - sequence_log2_probability_b
        )
        ratio_a_over_b, ratio_status = (
            probability_ratio_from_log2_advantage(
                information_advantage
            )
        )

        root_js = float(persistence[0]["js_divergence_bits"])
        js_values = [
            float(point["js_divergence_bits"])
            for point in persistence
        ]
        persistent_half_life_step = persistent_half_life(js_values)
        half_threshold = root_js / 2.0
        js_area = trapezoid_area(js_values)
        token_effects = [
            float(point["next_token"][
                "information_advantage_a_over_b_bits"
            ])
            for point in persistence
            if point["next_token"] is not None
        ]

        target_tokens = [
            {
                "index": index,
                "token_id": int(token_id),
                "text": piece,
                "display": visible_token(
                    piece,
                    f"token {token_id}",
                ),
            }
            for index, (token_id, piece) in enumerate(
                zip(
                    continuation_ids,
                    continuation_pieces,
                    strict=True,
                )
            )
        ]
        warnings: list[dict[str, str]] = [
            {
                "code": "first_token_access_curve",
                "message": (
                    "Křivka zpřístupňování sleduje první token cílového "
                    "pokračování. Pravděpodobnost celého pokračování se "
                    "počítá pouze za úplnými prefixy."
                ),
            },
            {
                "code": "descriptive_context_contrast",
                "message": (
                    "Kontrast popisuje dvě podmíněné jazykové distribuce; "
                    "není pravděpodobností faktu ani kauzálním zásahem."
                ),
            },
        ]
        if self.is_quantized():
            warnings.append(
                {
                    "code": "quantized_model",
                    "message": (
                        "Výsledek pochází z kvantovaného nebo hybridního "
                        "modelu; těsné rozdíly mohou být numericky citlivé."
                    ),
                }
            )

        total_ms = (time.perf_counter() - total_started) * 1_000
        response = {
            "analysis_kind": "context_contrast",
            "definition": {
                "information_advantage": (
                    "log2 P(continuation|A) - "
                    "log2 P(continuation|B)"
                ),
                "equivalent_surprisal_difference": "I_B - I_A",
                "positive_direction": (
                    "positive means A makes the exact continuation "
                    "more probable than B"
                ),
                "field_distance": (
                    "Jensen-Shannon divergence over the full vocabulary "
                    "at temperature 1"
                ),
                "persistence_alignment": (
                    "same continuation token_ids teacher-forced after "
                    "both exact prefixes"
                ),
            },
            "inputs": {
                "prefix_a": prepared["prefix_a"],
                "prefix_b": prepared["prefix_b"],
                "continuation": prepared["continuation"],
                "normalization": prepared["normalization"],
            },
            "tokenization": {
                "prefix_a_token_ids": prefix_ids_a,
                "prefix_b_token_ids": prefix_ids_b,
                "prefix_a_token_count": prefix_count_a,
                "prefix_b_token_count": prefix_count_b,
                "continuation_token_ids": continuation_ids,
                "continuation_token_count": len(continuation_ids),
                "continuation_tokens": target_tokens,
                "boundary_policy": (
                    "exact_prefix_suffix_and_identical_continuation_"
                    "token_ids_across_contexts"
                ),
            },
            "target_score": {
                "continuation": prepared["continuation"],
                "token_count": len(continuation_ids),
                "log2_probability_a": sequence_log2_probability_a,
                "log2_probability_b": sequence_log2_probability_b,
                "information_advantage_a_over_b_bits": (
                    information_advantage
                ),
                "mean_information_advantage_bits_per_token": (
                    information_advantage / len(continuation_ids)
                ),
                "probability_ratio_a_over_b": ratio_a_over_b,
                "probability_ratio_status": ratio_status,
                "favored_context": (
                    "a"
                    if information_advantage > 0.0
                    else "b"
                    if information_advantage < 0.0
                    else "equal"
                ),
            },
            "root_field": {
                "js_divergence_bits": root_js,
                "total_variation": float(
                    persistence[0]["total_variation"]
                ),
                "entropy_bits_a": float(
                    persistence[0]["entropy_bits_a"]
                ),
                "entropy_bits_b": float(
                    persistence[0]["entropy_bits_b"]
                ),
                "top_tokens_a": self._context_contrast_top_tokens(
                    root_logits_a,
                    root_probabilities_a,
                    top_k,
                ),
                "top_tokens_b": self._context_contrast_top_tokens(
                    root_logits_b,
                    root_probabilities_b,
                    top_k,
                ),
            },
            "access_curve": {
                "target_token": target_tokens[0],
                "scope": "first_continuation_token_only",
                "a": access_a,
                "b": access_b,
            },
            "persistence": {
                "states": persistence,
                "summary": {
                    "root_js_divergence_bits": root_js,
                    "final_js_divergence_bits": float(
                        persistence[-1]["js_divergence_bits"]
                    ),
                    "maximum_js_divergence_bits": max(
                        float(point["js_divergence_bits"])
                        for point in persistence
                    ),
                    "minimum_js_divergence_bits": min(
                        float(point["js_divergence_bits"])
                        for point in persistence
                    ),
                    "js_area_bit_token_steps": js_area,
                    "persistent_half_life_step": (
                        persistent_half_life_step
                    ),
                    "persistent_half_life_threshold_bits": (
                        half_threshold if root_js > 0.0 else None
                    ),
                    "half_life_definition": (
                        "first consumed-token count after which all "
                        "observed JS values remain at or below half "
                        "the root JS"
                    ),
                    "largest_absolute_token_advantage_bits": max(
                        (abs(value) for value in token_effects),
                        default=0.0,
                    ),
                    "final_cumulative_information_advantage_bits": (
                        cumulative_information_advantage
                    ),
                },
            },
            "temperature": 1.0,
            "probability_kind": "raw_model",
            "warnings": warnings,
            "execution": {
                "mode": "paired_context_teacher_forcing",
                "forward_passes": 2,
                "generation": False,
                "steering": False,
                "worker": "single_serial_worker_job",
                "partial_success": False,
                "cache_policy": (
                    "request_local_two_cache_free_forwards_then_"
                    "bounded_result_cache_no_global_kv_mutation"
                ),
            },
            "model": self.model_name,
            "model_id": self.active_model_id,
            "model_label": self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            "model_generation": self.model_generation,
            "transition_id": self.transition_id,
            "server_instance_id": self.server_instance_id,
            "provenance": {
                "server": {
                    "runtime": dict(RUNTIME_PROVENANCE),
                    "model": self._model_provenance_snapshot_unlocked(),
                }
            },
            "top_k": top_k,
            "cached": False,
            "timing_ms": {
                "forward_a": forward_a_ms,
                "forward_b": forward_b_ms,
                "total": total_ms,
            },
            "latency_ms": total_ms,
        }
        with self.cache_lock:
            self.context_contrast_cache[cache_key] = response
            self.context_contrast_cache.move_to_end(cache_key)
            while (
                len(self.context_contrast_cache)
                > CONTEXT_CONTRAST_CACHE_SIZE
            ):
                self.context_contrast_cache.popitem(last=False)
        return response

    def _lens_token_payload(self, token_id: int) -> dict[str, Any]:
        special_ids = (
            set(getattr(self.tokenizer, "all_special_ids", []))
            if not self.mock
            else set()
        )
        raw_token: str | None = None
        if self.mock:
            piece = MOCK_WORDS[token_id % len(MOCK_WORDS)]
        else:
            piece = safe_decode(
                self.tokenizer,
                [token_id],
                skip_special=False,
            )
            try:
                converted = self.tokenizer.convert_ids_to_tokens(token_id)
                raw_token = (
                    str(converted) if converted is not None else None
                )
            except Exception:
                raw_token = None
            if not piece:
                piece = raw_token or f"token {token_id}"
        display_piece = (
            raw_token
            if raw_token and ("\ufffd" in piece or not piece)
            else piece
        )
        return {
            "token_id": token_id,
            "text": piece,
            "display": visible_token(display_piece, f"token {token_id}"),
            "tokenizer_token": raw_token,
            "is_special": token_id in special_ids,
        }

    def _lens_output_head_row_norms(
        self,
        token_ids: np.ndarray,
    ) -> list[float | None]:
        """Return cheap per-candidate unembedding norms when a head exists."""

        if self.mock:
            return [None] * len(token_ids)
        import mlx.core as mx

        lm_head = getattr(self.model, "lm_head", None)
        weight = getattr(lm_head, "weight", None)
        if weight is None:
            return [None] * len(token_ids)
        rows = weight[
            mx.array([int(token_id) for token_id in token_ids])
        ].astype(mx.float32)
        norms = mx.sqrt(mx.sum(rows * rows, axis=-1))
        mx.eval(norms)
        return [float(value) for value in np.asarray(norms)]

    def _mock_lens_logits(
        self,
        text: str,
    ) -> tuple[np.ndarray, list[int]]:
        """Build deterministic mock logits at the active decoder's depth."""
        token_ids = list(range(max(1, len(text.split()))))
        seed = sum((index + 1) * ord(char) for index, char in enumerate(text))
        rng = np.random.default_rng(seed % (2**32))
        vocabulary_size = len(MOCK_WORDS)

        early = rng.normal(0.0, 0.65, size=vocabulary_size)
        middle = rng.normal(0.0, 0.55, size=vocabulary_size)
        final = rng.normal(0.0, 0.45, size=vocabulary_size)
        targets = rng.permutation(vocabulary_size)[:6]
        early[int(targets[0])] += 5.2
        early[int(targets[1])] += 3.3
        middle[int(targets[2])] += 5.5
        middle[int(targets[3])] += 3.4
        final[int(targets[4])] += 6.4
        final[int(targets[5])] += 3.7

        layer_logits: list[np.ndarray] = []
        layer_count = self.representation_layer_count()
        for layer_index in range(layer_count):
            progress = layer_index / max(1, layer_count - 1)
            if progress < 0.5:
                local = progress * 2.0
                logits = (1.0 - local) * early + local * middle
            else:
                local = (progress - 0.5) * 2.0
                smooth = local * local * (3.0 - 2.0 * local)
                logits = (1.0 - smooth) * middle + smooth * final
            if layer_index < layer_count - 1:
                logits = logits + rng.normal(
                    0.0,
                    0.16 * (1.0 - progress),
                    size=vocabulary_size,
                )
            layer_logits.append(logits.astype(np.float32))
        layer_logits[-1] = final.astype(np.float32)
        return np.stack(layer_logits, axis=0), token_ids

    def representation_layer_count(self) -> int:
        count = self.model_descriptor(self.active_model_id).get("layers")
        if type(count) is not int or count < 1:
            raise ValueError("Model nemá známý počet decoderových vrstev.")
        return count

    def _mlx_lens_logits(
        self,
        token_ids: list[int],
    ) -> tuple[np.ndarray, float, float]:
        """Read native states; use the native prefix shape for the final row.

        Quantized GEMMs may round differently at [layers, width] versus
        [1, prefix_length, width]. Intermediate readouts are batched; the
        final row uses the ordinary model's full-prefix projection layout.
        """
        import mlx.core as mx
        self._steering_body()
        forward_started = time.perf_counter()
        body, states, final_hidden = trace_last_position(self.model, token_ids)
        forward_ms = (time.perf_counter() - forward_started) * 1_000
        projection_started = time.perf_counter()
        projected = representation_readout(self.model, body, states[1:]).astype(mx.float32)
        final_projected = representation_readout(self.model, body, final_hidden)[0, -1].astype(mx.float32)
        mx.eval(projected, final_projected)
        logits = np.asarray(projected).copy()
        logits[-1] = np.asarray(final_projected)
        projection_ms = (
            time.perf_counter() - projection_started
        ) * 1_000
        return logits, forward_ms, projection_ms

    @staticmethod
    def _lens_js_divergence_bits(
        probabilities: np.ndarray,
        final_probabilities: np.ndarray,
    ) -> float:
        midpoint = 0.5 * (probabilities + final_probabilities)

        def kl_to_midpoint(distribution: np.ndarray) -> float:
            positive = distribution > 0
            return float(
                np.sum(
                    distribution[positive]
                    * np.log2(
                        distribution[positive] / midpoint[positive]
                    )
                )
            )

        return min(
            1.0,
            max(
                0.0,
                0.5
                * (
                    kl_to_midpoint(probabilities)
                    + kl_to_midpoint(final_probabilities)
                ),
            ),
        )

    def _summarize_lens_logits(
        self,
        logits_by_layer: np.ndarray,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
        layer_count = self.representation_layer_count()
        if logits_by_layer.ndim != 2 or logits_by_layer.shape[0] != layer_count:
            raise ValueError(
                f"RAW logit lens musí vrátit matici {layer_count} vrstev × slovník."
            )
        final_probabilities = probabilities_at_temperature(
            logits_by_layer[-1],
            1.0,
        )
        final_top1_token_id = int(
            np.flatnonzero(
                logits_by_layer[-1] == np.max(logits_by_layer[-1])
            )[0]
        )

        layer_payloads: list[dict[str, Any]] = []
        top1_ids: list[int] = []
        for layer_index, layer_logits in enumerate(logits_by_layer):
            probabilities = probabilities_at_temperature(
                layer_logits,
                1.0,
            )
            (
                top_ids,
                top_probabilities,
                other_probability,
                entropy_bits,
                effective_width,
                normalized_entropy,
            ) = summarize_probabilities(probabilities, top_k)
            candidates = []
            for token_id_value, probability in zip(
                top_ids,
                top_probabilities,
            ):
                token_id = int(token_id_value)
                candidate = self._lens_token_payload(token_id)
                candidate["probability"] = float(probability)
                candidates.append(candidate)

            top1_token_id = int(top_ids[0])
            top1_ids.append(top1_token_id)
            if probabilities.size >= 2:
                top2_token_id = int(top_ids[1])
                probability_gap = max(
                    0.0,
                    float(
                        probabilities[top1_token_id]
                        - probabilities[top2_token_id]
                    ),
                )
                gap_bits = max(
                    0.0,
                    float(
                        (
                            float(layer_logits[top1_token_id])
                            - float(layer_logits[top2_token_id])
                        )
                        / math.log(2.0)
                    ),
                )
            else:
                top2_token_id = None
                probability_gap = None
                gap_bits = None

            layer_payloads.append(
                {
                    "layer": layer_index + 1,
                    "relative_depth": (layer_index + 1) / layer_count,
                    "candidates": candidates,
                    "top1_token_id": top1_token_id,
                    "top2_token_id": top2_token_id,
                    "top_k_mass": 1.0 - other_probability,
                    "other_probability": other_probability,
                    "entropy_bits": entropy_bits,
                    "effective_width": effective_width,
                    "effective_tokens": effective_width,
                    "normalized_entropy": normalized_entropy,
                    "top1_top2_gap": probability_gap,
                    "top1_top2_gap_bits": gap_bits,
                    "js_divergence_to_final_bits": (
                        self._lens_js_divergence_bits(
                            probabilities,
                            final_probabilities,
                        )
                    ),
                    "final_top1_token_id": final_top1_token_id,
                    "final_top1_rank": 1
                    + int(
                        np.count_nonzero(
                            layer_logits
                            > layer_logits[final_top1_token_id]
                        )
                    ),
                    "final_top1_probability": float(
                        probabilities[final_top1_token_id]
                    ),
                }
            )

        stable_from_layer = len(top1_ids)
        for layer_index in range(len(top1_ids) - 1, -1, -1):
            if top1_ids[layer_index] != final_top1_token_id:
                break
            stable_from_layer = layer_index + 1

        final_top1 = self._lens_token_payload(final_top1_token_id)
        final_top1["probability"] = float(
            final_probabilities[final_top1_token_id]
        )
        return layer_payloads, final_top1, stable_from_layer

    def _lens_local(
        self,
        text: str,
        top_k: int,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        tokenization_started = time.perf_counter()
        if self.mock:
            token_ids = list(range(max(1, len(text.split()))))
        else:
            token_ids = encode_with_bos(self.tokenizer, text)
        if not token_ids:
            raise ValueError("Tokenizer nevrátil žádné tokeny.")
        if len(token_ids) > MAX_LENS_INPUT_TOKENS:
            raise ValueError(
                f"Text má {len(token_ids)} tokenů; RAW logit lens je omezen "
                f"na {MAX_LENS_INPUT_TOKENS} tokenů."
            )
        tokenization_ms = (
            time.perf_counter() - tokenization_started
        ) * 1_000

        cache_key = (self.model_generation, text, top_k)
        with self.cache_lock:
            cached_response = self.lens_cache.get(cache_key)
            if cached_response is not None:
                self.lens_cache.move_to_end(cache_key)
                response = dict(cached_response)
                response["cached"] = True
                response["timing_ms"] = {
                    "tokenization": tokenization_ms,
                    "forward": 0.0,
                    "projection": 0.0,
                    "summary": 0.0,
                    "total": (
                        time.perf_counter() - total_started
                    )
                    * 1_000,
                }
                return response

        if self.mock:
            forward_started = time.perf_counter()
            logits_by_layer, token_ids = self._mock_lens_logits(text)
            forward_ms = (
                time.perf_counter() - forward_started
            ) * 1_000
            projection_ms = 0.0
        else:
            (
                logits_by_layer,
                forward_ms,
                projection_ms,
            ) = self._mlx_lens_logits(token_ids)

        summary_started = time.perf_counter()
        layers, final_top1, stable_from_layer = (
            self._summarize_lens_logits(logits_by_layer, top_k)
        )
        input_tokens = []
        for position, token_id in enumerate(token_ids):
            token = self._lens_token_payload(int(token_id))
            token["position"] = position
            input_tokens.append(token)
        summary_ms = (time.perf_counter() - summary_started) * 1_000

        response = {
            "protocol": "raw-layers-v2",
            "projection_execution": "batched_intermediate_states_final_native_prefix_shape",
            "text": text,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "input_tokens": input_tokens,
            "top_k": top_k,
            "temperature": 1.0,
            "probability_kind": "raw_model",
            "lens_kind": "raw_logit_lens",
            "layer_count": len(layers),
            "vocab_size": int(logits_by_layer.shape[1]),
            "layers": layers,
            "final_top1": final_top1,
            "stable_from_layer": stable_from_layer,
            "stable_from_relative_depth": stable_from_layer / len(layers),
            "tokenizer": self._tokenizer_identity_unlocked(),
            "model": self.model_name,
            "model_id": self.active_model_id,
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
            "provenance": {
                "server": {
                    "runtime": dict(RUNTIME_PROVENANCE),
                    "model": self._model_provenance_unlocked(),
                }
            },
            "cached": False,
            "timing_ms": {
                "tokenization": tokenization_ms,
                "forward": forward_ms,
                "projection": projection_ms,
                "summary": summary_ms,
                "total": (
                    time.perf_counter() - total_started
                )
                * 1_000,
            },
        }
        with self.cache_lock:
            self.lens_cache[cache_key] = response
            self.lens_cache.move_to_end(cache_key)
            while len(self.lens_cache) > LENS_CACHE_SIZE:
                self.lens_cache.popitem(last=False)
        return response

    def _model_content_fingerprint_local(self) -> str:
        if self.mock:
            return "mock"
        provenance = self._model_provenance_unlocked()
        cache_key = (
            f"{self.active_model_id}:"
            f"{provenance['metadata_fingerprint']}"
        )
        cached = self.model_content_fingerprint_cache.get(cache_key)
        if cached is not None:
            return cached
        cache_path = (
            self.tuned_lens_root
            / "_model_fingerprints"
            / f"{self.active_model_id}.json"
        )
        fingerprint, _inventory = model_content_fingerprint(
            self.active_model_id,
            self._model_sources_unlocked(),
            cache_path=cache_path,
        )
        self.model_content_fingerprint_cache[cache_key] = fingerprint
        return fingerprint

    @staticmethod
    def _log_probabilities(logits: np.ndarray) -> np.ndarray:
        return teacher_forced_log_probabilities(logits)

    def _summarize_reader_logits(
        self,
        logits: np.ndarray,
        final_logits: np.ndarray,
        top_k: int,
    ) -> dict[str, Any]:
        probabilities = probabilities_at_temperature(logits, 1.0)
        final_probabilities = probabilities_at_temperature(final_logits, 1.0)
        final_log_probabilities = self._log_probabilities(final_logits)
        reader_log_probabilities = self._log_probabilities(logits)
        (
            top_ids,
            top_probabilities,
            other_probability,
            entropy_bits,
            effective_width,
            normalized_entropy,
        ) = summarize_probabilities(probabilities, top_k)
        candidates = []
        output_head_norms = self._lens_output_head_row_norms(top_ids)
        top1_logit = float(logits[int(top_ids[0])])
        for rank, (token_id_value, probability, output_head_norm) in enumerate(
            zip(top_ids, top_probabilities, output_head_norms),
            start=1,
        ):
            token_id = int(token_id_value)
            candidate = self._lens_token_payload(token_id)
            candidate["probability"] = float(probability)
            candidate["rank"] = rank
            candidate["surprisal_bits"] = max(
                0.0,
                float(
                    -reader_log_probabilities[token_id]
                    / math.log(2.0)
                ),
            )
            candidate["logit_gap_from_top1_bits"] = max(
                0.0,
                float(
                    (
                        top1_logit
                        - float(logits[token_id])
                    )
                    / math.log(2.0)
                ),
            )
            candidate["final_probability"] = float(
                final_probabilities[token_id]
            )
            candidate["final_rank"] = 1 + int(
                np.count_nonzero(final_logits > final_logits[token_id])
            )
            candidate["final_surprisal_bits"] = max(
                0.0,
                float(
                    -final_log_probabilities[token_id]
                    / math.log(2.0)
                ),
            )
            if output_head_norm is not None:
                candidate["output_head_row_norm"] = output_head_norm
            candidates.append(candidate)

        final_top1_token_id = int(
            np.flatnonzero(final_logits == np.max(final_logits))[0]
        )
        top1_token_id = int(top_ids[0])
        if probabilities.size >= 2:
            top2_token_id = int(top_ids[1])
            top1_top2_gap = max(
                0.0,
                float(
                    probabilities[top1_token_id]
                    - probabilities[top2_token_id]
                ),
            )
            top1_top2_gap_bits = max(
                0.0,
                float(
                    (
                        float(logits[top1_token_id])
                        - float(logits[top2_token_id])
                    )
                    / math.log(2.0)
                ),
            )
        else:
            top2_token_id = None
            top1_top2_gap = None
            top1_top2_gap_bits = None
        kl_bits = float(
            np.sum(
                final_probabilities
                * (
                    final_log_probabilities
                    - reader_log_probabilities
                )
            )
            / math.log(2.0)
        )
        return {
            "candidates": candidates,
            "top1_token_id": top1_token_id,
            "top2_token_id": top2_token_id,
            "top_k_mass": 1.0 - other_probability,
            "other_probability": other_probability,
            "entropy_bits": entropy_bits,
            "effective_width": effective_width,
            "effective_tokens": effective_width,
            "normalized_entropy": normalized_entropy,
            "top1_top2_gap": top1_top2_gap,
            "top1_top2_gap_bits": top1_top2_gap_bits,
            "kl_divergence_to_final_bits": max(0.0, kl_bits),
            "js_divergence_to_final_bits": self._lens_js_divergence_bits(
                probabilities,
                final_probabilities,
            ),
            "final_top1_token_id": final_top1_token_id,
            "final_top1_rank": 1
            + int(np.count_nonzero(logits > logits[final_top1_token_id])),
            "final_top1_probability": float(
                probabilities[final_top1_token_id]
            ),
            "top1_matches_final": top1_token_id == final_top1_token_id,
        }

    def _mock_tuned_lens_logits(
        self,
        text: str,
        selected_layers: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], float, float]:
        started = time.perf_counter()
        all_logits, token_ids = self._mock_lens_logits(text)
        raw_logits = np.stack(
            [all_logits[layer - 1] for layer in selected_layers],
            axis=0,
        )
        final_logits = all_logits[-1]
        # This is a UI and API simulation only.  Interpolation guarantees that
        # the synthetic learned reader approaches its synthetic teacher.
        tuned_logits = np.stack(
            [
                0.42 * raw + 0.58 * final_logits
                for raw in raw_logits
            ],
            axis=0,
        ).astype(np.float32)
        forward_ms = (time.perf_counter() - started) * 1_000
        return (
            raw_logits,
            tuned_logits,
            final_logits,
            token_ids,
            forward_ms,
            0.0,
        )

    def _apertus_tuned_lens_logits(
        self,
        token_ids: list[int],
        manifest: dict[str, Any],
        weights: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Run selected RAW and learned readers over one shared forward."""

        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        if (
            getattr(self.model, "model_type", None) != "apertus"
            or not hasattr(self.model, "model")
        ):
            raise RuntimeError(
                "RAW × TUNED je implementováno pouze pro Apertus."
            )
        body = self.model.model
        layers = getattr(body, "layers", None)
        if layers is None or len(layers) != 32:
            raise RuntimeError(
                "RAW × TUNED očekává 32 decoderových vrstev Apertusu."
            )
        selected_layers = list(manifest["reader"]["selected_layers"])

        forward_started = time.perf_counter()
        ids = mx.array([token_ids])
        hidden = body.embed_tokens(ids)
        causal_mask = create_attention_mask(hidden, None)
        selected_states: dict[int, Any] = {}
        for layer_number, layer in enumerate(layers, start=1):
            hidden = layer(hidden, mask=causal_mask, cache=None)
            if layer_number in selected_layers:
                selected_states[layer_number] = hidden[0, -1, :]
        final_state = hidden[0, -1, :]
        stacked_states = mx.stack(
            [selected_states[layer] for layer in selected_layers],
            axis=0,
        )
        mx.eval(stacked_states, final_state)
        forward_ms = (time.perf_counter() - forward_started) * 1_000

        projection_started = time.perf_counter()
        translated_states = mx.stack(
            [
                apply_translator(
                    selected_states[layer],
                    layer=layer,
                    manifest=manifest,
                    weights=weights,
                )
                for layer in selected_layers
            ],
            axis=0,
        )
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            raise RuntimeError(
                "Tuned-lens pilot vyžaduje samostatný lm_head Apertusu."
            )
        readout_states = mx.concatenate(
            [
                stacked_states,
                translated_states,
                final_state[None, :],
            ],
            axis=0,
        )
        projected = lm_head(body.norm(readout_states)).astype(mx.float32)
        mx.eval(projected)
        layer_count = len(selected_layers)
        projected_array = np.asarray(projected)
        raw_logits = projected_array[:layer_count]
        tuned_logits = projected_array[layer_count : 2 * layer_count]
        final_logits = projected_array[-1]
        projection_ms = (
            time.perf_counter() - projection_started
        ) * 1_000
        return (
            raw_logits,
            tuned_logits,
            final_logits,
            forward_ms,
            projection_ms,
        )

    def _tuned_lens_local(
        self,
        text: str,
        top_k: int,
        expected_artifact_id: str | None,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        tokenization_started = time.perf_counter()
        status = self._tuned_lens_status_unlocked()
        if status["status"] == "missing":
            raise TunedLensUnavailableError(status["message"])
        if status["status"] == "incompatible":
            raise CalibrationIncompatibleError(status["message"])
        if status["status"] != "ready":
            raise TunedLensUnavailableError(status["message"])
        calibration = status["calibration"]
        artifact_id = str(calibration["artifact_id"])
        if (
            expected_artifact_id is not None
            and expected_artifact_id != artifact_id
        ):
            raise CalibrationChangedError(
                "Aktivní kalibrace se od odeslání požadavku změnila."
            )
        selected_layers = list(
            calibration["reader"]["selected_layers"]
        )
        if self.mock:
            token_ids = list(range(max(1, len(text.split()))))
        else:
            token_ids = encode_with_bos(self.tokenizer, text)
        if not token_ids:
            raise ValueError("Tokenizer nevrátil žádné tokeny.")
        if len(token_ids) > MAX_LENS_INPUT_TOKENS:
            raise ValueError(
                f"Text má {len(token_ids)} tokenů; RAW × TUNED je omezeno "
                f"na {MAX_LENS_INPUT_TOKENS} tokenů."
            )
        tokenization_ms = (
            time.perf_counter() - tokenization_started
        ) * 1_000

        verification_started = time.perf_counter()
        if self.mock:
            manifest = calibration
            weights: dict[str, Any] = {}
        else:
            if self.tuned_lens_catalog is None:
                raise TunedLensUnavailableError(
                    "Tuned-lens katalog není inicializován."
                )
            provenance = self._model_provenance_unlocked()
            content_fingerprint = self._model_content_fingerprint_local()
            manifest, weights = self.tuned_lens_catalog.load(
                model_id=self.active_model_id,
                metadata_fingerprint=provenance[
                    "metadata_fingerprint"
                ],
                content_fingerprint=content_fingerprint,
                expected_artifact_id=artifact_id,
            )
            calibration = self._tuned_lens_status_unlocked()[
                "calibration"
            ]
        verification_ms = (
            time.perf_counter() - verification_started
        ) * 1_000
        calibration_digest = canonical_json_sha256(calibration)
        cache_key = (
            self.model_generation,
            artifact_id,
            calibration["weights"]["sha256"],
            calibration_digest,
            text,
            top_k,
        )
        with self.cache_lock:
            cached_response = self.tuned_lens_cache.get(cache_key)
            if cached_response is not None:
                self.tuned_lens_cache.move_to_end(cache_key)
                response = dict(cached_response)
                response["cached"] = True
                response["timing_ms"] = {
                    "tokenization": tokenization_ms,
                    "artifact_verification": verification_ms,
                    "forward": 0.0,
                    "projection": 0.0,
                    "summary": 0.0,
                    "total": (
                        time.perf_counter() - total_started
                    )
                    * 1_000,
                }
                return response

        if self.mock:
            (
                raw_logits,
                tuned_logits,
                final_logits,
                token_ids,
                forward_ms,
                projection_ms,
            ) = self._mock_tuned_lens_logits(
                text,
                selected_layers,
            )
        else:
            (
                raw_logits,
                tuned_logits,
                final_logits,
                forward_ms,
                projection_ms,
            ) = self._apertus_tuned_lens_logits(
                token_ids,
                manifest,
                weights,
            )

        summary_started = time.perf_counter()
        layers_payload = []
        tuned_closer_count = 0
        for index, layer in enumerate(selected_layers):
            raw = self._summarize_reader_logits(
                raw_logits[index],
                final_logits,
                top_k,
            )
            tuned = self._summarize_reader_logits(
                tuned_logits[index],
                final_logits,
                top_k,
            )
            kl_tuned_minus_raw = (
                tuned["kl_divergence_to_final_bits"]
                - raw["kl_divergence_to_final_bits"]
            )
            js_tuned_minus_raw = (
                tuned["js_divergence_to_final_bits"]
                - raw["js_divergence_to_final_bits"]
            )
            tuned_is_closer = kl_tuned_minus_raw < 0.0
            tuned_closer_count += int(tuned_is_closer)
            layers_payload.append(
                {
                    "layer": layer,
                    "raw": raw,
                    "tuned": tuned,
                    "comparison": {
                        "kl_tuned_minus_raw_bits": kl_tuned_minus_raw,
                        "kl_improvement_bits": -kl_tuned_minus_raw,
                        "js_tuned_minus_raw_bits": js_tuned_minus_raw,
                        "js_improvement_bits": -js_tuned_minus_raw,
                        "final_top1_rank_delta": (
                            tuned["final_top1_rank"]
                            - raw["final_top1_rank"]
                        ),
                        "final_top1_probability_delta": (
                            tuned["final_top1_probability"]
                            - raw["final_top1_probability"]
                        ),
                        "raw_top1_matches_final": raw[
                            "top1_matches_final"
                        ],
                        "tuned_top1_matches_final": tuned[
                            "top1_matches_final"
                        ],
                        "tuned_is_closer_by_kl": tuned_is_closer,
                    },
                }
            )
        final_probabilities = probabilities_at_temperature(
            final_logits,
            1.0,
        )
        final_top1_token_id = int(
            np.flatnonzero(final_logits == np.max(final_logits))[0]
        )
        final_top1 = self._lens_token_payload(final_top1_token_id)
        final_top1["probability"] = float(
            final_probabilities[final_top1_token_id]
        )
        input_tokens = []
        for position, token_id in enumerate(token_ids):
            token = self._lens_token_payload(int(token_id))
            token["position"] = position
            input_tokens.append(token)
        summary_ms = (time.perf_counter() - summary_started) * 1_000

        response = {
            "text": text,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "input_tokens": input_tokens,
            "top_k": top_k,
            "temperature": 1.0,
            "probability_kind": "learned_reader_vs_raw_model",
            "lens_kind": "raw_tuned_comparison",
            "reference_layer": 32,
            "selected_layers": selected_layers,
            "layers": layers_payload,
            "final_top1": final_top1,
            "summary": {
                "tuned_closer_by_kl_count": tuned_closer_count,
                "layer_count": len(layers_payload),
            },
            "calibration": calibration,
            "validation": calibration.get("validation", {}),
            "model": self.model_name,
            "model_id": self.active_model_id,
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
            "provenance": {
                "server": {
                    "runtime": dict(RUNTIME_PROVENANCE),
                    "model": self._model_provenance_unlocked(),
                }
            },
            "cached": False,
            "timing_ms": {
                "tokenization": tokenization_ms,
                "artifact_verification": verification_ms,
                "forward": forward_ms,
                "projection": projection_ms,
                "summary": summary_ms,
                "total": (
                    time.perf_counter() - total_started
                )
                * 1_000,
            },
        }
        with self.cache_lock:
            self.tuned_lens_cache[cache_key] = response
            self.tuned_lens_cache.move_to_end(cache_key)
            while len(self.tuned_lens_cache) > TUNED_LENS_CACHE_SIZE:
                self.tuned_lens_cache.popitem(last=False)
        return response

    def _capture_apertus_jacobian_states(
        self,
        token_ids: list[int],
        selected_layers: list[int],
    ) -> tuple[dict[int, Any], Any, float]:
        """Capture post-block final-position residuals in one cache-free pass."""

        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        if self.active_model_id != "bf16":
            raise JacobianLensIncompatibleError(
                "Tento J-Lens je ověřen pouze pro BF16 Apertus."
            )
        if (
            getattr(self.model, "model_type", None) != "apertus"
            or not hasattr(self.model, "model")
        ):
            raise JacobianLensIncompatibleError(
                "RAW × J-LENS je implementováno pouze pro Apertus."
            )
        body = self.model.model
        layers = getattr(body, "layers", None)
        if layers is None or len(layers) != 32:
            raise JacobianLensIncompatibleError(
                "J-Lens očekává 32 decoderových vrstev Apertusu."
            )
        requested = sorted(set(selected_layers))
        if (
            not requested
            or any(
                type(layer) is not int or not 1 <= layer <= len(layers)
                for layer in requested
            )
        ):
            raise ValueError("Neplatný výběr zdrojových vrstev J-Lens.")

        started = time.perf_counter()
        ids = mx.array([token_ids])
        hidden = body.embed_tokens(ids)
        causal_mask = create_attention_mask(hidden, None)
        selected_states: dict[int, Any] = {}
        for layer_number, layer in enumerate(layers, start=1):
            hidden = layer(hidden, mask=causal_mask, cache=None)
            if layer_number in requested:
                selected_states[layer_number] = hidden[0, -1, :]
        final_state = hidden[0, -1, :]
        mx.eval(*selected_states.values(), final_state)
        return (
            selected_states,
            final_state,
            (time.perf_counter() - started) * 1_000,
        )

    def _project_jacobian_readouts(
        self,
        selected_states: dict[int, Any],
        final_state: Any,
        selected_layers: list[int],
        manifest: dict[str, Any],
        weights: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Project RAW, transported and final states through one shared head."""

        import mlx.core as mx

        body = self.model.model
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            raise JacobianLensIncompatibleError(
                "J-Lens vyžaduje samostatný výstupní lm_head Apertusu."
            )
        started = time.perf_counter()
        raw_states = mx.stack(
            [selected_states[layer] for layer in selected_layers],
            axis=0,
        )
        transported_states = mx.stack(
            [
                apply_jacobian(
                    selected_states[layer],
                    layer=layer,
                    manifest=manifest,
                    weights=weights,
                )
                for layer in selected_layers
            ],
            axis=0,
        )
        # The Jacobian multiplication is FP32, but Anthropic's HF readout casts
        # every residual back to the lm_head weight dtype before final RMSNorm
        # and unembedding.  Mirroring that BF16 boundary is part of the transfer
        # contract; keeping transported states in FP32 here changes top-k ties.
        readout_dtype = lm_head.weight.dtype
        readout_states = mx.concatenate(
            [
                raw_states.astype(readout_dtype),
                transported_states.astype(readout_dtype),
                final_state[None, :].astype(readout_dtype),
            ],
            axis=0,
        )
        projected = lm_head(body.norm(readout_states)).astype(mx.float32)
        mx.eval(projected)
        projected_array = np.asarray(projected)
        layer_count = len(selected_layers)
        return (
            projected_array[:layer_count],
            projected_array[layer_count : 2 * layer_count],
            projected_array[-1],
            (time.perf_counter() - started) * 1_000,
        )

    @staticmethod
    def _top_token_ids(logits: np.ndarray, count: int = 20) -> list[int]:
        return [
            int(value)
            for value in np.argsort(-logits, kind="stable")[:count]
        ]

    def _run_jacobian_transfer_probe_local(
        self,
        manifest: dict[str, Any],
        weights: dict[str, Any],
        probe_tensors: dict[str, Any],
        probe_json: dict[str, Any],
        model_fingerprint: str,
    ) -> dict[str, Any]:
        """Attest tokenizer and post-block residual parity before J-Lens use."""

        if self.jacobian_lens_catalog is None:
            raise JacobianLensUnavailableError(
                "J-Lens katalog není inicializován."
            )
        selected_layers = list(manifest["reader"]["source_layers"])
        actual_input_ids: dict[int, list[int]] = {}
        actual_residuals: dict[tuple[int, int], np.ndarray] = {}
        transport_checks: list[dict[str, Any]] = []
        jacobian_output_checks: list[dict[str, Any]] = []
        final_output_checks: list[dict[str, Any]] = []
        import mlx.core as mx

        for record in probe_json["records"]:
            prompt_index = int(record["prompt_index"])
            token_ids = encode_with_bos(self.tokenizer, record["prompt"])
            actual_input_ids[prompt_index] = token_ids
            selected_states, final_state, _forward_ms = (
                self._capture_apertus_jacobian_states(
                    token_ids,
                    selected_layers,
                )
            )
            expected_layers = {
                int(item["layer"]): item for item in record["layers"]
            }
            body = self.model.model
            lm_head = getattr(self.model, "lm_head", None)
            if lm_head is None:
                raise JacobianLensIncompatibleError(
                    "Přenosová sonda vyžaduje samostatný lm_head."
                )
            for layer in selected_layers:
                state = selected_states[layer].astype(mx.float32)
                mx.eval(state)
                actual_residuals[(prompt_index, layer)] = np.asarray(state)
                transported = apply_jacobian(
                    state,
                    layer=layer,
                    manifest=manifest,
                    weights=weights,
                )
                transported_comparison = compare_vectors(
                    transported,
                    probe_tensors[
                        f"prompt_{prompt_index:02d}."
                        f"layer_{layer:02d}.transported"
                    ],
                )
                transport_checks.append(
                    {
                        "prompt_index": prompt_index,
                        "layer": layer,
                        **asdict(transported_comparison),
                    }
                )
                jacobian_logits = lm_head(
                    body.norm(
                        transported[None, :].astype(lm_head.weight.dtype)
                    )
                ).astype(mx.float32)
                mx.eval(jacobian_logits)
                actual_jacobian_top20 = self._top_token_ids(
                    np.asarray(jacobian_logits)[0],
                    20,
                )
                expected_jacobian_top20 = [
                    int(item["token_id"])
                    for item in expected_layers[layer]["jlens_top20"]
                ]
                jacobian_top1_matches = (
                    actual_jacobian_top20[0]
                    == expected_jacobian_top20[0]
                )
                jacobian_top20_overlap = (
                    len(
                        set(actual_jacobian_top20)
                        & set(expected_jacobian_top20)
                    )
                    / 20.0
                )
                jacobian_output_checks.append(
                    {
                        "prompt_index": prompt_index,
                        "layer": layer,
                        "passed": (
                            jacobian_top1_matches
                            and jacobian_top20_overlap
                            >= RUNTIME_JACOBIAN_TOP20_OVERLAP_MIN
                        ),
                        "top1_matches": jacobian_top1_matches,
                        "top20_overlap": jacobian_top20_overlap,
                        "actual_top20": actual_jacobian_top20,
                        "expected_top20": expected_jacobian_top20,
                    }
                )

            final_logits = lm_head(
                body.norm(
                    final_state[None, :].astype(lm_head.weight.dtype)
                )
            ).astype(mx.float32)
            mx.eval(final_logits)
            actual_top20 = self._top_token_ids(
                np.asarray(final_logits)[0],
                20,
            )
            expected_top20 = [
                int(item["token_id"]) for item in record["final_top20"]
            ]
            overlap = len(set(actual_top20) & set(expected_top20)) / 20.0
            final_output_checks.append(
                {
                    "prompt_index": prompt_index,
                    "passed": (
                        overlap >= RUNTIME_FINAL_TOP20_OVERLAP_MIN
                    ),
                    "top1_matches": (
                        bool(actual_top20)
                        and bool(expected_top20)
                        and actual_top20[0] == expected_top20[0]
                    ),
                    "top20_overlap": overlap,
                    "actual_top20": actual_top20,
                    "expected_top20": expected_top20,
                }
            )

        built = runtime_probe_report(
            manifest,
            probe_json,
            probe_tensors,
            actual_input_ids=actual_input_ids,
            actual_residuals=actual_residuals,
        )
        report = (
            built.to_dict()
            if hasattr(built, "to_dict")
            else dict(built)
        )
        report["output_checks"] = final_output_checks
        report["transport_checks"] = transport_checks
        report["jacobian_output_checks"] = jacobian_output_checks
        report["output_thresholds"] = {
            "jacobian_top1_exact": True,
            "jacobian_top20_overlap_min": (
                RUNTIME_JACOBIAN_TOP20_OVERLAP_MIN
            ),
            "jacobian_top20_overlap_mean_min": (
                RUNTIME_JACOBIAN_TOP20_OVERLAP_MEAN_MIN
            ),
            "jacobian_mean_noninferiority_margin": (
                RUNTIME_JACOBIAN_MEAN_NONINFERIORITY_MARGIN
            ),
            "final_top20_overlap_min": RUNTIME_FINAL_TOP20_OVERLAP_MIN,
        }
        report["residual_alignment_passed"] = bool(report.get("passed"))
        jacobian_mean_overlap = (
            sum(
                check["top20_overlap"]
                for check in jacobian_output_checks
            )
            / len(jacobian_output_checks)
        )
        report["jacobian_top20_overlap_mean"] = jacobian_mean_overlap
        final_mean_overlap = (
            sum(check["top20_overlap"] for check in final_output_checks)
            / len(final_output_checks)
        )
        report["final_top20_overlap_mean"] = final_mean_overlap
        report["rank_transfer_scope"] = (
            "verified_on_3_frozen_pytorch_bf16_to_mlx_probes"
        )
        report["passed"] = (
            all(check["passed"] for check in report["token_checks"])
            and all(
                check["passed"] for check in jacobian_output_checks
            )
            and jacobian_mean_overlap
            >= RUNTIME_JACOBIAN_TOP20_OVERLAP_MEAN_MIN
            and all(check["passed"] for check in final_output_checks)
            and jacobian_mean_overlap
            >= final_mean_overlap
            - RUNTIME_JACOBIAN_MEAN_NONINFERIORITY_MARGIN
        )
        report["warnings"] = [
            {
                "prompt_index": check["prompt_index"],
                "code": "final_top1_differs_within_compatible_top20",
                "top20_overlap": check["top20_overlap"],
            }
            for check in final_output_checks
            if not check["top1_matches"]
        ]
        if not report["residual_alignment_passed"]:
            report["warnings"].append(
                {
                    "code": "raw_residuals_differ_across_frameworks",
                    "worst_relative_l2": report["worst_relative_l2"],
                    "worst_cosine": report["worst_cosine"],
                    "interpretation": (
                        "Přenos je přijat podle přímé parity J-Lens "
                        "readoutu; syrové reziduály zůstávají diagnostikou."
                    ),
                }
            )
        weights_sha = manifest["files"]["weights"]["sha256"]
        probe_cache_key = (
            manifest["artifact_id"],
            weights_sha,
            model_fingerprint,
        )
        # Keep the failed report available to diagnostics and the UI as well.
        # A failure never marks the artifact as enabled, but hiding the measured
        # mismatch would make a PyTorch → MLX transfer error impossible to audit.
        self.jacobian_probe_report_cache[probe_cache_key] = report
        if not report["passed"]:
            raise TransferProbeError(
                "BF16 přenosová sonda J-Lens neprošla; čtečka zůstává vypnutá."
            )
        self.jacobian_lens_catalog.mark_runtime_probe(
            artifact_id=manifest["artifact_id"],
            weights_sha256=weights_sha,
            model_fingerprint=model_fingerprint,
            report=report,
        )
        return report

    def _mock_jacobian_lens_logits(
        self,
        text: str,
        selected_layers: list[int],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[int],
        float,
        float,
    ]:
        started = time.perf_counter()
        all_logits, token_ids = self._mock_lens_logits(text)
        raw_logits = np.stack(
            [all_logits[layer - 1] for layer in selected_layers],
            axis=0,
        )
        final_logits = all_logits[-1]
        jacobian_logits = np.stack(
            [
                0.72 * raw + 0.28 * np.roll(final_logits, layer % 7)
                for raw, layer in zip(raw_logits, selected_layers)
            ],
            axis=0,
        ).astype(np.float32)
        return (
            raw_logits,
            jacobian_logits,
            final_logits,
            token_ids,
            (time.perf_counter() - started) * 1_000,
            0.0,
        )

    def _jacobian_lens_local(
        self,
        text: str,
        top_k: int,
        source_layers: list[int] | None,
        expected_artifact_id: str | None,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        if self.jacobian_lens_catalog is None:
            raise JacobianLensUnavailableError(
                "J-Lens katalog není inicializován."
            )
        if not self.mock and self.active_model_id != "bf16":
            raise JacobianLensIncompatibleError(
                "Tento J-Lens je dostupný pouze pro BF16 Apertus."
            )

        verification_started = time.perf_counter()
        if self.mock:
            manifest = {
                "artifact_id": "mock-average-jacobian-v1",
                "reader": {
                    "source_layers": [4, 16, 17, 22, 23, 24, 31],
                    "target_layer": 32,
                    "transport": "residual @ jacobian.T",
                    "readout": (
                        "lm_head(final_rms_norm(transported_residual))"
                    ),
                },
                "model": {"mode": "bf16"},
                "compatibility": {
                    "allowed_model_modes": ["bf16"],
                    "hybrid_transfer": "not_validated",
                },
                "limitations": ["Syntetická simulace bez Jacobianů."],
                "files": {
                    "weights": {
                        "sha256": "0" * 64,
                        "bytes": 1,
                    }
                },
            }
            if (
                expected_artifact_id is not None
                and expected_artifact_id != manifest["artifact_id"]
            ):
                raise JacobianLensChangedError(
                    "J-Lens artefakt se od načtení stránky změnil."
                )
            weights = {}
            probe_tensors = {}
            probe_json = {}
        else:
            manifest, weights, probe_tensors, probe_json = (
                self.jacobian_lens_catalog.load(
                    model_id=self.active_model_id,
                    expected_artifact_id=expected_artifact_id,
                )
            )
            validate_bf16_runtime_contract(
                self.model,
                model_id=self.active_model_id,
                manifest=manifest,
            )
        available_layers = list(manifest["reader"]["source_layers"])
        selected_layers = (
            available_layers
            if source_layers is None
            else sorted(set(source_layers))
        )
        if not selected_layers or any(
            layer not in available_layers for layer in selected_layers
        ):
            raise ValueError(
                "source_layers musejí být neprázdnou podmnožinou vrstev artefaktu."
            )
        if self.mock:
            model_fingerprint = "mock"
            probe_report = {
                "kind": "mock_runtime_probe",
                "passed": True,
                "content_verification": "mock",
            }
        else:
            model_fingerprint = self._model_content_fingerprint_local()
            probe_key = (
                manifest["artifact_id"],
                manifest["files"]["weights"]["sha256"],
                model_fingerprint,
            )
            probe_report = self.jacobian_probe_report_cache.get(probe_key)
            if probe_report is None:
                probe_report = self._run_jacobian_transfer_probe_local(
                    manifest,
                    weights,
                    probe_tensors,
                    probe_json,
                    model_fingerprint,
                )
            elif probe_report.get("passed") is not True:
                raise TransferProbeError(
                    "Poslední BF16 přenosová sonda J-Lens neprošla; "
                    "čtečka zůstává vypnutá."
                )
        verification_ms = (
            time.perf_counter() - verification_started
        ) * 1_000

        tokenization_started = time.perf_counter()
        if self.mock:
            token_ids = list(range(max(1, len(text.split()))))
        else:
            token_ids = encode_with_bos(self.tokenizer, text)
        if not token_ids:
            raise ValueError("Tokenizer nevrátil žádné tokeny.")
        if len(token_ids) > MAX_LENS_INPUT_TOKENS:
            raise ValueError(
                f"Text má {len(token_ids)} tokenů; RAW × J-LENS je omezeno "
                f"na {MAX_LENS_INPUT_TOKENS} tokenů."
            )
        tokenization_ms = (
            time.perf_counter() - tokenization_started
        ) * 1_000

        artifact = self.jacobian_lens_catalog.status(
            model_id=("bf16" if self.mock else self.active_model_id),
            mock=self.mock,
            model_fingerprint=model_fingerprint,
            expected_artifact_id=manifest["artifact_id"],
        ).get("calibration")
        if self.mock:
            artifact = self._jacobian_lens_status_unlocked()["artifact"]
        if artifact is None:
            from_jacobian_manifest = {
                "artifact_id": manifest["artifact_id"],
                "reader": manifest["reader"],
                "model": manifest["model"],
                "compatibility": manifest["compatibility"],
                "limitations": manifest.get("limitations", []),
                "weights": {
                    "sha256": manifest["files"]["weights"]["sha256"],
                    "bytes": manifest["files"]["weights"]["bytes"],
                },
            }
            artifact = from_jacobian_manifest
        artifact_digest = canonical_jacobian_json_sha256(artifact)
        cache_key = (
            self.model_generation,
            manifest["artifact_id"],
            manifest["files"]["weights"]["sha256"],
            artifact_digest,
            text,
            top_k,
            tuple(selected_layers),
        )
        with self.cache_lock:
            cached_response = self.jacobian_lens_cache.get(cache_key)
            if cached_response is not None:
                self.jacobian_lens_cache.move_to_end(cache_key)
                response = dict(cached_response)
                response["cached"] = True
                response["timing_ms"] = {
                    "tokenization": tokenization_ms,
                    "artifact_and_transfer_verification": verification_ms,
                    "forward": 0.0,
                    "projection": 0.0,
                    "summary": 0.0,
                    "total": (
                        time.perf_counter() - total_started
                    )
                    * 1_000,
                }
                return response

        if self.mock:
            (
                raw_logits,
                jacobian_logits,
                final_logits,
                token_ids,
                forward_ms,
                projection_ms,
            ) = self._mock_jacobian_lens_logits(
                text,
                selected_layers,
            )
        else:
            selected_states, final_state, forward_ms = (
                self._capture_apertus_jacobian_states(
                    token_ids,
                    selected_layers,
                )
            )
            (
                raw_logits,
                jacobian_logits,
                final_logits,
                projection_ms,
            ) = self._project_jacobian_readouts(
                selected_states,
                final_state,
                selected_layers,
                manifest,
                weights,
            )

        summary_started = time.perf_counter()
        layers_payload = []
        jacobian_closer_count = 0
        top1_disagreement_count = 0
        for index, layer in enumerate(selected_layers):
            raw = self._summarize_reader_logits(
                raw_logits[index],
                final_logits,
                top_k,
            )
            jacobian = self._summarize_reader_logits(
                jacobian_logits[index],
                final_logits,
                top_k,
            )
            js_delta = (
                jacobian["js_divergence_to_final_bits"]
                - raw["js_divergence_to_final_bits"]
            )
            kl_delta = (
                jacobian["kl_divergence_to_final_bits"]
                - raw["kl_divergence_to_final_bits"]
            )
            jacobian_closer_count += int(js_delta < 0.0)
            top1_disagreement_count += int(
                raw["top1_token_id"] != jacobian["top1_token_id"]
            )
            layers_payload.append(
                {
                    "layer": layer,
                    "raw": raw,
                    "jacobian": jacobian,
                    "comparison": {
                        "js_jacobian_minus_raw_bits": js_delta,
                        "kl_jacobian_minus_raw_bits": kl_delta,
                        "final_top1_rank_delta": (
                            jacobian["final_top1_rank"]
                            - raw["final_top1_rank"]
                        ),
                        "final_top1_probability_delta": (
                            jacobian["final_top1_probability"]
                            - raw["final_top1_probability"]
                        ),
                        "top1_disagrees": (
                            raw["top1_token_id"]
                            != jacobian["top1_token_id"]
                        ),
                        "jacobian_is_closer_by_js": js_delta < 0.0,
                    },
                }
            )
        final_probabilities = probabilities_at_temperature(
            final_logits,
            1.0,
        )
        final_top1_token_id = int(
            np.flatnonzero(final_logits == np.max(final_logits))[0]
        )
        final_top1 = self._lens_token_payload(final_top1_token_id)
        final_top1["probability"] = float(
            final_probabilities[final_top1_token_id]
        )
        input_tokens = []
        for position, token_id in enumerate(token_ids):
            token = self._lens_token_payload(int(token_id))
            token["position"] = position
            input_tokens.append(token)
        summary_ms = (time.perf_counter() - summary_started) * 1_000

        response = {
            "text": text,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "input_tokens": input_tokens,
            "top_k": top_k,
            "temperature": 1.0,
            "probability_kind": "alternative_reader_vs_raw_model",
            "lens_kind": "raw_jacobian_comparison",
            "reader": {
                "kind": "raw_vs_jacobian_lens",
                "transport": manifest["reader"]["transport"],
                "readout": manifest["reader"]["readout"],
            },
            "target_layer": manifest["reader"]["target_layer"],
            "source_layers": selected_layers,
            "layers": layers_payload,
            "final_top1": final_top1,
            "summary": {
                "layer_count": len(layers_payload),
                "top1_disagreement_count": top1_disagreement_count,
                "jacobian_closer_by_js_count": jacobian_closer_count,
                "jacobian_farther_by_js_count": (
                    len(layers_payload) - jacobian_closer_count
                ),
            },
            "artifact_id": manifest["artifact_id"],
            "artifact": artifact,
            "transfer_probe": probe_report,
            "model": self.model_name,
            "model_id": self.active_model_id,
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
            "provenance": {
                "server": {
                    "runtime": dict(RUNTIME_PROVENANCE),
                    "model": self._model_provenance_unlocked(),
                }
            },
            "warnings": [
                {
                    "code": "unstable_split_layer",
                    "layer": 4,
                    "message": (
                        "L4 má slabší shodu obou fitovacích polovin; "
                        "čti ji explorativně."
                    ),
                }
            ],
            "cached": False,
            "timing_ms": {
                "tokenization": tokenization_ms,
                "artifact_and_transfer_verification": verification_ms,
                "forward": forward_ms,
                "projection": projection_ms,
                "summary": summary_ms,
                "total": (
                    time.perf_counter() - total_started
                )
                * 1_000,
            },
        }
        with self.cache_lock:
            self.jacobian_lens_cache[cache_key] = response
            self.jacobian_lens_cache.move_to_end(cache_key)
            while len(self.jacobian_lens_cache) > JACOBIAN_LENS_CACHE_SIZE:
                self.jacobian_lens_cache.popitem(last=False)
        return response

    def _component_attribution_token(
        self,
        query: str,
    ) -> tuple[int, dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Token musí být neprázdný řetězec nebo #ID.")
        stripped = query.strip()
        id_match = re.fullmatch(r"#(\d+)", stripped)
        if id_match is not None:
            token_id = int(id_match.group(1))
        else:
            token_text = query.replace("␠", " ")
            if self.mock:
                try:
                    token_id = MOCK_WORDS.index(token_text)
                except ValueError:
                    digest = hashlib.sha256(
                        token_text.encode("utf-8")
                    ).digest()
                    token_id = int.from_bytes(digest[:4], "big") % len(
                        MOCK_WORDS
                    )
            else:
                try:
                    encoded = list(
                        self.tokenizer.encode(
                            token_text,
                            add_special_tokens=False,
                        )
                    )
                except TypeError:
                    encoded = list(self.tokenizer.encode(token_text))
                    bos = getattr(self.tokenizer, "bos_token_id", None)
                    if bos is not None and encoded[:1] == [bos]:
                        encoded = encoded[1:]
                if len(encoded) != 1:
                    pieces = [
                        visible_token(
                            safe_decode(
                                self.tokenizer,
                                [int(token_id_value)],
                                skip_special=False,
                            ),
                            f"token {token_id_value}",
                        )
                        for token_id_value in encoded[:12]
                    ]
                    preview = ", ".join(pieces) or "žádný token"
                    raise ValueError(
                        "Kontrast vyžaduje právě jeden token na každé "
                        f"straně; zadaný text se rozpadl na "
                        f"{len(encoded)} tokenů ({preview}). Použij přesný "
                        "token včetně úvodní mezery nebo #ID."
                    )
                token_id = int(encoded[0])

        if self.mock:
            vocabulary_size = len(MOCK_WORDS)
        else:
            arguments = getattr(self.model, "args", None)
            vocabulary_size = int(
                getattr(
                    arguments,
                    "vocab_size",
                    getattr(self.tokenizer, "vocab_size", 0),
                )
            )
        if not 0 <= token_id < vocabulary_size:
            raise ValueError(
                f"Token #{token_id} leží mimo slovník 0–"
                f"{vocabulary_size - 1}."
            )
        payload = self._lens_token_payload(token_id)
        payload["query"] = query
        return token_id, payload

    def _mock_component_attribution_tensors(
        self,
        text: str,
        token_a: int,
        token_b: int,
    ) -> dict[str, Any]:
        # One fixed synthetic model/forward per prefix: swapping A and B must
        # only reverse the contrast, never regenerate the residual trajectory.
        seed_material = f"{text}\0component-attribution-v1".encode("utf-8")
        seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(seed)
        width = 48
        layer_count = 32
        vocabulary_size = len(MOCK_WORDS)
        head_rows = rng.normal(
            0.0,
            0.22,
            size=(vocabulary_size, width),
        ).astype(np.float32)
        norm_weight = rng.normal(1.0, 0.025, size=width).astype(np.float32)
        embedding = rng.normal(0.0, 0.55, size=width).astype(np.float32)
        attention_updates = rng.normal(
            0.0,
            0.055,
            size=(layer_count, width),
        ).astype(np.float32)
        mlp_updates = rng.normal(
            0.0,
            0.075,
            size=(layer_count, width),
        ).astype(np.float32)

        contrast_direction = (
            head_rows[0].astype(np.float64)
            - head_rows[1].astype(np.float64)
        )
        contrast_direction /= max(
            float(np.linalg.norm(contrast_direction)),
            1e-12,
        )
        profile = np.exp(
            -0.5 * ((np.arange(layer_count) - 20.0) / 4.5) ** 2
        )
        attention_updates += (
            0.09 * profile[:, None] * contrast_direction[None, :]
        ).astype(np.float32)
        mlp_updates += (
            -0.05
            * np.roll(profile, 3)[:, None]
            * contrast_direction[None, :]
        ).astype(np.float32)

        state = embedding.astype(np.float32, copy=True)
        after_attention_states = []
        after_layer_states = []
        for layer_index in range(layer_count):
            state = (state + attention_updates[layer_index]).astype(
                np.float32
            )
            after_attention_states.append(state.copy())
            state = (state + mlp_updates[layer_index]).astype(np.float32)
            after_layer_states.append(state.copy())
        final_residual = state
        epsilon = 1e-5
        normalized = (
            final_residual.astype(np.float64)
            * norm_weight.astype(np.float64)
            / math.sqrt(
                float(
                    np.mean(final_residual.astype(np.float64) ** 2)
                )
                + epsilon
            )
        )
        final_logits = normalized @ head_rows.astype(np.float64).T
        token_ids = [
            index % vocabulary_size
            for index in range(max(1, len(text.split()) + 1))
        ]
        return {
            "embedding": embedding,
            "attention_updates": attention_updates,
            "mlp_updates": mlp_updates,
            "after_attention_states": np.stack(
                after_attention_states,
                axis=0,
            ),
            "after_layer_states": np.stack(after_layer_states, axis=0),
            "final_residual": final_residual,
            "norm_weight": norm_weight,
            "row_a": head_rows[token_a],
            "row_b": head_rows[token_b],
            "epsilon": epsilon,
            "final_logits": final_logits,
            "token_ids": token_ids,
            "forward_ms": 4.0,
            "projection_ms": 1.0,
        }

    def _apertus_component_attribution_tensors(
        self,
        token_ids: list[int],
        token_a: int,
        token_b: int,
    ) -> dict[str, Any]:
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        if "attribution" not in self.model_descriptor(self.active_model_id)["capabilities"]:
            raise RuntimeError(
                "Komponentový rozklad zatím vyžaduje plnou BF16 "
                "výstupní hlavu; použij Hybrid nebo BF16."
            )
        if getattr(self.model, "model_type", None) != "apertus" or not hasattr(
            self.model,
            "model",
        ):
            raise RuntimeError(
                "Komponentový rozklad je implementován pouze pro Apertus."
            )
        body = self.model.model
        layers = getattr(body, "layers", None)
        if layers is None or len(layers) != 32:
            raise RuntimeError(
                "Komponentový rozklad očekává 32 vrstev Apertusu."
            )
        lm_head = getattr(self.model, "lm_head", None)
        weight = getattr(lm_head, "weight", None)
        hidden_size = int(getattr(self.model.args, "hidden_size", 0))
        vocabulary_size = int(getattr(self.model.args, "vocab_size", 0))
        if (
            lm_head is None
            or weight is None
            or getattr(weight, "ndim", None) != 2
            or tuple(weight.shape) != (vocabulary_size, hidden_size)
        ):
            raise RuntimeError(
                "Aktivní model nemá přímo adresovatelnou plnou výstupní "
                "hlavu. Přepni na Hybrid nebo BF16."
            )
        norm_weight = getattr(body.norm, "weight", None)
        if norm_weight is None or tuple(norm_weight.shape) != (hidden_size,):
            raise RuntimeError(
                "Finální RMSNorm Apertusu nemá očekávanou diagonální váhu."
            )

        forward_started = time.perf_counter()
        ids = mx.array([token_ids])
        hidden = body.embed_tokens(ids)
        causal_mask = create_attention_mask(hidden, None)
        embedding_last = hidden[0, -1, :]
        attention_updates: list[Any] = []
        mlp_updates: list[Any] = []
        after_attention_states: list[Any] = []
        after_layer_states: list[Any] = []
        for layer in layers:
            attention_update = layer.self_attn(
                layer.attention_layernorm(hidden),
                mask=causal_mask,
                cache=None,
            )
            after_attention = hidden + attention_update
            mlp_update = layer.mlp(
                layer.feedforward_layernorm(after_attention)
            )
            hidden = after_attention + mlp_update
            attention_updates.append(attention_update[0, -1, :])
            mlp_updates.append(mlp_update[0, -1, :])
            after_attention_states.append(after_attention[0, -1, :])
            after_layer_states.append(hidden[0, -1, :])

        final_residual = hidden[0, -1, :]
        stacked_attention = mx.stack(attention_updates, axis=0).astype(
            mx.float32
        )
        stacked_mlp = mx.stack(mlp_updates, axis=0).astype(mx.float32)
        stacked_after_attention = mx.stack(
            after_attention_states,
            axis=0,
        ).astype(mx.float32)
        stacked_after_layer = mx.stack(
            after_layer_states,
            axis=0,
        ).astype(mx.float32)
        evaluated = [
            embedding_last.astype(mx.float32),
            stacked_attention,
            stacked_mlp,
            stacked_after_attention,
            stacked_after_layer,
            final_residual.astype(mx.float32),
            norm_weight.astype(mx.float32),
            weight[token_a].astype(mx.float32),
            weight[token_b].astype(mx.float32),
        ]
        mx.eval(*evaluated)
        forward_ms = (time.perf_counter() - forward_started) * 1_000

        projection_started = time.perf_counter()
        final_normalized = body.norm(hidden)[0, -1, :]
        final_logits_array = lm_head(final_normalized[None, :])[0].astype(
            mx.float32
        )
        mx.eval(final_logits_array)
        final_logits = np.asarray(final_logits_array)
        projection_ms = (
            time.perf_counter() - projection_started
        ) * 1_000
        return {
            "embedding": np.asarray(evaluated[0]),
            "attention_updates": np.asarray(evaluated[1]),
            "mlp_updates": np.asarray(evaluated[2]),
            "after_attention_states": np.asarray(evaluated[3]),
            "after_layer_states": np.asarray(evaluated[4]),
            "final_residual": np.asarray(evaluated[5]),
            "norm_weight": np.asarray(evaluated[6]),
            "row_a": np.asarray(evaluated[7]),
            "row_b": np.asarray(evaluated[8]),
            "epsilon": float(getattr(body.norm, "eps", 1e-5)),
            "final_logits": final_logits,
            "token_ids": token_ids,
            "forward_ms": forward_ms,
            "projection_ms": projection_ms,
        }

    def _component_attribution_local(
        self,
        text: str,
        token_a_query: str,
        token_b_query: str,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        tokenization_started = time.perf_counter()
        token_a, token_a_payload = self._component_attribution_token(
            token_a_query
        )
        token_b, token_b_payload = self._component_attribution_token(
            token_b_query
        )
        if token_a == token_b:
            raise ValueError(
                "Obě strany kontrastu označují tentýž token; zvol dva "
                "různé tokeny."
            )
        if self.mock:
            token_ids = [
                index % len(MOCK_WORDS)
                for index in range(max(1, len(text.split()) + 1))
            ]
        else:
            token_ids = encode_with_bos(self.tokenizer, text)
        if not token_ids:
            raise ValueError("Tokenizer nevrátil žádné tokeny.")
        if len(token_ids) > MAX_LENS_INPUT_TOKENS:
            raise ValueError(
                f"Prefix má {len(token_ids)} tokenů; komponentový rozklad "
                f"je omezen na {MAX_LENS_INPUT_TOKENS} tokenů."
            )
        tokenization_ms = (
            time.perf_counter() - tokenization_started
        ) * 1_000

        cache_key = (
            self.model_generation,
            text,
            token_a,
            token_b,
        )
        with self.cache_lock:
            cached_response = self.component_attribution_cache.get(cache_key)
            if cached_response is not None:
                self.component_attribution_cache.move_to_end(cache_key)
                response = dict(cached_response)
                response["cached"] = True
                response["timing_ms"] = {
                    "tokenization": tokenization_ms,
                    "forward": 0.0,
                    "projection": 0.0,
                    "accounting": 0.0,
                    "total": (
                        time.perf_counter() - total_started
                    )
                    * 1_000,
                }
                return response

        if self.mock:
            tensors = self._mock_component_attribution_tensors(
                text,
                token_a,
                token_b,
            )
        else:
            tensors = self._apertus_component_attribution_tensors(
                token_ids,
                token_a,
                token_b,
            )

        accounting_started = time.perf_counter()
        final_logits = np.asarray(
            tensors["final_logits"],
            dtype=np.float64,
        )
        attribution = decompose_residual_contrast(
            embedding=tensors["embedding"],
            attention_updates=tensors["attention_updates"],
            mlp_updates=tensors["mlp_updates"],
            final_residual=tensors["final_residual"],
            norm_weight=tensors["norm_weight"],
            lm_head_row_a=tensors["row_a"],
            lm_head_row_b=tensors["row_b"],
            epsilon=float(tensors["epsilon"]),
            after_attention_states=tensors["after_attention_states"],
            after_layer_states=tensors["after_layer_states"],
            actual_logit_a=float(final_logits[token_a]),
            actual_logit_b=float(final_logits[token_b]),
        )
        probabilities = probabilities_at_temperature(final_logits, 1.0)
        final_top1_token_id = int(
            np.flatnonzero(final_logits == np.max(final_logits))[0]
        )
        final_top1 = self._lens_token_payload(final_top1_token_id)
        final_top1["probability"] = float(
            probabilities[final_top1_token_id]
        )
        attribution["final"].update(
            {
                "probability_a": float(probabilities[token_a]),
                "probability_b": float(probabilities[token_b]),
                "rank_a": 1
                + int(np.count_nonzero(final_logits > final_logits[token_a])),
                "rank_b": 1
                + int(np.count_nonzero(final_logits > final_logits[token_b])),
                "top1": final_top1,
            }
        )
        input_tokens = []
        for position, token_id in enumerate(tensors["token_ids"]):
            payload = self._lens_token_payload(int(token_id))
            payload["position"] = position
            input_tokens.append(payload)
        accounting_ms = (
            time.perf_counter() - accounting_started
        ) * 1_000

        response = {
            "kind": "component_attribution",
            "text": text,
            "token_ids": [int(value) for value in tensors["token_ids"]],
            "input_tokens": input_tokens,
            "contrast": {
                "definition": "logit(A) - logit(B)",
                "a": token_a_payload,
                "b": token_b_payload,
            },
            **attribution,
            "model": self.model_name,
            "model_id": self.active_model_id,
            "model_label": self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            "model_generation": self.model_generation,
            "transition_id": self.transition_id,
            "server_instance_id": self.server_instance_id,
            "provenance": {
                "runtime": dict(RUNTIME_PROVENANCE),
                "model": self._model_provenance_unlocked(),
            },
            "execution": {
                "mode": "single_cache_free_forward_component_capture",
                "forward_passes": 1,
                "generation": False,
                "steering": False,
                "attention_weights_used": False,
                "captured_components": (
                    "embedding_plus_32_attention_plus_32_mlp"
                ),
                "output_head": "two_rows_plus_one_full_final_projection",
            },
            "warnings": [
                {
                    "code": "descriptive_not_causal",
                    "message": (
                        "Příspěvek komponenty není ablace. Kauzalitu by "
                        "muselo ověřit vypnutí, patching nebo jiný zásah."
                    ),
                },
                {
                    "code": "fixed_final_normalization",
                    "message": (
                        "Aditivní waterfall používá pro všechny komponenty "
                        "tutéž RMSNorm škálu z finálního rezidua."
                    ),
                },
            ],
            "cached": False,
            "timing_ms": {
                "tokenization": tokenization_ms,
                "forward": float(tensors["forward_ms"]),
                "projection": float(tensors["projection_ms"]),
                "accounting": accounting_ms,
                "total": (time.perf_counter() - total_started) * 1_000,
            },
        }
        if self.active_model_id == HYBRID_MODEL_ID:
            response["warnings"].append(
                {
                    "code": "hybrid_quantized_body",
                    "message": (
                        "Hybrid používá 8bitové tělo a BF16 vstupní i "
                        "výstupní slovník; účetnictví popisuje právě tento "
                        "hybridní průchod."
                    ),
                }
            )

        with self.cache_lock:
            self.component_attribution_cache[cache_key] = response
            self.component_attribution_cache.move_to_end(cache_key)
            while (
                len(self.component_attribution_cache)
                > COMPONENT_ATTRIBUTION_CACHE_SIZE
            ):
                self.component_attribution_cache.popitem(last=False)
        return response

    @staticmethod
    def _activation_state_diagnostics(
        source_state: np.ndarray,
        target_state: np.ndarray,
    ) -> dict[str, float]:
        source = np.asarray(source_state, dtype=np.float64)
        target = np.asarray(target_state, dtype=np.float64)
        delta = source - target
        source_norm = float(np.linalg.norm(source))
        target_norm = float(np.linalg.norm(target))
        denominator = source_norm * target_norm
        cosine = (
            float(np.dot(source, target) / denominator)
            if denominator > 1e-12
            else 0.0
        )
        return {
            "source_norm": source_norm,
            "target_norm": target_norm,
            "delta_norm": float(np.linalg.norm(delta)),
            "cosine": float(np.clip(cosine, -1.0, 1.0)),
        }

    def _activation_patching_alignment(
        self,
        source_text: str,
        target_text: str,
    ) -> tuple[list[int], list[int], list[dict[str, Any]]]:
        source_trace = self._tokenization_text_payload(
            source_text,
            label="Reference A",
        )
        target_trace = self._tokenization_text_payload(
            target_text,
            label="Příjemce B",
        )
        source_ids = [
            int(value)
            for value in source_trace["encoded_with_special_token_ids"]
        ]
        target_ids = [
            int(value)
            for value in target_trace["encoded_with_special_token_ids"]
        ]
        if len(source_ids) != len(target_ids):
            raise ValueError(
                "Kauzální přenos v první verzi vyžaduje přesně stejně "
                "dlouhé tokenové stopy včetně BOS. A má "
                f"{len(source_ids)} tokenů, B má {len(target_ids)}."
            )
        if not source_ids:
            raise ValueError("Tokenizer nevrátil žádné tokeny.")
        if len(source_ids) > MAX_ACTIVATION_PATCHING_TOKENS:
            raise ValueError(
                f"Každý prefix může mít nejvýše "
                f"{MAX_ACTIVATION_PATCHING_TOKENS} tokenů včetně BOS; "
                f"zadaný pár má {len(source_ids)}."
            )

        def ordered_tokens(trace: dict[str, Any]) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            for token in trace.get("special_prefix_tokens", []):
                result.append(
                    {
                        "token_id": int(token["token_id"]),
                        "display": str(token["display"]),
                        "source_text": token.get("source_text"),
                        "tokenizer_token": token.get("tokenizer_token"),
                        "is_special": True,
                        "role": token.get("role"),
                    }
                )
            for token in trace["tokens"]:
                result.append(
                    {
                        "token_id": int(token["token_id"]),
                        "display": str(token["display"]),
                        "source_text": token.get("source_text"),
                        "tokenizer_token": token.get("tokenizer_token"),
                        "is_special": bool(token.get("is_special", False)),
                        "role": None,
                    }
                )
            return result

        source_tokens = ordered_tokens(source_trace)
        target_tokens = ordered_tokens(target_trace)
        if (
            len(source_tokens) != len(source_ids)
            or len(target_tokens) != len(target_ids)
        ):
            raise RuntimeError(
                "Tokenizer nevrátil konzistentní mapování modelových pozic."
            )
        alignment = []
        for position, (source, target) in enumerate(
            zip(source_tokens, target_tokens)
        ):
            alignment.append(
                {
                    "position": position,
                    "source": source,
                    "target": target,
                    "same_token_id": (
                        source["token_id"] == target["token_id"]
                    ),
                    "kind": (
                        "special"
                        if source["is_special"] or target["is_special"]
                        else "shared"
                        if source["token_id"] == target["token_id"]
                        else "different"
                    ),
                }
            )
        return source_ids, target_ids, alignment

    def _activation_distribution_payload(
        self,
        logits: np.ndarray,
        probabilities: np.ndarray,
        token_a: int,
        token_b: int,
        top_k: int,
    ) -> dict[str, Any]:
        values = np.asarray(logits, dtype=np.float64)
        probs = np.asarray(probabilities, dtype=np.float64)
        ordered = np.lexsort(
            (np.arange(values.size, dtype=np.int64), -values)
        )
        top_ids = ordered[:top_k]
        top_tokens = []
        for rank, token_id_value in enumerate(top_ids, start=1):
            token_id = int(token_id_value)
            payload = self._lens_token_payload(token_id)
            payload.update(
                {
                    "rank": rank,
                    "probability": float(probs[token_id]),
                    "logit": float(values[token_id]),
                }
            )
            top_tokens.append(payload)
        top1_token_id = int(top_ids[0])
        return {
            "contrast": float(values[token_a] - values[token_b]),
            "probability_a": float(probs[token_a]),
            "probability_b": float(probs[token_b]),
            "rank_a": 1 + int(np.count_nonzero(values > values[token_a])),
            "rank_b": 1 + int(np.count_nonzero(values > values[token_b])),
            "entropy_bits": patching_entropy_bits(probs),
            "top1": {
                **self._lens_token_payload(top1_token_id),
                "probability": float(probs[top1_token_id]),
            },
            "top_tokens": top_tokens,
        }

    def _activation_patching_movers(
        self,
        target_probabilities: np.ndarray,
        patched_probabilities: np.ndarray,
        *,
        count: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        delta = np.asarray(
            patched_probabilities - target_probabilities,
            dtype=np.float64,
        )
        actual_count = min(count, delta.size)
        gain_candidates = np.argpartition(
            delta,
            delta.size - actual_count,
        )[-actual_count:]
        loss_candidates = np.argpartition(
            delta,
            actual_count - 1,
        )[:actual_count]
        gains = gain_candidates[
            np.lexsort(
                (gain_candidates, -delta[gain_candidates])
            )
        ]
        losses = loss_candidates[
            np.lexsort(
                (loss_candidates, delta[loss_candidates])
            )
        ]

        def payload(values: np.ndarray) -> list[dict[str, Any]]:
            entries = []
            for token_id_value in values:
                token_id = int(token_id_value)
                entries.append(
                    {
                        **self._lens_token_payload(token_id),
                        "delta_probability": float(delta[token_id]),
                        "target_probability": float(
                            target_probabilities[token_id]
                        ),
                        "patched_probability": float(
                            patched_probabilities[token_id]
                        ),
                    }
                )
            return entries

        return {"gains": payload(gains), "losses": payload(losses)}

    def _activation_patching_cell(
        self,
        reference: Any,
        patched_logits: np.ndarray,
        *,
        layer: int,
        position: int,
        state_diagnostics: dict[str, float],
    ) -> dict[str, Any]:
        metrics, probabilities = summarize_patch(
            reference,
            patched_logits,
        )
        top1 = self._lens_token_payload(
            int(metrics.pop("top1_token_id"))
        )
        top1["probability"] = float(metrics.pop("top1_probability"))
        return {
            "layer": layer,
            "position": position,
            **metrics,
            "matched_baselines": {
                "scope": "same_fixed_forward_batch",
                "source_contrast": float(reference.source_contrast),
                "target_contrast": float(reference.target_contrast),
                "denominator": float(reference.contrast_denominator),
                "js_source_target_bits": float(
                    reference.baseline_js_bits
                ),
            },
            "top1": top1,
            "state": state_diagnostics,
            "movers": self._activation_patching_movers(
                reference.target_probabilities,
                probabilities,
            ),
        }

    def _mock_activation_patching_logits(
        self,
        source_text: str,
        target_text: str,
        token_a: int,
        token_b: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        seed = int.from_bytes(
            hashlib.sha256(
                (
                    "activation-patching-mock-v1\0"
                    f"{source_text}\0{target_text}"
                ).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(seed)
        base = rng.normal(0.0, 0.65, size=len(MOCK_WORDS))
        source = base + rng.normal(0.0, 0.18, size=len(MOCK_WORDS))
        target = base + rng.normal(0.0, 0.18, size=len(MOCK_WORDS))
        source[token_a] += 2.4
        source[token_b] -= 0.6
        target[token_a] -= 0.6
        target[token_b] += 2.4
        return source.astype(np.float32), target.astype(np.float32)

    def _mock_activation_patching_cells(
        self,
        source_text: str,
        target_text: str,
        source_ids: list[int],
        target_ids: list[int],
        layers: list[int],
        positions: list[int],
        reference: Any,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, float | int],
    ]:
        cells: list[dict[str, Any]] = []
        controls: list[dict[str, Any]] = []
        sequence_denominator = max(1, len(source_ids) - 1)
        for layer in layers:
            layer_profile = math.exp(
                -0.5 * ((layer - 22.0) / 6.2) ** 2
            )
            for position in positions:
                position_profile = 0.18 + 0.82 * (
                    position / sequence_denominator
                )
                difference_bonus = (
                    0.18
                    if source_ids[position] != target_ids[position]
                    else 0.0
                )
                recovery = min(
                    1.25,
                    layer_profile * position_profile + difference_bonus,
                )
                patched_logits = (
                    reference.target_logits
                    + recovery
                    * (
                        reference.source_logits
                        - reference.target_logits
                    )
                )
                width = 64
                source_state = self._mock_steering_state(
                    f"{source_text}\0{position}",
                    layer,
                )[:width]
                target_state = self._mock_steering_state(
                    f"{target_text}\0{position}",
                    layer,
                )[:width]
                cells.append(
                    self._activation_patching_cell(
                        reference,
                        patched_logits,
                        layer=layer,
                        position=position,
                        state_diagnostics=(
                            self._activation_state_diagnostics(
                                source_state,
                                target_state,
                            )
                        ),
                    )
                )
            controls.append(
                {
                    "layer": layer,
                    **summarize_patching_identity(
                        reference,
                        reference.target_logits.copy(),
                    ),
                }
            )
        patch_batches = sum(
            math.ceil(
                len(positions) / ACTIVATION_PATCHING_BATCH_SIZE
            )
            for _layer in layers
        )
        batches_per_layer = math.ceil(
            len(positions) / ACTIVATION_PATCHING_BATCH_SIZE
        )
        for control in controls:
            control.update(
                {
                    "batch_count": batches_per_layer,
                    "matched_reference_used": True,
                    "source_max_abs_logit_error": 0.0,
                    "source_contrast_delta": 0.0,
                    "source_js_to_baseline_bits": 0.0,
                }
            )
        return (
            cells,
            controls,
            {
                "baseline_forward_ms": 3.0,
                "patch_forward_ms": float(patch_batches * 1.5),
                "projection_ms": float((patch_batches + 1) * 0.3),
                "forward_batches": 1 + patch_batches,
                "matched_control_variants": patch_batches * 2,
            },
        )

    def _apertus_activation_patching_baselines(
        self,
        source_ids: list[int],
        target_ids: list[int],
    ) -> dict[str, Any]:
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        if (
            getattr(self.model, "model_type", None) != "apertus"
            or not hasattr(self.model, "model")
        ):
            raise RuntimeError(
                "Kauzální přenos je implementován pouze pro Apertus."
            )
        body = self.model.model
        layers = getattr(body, "layers", None)
        if layers is None or len(layers) != 32:
            raise RuntimeError(
                "Kauzální přenos očekává 32 decoderových vrstev Apertusu."
            )
        # Quantized kernels can change their rounding path with batch shape.
        # Keep the baseline and every intervention at one fixed layout:
        # [four B patch slots, A control, B control].
        ids = mx.array(
            [target_ids] * ACTIVATION_PATCHING_BATCH_SIZE
            + [source_ids, target_ids]
        )
        hidden = body.embed_tokens(ids)
        causal_mask = create_attention_mask(hidden, None)
        pre_states: list[Any] = []
        forward_started = time.perf_counter()
        for decoder_layer in layers:
            pre_states.append(hidden)
            hidden = decoder_layer(
                hidden,
                mask=causal_mask,
                cache=None,
            )
        final_states = hidden[:, -1, :]
        mx.eval(*pre_states, final_states)
        forward_ms = (time.perf_counter() - forward_started) * 1_000

        projection_started = time.perf_counter()
        normalized = body.norm(final_states)
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is not None:
            projected = lm_head(normalized)
        else:
            as_linear = getattr(body.embed_tokens, "as_linear", None)
            if not callable(as_linear):
                raise RuntimeError(
                    "Model nemá lm_head ani vázanou výstupní projekci."
                )
            projected = as_linear(normalized)
        projected = projected.astype(mx.float32)
        mx.eval(projected)
        logits = np.asarray(projected).astype(np.float32, copy=True)
        projection_ms = (
            time.perf_counter() - projection_started
        ) * 1_000
        return {
            "body": body,
            "layers": layers,
            "pre_states": pre_states,
            "source_logits": logits[
                ACTIVATION_PATCHING_SOURCE_CONTROL_INDEX
            ],
            "target_logits": logits[
                ACTIVATION_PATCHING_TARGET_CONTROL_INDEX
            ],
            "source_control_index": (
                ACTIVATION_PATCHING_SOURCE_CONTROL_INDEX
            ),
            "target_control_index": (
                ACTIVATION_PATCHING_TARGET_CONTROL_INDEX
            ),
            "fixed_batch_size": ACTIVATION_PATCHING_FIXED_BATCH_SIZE,
            "forward_ms": forward_ms,
            "projection_ms": projection_ms,
        }

    def _apertus_activation_patching_cells(
        self,
        baseline: dict[str, Any],
        layers: list[int],
        positions: list[int],
        reference: Any,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, float | int],
    ]:
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        body = baseline["body"]
        decoder_layers = baseline["layers"]
        pre_states = baseline["pre_states"]
        cells: list[dict[str, Any]] = []
        controls: list[dict[str, Any]] = []
        patch_forward_ms = 0.0
        projection_ms = 0.0
        forward_batches = 1
        matched_control_variants = 0
        source_control_index = int(baseline["source_control_index"])
        target_control_index = int(baseline["target_control_index"])

        for layer in layers:
            source_state = pre_states[layer - 1][source_control_index]
            target_state = pre_states[layer - 1][target_control_index]
            source_array = np.asarray(
                source_state.astype(mx.float32)
            )
            target_array = np.asarray(
                target_state.astype(mx.float32)
            )
            state_diagnostics = {
                position: self._activation_state_diagnostics(
                    source_array[position],
                    target_array[position],
                )
                for position in positions
            }
            layer_target_controls: list[dict[str, float]] = []
            layer_source_controls: list[dict[str, float]] = []
            for chunk_start in range(
                0,
                len(positions),
                ACTIVATION_PATCHING_BATCH_SIZE,
            ):
                chunk = positions[
                    chunk_start:
                    chunk_start + ACTIVATION_PATCHING_BATCH_SIZE
                ]
                run_started = time.perf_counter()
                padding = (
                    ACTIVATION_PATCHING_BATCH_SIZE - len(chunk)
                )
                slot_positions = chunk + [0] * padding
                slot_validity = [1] * len(chunk) + [0] * padding
                position_values = mx.array(slot_positions)
                sequence_positions = mx.arange(target_state.shape[0])
                mask = (
                    position_values[:, None]
                    == sequence_positions[None, :]
                )[:, :, None].astype(target_state.dtype) * mx.array(
                    slot_validity
                )[:, None, None].astype(target_state.dtype)
                delta = source_state - target_state
                target_batch = mx.broadcast_to(
                    target_state[None, :, :],
                    (
                        ACTIVATION_PATCHING_BATCH_SIZE,
                        target_state.shape[0],
                        target_state.shape[1],
                    ),
                )
                patch_slots = target_batch + mask * delta[None, :, :]
                patched = mx.concatenate(
                    (
                        patch_slots,
                        source_state[None, :, :],
                        target_state[None, :, :],
                    ),
                    axis=0,
                )
                downstream_mask = create_attention_mask(patched, None)
                for decoder_layer in decoder_layers[layer - 1:]:
                    patched = decoder_layer(
                        patched,
                        mask=downstream_mask,
                        cache=None,
                    )
                final_states = patched[:, -1, :]
                mx.eval(final_states)
                patch_forward_ms += (
                    time.perf_counter() - run_started
                ) * 1_000
                forward_batches += 1

                projection_started = time.perf_counter()
                normalized = body.norm(final_states)
                lm_head = getattr(self.model, "lm_head", None)
                if lm_head is not None:
                    projected = lm_head(normalized)
                else:
                    as_linear = getattr(
                        body.embed_tokens,
                        "as_linear",
                        None,
                    )
                    if not callable(as_linear):
                        raise RuntimeError(
                            "Model nemá lm_head ani vázanou výstupní "
                            "projekci."
                        )
                    projected = as_linear(normalized)
                projected = projected.astype(mx.float32)
                mx.eval(projected)
                logits = np.asarray(projected).astype(
                    np.float32,
                    copy=True,
                )
                projection_ms += (
                    time.perf_counter() - projection_started
                ) * 1_000

                matched_reference = build_patching_reference(
                    logits[source_control_index],
                    logits[target_control_index],
                    reference.token_a,
                    reference.token_b,
                )
                for row, position in enumerate(chunk):
                    cells.append(
                        self._activation_patching_cell(
                            matched_reference,
                            logits[row],
                            layer=layer,
                            position=position,
                            state_diagnostics=state_diagnostics[position],
                        )
                    )
                layer_target_controls.append(
                    summarize_patching_identity(
                        reference,
                        logits[target_control_index],
                    )
                )
                layer_source_controls.append(
                    summarize_patching_recomputation(
                        reference.source_logits,
                        logits[source_control_index],
                        reference.token_a,
                        reference.token_b,
                    )
                )
                matched_control_variants += 2

            def largest_absolute(
                rows: list[dict[str, float]],
                key: str,
            ) -> float:
                return float(
                    max(
                        rows,
                        key=lambda row: abs(float(row[key])),
                    )[key]
                )

            controls.append(
                {
                    "layer": layer,
                    "batch_count": len(layer_target_controls),
                    "matched_reference_used": True,
                    "max_abs_logit_error": max(
                        float(row["max_abs_logit_error"])
                        for row in layer_target_controls
                    ),
                    "contrast_delta": largest_absolute(
                        layer_target_controls,
                        "contrast_delta",
                    ),
                    "js_to_target_bits": max(
                        float(row["js_to_target_bits"])
                        for row in layer_target_controls
                    ),
                    "source_max_abs_logit_error": max(
                        float(row["max_abs_logit_error"])
                        for row in layer_source_controls
                    ),
                    "source_contrast_delta": largest_absolute(
                        layer_source_controls,
                        "contrast_delta",
                    ),
                    "source_js_to_baseline_bits": max(
                        float(row["js_to_expected_bits"])
                        for row in layer_source_controls
                    ),
                }
            )

        return (
            cells,
            controls,
            {
                "baseline_forward_ms": float(baseline["forward_ms"]),
                "patch_forward_ms": patch_forward_ms,
                "projection_ms": (
                    float(baseline["projection_ms"]) + projection_ms
                ),
                "forward_batches": forward_batches,
                "matched_control_variants": matched_control_variants,
            },
        )

    @staticmethod
    def _activation_patching_extrema(
        cells: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any] | None]:
        def compact(cell: dict[str, Any] | None) -> dict[str, Any] | None:
            if cell is None:
                return None
            return {
                "layer": int(cell["layer"]),
                "position": int(cell["position"]),
                "contrast_recovery": float(cell["contrast_recovery"]),
                "distribution_recovery": (
                    float(cell["distribution_recovery"])
                    if cell["distribution_recovery"] is not None
                    else None
                ),
                "delta_contrast_from_target": float(
                    cell["delta_contrast_from_target"]
                ),
            }

        distribution_cells = [
            cell
            for cell in cells
            if cell["distribution_recovery"] is not None
        ]
        return {
            "strongest_toward_source": compact(
                max(
                    cells,
                    key=lambda cell: cell["contrast_recovery"],
                    default=None,
                )
            ),
            "strongest_away_from_source": compact(
                min(
                    cells,
                    key=lambda cell: cell["contrast_recovery"],
                    default=None,
                )
            ),
            "best_distribution_recovery": compact(
                max(
                    distribution_cells,
                    key=lambda cell: cell["distribution_recovery"],
                    default=None,
                )
            ),
        }

    def _activation_patching_local(
        self,
        source_text: str,
        target_text: str,
        token_a_query: str,
        token_b_query: str,
        layers: list[int],
        positions: list[int],
        top_k: int,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        tokenization_started = time.perf_counter()
        if source_text == target_text:
            raise ValueError(
                "Reference A a příjemce B musí být dva odlišné prefixy."
            )
        token_a, token_a_payload = self._component_attribution_token(
            token_a_query
        )
        token_b, token_b_payload = self._component_attribution_token(
            token_b_query
        )
        if token_a == token_b:
            raise ValueError(
                "Obě strany výstupního kontrastu označují tentýž token."
            )
        source_ids, target_ids, alignment = (
            self._activation_patching_alignment(
                source_text,
                target_text,
            )
        )
        if (
            not layers
            or len(layers) != len(set(layers))
            or any(not 1 <= layer <= 32 for layer in layers)
        ):
            raise ValueError(
                "layers musí být neprázdný seznam unikátních vrstev L1–L32."
            )
        if (
            not positions
            or len(positions) != len(set(positions))
            or len(positions) > MAX_ACTIVATION_PATCHING_POSITIONS
            or any(
                not 0 <= position < len(source_ids)
                for position in positions
            )
        ):
            raise ValueError(
                "positions musí být neprázdný seznam nejvýše "
                f"{MAX_ACTIVATION_PATCHING_POSITIONS} unikátních "
                "modelových pozic v rozsahu tokenové stopy."
            )
        if len(layers) * len(positions) > MAX_ACTIVATION_PATCHING_CELLS:
            raise ValueError(
                "Požadovaná heatmapa má "
                f"{len(layers) * len(positions)} buněk; limit je "
                f"{MAX_ACTIVATION_PATCHING_CELLS}. Zúž vrstvy nebo pozice."
            )
        layers = sorted(layers)
        positions = sorted(positions)
        tokenization_ms = (
            time.perf_counter() - tokenization_started
        ) * 1_000

        cache_key = (
            self.model_generation,
            source_text,
            target_text,
            token_a,
            token_b,
            tuple(layers),
            tuple(positions),
            top_k,
        )
        with self.cache_lock:
            cached_response = self.activation_patching_cache.get(cache_key)
            if cached_response is not None:
                self.activation_patching_cache.move_to_end(cache_key)
                response = dict(cached_response)
                response["cached"] = True
                response["timing_ms"] = {
                    "tokenization": tokenization_ms,
                    "baseline_forward": 0.0,
                    "patch_forward": 0.0,
                    "projection": 0.0,
                    "summary": 0.0,
                    "total": (
                        time.perf_counter() - total_started
                    )
                    * 1_000,
                }
                return response

        if self.mock:
            source_logits, target_logits = (
                self._mock_activation_patching_logits(
                    source_text,
                    target_text,
                    token_a,
                    token_b,
                )
            )
            reference = build_patching_reference(
                source_logits,
                target_logits,
                token_a,
                token_b,
            )
            cells, controls, execution_timing = (
                self._mock_activation_patching_cells(
                    source_text,
                    target_text,
                    source_ids,
                    target_ids,
                    layers,
                    positions,
                    reference,
                )
            )
        else:
            baseline = self._apertus_activation_patching_baselines(
                source_ids,
                target_ids,
            )
            reference = build_patching_reference(
                baseline["source_logits"],
                baseline["target_logits"],
                token_a,
                token_b,
            )
            cells, controls, execution_timing = (
                self._apertus_activation_patching_cells(
                    baseline,
                    layers,
                    positions,
                    reference,
                )
            )

        summary_started = time.perf_counter()
        selected = set(positions)
        for entry in alignment:
            entry["selected"] = entry["position"] in selected
        source_payload = self._activation_distribution_payload(
            reference.source_logits,
            reference.source_probabilities,
            token_a,
            token_b,
            top_k,
        )
        target_payload = self._activation_distribution_payload(
            reference.target_logits,
            reference.target_probabilities,
            token_a,
            token_b,
            top_k,
        )
        max_identity_error = max(
            (
                abs(float(control["max_abs_logit_error"]))
                for control in controls
            ),
            default=0.0,
        )
        max_source_control_error = max(
            (
                abs(float(control["source_max_abs_logit_error"]))
                for control in controls
            ),
            default=0.0,
        )
        max_control_error = max(
            max_identity_error,
            max_source_control_error,
        )
        warnings: list[dict[str, Any]] = [
            {
                "code": "sufficiency_not_necessity",
                "message": (
                    "Patch ukazuje kauzální dostatečnost transplantovaného "
                    "stavu pro tento průchod, nikoli nezbytnost nebo "
                    "jedinečnou přirozenou cestu."
                ),
            },
            {
                "code": "full_residual_bundle",
                "message": (
                    "Přenáší se celý resid_pre vektor. Buňka proto není "
                    "čistým atomem jednoho významu."
                ),
            },
            {
                "code": "position_alignment_only",
                "message": (
                    "Pozice jsou zarovnány přesně podle modelového indexu; "
                    "výpočet nepředpokládá slovní ekvivalenci rozdílných "
                    "tokenů."
                ),
            },
        ]
        if (
            max_control_error
            > ACTIVATION_PATCHING_IDENTITY_LOGIT_TOLERANCE
        ):
            warnings.append(
                {
                    "code": "identity_control_drift",
                    "message": (
                        "Kontrolní A/B přepočet se od hlavní baseline liší "
                        "více než provozní tolerance. Recovery každé buňky "
                        "se přesto počítá vůči kontrolám z téhož pevného "
                        "batche; odchylka se týká srovnání s hlavním "
                        "souhrnem."
                    ),
                    "max_abs_logit_error": max_control_error,
                    "tolerance": (
                        ACTIVATION_PATCHING_IDENTITY_LOGIT_TOLERANCE
                    ),
                }
            )
        if self.is_quantized():
            warnings.append(
                {
                    "code": "quantized_body",
                    "message": (
                        "Intervence proběhla v aktivním kvantovaném nebo "
                        "hybridním modelu a popisuje právě jeho výpočet."
                    ),
                }
            )
        summary_ms = (
            time.perf_counter() - summary_started
        ) * 1_000
        total_ms = (time.perf_counter() - total_started) * 1_000
        response = {
            "analysis_kind": "activation_patching",
            "method": {
                "kind": "resid_pre_interchange_intervention",
                "hook": "pre_decoder_block_residual",
                "direction": "source_A_into_recipient_B",
                "contrast_definition": "logit(A) - logit(B)",
                "interpretation": "causal_sufficiency_not_necessity",
                "distribution_recovery_definition": (
                    "1 - JS(patched, source_A) / JS(recipient_B, source_A)"
                ),
                "contrast_recovery_definition": (
                    "(d_patched - d_B) / (d_A - d_B)"
                ),
                "cell_reference": (
                    "matched_A_B_controls_in_same_fixed_forward_batch"
                ),
            },
            "source_text": source_text,
            "target_text": target_text,
            "contrast": {
                "definition": "logit(A) - logit(B)",
                "a": token_a_payload,
                "b": token_b_payload,
                "denominator": float(reference.contrast_denominator),
            },
            "tokenization": {
                "alignment_kind": "exact_model_position",
                "equal_token_count": True,
                "token_count_including_special": len(source_ids),
                "source_token_ids": source_ids,
                "target_token_ids": target_ids,
                "alignment": alignment,
                "selected_positions": positions,
            },
            "layers": layers,
            "baselines": {
                "source": source_payload,
                "target": target_payload,
                "js_source_target_bits": float(
                    reference.baseline_js_bits
                ),
            },
            "cells": cells,
            "identity_controls": controls,
            "identity_control_summary": {
                "max_abs_logit_error": max_identity_error,
                "source_max_abs_logit_error": (
                    max_source_control_error
                ),
                "max_any_abs_logit_error": max_control_error,
                "tolerance": (
                    ACTIVATION_PATCHING_IDENTITY_LOGIT_TOLERANCE
                ),
                "passed": (
                    max_control_error
                    <= ACTIVATION_PATCHING_IDENTITY_LOGIT_TOLERANCE
                ),
                "matched_controls_used_for_cells": True,
            },
            "extrema": self._activation_patching_extrema(cells),
            "execution": {
                "mode": "cache_free_batched_interchange_intervention",
                "forward_batches": int(
                    execution_timing["forward_batches"]
                ),
                "baseline_sequences": 2,
                "baseline_batch_rows": (
                    ACTIVATION_PATCHING_FIXED_BATCH_SIZE
                ),
                "patched_variants": len(cells),
                "identity_control_variants": len(controls),
                "matched_control_variants": int(
                    execution_timing["matched_control_variants"]
                ),
                "patch_slots_per_batch": (
                    ACTIVATION_PATCHING_BATCH_SIZE
                ),
                "batch_size": ACTIVATION_PATCHING_BATCH_SIZE,
                "fixed_total_batch_size": (
                    ACTIVATION_PATCHING_FIXED_BATCH_SIZE
                ),
                "matched_source_target_controls_per_batch": True,
                "downstream_recomputed": True,
                "kv_cache_used": False,
                "global_cache_mutated": False,
                "generation": False,
                "teacher_forced_next_token": True,
            },
            "warnings": warnings,
            "model": self.model_name,
            "model_id": self.active_model_id,
            "model_label": self.model_label(
                self.active_model_id,
                self.active_model_id,
            ),
            "model_generation": self.model_generation,
            "server_instance_id": self.server_instance_id,
            "provenance": {
                "runtime": dict(RUNTIME_PROVENANCE),
                "model": self._model_provenance_unlocked(),
            },
            "cached": False,
            "timing_ms": {
                "tokenization": tokenization_ms,
                "baseline_forward": float(
                    execution_timing["baseline_forward_ms"]
                ),
                "patch_forward": float(
                    execution_timing["patch_forward_ms"]
                ),
                "projection": float(
                    execution_timing["projection_ms"]
                ),
                "summary": summary_ms,
                "total": total_ms,
            },
            "latency_ms": total_ms,
        }
        with self.cache_lock:
            self.activation_patching_cache[cache_key] = response
            self.activation_patching_cache.move_to_end(cache_key)
            while (
                len(self.activation_patching_cache)
                > ACTIVATION_PATCHING_CACHE_SIZE
            ):
                self.activation_patching_cache.popitem(last=False)
        return response

    def _validate_request_identity_unlocked(
        self,
        expected_generation: int | None,
        expected_server_instance_id: str | None,
        expected_model_id: str | None = None,
    ) -> None:
        if (
            expected_server_instance_id is not None
            and expected_server_instance_id != self.server_instance_id
        ):
            raise ModelChangedError(
                "Server byl od vytvoření této zahrady restartován."
            )
        if (
            expected_generation is not None
            and expected_generation != self.model_generation
        ):
            raise ModelChangedError(
                "Model se od vytvoření této zahrady změnil."
            )
        if (
            expected_model_id is not None
            and expected_model_id != self.active_model_id
        ):
            raise ModelChangedError(
                "Požadavek míří na jiný než právě aktivní model."
            )

    def _submit_ready(
        self,
        operation: str,
        arguments: tuple[Any, ...],
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
        expected_model_id: str | None = None,
    ) -> dict[str, Any]:
        done = threading.Event()
        result: dict[str, Any] = {}
        with self.state_lock:
            if self.status != "ready":
                raise RuntimeError("Model ještě není připraven.")
            self._validate_request_identity_unlocked(
                expected_generation,
                expected_server_instance_id,
                expected_model_id,
            )
            if self.outstanding_operation is not None:
                raise RuntimeError(
                    "Model právě dokončuje předchozí výpočet; další "
                    "požadavek zatím nebyl zařazen."
                )
            self.outstanding_operation = operation
            self.outstanding_operation_started_monotonic = time.monotonic()
            try:
                self.work_queue.put_nowait(
                    (operation, arguments, done, result)
                )
            except queue.Full as exc:
                self.outstanding_operation = None
                self.outstanding_operation_started_monotonic = None
                raise RuntimeError(
                    "Výpočetní fronta je plná; počkej na dokončení "
                    "předchozích požadavků."
                ) from exc
        timeout_seconds = (
            STEERING_CONTROL_JOB_TIMEOUT_SECONDS
            if operation
            in {
                "steering_sham_control",
                "steering_layer_sweep",
                "activation_patching",
            }
            else DEFAULT_MODEL_JOB_TIMEOUT_SECONDS
        )
        if not done.wait(timeout=timeout_seconds):
            raise RuntimeError(
                "Výpočet modelu překročil časový limit "
                f"{timeout_seconds // 60} minut."
            )
        if "error" in result:
            raise result["error"]
        return result["value"]

    def next_tokens(
        self,
        text: str,
        temperature: float,
        top_k: int,
        provided_token_ids: list[int] | None = None,
        pinned_token_ids: list[int] | None = None,
        cache_only: bool = False,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        if self.worker_thread is None:
            # Useful for isolated unit checks; the running app always uses start().
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                )
            return self._next_tokens_local(
                text,
                temperature,
                top_k,
                provided_token_ids,
                pinned_token_ids,
                cache_only,
            )
        return self._submit_ready(
            "next",
            (
                text,
                temperature,
                top_k,
                provided_token_ids,
                pinned_token_ids,
                cache_only,
            ),
            expected_generation,
            expected_server_instance_id,
        )

    def search_tokens(
        self,
        text: str,
        query: str,
        temperature: float,
        provided_token_ids: list[int] | None = None,
        limit: int = SEARCH_RESULT_LIMIT,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (text, query, temperature, provided_token_ids, limit)
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                )
            return self._search_tokens_local(*arguments)
        return self._submit_ready(
            "search",
            arguments,
            expected_generation,
            expected_server_instance_id,
        )

    def lookup_tokens(
        self,
        text: str,
        requested_token_ids: list[int],
        temperature: float,
        provided_token_ids: list[int] | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (
            text,
            requested_token_ids,
            temperature,
            provided_token_ids,
        )
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                )
            return self._lookup_tokens_local(*arguments)
        return self._submit_ready(
            "lookup",
            arguments,
            expected_generation,
            expected_server_instance_id,
        )

    def compare_distributions(
        self,
        left_text: str,
        left_token_ids: list[int],
        right_text: str,
        right_token_ids: list[int],
        temperature: float,
        mover_count: int,
        overlap_k: int,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (
            left_text,
            left_token_ids,
            right_text,
            right_token_ids,
            temperature,
            mover_count,
            overlap_k,
        )
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                )
            return self._compare_distributions_local(*arguments)
        return self._submit_ready(
            "compare",
            arguments,
            expected_generation,
            expected_server_instance_id,
        )

    def steering(
        self,
        contrast_pairs: list[dict[str, Any]],
        test_text: str,
        layer: int,
        alpha: float,
        temperature: float,
        top_k: int,
        mover_count: int,
        overlap_k: int,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (
            contrast_pairs,
            test_text,
            layer,
            alpha,
            temperature,
            top_k,
            mover_count,
            overlap_k,
        )
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._steering_local(*arguments)
        return self._submit_ready(
            "steering",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def steering_trajectory(
        self,
        contrast_pairs: list[dict[str, Any]],
        test_text: str,
        layer: int,
        alpha_magnitude: float,
        temperature: float,
        max_new_tokens: int,
        generation_mode: str,
        seed: int,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (
            contrast_pairs,
            test_text,
            layer,
            alpha_magnitude,
            temperature,
            max_new_tokens,
            generation_mode,
            seed,
        )
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._steering_trajectory_local(*arguments)
        return self._submit_ready(
            "steering_trajectory",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def steering_aligned_path(
        self,
        contrast_pairs: list[dict[str, Any]],
        test_text: str,
        continuation_text: str,
        layer: int,
        alpha_magnitude: float,
        temperature: float,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (
            contrast_pairs,
            test_text,
            continuation_text,
            layer,
            alpha_magnitude,
            temperature,
        )
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._steering_aligned_path_local(*arguments)
        return self._submit_ready(
            "steering_aligned_path",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def steering_phrase_score(
        self,
        contrast_pairs: list[dict[str, Any]],
        test_text: str,
        continuation_pairs: list[dict[str, Any]],
        layer: int,
        alpha_magnitude: float,
        temperature: float,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (
            contrast_pairs,
            test_text,
            continuation_pairs,
            layer,
            alpha_magnitude,
            temperature,
        )
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._steering_phrase_score_local(*arguments)
        return self._submit_ready(
            "steering_phrase_score",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def steering_sham_control(
        self,
        contrast_pairs: list[dict[str, Any]],
        test_text: str,
        continuation_pairs: list[dict[str, Any]],
        layer: int,
        alpha_magnitude: float,
        temperature: float,
        sample_count: int,
        seed: int,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (
            contrast_pairs,
            test_text,
            continuation_pairs,
            layer,
            alpha_magnitude,
            temperature,
            sample_count,
            seed,
        )
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._steering_sham_control_local(*arguments)
        return self._submit_ready(
            "steering_sham_control",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def steering_layer_sweep(
        self,
        contrast_pairs: list[dict[str, Any]],
        test_text: str,
        continuation_pairs: list[dict[str, Any]],
        layers: list[int],
        reference_layer: int,
        alpha_magnitude: float,
        temperature: float,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (
            contrast_pairs,
            test_text,
            continuation_pairs,
            layers,
            reference_layer,
            alpha_magnitude,
            temperature,
        )
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._steering_layer_sweep_local(*arguments)
        return self._submit_ready(
            "steering_layer_sweep",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def lens(
        self,
        text: str,
        top_k: int,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (text, top_k)
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._lens_local(*arguments)
        return self._submit_ready(
            "lens",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def text_surprisal(
        self,
        text: str,
        top_k: int,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (text, top_k)
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._text_surprisal_local(*arguments)
        return self._submit_ready(
            "text_surprisal",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def _predictive_stylometry_local(
        self,
        text: str,
        *,
        capture_window: Any = None,
    ) -> dict[str, Any]:
        if self.predictive_detector_engine is None:
            self.predictive_detector_engine = self.experimental_addons.create_detector() if self.experimental_addons is not None else PredictiveDetector(
                model_dir=self.model_paths["bf16"],
            )
        def score_window(window: str) -> dict[str, Any]:
            response = self._text_surprisal_local(window, 1)
            if capture_window is not None:
                capture_window(window, response)
            return response

        return self.predictive_detector_engine.analyze(
            text,
            score_window=score_window,
            model_id=self.active_model_id,
            model_generation=self.model_generation,
            server_instance_id=self.server_instance_id,
        )

    def predictive_stylometry(
        self,
        text: str,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (text,)
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._predictive_stylometry_local(*arguments)
        return self._submit_ready(
            "predictive_stylometry",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def tokenization(
        self,
        text_a: str,
        text_b: str | None = None,
    ) -> dict[str, Any]:
        # Tokenization is deterministic and does not use model weights.  The
        # state lock prevents a concurrent model switch from releasing the
        # shared tokenizer while this short analysis is in progress.
        with self.state_lock:
            if (
                self.status != "ready"
                or (self.tokenizer is None and not self.mock)
            ):
                raise RuntimeError("Tokenizer ještě není připraven.")
            return self._tokenization_local(text_a, text_b)

    def embeddings(
        self,
        contexts: list[dict[str, Any]],
        neighbor_count: int,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
        analysis_mode: str = "token",
    ) -> dict[str, Any]:
        arguments = (contexts, neighbor_count, analysis_mode)
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._embeddings_local(*arguments)
        return self._submit_ready(
            "embeddings",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def embedding_neighborhood(
        self,
        token_id: int,
        neighbor_count: int,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (token_id, neighbor_count)
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._embedding_neighborhood_local(*arguments)
        return self._submit_ready(
            "embedding_neighborhood",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def continuation_surprisal(
        self,
        prefix: str,
        continuation: str,
        top_k: int,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (prefix, continuation, top_k)
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._continuation_surprisal_local(*arguments)
        return self._submit_ready(
            "continuation_surprisal",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def text_surprisal_comparison(
        self,
        text_a: str,
        text_b: str,
        label_a: str,
        label_b: str,
        top_k: int,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (text_a, text_b, label_a, label_b, top_k)
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._text_surprisal_comparison_local(*arguments)
        return self._submit_ready(
            "text_surprisal_comparison",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def context_contrast(
        self,
        prefix_a: str,
        prefix_b: str,
        continuation: str,
        top_k: int,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (prefix_a, prefix_b, continuation, top_k)
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._context_contrast_local(*arguments)
        return self._submit_ready(
            "context_contrast",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def tuned_lens(
        self,
        text: str,
        top_k: int,
        expected_artifact_id: str | None = None,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (text, top_k, expected_artifact_id)
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._tuned_lens_local(*arguments)
        return self._submit_ready(
            "tuned_lens",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def jacobian_lens(
        self,
        text: str,
        top_k: int,
        source_layers: list[int] | None = None,
        expected_artifact_id: str | None = None,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (
            text,
            top_k,
            source_layers,
            expected_artifact_id,
        )
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._jacobian_lens_local(*arguments)
        return self._submit_ready(
            "jacobian_lens",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def component_attribution(
        self,
        text: str,
        token_a: str,
        token_b: str,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (text, token_a, token_b)
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._component_attribution_local(*arguments)
        return self._submit_ready(
            "component_attribution",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )

    def activation_patching(
        self,
        source_text: str,
        target_text: str,
        token_a: str,
        token_b: str,
        layers: list[int],
        positions: list[int],
        top_k: int,
        expected_model_id: str | None = None,
        expected_generation: int | None = None,
        expected_server_instance_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = (
            source_text,
            target_text,
            token_a,
            token_b,
            layers,
            positions,
            top_k,
        )
        if self.worker_thread is None:
            with self.state_lock:
                if self.status != "ready":
                    raise RuntimeError("Model ještě není připraven.")
                self._validate_request_identity_unlocked(
                    expected_generation,
                    expected_server_instance_id,
                    expected_model_id,
                )
            return self._activation_patching_local(*arguments)
        return self._submit_ready(
            "activation_patching",
            arguments,
            expected_generation,
            expected_server_instance_id,
            expected_model_id,
        )


class GardenHandler(BaseHTTPRequestHandler):
    server_version = SERVER_VERSION

    @property
    def state(self) -> ModelState:
        return self.server.state  # type: ignore[attr-defined]

    @property
    def experiment_store(self) -> ExperimentStore:
        return self.server.experiment_store  # type: ignore[attr-defined]

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[garden] {self.address_string()} {format_string % args}")

    def send_json(
        self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_download(
        self,
        body: bytes,
        *,
        content_type: str,
        filename: str,
    ) -> None:
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", content_type)
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="{filename}"',
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_experiment_error(
        self,
        error: Exception,
        status: HTTPStatus,
        *,
        code: str,
    ) -> None:
        self.send_json(
            {
                "code": code,
                "error": str(error),
            },
            status,
        )

    @staticmethod
    def _normalized_host_port(
        authority: str,
        scheme: str,
    ) -> tuple[str, int] | None:
        try:
            parsed = urlparse(f"//{authority}")
            hostname = (parsed.hostname or "").rstrip(".").casefold()
            if not hostname or parsed.username is not None:
                return None
            port = parsed.port
        except ValueError:
            return None
        if port is None:
            port = 443 if scheme == "https" else 80
        return hostname, port

    def _client_is_loopback(self) -> bool:
        raw_address = str(self.client_address[0]).split("%", 1)[0]
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError:
            return False
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        return address.is_loopback

    @staticmethod
    def _hostname_is_loopback(hostname: str) -> bool:
        normalized = hostname.rstrip(".").casefold()
        if normalized == "localhost" or normalized.endswith(".localhost"):
            return True
        try:
            address = ipaddress.ip_address(normalized.split("%", 1)[0])
        except ValueError:
            return False
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        return address.is_loopback

    def _request_host_is_loopback(self) -> bool:
        authority = self._normalized_host_port(
            self.headers.get("Host", ""),
            "http",
        )
        return bool(authority and self._hostname_is_loopback(authority[0]))

    def require_safe_archive_read(self) -> bool:
        """Keep plaintext archive records behind an explicit loopback Host."""
        if self._client_is_loopback() and self._request_host_is_loopback():
            return True
        self.send_json(
            {
                "code": "archive_host_forbidden",
                "error": "Lokální archiv je dostupný jen přes loopback adresu.",
            },
            HTTPStatus.FORBIDDEN,
        )
        return False

    def require_safe_mutation_origin(self) -> bool:
        """Allow same-origin browser requests and direct loopback API clients."""
        if not self._client_is_loopback() or not self._request_host_is_loopback():
            self.send_json(
                {
                    "code": "origin_forbidden",
                    "error": "Lokální API je povoleno jen přes loopback adresu.",
                },
                HTTPStatus.FORBIDDEN,
            )
            return False
        fetch_site = self.headers.get("Sec-Fetch-Site", "").strip().casefold()
        if fetch_site and fetch_site not in {"same-origin", "none"}:
            self.send_json(
                {
                    "code": "origin_forbidden",
                    "error": "Lokální API je povoleno jen ze stejného originu.",
                },
                HTTPStatus.FORBIDDEN,
            )
            return False

        origin = self.headers.get("Origin")
        if origin:
            parsed_origin = None
            origin_scheme = ""
            try:
                parsed_origin = urlparse(origin)
                origin_scheme = parsed_origin.scheme
                origin_authority = self._normalized_host_port(
                    parsed_origin.netloc,
                    origin_scheme,
                )
            except ValueError:
                origin_authority = None
            host_authority = self._normalized_host_port(
                self.headers.get("Host", ""),
                origin_scheme or "http",
            )
            if (
                parsed_origin is None
                or origin_scheme not in {"http", "https"}
                or origin_authority is None
                or host_authority is None
                or origin_authority != host_authority
                or parsed_origin.username is not None
                or parsed_origin.password is not None
            ):
                self.send_json(
                    {
                        "code": "origin_forbidden",
                        "error": (
                            "Lokální API je povoleno jen ze stejného "
                            "originu."
                        ),
                    },
                    HTTPStatus.FORBIDDEN,
                )
                return False
            return True

        if fetch_site == "same-origin" or self._client_is_loopback():
            return True
        self.send_json(
            {
                "code": "origin_forbidden",
                "error": (
                    "Požadavek bez hlavičky Origin je povolen jen z loopbacku."
                ),
            },
            HTTPStatus.FORBIDDEN,
        )
        return False

    def send_html_file(self, path: Path, missing_message: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, missing_message)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_static_file(
        self,
        path: Path,
        content_type: str,
        missing_message: str,
    ) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, missing_message)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_index(self) -> None:
        self.send_html_file(INDEX_FILE, "Chybí static/index.html")

    @staticmethod
    def _single_query_value(
        query: dict[str, list[str]],
        name: str,
    ) -> str | None:
        values = query.get(name)
        if values is None:
            return None
        if len(values) != 1:
            raise ExperimentValidationError(
                f"Parametr {name} smí být uveden nejvýše jednou."
            )
        return values[0]

    def _experiment_query(
        self,
        parsed_url: Any,
    ) -> tuple[str | None, dict[str, list[str]]]:
        query = parse_qs(parsed_url.query, keep_blank_values=True)
        kind = self._single_query_value(query, "kind")
        if kind == "":
            raise ExperimentValidationError(
                "Parametr kind nesmí být prázdný."
            )
        return kind, query

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        if path in {"/sprava-modelu", "/sprava-modelu/", "/model-admin"}:
            self.send_html_file(APP_DIR / "static/model-admin.html", "Missing model-admin.html")
            return
        if path in {"/static/model-admin.js", "/static/model-admin.css"}:
            self.send_static_file(APP_DIR / path.removeprefix("/"),
                                  "text/javascript; charset=utf-8" if path.endswith(".js") else "text/css; charset=utf-8", "Missing model administration asset")
            return
        if path in {"/api/model-admin", "/api/model-admin/addons"} or path.startswith("/api/model-admin/jobs/"):
            if not self.require_safe_archive_read():
                return
            try:
                admin = self.server.model_admin
                self.send_json(admin.snapshot() if path == "/api/model-admin" else admin.addon_snapshot() if path == "/api/model-admin/addons" else admin.job_status(path.rsplit("/",1)[-1]))
            except (AdminError, AddonError) as exc:
                self.send_json({"code":exc.code,"error":str(exc),"details":exc.details}, HTTPStatus.NOT_FOUND if exc.code == "unknown_job" else HTTPStatus.CONFLICT)
            except (OSError, ValueError) as exc:
                self.send_json({"code":"registry_unavailable","error":str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if path in {"/porovnani-modelu", "/porovnani-modelu/", "/model-comparison"}:
            self.send_html_file(MODEL_COMPARISON_FILE, "Chybí static/model-comparison.html")
            return
        if path == "/static/model-comparison.js":
            self.send_static_file(MODEL_COMPARISON_JS_FILE, "text/javascript; charset=utf-8", "Chybí model-comparison.js")
            return
        if path == "/static/steering-comparison.js":
            self.send_static_file(STEERING_COMPARISON_JS_FILE, "text/javascript; charset=utf-8", "Chybí steering-comparison.js")
            return
        if path == "/api/model-comparison/examples":
            self.send_json(json.loads((APP_DIR / "model_comparison_examples.json").read_text()))
            return
        if path.startswith("/api/model-comparison/jobs/"):
            if not self.require_safe_archive_read():
                return
            try:
                self.send_json(self.state.model_comparison_status(path.rsplit("/", 1)[-1]))
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        if path.startswith("/api/detector-readers/jobs/"):
            if not self.require_safe_archive_read():
                return
            try:
                self.send_json(self.state.detector_reader_status(path.rsplit("/", 1)[-1]))
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        if path == "/favicon.ico":
            # Chrome requests a favicon even when the local workbench does not
            # declare one. Returning an empty success keeps the console free of
            # a meaningless 404 without introducing another static artefact.
            self.send_response(HTTPStatus.NO_CONTENT.value)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if path in {"/", "/index.html"}:
            if self.state.status == "setup_required":
                self.send_html_file(APP_DIR / "static/model-admin.html", "Missing model-admin.html")
            else:
                self.send_index()
            return
        if path in {
            "/jak-to-cist",
            "/jak-to-cist/",
            "/jak-to-cist.html",
            "/guide.html",
        }:
            self.send_html_file(GUIDE_FILE, "Chybí static/guide.html")
            return
        if path in {"/vrstvy", "/vrstvy/", "/lens", "/lens.html"}:
            self.send_html_file(LENS_FILE, "Chybí static/lens.html")
            return
        if path in {
            "/tuned-lens",
            "/tuned-lens/",
            "/tuned",
            "/tuned.html",
        }:
            self.send_html_file(
                TUNED_LENS_FILE,
                "Chybí static/tuned.html",
            )
            return
        if path in {
            "/j-lens",
            "/j-lens/",
            "/jacobian-lens",
            "/jacobian-lens/",
            "/jlens.html",
        }:
            self.send_html_file(
                JACOBIAN_LENS_FILE,
                "Chybí static/jlens.html",
            )
            return
        if path in {
            "/atribuce",
            "/atribuce/",
            "/component-attribution",
            "/component-attribution/",
            "/attribution.html",
        }:
            self.send_html_file(
                COMPONENT_ATTRIBUTION_FILE,
                "Chybí static/attribution.html",
            )
            return
        if path in {
            "/kauzalni-prenos",
            "/kauzalni-prenos/",
            "/activation-patching",
            "/activation-patching/",
            "/patching.html",
        }:
            self.send_html_file(
                ACTIVATION_PATCHING_FILE,
                "Chybí static/patching.html",
            )
            return
        if path in {
            "/kontrast",
            "/kontrast/",
            "/context-contrast",
            "/context-contrast/",
            "/contrast.html",
        }:
            self.send_html_file(
                CONTEXT_CONTRAST_FILE,
                "Chybí static/contrast.html",
            )
            return
        if path in {
            "/steering",
            "/steering/",
            "/steering.html",
        }:
            self.send_html_file(
                STEERING_FILE,
                "Chybí static/steering.html",
            )
            return
        if path in {
            "/prekvapeni",
            "/prekvapeni/",
            "/prekvapeni.html",
            "/surprise",
            "/surprise/",
            "/surprise.html",
        }:
            self.send_html_file(
                SURPRISE_FILE,
                "Chybí static/surprise.html",
            )
            return
        if path in {
            "/detektor",
            "/detektor/",
            "/ai-detektor",
            "/ai-detektor/",
            "/predictive-stylometry",
            "/predictive-stylometry/",
            "/detector.html",
        }:
            self.send_html_file(
                PREDICTIVE_DETECTOR_FILE,
                "Chybí static/detector.html",
            )
            return
        if path in {
            "/tokenizace",
            "/tokenizace/",
            "/tokenization",
            "/tokenization/",
            "/tokenization.html",
        }:
            self.send_html_file(
                TOKENIZATION_FILE,
                "Chybí static/tokenization.html",
            )
            return
        if path in {
            "/embeddingy",
            "/embeddingy/",
            "/embeddings",
            "/embeddings/",
            "/embedding.html",
        }:
            self.send_html_file(
                EMBEDDINGS_FILE,
                "Chybí static/embedding.html",
            )
            return
        if path in {"/archiv", "/archiv/", "/archive", "/archive.html"}:
            self.send_html_file(ARCHIVE_FILE, "Chybí static/archive.html")
            return
        if path == "/static/workbench-shell.css":
            self.send_static_file(
                WORKBENCH_SHELL_CSS_FILE,
                "text/css; charset=utf-8",
                "Chybí static/workbench-shell.css",
            )
            return
        if path == "/static/i18n-catalog.generated.js":
            self.send_static_file(
                I18N_CATALOG_JS_FILE,
                "text/javascript; charset=utf-8",
                "Chybí static/i18n-catalog.generated.js",
            )
            return
        if path == "/static/i18n.js":
            self.send_static_file(
                I18N_JS_FILE,
                "text/javascript; charset=utf-8",
                "Chybí static/i18n.js",
            )
            return
        if path == "/static/workbench-shell.js":
            self.send_static_file(
                WORKBENCH_SHELL_JS_FILE,
                "text/javascript; charset=utf-8",
                "Chybí static/workbench-shell.js",
            )
            return
        if path == "/static/api-client.js":
            self.send_static_file(
                API_CLIENT_JS_FILE,
                "text/javascript; charset=utf-8",
                "Chybí static/api-client.js",
            )
            return
        if path == "/api/health":
            health = self.state.health()
            try:
                health["storage"] = self.experiment_store.status()
            except Exception as exc:
                self.log_error("Experiment storage health failed: %s", exc)
                health["storage"] = {
                    "status": "error",
                    "error": "Lokální archiv není dostupný.",
                }
            self.send_json(health)
            return
        if path == "/api/tuned-lens/status":
            self.send_json(self.state.tuned_lens_status())
            return
        if path == "/api/jacobian-lens/status":
            query = parse_qs(parsed_url.query, keep_blank_values=True)
            artifact_values = query.get("artifact_id", [])
            if (
                len(artifact_values) > 1
                or (artifact_values and not artifact_values[0].strip())
            ):
                self.send_json(
                    {
                        "code": "invalid_request",
                        "error": (
                            "artifact_id smí být uveden jednou a nesmí "
                            "být prázdný."
                        ),
                    },
                    HTTPStatus.BAD_REQUEST,
                )
                return
            expected_artifact_id = (
                artifact_values[0] if artifact_values else None
            )
            self.send_json(
                self.state.jacobian_lens_status(expected_artifact_id)
            )
            return

        if path in {
            "/api/experiments",
            "/api/experiments/export.json",
            "/api/experiments/export.csv",
        } or path.startswith("/api/experiments/"):
            if not self.require_safe_archive_read():
                return
            try:
                kind, query = self._experiment_query(parsed_url)
                if path == "/api/experiments":
                    limit_text = self._single_query_value(query, "limit")
                    offset_text = self._single_query_value(query, "offset")
                    limit = 100 if limit_text is None else int(limit_text)
                    offset = 0 if offset_text is None else int(offset_text)
                    result = self.experiment_store.list_summaries(
                        kind=kind,
                        limit=limit,
                        offset=offset,
                    )
                    self.send_json(result)
                    return
                if path == "/api/experiments/export.json":
                    body = self.experiment_store.export_json(kind=kind)
                    self.send_download(
                        body,
                        content_type="application/json; charset=utf-8",
                        filename="apertus-experiments.json",
                    )
                    return
                if path == "/api/experiments/export.csv":
                    body = self.experiment_store.export_csv(kind=kind)
                    self.send_download(
                        body,
                        content_type="text/csv; charset=utf-8",
                        filename="apertus-experiments.csv",
                    )
                    return

                export_format = None
                export_suffix = None
                if path.endswith("/export.json"):
                    export_format = "json"
                    export_suffix = "/export.json"
                elif path.endswith("/export.csv"):
                    export_format = "csv"
                    export_suffix = "/export.csv"
                if export_format is not None and export_suffix is not None:
                    record_id = path.removeprefix(
                        "/api/experiments/"
                    ).removesuffix(export_suffix)
                    if not record_id or "/" in record_id:
                        raise ExperimentValidationError(
                            "Neplatné ID experimentu."
                        )
                    record = self.experiment_store.get(record_id)
                    if record is None:
                        self.send_json(
                            {
                                "code": "experiment_not_found",
                                "error": "Experiment nebyl nalezen.",
                            },
                            HTTPStatus.NOT_FOUND,
                        )
                        return
                    if export_format == "json":
                        body = self.experiment_store.export_json(
                            record_id=record_id
                        )
                        content_type = "application/json; charset=utf-8"
                    else:
                        body = self.experiment_store.export_csv(
                            record_id=record_id
                        )
                        content_type = "text/csv; charset=utf-8"
                    self.send_download(
                        body,
                        content_type=content_type,
                        filename=(
                            f"apertus-experiment-{record_id}.{export_format}"
                        ),
                    )
                    return

                record_id = path.removeprefix("/api/experiments/")
                if not record_id or "/" in record_id:
                    raise ExperimentValidationError(
                        "Neplatné ID experimentu."
                    )
                record = self.experiment_store.get(record_id)
                if record is None:
                    self.send_json(
                        {
                            "code": "experiment_not_found",
                            "error": "Experiment nebyl nalezen.",
                        },
                        HTTPStatus.NOT_FOUND,
                    )
                    return
                self.send_json({"record": record})
                return
            except ExperimentSchemaVersionError as exc:
                self.send_experiment_error(
                    exc,
                    HTTPStatus.CONFLICT,
                    code="schema_version_unsupported",
                )
                return
            except (ExperimentValidationError, ValueError) as exc:
                self.send_experiment_error(
                    exc,
                    HTTPStatus.BAD_REQUEST,
                    code="invalid_experiment_request",
                )
                return
            except ExperimentError as exc:
                self.log_error("Experiment storage request failed: %s", exc)
                self.send_experiment_error(
                    RuntimeError("Lokální archiv není dostupný."),
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    code="storage_unavailable",
                )
                return
            except Exception as exc:
                self.log_error("Experiment storage request failed: %s", exc)
                self.send_experiment_error(
                    RuntimeError("Lokální archiv není dostupný."),
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    code="storage_unavailable",
                )
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    @staticmethod
    def _teacher_forced_request_fields(
        request: dict[str, Any],
        max_layers: int = 32,
    ) -> tuple[
        list[dict[str, Any]],
        str,
        int,
        float,
        float,
        str,
        int,
        str,
    ]:
        contrast_pairs = request.get("contrast_pairs")
        test_text = request.get("test_text")
        layer = request.get("layer", 16)
        alpha_value = request.get("alpha_magnitude", 1.0)
        temperature_value = request.get("temperature", 1.0)
        expected_model_id = request.get("model_id")
        expected_generation = request.get("model_generation")
        expected_server_instance_id = request.get("server_instance_id")
        if (
            not isinstance(contrast_pairs, list)
            or not MIN_STEERING_PAIRS
            <= len(contrast_pairs)
            <= MAX_STEERING_PAIRS
        ):
            raise ValueError(
                f"contrast_pairs musí být seznam "
                f"{MIN_STEERING_PAIRS} až "
                f"{MAX_STEERING_PAIRS} párů."
            )
        if not isinstance(test_text, str) or not test_text.strip():
            raise ValueError(
                "Pole test_text musí být neprázdný řetězec."
            )
        if type(layer) is not int or not 1 <= layer <= max_layers:
            raise ValueError(f"layer musí být celé číslo mezi 1 a {max_layers}.")
        if (
            isinstance(alpha_value, bool)
            or not isinstance(alpha_value, (int, float))
        ):
            raise ValueError("alpha_magnitude musí být číslo.")
        alpha_magnitude = float(alpha_value)
        if (
            not math.isfinite(alpha_magnitude)
            or not 0.0 < alpha_magnitude <= 2.0
        ):
            raise ValueError(
                "alpha_magnitude musí být větší než 0 a nejvýše 2."
            )
        if (
            isinstance(temperature_value, bool)
            or not isinstance(temperature_value, (int, float))
        ):
            raise ValueError("temperature musí být číslo.")
        temperature = float(temperature_value)
        if (
            not math.isfinite(temperature)
            or not 0.2 <= temperature <= 2.0
        ):
            raise ValueError("Teplota musí být mezi 0,2 a 2,0.")
        if (
            not isinstance(expected_model_id, str)
            or not expected_model_id
        ):
            raise ValueError(
                "model_id je pro steering povinný řetězec."
            )
        if (
            not isinstance(expected_server_instance_id, str)
            or not expected_server_instance_id
        ):
            raise ValueError(
                "server_instance_id je pro steering povinný řetězec."
            )
        if type(expected_generation) is not int:
            raise ValueError(
                "model_generation je pro steering povinné celé číslo."
            )
        return (
            contrast_pairs,
            test_text,
            layer,
            alpha_magnitude,
            temperature,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )

    @staticmethod
    def _tokenization_request_fields(
        request: dict[str, Any],
    ) -> tuple[str, str | None]:
        return tokenization_request_fields(request)

    @staticmethod
    def _embeddings_request_fields(
        request: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int, str, int, str, str]:
        return embeddings_request_fields(request)

    @staticmethod
    def _embedding_neighborhood_request_fields(
        request: dict[str, Any],
    ) -> tuple[int, int, str, int, str]:
        return embedding_neighborhood_request_fields(request)

    @staticmethod
    def _text_surprisal_request_fields(
        request: dict[str, Any],
    ) -> tuple[str, int, str, int, str]:
        return text_surprisal_request_fields(request)

    @staticmethod
    def _predictive_stylometry_request_fields(
        request: dict[str, Any],
    ) -> tuple[str, str, int, str]:
        text = request.get("text")
        expected_model_id = request.get("model_id")
        expected_generation = request.get("model_generation")
        expected_server_instance_id = request.get("server_instance_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Pole text musí být neprázdný řetězec.")
        if len(text) > MAX_PREDICTIVE_INPUT_CHARACTERS:
            raise ValueError(
                f"Text může mít nejvýše "
                f"{MAX_PREDICTIVE_INPUT_CHARACTERS} znaků."
            )
        if len(text.encode("utf-8")) > MAX_PREDICTIVE_INPUT_UTF8_BYTES:
            raise ValueError(
                f"Text může mít nejvýše "
                f"{MAX_PREDICTIVE_INPUT_UTF8_BYTES} UTF-8 bajtů."
            )
        if (
            not isinstance(expected_model_id, str)
            or not expected_model_id
        ):
            raise ValueError(
                "model_id je pro prediktivní stylometrii povinný "
                "řetězec."
            )
        if type(expected_generation) is not int:
            raise ValueError(
                "model_generation je pro prediktivní stylometrii "
                "povinné celé číslo."
            )
        if (
            not isinstance(expected_server_instance_id, str)
            or not expected_server_instance_id
        ):
            raise ValueError(
                "server_instance_id je pro prediktivní stylometrii "
                "povinný řetězec."
            )
        return (
            text,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )

    @staticmethod
    def _continuation_surprisal_request_fields(
        request: dict[str, Any],
    ) -> tuple[str, str, int, str, int, str]:
        return continuation_surprisal_request_fields(request)

    @staticmethod
    def _text_surprisal_comparison_request_fields(
        request: dict[str, Any],
    ) -> tuple[str, str, str, str, int, str, int, str]:
        return text_surprisal_comparison_request_fields(request)

    @staticmethod
    def _context_contrast_request_fields(
        request: dict[str, Any],
    ) -> tuple[str, str, str, int, str, int, str]:
        return context_contrast_request_fields(request)

    def _post_experiments(self, _path: str, request: Any) -> None:
        if "record" not in request:
            raise ExperimentValidationError(
                "Tělo požadavku musí obsahovat objekt record."
            )
        record = request["record"]
        stored_record, created = self.experiment_store.save(record)
        self.send_json(
            {
                "record": stored_record,
                "created": created,
            },
            HTTPStatus.CREATED if created else HTTPStatus.OK,
        )
        return

    def _post_experiments_import(self, _path: str, request: Any) -> None:
        result = self.experiment_store.import_payload(request)
        self.send_json(result)
        return

    def _post_model(self, _path: str, request: Any) -> None:
        target_model_id = request.get("model_id")
        if not isinstance(target_model_id, str):
            raise ValueError("model_id musí být řetězec.")
        result = self.state.request_model_switch(target_model_id)
        self.send_json(
            result,
            HTTPStatus.ACCEPTED
            if result.get("accepted")
            else HTTPStatus.OK,
        )
        return

    def _post_model_comparison(self, _path: str, request: Any) -> None:
        self.send_json(self.state.start_model_comparison(request), HTTPStatus.ACCEPTED)

    def _post_model_comparison_cancel(self, _path: str, request: Any) -> None:
        job_id = request.get("job_id")
        if not isinstance(job_id, str):
            raise ValueError("job_id musí být řetězec.")
        self.send_json(self.state.model_comparison_status(job_id, cancel=True))

    def _post_detector_readers(self, _path: str, request: Any) -> None:
        self.send_json(self.state.start_detector_reader_comparison(
            *self._predictive_stylometry_request_fields(request)), HTTPStatus.ACCEPTED)

    def _post_detector_readers_cancel(self, _path: str, request: Any) -> None:
        job_id = request.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id musí být neprázdný řetězec.")
        self.send_json(self.state.detector_reader_status(job_id, cancel=True))

    def _post_tokenization(self, _path: str, request: Any) -> None:
        text_a, text_b = self._tokenization_request_fields(request)
        self.send_json(self.state.tokenization(text_a, text_b))
        return

    def _post_embeddings(self, _path: str, request: Any) -> None:
        (
            contexts,
            neighbor_count,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
            analysis_mode,
        ) = self._embeddings_request_fields(request)
        result = self.state.embeddings(
            contexts,
            neighbor_count,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
            analysis_mode,
        )
        self.send_json(result)
        return

    def _post_embedding_neighborhood(self, _path: str, request: Any) -> None:
        (
            token_id,
            neighbor_count,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        ) = self._embedding_neighborhood_request_fields(request)
        result = self.state.embedding_neighborhood(
            token_id,
            neighbor_count,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_text_surprisal(self, _path: str, request: Any) -> None:
        (
            text,
            top_k,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        ) = self._text_surprisal_request_fields(request)
        result = self.state.text_surprisal(
            text,
            top_k,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_predictive_stylometry(self, _path: str, request: Any) -> None:
        (
            text,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        ) = self._predictive_stylometry_request_fields(request)
        result = self.state.predictive_stylometry(
            text,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_continuation_surprisal(self, _path: str, request: Any) -> None:
        (
            prefix,
            continuation,
            top_k,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        ) = self._continuation_surprisal_request_fields(request)
        result = self.state.continuation_surprisal(
            prefix,
            continuation,
            top_k,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_text_surprisal_comparison(self, _path: str, request: Any) -> None:
        (
            text_a,
            text_b,
            label_a,
            label_b,
            top_k,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        ) = self._text_surprisal_comparison_request_fields(
            request
        )
        result = self.state.text_surprisal_comparison(
            text_a,
            text_b,
            label_a,
            label_b,
            top_k,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_context_contrast(self, _path: str, request: Any) -> None:
        (
            prefix_a,
            prefix_b,
            continuation,
            top_k,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        ) = self._context_contrast_request_fields(request)
        result = self.state.context_contrast(
            prefix_a,
            prefix_b,
            continuation,
            top_k,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_steering_aligned_path(self, _path: str, request: Any) -> None:
        (
            contrast_pairs,
            test_text,
            layer,
            alpha_magnitude,
            temperature,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        ) = self._teacher_forced_request_fields(request, self.state.steering_layer_count())
        continuation_text = request.get("continuation_text")
        if (
            not isinstance(continuation_text, str)
            or not continuation_text.strip()
        ):
            raise ValueError(
                "continuation_text musí být neprázdný řetězec."
            )
        result = self.state.steering_aligned_path(
            contrast_pairs,
            test_text,
            continuation_text,
            layer,
            alpha_magnitude,
            temperature,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_steering_phrase_score(self, _path: str, request: Any) -> None:
        (
            contrast_pairs,
            test_text,
            layer,
            alpha_magnitude,
            temperature,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        ) = self._teacher_forced_request_fields(request, self.state.steering_layer_count())
        continuation_pairs = request.get("continuation_pairs")
        if (
            not isinstance(continuation_pairs, list)
            or not 1
            <= len(continuation_pairs)
            <= MAX_STEERING_PHRASE_PAIRS
        ):
            raise ValueError(
                "continuation_pairs musí obsahovat 1 až "
                f"{MAX_STEERING_PHRASE_PAIRS} párů."
            )
        result = self.state.steering_phrase_score(
            contrast_pairs,
            test_text,
            continuation_pairs,
            layer,
            alpha_magnitude,
            temperature,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_steering_sham_control(self, _path: str, request: Any) -> None:
        (
            contrast_pairs,
            test_text,
            layer,
            alpha_magnitude,
            temperature,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        ) = self._teacher_forced_request_fields(request, self.state.steering_layer_count())
        continuation_pairs = request.get("continuation_pairs")
        if (
            not isinstance(continuation_pairs, list)
            or not 1
            <= len(continuation_pairs)
            <= MAX_STEERING_PHRASE_PAIRS
        ):
            raise ValueError(
                "continuation_pairs musí obsahovat 1 až "
                f"{MAX_STEERING_PHRASE_PAIRS} párů."
            )
        sample_count = request.get("sample_count", 20)
        seed = request.get("seed", 0)
        if type(sample_count) is not int or not (
            MIN_STEERING_SHAM_SAMPLES
            <= sample_count
            <= MAX_STEERING_SHAM_SAMPLES
        ):
            raise ValueError(
                "sample_count musí být celé číslo mezi "
                f"{MIN_STEERING_SHAM_SAMPLES} a "
                f"{MAX_STEERING_SHAM_SAMPLES}."
            )
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("seed musí být uint32.")
        result = self.state.steering_sham_control(
            contrast_pairs,
            test_text,
            continuation_pairs,
            layer,
            alpha_magnitude,
            temperature,
            sample_count,
            seed,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_steering_layer_sweep(self, _path: str, request: Any) -> None:
        (
            contrast_pairs,
            test_text,
            default_reference_layer,
            alpha_magnitude,
            temperature,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        ) = self._teacher_forced_request_fields(request, self.state.steering_layer_count())
        continuation_pairs = request.get("continuation_pairs")
        if (
            not isinstance(continuation_pairs, list)
            or not 1
            <= len(continuation_pairs)
            <= MAX_STEERING_PHRASE_PAIRS
        ):
            raise ValueError(
                "continuation_pairs musí obsahovat 1 až "
                f"{MAX_STEERING_PHRASE_PAIRS} párů."
            )
        layers = request.get("layers")
        if not isinstance(layers, list) or not (
            MIN_STEERING_SWEEP_LAYERS
            <= len(layers)
            <= MAX_STEERING_SWEEP_LAYERS
        ):
            raise ValueError(
                "layers musí obsahovat "
                f"{MIN_STEERING_SWEEP_LAYERS} až "
                f"{MAX_STEERING_SWEEP_LAYERS} vrstev."
            )
        if any(
            type(candidate) is not int or not 1 <= candidate <= self.state.steering_layer_count()
            for candidate in layers
        ):
            raise ValueError(
                "Každá položka layers musí být celé číslo "
                f"mezi 1 a {self.state.steering_layer_count()}."
            )
        if len(set(layers)) != len(layers):
            raise ValueError("layers nesmí obsahovat duplicity.")
        reference_layer = request.get(
            "reference_layer",
            default_reference_layer,
        )
        if (
            type(reference_layer) is not int
            or reference_layer not in layers
        ):
            raise ValueError(
                "reference_layer musí být jednou z layers."
            )
        result = self.state.steering_layer_sweep(
            contrast_pairs,
            test_text,
            continuation_pairs,
            layers,
            reference_layer,
            alpha_magnitude,
            temperature,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_steering_trajectory(self, _path: str, request: Any) -> None:
        contrast_pairs = request.get("contrast_pairs")
        test_text = request.get("test_text")
        layer = request.get("layer", 16)
        alpha_value = request.get("alpha_magnitude", 1.0)
        temperature_value = request.get("temperature", 1.0)
        max_new_tokens = request.get("max_new_tokens", 32)
        generation_mode = request.get(
            "generation_mode",
            "greedy",
        )
        seed = request.get("seed", 0)
        expected_model_id = request.get("model_id")
        expected_generation = request.get("model_generation")
        expected_server_instance_id = request.get(
            "server_instance_id"
        )
        if (
            not isinstance(contrast_pairs, list)
            or not MIN_STEERING_PAIRS
            <= len(contrast_pairs)
            <= MAX_STEERING_PAIRS
        ):
            raise ValueError(
                f"contrast_pairs musí být seznam "
                f"{MIN_STEERING_PAIRS} až "
                f"{MAX_STEERING_PAIRS} párů."
            )
        if not isinstance(test_text, str) or not test_text.strip():
            raise ValueError(
                "Pole test_text musí být neprázdný řetězec."
            )
        if type(layer) is not int or not 1 <= layer <= self.state.steering_layer_count():
            raise ValueError(
                f"layer musí být celé číslo mezi 1 a {self.state.steering_layer_count()}."
            )
        if (
            isinstance(alpha_value, bool)
            or not isinstance(alpha_value, (int, float))
        ):
            raise ValueError("alpha_magnitude musí být číslo.")
        alpha_magnitude = float(alpha_value)
        if (
            not math.isfinite(alpha_magnitude)
            or not 0.0 < alpha_magnitude <= 2.0
        ):
            raise ValueError(
                "alpha_magnitude musí být větší než 0 a nejvýše 2."
            )
        if (
            isinstance(temperature_value, bool)
            or not isinstance(temperature_value, (int, float))
        ):
            raise ValueError("temperature musí být číslo.")
        temperature = float(temperature_value)
        if (
            not math.isfinite(temperature)
            or not 0.2 <= temperature <= 2.0
        ):
            raise ValueError(
                "Teplota musí být mezi 0,2 a 2,0."
            )
        if (
            type(max_new_tokens) is not int
            or not 1
            <= max_new_tokens
            <= MAX_STEERING_TRAJECTORY_TOKENS
        ):
            raise ValueError(
                "max_new_tokens musí být celé číslo mezi 1 a "
                f"{MAX_STEERING_TRAJECTORY_TOKENS}."
            )
        if generation_mode not in {"greedy", "sample"}:
            raise ValueError(
                "generation_mode musí být greedy nebo sample."
            )
        if (
            type(seed) is not int
            or not 0 <= seed <= 2**32 - 1
        ):
            raise ValueError(
                "seed musí být celé číslo mezi 0 a 2^32−1."
            )
        if (
            not isinstance(expected_model_id, str)
            or not expected_model_id
        ):
            raise ValueError(
                "model_id je pro steering povinný řetězec."
            )
        if (
            not isinstance(expected_server_instance_id, str)
            or not expected_server_instance_id
        ):
            raise ValueError(
                "server_instance_id je pro steering povinný "
                "řetězec."
            )
        if type(expected_generation) is not int:
            raise ValueError(
                "model_generation je pro steering povinné celé "
                "číslo."
            )
        result = self.state.steering_trajectory(
            contrast_pairs,
            test_text,
            layer,
            alpha_magnitude,
            temperature,
            max_new_tokens,
            generation_mode,
            seed,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_steering(self, _path: str, request: Any) -> None:
        contrast_pairs = request.get("contrast_pairs")
        test_text = request.get("test_text")
        layer = request.get("layer", 16)
        alpha_value = request.get("alpha", 1.0)
        temperature_value = request.get("temperature", 1.0)
        top_k = request.get("top_k", 8)
        mover_count = request.get("mover_count", 8)
        overlap_k = request.get("overlap_k", 10)
        expected_model_id = request.get("model_id")
        expected_generation = request.get("model_generation")
        expected_server_instance_id = request.get(
            "server_instance_id"
        )
        if (
            not isinstance(contrast_pairs, list)
            or not MIN_STEERING_PAIRS
            <= len(contrast_pairs)
            <= MAX_STEERING_PAIRS
        ):
            raise ValueError(
                f"contrast_pairs musí být seznam "
                f"{MIN_STEERING_PAIRS} až "
                f"{MAX_STEERING_PAIRS} párů."
            )
        if not isinstance(test_text, str) or not test_text.strip():
            raise ValueError(
                "Pole test_text musí být neprázdný řetězec."
            )
        if type(layer) is not int or not 1 <= layer <= self.state.steering_layer_count():
            raise ValueError(
                f"layer musí být celé číslo mezi 1 a {self.state.steering_layer_count()}."
            )
        if (
            isinstance(alpha_value, bool)
            or not isinstance(alpha_value, (int, float))
        ):
            raise ValueError("alpha musí být číslo.")
        alpha = float(alpha_value)
        if not math.isfinite(alpha) or not -2.0 <= alpha <= 2.0:
            raise ValueError("alpha musí být mezi -2 a 2.")
        if (
            isinstance(temperature_value, bool)
            or not isinstance(temperature_value, (int, float))
        ):
            raise ValueError("temperature musí být číslo.")
        temperature = float(temperature_value)
        if (
            not math.isfinite(temperature)
            or not 0.2 <= temperature <= 2.0
        ):
            raise ValueError(
                "Teplota musí být mezi 0,2 a 2,0."
            )
        if type(top_k) is not int or not 3 <= top_k <= 12:
            raise ValueError(
                "top_k musí být celé číslo mezi 3 a 12."
            )
        if (
            type(mover_count) is not int
            or not 1 <= mover_count <= COMPARE_MOVER_LIMIT
        ):
            raise ValueError(
                "mover_count musí být celé číslo mezi 1 a "
                f"{COMPARE_MOVER_LIMIT}."
            )
        if (
            type(overlap_k) is not int
            or not 1 <= overlap_k <= COMPARE_OVERLAP_LIMIT
        ):
            raise ValueError(
                "overlap_k musí být celé číslo mezi 1 a "
                f"{COMPARE_OVERLAP_LIMIT}."
            )
        if (
            not isinstance(expected_model_id, str)
            or not expected_model_id
        ):
            raise ValueError(
                "model_id je pro steering povinný řetězec."
            )
        if (
            not isinstance(expected_server_instance_id, str)
            or not expected_server_instance_id
        ):
            raise ValueError(
                "server_instance_id je pro steering povinný "
                "řetězec."
            )
        if type(expected_generation) is not int:
            raise ValueError(
                "model_generation je pro steering povinné celé "
                "číslo."
            )
        result = self.state.steering(
            contrast_pairs,
            test_text,
            layer,
            alpha,
            temperature,
            top_k,
            mover_count,
            overlap_k,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_lens(self, _path: str, request: Any) -> None:
        text = request.get("text")
        top_k = request.get("top_k", 5)
        expected_model_id = request.get("model_id")
        expected_generation = request.get("model_generation")
        expected_server_instance_id = request.get(
            "server_instance_id"
        )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Pole text musí být neprázdný řetězec."
            )
        if type(top_k) is not int or not 3 <= top_k <= 8:
            raise ValueError(
                "top_k pro RAW logit lens musí být celé číslo "
                "mezi 3 a 8."
            )
        if (
            expected_model_id is not None
            and not isinstance(expected_model_id, str)
        ):
            raise ValueError("model_id musí být řetězec.")
        if (
            expected_generation is not None
            and type(expected_generation) is not int
        ):
            raise ValueError(
                "model_generation musí být celé číslo."
            )
        if (
            expected_server_instance_id is not None
            and not isinstance(expected_server_instance_id, str)
        ):
            raise ValueError(
                "server_instance_id musí být řetězec."
            )
        result = self.state.lens(
            text,
            top_k,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_tuned_lens(self, _path: str, request: Any) -> None:
        text = request.get("text")
        top_k = request.get("top_k", 5)
        expected_artifact_id = request.get("calibration_id")
        expected_model_id = request.get("model_id")
        expected_generation = request.get("model_generation")
        expected_server_instance_id = request.get(
            "server_instance_id"
        )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Pole text musí být neprázdný řetězec."
            )
        if type(top_k) is not int or not 3 <= top_k <= 8:
            raise ValueError(
                "top_k pro RAW × TUNED musí být celé číslo "
                "mezi 3 a 8."
            )
        for name, value in (
            ("calibration_id", expected_artifact_id),
            ("model_id", expected_model_id),
            ("server_instance_id", expected_server_instance_id),
        ):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} musí být řetězec.")
        if (
            expected_generation is not None
            and type(expected_generation) is not int
        ):
            raise ValueError(
                "model_generation musí být celé číslo."
            )
        result = self.state.tuned_lens(
            text,
            top_k,
            expected_artifact_id,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_jacobian_lens(self, _path: str, request: Any) -> None:
        text = request.get("text")
        top_k = request.get("top_k", 5)
        source_layers = request.get("source_layers")
        expected_artifact_id = request.get("artifact_id")
        expected_model_id = request.get("model_id")
        expected_generation = request.get("model_generation")
        expected_server_instance_id = request.get(
            "server_instance_id"
        )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Pole text musí být neprázdný řetězec."
            )
        if type(top_k) is not int or not 3 <= top_k <= 20:
            raise ValueError(
                "top_k pro RAW × J-LENS musí být celé číslo "
                "mezi 3 a 20."
            )
        if (
            not isinstance(source_layers, list)
            or not source_layers
            or len(source_layers) > 31
            or any(
                type(layer) is not int or not 1 <= layer <= 31
                for layer in source_layers
            )
            or len(source_layers) != len(set(source_layers))
        ):
            raise ValueError(
                "source_layers musí být neprázdný seznam "
                "unikátních vrstev L1–L31."
            )
        for name, value in (
            ("artifact_id", expected_artifact_id),
            ("model_id", expected_model_id),
            ("server_instance_id", expected_server_instance_id),
        ):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} musí být řetězec.")
        if (
            expected_generation is not None
            and type(expected_generation) is not int
        ):
            raise ValueError(
                "model_generation musí být celé číslo."
            )
        result = self.state.jacobian_lens(
            text,
            top_k,
            source_layers,
            expected_artifact_id,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_component_attribution(self, _path: str, request: Any) -> None:
        text = request.get("text")
        token_a = request.get("token_a")
        token_b = request.get("token_b")
        expected_model_id = request.get("model_id")
        expected_generation = request.get("model_generation")
        expected_server_instance_id = request.get(
            "server_instance_id"
        )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Pole text musí být neprázdný řetězec."
            )
        if len(text) > 4_000:
            raise ValueError(
                "Prefix může mít nejvýše 4000 znaků."
            )
        for name, value in (
            ("token_a", token_a),
            ("token_b", token_b),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{name} musí být neprázdný token nebo #ID."
                )
            if len(value) > 160:
                raise ValueError(
                    f"{name} může mít nejvýše 160 znaků."
                )
        if (
            not isinstance(expected_model_id, str)
            or not expected_model_id
        ):
            raise ValueError(
                "model_id je pro komponentový rozklad povinný "
                "řetězec."
            )
        if type(expected_generation) is not int:
            raise ValueError(
                "model_generation je pro komponentový rozklad "
                "povinné celé číslo."
            )
        if (
            not isinstance(expected_server_instance_id, str)
            or not expected_server_instance_id
        ):
            raise ValueError(
                "server_instance_id je pro komponentový rozklad "
                "povinný řetězec."
            )
        result = self.state.component_attribution(
            text,
            token_a,
            token_b,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_activation_patching(self, _path: str, request: Any) -> None:
        source_text = request.get("source_text")
        target_text = request.get("target_text")
        token_a = request.get("token_a")
        token_b = request.get("token_b")
        layers = request.get("layers")
        positions = request.get("positions")
        top_k = request.get("top_k", 6)
        expected_model_id = request.get("model_id")
        expected_generation = request.get("model_generation")
        expected_server_instance_id = request.get(
            "server_instance_id"
        )
        for name, value in (
            ("source_text", source_text),
            ("target_text", target_text),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{name} musí být neprázdný řetězec."
                )
            if len(value) > 4_000:
                raise ValueError(
                    f"{name} může mít nejvýše 4000 znaků."
                )
        for name, value in (
            ("token_a", token_a),
            ("token_b", token_b),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{name} musí být neprázdný token nebo #ID."
                )
            if len(value) > 160:
                raise ValueError(
                    f"{name} může mít nejvýše 160 znaků."
                )
        if (
            not isinstance(layers, list)
            or not layers
            or len(layers) > 32
            or len(layers) != len(set(layers))
            or any(
                type(layer) is not int or not 1 <= layer <= 32
                for layer in layers
            )
        ):
            raise ValueError(
                "layers musí být neprázdný seznam unikátních "
                "vrstev L1–L32."
            )
        if (
            not isinstance(positions, list)
            or not positions
            or len(positions) > MAX_ACTIVATION_PATCHING_POSITIONS
            or len(positions) != len(set(positions))
            or any(
                type(position) is not int or position < 0
                for position in positions
            )
        ):
            raise ValueError(
                "positions musí být neprázdný seznam nejvýše "
                f"{MAX_ACTIVATION_PATCHING_POSITIONS} unikátních "
                "nezáporných modelových pozic."
            )
        if len(layers) * len(positions) > MAX_ACTIVATION_PATCHING_CELLS:
            raise ValueError(
                "Požadovaná heatmapa překračuje limit "
                f"{MAX_ACTIVATION_PATCHING_CELLS} buněk."
            )
        if type(top_k) is not int or not 3 <= top_k <= 12:
            raise ValueError(
                "top_k musí být celé číslo mezi 3 a 12."
            )
        if (
            not isinstance(expected_model_id, str)
            or not expected_model_id
        ):
            raise ValueError(
                "model_id je pro kauzální přenos povinný řetězec."
            )
        if type(expected_generation) is not int:
            raise ValueError(
                "model_generation je pro kauzální přenos povinné "
                "celé číslo."
            )
        if (
            not isinstance(expected_server_instance_id, str)
            or not expected_server_instance_id
        ):
            raise ValueError(
                "server_instance_id je pro kauzální přenos "
                "povinný řetězec."
            )
        result = self.state.activation_patching(
            source_text,
            target_text,
            token_a,
            token_b,
            layers,
            positions,
            top_k,
            expected_model_id,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_compare(self, _path: str, request: Any) -> None:
        left = request.get("left")
        right = request.get("right")
        temperature = float(request.get("temperature", 1.0))
        mover_count = request.get("mover_count", 8)
        overlap_k = request.get("overlap_k", 10)
        expected_generation = request.get("model_generation")
        expected_server_instance_id = request.get(
            "server_instance_id"
        )

        def validate_node(
            node: Any,
            name: str,
        ) -> tuple[str, list[int]]:
            if not isinstance(node, dict):
                raise ValueError(
                    f"Pole {name} musí být JSON objekt."
                )
            node_text = node.get("text")
            node_token_ids = node.get("token_ids")
            if (
                not isinstance(node_text, str)
                or not node_text.strip()
            ):
                raise ValueError(
                    f"Pole {name}.text musí být neprázdný řetězec."
                )
            if (
                not isinstance(node_token_ids, list)
                or not node_token_ids
                or len(node_token_ids) > MAX_INPUT_TOKENS
                or any(
                    type(token_id) is not int or token_id < 0
                    for token_id in node_token_ids
                )
            ):
                raise ValueError(
                    f"Pole {name}.token_ids musí být neprázdný "
                    "seznam celých čísel."
                )
            return node_text, list(node_token_ids)

        left_text, left_token_ids = validate_node(left, "left")
        right_text, right_token_ids = validate_node(right, "right")
        if not 0.2 <= temperature <= 2.0:
            raise ValueError("Teplota musí být mezi 0,2 a 2,0.")
        if (
            type(mover_count) is not int
            or not 1 <= mover_count <= COMPARE_MOVER_LIMIT
        ):
            raise ValueError(
                "mover_count musí být celé číslo mezi 1 a "
                f"{COMPARE_MOVER_LIMIT}."
            )
        if (
            type(overlap_k) is not int
            or not 1 <= overlap_k <= COMPARE_OVERLAP_LIMIT
        ):
            raise ValueError(
                "overlap_k musí být celé číslo mezi 1 a "
                f"{COMPARE_OVERLAP_LIMIT}."
            )
        if (
            expected_generation is not None
            and type(expected_generation) is not int
        ):
            raise ValueError("model_generation musí být celé číslo.")
        if (
            expected_server_instance_id is not None
            and not isinstance(expected_server_instance_id, str)
        ):
            raise ValueError(
                "server_instance_id musí být řetězec."
            )
        result = self.state.compare_distributions(
            left_text,
            left_token_ids,
            right_text,
            right_token_ids,
            temperature,
            mover_count,
            overlap_k,
            expected_generation,
            expected_server_instance_id,
        )
        self.send_json(result)
        return

    def _post_garden_query(
        self,
        path: str,
        request: dict[str, Any],
    ) -> None:
        text = request.get("text")
        temperature = float(request.get("temperature", 1.0))
        provided_token_ids = request.get("token_ids")
        pinned_token_ids = request.get("pinned_token_ids")
        cache_only = request.get("cache_only", False)
        expected_generation = request.get("model_generation")
        expected_server_instance_id = request.get("server_instance_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Pole text musí být neprázdný řetězec.")
        if not 0.2 <= temperature <= 2.0:
            raise ValueError("Teplota musí být mezi 0,2 a 2,0.")
        if provided_token_ids is not None:
            if (
                not isinstance(provided_token_ids, list)
                or not provided_token_ids
                or len(provided_token_ids) > MAX_INPUT_TOKENS
                or any(
                    type(token_id) is not int or token_id < 0
                    for token_id in provided_token_ids
                )
            ):
                raise ValueError("token_ids musí být neprázdný seznam celých čísel.")
        if pinned_token_ids is not None:
            if (
                not isinstance(pinned_token_ids, list)
                or len(pinned_token_ids) > MAX_PINNED_TOKENS
                or any(
                    type(token_id) is not int or token_id < 0
                    for token_id in pinned_token_ids
                )
            ):
                raise ValueError(
                    "pinned_token_ids musí být seznam nejvýše "
                    f"{MAX_PINNED_TOKENS} celých čísel."
                )
        if not isinstance(cache_only, bool):
            raise ValueError("cache_only musí být boolean.")
        if (
            expected_generation is not None
            and type(expected_generation) is not int
        ):
            raise ValueError("model_generation musí být celé číslo.")
        if (
            expected_server_instance_id is not None
            and not isinstance(expected_server_instance_id, str)
        ):
            raise ValueError("server_instance_id musí být řetězec.")
        if path == "/api/next":
            top_k = int(request.get("top_k", 6))
            if not 3 <= top_k <= 12:
                raise ValueError("top_k musí být mezi 3 a 12.")
            result = self.state.next_tokens(
                text,
                temperature,
                top_k,
                provided_token_ids,
                pinned_token_ids,
                cache_only,
                expected_generation,
                expected_server_instance_id,
            )
        elif path == "/api/search":
            query = request.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Pole query musí být neprázdný řetězec.")
            if len(query) > 64:
                raise ValueError("Hledaný výraz může mít nejvýše 64 znaků.")
            if provided_token_ids is None:
                raise ValueError("Pro hledání je nutné poslat token_ids uzlu.")
            result = self.state.search_tokens(
                text,
                query,
                temperature,
                provided_token_ids,
                SEARCH_RESULT_LIMIT,
                expected_generation,
                expected_server_instance_id,
            )
        else:
            requested_token_ids = request.get("candidate_token_ids")
            if (
                not isinstance(requested_token_ids, list)
                or not requested_token_ids
                or len(requested_token_ids) > LOOKUP_TOKEN_LIMIT
                or any(
                    type(token_id) is not int or token_id < 0
                    for token_id in requested_token_ids
                )
            ):
                raise ValueError(
                    "candidate_token_ids musí být neprázdný seznam nejvýše "
                    f"{LOOKUP_TOKEN_LIMIT} celých čísel."
                )
            if provided_token_ids is None:
                raise ValueError("Pro hledání je nutné poslat token_ids uzlu.")
            result = self.state.lookup_tokens(
                text,
                requested_token_ids,
                temperature,
                provided_token_ids,
                expected_generation,
                expected_server_instance_id,
            )
        self.send_json(result)

    def _post_model_admin(self, path: str, request: dict) -> None:
        admin = self.server.model_admin
        if path == "/api/model-admin/addons":
            result = admin.addon_action(request)
        elif path.endswith("/inspect"):
            result = admin.inspect(request)
        elif path.endswith("/edit"):
            result = admin.edit(request)
        elif path.endswith("/plan"):
            result = admin.plan_install(request)
        elif path.endswith("/jobs"):
            result = admin.start_job(request)
        elif path.endswith("/cancel"):
            result = admin.job_status(request.get("job_id"), cancel=True)
        elif path.endswith("/reload"):
            result = admin.reload(request)
        else:
            raise ValueError("Unknown model administration route.")
        self.send_json(result)

    def _dispatch_post_request(
        self,
        path: str,
        request: Any,
    ) -> None:
        route = POST_ROUTES[path]
        capability = {
            "_post_embeddings": "embeddings", "_post_embedding_neighborhood": "embeddings",
            "_post_predictive_stylometry": "detector", "_post_lens": "raw",
            "_post_tuned_lens": "tuned", "_post_jacobian_lens": "jlens",
            "_post_component_attribution": "attribution", "_post_activation_patching": "patching",
        }.get(route.controller)
        if route.controller.startswith("_post_steering"):
            capability = "steering"
        if capability:
            self.state.require_capability(capability)
        controller = getattr(self, route.controller)
        controller(path, request)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        route = POST_ROUTES.get(path)
        if route is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(
                self.headers.get("Content-Length", "0")
            )
            request_limit = {
                "default": MAX_REQUEST_BYTES,
                "experiment": MAX_EXPERIMENT_REQUEST_BYTES,
                "import": MAX_IMPORT_REQUEST_BYTES,
            }[route.request_limit]
            if content_length > request_limit:
                self.send_json(
                    {
                        "code": "request_too_large",
                        "error": (
                            f"Požadavek překračuje limit "
                            f"{request_limit} bajtů."
                        ),
                    },
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
                return
            if content_length <= 0:
                raise ValueError("Neplatná velikost požadavku.")
            content_type = self.headers.get(
                "Content-Type", ""
            )
            media_type = (
                content_type.split(";", 1)[0].strip().casefold()
            )
            if media_type != "application/json":
                self.send_json(
                    {
                        "code": "unsupported_media_type",
                        "error": (
                            "Lokální API přijímá POST pouze jako "
                            "application/json."
                        ),
                    },
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                )
                return
            if not self.require_safe_mutation_origin():
                return
            body = self.rfile.read(content_length)
            request = json.loads(body)
            if route.body_kind == "object" and not isinstance(
                request, dict
            ):
                raise ValueError(
                    "Tělo požadavku musí být JSON objekt."
                )
            self._dispatch_post_request(path, request)
        except (AdminError, AddonError) as exc:
            self.send_json({"code":exc.code,"error":str(exc),"details":exc.details},
                           HTTPStatus.CONFLICT if exc.code in {"registry_conflict","reload_required","model_busy","active_model","default_model","model_dependency","target_exists","profile_conflict"} else HTTPStatus.BAD_REQUEST)
        except ExperimentConflictError as exc:
            self.send_json(
                {
                    "code": "id_conflict",
                    "error": str(exc),
                    "id": getattr(exc, "record_id", None),
                },
                HTTPStatus.CONFLICT,
            )
        except ExperimentSchemaVersionError as exc:
            self.send_experiment_error(
                exc,
                HTTPStatus.CONFLICT,
                code="schema_version_unsupported",
            )
        except ExperimentValidationError as exc:
            self.send_experiment_error(
                exc,
                HTTPStatus.BAD_REQUEST,
                code="invalid_experiment_record",
            )
        except ExperimentError as exc:
            self.log_error("Experiment storage mutation failed: %s", exc)
            self.send_experiment_error(
                RuntimeError("Lokální archiv není dostupný."),
                HTTPStatus.SERVICE_UNAVAILABLE,
                code="storage_unavailable",
            )
        except ModelChangedError as exc:
            self.send_json(
                {
                    "code": "model_changed",
                    "error": str(exc),
                    "health": self.state.health(),
                },
                HTTPStatus.CONFLICT,
            )
        except (
            TunedLensUnavailableError,
            CalibrationIncompatibleError,
            CalibrationChangedError,
        ) as exc:
            self.send_json(
                {
                    "code": exc.code,
                    "error": str(exc),
                    "tuned_lens": self.state.tuned_lens_status(),
                },
                (
                    HTTPStatus.CONFLICT
                    if isinstance(
                        exc,
                        (
                            CalibrationIncompatibleError,
                            CalibrationChangedError,
                        ),
                    )
                    else HTTPStatus.SERVICE_UNAVAILABLE
                ),
            )
        except TunedLensError as exc:
            self.send_json(
                {
                    "code": exc.code,
                    "error": str(exc),
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        except (
            JacobianLensUnavailableError,
            JacobianLensIncompatibleError,
            JacobianLensChangedError,
            TransferProbeError,
        ) as exc:
            self.send_json(
                {
                    "code": exc.code,
                    "error": str(exc),
                    "jacobian_lens": self.state.jacobian_lens_status(),
                },
                (
                    HTTPStatus.CONFLICT
                    if isinstance(
                        exc,
                        (
                            JacobianLensIncompatibleError,
                            JacobianLensChangedError,
                            TransferProbeError,
                        ),
                    )
                    else HTTPStatus.SERVICE_UNAVAILABLE
                ),
            )
        except JacobianLensError as exc:
            self.send_json(
                {
                    "code": exc.code,
                    "error": str(exc),
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        except LogitsNotCachedError as exc:
            self.send_json(
                {
                    "code": "logits_not_cached",
                    "error": str(exc),
                    "side": exc.side,
                },
                HTTPStatus.CONFLICT,
            )
        except RuntimeError as exc:
            self.send_json(
                {"error": str(exc), "health": self.state.health()},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - defensive API boundary
            if path.startswith("/api/experiments"):
                self.log_error(
                    "Experiment storage mutation failed: %s",
                    exc,
                )
                self.send_experiment_error(
                    RuntimeError("Lokální archiv není dostupný."),
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    code="storage_unavailable",
                )
            else:
                self.send_json(
                    {"error": f"{type(exc).__name__}: {exc}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        prefix = "/api/experiments/"
        if not path.startswith(prefix):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        record_id = path.removeprefix(prefix)
        if not record_id or "/" in record_id:
            self.send_json(
                {
                    "code": "invalid_experiment_request",
                    "error": "Neplatné ID experimentu.",
                },
                HTTPStatus.BAD_REQUEST,
            )
            return
        if not self.require_safe_mutation_origin():
            return
        try:
            deleted = self.experiment_store.delete(record_id)
            if not deleted:
                self.send_json(
                    {
                        "code": "experiment_not_found",
                        "error": "Experiment nebyl nalezen.",
                    },
                    HTTPStatus.NOT_FOUND,
                )
                return
            self.send_json(
                {
                    "id": record_id,
                    "deleted": True,
                }
            )
        except ExperimentValidationError as exc:
            self.send_experiment_error(
                exc,
                HTTPStatus.BAD_REQUEST,
                code="invalid_experiment_request",
            )
        except ExperimentError as exc:
            self.log_error("Experiment storage deletion failed: %s", exc)
            self.send_experiment_error(
                RuntimeError("Lokální archiv není dostupný."),
                HTTPStatus.SERVICE_UNAVAILABLE,
                code="storage_unavailable",
            )
        except Exception as exc:
            self.log_error("Experiment storage deletion failed: %s", exc)
            self.send_experiment_error(
                RuntimeError("Lokální archiv není dostupný."),
                HTTPStatus.SERVICE_UNAVAILABLE,
                code="storage_unavailable",
            )


class GardenServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        state: ModelState,
        experiment_store: ExperimentStore,
        model_admin_data_dir: Path | None = None,
    ):
        super().__init__(address, GardenHandler)
        self.state = state
        self.experiment_store = experiment_store
        state.comparison_store = experiment_store
        self.model_admin = ModelAdmin(state, model_admin_data_dir)

    def server_close(self) -> None:
        if hasattr(self, "model_admin"):
            self.model_admin.close()
        super().server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--model-eurollm9", type=Path, default=None, help="Adresář EuroLLM 9B; MLX 6bit nebo původní HF váhy podle backendu.")
    parser.add_argument("--backend", choices=["mlx", "torch"], default=None)
    parser.add_argument("--registry", type=Path, default=None, help="TOML model registry (models.toml or built-in presets).")
    parser.add_argument("--data-dir", type=Path, default=None, help="User-owned archive and artifact directory.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda nebo cuda:0.")
    parser.add_argument(
        "--model-4bit",
        type=Path,
        default=None,
        help="Cesta k rychlé 4bit variantě Apertusu.",
    )
    parser.add_argument(
        "--model-8bit",
        type=Path,
        default=None,
        help="Cesta k 8bit variantě Apertusu.",
    )
    parser.add_argument(
        "--initial-model",
        default=None,
        help="Registered model ID loaded on startup (registry default_model).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--setup-only", action="store_true", help="Open model management without loading weights.")
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help=(
            "Cesta k lokální SQLite databázi experimentů "
            "(default: user data directory)."
        ),
    )
    parser.add_argument(
        "--tuned-lens-root",
        type=Path,
        default=None,
        help=(
            "Adresář verzovaných tuned-lens artefaktů "
            "(default: user data directory / tuned_lens)."
        ),
    )
    parser.add_argument(
        "--jacobian-lens-root",
        type=Path,
        default=None,
        help=(
            "Adresář verzovaných J-Lens artefaktů "
            "(default: user data directory / jacobian_lens)."
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Spustí rychlá demonstrační data bez načtení Apertusu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("This release is local-only: use --host 127.0.0.1 (no public authentication or multi-user isolation).")
    run_data = (args.data_dir or data_directory()).expanduser().resolve()
    database_path = (args.database or run_data / "experiments.sqlite3").expanduser().resolve()
    legacy_paths = any((args.model, args.model_8bit, args.model_4bit, args.model_eurollm9))
    registry = None
    if not args.mock and (args.registry is not None or not legacy_paths):
        try:
            registry = load_registry(args.registry or registry_path(data_root=run_data))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Cannot load model registry: {exc}") from exc
    if registry and args.backend and args.backend != registry.backend:
        raise SystemExit("--backend conflicts with the registry. Set backend in models.toml; one server uses one numerical backend.")
    backend_name = args.backend or (registry.backend if registry else "mlx")
    initial_model = args.initial_model or (registry.default_model if registry else HYBRID_MODEL_ID)
    try:
        experiment_store = ExperimentStore(database_path)
    except Exception as exc:
        raise SystemExit(
            f"Lokální archiv experimentů nelze otevřít: {exc}"
        ) from exc
    model_paths = registry.model_paths() if registry else {
        "bf16": (args.model or DEFAULT_MODEL).expanduser().resolve(),
        HYBRID_MODEL_ID: (args.model_8bit or DEFAULT_8BIT_MODEL).expanduser().resolve(),
        "8bit": (args.model_8bit or DEFAULT_8BIT_MODEL).expanduser().resolve(),
        "4bit": (args.model_4bit or DEFAULT_QUANTIZED_MODEL).expanduser().resolve(),
        EUROLLM_ID: (args.model_eurollm9 or (
            DEFAULT_EUROLLM_MODEL
        )).expanduser().resolve(),
    }
    if initial_model not in model_paths:
        raise SystemExit(f"Unknown model ID {initial_model!r}; registered IDs: {', '.join(model_paths)}")
    model_path = model_paths[initial_model]
    state = ModelState(
        model_path=model_path,
        model_paths=model_paths,
        tuned_lens_root=(args.tuned_lens_root or run_data / "tuned_lens").expanduser().resolve(),
        jacobian_lens_root=(args.jacobian_lens_root or run_data / "jacobian_lens").expanduser().resolve(),
        active_model_id=initial_model,
        backend_name=backend_name,
        registry=registry,
        device=args.device,
        mock=args.mock,
    )
    load_initial = not args.setup_only and state._model_available_unlocked(initial_model, model_path)

    server = GardenServer(
        (args.host, args.port),
        state,
        experiment_store,
        model_admin_data_dir=run_data,
    )
    state.start(load_initial=load_initial)
    url = f"http://{args.host}:{args.port}"
    print(f"Pravděpodobnostní zahrada: {url}")
    if not load_initial:
        print(f"Model setup: {url}/sprava-modelu")
    if args.mock:
        print("Vývojový režim: demonstrační data.")
    elif load_initial:
        print("Model is loading in the background…")
    else:
        print("No weights are loaded or downloaded. Choose a model in Model management.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nZastavuji zahradu.")
    finally:
        server.server_close()
        state.stop()
        if state.worker_thread is not None:
            state.worker_thread.join(timeout=12)


if __name__ == "__main__":
    main()
