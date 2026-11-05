"""Validated Jacobian-Lens runtime artifacts for the Apertus garden.

This module deliberately does not import MLX at import time.  The 224 MiB
runtime tensors are loaded only when a compatible BF16 artifact is requested.
The PyTorch-to-MLX transfer probe is part of the artifact contract, not an
optional diagnostic: it establishes tokenizer, hook and layer alignment for
the locally converted model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ARTIFACT_FORMAT = "apertus.jacobian-lens-artifact"
ARTIFACT_SCHEMA_VERSION = 1
WEIGHTS_FORMAT = "apertus.jacobian-lens-weights"
PROBE_FORMAT = "apertus.jacobian-lens-transfer-probe"
PROBE_SCHEMA_VERSION = 1
BF16_MODEL_ID = "bf16"
EXPECTED_HF_REPO = "swiss-ai/Apertus-8B-2509"
EXPECTED_HF_REVISION = "3162c99675aa588097cecd4a24b9aa1f712af477"
EXPECTED_HIDDEN_SIZE = 4096
EXPECTED_VOCAB_SIZE = 131072
MODEL_LAYER_COUNT = 32
SUPPORTED_TARGET_LAYERS = (31, 32)
STATIC_TRANSPORT_RELATIVE_L2_MAX = 1.0e-4
STATIC_TRANSPORT_COSINE_MIN = 0.999999
RUNTIME_RESIDUAL_RELATIVE_L2_MAX = 5.0e-2
RUNTIME_RESIDUAL_COSINE_MIN = 0.995
RUNTIME_JACOBIAN_TOP20_OVERLAP_MIN = 0.60
RUNTIME_JACOBIAN_TOP20_OVERLAP_MEAN_MIN = 0.80
RUNTIME_JACOBIAN_MEAN_NONINFERIORITY_MARGIN = 0.05
RUNTIME_FINAL_TOP20_OVERLAP_MIN = 0.80
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class JacobianLensError(RuntimeError):
    """Base exception with a stable API code."""

    code = "jacobian_lens_error"


class JacobianLensUnavailableError(JacobianLensError):
    code = "jacobian_lens_unavailable"


class JacobianLensIncompatibleError(JacobianLensError):
    code = "jacobian_lens_incompatible"


class JacobianLensChangedError(JacobianLensError):
    code = "jacobian_lens_changed"


class ArtifactValidationError(JacobianLensError):
    code = "invalid_jacobian_lens_artifact"


class TransferProbeError(JacobianLensError):
    code = "jacobian_lens_transfer_probe_failed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _detached_json(value: Any, field: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"{field} smí obsahovat pouze konečné JSON hodnoty."
        ) from exc


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{field} musí být JSON objekt.")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{field} musí být neprázdný řetězec.")
    return value


def _safe_id(value: Any, field: str) -> str:
    result = _text(value, field)
    if not SAFE_ID_PATTERN.fullmatch(result):
        raise ArtifactValidationError(f"{field} není bezpečný identifikátor.")
    return result


def _sha256(value: Any, field: str) -> str:
    result = _text(value, field)
    if not SHA256_PATTERN.fullmatch(result):
        raise ArtifactValidationError(f"{field} musí být SHA-256 v malém hexu.")
    return result


def _git_commit(value: Any, field: str) -> str:
    result = _text(value, field)
    if not GIT_COMMIT_PATTERN.fullmatch(result):
        raise ArtifactValidationError(f"{field} musí být 40znakový git commit.")
    return result


def _basename(value: Any, field: str) -> str:
    result = _text(value, field)
    if Path(result).name != result or "\\" in result or "\x00" in result:
        raise ArtifactValidationError(f"{field} musí být prosté jméno souboru.")
    return result


def _timestamp(value: Any, field: str) -> str:
    result = _text(value, field)
    candidate = result[:-1] + "+00:00" if result.endswith("Z") else result
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ArtifactValidationError(
            f"{field} musí být platné ISO-8601 datum."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ArtifactValidationError(f"{field} musí obsahovat časové pásmo.")
    return result


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ArtifactValidationError(f"{field} musí být kladné celé číslo.")
    return value


def _finite_number(value: Any, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ArtifactValidationError(f"{field} musí být konečné číslo.")
    return float(value)


def _file_entry(
    files: Mapping[str, Any],
    key: str,
    *,
    require_bytes: bool = False,
) -> dict[str, Any]:
    entry = _mapping(files.get(key), f"files.{key}")
    _basename(entry.get("name"), f"files.{key}.name")
    _sha256(entry.get("sha256"), f"files.{key}.sha256")
    if require_bytes:
        _positive_int(entry.get("bytes"), f"files.{key}.bytes")
    return entry


def runtime_file_entries(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return the files which must physically be in the runtime bundle."""

    files = _mapping(manifest.get("files"), "files")
    result = {
        "weights": _file_entry(files, "weights", require_bytes=True),
        "metrics": _file_entry(files, "metrics"),
        "transfer_probe_tensors": _file_entry(
            files,
            "transfer_probe_tensors",
        ),
        "transfer_probe_json": _file_entry(files, "transfer_probe_json"),
        "corpus_manifest": _file_entry(files, "corpus_manifest"),
        "run_manifest": _file_entry(files, "run_manifest"),
        "split_summaries": _file_entry(files, "split_summaries"),
        "first_prompt_benchmark": _file_entry(
            files,
            "first_prompt_benchmark",
        ),
    }
    names = [entry["name"] for entry in result.values()]
    if len(names) != len(set(names)) or "manifest.json" in names:
        raise ArtifactValidationError(
            "Runtime soubory musejí mít unikátní jména."
        )
    return result


def validate_manifest(
    submitted: Mapping[str, Any],
    *,
    artifact_dir: Path | None = None,
    verify_files: bool = False,
    expected_hidden_size: int = EXPECTED_HIDDEN_SIZE,
    expected_vocab_size: int = EXPECTED_VOCAB_SIZE,
) -> dict[str, Any]:
    """Validate and detach one J-Lens manifest.

    The model compatibility gate is intentionally stricter than a config-file
    hash: the converted MLX config is not byte-identical to Transformers 5.14.
    Runtime compatibility is established later by the bundled transfer probe.
    """

    manifest = _detached_json(submitted, "manifest")
    if not isinstance(manifest, dict):
        raise ArtifactValidationError("Manifest musí být JSON objekt.")
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ArtifactValidationError(
            f"format musí být {ARTIFACT_FORMAT!r}."
        )
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError(
            "Nepodporovaná verze Jacobian-Lens artefaktu."
        )
    artifact_id = _safe_id(manifest.get("artifact_id"), "artifact_id")
    _timestamp(manifest.get("created_at_utc"), "created_at_utc")
    _text(manifest.get("artifact_scope"), "artifact_scope")

    compatibility = _mapping(
        manifest.get("compatibility"),
        "compatibility",
    )
    if compatibility.get("allowed_model_modes") != [BF16_MODEL_ID]:
        raise ArtifactValidationError(
            "Tento J-Lens artefakt musí být výhradně BF16."
        )
    if compatibility.get("hybrid_transfer") != "not_validated":
        raise ArtifactValidationError(
            "Hybridní přenos nesmí být bez auditu deklarován jako validní."
        )
    if compatibility.get("mlx_enablement_requires_transfer_probe_match") is not True:
        raise ArtifactValidationError(
            "MLX použití musí vyžadovat shodu přenosové sondy."
        )

    reader = _mapping(manifest.get("reader"), "reader")
    expected_reader = {
        "kind": "average_jacobian_lens",
        "estimator": "current_and_future_targets",
        "hook": "post_decoder_block_residual",
        "layer_numbering": "one_based",
        "transport": "residual @ jacobian.T",
        "readout": "lm_head(final_rms_norm(transported_residual))",
    }
    for key, expected in expected_reader.items():
        if reader.get(key) != expected:
            raise ArtifactValidationError(
                f"reader.{key} neodpovídá runtime kontraktu."
            )
    target_layer = reader.get("target_layer")
    if type(target_layer) is not int or target_layer not in SUPPORTED_TARGET_LAYERS:
        raise ArtifactValidationError(
            "reader.target_layer musí být podporovaný post-block target L31 nebo L32."
        )
    paper_primary = reader.get("paper_primary_target_variant")
    if (
        paper_primary is not None
        and (
            not isinstance(paper_primary, bool)
            or paper_primary is not (target_layer == 31)
        )
    ):
        raise ArtifactValidationError(
            "reader.paper_primary_target_variant neodpovídá targetu."
        )
    if reader.get("hidden_size") != expected_hidden_size:
        raise ArtifactValidationError(
            f"reader.hidden_size musí být {expected_hidden_size}."
        )
    if reader.get("vocab_size") != expected_vocab_size:
        raise ArtifactValidationError(
            f"reader.vocab_size musí být {expected_vocab_size}."
        )
    layers = reader.get("source_layers")
    if (
        not isinstance(layers, list)
        or not layers
        or any(
            type(layer) is not int
            or not 1 <= layer < target_layer
            for layer in layers
        )
        or layers != sorted(set(layers))
    ):
        raise ArtifactValidationError(
            "reader.source_layers musejí být vzestupné, unikátní a před targetem."
        )
    for field in ("n_prompts", "max_seq_len", "skip_first"):
        _positive_int(reader.get(field), f"reader.{field}")

    model = _mapping(manifest.get("model"), "model")
    if model.get("hf_repo") != EXPECTED_HF_REPO:
        raise ArtifactValidationError("Artefakt patří jinému HF modelu.")
    if model.get("hf_revision") != EXPECTED_HF_REVISION:
        raise ArtifactValidationError("Artefakt patří jiné revizi Apertusu.")
    if model.get("mode") != BF16_MODEL_ID:
        raise ArtifactValidationError("Artefakt nebyl fitován nad BF16.")
    if model.get("architectures") != ["ApertusForCausalLM"]:
        raise ArtifactValidationError("Neočekávaná architektura modelu.")
    if model.get("tie_word_embeddings") is not False:
        raise ArtifactValidationError("Apertus musí mít nevázaný lm_head.")
    _sha256(model.get("config_sha256"), "model.config_sha256")
    _sha256(model.get("tokenizer_sha256"), "model.tokenizer_sha256")

    corpus = _mapping(manifest.get("corpus"), "corpus")
    if corpus.get("languages") != ["cs", "en"]:
        raise ArtifactValidationError(
            "Pilotní korpus musí deklarovat jazyky ['cs', 'en']."
        )
    _sha256(corpus.get("corpus_sha256"), "corpus.corpus_sha256")

    files = _mapping(manifest.get("files"), "files")
    entries = runtime_file_entries(manifest)
    weights = entries["weights"]
    if weights.get("dtype") != "float16":
        raise ArtifactValidationError("Jacobiany musejí být uloženy ve FP16.")
    expected_weight_keys = [
        f"layer_{layer:02d}.jacobian" for layer in layers
    ]
    if weights.get("keys") != expected_weight_keys:
        raise ArtifactValidationError(
            "Seznam matic neodpovídá reader.source_layers."
        )
    probe_tensors = entries["transfer_probe_tensors"]
    if (
        probe_tensors.get("residual_dtype") != "bfloat16"
        or probe_tensors.get("transported_dtype") != "float32"
    ):
        raise ArtifactValidationError(
            "Přenosová sonda musí mít BF16 rezidua a FP32 transport."
        )
    canonical_master = _mapping(
        files.get("canonical_master"),
        "files.canonical_master",
    )
    _basename(
        canonical_master.get("name"),
        "files.canonical_master.name",
    )
    _sha256(
        canonical_master.get("sha256"),
        "files.canonical_master.sha256",
    )

    software = _mapping(manifest.get("software"), "software")
    _text(software.get("transformers"), "software.transformers")
    fit = _mapping(manifest.get("fit"), "fit")
    _git_commit(fit.get("jlens_commit"), "fit.jlens_commit")

    limitations = manifest.get("limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(item, str) or not item for item in limitations)
    ):
        raise ArtifactValidationError(
            "limitations musí být neprázdný seznam textů."
        )

    if artifact_dir is not None:
        if artifact_dir.name != artifact_id:
            raise ArtifactValidationError(
                "artifact_id musí odpovídat názvu adresáře."
            )
        for key, entry in entries.items():
            path = artifact_dir / entry["name"]
            if not path.is_file():
                raise ArtifactValidationError(
                    f"Chybí runtime soubor {entry['name']}."
                )
            if key == "weights" and path.stat().st_size != entry["bytes"]:
                raise ArtifactValidationError(
                    "Velikost jacobians.safetensors nesouhlasí."
                )
            if verify_files and sha256_file(path) != entry["sha256"]:
                raise ArtifactValidationError(
                    f"SHA-256 souboru {entry['name']} nesouhlasí."
                )
    return manifest


def public_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    files = runtime_file_entries(manifest)
    return {
        "artifact_id": manifest.get("artifact_id"),
        "artifact_scope": manifest.get("artifact_scope"),
        "canonical_manifest_sha256": canonical_json_sha256(manifest),
        "created_at_utc": manifest.get("created_at_utc"),
        "schema_version": manifest.get("schema_version"),
        "compatibility": manifest.get("compatibility"),
        "reader": manifest.get("reader"),
        "model": manifest.get("model"),
        "corpus": manifest.get("corpus"),
        "weights": {
            "sha256": files["weights"]["sha256"],
            "bytes": files["weights"]["bytes"],
            "dtype": files["weights"].get("dtype"),
        },
        "metrics": {
            "sha256": files["metrics"]["sha256"],
        },
        "transfer_probe": {
            "json_sha256": files["transfer_probe_json"]["sha256"],
            "tensors_sha256": files["transfer_probe_tensors"]["sha256"],
        },
        "limitations": manifest.get("limitations", []),
    }


def _dtype_name(value: Any) -> str:
    name = str(getattr(value, "dtype", ""))
    return name.removeprefix("torch.").removeprefix("mlx.core.")


def validate_bf16_runtime_contract(
    model: Any,
    *,
    model_id: str,
    manifest: Mapping[str, Any],
) -> None:
    """Reject a quantized body before the more expensive transfer probe."""

    if model_id != BF16_MODEL_ID:
        raise JacobianLensIncompatibleError(
            "J-Lens runtime vyžaduje aktivní BF16 model."
        )
    if (
        getattr(model, "model_type", None) != "apertus"
        or not hasattr(model, "model")
        or not hasattr(model, "lm_head")
    ):
        raise JacobianLensIncompatibleError(
            "J-Lens runtime očekává Apertus s odděleným lm_head."
        )
    body = model.model
    layers = getattr(body, "layers", None)
    args = getattr(model, "args", None)
    if (
        layers is None
        or len(layers) != MODEL_LAYER_COUNT
        or args is None
        or int(getattr(args, "hidden_size", -1))
        != int(manifest["reader"]["hidden_size"])
        or int(getattr(args, "vocab_size", -1))
        != int(manifest["reader"]["vocab_size"])
        or not hasattr(body, "norm")
        or not hasattr(body, "embed_tokens")
    ):
        raise JacobianLensIncompatibleError(
            "Struktura aktivního modelu neodpovídá J-Lens manifestu."
        )
    for label, weight in (
        ("embed_tokens", getattr(body.embed_tokens, "weight", None)),
        ("lm_head", getattr(model.lm_head, "weight", None)),
    ):
        if weight is None or _dtype_name(weight) not in {"bfloat16", "bf16"}:
            raise JacobianLensIncompatibleError(
                f"{label} aktivního modelu není BF16."
            )


def _array_finite(value: Any) -> bool:
    try:
        array = np.asarray(value)
        return bool(np.isfinite(array.astype(np.float32, copy=False)).all())
    except Exception:
        import mlx.core as mx

        converted = value.astype(mx.float32)
        finite = mx.all(mx.isfinite(converted))
        mx.eval(converted, finite)
        return bool(finite.item())


def validate_weight_shapes(
    manifest: Mapping[str, Any],
    weights: Mapping[str, Any],
) -> None:
    reader = _mapping(manifest.get("reader"), "reader")
    hidden_size = int(reader["hidden_size"])
    expected = {
        f"layer_{layer:02d}.jacobian": (hidden_size, hidden_size)
        for layer in reader["source_layers"]
    }
    if set(weights) != set(expected):
        missing = sorted(set(expected) - set(weights))
        extra = sorted(set(weights) - set(expected))
        raise ArtifactValidationError(
            f"Neplatné klíče Jacobianů; chybí {missing}, navíc {extra}."
        )
    for key, shape in expected.items():
        value = weights[key]
        if tuple(int(item) for item in value.shape) != shape:
            raise ArtifactValidationError(
                f"{key} má tvar {tuple(value.shape)}, očekává se {shape}."
            )
        if _dtype_name(value) not in {"float16", "f16"}:
            raise ArtifactValidationError(f"{key} není FP16.")
        if not _array_finite(value):
            raise ArtifactValidationError(f"{key} obsahuje NaN nebo inf.")


def validate_probe_payload(
    manifest: Mapping[str, Any],
    probe_json: Mapping[str, Any],
    probe_tensors: Mapping[str, Any],
) -> dict[str, Any]:
    probe = _detached_json(probe_json, "transfer_probe.json")
    if not isinstance(probe, dict):
        raise ArtifactValidationError("transfer_probe.json musí být objekt.")
    if (
        probe.get("format") != PROBE_FORMAT
        or probe.get("schema_version") != PROBE_SCHEMA_VERSION
    ):
        raise ArtifactValidationError("Neplatný formát přenosové sondy.")
    model = manifest["model"]
    if (
        probe.get("model_id") != model["hf_repo"]
        or probe.get("model_revision") != model["hf_revision"]
        or probe.get("tokenizer_sha256") != model["tokenizer_sha256"]
    ):
        raise ArtifactValidationError(
            "Identita přenosové sondy neodpovídá manifestu."
        )
    records = probe.get("records")
    if not isinstance(records, list) or not records:
        raise ArtifactValidationError(
            "Přenosová sonda musí obsahovat alespoň jeden prompt."
        )
    layers = manifest["reader"]["source_layers"]
    hidden_size = manifest["reader"]["hidden_size"]
    expected_keys: set[str] = set()
    for prompt_index, record_value in enumerate(records):
        record = _mapping(
            record_value,
            f"transfer_probe.records[{prompt_index}]",
        )
        if record.get("prompt_index") != prompt_index:
            raise ArtifactValidationError(
                "Prompt indexy přenosové sondy nejsou kanonické."
            )
        _text(record.get("prompt"), f"probe prompt {prompt_index}")
        input_ids = record.get("input_ids")
        if (
            not isinstance(input_ids, list)
            or not input_ids
            or any(type(item) is not int or item < 0 for item in input_ids)
        ):
            raise ArtifactValidationError("Neplatné token IDs v sondě.")
        if record.get("last_token_id") != input_ids[-1]:
            raise ArtifactValidationError("last_token_id sondy nesouhlasí.")
        input_key = f"prompt_{prompt_index:02d}.input_ids"
        expected_keys.add(input_key)
        if input_key not in probe_tensors:
            raise ArtifactValidationError(f"Chybí {input_key}.")
        tensor_ids = np.asarray(probe_tensors[input_key]).astype(np.int64)
        if tensor_ids.ndim != 1 or tensor_ids.tolist() != input_ids:
            raise ArtifactValidationError(
                "JSON a tensorová token IDs sondy se liší."
            )
        layer_rows = record.get("layers")
        if (
            not isinstance(layer_rows, list)
            or [row.get("layer") for row in layer_rows] != layers
        ):
            raise ArtifactValidationError(
                "Vrstvy přenosové sondy neodpovídají manifestu."
            )
        for layer in layers:
            for suffix, dtype in (
                ("residual", {"bfloat16", "bf16"}),
                ("transported", {"float32", "f32"}),
            ):
                key = (
                    f"prompt_{prompt_index:02d}."
                    f"layer_{layer:02d}.{suffix}"
                )
                expected_keys.add(key)
                value = probe_tensors.get(key)
                if value is None:
                    raise ArtifactValidationError(f"Chybí {key}.")
                if tuple(int(item) for item in value.shape) != (hidden_size,):
                    raise ArtifactValidationError(f"{key} má neplatný tvar.")
                if _dtype_name(value) not in dtype:
                    raise ArtifactValidationError(f"{key} má neplatný dtype.")
                if not _array_finite(value):
                    raise ArtifactValidationError(
                        f"{key} obsahuje NaN nebo inf."
                    )
    if set(probe_tensors) != expected_keys:
        raise ArtifactValidationError(
            "Přenosová sonda obsahuje chybějící nebo neočekávané tensory."
        )
    return probe


def apply_jacobian(
    state: Any,
    *,
    layer: int,
    manifest: Mapping[str, Any],
    weights: Mapping[str, Any],
) -> Any:
    """Apply ``residual @ J.T`` in FP32 and keep the result in FP32."""

    import mlx.core as mx

    if layer not in manifest["reader"]["source_layers"]:
        raise ArtifactValidationError(f"Vrstva L{layer} není v artefaktu.")
    hidden_size = int(manifest["reader"]["hidden_size"])
    if not getattr(state, "shape", ()) or int(state.shape[-1]) != hidden_size:
        raise ArtifactValidationError(
            f"Reziduál musí mít poslední rozměr {hidden_size}."
        )
    key = f"layer_{layer:02d}.jacobian"
    jacobian = weights.get(key)
    if jacobian is None:
        raise ArtifactValidationError(f"V artefaktu chybí {key}.")
    return mx.matmul(
        state.astype(mx.float32),
        mx.transpose(jacobian.astype(mx.float32)),
    )


@dataclass(frozen=True)
class VectorComparison:
    relative_l2: float
    cosine: float
    max_absolute_error: float


@dataclass(frozen=True)
class RuntimeProbeReport:
    kind: str
    passed: bool
    thresholds: dict[str, float]
    token_checks: tuple[dict[str, Any], ...]
    residual_checks: tuple[dict[str, Any], ...]
    worst_relative_l2: float | None
    worst_cosine: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_passed(self) -> None:
        if not self.passed:
            raise TransferProbeError(
                "PyTorch BF16 → MLX přenosová sonda neprošla."
            )


def compare_vectors(actual: Any, expected: Any) -> VectorComparison:
    def as_float64(value: Any) -> np.ndarray:
        try:
            return np.asarray(value, dtype=np.float64).reshape(-1)
        except Exception:
            import mlx.core as mx

            converted = value.astype(mx.float32)
            mx.eval(converted)
            return np.asarray(converted, dtype=np.float64).reshape(-1)

    left = as_float64(actual)
    right = as_float64(expected)
    if left.shape != right.shape or left.size == 0:
        raise TransferProbeError("Porovnávané vektory mají jiný tvar.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise TransferProbeError("Porovnávané vektory obsahují NaN nebo inf.")
    difference = left - right
    right_norm = float(np.linalg.norm(right))
    left_norm = float(np.linalg.norm(left))
    difference_norm = float(np.linalg.norm(difference))
    if right_norm == 0.0:
        relative_l2 = 0.0 if difference_norm == 0.0 else math.inf
    else:
        relative_l2 = difference_norm / right_norm
    if left_norm == 0.0 or right_norm == 0.0:
        cosine = 1.0 if difference_norm == 0.0 else 0.0
    else:
        cosine = float(np.dot(left, right) / (left_norm * right_norm))
        cosine = min(1.0, max(-1.0, cosine))
    return VectorComparison(
        relative_l2=relative_l2,
        cosine=cosine,
        max_absolute_error=float(np.max(np.abs(difference))),
    )


def _default_transport(residual: Any, jacobian: Any) -> Any:
    try:
        left = np.asarray(residual, dtype=np.float32)
        matrix = np.asarray(jacobian, dtype=np.float32)
        return left @ matrix.T
    except Exception:
        import mlx.core as mx

        result = mx.matmul(
            residual.astype(mx.float32),
            mx.transpose(jacobian.astype(mx.float32)),
        )
        mx.eval(result)
        return result


def static_transport_probe_report(
    manifest: Mapping[str, Any],
    weights: Mapping[str, Any],
    probe_tensors: Mapping[str, Any],
    probe_json: Mapping[str, Any],
    *,
    transport: Callable[[Any, Any], Any] = _default_transport,
    relative_l2_max: float = STATIC_TRANSPORT_RELATIVE_L2_MAX,
    cosine_min: float = STATIC_TRANSPORT_COSINE_MIN,
) -> dict[str, Any]:
    """Check matrix orientation and dtype independently of the 8B model."""

    probe = validate_probe_payload(manifest, probe_json, probe_tensors)
    checks = []
    for record in probe["records"]:
        prompt_index = int(record["prompt_index"])
        for layer in manifest["reader"]["source_layers"]:
            prefix = f"prompt_{prompt_index:02d}.layer_{layer:02d}"
            actual = transport(
                probe_tensors[f"{prefix}.residual"],
                weights[f"layer_{layer:02d}.jacobian"],
            )
            comparison = compare_vectors(
                actual,
                probe_tensors[f"{prefix}.transported"],
            )
            passed = (
                comparison.relative_l2 <= relative_l2_max
                and comparison.cosine >= cosine_min
            )
            checks.append(
                {
                    "prompt_index": prompt_index,
                    "layer": layer,
                    "passed": passed,
                    **asdict(comparison),
                }
            )
    return {
        "kind": "static_transport_probe",
        "passed": all(check["passed"] for check in checks),
        "thresholds": {
            "relative_l2_max": relative_l2_max,
            "cosine_min": cosine_min,
        },
        "checks": checks,
        "worst_relative_l2": max(
            check["relative_l2"] for check in checks
        ),
        "worst_cosine": min(check["cosine"] for check in checks),
    }


def runtime_probe_report(
    manifest: Mapping[str, Any],
    probe_json: Mapping[str, Any],
    probe_tensors: Mapping[str, Any],
    *,
    actual_input_ids: Mapping[int, Sequence[int]],
    actual_residuals: Mapping[tuple[int, int], Any],
    relative_l2_max: float = RUNTIME_RESIDUAL_RELATIVE_L2_MAX,
    cosine_min: float = RUNTIME_RESIDUAL_COSINE_MIN,
) -> RuntimeProbeReport:
    """Compare MLX tokenization and post-block residuals to PyTorch BF16."""

    probe = validate_probe_payload(manifest, probe_json, probe_tensors)
    token_checks = []
    residual_checks = []
    for record in probe["records"]:
        prompt_index = int(record["prompt_index"])
        expected_ids = list(record["input_ids"])
        found_ids = list(actual_input_ids.get(prompt_index, ()))
        token_checks.append(
            {
                "prompt_index": prompt_index,
                "passed": found_ids == expected_ids,
                "expected": expected_ids,
                "actual": found_ids,
            }
        )
        for layer in manifest["reader"]["source_layers"]:
            actual = actual_residuals.get((prompt_index, layer))
            if actual is None:
                residual_checks.append(
                    {
                        "prompt_index": prompt_index,
                        "layer": layer,
                        "passed": False,
                        "error": "missing_runtime_residual",
                    }
                )
                continue
            key = (
                f"prompt_{prompt_index:02d}."
                f"layer_{layer:02d}.residual"
            )
            try:
                comparison = compare_vectors(actual, probe_tensors[key])
            except TransferProbeError as exc:
                residual_checks.append(
                    {
                        "prompt_index": prompt_index,
                        "layer": layer,
                        "passed": False,
                        "error": str(exc),
                    }
                )
                continue
            passed = (
                comparison.relative_l2 <= relative_l2_max
                and comparison.cosine >= cosine_min
            )
            residual_checks.append(
                {
                    "prompt_index": prompt_index,
                    "layer": layer,
                    "passed": passed,
                    **asdict(comparison),
                }
            )
    measured = [
        check for check in residual_checks if "relative_l2" in check
    ]
    return RuntimeProbeReport(
        kind="pytorch_bf16_to_mlx_runtime_probe",
        passed=(
            all(check["passed"] for check in token_checks)
            and all(check["passed"] for check in residual_checks)
        ),
        thresholds={
            "relative_l2_max": relative_l2_max,
            "cosine_min": cosine_min,
        },
        token_checks=tuple(token_checks),
        residual_checks=tuple(residual_checks),
        worst_relative_l2=(
            max(check["relative_l2"] for check in measured)
            if measured
            else None
        ),
        worst_cosine=(
            min(check["cosine"] for check in measured)
            if measured
            else None
        ),
    )


build_runtime_probe_report = runtime_probe_report


@dataclass(frozen=True)
class ArtifactMatch:
    directory: Path
    manifest: dict[str, Any]


class JacobianLensCatalog:
    """Discover, verify and lazily load BF16 Jacobian-Lens artifacts."""

    def __init__(
        self,
        root: Path,
        *,
        tensor_loader: Callable[[Path], Mapping[str, Any]] | None = None,
        expected_hidden_size: int = EXPECTED_HIDDEN_SIZE,
        expected_vocab_size: int = EXPECTED_VOCAB_SIZE,
    ):
        self.root = root
        self._tensor_loader = tensor_loader
        self._expected_hidden_size = expected_hidden_size
        self._expected_vocab_size = expected_vocab_size
        self._loaded_key: tuple[str, str] | None = None
        self._loaded: tuple[
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ] | None = None
        self._invalid: dict[tuple[str, str], str] = {}
        self._runtime_attestations: set[tuple[str, str, str]] = set()

    def clear_runtime_cache(self) -> None:
        self._loaded_key = None
        self._loaded = None
        self._runtime_attestations.clear()

    def _load_tensors(self, path: Path) -> dict[str, Any]:
        if self._tensor_loader is not None:
            return dict(self._tensor_loader(path))
        import mlx.core as mx

        return {str(key): value for key, value in mx.load(str(path)).items()}

    @staticmethod
    def _selection_key(match: ArtifactMatch) -> tuple[bool, str, str]:
        # Adding a paired target must not silently replace the established
        # final-target reader. Explicit artifact selection still wins.
        return (
            match.manifest["reader"]["target_layer"] == MODEL_LAYER_COUNT,
            match.manifest["created_at_utc"],
            match.manifest["artifact_id"],
        )

    def _manifests(self) -> list[ArtifactMatch]:
        if not self.root.is_dir():
            return []
        matches = []
        for path in sorted(self.root.glob("*/manifest.json")):
            try:
                submitted = json.loads(path.read_text(encoding="utf-8"))
                manifest = validate_manifest(
                    submitted,
                    artifact_dir=path.parent,
                    verify_files=False,
                    expected_hidden_size=self._expected_hidden_size,
                    expected_vocab_size=self._expected_vocab_size,
                )
            except (OSError, json.JSONDecodeError, JacobianLensError):
                continue
            matches.append(ArtifactMatch(path.parent, manifest))
        return matches

    @staticmethod
    def mock_status() -> dict[str, Any]:
        return {
            "status": "ready",
            "message": "Vývojová simulace J-Lens je připravena.",
            "content_verification": "mock",
        }

    def _public_artifact(
        self,
        match: ArtifactMatch,
        *,
        model_fingerprint: str | None,
    ) -> dict[str, Any]:
        public = public_manifest(match.manifest)
        metrics_entry = runtime_file_entries(match.manifest)["metrics"]
        metrics_path = match.directory / metrics_entry["name"]
        split_agreement: list[dict[str, Any]] = []
        heldout_summary: list[dict[str, Any]] = []
        readout_collapse_summary: list[dict[str, Any]] = []
        paired_target_experiment: dict[str, Any] = {}
        if (
            metrics_path.is_file()
            and sha256_file(metrics_path) == metrics_entry["sha256"]
        ):
            try:
                metrics_payload = json.loads(
                    metrics_path.read_text(encoding="utf-8")
                )
                if (
                    isinstance(metrics_payload, dict)
                    and metrics_payload.get("format")
                    == "apertus.jacobian-lens-metrics"
                    and metrics_payload.get("schema_version") == 1
                ):
                    split_agreement = list(
                        metrics_payload.get("split_agreement") or []
                    )
                    heldout_summary = list(
                        metrics_payload.get("heldout_summary") or []
                    )
                    readout_collapse_summary = list(
                        metrics_payload.get("readout_collapse_summary") or []
                    )
                    paired = metrics_payload.get("paired_target_experiment")
                    if isinstance(paired, dict):
                        paired_target_experiment = _detached_json(
                            paired,
                            "metrics.paired_target_experiment",
                        )
            except (
                OSError,
                json.JSONDecodeError,
                TypeError,
                ArtifactValidationError,
            ):
                split_agreement = []
                heldout_summary = []
                readout_collapse_summary = []
                paired_target_experiment = {}
        public["split_agreement"] = split_agreement
        public["heldout_summary"] = heldout_summary
        public["readout_collapse_summary"] = readout_collapse_summary
        public["paired_target_experiment"] = paired_target_experiment
        weight_sha = match.manifest["files"]["weights"]["sha256"]
        attested = (
            model_fingerprint is not None
            and (
                match.manifest["artifact_id"],
                weight_sha,
                model_fingerprint,
            )
            in self._runtime_attestations
        )
        public["runtime_status"] = "ready" if attested else "pending_probe"
        public["content_verification"] = "verified" if attested else "pending"
        return public

    def status(
        self,
        *,
        model_id: str,
        mock: bool = False,
        model_fingerprint: str | None = None,
        expected_artifact_id: str | None = None,
    ) -> dict[str, Any]:
        if mock:
            return self.mock_status()
        matches = self._manifests()
        candidates = [
            match
            for match in matches
            if (
                match.manifest["artifact_id"],
                match.manifest["files"]["weights"]["sha256"],
            )
            not in self._invalid
        ]
        public_candidates = [
            self._public_artifact(
                match,
                model_fingerprint=model_fingerprint,
            )
            for match in candidates
        ]
        if not candidates:
            if matches:
                latest = max(
                    matches,
                    key=lambda item: item.manifest["created_at_utc"],
                )
                key = (
                    latest.manifest["artifact_id"],
                    latest.manifest["files"]["weights"]["sha256"],
                )
                return {
                    "status": "invalid",
                    "message": self._invalid.get(
                        key,
                        "J-Lens artefakt selhal při ověření.",
                    ),
                    "content_verification": "failed",
                    "artifacts": [],
                }
            return {
                "status": "missing",
                "message": "J-Lens artefakt není nainstalován.",
                "content_verification": "not_started",
                "artifacts": [],
            }
        selection = candidates
        if expected_artifact_id is not None:
            selection = [
                match
                for match in candidates
                if match.manifest["artifact_id"] == expected_artifact_id
            ]
            if not selection:
                invalid_match = next(
                    (
                        match
                        for match in matches
                        if match.manifest["artifact_id"]
                        == expected_artifact_id
                    ),
                    None,
                )
                if invalid_match is not None:
                    invalid_key = (
                        invalid_match.manifest["artifact_id"],
                        invalid_match.manifest["files"]["weights"]["sha256"],
                    )
                    return {
                        "status": "invalid",
                        "message": self._invalid.get(
                            invalid_key,
                            "Požadovaný J-Lens artefakt selhal při ověření.",
                        ),
                        "content_verification": "failed",
                        "artifact_id": expected_artifact_id,
                        "artifacts": public_candidates,
                    }
                return {
                    "status": "missing",
                    "message": "Požadovaný J-Lens artefakt není nainstalován.",
                    "content_verification": "not_started",
                    "artifact_id": expected_artifact_id,
                    "artifacts": public_candidates,
                }
        selected = max(
            selection,
            key=self._selection_key,
        )
        public = next(
            item
            for item in public_candidates
            if item["artifact_id"] == selected.manifest["artifact_id"]
        )
        selected_payload = {
            "calibration": public,
            "artifact": public,
            "artifact_id": public["artifact_id"],
            "source_layers": list(public["reader"]["source_layers"]),
            "target_layer": public["reader"]["target_layer"],
            "split_agreement": public["split_agreement"],
            "heldout_summary": public["heldout_summary"],
            "readout_collapse_summary": public[
                "readout_collapse_summary"
            ],
            "paired_target_experiment": public[
                "paired_target_experiment"
            ],
            "artifacts": public_candidates,
        }
        if model_id != BF16_MODEL_ID:
            return {
                "status": "incompatible",
                "message": "J-Lens byl ověřen pouze pro BF16 Apertus.",
                "content_verification": "not_started",
                **selected_payload,
            }
        attested = public["runtime_status"] == "ready"
        return {
            "status": "ready" if attested else "pending_probe",
            "message": (
                "J-Lens je připraven a přenosová sonda souhlasí."
                if attested
                else "J-Lens čeká na BF16 přenosovou sondu."
            ),
            "content_verification": "verified" if attested else "pending",
            **selected_payload,
        }

    def load(
        self,
        *,
        model_id: str,
        expected_artifact_id: str | None = None,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        if model_id != BF16_MODEL_ID:
            raise JacobianLensIncompatibleError(
                "J-Lens lze načíst pouze nad BF16 Apertusem."
            )
        matches = self._manifests()
        if expected_artifact_id is not None:
            matches = [
                item
                for item in matches
                if item.manifest["artifact_id"] == expected_artifact_id
            ]
        if not matches:
            raise JacobianLensUnavailableError(
                "Požadovaný J-Lens artefakt není nainstalován."
            )
        selected = max(
            matches,
            key=self._selection_key,
        )
        artifact_id = selected.manifest["artifact_id"]
        entries = runtime_file_entries(selected.manifest)
        weight_sha = entries["weights"]["sha256"]
        cache_key = (artifact_id, weight_sha)
        if cache_key in self._invalid:
            raise ArtifactValidationError(self._invalid[cache_key])
        try:
            validate_manifest(
                selected.manifest,
                artifact_dir=selected.directory,
                verify_files=True,
                expected_hidden_size=self._expected_hidden_size,
                expected_vocab_size=self._expected_vocab_size,
            )
            if self._loaded_key != cache_key or self._loaded is None:
                weights = self._load_tensors(
                    selected.directory / entries["weights"]["name"]
                )
                probe_tensors = self._load_tensors(
                    selected.directory
                    / entries["transfer_probe_tensors"]["name"]
                )
                probe_json = json.loads(
                    (
                        selected.directory
                        / entries["transfer_probe_json"]["name"]
                    ).read_text(encoding="utf-8")
                )
                metrics = json.loads(
                    (
                        selected.directory / entries["metrics"]["name"]
                    ).read_text(encoding="utf-8")
                )
                metrics = _detached_json(metrics, "metrics.json")
                if (
                    not isinstance(metrics, dict)
                    or metrics.get("format")
                    != "apertus.jacobian-lens-metrics"
                    or metrics.get("schema_version") != 1
                ):
                    raise ArtifactValidationError(
                        "metrics.json má neplatný formát."
                    )
                validate_weight_shapes(selected.manifest, weights)
                validate_probe_payload(
                    selected.manifest,
                    probe_json,
                    probe_tensors,
                )
                static_report = static_transport_probe_report(
                    selected.manifest,
                    weights,
                    probe_tensors,
                    probe_json,
                )
                if not static_report["passed"]:
                    raise TransferProbeError(
                        "Statická kontrola residual @ J.T neprošla."
                    )
                self._loaded_key = cache_key
                self._loaded = (weights, probe_tensors, probe_json)
        except Exception as exc:
            self._loaded_key = None
            self._loaded = None
            message = str(exc) or "J-Lens artefakt nelze bezpečně načíst."
            self._invalid[cache_key] = message
            if isinstance(exc, JacobianLensError):
                raise
            raise ArtifactValidationError(message) from exc
        assert self._loaded is not None
        weights, probe_tensors, probe_json = self._loaded
        return selected.manifest, weights, probe_tensors, probe_json

    def mark_runtime_probe(
        self,
        *,
        artifact_id: str,
        weights_sha256: str,
        model_fingerprint: str,
        report: Mapping[str, Any] | RuntimeProbeReport,
    ) -> None:
        payload = (
            report.to_dict()
            if isinstance(report, RuntimeProbeReport)
            else report
        )
        if payload.get("kind") != "pytorch_bf16_to_mlx_runtime_probe":
            raise TransferProbeError("Neznámý typ runtime sondy.")
        if payload.get("passed") is not True:
            raise TransferProbeError(
                "Neúspěšnou runtime sondu nelze označit jako ověření."
            )
        _safe_id(artifact_id, "artifact_id")
        _sha256(weights_sha256, "weights_sha256")
        _sha256(model_fingerprint, "model_fingerprint")
        if self._loaded_key != (artifact_id, weights_sha256):
            raise JacobianLensChangedError(
                "Runtime sonda nepatří právě načtenému artefaktu."
            )
        thresholds = _mapping(
            payload.get("thresholds"),
            "runtime_probe.thresholds",
        )
        relative_limit = _finite_number(
            thresholds.get("relative_l2_max"),
            "runtime_probe.thresholds.relative_l2_max",
        )
        cosine_limit = _finite_number(
            thresholds.get("cosine_min"),
            "runtime_probe.thresholds.cosine_min",
        )
        if (
            relative_limit > RUNTIME_RESIDUAL_RELATIVE_L2_MAX
            or cosine_limit < RUNTIME_RESIDUAL_COSINE_MIN
        ):
            raise TransferProbeError(
                "Runtime sonda použila příliš volné tolerance."
            )
        for field in ("token_checks",):
            checks = payload.get(field)
            if (
                not isinstance(checks, (list, tuple))
                or not checks
                or any(
                    not isinstance(check, dict)
                    or check.get("passed") is not True
                    for check in checks
                )
            ):
                raise TransferProbeError(
                    f"Runtime sonda nemá platné úspěšné {field}."
                )
        residual_checks = payload.get("residual_checks")
        if (
            not isinstance(residual_checks, (list, tuple))
            or not residual_checks
            or any(not isinstance(check, dict) for check in residual_checks)
        ):
            raise TransferProbeError(
                "Runtime sonda nemá platnou diagnostiku reziduálů."
            )
        jacobian_checks = payload.get("jacobian_output_checks")
        if (
            not isinstance(jacobian_checks, (list, tuple))
            or not jacobian_checks
            or any(
                not isinstance(check, dict)
                or check.get("passed") is not True
                or check.get("top1_matches") is not True
                or _finite_number(
                    check.get("top20_overlap"),
                    "runtime_probe.jacobian_output_checks.top20_overlap",
                )
                < RUNTIME_JACOBIAN_TOP20_OVERLAP_MIN
                for check in jacobian_checks
            )
        ):
            raise TransferProbeError(
                "Runtime sonda nemá úspěšnou přímou kontrolu J-Lens výstupů."
            )
        jacobian_mean_overlap = sum(
            float(check["top20_overlap"]) for check in jacobian_checks
        ) / len(jacobian_checks)
        if jacobian_mean_overlap < RUNTIME_JACOBIAN_TOP20_OVERLAP_MEAN_MIN:
            raise TransferProbeError(
                "Průměrná shoda top-20 J-Lens výstupů je příliš nízká."
            )
        final_checks = payload.get("output_checks")
        if (
            not isinstance(final_checks, (list, tuple))
            or not final_checks
            or any(
                not isinstance(check, dict)
                or check.get("passed") is not True
                or _finite_number(
                    check.get("top20_overlap"),
                    "runtime_probe.output_checks.top20_overlap",
                )
                < RUNTIME_FINAL_TOP20_OVERLAP_MIN
                for check in final_checks
            )
        ):
            raise TransferProbeError(
                "Runtime sonda nemá kompatibilní finální top-20 výstupy."
            )
        final_mean_overlap = sum(
            float(check["top20_overlap"]) for check in final_checks
        ) / len(final_checks)
        if (
            jacobian_mean_overlap
            < final_mean_overlap
            - RUNTIME_JACOBIAN_MEAN_NONINFERIORITY_MARGIN
        ):
            raise TransferProbeError(
                "J-Lens rank-transfer je horší než mezibackendová baseline."
            )
        self._runtime_attestations.add(
            (artifact_id, weights_sha256, model_fingerprint)
        )
