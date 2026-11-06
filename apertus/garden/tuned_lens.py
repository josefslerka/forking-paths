"""Versioned tuned-lens pilot artifacts for the Apertus garden.

The live server imports this module without importing MLX.  MLX is loaded only
when a compatible artifact is actually used.  The artifact deliberately lives
outside the SQLite experiment archive: a saved experiment references immutable
weight digests, but never embeds weights or activations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


ARTIFACT_FORMAT = "apertus.tuned-lens-artifact"
ARTIFACT_SCHEMA_VERSION = 1
REFERENCE_LAYER = 32
DEFAULT_PILOT_LAYERS = (4, 16, 17, 22, 23, 24, 31)
SUPPORTED_TRANSLATORS = {
    "low_rank_residual_v1",
    "bias_only_residual_v1",
    "full_affine_residual_v1",
}
USABLE_VALIDATION_STATUSES = {"pilot", "validated"}
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MODEL_METADATA_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
)


class TunedLensError(RuntimeError):
    """Base error with a stable API code."""

    code = "tuned_lens_error"


class TunedLensUnavailableError(TunedLensError):
    code = "tuned_lens_unavailable"


class CalibrationIncompatibleError(TunedLensError):
    code = "calibration_incompatible"


class CalibrationChangedError(TunedLensError):
    code = "calibration_changed"


class ArtifactValidationError(TunedLensError):
    code = "invalid_tuned_lens_artifact"


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


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{field} musí být JSON objekt.")
    return value


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{field} musí být neprázdný řetězec.")
    return value


def _require_safe_id(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if not SAFE_ID_PATTERN.fullmatch(text):
        raise ArtifactValidationError(
            f"{field} smí obsahovat jen písmena, číslice, tečku, "
            "podtržítko a pomlčku."
        )
    return text


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ArtifactValidationError(f"{field} musí být SHA-256 v hexu.")
    return text


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_ready_acceptance(
    validation: Mapping[str, Any],
    *,
    selected_layers: list[int],
    dataset: Mapping[str, Any],
) -> None:
    acceptance = _require_mapping(
        validation.get("acceptance"),
        "validation.acceptance",
    )
    if acceptance.get("passed") is not True:
        raise ArtifactValidationError(
            "Přijatý artefakt musí mít acceptance.passed=true."
        )
    if acceptance.get("data_floor_passed") is not True:
        raise ArtifactValidationError(
            "Přijatý artefakt musí projít datovým minimem."
        )
    by_layer = _require_mapping(
        acceptance.get("by_layer"),
        "validation.acceptance.by_layer",
    )
    expected_layer_keys = {str(layer) for layer in selected_layers}
    if set(by_layer) != expected_layer_keys or any(
        value is not True for value in by_layer.values()
    ):
        raise ArtifactValidationError(
            "Přijímací podmínka musí explicitně projít ve všech vrstvách."
        )
    data_floor = _require_mapping(
        acceptance.get("data_floor"),
        "validation.acceptance.data_floor",
    )
    for split in ("train", "validation", "test"):
        split_data = _require_mapping(
            data_floor.get(split),
            f"validation.acceptance.data_floor.{split}",
        )
        for language in ("cs", "en"):
            values = _require_mapping(
                split_data.get(language),
                (
                    "validation.acceptance.data_floor."
                    f"{split}.{language}"
                ),
            )
            documents = values.get("documents")
            positions = values.get("positions")
            minimum_documents = 32 if split == "train" else 8
            minimum_positions = 4096 if split == "train" else 1
            if (
                type(documents) is not int
                or documents < minimum_documents
                or type(positions) is not int
                or positions < minimum_positions
            ):
                raise ArtifactValidationError(
                    "Datové minimum neodpovídá deklarovanému přijetí."
                )
            dataset_documents = dataset["document_counts"][split][language]
            if documents != dataset_documents:
                raise ArtifactValidationError(
                    "Datové minimum a dokumentový manifest si odporují."
                )

    held_out = _require_mapping(
        validation.get("held_out"),
        "validation.held_out",
    )
    if held_out.get("split") != "test":
        raise ArtifactValidationError(
            "Přijímací metriky musí pocházet z test splitu."
        )
    rows = held_out.get("by_layer")
    if not isinstance(rows, list):
        raise ArtifactValidationError(
            "validation.held_out.by_layer musí být seznam."
        )
    rows_by_layer: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("layer")) is not int:
            raise ArtifactValidationError(
                "Held-out vrstva musí být JSON objekt s číslem vrstvy."
            )
        layer = int(row["layer"])
        if layer in rows_by_layer:
            raise ArtifactValidationError(
                "Held-out metriky obsahují duplicitní vrstvu."
            )
        rows_by_layer[layer] = row
    if set(rows_by_layer) != set(selected_layers):
        raise ArtifactValidationError(
            "Held-out metriky musí pokrýt právě vybrané vrstvy."
        )
    for layer, row in rows_by_layer.items():
        languages = _require_mapping(
            row.get("by_language"),
            f"validation.held_out.L{layer}.by_language",
        )
        if set(languages) != {"cs", "en"}:
            raise ArtifactValidationError(
                "Held-out metriky musí odděleně obsahovat cs a en."
            )
        for language in ("cs", "en"):
            metrics = _require_mapping(
                languages.get(language),
                f"validation.held_out.L{layer}.{language}",
            )
            improvement = metrics.get("kl_improvement_bits")
            interval = metrics.get(
                "kl_improvement_ci95_document_bootstrap"
            )
            if (
                not _finite_number(improvement)
                or float(improvement) <= 0.0
                or not isinstance(interval, list)
                or len(interval) != 2
                or not all(_finite_number(value) for value in interval)
                or float(interval[0]) <= 0.0
                or float(interval[0]) > float(interval[1])
            ):
                raise ArtifactValidationError(
                    "Held-out KL zlepšení musí mít v cs i en "
                    "kladnou 95% bootstrapovou mez."
                )


def artifact_is_usable(manifest: Mapping[str, Any]) -> bool:
    validation = manifest["validation"]
    return (
        validation["status"] in USABLE_VALIDATION_STATUSES
        and validation["acceptance"]["passed"] is True
    )


def _validate_timestamp(value: Any, field: str) -> str:
    text = _require_text(value, field)
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ArtifactValidationError(
            f"{field} musí být platné ISO-8601 datum."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ArtifactValidationError(f"{field} musí obsahovat časové pásmo.")
    return text


def validate_manifest(
    submitted: Mapping[str, Any],
    *,
    artifact_dir: Path | None = None,
    verify_weights: bool = False,
) -> dict[str, Any]:
    """Validate and detach one manifest.

    ``verify_weights=False`` is intentionally cheap enough for ``/api/health``.
    The full weight digest is checked before first inference.
    """

    if not isinstance(submitted, dict):
        raise ArtifactValidationError("Manifest musí být JSON objekt.")
    try:
        manifest = json.loads(
            json.dumps(
                submitted,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            "Manifest musí obsahovat pouze konečné JSON hodnoty."
        ) from exc
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ArtifactValidationError(
            f"format musí být {ARTIFACT_FORMAT!r}."
        )
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError(
            "Nepodporovaná verze tuned-lens artefaktu."
        )
    artifact_id = _require_safe_id(manifest.get("artifact_id"), "artifact_id")
    _validate_timestamp(manifest.get("created_at"), "created_at")

    reader = _require_mapping(manifest.get("reader"), "reader")
    if reader.get("reference_layer") != REFERENCE_LAYER:
        raise ArtifactValidationError("Referenční vrstva musí být L32.")
    if reader.get("hook") != "post_decoder_block_residual":
        raise ArtifactValidationError(
            "Pilot očekává post-block reziduální proud."
        )
    translator_kind = reader.get("translator_kind")
    if translator_kind not in SUPPORTED_TRANSLATORS:
        raise ArtifactValidationError(
            f"Nepodporovaný translator_kind: {translator_kind!r}."
        )
    layers = reader.get("selected_layers")
    if (
        not isinstance(layers, list)
        or not layers
        or any(type(layer) is not int or not 1 <= layer < REFERENCE_LAYER for layer in layers)
        or layers != sorted(set(layers))
    ):
        raise ArtifactValidationError(
            "selected_layers musí být vzestupný unikátní seznam L1–L31."
        )
    hidden_size = reader.get("hidden_size")
    if hidden_size != 4096:
        raise ArtifactValidationError(
            "Apertus 8B tuned lens očekává hidden_size 4096."
        )
    rank = reader.get("rank")
    if translator_kind == "low_rank_residual_v1":
        if type(rank) is not int or not 1 <= rank <= hidden_size:
            raise ArtifactValidationError(
                "Low-rank translator musí mít platný reader.rank."
            )
    elif rank is not None:
        raise ArtifactValidationError(
            "reader.rank patří pouze low-rank translatoru."
        )
    expected_reader_kind = (
        "canonical_tuned_lens"
        if translator_kind == "full_affine_residual_v1"
        else "tuned_lens_style_probe"
    )
    if reader.get("kind") != expected_reader_kind:
        raise ArtifactValidationError(
            "reader.kind neodpovídá typu translatoru."
        )
    if reader.get("layer_numbering") != "one_based":
        raise ArtifactValidationError(
            "Číslování vrstev musí být one_based."
        )
    if reader.get("objective") != "full_vocab_kl_final_to_reader":
        raise ArtifactValidationError(
            "Pilot vyžaduje full-vocab KL(final || reader)."
        )
    if (
        reader.get("readout")
        != "lm_head(final_rms_norm(translated_residual))"
    ):
        raise ArtifactValidationError(
            "reader.readout neodpovídá produkční čtečce."
        )
    temperature = reader.get("temperature")
    if not _finite_number(temperature) or float(temperature) != 1.0:
        raise ArtifactValidationError(
            "Kalibrační teplota musí být přesně T=1."
        )

    model = _require_mapping(manifest.get("model"), "model")
    model_id = _require_safe_id(model.get("model_id"), "model.model_id")
    _require_sha256(
        model.get("metadata_fingerprint"),
        "model.metadata_fingerprint",
    )
    _require_sha256(
        model.get("content_fingerprint"),
        "model.content_fingerprint",
    )
    if model.get("quantization_mode") != model_id:
        raise ArtifactValidationError(
            "model.quantization_mode musí odpovídat model.model_id."
        )
    dataset = _require_mapping(manifest.get("dataset"), "dataset")
    _require_safe_id(dataset.get("dataset_id"), "dataset.dataset_id")
    if dataset.get("languages") != ["cs", "en"]:
        raise ArtifactValidationError(
            "Kalibrační dataset musí deklarovat jazyky ['cs', 'en']."
        )
    _require_sha256(
        dataset.get("document_manifest_sha256"),
        "dataset.document_manifest_sha256",
    )
    _require_sha256(
        dataset.get("source_file_sha256"),
        "dataset.source_file_sha256",
    )
    _require_sha256(
        dataset.get("state_cache_index_sha256"),
        "dataset.state_cache_index_sha256",
    )
    for field in (
        "split_seed",
        "sample_seed",
        "max_positions_per_document",
    ):
        if type(dataset.get(field)) is not int or dataset[field] <= 0:
            raise ArtifactValidationError(
                f"dataset.{field} musí být kladné celé číslo."
            )
    document_counts = _require_mapping(
        dataset.get("document_counts"),
        "dataset.document_counts",
    )
    for split in ("train", "validation", "test"):
        split_counts = _require_mapping(
            document_counts.get(split),
            f"dataset.document_counts.{split}",
        )
        for language in ("cs", "en"):
            count = split_counts.get(language)
            if type(count) is not int or count <= 0:
                raise ArtifactValidationError(
                    "Dataset musí mít kladný počet dokumentů v každém "
                    "splitu a jazyce."
                )
    validation = _require_mapping(manifest.get("validation"), "validation")
    status = validation.get("status")
    if status not in {"smoke", "unvalidated", *USABLE_VALIDATION_STATUSES}:
        raise ArtifactValidationError(
            "validation.status musí být smoke, unvalidated, pilot nebo validated."
        )
    acceptance = _require_mapping(
        validation.get("acceptance"),
        "validation.acceptance",
    )
    if not isinstance(acceptance.get("passed"), bool):
        raise ArtifactValidationError(
            "validation.acceptance.passed musí být boolean."
        )
    if status in USABLE_VALIDATION_STATUSES:
        _validate_ready_acceptance(
            validation,
            selected_layers=layers,
            dataset=dataset,
        )
    elif acceptance.get("passed") is not False:
        raise ArtifactValidationError(
            "Smoke/unvalidated artefakt nesmí deklarovat přijetí."
        )

    weights = _require_mapping(manifest.get("weights"), "weights")
    weight_name = _require_text(weights.get("file"), "weights.file")
    if Path(weight_name).name != weight_name:
        raise ArtifactValidationError(
            "weights.file musí být prosté jméno souboru."
        )
    expected_sha = _require_sha256(
        weights.get("sha256"),
        "weights.sha256",
    )
    expected_bytes = weights.get("bytes")
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ArtifactValidationError("weights.bytes musí být kladné celé číslo.")

    if artifact_dir is not None:
        if artifact_dir.name != artifact_id:
            raise ArtifactValidationError(
                "artifact_id musí odpovídat názvu adresáře artefaktu."
            )
        weight_path = artifact_dir / weight_name
        if not weight_path.is_file():
            raise ArtifactValidationError("Soubor vah podle manifestu chybí.")
        if weight_path.stat().st_size != expected_bytes:
            raise ArtifactValidationError("Velikost souboru vah nesouhlasí.")
        if verify_weights and sha256_file(weight_path) != expected_sha:
            raise ArtifactValidationError("SHA-256 souboru vah nesouhlasí.")
    metrics = _require_mapping(manifest.get("metrics"), "metrics")
    metrics_name = _require_text(metrics.get("file"), "metrics.file")
    if Path(metrics_name).name != metrics_name:
        raise ArtifactValidationError(
            "metrics.file musí být prosté jméno souboru."
        )
    metrics_sha = _require_sha256(
        metrics.get("sha256"),
        "metrics.sha256",
    )
    metrics_bytes = metrics.get("bytes")
    if type(metrics_bytes) is not int or metrics_bytes <= 0:
        raise ArtifactValidationError(
            "metrics.bytes musí být kladné celé číslo."
        )
    if artifact_dir is not None:
        metrics_path = artifact_dir / metrics_name
        if (
            not metrics_path.is_file()
            or metrics_path.stat().st_size != metrics_bytes
        ):
            raise ArtifactValidationError(
                "Soubor metrik podle manifestu chybí nebo má jinou velikost."
            )
        if verify_weights and sha256_file(metrics_path) != metrics_sha:
            raise ArtifactValidationError(
                "SHA-256 souboru metrik nesouhlasí."
            )
    return manifest


def public_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded, path-free subset safe for the browser and archive."""

    reader = _require_mapping(manifest.get("reader"), "reader")
    model = _require_mapping(manifest.get("model"), "model")
    dataset = _require_mapping(manifest.get("dataset"), "dataset")
    validation = _require_mapping(manifest.get("validation"), "validation")
    weights = _require_mapping(manifest.get("weights"), "weights")
    metrics = _require_mapping(manifest.get("metrics"), "metrics")
    return {
        "artifact_id": manifest.get("artifact_id"),
        "canonical_manifest_sha256": canonical_json_sha256(manifest),
        "created_at": manifest.get("created_at"),
        "schema_version": manifest.get("schema_version"),
        "reader": {
            key: reader.get(key)
            for key in (
                "kind",
                "translator_kind",
                "rank",
                "selected_layers",
                "reference_layer",
                "hook",
                "layer_numbering",
                "objective",
                "temperature",
                "hidden_size",
            )
            if key in reader
        },
        "model": {
            key: model.get(key)
            for key in (
                "model_id",
                "model_label",
                "metadata_fingerprint",
                "content_fingerprint",
                "quantization_mode",
            )
            if key in model
        },
        "dataset": dataset,
        "validation": validation,
        "weights": {
            "sha256": weights.get("sha256"),
            "bytes": weights.get("bytes"),
        },
        "metrics": {
            "sha256": metrics.get("sha256"),
            "bytes": metrics.get("bytes"),
        },
        "limitations": manifest.get("limitations", []),
    }


@dataclass(frozen=True)
class ArtifactMatch:
    directory: Path
    manifest: dict[str, Any]


class TunedLensCatalog:
    """Discover, validate and lazily load model-specific artifacts."""

    def __init__(self, root: Path):
        self.root = root
        self._loaded_key: tuple[str, str] | None = None
        self._loaded_weights: dict[str, Any] | None = None
        self._verified_artifacts: set[tuple[str, str, str]] = set()
        self._invalid_artifacts: dict[tuple[str, str], str] = {}

    def clear_runtime_cache(self) -> None:
        self._loaded_key = None
        self._loaded_weights = None
        self._verified_artifacts.clear()

    def _manifests(self) -> Iterable[ArtifactMatch]:
        if not self.root.is_dir():
            return []
        matches: list[ArtifactMatch] = []
        paths = [self.root / "manifest.json"] if (self.root / "manifest.json").is_file() else sorted(self.root.glob("*/manifest.json"))
        for path in paths:
            try:
                submitted = json.loads(path.read_text(encoding="utf-8"))
                manifest = validate_manifest(
                    submitted,
                    artifact_dir=path.parent,
                    verify_weights=False,
                )
            except (OSError, json.JSONDecodeError, TunedLensError):
                continue
            matches.append(ArtifactMatch(path.parent, manifest))
        return matches

    @staticmethod
    def mock_status() -> dict[str, Any]:
        layers = list(DEFAULT_PILOT_LAYERS)
        return {
            "status": "ready",
            "message": "Vývojová simulace RAW × TUNED je připravena.",
            "content_verification": "mock",
            "calibration": {
                "artifact_id": "mock-low-rank-pilot-v1",
                "canonical_manifest_sha256": "f" * 64,
                "created_at": "2026-07-28T00:00:00Z",
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "reader": {
                    "kind": "tuned_lens_style_probe",
                    "translator_kind": "low_rank_residual_v1",
                    "rank": 8,
                    "selected_layers": layers,
                    "reference_layer": REFERENCE_LAYER,
                    "hook": "post_decoder_block_residual",
                    "layer_numbering": "one_based",
                    "objective": "full_vocab_kl_final_to_reader",
                    "temperature": 1.0,
                    "hidden_size": 32,
                },
                "model": {
                    "model_id": "mock",
                    "model_label": "Vývojová simulace",
                    "metadata_fingerprint": "mock",
                    "content_fingerprint": "mock",
                    "quantization_mode": "mock",
                },
                "dataset": {
                    "dataset_id": "mock-balanced-cs-en",
                    "languages": ["cs", "en"],
                    "train_documents": 20,
                    "validation_documents": 8,
                    "test_documents": 8,
                    "train_positions": 2048,
                    "validation_positions": 512,
                    "test_positions": 512,
                },
                "validation": {
                    "status": "pilot",
                    "acceptance": {
                        "passed": True,
                        "criterion": "synthetic_mock_only",
                    },
                    "held_out": {
                        "split": "synthetic_test",
                        "by_layer": [
                            {
                                "layer": layer,
                                "positions": 512,
                                "raw_kl_bits": 0.8 / math.sqrt(layer),
                                "tuned_kl_bits": 0.48 / math.sqrt(layer),
                                "raw_js_bits": 0.16 / math.sqrt(layer),
                                "tuned_js_bits": 0.095 / math.sqrt(layer),
                                "raw_top1_agreement": min(0.98, 0.25 + layer / 45),
                                "tuned_top1_agreement": min(0.99, 0.42 + layer / 45),
                            }
                            for layer in layers
                        ],
                    },
                },
                "weights": {
                    "sha256": "0" * 64,
                    "bytes": 1,
                },
                "metrics": {
                    "sha256": "e" * 64,
                    "bytes": 1,
                },
                "limitations": [
                    "Pouze vývojová simulace bez interpretovatelné kalibrace."
                ],
            },
        }

    def status(
        self,
        *,
        model_id: str,
        metadata_fingerprint: str,
        mock: bool = False,
    ) -> dict[str, Any]:
        if mock:
            return self.mock_status()
        matches = list(self._manifests())
        same_model = [
            match
            for match in matches
            if match.manifest["model"]["model_id"] == model_id
        ]
        exact = [
            match
            for match in same_model
            if match.manifest["model"]["metadata_fingerprint"]
            == metadata_fingerprint
        ]
        if not exact:
            if same_model:
                return {
                    "status": "incompatible",
                    "message": (
                        "Kalibrace existuje, ale patří jiné sadě vah nebo "
                        "konfiguraci tohoto režimu."
                    ),
                    "content_verification": "not_started",
                }
            return {
                "status": "missing",
                "message": (
                    "Pro právě aktivní modelový režim zatím není kalibrace."
                ),
                "content_verification": "not_started",
            }
        usable_matches = [
            match
            for match in exact
            if artifact_is_usable(match.manifest)
            and (
                match.manifest["artifact_id"],
                match.manifest["weights"]["sha256"],
            )
            not in self._invalid_artifacts
        ]
        selected_pool = usable_matches or [
            match
            for match in exact
            if (
                match.manifest["artifact_id"],
                match.manifest["weights"]["sha256"],
            )
            not in self._invalid_artifacts
        ]
        if not selected_pool:
            selected = max(
                exact,
                key=lambda match: (
                    match.manifest.get("created_at", ""),
                    match.manifest.get("artifact_id", ""),
                ),
            )
            key = (
                selected.manifest["artifact_id"],
                selected.manifest["weights"]["sha256"],
            )
            return {
                "status": "invalid",
                "message": self._invalid_artifacts.get(
                    key,
                    "Kalibrační artefakt selhal při ověření integrity.",
                ),
                "content_verification": "failed",
                "calibration": public_manifest(selected.manifest),
            }
        selected = max(
            selected_pool,
            key=lambda match: (
                match.manifest.get("created_at", ""),
                match.manifest.get("artifact_id", ""),
            ),
        )
        public = public_manifest(selected.manifest)
        validation = selected.manifest["validation"]
        usable = artifact_is_usable(selected.manifest)
        content_fingerprint = selected.manifest["model"]["content_fingerprint"]
        verified_key = (
            selected.manifest["artifact_id"],
            selected.manifest["weights"]["sha256"],
            content_fingerprint,
        )
        verified = verified_key in self._verified_artifacts
        return {
            "status": "ready" if usable else "unvalidated",
            "message": (
                "Kalibrace je připravena."
                if usable
                else (
                    "Artefakt existuje, ale neprošel held-out přijímací "
                    "podmínkou."
                )
            ),
            "content_verification": "verified" if verified else "pending",
            "calibration": public,
        }

    def load(
        self,
        *,
        model_id: str,
        metadata_fingerprint: str,
        content_fingerprint: str,
        expected_artifact_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        status = self.status(
            model_id=model_id,
            metadata_fingerprint=metadata_fingerprint,
        )
        if status["status"] == "missing":
            raise TunedLensUnavailableError(status["message"])
        if status["status"] == "incompatible":
            raise CalibrationIncompatibleError(status["message"])
        if status["status"] != "ready":
            raise TunedLensUnavailableError(status["message"])
        public = status["calibration"]
        artifact_id = public["artifact_id"]
        if (
            expected_artifact_id is not None
            and expected_artifact_id != artifact_id
        ):
            raise CalibrationChangedError(
                "Aktivní tuned-lens artefakt se od odeslání požadavku změnil."
            )

        weight_digest = public["weights"]["sha256"]
        candidates = [
            item
            for item in self._manifests()
            if item.manifest["artifact_id"] == artifact_id
            and item.manifest["model"]["model_id"] == model_id
            and item.manifest["model"]["metadata_fingerprint"]
            == metadata_fingerprint
            and item.manifest["weights"]["sha256"] == weight_digest
        ]
        match = candidates[0] if len(candidates) == 1 else None
        if match is None:
            raise CalibrationChangedError(
                "Tuned-lens artefakt během požadavku zmizel nebo změnil identitu."
            )
        expected_content = match.manifest["model"]["content_fingerprint"]
        if content_fingerprint != expected_content:
            raise CalibrationIncompatibleError(
                "Úplný SHA-256 fingerprint použitých vah neodpovídá kalibraci."
            )
        try:
            validate_manifest(
                match.manifest,
                artifact_dir=match.directory,
                verify_weights=True,
            )
        except ArtifactValidationError as exc:
            self._invalid_artifacts[(artifact_id, weight_digest)] = str(exc)
            raise
        cache_key = (artifact_id, weight_digest)
        if self._loaded_key != cache_key or self._loaded_weights is None:
            try:
                import mlx.core as mx

                weight_path = (
                    match.directory / match.manifest["weights"]["file"]
                )
                loaded = mx.load(str(weight_path))
                weights = {
                    str(key): value for key, value in loaded.items()
                }
                validate_weight_shapes(match.manifest, weights)
            except ArtifactValidationError as exc:
                self._loaded_key = None
                self._loaded_weights = None
                self._invalid_artifacts[cache_key] = str(exc)
                raise
            except Exception as exc:
                self._loaded_key = None
                self._loaded_weights = None
                wrapped = ArtifactValidationError(
                    "Soubor vah tuned-lens artefaktu nelze bezpečně načíst."
                )
                self._invalid_artifacts[cache_key] = str(wrapped)
                raise wrapped from exc
            self._loaded_key = cache_key
            self._loaded_weights = weights
        self._verified_artifacts.add(
            (artifact_id, weight_digest, content_fingerprint)
        )
        return match.manifest, self._loaded_weights


def validate_weight_shapes(
    manifest: Mapping[str, Any],
    weights: Mapping[str, Any],
) -> None:
    reader = _require_mapping(manifest.get("reader"), "reader")
    hidden_size = int(reader["hidden_size"])
    kind = str(reader["translator_kind"])
    rank = int(reader.get("rank") or 0)
    expected_keys: dict[str, tuple[int, ...]] = {}
    for layer in reader["selected_layers"]:
        prefix = f"layer_{int(layer):02d}"
        expected_keys[f"{prefix}.bias"] = (hidden_size,)
        if kind == "low_rank_residual_v1":
            expected_keys[f"{prefix}.diag"] = (hidden_size,)
            expected_keys[f"{prefix}.down"] = (rank, hidden_size)
            expected_keys[f"{prefix}.up"] = (hidden_size, rank)
        elif kind == "full_affine_residual_v1":
            expected_keys[f"{prefix}.weight"] = (hidden_size, hidden_size)
    missing = sorted(set(expected_keys) - set(weights))
    if missing:
        raise ArtifactValidationError(
            "V artefaktu chybějí váhy: " + ", ".join(missing)
        )
    unexpected = sorted(set(weights) - set(expected_keys))
    if unexpected:
        raise ArtifactValidationError(
            "Artefakt obsahuje neočekávané váhy: " + ", ".join(unexpected)
        )
    for key, expected_shape in expected_keys.items():
        actual = tuple(int(value) for value in weights[key].shape)
        if actual != expected_shape:
            raise ArtifactValidationError(
                f"Váha {key} má tvar {actual}, očekává se {expected_shape}."
            )


def apply_translator(
    hidden: Any,
    *,
    layer: int,
    manifest: Mapping[str, Any],
    weights: Mapping[str, Any],
) -> Any:
    """Apply one residual translator to an MLX array."""

    import mlx.core as mx

    reader = manifest["reader"]
    prefix = f"layer_{layer:02d}"
    bias = weights[f"{prefix}.bias"].astype(hidden.dtype)
    kind = reader["translator_kind"]
    if kind == "bias_only_residual_v1":
        return hidden + bias
    if kind == "full_affine_residual_v1":
        weight = weights[f"{prefix}.weight"].astype(hidden.dtype)
        return hidden + mx.matmul(hidden, mx.transpose(weight)) + bias
    if kind == "low_rank_residual_v1":
        diag = weights[f"{prefix}.diag"].astype(hidden.dtype)
        down = weights[f"{prefix}.down"].astype(hidden.dtype)
        up = weights[f"{prefix}.up"].astype(hidden.dtype)
        low_rank = mx.matmul(
            mx.matmul(hidden, mx.transpose(down)),
            mx.transpose(up),
        )
        return hidden + hidden * diag + low_rank + bias
    raise ArtifactValidationError(f"Nepodporovaný translator_kind: {kind!r}.")


def _source_weight_files(
    model_id: str,
    sources: Iterable[tuple[str, Path]],
) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    for role, model_path in sources:
        model_path = model_path.resolve()
        if model_id == "hybrid" and role == "bf16_io":
            index_path = model_path / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
            shard_names = sorted(
                {
                    weight_map["model.embed_tokens.weight"],
                    weight_map["lm_head.weight"],
                }
            )
        else:
            index_path = model_path / "model.safetensors.index.json"
            if index_path.is_file():
                index = json.loads(index_path.read_text(encoding="utf-8"))
                shard_names = sorted(set(index.get("weight_map", {}).values()))
            else:
                shard_names = sorted(
                    path.name for path in model_path.glob("*.safetensors")
                )
        for shard_name in shard_names:
            shard_path = model_path / str(shard_name)
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"Chybí modelový shard {shard_path.name}."
                )
            selected.append((role, shard_path))
    return selected


def model_content_fingerprint(
    model_id: str,
    sources: Iterable[tuple[str, Path]],
    *,
    cache_path: Path | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Hash every model byte consumed by the selected runtime regime.

    The optional cache is accepted only when name, size and nanosecond mtime of
    every file still match.  The fingerprint itself is based on file contents,
    roles and relative names; absolute local paths never become identity.
    """

    source_list = [(role, path.resolve()) for role, path in sources]
    files: list[tuple[str, Path]] = []
    for role, model_path in source_list:
        for filename in MODEL_METADATA_FILES:
            path = model_path / filename
            if path.is_file():
                files.append((role, path))
    files.extend(_source_weight_files(model_id, source_list))
    deduplicated: dict[tuple[str, str], tuple[str, Path]] = {}
    for role, path in files:
        deduplicated[(role, path.name)] = (role, path)
    files = [deduplicated[key] for key in sorted(deduplicated)]
    stats = []
    for role, path in files:
        stat = path.stat()
        stats.append(
            {
                "role": role,
                "source": str(path.parent),
                "name": path.name,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
                "device": stat.st_dev,
                "inode": stat.st_ino,
            }
        )

    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("format") == "apertus.model-content-fingerprint"
                and cached.get("schema_version") == 2
                and cached.get("model_id") == model_id
                and cached.get("stats") == stats
                and isinstance(cached.get("fingerprint"), str)
                and isinstance(cached.get("files"), list)
            ):
                return cached["fingerprint"], cached["files"]
        except (OSError, json.JSONDecodeError):
            pass

    digest = hashlib.sha256()
    digest.update(b"apertus-model-content-v1\0")
    digest.update(model_id.encode("utf-8"))
    inventory = []
    for role, path in files:
        file_sha = sha256_file(path)
        digest.update(b"\0role\0")
        digest.update(role.encode("utf-8"))
        digest.update(b"\0file\0")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0sha256\0")
        digest.update(file_sha.encode("ascii"))
        inventory.append(
            {
                "role": role,
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha,
            }
        )
    fingerprint = digest.hexdigest()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "apertus.model-content-fingerprint",
            "schema_version": 2,
            "model_id": model_id,
            "stats": stats,
            "fingerprint": fingerprint,
            "files": inventory,
        }
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return fingerprint, inventory
