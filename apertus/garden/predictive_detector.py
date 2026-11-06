"""Runtime inference for the predictive-stylometry research preview."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import tempfile
import threading
import time
import uuid
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np

try:
    from .settings import cache_directory, data_directory
except ImportError:
    from settings import cache_directory, data_directory


GARDEN_DIR = Path(__file__).resolve().parent
PREDICTIVE_ROOT = data_directory() / "artifacts" / "predictive_stylometry"
DEFAULT_BUNDLE = (
    PREDICTIVE_ROOT
    / "expansion_v1"
    / "model_v2"
    / "inference_bundle_v2.json"
)
DEFAULT_MODEL_LOCK = (
    PREDICTIVE_ROOT
    / "expansion_v1"
    / "model_v2"
    / "model_lock.json"
)
DEFAULT_DEPLOYMENT = (
    PREDICTIVE_ROOT
    / "runtime"
    / "detector_v2_deployment.json"
)
DEFAULT_CHALLENGER_BUNDLE = (
    PREDICTIVE_ROOT
    / "pair_enrichment_v1"
    / "inference_bundle_pair_challenger.json"
)
DEFAULT_CHALLENGER_MANIFEST = (
    PREDICTIVE_ROOT
    / "runtime"
    / "detector_pair_challenger.json"
)
DEFAULT_MODEL_DIR = cache_directory() / "models" / "bf16"
DEFAULT_TIMING_LOG = (
    PREDICTIVE_ROOT
    / "runtime"
    / "app_analysis_timings.csv"
)
SCORER_SCRIPT = PREDICTIVE_ROOT / "scripts" / "score_pilot.py"
STYLOMETRY_SCRIPT = (
    PREDICTIVE_ROOT
    / "scripts"
    / "prepare_fusion_stylometry.py"
)
MAX_INPUT_CHARACTERS = 100_000
MAX_INPUT_UTF8_BYTES = 400_000
TIMING_FIELDS = (
    "analysis_id",
    "completed_at",
    "model_id",
    "model_generation",
    "character_count",
    "utf8_byte_count",
    "normalized_character_count",
    "full_token_count",
    "scored_token_count",
    "quality_status",
    "stylometry_ms",
    "apertus_ms",
    "model_prediction_ms",
    "challenger_model_ms",
    "total_ms",
)
LEGACY_TIMING_FIELDS_V2 = tuple(
    field
    for field in TIMING_FIELDS
    if field != "challenger_model_ms"
)

ScoreWindow = Callable[[str], dict[str, Any]]

# Reviewed legacy feature implementation. A package cannot authorize new Python
# code by declaring its own hash: changing these requires a code review/tests.
LEGACY_CODE_SHA256 = {
    "score_pilot.py": "f15fedec475691773832328c99b001992cf44e598c47010f878ebef1b097c3e3",
    "prepare_fusion_stylometry.py": "ebc6221e07a548d2a7e5853d6ccf8b3bd9af501287e9ead06c23327991372e06",
    "normalize_corpus_typography.py": "857ee5d1998819af2029faa36036e8590665c97698a12a044f6a288fbeb26797",
    "analyze_lexical_diversity.py": "f3dd4835aa13e3328a559d3b9ae99753095a72a923d18ba349b9772db0d29489",
    "analyze_sentence_punctuation.py": "2d3cad509f87855e2134a598b2ed9d6337e1ccee79342ec6b02dc2709bd56ad6",
    "analyze_syntactic_readability.py": "4d85b464b2348f530a0dc6e5bd29097f1b16a2f690f84d342b1a11aa74920909",
    "analyze_function_words.py": "3dfd44cf876c452c4c52a90a59634076884914525c8193b61befd898fdd4e2e2",
    "analyze_information_aesthetics.py": "83161e1edcdf267cb918fedc7cc51afc4c2a3808fe188bde436871077cffd27f",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nelze načíst výzkumný modul: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_legacy_runtime_code(root: Path, bundle: dict) -> dict[str, Path]:
    """Recognize a frozen legacy implementation without executing it."""
    result = {"scorer": root / "scripts/score_pilot.py", "stylometry": root / "scripts/prepare_fusion_stylometry.py"}
    extractors = bundle.get("stylometry", {}).get("legacy_extractors", {})
    if set(extractors) != {"normalization", "lexical", "sentence", "syntactic", "function_words", "information_aesthetics"}:
        raise RuntimeError("Unsupported frozen detector feature extractors.")
    for name, entry in extractors.items():
        path = Path(entry["path"])
        if entry.get("sha256") != LEGACY_CODE_SHA256.get(path.name):
            raise RuntimeError("Unreviewed detector Python code is not allowed.")
        result[name] = path
    for path in result.values():
        expected = LEGACY_CODE_SHA256.get(path.name)
        if expected is None or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Missing or changed reviewed detector code: {path.name}")
    return result


def sigmoid(value: float) -> float:
    if value >= 0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


class PredictiveDetector:
    """Load a frozen JSON model and score one arbitrary local text."""

    def __init__(
        self,
        *,
        bundle_path: Path = DEFAULT_BUNDLE,
        model_lock_path: Path = DEFAULT_MODEL_LOCK,
        deployment_path: Path = DEFAULT_DEPLOYMENT,
        challenger_bundle_path: Path = DEFAULT_CHALLENGER_BUNDLE,
        challenger_manifest_path: Path = DEFAULT_CHALLENGER_MANIFEST,
        model_dir: Path = DEFAULT_MODEL_DIR,
        timing_log_path: Path | None = DEFAULT_TIMING_LOG,
        artifact_root: Path | None = None,
    ) -> None:
        self.artifact_root = artifact_root.expanduser().resolve() if artifact_root is not None else None
        if self.artifact_root is not None:
            bundle_path = self.artifact_root / "expansion_v1/model_v2/inference_bundle_v2.json"
            model_lock_path = self.artifact_root / "expansion_v1/model_v2/model_lock.json"
            deployment_path = self.artifact_root / "runtime/detector_v2_deployment.json"
            challenger_bundle_path = self.artifact_root / "pair_enrichment_v1/inference_bundle_pair_challenger.json"
            challenger_manifest_path = self.artifact_root / "runtime/detector_pair_challenger.json"
        self.bundle_path = bundle_path.resolve()
        self.model_lock_path = model_lock_path.resolve()
        self.deployment_path = deployment_path.resolve()
        self.challenger_bundle_path = challenger_bundle_path.resolve()
        self.challenger_manifest_path = (
            challenger_manifest_path.resolve()
        )
        self.model_dir = model_dir.resolve()
        self.timing_log_path = (
            timing_log_path.resolve()
            if timing_log_path is not None
            else None
        )
        self._load_lock = threading.Lock()
        self._timing_log_lock = threading.Lock()
        self._loaded = False
        self.bundle: dict[str, Any] = {}
        self.bundle_sha256 = ""
        self.model_lock: dict[str, Any] = {}
        self.model_lock_sha256 = ""
        self.deployment: dict[str, Any] = {}
        self.deployment_sha256 = ""
        self.challenger_bundle: dict[str, Any] = {}
        self.challenger_bundle_sha256 = ""
        self.challenger_manifest: dict[str, Any] = {}
        self.challenger_manifest_sha256 = ""
        self.tokenizer: Any = None
        self.scorer: ModuleType | None = None
        self.stylometry: ModuleType | None = None
        self.stylometry_modules: dict[str, ModuleType] = {}

    def _verify_file(self, path: Path, expected_sha256: str) -> None:
        if not path.is_file():
            raise RuntimeError(f"Chybí zmrazený soubor: {path}")
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Změnil se zmrazený soubor {path.name}: "
                f"{actual} != {expected_sha256}"
            )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            if not self.deployment_path.is_file():
                raise RuntimeError("Chybí deployment manifest detektoru v2.")
            deployment_bytes = self.deployment_path.read_bytes()
            deployment = json.loads(deployment_bytes)
            if (
                deployment.get("schema_version") != 1
                or deployment.get("status") != "deployed_frozen_v2"
                or not all(deployment.get("checks", {}).values())
            ):
                raise RuntimeError("Deployment manifest v2 není platný.")

            deployment_bundle = deployment["bundle"]
            if (
                Path(deployment_bundle["path"]).resolve()
                != self.bundle_path
            ):
                raise RuntimeError(
                    "Deployment ukazuje na jiný inference bundle."
                )
            deployment_lock = deployment["model_lock"]
            if (
                Path(deployment_lock["path"]).resolve()
                != self.model_lock_path
            ):
                raise RuntimeError("Deployment ukazuje na jiný model lock.")
            for artifact in deployment["runtime_artifacts"].values():
                self._verify_file(
                    Path(artifact["path"]),
                    artifact["sha256"],
                )

            if not self.model_lock_path.is_file():
                raise RuntimeError("Chybí zmrazený model lock v2.")
            model_lock_bytes = self.model_lock_path.read_bytes()
            model_lock = json.loads(model_lock_bytes)
            if (
                model_lock.get("status")
                != "frozen_before_gemini_reopen"
                or not all(model_lock.get("checks", {}).values())
            ):
                raise RuntimeError("Zmrazený model lock v2 není platný.")
            if (
                hashlib.sha256(model_lock_bytes).hexdigest()
                != deployment_lock["sha256"]
            ):
                raise RuntimeError("Model lock se po deploymentu změnil.")
            for collection in ("inputs", "outputs"):
                for artifact in model_lock[collection].values():
                    self._verify_file(
                        Path(artifact["path"]),
                        artifact["sha256"],
                    )

            if not self.bundle_path.is_file():
                raise RuntimeError(
                    "Chybí inference bundle prediktivní stylometrie."
                )
            bundle_bytes = self.bundle_path.read_bytes()
            bundle = json.loads(bundle_bytes)
            if (
                bundle.get("schema_version") != 2
                or bundle.get("status")
                != "frozen_before_gemini_reopen"
            ):
                raise RuntimeError("Inference bundle má neznámou verzi.")
            if (
                hashlib.sha256(bundle_bytes).hexdigest()
                != deployment_bundle["sha256"]
            ):
                raise RuntimeError("Inference bundle se po deploymentu změnil.")
            decision_display = deployment.get("decision_display", {})
            expected_strict_threshold = float(
                bundle["scopes"]["quality_unflagged"]["models"]["fusion"][
                    "thresholds"
                ]["strict"]["threshold"]
            )
            display_strict_threshold = float(
                decision_display.get("strict_threshold", math.nan)
            )
            borderline_upper_score = float(
                decision_display.get("borderline_upper_score", math.nan)
            )
            if (
                decision_display.get("status")
                != "presentation_band_derived_from_frozen_calibration"
                or not math.isclose(
                    display_strict_threshold,
                    expected_strict_threshold,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not (
                    display_strict_threshold
                    < borderline_upper_score
                    <= 1.0
                )
            ):
                raise RuntimeError(
                    "Prezentační pásmo rozhodovací rezervy není platné."
                )

            if not self.challenger_manifest_path.is_file():
                raise RuntimeError(
                    "Chybí integrační manifest párového challengera."
                )
            challenger_manifest_bytes = (
                self.challenger_manifest_path.read_bytes()
            )
            challenger_manifest = json.loads(challenger_manifest_bytes)
            deployment_challenger = deployment.get(
                "parallel_challenger",
                {},
            )
            if (
                Path(deployment_challenger.get("path", "")).resolve()
                != self.challenger_manifest_path
                or deployment_challenger.get("sha256")
                != hashlib.sha256(
                    challenger_manifest_bytes
                ).hexdigest()
                or deployment_challenger.get("decision_authority")
                is not False
                or deployment_challenger.get("status") != "research_only"
                or deployment_challenger.get("role")
                != "paired_stylometry_v5"
            ):
                raise RuntimeError(
                    "Deployment neukazuje na uzamčený manifest "
                    "párového challengera."
                )
            required_challenger_checks = (
                "bundle_hash_verified",
                "pair_protocol_hash_verified",
                "runtime_protocol_hash_verified",
                "feature_schema_matches_v2_stylometry",
                "reader_signature_matches_v2",
                "candidate_is_stylometry_only",
                "paired_loss_model_verified",
                "decision_authority_disabled",
            )
            challenger_checks = challenger_manifest.get("checks", {})
            if (
                challenger_manifest.get("schema_version") != 1
                or challenger_manifest.get("status")
                != "parallel_research_challenger"
                or challenger_manifest.get("role")
                != "research_only_paired_stylometry_v5"
                or challenger_manifest.get("decision_authority") is not False
                or challenger_manifest.get(
                    "model_or_primary_threshold_changed"
                )
                is not False
                or any(
                    challenger_checks.get(check) is not True
                    for check in required_challenger_checks
                )
                or challenger_checks.get(
                    "independent_validation_completed"
                )
                is not False
                or challenger_checks.get("promotion_gate_passed")
                is not False
            ):
                raise RuntimeError(
                    "Integrační manifest párového challengera není platný."
                )
            challenger_manifest_bundle = challenger_manifest["bundle"]
            if (
                Path(challenger_manifest_bundle["path"]).resolve()
                != self.challenger_bundle_path
            ):
                raise RuntimeError(
                    "Manifest challengera ukazuje na jiný bundle."
                )
            self._verify_file(
                self.challenger_bundle_path,
                challenger_manifest_bundle["sha256"],
            )
            for artifact_name in (
                "protocol",
                "runtime_protocol",
                "selection",
                "decision",
                "source_model",
                "report",
                "posthoc_interpretation",
                "builder",
            ):
                artifact = challenger_manifest[artifact_name]
                self._verify_file(
                    Path(artifact["path"]),
                    artifact["sha256"],
                )

            challenger_bundle_bytes = (
                self.challenger_bundle_path.read_bytes()
            )
            challenger_bundle = json.loads(challenger_bundle_bytes)
            challenger_model = challenger_bundle.get("model", {})
            challenger_protocol = challenger_bundle.get("protocol", {})
            challenger_runtime_protocol = challenger_bundle.get(
                "runtime_protocol",
                {},
            )
            challenger_columns = challenger_model.get(
                "feature_columns",
                [],
            )
            if (
                challenger_bundle.get("schema_version") != 1
                or challenger_bundle.get("status")
                != "research_only_paired_challenger_posthoc"
                or challenger_bundle.get("candidate_name")
                != "paired_stylometry"
                or challenger_bundle.get("display_name_cs")
                != "Párová stylometrie v5"
                or challenger_bundle.get(
                    "next_required_action"
                )
                != (
                    "Freeze and run a new independent multi-model, "
                    "multi-publisher holdout before any "
                    "decision-authority change."
                )
                or challenger_bundle.get("reader") != bundle["reader"]
                or challenger_bundle.get("quality_rule")
                != bundle["quality_rule"]
                or challenger_columns
                != bundle["stylometry"]["feature_columns"]
                or len(challenger_columns) != 120
                or any(
                    not column.startswith("stylo__")
                    for column in challenger_columns
                )
                or Path(challenger_protocol.get("path", "")).resolve()
                != Path(
                    challenger_manifest["protocol"]["path"]
                ).resolve()
                or challenger_protocol.get("sha256")
                != challenger_manifest["protocol"]["sha256"]
                or challenger_protocol.get("status")
                != challenger_manifest["protocol"]["status"]
                or Path(
                    challenger_runtime_protocol.get("path", "")
                ).resolve()
                != Path(
                    challenger_manifest["runtime_protocol"]["path"]
                ).resolve()
                or challenger_runtime_protocol.get("sha256")
                != challenger_manifest["runtime_protocol"]["sha256"]
                or challenger_bundle.get(
                    "promotion_decision",
                    {},
                ).get("passed")
                is not False
                or challenger_bundle.get(
                    "promotion_decision",
                    {},
                ).get("application_integration_authorised")
                is not False
                or float(challenger_model.get("pair_strength", 0.0))
                <= 0.0
            ):
                raise RuntimeError(
                    "Výzkumný bundle párového challengera není "
                    "kompatibilní."
                )
            strict_threshold = float(
                challenger_model["thresholds"]["strict"]["threshold"]
            )
            screening_threshold = float(
                challenger_model["thresholds"]["screening"]["threshold"]
            )
            for threshold_name in ("strict", "screening"):
                threshold = float(
                    challenger_model["thresholds"][threshold_name][
                        "threshold"
                    ]
                )
                if not 0.0 < threshold < 1.0:
                    raise RuntimeError(
                        "Párový challenger má neplatný výzkumný práh."
                    )
            if screening_threshold > strict_threshold:
                raise RuntimeError(
                    "Screeningový práh challengera převyšuje přísný."
                )

            tokenizer_path = self.model_dir / "tokenizer.json"
            self._verify_file(
                tokenizer_path,
                bundle["reader"]["tokenizer_json_sha256"],
            )
            for extractor in bundle["stylometry"][
                "legacy_extractors"
            ].values():
                self._verify_file(
                    Path(extractor["path"]),
                    extractor["sha256"],
                )

            code = verify_legacy_runtime_code(self.artifact_root, bundle) if self.artifact_root is not None else None
            scorer = load_module(
                code["scorer"] if code else SCORER_SCRIPT,
                "predictive_detector_scorer",
            )
            stylometry = load_module(
                code["stylometry"] if code else STYLOMETRY_SCRIPT,
                "predictive_detector_stylometry",
            )
            stylometry_modules = {
                name: load_module(
                    code[name] if code else Path(metadata["path"]),
                    f"predictive_detector_{name}",
                )
                for name, metadata in bundle["stylometry"][
                    "legacy_extractors"
                ].items()
            }
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Lokální prostředí nemá knihovnu transformers."
                ) from exc
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                local_files_only=True,
                trust_remote_code=False,
            )

            self.bundle = bundle
            self.bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
            self.model_lock = model_lock
            self.model_lock_sha256 = hashlib.sha256(
                model_lock_bytes
            ).hexdigest()
            self.deployment = deployment
            self.deployment_sha256 = hashlib.sha256(
                deployment_bytes
            ).hexdigest()
            self.challenger_bundle = challenger_bundle
            self.challenger_bundle_sha256 = hashlib.sha256(
                challenger_bundle_bytes
            ).hexdigest()
            self.challenger_manifest = challenger_manifest
            self.challenger_manifest_sha256 = hashlib.sha256(
                challenger_manifest_bytes
            ).hexdigest()
            self.scorer = scorer
            self.stylometry = stylometry
            self.stylometry_modules = stylometry_modules
        self.tokenizer = tokenizer
        self._loaded = True

    def _append_timing(self, row: dict[str, Any]) -> None:
        if self.timing_log_path is None:
            return
        with self._timing_log_lock:
            self.timing_log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            exists = (
                self.timing_log_path.is_file()
                and self.timing_log_path.stat().st_size > 0
            )
            if exists:
                with self.timing_log_path.open(
                    encoding="utf-8",
                    newline="",
                ) as handle:
                    reader = csv.DictReader(handle)
                    previous_fields = tuple(reader.fieldnames or ())
                    previous_rows = list(reader)
                if previous_fields != TIMING_FIELDS:
                    if previous_fields != LEGACY_TIMING_FIELDS_V2:
                        raise OSError(
                            "Časový log má neznámé schéma; nebyl změněn."
                        )
                    temporary_path: Path | None = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="w",
                            encoding="utf-8",
                            newline="",
                            dir=self.timing_log_path.parent,
                            prefix=(
                                f".{self.timing_log_path.name}."
                            ),
                            suffix=".tmp",
                            delete=False,
                        ) as temporary_handle:
                            temporary_path = Path(temporary_handle.name)
                            writer = csv.DictWriter(
                                temporary_handle,
                                fieldnames=list(TIMING_FIELDS),
                                extrasaction="ignore",
                            )
                            writer.writeheader()
                            writer.writerows(previous_rows)
                            temporary_handle.flush()
                            os.fsync(temporary_handle.fileno())
                        os.replace(
                            temporary_path,
                            self.timing_log_path,
                        )
                    finally:
                        if (
                            temporary_path is not None
                            and temporary_path.exists()
                        ):
                            temporary_path.unlink()
            with self.timing_log_path.open(
                "a",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(TIMING_FIELDS),
                    extrasaction="ignore",
                )
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())

    def _quality_flags(
        self,
        diagnostics: dict[str, Any],
    ) -> list[dict[str, str]]:
        rule = self.bundle["quality_rule"]
        flags: list[dict[str, str]] = []
        if (
            diagnostics["input_non_latin_letter_ratio"]
            > float(rule["non_latin_ratio_threshold"])
            or diagnostics["input_unicode_script_count"]
            >= int(rule["script_count_threshold"])
        ):
            flags.append(
                {
                    "code": "mixed_or_non_latin_script",
                    "message_cs": (
                        "Text výrazně míchá písma nebo používá jiné než "
                        "latinské písmo."
                    ),
                }
            )
        if (
            diagnostics["input_english_function_word_rate"]
            >= float(rule["english_function_word_rate_threshold"])
            and diagnostics["input_english_function_word_rate"]
            > diagnostics["input_czech_function_word_rate"]
        ):
            flags.append(
                {
                    "code": "english_function_words_dominate",
                    "message_cs": (
                        "Text nevypadá jako čeština; anglická funkční slova "
                        "převažují."
                    ),
                }
            )
        if diagnostics["input_replacement_character_count"] > int(
            rule["replacement_character_threshold"]
        ):
            flags.append(
                {
                    "code": "unicode_replacement_character",
                    "message_cs": (
                        "Text obsahuje náhradní znak Unicode a může být "
                        "poškozený."
                    ),
                }
            )
        return flags

    def _extract_stylometry(
        self,
        text: str,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        assert self.stylometry is not None
        normalization = self.stylometry_modules["normalization"]
        normalized = normalization.normalize_text(text)
        sampling = self.bundle["stylometry"]["sampling"]
        span_characters = int(sampling["span_characters"])
        span_count = int(sampling["span_count"])
        minimum_characters = span_characters * span_count
        if len(normalized) < minimum_characters:
            raise ValueError(
                f"Pro společné skóre je potřeba alespoň "
                f"{minimum_characters} normalizovaných znaků; text má "
                f"{len(normalized)}."
            )
        sampled, starts = self.stylometry.sampled_text(
            normalized,
            span_characters=span_characters,
            span_count=span_count,
        )
        with tempfile.TemporaryDirectory(
            prefix="apertus-predictive-detector-"
        ) as temporary:
            # The frozen legacy extractors parse ``<author>_block<N>`` from
            # filenames even though author/block are excluded from the model.
            sampled_path = Path(temporary) / "anonymous_block1.txt"
            sampled_path.write_text(sampled, encoding="utf-8")
            features = self.stylometry.extract_feature_row(
                sampled_path,
                self.stylometry_modules,
            )
        expected = self.bundle["stylometry"]["feature_columns"]
        if set(features) != set(expected):
            missing = sorted(set(expected) - set(features))
            unexpected = sorted(set(features) - set(expected))
            raise RuntimeError(
                "Stylometrické schéma se změnilo. "
                f"Chybí={missing[:5]}, navíc={unexpected[:5]}"
            )
        return features, {
            "raw_character_count": len(text),
            "normalized_character_count": len(normalized),
            "sampled_character_count": len(sampled),
            "span_characters": span_characters,
            "span_count": span_count,
            "span_starts": starts,
        }

    def _score_apertus(
        self,
        text: str,
        score_window: ScoreWindow,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        assert self.scorer is not None
        configuration = self.bundle["reader"]["configuration"]
        context_tokens = int(configuration["context_tokens"])
        target_tokens = int(configuration["target_tokens"])
        burn_in_tokens = int(configuration["burn_in_tokens"])
        window_count = int(configuration["windows_per_document"])
        minimum_target_tokens = int(
            configuration["minimum_target_tokens"]
        )
        document_ids = list(
            self.tokenizer.encode(text, add_special_tokens=False)
        )
        starts = self.scorer.evenly_spaced_starts(
            len(document_ids),
            burn_in_tokens=burn_in_tokens,
            target_tokens=target_tokens,
            window_count=window_count,
        )
        if len(starts) != window_count:
            raise ValueError(
                f"Text má jen {len(document_ids)} Apertus tokenů a "
                f"nevytvoří {window_count} odlišná měřicí okna."
            )

        token_rows: list[dict[str, Any]] = []
        selected_character_count = 0
        selected_non_whitespace_count = 0
        windows: list[dict[str, Any]] = []
        scoring_seconds = 0.0
        for window_index, (position_label, target_start) in enumerate(
            starts,
            start=1,
        ):
            context_start = max(0, target_start - context_tokens)
            target_end = min(
                len(document_ids),
                target_start + target_tokens,
            )
            (
                window_text,
                window_ids,
                target_end,
                context_start,
            ) = self.scorer.decode_window(
                self.tokenizer,
                document_ids,
                context_start=context_start,
                target_start=target_start,
                target_end=target_end,
            )
            target_offset = target_start - context_start
            target_count = target_end - target_start
            if target_count < minimum_target_tokens:
                raise ValueError(
                    f"Okno {window_index} má jen {target_count} "
                    "hodnocených tokenů."
                )
            started = time.perf_counter()
            result = score_window(window_text)
            scoring_seconds += time.perf_counter() - started
            if result.get("analysis_kind") != "text_surprisal":
                raise RuntimeError("Apertus vrátil neočekávaný typ analýzy.")
            if result.get("token_ids") != window_ids:
                raise RuntimeError(
                    "Tokenizace inference bundle se liší od serveru."
                )
            target_end_offset = target_offset + target_count
            target_payload = result["tokens"][
                target_offset:target_end_offset
            ]
            if len(target_payload) != target_count:
                raise RuntimeError("Apertus vrátil neúplné měřicí okno.")
            target_text = self.tokenizer.decode(
                document_ids[target_start:target_end],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            selected_character_count += len(target_text)
            selected_non_whitespace_count += sum(
                not character.isspace()
                for character in target_text
            )
            token_rows.extend(target_payload)
            values = [
                float(token["surprisal_bits"])
                for token in target_payload
            ]
            windows.append(
                {
                    "index": window_index,
                    "position": position_label,
                    "context_start": context_start,
                    "target_start": target_start,
                    "target_end": target_end,
                    "target_token_count": target_count,
                    "mean_surprisal_bits": (
                        math.fsum(values) / len(values)
                    ),
                    "server_forward_ms": float(
                        result["timing_ms"]["forward"]
                    ),
                    "server_total_ms": float(
                        result["timing_ms"]["total"]
                    ),
                }
            )

        surprises = [
            float(token["surprisal_bits"])
            for token in token_rows
        ]
        entropies = [
            float(token["entropy_bits"])
            for token in token_rows
        ]
        if not surprises or selected_non_whitespace_count <= 0:
            raise RuntimeError("Apertus nevytvořil použitelné rysy.")
        q1 = self.scorer.quantile(surprises, 0.25)
        q3 = self.scorer.quantile(surprises, 0.75)
        p95 = self.scorer.quantile(surprises, 0.95)
        top_share = self.scorer.concentration_share(surprises, 0.05)
        if None in (q1, q3, p95, top_share):
            raise RuntimeError("Apertus rysy nejsou úplné.")
        total_surprisal = math.fsum(surprises)
        features = {
            "surprisal_bits_per_non_whitespace_character": (
                total_surprisal / selected_non_whitespace_count
            ),
            "token_surprisal_iqr": float(q3) - float(q1),
            "token_surprisal_p95": float(p95),
            "top_5pct_surprisal_share": float(top_share),
            "mean_surprisal_minus_entropy_bits": (
                math.fsum(
                    surprise - entropy
                    for surprise, entropy in zip(
                        surprises,
                        entropies,
                        strict=True,
                    )
                )
                / len(surprises)
            ),
        }
        return features, {
            "full_token_count": len(document_ids),
            "scored_token_count": len(token_rows),
            "scored_character_count": selected_character_count,
            "scored_non_whitespace_character_count": (
                selected_non_whitespace_count
            ),
            "windows": windows,
            "scoring_seconds": scoring_seconds,
        }

    @staticmethod
    def _predict_model(
        model: dict[str, Any],
        feature_values: dict[str, float],
    ) -> dict[str, Any]:
        columns = model["feature_columns"]
        raw = np.asarray(
            [feature_values.get(column, math.nan) for column in columns],
            dtype=np.float64,
        )
        statistics = np.asarray(
            model["imputer_statistics"],
            dtype=np.float64,
        )
        missing = ~np.isfinite(raw)
        imputed = np.where(missing, statistics, raw)
        scaled = (
            imputed
            - np.asarray(model["scaler_mean"], dtype=np.float64)
        ) / np.asarray(model["scaler_scale"], dtype=np.float64)
        coefficients = np.asarray(
            model["logistic_coefficients"],
            dtype=np.float64,
        )
        contributions = scaled * coefficients
        intercept = float(model["logistic_intercept"])
        logit = intercept + float(np.sum(contributions))
        score = sigmoid(logit)
        strict = model["thresholds"]["strict"]
        screening = model["thresholds"]["screening"]
        strict_threshold = float(strict["threshold"])
        screening_threshold = float(screening["threshold"])
        block_contributions = {
            "intercept": intercept,
            "stylometry": float(
                sum(
                    contribution
                    for column, contribution in zip(
                        columns,
                        contributions,
                        strict=True,
                    )
                    if column.startswith("stylo__")
                )
            ),
            "apertus": float(
                sum(
                    contribution
                    for column, contribution in zip(
                        columns,
                        contributions,
                        strict=True,
                    )
                    if not column.startswith("stylo__")
                )
            ),
        }
        top_contributions = sorted(
            (
                {
                    "feature": column,
                    "logit_contribution": float(contribution),
                }
                for column, contribution in zip(
                    columns,
                    contributions,
                    strict=True,
                )
            ),
            key=lambda row: abs(row["logit_contribution"]),
            reverse=True,
        )[:8]
        calibration = model["calibration"]
        return {
            "score": score,
            "score_percent": score * 100.0,
            "logit": logit,
            # Compatibility alias retained for the current page contract.
            "locked_low_fpr_threshold": strict_threshold,
            "strict_threshold": strict_threshold,
            "screening_threshold": screening_threshold,
            "strict_ai_signal": score >= strict_threshold,
            "screening_ai_signal": score >= screening_threshold,
            "above_balanced_midpoint": score >= 0.5,
            "imputed_feature_count": int(np.count_nonzero(missing)),
            "block_logit_contributions": block_contributions,
            "top_feature_contributions": top_contributions,
            "calibration": {
                "pair_count": calibration["pair_count"],
                "roc_auc": calibration["roc_auc"],
                "pr_auc": calibration["pr_auc"],
                "pair_ranking_accuracy": calibration[
                    "pair_ranking_accuracy"
                ],
                "strict": {
                    "fpr": calibration["strict"]["fpr"],
                    "recall": calibration["strict"]["ai_recall"],
                    "precision": calibration["strict"]["precision"],
                },
                "screening": {
                    "fpr": calibration["screening"]["fpr"],
                    "recall": calibration["screening"]["ai_recall"],
                    "precision": calibration["screening"]["precision"],
                },
                "interpretation": calibration["interpretation"],
            },
        }

    @staticmethod
    def _predict_challenger_model(
        model: dict[str, Any],
        feature_values: dict[str, float],
    ) -> dict[str, Any]:
        columns = model["feature_columns"]
        raw = np.asarray(
            [feature_values.get(column, math.nan) for column in columns],
            dtype=np.float64,
        )
        statistics = np.asarray(
            model["imputer_statistics"],
            dtype=np.float64,
        )
        means = np.asarray(model["scaler_mean"], dtype=np.float64)
        scales = np.asarray(model["scaler_scale"], dtype=np.float64)
        coefficients = np.asarray(
            model["logistic_coefficients"],
            dtype=np.float64,
        )
        if any(
            values.shape != raw.shape
            for values in (statistics, means, scales, coefficients)
        ):
            raise RuntimeError(
                "Párový challenger má nekompatibilní rozměry modelu."
            )
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
            raise RuntimeError(
                "Párový challenger má neplatné škálovací konstanty."
            )
        missing = ~np.isfinite(raw)
        imputed = np.where(missing, statistics, raw)
        scaled = (imputed - means) / scales
        contributions = scaled * coefficients
        intercept = float(model["logistic_intercept"])
        logit = intercept + float(np.sum(contributions))
        score = sigmoid(logit)
        strict_threshold = float(
            model["thresholds"]["strict"]["threshold"]
        )
        screening_threshold = float(
            model["thresholds"]["screening"]["threshold"]
        )
        top_contributions = sorted(
            (
                {
                    "feature": column,
                    "logit_contribution": float(contribution),
                }
                for column, contribution in zip(
                    columns,
                    contributions,
                    strict=True,
                )
            ),
            key=lambda row: abs(row["logit_contribution"]),
            reverse=True,
        )[:8]
        return {
            "score": score,
            "score_percent": score * 100.0,
            "logit": logit,
            "strict_threshold": strict_threshold,
            "screening_threshold": screening_threshold,
            "strict_ai_signal": score >= strict_threshold,
            "screening_ai_signal": score >= screening_threshold,
            "imputed_feature_count": int(np.count_nonzero(missing)),
            "feature_count": len(columns),
            "top_feature_contributions": top_contributions,
        }

    @staticmethod
    def _challenger_comparison(
        quality_status: str,
        apertus: dict[str, Any],
        challenger: dict[str, Any],
    ) -> dict[str, Any]:
        def band(model: dict[str, Any]) -> str:
            if model["strict_ai_signal"]:
                return "strict"
            if model["screening_ai_signal"]:
                return "screening"
            return "low"

        apertus_band = band(apertus)
        challenger_band = band(challenger)
        delta_points = (
            float(challenger["score"]) - float(apertus["score"])
        ) * 100.0
        if quality_status == "abstain":
            status = "abstain"
            label_cs = "Bez srovnání kvůli quality gate"
            interpretation_cs = (
                "Obě skóre zůstávají diagnostická; vstup neprošel "
                "kontrolou kvality."
            )
        elif apertus_band == challenger_band:
            status = f"agreement_{apertus_band}"
            labels = {
                "strict": "Shoda: oba přísný signál",
                "screening": "Shoda: oba v šedém pásmu",
                "low": "Shoda: oba pod screeningem",
            }
            label_cs = labels[apertus_band]
            interpretation_cs = (
                "Apertus a párová stylometrie v5 leží ve stejném "
                "pásmu svých vlastních prahů."
            )
        else:
            rank = {"low": 0, "screening": 1, "strict": 2}
            challenger_higher = (
                rank[challenger_band] > rank[apertus_band]
            )
            status = (
                "challenger_higher"
                if challenger_higher
                else "apertus_higher"
            )
            label_cs = (
                "Neshoda: výš je párová stylometrie"
                if challenger_higher
                else "Neshoda: výš je Apertus"
            )
            interpretation_cs = (
                "Modely leží v různých pásmech. Jde o výzkumný záznam "
                "neshody; hlavní verdikt se řídí pouze Apertusem."
            )
        return {
            "status": status,
            "label_cs": label_cs,
            "interpretation_cs": interpretation_cs,
            "apertus_band": apertus_band,
            "challenger_band": challenger_band,
            "score_delta_points": delta_points,
            "changes_primary_verdict": False,
        }

    def _scope_scores(
        self,
        scope: str,
        feature_values: dict[str, float],
    ) -> dict[str, Any]:
        scope_bundle = self.bundle["scopes"][scope]
        training_pair_count = int(scope_bundle["training_pair_count"])
        return {
            "scope": scope,
            "scope_label_cs": (
                f"všech {training_pair_count} trénovacích párů"
                if scope == "all"
                else (
                    f"{training_pair_count} quality-gated "
                    "trénovacích párů"
                )
            ),
            "training_pair_count": training_pair_count,
            "models": {
                name: self._predict_model(model, feature_values)
                for name, model in scope_bundle["models"].items()
            },
        }

    def _predictive_profile(
        self,
        features: dict[str, float],
    ) -> dict[str, Any]:
        human_reference_count = len(
            self.bundle["human_reference"][
                self.bundle["apertus_features"][0]
            ]["human_values"]
        )
        rows: list[dict[str, Any]] = []
        for feature in self.bundle["apertus_features"]:
            reference = self.bundle["human_reference"][feature]
            values = list(reference["human_values"])
            value = float(features[feature])
            left = bisect_left(values, value)
            right = bisect_right(values, value)
            percentile = (
                (left + 0.5 * (right - left)) / len(values) * 100.0
            )
            ai_direction_percentile = (
                100.0 - percentile
                if reference["ai_direction"] == "lower"
                else percentile
            )
            rows.append(
                {
                    "feature": feature,
                    "label_cs": reference["label_cs"],
                    "value": value,
                    "human_percentile": percentile,
                    "ai_direction": reference["ai_direction"],
                    "ai_direction_percentile": (
                        ai_direction_percentile
                    ),
                    "human_q1": reference["human_q1"],
                    "human_median": reference["human_median"],
                    "human_q3": reference["human_q3"],
                }
            )
        return {
            "ai_direction_index": float(
                np.mean(
                    [
                        row["ai_direction_percentile"]
                        for row in rows
                    ]
                )
            ),
            "human_reference_count": human_reference_count,
            "index_semantics_cs": (
                f"Průměr směrových percentilů vůči "
                f"{human_reference_count} lidským tréninkovým "
                "textům; nejde o pravděpodobnost ani klasifikátor."
            ),
            "features": rows,
        }

    def analyze(
        self,
        text: str,
        *,
        score_window: ScoreWindow,
        model_id: str,
        model_generation: int,
        server_instance_id: str,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        self._ensure_loaded()
        if model_id != self.bundle["reader"]["required_model_id"]:
            raise ValueError(
                "Prediktivní detektor je zmrazen pro model Hybrid. "
                "Přepni globální model na Hybrid."
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text musí být neprázdný řetězec.")
        if len(text) > MAX_INPUT_CHARACTERS:
            raise ValueError(
                f"Text může mít nejvýše {MAX_INPUT_CHARACTERS} znaků."
            )
        utf8_bytes = len(text.encode("utf-8"))
        if utf8_bytes > MAX_INPUT_UTF8_BYTES:
            raise ValueError(
                f"Text může mít nejvýše {MAX_INPUT_UTF8_BYTES} UTF-8 bajtů."
            )

        diagnostics = self.scorer.input_text_diagnostics(text)
        quality_flags = self._quality_flags(diagnostics)

        stylometry_started = time.perf_counter()
        style_features, style_sampling = self._extract_stylometry(text)
        stylometry_ms = (
            time.perf_counter() - stylometry_started
        ) * 1_000

        apertus_started = time.perf_counter()
        apertus_features, apertus_sampling = self._score_apertus(
            text,
            score_window,
        )
        apertus_ms = (
            time.perf_counter() - apertus_started
        ) * 1_000

        prediction_started = time.perf_counter()
        feature_values = {**style_features, **apertus_features}
        all_scope = self._scope_scores("all", feature_values)
        unflagged_scope = (
            self._scope_scores("quality_unflagged", feature_values)
            if not quality_flags
            else None
        )
        primary_scope = (
            "quality_unflagged"
            if unflagged_scope is not None
            else "all"
        )
        primary_scores = unflagged_scope or all_scope
        primary_apertus = primary_scores["models"]["apertus"]
        verdict = (
            "abstain"
            if quality_flags
            else "ai_signal"
            if primary_apertus["strict_ai_signal"]
            else "screening_signal"
            if primary_apertus["screening_ai_signal"]
            else "ai_signal_not_detected"
        )
        profile = self._predictive_profile(apertus_features)
        prediction_ms = (
            time.perf_counter() - prediction_started
        ) * 1_000

        challenger_started = time.perf_counter()
        challenger_model = self._predict_challenger_model(
            self.challenger_bundle["model"],
            style_features,
        )
        challenger_model_ms = (
            time.perf_counter() - challenger_started
        ) * 1_000
        quality_status = "abstain" if quality_flags else "pass"
        challenger_comparison = self._challenger_comparison(
            quality_status,
            primary_apertus,
            challenger_model,
        )

        warnings = list(self.bundle["limitations_cs"])
        if quality_flags:
            warnings.insert(
                0,
                "Quality gate se zdržel autorství; zobrazená skóre jsou "
                "pouze diagnostická.",
            )
        timing = {
            "stylometry": stylometry_ms,
            "apertus": apertus_ms,
            "model_prediction": prediction_ms,
            "challenger_model": challenger_model_ms,
            "total": (
                time.perf_counter() - total_started
            )
            * 1_000,
        }
        analysis_id = uuid.uuid4().hex
        timing_stored = False
        timing_log_error: str | None = None
        if self.timing_log_path is not None:
            try:
                self._append_timing(
                    {
                        "analysis_id": analysis_id,
                        "completed_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "model_id": model_id,
                        "model_generation": model_generation,
                        "character_count": len(text),
                        "utf8_byte_count": utf8_bytes,
                        "normalized_character_count": style_sampling[
                            "normalized_character_count"
                        ],
                        "full_token_count": apertus_sampling[
                            "full_token_count"
                        ],
                        "scored_token_count": apertus_sampling[
                            "scored_token_count"
                        ],
                        "quality_status": (
                            "abstain" if quality_flags else "pass"
                        ),
                        "stylometry_ms": stylometry_ms,
                        "apertus_ms": apertus_ms,
                        "model_prediction_ms": prediction_ms,
                        "challenger_model_ms": challenger_model_ms,
                        "total_ms": timing["total"],
                    }
                )
                timing_stored = True
            except OSError as exc:
                timing_log_error = f"{type(exc).__name__}: {exc}"
                warnings.append(
                    "Technickou časovou stopu se nepodařilo uložit."
                )

        response = {
            "analysis_kind": "predictive_stylometry",
            "schema_version": 2,
            "research_preview": True,
            "analysis_id": analysis_id,
            "verdict": verdict,
            "primary_decision": {
                "model": "apertus",
                "scope": primary_scope,
                "challenger_changes_verdict": False,
            },
            "quality": {
                "status": quality_status,
                "flags": quality_flags,
                "diagnostics": diagnostics,
            },
            "score_semantics": self.bundle["score_semantics"],
            "primary_scope": primary_scope,
            "scores": {
                "all": all_scope,
                "quality_unflagged": unflagged_scope,
                "primary": primary_scores,
            },
            "challenger": {
                "status": self.challenger_manifest["status"],
                "role": self.challenger_manifest["role"],
                "decision_authority": False,
                "model_name": self.challenger_bundle[
                    "candidate_name"
                ],
                "model_label_cs": self.challenger_bundle[
                    "display_name_cs"
                ],
                "model": challenger_model,
                "comparison": challenger_comparison,
                "score_semantics": self.challenger_bundle[
                    "score_semantics"
                ],
                "exploratory_performance": self.challenger_bundle[
                    "exploratory_performance"
                ]["source_blocked_news"],
                "promotion_decision": self.challenger_bundle[
                    "promotion_decision"
                ],
                "training": {
                    "document_count": self.challenger_bundle[
                        "training_data"
                    ]["quality_document_count"],
                    "human_count": self.challenger_bundle[
                        "training_data"
                    ]["quality_human_count"],
                    "ai_count": self.challenger_bundle[
                        "training_data"
                    ]["quality_ai_count"],
                    "news_ai_source_count": self.challenger_bundle[
                        "training_data"
                    ]["news_ai_source_count"],
                    "pair_count": self.challenger_bundle[
                        "training_data"
                    ]["pair_counts"]["all_pairs"],
                    "pair_source_count": self.challenger_bundle[
                        "training_data"
                    ]["pair_counts"]["all_sources"],
                },
                "limitations_cs": self.challenger_bundle[
                    "limitations_cs"
                ],
                "bundle": {
                    "schema_version": self.challenger_bundle[
                        "schema_version"
                    ],
                    "status": self.challenger_bundle["status"],
                    "created_at": self.challenger_bundle["created_at"],
                    "sha256": self.challenger_bundle_sha256,
                    "protocol_sha256": self.challenger_bundle[
                        "protocol"
                    ]["sha256"],
                    "runtime_protocol_sha256": self.challenger_bundle[
                        "runtime_protocol"
                    ]["sha256"],
                    "manifest_sha256": (
                        self.challenger_manifest_sha256
                    ),
                },
            },
            "predictive_profile": profile,
            "apertus_features": apertus_features,
            "sampling": {
                "stylometry": style_sampling,
                "apertus": apertus_sampling,
            },
            "input": {
                "character_count": len(text),
                "utf8_byte_count": utf8_bytes,
                "sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
            },
            "reader": {
                **self.bundle["reader"],
                "model_id": model_id,
                "model_generation": model_generation,
                "server_instance_id": server_instance_id,
            },
            "bundle": {
                "schema_version": self.bundle["schema_version"],
                "model_version": self.deployment["model_version"],
                "status": self.bundle["status"],
                "created_at": self.bundle["created_at"],
                "sha256": self.bundle_sha256,
                "model_lock_sha256": self.model_lock_sha256,
                "protocol_sha256": self.bundle["provenance"][
                    "expansion_protocol_sha256"
                ],
            },
            "deployment": {
                "status": self.deployment["status"],
                "sha256": self.deployment_sha256,
                "independent_audit": self.deployment[
                    "independent_audit"
                ]["primary"],
                "decision_display": self.deployment["decision_display"],
                "next_holdout": {
                    key: value
                    for key, value in self.deployment[
                        "next_holdout"
                    ].items()
                    if key != "protocol"
                },
            },
            "warnings": warnings,
            "timing_ms": timing,
            "timing_log": {
                "stored": timing_stored,
                "contains_text": False,
                "path": (
                    str(
                        self.timing_log_path.relative_to(
                            PREDICTIVE_ROOT
                        )
                    )
                    if self.timing_log_path is not None
                    and self.timing_log_path.is_relative_to(
                        PREDICTIVE_ROOT
                    )
                    else None
                ),
                "error": timing_log_error,
            },
            "privacy": {
                "stored": False,
                "stored_text": False,
                "stored_timing": timing_stored,
                "message_cs": (
                    "Text se neukládá; lokálně se ukládá jen technický "
                    "čas běhu bez obsahu textu."
                    if timing_stored
                    else "Text byl analyzován lokálně a neukládá se."
                ),
            },
        }
        return response
