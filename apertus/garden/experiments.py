"""Durable storage and lossless exports for Apertus Garden experiments.

The store deliberately knows very little about individual experiment payloads.
The canonical record envelope is validated here; kind-specific payloads remain
JSON so new experiment kinds do not require a database migration.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


DATABASE_SCHEMA_VERSION = 1
RECORD_SCHEMA = "apertus.experiment-record"
EXPORT_FORMAT = "apertus.experiment-export"
RECORD_SCHEMA_VERSION = 1
MAX_KIND_LENGTH = 128
MAX_LIST_LIMIT = 500
DEFAULT_LIST_LIMIT = 100

_KIND_RE = re.compile(rf"^[A-Za-z0-9][A-Za-z0-9_.:-]{{0,{MAX_KIND_LENGTH - 1}}}$")
_FORMULA_RE = re.compile(r"^\s*[=+\-@]")
_REQUIRED_RECORD_KEYS = {
    "schema",
    "schema_version",
    "id",
    "kind",
    "captured_at",
    "annotation",
    "capture",
    "provenance",
    "payload",
    "diagnostics",
}


class ExperimentError(RuntimeError):
    """Base class for errors safe to translate into an API response."""

    code = "experiment_error"


class ValidationError(ExperimentError):
    """The submitted value is not a canonical experiment record."""

    code = "validation_error"


class ConflictError(ExperimentError):
    """A record UUID already exists with different canonical content."""

    code = "id_conflict"

    def __init__(self, record_id: str):
        self.record_id = record_id
        super().__init__(
            f"Experiment {record_id!r} already exists with different content."
        )


class SchemaVersionError(ExperimentError):
    """The database or serialized envelope has an unsupported version."""

    code = "schema_version_unsupported"

    def __init__(self, message: str, *, found: int | None = None):
        self.found = found
        super().__init__(message)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _validate_timestamp(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise ValidationError(f"{field} must be a bounded ISO-8601 string.")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValidationError(f"{field} must be a valid ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{field} must include a UTC offset.")
    return value


def _validate_uuid(value: Any) -> str:
    if not isinstance(value, str):
        raise ValidationError("id must be a canonical UUID string.")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValidationError("id must be a canonical UUID string.") from exc
    if str(parsed) != value:
        raise ValidationError("id must use canonical lowercase UUID syntax.")
    return value


def _validate_kind(value: Any) -> str:
    if not isinstance(value, str) or not _KIND_RE.fullmatch(value):
        raise ValidationError(
            "kind must be a non-empty token of at most "
            f"{MAX_KIND_LENGTH} ASCII letters, digits, '.', '_', ':' or '-'."
        )
    return value


def _validate_json_value(value: Any, field: str, *, depth: int = 0) -> None:
    if depth > 100:
        raise ValidationError(f"{field} is nested too deeply.")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{field} contains a non-finite number.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field}[{index}]", depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{field} contains a non-string object key.")
            _validate_json_value(item, f"{field}.{key}", depth=depth + 1)
        return
    raise ValidationError(f"{field} contains a value that JSON cannot represent.")


def _canonical_record(
    record: Mapping[str, Any],
    *,
    allow_saved_at: bool = False,
) -> tuple[dict[str, Any], str, str | None]:
    if not isinstance(record, dict):
        raise ValidationError("An experiment record must be a JSON object.")
    missing = sorted(_REQUIRED_RECORD_KEYS - record.keys())
    if missing:
        raise ValidationError(f"Experiment record is missing: {', '.join(missing)}.")
    if record.get("schema") != RECORD_SCHEMA:
        raise ValidationError(f"schema must be {RECORD_SCHEMA!r}.")
    version = record.get("schema_version")
    if type(version) is not int or version != RECORD_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"Experiment record schema_version {version!r} is unsupported.",
            found=version if type(version) is int else None,
        )
    _validate_uuid(record.get("id"))
    _validate_kind(record.get("kind"))
    _validate_timestamp(record.get("captured_at"), "captured_at")
    if not isinstance(record.get("annotation"), dict):
        raise ValidationError("annotation must be a JSON object.")
    for field in ("capture", "provenance", "payload", "diagnostics"):
        if not isinstance(record.get(field), dict):
            raise ValidationError(f"{field} must be a JSON object.")
    if "format" in record:
        raise ValidationError("Record discriminator is 'schema', not 'format'.")

    saved_at = record.get("saved_at")
    if saved_at is not None:
        if not allow_saved_at:
            raise ValidationError("saved_at is assigned by the experiment store.")
        _validate_timestamp(saved_at, "saved_at")

    canonical = {key: value for key, value in record.items() if key != "saved_at"}
    _validate_json_value(canonical, "record")
    try:
        canonical_json = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError("Experiment record is not valid JSON.") from exc
    # Round-tripping also detaches stored content from caller-owned mutable values.
    return json.loads(canonical_json), canonical_json, saved_at


def _stored_record(canonical_json: str, saved_at: str) -> dict[str, Any]:
    record = json.loads(canonical_json)
    record["saved_at"] = saved_at
    return record


class ExperimentStore:
    """SQLite-backed immutable experiment records.

    A fresh SQLite connection is used for every public operation. This makes
    the class safe to call from ``ThreadingHTTPServer`` request threads without
    sharing connection state.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        if str(path) == ":memory:":
            raise ValidationError("ExperimentStore requires a file-backed database.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _raw_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path),
            timeout=5.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._raw_connection()
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version != DATABASE_SCHEMA_VERSION:
                raise SchemaVersionError(
                    "Experiment database user_version "
                    f"{version} is unsupported; expected {DATABASE_SCHEMA_VERSION}.",
                    found=version,
                )
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection
        finally:
            connection.close()

    @staticmethod
    @contextmanager
    def _write_transaction(connection: sqlite3.Connection) -> Iterator[None]:
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

    def _initialize(self) -> None:
        connection = self._raw_connection()
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > DATABASE_SCHEMA_VERSION:
                raise SchemaVersionError(
                    "Experiment database was created by a newer application "
                    f"(user_version {version}).",
                    found=version,
                )
            connection.execute("PRAGMA journal_mode = WAL")
            with self._write_transaction(connection):
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS experiment_records (
                        id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        captured_at TEXT NOT NULL,
                        saved_at TEXT NOT NULL,
                        canonical_json TEXT NOT NULL,
                        canonical_sha256 TEXT NOT NULL
                            CHECK(length(canonical_sha256) = 64)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS experiment_records_kind_saved
                    ON experiment_records(kind, saved_at DESC, id DESC)
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS experiment_store_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO experiment_store_meta (key, value)
                    VALUES ('archive_id', ?)
                    """,
                    (str(uuid.uuid4()),),
                )
                if version == 0:
                    connection.execute(
                        f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}"
                    )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(experiment_records)"
                ).fetchall()
            }
            expected = {
                "id",
                "kind",
                "captured_at",
                "saved_at",
                "canonical_json",
                "canonical_sha256",
            }
            if not expected.issubset(columns):
                raise SchemaVersionError(
                    "Experiment database schema is incomplete.",
                    found=version,
                )
        finally:
            connection.close()

    def status(self) -> dict[str, Any]:
        with self._connection() as connection:
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM experiment_records"
                ).fetchone()[0]
            )
            journal_mode = str(
                connection.execute("PRAGMA journal_mode").fetchone()[0]
            ).lower()
            archive_id = str(
                connection.execute(
                    """
                    SELECT value
                    FROM experiment_store_meta
                    WHERE key = 'archive_id'
                    """
                ).fetchone()[0]
            )
        return {
            "status": "ready",
            "ready": True,
            "archive_id": archive_id,
            "path_label": self.path.name,
            "schema_version": DATABASE_SCHEMA_VERSION,
            "count": count,
            "record_count": count,
            "journal_mode": journal_mode,
        }

    def save(self, record: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        canonical, canonical_json, _ = _canonical_record(record)
        return self._save_canonical(canonical, canonical_json, saved_at=None)

    def _save_canonical(
        self,
        canonical: Mapping[str, Any],
        canonical_json: str,
        *,
        saved_at: str | None,
    ) -> tuple[dict[str, Any], bool]:
        digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        assigned_saved_at = saved_at or _utc_now()
        with self._connection() as connection:
            with self._write_transaction(connection):
                existing = connection.execute(
                    """
                    SELECT canonical_json, canonical_sha256, saved_at
                    FROM experiment_records
                    WHERE id = ?
                    """,
                    (canonical["id"],),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["canonical_sha256"] != digest
                        or existing["canonical_json"] != canonical_json
                    ):
                        raise ConflictError(str(canonical["id"]))
                    return (
                        _stored_record(
                            existing["canonical_json"],
                            existing["saved_at"],
                        ),
                        False,
                    )
                connection.execute(
                    """
                    INSERT INTO experiment_records (
                        id, kind, captured_at, saved_at,
                        canonical_json, canonical_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        canonical["id"],
                        canonical["kind"],
                        canonical["captured_at"],
                        assigned_saved_at,
                        canonical_json,
                        digest,
                    ),
                )
        return _stored_record(canonical_json, assigned_saved_at), True

    def get(self, record_id: str) -> dict[str, Any] | None:
        _validate_uuid(record_id)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT canonical_json, saved_at
                FROM experiment_records
                WHERE id = ?
                """,
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        return _stored_record(row["canonical_json"], row["saved_at"])

    def list_summaries(
        self,
        kind: str | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        if kind is not None:
            _validate_kind(kind)
        if type(limit) is not int or not 1 <= limit <= MAX_LIST_LIMIT:
            raise ValidationError(f"limit must be between 1 and {MAX_LIST_LIMIT}.")
        if type(offset) is not int or offset < 0:
            raise ValidationError("offset must be a non-negative integer.")

        where = " WHERE kind = ?" if kind is not None else ""
        parameters: list[Any] = [kind] if kind is not None else []
        with self._connection() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM experiment_records{where}",
                    parameters,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT canonical_json, saved_at
                FROM experiment_records
                {where}
                ORDER BY saved_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchall()
        summaries = []
        for row in rows:
            record = _stored_record(row["canonical_json"], row["saved_at"])
            summary = {
                key: record[key]
                for key in (
                    "schema",
                    "schema_version",
                    "id",
                    "kind",
                    "captured_at",
                    "saved_at",
                    "annotation",
                    "capture",
                    "provenance",
                    "diagnostics",
                )
            }
            annotation = _mapping(record.get("annotation"))
            provenance = _mapping(record.get("provenance"))
            provenance_model = _mapping(provenance.get("model"))
            summary.update(
                {
                    "title": annotation.get("title"),
                    "notes": annotation.get("notes"),
                    "preview": annotation.get("notes") or annotation.get("title"),
                    "model_id": _value(
                        provenance_model, "model_id", "id"
                    ) or _value(provenance, "model_id"),
                    "capture_status": _value(record.get("capture"), "status"),
                }
            )
            summaries.append(summary)
        return {
            "records": summaries,
            "total": total,
            "kind": kind,
            "limit": limit,
            "offset": offset,
        }

    def import_payload(self, payload: Any) -> dict[str, Any]:
        records, preserve_saved_at = self._unwrap_import(payload)
        results: list[dict[str, Any]] = []
        counts = {"created": 0, "existing": 0, "conflicts": 0, "rejected": 0}
        for index, submitted in enumerate(records):
            raw_id = submitted.get("id") if isinstance(submitted, dict) else None
            try:
                normalized = self._normalize_import_record(submitted)
                canonical, canonical_json, imported_saved_at = _canonical_record(
                    normalized,
                    allow_saved_at=preserve_saved_at,
                )
                stored, created = self._save_canonical(
                    canonical,
                    canonical_json,
                    saved_at=imported_saved_at if preserve_saved_at else None,
                )
            except ConflictError as exc:
                counts["conflicts"] += 1
                results.append(
                    {
                        "index": index,
                        "id": exc.record_id,
                        "status": "conflict",
                        "code": exc.code,
                        "error": str(exc),
                    }
                )
            except ExperimentError as exc:
                counts["rejected"] += 1
                results.append(
                    {
                        "index": index,
                        "id": raw_id,
                        "status": "rejected",
                        "code": exc.code,
                        "error": str(exc),
                    }
                )
            else:
                status = "created" if created else "existing"
                counts[status] += 1
                results.append(
                    {
                        "index": index,
                        "id": stored["id"],
                        "status": status,
                        "saved_at": stored["saved_at"],
                    }
                )
        return {
            "total": len(records),
            "accepted": counts["created"] + counts["existing"],
            "imported": counts["created"],
            **counts,
            "results": results,
        }

    @staticmethod
    def _unwrap_import(payload: Any) -> tuple[list[Any], bool]:
        if isinstance(payload, list):
            return payload, False
        if not isinstance(payload, dict):
            raise ValidationError("Import payload must be a record, list, or wrapper.")
        if payload.get("format") == EXPORT_FORMAT:
            version = payload.get("schema_version")
            if type(version) is not int or version != RECORD_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"Experiment export schema_version {version!r} is unsupported.",
                    found=version if type(version) is int else None,
                )
            records = payload.get("records")
            if not isinstance(records, list):
                raise ValidationError("Experiment export records must be a list.")
            return records, True
        if "format" in payload and payload.get("format") != RECORD_SCHEMA:
            raise ValidationError("Unknown import format.")
        if "records" in payload and "schema" not in payload:
            records = payload["records"]
            if not isinstance(records, list):
                raise ValidationError("Import records must be a list.")
            return records, False
        return [payload], False

    @staticmethod
    def _normalize_import_record(submitted: Any) -> Any:
        if not isinstance(submitted, dict):
            return submitted
        record = dict(submitted)
        if "schema" not in record and record.get("format") == RECORD_SCHEMA:
            record["schema"] = record.pop("format")
        annotation = record.get("annotation")
        if annotation is None:
            record["annotation"] = {"title": "", "notes": "", "tags": []}
        elif isinstance(annotation, str):
            record["annotation"] = {
                "title": "",
                "notes": annotation,
                "tags": [],
            }
        return record

    def delete(self, record_id: str) -> bool:
        _validate_uuid(record_id)
        with self._connection() as connection:
            with self._write_transaction(connection):
                cursor = connection.execute(
                    "DELETE FROM experiment_records WHERE id = ?",
                    (record_id,),
                )
                deleted = cursor.rowcount > 0
        return deleted

    def _export_records(
        self,
        kind: str | None,
        *,
        record_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if kind is not None and record_id is not None:
            raise ValidationError(
                "Export may be filtered by kind or record_id, not both."
            )
        if kind is not None:
            _validate_kind(kind)
        if record_id is not None:
            _validate_uuid(record_id)
            where = " WHERE id = ?"
            parameters: Sequence[Any] = (record_id,)
        elif kind is not None:
            where = " WHERE kind = ?"
            parameters = (kind,)
        else:
            where = ""
            parameters = ()
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT canonical_json, saved_at
                FROM experiment_records
                {where}
                ORDER BY saved_at ASC, id ASC
                """,
                parameters,
            ).fetchall()
        return [
            _stored_record(row["canonical_json"], row["saved_at"])
            for row in rows
        ]

    def export_json(
        self,
        kind: str | None = None,
        *,
        record_id: str | None = None,
    ) -> bytes:
        records = self._export_records(kind, record_id=record_id)
        wrapper: dict[str, Any] = {
            "format": EXPORT_FORMAT,
            "schema_version": RECORD_SCHEMA_VERSION,
            "exported_at": _utc_now(),
            "record_count": len(records),
            "records": records,
        }
        if kind is not None:
            wrapper["kind"] = kind
        if record_id is not None:
            wrapper["record_id"] = record_id
        return (
            json.dumps(
                wrapper,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    def export_csv(
        self,
        kind: str | None = None,
        *,
        record_id: str | None = None,
    ) -> bytes:
        records = self._export_records(kind, record_id=record_id)
        rows: list[dict[str, Any]] = []
        for record in records:
            normalized_kind = record["kind"].lower().replace("-", "_")
            if normalized_kind == "model_comparison":
                rows.extend(_model_comparison_rows(record))
            elif normalized_kind == "text_surprisal_comparison":
                rows.extend(_text_surprisal_comparison_rows(record))
            elif normalized_kind == "text_surprisal":
                rows.extend(_text_surprisal_rows(record))
            elif "comparison" in normalized_kind:
                rows.extend(_comparison_rows(record))
            elif "garden" in normalized_kind:
                rows.extend(_garden_rows(record))
            elif "lens" in normalized_kind:
                rows.extend(_lens_rows(record))
            else:
                rows.append(_base_csv_row(record, "record"))

        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=_CSV_FIELDS,
            extrasaction="ignore",
            quoting=csv.QUOTE_ALL,
            lineterminator="\r\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in _CSV_FIELDS})
        return buffer.getvalue().encode("utf-8")


def _value(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _json_cell(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _model_comparison_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = _mapping(record.get("payload"))
    rows = []
    for item in payload.get("comparisons", []):
        units = item.get("units", [item])
        for unit in units:
            row = _base_csv_row(record, "model_comparison")
            row.update({
                "comparison_item_id": item.get("id"), "comparison_item_kind": item.get("kind"),
                "comparison_model_a": item.get("model_a"), "comparison_model_b": item.get("model_b"),
                "comparison_unit_text": unit.get("text"), "comparison_character_start": unit.get("character_start"),
                "comparison_character_end": unit.get("character_end"), "comparison_complete": unit.get("complete"),
                "comparison_a_bits": unit.get("a_bits", item.get("steering_effect_bits", item.get("context_effect_bits", {})).get("a")),
                "comparison_b_bits": unit.get("b_bits", item.get("steering_effect_bits", item.get("context_effect_bits", {})).get("b")),
                "comparison_delta_b_minus_a_bits": unit.get("delta_b_minus_a_bits", item.get("delta_steering_effect_b_minus_a_bits", item.get("delta_context_effect_b_minus_a_bits"))),
                "comparison_alignment_reasons_json": {"a": unit.get("a_reasons", []), "b": unit.get("b_reasons", [])},
            })
            if item.get("kind") == "steering":
                row["comparison_complete"] = True
                for side in ("a", "b"):
                    measurement = item[side]
                    for key in ("layer", "layer_count", "prefix_relative_shift_norm", "direction_sha256", "sham_sample_count", "sham_mean_effect_bits"):
                        row[f"steering_{side}_{key}"] = measurement.get(key)
                    for branch in ("negative", "baseline", "positive"):
                        row[f"steering_{side}_{branch}_contrast_bits"] = measurement["contrast_bits"][branch]
            # Canonical JSON is provided once; do not duplicate a whole batch on every CSV line.
            if rows:
                row["payload_json"] = None
            rows.append(row)
    return rows or [_base_csv_row(record, "model_comparison")]


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (dict, list)):
        text = _json_cell(value)
    else:
        text = str(value)
    if text.startswith(("\t", "\r", "\n")) or _FORMULA_RE.match(text):
        text = "'" + text
    return text


def _base_csv_row(record: Mapping[str, Any], export_scope: str) -> dict[str, Any]:
    capture = _mapping(record.get("capture"))
    provenance = _mapping(record.get("provenance"))
    provenance_model = _mapping(provenance.get("model"))
    provenance_execution = _mapping(provenance.get("execution"))
    payload = _mapping(record.get("payload"))
    payload_snapshot = _mapping(payload.get("snapshot"))
    payload_response = _mapping(payload.get("response"))
    payload_controls = _mapping(payload.get("controls"))
    payload_request = _mapping(payload.get("request"))
    payload_result = _mapping(payload.get("result"))
    return {
        "schema": record.get("schema"),
        "schema_version": record.get("schema_version"),
        "record_id": record.get("id"),
        "kind": record.get("kind"),
        "captured_at": record.get("captured_at"),
        "saved_at": record.get("saved_at"),
        "annotation_json": record.get("annotation"),
        "export_scope": export_scope,
        "capture_status": capture.get("status"),
        "model_id": _value(
            provenance_model, "model_id", "id"
        ) or _value(provenance, "model_id") or _value(
            capture, "model_id"
        ) or _value(payload_response, "model_id") or _value(
            payload_result, "model_id"
        ) or _value(payload, "model_id"),
        "model_generation": _value(
            provenance_model, "model_generation"
        ) or _value(provenance_execution, "model_generation") or _value(
            provenance, "model_generation"
        ) or _value(capture, "model_generation") or _value(
            payload_snapshot, "model_generation"
        ) or _value(payload_response, "model_generation") or _value(
            payload_result, "model_generation"
        ) or _value(payload, "model_generation"),
        "server_instance_id": _value(
            provenance_model, "server_instance_id"
        ) or _value(provenance_execution, "server_instance_id") or _value(
            provenance, "server_instance_id"
        ) or _value(capture, "server_instance_id") or _value(
            payload_snapshot, "server_instance_id"
        ) or _value(payload_response, "server_instance_id") or _value(
            payload_result, "server_instance_id"
        ) or _value(payload, "server_instance_id"),
        "temperature": _value(capture, "temperature")
        if _value(capture, "temperature") is not None
        else (
            _value(payload_snapshot, "temperature")
            if _value(payload_snapshot, "temperature") is not None
            else (
                _value(payload_controls, "temperature")
                if _value(payload_controls, "temperature") is not None
                else (
                    _value(payload_request, "temperature")
                    if _value(payload_request, "temperature") is not None
                    else _value(payload_result, "temperature")
                )
            )
        ),
        "probability_kind": _value(
            capture, "probability_kind"
        ) or _value(payload_response, "probability_kind") or _value(
            payload_result, "probability_kind"
        ) or _value(payload, "probability_kind"),
        "capture_json": capture,
        "provenance_json": provenance,
        "diagnostics_json": record.get("diagnostics"),
        "payload_json": payload,
    }


def _comparison_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = _mapping(record.get("payload"))
    response = _mapping(payload.get("response")) or payload
    capture = _mapping(record.get("capture"))
    snapshot = (
        _mapping(payload.get("snapshot"))
        or _mapping(capture.get("snapshot"))
        or capture
    )
    left_snapshot = _mapping(_value(snapshot, "left", "a"))
    right_snapshot = _mapping(_value(snapshot, "right", "b"))
    left = _mapping(_value(response, "left", "a"))
    right = _mapping(_value(response, "right", "b"))
    base = _base_csv_row(record, "displayed_movers")
    base.update(
        {
            "a_prefix": _value(left_snapshot, "prefix", "text")
            or _value(left, "prefix", "text"),
            "b_prefix": _value(right_snapshot, "prefix", "text")
            or _value(right, "prefix", "text"),
            "entropy_a_bits": _value(left, "entropy_bits"),
            "entropy_b_bits": _value(right, "entropy_bits"),
            "delta_entropy_b_minus_a_bits": _value(
                response, "delta_entropy_bits"
            ),
            "effective_tokens_a": _value(
                left, "effective_tokens", "effective_width"
            ),
            "effective_tokens_b": _value(
                right, "effective_tokens", "effective_width"
            ),
            "effective_width_b_over_a": _value(
                response, "effective_support_ratio"
            ),
            "js_divergence_bits": _value(response, "js_divergence_bits"),
            "total_variation": _value(response, "total_variation"),
        }
    )
    overlap = _mapping(response.get("top_k_overlap"))
    base.update(
        {
            "top_k": _value(overlap, "k"),
            "top_k_overlap": _value(overlap, "count"),
            "top_k_overlap_fraction": _value(overlap, "fraction"),
        }
    )
    mover_container = _mapping(response.get("movers"))
    movers = [
        *[("gain_in_b", item) for item in _list(response.get("gains"))],
        *[("loss_in_b", item) for item in _list(response.get("losses"))],
    ]
    if not movers:
        movers = [
            *[("gain_in_b", item) for item in _list(mover_container.get("gains"))],
            *[("loss_in_b", item) for item in _list(mover_container.get("losses"))],
        ]
    if not movers:
        return [base]
    rows = []
    for direction, item in movers:
        mover = _mapping(item)
        row = dict(base)
        row.update(
            {
                "mover_direction": direction,
                "token_id": _value(mover, "token_id", "id"),
                "token": _value(mover, "display", "token", "text"),
                "is_special": _value(mover, "is_special"),
                "probability_a": _value(
                    mover, "probability_left", "probability_a"
                ),
                "probability_b": _value(
                    mover, "probability_right", "probability_b"
                ),
                "delta_b_minus_a": _value(mover, "delta"),
                "rank_a": _value(mover, "rank_left", "rank_a"),
                "rank_b": _value(mover, "rank_right", "rank_b"),
            }
        )
        rows.append(row)
    return rows


def _garden_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = _mapping(record.get("payload"))
    garden = _mapping(payload.get("garden")) or payload
    columns = _list(garden.get("columns"))
    if not columns:
        columns = [{"nodes": _list(garden.get("nodes"))}]
    rows: list[dict[str, Any]] = []
    for column_index, column_value in enumerate(columns):
        column = _mapping(column_value)
        for node_index, node_value in enumerate(_list(column.get("nodes"))):
            node = _mapping(node_value)
            data = _mapping(node.get("distribution")) or _mapping(node.get("data"))
            incoming = _mapping(node.get("incoming"))
            base = _base_csv_row(record, "captured_nodes")
            base.update(
                {
                    "column_id": _value(
                        column, "session_column_id", "id", "column_id"
                    ),
                    "column_index": _value(column, "index", "column_index")
                    if _value(column, "index", "column_index") is not None
                    else column_index,
                    "column_label": _value(column, "label", "title", "prompt"),
                    "node_id": _value(
                        node, "session_node_id", "id", "node_id"
                    ),
                    "parent_node_id": _value(
                        node, "session_parent_id", "parent_id", "parentId"
                    ),
                    "node_prefix": _value(node, "prefix", "text")
                    or _value(data, "text"),
                    "node_depth": _value(node, "depth")
                    if _value(node, "depth") is not None
                    else node_index,
                    "node_active": _value(node, "active", "is_active"),
                    "node_stale": _value(node, "stale"),
                    "incoming_token_id": _value(
                        node, "incoming_token_id", "incomingTokenId"
                    )
                    if _value(node, "incoming_token_id", "incomingTokenId")
                    is not None
                    else _value(incoming, "token_id"),
                    "incoming_token_probability": _value(
                        node, "incoming_probability", "incomingProbability"
                    )
                    if _value(
                        node, "incoming_probability", "incomingProbability"
                    )
                    is not None
                    else _value(incoming, "probability"),
                    "other_probability": _value(data, "other_probability"),
                    "entropy_bits": _value(data, "entropy_bits"),
                    "effective_tokens": _value(
                        data, "effective_tokens", "effective_width"
                    ),
                    "normalized_entropy": _value(data, "normalized_entropy"),
                    "node_json": node,
                }
            )
            rows.append(base)
            candidate_groups = (
                ("candidate", _list(node.get("candidates")) or _list(data.get("candidates"))),
                (
                    "pinned_candidate",
                    _list(node.get("pinned_candidates"))
                    or _list(data.get("pinned_candidates")),
                ),
            )
            candidate_index = 0
            for candidate_kind, candidates in candidate_groups:
                for candidate_value in candidates:
                    candidate = _mapping(candidate_value)
                    row = dict(base)
                    row["export_scope"] = "displayed_candidates"
                    row.update(
                        {
                            "candidate_index": candidate_index,
                            "candidate_kind": candidate_kind,
                            "candidate_token_id": _value(candidate, "token_id", "id"),
                            "candidate_token": _value(
                                candidate, "display", "token", "text"
                            ),
                            "candidate_is_special": _value(candidate, "is_special"),
                            "candidate_probability": _value(
                                candidate, "probability"
                            ),
                            "candidate_raw_probability": _value(
                                candidate, "raw_probability"
                            ),
                            "candidate_json": candidate,
                        }
                    )
                    rows.append(row)
                    candidate_index += 1
    return rows or [_base_csv_row(record, "captured_nodes")]


def _lens_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = _mapping(record.get("payload"))
    reader = _mapping(payload.get("reader"))
    reader_kind = _value(reader, "kind")
    lens = (
        _mapping(payload.get("result"))
        or _mapping(payload.get("response"))
        or _mapping(payload.get("lens"))
        or payload
    )
    calibration = (
        _mapping(lens.get("calibration"))
        or _mapping(payload.get("calibration"))
        or _mapping(reader.get("calibration"))
    )
    artifact = (
        _mapping(lens.get("artifact"))
        or _mapping(payload.get("artifact"))
        or calibration
    )
    transfer_probe = (
        _mapping(lens.get("transfer_probe"))
        or _mapping(payload.get("transfer_probe"))
    )
    validation = (
        _mapping(lens.get("validation"))
        or _mapping(payload.get("validation"))
        or _mapping(artifact.get("validation"))
    )
    validation_status = _value(validation, "status") or _value(
        _mapping(validation.get("acceptance")),
        "status",
    )
    artifact_id = _value(
        artifact,
        "artifact_id",
        "calibration_id",
        "id",
    ) or _value(reader, "artifact_id", "calibration_id")
    tuned_reader = _mapping(reader.get("tuned"))
    artifact_manifest_sha256 = _value(
        artifact,
        "canonical_manifest_sha256",
        "artifact_sha256",
        "manifest_sha256",
    ) or _value(tuned_reader, "artifact_sha256")
    weights_sha256 = _value(
        _mapping(artifact.get("weights")),
        "sha256",
    ) or _value(tuned_reader, "weights_sha256")
    metrics_sha256 = _value(
        _mapping(artifact.get("metrics")),
        "sha256",
    ) or _value(tuned_reader, "metrics_sha256")
    artifact_model = _mapping(artifact.get("model"))
    artifact_reader = _mapping(artifact.get("reader"))
    artifact_compatibility = _mapping(artifact.get("compatibility"))
    rows: list[dict[str, Any]] = []
    for layer_index, layer_value in enumerate(_list(lens.get("layers"))):
        layer = _mapping(layer_value)
        paired_variants = [
            (variant, _mapping(layer.get(variant)))
            for variant in ("raw", "tuned", "jacobian")
            if isinstance(layer.get(variant), dict)
        ]
        variants = paired_variants or [("raw", layer)]
        for variant, reading in variants:
            if not reading:
                continue
            base = _base_csv_row(record, "captured_layers")
            base.update(
                {
                    "reader_kind": reader_kind,
                    "lens_variant": variant,
                    "artifact_id": artifact_id,
                    "artifact_manifest_sha256": artifact_manifest_sha256,
                    "artifact_weights_sha256": weights_sha256,
                    "artifact_metrics_sha256": metrics_sha256,
                    "artifact_format": _value(artifact, "format"),
                    "artifact_schema_version": _value(
                        artifact, "schema_version"
                    ),
                    "artifact_scope": _value(artifact, "artifact_scope"),
                    "artifact_model_mode": _value(
                        artifact_model, "mode", "model_id"
                    ),
                    "artifact_source_layers_json": _value(
                        artifact_reader, "source_layers", "selected_layers"
                    ),
                    "artifact_target_layer": _value(
                        artifact_reader, "target_layer", "reference_layer"
                    ),
                    "artifact_compatibility_json": artifact_compatibility,
                    "transfer_probe_status": _value(
                        transfer_probe, "status"
                    ),
                    "transfer_probe_json": transfer_probe,
                    "validation_status": validation_status,
                    "layer": _value(layer, "layer")
                    if _value(layer, "layer") is not None
                    else layer_index + 1,
                    "layer_index": layer_index,
                    "top1_token_id": _value(reading, "top1_token_id"),
                    "top2_token_id": _value(reading, "top2_token_id"),
                    "top_k_mass": _value(reading, "top_k_mass"),
                    "lens_other_probability": _value(
                        reading, "other_probability"
                    ),
                    "lens_entropy_bits": _value(reading, "entropy_bits"),
                    "effective_width": _value(
                        reading, "effective_width", "effective_tokens"
                    ),
                    "kl_divergence_to_final_bits": _value(
                        reading, "kl_divergence_to_final_bits"
                    ),
                    "js_divergence_to_final_bits": _value(
                        reading, "js_divergence_to_final_bits"
                    ),
                    "final_top1_token_id": _value(
                        reading, "final_top1_token_id"
                    ),
                    "final_top1_rank": _value(
                        reading, "final_top1_rank"
                    ),
                    "final_top1_probability": _value(
                        reading, "final_top1_probability"
                    ),
                    "top1_top2_gap": _value(
                        reading, "top1_top2_gap"
                    ),
                    "top1_top2_gap_bits": _value(
                        reading, "top1_top2_gap_bits"
                    ),
                    "layer_json": {
                        "layer": _value(layer, "layer"),
                        "variant": variant,
                        "reading": reading,
                        "comparison": _mapping(
                            layer.get("comparison")
                        ),
                    },
                }
            )
            rows.append(base)
            for candidate_index, candidate_value in enumerate(
                _list(reading.get("candidates"))
            ):
                candidate = _mapping(candidate_value)
                row = dict(base)
                row["export_scope"] = "displayed_candidates"
                row.update(
                    {
                        "lens_candidate_index": candidate_index,
                        "lens_candidate_token_id": _value(
                            candidate, "token_id", "id"
                        ),
                        "lens_candidate_token": _value(
                            candidate, "display", "token", "text"
                        ),
                        "lens_candidate_is_special": _value(
                            candidate, "is_special"
                        ),
                        "lens_candidate_probability": _value(
                            candidate, "probability"
                        ),
                        "lens_candidate_json": candidate,
                    }
                )
                rows.append(row)
    return rows or [_base_csv_row(record, "captured_layers")]


def _text_surprisal_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten a text-surprisal capture without replacing its canonical JSON.

    The summary, observed-token and alternative rows deliberately repeat the
    common run metadata. This keeps filtered CSVs useful on their own while
    ``payload_json`` remains the lossless source of truth.
    """

    payload = _mapping(record.get("payload"))
    result = _mapping(payload.get("result"))
    response = (
        _mapping(payload.get("response"))
        or _mapping(result.get("response"))
        or result
        or payload
    )
    request = _mapping(payload.get("request"))
    summary = _mapping(response.get("summary")) or _mapping(payload.get("summary"))
    tokens = _list(response.get("tokens")) or _list(payload.get("tokens"))
    aggregations = _mapping(response.get("aggregations")) or _mapping(
        payload.get("aggregations")
    )
    aggregation_method = _mapping(
        response.get("aggregation_method")
    ) or _mapping(payload.get("aggregation_method"))

    original_text = response.get("original_text")
    if original_text is None:
        original_text = request.get("text")
    if original_text is None:
        original_text = response.get("text")
    analyzed_text = response.get("text")
    if analyzed_text is None:
        analyzed_text = original_text

    base = _base_csv_row(record, "text_surprisal_summary")
    response_temperature = _value(response, "temperature")
    base.update(
        {
            "analysis_kind": _value(response, "analysis_kind"),
            "original_text": original_text,
            "normalized_text": _value(response, "normalized_text", "text"),
            "text": analyzed_text,
            "normalization": _value(response, "normalization"),
            "character_count": _value(response, "character_count"),
            "utf8_byte_count": _value(response, "utf8_byte_count"),
            "token_count": _value(response, "token_count")
            if _value(response, "token_count") is not None
            else len(tokens),
            "top_k": _value(response, "top_k")
            if _value(response, "top_k") is not None
            else _value(request, "top_k"),
            "temperature": response_temperature
            if response_temperature is not None
            else base.get("temperature"),
            "total_surprisal_bits": _value(summary, "total_surprisal_bits"),
            "mean_surprisal_bits": _value(summary, "mean_surprisal_bits"),
            "perplexity": _value(summary, "perplexity"),
            "bits_per_character": _value(summary, "bits_per_character"),
            "bits_per_utf8_byte": _value(summary, "bits_per_utf8_byte"),
            "mean_entropy_bits": _value(summary, "mean_entropy_bits"),
            "max_surprisal_token_json": summary.get("max_surprisal_token"),
            "transition_id": _value(response, "transition_id"),
            "latency_ms": _value(response, "latency_ms"),
            "execution_json": response.get("execution"),
            "timing_ms_json": response.get("timing_ms"),
            "surprisal_summary_json": summary,
            "aggregation_method_json": aggregation_method,
            "aggregations_json": aggregations,
        }
    )

    rows = [base]
    canonical_aggregation_types = (
        "whitespace_span",
        "line",
        "sentence_like",
    )
    aggregation_types = [
        aggregation_type
        for aggregation_type in canonical_aggregation_types
        if aggregation_type in aggregations
    ]
    aggregation_types.extend(
        sorted(
            aggregation_type
            for aggregation_type in aggregations
            if aggregation_type not in canonical_aggregation_types
        )
    )
    for aggregation_type in aggregation_types:
        aggregation = _mapping(aggregations.get(aggregation_type))
        aggregation_summary = _mapping(aggregation.get("summary"))
        aggregation_spans = _list(aggregation.get("spans"))
        aggregation_total_token_count = _value(
            aggregation_summary, "total_token_count"
        )
        if aggregation_total_token_count is None:
            aggregation_token_counts = [
                _value(aggregation_summary, key)
                for key in (
                    "assigned_token_count",
                    "ambiguous_token_count",
                    "unassigned_token_count",
                )
            ]
            if all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in aggregation_token_counts
            ):
                aggregation_total_token_count = sum(aggregation_token_counts)

        summary_row = dict(base)
        summary_row["export_scope"] = "text_surprisal_aggregation_summary"
        summary_row.update(
            {
                "aggregation_type": aggregation_type,
                "aggregation_label": _value(aggregation, "label_cs", "label"),
                "aggregation_unit": _value(aggregation, "unit")
                or aggregation_type,
                "aggregation_span_count": _value(
                    aggregation_summary, "span_count"
                )
                if _value(aggregation_summary, "span_count") is not None
                else len(aggregation_spans),
                "aggregation_complete_span_count": _value(
                    aggregation_summary, "complete_span_count", "count"
                ),
                "aggregation_incomplete_span_count": _value(
                    aggregation_summary, "incomplete_span_count"
                ),
                "aggregation_assigned_token_count": _value(
                    aggregation_summary, "assigned_token_count"
                ),
                "aggregation_ambiguous_token_count": _value(
                    aggregation_summary, "ambiguous_token_count"
                ),
                "aggregation_unassigned_token_count": _value(
                    aggregation_summary, "unassigned_token_count"
                ),
                "aggregation_total_token_count": aggregation_total_token_count,
                "aggregation_coverage_ratio": _value(
                    aggregation_summary, "coverage_ratio"
                ),
                "aggregation_mean": _value(aggregation_summary, "mean"),
                "aggregation_median": _value(aggregation_summary, "median"),
                "aggregation_q1": _value(aggregation_summary, "q1"),
                "aggregation_q3": _value(aggregation_summary, "q3"),
                "aggregation_iqr": _value(aggregation_summary, "iqr"),
                "aggregation_raw_mad": _value(
                    aggregation_summary, "raw_mad", "mad"
                ),
                "aggregation_min": _value(aggregation_summary, "min"),
                "aggregation_max": _value(aggregation_summary, "max"),
                "aggregation_summary_json": aggregation_summary,
            }
        )
        rows.append(summary_row)

        for fallback_span_index, span_value in enumerate(aggregation_spans):
            span = _mapping(span_value)
            ambiguous_token_indices = _list(
                span.get("ambiguous_token_indices")
            )
            span_row = dict(summary_row)
            span_row["export_scope"] = "text_surprisal_aggregation_spans"
            span_row.update(
                {
                    "aggregation_span_index": _value(span, "index")
                    if _value(span, "index") is not None
                    else fallback_span_index,
                    "aggregation_character_start": _value(
                        span, "character_start"
                    ),
                    "aggregation_character_end": _value(span, "character_end"),
                    "aggregation_text": _value(span, "text"),
                    "aggregation_token_indices_json": _value(
                        span, "token_indices"
                    ),
                    "aggregation_token_ids_json": _value(span, "token_ids"),
                    "aggregation_token_count": _value(span, "token_count"),
                    "aggregation_total_surprisal_bits": _value(
                        span, "total_surprisal_bits"
                    ),
                    "aggregation_surprisal_bits_per_token": _value(
                        span, "surprisal_bits_per_token"
                    ),
                    "aggregation_surprisal_bits_per_non_whitespace_character": (
                        _value(
                            span,
                            "surprisal_bits_per_non_whitespace_character",
                            "surprisal_bits_per_character",
                        )
                    ),
                    "aggregation_character_count": _value(
                        span, "character_count"
                    ),
                    "aggregation_non_whitespace_character_count": _value(
                        span, "non_whitespace_character_count"
                    ),
                    "aggregation_covered_non_whitespace_character_count": _value(
                        span, "covered_non_whitespace_character_count"
                    ),
                    "aggregation_character_coverage_ratio": _value(
                        span, "character_coverage_ratio"
                    ),
                    "aggregation_complete": _value(span, "complete"),
                    "aggregation_span_ambiguous_token_count": _value(
                        span, "ambiguous_token_count"
                    ),
                    "aggregation_ambiguous_token_indices_json": _value(
                        span, "ambiguous_token_indices"
                    ),
                    "aggregation_span_json": span_value,
                }
            )
            if span_row["aggregation_span_ambiguous_token_count"] is None:
                span_row["aggregation_span_ambiguous_token_count"] = len(
                    ambiguous_token_indices
                )
            rows.append(span_row)

    for fallback_index, token_value in enumerate(tokens):
        token = _mapping(token_value)
        token_index = _value(token, "index")
        if token_index is None:
            token_index = fallback_index
        token_row = dict(base)
        token_row["export_scope"] = "text_surprisal_tokens"
        token_row.update(
            {
                "text_token_index": token_index,
                "text_token_id": _value(token, "token_id", "id"),
                "text_token": _value(token, "text", "piece", "display"),
                "text_token_piece": _value(token, "piece", "text"),
                "text_token_display": _value(token, "display", "text", "piece"),
                "tokenizer_token": _value(token, "tokenizer_token"),
                "token_character_start": _value(token, "character_start"),
                "token_character_end": _value(token, "character_end"),
                "token_probability": _value(token, "probability"),
                "token_logprob": _value(token, "logprob"),
                "token_surprisal_bits": _value(token, "surprisal_bits"),
                "token_rank": _value(token, "rank"),
                "token_entropy_bits": _value(token, "entropy_bits"),
                "token_effective_tokens": _value(
                    token, "effective_tokens", "effective_width"
                ),
                "token_top1_gap_bits": _value(token, "top1_gap_bits"),
                "token_word_start_hint": _value(token, "word_start_hint"),
                "surprisal_token_json": token_value,
            }
        )
        rows.append(token_row)

        for alternative_index, alternative_value in enumerate(
            _list(token.get("alternatives"))
        ):
            alternative = _mapping(alternative_value)
            row = dict(token_row)
            row["export_scope"] = "text_surprisal_alternatives"
            row.update(
                {
                    "alternative_index": alternative_index,
                    "alternative_token_id": _value(
                        alternative, "token_id", "id"
                    ),
                    "alternative_token": _value(
                        alternative, "text", "piece", "display"
                    ),
                    "alternative_display": _value(
                        alternative, "display", "text", "piece"
                    ),
                    "alternative_is_special": _value(
                        alternative, "is_special"
                    ),
                    "alternative_probability": _value(
                        alternative, "probability"
                    ),
                    "alternative_rank": _value(alternative, "rank"),
                    "surprisal_alternative_json": alternative_value,
                }
            )
            rows.append(row)
    return rows


def _text_surprisal_comparison_side_response(value: Any) -> dict[str, Any]:
    """Read a full comparison side without requiring one UI wrapper shape."""

    side = _mapping(value)
    response = _mapping(side.get("response"))
    if response:
        return response
    result = _mapping(side.get("result"))
    nested_response = _mapping(result.get("response"))
    if nested_response:
        return nested_response
    return result or side


def _comparison_method_identifier(value: Any) -> Any:
    if isinstance(value, dict):
        return _value(value, "id", "name")
    return value


def _text_surprisal_comparison_rows(
    record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Flatten paired teacher-forced captures while keeping their pairing.

    A and B remain independent token sequences.  The export repeats shared
    provenance and the complete canonical payload, but never invents a token
    alignment.  Comparison rows are limited to the explicitly stored summary
    metrics whose delta definition is part of the record.
    """

    payload = _mapping(record.get("payload"))
    comparison = _mapping(payload.get("comparison"))
    labels = _mapping(payload.get("labels")) or _mapping(
        comparison.get("labels")
    )
    comparison_method_value = _value(comparison, "method") or _value(
        payload, "comparison_method"
    )
    comparison_method = _comparison_method_identifier(
        comparison_method_value
    )
    comparison_provenance = _mapping(
        comparison.get("provenance")
    ) or _mapping(payload.get("comparison_provenance")) or _mapping(
        record.get("provenance")
    )
    delta_definition = _value(
        comparison, "delta_definition"
    ) or "B - A"
    rows: list[dict[str, Any]] = []
    scope_names = {
        "text_surprisal_summary": (
            "text_surprisal_comparison_side_summary"
        ),
        "text_surprisal_aggregation_summary": (
            "text_surprisal_comparison_side_aggregation_summary"
        ),
        "text_surprisal_aggregation_spans": (
            "text_surprisal_comparison_side_aggregation_spans"
        ),
        "text_surprisal_tokens": (
            "text_surprisal_comparison_side_tokens"
        ),
        "text_surprisal_alternatives": (
            "text_surprisal_comparison_side_alternatives"
        ),
    }

    for side_key, side_name in (("a", "A"), ("b", "B")):
        side_payload = _mapping(payload.get(side_key))
        side_response = _text_surprisal_comparison_side_response(
            side_payload
        )
        label = _value(labels, side_key, side_name)
        if label is None:
            label = _value(side_payload, "label", "name")
        if label is None:
            label = side_name

        synthetic_record = dict(record)
        synthetic_record["payload"] = {"response": side_response}
        for row in _text_surprisal_rows(synthetic_record):
            row["export_scope"] = scope_names.get(
                str(row.get("export_scope")),
                "text_surprisal_comparison_side",
            )
            row.update(
                {
                    "comparison_side": side_name,
                    "comparison_side_label": label,
                    "comparison_side_json": side_payload,
                    "comparison_delta_definition": delta_definition,
                    "comparison_method": comparison_method,
                    "comparison_method_json": comparison_method_value,
                    "comparison_provenance_json": comparison_provenance,
                    "comparison_labels_json": labels,
                    "comparison_json": comparison,
                    # Restore the paired source of truth after the synthetic
                    # response was used solely to reuse single-side flattening.
                    "payload_json": payload,
                }
            )
            rows.append(row)

    metrics = _mapping(comparison.get("metrics"))
    canonical_metric_names = (
        "mean_surprisal_bits",
        "bits_per_non_whitespace_character",
        "median_whitespace_span_bits_per_non_whitespace_character",
        "whitespace_span_iqr_bits_per_non_whitespace_character",
        "whitespace_span_raw_mad_bits_per_non_whitespace_character",
    )
    metric_names = [
        name for name in canonical_metric_names if name in metrics
    ]
    metric_names.extend(
        sorted(name for name in metrics if name not in canonical_metric_names)
    )
    for metric_name in metric_names:
        metric = _mapping(metrics.get(metric_name))
        metric_method_value = _value(metric, "method")
        metric_method = _comparison_method_identifier(metric_method_value)
        if metric_method is None:
            metric_method = comparison_method
        metric_provenance = _mapping(metric.get("provenance"))
        if not metric_provenance:
            metric_provenance = comparison_provenance
        row = _base_csv_row(
            record,
            "text_surprisal_comparison_metric",
        )
        row.update(
            {
                "comparison_metric": metric_name,
                "comparison_metric_a": _value(metric, "a"),
                "comparison_metric_b": _value(metric, "b"),
                "comparison_metric_delta_b_minus_a": _value(
                    metric, "delta_b_minus_a"
                ),
                "comparison_metric_unit": _value(metric, "unit"),
                "comparison_metric_method": metric_method,
                "comparison_metric_provenance_json": metric_provenance,
                "comparison_delta_definition": delta_definition,
                "comparison_method": comparison_method,
                "comparison_method_json": comparison_method_value,
                "comparison_provenance_json": comparison_provenance,
                "comparison_labels_json": labels,
                "comparison_json": comparison,
            }
        )
        rows.append(row)

    return rows or [
        _base_csv_row(record, "text_surprisal_comparison")
    ]


_CSV_FIELDS = [
    "schema",
    "schema_version",
    "record_id",
    "kind",
    "captured_at",
    "saved_at",
    "annotation_json",
    "export_scope",
    "capture_status",
    "model_id",
    "model_generation",
    "server_instance_id",
    "temperature",
    "probability_kind",
    "analysis_kind",
    "comparison_side",
    "comparison_side_label",
    "comparison_metric",
    "comparison_metric_a",
    "comparison_metric_b",
    "comparison_metric_delta_b_minus_a",
    "comparison_metric_unit",
    "comparison_metric_method",
    "comparison_delta_definition",
    "comparison_method",
    "original_text",
    "normalized_text",
    "text",
    "normalization",
    "character_count",
    "utf8_byte_count",
    "token_count",
    "total_surprisal_bits",
    "mean_surprisal_bits",
    "perplexity",
    "bits_per_character",
    "bits_per_utf8_byte",
    "mean_entropy_bits",
    "max_surprisal_token_json",
    "aggregation_type",
    "aggregation_label",
    "aggregation_unit",
    "aggregation_span_index",
    "aggregation_character_start",
    "aggregation_character_end",
    "aggregation_text",
    "aggregation_token_indices_json",
    "aggregation_token_ids_json",
    "aggregation_token_count",
    "aggregation_total_surprisal_bits",
    "aggregation_surprisal_bits_per_token",
    "aggregation_surprisal_bits_per_non_whitespace_character",
    "aggregation_character_count",
    "aggregation_non_whitespace_character_count",
    "aggregation_covered_non_whitespace_character_count",
    "aggregation_character_coverage_ratio",
    "aggregation_complete",
    "aggregation_span_ambiguous_token_count",
    "aggregation_ambiguous_token_indices_json",
    "aggregation_span_count",
    "aggregation_complete_span_count",
    "aggregation_incomplete_span_count",
    "aggregation_assigned_token_count",
    "aggregation_ambiguous_token_count",
    "aggregation_unassigned_token_count",
    "aggregation_total_token_count",
    "aggregation_coverage_ratio",
    "aggregation_mean",
    "aggregation_median",
    "aggregation_q1",
    "aggregation_q3",
    "aggregation_iqr",
    "aggregation_raw_mad",
    "aggregation_min",
    "aggregation_max",
    "transition_id",
    "latency_ms",
    "text_token_index",
    "text_token_id",
    "text_token",
    "text_token_piece",
    "text_token_display",
    "tokenizer_token",
    "token_character_start",
    "token_character_end",
    "token_probability",
    "token_logprob",
    "token_surprisal_bits",
    "token_rank",
    "token_entropy_bits",
    "token_effective_tokens",
    "token_top1_gap_bits",
    "token_word_start_hint",
    "alternative_index",
    "alternative_token_id",
    "alternative_token",
    "alternative_display",
    "alternative_is_special",
    "alternative_probability",
    "alternative_rank",
    "a_prefix",
    "b_prefix",
    "entropy_a_bits",
    "entropy_b_bits",
    "delta_entropy_b_minus_a_bits",
    "effective_tokens_a",
    "effective_tokens_b",
    "effective_width_b_over_a",
    "js_divergence_bits",
    "total_variation",
    "top_k",
    "top_k_overlap",
    "top_k_overlap_fraction",
    "mover_direction",
    "token_id",
    "token",
    "is_special",
    "probability_a",
    "probability_b",
    "delta_b_minus_a",
    "rank_a",
    "rank_b",
    "column_id",
    "column_index",
    "column_label",
    "node_id",
    "parent_node_id",
    "node_prefix",
    "node_depth",
    "node_active",
    "node_stale",
    "incoming_token_id",
    "incoming_token_probability",
    "candidate_index",
    "candidate_kind",
    "candidate_token_id",
    "candidate_token",
    "candidate_is_special",
    "candidate_probability",
    "candidate_raw_probability",
    "other_probability",
    "entropy_bits",
    "effective_tokens",
    "normalized_entropy",
    "reader_kind",
    "lens_variant",
    "artifact_id",
    "artifact_manifest_sha256",
    "artifact_weights_sha256",
    "artifact_metrics_sha256",
    "artifact_format",
    "artifact_schema_version",
    "artifact_scope",
    "artifact_model_mode",
    "artifact_source_layers_json",
    "artifact_target_layer",
    "artifact_compatibility_json",
    "transfer_probe_status",
    "validation_status",
    "layer",
    "layer_index",
    "top1_token_id",
    "top2_token_id",
    "top_k_mass",
    "lens_other_probability",
    "lens_entropy_bits",
    "effective_width",
    "kl_divergence_to_final_bits",
    "js_divergence_to_final_bits",
    "final_top1_token_id",
    "final_top1_rank",
    "final_top1_probability",
    "top1_top2_gap",
    "top1_top2_gap_bits",
    "lens_candidate_index",
    "lens_candidate_token_id",
    "lens_candidate_token",
    "lens_candidate_is_special",
    "lens_candidate_probability",
    "capture_json",
    "provenance_json",
    "diagnostics_json",
    "payload_json",
    "node_json",
    "candidate_json",
    "layer_json",
    "lens_candidate_json",
    "transfer_probe_json",
    "surprisal_summary_json",
    "execution_json",
    "timing_ms_json",
    "surprisal_token_json",
    "surprisal_alternative_json",
    "aggregation_method_json",
    "aggregations_json",
    "aggregation_summary_json",
    "aggregation_span_json",
    "comparison_side_json",
    "comparison_metric_provenance_json",
    "comparison_provenance_json",
    "comparison_labels_json",
    "comparison_json",
    "comparison_method_json",
    "comparison_item_id", "comparison_item_kind", "comparison_model_a", "comparison_model_b",
    "comparison_unit_text", "comparison_character_start", "comparison_character_end", "comparison_complete",
    "comparison_a_bits", "comparison_b_bits", "comparison_delta_b_minus_a_bits", "comparison_alignment_reasons_json",
    "steering_a_layer",
    "steering_a_layer_count",
    "steering_a_prefix_relative_shift_norm",
    "steering_a_direction_sha256",
    "steering_a_sham_sample_count",
    "steering_a_sham_mean_effect_bits",
    "steering_a_negative_contrast_bits",
    "steering_a_baseline_contrast_bits",
    "steering_a_positive_contrast_bits",
    "steering_b_layer",
    "steering_b_layer_count",
    "steering_b_prefix_relative_shift_norm",
    "steering_b_direction_sha256",
    "steering_b_sham_sample_count",
    "steering_b_sham_mean_effect_bits",
    "steering_b_negative_contrast_bits",
    "steering_b_baseline_contrast_bits",
    "steering_b_positive_contrast_bits",

]
