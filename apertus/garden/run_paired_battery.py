#!/usr/bin/env python3
"""Run the locked nomination-vs-description battery through the local API.

The runner deliberately uses urllib instead of requests/pandas and never calls
an archive mutation endpoint. The only non-stdlib import is the Rust-backed
``tokenizers.Tokenizer`` required to derive exact, contextual token IDs from
the model's tokenizer.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import ipaddress
import json
import math
import os
import platform
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


APP_DIR = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = APP_DIR / "paired_battery_protocol.json"
try:
    from .settings import cache_directory, data_directory
except ImportError:
    from settings import cache_directory, data_directory
DEFAULT_TOKENIZER = cache_directory() / "models/bf16/tokenizer.json"
DEFAULT_RAW_ROOT = data_directory() / "paired_batteries"
DEFAULT_OUTPUT_DIR = data_directory() / "results_nomination_description"
FORMAT = "apertus.nomination-description-battery"
SCHEMA_VERSION = 1
SERIES_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ALLOWED_POST_ENDPOINTS = frozenset(
    {"/api/next", "/api/lookup", "/api/compare", "/api/tuned-lens"}
)
FRAME_ORDER = ("active", "nominal")
PATH_ORDER = ("direct", "descriptor")


class BatteryError(RuntimeError):
    """A protocol, identity, API, or persistence invariant failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    atomic_write(path, pretty_json_bytes(value) if pretty else canonical_json_bytes(value))


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatteryError(f"Nelze načíst JSON {path}: {exc}") from exc


def ensure_loopback_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username:
        raise BatteryError("API musí být obyčejné http na loopback adrese.")
    hostname = parsed.hostname.rstrip(".").casefold()
    try:
        is_loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        is_loopback = hostname == "localhost" or hostname.endswith(".localhost")
    if not is_loopback:
        raise BatteryError("Runner z bezpečnostních důvodů volá pouze loopback API.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise BatteryError("--base-url nesmí obsahovat cestu, query ani fragment.")
    return base_url.rstrip("/")


def archive_identity(health: dict[str, Any]) -> dict[str, Any]:
    storage = health.get("storage")
    if not isinstance(storage, dict) or storage.get("status") != "ready":
        raise BatteryError("Lokální archiv není ve health odpovědi připraven.")
    archive_id = storage.get("archive_id")
    count = storage.get("record_count", storage.get("count"))
    if not isinstance(archive_id, str) or type(count) is not int:
        raise BatteryError("Health neobsahuje platnou identitu a počet archivu.")
    return {
        "archive_id": archive_id,
        "record_count": count,
        "path_label": storage.get("path_label"),
        "schema_version": storage.get("schema_version"),
    }


def health_identity(health: dict[str, Any]) -> dict[str, Any]:
    active_model = health.get("active_model")
    tuned = health.get("tuned_lens")
    if not isinstance(active_model, dict) or not isinstance(tuned, dict):
        raise BatteryError("Health neobsahuje identitu modelu nebo kalibrace.")
    calibration = tuned.get("calibration")
    if not isinstance(calibration, dict):
        raise BatteryError("Health neobsahuje aktivní tuned-lens kalibraci.")
    provenance = health.get("provenance", {})
    model_provenance = provenance.get("model", {}) if isinstance(provenance, dict) else {}
    model_calibration = calibration.get("model", {})
    weights = calibration.get("weights", {})
    return {
        "model_id": active_model.get("id"),
        "model_generation": health.get("model_generation"),
        "server_instance_id": health.get("server_instance_id"),
        "model_metadata_fingerprint": (
            model_provenance.get("metadata_fingerprint")
            if isinstance(model_provenance, dict)
            else None
        ),
        "calibration_id": calibration.get("artifact_id"),
        "calibration_manifest_sha256": calibration.get(
            "canonical_manifest_sha256"
        ),
        "calibration_weights_sha256": (
            weights.get("sha256") if isinstance(weights, dict) else None
        ),
        "calibration_model_id": (
            model_calibration.get("model_id")
            if isinstance(model_calibration, dict)
            else None
        ),
    }


def validate_health(
    health: dict[str, Any],
    *,
    required_model_id: str,
    expected_identity: dict[str, Any] | None = None,
    expected_archive: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if health.get("status") != "ready" or health.get("phase") is not None:
        raise BatteryError(
            f"Model není připraven: status={health.get('status')!r}, "
            f"phase={health.get('phase')!r}."
        )
    if health.get("mock") is not False:
        raise BatteryError("Měření odmítá mock server.")
    identity = health_identity(health)
    if identity["model_id"] != required_model_id:
        raise BatteryError(
            f"Je vyžadován model {required_model_id!r}, aktivní je "
            f"{identity['model_id']!r}."
        )
    tuned = health["tuned_lens"]
    calibration = tuned["calibration"]
    acceptance = calibration.get("validation", {}).get("acceptance", {})
    if tuned.get("status") != "ready" or acceptance.get("passed") is not True:
        raise BatteryError("Aktivní tuned-lens kalibrace není ready/accepted.")
    if identity["calibration_model_id"] != required_model_id:
        raise BatteryError("Tuned-lens artefakt nepatří aktivnímu hybridu.")
    required_identity_fields = (
        "model_id",
        "model_generation",
        "server_instance_id",
        "model_metadata_fingerprint",
        "calibration_id",
        "calibration_manifest_sha256",
        "calibration_weights_sha256",
    )
    if any(identity.get(name) in {None, ""} for name in required_identity_fields):
        raise BatteryError("Health vrací neúplnou modelovou identitu.")
    archive = archive_identity(health)
    if expected_identity is not None and identity != expected_identity:
        raise BatteryError("Model/server/kalibrace se během série změnily.")
    if expected_archive is not None and archive != expected_archive:
        raise BatteryError(
            "Identita nebo počet SQLite archivu se během série změnily; "
            "runner do archivu nezapisuje a z bezpečnostních důvodů končí."
        )
    return identity, archive


class LocalApi:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        identity: dict[str, Any],
    ) -> None:
        self.base_url = ensure_loopback_base_url(base_url)
        self.timeout = timeout
        self.identity = identity
        self.call_count = 0
        self.endpoint_counts: dict[str, int] = {}

    def get_health(self) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/api/health",
            headers={"Accept": "application/json"},
            method="GET",
        )
        return self._send(request, endpoint="/api/health")["response"]

    def post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        if endpoint not in ALLOWED_POST_ENDPOINTS:
            raise BatteryError(
                f"Endpoint {endpoint!r} není v neměnném read/inference allowlistu."
            )
        submitted = dict(payload)
        submitted.setdefault("model_generation", self.identity["model_generation"])
        submitted.setdefault(
            "server_instance_id", self.identity["server_instance_id"]
        )
        body = canonical_json_bytes(submitted)
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        record = self._send(request, endpoint=endpoint, submitted=submitted)
        self._validate_response_identity(record["response"], endpoint)
        return record

    def _send(
        self,
        request: urllib.request.Request,
        *,
        endpoint: str,
        submitted: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started_at = utc_now()
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = response.status
                data = response.read()
        except urllib.error.HTTPError as exc:
            data = exc.read()
            try:
                detail = json.loads(data)
            except json.JSONDecodeError:
                detail = data.decode("utf-8", errors="replace")
            raise BatteryError(
                f"{endpoint} vrátil HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BatteryError(f"{endpoint} selhal: {exc}") from exc
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise BatteryError(f"{endpoint} nevrátil platný JSON.") from exc
        if not isinstance(decoded, dict):
            raise BatteryError(f"{endpoint} nevrátil JSON objekt.")
        self.call_count += 1
        self.endpoint_counts[endpoint] = (
            self.endpoint_counts.get(endpoint, 0) + 1
        )
        return {
            "call_index": self.call_count,
            "started_at": started_at,
            "elapsed_client_ms": (time.perf_counter() - started) * 1000,
            "method": request.method,
            "endpoint": endpoint,
            "request": submitted,
            "http_status": status,
            "response": decoded,
        }

    def _validate_response_identity(
        self, response: dict[str, Any], endpoint: str
    ) -> None:
        for field in ("model_id", "model_generation", "server_instance_id"):
            if response.get(field) != self.identity[field]:
                raise BatteryError(
                    f"{endpoint}: odpověď změnila identitu {field}."
                )
        if endpoint == "/api/tuned-lens":
            calibration = response.get("calibration")
            if not isinstance(calibration, dict):
                raise BatteryError("RAW × TUNED odpověď nemá kalibraci.")
            expected = {
                "artifact_id": self.identity["calibration_id"],
                "canonical_manifest_sha256": self.identity[
                    "calibration_manifest_sha256"
                ],
            }
            for field, value in expected.items():
                if calibration.get(field) != value:
                    raise BatteryError(
                        f"RAW × TUNED odpověď změnila {field}."
                    )


def load_tokenizer(path: Path) -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise BatteryError(
            "Chybí balík tokenizers. Nainstaluj zvolený backend do stejného Python prostředí."
        ) from exc
    try:
        return Tokenizer.from_file(str(path))
    except Exception as exc:
        raise BatteryError(f"Tokenizer nelze načíst z {path}: {exc}") from exc


def validate_tokenizer_against_health(
    tokenizer_path: Path, tokenizer_sha: str, health: dict[str, Any]
) -> None:
    model = health.get("provenance", {}).get("model", {})
    hashes: list[str] = []
    if isinstance(model, dict):
        for source in model.get("sources", []):
            if not isinstance(source, dict):
                continue
            for item in source.get("metadata_files", []):
                if isinstance(item, dict) and item.get("name") == "tokenizer.json":
                    value = item.get("sha256")
                    if isinstance(value, str):
                        hashes.append(value)
    if not hashes or any(value != tokenizer_sha for value in hashes):
        raise BatteryError(
            f"Lokální {tokenizer_path.name} se neshoduje s aktivním modelem."
        )


def validate_protocol(protocol: dict[str, Any]) -> None:
    if (
        protocol.get("format") != "apertus.nomination-description-protocol"
        or protocol.get("schema_version") != 1
    ):
        raise BatteryError("Nepodporovaný formát protokolu.")
    design = protocol.get("design")
    pairs = protocol.get("pairs")
    if not isinstance(design, dict) or not isinstance(pairs, list) or not pairs:
        raise BatteryError("Protokol postrádá design nebo páry.")
    ids: set[str] = set()
    core_count = 0
    female_count = 0
    for pair in pairs:
        if not isinstance(pair, dict):
            raise BatteryError("Každý pár musí být JSON objekt.")
        pair_id = pair.get("pair_id")
        if (
            not isinstance(pair_id, str)
            or not SERIES_RE.fullmatch(pair_id)
            or pair_id in ids
        ):
            raise BatteryError(f"Neplatné nebo duplicitní pair_id: {pair_id!r}.")
        ids.add(pair_id)
        role = pair.get("role")
        core_count += role == "core"
        female_count += role == "female_replication"
        frames = pair.get("frames")
        paths = pair.get("paths")
        if not isinstance(frames, dict) or set(frames) != set(FRAME_ORDER):
            raise BatteryError(f"{pair_id}: rámce musí být active/nominal.")
        if not isinstance(paths, dict) or set(paths) != set(PATH_ORDER):
            raise BatteryError(f"{pair_id}: cesty musí být direct/descriptor.")
        for prompt in frames.values():
            if not isinstance(prompt, str) or not prompt.strip() or prompt != prompt.rstrip():
                raise BatteryError(f"{pair_id}: prefix nesmí mít koncovou mezeru.")
        for continuation in paths.values():
            if (
                not isinstance(continuation, str)
                or not continuation.startswith(" ")
                or continuation.startswith("  ")
                or continuation != continuation.rstrip()
            ):
                raise BatteryError(
                    f"{pair_id}: cesta musí začít jednou mezerou a nesmí končit mezerou."
                )
    if core_count != 6 or female_count != 1:
        raise BatteryError("Protokol musí mít 6 core párů a 1 ženskou sondu.")


def tokenize_pair(tokenizer: Any, pair: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pair_id": pair["pair_id"],
        "frames": {},
    }
    first_ids: dict[str, set[int]] = {kind: set() for kind in PATH_ORDER}
    suffixes: dict[str, list[list[int]]] = {kind: [] for kind in PATH_ORDER}
    for frame in FRAME_ORDER:
        prompt = pair["frames"][frame]
        root = tokenizer.encode(prompt, add_special_tokens=True)
        if not root.ids:
            raise BatteryError(f"{pair['pair_id']}/{frame}: prázdná tokenizace.")
        frame_data = {
            "prompt": prompt,
            "root_token_ids": list(root.ids),
            "root_tokens": list(root.tokens),
            "paths": {},
        }
        for kind in PATH_ORDER:
            continuation = pair["paths"][kind]
            full = tokenizer.encode(prompt + continuation, add_special_tokens=True)
            if full.ids[: len(root.ids)] != root.ids:
                raise BatteryError(
                    f"{pair['pair_id']}/{frame}/{kind}: kořen není tokenovým "
                    "prefixem předepsané cesty."
                )
            ids = list(full.ids[len(root.ids) :])
            tokens = list(full.tokens[len(root.ids) :])
            if not ids:
                raise BatteryError(
                    f"{pair['pair_id']}/{frame}/{kind}: cesta nemá tokeny."
                )
            decoded = tokenizer.decode(full.ids, skip_special_tokens=True)
            if decoded != prompt + continuation:
                raise BatteryError(
                    f"{pair['pair_id']}/{frame}/{kind}: tokenizer cestu "
                    "nedekóduje bezeztrátově."
                )
            first_ids[kind].add(ids[0])
            suffixes[kind].append(ids)
            frame_data["paths"][kind] = {
                "continuation": continuation,
                "token_ids": ids,
                "tokens": tokens,
                "first_token_id": ids[0],
            }
        if (
            frame_data["paths"]["direct"]["first_token_id"]
            == frame_data["paths"]["descriptor"]["first_token_id"]
        ):
            raise BatteryError(
                f"{pair['pair_id']}/{frame}: direct a descriptor nezačínají "
                "různými token_id."
            )
        result["frames"][frame] = frame_data
    for kind in PATH_ORDER:
        if len(first_ids[kind]) != 1 or suffixes[kind][0] != suffixes[kind][1]:
            raise BatteryError(
                f"{pair['pair_id']}: {kind} nemá stejnou tokenovou cestu "
                "pod oběma rámci."
            )
    return result


def choose_pairs(
    protocol: dict[str, Any],
    pair_ids: list[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    all_pairs = protocol["pairs"]
    by_id = {pair["pair_id"]: pair for pair in all_pairs}
    if pair_ids:
        unknown = [pair_id for pair_id in pair_ids if pair_id not in by_id]
        if unknown:
            raise BatteryError(f"Neznámé pair_id: {', '.join(unknown)}.")
        wanted = set(pair_ids)
        selected = [pair for pair in all_pairs if pair["pair_id"] in wanted]
    else:
        selected = list(all_pairs)
    if limit is not None:
        if limit < 1:
            raise BatteryError("--limit musí být kladný.")
        selected = selected[:limit]
    if not selected:
        raise BatteryError("Výběr neobsahuje žádný pár.")
    return selected


def response(call: dict[str, Any]) -> dict[str, Any]:
    return call["response"]


def lookup_map(call: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(item["token_id"]): item
        for item in response(call).get("results", [])
    }


def distribution_metrics(next_response: dict[str, Any]) -> dict[str, Any]:
    raw = next_response["raw_metrics"]
    candidates = next_response.get("candidates", [])
    top1 = candidates[0] if candidates else {}
    top2 = candidates[1] if len(candidates) > 1 else {}
    top1_probability = float(raw["top1_probability"])
    top2_probability = (
        float(raw["top2_probability"])
        if raw.get("top2_probability") is not None
        else None
    )
    top5_mass = sum(
        float(candidate["raw_probability"]) for candidate in candidates[:5]
    )
    return {
        "entropy_bits": float(raw["entropy_bits"]),
        "effective_tokens": float(raw["effective_tokens"]),
        "normalized_entropy": float(raw["normalized_entropy"]),
        "top5_mass": top5_mass,
        "tail_top5_probability": 1.0 - top5_mass,
        "top12_mass": float(next_response["top_k_mass"]),
        "tail_top12_probability": float(next_response["other_probability"]),
        "top1_token_id": raw["top1_token_id"],
        "top1_display": top1.get("display"),
        "top1_probability": top1_probability,
        "top2_token_id": raw.get("top2_token_id"),
        "top2_display": top2.get("display"),
        "top2_probability": top2_probability,
        "top1_top2_probability_gap": (
            top1_probability - top2_probability
            if top2_probability is not None
            else None
        ),
        "top1_top2_gap_bits": raw.get("top1_top2_gap_bits"),
        "top1_top2_tie_at_1e_12": (
            math.isclose(
                top1_probability,
                top2_probability,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            if top2_probability is not None
            else False
        ),
    }


def telemetry_row(
    next_response: dict[str, Any],
    candidate: dict[str, Any],
    *,
    step: int,
    cumulative_surprisal: float | None,
) -> tuple[dict[str, Any], float | None]:
    probability = float(candidate["raw_probability"])
    surprisal = -math.log2(probability) if probability > 0.0 else None
    cumulative = (
        None
        if surprisal is None or cumulative_surprisal is None
        else cumulative_surprisal + surprisal
    )
    metrics = distribution_metrics(next_response)
    item = {
        "step": step,
        "before_text": next_response["text"],
        "before_token_ids": next_response["token_ids"],
        "token_id": int(candidate["token_id"]),
        "token_text": candidate.get("text"),
        "token_display": candidate.get("display"),
        "after_text": candidate.get("next_text"),
        "after_token_ids": candidate.get("next_token_ids"),
        "probability": probability,
        "rank": int(candidate["rank"]),
        "surprisal_bits": surprisal,
        "cumulative_surprisal_bits": cumulative,
        "mean_surprisal_bits": (
            cumulative / step if cumulative is not None else None
        ),
        "in_top12": any(
            int(item["token_id"]) == int(candidate["token_id"])
            for item in next_response.get("candidates", [])
        ),
        **metrics,
    }
    return item, cumulative


def make_next_payload(
    text: str, token_ids: list[int], design: dict[str, Any]
) -> dict[str, Any]:
    return {
        "text": text,
        "token_ids": token_ids,
        "temperature": design["temperature"],
        "top_k": design["path_top_k"],
    }


def run_path(
    api: LocalApi,
    *,
    frame_tokenization: dict[str, Any],
    kind: str,
    root_next_call: dict[str, Any],
    root_lookup_call: dict[str, Any],
    design: dict[str, Any],
) -> dict[str, Any]:
    target_ids = frame_tokenization["paths"][kind]["token_ids"]
    root_next = response(root_next_call)
    first = lookup_map(root_lookup_call).get(target_ids[0])
    if first is None:
        raise BatteryError(f"Root lookup nevrátil cílový token {target_ids[0]}.")
    telemetry: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    item, cumulative = telemetry_row(
        root_next, first, step=1, cumulative_surprisal=0.0
    )
    telemetry.append(item)
    current_text = first["next_text"]
    current_ids = list(first["next_token_ids"])
    for step, token_id in enumerate(target_ids[1:], start=2):
        next_call = api.post(
            "/api/next", make_next_payload(current_text, current_ids, design)
        )
        lookup_call = api.post(
            "/api/lookup",
            {
                "text": current_text,
                "token_ids": current_ids,
                "candidate_token_ids": [token_id],
                "temperature": design["temperature"],
            },
        )
        selected = lookup_map(lookup_call).get(token_id)
        if selected is None:
            raise BatteryError(f"Path lookup nevrátil cílový token {token_id}.")
        item, cumulative = telemetry_row(
            response(next_call),
            selected,
            step=step,
            cumulative_surprisal=cumulative,
        )
        telemetry.append(item)
        calls.append({"next": next_call, "lookup": lookup_call})
        current_text = selected["next_text"]
        current_ids = list(selected["next_token_ids"])
    expected = (
        frame_tokenization["prompt"]
        + frame_tokenization["paths"][kind]["continuation"]
    )
    if current_text != expected:
        raise BatteryError(
            f"Předepsaná cesta skončila textem {current_text!r}, očekáváno {expected!r}."
        )
    return {
        "kind": kind,
        "continuation": frame_tokenization["paths"][kind]["continuation"],
        "target_token_ids": target_ids,
        "telemetry": telemetry,
        "additional_calls": calls,
        "final_text": current_text,
        "final_token_ids": current_ids,
        "total_surprisal_bits": cumulative,
        "mean_surprisal_bits": (
            cumulative / len(target_ids) if cumulative is not None else None
        ),
    }


def run_pair(
    api: LocalApi,
    *,
    pair: dict[str, Any],
    tokenization: dict[str, Any],
    design: dict[str, Any],
    expected_identity: dict[str, Any],
    expected_archive: dict[str, Any],
    index: int,
    total: int,
) -> dict[str, Any]:
    pair_id = pair["pair_id"]
    print(f"[{index:02d}/{total:02d}] {pair_id}: kořeny", flush=True)
    roots: dict[str, Any] = {}
    for frame in FRAME_ORDER:
        frame_tok = tokenization["frames"][frame]
        next_call = api.post(
            "/api/next",
            {
                "text": frame_tok["prompt"],
                "temperature": design["temperature"],
                "top_k": design["root_top_k"],
            },
        )
        root_response = response(next_call)
        if root_response.get("token_ids") != frame_tok["root_token_ids"]:
            raise BatteryError(
                f"{pair_id}/{frame}: server a tokenizers se neshodují na kořeni."
            )
        target_ids = [
            frame_tok["paths"][kind]["first_token_id"] for kind in PATH_ORDER
        ]
        lookup_call = api.post(
            "/api/lookup",
            {
                "text": frame_tok["prompt"],
                "token_ids": root_response["token_ids"],
                "candidate_token_ids": target_ids,
                "temperature": design["temperature"],
            },
        )
        if set(lookup_map(lookup_call)) != set(target_ids):
            raise BatteryError(f"{pair_id}/{frame}: neúplný targeted root lookup.")
        roots[frame] = {"next": next_call, "target_lookup": lookup_call}

    print(f"[{index:02d}/{total:02d}] {pair_id}: kontrast", flush=True)
    compare_call = api.post(
        "/api/compare",
        {
            "left": {
                "text": tokenization["frames"]["active"]["prompt"],
                "token_ids": response(roots["active"]["next"])["token_ids"],
            },
            "right": {
                "text": tokenization["frames"]["nominal"]["prompt"],
                "token_ids": response(roots["nominal"]["next"])["token_ids"],
            },
            "temperature": design["temperature"],
            "mover_count": design["compare_mover_count"],
            "overlap_k": design["compare_overlap_k"],
        },
    )

    paths: dict[str, Any] = {}
    for frame in FRAME_ORDER:
        paths[frame] = {}
        for kind in PATH_ORDER:
            print(
                f"[{index:02d}/{total:02d}] {pair_id}: {frame}/{kind}",
                flush=True,
            )
            paths[frame][kind] = run_path(
                api,
                frame_tokenization=tokenization["frames"][frame],
                kind=kind,
                root_next_call=roots[frame]["next"],
                root_lookup_call=roots[frame]["target_lookup"],
                design=design,
            )

    print(f"[{index:02d}/{total:02d}] {pair_id}: RAW × TUNED", flush=True)
    tuned: dict[str, Any] = {}
    for frame in FRAME_ORDER:
        tuned[frame] = api.post(
            "/api/tuned-lens",
            {
                "text": tokenization["frames"][frame]["prompt"],
                "top_k": design["tuned_lens_top_k"],
                "calibration_id": expected_identity["calibration_id"],
                "model_id": expected_identity["model_id"],
            },
        )
    health_after = api.get_health()
    validate_health(
        health_after,
        required_model_id=design["required_model_id"],
        expected_identity=expected_identity,
        expected_archive=expected_archive,
    )
    return {
        "format": f"{FORMAT}.pair",
        "schema_version": SCHEMA_VERSION,
        "completed_at": utc_now(),
        "pair": pair,
        "tokenization": tokenization,
        "roots": roots,
        "comparison": compare_call,
        "paths": paths,
        "tuned_lens": tuned,
        "health_after_pair": health_after,
    }


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    atomic_write(path, output.getvalue().encode("utf-8"))


def flatten_pair_results(
    series_id: str, pair_results: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    for bundle in pair_results:
        pair = bundle["pair"]
        pair_id = pair["pair_id"]
        root_targets: dict[str, dict[str, dict[str, Any]]] = {}
        for frame in FRAME_ORDER:
            next_response = response(bundle["roots"][frame]["next"])
            metrics = distribution_metrics(next_response)
            target_map = lookup_map(bundle["roots"][frame]["target_lookup"])
            targets: dict[str, dict[str, Any]] = {}
            for kind in PATH_ORDER:
                token_id = bundle["tokenization"]["frames"][frame]["paths"][kind][
                    "first_token_id"
                ]
                targets[kind] = target_map[token_id]
            root_targets[frame] = targets
            observations.append(
                {
                    "series_id": series_id,
                    "pair_id": pair_id,
                    "pair_role": pair["role"],
                    "work": pair["work"],
                    "expected_person": pair["expected_person"],
                    "frame": frame,
                    "prompt": pair["frames"][frame],
                    "prompt_token_count": next_response["token_count"],
                    **metrics,
                    "direct_first_token_id": targets["direct"]["token_id"],
                    "direct_first_display": targets["direct"]["display"],
                    "direct_first_probability": targets["direct"]["raw_probability"],
                    "direct_first_rank": targets["direct"]["rank"],
                    "direct_first_surprisal_bits": (
                        -math.log2(targets["direct"]["raw_probability"])
                        if targets["direct"]["raw_probability"] > 0
                        else None
                    ),
                    "descriptor_first_token_id": targets["descriptor"]["token_id"],
                    "descriptor_first_display": targets["descriptor"]["display"],
                    "descriptor_first_probability": targets["descriptor"][
                        "raw_probability"
                    ],
                    "descriptor_first_rank": targets["descriptor"]["rank"],
                    "descriptor_first_surprisal_bits": (
                        -math.log2(targets["descriptor"]["raw_probability"])
                        if targets["descriptor"]["raw_probability"] > 0
                        else None
                    ),
                    "latency_ms": next_response["latency_ms"],
                    "cached": next_response["cached"],
                }
            )
            for kind in PATH_ORDER:
                path = bundle["paths"][frame][kind]
                for item in path["telemetry"]:
                    paths.append(
                        {
                            "series_id": series_id,
                            "pair_id": pair_id,
                            "pair_role": pair["role"],
                            "frame": frame,
                            "path_kind": kind,
                            "prompt": pair["frames"][frame],
                            "continuation": path["continuation"],
                            **item,
                        }
                    )
            tuned_response = response(bundle["tuned_lens"][frame])
            for layer in tuned_response["layers"]:
                raw = layer["raw"]
                tuned = layer["tuned"]
                comparison = layer["comparison"]
                layers.append(
                    {
                        "series_id": series_id,
                        "pair_id": pair_id,
                        "pair_role": pair["role"],
                        "frame": frame,
                        "prompt": pair["frames"][frame],
                        "layer": layer["layer"],
                        "final_top1_token_id": tuned_response["final_top1"][
                            "token_id"
                        ],
                        "final_top1_display": tuned_response["final_top1"]["display"],
                        "final_top1_probability": tuned_response["final_top1"][
                            "probability"
                        ],
                        "raw_top1_token_id": raw["top1_token_id"],
                        "raw_top1_display": raw["candidates"][0]["display"],
                        "raw_top1_probability": raw["candidates"][0]["probability"],
                        "raw_kl_to_final_bits": raw[
                            "kl_divergence_to_final_bits"
                        ],
                        "raw_js_to_final_bits": raw[
                            "js_divergence_to_final_bits"
                        ],
                        "raw_final_top1_rank": raw["final_top1_rank"],
                        "raw_final_top1_probability": raw[
                            "final_top1_probability"
                        ],
                        "raw_top1_matches_final": raw["top1_matches_final"],
                        "raw_final_top1_in_top8": any(
                            candidate["token_id"]
                            == tuned_response["final_top1"]["token_id"]
                            for candidate in raw["candidates"]
                        ),
                        "tuned_top1_token_id": tuned["top1_token_id"],
                        "tuned_top1_display": tuned["candidates"][0]["display"],
                        "tuned_top1_probability": tuned["candidates"][0][
                            "probability"
                        ],
                        "tuned_kl_to_final_bits": tuned[
                            "kl_divergence_to_final_bits"
                        ],
                        "tuned_js_to_final_bits": tuned[
                            "js_divergence_to_final_bits"
                        ],
                        "tuned_final_top1_rank": tuned["final_top1_rank"],
                        "tuned_final_top1_probability": tuned[
                            "final_top1_probability"
                        ],
                        "tuned_top1_matches_final": tuned[
                            "top1_matches_final"
                        ],
                        "tuned_final_top1_in_top8": any(
                            candidate["token_id"]
                            == tuned_response["final_top1"]["token_id"]
                            for candidate in tuned["candidates"]
                        ),
                        "kl_improvement_bits": comparison["kl_improvement_bits"],
                        "tuned_is_closer_by_kl": comparison[
                            "tuned_is_closer_by_kl"
                        ],
                        "calibration_id": tuned_response["calibration"][
                            "artifact_id"
                        ],
                    }
                )
        compare = response(bundle["comparison"])
        active_paths = bundle["paths"]["active"]
        nominal_paths = bundle["paths"]["nominal"]
        root_surprisal: dict[str, dict[str, float]] = {}
        for frame in FRAME_ORDER:
            root_surprisal[frame] = {}
            for kind in PATH_ORDER:
                probability = float(
                    root_targets[frame][kind]["raw_probability"]
                )
                root_surprisal[frame][kind] = -math.log2(probability)

        path_parts: dict[str, dict[str, dict[str, Any]]] = {}
        for frame in FRAME_ORDER:
            path_parts[frame] = {}
            for kind in PATH_ORDER:
                telemetry = active_paths[kind]["telemetry"]
                if frame == "nominal":
                    telemetry = nominal_paths[kind]["telemetry"]
                if not telemetry or telemetry[-1].get("token_text") != ".":
                    raise BatteryError(
                        f"{pair_id}/{frame}/{kind}: protokol očekává "
                        "terminální tečku jako samostatný poslední token."
                    )
                lexical_steps = telemetry[:-1]
                lexical_surprisal = sum(
                    float(item["surprisal_bits"]) for item in lexical_steps
                )
                path_parts[frame][kind] = {
                    "token_count": len(telemetry),
                    "lexical_token_count": len(lexical_steps),
                    "lexical_surprisal_bits": lexical_surprisal,
                    "lexical_mean_surprisal_bits": (
                        lexical_surprisal / len(lexical_steps)
                        if lexical_steps
                        else None
                    ),
                    "terminal_surprisal_bits": float(
                        telemetry[-1]["surprisal_bits"]
                    ),
                }

        delta_direct_first_surprisal = (
            root_surprisal["nominal"]["direct"]
            - root_surprisal["active"]["direct"]
        )
        delta_descriptor_first_surprisal = (
            root_surprisal["nominal"]["descriptor"]
            - root_surprisal["active"]["descriptor"]
        )
        delta_direct_total_surprisal = (
            nominal_paths["direct"]["total_surprisal_bits"]
            - active_paths["direct"]["total_surprisal_bits"]
        )
        delta_descriptor_total_surprisal = (
            nominal_paths["descriptor"]["total_surprisal_bits"]
            - active_paths["descriptor"]["total_surprisal_bits"]
        )
        delta_direct_lexical_surprisal = (
            path_parts["nominal"]["direct"]["lexical_surprisal_bits"]
            - path_parts["active"]["direct"]["lexical_surprisal_bits"]
        )
        delta_descriptor_lexical_surprisal = (
            path_parts["nominal"]["descriptor"]["lexical_surprisal_bits"]
            - path_parts["active"]["descriptor"]["lexical_surprisal_bits"]
        )
        delta_direct_lexical_mean_surprisal = (
            path_parts["nominal"]["direct"]["lexical_mean_surprisal_bits"]
            - path_parts["active"]["direct"]["lexical_mean_surprisal_bits"]
        )
        delta_descriptor_lexical_mean_surprisal = (
            path_parts["nominal"]["descriptor"][
                "lexical_mean_surprisal_bits"
            ]
            - path_parts["active"]["descriptor"][
                "lexical_mean_surprisal_bits"
            ]
        )
        contrasts.append(
            {
                "series_id": series_id,
                "pair_id": pair_id,
                "pair_role": pair["role"],
                "work": pair["work"],
                "expected_person": pair["expected_person"],
                "active_prompt": pair["frames"]["active"],
                "nominal_prompt": pair["frames"]["nominal"],
                "entropy_active_bits": compare["left"]["entropy_bits"],
                "entropy_nominal_bits": compare["right"]["entropy_bits"],
                "delta_entropy_nominal_minus_active_bits": compare[
                    "delta_entropy_bits"
                ],
                "effective_support_ratio_nominal_over_active": compare[
                    "effective_support_ratio"
                ],
                "js_divergence_bits": compare["js_divergence_bits"],
                "total_variation": compare["total_variation"],
                "top12_overlap_count": compare["top_k_overlap"]["count"],
                "top12_overlap_fraction": compare["top_k_overlap"]["fraction"],
                "direct_first_token_probability_active": root_targets["active"]["direct"][
                    "raw_probability"
                ],
                "direct_first_token_rank_active": root_targets["active"][
                    "direct"
                ]["rank"],
                "direct_first_token_probability_nominal": root_targets["nominal"]["direct"][
                    "raw_probability"
                ],
                "direct_first_token_rank_nominal": root_targets["nominal"][
                    "direct"
                ]["rank"],
                "direct_first_token_surprisal_active_bits": root_surprisal[
                    "active"
                ]["direct"],
                "direct_first_token_surprisal_nominal_bits": root_surprisal[
                    "nominal"
                ]["direct"],
                "delta_direct_first_token_surprisal_nominal_minus_active_bits": (
                    delta_direct_first_surprisal
                ),
                "delta_direct_first_token_probability_nominal_minus_active": (
                    root_targets["nominal"]["direct"]["raw_probability"]
                    - root_targets["active"]["direct"]["raw_probability"]
                ),
                "descriptor_first_token_probability_active": root_targets["active"][
                    "descriptor"
                ]["raw_probability"],
                "descriptor_first_token_rank_active": root_targets["active"][
                    "descriptor"
                ]["rank"],
                "descriptor_first_token_probability_nominal": root_targets["nominal"][
                    "descriptor"
                ]["raw_probability"],
                "descriptor_first_token_rank_nominal": root_targets["nominal"][
                    "descriptor"
                ]["rank"],
                "descriptor_first_token_surprisal_active_bits": root_surprisal[
                    "active"
                ]["descriptor"],
                "descriptor_first_token_surprisal_nominal_bits": root_surprisal[
                    "nominal"
                ]["descriptor"],
                "delta_descriptor_first_token_surprisal_nominal_minus_active_bits": (
                    delta_descriptor_first_surprisal
                ),
                "delta_descriptor_first_token_probability_nominal_minus_active": (
                    root_targets["nominal"]["descriptor"]["raw_probability"]
                    - root_targets["active"]["descriptor"]["raw_probability"]
                ),
                "root_descriptor_relative_log2_shift_nominal_vs_active": (
                    delta_direct_first_surprisal
                    - delta_descriptor_first_surprisal
                ),
                "direct_total_surprisal_active_bits": active_paths["direct"][
                    "total_surprisal_bits"
                ],
                "direct_total_surprisal_nominal_bits": nominal_paths["direct"][
                    "total_surprisal_bits"
                ],
                "delta_direct_surprisal_nominal_minus_active_bits": (
                    delta_direct_total_surprisal
                ),
                "descriptor_total_surprisal_active_bits": active_paths[
                    "descriptor"
                ]["total_surprisal_bits"],
                "descriptor_total_surprisal_nominal_bits": nominal_paths[
                    "descriptor"
                ]["total_surprisal_bits"],
                "delta_descriptor_surprisal_nominal_minus_active_bits": (
                    delta_descriptor_total_surprisal
                ),
                "full_path_descriptor_relative_log2_shift_nominal_vs_active": (
                    delta_direct_total_surprisal
                    - delta_descriptor_total_surprisal
                ),
                "direct_lexical_token_count": path_parts["active"]["direct"][
                    "lexical_token_count"
                ],
                "direct_lexical_surprisal_active_bits": path_parts["active"][
                    "direct"
                ]["lexical_surprisal_bits"],
                "direct_lexical_surprisal_nominal_bits": path_parts["nominal"][
                    "direct"
                ]["lexical_surprisal_bits"],
                "delta_direct_lexical_surprisal_nominal_minus_active_bits": (
                    delta_direct_lexical_surprisal
                ),
                "descriptor_lexical_token_count": path_parts["active"][
                    "descriptor"
                ]["lexical_token_count"],
                "descriptor_lexical_surprisal_active_bits": path_parts["active"][
                    "descriptor"
                ]["lexical_surprisal_bits"],
                "descriptor_lexical_surprisal_nominal_bits": path_parts[
                    "nominal"
                ]["descriptor"]["lexical_surprisal_bits"],
                "delta_descriptor_lexical_surprisal_nominal_minus_active_bits": (
                    delta_descriptor_lexical_surprisal
                ),
                "lexical_descriptor_relative_log2_shift_nominal_vs_active": (
                    delta_direct_lexical_surprisal
                    - delta_descriptor_lexical_surprisal
                ),
                "delta_direct_lexical_mean_surprisal_nominal_minus_active_bits": (
                    delta_direct_lexical_mean_surprisal
                ),
                "delta_descriptor_lexical_mean_surprisal_nominal_minus_active_bits": (
                    delta_descriptor_lexical_mean_surprisal
                ),
                "lexical_mean_descriptor_relative_log2_shift_nominal_vs_active": (
                    delta_direct_lexical_mean_surprisal
                    - delta_descriptor_lexical_mean_surprisal
                ),
                "direct_terminal_period_surprisal_active_bits": path_parts[
                    "active"
                ]["direct"]["terminal_surprisal_bits"],
                "direct_terminal_period_surprisal_nominal_bits": path_parts[
                    "nominal"
                ]["direct"]["terminal_surprisal_bits"],
                "delta_direct_terminal_period_surprisal_nominal_minus_active_bits": (
                    path_parts["nominal"]["direct"]["terminal_surprisal_bits"]
                    - path_parts["active"]["direct"]["terminal_surprisal_bits"]
                ),
                "descriptor_terminal_period_surprisal_active_bits": path_parts[
                    "active"
                ]["descriptor"]["terminal_surprisal_bits"],
                "descriptor_terminal_period_surprisal_nominal_bits": path_parts[
                    "nominal"
                ]["descriptor"]["terminal_surprisal_bits"],
                "delta_descriptor_terminal_period_surprisal_nominal_minus_active_bits": (
                    path_parts["nominal"]["descriptor"][
                        "terminal_surprisal_bits"
                    ]
                    - path_parts["active"]["descriptor"][
                        "terminal_surprisal_bits"
                    ]
                ),
                "gains_json": json.dumps(
                    compare["gains"], ensure_ascii=False, separators=(",", ":")
                ),
                "losses_json": json.dumps(
                    compare["losses"], ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
    return observations, contrasts, paths, layers


def numeric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def summarize_contrast_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(field: str) -> list[float]:
        return [
            float(row[field])
            for row in rows
            if row.get(field) is not None
        ]

    direct_root_probability = values(
        "delta_direct_first_token_probability_nominal_minus_active"
    )
    descriptor_root_probability = values(
        "delta_descriptor_first_token_probability_nominal_minus_active"
    )
    root_relative_shift = values(
        "root_descriptor_relative_log2_shift_nominal_vs_active"
    )
    direct_lexical = values(
        "delta_direct_lexical_surprisal_nominal_minus_active_bits"
    )
    descriptor_lexical = values(
        "delta_descriptor_lexical_surprisal_nominal_minus_active_bits"
    )
    lexical_relative_shift = values(
        "lexical_descriptor_relative_log2_shift_nominal_vs_active"
    )
    full_relative_shift = values(
        "full_path_descriptor_relative_log2_shift_nominal_vs_active"
    )
    direct_period = values(
        "delta_direct_terminal_period_surprisal_nominal_minus_active_bits"
    )
    descriptor_period = values(
        "delta_descriptor_terminal_period_surprisal_nominal_minus_active_bits"
    )
    entropy = values("delta_entropy_nominal_minus_active_bits")
    return {
        "n": len(rows),
        "pair_ids": [row["pair_id"] for row in rows],
        "root": {
            "delta_entropy_nominal_minus_active_bits": numeric_summary(
                entropy
            ),
            "effective_support_ratio_nominal_over_active": numeric_summary(
                values("effective_support_ratio_nominal_over_active")
            ),
            "geometric_mean_effective_support_ratio_nominal_over_active": (
                2.0 ** statistics.fmean(entropy) if entropy else None
            ),
            "js_divergence_bits": numeric_summary(
                values("js_divergence_bits")
            ),
            "total_variation": numeric_summary(
                values("total_variation")
            ),
            "top12_overlap_fraction": numeric_summary(
                values("top12_overlap_fraction")
            ),
            "delta_direct_first_token_probability_nominal_minus_active": (
                numeric_summary(direct_root_probability)
            ),
            "delta_descriptor_first_token_probability_nominal_minus_active": (
                numeric_summary(descriptor_root_probability)
            ),
            "descriptor_relative_log2_shift_nominal_vs_active": (
                numeric_summary(root_relative_shift)
            ),
            "nominal_widens_field": sum(value > 0 for value in entropy),
            "nominal_narrows_field": sum(value < 0 for value in entropy),
            "active_has_more_direct_first_token_proxy": sum(
                value < 0 for value in direct_root_probability
            ),
            "nominal_has_more_descriptor_first_token_proxy": sum(
                value > 0 for value in descriptor_root_probability
            ),
            "nominal_relatively_favors_descriptor_onset": sum(
                value > 0 for value in root_relative_shift
            ),
            "nominal_relatively_favors_direct_onset": sum(
                value < 0 for value in root_relative_shift
            ),
        },
        "lexical_paths_without_terminal_period": {
            "delta_direct_surprisal_nominal_minus_active_bits": (
                numeric_summary(direct_lexical)
            ),
            "delta_descriptor_surprisal_nominal_minus_active_bits": (
                numeric_summary(descriptor_lexical)
            ),
            "descriptor_relative_log2_shift_nominal_vs_active": (
                numeric_summary(lexical_relative_shift)
            ),
            "direct_path_easier_under_active": sum(
                value > 0 for value in direct_lexical
            ),
            "descriptor_path_easier_under_nominal": sum(
                value < 0 for value in descriptor_lexical
            ),
            "nominal_relatively_favors_descriptor_path": sum(
                value > 0 for value in lexical_relative_shift
            ),
            "nominal_relatively_favors_direct_path": sum(
                value < 0 for value in lexical_relative_shift
            ),
            "positive_shift_pair_ids": [
                row["pair_id"]
                for row in rows
                if float(
                    row[
                        "lexical_descriptor_relative_log2_shift_nominal_vs_active"
                    ]
                )
                > 0
            ],
            "negative_shift_pair_ids": [
                row["pair_id"]
                for row in rows
                if float(
                    row[
                        "lexical_descriptor_relative_log2_shift_nominal_vs_active"
                    ]
                )
                < 0
            ],
        },
        "sentence_closure": {
            "delta_direct_terminal_period_surprisal_nominal_minus_active_bits": (
                numeric_summary(direct_period)
            ),
            "delta_descriptor_terminal_period_surprisal_nominal_minus_active_bits": (
                numeric_summary(descriptor_period)
            ),
            "terminal_period_easier_under_nominal": sum(
                value < 0 for value in direct_period + descriptor_period
            ),
            "terminal_period_comparisons": (
                len(direct_period) + len(descriptor_period)
            ),
            "full_path_descriptor_relative_log2_shift_nominal_vs_active": (
                numeric_summary(full_relative_shift)
            ),
        },
    }


def summarize_layer_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    layer_summaries = []
    for layer, layer_rows in sorted(by_layer.items()):
        tuned_closer = sum(
            bool(row["tuned_is_closer_by_kl"]) for row in layer_rows
        )
        layer_summaries.append(
            {
                "layer": layer,
                "n": len(layer_rows),
                "raw_mean_kl_to_final_bits": statistics.fmean(
                    float(row["raw_kl_to_final_bits"])
                    for row in layer_rows
                ),
                "tuned_mean_kl_to_final_bits": statistics.fmean(
                    float(row["tuned_kl_to_final_bits"])
                    for row in layer_rows
                ),
                "tuned_closer_by_kl": tuned_closer,
                "tuned_top1_matches_final": sum(
                    bool(row["tuned_top1_matches_final"])
                    for row in layer_rows
                ),
                "tuned_final_top1_in_top8": sum(
                    bool(row["tuned_final_top1_in_top8"])
                    for row in layer_rows
                ),
            }
        )
    return {
        "row_count": len(rows),
        "layers": layer_summaries,
    }


def build_summary(
    series_id: str,
    selected_pairs: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    core_rows = [
        row for row in contrasts if row["pair_role"] == "core"
    ]
    female_rows = [
        row
        for row in contrasts
        if row["pair_role"] == "female_replication"
    ]
    tuned_count = sum(bool(row["tuned_is_closer_by_kl"]) for row in layer_rows)
    core_layer_rows = [
        row for row in layer_rows if row["pair_role"] == "core"
    ]
    return {
        "format": f"{FORMAT}.summary",
        "schema_version": SCHEMA_VERSION,
        "series_id": series_id,
        "generated_at": utc_now(),
        "pair_count": len(selected_pairs),
        "core_pair_count": len(core_rows),
        "female_replication_count": len(female_rows),
        "pair_ids": [pair["pair_id"] for pair in selected_pairs],
        "groups": {
            "core": summarize_contrast_rows(core_rows),
            "female_replication": summarize_contrast_rows(female_rows),
            "all": summarize_contrast_rows(contrasts),
        },
        "tuned_lens": {
            "row_count": len(layer_rows),
            "tuned_closer_by_kl_count": tuned_count,
            "tuned_closer_by_kl_fraction": (
                tuned_count / len(layer_rows) if layer_rows else None
            ),
            "kl_improvement_bits": numeric_summary(
                [float(row["kl_improvement_bits"]) for row in layer_rows]
            ),
            "all": summarize_layer_rows(layer_rows),
            "core": summarize_layer_rows(core_layer_rows),
        },
        "interpretation_limits": [
            "Deskriptivní explorativní souhrn bez inferenčního testu.",
            "Pravděpodobnost tokenu není pravda, mínění ani jistota.",
            "Surprisal se srovnává jen pro tutéž cestu mezi rámci.",
            "Lexikální cesta se hlásí bez záměrné terminální tečky; uzavření věty se hlásí odděleně.",
            "Babička není součástí primárního šestipárového souhrnu.",
            "První-token proxy není masa celé sémantické třídy.",
        ],
    }


def build_report(summary: dict[str, Any], metadata: dict[str, Any]) -> str:
    core = summary["groups"]["core"]
    female = summary["groups"]["female_replication"]
    roots = core["root"]
    lexical = core["lexical_paths_without_terminal_period"]
    closure = core["sentence_closure"]
    female_roots = female["root"]
    positive_ids = ", ".join(lexical["positive_shift_pair_ids"])
    negative_ids = ", ".join(lexical["negative_shift_pair_ids"])
    core_l31 = next(
        (
            item
            for item in summary["tuned_lens"]["core"]["layers"]
            if item["layer"] == 31
        ),
        {
            "n": 0,
            "tuned_final_top1_in_top8": 0,
            "tuned_top1_matches_final": 0,
        },
    )
    return f"""# Přímá nominace vs. kategoriální popis

Série `{summary['series_id']}` obsahuje {summary['pair_count']} předem
uzamčených párů ({summary['core_pair_count']} core a
{summary['female_replication_count']} ženskou sondu).
Měřeno na režimu `{metadata['model_identity']['model_id']}` při T = 1;
kořeny i kroky cest používají top 12 a RAW × TUNED top 8.

## Co bylo měřeno

U každého páru se porovnává aktivní rámec s jmenným autorským rámcem.
Zaznamenána je úplná kořenová telemetrie, cílené pořadí prvního tokenu jména
a popisu, JS/TV kontrast, krokový surprisal dvou totožných předepsaných cest
a sedm řezů RAW × TUNED.

## Hlavní šestipárový výsledek

- Šířka pole se neposunula jednotným směrem: jmenný rámec pole ve 3 z 6
  případů rozšířil a ve 3 z 6 zúžil. Medián ΔH (jmenný − aktivní) je
  {roots['delta_entropy_nominal_minus_active_bits']['median']:.3f} bitu.
- Přesto se obsah pole výrazně přeskupil: průměrná total variation je
  {roots['total_variation']['mean']:.3f} a průměrný překryv top 12 jen
  {100 * roots['top12_overlap_fraction']['mean']:.1f} %.
- Odvozená explorativní log-odds interakce mezi přesně předepsaným prvním
  tokenem popisu a prvním tokenem přímého jména se pod jmenným rámcem
  posunula k popisu ve
  {roots['nominal_relatively_favors_descriptor_onset']} z 6 případů.
- V post-hoc citlivostním řezu po odečtení závěrečné tečky se stejný
  relativní posun objevil ve
  {lexical['nominal_relatively_favors_descriptor_path']} z 6 lexikálních
  cest. Kladné případy: {positive_ids}. Záporné případy: {negative_ids}.
- Obě záporné core položky jsou obrazy (`guernica`, `vykrik`). To je nápadný
  post-hoc vzorec, ale se dvěma položkami není důkazem žánrového efektu.

## Tečka není detail

Terminální tečka byla předem a záměrně součástí protokolu. Předem určený
whole-string výsledek proto zůstává zachovaný. Rozklad po krocích ale ukazuje,
že pod jmenným rámcem měla tečka
nižší surprisal ve {closure['terminal_period_easier_under_nominal']} ze
{closure['terminal_period_comparisons']} core srovnání. Souhrn celé věty by
proto hlavně měřil snadnost uzavření konstrukce `Autorem … je … .`, nikoli
jen volbu mezi jménem a popisem. Lexikální řez bez poslední tečky je
transparentně označený post-hoc sensitivity check; uzavření věty je
samostatný nález.

## Ženská sonda

`Knihu Babička napsala` proti `Autorkou knihy Babička je` je odlehlé
pozorování: ΔH je
{female_roots['delta_entropy_nominal_minus_active_bits']['mean']:.3f} bitu
a efektivní šířka se změnila přibližně
{female_roots['effective_support_ratio_nominal_over_active']['mean']:.1f}×.
Položka se proto nepočítá do primárního souhrnu. Raw protokol ji z
auditních důvodů označuje jako `female_replication`, empiricky je to však
jediná ženská sonda n = 1, nikoli potvrzená replikace. Může zachycovat
řídkost nebo nezvyklost konkrétního rámce, ne obecný genderový efekt.

## RAW × TUNED

TUNED čtečka měla nižší KL k finální distribuci ve
{summary['tuned_lens']['tuned_closer_by_kl_count']} z
{summary['tuned_lens']['row_count']} zobrazených řezů. To je technická
vlastnost kalibrované čtečky, nikoli nezávislý důkaz interpretace syntaxe.
V L31 byl finální top-1 mezi osmi zobrazenými TUNED kandidáty ve
{core_l31['tuned_final_top1_in_top8']} z {core_l31['n']} core rámců a přímo
top-1 v {core_l31['tuned_top1_matches_final']} z {core_l31['n']}. Přesný
povrchový výběr se tedy většinou ustaluje velmi pozdě.

## Jak to číst

Jde o explorativní popis malé, záměrně vybrané baterie. Výsledek ukazuje, že
konstrukční rámec dokáže přeskládat pole bez jednotné změny jeho šířky; sám
o sobě ale nedokazuje obecný zákon češtiny, pravdivost odpovědi ani „mínění“
modelu. Kořenové cílové hodnoty jsou pouze proxy prvního tokenu jména či
popisu, nikoli součet pravděpodobnosti všech možných jmen nebo popisů.
Například `nor`, `norsk` a `Norweg` jsou různé tokenové začátky téže lidsky
čtené kategorie.
Přímá a popisná cesta mají různou délku, proto se jejich celkové
pravděpodobnosti neporovnávají mezi sebou.

Raw checkpoint je v `{metadata['raw_bundle']}`. Odvozené CSV jsou
reprodukovatelné z neměnných pair JSONů tohoto bundle.
"""


def write_derived_outputs(
    *,
    output_dir: Path,
    series_id: str,
    selected_pairs: list[dict[str, Any]],
    pair_results: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    observations, contrasts, paths, layers = flatten_pair_results(
        series_id, pair_results
    )
    summary = build_summary(series_id, selected_pairs, contrasts, layers)
    write_csv(
        output_dir / "observations.csv",
        list(observations[0]) if observations else [],
        observations,
    )
    write_csv(
        output_dir / "pair_contrasts.csv",
        list(contrasts[0]) if contrasts else [],
        contrasts,
    )
    write_csv(output_dir / "paths.csv", list(paths[0]) if paths else [], paths)
    write_csv(
        output_dir / "layer_trajectories.csv",
        list(layers[0]) if layers else [],
        layers,
    )
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(output_dir / "run_metadata.json", metadata)
    atomic_write(
        output_dir / "report.md",
        build_report(summary, metadata).encode("utf-8"),
    )


def checkpoint_path(series_dir: Path) -> Path:
    return series_dir / "checkpoint.json"


def save_checkpoint(series_dir: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(checkpoint_path(series_dir), checkpoint)


def create_manifest(series_dir: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    inventory = []
    for path in sorted(series_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            inventory.append(
                {
                    "path": str(path.relative_to(series_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "format": f"{FORMAT}.manifest",
        "schema_version": SCHEMA_VERSION,
        "series_id": checkpoint["series_id"],
        "status": "complete",
        "completed_at": checkpoint["completed_at"],
        "selected_pair_ids": checkpoint["selected_pair_ids"],
        "files": inventory,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sekvenční baterie přímé nominace vs. popisu."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--series-id")
    parser.add_argument(
        "--pair-id",
        action="append",
        default=[],
        help="Opakovatelné pair_id; bez volby se měří všech sedm.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Jen ověří protokol a tokenizaci; nevolá API ani nezapisuje.",
    )
    parser.add_argument("--timeout", type=float, default=305.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = args.protocol.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    protocol = load_json(protocol_path)
    validate_protocol(protocol)
    protocol_sha = sha256_file(protocol_path)
    tokenizer_sha = sha256_file(tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    initially_selected = choose_pairs(protocol, args.pair_id, args.limit)
    tokenizations_all = {
        pair["pair_id"]: tokenize_pair(tokenizer, pair)
        for pair in protocol["pairs"]
    }
    if args.validate_only:
        selected_ids = [pair["pair_id"] for pair in initially_selected]
        token_count = sum(
            len(tokenizations_all[pair_id]["frames"][frame]["paths"][kind][
                "token_ids"
            ])
            for pair_id in selected_ids
            for frame in FRAME_ORDER
            for kind in PATH_ORDER
        )
        print(
            json.dumps(
                {
                    "status": "valid",
                    "protocol_id": protocol["protocol_id"],
                    "selected_pair_ids": selected_ids,
                    "prescribed_path_token_steps": token_count,
                    "protocol_sha256": protocol_sha,
                    "tokenizer_sha256": tokenizer_sha,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.timeout <= 0:
        raise BatteryError("--timeout musí být kladný.")
    if args.series_id is None:
        if args.resume:
            raise BatteryError("--resume vyžaduje explicitní --series-id.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        series_id = f"nomination-description-{stamp}-{protocol_sha[:8]}"
    else:
        series_id = args.series_id
    if not SERIES_RE.fullmatch(series_id):
        raise BatteryError("Neplatné --series-id.")

    raw_root = args.raw_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    series_dir = raw_root / series_id
    base_url = ensure_loopback_base_url(args.base_url)

    if args.resume:
        checkpoint = load_json(checkpoint_path(series_dir))
        if checkpoint.get("format") != FORMAT or checkpoint.get("series_id") != series_id:
            raise BatteryError("Resume checkpoint má jiný formát nebo series_id.")
        if checkpoint.get("protocol_sha256") != protocol_sha:
            raise BatteryError("Protokol se od zahájení série změnil.")
        if checkpoint.get("tokenizer_sha256") != tokenizer_sha:
            raise BatteryError("Tokenizer se od zahájení série změnil.")
        stored_ids = checkpoint.get("selected_pair_ids")
        if not isinstance(stored_ids, list):
            raise BatteryError("Checkpoint nemá výběr párů.")
        if args.pair_id or args.limit is not None:
            requested_ids = [pair["pair_id"] for pair in initially_selected]
            if requested_ids != stored_ids:
                raise BatteryError("Resume musí použít původní výběr párů.")
        selected_pairs = choose_pairs(protocol, stored_ids, None)
        initial_health = load_json(series_dir / "health_initial.json")
        expected_identity = checkpoint["model_identity"]
        expected_archive = checkpoint["archive_initial"]
        bootstrap_api = LocalApi(base_url, args.timeout, expected_identity)
        current_health = bootstrap_api.get_health()
        validate_health(
            current_health,
            required_model_id=protocol["design"]["required_model_id"],
            expected_identity=expected_identity,
            expected_archive=expected_archive,
        )
    else:
        if series_dir.exists():
            raise BatteryError(
                f"Série {series_dir} už existuje; použij --resume nebo jiné ID."
            )
        series_dir.mkdir(parents=True)
        selected_pairs = initially_selected
        temporary_identity = {
            "model_generation": None,
            "server_instance_id": None,
        }
        bootstrap_api = LocalApi(base_url, args.timeout, temporary_identity)
        initial_health = bootstrap_api.get_health()
        expected_identity, expected_archive = validate_health(
            initial_health,
            required_model_id=protocol["design"]["required_model_id"],
        )
        validate_tokenizer_against_health(
            tokenizer_path, tokenizer_sha, initial_health
        )
        checkpoint = {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "series_id": series_id,
            "status": "incomplete",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha,
            "tokenizer_sha256": tokenizer_sha,
            "tokenizer_path": str(tokenizer_path),
            "base_url": base_url,
            "model_identity": expected_identity,
            "archive_initial": expected_archive,
            "selected_pair_ids": [pair["pair_id"] for pair in selected_pairs],
            "completed_pairs": {},
            "last_error": None,
        }
        atomic_write_json(series_dir / "protocol.json", protocol)
        atomic_write_json(series_dir / "health_initial.json", initial_health)
        atomic_write_json(
            series_dir / "tokenization.json",
            {
                "tokenizer_path": str(tokenizer_path),
                "tokenizer_sha256": tokenizer_sha,
                "pairs": {
                    pair["pair_id"]: tokenizations_all[pair["pair_id"]]
                    for pair in selected_pairs
                },
            },
        )
        save_checkpoint(series_dir, checkpoint)

    api = LocalApi(base_url, args.timeout, expected_identity)
    completed = checkpoint["completed_pairs"]
    try:
        for index, pair in enumerate(selected_pairs, start=1):
            pair_id = pair["pair_id"]
            if pair_id in completed:
                pair_path = series_dir / completed[pair_id]["file"]
                if sha256_file(pair_path) != completed[pair_id]["sha256"]:
                    raise BatteryError(f"Checkpoint {pair_id} má poškozený hash.")
                print(
                    f"[{index:02d}/{len(selected_pairs):02d}] {pair_id}: checkpoint",
                    flush=True,
                )
                continue
            pair_result = run_pair(
                api,
                pair=pair,
                tokenization=tokenizations_all[pair_id],
                design=protocol["design"],
                expected_identity=expected_identity,
                expected_archive=expected_archive,
                index=index,
                total=len(selected_pairs),
            )
            pair_rel = Path("pairs") / f"{pair_id}.json"
            pair_path = series_dir / pair_rel
            atomic_write_json(pair_path, pair_result)
            completed[pair_id] = {
                "file": str(pair_rel),
                "sha256": sha256_file(pair_path),
                "completed_at": pair_result["completed_at"],
            }
            checkpoint["last_error"] = None
            save_checkpoint(series_dir, checkpoint)

        final_health = api.get_health()
        validate_health(
            final_health,
            required_model_id=protocol["design"]["required_model_id"],
            expected_identity=expected_identity,
            expected_archive=expected_archive,
        )
        atomic_write_json(series_dir / "health_final.json", final_health)
        checkpoint["status"] = "complete"
        checkpoint["completed_at"] = utc_now()
        checkpoint["archive_final"] = archive_identity(final_health)
        checkpoint["api_call_count_this_process"] = api.call_count
        checkpoint["api_endpoint_counts_this_process"] = dict(
            sorted(api.endpoint_counts.items())
        )
        checkpoint["last_error"] = None
        save_checkpoint(series_dir, checkpoint)
        manifest = create_manifest(series_dir, checkpoint)
        atomic_write_json(series_dir / "manifest.json", manifest)
    except Exception as exc:
        checkpoint["status"] = "incomplete"
        checkpoint["last_error"] = {
            "at": utc_now(),
            "type": type(exc).__name__,
            "message": str(exc),
        }
        save_checkpoint(series_dir, checkpoint)
        raise

    pair_results = [
        load_json(series_dir / checkpoint["completed_pairs"][pair["pair_id"]]["file"])
        for pair in selected_pairs
    ]
    runner_path = Path(__file__).resolve()
    metadata = {
        "format": f"{FORMAT}.run-metadata",
        "schema_version": SCHEMA_VERSION,
        "series_id": series_id,
        "started_at": checkpoint["started_at"],
        "completed_at": checkpoint["completed_at"],
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "runner_sha256": sha256_file(runner_path),
        "tokenizer_sha256": tokenizer_sha,
        "model_identity": expected_identity,
        "archive_initial": expected_archive,
        "archive_final": checkpoint["archive_final"],
        "archive_unchanged": expected_archive == checkpoint["archive_final"],
        "archive_mutation_endpoints_called": 0,
        "selected_pair_ids": checkpoint["selected_pair_ids"],
        "raw_bundle": str(series_dir),
        "raw_manifest_sha256": sha256_file(series_dir / "manifest.json"),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    write_derived_outputs(
        output_dir=output_dir,
        series_id=series_id,
        selected_pairs=selected_pairs,
        pair_results=pair_results,
        metadata=metadata,
    )
    print(f"Raw bundle: {series_dir}")
    print(f"Derived outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BatteryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
