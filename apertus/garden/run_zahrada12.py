#!/usr/bin/env python3
"""Run and evaluate the frozen Zahrada 12 surprisal battery.

The collector calls the same local, teacher-forced endpoint as the
``Překvapení textu`` page.  It never generates text, switches a model, or
writes into the Garden archive.  The frozen material is expanded to 120
unique full texts, each scored once from BOS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import ipaddress
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


APP_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = APP_DIR.parents[1]
DEFAULT_MATERIALS = REPOSITORY_ROOT / "experiment-zahrada12-materials.json"
DEFAULT_RAW_ROOT = APP_DIR / "data" / "zahrada12"
FROZEN_MATERIALS_SHA256 = (
    "d0b1ba9c7abae9c95defc443cb804757d8a26f71188a33b912dd07bbed9bf46a"
)
FORMAT = "apertus.zahrada12-normative-status-battery"
SCHEMA_VERSION = 2
REQUIRED_SERIES_ID = "zahrada12-normative-status-v1"
REQUIRED_MODEL_ID = "hybrid"
TOP_K = 5
BOOTSTRAP_SEED = 20260802
BOOTSTRAP_SAMPLES = 10_000
PREFIX_ORDER = (
    "normative_failure",
    "epistemic_failure",
    "normative_success",
    "epistemic_success",
)
CANDIDATE_ORDER = (
    "normative_failure",
    "normative_success",
    "epistemic_failure",
    "epistemic_success",
)
SCORING_MATRIX = {
    "normative_failure": (
        "normative_failure",
        "epistemic_failure",
        "normative_success",
    ),
    "epistemic_failure": (
        "normative_failure",
        "epistemic_failure",
        "epistemic_success",
    ),
    "normative_success": (
        "normative_success",
        "epistemic_success",
        "normative_failure",
    ),
    "epistemic_success": (
        "normative_success",
        "epistemic_success",
        "epistemic_failure",
    ),
}
EXPECTED_CANDIDATES = {
    "normative_failure": " nedodržení daného slova",
    "normative_success": " dodržení daného slova",
    "epistemic_failure": " nesprávné očekávání",
    "epistemic_success": " správné očekávání",
}


class BatteryError(RuntimeError):
    """A frozen-design, API, identity, or persistence invariant failed."""


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BatteryError(f"Nelze hashovat {path}: {exc}") from exc
    return digest.hexdigest()


def pretty_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BatteryError(f"Objekt nelze zapsat jako JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BatteryError(
            f"Objekt nelze zapsat jako kanonický JSON: {exc}"
        ) from exc
    return (text + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write(path, pretty_json_bytes(value))


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatteryError(f"Nelze načíst JSON {path}: {exc}") from exc


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BatteryError(f"{label} musí být neprázdný řetězec.")
    return value


def validate_materials(materials: dict[str, Any]) -> None:
    if materials.get("series_id") != REQUIRED_SERIES_ID:
        raise BatteryError("Materiál má jiné nebo chybějící series_id.")
    if materials.get("language") != "cs":
        raise BatteryError("Jazyk materiálu musí být cs.")
    if materials.get("primary_model") != "Apertus 8B Hybrid 8bit/BF16":
        raise BatteryError("Primární model v materiálu není uzamčený Hybrid.")
    if materials.get("primary_tool") != "Překvapení textu":
        raise BatteryError("Primárním nástrojem musí být Překvapení textu.")
    if materials.get("scoring") != "teacher_forced_total_surprisal":
        raise BatteryError("Materiál nemá uzamčené teacher-forced skórování.")
    if materials.get("temperature") != 1:
        raise BatteryError("Teplota musí být přesně 1.")
    if materials.get("candidates") != EXPECTED_CANDIDATES:
        raise BatteryError("Zmrazená kandidátní pokračování se změnila.")
    shared_prefix = _nonempty_string(
        materials.get("shared_evaluation_prefix"),
        "shared_evaluation_prefix",
    )
    if shared_prefix != "Šlo především o":
        raise BatteryError("Společný hodnoticí prefix se změnil.")

    items = materials.get("items")
    if not isinstance(items, list) or len(items) != 10:
        raise BatteryError("Materiál musí obsahovat přesně 10 položek.")
    expected_ids = [f"z12_{index:02d}" for index in range(1, 11)]
    if [item.get("id") for item in items if isinstance(item, dict)] != expected_ids:
        raise BatteryError("Položky nemají přesně uzamčené pořadí a ID.")
    domains: set[str] = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise BatteryError(f"Položka {index} není objekt.")
        expected_block = (
            "slibil_odhadl" if index <= 5 else "zarucil_predpokladal"
        )
        if item.get("lexical_block") != expected_block:
            raise BatteryError(
                f"{item.get('id')}: nesouhlasí lexikální blok."
            )
        domain = _nonempty_string(item.get("domain"), f"{item['id']}.domain")
        if domain in domains:
            raise BatteryError(f"Doména {domain!r} je použita vícekrát.")
        domains.add(domain)
        prefixes = item.get("prefixes")
        if not isinstance(prefixes, dict) or set(prefixes) != set(PREFIX_ORDER):
            raise BatteryError(
                f"{item['id']}: chybí přesně čtyři uzamčené prefixy."
            )
        for prefix_id in PREFIX_ORDER:
            prefix = _nonempty_string(
                prefixes[prefix_id],
                f"{item['id']}.{prefix_id}",
            )
            if prefix != prefix.rstrip():
                raise BatteryError(
                    f"{item['id']}.{prefix_id}: prefix má koncovou mezeru."
                )
            if not prefix.endswith(shared_prefix):
                raise BatteryError(
                    f"{item['id']}.{prefix_id}: prefix nekončí společným řezem."
                )

    for candidate_id, candidate in EXPECTED_CANDIDATES.items():
        if not candidate.startswith(" ") or candidate.startswith("  "):
            raise BatteryError(
                f"{candidate_id}: pokračování musí začínat právě jednou mezerou."
            )
        if candidate != candidate.rstrip() or candidate.endswith("."):
            raise BatteryError(
                f"{candidate_id}: pokračování nesmí mít konečnou mezeru ani tečku."
            )


def load_frozen_materials(
    path: Path,
) -> tuple[dict[str, Any], bytes, str]:
    try:
        raw = path.read_bytes()
        materials = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatteryError(f"Nelze načíst materiál {path}: {exc}") from exc
    digest = sha256_bytes(raw)
    if digest != FROZEN_MATERIALS_SHA256:
        raise BatteryError(
            "Hash materiálu nesouhlasí s preregistrací: "
            f"čekáno {FROZEN_MATERIALS_SHA256}, nalezeno {digest}."
        )
    if not isinstance(materials, dict):
        raise BatteryError("Kořen materiálu musí být JSON objekt.")
    validate_materials(materials)
    return materials, raw, digest


def expand_materials(materials: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = materials["candidates"]
    rows: list[dict[str, Any]] = []
    for item in materials["items"]:
        for prefix_id in PREFIX_ORDER:
            prefix = item["prefixes"][prefix_id]
            for candidate_id in SCORING_MATRIX[prefix_id]:
                candidate = candidates[candidate_id]
                rows.append(
                    {
                        "run_id": (
                            f"{item['id']}__{prefix_id}__{candidate_id}"
                        ),
                        "item_id": item["id"],
                        "domain": item["domain"],
                        "lexical_block": item["lexical_block"],
                        "prefix_id": prefix_id,
                        "candidate_id": candidate_id,
                        "prefix": prefix,
                        "candidate": candidate,
                        "candidate_character_start": len(prefix),
                        "text": prefix + candidate,
                    }
                )
    run_ids = [row["run_id"] for row in rows]
    texts = [row["text"] for row in rows]
    if len(rows) != 120:
        raise BatteryError(f"Expanze vytvořila {len(rows)} místo 120 textů.")
    if len(set(run_ids)) != len(rows):
        raise BatteryError("Expanze vytvořila duplicitní run_id.")
    if len(set(texts)) != len(rows):
        raise BatteryError("Expanze vytvořila duplicitní úplné texty.")
    return rows


def ensure_loopback_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username:
        raise BatteryError("API musí být obyčejné http na loopback adrese.")
    hostname = parsed.hostname.rstrip(".").casefold()
    try:
        loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        loopback = hostname == "localhost" or hostname.endswith(".localhost")
    if not loopback:
        raise BatteryError("Runner z bezpečnostních důvodů volá jen loopback API.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise BatteryError("--base-url nesmí obsahovat cestu, query ani fragment.")
    return base_url.rstrip("/")


def _http_json(
    request: urllib.request.Request,
    *,
    timeout: float,
) -> tuple[int, dict[str, Any], float]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            detail: Any = json.loads(raw)
        except json.JSONDecodeError:
            detail = raw.decode("utf-8", errors="replace")
        raise BatteryError(
            f"{request.full_url} vrátil HTTP {exc.code}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BatteryError(f"{request.full_url} není dostupné: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BatteryError(f"{request.full_url} nevrátilo platný JSON.") from exc
    if not isinstance(payload, dict):
        raise BatteryError(f"{request.full_url} nevrátilo JSON objekt.")
    return status, payload, (time.perf_counter() - started) * 1000


def get_health(base_url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/api/health",
        headers={"Accept": "application/json"},
        method="GET",
    )
    _status, health, _elapsed = _http_json(request, timeout=timeout)
    return health


def validate_health(health: dict[str, Any]) -> dict[str, Any]:
    active_model = health.get("active_model")
    worker = health.get("worker")
    provenance = health.get("provenance")
    model_provenance = (
        provenance.get("model") if isinstance(provenance, dict) else None
    )
    if health.get("status") != "ready" or health.get("phase") is not None:
        raise BatteryError(
            "Model není připraven: "
            f"status={health.get('status')!r}, phase={health.get('phase')!r}."
        )
    if health.get("mock") is not False:
        raise BatteryError("Registrovaný běh odmítá mock server.")
    if (
        not isinstance(active_model, dict)
        or active_model.get("id") != REQUIRED_MODEL_ID
    ):
        raise BatteryError(
            "Aktivní model musí být Hybrid · 8bit/BF16; "
            f"nalezeno {active_model!r}."
        )
    if (
        not isinstance(worker, dict)
        or worker.get("busy") is not False
        or worker.get("queue_depth") != 0
    ):
        raise BatteryError("Modelový worker není před během volný.")
    if not isinstance(model_provenance, dict):
        raise BatteryError("Health neobsahuje provenienci modelu.")
    metadata_fingerprint = model_provenance.get("metadata_fingerprint")
    if not isinstance(metadata_fingerprint, str) or not metadata_fingerprint:
        raise BatteryError("Health neobsahuje fingerprint modelu.")
    generation = health.get("model_generation")
    server_instance_id = health.get("server_instance_id")
    if type(generation) is not int:
        raise BatteryError("Health neobsahuje celé model_generation.")
    if not isinstance(server_instance_id, str) or not server_instance_id:
        raise BatteryError("Health neobsahuje server_instance_id.")
    return {
        "model_id": REQUIRED_MODEL_ID,
        "model_label": active_model.get("label"),
        "model_generation": generation,
        "server_instance_id": server_instance_id,
        "transition_id": health.get("transition_id"),
        "metadata_fingerprint": metadata_fingerprint,
        "model_provenance": model_provenance,
        "runtime_provenance": (
            provenance.get("runtime")
            if isinstance(provenance, dict)
            else None
        ),
    }


def stable_model_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Identity that must survive even when a stopped run resumes in a new server."""
    return {
        "model_id": identity["model_id"],
        "model_label": identity["model_label"],
        "metadata_fingerprint": identity["metadata_fingerprint"],
        "model_provenance": identity["model_provenance"],
    }


class LocalSurprisalApi:
    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        session_identity: dict[str, Any],
    ) -> None:
        self.base_url = ensure_loopback_base_url(base_url)
        self.timeout = timeout
        self.identity = session_identity

    def score(self, prefix: str, continuation: str) -> dict[str, Any]:
        submitted = {
            "prefix": prefix,
            "continuation": continuation,
            "top_k": TOP_K,
            "model_id": self.identity["model_id"],
            "model_generation": self.identity["model_generation"],
            "server_instance_id": self.identity["server_instance_id"],
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/continuation-surprisal",
            data=canonical_json_bytes(submitted),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        status, response, elapsed_ms = _http_json(
            request,
            timeout=self.timeout,
        )
        for field in ("model_id", "model_generation", "server_instance_id"):
            if response.get(field) != self.identity[field]:
                raise BatteryError(
                    f"/api/continuation-surprisal změnilo identitu {field}."
                )
        return {
            "started_at": utc_now(),
            "elapsed_client_ms": elapsed_ms,
            "http_status": status,
            "endpoint": "/api/continuation-surprisal",
            "request": submitted,
            "response": response,
        }


def extract_measurement(
    row: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    if response.get("analysis_kind") != "continuation_surprisal":
        raise BatteryError(f"{row['run_id']}: odpověď má jiný druh analýzy.")
    if (
        response.get("prefix") != row["prefix"]
        or response.get("continuation") != row["candidate"]
        or response.get("full_text") != row["text"]
    ):
        raise BatteryError(f"{row['run_id']}: server vrátil jiné vstupy.")
    if response.get("top_k") != TOP_K or response.get("temperature") != 1.0:
        raise BatteryError(f"{row['run_id']}: nesouhlasí top-k nebo teplota.")
    if response.get("probability_kind") != "raw_model":
        raise BatteryError(f"{row['run_id']}: nejde o raw pravděpodobnost.")
    execution = response.get("execution")
    if (
        not isinstance(execution, dict)
        or execution.get("mode") != "teacher_forced_incremental_kv"
        or execution.get("generation") is not False
        or execution.get("steering") is not False
        or execution.get("future_continuation_tokens_in_forward") is not False
    ):
        raise BatteryError(f"{row['run_id']}: porušen režim teacher forcing.")
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise BatteryError(f"{row['run_id']}: chybí tokenová stopa.")

    boundary = row["candidate_character_start"]
    candidate_tokens: list[dict[str, Any]] = []
    for token in tokens:
        if not isinstance(token, dict):
            raise BatteryError(f"{row['run_id']}: token není objekt.")
        start = token.get("character_start")
        end = token.get("character_end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise BatteryError(
                f"{row['run_id']}: token nemá spolehlivé znakové zarovnání."
            )
        if start >= boundary:
            candidate_tokens.append(token)
        else:
            raise BatteryError(
                f"{row['run_id']}: token neleží celý v pokračování."
            )

    candidate_text = "".join(
        str(token.get("text", "")) for token in candidate_tokens
    )
    if candidate_text != row["candidate"]:
        raise BatteryError(
            f"{row['run_id']}: tokeny neskládají přesné pokračování."
        )
    if not candidate_tokens:
        raise BatteryError(f"{row['run_id']}: pokračování nemá žádný token.")

    def checked_surprisal(token: dict[str, Any]) -> float:
        value = token.get("surprisal_bits")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise BatteryError(
                f"{row['run_id']}: token má nekonečné nebo chybějící I."
            )
        return float(value)

    candidate_surprisal = math.fsum(
        checked_surprisal(token) for token in candidate_tokens
    )
    summary = response.get("summary")
    server_total = (
        summary.get("total_continuation_surprisal_bits")
        if isinstance(summary, dict)
        else None
    )
    if (
        not isinstance(server_total, (int, float))
        or not math.isfinite(float(server_total))
        or not math.isclose(
            candidate_surprisal,
            float(server_total),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    ):
        raise BatteryError(
            f"{row['run_id']}: tokenový součet nesedí na pokračování."
        )
    prefix_token_ids = response.get("prefix_token_ids")
    prompt_logits_sha256 = response.get("prompt_logits_sha256")
    if (
        not isinstance(prefix_token_ids, list)
        or not prefix_token_ids
        or not all(type(value) is int for value in prefix_token_ids)
    ):
        raise BatteryError(f"{row['run_id']}: chybí tokeny prefixu.")
    if (
        not isinstance(prompt_logits_sha256, str)
        or len(prompt_logits_sha256) != 64
    ):
        raise BatteryError(f"{row['run_id']}: chybí hash kořenových logitů.")
    return {
        "conditional_surprisal_bits": candidate_surprisal,
        "prefix_token_ids": list(prefix_token_ids),
        "prompt_logits_sha256": prompt_logits_sha256,
        "candidate_token_ids": [
            int(token["token_id"]) for token in candidate_tokens
        ],
        "candidate_token_count": len(candidate_tokens),
        "candidate_character_start": boundary,
        "candidate_character_end": len(row["text"]),
    }


def validate_prefix_cancellation(
    expanded: list[dict[str, Any]],
    measurements: dict[str, dict[str, Any]],
) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_run = {row["run_id"]: row for row in expanded}
    for run_id, measurement in measurements.items():
        row = by_run[run_id]
        grouped.setdefault((row["item_id"], row["prefix_id"]), []).append(
            measurement
        )
    if len(grouped) != 40:
        raise BatteryError("Hotový běh nemá 40 úplných prefixových skupin.")
    for key, values in grouped.items():
        if len(values) != 3:
            raise BatteryError(f"{key}: prefix nemá přesně tři pokračování.")
        first_ids = values[0]["prefix_token_ids"]
        first_logits_hash = values[0]["prompt_logits_sha256"]
        for value in values[1:]:
            if value["prefix_token_ids"] != first_ids:
                raise BatteryError(
                    f"{key}: tokenizace společného prefixu se mezi kandidáty liší."
                )
            if value["prompt_logits_sha256"] != first_logits_hash:
                raise BatteryError(
                    f"{key}: kořenové logity společného prefixu se mezi "
                    "kandidáty liší."
                )


def mean(values: Iterable[float]) -> float:
    checked = [float(value) for value in values]
    if not checked:
        raise BatteryError("Nelze počítat průměr prázdného seznamu.")
    return statistics.fmean(checked)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise BatteryError("Nelze počítat percentil prázdného seznamu.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    fraction = position - left
    return ordered[left] + fraction * (ordered[right] - ordered[left])


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> list[float]:
    if not values:
        raise BatteryError("Bootstrap nemá žádné položky.")
    rng = random.Random(seed)
    size = len(values)
    replicates = [
        mean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    ]
    return [percentile(replicates, 0.025), percentile(replicates, 0.975)]


def exact_one_sided_sign_flip_p(values: list[float]) -> float:
    if not values:
        raise BatteryError("Sign-flip test nemá žádné položky.")
    observed = mean(values)
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        null_mean = mean(
            sign * value for sign, value in zip(signs, values, strict=True)
        )
        if null_mean >= observed - 1e-12:
            extreme += 1
    return extreme / (2 ** len(values))


def compute_analysis(
    materials: dict[str, Any],
    expanded: list[dict[str, Any]],
    measurements: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(measurements) != {row["run_id"] for row in expanded}:
        raise BatteryError("Analýza vyžaduje všech 120 úplných měření.")
    validate_prefix_cancellation(expanded, measurements)
    scores = {
        (
            row["item_id"],
            row["prefix_id"],
            row["candidate_id"],
        ): float(measurements[row["run_id"]]["conditional_surprisal_bits"])
        for row in expanded
    }

    item_rows: list[dict[str, Any]] = []
    for item in materials["items"]:
        item_id = item["id"]

        def score(prefix_id: str, candidate_id: str) -> float:
            try:
                return scores[(item_id, prefix_id, candidate_id)]
            except KeyError as exc:
                raise BatteryError(
                    f"{item_id}: chybí {prefix_id}/{candidate_id}."
                ) from exc

        l_failure_normative = (
            score("normative_failure", "epistemic_failure")
            - score("normative_failure", "normative_failure")
        )
        l_failure_epistemic = (
            score("epistemic_failure", "epistemic_failure")
            - score("epistemic_failure", "normative_failure")
        )
        d_failure = l_failure_normative - l_failure_epistemic

        l_success_normative = (
            score("normative_success", "epistemic_success")
            - score("normative_success", "normative_success")
        )
        l_success_epistemic = (
            score("epistemic_success", "epistemic_success")
            - score("epistemic_success", "normative_success")
        )
        d_success = l_success_normative - l_success_epistemic
        d_item = (d_failure + d_success) / 2

        c_normative = 0.5 * (
            score("normative_failure", "normative_success")
            - score("normative_failure", "normative_failure")
            + score("normative_success", "normative_failure")
            - score("normative_success", "normative_success")
        )
        c_epistemic = 0.5 * (
            score("epistemic_failure", "epistemic_success")
            - score("epistemic_failure", "epistemic_failure")
            + score("epistemic_success", "epistemic_failure")
            - score("epistemic_success", "epistemic_success")
        )
        item_rows.append(
            {
                "item_id": item_id,
                "domain": item["domain"],
                "lexical_block": item["lexical_block"],
                "L_failure_normative_bits": l_failure_normative,
                "L_failure_epistemic_bits": l_failure_epistemic,
                "D_failure_bits": d_failure,
                "L_success_normative_bits": l_success_normative,
                "L_success_epistemic_bits": l_success_epistemic,
                "D_success_bits": d_success,
                "D_bits": d_item,
                "C_normative_bits": c_normative,
                "C_epistemic_bits": c_epistemic,
            }
        )

    d_values = [row["D_bits"] for row in item_rows]
    d_failure_values = [row["D_failure_bits"] for row in item_rows]
    d_success_values = [row["D_success_bits"] for row in item_rows]
    c_normative_values = [row["C_normative_bits"] for row in item_rows]
    c_epistemic_values = [row["C_epistemic_bits"] for row in item_rows]
    block_means = {
        block: mean(
            row["D_bits"]
            for row in item_rows
            if row["lexical_block"] == block
        )
        for block in ("slibil_odhadl", "zarucil_predpokladal")
    }
    sign_flip_p = exact_one_sided_sign_flip_p(d_values)
    checks = {
        "mean_D_positive": mean(d_values) > 0,
        "sign_flip_p_below_0_05": sign_flip_p < 0.05,
        "at_least_8_of_10_D_positive": sum(value > 0 for value in d_values) >= 8,
        "mean_D_failure_positive": mean(d_failure_values) > 0,
        "mean_D_success_positive": mean(d_success_values) > 0,
        "normative_outcome_control": (
            mean(c_normative_values) > 0
            and sum(value > 0 for value in c_normative_values) >= 8
        ),
        "epistemic_outcome_control": (
            mean(c_epistemic_values) > 0
            and sum(value > 0 for value in c_epistemic_values) >= 8
        ),
        "both_lexical_blocks_positive": all(
            value > 0 for value in block_means.values()
        ),
    }
    status_gate_names = (
        "mean_D_positive",
        "sign_flip_p_below_0_05",
        "at_least_8_of_10_D_positive",
        "mean_D_failure_positive",
        "mean_D_success_positive",
        "both_lexical_blocks_positive",
    )
    status_gate = all(checks[name] for name in status_gate_names)
    full_gate = all(checks.values())
    if full_gate:
        outcome = "A_differentiated_status_and_outcome"
    elif status_gate and (
        not checks["normative_outcome_control"]
        or not checks["epistemic_outcome_control"]
    ):
        outcome = "B_status_without_validated_polarity"
    elif (
        checks["mean_D_failure_positive"]
        != checks["mean_D_success_positive"]
    ):
        outcome = "C_single_outcome_block"
    elif not checks["both_lexical_blocks_positive"] and mean(d_values) > 0:
        outcome = "D_lexically_limited"
    else:
        outcome = "E_null_or_opposite"

    return {
        "format": f"{FORMAT}.analysis",
        "schema_version": SCHEMA_VERSION,
        "series_id": materials["series_id"],
        "computed_at": utc_now(),
        "primary": {
            "mean_D_bits": mean(d_values),
            "D_positive_count": sum(value > 0 for value in d_values),
            "exact_one_sided_sign_flip_p": sign_flip_p,
            "bootstrap_mean_D_ci95_bits": bootstrap_mean_ci(d_values),
        },
        "outcome_blocks": {
            "mean_D_failure_bits": mean(d_failure_values),
            "mean_D_success_bits": mean(d_success_values),
        },
        "outcome_controls": {
            "mean_C_normative_bits": mean(c_normative_values),
            "C_normative_positive_count": sum(
                value > 0 for value in c_normative_values
            ),
            "bootstrap_mean_C_normative_ci95_bits": bootstrap_mean_ci(
                c_normative_values
            ),
            "mean_C_epistemic_bits": mean(c_epistemic_values),
            "C_epistemic_positive_count": sum(
                value > 0 for value in c_epistemic_values
            ),
            "bootstrap_mean_C_epistemic_ci95_bits": bootstrap_mean_ci(
                c_epistemic_values
            ),
        },
        "lexical_block_mean_D_bits": block_means,
        "decision": {
            "checks": checks,
            "full_confirmatory_gate_passed": full_gate,
            "outcome": outcome,
        },
        "items": item_rows,
        "interpretation_limit": (
            "Jde o statusově podmíněnou dostupnost pokračování v tomto "
            "modelu a materiálu, nikoli o důkaz, že model nese závazky."
        ),
    }


def items_csv_bytes(analysis: dict[str, Any]) -> bytes:
    rows = analysis["items"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def build_report(analysis: dict[str, Any]) -> str:
    primary = analysis["primary"]
    blocks = analysis["outcome_blocks"]
    controls = analysis["outcome_controls"]
    decision = analysis["decision"]
    block_means = analysis["lexical_block_mean_D_bits"]
    checks = "\n".join(
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in decision["checks"].items()
    )
    return f"""# Zahrada 12 — výsledek uzamčené série

**Stav:** `{decision['outcome']}`\\
**Plná konfirmační brána:** {'prošla' if decision['full_confirmatory_gate_passed'] else 'neprošla'}

## Primární výsledek

- průměr D: **{primary['mean_D_bits']:.4f} bitu**
- kladné D: **{primary['D_positive_count']}/10**
- exaktní jednostranný sign-flip: **p = {primary['exact_one_sided_sign_flip_p']:.6f}**
- bootstrap 95% CI průměru D: **[{primary['bootstrap_mean_D_ci95_bits'][0]:.4f}; {primary['bootstrap_mean_D_ci95_bits'][1]:.4f}] bitu**

## Kontroly

- průměr D, neúspěch: **{blocks['mean_D_failure_bits']:.4f} bitu**
- průměr D, úspěch: **{blocks['mean_D_success_bits']:.4f} bitu**
- normativní polarita: **{controls['mean_C_normative_bits']:.4f} bitu**, kladná u **{controls['C_normative_positive_count']}/10**
- epistemická polarita: **{controls['mean_C_epistemic_bits']:.4f} bitu**, kladná u **{controls['C_epistemic_positive_count']}/10**
- `slíbil × odhadl`: průměr D **{block_means['slibil_odhadl']:.4f} bitu**
- `zaručil × předpokládal`: průměr D **{block_means['zarucil_predpokladal']:.4f} bitu**

## Rozhodovací brána

{checks}

## Hranice interpretace

{analysis['interpretation_limit']}
"""


def checkpoint_path(series_dir: Path) -> Path:
    return series_dir / "checkpoint.json"


def save_checkpoint(series_dir: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(checkpoint_path(series_dir), checkpoint)


def raw_path(series_dir: Path, run_id: str) -> Path:
    return series_dir / "raw" / f"{run_id}.json"


def load_measurements_from_checkpoint(
    series_dir: Path,
    checkpoint: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    measurements: dict[str, dict[str, Any]] = {}
    completed = checkpoint.get("completed")
    if not isinstance(completed, dict):
        raise BatteryError("Checkpoint nemá objekt completed.")
    for run_id, entry in completed.items():
        if not isinstance(entry, dict):
            raise BatteryError(f"{run_id}: neplatný checkpoint záznam.")
        relative = entry.get("raw_path")
        expected_sha = entry.get("raw_sha256")
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise BatteryError(f"{run_id}: checkpoint postrádá raw identitu.")
        path = (series_dir / relative).resolve()
        try:
            path.relative_to(series_dir.resolve())
        except ValueError as exc:
            raise BatteryError(f"{run_id}: raw cesta opouští sérii.") from exc
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise BatteryError(f"{run_id}: raw soubor chybí nebo změnil hash.")
        bundle = load_json(path)
        measurement = bundle.get("measurement")
        if not isinstance(measurement, dict):
            raise BatteryError(f"{run_id}: raw soubor nemá measurement.")
        measurements[run_id] = measurement
    return measurements


def create_manifest(series_dir: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    files = []
    for path in sorted(series_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
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
        "status": checkpoint["status"],
        "completed_at": checkpoint.get("completed_at"),
        "materials_sha256": checkpoint["materials_sha256"],
        "runner_sha256": checkpoint["runner_sha256"],
        "stable_model_identity": checkpoint["stable_model_identity"],
        "run_count": len(checkpoint["completed"]),
        "files": files,
    }


def validate_series_id(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if not value or len(value) > 128 or value[0] not in allowed:
        raise BatteryError("Neplatné --series-id.")
    if any(character not in allowed for character in value):
        raise BatteryError("Neplatné --series-id.")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Uzamčená 120textová teacher-forced baterie Zahrada 12."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--materials", type=Path, default=DEFAULT_MATERIALS)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--series-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=305.0)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    materials_path = args.materials.expanduser().resolve()
    materials, materials_raw, materials_sha = load_frozen_materials(
        materials_path
    )
    expanded = expand_materials(materials)
    runner_sha = sha256_file(Path(__file__).resolve())
    if args.validate_only:
        return {
            "status": "valid",
            "series_id": materials["series_id"],
            "materials_path": str(materials_path),
            "materials_sha256": materials_sha,
            "item_count": len(materials["items"]),
            "prefix_count": len(materials["items"]) * len(PREFIX_ORDER),
            "unique_full_text_count": len(expanded),
            "model_forward_executed": False,
            "required_model_id": REQUIRED_MODEL_ID,
            "top_k_diagnostic_only": TOP_K,
        }
    if args.series_id is None:
        raise BatteryError(
            "Ostrý běh vyžaduje explicitní --series-id, např. "
            "zahrada12-normative-status-v1-run1."
        )
    series_id = validate_series_id(args.series_id)
    if args.resume is True and args.series_id is None:
        raise BatteryError("--resume vyžaduje --series-id.")

    base_url = ensure_loopback_base_url(args.base_url)
    health = get_health(base_url, args.timeout)
    session_identity = validate_health(health)
    stable_identity = stable_model_identity(session_identity)
    raw_root = args.raw_root.expanduser().resolve()
    series_dir = raw_root / series_id
    checkpoint: dict[str, Any]

    if args.resume:
        if not series_dir.is_dir():
            raise BatteryError(f"Resume série neexistuje: {series_dir}.")
        checkpoint = load_json(checkpoint_path(series_dir))
        if (
            checkpoint.get("format") != FORMAT
            or checkpoint.get("schema_version") != SCHEMA_VERSION
            or checkpoint.get("series_id") != series_id
        ):
            raise BatteryError("Resume checkpoint má jiný formát nebo sérii.")
        expected = {
            "materials_sha256": materials_sha,
            "runner_sha256": runner_sha,
            "run_order": [row["run_id"] for row in expanded],
            "stable_model_identity": stable_identity,
        }
        for field, value in expected.items():
            if checkpoint.get(field) != value:
                raise BatteryError(f"Resume nesouhlasí v poli {field}.")
        if not isinstance(checkpoint.get("prefix_roots"), dict):
            raise BatteryError("Resume checkpoint postrádá audity prefixů.")
        if checkpoint.get("status") == "complete":
            raise BatteryError("Série už je kompletní; znovu ji nespouštím.")
        load_measurements_from_checkpoint(series_dir, checkpoint)
        checkpoint.setdefault("sessions", []).append(
            {
                "started_at": utc_now(),
                "resumed": True,
                "session_identity": session_identity,
            }
        )
        save_checkpoint(series_dir, checkpoint)
    else:
        if series_dir.exists():
            raise BatteryError(
                f"Cílová série už existuje: {series_dir}; "
                "použij --resume nebo jiné ID."
            )
        series_dir.mkdir(parents=True)
        atomic_write(series_dir / "materials.json", materials_raw)
        checkpoint = {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "series_id": series_id,
            "registered_series_id": materials["series_id"],
            "status": "incomplete",
            "started_at": utc_now(),
            "materials_sha256": materials_sha,
            "runner_sha256": runner_sha,
            "top_k_diagnostic_only": TOP_K,
            "temperature": 1,
            "run_order": [row["run_id"] for row in expanded],
            "stable_model_identity": stable_identity,
            "completed": {},
            "prefix_roots": {},
            "sessions": [
                {
                    "started_at": utc_now(),
                    "resumed": False,
                    "session_identity": session_identity,
                }
            ],
            "last_error": None,
        }
        save_checkpoint(series_dir, checkpoint)

    api = LocalSurprisalApi(
        base_url=base_url,
        timeout=args.timeout,
        session_identity=session_identity,
    )
    try:
        for index, row in enumerate(expanded, start=1):
            if row["run_id"] in checkpoint["completed"]:
                continue
            print(
                f"[{index:03d}/120] {row['run_id']}",
                file=sys.stderr,
                flush=True,
            )
            call = api.score(row["prefix"], row["candidate"])
            measurement = extract_measurement(row, call["response"])
            prefix_group = f"{row['item_id']}__{row['prefix_id']}"
            observed_root = {
                "prefix_token_ids": measurement["prefix_token_ids"],
                "prompt_logits_sha256": measurement[
                    "prompt_logits_sha256"
                ],
            }
            stored_root = checkpoint["prefix_roots"].get(prefix_group)
            if stored_root is None:
                checkpoint["prefix_roots"][prefix_group] = observed_root
            elif stored_root != observed_root:
                raise BatteryError(
                    f"{prefix_group}: inkrementální kořen se mezi kandidáty "
                    "změnil; série končí okamžitě."
                )
            bundle = {
                "format": f"{FORMAT}.raw-text",
                "schema_version": SCHEMA_VERSION,
                "registered_series_id": materials["series_id"],
                "collection_series_id": series_id,
                "materials_sha256": materials_sha,
                "runner_sha256": runner_sha,
                "captured_at": utc_now(),
                "design_row": row,
                "session_identity": session_identity,
                "api": call,
                "measurement": measurement,
            }
            path = raw_path(series_dir, row["run_id"])
            atomic_write_json(path, bundle)
            relative = str(path.relative_to(series_dir))
            checkpoint["completed"][row["run_id"]] = {
                "raw_path": relative,
                "raw_sha256": sha256_file(path),
                "conditional_surprisal_bits": measurement[
                    "conditional_surprisal_bits"
                ],
            }
            checkpoint["last_error"] = None
            save_checkpoint(series_dir, checkpoint)

        final_health = get_health(base_url, args.timeout)
        final_identity = validate_health(final_health)
        if final_identity != session_identity:
            raise BatteryError("Identita serveru nebo modelu se během běhu změnila.")
        measurements = load_measurements_from_checkpoint(series_dir, checkpoint)
        analysis = compute_analysis(materials, expanded, measurements)
        atomic_write_json(series_dir / "analysis.json", analysis)
        atomic_write(
            series_dir / "items.csv",
            items_csv_bytes(analysis),
        )
        atomic_write(
            series_dir / "report.md",
            build_report(analysis).encode("utf-8"),
        )
        checkpoint["status"] = "complete"
        checkpoint["completed_at"] = utc_now()
        checkpoint["sessions"][-1]["completed_at"] = checkpoint["completed_at"]
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
        checkpoint["sessions"][-1]["ended_at"] = utc_now()
        save_checkpoint(series_dir, checkpoint)
        raise

    return {
        "status": "complete",
        "series_id": series_id,
        "series_dir": str(series_dir),
        "run_count": len(checkpoint["completed"]),
        "outcome": analysis["decision"]["outcome"],
        "full_confirmatory_gate_passed": analysis["decision"][
            "full_confirmatory_gate_passed"
        ],
        "mean_D_bits": analysis["primary"]["mean_D_bits"],
        "sign_flip_p": analysis["primary"]["exact_one_sided_sign_flip_p"],
        "report": str(series_dir / "report.md"),
        "analysis": str(series_dir / "analysis.json"),
        "manifest": str(series_dir / "manifest.json"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except BatteryError as exc:
        print(f"CHYBA: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
