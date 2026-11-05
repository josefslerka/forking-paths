"""Memory-only EuroLLM diagnostics beside the frozen Apertus detector.

The calibrated detector and its sampling remain untouched. Paired measurements
use identical literal context/target spans and native, canonical token paths.
No classifier, thresholds, training labels or persistent experiment store are
used by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .domains.tokenization import exact_byte_offsets
except ImportError:
    from domains.tokenization import exact_byte_offsets


PROTOCOL = {
    "id": "detector-readers-v1",
    "temperature": 1.0,
    "chat_template": False,
    "execution": "teacher_forced_full_window",
    "sampling": "frozen_apertus_detector_three_windows",
    "paired_target_start": "first_shared_canonical_utf8_token_boundary",
    "maximum_trimmed_apertus_target_tokens": 16,
    "minimum_target_tokens_per_reader": 32,
    "target_end": "unchanged_frozen_window_end",
    "metric": "canonical_token_path_surprisal_bits_per_non_whitespace_character",
    "word_boundary_correction": False,
    "native_token_features": "descriptive_only_not_directly_comparable_across_vocabularies",
    "delta_definition": "eurollm_minus_apertus",
    "decision_authority": False,
}
FEATURE_LABELS = (
    "surprisal_bits_per_non_whitespace_character",
    "token_surprisal_iqr",
    "token_surprisal_p95",
    "top_5pct_surprisal_share",
    "mean_surprisal_minus_entropy_bits",
)
JOB_TTL_SECONDS = 900
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _features(rows: list[dict[str, Any]], non_whitespace_characters: int) -> dict[str, float]:
    if not rows or non_whitespace_characters <= 0:
        raise ValueError("Měřicí úsek neobsahuje použitelné tokeny a znaky.")
    surprises = [float(row["surprisal_bits"]) for row in rows]
    entropies = [float(row["entropy_bits"]) for row in rows]
    if any(not math.isfinite(value) or value < 0 for value in surprises + entropies):
        raise ValueError("Model vrátil neplatné překvapení nebo entropii.")
    total = math.fsum(surprises)
    if total <= 0:
        raise ValueError("Model vrátil nulové celkové překvapení.")
    q1, q3, p95 = np.quantile(surprises, [0.25, 0.75, 0.95])
    tail_count = max(1, math.ceil(len(surprises) * .05))
    return {
        FEATURE_LABELS[0]: total / non_whitespace_characters,
        FEATURE_LABELS[1]: float(q3 - q1),
        FEATURE_LABELS[2]: float(p95),
        FEATURE_LABELS[3]: math.fsum(sorted(surprises, reverse=True)[:tail_count]) / total,
        FEATURE_LABELS[4]: total / len(rows) - math.fsum(entropies) / len(rows),
    }


def _ranges(response: dict[str, Any], algorithm: str | None, *, mock: bool) -> list[dict[str, Any]] | None:
    text, tokens = response["normalized_text"], response["tokens"]
    if len(tokens) != len(response["token_ids"]):
        return None
    if not mock:
        recovered = exact_byte_offsets(text, tokens, algorithm)
        return recovered[1] if recovered else None
    # Development fixtures have exact character offsets and no raw byte tokens.
    spans = []
    for row in tokens:
        start, end = row.get("character_start"), row.get("character_end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            return None
        spans.append({"utf8_byte_start": len(text[:start].encode()),
                      "utf8_byte_end": len(text[:end].encode())})
    if not spans or spans[0]["utf8_byte_start"] != 0 or spans[-1]["utf8_byte_end"] != len(text.encode()):
        return None
    if any(a["utf8_byte_end"] != b["utf8_byte_start"] for a, b in zip(spans, spans[1:])):
        return None
    return spans


def _canonical_prefix(tokenizer: Any, prefix: str, ids: list[int], index: int, *, mock: bool) -> bool:
    if mock:
        return True
    return list(tokenizer.encode(prefix, add_special_tokens=False)) == ids[:index]


def paired_target_plan(
    sampling: dict[str, Any],
    a: dict[str, Any],
    b: dict[str, Any],
    tokenizer_a: Any,
    tokenizer_b: Any,
    algorithm_a: str | None,
    algorithm_b: str | None,
    *,
    mock: bool = False,
) -> tuple[dict[str, Any], tuple[int, int] | None]:
    """Trim at most 16 Apertus targets, using boundaries only, never scores.

    Whole windows are excluded when literal byte reconstruction, a canonical
    prefix or sufficient target length cannot be verified for both readers.
    The original Apertus classifier still uses its original untrimmed targets.
    """
    output = {"index": sampling["index"], "position": sampling["position"],
              "status": "excluded", "reason": None,
              "original_apertus_target_tokens": sampling["target_token_count"]}
    if a.get("analysis_kind") != "text_surprisal" or b.get("analysis_kind") != "text_surprisal":
        raise RuntimeError("Čtenář vrátil jiný druh analýzy.")
    text = a["normalized_text"]
    if b["normalized_text"] != text:
        raise RuntimeError("Čtenáři nezměřili totožný text.")
    output["window_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    output["window_character_count"] = len(text)
    spans_a = _ranges(a, algorithm_a, mock=mock)
    spans_b = _ranges(b, algorithm_b, mock=mock)
    if spans_a is None or spans_b is None:
        output["reason"] = "unreliable_apertus_offsets" if spans_a is None else "unreliable_eurollm_offsets"
        return output, None
    target_index = sampling["target_start"] - sampling["context_start"]
    if not 0 < target_index < len(spans_a) or target_index + sampling["target_token_count"] != len(spans_a):
        raise RuntimeError("Původní měřicí okno nemá očekávané hranice.")
    byte_starts_b = {span["utf8_byte_start"]: i for i, span in enumerate(spans_b)
                     if span["utf8_byte_end"] > span["utf8_byte_start"]}
    minimum = PROTOCOL["minimum_target_tokens_per_reader"]
    max_trim = PROTOCOL["maximum_trimmed_apertus_target_tokens"]
    encoded = text.encode()
    chosen = None
    for ia in range(target_index, min(target_index + max_trim + 1, len(spans_a))):
        start = spans_a[ia]["utf8_byte_start"]
        ib = byte_starts_b.get(start)
        if ib is None or ib == 0 or len(spans_a) - ia < minimum or len(spans_b) - ib < minimum:
            continue
        try:
            prefix = encoded[:start].decode("utf-8", errors="strict")
            target = encoded[start:].decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        if (not prefix or not target.strip()
                or not _canonical_prefix(tokenizer_a, prefix, a["token_ids"], ia, mock=mock)
                or not _canonical_prefix(tokenizer_b, prefix, b["token_ids"], ib, mock=mock)):
            continue
        chosen = ia, ib, start, prefix, target
        break
    if chosen is None:
        output["reason"] = "no_shared_canonical_boundary"
        return output, None
    ia, ib, start, prefix, target = chosen
    characters = sum(not character.isspace() for character in target)
    output.update(status="paired", reason=None,
                  trimmed_apertus_target_tokens=ia - target_index,
                  context_character_count=len(prefix), target_character_count=len(target),
                  target_non_whitespace_character_count=characters,
                  target_utf8_byte_start=start, target_utf8_byte_end=len(encoded),
                  target_sha256=hashlib.sha256(target.encode()).hexdigest())
    return output, (ia, ib)


def pair_window(
    sampling: dict[str, Any],
    a: dict[str, Any],
    b: dict[str, Any],
    tokenizer_a: Any,
    tokenizer_b: Any,
    algorithm_a: str | None,
    algorithm_b: str | None,
    *,
    mock: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    output, indices = paired_target_plan(
        sampling, a, b, tokenizer_a, tokenizer_b, algorithm_a,
        algorithm_b, mock=mock)
    if indices is None:
        return output, [], []
    ia, ib = indices
    rows_a, rows_b = a["tokens"][ia:], b["tokens"][ib:]
    characters = output["target_non_whitespace_character_count"]
    measurements = {}
    for key, rows, response in (("apertus", rows_a, a), ("eurollm", rows_b, b)):
        measurements[key] = {
            "token_count": len(rows), "total_surprisal_bits": math.fsum(float(row["surprisal_bits"]) for row in rows),
            "features": _features(rows, characters),
            "forward_ms": response.get("timing_ms", {}).get("forward"),
        }
    output.update(measurements=measurements,
                  delta_bits_per_character=(measurements["eurollm"]["features"][FEATURE_LABELS[0]]
                                            - measurements["apertus"]["features"][FEATURE_LABELS[0]]))
    return output, rows_a, rows_b


def new_job(text: str, identity: dict[str, Any], euro_model_id: str) -> dict[str, Any]:
    return {"id": str(uuid.uuid4()), "status": "queued", "phase": "queued",
            "created_monotonic": time.monotonic(), "text": text,
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "request_identity": dict(identity), "euro_model_id": euro_model_id,
            "cancel": threading.Event(), "completed_measurements": 0, "total_measurements": 6,
            "errors": [], "result": None, "restored_identity": None}


class _Cancelled(Exception):
    pass


def run_comparison(state: Any, job: dict[str, Any]) -> None:
    """Own the serial worker through both measurements and model restoration."""
    started = time.perf_counter()
    original = state.active_model_id
    captures: list[dict[str, Any]] = []
    paired_rows = {"apertus": [], "eurollm": []}
    comparison = {"analysis_kind": "detector_reader_comparison", "schema_version": 1,
                  "research_only": True, "decision_authority": False,
                  "input_sha256": job["input_sha256"],
                  "protocol": dict(PROTOCOL), "protocol_sha256": _hash(PROTOCOL),
                  "implementation_sha256": IMPLEMENTATION_SHA256,
                  "models": {}, "windows": [], "summary": {}, "status": "running"}
    def capture(text: str, response: dict[str, Any]) -> None:
        captures.append({"text": text, "response": response})
        with state.state_lock:
            job["completed_measurements"] = len(captures)
        if job["cancel"].is_set():
            raise _Cancelled()
    def descriptor() -> dict[str, Any]:
        return {**state.model_descriptor(state.active_model_id),
                "model_generation": state.model_generation,
                "provenance": state._model_provenance_snapshot_unlocked(),
                "tokenizer": state._tokenizer_identity_unlocked()}
    with state.state_lock:
        job.update(status="running", phase="apertus")
    try:
        if job["cancel"].is_set():
            raise _Cancelled()
        comparison["models"]["apertus"] = descriptor()
        result = state._predictive_stylometry_local(job["text"], capture_window=capture)
        with state.state_lock:
            job["result"] = result
        if job["cancel"].is_set():
            raise _Cancelled()
        tokenizer_a = state.predictive_detector_engine.tokenizer if not state.mock else None
        windows = result["sampling"]["apertus"]["windows"]
        if len(captures) != 3 or len(windows) != len(captures):
            raise RuntimeError("Apertus nevytvořil tři očekávaná okna.")
        with state.state_lock:
            job["phase"] = "loading_eurollm"
        load_started = time.perf_counter()
        state._switch_model_local(job["euro_model_id"])
        if state.status != "ready":
            raise RuntimeError(state.error or "EuroLLM se nepodařilo načíst.")
        comparison["load_eurollm_ms"] = (time.perf_counter() - load_started) * 1000
        comparison["models"]["eurollm"] = descriptor()
        with state.state_lock:
            job["phase"] = "eurollm"
        for sampling, captured in zip(windows, captures, strict=True):
            if job["cancel"].is_set():
                raise _Cancelled()
            response = state._text_surprisal_local(captured["text"], 1)
            paired, a, b = pair_window(
                sampling, captured["response"], response, tokenizer_a, state.tokenizer,
                comparison["models"]["apertus"]["tokenizer"].get("algorithm"),
                comparison["models"]["eurollm"]["tokenizer"].get("algorithm"), mock=state.mock)
            comparison["windows"].append(paired)
            paired_rows["apertus"].extend(a)
            paired_rows["eurollm"].extend(b)
            with state.state_lock:
                job["completed_measurements"] += 1
        with state.state_lock:
            job["status"] = "cancelled" if job["cancel"].is_set() else "completed"
    except _Cancelled:
        with state.state_lock:
            job["status"] = "cancelled"
    except Exception as exc:
        with state.state_lock:
            job["status"] = "failed"
            job["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        with state.state_lock:
            job["phase"] = "restoring"
        restore_started = time.perf_counter()
        try:
            if state.active_model_id != original or state.status != "ready":
                state._switch_model_local(original)
            if state.status != "ready":
                raise RuntimeError(state.error or "Původní model se nepodařilo obnovit.")
            with state.state_lock:
                job["restored_identity"] = {"model_id": state.active_model_id,
                                            "model_generation": state.model_generation,
                                            "server_instance_id": state.server_instance_id}
        except Exception as exc:
            with state.state_lock:
                job["status"] = "failed"
                job["restoration_error"] = str(exc)
                job["errors"].append(f"Obnova původního modelu selhala: {exc}")
        comparison["restore_model_ms"] = (time.perf_counter() - restore_started) * 1000
        if job["result"] is not None:
            measured_indices = {window["index"] for window in comparison["windows"]}
            for window in job["result"]["sampling"]["apertus"]["windows"]:
                if window["index"] not in measured_indices:
                    comparison["windows"].append({"index": window["index"], "position": window["position"],
                                                   "status": "unmeasured", "reason": "measurement_incomplete"})
            comparison["windows"].sort(key=lambda window: window["index"])
        paired = [window for window in comparison["windows"] if window["status"] == "paired"]
        characters = sum(window["target_non_whitespace_character_count"] for window in paired)
        summary = {"paired_windows": len(paired), "requested_windows": 3,
                   "target_non_whitespace_character_count": characters, "measurements": {}}
        for key, rows in paired_rows.items():
            if rows:
                summary["measurements"][key] = {"token_count": len(rows), "features": _features(rows, characters),
                                                 "total_surprisal_bits": math.fsum(float(row["surprisal_bits"]) for row in rows)}
        summary["delta_bits_per_character"] = (
            summary["measurements"]["eurollm"]["features"][FEATURE_LABELS[0]]
            - summary["measurements"]["apertus"]["features"][FEATURE_LABELS[0]] if paired else None)
        comparison.update(summary=summary, status=job["status"],
                          timing_ms={"total": (time.perf_counter() - started) * 1000},
                          errors=list(job["errors"]),
                          privacy={"stored": False, "stored_text": False, "contains_text": False})
        with state.state_lock:
            if job["result"] is not None:
                job["result"]["reader_comparison"] = comparison
            # No document, fragments, tokenizer strings or per-token text survive
            # in the bounded job cache. Results contain only metrics and hashes.
            job.pop("text", None)
            job["phase"] = "done"
