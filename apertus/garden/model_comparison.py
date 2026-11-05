"""Paired, tokenizer-aware measurements and immutable experiment records."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

try:
    from .domains.tokenization import exact_byte_offsets
    from . import steering_comparison
except ImportError:
    from domains.tokenization import exact_byte_offsets
    import steering_comparison

PROTOCOL = {
    "id": "model-comparison-v2", "temperature": 1.0, "chat_template": False,
    "normalization": "full_native_vocabulary", "text_normalization": "trim_trailing_whitespace_only",
    "span_policy": "whitespace_units_with_leading_separator_and_attached_punctuation",
    "metric": "canonical_token_path_surprisal_bits", "word_boundary_correction": False,
    "delta_definition": "model_b_minus_model_a",
    "context_effect": "S(continuation|prefix_b)-S(continuation|prefix_a)",
    "interpretation": "Cross-model robustness; training, architecture and precision are not isolated.",
}
MAX_ITEMS = 32


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    models = request.get("model_ids")
    if (not isinstance(models, list) or len(models) != 2
            or any(not isinstance(v, str) or not v for v in models) or models[0] == models[1]):
        raise ValueError("Vyber dva různé modely v model_ids.")
    steering = steering_comparison.validate_options(request["steering"]) if "steering" in request else None
    items = request.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS:
        raise ValueError(f"items musí obsahovat 1 až {MAX_ITEMS} zadání.")
    if steering and len(items) > steering_comparison.MAX_ITEMS:
        raise ValueError(f"Steeringová dávka může obsahovat nejvýše {steering_comparison.MAX_ITEMS} zadání.")
    checked = []
    seen = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("Každé zadání musí být JSON objekt.")
        item_id = item.get("id", f"item-{index + 1}")
        if not isinstance(item_id, str) or not item_id or len(item_id) > 120 or item_id in seen:
            raise ValueError("Každé zadání musí mít jedinečné krátké id.")
        seen.add(item_id)
        if steering and any(key in item for key in ("text", "prefix_a", "prefix_b", "continuation")):
            raise ValueError("Steeringové zadání obsahuje test_text a dvě pokračování.")
        fields = ("test_text", "negative_text", "positive_text") if steering else (("text",) if "text" in item else ("prefix_a", "prefix_b", "continuation"))
        if "text" in item and any(key in item for key in ("prefix_a", "prefix_b", "continuation")):
            raise ValueError("Zadání obsahuje buď text, nebo dva prefixy a continuation.")
        result = {"id": item_id, "kind": "steering" if steering else ("text" if fields == ("text",) else "context")}
        if "group" in item:
            if not isinstance(item["group"], str) or len(item["group"]) > 80:
                raise ValueError("group musí být krátký řetězec.")
            result["group"] = item["group"]
        for key in fields:
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{item_id}: {key} musí obsahovat neprázdný text.")
            if len(value) > 8192 or len(value.encode()) > 32768:
                raise ValueError(f"{item_id}: {key} překračuje limit délky.")
            if fields != ("text",) and value != value.rstrip():
                raise ValueError(f"{item_id}: koncové mezery nejsou povolené; oddělovač patří do continuation.")
            result[key] = value
        if result["kind"] == "context":
            for prefix in (result["prefix_a"], result["prefix_b"]):
                full = prefix + result["continuation"]
                if len(full) > 8192 or len(full.encode()) > 32768:
                    raise ValueError(f"{item_id}: prefix s pokračováním překračuje limit délky.")
        if steering:
            steering_comparison.validate_held_out_item(result, steering)
        checked.append(result)
    title = request.get("title", "Srovnání modelů")
    if not isinstance(title, str) or len(title) > 200:
        raise ValueError("title musí být řetězec nejvýše 200 znaků.")
    result = {"model_ids": list(models), "items": checked, "title": title}
    if steering:
        result["steering"] = steering
    return result



def aligned_units(response: dict[str, Any], tokenizer: Any = None, *, mock: bool = False, algorithm: str | None = None) -> dict[str, Any]:
    """Each token is assigned at most once, including standalone separator tokens.

    Shared character spans do not imply shared token events. Ambiguous offsets or
    non-canonical prefix boundaries are explicitly excluded from paired metrics.
    """
    text = response["normalized_text"]
    tokens = response["tokens"]
    recovered = exact_byte_offsets(text, tokens, algorithm) if not mock else None
    offset_source = "response_character_offsets"
    ranges = None
    if recovered:
        offset_source, ranges = recovered
        tokens = [{**token, **span} for token, span in zip(tokens, ranges, strict=True)]
    matches = list(re.finditer(r"\S+", text))
    units = []
    previous_end = 0
    for match in matches:
        units.append({"text": match.group(), "character_start": match.start(),
                      "character_end": match.end(), "path_character_start": previous_end,
                      "token_indices": [], "reasons": []})
        previous_end = match.end()
    unassigned = []
    for index, token in enumerate(tokens):
        start, end = token.get("character_start"), token.get("character_end")
        if token.get("synthetic_leading_marker") and units:
            units[0]["token_indices"].append(index)
            continue
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            unassigned.append(index)
            # Unknown offsets cannot be localized safely.
            for unit in units:
                unit["reasons"].append("unreliable_token_offsets")
            continue
        overlap = [unit for unit in units if start < unit["character_end"] and end > unit["path_character_start"]]
        if len(overlap) != 1:
            unassigned.append(index)
            for unit in overlap:
                unit["reasons"].append("token_crosses_unit_boundary")
            continue
        overlap[0]["token_indices"].append(index)
    from_token = []
    if not mock and tokenizer is not None:
        # Local import also supports server.py direct execution.
        try:
            from .domains.tokenization import encode_with_bos
        except ImportError:
            from domains.tokenization import encode_with_bos
        from_token = encode_with_bos(tokenizer, text)
    for unit in units:
        indices = unit["token_indices"]
        if not indices:
            unit["reasons"].append("no_assigned_tokens")
        else:
            covered = set()
            for i in indices:
                covered.update(range(tokens[i]["character_start"], tokens[i]["character_end"]))
            if not set(range(unit["path_character_start"], unit["character_end"])) <= covered:
                unit["reasons"].append("incomplete_character_coverage")
            if indices != list(range(indices[0], indices[-1] + 1)):
                unit["reasons"].append("non_contiguous_token_path")
            if from_token:
                before = encode_with_bos(tokenizer, text[:unit["path_character_start"]])
                after = encode_with_bos(tokenizer, text[:unit["character_end"]])
                if before != from_token[:indices[0] + 1] or after != from_token[:indices[-1] + 2]:
                    unit["reasons"].append("non_canonical_prefix_boundary")
        unit["reasons"] = sorted(set(unit["reasons"]))
        unit["complete"] = not unit["reasons"]
        unit["token_count"] = len(indices)
        unit["surprisal_bits"] = math.fsum(tokens[i]["surprisal_bits"] for i in indices) if unit["complete"] else None
    return {"units": units, "unassigned_token_indices": unassigned,
            "assigned_token_count": sum(unit["token_count"] for unit in units),
            "token_count": len(tokens), "method": PROTOCOL["span_policy"],
            "token_offset_source": offset_source, "recovered_token_ranges": ranges}


def paired_item(item: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    if item["kind"] == "steering":
        return steering_comparison.paired_item(item, a, b)
    output = {"id": item["id"], "kind": item["kind"], "model_a": a["model_id"], "model_b": b["model_id"]}
    if item["kind"] == "text":
        rows = []
        left, right = a["alignment"]["units"], b["alignment"]["units"]
        if [(u["character_start"], u["character_end"]) for u in left] != [(u["character_start"], u["character_end"]) for u in right]:
            raise RuntimeError("Modely nevrátily totožné znakové úseky.")
        for ua, ub in zip(left, right, strict=True):
            complete = ua["complete"] and ub["complete"]
            rows.append({"text": ua["text"], "character_start": ua["character_start"], "character_end": ua["character_end"],
                         "a_bits": ua["surprisal_bits"], "b_bits": ub["surprisal_bits"], "complete": complete,
                         "delta_b_minus_a_bits": ub["surprisal_bits"] - ua["surprisal_bits"] if complete else None,
                         "a_reasons": ua["reasons"], "b_reasons": ub["reasons"]})
        output["units"] = rows
        output["total_path_bits"] = {side: value["response"]["summary"]["total_surprisal_bits"] for side, value in (("a", a), ("b", b))}
        output["complete_paired_units"] = sum(row["complete"] for row in rows)
    else:
        effects = {"a": a["context_effect_bits"], "b": b["context_effect_bits"]}
        output.update(context_effect_bits=effects, delta_context_effect_b_minus_a_bits=effects["b"] - effects["a"],
                      same_effect_direction=(effects["a"] > 0) == (effects["b"] > 0) if all(v != 0 for v in effects.values()) else None)
    return output


def make_record(job: dict[str, Any]) -> dict[str, Any]:
    request = job["request"]
    models = request["model_ids"]
    rows = []
    for item in request["items"]:
        a = job["measurements"].get(models[0], {}).get(item["id"])
        b = job["measurements"].get(models[1], {}).get(item["id"])
        if a and b:
            rows.append(paired_item(item, a, b))
    protocol = steering_comparison.PROTOCOL if "steering" in request else PROTOCOL
    descriptors = job.get("models", {})
    context_rows = [row for row in rows if row["kind"] == "context"]
    direction_rows = [row for row in context_rows if row["same_effect_direction"] is not None]
    steering_rows = [row for row in rows if row["kind"] == "steering"]
    steering_direction_rows = [row for row in steering_rows if row["same_effect_direction"] is not None]
    return {
        "schema": "apertus.experiment-record", "schema_version": 1, "id": job["id"], "kind": "model_comparison",
        "captured_at": utc_now(), "annotation": {"title": request["title"], "notes": "", "tags": ["model-comparison"]},
        "capture": {"status": "complete" if job["status"] == "completed" else job["status"], "warnings": list(job["errors"]), "temperature": 1.0},
        "provenance": {"model": {"model_id": " × ".join(models)}, "models": descriptors,
                       "protocol": protocol, "protocol_sha256": canonical_hash(protocol), "input_sha256": canonical_hash(request),
                       "environment": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
                       "server_instance_id": job["server_instance_id"]},
        "payload": {"request": request, "measurements": job["measurements"], "comparisons": rows,
                    "summary": {"paired_items": len(rows), "requested_items": len(request["items"]),
                                "context_direction_agreement_count": sum(row["same_effect_direction"] for row in direction_rows),
                                "context_direction_eligible_count": len(direction_rows),
                                "steering_direction_agreement_count": sum(row["same_effect_direction"] for row in steering_direction_rows),
                                "steering_direction_eligible_count": len(steering_direction_rows)}, "protocol": protocol},
        "diagnostics": {"timing_ms": job.get("timing_ms", {}), "load_ms": job.get("load_ms", {}),
                        "restoration_error": job.get("restoration_error"), "errors": job["errors"]},
    }


def run_comparison(state: Any, job: dict[str, Any]) -> None:
    """Run inside the existing single model worker; hold the reservation until restoration."""
    started = time.perf_counter()
    original = state.active_model_id
    with state.state_lock:
        job["status"] = "running"
    try:
        for model_id in job["request"]["model_ids"]:
            if job["cancel"].is_set():
                break
            with state.state_lock:
                job["current_model_id"] = model_id
                job["phase"] = "loading"
            load_started = time.perf_counter()
            if state.active_model_id != model_id:
                state._switch_model_local(model_id)
            if state.status != "ready":
                raise RuntimeError(state.error or "Model se nepodařilo načíst.")
            descriptor = dict(state.model_descriptor(model_id))
            descriptor["provenance"] = state._model_provenance_snapshot_unlocked()
            descriptor["tokenizer"] = state._tokenizer_identity_unlocked()
            with state.state_lock:
                job["load_ms"][model_id] = (time.perf_counter() - load_started) * 1000
                job["models"][model_id] = descriptor
                job["measurements"][model_id] = {}
                job["phase"] = "scoring"
            for item in job["request"]["items"]:
                if job["cancel"].is_set():
                    break
                if item["kind"] == "text":
                    response = state._text_surprisal_local(item["text"], 3)
                    measurement = {"model_id": model_id, "response": response,
                                   "alignment": aligned_units(response, state.tokenizer, mock=state.mock, algorithm=descriptor["tokenizer"]["algorithm"])}
                elif item["kind"] == "steering":
                    state.require_capability("steering")
                    options = job["request"]["steering"]
                    layer_count = state.steering_layer_count()
                    layer = steering_comparison.resolve_layer(options["relative_depth"], layer_count)
                    response = state._steering_sham_control_local(
                        options["contrast_pairs"], item["test_text"],
                        [{"label": item["id"], "negative_text": item["negative_text"], "positive_text": item["positive_text"]}],
                        layer, options["alpha_magnitude"], 1.0, options["sham_samples"], options["seed"],
                    )
                    measurement = {"model_id": model_id, "response": response,
                                   "summary": steering_comparison.summarize_measurement(response, layer_count)}
                else:
                    a = state._continuation_surprisal_local(item["prefix_a"], item["continuation"], 3)
                    b = state._continuation_surprisal_local(item["prefix_b"], item["continuation"], 3)
                    measurement = {"model_id": model_id, "a": a, "b": b,
                                   "context_effect_bits": b["summary"]["total_continuation_surprisal_bits"] - a["summary"]["total_continuation_surprisal_bits"]}
                with state.state_lock:
                    job["measurements"][model_id][item["id"]] = measurement
                    job["completed_measurements"] += 1
        with state.state_lock:
            job["status"] = "cancelled" if job["cancel"].is_set() else "completed"
    except Exception as exc:
        with state.state_lock:
            job["status"] = "failed"
            job["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        with state.state_lock:
            job["phase"] = "restoring"
        if state.active_model_id != original or state.status != "ready":
            state._switch_model_local(original)
            if state.status != "ready":
                job["restoration_error"] = state.error
                job["errors"].append(f"Obnova původního modelu selhala: {state.error}")
                job["status"] = "failed"
        with state.state_lock:
            job["timing_ms"]["total"] = (time.perf_counter() - started) * 1000
            job["timing_ms"]["process_peak_rss_bytes"] = state.health()["performance"]["process_peak_rss_bytes"]
            try:
                record = make_record(job)
            except Exception as exc:
                job["errors"].append(f"Párování výsledků selhalo: {exc}")
                job["status"] = "failed"
                # Preserve raw measurements even if the paired renderer cannot be built.
                record = {
                    "schema": "apertus.experiment-record", "schema_version": 1, "id": job["id"], "kind": "model_comparison",
                    "captured_at": utc_now(), "annotation": {"title": job["request"]["title"]},
                    "capture": {"status": "failed", "warnings": job["errors"]}, "provenance": {"models": job["models"], "protocol": steering_comparison.PROTOCOL if "steering" in job["request"] else PROTOCOL},
                    "payload": {"request": job["request"], "measurements": job["measurements"], "comparisons": [],
                                "summary": {"paired_items": 0, "requested_items": len(job["request"]["items"]),
                                            "context_direction_agreement_count": 0, "context_direction_eligible_count": 0}},
                    "diagnostics": {"errors": job["errors"]},
                }
            job["record"] = record
            job["phase"] = "saving"
        try:
            if state.comparison_store is not None:
                state.comparison_store.save(record)
                job["saved"] = True
        except Exception as exc:
            job["storage_error"] = str(exc)
        with state.state_lock:
            job["phase"] = "done"


def new_job(request: dict[str, Any], server_instance_id: str) -> dict[str, Any]:
    import threading
    return {"id": str(uuid.uuid4()), "status": "queued", "phase": "queued", "request": request,
            "server_instance_id": server_instance_id, "cancel": threading.Event(), "models": {},
            "measurements": {}, "completed_measurements": 0, "total_measurements": 2 * len(request["items"]),
            "errors": [], "load_ms": {}, "timing_ms": {}, "saved": False}
