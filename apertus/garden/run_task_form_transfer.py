"""Collect and derive the 180-item Apertus/EuroLLM task/form transfer locally.

Each model is released before the next is loaded. This process has no server,
database, UI, cloud or network side effects. Raw bundles are sealed by complete
SHA-256 manifests; reports can be regenerated without MLX via --derive-only.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

GARDEN = Path(__file__).resolve().parent
ROOT = GARDEN.parents[1]
sys.path.insert(0, str(ROOT))

import run_task_form_battery as parent
import task_form_transfer as transfer
from settings import cache_directory, data_directory

MODEL_PATHS = {
    "hybrid": cache_directory() / "models/8bit",
    "8bit": cache_directory() / "models/8bit",
    "bf16": cache_directory() / "models/bf16",
    "eurollm9": cache_directory() / "models/eurollm9",
}
MODEL_IDS = ("hybrid", "eurollm9")
SOURCES = ("run_task_form_transfer.py", "task_form_transfer.py", "run_task_form_battery.py",
           "task_form_analysis.py", "task_form_variance.py", "derive_task_form_battery.py",
           "derive_task_form_transfer.py", "domains/representations.py", "domains/steering.py",
           "server.py", "model_registry.py", "backends.py")


def prepare() -> tuple[dict, dict, list]:
    source = parent.load_json(parent.DEFAULT_PROTOCOL)
    stimuli = parent.expand_protocol(source)
    protocols = {model_id: transfer.build_protocol(source, model_id) for model_id in MODEL_IDS}
    tokenizations = {}
    for model_id in MODEL_IDS:
        tokenizer_path = MODEL_PATHS["bf16" if model_id == "hybrid" else model_id] / "tokenizer.json"
        tokenizer = parent.load_tokenizer(tokenizer_path)
        tokenizations[model_id] = parent.tokenize_stimuli(tokenizer, stimuli)
    lock = {
        "format": "apertus.task-form-transfer-lock", "schema_version": 1,
        "source_protocol_file_sha256": parent.sha256_file(parent.DEFAULT_PROTOCOL),
        "source_sha256": {str((GARDEN / name).relative_to(ROOT)): parent.sha256_file(GARDEN / name) for name in SOURCES},
        "protocols": protocols,
        "tokenizer_sha256": {model_id: parent.sha256_file(MODEL_PATHS[model_id] / "tokenizer.json") for model_id in MODEL_IDS},
        "model_config_sha256": {model_id: parent.sha256_file(MODEL_PATHS[model_id] / "config.json") for model_id in MODEL_IDS},
        "comparison_plan": {
            "status": "exploratory_cross_model_comparison; EuroLLM_window_fixed_before_its_residual_inspection",
            "primary_position": "response_boundary", "relative_window_rule": transfer.WINDOW_RULE,
            "difference": "EuroLLM minus Apertus, matched country means in each model's central window",
            "G_V_intervals": "5000 paired country-block percentile bootstrap replicates; fixed-set stability, no population inference",
            "alignment_difference": "descriptive mean and paired outer-country-deletion refits; no naive independent-score CI",
            "thresholds": "reuse parent within-model conjunction; no threshold on cross-model differences or transition layer",
            "numerics": "native residuals; float32 storage; native full-prefix final projection; reference atol=1e-5 rtol=0",
            "model_difference_interpretation": "architecture, training, tokenizer, size and quantization remain confounded",
        },
    }
    return lock, tokenizations, stimuli


def verify_inventory(bundle: Path) -> dict:
    manifest = parent.load_json(bundle / "manifest.json")
    if manifest.get("format") != "apertus.task-form-transfer-raw" or manifest.get("status") != "complete":
        raise parent.BatteryError("Raw přenos není úplný nebo má nepodporovaný formát.")
    actual = parent.inventory_files(bundle)
    if actual != manifest["files"]:
        raise parent.BatteryError("Raw přenos neodpovídá úplnému SHA-256 manifestu.")
    return manifest


def check_saved_item(bundle: Path, stimulus: dict, entry: dict, shape: list[int], tokenization: dict) -> dict:
    from safetensors.numpy import load_file

    stimulus_id = stimulus["stimulus_id"]
    metadata_path = bundle / "items" / f"{stimulus_id}.json"
    state_path = bundle / "states" / f"{stimulus_id}.safetensors"
    if parent.sha256_file(metadata_path) != entry["metadata_sha256"] or parent.sha256_file(state_path) != entry["state_sha256"]:
        raise parent.BatteryError(f"Změněný dokončený stimul {stimulus_id}.")
    metadata = parent.load_json(metadata_path)
    if metadata["stimulus"] != stimulus or metadata["tokenization"] != tokenization or metadata["state_shape"] != shape:
        raise parent.BatteryError(f"Jiný design nebo tokenizace stimulu {stimulus_id}.")
    arrays = load_file(str(state_path))
    if set(arrays) != {"states"}:
        raise parent.BatteryError(f"Jiné tensory v {stimulus_id}.")
    states = arrays["states"]
    if list(states.shape) != shape or states.dtype != np.float32 or not np.isfinite(states).all() or np.any(np.linalg.norm(states, axis=-1) == 0):
        raise parent.BatteryError(f"Neplatný tensor v {stimulus_id}.")
    return metadata


def collect_model(model_id: str, bundle: Path, lock: dict, tokenizations: dict, stimuli: list, resume: bool) -> None:
    from safetensors.numpy import save
    from apertus.garden.server import ModelState, encode_with_bos, RUNTIME_PROVENANCE
    from apertus.garden.domains.representations import trace_positions, readout
    import mlx.core as mx

    protocol = lock["protocols"][model_id]
    shape = [transfer.MODEL_DEPTHS[model_id] + 1, 2, 4096]
    lock_sha = parent.sha256_bytes(parent.canonical_json_bytes(lock))
    if bundle.exists():
        if not resume:
            raise parent.BatteryError(f"Raw přenos již existuje: {bundle}")
        progress = parent.load_json(bundle / "progress.json")
        if progress["lock_sha256"] != lock_sha:
            raise parent.BatteryError("Resume odmítá změnu protokolu nebo implementace.")
        for stimulus in stimuli:
            entry = progress["completed"].get(stimulus["stimulus_id"])
            if entry is not None:
                check_saved_item(bundle, stimulus, entry, shape, tokenizations[stimulus["stimulus_id"]])
        if (bundle / "manifest.json").exists():
            verify_inventory(bundle)
            print(json.dumps({"model": model_id, "phase": "already_complete"}), flush=True)
            return
    else:
        bundle.mkdir(parents=True)
        progress = {"lock_sha256": lock_sha, "model_id": model_id, "started_at": parent.utc_now(), "completed": {}}
        parent.atomic_write_json(bundle / "protocol.json", protocol)
        parent.atomic_write_json(bundle / "progress.json", progress)

    state = ModelState(MODEL_PATHS[model_id], model_paths=MODEL_PATHS, active_model_id=model_id)
    started = time.perf_counter()
    try:
        # Bound this standalone process's temporary Metal allocations while the
        # user's idle Workbench may retain its own model. No global app setting.
        mx.set_cache_limit(64 * 1024**2)
        mx.set_memory_limit(9 * 1024**3)
        if not state._model_available_unlocked(model_id, MODEL_PATHS[model_id]):
            raise parent.BatteryError(f"Neúplný místní model {model_id}.")
        state.load()
        if state.status != "ready":
            raise parent.BatteryError(state.error)
        descriptor = state.model_descriptor(model_id)
        if descriptor["layers"] != shape[0] - 1:
            raise parent.BatteryError("Model má jiný počet vrstev než uzamčený protokol.")
        identity = parent.model_identity(state)
        identity_path = bundle / "model.json"
        if identity_path.exists() and parent.load_json(identity_path)["identity"] != identity:
            raise parent.BatteryError("Resume odmítá změnu modelových vah.")
        if not identity_path.exists():
            parent.atomic_write_json(identity_path, {"identity": identity, "descriptor": descriptor,
                                                     "runtime": RUNTIME_PROVENANCE, "memory_limit_bytes": 9 * 1024**3})
        # Audit every prefix before any measured forward, not just the pilot.
        for stimulus in stimuli:
            ids = list(encode_with_bos(state.tokenizer, stimulus["prompt"]))
            if ids != tokenizations[stimulus["stimulus_id"]]["prompt_token_ids"]:
                raise parent.BatteryError("Runtime tokenizer nesouhlasí s předběžným auditem.")
        rust_tokenizer = parent.load_tokenizer(MODEL_PATHS[model_id] / "tokenizer.json")
        print(json.dumps({"model": model_id, "phase": "loaded_and_audited", "items": len(stimuli)}), flush=True)
        for index, stimulus in enumerate(stimuli, 1):
            stimulus_id = stimulus["stimulus_id"]
            if stimulus_id in progress["completed"]:
                continue
            trace = tokenizations[stimulus_id]
            ids = trace["prompt_token_ids"]
            item_started = time.perf_counter()
            body, native_states, hidden = trace_positions(state.model, ids, trace["anchor_token_indices"])
            logits = readout(state.model, body, hidden)[0, -1].astype(mx.float32)
            states = native_states.astype(mx.float32)
            reference = state.model(mx.array([ids]))[0, -1].astype(mx.float32)
            mx.eval(states, logits, reference)
            stored = np.ascontiguousarray(np.asarray(states), dtype=np.float32)
            projected, independent = np.asarray(logits), np.asarray(reference)
            error = float(np.max(np.abs(projected - independent)))
            np.testing.assert_allclose(projected, independent, atol=1e-5, rtol=0)
            if list(stored.shape) != shape or not np.isfinite(stored).all():
                raise parent.BatteryError("Neplatný tvar nebo hodnoty nativních reziduálů.")
            prediction = parent.summarize_final_distribution(logits=projected,
                expected_first_token_id=trace["target"]["first_token_id"], tokenizer=rust_tokenizer, top_k=8)
            prediction["source"] = f"L{shape[0]-1}_native_full_prefix_final_norm_and_head"
            state_path = bundle / "states" / f"{stimulus_id}.safetensors"
            parent.atomic_write(state_path, save({"states": stored}, metadata={"axes": "layer,position,hidden", "positions": "country_end,response_boundary"}))
            metadata = {"model_id": model_id, "stimulus": stimulus, "tokenization": trace,
                        "captured_at": parent.utc_now(),
                        "state_shape": shape, "state_sha256": parent.sha256_file(state_path),
                        "native_final_max_absolute_logit_error": error, "final_distribution": prediction,
                        "forward_and_verification_seconds": time.perf_counter() - item_started,
                        "timing_ms": {"forward_and_projection_ms": (time.perf_counter() - item_started) * 1000}}
            metadata_path = bundle / "items" / f"{stimulus_id}.json"
            parent.atomic_write_json(metadata_path, metadata)
            progress["completed"][stimulus_id] = {"state_sha256": metadata["state_sha256"], "metadata_sha256": parent.sha256_file(metadata_path),
                "metadata_file": f"items/{stimulus_id}.json", "state_file": f"states/{stimulus_id}.safetensors"}
            parent.atomic_write_json(bundle / "progress.json", progress)
            if index % 18 == 0:
                print(json.dumps({"model": model_id, "completed": index, "total": len(stimuli), "max_logit_error": error}), flush=True)
        if state.active_ids is not None or state.active_cache is not None:
            raise parent.BatteryError("Sběr změnil KV cache zahrady.")
        parent.verify_model_identity_unchanged(state, identity)
        for name, expected in lock["source_sha256"].items():
            if parent.sha256_file(ROOT / name) != expected:
                raise parent.BatteryError("Implementace se během měření změnila.")
        for stimulus in stimuli:
            check_saved_item(bundle, stimulus, progress["completed"][stimulus["stimulus_id"]], shape, tokenizations[stimulus["stimulus_id"]])
        progress.update(status="complete", completed_at=parent.utc_now(), session_elapsed_seconds=time.perf_counter() - started)
        parent.atomic_write_json(bundle / "progress.json", progress)
        parent.atomic_write_json(bundle / "manifest.json", {"format": "apertus.task-form-transfer-raw", "schema_version": 1,
            "status": "complete", "model_id": model_id, "lock_sha256": lock_sha, "state_shape": shape,
            "stimulus_ids": [item["stimulus_id"] for item in stimuli], "files": parent.inventory_files(bundle)})
        verify_inventory(bundle)
    finally:
        state._release_model_local()
        del state
        gc.collect()
        mx.synchronize()
        mx.clear_cache()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=data_directory() / "task_form_transfers")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--derive-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    raw = args.raw_root.resolve() / output.name
    if not parent.SERIES_RE.fullmatch(output.name):
        raise parent.BatteryError("Neplatné označení série.")
    if args.derive_only:
        from derive_task_form_transfer import derive
        derive(output, raw)
        return 0
    if args.validate_only:
        source = parent.load_json(parent.DEFAULT_PROTOCOL)
        stimuli = parent.expand_protocol(source)
        protocols = {model_id: transfer.build_protocol(source, model_id) for model_id in MODEL_IDS}
        print(json.dumps({"valid": True, "stimuli_per_model": len(stimuli), "mlx_or_model_loaded": False,
            "windows": {key: value["design"]["primary_layers"] for key, value in protocols.items()},
            "tokenization": "not_checked_without_model_assets"},ensure_ascii=False))
        return 0
    lock, tokenizations, stimuli = prepare()
    lock_path = output / "protocol_lock.json"
    if lock_path.exists():
        if not args.resume or parent.load_json(lock_path) != lock:
            raise parent.BatteryError("Série existuje; resume vyžaduje stejný uzamčený protokol i kód.")
    else:
        if output.exists() or raw.exists():
            raise parent.BatteryError("Vyber nový výstupní adresář pro neexistující sérii.")
        output.mkdir(parents=True)
        parent.atomic_write_json(lock_path, lock)
        parent.atomic_write_json(output / "tokenizations.json", tokenizations)
    for model_id in MODEL_IDS:
        collect_model(model_id, raw / model_id, lock, tokenizations[model_id], stimuli, args.resume)
    from derive_task_form_transfer import derive
    derive(output, raw)
    print(json.dumps({"complete": True, "stimuli_per_model": len(stimuli), "output": str(output)},ensure_ascii=False),flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except parent.BatteryError as exc:
        raise SystemExit(str(exc)) from exc
