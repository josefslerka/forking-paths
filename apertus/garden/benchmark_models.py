#!/usr/bin/env python3
"""Matched workbench API timings, with model restoration and explicit cache states."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

try:
    from .run_model_comparison import call
except ImportError:
    from run_model_comparison import call

TEXTS = {
    "short": "Mám rád Prahu.",
    "medium": "Výzkumník porovnal dvě věty ve stejném kontextu. Zajímalo ho, zda změna jediného slova posune očekávání podobným směrem v obou modelech.",
    "long": "Výzkumník porovnal dvě věty ve stejném kontextu. Zajímalo ho, zda změna jediného slova posune očekávání podobným směrem v obou modelech. " * 3,
}


def switch(base: str, model_id: str) -> dict:
    accepted = call(base, "/api/model", {"model_id": model_id})
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        health = call(base, "/api/health")
        if health["status"] == "error":
            raise RuntimeError(health["error"])
        if health["status"] == "ready" and health["active_model"]["id"] == model_id and not health["worker"]["busy"]:
            return {"health": health, "accepted": accepted.get("accepted", False)}
        time.sleep(.25)
    raise RuntimeError("Model switch timed out")


def benchmark(base: str, model_ids: list[str], repeats: int = 3) -> dict:
    initial = call(base, "/api/health")
    if initial["status"] != "ready" or initial["worker"]["busy"]:
        raise RuntimeError("Workbench must be ready and idle")
    original = initial["active_model"]["id"]
    result = {"protocol": "workbench-api-benchmark-v1", "repeats": repeats, "texts": TEXTS,
              "notes": ["Same visible text; native token counts may differ.",
                        "Cold prefix means a logit-cache miss, not a filesystem-cold model load.",
                        "Full-text timing uses one warm-up then repeated cache-free forwards.",
                        "RSS is a lifetime process peak, not per-model GPU memory.",
                        "Incremental branch consumes one native predicted token per model."], "models": {}}
    try:
        for model_id in model_ids:
            print(f"Benchmark {model_id}", flush=True)
            started = time.perf_counter()
            transition = switch(base, model_id)
            health = transition["health"]
            model = {"descriptor": health["active_model"], "runtime": health["provenance"],
                     "model_switch_wall_ms": (time.perf_counter() - started) * 1000,
                     "model_switch_performed": transition["accepted"], "measurements": [], "summary": {}}
            result["models"][model_id] = model
            identity = {"model_id": model_id, "model_generation": health["model_generation"], "server_instance_id": health["server_instance_id"]}
            for text_id, text in TEXTS.items():
                text = text.rstrip()
                for repetition in range(repeats + 1):
                    start = time.perf_counter()
                    scored = call(base, "/api/text-surprisal", {"text": text, "top_k": 3, **identity})
                    model["measurements"].append({"mode": "text_surprisal", "text_id": text_id, "warmup": repetition == 0,
                                                  "wall_ms": (time.perf_counter() - start) * 1000, "server_timing_ms": scored["timing_ms"],
                                                  "token_count": scored["token_count"], "character_count": len(text)})
            for repetition in range(repeats):
                prefix = f"Ve zkušebním příkladu číslo {repetition + 1} je hlavním městem České republiky"
                def query(mode: str, text: str, ids: list[int] | None = None) -> dict:
                    payload = {"text": text, "top_k": 5, "temperature": 1, **identity}
                    if ids is not None:
                        payload["token_ids"] = ids
                    start = time.perf_counter()
                    output = call(base, "/api/next", payload)
                    model["measurements"].append({"mode": mode, "wall_ms": (time.perf_counter() - start) * 1000,
                                                  "server_ms": output["latency_ms"], "cached": output["cached"], "token_count": output["token_count"]})
                    return output
                root = query("prefix_cache_miss", prefix)
                cached = query("cached_prefix", prefix)
                if not cached["cached"]:
                    raise RuntimeError("Warm prefix was not cached")
                candidate = root["candidates"][0]
                query("incremental_token", prefix + candidate.get("text", ""), [*root["token_ids"], candidate["token_id"]])
            for mode in ("prefix_cache_miss", "cached_prefix", "incremental_token"):
                values = [row["wall_ms"] for row in model["measurements"] if row["mode"] == mode]
                model["summary"][mode] = {"median_wall_ms": statistics.median(values), "count": len(values)}
            for text_id in TEXTS:
                rows = [row for row in model["measurements"] if row["mode"] == "text_surprisal" and row["text_id"] == text_id and not row["warmup"]]
                model["summary"][f"text_{text_id}"] = {"median_wall_ms": statistics.median(row["wall_ms"] for row in rows),
                                                          "median_forward_ms": statistics.median(row["server_timing_ms"]["forward"] for row in rows),
                                                          "token_count": rows[0]["token_count"]}
            model["process_peak_rss_bytes"] = call(base, "/api/health")["performance"]["process_peak_rss_bytes"]
    finally:
        switch(base, original)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--models", nargs=2, default=["hybrid", "eurollm9"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists; use a new filename")
    if not 1 <= args.repeats <= 10:
        raise SystemExit("repeats must be 1–10")
    result = benchmark(args.base_url, args.models, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
