#!/usr/bin/env python3
"""Run a paired batch against the same API used by the browser and save its record."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def call(base: str, path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(base.rstrip("/") + path, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--model-a", default="hybrid")
    parser.add_argument("--model-b", default="eurollm9")
    parser.add_argument("--items", type=Path, default=Path(__file__).with_name("model_comparison_examples.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Apertus × EuroLLM · průzkumné porovnání")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Výstup již existuje; zvol nový soubor pro nové měření.")
    data = json.loads(args.items.read_text())
    items = data["items"] if isinstance(data, dict) else data
    job = call(args.base_url, "/api/model-comparison", {"model_ids": [args.model_a, args.model_b], "items": items, "title": args.title})
    print(f"Job {job['id']}", flush=True)
    deadline = time.monotonic() + args.timeout
    previous = None
    try:
        while time.monotonic() < deadline:
            job = call(args.base_url, f"/api/model-comparison/jobs/{job['id']}")
            progress = (job["phase"], job.get("current_model_id"), job["completed_measurements"])
            if progress != previous:
                print(f"{progress[0]} · {progress[1]} · {progress[2]}/{job['total_measurements']}", flush=True)
                previous = progress
            if job["phase"] == "done":
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(job, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
                print(f"Saved {args.output}; archive saved={job['saved']}; status={job['status']}", flush=True)
                if job["status"] != "completed" or not job["saved"]:
                    raise SystemExit(1)
                return
            time.sleep(1)
        call(args.base_url, "/api/model-comparison/cancel", {"job_id": job["id"]})
        raise SystemExit("Časový limit; běh byl požádán o přerušení. Výsledek bude v archivu.")
    except KeyboardInterrupt:
        call(args.base_url, "/api/model-comparison/cancel", {"job_id": job["id"]})
        raise SystemExit("Požadavek na přerušení odeslán; dokončená měření zůstanou v archivu.")


if __name__ == "__main__":
    main()
