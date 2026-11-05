from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

from apertus.garden.experiments import ExperimentStore
from apertus.garden.server import GardenServer, ModelState


def test_http_job_archive_and_reload_use_one_immutable_record(tmp_path) -> None:
    state = ModelState(tmp_path, mock=True)
    store = ExperimentStore(tmp_path / "http.sqlite3")
    server = GardenServer(("127.0.0.1", 0), state, store)
    state.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    def call(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.load(response)
    try:
        deadline = time.monotonic() + 3
        while call("/api/health")["status"] != "ready" and time.monotonic() < deadline:
            time.sleep(.01)
        accepted = call("/api/model-comparison", {"model_ids": ["mock", "mock-eurollm9"], "items": [{"id": "same-input", "text": "Mám rád Prahu."}]})
        while time.monotonic() < deadline:
            job = call(f"/api/model-comparison/jobs/{accepted['id']}")
            if job["phase"] == "done":
                break
            time.sleep(.01)
        assert job["status"] == "completed" and job["saved"]
        record = call(f"/api/experiments/{accepted['id']}")["record"]
        assert record["payload"] == job["record"]["payload"]
        reopened = ExperimentStore(tmp_path / "http.sqlite3")
        assert reopened.get(accepted["id"])["payload"] == record["payload"]
        health = call("/api/health")
        assert health["active_model"]["id"] == "mock"
        with urllib.request.urlopen(base + "/porovnani-modelu") as response:
            assert b"comparison-form" in response.read()
        with urllib.request.urlopen(base + "/static/steering-comparison.js") as response:
            assert "javascript" in response.headers["Content-Type"]
            assert b"ProbabilitySteeringComparison" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        state.stop()
        state.worker_thread.join(3)
