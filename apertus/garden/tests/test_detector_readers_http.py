from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from apertus.garden.experiments import ExperimentStore
from apertus.garden.server import GardenServer, ModelState
from apertus.garden.tests.test_detector_readers import fake_detector


def test_memory_only_http_job_identity_validation_and_numeric_result(tmp_path, monkeypatch):
    state = ModelState(tmp_path, mock=True)
    fake_detector(state, monkeypatch)
    store = ExperimentStore(tmp_path / "http.sqlite3")
    server = GardenServer(("127.0.0.1", 0), state, store)
    state.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    def call(path, payload=None, origin=None, host=None):
        req = urllib.request.Request(base + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **({"Origin": origin} if origin else {}),
                     **({"Host": host} if host else {})})
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.load(response)
    try:
        deadline = time.monotonic() + 5
        health = call("/api/health")[1]
        while health["status"] != "ready" and time.monotonic() < deadline:
            time.sleep(.01)
            health = call("/api/health")[1]
        request = {"text": "Tento dokument se nesmí uložit.", "model_id": "mock",
                   "model_generation": health["model_generation"], "server_instance_id": health["server_instance_id"]}
        with pytest.raises(urllib.error.HTTPError) as stale:
            call("/api/detector-readers", {**request, "model_generation": 999})
        assert stale.value.code == 409
        assert json.load(stale.value)["code"] == "model_changed"
        with pytest.raises(urllib.error.HTTPError) as missing_identity:
            call("/api/detector-readers", {"text": "private document"})
        assert missing_identity.value.code == 400
        status, accepted = call("/api/detector-readers", request)
        assert status == 202 and "text" not in accepted
        path = f"/api/detector-readers/jobs/{accepted['id']}"
        while time.monotonic() < deadline:
            job = call(path)[1]
            if job["phase"] == "done":
                break
            time.sleep(.01)
        assert job["status"] == "completed"
        assert job["result"]["reader_comparison"]["summary"]["paired_windows"] == 3
        assert job["restored_identity"]["model_id"] == "mock"
        assert request["text"] not in json.dumps(job, ensure_ascii=False)
        assert call("/api/experiments")[1]["total"] == 0
        with pytest.raises(urllib.error.HTTPError) as unsafe:
            call(path, host="untrusted.invalid")
        assert unsafe.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as unsafe_cancel:
            call("/api/detector-readers/cancel", {"job_id": accepted["id"]}, origin="https://untrusted.invalid")
        assert unsafe_cancel.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as missing:
            call("/api/detector-readers/jobs/missing")
        assert missing.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        state.stop()
        state.worker_thread.join(3)
