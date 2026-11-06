from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from apertus.garden.experiments import ExperimentStore
from apertus.garden.model_admin import registry_bytes
from apertus.garden.registry_config import load_registry
from apertus.garden.server import GardenServer, ModelState


def test_admin_transport_guards_registry_writes_and_serves_first_launch(tmp_path):
    path=tmp_path / "models.toml"
    path.write_bytes(registry_bytes({"schema_version":1,"backend":"torch","default_model":"draft","models":{"draft":{"path":"missing"}}}))
    registry=load_registry(path)
    state=ModelState(registry.models["draft"].path,registry=registry,model_paths=registry.model_paths(),active_model_id="draft",backend_name="torch")
    server=GardenServer(("127.0.0.1",0),state,ExperimentStore(tmp_path / "archive.sqlite3"),model_admin_data_dir=tmp_path / "data")
    state.start(load_initial=False)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f"http://127.0.0.1:{server.server_address[1]}"
    def call(route, data=None, headers=None):
        request=Request(base+route,data=None if data is None else json.dumps(data).encode(),headers={"Content-Type":"application/json",**(headers or {})})
        return urlopen(request,timeout=5)
    try:
        with call("/") as response:
            assert b'data-workbench-page="model-admin"' in response.read()
        with call("/static/model-admin.css") as response:
            assert "text/css" in response.headers["Content-Type"]
        with call("/api/model-admin") as response:
            snapshot=json.load(response)
        assert snapshot["status"]=="setup_required" and not snapshot["readonly"]
        original=path.read_bytes()
        with pytest.raises(HTTPError) as error:
            call("/api/model-admin",headers={"Host":"attacker.example"})
        assert error.value.code==403
        with pytest.raises(HTTPError) as error:
            call("/api/model-admin/edit",{"action":"remove","model_id":"draft","registry_sha256":snapshot["registry_sha256"]},headers={"Origin":"https://attacker.example"})
        assert error.value.code==403 and path.read_bytes()==original
        with pytest.raises(HTTPError) as error:
            call("/api/model-admin/edit",{"action":"set_default","model_id":"draft","registry_sha256":"stale"})
        assert error.value.code==409
        assert json.load(error.value)["code"]=="registry_conflict"
        with pytest.raises(HTTPError) as error:
            call("/api/model-admin/jobs/no-such-job")
        assert error.value.code==404
        assert path.read_bytes()==original and not (tmp_path / "data/model-registry-backups").exists()
    finally:
        server.shutdown();server.server_close();thread.join(3);state.stop();state.worker_thread.join(3)
