"""Install a wheel in a disposable, model-free venv and smoke-test HTTP routes.

This explicitly invokes pip (package network access may be needed). It never
downloads model weights and never uses the normal application archive.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import tempfile
import time
import venv
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def check_model_management(python: Path, root: Path, env: dict) -> None:
    """Exercise a real model-free first launch with only the installed wheel."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    data = (root / "first-launch-data").resolve()
    with (root / "first-launch.log").open("w+") as log:
        process = subprocess.Popen([str(python),"-m","apertus.garden","serve","--setup-only",
                                    "--port",str(port),"--data-dir",str(data)],
                                   cwd=root,env={**env,"APERTUS_DATA_DIR":str(data)},stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while True:
                try:
                    with urlopen(base + "/api/health",timeout=1) as response:
                        health = json.load(response)
                    if health["status"] == "setup_required":
                        break
                except (OSError,URLError):
                    pass
                if process.poll() is not None or time.monotonic() > deadline:
                    log.seek(0)
                    raise RuntimeError("Wheel first launch failed: " + log.read()[-3000:])
                time.sleep(.05)
            with urlopen(base + "/") as response:
                assert b'data-workbench-page="model-admin"' in response.read()
            for route in ("/static/model-admin.js","/static/model-admin.css"):
                with urlopen(base + route) as response:
                    assert response.status == 200 and len(response.read()) > 100
            with urlopen(base + "/api/model-admin") as response:
                settings = json.load(response)
            assert not settings["readonly"] and not settings["backend_installed"]
            model = root / "synthetic-local-model"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"gpt2","num_hidden_layers":2}')
            (model / "tokenizer.json").write_text("{}")
            weights = b"Synthetic file-preflight fixture, not numerical model weights"
            (model / "model.safetensors").write_bytes(weights)
            payload = {"action":"add","model_id":"wheel-local","label":"Wheel check",
                       "path":str(model),"registry_sha256":settings["registry_sha256"]}
            request = Request(base + "/api/model-admin/edit",data=json.dumps(payload).encode(),
                              headers={"Content-Type":"application/json","Origin":base})
            with urlopen(request) as response:
                saved = json.load(response)
            assert Path(saved["registry_path"]) == data / "models.toml"
            assert saved["registry_sha256"] == saved["applied_sha256"]
            assert any(item["id"] == "wheel-local" and item["files_ok"] for item in saved["models"])
            assert (model / "model.safetensors").read_bytes() == weights
            assert list((data / "model-registry-backups").glob("*.toml"))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="apertus-wheel-check-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", str(wheel)], check=True, cwd=root)
        subprocess.run([str(python), "-c", "import importlib.util; assert all(importlib.util.find_spec(name) is None for name in ('mlx','torch','transformers'))"], check=True, cwd=root)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        env = {key:value for key,value in os.environ.items()
               if key not in {"APERTUS_MODEL_REGISTRY","APERTUS_DATA_DIR","APERTUS_CACHE_DIR","PYTHONPATH","PYTHONHOME"}}
        env.update({"APERTUS_DATA_DIR":str(root / "data"),"APERTUS_CACHE_DIR":str(root / "cache")})
        with (root / "server.log").open("w+") as log:
            process = subprocess.Popen([str(python), "-m", "apertus.garden", "demo", "--port", str(port),
                                        "--data-dir", str(root / "data")], cwd=root,
                                       env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 15
                while True:
                    try:
                        with urlopen(base + "/api/health", timeout=1) as response:
                            health = json.load(response)
                        if health.get("status") == "ready":
                            break
                    except (OSError, URLError):
                        pass
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        raise RuntimeError("Wheel demo failed: " + log.read()[-3000:])
                    time.sleep(0.05)
                assert health["active_model"]["id"] == "mock", health
                for route in ("/", "/embeddingy", "/prekvapeni", "/static/workbench-shell.css", "/static/i18n.js"):
                    with urlopen(base + route) as response:
                        assert response.status == 200 and len(response.read()) > 100
                request = Request(base + "/api/next", data=json.dumps({"text":"Praha", "temperature":1.0,"top_k":5}).encode(),
                                  headers={"Content-Type":"application/json", "Origin":base})
                with urlopen(request) as response:
                    result = json.load(response)
                assert result["candidates"] and all(0 <= item["probability"] <= 1 for item in result["candidates"])
                check_model_management(python,root,env)
                print(json.dumps({"status":"passed", "wheel":wheel.name, "isolated_venv":True,
                                  "inference_frameworks_installed":False, "model_weights_downloaded":False,
                                  "static_routes":8,"synthetic_probability_request":True,
                                  "model_free_first_launch":True,"local_registration_and_backup":True}))
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
