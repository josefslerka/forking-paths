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
        with (root / "server.log").open("w+") as log:
            process = subprocess.Popen([str(python), "-m", "apertus.garden", "demo", "--port", str(port),
                                        "--data-dir", str(root / "data")], cwd=root,
                                       env={**os.environ, "APERTUS_DATA_DIR":str(root / "data")},
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
                print(json.dumps({"status":"passed", "wheel":wheel.name, "isolated_venv":True,
                                  "inference_frameworks_installed":False, "model_weights_downloaded":False,
                                  "static_routes":5,"synthetic_probability_request":True}))
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
