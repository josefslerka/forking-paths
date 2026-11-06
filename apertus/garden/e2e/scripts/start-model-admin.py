"""Isolated administration E2E fixture. All weights/inference/downloads are fake.

Only this test launcher installs these doubles. The normal server has no
environment switch or API flag that enables them.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from apertus.garden import server, model_admin


def fixture_files(path, *, family="gpt2", layers=2, vocab=8):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type":family,"num_hidden_layers":layers,"vocab_size":vocab,"hidden_size":4096 if family=="apertus" else 16}))
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"E2E synthetic placeholder, NOT real model weights")


class FixtureTokenizer:
    all_special_ids = [0]
    def encode(self,text,**kwargs): return [0]+[1]*len(text.split())
    def decode(self,ids,**kwargs): return " ".join(f"token{i}" for i in ids)


class FixtureBackend:
    name="torch"
    def load(self,path,**kwargs):
        if (Path(path)/"fail-load").exists(): raise RuntimeError("Deliberate E2E load failure")
        config=json.loads((Path(path)/"config.json").read_text())
        return SimpleNamespace(parameters=lambda:iter([SimpleNamespace(dtype="float32")]),device="cpu",vocab_size=config["vocab_size"]),FixtureTokenizer()
    def synchronize(self): pass
    def clear_cache(self): pass
    def make_cache(self,model): return {}
    def last_logits(self,model,ids,cache): return np.linspace(-2,2,model.vocab_size,dtype=np.float32),cache


def simulated_setup(self,job,step,registry):
    for _index in range(35):
        self._cancelled(job)
        time.sleep(.05)
    target=registry.models[step["model_id"]].path
    fixture_files(target,family="apertus" if step["model_id"]=="bf16" else "llama",layers=32 if step["model_id"]=="bf16" else 42,vocab=131072 if step["model_id"]=="bf16" else 128000)
    self._log(job,"Synthetic E2E setup only. No network download or real weights.")


with tempfile.TemporaryDirectory(prefix="apertus-admin-e2e-") as temporary:
    directory=Path(temporary)
    os.environ["APERTUS_CACHE_DIR"]=str(directory/"cache")
    os.environ["APERTUS_DATA_DIR"]=str(directory/"data")
    fixture_files(directory/"cache/models/local-example")
    fixture_files(directory/"cache/models/bad-example")
    (directory/"cache/models/bad-example/fail-load").touch()
    registry=directory/"models.toml"
    registry.write_bytes(model_admin.registry_bytes({"schema_version":1,"backend":"torch","default_model":"draft","models":{"draft":{"label":"E2E not installed","path":"missing","adapter":"causal-lm"}}}))
    server.create_backend=lambda *args,**kwargs:FixtureBackend()
    model_admin.ModelAdmin._setup_child=simulated_setup
    # The fake installer must not depend on optional real download/inference stacks.
    model_admin.dependency_available=lambda module:True
    model_admin.shutil.disk_usage=lambda path:SimpleNamespace(free=10**12)
    sys.argv=["server","--registry",str(registry),"--data-dir",str(directory/"data"),"--port","8896","--setup-only"]
    server.main()
