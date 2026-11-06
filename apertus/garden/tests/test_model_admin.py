from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apertus.garden.model_admin import AdminError, Cancelled, ModelAdmin, registry_bytes
from apertus.garden.registry_config import load_registry
from apertus.garden.server import ModelState


def local_files(path: Path, *, family="gpt2"):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type":family,"num_hidden_layers":2,"vocab_size":8}))
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"synthetic test fixture, not real weights")


class TestTokenizer:
    all_special_ids = [0]
    def encode(self, text, **kwargs):
        return [0] + [1] * len(text.split())
    def decode(self, ids, **kwargs):
        return " ".join(f"token{i}" for i in ids)


class TestBackend:
    name = "torch"
    def load(self, path, **kwargs):
        if (Path(path) / "fail-load").exists():
            raise RuntimeError("Synthetic load failure")
        model = SimpleNamespace(parameters=lambda:iter([SimpleNamespace(dtype="float32")]),device="cpu")
        return model, TestTokenizer()
    def synchronize(self): pass
    def clear_cache(self): pass
    def make_cache(self, model): return {}
    def last_logits(self, model, ids, cache): return np.arange(8,dtype=np.float32), cache


@pytest.fixture
def managed(tmp_path, monkeypatch):
    # This fixture supplies its own numerical backend, not optional libraries.
    monkeypatch.setattr("apertus.garden.model_admin.dependency_available", lambda module: True)
    for name in ("one","two"):
        local_files(tmp_path / name)
    path = tmp_path / "models.toml"
    path.write_bytes(registry_bytes({"schema_version":1,"backend":"torch","default_model":"one","models":{
        "one":{"path":"one","label":"First","adapter":"causal-lm"},"two":{"path":"two","label":"Second","adapter":"causal-lm"}}}))
    registry = load_registry(path)
    state = ModelState(registry.models["one"].path, registry=registry, model_paths=registry.model_paths(), active_model_id="one", backend_name="torch")
    state.backend = TestBackend()
    state.start(load_initial=False)
    admin = ModelAdmin(state,tmp_path / "data")
    yield admin, state, path
    admin.close();state.stop();state.worker_thread.join(5)


def edit(admin, action, model_id, **extra):
    return admin.edit({"action":action,"model_id":model_id,"registry_sha256":admin.snapshot()["registry_sha256"],**extra})


def wait_job(admin, job):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = admin.job_status(job["id"])
        if result["phase"] == "done" and admin.state.outstanding_operation is None:
            return result
        time.sleep(.01)
    pytest.fail("Model administration job did not finish")


def test_setup_mode_does_not_import_or_load_backend_and_supports_health(managed):
    admin,state,path=managed
    assert state.health()["status"] == "setup_required"
    assert state.model is None
    assert "model-admin" in state.health()["capabilities"]
    assert admin.snapshot()["readonly"] is False


def test_local_add_is_validated_applied_and_backed_up_without_copying_weights(managed,tmp_path):
    admin,state,path=managed
    original=path.read_bytes()
    local_files(tmp_path / "third")
    result=admin.inspect({"path":str(tmp_path / "third")})
    assert result["files_ok"] and result["adapter"]=="causal-lm"
    saved=edit(admin,"add","third",path=str(tmp_path / "third"),label="Third")
    assert "third" in state.model_paths
    assert load_registry(path).models["third"].path == tmp_path / "third"
    assert len(list((tmp_path / "data/model-registry-backups").glob("*.toml")))==1
    assert next((tmp_path / "data/model-registry-backups").glob("*.toml")).read_bytes()==original
    assert saved["registry_sha256"]==state.registry.sha256
    assert not any((tmp_path / "data").glob("*.safetensors"))


def test_optimistic_lock_rejects_stale_edits_and_keeps_manual_content(managed):
    admin,state,path=managed
    old=admin.snapshot()["registry_sha256"]
    path.write_text(path.read_text()+"# manual edit\n")
    before=path.read_bytes()
    with pytest.raises(AdminError,match="Registry changed"):
        admin.edit({"action":"set_default","model_id":"two","registry_sha256":old})
    assert path.read_bytes()==before
    with pytest.raises(AdminError) as error:
        edit(admin,"set_default","two")
    assert error.value.code=="reload_required"
    admin.reload({"registry_sha256":admin.snapshot()["registry_sha256"]})
    assert edit(admin,"set_default","two")["applied_sha256"]==state.registry.sha256


def test_hide_restore_default_and_disconnect_never_delete_model_files(managed):
    admin,state,path=managed
    edit(admin,"set_enabled","two",enabled=False)
    assert "two" not in state.model_paths
    assert path.parent.joinpath("two/model.safetensors").is_file()
    edit(admin,"set_enabled","two",enabled=True)
    edit(admin,"set_default","two")
    edit(admin,"remove","one")
    assert "one" not in load_registry(path).models
    assert path.parent.joinpath("one/model.safetensors").is_file()


def test_default_and_active_registration_are_protected(managed):
    admin,state,path=managed
    with pytest.raises(AdminError) as error:
        edit(admin,"remove","one")
    assert error.value.code=="default_model"
    state.load()
    with pytest.raises(AdminError) as error:
        edit(admin,"set_enabled","one",enabled=False)
    assert error.value.code=="active_model"


def test_core_smoke_loads_one_model_and_restores_previous_after_success_and_failure(managed):
    admin,state,path=managed
    state.load()
    old_generation=state.model_generation
    job=admin.start_job({"action":"smoke","model_id":"two","registry_sha256":admin.snapshot()["registry_sha256"]})
    result=wait_job(admin,job)
    assert result["status"]=="completed",result
    assert result["result"]["passed"] and state.active_model_id=="one" and state.status=="ready"
    assert state.model_generation > old_generation
    path.parent.joinpath("two/fail-load").touch()
    job=admin.start_job({"action":"smoke","model_id":"two","registry_sha256":admin.snapshot()["registry_sha256"]})
    result=wait_job(admin,job)
    assert result["status"]=="failed" and result["error_code"]=="load_failed"
    assert state.active_model_id=="one" and state.status=="ready"


def test_smoke_from_first_run_keeps_successful_model_loaded(managed):
    admin,state,path=managed
    job=admin.start_job({"action":"smoke","model_id":"two","registry_sha256":admin.snapshot()["registry_sha256"]})
    assert wait_job(admin,job)["status"]=="completed"
    assert state.active_model_id=="two" and state.status=="ready"
    assert next(model for model in admin.snapshot()["models"] if model["id"]=="two")["check"]["passed"]


def test_invalid_directory_and_unsupported_adapter_do_not_edit_registry(managed,tmp_path):
    admin,state,path=managed
    before=path.read_bytes()
    with pytest.raises(AdminError) as error:
        edit(admin,"add","bad",path=str(tmp_path),label="Bad")
    assert error.value.code=="invalid_model_files"
    with pytest.raises(AdminError) as error:
        admin.inspect({"path":str(tmp_path / "one"),"adapter":"apertus"})
    assert error.value.code=="invalid_adapter" and path.read_bytes()==before


def test_curated_plan_is_explicit_pinned_and_does_not_touch_disk(managed,tmp_path,monkeypatch):
    admin,state,path=managed
    monkeypatch.setenv("APERTUS_CACHE_DIR",str(tmp_path / "cache"))
    monkeypatch.setattr("apertus.garden.model_admin.shutil.disk_usage",lambda path:SimpleNamespace(free=10**12))
    before=set(tmp_path.iterdir())
    plan=admin.plan_install({"profile_id":"bf16","registry_sha256":admin.snapshot()["registry_sha256"]})
    assert plan["estimated_bytes"]==17_000_000_000
    assert len(plan["steps"][0]["revision"])==40
    assert set(tmp_path.iterdir())==before
    with pytest.raises(AdminError) as error:
        admin.start_job({"action":"install","plan_id":plan["id"],"registry_sha256":plan["registry_sha256"]})
    assert error.value.code=="confirmation_required"


def test_hybrid_plan_resolves_dependency_graph_before_validation(tmp_path,monkeypatch):
    monkeypatch.setenv("APERTUS_CACHE_DIR",str(tmp_path / "cache"))
    monkeypatch.setattr("apertus.garden.model_admin.shutil.disk_usage",lambda path:SimpleNamespace(free=10**12))
    path=tmp_path / "models.toml"
    path.write_bytes(registry_bytes({"schema_version":1,"backend":"mlx","default_model":"local","models":{"local":{"path":"missing"}}}))
    registry=load_registry(path)
    state=ModelState(registry.models["local"].path,registry=registry,model_paths=registry.model_paths(),active_model_id="local",status="setup_required")
    admin=ModelAdmin(state,tmp_path / "data")
    plan=admin.plan_install({"profile_id":"hybrid","registry_sha256":registry.sha256})
    assert [step["model_id"] for step in plan["steps"]]==["bf16","8bit","hybrid"]
    assert plan["estimated_bytes"]==26_000_000_000
    assert not (tmp_path / "cache").exists()


def test_demo_admin_is_read_only_and_does_not_write_user_registry(tmp_path):
    state=ModelState(tmp_path,mock=True)
    admin=ModelAdmin(state,tmp_path / "data")
    with pytest.raises(AdminError) as error:
        admin.edit({"action":"remove","model_id":"bf16"})
    assert error.value.code=="demo_readonly" and not (tmp_path / "data").exists()


def test_cancelling_child_cleans_only_fresh_owned_workspace(managed,monkeypatch,tmp_path):
    admin,state,path=managed
    real_popen=subprocess.Popen
    def child(*args,**kwargs):
        temporary=args[0][-1]
        script='import pathlib,time,sys; pathlib.Path(sys.argv[1],"partial").write_bytes(b"partial"); print("starting",flush=True); time.sleep(30)'
        return real_popen([sys.executable,"-u","-c",script,temporary],**kwargs)
    monkeypatch.setattr("apertus.garden.model_admin.subprocess.Popen",child)
    job={"cancel":threading.Event(),"logs":[]}
    timer=threading.Timer(.25,job["cancel"].set);timer.start()
    before=(tmp_path / "two/model.safetensors").read_bytes()
    try:
        with pytest.raises(Cancelled):
            admin._setup_child(job,{"model_id":"two","operation":"download"},state.registry)
    finally:
        timer.cancel()
    assert admin.child is None
    assert not list(tmp_path.glob(".workbench-install-*"))
    assert (tmp_path / "two/model.safetensors").read_bytes()==before


@pytest.mark.skipif(os.name != "posix", reason="MLX conversion process groups are POSIX-only")
def test_cancellation_also_stops_owned_conversion_grandchildren(managed,monkeypatch):
    admin,state,path=managed
    real_popen=subprocess.Popen
    grandchild_pid=[]
    def child(*args,**kwargs):
        script='import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-u","-c","import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print(\\\"grandchild ready\\\",flush=True); time.sleep(30)"]); print("grandchild pid",p.pid,flush=True); time.sleep(30)'
        return real_popen([sys.executable,"-u","-c",script],**kwargs)
    monkeypatch.setattr("apertus.garden.model_admin.subprocess.Popen",child)
    original_log=admin._log
    job={"cancel":threading.Event(),"logs":[]}
    def record(current,line):
        original_log(current,line)
        if line.startswith("grandchild pid "):
            grandchild_pid.append(int(line.rsplit(" ",1)[1]))
        if line == "grandchild ready":
            job["cancel"].set()
    monkeypatch.setattr(admin,"_log",record)
    try:
        with pytest.raises(Cancelled):
            admin._setup_child(job,{"model_id":"two","operation":"convert"},state.registry)
        assert grandchild_pid
        monkeypatch.setattr("apertus.garden.model_admin.subprocess.Popen",real_popen)
        for _attempt in range(50):
            result=subprocess.run(["ps","-p",str(grandchild_pid[0]),"-o","stat="],capture_output=True,text=True)
            if result.returncode or result.stdout.strip().startswith("Z"):
                break
            time.sleep(.02)
        assert result.returncode or result.stdout.strip().startswith("Z"), result.stdout
    finally:
        if grandchild_pid:
            try: os.kill(grandchild_pid[0],signal.SIGKILL)
            except ProcessLookupError: pass


def test_installation_requires_fresh_confirmation_and_blocks_other_edits(managed,monkeypatch,tmp_path):
    admin,state,path=managed
    monkeypatch.setenv("APERTUS_CACHE_DIR",str(tmp_path / "cache"))
    monkeypatch.setattr("apertus.garden.model_admin.shutil.disk_usage",lambda path:SimpleNamespace(free=10**12))
    monkeypatch.setattr("apertus.garden.model_admin.dependency_available",lambda module:True)
    def fake_setup(job,step,registry):
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            admin._cancelled(job);time.sleep(.01)
        local_files(registry.models[step["model_id"]].path,family="apertus")
    monkeypatch.setattr(admin,"_setup_child",fake_setup)
    plan=admin.plan_install({"profile_id":"bf16","registry_sha256":admin.snapshot()["registry_sha256"]})
    job=admin.start_job({"action":"install","plan_id":plan["id"],"registry_sha256":plan["registry_sha256"],"confirmed":True})
    with pytest.raises(AdminError) as error:
        edit(admin,"set_default","two")
    assert error.value.code=="model_busy"
    admin.job_status(job["id"],cancel=True)
    assert wait_job(admin,job)["status"]=="cancelled"
    assert state.status=="setup_required" and (tmp_path / "one/model.safetensors").exists()


def test_missing_download_dependencies_do_not_change_the_registry(managed,monkeypatch,tmp_path):
    admin,state,path=managed
    monkeypatch.setenv("APERTUS_CACHE_DIR",str(tmp_path / "cache"))
    monkeypatch.setattr("apertus.garden.model_admin.shutil.disk_usage",lambda path:SimpleNamespace(free=10**12))
    monkeypatch.setattr("apertus.garden.model_admin.dependency_available",lambda module:False)
    plan=admin.plan_install({"profile_id":"bf16","registry_sha256":state.registry.sha256})
    original=path.read_bytes()
    with pytest.raises(AdminError) as error:
        admin.start_job({"action":"install","plan_id":plan["id"],"registry_sha256":plan["registry_sha256"],"confirmed":True})
    assert error.value.code=="dependency_missing"
    assert path.read_bytes()==original and not admin.jobs and not (tmp_path / "cache").exists()


def test_saving_factory_presets_creates_a_user_registry_not_a_package_edit(tmp_path,monkeypatch):
    monkeypatch.setenv("APERTUS_CACHE_DIR",str(tmp_path / "cache"))
    registry=load_registry(Path(__file__).parents[1] / "model_presets.toml")
    original=registry.path.read_bytes()
    state=ModelState(registry.models["bf16"].path,registry=registry,model_paths=registry.model_paths(),active_model_id="bf16",status="setup_required")
    admin=ModelAdmin(state,tmp_path / "data")
    local_files(tmp_path / "local")
    edit(admin,"add","own",path=str(tmp_path / "local"),label="Local")
    assert state.registry.path==tmp_path / "data/models.toml"
    assert registry.path.read_bytes()==original


def test_hidden_hybrid_sources_stay_registered_and_default_cannot_be_hidden(tmp_path):
    path=tmp_path / "models.toml"
    path.write_bytes(registry_bytes({"schema_version":1,"backend":"mlx","default_model":"hybrid","models":{
        "bf16":{"path":"bf16","adapter":"apertus"},"8bit":{"path":"q8","adapter":"apertus"},"hybrid":{"kind":"apertus-hybrid","adapter":"apertus"}}}))
    registry=load_registry(path)
    state=ModelState(registry.models["hybrid"].path,registry=registry,model_paths=registry.model_paths(),active_model_id="hybrid",status="setup_required")
    admin=ModelAdmin(state,tmp_path / "data")
    with pytest.raises(AdminError) as error:
        edit(admin,"remove","bf16")
    assert error.value.code=="model_dependency"
    edit(admin,"set_enabled","bf16",enabled=False)
    assert "bf16" in state.registry.models and "bf16" not in state.model_paths and "hybrid" in state.model_paths
    with pytest.raises(ValueError):
        edit(admin,"set_enabled","hybrid",enabled=False)


@pytest.mark.parametrize("model_id",["mock-user","bad/id","hybrid","<script>"])
def test_unsafe_or_reserved_model_ids_never_rewrite_configuration(managed,tmp_path,model_id):
    admin,state,path=managed
    before=path.read_bytes()
    with pytest.raises(ValueError):
        edit(admin,"add",model_id,path=str(tmp_path / "two"),label="Invalid ID")
    assert path.read_bytes()==before
