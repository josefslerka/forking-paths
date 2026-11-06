from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from apertus.garden.experimental_addons import AddonError
from apertus.garden.model_admin import ModelAdmin, registry_bytes
from apertus.garden.predictive_detector import verify_legacy_runtime_code
from apertus.garden.registry_config import load_registry
from apertus.garden.server import ModelState
from apertus.garden.tuned_lens import TunedLensCatalog
from test_tuned_lens import artifact_manifest
from test_jacobian_lens import fixture as jacobian_fixture


@pytest.fixture
def managed_addons(tmp_path):
    for name in ("bf16", "other"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps({"model_type":"apertus","num_hidden_layers":32,"hidden_size":4096,"vocab_size":131072}))
        (folder / "tokenizer.json").write_text("{}")
        (folder / "model.safetensors").write_bytes(b"synthetic model preflight, not real weights")
    path = tmp_path / "models.toml"
    path.write_bytes(registry_bytes({"schema_version":1,"backend":"mlx","default_model":"bf16","models":{
        name:{"path":name,"adapter":"apertus"} for name in ("bf16","other")}}))
    registry = load_registry(path)
    state = ModelState(registry.models["bf16"].path,registry=registry,model_paths=registry.model_paths(),active_model_id="bf16")
    state.start(load_initial=False)
    admin = ModelAdmin(state,tmp_path / "data")
    yield admin,state
    admin.close(); state.stop(); state.worker_thread.join(5)


def make_reader(admin, root, **overrides):
    folder = root / "reader-fixture"
    folder.mkdir()
    manifest = artifact_manifest(folder,artifact_id=folder.name,model_id="bf16",
                                 metadata_fingerprint=admin.addons.model_identity("bf16")["metadata_fingerprint"],**overrides)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def edit(admin, action, kind="tuned", **extra):
    return admin.addon_action({"action":action,"kind":kind,"addon_revision":admin.addons.revision,
                               "registry_sha256":admin.state.registry.sha256,**extra})


def verify(admin, kind="tuned"):
    job=admin.start_job({"action":"addon_verify","kind":kind,"addon_revision":admin.addons.revision,
                         "registry_sha256":admin.state.registry.sha256})
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        result=admin.job_status(job["id"])
        if result["phase"]=="done" and admin.state.outstanding_operation is None:
            return result
        time.sleep(.01)
    pytest.fail("Addon verification did not finish")


def test_unconfigured_addons_are_explicit_and_do_not_create_files(managed_addons,tmp_path):
    admin,state=managed_addons
    assert {item["status"] for item in admin.addons.snapshot()["items"]}=={"missing"}
    assert not (tmp_path / "data").exists()
    assert {"tuned","jlens","detector"}.isdisjoint(state.model_descriptor("bf16")["capabilities"])
    assert admin.addons.availability("bf16")["tuned"]["code"]=="addon_missing"


def test_single_artifact_and_catalog_inspection_are_readonly(managed_addons,tmp_path):
    admin,state=managed_addons
    folder=make_reader(admin,tmp_path)
    assert edit(admin,"inspect",path=str(folder))["compatible_models"]==["bf16"]
    assert edit(admin,"inspect",path=str(tmp_path))["files_ok"]
    assert not admin.addons.path.exists()
    assert TunedLensCatalog(folder).status(model_id="bf16",metadata_fingerprint=admin.addons.model_identity("bf16")["metadata_fingerprint"])["status"]=="ready"


def test_connect_verify_enable_and_restart_preserve_model_and_packages(managed_addons,tmp_path):
    admin,state=managed_addons
    folder=make_reader(admin,tmp_path)
    weights=(folder / "translators.safetensors").read_bytes()
    edit(admin,"connect",path=str(folder))
    original=admin.addons.path.read_bytes()
    with pytest.raises(AddonError) as error:
        edit(admin,"set_enabled",enabled=True)
    assert error.value.code=="addon_unverified"
    model=object(); state.model=model; state.status="ready"
    result=verify(admin)
    assert result["status"]=="completed",result
    assert state.model is model and state.active_model_id=="bf16"
    assert "tuned" not in state.model_descriptor("bf16")["capabilities"]
    edit(admin,"set_enabled",enabled=True)
    assert "tuned" in state.model_descriptor("bf16")["capabilities"]
    assert "tuned" not in state.model_descriptor("other")["capabilities"]
    assert admin.addons.availability("other")["tuned"]["code"]=="addon_incompatible"
    assert state.tuned_lens_catalog.root==folder
    backups=list((tmp_path / "data/experimental-addon-backups").glob("*.json"))
    assert any(path.read_bytes()==original for path in backups)
    fresh=ModelState(state.model_path,registry=state.registry,model_paths=state.registry.model_paths(),active_model_id="bf16",status="setup_required")
    ModelAdmin(fresh,tmp_path / "data")
    assert "tuned" in fresh.model_descriptor("bf16")["capabilities"]
    assert (folder / "translators.safetensors").read_bytes()==weights


def test_changed_artifact_disables_capability_and_checksum_reverification_fails(managed_addons,tmp_path):
    admin,state=managed_addons
    folder=make_reader(admin,tmp_path)
    edit(admin,"connect",path=str(folder)); assert verify(admin)["status"]=="completed"
    edit(admin,"set_enabled",enabled=True)
    (folder / "translators.safetensors").write_bytes(b"different malicious bytes")
    assert "tuned" not in state.model_descriptor("bf16")["capabilities"]
    assert admin.addons.availability("bf16")["tuned"]["code"]=="addon_unverified"
    result=verify(admin)
    assert result["status"]=="failed" and result["error_code"]=="invalid_addon_files"


def test_changed_model_files_invalidate_receipt(managed_addons,tmp_path):
    admin,state=managed_addons
    folder=make_reader(admin,tmp_path)
    edit(admin,"connect",path=str(folder)); assert verify(admin)["status"]=="completed"
    edit(admin,"set_enabled",enabled=True)
    (tmp_path / "bf16/model.safetensors").write_bytes(b"replaced model weights")
    assert not admin.addons.current_report("tuned")
    assert "tuned" not in state.model_descriptor("bf16")["capabilities"]


def test_unvalidated_or_wrong_metadata_reader_is_not_enabled(managed_addons,tmp_path):
    admin,state=managed_addons
    folder=make_reader(admin,tmp_path,validation_status="unvalidated",accepted=False)
    assert not edit(admin,"inspect",path=str(folder))["files_ok"]
    with pytest.raises(AddonError): edit(admin,"connect",path=str(folder))
    manifest=artifact_manifest(folder,artifact_id=folder.name,model_id="bf16",metadata_fingerprint="0"*64)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    edit(admin,"connect",path=str(folder))
    assert verify(admin)["status"]=="completed"
    assert admin.addons.snapshot()["items"][0]["status"]=="incompatible"
    with pytest.raises(AddonError): edit(admin,"set_enabled",enabled=True)


def test_torch_does_not_gain_internal_addons_from_matching_model_name(managed_addons,tmp_path):
    admin,state=managed_addons
    folder=make_reader(admin,tmp_path)
    state.registry=replace(state.registry,backend="torch")
    assert admin.addons.inspect("tuned",str(folder))["compatible_models"]==[]


def test_adapter_change_invalidates_cached_model_identity(managed_addons,tmp_path):
    admin,state=managed_addons
    folder=make_reader(admin,tmp_path)
    edit(admin,"connect",path=str(folder));assert verify(admin)["status"]=="completed"
    edit(admin,"set_enabled",enabled=True)
    models={**state.registry.models,"bf16":replace(state.registry.models["bf16"],adapter="causal-lm")}
    state.registry=replace(state.registry,models=models)
    assert admin.addons.current_report("tuned") is None
    assert not admin.addons.availability("bf16")["tuned"]["available"]
    assert admin.addons.inspect("tuned",str(folder))["compatible_models"]==[]


def test_jacobian_package_is_bf16_only_and_keeps_native_probe_requirement(managed_addons,tmp_path,monkeypatch):
    admin,state=managed_addons
    manifest,contents,_tensors=jacobian_fixture()
    manifest["reader"]["hidden_size"]=4096;manifest["reader"]["vocab_size"]=131072
    folder=tmp_path / manifest["artifact_id"];folder.mkdir()
    for name,content in contents.items(): (folder / name).write_bytes(content)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    # Synthetic tensor files cannot run a tokenizer/native transfer probe. The
    # real probe implementation and real packages are tested separately.
    monkeypatch.setattr(admin.addons,"probe_tokenizer_matches",lambda *args:True)
    edit(admin,"connect",kind="jlens",path=str(folder))
    result=verify(admin,"jlens");assert result["status"]=="completed",result
    edit(admin,"set_enabled",kind="jlens",enabled=True)
    assert "jlens" in state.model_descriptor("bf16")["capabilities"]
    assert not admin.addons.availability("hybrid")["jlens"]["available"]
    assert admin.addons.current_report("jlens")["artifacts"][0]["runtime_check"]=="bf16_transfer_probe"


def test_disconnect_only_edits_settings_and_stale_revisions_are_rejected(managed_addons,tmp_path):
    admin,state=managed_addons
    folder=make_reader(admin,tmp_path)
    stale=admin.addons.revision
    edit(admin,"connect",path=str(folder))
    with pytest.raises(AddonError) as error:
        admin.addon_action({"action":"disconnect","kind":"tuned","addon_revision":stale,"registry_sha256":state.registry.sha256})
    assert error.value.code=="addon_conflict"
    edit(admin,"disconnect")
    assert (folder / "translators.safetensors").is_file()
    assert admin.addons.availability("bf16")["tuned"]["code"]=="addon_missing"


def test_cancellation_leaves_verification_uncommitted(managed_addons,tmp_path,monkeypatch):
    admin,state=managed_addons
    folder=make_reader(admin,tmp_path);edit(admin,"connect",path=str(folder))
    original=admin.addons.path.read_bytes()
    monkeypatch.setattr(admin,"_log",lambda *args:admin.close())
    assert verify(admin)["status"]=="cancelled"
    assert admin.addons.path.read_bytes()==original


def test_self_declared_code_hash_cannot_authorize_a_python_plugin(tmp_path):
    script=tmp_path / "unreviewed.py"
    script.write_text('raise AssertionError("must never execute")')
    digest=hashlib.sha256(script.read_bytes()).hexdigest()
    bundle={"stylometry":{"legacy_extractors":{name:{"path":str(script),"sha256":digest} for name in (
        "normalization","lexical","sentence","syntactic","function_words","information_aesthetics")}}}
    with pytest.raises(RuntimeError,match="Unreviewed"):
        verify_legacy_runtime_code(tmp_path,bundle)
