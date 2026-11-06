from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from apertus.garden import runtime_environment as runtime
from apertus.garden.model_admin import AdminError, ModelAdmin, registry_bytes
from apertus.garden.registry_config import load_registry
from apertus.garden.server import ModelState
from apertus.garden.settings import registry_path


@pytest.mark.parametrize("system,machine,expected", [
    ("linux", "x86_64", "torch"), ("linux", "aarch64", "torch"),
    ("win32", "AMD64", "torch"), ("darwin", "x86_64", "torch"),
    ("darwin", "arm64", "mlx"), ("darwin", "aarch64", "mlx"),
])
def test_defaults_match_the_supported_host_without_importing_frameworks(monkeypatch, tmp_path, system, machine, expected):
    monkeypatch.setattr(sys, "platform", system)
    monkeypatch.setattr(runtime.platform, "machine", lambda: machine)
    monkeypatch.delenv("APERTUS_MODEL_REGISTRY", raising=False)
    monkeypatch.chdir(tmp_path)
    assert runtime.recommended_backend() == expected
    selected = registry_path(data_root=tmp_path / "empty")
    assert load_registry(selected).backend == expected
    if expected == "torch":
        assert set(load_registry(selected).models) == {"bf16", "eurollm9-hf"}
    else:
        assert "hybrid" in load_registry(selected).models


def test_mlx_is_blocked_on_linux_even_if_packages_are_present(monkeypatch):
    monkeypatch.setattr(runtime, "recommended_backend", lambda: "torch")
    report = runtime.environment_report("mlx", available=lambda name: True)
    assert report["code"] == "unsupported_platform" and not report["ready"]
    assert report["install_backend"] == "torch"
    assert "[torch]" in report["install_command"]


def test_dependency_advice_uses_the_actual_server_interpreter(monkeypatch):
    monkeypatch.setattr(sys, "executable", "/tmp/fixture python/bin/python")
    report = runtime.environment_report("torch", available=lambda name: name != "accelerate")
    assert report["missing_modules"] == ["accelerate"]
    assert report["code"] == "dependency_missing"
    if sys.platform != "win32":
        assert shlex.split(report["install_command"])[0] == sys.executable


def mlx_registry(tmp_path):
    for name in ("bf16", "q4"):
        root = tmp_path / name
        root.mkdir()
        config = {"model_type":"apertus", "num_hidden_layers":32, "vocab_size":8}
        if name == "q4":
            config["quantization"] = {"bits":4, "mode":"affine"}
        (root / "config.json").write_text(json.dumps(config))
        (root / "tokenizer.json").write_text("{}")
        (root / "model.safetensors").write_bytes(b"synthetic fixture, not a model")
    path = tmp_path / "models.toml"
    path.write_bytes(registry_bytes({"schema_version":1,"backend":"mlx","default_model":"4bit","models":{
        "bf16":{"path":"bf16","adapter":"apertus"},
        "4bit":{"path":"q4","adapter":"apertus","convert_from":"bf16","bits":4}}}))
    selected = load_registry(path)
    state = ModelState(selected.models["4bit"].path, registry=selected, model_paths=selected.model_paths(),
                       active_model_id="4bit", backend_name="mlx", status="setup_required")
    return ModelAdmin(state, tmp_path / "data"), state, path


def test_unsupported_install_is_rejected_before_registry_jobs_or_downloads(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "recommended_backend", lambda: "torch")
    monkeypatch.setattr("apertus.garden.model_admin.dependency_available", lambda name: True)
    admin, state, path = mlx_registry(tmp_path)
    before = path.read_bytes()
    assert admin.snapshot()["profiles"] == []
    plan = admin.plan_install({"profile_id":"4bit","registry_sha256":state.registry.sha256})
    assert plan["environment"]["code"] == "unsupported_platform"
    with pytest.raises(AdminError) as error:
        admin.start_job({"action":"install","plan_id":plan["id"],"confirmed":True,"registry_sha256":state.registry.sha256})
    assert error.value.code == "unsupported_platform"
    assert path.read_bytes() == before and not admin.jobs
    assert not (tmp_path / "data").exists()


def test_switch_to_torch_reuses_bf16_and_backs_up_mlx_registrations(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "recommended_backend", lambda: "torch")
    admin, state, path = mlx_registry(tmp_path)
    before = path.read_bytes()
    weights = {p: p.read_bytes() for p in tmp_path.glob("*/*.safetensors")}
    result = admin.edit({"action":"use_torch","registry_sha256":state.registry.sha256})
    assert state.backend_name == "torch" and state.status == "setup_required"
    assert state.registry.models["bf16"].path == tmp_path / "bf16"
    assert state.registry.default_model == "bf16" and "4bit" not in state.registry.models
    assert set(row["id"] for row in result["profiles"]) == {"bf16", "eurollm9-hf"}
    assert next((tmp_path / "data/model-registry-backups").glob("*.toml")).read_bytes() == before
    assert all(p.read_bytes() == content for p, content in weights.items())
    assert not admin.jobs


def test_switch_cannot_change_a_loaded_model(monkeypatch, tmp_path):
    admin, state, path = mlx_registry(tmp_path)
    state.model = object()
    before = path.read_bytes()
    with pytest.raises(AdminError) as error:
        admin.edit({"action":"use_torch","registry_sha256":state.registry.sha256})
    assert error.value.code == "active_model" and path.read_bytes() == before


def test_switch_preserves_the_path_of_a_hidden_original_bf16_source(tmp_path):
    admin, state, path = mlx_registry(tmp_path)
    import tomllib
    value = tomllib.loads(path.read_text())
    value["models"]["bf16"].update(enabled=False, kind="source")
    path.write_bytes(registry_bytes(value))
    admin.reload({"registry_sha256":load_registry(path).sha256})
    admin.edit({"action":"use_torch","registry_sha256":state.registry.sha256})
    assert state.registry.models["bf16"].path == tmp_path / "bf16"
    assert state.registry.models["bf16"].enabled
    assert state.registry.models["bf16"].kind == "native"


def test_torch_factory_presets_are_never_modified(monkeypatch, tmp_path):
    selected = load_registry(Path(runtime.__file__).with_name("model_presets.torch.toml"))
    before = selected.path.read_bytes()
    state = ModelState(selected.models["bf16"].path, registry=selected, model_paths=selected.model_paths(),
                       active_model_id="bf16", backend_name="torch", status="setup_required")
    admin = ModelAdmin(state, tmp_path / "data")
    admin.edit({"action":"set_default","model_id":"eurollm9-hf","registry_sha256":selected.sha256})
    assert state.registry.path == tmp_path / "data/models.toml"
    assert selected.path.read_bytes() == before
