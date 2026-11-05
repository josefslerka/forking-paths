from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from apertus.garden.cli import main
from apertus.garden.model_registry import model_descriptor
from apertus.garden.model_setup import staged_target
from apertus.garden.registry_config import ADAPTERS, load_registry, validate_model_files
from apertus.garden.predictive_detector import sigmoid


def registry_file(tmp_path: Path, tables: str = '[models.other_name]\npath = "weights"\n', **values) -> Path:
    root = {"schema_version": 1, "backend": "torch", "default_model": "other_name", **values}
    path = tmp_path / "models.toml"
    path.write_text("\n".join(f"{key} = {json.dumps(value)}" for key, value in root.items()) + "\n" + tables)
    return path


def fake_files(path: Path, family: str = "llama") -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": family, "num_hidden_layers": 7}))
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"synthetic placeholder, not real weights")


def test_new_model_id_paths_and_registry_snapshot_are_not_hardcoded(tmp_path, monkeypatch):
    path = registry_file(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    registry = load_registry(path)
    assert registry.default_model == "other_name"
    assert registry.model_paths() == {"other_name": tmp_path / "weights"}
    assert registry.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    before = registry.sha256
    path.write_text(path.read_text() + "# changed\n")
    assert registry.sha256 == before != load_registry(path).sha256


@pytest.mark.parametrize("tables,values", [
    ('[models.other_name]\nadapter="made-up"', {}),
    ('[models.other_name]\ncapabilities=["steering"]', {}),
    ('[models.other_name]\nrevision="main"', {}),
    ('[models.other_name]\nconvert_from="other_name"\nbits=6', {}),
    ('[models.other_name]\nconvert_from="second"\nbits=6\n[models.second]\nconvert_from="other_name"\nbits=4', {}),
    ('[models.other_name]\nkind="source"', {}),
    ('[models.other_name]\nadapter=["apertus"]', {}),
    ('[models.other_name]\nbits=3', {}),
    ('[models.other_name]\nconvert_from="absent"\nbits=6', {}),
    ('[models.other_name]\nlabel=""', {}),
    ('[models.other_name]', {"backend": ["mlx"]}),
    ('[models.other_name]', {"default_model": ["other_name"]}),
    ('[models.other_name]', {"schema_version": True}),
    ('[models.mock-real]', {"default_model": "mock-real"}),
    ('[models.hybrid]\nkind="apertus-hybrid"', {"default_model": "hybrid"}),
    ('[models.bf16]\nadapter="apertus"\nconvert_from="hybrid"\nbits=8\n[models."8bit"]\nadapter="apertus"\n[models.hybrid]\nkind="apertus-hybrid"', {"backend":"mlx","default_model":"hybrid"}),
    ('[models.bf16]\nadapter="apertus"\n[models."8bit"]\nadapter="apertus"\n[models.hybrid]\nkind="apertus-hybrid"\nrepository="test/model"', {"backend":"mlx","default_model":"hybrid"}),
])
def test_invalid_registry_fails_before_loading_or_mutation(tmp_path, tables, values):
    path = registry_file(tmp_path, tables, **values)
    with pytest.raises(ValueError):
        load_registry(path)
    assert list(tmp_path.iterdir()) == [path]


def test_source_only_entries_are_hidden_and_default_cache_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("APERTUS_CACHE_DIR", str(tmp_path / "cache"))
    path = registry_file(tmp_path, '[models.other_name]\n[models.source]\nkind="source"')
    registry = load_registry(path)
    assert registry.model_paths() == {"other_name": tmp_path / "cache/models/other_name"}
    assert "source" in registry.models
    assert not (tmp_path / "cache").exists()


def test_quantized_model_exports_its_declared_upstream_revision(tmp_path):
    path = registry_file(tmp_path, '[models.other_name]\nconvert_from="source"\nbits=6\n[models.source]\nkind="source"\nrepository="test/model"\nrevision="' + "b"*40 + '"')
    identity = load_registry(path).source_identity("other_name")
    assert identity["bits"] == 6
    assert identity["conversion_source"]["repository"] == "test/model"
    assert identity["conversion_source"]["revision"] == "b"*40
    assert identity["verification"] == "declared_registry_metadata_not_a_runtime_weight_hash"


def test_single_shard_and_unsafe_or_missing_index_are_checked(tmp_path):
    path = tmp_path / "weights"
    fake_files(path)
    assert validate_model_files(path) == []
    index = path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"tensor": "../elsewhere.safetensors"}}))
    assert any("unsafe" in error for error in validate_model_files(path))
    index.write_text(json.dumps({"weight_map": {"tensor": "missing.safetensors"}}))
    assert any("Missing" in error for error in validate_model_files(path))
    index.write_text(json.dumps({"weight_map": {"tensor": "pickle.bin"}}))
    assert any("unsafe" in error for error in validate_model_files(path))


def test_standard_hf_cache_blobs_are_supported_but_arbitrary_links_are_not(tmp_path):
    root = tmp_path / "models--test--model"
    snapshot = root / "snapshots" / ("a" * 40)
    snapshot.parent.mkdir(parents=True)
    fake_files(snapshot)
    (root / "blobs").mkdir()
    blob = root / "blobs/weight-hash"
    blob.write_bytes(b"cached weights")
    shard = snapshot / "model.safetensors"
    shard.unlink()
    shard.symlink_to(blob)
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map":{"tensor":"model.safetensors"}}))
    assert validate_model_files(snapshot) == []
    shard.unlink()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"unrelated user file")
    shard.symlink_to(outside)
    assert any("unsafe" in error for error in validate_model_files(snapshot))


def test_adapter_capabilities_are_guarded_by_family_backend_and_layout(tmp_path):
    model = tmp_path / "weights"
    fake_files(model)
    path = registry_file(tmp_path, '[models.other_name]\npath="weights"\nadapter="llama-full-attention"', backend="mlx")
    registry = load_registry(path)
    registration = registry.models["other_name"]
    descriptor = model_descriptor("other_name", model, "ignored", backend="mlx", mock=False, registration=registration)
    assert descriptor["layers"] == 7
    assert {"embeddings", "raw", "steering"} <= set(descriptor["capabilities"])
    assert {"tuned", "jlens", "patching", "detector", "attribution"}.isdisjoint(descriptor["capabilities"])
    torch = model_descriptor("other_name", model, "ignored", backend="torch", mock=False, registration=registration)
    assert "raw" not in torch["capabilities"]
    for config in ({"model_type":"apertus", "num_hidden_layers":32}, {"model_type":"llama", "num_hidden_layers":7, "sliding_window":128}):
        with pytest.raises(ValueError):
            ADAPTERS["llama-full-attention"].validate(config)


def test_core_profile_does_not_gain_hooks_from_model_name(tmp_path):
    model = tmp_path / "weights"
    fake_files(model, "apertus")
    path = registry_file(tmp_path, '[models.bf16]\npath="weights"', default_model="bf16", backend="mlx")
    registration = load_registry(path).models["bf16"]
    result = model_descriptor("bf16", model, "BF16", backend="mlx", mock=False, registration=registration)
    assert "embeddings" not in result["capabilities"]
    assert "detector" not in result["capabilities"]


def test_setup_refuses_overwrite_and_cleans_only_its_own_stage(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("user-owned")
    with pytest.raises(ValueError, match="overwrite"):
        with staged_target(target):
            pytest.fail("must not enter")
    assert marker.read_text() == "user-owned"
    fresh = tmp_path / "new"
    with pytest.raises(RuntimeError):
        with staged_target(fresh) as stage:
            stage.mkdir()
            raise RuntimeError("failed preparation")
    assert not fresh.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["existing"]


def test_cli_check_uses_registry_default_and_does_not_import_inference(tmp_path, capsys):
    fake_files(tmp_path / "weights")
    path = registry_file(tmp_path)
    main(["models", "check", "--registry", str(path)])
    assert "other_name" in capsys.readouterr().out


def test_pure_detector_math_remains_covered_without_private_calibration():
    assert sigmoid(1000.0) == 1.0
    assert sigmoid(-1000.0) == 0.0
    assert sigmoid(0.0) == 0.5
