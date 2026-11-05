from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from apertus.garden.model_setup import download_model
from apertus.garden.registry_config import load_registry


def fake_hub(monkeypatch, content, *, bad_hash=False):
    revision = "a" * 40
    siblings = [SimpleNamespace(rfilename=name, size=len(data),
                 lfs=SimpleNamespace(sha256=("0" * 64 if bad_hash and name.endswith("safetensors") else hashlib.sha256(data).hexdigest())))
                for name, data in content.items()]
    class HfApi:
        def __init__(self, token):
            assert token is False
        def model_info(self, repository, **kwargs):
            assert repository == "test/local-fixture" and kwargs["revision"] == revision
            return SimpleNamespace(sha=revision, siblings=siblings)
    def fetch(repository, filename, *, local_dir, revision, token):
        assert token is False
        path = local_dir / filename
        path.write_bytes(content[filename])
        return path
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=HfApi, hf_hub_download=fetch))


@pytest.mark.parametrize("bad_hash", [False, True])
def test_explicit_pinned_download_validates_hashes_and_does_not_fetch_code(tmp_path, monkeypatch, bad_hash):
    content = {"config.json": b'{"model_type":"llama"}', "tokenizer.json":b'{}',
               "model.safetensors":b"fake tensor bytes for download plumbing only", "remote_code.py":b"must not be fetched"}
    fake_hub(monkeypatch, content, bad_hash=bad_hash)
    path = tmp_path / "models.toml"
    path.write_text('schema_version=1\nbackend="torch"\ndefault_model="custom"\n[models.custom]\npath="weights"\nrepository="test/local-fixture"\nrevision="' + "a"*40 + '"\n')
    registry = load_registry(path)
    if bad_hash:
        with pytest.raises(ValueError, match="SHA-256"):
            download_model(registry,"custom")
        assert not (tmp_path / "weights").exists()
    else:
        download_model(registry,"custom")
        receipt = json.loads((tmp_path / "weights/workbench-receipt.json").read_text())
        assert receipt["revision"] == "a" * 40
        assert receipt["files"]["model.safetensors"]["sha256"] == hashlib.sha256(content["model.safetensors"]).hexdigest()
        assert not (tmp_path / "weights/remote_code.py").exists()


def test_unpinned_download_is_rejected_without_network(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text('schema_version=1\nbackend="torch"\ndefault_model="custom"\n[models.custom]\npath="weights"\nrepository="test/local-fixture"\n')
    with pytest.raises(ValueError, match="pinned"):
        download_model(load_registry(path), "custom")
    assert list(tmp_path.iterdir()) == [path]
