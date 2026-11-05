"""Declarative model registrations and safe, backend-independent preflight.

TOML selects a known adapter, never a Python module or arbitrary executable.
One server uses one backend. New compatible checkpoints need only a new table.
"""
from __future__ import annotations

import json
import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .settings import cache_directory
except ImportError:
    from settings import cache_directory

CORE_CAPABILITIES = ("tokenization", "garden", "surprise", "contrast", "model-comparison", "archive", "guide")


@dataclass(frozen=True)
class Adapter:
    """Metadata contract for an implemented adapter, not a user feature flag."""

    id: str
    families: tuple[str, ...] = ()
    internal_capabilities: tuple[str, ...] = ()

    def validate(self, config: dict[str, Any]) -> None:
        if self.families and config.get("model_type") not in self.families:
            raise ValueError(f"Adapter {self.id} requires model_type in {self.families}; use causal-lm for basic measurements.")
        if self.internal_capabilities:
            depth = config.get("num_hidden_layers")
            if type(depth) is not int or depth < 1:
                raise ValueError(f"Adapter {self.id} requires a positive num_hidden_layers.")
            if config.get("sliding_window") or config.get("use_sliding_window"):
                raise ValueError(f"Adapter {self.id} does not support local/sliding attention. Use causal-lm.")
            layer_types = config.get("layer_types")
            if layer_types is not None and (not isinstance(layer_types, list) or any(kind != "full_attention" for kind in layer_types)):
                raise ValueError(f"Adapter {self.id} requires full_attention in every decoder block.")


ADAPTERS: dict[str, Adapter] = {
    "causal-lm": Adapter("causal-lm"),
    "apertus": Adapter("apertus", ("apertus",), ("embeddings", "raw", "steering")),
    "llama-full-attention": Adapter("llama-full-attention", ("llama",), ("embeddings", "raw", "steering")),
}


def register_adapter(adapter: Adapter) -> None:
    """Extension point for a reviewed implementation plus its numerical tests.

    Registering metadata is not sufficient for new internal hooks: implement
    the corresponding native decoder path in domains/steering.py as documented.
    No code is imported from model configuration files.
    """
    if adapter.id in ADAPTERS:
        raise ValueError(f"Adapter already registered: {adapter.id}")
    ADAPTERS[adapter.id] = adapter


@dataclass(frozen=True)
class ModelRegistration:
    id: str
    label: str
    path: Path
    adapter: str = "causal-lm"
    kind: str = "native"
    repository: str | None = None
    revision: str | None = None
    convert_from: str | None = None
    bits: int | None = None
    group_size: int = 64

    def source(self) -> dict[str, Any]:
        return {"repository": self.repository, "revision": self.revision,
                "convert_from": self.convert_from, "bits": self.bits, "group_size": self.group_size if self.bits else None,
                "verification": "declared_registry_metadata_not_a_runtime_weight_hash"}


@dataclass(frozen=True)
class ModelRegistry:
    path: Path
    backend: str
    default_model: str
    models: dict[str, ModelRegistration]
    enable_research_artifacts: bool = False
    sha256: str = ""

    def model_paths(self) -> dict[str, Path]:
        return {key: model.path for key, model in self.models.items() if model.kind != "source"}

    def source_identity(self, model_id: str) -> dict[str, Any]:
        model = self.models[model_id]
        identity = model.source()
        if model.convert_from:
            identity["conversion_source"] = self.source_identity(model.convert_from)
        if model.kind == "apertus-hybrid":
            if model.convert_from or model.bits or model.repository or model.revision:
                raise ValueError("Hybrid is a composition of bf16 and 8bit, not a downloadable/conversion checkpoint.")
            identity["composition"] = {key: self.source_identity(key) for key in ("bf16", "8bit")}
        return identity


def _unknown(value: dict[str, Any], allowed: set[str], context: str) -> None:
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"Unknown fields in {context}: {', '.join(sorted(extra))}")


def load_registry(path: Path) -> ModelRegistry:
    path = path.expanduser().resolve()
    if path.stat().st_size > 1_000_000:
        raise ValueError("Model registry exceeds 1 MB.")
    raw = path.read_bytes()
    value = tomllib.loads(raw.decode("utf-8"))
    _unknown(value, {"schema_version", "backend", "default_model", "models", "enable_research_artifacts"}, "registry")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("Model registry requires schema_version = 1.")
    backend = value.get("backend", "mlx")
    if not isinstance(backend, str) or backend not in {"mlx", "torch"}:
        raise ValueError("backend must be mlx or torch.")
    enabled_artifacts = value.get("enable_research_artifacts", False)
    if type(enabled_artifacts) is not bool:
        raise ValueError("enable_research_artifacts must be true or false.")
    raw_models = value.get("models")
    if not isinstance(raw_models, dict) or not 1 <= len(raw_models) <= 64:
        raise ValueError("Register between 1 and 64 models.")
    models = {}
    for model_id, entry in raw_models.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", model_id) or model_id.startswith("mock"):
            raise ValueError(f"Invalid or reserved model ID: {model_id}")
        if not isinstance(entry, dict):
            raise ValueError(f"models.{model_id} must be a table.")
        _unknown(entry, {"label", "path", "adapter", "kind", "repository", "revision", "convert_from", "bits", "group_size"}, f"models.{model_id}")
        label = entry.get("label", model_id)
        if not isinstance(label, str) or not label.strip() or len(label) > 120:
            raise ValueError(f"Invalid label for {model_id}.")
        adapter_id = entry.get("adapter", "causal-lm")
        if not isinstance(adapter_id, str) or adapter_id not in ADAPTERS:
            raise ValueError(f"Unknown adapter {adapter_id}; available: {', '.join(ADAPTERS)}")
        kind = entry.get("kind", "native")
        if not isinstance(kind, str) or kind not in {"native", "source", "apertus-hybrid"}:
            raise ValueError(f"Unknown registration kind: {kind}")
        if (model_id == "hybrid") != (kind == "apertus-hybrid"):
            raise ValueError("The reserved hybrid ID must use kind = apertus-hybrid, and vice versa.")
        model_path = entry.get("path")
        if model_path is not None and (not isinstance(model_path, str) or not model_path.strip()):
            raise ValueError(f"Invalid path for {model_id}.")
        if model_path is None:
            resolved = cache_directory() / "models" / model_id
        else:
            expanded = Path(model_path).expanduser()
            resolved = expanded if expanded.is_absolute() else path.parent / expanded
        repository = entry.get("repository")
        if repository is not None and (not isinstance(repository, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repository)):
            raise ValueError(f"Invalid Hugging Face repository for {model_id}.")
        revision = entry.get("revision")
        if revision is not None and (not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision)):
            raise ValueError(f"{model_id}: revision must be a full 40-character commit hash, not main.")
        bits = entry.get("bits")
        if bits is not None and (type(bits) is not int or bits not in {4, 6, 8}):
            raise ValueError("Supported affine quantization bits: 4, 6, 8.")
        group_size = entry.get("group_size", 64)
        if type(group_size) is not int or group_size not in {32, 64, 128}:
            raise ValueError("Supported quantization group sizes: 32, 64, 128.")
        models[model_id] = ModelRegistration(model_id, label.strip(), resolved.resolve(), adapter_id, kind,
                                              repository, revision, entry.get("convert_from"), bits, group_size)
    for model_id, model in list(models.items()):
        if model.kind == "apertus-hybrid":
            if model.convert_from or model.bits or model.repository or model.revision:
                raise ValueError("Hybrid is a composition of bf16 and 8bit, not a downloadable/conversion checkpoint.")
            if backend != "mlx" or not {"bf16", "8bit"} <= set(models):
                raise ValueError("Apertus Hybrid requires MLX and registrations bf16 and 8bit.")
            if any(models[key].adapter != "apertus" for key in ("bf16", "8bit")):
                raise ValueError("Both Hybrid sources must use the apertus adapter.")
            models[model_id] = ModelRegistration(model_id, model.label, models["8bit"].path, "apertus", kind="apertus-hybrid")
        if model.convert_from is not None and (not isinstance(model.convert_from, str) or model.convert_from not in models or model.convert_from == model_id):
            raise ValueError(f"Invalid conversion source for {model_id}.")
        if model.convert_from is not None and models[model.convert_from].kind == "apertus-hybrid":
            raise ValueError("A composed Hybrid runtime cannot be a conversion source; use original BF16 files.")
        if model.bits is not None and model.convert_from is None:
            raise ValueError(f"{model_id}: bits requires convert_from.")
        if model.convert_from is not None and model.bits is None:
            raise ValueError(f"{model_id}: convert_from requires bits.")
    # Reject dependency cycles before any download, load, or filesystem mutation.
    def visit(model_id: str, visiting: set[str]) -> None:
        if model_id in visiting:
            raise ValueError("Model conversion dependencies contain a cycle.")
        source = models[model_id].convert_from
        if source is not None:
            visit(source, visiting | {model_id})
    for model_id in models:
        visit(model_id, set())
    default = value.get("default_model", next(iter(models)))
    if not isinstance(default, str) or default not in models:
        raise ValueError("default_model is not registered.")
    if models[default].kind == "source":
        raise ValueError("A conversion-only source cannot be the default model.")
    return ModelRegistry(path, backend, default, models, enabled_artifacts, hashlib.sha256(raw).hexdigest())


def resolve_weight_file(path: Path, name: str) -> Path:
    """Reject index traversal; allow the standard HF snapshot -> blobs symlink."""
    if not isinstance(name, str) or not name.endswith(".safetensors"):
        raise ValueError("Weight shard is not safetensors")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Weight shard path escapes model directory")
    candidate = path / relative
    resolved = candidate.resolve()
    root = path.resolve()
    if not resolved.is_relative_to(root):
        blob_root = root.parent.parent / "blobs"
        if not (root.parent.name == "snapshots" and candidate.is_symlink() and resolved.is_relative_to(blob_root.resolve())):
            raise ValueError("Weight symlink escapes model directory and is not a Hugging Face snapshot blob")
    return resolved


def validate_model_files(path: Path) -> list[str]:
    """Report missing/unsafe files without importing an inference framework."""
    errors = []
    if not path.is_dir():
        return [f"Model directory does not exist: {path}"]
    config = path / "config.json"
    if not config.is_file():
        errors.append("Missing config.json")
    else:
        try:
            if not isinstance(json.loads(config.read_text()), dict):
                errors.append("config.json is not an object")
        except (OSError, ValueError):
            errors.append("Invalid config.json")
    if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        errors.append("Missing tokenizer.json, tokenizer.model or vocab.json")
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            mapping = index["weight_map"]
            if not isinstance(mapping, dict):
                raise ValueError("Weight map is not an object")
            names = set(mapping.values())
            if not names:
                raise ValueError("Empty weight map")
            for name in names:
                candidate = resolve_weight_file(path, name)
                if not candidate.is_file() or candidate.stat().st_size == 0:
                    errors.append(f"Missing or empty weight shard: {name}")
        except (OSError, ValueError, KeyError, TypeError):
            errors.append("Invalid or unsafe weight index")
    elif not any(candidate.stat().st_size > 0 for candidate in path.glob("*.safetensors") if candidate.is_file()):
        errors.append("Missing safetensors weights (pickle-based .bin checkpoints are not enabled)")
    return errors
