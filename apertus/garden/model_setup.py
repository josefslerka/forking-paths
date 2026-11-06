"""Explicit checkpoint acquisition/conversion; never runs on server startup.

Downloads require a pinned revision. Existing target directories are never
overwritten. Only safetensors and non-executable tokenizer metadata are fetched.
"""
from __future__ import annotations

import gc
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from .registry_config import ADAPTERS, ModelRegistry, validate_model_files

METADATA_FILES = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer.model",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "vocab.json", "merges.txt", "chat_template.jinja", "README.md",
    "LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE",
}
RECEIPT = "workbench-receipt.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_receipt(directory: Path, payload: dict) -> None:
    payload = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), **payload}
    payload["files"] = {file.name: {"bytes": file.stat().st_size, "sha256": sha256_file(file)}
                        for file in sorted(directory.iterdir()) if file.is_file() and file.name != RECEIPT}
    (directory / RECEIPT).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


@contextmanager
def staged_target(target: Path, temporary_root: Path | None = None):
    if target.exists() or target.is_symlink():
        raise ValueError(f"Refusing to overwrite existing target: {target}. Choose another model path.")
    target.parent.mkdir(parents=True, exist_ok=True)
    # TemporaryDirectory owns exactly this fresh path; never delete a user model.
    if temporary_root is not None and temporary_root.resolve().parent != target.parent.resolve():
        raise ValueError("Temporary setup workspace must be next to the target for atomic publication.")
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-setup-", dir=temporary_root or target.parent) as temporary:
        stage = Path(temporary) / "model"
        yield stage
        problems = validate_model_files(stage)
        if problems:
            raise ValueError("Prepared model failed file validation: " + "; ".join(problems))
        if target.exists() or target.is_symlink():
            raise ValueError(f"Target appeared during setup; not overwriting: {target}")
        stage.rename(target)
    print(f"Ready: {target}", flush=True)


def download_model(registry: ModelRegistry, model_id: str, *, temporary_root: Path | None = None) -> None:
    model = registry.models[model_id]
    if model.kind == "apertus-hybrid":
        raise ValueError("Hybrid combines bf16 and 8bit; download bf16, then convert 8bit. It has no separate weights.")
    if model.convert_from:
        raise ValueError(f"Use `models convert {model_id}` after downloading {model.convert_from}.")
    if not model.repository or not model.revision:
        raise ValueError("Downloading requires repository and a full pinned revision in the registry. Local paths do not need them.")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise ImportError("Install download support: python -m pip install -e '.[download]'") from exc
    # Public, unauthenticated download; no implicit credential use or login.
    info = HfApi(token=False).model_info(model.repository, revision=model.revision, files_metadata=True)
    if info.sha != model.revision:
        raise ValueError("The registry revision does not match the remote commit.")
    remote = {item.rfilename: item for item in info.siblings}
    metadata = sorted(set(remote) & METADATA_FILES)
    if "config.json" not in metadata:
        raise ValueError("The remote repository has no supported root config.json.")
    with staged_target(model.path, temporary_root) as stage:
        stage.mkdir()

        def fetch(name: str) -> Path:
            if name not in remote or Path(name).name != name or name.startswith("."):
                raise ValueError(f"Unsafe or missing remote filename: {name}")
            print(f"Downloading {name} ({remote[name].size or 0:,} bytes)", flush=True)
            path = Path(hf_hub_download(model.repository, name, revision=model.revision,
                                      local_dir=stage, token=False))
            if remote[name].size is not None and path.stat().st_size != remote[name].size:
                raise ValueError(f"Downloaded size mismatch: {name}")
            lfs = remote[name].lfs
            if lfs is not None and sha256_file(path) != lfs.sha256:
                raise ValueError(f"Downloaded SHA-256 mismatch: {name}")
            return path

        for name in metadata:
            fetch(name)
        ADAPTERS[model.adapter].validate(json.loads((stage / "config.json").read_text()))
        if "model.safetensors.index.json" in remote:
            index = json.loads(fetch("model.safetensors.index.json").read_text())
            mapping = index.get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("Invalid remote safetensors index.")
            shards = sorted(set(mapping.values()))
        elif "model.safetensors" in remote:
            shards = ["model.safetensors"]
        else:
            raise ValueError("Expected model.safetensors or model.safetensors.index.json; pickle/GGUF/custom loaders are not supported.")
        for name in shards:
            if not isinstance(name, str) or not name.endswith(".safetensors"):
                raise ValueError("The index references non-safetensors weights.")
            fetch(name)
        write_receipt(stage, {"operation": "download", "repository": model.repository,
                              "revision": model.revision, "remote_lfs_sha256_verified": True})


def _stream_llama(source: Path, target: Path, bits: int, group_size: int) -> None:
    """Full-attention Llama: process one source shard at a time, F32 -> BF16 -> Q.

    This deliberately supports one known weight layout, not every HF model.
    It avoids holding all 36+ GB of the official EuroLLM F32 checkpoint in RAM.
    """
    import mlx.core as mx
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "llama" or config.get("quantization") or config.get("quantization_config"):
        raise ValueError("Streaming conversion requires original, unquantized Llama safetensors.")
    ADAPTERS["llama-full-attention"].validate(config)
    target.mkdir()
    for name in METADATA_FILES - {"config.json"}:
        if (source / name).is_file():
            shutil.copyfile(source / name, target / name)
    index = source / "model.safetensors.index.json"
    mapping = json.loads(index.read_text())["weight_map"] if index.is_file() else None
    shards = sorted(set(mapping.values())) if mapping else ["model.safetensors"]
    output_map, seen = {}, set()
    total_size = 0
    for number, name in enumerate(shards, 1):
        print(f"Converting shard {number}/{len(shards)}: {name}", flush=True)
        weights = mx.load(str(source / name))
        output = {}
        for key, value in weights.items():
            if key in seen or (mapping is not None and mapping.get(key) != name):
                raise ValueError(f"Duplicate or unindexed source tensor: {key}")
            seen.add(key)
            if value.dtype not in {mx.float32, mx.bfloat16, mx.float16}:
                raise ValueError(f"Expected a floating tensor: {key}")
            value = value.astype(mx.bfloat16)
            if key.endswith(".weight") and value.ndim == 2:
                if value.shape[-1] % group_size:
                    raise ValueError(f"{key} is not divisible by quantization group size {group_size}.")
                quantized, scales, biases = mx.quantize(value, group_size=group_size, bits=bits, mode="affine")
                output.update({key: quantized, key[:-7] + ".scales": scales, key[:-7] + ".biases": biases})
            else:
                output[key] = value
            mx.eval(*output.values())
        mx.save_safetensors(str(target / name), output, metadata={"format": "mlx"})
        total_size += sum(array.nbytes for array in output.values())
        output_map.update({key: name for key in output})
        del output, weights, value
        gc.collect()
        mx.clear_cache()
    if mapping is not None and seen != set(mapping):
        raise ValueError("Source tensor set does not match its index.")
    config.pop("dtype", None)
    config["torch_dtype"] = "bfloat16"
    config["quantization"] = {"group_size": group_size, "bits": bits, "mode": "affine"}
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (target / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total_size}, "weight_map": output_map}, indent=2) + "\n")


def convert_model(registry: ModelRegistry, model_id: str, *, temporary_root: Path | None = None) -> None:
    model = registry.models[model_id]
    if registry.backend != "mlx" or not model.convert_from or model.bits is None:
        raise ValueError("Conversion needs backend mlx, convert_from and bits in the registry.")
    if sys.platform != "darwin":
        raise ValueError("MLX conversion requires Apple silicon macOS.")
    source = registry.models[model.convert_from]
    problems = validate_model_files(source.path)
    if problems:
        raise ValueError(f"Prepare source {source.id} first: " + "; ".join(problems))
    config = json.loads((source.path / "config.json").read_text())
    ADAPTERS[model.adapter].validate(config)
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("Do not quantize an already quantized source.")
    with staged_target(model.path, temporary_root) as stage:
        if config.get("model_type") == "llama":
            _stream_llama(source.path, stage, model.bits, model.group_size)
        else:
            subprocess.run([sys.executable, "-m", "mlx_lm.convert", "--hf-path", str(source.path),
                            "--mlx-path", str(stage), "-q", "--q-bits", str(model.bits),
                            "--q-group-size", str(model.group_size), "--q-mode", "affine"], check=True)
            # Conversion can rewrite tokenizer metadata. Preserve the exact source
            # tokenizer and non-architectural config so Hybrid remains auditable.
            for name in ("tokenizer.json", "tokenizer_config.json"):
                if (source.path / name).is_file():
                    shutil.copyfile(source.path / name, stage / name)
            converted = json.loads((stage / "config.json").read_text())
            quantization = converted.get("quantization", converted.get("quantization_config"))
            config["quantization"] = quantization
            (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        write_receipt(stage, {"operation": "conversion", "source_id": source.id,
                              "source_config_sha256": sha256_file(source.path / "config.json"),
                              "source_receipt_sha256": sha256_file(source.path / RECEIPT) if (source.path / RECEIPT).is_file() else None,
                              "source_repository": source.repository, "source_revision": source.revision,
                              "quantization": {"bits": model.bits, "group_size": model.group_size, "mode": "affine"},
                              "mlx_lm_version": version("mlx-lm")})


def main() -> None:
    """Private subprocess entry point used by the local model manager."""
    import argparse
    import signal
    from .registry_config import load_registry
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["download", "convert"])
    parser.add_argument("model_id")
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--temporary-root", required=True, type=Path)
    args = parser.parse_args()
    def interrupted(_number, _frame):
        raise KeyboardInterrupt("Model setup cancelled")
    signal.signal(signal.SIGTERM, interrupted)
    registry = load_registry(args.registry)
    action = download_model if args.operation == "download" else convert_model
    action(registry, args.model_id, temporary_root=args.temporary_root)


if __name__ == "__main__":
    main()
