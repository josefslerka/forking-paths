"""Model metadata and capabilities independent of the inference runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .registry_config import ADAPTERS, CORE_CAPABILITIES, ModelRegistration
except ImportError:
    from registry_config import ADAPTERS, CORE_CAPABILITIES, ModelRegistration

EUROLLM_ID = "eurollm9"
EUROLLM_REVISION = "3f5bacd0762eb45170985d6b699035f6a385ee3e"
EUROLLM_REPO = "utter-project/EuroLLM-9B-2512"
COMMON_CAPABILITIES = ["tokenization", "garden", "surprise", "contrast", "model-comparison", "archive", "guide"]
APERTUS_CAPABILITIES = ["embeddings", "raw", "tuned", "jlens", "attribution", "patching", "steering", "detector"]


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def model_descriptor(model_id: str, path: Path, label: str, *, backend: str, mock: bool,
                     registration: ModelRegistration | None = None,
                     enable_research_artifacts: bool = True) -> dict[str, Any]:
    config = read_json(path / "config.json") if not mock else {}
    family = config.get("model_type", "apertus" if model_id in {"bf16", "hybrid", "8bit", "4bit", "mock"} else "llama")
    layer_count = (42 if model_id == "mock-eurollm9" else 32) if mock else config.get("num_hidden_layers")
    quantization = config.get("quantization", config.get("quantization_config"))
    capabilities = list(COMMON_CAPABILITIES)
    adapter_error = None
    if registration is not None and not mock:
        adapter = ADAPTERS[registration.adapter]
        label = registration.label
        if config:
            try:
                adapter.validate(config)
                if backend == "mlx":
                    capabilities += list(adapter.internal_capabilities)
                    if family == "apertus" and layer_count == 32:
                        capabilities += ["patching"]
                        if not quantization or registration.kind == "apertus-hybrid":
                            capabilities += ["attribution"]
                        if enable_research_artifacts and model_id in {"bf16", "hybrid"}:
                            capabilities += ["tuned", "jlens", "detector"]
            except ValueError as exc:
                adapter_error = str(exc)
    elif family == "apertus" and (mock or backend == "mlx" and config.get("num_hidden_layers") == 32):
        capabilities += APERTUS_CAPABILITIES
    elif family == "llama" and (mock or backend == "mlx" and model_id == EUROLLM_ID
                               and layer_count == 42 and not config.get("sliding_window")):
        capabilities += ["steering", "raw", "embeddings"]
    if model_id == EUROLLM_ID and not mock and registration is None:
        label = f"EuroLLM 9B · {quantization['bits']}bit" if quantization and "bits" in quantization else "EuroLLM 9B · HF"
    tokenizer_file = path / "tokenizer.json"
    tokenizer_hash = hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() if tokenizer_file.is_file() else None
    return {
        "id": model_id, "label": label, "family": family, "backend": "mock" if mock else backend,
        "layers": layer_count, "vocab_size": config.get("vocab_size"),
        "quantization": quantization, "source_dtype": config.get("torch_dtype", config.get("dtype")),
        "capabilities": capabilities, "tokenizer_sha256": tokenizer_hash,
        "adapter": registration.adapter if registration else None,
        "adapter_error": adapter_error,
        "checkpoint": registration.source() if registration else checkpoint_identity(model_id, path),
    }


def checkpoint_identity(model_id: str, path: Path) -> dict[str, Any]:
    # A receipt identifies its source; it is not a fresh runtime hash of all weights.
    manifest = read_json(path / "workbench-receipt.json")
    if manifest:
        return {"repository": manifest.get("repository"),
                "revision": manifest.get("revision"),
                "verification": "local_setup_receipt_not_runtime_rehash"}
    return {"repository": None, "revision": None, "verification": "local_metadata_manifest"}
