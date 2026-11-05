"""Numerical model adapters. Importing the web application needs neither MLX nor Torch."""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ModelBackend(Protocol):
    name: str

    def load(self, path: str, *, lazy: bool = False) -> tuple[Any, Any]: ...
    def make_cache(self, model: Any) -> Any: ...
    def last_logits(self, model: Any, ids: list[int], cache: Any) -> tuple[np.ndarray, Any]: ...
    def full_logits(self, model: Any, ids: list[int]) -> np.ndarray: ...
    def synchronize(self) -> None: ...
    def clear_cache(self) -> None: ...


class MLXBackend:
    name = "mlx"

    def load(self, path: str, *, lazy: bool = False) -> tuple[Any, Any]:
        import mlx.core as mx
        from mlx_lm import load
        mx.default_stream(mx.default_device())
        return load(path, lazy=lazy, tokenizer_config={"trust_remote_code": False})

    def make_cache(self, model: Any) -> Any:
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(model)

    @staticmethod
    def _output(model: Any, ids: Any, cache: Any, last: bool) -> Any:
        # Avoid materializing a prefix × vocabulary matrix for next-token queries.
        if (getattr(model, "model_type", None) == "apertus"
                and hasattr(model, "model") and hasattr(model, "lm_head")
                and not getattr(getattr(model, "args", None), "tie_word_embeddings", False)):
            hidden = model.model(ids, cache=cache)
            return model.lm_head(hidden[:, -1:, :] if last else hidden)
        output = model(ids, cache=cache)
        if hasattr(output, "logits"):
            return output.logits
        return output[0] if isinstance(output, (tuple, list)) else output

    def last_logits(self, model: Any, ids: list[int], cache: Any) -> tuple[np.ndarray, Any]:
        import mlx.core as mx
        output = self._output(model, mx.array([ids]), cache, True)
        last = output[0, -1].astype(mx.float32)
        mx.eval(last, [item.state for item in cache])
        return np.asarray(last).copy(), cache

    def full_logits(self, model: Any, ids: list[int]) -> np.ndarray:
        import mlx.core as mx
        output = self._output(model, mx.array([ids]), None, False)[0].astype(mx.float32)
        mx.eval(output)
        return np.asarray(output).copy()

    def synchronize(self) -> None:
        import mlx.core as mx
        mx.synchronize()

    def clear_cache(self) -> None:
        import mlx.core as mx
        mx.clear_cache()


class TorchBackend:
    """Portable full-precision backend for original Hugging Face weights."""
    name = "torch"

    def __init__(self, device: str = "auto") -> None:
        self.device = device

    def load(self, path: str, *, lazy: bool = False) -> tuple[Any, Any]:
        import json
        from pathlib import Path
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        config = json.loads((Path(path) / "config.json").read_text())
        if config.get("quantization", {}).get("mode") == "affine":
            raise ValueError("Torch vyžaduje původní HF váhy; MLX affine váhy nejsou přenositelné.")
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, torch_dtype=dtype,
            low_cpu_mem_usage=True, attn_implementation="eager", device_map={"": device},
            use_safetensors=True, trust_remote_code=False,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        return model, tokenizer

    def make_cache(self, model: Any) -> Any:
        return None

    def last_logits(self, model: Any, ids: list[int], cache: Any) -> tuple[np.ndarray, Any]:
        import torch
        with torch.inference_mode():
            output = model(
                torch.tensor([ids], device=model.device), past_key_values=cache, use_cache=True,
            )
            return output.logits[0, -1].float().cpu().numpy().copy(), output.past_key_values

    def full_logits(self, model: Any, ids: list[int]) -> np.ndarray:
        import torch
        with torch.inference_mode():
            output = model(torch.tensor([ids], device=model.device), use_cache=False)
            return output.logits[0].float().cpu().numpy().copy()

    def synchronize(self) -> None:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def clear_cache(self) -> None:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_backend(name: str, device: str = "auto") -> ModelBackend:
    if name == "mlx":
        return MLXBackend()
    if name == "torch":
        return TorchBackend(device)
    raise ValueError(f"Neznámý výpočetní backend: {name}")
