"""Structural checks for cache-isolated steering on supported MLX decoders."""
from __future__ import annotations

from typing import Any


def decoder_body(model: Any) -> Any:
    """Accept only the full-attention Apertus/Llama path reproduced by steering.

    The model's own blocks retain their norms, RoPE, attention and MLP. Models
    with sliding attention need per-layer masks and are deliberately rejected.
    """
    if getattr(model, "model_type", None) not in {"apertus", "llama"}:
        raise RuntimeError("Steering podporuje MLX dekodéry Apertus a EuroLLM (Llama).")
    body = getattr(model, "model", None)
    layers = getattr(body, "layers", None)
    if not layers or not callable(getattr(body, "embed_tokens", None)) or not callable(getattr(body, "norm", None)):
        raise RuntimeError("Model nemá podporovanou strukturu decoderových bloků.")
    expected = getattr(getattr(model, "args", None), "num_hidden_layers", len(layers))
    if len(layers) != expected:
        raise RuntimeError("Počet načtených bloků nesouhlasí s konfigurací modelu.")
    if any(getattr(layer, "use_sliding", False) for layer in layers):
        raise RuntimeError("Steering pro vrstvy s lokální attention zatím není podporován.")
    if not callable(getattr(model, "lm_head", None)) and not callable(getattr(body.embed_tokens, "as_linear", None)):
        raise RuntimeError("Model nemá podporovanou výstupní projekci.")
    return body
