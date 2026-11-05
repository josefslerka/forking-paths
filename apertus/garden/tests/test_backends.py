from __future__ import annotations

import numpy as np
import pytest

from apertus.garden.backends import TorchBackend, create_backend


def test_portable_causal_backend_matches_full_prefix_and_incremental_cache() -> None:
    torch = pytest.importorskip("torch")
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(41)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)).eval()
    backend = TorchBackend("cpu")
    full = backend.full_logits(model, [1, 7, 9, 11])
    prefix, cache = backend.last_logits(model, [1, 7, 9], backend.make_cache(model))
    next_row, cache = backend.last_logits(model, [11], cache)
    np.testing.assert_allclose(prefix, full[2], atol=1e-6)
    np.testing.assert_allclose(next_row, full[3], atol=1e-6)
    changed = backend.full_logits(model, [1, 7, 9, 22])
    np.testing.assert_allclose(full[:3], changed[:3], atol=1e-6)
    assert cache is not None


def test_backend_selection_is_explicit() -> None:
    assert create_backend("mlx").name == "mlx"
    assert create_backend("torch", "cpu").name == "torch"
    with pytest.raises(ValueError):
        create_backend("unknown")


def test_torch_backend_loads_and_serves_the_existing_probability_api(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from apertus.garden.server import ModelState
    from apertus.garden.registry_config import load_registry
    vocabulary = {"<unk>": 0, "<s>": 1, "</s>": 2, "Ahoj": 3, "světe": 4}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>", unk_token="<unk>").save_pretrained(tmp_path)
    torch.manual_seed(42)
    LlamaForCausalLM(LlamaConfig(vocab_size=5, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)).save_pretrained(tmp_path)
    registry_path = tmp_path / "model-registry.toml"
    registry_path.write_text('schema_version=1\nbackend="torch"\ndefault_model="third-party-test"\n[models.third-party-test]\nlabel="Tiny local test"\npath="."\nadapter="causal-lm"\n')
    registry = load_registry(registry_path)
    state = ModelState(tmp_path, model_paths=registry.model_paths(), active_model_id=registry.default_model,
                       registry=registry, backend_name="torch", device="cpu")
    state.load()
    assert state.status == "ready", state.error
    result = state.next_tokens("Ahoj", 1.0, 3)
    assert abs(sum(c["probability"] for c in result["candidates"]) + result["other_probability"] - 1) < 1e-12
    scored = state.text_surprisal("Ahoj světe", 3)
    assert scored["token_count"] == 2
    assert scored["provenance"]["server"]["model"]["backend"] == "torch"
    assert scored["provenance"]["server"]["model"]["registry_sha256"] == registry.sha256
    with pytest.raises(ValueError):
        state.require_capability("embeddings")
    state._release_model_local()
