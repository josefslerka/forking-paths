"""Opt-in Metal tests with tiny local random weights, never a network download.

Run: APERTUS_TEST_MLX=1 python -m pytest -q .../test_registry_mlx.py
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("APERTUS_TEST_MLX") != "1", reason="Opt-in native Metal numerical check (APERTUS_TEST_MLX=1)")


@pytest.mark.parametrize("tied", [True, False])
def test_new_mlx_checkpoint_registry_conversion_and_replay(tmp_path, tied):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models.llama import Model, ModelArgs
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    from apertus.garden.model_setup import convert_model, sha256_file
    from apertus.garden.registry_config import load_registry
    from apertus.garden.server import ModelState
    from apertus.garden.domains.representations import readout, trace_last_position

    source = tmp_path / "source"
    source.mkdir()
    mx.random.seed(137)
    args = ModelArgs(model_type="llama", hidden_size=32, intermediate_size=64,
                     num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                     rms_norm_eps=1e-5, vocab_size=32, tie_word_embeddings=tied, max_position_embeddings=128)
    model = Model(args)
    mx.eval(model.parameters())
    mx.save_safetensors(str(source / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (source / "config.json").write_text(json.dumps(asdict(args)))
    vocabulary = {"<unk>":0, "<s>":1, "</s>":2, "Ahoj":3, "světe":4, **{f"unit{i}":i for i in range(5,32)}}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>",1)])
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>", unk_token="<unk>").save_pretrained(source)
    registry_path = tmp_path / "models.toml"
    registry_path.write_text('schema_version=1\nbackend="mlx"\ndefault_model="new-model"\n[models.new-model]\npath="source"\nadapter="llama-full-attention"\n[models.new-q6]\npath="quantized"\nadapter="llama-full-attention"\nconvert_from="new-model"\nbits=6\ngroup_size=32\n')
    registry = load_registry(registry_path)
    convert_model(registry, "new-q6")
    receipt = json.loads((tmp_path / "quantized/workbench-receipt.json").read_text())
    assert receipt["files"]["model.safetensors"]["sha256"] == sha256_file(tmp_path / "quantized/model.safetensors")
    assert receipt["quantization"]["bits"] == 6
    for model_id in ("new-model", "new-q6"):
        state = ModelState(registry.models[model_id].path, model_paths=registry.model_paths(),
                           registry=registry, active_model_id=model_id, backend_name="mlx")
        state.load()
        assert state.status == "ready", state.error
        distribution = state.next_tokens("Ahoj", 1.0, 3)
        assert abs(sum(item["probability"] for item in distribution["candidates"]) + distribution["other_probability"] - 1) < 1e-10
        ids = [1,3,4]
        native = state.backend.full_logits(state.model, ids)
        body, states, _hidden = trace_last_position(state.model, ids)
        replay = np.asarray(readout(state.model, body, states[-1:]).astype(mx.float32))[0]
        np.testing.assert_allclose(replay, native[-1], atol=1e-5)
        assert tuple(states.shape) == (4,32)
        changed = state.backend.full_logits(state.model, [1,3,4,8])
        np.testing.assert_allclose(native, changed[:3], atol=1e-5)
        assert np.isfinite(native).all()
        assert state.model_descriptor(model_id)["layers"] == 3
        state._release_model_local()

    # First-run loading, then test another real precision and restore the original.
    from apertus.garden.model_admin import ModelAdmin
    managed = ModelState(source, model_paths=registry.model_paths(), registry=registry,
                         active_model_id="new-model", backend_name="mlx")
    admin = ModelAdmin(managed, tmp_path / "admin-data")
    managed.start(load_initial=False)
    try:
        for model_id in ("new-model", "new-q6"):
            job = admin.start_job({"action":"smoke","model_id":model_id,"registry_sha256":registry.sha256})
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                result = admin.job_status(job["id"])
                if result["phase"] == "done" and managed.outstanding_operation is None:
                    break
                time.sleep(.02)
            assert result["status"] == "completed", result
            assert result["result"]["passed"] and managed.status == "ready"
            assert managed.active_model_id == "new-model"
        assert all(model["check"]["passed"] for model in admin.snapshot()["models"])
        assert managed.next_tokens("Ahoj",1.0,3)["candidates"]
    finally:
        admin.close(); managed.stop(); managed.worker_thread.join(10)
        managed._release_model_local()


def test_apertus_native_conversion_remains_compatible_with_hybrid(tmp_path):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models.apertus import Model, ModelArgs
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    from apertus.garden.cli import check_model
    from apertus.garden.model_setup import convert_model
    from apertus.garden.registry_config import load_registry
    from apertus.garden.server import ModelState

    source = tmp_path / "bf16"
    source.mkdir()
    args = ModelArgs(model_type="apertus", hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     mlp_bias=False, num_attention_heads=4, attention_bias=False, rms_norm_eps=1e-5,
                     vocab_size=32, num_key_value_heads=2, max_position_embeddings=128,
                     rope_theta=10000, post_norm=False, qk_norm=True, tie_word_embeddings=False)
    model = Model(args)
    flat = {key: value.astype(mx.bfloat16) for key,value in tree_flatten(model.parameters())}
    mx.save_safetensors(str(source / "model.safetensors"), flat)
    config = {**asdict(args), "torch_dtype":"bfloat16"}
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map":{key:"model.safetensors" for key in flat}}))
    tokenizer = Tokenizer(models.WordLevel({"<unk>":0,"<s>":1,"</s>":2,"Ahoj":3,**{f"unit{i}":i for i in range(4,32)}},unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>",1)])
    PreTrainedTokenizerFast(tokenizer_object=tokenizer,bos_token="<s>",eos_token="</s>",unk_token="<unk>").save_pretrained(source)
    path = tmp_path / "models.toml"
    path.write_text('schema_version=1\nbackend="mlx"\ndefault_model="hybrid"\n[models.bf16]\npath="bf16"\nadapter="apertus"\n[models."8bit"]\npath="q8"\nadapter="apertus"\nconvert_from="bf16"\nbits=8\ngroup_size=64\n[models.hybrid]\nkind="apertus-hybrid"\nadapter="apertus"\n')
    registry = load_registry(path)
    convert_model(registry, "8bit")
    assert check_model(registry, "hybrid") == []
    state = ModelState(registry.models["hybrid"].path, model_paths=registry.model_paths(), registry=registry, active_model_id="hybrid")
    state.load()
    assert state.status == "ready", state.error
    assert state.model.model.embed_tokens.weight.dtype == mx.bfloat16
    assert state.model.lm_head.weight.dtype == mx.bfloat16
    np.testing.assert_array_equal(np.asarray(state.model.model.embed_tokens.weight.astype(mx.float32)), np.asarray(flat["model.embed_tokens.weight"].astype(mx.float32)))
    assert np.isfinite(state.backend.full_logits(state.model,[1,3])).all()
    state._release_model_local()
