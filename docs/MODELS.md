# Model registry and adapters

The registry separates **which checkpoint to load** from **how its architecture
can be inspected**. A new compatible checkpoint needs a new TOML table, not a
server edit. An actually new internal architecture needs tested code.

## A minimal registration

Copy `models.example.toml` (MLX) or `models.torch.example.toml` (PyTorch) to
`models.toml`. This local file is ignored by Git.

```toml
schema_version = 1
backend = "torch"
default_model = "my_experiment"

[models.my_experiment]
label = "My experiment model"
path = "models/my-checkpoint"
adapter = "causal-lm"
```

The relative path above is relative to the registry file, not the terminal's
working directory. Absolute paths and `~` are supported. Without `path`, a
registration uses the user cache directory's `models/<id>` subdirectory.
Run `apertus-workbench paths` or `models list` to see resolved locations.

```sh
apertus-workbench models check my_experiment --registry models.toml
apertus-workbench serve --registry models.toml
```

Restart after changing the registry. The server snapshots registrations for
its lifetime. One process uses **one backend**; cross-model comparison currently
requires both models to work with that backend. There is no automatic switch
between MLX, Torch, Ollama, and a remote API.

## Fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Currently `1`; required. |
| `backend` | `mlx` or `torch` for the whole server. |
| `default_model` | Any registered inference ID. |
| `models.<id>.label` | Display name; not an architecture selector. |
| `path` | Local directory with config, tokenizer and safetensors. |
| `adapter` | Reviewed adapter ID; defaults to `causal-lm`. |
| `repository`, `revision` | Optional HF source metadata; both needed to download. Revision must be a full 40-character commit hash. |
| `convert_from`, `bits`, `group_size` | Explicit MLX conversion recipe; bits 4/6/8, groups 32/64/128. |
| `kind` | `native` (default), `source` (conversion input, hidden from toolbar), or reserved `apertus-hybrid`. |
| `enable_research_artifacts` | Top-level opt-in for separately validated Apertus lens/detector artifacts; defaults to false. Not a download switch. |

Unknown fields, unknown adapters and cyclic conversion dependencies fail
early. IDs beginning with `mock` are reserved for clearly labelled synthetic
fixtures. The ID `hybrid` is reserved for the documented Apertus composition
with sources `bf16` and `8bit`; do not reuse it for another architecture.

`models check` checks metadata, adapter compatibility, tokenizer presence and
weight-file completeness. It does **not** prove numerical equivalence, model
quality, or the identity of weight contents. Setup creates a SHA-256 receipt;
runtime reports distinguish declared registry provenance from a fresh full
rehash. Moving or replacing local files invalidates assumptions behind earlier
measurements; restart and revalidate after changes.

## Available adapters

| Adapter | Core probability tools | MLX internals | PyTorch internals |
| --- | --- | --- | --- |
| `causal-lm` | Yes, if the backend supports the causal checkpoint | No | No |
| `apertus` | Yes | Embeddings, RAW, steering; reviewed 32-layer Apertus also patching, and attribution with a full output head | No |
| `llama-full-attention` | Yes | Embeddings, RAW, steering; dynamic layer count | No |

Core means tokenization, next-token distribution/generation, text surprisal,
context contrasts, archive, and paired model measurements. A registered
checkpoint can still fail to load if its backend does not implement it.
`causal-lm` is not a promise to support every HF repository. Encoder-only,
embedding-only, multimodal and custom-code models are outside this contract.
Remote Python code and pickle-based checkpoints are not enabled.

The internal Llama adapter requires the native full-attention decoder layout
with embeddings, decoder blocks, final normalization and output projection.
Sliding/local attention is rejected; the model's actual layer count is used.
Do not select it just to make disabled menu items clickable.

## Included presets

The installed `apertus/garden/model_presets.toml` records pinned official sources:

- Apertus: `swiss-ai/Apertus-8B-2509`, revision
  `3162c99675aa588097cecd4a24b9aa1f712af477`; original BF16 plus explicit 8/4-bit
  affine MLX conversions and Hybrid composition.
- EuroLLM: `utter-project/EuroLLM-9B-2512`, revision
  `3f5bacd0762eb45170985d6b699035f6a385ee3e`; official F32 source followed by
  a BF16-to-6-bit affine MLX conversion.

To use the built-in EuroLLM recipe, either add its two tables from the presets
to your registry or explicitly select the preset file. You can print its path:

```sh
python -c 'from pathlib import Path; import apertus.garden; print(Path(apertus.garden.__file__).with_name("model_presets.toml"))'
```

Then use that path as `--registry PRESETS_PATH` on each command:

```sh
apertus-workbench models download eurollm9-source --registry PRESETS_PATH
apertus-workbench models convert eurollm9 --registry PRESETS_PATH
apertus-workbench models check eurollm9 --registry PRESETS_PATH
apertus-workbench serve --initial-model eurollm9 --registry PRESETS_PATH
```

The official EuroLLM source alone is roughly 36 GB. Conversion uses one source
shard at a time rather than holding the entire F32 model in memory; enough
space for source, output and download cache is still required. The converter
does **not** delete sources after conversion. Model acquisition is explicit,
public/unauthenticated, pinned, safetensors-only, and refuses existing output
directories. A failed staged setup may require re-downloading files; it is not
a general model-management application. For private/gated models, acquire the
weights yourself under their terms and register the local path.

GGUF files from Ollama are not supported by these numerical adapters. Renaming
one or changing the registry does not convert its tokenizer or weights.

## Implement a new internal architecture

1. First make the checkpoint work through `causal-lm` and the backend's native
   forward path. Check tokenizer boundaries, BOS policy and finite normalized
   probabilities.
2. Implement reviewed native block/readout hooks. Current shared hooks are in
   `domains/steering.py` (`decoder_body`, block splitting/replay) and
   `domains/representations.py` (trace/readout). Keep model-specific normalization,
   masks, position encoding and output projection faithful to its real forward.
3. Register an `Adapter` in `registry_config.py` with its family guard and only
   capabilities that have actual implementations. Do not load arbitrary modules
   from TOML. Update dispatch if the architecture needs a distinct path.
4. Add tiny-architecture and real-checkpoint numerical checks: replay vs native
   logits, causal prefix invariance, layer-0 identity, final-readout agreement,
   and zero/identity intervention controls. Test quantized and tied heads if
   supported. Refuse unsupported variants clearly.
5. Document the tested checkpoint/version/precision, then enable the capability.

Attribution and patching remain Apertus-specific. TUNED/J-Lens and the detector
also need trained/calibrated artifacts matching the checkpoint, not just a
compatible tensor shape. Those artifacts and private corpora are not bundled.

Specialized historical battery runners in this source tree retain locked
experimental designs. They are not a general multi-model API: use their
explicit options and matching assets, or the workbench comparison page for new
registered models. Their protocols are examples, not established findings.
