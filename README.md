[English](README.md) · [Čeština](README.cs.md)

# Forking Paths / Rozcestí

**A research and teaching workbench for exploring language models.**
The application is called **Rozcestí** in Czech and **Forking Paths** in English.
Apertus and EuroLLM are supported models, not the name of the application.
The branching-probability view remains **Probability Garden / Pravděpodobnostní
zahrada**.

**Česky:** Lokální výzkumná a výuková laboratoř jazykových modelů pro
humanitní badatele a psycholingvisty: tokenizace, pravděpodobnosti,
překvapení textu, vliv kontextu a podporované pohledy do modelu.
[Český popis a rychlý start →](README.cs.md)

A local research and teaching workbench for looking at what a language model
expects next — and how those expectations change with context. The interface
is available in Czech and English. It is aimed at humanities researchers and
psycholinguists, not just model developers.

You can inspect tokenization, next-token probabilities, word/token surprisal,
contrasts between contexts, and paired model measurements. Supported MLX
architectures also expose contextual representations, layer readouts and
interventions. The application is **not** a chat service, a model of human
reading time, or evidence that a model understands a text as a person does.

**Early alpha — runs locally.** No account or cloud inference API is needed. Model
weights, private research data, calibrated readouts, and scanned documents are
not included. No model is downloaded on startup.

For compatibility, the Python distribution/import namespace, `APERTUS_*`
environment variables and existing data/cache directories retain their original
technical names. `forking-paths` and the existing `apertus-workbench` command
run the same application. Rebranding does not move archives, weights or model
registrations. The Electron application is named **Forking Paths**.

[Český instalační návod](docs/INSTALL.cs.md) ·
[Adding a model](docs/MODELS.md) · [Methods and limitations](docs/METHODS.md) ·
[Optional desktop shell](docs/DESKTOP.md) · [Local verification record](docs/VALIDATION.md)

## 1. Try the interface without a model

You need Git and Python **3.11 or 3.12**. Download the source and install it in
an isolated environment:

```sh
git clone https://github.com/josefslerka/forking-paths.git
cd forking-paths
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
forking-paths demo
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.
Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Stop the server with Ctrl+C.
If the port is occupied, use `forking-paths demo --port 8766`.

**Demo values are synthetic fixtures, not measurements from Apertus or
EuroLLM.** The interface labels this mode; use it to check the installation and
learn the controls, not to draw linguistic conclusions.

## 2. Run a real model

Choose **one** numerical backend per server. Models in a paired comparison
must use the same backend. You do not need both optional installations.

### Apple silicon Mac: MLX

```sh
python -m pip install -e '.[mlx]'
cp models.example.toml models.toml
forking-paths serve
```

Open **Model management / Správa modelů** in the navigation, or go to
[http://127.0.0.1:8765/sprava-modelu](http://127.0.0.1:8765/sprava-modelu).
If no weights are ready, the application opens this page automatically:

- **Connect a local model:** paste its directory path (or use the native folder
  picker in Electron), check it and choose a display name. Files are not copied.
- **Install a prepared profile:** review the pinned sources, estimated disk
  space and conversion steps, then explicitly confirm the download. Apertus
  BF16/8-bit/4-bit/Hybrid and EuroLLM 9B 6-bit are included for MLX.
- **Test computation** checks loading and one next-token distribution.
  **Use** loads the model now; **Set as default** selects it for the next launch.

Hiding or disconnecting a registration **never deletes its weights**. Settings
edits are backed up and applied without restarting. Demo mode is read-only.
You still install the numerical Python extra once in the terminal; the GUI does
not run pip or install arbitrary Python code.

**Experimental addons** on the same page connect optional TUNED, J-LENS and
detector packages: **Connect directory → Verify package → Enable**. Each package
lists compatible models/precisions. These are local connections, not bundled
downloads: artifacts, calibration corpora and private provenance stay outside
Git. The current J-LENS is BF16-only; the frozen detector requires Apertus Hybrid.
See [addon formats and verification](docs/MODELS.md#experimental-addons).

The command line remains available for reproducible setup:

```sh
forking-paths models download bf16
forking-paths models check bf16
forking-paths serve
```

The BF16 checkpoint is roughly 16 GB of weights; allow additional RAM and disk
space for inference, conversion and download caching. It is not a small
browser-only app. For a smaller running model, prepare the 8-bit variant:

```sh
forking-paths models convert 8bit
forking-paths serve --initial-model 8bit
```

The optional `hybrid` registration combines that 8-bit body with exact BF16
input embeddings/output head. It still needs **both** source directories:

```sh
forking-paths models check hybrid
forking-paths serve --initial-model hybrid
```

The built-in presets additionally contain 4-bit Apertus and 6-bit EuroLLM 9B.
See [model setup and disk requirements](docs/MODELS.md#included-presets).

### Linux/Windows or a non-MLX machine: PyTorch

```sh
python -m pip install -e '.[torch]'
cp models.torch.example.toml models.toml
```

Edit `models.toml` to point to **original HF safetensors** for a causal language
model supported by Transformers, or run `serve` and connect it through
**Model management**. Its prepared downloads are Apertus BF16 and original
EuroLLM 9B HF weights (not the MLX conversions). For the command-line route:

```sh
forking-paths models check
forking-paths serve --device auto
```

`auto` uses CUDA if available, otherwise CPU. CPU inference uses FP32 and can
be slow and memory-intensive. Install the PyTorch build appropriate for your
hardware if you need a particular CUDA version. MLX affine checkpoints and
Ollama/GGUF files are not interchangeable with HF safetensors. This backend
currently supports the core probability tools, **not** internal layer tools.

## Add your own model

Use **Model management → Connect a local model**. The application detects a
reviewed adapter from the configuration, or falls back to core probability
tools. Registration does not implement missing architecture hooks.

For a version-controlled setup recipe, add a table to a local `models.toml`:

```toml
[models.my_model]
label = "My local causal model"
path = "/absolute/path/to/a/local/checkpoint"
adapter = "causal-lm"
```

Use `default_model = "my_model"` to select it at startup. The ID does not have
to be Apertus or EuroLLM. The toolbar discovers registered models from the
server; incomplete models are unavailable rather than silently substituted.
For a compatible, full-attention Llama model on MLX, the reviewed
`llama-full-attention` adapter enables representations, RAW and steering.
An unfamiliar architecture starts with `causal-lm`; adding a genuinely new
internal architecture requires an adapter implementation and numerical tests,
not an unchecked feature flag. See [the registry guide](docs/MODELS.md).
After a valid manual edit, use **Apply file changes** in Model management.
Changes affecting a model that is already loaded require a restart.

## Your data stays separate

The local archive stores text you choose to save. Application data and model
cache live in your user profile, not in the source checkout:

```sh
forking-paths paths
```

Override with `APERTUS_DATA_DIR`, `APERTUS_CACHE_DIR`, or
`APERTUS_MODEL_REGISTRY`. `--registry path/to/models.toml` selects a registry
explicitly; relative model paths are resolved against **that file**.
`--data-dir` changes the server's archive/readout directory. For research
detector artifacts, set `APERTUS_DATA_DIR` before launching as well.
Never commit local archives, private texts, model weights or local registries.

The server binds to loopback only. This release has no user authentication or
multi-user isolation; **do not expose it publicly**, tunnel it to the internet,
or treat it as a shared hosted service. See [SECURITY.md](SECURITY.md).

## Development and verification

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
cd apertus/garden/e2e
npm ci
npx playwright install chromium
npm run test:stub
npm run test:admin
npm run test:addons
```

The default tests do not download model weights. Numerical backend tests are
optional; tiny randomly initialized local models test adapter plumbing, not
linguistic validity. Private detector calibration regressions are explicitly
skipped when their external fixtures are absent. Synthetic UI tests still run.
The desktop helper has separate Node tests; see [CONTRIBUTING.md](CONTRIBUTING.md).

## License and attribution

Workbench code and original interface assets: **Apache License 2.0**
([LICENSE](LICENSE), [NOTICE](NOTICE)). Model weights and dependencies retain
their own licenses and attribution requirements; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This is an independent research
workbench, not an official product of the model providers.

Please cite the model checkpoint/revision, backend, precision/quantization,
workbench version, and your exact stimulus/scoring policy in research reports.
[CITATION.cff](CITATION.cff) supplies basic software authorship metadata.
