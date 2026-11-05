# Local preparation record — 5 November 2026

This is a record of checks actually run during local repository preparation,
not a claim that all platforms or every compatible checkpoint have been tested.
Nothing was uploaded to GitHub, PyPI, npm or a release server.

## Passed

- Fresh Python 3.11 environment: editable installation of the base package and
  development dependencies, with no MLX, Torch or Transformers initially
  present. Full Python suite: **451 passed, 17 explicitly skipped**.
- A fresh MLX-extra installation in that isolated environment: full Python
  suite also **451 passed, 17 skipped**. Opt-in native Metal suite:
  **3 passed**. Installed versions included MLX 0.32.4, MLX-LM 0.31.3,
  Transformers 5.22.1, Hugging Face Hub 1.33.0 and NumPy 2.4.6.
- Focused registry/backend checks in the existing numerical environment:
  **33 passed**, including Torch CPU forward/cache agreement, an arbitrarily
  named registered checkpoint, and the three native MLX checks. That environment
  used Torch 2.8.0, Transformers 5.14.1 and MLX 0.32.0.
- Native checks used tiny, randomly initialized **local** models: full-attention
  Llama with three layers, tied and untied heads, F32→BF16→6-bit streaming
  conversion, readout/native-logit agreement and causal-prefix invariance.
  A tiny Apertus additionally checked native 8-bit conversion and Hybrid's
  exact BF16 input/output installation. These are plumbing/numerical checks,
  not meaningful linguistic results from tiny models.
- Synthetic browser suite: **32 passed** against a dedicated loopback server,
  using Chrome. It covered Czech/English views, tokenization, word surprisal,
  embeddings, context contrast, model comparison, interventions and archive.
- Desktop helper tests: **9 passed**. Unsigned Apple-silicon Electron app
  directory built with Electron 45.2.3 / electron-builder 26.18.1. Publishing
  and automatic code-signing identity discovery were disabled. Own-code,
  Electron and Chromium license/notice files were present in the app resources.
- Desktop `npm audit` after the dependency updates: **0 reported
  vulnerabilities** at the time of this check. Some transitive packages still
  issue deprecation warnings; this is not a full security audit.
- Wheel and source distribution built. Wheel inspection found the CLI,
  registry presets, static interface, LICENSE and NOTICE; no model weights,
  SQLite archive, local `models.toml`, private arrays or node_modules.
- The wheel was installed into another disposable venv **outside the source
  checkout**, with no inference frameworks. Its demo served five page/static
  routes and a synthetic probability request successfully, using a temporary
  archive and random loopback port. No model weights were downloaded.
- The tracked-file hygiene gate found no matching personal absolute paths,
  credential patterns, private artifact types or files over 5 MB. The source
  snapshot is about 4.5 MB, excluding environments, dependencies and builds.
- Existing local Hybrid/EuroLLM directories passed registry/file preflight
  through a Git-ignored machine-specific configuration. No weights were copied
  into this repository and the original running application was not changed.

## Deliberately not claimed

The skipped checks require private detector calibration fixtures, an exact
external tokenizer, or an explicitly installed/enabled numerical backend.
Synthetic UI fixtures do not validate detector accuracy or linguistic findings.
Large official checkpoints were not freshly downloaded or benchmarked for this
release preparation. File preflight is not a fresh full weight-content hash or
a numerical equivalence test.

Windows/Linux, additional architectures, signed/notarized installers and a
complete end-user GUI lifecycle on the updated Electron were not exhaustively
tested. GitHub Actions definitions are included but have **not run on GitHub**.
The repository is local, and a public release still requires the owner's
explicit approval and the checklist in [RELEASE.md](RELEASE.md).
