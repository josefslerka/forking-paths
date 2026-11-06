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

## Model-management update — 6 November 2026

Added `/sprava-modelu` / `/model-admin`, local registration, prepared-profile
plans, explicit download consent, cancellation, computation checks and atomic
registry editing. All work remains in the new local repository; no remote was
configured and nothing was published.

Checks actually completed for this update:

- Full Python suite in the MLX-extra development environment: **475 passed,
  17 explicitly skipped**. New checks cover exact backups, registry discovery
  and conflicting revisions, protected active/default/dependency entries,
  hidden sources, first-run setup, worker restoration after failure, missing
  dependencies, and transport Host/Origin guards. Cancellation tests use owned
  dummy child/grandchild processes, not real downloads or user model paths.
- Optional numerical checks: **3 MLX passed** and **3 Torch passed**, using
  tiny random local weights. Admin jobs loaded a first model on the numerical
  worker; MLX also tested another precision and restored the original model.
  These remain numerical plumbing tests, not linguistic model validation.
- Existing synthetic browser regression suite: **33 passed**. Dedicated
  administration suite: **6 passed**, including CZ/EN, model-free first launch,
  add/default/test/use, loading failure and recovery, hide/disconnect without
  deletion, install consent and cancellation/restoration. Download and inference
  doubles exist only in the isolated E2E launcher. No large model was fetched.
- Desktop helper suite: **11 passed**, including directory-picker origin/route
  restrictions and user-registry discovery. Updated unsigned Apple-silicon app
  directory built locally; publishing and signing-identity discovery disabled.
  The native directory dialog was not exhaustively tested interactively on all
  operating systems. Python and model weights remain external to the app.
- Inspected Czech first-run, normal cards, English and mobile browser captures.
  No horizontal overflow at the tested 390-pixel width.
- Built wheel/sdist, then installed the wheel outside the checkout in a
  disposable venv with **only NumPy**, no MLX/Torch/Transformers. Demo routes
  and synthetic inference passed. A separate real, model-free server served
  the management page/assets and saved a synthetic local-file registration
  with a backup into its temporary user-data registry. No existing registry or
  model weights were edited, copied or downloaded by these checks.

Large prepared downloads/conversions are implemented but were not repeated
end-to-end for this update. Platform, authentication and numerical limitations
above still apply. The GUI is a privileged **local-only** interface, not a
multi-user administrator service. A passing one-distribution computation check
does not validate every tool or the full weight provenance.

## Experimental-addons update — 6 November 2026

- Full Python gate: **487 passed, 17 skipped**. Twelve new synthetic checks
  cover read-only inspection, direct artifact directories/catalogs, separate
  connect/verify/enable, persisted receipts, checksum/model-change invalidation,
  acceptance gates, BF16-only J-LENS, Torch exclusion, exact settings backups,
  cancellation, non-destructive disconnect and rejection of self-authorized code.
- Browser suites: **33 general + 6 administration + 4 addon tests passed**.
  Addon tests use a dedicated synthetic server/package directory and exercise
  compatible vs unrelated models, tampered files, Czech/English and mobile
  layout. Their placeholder tensors are not native numerical reader validation.
- Desktop helper gate: **11 passed**. The addon form uses the existing guarded
  native directory picker; no new filesystem or execution API was exposed.
- Existing private packages were connected **in place**, only through ignored
  local user-data settings. No packages, corpora or calibration results were
  copied into the repository, and nothing was downloaded or published.
- Real local native checks completed: TUNED inference on Apertus Hybrid with
  its original accepted reader/full-content gate; frozen detector inference on
  explicitly synthetic Czech text; J-LENS inference on BF16 Apertus with the
  original frozen transfer probe **passed**. The previous Hybrid was restored.
  BF16's first content/probe check was substantially slower than interface tests.
  These checks establish integration, not detector accuracy or broad method
  validity. Research datasets and historical acceptance limitations are unchanged.
- Updated wheel/sdist built. The wheel passed disposable-venv checks outside
  the checkout with only NumPy, including a model-free management launch.

The current connections deliberately have different scopes: this TUNED reader
and the frozen detector are Hybrid-specific, while J-LENS is BF16-only. A green
package badge does not override the native reader's stricter runtime checks.

## Forking Paths / Rozcestí naming — 6 November 2026

The application brand is now **Rozcestí** in Czech and **Forking Paths** in
English. **Probability Garden / Pravděpodobnostní zahrada** remains the name of
the branching-probability view; Apertus and EuroLLM remain model identities.
No numerical method, checkpoint or experimental stimulus was changed.

- Full Python suite: **488 passed, 17 skipped**. Added contracts distinguish
  the product brand from the garden view and require product-branded page titles.
- Browser checks with the installed Chrome: **34 general + 6 administration +
  4 addon tests passed**. A new regression switches the brand, accessible home
  link and page titles between Czech and English, including navigation to model
  management, without changing the garden view's name.
- Desktop helper suite: **12 passed**, including the original profile path,
  unchanged bundle identifier and a browser-safe shared branding module.
- The unsigned Apple-silicon **Forking Paths.app** built successfully. The
  native window was opened on the new repository's local server, the real
  Hybrid distribution loaded, and Czech → English → Czech branding was checked
  interactively. The application was left open in Czech.
- Existing Hybrid TUNED/detector connections remained available; the J-LENS
  connection remained BF16-only. This update did not repeat their earlier
  numerical reader/detector validation or edit the external packages.
- Both `forking-paths` and the original `apertus-workbench` command resolve to
  the same entry point and report identical data/cache/registry locations.
  The Python namespace, environment variables, data paths and native bundle
  identity are deliberately unchanged to preserve existing installations.
- Wheel/sdist built successfully. The wheel passed the disposable-environment
  gate outside the checkout with only NumPy, including both command names,
  synthetic probability requests and model-free model-management setup.
  No model weights or inference frameworks were downloaded by that check.

The repository is still local. No GitHub repository, push or package publication
was performed during this naming update.
