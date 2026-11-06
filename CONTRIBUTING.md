# Contributing

Use Python 3.11/3.12 and an isolated environment. Keep model weights, local
paths, corpora, archives and credentials out of Git. Contributions are made
under the project's Apache-2.0 license.

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
python tools/verify_wheel.py dist/apertus_workbench-0.1.0a1-py3-none-any.whl
cd apertus/garden/e2e
npm ci
npx playwright install chromium
npm run test:stub
npm run test:admin
npm run test:addons
cd ../desktop
npm ci
npm test
```

UI tests use synthetic fixtures. `GARDEN_PYTHON` can select the test server's
interpreter. `GARDEN_BROWSER_CHANNEL=chrome` optionally uses an installed Chrome
instead of Playwright's Chromium. Tests use dedicated ports/data, not your
normal archive. Real-model suites are explicit opt-ins and may require large
amounts of RAM; see the test code's checkpoint assumptions.

`test:admin` uses a separate first-run server and temporary registry. Its
download/inference doubles exist only in the test launcher, never as a
production environment switch. Cancellation, conflict guards and exact backup
behaviour also have Python tests; tiny numerical suites exercise real admin
loading/restoration on MLX and Torch. Keep all such tests free of large downloads.

`test:addons` runs a separate synthetic MLX-metadata server on its own port.
It covers connect/verify/enable, model-specific availability, checksum failure,
CZ/EN and non-destructive disconnection. Placeholder tensors are not numerical
reader validation. Real artifact checks are separate and use private local
packages; never copy those packages, receipts, corpora or absolute paths into Git.

`test_predictive_detector.py` contains historical regressions against private
calibration assets and skips them if absent. No private corpus is required by
the default portable gate. Exact Apertus token-path tests can use
`APERTUS_TEST_TOKENIZER=/path/to/tokenizer.json`. Tiny Torch tests run only when
the optional Torch stack is installed. These skips are not numerical model
validation; report them honestly.

Native Metal checks use tiny random local models and never download weights:
`APERTUS_TEST_MLX=1 python -m pytest apertus/garden/tests/test_registry_mlx.py -q`.
Run them in an environment with the MLX extra and access to the GPU.

Add new checkpoints through the [registry](docs/MODELS.md). New architecture
hooks must be numerically checked against the backend's native forward. Do not
grant every capability to an unfamiliar model. Keep raw logits, probabilities,
readout and intervention claims distinct.

Maintain Czech and English together. Translations are served locally; the
catalogue-generation utility is a developer tool, not a runtime dependency.
Keep the English `README.md` and Czech `README.cs.md` in sync as well.
Do not send private text to a translation or inference service.
Curated model-management translations live in
`apertus/garden/scripts/model-admin-en.json`; regenerate the local catalogue
without a translation service using
`python apertus/garden/scripts/build-i18n-catalog.py --curated-only`.

Before publication, follow [docs/RELEASE.md](docs/RELEASE.md). Do not attach real
user documents or detector results to bug reports. Minimize examples to safe,
synthetic text and include backend/version/precision and a reproducible trace.
