# Third-party components and model weights

The root Apache-2.0 license applies to this workbench's own code/assets. It does
not relicense external packages, downloaded models, private calibration data,
or an Electron/Chromium binary. Preserve the actual notices shipped with any
components you redistribute. This file is an inventory, not a substitute for
their complete licenses.

Model weights are **not included**. Source and license references for presets:

- [Apertus-8B-2509](https://huggingface.co/swiss-ai/Apertus-8B-2509): official
  checkpoint/license in its source repository (Apache-2.0).
- [EuroLLM-9B-2512](https://huggingface.co/utter-project/EuroLLM-9B-2512): official
  checkpoint/license in its source repository (Apache-2.0).

Your added models may have different terms. Quantizing a model does not remove
the original terms or attribution obligations. Include exact model citations
and checkpoint revisions in research outputs.

Dependencies are acquired separately by pip/npm. Important projects include
NumPy (BSD), MLX/MLX-LM (MIT), PyTorch (BSD-style plus bundled third-party
notices), Transformers/Hugging Face Hub/Accelerate (Apache-2.0), safetensors
(Apache-2.0), pytest (MIT), Beautiful Soup (MIT), Playwright (Apache-2.0),
Electron (MIT and Chromium/other bundled notices), and electron-builder (MIT).
Consult the installed package's license files for the precise version and its
transitive dependencies. Optional extras may install additional dependencies.

No third-party JavaScript frameworks, external fonts, or model weights are
vendored into the web interface. Its illustrations/icons are source assets of
the workbench. Model/provider names are descriptive; no endorsement is implied.
