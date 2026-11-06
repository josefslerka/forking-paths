"""Host support and dependency preflight without importing numerical libraries.

This release's MLX adapter uses Apple Metal. Linux MLX/CUDA is a separate,
unvalidated runtime, not enabled merely because an mlx package is installed.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import shlex
import sys
from pathlib import Path


def recommended_backend() -> str:
    return "mlx" if sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"} else "torch"


def builtin_registry_path() -> Path:
    name = "model_presets.toml" if recommended_backend() == "mlx" else "model_presets.torch.toml"
    return Path(__file__).with_name(name)


def environment_report(backend: str, *, available=None, download: bool = False) -> dict:
    available = available or (lambda module: importlib.util.find_spec(module) is not None)
    supported = backend == "torch" or backend == "mlx" and recommended_backend() == "mlx"
    modules = ["mlx", "mlx_lm", "transformers", "safetensors"] if backend == "mlx" else ["torch", "transformers", "safetensors", "accelerate"]
    if download:
        modules.append("huggingface_hub")
    missing = [module for module in modules if not available(module)]
    root = Path(__file__).resolve().parents[2]
    install_backend = backend if supported else recommended_backend()
    requirement = f"{root}[{install_backend}]" if (root / "pyproject.toml").is_file() else f".[{install_backend}]"
    arguments = [sys.executable, "-m", "pip", "install", "-e", requirement]
    command = ("& " + " ".join("'" + value.replace("'", "''") + "'" for value in arguments)) if os.name == "nt" else shlex.join(arguments)
    return {"backend":backend, "supported":supported, "recommended_backend":recommended_backend(),
            "system":sys.platform, "machine":platform.machine(), "missing_modules":missing,
            "python_executable":sys.executable, "install_command":command,
            "install_backend":install_backend,
            "ready":supported and not missing,
            "code":"unsupported_platform" if not supported else "dependency_missing" if missing else "ready"}
