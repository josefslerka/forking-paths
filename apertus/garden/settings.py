"""User-owned data and model locations; never depend on the author's checkout."""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from .runtime_environment import builtin_registry_path
except ImportError:
    from runtime_environment import builtin_registry_path


def data_directory() -> Path:
    override = os.environ.get("APERTUS_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/Apertus Workbench"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "Apertus Workbench"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "apertus-workbench"


def cache_directory() -> Path:
    override = os.environ.get("APERTUS_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library/Caches/Apertus Workbench"
    if os.name == "nt":
        return data_directory() / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "apertus-workbench"


def registry_path(*, data_root: Path | None = None) -> Path:
    override = os.environ.get("APERTUS_MODEL_REGISTRY")
    if override:
        return Path(override).expanduser().resolve()
    local = Path.cwd() / "models.toml"
    if local.is_file():
        return local
    user_registry = (data_root or data_directory()) / "models.toml"
    return user_registry if user_registry.is_file() else builtin_registry_path()
