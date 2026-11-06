"""Small command-line entry point; inference dependencies stay optional."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .registry_config import ADAPTERS, load_registry, validate_model_files
    from .settings import cache_directory, data_directory, registry_path
except ImportError:
    from registry_config import ADAPTERS, load_registry, validate_model_files
    from settings import cache_directory, data_directory, registry_path


def check_model(registry, model_id: str) -> list[str]:
    registration = registry.models[model_id]
    required = [registry.models[key] for key in ("bf16", "8bit")] if registration.kind == "apertus-hybrid" else [registration]
    errors = []
    for model in required:
        issues = validate_model_files(model.path)
        errors += [f"{model.id}: {issue}" for issue in issues]
        if not issues:
            config = json.loads((model.path / "config.json").read_text())
            try:
                ADAPTERS[model.adapter].validate(config)
            except ValueError as exc:
                errors.append(f"{model.id}: {exc}")
            if registry.backend == "torch" and (config.get("quantization") or config.get("quantization_config")):
                errors.append(f"{model.id}: Torch requires original, unquantized HF safetensors, not MLX/GGUF weights.")
    if not errors and registration.kind == "apertus-hybrid":
        # Same compatibility gate used by the live loader, without initializing Metal.
        try:
            from .server import ModelState
        except ImportError:
            from server import ModelState
        state = ModelState(model_path=registration.path, model_paths=registry.model_paths(),
                           active_model_id=model_id, backend_name=registry.backend, registry=registry)
        if not state._model_available_unlocked(model_id, registration.path):
            errors.append("Hybrid sources have incompatible configuration/tokenizers or missing BF16 input/output weights.")
    return errors


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Pass through server flags so there is only one authoritative parser.
    if argv and argv[0] in {"serve", "demo"}:
        demo = argv.pop(0) == "demo"
        if demo:
            print("DEMO: synthetic fixtures only — not model measurements.", flush=True)
            argv = ["--mock", *argv]
        from . import server
        previous = sys.argv
        try:
            sys.argv = [previous[0], *argv]
            server.main()
        finally:
            sys.argv = previous
        return
    parser = argparse.ArgumentParser(description="Local Apertus Workbench (nothing is downloaded on startup).")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="Run a local model (see serve --help)")
    sub.add_parser("demo", help="Synthetic interface preview, no model weights")
    sub.add_parser("paths", help="Show user data/cache/config locations")
    models = sub.add_parser("models", help="List, validate, download or convert registered checkpoints")
    operations = models.add_subparsers(dest="operation", required=True)
    for name in ("list", "check", "download", "convert"):
        operation = operations.add_parser(name)
        if name != "list":
            operation.add_argument("model_id", nargs="?" if name == "check" else None)
        operation.add_argument("--registry", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "paths":
        print(json.dumps({"data": str(data_directory()), "cache": str(cache_directory()),
                          "registry": str(registry_path())}, indent=2))
        return
    if args.command != "models":
        parser.print_help()
        return
    try:
        registry = load_registry(args.registry or registry_path())
        if args.operation == "list":
            print(f"Registry: {registry.path}\nBackend: {registry.backend}; default: {registry.default_model}")
            for model_id, model in registry.models.items():
                ready = not check_model(registry, model_id)
                print(f"{model_id:20} {'ready' if ready else 'not ready':10} {model.label}\n  {model.path} [{model.adapter}; {model.kind}]")
            return
        if args.model_id is None:
            args.model_id = registry.default_model
        if args.model_id not in registry.models:
            raise ValueError(f"Unknown model ID: {args.model_id}; available: {', '.join(registry.models)}")
        if args.operation == "check":
            errors = check_model(registry, args.model_id)
            if errors:
                raise ValueError("\n".join(errors))
            print(f"{args.model_id}: local files and adapter metadata OK. Weight contents are checked when loaded; this is not a numerical validation.")
        else:
            from .model_setup import convert_model, download_model
            action = download_model if args.operation == "download" else convert_model
            action(registry, args.model_id)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
