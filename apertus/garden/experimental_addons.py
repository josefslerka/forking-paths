"""Explicit local connections to reviewed research-artifact formats.

Configuration/verification receipts belong to user data, never the checkout.
Connecting does not execute code, load an LLM, copy, download or delete packages.
File verification is serialized on the existing numerical worker. Native readers
retain their stricter full-content/transfer checks before actual inference.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from . import jacobian_lens, tuned_lens
    from .model_registry import read_json
    from .predictive_detector import PredictiveDetector, verify_legacy_runtime_code
except ImportError:
    import jacobian_lens
    import tuned_lens
    from model_registry import read_json
    from predictive_detector import PredictiveDetector, verify_legacy_runtime_code

KINDS = {"tuned": "RAW × TUNED", "jlens": "RAW × J-LENS", "detector": "AI detektor"}


class AddonError(ValueError):
    def __init__(self, code: str, message: str, *, details=None):
        super().__init__(message)
        self.code, self.details = code, details


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def stamp(path: Path) -> list:
    try:
        value = path.stat()
        return [str(path), value.st_size, value.st_mtime_ns]
    except FileNotFoundError:
        return [str(path), None, None]


def read_document(path: Path) -> dict:
    if path.stat().st_size > 2_000_000:
        raise ValueError("Artifact metadata exceeds 2 MB.")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("Artifact metadata must be a JSON object.")
    return value


class ExperimentalAddons:
    def __init__(self, state, data_root: Path, *, config_lock: Callable, readonly=False):
        self.state, self.data_root, self.readonly = state, data_root, readonly
        self.path = data_root / "experimental-addons.json"
        self.config_lock = config_lock
        self._identity_cache = {}
        self.config, self.revision = self._read()

    def _read(self) -> tuple[dict, str]:
        raw = self.path.read_bytes() if self.path.exists() else b""
        if len(raw) > 2_000_000:
            raise AddonError("addon_config", "Experimental-addon settings exceed 2 MB.")
        value = json.loads(raw) if raw else {"schema_version": 1, "addons": {}}
        if not isinstance(value, dict) or value.get("schema_version") != 1 or set(value) != {"schema_version", "addons"} or not isinstance(value["addons"], dict):
            raise AddonError("addon_config", "Unsupported experimental-addon settings.")
        for kind, entry in value["addons"].items():
            if kind not in KINDS or not isinstance(entry, dict) or set(entry) - {"path", "enabled", "verification"}:
                raise AddonError("addon_config", "Unknown experimental-addon registration.")
            if not isinstance(entry.get("path"), str) or not Path(entry["path"]).is_absolute() or type(entry.get("enabled")) is not bool:
                raise AddonError("addon_config", "Invalid experimental-addon path or enabled state.")
            if entry.get("verification") is not None and not isinstance(entry["verification"], dict):
                raise AddonError("addon_config", "Invalid experimental-addon verification receipt.")
        return value, hashlib.sha256(raw).hexdigest()

    def check_revision(self, expected):
        _value, current = self._read()
        if expected != current or current != self.revision:
            raise AddonError("addon_conflict", "Experimental-addon settings changed. Refresh or restart after an external edit.")

    def _save(self, value: dict):
        if self.readonly:
            raise AddonError("demo_readonly", "Experimental addons are read-only in this launch mode.")
        self.data_root.mkdir(parents=True, exist_ok=True)
        with self.config_lock(self.path):
            self.check_revision(self.revision)
            previous = self.path.read_bytes() if self.path.exists() else b""
            if previous:
                backups = self.data_root / "experimental-addon-backups"
                backups.mkdir(exist_ok=True)
                descriptor, backup = tempfile.mkstemp(prefix="addons-", suffix=".json", dir=backups)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(previous)
            raw = canonical(value) + b"\n"
            descriptor, temporary = tempfile.mkstemp(prefix=".addons-", suffix=".json", dir=self.data_root)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(raw); stream.flush(); os.fsync(stream.fileno())
                self.check_revision(self.revision)
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)  # Exact fresh temporary file owned by this edit.
            self.config, self.revision = value, hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _kind(kind):
        if not isinstance(kind, str) or kind not in KINDS:
            raise AddonError("unknown_addon", "Unknown experimental-addon format.")
        return kind

    def model_identity(self, model_id: str) -> dict | None:
        registry = self.state.registry
        if not registry or model_id not in registry.models:
            return None
        model = registry.models[model_id]
        sources = [("quantized_body", model.path), ("bf16_io", registry.models["bf16"].path)] if model.kind == "apertus-hybrid" else [("model", model.path)]
        files = []
        for _role, directory in sources:
            files += [directory / name for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")]
            files += sorted(directory.glob("*.safetensors"))
        signature = hashlib.sha256(canonical([registry.backend, model.adapter, model.kind, registry.source_identity(model_id), [stamp(path) for path in files]])).hexdigest()
        key = (model_id, signature)
        if key not in self._identity_cache:
            try:
                from .server import metadata_fingerprint
            except ImportError:
                from server import metadata_fingerprint
            fingerprint, _inventory = metadata_fingerprint(model_id, sources)
            config = read_json(model.path / "config.json")
            tokenizer = model.path / "tokenizer.json"
            self._identity_cache = {**dict(list(self._identity_cache.items())[-31:]), key: {
                "signature": signature, "metadata_fingerprint": fingerprint, "config": config,
                "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest() if tokenizer.is_file() else None,
                "backend": registry.backend, "adapter": model.adapter,
            }}
        return self._identity_cache[key]

    @staticmethod
    def probe_tokenizer_matches(model_path: Path, probe_path: Path) -> bool:
        # HF->MLX tokenizer JSON serialization can differ bytewise. The existing
        # J-Lens contract uses exact frozen input IDs and a later numeric probe.
        try:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
            probe = read_document(probe_path)
            records = probe.get("records", [])
            return bool(records) and all(tokenizer.encode(row["prompt"]).ids == row["input_ids"] for row in records)
        except (ImportError, OSError, ValueError, KeyError, TypeError):
            return False

    def inspect(self, kind: str, raw_path: str) -> dict:
        self._kind(kind)
        if not isinstance(raw_path, str) or not raw_path.strip() or len(raw_path) > 4096 or "\0" in raw_path or not Path(raw_path).expanduser().is_absolute():
            raise AddonError("invalid_addon_path", "Choose an absolute local artifact directory.")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise AddonError("invalid_addon_path", "The artifact directory does not exist.")
        artifacts, files, problems = [], {}, []
        def remember(file: Path, expected: str | None = None):
            if not file.is_file():
                raise ValueError(f"Missing artifact file: {file.name}")
            files[str(file.resolve())] = {"path": str(file.resolve()), "expected_sha256": expected}
        if kind in {"tuned", "jlens"}:
            manifests = [path / "manifest.json"] if (path / "manifest.json").is_file() else sorted(path.glob("*/manifest.json"))
            if not 1 <= len(manifests) <= 32:
                raise AddonError("invalid_addon_files", "Choose one artifact or a catalog containing 1–32 artifact manifests.")
            for file in manifests:
                try:
                    submitted = read_document(file)
                    if kind == "tuned":
                        manifest = tuned_lens.validate_manifest(submitted, artifact_dir=file.parent)
                        if not tuned_lens.artifact_is_usable(manifest):
                            raise ValueError("The reader did not pass its held-out acceptance gate.")
                        entries = [{"name": manifest[key]["file"], "sha256": manifest[key]["sha256"]} for key in ("weights", "metrics")]
                        modes = [manifest["model"]["model_id"]]
                    else:
                        manifest = jacobian_lens.validate_manifest(submitted, artifact_dir=file.parent)
                        entries = list(jacobian_lens.runtime_file_entries(manifest).values())
                        modes = ["bf16"]
                    remember(file)
                    for entry in entries:
                        target = (file.parent / entry["name"]).resolve()
                        if not target.is_relative_to(file.parent.resolve()):
                            raise ValueError("Artifact file escapes its selected directory.")
                        remember(target, entry["sha256"])
                    compatible = []
                    for model_id in modes:
                        identity = self.model_identity(model_id)
                        if not identity or identity["backend"] != "mlx" or identity["adapter"] != "apertus":
                            continue
                        config = identity["config"]
                        if config.get("model_type") != "apertus" or config.get("num_hidden_layers") != 32:
                            continue
                        matches = identity["metadata_fingerprint"] == manifest["model"]["metadata_fingerprint"] if kind == "tuned" else (
                            not config.get("quantization", config.get("quantization_config")) and config.get("hidden_size") == 4096 and
                            config.get("vocab_size") == 131072 and self.probe_tokenizer_matches(
                                self.state.registry.models[model_id].path,
                                file.parent / manifest["files"]["transfer_probe_json"]["name"]))
                        if matches:
                            compatible.append(model_id)
                    artifacts.append({"id": manifest["artifact_id"], "model_modes": modes, "compatible_models": compatible,
                                      "runtime_check": "full_model_content_hash" if kind == "tuned" else "bf16_transfer_probe"})
                except (OSError, ValueError, KeyError, TypeError, tuned_lens.TunedLensError, jacobian_lens.JacobianLensError) as exc:
                    problems.append(f"{file.parent.name}: {exc}")
        else:
            try:
                deployment_path = path / "runtime/detector_v2_deployment.json"
                lock_path = path / "expansion_v1/model_v2/model_lock.json"
                bundle_path = path / "expansion_v1/model_v2/inference_bundle_v2.json"
                challenger_path = path / "runtime/detector_pair_challenger.json"
                deployment, lock, bundle, challenger = [read_document(item) for item in (deployment_path, lock_path, bundle_path, challenger_path)]
                if deployment.get("status") != "deployed_frozen_v2" or bundle.get("schema_version") != 2 or bundle.get("reader", {}).get("required_model_id") != "hybrid":
                    raise ValueError("Unsupported frozen detector package.")
                for file in (deployment_path, lock_path, bundle_path, challenger_path):
                    remember(file)
                # A frozen legacy detector includes external provenance references.
                # These are checked as data. Executable extractors have a separate
                # reviewed hash allowlist, never trust a hash supplied by a package.
                def references(item):
                    if isinstance(item, dict):
                        if isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str):
                            remember(Path(item["path"]), item["sha256"])
                        for value in item.values(): references(value)
                    elif isinstance(item, list):
                        for value in item: references(value)
                for item in (deployment, lock, bundle, challenger):
                    references(item)
                verify_legacy_runtime_code(path, bundle)
                identity = self.model_identity("hybrid")
                config = identity["config"] if identity else {}
                quantization = config.get("quantization", config.get("quantization_config", {})) or {}
                compatible = ["hybrid"] if identity and identity["backend"] == "mlx" and identity["adapter"] == "apertus" and config.get("model_type") == "apertus" and config.get("num_hidden_layers") == 32 and quantization.get("bits") == 8 and identity["tokenizer_sha256"] == bundle["reader"]["tokenizer_json_sha256"] else []
                artifacts.append({"id": "predictive-stylometry-frozen-v2", "model_modes": ["hybrid"],
                                  "compatible_models": compatible, "runtime_check": "frozen_runtime_and_local_tokenizer"})
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                problems.append(str(exc))
        compatible = sorted({model for artifact in artifacts for model in artifact["compatible_models"]})
        return {"kind": kind, "path": str(path), "files_ok": bool(artifacts) and not problems, "errors": problems,
                "artifacts": artifacts, "compatible_models": compatible, "files": list(files.values())}

    def connect(self, kind, path):
        inspected = self.inspect(kind, path)
        if not inspected["files_ok"]:
            raise AddonError("invalid_addon_files", "The artifact directory is incomplete or unsupported.", details=inspected["errors"])
        value = copy.deepcopy(self.config)
        value["addons"][kind] = {"path": inspected["path"], "enabled": False}
        self._save(value)
        self.state.refresh_experimental_addons()

    def verify(self, kind, *, cancelled=lambda: None, progress=lambda text: None):
        self._kind(kind)
        entry = self.config["addons"].get(kind)
        if not entry:
            raise AddonError("addon_missing", "Connect the package first.")
        inspected = self.inspect(kind, entry["path"])
        if not inspected["files_ok"]:
            raise AddonError("invalid_addon_files", "The artifact directory did not pass validation.", details=inspected["errors"])
        checked = []
        for item in inspected["files"]:
            cancelled(); file = Path(item["path"]); before = stamp(file)
            progress(file.name)
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    cancelled(); digest.update(block)
            after = stamp(file)
            if before != after or (item["expected_sha256"] and digest.hexdigest() != item["expected_sha256"]):
                raise AddonError("invalid_addon_files", f"Artifact file changed or its SHA-256 does not match: {file.name}")
            checked.append({"stamp": after, "sha256": digest.hexdigest()})
        if kind == "detector":
            if "hybrid" not in inspected["compatible_models"]:
                raise AddonError("addon_incompatible", "The frozen detector requires its matching Apertus Hybrid.")
            self.create_detector(Path(entry["path"]), timing_log=False)._ensure_loaded()
        report = {"checked_at": datetime.now(timezone.utc).isoformat(), "files": checked,
                  "artifacts": inspected["artifacts"], "compatible_models": inspected["compatible_models"],
                  "model_signatures": {model: self.model_identity(model)["signature"] for model in inspected["compatible_models"]},
                  "scope": "artifact_sha256_and_model_metadata; native_runtime_checks_still_required"}
        cancelled()
        value = copy.deepcopy(self.config)
        value["addons"][kind]["verification"] = report
        self._save(value)
        self.state.refresh_experimental_addons()
        return {"kind": kind, "package_verified": True, "compatible_models": inspected["compatible_models"]}

    def current_report(self, kind):
        entry = self.config["addons"].get(kind, {})
        report = entry.get("verification")
        if not isinstance(report, dict) or not isinstance(report.get("files"), list) or not report["files"] or not isinstance(report.get("compatible_models"), list) or not isinstance(report.get("model_signatures"), dict):
            return None
        try:
            if any(stamp(Path(item["stamp"][0])) != item["stamp"] for item in report["files"]):
                return None
            if any(not self.model_identity(model) or self.model_identity(model)["signature"] != signature for model, signature in report.get("model_signatures", {}).items()):
                return None
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
            return None
        return report

    def edit(self, kind, *, action, enabled=None):
        self._kind(kind)
        value = copy.deepcopy(self.config)
        if kind not in value["addons"]:
            raise AddonError("addon_missing", "Connect the package first.")
        if action == "disconnect":
            del value["addons"][kind]
        elif action == "set_enabled" and type(enabled) is bool:
            report = self.current_report(kind)
            if enabled and (not report or not report["compatible_models"]):
                raise AddonError("addon_unverified", "Verify a compatible artifact package before enabling it.")
            value["addons"][kind]["enabled"] = enabled
        else:
            raise AddonError("invalid_action", "Unknown experimental-addon action.")
        self._save(value)
        self.state.refresh_experimental_addons()

    def availability(self, model_id):
        result = {}
        for kind in KINDS:
            entry = self.config["addons"].get(kind)
            report = self.current_report(kind) if entry else None
            code = "addon_missing" if not entry else "addon_unverified" if not report else "addon_incompatible" if model_id not in report["compatible_models"] else "addon_disabled" if not entry["enabled"] else "ready"
            result[kind] = {"available": code == "ready", "code": code,
                            "compatible_models": report["compatible_models"] if report else []}
        return result

    def snapshot(self):
        models = self.state.registry.models if self.state.registry else {}
        active = self.state.active_model_id if self.state.model is not None else None
        rows = []
        for kind, label in KINDS.items():
            entry = self.config["addons"].get(kind)
            report = self.current_report(kind) if entry else None
            rows.append({"kind": kind, "label": label, "connected": entry is not None,
                         "path": entry["path"] if entry else None, "enabled": entry.get("enabled", False) if entry else False,
                         "status": "missing" if not entry else "unverified" if not report else "incompatible" if not report["compatible_models"] else "enabled" if entry["enabled"] else "ready",
                         "compatible_models": [{"id": model, "label": models[model].label} for model in report["compatible_models"] if model in models] if report else [],
                         "active_available": bool(active and self.availability(active)[kind]["available"]),
                         "verification": report})
        return {"config_path": str(self.path), "revision": self.revision, "readonly": self.readonly, "items": rows}

    def root(self, kind):
        entry = self.config["addons"].get(kind)
        return Path(entry["path"]) if entry else None

    def create_detector(self, root=None, *, timing_log=True):
        root = root or self.root("detector")
        registry = self.state.registry
        if root is None or not registry or "bf16" not in registry.models:
            raise AddonError("addon_missing", "The detector package or its original tokenizer is not connected.")
        return PredictiveDetector(artifact_root=root, model_dir=registry.models["bf16"].path,
                                  timing_log_path=self.data_root / "detector-runtime/app_analysis_timings.csv" if timing_log else None)
