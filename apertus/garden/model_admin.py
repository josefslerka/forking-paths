"""Local model administration. One TOML registry, atomic edits, no weight deletion.

Long operations run on the model worker. Downloads/conversions use cancellable
child processes and fresh, owned staging directories, never a shell command.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import unicodedata
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .experimental_addons import AddonError, ExperimentalAddons
    from .cli import check_model
    from .model_registry import model_descriptor, read_json
    from .registry_config import ADAPTERS, ModelRegistry, load_registry, parse_registry, validate_model_files
    from .settings import cache_directory, data_directory
except ImportError:
    from experimental_addons import AddonError, ExperimentalAddons
    from cli import check_model
    from model_registry import model_descriptor, read_json
    from registry_config import ADAPTERS, ModelRegistry, load_registry, parse_registry, validate_model_files
    from settings import cache_directory, data_directory

HERE = Path(__file__).resolve().parent
PRESETS = HERE / "model_presets.toml"
CAPABILITIES = ("tokenization", "garden", "surprise", "contrast", "model-comparison", "embeddings", "raw", "steering", "attribution", "patching", "tuned", "jlens", "detector")
# Rounded planning estimates, not a remote manifest or a RAM-fit guarantee.
SOURCE_BYTES = {"bf16": 17_000_000_000, "eurollm9-source": 37_000_000_000, "eurollm9-hf": 37_000_000_000}
OUTPUT_BYTES = {"8bit": 9_000_000_000, "4bit": 5_000_000_000, "eurollm9": 8_000_000_000}


class AdminError(ValueError):
    def __init__(self, code: str, message: str, *, details: Any = None):
        super().__init__(message)
        self.code, self.details = code, details


class Cancelled(Exception):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dependency_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def registry_bytes(value: dict) -> bytes:
    """A deterministic serializer for the deliberately small registry schema."""
    lines = ["# Managed by Apertus Workbench. Previous content is backed up before edits."]
    for key in ("schema_version", "backend", "default_model", "enable_research_artifacts"):
        if key in value:
            lines.append(f"{key} = {json.dumps(value[key], ensure_ascii=False)}")
    for model_id, fields in value["models"].items():
        lines += ["", f"[models.{json.dumps(model_id)}]"]
        lines += [f"{key} = {json.dumps(item, ensure_ascii=False)}" for key, item in fields.items()]
    return ("\n".join(lines) + "\n").encode("utf-8")


@contextmanager
def registry_lock(path: Path):
    """Cross-process advisory lock; optimistic hashes also protect manual edits."""
    handle = os.fdopen(os.open(path.with_name(path.name + ".lock"), os.O_CREAT | os.O_RDWR, 0o600), "r+b")
    try:
        if os.name == "nt":
            import msvcrt
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"0"); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    except (BlockingIOError, PermissionError) as exc:
        raise AdminError("registry_conflict", "The registry is being edited by another process.") from exc
    finally:
        handle.close()  # Closing releases the OS lock, including after an error.


def _stamp(path: Path) -> tuple:
    paths = [path, *(path / name for name in ("config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "model.safetensors.index.json"))]
    if path.is_dir():
        paths += sorted(path.glob("*.safetensors"))
    result = []
    for item in paths:
        try:
            stat = item.stat()
            result.append((str(item), stat.st_size, stat.st_mtime_ns))
        except FileNotFoundError:
            result.append((str(item), None))
    return tuple(result)


class ModelAdmin:
    def __init__(self, state: Any, data_root: Path | None = None):
        self.state = state
        self.data_root = (data_root or data_directory()).resolve()
        self.lock = threading.RLock()
        self.readonly = state.mock or state.registry is None
        self.registry = state.registry or load_registry(PRESETS)
        self.jobs: OrderedDict[str, dict] = OrderedDict()
        self.plans: OrderedDict[str, dict] = OrderedDict()
        self.checks: dict[str, dict] = {}
        self.cache: dict[tuple, dict] = {}
        self.child: subprocess.Popen | None = None
        self.addons = ExperimentalAddons(state, self.data_root, config_lock=registry_lock, readonly=self.readonly)
        if not self.readonly:
            state.experimental_addons = self.addons
            state.refresh_experimental_addons(initial=True)

    def _current(self) -> tuple[ModelRegistry, dict, bytes]:
        registry = load_registry(self.registry.path)
        raw = self.registry.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != registry.sha256:
            raise AdminError("registry_conflict", "Registry changed while reading it. Refresh and retry.")
        return registry, tomllib.loads(raw.decode()), raw

    def _idle(self):
        if self.readonly:
            raise AdminError("demo_readonly" if self.state.mock else "registry_required", "Model administration is read-only in this launch mode.")
        if self.active_job() is not None or self.state.outstanding_operation is not None or self.state.status in {"loading", "switching", "maintenance"}:
            raise AdminError("model_busy", "Wait for the current operation to finish.")

    def active_job(self) -> dict | None:
        return next((job for job in reversed(self.jobs.values()) if job["status"] in {"queued", "running", "cancelling"}), None)

    @staticmethod
    def _revision(registry: ModelRegistry, request: dict):
        if request.get("registry_sha256") != registry.sha256:
            raise AdminError("registry_conflict", "Registry changed. Refresh before making this edit.")

    def _info(self, registry: ModelRegistry, model_id: str) -> dict:
        model = registry.models[model_id]
        required = (registry.models["bf16"].path, registry.models["8bit"].path) if model.kind == "apertus-hybrid" else (model.path,)
        stamp = tuple(_stamp(path) for path in required)
        key = (registry.sha256, model_id, stamp)
        if key not in self.cache:
            errors = check_model(registry, model_id)
            descriptor = model_descriptor(model_id, model.path, model.label, backend=registry.backend, mock=False,
                                          registration=model, enable_research_artifacts=registry.enable_research_artifacts)
            size = sum(file.stat().st_size for file in model.path.glob("*.safetensors") if file.is_file()) if model.path.is_dir() else 0
            signature = hashlib.sha256(repr((model.path, model.adapter, registry.backend, stamp)).encode()).hexdigest()
            self.cache[key] = {"id":model_id,"label":model.label,"path":str(model.path),"adapter":model.adapter,
                               "kind":model.kind,"enabled":model.enabled,"files_ok":not errors,"errors":errors,
                               "bytes":size,"layers":descriptor["layers"],"family":descriptor["family"],
                               "capabilities":descriptor["capabilities"],"checkpoint":registry.source_identity(model_id),
                               "signature":signature}
            if len(self.cache) > 128:
                self.cache.pop(next(iter(self.cache)))
        value = copy.deepcopy(self.cache[key])
        if self.state.experimental_addons is not None:
            availability = self.addons.availability(model_id)
            value["capabilities"] = [item for item in value["capabilities"] if item not in availability]
            value["capabilities"] += [kind for kind, item in availability.items() if item["available"]]
            value["experimental_addons"] = availability
        value.update({"default":model_id == registry.default_model,
                      "active":not self.state.mock and self.state.model is not None and model_id == self.state.active_model_id,
                      "check":copy.deepcopy(self.checks.get(value["signature"]))})
        value["required_by"] = [other.id for other in registry.models.values() if other.convert_from == model_id or
                                (other.kind == "apertus-hybrid" and model_id in {"bf16", "8bit"})]
        return value

    def snapshot(self) -> dict:
        with self.lock, self.state.state_lock:
            registry, _data, _raw = self._current()
            backend_module = "mlx_lm" if registry.backend == "mlx" else "torch"
            modules = (backend_module,"transformers","safetensors")
            installed = all(dependency_available(module) for module in modules)
            legacy = self.readonly and not self.state.mock
            profiles = ["bf16", "8bit", "4bit", "hybrid", "eurollm9"] if registry.backend == "mlx" else ["bf16", "eurollm9-hf"]
            return {"registry_sha256":registry.sha256,"applied_sha256":self.state.registry.sha256 if self.state.registry else None,
                    "registry_path":str(registry.path) if not legacy else "", "backend":registry.backend if not legacy else self.state.backend_name,"backend_installed":installed,
                    "readonly":self.readonly,"mock":self.state.mock,"status":self.state.status,
                    "error":self.state.error,
                    "active_model_id":self.state.active_model_id if self.state.model is not None else None,
                    "models":[self._info(registry, model_id) for model_id in registry.models] if not legacy else [],
                    "profiles":[{"id":model_id,"license":"Apache-2.0"} for model_id in profiles] if not legacy else [],
                    "cache_path":str(cache_directory() / "models"),
                    "job":self.job_status(next(reversed(self.jobs))) if self.jobs else None,
                    "busy":bool(self.active_job() or self.state.outstanding_operation),
                    "unavailable_capabilities":[capability for capability in CAPABILITIES if capability not in {"tokenization","garden","surprise","contrast","model-comparison"}],
                    "backup_directory":str(self.data_root / "model-registry-backups")}

    def addon_action(self, request: dict) -> dict:
        with self.lock, self.state.state_lock:
            self._idle()
            registry, _value, _raw = self._current()
            self._revision(registry, request)
            if self.state.registry.sha256 != registry.sha256:
                raise AdminError("reload_required", "Apply external registry changes first.")
            self.addons.check_revision(request.get("addon_revision"))
            action, kind = request.get("action"), request.get("kind")
            self.addons._kind(kind)
            if action == "inspect":
                return self.addons.inspect(kind, request.get("path"))
            if action == "connect":
                self.addons.connect(kind, request.get("path"))
            elif action in {"set_enabled", "disconnect"}:
                self.addons.edit(kind, action=action, enabled=request.get("enabled"))
            else:
                raise AdminError("invalid_action", "Unknown experimental-addon action.")
            self.cache.clear()
            return self.addons.snapshot()

    def addon_snapshot(self) -> dict:
        with self.lock, self.state.state_lock:
            return self.addons.snapshot()

    def inspect(self, request: dict) -> dict:
        with self.lock:
            registry, _data, _raw = self._current()
            raw_path = request.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip() or len(raw_path) > 4096 or "\0" in raw_path:
                raise AdminError("invalid_model_path", "Choose an absolute local model directory.")
            path = Path(raw_path.strip()).expanduser()
            if not path.is_absolute():
                raise AdminError("invalid_model_path", "Choose an absolute local model directory.")
            path = path.resolve()
            errors = validate_model_files(path)
            config = read_json(path / "config.json")
            adapter = "causal-lm"
            candidate = {"apertus":"apertus", "llama":"llama-full-attention"}.get(config.get("model_type"))
            if candidate:
                try:
                    ADAPTERS[candidate].validate(config)
                    adapter = candidate
                except ValueError:
                    pass
            explicit = request.get("adapter")
            if explicit is not None:
                if not isinstance(explicit, str) or explicit not in ADAPTERS:
                    raise AdminError("invalid_adapter", "Unknown adapter.")
                try:
                    ADAPTERS[explicit].validate(config)
                except ValueError as exc:
                    raise AdminError("invalid_adapter", str(exc)) from exc
                adapter = explicit
            label = path.name or "Local model"
            slug = re.sub(r"[^a-z0-9_-]+", "-", unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode().lower()).strip("-_")[:48] or "local-model"
            if slug.startswith("mock") or slug == "hybrid":
                slug = "local-" + slug
            model_id, suffix = slug, 2
            while model_id in registry.models:
                model_id, suffix = f"{slug}-{suffix}", suffix + 1
            if registry.backend == "torch" and (config.get("quantization") or config.get("quantization_config")):
                errors.append("Torch requires original HF safetensors, not MLX affine weights.")
            return {"path":str(path),"model_id":model_id,"label":label,"adapter":adapter,
                    "family":config.get("model_type"),"layers":config.get("num_hidden_layers"),"files_ok":not errors,
                    "errors":errors,"already_registered":[model.id for model in registry.models.values() if model.path == path and model.kind != "apertus-hybrid"]}

    def _commit(self, value: dict, expected: ModelRegistry, before: bytes) -> ModelRegistry:
        if self.state.registry and self.state.registry.sha256 != expected.sha256:
            raise AdminError("reload_required", "Apply the externally edited registry before saving another change.")
        source = expected.path
        target = self.data_root / "models.toml" if source == PRESETS.resolve() else source
        if source.parent != target.parent:
            for entry in value["models"].values():
                if "path" in entry:
                    entry["path"] = str((source.parent / Path(entry["path"]).expanduser()).resolve())
        content = registry_bytes(value)
        candidate = parse_registry(content, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with registry_lock(target):
            if source.read_bytes() != before or (source != target and target.exists()):
                raise AdminError("registry_conflict", "Registry changed on disk; nothing was overwritten.")
            backup_dir = self.data_root / "model-registry-backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8] + ".toml")
            fd = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(before)
            fd, temporary = tempfile.mkstemp(prefix=".models-", suffix=".toml", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(content); stream.flush(); os.fsync(stream.fileno())
                if source.read_bytes() != before:
                    raise AdminError("registry_conflict", "Registry changed on disk; nothing was overwritten.")
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)  # Exact fresh temporary file owned by this edit.
        self.registry = candidate
        self.state.apply_registry(candidate)
        return candidate

    def edit(self, request: dict) -> dict:
        with self.lock, self.state.state_lock:
            self._idle()
            registry, value, raw = self._current()
            self._revision(registry, request)
            action, model_id = request.get("action"), request.get("model_id")
            if action == "add":
                inspected = self.inspect(request)
                if not inspected["files_ok"]:
                    raise AdminError("invalid_model_files", "The selected directory is not a complete compatible model.", details=inspected["errors"])
                if model_id in registry.models:
                    raise AdminError("duplicate_model", "This model ID is already registered.")
                label = request.get("label")
                if not isinstance(model_id, str) or not isinstance(label, str):
                    raise AdminError("invalid_registration", "Provide a model ID and display name.")
                value["models"][model_id] = {"label":label,"path":inspected["path"],"adapter":inspected["adapter"]}
            else:
                if model_id not in registry.models:
                    raise AdminError("unknown_model", "Unknown model ID.")
                model = registry.models[model_id]
                if action == "set_default":
                    value["default_model"] = model_id
                elif action in {"remove", "set_enabled"}:
                    if model_id == self.state.active_model_id and self.state.model is not None:
                        raise AdminError("active_model", "Select another model before hiding or disconnecting this one.")
                    if model_id == registry.default_model and (action == "remove" or request.get("enabled") is not True):
                        raise AdminError("default_model", "Choose another default model first.")
                    if action == "remove":
                        required = self._info(registry, model_id)["required_by"]
                        if required:
                            raise AdminError("model_dependency", "This registration is required by other profiles.", details=required)
                        del value["models"][model_id]
                    else:
                        if type(request.get("enabled")) is not bool:
                            raise AdminError("invalid_registration", "enabled must be true or false.")
                        value["models"][model_id]["enabled"] = request["enabled"]
                else:
                    raise AdminError("invalid_action", "Unknown registry action.")
            self._commit(value, registry, raw)
            return self.snapshot()

    def _profile_plan(self, registry: ModelRegistry, value: dict, profile_id: str) -> dict:
        presets = tomllib.loads(PRESETS.read_text())
        if registry.backend == "torch":
            if profile_id not in {"bf16", "eurollm9-hf"}:
                raise AdminError("incompatible_profile", "This profile requires the MLX backend.")
            presets["models"]["eurollm9-hf"] = {**presets["models"]["eurollm9-source"], "kind":"native", "label":"EuroLLM 9B · HF"}
        elif profile_id not in {"bf16", "8bit", "4bit", "hybrid", "eurollm9"}:
            raise AdminError("unknown_profile", "Unknown prepared profile.")
        merged, steps, visited, order = copy.deepcopy(value), [], set(), []
        def include(model_id: str):
            if model_id in visited:
                return
            visited.add(model_id)
            desired = presets["models"][model_id]
            entry = merged["models"].get(model_id)
            if entry is None:
                entry = {**desired, "path":str(cache_directory() / "models" / model_id)}
                merged["models"][model_id] = entry
            elif entry.get("kind", "native") != desired.get("kind", "native") or entry.get("adapter", "causal-lm") != desired.get("adapter", "causal-lm"):
                raise AdminError("profile_conflict", "An existing registration with this ID has a different adapter or kind.", details=model_id)
            if not entry.get("enabled", True) and entry.get("kind") != "source":
                raise AdminError("profile_disabled", "Enable this profile before installing it.", details=model_id)
            dependencies = ("bf16", "8bit") if model_id == "hybrid" else (desired["convert_from"],) if desired.get("convert_from") else ()
            for dependency in dependencies:
                include(dependency)
            order.append(model_id)
        include(profile_id)
        # Build the whole graph before validating (Hybrid needs both sources).
        candidate = parse_registry(registry_bytes(merged), registry.path)
        for model_id in order:
            desired = presets["models"][model_id]
            model = candidate.models[model_id]
            ready = not check_model(candidate, model_id)
            operation = "compose" if model.kind == "apertus-hybrid" else "convert" if desired.get("convert_from") else "download"
            if not ready and model.kind != "apertus-hybrid" and model.path.exists():
                raise AdminError("target_exists", "An incomplete target already exists. It will not be overwritten.", details=str(model.path))
            if operation == "download" and not ready and (model.repository != desired.get("repository") or model.revision != desired.get("revision")):
                raise AdminError("profile_conflict", "This registration points to a different source; it will not be replaced.", details=model_id)
            if operation == "convert" and (model.convert_from != desired["convert_from"] or model.bits != desired["bits"]):
                raise AdminError("profile_conflict", "This registration uses a different conversion recipe.", details=model_id)
            steps.append({"model_id":model_id,"operation":operation,"ready":ready,"path":str(model.path),
                          "estimated_bytes":0 if ready else SOURCE_BYTES.get(model_id, OUTPUT_BYTES.get(model_id, 0)),
                          "repository":model.repository,"revision":model.revision})
        estimate = sum(step["estimated_bytes"] for step in steps)
        free = None
        if estimate:
            # Each target can be on a different filesystem; check each location.
            disks = {}
            for step in steps:
                if step["ready"] or not step["estimated_bytes"]:
                    continue
                anchor = Path(step["path"]).parent
                while not anchor.exists():
                    anchor = anchor.parent
                disk = shutil.disk_usage(anchor)
                device = anchor.stat().st_dev
                item = disks.setdefault(device,{"path":str(anchor),"free_bytes":disk.free,"required_bytes":0})
                item["required_bytes"] += step["estimated_bytes"]
            free = list(disks.values())
            if any(item["free_bytes"] < item["required_bytes"] * 1.1 for item in free):
                raise AdminError("disk_space", "Insufficient free disk space for the estimated source and output weights.", details=free)
        return {"profile_id":profile_id,"steps":steps,"estimated_bytes":estimate,"disk":free,"value":merged}

    def plan_install(self, request: dict) -> dict:
        with self.lock:
            registry, value, _raw = self._current()
            self._revision(registry, request)
            profile_id = request.get("profile_id")
            if not isinstance(profile_id, str):
                raise AdminError("unknown_profile", "Select a prepared profile.")
            plan = self._profile_plan(registry, value, profile_id)
            plan.update({"id":uuid.uuid4().hex,"registry_sha256":registry.sha256,"created":time.monotonic()})
            self.plans[plan["id"]] = plan
            while len(self.plans) > 16:
                self.plans.popitem(last=False)
            return {key:copy.deepcopy(item) for key,item in plan.items() if key not in {"value", "created"}} | {"license":"Apache-2.0", "estimated":True}

    def reload(self, request: dict) -> dict:
        with self.lock, self.state.state_lock:
            self._idle()
            registry, _value, _raw = self._current()
            self._revision(registry, request)
            if self.state.model is not None:
                old = self.state.registry
                old_model = old.models[self.state.active_model_id]
                new_model = registry.models.get(self.state.active_model_id)
                if registry.backend != old.backend or new_model is None or not new_model.enabled or (new_model.path, new_model.adapter, new_model.kind) != (old_model.path, old_model.adapter, old_model.kind):
                    raise AdminError("active_model", "The external edit changes the loaded model. Restart after finishing your work.")
            self.state.apply_registry(registry)
            return self.snapshot()

    def start_job(self, request: dict) -> dict:
        with self.lock, self.state.state_lock:
            self._idle()
            registry, _value, raw = self._current()
            self._revision(registry, request)
            if self.state.registry.sha256 != registry.sha256:
                raise AdminError("reload_required", "Apply external registry changes first.")
            action = request.get("action")
            if action == "install":
                plan = self.plans.get(request.get("plan_id"))
                if not plan or time.monotonic() - plan["created"] > 600 or plan["registry_sha256"] != registry.sha256:
                    raise AdminError("plan_expired", "Review a new installation plan before starting.")
                if request.get("confirmed") is not True:
                    raise AdminError("confirmation_required", "Explicitly confirm the disk usage and source downloads.")
                if any(step["operation"]=="convert" and not step["ready"] for step in plan["steps"]) and not dependency_available("mlx_lm"):
                    raise AdminError("dependency_missing", "Install the MLX extra before starting a conversion recipe.")
                if any(step["operation"]=="download" and not step["ready"] for step in plan["steps"]) and not dependency_available("huggingface_hub"):
                    raise AdminError("dependency_missing", "Install the selected inference/download extra before starting a download.")
                registry = self._commit(copy.deepcopy(plan["value"]), registry, raw)
                steps, model_id = copy.deepcopy(plan["steps"]), plan["profile_id"]
            elif action == "addon_verify":
                kind = self.addons._kind(request.get("kind"))
                self.addons.check_revision(request.get("addon_revision"))
                if self.addons.root(kind) is None:
                    raise AddonError("addon_missing", "Connect the package first.")
                steps, model_id = [], registry.default_model
            elif action == "smoke":
                model_id = request.get("model_id")
                if model_id not in registry.model_paths():
                    raise AdminError("unknown_model", "Enable a registered inference model first.")
                errors = check_model(registry, model_id)
                if errors:
                    raise AdminError("invalid_model_files", "Prepare the model files first.", details=errors)
                steps = []
            else:
                raise AdminError("invalid_action", "Unknown background operation.")
            job = {"id":uuid.uuid4().hex,"action":action,"model_id":model_id,"status":"queued","phase":"queued",
                   "steps":steps,"step":0,"logs":[],"error":None,"error_code":None,"result":None,
                   "started_at":utc_now(),"finished_at":None,"cancel":threading.Event()}
            if action == "addon_verify":
                job["kind"] = kind
            self.jobs[job["id"]] = job
            while len(self.jobs) > 8:
                self.jobs.popitem(last=False)
            try:
                self.state.enqueue_model_admin(lambda: self._run_job(job))
            except Exception:
                del self.jobs[job["id"]]
                raise
            return self.job_status(job["id"])

    def job_status(self, job_id: str, *, cancel: bool = False) -> dict:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise AdminError("unknown_job", "This operation no longer exists.")
            if cancel and job["status"] in {"queued", "running", "cancelling"}:
                job["cancel"].set()
                job["status"] = "cancelling"
            return copy.deepcopy({key:value for key,value in job.items() if key != "cancel"})

    def _progress(self, job: dict, **patch):
        with self.lock:
            job.update(patch)

    def _log(self, job: dict, line: str):
        with self.lock:
            job["logs"] = [*job["logs"][-19:], line[:800]]

    @staticmethod
    def _cancelled(job):
        if job["cancel"].is_set():
            raise Cancelled()

    @staticmethod
    def _stop_child(child: subprocess.Popen, *, force: bool = False):
        try:
            if os.name == "posix":
                # Conversion may itself run mlx_lm.convert. Stop the whole owned
                # session, not just its Python supervisor, before removing stage.
                os.killpg(child.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif child.poll() is None:
                child.kill() if force else child.terminate()
        except ProcessLookupError:
            pass

    def _setup_child(self, job: dict, step: dict, registry: ModelRegistry):
        target = registry.models[step["model_id"]].path
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".workbench-install-", dir=target.parent) as temporary:
            config = Path(temporary) / "registry.toml"
            # Absolute paths preserve references when the child has a different cwd.
            value = {"schema_version":1,"backend":registry.backend,"default_model":registry.default_model,
                     "enable_research_artifacts":registry.enable_research_artifacts,"models":{}}
            for model_id, model in registry.models.items():
                fields = {"label":model.label,"path":str(model.path),"adapter":model.adapter,"kind":model.kind,
                          "enabled":model.enabled,"group_size":model.group_size,"repository":model.repository,
                          "revision":model.revision,"convert_from":model.convert_from,"bits":model.bits}
                value["models"][model_id] = {key:item for key,item in fields.items() if item is not None}
            config.write_bytes(registry_bytes(value))
            env = {**os.environ, "PYTHONUNBUFFERED":"1", "HF_HUB_DISABLE_PROGRESS_BARS":"1",
                   "PYTHONPATH":str(HERE.parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", "")}
            child = subprocess.Popen([sys.executable,"-m","apertus.garden.model_setup",step["operation"],step["model_id"],
                                      "--registry",str(config),"--temporary-root",temporary],
                                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,
                                     start_new_session=os.name == "posix")
            self.child = child
            lines = queue.Queue(maxsize=64)
            def read_output():
                for line in child.stdout:
                    try:
                        lines.put_nowait(line.rstrip())
                    except queue.Full:
                        pass
            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            terminated = None
            try:
                while child.poll() is None:
                    if job["cancel"].is_set():
                        if terminated is None:
                            self._stop_child(child); terminated = time.monotonic()
                        elif time.monotonic() - terminated > 5:
                            self._stop_child(child,force=True)
                    try:
                        self._log(job, lines.get(timeout=.1))
                    except queue.Empty:
                        pass
                reader.join(2)
                while not lines.empty():
                    self._log(job, lines.get_nowait())
                self._cancelled(job)
                if child.returncode:
                    raise AdminError("setup_failed", "Model setup failed; existing weights were not overwritten.", details=job["logs"][-5:])
            finally:
                if child.poll() is None:
                    self._stop_child(child,force=True); child.wait(timeout=10)
                elif job["cancel"].is_set() or child.returncode:
                    self._stop_child(child,force=True)
                child.stdout.close()
                self.child = None

    def _run_job(self, job: dict):
        state = self.state
        previous = state.active_model_id if state.model is not None else None
        success = False
        self._progress(job, status="running" if not job["cancel"].is_set() else "cancelling", phase="preparing")
        try:
            self._cancelled(job)
            if job["action"] == "install":
                if state.model is not None:
                    state._release_model_local()
                for index, step in enumerate(job["steps"]):
                    self._cancelled(job)
                    self._progress(job, step=index + 1, phase=step["operation"])
                    if step["operation"] != "compose" and check_model(self.registry, step["model_id"]):
                        self._setup_child(job, step, self.registry)
                    problems = check_model(self.registry, step["model_id"])
                    if problems:
                        raise AdminError("invalid_model_files", "Prepared files did not pass validation.", details=problems)
                self._progress(job, result={"files_ok":True,"model_id":job["model_id"]})
            elif job["action"] == "addon_verify":
                self._progress(job, phase="verifying_addon")
                result = self.addons.verify(job["kind"], cancelled=lambda:self._cancelled(job),
                                            progress=lambda filename:self._log(job,filename))
                self._progress(job, result=result)
            else:
                self._progress(job, phase="loading")
                if state.model is None or state.active_model_id != job["model_id"]:
                    state._switch_model_local(job["model_id"])
                    if state.status != "ready":
                        raise AdminError("load_failed", state.error or "The model could not be loaded.")
                with state.state_lock:
                    state.status = "maintenance"
                self._cancelled(job)
                self._progress(job, phase="checking")
                started = time.perf_counter()
                result = state._next_tokens_local("The capital of France is", 1.0, 3, None)
                probabilities = [item["probability"] for item in result["candidates"]]
                mass = sum(probabilities) + result["other_probability"]
                if not probabilities or any(not math.isfinite(number) or not 0 <= number <= 1 for number in probabilities) or not math.isfinite(mass) or abs(mass - 1) > 1e-6:
                    raise AdminError("invalid_output", "The model returned invalid probabilities.")
                self._cancelled(job)
                with self.lock:
                    info = self._info(self.registry, job["model_id"])
                report = {"passed":True,"checked_at":utc_now(),"duration_ms":round((time.perf_counter()-started)*1000),
                          "scope":"load_and_one_next_token_distribution","probability_sum":mass,"backend":state.backend_name}
                with self.lock:
                    self.checks[info["signature"]] = report
                self._progress(job, result=report)
            success = True
        except Cancelled:
            self._progress(job, status="cancelled", error_code="cancelled")
        except Exception as exc:
            self._progress(job, status="failed", error=str(exc), error_code=getattr(exc,"code","operation_failed"))
        finally:
            with self.lock:
                self.cache.clear()
            if previous and (state.model is None or state.active_model_id != previous):
                self._progress(job, phase="restoring")
                state._switch_model_local(previous)
                if state.status != "ready":
                    self._progress(job, status="failed", error_code="restoration_failed", error=state.error)
            elif state.model is not None:
                with state.state_lock:
                    state.status, state.phase = "ready", None
            else:
                with state.state_lock:
                    state.status, state.phase = "setup_required", None
            if success and job["status"] not in {"failed", "cancelled"}:
                self._progress(job, status="completed")
            self._progress(job, phase="done", finished_at=utc_now())

    def close(self):
        with self.lock:
            job = self.active_job()
            if job:
                job["cancel"].set()
