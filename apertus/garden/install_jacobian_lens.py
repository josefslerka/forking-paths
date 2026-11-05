#!/usr/bin/env python3
"""Safely install an Apertus Jacobian-Lens runtime ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

try:
    from .jacobian_lens import (
        ArtifactValidationError,
        runtime_file_entries,
        sha256_file,
        validate_manifest,
    )
except ImportError:
    from jacobian_lens import (  # type: ignore
        ArtifactValidationError,
        runtime_file_entries,
        sha256_file,
        validate_manifest,
    )


MAX_MANIFEST_BYTES = 64 * 1024
MAX_MEMBER_BYTES = 300 * 1024 * 1024
MAX_TOTAL_BYTES = 400 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200


def _safe_member_name(name: str) -> bool:
    if not name or "\\" in name or "\x00" in name:
        return False
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and len(path.parts) == 1
        and path.parts[0] not in {".", ".."}
    )


def _member_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return mode != 0 and stat.S_ISLNK(mode)


def install_runtime_zip(
    zip_path: Path,
    root: Path,
    *,
    expected_zip_sha256: str | None = None,
    expected_hidden_size: int = 4096,
    expected_vocab_size: int = 131072,
) -> Path:
    """Verify every byte and atomically install a runtime artifact."""

    zip_path = zip_path.expanduser().resolve()
    root = root.expanduser().resolve()
    if expected_zip_sha256 is not None:
        actual_archive_sha = sha256_file(zip_path)
        if actual_archive_sha != expected_zip_sha256:
            raise ArtifactValidationError("SHA-256 runtime ZIPu nesouhlasí.")

    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ArtifactValidationError("ZIP obsahuje duplicitní jméno.")
        if any(
            info.flag_bits & 0x1
            or info.is_dir()
            or _member_is_symlink(info)
            or not _safe_member_name(info.filename)
            for info in infos
        ):
            raise ArtifactValidationError(
                "ZIP obsahuje nepovolenou cestu, odkaz nebo šifrovaný soubor."
            )
        if "manifest.json" not in names:
            raise ArtifactValidationError("ZIP neobsahuje manifest.json.")
        manifest_info = archive.getinfo("manifest.json")
        if manifest_info.file_size > MAX_MANIFEST_BYTES:
            raise ArtifactValidationError("manifest.json je příliš velký.")
        manifest_bytes = archive.read(manifest_info)
        try:
            submitted = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactValidationError("manifest.json není platný JSON.") from exc
        manifest = validate_manifest(
            submitted,
            expected_hidden_size=expected_hidden_size,
            expected_vocab_size=expected_vocab_size,
        )
        entries = runtime_file_entries(manifest)
        expected_names = {
            "manifest.json",
            *(entry["name"] for entry in entries.values()),
        }
        if set(names) != expected_names:
            missing = sorted(expected_names - set(names))
            extra = sorted(set(names) - expected_names)
            raise ArtifactValidationError(
                f"ZIP má neplatný obsah; chybí {missing}, navíc {extra}."
            )
        total_bytes = sum(info.file_size for info in infos)
        if total_bytes > MAX_TOTAL_BYTES:
            raise ArtifactValidationError("Rozbalený ZIP je příliš velký.")
        for info in infos:
            if info.file_size > MAX_MEMBER_BYTES:
                raise ArtifactValidationError(
                    f"{info.filename} je příliš velký."
                )
            if (
                info.file_size > 0
                and info.compress_size == 0
                or (
                    info.compress_size > 0
                    and info.file_size / info.compress_size
                    > MAX_COMPRESSION_RATIO
                )
            ):
                raise ArtifactValidationError(
                    f"{info.filename} má podezřelý kompresní poměr."
                )

        root.mkdir(parents=True, exist_ok=True)
        destination = root / manifest["artifact_id"]
        if destination.exists():
            existing = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            existing_manifest = validate_manifest(
                existing,
                artifact_dir=destination,
                verify_files=True,
                expected_hidden_size=expected_hidden_size,
                expected_vocab_size=expected_vocab_size,
            )
            if existing_manifest == manifest:
                return destination
            raise ArtifactValidationError(
                "Adresář stejného artifact_id už obsahuje jiný artefakt."
            )

        temporary_root = Path(
            tempfile.mkdtemp(prefix=".jlens-install-", dir=root)
        )
        temporary_artifact = temporary_root / manifest["artifact_id"]
        temporary_artifact.mkdir()
        try:
            expected_by_name = {
                entry["name"]: entry["sha256"]
                for entry in entries.values()
            }
            for info in infos:
                output = temporary_artifact / info.filename
                digest = hashlib.sha256()
                with archive.open(info, "r") as source, output.open("xb") as target:
                    while chunk := source.read(8 * 1024 * 1024):
                        digest.update(chunk)
                        target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                expected = expected_by_name.get(info.filename)
                if expected is not None and digest.hexdigest() != expected:
                    raise ArtifactValidationError(
                        f"SHA-256 souboru {info.filename} nesouhlasí."
                    )
            validate_manifest(
                manifest,
                artifact_dir=temporary_artifact,
                verify_files=True,
                expected_hidden_size=expected_hidden_size,
                expected_vocab_size=expected_vocab_size,
            )
            os.replace(temporary_artifact, destination)
            return destination
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "jacobian_lens",
    )
    parser.add_argument("--sha256")
    args = parser.parse_args()
    installed = install_runtime_zip(
        args.zip_path,
        args.root,
        expected_zip_sha256=args.sha256,
    )
    print(installed)


if __name__ == "__main__":
    main()
