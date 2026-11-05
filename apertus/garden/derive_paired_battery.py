#!/usr/bin/env python3
"""Regenerate paired-battery tables and report without new model inference."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Any

import run_paired_battery as battery


APP_DIR = Path(__file__).resolve().parent


def verify_raw_manifest(series_dir: Path) -> dict[str, Any]:
    manifest_path = series_dir / "manifest.json"
    manifest = battery.load_json(manifest_path)
    if (
        manifest.get("format")
        != f"{battery.FORMAT}.manifest"
        or manifest.get("status") != "complete"
    ):
        raise battery.BatteryError("Raw manifest není úplný nebo podporovaný.")
    for item in manifest.get("files", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise battery.BatteryError("Raw manifest obsahuje neplatnou položku.")
        path = series_dir / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or battery.sha256_file(path) != item.get("sha256")
        ):
            raise battery.BatteryError(
                f"Raw bundle neodpovídá manifestu: {item['path']}."
            )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Odvodí tabulky a report z hotové párové baterie."
    )
    parser.add_argument("--series-id", required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=battery.DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=battery.DEFAULT_RAW_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=battery.DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not battery.SERIES_RE.fullmatch(args.series_id):
        raise battery.BatteryError("Neplatné --series-id.")
    protocol_path = args.protocol.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    series_dir = raw_root / args.series_id

    protocol = battery.load_json(protocol_path)
    battery.validate_protocol(protocol)
    checkpoint = battery.load_json(battery.checkpoint_path(series_dir))
    if (
        checkpoint.get("format") != battery.FORMAT
        or checkpoint.get("series_id") != args.series_id
        or checkpoint.get("status") != "complete"
    ):
        raise battery.BatteryError("Série nemá úplný checkpoint.")
    protocol_sha = battery.sha256_file(protocol_path)
    if checkpoint.get("protocol_sha256") != protocol_sha:
        raise battery.BatteryError("Aktuální protokol neodpovídá raw sérii.")
    manifest = verify_raw_manifest(series_dir)

    selected_pairs = battery.choose_pairs(
        protocol,
        list(checkpoint["selected_pair_ids"]),
        None,
    )
    pair_results = []
    for pair in selected_pairs:
        entry = checkpoint["completed_pairs"].get(pair["pair_id"])
        if not isinstance(entry, dict):
            raise battery.BatteryError(
                f"Checkpoint postrádá pár {pair['pair_id']}."
            )
        path = series_dir / entry["file"]
        if battery.sha256_file(path) != entry.get("sha256"):
            raise battery.BatteryError(
                f"Hash páru {pair['pair_id']} nesedí."
            )
        pair_results.append(battery.load_json(path))

    previous_metadata_path = output_dir / "run_metadata.json"
    previous_metadata = (
        battery.load_json(previous_metadata_path)
        if previous_metadata_path.is_file()
        else {}
    )
    collection_runner_sha = (
        previous_metadata.get("collection_runner_sha256")
        or previous_metadata.get("runner_sha256")
    )
    analyzer_path = Path(__file__).resolve()
    analysis_library_path = Path(battery.__file__).resolve()
    metadata = {
        "format": f"{battery.FORMAT}.run-metadata",
        "schema_version": battery.SCHEMA_VERSION,
        "series_id": args.series_id,
        "started_at": checkpoint["started_at"],
        "completed_at": checkpoint["completed_at"],
        "derived_at": battery.utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "collection_runner_sha256": collection_runner_sha,
        "analysis_entrypoint_sha256": battery.sha256_file(analyzer_path),
        "analysis_library_sha256": battery.sha256_file(
            analysis_library_path
        ),
        "tokenizer_sha256": checkpoint["tokenizer_sha256"],
        "model_identity": checkpoint["model_identity"],
        "archive_initial": checkpoint["archive_initial"],
        "archive_final": checkpoint["archive_final"],
        "archive_unchanged": (
            checkpoint["archive_initial"] == checkpoint["archive_final"]
        ),
        "archive_mutation_endpoints_called": 0,
        "selected_pair_ids": checkpoint["selected_pair_ids"],
        "raw_bundle": str(series_dir),
        "raw_manifest_sha256": battery.sha256_file(
            series_dir / "manifest.json"
        ),
        "raw_manifest_file_count": len(manifest.get("files", [])),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    battery.write_derived_outputs(
        output_dir=output_dir,
        series_id=args.series_id,
        selected_pairs=selected_pairs,
        pair_results=pair_results,
        metadata=metadata,
    )
    print(f"Verified raw bundle: {series_dir}")
    print(f"Regenerated outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except battery.BatteryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
