#!/usr/bin/env python3
"""Build a revision-pinned Czech/English Wikipedia pilot corpus.

This is a convenience collector, not a claim that Wikipedia alone is a
genre-balanced research corpus.  It writes document-level JSONL suitable for
``train_tuned_lens.py`` and records the license returned by each wiki's own
siteinfo API.  Random selection is made once; page IDs, revisions and content
hashes make the resulting file immutable and auditable afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    from .calibration_policy import EXCLUDED_EXHIBIT_FRAGMENTS
except ImportError:
    from calibration_policy import EXCLUDED_EXHIBIT_FRAGMENTS


LANGUAGES = ("cs", "en")
SPACE = re.compile(r"\s+")
WORD = re.compile(r"\w+", flags=re.UNICODE)
USER_AGENT = (
    "ApertusProbabilityGarden/1.0 "
    "(local academic calibration corpus collector)"
)
HUGGINGFACE_DATASET = "wikimedia/wikipedia"
HUGGINGFACE_CONFIG_DATE = "20231101"


def normalize(text: str) -> str:
    return SPACE.sub(" ", text).strip()


def api_json(language: str, parameters: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "format": "json",
            "formatversion": 2,
            **parameters,
        }
    )
    request = urllib.request.Request(
        f"https://{language}.wikipedia.org/w/api.php?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.load(response)
    if isinstance(payload, dict) and payload.get("error"):
        error = payload["error"]
        raise RuntimeError(
            f"MediaWiki API: {error.get('code', 'error')}: "
            f"{error.get('info', 'neznámá chyba')}"
        )
    return payload


def rights_info(language: str) -> dict[str, str]:
    payload = api_json(
        language,
        {
            "action": "query",
            "meta": "siteinfo",
            "siprop": "rightsinfo",
        },
    )
    rights = payload["query"]["rightsinfo"]
    return {
        "text": str(rights.get("text") or "Wikimedia project license"),
        "url": str(rights.get("url") or ""),
    }


def random_batch(language: str, limit: int) -> list[dict[str, Any]]:
    payload = api_json(
        language,
        {
            "action": "query",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": min(50, limit),
            "prop": "extracts|revisions|pageprops",
            "explaintext": 1,
            "exsectionformat": "plain",
            "rvprop": "ids|timestamp",
            "rvlimit": 1,
        },
    )
    return list(payload.get("query", {}).get("pages", []))


def simhash(text: str) -> int:
    words = WORD.findall(text.casefold())
    shingles = (
        [" ".join(words[index : index + 5]) for index in range(len(words) - 4)]
        if len(words) >= 5
        else words
    )
    vector = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(
            hashlib.blake2b(
                shingle.encode("utf-8"),
                digest_size=8,
            ).digest(),
            "big",
        )
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            result |= 1 << bit
    return result


def too_near(candidate: int, accepted: list[int], threshold: int) -> bool:
    return any(
        (candidate ^ previous).bit_count() <= threshold
        for previous in accepted
    )


def collect(
    *,
    language: str,
    count: int,
    minimum_characters: int,
    maximum_characters: int,
    near_duplicate_hamming: int,
    pause_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rights = rights_info(language)
    accepted: list[dict[str, Any]] = []
    seen_page_ids: set[int] = set()
    seen_hashes: set[str] = set()
    accepted_simhashes: list[int] = []
    attempts = 0
    while len(accepted) < count:
        attempts += 1
        if attempts > max(50, count * 10):
            raise RuntimeError(
                f"{language}: po {attempts} dávkách mám jen "
                f"{len(accepted)} dokumentů."
            )
        try:
            pages = random_batch(language, count - len(accepted))
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            print(f"{language}: síťová chyba, opakuji: {exc}", flush=True)
            time.sleep(min(60.0, max(2.0, pause_seconds) * attempts))
            continue
        for page in pages:
            page_id = int(page.get("pageid") or 0)
            if not page_id or page_id in seen_page_ids:
                continue
            seen_page_ids.add(page_id)
            if "disambiguation" in (page.get("pageprops") or {}):
                continue
            text = normalize(str(page.get("extract") or ""))
            if len(text) < minimum_characters:
                continue
            text = text[:maximum_characters].rsplit(" ", 1)[0]
            folded = text.casefold()
            if any(fragment in folded for fragment in EXCLUDED_EXHIBIT_FRAGMENTS):
                continue
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_hash in seen_hashes:
                continue
            fingerprint = simhash(text)
            if too_near(
                fingerprint,
                accepted_simhashes,
                near_duplicate_hamming,
            ):
                continue
            revisions = page.get("revisions") or []
            if not revisions:
                continue
            revision = revisions[0]
            wikidata_id = str(
                (page.get("pageprops") or {}).get("wikibase_item") or ""
            ).strip()
            topic_id = (
                f"wikidata-{wikidata_id}"
                if wikidata_id
                else f"{language}-wiki-{page_id}"
            )
            seen_hashes.add(text_hash)
            accepted_simhashes.append(fingerprint)
            title = str(page.get("title") or f"page-{page_id}")
            accepted.append(
                {
                    "doc_id": f"{language}-wiki-{page_id}-{revision['revid']}",
                    "pair_id": topic_id,
                    "language": language,
                    "source": (
                        f"https://{language}.wikipedia.org/?curid={page_id}"
                    ),
                    "revision": str(revision["revid"]),
                    "revision_timestamp": revision.get("timestamp"),
                    "license": rights["text"],
                    "license_url": rights["url"],
                    "title": title,
                    "text": text,
                    "text_sha256": text_hash,
                }
            )
            if len(accepted) >= count:
                break
        print(
            f"{language}: {len(accepted):4d}/{count} dokumentů "
            f"({len(seen_page_ids)} náhodných stránek)",
            flush=True,
        )
        time.sleep(pause_seconds)
    return accepted, rights


def collect_huggingface(
    *,
    language: str,
    count: int,
    minimum_characters: int,
    maximum_characters: int,
    near_duplicate_hamming: int,
    seed: int,
    revision: str | None,
) -> tuple[list[dict[str, Any]], dict[str, str], str]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    pinned_revision = revision or HfApi().dataset_info(
        HUGGINGFACE_DATASET
    ).sha
    dataset = load_dataset(
        HUGGINGFACE_DATASET,
        f"{HUGGINGFACE_CONFIG_DATE}.{language}",
        split="train",
        streaming=True,
        revision=pinned_revision,
    ).shuffle(
        seed=seed + LANGUAGES.index(language),
        buffer_size=max(2_000, count * 20),
    )
    accepted: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    accepted_simhashes: list[int] = []
    inspected = 0
    for row in dataset:
        inspected += 1
        text = normalize(str(row.get("text") or ""))
        if len(text) < minimum_characters:
            continue
        text = text[:maximum_characters].rsplit(" ", 1)[0]
        folded = text.casefold()
        if any(fragment in folded for fragment in EXCLUDED_EXHIBIT_FRAGMENTS):
            continue
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in seen_hashes:
            continue
        fingerprint = simhash(text)
        if too_near(
            fingerprint,
            accepted_simhashes,
            near_duplicate_hamming,
        ):
            continue
        article_id = str(row.get("id") or "").strip()
        if not article_id:
            continue
        wikidata_id = str(
            row.get("wikidata_id")
            or row.get("wikibase_item")
            or ""
        ).strip()
        topic_id = (
            f"wikidata-{wikidata_id}"
            if wikidata_id
            else f"{language}-wiki-{article_id}"
        )
        seen_hashes.add(text_hash)
        accepted_simhashes.append(fingerprint)
        accepted.append(
            {
                "doc_id": (
                    f"{language}-wiki-{HUGGINGFACE_CONFIG_DATE}-"
                    f"{article_id}"
                ),
                "pair_id": topic_id,
                "language": language,
                "source": str(row.get("url") or ""),
                "revision": (
                    f"huggingface:{pinned_revision};"
                    f"dump:{HUGGINGFACE_CONFIG_DATE};article:{article_id}"
                ),
                "license": "CC BY-SA 3.0 and GFDL",
                "license_url": (
                    "https://huggingface.co/datasets/wikimedia/wikipedia"
                ),
                "title": str(row.get("title") or ""),
                "text": text,
                "text_sha256": text_hash,
            }
        )
        if len(accepted) % 20 == 0 or len(accepted) == count:
            print(
                f"{language}: {len(accepted):4d}/{count} dokumentů "
                f"({inspected} streamovaných článků)",
                flush=True,
            )
        if len(accepted) >= count:
            break
    if len(accepted) < count:
        raise RuntimeError(
            f"{language}: stream skončil po {len(accepted)} dokumentech."
        )
    rights = {
        "text": "CC BY-SA 3.0 and GFDL",
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia",
    }
    return accepted, rights, pinned_revision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--backend",
        choices=("huggingface", "mediawiki"),
        default="huggingface",
    )
    parser.add_argument(
        "--dataset-revision",
        help="Volitelný přesný commit wikimedia/wikipedia.",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--documents-per-language", type=int, default=120)
    parser.add_argument("--minimum-characters", type=int, default=800)
    parser.add_argument("--maximum-characters", type=int, default=2400)
    parser.add_argument("--near-duplicate-hamming", type=int, default=3)
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.documents_per_language < 20:
        raise SystemExit(
            "Pro dokumentové train/validation/test dělení zvol alespoň 20."
        )
    if not 100 <= args.minimum_characters < args.maximum_characters:
        raise SystemExit("Neplatné rozmezí délky dokumentu.")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Výstup už existuje a nebude přepsán: {output}")
    documents = []
    rights_by_language = {}
    dataset_revision = None
    for language in LANGUAGES:
        if args.backend == "huggingface":
            selected, rights, pinned_revision = collect_huggingface(
                language=language,
                count=args.documents_per_language,
                minimum_characters=args.minimum_characters,
                maximum_characters=args.maximum_characters,
                near_duplicate_hamming=args.near_duplicate_hamming,
                seed=args.seed,
                revision=args.dataset_revision,
            )
            if (
                dataset_revision is not None
                and dataset_revision != pinned_revision
            ):
                raise RuntimeError(
                    "HF dataset změnil commit mezi jazykovými streamy."
                )
            dataset_revision = pinned_revision
        else:
            selected, rights = collect(
                language=language,
                count=args.documents_per_language,
                minimum_characters=args.minimum_characters,
                maximum_characters=args.maximum_characters,
                near_duplicate_hamming=args.near_duplicate_hamming,
                pause_seconds=args.pause_seconds,
            )
        documents.extend(selected)
        rights_by_language[language] = rights
    documents.sort(key=lambda document: document["doc_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for document in documents:
            handle.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "format": "apertus.wikipedia-calibration-corpus",
        "schema_version": 1,
        "output_file": output.name,
        "output_sha256": digest,
        "documents": len(documents),
        "documents_by_language": {
            language: sum(
                document["language"] == language
                for document in documents
            )
            for language in LANGUAGES
        },
        "selection": {
            "method": (
                "Hugging Face streaming shuffle over wikimedia/wikipedia"
                if args.backend == "huggingface"
                else "MediaWiki generator=random, namespace 0"
            ),
            "backend": args.backend,
            "dataset": (
                HUGGINGFACE_DATASET
                if args.backend == "huggingface"
                else None
            ),
            "dataset_revision": dataset_revision,
            "seed": args.seed,
            "minimum_characters": args.minimum_characters,
            "maximum_characters": args.maximum_characters,
            "near_duplicate": (
                "64-bit word-5-gram SimHash, reject Hamming <= "
                f"{args.near_duplicate_hamming}"
            ),
            "excluded_exhibit_fragments": list(
                EXCLUDED_EXHIBIT_FRAGMENTS
            ),
        },
        "rights_info": rights_by_language,
        "limitations": [
            "One genre (Wikipedia); this is a calibration pilot, not a genre-balanced corpus.",
            "Random pages are frozen by page ID, revision ID and output SHA-256 after collection.",
            (
                "Cross-language pages share a split only when the selected "
                "backend exposes a common Wikidata identifier."
            ),
        ],
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"Hotovo: {output}\nSHA-256: {digest}", flush=True)


if __name__ == "__main__":
    main()
