#!/usr/bin/env python3
"""Collect the locked task-vs-form residual-state battery locally.

Validation uses only the Rust-backed ``tokenizers`` package.  Model loading and
all MLX imports are deliberately delayed until after ``--validate-only`` has
returned.  A full run executes one cache-free Apertus forward per stimulus and
stores the two preregistered residual positions for L0 through L32.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


APP_DIR = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = APP_DIR / "task_form_protocol.json"
try:
    from .settings import cache_directory, data_directory
except ImportError:
    from settings import cache_directory, data_directory
DEFAULT_BF16_MODEL = cache_directory() / "models/bf16"
DEFAULT_8BIT_MODEL = cache_directory() / "models/8bit"
DEFAULT_RAW_ROOT = data_directory() / "task_form_batteries"
DEFAULT_RESULTS_DIR = data_directory() / "results_task_vs_form"

FORMAT = "apertus.task-vs-form-battery"
SCHEMA_VERSION = 1
POSITION_ORDER = ("country_end", "response_boundary")
LAYER_ORDER = tuple(range(33))
STATE_SHAPE = (33, 2, 4096)
STATE_KEY = "states"
MAX_INPUT_TOKENS = 2_048
MODEL_MODES = ("hybrid", "bf16")
SERIES_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class BatteryError(RuntimeError):
    """A locked-design, tokenizer, model, or persistence invariant failed."""


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BatteryError(f"Nelze hashovat {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BatteryError(f"Objekt nelze zapsat jako kanonický JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BatteryError(f"Objekt nelze zapsat jako JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    """Best-effort durability for an already atomically replaced file."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    atomic_write(path, pretty_json_bytes(value) if pretty else canonical_json_bytes(value))


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatteryError(f"Nelze načíst JSON {path}: {exc}") from exc


def load_protocol(path: Path) -> tuple[dict[str, Any], bytes, str]:
    try:
        contents = path.read_bytes()
        protocol = json.loads(contents)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatteryError(f"Nelze načíst protokol {path}: {exc}") from exc
    if not isinstance(protocol, dict):
        raise BatteryError("Kořen protokolu musí být JSON objekt.")
    return protocol, contents, sha256_bytes(contents)


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BatteryError(f"{label} musí být neprázdný řetězec.")
    return value


def _require_template(value: Any, label: str, fields: Iterable[str]) -> str:
    template = _require_nonempty_string(value, label)
    if template != template.strip():
        raise BatteryError(f"{label} nesmí mít okrajové mezery.")
    for field in fields:
        if template.count("{" + field + "}") != 1:
            raise BatteryError(
                f"{label} musí obsahovat právě jeden zástupný znak {{{field}}}."
            )
    return template


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Validate the locked factorial design without tokenizing or loading MLX."""

    if (
        protocol.get("format") != "apertus.task-vs-form-protocol"
        or protocol.get("schema_version") != 1
    ):
        raise BatteryError("Nepodporovaný formát nebo verze task-vs-form protokolu.")
    protocol_id = protocol.get("protocol_id")
    if not isinstance(protocol_id, str) or not SERIES_RE.fullmatch(protocol_id):
        raise BatteryError("Protokol nemá platné protocol_id.")

    design = protocol.get("design")
    relations = protocol.get("relations")
    verification = protocol.get("verification")
    countries = protocol.get("countries")
    false_donors = protocol.get("false_donors")
    audit = protocol.get("audit")
    if not isinstance(design, dict):
        raise BatteryError("Protokol postrádá objekt design.")
    if not isinstance(relations, dict) or set(relations) != {"capital", "currency"}:
        raise BatteryError("Protokol musí obsahovat relace capital a currency.")
    if not isinstance(verification, dict):
        raise BatteryError("Protokol postrádá verifikační rodinu.")
    if not isinstance(countries, list) or not countries:
        raise BatteryError("Protokol neobsahuje země.")
    if not isinstance(audit, dict):
        raise BatteryError("Protokol postrádá auditní očekávání.")

    if design.get("allowed_model_modes") != ["hybrid", "bf16"]:
        raise BatteryError("Povolené modelové režimy musí být hybrid a bf16.")
    if design.get("required_primary_model_mode") != "hybrid":
        raise BatteryError("Primární modelový režim musí být hybrid.")
    if design.get("layers") != list(LAYER_ORDER):
        raise BatteryError("Protokol musí uzamykat vrstvy L0–L32.")
    if design.get("primary_layers") != list(range(9, 25)):
        raise BatteryError("Primární okno musí být předem uzamčeno na L9–L24.")
    if design.get("layer_bands") != {
        "early": list(range(1, 9)),
        "middle": list(range(9, 25)),
        "late": list(range(25, 33)),
    }:
        raise BatteryError("Časné, střední a pozdní pásmo není uzamčeno.")
    if design.get("measurement_positions") != list(POSITION_ORDER):
        raise BatteryError(
            "Měřené pozice musí být country_end a response_boundary v tomto pořadí."
        )
    if design.get("response_boundary_text") != "je":
        raise BatteryError("Společná hranice odpovědi musí být přesně „je“.")
    if design.get("state_hook") != "post_decoder_block_residual":
        raise BatteryError("Nepodporovaný residual-state hook.")
    if design.get("layer_numbering") != "L0_embedding_then_one_based_decoder_blocks":
        raise BatteryError("Nepodporované číslování vrstev.")
    if design.get("state_storage_dtype") != "float32":
        raise BatteryError("Raw stavy musí být uzamčeny jako float32.")
    baseline = design.get("baseline")
    if (
        not isinstance(baseline, dict)
        or baseline.get("distance") != "cosine"
        or baseline.get("pooling_keys")
        != ["family", "truth_value", "layer", "position"]
        or baseline.get("matched_pair_keys") != ["relation_id", "form_id"]
        or baseline.get("country_pairs")
        != "all unordered pairs of distinct locked countries"
        or baseline.get("invalid_if") != "zero or non-finite from L1 onward"
        or baseline.get("analysis_scope")
        != "L1-L32; L0 is persisted only as an embedding sanity check"
    ):
        raise BatteryError("Normalizační baseline není jednoznačně uzamčena.")
    if (
        type(design.get("bootstrap_seed")) is not int
        or type(design.get("bootstrap_samples")) is not int
        or design["bootstrap_samples"] < 1_000
    ):
        raise BatteryError("Bootstrap seed nebo počet replikací není platný.")

    assignment = design.get("false_statement_assignment")
    if (
        not isinstance(assignment, dict)
        or assignment.get("kind") != "external_locked_donor"
        or assignment.get("one_to_one") is not True
    ):
        raise BatteryError("Neplatné externí přiřazení nepravdivých výroků.")

    generative_form_ids: list[str] | None = None
    verification_form_ids: list[str] | None = None
    answer_fields: set[str] = set()
    for relation_id, relation in relations.items():
        if not isinstance(relation, dict):
            raise BatteryError(f"Relace {relation_id} musí být objekt.")
        _require_nonempty_string(relation.get("label"), f"{relation_id}.label")
        answer_field = _require_nonempty_string(
            relation.get("answer_field"), f"{relation_id}.answer_field"
        )
        if answer_field in answer_fields:
            raise BatteryError("Každá relace musí mít jiné answer_field.")
        answer_fields.add(answer_field)
        forms = relation.get("generative_forms")
        if not isinstance(forms, dict) or len(forms) != 3:
            raise BatteryError(f"{relation_id} musí mít právě tři generativní formy.")
        current_ids: list[str] = []
        for form_id, template in forms.items():
            if not isinstance(form_id, str) or not ID_RE.fullmatch(form_id):
                raise BatteryError(f"Neplatné ID generativní formy: {form_id!r}.")
            current_ids.append(form_id)
            _require_template(
                template,
                f"{relation_id}.{form_id}",
                ("country_genitive",),
            )
        if generative_form_ids is None:
            generative_form_ids = current_ids
        elif current_ids != generative_form_ids:
            raise BatteryError(
                "Obě relace musí používat stejné generativní formy ve stejném pořadí."
            )
        relation_verification_forms = relation.get("verification_forms")
        if (
            not isinstance(relation_verification_forms, dict)
            or len(relation_verification_forms) != 3
        ):
            raise BatteryError(
                f"{relation_id} musí mít právě tři verifikační parafráze."
            )
        current_verification_ids: list[str] = []
        for form_id, template in relation_verification_forms.items():
            if not isinstance(form_id, str) or not ID_RE.fullmatch(form_id):
                raise BatteryError(
                    f"Neplatné ID verifikační formy: {form_id!r}."
                )
            current_verification_ids.append(form_id)
            _require_template(
                template,
                f"{relation_id}.verification_forms.{form_id}",
                ("country_genitive", "answer"),
            )
        if verification_form_ids is None:
            verification_form_ids = current_verification_ids
        elif current_verification_ids != verification_form_ids:
            raise BatteryError(
                "Obě relace musí používat stejné verifikační formy ve stejném pořadí."
            )

    truth_values = verification.get("truth_values")
    if not isinstance(truth_values, dict) or set(truth_values) != {"true", "false"}:
        raise BatteryError("Verifikace musí obsahovat truth values true a false.")
    outputs: set[str] = set()
    for truth_value, settings in truth_values.items():
        if not isinstance(settings, dict):
            raise BatteryError(f"truth value {truth_value} musí být objekt.")
        output = _require_nonempty_string(
            settings.get("expected_output"),
            f"verification.{truth_value}.expected_output",
        )
        if output != output.strip():
            raise BatteryError("Verifikační výstup nesmí mít okrajové mezery.")
        outputs.add(output)
    if len(outputs) != 2:
        raise BatteryError("Pravdivý a nepravdivý výrok musí mít různé štítky.")
    _require_template(
        verification.get("wrapper"),
        "verification.wrapper",
        ("statement",),
    )
    if (
        not isinstance(verification.get("form_ids"), list)
        or verification["form_ids"] != verification_form_ids
    ):
        raise BatteryError(
            "verification.form_ids musí přesně odpovídat relačním parafrázím."
        )
    primary_index_forms = design.get("primary_index_forms")
    if (
        not isinstance(primary_index_forms, dict)
        or primary_index_forms.get("generative")
        != ["g1_predicate", "g2_question"]
        or primary_index_forms.get("verification") != verification_form_ids
        or any(
            form_id not in (generative_form_ids or [])
            for form_id in primary_index_forms.get("generative", [])
        )
    ):
        raise BatteryError("Formy primárního indexu nejsou správně uzamčeny.")

    expected_country_count = audit.get("expected_counts", {}).get("countries")
    if type(expected_country_count) is not int or len(countries) != expected_country_count:
        raise BatteryError("Počet zemí neodpovídá uzamčenému auditnímu počtu.")

    country_ids: set[str] = set()
    country_names: set[str] = set()
    genitives: set[str] = set()
    answer_values: dict[str, set[str]] = {field: set() for field in answer_fields}
    for number, country in enumerate(countries, start=1):
        if not isinstance(country, dict):
            raise BatteryError(f"Země #{number} musí být objekt.")
        country_id = country.get("country_id")
        if (
            not isinstance(country_id, str)
            or not ID_RE.fullmatch(country_id)
            or country_id in country_ids
        ):
            raise BatteryError(f"Neplatné nebo duplicitní country_id: {country_id!r}.")
        country_ids.add(country_id)
        country_name = _require_nonempty_string(
            country.get("country"), f"{country_id}.country"
        )
        genitive = _require_nonempty_string(
            country.get("country_genitive"), f"{country_id}.country_genitive"
        )
        if country_name.casefold() in country_names:
            raise BatteryError(f"Duplicitní název země: {country_name}.")
        if genitive.casefold() in genitives:
            raise BatteryError(f"Duplicitní genitiv země: {genitive}.")
        country_names.add(country_name.casefold())
        genitives.add(genitive.casefold())
        _require_nonempty_string(country.get("region"), f"{country_id}.region")
        for field in answer_fields:
            answer = _require_nonempty_string(country.get(field), f"{country_id}.{field}")
            folded = answer.casefold()
            if folded in answer_values[field]:
                raise BatteryError(f"Duplicitní hodnota {field}: {answer}.")
            answer_values[field].add(folded)

    numeric_ids = design.get("numeric_audit_country_ids")
    if (
        not isinstance(numeric_ids, list)
        or not numeric_ids
        or len(numeric_ids) != len(set(numeric_ids))
        or any(country_id not in country_ids for country_id in numeric_ids)
    ):
        raise BatteryError("numeric_audit_country_ids nejsou platnou podmnožinou zemí.")

    if not isinstance(false_donors, list) or len(false_donors) != len(countries):
        raise BatteryError(
            "Protokol musí mít právě jednoho externího false donora pro každou zemi."
        )
    ordered_target_ids: list[str] = []
    donor_ids: set[str] = set()
    donor_answers: dict[str, set[str]] = {
        field: set() for field in answer_fields
    }
    countries_by_id = {
        country["country_id"]: country for country in countries
    }
    for donor in false_donors:
        if not isinstance(donor, dict):
            raise BatteryError("False donor musí být objekt.")
        target_country_id = _require_nonempty_string(
            donor.get("target_country_id"),
            "false_donor.target_country_id",
        )
        donor_id = _require_nonempty_string(
            donor.get("donor_id"),
            f"false_donor.{target_country_id}.donor_id",
        )
        if (
            target_country_id not in countries_by_id
            or donor_id in donor_ids
            or donor_id in country_ids
            or not ID_RE.fullmatch(donor_id)
        ):
            raise BatteryError(
                f"Neplatný nebo duplicitní false donor {donor_id!r}."
            )
        ordered_target_ids.append(target_country_id)
        donor_ids.add(donor_id)
        for field in answer_fields:
            answer = _require_nonempty_string(
                donor.get(field),
                f"false_donor.{donor_id}.{field}",
            )
            folded = answer.casefold()
            if (
                folded in donor_answers[field]
                or folded in answer_values[field]
            ):
                raise BatteryError(
                    f"False donor {donor_id} má duplicitní nebo pravdivou hodnotu {field}."
                )
            donor_answers[field].add(folded)
    if ordered_target_ids != [country["country_id"] for country in countries]:
        raise BatteryError(
            "False donoři musí jednou a v uzamčeném pořadí pokrýt všechny země."
        )

    expected_counts = audit.get("expected_counts")
    for key in ("generative_stimuli", "verification_stimuli", "total_stimuli"):
        if not isinstance(expected_counts, dict) or type(expected_counts.get(key)) is not int:
            raise BatteryError(f"Auditní počet {key} musí být celé číslo.")


def _format_template(template: str, fields: dict[str, Any], label: str) -> str:
    try:
        rendered = template.format_map(fields)
    except (KeyError, ValueError) as exc:
        raise BatteryError(f"Nelze rozvinout šablonu {label}: {exc}") from exc
    if not rendered or rendered != rendered.rstrip():
        raise BatteryError(f"Rozvinutý prefix {label} je prázdný nebo končí mezerou.")
    return rendered


def expand_protocol(
    protocol: dict[str, Any],
    country_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Expand the locked protocol to its 180 explicit, deterministic stimuli."""

    validate_protocol(protocol)
    countries = protocol["countries"]
    relations = protocol["relations"]
    verification = protocol["verification"]
    false_donors = {
        donor["target_country_id"]: donor
        for donor in protocol["false_donors"]
    }
    boundary = protocol["design"]["response_boundary_text"]
    stimuli: list[dict[str, Any]] = []

    for country in countries:
        country_id = country["country_id"]
        false_donor = false_donors[country_id]

        for relation_id, relation in relations.items():
            answer_field = relation["answer_field"]
            expected_answer = country[answer_field]
            for form_id, template in relation["generative_forms"].items():
                prompt = _format_template(
                    template,
                    country,
                    f"{country_id}/{relation_id}/{form_id}",
                )
                stimulus_id = f"g__{country_id}__{relation_id}__{form_id}"
                stimuli.append(
                    {
                        "stimulus_id": stimulus_id,
                        "family": "generative",
                        "country_id": country_id,
                        "country": country["country"],
                        "country_genitive": country["country_genitive"],
                        "region": country["region"],
                        "relation_id": relation_id,
                        "relation_label": relation["label"],
                        "form_id": form_id,
                        "truth_value": None,
                        "statement_answer_country_id": country_id,
                        "statement_answer": expected_answer,
                        "expected_output": expected_answer,
                        "target_continuation": f" {expected_answer}",
                        "prompt": prompt,
                    }
                )

        for relation_id, relation in relations.items():
            answer_field = relation["answer_field"]
            for truth_value, truth_settings in verification["truth_values"].items():
                statement_source = (
                    country if truth_value == "true" else false_donor
                )
                statement_answer = statement_source[answer_field]
                expected_output = truth_settings["expected_output"]
                for form_id, statement_template in relation[
                    "verification_forms"
                ].items():
                    statement = _format_template(
                        statement_template,
                        {
                            "country_genitive": country["country_genitive"],
                            "answer": statement_answer,
                        },
                        f"{country_id}/{relation_id}/{truth_value}/{form_id}/statement",
                    )
                    prompt = _format_template(
                        verification["wrapper"],
                        {"statement": statement},
                        f"{country_id}/{relation_id}/{truth_value}/{form_id}",
                    )
                    stimulus_id = (
                        f"v__{country_id}__{relation_id}__{truth_value}__{form_id}"
                    )
                    stimuli.append(
                        {
                            "stimulus_id": stimulus_id,
                            "family": "verification",
                            "country_id": country_id,
                            "country": country["country"],
                            "country_genitive": country["country_genitive"],
                            "region": country["region"],
                            "relation_id": relation_id,
                            "relation_label": relation["label"],
                            "form_id": form_id,
                            "truth_value": truth_value,
                            "statement_answer_country_id": (
                                country_id
                                if truth_value == "true"
                                else false_donor["donor_id"]
                            ),
                            "statement_answer": statement_answer,
                            "statement": statement,
                            "expected_output": expected_output,
                            "target_continuation": f" {expected_output}",
                            "prompt": prompt,
                        }
                    )

    stimulus_ids = [item["stimulus_id"] for item in stimuli]
    prompts = [item["prompt"] for item in stimuli]
    if len(stimulus_ids) != len(set(stimulus_ids)):
        raise BatteryError("Rozvinutý protokol obsahuje duplicitní stimulus_id.")
    if len(prompts) != len(set(prompts)):
        raise BatteryError("Rozvinutý protokol obsahuje duplicitní prefixy.")
    for stimulus in stimuli:
        prompt = stimulus["prompt"]
        if not prompt.endswith(f" {boundary}"):
            raise BatteryError(
                f"{stimulus['stimulus_id']}: prefix nekončí společnou hranicí „{boundary}“."
            )
        country_text = stimulus["country_genitive"]
        starts = [
            index
            for index in range(len(prompt))
            if prompt.startswith(country_text, index)
        ]
        if len(starts) != 1:
            raise BatteryError(
                f"{stimulus['stimulus_id']}: genitiv země nemá právě jeden znakový výskyt."
            )

    expected = protocol["audit"]["expected_counts"]
    generative_count = sum(item["family"] == "generative" for item in stimuli)
    verification_count = sum(item["family"] == "verification" for item in stimuli)
    actual = {
        "generative_stimuli": generative_count,
        "verification_stimuli": verification_count,
        "total_stimuli": len(stimuli),
    }
    for key, count in actual.items():
        if count != expected[key]:
            raise BatteryError(
                f"Rozvinutý počet {key}={count}, protokol uzamyká {expected[key]}."
            )
    if country_ids is None:
        return stimuli
    selected_countries = choose_countries(protocol, country_ids, None)
    return select_stimuli(
        stimuli,
        [country["country_id"] for country in selected_countries],
    )


# A descriptive alias retained for test and analysis callers.
expand_stimuli = expand_protocol


def load_tokenizer(path: Path) -> Any:
    """Load only tokenizer.json through the Rust-backed tokenizers library."""

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise BatteryError(
            "Chybí balík tokenizers; validační režim nesmí používat MLX tokenizer."
        ) from exc
    try:
        return Tokenizer.from_file(str(path))
    except Exception as exc:
        raise BatteryError(f"Tokenizer nelze načíst z {path}: {exc}") from exc


def _encoding_data(encoding: Any, label: str) -> tuple[list[int], list[str], list[list[int]]]:
    ids = list(encoding.ids)
    tokens = list(encoding.tokens)
    offsets = [[int(start), int(end)] for start, end in encoding.offsets]
    if not ids or len(ids) != len(tokens) or len(ids) != len(offsets):
        raise BatteryError(f"{label}: tokenizer vrátil neúplnou tokenizaci.")
    return ids, tokens, offsets


def _decode_exact(tokenizer: Any, ids: list[int], expected: str, label: str) -> None:
    try:
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
    except Exception as exc:
        raise BatteryError(f"{label}: tokenizer nedokáže dekódovat: {exc}") from exc
    if decoded != expected:
        raise BatteryError(
            f"{label}: tokenizace není bezeztrátová ({decoded!r} != {expected!r})."
        )


def _unique_text_span(text: str, needle: str, label: str) -> tuple[int, int]:
    starts = [index for index in range(len(text)) if text.startswith(needle, index)]
    if len(starts) != 1:
        raise BatteryError(f"{label}: text {needle!r} nemá právě jeden výskyt.")
    start = starts[0]
    return start, start + len(needle)


def _country_anchor(
    *,
    prompt: str,
    country_text: str,
    token_ids: list[int],
    tokens: list[str],
    offsets: list[list[int]],
    label: str,
) -> dict[str, Any]:
    char_start, char_end = _unique_text_span(prompt, country_text, label)
    overlapping = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end
    ]
    if not overlapping:
        raise BatteryError(f"{label}: genitiv země nemá tokenový span.")
    if overlapping != list(range(overlapping[0], overlapping[-1] + 1)):
        raise BatteryError(f"{label}: tokenový span země není souvislý.")
    token_start = overlapping[0]
    token_end = overlapping[-1] + 1
    union_start = offsets[token_start][0]
    union_end = offsets[token_end - 1][1]
    if union_end != char_end:
        raise BatteryError(f"{label}: poslední token země nekončí s jejím textem.")
    if union_start > char_start or not prompt[union_start:char_start].isspace():
        raise BatteryError(f"{label}: první token země překračuje jiný než whitespace.")
    ending = [
        index
        for index, (start, end) in enumerate(offsets)
        if start < char_end and end == char_end
    ]
    if ending != [token_end - 1]:
        raise BatteryError(f"{label}: country_end není jednoznačný tokenový offset.")
    return {
        "name": "country_end",
        "char_span": [char_start, char_end],
        "token_span": [token_start, token_end],
        "token_index": token_end - 1,
        "token_id": token_ids[token_end - 1],
        "token": tokens[token_end - 1],
        "offset": offsets[token_end - 1],
        "text": country_text,
    }


def _response_anchor(
    *,
    prompt: str,
    boundary: str,
    token_ids: list[int],
    tokens: list[str],
    offsets: list[list[int]],
    label: str,
) -> dict[str, Any]:
    if not prompt.endswith(f" {boundary}"):
        raise BatteryError(f"{label}: prefix nekončí přesně hranicí „ {boundary}“.")
    char_start = len(prompt) - len(boundary)
    char_end = len(prompt)
    candidates = [
        index
        for index, (start, end) in enumerate(offsets)
        if start < char_end and end == char_end
    ]
    if len(candidates) != 1:
        raise BatteryError(f"{label}: response_boundary není jednoznačný token.")
    token_index = candidates[0]
    if token_index != len(token_ids) - 1:
        raise BatteryError(f"{label}: response_boundary není poslední token prefixu.")
    start, end = offsets[token_index]
    if start > char_start or end != char_end:
        raise BatteryError(f"{label}: token hranice nepokrývá celý text „{boundary}“.")
    if prompt[start:end].strip() != boundary:
        raise BatteryError(
            f"{label}: poslední token není samostatná hranice „{boundary}“."
        )
    if start < char_start and not prompt[start:char_start].isspace():
        raise BatteryError(f"{label}: token hranice pohltil jiný než whitespace.")
    return {
        "name": "response_boundary",
        "char_span": [char_start, char_end],
        "token_span": [token_index, token_index + 1],
        "token_index": token_index,
        "token_id": token_ids[token_index],
        "token": tokens[token_index],
        "offset": offsets[token_index],
        "text": boundary,
    }


def tokenize_stimulus(
    tokenizer: Any,
    stimulus: dict[str, Any],
    *,
    response_boundary_text: str = "je",
) -> dict[str, Any]:
    """Tokenize one prefix, both anchors, and its contextual target suffix."""

    stimulus_id = stimulus["stimulus_id"]
    prompt = stimulus["prompt"]
    continuation = stimulus["target_continuation"]
    try:
        root_encoding = tokenizer.encode(prompt, add_special_tokens=True)
        full_encoding = tokenizer.encode(
            prompt + continuation,
            add_special_tokens=True,
        )
    except Exception as exc:
        raise BatteryError(f"{stimulus_id}: tokenizace selhala: {exc}") from exc
    root_ids, root_tokens, root_offsets = _encoding_data(
        root_encoding, f"{stimulus_id}/root"
    )
    full_ids, full_tokens, full_offsets = _encoding_data(
        full_encoding, f"{stimulus_id}/target"
    )
    if len(root_ids) > MAX_INPUT_TOKENS or len(full_ids) > MAX_INPUT_TOKENS:
        raise BatteryError(
            f"{stimulus_id}: tokenizace překročila limit {MAX_INPUT_TOKENS} tokenů."
        )
    _decode_exact(tokenizer, root_ids, prompt, f"{stimulus_id}/root")
    _decode_exact(
        tokenizer,
        full_ids,
        prompt + continuation,
        f"{stimulus_id}/target",
    )
    if full_ids[: len(root_ids)] != root_ids:
        raise BatteryError(
            f"{stimulus_id}: prefix není tokenovým prefixem cílového pokračování."
        )
    target_ids = full_ids[len(root_ids) :]
    target_tokens = full_tokens[len(root_ids) :]
    target_offsets = full_offsets[len(root_ids) :]
    if not target_ids:
        raise BatteryError(f"{stimulus_id}: cílové pokračování nemá žádný token.")

    country = _country_anchor(
        prompt=prompt,
        country_text=stimulus["country_genitive"],
        token_ids=root_ids,
        tokens=root_tokens,
        offsets=root_offsets,
        label=stimulus_id,
    )
    response = _response_anchor(
        prompt=prompt,
        boundary=response_boundary_text,
        token_ids=root_ids,
        tokens=root_tokens,
        offsets=root_offsets,
        label=stimulus_id,
    )
    if country["token_index"] >= response["token_index"]:
        raise BatteryError(f"{stimulus_id}: country_end musí předcházet hranici odpovědi.")

    return {
        "stimulus_id": stimulus_id,
        "prompt": prompt,
        "prompt_token_ids": root_ids,
        "prompt_tokens": root_tokens,
        "prompt_offsets": root_offsets,
        "prompt_token_count": len(root_ids),
        "anchors": {
            "country_end": country,
            "response_boundary": response,
        },
        "anchor_token_indices": [
            country["token_index"],
            response["token_index"],
        ],
        "target": {
            "expected_output": stimulus["expected_output"],
            "continuation": continuation,
            "token_ids": target_ids,
            "tokens": target_tokens,
            "offsets": target_offsets,
            "first_token_id": target_ids[0],
            "first_token": target_tokens[0],
        },
    }


def tokenize_stimuli(
    tokenizer: Any,
    stimuli: list[dict[str, Any]],
    *,
    response_boundary_text: str = "je",
) -> dict[str, dict[str, Any]]:
    tokenizations = {
        stimulus["stimulus_id"]: tokenize_stimulus(
            tokenizer,
            stimulus,
            response_boundary_text=response_boundary_text,
        )
        for stimulus in stimuli
    }
    boundary_ids = {
        item["anchors"]["response_boundary"]["token_id"]
        for item in tokenizations.values()
    }
    boundary_tokens = {
        item["anchors"]["response_boundary"]["token"]
        for item in tokenizations.values()
    }
    if len(boundary_ids) != 1 or len(boundary_tokens) != 1:
        raise BatteryError(
            "Všech 180 prefixů musí končit týmž tokenem response_boundary."
        )
    target_paths: dict[str, list[int]] = {}
    for item in tokenizations.values():
        expected_output = item["target"]["expected_output"]
        target_ids = item["target"]["token_ids"]
        previous = target_paths.setdefault(expected_output, target_ids)
        if previous != target_ids:
            raise BatteryError(
                f"Cílový výstup {expected_output!r} nemá ve všech kontextech "
                "stejnou tokenovou cestu."
            )
    return tokenizations


def choose_countries(
    protocol: dict[str, Any],
    country_ids: list[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    countries = protocol["countries"]
    by_id = {country["country_id"]: country for country in countries}
    if len(country_ids) != len(set(country_ids)):
        raise BatteryError("--country-id nesmí obsahovat duplicity.")
    unknown = [country_id for country_id in country_ids if country_id not in by_id]
    if unknown:
        raise BatteryError(f"Neznámé country_id: {', '.join(unknown)}.")
    if country_ids:
        wanted = set(country_ids)
        selected = [
            country for country in countries if country["country_id"] in wanted
        ]
    else:
        selected = list(countries)
    if limit is not None:
        if limit < 1:
            raise BatteryError("--limit musí být kladný.")
        selected = selected[:limit]
    if not selected:
        raise BatteryError("Výběr neobsahuje žádnou zemi.")
    return selected


def select_stimuli(
    stimuli: list[dict[str, Any]], selected_country_ids: list[str]
) -> list[dict[str, Any]]:
    wanted = set(selected_country_ids)
    selected = [item for item in stimuli if item["country_id"] in wanted]
    expected = 18 * len(selected_country_ids)
    if len(selected) != expected:
        raise BatteryError(
            f"Vyvážený výběr má {len(selected)} stimulů, očekává se {expected}."
        )
    return selected


def tokenizer_audit_payload(
    *,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    protocol_sha256: str,
    stimuli: list[dict[str, Any]],
    tokenizations: dict[str, dict[str, Any]],
    selected_stimulus_ids: list[str],
) -> dict[str, Any]:
    boundary = next(iter(tokenizations.values()))["anchors"]["response_boundary"]
    family_counts = {
        family: sum(item["family"] == family for item in stimuli)
        for family in ("generative", "verification")
    }
    return {
        "format": f"{FORMAT}.tokenization",
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol_sha256,
        "tokenizer": {
            "file": tokenizer_path.name,
            "model_directory": tokenizer_path.parent.name,
            "sha256": tokenizer_sha256,
            "implementation": "tokenizers.Tokenizer",
        },
        "stimulus_count": len(stimuli),
        "family_counts": family_counts,
        "selected_stimulus_ids": selected_stimulus_ids,
        "position_order": list(POSITION_ORDER),
        "common_response_boundary": {
            "text": boundary["text"],
            "token_id": boundary["token_id"],
            "token": boundary["token"],
        },
        "stimuli": tokenizations,
    }


def _load_server_module() -> Any:
    """Import server helpers only after validation-only has exited."""

    app_dir_text = str(APP_DIR)
    if app_dir_text not in sys.path:
        sys.path.insert(0, app_dir_text)
    try:
        return importlib.import_module("server")
    except Exception as exc:
        raise BatteryError(f"Nelze načíst ModelState ze server.py: {exc}") from exc


def load_model_state(
    *,
    model_mode: str,
    bf16_model_path: Path,
    eight_bit_model_path: Path,
) -> tuple[Any, Any]:
    server = _load_server_module()
    model_paths = {
        "bf16": bf16_model_path,
        "hybrid": eight_bit_model_path,
        "8bit": eight_bit_model_path,
    }
    model_path = model_paths[model_mode]
    state = server.ModelState(
        model_path=model_path,
        model_paths=model_paths,
        active_model_id=model_mode,
        tuned_lens_root=server.DEFAULT_TUNED_LENS_ROOT,
        mock=False,
    )
    if not state._model_available_unlocked(model_mode, model_path):
        raise BatteryError(f"Modelový režim {model_mode} není kompletní: {model_path}")
    print(f"Načítám Apertus ({model_mode})…", flush=True)
    state.load()
    if state.status != "ready" or state.model is None or state.tokenizer is None:
        raise BatteryError(f"Model se nenačetl: {state.error or state.status}")
    return state, server


def model_identity(state: Any) -> dict[str, Any]:
    try:
        provenance = state._model_provenance_unlocked()
        content_fingerprint = state._model_content_fingerprint_local()
    except Exception as exc:
        raise BatteryError(f"Nelze vytvořit modelový fingerprint: {exc}") from exc
    return {
        "model_id": state.active_model_id,
        "metadata_fingerprint": provenance["metadata_fingerprint"],
        "content_fingerprint": content_fingerprint,
        "fingerprint_scope": "all_runtime_model_bytes",
        "sources": provenance["sources"],
    }


def verify_model_identity_unchanged(state: Any, expected: dict[str, Any]) -> None:
    """Recheck disk metadata and cached per-file stats before sealing raw data."""

    state.model_provenance_cache.clear()
    state.model_content_fingerprint_cache.clear()
    observed = model_identity(state)
    if observed != expected:
        raise BatteryError("Modelový fingerprint se během série změnil.")


def runtime_token_ids(server: Any, tokenizer: Any, text: str) -> list[int]:
    try:
        return server.encode_with_bos(tokenizer, text)
    except Exception as exc:
        raise BatteryError(f"Runtime tokenizer nedokáže zakódovat prefix: {exc}") from exc


def capture_states_and_logits(
    *,
    state: Any,
    token_ids: list[int],
    anchor_token_indices: list[int],
) -> tuple[Any, Any, dict[str, float]]:
    """Run one cache-free Apertus forward and return float32 states and logits."""

    try:
        import mlx.core as mx
        import numpy as np
        from mlx_lm.models.base import create_attention_mask
    except ImportError as exc:
        raise BatteryError(f"Plný běh vyžaduje MLX, MLX-LM a NumPy: {exc}") from exc

    model = state.model
    if (
        getattr(model, "model_type", None) != "apertus"
        or not hasattr(model, "model")
    ):
        raise BatteryError("Residual runner podporuje pouze model Apertus.")
    body = model.model
    layers = getattr(body, "layers", None)
    if layers is None or len(layers) != 32:
        raise BatteryError("Apertus musí mít právě 32 decoderových vrstev.")
    hidden_size = int(getattr(model.args, "hidden_size", 0))
    if hidden_size != STATE_SHAPE[2]:
        raise BatteryError(
            f"Model má hidden_size={hidden_size}, očekává se {STATE_SHAPE[2]}."
        )
    if (
        len(anchor_token_indices) != 2
        or anchor_token_indices != sorted(anchor_token_indices)
        or anchor_token_indices[0] < 0
        or anchor_token_indices[-1] >= len(token_ids)
    ):
        raise BatteryError("Neplatné tokenové indexy měřených pozic.")

    started = time.perf_counter()
    ids = mx.array([token_ids])
    hidden = body.embed_tokens(ids)
    causal_mask = create_attention_mask(hidden, None)
    selected_states = [hidden[0, anchor_token_indices, :]]
    for layer in layers:
        hidden = layer(hidden, mask=causal_mask, cache=None)
        selected_states.append(hidden[0, anchor_token_indices, :])
    stacked_native = mx.stack(selected_states, axis=0)
    stacked = stacked_native.astype(mx.float32)

    # Project the model-native L32 state, matching the ordinary Apertus
    # forward.  The separately persisted copy is converted to locked float32.
    final_response_state = stacked_native[-1, 1, :]
    normalized = body.norm(final_response_state)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        projected = lm_head(normalized)
    else:
        as_linear = getattr(body.embed_tokens, "as_linear", None)
        if not callable(as_linear):
            raise BatteryError("Model nemá výstupní projekci.")
        projected = as_linear(normalized)
    logits = projected.astype(mx.float32)
    mx.eval(stacked, logits)
    elapsed_ms = (time.perf_counter() - started) * 1_000

    states_array = np.ascontiguousarray(np.asarray(stacked), dtype=np.float32)
    logits_array = np.ascontiguousarray(np.asarray(logits), dtype=np.float32)
    if tuple(states_array.shape) != STATE_SHAPE:
        raise BatteryError(
            f"Residual states mají tvar {states_array.shape}, očekává se {STATE_SHAPE}."
        )
    if logits_array.ndim != 1 or logits_array.size < 2:
        raise BatteryError("Finální projekce nevrátila vektor přes slovník.")
    if not np.all(np.isfinite(states_array)) or not np.all(np.isfinite(logits_array)):
        raise BatteryError("Model vrátil NaN nebo nekonečnou hodnotu.")
    norms = np.linalg.norm(states_array.astype(np.float64), axis=2)
    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
        raise BatteryError("Model vrátil nulový nebo neplatný residual state.")
    return states_array, logits_array, {"forward_and_projection_ms": elapsed_ms}


def summarize_final_distribution(
    *,
    logits: Any,
    expected_first_token_id: int,
    tokenizer: Any,
    top_k: int = 8,
) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size < top_k or not np.all(np.isfinite(values)):
        raise BatteryError("Nelze shrnout neplatný vektor logitů.")
    if not 0 <= expected_first_token_id < values.size:
        raise BatteryError("Cílový první token leží mimo slovník modelu.")

    count = min(top_k, values.size)
    cutoff_index = values.size - count
    cutoff = float(np.partition(values, cutoff_index)[cutoff_index])
    higher_ids = np.flatnonzero(values > cutoff)
    tied_ids = np.flatnonzero(values == cutoff)
    candidate_ids = np.concatenate(
        (higher_ids, tied_ids[: count - higher_ids.size])
    )
    ordered_ids = sorted(
        (int(token_id) for token_id in candidate_ids),
        key=lambda token_id: (-float(values[token_id]), token_id),
    )
    maximum = float(np.max(values))
    denominator = float(np.sum(np.exp(values - maximum)))
    log_denominator = maximum + math.log(denominator)

    def probability(token_id: int) -> float:
        return math.exp(float(values[token_id]) - log_denominator)

    def token_payload(token_id: int, rank: int) -> dict[str, Any]:
        raw_token = tokenizer.id_to_token(token_id)
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            decoded = None
        return {
            "rank": rank,
            "token_id": token_id,
            "token": raw_token,
            "decoded": decoded,
            "probability": probability(token_id),
            "logit": float(values[token_id]),
        }

    top = [
        token_payload(token_id, rank)
        for rank, token_id in enumerate(ordered_ids, start=1)
    ]
    expected_logit = float(values[expected_first_token_id])
    expected_rank = 1 + int(np.count_nonzero(values > expected_logit))
    expected = token_payload(expected_first_token_id, expected_rank)
    expected["in_top8"] = expected_first_token_id in ordered_ids
    return {
        "source": "L32_response_boundary_after_final_rms_norm_and_lm_head",
        "temperature": 1.0,
        "rank_definition": "1 + number of vocabulary logits strictly greater",
        "vocab_size": int(values.size),
        "top8": top,
        "expected_first_token": expected,
    }


def atomic_write_safetensors(path: Path, states: Any) -> None:
    try:
        import numpy as np
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise BatteryError(f"Uložení stavů vyžaduje safetensors a NumPy: {exc}") from exc
    array = np.ascontiguousarray(states, dtype=np.float32)
    if tuple(array.shape) != STATE_SHAPE:
        raise BatteryError(f"Nelze uložit states s tvarem {array.shape}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        save_file(
            {STATE_KEY: array},
            str(temporary),
            metadata={
                "format": f"{FORMAT}.states",
                "schema_version": str(SCHEMA_VERSION),
                "axes": "layer,position,hidden",
                "layers": "L0-L32",
                "positions": ",".join(POSITION_ORDER),
                "dtype": "float32",
            },
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def verify_state_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise BatteryError(f"Chybí nebo je nepovolený state soubor: {path}.")
    if sha256_file(path) != expected_sha256:
        raise BatteryError(f"Nesouhlasí hash state souboru {path.name}.")
    try:
        import numpy as np
        from safetensors.numpy import load_file

        arrays = load_file(str(path))
    except Exception as exc:
        raise BatteryError(f"Nelze načíst state soubor {path}: {exc}") from exc
    if set(arrays) != {STATE_KEY}:
        raise BatteryError(f"{path.name}: safetensors musí obsahovat jen klíč states.")
    states = arrays[STATE_KEY]
    if tuple(states.shape) != STATE_SHAPE or states.dtype != np.float32:
        raise BatteryError(
            f"{path.name}: očekává se float32 {STATE_SHAPE}, nalezeno "
            f"{states.dtype} {states.shape}."
        )
    if not np.all(np.isfinite(states)):
        raise BatteryError(f"{path.name}: states obsahují neplatné hodnoty.")


def _safe_relative_file(series_dir: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise BatteryError(f"{label}: chybí relativní cesta.")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise BatteryError(f"{label}: nepovolená cesta {relative!r}.")
    path = series_dir.joinpath(*candidate.parts)
    try:
        path.resolve().relative_to(series_dir.resolve())
    except (OSError, ValueError) as exc:
        raise BatteryError(f"{label}: cesta opouští raw bundle.") from exc
    return path


def verify_completed_entry(
    *,
    series_dir: Path,
    stimulus_id: str,
    entry: dict[str, Any],
    checkpoint: dict[str, Any],
) -> None:
    if not isinstance(entry, dict):
        raise BatteryError(f"Checkpoint entry {stimulus_id} není objekt.")
    metadata_path = _safe_relative_file(
        series_dir, entry.get("metadata_file"), f"{stimulus_id}/metadata"
    )
    state_path = _safe_relative_file(
        series_dir, entry.get("state_file"), f"{stimulus_id}/state"
    )
    if (
        not metadata_path.is_file()
        or metadata_path.is_symlink()
        or sha256_file(metadata_path) != entry.get("metadata_sha256")
    ):
        raise BatteryError(f"{stimulus_id}: metadata chybí nebo mají jiný hash.")
    verify_state_file(state_path, entry.get("state_sha256"))
    metadata = load_json(metadata_path)
    if (
        metadata.get("format") != f"{FORMAT}.stimulus"
        or metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("series_id") != checkpoint.get("series_id")
        or metadata.get("stimulus", {}).get("stimulus_id") != stimulus_id
        or metadata.get("protocol_sha256") != checkpoint.get("protocol_sha256")
        or metadata.get("tokenizer_sha256") != checkpoint.get("tokenizer_sha256")
        or metadata.get("runner_sha256") != checkpoint.get("runner_sha256")
        or metadata.get("model_identity") != checkpoint.get("model_identity")
        or metadata.get("model_metadata_fingerprint")
        != checkpoint.get("model_metadata_fingerprint")
        or metadata.get("model_content_fingerprint")
        != checkpoint.get("model_content_fingerprint")
    ):
        raise BatteryError(f"{stimulus_id}: metadata neodpovídají checkpointu.")
    state_meta = metadata.get("state")
    if (
        not isinstance(state_meta, dict)
        or state_meta.get("file") != entry.get("state_file")
        or state_meta.get("sha256") != entry.get("state_sha256")
        or state_meta.get("key") != STATE_KEY
        or state_meta.get("shape") != list(STATE_SHAPE)
        or state_meta.get("dtype") != "float32"
        or state_meta.get("layers") != list(LAYER_ORDER)
        or state_meta.get("positions") != list(POSITION_ORDER)
    ):
        raise BatteryError(f"{stimulus_id}: neplatný popis state tensoru.")


def checkpoint_path(series_dir: Path) -> Path:
    return series_dir / "checkpoint.json"


def save_checkpoint(series_dir: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(checkpoint_path(series_dir), checkpoint)


def expected_incomplete_files(
    checkpoint: dict[str, Any],
) -> set[str]:
    expected = {"checkpoint.json", "protocol.json", "tokenization.json"}
    claimed_paths: set[str] = set()
    completed = checkpoint.get("completed_stimuli")
    if not isinstance(completed, dict):
        raise BatteryError("Checkpoint nemá mapu completed_stimuli.")
    for stimulus_id, entry in completed.items():
        if not isinstance(stimulus_id, str) or not isinstance(entry, dict):
            raise BatteryError("Checkpoint obsahuje neplatnou dokončenou položku.")
        for key in ("metadata_file", "state_file"):
            relative = entry.get(key)
            if not isinstance(relative, str):
                raise BatteryError(f"{stimulus_id}: checkpoint postrádá {key}.")
            if relative in claimed_paths:
                raise BatteryError(
                    f"Checkpoint odkazuje vícekrát na raw cestu {relative}."
                )
            claimed_paths.add(relative)
            expected.add(relative)
    return expected


def inventory_files(series_dir: Path, *, exclude_manifest: bool = True) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(series_dir.rglob("*")):
        if path.is_symlink():
            raise BatteryError(f"Raw bundle nesmí obsahovat symlink: {path}.")
        if not path.is_file():
            continue
        relative = path.relative_to(series_dir).as_posix()
        if exclude_manifest and relative == "manifest.json":
            continue
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise BatteryError(f"Raw bundle obsahuje nebezpečnou cestu: {relative}.")
        inventory.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def verify_incomplete_inventory(series_dir: Path, checkpoint: dict[str, Any]) -> None:
    actual = {item["path"] for item in inventory_files(series_dir)}
    expected = expected_incomplete_files(checkpoint)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BatteryError(
            "Incomplete raw inventář nesouhlasí s checkpointem; "
            f"missing={missing}, extra={extra}."
        )


def create_manifest(series_dir: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    if checkpoint.get("status") != "complete":
        raise BatteryError("Manifest lze vytvořit jen pro úplný checkpoint.")
    return {
        "format": f"{FORMAT}.manifest",
        "schema_version": SCHEMA_VERSION,
        "series_id": checkpoint["series_id"],
        "status": "complete",
        "completed_at": checkpoint["completed_at"],
        "protocol_sha256": checkpoint["protocol_sha256"],
        "tokenizer_sha256": checkpoint["tokenizer_sha256"],
        "runner_sha256": checkpoint["runner_sha256"],
        "model_identity": checkpoint["model_identity"],
        "model_metadata_fingerprint": checkpoint["model_metadata_fingerprint"],
        "model_content_fingerprint": checkpoint["model_content_fingerprint"],
        "selected_country_ids": checkpoint["selected_country_ids"],
        "selected_stimulus_ids": checkpoint["selected_stimulus_ids"],
        "files": inventory_files(series_dir),
    }


def verify_manifest(series_dir: Path) -> dict[str, Any]:
    manifest_path = series_dir / "manifest.json"
    if manifest_path.is_symlink():
        raise BatteryError("Raw manifest nesmí být symlink.")
    manifest = load_json(manifest_path)
    if (
        manifest.get("format") != f"{FORMAT}.manifest"
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise BatteryError("Raw manifest není úplný nebo podporovaný.")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise BatteryError("Raw manifest nemá inventář.")
    declared: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise BatteryError("Raw manifest obsahuje neplatnou položku.")
        relative = item["path"]
        if relative in declared:
            raise BatteryError(f"Raw manifest duplikuje cestu {relative}.")
        path = _safe_relative_file(series_dir, relative, "manifest")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise BatteryError(f"Raw bundle neodpovídá manifestu: {relative}.")
        declared[relative] = item
    actual = {item["path"] for item in inventory_files(series_dir)}
    if set(declared) != actual:
        raise BatteryError("Manifest neobsahuje přesně všechny raw soubory.")
    checkpoint = load_json(checkpoint_path(series_dir))
    if checkpoint.get("status") != "complete":
        raise BatteryError("Manifest odkazuje na neúplný checkpoint.")
    for field in (
        "series_id",
        "completed_at",
        "protocol_sha256",
        "tokenizer_sha256",
        "runner_sha256",
        "model_identity",
        "model_metadata_fingerprint",
        "model_content_fingerprint",
        "selected_country_ids",
        "selected_stimulus_ids",
    ):
        if manifest.get(field) != checkpoint.get(field):
            raise BatteryError(f"Manifest a checkpoint se liší v poli {field}.")
    return manifest


def _new_series_id(protocol_sha256: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"task-vs-form-{stamp}-{protocol_sha256[:8]}"


def _validate_series_id(series_id: str) -> None:
    if not SERIES_RE.fullmatch(series_id):
        raise BatteryError("Neplatné --series-id.")


def _checkpoint_base_validation(
    *,
    checkpoint: dict[str, Any],
    series_id: str,
    protocol_sha256: str,
    tokenizer_sha256: str,
    runner_sha256: str,
    model_mode: str,
) -> None:
    if (
        checkpoint.get("format") != FORMAT
        or checkpoint.get("schema_version") != SCHEMA_VERSION
        or checkpoint.get("series_id") != series_id
    ):
        raise BatteryError("Resume checkpoint má jiný formát, verzi nebo series_id.")
    expected = {
        "protocol_sha256": protocol_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "runner_sha256": runner_sha256,
        "model_mode": model_mode,
        "position_order": list(POSITION_ORDER),
        "layer_order": list(LAYER_ORDER),
        "state_shape": list(STATE_SHAPE),
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise BatteryError(f"Resume odmítá změněnou identitu {field}.")
    if checkpoint.get("status") not in {"incomplete", "complete"}:
        raise BatteryError("Checkpoint má neplatný status.")


def _selected_for_resume(
    *,
    protocol: dict[str, Any],
    requested_country_ids: list[str],
    limit: int | None,
    checkpoint: dict[str, Any],
) -> list[dict[str, Any]]:
    stored = checkpoint.get("selected_country_ids")
    if not isinstance(stored, list) or not stored:
        raise BatteryError("Checkpoint nemá výběr zemí.")
    if requested_country_ids or limit is not None:
        requested = [
            item["country_id"]
            for item in choose_countries(protocol, requested_country_ids, limit)
        ]
        if requested != stored:
            raise BatteryError("Resume musí použít původní výběr zemí.")
    return choose_countries(protocol, list(stored), None)


def _write_stimulus_result(
    *,
    series_dir: Path,
    series_id: str,
    stimulus: dict[str, Any],
    tokenization: dict[str, Any],
    states: Any,
    prediction: dict[str, Any],
    timing: dict[str, float],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    stimulus_id = stimulus["stimulus_id"]
    state_relative = Path("states") / f"{stimulus_id}.safetensors"
    metadata_relative = Path("stimuli") / f"{stimulus_id}.json"
    state_path = series_dir / state_relative
    metadata_path = series_dir / metadata_relative
    if state_path.exists() or metadata_path.exists():
        raise BatteryError(
            f"{stimulus_id}: cílový raw soubor už existuje mimo checkpoint."
        )
    atomic_write_safetensors(state_path, states)
    state_sha = sha256_file(state_path)
    captured_at = utc_now()
    metadata = {
        "format": f"{FORMAT}.stimulus",
        "schema_version": SCHEMA_VERSION,
        "series_id": series_id,
        "captured_at": captured_at,
        "protocol_sha256": checkpoint["protocol_sha256"],
        "tokenizer_sha256": checkpoint["tokenizer_sha256"],
        "runner_sha256": checkpoint["runner_sha256"],
        "model_identity": checkpoint["model_identity"],
        "model_metadata_fingerprint": checkpoint["model_metadata_fingerprint"],
        "model_content_fingerprint": checkpoint["model_content_fingerprint"],
        "stimulus": stimulus,
        "tokenization": tokenization,
        "state": {
            "file": state_relative.as_posix(),
            "sha256": state_sha,
            "key": STATE_KEY,
            "shape": list(STATE_SHAPE),
            "dtype": "float32",
            "axes": ["layer", "position", "hidden"],
            "layers": list(LAYER_ORDER),
            "positions": list(POSITION_ORDER),
            "hook": "post_decoder_block_residual",
        },
        "final_distribution": prediction,
        "timing_ms": timing,
    }
    atomic_write_json(metadata_path, metadata)
    return {
        "metadata_file": metadata_relative.as_posix(),
        "metadata_sha256": sha256_file(metadata_path),
        "state_file": state_relative.as_posix(),
        "state_sha256": state_sha,
        "completed_at": captured_at,
    }


def _derive_if_available(series_dir: Path, results_dir: Path) -> bool:
    """Call the optional analysis writer without making collection depend on it."""

    for module_name in ("derive_task_form_battery", "task_form_analysis"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise BatteryError(
                    f"Derived writer {module_name} má chybějící závislost: {exc}"
                ) from exc
            continue
        hook = getattr(module, "write_derived_bundle", None)
        if not callable(hook):
            continue
        if module_name == "derive_task_form_battery":
            hook(
                series_dir=series_dir,
                results_dir=results_dir,
                protocol_path=None,
            )
        else:
            hook(series_dir=series_dir, results_dir=results_dir)
        return True
    print(
        "Raw sběr je úplný. Derived hook je připraven pro "
        "derive_task_form_battery.write_derived_bundle("
        "series_dir=…, results_dir=…, protocol_path=None).",
        flush=True,
    )
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--series-id")
    parser.add_argument(
        "--model-mode",
        choices=MODEL_MODES,
        default="hybrid",
        help="Hybrid je primární běh; BF16 slouží jako referenční replikace.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_BF16_MODEL,
        help="Cesta k BF16 modelu (stejný výchozí model jako server.py).",
    )
    parser.add_argument(
        "--model-8bit",
        type=Path,
        default=DEFAULT_8BIT_MODEL,
        help="Cesta k 8bit tělu hybridu (stejný výchozí model jako server.py).",
    )
    parser.add_argument(
        "--country-id",
        action="append",
        default=[],
        help="Opakovatelné country_id; bez volby se použije všech deset zemí.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Rozvine a tokenizačně ověří všech 180 stimulů bez importu MLX/modelu.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-derive",
        action="store_true",
        help=(
            "Po dokončení ponechá jen ověřený raw bundle; hodí se pro "
            "jednozemní časovací smoke test."
        ),
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Kořen odvozených výstupů; každá série dostane vlastní "
            "podadresář <series-id>."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = args.protocol.expanduser().resolve()
    bf16_model_path = args.model.expanduser().resolve()
    eight_bit_model_path = args.model_8bit.expanduser().resolve()
    tokenizer_path = bf16_model_path / "tokenizer.json"
    runner_path = Path(__file__).resolve()

    protocol, protocol_bytes, protocol_sha = load_protocol(protocol_path)
    validate_protocol(protocol)
    stimuli_all = expand_protocol(protocol)
    selected_countries_initial = choose_countries(
        protocol, args.country_id, args.limit
    )
    selected_country_ids_initial = [
        item["country_id"] for item in selected_countries_initial
    ]
    selected_stimuli_initial = select_stimuli(
        stimuli_all, selected_country_ids_initial
    )

    tokenizer_sha_before = sha256_file(tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizations_all = tokenize_stimuli(
        tokenizer,
        stimuli_all,
        response_boundary_text=protocol["design"]["response_boundary_text"],
    )
    tokenizer_sha = sha256_file(tokenizer_path)
    if tokenizer_sha != tokenizer_sha_before:
        raise BatteryError("tokenizer.json se během validace změnil.")
    runner_sha = sha256_file(runner_path)

    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "protocol_id": protocol["protocol_id"],
                    "protocol_sha256": protocol_sha,
                    "tokenizer_sha256": tokenizer_sha,
                    "validated_stimuli": len(stimuli_all),
                    "generative_stimuli": sum(
                        item["family"] == "generative" for item in stimuli_all
                    ),
                    "verification_stimuli": sum(
                        item["family"] == "verification" for item in stimuli_all
                    ),
                    "selected_country_ids": selected_country_ids_initial,
                    "selected_stimuli": len(selected_stimuli_initial),
                    "position_order": list(POSITION_ORDER),
                    "common_response_boundary": next(
                        iter(tokenizations_all.values())
                    )["anchors"]["response_boundary"],
                    "mlx_or_model_loaded": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.resume and args.series_id is None:
        raise BatteryError("--resume vyžaduje explicitní --series-id.")
    series_id = args.series_id or _new_series_id(protocol_sha)
    _validate_series_id(series_id)
    raw_root = args.raw_root.expanduser().resolve()
    results_root = args.results_dir.expanduser().resolve()
    series_dir = raw_root / series_id
    derived_results_dir = results_root / series_id

    checkpoint: dict[str, Any] | None = None
    if args.resume:
        if not series_dir.is_dir():
            raise BatteryError(f"Resume raw bundle neexistuje: {series_dir}.")
        checkpoint = load_json(checkpoint_path(series_dir))
        _checkpoint_base_validation(
            checkpoint=checkpoint,
            series_id=series_id,
            protocol_sha256=protocol_sha,
            tokenizer_sha256=tokenizer_sha,
            runner_sha256=runner_sha,
            model_mode=args.model_mode,
        )
        selected_countries = _selected_for_resume(
            protocol=protocol,
            requested_country_ids=args.country_id,
            limit=args.limit,
            checkpoint=checkpoint,
        )
    else:
        if series_dir.exists():
            raise BatteryError(
                f"Série {series_dir} už existuje; použij --resume nebo jiné ID."
            )
        selected_countries = selected_countries_initial

    selected_country_ids = [item["country_id"] for item in selected_countries]
    selected_stimuli = select_stimuli(stimuli_all, selected_country_ids)
    selected_stimulus_ids = [item["stimulus_id"] for item in selected_stimuli]
    if checkpoint is not None:
        if checkpoint.get("selected_stimulus_ids") != selected_stimulus_ids:
            raise BatteryError("Resume checkpoint má jiný nebo nevyvážený výběr stimulů.")
        if checkpoint.get("status") == "complete":
            manifest = verify_manifest(series_dir)
            if manifest.get("series_id") != series_id:
                raise BatteryError("Manifest patří jiné sérii.")

    state = None
    try:
        state, server = load_model_state(
            model_mode=args.model_mode,
            bf16_model_path=bf16_model_path,
            eight_bit_model_path=eight_bit_model_path,
        )
        observed_model_identity = model_identity(state)

        if checkpoint is not None:
            if checkpoint.get("model_identity") != observed_model_identity:
                raise BatteryError("Resume odmítá jiný modelový fingerprint.")
            if (
                checkpoint.get("model_metadata_fingerprint")
                != observed_model_identity["metadata_fingerprint"]
                or checkpoint.get("model_content_fingerprint")
                != observed_model_identity["content_fingerprint"]
            ):
                raise BatteryError("Resume odmítá nekonzistentní modelové hashe.")
            stored_tokenization_sha = checkpoint.get("tokenization_sha256")
            if (
                sha256_file(series_dir / "protocol.json") != protocol_sha
                or sha256_file(series_dir / "tokenization.json")
                != stored_tokenization_sha
            ):
                raise BatteryError("Raw protokol nebo tokenizace se změnily.")
            completed = checkpoint.get("completed_stimuli")
            if not isinstance(completed, dict):
                raise BatteryError("Checkpoint nemá completed_stimuli.")
            if any(key not in selected_stimulus_ids for key in completed):
                raise BatteryError("Checkpoint obsahuje stimulus mimo uzamčený výběr.")
            for stimulus_id, entry in completed.items():
                verify_completed_entry(
                    series_dir=series_dir,
                    stimulus_id=stimulus_id,
                    entry=entry,
                    checkpoint=checkpoint,
                )
            if checkpoint["status"] == "complete":
                print(f"Raw bundle je již úplný a ověřený: {series_dir}", flush=True)
                state._release_model_local()
                state = None
                if not args.no_derive:
                    try:
                        _derive_if_available(series_dir, derived_results_dir)
                    finally:
                        verify_manifest(series_dir)
                return 0
            if (series_dir / "manifest.json").exists():
                raise BatteryError("Neúplný checkpoint nesmí mít finální manifest.")
            verify_incomplete_inventory(series_dir, checkpoint)
            checkpoint["sessions"].append(
                {
                    "session_id": uuid.uuid4().hex,
                    "started_at": utc_now(),
                    "resumed": True,
                }
            )
            save_checkpoint(series_dir, checkpoint)
        else:
            series_dir.mkdir(parents=True)
            tokenization_payload = tokenizer_audit_payload(
                tokenizer_path=tokenizer_path,
                tokenizer_sha256=tokenizer_sha,
                protocol_sha256=protocol_sha,
                stimuli=stimuli_all,
                tokenizations=tokenizations_all,
                selected_stimulus_ids=selected_stimulus_ids,
            )
            atomic_write(series_dir / "protocol.json", protocol_bytes)
            atomic_write_json(series_dir / "tokenization.json", tokenization_payload)
            checkpoint = {
                "format": FORMAT,
                "schema_version": SCHEMA_VERSION,
                "series_id": series_id,
                "status": "incomplete",
                "started_at": utc_now(),
                "updated_at": utc_now(),
                "completed_at": None,
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha,
                "tokenizer_sha256": tokenizer_sha,
                "tokenization_sha256": sha256_file(
                    series_dir / "tokenization.json"
                ),
                "runner_sha256": runner_sha,
                "model_mode": args.model_mode,
                "model_identity": observed_model_identity,
                "model_metadata_fingerprint": observed_model_identity[
                    "metadata_fingerprint"
                ],
                "model_content_fingerprint": observed_model_identity[
                    "content_fingerprint"
                ],
                "selected_country_ids": selected_country_ids,
                "selected_stimulus_ids": selected_stimulus_ids,
                "position_order": list(POSITION_ORDER),
                "layer_order": list(LAYER_ORDER),
                "state_shape": list(STATE_SHAPE),
                "completed_stimuli": {},
                "runtime": {
                    "python_executable": sys.executable,
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                },
                "sessions": [
                    {
                        "session_id": uuid.uuid4().hex,
                        "started_at": utc_now(),
                        "resumed": False,
                    }
                ],
                "last_error": None,
            }
            save_checkpoint(series_dir, checkpoint)

        assert checkpoint is not None
        completed = checkpoint["completed_stimuli"]
        try:
            for index, stimulus in enumerate(selected_stimuli, start=1):
                stimulus_id = stimulus["stimulus_id"]
                if stimulus_id in completed:
                    print(
                        f"[{index:03d}/{len(selected_stimuli):03d}] "
                        f"{stimulus_id}: checkpoint",
                        flush=True,
                    )
                    continue
                tokenization = tokenizations_all[stimulus_id]
                runtime_ids = runtime_token_ids(
                    server, state.tokenizer, stimulus["prompt"]
                )
                if runtime_ids != tokenization["prompt_token_ids"]:
                    raise BatteryError(
                        f"{stimulus_id}: runtime tokenizer se neshoduje s "
                        "Rust tokenizers auditem."
                    )
                print(
                    f"[{index:03d}/{len(selected_stimuli):03d}] "
                    f"{stimulus_id}: cache-free forward",
                    flush=True,
                )
                states, logits, timing = capture_states_and_logits(
                    state=state,
                    token_ids=runtime_ids,
                    anchor_token_indices=tokenization[
                        "anchor_token_indices"
                    ],
                )
                prediction = summarize_final_distribution(
                    logits=logits,
                    expected_first_token_id=tokenization["target"][
                        "first_token_id"
                    ],
                    tokenizer=tokenizer,
                    top_k=8,
                )
                entry = _write_stimulus_result(
                    series_dir=series_dir,
                    series_id=series_id,
                    stimulus=stimulus,
                    tokenization=tokenization,
                    states=states,
                    prediction=prediction,
                    timing=timing,
                    checkpoint=checkpoint,
                )
                completed[stimulus_id] = entry
                checkpoint["last_error"] = None
                save_checkpoint(series_dir, checkpoint)

            if (
                len(completed) != len(selected_stimulus_ids)
                or set(completed) != set(selected_stimulus_ids)
            ):
                raise BatteryError(
                    "Množina dokončených stimulů neodpovídá protokolu."
                )
            if sha256_file(protocol_path) != protocol_sha:
                raise BatteryError("Protokol se během série změnil.")
            if sha256_file(tokenizer_path) != tokenizer_sha:
                raise BatteryError("Tokenizer se během série změnil.")
            if sha256_file(runner_path) != runner_sha:
                raise BatteryError("Collection runner se během série změnil.")
            verify_model_identity_unchanged(state, observed_model_identity)
            for stimulus_id, entry in completed.items():
                verify_completed_entry(
                    series_dir=series_dir,
                    stimulus_id=stimulus_id,
                    entry=entry,
                    checkpoint=checkpoint,
                )
            checkpoint["status"] = "complete"
            checkpoint["completed_at"] = utc_now()
            checkpoint["sessions"][-1]["completed_at"] = checkpoint["completed_at"]
            checkpoint["last_error"] = None
            save_checkpoint(series_dir, checkpoint)
            manifest = create_manifest(series_dir, checkpoint)
            atomic_write_json(series_dir / "manifest.json", manifest)
            verify_manifest(series_dir)
        except Exception as exc:
            checkpoint["status"] = "incomplete"
            checkpoint["last_error"] = {
                "at": utc_now(),
                "type": type(exc).__name__,
                "message": str(exc),
            }
            save_checkpoint(series_dir, checkpoint)
            raise
    finally:
        if state is not None:
            try:
                state._release_model_local()
            except Exception:
                pass

    print(f"Raw bundle: {series_dir}", flush=True)
    derived = (
        False
        if args.no_derive
        else _derive_if_available(series_dir, derived_results_dir)
    )
    verify_manifest(series_dir)
    if derived:
        print(f"Derived outputs: {derived_results_dir}", flush=True)
    elif args.no_derive:
        print("Derived outputs skipped (--no-derive).", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BatteryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
