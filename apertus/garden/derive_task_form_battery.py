#!/usr/bin/env python3
"""Regenerate task-vs-form tables and report without model inference."""

from __future__ import annotations

import argparse
import csv
import io
import math
import platform
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

import run_task_form_battery as battery
import task_form_analysis as analysis
import task_form_variance as variance


TABLE_NAMES = (
    "stimuli",
    "residual_norms",
    "pairwise_distances",
    "surface_distances",
    "layer_indices",
    "task_direction_alignment",
    "factor_variance",
    "commutation_summary",
    "knowledge_checks",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _manifest_path(series_dir: Path, value: Any) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise battery.BatteryError("Raw manifest obsahuje neplatnou cestu.")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or value == "manifest.json"
        or "\\" in value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise battery.BatteryError(
            f"Raw manifest obsahuje nebezpečnou cestu: {value!r}."
        )
    path = series_dir.joinpath(*relative.parts)
    cursor = series_dir
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise battery.BatteryError(
                f"Raw bundle nesmí obsahovat symbolický odkaz: {value}."
            )
    try:
        path.resolve().relative_to(series_dir.resolve())
    except ValueError as exc:
        raise battery.BatteryError(
            f"Raw manifest míří mimo bundle: {value!r}."
        ) from exc
    return relative.as_posix(), path


def verify_raw_manifest(
    series_dir: Path, *, expected_series_id: str | None = None
) -> dict[str, Any]:
    """Verify the complete, exact raw inventory before reading any state."""

    manifest_path = series_dir / "manifest.json"
    if manifest_path.is_symlink():
        raise battery.BatteryError("Raw manifest nesmí být symbolický odkaz.")
    manifest = battery.load_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != f"{battery.FORMAT}.manifest"
        or manifest.get("schema_version") != battery.SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise battery.BatteryError("Raw manifest není úplný nebo podporovaný.")
    series_id = manifest.get("series_id")
    if (
        not isinstance(series_id, str)
        or (expected_series_id is not None and series_id != expected_series_id)
    ):
        raise battery.BatteryError("Raw manifest má jiné nebo neplatné series_id.")
    items = manifest.get("files")
    if not isinstance(items, list) or not items:
        raise battery.BatteryError("Raw manifest nemá úplný inventář souborů.")

    declared: dict[str, tuple[Path, dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise battery.BatteryError("Raw manifest obsahuje neplatnou položku.")
        relative, path = _manifest_path(series_dir, item.get("path"))
        if relative in declared:
            raise battery.BatteryError(
                f"Raw manifest obsahuje duplicitní cestu: {relative}."
            )
        size = item.get("bytes")
        digest = item.get("sha256")
        if (
            type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or not SHA256_RE.fullmatch(digest)
        ):
            raise battery.BatteryError(
                f"Raw manifest má neplatnou velikost nebo SHA-256: {relative}."
            )
        declared[relative] = (path, item)

    actual: set[str] = set()
    for path in series_dir.rglob("*"):
        relative = path.relative_to(series_dir).as_posix()
        if path.is_symlink():
            raise battery.BatteryError(
                f"Raw bundle nesmí obsahovat symbolický odkaz: {relative}."
            )
        if path.is_file() and relative != "manifest.json":
            actual.add(relative)
    missing = sorted(set(declared) - actual)
    extra = sorted(actual - set(declared))
    if missing or extra:
        details = []
        if missing:
            details.append(f"chybí: {', '.join(missing)}")
        if extra:
            details.append(f"navíc: {', '.join(extra)}")
        raise battery.BatteryError(
            "Raw bundle neodpovídá úplnému manifestu (" + "; ".join(details) + ")."
        )

    for relative, (path, item) in declared.items():
        if (
            path.stat().st_size != item["bytes"]
            or battery.sha256_file(path) != item["sha256"]
        ):
            raise battery.BatteryError(
                f"Raw bundle neodpovídá manifestu: {relative}."
            )
    return manifest


def _load_states(path: Path) -> np.ndarray:
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise battery.BatteryError(
            "Pro odvození je nutný balík safetensors s numpy rozhraním."
        ) from exc
    try:
        tensors = load_file(str(path))
    except Exception as exc:
        raise battery.BatteryError(
            f"Nelze načíst SafeTensors {path}: {exc}"
        ) from exc
    if set(tensors) != {"states"}:
        raise battery.BatteryError(
            f"{path.name}: SafeTensors musí obsahovat právě tensor 'states'."
        )
    states = tensors["states"]
    if states.dtype != np.float32 or tuple(states.shape) != tuple(battery.STATE_SHAPE):
        raise battery.BatteryError(
            f"{path.name}: očekáváno float32 {tuple(battery.STATE_SHAPE)}, "
            f"nalezeno {states.dtype} {tuple(states.shape)}."
        )
    if not np.isfinite(states).all():
        raise battery.BatteryError(f"{path.name}: stavy obsahují NaN nebo infinity.")
    norms = np.linalg.norm(states, axis=-1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise battery.BatteryError(f"{path.name}: stav má nulovou nebo neplatnou normu.")
    return states


def _completed_paths(entry: dict[str, Any]) -> tuple[str, str]:
    metadata_file = entry.get("metadata_file")
    state_file = entry.get("state_file")
    if not isinstance(metadata_file, str) or not isinstance(state_file, str):
        raise battery.BatteryError(
            "Checkpoint má neplatnou dokončenou položku stimulu."
        )
    return metadata_file, state_file


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise battery.BatteryError("Analýza vrátila neplatný řádek CSV.")
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    if not fields:
        raise battery.BatteryError("Analýza vrátila prázdnou tabulku.")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _levenshtein_distance(left: list[int], right: list[int]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + int(left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def build_surface_distance_rows(
    *,
    pairwise_rows: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Audit lexical overlap once per primary residual-state pair."""

    primary_kinds = {
        "different_relation_same_form",
        "same_relation_different_form",
    }
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in pairwise_rows:
        if row.get("pair_kind") not in primary_kinds:
            continue
        key = (
            row["pair_kind"],
            row["stimulus_id_left"],
            row["stimulus_id_right"],
        )
        unique.setdefault(key, row)
    rows: list[dict[str, Any]] = []
    for pair in unique.values():
        left_ids = list(
            records_by_id[pair["stimulus_id_left"]]["tokenization"][
                "prompt_token_ids"
            ]
        )
        right_ids = list(
            records_by_id[pair["stimulus_id_right"]]["tokenization"][
                "prompt_token_ids"
            ]
        )
        if not left_ids or not right_ids:
            raise battery.BatteryError(
                "Surface audit vyžaduje neprázdnou tokenizaci s BOS."
            )
        if left_ids[0] != right_ids[0]:
            raise battery.BatteryError(
                "Surface audit našel neshodný BOS token."
            )
        left_without_bos = left_ids[1:]
        right_without_bos = right_ids[1:]
        edit = _levenshtein_distance(
            left_without_bos, right_without_bos
        )
        denominator = max(
            len(left_without_bos), len(right_without_bos), 1
        )
        left_set = set(left_without_bos)
        right_set = set(right_without_bos)
        union = left_set | right_set
        jaccard = (
            len(left_set & right_set) / len(union) if union else 1.0
        )
        rows.append(
            {
                "pair_kind": pair["pair_kind"],
                "family": pair["family"],
                "stimulus_id_left": pair["stimulus_id_left"],
                "stimulus_id_right": pair["stimulus_id_right"],
                "country_id_left": pair["country_id_left"],
                "country_id_right": pair["country_id_right"],
                "relation_id_left": pair["relation_id_left"],
                "relation_id_right": pair["relation_id_right"],
                "form_id_left": pair["form_id_left"],
                "form_id_right": pair["form_id_right"],
                "truth_value_left": pair["truth_value_left"],
                "truth_value_right": pair["truth_value_right"],
                "bos_removed": True,
                "bos_token_id": left_ids[0],
                "token_count_left_without_bos": len(left_without_bos),
                "token_count_right_without_bos": len(right_without_bos),
                "token_edit_distance": edit,
                "normalized_token_edit_distance": edit / denominator,
                "token_set_jaccard_similarity": jaccard,
                "token_set_jaccard_distance": 1.0 - jaccard,
            }
        )
    if not rows:
        raise battery.BatteryError(
            "Analýza nevrátila žádné primární páry pro surface audit."
        )
    return rows


def _number(value: Any) -> str:
    if value is None:
        return "není k dispozici"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "není k dispozici"
    return f"{number:.4f}"


def build_structural_summary(
    *,
    protocol: dict[str, Any],
    variance_rows: list[dict[str, Any]],
    commutation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the balanced factorial map in the locked primary window."""

    primary_layers = set(protocol["design"]["primary_layers"])
    primary_position = protocol["preregistered_analysis"][
        "primary_position"
    ]

    def truth_equal_mean(
        rows: list[dict[str, Any]], field: str
    ) -> float:
        by_truth: dict[Any, list[float]] = {}
        for row in rows:
            value = float(row[field])
            if not math.isfinite(value):
                raise battery.BatteryError(
                    f"Strukturální mapa obsahuje neplatné {field}."
                )
            by_truth.setdefault(row.get("truth_value"), []).append(value)
        if not by_truth:
            raise battery.BatteryError(
                "Strukturální mapa nemá řádky v primárním okně."
            )
        truth_means = [
            sum(values) / len(values) for values in by_truth.values()
        ]
        return sum(truth_means) / len(truth_means)

    families: dict[str, Any] = {}
    for label, family in (("G", "generative"), ("V", "verification")):
        family_variance = [
            row
            for row in variance_rows
            if row["family"] == family
            and row["position"] == primary_position
            and row["layer"] in primary_layers
        ]
        family_commutation = [
            row
            for row in commutation_rows
            if row["family"] == family
            and row["position"] == primary_position
            and row["layer"] in primary_layers
        ]
        component_proportions = {
            component: truth_equal_mean(
                [
                    row
                    for row in family_variance
                    if row["component"] == component
                ],
                "proportion_total",
            )
            for component in variance.COMPONENTS
        }
        noncommuting_share = truth_equal_mean(
            family_commutation, "noncommuting_share"
        )
        commutation_score = truth_equal_mean(
            family_commutation, "commutation_score"
        )
        if not math.isclose(
            commutation_score,
            1.0 - noncommuting_share,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise battery.BatteryError(
                "Průměrné komutační skóre neodpovídá interakčnímu podílu."
            )
        families[label] = {
            "family": family,
            "component_proportion_of_total": component_proportions,
            "noncommuting_share": noncommuting_share,
            "commutation_score": commutation_score,
            "truth_strata_equal_weighted": family == "verification",
            "layer_count": len(primary_layers),
        }

    return {
        "status": "descriptive_structural_map_no_support_threshold",
        "position": primary_position,
        "layers": sorted(primary_layers),
        "state_scale": "unit_normalized_residual_vectors",
        "country_factor_label": "country_item",
        "components": list(variance.COMPONENTS),
        "noncommuting_components": [
            "relation_x_form",
            "country_x_relation_x_form",
        ],
        "noncommuting_share_denominator": (
            "R+F+CR+CF+RF+CRF; country_item main effect excluded"
        ),
        "interpretation": (
            "The interaction share measures failure of additive "
            "relation-by-form closure. It is not proof that learned "
            "operators commute, and country_item is not a pure lexical factor."
        ),
        "families": families,
    }


def build_exploratory_transition_summary(
    *,
    protocol: dict[str, Any],
    index_rows: list[dict[str, Any]],
    variance_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe, without a new decision threshold, the G layer trajectory."""

    position = protocol["preregistered_analysis"]["primary_position"]
    layers = [
        layer for layer in protocol["design"]["layers"] if layer != 0
    ]
    selected_country_ids = {
        row["country_id"]
        for row in index_rows
        if row["family"] == "generative"
    }
    expected_country_count = len(selected_country_ids)
    if expected_country_count < 1:
        raise battery.BatteryError(
            "Explorativní profil nemá generativní země."
        )

    layer_rows: list[dict[str, Any]] = []
    for layer in layers:
        country_values = {
            row["country_id"]: float(row["normalized_index"])
            for row in index_rows
            if row["family"] == "generative"
            and row["position"] == position
            and row["layer"] == layer
        }
        if set(country_values) != selected_country_ids:
            raise battery.BatteryError(
                f"Explorativní profil L{layer} nemá úplný blok zemí."
            )

        def component_mean(component: str) -> float:
            values = [
                float(row["proportion_total"])
                for row in variance_rows
                if row["family"] == "generative"
                and row["position"] == position
                and row["layer"] == layer
                and row["component"] == component
            ]
            if len(values) != 1 or not math.isfinite(values[0]):
                raise battery.BatteryError(
                    f"Explorativní profil L{layer} nemá komponentu {component}."
                )
            return values[0]

        values = list(country_values.values())
        layer_rows.append(
            {
                "layer": layer,
                "mean_normalized_index": sum(values) / len(values),
                "positive_country_count": sum(value > 0.0 for value in values),
                "country_count": expected_country_count,
                "form_proportion_total": component_mean("form"),
                "relation_proportion_total": component_mean("relation"),
            }
        )

    first_positive_mean = next(
        (
            row["layer"]
            for row in layer_rows
            if row["mean_normalized_index"] > 0.0
        ),
        None,
    )
    first_relation_over_form = next(
        (
            row["layer"]
            for row in layer_rows
            if row["relation_proportion_total"]
            > row["form_proportion_total"]
        ),
        None,
    )

    longest_start: int | None = None
    longest_end: int | None = None
    run_start: int | None = None
    previous_layer: int | None = None
    for row in layer_rows:
        all_positive = (
            row["positive_country_count"] == row["country_count"]
        )
        if all_positive and (
            run_start is None
            or previous_layer is None
            or row["layer"] == previous_layer + 1
        ):
            if run_start is None:
                run_start = row["layer"]
        else:
            if run_start is not None and previous_layer is not None:
                if (
                    longest_start is None
                    or previous_layer - run_start
                    > (longest_end or 0) - longest_start
                ):
                    longest_start, longest_end = run_start, previous_layer
            run_start = row["layer"] if all_positive else None
        previous_layer = row["layer"]
    if run_start is not None and previous_layer is not None:
        if (
            longest_start is None
            or previous_layer - run_start
            > (longest_end or 0) - longest_start
        ):
            longest_start, longest_end = run_start, previous_layer

    return {
        "status": "exploratory_post_hoc_no_support_threshold",
        "family": "generative",
        "position": position,
        "first_layer_with_positive_group_mean_index": first_positive_mean,
        "first_layer_where_relation_ss_share_exceeds_form": (
            first_relation_over_form
        ),
        "crossings_coincide": (
            first_positive_mean is not None
            and first_positive_mean == first_relation_over_form
        ),
        "longest_contiguous_all_countries_positive_run": {
            "start_layer": longest_start,
            "end_layer": longest_end,
        },
        "interpretation": (
            "Descriptive layer localization inspected after the locked "
            "primary-window decision; it cannot replace that decision."
        ),
        "layer_rows": layer_rows,
    }


def build_report(summary: dict[str, Any], metadata: dict[str, Any]) -> str:
    """Render the three preregistered evidential components separately."""

    primary = summary.get("primary", {})
    families = primary.get("families", {})
    generative = families.get("G", {})
    verification = families.get("V", {})
    alignment = primary.get("alignment_G", {})
    support = primary.get("support_rule", {})
    knowledge = summary.get("knowledge_sensitivity", {})
    structural = summary.get("structural_map", {})
    structural_families = structural.get("families", {})
    structural_g = structural_families.get("G", {})
    structural_v = structural_families.get("V", {})
    transition = summary.get("exploratory_transition", {})
    transition_run = transition.get(
        "longest_contiguous_all_countries_positive_run", {}
    )
    eligibility_note = (
        "Tato série obsahuje všech deset uzamčených zemí v primárním Hybrid "
        "režimu, takže pravidlo smí vydat společné rozhodnutí."
        if support.get("decision_eligible")
        else "Tato série je podmnožina nebo numerický audit; endpointy jsou "
        "pouze popisné a společné rozhodnutí se nevydává."
    )
    return f"""# Požadovaná relace vs. povrchová forma

Série `{summary.get('series_id', metadata['series_id'])}` obsahuje
{summary.get('stimulus_count', metadata.get('stimulus_count'))} uzamčených
stimulů nad {summary.get('country_count', metadata.get('country_count'))}
zeměmi. Všechny tabulky byly odvozeny z kompletního, hashově ověřeného raw
bundlu bez nové inference.

## Generativní rodina

Primární normalizovaný index na `response_boundary`, zprůměrovaný přes
L9–L24, je **{_number(generative.get('mean'))}** (95% bootstrap stabilitní
interval pevné sady zemí
{_number(generative.get('ci_low'))} až {_number(generative.get('ci_high'))}).
L9–L24 je centrální polovina 32 decoderových bloků, uzamčená před kontrolou
těchto reziduálů.
Kladná hodnota znamená, že různé formulace téže relace jsou si v průměru
bližší než stejná formulace rozdílných relací. V této rodině je však relace
spojená také s identitou očekávané odpovědi; výsledek proto sám o sobě
neodděluje úlohu od výstupního cíle. Do indexu vstupují jen
`g1_predicate` a `g2_question`; `g3_lexical_holdout` je vyhrazen výhradně
pro následující cross-fit test.

## Verifikační rodina

Při porovnání uvnitř stejné pravdivostní hodnoty — a tedy při stejném
bezprostředním výstupním štítku — je primární normalizovaný index
**{_number(verification.get('mean'))}** (95% bootstrap stabilitní interval
{_number(verification.get('ci_low'))} až {_number(verification.get('ci_high'))}).
Tato rodina
kontroluje bezprostřední cíl `pravdivý`/`nepravdivý`, zároveň ale měří
posouzení tvrzení, nikoli generování faktické odpovědi.

## Lexikálně odložená formulace

Kosinové zarovnání směru capital–currency odhadnutého na
`g1_predicate` a `g2_question` se směrem v nezahrnuté formulaci
`g3_lexical_holdout` je **{_number(alignment.get('mean'))}** (95% bootstrap
podmíněný popisný interval {_number(alignment.get('ci_low'))} až
{_number(alignment.get('ci_high'))}).
Jde o test přenosu relačního směru na formulaci, z níž nebyl odhadnut.
Pro každou zemi je směr odhadnut také bez této země. Protože se devítizemní
trénovací množiny překrývají, tento interval nevstupuje do společného
rozhodnutí; stabilita alignmentu se ověřuje opakovaným vynecháním vnější
země a novým přepočtem všech zbývajících cross-fit směrů.

## Předregistrované společné pravidlo

Konjunkční opora stability relačně asociované geometrie je označena jako
**{support.get('label', support.get('supported', 'neurčeno'))}**. Vyžaduje
u G a V současně kladný průměr, kladnou dolní mez 95% country-bootstrap
intervalu a kladný průměr u každé vynechané země. Alignment musí mít kladný
celkový průměr a zůstat kladný po každém vnějším vynechání země s úplným
refitem. Exact one-sided sign-flip zůstává diagnostikou. Chybějící souhlas
se interpretuje jen jako rodinově specifický nebo explorativní výsledek.
{eligibility_note}

## Strukturální mapa a nekomutativita

Vedle konfirmačního cross-indexu se stejné unit-normalizované reziduály
rozkládají ve vyváženém faktoriálním designu na `country_item`, relaci,
formu a jejich čtyři interakce. V primárním okně je podíl energie nesený
interakcemi relace × forma a country_item × relace × forma
**{_number(structural_g.get('noncommuting_share'))}** v G a
**{_number(structural_v.get('noncommuting_share'))}** ve V. Doplňkové
komutační skóre `1 − podíl` je
**{_number(structural_g.get('commutation_score'))}** v G a
**{_number(structural_v.get('commutation_score'))}** ve V.

Tato osa odpovídá na jinou otázku než cross-index. Cross-index měří, zda
geometrie třídí stavy spíš podle relace než podle formulace; interakční podíl
měří, nakolik se efekt relace mění s formulací, tedy selhání aditivního
uzavření relačně-formového rovnoběžníku. Je to popisná geometrická
operacionalizace, ne důkaz komutativity naučených operátorů. Faktor
`country_item` spojuje lexikální, faktické i země-specifické rozdíly a
nesmí být vydáván za čistý faktor lexika.

## Explorativní profil po vrstvách

Tento odstavec je **post-hoc popis**, nikoli nové rozhodovací pravidlo.
V generativní rodině se průměrný cross-index poprvé obrací do kladna v
L{transition.get('first_layer_with_positive_group_mean_index', '?')}.
Ve stejné vrstvě poprvé převýší podíl relačního SS podíl formového SS:
L{transition.get('first_layer_where_relation_ss_share_exceeds_form', '?')}.
Nejdelší souvislý úsek, v němž je index kladný pro všechny analyzované země,
vede od L{transition_run.get('start_layer', '?')} do
L{transition_run.get('end_layer', '?')}.

Tento souběh vysvětluje zdánlivé napětí mezi záporným generativním výsledkem
v celém předem uzamčeném okně a kladným přenosem relačního směru do `g3`.
Rané a první střední vrstvy nesou větší hlavní efekt formy; v pozdější části
se relační efekt stane větším, zatímco nízká relačně-formová interakce znamená,
že směr může být stabilní i v geometrii zpočátku dominované formou. Jde o
hypotézu pro novou replikační baterii s předem uzamčeným bodem přechodu,
nikoli o záchranu neúspěšné primární predikce této série.

## Audit povrchového překryvu

`surface_distances.csv` hlásí pro každý primární pár normalizovanou tokenovou
editační vzdálenost a token-set Jaccard po odebrání BOS. Jde o audit
lexikálního překryvu, nikoli o dodatečné vylučovací pravidlo ani náhradu
reziduálního kontrastu.

## Behaviorální kontrola

`knowledge_checks.csv` obsahuje rank a pravděpodobnost očekávaného prvního
tokenu u všech G odpovědí i V štítků pravdivosti. Citlivostní G řez
kvalifikoval {knowledge.get('qualified_country_count', 0)} zemí; vyžaduje
nejméně dvě totožná `form_id`, v nichž limit splní současně capital i
currency. Tento řez je
**{'reportován' if knowledge.get('reported') else 'nereportován'}** a nikdy
nenahrazuje úplnou uzamčenou sadu.

## Omezení

Experiment nepředpovídá obrácené U. Pozdní vrstvy mohou parafráze téže
relace dál sbližovat, protože vedou ke stejné odpovědi. Časné, střední a
pozdní profily jsou proto popisné diagnostiky, nikoli test návratu povrchové
formy. Kosinová podobnost reziduálů ukazuje reprezentanční invarianci, ne
lokalizovanou „substanci významu“. Knowledge-qualified řez je citlivostní
analýza a nenahrazuje úplnou uzamčenou množinu. Země jsou účelově vybraná
pevná sada, nikoli populační vzorek; bootstrap vyjadřuje stabilitu této sady.
L0 je uložený vstupní sanity check; index a zarovnání začínají v L1, protože
na společném tokenu je embedding mezi podmínkami totožný a kosinus
rozdílového nulového vektoru není definovaný.
Generativní rodina navíc směšuje relaci s identitou a třídou odpovědi, proto
je verifikační rodina nutnou kontrolou. Nejsilnější dovolený závěr je, že
relačně asociovaná geometrie je stabilní napříč uzamčenými parafrázemi —
nikoli že je úloha reprezentována nezávisle na lexikálním povrchu.

Raw bundle: `{metadata['raw_bundle']}`.
"""


def apply_decision_eligibility(
    *,
    summary: dict[str, Any],
    protocol: dict[str, Any],
    checkpoint: dict[str, Any],
    selected_country_ids: list[str],
) -> bool:
    """Allow a joint decision only for the full locked primary-model run."""

    try:
        support_rule = summary["primary"]["support_rule"]
        criteria_passed = bool(support_rule["supported"])
    except (KeyError, TypeError) as exc:
        raise battery.BatteryError(
            "Souhrn nemá platné společné rozhodovací pravidlo."
        ) from exc
    full_locked_country_ids = [
        country["country_id"] for country in protocol["countries"]
    ]
    decision_eligible = bool(
        selected_country_ids == full_locked_country_ids
        and checkpoint.get("model_mode")
        == protocol["design"]["required_primary_model_mode"]
    )
    support_rule["criteria_passed"] = criteria_passed
    support_rule["decision_eligible"] = decision_eligible
    support_rule["eligibility_requires"] = {
        "all_locked_countries": full_locked_country_ids,
        "model_mode": protocol["design"]["required_primary_model_mode"],
    }
    if not decision_eligible:
        support_rule["supported"] = None
        support_rule["label"] = "descriptive_only"
    return decision_eligible


def _validate_completed_entry(
    *,
    series_dir: Path,
    stimulus_id: str,
    entry: Any,
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    if not isinstance(entry, dict):
        raise battery.BatteryError(
            f"Checkpoint postrádá dokončený stimul {stimulus_id}."
        )
    metadata_file, state_file = _completed_paths(entry)
    _, metadata_path = _manifest_path(series_dir, metadata_file)
    _, state_path = _manifest_path(series_dir, state_file)
    for label, path, expected_hash in (
        ("metadata", metadata_path, entry.get("metadata_sha256")),
        ("states", state_path, entry.get("state_sha256")),
    ):
        if (
            not isinstance(expected_hash, str)
            or not SHA256_RE.fullmatch(expected_hash)
            or not path.is_file()
            or battery.sha256_file(path) != expected_hash
        ):
            raise battery.BatteryError(
                f"{stimulus_id}: hash {label} v checkpointu nesedí."
            )
    battery.verify_completed_entry(
        series_dir=series_dir,
        stimulus_id=stimulus_id,
        entry=entry,
        checkpoint=checkpoint,
    )
    metadata = battery.load_json(metadata_path)
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != f"{battery.FORMAT}.stimulus"
        or metadata.get("schema_version") != battery.SCHEMA_VERSION
        or metadata.get("series_id") != checkpoint.get("series_id")
        or metadata.get("stimulus", {}).get("stimulus_id") != stimulus_id
    ):
        raise battery.BatteryError(
            f"{stimulus_id}: metadata stimulu mají jiný formát nebo identitu."
        )
    return metadata, _load_states(state_path)


def build_audit_rows(
    *,
    protocol: dict[str, Any],
    stimuli: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    completed: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Flatten collection metadata and preregistered knowledge diagnostics."""

    stimulus_rows: list[dict[str, Any]] = []
    knowledge_rows: list[dict[str, Any]] = []
    maximum_rank = protocol["design"]["knowledge_sensitivity"][
        "target_first_token_max_rank"
    ]
    minimum_forms = protocol["design"]["knowledge_sensitivity"][
        "minimum_supported_generative_forms_per_relation"
    ]

    for stimulus in stimuli:
        stimulus_id = stimulus["stimulus_id"]
        record = records_by_id[stimulus_id]
        tokenization = record.get("tokenization")
        if not isinstance(tokenization, dict):
            raise battery.BatteryError(
                f"{stimulus_id}: chybí tokenizační audit."
            )
        anchors = tokenization.get("anchors")
        target = tokenization.get("target")
        final_distribution = record.get("final_distribution")
        expected = (
            final_distribution.get("expected_first_token")
            if isinstance(final_distribution, dict)
            else None
        )
        if (
            not isinstance(anchors, dict)
            or not isinstance(target, dict)
            or not isinstance(expected, dict)
            or type(expected.get("rank")) is not int
            or expected["rank"] < 1
            or not isinstance(expected.get("probability"), (int, float))
            or isinstance(expected.get("probability"), bool)
            or not math.isfinite(float(expected["probability"]))
            or not 0.0 <= float(expected["probability"]) <= 1.0
            or type(expected.get("in_top8")) is not bool
        ):
            raise battery.BatteryError(
                f"{stimulus_id}: neplatný audit cílového prvního tokenu."
            )
        timing = record.get("timing_ms", {})
        entry = completed[stimulus_id]
        row = dict(stimulus)
        row.update(
            {
                "captured_at": record.get("captured_at"),
                "prompt_token_count": tokenization["prompt_token_count"],
                "country_end_token_index": anchors["country_end"]["token_index"],
                "country_end_token_id": anchors["country_end"]["token_id"],
                "response_boundary_token_index": anchors[
                    "response_boundary"
                ]["token_index"],
                "response_boundary_token_id": anchors[
                    "response_boundary"
                ]["token_id"],
                "target_first_token_id": target["first_token_id"],
                "target_first_token": target["first_token"],
                "expected_first_token_rank": expected["rank"],
                "expected_first_token_probability": expected["probability"],
                "expected_first_token_in_top8": expected["in_top8"],
                "forward_and_projection_ms": timing.get(
                    "forward_and_projection_ms"
                ),
                "metadata_file": entry["metadata_file"],
                "metadata_sha256": entry["metadata_sha256"],
                "state_file": entry["state_file"],
                "state_sha256": entry["state_sha256"],
            }
        )
        stimulus_rows.append(row)
        knowledge_rows.append(
            {
                "stimulus_id": stimulus_id,
                "family": stimulus["family"],
                "country_id": stimulus["country_id"],
                "relation_id": stimulus["relation_id"],
                "form_id": stimulus["form_id"],
                "truth_value": stimulus["truth_value"],
                "knowledge_role": (
                    "factual_answer"
                    if stimulus["family"] == "generative"
                    else "truth_label"
                ),
                "expected_output": stimulus["expected_output"],
                "expected_first_token_id": target["first_token_id"],
                "expected_first_token": target["first_token"],
                "expected_first_token_rank": expected["rank"],
                "expected_first_token_probability": expected[
                    "probability"
                ],
                "expected_first_token_in_top8": expected["in_top8"],
                "rank_threshold": maximum_rank,
                "form_supported": expected["rank"] <= maximum_rank,
            }
        )

    generative_rows = [
        row for row in knowledge_rows if row["family"] == "generative"
    ]
    paired_form_support: dict[tuple[str, str], bool] = {}
    for country_id in {
        row["country_id"] for row in generative_rows
    }:
        for form_id in {
            row["form_id"] for row in generative_rows
        }:
            matching = [
                row
                for row in generative_rows
                if row["country_id"] == country_id
                and row["form_id"] == form_id
            ]
            paired_form_support[(country_id, form_id)] = (
                {row["relation_id"] for row in matching}
                == set(protocol["relations"])
                and all(row["form_supported"] for row in matching)
            )
    shared_supported_counts: dict[str, int] = {}
    for (country_id, _), supported in paired_form_support.items():
        shared_supported_counts[country_id] = (
            shared_supported_counts.get(country_id, 0) + int(supported)
        )
    qualified_ids: list[str] = []
    for country in protocol["countries"]:
        country_id = country["country_id"]
        if shared_supported_counts.get(country_id, 0) >= minimum_forms:
            qualified_ids.append(country_id)
    qualified = set(qualified_ids)
    for row in knowledge_rows:
        if row["family"] == "generative":
            row["paired_relation_form_supported"] = paired_form_support[
                (row["country_id"], row["form_id"])
            ]
        else:
            row["paired_relation_form_supported"] = None
        row["shared_supported_generative_form_count"] = (
            shared_supported_counts.get(row["country_id"], 0)
        )
        row["minimum_shared_supported_generative_forms"] = minimum_forms
        row["country_qualified"] = row["country_id"] in qualified
    return stimulus_rows, knowledge_rows, qualified_ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Odvodí tabulky a report z hotové task-vs-form baterie."
    )
    parser.add_argument("--series-id", required=True)
    parser.add_argument("--protocol", type=Path, default=battery.DEFAULT_PROTOCOL)
    parser.add_argument("--raw-root", type=Path, default=battery.DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Přesný cílový adresář; výchozí je "
            "results_task_vs_form/<series-id>."
        ),
    )
    return parser.parse_args(argv)


def write_derived_bundle(
    *,
    series_dir: Path,
    results_dir: Path,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    """Verify one complete raw series, analyze it, and atomically write outputs."""

    series_dir = series_dir.expanduser().resolve()
    results_dir = results_dir.expanduser().resolve()
    try:
        results_dir.relative_to(series_dir)
    except ValueError:
        pass
    else:
        raise battery.BatteryError(
            "Odvozené výstupy nesmějí být uvnitř neměnného raw bundlu."
        )
    series_id = series_dir.name
    if not battery.SERIES_RE.fullmatch(series_id):
        raise battery.BatteryError("Raw adresář nemá platné series_id.")
    manifest = verify_raw_manifest(
        series_dir, expected_series_id=series_id
    )
    checkpoint = battery.load_json(series_dir / "checkpoint.json")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != battery.FORMAT
        or checkpoint.get("schema_version") != battery.SCHEMA_VERSION
        or checkpoint.get("series_id") != series_id
        or checkpoint.get("status") != "complete"
    ):
        raise battery.BatteryError("Série nemá úplný podporovaný checkpoint.")
    for field in (
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
            raise battery.BatteryError(
                f"Raw manifest a checkpoint se liší v {field}."
            )

    raw_protocol_path = series_dir / "protocol.json"
    protocol = battery.load_json(raw_protocol_path)
    battery.validate_protocol(protocol)
    raw_protocol_sha = battery.sha256_file(raw_protocol_path)
    if checkpoint.get("protocol_sha256") != raw_protocol_sha:
        raise battery.BatteryError("Hash raw protokolu neodpovídá checkpointu.")
    if protocol_path is not None:
        current_path = protocol_path.expanduser().resolve()
        current_protocol = battery.load_json(current_path)
        battery.validate_protocol(current_protocol)
        if (
            battery.sha256_file(current_path) != raw_protocol_sha
            or current_protocol != protocol
        ):
            raise battery.BatteryError(
                "Aktuální uzamčený protokol neodpovídá raw sérii."
            )

    all_stimuli = battery.expand_stimuli(protocol)
    selected_country_ids = checkpoint.get("selected_country_ids")
    if (
        not isinstance(selected_country_ids, list)
        or not selected_country_ids
        or not all(isinstance(value, str) for value in selected_country_ids)
    ):
        raise battery.BatteryError("Checkpoint nemá platný výběr zemí.")
    selected_countries = battery.choose_countries(
        protocol, list(selected_country_ids), None
    )
    ordered_country_ids = [
        country["country_id"] for country in selected_countries
    ]
    if ordered_country_ids != selected_country_ids:
        raise battery.BatteryError(
            "Pořadí vybraných zemí neodpovídá uzamčenému protokolu."
        )
    selected_stimuli = battery.select_stimuli(
        all_stimuli, selected_country_ids
    )
    completed = checkpoint.get("completed_stimuli")
    expected_ids = [stimulus["stimulus_id"] for stimulus in selected_stimuli]
    if (
        not isinstance(completed, dict)
        or set(completed) != set(expected_ids)
        or len(expected_ids) == 0
        or checkpoint.get("selected_stimulus_ids") != expected_ids
    ):
        raise battery.BatteryError(
            "Checkpoint neobsahuje přesně všechny vybrané stimuly."
        )

    records_by_id: dict[str, dict[str, Any]] = {}
    states_by_id: dict[str, np.ndarray] = {}
    stimuli_by_id = {
        stimulus["stimulus_id"]: stimulus for stimulus in selected_stimuli
    }
    for stimulus_id in expected_ids:
        metadata, states = _validate_completed_entry(
            series_dir=series_dir,
            stimulus_id=stimulus_id,
            entry=completed.get(stimulus_id),
            checkpoint=checkpoint,
        )
        records_by_id[stimulus_id] = metadata
        states_by_id[stimulus_id] = states
        embedded = metadata.get("stimulus")
        if embedded != stimuli_by_id[stimulus_id]:
            raise battery.BatteryError(
                f"{stimulus_id}: metadata neodpovídají raw protokolu."
            )

    stimulus_rows, knowledge_rows, qualified_country_ids = build_audit_rows(
        protocol=protocol,
        stimuli=selected_stimuli,
        records_by_id=records_by_id,
        completed=completed,
    )
    result = analysis.analyze_states(
        protocol, selected_stimuli, states_by_id
    )
    if not isinstance(result, dict):
        raise battery.BatteryError("Analýza nevrátila podporovaný výsledek.")
    summary = result.get("summary")
    row_keys = (
        "norm_rows",
        "pairwise_rows",
        "index_rows",
        "alignment_rows",
    )
    if not isinstance(summary, dict) or any(
        not isinstance(result.get(key), list) for key in row_keys
    ):
        raise battery.BatteryError("Analýza postrádá tabulky nebo souhrn.")
    try:
        variance_result = variance.analyze_states(
            protocol, selected_stimuli, states_by_id
        )
    except variance.VarianceError as exc:
        raise battery.BatteryError(
            f"Faktoriální rozklad selhal: {exc}"
        ) from exc
    factor_variance_rows = variance_result.get("variance_rows")
    commutation_rows = variance_result.get("cell_summary_rows")
    if not isinstance(factor_variance_rows, list) or not isinstance(
        commutation_rows, list
    ):
        raise battery.BatteryError(
            "Faktoriální rozklad nevrátil očekávané tabulky."
        )
    structural_summary = build_structural_summary(
        protocol=protocol,
        variance_rows=factor_variance_rows,
        commutation_rows=commutation_rows,
    )
    exploratory_transition = build_exploratory_transition_summary(
        protocol=protocol,
        index_rows=result["index_rows"],
        variance_rows=factor_variance_rows,
    )
    surface_rows = build_surface_distance_rows(
        pairwise_rows=result["pairwise_rows"],
        records_by_id=records_by_id,
    )
    minimum_qualified = protocol["design"]["knowledge_sensitivity"][
        "minimum_qualified_countries_to_report"
    ]
    qualified_primary: dict[str, Any] | None = None
    if len(qualified_country_ids) >= minimum_qualified:
        qualified = set(qualified_country_ids)
        qualified_stimuli = [
            stimulus
            for stimulus in selected_stimuli
            if stimulus["country_id"] in qualified
        ]
        qualified_states = {
            stimulus["stimulus_id"]: states_by_id[
                stimulus["stimulus_id"]
            ]
            for stimulus in qualified_stimuli
        }
        qualified_result = analysis.analyze_states(
            protocol, qualified_stimuli, qualified_states
        )
        qualified_primary = qualified_result["summary"]["primary"]
    tables = {
        "stimuli": stimulus_rows,
        "residual_norms": result["norm_rows"],
        "pairwise_distances": result["pairwise_rows"],
        "surface_distances": surface_rows,
        "layer_indices": result["index_rows"],
        "task_direction_alignment": result["alignment_rows"],
        "factor_variance": factor_variance_rows,
        "commutation_summary": commutation_rows,
        "knowledge_checks": knowledge_rows,
    }

    summary = dict(summary)
    summary["format"] = f"{battery.FORMAT}.summary"
    summary["schema_version"] = battery.SCHEMA_VERSION
    summary["series_id"] = series_id
    summary["generated_at"] = battery.utc_now()
    summary["country_count"] = len(selected_country_ids)
    summary["stimulus_count"] = len(expected_ids)
    summary["structural_map"] = structural_summary
    summary["exploratory_transition"] = exploratory_transition
    apply_decision_eligibility(
        summary=summary,
        protocol=protocol,
        checkpoint=checkpoint,
        selected_country_ids=list(selected_country_ids),
    )
    summary["knowledge_sensitivity"] = {
        "qualified_country_ids": qualified_country_ids,
        "qualified_country_count": len(qualified_country_ids),
        "qualification_rule": (
            "at_least_two_identical_generative_form_ids_supported_for_both_relations"
        ),
        "minimum_qualified_countries_to_report": minimum_qualified,
        "reported": qualified_primary is not None,
        "primary": qualified_primary,
        "scope": "sensitivity_only_complete_locked_set_remains_primary",
    }
    metadata = {
        "format": f"{battery.FORMAT}.run-metadata",
        "schema_version": battery.SCHEMA_VERSION,
        "series_id": series_id,
        "started_at": checkpoint.get("started_at"),
        "completed_at": checkpoint.get("completed_at"),
        "derived_at": battery.utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": raw_protocol_sha,
        "collection_runner_sha256": checkpoint.get("runner_sha256"),
        "analysis_entrypoint_sha256": battery.sha256_file(Path(__file__).resolve()),
        "analysis_library_sha256": battery.sha256_file(
            Path(analysis.__file__).resolve()
        ),
        "variance_library_sha256": battery.sha256_file(
            Path(variance.__file__).resolve()
        ),
        "tokenizer_sha256": checkpoint.get("tokenizer_sha256"),
        "model_identity": checkpoint.get("model_identity"),
        "model_metadata_fingerprint": checkpoint.get(
            "model_metadata_fingerprint"
        ),
        "model_content_fingerprint": checkpoint.get(
            "model_content_fingerprint"
        ),
        "selected_country_ids": list(selected_country_ids),
        "country_count": len(selected_country_ids),
        "stimulus_count": len(expected_ids),
        "raw_bundle": str(series_dir),
        "raw_manifest_sha256": battery.sha256_file(
            series_dir / "manifest.json"
        ),
        "raw_manifest_file_count": len(manifest["files"]),
        "model_or_api_calls_during_derivation": 0,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    # Close the verification window before publishing any derived file.
    verify_raw_manifest(series_dir, expected_series_id=series_id)
    csv_payloads: dict[str, bytes] = {}
    for name in TABLE_NAMES:
        rows = tables[name]
        if not isinstance(rows, list):
            raise battery.BatteryError(f"Tabulka {name} není seznam řádků.")
        csv_payloads[name] = _csv_bytes(rows)
    summary_payload = battery.pretty_json_bytes(summary)
    metadata_payload = battery.pretty_json_bytes(metadata)
    report_payload = build_report(summary, metadata).encode("utf-8")

    results_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLE_NAMES:
        battery.atomic_write(
            results_dir / f"{name}.csv", csv_payloads[name]
        )
    battery.atomic_write(results_dir / "summary.json", summary_payload)
    battery.atomic_write(
        results_dir / "run_metadata.json", metadata_payload
    )
    battery.atomic_write(
        results_dir / "report.md",
        report_payload,
    )
    return {"summary": summary, "metadata": metadata}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not battery.SERIES_RE.fullmatch(args.series_id):
        raise battery.BatteryError("Neplatné --series-id.")
    protocol_path = args.protocol.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (battery.DEFAULT_RESULTS_DIR / args.series_id).resolve()
    )
    series_dir = raw_root / args.series_id
    write_derived_bundle(
        series_dir=series_dir,
        results_dir=output_dir,
        protocol_path=protocol_path,
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
