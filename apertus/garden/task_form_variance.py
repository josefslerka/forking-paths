"""Balanced task/form variance decomposition of residual states.

The analysis is deliberately descriptive.  For every
``family × truth_value × layer × position`` cell it unit-normalizes each
residual-state vector and constructs the balanced tensor
``country_item × relation × form × hidden``.  The centered tensor is then
split into the seven mutually orthogonal factorial components.

Generative cells use only the preregistered g1/g2 forms.  Verification cells
use all three preregistered forms.  L0 is never analyzed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


FORMAT = "apertus.task-form-variance"
SCHEMA_VERSION = 1

COMPONENTS = (
    "country_item",
    "relation",
    "form",
    "country_x_relation",
    "country_x_form",
    "relation_x_form",
    "country_x_relation_x_form",
)

COMPONENT_CODES = {
    "country_item": "C",
    "relation": "R",
    "form": "F",
    "country_x_relation": "CR",
    "country_x_form": "CF",
    "relation_x_form": "RF",
    "country_x_relation_x_form": "CRF",
}


class VarianceError(RuntimeError):
    """The balanced design or a numeric invariant was violated."""


def _as_finite_float_tensor(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if (
        array.ndim != 4
        or any(size < 1 for size in array.shape)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise VarianceError(
            "Rozklad vyžaduje reálný neprázdný tensor C × R × F × hidden."
        )
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise VarianceError("Rozklad odmítá NaN a infinity.")
    return result


def unit_normalize_states(values: Any) -> np.ndarray:
    """Return unit-normalized vectors along the hidden-state axis."""

    array = _as_finite_float_tensor(values)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        raise VarianceError(
            "Unit-normalizace narazila na nulový nebo neplatný jmenovatel."
        )
    normalized = array / norms
    if not np.all(np.isfinite(normalized)):
        raise VarianceError("Unit-normalizace vytvořila neplatné hodnoty.")
    return normalized


def decompose_balanced_tensor(values: Any) -> dict[str, Any]:
    """Decompose a balanced C × R × F × hidden tensor.

    ``values`` are assumed to be on the scale that should be decomposed.  The
    high-level :func:`analyze_states` entry point unit-normalizes raw residual
    states before calling this function.

    The returned component arrays all have the full input shape, so their sum
    directly reconstructs ``centered``.  Their squared Frobenius norms are the
    factorial sums of squares with the appropriate replication multiplicity.
    """

    tensor = _as_finite_float_tensor(values)
    grand_mean = np.mean(tensor, axis=(0, 1, 2), keepdims=True)
    centered = tensor - grand_mean

    country = np.mean(tensor, axis=(1, 2), keepdims=True) - grand_mean
    relation = np.mean(tensor, axis=(0, 2), keepdims=True) - grand_mean
    form = np.mean(tensor, axis=(0, 1), keepdims=True) - grand_mean

    country_relation = (
        np.mean(tensor, axis=2, keepdims=True)
        - grand_mean
        - country
        - relation
    )
    country_form = (
        np.mean(tensor, axis=1, keepdims=True)
        - grand_mean
        - country
        - form
    )
    relation_form = (
        np.mean(tensor, axis=0, keepdims=True)
        - grand_mean
        - relation
        - form
    )

    shape = tensor.shape
    components: dict[str, np.ndarray] = {
        "country_item": np.broadcast_to(country, shape),
        "relation": np.broadcast_to(relation, shape),
        "form": np.broadcast_to(form, shape),
        "country_x_relation": np.broadcast_to(country_relation, shape),
        "country_x_form": np.broadcast_to(country_form, shape),
        "relation_x_form": np.broadcast_to(relation_form, shape),
    }
    lower_order = np.zeros_like(centered)
    for component in components.values():
        lower_order += component
    components["country_x_relation_x_form"] = centered - lower_order

    component_ss = {
        name: float(np.sum(component * component, dtype=np.float64))
        for name, component in components.items()
    }
    total_ss = float(np.sum(centered * centered, dtype=np.float64))
    if not np.isfinite(total_ss) or total_ss <= 0.0:
        raise VarianceError(
            "Celkový centrovaný součet čtverců musí být konečný a kladný."
        )
    if any(not np.isfinite(value) or value < 0.0 for value in component_ss.values()):
        raise VarianceError("Komponentní součet čtverců není platný.")

    return {
        "grand_mean": grand_mean,
        "centered": centered,
        "components": components,
        "component_ss": component_ss,
        "total_ss": total_ss,
    }


def _string_sequence(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise VarianceError(f"{label} musí být neprázdný seznam řetězců.")
    result = list(value)
    if len(result) != len(set(result)):
        raise VarianceError(f"{label} obsahuje duplicity.")
    return result


def _protocol_axes(
    protocol: Mapping[str, Any],
) -> tuple[
    list[Any],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    try:
        design = protocol["design"]
        relations_config = protocol["relations"]
        verification = protocol["verification"]
        countries_config = protocol["countries"]
    except (KeyError, TypeError) as exc:
        raise VarianceError("Protokol postrádá faktorový design.") from exc
    if not isinstance(design, Mapping):
        raise VarianceError("protocol.design musí být objekt.")
    if not isinstance(relations_config, Mapping):
        raise VarianceError("protocol.relations musí být objekt.")
    if not isinstance(verification, Mapping):
        raise VarianceError("protocol.verification musí být objekt.")

    layers = list(design.get("layers", []))
    if not layers or len(layers) != len(set(layers)):
        raise VarianceError("Vrstvy musí být neprázdné a unikátní.")
    positions = _string_sequence(
        design.get("measurement_positions"), "measurement_positions"
    )
    relation_ids = _string_sequence(
        list(relations_config), "relation_ids"
    )

    primary_forms = design.get("primary_index_forms")
    if not isinstance(primary_forms, Mapping):
        raise VarianceError("Chybí design.primary_index_forms.")
    generative_forms = _string_sequence(
        primary_forms.get("generative"), "generative primary forms"
    )
    if generative_forms != ["g1_predicate", "g2_question"]:
        raise VarianceError(
            "Generativní rozklad musí používat právě g1_predicate a g2_question."
        )
    verification_forms = _string_sequence(
        verification.get("form_ids"), "verification form_ids"
    )
    if len(verification_forms) != 3:
        raise VarianceError("Verifikační rozklad musí používat právě tři formy.")
    if list(primary_forms.get("verification", [])) != verification_forms:
        raise VarianceError(
            "Primární verifikační formy musí odpovídat všem třem form_ids."
        )

    truth_config = verification.get("truth_values")
    if isinstance(truth_config, Mapping):
        truth_values = _string_sequence(
            list(truth_config), "verification truth_values"
        )
    else:
        truth_values = _string_sequence(
            truth_config, "verification truth_values"
        )

    if (
        not isinstance(countries_config, Sequence)
        or isinstance(countries_config, (str, bytes))
        or not countries_config
    ):
        raise VarianceError("protocol.countries musí být neprázdný seznam.")
    country_ids: list[str] = []
    for country in countries_config:
        if not isinstance(country, Mapping):
            raise VarianceError("Každá země v protokolu musí být objekt.")
        country_id = country.get("country_id")
        if not isinstance(country_id, str) or not country_id:
            raise VarianceError("Země postrádá country_id.")
        country_ids.append(country_id)
    if len(country_ids) != len(set(country_ids)):
        raise VarianceError("country_id v protokolu nejsou unikátní.")

    return (
        layers,
        positions,
        relation_ids,
        generative_forms,
        verification_forms,
        truth_values,
    )


def _indexed_cells(
    *,
    protocol: Mapping[str, Any],
    stimuli: Sequence[Mapping[str, Any]],
    states_by_id: Mapping[str, Any],
    relation_ids: list[str],
    generative_forms: list[str],
    verification_forms: list[str],
    truth_values: list[str],
) -> tuple[
    dict[tuple[str, str | None, str, str, str], str],
    list[str],
]:
    country_order = [
        country["country_id"] for country in protocol["countries"]
    ]
    allowed_countries = set(country_order)
    allowed_relations = set(relation_ids)
    allowed_truth = set(truth_values)
    selected_forms = {
        "generative": set(generative_forms),
        "verification": set(verification_forms),
    }

    cells: dict[tuple[str, str | None, str, str, str], str] = {}
    seen_countries: set[str] = set()
    for stimulus in stimuli:
        if not isinstance(stimulus, Mapping):
            raise VarianceError("Každý stimul musí být objekt.")
        family = stimulus.get("family")
        if family not in selected_forms:
            continue
        form_id = stimulus.get("form_id")
        if form_id not in selected_forms[family]:
            continue
        country_id = stimulus.get("country_id")
        relation_id = stimulus.get("relation_id")
        truth_value = stimulus.get("truth_value")
        stimulus_id = stimulus.get("stimulus_id")
        if country_id not in allowed_countries:
            raise VarianceError(f"Neznámý country_item: {country_id!r}.")
        if relation_id not in allowed_relations:
            raise VarianceError(f"Neznámá relace: {relation_id!r}.")
        if family == "generative" and truth_value is not None:
            raise VarianceError("Generativní buňka nesmí mít truth_value.")
        if family == "verification" and truth_value not in allowed_truth:
            raise VarianceError(
                f"Neznámá verifikační truth_value: {truth_value!r}."
            )
        if not isinstance(stimulus_id, str) or not stimulus_id:
            raise VarianceError("Analyzovaný stimul postrádá stimulus_id.")
        if stimulus_id not in states_by_id:
            raise VarianceError(f"Chybí residual states pro {stimulus_id}.")
        key = (
            str(family),
            truth_value,
            str(country_id),
            str(relation_id),
            str(form_id),
        )
        if key in cells:
            raise VarianceError(f"Duplicitní faktorová buňka: {key}.")
        cells[key] = stimulus_id
        seen_countries.add(str(country_id))

    selected_countries = [
        country_id for country_id in country_order if country_id in seen_countries
    ]
    if not selected_countries:
        raise VarianceError("Analýza neobdržela žádný vybraný country_item.")

    strata: list[tuple[str, str | None, list[str]]] = [
        ("generative", None, generative_forms)
    ]
    strata.extend(
        ("verification", truth_value, verification_forms)
        for truth_value in truth_values
    )
    for family, truth_value, forms in strata:
        for country_id in selected_countries:
            for relation_id in relation_ids:
                for form_id in forms:
                    key = (
                        family,
                        truth_value,
                        country_id,
                        relation_id,
                        form_id,
                    )
                    if key not in cells:
                        raise VarianceError(
                            "Stimuly netvoří úplný vyvážený blok; "
                            f"chybí buňka {key}."
                        )
    return cells, selected_countries


def analyze_states(
    protocol: Mapping[str, Any],
    stimuli: Sequence[Mapping[str, Any]],
    states_by_id: Mapping[str, Any],
) -> dict[str, Any]:
    """Return long-form factorial SS rows and commutation summaries."""

    if not isinstance(protocol, Mapping):
        raise VarianceError("protocol musí být objekt.")
    if (
        not isinstance(stimuli, Sequence)
        or isinstance(stimuli, (str, bytes))
        or not stimuli
    ):
        raise VarianceError("stimuli musí být neprázdný seznam.")
    if not isinstance(states_by_id, Mapping):
        raise VarianceError("states_by_id musí být mapování.")

    (
        layers,
        positions,
        relation_ids,
        generative_forms,
        verification_forms,
        truth_values,
    ) = _protocol_axes(protocol)
    analyzed_layers = [
        (index, layer) for index, layer in enumerate(layers) if layer != 0
    ]
    if not analyzed_layers:
        raise VarianceError("Rozklad potřebuje alespoň jednu vrstvu mimo L0.")

    cells, country_ids = _indexed_cells(
        protocol=protocol,
        stimuli=stimuli,
        states_by_id=states_by_id,
        relation_ids=relation_ids,
        generative_forms=generative_forms,
        verification_forms=verification_forms,
        truth_values=truth_values,
    )

    relevant_ids = set(cells.values())
    converted: dict[str, np.ndarray] = {}
    hidden_size: int | None = None
    for stimulus_id in relevant_ids:
        states = np.asarray(states_by_id[stimulus_id])
        if (
            states.ndim != 3
            or states.shape[:2] != (len(layers), len(positions))
            or states.shape[2] < 1
            or not np.issubdtype(states.dtype, np.number)
            or np.issubdtype(states.dtype, np.complexfloating)
        ):
            raise VarianceError(
                f"{stimulus_id}: neplatný tvar residual states {states.shape}."
            )
        if hidden_size is None:
            hidden_size = int(states.shape[2])
        elif states.shape[2] != hidden_size:
            raise VarianceError("Residual states nemají shodný hidden rozměr.")
        converted[stimulus_id] = states
    if hidden_size is None:
        raise VarianceError("Analýza nemá žádné residual states.")

    strata: list[tuple[str, str | None, list[str]]] = [
        ("generative", None, generative_forms)
    ]
    strata.extend(
        ("verification", truth_value, verification_forms)
        for truth_value in truth_values
    )

    variance_rows: list[dict[str, Any]] = []
    cell_summary_rows: list[dict[str, Any]] = []
    for family, truth_value, form_ids in strata:
        for layer_index, layer in analyzed_layers:
            for position_index, position in enumerate(positions):
                tensor = np.empty(
                    (
                        len(country_ids),
                        len(relation_ids),
                        len(form_ids),
                        hidden_size,
                    ),
                    dtype=np.float64,
                )
                for country_index, country_id in enumerate(country_ids):
                    for relation_index, relation_id in enumerate(relation_ids):
                        for form_index, form_id in enumerate(form_ids):
                            stimulus_id = cells[
                                (
                                    family,
                                    truth_value,
                                    country_id,
                                    relation_id,
                                    form_id,
                                )
                            ]
                            tensor[
                                country_index,
                                relation_index,
                                form_index,
                            ] = np.asarray(
                                converted[stimulus_id][
                                    layer_index, position_index
                                ],
                                dtype=np.float64,
                            )

                normalized = unit_normalize_states(tensor)
                decomposition = decompose_balanced_tensor(normalized)
                component_ss = decomposition["component_ss"]
                total_ss = float(decomposition["total_ss"])
                proportions = {
                    name: float(component_ss[name] / total_ss)
                    for name in COMPONENTS
                }
                if any(
                    not np.isfinite(value) or value < 0.0
                    for value in proportions.values()
                ):
                    raise VarianceError("Proporce celkového SS není platná.")

                common = {
                    "family": family,
                    "truth_value": truth_value,
                    "layer": layer,
                    "position": position,
                }
                for component in COMPONENTS:
                    variance_rows.append(
                        {
                            **common,
                            "component": component,
                            "component_code": COMPONENT_CODES[component],
                            "ss": float(component_ss[component]),
                            "total_ss": total_ss,
                            "proportion_total": proportions[component],
                            "n_country_items": len(country_ids),
                            "n_relations": len(relation_ids),
                            "n_forms": len(form_ids),
                        }
                    )

                transformation_denominator_ss = float(
                    sum(
                        component_ss[name]
                        for name in (
                            "relation",
                            "form",
                            "country_x_relation",
                            "country_x_form",
                            "relation_x_form",
                            "country_x_relation_x_form",
                        )
                    )
                )
                noncommuting_ss = float(
                    component_ss["relation_x_form"]
                    + component_ss["country_x_relation_x_form"]
                )
                if (
                    not np.isfinite(transformation_denominator_ss)
                    or transformation_denominator_ss <= 0.0
                ):
                    raise VarianceError(
                        "Jmenovatel transformačního podílu musí být "
                        "konečný a kladný."
                    )
                noncommuting_share = float(
                    noncommuting_ss / transformation_denominator_ss
                )
                commutation_score = float(1.0 - noncommuting_share)
                if (
                    not np.isfinite(noncommuting_share)
                    or not 0.0 <= noncommuting_share <= 1.0
                    or not np.isfinite(commutation_score)
                ):
                    raise VarianceError("Komutační skóre není platné.")

                cell_summary_rows.append(
                    {
                        **common,
                        "total_ss": total_ss,
                        **{
                            f"{COMPONENT_CODES[name].lower()}_ss": float(
                                component_ss[name]
                            )
                            for name in COMPONENTS
                        },
                        "transformation_denominator_ss": (
                            transformation_denominator_ss
                        ),
                        "noncommuting_ss": noncommuting_ss,
                        "noncommuting_share": noncommuting_share,
                        "commutation_score": commutation_score,
                        "n_country_items": len(country_ids),
                        "n_relations": len(relation_ids),
                        "n_forms": len(form_ids),
                    }
                )

    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "variance_rows": variance_rows,
        "cell_summary_rows": cell_summary_rows,
    }


analyze_task_form_variance = analyze_states
