"""Pure NumPy analysis for the locked task-vs-form residual battery."""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Sequence

import numpy as np


FORMAT = "apertus.task-vs-form-analysis"
SCHEMA_VERSION = 1
FAMILIES = ("generative", "verification")


class AnalysisError(RuntimeError):
    """The balanced design or a numeric invariant was violated."""


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Reuse the collection-side lock without importing MLX or loading files."""

    try:
        import run_task_form_battery as runner

        if protocol.get("format") == "apertus.task-vs-form-transfer-protocol":
            import task_form_transfer
            task_form_transfer.validate_protocol(protocol)
        else:
            runner.validate_protocol(protocol)
    except AnalysisError:
        raise
    except Exception as exc:
        raise AnalysisError(f"Neplatný task-vs-form protokol: {exc}") from exc

    design = protocol["design"]
    if design.get("primary_distance") != "cosine":
        raise AnalysisError("Primární vzdálenost musí být cosine.")
    primary_layers = design.get("primary_layers")
    layers = design.get("layers")
    if (
        not isinstance(primary_layers, list)
        or not primary_layers
        or any(layer not in layers for layer in primary_layers)
    ):
        raise AnalysisError("Primární vrstvy nejsou platnou podmnožinou.")
    bands = design.get("layer_bands")
    if (
        not isinstance(bands, dict)
        or set(bands) != {"early", "middle", "late"}
        or bands["middle"] != primary_layers
    ):
        raise AnalysisError("Vrstevná pásma neodpovídají primárnímu oknu.")
    if type(design.get("bootstrap_seed")) is not int:
        raise AnalysisError("Bootstrap seed musí být celé číslo.")
    if (
        type(design.get("bootstrap_samples")) is not int
        or design["bootstrap_samples"] < 1
    ):
        raise AnalysisError("Počet bootstrap replikací musí být kladný.")


def expand_protocol(
    protocol: dict[str, Any],
    country_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the collection-identical deterministic stimulus expansion."""

    validate_protocol(protocol)
    try:
        import run_task_form_battery as runner

        if protocol.get("format") == "apertus.task-vs-form-transfer-protocol":
            import task_form_transfer
            stimuli = task_form_transfer.expand_protocol(protocol)
        else:
            stimuli = runner.expand_protocol(protocol)
        if country_ids is None:
            return stimuli
        selected = runner.choose_countries(protocol, list(country_ids), None)
        ordered_ids = [country["country_id"] for country in selected]
        return runner.select_stimuli(stimuli, ordered_ids)
    except AnalysisError:
        raise
    except Exception as exc:
        raise AnalysisError(f"Protokol nelze rozvinout: {exc}") from exc


def cosine_distance(left: Any, right: Any) -> float | np.ndarray:
    """Cosine distance along the last axis, with strict finite/nonzero checks."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 1 or a.shape[-1] < 1:
        raise AnalysisError(
            f"Kosinová vzdálenost vyžaduje stejný neprázdný tvar; "
            f"nalezeno {a.shape} a {b.shape}."
        )
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise AnalysisError("Kosinová vzdálenost odmítá NaN a infinity.")
    left_norm = np.linalg.norm(a, axis=-1)
    right_norm = np.linalg.norm(b, axis=-1)
    denominator = left_norm * right_norm
    if np.any(denominator == 0.0) or not np.all(np.isfinite(denominator)):
        raise AnalysisError("Kosinová vzdálenost odmítá nulový vektor.")
    similarity = np.sum(a * b, axis=-1) / denominator
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    if np.ndim(distance) == 0:
        return float(distance)
    return np.asarray(distance, dtype=np.float64)


def country_bootstrap(
    values: Iterable[float],
    *,
    seed: int,
    samples: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Percentile interval from resampling the country-level units."""

    array = np.asarray(list(values), dtype=np.float64)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.all(np.isfinite(array))
    ):
        raise AnalysisError("Bootstrap vyžaduje neprázdné konečné hodnoty.")
    if type(seed) is not int or type(samples) is not int or samples < 1:
        raise AnalysisError("Bootstrap vyžaduje celočíselný seed a počet vzorků.")
    if not 0.0 < confidence < 1.0:
        raise AnalysisError("Bootstrap confidence musí ležet mezi 0 a 1.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, array.size, size=(samples, array.size), endpoint=False
    )
    means = np.mean(array[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "n_countries": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "confidence": float(confidence),
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def exact_sign_flip(values: Iterable[float]) -> dict[str, Any]:
    """Exact randomization diagnostic over all country-level sign flips."""

    array = np.asarray(list(values), dtype=np.float64)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.all(np.isfinite(array))
    ):
        raise AnalysisError("Sign-flip vyžaduje neprázdné konečné hodnoty.")
    if array.size > 20:
        raise AnalysisError("Přesný sign-flip je omezen na nejvýše 20 zemí.")
    count = 1 << int(array.size)
    assignments = np.arange(count, dtype=np.uint64)[:, None]
    bit_positions = np.arange(array.size, dtype=np.uint64)[None, :]
    signs = np.where(
        ((assignments >> bit_positions) & np.uint64(1)) == 0,
        -1.0,
        1.0,
    )
    means = np.mean(signs * array[None, :], axis=1)
    observed = float(np.mean(array))
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(observed)) * 8.0
    return {
        "n_countries": int(array.size),
        "assignments": count,
        "observed_mean": observed,
        "two_sided_p": float(
            np.mean(np.abs(means) >= abs(observed) - tolerance)
        ),
        "positive_one_sided_p": float(
            np.mean(means >= observed - tolerance)
        ),
        "positive_count": int(np.count_nonzero(array > 0.0)),
        "negative_count": int(np.count_nonzero(array < 0.0)),
        "zero_count": int(np.count_nonzero(array == 0.0)),
    }


def _validated_inputs(
    protocol: dict[str, Any],
    stimuli: list[dict[str, Any]],
    states_by_id: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, np.ndarray],
    list[str],
    list[int],
    list[str],
]:
    validate_protocol(protocol)
    if not isinstance(stimuli, list) or not stimuli:
        raise AnalysisError("Analýza neobdržela žádné stimuly.")
    if not isinstance(states_by_id, dict):
        raise AnalysisError("states_by_id musí být slovník.")
    ids = [stimulus.get("stimulus_id") for stimulus in stimuli]
    if any(not isinstance(value, str) for value in ids) or len(ids) != len(
        set(ids)
    ):
        raise AnalysisError("Stimuly mají chybějící nebo duplicitní ID.")
    country_set = {
        stimulus.get("country_id")
        for stimulus in stimuli
        if isinstance(stimulus, dict)
    }
    protocol_country_ids = [
        country["country_id"]
        for country in protocol["countries"]
        if country["country_id"] in country_set
    ]
    if country_set != set(protocol_country_ids) or len(protocol_country_ids) < 2:
        raise AnalysisError(
            "Analýza vyžaduje nejméně dvě platné, úplné země."
        )
    expected = expand_protocol(protocol, protocol_country_ids)
    expected_by_id = {stimulus["stimulus_id"]: stimulus for stimulus in expected}
    if set(ids) != set(expected_by_id) or set(states_by_id) != set(ids):
        raise AnalysisError(
            "Stimuly a stavy netvoří úplný vyvážený blok vybraných zemí."
        )
    supplied_by_id = {stimulus["stimulus_id"]: stimulus for stimulus in stimuli}
    for stimulus_id, locked in expected_by_id.items():
        if supplied_by_id[stimulus_id] != locked:
            raise AnalysisError(
                f"{stimulus_id}: stimul neodpovídá uzamčenému protokolu."
            )

    layers = list(protocol["design"]["layers"])
    positions = list(protocol["design"]["measurement_positions"])
    converted: dict[str, np.ndarray] = {}
    hidden_size: int | None = None
    for stimulus_id in [stimulus["stimulus_id"] for stimulus in expected]:
        states = np.asarray(states_by_id[stimulus_id])
        if (
            states.ndim != 3
            or states.shape[:2] != (len(layers), len(positions))
            or states.shape[2] < 2
            or not np.issubdtype(states.dtype, np.number)
        ):
            raise AnalysisError(
                f"{stimulus_id}: neplatný tvar stavů {states.shape}."
            )
        if hidden_size is None:
            hidden_size = int(states.shape[2])
        elif states.shape[2] != hidden_size:
            raise AnalysisError("Stimuly nemají shodný skrytý rozměr.")
        array = np.asarray(states, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise AnalysisError(f"{stimulus_id}: stav obsahuje NaN nebo infinity.")
        norms = np.linalg.norm(array, axis=2)
        if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
            raise AnalysisError(
                f"{stimulus_id}: stav má nulovou nebo neplatnou normu."
            )
        converted[stimulus_id] = array
    return expected, converted, protocol_country_ids, layers, positions


def _state_index(
    stimuli: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str | None], dict[str, Any]]:
    index: dict[
        tuple[str, str, str, str | None], dict[str, Any]
    ] = {}
    for stimulus in stimuli:
        key = (
            stimulus["country_id"],
            stimulus["relation_id"],
            stimulus["form_id"],
            stimulus["truth_value"],
        )
        family_key = (stimulus["family"],) + key
        if family_key in index:
            raise AnalysisError(f"Duplicitní analytická buňka: {family_key}.")
        index[family_key] = stimulus
    return index


def _distance_rows(
    *,
    distances: np.ndarray,
    left: dict[str, Any],
    right: dict[str, Any],
    pair_kind: str,
    layers: list[int],
    positions: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        for position_index, position in enumerate(positions):
            rows.append(
                {
                    "pair_kind": pair_kind,
                    "family": left["family"],
                    "layer": layer,
                    "position": position,
                    "stimulus_id_left": left["stimulus_id"],
                    "stimulus_id_right": right["stimulus_id"],
                    "country_id_left": left["country_id"],
                    "country_id_right": right["country_id"],
                    "relation_id_left": left["relation_id"],
                    "relation_id_right": right["relation_id"],
                    "form_id_left": left["form_id"],
                    "form_id_right": right["form_id"],
                    "truth_value_left": left["truth_value"],
                    "truth_value_right": right["truth_value"],
                    "cosine_distance": float(
                        distances[layer_index, position_index]
                    ),
                }
            )
    return rows


def _summary_stats(
    country_values: list[float],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    result = country_bootstrap(
        country_values, seed=seed, samples=samples
    )
    result["exact_sign_flip"] = exact_sign_flip(country_values)
    result["positive"] = result["mean"] > 0.0
    return result


def _profile_stats(
    rows: list[dict[str, Any]],
    *,
    value_field: str,
    countries: list[str],
    layers: set[int],
    seed: int,
    samples: int,
) -> dict[str, Any]:
    values: list[float] = []
    for country_id in countries:
        selected = [
            float(row[value_field])
            for row in rows
            if row["country_id"] == country_id and int(row["layer"]) in layers
        ]
        if not selected:
            raise AnalysisError(
                f"Profil postrádá hodnoty pro zemi {country_id}."
            )
        values.append(float(np.mean(selected)))
    result = _summary_stats(values, seed=seed, samples=samples)
    result["country_values"] = [
        {"country_id": country_id, "mean": value}
        for country_id, value in zip(countries, values)
    ]
    result["positive_country_count"] = sum(
        value > 0.0 for value in values
    )
    result["negative_country_count"] = sum(
        value < 0.0 for value in values
    )
    result["zero_country_count"] = sum(value == 0.0 for value in values)
    result["all_country_means_positive"] = all(
        value > 0.0 for value in values
    )
    leave_one_country_out = []
    for index, country_id in enumerate(countries):
        remaining = values[:index] + values[index + 1 :]
        if not remaining:
            raise AnalysisError(
                "Leave-one-country-out vyžaduje nejméně dvě země."
            )
        leave_one_country_out.append(
            {
                "excluded_country_id": country_id,
                "mean": float(np.mean(remaining)),
            }
        )
    result["leave_one_country_out"] = leave_one_country_out
    result["leave_one_country_out_minimum"] = min(
        item["mean"] for item in leave_one_country_out
    )
    result["leave_one_country_out_maximum"] = max(
        item["mean"] for item in leave_one_country_out
    )
    result["all_leave_one_country_out_means_positive"] = all(
        item["mean"] > 0.0 for item in leave_one_country_out
    )
    result["bootstrap_interpretation"] = (
        "stability_of_the_fixed_locked_country_set"
    )
    result["prediction_supported"] = bool(
        result["mean"] > 0.0
        and result["ci_low"] > 0.0
        and result["all_leave_one_country_out_means_positive"]
    )
    return result


def analyze_states(
    protocol: dict[str, Any],
    stimuli: list[dict[str, Any]],
    states_by_id: dict[str, Any],
) -> dict[str, Any]:
    """Compute preregistered tables and country-level inferential diagnostics."""

    (
        stimuli,
        states_by_id,
        country_ids,
        layers,
        positions,
    ) = _validated_inputs(protocol, stimuli, states_by_id)
    analysis_layer_indices = [
        index for index, layer in enumerate(layers) if layer != 0
    ]
    if not analysis_layer_indices:
        raise AnalysisError("Analýza potřebuje alespoň jednu post-block vrstvu.")
    cell = _state_index(stimuli)
    generative_forms = list(
        protocol["relations"]["capital"]["generative_forms"]
    )
    holdout_form = "g3_lexical_holdout"
    if holdout_form not in generative_forms:
        raise AnalysisError("Chybí g3_lexical_holdout.")
    training_forms = [
        form_id for form_id in generative_forms if form_id != holdout_form
    ]
    if len(training_forms) != 2:
        raise AnalysisError("Primární G index musí používat právě g1 a g2.")
    verification_forms = list(
        protocol["verification"].get(
            "forms", protocol["verification"].get("form_ids", [])
        )
    )
    if len(verification_forms) != 3:
        raise AnalysisError("Verifikace musí mít tři uzamčené formy.")
    relation_ids = list(protocol["relations"])
    truth_values = list(protocol["verification"]["truth_values"])
    primary_forms = {
        "generative": training_forms,
        "verification": verification_forms,
    }

    norm_rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        norms = np.linalg.norm(
            states_by_id[stimulus["stimulus_id"]], axis=2
        )
        for layer_index, layer in enumerate(layers):
            for position_index, position in enumerate(positions):
                norm_rows.append(
                    {
                        "stimulus_id": stimulus["stimulus_id"],
                        "family": stimulus["family"],
                        "country_id": stimulus["country_id"],
                        "relation_id": stimulus["relation_id"],
                        "form_id": stimulus["form_id"],
                        "truth_value": stimulus["truth_value"],
                        "layer": layer,
                        "position": position,
                        "residual_norm": float(
                            norms[layer_index, position_index]
                        ),
                    }
                )

    pairwise_rows: list[dict[str, Any]] = []
    baseline_arrays: dict[tuple[str, str | None], list[np.ndarray]] = {}
    strata: dict[
        tuple[str, str, str, str | None], list[dict[str, Any]]
    ] = {}
    for stimulus in stimuli:
        if stimulus["form_id"] not in primary_forms[stimulus["family"]]:
            continue
        key = (
            stimulus["family"],
            stimulus["relation_id"],
            stimulus["form_id"],
            stimulus["truth_value"],
        )
        strata.setdefault(key, []).append(stimulus)
    for stratum, members in strata.items():
        if len(members) != len(country_ids):
            raise AnalysisError(f"Neúplná baseline stratum: {stratum}.")
        for left, right in itertools.combinations(members, 2):
            if left["country_id"] == right["country_id"]:
                raise AnalysisError("Baseline smí spojovat jen různé země.")
            distances = np.asarray(
                cosine_distance(
                    states_by_id[left["stimulus_id"]],
                    states_by_id[right["stimulus_id"]],
                )
            )
            baseline_key = (left["family"], left["truth_value"])
            baseline_arrays.setdefault(baseline_key, []).append(distances)
            pairwise_rows.extend(
                _distance_rows(
                    distances=distances,
                    left=left,
                    right=right,
                    pair_kind="different_country_baseline",
                    layers=layers,
                    positions=positions,
                )
            )
    baseline_medians: dict[tuple[str, str | None], np.ndarray] = {}
    expected_baselines = [("generative", None)] + [
        ("verification", truth_value) for truth_value in truth_values
    ]
    for baseline_key in expected_baselines:
        if not baseline_arrays.get(baseline_key):
            raise AnalysisError(
                f"Stratum {baseline_key} nemá baseline páry."
            )
        median = np.median(
            np.stack(baseline_arrays[baseline_key], axis=0), axis=0
        )
        analyzed_median = median[analysis_layer_indices, :]
        if (
            not np.all(np.isfinite(analyzed_median))
            or np.any(analyzed_median <= 0.0)
        ):
            raise AnalysisError(
                f"Stratum {baseline_key} má od L1 nulovou nebo neplatnou baseline."
            )
        baseline_medians[baseline_key] = median
    index_rows: list[dict[str, Any]] = []
    for country_id in country_ids:
        for family in FAMILIES:
            analysis_truth_values: list[str | None] = (
                [None] if family == "generative" else truth_values
            )
            forms = primary_forms[family]
            for analysis_truth_value in analysis_truth_values:
                task_pairs: list[
                    tuple[dict[str, Any], dict[str, Any]]
                ] = []
                form_pairs: list[
                    tuple[dict[str, Any], dict[str, Any]]
                ] = []
                for form_id in forms:
                    task_pairs.append(
                        (
                            cell[
                                (
                                    family,
                                    country_id,
                                    relation_ids[0],
                                    form_id,
                                    analysis_truth_value,
                                )
                            ],
                            cell[
                                (
                                    family,
                                    country_id,
                                    relation_ids[1],
                                    form_id,
                                    analysis_truth_value,
                                )
                            ],
                        )
                    )
                for relation_id in relation_ids:
                    for left_form, right_form in itertools.combinations(
                        forms, 2
                    ):
                        form_pairs.append(
                            (
                                cell[
                                    (
                                        family,
                                        country_id,
                                        relation_id,
                                        left_form,
                                        analysis_truth_value,
                                    )
                                ],
                                cell[
                                    (
                                        family,
                                        country_id,
                                        relation_id,
                                        right_form,
                                        analysis_truth_value,
                                    )
                                ],
                            )
                        )
                task_distances: list[np.ndarray] = []
                for left, right in task_pairs:
                    distances = np.asarray(
                        cosine_distance(
                            states_by_id[left["stimulus_id"]],
                            states_by_id[right["stimulus_id"]],
                        )
                    )
                    task_distances.append(distances)
                    pairwise_rows.extend(
                        _distance_rows(
                            distances=distances,
                            left=left,
                            right=right,
                            pair_kind="different_relation_same_form",
                            layers=layers,
                            positions=positions,
                        )
                    )
                form_distances: list[np.ndarray] = []
                for left, right in form_pairs:
                    distances = np.asarray(
                        cosine_distance(
                            states_by_id[left["stimulus_id"]],
                            states_by_id[right["stimulus_id"]],
                        )
                    )
                    form_distances.append(distances)
                    pairwise_rows.extend(
                        _distance_rows(
                            distances=distances,
                            left=left,
                            right=right,
                            pair_kind="same_relation_different_form",
                            layers=layers,
                            positions=positions,
                        )
                    )
                task_mean = np.mean(
                    np.stack(task_distances, axis=0), axis=0
                )
                form_mean = np.mean(
                    np.stack(form_distances, axis=0), axis=0
                )
                raw_index = task_mean - form_mean
                baseline_key = (family, analysis_truth_value)
                normalized = np.full(raw_index.shape, np.nan, dtype=np.float64)
                normalized[analysis_layer_indices, :] = (
                    raw_index[analysis_layer_indices, :]
                    / baseline_medians[baseline_key][analysis_layer_indices, :]
                )
                for layer_index, layer in enumerate(layers):
                    if layer == 0:
                        continue
                    for position_index, position in enumerate(positions):
                        index_rows.append(
                            {
                                "family": family,
                                "country_id": country_id,
                                "truth_value": analysis_truth_value,
                                "layer": layer,
                                "position": position,
                                "different_relation_same_form_mean_distance": float(
                                    task_mean[layer_index, position_index]
                                ),
                                "same_relation_different_form_mean_distance": float(
                                    form_mean[layer_index, position_index]
                                ),
                                "index": float(
                                    raw_index[layer_index, position_index]
                                ),
                                "baseline_median": float(
                                    baseline_medians[baseline_key][
                                        layer_index, position_index
                                    ]
                                ),
                                "normalized_index": float(
                                    normalized[layer_index, position_index]
                                ),
                                "task_pair_count": len(task_pairs),
                                "form_pair_count": len(form_pairs),
                                "verification_truth_conditioned": (
                                    family == "verification"
                                ),
                            }
                        )

    def cross_fitted_alignment(
        heldout_country_id: str,
        training_country_ids: list[str],
    ) -> tuple[np.ndarray, int]:
        if not training_country_ids:
            raise AnalysisError("Směrový cross-fit nemá žádnou trénovací zemi.")
        training_directions: list[np.ndarray] = []
        for training_country_id in training_country_ids:
            for form_id in training_forms:
                capital = cell[
                    (
                        "generative",
                        training_country_id,
                        "capital",
                        form_id,
                        None,
                    )
                ]
                currency = cell[
                    (
                        "generative",
                        training_country_id,
                        "currency",
                        form_id,
                        None,
                    )
                ]
                training_directions.append(
                    states_by_id[capital["stimulus_id"]]
                    - states_by_id[currency["stimulus_id"]]
                )
        estimated_direction = np.mean(
            np.stack(training_directions, axis=0), axis=0
        )
        heldout_capital = cell[
            (
                "generative",
                heldout_country_id,
                "capital",
                holdout_form,
                None,
            )
        ]
        heldout_currency = cell[
            (
                "generative",
                heldout_country_id,
                "currency",
                holdout_form,
                None,
            )
        ]
        heldout_direction = (
            states_by_id[heldout_capital["stimulus_id"]]
            - states_by_id[heldout_currency["stimulus_id"]]
        )
        alignment = 1.0 - np.asarray(
            cosine_distance(
                estimated_direction[analysis_layer_indices, :, :],
                heldout_direction[analysis_layer_indices, :, :],
            )
        )
        return alignment, len(training_directions)

    alignment_rows: list[dict[str, Any]] = []
    for heldout_country_id in country_ids:
        training_country_ids = [
            country_id
            for country_id in country_ids
            if country_id != heldout_country_id
        ]
        alignment, training_direction_pair_count = cross_fitted_alignment(
            heldout_country_id, training_country_ids
        )
        for analyzed_index, layer_index in enumerate(analysis_layer_indices):
            layer = layers[layer_index]
            for position_index, position in enumerate(positions):
                alignment_rows.append(
                    {
                        "family": "generative",
                        "country_id": heldout_country_id,
                        "layer": layer,
                        "position": position,
                        "training_forms": ",".join(training_forms),
                        "heldout_form": holdout_form,
                        "leave_one_country_out": True,
                        "training_country_count": len(training_country_ids),
                        "training_direction_pair_count": (
                            training_direction_pair_count
                        ),
                        "cosine_alignment": float(
                            alignment[analyzed_index, position_index]
                        ),
                    }
                )

    seed = protocol["design"]["bootstrap_seed"]
    samples = protocol["design"]["bootstrap_samples"]
    primary_layers = set(protocol["design"]["primary_layers"])
    primary_position = protocol["preregistered_analysis"][
        "primary_position"
    ]
    primary: dict[str, Any] = {
        "position": primary_position,
        "layers": sorted(primary_layers),
        "families": {},
    }
    for label, family in (("G", "generative"), ("V", "verification")):
        family_rows = [
            row
            for row in index_rows
            if row["family"] == family
            and row["position"] == primary_position
        ]
        stats = _profile_stats(
            family_rows,
            value_field="normalized_index",
            countries=country_ids,
            layers=primary_layers,
            seed=seed,
            samples=samples,
        )
        stats["family"] = family
        stats["metric"] = f"mean_L{min(primary_layers)}_L{max(primary_layers)}_normalized_index_by_country"
        primary["families"][label] = stats
    primary_alignment_rows = [
        row
        for row in alignment_rows
        if row["position"] == primary_position
    ]
    alignment_stats = _profile_stats(
        primary_alignment_rows,
        value_field="cosine_alignment",
        countries=country_ids,
        layers=primary_layers,
        seed=seed,
        samples=samples,
    )
    alignment_stats["family"] = "generative"
    alignment_stats["metric"] = (
        f"mean_L{min(primary_layers)}_L{max(primary_layers)}_leave_one_country_out_g1_g2_to_g3_cosine_alignment"
    )
    alignment_stats["exact_sign_flip_interpretation"] = (
        "country_level_diagnostic_conditional_on_overlapping_cross_fitted_directions"
    )
    alignment_stats["conditional_country_bootstrap"] = {
        "ci_low": alignment_stats["ci_low"],
        "ci_high": alignment_stats["ci_high"],
        "interpretation": (
            "descriptive_only_scores_share_overlapping_training_countries"
        ),
    }
    primary_position_index = positions.index(primary_position)
    primary_alignment_offsets = [
        offset
        for offset, layer_index in enumerate(analysis_layer_indices)
        if layers[layer_index] in primary_layers
    ]
    outer_refit: list[dict[str, Any]] = []
    if len(country_ids) >= 3:
        for excluded_country_id in country_ids:
            retained_country_ids = [
                country_id
                for country_id in country_ids
                if country_id != excluded_country_id
            ]
            refitted_country_scores: list[float] = []
            invalid_reason: str | None = None
            for heldout_country_id in retained_country_ids:
                refit_training_ids = [
                    country_id
                    for country_id in retained_country_ids
                    if country_id != heldout_country_id
                ]
                try:
                    refitted_alignment, _ = cross_fitted_alignment(
                        heldout_country_id, refit_training_ids
                    )
                except AnalysisError as exc:
                    invalid_reason = str(exc)
                    break
                refitted_country_scores.append(
                    float(
                        np.mean(
                            refitted_alignment[
                                primary_alignment_offsets,
                                primary_position_index,
                            ]
                        )
                    )
                )
            outer_refit.append(
                {
                    "excluded_country_id": excluded_country_id,
                    "refitted_heldout_country_count": len(
                        refitted_country_scores
                    ),
                    "mean": (
                        float(np.mean(refitted_country_scores))
                        if invalid_reason is None
                        else None
                    ),
                    "valid": invalid_reason is None,
                    "invalid_reason": invalid_reason,
                }
            )
    valid_outer_refit_means = [
        item["mean"]
        for item in outer_refit
        if item["valid"] and item["mean"] is not None
    ]
    alignment_stats["outer_refit_leave_one_country_out"] = outer_refit
    alignment_stats["outer_refit_leave_one_country_out_minimum"] = (
        min(valid_outer_refit_means) if valid_outer_refit_means else None
    )
    alignment_stats[
        "all_outer_refit_leave_one_country_out_means_positive"
    ] = bool(
        outer_refit
        and len(valid_outer_refit_means) == len(outer_refit)
        and all(mean > 0.0 for mean in valid_outer_refit_means)
    )
    alignment_stats["prediction_supported"] = bool(
        alignment_stats["mean"] > 0.0
        and alignment_stats[
            "all_outer_refit_leave_one_country_out_means_positive"
        ]
    )
    alignment_stats["support_criterion"] = (
        "positive_grand_mean_and_every_outer_country_deletion_refit_mean_positive"
    )
    primary["alignment_G"] = alignment_stats
    primary["all_three_endpoints_supported"] = bool(
        primary["families"]["G"]["prediction_supported"]
        and primary["families"]["V"]["prediction_supported"]
        and primary["alignment_G"]["prediction_supported"]
    )
    # Compatibility alias for the first schema draft; this is stricter than
    # mere positivity because every endpoint also has to pass its CI and LOO
    # criteria.
    primary["all_three_predictions_positive"] = primary[
        "all_three_endpoints_supported"
    ]
    primary["support_rule"] = {
        "supported": primary["all_three_endpoints_supported"],
        "label": (
            "supported"
            if primary["all_three_endpoints_supported"]
            else "not_supported"
        ),
        "requires": [
            "G_and_V_grand_mean_above_zero",
            "G_and_V_95pct_country_bootstrap_lower_bound_above_zero",
            "G_and_V_every_leave_one_country_out_mean_above_zero",
            "alignment_grand_mean_above_zero",
            "alignment_every_outer_country_deletion_refit_mean_above_zero",
        ],
    }

    profiles: dict[str, Any] = {"indices": {}, "alignment_G": {}}
    for family in FAMILIES:
        profiles["indices"][family] = {}
        family_rows = [
            row for row in index_rows if row["family"] == family
        ]
        for position in positions:
            profiles["indices"][family][position] = {}
            position_rows = [
                row
                for row in family_rows
                if row["position"] == position
            ]
            for band, band_layers in protocol["design"][
                "layer_bands"
            ].items():
                profiles["indices"][family][position][band] = _profile_stats(
                    position_rows,
                    value_field="normalized_index",
                    countries=country_ids,
                    layers=set(band_layers),
                    seed=seed,
                    samples=samples,
                )
    for position in positions:
        profiles["alignment_G"][position] = {}
        position_rows = [
            row
            for row in alignment_rows
            if row["position"] == position
        ]
        for band, band_layers in protocol["design"]["layer_bands"].items():
            profiles["alignment_G"][position][band] = _profile_stats(
                position_rows,
                value_field="cosine_alignment",
                countries=country_ids,
                layers=set(band_layers),
                seed=seed,
                samples=samples,
            )

    summary = {
        "format": f"{FORMAT}.summary",
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "n_stimuli": len(stimuli),
        "n_countries": len(country_ids),
        "country_ids": country_ids,
        "positions": positions,
        "primary": primary,
        "profiles": profiles,
        "interpretation_limits": [
            "No preregistered inverted-U prediction.",
            "Generative relation is confounded with expected answer identity and answer class.",
            "Verification controls the immediate truth label but measures truth assessment.",
            "Residual cosine geometry is representational invariance, not localized meaning.",
            "L0 is a sanity check and is outside the primary window.",
            "Countries are a fixed purposive set; bootstrap describes set stability, not population inference.",
            "The strongest wording is stability of relation-associated geometry across locked paraphrases.",
        ],
    }
    return {
        "norm_rows": norm_rows,
        "pairwise_rows": pairwise_rows,
        "index_rows": index_rows,
        "alignment_rows": alignment_rows,
        "summary": summary,
    }
