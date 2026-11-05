"""Prospective paired steering measurements, with whole-path cross-model units."""
from __future__ import annotations

import math
from typing import Any

PROTOCOL = {
    "id": "steering-comparison-v1",
    "temperature": 1.0,
    "chat_template": False,
    "normalization": "full_native_vocabulary",
    "direction": "separate_per_model_from_identical_contrast_texts",
    "aggregation": "normalized_pair_mean_with_median_native_scale",
    "hook": "post_decoder_block_residual_last_position_at_each_scored_step",
    "layer_selection": "ceil(relative_depth * native_decoder_layer_count)",
    "alpha_scale": "native_median_contrast_pair_norm; not_equal_absolute_or_prefix_relative_norm",
    "branches": ["negative", "baseline", "positive"],
    "metric": "whole_canonical_continuation_log_odds_bits",
    "contrast": "C(alpha)=log2(P(positive_path|prefix,alpha)/P(negative_path|prefix,alpha))",
    "effect": "0.5*(C(+alpha)-C(-alpha))",
    "paired_delta": "effect_model_b_minus_effect_model_a",
    "word_boundary_correction": False,
    "teacher_forced": True,
    "controls": "per_model_post_cast_norm_matched_orthogonal_isotropic_directions",
    "control_interpretation": "descriptive_reference_not_exact_randomization_p",
    "interpretation": "Causal intervention in these model forwards; not a human reading-time or concept-localization claim.",
}
MAX_ITEMS = 12


def validate_options(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("steering musí obsahovat nastavení zásahu.")
    pairs = value.get("contrast_pairs")
    if not isinstance(pairs, list) or not 3 <= len(pairs) <= 8:
        raise ValueError("Steering vyžaduje 3 až 8 kontrastních párů.")
    checked = []
    seen = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("Kontrastní pár musí být objekt se dvěma texty.")
        texts = []
        for key in ("negative_text", "positive_text"):
            text = pair.get(key)
            if not isinstance(text, str) or not text.strip() or len(text) > 8192:
                raise ValueError("Obě strany kontrastního páru musí obsahovat neprázdný text.")
            texts.append(text.rstrip())
        key = tuple(sorted(texts))
        if texts[0] == texts[1] or key in seen:
            raise ValueError("Kontrastní páry musí být různé, bez duplicity a obrácených kopií.")
        seen.add(key)
        checked.append(dict(zip(("negative_text", "positive_text"), texts, strict=True)))
    depth = value.get("relative_depth", .5)
    alpha = value.get("alpha_magnitude", 1.0)
    for name, number, low, high in (("relative_depth", depth, 0.0, 1.0), ("alpha_magnitude", alpha, 0.0, 2.0)):
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not low < number <= high:
            raise ValueError(f"{name} musí být větší než {low:g} a nejvýše {high:g}.")
    samples = value.get("sham_samples", 9)
    seed = value.get("seed", 20261008)
    if type(samples) is not int or not 2 <= samples <= 99:
        raise ValueError("Počet kontrolních směrů musí být celé číslo mezi 2 a 99.")
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed musí být uint32.")
    return {"contrast_pairs": checked, "relative_depth": float(depth), "alpha_magnitude": float(alpha),
            "sham_samples": samples, "seed": seed}


def validate_held_out_item(item: dict[str, Any], options: dict[str, Any]) -> None:
    training_texts = {text for pair in options["contrast_pairs"] for text in pair.values()}
    if item["negative_text"] == item["positive_text"]:
        raise ValueError("Testovací pokračování musí být různá.")
    candidates = [item["test_text"], item["test_text"] + item["negative_text"],
                  item["test_text"] + item["positive_text"]]
    if any(text.rstrip() in training_texts for text in candidates):
        raise ValueError("Testovací prompt a jeho pokračování musí být oddělené od konstrukčních příkladů.")


def resolve_layer(relative_depth: float, layer_count: int) -> int:
    if type(layer_count) is not int or layer_count < 1:
        raise ValueError("Model nemá známý počet decoderových vrstev.")
    return max(1, math.ceil(relative_depth * layer_count))


def path_contrasts(score: dict[str, Any]) -> dict[str, float]:
    """Use total log probabilities, never average per differently sized token."""
    pairs = score["per_pair"]
    if not pairs:
        raise ValueError("Chybí měření testovacích pokračování.")
    result = {}
    for branch in ("negative", "baseline", "positive"):
        differences = []
        for pair in pairs:
            positive = pair["positive"]["total_logprobs"][branch]
            negative = pair["negative"]["total_logprobs"][branch]
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                       for v in (positive, negative)):
                raise ValueError("Skóre celé tokenové cesty musí být konečné.")
            differences.append((positive - negative) / math.log(2.0))
        result[branch] = math.fsum(differences) / len(differences)
    return result


def summarize_measurement(response: dict[str, Any], layer_count: int) -> dict[str, Any]:
    target = response["target"]
    contrasts = path_contrasts(target)
    effect = .5 * (contrasts["positive"] - contrasts["negative"])
    shams = [path_contrasts(sham) for sham in response["shams"]]
    sham_effects = [.5 * (row["positive"] - row["negative"]) for row in shams]
    return {
        "layer": response["layer"], "layer_count": layer_count,
        "realized_relative_depth": response["layer"] / layer_count,
        "contrast_bits": contrasts,
        "effect_bits": effect,
        "positive_minus_baseline_bits": contrasts["positive"] - contrasts["baseline"],
        "baseline_minus_negative_bits": contrasts["baseline"] - contrasts["negative"],
        "strictly_increasing": contrasts["negative"] < contrasts["baseline"] < contrasts["positive"],
        "direction_sha256": target["direction"]["sha256"],
        "native_scale": target["native_scale"],
        "effective_shift_norm": target["effective_shift_norm"],
        "prefix_relative_shift_norm": target["prefix_relative_shift_norm"],
        "sham_effect_bits": sham_effects,
        "sham_mean_effect_bits": math.fsum(sham_effects) / len(sham_effects),
        "sham_count_at_or_above_target": sum(value >= effect for value in sham_effects),
        "sham_sample_count": len(sham_effects),
        "sham_reference_only": True,
        "all_effective_norm_matches_passed": response["randomization"]["all_effective_norm_matches_passed"],
        "estimand": PROTOCOL["effect"],
    }


def paired_item(item: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    left, right = a["summary"], b["summary"]
    effects = {"a": left["effect_bits"], "b": right["effect_bits"]}
    return {"id": item["id"], "kind": "steering", "model_a": a["model_id"], "model_b": b["model_id"],
            "steering_effect_bits": effects, "delta_steering_effect_b_minus_a_bits": effects["b"] - effects["a"],
            "same_effect_direction": (effects["a"] > 0) == (effects["b"] > 0) if all(v != 0 for v in effects.values()) else None,
            "a": left, "b": right}
