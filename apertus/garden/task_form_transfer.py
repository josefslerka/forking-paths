"""A prospective model transfer of the unchanged 180-item task/form design.

The original Apertus protocol remains immutable. Only the model identity and
layer coordinates are adapted, with the complete parent design embedded for
offline validation. No MLX imports belong in this module.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

import run_task_form_battery as parent

FORMAT = "apertus.task-vs-form-transfer-protocol"
MODEL_DEPTHS = {"hybrid": 32, "eurollm9": 42}
WINDOW_RULE = "0.25 < layer / decoder_layer_count <= 0.75"


def central_layers(depth: int) -> list[int]:
    if type(depth) is not int or depth < 4:
        raise parent.BatteryError("Počet vrstev musí být celé číslo alespoň 4.")
    return [layer for layer in range(1, depth + 1) if depth < 4 * layer <= 3 * depth]


def build_protocol(source: dict[str, Any], model_id: str) -> dict[str, Any]:
    parent.validate_protocol(source)
    if model_id not in MODEL_DEPTHS:
        raise parent.BatteryError(f"Nepodporovaný model přenosu: {model_id}")
    depth = MODEL_DEPTHS[model_id]
    middle = central_layers(depth)
    window = f"L{middle[0]}-L{middle[-1]}"
    protocol = copy.deepcopy(source)
    protocol.update(format=FORMAT, schema_version=1,
                    protocol_id=f"task-vs-form-transfer-20261009-{model_id}-v1")
    design = protocol["design"]
    design.update(required_primary_model_mode=model_id, allowed_model_modes=[model_id],
                  layers=list(range(depth + 1)), primary_layers=middle,
                  layer_bands={"early": list(range(1, middle[0])), "middle": middle,
                               "late": list(range(middle[-1] + 1, depth + 1))},
                  primary_window_rationale=f"{WINDOW_RULE}; fixed before EuroLLM residual inspection.")
    design["baseline"]["analysis_scope"] = f"L1-L{depth}; L0 is an embedding sanity check only"
    design["factorial_decomposition"]["analysis_scope"] = (
        f"L1-L{depth} at both positions; descriptive primary map is {window} at response_boundary")
    protocol["preregistered_analysis"]["primary_window"] = f"mean across {window}"
    protocol["preregistered_analysis"]["aggregation_order"] = (
        "Compute J per country/layer/(truth); average true and false equally within verification; "
        f"average {window} within country; only then summarize country values.")
    protocol["transfer"] = {
        "model_id": model_id, "decoder_layer_count": depth, "relative_window_rule": WINDOW_RULE,
        "source_protocol": copy.deepcopy(source),
        "source_canonical_sha256": hashlib.sha256(parent.canonical_json_bytes(source)).hexdigest(),
        "stimuli_and_endpoint_definitions": "unchanged_from_parent",
        "status": "prospective_EuroLLM_transfer_and_exploratory_cross_model_comparison",
        "prior_knowledge": "Apertus results on these stimuli were already inspected; no independent stimulus replication claim",
    }
    return protocol


def validate_protocol(protocol: dict[str, Any]) -> None:
    try:
        transfer = protocol["transfer"]
        expected = build_protocol(transfer["source_protocol"], transfer["model_id"])
    except (KeyError, TypeError) as exc:
        raise parent.BatteryError("Přenosový protokol postrádá původ a identitu modelu.") from exc
    if protocol != expected:
        raise parent.BatteryError("Přenos mění uzamčený design, metriky nebo relativní okno.")


def expand_protocol(protocol: dict[str, Any], country_ids=None) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    return parent.expand_protocol(protocol["transfer"]["source_protocol"], country_ids)
