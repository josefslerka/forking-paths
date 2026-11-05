from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

GARDEN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GARDEN))

import run_task_form_battery as parent
import run_task_form_transfer as collector
import task_form_transfer as transfer
import task_form_analysis as analysis
import task_form_variance as variance
from derive_task_form_transfer import country_block_interval, paired_comparison, align_collection_order


def test_unchanged_stimuli_and_prospective_relative_window():
    source = parent.load_json(parent.DEFAULT_PROTOCOL)
    before = copy.deepcopy(source)
    expected = parent.expand_protocol(source)
    for model_id, depth, window in (("hybrid", 32, list(range(9, 25))), ("eurollm9", 42, list(range(11, 32)))):
        protocol = transfer.build_protocol(source, model_id)
        transfer.validate_protocol(protocol)
        assert transfer.expand_protocol(protocol) == expected
        assert len(protocol["design"]["layers"]) == depth + 1
        assert protocol["design"]["primary_layers"] == window
        protocol["design"]["primary_layers"].append(window[-1] + 1)
        with pytest.raises(parent.BatteryError):transfer.validate_protocol(protocol)
    assert source == before


def test_transfer_rejects_changed_text_or_endpoint():
    source = parent.load_json(parent.DEFAULT_PROTOCOL)
    protocol = transfer.build_protocol(source, "eurollm9")
    protocol["preregistered_analysis"]["primary_position"] = "country_end"
    with pytest.raises(parent.BatteryError):transfer.validate_protocol(protocol)
    protocol = transfer.build_protocol(source, "eurollm9")
    protocol["relations"]["capital"]["generative_forms"]["g1_predicate"] += "."
    with pytest.raises(parent.BatteryError):transfer.validate_protocol(protocol)


def test_analysis_and_factorial_map_use_all_42_layers():
    protocol = transfer.build_protocol(parent.load_json(parent.DEFAULT_PROTOCOL), "eurollm9")
    stimuli = transfer.expand_protocol(protocol)
    countries = {country["country_id"]: index for index,country in enumerate(protocol["countries"])}
    forms = {"generative": ["g1_predicate", "g2_question", "g3_lexical_holdout"],
             "verification": ["v1_predicate", "v2_answer_front", "v3_paraphrase"]}
    states = {}
    for item in stimuli:
        vector = np.zeros(18)
        vector[0] = 2
        vector[1] = 2 if item["relation_id"] == "capital" else -2
        vector[2 + forms[item["family"]].index(item["form_id"])] = .08
        vector[5 + countries[item["country_id"]]] = .2
        vector[15] = .15 if item["truth_value"] == "true" else 0
        vector[16] = .15 if item["truth_value"] == "false" else 0
        states[item["stimulus_id"]] = np.broadcast_to(vector, (43, 2, 18)).copy()
    result = analysis.analyze_states(protocol, stimuli, states)
    assert result["summary"]["primary"]["layers"] == list(range(11, 32))
    assert result["summary"]["primary"]["families"]["G"]["mean"] > 0
    assert "L11_L31" in result["summary"]["primary"]["families"]["V"]["metric"]
    assert {row["layer"] for row in result["index_rows"]} == set(range(1, 43))
    structural = variance.analyze_states(protocol, stimuli, states)
    assert {row["layer"] for row in structural["variance_rows"]} == set(range(1, 43))


def test_truth_strata_are_one_bootstrap_block_per_country():
    rows = [{"country_id":"a", "normalized_index":-10}, {"country_id":"a", "normalized_index":10},
            {"country_id":"b", "normalized_index":-20}, {"country_id":"b", "normalized_index":20}]
    stats = country_block_interval(rows, seed=12, samples=5000)
    assert stats["n_countries"] == 2
    assert stats["mean"] == stats["ci_low"] == stats["ci_high"] == 0


def test_sealed_collection_order_survives_json_key_sorting():
    source = parent.load_json(parent.DEFAULT_PROTOCOL)
    protocol = transfer.build_protocol(source, "eurollm9")
    before = transfer.expand_protocol(protocol)
    after = transfer.expand_protocol(json.loads(json.dumps(protocol, sort_keys=True)))
    ids = [item["stimulus_id"] for item in before]
    assert align_collection_order(ids, after) == before
    for invalid in (ids[:-1], ids + [ids[0]], [ids[0]] * len(ids)):
        with pytest.raises(parent.BatteryError):align_collection_order(invalid, after)


def test_model_difference_pairs_ids_and_refits():
    summaries = {}
    for model, offset in (("hybrid",0), ("eurollm9",1)):
        order = ["a", "b", "c"] if model == "hybrid" else ["c", "b", "a"]
        values = [{"country_id": key,"mean": ord(key) + offset} for key in order]
        summaries[model] = {"primary": {"families": {label:{"country_values":values} for label in ("G","V")},
            "alignment_G": {"mean":.2 + offset / 10,
                "outer_refit_leave_one_country_out":[{"excluded_country_id":key,"mean":.1 + offset / 10} for key in order]}}}
    result = paired_comparison(summaries, seed=12, samples=5000)
    for label in ("G","V"):
        assert result["families"][label]["mean"] == result["families"][label]["ci_low"] == result["families"][label]["ci_high"] == 1
    assert result["alignment_G"]["mean_difference"] == pytest.approx(.1)
    assert "ci_low" not in result["alignment_G"]


def test_manifest_rejects_tamper_and_extra_files(tmp_path):
    parent.atomic_write_json(tmp_path / "item.json", {"value":1})
    parent.atomic_write_json(tmp_path / "manifest.json", {"format":"apertus.task-form-transfer-raw","status":"complete","files":parent.inventory_files(tmp_path)})
    collector.verify_inventory(tmp_path)
    parent.atomic_write_json(tmp_path / "item.json", {"value":2})
    with pytest.raises(parent.BatteryError):collector.verify_inventory(tmp_path)
    parent.atomic_write_json(tmp_path / "item.json", {"value":1})
    (tmp_path / "extra.txt").write_text("unexpected")
    with pytest.raises(parent.BatteryError):collector.verify_inventory(tmp_path)


def test_validate_only_has_no_mlx_or_model_side_effects(tmp_path):
    (tmp_path / "mlx.py").write_text('raise RuntimeError("MLX must not be imported")\n')
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    output = tmp_path / "validation"
    result = subprocess.run([sys.executable,str(GARDEN / "run_task_form_transfer.py"),"--validate-only","--output",str(output)],env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["stimuli_per_model"] == 180 and data["mlx_or_model_loaded"] is False
    assert not output.exists()


def test_invalid_positions_fail_before_mlx_import():
    from apertus.garden.domains.representations import trace_positions
    for ids,positions in (([],[0]),([1,2],[]),([1,2],[-1]),([1,2],[2]),([1,2],[1,0]),([1,2],[0,0])):
        with pytest.raises(ValueError):trace_positions(None,ids,positions)
