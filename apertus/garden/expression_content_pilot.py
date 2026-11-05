"""Exploratory input wording / truth / output vocabulary pilot.

The balanced design and offline analysis do not import MLX.  In particular,
output-vocabulary geometry is not identified with Hjelmslev's content plane.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

import run_task_form_battery as parent

AXES = ("country", "relation", "truth", "input_form", "output_code")
CODES = {"cs": {"true": "ano", "false": "ne"},
         "en": {"true": "true", "false": "false"}}
FORMS = ("v1_predicate", "v2_answer_front")
DEPTHS = {"hybrid": 32, "eurollm9": 42}


def build_protocol(source: dict) -> dict:
    """All ten parent countries and locked false donors; no outcome selection."""
    source_items = parent.expand_protocol(source)
    stimuli = []
    for item in source_items:
        if item["family"] != "verification" or item["form_id"] not in FORMS:
            continue
        statement = item["statement"]
        for code, labels in CODES.items():
            prompt = (
                "Rozhodni o pravdivosti výroku. Použij stejný způsob odpovědi jako v příkladech.\n"
                f"Výrok: Dva plus dva jsou čtyři.\nOdpověď: {labels['true']}\n"
                f"Výrok: Dva plus dva jsou pět.\nOdpověď: {labels['false']}\n"
                f"Výrok: {statement}.\nOdpověď:"
            )
            stimuli.append({"stimulus_id": f"{item['stimulus_id']}__{code}",
                            "parent_stimulus_id": item["stimulus_id"],
                            "country": item["country_id"], "relation": item["relation_id"],
                            "truth": item["truth_value"], "input_form": item["form_id"],
                            "output_code": code, "statement": statement, "prompt": prompt,
                            "expected_output": labels[item["truth_value"]]})
    axes = {"country": [x["country_id"] for x in source["countries"]],
            "relation": list(source["relations"]), "truth": ["true", "false"],
            "input_form": list(FORMS), "output_code": list(CODES)}
    if len(stimuli) != 160 or len({tuple(x[a] for a in AXES) for x in stimuli}) != 160:
        raise ValueError("Pilot requires the complete balanced 10×2×2×2×2 design")
    return {"format": "apertus.expression-content-output-pilot", "schema_version": 1,
            "status": "exploratory_new_prompts_after_inspection_of_parent_profiles",
            "axes": axes, "stimuli": stimuli, "model_depths": DEPTHS,
            "position": "last prefix token; identical response-boundary token within tokenizer",
            "parent_protocol_sha256": parent.sha256_bytes(parent.canonical_json_bytes(source)),
            "output_labels": CODES,
            "plan": {
                "bands": {"early": "0 < l/N <= .25", "middle": ".25 < l/N <= .75", "late": ".75 < l/N <= 1"},
                "geometry": "unit residual vectors; complete five-factor orthogonal ANOVA; L1..LN",
                "truth_transfer": "train true-minus-false direction on nine countries, one input form and one output code; test on omitted country, other input form and other output code; average four ordered F×O transfers per country; descriptive cosine, no IID confidence interval",
                "readout": "native final norm/head RAW logit lens; intermediate states batched; last row replaced by native full-prefix projection verified against independent forward",
                "lexicalization_proxy": "full-vocabulary probability mass of the prescribed pair of single-token labels and selected-vs-other-code log odds; sensitive to uncalibrated early RAW readout",
                "adequacy_floor": ".75 within-code final truth accuracy and .75 final four-label code accuracy in EACH output code; descriptive floor only, no item exclusion",
                "interpretation": "No causal or confirmatory hourglass claim; input wording, output code cue and lexical realization are distinct. English labels also change output language. Truth and output-token identity remain coupled within a code; cross-code transfer is an imperfect control.",
                "statistics": "fixed purposive ten-country set, all items retained, no population p-values, no peak-selected primary window",
            }}


def tokenize(protocol: dict, tokenizer) -> dict:
    result = {}
    label_paths, boundaries = {}, set()
    for item in protocol["stimuli"]:
        encoding = tokenizer.encode(item["prompt"], add_special_tokens=True)
        ids = list(encoding.ids)
        if tokenizer.decode(ids, skip_special_tokens=True) != item["prompt"]:
            raise ValueError("Prefix is not losslessly tokenized")
        labels = {}
        for code, mapping in CODES.items():
            for truth, label in mapping.items():
                complete = tokenizer.encode(item["prompt"] + " " + label, add_special_tokens=True)
                if list(complete.ids[:len(ids)]) != ids:
                    raise ValueError("Label retokenizes the prefix")
                tail = list(complete.ids[len(ids):])
                if len(tail) != 1:
                    raise ValueError("All four output labels must be single tokens")
                previous = label_paths.setdefault(label, tail[0])
                if previous != tail[0]:
                    raise ValueError("Label path varies across conditions")
                labels[f"{code}_{truth}"] = tail[0]
        if len(set(labels.values())) != 4:
            raise ValueError("The two vocabularies must have disjoint label tokens")
        boundaries.add(ids[-1])
        result[item["stimulus_id"]] = {"ids": ids, "tokens": encoding.tokens,
            "boundary_index": len(ids)-1, "boundary_token_id": ids[-1], "label_token_ids": labels}
    if len(boundaries) != 1:
        raise ValueError("Response-boundary token differs across conditions")
    return result


def decompose(values: np.ndarray) -> dict:
    """General balanced multivariate ANOVA with a reconstruction invariant."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 6 or any(s < 2 for s in x.shape[:-1]) or not np.isfinite(x).all():
        raise ValueError("Expected finite C×R×T×F×O×hidden tensor")
    effects = {(): x.mean(axis=tuple(range(5)), keepdims=True)}
    sums = {}
    for size in range(1, 6):
        for subset in itertools.combinations(range(5), size):
            off = tuple(a for a in range(5) if a not in subset)
            effect = x.mean(axis=off, keepdims=True) if off else x.copy()
            for previous, component in effects.items():
                if set(previous).issubset(subset):
                    effect = effect - component
            effects[subset] = effect
            sums["x".join(AXES[a] for a in subset)] = float(np.sum(effect**2) * np.prod([x.shape[a] for a in off], dtype=np.int64))
    total = float(np.sum((x-effects[()])**2))
    if total <= 0 or not np.isclose(sum(sums.values()), total, rtol=1e-10, atol=1e-10):
        raise ValueError("Factor decomposition did not reconstruct total variance")
    return {"total_ss": total, "ss": sums, "share": {k: v/total for k,v in sums.items()}}


def ordered_tensor(protocol: dict, items: list[dict], values: dict) -> np.ndarray:
    expected = {tuple(x) for x in itertools.product(*(protocol["axes"][a] for a in AXES))}
    keys = [tuple(item[a] for a in AXES) for item in items]
    if len(keys) != len(set(keys)) or set(keys) != expected or set(values) != {x["stimulus_id"] for x in items}:
        raise ValueError("Missing, extra, or duplicate factorial cells")
    by_key = dict(zip(keys, items))
    ordered = [values[by_key[key]["stimulus_id"]] for key in itertools.product(*(protocol["axes"][a] for a in AXES))]
    array = np.asarray(ordered, dtype=np.float64)
    shape = [len(protocol["axes"][a]) for a in AXES]
    return array.reshape(*shape, *array.shape[1:])


def truth_transfer(unit_tensor: np.ndarray) -> list[dict]:
    """Cross both surface factors and omit the evaluated country from training."""
    c_count, r_count, t_count, f_count, o_count, _ = unit_tensor.shape
    if (t_count, f_count, o_count) != (2,2,2) or c_count < 3:
        raise ValueError("Unexpected truth-transfer design")
    rows = []
    for c in range(c_count):
        train = unit_tensor[np.arange(c_count) != c]
        for form, code in itertools.product(range(2), repeat=2):
            train_direction = (train[:, :, 0, form, code] - train[:, :, 1, form, code]).mean(axis=(0,1))
            test_direction = (unit_tensor[c, :, 0, 1-form, 1-code] - unit_tensor[c, :, 1, 1-form, 1-code]).mean(axis=0)
            denominator = np.linalg.norm(train_direction)*np.linalg.norm(test_direction)
            cosine = float(np.dot(train_direction,test_direction)/denominator) if denominator > 0 else None
            rows.append({"country_index": c, "train_form_index": form, "train_code_index": code, "cosine": cosine})
    return rows


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(maximum,axis=axis) + np.log(np.exp(x-maximum).sum(axis=axis))


def readout_rows(item: dict, label_ids: dict, logits: np.ndarray) -> list[dict]:
    """Full-vocabulary probabilities and controlled four-label decisions."""
    names = [f"{code}_{truth}" for code in CODES for truth in ("true", "false")]
    ids = [label_ids[name] for name in names]
    selected_indices = [i for i,name in enumerate(names) if name.startswith(item["output_code"]+"_")]
    other_indices = [i for i in range(4) if i not in selected_indices]
    correct_index = names.index(f"{item['output_code']}_{item['truth']}")
    wrong_index = names.index(f"{item['output_code']}_{'false' if item['truth']=='true' else 'true'}")
    labels = logits[:, ids].astype(np.float64)
    log_z = logsumexp(logits.astype(np.float64))
    selected_mass = logsumexp(labels[:, selected_indices])
    other_mass = logsumexp(labels[:, other_indices])
    top_indices = labels.argmax(axis=-1)
    result=[]
    for layer in range(len(logits)):
        result.append({"stimulus_id": item["stimulus_id"], **{a:item[a] for a in AXES}, "layer": layer,
            "selected_code_probability": float(np.exp(selected_mass[layer]-log_z[layer])),
            "expected_token_probability": float(np.exp(labels[layer,correct_index]-log_z[layer])),
            "selected_code_log_odds": float(selected_mass[layer]-other_mass[layer]),
            "truth_log_odds": float(labels[layer,correct_index]-labels[layer,wrong_index]),
            "within_code_truth_correct": bool(labels[layer,correct_index]>labels[layer,wrong_index]),
            "four_label_code_correct": bool(top_indices[layer] in selected_indices),
            "full_vocab_top1_correct": bool(np.argmax(logits[layer])==ids[correct_index]),
            "expected_token_rank": int(1+np.sum(logits[layer]>labels[layer,correct_index]))})
    return result


def csv_bytes(rows: list[dict]) -> bytes:
    import csv
    import io
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def analyze(protocol: dict, items: list[dict], states: dict, readouts: list[dict], depth: int) -> dict:
    profiles, factors, transfers = [], [], []
    tensor = ordered_tensor(protocol, items, states)
    if tensor.shape[-2:] != (depth+1,4096):
        raise ValueError("Unexpected native layer/hidden shape")
    for layer in range(1,depth+1):
        x=tensor[...,layer,:]
        norms=np.linalg.norm(x,axis=-1,keepdims=True)
        if np.any(norms<=0): raise ValueError("Zero state")
        x=x/norms
        decomposition=decompose(x)
        shares=decomposition["share"]
        for key,ss in decomposition["ss"].items():
            factors.append({"layer":layer,"relative_depth":layer/depth,"component":key,"ss":ss,
                            "total_ss":decomposition["total_ss"],"share":shares[key]})
        transfer=truth_transfer(x)
        for row in transfer:
            transfers.append({"layer":layer,"country":protocol["axes"]["country"][row["country_index"]],**row})
        cosines=[row["cosine"] for row in transfer if row["cosine"] is not None]
        rs=[row for row in readouts if row["layer"]==layer]
        if len(rs)!=160: raise ValueError("Readouts are not complete")
        profiles.append({"layer":layer,"relative_depth":layer/depth,
            "input_form_main_share":shares["input_form"],
            "input_form_any_share":sum(v for k,v in shares.items() if "input_form" in k.split("x")),
            "output_code_main_share":shares["output_code"],
            "output_code_any_share":sum(v for k,v in shares.items() if "output_code" in k.split("x")),
            "truth_main_share":shares["truth"],"relation_main_share":shares["relation"],
            "content_core_share":shares["relation"]+shares["truth"]+shares["relationxtruth"],
            "truth_output_interaction_share":shares["truthxoutput_code"],
            "truth_crossfit_cosine":float(np.mean(cosines)) if cosines else None,
            **{key:float(np.mean([row[key] for row in rs])) for key in (
                "selected_code_probability","expected_token_probability","selected_code_log_odds",
                "within_code_truth_correct","four_label_code_correct","full_vocab_top1_correct")}})
    adequacy={}
    for code in CODES:
        rs=[row for row in readouts if row["layer"]==depth and row["output_code"]==code]
        scores={key:float(np.mean([r[key] for r in rs])) for key in (
            "within_code_truth_correct","four_label_code_correct","full_vocab_top1_correct")}
        adequacy[code]={"n":len(rs),**scores,"floor_met":scores["within_code_truth_correct"]>=.75 and scores["four_label_code_correct"]>=.75}
    bands={}
    for band,low,high in (("early",0,.25),("middle",.25,.75),("late",.75,1)):
        rs=[r for r in profiles if low<r["relative_depth"]<=high]
        bands[band]={"layers":[r["layer"] for r in rs],**{
            key:float(np.mean([r[key] for r in rs if r[key] is not None])) if any(r[key] is not None for r in rs) else None
            for key in profiles[0] if key not in ("layer","relative_depth")}}
    return {"profiles":profiles,"factor_rows":factors,"transfer_rows":transfers,
            "summary":{"adequacy":adequacy,"all_codes_meet_floor":all(r["floor_met"] for r in adequacy.values()),
                       "bands":bands,"all_items_retained":True,"causal_test":False}}
