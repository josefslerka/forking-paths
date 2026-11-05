"""Offline analysis and Czech report for sealed task/form model transfers."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

import run_task_form_battery as parent
import run_task_form_transfer as collection
import task_form_analysis as analysis
import task_form_transfer as transfer
import task_form_variance as variance
import derive_task_form_battery as original_derivation


def country_block_interval(rows: list[dict], seed: int, samples: int) -> dict:
    """Average truth strata inside each country before resampling countries."""
    blocks: dict[str, list[float]] = {}
    for row in rows:
        blocks.setdefault(row["country_id"], []).append(float(row["normalized_index"]))
    return analysis.country_bootstrap([float(np.mean(values)) for values in blocks.values()], seed=seed, samples=samples)


def align_collection_order(stimulus_ids: list[str], expanded: list[dict]) -> list[dict]:
    """JSON object order is not a condition; preserve the sealed collection list."""
    by_id = {item["stimulus_id"]: item for item in expanded}
    if len(stimulus_ids) != len(expanded) or len(set(stimulus_ids)) != len(stimulus_ids) or set(stimulus_ids) != set(by_id):
        raise parent.BatteryError("Chybějící, nadbytečný nebo duplicitní stimul v raw manifestu.")
    return [by_id[stimulus_id] for stimulus_id in stimulus_ids]


def paired_comparison(summaries: dict, seed: int, samples: int) -> dict:
    """Pair country blocks; do not treat cross-fitted alignment scores as IID."""
    left, right = (summaries[key]["primary"] for key in collection.MODEL_IDS)
    result = {"direction": "EuroLLM minus Apertus", "status": "exploratory_fixed_country_set", "families": {}}
    for label in ("G", "V"):
        a = {item["country_id"]: item["mean"] for item in left["families"][label]["country_values"]}
        b = {item["country_id"]: item["mean"] for item in right["families"][label]["country_values"]}
        if set(a) != set(b):
            raise parent.BatteryError("Párové porovnání nemá stejné země.")
        differences = [{"country_id": key, "difference": b[key] - a[key]} for key in a]
        stats = analysis.country_bootstrap([item["difference"] for item in differences], seed=seed, samples=samples)
        stats.update(country_differences=differences, interpretation="paired fixed-country-set stability; not population inference")
        result["families"][label] = stats
    a_outer = {item["excluded_country_id"]: item["mean"] for item in left["alignment_G"]["outer_refit_leave_one_country_out"]}
    b_outer = {item["excluded_country_id"]: item["mean"] for item in right["alignment_G"]["outer_refit_leave_one_country_out"]}
    if set(a_outer) != set(b_outer) or any(value is None for value in [*a_outer.values(), *b_outer.values()]):
        raise parent.BatteryError("Neúplné vnější refity alignmentu.")
    result["alignment_G"] = {
        "mean_difference": right["alignment_G"]["mean"] - left["alignment_G"]["mean"],
        "outer_country_deletion_refit_differences": [{"excluded_country_id": key, "difference": b_outer[key] - a_outer[key]} for key in a_outer],
        "interpretation": "descriptive paired outer-country-deletion refits; no naive bootstrap or significance test of overlapping scores",
    }
    return result


def layer_profiles(model_id: str, protocol: dict, result: dict, structural: dict) -> list[dict]:
    depth = transfer.MODEL_DEPTHS[model_id]
    profiles = []
    for position in parent.POSITION_ORDER:
        for layer in range(1, depth + 1):
            for label, family in (("G", "generative"), ("V", "verification")):
                rows = [row for row in result["index_rows"] if row["family"] == family and row["position"] == position and row["layer"] == layer]
                stats = country_block_interval(rows, seed=protocol["design"]["bootstrap_seed"], samples=protocol["design"]["bootstrap_samples"])
                profiles.append({"model_id": model_id, "family": family, "position": position, "layer": layer,
                                 "relative_depth": layer / depth, "metric": "normalized_index", "mean": stats["mean"],
                                 "ci_low": stats["ci_low"], "ci_high": stats["ci_high"], "n_country_blocks": stats["n_countries"]})
            rows = [row for row in result["alignment_rows"] if row["position"] == position and row["layer"] == layer]
            profiles.append({"model_id": model_id, "family": "generative", "position": position, "layer": layer,
                             "relative_depth": layer / depth, "metric": "crossfit_alignment", "mean": float(np.mean([row["cosine_alignment"] for row in rows]))})
            for family in ("generative", "verification"):
                rows = [row for row in structural["cell_summary_rows"] if row["family"] == family and row["position"] == position and row["layer"] == layer]
                profiles.append({"model_id": model_id, "family": family, "position": position, "layer": layer,
                                 "relative_depth": layer / depth, "metric": "noncommuting_share", "mean": float(np.mean([row["noncommuting_share"] for row in rows]))})
    return profiles


def draw_figure(output: Path, profiles: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8), layout="constrained")
    colors = {"hybrid": "#286b4f", "eurollm9": "#a3601b"}
    labels = {"hybrid": "Apertus Hybrid · 32 vrstev", "eurollm9": "EuroLLM 9B · 6bit · 42 vrstev"}
    handles = []
    specs = (("normalized_index", "generative", "Generování faktické odpovědi", "Relační index J · symlog"),
             ("normalized_index", "verification", "Ověřování tvrzení při shodném štítku", "Relační index J · symlog"),
             ("crossfit_alignment", "generative", "Přenos směru do odložené formulace", "Kosinové zarovnání"),
             ("noncommuting_share", "verification", "Interakce relace × forma při ověřování", "Podíl interakční energie"))
    for axis, (metric, family, title, ylabel) in zip(axes.flat, specs):
        axis.axvspan(.25, .75, color="#787878", alpha=.08)
        for model_id in collection.MODEL_IDS:
            rows = [row for row in profiles if row["model_id"] == model_id and row["position"] == "response_boundary" and row["family"] == family and row["metric"] == metric]
            xs, ys = [row["relative_depth"] for row in rows], [row["mean"] for row in rows]
            line, = axis.plot(xs, ys, color=colors[model_id], label=labels[model_id], linewidth=2)
            if len(handles) < 2: handles.append(line)
            if metric == "normalized_index":
                axis.fill_between(xs, [row["ci_low"] for row in rows], [row["ci_high"] for row in rows], color=colors[model_id], alpha=.14)
        if metric == "normalized_index":axis.set_yscale("symlog", linthresh=1)
        if metric == "crossfit_alignment":axis.set_ylim(-1, 1)
        if metric == "noncommuting_share":axis.set_ylim(0, 1)
        axis.axhline(0, color="#777777", linewidth=.7)
        axis.set(xlim=(0, 1), title=title, xlabel="Relativní hloubka vrstvy", ylabel=ylabel)
        axis.grid(alpha=.15)
    fig.suptitle("Relace versus formulace · 180 stejných zadání na model", fontsize=16)
    fig.legend(handles=handles, loc="outside upper center", bbox_to_anchor=(.5, 1.045), ncols=2, frameon=False)
    fig.text(.5, -.035, "Šedá plocha: předem určené střední okno. Pásma J: 95% bootstrap stability pevné sady 10 zemí.\nRůzné tokenizery a kvantizace; nejde o pořadí kvality modelů.", ha="center", fontsize=10)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"comparison.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(summaries: dict, comparison: dict, verification: dict) -> str:
    lines = ["Relace versus formulace: Apertus a EuroLLM", "9. října 2026", "",
             "Stejných 180 českých zadání na každý model: 10 pevných zemí, hlavní město a měna; 60 generativních a 120 verifikačních prefixů.",
             "Texty, metriky, donory nepravdivých tvrzení, formulace vyhrazená pro test přenosu a pravidlo podpory zůstaly stejné jako v původní baterii.",
             "Relativní okno bylo uzamčeno před měřením EuroLLM: 0,25 < vrstva / počet bloků ≤ 0,75; Apertus L9–L24, EuroLLM L11–L31.",
             "Apertus byl v tomto běhu změřen znovu stejnou sběrnou cestou. Jeho starší výsledky již byly známé; jde o přenos na EuroLLM a průzkumné porovnání, nikoli nezávislou replikaci na nových stimulech.", "",
             "Primární výsledky na hranici odpovědi", "Kladný index J znamená bližší stavy různých formulací téže relace než stejné formulace různých relací. Normalizuje se vlastní mezizemní baseline každého modelu."]
    for model_id in collection.MODEL_IDS:
        summary = summaries[model_id]; primary = summary["primary"]
        name = "Apertus Hybrid" if model_id == "hybrid" else "EuroLLM 9B · 6bit"
        lines.extend(["", name])
        for label in ("G", "V"):
            stats = primary["families"][label]
            lines.append(f"  {label}: J = {stats['mean']:.4f}; 95% stabilitní interval [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]; kladné země {stats['positive_country_count']}/10.")
        a = primary["alignment_G"]
        lines.append(f"  Přenos relačního směru do odložené g3: {a['mean']:.4f}; minimum po vnějším vynechání země a úplném refitu {a['outer_refit_leave_one_country_out_minimum']:.4f}.")
        lines.append(f"  Společné převzaté pravidlo G + V + přenos: {primary['support_rule']['label']}.")
        for label in ("G", "V"):
            share = summary["structural_map"]["families"][label]["noncommuting_share"]
            lines.append(f"  Interakční podíl relace × forma ({label}): {share:.4f}.")
        lines.append(f"  Znalostní citlivostní řez: {summary['knowledge_sensitivity']['qualified_country_count']}/10 zemí; úplná sada zůstává primární.")
    lines.extend(["", "Párové rozdíly EuroLLM − Apertus"])
    for label in ("G", "V"):
        d = comparison["families"][label]
        lines.append(f"  {label}: ΔJ = {d['mean']:.4f}; 95% párový stabilitní interval [{d['ci_low']:.4f}, {d['ci_high']:.4f}].")
    d = comparison["alignment_G"]
    outer = [item["difference"] for item in d["outer_country_deletion_refit_differences"]]
    lines.append(f"  Přenos směru: rozdíl {d['mean_difference']:.4f}; rozsah rozdílů vnějších refitů [{min(outer):.4f}, {max(outer):.4f}]. Kvůli sdíleným trénovacím zemím není použit naivní bootstrap alignmentu.")
    lines.extend(["", "Výklad a omezení",
        "G směšuje požadovanou relaci s identitou a třídou faktické odpovědi. V porovnává relace uvnitř stejné pravdivostní hodnoty, a tím kontroluje bezprostřední výstupní štítek; současně však měří jinou úlohu, posouzení tvrzení.",
        "Přenos do odložené formulace testuje stabilitu relačně asociovaného směru. Kosinová geometrie ani interakční podíl samy nedokazují kauzální mechanismus nebo nezávislost významu na lexiku.",
        "Deset zemí je pevná účelová sada. Intervaly vyjadřují stabilitu vůči jejímu složení, nikoli populační nejistotu o jazyce nebo všech zemích. Přímé rozdíly mezi modely jsou průzkumné.",
        "Modely se současně liší vahami, tréninkem, architekturou, tokenizerem a kvantizací (Apertus 8bit tělo / BF16 I/O, EuroLLM 6bit). Vyšší index není sám o sobě lepší model a tento design neizoluje příčinu rozdílu.",
        "Kontrola znalostí používá první token očekávaného pokračování, nikoli úspěšnost celé odpovědi. Vrstevné přechody v profilech jsou popisné; nepřidává se dodatečně zvolené rozhodovací okno.", "", "Technické ověření",
        f"Dokončeno {verification['measurements']} měření; maximální absolutní chyba finálních logitů vůči nezávislému běžnému forwardu: {verification['max_native_final_logit_error']}.",
        "Nativní bloky a kauzální maska; uložené post-block reziduály L0–L32 a L0–L42 na country_end a response_boundary. Float32 je pouze formát uložené kopie. Cache zahrady se nepoužila.",
        "Tokenové kotvy všech 180 zadání byly validovány pro oba tokenizery před měřením; runtime tokenové cesty se přesně shodovaly. Každý raw bundle má úplný inventář a SHA-256 všech souborů; identita vah je založena na obsahu všech modelových souborů.",
        "Výstupy: comparison.json, comparison.png/svg, per-model summary.json a CSV tabulky. Raw data a cesta k nim jsou v verification.json. Odvození lze zopakovat bez inference příkazem run_task_form_transfer.py --derive-only --output <tento_adresář>."])
    if verification.get("derivation_validation_repair"):
        lines.extend(["", "Audit opravy odvození",
            "První odvození zastavila příliš přísná kontrola pořadí true/false po seřazení klíčů JSON. Opravená kontrola ověřuje úplnou jedinečnou množinu ID a zachovává pořadí sběru z neměnného manifestu. Žádný stimul, stav, metrika ani vrstvové okno se nezměnily. Původní kód je v source_snapshot.zip, přesné otisky před opravou a po ní v derivation_repair_1.json."])
    return "\n".join(lines) + "\n"


def derive(output: Path, raw: Path) -> dict:
    from safetensors.numpy import load_file

    lock = parent.load_json(output / "protocol_lock.json")
    lock_sha = parent.sha256_bytes(parent.canonical_json_bytes(lock))
    repair_path = output / "derivation_repair_1.json"
    repair = parent.load_json(repair_path) if repair_path.exists() else None
    for name in ("task_form_analysis.py", "task_form_variance.py", "task_form_transfer.py", "derive_task_form_transfer.py", "derive_task_form_battery.py"):
        relative = str((collection.GARDEN / name).relative_to(collection.ROOT))
        expected = lock["source_sha256"][relative]
        current = parent.sha256_file(collection.GARDEN / name)
        # The completed collection keeps its original lock. One documented
        # validation repair may replace this derivation entry point alone.
        change = repair.get("source_changes", {}).get(relative, {}) if repair else {}
        allowed_repair = (name == "derive_task_form_transfer.py" and repair
                          and repair.get("status") == "collection_complete_derivation_validation_repair"
                          and change.get("collected_sha256") == expected
                          and change.get("derived_sha256") == current)
        if current != expected and not allowed_repair:
            raise parent.BatteryError("Odvození odmítá jinou verzi analytického kódu.")
    summaries, profiles, all_errors = {}, [], []
    verification = {"derived_at": parent.utc_now(), "raw_root": str(raw), "models": {}, "model_or_api_calls_during_derivation": 0}
    if repair: verification["derivation_validation_repair"] = repair
    for model_id in collection.MODEL_IDS:
        bundle = raw / model_id
        manifest = collection.verify_inventory(bundle)
        protocol = lock["protocols"][model_id]
        transfer.validate_protocol(protocol)
        stimuli = align_collection_order(manifest["stimulus_ids"], transfer.expand_protocol(protocol))
        expected_ids = [item["stimulus_id"] for item in stimuli]
        if manifest["lock_sha256"] != lock_sha or manifest["stimulus_ids"] != expected_ids or manifest["model_id"] != model_id:
            raise parent.BatteryError("Raw bundle neodpovídá uzamčenému modelu a množině stimulů.")
        if repair and repair["raw_manifest_sha256"][model_id] != parent.sha256_file(bundle / "manifest.json"):
            raise parent.BatteryError("Raw bundle se při opravě odvození změnil.")
        states, records = {}, {}
        progress = parent.load_json(bundle / "progress.json")
        if set(progress["completed"]) != set(expected_ids):
            raise parent.BatteryError("Chybějící nebo nadbytečné měření.")
        for item in stimuli:
            stimulus_id = item["stimulus_id"]
            record = parent.load_json(bundle / "items" / f"{stimulus_id}.json")
            if record["stimulus"] != item or record["model_id"] != model_id:
                raise parent.BatteryError("Metadata jiného stimulu nebo modelu.")
            states[stimulus_id] = load_file(str(bundle / "states" / f"{stimulus_id}.safetensors"))["states"]
            records[stimulus_id] = record
            all_errors.append(record["native_final_max_absolute_logit_error"])
        result = analysis.analyze_states(protocol, stimuli, states)
        structural = variance.analyze_states(protocol, stimuli, states)
        summary = result["summary"]
        summary["structural_map"] = original_derivation.build_structural_summary(protocol=protocol, variance_rows=structural["variance_rows"], commutation_rows=structural["cell_summary_rows"])
        summary["exploratory_transition"] = original_derivation.build_exploratory_transition_summary(protocol=protocol, index_rows=result["index_rows"], variance_rows=structural["variance_rows"])
        stimulus_rows, knowledge, qualified_ids = original_derivation.build_audit_rows(protocol=protocol, stimuli=stimuli, records_by_id=records, completed=progress["completed"])
        qualified_primary = None
        if len(qualified_ids) >= protocol["design"]["knowledge_sensitivity"]["minimum_qualified_countries_to_report"]:
            if len(qualified_ids) == 10: qualified_primary = summary["primary"]
            else:
                selected = [item for item in stimuli if item["country_id"] in qualified_ids]
                qualified_primary = analysis.analyze_states(protocol, selected, {item["stimulus_id"]: states[item["stimulus_id"]] for item in selected})["summary"]["primary"]
        summary["knowledge_sensitivity"] = {"qualified_country_ids": qualified_ids, "qualified_country_count": len(qualified_ids), "primary": qualified_primary, "scope": "sensitivity_only"}
        summary["interpretation_status"] = "transferred_parent_within_model_rule; cross_model_comparison_exploratory"
        summaries[model_id] = summary
        profiles.extend(layer_profiles(model_id, protocol, result, structural))
        tables = {"stimuli": stimulus_rows, "knowledge_checks": knowledge, "residual_norms": result["norm_rows"],
                  "pairwise_distances": result["pairwise_rows"], "layer_indices": result["index_rows"],
                  "task_direction_alignment": result["alignment_rows"], "factor_variance": structural["variance_rows"],
                  "commutation_summary": structural["cell_summary_rows"],
                  "surface_distances": original_derivation.build_surface_distance_rows(pairwise_rows=result["pairwise_rows"], records_by_id=records)}
        for name, rows in tables.items():parent.atomic_write(output / model_id / f"{name}.csv", original_derivation._csv_bytes(rows))
        parent.atomic_write_json(output / model_id / "summary.json", summary)
        verification["models"][model_id] = {"items": len(stimuli), "state_shape": manifest["state_shape"],
            "manifest_sha256": parent.sha256_file(bundle / "manifest.json"), "raw_files": len(manifest["files"]),
            "model_identity": parent.load_json(bundle / "model.json")["identity"],
            "session_elapsed_seconds": progress["session_elapsed_seconds"],
            "knowledge_top1_by_family": {family: sum(row["expected_first_token_rank"] == 1 for row in knowledge if row["family"] == family) for family in ("generative", "verification")}}
        collection.verify_inventory(bundle)
        del states, records, result, structural
        gc.collect()
    comparison = paired_comparison(summaries, seed=20260729, samples=5000)
    verification.update(measurements=len(all_errors), max_native_final_logit_error=max(all_errors),
                        all_native_final_checks_passed=all(error <= 1e-5 for error in all_errors))
    parent.atomic_write_json(output / "comparison.json", comparison)
    parent.atomic_write_json(output / "profiles.json", {"profiles": profiles})
    parent.atomic_write_json(output / "verification.json", verification)
    parent.atomic_write(output / "report_cs.txt", build_report(summaries, comparison, verification).encode("utf-8"))
    draw_figure(output, profiles)
    return {"summaries": summaries, "comparison": comparison, "verification": verification}
