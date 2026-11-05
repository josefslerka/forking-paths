import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


GARDEN_DIR = Path(__file__).resolve().parents[1]
if str(GARDEN_DIR) not in sys.path:
    sys.path.insert(0, str(GARDEN_DIR))

import derive_task_form_battery as DERIVE  # noqa: E402
import run_task_form_battery as RUNNER  # noqa: E402
import task_form_analysis as ANALYSIS  # noqa: E402


class TaskFormProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (GARDEN_DIR / "task_form_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        cls.stimuli = RUNNER.expand_protocol(cls.protocol)

    def test_locked_protocol_validates_and_expands_to_180(self):
        RUNNER.validate_protocol(self.protocol)
        counts = Counter(item["family"] for item in self.stimuli)
        self.assertEqual(len(self.protocol["countries"]), 10)
        self.assertEqual(counts["generative"], 60)
        self.assertEqual(counts["verification"], 120)
        self.assertEqual(len(self.stimuli), 180)
        self.assertEqual(
            len({item["stimulus_id"] for item in self.stimuli}), 180
        )
        self.assertEqual(len({item["prompt"] for item in self.stimuli}), 180)

    def test_false_statements_use_locked_external_donors(self):
        donors = {
            donor["target_country_id"]: donor
            for donor in self.protocol["false_donors"]
        }
        false_rows = [
            item
            for item in self.stimuli
            if item["family"] == "verification"
            and item["truth_value"] == "false"
        ]
        self.assertEqual(len(false_rows), 60)
        for row in false_rows:
            assigned = donors[row["country_id"]]
            answer_field = self.protocol["relations"][row["relation_id"]][
                "answer_field"
            ]
            self.assertEqual(
                row["statement_answer_country_id"], assigned["donor_id"]
            )
            self.assertEqual(row["statement_answer"], assigned[answer_field])
            self.assertIn(assigned[answer_field], row["statement"])
            self.assertEqual(row["expected_output"], "nepravdivý")

    def test_verification_uses_one_wrapper_and_three_distinct_statements(self):
        wrapper = self.protocol["verification"]["wrapper"]
        groups = {}
        for row in self.stimuli:
            if row["family"] != "verification":
                continue
            self.assertEqual(
                row["prompt"], wrapper.format(statement=row["statement"])
            )
            key = (
                row["country_id"],
                row["relation_id"],
                row["truth_value"],
            )
            groups.setdefault(key, []).append(row)
        self.assertEqual(len(groups), 40)
        for rows in groups.values():
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({row["statement"] for row in rows}), 3)

    def test_all_prefixes_share_boundary_and_have_two_exact_anchors(self):
        tokenizer_path = (
            GARDEN_DIR.parent
            / "apertus-8b-base-bf16"
            / "tokenizer.json"
        )
        if not tokenizer_path.is_file():
            self.skipTest("Lokální tokenizer.json není dostupný.")
        try:
            tokenizer = RUNNER.load_tokenizer(tokenizer_path)
        except RUNNER.BatteryError as exc:
            if "tokenizers" in str(exc):
                self.skipTest("Balík tokenizers není v tomto Pythonu.")
            raise
        tokenizations = RUNNER.tokenize_stimuli(
            tokenizer,
            self.stimuli,
            response_boundary_text=self.protocol["design"][
                "response_boundary_text"
            ],
        )
        self.assertEqual(len(tokenizations), 180)
        boundary_ids = set()
        boundary_tokens = set()
        for stimulus in self.stimuli:
            item = tokenizations[stimulus["stimulus_id"]]
            country = item["anchors"]["country_end"]
            boundary = item["anchors"]["response_boundary"]
            self.assertEqual(country["text"], stimulus["country_genitive"])
            self.assertEqual(boundary["text"], "je")
            self.assertLess(country["token_index"], boundary["token_index"])
            self.assertEqual(boundary["token_index"], len(item["prompt_token_ids"]) - 1)
            boundary_ids.add(boundary["token_id"])
            boundary_tokens.add(boundary["token"])
        self.assertEqual(len(boundary_ids), 1)
        self.assertEqual(len(boundary_tokens), 1)

    def test_protocol_rejects_false_donor_with_true_answer(self):
        invalid = copy.deepcopy(self.protocol)
        invalid["false_donors"][0]["capital"] = invalid["countries"][0][
            "capital"
        ]
        with self.assertRaises(RUNNER.BatteryError):
            RUNNER.validate_protocol(invalid)

    def test_validate_only_subprocess_never_imports_mlx(self):
        if importlib.util.find_spec("tokenizers") is None:
            self.skipTest("Balík tokenizers není v tomto Pythonu.")
        tokenizer_path = (
            GARDEN_DIR.parent
            / "apertus-8b-base-bf16"
            / "tokenizer.json"
        )
        if not tokenizer_path.is_file():
            self.skipTest("Lokální tokenizer.json není dostupný.")
        with tempfile.TemporaryDirectory() as temporary:
            guard = Path(temporary)
            (guard / "mlx.py").write_text(
                'raise RuntimeError("validate-only imported mlx")\n',
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                value
                for value in (
                    str(guard),
                    environment.get("PYTHONPATH"),
                )
                if value
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(GARDEN_DIR / "run_task_form_battery.py"),
                    "--validate-only",
                ],
                cwd=GARDEN_DIR.parent.parent,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["validated_stimuli"], 180)
        self.assertIs(payload["mlx_or_model_loaded"], False)


class TaskFormAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (GARDEN_DIR / "task_form_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        cls.stimuli = RUNNER.expand_protocol(cls.protocol)

    def test_cosine_distance_known_cases(self):
        self.assertAlmostEqual(
            float(
                ANALYSIS.cosine_distance(
                    np.array([1.0, 0.0]), np.array([1.0, 0.0])
                )
            ),
            0.0,
        )
        self.assertAlmostEqual(
            float(
                ANALYSIS.cosine_distance(
                    np.array([1.0, 0.0]), np.array([0.0, 1.0])
                )
            ),
            1.0,
        )
        self.assertAlmostEqual(
            float(
                ANALYSIS.cosine_distance(
                    np.array([1.0, 0.0]), np.array([-1.0, 0.0])
                )
            ),
            2.0,
        )

    def _synthetic_states(self, *, relation_dominant: bool):
        countries = {
            country["country_id"]: index
            for index, country in enumerate(self.protocol["countries"])
        }
        generative_forms = list(
            self.protocol["relations"]["capital"]["generative_forms"]
        )
        verification_forms = list(
            self.protocol["verification"].get(
                "forms", self.protocol["verification"]["form_ids"]
            )
        )
        hidden = 24
        result = {}
        relation_scale = 2.0 if relation_dominant else 0.08
        form_scale = 0.08 if relation_dominant else 2.0
        for stimulus in self.stimuli:
            vector = np.zeros(hidden, dtype=np.float64)
            vector[0] = 2.0
            vector[1] = (
                relation_scale
                if stimulus["relation_id"] == "capital"
                else -relation_scale
            )
            forms = (
                generative_forms
                if stimulus["family"] == "generative"
                else verification_forms
            )
            vector[2 + forms.index(stimulus["form_id"])] = form_scale
            vector[5 + countries[stimulus["country_id"]]] = 0.20
            if stimulus["truth_value"] == "false":
                vector[15] = 0.15
            elif stimulus["truth_value"] == "true":
                vector[16] = 0.15
            states = np.empty((33, 2, hidden), dtype=np.float64)
            states[:] = vector
            result[stimulus["stimulus_id"]] = states
        return result

    @staticmethod
    def _middle_generative_mean(result):
        rows = [
            row
            for row in result["index_rows"]
            if row["family"] == "generative"
            and row["position"] == "response_boundary"
            and 9 <= int(row["layer"]) <= 24
        ]
        return float(
            np.mean([float(row["normalized_index"]) for row in rows])
        )

    def test_synthetic_relation_and_form_geometry_flip_index_sign(self):
        positive = ANALYSIS.analyze_states(
            self.protocol,
            self.stimuli,
            self._synthetic_states(relation_dominant=True),
        )
        negative = ANALYSIS.analyze_states(
            self.protocol,
            self.stimuli,
            self._synthetic_states(relation_dominant=False),
        )
        self.assertGreater(self._middle_generative_mean(positive), 0.0)
        self.assertLess(self._middle_generative_mean(negative), 0.0)
        self.assertTrue(
            positive["summary"]["primary"][
                "all_three_predictions_positive"
            ]
        )
        self.assertFalse(
            negative["summary"]["primary"][
                "all_three_predictions_positive"
            ]
        )
        for endpoint in (
            positive["summary"]["primary"]["families"]["G"],
            positive["summary"]["primary"]["families"]["V"],
            positive["summary"]["primary"]["alignment_G"],
        ):
            self.assertGreater(endpoint["mean"], 0.0)
            self.assertGreater(endpoint["ci_low"], 0.0)
            self.assertTrue(
                endpoint[
                    "all_leave_one_country_out_means_positive"
                ]
            )
            self.assertTrue(endpoint["prediction_supported"])

    def test_g3_is_exclusive_to_leave_one_country_out_alignment(self):
        result = ANALYSIS.analyze_states(
            self.protocol,
            self.stimuli,
            self._synthetic_states(relation_dominant=True),
        )
        for row in result["pairwise_rows"]:
            self.assertNotEqual(row["form_id_left"], "g3_lexical_holdout")
            self.assertNotEqual(row["form_id_right"], "g3_lexical_holdout")
        self.assertTrue(
            all(
                row["heldout_form"] == "g3_lexical_holdout"
                and row["leave_one_country_out"]
                for row in result["alignment_rows"]
            )
        )
        self.assertEqual(len(result["norm_rows"]), 11_880)
        self.assertEqual(len(result["pairwise_rows"]), 62_040)
        self.assertEqual(len(result["index_rows"]), 1_920)
        self.assertEqual(len(result["alignment_rows"]), 640)
        self.assertNotIn(0, {row["layer"] for row in result["index_rows"]})
        self.assertNotIn(0, {row["layer"] for row in result["alignment_rows"]})
        one_cell = [
            row
            for row in result["pairwise_rows"]
            if row["layer"] == 12
            and row["position"] == "response_boundary"
        ]
        baselines = [
            row
            for row in one_cell
            if row["pair_kind"] == "different_country_baseline"
        ]
        self.assertEqual(
            sum(row["family"] == "generative" for row in baselines), 180
        )
        self.assertEqual(
            Counter(
                row["truth_value_left"]
                for row in baselines
                if row["family"] == "verification"
            ),
            Counter({"true": 270, "false": 270}),
        )
        verification_indices = [
            row
            for row in result["index_rows"]
            if row["family"] == "verification"
        ]
        self.assertEqual(
            {row["truth_value"] for row in verification_indices},
            {"true", "false"},
        )

    def test_alignment_is_country_loo_and_reversing_g3_reverses_direction(self):
        states = self._synthetic_states(relation_dominant=True)
        original = ANALYSIS.analyze_states(
            self.protocol, self.stimuli, states
        )
        heldout_country = self.protocol["countries"][0]["country_id"]
        perturbed = {key: value.copy() for key, value in states.items()}
        for stimulus in self.stimuli:
            if (
                stimulus["country_id"] == heldout_country
                and stimulus["family"] == "generative"
                and stimulus["form_id"] != "g3_lexical_holdout"
            ):
                perturbed[stimulus["stimulus_id"]][..., 1] *= -7.0
        changed = ANALYSIS.analyze_states(
            self.protocol, self.stimuli, perturbed
        )
        original_values = [
            row["cosine_alignment"]
            for row in original["alignment_rows"]
            if row["country_id"] == heldout_country
        ]
        changed_values = [
            row["cosine_alignment"]
            for row in changed["alignment_rows"]
            if row["country_id"] == heldout_country
        ]
        np.testing.assert_allclose(original_values, changed_values)

        reversed_g3 = {
            key: value.copy() for key, value in states.items()
        }
        by_condition = {
            (
                stimulus["country_id"],
                stimulus["relation_id"],
                stimulus["form_id"],
            ): stimulus["stimulus_id"]
            for stimulus in self.stimuli
            if stimulus["family"] == "generative"
        }
        for country in self.protocol["countries"]:
            capital_id = by_condition[
                (country["country_id"], "capital", "g3_lexical_holdout")
            ]
            currency_id = by_condition[
                (country["country_id"], "currency", "g3_lexical_holdout")
            ]
            capital = reversed_g3[capital_id].copy()
            reversed_g3[capital_id] = reversed_g3[currency_id].copy()
            reversed_g3[currency_id] = capital
        reversed_result = ANALYSIS.analyze_states(
            self.protocol, self.stimuli, reversed_g3
        )
        self.assertLess(
            reversed_result["summary"]["primary"]["alignment_G"]["mean"],
            -0.99,
        )

    def test_zero_cross_country_baseline_is_rejected(self):
        states = self._synthetic_states(relation_dominant=True)
        for values in states.values():
            values[..., 5:15] = 0.0
        with self.assertRaises(ANALYSIS.AnalysisError):
            ANALYSIS.analyze_states(self.protocol, self.stimuli, states)

    def test_degenerate_embedding_layer_is_kept_as_sanity_only(self):
        states = self._synthetic_states(relation_dominant=True)
        common_embedding = np.zeros(24, dtype=np.float64)
        common_embedding[0] = 1.0
        for values in states.values():
            values[0, :, :] = common_embedding
        result = ANALYSIS.analyze_states(
            self.protocol, self.stimuli, states
        )
        self.assertIn(0, {row["layer"] for row in result["norm_rows"]})
        self.assertIn(0, {row["layer"] for row in result["pairwise_rows"]})
        self.assertNotIn(0, {row["layer"] for row in result["index_rows"]})
        self.assertNotIn(0, {row["layer"] for row in result["alignment_rows"]})


class TaskFormManifestTests(unittest.TestCase):
    def _make_bundle(self, root: Path) -> Path:
        series_dir = root / "manifest-test"
        series_dir.mkdir()
        (series_dir / "checkpoint.json").write_text(
            '{"status":"complete"}\n', encoding="utf-8"
        )
        states_dir = series_dir / "states"
        states_dir.mkdir()
        (states_dir / "one.bin").write_bytes(b"locked-state")
        manifest = {
            "format": f"{RUNNER.FORMAT}.manifest",
            "schema_version": RUNNER.SCHEMA_VERSION,
            "series_id": "manifest-test",
            "status": "complete",
            "files": RUNNER.inventory_files(series_dir),
        }
        RUNNER.atomic_write_json(series_dir / "manifest.json", manifest)
        return series_dir

    def test_complete_manifest_detects_corruption_and_extra_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            series_dir = self._make_bundle(Path(temporary))
            verified = DERIVE.verify_raw_manifest(
                series_dir, expected_series_id="manifest-test"
            )
            self.assertEqual(len(verified["files"]), 2)

            target = series_dir / "states" / "one.bin"
            target.write_bytes(b"corrupt")
            with self.assertRaises(RUNNER.BatteryError):
                DERIVE.verify_raw_manifest(
                    series_dir, expected_series_id="manifest-test"
                )

        with tempfile.TemporaryDirectory() as temporary:
            series_dir = self._make_bundle(Path(temporary))
            (series_dir / "unlisted.txt").write_text(
                "extra", encoding="utf-8"
            )
            with self.assertRaises(RUNNER.BatteryError):
                DERIVE.verify_raw_manifest(
                    series_dir, expected_series_id="manifest-test"
                )


class TaskFormDerivationHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (GARDEN_DIR / "task_form_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        cls.stimuli = RUNNER.expand_protocol(cls.protocol)

    def _records_and_completed(self):
        first_country = self.protocol["countries"][0]["country_id"]
        second_country = self.protocol["countries"][1]["country_id"]
        records = {}
        completed = {}
        for stimulus in self.stimuli:
            rank = 100
            if stimulus["family"] == "verification":
                rank = 5
            elif stimulus["country_id"] == first_country and stimulus[
                "form_id"
            ] in {"g1_predicate", "g2_question"}:
                rank = 10
            elif stimulus["country_id"] == second_country:
                supported = {
                    ("capital", "g1_predicate"),
                    ("capital", "g3_lexical_holdout"),
                    ("currency", "g2_question"),
                    ("currency", "g3_lexical_holdout"),
                }
                if (stimulus["relation_id"], stimulus["form_id"]) in supported:
                    rank = 10
            stimulus_id = stimulus["stimulus_id"]
            records[stimulus_id] = {
                "captured_at": "2026-07-29T00:00:00Z",
                "tokenization": {
                    "prompt_token_ids": [1, 10, 11],
                    "prompt_token_count": 3,
                    "anchors": {
                        "country_end": {"token_index": 1, "token_id": 10},
                        "response_boundary": {
                            "token_index": 2,
                            "token_id": 11,
                        },
                    },
                    "target": {
                        "first_token_id": 20,
                        "first_token": "target",
                    },
                },
                "final_distribution": {
                    "expected_first_token": {
                        "rank": rank,
                        "probability": 0.25,
                        "in_top8": rank <= 8,
                    }
                },
                "timing_ms": {"forward_and_projection_ms": 1.0},
            }
            completed[stimulus_id] = {
                "metadata_file": f"stimuli/{stimulus_id}.json",
                "metadata_sha256": "a" * 64,
                "state_file": f"states/{stimulus_id}.safetensors",
                "state_sha256": "b" * 64,
            }
        return records, completed

    def test_knowledge_rows_include_v_and_require_shared_g_forms(self):
        records, completed = self._records_and_completed()
        stimulus_rows, knowledge_rows, qualified = DERIVE.build_audit_rows(
            protocol=self.protocol,
            stimuli=self.stimuli,
            records_by_id=records,
            completed=completed,
        )
        self.assertEqual(len(stimulus_rows), 180)
        self.assertEqual(len(knowledge_rows), 180)
        self.assertEqual(
            sum(row["family"] == "verification" for row in knowledge_rows),
            120,
        )
        first_country = self.protocol["countries"][0]["country_id"]
        second_country = self.protocol["countries"][1]["country_id"]
        self.assertIn(first_country, qualified)
        self.assertNotIn(second_country, qualified)
        second_rows = [
            row
            for row in knowledge_rows
            if row["family"] == "generative"
            and row["country_id"] == second_country
        ]
        self.assertEqual(
            {row["shared_supported_generative_form_count"] for row in second_rows},
            {1},
        )
        self.assertEqual(
            {
                row["form_id"]
                for row in second_rows
                if row["paired_relation_form_supported"]
            },
            {"g3_lexical_holdout"},
        )

    def test_surface_distance_deduplicates_layers_and_removes_bos(self):
        records, _ = self._records_and_completed()
        left = self.stimuli[0]
        right = next(
            stimulus
            for stimulus in self.stimuli
            if stimulus["family"] == "generative"
            and stimulus["country_id"] == left["country_id"]
            and stimulus["relation_id"] != left["relation_id"]
            and stimulus["form_id"] == left["form_id"]
        )
        records[right["stimulus_id"]]["tokenization"][
            "prompt_token_ids"
        ] = [1, 10, 12, 13]
        pair = {
            "pair_kind": "different_relation_same_form",
            "family": "generative",
            "stimulus_id_left": left["stimulus_id"],
            "stimulus_id_right": right["stimulus_id"],
            "country_id_left": left["country_id"],
            "country_id_right": right["country_id"],
            "relation_id_left": left["relation_id"],
            "relation_id_right": right["relation_id"],
            "form_id_left": left["form_id"],
            "form_id_right": right["form_id"],
            "truth_value_left": None,
            "truth_value_right": None,
        }
        rows = DERIVE.build_surface_distance_rows(
            pairwise_rows=[
                dict(pair, layer=0, position="country_end"),
                dict(pair, layer=1, position="response_boundary"),
            ],
            records_by_id=records,
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(
            rows[0]["normalized_token_edit_distance"], 2 / 3
        )
        self.assertAlmostEqual(
            rows[0]["token_set_jaccard_similarity"], 0.25
        )
        self.assertTrue(rows[0]["bos_removed"])

    def test_joint_decision_requires_full_locked_hybrid_run(self):
        country_ids = [
            country["country_id"] for country in self.protocol["countries"]
        ]
        base_summary = {
            "primary": {
                "support_rule": {
                    "supported": True,
                    "label": "supported",
                }
            }
        }
        full = copy.deepcopy(base_summary)
        self.assertTrue(
            DERIVE.apply_decision_eligibility(
                summary=full,
                protocol=self.protocol,
                checkpoint={"model_mode": "hybrid"},
                selected_country_ids=country_ids,
            )
        )
        self.assertTrue(full["primary"]["support_rule"]["supported"])
        self.assertTrue(
            full["primary"]["support_rule"]["decision_eligible"]
        )

        smoke = copy.deepcopy(base_summary)
        self.assertFalse(
            DERIVE.apply_decision_eligibility(
                summary=smoke,
                protocol=self.protocol,
                checkpoint={"model_mode": "hybrid"},
                selected_country_ids=country_ids[:2],
            )
        )
        self.assertIsNone(smoke["primary"]["support_rule"]["supported"])
        self.assertEqual(
            smoke["primary"]["support_rule"]["label"],
            "descriptive_only",
        )

        bf16 = copy.deepcopy(base_summary)
        self.assertFalse(
            DERIVE.apply_decision_eligibility(
                summary=bf16,
                protocol=self.protocol,
                checkpoint={"model_mode": "bf16"},
                selected_country_ids=country_ids,
            )
        )
        self.assertIsNone(bf16["primary"]["support_rule"]["supported"])

    def test_structural_summary_truth_weights_v_equally(self):
        variance_rows = []
        commutation_rows = []
        component_share = 1.0 / len(DERIVE.variance.COMPONENTS)
        for family, truth_values in (
            ("generative", [None]),
            ("verification", ["true", "false"]),
        ):
            for truth_value in truth_values:
                noncommuting_share = (
                    0.2
                    if truth_value in (None, "true")
                    else 0.6
                )
                for layer in self.protocol["design"]["primary_layers"]:
                    for component in DERIVE.variance.COMPONENTS:
                        variance_rows.append(
                            {
                                "family": family,
                                "truth_value": truth_value,
                                "layer": layer,
                                "position": "response_boundary",
                                "component": component,
                                "proportion_total": component_share,
                            }
                        )
                    commutation_rows.append(
                        {
                            "family": family,
                            "truth_value": truth_value,
                            "layer": layer,
                            "position": "response_boundary",
                            "noncommuting_share": noncommuting_share,
                            "commutation_score": 1.0
                            - noncommuting_share,
                        }
                    )
        summary = DERIVE.build_structural_summary(
            protocol=self.protocol,
            variance_rows=variance_rows,
            commutation_rows=commutation_rows,
        )
        self.assertEqual(
            summary["status"],
            "descriptive_structural_map_no_support_threshold",
        )
        self.assertAlmostEqual(
            summary["families"]["G"]["noncommuting_share"], 0.2
        )
        self.assertAlmostEqual(
            summary["families"]["V"]["noncommuting_share"], 0.4
        )
        self.assertAlmostEqual(
            sum(
                summary["families"]["V"][
                    "component_proportion_of_total"
                ].values()
            ),
            1.0,
        )

    def test_exploratory_transition_localizes_crossing_and_all_country_run(
        self,
    ):
        country_ids = [
            country["country_id"] for country in self.protocol["countries"]
        ]
        index_rows = []
        variance_rows = []
        for layer in range(1, 33):
            for country_index, country_id in enumerate(country_ids):
                value = -1.0 if layer < 17 else 1.0
                if layer == 17 and country_index < 4:
                    value = -0.1
                if layer == 32 and country_index < 2:
                    value = -0.1
                index_rows.append(
                    {
                        "family": "generative",
                        "country_id": country_id,
                        "layer": layer,
                        "position": "response_boundary",
                        "normalized_index": value,
                    }
                )
            form_share, relation_share = (
                (0.6, 0.2) if layer < 17 else (0.2, 0.6)
            )
            for component, share in (
                ("form", form_share),
                ("relation", relation_share),
            ):
                variance_rows.append(
                    {
                        "family": "generative",
                        "truth_value": None,
                        "layer": layer,
                        "position": "response_boundary",
                        "component": component,
                        "proportion_total": share,
                    }
                )
        transition = DERIVE.build_exploratory_transition_summary(
            protocol=self.protocol,
            index_rows=index_rows,
            variance_rows=variance_rows,
        )
        self.assertEqual(
            transition["first_layer_with_positive_group_mean_index"], 17
        )
        self.assertEqual(
            transition["first_layer_where_relation_ss_share_exceeds_form"],
            17,
        )
        self.assertTrue(transition["crossings_coincide"])
        self.assertEqual(
            transition["longest_contiguous_all_countries_positive_run"],
            {"start_layer": 18, "end_layer": 31},
        )


if __name__ == "__main__":
    unittest.main()
