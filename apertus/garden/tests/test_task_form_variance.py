import sys
import unittest
from pathlib import Path

import numpy as np


GARDEN_DIR = Path(__file__).resolve().parents[1]
if str(GARDEN_DIR) not in sys.path:
    sys.path.insert(0, str(GARDEN_DIR))

import task_form_variance as VARIANCE  # noqa: E402


def _balanced_additive_tensor(
    country_count=4,
    relation_count=2,
    form_count=3,
):
    country = np.eye(country_count) - 1.0 / country_count
    relation = np.eye(relation_count) - 1.0 / relation_count
    form = np.eye(form_count) - 1.0 / form_count
    hidden = 1 + country_count + relation_count + form_count
    tensor = np.zeros(
        (country_count, relation_count, form_count, hidden),
        dtype=np.float64,
    )
    tensor[..., 0] = 3.0
    tensor[..., 1 : 1 + country_count] = country[:, None, None, :]
    start = 1 + country_count
    tensor[..., start : start + relation_count] = relation[None, :, None, :]
    start += relation_count
    tensor[..., start : start + form_count] = form[None, None, :, :]
    return VARIANCE.unit_normalize_states(tensor)


class BalancedTensorDecompositionTests(unittest.TestCase):
    def test_components_reconstruct_centered_tensor_and_ss_partition_total(self):
        rng = np.random.default_rng(9182)
        tensor = VARIANCE.unit_normalize_states(
            rng.normal(size=(5, 2, 3, 11))
        )
        result = VARIANCE.decompose_balanced_tensor(tensor)
        reconstructed = sum(
            (
                result["components"][name]
                for name in VARIANCE.COMPONENTS
            ),
            np.zeros_like(tensor),
        )
        np.testing.assert_allclose(
            reconstructed,
            result["centered"],
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            sum(result["component_ss"].values()),
            result["total_ss"],
            places=11,
        )

        for left_index, left_name in enumerate(VARIANCE.COMPONENTS):
            for right_name in VARIANCE.COMPONENTS[left_index + 1 :]:
                inner_product = float(
                    np.sum(
                        result["components"][left_name]
                        * result["components"][right_name]
                    )
                )
                self.assertAlmostEqual(inner_product, 0.0, places=11)

    def test_pure_additive_tensor_has_zero_rf_and_crf(self):
        result = VARIANCE.decompose_balanced_tensor(
            _balanced_additive_tensor()
        )
        self.assertLess(result["component_ss"]["relation_x_form"], 1e-25)
        self.assertLess(
            result["component_ss"]["country_x_relation_x_form"],
            1e-25,
        )

    def test_injected_relation_form_interaction_is_detected(self):
        additive = _balanced_additive_tensor()
        country_count, relation_count, form_count, old_hidden = additive.shape
        relation = np.eye(relation_count) - 1.0 / relation_count
        form = np.eye(form_count) - 1.0 / form_count
        interaction = np.einsum("ri,fj->rfij", relation, form).reshape(
            relation_count, form_count, relation_count * form_count
        )

        tensor = np.zeros(
            (
                country_count,
                relation_count,
                form_count,
                old_hidden + relation_count * form_count,
            ),
            dtype=np.float64,
        )
        tensor[..., :old_hidden] = additive
        tensor[..., old_hidden:] = 0.4 * interaction[None, ...]
        tensor = VARIANCE.unit_normalize_states(tensor)
        result = VARIANCE.decompose_balanced_tensor(tensor)

        self.assertGreater(
            result["component_ss"]["relation_x_form"],
            1e-3,
        )
        self.assertLess(
            result["component_ss"]["country_x_relation_x_form"],
            1e-25,
        )


class TaskFormVarianceAnalysisTests(unittest.TestCase):
    @staticmethod
    def _fixture():
        countries = ["c1", "c2", "c3"]
        relations = ["capital", "currency"]
        g_forms = ["g1_predicate", "g2_question", "g3_lexical_holdout"]
        v_forms = ["v1_predicate", "v2_answer_front", "v3_paraphrase"]
        truths = ["true", "false"]
        protocol = {
            "design": {
                "layers": [0, 1],
                "measurement_positions": ["response_boundary"],
                "primary_index_forms": {
                    "generative": g_forms[:2],
                    "verification": v_forms,
                },
            },
            "relations": {relation: {} for relation in relations},
            "verification": {
                "form_ids": v_forms,
                "truth_values": {truth: {} for truth in truths},
            },
            "countries": [
                {"country_id": country_id} for country_id in countries
            ],
        }

        country_effect = np.eye(len(countries)) - 1.0 / len(countries)
        relation_effect = np.eye(len(relations)) - 1.0 / len(relations)
        all_forms = g_forms + v_forms
        form_effect = np.eye(len(all_forms)) - 1.0 / len(all_forms)
        hidden = 1 + len(countries) + len(relations) + len(all_forms)
        stimuli = []
        states_by_id = {}

        def add_stimulus(family, truth, country, relation, form):
            stimulus_id = "__".join(
                [family, str(truth), country, relation, form]
            )
            stimuli.append(
                {
                    "stimulus_id": stimulus_id,
                    "family": family,
                    "truth_value": truth,
                    "country_id": country,
                    "relation_id": relation,
                    "form_id": form,
                }
            )
            if form == "g3_lexical_holdout":
                return
            vector = np.zeros(hidden, dtype=np.float64)
            vector[0] = 3.0
            country_index = countries.index(country)
            vector[
                1 : 1 + len(countries)
            ] = country_effect[country_index]
            start = 1 + len(countries)
            relation_index = relations.index(relation)
            vector[
                start : start + len(relations)
            ] = relation_effect[relation_index]
            start += len(relations)
            vector[
                start : start + len(all_forms)
            ] = form_effect[all_forms.index(form)]
            states = np.zeros((2, 1, hidden), dtype=np.float64)
            states[1, 0] = vector
            states_by_id[stimulus_id] = states

        for country in countries:
            for relation in relations:
                for form in g_forms:
                    add_stimulus(
                        "generative", None, country, relation, form
                    )
                for truth in truths:
                    for form in v_forms:
                        add_stimulus(
                            "verification",
                            truth,
                            country,
                            relation,
                            form,
                        )
        return protocol, stimuli, states_by_id

    def test_long_form_rows_use_locked_forms_and_skip_l0(self):
        protocol, stimuli, states_by_id = self._fixture()
        result = VARIANCE.analyze_states(
            protocol, stimuli, states_by_id
        )

        self.assertEqual(len(result["variance_rows"]), 3 * 7)
        self.assertEqual(len(result["cell_summary_rows"]), 3)
        self.assertEqual(
            {row["layer"] for row in result["variance_rows"]},
            {1},
        )
        self.assertEqual(
            {
                (row["family"], row["truth_value"], row["n_forms"])
                for row in result["cell_summary_rows"]
            },
            {
                ("generative", None, 2),
                ("verification", "true", 3),
                ("verification", "false", 3),
            },
        )
        for summary in result["cell_summary_rows"]:
            expected_share = (
                summary["rf_ss"] + summary["crf_ss"]
            ) / (
                summary["r_ss"]
                + summary["f_ss"]
                + summary["cr_ss"]
                + summary["cf_ss"]
                + summary["rf_ss"]
                + summary["crf_ss"]
            )
            self.assertAlmostEqual(
                summary["noncommuting_share"], expected_share
            )
            self.assertAlmostEqual(
                summary["commutation_score"],
                1.0 - expected_share,
            )


if __name__ == "__main__":
    unittest.main()
