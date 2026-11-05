import copy
import json
import math
import sys
import unittest
from pathlib import Path


GARDEN_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = GARDEN_DIR.parents[1]
if str(GARDEN_DIR) not in sys.path:
    sys.path.insert(0, str(GARDEN_DIR))

import run_zahrada12 as RUNNER  # noqa: E402


class Zahrada12MaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = REPOSITORY_ROOT / "experiment-zahrada12-materials.json"
        cls.materials, _raw, cls.digest = RUNNER.load_frozen_materials(
            cls.path
        )
        cls.expanded = RUNNER.expand_materials(cls.materials)

    def test_frozen_hash_and_balanced_expansion(self):
        self.assertEqual(
            self.digest,
            RUNNER.FROZEN_MATERIALS_SHA256,
        )
        self.assertEqual(len(self.materials["items"]), 10)
        self.assertEqual(len(self.expanded), 120)
        self.assertEqual(
            len({row["run_id"] for row in self.expanded}),
            120,
        )
        self.assertEqual(
            len({row["text"] for row in self.expanded}),
            120,
        )

    def test_each_prefix_has_exactly_three_locked_candidates(self):
        groups = {}
        for row in self.expanded:
            groups.setdefault((row["item_id"], row["prefix_id"]), []).append(
                row["candidate_id"]
            )
            self.assertEqual(
                row["text"],
                row["prefix"] + row["candidate"],
            )
            self.assertEqual(
                row["candidate_character_start"],
                len(row["prefix"]),
            )
        self.assertEqual(len(groups), 40)
        for (_item_id, prefix_id), candidate_ids in groups.items():
            self.assertEqual(
                tuple(candidate_ids),
                RUNNER.SCORING_MATRIX[prefix_id],
            )

    def test_material_validation_rejects_changed_candidate(self):
        changed = copy.deepcopy(self.materials)
        changed["candidates"]["normative_failure"] += "."
        with self.assertRaises(RUNNER.BatteryError):
            RUNNER.validate_materials(changed)


class Zahrada12MeasurementTests(unittest.TestCase):
    def test_extracts_only_candidate_tokens(self):
        row = {
            "run_id": "synthetic",
            "prefix": "Prefix o",
            "candidate": " pokračování",
            "candidate_character_start": 8,
            "text": "Prefix o pokračování",
        }
        response = {
            "analysis_kind": "continuation_surprisal",
            "prefix": row["prefix"],
            "continuation": row["candidate"],
            "full_text": row["text"],
            "top_k": RUNNER.TOP_K,
            "temperature": 1.0,
            "probability_kind": "raw_model",
            "execution": {
                "mode": "teacher_forced_incremental_kv",
                "generation": False,
                "steering": False,
                "future_continuation_tokens_in_forward": False,
            },
            "prefix_token_ids": [1, 2],
            "prompt_logits_sha256": "a" * 64,
            "tokens": [
                {
                    "token_id": 3,
                    "text": " pokrač",
                    "character_start": 8,
                    "character_end": 16,
                    "surprisal_bits": 4.0,
                },
                {
                    "token_id": 4,
                    "text": "ování",
                    "character_start": 16,
                    "character_end": 21,
                    "surprisal_bits": 5.0,
                },
            ],
            "summary": {"total_continuation_surprisal_bits": 9.0},
        }
        measurement = RUNNER.extract_measurement(row, response)
        self.assertEqual(measurement["prefix_token_ids"], [1, 2])
        self.assertEqual(measurement["candidate_token_ids"], [3, 4])
        self.assertEqual(measurement["conditional_surprisal_bits"], 9.0)

    def test_rejects_token_crossing_candidate_boundary(self):
        row = {
            "run_id": "synthetic",
            "prefix": "abc",
            "candidate": " def",
            "candidate_character_start": 3,
            "text": "abc def",
        }
        response = {
            "analysis_kind": "continuation_surprisal",
            "prefix": row["prefix"],
            "continuation": row["candidate"],
            "full_text": row["text"],
            "top_k": RUNNER.TOP_K,
            "temperature": 1.0,
            "probability_kind": "raw_model",
            "execution": {
                "mode": "teacher_forced_incremental_kv",
                "generation": False,
                "steering": False,
                "future_continuation_tokens_in_forward": False,
            },
            "prefix_token_ids": [1],
            "prompt_logits_sha256": "a" * 64,
            "tokens": [
                {
                    "token_id": 1,
                    "text": "abc ",
                    "character_start": 0,
                    "character_end": 4,
                    "surprisal_bits": 1.0,
                },
                {
                    "token_id": 2,
                    "text": "def",
                    "character_start": 4,
                    "character_end": 7,
                    "surprisal_bits": 1.0,
                },
            ],
            "summary": {"total_continuation_surprisal_bits": 2.0},
        }
        with self.assertRaises(RUNNER.BatteryError):
            RUNNER.extract_measurement(row, response)


class Zahrada12AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        materials_path = (
            REPOSITORY_ROOT / "experiment-zahrada12-materials.json"
        )
        cls.materials, _raw, _digest = RUNNER.load_frozen_materials(
            materials_path
        )
        cls.expanded = RUNNER.expand_materials(cls.materials)

    def synthetic_measurements(self):
        scores = {
            "normative_failure": {
                "normative_failure": 1.0,
                "epistemic_failure": 5.0,
                "normative_success": 4.0,
            },
            "epistemic_failure": {
                "normative_failure": 4.0,
                "epistemic_failure": 1.0,
                "epistemic_success": 4.0,
            },
            "normative_success": {
                "normative_success": 1.0,
                "epistemic_success": 5.0,
                "normative_failure": 4.0,
            },
            "epistemic_success": {
                "normative_success": 4.0,
                "epistemic_success": 1.0,
                "epistemic_failure": 4.0,
            },
        }
        measurements = {}
        for row in self.expanded:
            measurements[row["run_id"]] = {
                "conditional_surprisal_bits": scores[row["prefix_id"]][
                    row["candidate_id"]
                ],
                "prefix_token_ids": [1, 2, 3],
                "prompt_logits_sha256": "a" * 64,
            }
        return measurements

    def test_known_positive_pattern_passes_full_gate(self):
        analysis = RUNNER.compute_analysis(
            self.materials,
            self.expanded,
            self.synthetic_measurements(),
        )
        self.assertEqual(analysis["primary"]["mean_D_bits"], 7.0)
        self.assertEqual(analysis["primary"]["D_positive_count"], 10)
        self.assertTrue(
            analysis["decision"]["full_confirmatory_gate_passed"]
        )
        self.assertEqual(
            analysis["decision"]["outcome"],
            "A_differentiated_status_and_outcome",
        )
        self.assertAlmostEqual(
            analysis["primary"]["exact_one_sided_sign_flip_p"],
            1 / 1024,
        )
        self.assertTrue(
            all(
                math.isclose(row["C_normative_bits"], 3.0)
                and math.isclose(row["C_epistemic_bits"], 3.0)
                for row in analysis["items"]
            )
        )


if __name__ == "__main__":
    unittest.main()
