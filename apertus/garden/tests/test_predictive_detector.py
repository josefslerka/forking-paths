from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


GARDEN_DIR = Path(__file__).resolve().parents[1]
PREDICTIVE_ROOT = GARDEN_DIR.parent / "predictive_stylometry"
BUNDLE_PATH = (
    PREDICTIVE_ROOT
    / "expansion_v1"
    / "model_v2"
    / "inference_bundle_v2.json"
)
STYLE_PATH = (
    PREDICTIVE_ROOT
    / "expansion_v1"
    / "stylometry"
    / "calibration"
    / "stylometry_features.csv"
)
APERTUS_PATH = (
    PREDICTIVE_ROOT
    / "expansion_v1"
    / "apertus"
    / "calibration"
    / "document_features.csv"
)
PREDICTIONS_PATH = (
    PREDICTIVE_ROOT
    / "expansion_v1"
    / "model_v2"
    / "calibration_predictions.csv"
)
CHALLENGER_BUNDLE_PATH = (
    PREDICTIVE_ROOT
    / "pair_enrichment_v1"
    / "inference_bundle_pair_challenger.json"
)

sys.path.insert(0, str(GARDEN_DIR))
from predictive_detector import (  # noqa: E402
    LEGACY_TIMING_FIELDS_V2,
    TIMING_FIELDS,
    PredictiveDetector,
    sigmoid,
)


def first_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))


class PredictiveDetectorModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (BUNDLE_PATH, STYLE_PATH, APERTUS_PATH, PREDICTIONS_PATH, CHALLENGER_BUNDLE_PATH)
        if not all(path.is_file() for path in required):
            raise unittest.SkipTest("Private detector calibration fixtures are not distributed; synthetic detector/API tests still run.")
        cls.bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
        cls.challenger_bundle = json.loads(
            CHALLENGER_BUNDLE_PATH.read_text(encoding="utf-8")
        )

    def test_sigmoid_is_stable_for_large_logits(self) -> None:
        self.assertEqual(sigmoid(1000.0), 1.0)
        self.assertEqual(sigmoid(-1000.0), 0.0)
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_default_deployment_loads_frozen_v2(self) -> None:
        detector = PredictiveDetector(timing_log_path=None)
        detector._ensure_loaded()
        self.assertEqual(detector.bundle["schema_version"], 2)
        self.assertEqual(
            detector.challenger_bundle["status"],
            "research_only_paired_challenger_posthoc",
        )
        self.assertEqual(
            detector.challenger_manifest["status"],
            "parallel_research_challenger",
        )
        self.assertFalse(
            detector.challenger_manifest["decision_authority"]
        )
        self.assertFalse(
            detector.challenger_manifest["checks"][
                "independent_validation_completed"
            ]
        )
        self.assertFalse(
            detector.challenger_manifest["checks"][
                "promotion_gate_passed"
            ]
        )
        self.assertEqual(
            detector.challenger_manifest["role"],
            "research_only_paired_stylometry_v5",
        )
        self.assertEqual(
            detector.deployment["status"],
            "deployed_frozen_v2",
        )
        self.assertTrue(all(detector.model_lock["checks"].values()))
        self.assertEqual(
            detector.deployment["next_holdout"]["status"],
            "frozen_before_scoring",
        )
        display = detector.deployment["decision_display"]
        self.assertEqual(
            display["basis"],
            "quality_unflagged_fusion_calibration_ai_score_p10",
        )
        self.assertGreater(
            display["borderline_upper_score"],
            display["strict_threshold"],
        )
        self.assertEqual(display["calibration_ai_count"], 60)

    def test_json_runtime_reproduces_frozen_calibration_prediction(self) -> None:
        style = first_row(STYLE_PATH)
        apertus = first_row(APERTUS_PATH)
        self.assertEqual(style["sample_id"], apertus["sample_id"])
        feature_values = {
            key: float(value)
            for key, value in {**style, **apertus}.items()
            if key.startswith("stylo__")
            or key in self.bundle["apertus_features"]
        }
        model = self.bundle["scopes"]["quality_unflagged"]["models"][
            "fusion"
        ]
        actual = PredictiveDetector._predict_model(model, feature_values)

        with PREDICTIONS_PATH.open(encoding="utf-8", newline="") as handle:
            expected = next(
                row
                for row in csv.DictReader(handle)
                if row["scope"] == "quality_unflagged"
                and row["model"] == "fusion"
                and row["sample_id"] == style["sample_id"]
            )
        self.assertAlmostEqual(
            actual["score"],
            float(expected["ai_score"]),
            places=14,
        )
        self.assertEqual(
            actual["strict_ai_signal"],
            expected["strict_ai_signal"] == "True",
        )
        self.assertEqual(
            actual["screening_ai_signal"],
            expected["screening_ai_signal"] == "True",
        )
        self.assertAlmostEqual(
            actual["strict_threshold"],
            float(expected["strict_threshold"]),
        )
        self.assertAlmostEqual(
            actual["screening_threshold"],
            float(expected["screening_threshold"]),
        )
        self.assertEqual(actual["calibration"]["pair_count"], 60)
        self.assertTrue(math.isfinite(actual["logit"]))

    def test_paired_challenger_uses_only_120_style_features(self) -> None:
        style = first_row(STYLE_PATH)
        feature_values: dict[str, float] = {}
        for column in self.challenger_bundle["model"][
            "feature_columns"
        ]:
            try:
                value = float(style[column])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                feature_values[column] = value

        model = self.challenger_bundle["model"]
        actual = PredictiveDetector._predict_challenger_model(
            model,
            feature_values,
        )
        contributions = []
        for index, column in enumerate(model["feature_columns"]):
            value = feature_values.get(
                column,
                float(model["imputer_statistics"][index]),
            )
            scaled = (
                value - float(model["scaler_mean"][index])
            ) / float(model["scaler_scale"][index])
            contributions.append(
                scaled * float(model["logistic_coefficients"][index])
            )
        expected = sigmoid(
            float(model["logistic_intercept"])
            + math.fsum(contributions)
        )
        self.assertAlmostEqual(actual["score"], expected, places=14)
        self.assertEqual(actual["feature_count"], 120)
        self.assertTrue(
            all(
                row["feature"].startswith("stylo__")
                for row in actual["top_feature_contributions"]
            )
        )
        self.assertAlmostEqual(
            actual["strict_threshold"],
            model["thresholds"]["strict"]["threshold"],
        )
        self.assertAlmostEqual(
            actual["screening_threshold"],
            model["thresholds"]["screening"]["threshold"],
        )
        self.assertGreater(model["pair_strength"], 0.0)
        self.assertFalse(
            self.challenger_bundle["promotion_decision"]["passed"]
        )

    def test_challenger_comparison_never_changes_primary_verdict(
        self,
    ) -> None:
        apertus = {
            "score": 0.2,
            "strict_ai_signal": False,
            "screening_ai_signal": False,
        }
        challenger = {
            "score": 0.96,
            "strict_ai_signal": True,
            "screening_ai_signal": True,
        }
        comparison = PredictiveDetector._challenger_comparison(
            "pass",
            apertus,
            challenger,
        )
        self.assertEqual(comparison["status"], "challenger_higher")
        self.assertEqual(comparison["apertus_band"], "low")
        self.assertEqual(comparison["challenger_band"], "strict")
        self.assertFalse(comparison["changes_primary_verdict"])

    def test_quality_gate_abstains_on_non_czech_or_corrupt_input(self) -> None:
        detector = PredictiveDetector()
        detector.bundle = self.bundle
        clean = {
            "input_non_latin_letter_ratio": 0.0,
            "input_unicode_script_count": 1,
            "input_english_function_word_rate": 0.0,
            "input_czech_function_word_rate": 0.2,
            "input_replacement_character_count": 0,
        }
        self.assertEqual(detector._quality_flags(clean), [])

        flagged = dict(clean)
        flagged.update(
            {
                "input_non_latin_letter_ratio": 0.1,
                "input_english_function_word_rate": 0.2,
                "input_czech_function_word_rate": 0.0,
                "input_replacement_character_count": 1,
            }
        )
        codes = {
            flag["code"]
            for flag in detector._quality_flags(flagged)
        }
        self.assertEqual(
            codes,
            {
                "mixed_or_non_latin_script",
                "english_function_words_dominate",
                "unicode_replacement_character",
            },
        )

    def test_apertus_profile_is_descriptive_and_bounded(self) -> None:
        detector = PredictiveDetector()
        detector.bundle = self.bundle
        features = {
            feature: self.bundle["human_reference"][feature][
                "human_median"
            ]
            for feature in self.bundle["apertus_features"]
        }
        profile = detector._predictive_profile(features)
        self.assertEqual(len(profile["features"]), 5)
        self.assertIn(
            "nejde o pravděpodobnost ani klasifikátor",
            profile["index_semantics_cs"],
        )
        self.assertEqual(profile["human_reference_count"], 250)
        self.assertGreaterEqual(profile["ai_direction_index"], 0.0)
        self.assertLessEqual(profile["ai_direction_index"], 100.0)

    def test_timing_log_stores_no_text_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "timings.csv"
            detector = PredictiveDetector(timing_log_path=path)
            row = {
                "analysis_id": "test-analysis",
                "completed_at": "2026-08-03T00:00:00+00:00",
                "model_id": "hybrid",
                "model_generation": 1,
                "character_count": 3000,
                "utf8_byte_count": 3200,
                "normalized_character_count": 3000,
                "full_token_count": 900,
                "scored_token_count": 384,
                "quality_status": "pass",
                "stylometry_ms": 7.0,
                "apertus_ms": 19000.0,
                "model_prediction_ms": 1.0,
                "challenger_model_ms": 0.2,
                "total_ms": 19008.0,
            }
            detector._append_timing(row)
            stored = first_row(path)
            self.assertEqual(stored["analysis_id"], "test-analysis")
            self.assertEqual(stored["total_ms"], "19008.0")
            self.assertEqual(stored["challenger_model_ms"], "0.2")
            self.assertNotIn("text", stored)

    def test_timing_log_migrates_v2_header_without_losing_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "timings.csv"
            legacy_row = {
                field: f"legacy-{field}"
                for field in LEGACY_TIMING_FIELDS_V2
            }
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(LEGACY_TIMING_FIELDS_V2),
                )
                writer.writeheader()
                writer.writerow(legacy_row)

            detector = PredictiveDetector(timing_log_path=path)
            detector._append_timing(
                {
                    "analysis_id": "new-analysis",
                    "challenger_model_ms": 0.25,
                }
            )
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fields = tuple(reader.fieldnames or ())
            self.assertEqual(fields, TIMING_FIELDS)
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                rows[0]["analysis_id"],
                "legacy-analysis_id",
            )
            self.assertEqual(rows[0]["challenger_model_ms"], "")
            self.assertEqual(rows[1]["analysis_id"], "new-analysis")
            self.assertEqual(rows[1]["challenger_model_ms"], "0.25")
            self.assertNotIn("text", fields)


if __name__ == "__main__":
    unittest.main()
