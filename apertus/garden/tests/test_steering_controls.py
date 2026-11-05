from __future__ import annotations

import inspect
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from apertus.garden.api.post_routes import POST_ROUTES
from apertus.garden.server import (
    STEERING_CONTROL_JOB_TIMEOUT_SECONDS,
    STEERING_EFFECTIVE_NORM_RELATIVE_TOLERANCE,
    GardenHandler,
    ModelChangedError,
    ModelState,
)
from apertus.garden.tests.test_steering_teacher_forced import (
    continuation_pairs,
    contrast_pairs,
)


TEST_TEXT = "Nový soused zaklepal. Moje odpověď bude"


class MockSteeringControlsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ModelState(model_path=Path("."), mock=True)
        self.state.status = "ready"
        self.state.model_generation = 7

    def run_shams(
        self,
        *,
        sample_count: int = 4,
        seed: int = 20260730,
    ) -> dict:
        return self.state.steering_sham_control(
            contrast_pairs(),
            TEST_TEXT,
            continuation_pairs(),
            16,
            1.0,
            1.0,
            sample_count,
            seed,
            "mock",
            7,
            self.state.server_instance_id,
        )

    def run_sweep(
        self,
        *,
        layers: list[int] | None = None,
        reference_layer: int = 16,
    ) -> dict:
        return self.state.steering_layer_sweep(
            contrast_pairs(),
            TEST_TEXT,
            continuation_pairs(),
            layers if layers is not None else [8, 16, 24],
            reference_layer,
            1.0,
            1.0,
            "mock",
            7,
            self.state.server_instance_id,
        )

    def assert_effect_formula(self, result: dict) -> None:
        pair_effects = []
        for item in result["per_pair"]:
            contrasts = item["contrasts_nats_per_token"]
            expected = 0.5 * (contrasts["positive"] - contrasts["negative"])
            self.assertAlmostEqual(
                item["symmetric_effect_nats_per_token"],
                expected,
            )
            self.assertAlmostEqual(
                item["symmetric_effect_bits_per_token"],
                expected / math.log(2.0),
            )
            pair_effects.append(expected)
        self.assertAlmostEqual(
            result["effect_nats_per_token"],
            sum(pair_effects) / len(pair_effects),
        )
        self.assertAlmostEqual(
            result["effect_bits_per_token"],
            result["effect_nats_per_token"] / math.log(2.0),
        )

    def test_sham_control_uses_symmetric_matched_pair_effect(self) -> None:
        result = self.run_shams()

        self.assertEqual(result["analysis_kind"], "steering_sham_control")
        self.assertEqual(result["pair_count"], len(continuation_pairs()))
        self.assert_effect_formula(result["target"])
        for sham in result["shams"]:
            self.assert_effect_formula(sham)
            self.assertEqual(sham["kind"], "isotropic_norm_matched_sham")
            self.assertAlmostEqual(sham["direction"]["unit_norm"], 1.0)
            self.assertAlmostEqual(
                sham["direction"]["cosine_to_target"],
                0.0,
                places=12,
            )
            self.assertAlmostEqual(
                sham["nominal_applied_shift_norm"],
                result["target"]["applied_shift_norm"],
            )
            self.assertAlmostEqual(
                sham["effective_shift_norm"],
                result["target"]["effective_shift_norm"],
                places=5,
            )
            self.assertAlmostEqual(
                sham["prefix_relative_shift_norm"],
                sham["effective_shift_norm"]
                / sham["baseline_prefix_residual_norm"],
            )
            self.assertTrue(sham["effective_norm_match_passed"])
            self.assertLessEqual(
                sham["effective_norm_relative_error"],
                sham["effective_norm_match_tolerance"],
            )
            self.assertEqual(len(sham["direction"]["sha256"]), 64)
            self.assertTrue(
                math.isfinite(
                    sham["direction"][
                        "effective_cosine_to_target_after_cast"
                    ]
                )
            )
            self.assertGreater(sham["pre_cast_correction_factor"], 0.0)
            self.assertEqual(
                sham["effective_norm_match_tolerance"],
                STEERING_EFFECTIVE_NORM_RELATIVE_TOLERANCE,
            )
        self.assertEqual(len(result["target"]["direction"]["sha256"]), 64)
        self.assertEqual(
            len(result["target"]["direction"]["pair_norms"]),
            len(contrast_pairs()),
        )
        self.assertEqual(
            len(
                result["target"]["direction"][
                    "pair_cosines_to_direction"
                ]
            ),
            len(contrast_pairs()),
        )
        self.assertTrue(
            result["randomization"]["all_effective_norm_matches_passed"]
        )
        self.assertEqual(
            result["randomization"]["bit_generator"],
            "explicit_numpy_PCG64",
        )
        self.assertFalse(result["randomization"]["exchangeable_null"])
        self.assertFalse(
            result["randomization"]["exact_randomization_test"]
        )
        self.assertEqual(
            result["provenance"]["test"]["token_ids"],
            result["test_token_ids"],
        )
        source_control = result["provenance"]["runtime"]["source_control"]
        self.assertIn("available", source_control)
        self.assertIn("commit_sha", source_control)
        self.assertIn("dirty", source_control)

    def test_sham_randomization_is_reproducible_and_p_is_corrected(
        self,
    ) -> None:
        first = self.run_shams(sample_count=5, seed=1234)
        second = self.run_shams(sample_count=5, seed=1234)

        self.assertEqual(
            first["randomization"]["child_seeds"],
            second["randomization"]["child_seeds"],
        )
        self.assertEqual(
            [item["effect_nats_per_token"] for item in first["shams"]],
            [item["effect_nats_per_token"] for item in second["shams"]],
        )
        self.assertEqual(
            [item["direction"]["sha256"] for item in first["shams"]],
            [item["direction"]["sha256"] for item in second["shams"]],
        )
        summary = first["summary"]
        exceedances = sum(
            item["effect_nats_per_token"]
            >= first["target"]["effect_nats_per_token"]
            for item in first["shams"]
        )
        self.assertEqual(summary["target_exceedance_count"], exceedances)
        self.assertEqual(
            summary["monte_carlo_reference_tail_fraction"],
            (1 + exceedances) / 6,
        )
        self.assertNotIn("empirical_one_sided_p", summary)
        distribution = summary["sham_effect_nats_per_token"]
        expected_z = (
            first["target"]["effect_nats_per_token"] - distribution["mean"]
        ) / distribution["sample_sd"]
        self.assertAlmostEqual(summary["target_z_score"], expected_z)
        self.assertTrue(summary["inference"]["z_score_is_descriptive"])
        self.assertFalse(summary["inference"]["exchangeable_null"])
        self.assertFalse(summary["inference"]["exact_randomization_p"])

    def test_randomization_summary_uses_sample_sd_and_handles_ties(
        self,
    ) -> None:
        summary = self.state._sham_distribution_summary(
            1.0,
            [0.0, 1.0, 1.0, 2.0],
        )
        self.assertAlmostEqual(
            summary["sham_effect_nats_per_token"]["sample_sd"],
            float(np.std([0.0, 1.0, 1.0, 2.0], ddof=1)),
        )
        self.assertEqual(summary["target_exceedance_count"], 3)
        self.assertEqual(
            summary["monte_carlo_reference_tail_fraction"],
            4 / 5,
        )
        self.assertEqual(
            summary["target_percentile_midrank_fraction"],
            0.5,
        )
        with self.assertRaisesRegex(ValueError, "konečná"):
            self.state._sham_distribution_summary(
                float("nan"),
                [0.0, 1.0],
            )

    def test_health_exposes_busy_worker_and_submit_rejects_hidden_queue(
        self,
    ) -> None:
        self.state.outstanding_operation = "steering_sham_control"
        self.state.outstanding_operation_started_monotonic = 1.0
        health = self.state.health()
        self.assertTrue(health["worker"]["busy"])
        self.assertEqual(
            health["worker"]["operation"],
            "steering_sham_control",
        )
        self.assertEqual(health["worker"]["phase"], "queued")
        self.assertGreaterEqual(health["worker"]["elapsed_ms"], 0.0)
        with self.assertRaisesRegex(
            RuntimeError,
            "dokončuje předchozí výpočet",
        ):
            self.state._submit_ready(
                "steering_sham_control",
                (),
            )

    def test_submit_reserves_operation_atomically_before_worker_dequeue(
        self,
    ) -> None:
        with patch(
            "apertus.garden.server.threading.Event.wait",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "časový limit"):
                self.state._submit_ready(
                    "steering_sham_control",
                    (),
                )
        self.assertEqual(
            self.state.outstanding_operation,
            "steering_sham_control",
        )
        self.assertEqual(self.state.work_queue.qsize(), 1)
        with self.assertRaisesRegex(
            RuntimeError,
            "dokončuje předchozí výpočet",
        ):
            self.state._submit_ready(
                "steering_layer_sweep",
                (),
            )
        self.assertEqual(self.state.work_queue.qsize(), 1)

    def test_layer_sweep_rederives_direction_and_reports_scale(self) -> None:
        result = self.run_sweep()

        self.assertEqual(result["analysis_kind"], "steering_layer_sweep")
        self.assertEqual(
            result["direction_policy"],
            "locally_rederived_per_layer",
        )
        self.assertEqual(
            result["steering"]["direction_computations"],
            3,
        )
        self.assertEqual(
            [point["layer"] for point in result["points"]],
            [8, 16, 24],
        )
        for point in result["points"]:
            self.assert_effect_formula(point)
            self.assertGreater(point["native_scale"], 0.0)
            self.assertGreater(point["applied_shift_norm"], 0.0)
            self.assertGreater(point["effective_shift_norm"], 0.0)
            self.assertGreater(
                point["baseline_prefix_residual_norm"],
                0.0,
            )
            self.assertAlmostEqual(
                point["prefix_relative_shift_norm"],
                point["effective_shift_norm"]
                / point["baseline_prefix_residual_norm"],
            )
            self.assertEqual(len(point["direction_sha256"]), 64)
            self.assertIn(
                "resultant_length",
                point["direction_coherence"],
            )
            self.assertEqual(
                len(point["direction_coherence"]["pair_norms"]),
                len(contrast_pairs()),
            )
            self.assertEqual(
                len(
                    point["direction_coherence"][
                        "pair_cosines_to_direction"
                    ]
                ),
                len(contrast_pairs()),
            )
        reference = next(
            point for point in result["points"] if point["layer"] == 16
        )
        self.assertAlmostEqual(
            reference["cosine_to_reference_direction"],
            1.0,
        )
        self.assertFalse(result["summary"]["peak_claim"])

    def test_validation_rejects_bad_controls_and_stale_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_count"):
            self.run_shams(sample_count=1)
        with self.assertRaisesRegex(ValueError, "duplicity"):
            self.run_sweep(layers=[8, 8], reference_layer=8)
        with self.assertRaisesRegex(ValueError, "reference_layer"):
            self.run_sweep(layers=[8, 16], reference_layer=24)
        with self.assertRaises(ModelChangedError):
            self.state.steering_sham_control(
                contrast_pairs(),
                TEST_TEXT,
                continuation_pairs(),
                16,
                1.0,
                1.0,
                2,
                0,
                "bf16",
                7,
                self.state.server_instance_id,
            )
        with self.assertRaises(ModelChangedError):
            self.state.steering_layer_sweep(
                contrast_pairs(),
                TEST_TEXT,
                continuation_pairs(),
                [8, 16],
                16,
                1.0,
                1.0,
                "bf16",
                7,
                self.state.server_instance_id,
            )

    def test_each_control_is_one_serial_worker_job(self) -> None:
        state = ModelState(model_path=Path("."), mock=True)
        state.start()
        try:
            sham = state.steering_sham_control(
                contrast_pairs(),
                TEST_TEXT,
                continuation_pairs()[:1],
                16,
                0.5,
                1.0,
                2,
                0,
                "mock",
                1,
                state.server_instance_id,
            )
            sweep = state.steering_layer_sweep(
                contrast_pairs(),
                TEST_TEXT,
                continuation_pairs()[:1],
                [8, 16],
                16,
                0.5,
                1.0,
                "mock",
                1,
                state.server_instance_id,
            )
            self.assertEqual(
                sham["steering"]["execution"],
                "single_serial_worker_job",
            )
            self.assertEqual(
                sweep["steering"]["execution"],
                "single_serial_worker_job",
            )
        finally:
            state.stop()
            if state.worker_thread is not None:
                state.worker_thread.join(timeout=2)


class SteeringControlsHttpContractTest(unittest.TestCase):
    def test_routes_dispatch_to_state(self) -> None:
        for route, method in (
            (
                "/api/steering/sham-control",
                "self.state.steering_sham_control(",
            ),
            (
                "/api/steering/layer-sweep",
                "self.state.steering_layer_sweep(",
            ),
        ):
            spec = POST_ROUTES[route]
            source = inspect.getsource(
                getattr(GardenHandler, spec.controller)
            )
            self.assertIn(method, source)

    def test_long_controls_have_an_hour_server_wait_budget(self) -> None:
        self.assertEqual(STEERING_CONTROL_JOB_TIMEOUT_SECONDS, 3_600)
        source = inspect.getsource(ModelState._submit_ready)
        self.assertIn('"steering_sham_control"', source)
        self.assertIn('"steering_layer_sweep"', source)


if __name__ == "__main__":
    unittest.main()
