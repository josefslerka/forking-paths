from __future__ import annotations

import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

from apertus.garden.server import (
    MAX_STEERING_TOTAL_TOKENS,
    ModelChangedError,
    ModelState,
    aggregate_steering_deltas,
)


def contrast_pairs() -> list[dict[str, str]]:
    return [
        {
            "negative_text": "Ten tón působí chladně v tomto rámci",
            "positive_text": "Ten tón působí vřele v tomto rámci",
        },
        {
            "negative_text": "Odpověď je odmítavá v tomto rámci",
            "positive_text": "Odpověď je vstřícná v tomto rámci",
        },
        {
            "negative_text": "Vyznění zůstává odtažité v tomto rámci",
            "positive_text": "Vyznění zůstává přátelské v tomto rámci",
        },
    ]


class SteeringAggregationTest(unittest.TestCase):
    def test_pair_magnitudes_do_not_dominate_direction(self) -> None:
        ordinary = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
            ]
        )
        extreme = ordinary.copy()
        extreme[1] *= 1_000_000.0

        ordinary_direction, ordinary_diagnostics = (
            aggregate_steering_deltas(ordinary)
        )
        extreme_direction, extreme_diagnostics = (
            aggregate_steering_deltas(extreme)
        )

        np.testing.assert_allclose(
            ordinary_direction,
            extreme_direction,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            ordinary_direction,
            np.array([2.0, 1.0]) / np.sqrt(5.0),
            atol=1e-12,
        )
        self.assertEqual(ordinary_diagnostics["native_scale"], 1.0)
        self.assertEqual(extreme_diagnostics["native_scale"], 1.0)
        self.assertAlmostEqual(
            ordinary_diagnostics["resultant_length"],
            np.sqrt(5.0) / 3.0,
        )

    def test_zero_and_cancelling_deltas_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nulový"):
            aggregate_steering_deltas(
                np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 1.0],
                    ]
                )
            )
        with self.assertRaisesRegex(ValueError, "ruší"):
            aggregate_steering_deltas(
                np.array(
                    [
                        [1.0, 0.0],
                        [-1.0, 0.0],
                    ]
                )
            )

    def test_aggregation_is_permutation_invariant(self) -> None:
        deltas = np.array(
            [
                [2.0, 1.0, 0.0],
                [0.0, 4.0, 1.0],
                [1.0, 0.0, 3.0],
                [2.0, 2.0, 2.0],
            ]
        )
        direction, diagnostics = aggregate_steering_deltas(deltas)
        permuted_direction, permuted_diagnostics = (
            aggregate_steering_deltas(deltas[[2, 0, 3, 1]])
        )
        np.testing.assert_allclose(
            direction,
            permuted_direction,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            diagnostics["native_scale"],
            permuted_diagnostics["native_scale"],
        )
        self.assertAlmostEqual(
            diagnostics["resultant_length"],
            permuted_diagnostics["resultant_length"],
        )


class MockSteeringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ModelState(model_path=Path("."), mock=True)
        self.state.status = "ready"
        self.state.model_generation = 7

    def run_steering(
        self,
        *,
        pairs: list[dict[str, str]] | None = None,
        test_text: str = "Nový text v nezávislém rámci",
        alpha: float = 0.75,
    ) -> dict:
        return self.state.steering(
            pairs if pairs is not None else contrast_pairs(),
            test_text,
            16,
            alpha,
            1.0,
            5,
            6,
            10,
            "mock",
            7,
            self.state.server_instance_id,
        )

    def test_response_contains_paired_metrics_and_direction_diagnostics(
        self,
    ) -> None:
        result = self.run_steering()

        self.assertEqual(result["left"]["label"], "baseline")
        self.assertEqual(result["right"]["label"], "steered")
        self.assertEqual(len(result["left"]["top_tokens"]), 5)
        self.assertEqual(len(result["right"]["top_tokens"]), 5)
        self.assertGreater(result["js_divergence_bits"], 0.0)
        self.assertGreater(result["total_variation"], 0.0)
        self.assertIn("gains", result)
        self.assertIn("losses", result)
        self.assertEqual(
            result["movers"]["gains"],
            result["gains"],
        )
        steering = result["steering"]
        self.assertEqual(steering["layer"], 16)
        self.assertEqual(steering["pair_count"], 3)
        self.assertEqual(
            steering["aggregation"],
            "normalized_pair_mean_with_median_native_scale",
        )
        self.assertGreater(steering["native_scale"], 0.0)
        self.assertGreater(steering["effective_shift_norm"], 0.0)
        self.assertGreater(steering["resultant_length"], 0.0)
        self.assertLessEqual(steering["resultant_length"], 1.0)
        self.assertEqual(len(steering["pair_diagnostics"]), 3)
        self.assertTrue(
            all(
                pair["negative_token_ids"][-1]
                == pair["positive_token_ids"][-1]
                == pair["boundary_token_id"]
                for pair in steering["pair_diagnostics"]
            )
        )
        self.assertEqual(
            steering["cache_policy"],
            "cache_free_no_state_mutation",
        )
        self.assertEqual(result["model_id"], "mock")
        self.assertEqual(result["model_generation"], 7)
        self.assertFalse(result["cached"])

    def test_alpha_zero_is_exact_identity_and_does_not_mutate_caches(
        self,
    ) -> None:
        cached_logits = np.array([1.0, 2.0], dtype=np.float32)
        self.state.logits_cache = OrderedDict(
            [((1, 2), cached_logits.copy())]
        )
        self.state.lens_cache = OrderedDict(
            [((7, "lens", 5), {"kept": True})]
        )
        self.state.tuned_lens_cache = OrderedDict(
            [((7, "a", "b", "c", "d", 5), {"kept": True})]
        )
        self.state.active_ids = [1, 2]
        active_cache = object()
        self.state.active_cache = active_cache
        self.state.active_logits = cached_logits.copy()

        result = self.run_steering(alpha=0.0)

        self.assertEqual(result["js_divergence_bits"], 0.0)
        self.assertEqual(result["total_variation"], 0.0)
        self.assertEqual(result["delta_entropy_bits"], 0.0)
        self.assertEqual(result["gains"], [])
        self.assertEqual(result["losses"], [])
        self.assertEqual(
            result["left"]["top_tokens"],
            result["right"]["top_tokens"],
        )
        self.assertEqual(
            result["steering"]["effective_shift_norm"],
            0.0,
        )
        self.assertIn(
            "identity_control",
            [warning["code"] for warning in result["warnings"]],
        )

        self.assertEqual(list(self.state.logits_cache), [(1, 2)])
        np.testing.assert_array_equal(
            self.state.logits_cache[(1, 2)],
            cached_logits,
        )
        self.assertEqual(
            self.state.lens_cache[(7, "lens", 5)],
            {"kept": True},
        )
        self.assertEqual(
            self.state.tuned_lens_cache[(7, "a", "b", "c", "d", 5)],
            {"kept": True},
        )
        self.assertEqual(self.state.active_ids, [1, 2])
        self.assertIs(self.state.active_cache, active_cache)
        np.testing.assert_array_equal(
            self.state.active_logits,
            cached_logits,
        )

    def test_underflowed_nonzero_alpha_is_reported_as_ineffective(self) -> None:
        result = self.run_steering(alpha=1e-50)

        self.assertEqual(result["steering"]["effective_shift_norm"], 0.0)
        self.assertIn(
            "ineffective_shift",
            [warning["code"] for warning in result["warnings"]],
        )

    def test_trailing_whitespace_is_trimmed_but_leading_is_preserved(
        self,
    ) -> None:
        pairs = contrast_pairs()
        pairs[0]["negative_text"] = "  " + pairs[0]["negative_text"]
        for pair in pairs:
            pair["negative_text"] += "   "
            pair["positive_text"] += "\t"
        result = self.run_steering(
            pairs=pairs,
            test_text="  Nový text v nezávislém rámci   ",
        )
        self.assertEqual(
            result["left"]["text"],
            "  Nový text v nezávislém rámci",
        )
        self.assertTrue(
            all(
                pair["negative_text"] == pair["negative_text"].rstrip()
                and pair["positive_text"] == pair["positive_text"].rstrip()
                for pair in result["steering"]["pair_diagnostics"]
            )
        )
        self.assertTrue(
            result["steering"]["pair_diagnostics"][0][
                "negative_text"
            ].startswith("  ")
        )

    def test_boundary_mismatch_duplicate_pairs_and_reused_test_are_rejected(
        self,
    ) -> None:
        mismatch = contrast_pairs()
        mismatch[0]["positive_text"] = (
            "Ten tón působí vřele v jiném zakončení"
        )
        with self.assertRaisesRegex(ValueError, "stejným token_id"):
            self.run_steering(pairs=mismatch)

        duplicated = [contrast_pairs()[0] for _ in range(3)]
        with self.assertRaisesRegex(ValueError, "duplicitní"):
            self.run_steering(pairs=duplicated)

        pairs = contrast_pairs()
        reversed_duplicate = [
            pairs[0],
            {
                "negative_text": pairs[0]["positive_text"],
                "positive_text": pairs[0]["negative_text"],
            },
            pairs[1],
        ]
        with self.assertRaisesRegex(ValueError, "duplicitní"):
            self.run_steering(pairs=reversed_duplicate)

        with self.assertRaisesRegex(ValueError, "nezávislý"):
            self.run_steering(
                pairs=pairs,
                test_text=pairs[1]["positive_text"],
            )

    def test_total_token_budget_is_enforced(self) -> None:
        pairs: list[dict[str, str]] = []
        for index in range(8):
            shared = " ".join(
                [f"společný{index}"] * 126 + ["hranice"]
            )
            pairs.append(
                {
                    "negative_text": f"negativní{index} {shared}",
                    "positive_text": f"pozitivní{index} {shared}",
                }
            )
        with self.assertRaisesRegex(
            ValueError,
            str(MAX_STEERING_TOTAL_TOKENS),
        ):
            self.run_steering(pairs=pairs)

    def test_model_identity_is_checked_before_computation(self) -> None:
        with self.assertRaises(ModelChangedError):
            self.state.steering(
                contrast_pairs(),
                "Nový text v nezávislém rámci",
                16,
                1.0,
                1.0,
                5,
                6,
                10,
                "bf16",
                7,
                self.state.server_instance_id,
            )

    def test_swapping_poles_and_alpha_sign_is_equivalent(self) -> None:
        forward = self.run_steering(alpha=0.8)
        swapped = [
            {
                "negative_text": pair["positive_text"],
                "positive_text": pair["negative_text"],
            }
            for pair in contrast_pairs()
        ]
        reverse = self.run_steering(
            pairs=swapped,
            alpha=-0.8,
        )

        self.assertEqual(
            forward["right"]["top_tokens"],
            reverse["right"]["top_tokens"],
        )
        self.assertAlmostEqual(
            forward["js_divergence_bits"],
            reverse["js_divergence_bits"],
            places=12,
        )
        self.assertAlmostEqual(
            forward["total_variation"],
            reverse["total_variation"],
            places=12,
        )
        self.assertEqual(forward["gains"], reverse["gains"])
        self.assertEqual(forward["losses"], reverse["losses"])

    def test_worker_dispatches_steering_as_one_job(self) -> None:
        state = ModelState(model_path=Path("."), mock=True)
        state.start()
        try:
            result = state.steering(
                contrast_pairs(),
                "Nový text v nezávislém rámci",
                16,
                0.5,
                1.0,
                5,
                6,
                10,
                "mock",
                1,
                state.server_instance_id,
            )
            self.assertEqual(
                result["steering"]["execution"],
                "single_serial_worker_job",
            )
        finally:
            state.stop()
            if state.worker_thread is not None:
                state.worker_thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
