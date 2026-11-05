from __future__ import annotations

import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

from apertus.garden.server import (
    MAX_STEERING_TRAJECTORY_TOKENS,
    ModelChangedError,
    ModelState,
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


class MockSteeringTrajectoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ModelState(model_path=Path("."), mock=True)
        self.state.status = "ready"
        self.state.model_generation = 7

    def run_trajectory(
        self,
        *,
        pairs: list[dict[str, str]] | None = None,
        alpha_magnitude: float = 1.0,
        max_new_tokens: int = 8,
        generation_mode: str = "greedy",
        seed: int = 19,
    ) -> dict:
        return self.state.steering_trajectory(
            pairs if pairs is not None else contrast_pairs(),
            "Nový soused zaklepal. Moje odpověď bude",
            16,
            alpha_magnitude,
            1.0,
            max_new_tokens,
            generation_mode,
            seed,
            "mock",
            7,
            self.state.server_instance_id,
        )

    def test_response_contains_three_symmetric_branches_and_token_telemetry(
        self,
    ) -> None:
        result = self.run_trajectory()

        self.assertEqual(
            [branch["key"] for branch in result["branches"]],
            ["negative", "baseline", "positive"],
        )
        self.assertEqual(
            [branch["alpha"] for branch in result["branches"]],
            [-1.0, 0.0, 1.0],
        )
        self.assertTrue(
            result["steering"]["steering_applied_every_step"]
        )
        self.assertEqual(result["steering"]["direction_computations"], 1)
        self.assertEqual(
            result["steering"]["batch_strategy"],
            "three_branches_one_batched_forward",
        )
        self.assertIn(
            "no_global_mutation",
            result["steering"]["cache_policy"],
        )
        self.assertEqual(result["model_id"], "mock")
        self.assertEqual(result["model_generation"], 7)
        self.assertGreater(result["timing_ms"]["total"], 0.0)

        for branch in result["branches"]:
            self.assertTrue(branch["full_text"].startswith(result["test_text"]))
            self.assertEqual(
                branch["full_text"],
                result["test_text"] + branch["completion"],
            )
            self.assertLessEqual(
                branch["generated_token_count"],
                result["max_new_tokens"],
            )
            self.assertIn(
                branch["stop_reason"],
                {"sentence_end", "newline", "eos", "max_new_tokens"},
            )
            for expected_step, token in enumerate(
                branch["tokens"],
                start=1,
            ):
                self.assertEqual(token["step"], expected_step)
                self.assertEqual(token["step_index"], expected_step - 1)
                self.assertGreater(token["p_conditional"], 0.0)
                self.assertLessEqual(token["p_conditional"], 1.0)
                self.assertAlmostEqual(
                    token["surprisal_bits"],
                    -np.log2(token["p_conditional"]),
                )
                self.assertIn("entropy_bits", token)
                self.assertIn("logprob", token)
                self.assertIn("divergence_from_baseline", token)

    def test_divergence_is_only_reported_while_contexts_match(self) -> None:
        result = self.run_trajectory(max_new_tokens=6)
        first = result["first_divergence"]
        self.assertEqual(first["any_step"], 1)
        self.assertEqual(first["event"]["step_index"], 0)
        self.assertTrue(first["event"]["all_equal_before"])

        for branch in (result["branches"][0], result["branches"][2]):
            divergence_step = branch[
                "first_divergence_from_baseline_step"
            ]
            self.assertIsNotNone(divergence_step)
            assert divergence_step is not None
            at_divergence = branch["tokens"][divergence_step - 1][
                "divergence_from_baseline"
            ]
            self.assertTrue(at_divergence["context_aligned"])
            self.assertIsNotNone(at_divergence["js_divergence_bits"])
            if len(branch["tokens"]) > divergence_step:
                after = branch["tokens"][divergence_step][
                    "divergence_from_baseline"
                ]
                self.assertFalse(after["context_aligned"])
                self.assertIsNone(after["js_divergence_bits"])
                self.assertIsNone(after["total_variation"])

    def test_greedy_and_seeded_sampling_are_reproducible(self) -> None:
        greedy_a = self.run_trajectory(seed=1)
        greedy_b = self.run_trajectory(seed=999)
        self.assertEqual(
            [
                branch["generated_token_ids"]
                for branch in greedy_a["branches"]
            ],
            [
                branch["generated_token_ids"]
                for branch in greedy_b["branches"]
            ],
        )
        self.assertEqual(
            greedy_a["first_divergence"],
            greedy_b["first_divergence"],
        )

        sampled_a = self.run_trajectory(
            generation_mode="sample",
            seed=42,
        )
        sampled_b = self.run_trajectory(
            generation_mode="sample",
            seed=42,
        )
        self.assertEqual(
            [
                branch["generated_token_ids"]
                for branch in sampled_a["branches"]
            ],
            [
                branch["generated_token_ids"]
                for branch in sampled_b["branches"]
            ],
        )
        self.assertEqual(
            sampled_a["first_divergence"],
            sampled_b["first_divergence"],
        )

    def test_direction_reversal_swaps_negative_and_positive_trajectories(
        self,
    ) -> None:
        forward = self.run_trajectory(max_new_tokens=5)
        reversed_pairs = [
            {
                "negative_text": pair["positive_text"],
                "positive_text": pair["negative_text"],
            }
            for pair in contrast_pairs()
        ]
        reverse = self.run_trajectory(
            pairs=reversed_pairs,
            max_new_tokens=5,
        )

        self.assertEqual(
            forward["branches"][0]["generated_token_ids"],
            reverse["branches"][2]["generated_token_ids"],
        )
        self.assertEqual(
            forward["branches"][2]["generated_token_ids"],
            reverse["branches"][0]["generated_token_ids"],
        )
        self.assertEqual(
            forward["branches"][1]["generated_token_ids"],
            reverse["branches"][1]["generated_token_ids"],
        )

    def test_request_does_not_mutate_persistent_garden_caches(self) -> None:
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

        self.run_trajectory(max_new_tokens=4)

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

    def test_stop_detection_and_limits(self) -> None:
        self.assertEqual(
            self.state._trajectory_stop_reason(3, ".", set()),
            "sentence_end",
        )
        self.assertEqual(
            self.state._trajectory_stop_reason(3, ".\n", set()),
            "sentence_end",
        )
        self.assertEqual(
            self.state._trajectory_stop_reason(3, "\n", set()),
            "newline",
        )
        self.assertEqual(
            self.state._trajectory_stop_reason(3, "", {3}),
            "eos",
        )
        with self.assertRaisesRegex(
            ValueError,
            str(MAX_STEERING_TRAJECTORY_TOKENS),
        ):
            self.run_trajectory(
                max_new_tokens=MAX_STEERING_TRAJECTORY_TOKENS + 1
            )
        with self.assertRaisesRegex(ValueError, "alpha_magnitude"):
            self.run_trajectory(alpha_magnitude=0.0)

    def test_model_identity_is_checked_before_computation(self) -> None:
        with self.assertRaises(ModelChangedError):
            self.state.steering_trajectory(
                contrast_pairs(),
                "Nový text v nezávislém rámci",
                16,
                1.0,
                1.0,
                8,
                "greedy",
                0,
                "bf16",
                7,
                self.state.server_instance_id,
            )

    def test_worker_dispatches_trajectory_as_one_job(self) -> None:
        state = ModelState(model_path=Path("."), mock=True)
        state.start()
        try:
            result = state.steering_trajectory(
                contrast_pairs(),
                "Nový text v nezávislém rámci",
                16,
                0.5,
                1.0,
                4,
                "greedy",
                0,
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
