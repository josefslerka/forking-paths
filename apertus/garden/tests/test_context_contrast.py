from __future__ import annotations

import inspect
import math
import unittest
from pathlib import Path

from apertus.garden.api.post_routes import POST_ROUTES
from apertus.garden.server import (
    GardenHandler,
    ModelChangedError,
    ModelState,
)


class ContextContrastMathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ModelState(Path("."), mock=True)
        self.state.status = "ready"
        self.state.model_generation = 7

    def run_contrast(self) -> dict:
        return self.state.context_contrast(
            "Hlavním městem České republiky je",
            "Českou metropolí je",
            " Praha leží na Vltavě.",
            5,
            "mock",
            7,
            self.state.server_instance_id,
        )

    def test_two_forwards_measure_target_field_and_persistence(self) -> None:
        result = self.run_contrast()

        self.assertEqual(result["analysis_kind"], "context_contrast")
        self.assertEqual(result["temperature"], 1.0)
        self.assertEqual(result["probability_kind"], "raw_model")
        self.assertEqual(result["execution"]["forward_passes"], 2)
        self.assertFalse(result["execution"]["generation"])
        self.assertFalse(result["execution"]["steering"])
        token_count = result["tokenization"]["continuation_token_count"]
        self.assertGreater(token_count, 1)
        self.assertEqual(
            len(result["persistence"]["states"]),
            token_count + 1,
        )
        self.assertEqual(
            len(result["access_curve"]["a"]),
            result["tokenization"]["prefix_a_token_count"] + 1,
        )
        self.assertEqual(
            len(result["access_curve"]["b"]),
            result["tokenization"]["prefix_b_token_count"] + 1,
        )
        self.assertEqual(
            result["access_curve"]["scope"],
            "first_continuation_token_only",
        )
        self.assertTrue(
            0.0 <= result["root_field"]["js_divergence_bits"] <= 1.0
        )

    def test_sequence_advantage_is_sum_of_aligned_token_advantages(self) -> None:
        result = self.run_contrast()
        token_advantages = [
            state["next_token"][
                "information_advantage_a_over_b_bits"
            ]
            for state in result["persistence"]["states"]
            if state["next_token"] is not None
        ]
        total = math.fsum(token_advantages)

        self.assertAlmostEqual(
            total,
            result["target_score"][
                "information_advantage_a_over_b_bits"
            ],
            places=12,
        )
        self.assertAlmostEqual(
            total,
            result["persistence"]["summary"][
                "final_cumulative_information_advantage_bits"
            ],
            places=12,
        )
        self.assertEqual(
            result["definition"]["equivalent_surprisal_difference"],
            "I_B - I_A",
        )

    def test_repeat_uses_bounded_result_cache_without_global_kv_mutation(
        self,
    ) -> None:
        before = list(self.state.logits_cache.items())
        first = self.run_contrast()
        second = self.run_contrast()

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(before, list(self.state.logits_cache.items()))
        self.assertEqual(len(self.state.context_contrast_cache), 1)
        self.assertEqual(
            first["target_score"],
            second["target_score"],
        )

    def test_stale_identity_is_rejected_before_computation(self) -> None:
        with self.assertRaises(ModelChangedError):
            self.state.context_contrast(
                "A",
                "B",
                " C",
                5,
                "mock",
                6,
                self.state.server_instance_id,
            )


class ContextContrastHttpContractTest(unittest.TestCase):
    def test_request_validation_requires_identity_and_bounded_fields(self) -> None:
        fields = GardenHandler._context_contrast_request_fields(
            {
                "prefix_a": "A",
                "prefix_b": "B",
                "continuation": " C",
                "top_k": 5,
                "model_id": "hybrid",
                "model_generation": 3,
                "server_instance_id": "server",
            }
        )
        self.assertEqual(fields[:4], ("A", "B", " C", 5))
        with self.assertRaises(ValueError):
            GardenHandler._context_contrast_request_fields(
                {
                    "prefix_a": "A",
                    "prefix_b": "B",
                    "continuation": " C",
                    "top_k": 2,
                    "model_id": "hybrid",
                    "model_generation": 3,
                    "server_instance_id": "server",
                }
            )

    def test_page_and_api_routes_dispatch_to_context_contrast(self) -> None:
        get_source = inspect.getsource(GardenHandler.do_GET)
        route = POST_ROUTES["/api/context-contrast"]
        controller_source = inspect.getsource(
            getattr(GardenHandler, route.controller)
        )
        worker_source = inspect.getsource(ModelState._worker_loop)

        self.assertIn('"/kontrast"', get_source)
        self.assertIn("CONTEXT_CONTRAST_FILE", get_source)
        self.assertIn("self.state.context_contrast(", controller_source)
        self.assertIn('operation == "context_contrast"', worker_source)
        self.assertIn("_context_contrast_local", worker_source)


if __name__ == "__main__":
    unittest.main()
