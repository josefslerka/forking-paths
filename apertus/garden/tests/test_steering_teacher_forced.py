from __future__ import annotations

import inspect
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

from apertus.garden.api.post_routes import POST_ROUTES
from apertus.garden.server import GardenHandler, ModelChangedError, ModelState


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


def continuation_pairs() -> list[dict[str, str]]:
    return [
        {
            "label": "pozdrav",
            "negative_text": " strohá a odmítavá.",
            "positive_text": " vlídná a vstřícná.",
        },
        {
            "label": "nabídka",
            "negative_text": " že nemám zájem.",
            "positive_text": " že rád pomohu.",
        },
    ]


class MockTeacherForcedSteeringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ModelState(model_path=Path("."), mock=True)
        self.state.status = "ready"
        self.state.model_generation = 7

    def run_aligned(self, continuation_text: str = " velmi laskavá.") -> dict:
        return self.state.steering_aligned_path(
            contrast_pairs(),
            "Nový soused zaklepal. Moje odpověď bude",
            continuation_text,
            16,
            1.0,
            1.0,
            "mock",
            7,
            self.state.server_instance_id,
        )

    def run_phrase_score(
        self,
        pairs: list[dict[str, str]] | None = None,
    ) -> dict:
        return self.state.steering_phrase_score(
            contrast_pairs(),
            "Nový soused zaklepal. Moje odpověď bude",
            pairs if pairs is not None else continuation_pairs(),
            16,
            1.0,
            1.0,
            "mock",
            7,
            self.state.server_instance_id,
        )

    def test_aligned_path_scores_identical_tokens_in_aligned_contexts(
        self,
    ) -> None:
        result = self.run_aligned()

        self.assertEqual(
            result["analysis_kind"],
            "steering_aligned_path",
        )
        self.assertEqual(
            [branch["key"] for branch in result["branches"]],
            ["negative", "baseline", "positive"],
        )
        self.assertEqual(
            [branch["alpha"] for branch in result["branches"]],
            [-1.0, 0.0, 1.0],
        )
        self.assertTrue(result["steering"]["teacher_forced"])
        self.assertTrue(
            result["steering"]["shared_context_at_every_step"]
        )
        self.assertEqual(result["steering"]["direction_computations"], 1)
        self.assertEqual(
            result["steering"]["batch_strategy"],
            "three_alphas_one_batched_forward",
        )
        token_ids = result["continuation_token_ids"]
        self.assertGreater(len(token_ids), 0)
        for branch in result["branches"]:
            self.assertEqual(branch["continuation_token_ids"], token_ids)
            self.assertEqual(branch["token_count"], len(token_ids))
            self.assertAlmostEqual(
                branch["mean_logprob"],
                branch["cumulative_logprob"] / len(token_ids),
            )
            self.assertEqual(
                branch["total_logprob"],
                branch["cumulative_logprob"],
            )
            self.assertAlmostEqual(
                branch["length_normalized_logprob"],
                branch["mean_logprob"],
            )
            self.assertAlmostEqual(
                branch["perplexity"],
                np.exp(-branch["mean_logprob"]),
            )
            previous_prefix = result["test_text"]
            for expected_step, token in enumerate(
                branch["tokens"],
                start=1,
            ):
                self.assertEqual(token["step"], expected_step)
                self.assertEqual(
                    token["token_id"],
                    token_ids[expected_step - 1],
                )
                self.assertGreater(token["p_conditional"], 0.0)
                self.assertLessEqual(token["p_conditional"], 1.0)
                self.assertGreaterEqual(token["rank"], 1)
                self.assertTrue(
                    token["divergence_from_baseline"]["context_aligned"]
                )
                self.assertIsNotNone(
                    token["divergence_from_baseline"][
                        "js_divergence_bits"
                    ]
                )
                self.assertIsNotNone(
                    token["divergence_from_baseline"][
                        "total_variation"
                    ]
                )
                self.assertEqual(
                    token["prefix_before_text"],
                    previous_prefix,
                )
                self.assertEqual(
                    token["prefix_after_text"],
                    token["prefix_before_text"] + token["text"],
                )
                previous_prefix = token["prefix_after_text"]

    def test_aligned_path_does_not_mutate_global_caches(self) -> None:
        cached_logits = np.array([1.0, 2.0], dtype=np.float32)
        self.state.logits_cache = OrderedDict(
            [((1, 2), cached_logits.copy())]
        )
        self.state.active_ids = [1, 2]
        active_cache = object()
        self.state.active_cache = active_cache
        self.state.active_logits = cached_logits.copy()

        self.run_aligned()

        self.assertEqual(list(self.state.logits_cache), [(1, 2)])
        np.testing.assert_array_equal(
            self.state.logits_cache[(1, 2)],
            cached_logits,
        )
        self.assertEqual(self.state.active_ids, [1, 2])
        self.assertIs(self.state.active_cache, active_cache)
        np.testing.assert_array_equal(
            self.state.active_logits,
            cached_logits,
        )

    def test_phrase_score_returns_sequence_and_contrast_metrics(self) -> None:
        result = self.run_phrase_score()

        self.assertEqual(
            result["analysis_kind"],
            "steering_phrase_score",
        )
        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(result["steering"]["direction_computations"], 1)
        self.assertEqual(
            result["summary"]["contrast_definition"],
            "positive_mean_logprob_minus_negative_mean_logprob",
        )
        self.assertEqual(
            result["summary"]["alpha_order"],
            ["negative", "baseline", "positive"],
        )
        for item in result["items"]:
            self.assertIn(item["label"], {"pozdrav", "nabídka"})
            self.assertEqual(len(item["contrasts"]), 3)
            for pole in ("negative", "positive"):
                sequence = item[pole]
                self.assertGreater(sequence["token_count"], 0)
                self.assertEqual(len(sequence["scores"]), 3)
                for score in sequence["scores"]:
                    self.assertEqual(
                        score["token_count"],
                        sequence["token_count"],
                    )
                    self.assertAlmostEqual(
                        score["mean_logprob"],
                        score["cumulative_logprob"]
                        / score["token_count"],
                    )
                    self.assertEqual(
                        score["total_logprob"],
                        score["cumulative_logprob"],
                    )
            expected = [
                item["positive"]["scores"][index]["mean_logprob"]
                - item["negative"]["scores"][index]["mean_logprob"]
                for index in range(3)
            ]
            self.assertEqual(
                [
                    value["positive_minus_negative_mean_logprob"]
                    for value in item["contrasts"]
                ],
                expected,
            )
            diagnostic = item["monotonicity"]
            self.assertEqual(
                diagnostic["strictly_increasing"],
                expected[1] - expected[0] > diagnostic["tolerance"]
                and expected[2] - expected[1] > diagnostic["tolerance"],
            )
            self.assertEqual(
                diagnostic["nondecreasing"],
                expected[1] - expected[0] >= -diagnostic["tolerance"]
                and expected[2] - expected[1] >= -diagnostic["tolerance"],
            )

        summary = result["summary"]["monotonicity"]
        self.assertEqual(summary["pair_count"], 2)
        self.assertLessEqual(summary["pairs_nondecreasing"], 2)
        self.assertLessEqual(summary["pairs_strictly_increasing"], 2)

    def test_monotonicity_tolerance_is_explicit(self) -> None:
        strict = self.state._steering_monotonicity([0.0, 0.1, 0.2])
        flat = self.state._steering_monotonicity(
            [0.0, 0.25e-4, 0.5e-4]
        )
        falling = self.state._steering_monotonicity([0.0, -0.1, 0.2])

        self.assertEqual(strict["tolerance"], 1e-4)
        self.assertTrue(strict["strictly_increasing"])
        self.assertTrue(strict["nondecreasing"])
        self.assertFalse(flat["strictly_increasing"])
        self.assertTrue(flat["nondecreasing"])
        self.assertFalse(falling["nondecreasing"])

    def test_validation_rejects_duplicate_pairs_and_stale_identity(self) -> None:
        duplicated = [continuation_pairs()[0], continuation_pairs()[0]]
        with self.assertRaisesRegex(ValueError, "duplicitní"):
            self.run_phrase_score(duplicated)

        with self.assertRaises(ModelChangedError):
            self.state.steering_aligned_path(
                contrast_pairs(),
                "Nový soused zaklepal. Moje odpověď bude",
                " vlídná.",
                16,
                1.0,
                1.0,
                "bf16",
                7,
                self.state.server_instance_id,
            )

    def test_trajectory_tokens_include_exact_handoff_prefixes(self) -> None:
        result = self.state.steering_trajectory(
            contrast_pairs(),
            "Nový soused zaklepal. Moje odpověď bude",
            16,
            1.0,
            1.0,
            4,
            "greedy",
            0,
            "mock",
            7,
            self.state.server_instance_id,
        )
        for branch in result["branches"]:
            previous = result["test_text"]
            for token in branch["tokens"]:
                self.assertEqual(token["prefix_before_text"], previous)
                self.assertEqual(
                    token["prefix_after_text"],
                    token["prefix_before_text"] + token["text"],
                )
                previous = token["prefix_after_text"]

    def test_worker_dispatches_each_analysis_as_one_job(self) -> None:
        state = ModelState(model_path=Path("."), mock=True)
        state.start()
        try:
            aligned = state.steering_aligned_path(
                contrast_pairs(),
                "Nový soused zaklepal. Moje odpověď bude",
                " vlídná.",
                16,
                0.5,
                1.0,
                "mock",
                1,
                state.server_instance_id,
            )
            phrases = state.steering_phrase_score(
                contrast_pairs(),
                "Nový soused zaklepal. Moje odpověď bude",
                continuation_pairs(),
                16,
                0.5,
                1.0,
                "mock",
                1,
                state.server_instance_id,
            )
            self.assertEqual(
                aligned["steering"]["execution"],
                "single_serial_worker_job",
            )
            self.assertEqual(
                phrases["steering"]["execution"],
                "single_serial_worker_job",
            )
        finally:
            state.stop()
            if state.worker_thread is not None:
                state.worker_thread.join(timeout=2)


class BoundaryTokenizerTest(unittest.TestCase):
    class Tokenizer:
        all_special_ids: list[int] = []

        def encode(
            self,
            text: str,
            *,
            add_special_tokens: bool = True,
        ) -> list[int]:
            mapping = {
                "prefix": [1, 10],
                "prefixsuffix": [1, 11],
            }
            return mapping[text]

        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool = True,
        ) -> str:
            del skip_special_tokens
            return "decoded:" + ",".join(str(value) for value in token_ids)

    def test_boundary_retokenization_is_rejected(self) -> None:
        state = ModelState(model_path=Path("."), mock=False)
        state.tokenizer = self.Tokenizer()
        with self.assertRaisesRegex(ValueError, "hranici"):
            state._prepare_teacher_forced_continuation(
                "prefix",
                [1, 10],
                "suffix",
            )


class TeacherForcedHttpContractTest(unittest.TestCase):
    def test_request_parser_validates_shared_fields(self) -> None:
        fields = GardenHandler._teacher_forced_request_fields(
            {
                "contrast_pairs": contrast_pairs(),
                "test_text": "Nový soused zaklepal. Moje odpověď bude",
                "layer": 16,
                "alpha_magnitude": 0.5,
                "temperature": 1.0,
                "model_id": "mock",
                "model_generation": 1,
                "server_instance_id": "server",
            }
        )
        self.assertEqual(fields[1], "Nový soused zaklepal. Moje odpověď bude")
        self.assertEqual(fields[2:5], (16, 0.5, 1.0))
        self.assertEqual(fields[5:], ("mock", 1, "server"))

    def test_both_teacher_forced_routes_dispatch_to_state(self) -> None:
        for route, method in (
            (
                "/api/steering/aligned-path",
                "self.state.steering_aligned_path(",
            ),
            (
                "/api/steering/phrase-score",
                "self.state.steering_phrase_score(",
            ),
        ):
            spec = POST_ROUTES[route]
            source = inspect.getsource(
                getattr(GardenHandler, spec.controller)
            )
            self.assertIn(method, source)


if __name__ == "__main__":
    unittest.main()
