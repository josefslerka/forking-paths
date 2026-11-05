from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from apertus.garden.api.post_routes import POST_ROUTES
from apertus.garden.activation_patching import (
    ActivationPatchingError,
    build_reference,
    identity_control,
    recomputation_control,
    summarize_patch,
)
from apertus.garden.server import GardenHandler, ModelChangedError, ModelState


class ActivationPatchingAccountingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = np.array([4.0, 1.0, -0.5, 0.25])
        self.target = np.array([0.5, 2.5, -0.25, 0.5])
        self.reference = build_reference(
            self.source,
            self.target,
            0,
            1,
        )

    def test_target_source_and_overshoot_have_expected_recovery(self) -> None:
        target_metrics, _ = summarize_patch(
            self.reference,
            self.target,
        )
        source_metrics, _ = summarize_patch(
            self.reference,
            self.source,
        )
        overshoot = self.target + 1.5 * (self.source - self.target)
        overshoot_metrics, _ = summarize_patch(
            self.reference,
            overshoot,
        )

        self.assertAlmostEqual(target_metrics["contrast_recovery"], 0.0)
        self.assertAlmostEqual(source_metrics["contrast_recovery"], 1.0)
        self.assertAlmostEqual(
            overshoot_metrics["contrast_recovery"],
            1.5,
        )
        self.assertAlmostEqual(target_metrics["distribution_recovery"], 0.0)
        self.assertAlmostEqual(source_metrics["distribution_recovery"], 1.0)
        self.assertGreater(overshoot_metrics["js_to_target_bits"], 0.0)

    def test_identity_control_closes_exactly(self) -> None:
        control = identity_control(
            self.reference,
            self.target.copy(),
        )

        self.assertEqual(control["max_abs_logit_error"], 0.0)
        self.assertEqual(control["contrast_delta"], 0.0)
        self.assertEqual(control["js_to_target_bits"], 0.0)

    def test_matched_controls_remove_a_shared_numerical_shift(self) -> None:
        numerical_shift = np.array([0.4, -0.2, 0.1, -0.3])
        matched_source = self.source + numerical_shift
        matched_target = self.target + numerical_shift
        matched_reference = build_reference(
            matched_source,
            matched_target,
            0,
            1,
        )
        halfway = matched_target + 0.5 * (
            matched_source - matched_target
        )

        metrics, _ = summarize_patch(matched_reference, halfway)
        source_control = recomputation_control(
            matched_source,
            matched_source.copy(),
            0,
            1,
        )

        self.assertAlmostEqual(metrics["contrast_recovery"], 0.5)
        self.assertEqual(source_control["max_abs_logit_error"], 0.0)
        self.assertEqual(source_control["js_to_expected_bits"], 0.0)

    def test_tiny_baseline_contrast_difference_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ActivationPatchingError,
            "příliš malý",
        ):
            build_reference(
                self.source,
                self.source + 1e-10,
                0,
                1,
            )


class MockActivationPatchingServerTest(unittest.TestCase):
    SOURCE = (
        "V tomto příběhu je hlavním městem Česka Praha. "
        "Hlavním městem Česka je"
    )
    TARGET = (
        "V tomto příběhu je hlavním městem Česka Brno. "
        "Hlavním městem Česka je"
    )

    def setUp(self) -> None:
        self.state = ModelState(model_path=Path("."), mock=True)
        self.state.status = "ready"
        self.state.model_generation = 12

    def run_patching(self) -> dict:
        return self.state.activation_patching(
            self.SOURCE,
            self.TARGET,
            " invaze",
            " válka",
            [1, 16, 24, 32],
            [8, 9, 10, 11],
            6,
            "mock",
            12,
            self.state.server_instance_id,
        )

    def test_mock_response_has_causal_contract_and_identity_controls(
        self,
    ) -> None:
        result = self.run_patching()

        self.assertEqual(result["analysis_kind"], "activation_patching")
        self.assertEqual(
            result["method"]["hook"],
            "pre_decoder_block_residual",
        )
        self.assertEqual(
            result["method"]["interpretation"],
            "causal_sufficiency_not_necessity",
        )
        self.assertEqual(len(result["cells"]), 16)
        self.assertEqual(len(result["identity_controls"]), 4)
        self.assertTrue(result["identity_control_summary"]["passed"])
        self.assertEqual(result["execution"]["patched_variants"], 16)
        self.assertEqual(
            result["execution"]["identity_control_variants"],
            4,
        )
        self.assertTrue(
            result["execution"][
                "matched_source_target_controls_per_batch"
            ]
        )
        self.assertEqual(result["execution"]["patch_slots_per_batch"], 4)
        self.assertEqual(result["execution"]["fixed_total_batch_size"], 6)
        self.assertEqual(
            result["execution"]["matched_control_variants"],
            8,
        )
        self.assertTrue(result["execution"]["downstream_recomputed"])
        self.assertFalse(result["execution"]["kv_cache_used"])
        self.assertFalse(result["execution"]["global_cache_mutated"])
        self.assertGreater(
            result["baselines"]["js_source_target_bits"],
            0.0,
        )
        self.assertIn(
            "strongest_toward_source",
            result["extrema"],
        )
        first = result["cells"][0]
        self.assertIn("contrast_recovery", first)
        self.assertIn("distribution_recovery", first)
        self.assertEqual(
            first["matched_baselines"]["scope"],
            "same_fixed_forward_batch",
        )
        self.assertIn("state", first)
        self.assertIn("movers", first)
        self.assertEqual(
            result["method"]["cell_reference"],
            "matched_A_B_controls_in_same_fixed_forward_batch",
        )
        self.assertTrue(
            result["identity_control_summary"][
                "matched_controls_used_for_cells"
            ]
        )

    def test_real_path_uses_fixed_batch_and_matched_reference(self) -> None:
        baseline_source = inspect.getsource(
            ModelState._apertus_activation_patching_baselines
        )
        cells_source = inspect.getsource(
            ModelState._apertus_activation_patching_cells
        )

        self.assertIn(
            "ACTIVATION_PATCHING_FIXED_BATCH_SIZE",
            baseline_source,
        )
        self.assertIn("slot_validity", cells_source)
        self.assertIn("matched_reference", cells_source)
        self.assertIn(
            "logits[source_control_index]",
            cells_source,
        )
        self.assertIn(
            "logits[target_control_index]",
            cells_source,
        )

    def test_result_is_cached_by_full_intervention_grid(self) -> None:
        first = self.run_patching()
        second = self.run_patching()

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(len(self.state.activation_patching_cache), 1)

    def test_unequal_token_count_is_rejected_before_patch_forward(self) -> None:
        forward = MagicMock()
        self.state._mock_activation_patching_cells = forward  # type: ignore[method-assign]

        with self.assertRaisesRegex(ValueError, "stejně dlouhé"):
            self.state.activation_patching(
                "Krátký text",
                "O něco delší text",
                " invaze",
                " válka",
                [1],
                [1],
                5,
                "mock",
                12,
                self.state.server_instance_id,
            )

        forward.assert_not_called()

    def test_stale_model_identity_is_rejected_before_computation(self) -> None:
        local = MagicMock()
        self.state._activation_patching_local = local  # type: ignore[method-assign]

        with self.assertRaises(ModelChangedError):
            self.state.activation_patching(
                self.SOURCE,
                self.TARGET,
                " invaze",
                " válka",
                [1],
                [1],
                5,
                "bf16",
                12,
                self.state.server_instance_id,
            )

        local.assert_not_called()

    def test_handler_registers_page_and_api_routes(self) -> None:
        get_source = inspect.getsource(GardenHandler.do_GET)
        route = POST_ROUTES["/api/activation-patching"]
        controller_source = inspect.getsource(
            getattr(GardenHandler, route.controller)
        )

        self.assertIn('"/kauzalni-prenos"', get_source)
        self.assertIn("ACTIVATION_PATCHING_FILE", get_source)
        self.assertIn("state.activation_patching", controller_source)


if __name__ == "__main__":
    unittest.main()
