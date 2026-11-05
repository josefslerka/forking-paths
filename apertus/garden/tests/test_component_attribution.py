from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from apertus.garden.api.post_routes import POST_ROUTES
from apertus.garden.component_attribution import (
    ComponentAttributionError,
    decompose_residual_contrast,
)
from apertus.garden.server import GardenHandler, ModelChangedError, ModelState


class ResidualContrastAccountingTest(unittest.TestCase):
    def synthetic_inputs(self) -> dict:
        rng = np.random.default_rng(20260802)
        layers = 4
        width = 12
        embedding = rng.normal(size=width)
        attention = rng.normal(scale=0.2, size=(layers, width))
        mlp = rng.normal(scale=0.3, size=(layers, width))
        final = embedding + attention.sum(axis=0) + mlp.sum(axis=0)
        return {
            "embedding": embedding,
            "attention_updates": attention,
            "mlp_updates": mlp,
            "final_residual": final,
            "norm_weight": rng.normal(1.0, 0.05, size=width),
            "lm_head_row_a": rng.normal(size=width),
            "lm_head_row_b": rng.normal(size=width),
            "epsilon": 1e-5,
        }

    def test_fixed_final_rmsnorm_accounting_closes_the_contrast(self) -> None:
        result = decompose_residual_contrast(**self.synthetic_inputs())

        self.assertEqual(result["layer_count"], 4)
        self.assertEqual(
            result["method"]["contrast_definition"],
            "logit(A) - logit(B)",
        )
        self.assertEqual(
            result["method"]["status"],
            "descriptive_not_causal",
        )
        self.assertAlmostEqual(
            result["final"]["contrast"],
            result["final"]["reconstructed_contrast"],
            places=12,
        )
        self.assertAlmostEqual(
            result["final"]["reconstruction_error"],
            0.0,
            places=12,
        )
        self.assertEqual(
            [entry["layer"] for entry in result["layers"]],
            [1, 2, 3, 4],
        )
        self.assertEqual(len(result["components"]), 11)

    def test_rounding_remainder_is_explicit_instead_of_hidden(self) -> None:
        inputs = self.synthetic_inputs()
        inputs["final_residual"] = (
            np.asarray(inputs["final_residual"]).copy()
        )
        inputs["final_residual"][0] += 0.125

        result = decompose_residual_contrast(**inputs)

        self.assertGreater(
            result["diagnostics"]["residual_rounding_norm"],
            0.12,
        )
        self.assertEqual(
            result["residual_rounding"]["kind"],
            "residual_rounding",
        )
        self.assertAlmostEqual(
            result["final"]["reconstruction_error"],
            0.0,
            places=12,
        )

    def test_model_numeric_difference_is_a_named_closure(self) -> None:
        inputs = self.synthetic_inputs()
        result = decompose_residual_contrast(
            **inputs,
            actual_logit_a=3.0,
            actual_logit_b=-2.0,
        )

        self.assertEqual(result["final"]["contrast"], 5.0)
        self.assertEqual(
            result["numerical_closure"]["kind"],
            "numerical_closure",
        )
        self.assertAlmostEqual(
            result["final"]["reconstructed_contrast"],
            5.0,
            places=12,
        )

    def test_invalid_component_shapes_are_rejected(self) -> None:
        inputs = self.synthetic_inputs()
        inputs["mlp_updates"] = np.zeros((3, 12))

        with self.assertRaises(ComponentAttributionError):
            decompose_residual_contrast(**inputs)


class MockComponentAttributionServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ModelState(model_path=Path("."), mock=True)
        self.state.status = "ready"
        self.state.model_generation = 9

    def run_attribution(self) -> dict:
        return self.state.component_attribution(
            "Hlavním městem České republiky je",
            " invaze",
            " válka",
            "mock",
            9,
            self.state.server_instance_id,
        )

    def test_mock_response_has_full_component_contract(self) -> None:
        result = self.run_attribution()

        self.assertEqual(result["kind"], "component_attribution")
        self.assertEqual(result["model_id"], "mock")
        self.assertEqual(result["contrast"]["definition"], "logit(A) - logit(B)")
        self.assertEqual(len(result["layers"]), 32)
        self.assertEqual(result["execution"]["forward_passes"], 1)
        self.assertFalse(result["execution"]["attention_weights_used"])
        self.assertEqual(
            result["normalization"]["kind"],
            "fixed_final_rmsnorm",
        )
        self.assertAlmostEqual(
            result["final"]["reconstruction_error"],
            0.0,
            places=12,
        )
        for layer_number, layer in enumerate(result["layers"], start=1):
            self.assertEqual(layer["layer"], layer_number)
            self.assertEqual(layer["attention"]["kind"], "attention")
            self.assertEqual(layer["mlp"]["kind"], "mlp")
            self.assertIn("local_reader_after_layer", layer)
        self.assertEqual(len(result["strongest_components"]), 12)
        self.assertTrue(
            any(
                warning["code"] == "descriptive_not_causal"
                for warning in result["warnings"]
            )
        )

    def test_cache_is_keyed_by_contrast_and_prefix(self) -> None:
        first = self.run_attribution()
        second = self.run_attribution()
        reversed_result = self.state.component_attribution(
            "Hlavním městem České republiky je",
            " válka",
            " invaze",
            "mock",
            9,
            self.state.server_instance_id,
        )

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertFalse(reversed_result["cached"])
        self.assertEqual(len(self.state.component_attribution_cache), 2)
        self.assertAlmostEqual(
            first["final"]["contrast"],
            -reversed_result["final"]["contrast"],
            places=9,
        )

    def test_same_token_is_rejected_before_forward(self) -> None:
        forward = MagicMock()
        self.state._mock_component_attribution_tensors = forward  # type: ignore[method-assign]

        with self.assertRaisesRegex(ValueError, "tentýž token"):
            self.state.component_attribution(
                "Prefix",
                " invaze",
                "#0",
                "mock",
                9,
                self.state.server_instance_id,
            )

        forward.assert_not_called()

    def test_stale_model_identity_is_rejected_before_computation(self) -> None:
        local = MagicMock()
        self.state._component_attribution_local = local  # type: ignore[method-assign]

        with self.assertRaises(ModelChangedError):
            self.state.component_attribution(
                "Prefix",
                " invaze",
                " válka",
                "bf16",
                9,
                self.state.server_instance_id,
            )

        local.assert_not_called()

    def test_handler_registers_page_and_api_routes(self) -> None:
        get_source = inspect.getsource(GardenHandler.do_GET)
        route = POST_ROUTES["/api/component-attribution"]
        controller_source = inspect.getsource(
            getattr(GardenHandler, route.controller)
        )

        self.assertIn('"/atribuce"', get_source)
        self.assertIn("COMPONENT_ATTRIBUTION_FILE", get_source)
        self.assertIn("state.component_attribution", controller_source)


if __name__ == "__main__":
    unittest.main()
