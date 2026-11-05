from __future__ import annotations

import inspect
import threading
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

from apertus.garden.api.post_routes import POST_ROUTES
from apertus.garden.jacobian_lens import (
    JacobianLensChangedError,
    JacobianLensIncompatibleError,
    TransferProbeError,
)
from apertus.garden.server import (
    GardenHandler,
    ModelChangedError,
    ModelState,
)


MOCK_ARTIFACT_ID = "mock-average-jacobian-v1"
SOURCE_LAYERS = [4, 16, 17]


class MockJacobianLensServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ModelState(model_path=Path("."), mock=True)
        self.state.status = "ready"
        self.state.model_generation = 7

    def run_lens(
        self,
        *,
        text: str = "Hlavním městem České republiky je",
        source_layers: list[int] | None = None,
        top_k: int = 5,
    ) -> dict:
        return self.state.jacobian_lens(
            text,
            top_k,
            SOURCE_LAYERS if source_layers is None else source_layers,
            MOCK_ARTIFACT_ID,
            "mock",
            7,
            self.state.server_instance_id,
        )

    def test_mock_response_has_paired_reader_contract(self) -> None:
        result = self.run_lens()

        self.assertEqual(result["lens_kind"], "raw_jacobian_comparison")
        self.assertEqual(
            result["reader"]["kind"],
            "raw_vs_jacobian_lens",
        )
        self.assertEqual(result["artifact_id"], MOCK_ARTIFACT_ID)
        self.assertEqual(result["source_layers"], SOURCE_LAYERS)
        self.assertEqual(result["target_layer"], 32)
        self.assertEqual(result["model_id"], "mock")
        self.assertEqual(result["model_generation"], 7)
        self.assertEqual(
            result["server_instance_id"],
            self.state.server_instance_id,
        )
        self.assertTrue(result["transfer_probe"]["passed"])
        self.assertFalse(result["cached"])

        self.assertEqual(
            [entry["layer"] for entry in result["layers"]],
            SOURCE_LAYERS,
        )
        for entry in result["layers"]:
            self.assertEqual(
                set(entry),
                {"layer", "raw", "jacobian", "comparison"},
            )
            self.assertEqual(len(entry["raw"]["candidates"]), 5)
            self.assertEqual(len(entry["jacobian"]["candidates"]), 5)
            for side in ("raw", "jacobian"):
                for rank, candidate in enumerate(
                    entry[side]["candidates"],
                    start=1,
                ):
                    self.assertEqual(candidate["rank"], rank)
                    self.assertGreaterEqual(candidate["surprisal_bits"], 0.0)
                    self.assertGreaterEqual(
                        candidate["logit_gap_from_top1_bits"],
                        0.0,
                    )
                    self.assertGreaterEqual(candidate["final_probability"], 0.0)
                    self.assertGreaterEqual(candidate["final_rank"], 1)
                    self.assertGreaterEqual(
                        candidate["final_surprisal_bits"],
                        0.0,
                    )
                self.assertEqual(
                    entry[side]["candidates"][0][
                        "logit_gap_from_top1_bits"
                    ],
                    0.0,
                )
            self.assertIn(
                "js_jacobian_minus_raw_bits",
                entry["comparison"],
            )

    def test_response_can_return_twenty_candidates_per_reader(self) -> None:
        result = self.run_lens(top_k=20)

        self.assertEqual(result["top_k"], 20)
        for entry in result["layers"]:
            self.assertEqual(len(entry["raw"]["candidates"]), 20)
            self.assertEqual(len(entry["jacobian"]["candidates"]), 20)

    def test_cache_identity_includes_selected_layers(self) -> None:
        first = self.run_lens(source_layers=[4])
        repeated = self.run_lens(source_layers=[4])
        other_slice = self.run_lens(source_layers=[16])

        self.assertFalse(first["cached"])
        self.assertTrue(repeated["cached"])
        self.assertFalse(other_slice["cached"])
        self.assertEqual(len(self.state.jacobian_lens_cache), 2)
        self.assertEqual(first["source_layers"], [4])
        self.assertEqual(other_slice["source_layers"], [16])

    def test_invalid_source_layer_is_rejected_before_forward(self) -> None:
        forward = MagicMock()
        self.state._mock_jacobian_lens_logits = forward  # type: ignore[method-assign]

        with self.assertRaisesRegex(ValueError, "podmnožinou"):
            self.run_lens(source_layers=[5])

        forward.assert_not_called()

    def test_mock_rejects_a_stale_artifact_identity(self) -> None:
        with self.assertRaises(JacobianLensChangedError):
            self.state._jacobian_lens_local(
                "Text",
                5,
                [4],
                "different-artifact",
            )

    def test_model_identity_is_checked_before_local_computation(self) -> None:
        local = MagicMock()
        self.state._jacobian_lens_local = local  # type: ignore[method-assign]

        cases = (
            {
                "expected_model_id": "bf16",
                "expected_generation": 7,
                "expected_server_instance_id": self.state.server_instance_id,
            },
            {
                "expected_model_id": "mock",
                "expected_generation": 8,
                "expected_server_instance_id": self.state.server_instance_id,
            },
            {
                "expected_model_id": "mock",
                "expected_generation": 7,
                "expected_server_instance_id": "restarted-server",
            },
        )
        for identity in cases:
            with self.subTest(identity=identity):
                with self.assertRaises(ModelChangedError):
                    self.state.jacobian_lens(
                        "Text",
                        5,
                        [4],
                        MOCK_ARTIFACT_ID,
                        identity["expected_model_id"],
                        identity["expected_generation"],
                        identity["expected_server_instance_id"],
                    )

        local.assert_not_called()

    def test_hybrid_gate_precedes_catalog_load_and_mlx_imports(self) -> None:
        state = ModelState(model_path=Path("."), mock=True)
        state.mock = False
        state.active_model_id = "hybrid"
        state.status = "ready"
        catalog = MagicMock()
        state.jacobian_lens_catalog = catalog

        with self.assertRaisesRegex(
            JacobianLensIncompatibleError,
            "pouze pro BF16",
        ):
            state._jacobian_lens_local(
                "Text",
                5,
                [4],
                MOCK_ARTIFACT_ID,
            )

        catalog.load.assert_not_called()

    def test_cached_failed_probe_never_enables_the_reader(self) -> None:
        state = ModelState(model_path=Path("."), mock=True)
        state.mock = False
        state.active_model_id = "bf16"
        state.status = "ready"
        state.model = object()
        weights_sha = "a" * 64
        artifact_id = "cached-failed-probe"
        manifest = {
            "artifact_id": artifact_id,
            "reader": {
                "source_layers": [4],
                "target_layer": 32,
            },
            "files": {"weights": {"sha256": weights_sha}},
        }
        catalog = MagicMock()
        catalog.load.return_value = (manifest, {}, {}, {})
        state.jacobian_lens_catalog = catalog
        state._model_content_fingerprint_local = MagicMock(  # type: ignore[method-assign]
            return_value="b" * 64
        )
        state.jacobian_probe_report_cache[
            (artifact_id, weights_sha, "b" * 64)
        ] = {
            "kind": "pytorch_bf16_to_mlx_runtime_probe",
            "passed": False,
        }
        state._run_jacobian_transfer_probe_local = MagicMock()  # type: ignore[method-assign]
        state._forward_layer_states = MagicMock()  # type: ignore[method-assign]

        with patch(
            "apertus.garden.server.validate_bf16_runtime_contract"
        ), self.assertRaisesRegex(TransferProbeError, "zůstává vypnutá"):
            state._jacobian_lens_local(
                "Text",
                5,
                [4],
                artifact_id,
            )

        state._run_jacobian_transfer_probe_local.assert_not_called()
        state._forward_layer_states.assert_not_called()

    def test_worker_dispatches_jacobian_lens_as_one_serial_job(self) -> None:
        state = ModelState(model_path=Path("."), mock=True)
        original = state._jacobian_lens_local
        worker_names: list[str] = []

        def tracked(*arguments):
            worker_names.append(threading.current_thread().name)
            return original(*arguments)

        state._jacobian_lens_local = tracked  # type: ignore[method-assign]
        state.start()
        try:
            result = state.jacobian_lens(
                "Text",
                5,
                [4],
                MOCK_ARTIFACT_ID,
                "mock",
                1,
                state.server_instance_id,
            )
            self.assertEqual(result["source_layers"], [4])
            self.assertEqual(worker_names, ["apertus-model-worker"])
            self.assertIsNone(state.active_operation)
            self.assertIsNone(state.outstanding_operation)
        finally:
            state.stop()
            if state.worker_thread is not None:
                state.worker_thread.join(timeout=2)

    def test_model_release_clears_jacobian_runtime_state(self) -> None:
        self.state.jacobian_lens_cache = OrderedDict(
            [((7, "artifact", "weights", "digest", "text", 5, (4,)), {})]
        )
        self.state.jacobian_probe_report_cache = {
            ("artifact", "weights", "model"): {"passed": True}
        }
        self.state.model_content_fingerprint_cache = {
            "bf16:metadata": "fingerprint"
        }
        catalog = MagicMock()
        self.state.jacobian_lens_catalog = catalog

        self.state._release_model_local()

        self.assertEqual(self.state.jacobian_lens_cache, OrderedDict())
        self.assertEqual(self.state.jacobian_probe_report_cache, {})
        self.assertEqual(self.state.model_content_fingerprint_cache, {})
        catalog.clear_runtime_cache.assert_called_once_with()

    def test_status_exposes_artifact_and_request_identity(self) -> None:
        status = self.state.jacobian_lens_status()

        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["artifact_id"], MOCK_ARTIFACT_ID)
        self.assertEqual(
            status["source_layers"],
            [4, 16, 17, 22, 23, 24, 31],
        )
        self.assertEqual(status["target_layer"], 32)
        self.assertEqual(status["model_id"], "mock")
        self.assertEqual(status["model_generation"], 7)
        self.assertEqual(
            status["server_instance_id"],
            self.state.server_instance_id,
        )
        self.assertEqual(
            status["artifacts"][0]["artifact_id"],
            MOCK_ARTIFACT_ID,
        )

    def test_status_honors_requested_artifact_identity(self) -> None:
        missing = self.state.jacobian_lens_status("not-installed")

        self.assertEqual(missing["status"], "missing")
        self.assertEqual(missing["artifact_id"], "not-installed")
        self.assertEqual(
            missing["artifacts"][0]["artifact_id"],
            MOCK_ARTIFACT_ID,
        )


class JacobianLensRouteContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.get_source = inspect.getsource(GardenHandler.do_GET)
        route = POST_ROUTES["/api/jacobian-lens"]
        cls.post_source = inspect.getsource(
            getattr(GardenHandler, route.controller)
        )

    def test_page_status_and_post_routes_are_registered(self) -> None:
        self.assertIn('"/j-lens"', self.get_source)
        self.assertIn('"/jacobian-lens"', self.get_source)
        self.assertIn(
            'if path == "/api/jacobian-lens/status":',
            self.get_source,
        )
        self.assertIn('"artifact_id"', self.get_source)
        self.assertIn(
            "self.state.jacobian_lens_status(expected_artifact_id)",
            self.get_source,
        )
        self.assertIn(
            "self.state.jacobian_lens(",
            self.post_source,
        )

    def test_request_contract_carries_artifact_and_model_identity(self) -> None:
        route = self.post_source

        for field in (
            '"text"',
            '"top_k"',
            '"source_layers"',
            '"artifact_id"',
            '"model_id"',
            '"model_generation"',
            '"server_instance_id"',
        ):
            with self.subTest(field=field):
                self.assertIn(field, route)
        self.assertIn("len(source_layers) != len(set(source_layers))", route)
        self.assertIn("not 1 <= layer <= 31", route)
        self.assertIn("self.state.jacobian_lens(", route)


if __name__ == "__main__":
    unittest.main()
