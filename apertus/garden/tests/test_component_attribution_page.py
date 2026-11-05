from __future__ import annotations

import unittest
from pathlib import Path


GARDEN_DIR = Path(__file__).resolve().parents[1]
PAGE = GARDEN_DIR / "static" / "attribution.html"


class ComponentAttributionPageContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = PAGE.read_text(encoding="utf-8")

    def test_page_is_a_shared_workbench_view(self) -> None:
        self.assertIn(
            '<body data-workbench-page="attribution">',
            self.html,
        )
        self.assertIn(
            'href="/static/workbench-shell.css"',
            self.html,
        )
        self.assertIn(
            'src="/static/workbench-shell.js"',
            self.html,
        )
        self.assertIn('class="skip-link"', self.html)

    def test_method_is_explicit_about_scope_and_sign(self) -> None:
        self.assertIn("logit(A) − logit(B)", self.html)
        self.assertIn("touž finální RMSNorm", self.html)
        self.assertIn("popis jednoho výpočtu, ne důkaz příčiny", self.html)
        self.assertIn("ablaci, activation patching", self.html)
        self.assertIn("nikoli váhy attention", self.html)
        self.assertIn('<details id="method-details"', self.html)

    def test_controls_preserve_exact_token_contract(self) -> None:
        self.assertIn('id="token-a"', self.html)
        self.assertIn('value="␠Brno"', self.html)
        self.assertIn('id="token-b"', self.html)
        self.assertIn('value="␠Praha"', self.html)
        self.assertIn("Porovnáváme dva jednotlivé tokeny", self.html)
        self.assertIn("také přesným #ID", self.html)
        self.assertIn('id="swap-tokens"', self.html)

    def test_page_has_both_required_visualizations_and_fallback_table(self) -> None:
        self.assertIn('id="contribution-chart"', self.html)
        self.assertIn("Co přidaly jednotlivé vrstvy?", self.html)
        self.assertIn('id="trajectory-chart"', self.html)
        self.assertIn("Jak se náskok postupně skládá", self.html)
        self.assertIn('<details id="technical-results"', self.html)
        self.assertIn('id="local-raw-chart"', self.html)
        self.assertIn("Lokální RAW čtení", self.html)
        self.assertIn('id="layer-table"', self.html)
        self.assertIn("Attention Δlogit", self.html)
        self.assertIn("MLP Δlogit", self.html)

    def test_local_raw_has_an_optional_chart_and_independent_scale(self) -> None:
        self.assertIn(
            '<details class="result-panel secondary-chart" id="local-raw-panel">',
            self.html,
        )
        self.assertNotIn(
            '<details class="result-panel secondary-chart" id="local-raw-panel" open>',
            self.html,
        )
        self.assertIn("proto má graf samostatnou osu", self.html)
        self.assertIn("const renderLocalRawChart = (data) =>", self.html)
        self.assertIn("renderLocalRawChart(data);", self.html)
        trajectory_source = self.html.split(
            "const renderTrajectoryChart = (data) =>",
            1,
        )[1].split("const renderLocalRawChart = (data) =>", 1)[0]
        self.assertNotIn("local_reader_after_layer", trajectory_source)

    def test_loading_error_stale_and_cancel_states_are_visible(self) -> None:
        self.assertIn('id="compute-overlay"', self.html)
        self.assertIn('id="cancel-request"', self.html)
        self.assertIn("AbortController", self.html)
        self.assertIn("AbortError", self.html)
        self.assertIn('error.code === "model_changed"', self.html)
        self.assertIn('role="status"', self.html)

    def test_result_can_be_exported_with_full_diagnostics(self) -> None:
        self.assertIn('id="download-json"', self.html)
        self.assertIn("JSON.stringify(currentResult, null, 2)", self.html)
        self.assertIn("component-attribution-", self.html)
        self.assertIn("summary-reconstruction", self.html)
        self.assertIn("residual_rounding_relative_norm", self.html)

    def test_request_is_bound_to_model_identity(self) -> None:
        self.assertIn('fetchJson("/api/component-attribution"', self.html)
        self.assertIn("model_id: health.active_model?.id", self.html)
        self.assertIn("model_generation: health.model_generation", self.html)
        self.assertIn(
            "server_instance_id: health.server_instance_id",
            self.html,
        )
        self.assertIn('["hybrid", "bf16", "mock"]', self.html)


if __name__ == "__main__":
    unittest.main()
