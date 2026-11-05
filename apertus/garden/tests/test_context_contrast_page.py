from __future__ import annotations

import unittest
from pathlib import Path


GARDEN_DIR = Path(__file__).resolve().parents[1]
PAGE = GARDEN_DIR / "static" / "contrast.html"


class ContextContrastPageContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = PAGE.read_text(encoding="utf-8")

    def test_page_is_a_shared_workbench_view(self) -> None:
        self.assertIn(
            '<body data-workbench-page="contrast">',
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

    def test_inputs_preserve_exact_paired_context_contract(self) -> None:
        self.assertIn('id="prefix-a"', self.html)
        self.assertIn('id="prefix-b"', self.html)
        self.assertIn('id="continuation"', self.html)
        self.assertIn("> Praha leží na řece Vltavě.</textarea>", self.html)
        self.assertIn(
            "Úvodní mezera pokračování patří k cíli",
            self.html,
        )
        self.assertIn("model_id: check.identity.modelId", self.html)
        self.assertIn(
            "model_generation: check.identity.modelGeneration",
            self.html,
        )
        self.assertIn(
            "server_instance_id: check.identity.serverInstanceId",
            self.html,
        )

    def test_page_exposes_the_three_distinct_estimands(self) -> None:
        self.assertIn("Informační výhoda A − B", self.html)
        self.assertIn("Kořenová JS divergence", self.html)
        self.assertIn("Kdy se cíl zpřístupňuje", self.html)
        self.assertIn("Jak rozdíl doznívá", self.html)
        self.assertIn("Kumulativní výhoda přesné cesty", self.html)
        self.assertIn('id="access-chart"', self.html)
        self.assertIn('id="js-chart"', self.html)
        self.assertIn('id="advantage-chart"', self.html)
        self.assertIn('id="token-table-body"', self.html)

    def test_interpretation_does_not_claim_truth_or_causality(self) -> None:
        self.assertIn("nikoli pravděpodobností faktu ani kauzálním", self.html)
        self.assertIn("zásahem</strong>", self.html)
        self.assertIn(
            "popisuje pouze zadaný tokenový",
            self.html,
        )
        self.assertIn(
            "`řetězec; kořenová JS",
            self.html,
        )
        self.assertIn(
            "Top-k je jen viditelný výřez",
            self.html,
        )
        self.assertIn("JS výše je vždy spočtená", self.html)
        self.assertIn("nad celým", self.html)
        self.assertIn("slovníkem, včetně nezobrazeného ocasu", self.html)

    def test_live_request_has_loading_stale_and_model_change_states(self) -> None:
        self.assertIn('id="compute-overlay"', self.html)
        self.assertIn("AbortController", self.html)
        self.assertIn('fetchJson("/api/context-contrast"', self.html)
        self.assertIn('error.code === "model_changed"', self.html)
        self.assertIn("resultIsStale", self.html)
        self.assertIn("aria-busy", self.html)
        self.assertIn(
            '"probability-garden:workbench-state"',
            self.html,
        )
        self.assertIn('id="stale-notice"', self.html)
        self.assertIn("staleNotice.hidden = !stale", self.html)

    def test_tiny_ratios_and_missing_persistent_fall_are_not_misreported(self) -> None:
        self.assertIn("const formatRatio = (value) =>", self.html)
        self.assertIn('notation: "scientific"', self.html)
        self.assertIn("summaryRatio.textContent = formatRatio(ratio)", self.html)
        self.assertIn('"nepozorován"', self.html)
        self.assertIn(
            "Žádný pokles JS pod polovinu kořene nepřetrval do konce pozorovaného úseku.",
            self.html,
        )
        self.assertNotIn('"nenastal"', self.html)
        self.assertNotIn(
            "V pozorovaném úseku se JS trvale nedostala pod polovinu kořene.",
            self.html,
        )

    def test_result_can_be_exported_and_archived(self) -> None:
        self.assertIn('id="export-json"', self.html)
        self.assertIn('id="export-csv"', self.html)
        self.assertIn('id="save-experiment"', self.html)
        self.assertIn('kind: "context_contrast"', self.html)
        self.assertIn('href="/archiv?kind=context_contrast"', self.html)
        self.assertIn("rowsToCsv", self.html)
        self.assertIn("full_vocabulary_jensen_shannon_bits", self.html)


if __name__ == "__main__":
    unittest.main()
