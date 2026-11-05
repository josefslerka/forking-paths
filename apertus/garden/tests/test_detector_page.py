from __future__ import annotations

import unittest
from pathlib import Path

from apertus.garden.api.post_routes import POST_ROUTES


GARDEN_DIR = Path(__file__).resolve().parents[1]
PAGE = GARDEN_DIR / "static" / "detector.html"
SERVER = GARDEN_DIR / "server.py"


class DetectorPageContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = PAGE.read_text(encoding="utf-8")
        cls.server = SERVER.read_text(encoding="utf-8")

    def test_page_is_a_shared_accessible_workbench_view(self) -> None:
        self.assertIn('<html lang="cs">', self.html)
        self.assertIn('<body data-workbench-page="detector">', self.html)
        self.assertIn('class="skip-link"', self.html)
        self.assertIn('href="/static/workbench-shell.css"', self.html)
        self.assertIn('src="/static/workbench-shell.js"', self.html)
        self.assertIn('aria-live="polite"', self.html)
        self.assertIn("@media (prefers-reduced-motion: reduce)", self.html)

    def test_input_and_privacy_contract_are_explicit(self) -> None:
        self.assertIn('minlength="2100"', self.html)
        self.assertIn('maxlength="100000"', self.html)
        self.assertIn("Text byl analyzován lokálně", self.html)
        self.assertNotIn("localStorage", self.html)
        self.assertNotIn("/api/experiments", self.html)
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)

    def test_page_renders_champion_challenger_and_archive_scores(
        self,
    ) -> None:
        for label in (
            "Párová stylometrie v5",
            "Stylometrie v2 · původní skóre",
            "Apertus",
            "Experimentální fúze v2",
            "Popisný profil Apertusu",
        ):
            self.assertIn(label, self.html)
        self.assertIn("Bez klasifikátoru", self.html)
        self.assertIn("120 + 5 rysů · diagnostické", self.html)
        self.assertIn("ai_direction_index", self.html)
        self.assertIn("Apertus-first", self.html)
        self.assertIn("250 lidským textům", self.html)
        self.assertIn('id="challenger-score"', self.html)
        self.assertIn('id="challenger-comparison"', self.html)
        self.assertIn("renderChallenger", self.html)
        self.assertIn("timing.challenger_model", self.html)
        self.assertIn("promotion gate proto neprošla", self.html)
        self.assertIn("1 / 24", self.html)
        self.assertIn("5,3 %", self.html)
        self.assertIn("párový loss", self.html)
        self.assertIn('id="challenger-pair-win"', self.html)

    def test_apertus_first_decision_and_abstention_are_explicit(self) -> None:
        self.assertIn("apertusFirstDecision", self.html)
        self.assertIn("const primaryModel = models.apertus;", self.html)
        self.assertIn("Apertus zachytil přísný AI signál", self.html)
        self.assertIn("Nejisté: Apertus je v šedém pásmu", self.html)
        self.assertIn("sama AI verdikt nevyvolá", self.html)
        self.assertIn(
            "Párová stylometrie v5 ani modely v2",
            self.html,
        )
        self.assertIn(
            "const challengerModel = renderChallenger(data.challenger);",
            self.html,
        )
        self.assertIn(
            'data?.primary_decision?.model !== "apertus"',
            self.html,
        )
        self.assertIn(
            "data?.challenger?.decision_authority !== false",
            self.html,
        )
        self.assertIn("Výsledek je vodítko, ne důkaz autorství", self.html)
        self.assertIn("Quality gate se zdržel verdiktu", self.html)
        self.assertIn("Screening práh", self.html)
        self.assertIn('id="decision-margin"', self.html)
        self.assertIn('id="style-advisory"', self.html)
        self.assertNotIn("verdict.dataset.state = data.verdict", self.html)
        self.assertNotIn("expectedPrecision", self.html)

    def test_context_explains_both_news_audits(self) -> None:
        self.assertIn(
            "Audity 790 lidských + 120 AI přiřazení",
            self.html,
        )
        self.assertIn("Lidské · falešně AI", self.html)
        self.assertIn("2,2 %", self.html)
        self.assertIn("AI · přísně zachyceno", self.html)
        self.assertIn("1,8 %", self.html)
        self.assertIn("AI · screeningový záchyt", self.html)
        self.assertIn("17,0 %", self.html)
        self.assertIn("113 / 120", self.html)
        self.assertIn("Předem stanovená hypotéza neprošla", self.html)
        self.assertIn("0,654", self.html)
        self.assertIn("0,709", self.html)
        self.assertIn("0,750", self.html)
        self.assertIn("Inference bundle v2", self.html)

    def test_completed_confusion_matrix_is_explicit_and_filterable(
        self,
    ) -> None:
        self.assertIn("Rozšířená confusion matice", self.html)
        self.assertIn("HUMAN_CONFUSION_AUDIT", self.html)
        self.assertIn("AI_CONFUSION_AUDIT", self.html)
        self.assertIn("renderConfusionMatrix", self.html)
        self.assertIn("strictAi: 17", self.html)
        self.assertIn("uncertain: 42", self.html)
        self.assertIn("noSignal: 719", self.html)
        self.assertIn("abstain: 12", self.html)
        self.assertIn("strictAi: 2", self.html)
        self.assertIn("uncertain: 17", self.html)
        self.assertIn("noSignal: 93", self.html)
        self.assertIn("abstain: 8", self.html)
        self.assertIn('id="confusion-ai-strict"', self.html)
        self.assertIn('id="confusion-ai-uncertain"', self.html)
        self.assertIn('id="confusion-ai-no-signal"', self.html)
        self.assertIn('id="confusion-ai-abstain"', self.html)
        self.assertIn("nelze odvozovat populační precision", self.html)
        self.assertNotIn("čeká na 120 AI přepisů", self.html)
        for scope in (
            "overall",
            "echo24.cz",
            "irozhlas.cz",
            "denikalarm.cz",
            "denikreferendum.cz",
        ):
            self.assertIn(f'data-confusion-scope="{scope}"', self.html)

    def test_request_is_bound_to_current_model_identity(self) -> None:
        self.assertIn('fetchJson("/api/predictive-stylometry"', self.html)
        self.assertIn("model_id: check.identity.modelId", self.html)
        self.assertIn(
            "model_generation: check.identity.modelGeneration",
            self.html,
        )
        self.assertIn(
            "server_instance_id: check.identity.serverInstanceId",
            self.html,
        )
        self.assertIn('currentIdentity.modelId !== "hybrid"', self.html)

    def test_server_exposes_page_api_and_single_worker_operation(self) -> None:
        self.assertIn('"/detektor"', self.server)
        self.assertIn("/api/predictive-stylometry", POST_ROUTES)
        self.assertIn(
            'elif operation == "predictive_stylometry":',
            self.server,
        )
        self.assertIn(
            'self._submit_ready(\n            "predictive_stylometry"',
            self.server,
        )
        self.assertIn("PredictiveDetector(", self.server)


if __name__ == "__main__":
    unittest.main()
