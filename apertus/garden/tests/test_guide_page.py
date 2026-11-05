from __future__ import annotations

import re
import unittest
from pathlib import Path


GARDEN_DIR = Path(__file__).resolve().parents[1]
PAGE = GARDEN_DIR / "static" / "guide.html"


class GuidePageContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = PAGE.read_text(encoding="utf-8")

    def test_page_is_a_workbench_guide_not_a_garden_only_manual(self) -> None:
        self.assertIn("<h1>Průvodce workbenchem</h1>", self.html)
        self.assertIn("soustavu různých řezů týmž", self.html)
        self.assertIn("Začni otázkou, ne nástrojem", self.html)
        self.assertNotIn(
            "<h1>Jak číst Pravděpodobnostní zahradu</h1>",
            self.html,
        )
        self.assertNotIn("Tři druhy záznamu", self.html)

    def test_question_router_links_every_research_view(self) -> None:
        for href in (
            "/tokenizace",
            "/",
            "/prekvapeni",
            "/kontrast",
            "/vrstvy",
            "/tuned-lens",
            "/atribuce",
            "/kauzalni-prenos",
            "/steering",
            "/archiv",
        ):
            with self.subTest(href=href):
                self.assertRegex(
                    self.html,
                    rf'class="question-card" href="{re.escape(href)}"',
                )
        self.assertIn("RAW × TUNED / J‑LENS", self.html)
        self.assertIn("Co model skutečně dostane?", self.html)

    def test_evidential_ladder_separates_readout_accounting_and_intervention(
        self,
    ) -> None:
        self.assertIn("Od pozorování k zásahu", self.html)
        self.assertIn("Elementární vstup", self.html)
        self.assertIn("Readout mezistavu", self.html)
        self.assertIn("Aditivní účetnictví", self.html)
        self.assertIn("Interchange intervence", self.html)
        self.assertIn("Směrová intervence", self.html)
        self.assertIn(
            "Je kauzální vůči přesnému zásahu, ne automaticky",
            self.html,
        )
        self.assertIn(
            "Komponenty jsou přesné účetnictví jednoho realizovaného",
            self.html,
        )

    def test_each_new_view_has_a_question_metric_and_boundary(self) -> None:
        for phrase in (
            "Tokenizace",
            "Překvapení textu",
            "Kontrast kontextů",
            "RAW × TUNED",
            "RAW × J‑LENS",
            "Attention × MLP",
            "Kauzální přenos A → B",
            "Steering aktivací",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.html)
        self.assertIn("Výhoda řetězce není pravděpodobnost faktu", self.html)
        self.assertIn("Překvapení odhalilo autora či poezii", self.html)
        self.assertIn("Velký attention/MLP příspěvek je příčina", self.html)
        self.assertIn("Patching našel místo, kde sídlí význam", self.html)
        self.assertIn("Steeringový směr je čistý koncept", self.html)
        self.assertIn(
            "Deterministickou segmentaci za volbu modelu",
            self.html,
        )

    def test_reader_specific_failures_and_model_requirements_are_explicit(
        self,
    ) -> None:
        self.assertIn("Tentýž mezistav, různé čtečky", self.html)
        self.assertIn("token <code>.Qt</code>", self.html)
        self.assertIn("připravený Hybrid artefakt", self.html)
        self.assertIn("připravený BF16 artefakt", self.html)
        self.assertIn(
            "Dostupnost konkrétní čtečky je součástí pohledu",
            self.html,
        )

    def test_shared_measurement_contract_remains_visible(self) -> None:
        self.assertIn(
            "Nejdřív text, potom token, teprve pak distribuce",
            self.html,
        )
        self.assertIn("131&nbsp;072 tokenů", self.html)
        self.assertIn("Top‑k je vždy jen viditelný výřez", self.html)
        self.assertIn("Výzkumný výchozí bod je T = 1", self.html)
        self.assertIn("Informační výhoda cesty", self.html)
        self.assertIn("Teacher forcing", self.html)
        self.assertIn("Readout / čtečka", self.html)
        self.assertIn("0 forwardů · deterministický převod", self.html)
        self.assertIn(
            "Tokenizace neurčuje význam. Určuje však elementární",
            self.html,
        )

    def test_static_ids_are_unique(self) -> None:
        ids = re.findall(r'\bid="([^"]+)"', self.html)
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
