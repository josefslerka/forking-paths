from __future__ import annotations

import unittest
from pathlib import Path

from apertus.garden.api.post_routes import POST_ROUTES


GARDEN_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = GARDEN_DIR / "static"


class ActivationPatchingPageContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (STATIC_DIR / "patching.html").read_text(encoding="utf-8")
        cls.server = (GARDEN_DIR / "server.py").read_text(encoding="utf-8")

    def test_page_is_part_of_the_shared_workbench(self) -> None:
        self.assertIn(
            '<body data-workbench-page="patching">',
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

    def test_page_requires_visible_alignment_before_intervention(self) -> None:
        self.assertIn('fetchJson("/api/tokenization"', self.html)
        self.assertIn('id="alignment-track"', self.html)
        self.assertIn('id="select-shared-after"', self.html)
        self.assertIn(
            "První verze patchuje jen přesně zarovnané pozice",
            self.html,
        )

    def test_page_calls_patching_api_with_identity_fields(self) -> None:
        self.assertIn('fetchJson("/api/activation-patching"', self.html)
        for field in (
            "source_text",
            "target_text",
            "token_a",
            "token_b",
            "layers",
            "positions",
            "model_id",
            "model_generation",
            "server_instance_id",
        ):
            with self.subTest(field=field):
                self.assertIn(field, self.html)

    def test_page_exposes_heatmap_detail_controls_and_exports(self) -> None:
        for element_id in (
            "heatmap",
            "detail-grid",
            "mover-gains",
            "mover-losses",
            "summary-identity",
            "download-json",
            "download-csv",
        ):
            with self.subTest(element_id=element_id):
                self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn("contrast_recovery", self.html)
        self.assertIn("distribution_recovery", self.html)
        self.assertIn("js_to_source_bits", self.html)
        self.assertIn("max_abs_logit_error", self.html)

    def test_copy_limits_the_causal_claim(self) -> None:
        self.assertIn("Kauzální dostatečnost", self.html)
        self.assertIn("nikoli jedno", self.html)
        self.assertIn("ani nezbytnost", self.html)
        self.assertIn("Každá buňka je samostatný zásah", self.html)

    def test_server_exposes_page_and_api(self) -> None:
        self.assertIn('"/kauzalni-prenos"', self.server)
        self.assertIn('"/activation-patching"', self.server)
        self.assertIn("/api/activation-patching", POST_ROUTES)


if __name__ == "__main__":
    unittest.main()
