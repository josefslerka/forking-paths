from __future__ import annotations

import unittest
from pathlib import Path


GARDEN_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = GARDEN_DIR / "static"


class WorkbenchShellContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.javascript = (STATIC_DIR / "workbench-shell.js").read_text(
            encoding="utf-8"
        )
        cls.css = (STATIC_DIR / "workbench-shell.css").read_text(
            encoding="utf-8"
        )
        cls.server = (GARDEN_DIR / "server.py").read_text(encoding="utf-8")

    def test_product_brand_is_separate_from_the_garden_and_model(self) -> None:
        self.assertIn('<strong>Rozcestí</strong>', self.javascript)
        self.assertIn('"Rozcestí – domů"', self.javascript)
        self.assertIn('"Výzkumná a výuková laboratoř jazykových modelů"', self.javascript)
        self.assertIn('label: "Pravděpodobnostní zahrada"', self.javascript)
        for html_path in STATIC_DIR.glob("*.html"):
            with self.subTest(page=html_path.name):
                html = html_path.read_text(encoding="utf-8")
                if 'data-workbench-page=' in html:
                    title = html.split("<title>", 1)[1].split("</title>", 1)[0]
                    self.assertTrue(title.endswith(" · Rozcestí"), title)

    def test_every_view_loads_the_shared_shell(self) -> None:
        pages = {
            "tokenization.html": "tokenization",
            "embedding.html": "embeddings",
            "index.html": "garden",
            "surprise.html": "surprise",
            "detector.html": "detector",
            "contrast.html": "contrast",
            "lens.html": "raw",
            "tuned.html": "tuned",
            "jlens.html": "jlens",
            "attribution.html": "attribution",
            "patching.html": "patching",
            "steering.html": "steering",
            "guide.html": "guide",
            "archive.html": "archive",
        }
        for filename, page_id in pages.items():
            with self.subTest(filename=filename):
                html = (STATIC_DIR / filename).read_text(encoding="utf-8")
                self.assertIn(
                    f'<body data-workbench-page="{page_id}">',
                    html,
                )
                self.assertIn(
                    'href="/static/workbench-shell.css"',
                    html,
                )
                if filename == "detector.html":
                    self.assertNotIn(
                        'src="/static/api-client.js"',
                        html,
                    )
                else:
                    self.assertIn(
                        'src="/static/api-client.js"',
                        html,
                    )
                    self.assertLess(
                        html.index("/static/api-client.js"),
                        html.index("/static/workbench-shell.js"),
                    )
                self.assertIn(
                    'src="/static/workbench-shell.js"',
                    html,
                )
                self.assertIn(
                    'src="/static/i18n-catalog.generated.js"',
                    html,
                )
                self.assertIn('src="/static/i18n.js"', html)
                self.assertLess(
                    html.index("/static/i18n.js"),
                    html.index("/static/workbench-shell.js"),
                )
                self.assertIn('class="skip-link"', html)

    def test_server_exposes_shell_assets_with_explicit_content_types(self) -> None:
        self.assertIn('path == "/static/workbench-shell.css"', self.server)
        self.assertIn('"text/css; charset=utf-8"', self.server)
        self.assertIn('path == "/static/workbench-shell.js"', self.server)
        self.assertIn(
            'path == "/static/i18n-catalog.generated.js"',
            self.server,
        )
        self.assertIn('path == "/static/i18n.js"', self.server)
        self.assertIn('"text/javascript; charset=utf-8"', self.server)
        self.assertIn('path == "/static/api-client.js"', self.server)
        self.assertIn(
            'script.src = "/static/api-client.js"',
            self.javascript,
        )

    def test_shell_has_a_persistent_accessible_language_switch(self) -> None:
        i18n = (STATIC_DIR / "i18n.js").read_text(encoding="utf-8")
        self.assertIn("pg-language-switch", self.javascript)
        self.assertIn("pg-language-button", self.javascript)
        self.assertIn('String((i18n?.language ?? "cs") === language.id)', self.javascript)
        self.assertIn('url.searchParams.delete("lang")', self.javascript)
        self.assertIn("probability-garden:language", i18n)
        self.assertIn("window.localStorage.setItem", i18n)
        self.assertIn("document.documentElement.lang = language", i18n)
        self.assertIn(".pg-language-switch", self.css)
        self.assertIn('.pg-language-button[aria-pressed="true"]', self.css)

    def test_shell_keeps_global_views_and_non_linear_steering_map(self) -> None:
        for page_id in (
            "tokenization",
            "embeddings",
            "garden",
            "surprise",
            "detector",
            "contrast",
            "raw",
            "tuned",
            "jlens",
            "attribution",
            "patching",
            "steering",
            "guide",
            "archive",
        ):
            self.assertIn(f'id: "{page_id}"', self.javascript)
        for stage_id in (
            "direction",
            "immediate",
            "trajectory",
            "held-out",
            "sham",
            "sweep",
        ):
            self.assertIn(f'id: "{stage_id}"', self.javascript)
        self.assertIn(
            "Ostatní oblasti jsou samostatné větve",
            self.javascript,
        )
        self.assertIn('label: "Průvodce workbenchem"', self.javascript)
        self.assertIn('meta: "mapa metod"', self.javascript)
        self.assertIn('label: "Tokenizace"', self.javascript)
        self.assertIn('meta: "vstupní jednotky"', self.javascript)
        self.assertIn('label: "Embeddingy"', self.javascript)
        self.assertIn('meta: "výchozí geometrie"', self.javascript)
        self.assertNotIn("09 / 24", self.javascript)

    def test_shell_supports_live_context_and_accessible_stage_states(self) -> None:
        self.assertIn(
            '"probability-garden:workbench-state"',
            self.javascript,
        )
        self.assertIn(
            '"probability-garden:request-workbench-state"',
            self.javascript,
        )
        self.assertIn("aria-disabled", self.javascript)
        self.assertIn('aria-current", "location"', self.javascript)
        self.assertIn("const activateStage = (", self.javascript)
        self.assertIn('root.querySelector("#run-steering")', self.javascript)
        self.assertIn("primaryAction.disabled", self.javascript)
        self.assertIn(
            'activateStage("direction", {scroll: false})',
            self.javascript,
        )
        self.assertIn("MutationObserver", self.javascript)
        self.assertIn('api.fetchJson("/api/health"', self.javascript)
        self.assertIn("activeModel.label", self.javascript)
        self.assertIn("prefers-reduced-motion: reduce", self.javascript)

    def test_locked_stage_deep_link_is_replayed_after_steering(self) -> None:
        steering = (STATIC_DIR / "steering.html").read_text(encoding="utf-8")
        self.assertIn("pendingRequestedStage", self.javascript)
        self.assertIn("setPendingRequestedStage", self.javascript)
        self.assertIn("activateRequestedStage", self.javascript)
        self.assertIn(
            "Po dokončení se požadovaná etapa",
            self.javascript,
        )
        self.assertIn(
            "window.ProbabilityGardenShell?.activateRequestedStage?.({",
            steering,
        )
        self.assertIn(
            'window.addEventListener("hashchange"',
            self.javascript,
        )
        self.assertIn(".pg-stage-lock-notice", self.css)

    def test_model_is_a_global_context_control(self) -> None:
        self.assertIn("data-pg-model-slot", self.javascript)
        self.assertIn('root.querySelector("#model-mode")', self.javascript)
        self.assertIn(
            "const shellOwnsModelSelect = !pageModelSelect",
            self.javascript,
        )
        self.assertIn(
            "if (!shellOwnsModelSelect) {",
            self.javascript,
        )
        self.assertIn('modelSelect.id = "pg-model-mode"', self.javascript)
        self.assertIn('api.fetchJson("/api/model"', self.javascript)
        self.assertIn("api.eventName", self.javascript)
        self.assertIn("body.dataset.apiRequestState", self.javascript)
        self.assertIn("health?.worker?.busy", self.javascript)
        self.assertIn(
            "health.transition_id !== expectedTransitionId",
            self.javascript,
        )
        self.assertIn("window.location.reload()", self.javascript)
        self.assertIn(".pg-model-select", self.css)
        self.assertIn(
            'body[data-workbench-page="garden"] .model-control',
            self.css,
        )

    def test_startup_model_comes_from_registry_not_a_fixed_cli_list(self) -> None:
        self.assertIn("registry.default_model", self.server)
        self.assertIn("registry.model_paths()", self.server)
        self.assertNotIn('choices=["bf16", HYBRID_MODEL_ID', self.server)
        self.assertIn('"--registry"', self.server)

    def test_workbench_hero_scale_is_bounded(self) -> None:
        self.assertNotIn("8.6rem", self.css)
        self.assertIn(
            "font-size: clamp(3.2rem, 5.35vw, 5.65rem)",
            self.css,
        )
        self.assertIn(
            "font-size: clamp(2.7rem, 4.1vw, 4.45rem)",
            self.css,
        )

    def test_analytical_hero_descriptions_follow_the_heading(self) -> None:
        for page_id in (
            "surprise",
            "detector",
            "contrast",
            "raw",
            "tuned",
            "jlens",
            "attribution",
            "patching",
            "steering",
            "archive",
        ):
            with self.subTest(page_id=page_id):
                selector = (
                    f'body[data-workbench-page="{page_id}"] '
                    ".pg-content-root > .hero"
                )
                self.assertIn(selector, self.css)
                self.assertIn(f"{selector} .lede", self.css)
        self.assertIn("display: block;", self.css)
        self.assertIn("font-family: var(--pg-sans);", self.css)
        self.assertIn("max-width: 720px;", self.css)

    def test_shell_has_responsive_and_reduced_motion_modes(self) -> None:
        self.assertIn("@media (max-width: 980px)", self.css)
        self.assertIn("@media (max-width: 720px)", self.css)
        self.assertIn("@media (prefers-reduced-motion: reduce)", self.css)
        self.assertIn("scroll-margin-top", self.css)
        self.assertIn(".pg-stage-link[aria-disabled=\"true\"]", self.css)
        self.assertIn(".pg-stage-focus", self.css)
        mobile_start = self.css.index("@media (max-width: 720px)")
        reduced_start = self.css.index(
            "@media (prefers-reduced-motion: reduce)"
        )
        mobile_css = self.css[mobile_start:reduced_start]
        responsive_start = self.css.index("@media (max-width: 980px)")
        responsive_css = self.css[responsive_start:reduced_start]
        self.assertNotIn(".pg-view-nav {\n    display: none", mobile_css)
        self.assertNotIn(".pg-context {\n    display: none", responsive_css)
        self.assertIn(
            ".pg-context .pg-context-model {\n    display: grid",
            responsive_css,
        )


if __name__ == "__main__":
    unittest.main()
