from __future__ import annotations

import unittest
from pathlib import Path


GARDEN_DIR = Path(__file__).resolve().parents[1]
PAGE = GARDEN_DIR / "static" / "jlens.html"


class JacobianLensPageContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = PAGE.read_text(encoding="utf-8")

    def test_page_uses_workbench_shell_and_additive_reader_identity(self) -> None:
        self.assertIn('<body data-workbench-page="jlens">', self.html)
        self.assertIn('href="/static/workbench-shell.css"', self.html)
        self.assertIn('src="/static/workbench-shell.js"', self.html)
        self.assertIn("RAW × J‑LENS", self.html)
        self.assertIn("Nesoulad čteček", self.html)
        self.assertIn(
            "Nesoulad čteček je měření, nikoli chyba",
            self.html,
        )

    def test_page_requires_bf16_and_never_silently_falls_back(self) -> None:
        self.assertIn('const BF16_MODEL_ID = "bf16"', self.html)
        self.assertIn("activeModelIsBf16", self.html)
        self.assertIn(
            "Tento J‑Lens byl fitován pro BF16 Apertus.",
            self.html,
        )
        self.assertIn(
            "Přenos na Hybrid zatím nebyl validován.",
            self.html,
        )
        self.assertIn(
            "V horním pruhu přepni model na BF16.",
            self.html,
        )
        self.assertIn(
            'health?.active_model?.id === BF16_MODEL_ID',
            self.html,
        )

    def test_status_and_compute_use_dedicated_api_contracts(self) -> None:
        self.assertIn('"/api/jacobian-lens/status"', self.html)
        self.assertIn(
            "/api/jacobian-lens/status?artifact_id=",
            self.html,
        )
        self.assertIn('fetchJson("/api/jacobian-lens", {', self.html)
        for identity in (
            "model_id",
            "model_generation",
            "server_instance_id",
            "artifact_id",
        ):
            self.assertIn(identity, self.html)
        for state in (
            "loading",
            "ready",
            "missing",
            "incompatible",
            "invalid",
            "error",
            "unvalidated",
        ):
            self.assertIn(f"{state}:", self.html)

    def test_source_layers_are_manifest_driven_and_l4_is_visible(self) -> None:
        self.assertIn('id="source-layer-control"', self.html)
        self.assertIn('name="source_layer"', self.html)
        self.assertIn("source_layers", self.html)
        self.assertIn("target_layer", self.html)
        self.assertIn('id="l4-warning"', self.html)
        self.assertIn(
            "L4: slabší shoda dvou fitovacích polovin",
            self.html,
        )
        self.assertIn("l4.matrix_cosine", self.html)
        self.assertIn("l4.relative_frobenius_difference", self.html)
        self.assertNotIn('value="4" disabled', self.html)

    def test_page_can_compare_installed_target_variants(self) -> None:
        self.assertIn('id="artifact-select"', self.html)
        self.assertIn("availableArtifacts", self.html)
        self.assertIn("Paper-primary", self.html)
        self.assertIn("Final-target", self.html)
        self.assertIn("target L", self.html)
        self.assertIn("readout_collapse_summary", self.html)
        self.assertIn("Held-out diagnostika", self.html)
        self.assertIn(
            "Je to varování čtečky, ne vlastnost konkrétního prefixu.",
            self.html,
        )

    def test_result_is_neutral_and_reports_shared_reference(self) -> None:
        self.assertIn("J‑LENS blíž", self.html)
        self.assertIn("J‑LENS dál", self.html)
        self.assertIn("ΔJS (J‑LENS − RAW)", self.html)
        self.assertIn(
            "row?.comparison?.js_jacobian_minus_raw_bits",
            self.html,
        )
        self.assertIn(
            "row?.jacobian ?? row?.jlens",
            self.html,
        )
        self.assertIn("společná finální reference", self.html)
        self.assertIn(
            "Nečiní z čtečky přímý dekodér významu.",
            self.html,
        )
        self.assertNotIn("správnější čtečka", self.html.lower())
        self.assertNotIn("model věří", self.html.lower())
        self.assertNotIn("model si myslí", self.html.lower())

    def test_top_k_candidates_expand_with_stable_saturation_metrics(self) -> None:
        self.assertIn('<option value="12">Top 12</option>', self.html)
        self.assertIn('<option value="20">Top 20</option>', self.html)
        self.assertIn("data-candidate-toggle", self.html)
        self.assertIn("renderReaderCandidates", self.html)
        self.assertIn("surprisal_bits", self.html)
        self.assertIn("logit_gap_from_top1_bits", self.html)
        self.assertIn("final_rank", self.html)
        self.assertIn("final_surprisal_bits", self.html)
        self.assertIn(
            "Pravděpodobnost může být po softmaxu numericky 0 nebo 100 %",
            self.html,
        )
        self.assertIn(
            "téhož tokenu po posledním",
            self.html,
        )
        self.assertIn(
            'toggle.setAttribute("aria-expanded", String(!expanded))',
            self.html,
        )

    def test_loading_error_stale_and_cancel_states_are_explicit(self) -> None:
        self.assertIn('id="compute-overlay"', self.html)
        self.assertIn('id="cancel-compute"', self.html)
        self.assertIn("abortController?.abort()", self.html)
        self.assertIn(
            "Načítám a ověřuji J‑Lens artefakt",
            self.html,
        )
        self.assertIn(
            "Výsledek je zastaralý po změně prefixu",
            self.html,
        )
        self.assertIn("Hash, dtype nebo tvar", self.html)
        self.assertIn("model_changed", self.html)

    def test_archive_record_reuses_lens_trajectory_kind(self) -> None:
        self.assertIn('kind: "lens_trajectory"', self.html)
        self.assertIn('kind: "raw_vs_jacobian_lens"', self.html)
        self.assertIn(
            'tags: ["raw-vs-jlens", "jacobian-lens", "bf16"]',
            self.html,
        )
        self.assertIn(
            'href="/archiv?kind=lens_trajectory"',
            self.html,
        )
        self.assertIn(
            "Aktivace ani Jacobianové matice se neukládají.",
            self.html,
        )


if __name__ == "__main__":
    unittest.main()
