from __future__ import annotations

import unittest
from html.parser import HTMLParser
from pathlib import Path


STEERING_PAGE = (
    Path(__file__).resolve().parents[1] / "static" / "steering.html"
)


class _StaticReferenceCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.references: list[tuple[str, str]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(str(attributes["id"]))
        if attributes.get("for"):
            self.references.append(("for", str(attributes["for"])))
        for name in ("aria-labelledby", "aria-describedby"):
            value = attributes.get(name)
            if value:
                self.references.extend(
                    (name, target)
                    for target in str(value).split()
                )
        href = attributes.get("href")
        if href and str(href).startswith("#") and len(str(href)) > 1:
            self.references.append(("href", str(href)[1:]))


class SteeringPageContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = STEERING_PAGE.read_text(encoding="utf-8")

    def test_wait_overlay_hides_without_aborting_the_request(self) -> None:
        self.assertIn('id="hide-compute"', self.html)
        self.assertIn("Skrýt čekání", self.html)
        self.assertIn(
            'hideComputeButton.addEventListener("click", hideComputeWait)',
            self.html,
        )

        start = self.html.index("const hideComputeWait = () => {")
        end = self.html.index("const trapOverlayFocus", start)
        hide_handler = self.html[start:end]
        self.assertIn("hideOverlay({restoreFocus: false})", hide_handler)
        self.assertNotIn(".abort(", hide_handler)
        self.assertNotIn("state.busy = false", hide_handler)

        self.assertIn("runButton.textContent = state.busy", self.html)
        self.assertIn('"Počítám steering…"', self.html)
        self.assertIn('window.addEventListener("pagehide"', self.html)
        self.assertIn("state.controller?.abort()", self.html)

    def test_copy_explains_hide_only_semantics(self) -> None:
        self.assertIn(
            "Skrytí okna nezastaví výpočet; formulář zůstane uzamčený "
            "do dokončení.",
            self.html,
        )
        self.assertIn(
            "<strong>Skrytí čekacího okna výpočet nezruší.</strong>",
            self.html,
        )
        self.assertNotIn(">Zrušit</button>", self.html)
        self.assertNotIn("Výpočet steeringu byl zrušen.", self.html)

    def test_duplicate_validation_matches_backend_orientation_policy(self) -> None:
        start = self.html.index("const contrastPairKey = (pair) =>")
        end = self.html.index("const duplicateContrastCount", start)
        key_builder = self.html[start:end]
        self.assertIn("requestText(pair.negative_text)", key_builder)
        self.assertIn("requestText(pair.positive_text)", key_builder)
        self.assertIn("].sort()", key_builder)

        collision_start = self.html.index("const testPromptCollision = () =>")
        collision_end = self.html.index("const contrastPairKey", collision_start)
        collision_check = self.html[collision_start:collision_end]
        self.assertIn("const target = requestText(testPrompt.value)", collision_check)
        self.assertNotIn("canonicalText(text)", collision_check)

    def test_trajectory_uses_frozen_result_and_symmetric_alpha(self) -> None:
        self.assertIn('id="trajectory-section"', self.html)
        self.assertIn('id="run-trajectory"', self.html)
        self.assertIn(">Dopsat do tečky</button>", self.html)
        self.assertIn(
            'const TRAJECTORY_BRANCH_ORDER = ["negative", "baseline", "positive"]',
            self.html,
        )
        self.assertIn('fetchJson("/api/steering/trajectory"', self.html)
        self.assertIn("const frozen = eligibility.request", self.html)
        self.assertIn(
            "alpha_magnitude: eligibility.magnitude",
            self.html,
        )
        self.assertIn('generation_mode: "greedy"', self.html)
        self.assertIn("temperature: 1", self.html)

    def test_trajectory_validates_and_renders_token_level_provenance(self) -> None:
        self.assertIn("const assertTrajectoryResponse", self.html)
        self.assertIn(
            "data?.steering?.steering_applied_every_step !== true",
            self.html,
        )
        self.assertIn("const normalizeTrajectoryToken", self.html)
        self.assertIn("token?.token_id", self.html)
        self.assertIn("contextAligned: divergence.context_aligned", self.html)
        self.assertIn(
            'metrics.push("jiný kontext")',
            self.html,
        )
        self.assertIn(
            "Po prvním rozchodu má každá větev jiný kontext",
            self.html,
        )

    def test_trajectory_wait_can_hide_but_never_abort(self) -> None:
        self.assertIn('id="hide-trajectory-wait"', self.html)
        self.assertIn(
            'hideTrajectoryWaitButton.addEventListener(',
            self.html,
        )
        start = self.html.index("const hideTrajectoryWait = () => {")
        end = self.html.index("const prepareGardenHandoff", start)
        handler = self.html[start:end]
        self.assertIn("state.trajectoryWaitHidden = true", handler)
        self.assertIn("trajectoryLoading.hidden = true", handler)
        self.assertNotIn(".abort(", handler)
        self.assertNotIn("state.busy = false", handler)
        self.assertIn(
            "Skrytí čekání požadavek nezastavilo",
            handler,
        )
        self.assertIn(
            ".trajectory-loading[hidden]",
            self.html,
        )
        self.assertIn(
            ".trajectory-grid[hidden]",
            self.html,
        )

    def test_trajectory_has_accessible_busy_and_focus_states(self) -> None:
        self.assertIn('aria-busy="false"', self.html)
        self.assertIn('id="trajectory-status"', self.html)
        self.assertIn('tabindex="-1"', self.html)
        self.assertIn(
            'trajectorySection.setAttribute("aria-busy", "true")',
            self.html,
        )
        self.assertIn("trajectoryTitle.focus", self.html)
        self.assertIn(
            'window.addEventListener("pagehide"',
            self.html,
        )

    def test_interpretation_distinguishes_next_token_and_trajectory(self) -> None:
        self.assertIn(
            "Primární výsledek výše popisuje jediný next-token krok.",
            self.html,
        )

    def test_controlled_path_and_phrase_score_contracts_are_present(self) -> None:
        self.assertIn('id="aligned-path-tool"', self.html)
        self.assertIn('id="aligned-continuation"', self.html)
        self.assertIn('id="run-aligned-path"', self.html)
        self.assertIn('fetchJson("/api/steering/aligned-path"', self.html)
        self.assertIn(
            '"steering_aligned_path"',
            self.html,
        )
        self.assertIn('id="phrase-score-tool"', self.html)
        self.assertIn('id="phrase-pairs"', self.html)
        self.assertIn('id="run-phrase-score"', self.html)
        self.assertIn('fetchJson("/api/steering/phrase-score"', self.html)
        self.assertIn(
            "positive_minus_negative_mean_logprob",
            self.html,
        )
        self.assertIn("pairs_strictly_increasing", self.html)
        self.assertIn("const monotonicTrend", self.html)
        self.assertIn('return "flat"', self.html)
        self.assertIn("nerozhodné ≈", self.html)
        self.assertIn("partialPhrasePairCount", self.html)
        self.assertIn("invalidateControlledResults", self.html)
        self.assertIn("item?.negative?.token_count", self.html)

    def test_trajectory_token_handoff_preserves_exact_context(self) -> None:
        self.assertIn("const trajectoryTokenContexts", self.html)
        self.assertIn("prefixBeforeText", self.html)
        self.assertIn("prefixAfterText", self.html)
        self.assertIn("data-open-garden-branch", self.html)
        self.assertIn(
            'source: "steering_trajectory"',
            self.html,
        )
        self.assertIn(
            "token_ids: mode === \"after\" ? context.afterIds : context.beforeIds",
            self.html,
        )
        self.assertIn("window.localStorage.setItem", self.html)
        self.assertIn(
            "je samostatná steerovaná generace",
            self.html,
        )

    def test_specificity_controls_are_wired_to_long_running_endpoints(self) -> None:
        self.assertIn('id="specificity-section"', self.html)
        self.assertIn('id="sham-control-tool"', self.html)
        self.assertIn('id="layer-sweep-tool"', self.html)
        self.assertIn('fetchJson("/api/steering/sham-control"', self.html)
        self.assertIn('fetchJson("/api/steering/layer-sweep"', self.html)
        self.assertIn('"steering_sham_control"', self.html)
        self.assertIn('"steering_layer_sweep"', self.html)
        self.assertIn(
            "const VALIDATION_FETCH_TIMEOUT_MS = 3_660_000",
            self.html,
        )
        self.assertIn(
            'runShamControlButton.addEventListener("click", runShamControl)',
            self.html,
        )
        self.assertIn(
            'runLayerSweepButton.addEventListener("click", runLayerSweep)',
            self.html,
        )
        self.assertIn('showOverlay([', self.html)
        self.assertIn("modelWorkerBusy()", self.html)
        self.assertIn("health.worker?.busy", self.html)
        self.assertIn(
            'shamControlTool.setAttribute("aria-busy", "true")',
            self.html,
        )
        self.assertIn(
            'layerSweepTool.setAttribute("aria-busy", "true")',
            self.html,
        )

    def test_specificity_requires_complete_pairs_and_invalidates_stale_results(
        self,
    ) -> None:
        self.assertIn("const specificityEligibility = () =>", self.html)
        self.assertIn("partialPhrasePairCount()", self.html)
        self.assertIn(
            "parsedLayers.layers.includes(referenceLayer)",
            self.html,
        )
        self.assertIn(
            'invalidateSpecificityResults("matched pokračování se změnila")',
            self.html,
        )
        self.assertIn(
            'invalidateShamResult("počet shamů se změnil")',
            self.html,
        )
        self.assertIn(
            'invalidateLayerSweepResult("seznam vrstev se změnil")',
            self.html,
        )

    def test_specificity_copy_and_rendering_keep_inference_descriptive(self) -> None:
        self.assertIn("deskriptivní Monte Carlo tail fraction", self.html)
        self.assertIn("nikoli o obecně", self.html)
        self.assertIn("kalibrovanou p-hodnotu", self.html)
        self.assertIn("znovu lokálně", self.html)
        self.assertIn("deskriptivní maximum tohoto sweepu", self.html)
        self.assertIn("baseline_prefix_residual_norm", self.html)
        self.assertIn("prefix_relative_shift_norm", self.html)
        self.assertIn("direction.effective_cosine_to_target_after_cast", self.html)
        self.assertIn(
            "data?.randomization?.all_effective_norm_matches_passed !== true",
            self.html,
        )

    def test_specificity_exports_exact_json_and_useful_csv(self) -> None:
        self.assertIn('id="export-sham-json"', self.html)
        self.assertIn('id="export-sham-csv"', self.html)
        self.assertIn('id="export-layer-json"', self.html)
        self.assertIn('id="export-layer-csv"', self.html)
        self.assertIn(
            'exportShamJsonButton.addEventListener("click"',
            self.html,
        )
        self.assertIn(
            'exportShamCsvButton.addEventListener("click"',
            self.html,
        )
        self.assertIn(
            'exportLayerJsonButton.addEventListener("click"',
            self.html,
        )
        self.assertIn(
            'exportLayerCsvButton.addEventListener("click"',
            self.html,
        )
        self.assertIn("const exportEnvelope =", self.html)
        self.assertIn("request,", self.html)
        self.assertIn("response", self.html)
        self.assertIn("const csvCell =", self.html)
        self.assertIn('replaceAll(\'"\', \'""\')', self.html)
        self.assertIn('"raw_json"', self.html)
        self.assertIn("continuationPairExportRows", self.html)
        self.assertIn("direction_sha256", self.html)
        self.assertIn("contrast_negative_nats_per_token", self.html)
        self.assertIn("contrast_baseline_nats_per_token", self.html)
        self.assertIn("contrast_positive_nats_per_token", self.html)
        self.assertIn("negative_mean_logprob_at_negative_alpha", self.html)
        self.assertIn("positive_mean_logprob_at_positive_alpha", self.html)
        self.assertIn(
            "monte_carlo_reference_tail_fraction",
            self.html,
        )

    def test_redesign_has_non_linear_workbench_stages(self) -> None:
        for stage_id in (
            "stage-direction",
            "stage-immediate",
            "trajectory-section",
            "validation-section",
            "sham-control-tool",
            "layer-sweep-tool",
        ):
            self.assertIn(f'id="{stage_id}"', self.html)
        self.assertIn('data-stage="direction"', self.html)
        self.assertIn('data-stage="immediate"', self.html)
        self.assertIn('data-stage="trajectory"', self.html)
        self.assertIn('data-stage="held-out"', self.html)
        self.assertIn('data-stage="sham"', self.html)
        self.assertIn('data-stage="sweep"', self.html)
        self.assertIn("probability-garden:workbench-state", self.html)
        self.assertIn(
            "ProbabilityGardenShell?.activateStage?.(",
            self.html,
        )
        self.assertIn(
            '"(prefers-reduced-motion: reduce)"',
            self.html,
        )
        self.assertIn("trajectoryReady ? \"ready\" : \"locked\"", self.html)
        self.assertIn("specificity.shamValid ? \"ready\" : \"locked\"", self.html)

    def test_static_ids_and_accessibility_references_are_valid(self) -> None:
        collector = _StaticReferenceCollector()
        collector.feed(self.html)
        counts = {
            element_id: collector.ids.count(element_id)
            for element_id in collector.ids
        }
        duplicates = sorted(
            element_id
            for element_id, count in counts.items()
            if count > 1
        )
        self.assertEqual([], duplicates)
        known_ids = set(collector.ids)
        missing = sorted(
            (kind, target)
            for kind, target in collector.references
            if target not in known_ids
        )
        self.assertEqual([], missing)

    def test_immediate_effect_has_lossless_and_tabular_exports(self) -> None:
        self.assertIn('id="export-steering-json"', self.html)
        self.assertIn('id="export-steering-csv"', self.html)
        self.assertIn("const nextTokenCsvRows = () =>", self.html)
        self.assertIn('"steering_next_token"', self.html)
        self.assertIn("exportSteeringJsonButton.addEventListener", self.html)
        self.assertIn("exportSteeringCsvButton.addEventListener", self.html)

    def test_display_top_k_does_not_invalidate_the_frozen_source(self) -> None:
        source_start = self.html.index(
            "const sourceConfigurationSignature = ("
        )
        source_end = self.html.index(
            "const resultIsStale = () =>",
            source_start,
        )
        source_signature = self.html[source_start:source_end]
        self.assertNotIn("top_k:", source_signature)
        self.assertIn(
            "state.resultSourceSignature !== sourceConfigurationSignature()",
            self.html,
        )
        self.assertIn(
            "source_result: state.resultSourceSignature",
            self.html,
        )
        self.assertIn(
            "Změnil se pouze počet zobrazených top-k kandidátů.",
            self.html,
        )


if __name__ == "__main__":
    unittest.main()
