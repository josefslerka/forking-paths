from __future__ import annotations

import unittest
from pathlib import Path


GARDEN_DIR = Path(__file__).resolve().parents[1]
SURPRISE_PAGE = GARDEN_DIR / "static" / "surprise.html"
SHELL_JS = GARDEN_DIR / "static" / "workbench-shell.js"


class SurprisePageContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = SURPRISE_PAGE.read_text(encoding="utf-8")
        cls.shell = SHELL_JS.read_text(encoding="utf-8")

    def test_page_uses_workbench_shell_and_global_navigation(self) -> None:
        self.assertIn('<body data-workbench-page="surprise">', self.html)
        self.assertIn('href="/static/workbench-shell.css"', self.html)
        self.assertIn('src="/static/workbench-shell.js"', self.html)
        self.assertIn('id: "surprise"', self.shell)
        self.assertIn('href: "/prekvapeni"', self.shell)
        self.assertIn('label: "Překvapení textu"', self.shell)
        self.assertIn('meta: "hotový text"', self.shell)

    def test_tokenization_handoff_can_prefill_and_run_once(self) -> None:
        self.assertIn(
            "const initialParams = new URLSearchParams(window.location.search)",
            self.html,
        )
        self.assertIn('const initialText = initialParams.get("text")', self.html)
        self.assertIn(
            'const autoRunFromQuery = initialParams.get("run") === "1"',
            self.html,
        )
        self.assertIn("sourceText.value = initialText", self.html)
        self.assertIn("autoRunPending: Boolean(", self.html)
        self.assertIn("state.autoRunPending = false", self.html)
        self.assertIn("await analyze()", self.html)

    def test_analysis_posts_text_top_k_and_identity_safely(self) -> None:
        self.assertIn('const endpoint = check.compareMode', self.html)
        self.assertIn('? "/api/text-surprisal-comparison"', self.html)
        self.assertIn(': "/api/text-surprisal"', self.html)
        self.assertIn("const data = await fetchJson(endpoint", self.html)
        self.assertIn('text: check.text', self.html)
        self.assertIn('top_k: check.topK', self.html)
        self.assertIn('model_id: check.identity.modelId', self.html)
        self.assertIn(
            'model_generation: check.identity.modelGeneration',
            self.html,
        )
        self.assertIn(
            'server_instance_id: check.identity.serverInstanceId',
            self.html,
        )
        self.assertIn('"text_surprisal"', self.html)
        self.assertIn('"raw_model"', self.html)
        self.assertIn('data?.temperature !== 1', self.html)
        self.assertIn('state.signature !== currentSignature()', self.html)

    def test_busy_overlay_hides_without_aborting_request(self) -> None:
        self.assertIn('id="hide-wait"', self.html)
        self.assertIn(
            'hideWaitButton.addEventListener("click", hideWait)',
            self.html,
        )
        start = self.html.index("const hideWait = () => {")
        end = self.html.index("const trapOverlayFocus", start)
        handler = self.html[start:end]
        self.assertIn("hideOverlay({restoreFocus: true})", handler)
        self.assertIn("požadavek nezastavilo", handler)
        self.assertNotIn(".abort(", handler)
        self.assertNotIn("state.busy = false", handler)
        self.assertIn('window.addEventListener("pagehide"', self.html)
        self.assertIn("state.controller?.abort()", self.html)

    def test_copy_separates_surprisal_entropy_and_truth(self) -> None:
        self.assertIn("Realizované překvapení I = −log₂ p(token)", self.html)
        self.assertIn("Entropie H popisuje celé pole možností", self.html)
        self.assertIn("vysoké I + nízké H", self.html)
        self.assertIn("vysoké I + vysoké H", self.html)
        self.assertIn("Tokenová predikce není pravda", self.html)
        self.assertIn("nikoli pravdivost", self.html)
        self.assertIn("První token má k dispozici pouze BOS", self.html)

    def test_color_scale_is_fixed_and_not_run_normalized(self) -> None:
        for boundary in (
            "0–1 bit",
            "1–3 bity",
            "3–6 bitů",
            "6–10 bitů",
            "10+ bitů",
        ):
            self.assertIn(boundary, self.html)
        self.assertIn("škála se nepřizpůsobuje běhu", self.html)
        self.assertIn('if (bits < 1) return "surprise-0"', self.html)
        self.assertIn('if (bits < 3) return "surprise-1"', self.html)
        self.assertIn('if (bits < 6) return "surprise-2"', self.html)
        self.assertIn('if (bits < 10) return "surprise-3"', self.html)
        self.assertNotIn("maximumSurprisal", self.html)

    def test_raw_tokens_remain_primary_and_accessible(
        self,
    ) -> None:
        self.assertIn('id="token-stream"', self.html)
        self.assertIn('class="token-chip ${surprisalClass', self.html)
        self.assertIn('data-token-index="${index}"', self.html)
        self.assertIn('aria-pressed="false"', self.html)
        self.assertIn("ArrowLeft", self.html)
        self.assertIn("ArrowRight", self.html)
        self.assertIn("syrové tokeny zůstávají primárním měřením", self.html)
        self.assertIn("Zobrazení záměrně zachovává tokenovou rovinu", self.html)

    def test_derived_span_aggregation_is_accessible_and_conservative(
        self,
    ) -> None:
        self.assertIn('id="aggregation-title">Agregované úseky', self.html)
        self.assertIn('label for="aggregation-unit"', self.html)
        self.assertIn('id="aggregation-unit"', self.html)
        self.assertIn('value="whitespace_span"', self.html)
        self.assertIn('value="orthographic_word"', self.html)
        self.assertIn('value="line"', self.html)
        self.assertIn('value="sentence_like"', self.html)
        self.assertIn('aria-live="polite"', self.html)
        self.assertIn('aria-describedby="aggregation-caveat"', self.html)
        self.assertIn("nikoli autoritativně určené slovo", self.html)
        self.assertIn("Větný úsek vzniká pravidlovým dělením", self.html)
        self.assertIn(
            "spolehlivý znakový rozsah protíná právě jeden úsek",
            self.html,
        )
        self.assertIn("může proto", self.html)
        self.assertIn("nést i přilehlé bílé místo", self.html)
        self.assertIn(
            "Medián, IQR a MAD popisují pouze disperzi",
            self.html,
        )
        self.assertIn("Neměří jejich pořadí, strukturovanost", self.html)
        self.assertIn("poetičnost ani kvalitu textu", self.html)
        self.assertNotIn("rým", self.html.lower())
        self.assertNotIn("funkční slova", self.html.lower())
        self.assertIn('id="aggregation-status"', self.html)
        self.assertIn('role="status"', self.html)
        list_start = self.html.index('id="aggregation-list"')
        list_end = self.html.index("></ol>", list_start)
        self.assertNotIn("aria-live", self.html[list_start:list_end])

    def test_missing_numeric_values_are_not_rendered_as_zero(self) -> None:
        finite_start = self.html.index("const finite =")
        finite_end = self.html.index("const asArray", finite_start)
        finite = self.html[finite_start:finite_end]
        self.assertIn("value === null", finite)
        self.assertIn("value === undefined", finite)
        self.assertIn('value === ""', finite)

    def test_aggregation_summary_and_spans_use_neutral_metrics(self) -> None:
        for metric in (
            "Počet úseků",
            "Medián I / nebílý znak",
            "IQR I / nebílý znak",
            "MAD I / nebílý znak",
            "Pokrytí tokenů",
            "ΣI",
            "I / nebílý znak",
        ):
            self.assertIn(metric, self.html)
        self.assertIn("nejednoznačných tokenů", self.html)
        self.assertNotIn("hraničních tokenů", self.html)
        self.assertIn('class="aggregation-span"', self.html)
        self.assertIn("spanCharacterRange", self.html)
        self.assertIn("span?.token_count", self.html)
        self.assertIn("aggregation?.summary", self.html)
        self.assertIn("summary?.median", self.html)
        self.assertIn("summary?.iqr", self.html)
        self.assertIn("summary?.raw_mad", self.html)
        self.assertIn("summary?.complete_span_count", self.html)
        self.assertIn("span?.character_start", self.html)
        self.assertIn("span?.character_end", self.html)
        self.assertIn("span?.complete !== false", self.html)
        self.assertIn("— (neúplné)", self.html)
        render_start = self.html.index("const renderAggregations")
        render_end = self.html.index("const orthographicWords", render_start)
        render = self.html[render_start:render_end]
        self.assertNotIn("surprisalClass(", render)
        self.assertNotIn("surprise-", render)

    def test_missing_aggregation_data_falls_back_without_hiding_tokens(
        self,
    ) -> None:
        self.assertIn("if (!aggregations || typeof aggregations", self.html)
        self.assertIn("if (!aggregation || !spans.length)", self.html)
        self.assertIn("aggregationEmpty.hidden = false", self.html)
        self.assertIn("Syrová tokenová stopa", self.html)
        self.assertNotIn("data?.aggregations", self.html[
            self.html.index("const assertResponse"):
            self.html.index("const renderTokenDetails")
        ])

    def test_aggregation_selector_does_not_invalidate_or_recompute_result(
        self,
    ) -> None:
        signature = self.html[
            self.html.index("const currentSignature"):
            self.html.index("const resultIsStale")
        ]
        self.assertNotIn("aggregationUnit", signature)
        listener = self.html[
            self.html.index(
                'aggregationUnit.addEventListener("change"'
            ):
            self.html.index(
                'exportJsonButton.addEventListener("click"'
            )
        ]
        self.assertIn("renderAggregations()", listener)
        self.assertNotIn("analyze()", listener)

    def test_token_selection_shows_requested_diagnostics(self) -> None:
        for metric in (
            "Skutečná p",
            "Překvapení I",
            "Globální pořadí",
            "Entropie H",
            "Efektivní podpora",
            "Top‑1 gap",
        ):
            self.assertIn(metric, self.html)
        self.assertIn("token.alternatives", self.html)
        self.assertIn("alternative?.probability", self.html)
        self.assertIn("alternative?.rank", self.html)

    def test_psycholinguistic_word_view_is_explicit_and_traceable(self) -> None:
        for identifier in (
            'id="display-mode-tokens"',
            'id="display-mode-words"',
            'id="psycholinguistic-view"',
            'id="psych-word-list"',
            'id="export-word-csv"',
        ):
            self.assertIn(identifier, self.html)
        self.assertIn('initialParams.get("view")', self.html)
        self.assertIn('aggregationFor(trace, "orthographic_word")', self.html)
        self.assertIn('aggregationFor(trace, "punctuation")', self.html)
        self.assertIn("word?.word_surprisal_bits", self.html)
        self.assertIn("word?.pre_word_next_token_entropy_bits", self.html)
        self.assertIn("Slovní surprisal", self.html)
        self.assertIn("Entropie dalšího tokenu před slovem", self.html)
        self.assertIn("Surprisal interpunkce", self.html)
        self.assertIn("psych-punctuation-row", self.html)
        self.assertIn("ΔI oproti předchozímu slovu", self.html)
        self.assertIn("Tokenové střepy", self.html)
        self.assertIn("next_token_vocabulary", self.html)

    def test_word_comparison_delta_requires_exact_surface_and_position(self) -> None:
        self.assertIn("surfaceA === surfaceB", self.html)
        self.assertIn("exactSurfaceAlignment", self.html)
        self.assertIn("stejné pozici pozorované jednotky", self.html)
        self.assertIn("delta_surprisal_bits_b_minus_a", self.html)
        self.assertIn(
            '"exact_unicode_surface_and_kind_at_same_observed_unit_ordinal"',
            self.html,
        )
        self.assertIn("ΔI se nepočítá", self.html)

    def test_dedicated_word_csv_keeps_subwords_and_boundary_diagnostics(
        self,
    ) -> None:
        self.assertIn("const wordCsvRows = () =>", self.html)
        self.assertIn('? "orthographic_word"', self.html)
        self.assertIn(': "punctuation"', self.html)
        self.assertIn('row_type: "psycholinguistic_unit_pair"', self.html)
        self.assertIn("unit_kind", self.html)
        self.assertIn("pre_unit_next_token_entropy_bits", self.html)
        self.assertIn("token_fragments", self.html)
        self.assertIn("pre_word_entropy_scope", self.html)
        self.assertIn("boundary_contaminated_token_count", self.html)
        self.assertIn("rowsToCsv(rows)", self.html)
        self.assertIn("wordFilename()", self.html)

    def test_json_csv_and_archive_save_are_lossless(self) -> None:
        self.assertIn('id="export-json"', self.html)
        self.assertIn('id="export-csv"', self.html)
        self.assertIn('id="save-experiment"', self.html)
        self.assertIn('schema: "apertus.experiment-record"', self.html)
        self.assertIn('kind: "text_surprisal"', self.html)
        self.assertIn("result: data", self.html)
        self.assertIn("request: {...state.request}", self.html)
        self.assertIn("const csvRows = () =>", self.html)
        self.assertIn('row_type: "token"', self.html)
        self.assertIn('row_type: "alternative"', self.html)
        self.assertIn('row_type: "aggregation_summary"', self.html)
        self.assertIn('row_type: "span"', self.html)
        self.assertIn("aggregation_kind: aggregationKind", self.html)
        self.assertIn("surprisal_bits_per_character", self.html)
        for field in (
            "statistic_span_count: summary.count",
            "mean: summary.mean",
            "q1: summary.q1",
            "q3: summary.q3",
            "min: summary.min",
            "max: summary.max",
            "assigned_token_count: assignedTokenCount",
            "unassigned_token_count: unassignedTokenCount",
            "total_token_count: totalTokenCount",
            "complete_span_count: summary.complete_span_count",
            "incomplete_span_count: summary.incomplete_span_count",
            "aggregation_label: aggregation?.label_cs",
            "aggregation_unit: aggregation?.unit ?? aggregationKind",
            "token_indices: span.token_indices",
            "token_ids: span.token_ids",
            "surprisal_bits_per_token: span.surprisal_bits_per_token",
            "character_count: span.character_count",
            "span.non_whitespace_character_count",
            "span.covered_non_whitespace_character_count",
            "character_coverage_ratio: span.character_coverage_ratio",
            "complete: span.complete",
            "ambiguous_token_indices: span.ambiguous_token_indices",
        ):
            self.assertIn(field, self.html)
        self.assertIn('row_type: "provenance"', self.html)
        self.assertIn(
            "aggregation_method: data.aggregation_method ?? null",
            self.html,
        )
        self.assertIn("aggregations: data.aggregations ?? null", self.html)
        self.assertIn("raw_json: token", self.html)
        self.assertIn("/^\\s*[=+\\-@]/.test(text)", self.html)
        self.assertIn("Boolean(state.savedRecordId)", self.html)
        self.assertIn('fetchJson("/api/experiments"', self.html)
        self.assertIn("body: JSON.stringify({record})", self.html)

    def test_client_character_limit_matches_server_contract(self) -> None:
        self.assertIn('maxlength="8192"', self.html)
        self.assertIn("textA.length > 8192", self.html)
        self.assertIn("textB.length > 8192", self.html)
        self.assertNotIn('maxlength="20000"', self.html)

    def test_ab_mode_is_progressive_and_removable(self) -> None:
        self.assertIn('id="add-text-b"', self.html)
        self.assertIn("+ Přidat text B", self.html)
        self.assertIn('id="comparison-inputs"', self.html)
        self.assertIn('id="source-text-b"', self.html)
        self.assertIn('id="remove-text-b"', self.html)
        self.assertIn("Odebrat text B", self.html)
        self.assertIn("comparisonInputs.hidden = !state.compareMode", self.html)
        self.assertIn("addTextBButton.hidden = state.compareMode", self.html)
        self.assertIn("setComparisonMode(true)", self.html)
        self.assertIn("setComparisonMode(false)", self.html)

    def test_ab_request_is_one_atomic_server_job(self) -> None:
        analyze = self.html[
            self.html.index("const analyze = async"):
            self.html.index("const newRecordId", self.html.index(
                "const analyze = async"
            ))
        ]
        self.assertIn("text_a: check.textA", analyze)
        self.assertIn("text_b: check.textB", analyze)
        self.assertIn("{label_a: enteredLabelA}", analyze)
        self.assertIn("{label_b: enteredLabelB}", analyze)
        self.assertIn("...commonRequest", analyze)
        self.assertIn('"/api/text-surprisal-comparison"', analyze)
        self.assertEqual(analyze.count("await fetchJson(endpoint"), 1)
        self.assertIn("assertComparisonResponse(data, request)", analyze)

    def test_ab_stale_signature_tracks_both_texts_not_labels(self) -> None:
        signature = self.html[
            self.html.index("const currentSignature"):
            self.html.index("const resultIsStale")
        ]
        self.assertIn('mode: state.compareMode ? "comparison" : "single"', signature)
        self.assertIn("text_a: normalizedInput()", signature)
        self.assertIn("text_b: state.compareMode ? normalizedInputB() : null", signature)
        self.assertNotIn("labelA", signature)
        self.assertNotIn("labelB", signature)
        self.assertIn(
            'labelA.addEventListener("input", refreshComparisonLabels)',
            self.html,
        )

    def test_ab_uses_server_metrics_and_never_derives_alignment(self) -> None:
        for metric in (
            "mean_surprisal_bits",
            "bits_per_non_whitespace_character",
            "median_whitespace_span_bits_per_non_whitespace_character",
            "whitespace_span_iqr_bits_per_non_whitespace_character",
            "whitespace_span_raw_mad_bits_per_non_whitespace_character",
        ):
            self.assertIn(metric, self.html)
        self.assertIn("const metrics = data.comparison.metrics", self.html)
        self.assertIn("metric.delta_b_minus_a", self.html)
        self.assertIn('delta_definition !== "B - A"', self.html)
        self.assertIn("Dvě samostatná měření, žádné zarovnávání", self.html)
        self.assertIn("pozičně ani slovně nezarovnávají", self.html)
        self.assertIn("délkově závislé, bez Δ", self.html)
        self.assertNotIn("tokenAlignment", self.html)
        self.assertNotIn("alignedTokens", self.html)

    def test_ab_renders_two_clickable_raw_traces_on_one_scale(self) -> None:
        self.assertIn('data-comparison-token-stream="${side}"', self.html)
        self.assertIn('data-comparison-side="${side}"', self.html)
        self.assertIn('data-comparison-token-index="${index}"', self.html)
        self.assertIn(
            'class="token-chip ${surprisalClass(token.surprisal_bits)}"',
            self.html,
        )
        self.assertIn("comparisonTraceMarkup(data.a", self.html)
        self.assertIn("comparisonTraceMarkup(data.b", self.html)
        self.assertIn("renderComparisonTokenDetails(", self.html)
        self.assertIn("comparisonAggregationUnit.value", self.html)
        self.assertIn('["a", "b"].forEach((side)', self.html)

    def test_ab_record_and_csv_keep_the_pair_lossless(self) -> None:
        self.assertIn('kind: "text_surprisal_comparison"', self.html)
        self.assertIn("a: data.a", self.html)
        self.assertIn("b: data.b", self.html)
        self.assertIn("labels,", self.html)
        self.assertIn("comparison,", self.html)
        self.assertIn("response: data", self.html)
        self.assertIn('row_type: "comparison_payload"', self.html)
        self.assertIn('row_type: "normalized_metric"', self.html)
        self.assertIn("comparison_method_id", self.html)
        self.assertIn("delta_definition", self.html)
        self.assertIn('side: side.toUpperCase()', self.html)
        self.assertIn('row_type: "alternative"', self.html)
        self.assertIn('row_type: "span"', self.html)
        self.assertIn("const labels = currentDisplayLabels()", self.html)

    def test_ab_copy_accessibility_and_mobile_layout_are_conservative(
        self,
    ) -> None:
        self.assertIn("nevysvětluje jeho příčinu", self.html)
        self.assertIn("App neposuzuje, zda jsou texty srovnatelné", self.html)
        self.assertIn("určení „lepšího“ textu", self.html)
        self.assertIn('id="comparison-status"', self.html)
        self.assertIn(
            "Srovnání hotovo: dvě samostatné tokenové stopy A a B.",
            self.html,
        )
        status_start = self.html.index('id="comparison-status"')
        status_end = self.html.index("></p>", status_start)
        self.assertIn('aria-live="polite"', self.html[status_start:status_end])
        mobile = self.html[
            self.html.index("@media (max-width: 720px)"):
            self.html.index("@media (prefers-reduced-motion: reduce)")
        ]
        self.assertIn(".comparison-inputs", mobile)
        self.assertIn(".comparison-total-grid", mobile)
        self.assertIn("grid-template-columns: 1fr;", mobile)

    def test_page_is_responsive_and_respects_reduced_motion(self) -> None:
        self.assertIn("@media (max-width: 980px)", self.html)
        self.assertIn("@media (max-width: 720px)", self.html)
        self.assertIn(".aggregation-controls", self.html)
        self.assertIn(".aggregation-summary", self.html)
        self.assertIn(".aggregation-span", self.html)
        mobile_start = self.html.index("@media (max-width: 720px)")
        mobile_end = self.html.index(
            "@media (prefers-reduced-motion: reduce)",
            mobile_start,
        )
        mobile = self.html[mobile_start:mobile_end]
        self.assertIn("grid-template-columns: 1fr;", mobile)
        self.assertIn("white-space: normal;", mobile)
        self.assertIn("overflow-wrap: anywhere;", mobile)
        self.assertIn("@media (prefers-reduced-motion: reduce)", self.html)
        self.assertIn("animation-duration: 0.01ms !important", self.html)
        self.assertIn('"(prefers-reduced-motion: reduce)"', self.html)
        self.assertIn('behavior: reducedMotion ? "auto" : "smooth"', self.html)


if __name__ == "__main__":
    unittest.main()
