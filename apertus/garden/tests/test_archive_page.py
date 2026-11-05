from __future__ import annotations

import unittest
from pathlib import Path


GARDEN_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_PAGE = GARDEN_DIR / "static" / "archive.html"


class ArchivePageContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = ARCHIVE_PAGE.read_text(encoding="utf-8")

    def test_text_surprisal_is_filterable_and_has_a_detail_renderer(self) -> None:
        self.assertIn(
            '<option value="text_surprisal">Surprisal textu</option>',
            self.html,
        )
        self.assertIn('text_surprisal: "Surprisal textu"', self.html)
        self.assertIn("const renderTextSurprisal = (record) =>", self.html)
        self.assertIn(
            'record.kind === "text_surprisal"',
            self.html,
        )
        self.assertIn("? renderTextSurprisal", self.html)

    def test_detail_keeps_token_telemetry_numeric_and_shows_alternatives(self) -> None:
        self.assertIn('class="surprisal-telemetry"', self.html)
        self.assertIn("token.surprisal_bits", self.html)
        self.assertIn("token.logprob", self.html)
        self.assertIn("token.entropy_bits", self.html)
        self.assertIn("token.effective_tokens", self.html)
        self.assertIn("asArray(token.alternatives)", self.html)
        self.assertIn('class="surprisal-alternatives"', self.html)
        self.assertIn(
            "barva nenese informaci o jejich velikosti",
            self.html,
        )
        self.assertNotIn("--surprisal-color", self.html)
        self.assertNotIn("surprisal-heat", self.html)

    def test_detail_escapes_text_and_exposes_summary_and_provenance(self) -> None:
        self.assertIn(
            '<pre class="surprisal-source">${escapeHtml(originalText)}</pre>',
            self.html,
        )
        self.assertIn("summary.total_surprisal_bits", self.html)
        self.assertIn("summary.mean_surprisal_bits", self.html)
        self.assertIn("summary.perplexity", self.html)
        self.assertIn("summary.bits_per_character", self.html)
        self.assertIn("summary.bits_per_utf8_byte", self.html)
        self.assertIn("summary.mean_entropy_bits", self.html)
        self.assertIn("asObject(record.provenance)", self.html)
        self.assertIn("asObject(response.provenance)", self.html)
        self.assertIn("Provenience a podmínky výpočtu", self.html)
        self.assertIn('metric("Instance serveru", serverInstanceId)', self.html)
        self.assertIn(
            'metric("Normalizace", formatDescriptor(response.normalization))',
            self.html,
        )

    def test_detail_renders_general_aggregations_without_poetic_claims(self) -> None:
        self.assertIn(
            "const aggregations = asObject(",
            self.html,
        )
        self.assertIn(
            'response.aggregation_method || payload.aggregation_method',
            self.html,
        )
        self.assertIn(
            '"whitespace_span"',
            self.html,
        )
        self.assertIn(
            '"sentence_like"',
            self.html,
        )
        self.assertIn(
            "Obecné agregace úseků",
            self.html,
        )
        self.assertIn(
            "pravidlové segmentace, nikoli jazykově autoritativní analýza",
            self.html,
        )
        self.assertIn(
            "sama neměří",
            self.html,
        )
        self.assertIn(
            "strukturovanost, poetičnost ani pořadí",
            self.html,
        )
        self.assertIn(
            "aggregationSummary.raw_mad",
            self.html,
        )
        self.assertIn(
            "span.surprisal_bits_per_character",
            self.html,
        )
        self.assertIn(
            "span.ambiguous_token_indices",
            self.html,
        )
        self.assertIn(
            '<td><code>${escapeHtml(span.text ?? "")}</code></td>',
            self.html,
        )
        self.assertIn("const complete = span.complete === true", self.html)
        self.assertGreaterEqual(self.html.count('"— (neúplné)"'), 3)
        self.assertIn(
            "Sčítají se překvapení tokenů přiřazených podle jediného",
            self.html,
        )
        self.assertIn(
            "může\n                  nést i přilehlé bílé místo",
            self.html,
        )
        self.assertNotIn("rým", self.html.lower())
        self.assertNotIn("slovní druh", self.html.lower())

    def test_text_surprisal_comparison_is_filterable_and_neutral(self) -> None:
        self.assertIn(
            '<option value="text_surprisal_comparison">'
            "Srovnání surprisal textů</option>",
            self.html,
        )
        self.assertIn(
            'text_surprisal_comparison: "Srovnání surprisal textů"',
            self.html,
        )
        self.assertIn(
            "const renderTextSurprisalComparison = (record) =>",
            self.html,
        )
        self.assertIn(
            'record.kind === "text_surprisal_comparison"',
            self.html,
        )
        self.assertIn(
            "? renderTextSurprisalComparison",
            self.html,
        )

        start = self.html.index(
            "const renderTextSurprisalComparison = (record) =>"
        )
        end = self.html.index(
            "const renderDetail = (record) =>",
            start,
        )
        renderer = self.html[start:end]
        self.assertIn(
            '<pre class="surprisal-source">${escapeHtml(textA)}</pre>',
            renderer,
        )
        self.assertIn(
            '<pre class="surprisal-source">${escapeHtml(textB)}</pre>',
            renderer,
        )
        self.assertIn("escapeHtml(labelA)", renderer)
        self.assertIn("escapeHtml(labelB)", renderer)
        self.assertIn("comparison.delta_definition", renderer)
        self.assertIn("comparisonMetric.delta_b_minus_a", renderer)
        self.assertIn(
            "mean_surprisal_bits",
            renderer,
        )
        self.assertIn(
            "bits_per_non_whitespace_character",
            renderer,
        )
        self.assertIn(
            "median_whitespace_span_bits_per_non_whitespace_character",
            renderer,
        )
        self.assertIn(
            "whitespace_span_iqr_bits_per_non_whitespace_character",
            renderer,
        )
        self.assertIn(
            "whitespace_span_raw_mad_bits_per_non_whitespace_character",
            renderer,
        )
        self.assertIn("Společná provenience", renderer)
        self.assertIn(
            "jednotlivé tokenové řady A a B zůstávají oddělené",
            renderer,
        )
        self.assertNotIn(".tokens", renderer)
        self.assertNotIn("vítěz", renderer.lower())
        self.assertNotIn("žánr", renderer.lower())
        self.assertNotIn("kauz", renderer.lower())

    def test_context_contrast_is_filterable_and_has_a_detail_renderer(self) -> None:
        self.assertIn(
            '<option value="context_contrast">Kontrast kontextů</option>',
            self.html,
        )
        self.assertIn(
            'context_contrast: "Kontrast kontextů"',
            self.html,
        )
        self.assertIn(
            "const renderContextContrast = (record) =>",
            self.html,
        )
        self.assertIn(
            'record.kind === "context_contrast"',
            self.html,
        )
        self.assertIn("? renderContextContrast", self.html)

        start = self.html.index(
            "const renderContextContrast = (record) =>"
        )
        end = self.html.index("const renderDetail = (record) =>", start)
        renderer = self.html[start:end]
        self.assertIn("information_advantage_a_over_b_bits", renderer)
        self.assertIn("root.js_divergence_bits", renderer)
        self.assertIn("persistent_half_life_step", renderer)
        self.assertIn("persistence.states", renderer)
        self.assertIn("Dva prefixy a společná cesta", renderer)
        self.assertIn("formatRatio(score.probability_ratio_a_over_b)", renderer)
        self.assertIn('halfLife === null ? "nepozorován"', renderer)
        self.assertNotIn('halfLife === null ? "nenastal"', renderer)

    def test_exports_use_native_downloads_and_report_the_handoff(self) -> None:
        self.assertIn(
            'download="apertus-experiments.json"',
            self.html,
        )
        self.assertIn(
            'download="apertus-experiments.csv"',
            self.html,
        )
        self.assertIn(
            "const announceArchiveExport = (filename) =>",
            self.html,
        )
        self.assertIn(
            "Stažení ${filename} bylo předáno prohlížeči",
            self.html,
        )
        self.assertIn("Export archivu JSON", self.html)
        self.assertIn("Export archivu CSV", self.html)
        self.assertIn("Export typu JSON", self.html)
        self.assertIn("Export typu CSV", self.html)
        self.assertIn(
            'data-export-record',
            self.html,
        )
        self.assertIn(
            '}/export.json"',
            self.html,
        )
        self.assertIn(
            '}/export.csv"',
            self.html,
        )
        self.assertIn(
            'download="apertus-experiment-${escapeHtml(record.id)}.json"',
            self.html,
        )
        self.assertIn(
            'download="apertus-experiment-${escapeHtml(record.id)}.csv"',
            self.html,
        )
        self.assertIn(
            "announceArchiveExport(exportLink.dataset.exportFilename)",
            self.html,
        )
        self.assertNotIn("event.preventDefault()", self.html)
        self.assertNotIn("URL.createObjectURL(blob)", self.html)

    def test_lens_renderer_preserves_raw_vs_jacobian_and_provenance(self) -> None:
        start = self.html.index("const renderLens = (record) =>")
        end = self.html.index(
            "const renderTextSurprisal = (record) =>",
            start,
        )
        renderer = self.html[start:end]
        self.assertIn('reader.kind === "raw_vs_jacobian_lens"', renderer)
        self.assertIn(
            'layers.some((layer) => layer?.jacobian)',
            renderer,
        )
        self.assertIn(
            'const secondVariantKey = isJacobianComparison ? "jacobian" : "tuned"',
            renderer,
        )
        self.assertIn('"RAW × J‑LENS"', renderer)
        self.assertIn("result.artifact", renderer)
        self.assertIn("result.transfer_probe", renderer)
        self.assertIn('"Přenosová sonda"', renderer)
        self.assertIn("artifact.canonical_manifest_sha256", renderer)
        self.assertIn("artifact.weights?.sha256", renderer)
        self.assertIn("artifact.compatibility", renderer)
        self.assertIn(
            "RAW a J‑LENS jsou dvě čtečky téhož uloženého mezistavu",
            renderer,
        )
        self.assertIn(
            "Jejich shoda ani nesoulad samy o sobě neurčují",
            renderer,
        )
        self.assertNotIn("J‑LENS aproximuje", renderer)
        self.assertNotIn("J‑LENS je přesnější", renderer)


if __name__ == "__main__":
    unittest.main()
