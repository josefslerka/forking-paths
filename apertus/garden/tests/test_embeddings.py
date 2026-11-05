from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

from apertus.garden.api.post_routes import POST_ROUTES
from apertus.garden.server import GardenHandler, ModelState


GARDEN_DIR = Path(__file__).resolve().parents[1]
PAGE = GARDEN_DIR / "static" / "embedding.html"


class EmbeddingAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ModelState(model_path=Path("."), mock=True)
        self.state.status = "ready"
        self.state.model_generation = 7

    @staticmethod
    def contexts() -> list[dict]:
        return [
            {
                "label": "Kopec",
                "text": "Na kopci stojí zámek.",
                "token_index": 3,
            },
            {
                "label": "Dveře",
                "text": "Na dveřích je rozbitý zámek.",
                "token_index": 4,
            },
        ]

    def run_analysis(self) -> dict:
        return self.state.embeddings(
            self.contexts(),
            8,
            "mock",
            7,
            self.state.server_instance_id,
        )

    def test_same_id_has_one_row_and_contexts_diverge_after_l0(self) -> None:
        result = self.run_analysis()
        pair = result["pairs"][0]

        self.assertEqual(
            result["analysis_kind"],
            "input_embedding_context_trajectories",
        )
        self.assertEqual(result["embedding"]["dimension"], 64)
        self.assertEqual(result["execution"]["forward_passes"], 2)
        self.assertFalse(result["execution"]["future_suffix_used"])
        self.assertEqual(len(result["neighbors"]), 8)
        self.assertNotIn(
            result["selected_token"]["token_id"],
            {
                neighbor["token_id"]
                for neighbor in result["neighbors"]
            },
        )
        self.assertEqual(
            len(result["projection"]["points"]),
            len(result["neighbors"]) + 1,
        )
        self.assertEqual(len(result["contexts"][0]["layers"]), 33)
        self.assertAlmostEqual(pair["input_cosine_similarity"], 1.0)
        self.assertAlmostEqual(pair["input_cosine_distance"], 0.0)
        self.assertGreater(pair["max_distance"], 0.0)
        self.assertGreater(pair["final_cosine_distance"], 0.0)

    def test_each_context_exposes_its_actual_layer_state_and_shared_projection(self) -> None:
        result = self.run_analysis()
        projection = result["context_projection"]
        self.assertEqual(projection["method"], "shared_pca_unit_context_states_all_layers")
        self.assertEqual(projection["dimension"], 64)
        self.assertEqual(len(projection["contexts"]), 2)
        self.assertEqual(len(projection["contexts"][0]["points"]), 33)
        left, right = result["contexts"]
        self.assertEqual(left["layers"][0]["state_summary"], right["layers"][0]["state_summary"])
        self.assertEqual(left["layers"][0]["state_summary"]["float32_sha256"], result["embedding"]["float32_sha256"])
        self.assertNotEqual(left["layers"][-1]["state_summary"]["float32_sha256"], right["layers"][-1]["state_summary"]["float32_sha256"])
        for context in result["contexts"]:
            for point in context["layers"]:
                summary = point["state_summary"]
                self.assertEqual(summary["dimension"], 64)
                self.assertEqual(len(summary["coordinate_preview"]), 16)
                self.assertAlmostEqual(summary["norm"], point["norm"])

    def test_suffix_after_selected_occurrence_cannot_change_trajectory(
        self,
    ) -> None:
        first = self.state._embeddings_local(
            [
                {
                    "label": "A",
                    "text": "Na kopci zámek potom stojí.",
                    "token_index": 2,
                },
                {
                    "label": "B",
                    "text": "U dveří zámek zůstal viset.",
                    "token_index": 2,
                },
            ],
            5,
        )
        second = self.state._embeddings_local(
            [
                {
                    "label": "A",
                    "text": "Na kopci zámek úplně jiný konec.",
                    "token_index": 2,
                },
                {
                    "label": "B",
                    "text": "U dveří zámek a další cizí suffix.",
                    "token_index": 2,
                },
            ],
            5,
        )

        self.assertEqual(
            first["contexts"][0]["prefix_token_ids"],
            second["contexts"][0]["prefix_token_ids"],
        )
        self.assertEqual(
            first["contexts"][0]["layers"],
            second["contexts"][0]["layers"],
        )
        self.assertEqual(
            first["contexts"][1]["layers"],
            second["contexts"][1]["layers"],
        )
        self.assertEqual(
            first["pairs"][0]["points"],
            second["pairs"][0]["points"],
        )

    def test_mismatched_token_ids_are_rejected(self) -> None:
        contexts = self.contexts()
        contexts[1]["token_index"] = 0
        with self.assertRaisesRegex(ValueError, "stejné tokenové ID"):
            self.state._embeddings_local(contexts, 8)

    def test_repeat_uses_bounded_result_cache(self) -> None:
        first = self.run_analysis()
        second = self.run_analysis()

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(len(self.state.embedding_cache), 1)
        self.assertEqual(first["embedding"], second["embedding"])
        self.assertEqual(first["pairs"], second["pairs"])

    def test_on_demand_neighborhood_is_forward_free_and_cached(
        self,
    ) -> None:
        first = self.state._embedding_neighborhood_local(7, 6)
        second = self.state._embedding_neighborhood_local(7, 6)

        self.assertEqual(
            first["analysis_kind"],
            "input_embedding_token_neighborhood",
        )
        self.assertEqual(first["root"]["token_id"], 7)
        self.assertEqual(len(first["neighbors"]), 6)
        self.assertNotIn(
            7,
            {neighbor["token_id"] for neighbor in first["neighbors"]},
        )
        self.assertEqual(first["execution"]["forward_passes"], 0)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(second["timing_ms"]["total"], 0.0)
        self.assertEqual(len(self.state.embedding_neighborhood_cache), 1)
        self.assertEqual(
            first["graph_contract"]["layout"],
            "client_side_schematic_topology_not_embedding_geometry",
        )

    def test_on_demand_neighborhood_rejects_invalid_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "mimo slovník"):
            self.state._embedding_neighborhood_local(999, 6)
        with self.assertRaisesRegex(ValueError, "neighbor_count"):
            self.state._embedding_neighborhood_local(7, 2)

    def test_word_span_uses_closing_subtoken_with_honest_geometry(
        self,
    ) -> None:
        result = self.state._embeddings_local(
            [
                {
                    "label": "Kopec",
                    "text": "Na kopci stojí zámek.",
                    "token_index": 3,
                    "span_start_token_index": 3,
                    "span_end_token_index": 3,
                    "span_character_start": 15,
                    "span_character_end": 20,
                    "span_text": "zámek",
                },
                {
                    "label": "Dveře",
                    "text": "Na dveřích je rozbitý zámek.",
                    "token_index": 4,
                    "span_start_token_index": 4,
                    "span_end_token_index": 4,
                    "span_character_start": 22,
                    "span_character_end": 27,
                    "span_text": "zámek",
                },
            ],
            8,
            "word_span",
        )

        self.assertEqual(
            result["analysis_kind"],
            "word_span_closing_position_context_trajectories",
        )
        self.assertEqual(result["analysis_mode"], "word_span")
        self.assertEqual(result["word_span"]["text"], "zámek")
        self.assertEqual(result["word_span"]["token_count"], 1)
        self.assertEqual(len(result["neighbors"]), 8)
        self.assertIsNotNone(result["projection"])
        self.assertEqual(
            len(result["projection"]["points"]),
            len(result["neighbors"]) + result["word_span"]["token_count"],
        )
        self.assertEqual(
            result["projection"]["points"][0]["point_role"],
            "closing_fragment",
        )
        self.assertTrue(result["execution"]["vocabulary_scan"])
        self.assertEqual(
            result["embedding"]["role"],
            "closing_subtoken_input_row_not_whole_word",
        )
        self.assertTrue(result["contexts"][0]["tokens"][3]["selected"])
        self.assertTrue(
            result["contexts"][0]["tokens"][3][
                "closes_selected_span"
            ]
        )

    def test_multitoken_word_span_keeps_each_fragment_row(self) -> None:
        def fake_trace(text: str, *, label: str) -> dict:
            leading_id = 21 if text.startswith("A") else 22
            token_specs = [
                (leading_id, text[0:1], 0, 1),
                (12, text[1:4], 1, 4),
                (13, text[4:7], 4, 7),
                (14, text[7:8], 7, 8),
            ]
            tokens = []
            for index, (token_id, source, start, end) in enumerate(
                token_specs
            ):
                tokens.append(
                    {
                        "index": index,
                        "token_id": token_id,
                        "display": source,
                        "source_text": source,
                        "character_start": start,
                        "character_end": end,
                        "utf8_byte_start": len(
                            text[:start].encode("utf-8")
                        ),
                        "utf8_byte_end": len(
                            text[:end].encode("utf-8")
                        ),
                    }
                )
            boundaries = []
            ids = [item[0] for item in token_specs]
            for index in range(len(tokens) + 1):
                end = 0 if index == 0 else token_specs[index - 1][3]
                boundaries.append(
                    {
                        "prefix_text": text[:end],
                        "prefix_token_ids": [0, *ids[:index]],
                    }
                )
            return {
                "label": label,
                "text": text,
                "token_count": len(tokens),
                "tokens": tokens,
                "boundaries": boundaries,
            }

        contexts = [
            {
                "label": "A",
                "text": "A zámek.",
                "token_index": 2,
                "span_start_token_index": 1,
                "span_end_token_index": 2,
                "span_character_start": 2,
                "span_character_end": 7,
                "span_text": "zámek",
            },
            {
                "label": "B",
                "text": "B zámek.",
                "token_index": 2,
                "span_start_token_index": 1,
                "span_end_token_index": 2,
                "span_character_start": 2,
                "span_character_end": 7,
                "span_text": "zámek",
            },
        ]
        with patch.object(
            self.state,
            "_tokenization_text_payload",
            side_effect=fake_trace,
        ):
            result = self.state._embeddings_local(
                contexts,
                8,
                "word_span",
            )

        self.assertEqual(result["word_span"]["token_ids"], [12, 13])
        self.assertEqual(len(result["embedding_fragments"]), 2)
        self.assertFalse(
            result["embedding_fragments"][0]["is_closing_fragment"]
        )
        self.assertTrue(
            result["embedding_fragments"][1]["is_closing_fragment"]
        )
        self.assertEqual(
            result["contexts"][0]["prefix_token_ids"],
            [0, 21, 12, 13],
        )
        self.assertEqual(result["contexts"][0]["suffix_text"], ".")
        self.assertAlmostEqual(
            result["pairs"][0]["input_cosine_distance"],
            0.0,
        )
        self.assertEqual(
            [
                point["point_role"]
                for point in result["projection"]["points"][:2]
            ],
            ["closing_fragment", "word_fragment"],
        )

    def test_single_context_keeps_geometry_and_uses_one_forward(self) -> None:
        context = self.contexts()[0]
        result = self.state._embeddings_local([context], 8)

        self.assertEqual(len(result["contexts"]), 1)
        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["execution"]["forward_passes"], 1)
        self.assertEqual(len(result["neighbors"]), 8)
        self.assertEqual(len(result["projection"]["points"]), 9)
        self.assertEqual(len(result["contexts"][0]["layers"]), 33)

    def test_word_span_rejects_different_visible_forms(self) -> None:
        contexts = [
            {
                "label": "A",
                "text": "Na kopci stojí zámek.",
                "token_index": 3,
                "span_start_token_index": 3,
                "span_end_token_index": 3,
                "span_character_start": 15,
                "span_character_end": 20,
                "span_text": "zámek",
            },
            {
                "label": "B",
                "text": "Na dveřích je rozbitý zámek.",
                "token_index": 4,
                "span_start_token_index": 4,
                "span_end_token_index": 4,
                "span_character_start": 22,
                "span_character_end": 27,
                "span_text": "jinak",
            },
        ]
        with self.assertRaisesRegex(ValueError, "neodpovídá přesnému"):
            self.state._embeddings_local(contexts, 8, "word_span")

    def test_word_span_rejects_partial_and_multiword_ranges(self) -> None:
        base = {
            "label": "A",
            "text": "Na kopci stojí zámek.",
        }
        partial = {
            **base,
            "token_index": 3,
            "span_start_token_index": 3,
            "span_end_token_index": 3,
            "span_character_start": 15,
            "span_character_end": 18,
            "span_text": "zám",
        }
        phrase = {
            **base,
            "token_index": 3,
            "span_start_token_index": 2,
            "span_end_token_index": 3,
            "span_character_start": 9,
            "span_character_end": 20,
            "span_text": "stojí zámek",
        }

        for context in (partial, phrase):
            with self.subTest(span=context["span_text"]):
                with self.assertRaisesRegex(
                    ValueError,
                    "není celé maximální ortografické slovo",
                ):
                    self.state._embeddings_local(
                        [context],
                        8,
                        "word_span",
                    )

    def test_same_visible_word_with_different_tokenization_is_rejected(
        self,
    ) -> None:
        def fake_trace(text: str, *, label: str) -> dict:
            has_leading_space = text.startswith(" ")
            token_id = 13 if has_leading_space else 12
            token = {
                "index": 0,
                "token_id": token_id,
                "display": "zámek",
                "source_text": text,
                "character_start": 0,
                "character_end": len(text),
                "utf8_byte_start": 0,
                "utf8_byte_end": len(text.encode("utf-8")),
            }
            return {
                "label": label,
                "text": text,
                "token_count": 1,
                "tokens": [token],
                "boundaries": [
                    {
                        "prefix_text": "",
                        "prefix_token_ids": [0],
                    },
                    {
                        "prefix_text": text,
                        "prefix_token_ids": [0, token_id],
                    },
                ],
            }

        contexts = [
            {
                "label": "Samostatně",
                "text": "zámek",
                "token_index": 0,
                "span_start_token_index": 0,
                "span_end_token_index": 0,
                "span_character_start": 0,
                "span_character_end": 5,
                "span_text": "zámek",
            },
            {
                "label": "Po mezeře",
                "text": " zámek",
                "token_index": 0,
                "span_start_token_index": 0,
                "span_end_token_index": 0,
                "span_character_start": 1,
                "span_character_end": 6,
                "span_text": "zámek",
            },
        ]

        with patch.object(
            self.state,
            "_tokenization_text_payload",
            side_effect=fake_trace,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "shodnou posloupnost tokenových ID",
            ):
                self.state._embeddings_local(contexts, 8, "word_span")


class EmbeddingHttpAndPageContractTest(unittest.TestCase):
    def test_request_requires_identity_and_bounded_contexts(self) -> None:
        fields = GardenHandler._embeddings_request_fields(
            {
                "contexts": [
                    {
                        "label": "A",
                        "text": "Jeden text",
                        "token_index": 1,
                    },
                    {
                        "label": "B",
                        "text": "Druhý text",
                        "token_index": 1,
                    },
                ],
                "neighbor_count": 12,
                "model_id": "hybrid",
                "model_generation": 3,
                "server_instance_id": "server",
            }
        )
        self.assertEqual(fields[1:4], (12, "hybrid", 3))
        self.assertEqual(fields[5], "token")
        with self.assertRaises(ValueError):
            GardenHandler._embeddings_request_fields(
                {
                    "contexts": [],
                    "neighbor_count": 12,
                    "model_id": "hybrid",
                    "model_generation": 3,
                    "server_instance_id": "server",
                }
            )

    def test_page_api_and_worker_routes_are_registered(self) -> None:
        get_source = inspect.getsource(GardenHandler.do_GET)
        embedding_route = POST_ROUTES["/api/embeddings"]
        neighborhood_route = POST_ROUTES[
            "/api/embedding-neighborhood"
        ]
        embedding_source = inspect.getsource(
            getattr(GardenHandler, embedding_route.controller)
        )
        neighborhood_source = inspect.getsource(
            getattr(GardenHandler, neighborhood_route.controller)
        )
        worker_source = inspect.getsource(ModelState._worker_loop)

        self.assertIn('"/embeddingy"', get_source)
        self.assertIn("EMBEDDINGS_FILE", get_source)
        self.assertIn("self.state.embeddings(", embedding_source)
        self.assertIn('operation == "embeddings"', worker_source)
        self.assertIn(
            "self.state.embedding_neighborhood(",
            neighborhood_source,
        )
        self.assertIn(
            'operation == "embedding_neighborhood"',
            worker_source,
        )

    def test_neighborhood_request_requires_exact_identity_and_bounds(
        self,
    ) -> None:
        fields = GardenHandler._embedding_neighborhood_request_fields(
            {
                "token_id": 17,
                "neighbor_count": 6,
                "model_id": "hybrid",
                "model_generation": 3,
                "server_instance_id": "server",
            }
        )
        self.assertEqual(fields, (17, 6, "hybrid", 3, "server"))
        for invalid in (
            {**{
                "neighbor_count": 6,
                "model_id": "hybrid",
                "model_generation": 3,
                "server_instance_id": "server",
            }},
            {
                "token_id": True,
                "neighbor_count": 6,
                "model_id": "hybrid",
                "model_generation": 3,
                "server_instance_id": "server",
            },
            {
                "token_id": 17,
                "neighbor_count": 9,
                "model_id": "hybrid",
                "model_generation": 3,
                "server_instance_id": "server",
            },
        ):
            with self.subTest(request=invalid):
                with self.assertRaises(ValueError):
                    GardenHandler._embedding_neighborhood_request_fields(
                        invalid
                    )

    def test_page_exposes_row_neighbors_and_context_divergence(self) -> None:
        source = PAGE.read_text(encoding="utf-8")

        self.assertIn('data-workbench-page="embeddings"', source)
        self.assertIn('href="/static/workbench-shell.css"', source)
        self.assertIn('src="/static/workbench-shell.js"', source)
        self.assertIn('id="neighbor-chart"', source)
        self.assertIn('id="trajectory-chart"', source)
        self.assertIn('id="token-streams"', source)
        self.assertIn("/api/tokenization", source)
        self.assertIn("/api/embeddings", source)
        self.assertIn("/api/embedding-neighborhood", source)
        self.assertIn('id="neighbor-network-chart"', source)
        self.assertIn("Rozložení je schematické", source)
        self.assertIn("expandNeighborGraphToken", source)
        self.assertIn("MAX_EXPANDED_GRAPH_ROOTS", source)
        self.assertIn(
            "nikoli stejnou\n        kontextovou reprezentaci",
            source,
        )
        self.assertIn("stavy na pozici", source)
        self.assertIn("zvoleného výskytu", source)
        self.assertIn("Soused není synonymum", source)
        self.assertIn("model nic z šedého pokračování nepoužije", source)
        self.assertIn("+ Přidat druhý kontext", source)
        self.assertIn("defaultSingleWordSelection", source)
        self.assertIn("defaultSingleTokenSelection", source)
        self.assertIn("trajectorySeries", source)
        self.assertNotIn("Druhý kontext je povinný.", source)
        self.assertIn('value="word_span"', source)
        self.assertIn("Slovo není nový řádek", source)
        self.assertIn("posledního střepu", source)
        self.assertIn("Celé slovo · ortograficky", source)
        self.assertIn("Není to lemma ani morfologická analýza", source)
        self.assertIn(
            "Text neobsahuje žádné celé ortografické slovo.",
            source,
        )
        self.assertIn("režimu Tokenový řádek", source)
        self.assertIn("trace?.word_spans", source)
        self.assertNotIn("const tokenCharacterRange", source)


if __name__ == "__main__":
    unittest.main()
