from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GardenHandoffPageContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (ROOT / "static" / "index.html").read_text(
            encoding="utf-8"
        )

    def test_handoff_is_one_shot_and_validated(self) -> None:
        self.assertIn(
            'const GARDEN_HANDOFF_SCHEMA = "apertus.garden-handoff"',
            self.html,
        )
        self.assertIn("GARDEN_HANDOFF_MAX_AGE_MS", self.html)
        self.assertIn("window.localStorage.removeItem(key)", self.html)
        self.assertIn("validHandoffTokenIds", self.html)
        self.assertIn("cleanHandoffQuery()", self.html)

    def test_exact_token_ids_bypass_retokenizing_trim(self) -> None:
        self.assertIn(
            "const prompt = exactTokenIds ? rawPrompt : rawPrompt.trimEnd()",
            self.html,
        )
        self.assertIn(
            "selectedHandoffTokenId === null",
            self.html,
        )
        self.assertIn(
            "[selectedHandoffTokenId]",
            self.html,
        )
        self.assertIn(
            "candidate.handoff_selected = true",
            self.html,
        )
        self.assertIn(
            "token z trajektorie · otevřít větev",
            self.html,
        )
        self.assertIn(
            "String(column.handoff.model_id) !== String(activeModelId)",
            self.html,
        )

    def test_prompt_query_remains_safe_fallback(self) -> None:
        self.assertIn('const queryPrompt = params.get("prompt")', self.html)
        self.assertIn(
            "createColumn(initialGardenSeed.prompt, initialGardenSeed.handoff)",
            self.html,
        )


if __name__ == "__main__":
    unittest.main()
