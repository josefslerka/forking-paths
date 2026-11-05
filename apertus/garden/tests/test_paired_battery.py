import importlib.util
import json
import unittest
import os
from pathlib import Path


GARDEN_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_paired_battery",
    GARDEN_DIR / "run_paired_battery.py",
)
assert SPEC and SPEC.loader
BATTERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATTERY)


class PairedBatteryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (GARDEN_DIR / "paired_battery_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        path = Path(os.environ.get("APERTUS_TEST_TOKENIZER", BATTERY.DEFAULT_TOKENIZER))
        cls.tokenizer = BATTERY.load_tokenizer(path) if path.is_file() else None

    def test_locked_protocol_and_contextual_paths_validate(self):
        BATTERY.validate_protocol(self.protocol)
        roles = [pair["role"] for pair in self.protocol["pairs"]]
        self.assertEqual(roles.count("core"), 6)
        self.assertEqual(roles.count("female_replication"), 1)
        if self.tokenizer is None:
            self.skipTest("Exact Apertus token paths need APERTUS_TEST_TOKENIZER; no model tokenizer is bundled.")
        for pair in self.protocol["pairs"]:
            tokenization = BATTERY.tokenize_pair(self.tokenizer, pair)
            for frame in BATTERY.FRAME_ORDER:
                paths = tokenization["frames"][frame]["paths"]
                self.assertNotEqual(
                    paths["direct"]["first_token_id"],
                    paths["descriptor"]["first_token_id"],
                )

    def test_audited_wording_is_locked(self):
        pairs = {pair["pair_id"]: pair for pair in self.protocol["pairs"]}
        self.assertEqual(
            pairs["rusalka"]["frames"]["active"],
            "Hudbu k opeře Rusalka složil",
        )
        self.assertEqual(pairs["vykrik"]["expected_person"], "Edvard Munch")
        self.assertEqual(
            pairs["babicka"]["frames"]["active"],
            "Knihu Babička napsala",
        )

    def test_pair_filter_preserves_protocol_order(self):
        selected = BATTERY.choose_pairs(
            self.protocol, ["babicka", "orwell_1984"], None
        )
        self.assertEqual(
            [pair["pair_id"] for pair in selected],
            ["orwell_1984", "babicka"],
        )

    def test_archive_endpoints_are_not_callable(self):
        client = BATTERY.LocalApi(
            "http://127.0.0.1:8765",
            1.0,
            {
                "model_generation": 1,
                "server_instance_id": "test",
            },
        )
        with self.assertRaises(BATTERY.BatteryError):
            client.post("/api/experiments", {"record": {}})

    def test_summary_keeps_closure_and_female_replication_separate(self):
        def contrast(pair_id, role, entropy, root_shift, lexical_shift):
            return {
                "pair_id": pair_id,
                "pair_role": role,
                "delta_entropy_nominal_minus_active_bits": entropy,
                "effective_support_ratio_nominal_over_active": 2.0**entropy,
                "js_divergence_bits": 0.2,
                "delta_direct_first_token_probability_nominal_minus_active": -0.1,
                "delta_descriptor_first_token_probability_nominal_minus_active": 0.1,
                "root_descriptor_relative_log2_shift_nominal_vs_active": root_shift,
                "delta_direct_lexical_surprisal_nominal_minus_active_bits": 0.5,
                "delta_descriptor_lexical_surprisal_nominal_minus_active_bits": -0.5,
                "lexical_descriptor_relative_log2_shift_nominal_vs_active": lexical_shift,
                "full_path_descriptor_relative_log2_shift_nominal_vs_active": -3.0,
                "delta_direct_terminal_period_surprisal_nominal_minus_active_bits": -2.0,
                "delta_descriptor_terminal_period_surprisal_nominal_minus_active_bits": -1.0,
            }

        summary = BATTERY.build_summary(
            "test",
            [
                {"pair_id": "core", "role": "core"},
                {"pair_id": "female", "role": "female_replication"},
            ],
            [
                contrast("core", "core", -0.1, 1.0, 0.5),
                contrast("female", "female_replication", 6.0, 8.0, 4.0),
            ],
            [],
        )
        core = summary["groups"]["core"]
        female = summary["groups"]["female_replication"]
        self.assertEqual(core["n"], 1)
        self.assertEqual(female["n"], 1)
        self.assertAlmostEqual(
            core["root"]["delta_entropy_nominal_minus_active_bits"]["mean"],
            -0.1,
        )
        self.assertEqual(
            core["sentence_closure"]["terminal_period_easier_under_nominal"],
            2,
        )
        self.assertEqual(
            core["lexical_paths_without_terminal_period"][
                "nominal_relatively_favors_descriptor_path"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
