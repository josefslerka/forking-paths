from __future__ import annotations

import time
import unittest
from pathlib import Path

import numpy as np
import pytest

from apertus.garden.domains.tokenization import BYTE_LEVEL_UNICODE_TO_BYTE
from apertus.garden.server import GardenHandler, ModelState


@pytest.mark.parametrize("prefix,continuation", [("Ř", " A"), ("A", " Ťok.")])
def test_exact_byte_boundary_accepts_fragmented_unicode(tmp_path, prefix, continuation):
    class ByteTokenizer:
        bos_token_id = 0
        all_special_ids = [0]
        alphabet = {value: character for character, value in BYTE_LEVEL_UNICODE_TO_BYTE.items()}

        def encode(self, text, add_special_tokens=True):
            ids = [value + 1 for value in text.encode("utf-8")]
            return [0, *ids] if add_special_tokens else ids

        def decode(self, ids, **kwargs):
            return bytes(value - 1 for value in ids if value).decode("utf-8", errors="replace")

        def convert_ids_to_tokens(self, value):
            return self.alphabet[value - 1] if value else "<s>"

    class FixedBackend:
        def make_cache(self, model):
            return []

        def last_logits(self, model, ids, cache):
            cache.extend(ids)
            return np.zeros(257, dtype=np.float32), cache

    state = ModelState(tmp_path, model_paths={"bytes": tmp_path}, active_model_id="bytes")
    state.tokenizer = ByteTokenizer()
    state.backend = FixedBackend()
    state.tokenizer_identity_cache = {"algorithm": "byte_level_bpe"}
    result = state._continuation_surprisal_local(prefix, continuation, 3)
    assert result["execution"]["token_offset_source"] == "byte_level_bpe_exact_bytes"
    assert result["continuation_token_ids"] == state.tokenizer.encode(continuation, False)
    assert result["tokens"][0]["utf8_byte_start"] == len(prefix.encode("utf-8"))
    assert result["tokens"][-1]["utf8_byte_end"] == len((prefix + continuation).encode("utf-8"))
    assert result["summary"]["total_continuation_surprisal_bits"] == pytest.approx(len(continuation.encode("utf-8")) * np.log2(257))


class ContinuationSurprisalTest(unittest.TestCase):
    def test_same_prefix_has_identical_root_without_future_tokens(self) -> None:
        state = ModelState(model_path=Path("."), mock=True)
        state.start()
        deadline = time.monotonic() + 2.0
        while state.status != "ready" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(state.status, "ready")
        try:
            common = (
                5,
                "mock",
                state.model_generation,
                state.server_instance_id,
            )
            first = state.continuation_surprisal(
                "Společný prefix",
                " A B",
                *common,
            )
            second = state.continuation_surprisal(
                "Společný prefix",
                " A C",
                *common,
            )
            self.assertEqual(
                first["execution"]["mode"],
                "teacher_forced_incremental_kv",
            )
            self.assertIs(
                first["execution"]["future_continuation_tokens_in_forward"],
                False,
            )
            self.assertEqual(
                first["prompt_logits_sha256"],
                second["prompt_logits_sha256"],
            )
            self.assertEqual(
                first["tokens"][0]["token_id"],
                second["tokens"][0]["token_id"],
            )
            self.assertEqual(
                first["tokens"][0]["surprisal_bits"],
                second["tokens"][0]["surprisal_bits"],
            )
        finally:
            state.stop()
            if state.worker_thread is not None:
                state.worker_thread.join(timeout=2)

    def test_http_request_fields_are_lossless_and_complete(self) -> None:
        request = {
            "prefix": "Prefix",
            "continuation": " pokračování",
            "top_k": 5,
            "model_id": "hybrid",
            "model_generation": 3,
            "server_instance_id": "server",
        }
        self.assertEqual(
            GardenHandler._continuation_surprisal_request_fields(request),
            (
                "Prefix",
                " pokračování",
                5,
                "hybrid",
                3,
                "server",
            ),
        )
        for key in request:
            invalid = dict(request)
            invalid.pop(key)
            with self.subTest(missing=key):
                with self.assertRaises(ValueError):
                    GardenHandler._continuation_surprisal_request_fields(
                        invalid
                    )


if __name__ == "__main__":
    unittest.main()
