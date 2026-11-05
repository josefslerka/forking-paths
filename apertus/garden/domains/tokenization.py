"""Deterministic tokenization analysis independent of HTTP and model workers."""

from __future__ import annotations

import bisect
import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from ..api.contracts import (
        MAX_TOKENIZATION_CHARACTERS,
        MAX_TOKENIZATION_TOKENS,
        MAX_TOKENIZATION_UTF8_BYTES,
    )
except ImportError:  # Direct execution via `python3 server.py`.
    from api.contracts import (
        MAX_TOKENIZATION_CHARACTERS,
        MAX_TOKENIZATION_TOKENS,
        MAX_TOKENIZATION_UTF8_BYTES,
    )


def encode_with_bos(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=True))
    except TypeError:
        token_ids = list(tokenizer.encode(text))
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if (
            bos_token_id is not None
            and token_ids[:1] != [bos_token_id]
        ):
            token_ids.insert(0, bos_token_id)
        return token_ids


def safe_decode(
    tokenizer: Any,
    token_ids: list[int],
    *,
    skip_special: bool,
) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special,
        )
    except TypeError:
        return tokenizer.decode(token_ids)


def visible_token(text: str, fallback: str) -> str:
    if text == "":
        return f"⟨{fallback}⟩"
    return (
        text.replace(" ", "␠")
        .replace("\n", "↵")
        .replace("\t", "⇥")
        .replace("\r", "⏎")
    )


def _byte_level_unicode_to_byte() -> dict[str, int]:
    visible_bytes = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values = list(visible_bytes)
    unicode_values = list(visible_bytes)
    extra = 0
    for value in range(256):
        if value in byte_values:
            continue
        byte_values.append(value)
        unicode_values.append(256 + extra)
        extra += 1
    return {
        chr(unicode_value): byte_value
        for byte_value, unicode_value in zip(
            byte_values,
            unicode_values,
            strict=True,
        )
    }


BYTE_LEVEL_UNICODE_TO_BYTE = _byte_level_unicode_to_byte()


def byte_level_token_bytes(raw_token: str | None) -> bytes | None:
    if not isinstance(raw_token, str) or not raw_token:
        return None
    try:
        return bytes(
            BYTE_LEVEL_UNICODE_TO_BYTE[character]
            for character in raw_token
        )
    except KeyError:
        return None


def exact_byte_offsets(text: str, tokens: list[dict[str, Any]], algorithm: str | None = None) -> tuple[str, list[dict[str, Any]]] | None:
    """Recover ranges even when a token ends inside a UTF-8 character.

    Accept a representation only if concatenated bytes reproduce the exact input.
    Character ranges may overlap for fragments of the same Unicode character.
    """
    raw = [token.get("tokenizer_token", token.get("raw_token")) for token in tokens]
    if any(not isinstance(value, str) for value in raw):
        return None
    text_bytes = text.encode("utf-8")
    candidates = []
    byte_level = [byte_level_token_bytes(value) for value in raw]
    if algorithm in {None, "byte_level_bpe"} and all(value is not None for value in byte_level):
        candidates.append(("byte_level_bpe_exact_bytes", byte_level))
    sentencepiece = [bytes([int(value[3:5], 16)]) if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", value)
                     else value.replace("▁", " ").encode("utf-8") for value in raw]
    if algorithm in {None, "sentencepiece_bpe"}:
        candidates.append(("sentencepiece_exact_bytes", sentencepiece))
    boundaries = [0]
    for character in text:
        boundaries.append(boundaries[-1] + len(character.encode("utf-8")))
    for source, pieces in candidates:
        combined = b"".join(pieces)
        leading_strip = 0
        if combined != text_bytes:
            if source == "sentencepiece_exact_bytes" and combined == b" " + text_bytes:
                leading_strip = 1
            else:
                continue
        cursor = -leading_strip
        ranges = []
        for piece in pieces:
            start, end = max(0, cursor), max(0, cursor + len(piece))
            ranges.append({"character_start": bisect.bisect_right(boundaries, start) - 1,
                           "character_end": bisect.bisect_left(boundaries, end),
                           "utf8_byte_start": start, "utf8_byte_end": end,
                           "synthetic_leading_marker": start == end == 0 and leading_strip == 1})
            cursor += len(piece)
        return source, ranges
    return None


def build_tokenizer_identity(
    *,
    tokenizer: Any,
    mock: bool,
    mock_vocabulary_size: int,
    model_paths: dict[str, Path],
    active_model_id: str | None = None,
) -> dict[str, Any]:
    if mock:
        return {
            "kind": "deterministic_mock_tokenizer",
            "algorithm": "development_mock",
            "vocab_size": mock_vocabulary_size,
            "bos_token_id": 0,
            "eos_token_id": None,
            "shared_across_model_modes": True,
            "verification": "single_mock_runtime",
            "tokenizer_json_sha256": None,
            "model_files": [],
        }

    entries: list[dict[str, Any]] = []
    for model_id, model_path in model_paths.items():
        tokenizer_path = model_path / "tokenizer.json"
        if not tokenizer_path.is_file():
            continue
        try:
            contents = tokenizer_path.read_bytes()
        except OSError:
            continue
        entries.append(
            {
                "model_id": model_id,
                "directory": model_path.name,
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        )
    fingerprints = {
        str(entry["sha256"])
        for entry in entries
    }
    shared = bool(entries) and len(fingerprints) == 1
    active_entry = next((entry for entry in entries if entry["model_id"] == active_model_id), None)
    selected_id = active_model_id or (entries[0]["model_id"] if entries else None)
    algorithm = "unknown"
    if selected_id in model_paths:
        try:
            tokenizer_config = json.loads((model_paths[selected_id] / "tokenizer.json").read_text())
            decoder = json.dumps(tokenizer_config.get("decoder", {}), ensure_ascii=False)
            if tokenizer_config.get("model", {}).get("type") == "BPE":
                algorithm = "byte_level_bpe" if '"ByteLevel"' in decoder else "sentencepiece_bpe" if '"ByteFallback"' in decoder else "bpe"
            else:
                algorithm = str(tokenizer_config.get("model", {}).get("type", "unknown")).lower()
        except (OSError, ValueError):
            pass
    return {
        "kind": (
            f"{type(tokenizer).__module__}."
            f"{type(tokenizer).__qualname__}"
        ),
        "algorithm": algorithm,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "shared_across_model_modes": shared,
        "verification": (
            "exact_tokenizer_json_sha256"
            if shared
            else "tokenizer_files_differ_or_are_unavailable"
        ),
        "tokenizer_json_sha256": (
            active_entry["sha256"] if active_entry else next(iter(fingerprints))
            if len(fingerprints) == 1
            else None
        ),
        "model_files": entries,
    }


@dataclass(frozen=True)
class TokenizationService:
    """Analyze tokenizer units without owning model or server state."""

    tokenizer: Any
    mock: bool
    mock_words: tuple[str, ...]

    def text_tokenization(
        self,
        text: str,
    ) -> tuple[
        list[int],
        list[int],
        list[str],
        list[str | None],
        list[tuple[int, int] | None],
    ]:
        """Return ordinary targets and BOS-aligned contexts for fixed text."""
        if self.mock:
            pieces = re.findall(r"\s*\S+", text)
            token_ids = [
                int.from_bytes(
                    hashlib.sha256(piece.encode("utf-8")).digest()[:4],
                    "big",
                )
                % len(self.mock_words)
                for piece in pieces
            ]
            offsets: list[tuple[int, int] | None] = []
            cursor = 0
            for piece in pieces:
                end = cursor + len(piece)
                offsets.append((cursor, end))
                cursor = end
            return (
                token_ids,
                [0, *token_ids[:-1]],
                pieces,
                [None for _ in token_ids],
                offsets,
            )

        try:
            token_ids = list(
                self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            )
        except TypeError:
            token_ids = encode_with_bos(self.tokenizer, text)
            bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_token_id is not None and token_ids[:1] == [bos_token_id]:
                token_ids = token_ids[1:]
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None and token_ids[-1:] == [eos_token_id]:
                token_ids = token_ids[:-1]

        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if type(bos_token_id) is not int or bos_token_id < 0:
            raise RuntimeError(
                "Tokenizer nemá platný BOS token; první textový token "
                "nelze skórovat z definovaného kontextu."
            )

        raw_tokens: list[str | None] = []
        for token_id in token_ids:
            try:
                raw_token = self.tokenizer.convert_ids_to_tokens(token_id)
            except Exception:
                raw_token = None
            raw_tokens.append(
                str(raw_token) if raw_token is not None else None
            )

        pieces: list[str] | None = None
        offsets: list[tuple[int, int] | None] | None = None
        tokenizer_call = getattr(self.tokenizer, "__call__", None)
        if callable(tokenizer_call):
            try:
                encoded = tokenizer_call(
                    text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                offset_ids = list(encoded["input_ids"])
                offset_values = list(encoded["offset_mapping"])
                candidate_offsets = [
                    (int(start), int(end))
                    for start, end in offset_values
                ]
                candidate_pieces = [
                    text[start:end] for start, end in candidate_offsets
                ]
                contiguous = (
                    bool(candidate_offsets)
                    and candidate_offsets[0][0] == 0
                    and candidate_offsets[-1][1] == len(text)
                    and all(
                        left[1] == right[0]
                        for left, right in zip(
                            candidate_offsets,
                            candidate_offsets[1:],
                        )
                    )
                )
                if (
                    offset_ids == token_ids
                    and len(candidate_offsets) == len(token_ids)
                    and contiguous
                    and "".join(candidate_pieces) == text
                ):
                    pieces = candidate_pieces
                    offsets = list(candidate_offsets)
            except (
                KeyError,
                TypeError,
                ValueError,
                AttributeError,
            ):
                pass

        if pieces is None or offsets is None:
            decoded_pieces: list[str] = []
            running_ids: list[int] = []
            decoded_prefix = ""
            for token_id in token_ids:
                running_ids.append(token_id)
                decoded = safe_decode(
                    self.tokenizer,
                    running_ids,
                    skip_special=True,
                )
                if decoded.startswith(decoded_prefix):
                    piece = decoded[len(decoded_prefix) :]
                else:
                    piece = safe_decode(
                        self.tokenizer,
                        [token_id],
                        skip_special=False,
                    )
                decoded_pieces.append(piece)
                decoded_prefix = decoded
            if "".join(decoded_pieces) == text:
                pieces = decoded_pieces
                offsets = []
                cursor = 0
                for piece in pieces:
                    end = cursor + len(piece)
                    offsets.append((cursor, end))
                    cursor = end
            else:
                pieces = [
                    safe_decode(
                        self.tokenizer,
                        [token_id],
                        skip_special=False,
                    )
                    for token_id in token_ids
                ]
                offsets = [None for _ in token_ids]

        return (
            token_ids,
            [bos_token_id, *token_ids[:-1]],
            pieces,
            raw_tokens,
            offsets,
        )

    @staticmethod
    def unicode_text_payload(text: str) -> dict[str, Any]:
        normalized = {
            form: unicodedata.normalize(form, text)
            for form in ("NFC", "NFD", "NFKC", "NFKD")
        }
        return {
            "code_point_count": len(text),
            "utf8_byte_count": len(text.encode("utf-8")),
            "utf8_hex": " ".join(
                f"{value:02X}" for value in text.encode("utf-8")
            ),
            "normalization": {
                form: {
                    "is_normalized": value == text,
                    "changed": value != text,
                }
                for form, value in normalized.items()
            },
        }

    @staticmethod
    def orthographic_word_ranges(
        text: str,
    ) -> list[tuple[int, int]]:
        """Return maximal observable word-like spans in Unicode text."""

        def starts_word(character: str) -> bool:
            return unicodedata.category(character)[:1] in {"L", "N"}

        def continues_word(character: str) -> bool:
            return unicodedata.category(character)[:1] in {"L", "M", "N"}

        joiners = {"'", "’", "-", "‑"}
        ranges: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(text):
            if not starts_word(text[cursor]):
                cursor += 1
                continue
            start = cursor
            cursor += 1
            while cursor < len(text):
                if continues_word(text[cursor]):
                    cursor += 1
                    continue
                if (
                    text[cursor] in joiners
                    and cursor + 1 < len(text)
                    and starts_word(text[cursor + 1])
                ):
                    cursor += 1
                    continue
                break
            ranges.append((start, cursor))
        return ranges

    @classmethod
    def orthographic_word_spans(
        cls,
        text: str,
        tokens: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Map orthographic spans to the exact tokenizer units they touch."""
        spans: list[dict[str, Any]] = []
        for character_start, character_end in cls.orthographic_word_ranges(
            text
        ):
            byte_start = len(text[:character_start].encode("utf-8"))
            byte_end = len(text[:character_end].encode("utf-8"))
            token_indices: list[int] = []
            for token in tokens:
                token_byte_start = token.get("utf8_byte_start")
                token_byte_end = token.get("utf8_byte_end")
                if (
                    type(token_byte_start) is int
                    and type(token_byte_end) is int
                    and max(byte_start, token_byte_start)
                    < min(byte_end, token_byte_end)
                ):
                    token_indices.append(int(token["index"]))
            contiguous = all(
                index == 0
                or token_index == token_indices[index - 1] + 1
                for index, token_index in enumerate(token_indices)
            )
            if not token_indices or not contiguous:
                continue
            spans.append(
                {
                    "text": text[character_start:character_end],
                    "character_start": character_start,
                    "character_end": character_end,
                    "utf8_byte_start": byte_start,
                    "utf8_byte_end": byte_end,
                    "token_indices": token_indices,
                    "token_start_index": token_indices[0],
                    "token_end_index": token_indices[-1],
                    "token_ids": [
                        int(tokens[index]["token_id"])
                        for index in token_indices
                    ],
                    "token_count": len(token_indices),
                }
            )
        return spans

    def text_payload(
        self,
        text: str,
        *,
        label: str,
    ) -> dict[str, Any]:
        (
            token_ids,
            _context_token_ids,
            pieces,
            raw_tokens,
            offsets,
        ) = self.text_tokenization(text)
        if not token_ids:
            raise ValueError(
                f"{label} po tokenizaci neobsahuje žádný textový token."
            )
        if len(token_ids) > MAX_TOKENIZATION_TOKENS:
            raise ValueError(
                f"{label} má {len(token_ids)} tokenů; limit je "
                f"{MAX_TOKENIZATION_TOKENS}."
            )

        if self.mock:
            encoded_with_special = [0, *token_ids]
            decoded = "".join(pieces)
            special_ids = {0}
        else:
            encoded_with_special = encode_with_bos(self.tokenizer, text)
            decoded = safe_decode(
                self.tokenizer,
                token_ids,
                skip_special=True,
            )
            special_ids = set(
                getattr(self.tokenizer, "all_special_ids", [])
            )

        text_utf8 = text.encode("utf-8")
        raw_byte_pieces = [
            byte_level_token_bytes(raw_token)
            for raw_token in raw_tokens
        ]
        exact_byte_offsets: list[tuple[int, int]] | None = None
        if (
            raw_byte_pieces
            and all(piece is not None for piece in raw_byte_pieces)
            and b"".join(
                piece
                for piece in raw_byte_pieces
                if piece is not None
            ) == text_utf8
        ):
            exact_byte_offsets = []
            byte_cursor = 0
            for byte_piece in raw_byte_pieces:
                assert byte_piece is not None
                byte_end = byte_cursor + len(byte_piece)
                exact_byte_offsets.append((byte_cursor, byte_end))
                byte_cursor = byte_end

        byte_boundary_to_character = {0: 0}
        character_byte_ranges: list[tuple[int, int]] = []
        byte_cursor = 0
        for character_index, character in enumerate(text):
            byte_end = byte_cursor + len(character.encode("utf-8"))
            character_byte_ranges.append((byte_cursor, byte_end))
            byte_boundary_to_character[byte_end] = character_index + 1
            byte_cursor = byte_end

        tokens: list[dict[str, Any]] = []
        for index, (
            token_id_value,
            piece,
            raw_token,
            offset,
        ) in enumerate(
            zip(
                token_ids,
                pieces,
                raw_tokens,
                offsets,
                strict=True,
            )
        ):
            token_id = int(token_id_value)
            decoded_single = (
                piece
                if self.mock
                else safe_decode(
                    self.tokenizer,
                    [token_id],
                    skip_special=False,
                )
            )
            character_start: int | None = None
            character_end: int | None = None
            utf8_byte_start: int | None = None
            utf8_byte_end: int | None = None
            source_piece = piece
            if offset is not None:
                character_start, character_end = offset
                source_piece = text[character_start:character_end]
                utf8_byte_start = len(
                    text[:character_start].encode("utf-8")
                )
                utf8_byte_end = len(
                    text[:character_end].encode("utf-8")
                )
            character_range_exact = (
                character_start is not None
                and character_end is not None
            )
            byte_range_exact = character_range_exact
            covering_character_start: int | None = None
            covering_character_end: int | None = None
            if exact_byte_offsets is not None:
                utf8_byte_start, utf8_byte_end = exact_byte_offsets[index]
                byte_range_exact = True
                start_character = byte_boundary_to_character.get(
                    utf8_byte_start
                )
                end_character = byte_boundary_to_character.get(
                    utf8_byte_end
                )
                if (
                    start_character is not None
                    and end_character is not None
                ):
                    character_start = start_character
                    character_end = end_character
                    source_piece = text[character_start:character_end]
                    character_range_exact = True
                else:
                    character_start = None
                    character_end = None
                    character_range_exact = False
                    overlapping_characters = [
                        character_index
                        for character_index, (
                            character_byte_start,
                            character_byte_end,
                        ) in enumerate(character_byte_ranges)
                        if (
                            character_byte_end > utf8_byte_start
                            and character_byte_start < utf8_byte_end
                        )
                    ]
                    if overlapping_characters:
                        covering_character_start = min(
                            overlapping_characters
                        )
                        covering_character_end = (
                            max(overlapping_characters) + 1
                        )
                    source_piece = ""
            source_bytes = (
                text_utf8[utf8_byte_start:utf8_byte_end]
                if (
                    byte_range_exact
                    and utf8_byte_start is not None
                    and utf8_byte_end is not None
                )
                else source_piece.encode("utf-8")
            )
            splits_unicode_code_point = (
                byte_range_exact and not character_range_exact
            )
            display_piece = (
                f"⟦{'·'.join(f'{value:02X}' for value in source_bytes)}⟧"
                if splits_unicode_code_point
                else raw_token
                if (
                    offset is None
                    and raw_token
                    and ("\ufffd" in decoded_single or not decoded_single)
                )
                else source_piece
            )
            tokens.append(
                {
                    "index": index,
                    "token_id": token_id,
                    "display": visible_token(
                        display_piece,
                        f"token {token_id}",
                    ),
                    "source_text": (
                        source_piece if character_range_exact else None
                    ),
                    "source_text_cover": (
                        text[
                            covering_character_start:
                            covering_character_end
                        ]
                        if (
                            covering_character_start is not None
                            and covering_character_end is not None
                        )
                        else None
                    ),
                    "decoded_text": decoded_single,
                    "tokenizer_token": raw_token,
                    "is_special": token_id in special_ids,
                    "piece_is_lossless": character_range_exact,
                    "character_range_exact": character_range_exact,
                    "utf8_byte_range_exact": byte_range_exact,
                    "splits_unicode_code_point": (
                        splits_unicode_code_point
                    ),
                    "character_start": character_start,
                    "character_end": character_end,
                    "covering_character_start": (
                        covering_character_start
                    ),
                    "covering_character_end": covering_character_end,
                    "character_count": (
                        character_end - character_start
                        if character_start is not None
                        and character_end is not None
                        else None
                    ),
                    "utf8_byte_start": utf8_byte_start,
                    "utf8_byte_end": utf8_byte_end,
                    "utf8_byte_count": len(source_bytes),
                    "utf8_hex": " ".join(
                        f"{value:02X}" for value in source_bytes
                    ),
                    "code_points": [
                        f"U+{ord(character):04X}"
                        for character in (
                            source_piece
                            if character_range_exact
                            else (
                                text[
                                    covering_character_start:
                                    covering_character_end
                                ]
                                if (
                                    covering_character_start is not None
                                    and covering_character_end is not None
                                )
                                else ""
                            )
                        )
                    ],
                }
            )

        bos_token_id = (
            0
            if self.mock
            else getattr(self.tokenizer, "bos_token_id", None)
        )
        special_prefix: list[dict[str, Any]] = []
        ordinary_start = 0
        if (
            type(bos_token_id) is int
            and encoded_with_special[:1] == [bos_token_id]
        ):
            raw_bos: str | None = None
            decoded_bos = ""
            if not self.mock:
                try:
                    converted = self.tokenizer.convert_ids_to_tokens(
                        bos_token_id
                    )
                    raw_bos = (
                        str(converted) if converted is not None else None
                    )
                except Exception:
                    raw_bos = None
                decoded_bos = safe_decode(
                    self.tokenizer,
                    [bos_token_id],
                    skip_special=False,
                )
            special_prefix.append(
                {
                    "index": -1,
                    "token_id": bos_token_id,
                    "display": raw_bos or "<s>",
                    "source_text": "",
                    "decoded_text": decoded_bos,
                    "tokenizer_token": raw_bos or "<s>",
                    "is_special": True,
                    "role": "bos",
                    "piece_is_lossless": True,
                    "character_start": 0,
                    "character_end": 0,
                    "character_count": 0,
                    "utf8_byte_start": 0,
                    "utf8_byte_end": 0,
                    "utf8_byte_count": 0,
                    "utf8_hex": "",
                    "code_points": [],
                }
            )
            ordinary_start = 1

        boundaries: list[dict[str, Any]] = []
        for boundary_index in range(len(token_ids) + 1):
            if boundary_index == 0:
                prefix_text = ""
                character_offset: int | None = 0
                byte_offset: int | None = 0
            else:
                previous = tokens[boundary_index - 1]
                byte_offset = previous["utf8_byte_end"]
                character_offset = (
                    byte_boundary_to_character.get(byte_offset)
                    if type(byte_offset) is int
                    else None
                )
                if character_offset is not None:
                    prefix_text = text[:character_offset]
                else:
                    prefix_ids = token_ids[:boundary_index]
                    prefix_text = (
                        "".join(pieces[:boundary_index])
                        if self.mock
                        else safe_decode(
                            self.tokenizer,
                            prefix_ids,
                            skip_special=True,
                        )
                    )
            prefix_ids = list(
                encoded_with_special[
                    : ordinary_start + boundary_index
                ]
            )
            next_token = (
                dict(tokens[boundary_index])
                if boundary_index < len(tokens)
                else None
            )
            boundaries.append(
                {
                    "index": boundary_index,
                    "prefix_text": prefix_text,
                    "prefix_token_ids": prefix_ids,
                    "text_token_count": boundary_index,
                    "character_offset": character_offset,
                    "utf8_byte_offset": (
                        byte_offset
                        if type(byte_offset) is int
                        else (
                            len(prefix_text.encode("utf-8"))
                            if character_offset is not None
                            else None
                        )
                    ),
                    "next_token": next_token,
                    "at_end": boundary_index == len(tokens),
                    "prefix_roundtrip_exact": (
                        character_offset is not None
                        and text[:character_offset] == prefix_text
                    ),
                }
            )

        word_spans = self.orthographic_word_spans(text, tokens)
        exact_character_ranges = sum(
            token["character_range_exact"]
            for token in tokens
        )
        exact_byte_ranges = sum(
            token["utf8_byte_range_exact"]
            for token in tokens
        )
        unicode_payload = self.unicode_text_payload(text)
        return {
            "label": label,
            "text": text,
            "unicode": unicode_payload,
            "token_count": len(tokens),
            "token_ids": [int(token_id) for token_id in token_ids],
            "encoded_with_special_token_ids": [
                int(token_id) for token_id in encoded_with_special
            ],
            "special_prefix_tokens": special_prefix,
            "tokens": tokens,
            "word_spans": word_spans,
            "word_span_definition": {
                "kind": "maximal_unicode_orthographic_span_v1",
                "starts_with": "Unicode letter or number",
                "continues_with": "Unicode letter, mark, or number",
                "internal_joiners": ["'", "’", "-", "‑"],
                "linguistic_claim": (
                    "observable_text_span_not_lemma_or_morphological_word"
                ),
            },
            "boundaries": boundaries,
            "detokenized_text": decoded,
            "roundtrip_exact": decoded == text,
            "offset_coverage": {
                "exact_token_count": exact_character_ranges,
                "total_token_count": len(tokens),
                "ratio": exact_character_ranges / len(tokens),
                "exact_character_token_count": exact_character_ranges,
                "character_ratio": exact_character_ranges / len(tokens),
                "exact_utf8_byte_token_count": exact_byte_ranges,
                "utf8_byte_ratio": exact_byte_ranges / len(tokens),
            },
        }

    @staticmethod
    def comparison(
        a: dict[str, Any],
        b: dict[str, Any],
    ) -> dict[str, Any]:
        ids_a = list(a["token_ids"])
        ids_b = list(b["token_ids"])
        common_prefix = 0
        for left, right in zip(ids_a, ids_b):
            if left != right:
                break
            common_prefix += 1

        max_suffix = min(len(ids_a), len(ids_b)) - common_prefix
        common_suffix = 0
        while (
            common_suffix < max_suffix
            and ids_a[-1 - common_suffix] == ids_b[-1 - common_suffix]
        ):
            common_suffix += 1

        first_token_divergence: dict[str, Any] | None = None
        if ids_a != ids_b:
            first_token_divergence = {
                "index": common_prefix,
                "a": (
                    dict(a["tokens"][common_prefix])
                    if common_prefix < len(ids_a)
                    else None
                ),
                "b": (
                    dict(b["tokens"][common_prefix])
                    if common_prefix < len(ids_b)
                    else None
                ),
            }

        text_a = str(a["text"])
        text_b = str(b["text"])
        common_character_prefix = 0
        for left, right in zip(text_a, text_b):
            if left != right:
                break
            common_character_prefix += 1
        first_character_divergence = (
            None
            if text_a == text_b
            else {
                "index": common_character_prefix,
                "a": (
                    f"U+{ord(text_a[common_character_prefix]):04X}"
                    if common_character_prefix < len(text_a)
                    else None
                ),
                "b": (
                    f"U+{ord(text_b[common_character_prefix]):04X}"
                    if common_character_prefix < len(text_b)
                    else None
                ),
            }
        )

        return {
            "same_text": text_a == text_b,
            "same_utf8_bytes": text_a.encode("utf-8") == text_b.encode("utf-8"),
            "canonically_equivalent_nfc": (
                unicodedata.normalize("NFC", text_a)
                == unicodedata.normalize("NFC", text_b)
            ),
            "compatibility_equivalent_nfkc": (
                unicodedata.normalize("NFKC", text_a)
                == unicodedata.normalize("NFKC", text_b)
            ),
            "identical_token_ids": ids_a == ids_b,
            "token_count_a": len(ids_a),
            "token_count_b": len(ids_b),
            "token_count_delta_b_minus_a": len(ids_b) - len(ids_a),
            "common_token_prefix_count": common_prefix,
            "common_token_suffix_count": common_suffix,
            "first_token_divergence": first_token_divergence,
            "common_character_prefix_count": common_character_prefix,
            "first_character_divergence": first_character_divergence,
        }

    def analyze(
        self,
        text_a: str,
        text_b: str | None,
        *,
        tokenizer_payload: dict[str, Any],
        model_id: str,
        model_label: str,
        model_generation: int,
        server_instance_id: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        for label, text in (("Text A", text_a), ("Text B", text_b)):
            if text is None:
                continue
            if not isinstance(text, str) or text == "":
                raise ValueError(f"{label} musí být neprázdný řetězec.")
            if len(text) > MAX_TOKENIZATION_CHARACTERS:
                raise ValueError(
                    f"{label} může mít nejvýše "
                    f"{MAX_TOKENIZATION_CHARACTERS} znaků."
                )
            if len(text.encode("utf-8")) > MAX_TOKENIZATION_UTF8_BYTES:
                raise ValueError(
                    f"{label} může mít nejvýše "
                    f"{MAX_TOKENIZATION_UTF8_BYTES} UTF-8 bajtů."
                )

        a = self.text_payload(text_a, label="Text A")
        b = (
            self.text_payload(text_b, label="Text B")
            if text_b is not None
            else None
        )
        total_ms = (time.perf_counter() - started) * 1_000
        warnings: list[dict[str, str]] = []
        for side in (a, b):
            if side is None:
                continue
            if not side["roundtrip_exact"]:
                warnings.append(
                    {
                        "code": "roundtrip_not_exact",
                        "message": (
                            f"{side['label']}: detokenizace se přesně "
                            "nevrátila k původnímu textu."
                        ),
                    }
                )
            if side["offset_coverage"]["utf8_byte_ratio"] < 1.0:
                warnings.append(
                    {
                        "code": "offsets_incomplete",
                        "message": (
                            f"{side['label']}: některé tokeny nemají "
                            "spolehlivý bajtový rozsah."
                        ),
                    }
                )
            split_count = sum(
                bool(token["splits_unicode_code_point"])
                for token in side["tokens"]
            )
            if split_count:
                warnings.append(
                    {
                        "code": "tokens_split_utf8_code_point",
                        "message": (
                            f"{side['label']}: {split_count} tokenů dělí "
                            "UTF‑8 uvnitř Unicode kódového bodu; bajtové "
                            "rozsahy jsou přesné, samostatný znakový úsek "
                            "pro ně neexistuje."
                        ),
                    }
                )

        return {
            "analysis_kind": "tokenization",
            "a": a,
            "b": b,
            "comparison": (
                self.comparison(a, b)
                if b is not None
                else None
            ),
            "tokenizer": tokenizer_payload,
            "execution": {
                "mode": "deterministic_tokenization_only",
                "forward_passes": 0,
                "generation": False,
                "probabilities": False,
                "model_weights_used": False,
                "normalization_applied_to_text": False,
            },
            "warnings": warnings,
            "model_id": model_id,
            "model_label": model_label,
            "model_generation": model_generation,
            "server_instance_id": server_instance_id,
            "timing_ms": {
                "total": total_ms,
            },
            "latency_ms": total_ms,
        }


__all__ = [
    "TokenizationService",
    "build_tokenizer_identity",
    "byte_level_token_bytes",
    "encode_with_bos",
    "exact_byte_offsets",
    "safe_decode",
    "visible_token",
]
