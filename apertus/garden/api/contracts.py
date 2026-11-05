"""Pure request validation contracts shared by the HTTP handler and tests."""

from __future__ import annotations

from typing import Any


MAX_TEXT_SURPRISAL_CHARACTERS = 8_192
MAX_TEXT_SURPRISAL_UTF8_BYTES = 32_768
MAX_TEXT_SURPRISAL_TOKENS = 256
MAX_TEXT_SURPRISAL_TOP_K = 12
MAX_TEXT_SURPRISAL_LABEL_CHARACTERS = 160
MAX_TOKENIZATION_CHARACTERS = 8_192
MAX_TOKENIZATION_UTF8_BYTES = 32_768
MAX_TOKENIZATION_TOKENS = 2_048
MAX_EMBEDDING_CONTEXTS = 4
MAX_EMBEDDING_CHARACTERS = 4_000
MAX_EMBEDDING_UTF8_BYTES = 16_000
MAX_EMBEDDING_TOKENS = 256
MIN_EMBEDDING_NEIGHBORS = 5
MAX_EMBEDDING_NEIGHBORS = 30
MIN_EMBEDDING_GRAPH_NEIGHBORS = 3
MAX_EMBEDDING_GRAPH_NEIGHBORS = 8
MAX_CONTEXT_CONTRAST_CHARACTERS = 4_000
MAX_CONTEXT_CONTRAST_UTF8_BYTES = 16_000
MAX_CONTEXT_CONTRAST_PREFIX_TOKENS = 128
MAX_CONTEXT_CONTRAST_CONTINUATION_TOKENS = 64
MAX_CONTEXT_CONTRAST_TOP_K = 12


def tokenization_request_fields(
    request: dict[str, Any],
) -> tuple[str, str | None]:
    text_a = request.get("text_a")
    text_b = request.get("text_b")
    if not isinstance(text_a, str) or text_a == "":
        raise ValueError("Pole text_a musí být neprázdný řetězec.")
    if text_b is not None and (
        not isinstance(text_b, str) or text_b == ""
    ):
        raise ValueError(
            "Pole text_b musí být neprázdný řetězec, nebo se vynechá."
        )
    for field_name, text in (("text_a", text_a), ("text_b", text_b)):
        if text is None:
            continue
        if len(text) > MAX_TOKENIZATION_CHARACTERS:
            raise ValueError(
                f"Pole {field_name} může mít nejvýše "
                f"{MAX_TOKENIZATION_CHARACTERS} znaků."
            )
        if len(text.encode("utf-8")) > MAX_TOKENIZATION_UTF8_BYTES:
            raise ValueError(
                f"Pole {field_name} může mít nejvýše "
                f"{MAX_TOKENIZATION_UTF8_BYTES} UTF-8 bajtů."
            )
    return text_a, text_b


def embeddings_request_fields(
    request: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, str, int, str, str]:
    contexts = request.get("contexts")
    neighbor_count = request.get("neighbor_count", 12)
    analysis_mode = request.get("analysis_mode", "token")
    expected_model_id = request.get("model_id")
    expected_generation = request.get("model_generation")
    expected_server_instance_id = request.get("server_instance_id")
    if (
        not isinstance(contexts, list)
        or not 1 <= len(contexts) <= MAX_EMBEDDING_CONTEXTS
    ):
        raise ValueError(
            "contexts musí být seznam 1 až "
            f"{MAX_EMBEDDING_CONTEXTS} kontextů."
        )
    validated_contexts: list[dict[str, Any]] = []
    for index, context in enumerate(contexts):
        if not isinstance(context, dict):
            raise ValueError(
                f"contexts[{index}] musí být JSON objekt."
            )
        text = context.get("text")
        label = context.get("label", f"Kontext {index + 1}")
        token_index = context.get("token_index")
        if not isinstance(text, str) or text == "":
            raise ValueError(
                f"contexts[{index}].text musí být neprázdný řetězec."
            )
        if len(text) > MAX_EMBEDDING_CHARACTERS:
            raise ValueError(
                f"contexts[{index}].text může mít nejvýše "
                f"{MAX_EMBEDDING_CHARACTERS} znaků."
            )
        if len(text.encode("utf-8")) > MAX_EMBEDDING_UTF8_BYTES:
            raise ValueError(
                f"contexts[{index}].text může mít nejvýše "
                f"{MAX_EMBEDDING_UTF8_BYTES} UTF-8 bajtů."
            )
        if not isinstance(label, str) or len(label) > 120:
            raise ValueError(
                f"contexts[{index}].label musí být řetězec nejvýše "
                "120 znaků."
            )
        if type(token_index) is not int or token_index < 0:
            raise ValueError(
                f"contexts[{index}].token_index musí být nezáporné "
                "celé číslo."
            )
        validated_context = {
            "text": text,
            "label": label,
            "token_index": token_index,
        }
        if analysis_mode == "word_span":
            for field_name in (
                "span_start_token_index",
                "span_end_token_index",
                "span_character_start",
                "span_character_end",
            ):
                value = context.get(field_name)
                if type(value) is not int or value < 0:
                    raise ValueError(
                        f"contexts[{index}].{field_name} musí být "
                        "nezáporné celé číslo."
                    )
                validated_context[field_name] = value
            span_text = context.get("span_text")
            if not isinstance(span_text, str) or not span_text:
                raise ValueError(
                    f"contexts[{index}].span_text musí být "
                    "neprázdný řetězec."
                )
            validated_context["span_text"] = span_text
        validated_contexts.append(validated_context)
    if analysis_mode not in {"token", "word_span"}:
        raise ValueError(
            "analysis_mode musí být token nebo word_span."
        )
    if type(neighbor_count) is not int or not (
        MIN_EMBEDDING_NEIGHBORS
        <= neighbor_count
        <= MAX_EMBEDDING_NEIGHBORS
    ):
        raise ValueError(
            "neighbor_count musí být celé číslo mezi "
            f"{MIN_EMBEDDING_NEIGHBORS} a "
            f"{MAX_EMBEDDING_NEIGHBORS}."
        )
    if (
        not isinstance(expected_model_id, str)
        or not expected_model_id
    ):
        raise ValueError(
            "model_id je pro embeddingovou analýzu povinný řetězec."
        )
    if type(expected_generation) is not int:
        raise ValueError(
            "model_generation je pro embeddingovou analýzu povinné "
            "celé číslo."
        )
    if (
        not isinstance(expected_server_instance_id, str)
        or not expected_server_instance_id
    ):
        raise ValueError(
            "server_instance_id je pro embeddingovou analýzu "
            "povinný řetězec."
        )
    return (
        validated_contexts,
        neighbor_count,
        expected_model_id,
        expected_generation,
        expected_server_instance_id,
        analysis_mode,
    )


def embedding_neighborhood_request_fields(
    request: dict[str, Any],
) -> tuple[int, int, str, int, str]:
    token_id = request.get("token_id")
    neighbor_count = request.get("neighbor_count", 6)
    expected_model_id = request.get("model_id")
    expected_generation = request.get("model_generation")
    expected_server_instance_id = request.get("server_instance_id")
    if type(token_id) is not int or token_id < 0:
        raise ValueError(
            "token_id musí být nezáporné celé číslo."
        )
    if type(neighbor_count) is not int or not (
        MIN_EMBEDDING_GRAPH_NEIGHBORS
        <= neighbor_count
        <= MAX_EMBEDDING_GRAPH_NEIGHBORS
    ):
        raise ValueError(
            "neighbor_count musí být celé číslo mezi "
            f"{MIN_EMBEDDING_GRAPH_NEIGHBORS} a "
            f"{MAX_EMBEDDING_GRAPH_NEIGHBORS}."
        )
    if (
        not isinstance(expected_model_id, str)
        or not expected_model_id
    ):
        raise ValueError(
            "model_id je pro embeddingovou síť povinný řetězec."
        )
    if type(expected_generation) is not int:
        raise ValueError(
            "model_generation je pro embeddingovou síť povinné "
            "celé číslo."
        )
    if (
        not isinstance(expected_server_instance_id, str)
        or not expected_server_instance_id
    ):
        raise ValueError(
            "server_instance_id je pro embeddingovou síť povinný "
            "řetězec."
        )
    return (
        token_id,
        neighbor_count,
        expected_model_id,
        expected_generation,
        expected_server_instance_id,
    )


def text_surprisal_request_fields(
    request: dict[str, Any],
) -> tuple[str, int, str, int, str]:
    text = request.get("text")
    top_k = request.get("top_k")
    expected_model_id = request.get("model_id")
    expected_generation = request.get("model_generation")
    expected_server_instance_id = request.get("server_instance_id")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Pole text musí být neprázdný řetězec."
        )
    if len(text) > MAX_TEXT_SURPRISAL_CHARACTERS:
        raise ValueError(
            f"Text může mít nejvýše "
            f"{MAX_TEXT_SURPRISAL_CHARACTERS} znaků."
        )
    if len(text.encode("utf-8")) > MAX_TEXT_SURPRISAL_UTF8_BYTES:
        raise ValueError(
            f"Text může mít nejvýše "
            f"{MAX_TEXT_SURPRISAL_UTF8_BYTES} UTF-8 bajtů."
        )
    if type(top_k) is not int or not 1 <= top_k <= MAX_TEXT_SURPRISAL_TOP_K:
        raise ValueError(
            f"top_k musí být celé číslo mezi 1 a "
            f"{MAX_TEXT_SURPRISAL_TOP_K}."
        )
    if (
        not isinstance(expected_model_id, str)
        or not expected_model_id
    ):
        raise ValueError(
            "model_id je pro analýzu překvapení povinný řetězec."
        )
    if type(expected_generation) is not int:
        raise ValueError(
            "model_generation je pro analýzu překvapení povinné "
            "celé číslo."
        )
    if (
        not isinstance(expected_server_instance_id, str)
        or not expected_server_instance_id
    ):
        raise ValueError(
            "server_instance_id je pro analýzu překvapení povinný "
            "řetězec."
        )
    return (
        text,
        top_k,
        expected_model_id,
        expected_generation,
        expected_server_instance_id,
    )


def continuation_surprisal_request_fields(
    request: dict[str, Any],
) -> tuple[str, str, int, str, int, str]:
    prefix = request.get("prefix")
    continuation = request.get("continuation")
    top_k = request.get("top_k")
    expected_model_id = request.get("model_id")
    expected_generation = request.get("model_generation")
    expected_server_instance_id = request.get("server_instance_id")
    for field_name, value in (
        ("prefix", prefix),
        ("continuation", continuation),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Pole {field_name} musí být neprázdný řetězec."
            )
    assert isinstance(prefix, str)
    assert isinstance(continuation, str)
    full_text = prefix + continuation
    if len(full_text) > MAX_TEXT_SURPRISAL_CHARACTERS:
        raise ValueError(
            "Prefix a pokračování jsou dohromady příliš dlouhé."
        )
    if len(full_text.encode("utf-8")) > MAX_TEXT_SURPRISAL_UTF8_BYTES:
        raise ValueError(
            "Prefix a pokračování jsou dohromady příliš dlouhé v UTF-8."
        )
    if type(top_k) is not int or not 1 <= top_k <= MAX_TEXT_SURPRISAL_TOP_K:
        raise ValueError(
            f"top_k musí být celé číslo mezi 1 a "
            f"{MAX_TEXT_SURPRISAL_TOP_K}."
        )
    if not isinstance(expected_model_id, str) or not expected_model_id:
        raise ValueError("model_id je pro pokračování povinný řetězec.")
    if type(expected_generation) is not int:
        raise ValueError(
            "model_generation je pro pokračování povinné celé číslo."
        )
    if (
        not isinstance(expected_server_instance_id, str)
        or not expected_server_instance_id
    ):
        raise ValueError(
            "server_instance_id je pro pokračování povinný řetězec."
        )
    return (
        prefix,
        continuation,
        top_k,
        expected_model_id,
        expected_generation,
        expected_server_instance_id,
    )


def text_surprisal_comparison_request_fields(
    request: dict[str, Any],
) -> tuple[str, str, str, str, int, str, int, str]:
    text_a = request.get("text_a")
    text_b = request.get("text_b")
    label_a_value = request.get("label_a", "Text A")
    label_b_value = request.get("label_b", "Text B")
    top_k = request.get("top_k")
    expected_model_id = request.get("model_id")
    expected_generation = request.get("model_generation")
    expected_server_instance_id = request.get("server_instance_id")

    for field_name, text in (("text_a", text_a), ("text_b", text_b)):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Pole {field_name} musí být neprázdný řetězec."
            )
        if len(text) > MAX_TEXT_SURPRISAL_CHARACTERS:
            raise ValueError(
                f"Pole {field_name} může mít nejvýše "
                f"{MAX_TEXT_SURPRISAL_CHARACTERS} znaků."
            )
        if len(text.encode("utf-8")) > MAX_TEXT_SURPRISAL_UTF8_BYTES:
            raise ValueError(
                f"Pole {field_name} může mít nejvýše "
                f"{MAX_TEXT_SURPRISAL_UTF8_BYTES} UTF-8 bajtů."
            )

    labels: list[str] = []
    for field_name, label in (
        ("label_a", label_a_value),
        ("label_b", label_b_value),
    ):
        if not isinstance(label, str):
            raise ValueError(f"Pole {field_name} musí být řetězec.")
        if len(label) > MAX_TEXT_SURPRISAL_LABEL_CHARACTERS:
            raise ValueError(
                f"Pole {field_name} může mít nejvýše "
                f"{MAX_TEXT_SURPRISAL_LABEL_CHARACTERS} znaků."
            )
        default_label = (
            "Text A" if field_name == "label_a" else "Text B"
        )
        labels.append(label.strip() or default_label)

    if type(top_k) is not int or not 1 <= top_k <= MAX_TEXT_SURPRISAL_TOP_K:
        raise ValueError(
            f"top_k musí být celé číslo mezi 1 a "
            f"{MAX_TEXT_SURPRISAL_TOP_K}."
        )
    if (
        not isinstance(expected_model_id, str)
        or not expected_model_id
    ):
        raise ValueError(
            "model_id je pro párovou analýzu překvapení povinný "
            "řetězec."
        )
    if type(expected_generation) is not int:
        raise ValueError(
            "model_generation je pro párovou analýzu překvapení "
            "povinné celé číslo."
        )
    if (
        not isinstance(expected_server_instance_id, str)
        or not expected_server_instance_id
    ):
        raise ValueError(
            "server_instance_id je pro párovou analýzu překvapení "
            "povinný řetězec."
        )
    return (
        text_a,
        text_b,
        labels[0],
        labels[1],
        top_k,
        expected_model_id,
        expected_generation,
        expected_server_instance_id,
    )


def context_contrast_request_fields(
    request: dict[str, Any],
) -> tuple[str, str, str, int, str, int, str]:
    prefix_a = request.get("prefix_a")
    prefix_b = request.get("prefix_b")
    continuation = request.get("continuation")
    top_k = request.get("top_k", 5)
    expected_model_id = request.get("model_id")
    expected_generation = request.get("model_generation")
    expected_server_instance_id = request.get("server_instance_id")
    for field_name, text in (
        ("prefix_a", prefix_a),
        ("prefix_b", prefix_b),
        ("continuation", continuation),
    ):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Pole {field_name} musí být neprázdný řetězec."
            )
        if len(text) > MAX_CONTEXT_CONTRAST_CHARACTERS:
            raise ValueError(
                f"Pole {field_name} může mít nejvýše "
                f"{MAX_CONTEXT_CONTRAST_CHARACTERS} znaků."
            )
        if len(text.encode("utf-8")) > MAX_CONTEXT_CONTRAST_UTF8_BYTES:
            raise ValueError(
                f"Pole {field_name} může mít nejvýše "
                f"{MAX_CONTEXT_CONTRAST_UTF8_BYTES} UTF-8 bajtů."
            )
    if type(top_k) is not int or not (
        3 <= top_k <= MAX_CONTEXT_CONTRAST_TOP_K
    ):
        raise ValueError(
            "top_k musí být celé číslo mezi 3 a "
            f"{MAX_CONTEXT_CONTRAST_TOP_K}."
        )
    if (
        not isinstance(expected_model_id, str)
        or not expected_model_id
    ):
        raise ValueError(
            "model_id je pro kontrast kontextů povinný řetězec."
        )
    if type(expected_generation) is not int:
        raise ValueError(
            "model_generation je pro kontrast kontextů povinné "
            "celé číslo."
        )
    if (
        not isinstance(expected_server_instance_id, str)
        or not expected_server_instance_id
    ):
        raise ValueError(
            "server_instance_id je pro kontrast kontextů povinný "
            "řetězec."
        )
    return (
        prefix_a,
        prefix_b,
        continuation,
        top_k,
        expected_model_id,
        expected_generation,
        expected_server_instance_id,
    )


__all__ = [
    "MAX_CONTEXT_CONTRAST_CHARACTERS",
    "MAX_CONTEXT_CONTRAST_CONTINUATION_TOKENS",
    "MAX_CONTEXT_CONTRAST_PREFIX_TOKENS",
    "MAX_CONTEXT_CONTRAST_TOP_K",
    "MAX_CONTEXT_CONTRAST_UTF8_BYTES",
    "MAX_EMBEDDING_CHARACTERS",
    "MAX_EMBEDDING_CONTEXTS",
    "MAX_EMBEDDING_GRAPH_NEIGHBORS",
    "MAX_EMBEDDING_NEIGHBORS",
    "MAX_EMBEDDING_TOKENS",
    "MAX_EMBEDDING_UTF8_BYTES",
    "MAX_TEXT_SURPRISAL_CHARACTERS",
    "MAX_TEXT_SURPRISAL_LABEL_CHARACTERS",
    "MAX_TEXT_SURPRISAL_TOKENS",
    "MAX_TEXT_SURPRISAL_TOP_K",
    "MAX_TEXT_SURPRISAL_UTF8_BYTES",
    "MAX_TOKENIZATION_CHARACTERS",
    "MAX_TOKENIZATION_TOKENS",
    "MAX_TOKENIZATION_UTF8_BYTES",
    "MIN_EMBEDDING_GRAPH_NEIGHBORS",
    "MIN_EMBEDDING_NEIGHBORS",
    "context_contrast_request_fields",
    "continuation_surprisal_request_fields",
    "embedding_neighborhood_request_fields",
    "embeddings_request_fields",
    "text_surprisal_comparison_request_fields",
    "text_surprisal_request_fields",
    "tokenization_request_fields",
]
