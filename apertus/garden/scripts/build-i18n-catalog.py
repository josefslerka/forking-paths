#!/usr/bin/env python3
"""Build the English UI catalogue from the Czech static application sources.

The catalogue is deliberately generated offline through the locally installed
Ollama model.  Runtime translation never calls a remote service: the resulting
JavaScript file is checked into the application and served like any other
static asset.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import urllib.request
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
OUTPUT = STATIC / "i18n-catalog.generated.js"
CHECKPOINT = ROOT / "build" / "i18n-catalog-checkpoint.json"
MODEL = "qwen3.5:9b"
BATCH_SIZE = 40
MAX_BATCH_CHARACTERS = 1_800

PAGES = {
    "model-comparison.html": "model-comparison",
    "model-comparison.js": "model-comparison-runtime",
    "steering-comparison.js": "model-comparison-runtime",
    "archive.html": "archive",
    "attribution.html": "attribution",
    "contrast.html": "contrast",
    "detector.html": "detector",
    "embedding.html": "embeddings",
    "guide.html": "guide",
    "index.html": "garden",
    "jlens.html": "jlens",
    "lens.html": "raw",
    "patching.html": "patching",
    "steering.html": "steering",
    "surprise.html": "surprise",
    "tokenization.html": "tokenization",
    "tuned.html": "tuned",
    "workbench-shell.js": "shell",
    "api-client.js": "common",
}
PYTHON_PAGES = {
    "model_comparison.py": "comparison-api",
    "steering_comparison.py": "comparison-api",
    "domains/steering.py": "comparison-api",
    "detector_readers.py": "detector-readers-api",
    "backends.py": "backends",
    "server.py": "server",
}

TRANSLATABLE_ATTRIBUTES = (
    "aria-label",
    "aria-description",
    "placeholder",
    "title",
    "data-empty-label",
)

CZECH_SIGNAL = re.compile(
    r"[áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]"
    r"|\b(?:aby|agregace|ale|alternativy|archiv|bajt(?:y|ů)?|bez|bude|"
    r"celkem|celé|co|do|fingerprint|generace|hranice|hotovo|hodnoty|indexy|"
    r"instance|jak|jednotka|je|jsou|kandidát|kde|který|kvantily|latence|"
    r"metoda|model(?:u)?|na|nebo|není|normalizace|od|po|pokud|pole|populace|"
    r"prefix|pro|provenience|před|při|se|sekvence|serveru|slov(?:a|o)?|"
    r"smazat|souhrn|stav|stejn|text|token(?:y|u|ů)?|uloženo|verze|vrstva|"
    r"výsledek|z|zachyceno|ze|znak(?:y|ů)?)\b",
    re.IGNORECASE,
)

SKIP_EXACT = {
    "html",
    "attention",
    "MLP",
    "RAW",
    "TUNED",
    "J-LENS",
    "Apertus",
    "Hybrid",
    "JSON",
    "CSV",
}

SYSTEM_PROMPT = """You translate a Czech research interface for humanities
scholars and psycholinguists into concise, idiomatic academic English.

Terminology is fixed:
- Pravděpodobnostní zahrada = Probability Garden
- pole očekávání = field of expectations
- překvapení / surprisal = surprisal
- čtečka = readout
- mezistav = intermediate state
- reziduální proud = residual stream
- kontextový stav = contextual state
- střep (token fragment) = fragment
- kauzální přenos = causal transfer
- řízené pokračování = teacher-forced continuation
- směrování / steering = steering
- vrstva = layer
- slovník = vocabulary
- celý slovník / plnoslovníkový = full-vocabulary
- výchozí = baseline when it names an experimental condition, otherwise initial
- běh = run
- řez = slice or readout depending on context

Keep RAW, TUNED, J-LENS, BF16, MLP, attention, logits, token IDs, formulas,
Unicode symbols, keyboard shortcuts and placeholders unchanged. Do not add an
explanation. Keep labels short. Preserve sentence meaning and epistemic
qualifiers such as “does not prove”, “descriptive”, and “only”.
Placeholders such as __PG_SLOT_0__ are immutable: copy each one exactly once.
"""


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def useful(value: str) -> bool:
    value = normalize(value)
    if not value or value in SKIP_EXACT:
        return False
    if all(
        re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)+", part)
        for part in value.split()
    ):
        # CSS classes and element IDs are implementation details, not copy.
        # This also covers a whitespace-separated className assignment.
        return False
    if not re.search(r"\s", value) and (
        value.startswith(("#", ".", "/")) or "[" in value or "]" in value
    ):
        # Query selectors are implementation details as well. Human-facing
        # range notation is either whitespace-bearing or contains no letters.
        return False
    if not re.search(r"[A-Za-zÁ-ž]", value):
        return False
    return bool(CZECH_SIGNAL.search(value))


def extract_html(path: Path) -> set[str]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    values: set[str] = set()
    for node in soup.find_all(string=True):
        if node.parent.name in {"script", "style"}:
            continue
        value = normalize(str(node))
        if useful(value):
            values.add(value)
    for element in soup.find_all(True):
        for attribute in TRANSLATABLE_ATTRIBUTES:
            value = element.get(attribute)
            if isinstance(value, str):
                value = normalize(value)
                if useful(value):
                    values.add(value)
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and useful(str(meta.get("content", ""))):
        values.add(normalize(str(meta["content"])))
    return values


def decode_js_string(quote: str, body: str) -> str | None:
    if quote == "`" or "${" in body:
        return None
    try:
        return ast.literal_eval(f"{quote}{body}{quote}")
    except (SyntaxError, ValueError):
        return None


def extract_javascript(source: str) -> set[str]:
    values: set[str] = set()
    pattern = re.compile(r'(?P<q>["\'`])(?P<body>(?:\\.|(?!\1).)*)(?P=q)', re.S)
    for match in pattern.finditer(source):
        value = decode_js_string(match.group("q"), match.group("body"))
        if value is None or "<" in value or ">" in value:
            continue
        value = normalize(value)
        if useful(value) and len(value) <= 700:
            values.add(value)
    values.update(extract_template_texts(source))
    return values


def joined_python_string(node: ast.JoinedStr) -> str:
    """Render an f-string as stable placeholders for offline translation."""
    parts: list[str] = []
    slot = 0
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue):
            parts.append(f"__PG_SLOT_{slot}__")
            slot += 1
    return "".join(parts)


def extract_python(path: Path) -> set[str]:
    values: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        value: str | None = None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
        elif isinstance(node, ast.JoinedStr):
            value = joined_python_string(node)
        if value is None:
            continue
        value = normalize(value)
        if useful(value) and len(value) <= 700:
            values.add(value)
    return values


def _skip_js_expression(source: str, position: int) -> int:
    """Return the position after a balanced ${...} expression."""
    depth = 1
    quote: str | None = None
    line_comment = False
    block_comment = False
    while position < len(source) and depth:
        char = source[position]
        following = source[position + 1] if position + 1 < len(source) else ""
        if line_comment:
            if char in "\r\n":
                line_comment = False
            position += 1
            continue
        if block_comment:
            if char == "*" and following == "/":
                block_comment = False
                position += 2
            else:
                position += 1
            continue
        if quote:
            if char == "\\":
                position += 2
                continue
            if char == quote:
                quote = None
            position += 1
            continue
        if char == "/" and following == "/":
            line_comment = True
            position += 2
            continue
        if char == "/" and following == "*":
            block_comment = True
            position += 2
            continue
        if char in "'\"`":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        position += 1
    return position


def _template_literals(source: str) -> list[str]:
    templates: list[str] = []
    position = 0
    while position < len(source):
        if source[position] != "`" or (
            position and source[position - 1] == "\\"
        ):
            position += 1
            continue
        position += 1
        parts: list[str] = []
        slot = 0
        while position < len(source):
            char = source[position]
            if char == "\\" and position + 1 < len(source):
                escaped = source[position + 1]
                parts.append({"n": "\n", "r": "\r", "t": "\t"}.get(escaped, escaped))
                position += 2
                continue
            if char == "`":
                position += 1
                templates.append("".join(parts))
                break
            if char == "$" and position + 1 < len(source) and source[position + 1] == "{":
                parts.append(f"__PG_SLOT_{slot}__")
                slot += 1
                expression_start = position + 2
                position = _skip_js_expression(source, expression_start)
                expression = source[expression_start : position - 1]
                templates.extend(_template_literals(expression))
                # A template expression can contain user-facing branch text,
                # for example `${ok ? "Přesná shoda" : "Neshoda"}`.  The
                # outer JavaScript-string scan cannot see those literals
                # because they sit inside the already matched template, so
                # extract the expression as a self-contained source fragment.
                templates.extend(extract_javascript(expression))
                continue
            parts.append(char)
            position += 1
    return templates


def extract_template_texts(source: str) -> set[str]:
    values: set[str] = set()
    for template in _template_literals(source):
        if not CZECH_SIGNAL.search(template):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
            soup = BeautifulSoup(template, "html.parser")
        for node in soup.find_all(string=True):
            if node.parent.name in {"script", "style"}:
                continue
            value = normalize(str(node))
            if useful(value) and len(value) <= 700:
                values.add(value)
        for element in soup.find_all(True):
            for attribute in TRANSLATABLE_ATTRIBUTES:
                raw = element.get(attribute)
                if isinstance(raw, str):
                    value = normalize(raw)
                    if useful(value) and len(value) <= 700:
                        values.add(value)
    return values


def collect() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for filename, page in PAGES.items():
        path = STATIC / filename
        source = path.read_text(encoding="utf-8")
        values = extract_javascript(source)
        if path.suffix == ".html":
            values.update(extract_html(path))
        result[page] = sorted(values | set(result.get(page, [])), key=lambda value: (len(value), value))
    for filename, page in PYTHON_PAGES.items():
        values = extract_python(ROOT / filename)
        result[page] = sorted(values | set(result.get(page, [])), key=lambda value: (len(value), value))
    return result


def ollama_translate(page: str, items: list[tuple[int, str]]) -> dict[int, str]:
    schema = {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "en": {"type": "string"},
                    },
                    "required": ["id", "en"],
                },
            }
        },
        "required": ["translations"],
    }
    prompt = (
        SYSTEM_PROMPT
        + f"\nCurrent interface area: {page}.\n"
        + "Return one translation for every numbered item, preserving its id.\n"
        + json.dumps(
            [{"id": item_id, "cs": value} for item_id, value in items],
            ensure_ascii=False,
        )
    )
    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0, "num_predict": 8192},
            "keep_alive": "30m",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        envelope = json.load(response)
    parsed = json.loads(envelope["response"])
    return {
        int(row["id"]): normalize(str(row["en"]))
        for row in parsed["translations"]
    }


def validate_translation(source: str, translated: str) -> None:
    source_slots = re.findall(r"__PG_SLOT_\d+__", source)
    target_slots = re.findall(r"__PG_SLOT_\d+__", translated)
    if sorted(source_slots) != sorted(target_slots):
        raise ValueError(
            f"Placeholder mismatch: {source_slots!r} -> {target_slots!r}"
        )
    if not translated:
        raise ValueError("Empty translation")


def translation_batches(values: list[str]) -> list[list[str]]:
    """Bound both item count and prompt size for long methodological copy."""
    batches: list[list[str]] = []
    batch: list[str] = []
    characters = 0
    for value in values:
        if batch and (
            len(batch) >= BATCH_SIZE
            or characters + len(value) > MAX_BATCH_CHARACTERS
        ):
            batches.append(batch)
            batch = []
            characters = 0
        batch.append(value)
        characters += len(value)
    if batch:
        batches.append(batch)
    return batches


def save_checkpoint(data: dict[str, dict[str, str]]) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_javascript(data: dict[str, dict[str, str]]) -> None:
    payload = (
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    OUTPUT.write_text(
        "// Generated by scripts/build-i18n-catalog.py; reviewed overrides live in i18n.js.\n"
        f"window.ProbabilityGardenEnglishCatalogue = {payload};\n",
        encoding="utf-8",
    )


def main() -> int:
    sources = collect()
    translations: dict[str, dict[str, str]] = {}
    if CHECKPOINT.exists():
        translations = json.loads(CHECKPOINT.read_text(encoding="utf-8"))

    translations = {
        page: {
            source: translations.get(page, {})[source]
            for source in values
            if source in translations.get(page, {})
        }
        for page, values in sources.items()
    }
    known_by_source: dict[str, str] = {}
    for page_map in translations.values():
        for source, translated in page_map.items():
            known_by_source.setdefault(source, translated)
    reused = 0
    for page, values in sources.items():
        page_map = translations.setdefault(page, {})
        for source in values:
            if source not in page_map and source in known_by_source:
                page_map[source] = known_by_source[source]
                reused += 1
    if reused:
        save_checkpoint(translations)
        print(f"Reused {reused} translations shared by multiple views.", flush=True)

    total = sum(len(values) for values in sources.values())
    completed = sum(len(values) for values in translations.values())
    print(f"Collected {total} page-scoped strings; resuming after {completed}.")

    for page, values in sources.items():
        page_map = translations.setdefault(page, {})
        page_reused = 0
        for source in values:
            if source not in page_map and source in known_by_source:
                page_map[source] = known_by_source[source]
                page_reused += 1
        if page_reused:
            completed += page_reused
            save_checkpoint(translations)
        pending = [value for value in values if value not in page_map]
        for batch_values in translation_batches(pending):
            indexed = list(enumerate(batch_values))
            for attempt in range(3):
                try:
                    batch = ollama_translate(page, indexed)
                    if set(batch) != set(range(len(indexed))):
                        raise ValueError("Ollama returned an incomplete id set")
                    for item_id, source in indexed:
                        validate_translation(source, batch[item_id])
                    break
                except Exception as exc:  # noqa: BLE001 - command-line recovery
                    if attempt == 2:
                        raise
                    print(f"Retrying {page} batch after {exc}", file=sys.stderr)
            for item_id, source in indexed:
                page_map[source] = batch[item_id]
                known_by_source.setdefault(source, batch[item_id])
            save_checkpoint(translations)
            completed += len(indexed)
            print(f"{completed}/{total} · {page}", flush=True)

    write_javascript(translations)
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
