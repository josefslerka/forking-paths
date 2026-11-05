from __future__ import annotations

from pathlib import Path


GARDEN_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = GARDEN_DIR / "static"
API_CLIENT = (STATIC_DIR / "api-client.js").read_text(encoding="utf-8")
SERVER = (GARDEN_DIR / "server.py").read_text(encoding="utf-8")


def test_every_workbench_page_loads_api_client_before_shell() -> None:
    for path in sorted(STATIC_DIR.glob("*.html")):
        html = path.read_text(encoding="utf-8")
        if "/static/workbench-shell.js" not in html:
            continue
        if path.name == "detector.html":
            # The deployed detector page is a hash-locked research artefact.
            # Its external shell lazily adds the shared client.
            assert "/static/api-client.js" not in html
            continue
        assert "/static/api-client.js" in html, path.name
        assert html.index("/static/api-client.js") < html.index(
            "/static/workbench-shell.js"
        ), path.name


def test_api_client_owns_the_only_page_fetch_boundary() -> None:
    assert "const fetchJson = async" in API_CLIENT
    assert "GardenApiError" in API_CLIENT
    assert "GardenStaleResponseError" in API_CLIENT
    assert '"probability-garden:api-request-state"' in API_CLIENT
    assert '"request_timeout"' in API_CLIENT
    assert 'code: "network_error"' in API_CLIENT
    assert 'payload?.code === "model_changed"' in API_CLIENT
    for path in sorted(STATIC_DIR.glob("*.html")):
        html = path.read_text(encoding="utf-8")
        if path.name == "detector.html":
            continue
        assert "await fetch(" not in html, path.name


def test_server_exposes_api_client_as_javascript() -> None:
    assert 'path == "/static/api-client.js"' in SERVER
    assert "API_CLIENT_JS_FILE" in SERVER
    assert '"text/javascript; charset=utf-8"' in SERVER
