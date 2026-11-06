from __future__ import annotations

import inspect

from apertus.garden.api.post_routes import POST_ROUTES
from apertus.garden.server import GardenHandler


EXPECTED_POST_ROUTES = {
    "/api/next",
    "/api/search",
    "/api/lookup",
    "/api/compare",
    "/api/steering",
    "/api/steering/trajectory",
    "/api/steering/aligned-path",
    "/api/steering/phrase-score",
    "/api/steering/sham-control",
    "/api/steering/layer-sweep",
    "/api/tokenization",
    "/api/embeddings",
    "/api/embedding-neighborhood",
    "/api/text-surprisal",
    "/api/predictive-stylometry",
    "/api/detector-readers",
    "/api/detector-readers/cancel",
    "/api/continuation-surprisal",
    "/api/text-surprisal-comparison",
    "/api/context-contrast",
    "/api/lens",
    "/api/tuned-lens",
    "/api/jacobian-lens",
    "/api/component-attribution",
    "/api/activation-patching",
    "/api/model",
    "/api/model-admin/inspect",
    "/api/model-admin/edit",
    "/api/model-admin/plan",
    "/api/model-admin/jobs",
    "/api/model-admin/cancel",
    "/api/model-admin/reload",
    "/api/model-comparison",
    "/api/model-comparison/cancel",
    "/api/experiments",
    "/api/experiments/import",
}


def test_route_table_is_complete_and_controllers_exist() -> None:
    assert set(POST_ROUTES) == EXPECTED_POST_ROUTES
    for path, route in POST_ROUTES.items():
        controller = getattr(GardenHandler, route.controller, None)
        assert callable(controller), (path, route.controller)


def test_archive_routes_own_the_only_larger_body_budgets() -> None:
    assert POST_ROUTES["/api/experiments"].request_limit == "experiment"
    assert (
        POST_ROUTES["/api/experiments/import"].request_limit == "import"
    )
    assert (
        POST_ROUTES["/api/experiments/import"].body_kind == "any_json"
    )
    assert all(
        route.request_limit == "default"
        for path, route in POST_ROUTES.items()
        if path not in {"/api/experiments", "/api/experiments/import"}
    )


def test_do_post_is_transport_only() -> None:
    source = inspect.getsource(GardenHandler.do_POST)
    assert "POST_ROUTES.get(path)" in source
    assert "self._dispatch_post_request(path, request)" in source
    for business_call in (
        "next_tokens(",
        "tokenization(",
        "text_surprisal(",
        "context_contrast(",
        "steering(",
        "activation_patching(",
    ):
        assert business_call not in source
    assert "experiment_store" not in source
