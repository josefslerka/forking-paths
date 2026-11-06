"""Declarative POST route table for the local workbench API."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


RequestLimit = Literal["default", "experiment", "import"]
BodyKind = Literal["object", "any_json"]


@dataclass(frozen=True, slots=True)
class PostRoute:
    controller: str
    request_limit: RequestLimit = "default"
    body_kind: BodyKind = "object"


_ROUTES = {
    "/api/experiments": PostRoute(
        "_post_experiments",
        request_limit="experiment",
    ),
    "/api/experiments/import": PostRoute(
        "_post_experiments_import",
        request_limit="import",
        body_kind="any_json",
    ),
    "/api/model": PostRoute("_post_model"),
    "/api/model-admin/inspect": PostRoute("_post_model_admin"),
    "/api/model-admin/edit": PostRoute("_post_model_admin"),
    "/api/model-admin/plan": PostRoute("_post_model_admin"),
    "/api/model-admin/jobs": PostRoute("_post_model_admin"),
    "/api/model-admin/cancel": PostRoute("_post_model_admin"),
    "/api/model-admin/reload": PostRoute("_post_model_admin"),
    "/api/model-admin/addons": PostRoute("_post_model_admin"),
    "/api/model-comparison": PostRoute("_post_model_comparison"),
    "/api/model-comparison/cancel": PostRoute("_post_model_comparison_cancel"),
    "/api/tokenization": PostRoute("_post_tokenization"),
    "/api/embeddings": PostRoute("_post_embeddings"),
    "/api/embedding-neighborhood": PostRoute(
        "_post_embedding_neighborhood"
    ),
    "/api/text-surprisal": PostRoute("_post_text_surprisal"),
    "/api/predictive-stylometry": PostRoute(
        "_post_predictive_stylometry"
    ),
    "/api/detector-readers": PostRoute("_post_detector_readers"),
    "/api/detector-readers/cancel": PostRoute("_post_detector_readers_cancel"),
    "/api/continuation-surprisal": PostRoute(
        "_post_continuation_surprisal"
    ),
    "/api/text-surprisal-comparison": PostRoute(
        "_post_text_surprisal_comparison"
    ),
    "/api/context-contrast": PostRoute("_post_context_contrast"),
    "/api/steering/aligned-path": PostRoute(
        "_post_steering_aligned_path"
    ),
    "/api/steering/phrase-score": PostRoute(
        "_post_steering_phrase_score"
    ),
    "/api/steering/sham-control": PostRoute(
        "_post_steering_sham_control"
    ),
    "/api/steering/layer-sweep": PostRoute(
        "_post_steering_layer_sweep"
    ),
    "/api/steering/trajectory": PostRoute(
        "_post_steering_trajectory"
    ),
    "/api/steering": PostRoute("_post_steering"),
    "/api/lens": PostRoute("_post_lens"),
    "/api/tuned-lens": PostRoute("_post_tuned_lens"),
    "/api/jacobian-lens": PostRoute("_post_jacobian_lens"),
    "/api/component-attribution": PostRoute(
        "_post_component_attribution"
    ),
    "/api/activation-patching": PostRoute(
        "_post_activation_patching"
    ),
    "/api/compare": PostRoute("_post_compare"),
    "/api/next": PostRoute("_post_garden_query"),
    "/api/search": PostRoute("_post_garden_query"),
    "/api/lookup": PostRoute("_post_garden_query"),
}

POST_ROUTES = MappingProxyType(_ROUTES)


__all__ = ["BodyKind", "POST_ROUTES", "PostRoute", "RequestLimit"]
