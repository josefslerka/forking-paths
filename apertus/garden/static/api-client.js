(() => {
  "use strict";

  if (window.ProbabilityGardenApi) return;

  const REQUEST_EVENT = "probability-garden:api-request-state";
  const latestRequestByKey = new Map();
  let requestSequence = 0;
  let activeRequests = 0;
  let lastState = Object.freeze({
    phase: "idle",
    activeCount: 0,
    requestId: null,
    requestKey: null,
    url: null,
    method: null,
    code: null
  });

  class GardenApiError extends Error {
    constructor(message, details = {}) {
      super(message);
      this.name = "GardenApiError";
      Object.assign(this, details);
    }
  }

  class GardenStaleResponseError extends GardenApiError {
    constructor(message = "Dorazila zastaralá odpověď; výsledek nebyl použit.") {
      super(message, {code: "stale_response", stale: true});
      this.name = "GardenStaleResponseError";
    }
  }

  const emitState = (phase, detail = {}) => {
    lastState = Object.freeze({
      phase,
      activeCount: activeRequests,
      requestId: detail.requestId ?? null,
      requestKey: detail.requestKey ?? null,
      url: detail.url ?? null,
      method: detail.method ?? null,
      code: detail.code ?? null,
      silent: Boolean(detail.silent)
    });
    window.dispatchEvent(new CustomEvent(REQUEST_EVENT, {detail: lastState}));
  };

  const normalizeIdentity = (identity) => {
    if (!identity || typeof identity !== "object") return null;
    return {
      modelId: identity.modelId ?? identity.model_id ?? null,
      modelGeneration:
        identity.modelGeneration ?? identity.model_generation ?? null,
      serverInstanceId:
        identity.serverInstanceId ?? identity.server_instance_id ?? null
    };
  };

  const responseIdentity = (payload) => normalizeIdentity({
    model_id:
      payload?.model_id ??
      payload?.active_model?.id ??
      payload?.health?.active_model?.id,
    model_generation:
      payload?.model_generation ??
      payload?.health?.model_generation,
    server_instance_id:
      payload?.server_instance_id ??
      payload?.health?.server_instance_id
  });

  const identityChanged = (expected, actual) => {
    const left = normalizeIdentity(expected);
    const right = normalizeIdentity(actual);
    if (!left || !right) return false;
    return (
      (
        left.modelId !== null &&
        right.modelId !== null &&
        String(left.modelId) !== String(right.modelId)
      ) ||
      (
        left.modelGeneration !== null &&
        right.modelGeneration !== null &&
        Number(left.modelGeneration) !== Number(right.modelGeneration)
      ) ||
      (
        left.serverInstanceId !== null &&
        right.serverInstanceId !== null &&
        String(left.serverInstanceId) !== String(right.serverInstanceId)
      )
    );
  };

  const apiErrorFromResponse = (response, payload) => new GardenApiError(
    payload?.error || `Server odpověděl HTTP ${response.status}.`,
    {
      status: response.status,
      code: payload?.code ?? null,
      health: payload?.health ?? null,
      side: payload?.side ?? null,
      payload,
      stale: payload?.code === "model_changed",
      tunedStatus: payload?.tuned_lens ?? payload?.status_payload ?? null,
      readerStatus: payload?.jacobian_lens ?? payload?.status_payload ?? null
    }
  );

  const abortError = (timedOut, timeoutMs) => new GardenApiError(
    timedOut
      ? (
          timeoutMs
            ? `Požadavek překročil časový limit ${Math.ceil(timeoutMs / 1000)} s.`
            : "Požadavek překročil časový limit."
        )
      : "Požadavek byl zrušen.",
    {code: timedOut ? "request_timeout" : "request_cancelled"}
  );

  const fetchJson = async (url, options = {}, timeoutOverride = null) => {
    const {
      timeoutMs: optionTimeout = null,
      requestKey = null,
      expectedIdentity = null,
      silent = url === "/api/health",
      signal: externalSignal,
      ...fetchOptions
    } = options ?? {};
    const timeoutMs = timeoutOverride ?? optionTimeout;
    const requestId = ++requestSequence;
    const method = String(fetchOptions.method ?? "GET").toUpperCase();
    const controller = new AbortController();
    let timedOut = false;
    let timeoutId = null;
    const forwardAbort = () => controller.abort(externalSignal?.reason);

    if (externalSignal) {
      if (externalSignal.aborted) forwardAbort();
      else externalSignal.addEventListener("abort", forwardAbort, {once: true});
    }
    if (Number.isFinite(timeoutMs) && timeoutMs > 0) {
      timeoutId = window.setTimeout(() => {
        timedOut = true;
        controller.abort("timeout");
      }, timeoutMs);
    }
    if (requestKey) latestRequestByKey.set(requestKey, requestId);

    activeRequests += 1;
    emitState("loading", {
      requestId,
      requestKey,
      url,
      method,
      silent
    });

    let phase = "success";
    let outcomeCode = null;
    try {
      const response = await fetch(url, {
        cache: "no-store",
        ...fetchOptions,
        signal: controller.signal,
        headers: {
          Accept: "application/json",
          ...(fetchOptions.headers ?? {})
        }
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw apiErrorFromResponse(response, payload);
      if (
        requestKey &&
        latestRequestByKey.get(requestKey) !== requestId
      ) {
        throw new GardenStaleResponseError();
      }
      if (
        expectedIdentity &&
        identityChanged(expectedIdentity, responseIdentity(payload))
      ) {
        throw new GardenStaleResponseError(
          "Během výpočtu se změnil modelový běh; zastaralý výsledek nebyl použit."
        );
      }
      return payload;
    } catch (error) {
      let normalized = error;
      if (controller.signal.aborted && !(error instanceof GardenStaleResponseError)) {
        normalized = abortError(timedOut, timeoutMs);
      } else if (!(error instanceof Error)) {
        normalized = new GardenApiError(String(error), {
          code: "request_failed"
        });
      } else if (
        !(error instanceof GardenApiError) &&
        error.name === "TypeError"
      ) {
        normalized = new GardenApiError(
          "Server není dostupný nebo bylo spojení přerušeno.",
          {code: "network_error", cause: error}
        );
      }
      outcomeCode = normalized.code ?? null;
      phase = normalized.stale || normalized.code === "model_changed"
        ? "stale"
        : normalized.code === "request_cancelled"
        ? "cancelled"
        : "error";
      throw normalized;
    } finally {
      if (timeoutId !== null) window.clearTimeout(timeoutId);
      externalSignal?.removeEventListener?.("abort", forwardAbort);
      activeRequests = Math.max(0, activeRequests - 1);
      emitState(phase, {
        requestId,
        requestKey,
        url,
        method,
        code: outcomeCode,
        silent
      });
    }
  };

  const api = Object.freeze({
    fetchJson,
    GardenApiError,
    GardenStaleResponseError,
    isStaleError: (error) => (
      error instanceof GardenStaleResponseError ||
      error?.code === "stale_response"
    ),
    getState: () => lastState,
    normalizeIdentity,
    identityChanged,
    eventName: REQUEST_EVENT
  });

  window.ProbabilityGardenApi = api;
})();
