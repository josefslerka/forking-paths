import {
  expect,
  openWorkbench,
  test
} from "./helpers.mjs";

test.describe("Sdílený API klient", () => {
  test("publikuje loading, odmítne starou odpověď a rozliší změnu modelu", async ({appPage: page}) => {
    await openWorkbench(page, "/tokenizace", {mock: true});

    const result = await page.evaluate(async () => {
      const originalFetch = window.fetch;
      const events = [];
      let call = 0;
      window.addEventListener(
        window.ProbabilityGardenApi.eventName,
        (event) => {
          if (event.detail?.requestKey?.startsWith("e2e-")) {
            events.push({...event.detail});
          }
        }
      );
      window.fetch = async (url) => {
        if (String(url).includes("model-changed")) {
          return new Response(JSON.stringify({
            code: "model_changed",
            error: "Modelový běh se změnil."
          }), {
            status: 409,
            headers: {"Content-Type": "application/json"}
          });
        }
        call += 1;
        const current = call;
        await new Promise((resolve) => {
          window.setTimeout(resolve, current === 1 ? 60 : 5);
        });
        return new Response(JSON.stringify({call: current}), {
          status: 200,
          headers: {"Content-Type": "application/json"}
        });
      };

      try {
        const first = window.ProbabilityGardenApi.fetchJson(
          "/api/e2e-delayed",
          {requestKey: "e2e-same"}
        ).then(
          (value) => ({value}),
          (error) => ({code: error.code})
        );
        const loadingState = document.body.dataset.apiRequestState;
        const second = window.ProbabilityGardenApi.fetchJson(
          "/api/e2e-delayed",
          {requestKey: "e2e-same"}
        ).then(
          (value) => ({value}),
          (error) => ({code: error.code})
        );
        const [older, newer] = await Promise.all([first, second]);
        const changed = await window.ProbabilityGardenApi.fetchJson(
          "/api/model-changed",
          {requestKey: "e2e-model"}
        ).then(
          () => ({code: null}),
          (error) => ({code: error.code, stale: error.stale})
        );
        return {
          loadingState,
          older,
          newer,
          changed,
          phases: events.map((event) => event.phase),
          lastOutcome: document.body.dataset.apiLastOutcome
        };
      } finally {
        window.fetch = originalFetch;
      }
    });

    expect(result.loadingState).toBe("loading");
    expect(result.older.code).toBe("stale_response");
    expect(result.newer.value.call).toBe(2);
    expect(result.changed).toEqual({code: "model_changed", stale: true});
    expect(result.phases).toContain("loading");
    expect(result.phases).toContain("success");
    expect(result.phases).toContain("stale");
    expect(result.lastOutcome).toBe("stale");
  });
});
