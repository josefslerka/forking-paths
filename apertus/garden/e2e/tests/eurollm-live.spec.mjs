import {expect, expectFiniteText, openWorkbench, test} from "./helpers.mjs";

test.describe.configure({mode: "serial"});

test.describe("Živý EuroLLM a společné měření", () => {
  let original;
  async function switchModel(request, id) {
    await expect.poll(async () => {
      const h = await (await request.get("/api/health")).json();
      return `${h.status}:${h.worker.busy}`;
    }, {timeout: 180_000}).toBe("ready:false");
    const response = await request.post("/api/model", {data: {model_id: id}});
    expect(response.ok()).toBeTruthy();
    await expect.poll(async () => {
      const h = await (await request.get("/api/health")).json();
      return `${h.status}:${h.active_model?.id}:${h.worker.busy}`;
    }, {timeout: 180_000}).toBe(`ready:${id}:false`);
  }
  test.beforeAll(async ({request}) => {
    const h = await (await request.get("/api/health")).json();
    expect(h.mock).toBe(false);
    original = h.active_model.id;
    expect(h.available_models.find(m => m.id === "eurollm9")?.available).toBe(true);
    await switchModel(request, "eurollm9");
  });
  test.afterAll(async ({request}) => {
    if (original) await switchModel(request, original);
  });

  test("EuroLLM tokenizuje skutečný český text", async ({appPage: page}) => {
    await openWorkbench(page, "/tokenizace");
    await page.locator("#text-a").fill("Mám rád Prahu. Ťok přišel pozdě.");
    await page.locator("#text-b").fill("");
    await page.locator("#analyze").click();
    await expect(page.locator("#pipeline")).toContainText(/Přesná shoda/i);
    const h = await (await page.request.get("/api/health")).json();
    expect(h.active_model.family).toBe("llama");
    expect(h.active_model.capabilities).not.toContain("tuned");
  });

  test("EuroLLM otevře next-token distribuci a navazující větev", async ({appPage: page}) => {
    await openWorkbench(page, "/");
    const column = page.locator(".prompt-column").first();
    const root = column.locator(".level").first();
    await expect(root).toBeVisible();
    await expect(root).toContainText("EuroLLM");
    await root.locator("button[data-branch-column]").first().click();
    await expect(column.locator(".level")).toHaveCount(2);
    expect((await root.locator(".branch .probability").allTextContents()).join(" ")).not.toMatch(/NaN|Infinity/);
  });

  test("EuroLLM změří surprisal i kontrast kontextů", async ({appPage: page}) => {
    await openWorkbench(page, "/prekvapeni");
    await page.locator("#source-text").fill("Praha leží na řece Vltavě.");
    await page.locator("#analyze").click();
    await expect(page.locator("#results")).toBeVisible({timeout: 180_000});
    await expectFiniteText(page.locator("#summary-surprisal"));
    await expectFiniteText(page.locator("#summary-perplexity"));
    await openWorkbench(page, "/kontrast");
    await page.locator("#analyze").click();
    await expect(page.locator("#results")).toBeVisible({timeout: 180_000});
    await expectFiniteText(page.locator("#summary-advantage"));
    await expectFiniteText(page.locator("#summary-js"));
  });

  test("EuroLLM vykreslí všech 42 RAW vrstev", async ({appPage: page}) => {
    await openWorkbench(page, "/vrstvy");
    await page.locator("#prompt").fill("Hlavním městem České republiky je");
    await page.locator("#analyze-button").click();
    await expect(page.locator("#trajectory-body tr")).toHaveCount(42, {timeout: 180_000});
    await expect(page.locator("#trajectory-body tr").last()).toContainText("L42");
    await expect(page.locator("#layer-count")).toContainText("42");
    await expect(page.locator("[data-model-capability-notice]")).toHaveCount(0);
  });

  test("EuroLLM dovede slovní trajektorii až do L42", async ({appPage: page}) => {
    await openWorkbench(page, "/embeddingy");
    await page.locator("#tokenize").click();
    await page.locator("#measure").click();
    await expect(page.locator("#results")).toBeVisible({timeout: 180_000});
    await expect(page.locator("#metric-final-label")).toContainText("L42");
    await expect(page.locator("#context-cards")).toContainText("L42");
    await expectFiniteText(page.locator("#metric-final"));
  });

  test("živé porovnání uloží oba modely a úplné Unicode úseky", async ({appPage: page}) => {
    await openWorkbench(page, "/porovnani-modelu");
    await page.locator("#comparison-text").fill("Na stole je 300 knih. Ťok přišel pozdě.");
    await page.locator("#run-comparison").click();
    await expect(page.locator("#comparison-results svg")).toBeVisible({timeout: 180_000});
    await expect(page.locator("#comparison-status")).toContainText("dokončeno");
    await expect(page.locator("#comparison-results .legend")).toContainText("EuroLLM 9B");
    await expect(page.locator("#comparison-results table")).not.toContainText("neúplné");
    await page.locator("#comparison-results").scrollIntoViewIfNeeded();
    await page.screenshot({path: "test-results/model-comparison-live.png"});
    await page.locator("#comparison-results").getByRole("link", {name: "Otevřít v archivu"}).click();
    await expect(page.locator("#detail")).toContainText("Porovnání modelů");
    const h = await (await page.request.get("/api/health")).json();
    expect(h.active_model.id).toBe("eurollm9");
  });
});
