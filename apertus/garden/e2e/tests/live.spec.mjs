import {
  expect,
  expectFiniteText,
  expectNoBusyOverlay,
  expectWorkbenchReady,
  openWorkbench,
  test
} from "./helpers.mjs";

const LIVE_HYBRID_MAX_PEAK_RSS_BYTES = 32 * 1024 * 1024 * 1024;

test.describe.configure({mode: "serial"});

test.describe("Krátký živý průchod Apertusem", () => {
  test.beforeEach(async ({appPage: page}) => {
    await expectWorkbenchReady(page, {mock: false, timeout: 600_000});
    const health = await (await page.request.get("/api/health")).json();
    expect(health.active_model?.id).toBe("hybrid");
    expect(health.mock).toBe(false);
    expect(health.performance?.process_peak_rss_bytes).toBeGreaterThan(0);
    expect(health.performance?.process_peak_rss_bytes).toBeLessThanOrEqual(
      LIVE_HYBRID_MAX_PEAK_RSS_BYTES
    );
  });

  test("tokenizer zachová text a používá skutečný Apertus runtime", async ({appPage: page}) => {
    await openWorkbench(page, "/tokenizace");
    await page.locator("#text-a").fill("Mám rád Prahu.");
    await page.locator("#text-b").fill("");
    await page.locator("#analyze").click();

    await expect(page.locator("#pipeline")).toContainText(/Přesná shoda/i);
    await expect(page.locator("#pipeline")).not.toContainText(/mock/i);
  });

  test("zahrada vrátí živou distribuci a dovolí otevřít jednu větev", async ({appPage: page}) => {
    await openWorkbench(page, "/");
    const column = page.locator(".prompt-column").first();
    const root = column.locator(".level").first();
    await expect(root).toBeVisible({timeout: 180_000});
    await expect(root.locator("button[data-branch-column]").first()).toBeEnabled();

    const visiblePercentages = await root.locator(".branch .probability").allTextContents();
    expect(visiblePercentages.length).toBeGreaterThan(0);
    expect(visiblePercentages.join(" ")).not.toMatch(/NaN|Infinity/);

    await root.locator("button[data-branch-column]").first().click();
    await expect(column.locator(".level")).toHaveCount(2, {timeout: 180_000});
  });

  test("teacher-forced překvapení krátkého textu vrátí konečná čísla", async ({appPage: page}) => {
    await openWorkbench(page, "/prekvapeni");
    await page.locator("#source-text").fill("Praha leží na řece Vltavě.");
    await page.locator("#analyze").click();

    await expect(page.locator("#results")).toBeVisible({timeout: 180_000});
    await expectFiniteText(page.locator("#summary-surprisal"));
    await expectFiniteText(page.locator("#summary-perplexity"));
    await expect(page.locator("#token-stream button")).not.toHaveCount(0);
    await page.locator("#display-mode-words").click();
    await expect(page.locator("#psycholinguistic-view")).toBeVisible();
    await expect(page.locator("#psych-word-list .psych-word-row")).not.toHaveCount(0);
    await expect(
      page.locator("#psych-word-list .psych-punctuation-row").last()
    ).toBeVisible();
    await expect(
      page.locator(
        "#psych-word-list .psych-punctuation-row summary .observed-token"
      ).last()
    ).toHaveText(".");
    await expectFiniteText(
      page.locator("#psych-word-list .psych-surprisal-badge").first()
    );
    await expectNoBusyOverlay(page);
  });

  test("kontrast dvou kontextů změří celé pole i společné pokračování", async ({appPage: page}) => {
    await openWorkbench(page, "/kontrast");
    await expect(page.locator("#analyze")).toBeEnabled();
    await page.locator("#analyze").click();

    await expect(page.locator("#results")).toBeVisible({timeout: 180_000});
    await expectFiniteText(page.locator("#summary-advantage"));
    await expectFiniteText(page.locator("#summary-js"));
    await expectFiniteText(page.locator("#summary-half-life"));
    await expectNoBusyOverlay(page);
  });
});
