import {
  expect,
  expectFiniteText,
  expectNoBusyOverlay,
  expectReadableText,
  openWorkbench,
  test
} from "./helpers.mjs";

const untranslatedInterfaceCopy = (page) => page.evaluate(() => {
  const protectedSelector = [
    "script", "style", "code", "pre", "textarea",
    "[data-i18n-ignore]",
    ".archive-layout .prefix", ".archive-layout .token",
    ".context-future", ".context-prefix", ".detail-head h2", ".detail-note",
    "body[data-workbench-page=surprise] #detail-title",
    "body[data-workbench-page=tokenization] #detail-title",
    "body[data-workbench-page=embeddings] #result-title",
    ".experiment-prompt", ".experiment-text", ".experiment-prefix",
    ".field-prefix", ".heatmap-token", ".level-prefix",
    ".observed-token", ".path-text", ".prediction-token",
    ".record-button .record-preview", ".record-button .record-title",
    "body[data-workbench-page=archive] .metrics strong",
    "#experiment-left option", "#experiment-right option",
    "#row-heading", ".token-piece", ".token-cell", ".token-symbol",
    ".trajectory-completion", ".trajectory-prefix", ".trajectory-token-text",
    "body[data-workbench-page=contrast] #results-title"
  ].join(",");
  const walker = document.createTreeWalker(
    document.documentElement,
    NodeFilter.SHOW_TEXT
  );
  const found = [];
  let node;
  while ((node = walker.nextNode())) {
    const text = node.data.replace(/\s+/g, " ").trim();
    if (
      text &&
      (
        /[áčďéěíňóřšťúůýž]/i.test(text) ||
        /\b(?:znaky|indexy|souhrn|smazat|zachyceno|uloženo|metoda|verze|populace|kvantily|modelu|serveru|latence|normalizace|jazyky|pozic|vrstev|projekce|generace|pouze|oblasti|zdroj|lze|bitech|shoda)\b/i.test(text)
      ) &&
      !node.parentElement?.closest(protectedSelector)
    ) {
      found.push(text);
    }
  }
  return [...new Set(found)].slice(0, 20);
});

test.describe("Pravděpodobnostní zahrada — deset klíčových cest nad stubem", () => {
  test("00 · Anglické rozhraní je úplné, trvalé a nemění uživatelský text", async ({appPage: page}) => {
    await openWorkbench(page, "/?lang=en", {mock: true});

    await expect(page.locator("html")).toHaveAttribute("lang", "en");
    await expect(
      page.locator('.pg-language-button[data-language="en"]')
    ).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator("h1").first()).toContainText("Probability Garden");
    await expect(page.getByRole("link", {name: /Tokenization/})).toBeVisible();
    await expect(page.getByRole("link", {name: /Text surprisal/})).toBeVisible();
    await expect(page.getByRole("link", {name: /Causal transfer/})).toBeVisible();

    await page.getByRole("link", {name: /Tokenization/}).click();
    await expect(page).toHaveURL(/\/tokenizace$/);
    await expect(page.locator("html")).toHaveAttribute("lang", "en");
    await page.locator("#text-a").fill("Mám rád Prahu.");
    await page.waitForTimeout(50);
    await expect(page.locator("#text-a")).toHaveValue("Mám rád Prahu.");

    await page.locator('.pg-language-button[data-language="cs"]').click();
    await expect(page.locator("html")).toHaveAttribute("lang", "cs");
    await expect(page.getByRole("link", {name: /Překvapení textu/})).toBeVisible();
  });

  test("00b · Všech čtrnáct anglických pohledů má přeloženou navigaci a nadpisy", async ({appPage: page}) => {
    await page.addInitScript(() => {
      window.localStorage.setItem("probability-garden:language", "en");
    });
    const routes = new Map([
      ["/", "Probability Garden"],
      ["/tokenizace", "Tokenization"],
      ["/embeddingy", "Embeddings"],
      ["/prekvapeni", "Text surprisal"],
      ["/detektor", "AI detector"],
      ["/kontrast", "Context contrast"],
      ["/vrstvy", "RAW layers"],
      ["/tuned-lens", "RAW × TUNED"],
      ["/j-lens", "RAW × J-LENS"],
      ["/atribuce", "Preference"],
      ["/kauzalni-prenos", "Causal transfer"],
      ["/steering", "Steering"],
      ["/jak-to-cist", "Workbench guide"],
      ["/archiv", "Measurement archive"]
    ]);

    for (const [route, currentLabel] of routes) {
      await openWorkbench(page, route);
      await expect(page.locator("html")).toHaveAttribute("lang", "en");
      await expect(
        page.locator('.pg-view-link[aria-current="page"]')
      ).toContainText(currentLabel);
      await expect.poll(async () => {
        const headings = await page.locator("main h1").allTextContents();
        return headings.filter((heading) => /[áčďéěíňóřšťúůýž]/i.test(heading));
      }, {message: `The main heading on ${route} must be translated.`}).toEqual([]);
    }
  });

  test("00bb · Anglické pohledy nemají české ostrůvky mimo experimentální data", async ({appPage: page}) => {
    await page.addInitScript(() => {
      window.localStorage.setItem("probability-garden:language", "en");
    });
    const routes = [
      "/", "/tokenizace", "/embeddingy", "/prekvapeni", "/detektor",
      "/kontrast", "/vrstvy", "/tuned-lens", "/j-lens", "/atribuce",
      "/kauzalni-prenos", "/steering", "/jak-to-cist", "/archiv", "/sprava-modelu"
    ];

    for (const route of routes) {
      await openWorkbench(page, route);
      const untranslated = await untranslatedInterfaceCopy(page);
      expect(untranslated, `Untranslated copy on ${route}`).toEqual([]);
    }
  });

  test("00bc · Správa modelů je v demu jen ke čtení", async ({appPage: page, request}) => {
    await openWorkbench(page, "/sprava-modelu", {mock:true});
    await expect(page.locator("#demo-notice")).toBeVisible();
    await expect(page.locator("#inspect-model")).toBeDisabled();
    await expect(page.locator("#review-installation")).toBeDisabled();
    const result = await request.post("/api/model-admin/edit", {data:{action:"remove",model_id:"bf16"}});
    expect(result.status()).toBe(400);
    expect((await result.json()).code).toBe("demo_readonly");
  });

  test("00c · Angličtina překládá i dynamický výsledek a zachová jeho hodnoty", async ({appPage: page}) => {
    await page.addInitScript(() => {
      window.localStorage.setItem("probability-garden:language", "en");
    });
    await openWorkbench(page, "/kontrast", {mock: true});
    const prefixA = await page.locator("#prefix-a").inputValue();
    const prefixB = await page.locator("#prefix-b").inputValue();
    expect(prefixA).toBe("Hlavním městem České republiky je");
    expect(prefixB).toBe("Českou metropolí je");
    await page.locator("#analyze").click();

    await expect(page.locator("#results")).toBeVisible();
    await expect(page.locator("#message")).toContainText(
      /Done — two forward passes in [\d,.]+ ms\./
    );
    await expect(page.locator("#message")).not.toContainText("__PG_SLOT_");
    await expect(page.locator("#result-context")).toContainText(prefixA);
    await expect(page.locator("#result-context")).toContainText(prefixB);
    await expect(page.locator("#result-context")).toContainText(/target tokens/);
    await expect(page.locator("#result-context")).toContainText(
      /(?:live computation|from cache)/
    );
    await expect(page.locator("#result-context")).not.toContainText(
      /(?:tokenů cíle|živý výpočet|z cache)/
    );

    await page.evaluate(() => {
      const specimen = document.createElement("p");
      specimen.id = "i18n-observed-token-specimen";
      specimen.innerHTML = (
        '<span class="observed-token">je</span> · ' +
        '<span class="interface-copy">Hotovo</span>'
      );
      document.body.append(specimen);
    });
    await expect(page.locator("#i18n-observed-token-specimen .observed-token"))
      .toHaveText("je");
    await expect(page.locator("#i18n-observed-token-specimen .interface-copy"))
      .toHaveText("Done");
    expect(
      await untranslatedInterfaceCopy(page),
      "The English context-contrast result must not contain Czech UI copy."
    ).toEqual([]);
  });

  test("00cc · Malý nenulový poměr, trvalý pokles a stale výsledek jsou čitelné", async ({appPage: page}) => {
    await page.addInitScript(() => {
      window.localStorage.setItem("probability-garden:language", "en");
    });
    await openWorkbench(page, "/kontrast", {mock: true});
    await page.route("**/api/context-contrast", async (route) => {
      const response = await route.fetch();
      const data = await response.json();
      data.target_score.probability_ratio_a_over_b = 3.94e-12;
      data.target_score.probability_ratio_status = "finite";
      data.target_score.information_advantage_a_over_b_bits = Math.log2(3.94e-12);
      data.target_score.favored_context = "b";
      data.persistence.summary.persistent_half_life_step = null;
      await route.fulfill({
        response,
        contentType: "application/json",
        body: JSON.stringify(data)
      });
    });
    const originalPrefix = await page.locator("#prefix-a").inputValue();
    await page.locator("#analyze").click();
    await expect(page.locator("#results")).toBeVisible();
    await expect(page.locator("#summary-ratio")).toContainText(/E-12×$/);
    await expect(page.locator("#summary-ratio")).not.toHaveText("0×");
    await expect(page.locator("#summary-half-life")).toHaveText("not observed");
    await expect(page.locator("#summary-half-life-detail"))
      .toContainText("persisted to the end");

    await page.locator("#prefix-a").fill("Changed prefix");
    await expect(page.locator("#stale-notice")).toBeVisible();
    await expect(page.locator("#stale-notice")).toContainText(
      "does not match the current input"
    );
    await expect(page.locator("#result-badge")).toHaveText("stale");
    await page.locator("#prefix-a").fill(originalPrefix);
    await expect(page.locator("#stale-notice")).toBeHidden();

    await page.locator("#save-experiment").click();
    await expect(page.locator("#message")).toContainText(/saved|already in the archive/i);
    await openWorkbench(page, "/archiv");
    const savedContrast = page.locator("#record-list [data-record-id]").first();
    await expect(savedContrast).toBeVisible();
    await savedContrast.click();
    await expect(page.locator("#detail")).toContainText(/3\.94E-12×/);
    await expect(page.locator("#detail")).toContainText("not observed");
  });

  test("00d · Anglický embeddingový výsledek zachová slovo a přeloží výklad", async ({appPage: page}) => {
    await page.addInitScript(() => {
      window.localStorage.setItem("probability-garden:language", "en");
    });
    await openWorkbench(page, "/embeddingy", {mock: true});
    await page.locator("#text-a").fill("Hlavním");
    await page.locator("#tokenize").click();
    await page.locator("#token-streams button.token-chip").first().click();
    await page.locator("#measure").click();

    await expect(page.locator("#results")).toBeVisible();
    await expect(page.locator("#result-title")).toContainText(/^(?:Word|Token) Hlavním:/);
    await expect(page.locator("#result-title")).toContainText(/(?:fragment|row)/);
    await expect(page.locator("#result-title")).not.toContainText(
      /(?:Slovo|střep|trajektorie|jeden řádek)/i
    );
    await expect(page.locator("#row-heading")).toHaveText("Hlavním");
    await expect(page.locator("#form-status")).toContainText(/^Done —/);
    expect(
      await untranslatedInterfaceCopy(page),
      "The English embedding result must not contain Czech UI copy."
    ).toEqual([]);
  });

  test("00e · Angličtina pokrývá dynamické výsledky analytických pohledů", async ({appPage: page}) => {
    await page.addInitScript(() => {
      window.localStorage.setItem("probability-garden:language", "en");
    });

    await openWorkbench(page, "/prekvapeni", {mock: true});
    await page.locator("#source-text").fill("Krátký archivní regresní text.");
    await page.locator("#analyze").click();
    await expect(page.locator("#results")).toBeVisible();
    expect(await untranslatedInterfaceCopy(page), "Text surprisal result").toEqual([]);
    await page.locator("#display-mode-words").click();
    await expect(page.locator("#psycholinguistic-view")).toBeVisible();
    expect(
      await untranslatedInterfaceCopy(page),
      "Psycholinguistic word-level result"
    ).toEqual([]);

    await openWorkbench(page, "/vrstvy", {mock: true});
    await page.locator("#analyze-button").click();
    await expect(page.locator("#results")).toBeVisible();
    expect(await untranslatedInterfaceCopy(page), "RAW-layer result").toEqual([]);

    await openWorkbench(page, "/tuned-lens", {mock: true});
    await page.locator("#analyze-button").click();
    await expect(page.locator("#results")).toBeVisible();
    expect(await untranslatedInterfaceCopy(page), "RAW × TUNED result").toEqual([]);

    await openWorkbench(page, "/atribuce", {mock: true});
    await page.locator("#run-attribution").click();
    await expect(page.locator("#results")).toBeVisible();
    expect(await untranslatedInterfaceCopy(page), "Component result").toEqual([]);

    await openWorkbench(page, "/kauzalni-prenos", {mock: true});
    await page.locator("#check-alignment").click();
    await expect(page.locator("#alignment-panel")).toBeVisible();
    await page.locator("#select-shared-after").click();
    await page.locator("#layer-preset").selectOption("quarters");
    await page.locator("#run-patching").click();
    await expect(page.locator("#results")).toBeVisible();
    expect(await untranslatedInterfaceCopy(page), "Causal-transfer result").toEqual([]);

    await openWorkbench(page, "/steering", {mock: true});
    await page.locator("#run-steering").click();
    await expect(page.locator("#results")).toBeVisible();
    expect(await untranslatedInterfaceCopy(page), "Immediate steering result").toEqual([]);
    await page.locator('[data-stage-link="trajectory"]').click();
    await page.locator("#trajectory-max-tokens").selectOption("12");
    await page.locator("#run-trajectory").click();
    await expect(page.locator("#trajectory-grid > *")).toHaveCount(3);
    expect(await untranslatedInterfaceCopy(page), "Steering trajectories").toEqual([]);
  });

  test("00 · Všechny pohledy používají světlý výzkumný shell", async ({appPage: page}) => {
    const routes = [
      "/",
      "/tokenizace",
      "/embeddingy",
      "/prekvapeni",
      "/detektor",
      "/kontrast",
      "/vrstvy",
      "/tuned-lens",
      "/j-lens",
      "/atribuce",
      "/kauzalni-prenos",
      "/steering",
      "/jak-to-cist",
      "/archiv"
    ];

    for (const route of routes) {
      await openWorkbench(page, route);
      await expect(page.locator(".pg-global-rail")).toBeVisible();
      const theme = await page.evaluate(() => {
        const rootStyle = getComputedStyle(document.documentElement);
        const bodyStyle = getComputedStyle(document.body);
        const railStyle = getComputedStyle(
          document.querySelector(".pg-global-rail")
        );
        const currentStyle = getComputedStyle(
          document.querySelector('.pg-view-link[aria-current="page"]')
        );
        return {
          scheme: rootStyle.colorScheme,
          paper: bodyStyle.getPropertyValue("--bg").trim(),
          ink: bodyStyle.getPropertyValue("--text").trim(),
          rail: railStyle.backgroundColor,
          current: currentStyle.backgroundColor
        };
      });
      expect(theme, `Světlý motiv musí být kanonický i na ${route}.`).toEqual({
        scheme: "light",
        paper: "#f3f0e8",
        ink: "#202722",
        rail: "rgb(236, 239, 233)",
        current: "rgb(220, 235, 225)"
      });
      await expectReadableText(page.locator([
        ".method-note > strong:visible",
        ".methodology-drawer summary span:visible",
        ".hero-note:visible",
        ".control-foot:visible",
        ".mode-switch span:visible"
      ].join(",")));
    }
  });

  test("01 · Zahrada: shell, první distribuce, větev a uložení", async ({appPage: page}) => {
    await openWorkbench(page, "/", {mock: true});

    await expect(page.getByRole("link", {name: /Tokenizace/})).toBeVisible();
    await expect(page.getByRole("link", {name: /Steering/})).toBeVisible();
    await expect(page.locator("#model-state")).toContainText(/simulace|připraven/i);

    const column = page.locator(".prompt-column").first();
    await expect(column.locator(".level").first()).toBeVisible();
    await expect(column.locator("button[data-branch-column]").first()).toBeEnabled();
    await expectReadableText(column.locator(".focus-node"));
    await column.locator("button[data-branch-column]").first().click();
    await expect(column.locator(".level")).toHaveCount(2);

    await expect(page.locator("#save-garden")).toBeEnabled();
    await page.locator("#save-garden").click();
    await expect(page.locator("#experiment-records")).toContainText(
      /bezpečně uložena|uložená pod UUID/i
    );
  });

  test("02 · Tokenizace: Unicode divergence a přechod na hranici predikce", async ({appPage: page}) => {
    await openWorkbench(page, "/tokenizace", {mock: true});

    await page.locator("#text-a").fill("Mám rád Prahu.");
    await page.locator("#text-b").fill("Ma\u0301m rád Prahu.");
    await page.locator("#analyze").click();

    await expect(page.locator("#comparison-summary")).toBeVisible();
    await expect(page.locator("#comparison-summary")).toContainText(/Odlišný|divergence/i);
    await expect(page.locator("#pipeline")).toContainText(/Přesná shoda|tokenov/i);
    await expect(page.locator("#boundary-slider")).toBeEnabled();
    await expect(page.locator("#open-garden")).toHaveAttribute("href", /prompt=/);
    await expect(page.locator("#open-surprise")).toHaveAttribute("href", /text=/);

    const tokenLabels = page.locator(".token-card .token-display, .token-card .token-id, .token-card .token-size");
    await expectReadableText(tokenLabels);
    const ordinaryToken = page.locator(".token-card:not(.is-special):not(.is-divergence)").first();
    await ordinaryToken.hover();
    await expectReadableText(tokenLabels);
    await ordinaryToken.click();
    await ordinaryToken.focus();
    await expect(ordinaryToken).toHaveAttribute("aria-pressed", "true");
    await expectReadableText(tokenLabels);
  });

  test("03 · Embeddingy: jedno slovo, geometrie a rozkliknutí souseda", async ({appPage: page}) => {
    await openWorkbench(page, "/embeddingy", {mock: true});

    await page.locator("#text-a").fill("Hlavním");
    await page.locator("#tokenize").click();
    await expect(page.locator("#selection")).toBeVisible();

    const tokenChip = page.locator("#token-streams button.token-chip").first();
    await expect(tokenChip).toBeVisible();
    await tokenChip.click();
    await expect(page.locator("#measure")).toBeEnabled();
    await page.locator("#measure").click();

    await expect(page.locator("#results")).toBeVisible();
    await expect(page.locator(".metrics > .metric:first-child")).toHaveCSS(
      "background-color",
      "rgb(255, 253, 250)"
    );
    await expect(page.locator("#neighbor-chart")).toBeVisible();
    await expect(page.locator("#neighbor-chart")).toHaveCSS(
      "background-color",
      "rgb(246, 248, 244)"
    );
    await expect(page.locator("#neighbor-network-chart")).toHaveCSS(
      "background-color",
      "rgb(246, 248, 244)"
    );
    await expect(page.locator("#neighbor-list button[data-token-id]").first()).toBeVisible();
    const neighbor = page.locator("#neighbor-list button[data-token-id]").first();
    await neighbor.hover();
    await expect(neighbor).toHaveCSS("background-color", "rgb(231, 238, 232)");
    await expectReadableText(neighbor.locator("b, code, span"));
    await neighbor.focus();
    await expectReadableText(neighbor.locator("b, code, span"));
    await expect(page.locator("#input-technical-details")).not.toHaveAttribute("open", "");
    await expect(page.locator("#row-stats")).toBeHidden();
    await expect(page.locator("#coordinate-strip")).toBeHidden();
    await expect(page.locator("#context-cards .context-card")).toHaveCount(1);
    await expect(page.locator("#context-cards .state-details .row-stats")).toBeHidden();
    await page.locator("#input-technical-details > summary").click();
    await expect(page.locator("#row-stats")).toBeVisible();
    await expectReadableText(page.locator("#row-stats dd"));
    await page.locator("#input-technical-details > summary").click();
    const networkNodes = page.locator("#neighbor-network-nodes button[data-token-id]");
    const before = await networkNodes.count();
    await page.locator("#neighbor-list button[data-token-id]").first().click();
    await expect.poll(() => networkNodes.count()).toBeGreaterThan(before);
    await expect(page.locator("#neighbor-network-status")).toContainText(/rozbal|rozší/i);
    await expectNoBusyOverlay(page);
  });

  test("03b · Dva kontextové stavy sdílejí vrstvu, projekci i měřítko", async ({appPage: page}, testInfo) => {
    for (const [model, layers] of [["mock", 32], ["mock-eurollm9", 42]]) {
      await openWorkbench(page, "/embeddingy", {mock: true});
      if (await page.locator("#pg-model-mode").inputValue() !== model) {
        await page.locator("#pg-model-mode").selectOption(model);
        await expect.poll(async () => (await (await page.request.get("/api/health")).json()).active_model?.id).toBe(model);
        await openWorkbench(page, "/embeddingy", {mock: true});
      }
      await page.locator("#toggle-context-b").click();
      await page.locator("#text-a").fill("Na kopci stojí zámek.");
      await page.locator("#text-b").fill("Na dveřích je rozbitý zámek.");
      await page.locator("#tokenize").click();
      await page.locator('button.token-chip[data-side="0"]').filter({hasText: "zámek"}).click();
      const response = page.waitForResponse((item) => item.url().endsWith("/api/embeddings") && item.request().method() === "POST");
      await page.locator("#measure").click();
      const result = await (await response).json();
      await expect(page.locator("#context-state-heading")).toHaveText("Stejný začátek, dva kontextové stavy");
      await expect(page.locator("#context-cards .context-card")).toHaveCount(2);
      await expect(page.locator("#state-layer option")).toHaveCount(layers + 1);
      await expect(page.locator("#state-layer")).toHaveValue(String(layers));
      await expect(page.locator("#context-cards .state-details[open]")).toHaveCount(0);

      const canvases = page.locator(".context-state-chart");
      await expect(canvases.nth(0)).toHaveAttribute("data-layer", String(layers));
      const finalScale = await canvases.nth(0).getAttribute("data-projection-scale");
      expect(await canvases.nth(1).getAttribute("data-projection-scale")).toBe(finalScale);
      await page.locator("#state-show-input").click();
      await expect(page.locator("#state-layer")).toHaveValue("0");
      await expect(page.locator("#state-comparison")).toContainText("oba stavy totožné");
      expect(Number(await canvases.nth(0).getAttribute("data-projection-x"))).toBeCloseTo(Number(await canvases.nth(1).getAttribute("data-projection-x")), 10);
      expect(Number(await canvases.nth(0).getAttribute("data-projection-y"))).toBeCloseTo(Number(await canvases.nth(1).getAttribute("data-projection-y")), 10);
      expect(await canvases.nth(0).getAttribute("data-projection-scale")).toBe(finalScale);

      await page.locator("#state-layer").selectOption("8");
      await expect(page.locator("#context-cards .context-card[data-layer='8']")).toHaveCount(2);
      expect(await canvases.nth(0).getAttribute("data-projection-scale")).toBe(finalScale);
      await page.locator("#context-cards .state-details > summary").first().click();
      await expect(page.locator("#context-cards .state-details .row-stats").first()).toContainText(result.contexts[0].layers[8].state_summary.float32_sha256);
      await expectReadableText(page.locator("#context-cards .state-details .row-stats").first().locator("dd"));
      await page.locator("#state-layer").selectOption("12");
      await expect(page.locator("#context-cards .state-details").first()).toHaveAttribute("open", "");
      await expect(page.locator("#context-cards .state-details .row-stats").first()).toContainText(result.contexts[0].layers[12].state_summary.float32_sha256);
      await page.locator("#context-cards .state-details > summary").first().click();
      await page.locator("#state-show-max").click();
      await expect(page.locator("#state-layer")).toHaveValue(String(result.pairs[0].max_distance_layer));
      await page.locator("#context-state-section").scrollIntoViewIfNeeded();
      if (layers === 32) await page.screenshot({path: testInfo.outputPath("context-states.png")});
    }
    await page.setViewportSize({width: 390, height: 844});
    const boxes = await page.locator("#context-cards .context-card").evaluateAll((items) => items.map((item) => ({top: item.getBoundingClientRect().top, bottom: item.getBoundingClientRect().bottom})));
    expect(boxes[1].top).toBeGreaterThanOrEqual(boxes[0].bottom);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.locator("#pg-model-mode").selectOption("mock");
    await expect.poll(async () => (await (await page.request.get("/api/health")).json()).active_model?.id).toBe("mock");
  });

  test("04 · Překvapení: dva hotové texty a slovní telemetrie", async ({appPage: page}) => {
    await openWorkbench(page, "/prekvapeni", {mock: true});

    await page.locator("#source-text").fill(
      "Ráno se otevřelo okno a do pokoje vstoupilo světlo."
    );
    await page.locator("#add-text-b").click();
    await page.locator("#label-a").fill("Text A");
    await page.locator("#label-b").fill("Text B");
    await page.locator("#source-text-b").fill(
      "Ráno okno zpívalo a světlo se rozlilo po stropě."
    );
    await expect(page.locator("#analyze")).toBeEnabled();
    await page.locator("#analyze").click();

    await expect(page.locator("#results")).toBeVisible();
    await expect(page.locator("#comparison-view")).toBeVisible();
    await expectReadableText(page.locator(".legend-stop strong:visible, .aggregation-explanation:visible, .aggregation-span-metric:visible"));
    await expect(page.locator("#comparison-metric-body tr")).not.toHaveCount(0);
    await expectFiniteText(
      page.locator("#comparison-metric-body tr").first().locator("td").first()
    );
    await page.locator("#display-mode-words").click();
    await expect(page.locator("#psycholinguistic-view")).toBeVisible();
    await expect(page.locator("#comparison-view")).toBeHidden();
    await expect(page.locator("#psych-word-list .psych-word-row")).not.toHaveCount(0);
    const punctuationRow = page.locator(
      "#psych-word-list .psych-punctuation-row"
    ).last();
    await expect(punctuationRow).toBeVisible();
    await expect(
      punctuationRow.locator("summary .observed-token")
    ).toHaveText([".", "."]);
    const firstWord = page.locator("#psych-word-list .psych-word-row").first();
    await expect(firstWord.locator(".psych-metric").nth(1)).not.toContainText("—");
    await firstWord.locator("summary").click();
    await expect(firstWord.locator(".psych-subword-table")).toHaveCount(2);
    const wordDownload = page.waitForEvent("download");
    await page.locator("#export-word-csv").click();
    const downloadedWords = await wordDownload;
    expect(downloadedWords.suggestedFilename()).toMatch(
      /^psycholinguistic-surprisal-.+\.csv$/
    );
    expect(await downloadedWords.failure()).toBeNull();
    await expectNoBusyOverlay(page);
  });

  test("05 · Kontrast kontextů: konkrétní pokračování, celé pole a doznívání", async ({appPage: page}) => {
    await openWorkbench(page, "/kontrast", {mock: true});

    await expect(page.locator("#analyze")).toBeEnabled();
    await page.locator("#analyze").click();

    await expect(page.locator("#results")).toBeVisible();
    await expectFiniteText(page.locator("#summary-advantage"));
    await expectFiniteText(page.locator("#summary-js"));
    await expectFiniteText(page.locator("#summary-half-life"));
    await expectReadableText(page.locator(".field-card h4, .chart-card h4, .field-prefix, .token-chip, .warning-list li"));
  });

  test("06 · RAW vrstvy: vertikální řez má všech 32 vrstev", async ({appPage: page}) => {
    await openWorkbench(page, "/vrstvy", {mock: true});

    await expect(page.locator("#analyze-button")).toBeEnabled();
    await page.locator("#analyze-button").click();

    await expect(page.locator("#results")).toBeVisible();
    await expectFiniteText(page.locator("#final-token"));
    await expect(page.locator("#trajectory-body tr")).toHaveCount(32);
  });

  test("07 · Preference: lidský výklad, dvě možnosti a technické detaily", async ({appPage: page}, testInfo) => {
    await openWorkbench(page, "/atribuce", {mock: true});

    await expect(page.locator("h1")).toHaveText("Odkud se vzala preference?");
    await expect(page.locator("#token-a")).toHaveValue("␠Praha");
    await expect(page.locator("#token-b")).toHaveValue("␠Brno");
    await expect(page.locator("#method-details > .preference-details-content")).toBeHidden();
    await expect(page.locator("#run-attribution")).toBeEnabled();
    const firstResponse = page.waitForResponse((response) => response.url().endsWith("/api/component-attribution") && response.request().method() === "POST");
    await page.locator("#run-attribution").click();
    const first = await (await firstResponse).json();

    await expect(page.locator("#results")).toBeVisible();
    await expectFiniteText(page.locator("#summary-winner"));
    await expect(page.locator("#technical-results > .preference-details-content")).toBeHidden();
    await expect(page.locator("#summary-reconstruction")).toBeHidden();
    await expect(page.locator("#token-contract")).toBeHidden();
    await expect(page.locator("#contribution-direction")).toContainText(first.contrast.a.display);
    await expect(page.locator("#contribution-direction")).toContainText(first.contrast.b.display);
    await expect(page.locator("#summary-probability .probability-line")).toHaveCount(2);
    await expect(page.locator("#contribution-chart .attention-bar")).toHaveCount(32);
    await expect(page.locator("#contribution-chart .mlp-bar")).toHaveCount(32);
    await expect(page.locator("#contribution-chart .negative-bar")).toHaveCount(0);
    await expectReadableText(page.locator(".preference-summary .summary-label, .preference-summary .summary-value, .preference-summary .summary-detail"));
    await page.screenshot({path: testInfo.outputPath("preference-result.png")});

    await page.locator("#technical-results > summary").click();
    await expect(page.locator("#summary-reconstruction")).toContainText(/uzavřeno/i);
    await expect(page.locator("#summary-reconstruction")).toBeVisible();
    await expect(page.locator("#token-contract")).toBeVisible();
    await expect(page.locator("#layer-table tr")).toHaveCount(32);
    await expectReadableText(page.locator("#technical-results .technical-summary p, #token-contract dt, #token-contract dd"));
    await page.locator("#technical-results > summary").click();

    await page.locator("#swap-tokens").click();
    await expect(page.locator("#token-a")).toHaveValue("␠Brno");
    await expect(page.locator("#token-b")).toHaveValue("␠Praha");
    await expect(page.locator("#results")).toBeHidden();
    const swappedResponse = page.waitForResponse((response) => response.url().endsWith("/api/component-attribution") && response.request().method() === "POST");
    await page.locator("#run-attribution").click();
    const swapped = await (await swappedResponse).json();
    await expect(page.locator("#results")).toBeVisible();
    expect(swapped.final.contrast).toBeCloseTo(-first.final.contrast, 8);
    await expect(page.locator("#contribution-direction")).toContainText(swapped.contrast.a.display);
    await expect(page.locator("#technical-results > .preference-details-content")).toBeHidden();

    await page.setViewportSize({width: 390, height: 844});
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.locator("#prompt").fill("Změněný začátek textu");
    await expect(page.locator("#results")).toBeHidden();
  });

  test("08 · Steering: okamžitý efekt a tři trajektorie do tečky", async ({appPage: page}) => {
    await openWorkbench(page, "/steering", {mock: true});

    await expect(page.locator("#contrast-count")).toContainText(/3 platn/i);
    await expect(page.locator("#run-steering")).toBeEnabled();
    await page.locator("#run-steering").click();

    await expect(page.locator("#results")).toBeVisible();
    await expect(page.locator("#aligned-path-tool")).toHaveCSS(
      "background-color",
      "rgb(255, 253, 250)"
    );
    await expectFiniteText(page.locator("#summary-js"));
    await expectReadableText(page.locator(".warning-panel h3, .warning-panel li"));
    await page.locator('[data-stage-link="trajectory"]').click();
    await expect(page.locator("#trajectory-section")).toBeVisible();
    await expect(page.locator("#run-trajectory")).toBeEnabled();
    await page.locator("#trajectory-max-tokens").selectOption("12");
    await page.locator("#run-trajectory").click();

    await expect(page.locator("#trajectory-status")).toContainText(/Hotovo/i);
    await expect(page.locator("#trajectory-grid > *")).toHaveCount(3);
  });

  test("09 · Kauzální přenos: zarovnání, výběr pozic a patching", async ({appPage: page}) => {
    await openWorkbench(page, "/kauzalni-prenos", {mock: true});

    await expectReadableText(page.locator("#swap-ab"));
    await expect(page.locator("#check-alignment")).toBeEnabled();
    await page.locator("#check-alignment").click();
    await expect(page.locator("#alignment-panel")).toBeVisible();
    await page.locator("#select-shared-after").click();
    await page.locator("#layer-preset").selectOption("quarters");
    await expect(page.locator("#run-patching")).toBeEnabled();
    await page.locator("#run-patching").click();

    await expect(page.locator("#results")).toBeVisible();
    await expectFiniteText(page.locator("#summary-maximum"));
    await expect(page.locator("#heatmap")).toBeVisible();
  });

  test("10 · Archiv: uložit měření a exportovat právě jeden experiment", async ({appPage: page}) => {
    await openWorkbench(page, "/prekvapeni", {mock: true});
    await page.locator("#source-text").fill("Krátký archivní regresní text.");
    await page.locator("#analyze").click();
    await expect(page.locator("#results")).toBeVisible();
    await expect(page.locator("#save-experiment")).toBeEnabled();
    await page.locator("#save-experiment").click();
    await expect(page.locator("#message")).toContainText(/trvale uložen|už v archivu/i);

    await openWorkbench(page, "/archiv");
    const firstRecord = page.locator("#record-list [data-record-id]").first();
    await expect(firstRecord).toBeVisible();
    await firstRecord.click();
    const singleJson = page.locator("#detail [data-export-record]").first();
    await expect(singleJson).toBeVisible();

    const downloadPromise = page.waitForEvent("download");
    await singleJson.click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/^apertus-experiment-.+\.json$/);
    expect(await download.failure()).toBeNull();
  });
});
