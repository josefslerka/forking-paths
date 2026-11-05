import {expect, openWorkbench, test} from "./helpers.mjs";

test("porovnání obou modelů se uloží a otevře z archivu", async ({appPage: page}) => {
  await openWorkbench(page,"/porovnani-modelu",{mock:true});
  await expect(page.locator("#model-a")).toHaveValue("mock");
  await expect(page.locator("#model-b")).toHaveValue("mock-eurollm9");
  await page.locator("#comparison-text").fill("Mám rád Prahu. Na stole je 300 knih.");
  await page.locator("#run-comparison").click();
  await expect(page.locator("#comparison-results svg")).toBeVisible();
  await expect(page.locator("#comparison-results")).toContainText("Prahu.");
  const archive = page.locator("#comparison-results").getByRole("link",{name:"Otevřít v archivu"});
  await archive.click();
  await expect(page.locator("#detail")).toContainText("Porovnání modelů");
  await page.getByRole("link",{name:"Otevřít uložené porovnání"}).click();
  await expect(page.locator("#comparison-results svg")).toBeVisible();
  await expect(page.locator("#comparison-text")).toHaveValue("Mám rád Prahu. Na stole je 300 knih.");
  await expect(page.locator("#comparison-results")).toContainText("300");
  await page.locator("#comparison-results").scrollIntoViewIfNeeded();
  await page.screenshot({path:"test-results/model-comparison-cs.png"});
});

test("dávka kontextů zachová přesné pokračování a shodu směrů", async ({appPage: page}) => {
  await openWorkbench(page,"/porovnani-modelu",{mock:true});
  await page.locator("#load-examples").click();
  await expect(page.locator("#input-mode")).toHaveValue("batch");
  const items = JSON.parse(await page.locator("#batch-json").inputValue());
  expect(items).toHaveLength(30);
  await page.locator("#batch-json").fill(JSON.stringify(items.slice(0,2)));
  await page.locator("#run-comparison").click();
  await expect(page.locator("#comparison-results article")).toHaveCount(2);
  await expect(page.locator("#comparison-results")).toContainText("byla unavená.");
  const health = await (await page.request.get("/api/health")).json();
  expect(health.active_model.id).toBe("mock");
  const exportURL=await page.locator("#comparison-results").getByRole("link",{name:"Export JSON"}).getAttribute("href");
  const id=exportURL.split("/")[3];
  await openWorkbench(page,`/porovnani-modelu?experiment=${id}`,{mock:true});
  await expect(page.locator("#input-mode")).toHaveValue("batch");
  expect(JSON.parse(await page.locator("#batch-json").inputValue()).map(item=>item.continuation)).toEqual(items.slice(0,2).map(item=>item.continuation));
});

test("anglické porovnání zachová český text měření", async ({appPage: page}) => {
  await openWorkbench(page,"/porovnani-modelu?lang=en",{mock:true});
  await expect(page.locator("h1")).toHaveText("Model comparison");
  await expect(page.locator("#comparison-text")).toHaveValue("Na stole ležela kniha, kterou nikdo neotevřel.");
  await page.locator("#comparison-text").fill("Žlutý kůň šel domů.");
  await page.locator("#run-comparison").click();
  await expect(page.locator("#comparison-results")).toContainText("Žlutý kůň šel domů.");
  await expect(page.locator("#comparison-results")).toContainText("Alignment");
});

test("společný steering uloží vlastní vrstvy, kontrolní směry a obnoví model", async ({appPage: page}) => {
  await openWorkbench(page,"/porovnani-modelu?mode=steering",{mock:true});
  await expect(page.locator("#input-mode")).toHaveValue("steering");
  await expect(page.locator("[data-steering-pair]")).toHaveCount(3);
  await page.locator("#comparison-steering-shams").fill("2");
  await page.locator("#load-steering-examples").click();
  await page.locator("#run-comparison").click();
  await expect(page.locator("#comparison-results article")).toHaveCount(3);
  await expect(page.locator("#comparison-results")).toContainText("L16 / 32");
  await expect(page.locator("#comparison-results")).toContainText("L21 / 42");
  await expect(page.locator("#comparison-results")).toContainText("Kontroly ≥ efekt");
  await expect(page.locator("#comparison-results svg")).toHaveCount(3);
  const health=await (await page.request.get("/api/health")).json();
  expect(health.active_model.id).toBe("mock");
  const url=await page.locator("#comparison-results").getByRole("link",{name:"Export JSON"}).getAttribute("href");
  const wrapper=await (await page.request.get(url)).json();
  const record=wrapper.records[0];
  expect(record.provenance.protocol.id).toBe("steering-comparison-v1");
  expect(record.payload.comparisons.every((item)=>item.a.all_effective_norm_matches_passed && item.b.all_effective_norm_matches_passed)).toBe(true);
  await page.locator("#comparison-results").getByRole("link",{name:"Otevřít v archivu"}).click();
  await expect(page.locator("#detail")).toContainText("Shoda směru steeringového účinku");
  await page.getByRole("link",{name:"Otevřít uložené porovnání"}).click();
  await expect(page.locator("#input-mode")).toHaveValue("steering");
  await expect(page.locator("#comparison-steering-batch")).toBeChecked();
  await expect(page.locator("#comparison-steering-shams")).toHaveValue("2");
  expect(JSON.parse(await page.locator("#comparison-steering-items").inputValue())).toEqual(record.payload.request.items);
  await expect(page.locator("#comparison-results article")).toHaveCount(3);
  await page.locator("#comparison-results").scrollIntoViewIfNeeded();
  await page.screenshot({path:"test-results/steering-comparison-cs.png"});
});

test("anglický steering zachová české konstrukční a testovací texty", async ({appPage: page}) => {
  await openWorkbench(page,"/porovnani-modelu?mode=steering&lang=en",{mock:true});
  await expect(page.locator("#steering-input h2")).toHaveText("Direction-building contrasts");
  await expect(page.locator('[data-pole="negative"]').first()).toHaveValue("Ten tón působí chladně v tomto rámci");
  await page.locator("#comparison-steering-shams").fill("2");
  await page.locator("#run-comparison").click();
  await expect(page.locator("#comparison-results")).toContainText("Symmetric effect");
  await expect(page.locator("#comparison-results")).toContainText("Jeho reakce byla");
  await expect(page.locator("#comparison-results")).toContainText("velmi laskavá.");
  await page.locator("#comparison-results").scrollIntoViewIfNeeded();
  await page.screenshot({path:"test-results/steering-comparison-en.png"});
});

test("EuroLLM zpřístupní samostatný steering až po vrstvu 42", async ({appPage: page}) => {
  await openWorkbench(page,"/steering",{mock:true});
  await page.locator("#pg-model-mode").selectOption("mock-eurollm9");
  await expect(page.locator("#steering-layer option")).toHaveCount(42);
  await page.locator("#steering-layer").selectOption("42");
  await page.locator("#steering-alpha").evaluate((input)=>{input.value="0";input.dispatchEvent(new Event("input",{bubbles:true}));});
  await page.locator("#run-steering").click();
  await expect(page.locator("#steering-diagnostics")).toContainText("42");
  const health=await (await page.request.get("/api/health")).json();
  expect(health.capabilities).toContain("steering");
  const response=await page.request.post("/api/model",{data:{model_id:"mock"}});
  expect(response.ok()).toBe(true);
});
