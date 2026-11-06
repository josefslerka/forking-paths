import {test,expect} from "@playwright/test";

test.describe.serial("Local model management — synthetic filesystem/backend",()=>{
  let registry, local;
  const state=async(request)=>(await request.get("/api/model-admin")).json();
  test("first launch opens administration without loading or downloading weights",async({page,request})=>{
    await page.goto("/");await expect(page.locator("h1")).toHaveText("Správa modelů");
    await expect(page.locator("#setup-notice")).toBeVisible();
    registry=await state(request);local=registry.cache_path+"/local-example";
    expect(registry.status).toBe("setup_required");expect(registry.job).toBeNull();
    await expect(page.locator('[data-page="model-admin"]')).toBeVisible();
    await page.screenshot({path:"test-results/admin/model-management-first-run.png",fullPage:true});
  });
  test("connect a local model, set default, verify and use it without restart",async({page,request})=>{
    await page.goto("/sprava-modelu");await page.locator("#model-path").fill(local);
    await page.locator("#inspect-model").click();await expect(page.locator("#inspection-summary")).toContainText("potřebné soubory");
    await page.locator("#model-label").fill("E2E Local reader");await page.locator("#add-model").click();
    const card=page.locator('[data-model-id="local-example"]');await expect(card).toBeVisible();
    await card.getByRole("button",{name:"Nastavit jako výchozí",exact:true}).click();
    await expect(card.locator(".badge").filter({hasText:/^Výchozí$/})).toBeVisible();
    await card.getByRole("button",{name:"Ověřit výpočtem",exact:true}).click();
    await expect(page.locator("#job-message")).toContainText("proběhly v pořádku",{timeout:15000});
    await expect(card.locator(".badge").filter({hasText:"Právě načtený"})).toBeVisible();
    expect((await state(request)).active_model_id).toBe("local-example");
    await expect(card.getByRole("button",{name:"Odpojit",exact:true})).toBeDisabled();
    await page.screenshot({path:"test-results/admin/model-management-cs.png",fullPage:true});
  });
  test("hide, restore and disconnect a record without deleting its directory",async({page,request})=>{
    await page.goto("/sprava-modelu");await page.locator("#model-path").fill(registry.cache_path+"/bad-example");await page.locator("#inspect-model").click();
    await expect(page.locator("#add-model")).toBeEnabled();await page.locator("#add-model").click();
    const card=page.locator('[data-model-id="bad-example"]');
    await card.getByRole("button",{name:"Ověřit výpočtem",exact:true}).click();
    await expect(page.locator("#job-message")).toContainText("nepodařilo načíst",{timeout:15000});
    expect((await state(request)).active_model_id).toBe("local-example");
    await card.getByRole("button",{name:"Použít",exact:true}).click();
    await expect(page.locator("#load-error-notice")).toBeVisible();
    await page.locator('[data-model-id="local-example"]').getByRole("button",{name:"Použít",exact:true}).click();
    await expect(page.locator("#load-error-notice")).toBeHidden();
    await card.getByRole("button",{name:"Skrýt z nabídky",exact:true}).click();
    await expect(card.locator(".badge").filter({hasText:/^Skrytý$/})).toBeVisible();
    await card.getByRole("button",{name:"Vrátit do nabídky",exact:true}).click();
    await card.getByRole("button",{name:"Odpojit",exact:true}).click();await expect(page.locator("#disconnect-dialog")).toBeVisible();
    await page.locator("#confirm-disconnect").click();await expect(card).toHaveCount(0);
    const inspected=await request.post("/api/model-admin/inspect",{data:{path:registry.cache_path+"/bad-example"}});
    expect((await inspected.json()).files_ok).toBe(true);
  });
  test("review size and require consent before a simulated profile installation",async({page,request})=>{
    await page.goto("/sprava-modelu");await page.locator("#profile-id").selectOption("bf16");await page.locator("#review-installation").click();
    await expect(page.locator("#install-dialog")).toBeVisible();await expect(page.locator("#install-size")).toContainText("17");
    await expect(page.locator("#start-installation")).toBeDisabled();await page.locator("#confirm-download").check();await expect(page.locator("#start-installation")).toBeEnabled();
    await page.locator("#start-installation").click();await expect(page.locator("#job-message")).toContainText("Soubory jsou připravené",{timeout:15000});
    const value=await state(request);expect(value.active_model_id).toBe("local-example");expect(value.models.find(model=>model.id==="bf16").files_ok).toBe(true);
    await expect(page.locator('[data-model-id="bf16"]')).toBeVisible();
  });
  test("cancel a simulated installation and restore the loaded model",async({page,request})=>{
    await page.goto("/sprava-modelu");await page.locator("#profile-id").selectOption("eurollm9-hf");await page.locator("#review-installation").click();await page.locator("#confirm-download").check();await page.locator("#start-installation").click();
    await expect(page.locator("#cancel-job")).toBeEnabled();await page.locator("#cancel-job").click();await expect(page.locator("#job-message")).toContainText("přerušena",{timeout:15000});
    await expect.poll(async()=>(await state(request)).active_model_id).toBe("local-example");
  });
  test("English administration translates controls but preserves model names and paths",async({page})=>{
    await page.goto("/sprava-modelu?lang=en");await expect(page.locator("h1")).toHaveText("Model management");
    await expect(page.locator("#inspect-model")).toHaveText("Check directory");await expect(page.locator("#review-installation")).toHaveText("Review installation plan");
    await expect(page.locator('[data-model-id="local-example"] h3')).toHaveText("E2E Local reader");
    await expect(page.locator("#model-list")).not.toContainText("Ověřit výpočtem");
    await page.screenshot({path:"test-results/admin/model-management-en.png",fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.reload();await expect(page.locator("h1")).toBeVisible();
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
    await page.screenshot({path:"test-results/admin/model-management-mobile.png",fullPage:true});
  });
});
