import {test,expect} from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

test.describe.serial("Experimental addons — synthetic package files",()=>{
  let root,original;
  const card=page=>page.locator('[data-addon-kind="tuned"]');
  test("connect, verify, enable and explain availability for a matching model",async({page,request})=>{
    const state=await (await request.get("/api/model-admin")).json();root=path.join(state.cache_path,"reader-fixture");
    await page.goto("/sprava-modelu#experimental-addons");
    await expect(card(page)).toBeVisible();await expect(card(page).getByText("Nepřipojeno",{exact:true})).toBeVisible();
    await page.locator('[data-model-id="bf16"]').getByRole("button",{name:"Použít",exact:true}).click();
    await expect(page.locator('[data-model-id="bf16"]')).toContainText("Právě načtený");
    await expect(page.locator('[data-page="tuned"]')).toHaveAttribute("title","Není připojený potřebný experimentální doplněk.");
    await card(page).getByLabel("Složka balíčku").fill(root);
    await card(page).getByRole("button",{name:"Zkontrolovat složku",exact:true}).click();
    await card(page).getByRole("button",{name:"Připojit balíček",exact:true}).click();
    await expect(card(page).getByRole("button",{name:"Zapnout",exact:true})).toBeDisabled();
    await card(page).getByRole("button",{name:"Ověřit balíček",exact:true}).click();
    await expect(page.locator("#job-message")).toContainText("prošel kontrolou");
    await card(page).getByRole("button",{name:"Zapnout",exact:true}).click();
    await expect(card(page).getByText("Zapnuto",{exact:true})).toBeVisible();
    await expect(page.locator('[data-page="tuned"]')).toHaveAttribute("aria-disabled","false");
    await page.locator("#experimental-addons").scrollIntoViewIfNeeded();
    await page.screenshot({path:"test-results/addons/addons-cs.png",fullPage:true});
  });
  test("an enabled package does not unlock another model",async({page})=>{
    await page.goto("/sprava-modelu");
    await page.locator('[data-model-id="other"]').getByRole("button",{name:"Použít",exact:true}).click();
    await expect(page.locator('[data-model-id="other"]')).toContainText("Právě načtený");
    await expect(page.locator('[data-page="tuned"]')).toHaveAttribute("aria-disabled","true");
    await expect(page.locator('[data-page="tuned"]')).toHaveAttribute("title",/není kompatibilní/);
    await expect(card(page)).toContainText("Pro aktivní model není tento doplněk dostupný.");
  });
  test("changed files disable the tool; verification reports a checksum failure",async({page})=>{
    const weights=path.join(root,"translators.safetensors");original=fs.readFileSync(weights);fs.appendFileSync(weights,"changed");
    await page.goto("/sprava-modelu#experimental-addons");
    await expect(card(page)).toContainText("Čeká na ověření");
    await card(page).getByRole("button",{name:"Ověřit balíček",exact:true}).click();
    await expect(page.locator("#job-message")).toContainText("neprošel kontrolou");
    fs.writeFileSync(weights,original);
    await card(page).getByRole("button",{name:"Ověřit balíček",exact:true}).click();
    await expect(page.locator("#job-message")).toContainText("prošel kontrolou");
  });
  test("English controls and disconnect preserve the original files",async({page})=>{
    await page.goto("/sprava-modelu?lang=en#experimental-addons");
    await expect(page.locator("#addons-title")).toHaveText("Experimental addons");
    await expect(card(page).getByRole("button",{name:"Verify package",exact:true})).toBeEnabled();
    await expect(card(page)).not.toContainText("Ověřit balíček");
    await page.locator("#experimental-addons").scrollIntoViewIfNeeded();
    await page.screenshot({path:"test-results/addons/addons-en.png",fullPage:true});
    await card(page).getByRole("button",{name:"Disconnect",exact:true}).click();
    await expect(page.locator("#disconnect-title")).toHaveText("Disconnect addon?");
    await page.locator("#confirm-disconnect").click();
    await expect(card(page)).toContainText("Not connected");
    expect(fs.readFileSync(path.join(root,"translators.safetensors"))).toEqual(original);
    await page.setViewportSize({width:390,height:844});await page.reload();
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
  });
});
