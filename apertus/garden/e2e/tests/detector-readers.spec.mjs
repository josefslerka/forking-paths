import {createHash} from "node:crypto";
import fs from "node:fs/promises";
import {expect, openWorkbench, test} from "./helpers.mjs";

const text = "Na výzkumném setkání jsme porovnávali několik způsobů, jak modely čtou český text. ".repeat(35).trim();
const hash = (value) => createHash("sha256").update(value).digest("hex");
const feature = "surprisal_bits_per_non_whitespace_character";

function result(identity, source, {status = "completed", pairs = 3} = {}) {
  const score = {score: .2, strict_threshold: .479, screening_threshold: .479,
    strict_ai_signal: false, screening_ai_signal: false, calibration: {roc_auc: .8, pair_count: 60}};
  const features = (density) => ({[feature]: density, token_surprisal_iqr: 2,
    token_surprisal_p95: 8, top_5pct_surprisal_share: .2, mean_surprisal_minus_entropy_bits: .4});
  const windows = ["early", "middle_1", "late"].map((position, index) => ({index: index + 1, position,
    status: index < pairs ? "paired" : "unmeasured", reason: index < pairs ? null : "measurement_incomplete",
    target_non_whitespace_character_count: 400, trimmed_apertus_target_tokens: 1,
    measurements: {apertus: {token_count: 127, features: features(1.2 + index * .1)},
      eurollm: {token_count: 110, features: features(1.4 + index * .1)}}, delta_bits_per_character: .2}));
  return {analysis_kind: "predictive_stylometry", schema_version: 2, analysis_id: "test-analysis", reader: identity,
    primary_decision: {model: "apertus", challenger_changes_verdict: false}, quality: {status: "pass", flags: []},
    scores: {primary: {models: {apertus: score, stylometry: score, fusion: score}}},
    challenger: {decision_authority: false, model: score, comparison: {status: "agreement"}},
    input: {sha256: hash(source), character_count: source.length}, apertus_features: {original: 42},
    sampling: {apertus: {windows: windows.map(w => ({index: w.index, position: w.position, target_token_count: 128})), scored_token_count: 384}},
    reader_comparison: {analysis_kind: "detector_reader_comparison", schema_version: 1, research_only: true,
      decision_authority: false, input_sha256: hash(source), status,
      protocol: {id: "detector-readers-v1", decision_authority: false}, protocol_sha256: "abc",
      models: {apertus: {id: "hybrid", label: "Apertus Hybrid"}, eurollm: {id: "eurollm9", label: "EuroLLM 9B · 6bit"}},
      windows, summary: {paired_windows: pairs, requested_windows: 3, target_non_whitespace_character_count: pairs * 400,
        measurements: {apertus: {features: features(1.3)}, eurollm: {features: features(1.5)}}, delta_bits_per_character: .2},
      errors: status === "failed" ? ["test failure"] : []}}
}

async function setup(page, {available = true, pause = false, status = "completed", pairs = 3, wrongHash = false} = {}) {
  const base = await (await page.request.get("/api/health")).json();
  let generation = 1, request, complete = !pause, cancelled = false;
  const identity = () => ({model_id: "hybrid", model_generation: generation, server_instance_id: base.server_instance_id});
  const health = () => ({...base, model_generation: generation,
    active_model: {...base.active_model, id: "hybrid", label: "Hybrid · 8bit/BF16"},
    available_models: [{id: "hybrid", label: "Hybrid · 8bit/BF16", available: true},
      {id: "eurollm9", label: "EuroLLM 9B · 6bit", available}], capabilities: [...base.capabilities, "detector"],
    worker: {...base.worker, busy: false}});
  await page.route("**/api/health", route => route.fulfill({json: health()}));
  await page.route("**/api/predictive-stylometry", async route => {
    const body = route.request().postDataJSON();
    const data = result(identity(), body.text);
    delete data.reader_comparison;
    await route.fulfill({json: data});
  });
  const job = () => {
    const captured = {model_id: request.model_id, model_generation: request.model_generation,
      server_instance_id: request.server_instance_id};
    const done = complete || cancelled;
    if (done) generation = 3;
    return {id: "test-reader-job", phase: done ? "done" : "eurollm", status: done ? cancelled ? "cancelled" : status : "running",
      input_sha256: wrongHash ? "wrong" : hash(request.text), request_identity: captured,
      completed_measurements: done ? 6 : 3, total_measurements: 6,
      restored_identity: done ? identity() : null, errors: [],
      result: done ? result(captured, request.text, {status: cancelled ? "cancelled" : status, pairs}) : null};
  };
  await page.route("**/api/detector-readers", async route => {
    request = route.request().postDataJSON();
    await route.fulfill({status: 202, json: {id: "test-reader-job", phase: "queued", status: "queued",
      completed_measurements: 0, total_measurements: 6}});
  });
  await page.route("**/api/detector-readers/jobs/*", route => route.fulfill({json: job()}));
  await page.route("**/api/detector-readers/cancel", async route => {
    cancelled = true;
    await route.fulfill({json: job()});
  });
  return {finish: () => {complete = true;}, getRequest: () => request};
}

test("detektor přidá výzkumné EuroLLM, zachová verdikt a exportuje bez textu", async ({appPage: page}) => {
  const controls = await setup(page);
  await openWorkbench(page, "/detektor");
  await expect(page.locator("#include-eurollm")).toBeEnabled();
  await expect(page.locator("#include-eurollm")).not.toBeChecked();
  await page.locator("#source-text").fill(text);
  await page.locator("#analyze-button").click();
  await expect(page.locator("#results")).toBeVisible();
  const verdict = await page.locator("#verdict").textContent();
  const score = await page.locator("#apertus-score").textContent();
  await expect(page.locator("#reader-comparison")).toBeHidden();
  await page.locator("#include-eurollm").check();
  await page.locator("#analyze-button").click();
  await expect(page.locator("#reader-comparison")).toBeVisible();
  await expect(page.locator("#reader-status")).toContainText("3 / 3");
  await expect(page.locator("#reader-chart rect.bar-apertus")).toHaveCount(4);
  await expect(page.locator("#reader-windows tr")).toHaveCount(3);
  await expect(page.locator("#verdict")).toHaveText(verdict);
  await expect(page.locator("#apertus-score")).toHaveText(score);
  await expect(page.locator("#results")).toBeVisible();
  expect(controls.getRequest().text).toBe(text);
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#reader-export").click();
  const downloaded = await downloadPromise;
  const source = await fs.readFile(await downloaded.path(), "utf8");
  const data = JSON.parse(source);
  expect(data.comparison.decision_authority).toBe(false);
  expect(data.apertus_detector.scores.models.apertus.score).toBe(.2);
  expect(data.input.sha256).toBe(hash(text));
  expect(source).not.toContain("Na výzkumném setkání");
  await page.locator("#reader-comparison").scrollIntoViewIfNeeded();
  await page.screenshot({path: "test-results/detector-readers-cs.png"});
});

test("úprava textu během měření nepřipojí opožděný výsledek", async ({appPage: page}) => {
  const controls = await setup(page, {pause: true});
  await openWorkbench(page, "/detektor");
  await page.locator("#source-text").fill(text);
  await page.locator("#include-eurollm").check();
  await page.locator("#analyze-button").click();
  await expect(page.locator("#cancel-analysis")).toBeEnabled();
  await page.locator("#source-text").fill(text + " Jiný text.");
  controls.finish();
  await expect(page.locator("#form-message")).toContainText("Text se během měření změnil");
  await expect(page.locator("#results")).toBeHidden();
});

test("přerušení vrátí Hybrid a zobrazí pouze dokončené úseky", async ({appPage: page}) => {
  await setup(page, {pause: true, pairs: 1});
  await openWorkbench(page, "/detektor");
  await page.locator("#source-text").fill(text);
  await page.locator("#include-eurollm").check();
  await page.locator("#analyze-button").click();
  await expect(page.locator("#cancel-analysis")).toBeEnabled();
  await page.locator("#cancel-analysis").click();
  await expect(page.locator("#reader-status")).toContainText("Měření přerušeno");
  await expect(page.locator("#reader-status")).toContainText("1 / 3");
  await expect(page.locator("#reader-windows")).toContainText("Měření úseku nebylo dokončeno");
  await expect(page.locator("#verdict")).toContainText("Přísný AI signál nezachycen");
});

test("chybějící EuroLLM ponechá běžnou analýzu dostupnou", async ({appPage: page}) => {
  await setup(page, {available: false});
  await openWorkbench(page, "/detektor");
  await expect(page.locator("#include-eurollm")).toBeDisabled();
  await page.locator("#source-text").fill(text);
  await page.locator("#analyze-button").click();
  await expect(page.locator("#results")).toBeVisible();
  await expect(page.locator("#reader-comparison")).toBeHidden();
});

test("měření jiného dokumentu se odmítne", async ({appPage: page}) => {
  await setup(page, {wrongHash: true});
  await openWorkbench(page, "/detektor");
  await page.locator("#source-text").fill(text);
  await page.locator("#include-eurollm").check();
  await page.locator("#analyze-button").click();
  await expect(page.locator("#form-message")).toContainText("Měření nepatří k tomuto textu");
  await expect(page.locator("#results")).toBeHidden();
});

test("anglické měření přeloží graf i výsledky a zachová český vstup", async ({appPage: page}) => {
  await setup(page);
  await openWorkbench(page, "/detektor?lang=en");
  await expect(page.locator("#reader-option-copy")).toContainText("EuroLLM has no calibration");
  await page.locator("#source-text").fill(text);
  await page.locator("#include-eurollm").check();
  await page.locator("#analyze-button").click();
  await expect(page.locator("#reader-title")).toHaveText("Apertus and EuroLLM reading the same text");
  await expect(page.locator("#reader-status")).toContainText("Measurement complete.");
  await expect(page.locator("#reader-windows")).toContainText("Beginning");
  await expect(page.locator("#reader-alignment")).toContainText("Total boundary adjustment:");
  await expect(page.locator("#source-text")).toHaveValue(text);
  await page.locator("#reader-comparison details summary").click();
  await expect(page.locator("#reader-features")).toContainText("Mean token surprisal − entropy");
});
