import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  buildServerArguments,
  normalizeWorkbenchURL,
  fetchWorkbenchHealth,
  candidateApertusRoots,
  coerceApertusRoot,
  isValidHealthPayload,
  validateApertusRoot,
  isModelManagerSender,
} from "../lib/runtime.mjs";

function makeModel(root, name, shard = "model-00001-of-00001.safetensors") {
  const directory = path.join(root, name);
  fs.mkdirSync(directory, { recursive: true });
  fs.writeFileSync(path.join(directory, "config.json"), "{}\n");
  fs.writeFileSync(path.join(directory, "tokenizer.json"), "{}\n");
  fs.writeFileSync(path.join(directory, shard), "weights");
  fs.writeFileSync(
    path.join(directory, "model.safetensors.index.json"),
    `${JSON.stringify({ weight_map: { tensor: shard } })}\n`,
  );
}

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "apertus-desktop-test-"));
  const garden = path.join(root, "apertus", "garden");
  fs.mkdirSync(path.join(garden, "static"), { recursive: true });
  fs.writeFileSync(path.join(garden, "server.py"), "# server\n");
  fs.writeFileSync(path.join(garden, "static", "index.html"), "ok\n");
  fs.writeFileSync(path.join(garden, "model_presets.toml"), "schema_version = 1\n");
  fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\n");
  return fs.realpathSync(root);
}

test("native model folder picker accepts only the current model-manager origin and route", () => {
  const base = "http://127.0.0.1:8766";
  for (const route of ["/", "/sprava-modelu", "/sprava-modelu/?lang=en", "/model-admin"]) {
    assert.equal(isModelManagerSender(base + route, base), true);
  }
  for (const url of ["http://127.0.0.1:8765/sprava-modelu", "https://example.org/sprava-modelu", "file:///tmp/sprava-modelu", base+"/kontrast", "not a URL"]) {
    assert.equal(isModelManagerSender(url, base), false);
  }
});

test("first-run registry is found in user data without modifying the bundled preset", (t) => {
  const root = fixture();
  const oldData = process.env.APERTUS_DATA_DIR;
  const oldRegistry = process.env.APERTUS_MODEL_REGISTRY;
  const data = path.join(root, "test-user-data");
  fs.mkdirSync(data);
  fs.writeFileSync(path.join(data,"models.toml"), "# created through model management\n");
  process.env.APERTUS_DATA_DIR = data;
  delete process.env.APERTUS_MODEL_REGISTRY;
  t.after(() => {
    if (oldData === undefined) delete process.env.APERTUS_DATA_DIR; else process.env.APERTUS_DATA_DIR = oldData;
    if (oldRegistry === undefined) delete process.env.APERTUS_MODEL_REGISTRY; else process.env.APERTUS_MODEL_REGISTRY = oldRegistry;
    fs.rmSync(root, {recursive:true,force:true});
  });
  assert.equal(validateApertusRoot(root).registryPath, path.join(data,"models.toml"));
  fs.writeFileSync(path.join(root,"models.toml"), "# explicit checkout registry\n");
  assert.equal(validateApertusRoot(root).registryPath, path.join(root,"models.toml"));
  assert.equal(fs.readFileSync(path.join(root,"apertus/garden/model_presets.toml"),"utf8"), "schema_version = 1\n");
});

test("normalizes a remote workbench URL and rejects credentials and paths", () => {
  assert.equal(normalizeWorkbenchURL("http://127.0.0.1:8780/"), "http://127.0.0.1:8780");
  for (const value of ["file:///tmp/app", "https://user:secret@example.org", "https://example.org/path", "https://example.org/?token=secret"]) {
    assert.throws(() => normalizeWorkbenchURL(value));
  }
});

test("checks a remote server without requiring local Python or weights", async (t) => {
  const server = http.createServer((request, response) => {
    assert.equal(request.url, "/api/health");
    response.setHeader("Content-Type", "application/json");
    response.end(JSON.stringify({server_instance_id:"remote", model:"EuroLLM 9B", status:"ready"}));
  });
  await new Promise((resolve) => server.listen(0,"127.0.0.1",resolve));
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const health = await fetchWorkbenchHealth(`http://127.0.0.1:${server.address().port}`);
  assert.equal(health.server_instance_id, "remote");
});

test("validates source independently of optional external weights", (t) => {
  const root = fixture();
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const result = validateApertusRoot(root);
  assert.equal(result.ok, true);
  assert.equal(result.root, root);
  assert.equal(result.registryPath, path.join(root, "apertus", "garden", "model_presets.toml"));
  assert.notEqual(result.dataDir, path.join(root, "data"));
});

test("reports an incomplete source checkout before starting Python", (t) => {
  const root = fixture();
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  fs.rmSync(path.join(root, "pyproject.toml"));
  const result = validateApertusRoot(root);
  assert.equal(result.ok, false);
  assert.match(result.problems.join("\n"), /pyproject.toml/u);
});

test("accepts selecting the garden directory instead of its parent", (t) => {
  const root = fixture();
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  assert.equal(coerceApertusRoot(path.join(root, "apertus", "garden")), root);
});

test("discovers the Apertus root above the desktop project", (t) => {
  const root = fixture();
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const appPath = path.join(root, "apertus", "garden", "desktop");
  fs.mkdirSync(appPath);
  const candidates = candidateApertusRoots({
    appPath,
    cwd: appPath,
    homeDir: path.join(root, "unused-home"),
  });
  assert.ok(candidates.includes(root));
});

test("server arguments delegate model choice to the registry and use user data", (t) => {
  const root = fixture();
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const validation = validateApertusRoot(root);
  const args = buildServerArguments({ validation, port: 8811 });
  assert.deepEqual(args.slice(0, 4), ["-u", "-m", "apertus.garden", "serve"]);
  assert.ok(args.includes("8811"));
  assert.ok(args.includes(validation.registryPath));
  assert.ok(args.includes(validation.dataDir));
  assert.ok(!args.includes("--model"));
  assert.ok(!args.includes("--initial-model"));
  assert.ok(buildServerArguments({validation, initialModel:"custom-id"}).includes("custom-id"));
});

test("accepts the current Apertus health contract", () => {
  assert.equal(
    isValidHealthPayload({
      status: "ready",
      server_instance_id: "fixture-server",
      model: "Apertus 8B (mock)",
      active_model: { id: "mock", label: "Vývojová simulace" },
    }),
    true,
  );
  assert.equal(
    isValidHealthPayload({
      server_instance_id: "legacy-server",
      model: { id: "hybrid" },
    }),
    true,
  );
});

test("rejects unrelated services on the preferred port", () => {
  assert.equal(isValidHealthPayload({ status: "ok" }), false);
  assert.equal(
    isValidHealthPayload({ server_instance_id: "x", model: "Apertus" }, 503),
    false,
  );
});
