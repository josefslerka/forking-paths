import fs from "node:fs";
import http from "node:http";
import https from "node:https";
import net from "node:net";
import os from "node:os";
import path from "node:path";

export const DEFAULT_HOST = "127.0.0.1";
export const DEFAULT_PORT = 8765;
export const DEFAULT_MODEL = null; // Registry default, not a hard-coded model ID.
export const HEALTH_PATH = "/api/health";

// A display-name change must not create an empty Electron profile. Keep the
// established directory (settings, cookies, language and saved source path).
export function desktopUserDataPath(appData) {
  return path.join(appData, "Apertus Workbench");
}

export function isModelManagerSender(url, base) {
  try {
    const parsed = new URL(url);
    return parsed.origin === new URL(base).origin && ["/", "/sprava-modelu", "/sprava-modelu/", "/model-admin"].includes(parsed.pathname);
  } catch { return false; }
}

export function normalizeWorkbenchURL(value) {
  const url = new URL(value);
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password ||
      url.search || url.hash || !["", "/"].includes(url.pathname)) {
    throw new Error("Adresa workbenche musí být kořenová HTTP(S) adresa bez přihlašovacích údajů.");
  }
  return url.origin;
}

export function fetchWorkbenchHealth(base, timeoutMs = 5000) {
  const url = new URL(HEALTH_PATH, normalizeWorkbenchURL(base));
  return new Promise((resolve) => {
    const transport = url.protocol === "https:" ? https : http;
    const request = transport.get(url, (response) => {
      if (response.statusCode !== 200) {response.resume(); resolve(null); return;}
      let data = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => {data += chunk; if (data.length > 1_000_000) request.destroy();});
      response.on("end", () => {try {const payload=JSON.parse(data);resolve(isValidHealthPayload(payload) ? payload : null);} catch {resolve(null);}});
      response.on("error", () => resolve(null));
    });
    request.setTimeout(timeoutMs, () => request.destroy());
    request.on("error", () => resolve(null));
  });
}

function existingDirectory(candidate) {
  if (!candidate || typeof candidate !== "string") return null;
  try {
    const resolved = fs.realpathSync(path.resolve(candidate));
    return fs.statSync(resolved).isDirectory() ? resolved : null;
  } catch {
    return null;
  }
}

function uniquePaths(candidates) {
  const seen = new Set();
  const output = [];
  for (const candidate of candidates) {
    const resolved = existingDirectory(candidate);
    if (!resolved || seen.has(resolved)) continue;
    seen.add(resolved);
    output.push(resolved);
  }
  return output;
}

function ancestorCandidates(start, depth = 8) {
  const output = [];
  let current = existingDirectory(start);
  for (let index = 0; current && index < depth; index += 1) {
    output.push(current);
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return output;
}

export function candidateApertusRoots({
  explicitRoot,
  savedRoot,
  appPath,
  cwd = process.cwd(),
} = {}) {
  const fromApp = ancestorCandidates(appPath);
  const fromCwd = ancestorCandidates(cwd);
  return uniquePaths([
    explicitRoot,
    savedRoot,
    ...fromApp,
    ...fromCwd,
  ]);
}

export function coerceApertusRoot(selectedPath) {
  const selected = existingDirectory(selectedPath);
  if (!selected) return null;
  const candidates = uniquePaths([
    selected,
    path.dirname(selected),
    path.dirname(path.dirname(selected)),
  ]);
  return candidates.find((candidate) =>
    fs.existsSync(path.join(candidate, "apertus", "garden", "server.py"))
  ) ?? null;
}

function validateModelDirectory(modelPath) {
  const problems = [];
  for (const filename of [
    "config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
  ]) {
    if (!fs.existsSync(path.join(modelPath, filename))) {
      problems.push(`chybí ${path.basename(modelPath)}/${filename}`);
    }
  }
  const indexPath = path.join(modelPath, "model.safetensors.index.json");
  if (!fs.existsSync(indexPath)) return problems;
  try {
    const parsed = JSON.parse(fs.readFileSync(indexPath, "utf8"));
    const shards = [...new Set(Object.values(parsed.weight_map ?? {}))]
      .filter((item) => typeof item === "string");
    if (shards.length === 0) {
      problems.push(`${path.basename(modelPath)} nemá v indexu žádné shardy`);
    }
    for (const shard of shards) {
      const shardPath = path.join(modelPath, shard);
      if (!fs.existsSync(shardPath) || fs.statSync(shardPath).size === 0) {
        problems.push(`chybí ${path.basename(modelPath)}/${shard}`);
      }
    }
  } catch (error) {
    problems.push(
      `${path.basename(modelPath)} má nečitelný index: ${error.message}`,
    );
  }
  return problems;
}

export function validateApertusRoot(candidate) {
  const root = existingDirectory(candidate);
  if (!root) {
    return { ok: false, root: null, problems: ["adresář neexistuje"] };
  }
  const gardenDir = path.join(root, "apertus", "garden");
  const problems = [];
  for (const relativePath of [
    "pyproject.toml",
    "apertus/garden/server.py",
    "apertus/garden/static/index.html",
    "apertus/garden/model_presets.toml",
  ]) {
    if (!fs.existsSync(path.join(root, relativePath))) {
      problems.push(`chybí ${relativePath}`);
    }
  }
  const userData = process.platform === "darwin" ? path.join(os.homedir(), "Library", "Application Support", "Apertus Workbench") :
    process.platform === "win32" ? path.join(process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local"), "Apertus Workbench") :
    path.join(process.env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share"), "apertus-workbench");
  const dataDir = process.env.APERTUS_DATA_DIR || userData;
  const registryPath = process.env.APERTUS_MODEL_REGISTRY || (
    fs.existsSync(path.join(root, "models.toml")) ? path.join(root, "models.toml") :
    fs.existsSync(path.join(dataDir, "models.toml")) ? path.join(dataDir, "models.toml") : path.join(gardenDir, "model_presets.toml")
  );
  return {
    ok: problems.length === 0,
    root,
    gardenDir,
    registryPath: path.resolve(root, registryPath),
    dataDir,
    problems,
  };
}

export function resolveApertusRoot(options = {}) {
  const attempts = candidateApertusRoots(options).map((candidate) =>
    validateApertusRoot(candidate),
  );
  return {
    match: attempts.find((attempt) => attempt.ok) ?? null,
    attempts,
  };
}

export function pythonCandidates({ explicitPython, savedPython, root, env = process.env } = {}) {
  return [
    explicitPython,
    savedPython,
    env.GARDEN_PYTHON,
    root ? path.join(root, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python") : null,
    "/opt/homebrew/bin/python3",
    "/usr/local/bin/python3",
    "/usr/bin/python3",
    "python3",
  ].filter(Boolean);
}

export function buildServerArguments({
  validation,
  host = DEFAULT_HOST,
  port = DEFAULT_PORT,
  initialModel = DEFAULT_MODEL,
}) {
  const { dataDir, registryPath } = validation;
  return [
    "-u",
    "-m", "apertus.garden", "serve",
    "--host",
    host,
    "--port",
    String(port),
    ...(initialModel ? ["--initial-model", initialModel] : []),
    "--registry", registryPath,
    "--data-dir", dataDir,
  ];
}

export function healthURL(host = DEFAULT_HOST, port = DEFAULT_PORT) {
  return `http://${host}:${port}${HEALTH_PATH}`;
}

export function baseURL(host = DEFAULT_HOST, port = DEFAULT_PORT) {
  return `http://${host}:${port}`;
}

export function isValidHealthPayload(payload, statusCode = 200) {
  const modelIsNamed = typeof payload?.model === "string" && payload.model.length > 0;
  const modelIsStructured =
    payload?.model && typeof payload.model === "object" && !Array.isArray(payload.model);
  return (
    statusCode === 200 &&
    payload &&
    typeof payload === "object" &&
    typeof payload.server_instance_id === "string" &&
    payload.server_instance_id.length > 0 &&
    (modelIsNamed || modelIsStructured)
  );
}

export function fetchHealth(host = DEFAULT_HOST, port = DEFAULT_PORT, timeoutMs = 900) {
  return new Promise((resolve) => {
    const request = http.get(healthURL(host, port), {
      headers: { Accept: "application/json" },
      timeout: timeoutMs,
    });
    request.once("response", (response) => {
      const chunks = [];
      response.on("data", (chunk) => chunks.push(chunk));
      response.once("end", () => {
        try {
          const payload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
          resolve(isValidHealthPayload(payload, response.statusCode) ? payload : null);
        } catch {
          resolve(null);
        }
      });
    });
    request.once("timeout", () => request.destroy());
    request.once("error", () => resolve(null));
  });
}

export function isPortAvailable(host, port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.unref();
    server.once("error", () => resolve(false));
    server.listen({ host, port, exclusive: true }, () => {
      server.close(() => resolve(true));
    });
  });
}

export async function choosePort(
  host = DEFAULT_HOST,
  preferredPort = DEFAULT_PORT,
  attempts = 20,
) {
  for (let offset = 0; offset < attempts; offset += 1) {
    const port = preferredPort + offset;
    if (await isPortAvailable(host, port)) return port;
  }
  throw new Error(
    `Nenašel jsem volný lokální port v rozsahu ${preferredPort}–${preferredPort + attempts - 1}.`,
  );
}

export async function waitForHealth({
  host = DEFAULT_HOST,
  port,
  child,
  timeoutMs = 60_000,
  intervalMs = 250,
}) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (child && child.exitCode !== null) {
      throw new Error(`Server skončil během startu s kódem ${child.exitCode}.`);
    }
    const health = await fetchHealth(host, port);
    if (health) return health;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error(`Server se do ${Math.round(timeoutMs / 1000)} sekund neohlásil.`);
}
