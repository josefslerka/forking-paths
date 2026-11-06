import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  Menu,
  nativeTheme,
  session,
  shell,
} from "electron";
import { execFile, execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  baseURL,
  normalizeWorkbenchURL,
  fetchWorkbenchHealth,
  buildServerArguments,
  choosePort,
  coerceApertusRoot,
  DEFAULT_HOST,
  DEFAULT_MODEL,
  DEFAULT_PORT,
  fetchHealth,
  pythonCandidates,
  resolveApertusRoot,
  validateApertusRoot,
  waitForHealth,
  isModelManagerSender,
} from "./lib/runtime.mjs";

const moduleDir = path.dirname(fileURLToPath(import.meta.url));
const splashPath = path.join(moduleDir, "shell.html");

let mainWindow = null;
let serverProcess = null;
let ownsServer = false;
let currentBaseURL = null;
let currentValidation = null;
let currentPython = null;
let quittingAfterShutdown = false;
let stoppingServer = false;
let startupRun = 0;
let logStream = null;
let logPath = null;
let statusHistory = [];

app.setName("Apertus Workbench");
app.enableSandbox();
nativeTheme.themeSource = "light";

const hasLock = app.requestSingleInstanceLock();
if (!hasLock) app.quit();

function settingsPath() {
  return path.join(app.getPath("userData"), "settings.json");
}

function loadSettings() {
  try {
    const parsed = JSON.parse(fs.readFileSync(settingsPath(), "utf8"));
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function saveSettings(next) {
  const target = settingsPath();
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const temporary = `${target}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(next, null, 2)}\n`, "utf8");
  fs.renameSync(temporary, target);
}

function ensureLog() {
  if (logStream) return;
  const directory = path.join(app.getPath("userData"), "logs");
  fs.mkdirSync(directory, { recursive: true });
  logPath = path.join(directory, "apertus-workbench.log");
  logStream = fs.createWriteStream(logPath, { flags: "a" });
  logStream.write(`\n=== Desktop launch ${new Date().toISOString()} ===\n`);
}

function log(message) {
  ensureLog();
  const line = String(message).replace(/\s+$/u, "");
  logStream.write(`${line}\n`);
}

function publishStatus(payload) {
  const entry = {
    phase: payload.phase ?? "starting",
    title: payload.title ?? "Apertus Workbench",
    message: payload.message ?? "",
    details: Array.isArray(payload.details) ? payload.details : [],
  };
  statusHistory = [...statusHistory.slice(-7), entry];
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("desktop-status", entry);
  }
  log(`[${entry.phase}] ${entry.title}: ${entry.message}`);
  for (const detail of entry.details) log(`  ${detail}`);
}

function isSplashSender(event) {
  try {
    const url = new URL(event.senderFrame.url);
    return url.protocol === "file:" && fileURLToPath(url) === splashPath;
  } catch {
    return false;
  }
}

function isAllowedWorkbenchURL(target) {
  if (!currentBaseURL) return false;
  try {
    const url = new URL(target);
    return url.origin === new URL(currentBaseURL).origin;
  } catch {
    return false;
  }
}

function protectWebContents(contents) {
  contents.setWindowOpenHandler(({ url }) => {
    if (isAllowedWorkbenchURL(url)) return { action: "allow" };
    if (/^https?:\/\//u.test(url)) void shell.openExternal(url);
    return { action: "deny" };
  });
  contents.on("will-navigate", (event, url) => {
    let isSplash = false;
    try {
      const parsed = new URL(url);
      isSplash = parsed.protocol === "file:" && fileURLToPath(parsed) === splashPath;
    } catch {
      isSplash = false;
    }
    if (isSplash || isAllowedWorkbenchURL(url)) return;
    event.preventDefault();
    if (/^https?:\/\//u.test(url)) void shell.openExternal(url);
  });
  contents.on("context-menu", (_event, params) => {
    const template = [];
    if (params.isEditable) {
      template.push(
        { label: "Zpět", role: "undo", enabled: params.editFlags.canUndo },
        { label: "Znovu", role: "redo", enabled: params.editFlags.canRedo },
        { type: "separator" },
        { label: "Vyjmout", role: "cut", enabled: params.editFlags.canCut },
        { label: "Kopírovat", role: "copy", enabled: params.editFlags.canCopy },
        { label: "Vložit", role: "paste", enabled: params.editFlags.canPaste },
        { label: "Smazat", role: "delete", enabled: params.editFlags.canDelete },
        { type: "separator" },
        { label: "Vybrat vše", role: "selectAll" },
      );
    } else if (params.selectionText) {
      template.push({ label: "Kopírovat", role: "copy" });
    }
    if (template.length === 0) return;
    Menu.buildFromTemplate(template).popup({ window: mainWindow });
  });
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1040,
    minHeight: 700,
    backgroundColor: "#f5f2e9",
    show: false,
    title: "Apertus Workbench",
    webPreferences: {
      preload: path.join(moduleDir, "preload.cjs"),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      webSecurity: true,
    },
  });
  protectWebContents(mainWindow.webContents);
  mainWindow.once("ready-to-show", () => mainWindow?.show());
  await mainWindow.loadFile(splashPath);
  mainWindow.webContents.send("desktop-status-history", statusHistory);
}

function executablePath(candidate) {
  if (!candidate) return null;
  if (path.isAbsolute(candidate)) {
    try {
      fs.accessSync(candidate, fs.constants.X_OK);
      return candidate;
    } catch {
      return null;
    }
  }
  try {
    const resolved = execFileSync("/usr/bin/which", [candidate], {
      encoding: "utf8",
      timeout: 2_000,
    }).trim();
    return resolved || null;
  } catch {
    return null;
  }
}

function resolvePython(settings) {
  for (const candidate of pythonCandidates({
    explicitPython: process.env.APERTUS_PYTHON,
    savedPython: settings.pythonPath,
    root: currentValidation?.root,
  })) {
    const resolved = executablePath(candidate);
    if (resolved) return resolved;
  }
  return null;
}

function runPreflight(python, validation) {
  return new Promise((resolve, reject) => {
    execFile(
      python,
      ["-m", "apertus.garden", "models", "list", "--registry", validation.registryPath],
      {
        cwd: validation.root,
        encoding: "utf8",
        timeout: 45_000,
        maxBuffer: 2 * 1024 * 1024,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      },
      (error, stdout, stderr) => {
        if (stdout) log(stdout);
        if (stderr) log(stderr);
        if (error) {
          reject(new Error((stderr || stdout || error.message).trim()));
          return;
        }
        resolve(stdout);
      },
    );
  });
}

function pipeServerOutput(child) {
  for (const [stream, label] of [
    [child.stdout, "server"],
    [child.stderr, "server:error"],
  ]) {
    let buffered = "";
    stream?.setEncoding("utf8");
    stream?.on("data", (chunk) => {
      buffered += chunk;
      const lines = buffered.split(/\r?\n/u);
      buffered = lines.pop() ?? "";
      for (const line of lines) if (line) log(`[${label}] ${line}`);
    });
    stream?.on("end", () => {
      if (buffered) log(`[${label}] ${buffered}`);
    });
  }
}

function spawnServer(python, validation, port) {
  fs.mkdirSync(validation.dataDir, { recursive: true });
  const child = spawn(
    python,
    buildServerArguments({ validation, port }),
    {
      cwd: validation.root,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  pipeServerOutput(child);
  child.once("error", (error) => log(`[server:spawn] ${error.stack ?? error}`));
  child.once("exit", (code, signal) => {
    log(`[server:exit] code=${code} signal=${signal}`);
    if (quittingAfterShutdown || stoppingServer || !ownsServer) return;
    serverProcess = null;
    ownsServer = false;
    currentBaseURL = null;
    void mainWindow?.loadFile(splashPath).then(() => {
      publishStatus({
        phase: "error",
        title: "Server se zastavil",
        message: "Lokální proces Apertusu neočekávaně skončil.",
        details: [`Kód: ${code ?? "—"}`, `Signál: ${signal ?? "—"}`],
      });
    });
  });
  return child;
}

async function stopOwnedServer() {
  const child = serverProcess;
  if (!child || !ownsServer || child.exitCode !== null) {
    serverProcess = null;
    ownsServer = false;
    return;
  }
  stoppingServer = true;
  try {
    await new Promise((resolve) => {
      let killTimer = null;
      const terminateTimer = setTimeout(() => {
        if (child.exitCode === null) child.kill("SIGTERM");
        killTimer = setTimeout(() => {
          if (child.exitCode === null) child.kill("SIGKILL");
        }, 5_000);
        killTimer.unref();
      }, 10_000);
      terminateTimer.unref();
      child.once("exit", () => {
        clearTimeout(terminateTimer);
        if (killTimer) clearTimeout(killTimer);
        resolve();
      });
      child.kill("SIGINT");
    });
  } finally {
    serverProcess = null;
    ownsServer = false;
    currentBaseURL = null;
    stoppingServer = false;
  }
}

async function loadWorkbench(url) {
  currentBaseURL = url;
  publishStatus({
    phase: "ready",
    title: "Apertus je připraven",
    message: "Otevírám lokální workbench.",
  });
  await mainWindow.loadURL(`${url}/`);
}

async function startWorkbench({ forceRoot } = {}) {
  const run = ++startupRun;
  try {
    if (!mainWindow || mainWindow.isDestroyed()) await createWindow();
    if (!mainWindow.webContents.getURL().startsWith("file:")) {
      await mainWindow.loadFile(splashPath);
    }
    publishStatus({
      phase: "starting",
      title: "Spouštím Apertus",
      message: "Hledám modely a kontroluji lokální prostředí.",
    });

    const serverAddress = process.env.APERTUS_WORKBENCH_URL;
    if (serverAddress) {
      const url = normalizeWorkbenchURL(serverAddress);
      const health = await fetchWorkbenchHealth(url);
      if (run !== startupRun) return;
      if (!health) throw new Error("Výpočetní server neodpovídá jako platný workbench.");
      ownsServer = false;
      currentValidation = null;
      currentPython = null;
      await loadWorkbench(url);
      return;
    }

    const existing = await fetchHealth(DEFAULT_HOST, DEFAULT_PORT);
    if (run !== startupRun) return;
    if (existing) {
      ownsServer = false;
      currentValidation = null;
      currentPython = null;
      await loadWorkbench(baseURL(DEFAULT_HOST, DEFAULT_PORT));
      return;
    }

    const settings = loadSettings();
    const resolution = forceRoot
      ? { match: validateApertusRoot(forceRoot), attempts: [] }
      : resolveApertusRoot({
          explicitRoot: process.env.APERTUS_ROOT,
          savedRoot: settings.apertusRoot,
          appPath: app.getAppPath(),
        });
    if (!resolution.match?.ok) {
      const details = resolution.attempts
        .slice(0, 4)
        .flatMap((attempt) =>
          attempt.root
            ? [`${attempt.root}: ${attempt.problems.slice(0, 2).join(", ")}`]
            : [],
        );
      throw Object.assign(
        new Error("Vyberte repozitář apertus-workbench se souborem pyproject.toml."),
        { code: "root_missing", details },
      );
    }
    currentValidation = resolution.match;
    saveSettings({ ...settings, apertusRoot: currentValidation.root });
    publishStatus({
      phase: "starting",
      title: "Kontroluji prostředí",
      message: currentValidation.root,
    });

    currentPython = resolvePython(settings);
    if (!currentPython) {
      throw Object.assign(
        new Error("Nenašel jsem Python. V repozitáři vytvořte .venv podle instalačního návodu."),
        { code: "python_missing" },
      );
    }
    await runPreflight(currentPython, currentValidation);
    if (run !== startupRun) return;

    const port = await choosePort(DEFAULT_HOST, DEFAULT_PORT);
    publishStatus({
      phase: "starting",
      title: "Spouštím lokální server",
      message: `Python ${currentPython} · port ${port}`,
    });
    serverProcess = spawnServer(currentPython, currentValidation, port);
    ownsServer = true;
    const health = await waitForHealth({
      host: DEFAULT_HOST,
      port,
      child: serverProcess,
    });
    if (run !== startupRun) return;
    log(`[health] instance=${health.server_instance_id}`);
    await loadWorkbench(baseURL(DEFAULT_HOST, port));
  } catch (error) {
    if (run !== startupRun) return;
    if (serverProcess && ownsServer) await stopOwnedServer();
    log(error.stack ?? error.message ?? String(error));
    publishStatus({
      phase: "error",
      title: "Apertus se nepodařilo spustit",
      message: error.message ?? String(error),
      details: error.details ?? [],
      code: error.code,
    });
  }
}

async function chooseRoot(event) {
  if (!isSplashSender(event)) throw new Error("Nedůvěryhodný požadavek.");
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Vyberte repozitář apertus-workbench",
    message: "Složka s pyproject.toml a models.toml. Váhy mohou být uložené jinde.",
    properties: ["openDirectory"],
  });
  if (result.canceled || result.filePaths.length !== 1) return { canceled: true };
  const root = coerceApertusRoot(result.filePaths[0]);
  const validation = validateApertusRoot(root);
  if (!validation.ok) {
    return {
      canceled: false,
      ok: false,
      problems: validation.problems,
    };
  }
  const settings = loadSettings();
  saveSettings({ ...settings, apertusRoot: validation.root });
  void startWorkbench({ forceRoot: validation.root });
  return { canceled: false, ok: true, root: validation.root };
}

function installIPC() {
  ipcMain.handle("desktop:choose-model-directory", async (event) => {
    if (!currentBaseURL || !isModelManagerSender(event.senderFrame.url, currentBaseURL)) throw new Error("Nedůvěryhodný požadavek.");
    const result = await dialog.showOpenDialog(mainWindow, {title:"Vyberte adresář s modelem", properties:["openDirectory"]});
    return result.canceled || result.filePaths.length !== 1 ? {canceled:true} : {canceled:false,path:result.filePaths[0]};
  });
  ipcMain.handle("desktop:choose-root", chooseRoot);
  ipcMain.handle("desktop:retry", async (event) => {
    if (!isSplashSender(event)) throw new Error("Nedůvěryhodný požadavek.");
    await stopOwnedServer();
    void startWorkbench();
    return { ok: true };
  });
  ipcMain.handle("desktop:open-log", async (event) => {
    if (!isSplashSender(event)) throw new Error("Nedůvěryhodný požadavek.");
    ensureLog();
    return shell.openPath(logPath);
  });
}

function installMenu() {
  Menu.setApplicationMenu(
    Menu.buildFromTemplate([
      {
        label: "Apertus Workbench",
        submenu: [
          { role: "about" },
          { type: "separator" },
          { role: "hide" },
          { role: "hideOthers" },
          { role: "unhide" },
          { type: "separator" },
          { role: "quit" },
        ],
      },
      {
        label: "Workbench",
        submenu: [
          {
            label: "Obnovit stránku",
            accelerator: "CmdOrCtrl+R",
            click: () => mainWindow?.webContents.reload(),
          },
          {
            label: "Znovu spustit lokální server",
            click: async () => {
              await stopOwnedServer();
              await mainWindow?.loadFile(splashPath);
              void startWorkbench();
            },
          },
          { type: "separator" },
          {
            label: "Otevřít ve výchozím prohlížeči",
            click: () => {
              if (currentBaseURL) void shell.openExternal(currentBaseURL);
            },
          },
          {
            label: "Zobrazit diagnostický log",
            click: () => {
              ensureLog();
              void shell.openPath(logPath);
            },
          },
        ],
      },
      {
        label: "Úpravy",
        submenu: [
          { label: "Zpět", role: "undo" },
          { label: "Znovu", role: "redo" },
          { type: "separator" },
          { label: "Vyjmout", role: "cut" },
          { label: "Kopírovat", role: "copy" },
          { label: "Vložit", role: "paste" },
          { label: "Vložit bez formátování", role: "pasteAndMatchStyle" },
          { label: "Smazat", role: "delete" },
          { type: "separator" },
          { label: "Vybrat vše", role: "selectAll" },
        ],
      },
      {
        label: "Zobrazení",
        submenu: [
          { role: "resetZoom" },
          { role: "zoomIn" },
          { role: "zoomOut" },
          { type: "separator" },
          { role: "togglefullscreen" },
        ],
      },
      {
        label: "Okno",
        submenu: [{ role: "minimize" }, { role: "zoom" }, { role: "front" }],
      },
    ]),
  );
}

app.on("second-instance", () => {
  if (!mainWindow) return;
  if (mainWindow.isMinimized()) mainWindow.restore();
  mainWindow.show();
  mainWindow.focus();
});

app.whenReady().then(async () => {
  ensureLog();
  session.defaultSession.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false));
  installIPC();
  installMenu();
  await createWindow();
  void startWorkbench();
});

app.on("window-all-closed", () => app.quit());

app.on("before-quit", (event) => {
  if (!ownsServer || !serverProcess || serverProcess.exitCode !== null) return;
  if (quittingAfterShutdown) return;
  event.preventDefault();
  quittingAfterShutdown = true;
  void stopOwnedServer().finally(() => app.quit());
});

process.on("uncaughtException", (error) => {
  log(`[desktop:uncaught] ${error.stack ?? error}`);
  publishStatus({
    phase: "error",
    title: "Neočekávaná chyba desktopové aplikace",
    message: error.message ?? String(error),
  });
});
