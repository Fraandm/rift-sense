/* Lanzador de escritorio: servidor Python + ventana Chromium (Electron). */
const { app, BrowserWindow, Menu, dialog, shell } = require("electron");
const { spawn, execFile } = require("child_process");
const path = require("path");
const fs = require("fs");
const http = require("http");
const net = require("net");
const os = require("os");
const crypto = require("crypto");

const DEFAULT_PORT = 8000;
const FALLBACK_PORTS = Array.from({ length: 21 }, (_, i) => 8765 + i);
const BOOT_LOG = path.join(os.tmpdir(), "riftsense-boot.log");
const UPDATE_REPO = process.env.RIFTSENSE_UPDATE_REPO || "Fraandm/rift-sense";
const GH_BIN =
  process.env.RIFTSENSE_GH || (process.platform === "win32" ? "gh.exe" : "gh");

function bootLog(message) {
  try {
    fs.appendFileSync(
      BOOT_LOG,
      `[${new Date().toISOString()}] ${message}\n`,
      "utf8"
    );
  } catch (_) {
    // sin permisos
  }
}

bootLog(
  `arranque portableDir=${process.env.PORTABLE_EXECUTABLE_DIR || ""} ` +
    `portableFile=${process.env.PORTABLE_EXECUTABLE_FILE || ""} ` +
    `exec=${process.execPath} cwd=${process.cwd()} dirname=${__dirname}`
);

let mainWindow = null;
let serverProcess = null;
let ownsServer = false;
let serverPort = null;
let quitting = false;
let projectRoot = null;
let updateInProgress = false;

app.setAppUserModelId("com.fraandm.riftsense");

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (!mainWindow) return;
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  });
}

/* ------------------------------------------------------------------ */
/* Rutas                                                               */
/* ------------------------------------------------------------------ */
function uniqueRoots(values) {
  const seen = new Set();
  const result = [];
  for (const value of values) {
    if (!value) continue;
    try {
      const resolved = path.resolve(value);
      if (!seen.has(resolved.toLowerCase())) {
        seen.add(resolved.toLowerCase());
        result.push(resolved);
      }
    } catch (_) {
      // ruta invalida
    }
  }
  return result;
}

function rootCandidates() {
  const candidates = [];
  const add = (value) => {
    if (value) candidates.push(value);
  };

  add(process.env.RIFTSENSE_ROOT);
  add(process.env.PORTABLE_EXECUTABLE_DIR);
  try {
    const embedded = JSON.parse(
      fs.readFileSync(path.join(__dirname, "root.json"), "utf8")
    );
    add(embedded.root);
  } catch (_) {
    // build sin root.json
  }
  add(process.cwd());

  let dir = path.dirname(process.execPath);
  for (let i = 0; i < 7; i += 1) {
    add(dir);
    dir = path.dirname(dir);
  }

  add(path.resolve(__dirname, ".."));

  let appDir = app.getAppPath();
  for (let i = 0; i < 7; i += 1) {
    add(appDir);
    appDir = path.dirname(appDir);
  }

  const base = process.env.PORTABLE_EXECUTABLE_DIR || path.dirname(process.execPath);
  for (const name of ["rift-sense.config.json", "RiftSense.config.json"]) {
    try {
      const configPath = path.join(base, name);
      if (!fs.existsSync(configPath)) continue;
      const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
      if (config.root) add(path.resolve(base, config.root));
    } catch (_) {
      // config invalida
    }
  }

  return uniqueRoots(candidates);
}

function isProjectRoot(root) {
  return (
    fs.existsSync(path.join(root, ".venv", "Scripts", "python.exe")) &&
    fs.existsSync(path.join(root, "lolanalyzer", "api", "server.py")) &&
    fs.existsSync(path.join(root, "web", "index.html"))
  );
}

function findProjectRoot() {
  if (process.env.RIFTSENSE_PYTHON) {
    const python = process.env.RIFTSENSE_PYTHON;
    if (fs.existsSync(python)) {
      let dir = path.dirname(path.dirname(python));
      while (dir && dir !== path.dirname(dir)) {
        if (isProjectRoot(dir)) return dir;
        dir = path.dirname(dir);
      }
    }
  }
  for (const root of rootCandidates()) {
    try {
      if (isProjectRoot(root)) {
        bootLog(`raiz encontrada: ${root}`);
        return root;
      }
    } catch (_) {
      // sin permisos
    }
  }
  bootLog(`raiz NO encontrada. candidatos=${rootCandidates().join(" | ")}`);
  return null;
}

function findPython(root) {
  if (process.env.RIFTSENSE_PYTHON && fs.existsSync(process.env.RIFTSENSE_PYTHON)) {
    return process.env.RIFTSENSE_PYTHON;
  }
  const candidates = [
    path.join(root, ".venv", "Scripts", "python.exe"),
    path.join(root, ".venv", "Scripts", "pythonw.exe"),
  ];
  return candidates.find((candidate) => fs.existsSync(candidate)) || null;
}

function logFile() {
  return path.join(app.getPath("userData"), "server.log");
}

function launcherLogFile() {
  return path.join(app.getPath("userData"), "launcher.log");
}

function writeLauncherLog(message) {
  try {
    fs.appendFileSync(
      launcherLogFile(),
      `[${new Date().toISOString()}] ${message}\n`,
      "utf8"
    );
  } catch (_) {
    // sin permisos
  }
}

/* ------------------------------------------------------------------ */
/* Servidor                                                            */
/* ------------------------------------------------------------------ */
function getHealth(port) {
  return new Promise((resolve) => {
    const request = http.get(
      { host: "127.0.0.1", port, path: "/api/health", timeout: 1500 },
      (response) => {
        let body = "";
        response.setEncoding("utf8");
        response.on("data", (chunk) => {
          body += chunk;
        });
        response.on("end", () => {
          try {
            const data = JSON.parse(body);
            resolve(data && data.status === "ok");
          } catch (_) {
            resolve(false);
          }
        });
      }
    );
    request.on("error", () => resolve(false));
    request.on("timeout", () => {
      request.destroy();
      resolve(false);
    });
  });
}

function isPortFree(port) {
  return new Promise((resolve) => {
    const tester = net.createServer();
    tester.once("error", () => resolve(false));
    tester.once("listening", () => tester.close(() => resolve(true)));
    tester.listen(port, "127.0.0.1");
  });
}

async function choosePort() {
  for (const port of [DEFAULT_PORT, ...FALLBACK_PORTS]) {
    if (await getHealth(port)) return { port, reuse: true };
    if (await isPortFree(port)) return { port, reuse: false };
  }
  return null;
}

function startServer(root, python, port) {
  const log = fs.createWriteStream(logFile(), { flags: "a" });
  log.write(`\n=== RiftSense ${new Date().toISOString()} ===\n`);
  log.write(`python: ${python}\nroot: ${root}\nport: ${port}\n`);

  const child = spawn(
    python,
    ["-m", "lolanalyzer.api.server", "--host", "127.0.0.1", "--port", String(port)],
    {
      cwd: root,
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
      },
      stdio: ["ignore", "pipe", "pipe"],
    }
  );

  child.stdout.pipe(log);
  child.stderr.pipe(log);
  child.on("error", (error) => {
    log.write(`spawn error: ${error.message}\n`);
  });
  child.on("exit", (code, signal) => {
    log.write(`server exit code=${code} signal=${signal}\n`);
    if (!quitting && code !== 0) {
      writeLauncherLog(`El servidor termino con codigo ${code}`);
    }
    serverProcess = null;
    ownsServer = false;
  });
  return child;
}

function stopServer() {
  return new Promise((resolve) => {
    if (!serverProcess || !ownsServer) return resolve();
    const pid = serverProcess.pid;
    if (!pid) return resolve();
    execFile(
      "taskkill",
      ["/PID", String(pid), "/T", "/F"],
      { windowsHide: true },
      () => resolve()
    );
    setTimeout(resolve, 2500);
  });
}

async function waitForServer(port, timeoutMs = 60000) {
  const started = Date.now();
  let lastStatus = 0;
  while (Date.now() - started < timeoutMs) {
    if (await getHealth(port)) return true;
    if (!serverProcess) return false;
    const elapsed = Math.round((Date.now() - started) / 1000);
    if (Date.now() - lastStatus > 900) {
      lastStatus = Date.now();
      setSplashStatus(`Iniciando servidor local… ${elapsed}s`);
    }
    await delay(500);
  }
  return false;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function logTail(lines = 18) {
  try {
    const text = fs.readFileSync(logFile(), "utf8");
    return text.trim().split(/\r?\n/).slice(-lines).join("\n");
  } catch (_) {
    return "(sin log)";
  }
}

function showFailure(message, detail) {
  writeLauncherLog(`${message}\n${detail || ""}`);
  dialog.showErrorBox(
    "RiftSense",
    `${message}\n\n${detail || ""}\n\nLog:\n${logFile()}`
  );
}

/* ------------------------------------------------------------------ */
/* Actualizaciones (GitHub Releases)                                   */
/* ------------------------------------------------------------------ */
function runCommand(command, args, timeoutMs = 15000) {
  return new Promise((resolve) => {
    let settled = false;
    let stdout = "";
    let stderr = "";
    let child;
    try {
      child = spawn(command, args, { windowsHide: true });
    } catch (error) {
      resolve({ code: -1, stdout: "", stderr: String(error) });
      return;
    }

    const finish = (code, error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({ code, stdout, stderr, error });
    };

    const timer = setTimeout(() => {
      try {
        child.kill();
      } catch (_) {
        // sin permisos
      }
      finish(-1, "timeout");
    }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => finish(-1, String(error)));
    child.on("close", (code) => finish(code === null ? -1 : code));
  });
}

function normalizeVersion(value) {
  return String(value || "")
    .trim()
    .replace(/^v/i, "")
    .split("-")[0]
    .split("+")[0];
}

function compareVersions(a, b) {
  const pa = normalizeVersion(a).split(".").map((n) => parseInt(n, 10) || 0);
  const pb = normalizeVersion(b).split(".").map((n) => parseInt(n, 10) || 0);
  for (let i = 0; i < Math.max(pa.length, pb.length); i += 1) {
    const diff = (pa[i] || 0) - (pb[i] || 0);
    if (diff !== 0) return diff > 0 ? 1 : -1;
  }
  return 0;
}

function currentVersion() {
  return normalizeVersion(process.env.RIFTSENSE_CURRENT_VERSION || app.getVersion());
}

function updateTargetExe() {
  const direct = process.env.PORTABLE_EXECUTABLE_FILE;
  if (direct && fs.existsSync(direct)) return direct;
  const dir = process.env.PORTABLE_EXECUTABLE_DIR;
  if (dir) {
    const candidate = path.join(dir, "RiftSense.exe");
    if (fs.existsSync(candidate)) return candidate;
  }
  return null;
}

function updatesSupported() {
  return Boolean(updateTargetExe());
}

async function checkForUpdates() {
  const current = currentVersion();
  if (process.env.RIFTSENSE_DISABLE_UPDATES === "1" || !updatesSupported()) {
    return { available: false, current, disabled: true };
  }

  const result = await runCommand(
    GH_BIN,
    [
      "release",
      "view",
      "--repo",
      UPDATE_REPO,
      "--json",
      "tagName,name,body,assets,publishedAt",
    ],
    12000
  );
  if (result.code !== 0) {
    bootLog(`update check fallo: ${result.stderr.slice(0, 400)}`);
    return { available: false, current, error: "No se pudo consultar GitHub." };
  }

  let release;
  try {
    release = JSON.parse(result.stdout);
  } catch (_) {
    return { available: false, current, error: "Respuesta de GitHub invalida." };
  }
  const latest = normalizeVersion(release.tagName || release.name);
  if (!latest) {
    return { available: false, current, error: "Release sin version." };
  }
  const notes = String(release.body || "");
  if (compareVersions(latest, current) <= 0) {
    return { available: false, current, latest, notes, tag: release.tagName };
  }
  return {
    available: true,
    current,
    latest,
    notes,
    tag: release.tagName,
    assets: release.assets || [],
    repo: UPDATE_REPO,
  };
}

async function downloadReleaseAsset(tag, name, dir, timeoutMs = 600000) {
  return runCommand(
    GH_BIN,
    [
      "release",
      "download",
      tag,
      "--repo",
      UPDATE_REPO,
      "--pattern",
      name,
      "--dir",
      dir,
      "--clobber",
    ],
    timeoutMs
  );
}

function sha256File(file) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    const stream = fs.createReadStream(file);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}

async function promptUpdate(update) {
  if (process.env.RIFTSENSE_UPDATE_YES === "1") return true;
  const notes = String(update.notes || "Sin notas de la version.").slice(0, 2500);
  const result = await dialog.showMessageBox(mainWindow, {
    type: "info",
    title: "Actualizacion disponible",
    message: `RiftSense ${update.latest} esta disponible`,
    detail:
      `Version actual: ${update.current}\n` +
      `Version nueva: ${update.latest}\n\n` +
      notes,
    buttons: ["Actualizar ahora", "Mas tarde"],
    defaultId: 0,
    cancelId: 1,
    noLink: true,
  });
  return result.response === 0;
}

function updaterScript() {
  return String.raw`param(
  [int]$LauncherPid,
  [string]$Root,
  [string]$Exe,
  [string]$NewExe,
  [string]$ProjectZip,
  [string]$Port,
  [string]$Log
)

$ErrorActionPreference = "Stop"

function Write-UpdateLog([string]$Text) {
  try {
    Add-Content -LiteralPath $Log -Value ("[{0}] {1}" -f (Get-Date -Format "o"), $Text) -Encoding UTF8
  } catch {
  }
}

$backup = "$Exe.bak"
try {
  Write-UpdateLog "Esperando cierre del lanzador $LauncherPid"
  while (Get-Process -Id $LauncherPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Milliseconds 500
  }
  Start-Sleep -Milliseconds 800

  if ($Port) {
    try {
      $connection = Get-NetTCPConnection -LocalPort ([int]$Port) -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
      if ($connection) {
        Write-UpdateLog ("Deteniendo servidor local PID " + $connection.OwningProcess)
        Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 800
      }
    } catch {
      Write-UpdateLog ("No se pudo consultar el puerto: " + $_.Exception.Message)
    }
  }

  if (Test-Path -LiteralPath $Exe) {
    Copy-Item -LiteralPath $Exe -Destination $backup -Force
  }
  if (-not (Test-Path -LiteralPath $NewExe)) {
    throw "No existe el nuevo ejecutable descargado."
  }

  $copied = $false
  for ($i = 0; $i -lt 20; $i += 1) {
    try {
      Copy-Item -LiteralPath $NewExe -Destination $Exe -Force
      $copied = $true
      break
    } catch {
      Start-Sleep -Milliseconds 500
    }
  }
  if (-not $copied) {
    throw "No se pudo reemplazar RiftSense.exe."
  }
  Write-UpdateLog "Ejecutable actualizado."

  if ($ProjectZip -and (Test-Path -LiteralPath $ProjectZip)) {
    $extract = Join-Path ([IO.Path]::GetTempPath()) ("RiftSense-update-" + [Guid]::NewGuid().ToString("N"))
    Expand-Archive -LiteralPath $ProjectZip -DestinationPath $extract -Force
    foreach ($dir in @("lolanalyzer", "web", "scripts", "artifacts")) {
      $source = Join-Path $extract $dir
      if (Test-Path -LiteralPath $source) {
        $destination = Join-Path $Root $dir
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        Copy-Item -Path (Join-Path $source "*") -Destination $destination -Recurse -Force
      }
    }
    $processed = Join-Path $extract "data\processed"
    if (Test-Path -LiteralPath $processed) {
      $destination = Join-Path $Root "data\processed"
      New-Item -ItemType Directory -Path $destination -Force | Out-Null
      Copy-Item -Path (Join-Path $processed "*") -Destination $destination -Recurse -Force
    }
    foreach ($file in @("README.md", "requirements.txt", "start.ps1", ".env.example")) {
      $source = Join-Path $extract $file
      if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $Root $file) -Force
      }
    }
    Remove-Item -LiteralPath $extract -Recurse -Force -ErrorAction SilentlyContinue
    Write-UpdateLog "Codigo del proyecto actualizado."
  }

  $python = Join-Path $Root ".venv\Scripts\python.exe"
  $requirements = Join-Path $Root "requirements.txt"
  if ((Test-Path -LiteralPath $python) -and (Test-Path -LiteralPath $requirements)) {
    Write-UpdateLog "Revisando dependencias..."
    & $python -m pip install -r $requirements --disable-pip-version-check --no-input 2>&1 | Add-Content -LiteralPath $Log -Encoding UTF8
  }

  Remove-Item Env:RIFTSENSE_CURRENT_VERSION -ErrorAction SilentlyContinue
  Remove-Item Env:RIFTSENSE_UPDATE_YES -ErrorAction SilentlyContinue
  Remove-Item Env:RIFTSENSE_UPDATE_REPO -ErrorAction SilentlyContinue
  Remove-Item Env:RIFTSENSE_GH -ErrorAction SilentlyContinue

  Write-UpdateLog "Iniciando nueva version."
  Start-Process -FilePath $Exe -WorkingDirectory $Root

  $updateDir = Split-Path -Parent $PSCommandPath
  if ($updateDir -and (Split-Path -Leaf $updateDir) -like "RiftSense-update-*") {
    Start-Sleep -Milliseconds 1500
    Remove-Item -LiteralPath $updateDir -Recurse -Force -ErrorAction SilentlyContinue
  }
} catch {
  Write-UpdateLog ("ERROR: " + $_.Exception.Message)
  if (Test-Path -LiteralPath $backup) {
    try {
      Copy-Item -LiteralPath $backup -Destination $Exe -Force
    } catch {
    }
  }
  if (Test-Path -LiteralPath $Exe) {
    Start-Process -FilePath $Exe -WorkingDirectory $Root
  }
  exit 1
}
`;
}

function powershellExe() {
  if (process.env.RIFTSENSE_POWERSHELL) return process.env.RIFTSENSE_POWERSHELL;
  const systemRoot = process.env.SystemRoot || process.env.windir || "C:\\Windows";
  const candidate = path.join(
    systemRoot,
    "System32",
    "WindowsPowerShell",
    "v1.0",
    "powershell.exe"
  );
  return fs.existsSync(candidate) ? candidate : "powershell.exe";
}

function winQuote(value) {
  const text = String(value);
  if (text && !/[\s"]/.test(text)) return text;
  const escaped = text.replace(/(\\*)"/g, (_match, slashes) => `${slashes}${slashes}\\"`);
  return `"${escaped.replace(/(\\+)$/, "$1$1")}"`;
}

async function launchUpdater(commandLine) {
  const launcherPath = path.join(os.tmpdir(), "RiftSense-update-launch.ps1");
  fs.writeFileSync(
    launcherPath,
    fs.readFileSync(path.join(__dirname, "update-launch.ps1"), "utf8"),
    "utf8"
  );
  const result = await runCommand(
    powershellExe(),
    [
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      launcherPath,
      "-CommandLine",
      commandLine,
    ],
    20000
  );
  bootLog(
    `updater launcher code=${result.code} stdout=${result.stdout.trim()} ` +
      `stderr=${result.stderr.trim().slice(0, 500)}`
  );
  if (result.code !== 0) {
    throw new Error(
      `No se pudo iniciar el actualizador:\n${result.stderr.slice(0, 800)}`
    );
  }
  return result;
}

async function performUpdate(update) {
  if (updateInProgress) return;
  updateInProgress = true;

  const targetExe = updateTargetExe();
  if (!targetExe || !projectRoot) {
    updateInProgress = false;
    throw new Error(
      "La actualizacion automatica solo esta disponible en el ejecutable portable."
    );
  }

  const temp = path.join(os.tmpdir(), `RiftSense-update-${Date.now()}`);
  fs.mkdirSync(temp, { recursive: true });
  const updateLog = path.join(os.tmpdir(), "RiftSense-update.log");

  setSplashStatus(`Descargando actualizacion ${update.latest}…`);
  const manifestPath = path.join(temp, "latest.json");
  let result = await downloadReleaseAsset(update.tag, "latest.json", temp);
  if (result.code !== 0 || !fs.existsSync(manifestPath)) {
    throw new Error(`No se pudo descargar latest.json:\n${result.stderr.slice(0, 500)}`);
  }

  let manifest;
  try {
    const raw = fs.readFileSync(manifestPath, "utf8").replace(/^\uFEFF/, "");
    manifest = JSON.parse(raw);
  } catch (error) {
    throw new Error(`latest.json invalido: ${error.message}`);
  }
  if (normalizeVersion(manifest.version) !== update.latest) {
    throw new Error(
      `La release anuncia ${update.latest} pero el manifiesto dice ${manifest.version}.`
    );
  }

  const exeAsset = manifest.exe || { name: "RiftSense.exe" };
  if (!exeAsset.sha256) {
    throw new Error("El manifiesto no incluye el SHA-256 del ejecutable.");
  }
  const exeName = exeAsset.name || "RiftSense.exe";
  const newExe = path.join(temp, exeName);
  setSplashStatus(`Descargando ejecutable ${update.latest}…`);
  result = await downloadReleaseAsset(update.tag, exeName, temp);
  if (result.code !== 0 || !fs.existsSync(newExe)) {
    throw new Error(`No se pudo descargar ${exeName}:\n${result.stderr.slice(0, 500)}`);
  }
  const hash = await sha256File(newExe);
  if (hash.toLowerCase() !== String(exeAsset.sha256).toLowerCase()) {
    throw new Error("El SHA-256 del ejecutable no coincide.");
  }

  let projectZip = "";
  if (manifest.project && manifest.project.name) {
    if (!manifest.project.sha256) {
      throw new Error("El manifiesto no incluye el SHA-256 del proyecto.");
    }
    projectZip = path.join(temp, manifest.project.name);
    setSplashStatus("Descargando codigo del proyecto…");
    result = await downloadReleaseAsset(update.tag, manifest.project.name, temp);
    if (result.code !== 0 || !fs.existsSync(projectZip)) {
      throw new Error(
        `No se pudo descargar ${manifest.project.name}:\n${result.stderr.slice(0, 500)}`
      );
    }
    const hash = await sha256File(projectZip);
    if (hash.toLowerCase() !== String(manifest.project.sha256).toLowerCase()) {
      throw new Error("El SHA-256 del proyecto no coincide.");
    }
  }

  const scriptPath = path.join(temp, "update.ps1");
  fs.writeFileSync(scriptPath, "\ufeff" + updaterScript(), "utf8");
  setSplashStatus("Instalando actualizacion…");
  bootLog(`actualizacion ${update.latest}: lanzando updater`);

  const commandLine = [
    powershellExe(),
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-WindowStyle",
    "Hidden",
    "-File",
    scriptPath,
    "-LauncherPid",
    String(process.pid),
    "-Root",
    projectRoot,
    "-Exe",
    targetExe,
    "-NewExe",
    newExe,
    "-ProjectZip",
    projectZip,
    "-Port",
    String(serverPort || ""),
    "-Log",
    updateLog,
  ]
    .map((value) => winQuote(value))
    .join(" ");
  bootLog(`updater command: ${commandLine}`);
  await launchUpdater(commandLine);

  await stopServer();
  ownsServer = false;
  serverProcess = null;
  quitting = true;
  app.quit();
}

/* ------------------------------------------------------------------ */
/* Ventana                                                             */
/* ------------------------------------------------------------------ */
function setSplashStatus(text) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const safe = JSON.stringify(text);
  mainWindow.webContents
    .executeJavaScript(
      `(() => { const el = document.getElementById("status"); if (el) el.textContent = ${safe}; })()`
    )
    .catch(() => {});
}

function createWindow(iconPath) {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1000,
    minHeight: 680,
    show: false,
    backgroundColor: "#060a12",
    title: "RiftSense",
    icon: iconPath,
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      spellcheck: false,
    },
  });

  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (serverPort && url.startsWith(`http://127.0.0.1:${serverPort}`)) {
      return { action: "allow" };
    }
    if (/^https?:\/\//i.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });

  mainWindow.webContents.on("will-navigate", (event, url) => {
    if (url.startsWith("file://")) return;
    if (serverPort && url.startsWith(`http://127.0.0.1:${serverPort}`)) return;
    event.preventDefault();
    if (/^https?:\/\//i.test(url)) shell.openExternal(url);
  });

  mainWindow.webContents.on("before-input-event", (event, input) => {
    if (input.type !== "keyDown") return;
    if (input.key === "F5" || (input.control && input.key.toLowerCase() === "r")) {
      event.preventDefault();
      mainWindow.webContents.reload();
    } else if (
      input.key === "F12" ||
      (input.control && input.shift && input.key.toLowerCase() === "i")
    ) {
      event.preventDefault();
      mainWindow.webContents.toggleDevTools();
    }
  });

  mainWindow.webContents.on("did-fail-load", (_event, code, description, url, isMainFrame) => {
    if (!isMainFrame) return;
    if (url.startsWith("file://")) return;
    showFailure(
      `No se pudo cargar la interfaz (${code}: ${description}).`,
      "Comprueba que el servidor local sigue funcionando."
    );
  });

  return mainWindow;
}

async function main() {
  bootLog("main() inicio");
  const root = findProjectRoot();
  if (!root) {
    showFailure(
      "No se encontro la carpeta del proyecto RiftSense.",
      "Coloca RiftSense.exe dentro de la carpeta que contiene .venv, lolanalyzer y web.\n" +
        "Tambien puedes crear rift-sense.config.json junto al .exe con:\n" +
        '  { "root": "C:\\\\ruta\\\\al\\\\proyecto" }'
    );
    app.quit();
    return;
  }
  bootLog(`root=${root}`);
  projectRoot = root;

  const python = findPython(root);
  if (!python) {
    showFailure(
      "No se encontro el entorno Python del proyecto.",
      "Ejecuta .\\start.ps1 o crea el entorno con:\n" +
        "  py -m venv .venv\n" +
        "  .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
    );
    app.quit();
    return;
  }
  bootLog(`python=${python}`);

  const iconPath = path.join(__dirname, "build", "icon.png");
  createWindow(iconPath);
  await mainWindow.loadFile(path.join(__dirname, "splash.html"));
  mainWindow.show();
  bootLog("splash cargado");

  const selected = await choosePort();
  bootLog(`puerto seleccionado: ${selected ? JSON.stringify(selected) : "ninguno"}`);
  if (!selected) {
    showFailure(
      "No hay un puerto libre para el servidor local.",
      "Cierra otros procesos que esten usando los puertos 8000 o 8765-8785."
    );
    app.quit();
    return;
  }

  serverPort = selected.port;
  if (!selected.reuse) {
    setSplashStatus("Iniciando servidor local…");
    serverProcess = startServer(root, python, serverPort);
    ownsServer = true;
    const ready = await waitForServer(serverPort);
    bootLog(`servidor listo=${ready}`);
    if (!ready) {
      await stopServer();
      showFailure(
        "El servidor local no respondio a tiempo.",
        `Ultimas lineas del log:\n${logTail()}`
      );
      app.quit();
      return;
    }
  } else {
    writeLauncherLog(`Reutilizando servidor existente en el puerto ${serverPort}`);
    bootLog(`reutilizando servidor en ${serverPort}`);
  }

  setSplashStatus("Buscando actualizaciones…");
  try {
    const update = await checkForUpdates();
    if (update.available) {
      bootLog(`actualizacion disponible ${update.latest} (actual ${update.current})`);
      const accepted = await promptUpdate(update);
      if (accepted) {
        try {
          await performUpdate(update);
          return;
        } catch (error) {
          updateInProgress = false;
          bootLog(`update error ${error && error.stack ? error.stack : error}`);
          showFailure(
            "No se pudo completar la actualizacion.",
            `${String(error)}\n\nLog del actualizador:\n${path.join(
              os.tmpdir(),
              "RiftSense-update.log"
            )}`
          );
        }
      } else {
        bootLog("actualizacion pospuesta por el usuario");
      }
    } else if (update.latest) {
      bootLog(`sin actualizacion (actual ${update.current}, ultima ${update.latest})`);
    } else if (update.disabled) {
      bootLog("actualizaciones desactivadas o no soportadas en este modo");
    }
  } catch (error) {
    bootLog(`update check error ${error && error.stack ? error.stack : error}`);
  }

  setSplashStatus("Cargando interfaz…");
  const appUrl = `http://127.0.0.1:${serverPort}/`;
  bootLog(`loadURL ${appUrl}`);
  try {
    await mainWindow.loadURL(appUrl);
    mainWindow.setTitle("RiftSense · Analizador de draft y build");
    bootLog("interfaz cargada");
  } catch (error) {
    bootLog(`loadURL error ${error && error.stack ? error.stack : error}`);
    showFailure("No se pudo abrir la interfaz de RiftSense.", String(error));
    app.quit();
  }
}

app.whenReady().then(() => {
  bootLog("app ready");
  Menu.setApplicationMenu(null);
  main().catch((error) => {
    bootLog(`main error ${error && error.stack ? error.stack : error}`);
    showFailure("Error iniciando RiftSense.", String(error));
    app.quit();
  });
});

app.on("window-all-closed", () => {
  app.quit();
});

app.on("before-quit", (event) => {
  if (ownsServer && serverProcess && !quitting) {
    event.preventDefault();
    quitting = true;
    stopServer().finally(() => app.quit());
  }
});

process.on("uncaughtException", (error) => {
  bootLog(`uncaught: ${error && error.stack ? error.stack : error}`);
  writeLauncherLog(`uncaught: ${error && error.stack ? error.stack : error}`);
});

process.on("unhandledRejection", (reason) => {
  bootLog(`unhandledRejection: ${reason && reason.stack ? reason.stack : reason}`);
  writeLauncherLog(
    `unhandledRejection: ${reason && reason.stack ? reason.stack : reason}`
  );
});
