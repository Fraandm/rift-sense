/* Lanzador de escritorio: servidor Python + ventana Chromium (Electron). */
const { app, BrowserWindow, Menu, dialog, shell } = require("electron");
const { spawn, execFile } = require("child_process");
const path = require("path");
const fs = require("fs");
const http = require("http");
const net = require("net");
const os = require("os");

const DEFAULT_PORT = 8000;
const FALLBACK_PORTS = Array.from({ length: 21 }, (_, i) => 8765 + i);
const BOOT_LOG = path.join(os.tmpdir(), "riftsense-boot.log");

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
