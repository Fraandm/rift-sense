"""Localiza League of Legends en el PC y gestiona el lockfile del LCU.

Fuentes, en orden:
  1. Variables LOL_LOCKFILE / LOL_INSTALL_DIR
  2. Ruta guardada por el usuario (data/client_location.json)
  3. Metadata oficial del Riot Client (product_install_full_path)
  4. RiotClientInstalls.json
  5. Registro de Windows (desinstalador de League of Legends)
  6. Rutas habituales en todas las unidades
  7. Escaneo en segundo plano de todas las unidades fijas/extraibles
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import config
from ..utils import read_json, write_json

SETTINGS_PATH = config.DATA_DIR / "client_location.json"

SKIP_DIRS = {
    "windows",
    "$recycle.bin",
    "system volume information",
    "recovery",
    "perflogs",
    "config.msi",
    "node_modules",
    ".git",
    ".venv",
    "__pycache__",
    "appdata",
    "onedrive",
    "programdata",
}

COMMON_SUBPATHS = (
    "Riot Games/League of Legends",
    "Games/League of Legends",
    "Games/Riot Games/League of Legends",
    "Program Files/Riot Games/League of Legends",
    "Program Files (x86)/Riot Games/League of Legends",
    "League of Legends",
)


@dataclass
class LockfileInfo:
    path: Path
    pid: int
    port: int
    password: str
    protocol: str


@dataclass
class ClientLocation:
    install_dir: Path | None
    lockfile: Path | None
    source: str

    @property
    def drive(self) -> str | None:
        target = self.install_dir or self.lockfile
        if not target:
            return None
        return target.drive or None


def read_lockfile(path: Path) -> LockfileInfo | None:
    """Lee y valida un lockfile. Solo acepta el proceso LeagueClient."""
    try:
        if not path.is_file() or path.stat().st_size < 10:
            return None
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
    parts = raw.split(":", 4)
    if len(parts) < 5:
        return None
    if "leagueclient" not in parts[0].strip().lower():
        return None
    try:
        pid = int(parts[1])
        port = int(parts[2])
    except ValueError:
        return None
    password = parts[3]
    protocol = parts[4] or "https"
    if not password or not (0 < port < 65536):
        return None
    return LockfileInfo(path=path, pid=pid, port=port, password=password, protocol=protocol)


def drive_roots() -> list[Path]:
    """Unidades locales/extraibles disponibles (sin red ni CD-ROM)."""
    roots: list[Path] = []
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            mask = kernel32.GetLogicalDrives()
            for index in range(26):
                if not mask & (1 << index):
                    continue
                letter = chr(65 + index)
                root = f"{letter}:\\"
                drive_type = kernel32.GetDriveTypeW(root)
                if drive_type in (2, 3):  # removable / fixed
                    roots.append(Path(root))
            return roots
        except Exception:
            pass
    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        root = Path(f"{letter}:\\")
        if root.exists():
            roots.append(root)
    return roots


def _valid_install(path: Path) -> ClientLocation | None:
    """Comprueba si una carpeta es una instalacion de LoL (aunque este cerrada)."""
    path = path.expanduser()
    try:
        path = path.resolve()
    except OSError:
        return None
    if path.is_file():
        name = path.name.lower()
        if name == "lockfile":
            info = read_lockfile(path)
            return ClientLocation(path.parent, path, "manual") if info else None
        if name in {"leagueclient.exe", "leagueclientux.exe"}:
            path = path.parent
        else:
            return None
    if not path.is_dir():
        return None

    candidates = [path]
    if path.name.lower() == "riot games":
        candidates.append(path / "League of Legends")
    if (path / "League of Legends").is_dir():
        candidates.append(path / "League of Legends")

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        lock = candidate / "lockfile"
        if read_lockfile(lock):
            return ClientLocation(candidate, lock, "manual")
        if (candidate / "LeagueClient.exe").is_file() or (candidate / "LeagueClientUx.exe").is_file():
            return ClientLocation(candidate, lock if lock.is_file() else None, "manual")
    return None


def _location_from_path(raw: Any, source: str) -> ClientLocation | None:
    if raw is None:
        return None
    text = str(raw).strip().strip('"').strip("'")
    if not text:
        return None
    path = Path(text).expanduser()
    try:
        if not path.exists():
            return None
        location = _valid_install(path)
    except OSError:
        return None
    if location:
        location.source = source
    return location


def _from_env() -> ClientLocation | None:
    for name in ("LOL_INSTALL_DIR", "LOL_LOCKFILE"):
        value = os.environ.get(name)
        if not value:
            continue
        location = _location_from_path(value, "env")
        if location:
            return location
        # Si LOL_LOCKFILE apunta a un lockfile aun inexistente, probar su carpeta
        candidate = Path(value)
        if candidate.suffix.lower() == "" and candidate.is_dir():
            location = _location_from_path(candidate, "env")
            if location:
                return location
    return None


def _from_saved() -> ClientLocation | None:
    data = read_json(SETTINGS_PATH) or {}
    for key in ("install_dir", "lockfile"):
        location = _location_from_path(data.get(key), "saved")
        if location:
            return location
    return None


def _from_riot_installs() -> ClientLocation | None:
    drive = os.environ.get("ProgramData", "C:/ProgramData")
    path = Path(drive) / "Riot Games" / "RiotClientInstalls.json"
    data = read_json(path) or {}
    associated = data.get("associated_client") or {}
    if isinstance(associated, dict):
        for install in associated:
            if "pbe" in str(install).lower():
                continue
            location = _location_from_path(install, "riot-installs")
            if location:
                return location
    return None


def _read_product_path(yaml_path: Path) -> Path | None:
    try:
        text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    full = None
    root = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("product_install_full_path:"):
            full = stripped.split(":", 1)[1].strip().strip('"')
        elif stripped.startswith("product_install_root:"):
            root = stripped.split(":", 1)[1].strip().strip('"')
    if full:
        return Path(full)
    if root:
        return Path(root) / "League of Legends"
    return None


def _from_riot_metadata() -> ClientLocation | None:
    program_data = Path(os.environ.get("ProgramData", "C:/ProgramData"))
    base = program_data / "Riot Games" / "Metadata"
    if not base.is_dir():
        return None
    for product in ("league_of_legends.live", "league_of_legends.pbe"):
        if "pbe" in product:
            continue  # el analizador apunta al cliente live
        folder = base / product
        if not folder.is_dir():
            continue
        yaml_path = folder / f"{product}.product_settings.yaml"
        install = _read_product_path(yaml_path)
        if install:
            location = _location_from_path(install, "riot-metadata")
            if location:
                return location
    return None


def _from_registry() -> ClientLocation | None:
    if os.name != "nt":
        return None
    try:
        import winreg
    except ImportError:
        return None

    roots = (
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    )
    candidates: list[str] = []
    for hive, subkey in roots:
        try:
            key = winreg.OpenKey(hive, subkey)
        except OSError:
            continue
        try:
            count = winreg.QueryInfoKey(key)[0]
            for index in range(count):
                try:
                    name = winreg.EnumKey(key, index)
                    child = winreg.OpenKey(key, name)
                except OSError:
                    continue
                try:
                    try:
                        display = winreg.QueryValueEx(child, "DisplayName")[0]
                    except OSError:
                        display = ""
                    if not display or "pbe" in str(display).lower():
                        continue
                    if str(display).strip().lower() not in {"league of legends", "league of legends live"}:
                        continue
                    try:
                        location = winreg.QueryValueEx(child, "InstallLocation")[0]
                    except OSError:
                        location = ""
                    if location:
                        candidates.append(str(location))
                finally:
                    child.Close()
        finally:
            key.Close()

    for candidate in candidates:
        location = _location_from_path(candidate, "registry")
        if location:
            return location
    return None


def _from_common() -> ClientLocation | None:
    for root in drive_roots():
        for relative in COMMON_SUBPATHS:
            location = _location_from_path(root / relative, "common")
            if location:
                return location
    return None


def _fast_locations() -> ClientLocation | None:
    for finder in (
        _from_env,
        _from_saved,
        _from_riot_metadata,
        _from_riot_installs,
        _from_registry,
        _from_common,
    ):
        try:
            location = finder()
        except Exception:
            location = None
        if location:
            return location
    return None


class ClientLocator:
    """Buscador reanudable y en segundo plano de la instalacion de LoL."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._searching = False
        self._state = "idle"
        self._location: ClientLocation | None = None
        self._drives: list[str] = []
        self._scanned = 0
        self._current_drive: str | None = None
        self._started_at: str | None = None
        self._finished_at: str | None = None
        self._error: str | None = None
        self._incomplete = False

    # ------------------------------------------------------------------ #
    # API publica
    # ------------------------------------------------------------------ #
    def start(self, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._searching:
                return self.status()
            if not force and self._state == "found" and self._location:
                return self.status()

            roots = drive_roots()
            self._generation += 1
            generation = self._generation
            self._searching = True
            self._state = "searching"
            self._drives = [root.drive or str(root) for root in roots]
            self._scanned = 0
            self._current_drive = self._drives[0] if self._drives else None
            self._started_at = dt.datetime.now(dt.timezone.utc).isoformat()
            self._finished_at = None
            self._error = None
            self._incomplete = False

            thread = threading.Thread(
                target=self._run,
                args=(generation, roots),
                name="lol-client-locator",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            print(
                f"[lcu] Buscando League of Legends en {', '.join(self._drives) or 'el sistema'}...",
                flush=True,
            )
        return self.status()

    def ensure_started(self) -> None:
        if self._state == "idle" and not self._searching and not self._location:
            self.start(force=True)

    def status(self) -> dict[str, Any]:
        with self._lock:
            location = self._location
            install = location.install_dir if location else None
            lockfile = location.lockfile if location else None
            if location and location.install_dir and not lockfile:
                candidate = location.install_dir / "lockfile"
                if candidate.is_file():
                    lockfile = candidate
            if lockfile and not read_lockfile(lockfile):
                lockfile = None
            return {
                "state": self._state,
                "searching": self._searching,
                "installPath": str(install) if install else None,
                "lockfilePath": str(lockfile) if lockfile else None,
                "drive": location.drive if location else None,
                "source": location.source if location else None,
                "drives": list(self._drives),
                "scannedDirs": self._scanned,
                "currentDrive": self._current_drive,
                "startedAt": self._started_at,
                "finishedAt": self._finished_at,
                "error": self._error,
                "incomplete": self._incomplete,
                "saved": str(SETTINGS_PATH) if SETTINGS_PATH.exists() else None,
            }

    def lockfile_path(self) -> Path | None:
        self.ensure_started()
        with self._lock:
            location = self._location
        if location:
            for candidate in (location.lockfile, location.install_dir / "lockfile" if location.install_dir else None):
                if candidate and read_lockfile(candidate):
                    return candidate
        quick = _fast_locations()
        if quick:
            with self._lock:
                if not self._location:
                    self._location = quick
                    self._state = "found"
                    self._error = None
            if quick.lockfile and read_lockfile(quick.lockfile):
                return quick.lockfile
            if quick.install_dir:
                candidate = quick.install_dir / "lockfile"
                if read_lockfile(candidate):
                    return candidate
        return None

    def set_custom(self, raw_path: str) -> dict[str, Any]:
        location = _location_from_path(raw_path, "manual")
        if not location:
            location = _location_from_path(Path(raw_path) / "League of Legends", "manual")
        if not location:
            raise ValueError(
                "No se encontró LeagueClient.exe ni lockfile en esa ruta. "
                "Selecciona la carpeta que contiene LeagueClient.exe."
            )
        with self._lock:
            self._generation += 1
            self._searching = False
            self._state = "found"
            self._location = location
            self._error = None
            self._finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
            self._incomplete = False
            self._save(location)
        print(f"[lcu] Ruta manual: {location.install_dir}", flush=True)
        return self.status()

    # ------------------------------------------------------------------ #
    # Interno
    # ------------------------------------------------------------------ #
    def _run(self, generation: int, roots: list[Path]) -> None:
        location: ClientLocation | None = None
        error: str | None = None
        try:
            location = _fast_locations()
            if location is None:
                location = self._scan(generation, roots)
            if location is None and not self._incomplete:
                error = "No se encontró League of Legends en este PC."
        except Exception as exc:
            error = f"Error durante la búsqueda: {exc}"

        with self._lock:
            if generation != self._generation:
                return
            self._searching = False
            self._finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
            if location:
                self._location = location
                self._state = "found"
                self._error = None
                self._save(location)
                print(
                    f"[lcu] Cliente encontrado en {location.drive or '?'} "
                    f"-> {location.install_dir} ({location.source})",
                    flush=True,
                )
            else:
                self._state = "not_found"
                self._error = error or (
                    "Búsqueda incompleta: se agotó el tiempo. Pulsa Buscar de nuevo."
                )

    def _scan(self, generation: int, roots: list[Path]) -> ClientLocation | None:
        deadline = time.time() + 45.0
        scanned = 0
        for root in roots:
            with self._lock:
                if generation != self._generation:
                    return None
                self._current_drive = root.drive or str(root)
            queue: deque[tuple[Path, int]] = deque([(root, 0)])
            while queue:
                if time.time() > deadline:
                    with self._lock:
                        self._incomplete = True
                    return None
                path, depth = queue.popleft()
                if depth > 4:
                    continue
                try:
                    entries = list(os.scandir(path))
                except OSError:
                    continue
                scanned += 1
                if scanned % 25 == 0:
                    with self._lock:
                        if generation != self._generation:
                            return None
                        self._scanned = scanned

                for entry in entries:
                    try:
                        if entry.is_file(follow_symlinks=False) and entry.name.lower() == "leagueclient.exe":
                            location = _location_from_path(path, "scan")
                            if location:
                                with self._lock:
                                    self._scanned = scanned
                                return location
                    except OSError:
                        continue

                for entry in entries:
                    try:
                        if not entry.is_dir(follow_symlinks=False):
                            continue
                        name = entry.name.lower()
                        if name in SKIP_DIRS or "pbe" in name:
                            continue
                        queue.append((Path(entry.path), depth + 1))
                    except OSError:
                        continue

        with self._lock:
            self._scanned = scanned
        return None

    def _save(self, location: ClientLocation) -> None:
        write_json(
            SETTINGS_PATH,
            {
                "install_dir": str(location.install_dir) if location.install_dir else None,
                "lockfile": str(location.lockfile) if location.lockfile else None,
                "source": location.source,
                "updatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )


_LOCATOR = ClientLocator()


def get_locator() -> ClientLocator:
    return _LOCATOR


def find_lockfile() -> Path | None:
    return _LOCATOR.lockfile_path()
