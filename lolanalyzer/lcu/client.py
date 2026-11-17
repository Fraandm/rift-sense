"""Conector de solo lectura con el cliente local de League of Legends (LCU).

La localizacion del cliente y la lectura del lockfile viven en locator.py.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Any

import requests
import urllib3

from .locator import LockfileInfo, get_locator, read_lockfile

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LIVE_URLS = (
    "https://127.0.0.1:2999/liveclientdata/allgamedata",
    "http://127.0.0.1:2999/liveclientdata/allgamedata",
)


class LeagueClient:
    """Cliente LCU con redeteccion automatica del lockfile."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live_lock = threading.Lock()
        self._info: LockfileInfo | None = None
        self._session: requests.Session | None = None
        self._live_session: requests.Session | None = None
        self._last_error: str | None = None
        self._last_discovery = 0.0
        self._missing_since: float | None = None
        self.locator = get_locator()

    # ------------------------------------------------------------------ #
    # Conexion
    # ------------------------------------------------------------------ #
    def _rediscover(self) -> bool:
        path = self.locator.lockfile_path()
        search = self.locator.status()
        if not path:
            self._info = None
            self._session = None
            if search.get("searching"):
                self._last_error = "Buscando League of Legends en los discos..."
            elif search.get("installPath"):
                drive = search.get("drive") or "?"
                self._last_error = (
                    f"Cliente encontrado en el disco {drive} pero está cerrado. "
                    "Abre League of Legends para conectarlo."
                )
            else:
                self._last_error = (
                    "No se encontró el cliente de LoL. Revisa la instalación o fija la ruta."
                )
            self._missing_since = self._missing_since or time.time()
            self._last_discovery = time.time()
            return False

        info = read_lockfile(path)
        if not info:
            self._last_error = f"Lockfile inválido: {path}"
            self._last_discovery = time.time()
            return False

        session = requests.Session()
        session.trust_env = False
        session.verify = False
        session.auth = ("riot", info.password)
        session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "RiftSense/1.0 (LCU read-only)",
            }
        )
        self._info = info
        self._session = session
        self._last_error = None
        self._missing_since = None
        self._last_discovery = time.time()
        return True

    def _ensure_connection(self) -> bool:
        if self._info is not None and self._session is not None:
            return True
        # No reintentar el escaneo mas de una vez cada 3 s
        if time.time() - self._last_discovery < 3.0:
            return False
        return self._rediscover()

    def reset(self) -> None:
        with self._lock:
            self._info = None
            self._session = None
            self._last_discovery = 0.0

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #
    def get_json(self, path: str, timeout: float = 2.0) -> Any:
        with self._lock:
            if not self._ensure_connection():
                return None
            assert self._info and self._session
            url = f"{self._info.protocol}://127.0.0.1:{self._info.port}{path}"
            try:
                response = self._session.get(url, timeout=timeout)
            except requests.RequestException as exc:
                self._last_error = f"Cliente no accesible: {type(exc).__name__}"
                self._info = None
                self._session = None
                return None

            if response.status_code == 404:
                return None
            if response.status_code == 401:
                self._last_error = "El cliente rechazó las credenciales del lockfile."
                self._info = None
                self._session = None
                return None
            if response.status_code >= 500:
                self._last_error = f"LCU devolvio HTTP {response.status_code}."
                return None
            try:
                response.raise_for_status()
                return response.json() if response.content else None
            except (requests.RequestException, ValueError) as exc:
                self._last_error = f"Respuesta LCU invalida: {type(exc).__name__}"
                return None

    def phase(self) -> str | None:
        value = self.get_json("/lol-gameflow/v1/gameflow-phase")
        return value if isinstance(value, str) else None

    def current_summoner(self) -> dict[str, Any] | None:
        value = self.get_json("/lol-summoner/v1/current-summoner")
        return value if isinstance(value, dict) else None

    def champ_select_session(self) -> dict[str, Any] | None:
        value = self.get_json("/lol-champ-select/v1/session")
        return value if isinstance(value, dict) else None

    def gameflow_session(self) -> dict[str, Any] | None:
        value = self.get_json("/lol-gameflow/v1/session")
        return value if isinstance(value, dict) else None

    # ------------------------------------------------------------------ #
    # Live Client Data API
    # ------------------------------------------------------------------ #
    def live_gamedata(self, timeout: float = 1.5) -> dict[str, Any] | None:
        with self._live_lock:
            if self._live_session is None:
                session = requests.Session()
                session.trust_env = False
                session.verify = False
                session.headers.update({"Accept": "application/json", "User-Agent": "RiftSense/1.0"})
                self._live_session = session
            for url in LIVE_URLS:
                try:
                    response = self._live_session.get(url, timeout=timeout)
                except requests.RequestException:
                    continue
                if response.status_code == 200:
                    try:
                        data = response.json()
                    except ValueError:
                        continue
                    return data if isinstance(data, dict) else None
            return None

    # ------------------------------------------------------------------ #
    # Estado
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        phase = self.phase()
        # En partida se consulta el Live API; tambien como fallback con un
        # timeout corto si el LCU esta cerrado pero el juego sigue corriendo.
        if phase in ("InProgress", "Reconnect", "WaitingForStats"):
            live = self.live_gamedata(timeout=1.2)
        elif phase in (None, "None"):
            live = self.live_gamedata(timeout=0.35)
        else:
            live = None
        summoner = self.current_summoner() if phase else None
        connected = phase is not None
        search = self.locator.status()
        lockfile_path = str(self._info.path) if self._info else search.get("lockfilePath")

        display_name = None
        if summoner:
            game_name = summoner.get("gameName") or summoner.get("displayName")
            tag = summoner.get("tagLine")
            display_name = f"{game_name}#{tag}" if game_name and tag else game_name

        if connected:
            error = None
        elif live is not None:
            error = None
        elif search.get("searching"):
            error = "Buscando League of Legends en los discos..."
        elif search.get("installPath"):
            error = (
                f"Cliente encontrado en el disco {search.get('drive') or '?'} "
                "pero está cerrado. Abre League of Legends."
            )
        else:
            error = self._last_error or "Abre League of Legends para detectar el cliente."

        effective_phase = phase
        if live is not None and phase in (None, "None"):
            effective_phase = "InProgress"

        return {
            "connected": connected,
            "phase": effective_phase or ("InProgress" if live else None),
            "summoner": display_name,
            "summonerLevel": (summoner or {}).get("summonerLevel"),
            "lockfilePath": lockfile_path,
            "installPath": search.get("installPath"),
            "drive": search.get("drive"),
            "search": search,
            "liveClient": live is not None,
            "inGame": phase == "InProgress" or live is not None,
            "champSelect": phase == "ChampSelect",
            "error": error,
            "checkedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
