"""Rutas y configuracion global del proyecto."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Carga simple de .env (sin dependencias) para RIOT_API_KEY, LP_CONTACT, etc.
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    try:
        for _line in _env_file.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _value = _line.split("=", 1)
            os.environ.setdefault(_key.strip(), _value.strip().strip('"').strip("'"))
    except OSError:
        pass

DATA_DIR = Path(os.environ.get("LOL_DATA_DIR", BASE_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STATIC_DIR = DATA_DIR / "static"
CACHE_DIR = DATA_DIR / "cache"

ARTIFACTS_DIR = Path(os.environ.get("LOL_ARTIFACTS_DIR", BASE_DIR / "artifacts"))
WEB_DIR = BASE_DIR / "web"

for _p in (DATA_DIR, RAW_DIR, PROCESSED_DIR, STATIC_DIR, CACHE_DIR, ARTIFACTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --- Fuentes -----------------------------------------------------------------
LEAGUEPEDIA_API = "https://lol.fandom.com/api.php"
DDRAGON_BASE = "https://ddragon.leagueoflegends.com"
DDRAGON_VERSIONS = f"{DDRAGON_BASE}/api/versions.json"

# --- Dataset -----------------------------------------------------------------
DEFAULT_SINCE = os.environ.get("LP_SINCE", "2024-01-01")
# Las builds cambian con los parches: por defecto solo se entrena con lo reciente.
BUILD_MIN_PATCH = os.environ.get("BUILD_MIN_PATCH", "25.10")

ROLES = ["TOP", "JUNGLE", "MID", "BOT", "SUPPORT"]
ROLE_LABELS = {
    "TOP": "Top",
    "JUNGLE": "Jungla",
    "MID": "Mid",
    "BOT": "Bot / ADC",
    "SUPPORT": "Support",
}

# --- Modelos -----------------------------------------------------------------
DRAFT_MODEL_PATH = ARTIFACTS_DIR / "draft_net.pt"
BUILD_MODEL_PATH = ARTIFACTS_DIR / "build_net.pt"
META_PATH = ARTIFACTS_DIR / "meta.json"
PRIORS_PATH = PROCESSED_DIR / "priors.json"
DRAFT_REPORT_PATH = ARTIFACTS_DIR / "draft_report.json"
BUILD_REPORT_PATH = ARTIFACTS_DIR / "build_report.json"

RANDOM_SEED = 1337
