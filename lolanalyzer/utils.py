"""Utilidades compartidas: HTTP con reintentos, JSON atomico y normalizacion."""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
import unicodedata
import html
from pathlib import Path
from typing import Any, Iterable, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import CACHE_DIR


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def make_session(user_agent: str | None = None) -> requests.Session:
    ua = user_agent or os.environ.get(
        "LOL_USER_AGENT",
        "RiftSense/1.0 (proyecto de analisis; contacto: "
        + os.environ.get("LP_CONTACT", "sin-contacto@example.com")
        + ")",
    )
    session = requests.Session()
    session.headers.update({"User-Agent": ua, "Accept": "application/json"})
    retry = Retry(
        total=6,
        connect=4,
        read=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: int = 60,
    cache_key: str | None = None,
    cache_ttl: float | None = None,
) -> Any:
    """GET JSON con cache opcional en disco (util para datos estaticos)."""
    cache_file: Path | None = None
    if cache_key:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", cache_key)[:180]
        cache_file = CACHE_DIR / f"{safe}.json"
        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if cache_ttl is None or age < cache_ttl:
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass

    resp = session.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if cache_file is not None:
        write_json(cache_file, data)
    return data


# --------------------------------------------------------------------------- #
# JSON / ficheros
# --------------------------------------------------------------------------- #
def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=None, separators=(",", ":"))
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
            n += 1
    return n


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# --------------------------------------------------------------------------- #
# Normalizacion
# --------------------------------------------------------------------------- #
_APOS = re.compile(r"['\u2019`]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm_name(value: str | None) -> str:
    """Normaliza nombres: 'Kai'Sa' -> 'kaisa', 'Nunu & Willump' -> 'nunuwillump'."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _APOS.sub("", text).lower()
    return _NON_ALNUM.sub("", text)


def clean(value: Any) -> str:
    """Limpia celdas tipo 'None', 'N/A', '' de Leaguepedia."""
    if value is None:
        return ""
    text = html.unescape(str(value)).strip()
    if text.lower() in {"none", "n/a", "na", "null", "-", "nan", "tbd"}:
        return ""
    return text


def split_list(value: Any, sep: str = ",") -> list[str]:
    if isinstance(value, (list, tuple)):
        return [item for item in (clean(v) for v in value) if item]
    text = clean(value)
    if not text:
        return []
    return [part.strip() for part in text.split(sep) if part.strip() and clean(part)]


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).replace(",", ".")))
    except (TypeError, ValueError):
        return default


def parse_patch(value: str | None) -> tuple[int, int]:
    """Convierte '26.22' o '15.S1.3' en una tupla comparable."""
    if not value:
        return (0, 0)
    nums = [int(n) for n in re.findall(r"\d+", str(value))]
    if not nums:
        return (0, 0)
    if len(nums) == 1:
        return (nums[0], 0)
    # 15.S1.3 -> (15, 3); 26.22 -> (26, 22)
    return (nums[0], nums[-1])


def patch_tuple_at_least(value: str | None, minimum: str) -> bool:
    return parse_patch(value) >= parse_patch(minimum)


def chunked(seq: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
