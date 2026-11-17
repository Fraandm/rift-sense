"""Descarga y normaliza los datos estaticos de Data Dragon (campeones, objetos...)."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

import requests

from .. import config
from ..utils import get_json, make_session, write_json

CHAMPION_FILES = ("champion.json",)
ITEM_FILE = "item.json"


def latest_version(session: requests.Session, *, include_pbe: bool = False) -> str:
    versions: list[str] = get_json(
        session, config.DDRAGON_VERSIONS, cache_key="ddragon_versions", cache_ttl=3600
    )
    for version in versions:
        if include_pbe or "pbe" not in version.lower():
            return version
    return versions[0]


def fetch_static(version: str | None = None) -> tuple[str, Path]:
    """Descarga los JSON crudos de un parche concreto y devuelve (version, carpeta)."""
    session = make_session()
    version = version or latest_version(session)
    out_dir = config.STATIC_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "champion.json": "champion.json",
        "item.json": "item.json",
        "summoner.json": "summoner.json",
        "runesReforged.json": "runesReforged.json",
    }
    for remote, local in datasets.items():
        url = f"{config.DDRAGON_BASE}/cdn/{version}/data/en_US/{remote}"
        data = get_json(session, url, cache_key=f"ddragon_{version}_{local}")
        write_json(out_dir / local, data)
        print(f"  [ddragon] {local}: OK")
    return version, out_dir


def _damage_profile(info: dict[str, Any]) -> str:
    attack = float(info.get("attack", 0) or 0)
    magic = float(info.get("magic", 0) or 0)
    if attack - magic >= 3:
        return "AD"
    if magic - attack >= 3:
        return "AP"
    return "MIXTO"


def build_meta(version: str, static_dir: Path | None = None) -> dict[str, Any]:
    static_dir = static_dir or config.STATIC_DIR / version
    champions_raw = _load(static_dir / "champion.json")["data"]
    items_raw = _load(static_dir / "item.json")["data"]
    spells_raw = _load(static_dir / "summoner.json").get("data", {})
    runes_raw = _load(static_dir / "runesReforged.json")

    champions: list[dict[str, Any]] = []
    for champ in champions_raw.values():
        info = champ.get("info", {})
        champions.append(
            {
                "id": champ["id"],
                "key": int(champ["key"]),
                "name": champ["name"],
                "slug": champ["id"],
                "icon": champ["image"]["full"],
                "tags": champ.get("tags", []),
                "damage": _damage_profile(info),
                "info": {
                    "attack": info.get("attack", 0),
                    "defense": info.get("defense", 0),
                    "magic": info.get("magic", 0),
                    "difficulty": info.get("difficulty", 0),
                },
                "partype": champ.get("partype", ""),
            }
        )
    champions.sort(key=lambda c: c["name"])

    items: list[dict[str, Any]] = []
    for item_id, item in items_raw.items():
        tags = item.get("tags", [])
        maps = item.get("maps", {})
        gold = item.get("gold", {})
        purchasable = bool(gold.get("purchasable", False))
        in_store = item.get("inStore", True)
        total = int(gold.get("total", 0) or 0)
        excluded = {"Consumable", "Trinket", "GoldPer"}.intersection(tags)
        trainable = (
            purchasable
            and in_store
            and maps.get("11", False)
            and total >= 300
            and not excluded
            and "Turret" not in tags
        )
        items.append(
            {
                "id": int(item_id),
                "name": item["name"],
                "icon": item["image"]["full"],
                "gold": total,
                "base": int(gold.get("base", 0) or 0),
                "tags": tags,
                "stats": {k: v for k, v in (item.get("stats") or {}).items() if v},
                "from": [int(x) for x in item.get("from", [])],
                "into": [int(x) for x in item.get("into", [])],
                "depth": int(item.get("depth", 0) or 0),
                "description": _strip_html(item.get("description", "")),
                "plaintext": item.get("plaintext", ""),
                "boots": "Boots" in tags,
                "trainable": trainable,
                "maps": {k: bool(v) for k, v in maps.items() if k in {"11", "12"}},
            }
        )
    items.sort(key=lambda i: i["gold"])

    spells = [
        {
            "id": s["id"],
            "key": s["key"],
            "name": s["name"],
            "icon": s["image"]["full"],
            "description": _strip_html(s.get("description", "")),
        }
        for s in spells_raw.values()
    ]

    rune_trees = [
        {
            "id": tree["id"],
            "key": tree["key"],
            "name": tree["name"],
            "icon": tree["icon"],
            "slots": [
                [
                    {"id": rune["id"], "key": rune["key"], "name": rune["name"], "icon": rune["icon"]}
                    for rune in slot["runes"]
                ]
                for slot in tree["slots"]
            ],
        }
        for tree in runes_raw
    ]

    meta = {
        "ddragonVersion": version,
        "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "champions": champions,
        "items": items,
        "spells": spells,
        "runeTrees": rune_trees,
        "roles": config.ROLES,
        "roleLabels": config.ROLE_LABELS,
        "iconBase": f"{config.DDRAGON_BASE}/cdn/{version}/img",
    }
    write_json(config.META_PATH, meta)
    return meta


def _load(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _strip_html(text: str) -> str:
    import re

    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


def ensure_meta(version: str | None = None, *, force: bool = False) -> dict[str, Any]:
    if config.META_PATH.exists() and not force:
        meta = _load(config.META_PATH)
        if not version or meta.get("ddragonVersion") == version:
            return meta
    version = version or latest_version(make_session())
    if not (config.STATIC_DIR / version / "item.json").exists() or force:
        version, _ = fetch_static(version)
    return build_meta(version)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga datos estaticos de Data Dragon")
    parser.add_argument("--version", default=None, help="Version de DDragon (por defecto la ultima)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    meta = ensure_meta(args.version, force=args.force)
    trainable = sum(1 for i in meta["items"] if i["trainable"])
    print(
        f"Data Dragon {meta['ddragonVersion']}: "
        f"{len(meta['champions'])} campeones, {len(meta['items'])} objetos "
        f"({trainable} entrenables) -> {config.META_PATH}"
    )


if __name__ == "__main__":
    main()
