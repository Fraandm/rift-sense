"""Adaptador de la Riot API para partidas de high elo (Challenger/GM/Master).

Genera JSONL compatibles con el pipeline de Leaguepedia, de modo que puedes
mezclar partidas profesionales y soloQ de alto elo:

    python -m lolanalyzer.scraper.riot_api --region euw1 --players 25 --matches 20

Requiere RIOT_API_KEY (https://developer.riotgames.com/). Las claves de
desarrollo caducan cada 24 h. El adaptador tambien guarda los tiempos exactos
de compra de la timeline en `PurchaseTimes`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from pathlib import Path
from typing import Any

from .. import config
from ..utils import (
    append_jsonl,
    clean,
    iter_jsonl,
    make_session,
    parse_int,
    read_json,
    write_json,
)

PLATFORM_HOSTS = {
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "me1": "europe",
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "oc1": "americas",
    "kr": "asia",
    "jp1": "asia",
    "ph2": "sea",
    "sg2": "sea",
    "th2": "sea",
    "tw2": "sea",
    "vn2": "sea",
}

ROLE_MAP = {
    "TOP": "Top",
    "JUNGLE": "Jungle",
    "MIDDLE": "Mid",
    "BOTTOM": "Bot",
    "UTILITY": "Support",
}

RIOT_DIR = config.RAW_DIR / "riot"
CACHE_DIR = config.CACHE_DIR / "riot"


def load_dotenv() -> None:
    env_path = config.BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class RiotClient:
    def __init__(self, region: str, api_key: str, min_interval: float = 0.08):
        self.region = region
        self.regional = PLATFORM_HOSTS.get(region)
        if not self.regional:
            raise ValueError(f"Region no soportada: {region}")
        self.session = make_session()
        self.session.headers.update({"X-Riot-Token": api_key})
        self.min_interval = min_interval
        self.last_call = 0.0
        self.calls = 0

    def _wait(self) -> None:
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()

    def get(self, path: str, *, regional: bool = False, allow_404: bool = True) -> Any:
        host = self.regional if regional else self.region
        url = f"https://{host}.api.riotgames.com{path}"
        delay = 3.0
        for attempt in range(8):
            self._wait()
            resp = self.session.get(url, timeout=30)
            self.calls += 1
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After") or delay)
                print(f"  [riot 429] esperando {wait:.0f}s...", flush=True)
                time.sleep(wait)
                delay = min(delay * 2, 120)
                continue
            if resp.status_code == 404 and allow_404:
                return None
            if resp.status_code == 403:
                raise RuntimeError(
                    "Riot API devolvio 403: clave caducada o invalida. "
                    "Renueva RIOT_API_KEY en developer.riotgames.com."
                )
            if resp.status_code == 401:
                raise RuntimeError("Riot API devolvio 401: falta RIOT_API_KEY.")
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("Riot API sigue limitando las peticiones.")

    def apex_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for tier in ("challengerleagues", "grandmasterleagues", "masterleagues"):
            payload = self.get(f"/lol/league/v4/{tier}/by-queue/RANKED_SOLO_5x5")
            if payload:
                entries.extend(payload.get("entries", []))
        return entries

    def match_ids(self, puuid: str, count: int) -> list[str]:
        data = self.get(
            f"/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&type=ranked&count={count}",
            regional=True,
        )
        return data or []

    def match(self, match_id: str) -> dict[str, Any] | None:
        return self.get(f"/lol/match/v5/matches/{match_id}", regional=True)

    def timeline(self, match_id: str) -> dict[str, Any] | None:
        return self.get(f"/lol/match/v5/matches/{match_id}/timeline", regional=True)


def champion_maps(meta: dict[str, Any]) -> tuple[dict[int, str], dict[int, dict[str, Any]], dict[int, str]]:
    champs = {int(c["key"]): c["name"] for c in meta["champions"]}
    champ_data = {int(c["key"]): c for c in meta["champions"]}
    items_by_id = {int(i["id"]): i for i in meta["items"]}
    spell_ids = {}
    for spell in meta.get("spells", []):
        key = parse_int(spell.get("key"))
        if key:
            spell_ids[key] = spell.get("name", "")
    return champs, {"by_id": items_by_id, "champ_data": champ_data}, spell_ids


def purchases_by_participant(timeline: dict[str, Any], trainable_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    if not timeline:
        return result
    seen: dict[int, set[int]] = {}
    for frame in timeline.get("info", {}).get("frames", []):
        for event in frame.get("events", []):
            if event.get("type") != "ITEM_PURCHASED":
                continue
            pid = int(event.get("participantId", 0))
            item_id = int(event.get("itemId", 0))
            if item_id not in trainable_ids:
                continue
            seen.setdefault(pid, set())
            if item_id in seen[pid]:
                continue
            seen[pid].add(item_id)
            result.setdefault(pid, []).append(
                {
                    "itemId": item_id,
                    "minute": round(float(event.get("timestamp", 0)) / 60000.0, 2),
                }
            )
    return result


def convert_match(
    match: dict[str, Any],
    timeline: dict[str, Any] | None,
    region: str,
    meta: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    info = match.get("info", {})
    if info.get("gameMode") != "CLASSIC" or int(info.get("mapId", 0)) != 11:
        return None
    if int(info.get("queueId", 0)) != 420:
        return None
    duration = float(info.get("gameDuration", 0) or 0)
    if duration < 600:
        return None
    participants = info.get("participants", [])
    if len(participants) != 10:
        return None

    champs, helpers, spell_ids = champion_maps(meta)
    items_by_id = helpers["by_id"]
    champ_data = helpers["champ_data"]
    trainable_ids = {int(i["id"]) for i in meta["items"] if i.get("trainable")}

    teams = {t.get("teamId"): t for t in info.get("teams", [])}
    blue = teams.get(100) or {}
    red = teams.get(200) or {}
    blue_win = bool(blue.get("win"))
    win_team = "Blue" if blue_win else "Red"
    creation = dt.datetime.fromtimestamp(int(info.get("gameCreation", 0)) / 1000, tz=dt.timezone.utc)
    version = clean(info.get("gameVersion"))
    parts = version.split(".")
    patch = ".".join(parts[:2]) if len(parts) >= 2 else ""

    by_pid = {int(p.get("participantId", 0)): p for p in participants}
    purchases = purchases_by_participant(timeline, trainable_ids)
    match_id = clean(match.get("metadata", {}).get("matchId")) or clean(info.get("matchId"))

    def ordered_items(pid: int, p: dict[str, Any]) -> list[int]:
        final = [int(p.get(f"item{i}", 0) or 0) for i in range(7)]
        final = [i for i in final if i in trainable_ids]
        order = [int(x["itemId"]) for x in purchases.get(pid, [])]
        ordered = order + [i for i in final if i not in order]
        return ordered[:7]

    def role_of(p: dict[str, Any]) -> str:
        return ROLE_MAP.get(clean(p.get("teamPosition")), "")

    blue_picks: list[str] = []
    red_picks: list[str] = []
    for pid in range(1, 11):
        p = by_pid.get(pid)
        if not p:
            continue
        champ = champs.get(int(p.get("championId", 0)))
        role = role_of(p)
        if role:
            (blue_picks if int(p.get("teamId")) == 100 else red_picks).append(champ or "")

    game = {
        "GameId": match_id,
        "MatchId": match_id,
        "OverviewPage": f"HighElo/{region}",
        "Tournament": f"High Elo {region.upper()}",
        "Team1": "Blue",
        "Team2": "Red",
        "WinTeam": win_team,
        "Winner": 1 if blue_win else 2,
        "DateUtc": creation.strftime("%Y-%m-%d %H:%M:%S"),
        "Patch": patch,
        "Gamelength": round(duration / 60.0, 2),
        "Team1Bans": [champs.get(int(b.get("championId", 0)), "") for b in blue.get("bans", []) if int(b.get("championId", 0)) > 0],
        "Team2Bans": [champs.get(int(b.get("championId", 0)), "") for b in red.get("bans", []) if int(b.get("championId", 0)) > 0],
        "Team1Picks": blue_picks,
        "Team2Picks": red_picks,
        "Team1Kills": sum(int(p.get("kills", 0)) for p in participants if int(p.get("teamId")) == 100),
        "Team2Kills": sum(int(p.get("kills", 0)) for p in participants if int(p.get("teamId")) == 200),
        "Team1Gold": sum(int(p.get("goldEarned", 0)) for p in participants if int(p.get("teamId")) == 100),
        "Team2Gold": sum(int(p.get("goldEarned", 0)) for p in participants if int(p.get("teamId")) == 200),
        "Team1Towers": int(blue.get("objectives", {}).get("tower", {}).get("kills", 0)),
        "Team2Towers": int(red.get("objectives", {}).get("tower", {}).get("kills", 0)),
        "Team1Dragons": int(blue.get("objectives", {}).get("dragon", {}).get("kills", 0)),
        "Team2Dragons": int(red.get("objectives", {}).get("dragon", {}).get("kills", 0)),
        "Team1Barons": int(blue.get("objectives", {}).get("baron", {}).get("kills", 0)),
        "Team2Barons": int(red.get("objectives", {}).get("baron", {}).get("kills", 0)),
        "Team1Inhibitors": int(blue.get("objectives", {}).get("inhibitor", {}).get("kills", 0)),
        "Team2Inhibitors": int(red.get("objectives", {}).get("inhibitor", {}).get("kills", 0)),
        "Source": "riot",
    }

    player_rows: list[dict[str, Any]] = []
    for pid in range(1, 11):
        p = by_pid.get(pid)
        if not p:
            continue
        team_id = int(p.get("teamId", 0))
        items = ordered_items(pid, p)
        item_names = [items_by_id[i]["name"] for i in items if i in items_by_id]
        player_rows.append(
            {
                "GameId": match_id,
                "MatchId": match_id,
                "Tournament": game["Tournament"],
                "Player": clean(p.get("riotIdGameName")) or clean(p.get("summonerName")) or str(p.get("puuid", ""))[:10],
                "PlayerLink": clean(p.get("riotIdTagLine")),
                "Champion": champs.get(int(p.get("championId", 0)), ""),
                "Role": role_of(p),
                "Kills": int(p.get("kills", 0)),
                "Deaths": int(p.get("deaths", 0)),
                "Assists": int(p.get("assists", 0)),
                "Gold": int(p.get("goldEarned", 0)),
                "CS": int(p.get("totalMinionsKilled", 0)) + int(p.get("neutralMinionsKilled", 0)),
                "Damage": int(p.get("totalDamageDealtToChampions", 0)),
                "VisionScore": int(p.get("visionScore", 0)),
                "Items": item_names,
                "Trinket": "",
                "Keystone": "",
                "PrimaryTree": "",
                "SecondaryTree": "",
                "SummonerSpells": [
                    spell_ids.get(int(p.get("summoner1Id", 0)), ""),
                    spell_ids.get(int(p.get("summoner2Id", 0)), ""),
                ],
                "Team": "Blue" if team_id == 100 else "Red",
                "TeamVs": "Red" if team_id == 100 else "Blue",
                "Side": 1 if team_id == 100 else 2,
                "PlayerWin": "Yes" if bool(p.get("win")) else "No",
                "DateUtc": game["DateUtc"],
                "PurchaseTimes": [x for x in purchases.get(pid, [])][:12],
                "RiotPosition": clean(p.get("teamPosition")),
                "Source": "riot",
            }
        )
    return game, player_rows


def scrape_riot(
    region: str,
    *,
    players: int = 25,
    matches_per_player: int = 20,
    max_matches: int = 400,
    min_interval: float = 0.08,
    restart: bool = False,
) -> None:
    load_dotenv()
    api_key = os.environ.get("RIOT_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "Falta RIOT_API_KEY. Copia .env.example a .env o exporta la variable.\n"
            "Consigue una clave en https://developer.riotgames.com/"
        )
    meta = read_json(config.META_PATH)
    if not meta:
        raise SystemExit("Falta artifacts/meta.json. Ejecuta python -m lolanalyzer.scraper.ddragon")

    RIOT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    games_path = RIOT_DIR / "games.jsonl"
    players_path = RIOT_DIR / "players.jsonl"
    done_path = RIOT_DIR / "done.json"
    if restart:
        for path in (games_path, players_path, done_path):
            if path.exists():
                path.unlink()
    done = set(read_json(done_path, {"matches": []}).get("matches", [])) if done_path.exists() else set()

    client = RiotClient(region, api_key, min_interval=min_interval)
    print(f"Consultando apex leagues de {region}...")
    entries = client.apex_entries()
    entries = [e for e in entries if e.get("puuid")]
    entries.sort(key=lambda e: int(e.get("leaguePoints", 0)), reverse=True)
    chosen = entries[:players]
    print(f"  {len(entries):,} jugadores en apex | seleccionados {len(chosen)}")

    match_ids: list[str] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(chosen, 1):
        ids = client.match_ids(entry["puuid"], matches_per_player)
        for mid in ids:
            if mid not in seen_ids and mid not in done:
                seen_ids.add(mid)
                match_ids.append(mid)
        if i % 5 == 0:
            print(f"  ids: {i}/{len(chosen)} jugadores | {len(match_ids)} partidas nuevas")
    if len(match_ids) > max_matches:
        match_ids = match_ids[:max_matches]
    if not match_ids:
        print("No hay partidas nuevas. Prueba con --matches o --restart.")
        return

    print(f"Descargando {len(match_ids)} partidas + timelines...")
    processed = 0
    for i, mid in enumerate(match_ids, 1):
        cache_file = CACHE_DIR / f"{mid}.json"
        payload = read_json(cache_file)
        if not payload:
            match = client.match(mid)
            if not match:
                continue
            timeline = client.timeline(mid)
            payload = {"match": match, "timeline": timeline}
            write_json(cache_file, payload)
        converted = convert_match(payload["match"], payload.get("timeline"), region, meta)
        if converted:
            game, rows = converted
            append_jsonl(games_path, [game])
            append_jsonl(players_path, rows)
            processed += 1
            done.add(mid)
        if i % 10 == 0 or i == len(match_ids):
            write_json(done_path, {"matches": sorted(done)})
            print(f"  {i}/{len(match_ids)} | validas {processed}", flush=True)

    write_json(
        RIOT_DIR / "summary.json",
        {
            "region": region,
            "players": len(chosen),
            "requestedMatches": len(match_ids),
            "validMatches": processed,
            "apiCalls": client.calls,
            "finishedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    print(f"Listo: {processed} partidas high elo -> {RIOT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga partidas high elo de la Riot API")
    parser.add_argument("--region", default=os.environ.get("RIOT_REGION", "euw1"))
    parser.add_argument("--players", type=int, default=25, help="Jugadores apex a muestrear")
    parser.add_argument("--matches", type=int, default=20, help="Partidas por jugador")
    parser.add_argument("--max-matches", type=int, default=400)
    parser.add_argument("--interval", type=float, default=0.08, help="Segundos entre llamadas")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    scrape_riot(
        args.region,
        players=args.players,
        matches_per_player=args.matches,
        max_matches=args.max_matches,
        min_interval=args.interval,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
