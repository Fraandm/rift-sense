"""Scraper del dataset profesional de Leaguepedia (API Cargo de lol.fandom.com).

Descarga, de forma reanudable y con cache en disco:
  * ScoreboardGames   -> una fila por partida (picks, bans, ganador, objetivos...)
  * ScoreboardPlayers -> una fila por jugador y partida (campeon, rol, items, KDA...)

Uso:
    python -m lolanalyzer.scraper.leaguepedia --since 2024-01-01
    python -m lolanalyzer.scraper.leaguepedia --only games
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path
from typing import Any

from .. import config
from ..utils import append_jsonl, make_session, parse_int, read_json, write_json

URL = f"{config.LEAGUEPEDIA_API}"
EXPORT_URL = "https://lol.fandom.com/wiki/Special:CargoExport"
RAW = config.RAW_DIR / "leaguepedia"

GAME_FIELDS = ",".join(
    [
        "ScoreboardGames.GameId=GameId",
        "ScoreboardGames.MatchId=MatchId",
        "ScoreboardGames.OverviewPage=OverviewPage",
        "ScoreboardGames.Tournament=Tournament",
        "ScoreboardGames.Team1=Team1",
        "ScoreboardGames.Team2=Team2",
        "ScoreboardGames.WinTeam=WinTeam",
        "ScoreboardGames.Winner=Winner",
        "ScoreboardGames.DateTime_UTC=DateUtc",
        "ScoreboardGames.Patch=Patch",
        "ScoreboardGames.Gamelength_Number=Gamelength",
        "ScoreboardGames.Team1Bans=Team1Bans",
        "ScoreboardGames.Team2Bans=Team2Bans",
        "ScoreboardGames.Team1Picks=Team1Picks",
        "ScoreboardGames.Team2Picks=Team2Picks",
        "ScoreboardGames.Team1Kills=Team1Kills",
        "ScoreboardGames.Team2Kills=Team2Kills",
        "ScoreboardGames.Team1Gold=Team1Gold",
        "ScoreboardGames.Team2Gold=Team2Gold",
        "ScoreboardGames.Team1Towers=Team1Towers",
        "ScoreboardGames.Team2Towers=Team2Towers",
        "ScoreboardGames.Team1Dragons=Team1Dragons",
        "ScoreboardGames.Team2Dragons=Team2Dragons",
        "ScoreboardGames.Team1Barons=Team1Barons",
        "ScoreboardGames.Team2Barons=Team2Barons",
        "ScoreboardGames.Team1Inhibitors=Team1Inhibitors",
        "ScoreboardGames.Team2Inhibitors=Team2Inhibitors",
        "ScoreboardGames.Team1RiftHeralds=Team1RiftHeralds",
        "ScoreboardGames.Team2RiftHeralds=Team2RiftHeralds",
        "ScoreboardGames.Team1VoidGrubs=Team1VoidGrubs",
        "ScoreboardGames.Team2VoidGrubs=Team2VoidGrubs",
        "ScoreboardGames.Team1Atakhans=Team1Atakhans",
        "ScoreboardGames.Team2Atakhans=Team2Atakhans",
        "ScoreboardGames.RiotPlatformGameId=RiotGameId",
    ]
)

PLAYER_FIELDS = ",".join(
    [
        "ScoreboardPlayers.GameId=GameId",
        "ScoreboardPlayers.MatchId=MatchId",
        "ScoreboardPlayers.Tournament=Tournament",
        "ScoreboardPlayers.Name=Player",
        "ScoreboardPlayers.Link=PlayerLink",
        "ScoreboardPlayers.Champion=Champion",
        "ScoreboardPlayers.Role=Role",
        "ScoreboardPlayers.Kills=Kills",
        "ScoreboardPlayers.Deaths=Deaths",
        "ScoreboardPlayers.Assists=Assists",
        "ScoreboardPlayers.Gold=Gold",
        "ScoreboardPlayers.CS=CS",
        "ScoreboardPlayers.DamageToChampions=Damage",
        "ScoreboardPlayers.VisionScore=VisionScore",
        "ScoreboardPlayers.Items=Items",
        "ScoreboardPlayers.Trinket=Trinket",
        "ScoreboardPlayers.KeystoneRune=Keystone",
        "ScoreboardPlayers.PrimaryTree=PrimaryTree",
        "ScoreboardPlayers.SecondaryTree=SecondaryTree",
        "ScoreboardPlayers.SummonerSpells=SummonerSpells",
        "ScoreboardPlayers.Team=Team",
        "ScoreboardPlayers.TeamVs=TeamVs",
        "ScoreboardPlayers.Side=Side",
        "ScoreboardPlayers.PlayerWin=PlayerWin",
        "ScoreboardPlayers.DateTime_UTC=DateUtc",
    ]
)


def cargo_query(
    session,
    table: str,
    fields: str,
    where: str,
    order_by: str,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    params = {
        "action": "cargoquery",
        "format": "json",
        "tables": table,
        "fields": fields,
        "where": where,
        "order_by": order_by,
        "limit": str(limit),
        "offset": str(offset),
    }
    delay = 5.0
    for attempt in range(8):
        resp = session.get(URL, params=params, timeout=90)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After") or delay)
            print(f"  [rate-limit 429] esperando {wait:.0f}s...", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 120)
            continue
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            info = str(payload["error"].get("info", ""))
            if "rate limit" in info.lower() or payload["error"].get("code") == "ratelimited":
                print(f"  [rate-limit cargo] esperando {delay:.0f}s...", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise RuntimeError(f"Cargo error: {payload['error']}")
        rows = payload.get("cargoquery", [])
        return [row.get("title", {}) for row in rows]
    raise RuntimeError("Leaguepedia sigue limitando las peticiones. Espera unos minutos y reintenta.")


def count_rows(session, table: str, where: str) -> int | None:
    try:
        rows = cargo_query(session, table, "COUNT(*)=Count", where, "", 1, 0)
        return parse_int(rows[0].get("Count")) if rows else None
    except Exception:
        return None


def cargo_export(
    session,
    table: str,
    fields: str,
    where: str,
    order_by: str,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    """Descarga masiva via Special:CargoExport (hasta 5.000 filas por peticion)."""
    params = {
        "tables": table,
        "fields": fields,
        "where": where,
        "order_by": order_by,
        "limit": str(limit),
        "offset": str(offset),
        "format": "json",
    }
    delay = 4.0
    for attempt in range(8):
        resp = session.get(EXPORT_URL, params=params, timeout=180)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After") or delay)
            print(f"  [rate-limit 429] esperando {wait:.0f}s...", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 120)
            continue
        resp.raise_for_status()
        text = resp.text.strip()
        if not text:
            return []
        if text[0] not in "[{":
            lowered = text.lower()
            if "rate limit" in lowered or "ratelimit" in lowered:
                print(f"  [rate-limit export] esperando {delay:.0f}s...", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise RuntimeError(f"Respuesta inesperada de CargoExport: {text[:300]}")
        payload = resp.json()
        if isinstance(payload, dict):
            if "error" in payload:
                raise RuntimeError(f"CargoExport error: {payload['error']}")
            return []
        return payload
    raise RuntimeError("CargoExport sigue limitando las peticiones. Espera unos minutos.")


def scrape_table(
    session,
    *,
    table: str,
    fields: str,
    where: str,
    order_by: str,
    out_path: Path,
    page_size: int = 5000,
    max_pages: int | None = None,
    restart: bool = False,
    sleep: float = 0.6,
    engine: str = "export",
) -> int:
    ckpt_path = RAW / "checkpoint.json"
    ckpt = read_json(ckpt_path, {}) or {}
    key = f"{engine}|{table}|{where}|{order_by}"
    if restart:
        for old_key in [k for k in ckpt if k.endswith(f"{table}|{where}|{order_by}")]:
            ckpt.pop(old_key, None)
        if out_path.exists():
            out_path.unlink()
    state = ckpt.get(key, {"offset": 0, "rows": 0})
    offset = int(state.get("offset", 0))
    total_written = int(state.get("rows", 0))

    total = count_rows(session, table, where) if engine == "api" else None
    print(
        f"[{table}] filas estimadas: {total if total is not None else '?'} | "
        f"motor {engine} | reanudando en offset {offset}"
    )

    fetch = (
        lambda *a: cargo_export(session, *a)
        if engine == "export"
        else cargo_query(session, *a)
    )

    started = time.time()
    pages = 0
    while True:
        if max_pages is not None and pages >= max_pages:
            print(f"[{table}] limite de paginas alcanzado ({max_pages})")
            break
        rows = fetch(table, fields, where, order_by, page_size, offset)
        if not rows:
            print(f"[{table}] fin de datos en offset {offset}")
            break
        written = append_jsonl(out_path, rows)
        offset += len(rows)
        total_written += written
        pages += 1
        ckpt[key] = {"offset": offset, "rows": total_written, "updated": dt.datetime.now().isoformat()}
        write_json(ckpt_path, ckpt)

        elapsed = max(time.time() - started, 1e-6)
        rate = (offset - int(state.get("offset", 0))) / elapsed
        eta = ""
        if total:
            remaining = max(total - offset, 0)
            eta = f" | ETA {remaining / max(rate, 1e-6) / 60:.1f} min"
        print(
            f"[{table}] {offset}/{total if total is not None else '?'} filas "
            f"({rate:.0f} filas/s){eta}",
            flush=True,
        )
        if total is not None and offset >= total:
            break
        if len(rows) < page_size:
            break
        time.sleep(sleep)

    return total_written


def scrape(since: str, until: str, only: str = "both", max_pages: int | None = None,
           restart: bool = False, engine: str = "export") -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    session = make_session()
    game_where = (
        f"ScoreboardGames.DateTime_UTC >= '{since}' "
        f"AND ScoreboardGames.DateTime_UTC < '{until}'"
    )
    player_where = (
        f"ScoreboardPlayers.DateTime_UTC >= '{since}' "
        f"AND ScoreboardPlayers.DateTime_UTC < '{until}'"
    )

    started = time.time()
    if only in {"both", "games"}:
        scrape_table(
            session,
            table="ScoreboardGames",
            fields=GAME_FIELDS,
            where=game_where,
            order_by="ScoreboardGames.DateTime_UTC DESC",
            out_path=RAW / "games.jsonl",
            max_pages=max_pages,
            restart=restart,
            engine=engine,
        )
    if only in {"both", "players"}:
        scrape_table(
            session,
            table="ScoreboardPlayers",
            fields=PLAYER_FIELDS,
            where=player_where,
            order_by="ScoreboardPlayers.DateTime_UTC DESC",
            out_path=RAW / "players.jsonl",
            page_size=5000,
            max_pages=max_pages,
            restart=restart,
            engine=engine,
            sleep=1.0,
        )

    summary = {
        "since": since,
        "until": until,
        "finishedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsedMinutes": round((time.time() - started) / 60, 2),
        "gamesFile": str(RAW / "games.jsonl"),
        "playersFile": str(RAW / "players.jsonl"),
    }
    write_json(RAW / "summary.json", summary)
    print(f"Scraping completado en {summary['elapsedMinutes']} min -> {RAW}")


def main() -> None:
    today = dt.date.today()
    parser = argparse.ArgumentParser(description="Descarga partidas pro de Leaguepedia")
    parser.add_argument("--since", default=config.DEFAULT_SINCE, help="Fecha inicial YYYY-MM-DD")
    parser.add_argument("--until", default=str(today + dt.timedelta(days=1)), help="Fecha final YYYY-MM-DD")
    parser.add_argument("--only", choices=["both", "games", "players"], default="both")
    parser.add_argument("--max-pages", type=int, default=None, help="Limite de paginas (pruebas)")
    parser.add_argument("--restart", action="store_true", help="Borra y empieza de cero")
    parser.add_argument(
        "--engine",
        choices=["export", "api"],
        default="export",
        help="export = Special:CargoExport (5.000 filas/peticion); api = cargoquery (500)",
    )
    args = parser.parse_args()
    scrape(args.since, args.until, args.only, args.max_pages, args.restart, args.engine)


if __name__ == "__main__":
    main()
