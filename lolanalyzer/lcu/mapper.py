"""Traduce respuestas del LCU / Live Client Data al esquema de RiftSense."""
from __future__ import annotations

import datetime as dt
from functools import lru_cache
from typing import Any

from .. import config
from ..utils import norm_name, read_json

ROLES = ["TOP", "JUNGLE", "MID", "BOT", "SUPPORT"]

ROLE_MAP = {
    "top": "TOP",
    "jungle": "JUNGLE",
    "jgl": "JUNGLE",
    "jungler": "JUNGLE",
    "middle": "MID",
    "mid": "MID",
    "bottom": "BOT",
    "bot": "BOT",
    "adc": "BOT",
    "utility": "SUPPORT",
    "support": "SUPPORT",
    "sup": "SUPPORT",
}


@lru_cache(maxsize=1)
def _champions() -> dict[str, Any]:
    meta = read_json(config.META_PATH) or {}
    by_key: dict[int, str] = {}
    by_norm: dict[str, str] = {}
    for champ in meta.get("champions", []):
        try:
            by_key[int(champ["key"])] = champ["name"]
        except (KeyError, TypeError, ValueError):
            continue
        by_norm[norm_name(champ["name"])] = champ["name"]
    return {"by_key": by_key, "by_norm": by_norm}


@lru_cache(maxsize=1)
def _priors() -> dict[str, Any]:
    return read_json(config.PRIORS_PATH) or {}


@lru_cache(maxsize=1)
def _item_ids() -> set[int]:
    vocab = read_json(config.PROCESSED_DIR / "vocab.json") or {}
    return {int(key) for key in (vocab.get("items") or {})}


def champion_from_id(champion_id: Any) -> str | None:
    try:
        cid = int(champion_id or 0)
    except (TypeError, ValueError):
        return None
    if cid <= 0:
        return None
    return _champions()["by_key"].get(cid)


def canonical_champion(name: Any) -> str | None:
    if not name:
        return None
    return _champions()["by_norm"].get(norm_name(str(name)))


def infer_role(champion: str | None) -> str:
    if not champion:
        return "MID"
    data = (_priors().get("champions") or {}).get(champion) or {}
    roles = data.get("roles") or {}
    if not roles:
        return "MID"
    role = max(roles.items(), key=lambda kv: int((kv[1] or {}).get("n", 0)))[0]
    return role if role in ROLES else "MID"


def role_index(value: Any) -> int | None:
    if not value:
        return None
    role = ROLE_MAP.get(str(value).strip().lower().replace("_", ""))
    return ROLES.index(role) if role else None


def _fill_team_by_id(players: list[dict[str, Any]], position_key: str) -> list[str | None]:
    result: list[str | None] = [None] * 5
    for player in players or []:
        champion = champion_from_id(player.get("championId")) or champion_from_id(
            player.get("championPickIntent")
        )
        if not champion or champion in result:
            continue
        index = role_index(player.get(position_key))
        if index is None and player.get("cellId") is not None:
            try:
                index = int(player["cellId"]) % 5
            except (TypeError, ValueError):
                index = None
        if index is None or result[index] is not None:
            index = next((i for i, value in enumerate(result) if value is None), None)
        if index is not None:
            result[index] = champion
    return result


def _fill_team_by_name(players: list[dict[str, Any]]) -> list[str | None]:
    result: list[str | None] = [None] * 5
    for player in players or []:
        champion = canonical_champion(player.get("championName"))
        if not champion or champion in result:
            continue
        index = role_index(player.get("position"))
        if index is None or result[index] is not None:
            index = next((i for i, value in enumerate(result) if value is None), None)
        if index is not None:
            result[index] = champion
    return result


def _team_side(my_team: list[dict[str, Any]], gameflow: dict[str, Any] | None) -> str:
    for player in my_team or []:
        team = player.get("team")
        if team in (1, 2):
            return "blue" if int(team) == 1 else "red"
        if team in (100, 200):
            return "blue" if int(team) == 100 else "red"

    my_ids: set[Any] = set()
    for player in my_team or []:
        for key in ("summonerId", "puuid"):
            if player.get(key) is not None:
                my_ids.add(player[key])
    if not my_ids:
        return "blue"

    game_data = (gameflow or {}).get("gameData") or {}

    def ids_from(key: str) -> set[Any]:
        values = game_data.get(key) or []
        result: set[Any] = set()
        for value in values:
            if isinstance(value, dict):
                for field in ("summonerId", "puuid"):
                    if value.get(field) is not None:
                        result.add(value[field])
            elif value is not None:
                result.add(value)
        return result

    team_one = ids_from("teamOne")
    team_two = ids_from("teamTwo")
    in_one = bool(my_ids & team_one)
    in_two = bool(my_ids & team_two)
    if in_two and not in_one:
        return "red"
    return "blue"


def _map_bans(session: dict[str, Any]) -> list[str]:
    raw: list[Any] = []
    bans = session.get("bans") or {}
    for key in ("myTeamBans", "theirTeamBans"):
        raw.extend(bans.get(key) or [])
    for group in session.get("actions") or []:
        if not isinstance(group, list):
            continue
        for action in group:
            if not isinstance(action, dict):
                continue
            if action.get("type") == "ban" and action.get("completed"):
                raw.append(action.get("championId"))

    names: list[str] = []
    for champion_id in raw:
        name = champion_from_id(champion_id)
        if name and name not in names:
            names.append(name)
    return names[:10]


def _timer_seconds(session: dict[str, Any]) -> float | None:
    timer = session.get("timer") or {}
    value = timer.get("adjustedTimeLeftInPhase") or timer.get("timeLeftInPhase")
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number / 1000.0 if number > 1000 else number, 1)


def _place_champion(team: list[str | None], champion: str | None, role: str) -> None:
    if not champion:
        return
    if role not in ROLES:
        role = infer_role(champion)
    index = ROLES.index(role)
    # Evita duplicados si el campeon ya estaba colocado por posicion
    for i, value in enumerate(team):
        if value == champion and i != index:
            team[i] = None
    team[index] = champion


def map_champ_select(
    session: dict[str, Any],
    gameflow: dict[str, Any] | None = None,
    status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(session, dict):
        return {"available": False, "error": "Sesión de champ select no disponible."}

    my_team = session.get("myTeam") or []
    their_team = session.get("theirTeam") or []
    if not my_team:
        return {"available": False, "error": "El champ select no tiene equipo propio."}

    local_cell = session.get("localPlayerCellId")
    me = None
    for player in my_team:
        if local_cell is not None and player.get("cellId") == local_cell:
            me = player
            break
    me = me or my_team[0]

    my_champion = champion_from_id(me.get("championId")) or champion_from_id(
        me.get("championPickIntent")
    )
    role = ROLE_MAP.get(str(me.get("assignedPosition") or "").lower())
    if not role:
        role = infer_role(my_champion)

    allies = _fill_team_by_id(my_team, "assignedPosition")
    enemies = _fill_team_by_id(their_team, "assignedPosition")
    _place_champion(allies, my_champion, role)

    return {
        "available": True,
        "source": "lcu",
        "phase": (session.get("timer") or {}).get("phase") or (status or {}).get("phase"),
        "side": _team_side(my_team, gameflow),
        "role": role,
        "allies": allies,
        "enemies": enemies,
        "bans": _map_bans(session),
        "myChampion": my_champion,
        "localPlayerCellId": local_cell,
        "timerSeconds": _timer_seconds(session),
        "queueId": (((gameflow or {}).get("gameData") or {}).get("queue") or {}).get("queueId"),
        "summoner": (status or {}).get("summoner"),
        "updatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _same_player(active_name: str, player: dict[str, Any]) -> bool:
    candidates = {
        str(player.get("summonerName") or "").strip(),
        str(player.get("riotId") or "").strip(),
        str(player.get("riotIdGameName") or "").strip(),
    }
    if active_name in candidates:
        return True
    active_base = active_name.split("#")[0].strip().lower()
    for candidate in candidates:
        if candidate and candidate.split("#")[0].strip().lower() == active_base:
            return True
    return False


def map_live_gamedata(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"available": False, "error": "Live Client Data no disponible."}

    active = data.get("activePlayer") or {}
    players = data.get("allPlayers") or []
    active_name = str(active.get("summonerName") or active.get("riotIdGameName") or "").strip()
    if not active_name:
        return {"available": False, "error": "No se pudo identificar al jugador activo."}

    me = next((player for player in players if _same_player(active_name, player)), None)
    if me is None:
        return {
            "available": False,
            "error": "El jugador activo no aparece en la lista de jugadores.",
        }

    champion = canonical_champion(me.get("championName"))
    if not champion:
        return {
            "available": False,
            "error": f"Campeón no reconocido: {me.get('championName')}",
        }

    role = ROLES[role_index(me.get("position"))] if role_index(me.get("position")) is not None else infer_role(champion)
    team = str(me.get("team") or "").upper()
    side = "red" if team == "CHAOS" else "blue"
    team_players = [player for player in players if str(player.get("team") or "").upper() == team]
    enemy_players = [player for player in players if str(player.get("team") or "").upper() != team]

    allies = _fill_team_by_name(team_players)
    enemies = _fill_team_by_name(enemy_players)
    _place_champion(allies, champion, role)

    raw_items = [
        item
        for item in (me.get("items") or [])
        if isinstance(item, dict) and int(item.get("itemID", 0) or 0) > 0
    ]
    raw_items.sort(key=lambda item: int(item.get("slot", 0) or 0))
    valid_items = _item_ids()
    items = [
        {
            "id": int(item["itemID"]),
            "name": str(item.get("displayName") or ""),
            "price": int(item.get("price", 0) or 0),
            "slot": int(item.get("slot", 0) or 0),
        }
        for item in raw_items
    ]
    owned = [item["id"] for item in items if item["id"] in valid_items]

    game_data = data.get("gameData") or {}
    game_time = float(game_data.get("gameTime", 0) or 0)
    scores = me.get("scores") or {}

    return {
        "available": True,
        "source": "liveclientdata",
        "champion": champion,
        "role": role,
        "side": side,
        "allies": allies,
        "enemies": enemies,
        "bans": [],
        "owned": owned,
        "items": items,
        "gold": int(float(active.get("currentGold", 0) or 0)),
        "minute": round(game_time / 60.0, 1),
        "gameTime": round(game_time, 1),
        "level": int(me.get("level", 0) or 0),
        "scores": {
            "kills": int(scores.get("kills", 0) or 0),
            "deaths": int(scores.get("deaths", 0) or 0),
            "assists": int(scores.get("assists", 0) or 0),
            "cs": int(scores.get("creepScore", 0) or 0),
            "wards": int(scores.get("wardScore", 0) or 0),
        },
        "team": team,
        "gameMode": game_data.get("gameMode"),
        "mapName": game_data.get("mapName"),
        "summoner": active_name,
        "updatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
