"""Construye el dataset procesado (draft + builds) y las estadisticas previas.

Lee los JSONL de Leaguepedia, resuelve nombres contra Data Dragon, valida las
partidas y genera:
    data/processed/dataset.npz   matrices listas para entrenar
    data/processed/vocab.json    indices de campeones y objetos
    data/processed/priors.json   winrates/presencia suavizados (solo train)
    data/processed/summary.json  cobertura y estadisticas del dataset
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import statistics
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .. import config
from ..utils import (
    clean,
    iter_jsonl,
    norm_name,
    parse_float,
    parse_int,
    parse_patch,
    read_json,
    split_list,
    write_json,
)

RAW = config.RAW_DIR / "leaguepedia"
PROCESSED = config.PROCESSED_DIR

ROLE_MAP = {
    "top": "TOP",
    "jungle": "JUNGLE",
    "jungler": "JUNGLE",
    "mid": "MID",
    "middle": "MID",
    "bot": "BOT",
    "adc": "BOT",
    "ad carry": "BOT",
    "bottom": "BOT",
    "support": "SUPPORT",
    "supp": "SUPPORT",
    "utility": "SUPPORT",
}

# Nombres que requieren ayuda extra al cruzar Leaguepedia con Data Dragon.
CHAMPION_ALIASES = {
    "wukong": "Wukong",
    "nunu": "Nunu & Willump",
    "renata": "Renata Glasc",
    "jarvan": "Jarvan IV",
    "yi": "Master Yi",
    "mf": "Miss Fortune",
    "sol": "Aurelion Sol",
}

ITEM_ALIASES = {
    "ionianbootsoflucidity": "Ionian Boots of Lucidity",
    "mercurystreads": "Mercury's Treads",
    "platedsteelcaps": "Plated Steelcaps",
    "sorcerersshoes": "Sorcerer's Shoes",
    "berserkersgreaves": "Berserker's Greaves",
    "bootsofswiftness": "Boots of Swiftness",
    "zhonyashourglass": "Zhonya's Hourglass",
    "rabadonsdeathcap": "Rabadon's Deathcap",
    "rylaiscrystalscepter": "Rylai's Crystal Scepter",
    "liandrystorment": "Liandry's Torment",
    "ludenscompanion": "Luden's Companion",
    "youmuusghostblade": "Youmuu's Ghostblade",
    "lorddominiksregards": "Lord Dominik's Regards",
    "bansheesveil": "Banshee's Veil",
    "nashorstooth": "Nashor's Tooth",
    "guinsoosrageblade": "Guinsoo's Rageblade",
    "titanichydra": "Titanic Hydra",
    "ravenoushydra": "Ravenous Hydra",
    "profanehydra": "Profane Hydra",
    "kaenicrookern": "Kaenic Rookern",
    "deadmansplate": "Dead Man's Plate",
    "abyssalmask": "Abyssal Mask",
    "statikkshiv": "Statikk Shiv",
    "krakenslayer": "Kraken Slayer",
    "mortalreminder": "Mortal Reminder",
    "phantomdancer": "Phantom Dancer",
    "rapidfirecannon": "Rapid Firecannon",
    "navoriflickerblade": "Navori Flickerblade",
    "essencereaver": "Essence Reaver",
    "thecollector": "The Collector",
    "sunderedsky": "Sundered Sky",
    "spearofshojin": "Spear of Shojin",
    "blackcleaver": "Black Cleaver",
    "steraksgage": "Sterak's Gage",
    "trinityforce": "Trinity Force",
    "divinesunderer": "Divine Sunderer",
    "jaksho": "Jak'Sho, The Protean",
    "iceborngauntlet": "Iceborn Gauntlet",
    "frozenheart": "Frozen Heart",
    "randuinsomen": "Randuin's Omen",
    "spiritvisage": "Spirit Visage",
    "forceofnature": "Force of Nature",
    "knightsvow": "Knight's Vow",
    "zekesconvergence": "Zeke's Convergence",
    "mikaelsblessing": "Mikael's Blessing",
    "ardentcenser": "Ardent Censer",
    "staffofwater": "Staff of Flowing Water",
    "moonstonerenewer": "Moonstone Renewer",
    "shurelyasbattlesong": "Shurelya's Battlesong",
    "locketdiamond": "Locket of the Iron Solari",
    "imperialmandate": "Imperial Mandate",
    "echoesofhelia": "Echoes of Helia",
    "worldatlas": "World Atlas",
    "runiccompass": "Runic Compass",
    "bountyofworlds": "Bounty of Worlds",
    "tearofthegoddess": "Tear of the Goddess",
    "archangelsstaff": "Archangel's Staff",
    "ludenscompanion": "Luden's Echo",
    "ludenstempest": "Luden's Echo",
    "navoriquickblades": "Navori Flickerblade",
    "moonstone": "Moonstone Renewer",
}


def item_rank(item: dict[str, Any]) -> tuple[int, int, int, int]:
    """Prioriza variantes de la Grieta sobre las de Arena."""
    maps = item.get("maps") or {}
    item_id = int(item.get("id", 0))
    return (
        1 if item.get("trainable") else 0,
        1 if maps.get("11") else 0,
        1 if item_id < 100000 else 0,
        -item_id,
    )


class NameResolver:
    """Resuelve nombres a entradas de Data Dragon (exacto, alias y fuzzy).

    Data Dragon repite nombres entre variantes de Grieta y Arena. `prefer`
    decide que entrada gana cuando hay colision (por defecto, la primera).
    """

    def __init__(
        self,
        entries: list[dict[str, Any]],
        aliases: dict[str, str] | None = None,
        prefer: Any = None,
    ):
        self.by_norm: dict[str, dict[str, Any]] = {}
        self.prefer = prefer
        for entry in entries:
            self._add(norm_name(entry["name"]), entry)
            if entry.get("id"):
                self._add(norm_name(str(entry["id"])), entry)
        self.aliases = aliases or {}
        self.misses: Counter[str] = Counter()
        self.fuzzy_hits: Counter[str] = Counter()

    def _add(self, key: str, entry: dict[str, Any]) -> None:
        current = self.by_norm.get(key)
        if current is None:
            self.by_norm[key] = entry
            return
        if self.prefer and self.prefer(entry) > self.prefer(current):
            self.by_norm[key] = entry

    def resolve(self, name: str | None) -> dict[str, Any] | None:
        raw = clean(name)
        if not raw:
            return None
        key = norm_name(raw)
        if key in self.by_norm:
            return self.by_norm[key]
        alias = self.aliases.get(key)
        if alias:
            target = self.by_norm.get(norm_name(alias))
            if target:
                return target
        matches = difflib.get_close_matches(key, self.by_norm.keys(), n=1, cutoff=0.9)
        if matches:
            self.fuzzy_hits[f"{raw} -> {self.by_norm[matches[0]]['name']}"] += 1
            return self.by_norm[matches[0]]
        self.misses[raw] += 1
        return None


def load_meta() -> dict[str, Any]:
    meta = read_json(config.META_PATH)
    if not meta:
        raise SystemExit("Falta artifacts/meta.json. Ejecuta: python -m lolanalyzer.scraper.ddragon")
    return meta


def build_vocab(meta: dict[str, Any]) -> dict[str, Any]:
    champions = sorted(meta["champions"], key=lambda c: c["name"])
    best_items: dict[str, dict[str, Any]] = {}
    for item in meta["items"]:
        if not item["trainable"]:
            continue
        key = norm_name(item["name"])
        current = best_items.get(key)
        if current is None or item_rank(item) > item_rank(current):
            best_items[key] = item
    items = sorted(best_items.values(), key=lambda i: int(i["id"]))
    return {
        "ddragonVersion": meta["ddragonVersion"],
        "champions": {c["name"]: idx + 1 for idx, c in enumerate(champions)},
        "championList": [
            {
                "idx": idx + 1,
                "id": c["id"],
                "key": c["key"],
                "name": c["name"],
                "icon": c["icon"],
                "tags": c["tags"],
                "damage": c["damage"],
                "info": c.get("info", {}),
            }
            for idx, c in enumerate(champions)
        ],
        "items": {str(i["id"]): idx + 1 for idx, i in enumerate(items)},
        "itemList": [
            {
                "idx": idx + 1,
                "id": i["id"],
                "name": i["name"],
                "icon": i["icon"],
                "gold": i["gold"],
                "tags": i["tags"],
                "stats": i["stats"],
                "from": i["from"],
                "into": i["into"],
                "boots": i["boots"],
            }
            for idx, i in enumerate(items)
        ],
        "roles": config.ROLES,
        "roleIdx": {r: i + 1 for i, r in enumerate(config.ROLES)},
    }


def load_games(paths: list[Path]) -> dict[str, dict[str, Any]]:
    games: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            gid = clean(row.get("GameId"))
            if gid and gid not in games:
                games[gid] = row
    return games


def load_players(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    players: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            gid = clean(row.get("GameId"))
            if not gid:
                continue
            key = (gid, clean(row.get("Player")), clean(row.get("Champion")))
            if key in seen:
                continue
            seen.add(key)
            players[gid].append(row)
    return players


def data_sources(spec: str) -> list[str]:
    if spec.strip().lower() == "auto":
        sources = ["leaguepedia"]
        if (config.RAW_DIR / "riot" / "games.jsonl").exists():
            sources.append("riot")
        return sources
    return [s.strip().lower() for s in spec.split(",") if s.strip()]


def source_paths(names: list[str], kind: str) -> list[Path]:
    paths: list[Path] = []
    if "leaguepedia" in names:
        paths.append(RAW / f"{kind}.jsonl")
    if "riot" in names:
        paths.append(config.RAW_DIR / "riot" / f"{kind}.jsonl")
    return paths


def winner_side(game: dict[str, Any]) -> int:
    """0 = gana Team1, 1 = gana Team2, -1 desconocido."""
    win_team = clean(game.get("WinTeam"))
    team1 = clean(game.get("Team1"))
    team2 = clean(game.get("Team2"))
    if win_team and win_team == team1:
        return 0
    if win_team and win_team == team2:
        return 1
    winner = clean(game.get("Winner"))
    if winner in {"1", "2"}:
        return int(winner) - 1
    return -1


def game_date(game: dict[str, Any]) -> dt.date | None:
    raw = clean(game.get("DateUtc"))
    if not raw:
        return None
    try:
        return dt.datetime.fromisoformat(raw.replace("Z", "")).date()
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye el dataset procesado")
    parser.add_argument("--min-games", type=int, default=500)
    parser.add_argument(
        "--sources",
        default="auto",
        help="auto, leaguepedia, riot o lista separada por comas (ej: leaguepedia,riot)",
    )
    args = parser.parse_args()

    meta = load_meta()
    vocab = build_vocab(meta)
    champ_idx = vocab["champions"]
    item_idx = {int(k): v for k, v in vocab["items"].items()}
    champ_resolver = NameResolver(meta["champions"], CHAMPION_ALIASES)
    item_resolver = NameResolver(meta["items"], ITEM_ALIASES, prefer=item_rank)
    trainable_items = {i["id"]: i for i in meta["items"] if i["trainable"]}

    print("Cargando JSONL...")
    sources = data_sources(args.sources)
    if not sources:
        raise SystemExit("No hay fuentes de datos disponibles.")
    games = load_games(source_paths(sources, "games"))
    players = load_players(source_paths(sources, "players"))
    total_rows = sum(len(v) for v in players.values())
    print(f"  fuentes: {', '.join(sources)}")
    print(f"  partidas distintas: {len(games):,} | filas de jugador: {total_rows:,}")

    role_to_idx = vocab["roleIdx"]
    epoch = dt.date(2024, 1, 1)

    draft_rows: list[dict[str, Any]] = []
    player_index: list[dict[str, Any]] = []  # filas prometedoras para builds
    unresolved_champs: Counter[str] = Counter()
    unresolved_items: Counter[str] = Counter()
    excluded_items: Counter[str] = Counter()
    played_roles: Counter[str] = Counter()
    patches: Counter[str] = Counter()
    tournaments: Counter[str] = Counter()
    idx_to_item_id = {idx: iid for iid, idx in item_idx.items()}
    timing_samples: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for gid, game in games.items():
        plist = players.get(gid, [])
        if len(plist) < 10:
            continue
        win = winner_side(game)
        team1 = clean(game.get("Team1"))
        team2 = clean(game.get("Team2"))
        date = game_date(game)
        if win < 0 or not team1 or not team2 or date is None:
            continue
        patch = clean(game.get("Patch"))

        sides: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
        for prow in plist:
            team = clean(prow.get("Team"))
            side = 0 if team == team1 else 1 if team == team2 else -1
            if side < 0:
                continue
            role = ROLE_MAP.get(clean(prow.get("Role")).lower())
            champ = champ_resolver.resolve(prow.get("Champion"))
            if not champ:
                unresolved_champs[clean(prow.get("Champion"))] += 1
                continue
            if not role:
                continue
            sides[side].append(
                {"champ": champ, "role": role, "role_idx": role_to_idx[role], "raw": prow}
            )

        bad = False
        ordered: list[dict[str, Any]] = []
        for side in (0, 1):
            by_role = {p["role"]: p for p in sides[side]}
            if set(by_role) != set(config.ROLES):
                bad = True
                break
            ordered.extend(by_role[r] for r in config.ROLES)
        if bad:
            continue

        pick_ids = [champ_idx[p["champ"]["name"]] for p in ordered]
        pick_roles = [p["role_idx"] for p in ordered]
        pick_side = [0] * 5 + [1] * 5

        bans: list[tuple[int, int]] = []
        picked_keys = {p["champ"]["key"] for p in ordered}
        for side, field in ((0, "Team1Bans"), (1, "Team2Bans")):
            seen: set[int] = set()
            for raw_name in split_list(game.get(field)):
                champ = champ_resolver.resolve(raw_name)
                if champ and champ["key"] not in seen and champ["key"] not in picked_keys:
                    seen.add(champ["key"])
                    bans.append((champ_idx[champ["name"]], side))
        bans = bans[:10]
        ban_ids = [b[0] for b in bans] + [0] * (10 - len(bans))
        ban_sides = [b[1] for b in bans] + [0] * (10 - len(bans))

        duration = parse_float(game.get("Gamelength")) * 60.0
        draft_rows.append(
            {
                "gid": gid,
                "date": (date - epoch).days,
                "patch": parse_patch(patch),
                "patch_str": patch,
                "tournament": clean(game.get("Tournament")),
                "picks": pick_ids,
                "pick_roles": pick_roles,
                "pick_side": pick_side,
                "bans": ban_ids,
                "ban_sides": ban_sides,
                "y": win,
                "duration": duration,
                "team1": team1,
                "team2": team2,
            }
        )
        patches[patch] += 1
        tournaments[clean(game.get("Tournament"))] += 1

        for p in ordered:
            prow = p["raw"]
            seq: list[int] = []
            seen_items: set[int] = set()
            for raw_item in [clean(x) for x in split_list(prow.get("Items"), ";")]:
                item = item_resolver.resolve(raw_item)
                if not item:
                    if raw_item:
                        unresolved_items[raw_item] += 1
                    continue
                if item["id"] not in trainable_items:
                    excluded_items[item["name"]] += 1
                    continue
                if item["id"] in seen_items:
                    continue
                seen_items.add(item["id"])
                seq.append(item_idx[item["id"]])
            played_roles[p["role"]] += 1
            if len(seq) < 2:
                continue

            # Tiempos exactos de compra (solo disponibles con Riot timelines)
            purchase_minutes: dict[int, float] = {}
            for entry in prow.get("PurchaseTimes") or []:
                if not isinstance(entry, dict):
                    continue
                iid = parse_int(entry.get("itemId"))
                minute_value = parse_float(entry.get("minute"), -1)
                if iid in trainable_items and minute_value >= 0:
                    purchase_minutes[iid] = minute_value
            if purchase_minutes:
                for slot_idx, item_value in enumerate(seq):
                    iid = idx_to_item_id.get(item_value)
                    if iid in purchase_minutes:
                        timing_samples[str(iid)][p["role"]][slot_idx].append(purchase_minutes[iid])

            player_index.append(
                {
                    "gid": gid,
                    "champ_idx": champ_idx[p["champ"]["name"]],
                    "role_idx": p["role_idx"],
                    "side": side,
                    "items": seq[:7],
                    "gold": parse_int(prow.get("Gold")),
                    "duration": duration,
                    "win": win,
                }
            )

    n_games = len(draft_rows)
    if n_games < args.min_games:
        raise SystemExit(f"Muy pocas partidas validas ({n_games}). Revisa el scraping.")

    draft_rows.sort(key=lambda r: (r["date"], r["gid"]))
    gid_to_idx = {r["gid"]: i for i, r in enumerate(draft_rows)}
    for p in player_index:
        p["game_idx"] = gid_to_idx[p["gid"]]

    n = len(draft_rows)
    n_test = max(int(round(n * 0.05)), 1)
    n_val = max(int(round(n * 0.10)), 1)
    n_train = n - n_val - n_test
    split = np.zeros(n, dtype=np.int8)
    split[n_train : n_train + n_val] = 1
    split[n_train + n_val :] = 2

    draft_rows = [r for r in draft_rows]
    date_arr = np.array([r["date"] for r in draft_rows], dtype=np.int32)
    patch_arr = np.array([r["patch"][0] * 100 + r["patch"][1] for r in draft_rows], dtype=np.int32)
    picks = np.array([r["picks"] for r in draft_rows], dtype=np.int16)
    pick_roles_arr = np.array([r["pick_roles"] for r in draft_rows], dtype=np.int8)
    pick_side_arr = np.array([r["pick_side"] for r in draft_rows], dtype=np.int8)
    bans_arr = np.array([r["bans"] for r in draft_rows], dtype=np.int16)
    ban_sides_arr = np.array([r["ban_sides"] for r in draft_rows], dtype=np.int8)
    y = np.array([r["y"] for r in draft_rows], dtype=np.int8)

    build_rows = [p for p in player_index if p["game_idx"] < n]
    n_build = len(build_rows)
    b_champ = np.array([b["champ_idx"] for b in build_rows], dtype=np.int16)
    b_role = np.array([b["role_idx"] for b in build_rows], dtype=np.int8)
    b_side = np.array([b["side"] for b in build_rows], dtype=np.int8)
    b_items = np.array([b["items"] + [0] * (7 - len(b["items"])) for b in build_rows], dtype=np.int16)
    b_len = np.array([len(b["items"]) for b in build_rows], dtype=np.int8)
    b_gold = np.array([b["gold"] for b in build_rows], dtype=np.int32)
    b_duration = np.array([b["duration"] for b in build_rows], dtype=np.float32)
    b_win = np.array([b["win"] for b in build_rows], dtype=np.int8)
    b_game = np.array([b["game_idx"] for b in build_rows], dtype=np.int32)

    # Aliados/enemigos segun el lado real de cada jugador
    b_ally = np.zeros((n_build, 5), dtype=np.int16)
    b_enemy = np.zeros((n_build, 5), dtype=np.int16)
    for i, b in enumerate(build_rows):
        g = draft_rows[b["game_idx"]]
        if b["side"] == 0:
            b_ally[i] = g["picks"][:5]
            b_enemy[i] = g["picks"][5:]
        else:
            b_ally[i] = g["picks"][5:]
            b_enemy[i] = g["picks"][:5]

    np.savez_compressed(
        PROCESSED / "dataset.npz",
        picks=picks,
        pick_roles=pick_roles_arr,
        pick_side=pick_side_arr,
        bans=bans_arr,
        ban_sides=ban_sides_arr,
        y=y,
        split=split,
        date=date_arr,
        patch=patch_arr,
        b_champ=b_champ,
        b_role=b_role,
        b_side=b_side,
        b_ally=b_ally,
        b_enemy=b_enemy,
        b_items=b_items,
        b_len=b_len,
        b_gold=b_gold,
        b_duration=b_duration,
        b_win=b_win,
        b_game=b_game,
    )
    write_json(PROCESSED / "vocab.json", vocab)

    dates = [epoch + dt.timedelta(days=int(d)) for d in date_arr]
    summary = {
        "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "Leaguepedia (Cargo API) + Riot Data Dragon"
        + (" + Riot API high elo" if "riot" in sources else ""),
        "sources": sources,
        "ddragonVersion": meta["ddragonVersion"],
        "games": n,
        "players": n_build,
        "rawPlayers": total_rows,
        "tournaments": len(tournaments),
        "topTournaments": [{"name": k, "games": v} for k, v in tournaments.most_common(12)],
        "patches": [
            {"patch": k, "games": v}
            for k, v in sorted(patches.items(), key=lambda kv: parse_patch(kv[0]))[-20:]
        ],
        "dateRange": [str(dates[0]), str(dates[-1])],
        "splits": {
            "train": int((split == 0).sum()),
            "val": int((split == 1).sum()),
            "test": int((split == 2).sum()),
        },
        "roles": dict(played_roles),
        "coverage": {
            "champions": round(1 - sum(unresolved_champs.values()) / max(total_rows, 1), 4),
            "items": round(1 - sum(unresolved_items.values()) / max(total_rows, 1), 4),
        },
        "unresolvedChampions": unresolved_champs.most_common(20),
        "unresolvedItems": unresolved_items.most_common(40),
        "excludedItems": excluded_items.most_common(25),
        "championFuzzy": champ_resolver.fuzzy_hits.most_common(10),
        "itemFuzzy": item_resolver.fuzzy_hits.most_common(10),
    }
    write_json(PROCESSED / "summary.json", summary)

    if timing_samples:
        timings = {
            iid: {
                role: {
                    str(slot): {
                        "n": len(values),
                        "median": round(float(statistics.median(values)), 2),
                    }
                    for slot, values in slots.items()
                    if values
                }
                for role, slots in roles.items()
            }
            for iid, roles in timing_samples.items()
        }
        write_json(PROCESSED / "timings.json", timings)
        print(f"Timings de compra -> {PROCESSED / 'timings.json'}")

    build_priors(draft_rows, build_rows, split, vocab, meta, dates)

    print(
        f"Dataset: {n:,} partidas ({dates[0]} -> {dates[-1]}), {n_build:,} builds de jugador\n"
        f"  split train/val/test = {summary['splits']}\n"
        f"  cobertura campeones {summary['coverage']['champions']:.2%} | objetos {summary['coverage']['items']:.2%}"
    )
    for label, counter in (("Campeones", unresolved_champs), ("Objetos", unresolved_items)):
        if counter:
            print(f"  {label} sin resolver: {counter.most_common(10)}")


def build_priors(
    draft_rows: list[dict[str, Any]],
    build_rows: list[dict[str, Any]],
    split: np.ndarray,
    vocab: dict[str, Any],
    meta: dict[str, Any],
    dates: list[dt.date],
) -> None:
    """Estadisticas empiricas suavizadas, calculadas SOLO con el split de train."""
    train_games = [r for r, s in zip(draft_rows, split) if s == 0]
    train_builds = [b for b in build_rows if split[b["game_idx"]] == 0]
    print(f"Calculando priors con {len(train_games):,} partidas y {len(train_builds):,} builds...")

    champ_names = {v: k for k, v in vocab["champions"].items()}
    item_names = {v: k for k, v in vocab["items"].items()}
    item_meta = {i["id"]: i for i in meta["items"]}

    champ_role: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    champ_total: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    role_total: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    bans: Counter[str] = Counter()
    matchup: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    synergy: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    for r in train_games:
        names = [champ_names.get(c) for c in r["picks"]]
        for i, name in enumerate(names):
            if not name:
                continue
            role = config.ROLES[i % 5]
            won = 1 if (r["y"] == 0) == (i < 5) else 0
            champ_role[name][role][0] += 1
            champ_role[name][role][1] += won
            champ_total[name][0] += 1
            champ_total[name][1] += won
            role_total[role][0] += 1
            role_total[role][1] += won
        for champ_id in r["bans"]:
            if champ_id:
                name = champ_names.get(champ_id)
                if name:
                    bans[name] += 1
        team1 = names[:5]
        team2 = names[5:]
        for team, rivals, team_won in ((team1, team2, r["y"] == 0), (team2, team1, r["y"] == 1)):
            for a in team:
                if not a:
                    continue
                for b in team:
                    if b and b != a:
                        entry = synergy[a][b]
                        entry[0] += 1
                        entry[1] += int(team_won)
                for b in rivals:
                    if b:
                        entry = matchup[a][b]
                        entry[0] += 1
                        entry[1] += int(team_won)

    # Items por campeon+rol, en una sola pasada
    champ_role_items: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    )
    item_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "wins": 0, "slots": 0, "roles": defaultdict(lambda: [0, 0, 0])}
    )
    item_count: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    first_items: dict[str, Counter] = defaultdict(Counter)
    for b in train_builds:
        cmp_name = champ_names.get(b["champ_idx"])
        if not cmp_name:
            continue
        role = config.ROLES[b["role_idx"] - 1]
        won = 1 if b["side"] == b["win"] else 0
        item_count[min(len(b["items"]), 7)][0] += 1
        item_count[min(len(b["items"]), 7)][1] += won
        if b["items"]:
            iid_first = item_names.get(b["items"][0])
            if iid_first:
                first_items[role][iid_first] += 1
        for slot, item_idx in enumerate(b["items"]):
            if not item_idx:
                continue
            iid = item_names.get(item_idx)
            if not iid:
                continue
            entry = champ_role_items[cmp_name][role][iid]
            entry[0] += 1
            entry[1] += won
            entry[2] += slot
            st = item_stats[iid]
            st["n"] += 1
            st["wins"] += won
            st["slots"] += slot
            st["roles"][role][0] += 1
            st["roles"][role][1] += won
            st["roles"][role][2] += slot

    def smoothed(wins: int, n: int, prior: float, k: float = 25.0) -> float:
        return (wins + prior * k) / (n + k)

    champ_data: dict[str, Any] = {}
    for name, roles in champ_role.items():
        total_n, total_w = champ_total[name]
        champ_data[name] = {
            "n": total_n,
            "wins": total_w,
            "winrate": round(smoothed(total_w, total_n, 0.5), 4),
            "bans": bans.get(name, 0),
            "roles": {},
            "matchups": {},
            "synergy": {},
            "items": {},
        }
        for role, (rn, rw) in roles.items():
            base = role_total[role]
            prior = (base[1] + 1) / (base[0] + 2) if base[0] else 0.5
            champ_data[name]["roles"][role] = {
                "n": rn,
                "wins": rw,
                "winrate": round(smoothed(rw, rn, prior), 4),
            }
            rolls = champ_role_items.get(name, {}).get(role, {})
            champ_data[name]["items"][role] = {
                iid: {"n": int(v[0]), "wins": int(v[1]), "avgSlot": round(v[2] / max(v[0], 1), 2)}
                for iid, v in sorted(rolls.items(), key=lambda kv: -kv[1][0])[:40]
            }
        champ_data[name]["matchups"] = {
            enemy: {"n": int(v[0]), "wins": int(v[1])}
            for enemy, v in sorted(matchup[name].items(), key=lambda kv: -kv[1][0])[:80]
            if v[0] >= 3
        }
        champ_data[name]["synergy"] = {
            ally: {"n": int(v[0]), "wins": int(v[1])}
            for ally, v in sorted(synergy[name].items(), key=lambda kv: -kv[1][0])[:30]
            if v[0] >= 3
        }

    items_out: dict[str, Any] = {}
    for iid, st in item_stats.items():
        meta_item = item_meta.get(int(iid), {})
        items_out[iid] = {
            "name": meta_item.get("name", iid),
            "icon": meta_item.get("icon", ""),
            "gold": meta_item.get("gold", 0),
            "n": st["n"],
            "winrate": round(smoothed(st["wins"], st["n"], 0.5, 50.0), 4),
            "avgSlot": round(st["slots"] / max(st["n"], 1), 2),
            "roles": {
                role: {
                    "n": v[0],
                    "winrate": round(smoothed(v[1], v[0], 0.5, 25.0), 4),
                    "avgSlot": round(v[2] / max(v[0], 1), 2),
                }
                for role, v in st["roles"].items()
            },
        }

    priors = {
        "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "train split",
        "games": len(train_games),
        "builds": len(train_builds),
        "dateRange": [str(dates[0]), str(dates[-1])],
        "roleBaseline": {
            role: {
                "n": role_total[role][0],
                "winrate": round((role_total[role][1] + 1) / (role_total[role][0] + 2), 4)
                if role_total[role][0]
                else 0.5,
            }
            for role in config.ROLES
        },
        "itemCountWinrate": {
            str(k): {
                "n": v[0],
                "winrate": round((v[1] + 1) / (v[0] + 2), 4),
            }
            for k, v in sorted(item_count.items())
        },
        "firstItems": {
            role: [
                {
                    "id": int(iid),
                    "name": item_meta.get(int(iid), {}).get("name", iid),
                    "n": n,
                }
                for iid, n in counter.most_common(8)
            ]
            for role, counter in first_items.items()
        },
        "champions": champ_data,
        "items": items_out,
    }
    write_json(PROCESSED / "priors.json", priors)
    print(f"Priors -> {PROCESSED / 'priors.json'} ({len(champ_data)} campeones, {len(items_out)} objetos)")


if __name__ == "__main__":
    main()
