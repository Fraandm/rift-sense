"""Motor de recomendación: combina las redes con el conocimiento empirico del dataset."""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .. import config
from ..models.build_net import load_build_net
from ..models.draft_net import load_draft_net
from ..utils import norm_name, read_json

ROLE_ALIASES = {
    "top": "TOP",
    "jgl": "JUNGLE",
    "jungle": "JUNGLE",
    "jungla": "JUNGLE",
    "mid": "MID",
    "middle": "MID",
    "bot": "BOT",
    "adc": "BOT",
    "tirador": "BOT",
    "support": "SUPPORT",
    "sup": "SUPPORT",
    "soporte": "SUPPORT",
}

SIDE_ALIASES = {"blue": 0, "azul": 0, "red": 1, "rojo": 1, "0": 0, "1": 1}

DAMAGE_STAT_KEYS = {
    "armor": ("FlatArmorMod", "Armor"),
    "mr": ("FlatSpellBlockMod", "SpellBlock"),
    "hp": ("FlatHPPoolMod", "Health"),
    "ad": ("FlatPhysicalDamageMod", "Damage"),
    "ap": ("FlatMagicDamageMod", "SpellDamage"),
}


def _standardize(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return np.zeros_like(arr)
    std = float(arr.std())
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


class Engine:
    """Carga artefactos y resuelve recomendaciones de draft y build."""

    def __init__(self) -> None:
        self.meta = read_json(config.META_PATH) or {}
        self.vocab = read_json(config.PROCESSED_DIR / "vocab.json") or {}
        self.priors = read_json(config.PRIORS_PATH) or {}
        self.summary = read_json(config.PROCESSED_DIR / "summary.json") or {}
        self.draft_report = read_json(config.DRAFT_REPORT_PATH) or {}
        self.build_report = read_json(config.BUILD_REPORT_PATH) or {}
        self.timings = read_json(config.PROCESSED_DIR / "timings.json") or {}

        if not self.vocab or not self.priors:
            raise RuntimeError(
                "Faltan datos procesados. Ejecuta el pipeline: "
                "python scripts/build_all.py"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.champions: list[dict[str, Any]] = self.vocab["championList"]
        self.items: list[dict[str, Any]] = self.vocab["itemList"]
        self.roles: list[str] = self.vocab["roles"]
        self.role_labels: dict[str, str] = self.meta.get("roleLabels", {})
        self.role_idx: dict[str, int] = self.vocab["roleIdx"]
        self.unk = len(self.champions) + 1

        self.champ_by_norm = {norm_name(c["name"]): c for c in self.champions}
        self.champ_by_key = {int(c["key"]): c for c in self.champions}
        self.item_by_id = {int(i["id"]): i for i in self.items}
        self.item_by_norm = {norm_name(i["name"]): i for i in self.items}
        self.all_items_by_id = {int(i["id"]): i for i in self.meta.get("items", [])}
        self.priors_games = max(int(self.priors.get("games", 1)), 1)
        self.priors_champs = self.priors.get("champions", {})
        self.priors_items = self.priors.get("items", {})

        self.draft_model = None
        self.build_model = None
        self.model_notes: list[str] = []
        if config.DRAFT_MODEL_PATH.exists():
            try:
                self.draft_model, _ = load_draft_net(str(config.DRAFT_MODEL_PATH), self.device)
                if self.draft_model.n_champions != len(self.champions):
                    raise ValueError(
                        f"vocabulario incompatible ({self.draft_model.n_champions} vs {len(self.champions)})"
                    )
            except Exception as exc:  # modelo corrupto o incompatible
                self.draft_model = None
                self.model_notes.append(f"No se pudo cargar DraftNet: {exc}")
        else:
            self.model_notes.append("DraftNet no entrenada: se usan estadísticas empiricas.")
        if config.BUILD_MODEL_PATH.exists():
            try:
                self.build_model, _ = load_build_net(str(config.BUILD_MODEL_PATH), self.device)
                if self.build_model.n_items != len(self.items):
                    raise ValueError(
                        f"vocabulario de objetos incompatible ({self.build_model.n_items} vs {len(self.items)})"
                    )
            except Exception as exc:
                self.build_model = None
                self.model_notes.append(f"No se pudo cargar BuildNet: {exc}")
        else:
            self.model_notes.append("BuildNet no entrenada: se usan estadísticas empiricas.")

    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #
    @property
    def patch(self) -> str:
        return str(self.meta.get("ddragonVersion", ""))

    def resolve_champion(self, value: Any) -> dict[str, Any] | None:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            champ = self.champ_by_key.get(int(value))
            if champ:
                return champ
            return self.champ_by_norm.get(norm_name(str(int(value))))
        text = str(value).strip()
        champ = self.champ_by_norm.get(norm_name(text))
        if champ:
            return champ
        if text.isdigit():
            return self.champ_by_key.get(int(text))
        return None

    def resolve_item(self, value: Any) -> dict[str, Any] | None:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return self.item_by_id.get(int(value))
        text = str(value).strip()
        if text.isdigit():
            return self.item_by_id.get(int(text))
        return self.item_by_norm.get(norm_name(text))

    def resolve_side(self, value: Any) -> int:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return 0 if int(value) == 0 else 1
        text = str(value).strip().lower()
        if text in SIDE_ALIASES:
            return SIDE_ALIASES[text]
        raise ValueError("Lado invalido: usa 'blue'/'red' o 0/1")

    def resolve_role(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        role = ROLE_ALIASES.get(text) or (text.upper() if text.upper() in self.roles else None)
        if role not in self.roles:
            raise ValueError(f"Rol invalido: {value}")
        return role

    def _team(self, names: Iterable[Any] | None) -> list[dict[str, Any] | None]:
        values = list(names or [])
        values = (values + [None] * 5)[:5]
        team: list[dict[str, Any] | None] = []
        seen: set[int] = set()
        for value in values:
            champ = self.resolve_champion(value)
            if champ:
                if champ["key"] in seen:
                    raise ValueError(f"Campeón duplicado: {champ['name']}")
                seen.add(champ["key"])
            team.append(champ)
        return team

    # ------------------------------------------------------------------ #
    # Draft
    # ------------------------------------------------------------------ #
    def _draft_inputs(
        self,
        side: int,
        role: str,
        allies: list[dict[str, Any] | None],
        enemies: list[dict[str, Any] | None],
        bans: list[dict[str, Any]],
    ) -> tuple[np.ndarray, ...]:
        champ = np.zeros((1, 20), dtype=np.int64)
        role_ids = np.zeros((1, 20), dtype=np.int64)
        sides = np.zeros((1, 20), dtype=np.int64)
        kinds = np.zeros((1, 20), dtype=np.int64)
        mask = np.zeros((1, 20), dtype=bool)

        team1 = allies if side == 0 else enemies
        team2 = enemies if side == 0 else allies
        for slot, team in ((0, team1), (1, team2)):
            for r in range(5):
                pos = slot * 5 + r
                champ[0, pos] = team[r]["idx"] if team[r] else self.unk
                role_ids[0, pos] = r + 1
                sides[0, pos] = slot
                kinds[0, pos] = 1
                mask[0, pos] = True

        for i, ban in enumerate(bans[:10]):
            pos = 10 + i
            champ[0, pos] = ban["idx"]
            sides[0, pos] = 0 if i < 5 else 1
            kinds[0, pos] = 2
            mask[0, pos] = True
        return champ, role_ids, sides, kinds, mask

    def _predict_draft(self, tensors: tuple[np.ndarray, ...]) -> np.ndarray:
        if self.draft_model is None:
            return np.full(tensors[0].shape[0], 0.5)
        with torch.inference_mode():
            out = self.draft_model(*[torch.from_numpy(t).to(self.device) for t in tensors])
            return torch.sigmoid(out).cpu().numpy()

    def _champ_edge(self, name: str, role: str) -> dict[str, Any]:
        data = self.priors_champs.get(name) or {}
        role_data = (data.get("roles") or {}).get(role) or {}
        baseline = (self.priors.get("roleBaseline") or {}).get(role, {}).get("winrate", 0.5)
        role_games = int((self.priors.get("roleBaseline") or {}).get(role, {}).get("n", 0)) or 1
        n = int(role_data.get("n", 0))
        wins = int(role_data.get("wins", 0))
        # Suavizado fuerte: muestras pequenas no deben dominar el ranking
        wr = (wins + baseline * 80.0) / (n + 80.0) if n else baseline
        return {
            "n": n,
            "wins": wins,
            "winrate": wr,
            "edge": wr - baseline,
            "baseline": baseline,
            "rolePickRate": n / role_games,
            "totalGames": int(data.get("n", 0)),
            "bans": int(data.get("bans", 0)),
            "presence": (int(data.get("n", 0)) + int(data.get("bans", 0))) / self.priors_games,
        }

    def _matchup_edge(self, name: str, enemies: list[dict[str, Any] | None]) -> dict[str, Any]:
        data = self.priors_champs.get(name) or {}
        matchups = data.get("matchups") or {}
        n_total = 0
        w_total = 0.0
        notes: list[dict[str, Any]] = []
        for enemy in enemies:
            if not enemy:
                continue
            entry = matchups.get(enemy["name"])
            if not entry:
                continue
            n = int(entry.get("n", 0))
            wins = int(entry.get("wins", 0))
            if n <= 0:
                continue
            wr = (wins + 0.5 * 20.0) / (n + 20.0)
            n_total += n
            w_total += wr * n
            notes.append({"enemy": enemy["name"], "winrate": wr, "n": n, "icon": enemy["icon"]})
        if not n_total:
            return {"winrate": 0.5, "n": 0, "edge": 0.0, "notes": []}
        wr = w_total / n_total
        confidence = n_total / (n_total + 80.0)
        return {
            "winrate": wr,
            "n": n_total,
            "edge": (wr - 0.5) * confidence,
            "notes": sorted(notes, key=lambda x: -x["n"])[:4],
        }

    def _synergy_edge(self, name: str, allies: list[dict[str, Any] | None]) -> dict[str, Any]:
        data = self.priors_champs.get(name) or {}
        synergy = data.get("synergy") or {}
        n_total = 0
        w_total = 0.0
        notes: list[dict[str, Any]] = []
        for ally in allies:
            if not ally or ally["name"] == name:
                continue
            entry = synergy.get(ally["name"])
            if not entry:
                continue
            n = int(entry.get("n", 0))
            wins = int(entry.get("wins", 0))
            if n <= 0:
                continue
            wr = (wins + 0.5 * 20.0) / (n + 20.0)
            n_total += n
            w_total += wr * n
            notes.append({"ally": ally["name"], "winrate": wr, "n": n, "icon": ally["icon"]})
        if not n_total:
            return {"winrate": 0.5, "n": 0, "edge": 0.0, "notes": []}
        wr = w_total / n_total
        confidence = n_total / (n_total + 80.0)
        return {
            "winrate": wr,
            "n": n_total,
            "edge": (wr - 0.5) * confidence,
            "notes": sorted(notes, key=lambda x: -x["n"])[:4],
        }

    def draft_recommend(
        self,
        *,
        role: str,
        side: int,
        bans: Sequence[Any] | None,
        allies: Sequence[Any] | None,
        enemies: Sequence[Any] | None,
        limit: int = 10,
    ) -> dict[str, Any]:
        role = self.resolve_role(role)
        side = self.resolve_side(side)
        ally_team = self._team(allies)
        enemy_team = self._team(enemies)
        ban_list: list[dict[str, Any]] = []
        seen_bans: set[int] = set()
        for value in list(bans or [])[:10]:
            champ = self.resolve_champion(value)
            if champ and champ["key"] not in seen_bans:
                seen_bans.add(champ["key"])
                ban_list.append(champ)

        picked = {c["key"] for c in ally_team + enemy_team if c}
        used = picked | seen_bans
        my_role_idx = self.role_idx[role]
        my_slot = (0 if side == 0 else 5) + my_role_idx - 1

        tensors = self._draft_inputs(side, role, ally_team, enemy_team, ban_list)
        base_p = float(self._predict_draft(tensors)[0])
        base_user_p = base_p if side == 0 else 1.0 - base_p

        my_champ = ally_team[my_role_idx - 1]
        if my_champ:
            my_edge = self._champ_edge(my_champ["name"], role)
            return {
                "mode": "model" if self.draft_model else "priors",
                "role": role,
                "roleLabel": self.role_labels.get(role, role),
                "side": "blue" if side == 0 else "red",
                "baselineWinProbability": round(base_user_p, 4),
                "myChampion": {k: my_champ[k] for k in ("id", "key", "name", "icon", "tags", "damage")},
                "myChampionStats": {
                    "winrate": round(my_edge["winrate"], 4),
                    "games": my_edge["n"],
                    "presence": round(my_edge["presence"], 4),
                },
                "knownAllies": [c["name"] if c else None for c in ally_team],
                "knownEnemies": [c["name"] if c else None for c in enemy_team],
                "bans": [c["name"] for c in ban_list],
                "recommendations": [],
                "banPriority": self.ban_priority(
                    exclude=[c["name"] for c in ban_list] + [c["name"] for c in ally_team + enemy_team if c],
                    limit=8,
                ),
                "notes": self.model_notes
                + ["Tu campeón ya está fijado: quítalo del slot para comparar alternativas."],
            }

        candidates = []
        for champ in self.champions:
            if int(champ["key"]) in used:
                continue
            edge = self._champ_edge(champ["name"], role)
            if edge["totalGames"] > 0 and edge["n"] == 0:
                continue
            if edge["totalGames"] < 3:
                continue
            role_share = edge["n"] / max(edge["totalGames"], 1)
            # Filtro estricto: el pick debe tener muestra suficiente en el rol
            # y no ser un caso residual de otro rol.
            if edge["n"] < 100 or role_share < 0.10:
                continue
            candidates.append((champ, edge))

        # Fallback para parches nuevos con poca muestra
        if len(candidates) < 6:
            candidates = []
            for champ in self.champions:
                if int(champ["key"]) in used:
                    continue
                edge = self._champ_edge(champ["name"], role)
                role_share = edge["n"] / max(edge["totalGames"], 1)
                if edge["n"] >= 20 and role_share >= 0.05:
                    candidates.append((champ, edge))
        if not candidates:
            raise ValueError("No hay campeones candidatos con datos suficientes.")

        # Prediccion del modelo en lote: un draft por candidato
        batch_n = len(candidates)
        champ = np.repeat(tensors[0], batch_n, axis=0)
        role_ids = np.repeat(tensors[1], batch_n, axis=0)
        sides = np.repeat(tensors[2], batch_n, axis=0)
        kinds = np.repeat(tensors[3], batch_n, axis=0)
        mask = np.repeat(tensors[4], batch_n, axis=0)
        for i, (champ_info, _) in enumerate(candidates):
            champ[i, my_slot] = champ_info["idx"]
        probs = self._predict_draft((champ, role_ids, sides, kinds, mask))
        user_probs = probs if side == 0 else 1.0 - probs

        model_edges = user_probs - base_user_p
        prior_edges = [edge["edge"] for _, edge in candidates]
        matchup = [self._matchup_edge(c["name"], enemy_team) for c, _ in candidates]
        synergy = [self._synergy_edge(c["name"], ally_team) for c, _ in candidates]
        presence = [edge["presence"] for _, edge in candidates]
        role_pick = [edge["rolePickRate"] for _, edge in candidates]

        if self.draft_model:
            weights = {
                "model": 0.35,
                "prior": 0.25,
                "matchup": 0.15 if any(enemy_team) else 0.0,
                "synergy": 0.10 if any(ally_team) else 0.0,
                "meta": 0.15,
            }
        else:
            weights = {
                "model": 0.0,
                "prior": 0.35,
                "matchup": 0.15 if any(enemy_team) else 0.0,
                "synergy": 0.10 if any(ally_team) else 0.0,
                "meta": 0.40,
            }
        total_w = sum(weights.values()) or 1.0
        weights = {k: v / total_w for k, v in weights.items()}

        z_model = _standardize(model_edges)
        z_prior = _standardize(prior_edges)
        z_matchup = _standardize([m["edge"] for m in matchup])
        z_synergy = _standardize([s["edge"] for s in synergy])
        z_meta = 0.5 * _standardize(presence) + 0.5 * _standardize(role_pick)
        known_picks = sum(1 for c in ally_team + enemy_team if c)
        model_confidence = 0.25 + 0.75 * (known_picks / 10.0)
        composite = (
            weights["model"] * z_model * model_confidence
            + weights["prior"] * z_prior
            + weights["matchup"] * z_matchup
            + weights["synergy"] * z_synergy
            + weights["meta"] * z_meta
        )

        results: list[dict[str, Any]] = []
        for i, (champ, edge) in enumerate(candidates):
            score = _clamp(50.0 + 18.0 * float(composite[i]), 1.0, 99.0)
            m = matchup[i]
            s = synergy[i]
            reasons: list[str] = []
            if edge["n"] >= 10:
                reasons.append(
                    f"Winrate {_pct(edge['winrate'])} en {self.role_labels.get(role, role)} "
                    f"({edge['n']} partidas pro)"
                )
            if m["n"] >= 15 and m["winrate"] >= 0.515:
                best = m["notes"][0] if m["notes"] else None
                if best:
                    reasons.append(
                        f"Buen matchup contra {best['enemy']} ({_pct(best['winrate'], 0)}, n={best['n']})"
                    )
            if s["n"] >= 15 and s["winrate"] >= 0.515:
                best = s["notes"][0] if s["notes"] else None
                if best:
                    reasons.append(
                        f"Sinergia con {best['ally']} ({_pct(best['winrate'], 0)}, n={best['n']})"
                    )
            if edge["presence"] >= 0.08:
                reasons.append(f"Presencia en draft: {_pct(edge['presence'], 0)}")
            if self.draft_model:
                delta = float(model_edges[i]) * 100
                if abs(delta) >= 0.05:
                    reasons.append(
                        f"Modelo: {_pct(float(user_probs[i]))} de victoria ({delta:+.1f} pp)"
                    )
            results.append(
                {
                    "champion": {
                        "id": champ["id"],
                        "key": champ["key"],
                        "name": champ["name"],
                        "icon": champ["icon"],
                        "tags": champ.get("tags", []),
                        "damage": champ.get("damage", ""),
                    },
                    "role": role,
                    "score": round(score, 1),
                    "winProbability": round(float(user_probs[i]), 4),
                    "deltaProbability": round(float(model_edges[i]), 4),
                    "evidence": {
                        "modelEdge": round(float(model_edges[i]), 4),
                        "priorWinrate": round(edge["winrate"], 4),
                        "priorGames": edge["n"],
                        "presence": round(edge["presence"], 4),
                        "rolePickRate": round(edge["rolePickRate"], 4),
                        "matchupWinrate": round(m["winrate"], 4),
                        "matchupGames": m["n"],
                        "synergyWinrate": round(s["winrate"], 4),
                        "synergyGames": s["n"],
                    },
                    "reasons": reasons[:4],
                }
            )

        results.sort(key=lambda r: -r["score"])
        return {
            "mode": "model" if self.draft_model else "priors",
            "role": role,
            "roleLabel": self.role_labels.get(role, role),
            "side": "blue" if side == 0 else "red",
            "baselineWinProbability": round(base_user_p, 4),
            "knownAllies": [c["name"] if c else None for c in ally_team],
            "knownEnemies": [c["name"] if c else None for c in enemy_team],
            "bans": [c["name"] for c in ban_list],
            "recommendations": results[:limit],
            "banPriority": self.ban_priority(exclude=[c["name"] for c in ban_list] + [c["name"] for c in ally_team + enemy_team if c], limit=8),
            "notes": self.model_notes,
        }

    def draft_evaluate(
        self,
        *,
        role: str,
        side: int,
        bans: Sequence[Any] | None,
        allies: Sequence[Any] | None,
        enemies: Sequence[Any] | None,
    ) -> dict[str, Any]:
        """Probabilidad estimada del draft actual (sin recomendar pick)."""
        role = self.resolve_role(role)
        side = self.resolve_side(side)
        ally_team = self._team(allies)
        enemy_team = self._team(enemies)
        ban_list: list[dict[str, Any]] = []
        seen: set[int] = set()
        for value in list(bans or [])[:10]:
            champ = self.resolve_champion(value)
            if champ and champ["key"] not in seen:
                seen.add(champ["key"])
                ban_list.append(champ)

        tensors = self._draft_inputs(side, role, ally_team, enemy_team, ban_list)
        p = float(self._predict_draft(tensors)[0])
        p_user = p if side == 0 else 1.0 - p

        def team_edge(team: list[dict[str, Any] | None]) -> float:
            values = []
            for i, champ in enumerate(team):
                if champ:
                    values.append(self._champ_edge(champ["name"], self.roles[i])["edge"])
            return float(np.mean(values)) if values else 0.0

        ally_edge = team_edge(ally_team)
        enemy_edge = team_edge(enemy_team)
        p_prior = _clamp(0.5 + 1.2 * (ally_edge - enemy_edge), 0.03, 0.97)
        if self.draft_model:
            # El modelo de draft es modesto (AUC ~0.54): suavizamos los extremos
            # y lo mezclamos con la evidencia empirica para no mostrar 30% por ruido.
            shrunk = 0.5 + 0.65 * (p_user - 0.5)
            p_final = _clamp(0.6 * shrunk + 0.4 * p_prior, 0.03, 0.97)
        else:
            p_final = p_prior

        my_champ = ally_team[self.role_idx[role] - 1]
        known = sum(1 for c in ally_team + enemy_team if c)
        return {
            "mode": "model" if self.draft_model else "priors",
            "winProbability": round(p_final, 4),
            "modelProbability": round(p_user, 4) if self.draft_model else None,
            "priorProbability": round(p_prior, 4),
            "myChampion": (
                {k: my_champ[k] for k in ("id", "key", "name", "icon", "tags", "damage")}
                if my_champ
                else None
            ),
            "allyEdge": round(ally_edge, 4),
            "enemyEdge": round(enemy_edge, 4),
            "knownPicks": known,
            "complete": known == 10,
            "notes": self.model_notes,
        }

    def ban_priority(self, *, exclude: Sequence[str] | None = None, limit: int = 8) -> list[dict[str, Any]]:
        excluded = {norm_name(x) for x in (exclude or [])}
        rows = []
        for name, data in self.priors_champs.items():
            if norm_name(name) in excluded:
                continue
            presence = (int(data.get("n", 0)) + int(data.get("bans", 0))) / self.priors_games
            wr = float(data.get("winrate", 0.5))
            rows.append(
                {
                    "name": name,
                    "presence": presence,
                    "winrate": wr,
                    "games": int(data.get("n", 0)),
                    "bans": int(data.get("bans", 0)),
                }
            )
        if not rows:
            return []
        pres_z = _standardize([r["presence"] for r in rows])
        wr_z = _standardize([r["winrate"] for r in rows])
        for i, row in enumerate(rows):
            row["priority"] = round(_clamp(50 + 18 * (0.65 * pres_z[i] + 0.35 * wr_z[i]), 1, 99), 1)
            champ = self.resolve_champion(row["name"])
            if champ:
                row["icon"] = champ["icon"]
                row["key"] = champ["key"]
        rows.sort(key=lambda r: -r["priority"])
        for row in rows:
            row["reason"] = (
                f"Presencia {_pct(row['presence'], 0)} · winrate {_pct(row['winrate'])}"
            )
        return rows[:limit]

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def _enemy_profile(self, enemies: list[dict[str, Any] | None]) -> dict[str, Any]:
        ad = 0.0
        ap = 0.0
        for champ in enemies:
            if not champ:
                continue
            info = champ.get("info") or {}
            tags = set(champ.get("tags", []))
            a = float(info.get("attack", 0) or 0) + (2.0 if tags & {"Marksman", "Fighter", "Assassin"} else 0.0)
            m = float(info.get("magic", 0) or 0) + (2.0 if "Mage" in tags else 0.0)
            if champ.get("damage") == "AD":
                a += 2.0
            elif champ.get("damage") == "AP":
                m += 2.0
            ad += a
            ap += m
        total = ad + ap
        if total <= 0:
            return {"physical": 0.5, "magic": 0.5, "label": "Mixta", "known": 0}
        physical = ad / total
        label = "AD" if physical >= 0.62 else "AP" if physical <= 0.38 else "Mixta"
        return {
            "physical": round(physical, 3),
            "magic": round(1 - physical, 3),
            "label": label,
            "known": sum(1 for c in enemies if c),
        }

    def _item_prior(self, champ_name: str, role: str, item_id: int) -> dict[str, Any]:
        champ_data = self.priors_champs.get(champ_name) or {}
        role_data = (champ_data.get("roles") or {}).get(role) or {}
        champ_games = int(role_data.get("n", 0))
        champ_items = ((champ_data.get("items") or {}).get(role) or {})
        entry = champ_items.get(str(item_id))
        if entry:
            return {
                "n": int(entry.get("n", 0)),
                "wins": int(entry.get("wins", 0)),
                "avgSlot": float(entry.get("avgSlot", 0)),
                "scope": "champ",
                "champGames": champ_games,
            }
        # Si el campeón tiene datos suficientes, un objeto que nunca compra no
        # debe heredar la popularidad global del rol.
        if champ_games >= 50:
            return {"n": 0, "wins": 0, "avgSlot": 0.0, "scope": "none", "champGames": champ_games}
        role_items = ((self.priors_items.get(str(item_id)) or {}).get("roles") or {}).get(role)
        if role_items:
            return {
                "n": int(role_items.get("n", 0)),
                "wins": int(role_items.get("n", 0) * float(role_items.get("winrate", 0.5))),
                "avgSlot": float(role_items.get("avgSlot", 0)),
                "scope": "role",
                "champGames": champ_games,
            }
        return {"n": 0, "wins": 0, "avgSlot": 0.0, "scope": "none", "champGames": champ_games}

    def _defensive_fit(self, item: dict[str, Any], profile: dict[str, Any]) -> tuple[float, str]:
        stats = item.get("stats") or {}
        tags = set(item.get("tags", []))
        armor = float(stats.get("FlatArmorMod", 0) or 0) + (1.0 if "Armor" in tags else 0.0)
        mr = float(stats.get("FlatSpellBlockMod", 0) or 0) + (1.0 if "SpellBlock" in tags else 0.0)
        hp = float(stats.get("FlatHPPoolMod", 0) or 0)
        physical = profile["physical"]
        magic = profile["magic"]
        score = 0.0
        note = ""
        if armor and physical >= 0.55:
            score = min(armor / 60.0, 1.3) * (physical - 0.5) * 2.0
            note = "armadura contra daño físico"
        if mr and magic >= 0.55:
            mr_score = min(mr / 55.0, 1.3) * (magic - 0.5) * 2.0
            if mr_score > score:
                score = mr_score
                note = "resistencia mágica contra daño AP"
        if hp and profile["label"] == "Mixta" and hp >= 300:
            hp_score = min(hp / 700.0, 1.0) * 0.5
            if hp_score > score:
                score = hp_score
                note = "vida contra daño mixto"
        return _clamp(score, 0.0, 1.5), note

    def _score_items(
        self,
        *,
        champ: dict[str, Any],
        role: str,
        side: int,
        ally_team: list[dict[str, Any] | None],
        enemy_team: list[dict[str, Any] | None],
        owned: list[dict[str, Any]],
        gold: int,
        minute: float,
        use_model: bool = True,
    ) -> list[dict[str, Any]]:
        owned_ids = {int(i["id"]) for i in owned}
        slot = len(owned)
        profile = self._enemy_profile(enemy_team)

        items_tensor = np.zeros((1, 7), dtype=np.int64)
        for i, item in enumerate(owned[:7]):
            items_tensor[0, i] = int(item["idx"])
        champ_tensor = np.array([int(champ["idx"])], dtype=np.int64)
        role_tensor = np.array([self.role_idx[role]], dtype=np.int64)
        side_tensor = np.array([int(side)], dtype=np.int64)
        ally_tensor = np.array([[(c["idx"] if c else 0) for c in ally_team[:5]]], dtype=np.int64)
        enemy_tensor = np.array([[(c["idx"] if c else 0) for c in enemy_team[:5]]], dtype=np.int64)

        if use_model and self.build_model is not None:
            with torch.inference_mode():
                logits = self.build_model(
                    torch.from_numpy(champ_tensor).to(self.device),
                    torch.from_numpy(role_tensor).to(self.device),
                    torch.from_numpy(side_tensor).to(self.device),
                    torch.from_numpy(ally_tensor).to(self.device),
                    torch.from_numpy(enemy_tensor).to(self.device),
                    torch.from_numpy(items_tensor).to(self.device),
                    torch.tensor([slot], dtype=torch.long, device=self.device),
                )
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        else:
            probs = np.zeros(len(self.items) + 1)

        candidates = []
        for item in self.items:
            item_id = int(item["id"])
            if item_id in owned_ids:
                continue
            if slot >= 6:
                continue
            if item.get("boots") and any(i.get("boots") for i in owned):
                continue
            prior = self._item_prior(champ["name"], role, item_id)
            avg_slot = prior["avgSlot"]
            if prior["scope"] == "none":
                slot_fit = 0.30
            else:
                slot_fit = math.exp(-0.5 * ((avg_slot - slot) / 1.6) ** 2)
            wr = (prior["wins"] + 0.5 * 25.0) / (prior["n"] + 25.0)
            wr_edge = wr - 0.5
            defense, defense_note = self._defensive_fit(item, profile)
            candidates.append(
                {
                    "item": item,
                    "modelProb": float(probs[int(item["idx"])]) if probs.size else 0.0,
                    "priorN": prior["n"],
                    "avgSlot": avg_slot,
                    "slotFit": slot_fit,
                    "winrate": wr,
                    "wrEdge": wr_edge,
                    "defense": defense,
                    "defenseNote": defense_note,
                }
            )
        if not candidates:
            return []

        weights = {
            "model": 0.50 if (use_model and self.build_model is not None) else 0.0,
            "popularity": 0.20,
            "slot": 0.15,
            "winrate": 0.05,
            "defense": 0.10,
        }
        total_w = sum(weights.values()) or 1.0
        weights = {k: v / total_w for k, v in weights.items()}

        max_prior = max((c["priorN"] for c in candidates), default=1) or 1
        z_model = _standardize([c["modelProb"] for c in candidates])
        z_pop = _standardize([c["priorN"] / max_prior for c in candidates])
        z_slot = _standardize([c["slotFit"] for c in candidates])
        z_wr = _standardize([c["wrEdge"] for c in candidates])
        z_def = _standardize([c["defense"] for c in candidates])

        for i, c in enumerate(candidates):
            composite = (
                weights["model"] * z_model[i]
                + weights["popularity"] * z_pop[i]
                + weights["slot"] * z_slot[i]
                + weights["winrate"] * z_wr[i]
                + weights["defense"] * z_def[i]
            )
            c["score"] = _clamp(50 + 11 * composite, 1, 96)
            c["affordable"] = gold >= int(c["item"]["gold"])
            c["missingGold"] = max(int(c["item"]["gold"]) - gold, 0)

        candidates.sort(key=lambda c: -c["score"])
        return candidates

    def _components(self, item: dict[str, Any], gold: int) -> list[dict[str, Any]]:
        components = []
        for cid in item.get("from", []):
            comp = self.all_items_by_id.get(int(cid)) or self.item_by_id.get(int(cid))
            if not comp:
                continue
            cost = int(comp.get("gold", 0) or 0)
            components.append(
                {
                    "id": int(comp["id"]),
                    "name": comp.get("name", ""),
                    "icon": comp.get("icon", ""),
                    "gold": cost,
                    "affordable": gold >= cost,
                }
            )
        components.sort(key=lambda c: -c["gold"])
        return components[:5]

    def _format_item(self, candidate: dict[str, Any], gold: int, slot: int, role: str) -> dict[str, Any]:
        item = candidate["item"]
        prior_n = candidate["priorN"]
        avg_slot = candidate["avgSlot"]
        typical_minute = None
        if self.timings:
            slot_timings = ((self.timings.get(str(item["id"])) or {}).get(role) or {})
            entry = slot_timings.get(str(slot)) or {}
            typical_minute = entry.get("median")
        reasons: list[str] = []
        if prior_n >= 10:
            reasons.append(f"Aparece en {prior_n} builds analizadas")
            if avg_slot:
                reasons.append(f"Posición media en inventario: {avg_slot + 1:.1f}")
        if typical_minute:
            reasons.append(f"Minuto típico: ~{typical_minute:.1f}")
        if candidate["winrate"] >= 0.515 and prior_n >= 20:
            reasons.append(f"Winrate {_pct(candidate['winrate'])} con este objeto")
        if candidate["defenseNote"]:
            reasons.append(f"Aporta {candidate['defenseNote']}")
        if candidate["affordable"]:
            reasons.append("Puedes comprarlo ahora")
        else:
            reasons.append(f"Te faltan {candidate['missingGold']} de oro")
        components = self._components(item, gold)
        if components and not candidate["affordable"]:
            cheap = [c for c in components if c["affordable"]]
            if cheap:
                reasons.append("Componente asequible: " + cheap[0]["name"])
        return {
            "id": int(item["id"]),
            "name": item["name"],
            "icon": item["icon"],
            "gold": int(item["gold"]),
            "tags": item.get("tags", []),
            "score": round(float(candidate["score"]), 1),
            "affordable": bool(candidate["affordable"]),
            "missingGold": int(candidate["missingGold"]),
            "avgSlot": round(float(avg_slot) + 1, 2) if prior_n else None,
            "typicalMinute": round(float(typical_minute), 2) if typical_minute else None,
            "priorGames": prior_n,
            "winrate": round(float(candidate["winrate"]), 4),
            "modelProb": round(float(candidate["modelProb"]), 5),
            "components": components,
            "reasons": reasons,
            "slot": slot + 1,
        }

    def build_recommend(
        self,
        *,
        champion: Any,
        role: str,
        side: int,
        allies: Sequence[Any] | None = None,
        enemies: Sequence[Any] | None = None,
        owned: Sequence[Any] | None = None,
        gold: int = 0,
        minute: float = 0.0,
        limit: int = 10,
        simulate: bool = True,
    ) -> dict[str, Any]:
        champ = self.resolve_champion(champion)
        if not champ:
            raise ValueError(f"Campeón desconocido: {champion}")
        role = self.resolve_role(role)
        side = self.resolve_side(side)
        ally_team = self._team(allies)
        enemy_team = self._team(enemies)
        gold = int(max(0, min(int(gold or 0), 100000)))
        minute = float(max(0.0, min(float(minute or 0.0), 120.0)))

        owned_items: list[dict[str, Any]] = []
        seen_items: set[int] = set()
        for value in list(owned or [])[:8]:
            item = self.resolve_item(value)
            if item and int(item["id"]) not in seen_items:
                seen_items.add(int(item["id"]))
                owned_items.append(item)
        slot = min(len(owned_items), 6)

        profile = self._enemy_profile(enemy_team)
        if slot >= 6:
            return {
                "mode": "complete",
                "champion": {k: champ[k] for k in ("id", "key", "name", "icon", "tags", "damage")},
                "role": role,
                "slot": 6,
                "enemyProfile": profile,
                "items": [],
                "buildPlan": [],
                "notes": ["Build completa: 6 objetos. Considera el trinket y consumibles."],
            }

        candidates = self._score_items(
            champ=champ,
            role=role,
            side=side,
            ally_team=ally_team,
            enemy_team=enemy_team,
            owned=owned_items,
            gold=gold,
            minute=minute,
        )
        items_out = [self._format_item(c, gold, slot, role) for c in candidates[:limit]]
        affordable = [c for c in candidates if gold >= int(c["item"]["gold"])]
        next_purchase = self._format_item(affordable[0], gold, slot, role) if affordable else None

        build_plan: list[dict[str, Any]] = []
        if simulate:
            sim_owned = list(owned_items)
            sim_gold = gold
            total_cost = 0
            for _ in range(6 - slot):
                sims = self._score_items(
                    champ=champ,
                    role=role,
                    side=side,
                    ally_team=ally_team,
                    enemy_team=enemy_team,
                    owned=sim_owned,
                    gold=sim_gold,
                    minute=minute,
                )
                if not sims:
                    break
                best = sims[0]
                item = best["item"]
                build_plan.append(
                    {
                        "id": int(item["id"]),
                        "name": item["name"],
                        "icon": item["icon"],
                        "gold": int(item["gold"]),
                        "slot": len(sim_owned) + 1,
                        "score": round(float(best["score"]), 1),
                        "reasons": [
                            r
                            for r in [
                                f"Score {best['score']:.0f}",
                                f"Winrate {_pct(best['winrate'])}" if best["priorN"] >= 20 else "",
                            ]
                            if r
                        ],
                    }
                )
                total_cost += int(item["gold"])
                sim_owned.append(item)
                sim_gold = max(0, sim_gold - int(item["gold"]))

        item_count = (self.priors.get("itemCountWinrate") or {})
        spike = item_count.get(str(slot + 1)) or item_count.get(str(slot)) or {}
        others = {
            k: v
            for k, v in item_count.items()
            if k in {str(x) for x in range(slot, min(slot + 3, 8))}
        }

        notes = list(self.model_notes)
        if not candidates:
            notes.append("No hay objetos candidatos con los filtros actuales.")
        if profile["known"] < 5:
            notes.append("Faltan campeones enemigos: el perfil de daño es aproximado.")

        return {
            "mode": "model" if self.build_model else "priors",
            "champion": {k: champ[k] for k in ("id", "key", "name", "icon", "tags", "damage")},
            "role": role,
            "roleLabel": self.role_labels.get(role, role),
            "side": "blue" if side == 0 else "red",
            "slot": slot + 1,
            "gold": gold,
            "minute": minute,
            "enemyProfile": profile,
            "items": items_out,
            "nextPurchase": next_purchase,
            "buildPlan": build_plan,
            "powerSpike": {
                "slot": slot + 1,
                "n": int(spike.get("n", 0)),
                "winrate": spike.get("winrate"),
                "nearby": others,
            },
            "notes": notes,
        }

    # ------------------------------------------------------------------ #
    # Meta / estadísticas
    # ------------------------------------------------------------------ #
    def champion_meta(self, role: str | None = None) -> list[dict[str, Any]]:
        role = self.resolve_role(role) if role else None
        rows = []
        for name, data in self.priors_champs.items():
            champ = self.resolve_champion(name)
            if not champ:
                continue
            roles_data = data.get("roles") or {}
            if role:
                rd = roles_data.get(role)
                if not rd or int(rd.get("n", 0)) == 0:
                    continue
                pick_rate = int(rd.get("n", 0)) / self.priors_games
                wr = float(rd.get("winrate", 0.5))
                games = int(rd.get("n", 0))
            else:
                pick_rate = int(data.get("n", 0)) / self.priors_games
                wr = float(data.get("winrate", 0.5))
                games = int(data.get("n", 0))
            bans = int(data.get("bans", 0))
            rows.append(
                {
                    "name": name,
                    "key": champ["key"],
                    "id": champ["id"],
                    "icon": champ["icon"],
                    "tags": champ.get("tags", []),
                    "damage": champ.get("damage", ""),
                    "games": games,
                    "winrate": round(wr, 4),
                    "pickRate": round(pick_rate, 4),
                    "banRate": round(bans / self.priors_games, 4),
                    "presence": round(min(1.0, pick_rate + bans / self.priors_games), 4),
                    "roles": {
                        r: {
                            "n": int(v.get("n", 0)),
                            "winrate": float(v.get("winrate", 0.5)),
                        }
                        for r, v in roles_data.items()
                    },
                }
            )
        rows.sort(key=lambda r: -r["presence"])
        return rows

    def meta_payload(self) -> dict[str, Any]:
        champion_stats = self.champion_meta()
        stats_by_name = {row["name"]: row for row in champion_stats}
        champions = []
        for champ in self.champions:
            stat = stats_by_name.get(champ["name"], {})
            champions.append(
                {
                    **{k: champ.get(k) for k in ("id", "key", "name", "icon", "tags", "damage", "info")},
                    "games": stat.get("games", 0),
                    "winrate": stat.get("winrate", 0.5),
                    "pickRate": stat.get("pickRate", 0.0),
                    "banRate": stat.get("banRate", 0.0),
                    "presence": stat.get("presence", 0.0),
                    "roles": stat.get("roles", {}),
                }
            )
        return {
            "patch": self.patch,
            "generatedAt": self.meta.get("generatedAt"),
            "iconBase": self.meta.get("iconBase", ""),
            "roles": self.roles,
            "roleLabels": self.role_labels,
            "champions": champions,
            "items": [
                {
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
                for i in self.items
            ],
            "models": {
                "draft": self.draft_model is not None,
                "build": self.build_model is not None,
                "device": self.device,
                "notes": self.model_notes,
            },
            "dataset": {
                "games": self.summary.get("games", 0),
                "players": self.summary.get("players", 0),
                "dateRange": self.summary.get("dateRange", []),
                "tournaments": self.summary.get("tournaments", 0),
                "source": self.summary.get("source", ""),
            },
        }

    def dataset_payload(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "priors": {
                "games": self.priors.get("games"),
                "builds": self.priors.get("builds"),
                "dateRange": self.priors.get("dateRange"),
                "roleBaseline": self.priors.get("roleBaseline"),
                "itemCountWinrate": self.priors.get("itemCountWinrate"),
                "firstItems": self.priors.get("firstItems"),
            },
            "champions": self.champion_meta(),
            "coverage": {
                "championsWithData": len(self.priors_champs),
                "itemsWithData": len(self.priors_items),
                "championPool": len(self.champions),
                "itemPool": len(self.items),
            },
        }

    def model_payload(self) -> dict[str, Any]:
        def artifact(path, report):
            return {
                "exists": path.exists(),
                "sizeMB": round(path.stat().st_size / 1e6, 2) if path.exists() else 0,
                "report": report,
            }

        return {
            "draft": artifact(config.DRAFT_MODEL_PATH, self.draft_report),
            "build": artifact(config.BUILD_MODEL_PATH, self.build_report),
            "device": self.device,
            "notes": self.model_notes,
            "patch": self.patch,
        }

    def icon(self, kind: str, file: str) -> str:
        base = self.meta.get("iconBase", "")
        folder = {"champion": "champion", "item": "item", "spell": "spell", "rune": "rune"}.get(kind)
        if not base or not folder:
            return ""
        return f"{base}/{folder}/{file}"
