"""Servidor FastAPI: API de recomendaciones + interfaz web."""
from __future__ import annotations

import argparse
from functools import lru_cache

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import config
from ..lcu import LeagueClient, get_locator, map_champ_select, map_live_gamedata
from . import engine as engine_module
from .schemas import (
    BansRequest,
    BuildRequest,
    ClientPathRequest,
    DraftRequest,
    HealthResponse,
    LiveRequest,
)

app = FastAPI(
    title="RiftSense API",
    version="1.0.0",
    description="Recomendaciones de draft y build para League of Legends",
)

locator = get_locator()
lcu = LeagueClient()
locator.start(force=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_assets(request, call_next):
    """Evita que el navegador cachee JS/CSS durante el desarrollo local."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/assets/") or path in {"/", "/index.html"}:
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


@lru_cache(maxsize=1)
def get_engine() -> engine_module.Engine:
    return engine_module.Engine()


def load_engine() -> engine_module.Engine:
    try:
        return get_engine()
    except Exception as exc:  # datos o modelos ausentes
        raise HTTPException(
            status_code=503,
            detail=f"El motor no está listo: {exc}",
        ) from exc


@app.get("/api/health", response_model=HealthResponse)
def health() -> dict:
    try:
        eng = get_engine()
    except Exception as exc:
        return {"status": "not_ready", "error": str(exc)}
    return {
        "status": "ok",
        "patch": eng.patch,
        "models": {
            "draft": eng.draft_model is not None,
            "build": eng.build_model is not None,
            "device": eng.device,
        },
        "dataset": {
            "games": eng.summary.get("games"),
            "players": eng.summary.get("players"),
            "dateRange": eng.summary.get("dateRange"),
        },
    }


@app.get("/api/meta")
def meta() -> dict:
    return load_engine().meta_payload()


@app.get("/api/meta/champions")
def meta_champions(role: str | None = None) -> dict:
    eng = load_engine()
    return {"role": role, "champions": eng.champion_meta(role)}


@app.post("/api/draft")
def draft(req: DraftRequest) -> dict:
    eng = load_engine()
    try:
        return eng.draft_recommend(
            role=req.role,
            side=req.side,
            bans=req.bans,
            allies=req.allies,
            enemies=req.enemies,
            limit=req.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error calculando el draft: {exc}") from exc


@app.post("/api/draft/bans")
def draft_bans(req: BansRequest) -> dict:
    eng = load_engine()
    try:
        return {"bans": eng.ban_priority(exclude=req.exclude, limit=req.limit)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error calculando bans: {exc}") from exc


@app.post("/api/build")
def build(req: BuildRequest, simulate: bool | None = None) -> dict:
    eng = load_engine()
    try:
        return eng.build_recommend(
            champion=req.champion,
            role=req.role,
            side=req.side,
            allies=req.allies,
            enemies=req.enemies,
            owned=req.owned,
            gold=req.gold,
            minute=req.minute,
            limit=req.limit,
            simulate=req.simulate if simulate is None else simulate,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error calculando la build: {exc}") from exc


@app.post("/api/live")
def live(req: LiveRequest) -> dict:
    eng = load_engine()
    try:
        draft_state = eng.draft_evaluate(
            role=req.role,
            side=req.side,
            bans=req.bans,
            allies=req.allies,
            enemies=req.enemies,
        )
        build_state = eng.build_recommend(
            champion=req.champion,
            role=req.role,
            side=req.side,
            allies=req.allies,
            enemies=req.enemies,
            owned=req.owned,
            gold=req.gold,
            minute=req.minute,
            limit=req.limit,
            simulate=req.simulate,
        )
        return {"draft": draft_state, "build": build_state}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en el análisis en vivo: {exc}") from exc


@app.get("/api/dataset")
def dataset() -> dict:
    return load_engine().dataset_payload()


@app.get("/api/model")
def model() -> dict:
    return load_engine().model_payload()


@app.post("/api/reload")
def reload_models() -> dict:
    get_engine.cache_clear()
    eng = load_engine()
    return {
        "reloaded": True,
        "draft": eng.draft_model is not None,
        "build": eng.build_model is not None,
        "patch": eng.patch,
    }


# --------------------------------------------------------------------------- #
# Cliente local (LCU + Live Client Data)
# --------------------------------------------------------------------------- #
@app.get("/api/client/status")
def client_status() -> dict:
    try:
        return lcu.status()
    except Exception as exc:
        return {
            "connected": False,
            "phase": None,
            "liveClient": False,
            "inGame": False,
            "champSelect": False,
            "error": f"Error consultando el cliente: {exc}",
        }


@app.get("/api/client/draft")
def client_draft() -> dict:
    try:
        status = lcu.status()
        if not status.get("connected") or status.get("phase") != "ChampSelect":
            return {
                "available": False,
                "status": status,
                "message": "Abre un champ select en League of Legends.",
            }
        session = lcu.champ_select_session()
        if not session:
            return {
                "available": False,
                "status": status,
                "message": "LCU esta en champ select pero no devolvio la sesion.",
            }
        gameflow = lcu.gameflow_session() or {}
        mapped = map_champ_select(session, gameflow, status)
        return {"available": bool(mapped.get("available")), "status": status, "draft": mapped}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error leyendo el champ select: {exc}") from exc


@app.get("/api/client/live")
def client_live() -> dict:
    try:
        data = lcu.live_gamedata()
        if not data:
            return {
                "available": False,
                "message": "No hay una partida en curso o el Live Client API no responde en el puerto 2999.",
            }
        mapped = map_live_gamedata(data)
        return {"available": bool(mapped.get("available")), "state": mapped}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error leyendo la partida en vivo: {exc}") from exc


@app.get("/api/client/search")
def client_search_status() -> dict:
    return {"search": locator.status()}


@app.post("/api/client/search")
def client_search() -> dict:
    search = locator.start(force=True)
    return {"started": True, "search": search}


@app.post("/api/client/path")
def client_path(req: ClientPathRequest) -> dict:
    try:
        search = locator.set_custom(req.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    lcu.reset()
    return {"ok": True, "search": search, "status": lcu.status()}


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #
if config.WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(config.WEB_DIR / "assets")), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.WEB_DIR / "index.html")


@app.get("/favicon.svg")
def favicon() -> FileResponse:
    return FileResponse(config.WEB_DIR / "assets" / "favicon.svg")


def main() -> None:
    parser = argparse.ArgumentParser(description="Servidor de RiftSense")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    print(f"RiftSense -> http://{args.host}:{args.port}")
    if args.reload:
        uvicorn.run("lolanalyzer.api.server:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
