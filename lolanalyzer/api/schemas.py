"""Esquemás de entrada/salida de la API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


ChampionRef = str | int


class DraftRequest(BaseModel):
    role: str = Field("MID", description="TOP, JUNGLE, MID, BOT o SUPPORT")
    side: str | int = Field("blue", description="blue/red o 0/1")
    bans: list[ChampionRef] = Field(default_factory=list, max_length=10)
    allies: list[ChampionRef | None] = Field(default_factory=list, max_length=5)
    enemies: list[ChampionRef | None] = Field(default_factory=list, max_length=5)
    limit: int = Field(10, ge=1, le=25)


class BansRequest(BaseModel):
    exclude: list[ChampionRef] = Field(default_factory=list, max_length=20)
    limit: int = Field(8, ge=1, le=20)


class BuildRequest(BaseModel):
    champion: ChampionRef
    role: str = "MID"
    side: str | int = "blue"
    allies: list[ChampionRef | None] = Field(default_factory=list, max_length=5)
    enemies: list[ChampionRef | None] = Field(default_factory=list, max_length=5)
    owned: list[int | str] = Field(default_factory=list, max_length=8, description="IDs de objeto en orden de compra")
    gold: int = Field(0, ge=0, le=100000)
    minute: float = Field(0.0, ge=0, le=120)
    limit: int = Field(10, ge=1, le=25)
    simulate: bool = True


class LiveRequest(BuildRequest):
    bans: list[ChampionRef] = Field(default_factory=list, max_length=10)


class ClientPathRequest(BaseModel):
    path: str = Field(..., min_length=2, max_length=500, description="Carpeta de LoL o lockfile")


class ApiError(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    patch: str | None = None
    models: dict[str, Any] | None = None
    dataset: dict[str, Any] | None = None
    error: str | None = None
