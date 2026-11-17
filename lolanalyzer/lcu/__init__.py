"""Integracion de solo lectura con el cliente local de League of Legends."""

from .client import LeagueClient
from .locator import ClientLocator, find_lockfile, get_locator
from .mapper import map_champ_select, map_live_gamedata

__all__ = [
    "LeagueClient",
    "ClientLocator",
    "find_lockfile",
    "get_locator",
    "map_champ_select",
    "map_live_gamedata",
]
