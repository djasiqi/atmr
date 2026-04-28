"""Routage read/write Phase 4 (primary + replicas).

Objectif:
- centraliser la politique de vérité data-plane
- exposer un bind SQLAlchemy cohérent selon le type de lecture
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass


def _parse_csv_env(name: str) -> list[str]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class ReadContext:
    """Contexte de lecture minimal pour choisir le bind."""

    critical_post_write: bool = False
    analytical: bool = False


class DataPlaneRouter:
    """Sélecteur de bind read/write pour SQLAlchemy."""

    def __init__(self) -> None:
        self._replicas = _parse_csv_env("REPLICA_DATABASE_URLS")
        if not self._replicas:
            replica_single = (os.getenv("REPLICA_DATABASE_URL") or "").strip()
            if replica_single:
                self._replicas = [replica_single]
        self._rr_index = itertools.count(0)

    def write_bind(self) -> str:
        return "primary"

    def read_bind(self, context: ReadContext | None = None) -> str:
        ctx = context or ReadContext()
        # Politique Phase 4: post-write critique => primary
        if ctx.critical_post_write:
            return "primary"

        # Si pas de replica configurée, on reste sur primary
        if not self._replicas:
            return "primary"

        # Volumétrique/analytique: replica
        return "replica"

    def next_replica_url(self) -> str | None:
        """Expose l'URL replica suivante pour consommateurs non-ORM."""
        if not self._replicas:
            return None
        idx = next(self._rr_index) % len(self._replicas)
        return self._replicas[idx]


data_plane_router = DataPlaneRouter()
