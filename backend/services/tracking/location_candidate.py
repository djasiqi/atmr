"""P5-B — candidat de localisation (scaffold, non branché sur le hot path).

Cible : séparer évaluation / promotion d'un point GPS du ``LocationService``
actuel, sans casser les callers existants. Implémentation complète hors scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class LocationCandidate:
    """Point GPS candidat avant arbitrage / promotion canonique (P5-B)."""

    driver_id: int
    latitude: float
    longitude: float
    recorded_at: datetime | None = None
    mission_id: int | None = None
    location_mode: str = "mission_live"
    accuracy: float | None = None
    transport: str = "http"
    raw_lat: float | None = None
    raw_lon: float | None = None
    meta: dict[str, Any] | None = None


def evaluate_location_candidate(
    candidate: LocationCandidate,
    *,
    context: dict[str, Any] | None = None,  # noqa: ARG001
) -> dict[str, Any]:
    """Évalue un candidat (admissibilité, téléport, fraîcheur).

    P5-B target : centraliser ici les règles aujourd'hui dispersées dans
    ``LocationService._store_location`` / firewall / arbitrage. Stub — ne pas
    appeler depuis le hot path tant que non branché.
    """
    return {
        "ok": True,
        "disposition": "pending",
        "reason": "p5b_scaffold_not_implemented",
        "candidate_driver_id": candidate.driver_id,
    }


def promote_location_candidate(
    candidate: LocationCandidate,
    *,
    evaluation: dict[str, Any] | None = None,  # noqa: ARG001
) -> dict[str, Any]:
    """Promeut un candidat vers canonical / ledger / fanout.

    P5-B target : écrire Redis canonical + éventuellement ledger après une
    évaluation positive. Stub — ne pas appeler depuis le hot path.
    """
    return {
        "ok": False,
        "promoted": False,
        "reason": "p5b_scaffold_not_implemented",
        "candidate_driver_id": candidate.driver_id,
    }
