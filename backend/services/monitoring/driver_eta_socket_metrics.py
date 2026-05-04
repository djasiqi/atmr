"""Métriques P1 — émissions `eta_changed` vs localisations (ratio santé < 0,1 cible)."""

from __future__ import annotations

try:
    from prometheus_client import Counter
except ImportError:  # pragma: no cover
    Counter = None  # type: ignore[misc, assignment]

_EMITTED: Counter | None = None
_SKIPPED: Counter | None = None
_LOC_FOR_RATIO: Counter | None = None


def _emitted() -> Counter | None:
    global _EMITTED
    if Counter is None:
        return None
    if _EMITTED is None:
        _EMITTED = Counter(
            "driver_eta_changed_emitted_total",
            "Événements socket eta_changed émis vers le chauffeur",
        )
    return _EMITTED


def _skipped() -> Counter | None:
    global _SKIPPED
    if Counter is None:
        return None
    if _SKIPPED is None:
        _SKIPPED = Counter(
            "driver_eta_changed_skipped_total",
            "Émissions eta_changed ignorées (throttle, delta, erreur)",
            ["reason"],
        )
    return _SKIPPED


def _loc_ratio() -> Counter | None:
    global _LOC_FOR_RATIO
    if Counter is None:
        return None
    if _LOC_FOR_RATIO is None:
        _LOC_FOR_RATIO = Counter(
            "driver_eta_location_ingested_for_ratio_total",
            "Points driver_location canoniques (dénominateur ratio vs eta_changed)",
        )
    return _LOC_FOR_RATIO


def inc_eta_changed_emitted() -> None:
    c = _emitted()
    if c:
        c.inc()


def inc_eta_changed_skipped(*, reason: str) -> None:
    c = _skipped()
    if c:
        c.labels(reason=reason).inc()


def inc_driver_location_ingested_for_eta_ratio() -> None:
    c = _loc_ratio()
    if c:
        c.inc()
