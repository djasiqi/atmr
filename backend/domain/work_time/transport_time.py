"""Temps réel d'une course : Arrivé → Terminé, avec qualité de la donnée."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from domain.work_time.clock import (
    elapsed_minutes,
    scheduled_zurich_date,
    split_minutes_by_zurich_day,
    zurich_date_of_instant,
)

QUALITY_VERIFIED = "verified"
QUALITY_ESTIMATED = "estimated_historical"
QUALITY_ADJUSTED = "adjusted"
QUALITY_INCOMPLETE = "incomplete"

SOURCE_VERIFIED = "arrived_to_completed"
SOURCE_ESTIMATED = "historical_pickup_to_completed"
SOURCE_ADJUSTED = "admin_adjustment"


@dataclass(frozen=True)
class AdjustmentView:
    """Dernière correction : snapshot complet des deux instants effectifs."""

    corrected_arrived_at: datetime | None
    corrected_completed_at: datetime | None
    reason: str = ""
    created_at: datetime | None = None
    adjustment_id: int | None = None


@dataclass(frozen=True)
class TransportInput:
    """Données minimales d'un tronçon pour le calcul de durée."""

    booking_id: int
    arrived_at: datetime | None
    boarded_at: datetime | None
    completed_at: datetime | None
    scheduled_time: datetime | None
    status: str


@dataclass
class TransportWorkTime:
    """Résultat de durée d'un tronçon."""

    effective_arrived_at: datetime | None
    effective_completed_at: datetime | None
    worked_minutes: int | None
    worked_minutes_source: str | None
    quality: str
    classification_date_zurich: str | None
    day_slices: list[tuple[str, int]] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)


def _completed_status(status: str) -> bool:
    return status.upper() in {"COMPLETED", "RETURN_COMPLETED"}


def business_policy_date(
    *,
    effective_completed_at: datetime | None,
    corrected_completed_at: datetime | None,
    scheduled_time: datetime | None,
) -> str | None:
    """Jour civil Europe/Zurich de la première fin exploitable.

    L'ordre est écrit même lorsque la fin effective reprend déjà la rectification :
    1. fin effective, si elle existe ;
    2. sinon fin rectifiée ;
    3. sinon le jour de ``scheduled_time``, uniquement sans aucune fin.
    """
    if effective_completed_at is not None:
        return zurich_date_of_instant(effective_completed_at)
    if corrected_completed_at is not None:
        return zurich_date_of_instant(corrected_completed_at)
    return scheduled_zurich_date(scheduled_time)


def compute_transport_work_time(
    row: TransportInput,
    adjustment: AdjustmentView | None,
    cutover: datetime,
) -> TransportWorkTime:
    """Calcule la durée réelle sans inventer d'heure manquante.

    Avant ``cutover``, l'absence de ``arrived_at`` avec prise en charge et fin
    produit une durée estimée historique. Après, c'est une anomalie.
    ``scheduled_time`` ne sert qu'à classer une course sans heure de fin.
    """
    if adjustment is not None:
        arrived = adjustment.corrected_arrived_at
        completed = adjustment.corrected_completed_at
        adjusted = True
    else:
        arrived = row.arrived_at
        completed = row.completed_at
        adjusted = False

    corrected = adjustment.corrected_completed_at if adjustment is not None else None
    policy_day = business_policy_date(
        effective_completed_at=completed,
        corrected_completed_at=corrected,
        scheduled_time=row.scheduled_time,
    )

    if not _completed_status(row.status):
        return TransportWorkTime(
            effective_arrived_at=arrived,
            effective_completed_at=completed,
            worked_minutes=None,
            worked_minutes_source=None,
            quality=QUALITY_INCOMPLETE,
            classification_date_zurich=scheduled_zurich_date(row.scheduled_time),
            anomalies=["segment_not_completed"],
        )

    if completed is None:
        return TransportWorkTime(
            effective_arrived_at=arrived,
            effective_completed_at=None,
            worked_minutes=None,
            worked_minutes_source=None,
            quality=QUALITY_INCOMPLETE,
            classification_date_zurich=policy_day,
            anomalies=["completion_time_missing"],
        )

    classification = policy_day

    if arrived is None:
        before_cutover = completed < cutover
        if (
            not adjusted
            and before_cutover
            and row.boarded_at is not None
            and row.completed_at is not None
            and row.completed_at >= row.boarded_at
        ):
            minutes = elapsed_minutes(row.boarded_at, row.completed_at)
            return TransportWorkTime(
                effective_arrived_at=row.boarded_at,
                effective_completed_at=row.completed_at,
                worked_minutes=minutes,
                worked_minutes_source=SOURCE_ESTIMATED,
                quality=QUALITY_ESTIMATED,
                classification_date_zurich=policy_day,
                day_slices=split_minutes_by_zurich_day(
                    row.boarded_at, row.completed_at
                ),
                anomalies=[],
            )
        return TransportWorkTime(
            effective_arrived_at=None,
            effective_completed_at=completed,
            worked_minutes=None,
            worked_minutes_source=None,
            quality=QUALITY_INCOMPLETE,
            classification_date_zurich=classification,
            anomalies=["arrival_not_recorded"],
        )

    if completed < arrived:
        return TransportWorkTime(
            effective_arrived_at=arrived,
            effective_completed_at=completed,
            worked_minutes=None,
            worked_minutes_source=None,
            quality=QUALITY_INCOMPLETE,
            classification_date_zurich=classification,
            anomalies=["completed_before_arrival"],
        )

    minutes = elapsed_minutes(arrived, completed)
    quality = QUALITY_ADJUSTED if adjusted else QUALITY_VERIFIED
    source = SOURCE_ADJUSTED if adjusted else SOURCE_VERIFIED
    anomalies = ["schedule_adjusted"] if adjusted else []
    return TransportWorkTime(
        effective_arrived_at=arrived,
        effective_completed_at=completed,
        worked_minutes=minutes,
        worked_minutes_source=source,
        quality=quality,
        classification_date_zurich=classification,
        day_slices=split_minutes_by_zurich_day(arrived, completed),
        anomalies=anomalies,
    )
