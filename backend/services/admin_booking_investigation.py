"""Règles d'investigation admin transports — parité SQL / Python.

Source de vérité pour :
- filtres / compteurs `needs_investigation` et `incomplete_data` ;
- raisons structurées sur liste et détail ;
- diagnostic support (`action_required` / `attention` / `ok`).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, exists, func, or_, select

from models import Booking, BookingStatus
from models.booking_transfer import BookingTransfer
from models.enums import TransferStatus

# Statuts métier qui exigent un chauffeur (aligné contraintes modèle).
_STATUSES_REQUIRING_DRIVER = (
    BookingStatus.ASSIGNED,
    BookingStatus.EN_ROUTE,
    BookingStatus.IN_PROGRESS,
)

SEVERITY_BLOCKING = "blocking"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

DIAG_ACTION_REQUIRED = "action_required"
DIAG_ATTENTION = "attention"
DIAG_OK = "ok"


def incomplete_data_sql_condition():
    """Données minimales manquantes (sémantique historique inchangée)."""
    return or_(
        Booking.scheduled_time.is_(None),
        func.coalesce(func.trim(Booking.customer_name), "") == "",
        func.coalesce(func.trim(Booking.pickup_location), "") == "",
        func.coalesce(func.trim(Booking.dropoff_location), "") == "",
    )


def _transfer_pending_exists_sql():
    return exists(
        select(BookingTransfer.id).where(
            and_(
                BookingTransfer.booking_id == Booking.id,
                BookingTransfer.status == TransferStatus.PENDING,
            )
        )
    )


def _past_due_pending_sql(now: datetime):
    stale_threshold = now - timedelta(hours=24)
    return and_(
        Booking.status == BookingStatus.PENDING,
        Booking.scheduled_time.isnot(None),
        Booking.scheduled_time < stale_threshold,
    )


def _driver_invariant_broken_sql():
    return and_(
        Booking.status.in_(_STATUSES_REQUIRING_DRIVER),
        Booking.driver_id.is_(None),
    )


def needs_investigation_sql_condition(now: datetime | None = None):
    """OR des règles blocking — alimente filtre et summary."""
    now = now or datetime.now(UTC)
    return or_(
        incomplete_data_sql_condition(),
        _past_due_pending_sql(now),
        _transfer_pending_exists_sql(),
        _driver_invariant_broken_sql(),
    )


def evaluate_incomplete(booking: Booking) -> bool:
    if booking.scheduled_time is None:
        return True
    if not (booking.customer_name or "").strip():
        return True
    if not (booking.pickup_location or "").strip():
        return True
    return not (booking.dropoff_location or "").strip()


def _status_key(booking: Booking) -> str:
    st = booking.status
    return st.value if hasattr(st, "value") else str(st).upper()


def _reason(
    code: str,
    *,
    severity: str,
    label: str,
    recommended_action: str | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "label": label,
        "recommended_action": recommended_action,
    }


def build_investigation_reasons(
    booking: Booking,
    *,
    created_by: dict[str, Any] | None = None,
    has_pending_transfer: bool | None = None,
    now: datetime | None = None,
    institution_present: bool | None = None,
) -> list[dict[str, Any]]:
    """Évalue les raisons (blocking / warning / info) pour un transport."""
    now = now or datetime.now(UTC)
    reasons: list[dict[str, Any]] = []

    if booking.scheduled_time is None:
        reasons.append(
            _reason(
                "MISSING_SCHEDULED_TIME",
                severity=SEVERITY_BLOCKING,
                label="Date et heure du transport manquantes",
                recommended_action="request_or_correct_schedule",
            )
        )
    if not (booking.customer_name or "").strip():
        reasons.append(
            _reason(
                "MISSING_CUSTOMER_NAME",
                severity=SEVERITY_BLOCKING,
                label="Nom du client manquant",
                recommended_action="request_or_correct_customer_name",
            )
        )
    if not (booking.pickup_location or "").strip():
        reasons.append(
            _reason(
                "MISSING_PICKUP",
                severity=SEVERITY_BLOCKING,
                label="Lieu de départ manquant",
                recommended_action="request_or_correct_pickup",
            )
        )
    if not (booking.dropoff_location or "").strip():
        reasons.append(
            _reason(
                "MISSING_DROPOFF",
                severity=SEVERITY_BLOCKING,
                label="Lieu d'arrivée manquant",
                recommended_action="request_or_correct_dropoff",
            )
        )

    if booking.status == BookingStatus.PENDING and booking.scheduled_time:
        st = booking.scheduled_time
        if st.tzinfo is None:
            st = st.replace(tzinfo=UTC)
        if st < now - timedelta(hours=24):
            reasons.append(
                _reason(
                    "PAST_DUE_PENDING_24H",
                    severity=SEVERITY_BLOCKING,
                    label=(
                        "Transport toujours en attente plus de 24 heures "
                        "après l'heure prévue"
                    ),
                    recommended_action="retry_dispatch_or_assign",
                )
            )

    pending = has_pending_transfer
    if pending is None:
        try:
            pending = (
                BookingTransfer.query.filter_by(booking_id=booking.id)
                .filter_by(status=TransferStatus.PENDING)
                .first()
                is not None
            )
        except Exception:
            pending = False
    if pending:
        reasons.append(
            _reason(
                "PENDING_TRANSFER_REQUIRES_REVIEW",
                severity=SEVERITY_BLOCKING,
                label="Transfert en attente",
                recommended_action="review_pending_transfer",
            )
        )

    status_key = _status_key(booking)
    requires_driver = status_key in {s.value for s in _STATUSES_REQUIRING_DRIVER}
    if requires_driver and getattr(booking, "driver_id", None) is None:
        reasons.append(
            _reason(
                "DRIVER_INVARIANT_BROKEN",
                severity=SEVERITY_BLOCKING,
                label="Chauffeur manquant pour un statut qui l'exige",
                recommended_action="open_investigation",
            )
        )

    created = created_by or {}
    source = created.get("source") or "unknown"
    if institution_present is None:
        institution_present = False
        cli = getattr(booking, "client", None)
        if cli is not None:
            if getattr(cli, "linked_institution_id", None):
                institution_present = True
            elif getattr(cli, "linked_institution", None) is not None:
                institution_present = True

    if source == "institution_request" and not institution_present:
        reasons.append(
            _reason(
                "MISSING_INSTITUTION",
                severity=SEVERITY_WARNING,
                label="Institution attendue non identifiée",
                recommended_action="request_institution_identification",
            )
        )

    if source == "unknown":
        reasons.append(
            _reason(
                "MISSING_CREATOR",
                severity=SEVERITY_INFO,
                label="Auteur de la création non identifié",
                recommended_action=None,
            )
        )

    return reasons


def has_blocking_reason(reasons: list[dict[str, Any]]) -> bool:
    return any(r.get("severity") == SEVERITY_BLOCKING for r in reasons)


def compute_needs_investigation_booking(
    booking: Booking,
    *,
    has_pending_transfer: bool | None = None,
    created_by: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> bool:
    """Booléen aligné sur les raisons blocking (et donc sur le SQL)."""
    reasons = build_investigation_reasons(
        booking,
        created_by=created_by,
        has_pending_transfer=has_pending_transfer,
        now=now,
    )
    return has_blocking_reason(reasons)


def build_support_diagnostic(
    reasons: list[dict[str, Any]],
    *,
    status_label: str | None = None,
    current_company_name: str | None = None,
) -> dict[str, Any]:
    """Construit le bloc support_diagnostic à partir des reasons."""
    blocking = [r for r in reasons if r.get("severity") == SEVERITY_BLOCKING]
    warnings = [r for r in reasons if r.get("severity") == SEVERITY_WARNING]

    if blocking:
        status = DIAG_ACTION_REQUIRED
        severity = SEVERITY_BLOCKING
        primary = blocking[0]
    elif warnings:
        status = DIAG_ATTENTION
        severity = SEVERITY_WARNING
        primary = warnings[0]
    else:
        status = DIAG_OK
        severity = None
        primary = None

    company = (current_company_name or "").strip() or None
    status_fr = (status_label or "").strip() or None

    if primary and primary["code"] == "MISSING_SCHEDULED_TIME":
        headline = "Horaire du transport manquant"
        if company and status_fr:
            summary = (
                f"Le transport est {status_fr.lower()} par {company}, "
                f"mais aucune date ni heure n'est renseignée."
            )
        elif company:
            summary = (
                f"Course prise en charge par {company}, "
                f"mais aucune date ni heure n'est renseignée."
            )
        else:
            summary = "Aucune date ni heure de transport n'est renseignée."
    elif primary:
        headline = primary["label"]
        parts: list[str] = []
        if status_fr and company:
            parts.append(f"Statut : {status_fr} ({company}).")
        elif status_fr:
            parts.append(f"Statut : {status_fr}.")
        parts.append(primary["label"] + ".")
        summary = " ".join(parts)
    elif status_fr and company:
        headline = f"{status_fr} — {company}"
        summary = f"Transport {status_fr.lower()} ; aucune anomalie bloquante détectée."
    else:
        headline = "Transport sans anomalie détectée"
        summary = "Aucune anomalie bloquante ni avertissement."

    return {
        "status": status,
        "severity": severity,
        "needs_investigation": bool(blocking),
        "primary_reason_code": primary["code"] if primary else None,
        "headline": headline,
        "summary": summary,
        "recommended_action": (primary.get("recommended_action") if primary else None),
        "reasons": reasons,
    }
