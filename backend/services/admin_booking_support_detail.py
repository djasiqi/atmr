"""Projection DTO et chronologie pour la console support admin (lecture seule)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from models import Booking, BookingStatus
from security.audit_log import AuditLog
from services.admin_booking_investigation import (
    build_investigation_reasons,
    build_support_diagnostic,
)
from services.admin_booking_labels import booking_status_label_fr

_MAX_DETAIL_KEYS = 12
_MAX_DETAIL_DEPTH = 2
_MAX_DETAIL_STR_LEN = 200
_SENSITIVE_KEY_FRAGMENTS = (
    "password",
    "secret",
    "token",
    "authorization",
    "api_key",
    "apikey",
    "cookie",
    "credential",
    "private_key",
)

# Présentation allowlistée des audits (clés autorisées dans details).
AUDIT_PRESENTATION: dict[str, dict[str, Any]] = {
    "booking_created_from_request": {
        "label": "Transport créé depuis une demande",
        "allowed_keys": {"request_id", "institution_id", "company_id"},
        "dedupe_create": True,
    },
    "booking_released_for_redispatch": {
        "label": "Transport relâché pour redispatch",
        "allowed_keys": {"reason", "previous_company_id"},
    },
    "booking_billing_updated": {
        "label": "Facturation mise à jour",
        "allowed_keys": {"field", "from", "to"},
    },
    "dispatch_complete": {
        "label": "Dispatch terminé",
        "allowed_keys": {"result", "companies_count", "mode"},
    },
    "driver_assign": {
        "label": "Chauffeur affecté",
        "allowed_keys": {"driver_id", "company_id"},
    },
    "booking_update": {
        "label": "Transport modifié",
        "allowed_keys": {"fields", "reason"},
    },
    "booking_status_changed": {
        "label": "Statut modifié",
        "allowed_keys": {"from_status", "to_status", "reason"},
    },
    "STATUS_CHANGED": {
        "label": "Statut modifié",
        "allowed_keys": {"from_status", "to_status", "reason", "from", "to"},
    },
}

_CREATE_AUDIT_TYPES = frozenset(
    k for k, v in AUDIT_PRESENTATION.items() if v.get("dedupe_create")
) | frozenset({"booking_created", "create_booking"})


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def _age_seconds(dt: datetime | None, *, now: datetime | None = None) -> int | None:
    if dt is None:
        return None
    now = now or datetime.now(UTC)
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    now_aware = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return max(0, int((now_aware - aware).total_seconds()))


def _actor_ref(*, id_: int | None = None, label: str | None = None) -> dict[str, Any] | None:
    if id_ is None and not (label or "").strip():
        return None
    return {"id": id_, "label": (label or "").strip() or None}


def serialize_admin_support_transport(
    booking: Booking, *, now: datetime | None = None
) -> dict[str, Any]:
    """Projection minimale — sans PII médicale / codes / GPS / paiement."""
    now = now or datetime.now(UTC)
    status_val = booking.status
    key = status_val.value if hasattr(status_val, "value") else str(status_val).upper()
    scheduled = getattr(booking, "scheduled_time", None)
    created = getattr(booking, "created_at", None)
    updated = getattr(booking, "updated_at", None)
    cancelled = getattr(booking, "cancelled_at", None)
    amount = getattr(booking, "amount", None)
    is_canceled = status_val == BookingStatus.CANCELED

    return {
        "status": key.lower(),
        "status_label": booking_status_label_fr(status_val),
        "scheduled_at": _iso(scheduled),
        "pickup": (booking.pickup_location or None),
        "dropoff": (booking.dropoff_location or None),
        "amount_chf": float(amount) if amount is not None else None,
        "mission_type": getattr(booking, "mission_type", None),
        "is_round_trip": bool(getattr(booking, "is_round_trip", False)),
        "is_return": bool(getattr(booking, "is_return", False)),
        "created_at": _iso(created),
        "last_updated_at": _iso(updated),
        "last_updated_age_seconds": _age_seconds(updated, now=now),
        "edit_version": int(getattr(booking, "edit_version", None) or 1),
        "cancelled_at": _iso(cancelled) if is_canceled else None,
    }


def _sanitize_details(
    raw: Any,
    *,
    allowed_keys: set[str] | None,
    depth: int = 0,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    if depth > _MAX_DETAIL_DEPTH:
        return None
    out: dict[str, Any] = {}
    for i, (k, v) in enumerate(raw.items()):
        if i >= _MAX_DETAIL_KEYS:
            break
        key = str(k)
        low = key.lower()
        if any(frag in low for frag in _SENSITIVE_KEY_FRAGMENTS):
            continue
        if allowed_keys is not None and key not in allowed_keys:
            continue
        if isinstance(v, dict):
            nested = _sanitize_details(
                v, allowed_keys=None, depth=depth + 1
            )
            if nested:
                out[key] = nested
        elif isinstance(v, (list, tuple)):
            out[key] = [
                (
                    str(x)[:_MAX_DETAIL_STR_LEN]
                    if not isinstance(x, (int, float, bool, type(None)))
                    else x
                )
                for x in list(v)[:_MAX_DETAIL_KEYS]
            ]
        elif isinstance(v, (int, float, bool)) or v is None:
            out[key] = v
        else:
            out[key] = str(v)[:_MAX_DETAIL_STR_LEN]
    return out or None


def _detail_from_filtered(details: dict[str, Any] | None) -> str | None:
    if not details:
        return None
    parts = [f"{k}={v}" for k, v in details.items()]
    text = ", ".join(parts)
    return text[:_MAX_DETAIL_STR_LEN] if text else None


def _present_audit_event(row: AuditLog) -> dict[str, Any] | None:
    action_type = row.action_type or ""
    try:
        raw_details = json.loads(row.action_details or "{}")
    except json.JSONDecodeError:
        raw_details = {}

    presentation = AUDIT_PRESENTATION.get(action_type)
    if presentation is None:
        return {
            "type": f"audit:{action_type}" if action_type else "audit:unknown",
            "at": _iso(row.created_at),
            "label": "Événement technique",
            "detail": "Un événement d'audit a été enregistré.",
            "actor": row.user_type,
            "source": "audit",
            "details": None,
        }

    allowed = set(presentation.get("allowed_keys") or ())
    filtered = _sanitize_details(raw_details, allowed_keys=allowed)
    return {
        "type": f"audit:{action_type}",
        "at": _iso(row.created_at),
        "label": presentation["label"],
        "detail": _detail_from_filtered(filtered),
        "actor": row.user_type,
        "source": "audit",
        "details": filtered,
    }


def build_support_timeline(
    booking: Booking,
    *,
    created_by: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Chronologie métier + audits filtrés."""
    timeline: list[dict[str, Any]] = []
    created_at = getattr(booking, "created_at", None)
    create_detail = None
    if created_by:
        if created_by.get("label"):
            create_detail = created_by["label"]
        elif created_by.get("source") == "unknown":
            create_detail = "Auteur non identifié"

    if created_at is not None:
        timeline.append(
            {
                "type": "transport_created",
                "at": _iso(created_at),
                "label": "Transport créé",
                "detail": create_detail,
                "actor": None,
                "source": "booking",
                "details": None,
            }
        )

    tl_inst = None
    try:
        tl_inst = booking._get_institution_timeline()
    except Exception:
        tl_inst = None
    if tl_inst:
        if tl_inst.get("sent_at"):
            timeline.append(
                {
                    "type": "request_sent",
                    "at": tl_inst["sent_at"],
                    "label": "Demande envoyée (institution)",
                    "detail": tl_inst.get("institution_name"),
                    "actor": None,
                    "source": "institution",
                    "details": None,
                }
            )
        if tl_inst.get("accepted_at"):
            timeline.append(
                {
                    "type": "request_accepted",
                    "at": tl_inst["accepted_at"],
                    "label": "Acceptée par entreprise",
                    "detail": tl_inst.get("accepted_by_company_name"),
                    "actor": None,
                    "source": "institution",
                    "details": None,
                }
            )

    cancelled_at = getattr(booking, "cancelled_at", None)
    if cancelled_at is not None and booking.status == BookingStatus.CANCELED:
        role = booking.cancelled_by_role
        timeline.append(
            {
                "type": "cancelled",
                "at": _iso(cancelled_at),
                "label": "Annulation",
                "detail": str(role) if role else None,
                "actor": role,
                "source": "booking",
                "details": None,
            }
        )

    create_at_iso = _iso(created_at)
    audit_rows = (
        AuditLog.query.filter_by(booking_id=booking.id)
        .order_by(AuditLog.created_at.asc())
        .limit(200)
        .all()
    )
    for row in audit_rows:
        action_type = row.action_type or ""
        # Dédup : audit de création redondant avec transport_created
        if action_type in _CREATE_AUDIT_TYPES and create_at_iso:
            row_at = _iso(row.created_at)
            if row_at and row_at[:19] == create_at_iso[:19]:
                continue
        presented = _present_audit_event(row)
        if presented:
            timeline.append(presented)

    timeline.sort(key=lambda x: (x.get("at") or "",))
    return timeline


def build_support_actors(
    booking: Booking,
    *,
    created_by: dict[str, Any] | None,
    cancelled_by: dict[str, Any] | None,
    previous_company: dict[str, Any] | None,
) -> dict[str, Any]:
    client_label = booking.customer_full_name
    client_id = None
    institution = None
    cli = booking.client
    if cli is not None:
        client_id = getattr(cli, "id", None)
        li = getattr(cli, "linked_institution", None)
        if li is not None:
            institution = _actor_ref(id_=li.id, label=getattr(li, "name", None))
        elif getattr(cli, "linked_institution_id", None):
            institution = _actor_ref(
                id_=cli.linked_institution_id,
                label=getattr(cli, "institution_name", None),
            )

    requester = None
    if created_by and created_by.get("source") != "unknown":
        requester = _actor_ref(label=created_by.get("label"))
        if created_by.get("source") == "institution_request" and created_by.get(
            "institution_name"
        ):
            requester = {
                "id": None,
                "label": created_by.get("label"),
                "institution_name": created_by.get("institution_name"),
                "source": "institution_request",
            }
        elif created_by:
            requester = {
                "id": None,
                "label": created_by.get("label"),
                "source": created_by.get("source"),
            }

    current = booking.executing_company or booking.company
    current_company = (
        _actor_ref(id_=current.id, label=current.name) if current else None
    )
    executing = None
    if booking.executing_company_id and booking.executing_company:
        executing = _actor_ref(
            id_=booking.executing_company.id,
            label=booking.executing_company.name,
        )

    previous = None
    if previous_company:
        previous = _actor_ref(
            id_=previous_company.get("id"),
            label=previous_company.get("name"),
        )

    driver = None
    driver_id = getattr(booking, "driver_id", None)
    if driver_id is not None:
        label = None
        drv = getattr(booking, "driver", None)
        if drv is not None:
            from shared.driver_display import format_driver_display_name

            try:
                label = format_driver_display_name(drv)
            except Exception:
                label = getattr(booking, "driver_name", None)
        else:
            label = getattr(booking, "driver_name", None)
        driver = _actor_ref(id_=driver_id, label=label)

    cancelled_actor = None
    if cancelled_by and booking.status == BookingStatus.CANCELED:
        cancelled_actor = {
            "role": cancelled_by.get("role"),
            "cancelled_at": cancelled_by.get("cancelled_at"),
            "reason_code": cancelled_by.get("reason_code"),
        }

    return {
        "client": _actor_ref(id_=client_id, label=client_label),
        "requester": requester,
        "institution": institution,
        "current_company": current_company,
        "executing_company": executing,
        "previous_company": previous,
        "driver": driver,
        "cancelled_by": cancelled_actor,
    }


def build_admin_support_detail_payload(
    booking: Booking,
    *,
    created_by: dict[str, Any],
    cancelled_by: dict[str, Any] | None,
    previous_company: dict[str, Any] | None,
    has_pending_transfer: bool | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Contrat GET /admin/bookings/:id (console support)."""
    now = now or datetime.now(UTC)
    transport = serialize_admin_support_transport(booking, now=now)
    institution_present = False
    cli = booking.client
    if cli is not None:
        if getattr(cli, "linked_institution_id", None) or getattr(
            cli, "linked_institution", None
        ):
            institution_present = True

    reasons = build_investigation_reasons(
        booking,
        created_by=created_by,
        has_pending_transfer=has_pending_transfer,
        now=now,
        institution_present=institution_present,
    )
    current = booking.executing_company or booking.company
    diagnostic = build_support_diagnostic(
        reasons,
        status_label=transport["status_label"],
        current_company_name=current.name if current else None,
    )
    actors = build_support_actors(
        booking,
        created_by=created_by,
        cancelled_by=cancelled_by,
        previous_company=previous_company,
    )
    timeline = build_support_timeline(booking, created_by=created_by)

    return {
        "id": booking.id,
        "transport": transport,
        "support_diagnostic": diagnostic,
        "actors": actors,
        "timeline": timeline,
        "references": {"booking_id": booking.id},
    }
