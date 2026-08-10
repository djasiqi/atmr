# backend/services/companies/dashboard_action_queue.py
"""Exécution idempotente d'une action de la file du dashboard entreprise
(PR3 — `schema_version=2`, additif).

Contexte:
    La file d'actions (voir ``build_dashboard_action_queue``)
    expose des items ``{action_id: "<kind>:<booking_id>", version, allowed_actions}``.
    Ce module gère leur exécution via
    ``POST /companies/me/action-queue/<action_id>/execute`` avec :

    - idempotence (même ``idempotency_key`` + même payload → même résultat rejoué,
      sans ré-exécution métier) ;
    - concurrence optimiste (``expected_version`` comparée à
      ``Booking.edit_version`` — même mécanisme que les autres flux d'édition de
      réservation, voir ``services/institutions/booking_change_service.py``).

Implémentation volontairement minimale : gère les décisions en attente
(accept/reject) directement. L'assignation chauffeur réelle reste dans les flux
dispatch existants (nécessite un choix de chauffeur, hors contrat de ce endpoint).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

from models import Booking
from models.enums import BookingStatus

logger = logging.getLogger(__name__)


CRITICAL_DELAY_DEFAULT_MINUTES = 15

_ACTIVE_DELAYABLE_STATUS_VALUES = frozenset(
    {"accepted", "assigned", "en_route", "in_progress"}
)


def delay_severity_for_minutes(
    delay_minutes: float, *, critical_delay_minutes: int
) -> str | None:
    """critical si retard >= seuil KPI, warning si > 0, sinon None."""
    if delay_minutes <= 0:
        return None
    if delay_minutes >= critical_delay_minutes:
        return "critical"
    return "warning"


def _booking_status_str(booking) -> str:
    status = (
        booking.status.value
        if hasattr(booking.status, "value")
        else str(booking.status)
    )
    return (status or "").lower()


def build_dashboard_action_queue(bookings, kpi):
    """Projection legere action_queue (v2) — n'utilise pas serialize_dashboard."""
    from models.enums import BookingStatus
    from shared.time_utils import now_local

    critical_delay_minutes = int(
        (kpi or {}).get("critical_delay_minutes") or CRITICAL_DELAY_DEFAULT_MINUTES
    )
    now = now_local()

    items = []
    for b in bookings or []:
        st = _booking_status_str(b)
        kind = None
        allowed: list[str] = []
        delay_minutes = 0.0
        if st == BookingStatus.PENDING.value.lower() or st == "pending":
            kind = "pending_decision"
            allowed = ["accept", "reject"]
        elif st in ("accepted",) and not getattr(b, "driver_id", None):
            kind = "unassigned"
            allowed = ["assign"]
        elif st in _ACTIVE_DELAYABLE_STATUS_VALUES and getattr(
            b, "scheduled_time", None
        ):
            delay_minutes = (now - b.scheduled_time).total_seconds() / 60.0
            if delay_minutes >= critical_delay_minutes:
                kind = "critical_delay"
                allowed = ["acknowledge"]
        if not kind:
            continue
        status_display = b.status.value if hasattr(b.status, "value") else str(b.status)
        item = {
            "action_id": f"{kind}:{b.id}",
            "dedupe_key": f"{kind}:{b.id}",
            "entity_type": "booking",
            "entity_id": b.id,
            # Alias explicite : les items action_queue portent toujours sur un booking.
            "booking_id": b.id,
            "kind": kind,
            "priority": {
                "critical_delay": 0,
                "pending_decision": 10,
                "unassigned": 20,
            }.get(kind, 99),
            "deadline_at": b.scheduled_time.isoformat()
            if getattr(b, "scheduled_time", None)
            else None,
            "reason": kind,
            "action_required_by": "company",
            "allowed_actions": allowed,
            "booking_summary": {
                "id": b.id,
                "status": status_display,
                "scheduled_time": b.scheduled_time.isoformat()
                if getattr(b, "scheduled_time", None)
                else None,
                "pickup_location": getattr(b, "pickup_location", None),
                "dropoff_location": getattr(b, "dropoff_location", None),
            },
            "version": int(getattr(b, "edit_version", None) or 1),
        }
        if getattr(b, "scheduled_time", None):
            if delay_minutes <= 0 and st in _ACTIVE_DELAYABLE_STATUS_VALUES:
                delay_minutes = (now - b.scheduled_time).total_seconds() / 60.0
            severity = delay_severity_for_minutes(
                delay_minutes, critical_delay_minutes=critical_delay_minutes
            )
            if severity:
                item["delay_minutes"] = round(delay_minutes, 1)
                item["delay_severity"] = severity
        items.append(item)
    items.sort(
        key=lambda x: (
            x.get("priority") or 99,
            x.get("deadline_at") or "",
            x.get("entity_id") or 0,
        )
    )
    return items


def _build_delay_summary(bookings, kpi):
    from shared.time_utils import now_local

    critical_delay_minutes = int(
        (kpi or {}).get("critical_delay_minutes") or CRITICAL_DELAY_DEFAULT_MINUTES
    )
    now = now_local()
    rows = []
    for b in bookings or []:
        st = _booking_status_str(b)
        if st not in _ACTIVE_DELAYABLE_STATUS_VALUES or not getattr(
            b, "scheduled_time", None
        ):
            continue
        delay_minutes = (now - b.scheduled_time).total_seconds() / 60.0
        severity = delay_severity_for_minutes(
            delay_minutes, critical_delay_minutes=critical_delay_minutes
        )
        if not severity:
            continue
        rows.append(
            {
                "booking_id": b.id,
                "delay_minutes": round(delay_minutes, 1),
                "delay_severity": severity,
                "scheduled_time": b.scheduled_time.isoformat(),
                "status": st,
            }
        )
    rows.sort(key=lambda r: (-(r["delay_minutes"] or 0), r["booking_id"]))
    return {
        "delay_count": int((kpi or {}).get("delay_count") or 0),
        "critical_delay_count": int((kpi or {}).get("critical_delay_count") or 0),
        "critical_delay_minutes": critical_delay_minutes,
        "items": rows,
    }


def _build_upcoming_bookings_light(bookings, _kpi, *, limit: int = 30):
    from shared.time_utils import now_local

    now = now_local()
    upcoming = []
    for b in bookings or []:
        st = _booking_status_str(b)
        if st in ("completed", "return_completed", "canceled", "cancelled"):
            continue
        scheduled = getattr(b, "scheduled_time", None)
        if scheduled is None or scheduled < now:
            continue
        upcoming.append(
            {
                "id": b.id,
                "status": st,
                "scheduled_time": scheduled.isoformat(),
                "pickup_location": getattr(b, "pickup_location", None),
                "dropoff_location": getattr(b, "dropoff_location", None),
            }
        )
    upcoming.sort(key=lambda x: (x["scheduled_time"], x["id"]))
    return upcoming[:limit]


def serialize_dashboard_v2_extras(
    bookings, kpi, *, action_queue_limit: int | None = None
):
    import os

    limit = action_queue_limit
    if limit is None:
        limit = int(os.getenv("LIRIE_DASHBOARD_ACTION_QUEUE_LIMIT", "50") or "50")
    full_action_queue = build_dashboard_action_queue(bookings, kpi)
    action_queue_total = (
        int(kpi.get("pending_decision", 0))
        + int(kpi.get("unassigned", 0))
        + int(kpi.get("critical_delay_count", 0))
    )
    action_queue = full_action_queue[:limit]
    action_queue_truncated = len(full_action_queue) > limit or action_queue_total > len(
        full_action_queue
    )
    return {
        "summary": {**(kpi or {}), "to_handle": action_queue_total},
        "action_queue": action_queue,
        "action_queue_total": action_queue_total,
        "action_queue_truncated": action_queue_truncated,
        "action_queue_next_cursor": str(limit) if action_queue_truncated else None,
        "delay_summary": _build_delay_summary(bookings, kpi),
        "upcoming_bookings": _build_upcoming_bookings_light(bookings, kpi),
    }


_IDEMPOTENCY_KEY_PREFIX = "lirie:action_queue:idem:"
_IDEMPOTENCY_TTL_SECONDS = 86400

# Repli mémoire process-local si Redis est indisponible : garantit l'idempotence
# au moins pour la durée de vie du process (dégradé mais jamais silencieusement
# désactivé — voir contrat `execute_action`). Purgé au redémarrage du process.
_MEMORY_IDEMPOTENCY_STORE: dict[str, tuple[float, dict[str, Any]]] = {}


def _current_version(booking: Booking) -> int:
    return int(getattr(booking, "edit_version", None) or 1)


def _idempotency_redis_key(
    company_id: int, action_id: str, idempotency_key: str
) -> str:
    return f"{_IDEMPOTENCY_KEY_PREFIX}{company_id}:{action_id}:{idempotency_key}"


def _payload_fingerprint(action: str, expected_version: Any) -> str:
    raw = json.dumps(
        {"action": action, "expected_version": expected_version},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _memory_store_get(key: str) -> dict[str, Any] | None:
    import time

    entry = _MEMORY_IDEMPOTENCY_STORE.get(key)
    if entry is None:
        return None
    expires_at, payload = entry
    if expires_at < time.time():
        _MEMORY_IDEMPOTENCY_STORE.pop(key, None)
        return None
    return payload


def _memory_store_set(key: str, payload: dict[str, Any]) -> None:
    import time

    # Purge best-effort pour éviter une croissance non bornée en l'absence de Redis.
    if len(_MEMORY_IDEMPOTENCY_STORE) > 5000:
        _MEMORY_IDEMPOTENCY_STORE.clear()
    _MEMORY_IDEMPOTENCY_STORE[key] = (time.time() + _IDEMPOTENCY_TTL_SECONDS, payload)


def _load_idempotent_result(redis_client: Any, redis_key: str) -> dict[str, Any] | None:
    """Lit le résultat idempotent stocké — Redis en priorité, repli mémoire sinon."""
    if redis_client is not None:
        try:
            cached_raw = redis_client.get(redis_key)
        except Exception:
            logger.warning(
                "[dashboard_action_queue] échec lecture idempotency (key=%s)",
                redis_key,
                exc_info=True,
            )
            cached_raw = None
        if cached_raw:
            try:
                return json.loads(cached_raw)
            except (TypeError, ValueError):
                return None
        return None
    return _memory_store_get(redis_key)


def _store_idempotent_result(
    redis_client: Any,
    redis_key: str,
    fingerprint: str,
    result: dict[str, Any],
    status_code: int,
) -> None:
    payload = {"fingerprint": fingerprint, "result": result, "status_code": status_code}
    if redis_client is None:
        _memory_store_set(redis_key, payload)
        return
    try:
        redis_client.setex(
            redis_key,
            _IDEMPOTENCY_TTL_SECONDS,
            json.dumps(payload, default=str),
        )
    except Exception:
        logger.warning(
            "[dashboard_action_queue] échec stockage idempotency (key=%s), repli mémoire",
            redis_key,
            exc_info=True,
        )
        _memory_store_set(redis_key, payload)


def _apply_action(
    booking: Booking, kind: str, action: str
) -> tuple[dict[str, Any], int]:
    """Applique la transition métier — voir avertissement d'implémentation minimale
    en tête de module (seul `pending_decision` mute réellement l'état)."""
    from ext import db

    normalized_action = (action or "").strip().lower()

    if kind == "pending_decision":
        if booking.status != BookingStatus.PENDING:
            return (
                {
                    "error": "invalid_state",
                    "message": "Réservation déjà traitée (statut différent de PENDING).",
                },
                409,
            )
        if normalized_action == "accept":
            booking.status = BookingStatus.ACCEPTED
        elif normalized_action in ("reject", "refuse"):
            booking.status = BookingStatus.CANCELED
            booking.cancelled_at = datetime.now(UTC)
            booking.cancelled_by_role = "company"
        else:
            return (
                {
                    "error": "invalid_action",
                    "message": f"Action '{action}' non supportée pour une décision en attente.",
                },
                400,
            )
    elif kind == "unassigned":
        return (
            {
                "error": "invalid_action",
                "message": (
                    "L'assignation d'un chauffeur nécessite le flux dispatch "
                    "(choix du chauffeur) — non disponible via cet endpoint."
                ),
            },
            400,
        )
    elif kind == "critical_delay":
        if normalized_action != "acknowledge":
            return (
                {
                    "error": "invalid_action",
                    "message": f"Action '{action}' non supportée pour un retard critique.",
                },
                400,
            )
        # Acquittement uniquement (aucune mutation métier) — la résolution réelle du
        # retard reste gérée par les flux dispatch existants (voir routes/dispatch).
    else:
        return {
            "error": "unknown_action_kind",
            "message": "Type d'action inconnu.",
        }, 400

    booking.edit_version = _current_version(booking) + 1
    booking.updated_at = datetime.now(UTC)
    db.session.commit()
    return (
        {
            "action_id": f"{kind}:{booking.id}",
            "booking_id": booking.id,
            "status": getattr(booking.status, "value", str(booking.status)).lower(),
            "new_version": booking.edit_version,
        },
        200,
    )


def execute_action(
    *,
    company_id: int,
    action_id: str,
    action: str,
    expected_version: Any,
    idempotency_key: str | None,
) -> tuple[dict[str, Any], int]:
    """Exécute une action de la file (idempotente + concurrence optimiste).

    Contrat:
        - même ``idempotency_key`` + même payload (``action``, ``expected_version``)
          → même résultat renvoyé (rejoué depuis Redis, aucune ré-exécution) ;
        - même ``idempotency_key`` + payload différent → 409 ``idempotency_conflict`` ;
        - ``expected_version`` différente de ``Booking.edit_version`` courante
          → 409 ``stale_action`` (la file affichée côté client est obsolète).
    """
    if not idempotency_key:
        return (
            {
                "error": "idempotency_key_required",
                "message": "idempotency_key est obligatoire.",
            },
            400,
        )

    try:
        kind, booking_id_str = action_id.rsplit(":", 1)
        booking_id = int(booking_id_str)
    except (ValueError, AttributeError):
        return {"error": "invalid_action_id", "message": "action_id invalide."}, 400

    fingerprint = _payload_fingerprint(action or "", expected_version)
    redis_key = _idempotency_redis_key(company_id, action_id, idempotency_key)

    from ext import redis_client

    cached = _load_idempotent_result(redis_client, redis_key)
    if cached is not None:
        if cached.get("fingerprint") == fingerprint:
            return cached["result"], cached["status_code"]
        return (
            {
                "error": "idempotency_conflict",
                "message": (
                    "Cette clé d'idempotence a déjà été utilisée avec un payload différent."
                ),
            },
            409,
        )

    booking = Booking.query.filter_by(id=booking_id, company_id=company_id).first()
    if booking is None:
        return {"error": "not_found", "message": "Réservation introuvable."}, 404

    current_version = _current_version(booking)
    try:
        expected_version_int = (
            int(expected_version) if expected_version is not None else None
        )
    except (TypeError, ValueError):
        return {
            "error": "invalid_expected_version",
            "message": "expected_version invalide.",
        }, 400
    if expected_version_int is not None and expected_version_int != current_version:
        result, status_code = (
            {
                "error": "stale_action",
                "message": "Cette action n'est plus à jour (réservation modifiée entre-temps).",
                "current_version": current_version,
            },
            409,
        )
        _store_idempotent_result(
            redis_client, redis_key, fingerprint, result, status_code
        )
        return result, status_code

    result, status_code = _apply_action(booking, kind, action)
    _store_idempotent_result(redis_client, redis_key, fingerprint, result, status_code)
    return result, status_code
