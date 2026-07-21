# services/events/institution_events.py
"""ÉTAPE 5: Événements temps réel pour les institutions.

Ce module gère les événements Socket.IO envoyés aux institutions:
- request_sent: Demande envoyée aux transporteurs
- offer_accepted: Offre acceptée (socket UI ; pas de notif cloche — voir request_converted)
- request_converted: Demande convertie en booking (+ notification cloche unique)
- booking_status_updated: Statut du booking mis à jour
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime
from typing import Any

from ext import db, socketio
from schemas.socket_events import EVENT_VERSION, SocketEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Room helpers pour institutions
# ---------------------------------------------------------------------------
def get_institution_room(institution_id: int) -> str:
    """Room d'institution (ex: institution_42)."""
    return f"institution_{institution_id}"


# ---------------------------------------------------------------------------
# Émission vers institutions
# ---------------------------------------------------------------------------
def emit_institution_event(
    institution_id: int,
    event_name: str,
    payload: dict[str, Any],
) -> bool:
    """Émet un événement Socket.IO vers une institution.

    Args:
        institution_id: ID de l'institution
        event_name: Nom de l'événement
        payload: Données de l'événement

    Returns:
        True si émis, False si erreur
    """
    if not institution_id:
        logger.warning("[InstitutionEvents] institution_id manquant, skip emit")
        return False

    try:
        room = get_institution_room(institution_id)

        # Enrichir le payload avec métadonnées
        enriched = SocketEvent.create(
            event_type=event_name,
            payload=payload,
            version=EVENT_VERSION,
        )

        socketio.emit(
            event_name,
            enriched,
            to=room,
            namespace="/",
        )

        logger.debug(
            "[InstitutionEvents] Emitted %s to %s",
            event_name,
            room,
        )
        return True

    except Exception as e:
        logger.error(
            "[InstitutionEvents] Error emitting %s to institution %s: %s",
            event_name,
            institution_id,
            e,
        )
        return False


# ---------------------------------------------------------------------------
# Notification persistence helper
# ---------------------------------------------------------------------------
_EVENT_TITLES: dict[str, str] = {
    "request_sent": "Demande envoyée",
    "offer_accepted": "Transporteur trouvé",
    "request_converted": "Transport confirmé",
    "booking_status_updated": "Mise à jour transport",
    "request_cancelled": "Demande annulée",
    "booking_cancelled": "Transport annulé",
    "booking_message": "Nouveau message",
}

_BOOKING_STATUS_LABELS: dict[str, str] = {
    "pending": "En attente",
    "confirmed": "Confirmé",
    "assigned": "Assigné au chauffeur",
    "en_route": "En route",
    "in_progress": "En cours",
    "completed": "Terminé",
    "cancelled": "Annulé",
}


def _build_dedupe_key(
    event_type: str,
    metadata: dict[str, Any] | None,
    explicit_key: str | None = None,
) -> str | None:
    """Construit un dedupe_key stable a partir du contexte.

    Format: "{event_type}:{booking_id}:{status}:{actor_role}:{actor_id}"
    Retourne None si pas de booking_id (notifications non-booking = pas de dedup).
    """
    if explicit_key:
        return explicit_key
    if not metadata:
        return None
    booking_id = metadata.get("booking_id")
    if not booking_id:
        return None
    status = metadata.get("new_status") or metadata.get("status") or ""
    actor_role = metadata.get("actor_role") or ""
    actor_id = metadata.get("actor_id") or ""
    return f"{event_type}:{booking_id}:{status}:{actor_role}:{actor_id}"


def _persist_notification(
    institution_id: int,
    event_type: str,
    message: str,
    metadata: dict[str, Any] | None = None,
    dedupe_key: str | None = None,
) -> dict[str, Any] | None:
    """Persiste une notification en base et emet un evenement socket dedie.

    Args:
        dedupe_key: si fourni, evite les doublons via contrainte unique
            (institution_id, dedupe_key). Auto-genere depuis metadata sinon.

    Returns:
        Le dict serialise de la notification ou None si erreur / doublon.
    """
    resolved_key = _build_dedupe_key(event_type, metadata, dedupe_key)
    try:
        from models.institution_notification import InstitutionNotification

        title = _EVENT_TITLES.get(event_type, "Notification")

        notif = InstitutionNotification()
        notif.institution_id = institution_id
        notif.event_type = event_type
        notif.title = title
        notif.message = message
        notif.metadata_json = metadata or {}
        if resolved_key:
            notif.dedupe_key = resolved_key
        db.session.add(notif)
        db.session.commit()

        # Émettre un événement socket dédié pour le temps réel
        serialized = notif.serialize
        emit_institution_event(
            institution_id=institution_id,
            event_name="new_notification",
            payload=serialized,
        )

        logger.info(
            "[notification_persisted] table=institution_notifications institution_id=%s dedupe_key=%s result=ok",
            institution_id,
            resolved_key,
        )
        return serialized

    except Exception as e:
        with contextlib.suppress(Exception):
            db.session.rollback()
        exc_str = str(e).lower()
        if resolved_key and ("unique" in exc_str or "duplicate" in exc_str):
            logger.info(
                "[notification_persisted] table=institution_notifications dedupe_key=%s result=duplicate",
                resolved_key,
            )
            return None
        logger.error(
            "[InstitutionEvents] Error persisting notification %s: %s",
            event_type,
            e,
        )
        return None


# ---------------------------------------------------------------------------
# Événements spécifiques
# ---------------------------------------------------------------------------
def emit_request_sent(
    institution_id: int,
    request_id: int,
    public_id: str,
    external_reference: str | None,
    mode: str,  # "sequential" ou "broadcast"
    offers_created: int,
) -> bool:
    """Émet l'événement request_sent.

    Appelé après envoi d'une demande aux transporteurs.
    """
    _persist_notification(
        institution_id=institution_id,
        event_type="request_sent",
        message=f"Demande #{request_id} envoyée à {offers_created} transporteur(s)",
        metadata={
            "request_id": request_id,
            "public_id": public_id,
            "external_reference": external_reference,
        },
    )
    return emit_institution_event(
        institution_id=institution_id,
        event_name="request_sent",
        payload={
            "request_id": request_id,
            "public_id": public_id,
            "external_reference": external_reference,
            "mode": mode,
            "offers_created": offers_created,
            "sent_at": datetime.now(UTC).isoformat(),
        },
    )


def emit_offer_accepted(
    institution_id: int,
    request_id: int,
    public_id: str,
    offer_id: int,
    company_name: str | None = None,  # Optionnel, peut être masqué
) -> bool:
    """Émet l'événement offer_accepted (socket uniquement).

    Pas de notification cloche ici : ``emit_request_converted`` couvre
    le même moment métier (acceptation = conversion) et évite le doublon
    « Transporteur trouvé » / « Transport confirmé ».
    """
    return emit_institution_event(
        institution_id=institution_id,
        event_name="offer_accepted",
        payload={
            "request_id": request_id,
            "public_id": public_id,
            "offer_id": offer_id,
            "company_name": company_name,
            "accepted_at": datetime.now(UTC).isoformat(),
        },
    )


def emit_request_converted(
    institution_id: int,
    request_id: int,
    public_id: str,
    booking_id: int,
    company_id: int | None = None,  # Masqué si None
    company_name: str | None = None,
) -> bool:
    """Émet l'événement request_converted + notification cloche unique.

    Appelé quand une demande est convertie en booking (acceptation transporteur).
    """
    name = company_name or "un transporteur"
    _persist_notification(
        institution_id=institution_id,
        event_type="request_converted",
        message=f"Demande #{request_id} confirmée par {name}",
        metadata={
            "request_id": request_id,
            "public_id": public_id,
            "booking_id": booking_id,
            "company_name": company_name,
        },
        dedupe_key=f"request_converted:{request_id}",
    )
    return emit_institution_event(
        institution_id=institution_id,
        event_name="request_converted",
        payload={
            "request_id": request_id,
            "public_id": public_id,
            "booking_id": booking_id,
            "company_id": company_id,
            "company_name": company_name,
            "converted_at": datetime.now(UTC).isoformat(),
        },
    )


def emit_booking_status_updated(
    institution_id: int,
    booking_id: int,
    request_id: int | None,
    public_id: str | None,
    old_status: str,
    new_status: str,
    driver_name: str | None = None,  # Optionnel
    eta: str | None = None,  # ISO8601 si disponible
) -> bool:
    """Émet l'événement booking_status_updated.

    Appelé lors des changements de statut: EN_ROUTE, IN_PROGRESS, COMPLETED, etc.
    """
    status_label = _BOOKING_STATUS_LABELS.get(new_status, new_status)
    driver_info = f" par {driver_name}" if driver_name else ""
    req_ref = f"demande #{request_id}" if request_id else f"transport #{booking_id}"
    _persist_notification(
        institution_id=institution_id,
        event_type="booking_status_updated",
        message=f"Statut {req_ref}: {status_label}{driver_info}",
        metadata={
            "booking_id": booking_id,
            "request_id": request_id,
            "public_id": public_id,
            "old_status": old_status,
            "new_status": new_status,
            "driver_name": driver_name,
        },
    )
    return emit_institution_event(
        institution_id=institution_id,
        event_name="booking_status_updated",
        payload={
            "booking_id": booking_id,
            "request_id": request_id,
            "public_id": public_id,
            "old_status": old_status,
            "new_status": new_status,
            "driver_name": driver_name,
            "eta": eta,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def emit_booking_assigned_to_institution(
    institution_id: int,
    booking_id: int,
    request_id: int | None,
    public_id: str | None,
) -> bool:
    """Notifie l'institution qu'un chauffeur a ete assigne a sa course."""
    req_ref = f"demande #{request_id}" if request_id else f"transport #{booking_id}"
    _persist_notification(
        institution_id=institution_id,
        event_type="booking_assigned",
        message=f"Chauffeur assigne pour {req_ref}",
        metadata={
            "booking_id": booking_id,
            "request_id": request_id,
            "public_id": public_id,
        },
    )
    return emit_institution_event(
        institution_id=institution_id,
        event_name="booking_assigned",
        payload={
            "booking_id": booking_id,
            "request_id": request_id,
            "public_id": public_id,
            "assigned_at": datetime.now(UTC).isoformat(),
        },
    )


def emit_request_cancelled(
    institution_id: int,
    request_id: int,
    public_id: str,
    reason: str | None = None,
) -> bool:
    """Émet l'événement request_cancelled.

    Appelé quand une demande est annulée par l'institution.
    """
    reason_info = f" — {reason}" if reason else ""
    _persist_notification(
        institution_id=institution_id,
        event_type="request_cancelled",
        message=f"Demande #{request_id} annulée{reason_info}",
        metadata={
            "request_id": request_id,
            "public_id": public_id,
            "reason": reason,
        },
    )
    return emit_institution_event(
        institution_id=institution_id,
        event_name="request_cancelled",
        payload={
            "request_id": request_id,
            "public_id": public_id,
            "reason": reason,
            "cancelled_at": datetime.now(UTC).isoformat(),
        },
    )


def emit_booking_cancelled(
    institution_id: int,
    booking_id: int,
    request_id: int | None,
    public_id: str | None,
    is_billable: bool,
    reason_code: str | None = None,
    display_label: str | None = None,
) -> bool:
    """Émet l'événement booking_cancelled.

    Appelé quand un booking est annulé.
    """
    req_ref = f"demande #{request_id}" if request_id else f"transport #{booking_id}"
    parts = [f"Transport {req_ref} annulé par l'entreprise"]
    if display_label:
        parts.append(f"— {display_label}")
    if is_billable:
        parts.append("(facturé)")
    else:
        parts.append("(non facturé)")
    message = " ".join(parts)

    _persist_notification(
        institution_id=institution_id,
        event_type="booking_cancelled",
        message=message,
        metadata={
            "booking_id": booking_id,
            "request_id": request_id,
            "public_id": public_id,
            "is_billable": is_billable,
            "reason_code": reason_code,
            "display_label": display_label,
        },
    )
    return emit_institution_event(
        institution_id=institution_id,
        event_name="booking_cancelled",
        payload={
            "booking_id": booking_id,
            "request_id": request_id,
            "public_id": public_id,
            "is_billable": is_billable,
            "reason_code": reason_code,
            "display_label": display_label,
            "cancelled_at": datetime.now(UTC).isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Mini-chat booking: emission bidirectionnelle
# ---------------------------------------------------------------------------
def emit_booking_message(
    company_id: int | None,
    institution_id: int | None,
    booking_id: int,
    message_data: dict[str, Any],
    sender_type: str | None = None,
    request_id: int | None = None,
) -> bool:
    """Emet un message booking vers les rooms company et institution.

    Si sender_type == "COMPANY" et institution_id, persiste une notification
    côté institution pour la cloche.

    Guards null: n'emet pas vers une room si l'ID est None.
    """
    payload = {
        "booking_id": booking_id,
        "message": message_data,
    }
    emitted = False
    try:
        if company_id:
            socketio.emit("booking_message", payload, to=f"company_{company_id}")
            emitted = True
        if institution_id:
            socketio.emit(
                "booking_message", payload, to=f"institution_{institution_id}"
            )
            emitted = True

        # Propriétaire du booking (portail client) — reçoit aussi les réponses entreprise / institution
        try:
            from models.booking import Booking as BookingRow
            from models.client import Client as ClientRow

            b_row = db.session.get(BookingRow, booking_id)
            if b_row and getattr(b_row, "client_id", None):
                cli_row = getattr(b_row, "client", None) or ClientRow.query.get(
                    b_row.client_id
                )
                u_row = getattr(cli_row, "user", None) if cli_row else None
                pub = (getattr(u_row, "public_id", None) or "").strip()
                if pub:
                    socketio.emit("booking_message", payload, to=f"client_{pub}")
                    emitted = True
        except Exception as room_err:
            logger.debug(
                "[BookingChat] client portal room emit skipped booking=%s: %s",
                booking_id,
                room_err,
            )

        if emitted:
            logger.debug(
                "[BookingChat] Emitted booking_message for booking %s (company=%s, institution=%s)",
                booking_id,
                company_id,
                institution_id,
            )

        # Persist notification for the *receiving* side
        sender_label = message_data.get("sender_label", "")
        content_preview = (message_data.get("content") or "")[:80]

        if sender_type == "COMPANY" and institution_id:
            _persist_notification(
                institution_id=institution_id,
                event_type="booking_message",
                message=f"{sender_label}: {content_preview}",
                metadata={
                    "booking_id": booking_id,
                    "request_id": request_id,
                    "sender_label": sender_label,
                },
            )

        if sender_type == "INSTITUTION" and company_id:
            persist_company_notification(
                company_id=company_id,
                event_type="booking_message",
                title="Nouveau message",
                message=f"{sender_label}: {content_preview}",
                metadata={
                    "booking_id": booking_id,
                    "sender_label": sender_label,
                },
            )

        if sender_type == "CLIENT" and company_id:
            persist_company_notification(
                company_id=company_id,
                event_type="booking_message",
                title="Message client",
                message=f"{sender_label}: {content_preview}",
                metadata={
                    "booking_id": booking_id,
                    "sender_label": sender_label,
                },
            )

        if sender_type == "CLIENT" and institution_id:
            _persist_notification(
                institution_id=institution_id,
                event_type="booking_message",
                message=f"{sender_label}: {content_preview}",
                metadata={
                    "booking_id": booking_id,
                    "request_id": request_id,
                    "sender_label": sender_label,
                },
            )

    except Exception as e:
        logger.error("[BookingChat] Error emitting booking_message: %s", e)
        return False

    return emitted


# ---------------------------------------------------------------------------
# Company notification persistence
# ---------------------------------------------------------------------------
def persist_company_notification(
    company_id: int,
    event_type: str,
    title: str,
    message: str,
    metadata: dict[str, Any] | None = None,
    dedupe_key: str | None = None,
) -> dict[str, Any] | None:
    """Persiste une notification company et emet un evenement socket.

    Args:
        dedupe_key: cle de deduplication explicite. Auto-generee depuis metadata sinon.
    """
    resolved_key = _build_dedupe_key(event_type, metadata, dedupe_key)
    try:
        from models.company_notification import CompanyNotification

        notif = CompanyNotification()
        notif.company_id = company_id
        notif.event_type = event_type
        notif.title = title
        notif.message = message
        notif.metadata_json = metadata or {}
        if resolved_key:
            notif.dedupe_key = resolved_key
        db.session.add(notif)
        db.session.commit()

        serialized = notif.serialize
        socketio.emit(
            "new_company_notification",
            serialized,
            to=f"company_{company_id}",
            namespace="/",
        )

        logger.info(
            "[notification_persisted] table=company_notifications company_id=%s dedupe_key=%s result=ok",
            company_id,
            resolved_key,
        )
        return serialized

    except Exception as e:
        with contextlib.suppress(Exception):
            db.session.rollback()
        exc_str = str(e).lower()
        if resolved_key and ("unique" in exc_str or "duplicate" in exc_str):
            logger.info(
                "[notification_persisted] table=company_notifications dedupe_key=%s result=duplicate",
                resolved_key,
            )
            return None
        logger.error("[CompanyNotif] Error persisting %s: %s", event_type, e)
        return None


# ---------------------------------------------------------------------------
# Helper pour retrouver l'institution d'un booking
# ---------------------------------------------------------------------------
def get_institution_from_booking(booking_id: int) -> int | None:
    """Retrouve l'institution_id d'un booking via sa TransportRequest source.

    Args:
        booking_id: ID du booking

    Returns:
        institution_id ou None si non trouvé
    """
    from models import TransportRequest

    transport_req = TransportRequest.query.filter_by(
        booking_id=booking_id,
    ).first()

    if transport_req:
        return transport_req.institution_id

    return None


def emit_offer_unavailable(
    *,
    company_id: int,
    offer_id: int,
    transport_request: Any,
    reason: str = "accepted_by_peer",
    accepted_by_company_id: int | None = None,
    accepted_by_company_name: str | None = None,
) -> bool:
    """Notifie une entreprise qu'une offre n'est plus disponible (broadcast concurrent)."""
    if not company_id or not offer_id:
        return False

    metadata: dict[str, Any] = {
        "offer_id": offer_id,
        "request_id": transport_request.id,
        "public_id": str(getattr(transport_request, "public_id", "")),
        "reason": reason,
        "accepted_by_company_id": accepted_by_company_id,
        "accepted_by_company_name": accepted_by_company_name,
    }
    institution = getattr(transport_request, "institution", None)
    institution_name = getattr(institution, "name", None) if institution else None
    if institution_name:
        metadata["institution_name"] = institution_name

    title = "Demande plus disponible"
    message = (
        f"La demande #{transport_request.id} a été acceptée par un autre transporteur"
    )
    if institution_name:
        message = f"{institution_name} — demande déjà prise par un autre transporteur"

    dedupe_key = f"offer_unavailable:{transport_request.id}:{company_id}:{reason}"

    try:
        notif = persist_company_notification(
            company_id=company_id,
            event_type="offer_unavailable",
            title=title,
            message=message,
            metadata=metadata,
            dedupe_key=dedupe_key,
        )
        if notif is None:
            return True

        from services.realtime.socketio import emit_company_event

        emit_company_event(
            company_id,
            "offer_unavailable",
            SocketEvent.create(
                "offer_unavailable",
                {
                    **metadata,
                    "transport_request_id": transport_request.id,
                    "institution_name": institution_name,
                },
            ),
        )

        from services.metrics.institution_metrics import track_offer_unavailable_emitted

        track_offer_unavailable_emitted(
            company_id=company_id,
            offer_id=offer_id,
            transport_request_id=transport_request.id,
            reason=reason,
        )

        try:
            from services.notifications.institution_new_request_push import (
                enqueue_institution_company_push_message,
            )
            from services.notifications.push_message_builder import (
                build_push_for_institution_offer_unavailable,
            )

            push_msg = build_push_for_institution_offer_unavailable(
                transport_request=transport_request,
                offer_id=offer_id,
                company_id=company_id,
                institution_name=institution_name,
                title=title,
                message=message,
                dedupe_key=dedupe_key,
                reason=reason,
            )
            enqueue_institution_company_push_message(
                company_id=company_id,
                msg=push_msg,
            )
        except Exception as push_err:
            logger.warning(
                "[InstitutionEvents] Push offer_unavailable company=%s: %s",
                company_id,
                push_err,
            )
        return True
    except Exception as e:
        logger.error(
            "[InstitutionEvents] Error emitting offer_unavailable to company %s: %s",
            company_id,
            e,
        )
        return False


def _format_request_info(transport_req: Any) -> dict[str, Any]:
    return {
        "request_id": transport_req.id,
        "public_id": transport_req.public_id,
        "institution_id": transport_req.institution_id,
        "external_reference": transport_req.external_reference,
    }


def get_request_info_from_booking(booking_id: int) -> dict[str, Any] | None:
    """Retrouve les infos de la TransportRequest source d'un booking.

    Gère les bookings retour (is_return=True) en résolvant via parent_booking_id.

    Args:
        booking_id: ID du booking

    Returns:
        Dict avec request_id, public_id, institution_id ou None
    """
    from ext import db
    from models import Booking, TransportRequest

    transport_req = TransportRequest.query.filter_by(
        booking_id=booking_id,
    ).first()
    if transport_req:
        return _format_request_info(transport_req)

    # Fallback: booking retour → résoudre via parent_booking_id
    booking = db.session.get(Booking, booking_id)
    if booking is not None:
        is_return = bool(getattr(booking, "is_return", False))
        parent_booking_id = getattr(booking, "parent_booking_id", None)
    else:
        is_return = False
        parent_booking_id = None

    if is_return and parent_booking_id:
        transport_req = TransportRequest.query.filter_by(
            booking_id=parent_booking_id,
        ).first()
        if transport_req:
            return _format_request_info(transport_req)

    return None
