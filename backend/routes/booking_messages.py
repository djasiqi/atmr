# routes/booking_messages.py
# pyright: reportArgumentType=false, reportOperatorIssue=false
"""Mini-canal de communication booking.

Endpoints:
- GET  /api/v1/bookings/<booking_id>/messages  — Historique paginé (before_id)
- POST /api/v1/bookings/<booking_id>/messages  — Envoyer un message (1-500 chars)
"""
# ruff: noqa: I001
from __future__ import annotations

import logging

from flask import abort, request as flask_request
from flask_jwt_extended import get_jwt, get_jwt_identity, verify_jwt_in_request
from flask_restx import Namespace, Resource

from ext import db
from models.booking import Booking
from models.booking_transfer import BookingTransfer
from models.booking_message import BookingMessage, BookingMessageSender
from models.client import Client
from models.enums import TransferStatus, UserRole
from models.transport_request import TransportRequest
from models.user import User

MAX_MESSAGE_LENGTH = 500

logger = logging.getLogger(__name__)

booking_messages_ns = Namespace(
    "booking_messages",
    description="Messages booking (institution <-> entreprise, client <-> entreprise)",
)


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
def _get_booking_with_auth(booking_id: int):  # noqa: RET503
    """Charge le booking, verifie l'acces.

    Retourne
    (booking, sender_type, sender_label, institution_id, request_id, peer_company_id).

    - abort(404) si booking introuvable ou aucun canal disponible
    - abort(403) si l'utilisateur n'appartient ni a la company ni a l'institution
      (ou au partenaire transferé)
    """
    booking = Booking.query.get_or_404(booking_id)

    # A/R: rediriger le retour vers l'aller (thread unifié R2)
    effective_booking_id = booking.id
    if booking.is_return and booking.parent_booking_id:
        effective_booking_id = booking.parent_booking_id
        booking = Booking.query.get_or_404(effective_booking_id)

    verify_jwt_in_request()
    claims = get_jwt()

    source_req = TransportRequest.query.filter_by(booking_id=effective_booking_id).first()
    institution_id = source_req.institution_id if source_req else None
    request_id = source_req.id if source_req else None

    # Canal entreprise <-> entreprise pour reservations transférées
    transfer = (
        BookingTransfer.query.filter_by(booking_id=effective_booking_id)
        .filter(
            BookingTransfer.status.in_(
                [
                    TransferStatus.PENDING,
                    TransferStatus.ACCEPTED,
                    TransferStatus.COMPLETED,
                ]
            )
        )
        .order_by(BookingTransfer.id.desc())
        .first()
    )

    # Resolve user display name from JWT
    user_id = claims.get("user_id")
    user = User.query.get(user_id) if user_id else None
    if user is None:
        pub = get_jwt_identity()
        if pub:
            user = User.query.filter_by(public_id=str(pub)).first()
    if user:
        first = (getattr(user, "first_name", None) or "").strip()
        last = (getattr(user, "last_name", None) or "").strip()
        user_display = f"{first} {last}".strip()
    else:
        user_display = ""

    # Company user
    user_company_id = claims.get("company_id")
    if user_company_id and int(user_company_id) == booking.company_id:
        label = user_display or (booking.company.name if booking.company else "Entreprise")
        peer_company_id = transfer.executing_company_id if transfer else None
        return (
            booking,
            BookingMessageSender.COMPANY,
            label,
            institution_id,
            request_id,
            peer_company_id,
        )

    # Executing company (partenaire) sur booking transféré
    if transfer and user_company_id and int(user_company_id) == transfer.executing_company_id:
        label = user_display or "Partenaire"
        return (
            booking,
            BookingMessageSender.COMPANY,
            label,
            institution_id,
            request_id,
            transfer.owner_company_id,
        )

    # Institution user
    user_institution_id = claims.get("institution_id")
    if user_institution_id and institution_id and int(user_institution_id) == institution_id:
        inst = source_req.institution if source_req else None
        label = user_display or (inst.name if inst else "Institution")
        return (
            booking,
            BookingMessageSender.INSTITUTION,
            label,
            institution_id,
            request_id,
            None,
        )

    # Client portail (course avec entreprise retenue)
    if user and getattr(user, "role", None) == UserRole.client:
        client_profile = Client.query.filter_by(user_id=user.id).first()
        if (
            client_profile
            and booking.client_id
            and int(client_profile.id) == int(booking.client_id)
            and booking.company_id
        ):
            label = user_display or "Client"
            return (
                booking,
                BookingMessageSender.CLIENT,
                label,
                institution_id,
                request_id,
                None,
            )

    if not source_req and not transfer:
        abort(404, description="Pas de canal de communication pour cette reservation")

    abort(403, description="Acces non autorise a cette conversation")


# ---------------------------------------------------------------------------
# GET messages
# ---------------------------------------------------------------------------
@booking_messages_ns.route("/<int:booking_id>/messages")
class BookingMessageList(Resource):
    """Liste et envoi de messages pour un booking."""

    @booking_messages_ns.doc(
        description="Historique pagine des messages d'un booking",
        security="BearerAuth",
        params={
            "limit": "Nombre max de messages (defaut 30, max 100)",
            "before_id": "ID du message pour pagination (retourne les messages plus anciens)",
        },
    )
    def get(self, booking_id):
        """Retourne les messages du booking (ASC)."""
        try:
            booking, _sender_type, _label, _inst_id, _req_id, _peer_company_id = _get_booking_with_auth(booking_id)

            limit = min(int(flask_request.args.get("limit", 30)), 100)
            before_id = flask_request.args.get("before_id", type=int)

            query = BookingMessage.query.filter_by(booking_id=booking.id)
            if before_id:
                query = query.filter(BookingMessage.id < before_id)

            results = query.order_by(BookingMessage.id.desc()).limit(limit + 1).all()

            has_more = len(results) > limit
            messages = list(reversed(results[:limit]))

            return {
                "messages": [m.serialize for m in messages],
                "has_more": has_more,
            }, 200

        except Exception as exc:
            if hasattr(exc, "code"):
                raise
            logger.exception("[BookingMessages] GET error booking=%s", booking_id)
            return {"error": "Erreur serveur"}, 500

    # ---------------------------------------------------------------------------
    # POST message
    # ---------------------------------------------------------------------------
    @booking_messages_ns.doc(
        description="Envoyer un message dans le mini-chat du booking",
        security="BearerAuth",
    )
    def post(self, booking_id):
        """Cree un nouveau message."""
        try:
            booking, sender_type, sender_label, institution_id, request_id, peer_company_id = _get_booking_with_auth(
                booking_id
            )

            # Bloquer l'envoi si le booking est terminé ou annulé
            closed_statuses = {"COMPLETED", "RETURN_COMPLETED", "CANCELED", "CANCELLED"}
            if booking.status and booking.status.upper() in closed_statuses:
                return {"error": "Conversation cloturee, le transport est termine"}, 403

            body = flask_request.get_json(silent=True) or {}
            content = (body.get("content") or "").strip()

            if not content:
                return {"error": "Contenu requis"}, 400
            if len(content) > MAX_MESSAGE_LENGTH:
                return {"error": f"{MAX_MESSAGE_LENGTH} caracteres maximum"}, 400

            verify_jwt_in_request()
            claims = get_jwt()
            sender_user_id = claims.get("user_id")

            msg = BookingMessage()
            msg.booking_id = booking.id
            msg.sender_user_id = sender_user_id
            msg.sender_type = sender_type
            msg.sender_label = sender_label
            msg.content = content
            db.session.add(msg)
            db.session.commit()

            # Emit socket event
            from services.events.institution_events import emit_booking_message

            emit_booking_message(
                company_id=booking.company_id,
                institution_id=institution_id,
                booking_id=booking.id,
                message_data=msg.serialize,
                sender_type=sender_type.value if sender_type else None,
                request_id=request_id,
            )
            if peer_company_id:
                # Canal partenaire: diffuser aussi au partenaire transféré
                from ext import socketio
                from services.events.institution_events import persist_company_notification

                payload = {
                    "booking_id": booking.id,
                    "message": msg.serialize,
                }
                socketio.emit("booking_message", payload, to=f"company_{peer_company_id}")

                verify_jwt_in_request()
                claims = get_jwt()
                sender_company_id = claims.get("company_id")
                if sender_company_id and int(sender_company_id) != int(peer_company_id):
                    content_preview = (msg.content or "")[:80]
                    persist_company_notification(
                        company_id=int(peer_company_id),
                        event_type="booking_message",
                        title="Nouveau message partenaire",
                        message=f"{sender_label}: {content_preview}",
                        metadata={
                            "booking_id": booking.id,
                            "sender_label": sender_label,
                        },
                    )

            return {"message": msg.serialize}, 201

        except Exception as exc:
            if hasattr(exc, "code"):
                raise
            logger.exception("[BookingMessages] POST error booking=%s", booking_id)
            db.session.rollback()
            return {"error": "Erreur serveur"}, 500
