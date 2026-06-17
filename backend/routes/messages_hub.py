"""REST endpoints for message hub (driver + company mobile)."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource

from models import SenderRole
from repositories.message_repository import MessageRepository
from repositories.user_repository import UserRepository
from services.drivers.request_driver import resolve_request_driver
from services.messages.hub_service import (
    THREAD_DISPATCH,
    ack_message,
    build_driver_threads,
    count_company_team_members,
    count_unread,
    get_thread_messages,
    list_driver_colleagues,
    mark_thread_read,
    report_hub_emergency,
    send_company_hub_message,
    send_driver_hub_message,
)

logger = logging.getLogger(__name__)

messages_hub_ns = Namespace("messages_hub", description="Hub messages mobile")
user_repo = UserRepository()
message_repo = MessageRepository()


def _user_role(user) -> str:
    return str(getattr(user.role, "value", user.role)).upper()


def _resolve_hub_access(
    user, company_id: int
) -> tuple[dict[str, Any] | None, tuple[dict, int] | None]:
    role = _user_role(user)
    active_ctx = (request.headers.get("X-Active-Context-Id") or "").strip()
    in_driver_context = active_ctx.startswith("driver:")

    if role == "COMPANY" and not in_driver_context:
        company = getattr(user, "company", None)
        if company is None:
            return None, ({"error": "Profil entreprise introuvable"}, 404)
        if int(company.id) != int(company_id):
            return None, ({"error": "Accès refusé"}, 403)
        return {"kind": "company", "user": user, "company_id": int(company.id)}, None

    driver, driver_err = resolve_request_driver(user)
    if driver:
        resolved = getattr(driver, "company_id", None)
        if not resolved:
            return None, ({"error": "Entreprise chauffeur introuvable"}, 404)
        if int(resolved) != int(company_id):
            return None, ({"error": "Accès refusé"}, 403)
        return {
            "kind": "driver",
            "user": user,
            "driver": driver,
            "company_id": int(resolved),
        }, None

    if driver_err:
        return None, driver_err

    return None, ({"error": "Rôle non autorisé"}, 403)


def _hub_user():
    return user_repo.find_by_public_id_with_driver_and_company(get_jwt_identity())


@messages_hub_ns.route("/<int:company_id>/hub/threads")
class MessageHubThreads(Resource):
    @jwt_required()
    def get(self, company_id: int):
        from services.monitoring.route_timing import route_duration_span

        user = _hub_user()
        if not user:
            return {"error": "Utilisateur introuvable"}, 404
        actor, err = _resolve_hub_access(user, company_id)
        if err:
            return err

        try:
            from services.messaging.conversation_service import ConversationService

            with route_duration_span(
                "messages_hub.threads",
                company_id=company_id,
                actor_kind=actor["kind"],
            ):
                return self._get_threads_payload(actor)
        except Exception:
            logger.exception("ConversationService inbox fallback to hub_service")
            return self._hub_threads_fallback(actor, company_id)

    def _get_threads_payload(self, actor: dict[str, Any]) -> tuple[dict, int]:
        from services.messaging.conversation_service import ConversationService

        if actor["kind"] == "company":
            user = actor["user"]
            inbox = ConversationService.build_company_inbox(user)
            threads = ConversationService.hub_threads_for_company(user, inbox=inbox)
            unread = int(inbox.get("unread_total") or 0)
        else:
            driver = actor["driver"]
            inbox = ConversationService.build_driver_inbox(driver)
            threads = ConversationService.hub_threads_for_driver(driver, inbox=inbox)
            unread = int(inbox.get("unread_total") or 0)

        return {
            "threads": threads,
            "unread_total": unread,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }, 200

    def _hub_threads_fallback(
        self, actor: dict[str, Any], company_id: int
    ) -> tuple[dict, int]:
        if actor["kind"] == "company":
            threads = [
                {
                    "thread_id": THREAD_DISPATCH,
                    "section": "dispatch",
                    "title": "Dispatch",
                    "subtitle": "Exploitation & régulation",
                    "unread_count": 0,
                    "priority": "normal",
                }
            ]
            unread = count_unread(company_id)
        else:
            threads = build_driver_threads(company_id, actor["driver"])
            unread = count_unread(company_id)

        return {
            "threads": threads,
            "unread_total": unread,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }, 200


@messages_hub_ns.route("/<int:company_id>/hub/colleagues")
class MessageHubColleagues(Resource):
    @jwt_required()
    def get(self, company_id: int):
        user = _hub_user()
        if not user:
            return {"error": "Utilisateur introuvable"}, 404
        actor, err = _resolve_hub_access(user, company_id)
        if err:
            return err
        if actor["kind"] != "driver":
            return {"error": "Réservé aux chauffeurs"}, 403
        colleagues = list_driver_colleagues(company_id, actor["driver"])
        return {
            "colleagues": colleagues,
            "team_member_count": count_company_team_members(company_id),
        }, 200


@messages_hub_ns.route("/<int:company_id>/hub/threads/<string:thread_id>/messages")
class MessageHubThreadMessages(Resource):
    @jwt_required()
    def get(self, company_id: int, thread_id: str):
        user = _hub_user()
        if not user:
            return {"error": "Utilisateur introuvable"}, 404
        actor, err = _resolve_hub_access(user, company_id)
        if err:
            return err

        try:
            limit = max(1, min(100, int(request.args.get("limit", 40))))
        except ValueError:
            return {"error": "Paramètre limit invalide"}, 400

        before_raw = request.args.get("before")
        dt_before = None
        if before_raw:
            try:
                dt_before = datetime.fromisoformat(before_raw.rstrip("Z"))
            except ValueError:
                return {"error": "Timestamp invalide"}, 400

        if actor["kind"] == "company":
            messages = get_thread_messages(
                company_id,
                thread_id,
                before=dt_before,
                limit=limit,
                reader_user=user,
            )
        else:
            reader_user_id = getattr(actor["driver"], "user_id", None)
            messages = get_thread_messages(
                company_id,
                thread_id,
                before=dt_before,
                limit=limit,
                reader_user_id=int(reader_user_id) if reader_user_id else None,
                driver=actor["driver"],
                reader_user=user,
            )
        return {
            "messages": [m.serialize for m in messages],
            "thread_id": thread_id,
        }, 200

    @jwt_required()
    def post(self, company_id: int, thread_id: str):
        user = _hub_user()
        if not user:
            return {"error": "Utilisateur introuvable"}, 404
        actor, err = _resolve_hub_access(user, company_id)
        if err:
            return err

        body = request.get_json(silent=True) or {}
        if actor["kind"] == "company":
            payload, error = send_company_hub_message(
                user,
                company_id,
                thread_id=str(thread_id),
                body=body,
            )
        else:
            payload, error = send_driver_hub_message(
                user,
                actor["driver"],
                company_id,
                thread_id=str(thread_id),
                body=body,
            )
        if error:
            return error[0], error[1]
        return {"message": payload, "thread_id": thread_id}, 201


@messages_hub_ns.route("/<int:company_id>/hub/read")
class MessageHubMarkRead(Resource):
    @jwt_required()
    def post(self, company_id: int):
        user = _hub_user()
        if not user:
            return {"error": "Utilisateur introuvable"}, 404
        actor, err = _resolve_hub_access(user, company_id)
        if err:
            return err

        body = request.get_json(silent=True) or {}
        thread_id = body.get("thread_id") or THREAD_DISPATCH
        try:
            from services.messaging.conversation_service import ConversationService

            if actor["kind"] == "company":
                conv = ConversationService.resolve_by_legacy_thread_for_company(
                    company_id, str(thread_id)
                )
                if conv:
                    updated = ConversationService.mark_read(conv, user)
                    return {"thread_id": thread_id, "marked_read": updated}, 200
            else:
                conv = ConversationService.resolve_by_legacy_thread(
                    company_id, str(thread_id), actor["driver"]
                )
                if conv:
                    updated = ConversationService.mark_read(conv, user)
                    return {"thread_id": thread_id, "marked_read": updated}, 200
        except PermissionError:
            logger.info(
                "mark_read refusé (permissions), fallback legacy thread_id=%s",
                thread_id,
            )
        except Exception:
            logger.exception("mark_read conversation fallback")

        reader_user_id = (
            int(user.id)
            if actor["kind"] == "company"
            else getattr(actor["driver"], "user_id", None)
        )
        reader_role = (
            SenderRole.COMPANY if actor["kind"] == "company" else SenderRole.DRIVER
        )
        updated = mark_thread_read(
            company_id,
            str(thread_id),
            reader_role,
            reader_user_id=int(reader_user_id) if reader_user_id else None,
        )
        return {"thread_id": thread_id, "marked_read": updated}, 200


@messages_hub_ns.route("/<int:company_id>/hub/ack/<int:message_id>")
class MessageHubAck(Resource):
    @jwt_required()
    def post(self, company_id: int, message_id: int):
        user = _hub_user()
        if not user:
            return {"error": "Utilisateur introuvable"}, 404
        actor, err = _resolve_hub_access(user, company_id)
        if err:
            return err
        ok = ack_message(message_id, company_id)
        if not ok:
            return {"error": "Message introuvable"}, 404
        return {"message_id": message_id, "acked": True}, 200


@messages_hub_ns.route("/<int:company_id>/hub/unread-count")
class MessageHubUnreadCount(Resource):
    @jwt_required()
    def get(self, company_id: int):
        user = _hub_user()
        if not user:
            return {"error": "Utilisateur introuvable"}, 404
        actor, err = _resolve_hub_access(user, company_id)
        if err:
            return err
        try:
            from services.messaging.conversation_service import ConversationService

            if actor["kind"] == "company":
                inbox = ConversationService.build_company_inbox(user)
                unread = int(inbox.get("unread_total") or 0)
            else:
                inbox = ConversationService.build_driver_inbox(actor["driver"])
                unread = int(inbox.get("unread_total") or 0)
        except Exception:
            unread = count_unread(company_id)
        return {"unread_count": unread}, 200


@messages_hub_ns.route("/<int:company_id>/hub/emergency")
class MessageHubEmergency(Resource):
    @jwt_required()
    def post(self, company_id: int):
        user = _hub_user()
        if not user:
            return {"error": "Utilisateur introuvable"}, 404
        actor, err = _resolve_hub_access(user, company_id)
        if err:
            return err
        if actor["kind"] != "driver":
            return {"error": "Réservé aux chauffeurs"}, 403

        body = request.get_json(silent=True) or {}
        issue_type = str(body.get("issue_type") or "").strip()
        if not issue_type:
            return {"error": "issue_type requis"}, 400

        booking_id = body.get("booking_id")
        try:
            booking_id_int = int(booking_id) if booking_id is not None else None
        except (TypeError, ValueError):
            booking_id_int = None

        lat = body.get("latitude")
        lon = body.get("longitude")
        try:
            lat_f = float(lat) if lat is not None else None
            lon_f = float(lon) if lon is not None else None
        except (TypeError, ValueError):
            lat_f, lon_f = None, None

        note = body.get("note") if isinstance(body.get("note"), str) else None
        result = report_hub_emergency(
            actor["driver"],
            issue_type=issue_type,
            booking_id=booking_id_int,
            latitude=lat_f,
            longitude=lon_f,
            note=note,
        )
        return result, 200
