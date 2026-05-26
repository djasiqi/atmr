"""REST API for multi-context conversations (V1 driver)."""

from __future__ import annotations

import logging
from datetime import datetime

from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource

from models import Conversation
from repositories.user_repository import UserRepository
from services.drivers.request_driver import resolve_request_driver
from services.messages.hub_service import report_hub_emergency
from services.messaging.conversation_service import ConversationService
from services.messaging.permission_service import MessagingPermissionService

logger = logging.getLogger(__name__)

conversations_ns = Namespace("conversations", description="Messagerie conversations")
user_repo = UserRepository()


def _normalized_role(user) -> str:
    return str(getattr(user.role, "value", user.role)).upper()


def _resolve_conversation_actor(user):
    """Accès messagerie : compte entreprise ou chauffeur (contexte actif)."""
    if not user:
        return None, ({"error": "Utilisateur introuvable"}, 404)
    role = _normalized_role(user)
    if role == "COMPANY":
        company = getattr(user, "company", None)
        if company is None:
            return None, ({"error": "Profil entreprise introuvable"}, 404)
        return {
            "kind": "company",
            "user": user,
            "company_id": int(company.id),
        }, None
    driver, err = resolve_request_driver(user)
    if driver:
        return {
            "kind": "driver",
            "user": user,
            "driver": driver,
            "company_id": int(driver.company_id),
        }, None
    if err:
        return None, err
    return None, ({"error": "Rôle non autorisé"}, 403)


@conversations_ns.route("/inbox")
class ConversationsInbox(Resource):
    @jwt_required()
    def get(self):
        user = user_repo.find_by_public_id_with_driver_and_company(get_jwt_identity())
        driver, err = resolve_request_driver(user)
        if err:
            return err
        inbox = ConversationService.build_driver_inbox(driver)
        return {
            **inbox,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }, 200


@conversations_ns.route("/resolve")
class ConversationResolveLegacy(Resource):
    """Resolve legacy thread_id to conversation_id."""

    @jwt_required()
    def get(self):
        user = user_repo.find_by_public_id_with_driver_and_company(get_jwt_identity())
        actor, err = _resolve_conversation_actor(user)
        if err:
            return err
        company_id = int(actor["company_id"])
        thread_id = request.args.get("thread_id", "").strip()
        if not thread_id:
            return {"error": "thread_id requis"}, 400
        if actor["kind"] == "company":
            conv = ConversationService.resolve_by_legacy_thread_for_company(
                company_id, thread_id
            )
        else:
            conv = ConversationService.resolve_by_legacy_thread(
                company_id, thread_id, actor["driver"]
            )
        if not conv:
            return {"error": "Conversation introuvable"}, 404
        return {
            "conversation_id": conv.id,
            "thread_id": thread_id,
            "can_manage": (
                actor["kind"] == "company"
                and ConversationService.is_company_managed_dispatch(conv)
            ),
        }, 200


@conversations_ns.route("/emergency")
class ConversationEmergency(Resource):
    """Signalement urgence chauffeur — company_id dérivé du profil (pas d'ID dans l'URL)."""

    @jwt_required()
    def post(self):
        user = user_repo.find_by_public_id_with_driver_and_company(get_jwt_identity())
        driver, err = resolve_request_driver(user)
        if err:
            return err

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
            driver,
            issue_type=issue_type,
            booking_id=booking_id_int,
            latitude=lat_f,
            longitude=lon_f,
            note=note,
        )
        return result, 200


@conversations_ns.route("/<int:conversation_id>")
class ConversationDetail(Resource):
    @jwt_required()
    def get(self, conversation_id: int):
        user = user_repo.find_by_public_id_with_driver_and_company(get_jwt_identity())
        if not user:
            return {"error": "Utilisateur introuvable"}, 404
        conv = Conversation.query.get_or_404(conversation_id)
        try:
            MessagingPermissionService.assert_can_read(user, conv)
        except PermissionError as e:
            return {"error": str(e)}, 403
        return {"conversation": conv.serialize}, 200


@conversations_ns.route("/<int:conversation_id>/messages")
class ConversationMessages(Resource):
    @jwt_required()
    def get(self, conversation_id: int):
        user = user_repo.find_by_public_id_with_driver_and_company(get_jwt_identity())
        if not user:
            return {"error": "Utilisateur introuvable"}, 404
        conv = Conversation.query.get_or_404(conversation_id)
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
        try:
            messages = ConversationService.get_messages(
                conv, user, before=dt_before, limit=limit
            )
        except PermissionError as e:
            return {"error": str(e)}, 403
        return {
            "conversation_id": conversation_id,
            "messages": [m.serialize for m in messages],
        }, 200


@conversations_ns.route("/<int:conversation_id>/read")
class ConversationMarkRead(Resource):
    @jwt_required()
    def post(self, conversation_id: int):
        user = user_repo.find_by_public_id_with_driver_and_company(get_jwt_identity())
        if not user:
            return {"error": "Utilisateur introuvable"}, 404
        conv = Conversation.query.get_or_404(conversation_id)
        try:
            updated = ConversationService.mark_read(conv, user)
        except PermissionError as e:
            return {"error": str(e)}, 403
        return {"conversation_id": conversation_id, "marked_read": updated}, 200


@conversations_ns.route("/<int:conversation_id>/participants")
class ConversationParticipants(Resource):
    @jwt_required()
    def get(self, conversation_id: int):
        user = user_repo.find_by_public_id_with_driver_and_company(get_jwt_identity())
        actor, err = _resolve_conversation_actor(user)
        if err:
            return err
        conv = Conversation.query.get_or_404(conversation_id)
        if int(conv.company_id) != int(actor["company_id"]):
            return {"error": "Accès refusé"}, 403
        try:
            payload = ConversationService.list_dispatch_participants(
                conv, actor["user"]
            )
        except PermissionError as e:
            return {"error": str(e)}, 403
        return payload, 200

    @jwt_required()
    def post(self, conversation_id: int):
        user = user_repo.find_by_public_id_with_driver_and_company(get_jwt_identity())
        actor, err = _resolve_conversation_actor(user)
        if err:
            return err
        if actor["kind"] != "company":
            return {"error": "Réservé à l'exploitation"}, 403
        conv = Conversation.query.get_or_404(conversation_id)
        if int(conv.company_id) != int(actor["company_id"]):
            return {"error": "Accès refusé"}, 403
        body = request.get_json(silent=True) or {}
        try:
            driver_id = int(body.get("driver_id"))
        except (TypeError, ValueError):
            return {"error": "driver_id requis"}, 400
        try:
            participant = ConversationService.add_dispatch_participant(
                conv, actor["user"], driver_id=driver_id
            )
        except PermissionError as e:
            return {"error": str(e)}, 403
        return {"participant": participant}, 201


@conversations_ns.route("/<int:conversation_id>/participants/<int:user_id>")
class ConversationParticipantRemove(Resource):
    @jwt_required()
    def delete(self, conversation_id: int, user_id: int):
        user = user_repo.find_by_public_id_with_driver_and_company(get_jwt_identity())
        actor, err = _resolve_conversation_actor(user)
        if err:
            return err
        if actor["kind"] != "company":
            return {"error": "Réservé à l'exploitation"}, 403
        conv = Conversation.query.get_or_404(conversation_id)
        if int(conv.company_id) != int(actor["company_id"]):
            return {"error": "Accès refusé"}, 403
        try:
            result = ConversationService.remove_dispatch_participant(
                conv, actor["user"], target_user_id=user_id
            )
        except PermissionError as e:
            return {"error": str(e)}, 403
        return result, 200


@conversations_ns.route("/<int:conversation_id>/manage")
class ConversationManage(Resource):
    """Vue gestion canal dispatch (entreprise)."""

    @jwt_required()
    def get(self, conversation_id: int):
        user = user_repo.find_by_public_id_with_driver_and_company(get_jwt_identity())
        actor, err = _resolve_conversation_actor(user)
        if err:
            return err
        conv = Conversation.query.get_or_404(conversation_id)
        if int(conv.company_id) != int(actor["company_id"]):
            return {"error": "Accès refusé"}, 403
        try:
            payload = ConversationService.get_dispatch_channel_manage(
                conv, actor["user"]
            )
        except PermissionError as e:
            return {"error": str(e)}, 403
        return payload, 200

    @jwt_required()
    def patch(self, conversation_id: int):
        user = user_repo.find_by_public_id_with_driver_and_company(get_jwt_identity())
        actor, err = _resolve_conversation_actor(user)
        if err:
            return err
        if actor["kind"] != "company":
            return {"error": "Réservé à l'exploitation"}, 403
        conv = Conversation.query.get_or_404(conversation_id)
        if int(conv.company_id) != int(actor["company_id"]):
            return {"error": "Accès refusé"}, 403
        body = request.get_json(silent=True) or {}
        title = body.get("title") if isinstance(body.get("title"), str) else None
        description = (
            body.get("description")
            if isinstance(body.get("description"), str)
            else None
        )
        try:
            payload = ConversationService.update_dispatch_channel(
                conv,
                actor["user"],
                title=title,
                description=description,
            )
        except PermissionError as e:
            return {"error": str(e)}, 403
        return payload, 200


@conversations_ns.route("/<int:conversation_id>/manage/clear-history")
class ConversationManageClearHistory(Resource):
    """Supprime tous les messages du canal dispatch (entreprise)."""

    @jwt_required()
    def post(self, conversation_id: int):
        user = user_repo.find_by_public_id_with_driver_and_company(get_jwt_identity())
        actor, err = _resolve_conversation_actor(user)
        if err:
            return err
        if actor["kind"] != "company":
            return {"error": "Réservé à l'exploitation"}, 403
        conv = Conversation.query.get_or_404(conversation_id)
        if int(conv.company_id) != int(actor["company_id"]):
            return {"error": "Accès refusé"}, 403
        try:
            payload = ConversationService.clear_dispatch_channel_history(
                conv, actor["user"]
            )
        except PermissionError as e:
            return {"error": str(e)}, 403
        return payload, 200


@conversations_ns.route("/<int:conversation_id>/attachments")
class ConversationAttachments(Resource):
    @jwt_required()
    def get(self, conversation_id: int):
        user = user_repo.find_by_public_id_with_driver_and_company(get_jwt_identity())
        actor, err = _resolve_conversation_actor(user)
        if err:
            return err
        conv = Conversation.query.get_or_404(conversation_id)
        if int(conv.company_id) != int(actor["company_id"]):
            return {"error": "Accès refusé"}, 403
        try:
            limit = max(1, min(200, int(request.args.get("limit", 80))))
        except ValueError:
            return {"error": "Paramètre limit invalide"}, 400
        try:
            attachments = ConversationService.list_conversation_attachments(
                conv, actor["user"], limit=limit
            )
        except PermissionError as e:
            return {"error": str(e)}, 403
        return {"conversation_id": conversation_id, "attachments": attachments}, 200
