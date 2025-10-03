from __future__ import annotations

from flask import request
from flask_restx import Namespace, Resource
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import Message, User, UserRole, Company
from sqlalchemy.orm import joinedload
from datetime import datetime
from typing import Any, cast

from ext import app_logger  # si tu utilises un logger structuré

messages_ns = Namespace("messages", description="Messagerie entreprise")


@messages_ns.route("/<int:company_id>")
class MessagesList(Resource):
    @jwt_required()
    def get(self, company_id: int):
        user_public_id = get_jwt_identity()

        # 🔍 Chargement de l’utilisateur + relations (avec cast pour Pylance)
        user = (
            User.query
            .options(
                joinedload(cast(Any, User.driver)),
                joinedload(cast(Any, User.company)),
            )
            .filter_by(public_id=user_public_id)
            .first()
        )
        if not user:
            app_logger.error(f"❌ Utilisateur introuvable pour public_id: {user_public_id}")
            return {"error": "Utilisateur introuvable"}, 404

        # 🔐 Contrôle d’accès
        if user.role == UserRole.driver:
            if not getattr(user, "driver", None) or user.driver.company_id != company_id:
                return {"error": "Accès refusé au chat de cette entreprise"}, 403
        elif user.role == UserRole.company:
            if not getattr(user, "company", None) or user.company.id != company_id:
                return {"error": "Accès refusé à cette entreprise"}, 403
        else:
            return {"error": "Rôle non autorisé"}, 403

        # 📦 Lecture des params de pagination
        try:
            limit = max(1, int(request.args.get("limit", 20)))
            before = request.args.get("before", None)
        except ValueError:
            return {"error": "Paramètres invalides"}, 400

        # 🔎 Construction de la requête
        query = Message.query.filter_by(company_id=company_id)
        if before:
            try:
                before_str = before.rstrip("Z")  # support basique ISO8601 avec 'Z'
                dt_before = datetime.fromisoformat(before_str)
                query = query.filter(Message.timestamp < dt_before)
            except ValueError:
                return {"error": "Timestamp invalide"}, 400

        # 🔄 Récupération des messages (avec relations préchargées)
        messages = (
            query
            .options(
                joinedload(cast(Any, Message.sender)),
                joinedload(cast(Any, Message.receiver)),
            )
            .order_by(Message.timestamp.desc())
            .limit(limit)
            .all()
        )

        # ↩️ On remet en ordre ascendant
        messages.reverse()

        # Précharger l’entreprise (évite une requête par message)
        company = Company.query.get(company_id)
        company_name = company.name if company and getattr(company, "name", None) else "Entreprise"

        # 🔧 Sérialisation
        results: list[dict[str, Any]] = []
        for m in messages:
            sender_name = company_name if m.sender_role == "company" else (
                m.sender.first_name if getattr(m, "sender", None) and getattr(m.sender, "first_name", None) else "Inconnu"
            )
            receiver_name = (
                m.receiver.first_name
                if getattr(m, "receiver", None) and getattr(m.receiver, "first_name", None)
                else None
            )
            results.append({
                "id": m.id,
                "company_id": m.company_id,
                "sender_role": m.sender_role,
                "sender_name": sender_name,
                "receiver_name": receiver_name,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if getattr(m, "timestamp", None) else None,
            })

        app_logger.info(f"📨 {len(results)} messages (limit={limit}, before={before}) pour company_id={company_id}")
        return results, 200
