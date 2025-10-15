from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource
from sqlalchemy.orm import joinedload

from ext import app_logger  # si tu utilises un logger structuré
from models import Company, Message, User, UserRole

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

        # 🔧 Sérialisation (s'aligne sur Message.serialize pour cohérence API)
        results: list[dict[str, Any]] = []
        for m in messages:
            try:
                base = m.serialize if hasattr(m, "serialize") else {}
            except Exception:
                base = {}
            if not base:
                # Fallback minimal si serialize indisponible
                base = {
                    "id": m.id,
                    "company_id": m.company_id,
                    "sender_id": getattr(m, "sender_id", None),
                    "receiver_id": getattr(m, "receiver_id", None),
                    "sender_role": getattr(m, "sender_role", None),
                    "content": getattr(m, "content", None),
                    "timestamp": m.timestamp.isoformat() if getattr(m, "timestamp", None) else None,
                }
                # enrichir noms
                base["sender_name"] = (
                    company_name if getattr(m, "sender_role", None) in ("COMPANY", "company") else (
                        getattr(getattr(m, "sender", None), "first_name", None)
                    )
                )
                base["receiver_name"] = getattr(getattr(m, "receiver", None), "first_name", None)

            results.append(base)

        app_logger.info(f"📨 {len(results)} messages (limit={limit}, before={before}) pour company_id={company_id}")
        return results, 200
