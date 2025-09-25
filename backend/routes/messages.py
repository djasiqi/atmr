from flask import request
from flask_restx import Namespace, Resource
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import Message, User, UserRole, db, Company
from sqlalchemy.orm import joinedload
from datetime import datetime
from ext import app_logger  # si tu utilises un logger structuré

messages_ns = Namespace("messages", description="Messagerie entreprise")


@messages_ns.route("/<int:company_id>")
class MessagesList(Resource):
    @jwt_required()
    def get(self, company_id):
        user_public_id = get_jwt_identity()

        # 🔍 Chargement de l’utilisateur + relations
        user = (
            User.query
            .options(joinedload(User.driver), joinedload(User.company))
            .filter_by(public_id=user_public_id)
            .first()
        )
        if not user:
            app_logger.error(f"❌ Utilisateur introuvable pour public_id: {user_public_id}")
            return {"error": "Utilisateur introuvable"}, 404

        # 🔐 Contrôle d’accès
        if user.role == UserRole.driver:
            if not user.driver or user.driver.company_id != company_id:
                return {"error": "Accès refusé au chat de cette entreprise"}, 403
        elif user.role == UserRole.company:
            if not user.company or user.company.id != company_id:
                return {"error": "Accès refusé à cette entreprise"}, 403
        else:
            return {"error": "Rôle non autorisé"}, 403

        # 📦 Lecture des params de pagination
        try:
            limit = int(request.args.get("limit", 20))
            before = request.args.get("before", None)
        except ValueError:
            return {"error": "Paramètres invalides"}, 400

        # 🔎 Construction de la requête
        query = Message.query.filter_by(company_id=company_id)
        if before:
            try:
                dt_before = datetime.fromisoformat(before)
                query = query.filter(Message.timestamp < dt_before)
            except ValueError:
                return {"error": "Timestamp invalide"}, 400

        # 🔄 On récupère d’abord en DESC pour prendre les plus récents
        messages = (
            query
            .order_by(Message.timestamp.desc())
            .limit(limit)
            .all()
        )
        # ↩️ On remet en ordre ascendant
        messages.reverse()

        # 🔧 Sérialisation uniforme
        results = []
        for m in messages:
            # Nom de l’émetteur
            if m.sender_role == "company":
                sender_name = (Company.query.get(m.company_id) or Company(name="Entreprise")).name
            else:
                sender_name = m.sender.first_name if m.sender else "Inconnu"

            # Nom du destinataire
            receiver_name = m.receiver.first_name if m.receiver else None

            results.append({
                "id"            : m.id,
                "company_id"    : m.company_id,
                "sender_role"   : m.sender_role,
                "sender_name"   : sender_name,
                "receiver_name" : receiver_name,
                "content"       : m.content,
                "timestamp"     : m.timestamp.isoformat(),
            })

        app_logger.info(f"📨 {len(results)} messages (limit={limit}, before={before}) pour company_id={company_id}")
        return results
