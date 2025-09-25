# backend/sockets/chat.py
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from flask import session, request
from flask_socketio import SocketIO, emit, join_room
from flask_jwt_extended import decode_token

from models import User, UserRole, Driver, Message, Company
from ext import db

logger = logging.getLogger("socketio")

# Petit index en mémoire pour le debug/nettoyage : sid -> infos
_SID_INDEX: Dict[str, Dict[str, Any]] = {}


def _extract_token(auth) -> Optional[str]:
    """Récupère le token JWT depuis Authorization, auth.token ou ?token="""
    # 1) Header Authorization: Bearer ...
    authz = request.headers.get("Authorization") or request.headers.get("AUTHORIZATION")
    if authz and authz.lower().startswith("bearer "):
        return authz.split(" ", 1)[1].strip()
    # 2) Payload auth envoyé par le client Socket.IO
    if isinstance(auth, dict):
        tok = auth.get("token") or auth.get("accessToken")
        if tok:
            return str(tok).strip()
    # 3) Paramètre de query string (secours)
    qs_tok = request.args.get("token")
    if qs_tok:
        return qs_tok.strip()
    return None


def init_chat_socket(socketio: SocketIO):

    @socketio.on("connect")
    def handle_connect(auth):
        client_ip = request.environ.get("REMOTE_ADDR")
        ua = request.headers.get("User-Agent", "Unknown")
        logger.info(f"🔌 SIO connect from {client_ip} UA={ua}")

        try:
            token = _extract_token(auth)
            if not token:
                logger.info("⛔ Refus: token JWT manquant")
                emit("unauthorized", {"error": "Token JWT manquant"})
                return False

            # Vérifie & décode (lève si invalide/expiré)
            decoded = decode_token(token)
            public_id = decoded.get("sub")
            if not public_id:
                emit("unauthorized", {"error": "Token sans 'sub'"})
                return False
            logger.info(f"🧾 Token validé pour user {public_id}")

            user = User.query.filter_by(public_id=public_id).first()
            if not user:
                logger.info(f"⛔ user not found: {public_id}")
                emit("unauthorized", {"error": "Utilisateur non trouvé"})
                return False

            # Stash session minimale
            session["user_id"] = user.id
            session["first_name"] = user.first_name
            session["role"] = user.role.value

            if user.role == UserRole.driver:
                driver = Driver.query.filter_by(user_id=user.id).first()
                if not driver or not driver.company_id:
                    raise Exception("Chauffeur ou entreprise associée introuvable")

                company_room = f"company_{driver.company_id}"
                driver_room = f"driver_{driver.id}"
                join_room(company_room)
                join_room(driver_room)

                emit("connected", {"message": "✅ Chauffeur connecté"})
                logger.info(f"🔌 Driver {driver.id} -> rooms: {company_room}, {driver_room}")

                _SID_INDEX[request.sid] = {
                    "user_public_id": public_id,
                    "driver_id": driver.id,
                    "company_id": driver.company_id,
                    "ip": client_ip,
                    "role": "driver",
                }

            elif user.role == UserRole.company:
                company = Company.query.filter_by(user_id=user.id).first()
                if not company:
                    emit("unauthorized", {"error": "Entreprise introuvable"})
                    return False

                room = f"company_{company.id}"
                join_room(room)
                emit("connected", {"message": f"✅ Entreprise connectée à {room}"})
                logger.info(f"🏢 Company {company.id} -> room: {room}")

                _SID_INDEX[request.sid] = {
                    "user_public_id": public_id,
                    "company_id": company.id,
                    "ip": client_ip,
                    "role": "company",
                }
            else:
                emit("unauthorized", {"error": "Rôle non autorisé pour le chat"})
                return False

            return True

        except Exception as e:
            logger.exception(f"❌ Erreur de connexion WebSocket : {e}")
            try:
                emit("unauthorized", {"error": str(e)})
            finally:
                return False

    @socketio.on("team_chat_message")
    def handle_team_chat(data):
        local_id = data.get("_localId")
        try:
            user_id = session.get("user_id")
            if not user_id:
                emit("error", {"error": "Session utilisateur introuvable."})
                return

            user = User.query.get(user_id)
            if not user:
                emit("error", {"error": "Utilisateur non reconnu."})
                return

            content = (data.get("content") or "").strip()
            receiver_id = data.get("receiver_id")
            timestamp = datetime.now(timezone.utc)
            if not content:
                emit("error", {"error": "Message vide non autorisé."})
                return

            if user.role == UserRole.driver:
                driver = Driver.query.filter_by(user_id=user.id).first()
                if not driver:
                    emit("error", {"error": "Chauffeur introuvable."})
                    return
                company_id = driver.company_id
                sender_role = "driver"
                sender_id = user.id
                company_obj = None
            elif user.role == UserRole.company:
                company_obj = Company.query.filter_by(user_id=user.id).first()
                if not company_obj:
                    emit("error", {"error": "Entreprise introuvable."})
                    return
                company_id = company_obj.id
                sender_role = "company"
                sender_id = None
            else:
                emit("error", {"error": "Rôle non autorisé pour le chat."})
                return

            message = Message(
                sender_id=sender_id,
                receiver_id=receiver_id,
                company_id=company_id,
                sender_role=sender_role,
                content=content,
                timestamp=timestamp,
            )
            db.session.add(message)
            db.session.commit()

            payload = {
                "id": message.id,
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "receiver_name": message.receiver.first_name if message.receiver else None,
                "sender_role": sender_role,
                "sender_name": session.get("first_name") or user.first_name,
                "content": content,
                "timestamp": timestamp.isoformat(),
                "type": "chat",
                "company_id": company_id,
                "company_name": (company_obj.name if sender_role == "company" and company_obj else None),
                "_localId": local_id,
            }

            room = f"company_{company_id}"
            emit("team_chat_message", payload, room=room)
            logger.info(f"📨 Message émis dans {room} par {sender_role} : {content}")

        except Exception as e:
            logger.exception(f"❌ Erreur team_chat_message : {e}")
            emit("error", {"error": "Erreur d’envoi de message."})

    @socketio.on("join_driver_room")
    def handle_join_driver_room():
        try:
            user_id = session.get("user_id")
            if not user_id:
                emit("error", {"error": "Session non trouvée"})
                return

            user = User.query.get(user_id)
            if not user or user.role != UserRole.driver:
                emit("error", {"error": "Seuls les chauffeurs peuvent rejoindre cette room."})
                return

            driver = Driver.query.filter_by(user_id=user.id).first()
            if not driver:
                emit("error", {"error": "Chauffeur introuvable"})
                return

            driver_room = f"driver_{driver.id}"
            join_room(driver_room)
            company_room = f"company_{driver.company_id}"
            join_room(company_room)
            logger.info(f"✅ Driver {driver.id} joined rooms [{driver_room}, {company_room}]")
            emit("joined_room", {"rooms": [driver_room, company_room]})

        except Exception as e:
            logger.exception(f"❌ Erreur join_driver_room : {e}")
            emit("error", {"error": str(e)})

    @socketio.on("driver_location")
    def handle_driver_location(data):
        try:
            user_id = session.get("user_id")
            user_role = session.get("role")
            if not user_id or user_role != "driver":
                emit("error", {"error": "Accès non autorisé pour l’envoi de localisation."})
                return

            driver = Driver.query.filter_by(user_id=user_id).first()
            if not driver or not driver.company_id:
                emit("error", {"error": "Chauffeur introuvable ou non lié à une entreprise."})
                return

            latitude = data.get("latitude")
            longitude = data.get("longitude")
            if latitude is None or longitude is None:
                emit("error", {"error": "Latitude et longitude requises."})
                return

            company_room = f"company_{driver.company_id}"
            emit(
                "driver_location_update",
                {
                    "driver_id": driver.id,
                    "first_name": driver.user.first_name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                room=company_room,
            )
            logger.info(f"📡 Loc -> {company_room} (driver {driver.id}) {latitude},{longitude}")

        except Exception as e:
            logger.exception(f"❌ Erreur driver_location : {e}")
            emit("error", {"error": str(e)})

    @socketio.on("disconnect")
    def handle_disconnect():
        info = _SID_INDEX.pop(request.sid, None)
        logger.info(f"👋 SIO disconnect sid={request.sid} info={info}")