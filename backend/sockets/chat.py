# backend/sockets/chat.py
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, cast, Mapping

from flask import session, request
from flask_socketio import SocketIO, emit, join_room
from flask_jwt_extended import decode_token

from models import User, UserRole, Driver, Message, Company
from ext import db, redis_client

logger = logging.getLogger("socketio")

# Petit index en mémoire pour le debug/nettoyage : sid -> infos
_SID_INDEX: Dict[str, Dict[str, Any]] = {}


def _get_sid() -> str:
    sid = getattr(request, "sid", None) or request.environ.get("socketio.sid")
    return str(sid) if sid is not None else ""


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

                _SID_INDEX[_get_sid()] = {
                    "user_public_id": public_id,
                    "user_id": user.id,
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

                _SID_INDEX[_get_sid()] = {
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

            MessageCtor = cast(Any, Message)
            message = MessageCtor(
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
            # Pylance ne déclare pas kwarg 'room' sur emit -> cast en Any
            cast(Any, emit)("team_chat_message", payload, room=room)
            logger.info(f"📨 Message émis dans {room} par {sender_role} : {content}")

        except Exception as e:
            logger.exception(f"❌ Erreur team_chat_message : {e}")
            emit("error", {"error": "Erreur d’envoi de message."})

    @socketio.on("join_driver_room")
    def handle_join_driver_room(data=None):
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
              
            # Fallback: si session vide, on récupère depuis _SID_INDEX
            if not user_id:
                sid_info = _SID_INDEX.get(_get_sid(), {})
                user_id = sid_info.get("user_id")
                user_role = sid_info.get("role")
              
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
            cast(Any, emit)(
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


       
    @socketio.on("join_company")
    def handle_join_company(data=None):
           try:
               user_id = session.get("user_id")
               user_role = session.get("role")
               if not user_id:
                   emit("error", {"error": "Session utilisateur introuvable."})
                   return

               if user_role == "company":
                   company = Company.query.filter_by(user_id=user_id).first()
                   if not company:
                       emit("error", {"error": "Entreprise introuvable."})
                       return

                   room = f"company_{company.id}"
                   join_room(room)
                   emit("joined_company", {"company_id": company.id, "room": room})
                   logger.info(f"🏢 Company {company.id} joined room: {room}")
               elif user_role == "driver":
                   driver = Driver.query.filter_by(user_id=user_id).first()
                   if not driver or not driver.company_id:
                       emit("error", {"error": "Chauffeur ou entreprise associée introuvable."})
                       return

                   room = f"company_{driver.company_id}"
                   join_room(room)
                   emit("joined_company", {"company_id": driver.company_id, "room": room})
                   logger.info(f"🚗 Driver {driver.id} joined company room: {room}")
               else:
                   emit("error", {"error": "Rôle non autorisé pour rejoindre une room entreprise."})
           except Exception as e:
               logger.exception(f"❌ Error in join_company: {e}")
               emit("error", {"error": str(e)})


    @socketio.on("get_driver_locations")
    def handle_get_driver_locations():
           try:
               user_id = session.get("user_id")
               user_role = session.get("role")
               if not user_id or user_role != "company":
                   emit("error", {"error": "Accès non autorisé pour la demande de localisation."})
                   return

               # Get company ID from session
               company_info = _SID_INDEX.get(_get_sid(), {})
               company_id = company_info.get("company_id")

               if not company_id:
                   emit("error", {"error": "Entreprise non identifiée."})
                   return

               # Get all drivers for this company
               drivers = Driver.query.filter_by(company_id=company_id).all()

               # For each driver, get location from Redis or DB
               for driver in drivers:
                   try:
                       # Try Redis first
                       key = f"driver:{driver.id}:loc"
                       h_raw = redis_client.hgetall(key)
                       # Calme Pylance: redis-py retourne un dict[bytes, bytes]
                       h: Mapping[bytes, Any] = cast(Mapping[bytes, Any], h_raw)

                       if h:
                           # Redis returns bytes -> decode
                           def _dec(v):
                               try:
                                   return v.decode()
                               except Exception:
                                   return v

                           loc_data = {
                               (k.decode() if isinstance(k, (bytes, bytearray)) else str(k)): _dec(v)
                               for k, v in h.items()
                           }

                           # Cast numeric fields
                           for kf in ("lat","lon","speed","heading","accuracy"):
                               if kf in loc_data:
                                   try:
                                       loc_data[kf] = float(loc_data[kf])
                                   except Exception:
                                       pass

                           # Emit location to the company room
                           cast(Any, emit)("driver_location_update", {
                               "driver_id": driver.id,
                               "first_name": driver.user.first_name,
                               "latitude": loc_data.get("lat"),
                               "longitude": loc_data.get("lon"),
                               "timestamp": loc_data.get("ts") or datetime.now(timezone.utc).isoformat(),
                           })
                       elif driver.latitude and driver.longitude:
                           # Fallback to DB if Redis doesnt have data
                           cast(Any, emit)("driver_location_update", {
                               "driver_id": driver.id,
                               "first_name": driver.user.first_name,
                               "latitude": driver.latitude,
                               "longitude": driver.longitude,
                               "timestamp": datetime.now(timezone.utc).isoformat(),
                           })
                   except Exception as e:
                       logger.exception(f"❌ Error sending driver location for driver {driver.id}: {e}")

               logger.info(f"📡 Sent locations for {len(drivers)} drivers to company {company_id}")

           except Exception as e:
               logger.exception(f"❌ Error in get_driver_locations: {e}")
               emit("error", {"error": str(e)})

    @socketio.on("disconnect")
    def handle_disconnect():
        sid = _get_sid()
        info = _SID_INDEX.pop(sid, None)
        logger.info(f"👋 SIO disconnect sid={sid} info={info}")