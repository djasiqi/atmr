# backend/sockets/chat.py
import logging
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, Dict, cast
from typing import cast as tcast

from flask import request, session
from flask_jwt_extended import decode_token
from flask_socketio import SocketIO, emit, join_room

from ext import db, redis_client
from models import Company, Driver, Message, User, UserRole

logger = logging.getLogger("socketio")

# Petit index en mémoire pour le debug/nettoyage : sid -> infos
_SID_INDEX: Dict[str, Dict[str, Any]] = {}


def _get_sid(fallback_request=None) -> str:
    """Récupère le SID de la requête Socket.IO actuelle."""
    if fallback_request is None:
        fallback_request = request

    sid = getattr(fallback_request, "sid", None) or fallback_request.environ.get("socketio.sid")
    return str(sid) if sid is not None else ""


def _extract_token(auth) -> str | None:
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
    logger.info("🔧 [INIT] Initialisation des handlers Socket.IO chat")

    @socketio.on("connect", namespace="/")
    def handle_connect(auth):
        logger.info(f"🔌 [CONNECT] HANDLER APPELÉ ! auth={auth}")
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
            session["role"] = user.role.value.lower()

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
            emit("unauthorized", {"error": str(e)})
            return False

    @socketio.on("team_chat_message")
    def handle_team_chat(data):
        local_id = data.get("_localId")
        try:
            # ✅ SECURITY: Utiliser JWT depuis _SID_INDEX au lieu de session Flask
            sid = _get_sid()
            sid_data = _SID_INDEX.get(sid, {})
            user_public_id = sid_data.get("user_public_id")

            if not user_public_id:
                emit("error", {"error": "Session JWT introuvable. Reconnectez-vous."})
                return

            user = User.query.filter_by(public_id=user_public_id).first()
            if not user:
                emit("error", {"error": "Utilisateur non reconnu."})
                return

            content = (data.get("content") or "").strip()

            # ✅ Validation longueur message
            if len(content) > 1000:
                emit("error", {"error": "Message trop long (max 1000 caractères)."})
                return

            receiver_id = data.get("receiver_id")
            timestamp = datetime.now(UTC)
            if not content:
                emit("error", {"error": "Message vide non autorisé."})
                return

            # ✅ Validation receiver_id si fourni
            if receiver_id is not None:
                try:
                    receiver_id = int(receiver_id)
                    if receiver_id <= 0:
                        raise ValueError()
                except (TypeError, ValueError):
                    emit("error", {"error": "receiver_id invalide."})
                    return

            if user.role == UserRole.driver:
                driver = Driver.query.filter_by(user_id=user.id).first()
                if not driver:
                    emit("error", {"error": "Chauffeur introuvable."})
                    return
                company_id = driver.company_id
                # ✅ FIX: Utiliser "DRIVER" en majuscules pour matcher l'Enum SenderRole
                sender_role = "DRIVER"
                sender_id = user.id
                company_obj = None
                logger.info(f"📨 Chat driver: user_id={user.id}, driver_id={driver.id}, company_id={company_id}")
            elif user.role == UserRole.company:
                company_obj = Company.query.filter_by(user_id=user.id).first()
                if not company_obj:
                    emit("error", {"error": "Entreprise introuvable."})
                    return
                company_id = company_obj.id
                # ✅ FIX: Utiliser "COMPANY" en majuscules pour matcher l'Enum SenderRole
                sender_role = "COMPANY"
                # ✅ FIX: Ne jamais mettre sender_id=None pour l'entreprise
                sender_id = user.id
                logger.info(f"📨 Chat company: user_id={user.id}, company_id={company_id}")
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
                "sender_name": user.first_name,  # ✅ Utiliser user.first_name directement
                "content": content,
                "timestamp": timestamp.isoformat(),
                "type": "chat",
                "company_id": company_id,
                # ✅ FIX: company_name disponible et test en MAJ
                "company_name": (company_obj.name if (sender_role == "COMPANY" and company_obj) else None),
                "_localId": local_id,
            }

            room = f"company_{company_id}"
            # Pylance ne déclare pas kwarg 'room' sur emit -> cast en Any
            cast(Any, emit)("team_chat_message", payload, room=room)
            logger.info(f"📨 Message émis dans {room} par {sender_role} : {content}")

            # ✅ Si un receiver_id (driver) est fourni, notifier aussi sa room dédiée
            if receiver_id:
                driver_room = f"driver_{receiver_id}"
                cast(Any, emit)("team_chat_message", payload, room=driver_room)
                logger.info(f"📨 Message relayé vers {driver_room}")

        except Exception as e:
            logger.exception(f"❌ Erreur team_chat_message : {e}")
            emit("error", {"error": "Erreur d'envoi de message."})

    @socketio.on("join_driver_room")
    def handle_join_driver_room(data=None):
        try:
            # ✅ SECURITY: Utiliser JWT depuis _SID_INDEX
            sid = _get_sid()
            sid_data = _SID_INDEX.get(sid, {})
            user_public_id = sid_data.get("user_public_id")

            if not user_public_id:
                emit("error", {"error": "Session JWT introuvable. Reconnectez-vous."})
                return

            user = User.query.filter_by(public_id=user_public_id).first()
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
        """
        Handler pour la réception de la localisation du chauffeur.
        ✅ FIX: Accepte driver_id dans payload + fallback robuste par user_id
        """
        try:
            # 1. Récupération du SID pour le debug
            current_sid = _get_sid()
            logger.info(f"📍 driver_location reçu, SID={current_sid}, data={data}")

            # ✅ SECURITY: Utiliser JWT depuis _SID_INDEX uniquement
            sid_info = _SID_INDEX.get(current_sid, {})
            user_public_id = sid_info.get("user_public_id")
            user_role = sid_info.get("role")

            if not user_public_id:
                logger.warning(f"⛔ driver_location sans JWT public_id pour SID={current_sid}")
                emit("error", {"error": "Session JWT introuvable"})
                return

            # Récupérer l'user_id depuis la DB
            user = User.query.filter_by(public_id=user_public_id).first()
            if not user:
                emit("error", {"error": "Utilisateur introuvable"})
                return
            user_id = user.id

            # 4. Nouvelle approche: extraire driver_id du payload si disponible
            payload_driver_id = data.get("driver_id")

            # 5. Déterminer le driver à utiliser
            driver: Driver | None = None

            if payload_driver_id and isinstance(payload_driver_id, (int, str)):
                # Priorité au driver_id du payload (plus fiable)
                try:
                    candidate_id = int(payload_driver_id)
                    driver = Driver.query.get(candidate_id)
                    if driver:
                        logger.info(f"✅ Driver trouvé via payload: {driver.id}")
                    else:
                        logger.warning(f"⚠️ Driver introuvable via payload_driver_id={candidate_id}")
                except (ValueError, TypeError):
                    logger.warning(f"⚠️ driver_id non convertible: {payload_driver_id}")

            if not driver and user_id and user_role == "driver":
                # Fallback: recherche via user_id
                driver = Driver.query.filter_by(user_id=user_id).first()
                if driver:
                    logger.info(f"✅ Driver trouvé via user_id: {driver.id}")
                else:
                    logger.warning(f"⚠️ Aucun driver associé à user_id={user_id}")

            # Évite l'évaluation booléenne d'une colonne SQLA : on récupère un int ou None
            company_id_val = tcast(int | None, getattr(driver, "company_id", None))
            if (driver is None) or (company_id_val is None):
                logger.error(
                    f"❌ Driver introuvable: payload_driver_id={payload_driver_id}, user_id={user_id}"
                )
                emit("error", {"error": "Chauffeur introuvable ou non lié à une entreprise."})
                return

            latitude = data.get("latitude")
            longitude = data.get("longitude")

            # ✅ Validation stricte lat/lon
            try:
                lat = float(latitude)
                lon = float(longitude)
            except (TypeError, ValueError):
                emit("error", {"error": "Latitude et longitude requises."})
                return

            # ✅ Validation bornes géographiques
            if not (-90 <= lat <= 90):
                emit("error", {"error": "Latitude invalide (doit être entre -90 et 90)."})
                return

            if not (-180 <= lon <= 180):
                emit("error", {"error": "Longitude invalide (doit être entre -180 et 180)."})
                return

            latitude, longitude = lat, lon

            company_room = f"company_{company_id_val}"
            cast(Any, emit)(
                "driver_location_update",
                {
                    "driver_id": driver.id,
                    "first_name": getattr(getattr(driver, "user", None), "first_name", None),
                    "latitude": latitude,
                    "longitude": longitude,
                    "timestamp": datetime.now(UTC).isoformat(),
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
               # ✅ SECURITY: Utiliser JWT depuis _SID_INDEX
               sid = _get_sid()
               sid_data = _SID_INDEX.get(sid, {})
               user_public_id = sid_data.get("user_public_id")
               user_role = sid_data.get("role")

               if not user_public_id:
                   emit("error", {"error": "Session JWT introuvable. Reconnectez-vous."})
                   return

               # Récupérer user depuis public_id
               user = User.query.filter_by(public_id=user_public_id).first()
               if not user:
                   emit("error", {"error": "Utilisateur introuvable."})
                   return

               if user_role == "company":
                   company = Company.query.filter_by(user_id=user.id).first()
                   if not company:
                       emit("error", {"error": "Entreprise introuvable."})
                       return

                   room = f"company_{company.id}"
                   join_room(room)
                   emit("joined_company", {"company_id": company.id, "room": room})
                   logger.info(f"🏢 Company {company.id} joined room: {room}")
               elif user_role == "driver":
                   driver = Driver.query.filter_by(user_id=user.id).first()
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
               # ✅ SECURITY: Utiliser JWT depuis _SID_INDEX
               sid = _get_sid()
               company_info = _SID_INDEX.get(sid, {})
               user_public_id = company_info.get("user_public_id")
               user_role = company_info.get("role")
               company_id = company_info.get("company_id")

               if not user_public_id or user_role != "company":
                   emit("error", {"error": "Accès non autorisé pour la demande de localisation."})
                   return

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
                                   with suppress(Exception):
                                       loc_data[kf] = float(loc_data[kf])

                           # Emit location to the company room
                           cast(Any, emit)("driver_location_update", {
                               "driver_id": driver.id,
                               "first_name": getattr(getattr(driver, "user", None), "first_name", None),
                               "latitude": loc_data.get("lat"),
                               "longitude": loc_data.get("lon"),
                               "timestamp": loc_data.get("ts") or datetime.now(UTC).isoformat(),
                           })
                       elif (driver.latitude is not None) and (driver.longitude is not None):
                           # Fallback to DB if Redis doesnt have data
                           cast(Any, emit)("driver_location_update", {
                               "driver_id": driver.id,
                               "first_name": getattr(getattr(driver, "user", None), "first_name", None),
                               "latitude": driver.latitude,
                               "longitude": driver.longitude,
                               "timestamp": datetime.now(UTC).isoformat(),
                           })
                   except Exception as e:
                       # driver vient du for → devrait exister, mais on défend le log quand même
                       safe_id = getattr(driver, "id", None)
                       logger.exception(f"❌ Error sending driver location for driver {safe_id}: {e}")

               logger.info(f"📡 Sent locations for {len(drivers)} drivers to company {company_id}")

           except Exception as e:
               logger.exception(f"❌ Error in get_driver_locations: {e}")
               emit("error", {"error": str(e)})

    @socketio.on("disconnect")
    def handle_disconnect():
        sid = _get_sid()
        info = _SID_INDEX.pop(sid, None)
        logger.info(f"👋 SIO disconnect sid={sid} info={info}")
