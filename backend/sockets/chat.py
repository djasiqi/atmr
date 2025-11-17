# backend/sockets/chat.py
"""Socket.IO handlers pour le chat et la localisation.
Les fonctions de handlers sont enregistrées via @socketio.on() et appelées par le framework.
"""

import logging
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Dict, cast
from typing import cast as tcast

from flask import request, session
from flask_jwt_extended import decode_token
from flask_socketio import SocketIO, emit, join_room

from ext import db, redis_client
from models import Company, Driver, Message, SenderRole, User, UserRole

# Constantes pour éviter les valeurs magiques
RECEIVER_ID_ZERO = 0
LAT_THRESHOLD = 90
LON_THRESHOLD = 180
MAX_MESSAGE_LENGTH = 1000

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger("socketio")

# Petit index en mémoire pour le debug/nettoyage : sid -> infos
_SID_INDEX: Dict[str, Dict[str, Any]] = {}

# Les handlers Socket.IO sont enregistrés par @socketio.on()


def _get_sid(fallback_request=None) -> str:
    """Récupère le SID de la requête Socket.IO actuelle."""
    if fallback_request is None:
        fallback_request = request

    sid = getattr(fallback_request, "sid", None) or fallback_request.environ.get("socketio.sid")
    return str(sid) if sid is not None else ""


def _extract_token(auth) -> str | None:
    """Récupère le token JWT depuis Authorization, auth.token ou ?token=."""
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
    def handle_connect(auth: dict[str, Any] | None) -> bool:  # noqa: PLR0911
        logger.info("🔌 [CONNECT] HANDLER APPELÉ ! auth=%s", auth)
        client_ip = request.environ.get("REMOTE_ADDR")
        ua = request.headers.get("User-Agent", "Unknown")
        logger.info("🔌 SIO connect from %s UA=%s", client_ip, ua)

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
            logger.info("🧾 Token validé pour user %s", public_id)

            user = User.query.filter_by(public_id=public_id).first()
            if not user:
                logger.info("⛔ user not found: %s", public_id)
                emit("unauthorized", {"error": "Utilisateur non trouvé"})
                return False

            # Stash session minimale
            session["user_id"] = user.id
            session["first_name"] = user.first_name
            session["role"] = user.role.value.lower()

            if user.role == UserRole.driver:
                driver = Driver.query.filter_by(user_id=user.id).first()
                if not driver or not driver.company_id:
                    msg = "Chauffeur ou entreprise associée introuvable"
                    raise Exception(msg)

                company_room = f"company_{driver.company_id}"
                driver_room = f"driver_{driver.id}"
                join_room(company_room)
                join_room(driver_room)

                emit("connected", {"message": "✅ Chauffeur connecté"})
                logger.info("🔌 Driver %s -> rooms: %s, %s", driver.id, company_room, driver_room)

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
                logger.info("🏢 Company %s -> room: %s", company.id, room)

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
            logger.exception("❌ Erreur de connexion WebSocket : %s", e)
            emit("unauthorized", {"error": str(e)})
            return False

    @socketio.on("team_chat_message")
    def handle_team_chat(data):  # noqa: PLR0911
        local_id = data.get("_localId")
        logger.info("📨 [CHAT] team_chat_message reçu: data=%s", data)
        try:
            # ✅ SECURITY: Utiliser JWT depuis _SID_INDEX au lieu de session Flask
            sid = _get_sid()
            sid_data = _SID_INDEX.get(sid, {})
            user_public_id = sid_data.get("user_public_id")
            logger.info("📨 [CHAT] SID=%s, user_public_id=%s, sid_data=%s", sid, user_public_id, sid_data)

            if not user_public_id:
                logger.error("❌ [CHAT] Session JWT introuvable pour SID=%s", sid)
                emit("error", {"error": "Session JWT introuvable. Reconnectez-vous."})
                return

            user = User.query.filter_by(public_id=user_public_id).first()
            if not user:
                logger.error("❌ [CHAT] Utilisateur non trouvé pour public_id=%s", user_public_id)
                emit("error", {"error": "Utilisateur non reconnu."})
                return

            content_raw = data.get("content")
            logger.info("📨 [CHAT] Content brut reçu: %s (type=%s)", content_raw, type(content_raw).__name__)
            content = (content_raw or "").strip()
            logger.info("📨 [CHAT] Content après strip: '%s' (len=%d, bool=%s)", content, len(content), bool(content))

            # ✅ Validation longueur message
            if len(content) > MAX_MESSAGE_LENGTH:
                emit("error", {"error": f"Message trop long (max {MAX_MESSAGE_LENGTH} caractères)."})
                return

            receiver_id = data.get("receiver_id")
            timestamp = datetime.now(UTC)
            if not content:
                logger.error("❌ [CHAT] Message vide détecté après validation")
                emit("error", {"error": "Message vide non autorisé."})
                return

            # ✅ Validation receiver_id si fourni
            if receiver_id is not None:
                try:
                    receiver_id = int(receiver_id)
                    if receiver_id <= RECEIVER_ID_ZERO:
                        raise ValueError
                except (TypeError, ValueError):
                    emit("error", {"error": "receiver_id invalide."})
                    return

            if user.role == UserRole.driver:
                driver = Driver.query.filter_by(user_id=user.id).first()
                if not driver:
                    emit("error", {"error": "Chauffeur introuvable."})
                    return
                company_id = driver.company_id
                # ✅ FIX: Utiliser l'Enum SenderRole au lieu d'une chaîne
                sender_role = SenderRole.DRIVER
                sender_id = user.id
                company_obj = None
                logger.info("📨 Chat driver: user_id=%s, driver_id=%s, company_id=%s", user.id, driver.id, company_id)
            elif user.role == UserRole.company:
                company_obj = Company.query.filter_by(user_id=user.id).first()
                if not company_obj:
                    emit("error", {"error": "Entreprise introuvable."})
                    return
                company_id = company_obj.id
                # ✅ FIX: Utiliser l'Enum SenderRole au lieu d'une chaîne
                sender_role = SenderRole.COMPANY
                # ✅ FIX: Ne jamais mettre sender_id=None pour l'entreprise
                sender_id = user.id
                logger.info("📨 Chat company: user_id=%s, company_id=%s", user.id, company_id)
            else:
                emit("error", {"error": "Rôle non autorisé pour le chat."})
                return

            logger.info(
                "📨 [CHAT] Création du message: sender_id=%s, receiver_id=%s, company_id=%s, sender_role=%s, content='%s' (len=%d)",
                sender_id,
                receiver_id,
                company_id,
                sender_role,
                content[:50],
                len(content),
            )
            MessageCtor = cast("Any", Message)
            try:
                # ✅ Vérifier que le contenu n'est pas vide avant de créer le message
                content_final = content.strip() if content else ""
                logger.info(
                    "📨 [CHAT] Contenu final avant création: '%s' (len=%d, type=%s)",
                    content_final,
                    len(content_final),
                    type(content_final).__name__,
                )
                if not content_final:
                    logger.error("❌ [CHAT] Contenu vide détecté juste avant création: content='%s'", content)
                    emit("error", {"error": "Le contenu du message ne peut pas être vide."})
                    return

                logger.info("📨 [CHAT] Création de l'objet Message avec content='%s'", content_final[:100])
                message = MessageCtor(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    company_id=company_id,
                    sender_role=sender_role,
                    content=content_final,  # ✅ FIX: Utiliser le contenu réel et s'assurer qu'il est stripé
                    timestamp=timestamp,
                )
                logger.info(
                    "📨 [CHAT] Message créé avec succès, id=%s, content vérifié='%s'",
                    getattr(message, "id", "N/A"),
                    getattr(message, "content", "N/A")[:50],
                )
                logger.info("📨 [CHAT] Message créé, ajout à la session...")
                db.session.add(message)
                logger.info("📨 [CHAT] Commit en cours...")
                db.session.commit()
                logger.info(
                    "✅ [CHAT] Message sauvegardé en DB: id=%s, content='%s', sender_role=%s",
                    message.id,
                    content[:50],
                    sender_role,
                )
            except Exception as commit_err:
                db.session.rollback()
                logger.exception("❌ [CHAT] Erreur lors du commit du message: %s", commit_err)
                emit("error", {"error": f"Erreur lors de la sauvegarde du message: {commit_err!s}"})
                return

            payload = {
                "id": message.id,
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "receiver_name": message.receiver.first_name if message.receiver else None,
                "sender_role": sender_role.value
                if hasattr(sender_role, "value")
                else str(sender_role),  # ✅ S'assurer que c'est une chaîne
                "sender_name": user.first_name,  # ✅ Utiliser user.first_name directement
                "content": content,
                "timestamp": timestamp.isoformat(),
                "type": "chat",
                "company_id": company_id,
                # ✅ FIX: company_name disponible
                "company_name": (company_obj.name if (sender_role == SenderRole.COMPANY and company_obj) else None),
                "_localId": local_id,
            }

            room = f"company_{company_id}"
            # Pylance ne déclare pas kwarg 'room' sur emit -> cast en Any
            cast("Any", emit)("team_chat_message", payload, room=room)
            logger.info("📨 Message émis dans %s par %s : %s", room, sender_role, content)

            # ✅ Si un receiver_id (driver) est fourni, notifier aussi sa room dédiée
            if receiver_id:
                driver_room = f"driver_{receiver_id}"
                cast("Any", emit)("team_chat_message", payload, room=driver_room)
                logger.info("📨 Message relayé vers %s", driver_room)

        except Exception as e:
            logger.exception("❌ Erreur team_chat_message : %s", e)
            emit("error", {"error": "Erreur d'envoi de message."})

    @socketio.on("join_driver_room")
    def handle_join_driver_room(data=None):  # noqa: ARG001
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
            logger.info("✅ Driver %s joined rooms [%s, %s]", driver.id, driver_room, company_room)
            emit("joined_room", {"rooms": [driver_room, company_room]})

        except Exception as e:
            logger.exception("❌ Erreur join_driver_room : %s", e)
            emit("error", {"error": str(e)})

    @socketio.on("driver_location")
    def handle_driver_location(data):
        """Handler pour la réception de la localisation du chauffeur.
        ✅ FIX: Accepte driver_id dans payload + fallback robuste par user_id.
        """
        try:
            # 1. Récupération du SID pour le debug
            current_sid = _get_sid()
            logger.info("📍 driver_location reçu, SID=%s, data=%s", current_sid, data)

            # ✅ SECURITY: Utiliser JWT depuis _SID_INDEX uniquement
            sid_info = _SID_INDEX.get(current_sid, {})
            user_public_id = sid_info.get("user_public_id")
            user_role = sid_info.get("role")

            if not user_public_id:
                logger.warning("⛔ driver_location sans JWT public_id pour SID=%s", current_sid)
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
                        logger.info("✅ Driver trouvé via payload: %s", driver.id)
                    else:
                        logger.warning("⚠️ Driver introuvable via payload_driver_id=%s", candidate_id)
                except (ValueError, TypeError):
                    logger.warning("⚠️ driver_id non convertible: %s", payload_driver_id)

            if not driver and user_id and user_role == "driver":
                # Fallback: recherche via user_id
                driver = Driver.query.filter_by(user_id=user_id).first()
                if driver:
                    logger.info("✅ Driver trouvé via user_id: %s", driver.id)
                else:
                    logger.warning("⚠️ Aucun driver associé à user_id=%s", user_id)

            # Évite l'évaluation booléenne d'une colonne SQLA : on récupère un int ou None
            company_id_val = tcast("int | None", getattr(driver, "company_id", None))
            if (driver is None) or (company_id_val is None):
                logger.error("❌ Driver introuvable: payload_driver_id=%s, user_id=%s", payload_driver_id, user_id)
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
            if not (-LAT_THRESHOLD <= lat <= LAT_THRESHOLD):
                emit("error", {"error": "Latitude invalide (doit être entre -90 et 90)."})
                return

            if not (-LON_THRESHOLD <= lon <= LON_THRESHOLD):
                emit("error", {"error": "Longitude invalide (doit être entre -180 et 180)."})
                return

            latitude, longitude = lat, lon

            # 6. Persister la dernière position (Redis + DB)
            now_iso = datetime.now(UTC).isoformat()
            try:
                # 🔹 Redis: clé courte pour get_driver_locations()
                key = f"driver:{driver.id}:loc"
                redis_client.hset(
                    key,
                    mapping={
                        "lat": str(latitude),
                        "lon": str(longitude),
                        "ts": now_iso,
                    },
                )
                # TTL raisonnable (par ex. 24h) pour éviter d'accumuler les clés mortes
                with suppress(Exception):
                    redis_client.expire(key, 24 * 3600)
            except Exception as e_redis:
                logger.exception("❌ Erreur écriture Redis driver_location pour driver %s: %s", driver.id, e_redis)

            try:
                # 🔹 DB: stocker aussi sur le modèle Driver (fallback si Redis down)
                driver.latitude = latitude
                driver.longitude = longitude
                db.session.add(driver)
                db.session.commit()
            except Exception as e_db:
                db.session.rollback()
                logger.exception("❌ Erreur mise à jour DB driver_location pour driver %s: %s", driver.id, e_db)

            # 7. Diffuser la position aux rooms de l'entreprise
            company_room = f"company_{company_id_val}"
            cast("Any", emit)(
                "driver_location_update",
                {
                    "driver_id": driver.id,
                    "first_name": getattr(getattr(driver, "user", None), "first_name", None),
                    "latitude": latitude,
                    "longitude": longitude,
                    "timestamp": now_iso,
                },
                room=company_room,
            )
            logger.info("📡 Loc -> %s (driver %s) %s,%s", company_room, driver.id, latitude, longitude)

        except Exception as e:
            logger.exception("❌ Erreur driver_location : %s", e)
            emit("error", {"error": str(e)})

    @socketio.on("driver_location_batch")
    def handle_driver_location_batch(data):
        """Handler pour la réception de batch de localisations du chauffeur.
        Traite chaque position du batch et les persiste.
        """
        try:
            current_sid = _get_sid()
            logger.info(
                "📍 driver_location_batch reçu, SID=%s, positions_count=%s", current_sid, len(data.get("positions", []))
            )

            sid_info = _SID_INDEX.get(current_sid, {})
            user_public_id = sid_info.get("user_public_id")
            user_role = sid_info.get("role")

            if not user_public_id:
                logger.warning("⛔ driver_location_batch sans JWT public_id pour SID=%s", current_sid)
                emit("error", {"error": "Session JWT introuvable"})
                return

            user = User.query.filter_by(public_id=user_public_id).first()
            if not user:
                emit("error", {"error": "Utilisateur introuvable"})
                return

            payload_driver_id = data.get("driver_id")
            driver: Driver | None = None

            if payload_driver_id and isinstance(payload_driver_id, (int, str)):
                try:
                    candidate_id = int(payload_driver_id)
                    driver = Driver.query.get(candidate_id)
                    if driver:
                        logger.info("✅ Driver trouvé via payload: %s", driver.id)
                except (ValueError, TypeError):
                    logger.warning("⚠️ driver_id non convertible: %s", payload_driver_id)

            if not driver and user_role == "driver":
                driver = Driver.query.filter_by(user_id=user.id).first()
                if driver:
                    logger.info("✅ Driver trouvé via user_id: %s", driver.id)

            company_id_val = tcast("int | None", getattr(driver, "company_id", None))
            if (driver is None) or (company_id_val is None):
                logger.error(
                    "❌ Driver introuvable pour driver_location_batch: payload_driver_id=%s, user_id=%s",
                    payload_driver_id,
                    user.id,
                )
                emit("error", {"error": "Chauffeur introuvable ou non lié à une entreprise."})
                return

            positions = data.get("positions", [])
            if not positions:
                logger.warning("⚠️ driver_location_batch vide")
                return

            company_room = f"company_{company_id_val}"
            now_iso = datetime.now(UTC).isoformat()

            # Traiter chaque position du batch
            for pos in positions:
                try:
                    latitude = float(pos.get("latitude", 0))
                    longitude = float(pos.get("longitude", 0))

                    if not (-LAT_THRESHOLD <= latitude <= LAT_THRESHOLD):
                        continue
                    if not (-LON_THRESHOLD <= longitude <= LON_THRESHOLD):
                        continue

                    # Persister dans Redis
                    try:
                        key = f"driver:{driver.id}:loc"
                        redis_client.hset(
                            key,
                            mapping={
                                "lat": str(latitude),
                                "lon": str(longitude),
                                "ts": now_iso,
                            },
                        )
                        redis_client.expire(key, 24 * 3600)
                    except Exception as e_redis:
                        logger.exception("❌ Erreur Redis driver_location_batch pour driver %s: %s", driver.id, e_redis)

                    # Persister dans DB (seulement la dernière position du batch)
                    if pos == positions[-1]:
                        try:
                            driver.latitude = latitude
                            driver.longitude = longitude
                            db.session.add(driver)
                            db.session.commit()
                        except Exception as e_db:
                            db.session.rollback()
                            logger.exception("❌ Erreur DB driver_location_batch pour driver %s: %s", driver.id, e_db)

                    # Diffuser chaque position
                    cast("Any", emit)(
                        "driver_location_update",
                        {
                            "driver_id": driver.id,
                            "first_name": getattr(getattr(driver, "user", None), "first_name", None),
                            "latitude": latitude,
                            "longitude": longitude,
                            "timestamp": pos.get("timestamp") or now_iso,
                        },
                        room=company_room,
                    )
                except (TypeError, ValueError) as e:
                    logger.warning("⚠️ Position invalide dans batch: %s", e)
                    continue

            logger.info("📡 Batch -> %s (driver %s) %s positions", company_room, driver.id, len(positions))

        except Exception as e:
            logger.exception("❌ Erreur driver_location_batch : %s", e)
            emit("error", {"error": str(e)})

    @socketio.on("join_company")
    def handle_join_company(data=None):  # noqa: ARG001
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
                logger.info("🏢 Company %s joined room: %s", company.id, room)
            elif user_role == "driver":
                driver = Driver.query.filter_by(user_id=user.id).first()
                if not driver or not driver.company_id:
                    emit("error", {"error": "Chauffeur ou entreprise associée introuvable."})
                    return

                room = f"company_{driver.company_id}"
                join_room(room)
                emit("joined_company", {"company_id": driver.company_id, "room": room})
                logger.info("🚗 Driver %s joined company room: %s", driver.id, room)
            else:
                emit("error", {"error": "Rôle non autorisé pour rejoindre une room entreprise."})
        except Exception as e:
            logger.exception("❌ Error in join_company: %s", e)
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
                    h: Mapping[bytes, Any] = cast("Mapping[bytes, Any]", h_raw)

                    if h:
                        # Redis returns bytes -> decode
                        def _dec(v):
                            try:
                                return v.decode()
                            except Exception:
                                return v

                        loc_data = {k.decode(): _dec(v) for k, v in h.items()}

                        # Cast numeric fields
                        for kf in ("lat", "lon", "speed", "heading", "accuracy"):
                            if kf in loc_data:
                                with suppress(Exception):
                                    loc_data[kf] = float(loc_data[kf])

                        # Emit location to the company room
                        cast("Any", emit)(
                            "driver_location_update",
                            {
                                "driver_id": driver.id,
                                "first_name": getattr(getattr(driver, "user", None), "first_name", None),
                                "latitude": loc_data.get("lat"),
                                "longitude": loc_data.get("lon"),
                                "timestamp": loc_data.get("ts") or datetime.now(UTC).isoformat(),
                            },
                        )
                    elif (driver.latitude is not None) and (driver.longitude is not None):
                        # Fallback to DB if Redis doesnt have data
                        cast("Any", emit)(
                            "driver_location_update",
                            {
                                "driver_id": driver.id,
                                "first_name": getattr(getattr(driver, "user", None), "first_name", None),
                                "latitude": driver.latitude,
                                "longitude": driver.longitude,
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )
                except Exception as e:
                    # driver vient du for → devrait exister, mais on défend le log quand même
                    safe_id = getattr(driver, "id", None)
                    logger.exception("❌ Error sending driver location for driver %s: %s", safe_id, e)

            logger.info("📡 Sent locations for %s drivers to company %s", len(drivers), company_id)

        except Exception as e:
            logger.exception("❌ Error in get_driver_locations: %s", e)
            emit("error", {"error": str(e)})

    @socketio.on("disconnect")
    def handle_disconnect():
        sid = _get_sid()
        info = _SID_INDEX.pop(sid, None)
        logger.info("👋 SIO disconnect sid=%s info=%s", sid, info)

    # Référencer les handlers pour indiquer qu'ils sont utilisés par Socket.IO
    _registered_handlers = (
        handle_connect,
        handle_team_chat,
        handle_join_driver_room,
        handle_driver_location,
        handle_driver_location_batch,
        handle_join_company,
        handle_get_driver_locations,
        handle_disconnect,
    )
    # Les handlers sont enregistrés via @socketio.on() ci-dessus
