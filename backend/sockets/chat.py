# backend/sockets/chat.py
# pyright: reportUnusedFunction=false
# Les fonctions handlers sont enregistrées via @socketio.on() et appelées
# par le framework Socket.IO, donc elles ne sont pas directement "accédées"
# dans le code Python.
"""Socket.IO handlers pour le chat et la localisation.
Les fonctions de handlers sont enregistrées via @socketio.on()
et appelées par le framework.
"""

import logging
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Dict, cast
from typing import cast as tcast

import jwt.exceptions as jwt_exceptions
from flask import request, session
from flask_jwt_extended import decode_token
from flask_socketio import SocketIO, emit, join_room

from ext import db, redis_client
from models import Company, Driver, Message, SenderRole, User, UserRole
from services.location_service import get_location_service
from services.spam_protection import can_send_message
from services.websocket_metrics import ws_metrics
from services.websocket_rate_limiter import ws_rate_limiter

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
# Note: Rate limiting géré par WebSocketRateLimiter (backend/services/websocket_rate_limiter.py)


def _get_sid(fallback_request=None) -> str:
    """Récupère le SID de la requête Socket.IO actuelle."""
    if fallback_request is None:
        fallback_request = request

    sid = getattr(fallback_request, "sid", None) or fallback_request.environ.get(
        "socketio.sid"
    )
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
        client_ip = request.environ.get("REMOTE_ADDR", "unknown")
        ua = request.headers.get("User-Agent", "Unknown")
        trace_id = request.headers.get("X-Trace-ID") or request.headers.get("Trace-Id")
        now = datetime.now(UTC)

        # ✅ Rate limiting : utiliser WebSocketRateLimiter (remplace ancien système)
        allowed, retry_after = ws_rate_limiter.check_rate_limit(
            "connect",
            client_ip=client_ip,
        )
        if not allowed:
            logger.warning(
                "socket_rate_limit_exceeded",
                extra={
                    "event": "rate_limit_exceeded",
                    "ip": client_ip,
                    "retry_after": retry_after or 0,
                    "timestamp": now.isoformat(),
                    "request_trace_id": trace_id,
                },
            )
            emit(
                "error",
                {
                    "error": f"Trop de tentatives de connexion. Réessayez dans {retry_after or 60} secondes.",
                    "retry_after": retry_after or 60,
                },
            )
            ws_metrics.on_error("rate_limit_exceeded")
            ws_metrics.on_rate_limit_hit("connect")
            return False

        logger.info(
            "socket_connect_attempt",
            extra={
                "event": "connect_attempt",
                "ip": client_ip,
                "user_agent": ua,
                "timestamp": now.isoformat(),
                "request_trace_id": trace_id,
            },
        )

        try:
            token = _extract_token(auth)
            if not token:
                logger.info(
                    "socket_connect_refused",
                    extra={
                        "event": "connect_refused",
                        "reason": "token_missing",
                        "ip": client_ip,
                        "timestamp": now.isoformat(),
                        "request_trace_id": trace_id,
                    },
                )
                emit("unauthorized", {"error": "Token JWT manquant"})
                ws_metrics.on_error("token_missing")
                return False

            # Vérifie & décode (lève si invalide/expiré)
            try:
                decoded = decode_token(token)
            except jwt_exceptions.ExpiredSignatureError:
                logger.info(
                    "socket_connect_error",
                    extra={
                        "event": "connect_error",
                        "reason": "token_expired",
                        "ip": client_ip,
                        "timestamp": now.isoformat(),
                        "request_trace_id": trace_id,
                    },
                )
                emit(
                    "unauthorized",
                    {"error": "Token expiré. Veuillez vous reconnecter."},
                )
                ws_metrics.on_error("token_expired")
                return False
            except jwt_exceptions.InvalidAudienceError:
                logger.info(
                    "socket_connect_error",
                    extra={
                        "event": "connect_error",
                        "reason": "token_invalid_audience",
                        "ip": client_ip,
                        "timestamp": now.isoformat(),
                        "request_trace_id": trace_id,
                    },
                )
                emit("unauthorized", {"error": "Token invalide (audience incorrecte)."})
                ws_metrics.on_error("token_invalid_audience")
                return False
            except Exception as e:
                logger.warning(
                    "socket_connect_error",
                    extra={
                        "event": "connect_error",
                        "reason": "token_decode_error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "ip": client_ip,
                        "timestamp": now.isoformat(),
                        "request_trace_id": trace_id,
                    },
                )
                emit("unauthorized", {"error": "Token invalide."})
                ws_metrics.on_error("token_decode_error")
                return False

            public_id = decoded.get("sub")
            if not public_id:
                logger.info(
                    "socket_connect_refused",
                    extra={
                        "event": "connect_refused",
                        "reason": "token_no_sub",
                        "ip": client_ip,
                        "timestamp": now.isoformat(),
                        "request_trace_id": trace_id,
                    },
                )
                emit("unauthorized", {"error": "Token sans 'sub'"})
                ws_metrics.on_error("token_invalid")
                return False

            logger.debug(
                "socket_token_validated",
                extra={
                    "event": "token_validated",
                    "user_public_id": public_id,
                    "ip": client_ip,
                    "timestamp": now.isoformat(),
                    "request_trace_id": trace_id,
                },
            )

            user = User.query.filter_by(public_id=public_id).first()
            if not user:
                logger.info(
                    "socket_connect_refused",
                    extra={
                        "event": "connect_refused",
                        "reason": "user_not_found",
                        "user_public_id": public_id,
                        "ip": client_ip,
                        "timestamp": now.isoformat(),
                        "request_trace_id": trace_id,
                    },
                )
                emit("unauthorized", {"error": "Utilisateur non trouvé"})
                ws_metrics.on_error("user_not_found")
                return False

            # Stash session minimale
            session["user_id"] = user.id
            session["first_name"] = user.first_name
            session["role"] = user.role.value.lower()

            sid = _get_sid()
            trace_id = trace_id or f"socket-{sid[:8]}"

            if user.role == UserRole.driver:
                driver = Driver.query.filter_by(user_id=user.id).first()
                if not driver or not driver.company_id:
                    msg = "Chauffeur ou entreprise associée introuvable"
                    logger.error(
                        "socket_connect_error",
                        extra={
                            "event": "connect_error",
                            "reason": "driver_or_company_not_found",
                            "user_id": user.id,
                            "ip": client_ip,
                            "timestamp": now.isoformat(),
                            "request_trace_id": trace_id,
                        },
                    )
                    raise Exception(msg)

                company_room = f"company_{driver.company_id}"
                driver_room = f"driver_{driver.id}"
                join_room(company_room)
                join_room(driver_room)

                emit("connected", {"message": "✅ Chauffeur connecté"})

                _SID_INDEX[sid] = {
                    "user_public_id": public_id,
                    "user_id": user.id,
                    "driver_id": driver.id,
                    "company_id": driver.company_id,
                    "ip": client_ip,
                    "role": "driver",
                }

                # ✅ Métriques
                ws_metrics.on_connect(company_id=driver.company_id, user_id=user.id)
                # ✅ Tracking rooms
                ws_metrics.on_room_join(company_room)
                ws_metrics.on_room_join(driver_room)

                logger.info(
                    "socket_connect_success",
                    extra={
                        "event": "connect_success",
                        "sid": sid,
                        "user_id": user.id,
                        "user_public_id": public_id,
                        "driver_id": driver.id,
                        "company_id": driver.company_id,
                        "role": "driver",
                        "rooms": [company_room, driver_room],
                        "ip": client_ip,
                        "timestamp": now.isoformat(),
                        "request_trace_id": trace_id,
                    },
                )

            elif user.role == UserRole.company:
                company = Company.query.filter_by(user_id=user.id).first()
                if not company:
                    logger.error(
                        "socket_connect_error",
                        extra={
                            "event": "connect_error",
                            "reason": "company_not_found",
                            "user_id": user.id,
                            "ip": client_ip,
                            "timestamp": now.isoformat(),
                            "request_trace_id": trace_id,
                        },
                    )
                    emit("unauthorized", {"error": "Entreprise introuvable"})
                    ws_metrics.on_error("company_not_found")
                    return False

                room = f"company_{company.id}"
                join_room(room)
                emit("connected", {"message": f"✅ Entreprise connectée à {room}"})

                _SID_INDEX[sid] = {
                    "user_public_id": public_id,
                    "user_id": user.id,
                    "company_id": company.id,
                    "ip": client_ip,
                    "role": "company",
                }

                # ✅ Métriques
                ws_metrics.on_connect(company_id=company.id, user_id=user.id)
                # ✅ Tracking rooms
                ws_metrics.on_room_join(room)

                logger.info(
                    "socket_connect_success",
                    extra={
                        "event": "connect_success",
                        "sid": sid,
                        "user_id": user.id,
                        "user_public_id": public_id,
                        "company_id": company.id,
                        "role": "company",
                        "rooms": [room],
                        "ip": client_ip,
                        "timestamp": now.isoformat(),
                        "request_trace_id": trace_id,
                    },
                )
            else:
                logger.warning(
                    "socket_connect_refused",
                    extra={
                        "event": "connect_refused",
                        "reason": "role_not_authorized",
                        "user_id": user.id,
                        "role": user.role.value,
                        "ip": client_ip,
                        "timestamp": now.isoformat(),
                        "request_trace_id": trace_id,
                    },
                )
                emit("unauthorized", {"error": "Rôle non autorisé pour le chat"})
                ws_metrics.on_error("role_not_authorized")
                return False

            return True

        except Exception as e:
            logger.exception(
                "socket_connect_error",
                extra={
                    "event": "connect_error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "ip": client_ip,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "request_trace_id": trace_id if "trace_id" in locals() else None,
                },
            )
            ws_metrics.on_error("connect_exception")
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
            logger.info(
                "📨 [CHAT] SID=%s, user_public_id=%s, sid_data=%s",
                sid,
                user_public_id,
                sid_data,
            )

            if not user_public_id:
                logger.error("❌ [CHAT] Session JWT introuvable pour SID=%s", sid)
                emit("error", {"error": "Session JWT introuvable. Reconnectez-vous."})
                return

            user = User.query.filter_by(public_id=user_public_id).first()
            if not user:
                logger.error(
                    "❌ [CHAT] Utilisateur non trouvé pour public_id=%s", user_public_id
                )
                emit("error", {"error": "Utilisateur non reconnu."})
                return

            # ✅ Rate limiting: vérifier avant anti-spam (plus restrictif)
            client_ip = request.environ.get("REMOTE_ADDR", "unknown")
            allowed, retry_after = ws_rate_limiter.check_rate_limit(
                "team_chat_message",
                user_id=user.id,
                client_ip=client_ip,
            )
            if not allowed:
                logger.warning(
                    "🚫 Rate limit team_chat_message dépassé pour user_id=%s, retry_after=%d",
                    user.id,
                    retry_after or 0,
                )
                emit(
                    "rate_limit_exceeded",
                    {
                        "event": "rate_limit_exceeded",
                        "message": f"Trop de messages. Réessayez dans {retry_after} secondes.",
                        "attempts": 1,
                        "retry_after_seconds": retry_after,
                    },
                )
                ws_metrics.on_error("rate_limit_exceeded")
                ws_metrics.on_rate_limit_hit("team_chat_message")
                return

            # ✅ Anti-spam: vérifier le taux d'envoi (1 message/seconde) - garde pour compatibilité
            allowed_spam, spam_error = can_send_message(user.id)
            if not allowed_spam:
                logger.warning(
                    "🚫 [CHAT] Anti-spam: Utilisateur %s - %s", user.id, spam_error
                )
                emit(
                    "error",
                    {"error": spam_error or "Trop de messages. Attendez 1 seconde."},
                )
                return

            content_raw = data.get("content")
            logger.info(
                "📨 [CHAT] Content brut reçu: %s (type=%s)",
                content_raw,
                type(content_raw).__name__,
            )
            content = (content_raw or "").strip() if content_raw else ""
            logger.info(
                "📨 [CHAT] Content après strip: '%s' (len=%d, bool=%s)",
                content,
                len(content),
                bool(content),
            )

            # Support pour images et PDF
            image_url = data.get("image_url") or data.get("image")
            pdf_url = data.get("pdf_url") or data.get("pdf")
            pdf_filename = data.get("pdf_filename")
            pdf_size = data.get("pdf_size")

            # ✅ Limite: 1 fichier par message (image OU PDF, pas les deux)
            has_image = bool(image_url)
            has_pdf = bool(pdf_url)
            if has_image and has_pdf:
                logger.error(
                    (
                        "❌ [CHAT] Limite: 1 fichier par message "
                        "(image OU PDF, pas les deux)"
                    )
                )
                emit(
                    "error",
                    {
                        "error": (
                            "Limite: 1 fichier par message. "
                            "Choisissez une image OU un PDF."
                        )
                    },
                )
                return

            # ✅ Validation : le message doit avoir au moins du contenu texte,
            # une image ou un PDF
            has_content = bool(content)
            if not (has_content or has_image or has_pdf):
                logger.error("❌ [CHAT] Message vide : ni contenu, ni image, ni PDF")
                emit(
                    "error",
                    {
                        "error": (
                            "Le message doit contenir du texte, une image ou un PDF."
                        )
                    },
                )
                return

            # ✅ Validation longueur message (si contenu texte présent)
            if has_content and len(content) > MAX_MESSAGE_LENGTH:
                emit(
                    "error",
                    {
                        "error": (
                            f"Message trop long (max {MAX_MESSAGE_LENGTH} caractères)."
                        )
                    },
                )
                return

            receiver_id = data.get("receiver_id")
            timestamp = datetime.now(UTC)

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
                logger.info(
                    "📨 Chat driver: user_id=%s, driver_id=%s, company_id=%s",
                    user.id,
                    driver.id,
                    company_id,
                )
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
                logger.info(
                    "📨 Chat company: user_id=%s, company_id=%s", user.id, company_id
                )
            else:
                emit("error", {"error": "Rôle non autorisé pour le chat."})
                return

            logger.info(
                (
                    "📨 [CHAT] Création du message: sender_id=%s, "
                    "receiver_id=%s, company_id=%s, sender_role=%s, "
                    "content='%s' (len=%d)"
                ),
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
                # Permettre None si seulement image/PDF
                content_final = content.strip() if content else None
                logger.info(
                    (
                        "📨 [CHAT] Contenu final avant création: '%s' "
                        "(len=%d, type=%s, has_image=%s, has_pdf=%s)"
                    ),
                    content_final or "(vide)",
                    len(content_final) if content_final else 0,
                    type(content_final).__name__ if content_final else "None",
                    has_image,
                    has_pdf,
                )
                # Validation déjà faite plus haut : au moins content, image ou PDF

                logger.info(
                    "📨 [CHAT] Création de l'objet Message avec content='%s'",
                    content_final[:100] if content_final else "(vide)",
                )
                message = MessageCtor(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    company_id=company_id,
                    sender_role=sender_role,
                    content=content_final
                    if content_final
                    else None,  # Permettre None si seulement image/PDF
                    timestamp=timestamp,
                    image_url=image_url if has_image else None,
                    pdf_url=pdf_url if has_pdf else None,
                    pdf_filename=pdf_filename if has_pdf else None,
                    pdf_size=int(pdf_size) if has_pdf and pdf_size else None,
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
                    (
                        "✅ [CHAT] Message sauvegardé en DB: id=%s, "
                        "content='%s', sender_role=%s"
                    ),
                    message.id,
                    content[:50],
                    sender_role,
                )
            except Exception as commit_err:
                db.session.rollback()
                logger.exception(
                    "❌ [CHAT] Erreur lors du commit du message: %s", commit_err
                )
                emit(
                    "error",
                    {
                        "error": (
                            f"Erreur lors de la sauvegarde du message: {commit_err!s}"
                        )
                    },
                )
                return

            payload = {
                "id": message.id,
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "receiver_name": message.receiver.first_name
                if message.receiver
                else None,
                "sender_role": sender_role.value
                if hasattr(sender_role, "value")
                else str(sender_role),  # ✅ S'assurer que c'est une chaîne
                "sender_name": user.first_name,  # ✅ Utiliser user.first_name
                # directement
                "content": content,
                "timestamp": timestamp.isoformat(),
                "type": "chat",
                "company_id": company_id,
                # ✅ FIX: company_name disponible
                "company_name": (
                    company_obj.name
                    if (sender_role == SenderRole.COMPANY and company_obj)
                    else None
                ),
                "_localId": local_id,
                # Support pour images et PDF
                "image_url": image_url if has_image else None,
                "pdf_url": pdf_url if has_pdf else None,
                "pdf_filename": pdf_filename if has_pdf else None,
                "pdf_size": int(pdf_size) if has_pdf and pdf_size else None,
            }

            room = f"company_{company_id}"
            # Pylance ne déclare pas kwarg 'room' sur emit -> cast en Any
            cast("Any", emit)("team_chat_message", payload, room=room)
            logger.info(
                "📨 Message émis dans %s par %s : %s", room, sender_role, content
            )

            # ✅ Si un receiver_id (driver) est fourni, notifier aussi sa room dédiée
            if receiver_id:
                driver_room = f"driver_{receiver_id}"
                cast("Any", emit)("team_chat_message", payload, room=driver_room)
                logger.info("📨 Message relayé vers %s", driver_room)

        except Exception as e:
            logger.exception("❌ Erreur team_chat_message : %s", e)
            emit("error", {"error": "Erreur d'envoi de message."})

    @socketio.on("typing_start")
    def handle_typing_start(data=None):  # noqa: ARG001
        """Handler pour l'indicateur de frappe (typing indicator)."""
        try:
            sid = _get_sid()
            sid_data = _SID_INDEX.get(sid, {})
            user_public_id = sid_data.get("user_public_id")
            company_id = sid_data.get("company_id")

            if not user_public_id or not company_id:
                return

            # ✅ Rate limiting: vérifier avant émission
            client_ip = request.environ.get("REMOTE_ADDR", "unknown")
            # Récupérer user_id depuis la DB si nécessaire pour le rate limiting
            user_id = None
            try:
                from models import User

                user = User.query.filter_by(public_id=user_public_id).first()
                if user:
                    user_id = user.id
            except Exception:
                pass  # Si User non disponible, utiliser seulement IP

            allowed, retry_after = ws_rate_limiter.check_rate_limit(
                "typing_start",
                user_id=user_id,
                client_ip=client_ip,
            )
            if not allowed:
                logger.warning(
                    "🚫 Rate limit typing_start dépassé pour user_id=%s, retry_after=%d",
                    user_id or user_public_id,
                    retry_after or 0,
                )
                emit(
                    "rate_limit_exceeded",
                    {
                        "event": "rate_limit_exceeded",
                        "message": f"Trop de requêtes. Réessayez dans {retry_after} secondes.",
                        "retry_after": retry_after or 0,
                    },
                )
                ws_metrics.on_rate_limit_hit("typing_start")
                return

            # Diffuser l'indicateur de frappe à la room de l'entreprise
            room = f"company_{company_id}"
            cast("Any", emit)(
                "typing_start", {"user_id": user_public_id}, room=room, skip_sid=sid
            )
            logger.debug("⌨️ typing_start de %s dans %s", user_public_id, room)

        except Exception as e:
            logger.exception("❌ Erreur typing_start : %s", e)

    @socketio.on("typing_stop")
    def handle_typing_stop(data=None):  # noqa: ARG001
        """Handler pour arrêter l'indicateur de frappe."""
        try:
            sid = _get_sid()
            sid_data = _SID_INDEX.get(sid, {})
            user_public_id = sid_data.get("user_public_id")
            company_id = sid_data.get("company_id")

            if not user_public_id or not company_id:
                return

            # ✅ Rate limiting: vérifier avant émission
            client_ip = request.environ.get("REMOTE_ADDR", "unknown")
            # Récupérer user_id depuis la DB si nécessaire pour le rate limiting
            user_id = None
            try:
                from models import User

                user = User.query.filter_by(public_id=user_public_id).first()
                if user:
                    user_id = user.id
            except Exception:
                pass  # Si User non disponible, utiliser seulement IP

            allowed, retry_after = ws_rate_limiter.check_rate_limit(
                "typing_stop",
                user_id=user_id,
                client_ip=client_ip,
            )
            if not allowed:
                logger.warning(
                    "🚫 Rate limit typing_stop dépassé pour user_id=%s, retry_after=%d",
                    user_id or user_public_id,
                    retry_after or 0,
                )
                emit(
                    "rate_limit_exceeded",
                    {
                        "event": "rate_limit_exceeded",
                        "message": f"Trop de requêtes. Réessayez dans {retry_after} secondes.",
                        "retry_after": retry_after or 0,
                    },
                )
                ws_metrics.on_rate_limit_hit("typing_stop")
                return

            # Diffuser l'arrêt de frappe à la room de l'entreprise
            room = f"company_{company_id}"
            cast("Any", emit)(
                "typing_stop", {"user_id": user_public_id}, room=room, skip_sid=sid
            )
            logger.debug("⌨️ typing_stop de %s dans %s", user_public_id, room)

        except Exception as e:
            logger.exception("❌ Erreur typing_stop : %s", e)

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
                emit(
                    "error",
                    {"error": "Seuls les chauffeurs peuvent rejoindre cette room."},
                )
                return

            driver = Driver.query.filter_by(user_id=user.id).first()
            if not driver:
                emit("error", {"error": "Chauffeur introuvable"})
                return

            driver_room = f"driver_{driver.id}"
            join_room(driver_room)
            company_room = f"company_{driver.company_id}"
            join_room(company_room)
            # ✅ Tracking rooms
            ws_metrics.on_room_join(driver_room)
            ws_metrics.on_room_join(company_room)
            logger.info(
                "✅ Driver %s joined rooms [%s, %s]",
                driver.id,
                driver_room,
                company_room,
            )
            emit("joined_room", {"rooms": [driver_room, company_room]})

        except Exception as e:
            logger.exception("❌ Erreur join_driver_room : %s", e)
            emit("error", {"error": str(e)})

    @socketio.on("driver_location")
    def handle_driver_location(data):  # noqa: PLR0911
        """Handler pour la réception de la localisation du chauffeur.
        ✅ FIX: Accepte driver_id dans payload + fallback robuste par user_id.

        Note: PLR0911 (too many returns) ignoré car les returns sont nécessaires
        pour la validation et la gestion d'erreurs (sécurité, rate limiting, etc.).
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
                logger.warning(
                    "⛔ driver_location sans JWT public_id pour SID=%s", current_sid
                )
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

            # ✅ Rate limiting: vérifier après avoir récupéré user_id et driver_id
            client_ip = request.environ.get("REMOTE_ADDR", "unknown")

            # Déterminer driver_id pour rate limiting
            driver_id_for_rate_limit = payload_driver_id
            if not driver_id_for_rate_limit:
                # Fallback: chercher via user_id
                driver = Driver.query.filter_by(user_id=user_id).first()
                if driver:
                    driver_id_for_rate_limit = driver.id

            if driver_id_for_rate_limit:
                allowed, retry_after = ws_rate_limiter.check_rate_limit(
                    "driver_location",
                    user_id=user_id,
                    driver_id=int(driver_id_for_rate_limit),
                    client_ip=client_ip,
                )
                if not allowed:
                    logger.warning(
                        "🚫 Rate limit driver_location dépassé pour driver_id=%s, retry_after=%d",
                        driver_id_for_rate_limit,
                        retry_after or 0,
                    )
                    emit(
                        "rate_limit_exceeded",
                        {
                            "event": "rate_limit_exceeded",
                            "message": f"Trop de mises à jour de position. Réessayez dans {retry_after} secondes.",
                            "attempts": 1,
                            "retry_after_seconds": retry_after,
                        },
                    )
                    ws_metrics.on_error("rate_limit_exceeded")
                    ws_metrics.on_rate_limit_hit("driver_location")
                    return

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
                        logger.warning(
                            "⚠️ Driver introuvable via payload_driver_id=%s",
                            candidate_id,
                        )
                except (ValueError, TypeError):
                    logger.warning(
                        "⚠️ driver_id non convertible: %s",
                        payload_driver_id,
                    )

            if not driver and user_id and user_role == "driver":
                # Fallback: recherche via user_id
                driver = Driver.query.filter_by(user_id=user_id).first()
                if driver:
                    logger.info("✅ Driver trouvé via user_id: %s", driver.id)
                else:
                    logger.warning("⚠️ Aucun driver associé à user_id=%s", user_id)

            # Évite l'évaluation booléenne d'une colonne SQLA :
            # on récupère un int ou None
            company_id_val = tcast("int | None", getattr(driver, "company_id", None))
            if (driver is None) or (company_id_val is None):
                logger.error(
                    "❌ Driver introuvable: payload_driver_id=%s, user_id=%s",
                    payload_driver_id,
                    user_id,
                )
                emit(
                    "error",
                    {"error": "Chauffeur introuvable ou non lié à une entreprise."},
                )
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
                emit(
                    "error", {"error": "Latitude invalide (doit être entre -90 et 90)."}
                )
                return

            if not (-LON_THRESHOLD <= lon <= LON_THRESHOLD):
                emit(
                    "error",
                    {"error": "Longitude invalide (doit être entre -180 et 180)."},
                )
                return

            latitude, longitude = lat, lon

            # ✅ 3.3.1: Utiliser LocationService pour centraliser la logique
            speed = data.get("speed")
            heading = data.get("heading")
            accuracy = data.get("accuracy")
            timestamp_str = data.get("timestamp")
            timestamp = (
                datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if timestamp_str
                else datetime.now(UTC)
            )

            snapped_lat, snapped_lon = latitude, longitude
            try:
                location_service = get_location_service()
                result = location_service.update_driver_location(
                    driver_id=driver.id,
                    latitude=latitude,
                    longitude=longitude,
                    speed=float(speed) if speed is not None else None,
                    heading=float(heading) if heading is not None else None,
                    accuracy=float(accuracy) if accuracy is not None else None,
                    source="gps",
                    timestamp=timestamp,
                )

                # Utiliser position snapée
                snapped_lat = result.snapped_lat
                snapped_lon = result.snapped_lon

                # Émettre events geofencing si détectés
                for event in result.geofence_events:
                    if event == "arrived_at_pickup":
                        emit("driver:arrived_at_pickup", {"driver_id": driver.id})
                    elif event == "arrived_at_dropoff":
                        emit("driver:arrived_at_dropoff", {"driver_id": driver.id})

                logger.info(
                    "[LocationService] Position updated: driver=%d source=%s geofence_events=%s trip_logged=%s",
                    driver.id,
                    result.source,
                    result.geofence_events,
                    result.trip_logged,
                )

            except Exception as e_loc:
                logger.exception(
                    "❌ Erreur LocationService pour driver %s: %s", driver.id, e_loc
                )
                # Fallback: utiliser position brute
                snapped_lat, snapped_lon = latitude, longitude

            # 7. Diffuser la position aux rooms de l'entreprise
            now_iso = datetime.now(UTC).isoformat()
            company_room = f"company_{company_id_val}"
            cast("Any", emit)(
                "driver_location_update",
                {
                    "driver_id": driver.id,
                    "first_name": getattr(
                        getattr(driver, "user", None), "first_name", None
                    ),
                    "latitude": snapped_lat,
                    "longitude": snapped_lon,
                    "timestamp": now_iso,
                },
                room=company_room,
            )
            logger.info(
                "📡 Loc -> %s (driver %s) %s,%s",
                company_room,
                driver.id,
                snapped_lat,
                snapped_lon,
            )

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
                "📍 driver_location_batch reçu, SID=%s, positions_count=%s",
                current_sid,
                len(data.get("positions", [])),
            )

            # ✅ SECURITY: Utiliser JWT depuis _SID_INDEX uniquement
            sid_info = _SID_INDEX.get(current_sid, {})
            user_public_id = sid_info.get("user_public_id")
            user_role = sid_info.get("role")

            if not user_public_id:
                logger.warning(
                    "⛔ driver_location_batch sans JWT public_id pour SID=%s",
                    current_sid,
                )
                emit("error", {"error": "Session JWT introuvable"})
                return

            user = User.query.filter_by(public_id=user_public_id).first()
            if not user:
                emit("error", {"error": "Utilisateur introuvable"})
                return
            user_id = user.id

            payload_driver_id = data.get("driver_id")

            # ✅ Rate limiting: vérifier après avoir récupéré user_id et driver_id
            client_ip = request.environ.get("REMOTE_ADDR", "unknown")

            # Déterminer driver_id pour rate limiting
            driver_id_for_rate_limit = payload_driver_id
            if not driver_id_for_rate_limit:
                # Fallback: chercher via user_id
                driver = Driver.query.filter_by(user_id=user_id).first()
                if driver:
                    driver_id_for_rate_limit = driver.id

            if driver_id_for_rate_limit:
                allowed, retry_after = ws_rate_limiter.check_rate_limit(
                    "driver_location_batch",
                    user_id=user_id,
                    driver_id=int(driver_id_for_rate_limit),
                    client_ip=client_ip,
                )
                if not allowed:
                    logger.warning(
                        "🚫 Rate limit driver_location_batch dépassé pour driver_id=%s, retry_after=%d",
                        driver_id_for_rate_limit,
                        retry_after or 0,
                    )
                    emit(
                        "rate_limit_exceeded",
                        {
                            "event": "rate_limit_exceeded",
                            "message": f"Trop de mises à jour batch. Réessayez dans {retry_after} secondes.",
                            "attempts": 1,
                            "retry_after_seconds": retry_after,
                        },
                    )
                    ws_metrics.on_error("rate_limit_exceeded")
                    ws_metrics.on_rate_limit_hit("driver_location_batch")
                    return

            payload_driver_id = data.get("driver_id")
            driver: Driver | None = None

            if payload_driver_id and isinstance(payload_driver_id, (int, str)):
                try:
                    candidate_id = int(payload_driver_id)
                    driver = Driver.query.get(candidate_id)
                    if driver:
                        logger.info(
                            "✅ Driver trouvé via payload: %s",
                            driver.id,
                        )
                except (ValueError, TypeError):
                    logger.warning(
                        "⚠️ driver_id non convertible: %s",
                        payload_driver_id,
                    )

            if not driver and user_role == "driver":
                driver = Driver.query.filter_by(user_id=user.id).first()
                if driver:
                    logger.info(
                        "✅ Driver trouvé via user_id: %s",
                        driver.id,
                    )

            company_id_val = tcast("int | None", getattr(driver, "company_id", None))
            if (driver is None) or (company_id_val is None):
                logger.error(
                    (
                        "❌ Driver introuvable pour driver_location_batch: "
                        "payload_driver_id=%s, user_id=%s"
                    ),
                    payload_driver_id,
                    user.id,
                )
                emit(
                    "error",
                    {"error": "Chauffeur introuvable ou non lié à une entreprise."},
                )
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

                    # ✅ 3.3.1: Utiliser LocationService pour chaque position du batch
                    speed = pos.get("speed")
                    heading = pos.get("heading")
                    accuracy = pos.get("accuracy")
                    timestamp_str = pos.get("timestamp")
                    timestamp = (
                        datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        if timestamp_str
                        else datetime.now(UTC)
                    )

                    snapped_lat, snapped_lon = latitude, longitude
                    try:
                        location_service = get_location_service()
                        result = location_service.update_driver_location(
                            driver_id=driver.id,
                            latitude=latitude,
                            longitude=longitude,
                            speed=float(speed) if speed is not None else None,
                            heading=float(heading) if heading is not None else None,
                            accuracy=float(accuracy) if accuracy is not None else None,
                            source="gps",
                            timestamp=timestamp,
                        )

                        # Utiliser position snapée
                        snapped_lat = result.snapped_lat
                        snapped_lon = result.snapped_lon

                        # Émettre events geofencing si détectés (seulement pour dernière position)
                        if pos == positions[-1]:
                            for event in result.geofence_events:
                                if event == "arrived_at_pickup":
                                    emit(
                                        "driver:arrived_at_pickup",
                                        {"driver_id": driver.id},
                                    )
                                elif event == "arrived_at_dropoff":
                                    emit(
                                        "driver:arrived_at_dropoff",
                                        {"driver_id": driver.id},
                                    )
                    except Exception as e_loc:
                        logger.debug(
                            "[LocationService] Batch position failed: %s", str(e_loc)
                        )
                        # Fallback: utiliser position brute
                        snapped_lat, snapped_lon = latitude, longitude

                    # Diffuser chaque position (snapée)
                    cast("Any", emit)(
                        "driver_location_update",
                        {
                            "driver_id": driver.id,
                            "first_name": getattr(
                                getattr(driver, "user", None), "first_name", None
                            ),
                            "latitude": snapped_lat,
                            "longitude": snapped_lon,
                            "timestamp": timestamp_str or now_iso,
                        },
                        room=company_room,
                    )
                except (TypeError, ValueError) as e:
                    logger.warning("⚠️ Position invalide dans batch: %s", e)
                    continue

            logger.info(
                "📡 Batch -> %s (driver %s) %s positions",
                company_room,
                driver.id,
                len(positions),
            )

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
                # ✅ Tracking rooms
                ws_metrics.on_room_join(room)
                emit("joined_company", {"company_id": company.id, "room": room})
                logger.info("🏢 Company %s joined room: %s", company.id, room)
            elif user_role == "driver":
                driver = Driver.query.filter_by(user_id=user.id).first()
                if not driver or not driver.company_id:
                    emit(
                        "error",
                        {"error": "Chauffeur ou entreprise associée introuvable."},
                    )
                    return

                room = f"company_{driver.company_id}"
                join_room(room)
                # ✅ Tracking rooms
                ws_metrics.on_room_join(room)
                emit("joined_company", {"company_id": driver.company_id, "room": room})
                logger.info("🚗 Driver %s joined company room: %s", driver.id, room)
            else:
                emit(
                    "error",
                    {"error": "Rôle non autorisé pour rejoindre une room entreprise."},
                )
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
                emit(
                    "error",
                    {"error": "Accès non autorisé pour la demande de localisation."},
                )
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
                    h: Mapping[bytes, Any] = {}
                    if (
                        redis_client
                    ):  # ✅ Vérification explicite pour satisfaire le linter
                        key = f"driver:{driver.id}:loc"
                        h_raw = redis_client.hgetall(key)
                        # Calme Pylance: redis-py retourne un dict[bytes, bytes]
                        h = cast("Mapping[bytes, Any]", h_raw)

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
                                "first_name": getattr(
                                    getattr(driver, "user", None), "first_name", None
                                ),
                                "latitude": loc_data.get("lat"),
                                "longitude": loc_data.get("lon"),
                                "timestamp": loc_data.get("ts")
                                or datetime.now(UTC).isoformat(),
                            },
                        )
                    elif (driver.latitude is not None) and (
                        driver.longitude is not None
                    ):
                        # Fallback to DB if Redis doesnt have data
                        cast("Any", emit)(
                            "driver_location_update",
                            {
                                "driver_id": driver.id,
                                "first_name": getattr(
                                    getattr(driver, "user", None), "first_name", None
                                ),
                                "latitude": driver.latitude,
                                "longitude": driver.longitude,
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )
                except Exception as e:
                    # driver vient du for → devrait exister,
                    # mais on défend le log quand même
                    safe_id = getattr(driver, "id", None)
                    logger.exception(
                        "❌ Error sending driver location for driver %s: %s", safe_id, e
                    )

            logger.info(
                "📡 Sent locations for %s drivers to company %s",
                len(drivers),
                company_id,
            )

        except Exception as e:
            logger.exception("❌ Error in get_driver_locations: %s", e)
            emit("error", {"error": str(e)})

    @socketio.on("disconnect")
    def handle_disconnect():
        sid = _get_sid()
        info = _SID_INDEX.pop(sid, None)
        trace_id = request.headers.get("X-Trace-ID") or request.headers.get("Trace-Id")
        now = datetime.now(UTC)

        company_id = info.get("company_id") if info else None
        user_id = info.get("user_id") if info else None
        driver_id = info.get("driver_id") if info else None
        role = info.get("role") if info else None

        # ✅ Tracking rooms : quitter les rooms appropriées
        if role == "driver" and driver_id and company_id:
            driver_room = f"driver_{driver_id}"
            company_room = f"company_{company_id}"
            ws_metrics.on_room_leave(driver_room)
            ws_metrics.on_room_leave(company_room)
        elif role == "company" and company_id:
            company_room = f"company_{company_id}"
            ws_metrics.on_room_leave(company_room)

        # ✅ Métriques
        ws_metrics.on_disconnect(company_id=company_id)

        logger.info(
            "socket_disconnect",
            extra={
                "event": "disconnect",
                "sid": sid,
                "user_id": user_id,
                "user_public_id": info.get("user_public_id") if info else None,
                "driver_id": info.get("driver_id") if info else None,
                "company_id": company_id,
                "role": info.get("role") if info else None,
                "ip": info.get("ip") if info else None,
                "timestamp": now.isoformat(),
                "request_trace_id": trace_id,
            },
        )

    @socketio.on("ping")
    def handle_ping():
        """Handler pour le heartbeat ping/pong avec logging structuré.
        Répond avec un pong contenant le timestamp actuel.
        """
        try:
            sid = _get_sid()
            sid_data = _SID_INDEX.get(sid, {})
            user_public_id = sid_data.get("user_public_id")
            role = sid_data.get("role", "unknown")
            driver_id = sid_data.get("driver_id")
            company_id = sid_data.get("company_id")
            trace_id = request.headers.get("X-Trace-ID") or request.headers.get(
                "Trace-Id"
            )
            now = datetime.now(UTC)

            logger.debug(
                "socket_heartbeat_ping",
                extra={
                    "event": "heartbeat_ping",
                    "sid": sid,
                    "user_public_id": user_public_id or "unknown",
                    "role": role,
                    "driver_id": driver_id,
                    "company_id": company_id,
                    "timestamp": now.isoformat(),
                    "request_trace_id": trace_id,
                },
            )
            emit("pong", {"timestamp": now.isoformat()})
        except Exception as e:
            logger.exception(
                "socket_heartbeat_ping_error",
                extra={
                    "event": "heartbeat_ping_error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            ws_metrics.on_error("heartbeat_ping_exception")
            # Envoyer pong même en cas d'erreur pour ne pas casser le heartbeat
            emit("pong", {"timestamp": datetime.now(UTC).isoformat()})

    @socketio.on("driver:heartbeat")
    def handle_driver_heartbeat(data):
        """Heartbeat applicatif avec métadonnées métier.

        Data:
            {
                "last_mission_id": 123,
                "location": {"lat": 46.2, "lon": 6.1},
                "timestamp": 1234567890
            }
        """
        try:
            sid = _get_sid()
            sid_data = _SID_INDEX.get(sid, {})
            driver_id = sid_data.get("driver_id")
            company_id = sid_data.get("company_id")
            user_public_id = sid_data.get("user_public_id")
            trace_id = request.headers.get("X-Trace-ID") or request.headers.get(
                "Trace-Id"
            )
            now = datetime.now(UTC)

            if not driver_id:
                logger.warning(
                    "socket_driver_heartbeat_error",
                    extra={
                        "event": "driver_heartbeat_error",
                        "reason": "driver_id_not_found",
                        "sid": sid,
                        "timestamp": now.isoformat(),
                        "request_trace_id": trace_id,
                    },
                )
                emit("error", {"error": "Driver ID introuvable"})
                ws_metrics.on_error("driver_heartbeat_no_driver_id")
                return

            # ✅ Mettre à jour last_seen_at en Redis (TTL 120s)
            if redis_client:
                try:
                    redis_client.setex(
                        f"driver:{driver_id}:last_seen",
                        120,  # TTL 2 minutes
                        now.isoformat(),
                    )

                    # Stocker métadonnées optionnelles
                    if data.get("location"):
                        redis_client.hset(
                            f"driver:{driver_id}:heartbeat",
                            mapping={
                                "last_mission_id": str(data.get("last_mission_id", "")),
                                "lat": str(data.get("location", {}).get("lat", "")),
                                "lon": str(data.get("location", {}).get("lon", "")),
                                "ts": now.isoformat(),
                            },
                        )
                        redis_client.expire(f"driver:{driver_id}:heartbeat", 120)
                except Exception as e_redis:
                    logger.warning(
                        "socket_driver_heartbeat_redis_error",
                        extra={
                            "event": "driver_heartbeat_redis_error",
                            "driver_id": driver_id,
                            "error": str(e_redis),
                            "timestamp": now.isoformat(),
                            "request_trace_id": trace_id,
                        },
                    )

            logger.debug(
                "socket_driver_heartbeat",
                extra={
                    "event": "driver_heartbeat",
                    "sid": sid,
                    "driver_id": driver_id,
                    "company_id": company_id,
                    "user_public_id": user_public_id,
                    "last_mission_id": data.get("last_mission_id"),
                    "has_location": bool(data.get("location")),
                    "timestamp": now.isoformat(),
                    "request_trace_id": trace_id,
                },
            )

            emit(
                "driver:heartbeat:ack",
                {"timestamp": now.isoformat(), "driver_id": driver_id},
            )
        except Exception as e:
            logger.exception(
                "socket_driver_heartbeat_error",
                extra={
                    "event": "driver_heartbeat_error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            ws_metrics.on_error("driver_heartbeat_exception")
            emit("error", {"error": str(e)})

    # Les handlers sont enregistrés via @socketio.on() ci-dessus
