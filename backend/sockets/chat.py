# backend/sockets/chat.py
# pyright: reportUnusedFunction=false
# Les fonctions handlers sont enregistrées via @socketio.on() et appelées
# par le framework Socket.IO, donc elles ne sont pas directement "accédées"
# dans le code Python.
"""Socket.IO handlers pour le chat et la localisation.
Les fonctions de handlers sont enregistrées via @socketio.on()
et appelées par le framework.
"""

# ruff: noqa: I001
import json
import logging
import os
import traceback
import urllib.request
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, cast
from typing import cast as tcast

import jwt.exceptions as jwt_exceptions
from flask import current_app, request, session
from flask_jwt_extended import decode_token
from flask_socketio import SocketIO, emit, join_room
from socketio.exceptions import ConnectionRefusedError as SocketConnectionRefusedError

from ext import db, redis_client
from models import Company, Driver, Message, SenderRole, User, UserRole
from schemas.socket_events import EVENT_VERSION, SocketEvent
from services.geolocation.location import get_location_service
from services.monitoring.websocket_rate_limiter import ws_rate_limiter
from services.monitoring.websocket_metrics import ws_metrics

# from services.notifications.push import send_push_message  # Unused, using fanout now
from services.security.spam import can_send_message

# Constantes pour éviter les valeurs magiques
RECEIVER_ID_ZERO = 0
LAT_THRESHOLD = 90
LON_THRESHOLD = 180
MAX_MESSAGE_LENGTH = 1000
MESSAGE_PREVIEW_LENGTH = 50
# ✅ errno pour "Bad file descriptor" - survient lors de déconnexions brutales
ERRNO_BAD_FILE_DESCRIPTOR = 9

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger("socketio")

# Petit index en mémoire pour le debug/nettoyage : sid -> infos
_SID_INDEX: Dict[str, Dict[str, Any]] = {}

# ✅ Tracking des erreurs token_expired par IP pour réduire le bruit dans les logs
# Format: {ip: (last_log_time | None, count)}
_TOKEN_EXPIRED_TRACKING: Dict[str, tuple[datetime | None, int]] = {}
_TOKEN_EXPIRED_LOG_INTERVAL = 60  # Logger au maximum toutes les 60 secondes par IP
_TOKEN_EXPIRED_MAX_COUNT = 5  # Après 5 erreurs, logger seulement toutes les 60s
_TOKEN_EXPIRED_TRACKING_MAX_SIZE = 1000  # Taille max du dictionnaire avant nettoyage


def _log_socketio_exception(
    exception: Exception,
    event_name: str,
    sid: str | None = None,
    user_id: int | None = None,
    company_id: int | None = None,
    driver_id: int | None = None,
    additional_context: Dict[str, Any] | None = None,
) -> None:
    """Helper pour logger les exceptions Socket.IO avec contexte complet.

    Args:
        exception: L'exception à logger
        event_name: Nom de l'événement Socket.IO (ex: "team_chat_message")
        sid: Socket ID (optionnel)
        user_id: ID utilisateur (optionnel)
        company_id: ID entreprise (optionnel)
        driver_id: ID chauffeur (optionnel)
        additional_context: Contexte additionnel à logger (optionnel)
    """
    # Récupérer le SID si non fourni
    if sid is None:
        sid = _get_sid()

    # Récupérer les infos depuis _SID_INDEX si disponibles
    sid_data = _SID_INDEX.get(sid, {}) if sid else {}
    if user_id is None:
        user_id = sid_data.get("user_id")
    if company_id is None:
        company_id = sid_data.get("company_id")
    if driver_id is None:
        driver_id = sid_data.get("driver_id")

    # Récupérer trace_id depuis headers
    trace_id = None
    with suppress(Exception):
        trace_id = request.headers.get("X-Trace-ID") or request.headers.get("Trace-Id")

    # Récupérer IP client
    client_ip = "unknown"
    with suppress(Exception):
        client_ip = request.environ.get("REMOTE_ADDR", "unknown")

    # Construire le contexte de logging
    log_context = {
        "event": f"{event_name}_error",
        "error": str(exception),
        "error_type": type(exception).__name__,
        "sid": sid or "unknown",
        "user_id": user_id,
        "company_id": company_id,
        "driver_id": driver_id,
        "ip": client_ip,
        "timestamp": datetime.now(UTC).isoformat(),
        "request_trace_id": trace_id,
    }

    # Ajouter contexte additionnel
    if additional_context:
        log_context.update(additional_context)

    # Logger avec stack trace
    logger.exception(
        "[Socket.IO] ❌ Exception dans %s",
        event_name,
        extra=log_context,
    )

    # Enregistrer métrique d'erreur
    ws_metrics.on_error(f"{event_name}_exception")


# Les handlers Socket.IO sont enregistrés par @socketio.on()
# Note: Rate limiting géré par WebSocketRateLimiter (backend/services/websocket_rate_limiter.py)


def _parse_timestamp(timestamp_value: Any) -> datetime:
    """Parse un timestamp qui peut être soit une chaîne ISO, soit un entier Unix (millisecondes).

    Args:
        timestamp_value: Timestamp sous forme de chaîne ISO (ex: "2025-12-07T12:34:56.789Z")
                        ou entier Unix en millisecondes (ex: 1701950096789)

    Returns:
        datetime: Objet datetime avec timezone UTC
    """
    if not timestamp_value:
        return datetime.now(UTC)

    # Si c'est un entier (Unix timestamp en millisecondes)
    if isinstance(timestamp_value, (int, float)):
        # Convertir millisecondes en secondes pour fromtimestamp
        timestamp_seconds = float(timestamp_value) / 1000.0
        return datetime.fromtimestamp(timestamp_seconds, tz=UTC)

    # Si c'est une chaîne, traiter comme ISO
    if isinstance(timestamp_value, str):
        # Remplacer "Z" par "+00:00" pour compatibilité avec fromisoformat
        timestamp_str = timestamp_value.replace("Z", "+00:00")
        return datetime.fromisoformat(timestamp_str)

    # Fallback: utiliser maintenant
    return datetime.now(UTC)


def _get_sid(fallback_request=None) -> str:
    """Récupère le SID de la requête Socket.IO actuelle."""
    if fallback_request is None:
        fallback_request = request

    sid = getattr(fallback_request, "sid", None) or fallback_request.environ.get(
        "socketio.sid"
    )
    return str(sid) if sid is not None else ""


def _enrich_payload_if_needed(
    payload: Dict[str, Any], event_name: str
) -> Dict[str, Any]:
    """Enrichit un payload avec event_id, version, timestamp si absents.

    Utilise le schéma centralisé SocketEvent pour garantir la cohérence.

    Args:
        payload: Payload d'événement (peut déjà contenir event_id)
        event_name: Nom de l'événement Socket.IO (ex: "team_chat_message")

    Returns:
        Payload enrichi (nouveau dict si enrichissement nécessaire, sinon original)
    """
    # Si event_id déjà présent, ne pas enrichir (évite doublon)
    if "event_id" in payload:
        return payload

    # ✅ Utiliser le schéma centralisé SocketEvent pour enrichir
    return SocketEvent.create(
        event_type=event_name, payload=payload, version=EVENT_VERSION
    )


def _extract_token(auth) -> str | None:
    """Récupère le token JWT depuis 3 sources dans l'ordre de priorité suivant :

    ✅ S1: Support query string supprimé pour éviter fuite de tokens dans logs/URLs.
    ✅ PROD: En production, rejeter payload auth si header/cookie présents (sécurité).

    **Ordre de priorité :**
    1. **Header Authorization** : `Authorization: Bearer <token>` (priorité la plus élevée)
    2. **Cookie access_token** : Cookie httpOnly (✅ Migration localStorage → cookies)
    3. **Payload auth** : `auth.token` ou `auth.accessToken` (envoyé par le client Socket.IO)

    **Justification de l'ordre :**
    - Header Authorization : Méthode standard HTTP, recommandée pour la sécurité
    - Cookie access_token : ✅ Migration localStorage → cookies httpOnly (priorité après header)
    - Payload auth : Supporte les clients Socket.IO qui envoient le token dans le payload de connexion

    **Comportement en production :**
    - Si header Authorization présent : ignorer cookie/payload (priorité stricte)
    - Payload auth : rejeté en production (sécurité)

    Args:
        auth: Payload d'authentification Socket.IO (dict ou None)

    Returns:
        Token JWT (str) si trouvé, None sinon
    """
    token_result: str | None = None

    # 1) Header Authorization: Bearer ... (PRIORITÉ 1 - Méthode standard HTTP)
    authz = request.headers.get("Authorization") or request.headers.get("AUTHORIZATION")
    if authz and authz.lower().startswith("bearer "):
        token = authz.split(" ", 1)[1].strip()
        # ✅ PROD: En production, ignorer cookie/payload si header présent
        try:
            is_prod = current_app.config.get("ENV") == "production"
        except RuntimeError:
            is_prod = False
        if is_prod and token:
            logger.debug(
                "socket_token_extracted",
                extra={
                    "event": "token_extracted",
                    "source": "header_authorization",
                    "has_token": bool(token),
                    "ignored_other_sources": True,
                },
            )
            token_result = token
        else:
            logger.debug(
                "socket_token_extracted",
                extra={
                    "event": "token_extracted",
                    "source": "header_authorization",
                    "has_token": bool(token),
                },
            )
            token_result = token

    # 2) ✅ Cookie access_token (PRIORITÉ 2 - Migration localStorage → cookies httpOnly)
    try:
        cookie_name = current_app.config.get("COOKIE_ACCESS_TOKEN_NAME", "access_token")
    except RuntimeError:
        # current_app n'est pas disponible dans ce contexte
        cookie_name = "access_token"
        logger.debug(
            "socket_token_current_app_unavailable",
            extra={
                "event": "current_app_unavailable",
                "using_fallback": "access_token",
            },
        )

    try:
        all_cookies = list(request.cookies.keys()) if request.cookies else []
        cookie_token = request.cookies.get(cookie_name)
        # #region agent log
        try:
            log_data_cookie_check = {
                "location": "chat.py:_extract_token",
                "message": "Cookie check",
                "data": {
                    "cookie_name": cookie_name,
                    "has_cookie": bool(cookie_token),
                    "all_cookies": all_cookies,
                    "has_cookies": bool(request.cookies),
                },
                "timestamp": int(datetime.now(UTC).timestamp() * 1000),
                "sessionId": "debug-session",
                "runId": "post-fix",
                "hypothesisId": "G",
            }
            req = urllib.request.Request(
                "http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74",
                data=json.dumps(log_data_cookie_check).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=0.1)
        except Exception as log_err:
            # Ne pas faire échouer l'extraction du token si le log échoue
            logger.debug("Failed to send debug log: %s", log_err)
        # #endregion
        if cookie_token:
            token = cookie_token.strip()
            logger.info(
                "socket_token_extracted",
                extra={
                    "event": "token_extracted",
                    "source": "cookie",
                    "has_token": bool(token),
                    "cookie_name": cookie_name,
                },
            )
            token_result = token
        # ✅ Log pour debug : vérifier si les cookies sont présents
        logger.debug(
            "socket_token_cookie_not_found",
            extra={
                "event": "cookie_not_found",
                "cookie_name": cookie_name,
                "all_cookies": all_cookies,
                "has_cookies": bool(request.cookies),
            },
        )
    except Exception as e:
        # ✅ Gérer toutes les exceptions pour éviter "server error"
        logger.error(
            "socket_token_cookie_error",
            extra={
                "event": "cookie_error",
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        # ✅ Fallback : essayer directement avec le nom du cookie
        try:
            cookie_token = request.cookies.get("access_token")
            if cookie_token:
                token = cookie_token.strip()
                logger.info(
                    "socket_token_extracted",
                    extra={
                        "event": "token_extracted",
                        "source": "cookie_fallback",
                        "has_token": bool(token),
                    },
                )
                token_result = token
        except Exception:
            pass

    # 3) Payload auth envoyé par le client Socket.IO (PRIORITÉ 3 - Uniquement si header/cookie absents)
    if not token_result and isinstance(auth, dict):
        tok = auth.get("token") or auth.get("accessToken")
        if tok:
            # ⚠️ Toujours logger un warning si payload auth utilisé (sécurité)
            try:
                is_prod = current_app.config.get("ENV") == "production"
                env = current_app.config.get("ENV", "unknown")
            except RuntimeError:
                is_prod = False
                env = "unknown"

            logger.warning(
                "socket_token_payload_auth_used",
                extra={
                    "event": "payload_auth_detected",
                    "has_header": bool(authz),
                    "has_cookie": bool(request.cookies.get("access_token")),
                    "env": env,
                    "is_production": is_prod,
                },
            )

            # Rejeter en production, accepter en dev uniquement
            if is_prod:
                logger.warning(
                    "socket_token_payload_auth_rejected",
                    extra={
                        "event": "payload_auth_rejected",
                        "reason": "production_security",
                        "env": env,
                    },
                )
                token_result = None
            else:
                # Dev: autoriser payload auth (mais avec warning)
                token = str(tok).strip()
                logger.debug(
                    "socket_token_extracted",
                    extra={
                        "event": "token_extracted",
                        "source": "auth_payload",
                        "has_token": bool(token),
                        "env": env,
                    },
                )
                token_result = token

    # ✅ S1: Support query string supprimé pour éviter fuite de tokens dans logs/URLs
    # Les clients doivent utiliser Header Authorization (mobile) ou Cookie (web)

    # Aucun token trouvé
    if not token_result:
        logger.debug(
            "socket_token_not_found",
            extra={
                "event": "token_not_found",
                "has_auth": isinstance(auth, dict),
                "has_authz_header": bool(authz),
            },
        )

    return token_result


def init_chat_socket(socketio: SocketIO):
    logger.info("🔧 [INIT] Initialisation des handlers Socket.IO chat")
    # #region agent log - DÉSACTIVÉ EN PRODUCTION
    if os.getenv("FLASK_ENV") == "development":
        try:
            debug_log_path = Path(".cursor/debug.log")
            debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with debug_log_path.open("a") as f:
                # json est déjà importé en haut du fichier
                f.write(
                    json.dumps(
                        {
                            "location": "chat.py:init_chat_socket",
                            "message": "init_chat_socket CALLED",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H3",
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass  # Ignore les erreurs de log en dev
    # #endregion

    @socketio.on("connect", namespace="/")
    def handle_connect(auth: dict[str, Any] | None) -> bool:
        # #region agent log - DÉSACTIVÉ EN PRODUCTION
        if os.getenv("FLASK_ENV") == "development":
            try:
                debug_log_path = Path(".cursor/debug.log")
                with debug_log_path.open("a") as f:
                    # json est déjà importé en haut du fichier
                    f.write(
                        json.dumps(
                            {
                                "location": "chat.py:handle_connect:ENTRY",
                                "message": "CONNECT HANDLER INVOKED",
                                "data": {"auth": str(auth)[:200] if auth else None},
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H3",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass  # Ignore les erreurs de log en dev
        # #endregion
        # ✅ Try-except global pour capturer TOUTES les exceptions, y compris celles du rate limiting
        try:
            client_ip = request.environ.get("REMOTE_ADDR", "unknown")
            ua = request.headers.get("User-Agent", "Unknown")
            trace_id = request.headers.get("X-Trace-ID") or request.headers.get(
                "Trace-Id"
            )
            now = datetime.now(UTC)
            # R1: device_id + session_diag depuis socket.auth pour corrélation (connect/disconnect)
            auth_dict = auth if isinstance(auth, dict) else {}
            device_id = auth_dict.get("device_id")
            session_diag = auth_dict.get("session_diag")

            # ✅ Log origin pour debug CORS
            origin = request.headers.get("Origin") or request.headers.get("ORIGIN")
            # ✅ Log explicite pour diagnostic (toujours affiché)
            print(
                f"🔌 [SOCKET.IO] handle_connect appelé - IP: {client_ip}, Origin: {origin}"
            )
            # #region agent log - DÉSACTIVÉ EN PRODUCTION
            if os.getenv("FLASK_ENV") == "development":
                try:
                    sid = getattr(request, "sid", None) or request.environ.get(
                        "socketio.sid", "unknown"
                    )
                    debug_log_path = os.getenv(
                        "DEBUG_LOG_PATH", r"c:\Users\jasiq\atmr\.cursor\debug.log"
                    )
                    log_path = Path(debug_log_path)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a", encoding="utf-8") as f:
                        log_data = {
                            "id": f"log_{int(datetime.now(UTC).timestamp() * 1000)}_connect",
                            "timestamp": int(datetime.now(UTC).timestamp() * 1000),
                            "location": "chat.py:handle_connect",
                            "message": "Socket.IO connect event",
                            "data": {
                                "sid": str(sid),
                                "worker_pid": os.getpid(),
                                "client_ip": client_ip,
                                "origin": origin or "none",
                            },
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "B",
                        }
                        f.write(json.dumps(log_data) + "\n")
                except Exception:
                    pass
            # #endregion
            logger.info(
                "socket_connect_attempt",
                extra={
                    "event": "connect_attempt",
                    "ip": client_ip,
                    "origin": origin,
                    "user_agent": ua,
                    "timestamp": now.isoformat(),
                    "request_trace_id": trace_id,
                },
            )

            # ✅ Rate limiting : utiliser WebSocketRateLimiter (remplace ancien système)
            allowed, retry_after = ws_rate_limiter.check_rate_limit(
                "connect",
                client_ip=client_ip,
            )
            if not allowed:
                # #region agent log
                log_data_rate_limit = {
                    "location": "chat.py:handle_connect",
                    "message": "Rate limit exceeded",
                    "data": {
                        "reason": "rate_limit_exceeded",
                        "ip": client_ip,
                        "retry_after": retry_after or 0,
                    },
                    "timestamp": int(now.timestamp() * 1000),
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "E",
                }
                try:
                    req = urllib.request.Request(
                        "http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74",
                        data=json.dumps(log_data_rate_limit).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=0.1)
                except Exception:
                    pass
                # #endregion
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
                ws_metrics.on_error("rate_limit_exceeded")
                ws_metrics.on_rate_limit_hit("connect")
                raise SocketConnectionRefusedError(
                    "RATE_LIMIT", {"retry_after": retry_after or 60}
                )

            # ✅ Logs conditionnels (désactiver en production)
            is_prod = current_app.config.get("ENV") == "production"
            if not is_prod:
                print("🔌 [SOCKET.IO] ✅✅✅ HANDLE_CONNECT APPELÉ ! ✅✅✅")
                print(f"🔌 [SOCKET.IO] Tentative de connexion depuis {client_ip}")
                print(f"🔌 [SOCKET.IO] User-Agent: {ua}")

            # ✅ Logger structuré (toujours activé mais niveau INFO en prod)
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
            # ✅ Log pour debug : vérifier les cookies reçus (uniquement en dev)
            all_cookies = list(request.cookies.keys()) if request.cookies else []
            if not is_prod:
                print(f"🔌 [SOCKET.IO] Cookies reçus: {all_cookies}")
                print(f"🔌 [SOCKET.IO] Has cookies: {bool(request.cookies)}")
                print(f"🔌 [SOCKET.IO] Has auth payload: {isinstance(auth, dict)}")
                print(
                    f"🔌 [SOCKET.IO] Has Authorization header: {bool(request.headers.get('Authorization') or request.headers.get('AUTHORIZATION'))}"
                )
            logger.info(
                "socket_connect_debug",
                extra={
                    "event": "connect_debug",
                    "ip": client_ip,
                    "has_cookies": bool(request.cookies),
                    # ✅ Ne pas logger les cookies/tokens en production
                    "cookie_keys": all_cookies if not is_prod else None,
                    "has_auth": isinstance(auth, dict),
                    "auth_keys": list(auth.keys())
                    if isinstance(auth, dict) and not is_prod
                    else [],
                    "has_authz_header": bool(
                        request.headers.get("Authorization")
                        or request.headers.get("AUTHORIZATION")
                    ),
                    "timestamp": now.isoformat(),
                    "request_trace_id": trace_id,
                },
            )

            token = _extract_token(auth)
            # ✅ Log explicite pour diagnostic (toujours affiché)
            print(
                f"🔌 [SOCKET.IO] Token extrait: {bool(token)}, Cookies: {bool(request.cookies)}, Auth payload: {isinstance(auth, dict)}"
            )
            # #region agent log
            log_data = {
                "location": "chat.py:handle_connect",
                "message": "Token extraction result",
                "data": {
                    "has_token": bool(token),
                    "token_length": len(token) if token else 0,
                    "has_cookies": bool(request.cookies),
                    "cookie_keys": all_cookies,
                    "has_auth_payload": isinstance(auth, dict),
                    "auth_keys": list(auth.keys()) if isinstance(auth, dict) else [],
                    "auth_token_key": "token" in auth
                    if isinstance(auth, dict)
                    else False,
                    "auth_accessToken_key": "accessToken" in auth
                    if isinstance(auth, dict)
                    else False,
                    "has_authz_header": bool(
                        request.headers.get("Authorization")
                        or request.headers.get("AUTHORIZATION")
                    ),
                },
                "timestamp": int(now.timestamp() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A",
            }
            try:
                req = urllib.request.Request(
                    "http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74",
                    data=json.dumps(log_data).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=0.1)
            except Exception:
                pass
            # #endregion
            if not token:
                # ✅ Log explicite pour diagnostic (toujours affiché)
                print("❌ [SOCKET.IO] Token manquant - Connexion refusée")
                print(f"   - Cookies présents: {bool(request.cookies)}")
                print(f"   - Clés cookies: {all_cookies}")
                print(f"   - Auth payload: {isinstance(auth, dict)}")
                print(
                    f"   - Auth keys: {list(auth.keys()) if isinstance(auth, dict) else []}"
                )
                print(
                    f"   - Authorization header: {bool(request.headers.get('Authorization') or request.headers.get('AUTHORIZATION'))}"
                )
                # #region agent log
                log_data_error = {
                    "location": "chat.py:handle_connect",
                    "message": "Token missing - connection refused",
                    "data": {
                        "reason": "token_missing",
                        "has_cookies": bool(request.cookies),
                        "cookie_keys": all_cookies,
                        "ip": client_ip,
                    },
                    "timestamp": int(now.timestamp() * 1000),
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                }
                try:
                    req = urllib.request.Request(
                        "http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74",
                        data=json.dumps(log_data_error).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=0.1)
                except Exception:
                    pass
                # #endregion
                logger.warning(
                    "socket_connect_refused",
                    extra={
                        "event": "connect_refused",
                        "reason": "token_missing",
                        "ip": client_ip,
                        "has_cookies": bool(request.cookies),
                        "cookie_keys": all_cookies,
                        "has_authz_header": bool(
                            request.headers.get("Authorization")
                            or request.headers.get("AUTHORIZATION")
                        ),
                        "has_auth_payload": isinstance(auth, dict),
                        "auth_keys": list(auth.keys())
                        if isinstance(auth, dict)
                        else [],
                        "timestamp": now.isoformat(),
                        "request_trace_id": trace_id,
                    },
                )
                ws_metrics.on_error("token_missing")
                raise SocketConnectionRefusedError("AUTH_REQUIRED")

            # ✅ Log conditionnel (uniquement en dev)
            try:
                is_prod = current_app.config.get("ENV") == "production"
            except RuntimeError:
                is_prod = False
            if not is_prod:
                print("✅ [SOCKET.IO] Token extrait avec succès")

            # Vérifie & décode (lève si invalide/expiré).
            # ✅ Robustesse: si plusieurs sources de token existent (header + auth payload),
            # essayer un fallback si la signature du premier token échoue.
            try:
                auth_token: str | None = None
                auth_access_token: str | None = None
                if isinstance(auth, dict):
                    with suppress(Exception):
                        v = auth.get("token")
                        auth_token = str(v).strip() if v else None
                    with suppress(Exception):
                        v = auth.get("accessToken")
                        auth_access_token = str(v).strip() if v else None

                candidate_tokens: list[str] = []
                if token:
                    candidate_tokens.append(token)
                if auth_token and auth_token not in candidate_tokens:
                    candidate_tokens.append(auth_token)
                if auth_access_token and auth_access_token not in candidate_tokens:
                    candidate_tokens.append(auth_access_token)

                decoded: dict[str, Any] | None = None
                last_decode_error: Exception | None = None
                for idx, candidate in enumerate(candidate_tokens):
                    try:
                        decoded = decode_token(candidate)
                        if idx > 0 and not is_prod:
                            print(
                                f"🔌 [SOCKET.IO] Token fallback utilisé (idx={idx}) après échec du premier token"
                            )
                        break
                    except jwt_exceptions.InvalidSignatureError as e:
                        last_decode_error = e
                        # ✅ Support legacy keys JWT (rotation) si la signature échoue
                        with suppress(Exception):
                            from security.jwt_legacy_keys import (  # local import (évite cycles)
                                try_decode_with_legacy_keys,
                            )

                            alg = current_app.config.get("JWT_ALGORITHM", "HS256")
                            legacy_payload, _ = try_decode_with_legacy_keys(
                                candidate, algorithms=[str(alg)]
                            )
                            if legacy_payload:
                                decoded = legacy_payload
                                if not is_prod:
                                    print(
                                        "🔌 [SOCKET.IO] Token accepté via legacy key (rotation)"
                                    )
                                break
                        continue
                    except Exception as e:
                        last_decode_error = e
                        break

                if decoded is None:
                    raise last_decode_error or Exception("Token decode error")

                # #region agent log
                log_data_decoded = {
                    "location": "chat.py:handle_connect",
                    "message": "Token decoded successfully",
                    "data": {
                        "has_decoded": bool(decoded),
                        "has_sub": "sub" in decoded if decoded else False,
                        "public_id": decoded.get("sub") if decoded else None,
                        "role": decoded.get("role") if decoded else None,
                    },
                    "timestamp": int(now.timestamp() * 1000),
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "D",
                }
                try:
                    req = urllib.request.Request(
                        "http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74",
                        data=json.dumps(log_data_decoded).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=0.1)
                except Exception:
                    pass
                # #endregion
            except jwt_exceptions.ExpiredSignatureError:
                # ✅ Réduire le bruit dans les logs : tracker les erreurs token_expired par IP
                # Logger seulement si c'est la première erreur ou si ça fait plus de 60s depuis le dernier log
                should_log = True
                last_log_time, error_count = _TOKEN_EXPIRED_TRACKING.get(
                    client_ip, (None, 0)
                )

                if last_log_time is not None:
                    time_since_last_log = (now - last_log_time).total_seconds()
                    if error_count < _TOKEN_EXPIRED_MAX_COUNT:
                        # Les premières erreurs sont toujours loggées
                        should_log = True
                    elif time_since_last_log < _TOKEN_EXPIRED_LOG_INTERVAL:
                        # Trop d'erreurs récentes, ne pas logger (réduire bruit)
                        should_log = False
                    else:
                        # Assez de temps écoulé, logger à nouveau
                        should_log = True

                # Mettre à jour le tracking
                if should_log:
                    _TOKEN_EXPIRED_TRACKING[client_ip] = (now, error_count + 1)
                else:
                    _TOKEN_EXPIRED_TRACKING[client_ip] = (
                        last_log_time,
                        error_count + 1,
                    )

                # Nettoyer les entrées anciennes (> 5 minutes) pour éviter fuite mémoire
                if len(_TOKEN_EXPIRED_TRACKING) > _TOKEN_EXPIRED_TRACKING_MAX_SIZE:
                    # Nettoyer les entrées plus anciennes que 5 minutes
                    cutoff_time = now - timedelta(minutes=5)
                    ips_to_remove = [
                        ip
                        for ip, (last_time, _) in _TOKEN_EXPIRED_TRACKING.items()
                        if last_time is not None and last_time < cutoff_time
                    ]
                    for ip in ips_to_remove:
                        _TOKEN_EXPIRED_TRACKING.pop(ip, None)

                # #region agent log (uniquement si should_log)
                if should_log:
                    log_data_expired = {
                        "location": "chat.py:handle_connect",
                        "message": "Token expired",
                        "data": {
                            "reason": "token_expired",
                            "ip": client_ip,
                            "error_count": error_count + 1,
                        },
                        "timestamp": int(now.timestamp() * 1000),
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "B",
                    }
                    try:
                        req = urllib.request.Request(
                            "http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74",
                            data=json.dumps(log_data_expired).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        urllib.request.urlopen(req, timeout=0.1)
                    except Exception:
                        pass
                # #endregion

                # ✅ Logger seulement si should_log (réduire bruit)
                if should_log:
                    log_level = (
                        logger.debug
                        if error_count >= _TOKEN_EXPIRED_MAX_COUNT
                        else logger.info
                    )
                    log_level(
                        "socket_connect_error",
                        extra={
                            "event": "connect_error",
                            "reason": "token_expired",
                            "ip": client_ip,
                            "error_count": error_count + 1,
                            "timestamp": now.isoformat(),
                            "request_trace_id": trace_id,
                        },
                    )

                ws_metrics.on_error("token_expired")
                raise SocketConnectionRefusedError("TOKEN_EXPIRED", {"retry_after": 5}) from None
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
                ws_metrics.on_error("token_invalid_audience")
                raise SocketConnectionRefusedError("AUTH_INVALID") from None
            except Exception as e:
                # ✅ Dev-only: afficher la cause exacte (sinon "Token invalide." est trop vague)
                with suppress(Exception):
                    try:
                        is_prod = current_app.config.get("ENV") == "production"
                    except RuntimeError:
                        is_prod = False
                    if not is_prod:
                        print(
                            f"🔌 [SOCKET.IO] Token decode error: {type(e).__name__}: {e}"
                        )
                # #region agent log
                log_data_decode_error = {
                    "location": "chat.py:handle_connect",
                    "message": "Token decode error",
                    "data": {
                        "reason": "token_decode_error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "ip": client_ip,
                    },
                    "timestamp": int(now.timestamp() * 1000),
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                }
                try:
                    req = urllib.request.Request(
                        "http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74",
                        data=json.dumps(log_data_decode_error).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=0.1)
                except Exception:
                    pass
                # #endregion
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
                ws_metrics.on_error("token_decode_error")
                raise SocketConnectionRefusedError("AUTH_INVALID") from e

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
                ws_metrics.on_error("token_invalid")
                raise SocketConnectionRefusedError("AUTH_INVALID")

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
                # #region agent log
                log_data_user_not_found = {
                    "location": "chat.py:handle_connect",
                    "message": "User not found",
                    "data": {
                        "reason": "user_not_found",
                        "user_public_id": public_id,
                        "ip": client_ip,
                    },
                    "timestamp": int(now.timestamp() * 1000),
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "C",
                }
                try:
                    req = urllib.request.Request(
                        "http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74",
                        data=json.dumps(log_data_user_not_found).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=0.1)
                except Exception:
                    pass
                # #endregion
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
                ws_metrics.on_error("user_not_found")
                raise SocketConnectionRefusedError("AUTH_INVALID")

            # Stash session minimale
            session["user_id"] = user.id
            session["first_name"] = user.first_name
            session["role"] = user.role.value.lower()

            sid = _get_sid()
            trace_id = trace_id or f"socket-{sid[:8]}"

            if user.role == UserRole.driver:
                driver = Driver.query.filter_by(user_id=user.id).first()
                if not driver or not driver.company_id:
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
                    raise SocketConnectionRefusedError("DRIVER_OR_COMPANY_NOT_FOUND")

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
                    "device_id": device_id,
                    "session_diag": session_diag,
                }

                # ✅ Métriques
                ws_metrics.on_connect(company_id=driver.company_id, user_id=user.id)
                # ✅ Tracking rooms
                ws_metrics.on_room_join(company_room)
                ws_metrics.on_room_join(driver_room)

                logger.info(
                    "socket_connect_success",
                    extra={
                        "event": "socket_connect_success",
                        "sid": sid,
                        "user_id": user.id,
                        "user_public_id": public_id,
                        "driver_id": driver.id,
                        "company_id": driver.company_id,
                        "role": "driver",
                        "rooms": [company_room, driver_room],
                        "ip": client_ip,
                        "device_id": device_id,
                        "session_diag": session_diag,
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
                    ws_metrics.on_error("company_not_found")
                    raise SocketConnectionRefusedError("COMPANY_NOT_FOUND")

                room = f"company_{company.id}"
                join_room(room)
                emit("connected", {"message": f"✅ Entreprise connectée à {room}"})

                _SID_INDEX[sid] = {
                    "user_public_id": public_id,
                    "user_id": user.id,
                    "company_id": company.id,
                    "ip": client_ip,
                    "role": "company",
                    "device_id": device_id,
                    "session_diag": session_diag,
                }

                # ✅ Métriques
                ws_metrics.on_connect(company_id=company.id, user_id=user.id)
                # ✅ Tracking rooms
                ws_metrics.on_room_join(room)

                logger.info(
                    "socket_connect_success",
                    extra={
                        "event": "socket_connect_success",
                        "sid": sid,
                        "user_id": user.id,
                        "user_public_id": public_id,
                        "company_id": company.id,
                        "role": "company",
                        "rooms": [room],
                        "ip": client_ip,
                        "device_id": device_id,
                        "session_diag": session_diag,
                        "timestamp": now.isoformat(),
                        "request_trace_id": trace_id,
                    },
                )
            else:
                # #region agent log
                log_data_role_not_authorized = {
                    "location": "chat.py:handle_connect",
                    "message": "Role not authorized",
                    "data": {
                        "reason": "role_not_authorized",
                        "user_id": user.id,
                        "role": user.role.value,
                        "ip": client_ip,
                    },
                    "timestamp": int(now.timestamp() * 1000),
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "D",
                }
                try:
                    req = urllib.request.Request(
                        "http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74",
                        data=json.dumps(log_data_role_not_authorized).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=0.1)
                except Exception:
                    pass
                # #endregion
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
                ws_metrics.on_error("role_not_authorized")
                raise SocketConnectionRefusedError("AUTH_FORBIDDEN")

            return True

        except SocketConnectionRefusedError:
            raise
        except Exception as e:
            # ✅ Gérer toutes les exceptions pour éviter "server error"
            # ✅ Récupérer les variables qui peuvent ne pas être définies si l'exception est levée tôt
            client_ip = (
                request.environ.get("REMOTE_ADDR", "unknown")
                if hasattr(request, "environ")
                else "unknown"
            )
            trace_id = None
            with suppress(Exception):
                trace_id = request.headers.get("X-Trace-ID") or request.headers.get(
                    "Trace-Id"
                )

            # #region agent log
            log_data_exception = {
                "location": "chat.py:handle_connect",
                "message": "Exception in handle_connect",
                "data": {
                    "reason": "exception",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "ip": client_ip,
                },
                "timestamp": int(datetime.now(UTC).timestamp() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "F",
            }
            try:
                req = urllib.request.Request(
                    "http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74",
                    data=json.dumps(log_data_exception).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=0.1)
            except Exception:
                pass
            # #endregion

            # ✅ Logs conditionnels (désactiver en production)
            try:
                is_prod = current_app.config.get("ENV") == "production"
            except RuntimeError:
                is_prod = False
            if not is_prod:
                print(
                    f"❌ [SOCKET.IO] EXCEPTION dans handle_connect: {type(e).__name__}: {e!s}"
                )
                print("❌ [SOCKET.IO] Traceback:")
                traceback.print_exc()
            # ✅ logger.exception reste actif (géré par le système de logging)
            logger.exception(
                "socket_connect_error",
                extra={
                    "event": "connect_error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "ip": client_ip,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "request_trace_id": trace_id,
                },
            )
            ws_metrics.on_error("connect_exception")
            raise SocketConnectionRefusedError("CONNECT_ERROR") from e

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
                    "❌ [CHAT] Limite: 1 fichier par message (image OU PDF, pas les deux)"
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

            # ✅ Enrichir payload avec event_id, version, timestamp
            enriched_payload = _enrich_payload_if_needed(payload, "team_chat_message")

            room = f"company_{company_id}"
            # Pylance ne déclare pas kwarg 'room' sur emit -> cast en Any
            cast("Any", emit)("team_chat_message", enriched_payload, room=room)
            logger.info(
                "📨 Message émis dans %s par %s : %s", room, sender_role, content
            )

            # ✅ Si un receiver_id (driver) est fourni, notifier aussi sa room dédiée
            # Note: Utiliser le même payload enrichi (même event_id pour les deux rooms)
            if receiver_id:
                driver_room = f"driver_{receiver_id}"
                cast("Any", emit)(
                    "team_chat_message", enriched_payload, room=driver_room
                )
                logger.info("📨 Message relayé vers %s", driver_room)

            # ✅ P0: Push notification pour message.new (fan-out hybride)
            # Utiliser le système de fan-out centralisé pour cohérence
            if receiver_id:
                try:
                    receiver_user = User.query.get(receiver_id)
                    if receiver_user and receiver_user.role == UserRole.driver:
                        driver = Driver.query.filter_by(user_id=receiver_id).first()
                        if driver:
                            # Préparer le preview du message
                            # ✅ Notification pro: identifier clairement l'émetteur (Entreprise / Chauffeur)
                            sender_label = "Entreprise" if user.role == UserRole.company else "Chauffeur"
                            sender_display = sender_label
                            try:
                                # Entreprise: utiliser le nom de l'entreprise si dispo
                                if user.role == UserRole.company and company_obj:
                                    company_name = getattr(company_obj, "name", None)
                                    if company_name:
                                        sender_display = f"{sender_label} {company_name}"
                                else:
                                    first_name = getattr(user, "first_name", None) or None
                                    if first_name:
                                        sender_display = f"{sender_label} {first_name}"
                            except Exception:
                                sender_display = sender_label
                            message_preview = (
                                content[:MESSAGE_PREVIEW_LENGTH]
                                if content
                                else "Nouveau message"
                            )
                            if content and len(content) > MESSAGE_PREVIEW_LENGTH:
                                message_preview += "..."

                            # ✅ Utiliser fanout_message_new pour fan-out centralisé
                            from services.events.fanout import fanout_message_new

                            fanout_message_new(
                                driver_id=driver.id,
                                message_id=message.id,
                                sender_name=sender_display,
                                message_preview=message_preview,
                                company_id=company_id,
                            )
                            logger.info(
                                "[chat] Push notification queued for driver %s (message %s)",
                                driver.id,
                                message.id,
                            )
                except (ValueError, TypeError, AttributeError) as e:
                    logger.error(
                        "[chat] Push notification failed (validation error: %s): %s",
                        type(e).__name__,
                        e,
                    )
                except (ConnectionError, OSError) as e:
                    logger.error(
                        "[chat] Push notification failed (network error: %s): %s",
                        type(e).__name__,
                        e,
                    )
                except Exception:
                    logger.exception("[chat] Push notification failed")

        except Exception as e:
            # ✅ 18. Améliorer gestion erreurs Socket.IO : Logger avec contexte complet
            try:
                sid = _get_sid()
                sid_data = _SID_INDEX.get(sid, {})
            except Exception:
                sid = None
                sid_data = {}
            _log_socketio_exception(
                exception=e,
                event_name="team_chat_message",
                sid=sid,
                user_id=sid_data.get("user_id"),
                company_id=sid_data.get("company_id"),
                driver_id=sid_data.get("driver_id"),
                additional_context={
                    "local_id": local_id,
                    "has_content": bool(
                        data.get("content") if isinstance(data, dict) else False
                    ),
                    "has_image": bool(
                        data.get("image_url") or data.get("image")
                        if isinstance(data, dict)
                        else False
                    ),
                    "has_pdf": bool(
                        data.get("pdf_url") or data.get("pdf")
                        if isinstance(data, dict)
                        else False
                    ),
                },
            )
            # Notifier l'utilisateur si possible
            with suppress(Exception):
                emit(
                    "error",
                    {
                        "error": "Erreur d'envoi de message.",
                        "event": "team_chat_message_error",
                        "local_id": local_id,
                    },
                )

    @socketio.on("typing_start")
    def handle_typing_start(data=None):  # noqa: ARG001
        """Handler pour l'indicateur de frappe (typing indicator)."""
        # Variables pour logging d'erreur
        user_id_log: int | None = None
        company_id_log: int | None = None
        user_public_id_log: str | None = None
        try:
            sid = _get_sid()
            sid_data = _SID_INDEX.get(sid, {})
            user_public_id = sid_data.get("user_public_id")
            company_id = sid_data.get("company_id")
            user_public_id_log = user_public_id
            company_id_log = company_id

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
                    user_id_log = user_id
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
            # ✅ 18. Améliorer gestion erreurs Socket.IO : Logger avec contexte complet
            _log_socketio_exception(
                exception=e,
                event_name="typing_start",
                user_id=user_id_log,
                company_id=company_id_log,
                additional_context={"user_public_id": user_public_id_log},
            )
            # Notifier l'utilisateur si possible
            with suppress(Exception):
                emit("error", {"error": "Erreur lors de l'indicateur de frappe."})

    @socketio.on("typing_stop")
    def handle_typing_stop(data=None):  # noqa: ARG001
        """Handler pour arrêter l'indicateur de frappe."""
        # Variables pour logging d'erreur
        user_id_log: int | None = None
        company_id_log: int | None = None
        user_public_id_log: str | None = None
        try:
            sid = _get_sid()
            sid_data = _SID_INDEX.get(sid, {})
            user_public_id = sid_data.get("user_public_id")
            company_id = sid_data.get("company_id")
            user_public_id_log = user_public_id
            company_id_log = company_id

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
                    user_id_log = user_id
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
            # ✅ 18. Améliorer gestion erreurs Socket.IO : Logger avec contexte complet
            _log_socketio_exception(
                exception=e,
                event_name="typing_stop",
                user_id=user_id_log,
                company_id=company_id_log,
                additional_context={"user_public_id": user_public_id_log},
            )
            # Notifier l'utilisateur si possible
            with suppress(Exception):
                emit("error", {"error": "Erreur lors de l'arrêt de frappe."})

    @socketio.on("join_driver_room")
    def handle_join_driver_room(data=None):  # noqa: ARG001
        # Variables pour logging d'erreur
        user_id_log: int | None = None
        user_public_id_log: str | None = None
        try:
            # ✅ SECURITY: Utiliser JWT depuis _SID_INDEX
            sid = _get_sid()
            sid_data = _SID_INDEX.get(sid, {})
            user_public_id = sid_data.get("user_public_id")
            user_public_id_log = user_public_id

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

            user_id_log = user.id

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
            # ✅ 18. Améliorer gestion erreurs Socket.IO : Logger avec contexte complet
            _log_socketio_exception(
                exception=e,
                event_name="join_driver_room",
                user_id=user_id_log,
                additional_context={"user_public_id": user_public_id_log},
            )
            # Notifier l'utilisateur si possible
            with suppress(Exception):
                emit(
                    "error",
                    {"error": "Erreur lors de la connexion à la room chauffeur."},
                )

    @socketio.on("driver_location")
    def handle_driver_location(data):  # noqa: PLR0911
        """Handler pour la réception de la localisation du chauffeur.
        ✅ FIX: Accepte driver_id dans payload + fallback robuste par user_id.

        Note: PLR0911 (too many returns) ignoré car les returns sont nécessaires
        pour la validation et la gestion d'erreurs (sécurité, rate limiting, etc.).
        """
        # Variables pour logging d'erreur
        current_sid_log: str | None = None
        user_id_log: int | None = None
        driver_id_log: int | None = None
        company_id_log: int | None = None
        payload_driver_id_log: Any = None
        try:
            # 1. Récupération du SID pour le debug
            current_sid = _get_sid()
            current_sid_log = current_sid
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
            user_id_log = user_id

            # 4. Nouvelle approche: extraire driver_id du payload si disponible
            payload_driver_id = data.get("driver_id")
            payload_driver_id_log = payload_driver_id

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
            if driver:
                driver_id_log = driver.id
            if company_id_val:
                company_id_log = company_id_val
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
            timestamp_value = data.get("timestamp")
            timestamp = _parse_timestamp(timestamp_value)

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
                        # ✅ FIX: Standardiser avec '_' au lieu de ':' pour cohérence avec mobile
                        emit("driver_arrived_at_pickup", {"driver_id": driver.id})
                    elif event == "arrived_at_dropoff":
                        # ✅ FIX: Standardiser avec '_' au lieu de ':' pour cohérence avec mobile
                        emit("driver_arrived_at_dropoff", {"driver_id": driver.id})

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
            # ✅ 18. Améliorer gestion erreurs Socket.IO : Logger avec contexte complet
            has_lat = False
            has_lon = False
            with suppress(Exception):
                if isinstance(data, dict):
                    has_lat = "latitude" in data
                    has_lon = "longitude" in data
            _log_socketio_exception(
                exception=e,
                event_name="driver_location",
                sid=current_sid_log,
                user_id=user_id_log,
                driver_id=driver_id_log,
                company_id=company_id_log,
                additional_context={
                    "payload_driver_id": payload_driver_id_log,
                    "has_latitude": has_lat,
                    "has_longitude": has_lon,
                },
            )
            # Notifier l'utilisateur si possible
            with suppress(Exception):
                emit(
                    "error", {"error": "Erreur lors de la mise à jour de localisation."}
                )

    @socketio.on("driver_location_batch")
    def handle_driver_location_batch(data):  # noqa: PLR0911
        """Handler pour la réception de batch de localisations du chauffeur.
        Traite chaque position du batch et les persiste.
        """
        # Variables pour logging d'erreur
        current_sid_log: str | None = None
        user_id_log: int | None = None
        driver_id_log: int | None = None
        company_id_log: int | None = None
        payload_driver_id_log: Any = None
        try:
            current_sid = _get_sid()
            current_sid_log = current_sid
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
                return {"success": False, "error": "Session JWT introuvable"}

            user = User.query.filter_by(public_id=user_public_id).first()
            if not user:
                emit("error", {"error": "Utilisateur introuvable"})
                return {"success": False, "error": "Utilisateur introuvable"}
            user_id = user.id
            user_id_log = user_id

            payload_driver_id = data.get("driver_id")
            payload_driver_id_log = payload_driver_id

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
                    return {
                        "success": False,
                        "error": "Rate limit exceeded",
                        "retry_after": retry_after,
                    }

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
            if driver:
                driver_id_log = driver.id
            if company_id_val:
                company_id_log = company_id_val
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
                return {
                    "success": False,
                    "error": "Chauffeur introuvable ou non lié à une entreprise",
                }

            positions = data.get("positions", [])
            if not positions:
                logger.warning("⚠️ driver_location_batch vide")
                return {"success": False, "error": "Batch vide"}

            company_room = f"company_{company_id_val}"
            now_iso = datetime.now(UTC).isoformat()

            # ✅ P2: Déduplication - vérifier si batch déjà traité
            if positions and redis_client:
                try:
                    first_ts = positions[0].get("timestamp")
                    batch_id = f"{driver.id}:{first_ts}"
                    processed_key = f"driver:{driver.id}:processed_batch"

                    # Vérifier si batch déjà traité
                    if redis_client.exists(processed_key):
                        batch_ids = cast(set[bytes], redis_client.smembers(processed_key))
                        if batch_id.encode() in batch_ids:
                            logger.info(
                                "⚠️ [Déduplication] Batch déjà traité: driver=%s, batch_id=%s",
                                driver.id,
                                batch_id,
                            )
                            # ✅ P0: Envoyer ACK de succès avec flag duplicate
                            return {
                                "success": True,
                                "positions_count": len(positions),
                                "driver_id": driver.id,
                                "timestamp": now_iso,
                                "duplicate": True,
                            }

                    # Marquer batch comme traité (TTL 5 min)
                    redis_client.sadd(processed_key, batch_id)
                    redis_client.expire(processed_key, 300)

                except Exception as dedup_err:
                    # Ne pas faire échouer l'envoi pour une erreur de déduplication
                    logger.warning(
                        "⚠️ [Déduplication] Erreur lors de la vérification: %s",
                        dedup_err,
                    )

            # Traiter chaque position du batch
            rejected_positions: list[dict[str, Any]] = []  # ✅ P2: Bug #7
            processed_count = 0  # ✅ P2: Compter positions traitées avec succès
            for idx, pos in enumerate(positions):
                try:
                    latitude = float(pos.get("latitude", 0))
                    longitude = float(pos.get("longitude", 0))

                    # ✅ P2: Bug #7 - Tracker rejets au lieu de skip silencieux
                    if not (-LAT_THRESHOLD <= latitude <= LAT_THRESHOLD):
                        rejected_positions.append(
                            {
                                "index": idx,
                                "reason": f"Latitude invalide: {latitude}",
                                "latitude": latitude,
                                "longitude": longitude,
                            }
                        )
                        continue
                    if not (-LON_THRESHOLD <= longitude <= LON_THRESHOLD):
                        rejected_positions.append(
                            {
                                "index": idx,
                                "reason": f"Longitude invalide: {longitude}",
                                "latitude": latitude,
                                "longitude": longitude,
                            }
                        )
                        continue

                    # ✅ 3.3.1: Utiliser LocationService pour chaque position du batch
                    speed = pos.get("speed")
                    heading = pos.get("heading")
                    accuracy = pos.get("accuracy")
                    timestamp_value = pos.get("timestamp")
                    timestamp = _parse_timestamp(timestamp_value)

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
                            "timestamp": timestamp.isoformat()
                            if timestamp
                            else now_iso,
                        },
                        room=company_room,
                    )

                    # ✅ P2: Incrémenter compteur de positions traitées avec succès
                    processed_count += 1

                except (TypeError, ValueError) as e:
                    # ✅ P2: Bug #7 - Tracker erreur au lieu de skip silencieux
                    rejected_positions.append(
                        {
                            "index": idx,
                            "reason": f"Erreur validation: {e!s}",
                            "position": pos,
                        }
                    )
                    logger.warning(
                        "⚠️ Position invalide dans batch (index %d): %s", idx, e
                    )
                    continue

            logger.info(
                "📡 Batch -> %s (driver %s) %s/%s positions traitées",
                company_room,
                driver.id,
                processed_count,
                len(positions),
            )

            # ✅ P2: Bug #7 - Inclure positions rejetées dans ACK
            if rejected_positions:
                logger.warning(
                    "⚠️ [Validation] %d positions rejetées sur %d",
                    len(rejected_positions),
                    len(positions),
                )

            # ✅ P0: Envoyer ACK de succès au client pour confirmer réception
            ack_response = {
                "success": True,
                "positions_count": processed_count,
                "total_positions": len(positions),
                "driver_id": driver.id,
                "timestamp": now_iso,
            }

            # ✅ P2: Bug #7 - Inclure rejets si présents
            if rejected_positions:
                ack_response["rejected"] = rejected_positions
                ack_response["rejected_count"] = len(rejected_positions)

            return ack_response

        except Exception as e:
            # ✅ 18. Améliorer gestion erreurs Socket.IO : Logger avec contexte complet
            positions_count = 0
            with suppress(Exception):
                if isinstance(data, dict):
                    positions_count = len(data.get("positions", []))
            _log_socketio_exception(
                exception=e,
                event_name="driver_location_batch",
                sid=current_sid_log,
                user_id=user_id_log,
                driver_id=driver_id_log,
                company_id=company_id_log,
                additional_context={
                    "payload_driver_id": payload_driver_id_log,
                    "positions_count": positions_count,
                },
            )
            # Notifier l'utilisateur si possible
            with suppress(Exception):
                emit(
                    "error",
                    {"error": "Erreur lors de la mise à jour batch de localisation."},
                )

            # ✅ P0: Envoyer ACK d'erreur au client pour indiquer retry
            return {
                "success": False,
                "error": "Internal error processing batch",
                "retry": True,  # Indique au client de retry
            }

    @socketio.on("join_company")
    def handle_join_company(data=None):  # noqa: ARG001
        # Variables pour logging d'erreur
        user_id_log: int | None = None
        company_id_log: int | None = None
        user_public_id_log: str | None = None
        user_role_log: str | None = None
        try:
            # ✅ SECURITY: Utiliser JWT depuis _SID_INDEX
            sid = _get_sid()
            sid_data = _SID_INDEX.get(sid, {})
            user_public_id = sid_data.get("user_public_id")
            user_role = sid_data.get("role")
            user_public_id_log = user_public_id
            user_role_log = user_role

            if not user_public_id:
                emit("error", {"error": "Session JWT introuvable. Reconnectez-vous."})
                return

            # Récupérer user depuis public_id
            user = User.query.filter_by(public_id=user_public_id).first()
            if not user:
                emit("error", {"error": "Utilisateur introuvable."})
                return

            user_id_log = user.id

            if user_role == "company":
                company = Company.query.filter_by(user_id=user.id).first()
                if not company:
                    emit("error", {"error": "Entreprise introuvable."})
                    return

                company_id_log = company.id

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

                company_id_log = driver.company_id

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
            # ✅ 18. Améliorer gestion erreurs Socket.IO : Logger avec contexte complet
            _log_socketio_exception(
                exception=e,
                event_name="join_company",
                user_id=user_id_log,
                company_id=company_id_log,
                additional_context={
                    "user_public_id": user_public_id_log,
                    "user_role": user_role_log,
                },
            )
            # Notifier l'utilisateur si possible
            with suppress(Exception):
                emit(
                    "error",
                    {"error": "Erreur lors de la connexion à la room entreprise."},
                )

    @socketio.on("get_driver_locations")
    def handle_get_driver_locations():
        # Variables pour logging d'erreur
        user_id_log: int | None = None
        company_id_log: int | None = None
        user_public_id_log: str | None = None
        user_role_log: str | None = None
        drivers_count_log: int | None = None
        try:
            # ✅ SECURITY: Utiliser JWT depuis _SID_INDEX
            sid = _get_sid()
            company_info = _SID_INDEX.get(sid, {})
            user_public_id = company_info.get("user_public_id")
            user_role = company_info.get("role")
            company_id = company_info.get("company_id")
            user_public_id_log = user_public_id
            user_role_log = user_role
            company_id_log = company_id

            if not user_public_id or user_role != "company":
                emit(
                    "error",
                    {"error": "Accès non autorisé pour la demande de localisation."},
                )
                return

            if not company_id:
                emit("error", {"error": "Entreprise non identifiée."})
                return

            # Récupérer user_id pour logging
            user = User.query.filter_by(public_id=user_public_id).first()
            if user:
                user_id_log = user.id

            # Get all drivers for this company
            drivers = Driver.query.filter_by(company_id=company_id).all()
            drivers_count_log = len(drivers)

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
            # ✅ 18. Améliorer gestion erreurs Socket.IO : Logger avec contexte complet
            _log_socketio_exception(
                exception=e,
                event_name="get_driver_locations",
                user_id=user_id_log,
                company_id=company_id_log,
                additional_context={
                    "user_public_id": user_public_id_log,
                    "user_role": user_role_log,
                    "drivers_count": drivers_count_log,
                },
            )
            # Notifier l'utilisateur si possible
            with suppress(Exception):
                emit(
                    "error",
                    {"error": "Erreur lors de la récupération des localisations."},
                )

    @socketio.on("disconnect")
    def handle_disconnect():
        try:
            sid = _get_sid()
            info = _SID_INDEX.pop(sid, None)
            trace_id = request.headers.get("X-Trace-ID") or request.headers.get(
                "Trace-Id"
            )
            now = datetime.now(UTC)

            company_id = info.get("company_id") if info else None
            user_id = info.get("user_id") if info else None
            driver_id = info.get("driver_id") if info else None
            role = info.get("role") if info else None

            # ✅ Tracking rooms : quitter les rooms appropriées
            if role == "driver" and driver_id and company_id:
                driver_room = f"driver_{driver_id}"
                company_room = f"company_{company_id}"
                try:
                    ws_metrics.on_room_leave(driver_room)
                    ws_metrics.on_room_leave(company_room)
                except (ConnectionError, OSError) as e:
                    # ✅ Ignorer les erreurs de socket déjà fermé (errno 9 = Bad file descriptor)
                    if getattr(e, "errno", None) != ERRNO_BAD_FILE_DESCRIPTOR:
                        logger.warning(
                            "Error leaving rooms for driver (network error: %s): %s",
                            type(e).__name__,
                            e,
                        )
                except Exception as e:
                    logger.warning("Unexpected error leaving rooms for driver: %s", e)
            elif role == "company" and company_id:
                company_room = f"company_{company_id}"
                try:
                    ws_metrics.on_room_leave(company_room)
                except (ConnectionError, OSError) as e:
                    # ✅ Ignorer les erreurs de socket déjà fermé (errno 9 = Bad file descriptor)
                    if getattr(e, "errno", None) != ERRNO_BAD_FILE_DESCRIPTOR:
                        logger.warning(
                            "Error leaving room for company (network error: %s): %s",
                            type(e).__name__,
                            e,
                        )
                except Exception as e:
                    logger.warning("Unexpected error leaving room for company: %s", e)

            # ✅ Métriques
            try:
                ws_metrics.on_disconnect(company_id=company_id)
            except (ConnectionError, OSError) as e:
                # ✅ Ignorer les erreurs de socket déjà fermé
                if getattr(e, "errno", None) != ERRNO_BAD_FILE_DESCRIPTOR:
                    logger.warning(
                        "Error updating metrics on disconnect (network error: %s): %s",
                        type(e).__name__,
                        e,
                    )
            except Exception as e:
                logger.warning("Unexpected error updating metrics on disconnect: %s", e)

            logger.info(
                "socket_disconnect",
                extra={
                    "event": "socket_disconnect",
                    "sid": sid,
                    "user_id": user_id,
                    "user_public_id": info.get("user_public_id") if info else None,
                    "driver_id": info.get("driver_id") if info else None,
                    "company_id": company_id,
                    "role": info.get("role") if info else None,
                    "ip": info.get("ip") if info else None,
                    "device_id": info.get("device_id") if info else None,
                    "session_diag": info.get("session_diag") if info else None,
                    "timestamp": now.isoformat(),
                    "request_trace_id": trace_id,
                },
            )
        except (ConnectionError, OSError) as e:
            # ✅ Gérer proprement les erreurs de socket fermé lors de la déconnexion
            if getattr(e, "errno", None) == ERRNO_BAD_FILE_DESCRIPTOR:
                # Bad file descriptor - socket déjà fermé, c'est normal lors d'une déconnexion
                logger.debug(
                    "Socket already closed during disconnect (errno 9), this is expected"
                )
            else:
                logger.warning(
                    "Network error during disconnect (error: %s): %s",
                    type(e).__name__,
                    e,
                )
        except Exception as e:
            # ✅ Ne pas crasher le handler de déconnexion, juste logger
            logger.exception("Unexpected error in disconnect handler: %s", e)

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
