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
import logging
import os
import time
import traceback
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, cast
from typing import cast as tcast

import jwt.exceptions as jwt_exceptions
from flask import current_app, request, session
from flask_jwt_extended import decode_token
from flask_socketio import SocketIO, emit, join_room
from socketio.exceptions import ConnectionRefusedError as SocketConnectionRefusedError

from ext import db, redis_client
from models import Company, Driver, Message, SenderRole, User, UserRole
from schemas.socket_events import EVENT_VERSION, SocketEvent
from services.company_driver_location_freshness import (
    last_seen_seconds_from_location_fields,
)
from services.geolocation.presence import (
    compute_last_seen_seconds,
    compute_location_status,
    normalize_location_mode,
    presence_status_from_location_status,
)
from services.geolocation.location import get_location_service
from services.monitoring.driver_location_metrics import (
    inc_batch_fallback_individual,
    inc_batch_points_canonical,
    inc_batch_points_observability,
    inc_batch_points_received,
    inc_batch_points_skipped,
    inc_batch_rate_limited,
    inc_received,
    inc_tracking_delivery_result,
    inc_tracking_id_propagated,
    observe_batch_latency_seconds,
    observe_driver_location_batch_ingest_size,
    observe_gps_quality,
)
from services.monitoring.location_correlation_log import log_driver_location_processed
from services.monitoring.websocket_rate_limiter import ws_rate_limiter
from services.monitoring.websocket_metrics import ws_metrics
from services.monitoring.chat_metrics import (
    inc_chat_message_rejected,
    inc_chat_message_sent,
    inc_chat_payload_validation_failed,
    inc_chat_sid_lookup_failed,
)
from services.realtime.presence_registry import register_presence, remove_presence
from services.realtime.sid_claims_registry import (
    delete_sid_claims,
    get_sid_claims,
    set_sid_claims,
)
from services.realtime.live_driver_status import (
    resolve_driver_status_for_fanout as _resolve_driver_status,
    resolve_mission_status_for_driver as _resolve_mission_status_for_driver,
    sanitize_fanout_mission_id as _sanitize_fanout_mission_id,
)

# from services.notifications.push import send_push_message  # Unused, using fanout now
from services.security.spam import can_send_message

# Constantes pour éviter les valeurs magiques
RECEIVER_ID_ZERO = 0
LAT_THRESHOLD = 90
LON_THRESHOLD = 180
MAX_MESSAGE_LENGTH = 1000
MESSAGE_PREVIEW_LENGTH = 50
MAX_PUSH_PREVIEW_LEN = 90  # longueur max preview pour push notification
# ✅ errno pour "Bad file descriptor" - survient lors de déconnexions brutales
ERRNO_BAD_FILE_DESCRIPTOR = 9

logger = logging.getLogger("socketio")


def _parse_tracking_session_timestamp(session_id: str) -> int | None:
    """Extrait le timestamp ms de trk_sess_{ts}_{suffix}."""
    if not session_id or not session_id.startswith("trk_sess_"):
        return None
    parts = session_id.split("_", 3)
    if len(parts) < 3:
        return None
    try:
        return int(parts[2])
    except (TypeError, ValueError):
        return None


# Cache local synchronisé avec sid_claims_registry (tests legacy peuvent patcher).
_SID_INDEX: Dict[str, Dict[str, Any]] = {}

# Option A — miroir Kafka du chemin socket (voie durable secondaire, opt-in).
# Quand activé, chaque position acceptée via `driver_location_batch` est aussi
# publiée fire-and-forget dans `driver.location.raw` (replay/analytics/persistance
# multi-instance), SANS dégrader la latence du fanout live (pas d'attente d'ACK).
_SOCKET_KAFKA_MIRROR_ENABLED = (
    os.getenv("TRACKING_SOCKET_KAFKA_MIRROR_ENABLED", "false").lower() == "true"
)


def _store_sid_claims(sid: str, data: Dict[str, Any]) -> None:
    set_sid_claims(sid, data)
    _SID_INDEX[sid] = data


# ✅ Tracking des erreurs token_expired par IP pour réduire le bruit dans les logs
# Format: {ip: (last_log_time | None, count)}
_TOKEN_EXPIRED_TRACKING: Dict[str, tuple[datetime | None, int]] = {}
_TOKEN_EXPIRED_LOG_INTERVAL = 60  # Logger au maximum toutes les 60 secondes par IP
_TOKEN_EXPIRED_MAX_COUNT = 5  # Après 5 erreurs, logger seulement toutes les 60s
_TOKEN_EXPIRED_TRACKING_MAX_SIZE = 1000  # Taille max du dictionnaire avant nettoyage
REALTIME_MAX_CONNECTIONS_PER_COMPANY = int(
    os.getenv("REALTIME_MAX_CONNECTIONS_PER_COMPANY", "5000")
)


def _company_conn_key(company_id: int) -> str:
    return f"realtime:company:{company_id}:active_connections"


def _try_acquire_company_slot(company_id: int) -> bool:
    if redis_client is None:
        return True
    try:
        key = _company_conn_key(company_id)
        current_raw = cast(Any, redis_client.incr(key))
        if hasattr(current_raw, "__await__"):
            # Defensive fallback si un client async est injecte par erreur.
            return True
        current = int(current_raw)
        redis_client.expire(key, 120)
        if current > REALTIME_MAX_CONNECTIONS_PER_COMPANY:
            redis_client.decr(key)
            return False
        return True
    except Exception:
        logger.exception("[socketio] company slot acquire failed")
        return True


def _release_company_slot(company_id: int) -> None:
    if redis_client is None:
        return
    try:
        key = _company_conn_key(company_id)
        redis_client.decr(key)
        redis_client.expire(key, 120)
    except Exception:
        logger.exception("[socketio] company slot release failed")


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

    sid_data = _get_sid_claims(sid) if sid else {}
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


def _get_sid_claims(sid: str | None) -> Dict[str, Any]:
    """Retourne les claims socket (Redis + cache local)."""
    if not sid:
        return {}
    claims = get_sid_claims(sid)
    if claims:
        _SID_INDEX[sid] = claims
        return claims
    raw = _SID_INDEX.get(sid)
    if isinstance(raw, dict):
        return raw
    return {}


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


def _parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    with suppress(Exception):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


def _compute_presence_from_signals(
    *,
    location_mode: str | None,
    loc_data: Mapping[str, Any],
    last_seen_ts: str | None,
) -> tuple[str, str, str, int | None]:
    """Retourne (presence_status, location_status, offline_reason, last_seen_seconds)."""
    normalized_mode = normalize_location_mode(location_mode)
    age = last_seen_seconds_from_location_fields(loc_data)
    if age is None:
        age = compute_last_seen_seconds(last_seen_ts)
    location_status = compute_location_status(
        mode=normalized_mode, last_seen_seconds=age
    )
    presence_status = presence_status_from_location_status(location_status)
    if presence_status == "offline":
        return (presence_status, location_status, "no_signal", age)
    if presence_status == "degraded":
        return (presence_status, location_status, "location_stale", age)
    return (presence_status, location_status, "", age)


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
    - Payload auth (socket.io) : accepté si header/cookie absents (navigateur, Expo web) — pas de fuite URL, même validation JWT

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
    # Important: ne pas écraser un token déjà extrait depuis Authorization.
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
        if not token_result:
            cookie_token = request.cookies.get(cookie_name)
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
            else:
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

            # Même jeton que Authorization / cookie ; requis pour clients web (handshake WS sans en-tête custom).
            token = str(tok).strip()
            logger.info(
                "socket_token_extracted",
                extra={
                    "event": "token_extracted",
                    "source": "auth_payload",
                    "has_token": bool(token),
                    "env": env,
                    "is_production": is_prod,
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

    @socketio.on("connect", namespace="/")
    def handle_connect(auth: dict[str, Any] | None) -> bool:
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
                raise SocketConnectionRefusedError(
                    "TOKEN_EXPIRED", {"retry_after": 5}
                ) from None
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

            # Multi-contexte (app unifiée) : un compte dont le rôle BDD/JWT est
            # ``company`` peut posséder un profil chauffeur et opérer en contexte
            # ``driver`` (sélecteur de contexte mobile). Le handshake Socket.IO
            # transporte ce contexte via la query (``surface`` / ``context_id``)
            # — il n'envoie PAS le header ``X-Active-Context-Id`` utilisé en HTTP.
            # On réplique ici la résolution de ``role_required`` (ext.py) pour que
            # ce socket soit authentifié comme chauffeur (sinon ``driver_location_batch``
            # est rejeté avec « Accès réservé aux chauffeurs »). La correspondance
            # ne grant que la propre identité chauffeur de l'utilisateur.
            driver_profile_for_ctx = getattr(user, "driver", None)
            requested_surface = (request.args.get("surface") or "").strip()
            requested_context_id = (request.args.get("context_id") or "").strip()
            is_driver_context = driver_profile_for_ctx is not None and (
                requested_surface == "driver"
                or requested_context_id == f"driver:{driver_profile_for_ctx.id}"
            )

            if user.role == UserRole.driver or is_driver_context:
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
                if not _try_acquire_company_slot(int(driver.company_id)):
                    ws_metrics.on_error("realtime_company_capacity_exceeded")
                    raise SocketConnectionRefusedError(
                        "COMPANY_REALTIME_CAPACITY_EXCEEDED"
                    )

                # Arbitrage de présence : un chauffeur ne doit avoir QU'UN seul
                # socket actif. Des sockets dupliqués (reconnexions / churn) font
                # que les ACK ``driver_location_batch`` partent vers un autre sid
                # que celui qui émet → la file mobile ne draine jamais → boucle de
                # retransmission → rate-limit permanent → canonical figé. On
                # déconnecte donc les anciens sockets du même driver avant
                # d'enregistrer le nouveau (single-socket canonique).
                try:
                    stale_driver_sids = [
                        old_sid
                        for old_sid, old_claims in list(_SID_INDEX.items())
                        if old_sid != sid
                        and isinstance(old_claims, dict)
                        and old_claims.get("driver_id") == driver.id
                    ]
                    for old_sid in stale_driver_sids:
                        logger.info(
                            "socket presence arbitration: driver_id=%s old_sid=%s new_sid=%s",
                            driver.id,
                            old_sid,
                            sid,
                        )
                        _SID_INDEX.pop(old_sid, None)
                        with suppress(Exception):
                            socketio.server.disconnect(old_sid, namespace="/")
                except Exception:
                    logger.debug(
                        "[socketio] driver presence arbitration skipped", exc_info=True
                    )

                company_room = f"company_{driver.company_id}"
                driver_room = f"driver_{driver.id}"
                join_room(company_room)
                join_room(driver_room)
                try:
                    from services.messaging.conversation_service import (
                        ConversationService,
                    )

                    if driver.user_id:
                        for cid in ConversationService.conversation_ids_for_user(
                            int(driver.user_id), limit=50
                        ):
                            conv_room = f"conversation_{cid}"
                            join_room(conv_room)
                            ws_metrics.on_room_join(conv_room)
                except Exception:
                    logger.exception(
                        "[socketio] join conversation rooms on connect failed"
                    )

                emit("connected", {"message": "✅ Chauffeur connecté"})

                _store_sid_claims(
                    sid,
                    {
                        "user_public_id": public_id,
                        "user_id": user.id,
                        "driver_id": driver.id,
                        "company_id": driver.company_id,
                        "ip": client_ip,
                        "role": "driver",
                        "device_id": device_id,
                        "session_diag": session_diag,
                    },
                )
                register_presence(
                    sid=sid,
                    user_id=user.id,
                    role="driver",
                    company_id=driver.company_id,
                    driver_id=driver.id,
                )

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
                if not _try_acquire_company_slot(int(company.id)):
                    ws_metrics.on_error("realtime_company_capacity_exceeded")
                    raise SocketConnectionRefusedError(
                        "COMPANY_REALTIME_CAPACITY_EXCEEDED"
                    )

                room = f"company_{company.id}"
                join_room(room)
                try:
                    from services.messaging.conversation_service import (
                        ConversationService,
                    )

                    company_inbox = ConversationService.build_company_inbox(user)
                    joined_conv = 0
                    for row in company_inbox.get("threads") or []:
                        cid = row.get("conversation_id")
                        if not cid or joined_conv >= 50:
                            continue
                        conv_room = f"conversation_{cid}"
                        join_room(conv_room)
                        ws_metrics.on_room_join(conv_room)
                        joined_conv += 1
                except Exception:
                    logger.exception(
                        "[socketio] join conversation rooms for company on connect failed"
                    )

                emit("connected", {"message": f"✅ Entreprise connectée à {room}"})

                _store_sid_claims(
                    sid,
                    {
                        "user_public_id": public_id,
                        "user_id": user.id,
                        "company_id": company.id,
                        "ip": client_ip,
                        "role": "company",
                        "device_id": device_id,
                        "session_diag": session_diag,
                    },
                )
                register_presence(
                    sid=sid,
                    user_id=user.id,
                    role="company",
                    company_id=company.id,
                )

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

            # =====================================================================
            # ÉTAPE 5/6: Institution users - portail institutionnel
            # =====================================================================
            elif user.role == UserRole.institution:
                institution_id = getattr(user, "institution_id", None)
                if not institution_id:
                    logger.error(
                        "socket_connect_error",
                        extra={
                            "event": "connect_error",
                            "reason": "institution_not_found",
                            "user_id": user.id,
                            "ip": client_ip,
                            "timestamp": now.isoformat(),
                            "request_trace_id": trace_id,
                        },
                    )
                    ws_metrics.on_error("institution_not_found")
                    raise SocketConnectionRefusedError("INSTITUTION_NOT_FOUND")

                room = f"institution_{institution_id}"
                join_room(room)
                emit("connected", {"message": f"✅ Institution connectée à {room}"})

                _store_sid_claims(
                    sid,
                    {
                        "user_public_id": public_id,
                        "user_id": user.id,
                        "institution_id": institution_id,
                        "ip": client_ip,
                        "role": "institution",
                        "device_id": device_id,
                        "session_diag": session_diag,
                    },
                )
                register_presence(
                    sid=sid,
                    user_id=user.id,
                    role="institution",
                )

                # ✅ Métriques
                ws_metrics.on_connect(user_id=user.id)
                # ✅ Tracking rooms
                ws_metrics.on_room_join(room)

                logger.info(
                    "socket_connect_success",
                    extra={
                        "event": "socket_connect_success",
                        "sid": sid,
                        "user_id": user.id,
                        "user_public_id": public_id,
                        "institution_id": institution_id,
                        "role": "institution",
                        "rooms": [room],
                        "ip": client_ip,
                        "device_id": device_id,
                        "session_diag": session_diag,
                        "timestamp": now.isoformat(),
                        "request_trace_id": trace_id,
                    },
                )

            elif user.role == UserRole.client:
                room = f"client_{public_id}"
                join_room(room)
                emit("connected", {"message": f"✅ Client connecté à {room}"})

                _store_sid_claims(
                    sid,
                    {
                        "user_public_id": public_id,
                        "user_id": user.id,
                        "ip": client_ip,
                        "role": "client",
                        "device_id": device_id,
                        "session_diag": session_diag,
                    },
                )
                register_presence(
                    sid=sid,
                    user_id=user.id,
                    role="client",
                )

                ws_metrics.on_connect(user_id=user.id)
                ws_metrics.on_room_join(room)

                logger.info(
                    "socket_connect_success",
                    extra={
                        "event": "socket_connect_success",
                        "sid": sid,
                        "user_id": user.id,
                        "user_public_id": public_id,
                        "role": "client",
                        "rooms": [room],
                        "ip": client_ip,
                        "device_id": device_id,
                        "session_diag": session_diag,
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
    def handle_team_chat(data):
        local_id = data.get("_localId") if isinstance(data, dict) else None
        try:
            if not isinstance(data, dict):
                inc_chat_payload_validation_failed(
                    event="team_chat_message", reason="not_dict"
                )
                emit("error", {"error": "Payload invalide."})
                return

            from pydantic import ValidationError

            from services.messaging.schemas import TeamChatInboundPayload

            try:
                inbound = TeamChatInboundPayload.model_validate(data)
            except ValidationError:
                inc_chat_payload_validation_failed(
                    event="team_chat_message", reason="schema"
                )
                emit("error", {"error": "Payload invalide."})
                return

            sid = _get_sid()
            sid_data = _get_sid_claims(sid)
            user_public_id = sid_data.get("user_public_id")
            logger.info(
                "[CHAT] team_chat_message sid=%s thread_id=%s has_content=%s",
                sid,
                inbound.thread_id,
                bool((inbound.content or "").strip()),
            )

            if not user_public_id:
                inc_chat_sid_lookup_failed(event="team_chat_message")
                logger.error("[CHAT] Session JWT introuvable pour SID=%s", sid)
                inc_chat_message_rejected("auth")
                emit("error", {"error": "Session JWT introuvable. Reconnectez-vous."})
                return

            user = User.query.filter_by(public_id=user_public_id).first()
            if not user:
                inc_chat_message_rejected("auth")
                emit("error", {"error": "Utilisateur non reconnu."})
                return

            client_ip = request.environ.get("REMOTE_ADDR", "unknown")
            allowed, retry_after = ws_rate_limiter.check_rate_limit(
                "team_chat_message",
                user_id=user.id,
                client_ip=client_ip,
            )
            if not allowed:
                inc_chat_message_rejected("rate_limit")
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

            allowed_spam, spam_error = can_send_message(user.id)
            if not allowed_spam:
                inc_chat_message_rejected("spam")
                emit(
                    "error",
                    {"error": spam_error or "Trop de messages. Attendez 1 seconde."},
                )
                return

            content = (inbound.content or "").strip() if inbound.content else ""
            image_url = inbound.resolved_image_url()
            pdf_url = inbound.resolved_pdf_url()
            pdf_filename = inbound.pdf_filename
            pdf_size = inbound.pdf_size
            audio_url = inbound.audio_url
            has_audio = bool(audio_url)
            has_image = bool(image_url)
            has_pdf = bool(pdf_url)
            if has_image and has_pdf:
                inc_chat_message_rejected("empty")
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

            has_content = bool(content)
            if not (has_content or has_image or has_pdf or has_audio):
                inc_chat_message_rejected("empty")
                emit(
                    "error",
                    {
                        "error": (
                            "Le message doit contenir du texte, une image, un PDF "
                            "ou un message vocal."
                        )
                    },
                )
                return

            if has_content and len(content) > MAX_MESSAGE_LENGTH:
                inc_chat_message_rejected("empty")
                emit(
                    "error",
                    {
                        "error": (
                            f"Message trop long (max {MAX_MESSAGE_LENGTH} caractères)."
                        )
                    },
                )
                return

            receiver_id = inbound.receiver_id
            thread_id = inbound.thread_id
            booking_id_raw = inbound.booking_id
            message_type = (inbound.message_type or "text").strip() or "text"
            priority = (inbound.priority or "normal").strip() or "normal"
            client_message_id = inbound.resolved_client_message_id() or local_id
            timestamp = datetime.now(UTC)

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

            from services.messaging.message_idempotence import (
                find_idempotent_message,
                note_duplicate_hit,
            )

            thread_type_metric = (
                str(thread_id).split(":", 1)[0]
                if thread_id and ":" in str(thread_id)
                else (str(thread_id) if thread_id else "unknown")
            )

            message: Message | None = None
            conversation_id_val = inbound.conversation_id
            booking_id = booking_id_raw

            if client_message_id:
                existing_msg = find_idempotent_message(
                    sender_id, str(client_message_id)
                )
                if existing_msg is not None:
                    note_duplicate_hit(channel="socket")
                    message = existing_msg

            MessageCtor = cast("Any", Message)
            if message is None:
                try:
                    content_final = content.strip() if content else None
                    if booking_id_raw is not None:
                        booking_id = int(booking_id_raw)
                    if not thread_id and booking_id:
                        thread_id = f"mission:{booking_id}"
                    if not thread_id:
                        if receiver_id:
                            thread_id = f"direct:{receiver_id}"
                        elif sender_role == SenderRole.COMPANY:
                            thread_id = "dispatch"
                        else:
                            thread_id = "team"

                    conv_obj = None
                    try:
                        from models import Conversation
                        from services.messaging.conversation_service import (
                            ConversationService,
                        )
                        from services.messaging.permission_service import (
                            MessagingPermissionService,
                        )

                        if conversation_id_val:
                            conv_obj = Conversation.query.get(int(conversation_id_val))
                        else:
                            driver_resolve = (
                                Driver.query.filter_by(user_id=user.id).first()
                                if user.role == UserRole.driver
                                else None
                            )
                            conv_obj = ConversationService.resolve_by_legacy_thread(
                                int(company_id),
                                str(thread_id),
                                driver_resolve,
                            )
                            if (
                                thread_id
                                and str(thread_id).startswith("direct:")
                                and conv_obj is None
                            ):
                                inc_chat_message_rejected("permission")
                                emit(
                                    "error",
                                    {
                                        "error": (
                                            "Collègue introuvable ou message "
                                            "direct refusé."
                                        )
                                    },
                                )
                                return
                        if conv_obj is not None:
                            MessagingPermissionService.assert_can_write(user, conv_obj)
                            conversation_id_val = conv_obj.id
                    except PermissionError as perm_err:
                        inc_chat_message_rejected("permission")
                        emit("error", {"error": str(perm_err)})
                        return
                    except Exception:
                        logger.exception("[CHAT] conversation resolve failed")

                    message = MessageCtor(
                        sender_id=sender_id,
                        receiver_id=receiver_id,
                        company_id=company_id,
                        sender_role=sender_role,
                        content=content_final if content_final else None,
                        timestamp=timestamp,
                        image_url=image_url if has_image else None,
                        pdf_url=pdf_url if has_pdf else None,
                        pdf_filename=pdf_filename if has_pdf else None,
                        pdf_size=int(pdf_size) if has_pdf and pdf_size else None,
                        audio_url=audio_url if has_audio else None,
                        thread_id=str(thread_id) if thread_id else None,
                        booking_id=booking_id,
                        message_type=(
                            "audio"
                            if has_audio and not (has_content or has_image or has_pdf)
                            else message_type
                        ),
                        priority=priority,
                        client_message_id=str(client_message_id)
                        if client_message_id
                        else None,
                        conversation_id=int(conversation_id_val)
                        if conversation_id_val
                        else None,
                        visibility_tags=["operational"],
                    )
                    db.session.add(message)
                    db.session.commit()
                except Exception as commit_err:
                    from sqlalchemy.exc import IntegrityError

                    db.session.rollback()
                    if client_message_id and isinstance(commit_err, IntegrityError):
                        raced = find_idempotent_message(
                            sender_id, str(client_message_id)
                        )
                        if raced is not None:
                            note_duplicate_hit(channel="socket")
                            message = raced
                        else:
                            logger.exception(
                                "[CHAT] commit IntegrityError sans message existant"
                            )
                            emit(
                                "error",
                                {"error": "Erreur lors de la sauvegarde du message."},
                            )
                            return
                    else:
                        logger.exception(
                            "[CHAT] Erreur lors du commit du message: %s",
                            commit_err,
                        )
                        emit(
                            "error",
                            {"error": ("Erreur lors de la sauvegarde du message.")},
                        )
                        return

            if message is None:
                emit("error", {"error": "Erreur lors de la sauvegarde du message."})
                return

            conversation_id_val = message.conversation_id or conversation_id_val
            thread_id = message.thread_id or thread_id
            content = (message.content or "").strip() if message.content else content
            inc_chat_message_sent(channel="socket", thread_type=thread_type_metric)

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
                "thread_id": getattr(message, "thread_id", None),
                "booking_id": getattr(message, "booking_id", None),
                "message_type": getattr(message, "message_type", None) or "text",
                "priority": getattr(message, "priority", None) or "normal",
                "audio_url": audio_url if has_audio else None,
                "conversation_id": conversation_id_val,
            }

            # ✅ Enrichir payload avec event_id, version, timestamp
            enriched_payload = _enrich_payload_if_needed(payload, "team_chat_message")

            from services.messaging.channel_routing import emit_chat_message

            emit_chat_message(
                cast("Any", emit),
                "team_chat_message",
                enriched_payload,
                company_id=int(company_id),
                thread_id=str(thread_id) if thread_id else None,
                conversation_id=int(conversation_id_val)
                if conversation_id_val
                else None,
                receiver_id=int(receiver_id) if receiver_id else None,
            )
            logger.info(
                "📨 Message routé (thread_id=%s, conversation_id=%s) par %s",
                thread_id,
                conversation_id_val,
                sender_role,
            )

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
                            sender_label = (
                                "Entreprise"
                                if user.role == UserRole.company
                                else "Chauffeur"
                            )
                            sender_display = sender_label
                            try:
                                # Entreprise: utiliser le nom de l'entreprise si dispo
                                if user.role == UserRole.company and company_obj:
                                    company_name = getattr(company_obj, "name", None)
                                    if company_name:
                                        sender_display = (
                                            f"{sender_label} {company_name}"
                                        )
                                else:
                                    first_name = (
                                        getattr(user, "first_name", None) or None
                                    )
                                    if first_name:
                                        sender_display = f"{sender_label} {first_name}"
                            except Exception:
                                sender_display = sender_label
                            # ✅ Preview normalisé (trim, \n→espace, max 90 chars)
                            message_preview = content[:90].strip() if content else ""
                            message_preview = message_preview.replace(
                                "\n", " "
                            ).replace("\r", " ")
                            while "  " in message_preview:
                                message_preview = message_preview.replace("  ", " ")
                            message_preview = message_preview.strip()
                            if not message_preview:
                                message_preview = "Nouveau message"
                            elif len(message_preview) > MAX_PUSH_PREVIEW_LEN:
                                message_preview = (
                                    message_preview[: MAX_PUSH_PREVIEW_LEN - 1] + "…"
                                )

                            # ✅ Utiliser fanout_message_new (templates + anti-spam)
                            from services.events.fanout import fanout_message_new

                            fanout_message_new(
                                driver_id=driver.id,
                                message_id=message.id,
                                sender_name=sender_display,
                                message_preview=message_preview,
                                company_id=company_id,
                                chat_type="direct",
                                thread_id=company_id,
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
                sid_data = _get_sid_claims(sid)
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

    @socketio.on("team_chat_typing")
    def handle_team_chat_typing(data=None):
        """Alias mobile unified-app → relay team_chat_typing to company room."""
        if not isinstance(data, dict):
            inc_chat_payload_validation_failed(
                event="team_chat_typing", reason="not_dict"
            )
            return
        try:
            from pydantic import ValidationError

            from services.messaging.schemas import TypingPayload

            inbound_typing = TypingPayload.model_validate(data)
        except ValidationError:
            inc_chat_payload_validation_failed(
                event="team_chat_typing", reason="schema"
            )
            return
        try:
            sid = _get_sid()
            sid_data = _get_sid_claims(sid)
            user_public_id = sid_data.get("user_public_id")
            company_id = sid_data.get("company_id")
            if not user_public_id or not company_id:
                return
            sender_name = inbound_typing.sender_name
            if not sender_name:
                user = User.query.filter_by(public_id=user_public_id).first()
                sender_name = getattr(user, "first_name", None) if user else None
            typing_payload = {
                "user_id": user_public_id,
                "sender_name": sender_name or "Chauffeur",
                "surface": inbound_typing.surface,
                "conversation_id": inbound_typing.conversation_id,
            }
            conv_id = inbound_typing.conversation_id
            if conv_id:
                conv_room = f"conversation_{conv_id}"
                cast("Any", emit)(
                    "team_chat_typing", typing_payload, room=conv_room, skip_sid=sid
                )
            room = f"company_{company_id}"
            cast("Any", emit)(
                "team_chat_typing",
                typing_payload,
                room=room,
                skip_sid=sid,
            )
        except Exception as e:
            _log_socketio_exception(exception=e, event_name="team_chat_typing")

    @socketio.on("typing_start")
    def handle_typing_start(data=None):  # noqa: ARG001
        """Handler pour l'indicateur de frappe (typing indicator)."""
        # Variables pour logging d'erreur
        user_id_log: int | None = None
        company_id_log: int | None = None
        user_public_id_log: str | None = None
        try:
            sid = _get_sid()
            sid_data = _get_sid_claims(sid)
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
            sid_data = _get_sid_claims(sid)
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
            sid_data = _get_sid_claims(sid)
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
            try:
                from services.messaging.conversation_service import ConversationService

                if driver.user_id:
                    for cid in ConversationService.conversation_ids_for_user(
                        int(driver.user_id), limit=50
                    ):
                        conv_room = f"conversation_{cid}"
                        join_room(conv_room)
                        ws_metrics.on_room_join(conv_room)
            except Exception:
                logger.exception("[socketio] join conversation rooms failed")
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
    def handle_driver_location(data):
        """Handler pour la réception de la localisation du chauffeur.

        Contrat modes / transports : ``backend/docs/DRIVER_LOCATION_CONTRACT.md``.
        ``availability_presence`` est refusé ici (HTTP uniquement) — erreur
        ``availability_presence_socket_forbidden``.

        ✅ FIX: Accepte driver_id dans payload + fallback robuste par user_id.

        Policy ASSIGNED (P1): Les chauffeurs ASSIGNED peuvent envoyer des positions
        comme les autres (même rate limit). Voir backend/docs/POLICY_DRIVER_LOCATION_ASSIGNED.md.

        Note: PLR0911 (too many returns) ignoré car les returns sont nécessaires
        pour la validation et la gestion d'erreurs (sécurité, rate limiting, etc.).
        """
        # Variables pour logging d'erreur
        current_sid_log: str | None = None
        user_id_log: int | None = None
        driver_id_log: int | None = None
        company_id_log: int | None = None
        payload_driver_id_log: Any = None
        t0 = time.perf_counter()
        try:
            # 1. Récupération du SID pour le debug
            current_sid = _get_sid()
            current_sid_log = current_sid
            logger.info("📍 driver_location reçu, SID=%s, data=%s", current_sid, data)

            if isinstance(data, dict) and data.get("batch_fallback") is True:
                inc_batch_fallback_individual()

            # ✅ SECURITY: Utiliser JWT depuis _SID_INDEX uniquement
            sid_info = _get_sid_claims(current_sid)
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

            # 4. Ownership: driver strictement dérivé du JWT
            payload_driver_id = data.get("driver_id")
            payload_driver_id_log = payload_driver_id

            if user_role != "driver" or not user_id:
                emit("error", {"error": "Accès réservé aux chauffeurs."})
                return

            driver = Driver.query.filter_by(user_id=user_id).first()
            if driver is None:
                emit("error", {"error": "Chauffeur introuvable."})
                return

            if payload_driver_id is not None and isinstance(
                payload_driver_id, (int, str)
            ):
                try:
                    candidate_id = int(payload_driver_id)
                    if candidate_id != int(driver.id):
                        logger.warning(
                            "⛔ driver_id payload invalide: payload=%s, jwt_driver=%s",
                            candidate_id,
                            driver.id,
                        )
                        emit(
                            "error", {"error": "driver_id invalide pour cette session."}
                        )
                        return
                except (ValueError, TypeError):
                    emit("error", {"error": "driver_id invalide."})
                    return

            company_id_val = tcast("int | None", getattr(driver, "company_id", None))
            driver_id_log = driver.id
            if company_id_val:
                company_id_log = company_id_val
            if company_id_val is None:
                emit("error", {"error": "Chauffeur non lié à une entreprise."})
                return

            # ✅ Rate limiting
            client_ip = request.environ.get("REMOTE_ADDR", "unknown")
            allowed, retry_after = ws_rate_limiter.check_rate_limit(
                "driver_location",
                user_id=user_id,
                driver_id=int(driver.id),
                client_ip=client_ip,
            )
            if not allowed:
                logger.warning(
                    "🚫 Rate limit driver_location dépassé pour driver_id=%s, retry_after=%d",
                    driver.id,
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

            latitude = data.get("latitude")
            longitude = data.get("longitude")

            # ✅ Validation stricte lat/lon
            if latitude is None or longitude is None:
                emit("error", {"error": "Latitude et longitude requises."})
                return
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
            location_mode = normalize_location_mode(
                tcast("str | None", data.get("location_mode"))
            )
            recorded_at_value = data.get("recorded_at") or timestamp_value
            sent_at_value = data.get("sent_at")
            mission_id = data.get("mission_id")
            is_background = bool(data.get("is_background", False))
            recorded_at_dt = _parse_timestamp(recorded_at_value)
            sent_at_dt = (
                _parse_timestamp(sent_at_value) if sent_at_value else datetime.now(UTC)
            )
            if data.get("location_mode") is None or data.get("recorded_at") is None:
                emit(
                    "error",
                    {
                        "error": "missing required fields",
                        "reason": "missing_required_fields",
                    },
                )
                return
            if location_mode == "availability_presence":
                inc_tracking_delivery_result(
                    mode="availability_presence",
                    transport="socket",
                    result="forbidden",
                )
                emit(
                    "error",
                    {
                        "error": "availability_presence not allowed on socket single",
                        "reason": "availability_presence_socket_forbidden",
                    },
                )
                return

            raw_mode_sock = str(data.get("location_mode") or "mission_live")
            loc_svc_sock = get_location_service()
            norm_mode_sock = loc_svc_sock.resolve_normalized_location_mode(
                company_id_val, raw_mode_sock
            )
            leid_sock = data.get("location_event_id")
            from services.geolocation.driver_location_dedup import (
                should_skip_location_ingest,
            )
            from services.monitoring.driver_location_metrics import (
                inc_dedup_skipped,
                inc_received,
            )

            skip_ingest_sock, skip_reason_sock = should_skip_location_ingest(
                driver.id,
                latitude,
                longitude,
                recorded_at_dt,
                location_mode,
                str(leid_sock) if leid_sock else None,
            )
            if skip_ingest_sock and skip_reason_sock:
                inc_dedup_skipped(
                    reason=skip_reason_sock,
                    location_mode=norm_mode_sock,
                    transport="socket",
                )
                return

            inc_received(transport="socket", location_mode=norm_mode_sock)

            snapped_lat, snapped_lon = latitude, longitude
            accept_status = "accepted_observability_only"
            received_at = datetime.now(UTC).isoformat()
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
                    location_mode=location_mode,
                    recorded_at=recorded_at_dt,
                    sent_at=sent_at_dt,
                    is_background=is_background,
                    mission_id=mission_id if isinstance(mission_id, int) else None,
                    transport="socket",
                )

                # Utiliser position snapée
                snapped_lat = result.snapped_lat
                snapped_lon = result.snapped_lon
                accept_status = result.accept_status
                received_at = result.received_at or received_at

                log_driver_location_processed(
                    driver_id=driver.id,
                    company_id=company_id_val,
                    transport="socket",
                    location_mode=norm_mode_sock,
                    accept_status=accept_status,
                    accept_reason=result.accept_reason,
                    location_event_id=str(leid_sock) if leid_sock else None,
                )

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

            # 7. P2: Fanout realtime unifié
            now_iso = datetime.now(UTC).isoformat()
            last_seen_seconds = last_seen_seconds_from_location_fields(
                {
                    "recorded_at": recorded_at_dt.isoformat()
                    if recorded_at_dt
                    else None,
                    "received_at": received_at,
                    "ts": timestamp.isoformat() if timestamp else None,
                }
            )
            location_status = compute_location_status(
                mode=location_mode, last_seen_seconds=last_seen_seconds
            )
            presence_status = presence_status_from_location_status(location_status)
            mission_status = _resolve_mission_status_for_driver(driver.id)
            driver_status = _resolve_driver_status(
                mission_status=mission_status,
                is_active=bool(getattr(driver, "is_active", True)),
                presence_status=presence_status,
            )
            fanout_mission_id = _sanitize_fanout_mission_id(
                driver.id,
                mission_id if isinstance(mission_id, int) else None,
            )
            from services.realtime.socketio import fanout_driver_location_update

            fanout_driver_location_update(
                company_id_val,
                {
                    "driver_id": driver.id,
                    "company_id": company_id_val,
                    "first_name": getattr(
                        getattr(driver, "user", None), "first_name", None
                    ),
                    "latitude": snapped_lat,
                    "longitude": snapped_lon,
                    "timestamp": recorded_at_dt.isoformat()
                    if recorded_at_dt
                    else now_iso,
                    "recorded_at": recorded_at_dt.isoformat()
                    if recorded_at_dt
                    else now_iso,
                    "received_at": received_at,
                    "location_mode": location_mode,
                },
                {
                    "driver_id": driver.id,
                    "company_id": company_id_val,
                    "lat": snapped_lat,
                    "lng": snapped_lon,
                    "timestamp": recorded_at_dt.isoformat()
                    if recorded_at_dt
                    else now_iso,
                    "status": driver_status,
                    "mission_status": mission_status,
                    "presence_status": presence_status,
                    "location_status": location_status,
                    "is_available": driver_status == "available",
                    "offline_reason": "location_stale"
                    if location_status == "stale"
                    else "",
                    "last_seen_seconds": last_seen_seconds,
                    "location_mode": location_mode,
                    "mission_id": fanout_mission_id,
                    "recorded_at": recorded_at_dt.isoformat()
                    if recorded_at_dt
                    else now_iso,
                    "received_at": received_at,
                },
                accept_status=accept_status,
            )
            if accept_status == "accepted_canonical":
                from services.geolocation.driver_eta_socket_fanout import (
                    maybe_emit_eta_changed_after_driver_location,
                )
                from services.monitoring.driver_eta_socket_metrics import (
                    inc_driver_location_ingested_for_eta_ratio,
                )

                inc_driver_location_ingested_for_eta_ratio()
                maybe_emit_eta_changed_after_driver_location(
                    driver_id=driver.id,
                    driver_lat=float(snapped_lat),
                    driver_lon=float(snapped_lon),
                    accept_status=accept_status,
                )
            elapsed = time.perf_counter() - t0
            ws_metrics.on_driver_location_latency(elapsed)
            logger.info(
                "📡 Loc -> company_%s (driver %s) %s,%s",
                company_id_val,
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
    def handle_driver_location_batch(data):
        """Handler pour la réception de batch de localisations du chauffeur.

        Contrat : ``backend/docs/DRIVER_LOCATION_CONTRACT.md``. Les entrées
        ``availability_presence`` sont rejetées (``availability_presence_socket_forbidden``) ;
        utiliser ``PUT /driver/me/location`` pour la présence.
        Traite chaque position du batch et les persiste.
        """
        # Variables pour logging d'erreur
        current_sid_log: str | None = None
        user_id_log: int | None = None
        driver_id_log: int | None = None
        company_id_log: int | None = None
        payload_driver_id_log: Any = None
        batch_t0 = time.perf_counter()
        batch_platform = "unknown"
        try:
            db.session.rollback()
            current_sid = _get_sid()
            current_sid_log = current_sid
            logger.info(
                "📍 driver_location_batch reçu, SID=%s, positions_count=%s",
                current_sid,
                len(data.get("positions", [])),
            )

            # ✅ SECURITY: Utiliser JWT depuis _SID_INDEX uniquement
            sid_info = _get_sid_claims(current_sid)
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

            if user_role != "driver":
                emit("error", {"error": "Accès réservé aux chauffeurs."})
                return {"success": False, "error": "Accès réservé aux chauffeurs"}

            driver = Driver.query.filter_by(user_id=user_id).first()
            if driver is None:
                emit("error", {"error": "Chauffeur introuvable."})
                return {"success": False, "error": "Chauffeur introuvable"}

            if payload_driver_id is not None and isinstance(
                payload_driver_id, (int, str)
            ):
                try:
                    candidate_id = int(payload_driver_id)
                    if candidate_id != int(driver.id):
                        logger.warning(
                            "⛔ driver_id payload invalide (batch): payload=%s, jwt_driver=%s",
                            candidate_id,
                            driver.id,
                        )
                        emit(
                            "error", {"error": "driver_id invalide pour cette session."}
                        )
                        return {
                            "success": False,
                            "error": "driver_id invalide pour cette session",
                        }
                except (ValueError, TypeError):
                    emit("error", {"error": "driver_id invalide."})
                    return {"success": False, "error": "driver_id invalide"}

            company_id_val = tcast("int | None", getattr(driver, "company_id", None))
            driver_id_log = driver.id
            if company_id_val:
                company_id_log = company_id_val
            if company_id_val is None:
                emit("error", {"error": "Chauffeur non lié à une entreprise."})
                return {"success": False, "error": "Chauffeur non lié à une entreprise"}

            # ✅ Rate limiting
            client_ip = request.environ.get("REMOTE_ADDR", "unknown")
            allowed, retry_after = ws_rate_limiter.check_rate_limit(
                "driver_location_batch",
                user_id=user_id,
                driver_id=int(driver.id),
                client_ip=client_ip,
            )
            if not allowed:
                logger.warning(
                    "🚫 Rate limit driver_location_batch dépassé pour driver_id=%s, retry_after=%d",
                    driver.id,
                    retry_after or 0,
                )
                inc_batch_rate_limited()
                emit(
                    "rate_limit_exceeded",
                    {
                        "event": "rate_limit_exceeded",
                        "message": f"Trop de mises à jour batch. Réessayez dans {retry_after} secondes.",
                        "attempts": 1,
                        "retry_after_seconds": retry_after,
                    },
                )
                # Anti-tempête : ACK quand même les tracking_event_id du batch rate-limité.
                # Sinon un batch jamais ACK est retransmis en boucle par le client →
                # re-rate-limité → famine permanente du canonical (le chauffeur reste
                # « figé »). Les positions ne sont PAS ingérées ici ; le client draine sa
                # file et le prochain batch autorisé (≤ fenêtre) portera des points frais.
                rl_positions = data.get("positions") if isinstance(data, dict) else None
                if isinstance(rl_positions, list) and rl_positions:
                    rl_acked_ids: list[str] = []
                    rl_last_seq: int | None = None
                    for _rl_pos in rl_positions:
                        if not isinstance(_rl_pos, dict):
                            continue
                        _rl_teid = _rl_pos.get("tracking_event_id")
                        if isinstance(_rl_teid, str):
                            rl_acked_ids.append(_rl_teid)
                        _rl_seq = _rl_pos.get("sequence_id")
                        if isinstance(_rl_seq, (int, str)):
                            with suppress(Exception):
                                _rl_seq_i = int(_rl_seq)
                                rl_last_seq = (
                                    _rl_seq_i
                                    if rl_last_seq is None
                                    else max(rl_last_seq, _rl_seq_i)
                                )
                    if rl_acked_ids or rl_last_seq is not None:
                        rl_ack: dict[str, Any] = {
                            "success": True,
                            "rate_limited": True,
                            "positions_count": 0,
                            "total_positions": len(rl_positions),
                            "driver_id": driver.id,
                            "tracking_event_ids": rl_acked_ids,
                        }
                        if rl_last_seq is not None:
                            rl_ack["ack_last_sequence_id"] = rl_last_seq
                        if retry_after is not None:
                            rl_ack["retry_after_seconds"] = retry_after
                        emit("driver_location_batch_ack", rl_ack)
                ws_metrics.on_error("rate_limit_exceeded")
                ws_metrics.on_rate_limit_hit("driver_location_batch")
                return {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after,
                }

            positions = data.get("positions", [])
            tracking_session_id = (
                str(data.get("tracking_session_id")).strip()
                if isinstance(data, dict) and data.get("tracking_session_id")
                else ""
            )
            if not tracking_session_id:
                return {"success": False, "error": "tracking_session_id_required"}
            if redis_client is not None:
                session_key = f"driver:{driver.id}:active_tracking_session"
                active_raw = redis_client.get(session_key)
                active_value = (
                    active_raw.decode("utf-8")
                    if isinstance(active_raw, bytes)
                    else str(active_raw)
                    if active_raw is not None
                    else None
                )
                if active_value and active_value != tracking_session_id:
                    active_ts = _parse_tracking_session_timestamp(active_value)
                    incoming_ts = _parse_tracking_session_timestamp(tracking_session_id)
                    if (
                        incoming_ts is not None
                        and active_ts is not None
                        and incoming_ts < active_ts
                    ):
                        logger.warning(
                            "⛔ driver_location_batch session stale: driver=%s active=%s incoming=%s",
                            driver.id,
                            active_value,
                            tracking_session_id,
                        )
                        stale_positions = (
                            data.get("positions") if isinstance(data, dict) else None
                        )
                        if isinstance(stale_positions, list) and stale_positions:
                            stale_acked: list[str] = []
                            stale_last_seq: int | None = None
                            for _sp in stale_positions:
                                if not isinstance(_sp, dict):
                                    continue
                                _steid = _sp.get("tracking_event_id")
                                if isinstance(_steid, str):
                                    stale_acked.append(_steid)
                                _sseq = _sp.get("sequence_id")
                                if isinstance(_sseq, (int, str)):
                                    with suppress(Exception):
                                        _sseq_i = int(_sseq)
                                        stale_last_seq = (
                                            _sseq_i
                                            if stale_last_seq is None
                                            else max(stale_last_seq, _sseq_i)
                                        )
                            stale_ack: dict[str, Any] = {
                                "success": True,
                                "session_conflict": True,
                                "positions_count": 0,
                                "total_positions": len(stale_positions),
                                "driver_id": driver.id,
                                "tracking_event_ids": stale_acked,
                            }
                            if stale_last_seq is not None:
                                stale_ack["ack_last_sequence_id"] = stale_last_seq
                            emit("driver_location_batch_ack", stale_ack)
                        return {
                            "success": False,
                            "error": "tracking_session_conflict",
                        }
                    logger.warning(
                        "♻️ tracking_session takeover driver=%s %s -> %s",
                        driver.id,
                        active_value,
                        tracking_session_id,
                    )
                redis_client.setex(session_key, 1800, tracking_session_id)
            if not positions:
                logger.warning("⚠️ driver_location_batch vide")
                return {"success": False, "error": "Batch vide"}

            from services.geolocation.driver_location_pipeline import (
                process_driver_location_points,
            )

            positions = process_driver_location_points(list(positions))
            if not positions:
                logger.warning("⚠️ driver_location_batch vide après filtre")
                return {"success": False, "error": "Batch vide"}

            observe_driver_location_batch_ingest_size(size=len(positions))

            first_pos_for_platform = positions[0] if positions else {}
            raw_platform = first_pos_for_platform.get("platform")
            if isinstance(raw_platform, str) and raw_platform.strip():
                batch_platform = raw_platform.strip().lower()
            else:
                ua = request.headers.get("User-Agent", "").lower()
                if "iphone" in ua or "ipad" in ua or "ios" in ua:
                    batch_platform = "ios"
                elif "android" in ua:
                    batch_platform = "android"

            for pos in positions:
                norm_mode_pre = normalize_location_mode(
                    tcast("str | None", pos.get("location_mode"))
                )
                inc_batch_points_received(location_mode=norm_mode_pre)
                observe_gps_quality(
                    platform=batch_platform,
                    location_mode=norm_mode_pre,
                    transport="socket_batch",
                    accuracy=float(pos["accuracy"])
                    if pos.get("accuracy") is not None
                    else None,
                    speed=float(pos["speed"]) if pos.get("speed") is not None else None,
                    heading=float(pos["heading"])
                    if pos.get("heading") is not None
                    else None,
                    provider=tcast("str | None", pos.get("provider")),
                )

            # Instrumentation: tracer les clés reçues pour diagnostiquer payload cassé
            first_pos = positions[0] if positions else {}
            pos_keys = list(first_pos.keys())
            has_loc_mode = (
                "location_mode" in first_pos
                and first_pos.get("location_mode") is not None
            )
            has_rec_at = (
                "recorded_at" in first_pos and first_pos.get("recorded_at") is not None
            )
            if not has_loc_mode or not has_rec_at:
                logger.warning(
                    "driver_location_batch position missing required fields: keys=%s has_location_mode=%s has_recorded_at=%s",
                    pos_keys,
                    has_loc_mode,
                    has_rec_at,
                )

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
                        batch_ids = cast(
                            set[bytes], redis_client.smembers(processed_key)
                        )
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
            acked_tracking_event_ids: list[str] = []
            ack_last_sequence_id: int | None = None
            for idx, raw_pos in enumerate(positions):
                try:
                    pos = raw_pos
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
                    location_mode = normalize_location_mode(
                        tcast("str | None", pos.get("location_mode"))
                    )
                    recorded_at_value = pos.get("recorded_at") or timestamp_value
                    sent_at_value = pos.get("sent_at")
                    mission_id = pos.get("mission_id")
                    is_background = bool(pos.get("is_background", False))
                    recorded_at_dt = _parse_timestamp(recorded_at_value)
                    sent_at_dt = (
                        _parse_timestamp(sent_at_value)
                        if sent_at_value
                        else datetime.now(UTC)
                    )
                    if pos.get("location_mode") is None or not recorded_at_value:
                        rejected_positions.append(
                            {
                                "index": idx,
                                "reason": "missing_required_fields",
                                "position": pos,
                            }
                        )
                        continue
                    if pos.get("recorded_at") is None:
                        pos_with_ts = dict(pos)
                        pos_with_ts["recorded_at"] = recorded_at_value
                    else:
                        pos_with_ts = pos
                    if location_mode == "availability_presence":
                        inc_tracking_delivery_result(
                            mode="availability_presence",
                            transport="socket_batch",
                            result="forbidden",
                        )
                        rejected_positions.append(
                            {
                                "index": idx,
                                "reason": "availability_presence_socket_forbidden",
                                "position": pos_with_ts,
                            }
                        )
                        inc_batch_points_skipped(
                            reason="forbidden_mode",
                            location_mode=location_mode,
                        )
                        continue

                    raw_mode_batch = str(
                        pos_with_ts.get("location_mode") or "mission_live"
                    )
                    loc_svc_batch = get_location_service()
                    norm_mode_batch = loc_svc_batch.resolve_normalized_location_mode(
                        company_id_val, raw_mode_batch
                    )
                    leid_b = pos_with_ts.get("location_event_id")
                    from services.geolocation.driver_location_dedup import (
                        should_skip_location_ingest,
                    )
                    from services.monitoring.driver_location_metrics import (
                        inc_dedup_skipped,
                    )

                    skip_ingest, skip_r = should_skip_location_ingest(
                        driver.id,
                        latitude,
                        longitude,
                        recorded_at_dt,
                        location_mode,
                        str(leid_b) if leid_b else None,
                    )
                    if skip_ingest and skip_r:
                        tracking_event_id_val = pos_with_ts.get("tracking_event_id")
                        if isinstance(tracking_event_id_val, str):
                            acked_tracking_event_ids.append(tracking_event_id_val)
                        seq_obj = pos_with_ts.get("sequence_id")
                        if isinstance(seq_obj, (int, str)):
                            with suppress(Exception):
                                seq = int(seq_obj)
                                ack_last_sequence_id = (
                                    seq
                                    if ack_last_sequence_id is None
                                    else max(ack_last_sequence_id, seq)
                                )
                        inc_dedup_skipped(
                            reason=skip_r,
                            location_mode=norm_mode_batch,
                            transport="socket_batch",
                        )
                        inc_batch_points_skipped(
                            reason="dedup",
                            location_mode=norm_mode_batch,
                        )
                        processed_count += 1
                        continue

                    inc_received(
                        transport="socket_batch", location_mode=norm_mode_batch
                    )

                    snapped_lat, snapped_lon = latitude, longitude
                    accept_status = "accepted_observability_only"
                    received_at = datetime.now(UTC).isoformat()
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
                            location_mode=location_mode,
                            recorded_at=recorded_at_dt,
                            sent_at=sent_at_dt,
                            is_background=is_background,
                            mission_id=mission_id
                            if isinstance(mission_id, int)
                            else None,
                            transport="socket_batch",
                        )

                        # Utiliser position snapée
                        snapped_lat = result.snapped_lat
                        snapped_lon = result.snapped_lon
                        accept_status = result.accept_status
                        received_at = result.received_at or received_at
                        if accept_status == "accepted_canonical":
                            inc_batch_points_canonical(location_mode=norm_mode_batch)
                        elif accept_status == "accepted_observability_only":
                            inc_batch_points_observability(
                                location_mode=norm_mode_batch
                            )

                        log_driver_location_processed(
                            driver_id=driver.id,
                            company_id=company_id_val,
                            transport="socket_batch",
                            location_mode=norm_mode_batch,
                            accept_status=accept_status,
                            accept_reason=result.accept_reason,
                            location_event_id=str(leid_b) if leid_b else None,
                        )
                        inc_tracking_delivery_result(
                            mode=norm_mode_batch,
                            transport="socket_batch",
                            result="success",
                        )

                        # Émettre events geofencing si détectés (seulement pour dernière position)
                        if idx == len(positions) - 1:
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

                    # P2: Fanout realtime unifié
                    mission_status = _resolve_mission_status_for_driver(driver.id)
                    last_seen_seconds = last_seen_seconds_from_location_fields(
                        {
                            "recorded_at": recorded_at_dt.isoformat()
                            if recorded_at_dt
                            else None,
                            "received_at": received_at,
                            "ts": timestamp.isoformat() if timestamp else None,
                        }
                    )
                    location_status = compute_location_status(
                        mode=location_mode, last_seen_seconds=last_seen_seconds
                    )
                    presence_status = presence_status_from_location_status(
                        location_status
                    )
                    driver_status = _resolve_driver_status(
                        mission_status=mission_status,
                        is_active=bool(getattr(driver, "is_active", True)),
                        presence_status=presence_status,
                    )
                    fanout_mission_id = _sanitize_fanout_mission_id(
                        driver.id,
                        mission_id if isinstance(mission_id, int) else None,
                    )
                    ts_str = (
                        recorded_at_dt.isoformat()
                        if recorded_at_dt
                        else (timestamp.isoformat() if timestamp else now_iso)
                    )
                    from services.realtime.socketio import fanout_driver_location_update

                    tracking_event_id_fanout = pos_with_ts.get("tracking_event_id")
                    tracking_event_id_str = (
                        str(tracking_event_id_fanout).strip()
                        if isinstance(tracking_event_id_fanout, str)
                        else None
                    )
                    inc_tracking_id_propagated(
                        transport="socket_batch",
                        propagated=bool(tracking_event_id_str),
                    )

                    location_payload = {
                        "driver_id": driver.id,
                        "company_id": company_id_val,
                        "first_name": getattr(
                            getattr(driver, "user", None), "first_name", None
                        ),
                        "latitude": snapped_lat,
                        "longitude": snapped_lon,
                        "timestamp": ts_str,
                        "recorded_at": ts_str,
                        "received_at": received_at,
                        "location_mode": location_mode,
                    }
                    live_state_payload = {
                        "driver_id": driver.id,
                        "company_id": company_id_val,
                        "lat": snapped_lat,
                        "lng": snapped_lon,
                        "timestamp": ts_str,
                        "status": driver_status,
                        "mission_status": mission_status,
                        "presence_status": presence_status,
                        "location_status": location_status,
                        "is_available": driver_status == "available",
                        "offline_reason": "location_stale"
                        if location_status == "stale"
                        else "",
                        "last_seen_seconds": last_seen_seconds,
                        "location_mode": location_mode,
                        "mission_id": fanout_mission_id,
                        "recorded_at": ts_str,
                        "received_at": received_at,
                    }
                    if tracking_event_id_str:
                        location_payload["tracking_event_id"] = tracking_event_id_str
                        live_state_payload["tracking_event_id"] = tracking_event_id_str

                    fanout_driver_location_update(
                        company_id_val,
                        location_payload,
                        live_state_payload,
                        accept_status=accept_status,
                    )

                    # Option A — miroir Kafka fire-and-forget (voie durable secondaire,
                    # opt-in). La voie live (canonical + fanout ci-dessus) reste la
                    # source du marqueur temps réel. Jamais bloquant ni d'exception.
                    if _SOCKET_KAFKA_MIRROR_ENABLED:
                        try:
                            from services.tracking import enqueue_tracking_event_nowait

                            _ff_payload: dict[str, Any] = {
                                "latitude": latitude,
                                "longitude": longitude,
                                "recorded_at": ts_str,
                                "timestamp": ts_str,
                                "location_mode": location_mode,
                            }
                            if tracking_event_id_str:
                                _ff_payload["tracking_event_id"] = tracking_event_id_str
                            if isinstance(company_id_val, int):
                                _ff_payload["company_id"] = company_id_val
                            enqueue_tracking_event_nowait(
                                driver_id=int(driver.id),
                                company_id=company_id_val
                                if isinstance(company_id_val, int)
                                else None,
                                source="socket_batch",
                                payload=_ff_payload,
                            )
                        except Exception:
                            logger.debug(
                                "[socket_batch] kafka mirror unavailable", exc_info=True
                            )

                    # ✅ P2: Incrémenter compteur de positions traitées avec succès
                    processed_count += 1
                    tracking_event_id_val = pos_with_ts.get("tracking_event_id")
                    if isinstance(tracking_event_id_val, str):
                        acked_tracking_event_ids.append(tracking_event_id_val)
                    seq_obj = pos_with_ts.get("sequence_id")
                    if isinstance(seq_obj, (int, str)):
                        with suppress(Exception):
                            seq = int(seq_obj)
                            ack_last_sequence_id = (
                                seq
                                if ack_last_sequence_id is None
                                else max(ack_last_sequence_id, seq)
                            )

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
                "tracking_event_ids": acked_tracking_event_ids,
            }
            if ack_last_sequence_id is not None:
                ack_response["ack_last_sequence_id"] = ack_last_sequence_id

            # ✅ P2: Bug #7 - Inclure rejets si présents
            if rejected_positions:
                ack_response["rejected"] = rejected_positions
                ack_response["rejected_count"] = len(rejected_positions)
            observe_batch_latency_seconds(seconds=time.perf_counter() - batch_t0)
            emit("driver_location_batch_ack", ack_response)
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
            sid_data = _get_sid_claims(sid)
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

    # =========================================================================
    # ÉTAPE 5/6: Handler join_institution pour le portail Institution
    # =========================================================================
    @socketio.on("join_institution")
    def handle_join_institution(data=None):  # noqa: ARG001
        """Permet à un utilisateur institution de rejoindre sa room.

        ÉTAPE 6: Les utilisateurs institution reçoivent les events:
        - request_sent
        - offer_accepted
        - request_converted
        - booking_status_updated
        """
        user_id_log: int | None = None
        institution_id_log: int | None = None
        user_public_id_log: str | None = None
        try:
            # ✅ SECURITY: Utiliser JWT depuis _SID_INDEX
            sid = _get_sid()
            sid_data = _get_sid_claims(sid)
            user_public_id = sid_data.get("user_public_id")
            user_role = sid_data.get("role")
            user_public_id_log = user_public_id

            if not user_public_id:
                emit("error", {"error": "Session JWT introuvable. Reconnectez-vous."})
                return

            # Vérifier que c'est un utilisateur institution
            if user_role != "institution":
                emit(
                    "error",
                    {
                        "error": "Seuls les utilisateurs institution peuvent rejoindre cette room."
                    },
                )
                return

            # Récupérer user depuis public_id
            user = User.query.filter_by(public_id=user_public_id).first()
            if not user:
                emit("error", {"error": "Utilisateur introuvable."})
                return

            user_id_log = user.id

            # Vérifier institution_id
            institution_id = getattr(user, "institution_id", None)
            if not institution_id:
                emit("error", {"error": "Institution non associée à cet utilisateur."})
                return

            institution_id_log = institution_id

            # Joindre la room institution
            room = f"institution_{institution_id}"
            join_room(room)
            # ✅ Tracking rooms
            ws_metrics.on_room_join(room)
            emit("joined_institution", {"institution_id": institution_id, "room": room})
            logger.info("🏥 Institution user %s joined room: %s", user.id, room)

        except Exception as e:
            _log_socketio_exception(
                exception=e,
                event_name="join_institution",
                user_id=user_id_log,
                additional_context={
                    "user_public_id": user_public_id_log,
                    "institution_id": institution_id_log,
                },
            )
            with suppress(Exception):
                emit(
                    "error",
                    {"error": "Erreur lors de la connexion à la room institution."},
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
            company_info = _get_sid_claims(sid)
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
                    # Try Redis first (canonical then legacy during migration).
                    h: Mapping[bytes, Any] = {}
                    redis_source: str | None = None
                    if (
                        redis_client
                    ):  # ✅ Vérification explicite pour satisfaire le linter
                        canonical_key = f"driver:{driver.id}:loc:canonical"
                        legacy_key = f"driver:{driver.id}:loc"
                        canonical_raw = redis_client.hgetall(canonical_key)
                        legacy_raw = (
                            redis_client.hgetall(legacy_key)
                            if not canonical_raw
                            else None
                        )
                        h_raw = canonical_raw or legacy_raw
                        if canonical_raw:
                            redis_source = "canonical"
                        elif legacy_raw:
                            redis_source = "legacy"
                        # Calme Pylance: redis-py retourne un dict[bytes, bytes]
                        h = cast("Mapping[bytes, Any]", h_raw)
                        last_seen_raw = redis_client.get(
                            f"driver:{driver.id}:last_seen"
                        )
                    else:
                        last_seen_raw = None

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

                        last_seen_str = None
                        if isinstance(last_seen_raw, bytes):
                            with suppress(Exception):
                                last_seen_str = last_seen_raw.decode()
                        elif isinstance(last_seen_raw, str):
                            last_seen_str = last_seen_raw
                        mission_status = _resolve_mission_status_for_driver(driver.id)
                        (
                            presence_status,
                            location_status,
                            offline_reason,
                            last_seen_seconds_redis,
                        ) = _compute_presence_from_signals(
                            location_mode=tcast(
                                "str | None", loc_data.get("location_mode")
                            ),
                            loc_data=loc_data,
                            last_seen_ts=last_seen_str,
                        )
                        status = _resolve_driver_status(
                            mission_status=mission_status,
                            is_active=bool(getattr(driver, "is_active", True)),
                            presence_status=presence_status,
                        )
                        ts_val = loc_data.get("ts") or datetime.now(UTC).isoformat()
                        from services.realtime.socketio import (
                            fanout_driver_location_update,
                        )

                        fanout_accept_status = (
                            "accepted_canonical"
                            if redis_source == "canonical"
                            else "accepted_observability_only"
                        )
                        fanout_driver_location_update(
                            company_id,
                            {
                                "driver_id": driver.id,
                                "company_id": company_id,
                                "first_name": getattr(
                                    getattr(driver, "user", None), "first_name", None
                                ),
                                "latitude": loc_data.get("lat"),
                                "longitude": loc_data.get("lon"),
                                "timestamp": ts_val,
                                "received_at": loc_data.get("received_at"),
                                "last_seen_seconds": last_seen_seconds_redis,
                            },
                            {
                                "driver_id": driver.id,
                                "company_id": company_id,
                                "lat": loc_data.get("lat"),
                                "lng": loc_data.get("lon"),
                                "timestamp": ts_val,
                                "status": status,
                                "mission_status": mission_status,
                                "presence_status": presence_status,
                                "location_status": location_status,
                                "is_available": status == "available",
                                "offline_reason": offline_reason,
                                "received_at": loc_data.get("received_at"),
                                "last_seen_seconds": last_seen_seconds_redis,
                            },
                            accept_status=fanout_accept_status,
                        )
                    elif (driver.latitude is not None) and (
                        driver.longitude is not None
                    ):
                        # Fallback to DB if Redis doesnt have data — P2: fanout unifié
                        mission_status = _resolve_mission_status_for_driver(driver.id)
                        status = _resolve_driver_status(
                            mission_status=mission_status,
                            is_active=bool(getattr(driver, "is_active", True)),
                            presence_status="degraded",
                        )
                        ts_val = datetime.now(UTC).isoformat()
                        from services.realtime.socketio import (
                            fanout_driver_location_update,
                        )

                        fanout_driver_location_update(
                            company_id,
                            {
                                "driver_id": driver.id,
                                "company_id": company_id,
                                "first_name": getattr(
                                    getattr(driver, "user", None), "first_name", None
                                ),
                                "latitude": driver.latitude,
                                "longitude": driver.longitude,
                                "timestamp": ts_val,
                                "received_at": ts_val,
                            },
                            {
                                "driver_id": driver.id,
                                "company_id": company_id,
                                "lat": driver.latitude,
                                "lng": driver.longitude,
                                "timestamp": ts_val,
                                "status": status,
                                "mission_status": mission_status,
                                "presence_status": "degraded",
                                "location_status": "stale",
                                "is_available": status == "available",
                                "offline_reason": "location_stale",
                                "received_at": ts_val,
                            },
                            accept_status="accepted_observability_only",
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
            info = delete_sid_claims(sid) or _SID_INDEX.pop(sid, None)
            trace_id = request.headers.get("X-Trace-ID") or request.headers.get(
                "Trace-Id"
            )
            now = datetime.now(UTC)

            company_id = info.get("company_id") if info else None
            user_id = info.get("user_id") if info else None
            driver_id = info.get("driver_id") if info else None
            role = info.get("role") if info else None
            if sid and user_id and role:
                remove_presence(sid=sid, user_id=int(user_id), role=str(role))
            if company_id and role in {"driver", "company"}:
                _release_company_slot(int(company_id))

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

            # ✅ ÉTAPE 5/6: Gestion déconnexion institution
            elif role == "institution":
                institution_id = info.get("institution_id") if info else None
                if institution_id:
                    institution_room = f"institution_{institution_id}"
                    try:
                        ws_metrics.on_room_leave(institution_room)
                    except (ConnectionError, OSError) as e:
                        if getattr(e, "errno", None) != ERRNO_BAD_FILE_DESCRIPTOR:
                            logger.warning(
                                "Error leaving room for institution (network error: %s): %s",
                                type(e).__name__,
                                e,
                            )
                    except Exception as e:
                        logger.warning(
                            "Unexpected error leaving room for institution: %s", e
                        )

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
            sid_data = _get_sid_claims(sid)
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
            sid_data = _get_sid_claims(sid)
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
