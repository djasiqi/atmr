# backend/services/push_service.py
"""Service de push notifications Expo.

Module séparé pour éviter les cycles d'import avec notification_service et socketio_service.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Dict, cast

import requests
from requests import RequestException, Timeout
from requests.exceptions import ConnectionError as RequestsConnectionError

from ext import app_logger, redis_client

# Import conditionnel pour métriques Prometheus
try:
    from services.monitoring.prometheus import (
        track_push_notification,
        track_push_rate_limit_hit,
        track_push_token_invalidated,
    )
except ImportError:
    # Si prometheus_metrics non disponible, fonction no-op
    def track_push_notification(*args, **kwargs):
        """No-op si Prometheus non disponible."""

    def track_push_rate_limit_hit(*args, **kwargs):
        """No-op si Prometheus non disponible."""

    def track_push_token_invalidated(*args, **kwargs):
        """No-op si Prometheus non disponible."""


# Configuration retry
MAX_RETRY_ATTEMPTS = 5
INITIAL_RETRY_DELAY = 1  # secondes
MAX_RETRY_DELAY = 8  # secondes
TOKEN_DISPLAY_LENGTH = 20  # Longueur du token à afficher dans les logs
TOKEN_MASK_LENGTH = 10  # Longueur du token à garder pour masquage
BODY_PREVIEW_LENGTH = 100  # Longueur du body à afficher dans les logs

# Configuration rate limiting
PUSH_RATE_LIMIT_PER_MINUTE = 10  # Max 10 pushes/minute par driver
PUSH_RATE_LIMIT_WINDOW = 60  # Fenêtre de 60 secondes

# Configuration circuit breaker
CIRCUIT_BREAKER_THRESHOLD = 10  # Nombre d'échecs avant d'ouvrir le circuit
CIRCUIT_BREAKER_WINDOW = 300  # Fenêtre de 5 minutes
CIRCUIT_BREAKER_COOLDOWN = 60  # Cooldown de 60 secondes avant de réessayer

# Configuration déduplication
DEDUP_WINDOW = 300  # ✅ CORRECTIF: Fenêtre de déduplication de 5 minutes (300s) pour couvrir retries Celery


def _safe_int(val: Any) -> int:
    """Convertit une valeur Redis (bytes|str|None) en int pour le circuit breaker."""
    if val is None or val == 0:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _check_circuit_breaker() -> tuple[bool, str | None]:
    """Vérifie si le circuit breaker est ouvert (trop d'échecs Expo Push).

    ✅ CORRECTIF #5: Sépare les erreurs réseau des tokens invalides.
    Les tokens invalides ne doivent pas bloquer tous les envois.

    Returns:
        Tuple (is_open, reason):
            - is_open: True si le circuit est ouvert (arrêt temporaire)
            - reason: Raison de l'ouverture si applicable
    """
    if not redis_client:
        return False, None  # Pas de circuit breaker sans Redis

    try:
        circuit_key = "push:circuit_breaker:expo"
        failure_count_key = "push:circuit_breaker:failures"
        token_invalid_count_key = "push:circuit_breaker:token_invalid"

        # Vérifier si le circuit est ouvert
        is_open = redis_client.get(circuit_key)
        if is_open:
            ttl = redis_client.ttl(circuit_key)
            return True, f"Circuit breaker ouvert (réessai dans {ttl}s)"

        # ✅ CORRECTIF #5: Séparer erreurs réseau des tokens invalides
        failure_count = _safe_int(redis_client.get(failure_count_key))
        token_invalid_count = _safe_int(redis_client.get(token_invalid_count_key))

        # Ne compter que les erreurs réseau pour le circuit breaker
        # Les tokens invalides ne doivent pas bloquer tous les envois
        network_failures = failure_count - token_invalid_count

        if network_failures >= CIRCUIT_BREAKER_THRESHOLD:
            redis_client.setex(circuit_key, CIRCUIT_BREAKER_COOLDOWN, "1")
            app_logger.error(
                "[push] Circuit breaker OUVERT : %d erreurs réseau en %ds",
                network_failures,
                CIRCUIT_BREAKER_WINDOW,
            )
            return (
                True,
                f"Circuit breaker ouvert après {network_failures} erreurs réseau",
            )

        return False, None
    except Exception as e:
        app_logger.warning(
            "[push] Circuit breaker check failed: %s. Continuing without breaker.",
            e,
        )
        return False, None  # Fail-open en cas d'erreur


def _record_push_failure(token_invalid: bool = False) -> None:
    """Enregistre un échec de push pour le circuit breaker.

    Args:
        token_invalid: Si True, enregistre comme token invalide (ne compte pas pour circuit breaker)
    """
    if not redis_client:
        return

    try:
        failure_count_key = "push:circuit_breaker:failures"
        token_invalid_count_key = "push:circuit_breaker:token_invalid"

        # Toujours incrémenter le compteur total (pour statistiques)
        count = redis_client.incr(failure_count_key)
        if count == 1:
            # Première erreur de la fenêtre : définir expiration
            redis_client.expire(failure_count_key, CIRCUIT_BREAKER_WINDOW)

        # ✅ CORRECTIF #5: Séparer les tokens invalides des erreurs réseau
        if token_invalid:
            token_invalid_count = redis_client.incr(token_invalid_count_key)
            if token_invalid_count == 1:
                redis_client.expire(token_invalid_count_key, CIRCUIT_BREAKER_WINDOW)
    except Exception:
        pass  # Silent fail pour ne pas bloquer l'envoi


def _record_push_success() -> None:
    """Enregistre un succès de push (réinitialise le compteur d'échecs)."""
    if not redis_client:
        return

    try:
        failure_count_key = "push:circuit_breaker:failures"
        token_invalid_count_key = "push:circuit_breaker:token_invalid"
        circuit_key = "push:circuit_breaker:expo"

        # Réinitialiser les compteurs d'échecs après un succès
        redis_client.delete(failure_count_key)
        redis_client.delete(token_invalid_count_key)

        # Si le circuit était ouvert, le fermer
        if redis_client.get(circuit_key):
            redis_client.delete(circuit_key)
            app_logger.info("[push] Circuit breaker FERMÉ après succès")
    except Exception:
        pass  # Silent fail


def _check_duplicate_notification(  # ✅ CORRECTIF: Simple underscore = publique dans le module  # pyright: ignore[reportUnusedFunction]
    driver_id: int,
    title: str,
    body: str,
    notification_type: str,
) -> bool:
    """Vérifie si une notification identique a déjà été envoyée récemment.

    Note: Cette fonction est utilisée par fanout.py via import dynamique,
    donc elle peut apparaître comme non utilisée dans l'analyse statique.

    Args:
        driver_id: ID du driver
        title: Titre de la notification
        body: Corps de la notification
        notification_type: Type de notification (booking, message, etc.)

    Returns:
        True si c'est un doublon (ne pas envoyer), False sinon
    """
    if not redis_client:
        return False  # Pas de déduplication sans Redis

    try:
        # Créer un hash du contenu de la notification
        content = f"{driver_id}:{notification_type}:{title}:{body}"
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        dedup_key = f"push:dedup:{driver_id}:{content_hash}"

        # SET NX EX atomique — évite race GET+SETEX sous charge multi-worker
        was_set = redis_client.set(dedup_key, "1", nx=True, ex=DEDUP_WINDOW)
        if not was_set:
            app_logger.debug(
                "[push] Notification dupliquée détectée pour driver %s (type: %s)",
                driver_id,
                notification_type,
            )
            return True  # C'est un doublon
        return False
    except Exception as e:
        app_logger.warning(
            "[push] Deduplication check failed: %s. Continuing without dedup.",
            e,
        )
        return False  # Fail-open en cas d'erreur


# Push token invalidation is handled via DeviceToken.is_active = False
# (see services/events/fanout.py and tasks/notification_tasks.py).
# No legacy invalidation path (driver.push_token) — DeviceToken is the source of truth.


def _log_fcm_pipeline_result(
    result: Dict[str, Any],
    *,
    driver_id: int | None,
    data: Dict[str, Any] | None,
    correlation_id: str | None,
    notification_type: str | None = None,
) -> None:
    """Journalise le résultat FCM dans le pipeline observabilité."""
    try:
        from services.notifications.notification_pipeline_observability import (
            log_notification_pipeline_event,
        )

        payload = data or {}
        ntype = notification_type or payload.get("type") or "unknown"
        booking_id = payload.get("booking_id") or payload.get("mission_id")
        notification_id = payload.get("event_id") or payload.get("trace_id")
        event = "notification_fcm_sent" if result.get("ok") else "notification_fcm_failed"
        log_notification_pipeline_event(
            event,
            notification_id=notification_id,
            booking_id=booking_id,
            driver_id=driver_id,
            notification_type=str(ntype),
            correlation_id=correlation_id or payload.get("correlation_id"),
            error=result.get("error"),
            token_invalid=bool(result.get("token_invalid")),
        )
    except Exception:
        pass


def send_push_message(
    token: str,
    title: str,
    body: str,
    data: Dict[str, Any] | None = None,
    *,
    timeout: int = 5,
    use_retry: bool = True,
    driver_id: int | None = None,
    bypass_rate_limit: bool = False,
    correlation_id: str | None = None,
    provider: str | None = None,
    platform: str | None = None,
    device_token_id: int | None = None,
) -> Dict[str, Any]:
    """Envoie une notification push (FCM natif ou Expo Push fallback).

    Routing:
    - provider="fcm" + platform="android" → data-only via firebase-admin
    - provider="fcm" + platform="ios" → notification+data via firebase-admin
    - provider="expo" / ExponentPushToken → Expo Push API (legacy)
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    # Auto-detect provider from token format
    if provider is None:
        provider = "expo" if token.startswith("ExponentPushToken") else "fcm"

    # FCM native path
    if provider == "fcm":
        from services.notifications.firebase_push import send_fcm_android, send_fcm_ios

        app_logger.info(
            "[push] FCM native send (platform=%s, driver=%s)",
            platform,
            driver_id,
            extra={"correlation_id": correlation_id},
        )
        if platform == "ios":
            result = send_fcm_ios(token, title, body, data)
        else:
            result = send_fcm_android(token, title, body, data)
        if device_token_id is not None:
            try:
                from services.notifications.device_token_lifecycle import (
                    apply_push_result_to_device_token,
                )

                apply_push_result_to_device_token(device_token_id, result)
            except Exception as e:
                app_logger.warning(
                    "[push] device_token lifecycle update failed (ignored): %s",
                    str(e)[:200],
                )
        _log_fcm_pipeline_result(
            result,
            driver_id=driver_id,
            data=data,
            correlation_id=correlation_id,
        )
        return result

    # Expo Push API path (legacy fallback)
    notification_type = data.get("type") if data else None

    # ✅ INSTRUMENTATION: Token hash pour anonymisation
    token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
    token_preview = (
        token[:TOKEN_DISPLAY_LENGTH] + "..."
        if len(token) > TOKEN_DISPLAY_LENGTH
        else token
    )

    app_logger.info(
        "[push] Sending push notification",
        extra={
            "correlation_id": correlation_id,
            "driver_id": driver_id,
            "token_preview": token_preview,
            "token_hash": token_hash,
            "notification_type": notification_type,
        },
    )

    # ✅ AMÉLIORATION: Circuit breaker pour éviter les storms
    is_open, reason = _check_circuit_breaker()
    if is_open:
        app_logger.warning(
            "[push] Push skipped: %s",
            reason,
            extra={"correlation_id": correlation_id},
        )
        return {"ok": False, "error": reason, "circuit_breaker_open": True}

    # Utiliser retry par défaut pour améliorer la robustesse
    if use_retry:
        return send_push_message_with_retry(
            token=token,
            title=title,
            body=body,
            data=data,
            timeout=timeout,
            driver_id=driver_id,
            bypass_rate_limit=bypass_rate_limit,
            correlation_id=correlation_id,
            provider=provider,
            platform=platform,
            device_token_id=device_token_id,
        )

    # Priorité haute par défaut pour que FCM livre immédiatement (Doze bypass).
    # Seuls les types purement informatifs ou silencieux restent en "default".
    LOW_PRIORITY_TYPES = {
        "info",
        "stats",
        "dispatch_completed",
        "silent_update",
        "missions_preload",
        "profile_sync",
        "maps_precache",
        "config_update",
    }
    priority = "default" if notification_type in LOW_PRIORITY_TYPES else "high"

    # P0: channelId à la RACINE du message Expo (pas seulement dans data)
    # Si channelId dans data mais pas à la racine, l'OS Android peut ne pas afficher
    # quand l'app est killed (Expo docs: channelId = champ racine Android-only)
    data_dict = data or {}
    channel_id = data_dict.get("channelId", "missions_v2")
    trace_id = data_dict.get("trace_id")

    message: Dict[str, Any] = {
        "to": token,
        "sound": "default",
        "title": title,
        "body": body,
        "data": data_dict,
        "priority": priority,
    }
    # Android: channelId à la racine pour affichage quand app killed
    if channel_id:
        message["channelId"] = channel_id

    # P2: Proof log — payload sanitized (sans token) pour diagnostic app killed
    app_logger.info(
        "[push] PUSH_PROOF payload (sanitized) title=%r body_len=%d priority=%s channelId=%s trace_id=%s",
        title[:50] if title else "",
        len(body or ""),
        priority,
        channel_id,
        trace_id or "-",
        extra={"correlation_id": correlation_id},
    )

    result: Dict[str, Any] = {"ok": False, "error": "Unknown error"}
    try:
        resp = requests.post(
            "https://exp.host/--/api/v2/push/send", json=message, timeout=timeout
        )
        resp.raise_for_status()
        response_data = cast(Dict[str, Any], resp.json())
        # ✅ Normaliser la réponse Expo Push : vérifier si "data" contient des erreurs
        # Expo Push retourne {"data": [{"status": "ok", ...}]} en cas de succès
        if "data" in response_data and isinstance(response_data["data"], list):
            # Vérifier si tous les tickets ont status "ok"
            all_ok = all(
                ticket.get("status") == "ok" for ticket in response_data["data"]
            )
            if all_ok:
                result = {"ok": True, "data": response_data.get("data")}
                _record_push_success()  # ✅ Enregistrer succès pour circuit breaker

                # ✅ P2: PUSH_PROOF ticket — pour diagnostic app killed (corrélation receipts)
                for ticket in response_data["data"]:
                    tid = ticket.get("id")
                    app_logger.info(
                        "[push] PUSH_PROOF ticket status=ok id=%s correlation_id=%s → fetch receipt: python -m scripts.fetch_expo_receipts %s",
                        tid,
                        correlation_id,
                        tid or "",
                        extra={"correlation_id": correlation_id, "expo_ticket_id": tid},
                    )
            else:
                # Au moins un ticket a échoué
                errors = []
                token_is_invalid = False
                for ticket in response_data["data"]:
                    tid = ticket.get("id")
                    tstatus = ticket.get("status")
                    tmsg = ticket.get("message", "")
                    tdetails = ticket.get("details", {}) or {}
                    # ✅ P2: PUSH_PROOF ticket — erreur immédiate (pas besoin receipt)
                    app_logger.warning(
                        "[push] PUSH_PROOF ticket status=error id=%s message=%s details=%s correlation_id=%s",
                        tid,
                        tmsg,
                        tdetails.get("error", "-"),
                        correlation_id,
                        extra={"correlation_id": correlation_id, "expo_ticket_id": tid},
                    )

                    if tstatus != "ok":
                        error_msg = ticket.get("message", "Unknown error")
                        error_type = ticket.get("details", {}).get("error")
                        errors.append(error_msg)

                        # ✅ AMÉLIORATION: Détecter les tokens invalides/expirés
                        # Expo renvoie ces erreurs quand le token n'est plus valide
                        if error_type in [
                            "DeviceNotRegistered",
                            "InvalidCredentials",
                            "MessageTooBig",
                            "MessageRateExceeded",
                        ]:
                            token_is_invalid = True
                            app_logger.warning(
                                "[push] Token invalide détecté: %s (error: %s)",
                                token[:TOKEN_DISPLAY_LENGTH]
                                if len(token) > TOKEN_DISPLAY_LENGTH
                                else token,
                                error_type,
                                extra={"correlation_id": correlation_id},
                            )
                            # ✅ INSTRUMENTATION: Métrique Prometheus pour token invalide
                            reason_map = {
                                "DeviceNotRegistered": "device_not_registered",
                                "InvalidCredentials": "invalid_credentials",
                                "MessageTooBig": "message_too_big",
                                "MessageRateExceeded": "rate_exceeded",
                            }
                            track_push_token_invalidated(
                                reason=reason_map.get(error_type, "unknown")
                            )

                result = {
                    "ok": False,
                    "error": "; ".join(errors),
                    "token_invalid": token_is_invalid,  # Flag pour invalidation
                }
                # ✅ CORRECTIF #5: Enregistrer séparément selon le type d'erreur
                _record_push_failure(token_invalid=token_is_invalid)
        else:
            # Format inattendu mais pas d'erreur HTTP
            result = {"ok": True, "data": response_data}
    except (RequestException, Timeout, RequestsConnectionError) as e:
        # Erreurs réseau attendues : connexion HTTP, timeout
        app_logger.warning(
            "[push] Expo push failed (network error: %s): %s",
            type(e).__name__,
            e,
            extra={"correlation_id": correlation_id},
        )
        result = {"ok": False, "error": str(e)}
        # ✅ CORRECTIF #5: Erreur réseau (pas token invalide)
        _record_push_failure(token_invalid=False)
    except (ValueError, TypeError, KeyError) as e:
        # Erreurs de validation attendues : JSON invalide
        app_logger.warning(
            "[push] Expo push failed (validation error: %s): %s",
            type(e).__name__,
            e,
            extra={"correlation_id": correlation_id},
        )
        result = {"ok": False, "error": str(e)}
    except Exception:
        # Erreur inattendue : logger avec trace complète
        app_logger.exception(
            "[push] Expo push failed", extra={"correlation_id": correlation_id}
        )
        result = {"ok": False, "error": "Internal error"}

    # ✅ INSTRUMENTATION: Log résultat avec correlation_id
    app_logger.info(
        "[push] Push notification result",
        extra={
            "correlation_id": correlation_id,
            "ok": result.get("ok"),
            "error": result.get("error"),
            "token_invalid": result.get("token_invalid"),
        },
    )

    if device_token_id is not None:
        try:
            from services.notifications.device_token_lifecycle import (
                apply_push_result_to_device_token,
            )

            apply_push_result_to_device_token(device_token_id, result)
        except Exception as e:
            app_logger.warning(
                "[push] device_token lifecycle update failed (ignored): %s",
                str(e)[:200],
            )

    return result


def _calculate_retry_delay(attempt: int) -> float:
    """Calcule le délai de retry avec exponential backoff.

    Args:
        attempt: Numéro de la tentative (1-indexed)

    Returns:
        Délai en secondes (1, 2, 4, 8, max 8)
    """
    delay = min(INITIAL_RETRY_DELAY * (2 ** (attempt - 1)), MAX_RETRY_DELAY)
    return float(delay)


def send_push_message_with_retry(
    token: str,
    title: str,
    body: str,
    data: Dict[str, Any] | None = None,
    *,
    timeout: int = 5,
    max_retries: int = MAX_RETRY_ATTEMPTS,
    retry_on_network_error: bool = True,
    driver_id: int | None = None,
    bypass_rate_limit: bool = False,
    correlation_id: str | None = None,
    provider: str | None = None,
    platform: str | None = None,
    device_token_id: int | None = None,
) -> Dict[str, Any]:
    """Envoie une notification push avec retry automatique.

    Args:
        token: Token Expo Push du destinataire
        title: Titre de la notification
        body: Corps du message
        data: Données additionnelles (pour deep linking, etc.)
        timeout: Timeout en secondes (défaut: 5)
        max_retries: Nombre maximum de tentatives (défaut: 5)
        retry_on_network_error: Si True, retry sur erreurs réseau uniquement (défaut: True)
        driver_id: ID du chauffeur pour rate limiting (optionnel)
        bypass_rate_limit: Si True, contourne le rate limiting (pour alertes urgentes)
        provider/platform/device_token_id: alignés sur ``send_push_message`` (lifecycle Expo)

    Returns:
        Dict avec "ok" (bool), "error" (str) ou "data", et "attempts" (int)
    """
    # ✅ Rate limiting global par driver (sauf si bypass activé)
    if driver_id is not None and not bypass_rate_limit and redis_client:
        rate_limit_key = f"push:rate_limit:driver:{driver_id}"
        try:
            count_result = redis_client.incr(rate_limit_key)
            # Convertir en int (redis.incr retourne int ou ResponseT selon version)
            count = count_result if isinstance(count_result, int) else 0
            if count == 1:
                # Première push de la fenêtre : définir expiration
                redis_client.expire(rate_limit_key, PUSH_RATE_LIMIT_WINDOW)
            if count > PUSH_RATE_LIMIT_PER_MINUTE:
                app_logger.warning(
                    "[push] Rate limit exceeded for driver %s (%d pushes in %d seconds)",
                    driver_id,
                    count,
                    PUSH_RATE_LIMIT_WINDOW,
                )
                # ✅ INSTRUMENTATION: Métrique Prometheus pour rate limit hit
                track_push_rate_limit_hit(driver_id)
                return {
                    "ok": False,
                    "error": "Rate limit exceeded",
                    "rate_limit_exceeded": True,
                }
        except Exception as e:
            # Si Redis indisponible, logger et continuer (fail-open)
            app_logger.warning(
                "[push] Rate limit check failed (Redis error): %s. Continuing without rate limit.",
                e,
            )

    # ✅ INSTRUMENTATION: Générer correlation_id une seule fois pour tous les retries
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    start_time = time.time()
    event_type = data.get("type", "unknown") if data else "unknown"
    last_error: str | None = None
    last_result: Dict[str, Any] | None = None

    for attempt in range(1, max_retries + 1):
        # ✅ FIX CRITIQUE: Passer use_retry=False pour éviter la récursion infinie
        # send_push_message_with_retry gère déjà les retries, donc on ne doit pas
        # appeler send_push_message avec use_retry=True
        result = send_push_message(
            token=token,
            title=title,
            body=body,
            data=data,
            timeout=timeout,
            correlation_id=correlation_id,
            use_retry=False,  # ✅ Éviter la récursion infinie
            driver_id=driver_id,
            bypass_rate_limit=bypass_rate_limit,
            provider=provider,
            platform=platform,
            device_token_id=device_token_id,
        )

        if result.get("ok"):
            if attempt > 1:
                app_logger.info(
                    "[push] Push succeeded after %d attempts (token: %s...)",
                    attempt,
                    token[:TOKEN_DISPLAY_LENGTH]
                    if len(token) > TOKEN_DISPLAY_LENGTH
                    else token,
                )
            # ✅ Tracking métriques Prometheus
            latency = time.time() - start_time
            track_push_notification(
                status="success",
                event_type=event_type,
                latency_seconds=latency,
                attempts=attempt,
            )
            final_result = dict(result)
            final_result["attempts"] = attempt
            return final_result

        # Échec : déterminer si on doit retry
        last_error = result.get("error", "Unknown error")
        last_result = result

        # Ne pas retry sur erreurs de validation (erreurs définitives)
        if not retry_on_network_error:
            break

        # Retry uniquement sur erreurs réseau/timeout
        error_str = str(last_error).lower()
        is_retryable = any(
            keyword in error_str
            for keyword in [
                "timeout",
                "connection",
                "network",
                "requestexception",
                "connectionerror",
            ]
        )

        if not is_retryable:
            app_logger.debug(
                "[push] Non-retryable error (attempt %d/%d): %s",
                attempt,
                max_retries,
                last_error,
            )
            break

        # Dernière tentative : ne pas retry
        if attempt >= max_retries:
            break

        # Calculer délai avec exponential backoff
        delay = _calculate_retry_delay(attempt)
        app_logger.warning(
            "[push] Push failed (attempt %d/%d): %s. Retrying in %.1fs...",
            attempt,
            max_retries,
            last_error,
            delay,
        )

        time.sleep(delay)

    # Toutes les tentatives ont échoué
    app_logger.error(
        "[push] Push failed after %d attempts (token: %s...): %s",
        max_retries,
        token[:TOKEN_DISPLAY_LENGTH] if len(token) > TOKEN_DISPLAY_LENGTH else token,
        last_error,
    )

    # Retourner le dernier résultat avec info sur les tentatives
    final_result = (
        dict(last_result)
        if last_result
        else {"ok": False, "error": last_error or "Unknown error"}
    )
    final_result["attempts"] = max_retries
    final_result["final_error"] = last_error

    # ✅ Tracking métriques Prometheus (échec)
    latency = time.time() - start_time
    track_push_notification(
        status="failed",
        event_type=event_type,
        latency_seconds=latency,
        attempts=max_retries,
    )

    # Persister l'échec définitif pour analyse (dead letter queue)
    _persist_failed_push(token, title, body, data, last_error, max_retries)

    return final_result


def _persist_failed_push(
    token: str,
    title: str,
    body: str,
    data: Dict[str, Any] | None,
    error: str | None,
    attempts: int,
) -> None:
    """Persiste un échec définitif de push notification pour analyse.

    Cette fonction log les échecs dans un format structuré pour analyse ultérieure.
    Pour une vraie dead letter queue, on pourrait utiliser Redis ou une table DB.

    Args:
        token: Token Expo Push (partiellement masqué pour sécurité)
        title: Titre de la notification
        body: Corps du message
        data: Données additionnelles
        error: Message d'erreur
        attempts: Nombre de tentatives effectuées
    """
    # Masquer le token pour la sécurité (garder seulement les premiers caractères)
    masked_token = (
        token[:TOKEN_MASK_LENGTH] + "..." if len(token) > TOKEN_MASK_LENGTH else token
    )

    # Log structuré pour analyse
    app_logger.error(
        "[push] Dead letter queue entry - Push failed definitively",
        extra={
            "token_preview": masked_token,
            "title": title,
            "body_preview": (
                body[:BODY_PREVIEW_LENGTH] if len(body) > BODY_PREVIEW_LENGTH else body
            ),
            "data_type": data.get("type") if data else None,
            "error": error,
            "attempts": attempts,
            "timestamp": time.time(),
        },
    )
