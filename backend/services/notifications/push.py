# backend/services/push_service.py
"""Service de push notifications Expo.

Module séparé pour éviter les cycles d'import avec notification_service et socketio_service.
"""

from __future__ import annotations

import time
from typing import Any, Dict, cast

import requests  # pyright: ignore[reportMissingModuleSource]
from requests import (  # pyright: ignore[reportMissingModuleSource]
    RequestException,
    Timeout,
)
from requests.exceptions import (  # pyright: ignore[reportMissingModuleSource]
    ConnectionError as RequestsConnectionError,
)

from ext import app_logger, redis_client

# Import conditionnel pour métriques Prometheus
try:
    from services.monitoring.prometheus import track_push_notification
except ImportError:
    # Si prometheus_metrics non disponible, fonction no-op
    def track_push_notification(*args, **kwargs):
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
) -> Dict[str, Any]:
    """Envoie une notification push via Expo Push Notification Service.

    Args:
        token: Token Expo Push du destinataire
        title: Titre de la notification
        body: Corps du message
        data: Données additionnelles (pour deep linking, etc.)
        timeout: Timeout en secondes (défaut: 5)
        use_retry: Si True, utilise le retry automatique (défaut: True)
        driver_id: ID du chauffeur pour rate limiting (optionnel)
        bypass_rate_limit: Si True, contourne le rate limiting (pour alertes urgentes)

    Returns:
        Dict avec "ok" (bool) et "error" (str) ou "data" selon le résultat.
        Si use_retry=True, contient aussi "attempts" (int) et "final_error" (str) en cas d'échec.
    """
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
        )

    message = {
        "to": token,
        "sound": "default",
        "title": title,
        "body": body,
        "data": data or {},
    }
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
            else:
                # Au moins un ticket a échoué
                errors = [
                    ticket.get("message", "Unknown error")
                    for ticket in response_data["data"]
                    if ticket.get("status") != "ok"
                ]
                result = {"ok": False, "error": "; ".join(errors)}
        else:
            # Format inattendu mais pas d'erreur HTTP
            result = {"ok": True, "data": response_data}
    except (RequestException, Timeout, RequestsConnectionError) as e:
        # Erreurs réseau attendues : connexion HTTP, timeout
        app_logger.warning(
            "[push] Expo push failed (network error: %s): %s",
            type(e).__name__,
            e,
        )
        result = {"ok": False, "error": str(e)}
    except (ValueError, TypeError, KeyError) as e:
        # Erreurs de validation attendues : JSON invalide
        app_logger.warning(
            "[push] Expo push failed (validation error: %s): %s",
            type(e).__name__,
            e,
        )
        result = {"ok": False, "error": str(e)}
    except Exception:
        # Erreur inattendue : logger avec trace complète
        app_logger.exception("[push] Expo push failed")
        result = {"ok": False, "error": "Internal error"}

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

    start_time = time.time()
    event_type = data.get("type", "unknown") if data else "unknown"
    last_error: str | None = None
    last_result: Dict[str, Any] | None = None

    for attempt in range(1, max_retries + 1):
        result = send_push_message(
            token=token,
            title=title,
            body=body,
            data=data,
            timeout=timeout,
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
