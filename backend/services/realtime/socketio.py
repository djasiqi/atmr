# backend/services/socketio_service.py
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Dict, cast

from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import joinedload

from ext import app_logger, socketio
from schemas.socket_events import EVENT_VERSION, SocketEvent
from services.monitoring.driver_location_metrics import inc_fanout

if TYPE_CHECKING:
    from models import Booking

from models import Driver
from repositories.driver_repository import DriverRepository

# ---------------------------------------------------------------------------
# Constantes simples
# ---------------------------------------------------------------------------
DEFAULT_NAMESPACE = "/"

# Annexe F — alias optionnels lirie.* (double emit si LIRIE_WS_V2_ALIASES=1)
_LIRIE_WS_V2_ALIASES = os.getenv("LIRIE_WS_V2_ALIASES", "").lower() in ("1", "true", "yes")
_LIRIE_EVENT_ALIASES: dict[str, str] = {
    "driver_location_update": "lirie.driver.map.updated",
    "driver_live_state_update": "lirie.driver.state.updated",
    "booking_updated": "lirie.reservation.updated",
    "dispatch_delay_detected": "lirie.dispatch.delay.updated",
    "dispatch_assignment": "lirie.dispatch.assignment.updated",
}

_LIRIE_DISPATCH_WS_UNIFIED = os.getenv("LIRIE_DISPATCH_WS_UNIFIED", "").lower() in (
    "1",
    "true",
    "yes",
)
_LIRIE_DISPATCH_DASHBOARD_WS = os.getenv("LIRIE_DISPATCH_DASHBOARD_WS", "").lower() in (
    "1",
    "true",
    "yes",
)
_DISPATCH_DB_THROTTLE_LAST: dict[int, float] = {}
_DISPATCH_DB_THROTTLE_SEC = 2.0


def _emit_alias_no_metrics(
    event: str,
    payload: dict[str, Any],
    *,
    room: str,
    namespace: str,
) -> None:
    """Émet un alias lirie.* sans compter deux fois les métriques Prometheus."""
    try:
        from services.monitoring.lirie_prometheus import inc_socketio_alias_emit

        inc_socketio_alias_emit(event)
    except Exception:
        pass
    try:
        kwargs: dict[str, Any] = {"namespace": namespace, "to": room}
        cast("Any", socketio).emit(event, payload, **kwargs)
    except TypeError:
        try:
            kwargs = {"namespace": namespace, "room": room}
            cast("Any", socketio).emit(event, payload, **kwargs)
        except Exception:
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers de rooms (source de vérité unique)
# ---------------------------------------------------------------------------
def get_company_room(company_id: int) -> str:
    """Room d'entreprise (ex: company_42)."""
    return f"company_{company_id}"


def get_driver_room(driver_id: int) -> str:
    """Room personnelle d'un chauffeur (ex: driver_101)."""
    return f"driver_{driver_id}"


def get_date_room(date_str: str) -> str:
    """Room par date locale 'YYYY-MM-DD' (ex: date_2025-09-20)."""
    return f"date_{date_str}"


# ---------------------------------------------------------------------------
# Garde-fous utilitaires
# ---------------------------------------------------------------------------
def _is_jsonable(x: Any) -> bool:
    """Vérifie qu'un payload est sérialisable en JSON
    (évite les plantages silencieux).
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, ValueError):
        # Erreurs de sérialisation JSON attendues : types non sérialisables
        return False
    except Exception:
        # Erreur inattendue lors de la sérialisation JSON
        return False


# ---------------------------------------------------------------------------
# Helper pour enrichir payload avec event_id si absent
# ---------------------------------------------------------------------------
def _enrich_payload_if_needed(
    payload: dict[str, Any], event_name: str
) -> dict[str, Any]:
    """Enrichit un payload avec event_id, version, timestamp si absents.

    Utilise le schéma centralisé SocketEvent pour garantir la cohérence.

    Args:
        payload: Payload d'événement (peut déjà contenir event_id)
        event_name: Nom de l'événement Socket.IO (ex: "new_booking", "dispatch:run:started")

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


# ---------------------------------------------------------------------------
# Émission thread-safe (depuis handlers HTTP, workers, threads…)
# - Flask-SocketIO >= 5: 'to='
# - Compat anciennes versions: 'room='
# ---------------------------------------------------------------------------
def _safe_emit(
    event: str,
    payload: dict[str, Any],
    *,
    room: str | None = None,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    """Émet un événement Socket.IO de façon sûre:
    - exige une room (sinon log d'erreur),
    - vérifie la sérialisabilité JSON,
    - gère la compatibilité Flask-SocketIO v4/v5,
    - ne remonte pas d'exception aux appelants.
    """
    if room is None:
        app_logger.error("[socketio] _safe_emit sans room: event=%s", event)
        return

    if not _is_jsonable(payload):
        app_logger.error(
            "[socketio] payload non-JSON pour event=%s room=%s", event, room
        )
        return

    emitted_ok = False
    try:
        # Flask-SocketIO >= 5 utilise 'to=' ; on passe par **kwargs pour éviter
        # l'analyse statique de Pylance sur des kwargs non déclarés dans les stubs.
        kwargs: dict[str, Any] = {"namespace": namespace, "to": room}
        cast("Any", socketio).emit(event, payload, **kwargs)
        emitted_ok = True
    except TypeError:
        # Compat < 5.x : fallback avec 'room='
        try:
            kwargs = {"namespace": namespace, "room": room}
            cast("Any", socketio).emit(event, payload, **kwargs)
            emitted_ok = True
        except (ConnectionError, OSError) as e:
            # Erreurs réseau attendues : Socket.IO indisponible
            app_logger.error(
                "[socketio] emit failed (compat) event=%s room=%s (network error: %s): %s",
                event,
                room,
                type(e).__name__,
                e,
            )
        except Exception:
            # Erreur inattendue : logger avec trace complète
            app_logger.exception(
                "[socketio] emit failed (compat) event=%s room=%s", event, room
            )
    except (ConnectionError, OSError) as e:
        # Erreurs réseau attendues : Socket.IO indisponible
        app_logger.error(
            "[socketio] emit failed event=%s room=%s (network error: %s): %s",
            event,
            room,
            type(e).__name__,
            e,
        )
    except Exception:
        # Erreur inattendue : logger avec trace complète
        app_logger.exception("[socketio] emit failed event=%s room=%s", event, room)

    if emitted_ok:
        try:
            from services.monitoring.lirie_prometheus import inc_socketio_event

            inc_socketio_event(event)
        except Exception:
            pass
        if _LIRIE_WS_V2_ALIASES:
            alias = _LIRIE_EVENT_ALIASES.get(event)
            if alias:
                _emit_alias_no_metrics(alias, payload, room=room, namespace=namespace)


# ---------------------------------------------------------------------------
# Helpers "métier" d'émission
# ---------------------------------------------------------------------------
def notify_driver_new_booking(
    driver_id: int, booking: Booking, *, namespace: str = DEFAULT_NAMESPACE
) -> None:
    """Émet 'new_booking' vers la room du chauffeur correspondant."""
    try:
        data = (
            booking.to_dict()
            if hasattr(booking, "to_dict")
            else {"id": getattr(booking, "id", None)}
        )
    except (AttributeError, TypeError, ValueError) as e:
        # Erreurs de sérialisation attendues : attributs manquants, types non sérialisables
        app_logger.debug(
            "[socketio] Fallback serialization (validation error: %s): %s",
            type(e).__name__,
            e,
        )
        data = {"id": getattr(booking, "id", None)}
    except Exception:
        # Erreur inattendue lors de la sérialisation
        app_logger.exception("[socketio] Fallback serialization")
        data = {"id": getattr(booking, "id", None)}

    _safe_emit(
        "new_booking", data, room=get_driver_room(driver_id), namespace=namespace
    )


def emit_driver_event(
    driver_id: int,
    event: str,
    payload: dict[str, Any],
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    """Émet un événement générique vers un chauffeur (room driver_...).

    Enrichit automatiquement le payload avec event_id, version, timestamp si absents.
    """
    enriched_payload = _enrich_payload_if_needed(payload, event)
    _safe_emit(
        event, enriched_payload, room=get_driver_room(driver_id), namespace=namespace
    )


def emit_company_event(
    company_id: int,
    event: str,
    payload: dict[str, Any],
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    """Émet un événement SocketIO dans la room de l'entreprise (thread-safe).
    Utilise 'to=' si dispo, sinon 'room=' (compat v4/v5).
    Ne lève pas d'exception : log l'erreur si l'envoi échoue.

    Enrichit automatiquement le payload avec event_id, version, timestamp si absents.
    """
    enriched_payload = _enrich_payload_if_needed(payload, event)
    _safe_emit(
        event, enriched_payload, room=get_company_room(company_id), namespace=namespace
    )


def _emit_dispatch_state_patch_if_enabled(
    *,
    company_id: int,
    op: str,
    reservation_id: int,
    driver_id: int,
    assignment_id: str,
    namespace: str,
    fields: dict[str, Any] | None = None,
) -> None:
    """Événement unifié V2 (dispatch assignment uniquement) — idempotence : correlation_id + occurred_at."""
    if not _LIRIE_DISPATCH_WS_UNIFIED:
        return
    occurred_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    correlation_id = str(uuid.uuid4())
    payload: dict[str, Any] = {
        "v": 1,
        "kind": "dispatch",
        "op": op,
        "company_id": company_id,
        "reservation_id": reservation_id,
        "driver_id": driver_id,
        "assignment_id": assignment_id,
        "correlation_id": correlation_id,
        "occurred_at": occurred_at,
    }
    if fields is not None:
        payload["fields"] = fields
    emit_company_event(
        company_id, "dispatch_state_patch", payload, namespace=namespace
    )


def _maybe_emit_dispatch_dashboard_snapshot(
    company_id: int, *, namespace: str = DEFAULT_NAMESPACE
) -> None:
    """Hint throttlé (max ~1 / 2s / company) pour invalider le GET dashboard realtime côté client."""
    if not _LIRIE_DISPATCH_DASHBOARD_WS:
        return
    if company_id <= 0:
        return
    now = time.monotonic()
    last = _DISPATCH_DB_THROTTLE_LAST.get(company_id, 0.0)
    if now - last < _DISPATCH_DB_THROTTLE_SEC:
        return
    _DISPATCH_DB_THROTTLE_LAST[company_id] = now
    emit_company_event(
        company_id,
        "dispatch_dashboard_snapshot",
        {
            "v": 1,
            "kind": "dispatch_dashboard",
            "correlation_id": str(uuid.uuid4()),
            "occurred_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
        namespace=namespace,
    )


def fanout_driver_location_update(
    company_id: int,
    location_payload: dict[str, Any],
    live_state_payload: dict[str, Any],
    *,
    accept_status: str,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    """P2: Fanout realtime — room entreprise.

    **Contrat produit (à ne pas « symétriser » sans revue)** :

    - ``driver_location_update`` : position pour la carte (canon **ou** observabilité).
    - ``driver_live_state_update`` : **canon uniquement** — jamais pour
      ``accepted_observability_only``. Présence / statut métier suivent la vérité Redis
      canonique, pas un point d'observation.

    - ``accepted_canonical`` : les deux événements ci-dessus.
    - ``accepted_observability_only`` : **uniquement** ``driver_location_update``.
    """
    if company_id <= 0:
        app_logger.error("[socketio] invalid company_id in fanout_driver_location_update")
        return
    if not accept_status.strip():
        raise ValueError("accept_status is required for fanout_driver_location_update")
    # Canonique OU observabilité : les deux doivent pouvoir alimenter la carte entreprise.
    # Avant : seul accepted_canonical fanoutait → la plupart des points (arbitrage, clé Redis
    # legacy sans :canonical, précision GPS médiocre, etc.) n'atteignaient jamais le frontend.
    if accept_status not in ("accepted_canonical", "accepted_observability_only"):
        drv = location_payload.get("driver_id") or live_state_payload.get("driver_id")
        app_logger.info(
            "[socketio] fanout skipped accept_status=%s driver_id=%s company_id=%s",
            accept_status,
            drv,
            company_id,
        )
        return
    payload_company = live_state_payload.get("company_id") or location_payload.get("company_id")
    if payload_company is not None and str(payload_company) != str(company_id):
        app_logger.error(
            "[socketio] cross_tenant_mismatch blocked in fanout company_id=%s payload_company=%s",
            company_id,
            payload_company,
        )
        return
    room = get_company_room(company_id)
    loc_out = {**location_payload, "accept_status": accept_status}
    _safe_emit("driver_location_update", loc_out, room=room, namespace=namespace)
    inc_fanout(event="driver_location_update", accept_status=accept_status)
    if accept_status == "accepted_canonical":
        live_out = {**live_state_payload, "accept_status": accept_status}
        _safe_emit("driver_live_state_update", live_out, room=room, namespace=namespace)
        inc_fanout(event="driver_live_state_update", accept_status=accept_status)


def emit_date_event(
    date_str: str,
    event: str,
    payload: dict[str, Any],
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    """Émet un événement vers la room d'une date (utile pour vues par journée).

    Enrichit automatiquement le payload avec event_id, version, timestamp si absents.
    """
    enriched_payload = _enrich_payload_if_needed(payload, event)
    _safe_emit(
        event, enriched_payload, room=get_date_room(date_str), namespace=namespace
    )


# Évènements typés du moteur/dispatch
def emit_dispatch_run_started(
    company_id: int,
    dispatch_run_id: str,
    date_str: str,
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    # ✅ FIX: Standardiser avec '_' au lieu de ':' pour cohérence
    emit_company_event(
        company_id,
        "dispatch_run_started",
        {"dispatch_run_id": dispatch_run_id, "date": date_str},
        namespace=namespace,
    )
    # Optionnel: cibler aussi la room date_YYYY-MM-DD
    emit_date_event(
        date_str,
        "dispatch_run_started",
        {"dispatch_run_id": dispatch_run_id, "date": date_str},
        namespace=namespace,
    )


def emit_dispatch_run_completed(
    company_id: int,
    dispatch_run_id: str,
    date_str: str,
    assignments_count: int,
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    payload = {
        "dispatch_run_id": dispatch_run_id,
        "date": date_str,
        "assignments_count": int(assignments_count),
    }
    # Change these event names to match what the frontend is expecting
    emit_company_event(
        company_id, "dispatch_run_completed", payload, namespace=namespace
    )
    emit_date_event(date_str, "dispatch_run_completed", payload, namespace=namespace)


def emit_dispatch_run_failed(
    company_id: int,
    dispatch_run_id: str,
    date_str: str,
    error: str,
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    payload = {
        "dispatch_run_id": dispatch_run_id,
        "date": date_str,
        "error": str(error),
    }
    # ✅ FIX: Standardiser avec '_' au lieu de ':' pour cohérence
    emit_company_event(company_id, "dispatch_run_failed", payload, namespace=namespace)
    emit_date_event(date_str, "dispatch_run_failed", payload, namespace=namespace)


def emit_assignment_created(
    company_id: int,
    booking_id: int,
    driver_id: int,
    assignment_id: str,
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    """Notifie la création d'une assignation :
    - room entreprise (tableau de bord),
    - room chauffeur (réception tâche),
    - (optionnel) room booking si vous la gérez côté front.
    """
    company_payload = {
        "assignment_id": assignment_id,
        "booking_id": booking_id,
        "driver_id": driver_id,
    }
    # ✅ FIX: Standardiser avec '_' au lieu de ':' pour cohérence
    emit_company_event(
        company_id, "dispatch_assignment_created", company_payload, namespace=namespace
    )
    _emit_dispatch_state_patch_if_enabled(
        company_id=company_id,
        op="assignment_created",
        reservation_id=booking_id,
        driver_id=driver_id,
        assignment_id=assignment_id,
        namespace=namespace,
    )
    _maybe_emit_dispatch_dashboard_snapshot(company_id, namespace=namespace)

    driver_payload = {
        "assignment_id": assignment_id,
        "booking_id": booking_id,
    }
    emit_driver_event(
        driver_id, "driver_assignment_received", driver_payload, namespace=namespace
    )


def emit_assignment_updated(
    company_id: int,
    assignment_id: str,
    booking_id: int,
    driver_id: int,
    fields: dict[str, Any],
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    payload = {
        "assignment_id": assignment_id,
        "booking_id": booking_id,
        "driver_id": driver_id,
        "fields": fields,
    }
    # ✅ FIX: Standardiser avec '_' au lieu de ':' pour cohérence
    emit_company_event(
        company_id, "dispatch_assignment_updated", payload, namespace=namespace
    )
    _emit_dispatch_state_patch_if_enabled(
        company_id=company_id,
        op="assignment_updated",
        reservation_id=booking_id,
        driver_id=driver_id,
        assignment_id=assignment_id,
        namespace=namespace,
        fields=fields,
    )
    _maybe_emit_dispatch_dashboard_snapshot(company_id, namespace=namespace)
    emit_driver_event(
        driver_id, "driver_assignment_updated", payload, namespace=namespace
    )


def emit_assignment_cancelled(
    company_id: int,
    assignment_id: str,
    booking_id: int,
    driver_id: int,
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    payload = {
        "assignment_id": assignment_id,
        "booking_id": booking_id,
        "driver_id": driver_id,
    }
    # ✅ FIX: Standardiser avec '_' au lieu de ':' pour cohérence
    emit_company_event(
        company_id, "dispatch_assignment_cancelled", payload, namespace=namespace
    )
    _emit_dispatch_state_patch_if_enabled(
        company_id=company_id,
        op="assignment_cancelled",
        reservation_id=booking_id,
        driver_id=driver_id,
        assignment_id=assignment_id,
        namespace=namespace,
    )
    _maybe_emit_dispatch_dashboard_snapshot(company_id, namespace=namespace)
    emit_driver_event(
        driver_id, "driver_assignment_cancelled", payload, namespace=namespace
    )


def emit_delay_detected(
    company_id: int,
    booking_id: int,
    assignment_id: str,
    driver_id: int,
    delay_minutes: float,
    *,
    has_alternative: bool = False,
    alternative_driver_id: int | None = None,
    alternative_delay_minutes: float | None = None,
    is_dropoff: bool = False,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    """Émet un événement de retard détecté avec les informations du chauffeur.

    Charge automatiquement les informations du chauffeur (nom, téléphone, véhicule)
    pour les afficher dans le frontend.
    """
    payload: dict[str, Any] = {
        "assignment_id": assignment_id,
        "booking_id": booking_id,
        "driver_id": driver_id,
        "delay_minutes": float(delay_minutes),
        "has_alternative": bool(has_alternative),
        "is_dropoff": bool(is_dropoff),
    }

    # Charger les informations du chauffeur pour l'affichage
    driver = None  # Initialiser avant le try pour éviter "possibly unbound"
    try:
        # ✅ Utilisation du repository pour découpler de SQLAlchemy
        driver_repo = DriverRepository()
        driver_dto = driver_repo.find_by_id(driver_id)
        if driver_dto:
            # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
            # avec eager loading de la relation user
            driver = (
                Driver.query.options(joinedload(Driver.user))
                .filter_by(id=driver_dto.id)
                .first()
            )

        if driver:
            user = getattr(driver, "user", None)
            if user:
                first_name = getattr(user, "first_name", None) or ""
                last_name = getattr(user, "last_name", None) or ""
                driver_name = f"{first_name} {last_name}".strip()
                if not driver_name:
                    driver_name = (
                        getattr(user, "username", None) or f"Chauffeur #{driver_id}"
                    )

                payload["driver_name"] = driver_name
                payload["driver_phone"] = getattr(user, "phone", None)

            # Récupérer la plaque d'immatriculation depuis le Driver
            license_plate = getattr(driver, "license_plate", None)
            if license_plate:
                payload["driver_vehicle"] = license_plate
            else:
                # Fallback sur vehicle_assigned si license_plate n'est pas disponible
                vehicle_assigned = getattr(driver, "vehicle_assigned", None)
                if vehicle_assigned:
                    payload["driver_vehicle"] = vehicle_assigned

    except (OperationalError, DBAPIError) as e:
        # Erreurs DB attendues : connexion, timeout
        app_logger.warning(
            "[socketio] Erreur lors du chargement des infos chauffeur pour driver_id=%s (DB error: %s): %s",
            driver_id,
            type(e).__name__,
            e,
        )
    except (AttributeError, TypeError) as e:
        # Erreurs de validation attendues : attributs manquants
        app_logger.warning(
            "[socketio] Erreur lors du chargement des infos chauffeur pour driver_id=%s (validation error: %s): %s",
            driver_id,
            type(e).__name__,
            e,
        )
    except Exception:
        # En cas d'erreur inattendue, continuer sans les infos du chauffeur (non bloquant)
        app_logger.exception(
            "[socketio] Erreur lors du chargement des infos chauffeur pour driver_id=%s",
            driver_id,
        )

    if has_alternative and alternative_driver_id is not None:
        payload["alternative_driver_id"] = int(alternative_driver_id)
    if alternative_delay_minutes is not None:
        payload["alternative_delay_minutes"] = float(alternative_delay_minutes)

    # ✅ FIX: Standardiser avec '_' au lieu de ':' pour cohérence
    emit_company_event(
        company_id, "dispatch_delay_detected", payload, namespace=namespace
    )
    emit_driver_event(driver_id, "driver_delay_detected", payload, namespace=namespace)

    # ✅ P0: Push notification pour delay.detected (fan-out hybride)
    # ✅ CORRECTIF #3: Utiliser DeviceToken pour support multi-device
    if driver:
        try:
            from ext import db
            from models import DeviceToken
            from services.notifications.device_token_lifecycle import (
                is_push_device_token_lifecycle_enabled,
            )
            from services.notifications.push import send_push_message

            device_tokens = DeviceToken.query.filter_by(
                driver_id=driver.id,
                is_active=True,
            ).all()

            if device_tokens:
                delay_text = (
                    f"{int(delay_minutes)} min" if delay_minutes >= 1 else "< 1 min"
                )
                # Envoyer à tous les devices actifs
                success_count = 0
                last_result: Dict[str, Any] | None = None
                for device_token in device_tokens:
                    result = send_push_message(
                        token=device_token.token,
                        title="Retard détecté",
                        body=f"Retard de {delay_text} sur la mission #{booking_id}",
                        data={
                            "type": "delay",
                            "booking_id": booking_id,
                            "assignment_id": assignment_id,
                            "delay_minutes": float(delay_minutes),
                            "deepLink": f"atmr://booking/{booking_id}?alert=delay",
                        },
                        timeout=5,
                        driver_id=driver.id,
                        bypass_rate_limit=False,  # Les delays ne sont pas critiques, respecter le rate limit
                        provider=getattr(device_token, "provider", None),
                        platform=getattr(device_token, "platform", None),
                        device_token_id=device_token.id,
                    )
                    last_result = result  # Garder le dernier résultat pour logging

                    if result.get("ok"):
                        success_count += 1
                    elif result.get("token_invalid") and not is_push_device_token_lifecycle_enabled():
                        device_token.is_active = False

                try:
                    db.session.commit()
                except Exception:
                    app_logger.exception("[socketio] commit after delay push lifecycle")

                if success_count > 0:
                    app_logger.info(
                        "[socketio] Push sent to driver %s for delay on booking %s (%d/%d devices)",
                        driver.id,
                        booking_id,
                        success_count,
                        len(device_tokens),
                    )
                else:
                    error_msg = (
                        last_result.get("error", "Unknown error")
                        if last_result
                        else "All devices failed"
                    )
                    app_logger.warning(
                        "[socketio] Push failed for all devices of driver %s: %s",
                        driver.id,
                        error_msg,
                    )
        except (ValueError, TypeError, AttributeError) as e:
            app_logger.error(
                "[socketio] Push notification failed (validation error: %s): %s",
                type(e).__name__,
                e,
            )
        except (ConnectionError, OSError) as e:
            app_logger.error(
                "[socketio] Push notification failed (network error: %s): %s",
                type(e).__name__,
                e,
            )
        except Exception:
            app_logger.exception("[socketio] Push notification failed")


# ---------------------------------------------------------------------------
# Helpers pour joindre/quitter des rooms côté serveur (utilisable hors handler)
# ---------------------------------------------------------------------------
def join_company_room(
    sid: str, company_id: int, namespace: str = DEFAULT_NAMESPACE
) -> None:
    """Ajoute un client (sid) à la room d'entreprise
    - utilisable hors contexte handler.
    """
    try:
        cast("Any", socketio).enter_room(
            sid, get_company_room(company_id), namespace=namespace
        )
    except (ConnectionError, OSError) as e:
        # Erreurs réseau attendues : Socket.IO indisponible
        app_logger.error(
            "[socketio] enter_room failed sid=%s company=%s (network error: %s): %s",
            sid,
            company_id,
            type(e).__name__,
            e,
        )
    except Exception:
        # Erreur inattendue : logger avec trace complète
        app_logger.exception(
            "[socketio] enter_room failed sid=%s company=%s", sid, company_id
        )


def leave_company_room(
    sid: str, company_id: int, namespace: str = DEFAULT_NAMESPACE
) -> None:
    """Retire un client (sid) de la room d'entreprise
    - utilisable hors contexte handler.
    """
    try:
        cast("Any", socketio).leave_room(
            sid, get_company_room(company_id), namespace=namespace
        )
    except (ConnectionError, OSError) as e:
        # Erreurs réseau attendues : Socket.IO indisponible
        app_logger.error(
            "[socketio] leave_room failed sid=%s company=%s (network error: %s): %s",
            sid,
            company_id,
            type(e).__name__,
            e,
        )
    except Exception:
        # Erreur inattendue : logger avec trace complète
        app_logger.exception(
            "[socketio] leave_room failed sid=%s company=%s", sid, company_id
        )


def join_date_room(sid: str, date_str: str, namespace: str = DEFAULT_NAMESPACE) -> None:
    """Ajoute un client (sid) à la room de date (YYYY-MM-DD)."""
    try:
        cast("Any", socketio).enter_room(
            sid, get_date_room(date_str), namespace=namespace
        )
    except (ConnectionError, OSError) as e:
        # Erreurs réseau attendues : Socket.IO indisponible
        app_logger.error(
            "[socketio] enter_room(date) failed sid=%s date=%s (network error: %s): %s",
            sid,
            date_str,
            type(e).__name__,
            e,
        )
    except Exception:
        # Erreur inattendue : logger avec trace complète
        app_logger.exception(
            "[socketio] enter_room(date) failed sid=%s date=%s", sid, date_str
        )


def leave_date_room(
    sid: str, date_str: str, namespace: str = DEFAULT_NAMESPACE
) -> None:
    """Retire un client (sid) de la room de date (YYYY-MM-DD)."""
    try:
        cast("Any", socketio).leave_room(
            sid, get_date_room(date_str), namespace=namespace
        )
    except (ConnectionError, OSError) as e:
        # Erreurs réseau attendues : Socket.IO indisponible
        app_logger.error(
            "[socketio] leave_room(date) failed sid=%s date=%s (network error: %s): %s",
            sid,
            date_str,
            type(e).__name__,
            e,
        )
    except Exception:
        # Erreur inattendue : logger avec trace complète
        app_logger.exception(
            "[socketio] leave_room(date) failed sid=%s date=%s", sid, date_str
        )


# ---------------------------------------------------------------------------
# Exports publics explicites (facultatif)
# ---------------------------------------------------------------------------
__all__ = [
    "DEFAULT_NAMESPACE",
    "_safe_emit",
    "emit_assignment_cancelled",
    "emit_assignment_created",
    "emit_assignment_updated",
    "emit_company_event",
    "emit_date_event",
    "emit_delay_detected",
    "emit_dispatch_run_completed",
    "emit_dispatch_run_failed",
    "emit_dispatch_run_started",
    "emit_driver_event",
    "fanout_driver_location_update",
    "get_company_room",
    "get_date_room",
    "get_driver_room",
    "join_company_room",
    "join_date_room",
    "leave_company_room",
    "leave_date_room",
    "notify_driver_new_booking",
]
