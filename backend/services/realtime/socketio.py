# backend/services/socketio_service.py
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import joinedload

from ext import app_logger, socketio
from schemas.socket_events import EVENT_VERSION, SocketEvent

if TYPE_CHECKING:
    from models import Booking

from models import Driver
from repositories.driver_repository import DriverRepository

# ---------------------------------------------------------------------------
# Constantes simples
# ---------------------------------------------------------------------------
DEFAULT_NAMESPACE = "/"


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

    try:
        # Flask-SocketIO >= 5 utilise 'to=' ; on passe par **kwargs pour éviter
        # l'analyse statique de Pylance sur des kwargs non déclarés dans les stubs.
        kwargs: dict[str, Any] = {"namespace": namespace, "to": room}
        cast("Any", socketio).emit(event, payload, **kwargs)
    except TypeError:
        # Compat < 5.x : fallback avec 'room='
        try:
            kwargs = {"namespace": namespace, "room": room}
            cast("Any", socketio).emit(event, payload, **kwargs)
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
    # Import depuis push_service pour éviter les cycles d'import
    if driver and hasattr(driver, "push_token") and driver.push_token:
        try:
            from services.notifications.push import send_push_message

            delay_text = (
                f"{int(delay_minutes)} min" if delay_minutes >= 1 else "< 1 min"
            )
            result = send_push_message(
                token=driver.push_token,
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
                driver_id=driver_id,
                bypass_rate_limit=False,  # Les delays ne sont pas critiques, respecter le rate limit
            )

            if result.get("ok"):
                app_logger.info(
                    "[socketio] Push sent to driver %s for delay on booking %s",
                    driver_id,
                    booking_id,
                )
            else:
                app_logger.warning(
                    "[socketio] Push failed for driver %s: %s",
                    driver_id,
                    result.get("error", "Unknown error"),
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
    "get_company_room",
    "get_date_room",
    "get_driver_room",
    "join_company_room",
    "join_date_room",
    "leave_company_room",
    "leave_date_room",
    "notify_driver_new_booking",
]
