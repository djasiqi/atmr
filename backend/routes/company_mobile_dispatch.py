from __future__ import annotations

# ruff: noqa: I001

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, cast

from flask import current_app, request
from flask_jwt_extended import (  # pyright: ignore[reportMissingImports]
    get_jwt_identity,
    jwt_required,
)
from flask_restx import Namespace, Resource  # pyright: ignore[reportMissingImports]
from sqlalchemy import or_

from ext import db, limiter, role_required
from models import (
    Assignment,
    AutonomousAction,
    Booking,
    Client,
    ClientType,
    Company,
    DispatchMode,
    Driver,
    Message,
    User,
    UserRole,
)
from models.enums import AssignmentStatus, BookingStatus, DriverType, SenderRole

# ✅ REFACTORING: get_company_from_token() n'est plus importé directement
# ✅ DDD: Utilisation de GetCurrentCompanyUseCase via _get_current_company_via_use_case()
from services.dispatch.agent.orchestrator import (
    get_agent_for_company,
    stop_agent_for_company,
)
from services.dispatch.agent.tools import AgentTools
from services.cache import cache_response
from infrastructure.dispatch import settings_module_adapter as dispatch_settings
from infrastructure.dispatch.heuristics_adapter import MAX_FAIRNESS_GAP
from infrastructure.dispatch.queue_adapter import trigger_job
from infrastructure.dispatch.realtime_optimizer_adapter import (
    check_opportunities_manual,
    get_optimizer_for_company,
)
from infrastructure.dispatch.validation_adapter import (
    check_existing_assignment_conflict,
)
from shared.geo_utils import haversine_distance
from shared.time_utils import day_local_bounds, now_local, parse_local_naive

company_mobile_dispatch_ns = Namespace(
    "company_mobile_dispatch",
    description="API mobile entreprise pour le pilotage dispatch (v1)",
)

logger = logging.getLogger(__name__)


def _abort_from_company_error(error: dict[str, Any] | None, code: int | None) -> None:
    message = (error or {}).get("error") if isinstance(error, dict) else "Accès refusé"
    company_mobile_dispatch_ns.abort(code or 403, message)


def _get_current_company() -> Company:
    """Récupère l'entreprise courante via use-case (DDD).

    ✅ DDD: Utilise use-case au lieu de service directement.
    """
    from routes.companies import _get_current_company_via_use_case

    company, err, code = _get_current_company_via_use_case()
    if err or company is None:
        _abort_from_company_error(err if isinstance(err, dict) else None, code)
        raise AssertionError("Company should not be None after abort") from None
    return company


def _get_company_context() -> tuple[Company, int]:
    company = _get_current_company()
    company_id_attr = getattr(company, "id", None)
    if company_id_attr is None:
        company_mobile_dispatch_ns.abort(400, "Company ID invalide.")
        raise AssertionError("Company ID should be defined after abort") from None
    try:
        company_id = int(company_id_attr)
    except (TypeError, ValueError) as exc:
        company_mobile_dispatch_ns.abort(400, "Company ID invalide.")
        raise AssertionError("Company ID should be convertible to int") from exc
    return company, company_id


def _get_current_user() -> User:
    """Récupère l'utilisateur courant à partir du token JWT."""
    from repositories.user_repository import UserRepository

    identity = get_jwt_identity()
    if identity is None:
        company_mobile_dispatch_ns.abort(401, "Token invalide ou expiré.")
        raise AssertionError("JWT identity missing after abort") from None
    user_repo = UserRepository()
    user = user_repo.find_model_by_public_id(public_id=identity)
    if user is None:
        company_mobile_dispatch_ns.abort(404, "Utilisateur introuvable.")
        raise AssertionError("User should exist after abort") from None
    return user


def _serialize_dispatch_settings(company: Company) -> Dict[str, Any]:
    """Retourne les paramètres de dispatch pertinents pour l'app mobile."""
    settings_obj = dispatch_settings.for_company(company)
    overrides = company.get_autonomous_config().get("dispatch_overrides", {}) or {}

    fairness_overrides = overrides.get("fairness", {}) or {}
    fairness_max_gap = fairness_overrides.get("max_gap", MAX_FAIRNESS_GAP)

    service_times = settings_obj.service_times
    emergency_policy = settings_obj.emergency

    return {
        "fairness": {
            "max_gap": int(fairness_max_gap),
        },
        "emergency": {
            "emergency_penalty": float(emergency_policy.emergency_penalty),
        },
        "service_times": {
            "pickup_service_min": int(service_times.pickup_service_min),
            "dropoff_service_min": int(service_times.dropoff_service_min),
            "min_transition_margin_min": int(service_times.min_transition_margin_min),
        },
    }


DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
ACTIVE_ASSIGNMENT_STATUSES = {
    AssignmentStatus.SCHEDULED,
    AssignmentStatus.EN_ROUTE_PICKUP,
    AssignmentStatus.ARRIVED_PICKUP,
    AssignmentStatus.ONBOARD,
    AssignmentStatus.EN_ROUTE_DROPOFF,
}


def _format_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _driver_display_name(driver: Driver) -> str:
    # ✅ Gérer le cas où driver.user pourrait être None ou manquant
    try:
        user = getattr(driver, "user", None)
        if user:
            first_name = (getattr(user, "first_name", None) or "").strip()
            last_name = (getattr(user, "last_name", None) or "").strip()
            full = f"{first_name} {last_name}".strip()
            if full:
                return full
            username = getattr(user, "username", None)
            if username:
                return username
    except (AttributeError, TypeError):
        pass  # Continuer avec le fallback

    # Fallback si user n'existe pas ou est invalide
    return (
        getattr(driver, "name", None)
        or f"Chauffeur #{driver.id if getattr(driver, 'id', None) is not None else '?'}"
    )


def _serialize_driver(driver: Driver | None) -> Dict[str, Any] | None:
    if not driver:
        return None
    # ✅ Gérer le cas où driver_type pourrait être manquant ou invalide
    try:
        if hasattr(driver, "driver_type"):
            if hasattr(driver.driver_type, "value"):
                driver_type = driver.driver_type.value
            else:
                driver_type = str(driver.driver_type)
        else:
            driver_type = DriverType.REGULAR.value  # Valeur par défaut
    except (AttributeError, TypeError):
        driver_type = DriverType.REGULAR.value  # Valeur par défaut en cas d'erreur

    is_emergency = str(driver_type).upper() == DriverType.EMERGENCY.value
    return {
        "id": str(driver.id),
        "name": _driver_display_name(driver),
        "is_emergency": is_emergency,
    }


def _resolve_booking_status(booking: Booking) -> str:
    status_value = getattr(booking.status, "value", str(booking.status or "")).upper()

    # #region agent log - Log EVERY status check
    try:
        import json
        from pathlib import Path

        with Path(".cursor/debug.log").open("a", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                    "sessionId": "debug-session",
                    "runId": "run7",
                    "hypothesisId": "H7",
                    "location": "company_mobile_dispatch.py:202",
                    "message": "resolve_booking_status entry",
                    "data": {
                        "booking_id": booking.id,
                        "raw_status": str(booking.status),
                        "status_value": status_value,
                        "status_type": type(booking.status).__name__,
                        "has_value_attr": hasattr(booking.status, "value"),
                        "driver_id": booking.driver_id,
                    },
                },
                f,
            )
            f.write("\n")
    except Exception:
        pass
    # #endregion

    if status_value in {"CANCELED", "CANCELLED"}:
        return "cancelled"
    if status_value == "RETURN_COMPLETED":
        return "return_completed"
    if status_value == "COMPLETED":
        return "completed"
    # ✅ Gérer le status PENDING (course en attente d'acceptation/refus)
    if status_value == "PENDING":
        # #region agent log
        try:
            import json
            from pathlib import Path

            with Path(".cursor/debug.log").open("a", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": __import__("datetime").datetime.now().isoformat(),
                        "sessionId": "debug-session",
                        "runId": "run7",
                        "hypothesisId": "H7-CONFIRMED",
                        "location": "company_mobile_dispatch.py:242",
                        "message": "Booking status PENDING detected",
                        "data": {
                            "booking_id": booking.id,
                            "booking_status": status_value,
                            "resolved_status": "pending",
                        },
                    },
                    f,
                )
                f.write("\n")
        except Exception:
            pass
        # #endregion
        return "pending"
    driver_id_value = getattr(booking, "driver_id", None)
    if isinstance(driver_id_value, int):
        return "assigned"
    return "unassigned"


def _get_active_assignment(booking: Booking) -> Assignment | None:
    # ✅ Gérer le cas où assignments pourrait être None ou non itérable
    try:
        assignments = getattr(booking, "assignments", []) or []
        if not isinstance(assignments, (list, tuple)):
            return None
        for assignment in assignments:
            if (
                assignment
                and hasattr(assignment, "status")
                and assignment.status in ACTIVE_ASSIGNMENT_STATUSES
            ):
                return assignment
    except (AttributeError, TypeError, ValueError):
        logger.warning(
            "[_get_active_assignment] Erreur lors de la récupération des assignations pour booking %s",
            booking.id if hasattr(booking, "id") else "?",
        )
    return None


def _build_ride_summary(
    booking: Booking, current_company_id: int | None = None
) -> Dict[str, Any]:
    active_assignment = _get_active_assignment(booking)
    driver = booking.driver or (
        active_assignment.driver
        if active_assignment and active_assignment.driver
        else None
    )
    drop_eta = None
    if active_assignment and active_assignment.eta_dropoff_at:
        drop_eta = _format_datetime(active_assignment.eta_dropoff_at)

    # ✅ Ne pas calculer le retard pour les courses terminées
    booking_status = _resolve_booking_status(booking)
    is_completed = booking_status in ("completed", "return_completed")

    delay_seconds = 0
    if not is_completed and active_assignment:
        delay_seconds = getattr(active_assignment, "delay_seconds", 0) or 0

    risk_delay = bool(booking.is_urgent) or (
        not is_completed
        and isinstance(delay_seconds, (int, float))
        and delay_seconds > 15 * 60
    )

    distance_meters = getattr(booking, "distance_meters", None)
    distance_km = (
        round(distance_meters / 1000.0, 1)
        if isinstance(distance_meters, (int, float))
        else None
    )

    client_priority = "HIGH" if getattr(booking, "is_urgent", False) else "NORMAL"
    client_id_value = getattr(booking, "client_id", None)

    # ✅ Récupérer les informations détaillées du client
    client_info: Dict[str, Any] = {
        "id": str(client_id_value) if client_id_value is not None else "None",
        "name": getattr(booking, "customer_full_name", None)
        or booking.customer_name
        or "Client",
        "priority": client_priority,
    }

    # ✅ Ajouter les informations supplémentaires si le client existe
    if booking.client:
        client = booking.client
        # Date de naissance depuis User
        if client.user and client.user.birth_date:
            client_info["birth_date"] = client.user.birth_date.isoformat()

        # Téléphone (priorité: contact_phone du client, sinon phone de l'user)
        if client.contact_phone:
            client_info["phone"] = client.contact_phone
        elif client.user and client.user.phone:
            client_info["phone"] = client.user.phone

        # Adresse de domicile
        domicile_parts = []
        if client.domicile_address:
            domicile_parts.append(client.domicile_address)
        if client.domicile_city:
            domicile_parts.append(client.domicile_city)
        if client.domicile_zip:
            domicile_parts.append(client.domicile_zip)

        if domicile_parts:
            client_info["home_address"] = ", ".join(domicile_parts)

        # Prénom et nom séparés (depuis User)
        if client.user:
            if client.user.first_name:
                client_info["first_name"] = client.user.first_name
            if client.user.last_name:
                client_info["last_name"] = client.user.last_name

    # ✅ Récupérer les informations de transfert
    transfer_info: Dict[str, Any] | None = None
    active_transfer = None

    # Chercher un transfert actif (PENDING ou ACCEPTED)
    if hasattr(booking, "transfers") and booking.transfers:
        from models.enums import TransferStatus as TransferStatusEnum

        for transfer in booking.transfers:
            if transfer.status in [
                TransferStatusEnum.PENDING,
                TransferStatusEnum.ACCEPTED,
            ]:
                active_transfer = transfer
                break

    if active_transfer:
        from models.enums import TransferStatus as TransferStatusEnum

        # Utiliser current_company_id si fourni, sinon fallback sur booking.company_id
        company_id_for_transfer = (
            current_company_id if current_company_id is not None else booking.company_id
        )

        transfer_info = {
            "id": str(active_transfer.id),
            "status": active_transfer.status.value
            if hasattr(active_transfer.status, "value")
            else str(active_transfer.status),
            "is_sender": company_id_for_transfer == active_transfer.owner_company_id,
            "is_receiver": company_id_for_transfer
            == active_transfer.executing_company_id,
            "partner_company_id": str(active_transfer.executing_company_id)
            if company_id_for_transfer == active_transfer.owner_company_id
            else str(active_transfer.owner_company_id),
            "partner_company_name": active_transfer.executing_company.name
            if hasattr(active_transfer, "executing_company")
            and active_transfer.executing_company
            else (
                active_transfer.owner_company.name
                if hasattr(active_transfer, "owner_company")
                and active_transfer.owner_company
                else None
            ),
        }

    summary: Dict[str, Any] = {
        "id": str(booking.id),
        "time": {
            "pickup_at": booking.scheduled_time.isoformat()
            if booking.scheduled_time
            else None,
            "drop_eta": drop_eta,
            "window_start": None,
            "window_end": None,
        },
        "client": client_info,
        "route": {
            "pickup_address": booking.pickup_location or "",
            "dropoff_address": booking.dropoff_location or "",
            "distance_km": distance_km,
        },
        "status": _resolve_booking_status(booking),
        "driver": _serialize_driver(driver),
        "transfer": transfer_info,  # ✅ Ajouter les informations de transfert
        "flags": {
            "risk_delay": risk_delay,
            "prefs_respected": True,
            "fairness_score": None,
            "override_pending": False,
        },
    }

    # #region agent log
    try:
        import json
        from pathlib import Path

        with Path(".cursor/debug.log").open("a", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                    "sessionId": "debug-session",
                    "runId": "run6",
                    "hypothesisId": "H4",
                    "location": "company_mobile_dispatch.py:367",
                    "message": "Ride summary built",
                    "data": {
                        "booking_id": booking.id,
                        "resolved_status": summary.get("status"),
                        "transfer_status": transfer_info.get("status")
                        if transfer_info
                        else None,
                        "raw_booking_status": getattr(
                            booking.status, "value", str(booking.status or "")
                        ),
                    },
                },
                f,
            )
            f.write("\n")
    except Exception:
        pass
    # #endregion

    return summary


def _compute_driver_suggestions(
    company_id: int, booking: Booking
) -> List[Dict[str, Any]]:
    """Calcule les suggestions de chauffeurs pour un booking donné.

    ✅ S'assure que seuls les chauffeurs de l'entreprise sont retournés.

    Args:
        company_id: ID de l'entreprise (filtré strictement)
        booking: Booking pour lequel calculer les suggestions

    Returns:
        Liste de TOUS les suggestions de chauffeurs disponibles, triées par score décroissant
    """
    try:
        pickup_lat = getattr(booking, "pickup_lat", None)
        pickup_lon = getattr(booking, "pickup_lon", None)
        # ✅ Ne pas retourner une liste vide si pas de GPS : inclure tous les chauffeurs quand même
        # (les chauffeurs sans GPS auront un score de 0.0)

        from repositories.driver_repository import DriverRepository

        driver_repo = DriverRepository()
        available_drivers = (
            driver_repo.find_models_by_company_active_available_with_user_eager_loading(
                company_id=company_id
            )
        )

        driver_id_value = getattr(booking, "driver_id", None)
        current_driver_id = (
            driver_id_value if isinstance(driver_id_value, int) else None
        )

        suggestions: List[Dict[str, Any]] = []
        for driver in available_drivers:
            # ✅ Vérification supplémentaire : s'assurer que le driver appartient à l'entreprise
            driver_company_id = getattr(driver, "company_id", None)
            if driver_company_id is None or int(driver_company_id) != int(company_id):
                logger.warning(
                    "[MobileDispatch] Driver %d n'appartient pas à l'entreprise %d (company_id=%s)",
                    driver.id,
                    company_id,
                    driver_company_id,
                )
                continue

            if current_driver_id is not None and driver.id == current_driver_id:
                continue

            # ✅ Inclure TOUS les chauffeurs, même ceux sans position GPS
            # ✅ Gérer le cas où driver_type pourrait être manquant ou invalide
            try:
                if hasattr(driver, "driver_type"):
                    if hasattr(driver.driver_type, "value"):
                        driver_type = driver.driver_type.value
                    else:
                        driver_type = str(driver.driver_type)
                else:
                    driver_type = DriverType.REGULAR.value  # Valeur par défaut
            except (AttributeError, TypeError):
                driver_type = (
                    DriverType.REGULAR.value
                )  # Valeur par défaut en cas d'erreur

            # Calculer la distance si positions GPS disponibles (booking ET driver)
            if (
                pickup_lat is not None
                and pickup_lon is not None
                and (lat_val := getattr(driver, "latitude", None)) is not None
                and (lon_val := getattr(driver, "longitude", None)) is not None
            ):
                try:
                    distance_km = haversine_distance(
                        float(pickup_lat),
                        float(pickup_lon),
                        float(lat_val),
                        float(lon_val),
                    )
                    score = round(1.0 / (1.0 + distance_km), 4)
                    reason = f"Distance estimée {distance_km:.1f} km"
                except Exception:
                    # Fallback si calcul distance échoue
                    score = 0.0
                    reason = "Position GPS disponible"
            else:
                # Chauffeur ou booking sans position GPS : score par défaut bas mais inclus
                score = 0.0
                if (
                    getattr(driver, "latitude", None) is None
                    or getattr(driver, "longitude", None) is None
                ):
                    reason = "Position GPS du chauffeur non disponible"
                else:
                    reason = "Position GPS de la course non disponible"

            suggestions.append(
                {
                    "driver_id": str(driver.id),
                    "driver_name": _driver_display_name(driver),
                    "score": score,
                    "fairness_delta": None,
                    "preferred_match": False,
                    "is_emergency": str(driver_type).upper()
                    == DriverType.EMERGENCY.value,
                    "reason": reason,
                }
            )

        suggestions.sort(key=lambda s: s["score"], reverse=True)
        return suggestions  # ✅ Retourner TOUS les chauffeurs, pas seulement les 3 meilleurs
    except Exception as e:
        logger.exception(
            "[_compute_driver_suggestions] Erreur lors du calcul des suggestions pour booking %s: %s",
            booking.id if hasattr(booking, "id") else "?",
            e,
        )
        # ✅ Retourner une liste vide en cas d'erreur plutôt que de faire planter l'endpoint
        # Le frontend pourra alors charger tous les chauffeurs manuellement
        return []


def _build_ride_history(
    booking: Booking, assignment: Assignment | None
) -> List[Dict[str, Any]]:
    """Construit l'historique des événements de la course.

    ✅ Formate les détails JSON en texte lisible pour l'entreprise.
    """
    history: List[Dict[str, Any]] = []
    created_at = getattr(booking, "created_at", None)

    # Mapping des statuts en français
    status_map = {
        "ACCEPTED": "Acceptée",
        "SCHEDULED": "Planifiée",
        "ASSIGNED": "Assignée",
        "IN_PROGRESS": "En cours",
        "COMPLETED": "Terminée",
        "CANCELLED": "Annulée",
        "PENDING": "En attente",
    }

    # Événement de création
    # ✅ Gérer le cas où booking.status pourrait être manquant ou invalide
    try:
        created_status = getattr(booking.status, "value", str(booking.status))
    except (AttributeError, TypeError):
        created_status = "UNKNOWN"
    history.append(
        {
            "ts": created_at.isoformat()
            if isinstance(created_at, datetime)
            else now_local().isoformat(),
            "event": "created",
            "actor": "system",
            "details": {"status": created_status},
            "details_formatted": f"Statut: {status_map.get(created_status, created_status)}",
        }
    )

    # Événement d'assignation
    if assignment:
        assigned_at = getattr(assignment, "updated_at", None) or getattr(
            assignment, "created_at", None
        )
        assigned_status = getattr(assignment.status, "value", str(assignment.status))
        history.append(
            {
                "ts": assigned_at.isoformat()
                if isinstance(assigned_at, datetime)
                else now_local().isoformat(),
                "event": "assigned",
                "actor": "dispatcher",
                "details": {
                    "driver_id": assignment.driver_id,
                    "status": assigned_status,
                },
                "details_formatted": f"Chauffeur assigné: #{assignment.driver_id}\nStatut: {status_map.get(assigned_status, assigned_status)}",
            }
        )

    return history


def _build_ride_conflicts(booking: Booking) -> List[Dict[str, Any]]:
    booking_driver_id = getattr(booking, "driver_id", None)
    if booking_driver_id is None or not booking.scheduled_time:
        return []
    # ✅ Unpacking robuste : compatible si la fonction renvoie 2, 3 ou plus de valeurs
    result = check_existing_assignment_conflict(
        driver_id=int(booking_driver_id),
        scheduled_time=booking.scheduled_time,
        booking_id=int(booking.id),
        tolerance_minutes=30,
    )
    has_conflict = result[0] if result else False
    message = result[1] if len(result) > 1 else None
    if not has_conflict:
        return []
    return [
        {
            "type": "temporal",
            "message": message or "Conflit temporel détecté",
            "blocking": True,
        }
    ]


def _log_mobile_action(
    tools: AgentTools, kind: str, payload: Dict[str, Any], reasoning: str
) -> str | None:
    try:
        result = tools.log_action(kind=kind, payload=payload, reasoning_brief=reasoning)
        return result.get("event_id")
    except Exception as exc:  # pragma: no cover - logging best effort
        logger.warning(
            "[MobileDispatch] Impossible de journaliser l'action %s: %s", kind, exc
        )
        return None


def _execute_assignment_action(
    company_id: int, booking_id: int, driver_id: int, action_kind: str
) -> Dict[str, Any]:
    tools = AgentTools(company_id)
    assign_result = tools.assign(job_id=booking_id, driver_id=driver_id)

    if not assign_result.get("ok"):
        error_message = assign_result.get("error") or "Assignation impossible"
        if assign_result.get("conflict"):
            company_mobile_dispatch_ns.abort(409, error_message)
            raise AssertionError("Conflict should abort") from None
        company_mobile_dispatch_ns.abort(400, error_message)
        raise AssertionError("Assign failure should abort") from None

    from repositories.booking_repository import BookingRepository
    from models.enums import TransferStatus
    from sqlalchemy.orm import joinedload

    booking_repo = BookingRepository()
    booking = booking_repo.find_model_by_id_with_full_eager_loading(
        booking_id=booking_id, company_id=company_id
    )

    # Si pas trouvé avec company_id, vérifier si c'est un transfert accepté dont on est le receveur
    if booking is None:
        # Récupérer la course sans filtre de company_id, avec les transferts
        booking = (
            Booking.query.filter(Booking.id == booking_id)
            .options(joinedload(Booking.transfers))
            .first()
        )

        # Vérifier si on est le receveur d'un transfert accepté
        if booking:
            has_accepted_transfer = False
            for transfer in booking.transfers:
                if (
                    transfer.status == TransferStatus.ACCEPTED
                    and transfer.executing_company_id == company_id
                ):
                    has_accepted_transfer = True
                    break

            if not has_accepted_transfer:
                booking = None  # Pas autorisé à gérer cette course

    if booking is None:
        company_mobile_dispatch_ns.abort(404, "Course introuvable après assignation")
        raise AssertionError("Booking should exist after assign") from None

    if booking.driver_id != driver_id:
        booking.driver_id = driver_id
        # ✅ Mettre à jour le statut de ACCEPTED à ASSIGNED lors de l'assignation
        if booking.status == BookingStatus.ACCEPTED:
            booking.status = BookingStatus.ASSIGNED
        db.session.add(booking)
        db.session.commit()

    # ✅ NOUVEAU: Récupérer les informations de groupe depuis assign_result
    is_grouped = assign_result.get("grouped", False)
    group_id = assign_result.get("group_id")

    event_id = (
        _log_mobile_action(
            tools,
            action_kind,
            payload={
                "booking_id": booking_id,
                "driver_id": driver_id,
                "source": "mobile_enterprise",
                "grouped": is_grouped,
                "group_id": group_id,
            },
            reasoning=f"{action_kind} {booking_id} -> {driver_id}"
            + (f" (groupé avec {group_id})" if is_grouped else ""),
        )
        or ""
    )

    scheduled_time = (
        booking.scheduled_time.isoformat() if booking.scheduled_time else None
    )
    diff = assign_result.get("diff", {})
    return {
        "ride_id": str(booking_id),
        "grouped": is_grouped,
        "group_id": group_id,
        "driver_id": str(driver_id),
        "scheduled_time": scheduled_time,
        "fairness_delta": 0.0,
        "audit_event_id": event_id,
        "message": diff.get("action", "assigned"),
    }


@company_mobile_dispatch_ns.route("/v1/status")
class MobileDispatchStatus(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, company_id = _get_company_context()

        # Date cible pour les KPI (défaut: aujourd'hui en heure locale)
        requested_date = request.args.get("date")
        if not requested_date:
            requested_date = now_local().strftime("%Y-%m-%d")

        window_start: datetime
        window_end: datetime
        try:
            window_start, window_end = day_local_bounds(requested_date)
        except Exception as exc:
            company_mobile_dispatch_ns.abort(
                400,
                f"Format de date invalide: {requested_date} (attendu: YYYY-MM-DD)",
            )
            raise AssertionError("Invalid date should abort") from exc

        try:
            # KPI principaux
            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            bookings_query = (
                booking_repo.find_models_by_company_with_time_range_and_statuses_query(
                    company_id=company_id,
                    start_datetime=window_start,
                    end_datetime=window_end,
                    statuses=[
                        BookingStatus.ACCEPTED,
                        BookingStatus.ASSIGNED,
                        BookingStatus.EN_ROUTE,
                        BookingStatus.IN_PROGRESS,
                        BookingStatus.COMPLETED,
                        BookingStatus.RETURN_COMPLETED,
                    ],
                )
            )

            total_bookings = bookings_query.count()
            assigned_bookings = bookings_query.filter(
                Booking.driver_id.isnot(None)
            ).count()
            assignment_rate = (
                assigned_bookings / total_bookings if total_bookings > 0 else 0.0
            )
            at_risk_count = bookings_query.filter(Booking.is_urgent.is_(True)).count()

            kpis = {
                "date": requested_date,
                "total_bookings": total_bookings,
                "assigned_bookings": assigned_bookings,
                "assignment_rate": round(assignment_rate, 4),
                "at_risk": at_risk_count,
            }

            # Santé OSRM via tools agent
            osrm_status_payload = {
                "status": "DOWN",
                "latency_ms": None,
                "last_check": None,
            }
            try:
                tools = AgentTools(company_id)
                with current_app.app_context():
                    osrm_health = tools.osrm_health()
                osrm_state = osrm_health.get("state", "OPEN")
                latency = osrm_health.get("latency_ms")
                test_successful = osrm_health.get("test_successful", False)

                latency_value = (
                    latency if isinstance(latency, int) and latency >= 0 else None
                )
                if osrm_state == "CLOSED" and test_successful:
                    osrm_status = "OK"
                elif osrm_state in {"HALF_OPEN", "OPEN"} and test_successful:
                    osrm_status = "WARNING"
                else:
                    osrm_status = "DOWN"

                osrm_status_payload = {
                    "status": osrm_status,
                    "latency_ms": latency_value,
                    "last_check": now_local().isoformat(),
                }
            except Exception as exc:  # pragma: no cover - fallback résilient
                logger.warning(
                    "[MobileDispatch] OSRM health check failed for company %s: %s",
                    company_id,
                    exc,
                )

            # Statut agent orchestrator
            agent_mode_value = getattr(company.dispatch_mode, "value", "manual")
            agent_mode = agent_mode_value.upper()
            agent_active = False
            agent_last_tick = None
            try:
                agent = get_agent_for_company(
                    company_id,
                    app=current_app._get_current_object(),
                )
                agent_status = agent.get_status()
                agent_active = bool(agent_status.get("running"))
                agent_last_tick = agent_status.get("last_tick")
            except Exception as exc:  # pragma: no cover
                warning_msg = (
                    "[MobileDispatch] Impossible de récupérer le statut agent "
                    + "pour company %s: %s"
                )
                logger.warning(
                    warning_msg,
                    company_id,
                    exc,
                )

            agent_payload = {
                "mode": agent_mode,
                "active": agent_active,
                "last_tick": agent_last_tick,
            }

            # Statut optimizer
            optimizer_payload = {
                "active": False,
                "next_window_start": None,
            }
            try:
                optimizer = get_optimizer_for_company(company_id)
                if optimizer:
                    optimizer_status = optimizer.get_status()
                    optimizer_active = bool(optimizer_status.get("running"))
                    last_check = optimizer_status.get("last_check")
                    interval_seconds = optimizer_status.get("check_interval_seconds")

                    next_window_iso = None
                    if last_check and interval_seconds:
                        try:
                            last_check_dt = datetime.fromisoformat(last_check)
                            next_window_iso = (
                                last_check_dt + timedelta(seconds=int(interval_seconds))
                            ).isoformat()
                        except (ValueError, TypeError):
                            next_window_iso = last_check

                    optimizer_payload = {
                        "active": optimizer_active,
                        "next_window_start": next_window_iso,
                    }
            except Exception as exc:  # pragma: no cover
                warning_msg = (
                    "[MobileDispatch] Impossible de récupérer le statut optimizer "
                    + "pour company %s: %s"
                )
                logger.warning(
                    warning_msg,
                    company_id,
                    exc,
                )

            response = {
                "osrm": osrm_status_payload,
                "agent": agent_payload,
                "optimizer": optimizer_payload,
                "kpis": kpis,
            }

            return response, 200
        except Exception as exc:
            db.session.rollback()
            logger.exception(
                "[MobileDispatch] Erreur récupération statut pour company %s: %s",
                company_id,
                exc,
            )
            company_mobile_dispatch_ns.abort(
                500, "Impossible de récupérer le statut dispatch mobile."
            )


@company_mobile_dispatch_ns.route("/v1/rides")
class MobileDispatchRides(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @cache_response("api:company_mobile:rides", ttl=3)
    def get(self):
        # #region agent log
        import json
        from pathlib import Path

        log_path = r"c:\Users\jasiq\atmr\.cursor\debug.log"
        try:
            with Path(log_path).open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "location": "company_mobile_dispatch.py:641",
                            "message": "GET /v1/rides entry",
                            "data": {"args": dict(request.args)},
                            "timestamp": datetime.now(UTC).isoformat(),
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "A",
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        try:
            _, company_id = _get_company_context()
            # #region agent log
            try:
                with Path(log_path).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "company_mobile_dispatch.py:643",
                                "message": "_get_company_context success",
                                "data": {"company_id": company_id},
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "A",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
        except Exception as e:
            # #region agent log
            try:
                with Path(log_path).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "company_mobile_dispatch.py:643",
                                "message": "_get_company_context error",
                                "data": {
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "A",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            raise

        requested_date = request.args.get("date") or now_local().strftime("%Y-%m-%d")
        status_filter = (request.args.get("status") or "").strip().lower() or None
        search_query = (request.args.get("q") or "").strip()

        try:
            page = int(request.args.get("page", "1"))
        except ValueError:
            page = 1
        page = max(1, page)

        try:
            page_size = int(request.args.get("page_size", str(DEFAULT_PAGE_SIZE)))
        except ValueError:
            page_size = DEFAULT_PAGE_SIZE
        page_size = max(1, min(page_size, MAX_PAGE_SIZE))

        window_start: datetime
        window_end: datetime
        try:
            window_start, window_end = day_local_bounds(requested_date)
            # #region agent log
            try:
                with Path(log_path).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "company_mobile_dispatch.py:663",
                                "message": "day_local_bounds success",
                                "data": {
                                    "window_start": window_start.isoformat(),
                                    "window_end": window_end.isoformat(),
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "B",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
        except Exception as exc:
            # #region agent log
            try:
                with Path(log_path).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "company_mobile_dispatch.py:664",
                                "message": "day_local_bounds error",
                                "data": {
                                    "error": str(exc),
                                    "error_type": type(exc).__name__,
                                    "requested_date": requested_date,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "B",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            company_mobile_dispatch_ns.abort(
                400,
                f"Format de date invalide: {requested_date} (attendu: YYYY-MM-DD)",
            )
            raise AssertionError("Invalid date should abort") from exc

        ACTIVE_BOOKING_STATUSES = [
            BookingStatus.PENDING,  # ✅ Inclure les courses transférées en attente
            BookingStatus.ACCEPTED,
            BookingStatus.ASSIGNED,
            BookingStatus.EN_ROUTE,
            BookingStatus.IN_PROGRESS,
            BookingStatus.COMPLETED,
            BookingStatus.RETURN_COMPLETED,
        ]

        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        try:
            bookings_query = booking_repo.find_models_by_company_with_time_range_or_none_and_statuses_query(
                company_id=company_id,
                start_datetime=window_start,
                end_datetime=window_end,
                statuses=ACTIVE_BOOKING_STATUSES,
            )
            # ✅ Eager load des transferts et partenaires pour éviter les requêtes N+1
            from sqlalchemy.orm import joinedload
            from models.booking_transfer import BookingTransfer

            bookings_query = bookings_query.options(
                joinedload(Booking.transfers).joinedload(BookingTransfer.partnership),
                joinedload(Booking.transfers).joinedload(
                    BookingTransfer.executing_company
                ),
            )
            # #region agent log
            try:
                with Path(log_path).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "company_mobile_dispatch.py:688",
                                "message": "find_models_by_company query created",
                                "data": {"query_type": type(bookings_query).__name__},
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "C",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
        except Exception as e:
            # #region agent log
            try:
                with Path(log_path).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "company_mobile_dispatch.py:688",
                                "message": "find_models_by_company error",
                                "data": {
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "C",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            raise

        if status_filter == "assigned":
            bookings_query = bookings_query.filter(Booking.driver_id.isnot(None))
        elif status_filter == "unassigned":
            bookings_query = bookings_query.filter(
                Booking.driver_id.is_(None),
                Booking.status.in_(
                    [
                        BookingStatus.ACCEPTED,
                        BookingStatus.ASSIGNED,
                        BookingStatus.EN_ROUTE,
                        BookingStatus.IN_PROGRESS,
                    ]
                ),
            )
        elif status_filter == "urgent":
            bookings_query = bookings_query.filter(Booking.is_urgent.is_(True))
        elif status_filter == "cancelled":
            bookings_query = bookings_query.filter(
                Booking.status.in_([BookingStatus.CANCELED])
            )

        if search_query:
            like_value = f"%{search_query}%"
            bookings_query = bookings_query.filter(
                or_(
                    Booking.customer_name.ilike(like_value),
                    Booking.pickup_location.ilike(like_value),
                    Booking.dropoff_location.ilike(like_value),
                )
            )

        try:
            total = bookings_query.count()
            # #region agent log
            try:
                with Path(log_path).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "company_mobile_dispatch.py:721",
                                "message": "count() success",
                                "data": {"total": total},
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "D",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
        except Exception as e:
            # #region agent log
            try:
                with Path(log_path).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "company_mobile_dispatch.py:721",
                                "message": "count() error",
                                "data": {
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "D",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            raise

        try:
            bookings = (
                bookings_query.order_by(
                    Booking.scheduled_time.is_(None),
                    Booking.scheduled_time.asc(),
                    Booking.id.asc(),
                )
                .offset((page - 1) * page_size)
                .limit(page_size)
                .all()
            )
            # #region agent log
            try:
                import json
                from pathlib import Path

                log_path_debug = Path(__file__).parent.parent / ".cursor" / "debug.log"
                with log_path_debug.open("a", encoding="utf-8") as f:
                    status_distribution = {}
                    for b in bookings:
                        status_str = (
                            str(b.status.value)
                            if hasattr(b.status, "value")
                            else str(b.status)
                        )
                        status_distribution[status_str] = (
                            status_distribution.get(status_str, 0) + 1
                        )
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run2",
                                "hypothesisId": "H2",
                                "location": "company_mobile_dispatch.py:1120",
                                "message": "Bookings fetched - status distribution",
                                "data": {
                                    "total_bookings": len(bookings),
                                    "status_distribution": status_distribution,
                                    "date": requested_date,
                                    "status_filter": status_filter,
                                },
                                "timestamp": int(__import__("time").time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            # #region agent log
            try:
                with Path(log_path).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "company_mobile_dispatch.py:732",
                                "message": "all() success",
                                "data": {"bookings_count": len(bookings)},
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "D",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
        except Exception as e:
            # #region agent log
            try:
                with Path(log_path).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "company_mobile_dispatch.py:732",
                                "message": "all() error",
                                "data": {
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "D",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            raise

        try:
            items = []
            for i, booking in enumerate(bookings):
                try:
                    item = _build_ride_summary(booking, current_company_id=company_id)
                    items.append(item)
                except Exception as e:
                    # #region agent log
                    try:
                        with Path(log_path).open("a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "location": "company_mobile_dispatch.py:734",
                                        "message": "_build_ride_summary error",
                                        "data": {
                                            "error": str(e),
                                            "error_type": type(e).__name__,
                                            "booking_id": getattr(booking, "id", None),
                                            "index": i,
                                        },
                                        "timestamp": datetime.now(UTC).isoformat(),
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "E",
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                    # #endregion
                    raise
            # #region agent log
            try:
                with Path(log_path).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "company_mobile_dispatch.py:734",
                                "message": "_build_ride_summary all success",
                                "data": {"items_count": len(items)},
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "E",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
        except Exception as e:
            # #region agent log
            try:
                with Path(log_path).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": "company_mobile_dispatch.py:734",
                                "message": "_build_ride_summary loop error",
                                "data": {
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "E",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            raise

        result = {
            "page": page,
            "page_size": page_size,
            "total": total,
            "items": items,
        }

        # #region agent log - Log FULL response
        try:
            import json
            from pathlib import Path

            # Log only the first item to avoid huge logs
            first_item_status = items[0].get("status") if items else None
            with Path(".cursor/debug.log").open("a", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": __import__("datetime").datetime.now().isoformat(),
                        "sessionId": "debug-session",
                        "runId": "run8",
                        "hypothesisId": "H8-FINAL",
                        "location": "company_mobile_dispatch.py:1427",
                        "message": "Response JSON about to be sent",
                        "data": {
                            "total_items": len(items),
                            "first_item_status": first_item_status,
                            "first_item_id": items[0].get("id") if items else None,
                        },
                    },
                    f,
                )
                f.write("\n")
        except Exception:
            pass
        # #endregion

        return result, 200

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Créer une nouvelle course depuis l'interface mobile."""
        try:
            _, company_id = _get_company_context()
        except Exception as e:
            logger.exception(
                "[MobileDispatchRides POST] Erreur lors de la récupération du contexte entreprise"
            )
            company_mobile_dispatch_ns.abort(
                500, "Impossible de récupérer le contexte entreprise."
            )
            raise AssertionError("Should have aborted") from e

        payload = request.get_json(silent=True) or {}

        # ✅ Validation des champs requis
        if not payload.get("pickup_address"):
            company_mobile_dispatch_ns.abort(400, "pickup_address est requis.")
            raise AssertionError("pickup_address should not be None after abort")

        if not payload.get("dropoff_address"):
            company_mobile_dispatch_ns.abort(400, "dropoff_address est requis.")
            raise AssertionError("dropoff_address should not be None after abort")

        if not payload.get("scheduled_time"):
            company_mobile_dispatch_ns.abort(400, "scheduled_time est requis.")
            raise AssertionError("scheduled_time should not be None after abort")

        # ✅ Client ou customer_name requis
        client_id = payload.get("client_id")
        customer_name = payload.get("customer_name")
        if not client_id and not customer_name:
            company_mobile_dispatch_ns.abort(
                400, "client_id ou customer_name est requis."
            )
            raise AssertionError(
                "client_id or customer_name should not be None after abort"
            )

        try:
            # ✅ Récupérer ou créer le client
            from repositories.client_repository import ClientRepository
            from shared.time_utils import parse_local_naive

            client_repo = ClientRepository()
            client = None
            display_name = customer_name or ""

            if client_id:
                try:
                    client_id_int = int(client_id)
                    client = client_repo.find_model_by_id_and_company(
                        client_id_int, company_id
                    )
                    if not client:
                        company_mobile_dispatch_ns.abort(
                            404,
                            f"Client {client_id} introuvable pour cette entreprise.",
                        )
                        raise AssertionError("Client should not be None after abort")

                    # Déterminer le nom d'affichage
                    user = getattr(client, "user", None)
                    if user:
                        full_name = f"{getattr(user, 'first_name', '')} {getattr(user, 'last_name', '')}".strip()
                        is_institution = getattr(client, "is_institution", False)
                        institution_name = getattr(client, "institution_name", None)
                        if is_institution and institution_name:
                            display_name = institution_name
                        else:
                            display_name = (
                                full_name or getattr(user, "username", "") or "Client"
                            )
                except (ValueError, TypeError) as err:
                    company_mobile_dispatch_ns.abort(
                        400, "client_id doit être un entier."
                    )
                    raise AssertionError("Invalid client_id should abort") from err

            # ✅ Parser la date/heure
            try:
                scheduled_time = parse_local_naive(payload["scheduled_time"])
            except Exception as e:
                logger.exception(
                    "[MobileDispatchRides POST] Erreur lors du parsing de scheduled_time"
                )
                company_mobile_dispatch_ns.abort(400, f"scheduled_time invalide: {e}")
                raise AssertionError("Invalid scheduled_time should abort") from e

            # ✅ Créer la réservation aller
            from models.enums import BookingStatus

            booking = Booking()
            booking.customer_name = display_name
            if client:
                booking.client_id = client.id
                # ✅ Utiliser l'utilisateur du client (harmonisation avec route web)
                user = getattr(client, "user", None)
                if user:
                    booking.user_id = user.id
            booking.scheduled_time = scheduled_time
            booking.is_round_trip = bool(payload.get("is_return", False))
            booking.pickup_location = payload["pickup_address"]
            booking.dropoff_location = payload["dropoff_address"]
            booking.amount = (
                float(payload.get("amount", 0)) if payload.get("amount") else None
            )
            booking.status = BookingStatus.ACCEPTED
            booking.company_id = company_id
            booking.booking_type = "manual"
            booking.is_return = False
            booking.pickup_lat = payload.get("pickup_lat")
            booking.pickup_lon = payload.get("pickup_lon")
            booking.dropoff_lat = payload.get("dropoff_lat")
            booking.dropoff_lon = payload.get("dropoff_lon")
            booking.notes_medical = payload.get("notes") or None
            booking.wheelchair_client_has = bool(
                payload.get("wheelchair_client_has", False)
            )
            booking.wheelchair_need = bool(payload.get("wheelchair_need", False))

            # ✅ Priorité
            priority = payload.get("priority", "NORMAL")
            if priority == "HIGH":
                booking.is_urgent = True

            db.session.add(booking)
            db.session.flush()

            # ✅ Créer la course retour si nécessaire
            return_booking = None
            DATE_ONLY_LENGTH = 10  # Format YYYY-MM-DD
            is_round_trip_value = bool(payload.get("is_return", False))
            if is_round_trip_value and payload.get("return_time"):
                try:
                    return_time_str = payload["return_time"]
                    # Si c'est seulement une date (YYYY-MM-DD), pas d'heure
                    if len(return_time_str) == DATE_ONLY_LENGTH:
                        return_scheduled = None  # À confirmer plus tard
                    else:
                        return_scheduled = parse_local_naive(return_time_str)

                    return_booking = Booking()
                    return_booking.customer_name = display_name
                    if client:
                        return_booking.client_id = client.id
                        # ✅ Utiliser l'utilisateur du client (harmonisation avec route web)
                        user = getattr(client, "user", None)
                        if user:
                            return_booking.user_id = user.id
                    return_booking.scheduled_time = return_scheduled
                    return_booking.is_round_trip = True
                    return_booking.pickup_location = payload[
                        "dropoff_address"
                    ]  # Retour = inverse
                    return_booking.dropoff_location = payload["pickup_address"]
                    return_booking.amount = booking.amount
                    return_booking.status = BookingStatus.ACCEPTED
                    return_booking.company_id = company_id
                    return_booking.booking_type = "manual"
                    return_booking.is_return = True
                    return_booking.pickup_lat = payload.get("dropoff_lat")
                    return_booking.pickup_lon = payload.get("dropoff_lon")
                    return_booking.dropoff_lat = payload.get("pickup_lat")
                    return_booking.dropoff_lon = payload.get("pickup_lon")
                    return_booking.notes_medical = booking.notes_medical
                    return_booking.wheelchair_client_has = booking.wheelchair_client_has
                    return_booking.wheelchair_need = booking.wheelchair_need
                    return_booking.is_urgent = booking.is_urgent

                    db.session.add(return_booking)
                    db.session.flush()
                except Exception as e:
                    logger.exception(
                        "[MobileDispatchRides POST] Erreur lors de la création de la course retour"
                    )
                    db.session.rollback()
                    company_mobile_dispatch_ns.abort(
                        400, f"Erreur lors de la création de la course retour: {e}"
                    )
                    raise AssertionError("Return booking creation should abort") from e

            db.session.commit()

            # ✅ Construire les réponses
            summary = _build_ride_summary(booking, current_company_id=company_id)
            return_summary = None
            if return_booking:
                return_summary = _build_ride_summary(
                    return_booking, current_company_id=company_id
                )

            response = {
                "summary": summary,
            }
            if return_summary:
                response["return_summary"] = return_summary

            return response, 201

        except Exception as e:
            logger.exception(
                "[MobileDispatchRides POST] Erreur lors de la création de la course"
            )
            db.session.rollback()
            company_mobile_dispatch_ns.abort(
                500,
                "Une erreur interne s'est produite lors de la création de la course.",
            )
            raise AssertionError("Should have aborted") from e


@company_mobile_dispatch_ns.route("/v1/rides/<string:ride_id>")
class MobileRideDetail(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, ride_id: str):
        _, company_id = _get_company_context()

        try:
            booking_id = int(ride_id)
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(400, "ride_id invalide (entier attendu)")
            raise AssertionError("Invalid ride_id should abort") from exc

        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        booking = booking_repo.find_model_by_id_with_full_eager_loading(
            booking_id=booking_id, company_id=company_id
        )

        if booking is None:
            company_mobile_dispatch_ns.abort(
                404, "Course introuvable pour cette entreprise"
            )
            raise AssertionError(
                "abort() should have raised an exception"
            )  # Type hint: abort() lève une exception

        try:
            active_assignment = _get_active_assignment(booking)
            summary = _build_ride_summary(booking, current_company_id=company_id)
            suggestions = _compute_driver_suggestions(company_id, booking)
            history = _build_ride_history(booking, active_assignment)
            conflicts = _build_ride_conflicts(booking)

            notes: List[str] = []
            notes_dispatch = getattr(booking, "notes_medical", None)
            if notes_dispatch:
                notes.append(str(notes_dispatch))

            detail_payload = {
                "summary": summary,
                "suggestions": suggestions,
                "history": history,
                "conflicts": conflicts,
                "notes": notes,
            }

            return detail_payload, 200
        except Exception as e:
            logger.exception(
                "[MobileRideDetail] Erreur lors de la construction des détails de la course %s: %s",
                booking_id,
                e,
            )
            # ✅ Retourner un message d'erreur clair et informatif
            # Ne pas exposer les détails techniques de l'exception à l'utilisateur
            error_message = (
                "Impossible de charger les détails de la course. "
                "Veuillez réessayer ou contacter le support si le problème persiste."
            )
            company_mobile_dispatch_ns.abort(500, error_message)
            raise AssertionError("abort() should have raised an exception") from e


@company_mobile_dispatch_ns.route("/v1/drivers/available")
class MobileAvailableDrivers(Resource):
    """Endpoint pour récupérer tous les chauffeurs disponibles de l'entreprise."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère tous les chauffeurs disponibles de l'entreprise.

        Returns:
            Liste de tous les chauffeurs disponibles avec leurs informations de base.
        """
        _, company_id = _get_company_context()

        try:
            from repositories.driver_repository import DriverRepository

            driver_repo = DriverRepository()
            available_drivers = driver_repo.find_models_by_company_active_available_with_user_eager_loading(
                company_id=company_id
            )

            drivers_list: List[Dict[str, Any]] = []
            for driver in available_drivers:
                # ✅ Vérification supplémentaire : s'assurer que le driver appartient à l'entreprise
                driver_company_id = getattr(driver, "company_id", None)
                if driver_company_id is None or int(driver_company_id) != int(
                    company_id
                ):
                    logger.warning(
                        "[MobileAvailableDrivers] Driver %d n'appartient pas à l'entreprise %d (company_id=%s)",
                        driver.id,
                        company_id,
                        driver_company_id,
                    )
                    continue

                # ✅ Gérer le cas où driver_type pourrait être manquant ou invalide
                try:
                    if hasattr(driver, "driver_type"):
                        if hasattr(driver.driver_type, "value"):
                            driver_type = driver.driver_type.value
                        else:
                            driver_type = str(driver.driver_type)
                    else:
                        driver_type = DriverType.REGULAR.value  # Valeur par défaut
                except (AttributeError, TypeError):
                    driver_type = (
                        DriverType.REGULAR.value
                    )  # Valeur par défaut en cas d'erreur

                drivers_list.append(
                    {
                        "driver_id": str(driver.id),
                        "driver_name": _driver_display_name(driver),
                        "is_emergency": str(driver_type).upper()
                        == DriverType.EMERGENCY.value,
                        "driver_type": driver_type,
                    }
                )

            # Trier par nom pour faciliter la sélection
            drivers_list.sort(key=lambda d: d["driver_name"])

            return {"drivers": drivers_list}, 200

        except Exception as e:
            logger.exception(
                "[MobileAvailableDrivers] Erreur lors de la récupération des chauffeurs: %s",
                e,
            )
            company_mobile_dispatch_ns.abort(
                500, "Impossible de récupérer la liste des chauffeurs."
            )
            raise AssertionError("abort() should have raised an exception") from e


@company_mobile_dispatch_ns.route("/v1/rides/<string:ride_id>/assign")
class MobileRideAssign(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, ride_id: str):
        _, company_id = _get_company_context()

        try:
            booking_id = int(ride_id)
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(400, "ride_id invalide (entier attendu)")
            raise AssertionError("Invalid ride_id should abort") from exc

        payload = request.get_json(silent=True) or {}
        driver_id_raw = payload.get("driver_id")
        if driver_id_raw is None:
            company_mobile_dispatch_ns.abort(400, "driver_id manquant dans la requête")
            raise AssertionError("driver_id should not be None after abort") from None

        try:
            driver_id = int(driver_id_raw)
        except (TypeError, ValueError) as exc:
            company_mobile_dispatch_ns.abort(400, "driver_id invalide (entier attendu)")
            raise AssertionError("Invalid driver_id should abort") from exc

        response_payload = _execute_assignment_action(
            company_id=company_id,
            booking_id=booking_id,
            driver_id=driver_id,
            action_kind="mobile_assign",
        )

        return response_payload, 200


@company_mobile_dispatch_ns.route("/v1/rides/<string:ride_id>/reassign")
class MobileRideReassign(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, ride_id: str):
        _, company_id = _get_company_context()

        try:
            booking_id = int(ride_id)
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(400, "ride_id invalide (entier attendu)")
            raise AssertionError("Invalid ride_id should abort") from exc

        payload = request.get_json(silent=True) or {}
        driver_id_raw = payload.get("driver_id")
        if driver_id_raw is None:
            company_mobile_dispatch_ns.abort(400, "driver_id manquant dans la requête")
            raise AssertionError("driver_id should not be None after abort") from None

        try:
            driver_id = int(driver_id_raw)
        except (TypeError, ValueError) as exc:
            company_mobile_dispatch_ns.abort(400, "driver_id invalide (entier attendu)")
            raise AssertionError("Invalid driver_id should abort") from exc

        response_payload = _execute_assignment_action(
            company_id=company_id,
            booking_id=booking_id,
            driver_id=driver_id,
            action_kind="mobile_reassign",
        )

        return response_payload, 200


@company_mobile_dispatch_ns.route("/v1/rides/<string:ride_id>/schedule")
class MobileRideSchedule(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, ride_id: str):
        _, company_id = _get_company_context()

        try:
            booking_id = int(ride_id)
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(400, "ride_id invalide (entier attendu)")
            raise AssertionError("Invalid ride_id should abort") from exc

        payload = request.get_json(silent=True) or {}
        pickup_at_raw = payload.get("pickup_at")
        delta_minutes_raw = payload.get("delta_minutes")

        if pickup_at_raw is None and delta_minutes_raw is None:
            company_mobile_dispatch_ns.abort(
                400, "Il faut fournir soit pickup_at, soit delta_minutes."
            )
            raise AssertionError("Schedule payload invalid") from None

        new_datetime: datetime | None = None

        if pickup_at_raw is not None:
            try:
                parsed_dt = parse_local_naive(pickup_at_raw)
            except Exception as exc:
                company_mobile_dispatch_ns.abort(
                    400, "Format pickup_at invalide (ISO attendu)"
                )
                raise AssertionError("pickup_at parse error") from exc
            if parsed_dt is None:
                company_mobile_dispatch_ns.abort(400, "pickup_at ne peut pas être nul")
                raise AssertionError("pickup_at null") from None
            new_datetime = parsed_dt
        elif delta_minutes_raw is not None:
            try:
                delta_minutes = int(delta_minutes_raw)
            except (TypeError, ValueError) as exc:
                company_mobile_dispatch_ns.abort(
                    400, "delta_minutes doit être un entier"
                )
                raise AssertionError("delta parse error") from exc
            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            booking = booking_repo.find_model_by_id_and_company(
                booking_id=booking_id, company_id=company_id
            )
            if booking is None:
                company_mobile_dispatch_ns.abort(404, "Course introuvable")
                raise AssertionError("Booking not found") from None
            base_dt = booking.scheduled_time or now_local()
            new_datetime = base_dt + timedelta(minutes=delta_minutes)
        else:
            company_mobile_dispatch_ns.abort(400, "Paramètres planning invalides")
            raise AssertionError("Invalid scheduling parameters") from None

        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        booking = booking_repo.find_model_by_id_and_company(
            booking_id=booking_id, company_id=company_id
        )
        if booking is None:
            company_mobile_dispatch_ns.abort(404, "Course introuvable")
            raise AssertionError("Booking not found") from None

        booking.scheduled_time = new_datetime
        db.session.add(booking)
        db.session.commit()

        tools = AgentTools(company_id)
        scheduled_time_val = booking.scheduled_time
        scheduled_str = (
            scheduled_time_val.isoformat() if scheduled_time_val is not None else "None"
        )
        event_id = (
            _log_mobile_action(
                tools,
                "mobile_schedule",
                payload={
                    "booking_id": booking_id,
                    "scheduled_time": scheduled_time_val.isoformat()
                    if scheduled_time_val is not None
                    else None,
                    "source": "mobile_enterprise",
                },
                reasoning=f"Planification mobile {booking_id} -> {scheduled_str}",
            )
            or ""
        )

        return {
            "ride_id": str(booking_id),
            "scheduled_time": scheduled_time_val.isoformat()
            if scheduled_time_val is not None
            else None,
            "audit_event_id": event_id,
        }, 200


@company_mobile_dispatch_ns.route("/v1/rides/<string:ride_id>/urgent")
class MobileRideUrgent(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, ride_id: str):
        _, company_id = _get_company_context()

        try:
            booking_id = int(ride_id)
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(400, "ride_id invalide (entier attendu)")
            raise AssertionError("Invalid ride_id should abort") from exc

        payload = request.get_json(silent=True) or {}
        extra_delay_minutes_raw = payload.get("extra_delay_minutes", 15)
        reason = payload.get("reason")

        try:
            extra_delay_minutes = int(extra_delay_minutes_raw)
        except (TypeError, ValueError) as exc:
            company_mobile_dispatch_ns.abort(
                400, "extra_delay_minutes doit être un entier"
            )
            raise AssertionError("extra delay parse error") from exc

        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        booking = booking_repo.find_model_by_id_and_company(
            booking_id=booking_id, company_id=company_id
        )
        if booking is None:
            company_mobile_dispatch_ns.abort(404, "Course introuvable")
            raise AssertionError("Booking not found") from None

        booking.is_urgent = True
        
        # ✅ Calculer la nouvelle heure planifiée
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        
        # Si scheduled_time est None, à minuit (00:00), ou dans le passé,
        # utiliser l'heure actuelle + délai
        if not booking.scheduled_time:
            booking.scheduled_time = now + timedelta(minutes=extra_delay_minutes)
        else:
            # Vérifier si l'heure est à minuit (00:00)
            is_midnight = (
                booking.scheduled_time.hour == 0 
                and booking.scheduled_time.minute == 0
            )
            # Vérifier si l'heure est dans le passé
            is_past = booking.scheduled_time < now
            
            if is_midnight or is_past:
                # Utiliser l'heure actuelle + délai
                booking.scheduled_time = now + timedelta(minutes=extra_delay_minutes)
            else:
                # Ajouter le délai à l'heure existante
                booking.scheduled_time = booking.scheduled_time + timedelta(
                    minutes=extra_delay_minutes
                )

        db.session.add(booking)
        db.session.commit()

        tools = AgentTools(company_id)
        event_id = (
            _log_mobile_action(
                tools,
                "mobile_mark_urgent",
                payload={
                    "booking_id": booking_id,
                    "reason": reason,
                    "extra_delay_minutes": extra_delay_minutes,
                    "source": "mobile_enterprise",
                },
                reasoning=(
                    f"Marquage urgent mobile {booking_id} (+{extra_delay_minutes} min)"
                ),
            )
            or ""
        )

        return {
            "ride_id": str(booking_id),
            "is_urgent": True,
            "scheduled_time": booking.scheduled_time.isoformat()
            if booking.scheduled_time
            else None,
            "audit_event_id": event_id,
        }, 200


@company_mobile_dispatch_ns.route("/v1/mode")
class MobileDispatchMode(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, company_id = _get_company_context()
        mode_value = getattr(company.dispatch_mode, "value", "manual")
        return {
            "company_id": company_id,
            "dispatch_mode": mode_value,
            "autonomous_config": company.get_autonomous_config(),
        }, 200

    @jwt_required()
    @role_required(UserRole.company)
    def put(self):
        company, company_id = _get_company_context()
        payload = request.get_json(silent=True) or {}
        new_mode = payload.get("dispatch_mode")
        reason = payload.get("reason")

        if not new_mode:
            company_mobile_dispatch_ns.abort(400, "dispatch_mode requis")
            raise AssertionError("dispatch_mode required") from None

        try:
            target_mode = DispatchMode(new_mode)
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(
                400, "Mode invalide. Utilisez manual, semi_auto ou fully_auto."
            )
            raise AssertionError("Invalid dispatch mode") from exc

        previous_mode = getattr(company.dispatch_mode, "value", None)
        if previous_mode == target_mode.value:
            return {
                "company_id": company_id,
                "dispatch_mode": previous_mode,
                "previous_mode": previous_mode,
                "effective_at": datetime.now(UTC).isoformat(),
                "message": "Aucun changement (mode identique).",
            }, 200

        cast("Any", company).dispatch_mode = target_mode

        try:
            if target_mode.value == "fully_auto":
                agent = get_agent_for_company(
                    company_id,
                    app=current_app._get_current_object(),
                )
                if not agent.state.running:
                    agent.start()
                    log_msg = (
                        "[Dispatch-Mobile] Agent démarré automatiquement "
                        + "pour company %s (mode fully_auto)"
                    )
                    logger.info(
                        log_msg,
                        company_id,
                    )
            elif previous_mode == "fully_auto":
                stop_agent_for_company(company_id)
                logger.info(
                    "[Dispatch-Mobile] Agent arrêté pour company %s (mode %s)",
                    company_id,
                    target_mode.value,
                )
        except Exception as agent_err:
            logger.warning(
                "[Dispatch-Mobile] Erreur lors du contrôle agent (%s): %s",
                company_id,
                agent_err,
            )

        try:
            db.session.add(company)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.exception(
                "[Dispatch-Mobile] Échec mise à jour mode dispatch pour company %s",
                company_id,
            )
            company_mobile_dispatch_ns.abort(
                500, f"Impossible de mettre à jour le mode dispatch: {exc}"
            )
            raise AssertionError("Commit failed") from exc

        logger.info(
            "[Dispatch-Mobile] Company %s mode %s -> %s (reason=%s)",
            company_id,
            previous_mode,
            target_mode.value,
            reason,
        )

        return {
            "company_id": company_id,
            "dispatch_mode": target_mode.value,
            "previous_mode": previous_mode,
            "effective_at": datetime.now(UTC).isoformat(),
            "message": "Mode mis à jour avec succès.",
        }, 200


@company_mobile_dispatch_ns.route("/v1/settings")
class MobileDispatchSettings(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, _ = _get_company_context()
        return _serialize_dispatch_settings(company), 200

    @jwt_required()
    @role_required(UserRole.company)
    def put(self):
        company, company_id = _get_company_context()
        payload = request.get_json(silent=True) or {}

        if not isinstance(payload, dict):
            company_mobile_dispatch_ns.abort(400, "Payload JSON invalide.")
            raise AssertionError("Invalid payload should abort") from None

        autonomous_config = company.get_autonomous_config() or {}
        dispatch_overrides = autonomous_config.get("dispatch_overrides", {}) or {}

        updated = False

        fairness_update = payload.get("fairness")
        if fairness_update is not None:
            if not isinstance(fairness_update, dict):
                company_mobile_dispatch_ns.abort(
                    400, "fairness doit être un objet JSON."
                )
                raise AssertionError("Invalid fairness payload should abort") from None
            fairness_overrides = dispatch_overrides.get("fairness", {}) or {}
            if "max_gap" in fairness_update:
                try:
                    max_gap_val = int(fairness_update["max_gap"])
                except (TypeError, ValueError) as exc:
                    company_mobile_dispatch_ns.abort(
                        400, "fairness.max_gap doit être un entier."
                    )
                    raise AssertionError("Invalid fairness value should abort") from exc
                fairness_overrides["max_gap"] = max_gap_val
                updated = True
            if fairness_overrides:
                dispatch_overrides["fairness"] = fairness_overrides

        emergency_update = payload.get("emergency")
        if emergency_update is not None:
            if not isinstance(emergency_update, dict):
                company_mobile_dispatch_ns.abort(
                    400, "emergency doit être un objet JSON."
                )
                raise AssertionError("Invalid emergency payload should abort") from None
            emergency_overrides = dispatch_overrides.get("emergency", {}) or {}
            if "emergency_penalty" in emergency_update:
                try:
                    penalty_val = float(emergency_update["emergency_penalty"])
                except (TypeError, ValueError) as exc:
                    company_mobile_dispatch_ns.abort(
                        400, "emergency.emergency_penalty doit être numérique."
                    )
                    raise AssertionError(
                        "Invalid emergency value should abort"
                    ) from exc
                emergency_overrides["emergency_penalty"] = penalty_val
                updated = True
            if emergency_overrides:
                dispatch_overrides["emergency"] = emergency_overrides

        service_times_update = payload.get("service_times")
        if service_times_update is not None:
            if not isinstance(service_times_update, dict):
                company_mobile_dispatch_ns.abort(
                    400, "service_times doit être un objet JSON."
                )
                raise AssertionError(
                    "Invalid service_times payload should abort"
                ) from None
            service_overrides = dispatch_overrides.get("service_times", {}) or {}
            for key in (
                "pickup_service_min",
                "dropoff_service_min",
                "min_transition_margin_min",
            ):
                if key in service_times_update:
                    try:
                        service_overrides[key] = int(service_times_update[key])
                    except (TypeError, ValueError) as exc:
                        company_mobile_dispatch_ns.abort(
                            400, f"service_times.{key} doit être un entier."
                        )
                        raise AssertionError(
                            "Invalid service_times value should abort"
                        ) from exc
                    updated = True
            if service_overrides:
                dispatch_overrides["service_times"] = service_overrides

        if not updated:
            return {"message": "Aucune modification détectée"}, 200

        autonomous_config["dispatch_overrides"] = dispatch_overrides
        company.set_autonomous_config(autonomous_config)

        try:
            db.session.add(company)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.exception(
                "[MobileDispatch] Échec mise à jour paramètres mobile: %s", exc
            )
            company_mobile_dispatch_ns.abort(
                500, "Impossible de sauvegarder les paramètres."
            )
            raise AssertionError("Settings update failed after abort") from exc

        tools = AgentTools(company_id)
        _log_mobile_action(
            tools,
            "mobile_update_settings",
            payload={
                "fairness": fairness_update,
                "emergency": emergency_update,
                "service_times": service_times_update,
                "source": "mobile_enterprise",
            },
            reasoning="Mise à jour des paramètres dispatch via mobile",
        )

        # Rafraîchir les settings après commit
        db.session.refresh(company)
        return _serialize_dispatch_settings(company), 200


@company_mobile_dispatch_ns.route("/v1/run")
class MobileDispatchRun(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit(
        "10000 per hour"
    )  # ⚠️ C2: Augmenté temporairement pour load testing (normalement 10/minute)
    def post(self):
        company, company_id = _get_company_context()
        body = request.get_json(silent=True) or {}
        target_date = body.get("date") or now_local().strftime("%Y-%m-%d")

        try:
            datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(
                400, "Format de date invalide (YYYY-MM-DD)."
            )
            raise AssertionError("Invalid dispatch date should abort") from exc

        params: Dict[str, Any] = {
            "company_id": company_id,
            "for_date": target_date,
            "regular_first": True,
            "allow_emergency": True,
            "mode": "auto",
            "dispatch_overrides": {
                "fairness": {
                    "enable_fairness": True,
                    "fairness_window_days": 2,
                    "fairness_weight": 0.7,
                    "reset_daily_load": True,
                },
                "heuristic": {
                    "driver_load_balance": 0.7,
                    "proximity": 0.2,
                    "priority": 0.08,
                    "return_urgency": 0.02,
                },
                "solver": {
                    "max_bookings_per_driver": 999,
                },
                "emergency": {
                    "allow_emergency_drivers": True,
                    "emergency_penalty": 600.0,
                },
            },
        }

        dispatch_overrides = params.get("dispatch_overrides") or {}
        if dispatch_overrides:
            try:
                base_settings = dispatch_settings.for_company(company)
                _, validation = dispatch_settings.merge_overrides(
                    base_settings,
                    dispatch_overrides,
                    return_validation=True,
                )
                log_msg = (
                    "[MobileDispatch] Validation overrides: "
                    + "applied=%s ignored=%s errors=%s"
                )
                logger.info(
                    log_msg,
                    validation.get("applied"),
                    validation.get("ignored"),
                    validation.get("errors"),
                )
                critical_errors = validation.get("critical_errors", [])
                if critical_errors:
                    message = "Paramètres critiques ignorés: " + ", ".join(
                        critical_errors
                    )
                    logger.warning(
                        "[MobileDispatch] Overrides rejetés (critique): %s",
                        critical_errors,
                    )
                    company_mobile_dispatch_ns.abort(400, message)
            except ValueError as exc:
                logger.exception(
                    "[MobileDispatch] Validation overrides échouée: %s", exc
                )
                company_mobile_dispatch_ns.abort(400, "Paramètres overrides invalides.")
            params["overrides"] = dispatch_overrides
            params.pop("dispatch_overrides", None)

        job = trigger_job(company_id, params)

        tools = AgentTools(company_id)
        _log_mobile_action(
            tools,
            "mobile_run_dispatch",
            payload={"params": params, "source": "mobile_enterprise"},
            reasoning=f"Lancement dispatch mobile pour {target_date}",
        )

        response = {
            "message": f"Dispatch lancé pour {target_date}",
            "job": job,
            "for_date": target_date,
        }
        return response, 202


@company_mobile_dispatch_ns.route("/v1/optimizer/run")
class MobileOptimizerRun(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit(
        "10000 per hour"
    )  # ⚠️ C2: Augmenté temporairement pour load testing (normalement 10/minute)
    def post(self):
        _, company_id = _get_company_context()
        body = request.get_json(silent=True) or {}
        target_date = body.get("date")
        if target_date:
            try:
                datetime.strptime(target_date, "%Y-%m-%d")
            except ValueError as exc:
                company_mobile_dispatch_ns.abort(
                    400, "Format de date invalide (YYYY-MM-DD)."
                )
                raise AssertionError("Invalid optimizer date should abort") from exc

        opportunities = check_opportunities_manual(
            company_id,
            target_date,
            app=current_app._get_current_object(),
        )
        payload = [opp.to_dict() for opp in opportunities]

        tools = AgentTools(company_id)
        _log_mobile_action(
            tools,
            "mobile_optimizer_manual",
            payload={
                "count": len(payload),
                "for_date": target_date,
                "source": "mobile_enterprise",
            },
            reasoning=f"Relance optimiseur mobile ({len(payload)} opportunités)",
        )

        return {
            "message": "Optimisation recalculée",
            "count": len(payload),
            "opportunities": payload,
            "for_date": target_date,
        }, 200


@company_mobile_dispatch_ns.route("/v1/reset")
class MobileDispatchReset(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        _, company_id = _get_company_context()
        body = request.get_json(silent=True) or {}
        date_str = body.get("date")

        start_datetime: datetime | None = None
        end_datetime: datetime | None = None

        if date_str:
            try:
                target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError as exc:
                company_mobile_dispatch_ns.abort(
                    400, "Format de date invalide. Utilisez YYYY-MM-DD."
                )
                raise AssertionError("Invalid reset date should abort") from exc
            start_datetime = datetime.combine(target_date, datetime.min.time()).replace(
                tzinfo=UTC
            )
            end_datetime = datetime.combine(target_date, datetime.max.time()).replace(
                tzinfo=UTC
            )

        try:
            from repositories.assignment_repository import AssignmentRepository

            assignment_repo = AssignmentRepository()
            query = assignment_repo.find_models_by_company_with_date_filter_query(
                company_id=company_id,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
            )

            assignments = query.all()
            booking_ids = [assignment.booking_id for assignment in assignments]

            assignments_deleted = len(assignments)
            for assignment in assignments:
                db.session.delete(assignment)

            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            bookings_query = booking_repo.find_models_by_company_with_filters_query(
                company_id=company_id,
                booking_ids=booking_ids if booking_ids else None,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
            )

            bookings_reset = 0
            for booking in bookings_query.all():
                if booking.status == BookingStatus.ASSIGNED:
                    booking.status = BookingStatus.ACCEPTED
                    booking.driver_id = None
                    bookings_reset += 1

            db.session.commit()

            tools = AgentTools(company_id)
            _log_mobile_action(
                tools,
                "mobile_reset_assignments",
                payload={
                    "date": date_str,
                    "assignments_deleted": assignments_deleted,
                    "bookings_reset": bookings_reset,
                    "source": "mobile_enterprise",
                },
                reasoning="Réinitialisation des assignations via mobile",
            )

            return {
                "message": "Réinitialisation effectuée",
                "assignments_deleted": assignments_deleted,
                "bookings_reset": bookings_reset,
                "date": date_str or "toutes les dates",
            }, 200
        except Exception as exc:
            db.session.rollback()
            logger.exception("[MobileDispatch] Erreur reset mobile: %s", exc)
            company_mobile_dispatch_ns.abort(500, "Erreur lors de la réinitialisation.")
            raise AssertionError("Reset failed after abort") from exc


@company_mobile_dispatch_ns.route("/v1/incidents")
class MobileDispatchIncidents(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        _, company_id = _get_company_context()
        payload = request.get_json(silent=True) or {}

        if not isinstance(payload, dict):
            company_mobile_dispatch_ns.abort(400, "Payload JSON invalide.")
            raise AssertionError("Invalid incident payload should abort") from None

        incident_type = payload.get("type") or "incident"
        severity = payload.get("severity") or "medium"

        ride_id_raw = payload.get("ride_id")
        driver_id_raw = payload.get("driver_id")
        booking_id = None
        driver_id = None

        if ride_id_raw is not None:
            try:
                booking_id = int(ride_id_raw)
            except (TypeError, ValueError) as exc:
                company_mobile_dispatch_ns.abort(400, "ride_id doit être un entier.")
                raise AssertionError("Invalid ride_id should abort") from exc

        if driver_id_raw is not None:
            try:
                driver_id = int(driver_id_raw)
            except (TypeError, ValueError) as exc:
                company_mobile_dispatch_ns.abort(400, "driver_id doit être un entier.")
                raise AssertionError("Invalid driver_id should abort") from exc

        note = payload.get("note")
        attachments = payload.get("attachments") or []
        if attachments and not isinstance(attachments, list):
            company_mobile_dispatch_ns.abort(400, "attachments doit être une liste.")
            raise AssertionError("Invalid attachments should abort") from None

        action = AutonomousAction()
        action.company_id = company_id
        action.booking_id = booking_id
        action.driver_id = driver_id
        action.action_type = "mobile_incident"
        action.action_description = (
            f"Incident mobile: {incident_type} (severity={severity})"
        )
        action.action_data = json.dumps(
            {
                "type": incident_type,
                "severity": severity,
                "note": note,
                "attachments": attachments,
                "source": "mobile_enterprise",
            }
        )
        action.trigger_source = "mobile_enterprise"
        action.success = True

        try:
            db.session.add(action)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.exception(
                "[MobileDispatch] Échec enregistrement incident mobile: %s", exc
            )
            company_mobile_dispatch_ns.abort(
                500, "Impossible d'enregistrer l'incident."
            )
            raise AssertionError("Incident insert failed after abort") from exc

        tools = AgentTools(company_id)
        _log_mobile_action(
            tools,
            "mobile_incident_report",
            payload={
                "incident_id": action.id,
                "type": incident_type,
                "severity": severity,
                "booking_id": booking_id,
                "driver_id": driver_id,
                "source": "mobile_enterprise",
            },
            reasoning=f"Signalement incident ({incident_type}) via mobile",
        )

        return {
            "message": "Incident enregistré",
            "incident_id": action.id,
        }, 201


@company_mobile_dispatch_ns.route("/v1/chat/messages")
class MobileDispatchChat(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        _, company_id = _get_company_context()

        try:
            limit = max(1, min(100, int(request.args.get("limit", 50))))
        except (TypeError, ValueError):
            company_mobile_dispatch_ns.abort(400, "Paramètre limit invalide.")
            raise AssertionError("Invalid chat limit should abort") from None

        before = request.args.get("before")
        before_dt: datetime | None = None
        if before:
            try:
                before_dt = datetime.fromisoformat(before.rstrip("Z"))
            except ValueError as exc:
                company_mobile_dispatch_ns.abort(
                    400, "Paramètre before invalide (ISO8601 attendu)."
                )
                raise AssertionError("Invalid before timestamp should abort") from exc

        from repositories.message_repository import MessageRepository

        message_repo = MessageRepository()
        messages = message_repo.find_models_by_company_with_timestamp_filter(
            company_id=company_id, before_timestamp=before_dt
        )
        # Limiter et inverser l'ordre
        messages = list(reversed(messages[:limit]))
        serialized = []
        for message in messages:
            try:
                serialized.append(message.serialize)
            except Exception:
                serialized.append(
                    {
                        "id": message.id,
                        "content": message.content,
                        "timestamp": (
                            message.timestamp.isoformat()
                            if getattr(message, "timestamp", None) is not None
                            else None
                        ),
                        "sender_role": getattr(
                            message.sender_role, "value", message.sender_role
                        ),
                        "sender_id": message.sender_id,
                        "receiver_id": message.receiver_id,
                    }
                )

        return {
            "messages": serialized,
            "count": len(serialized),
        }, 200

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        _, company_id = _get_company_context()
        user = _get_current_user()
        payload = request.get_json(silent=True) or {}

        content = payload.get("content")
        if not content or not str(content).strip():
            company_mobile_dispatch_ns.abort(400, "Le champ content est requis.")
            raise AssertionError("Chat content invalid should abort") from None

        receiver_id = None
        if "receiver_id" in payload and payload["receiver_id"] is not None:
            try:
                receiver_id = int(payload["receiver_id"])
            except (TypeError, ValueError) as exc:
                company_mobile_dispatch_ns.abort(
                    400, "receiver_id doit être un entier."
                )
                raise AssertionError("Invalid receiver_id should abort") from exc

        message = Message()
        message.company_id = company_id
        message.sender_id = getattr(user, "id", None)
        message.receiver_id = receiver_id
        message.sender_role = SenderRole.COMPANY
        message.content = str(content)

        try:
            db.session.add(message)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.exception("[MobileDispatch] Échec envoi message mobile: %s", exc)
            company_mobile_dispatch_ns.abort(500, "Impossible d'envoyer le message.")
            raise AssertionError("Chat insert failed after abort") from exc

        tools = AgentTools(company_id)
        _log_mobile_action(
            tools,
            "mobile_chat_message",
            payload={
                "message_id": message.id,
                "receiver_id": receiver_id,
                "source": "mobile_enterprise",
            },
            reasoning="Message envoyé via app mobile entreprise",
        )

        return message.serialize, 201


# Constantes pour le calcul des retards
DELAY_MINUTES_THRESHOLD = 5
DELAY_MINUTES_ZERO = 0


@company_mobile_dispatch_ns.route("/v1/dashboard/realtime")
class MobileRealtimeDashboard(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @company_mobile_dispatch_ns.doc(
        params={"date": "YYYY-MM-DD (optionnel, défaut: aujourd'hui)"}
    )
    def get(self):
        """Dashboard temps réel pour les dispatchers mobile.
        Combine métriques de qualité, retards, opportunités et charge chauffeurs.
        """
        _, company_id = _get_company_context()

        date_str = request.args.get("date")
        if not date_str:
            date_str = datetime.now(UTC).date().strftime("%Y-%m-%d")

        try:
            # 1. Métriques de qualité du dernier dispatch
            quality_metrics = None
            try:
                from infrastructure.dispatch.dispatch_metrics_adapter import (
                    DispatchMetricsCollector,
                )

                collector = DispatchMetricsCollector(company_id)
                metrics = collector.collect_for_date(date_str)
                quality_metrics = metrics.to_summary()
            except Exception as e:
                # ✅ Log plus informatif : distinguer absence de DispatchRun vs vraie erreur
                if "No DispatchRun found" in str(e):
                    logger.info(
                        "[Dashboard] No dispatch data for company %s on %s (normal if no dispatch run yet)",
                        company_id,
                        date_str,
                    )
                else:
                    logger.warning(
                        "[Dashboard] Failed to get quality metrics for company %s: %s",
                        company_id,
                        e,
                    )
                # Retourner des métriques par défaut dans tous les cas
                quality_metrics = {
                    "quality_score": 0,
                    "assignment_rate": 0,
                    "on_time_rate": 0,
                    "pooling_rate": 0,
                    "fairness": 0,
                    "avg_delay": 0,
                }

            # 2. Retards en cours (live)
            assigns = []
            try:
                d0, d1 = day_local_bounds(date_str)
                from repositories.assignment_repository import AssignmentRepository

                assignment_repo = AssignmentRepository()
                assigns = assignment_repo.find_models_by_company_with_time_range_and_excluded_statuses(
                    company_id=company_id,
                    start_datetime=d0,
                    end_datetime=d1,
                    excluded_statuses=[
                        BookingStatus.COMPLETED,
                        BookingStatus.RETURN_COMPLETED,
                        BookingStatus.CANCELED,
                    ],
                )

                current_delays = []
                from repositories.booking_repository import BookingRepository

                booking_repo = BookingRepository()
                for a in assigns:
                    b = booking_repo.find_model_by_id(
                        booking_id=cast(int, a.booking_id)
                    )
                    if not b or not b.scheduled_time:
                        continue

                    # Calculer retard simplifié
                    current_time = now_local()
                    if a.eta_pickup_at and b.scheduled_time:
                        delay_minutes = int(
                            (a.eta_pickup_at - b.scheduled_time).total_seconds() / 60
                        )
                    else:
                        # Fallback: comparer heure actuelle vs scheduled_time
                        delay_minutes = int(
                            (current_time - b.scheduled_time).total_seconds() / 60
                        )

                    if abs(delay_minutes) >= DELAY_MINUTES_THRESHOLD:
                        current_delays.append(
                            {
                                "assignment_id": a.id,
                                "booking_id": b.id,
                                "driver_id": a.driver_id,
                                "delay_minutes": delay_minutes,
                                "status": "late"
                                if delay_minutes > DELAY_MINUTES_ZERO
                                else "early",
                                "customer_name": b.customer_name,
                                "scheduled_time": b.scheduled_time.isoformat()
                                if b.scheduled_time
                                else None,
                            }
                        )

                # Trier par retard décroissant
                current_delays.sort(key=lambda x: -abs(x["delay_minutes"]))

            except Exception as e:
                logger.warning("[Dashboard] Failed to get current delays: %s", e)
                current_delays = []

            # 3. Opportunités d'optimisation
            opportunities = []
            try:
                optimizer = get_optimizer_for_company(company_id)
                if optimizer:
                    # ✅ Fix: Vérifier que get_status() retourne un dict avec la clé "running"
                    optimizer_status = optimizer.get_status()
                    if isinstance(optimizer_status, dict) and optimizer_status.get(
                        "running"
                    ):
                        opportunities = [
                            o.to_dict() for o in optimizer.get_current_opportunities()
                        ]
                    else:
                        # Optimizer existe mais n'est pas en mode running
                        opportunities = [
                            o.to_dict()
                            for o in check_opportunities_manual(company_id, date_str)
                        ]
                else:
                    # Pas d'optimizer actif, vérification manuelle
                    opportunities = [
                        o.to_dict()
                        for o in check_opportunities_manual(company_id, date_str)
                    ]
            except Exception as e:
                logger.warning("[Dashboard] Failed to get opportunities: %s", e)
                # ✅ Fix: Log complet de l'exception pour debugging
                logger.exception("[Dashboard] Detailed error getting opportunities")

            # 4. Charge par chauffeur
            driver_load = {}
            try:
                for a in assigns:
                    if bool(a.driver_id):
                        driver_load[a.driver_id] = driver_load.get(a.driver_id, 0) + 1

                # Enrichir avec infos chauffeur
                driver_load_details = []
                from repositories.driver_repository import DriverRepository

                driver_repo = DriverRepository()
                for driver_id, count in driver_load.items():
                    driver = driver_repo.find_model_by_id(driver_id=driver_id)
                    if driver and driver.user:
                        driver_load_details.append(
                            {
                                "driver_id": driver_id,
                                "name": (
                                    f"{driver.user.first_name} {driver.user.last_name}"
                                ),
                                "bookings_count": count,
                                "is_emergency": getattr(driver, "is_emergency", False),
                            }
                        )

                # Trier par charge décroissante
                driver_load_details.sort(key=lambda x: -x["bookings_count"])

            except Exception as e:
                logger.warning("[Dashboard] Failed to get driver load: %s", e)
                driver_load_details = []

            # 5. Statistiques rapides
            stats = {
                "total_bookings": len(assigns),
                "delayed_bookings": len(
                    [d for d in current_delays if d["status"] == "late"]
                ),
                "early_bookings": len(
                    [d for d in current_delays if d["status"] == "early"]
                ),
                "on_time_bookings": len(assigns) - len(current_delays),
                "critical_opportunities": len(
                    [o for o in opportunities if o.get("severity") == "critical"]
                ),
                "drivers_active": len(driver_load),
            }

            return (
                {
                    "date": date_str,
                    "timestamp": now_local().isoformat(),
                    "quality_metrics": quality_metrics,
                    "current_delays": current_delays[:20],  # Top 20
                    "opportunities": opportunities[:10],  # Top 10
                    "driver_load": driver_load_details[:15],  # Top 15
                    "stats": stats,
                },
                200,
            )

        except Exception as e:
            logger.exception(
                "[Dashboard] Failed to build realtime dashboard for company %s: %s",
                company_id,
                str(e),
            )
            # ✅ Fix: Retourner une réponse JSON valide avec message d'erreur détaillé
            return {
                "error": "Failed to load dashboard",
                "details": str(e),
                "date": date_str,
                "timestamp": datetime.now(UTC).isoformat(),
                # Retourner des données par défaut pour éviter crash frontend
                "quality_metrics": {
                    "quality_score": 0,
                    "assignment_rate": 0,
                    "on_time_rate": 0,
                    "pooling_rate": 0,
                    "fairness": 0,
                    "avg_delay": 0,
                },
                "current_delays": [],
                "opportunities": [],
                "driver_load": [],
                "stats": {
                    "total_bookings": 0,
                    "delayed_bookings": 0,
                    "early_bookings": 0,
                    "on_time_bookings": 0,
                    "critical_opportunities": 0,
                    "drivers_active": 0,
                },
            }, 500


# =====================================================
# Endpoints pour création, modification et annulation de courses
# =====================================================


@company_mobile_dispatch_ns.route("/v1/rides")
class MobileCreateRide(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("100 per hour")
    def post(self):
        """Crée une nouvelle course pour l'entreprise."""
        company, company_id = _get_company_context()
        payload = request.get_json(silent=True) or {}

        # Validation des champs requis : client_id OU customer_name
        client_id_raw = payload.get("client_id")
        customer_name = payload.get("customer_name", "").strip()

        client = None
        client_id = None
        user = None

        if client_id_raw:
            # Si client_id est fourni, l'utiliser
            try:
                client_id = int(client_id_raw)
            except (TypeError, ValueError) as exc:
                company_mobile_dispatch_ns.abort(400, "client_id doit être un entier")
                raise AssertionError("Invalid client_id") from exc

            # Vérifier que le client appartient à l'entreprise
            from repositories.client_repository import ClientRepository

            client_repo = ClientRepository()
            client = client_repo.find_model_by_id_and_company(
                client_id=client_id, company_id=company_id
            )
            if not client:
                company_mobile_dispatch_ns.abort(404, "Client introuvable")
                raise AssertionError("Client not found") from None

            user = client.user
            if not user:
                company_mobile_dispatch_ns.abort(
                    404, "Utilisateur associé au client introuvable"
                )
                raise AssertionError("User not found") from None
        elif not customer_name:
            # Si ni client_id ni customer_name n'est fourni
            company_mobile_dispatch_ns.abort(400, "client_id ou customer_name requis")
            raise AssertionError("client_id or customer_name required") from None
        else:
            # Si seulement customer_name est fourni, créer un client temporaire
            import uuid

            # Créer un utilisateur temporaire
            temp_user = User()
            temp_user.public_id = str(uuid.uuid4())
            temp_user.username = (
                f"temp_{customer_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
            )
            temp_user.email = f"temp_{uuid.uuid4().hex[:8]}@temp.local"
            temp_user.first_name = (
                customer_name.split()[0] if customer_name.split() else customer_name
            )
            temp_user.last_name = (
                " ".join(customer_name.split()[1:])
                if len(customer_name.split()) > 1
                else ""
            )
            temp_user.role = UserRole.client
            temp_user.set_password(str(uuid.uuid4()))  # Mot de passe aléatoire

            db.session.add(temp_user)
            db.session.flush()  # Pour obtenir l'ID

            # Créer un client temporaire
            temp_client = Client()
            temp_client.user_id = temp_user.id
            temp_client.company_id = company_id
            temp_client.client_type = ClientType.PRIVATE
            temp_client.is_active = True

            db.session.add(temp_client)
            db.session.flush()  # Pour obtenir l'ID

            client = temp_client
            client_id = temp_client.id
            user = temp_user

        # Validation des adresses
        pickup_address = payload.get("pickup_address", "").strip()
        dropoff_address = payload.get("dropoff_address", "").strip()
        if not pickup_address or not dropoff_address:
            company_mobile_dispatch_ns.abort(
                400, "pickup_address et dropoff_address requis"
            )
            raise AssertionError("Addresses required") from None

        # Validation de la date/heure
        scheduled_time_str = payload.get("scheduled_time")
        if not scheduled_time_str:
            company_mobile_dispatch_ns.abort(400, "scheduled_time requis")
            raise AssertionError("scheduled_time required") from None

        try:
            scheduled_time = parse_local_naive(scheduled_time_str)
        except Exception as exc:
            company_mobile_dispatch_ns.abort(
                400, f"Format scheduled_time invalide: {exc}"
            )
            raise AssertionError("Invalid scheduled_time") from exc

        # Coordonnées GPS
        pickup_lat = payload.get("pickup_lat")
        pickup_lon = payload.get("pickup_lon")
        dropoff_lat = payload.get("dropoff_lat")
        dropoff_lon = payload.get("dropoff_lon")

        # Géocodage si coordonnées manquantes
        if not pickup_lat or not pickup_lon:
            try:
                from services.geolocation.maps import geocode_address

                pickup_coords = geocode_address(pickup_address)
                if pickup_coords:
                    pickup_lat = pickup_coords.get("lat")
                    pickup_lon = pickup_coords.get("lon")
            except Exception as e:
                logger.warning("[MobileCreateRide] Géocodage pickup échoué: %s", e)

        if not dropoff_lat or not dropoff_lon:
            try:
                from services.geolocation.maps import geocode_address

                dropoff_coords = geocode_address(dropoff_address)
                if dropoff_coords:
                    dropoff_lat = dropoff_coords.get("lat")
                    dropoff_lon = dropoff_coords.get("lon")
            except Exception as e:
                logger.warning("[MobileCreateRide] Géocodage dropoff échoué: %s", e)

        # Calcul distance/durée avec OSRM (best-effort)
        dur_s = None
        dist_m = None
        if pickup_lat and pickup_lon and dropoff_lat and dropoff_lon:
            try:
                from config import Config
                from services.geolocation.osrm import _route

                osrm_url = getattr(Config, "UD_OSRM_URL", "http://osrm:5000")
                route_data = _route(
                    base_url=osrm_url,
                    profile="driving",
                    origin=(float(pickup_lat), float(pickup_lon)),
                    destination=(float(dropoff_lat), float(dropoff_lon)),
                    timeout=2,
                    overview="false",
                    geometries="geojson",
                    steps=False,
                    annotations=False,
                )
                if route_data.get("code") == "Ok" and route_data.get("routes"):
                    r0 = route_data["routes"][0]
                    dur_s = int(r0.get("duration", 0))
                    dist_m = int(r0.get("distance", 0))
            except Exception as e:
                logger.warning("[MobileCreateRide] OSRM échoué: %s", e)

        # Nom du client
        if client and user:
            full_name = (
                f"{getattr(user, 'first_name', '')} {getattr(user, 'last_name', '')}"
            ).strip()
            if not full_name:
                full_name = getattr(user, "username", "") or "Client"
        else:
            # Utiliser customer_name fourni
            full_name = customer_name or "Client"

        # Créer la réservation
        new_booking = Booking()
        new_booking.customer_name = full_name
        new_booking.client_id = client_id if client_id else None
        new_booking.company_id = company_id
        new_booking.scheduled_time = scheduled_time
        new_booking.pickup_location = pickup_address
        new_booking.dropoff_location = dropoff_address
        new_booking.pickup_lat = float(pickup_lat) if pickup_lat else None
        new_booking.pickup_lon = float(pickup_lon) if pickup_lon else None
        new_booking.dropoff_lat = float(dropoff_lat) if dropoff_lat else None
        new_booking.dropoff_lon = float(dropoff_lon) if dropoff_lon else None
        new_booking.status = BookingStatus.ACCEPTED
        new_booking.booking_type = "manual"
        new_booking.user_id = getattr(company, "user_id", None)
        new_booking.duration_seconds = dur_s
        new_booking.distance_meters = dist_m
        new_booking.amount = float(payload.get("amount", 0))
        new_booking.is_urgent = payload.get("priority") == "HIGH"
        new_booking.notes_medical = payload.get("notes", "").strip() or None
        new_booking.wheelchair_client_has = bool(
            payload.get("wheelchair_client_has", False)
        )
        new_booking.wheelchair_need = bool(payload.get("wheelchair_need", False))
        new_booking.is_return = False  # C'est la course aller
        new_booking.is_round_trip = bool(payload.get("is_return", False))

        # Gestion de la course retour si demandée
        is_return_trip = bool(payload.get("is_return", False))
        return_time_str = payload.get("return_time")
        return_booking = None

        if is_return_trip:
            # Parser la date/heure de retour
            return_time = None
            return_time_confirmed = True
            if return_time_str:
                try:
                    return_time = parse_local_naive(return_time_str)
                    return_time_confirmed = True
                except Exception as e:
                    logger.warning(
                        "[MobileCreateRide] Erreur parsing return_time: %s", e
                    )
                    # Si l'heure n'est pas valide, créer quand même avec date à minuit
                    try:
                        # Essayer d'extraire juste la date
                        date_part = return_time_str.split("T")[0]
                        return_time = parse_local_naive(f"{date_part}T00:00:00")
                        return_time_confirmed = False
                    except Exception:
                        pass

            # Créer la course retour
            return_booking = Booking()
            return_booking.parent_booking_id = None  # Sera mis à jour après commit
            return_booking.customer_name = full_name
            return_booking.client_id = client_id if client_id else None
            return_booking.company_id = company_id
            return_booking.scheduled_time = return_time
            return_booking.pickup_location = dropoff_address  # Inversé
            return_booking.dropoff_location = pickup_address  # Inversé
            return_booking.pickup_lat = float(dropoff_lat) if dropoff_lat else None
            return_booking.pickup_lon = float(dropoff_lon) if dropoff_lon else None
            return_booking.dropoff_lat = float(pickup_lat) if pickup_lat else None
            return_booking.dropoff_lon = float(pickup_lon) if pickup_lon else None
            return_booking.status = BookingStatus.ACCEPTED
            return_booking.booking_type = "manual"
            return_booking.user_id = getattr(company, "user_id", None)
            return_booking.is_return = True
            return_booking.is_round_trip = False
            return_booking.duration_seconds = dur_s
            return_booking.distance_meters = dist_m
            return_booking.amount = float(payload.get("amount", 0))
            return_booking.is_urgent = payload.get("priority") == "HIGH"
            return_booking.notes_medical = payload.get("notes", "").strip() or None
            return_booking.wheelchair_client_has = bool(
                payload.get("wheelchair_client_has", False)
            )
            return_booking.wheelchair_need = bool(payload.get("wheelchair_need", False))
            return_booking.time_confirmed = return_time_confirmed

        try:
            db.session.add(new_booking)
            db.session.flush()  # Pour obtenir l'ID de la course aller

            # Lier la course retour à la course aller
            if return_booking:
                return_booking.parent_booking_id = new_booking.id
                db.session.add(return_booking)

            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.exception("[MobileCreateRide] Échec création: %s", exc)
            company_mobile_dispatch_ns.abort(500, "Impossible de créer la course")
            raise AssertionError("Create failed") from exc

        # Journaliser l'action
        tools = AgentTools(company_id)
        _log_mobile_action(
            tools,
            "mobile_create_ride",
            payload={
                "booking_id": new_booking.id,
                "client_id": client_id,
                "is_return": is_return_trip,
                "return_booking_id": return_booking.id if return_booking else None,
                "source": "mobile_enterprise",
            },
            reasoning=f"Création course mobile {new_booking.id}"
            + (f" + retour {return_booking.id}" if return_booking else ""),
        )

        # Retourner le détail de la course créée
        summary = _build_ride_summary(new_booking, current_company_id=company_id)
        result = {"summary": summary}
        if return_booking:
            return_summary = _build_ride_summary(
                return_booking, current_company_id=company_id
            )
            result["return_summary"] = return_summary
        return result, 201


@company_mobile_dispatch_ns.route("/v1/rides/<string:ride_id>")
class MobileUpdateRide(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("100 per hour")
    def put(self, ride_id: str):
        """Met à jour une course existante."""
        _, company_id = _get_company_context()

        try:
            booking_id = int(ride_id)
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(400, "ride_id invalide (entier attendu)")
            raise AssertionError("Invalid ride_id") from exc

        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        booking = booking_repo.find_model_by_id_and_company(
            booking_id=booking_id, company_id=company_id
        )
        if not booking:
            company_mobile_dispatch_ns.abort(404, "Course introuvable")
            raise AssertionError("Booking not found") from None

        payload = request.get_json(silent=True) or {}

        # Mise à jour des adresses
        if "pickup_address" in payload:
            booking.pickup_location = payload["pickup_address"]
        if "dropoff_address" in payload:
            booking.dropoff_location = payload["dropoff_address"]

        # Mise à jour des coordonnées
        if "pickup_lat" in payload and "pickup_lon" in payload:
            booking.pickup_lat = float(payload["pickup_lat"])
            booking.pickup_lon = float(payload["pickup_lon"])
        if "dropoff_lat" in payload and "dropoff_lon" in payload:
            booking.dropoff_lat = float(payload["dropoff_lat"])
            booking.dropoff_lon = float(payload["dropoff_lon"])

        # Mise à jour de l'horaire
        if "scheduled_time" in payload:
            try:
                booking.scheduled_time = parse_local_naive(payload["scheduled_time"])
            except Exception as exc:
                company_mobile_dispatch_ns.abort(
                    400, f"Format scheduled_time invalide: {exc}"
                )
                raise AssertionError("Invalid scheduled_time") from exc

        # Mise à jour des notes
        if "notes" in payload:
            booking.notes_medical = payload["notes"].strip() or None

        # Mise à jour de la priorité
        if "priority" in payload:
            booking.is_urgent = payload["priority"] == "HIGH"

        # Mise à jour du chauffeur (null pour désassigner)
        if "driver_id" in payload:
            driver_id_raw = payload["driver_id"]
            if driver_id_raw is None:
                booking.driver_id = None
                if booking.status == BookingStatus.ASSIGNED:
                    booking.status = BookingStatus.ACCEPTED
            else:
                try:
                    driver_id = int(driver_id_raw)
                    # Vérifier que le chauffeur appartient à l'entreprise
                    from repositories.driver_repository import DriverRepository

                    driver_repo = DriverRepository()
                    driver = driver_repo.find_model_by_id_and_company(
                        driver_id=driver_id, company_id=company_id
                    )
                    if not driver:
                        company_mobile_dispatch_ns.abort(404, "Chauffeur introuvable")
                        raise AssertionError("Driver not found") from None
                    booking.driver_id = driver_id
                    if booking.status == BookingStatus.ACCEPTED:
                        booking.status = BookingStatus.ASSIGNED
                except (TypeError, ValueError) as exc:
                    company_mobile_dispatch_ns.abort(
                        400, "driver_id doit être un entier"
                    )
                    raise AssertionError("Invalid driver_id") from exc

        # Mise à jour du statut
        if "status" in payload:
            status_str = payload["status"].upper()
            try:
                booking.status = BookingStatus[status_str]
            except KeyError:
                company_mobile_dispatch_ns.abort(400, f"Statut invalide: {status_str}")
                raise AssertionError("Invalid status") from None

        try:
            db.session.add(booking)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.exception("[MobileUpdateRide] Échec mise à jour: %s", exc)
            company_mobile_dispatch_ns.abort(
                500, "Impossible de mettre à jour la course"
            )
            raise AssertionError("Update failed") from exc

        # Journaliser l'action
        tools = AgentTools(company_id)
        _log_mobile_action(
            tools,
            "mobile_update_ride",
            payload={
                "booking_id": booking_id,
                "updates": list(payload.keys()),
                "source": "mobile_enterprise",
            },
            reasoning=f"Mise à jour course mobile {booking_id}",
        )

        summary = _build_ride_summary(booking, current_company_id=company_id)
        return {"summary": summary}, 200


@company_mobile_dispatch_ns.route("/v1/rides/<string:ride_id>/cancel")
class MobileCancelRide(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("100 per hour")
    def post(self, ride_id: str):
        """Annule une course."""
        _, company_id = _get_company_context()

        try:
            booking_id = int(ride_id)
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(400, "ride_id invalide (entier attendu)")
            raise AssertionError("Invalid ride_id") from exc

        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        booking = booking_repo.find_model_by_id_and_company(
            booking_id=booking_id, company_id=company_id
        )
        if not booking:
            company_mobile_dispatch_ns.abort(404, "Course introuvable")
            raise AssertionError("Booking not found") from None

        # Vérifier que la course peut être annulée
        if booking.status in (
            BookingStatus.COMPLETED,
            BookingStatus.RETURN_COMPLETED,
            BookingStatus.CANCELED,
        ):
            company_mobile_dispatch_ns.abort(
                400, "La course ne peut pas être annulée (déjà terminée ou annulée)"
            )
            raise AssertionError("Booking cannot be cancelled") from None

        payload = request.get_json(silent=True) or {}
        reason_code = payload.get("reason_code", "OPERATOR_CANCELLED")
        note = payload.get("note", "").strip()

        # Annuler la course
        booking.status = BookingStatus.CANCELED
        if note:
            existing_notes = booking.notes_medical or ""
            booking.notes_medical = (
                f"{existing_notes}\n[Annulé: {note}]".strip()
                if bool(existing_notes)
                else f"[Annulé: {note}]"
            )

        # Désassigner le chauffeur si assigné
        if booking.driver_id:
            booking.driver_id = None

        try:
            db.session.add(booking)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.exception("[MobileCancelRide] Échec annulation: %s", exc)
            company_mobile_dispatch_ns.abort(500, "Impossible d'annuler la course")
            raise AssertionError("Cancel failed") from exc

        # Journaliser l'action
        tools = AgentTools(company_id)
        _log_mobile_action(
            tools,
            "mobile_cancel_ride",
            payload={
                "booking_id": booking_id,
                "reason_code": reason_code,
                "source": "mobile_enterprise",
            },
            reasoning=f"Annulation course mobile {booking_id} ({reason_code})",
        )

        return {
            "ride_id": str(booking_id),
            "status": "cancelled",
            "message": "Course annulée avec succès",
        }, 200


# =====================================================
# Endpoints pour recherche d'adresses et clients
# =====================================================


@company_mobile_dispatch_ns.route("/v1/addresses/search")
class MobileSearchAddresses(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("200 per hour")
    def get(self):
        """Recherche d'adresses avec autocomplétion."""
        MIN_QUERY_LENGTH = 2
        q = (request.args.get("q") or "").strip()
        if len(q) < MIN_QUERY_LENGTH:
            return [], 200

        # Limite de résultats
        try:
            limit = max(1, min(int(request.args.get("limit", 8)), 12))
        except (TypeError, ValueError):
            limit = 8

        results: List[Dict[str, Any]] = []

        # Utiliser l'endpoint geocode existant
        try:
            from routes.geocode import (
                GENEVA_CENTER,
                match_alias,
                normalize_google_places,
                normalize_photon,
                photon_query,
            )
            from services.geolocation.google_places import (
                GooglePlacesError,
                autocomplete_address,
            )

            # 1) Alias rapides
            alias = match_alias(q)
            if alias:
                results.append(
                    {
                        "label": alias["address"],
                        "address": alias["address"],
                        "lat": alias["lat"],
                        "lon": alias["lon"],
                        "category": alias.get("category"),
                    }
                )

            # 2) Google Places (si activé) - avec normalisation enrichie
            if len(results) < limit:
                try:
                    google_results = autocomplete_address(q, limit=limit - len(results))
                    if google_results:
                        # Normaliser les résultats Google Places pour avoir le format complet
                        normalized_google = normalize_google_places(google_results)
                        for r in normalized_google:
                            results.append(
                                {
                                    "label": r.get("label", r.get("address", "")),
                                    "address": r.get("address", r.get("label", "")),
                                    "lat": r.get("lat"),
                                    "lon": r.get("lon"),
                                    "place_id": r.get("place_id"),
                                }
                            )
                except GooglePlacesError:
                    pass
                except Exception as e:
                    logger.warning(
                        "[MobileSearchAddresses] Erreur normalisation Google Places: %s",
                        e,
                    )

            # 3) Photon fallback
            if len(results) < limit:
                try:
                    ph = photon_query(
                        q,
                        lat=GENEVA_CENTER[0],
                        lon=GENEVA_CENTER[1],
                        limit=limit - len(results),
                        hospital_hint=False,
                    )
                    photon_results = normalize_photon(ph)
                    results.extend(photon_results)
                except Exception:
                    pass

        except Exception as e:
            logger.warning("[MobileSearchAddresses] Erreur recherche: %s", e)

        return results[:limit], 200


@company_mobile_dispatch_ns.route("/v1/clients/search")
class MobileSearchClients(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("200 per hour")
    def get(self):
        """Recherche de clients de l'entreprise."""
        _, company_id = _get_company_context()
        q = (request.args.get("q") or "").strip()

        # Limite de résultats
        try:
            limit = max(1, min(int(request.args.get("limit", 20)), 50))
        except (TypeError, ValueError):
            limit = 20

        from repositories.client_repository import ClientRepository

        client_repo = ClientRepository()
        all_clients = client_repo.find_models_by_company_with_user_and_search(
            company_id=company_id, search=q if q else None
        )
        # Limiter les résultats
        clients = all_clients[:limit]

        results = []
        for client in clients:
            user = client.user
            if not user:
                continue

            full_name = (
                f"{getattr(user, 'first_name', '')} {getattr(user, 'last_name', '')}"
            ).strip()
            if not full_name:
                full_name = getattr(user, "username", "") or "Client"

            # Récupérer l'adresse de domicile complète
            domicile_address = getattr(client, "domicile_address", None)
            domicile_zip = getattr(client, "domicile_zip", None)
            domicile_city = getattr(client, "domicile_city", None)
            domicile_lat = getattr(client, "domicile_lat", None)
            domicile_lon = getattr(client, "domicile_lon", None)

            # ✅ Construire l'adresse complète avec code postal et ville
            domicile_full_address = None
            if domicile_address:
                address_parts = [domicile_address]
                if domicile_zip:
                    address_parts.append(domicile_zip)
                if domicile_city:
                    address_parts.append(domicile_city)
                domicile_full_address = ", ".join(address_parts)

            user_phone = getattr(user, "phone", None)
            user_email = getattr(user, "email", None)

            results.append(
                {
                    "id": str(client.id),
                    "name": full_name,
                    "phone": user_phone,
                    "email": user_email,
                    "contact_phone": user_phone,
                    "contact_email": user_email,
                    "domicile_address": domicile_full_address
                    or domicile_address,  # ✅ Retourner l'adresse complète
                    "domicile_zip": domicile_zip,  # ✅ Ajouter le code postal
                    "domicile_city": domicile_city,  # ✅ Ajouter la ville
                    "domicile_lat": float(domicile_lat)
                    if domicile_lat is not None
                    else None,
                    "domicile_lon": float(domicile_lon)
                    if domicile_lon is not None
                    else None,
                }
            )

        return results, 200
