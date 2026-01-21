# backend/routes/dispatch/dispatch_assignments.py
"""Endpoints pour la gestion des assignations."""

# ruff: noqa: I001  # Imports organisés manuellement pour meilleure lisibilité
import logging
from collections import defaultdict
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Any, cast

from flask import request
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_restx import Resource  # pyright: ignore[reportMissingImports]

from ext import db, redis_client, role_required
from models import Assignment
from models.enums import AssignmentStatus, BookingStatus, UserRole
from repositories.assignment_repository import AssignmentRepository
from repositories.booking_repository import BookingRepository
from repositories.driver_repository import DriverRepository
from repositories.rl_suggestion_metric_repository import RLSuggestionMetricRepository
from routes.dispatch import dispatch_ns
from routes.dispatch.dispatch_helpers import (
    _booking_time_expr,
    _current_company_id,
    _get_current_company,
    _parse_date,
)
from routes.dispatch.dispatch_schemas import (
    assignment_model,
    assignment_patch_model,
    reassign_model,
)
from shared.error_handlers import APIErrorHandler
from shared.time_utils import day_local_bounds, now_local

logger = logging.getLogger(__name__)

# Initialisation des repositories
assignment_repo = AssignmentRepository()
booking_repo = BookingRepository()
driver_repo = DriverRepository()
rl_suggestion_metric_repo = RLSuggestionMetricRepository()

# Shadow mode (optionnel)
try:
    from services.shadow_mode import ShadowModeManager  # type: ignore[reportMissingImports]

    SHADOW_MODE_AVAILABLE = True
    _shadow_manager = None
except ImportError:
    SHADOW_MODE_AVAILABLE = False
    ShadowModeManager = None
    _shadow_manager = None


def get_shadow_manager():
    """Récupère l'instance du shadow manager (singleton)."""
    global _shadow_manager  # noqa: PLW0603
    if not SHADOW_MODE_AVAILABLE or ShadowModeManager is None:
        return None
    if _shadow_manager is None:
        try:
            _shadow_manager = ShadowModeManager()
        except Exception as e:
            logger.warning("Failed to initialize shadow manager: %s", e)
            return None
    return _shadow_manager


@dispatch_ns.route("/assignments/validate")
class ValidateAssignmentsResource(Resource):
    """Valide les assignations existantes pour détecter les conflits temporels."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Vérifie les conflits temporels dans les assignations existantes.

        Query params:
            date: Date au format YYYY-MM-DD (défaut: aujourd'hui)

        Returns:
            {
                "valid": bool,
                "conflicts": List[Dict],
                "summary": Dict
            }
        """
        try:
            company_id = _current_company_id()
            date_str = request.args.get("date")

            target_date = (
                datetime.strptime(date_str, "%Y-%m-%d").date()
                if date_str
                else now_local().date()
            )

            from infrastructure.dispatch.validation_adapter import validate_assignments

            # Récupérer les assignations pour la date
            start_datetime = datetime.combine(target_date, datetime.min.time()).replace(
                tzinfo=UTC
            )
            end_datetime = datetime.combine(target_date, datetime.max.time()).replace(
                tzinfo=UTC
            )

            assignments_data = []
            # ✅ P1: Eager loading pour éviter N+1 queries
            assignments = assignment_repo.find_models_with_time_range_and_eager_loading(
                company_id=company_id,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                statuses=[
                    AssignmentStatus.SCHEDULED,
                    AssignmentStatus.EN_ROUTE_PICKUP,
                    AssignmentStatus.ARRIVED_PICKUP,
                    AssignmentStatus.ONBOARD,
                    AssignmentStatus.EN_ROUTE_DROPOFF,
                ],
            )

            for assignment in assignments:
                assignments_data.append(
                    {
                        "booking_id": assignment.booking_id,
                        "driver_id": assignment.driver_id,
                        "scheduled_time": assignment.booking.scheduled_time.isoformat()
                        if assignment.booking.scheduled_time
                        else None,
                    }
                )

            # Valider
            result = validate_assignments(assignments_data, strict=False)

            # ✅ P1: Protéger accès dictionnaires pour éviter KeyError
            warnings = result.get("warnings", [])
            errors = result.get("errors", [])
            return {
                "valid": result.get("valid", False),
                "conflicts": warnings + errors,
                "summary": result.get("stats", {}),
                "date": target_date.isoformat(),
                "total_assignments": len(assignments_data),
            }, HTTPStatus.OK

        except Exception as e:
            logger.exception("Erreur validation assignations: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/assignments")
class AssignmentsListResource(Resource):
    """Liste des assignations pour un jour."""

    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.doc(params={"date": "YYYY-MM-DD"})
    @dispatch_ns.marshal_list_with(assignment_model)
    def get(self):
        """Liste des assignations pour un jour.

        Retourne toutes les assignations pour la date donnée,
        avec les relations booking et driver chargées.
        """
        try:
            date_str = request.args.get("date")
            logger.info(
                "[Dispatch] /assignments request for date=%s company_id=%s",
                date_str,
                _current_company_id(),
            )

            d = _parse_date(date_str)
            # Utiliser day_local_bounds pour obtenir les bornes locales du jour (naïves)
            # Booking.scheduled_time est naïf local, donc on ne convertit PAS en UTC
            d0local, d1local = day_local_bounds(d.strftime("%Y-%m-%d"))
            # Pas de conversion UTC - on utilise directement les bornes locales
            d0, d1 = d0local, d1local

            logger.debug("[Dispatch] /assignments date bounds: %s to %s", d0, d1)

            # 🔒 Filtre multi-colonnes temps (comme le front)
            company = _get_current_company()
            time_expr = _booking_time_expr()

            # Ids des bookings du jour (entreprise courante),
            # en excluant les statuts terminés/annulés
            excluded_statuses = [
                s
                for s in [
                    getattr(BookingStatus, "COMPLETED", None),
                    getattr(BookingStatus, "RETURN_COMPLETED", None),
                    getattr(BookingStatus, "CANCELLED", None),
                    getattr(BookingStatus, "CANCELED", None),
                ]
                if s is not None
            ]
            bookings = booking_repo.find_models_by_company_with_time_expr_and_excluded_statuses(
                company_id=company.id,
                time_expr=time_expr,
                start_datetime=d0,
                end_datetime=d1,
                excluded_statuses=excluded_statuses,
            )
            booking_ids = [b.id for b in bookings]

            logger.info(
                "[Dispatch] /assignments found %d bookings for date=%s company_id=%s",
                len(booking_ids),
                date_str,
                company.id,
            )

            # Assignations pour ces bookings avec eager loading des relations
            assignments = []
            if booking_ids:
                assignments = (
                    assignment_repo.find_models_by_booking_ids_with_eager_loading(
                        booking_ids
                    )
                )

                logger.info(
                    (
                        "[Dispatch] /assignments found %d assignments "
                        "for %d bookings date=%s"
                    ),
                    len(assignments),
                    len(booking_ids),
                    date_str,
                )
            else:
                logger.debug(
                    (
                        "[Dispatch] /assignments no bookings found "
                        "for date=%s, returning empty assignments"
                    ),
                    date_str,
                )

            # Enrichir manuellement les champs flat pour Flask-RESTX
            for a in assignments:
                if a.driver and a.driver.user:
                    user = a.driver.user
                    # Ajouter les champs flat au driver pour le marshalling
                    a.driver.username = user.username
                    a.driver.first_name = user.first_name
                    a.driver.last_name = user.last_name
                    full = f"{user.first_name or ''} {user.last_name or ''}".strip()
                    a.driver.full_name = full or user.username

            logger.debug(
                "[Dispatch] /assignments returning %d assignments for date=%s",
                len(assignments),
                date_str,
            )

            return assignments

        except Exception as e:
            logger.exception(
                "[Dispatch] /assignments error for date=%s company_id=%s: %s",
                request.args.get("date"),
                _current_company_id(),
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/assignments/<int:assignment_id>")
class AssignmentResource(Resource):
    """Détail et modification d'une assignation."""

    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.marshal_with(assignment_model)
    def get(self, assignment_id: int):
        """Détail d'une assignation."""
        try:
            company = _get_current_company()
            a_opt: Assignment | None = (
                assignment_repo.find_model_by_id_with_company_check(
                    assignment_id, company.id
                )
            )
            if a_opt is None:
                raise APIErrorHandler.not_found(message="Assignment not found")

            return a_opt

        except Exception as e:
            logger.exception(
                "Erreur récupération assignation id=%s: %s", assignment_id, e
            )
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.expect(assignment_patch_model)
    @dispatch_ns.marshal_with(assignment_model)
    def patch(self, assignment_id: int):
        """Modifie une assignation."""
        try:
            company = _get_current_company()
            a_opt = assignment_repo.find_model_by_id_with_company_check(
                assignment_id, company.id
            )
            if a_opt is None:
                raise APIErrorHandler.not_found(message="Assignment not found")

            a = a_opt

            data = request.get_json() or {}
            # ✅ Détecter réassignation pour notifier l'ancien chauffeur
            old_driver_id: int | None = None
            try:
                old_driver_id = int(getattr(a, "driver_id", None) or 0) or None
            except Exception:
                old_driver_id = None

            # ✅ P1: Protéger accès dictionnaires pour éviter KeyError
            driver_id = data.get("driver_id")
            if driver_id is not None:
                a.driver_id = driver_id
                # Garder booking.driver_id cohérent si accessible
                from contextlib import suppress

                with suppress(Exception):
                    if getattr(a, "booking", None) is not None:
                        a.booking.driver_id = driver_id
            status = data.get("status")
            if status is not None:
                a.status = status

            cast("Any", a).updated_at = datetime.now(UTC)

            db.session.add(a)
            db.session.commit()

            # ✅ Notifier ancien chauffeur + nouveau chauffeur si changement
            try:
                new_driver_id = int(getattr(a, "driver_id", None) or 0) or None
                booking = getattr(a, "booking", None)
                booking_id = int(getattr(booking, "id", 0) or 0) if booking else 0
                if (
                    old_driver_id
                    and new_driver_id
                    and old_driver_id != new_driver_id
                    and booking_id
                ):
                    from application.events.event_bus import publish_event
                    from domain.events.events import (
                        DriverBookingReassignedEvent,
                        DriverNewBookingEvent,
                    )

                    publish_event(
                        DriverBookingReassignedEvent(
                            booking_id=booking_id,
                            old_driver_id=int(old_driver_id),
                            new_driver_id=int(new_driver_id),
                            company_id=company.id,
                        )
                    )
                    publish_event(
                        DriverNewBookingEvent(
                            booking_id=booking_id,
                            driver_id=int(new_driver_id),
                            company_id=company.id,
                        )
                    )
            except Exception:
                logger.exception(
                    "[Dispatch] Failed to publish reassignment events after PATCH"
                )
            return a

        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur MAJ assignation id=%s: %s", assignment_id, e)
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/assignments/<int:assignment_id>/reassign")
class ReassignResource(Resource):
    """Réassignation d'une assignation à un nouveau chauffeur."""

    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.expect(reassign_model, validate=True)
    @dispatch_ns.marshal_with(assignment_model)
    def post(self, assignment_id: int):
        """Réassignation d'une assignation à un nouveau chauffeur."""
        try:
            data = request.get_json() or {}
            # ✅ P1: Protéger accès dictionnaires pour éviter KeyError
            new_driver_id_raw = data.get("new_driver_id")
            if new_driver_id_raw is None:
                raise APIErrorHandler.bad_request(message="new_driver_id est requis")

            # Type narrowing: après la vérification, new_driver_id_raw n'est plus None
            new_driver_id_raw_typed = cast(int | str | float, new_driver_id_raw)

            # Initialiser new_driver_id à None pour aider le type checker
            new_driver_id: int | None = None
            try:
                new_driver_id = int(new_driver_id_raw_typed)
            except (ValueError, TypeError) as e:
                raise APIErrorHandler.bad_request(
                    message=f"new_driver_id doit être un entier valide: {e}"
                ) from e

            # Type narrowing: après le try/except, new_driver_id est garanti d'être un int
            assert new_driver_id is not None, "new_driver_id should not be None here"

            company = _get_current_company()

            # ✅ P1: Eager loading pour éviter N+1 query
            a_opt = assignment_repo.find_model_by_id_with_booking_eager_loading(
                assignment_id, company.id
            )
            if a_opt is None:
                raise APIErrorHandler.not_found(message="Assignment not found")

            a = a_opt
            # ✅ P1: Utiliser la relation eager-loaded au lieu de requête séparée
            booking = a.booking
            old_driver_id: int | None = None
            try:
                old_driver_id = int(getattr(a, "driver_id", None) or 0) or None
            except Exception:
                old_driver_id = None

            # ✅ SHADOW MODE: Prédiction DQN (NON-BLOQUANTE)
            shadow_prediction = None
            if SHADOW_MODE_AVAILABLE and booking:
                try:
                    shadow_mgr = get_shadow_manager()
                    if shadow_mgr:
                        available_drivers = (
                            driver_repo.find_models_by_company_available(company.id)
                        )

                        current_assignments = defaultdict(list)
                        # ✅ P1: Eager loading pour éviter N+1 queries
                        active_assignments = assignment_repo.find_models_by_company_with_active_statuses_eager_loading(
                            company_id=company.id,
                            statuses=[
                                AssignmentStatus.SCHEDULED,
                                AssignmentStatus.EN_ROUTE_PICKUP,
                            ],
                        )
                        for assign in active_assignments:
                            current_assignments[assign.driver_id].append(
                                assign.booking_id
                            )

                        shadow_prediction = shadow_mgr.predict_driver_assignment(
                            booking=booking,
                            available_drivers=available_drivers,
                            current_assignments=dict(current_assignments),
                        )
                        logger.debug(
                            "Shadow prediction for reassign: %s", shadow_prediction
                        )
                except Exception as e:
                    logger.warning("Shadow mode error (non-critique): %s", e)

            # ✅ SYSTÈME ACTUEL: Logique INCHANGÉE
            driver_opt = driver_repo.find_model_by_id_and_company_available(
                new_driver_id, company.id
            )
            if driver_opt is None:
                raise APIErrorHandler.not_found(message="Driver not found")

            # ✅ VALIDATION : Vérifier conflit temporel AVANT assignation
            if booking and booking.scheduled_time:
                from infrastructure.dispatch.validation_adapter import (
                    check_existing_assignment_conflict,
                )

                has_conflict, conflict_msg = check_existing_assignment_conflict(
                    driver_id=new_driver_id,
                    scheduled_time=booking.scheduled_time,
                    booking_id=booking.id,
                    tolerance_minutes=30,
                )

                if has_conflict:
                    logger.warning(
                        "[Dispatch] Tentative de réassignation créerait un conflit: %s",
                        conflict_msg,
                    )
                    raise APIErrorHandler.conflict(
                        message=f"❌ Impossible d'assigner ce chauffeur : {conflict_msg}"
                    )

            cast("Any", a).driver_id = new_driver_id
            cast("Any", a).updated_at = datetime.now(UTC)

            # ✅ Garder booking.driver_id cohérent
            if booking is not None:
                from contextlib import suppress

                with suppress(Exception):
                    booking.driver_id = int(new_driver_id)

            db.session.add(a)
            db.session.commit()

            # ✅ NOTIFICATION: Envoyer notification au nouveau chauffeur (même logique que l'assignation initiale)
            if booking:
                try:
                    from application.events.event_bus import publish_event
                    from domain.events.events import (
                        DriverBookingReassignedEvent,
                        DriverNewBookingEvent,
                    )

                    # ✅ Notifier l'ancien chauffeur si réassignation
                    try:
                        if old_driver_id and old_driver_id != int(new_driver_id):
                            publish_event(
                                DriverBookingReassignedEvent(
                                    booking_id=booking.id,
                                    old_driver_id=int(old_driver_id),
                                    new_driver_id=int(new_driver_id),
                                    company_id=company.id,
                                )
                            )
                    except Exception:
                        logger.exception(
                            "[Dispatch] Failed to publish DriverBookingReassignedEvent"
                        )

                    publish_event(
                        DriverNewBookingEvent(
                            booking_id=booking.id,
                            driver_id=new_driver_id,
                            company_id=company.id,
                        )
                    )
                except Exception as e:
                    # Fallback vers notification directe si événement échoue
                    logger.warning(
                        "[Dispatch] Event publish failed during reassignment, using direct notification: %s",
                        e,
                    )
                    from shared.notifications import notify_driver_new_booking

                    notify_driver_new_booking(new_driver_id, booking)

            # ✅ MÉTRIQUES : Marquer suggestion comme appliquée
            try:
                # Trouver la métrique correspondante (la plus récente non appliquée)
                metric = rl_suggestion_metric_repo.find_by_assignment_and_driver(
                    assignment_id, new_driver_id
                )

                if metric:
                    metric.applied_at = datetime.now(UTC)

                    # Calculer gain réel (approximation basée sur ETA)
                    # Note : Le gain réel précis nécessiterait
                    # de tracker l'ETA avant/après
                    # Pour l'instant, on marque comme "appliqué"
                    # et on calculera le gain plus tard
                    metric.was_successful = True  # Assume succès (à affiner)

                    db.session.add(metric)
                    db.session.commit()
                    logger.info(
                        "[RL] Metric %s marked as applied", metric.suggestion_id
                    )
                else:
                    logger.debug(
                        "[RL] No metric found for assignment %s, driver %s",
                        assignment_id,
                        new_driver_id,
                    )
            except Exception as e:
                db.session.rollback()
                logger.warning("[RL] Failed to update metric (non-critique): %s", e)

            # ✅ CACHE REDIS : Invalider cache suggestions après réassignation
            if redis_client:
                try:
                    # ✅ P1: Booking déjà chargé via join précédent, pas besoin de requête supplémentaire
                    if booking and booking.scheduled_time:
                        for_date_cache = booking.scheduled_time.date().isoformat()

                        # Supprimer toutes les clés de cache pour cette company/date
                        pattern = f"rl_suggestions:{company.id}:{for_date_cache}:*"
                        deleted_count = 0
                        for key in redis_client.scan_iter(match=pattern):
                            redis_client.delete(key)
                            deleted_count += 1

                        logger.info(
                            (
                                "[RL] Cache invalidated: %s keys deleted "
                                "for company %s, date %s"
                            ),
                            deleted_count,
                            company.id,
                            for_date_cache,
                        )
                except Exception as e:
                    logger.warning(
                        "[RL] Cache invalidation failed (non-critique): %s", e
                    )

            return a

        except Exception as e:
            db.session.rollback()
            logger.exception(
                "Erreur réassignation assignment_id=%s: %s", assignment_id, e
            )
            return APIErrorHandler.handle_exception(e, logger)
