"""Applique les assignations en base et émet événements/notifications."""

import logging
from contextlib import suppress
from datetime import date
from typing import Any

from application.events.event_bus import publish_event
from domain.events.events import BookingAssignedEvent, DispatchRunCompletedEvent
from ext import db
from models import Booking, DispatchRun
from repositories.booking_repository import BookingRepository
from services.unified_dispatch.core import settings as ud_settings
from services.unified_dispatch.optimization.assignment_applier import apply_assignments
from services.unified_dispatch.validation.constraints import validate_assignments
from shared.otel_setup import get_tracer

logger = logging.getLogger(__name__)

# ✅ D1: Tracer OpenTelemetry pour traces E2E
tracer = get_tracer("assignment_applier")


def _safe_int(v: Any) -> int | None:
    """Convertit n'importe quelle valeur en int Python ou retourne None."""
    try:
        return int(v)
    except Exception:
        return None


class AssignmentApplier:
    """Applique les assignations en base et émet événements/notifications.

    ✅ Validation temporelle stricte: Si des conflits sont détectés et que
    strict_temporal_validation est activé, rollback automatique.
    """

    def apply_and_emit(
        self,
        company: Any,
        assignments: list[Any],
        dispatch_run_id: int | None,
    ) -> None:
        """Applique les assignations en base et émet événements/notifications.

        Args:
            company: Objet Company
            assignments: Liste des assignations à appliquer
            dispatch_run_id: ID du DispatchRun optionnel
        """
        if not assignments:
            return

        # Session propre avant les writes
        with suppress(Exception):
            db.session.rollback()

        # ✅ Validation temporelle stricte avant application
        self._validate_assignments(company, assignments)

        # ✅ D1: Span persist
        with tracer.start_as_current_span("persist") as persist_span:
            persist_span.set_attribute("assignments_count", len(assignments))

            # 1) Apply en DB
            applied_count = self._apply_to_db(company, assignments, dispatch_run_id)
            persist_span.set_attribute("applied_count", applied_count)

            # 2) Notifications par booking
            self._notify_bookings(assignments)

        # 3) Notification globale de fin de run
        self._notify_dispatch_completion(company, dispatch_run_id, applied_count)

    def _validate_assignments(self, company: Any, assignments: list[Any]) -> None:
        """Valide les assignations avant application.

        Args:
            company: Objet Company
            assignments: Liste des assignations à valider

        Raises:
            ValueError: Si la validation échoue
        """
        try:
            company_settings = ud_settings.for_company(company)
            strict_validation = getattr(
                company_settings.features,
                "enable_strict_temporal_conflict_check",
                True,  # Par défaut activé
            )

            if not strict_validation:
                return

            # Convertir assignments en format dict pour validation
            assignments_dict = []
            for a in assignments:
                if isinstance(a, dict):
                    assignments_dict.append(a)
                else:
                    # Convertir objet en dict
                    assignment_dict = {
                        "driver_id": getattr(a, "driver_id", None),
                        "booking_id": getattr(a, "booking_id", None),
                        "scheduled_time": getattr(a, "scheduled_time", None),
                        "estimated_duration_minutes": getattr(
                            a, "estimated_duration_minutes", None
                        ),
                    }
                    assignments_dict.append(assignment_dict)

            validation_result = validate_assignments(assignments_dict, strict=True)

            if not validation_result["valid"]:
                # ✅ Rollback automatique si conflits détectés
                logger.error(
                    (
                        "[AssignmentApplier] ❌ Validation temporelle stricte échouée: "
                        "%d erreurs critiques détectées. Rollback automatique."
                    ),
                    len(validation_result["errors"]),
                )
                for error in validation_result["errors"]:
                    logger.error("[AssignmentApplier]   %s", error)

                # ✅ FIX RC2: Extraire les booking_ids affectés avant rollback
                affected_booking_ids = []
                for assignment in assignments:
                    booking_id = getattr(assignment, "booking_id", None) or (
                        assignment.get("booking_id")
                        if isinstance(assignment, dict)
                        else None
                    )
                    if booking_id:
                        affected_booking_ids.append(booking_id)

                # Rollback de la transaction
                db.session.rollback()
                # ✅ FIX RC2: Expirer tous les objets après rollback pour forcer le rechargement
                db.session.expire_all()

                # ✅ FIX RC2: Recharger les bookings depuis la DB
                if affected_booking_ids:
                    reloaded_bookings = (
                        db.session.query(Booking)
                        .filter(Booking.id.in_(affected_booking_ids))
                        .all()
                    )
                    for booking in reloaded_bookings:
                        db.session.refresh(booking)

                # Lever une exception pour arrêter le dispatch
                msg = (
                    "Validation temporelle stricte échouée: "
                    f"{len(validation_result['errors'])} conflits détectés. "
                    "Assignations non appliquées. Erreurs: "
                    f"{validation_result['errors'][:3]}"
                )
                raise ValueError(msg)

            if validation_result.get("warnings"):
                # Avertissements seulement (pas de rollback)
                logger.warning(
                    (
                        "[AssignmentApplier] ⚠️ Validation temporelle: %d avertissements "
                        "(non bloquants)"
                    ),
                    len(validation_result["warnings"]),
                )
                for warning in validation_result["warnings"][
                    :5
                ]:  # Limiter à 5 warnings
                    logger.warning("[AssignmentApplier]   %s", warning)
        except ImportError:
            # Module validation non disponible, continuer sans validation
            logger.warning(
                "[AssignmentApplier] Module validation non disponible, skip validation temporelle"
            )
        except ValueError:
            # Ré-élever l'exception de validation
            raise
        except Exception as e:
            # Erreur lors de la validation, logger mais continuer (mode défensif)
            logger.exception(
                "[AssignmentApplier] Erreur lors de la validation temporelle: %s", e
            )

    def _apply_to_db(
        self, company: Any, assignments: list[Any], dispatch_run_id: int | None
    ) -> int:
        """Applique les assignations en base de données.

        Args:
            company: Objet Company
            assignments: Liste des assignations à appliquer
            dispatch_run_id: ID du DispatchRun optionnel

        Returns:
            Nombre d'assignations appliquées
        """
        try:
            logger.info(
                "[AssignmentApplier] Applying assignments with dispatch_run_id=%s",
                dispatch_run_id,
            )
            company_id_int = _safe_int(getattr(company, "id", None)) or 0
            result = apply_assignments(
                company_id_int,
                assignments,
                dispatch_run_id=dispatch_run_id,
                return_pairs=True,
            )
            db.session.commit()
            applied_count = len(result.get("applied", []))
            logger.info(
                "[AssignmentApplier] Applied %d assignments with dispatch_run_id=%s",
                applied_count,
                dispatch_run_id,
            )
            return applied_count
        except Exception:
            logger.exception("[AssignmentApplier] DB apply failed")
            with suppress(Exception):
                db.session.rollback()
                # ✅ FIX: Expirer tous les objets après rollback pour forcer le rechargement
                db.session.expire_all()
            raise

    def _notify_bookings(self, assignments: list[Any]) -> None:
        """Émet les notifications pour chaque booking assigné.

        Args:
            assignments: Liste des assignations
        """
        # ✅ D1: Span ws_emit
        with tracer.start_as_current_span("ws_emit") as ws_span:
            applied_count = 0
            booking_repo = BookingRepository()
            assignments_by_booking_id = {
                int(booking_id): assignment
                for assignment in assignments
                if (booking_id := getattr(assignment, "booking_id", None)) is not None
            }
            # Charger toutes les courses en une seule requête évite un SELECT par
            # assignation lors d'un dispatch volumineux.
            bookings_by_id = {
                booking.id: booking
                for booking in booking_repo.find_by_ids(list(assignments_by_booking_id))
            }
            for booking_id, assignment in assignments_by_booking_id.items():
                with suppress(Exception):
                    booking_dto = bookings_by_id.get(booking_id)
                    if booking_dto:
                        publish_event(
                            BookingAssignedEvent(
                                booking_id=int(booking_dto.id),
                                company_id=(
                                    int(booking_dto.company_id)
                                    if booking_dto.company_id is not None
                                    else None
                                ),
                                driver_id=_safe_int(
                                    getattr(assignment, "driver_id", None)
                                ),
                            )
                        )
                        applied_count += 1
            ws_span.set_attribute("events_count", applied_count)

    def _notify_dispatch_completion(
        self, company: Any, dispatch_run_id: int | None, applied_count: int
    ) -> None:
        """Émet la notification globale de fin de run.

        Args:
            company: Objet Company
            dispatch_run_id: ID du DispatchRun optionnel
            applied_count: Nombre d'assignations appliquées
        """
        try:
            if not dispatch_run_id:
                return

            # Assainir la session avant un SELECT (évite InFailedSqlTransaction)
            with suppress(Exception):
                db.session.rollback()

            # Charger le DispatchRun proprement
            dr = None
            try:
                dr = db.session.get(DispatchRun, int(dispatch_run_id))
            except Exception as e:
                logger.warning(
                    "[AssignmentApplier] Failed to load DispatchRun %s: %s",
                    dispatch_run_id,
                    e,
                )

            date_str: str | None = None
            if dr is not None:
                dr_day = getattr(dr, "day", None)
                # ✅ évite le test booléen sur une Column; vérifie le type valeur
                if isinstance(dr_day, date):
                    date_str = dr_day.isoformat()

            # Notification défensive : ne doit jamais faire échouer le dispatch
            try:
                publish_event(
                    DispatchRunCompletedEvent(
                        company_id=_safe_int(getattr(company, "id", None)) or 0,
                        dispatch_run_id=int(dispatch_run_id),
                        assignments_count=applied_count,
                        date_str=date_str,
                    )
                )
            except Exception:
                logger.exception(
                    (
                        "[AssignmentApplier] Erreur notification dispatch_run_completed "
                        "(dispatch_run_id=%s) - continuation"
                    ),
                    dispatch_run_id,
                )
                # Ne pas relancer l'exception : les notifications ne doivent pas bloquer le dispatch
            logger.info(
                (
                    "[AssignmentApplier] Notified dispatch completion: company_id=%s, "
                    "dispatch_run_id=%s, assignments=%s, date=%s"
                ),
                getattr(company, "id", None),
                dispatch_run_id,
                applied_count,
                date_str,
            )
        except Exception:
            with suppress(Exception):
                logger.error("[AssignmentApplier] Notification/socket error")
