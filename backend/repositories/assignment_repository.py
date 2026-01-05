"""Repository pour l'accès aux données Assignment avec conversion DTO.

Ce repository découple les services de l'implémentation SQLAlchemy
en retournant des DTOs au lieu de modèles SQLAlchemy directs.
"""

from typing import Any, cast

from domain.assignment_dto import AssignmentDTO
from models import Assignment

logger = __import__("logging").getLogger(__name__)


class AssignmentRepository:
    """Repository pour l'accès aux données Assignment avec conversion DTO."""

    def _to_dto(self, assignment: Assignment) -> AssignmentDTO:
        """Convertit un modèle Assignment SQLAlchemy en DTO.

        Args:
            assignment: Modèle Assignment SQLAlchemy

        Returns:
            AssignmentDTO correspondant
        """
        return AssignmentDTO(
            id=assignment.id,
            dispatch_run_id=assignment.dispatch_run_id,
            booking_id=assignment.booking_id,  # type: ignore[reportGeneralTypeIssues]
            driver_id=assignment.driver_id,  # type: ignore[reportGeneralTypeIssues]
            status=assignment.status,
            planned_pickup_at=assignment.planned_pickup_at,
            planned_dropoff_at=assignment.planned_dropoff_at,
            actual_pickup_at=assignment.actual_pickup_at,
            actual_dropoff_at=assignment.actual_dropoff_at,
            eta_pickup_at=assignment.eta_pickup_at,
            eta_dropoff_at=assignment.eta_dropoff_at,
            delay_seconds=assignment.delay_seconds,
            decision_explanation=(
                dict(assignment.decision_explanation)
                if assignment.decision_explanation
                else None
            ),
            created_at=assignment.created_at,  # type: ignore[reportGeneralTypeIssues]
            updated_at=assignment.updated_at,  # type: ignore[reportGeneralTypeIssues]
        )

    def find_by_id(self, assignment_id: int) -> AssignmentDTO | None:
        """Trouve un assignment par son ID.

        Args:
            assignment_id: ID de l'assignment

        Returns:
            AssignmentDTO ou None si non trouvé
        """
        assignment = Assignment.query.get(assignment_id)
        if assignment is None:
            return None
        return self._to_dto(assignment)

    def find_by_booking_id(self, booking_id: int) -> list[AssignmentDTO]:
        """Trouve les assignments d'un booking.

        Args:
            booking_id: ID du booking

        Returns:
            Liste de AssignmentDTO
        """
        assignments = Assignment.query.filter_by(booking_id=booking_id).all()
        return [self._to_dto(a) for a in assignments]

    def find_by_driver_id(self, driver_id: int) -> list[AssignmentDTO]:
        """Trouve les assignments d'un driver.

        Args:
            driver_id: ID du driver

        Returns:
            Liste de AssignmentDTO
        """
        assignments = Assignment.query.filter_by(driver_id=driver_id).all()
        return [self._to_dto(a) for a in assignments]

    def find_by_dispatch_run_id(self, dispatch_run_id: int) -> list[AssignmentDTO]:
        """Trouve les assignments d'un dispatch run.

        Args:
            dispatch_run_id: ID du dispatch run

        Returns:
            Liste de AssignmentDTO
        """
        assignments = Assignment.query.filter_by(dispatch_run_id=dispatch_run_id).all()
        return [self._to_dto(a) for a in assignments]

    def find_by_booking_ids(self, booking_ids: list[int]) -> list[AssignmentDTO]:
        """Trouve les assignments pour plusieurs bookings.

        Args:
            booking_ids: Liste d'IDs de bookings

        Returns:
            Liste de AssignmentDTO
        """
        if not booking_ids:
            return []
        assignments = Assignment.query.filter(
            Assignment.booking_id.in_(booking_ids)
        ).all()
        return [self._to_dto(a) for a in assignments]

    def find_by_ids(self, assignment_ids: list[int]) -> list[AssignmentDTO]:
        """Trouve les assignments par leurs IDs.

        Args:
            assignment_ids: Liste d'IDs d'assignments

        Returns:
            Liste de AssignmentDTO correspondants
        """
        if not assignment_ids:
            return []
        assignments = Assignment.query.filter(Assignment.id.in_(assignment_ids)).all()
        return [self._to_dto(a) for a in assignments]

    def count_by_statuses(self, statuses: list[Any]) -> int:
        """Compte les assignments avec les statuts spécifiés.

        Args:
            statuses: Liste de statuts à filtrer

        Returns:
            Nombre d'assignments correspondants
        """
        return Assignment.query.filter(Assignment.status.in_(statuses)).count()

    def find_model_by_booking_id(self, booking_id: int) -> Assignment | None:
        """Trouve un assignment par booking_id (retourne le modèle SQLAlchemy).

        Args:
            booking_id: ID du booking

        Returns:
            Assignment ou None si non trouvé
        """
        return Assignment.query.filter_by(booking_id=booking_id).first()

    def delete_by_booking_id(self, booking_id: int) -> int:
        """Supprime tous les assignments associés à un booking (opération bulk).

        Args:
            booking_id: ID du booking

        Returns:
            Nombre d'assignments supprimés
        """
        return Assignment.query.filter_by(booking_id=booking_id).delete()

    def find_model_by_id_with_company_check(
        self, assignment_id: int, company_id: int
    ) -> Assignment | None:
        """Trouve un assignment par son ID avec vérification de company (retourne le modèle SQLAlchemy).

        Args:
            assignment_id: ID de l'assignment
            company_id: ID de l'entreprise

        Returns:
            Assignment ou None si non trouvé
        """
        from models import Booking

        return (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(Assignment.id == assignment_id, Booking.company_id == company_id)
            .first()
        )

    def find_model_by_id_with_booking_eager_loading(
        self, assignment_id: int, company_id: int
    ) -> Assignment | None:
        """Trouve un assignment par son ID avec eager loading de booking (retourne le modèle SQLAlchemy).

        Args:
            assignment_id: ID de l'assignment
            company_id: ID de l'entreprise

        Returns:
            Assignment ou None si non trouvé (avec booking chargé)
        """
        from sqlalchemy.orm import joinedload

        from models import Booking

        return (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .options(joinedload(Assignment.booking))
            .filter(Assignment.id == assignment_id, Booking.company_id == company_id)
            .first()
        )

    def find_models_with_time_range_and_eager_loading(
        self,
        company_id: int,
        start_datetime: Any,
        end_datetime: Any,
        statuses: list[Any] | None = None,
    ) -> list[Assignment]:
        """Trouve les assignments d'une entreprise dans une plage temporelle avec eager loading.

        Args:
            company_id: ID de l'entreprise
            start_datetime: Date/heure de début
            end_datetime: Date/heure de fin
            statuses: Liste de statuts à filtrer (optionnel)

        Returns:
            Liste de Assignment avec booking chargé
        """
        from sqlalchemy.orm import joinedload

        from models import Booking

        query = (
            Assignment.query.join(Booking)
            .options(joinedload(Assignment.booking))
            .filter(
                Assignment.company_id == company_id,
                Booking.scheduled_time >= start_datetime,
                Booking.scheduled_time <= end_datetime,
            )
        )
        if statuses:
            query = query.filter(Assignment.status.in_(statuses))
        return query.all()

    def find_models_by_booking_ids_with_eager_loading(
        self, booking_ids: list[int]
    ) -> list[Assignment]:
        """Trouve les assignments par booking_ids avec eager loading.

        Args:
            booking_ids: Liste d'IDs de bookings

        Returns:
            Liste de Assignment avec booking et driver+user chargés
        """
        from sqlalchemy.orm import joinedload

        from models import Driver

        if not booking_ids:
            return []
        return (
            Assignment.query.filter(Assignment.booking_id.in_(booking_ids))
            .options(
                joinedload(Assignment.booking),
                joinedload(Assignment.driver).joinedload(Driver.user),
            )
            .all()
        )

    def find_models_by_dispatch_run_with_eager_loading(
        self, dispatch_run_id: int, company_id: int
    ) -> list[Assignment]:
        """Trouve les assignments d'un dispatch run avec eager loading.

        Args:
            dispatch_run_id: ID du dispatch run
            company_id: ID de l'entreprise

        Returns:
            Liste de Assignment avec booking chargé
        """
        from sqlalchemy.orm import joinedload

        from models import Booking

        return (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .options(joinedload(Assignment.booking))
            .filter(
                Assignment.dispatch_run_id == dispatch_run_id,
                Booking.company_id == company_id,
            )
            .all()
        )

    def find_models_by_dispatch_run_with_full_eager_loading(
        self, dispatch_run_id: int, company_id: int
    ) -> list[Assignment]:
        """Trouve les assignments d'un dispatch run avec eager loading complet.

        Args:
            dispatch_run_id: ID du dispatch run
            company_id: ID de l'entreprise

        Returns:
            Liste de Assignment avec booking et driver chargés
        """
        from sqlalchemy.orm import joinedload

        from models import Booking

        return (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .options(
                joinedload(Assignment.booking),
                joinedload(Assignment.driver),
            )
            .filter(
                Assignment.dispatch_run_id == dispatch_run_id,
                Booking.company_id == company_id,
            )
            .all()
        )

    def find_models_by_driver_and_time_range(
        self,
        driver_id: int,
        company_id: int,
        current_date_start: Any,
        current_date_end: Any,
    ) -> list[Assignment]:
        """Trouve les assignments d'un driver dans une plage temporelle.

        Args:
            driver_id: ID du driver
            company_id: ID de l'entreprise
            current_date_start: Date/heure de début
            current_date_end: Date/heure de fin

        Returns:
            Liste de Assignment avec booking chargé
        """
        from sqlalchemy.orm import joinedload

        from models import Booking

        time_expr = (
            Booking.scheduled_time
        )  # Utiliser scheduled_time comme expression temporelle
        return (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .options(joinedload(Assignment.booking))
            .filter(
                Assignment.driver_id == driver_id,
                Booking.company_id == company_id,
                time_expr >= current_date_start,
                time_expr < current_date_end,
            )
            .order_by(Booking.scheduled_time.desc())
            .all()
        )

    def find_models_in_time_window(
        self, company_id: int, d0: Any, d1: Any
    ) -> list[Assignment]:
        """Trouve les assignments dans une fenêtre temporelle.

        Args:
            company_id: ID de l'entreprise
            d0: Date/heure de début
            d1: Date/heure de fin

        Returns:
            Liste de Assignment
        """
        from models import Booking

        time_expr = Booking.scheduled_time
        return (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(
                Booking.company_id == company_id,
                time_expr >= d0,
                time_expr < d1,
            )
            .all()
        )

    def find_models_by_company_with_time_range(
        self, company_id: int, d0: Any, d1: Any
    ) -> list[Assignment]:
        """Trouve les assignments d'une entreprise dans une plage temporelle.

        Args:
            company_id: ID de l'entreprise
            d0: Date/heure de début
            d1: Date/heure de fin

        Returns:
            Liste de Assignment avec booking chargé
        """
        from sqlalchemy.orm import joinedload

        from models import Booking

        return (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .options(joinedload(Assignment.booking))
            .filter(
                Booking.company_id == company_id,
                Booking.scheduled_time >= d0,
                Booking.scheduled_time < d1,
            )
            .all()
        )

    def find_models_by_company_query(self, company_id: int):
        """Retourne une query Assignment filtrée par company via Booking (pour compatibilité).

        Args:
            company_id: ID de l'entreprise

        Returns:
            Query SQLAlchemy filtrée
        """
        from models import Booking

        return Assignment.query.join(Booking).filter(Booking.company_id == company_id)

    def find_models_by_company_with_time_range_and_excluded_statuses(
        self,
        company_id: int,
        start_datetime: Any,
        end_datetime: Any,
        excluded_statuses: list[Any],
    ) -> list[Assignment]:
        """Trouve les assignments d'une entreprise dans une plage temporelle avec statuts exclus.

        Args:
            company_id: ID de l'entreprise
            start_datetime: Date/heure de début
            end_datetime: Date/heure de fin
            excluded_statuses: Liste de statuts de booking à exclure

        Returns:
            Liste de Assignment
        """
        from models import Booking

        return (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(
                Booking.company_id == company_id,
                Booking.scheduled_time >= start_datetime,
                Booking.scheduled_time < end_datetime,
            )
            .filter(cast("Any", Booking.status).notin_(excluded_statuses))
            .all()
        )

    def find_models_by_company_with_date_filter_query(
        self,
        company_id: int,
        start_datetime: Any | None = None,
        end_datetime: Any | None = None,
    ):
        """Retourne une query Assignment filtrée par company avec filtres de date optionnels.

        Args:
            company_id: ID de l'entreprise
            start_datetime: Date/heure de début (optionnel)
            end_datetime: Date/heure de fin (optionnel)

        Returns:
            Query SQLAlchemy filtrée
        """
        from models import Booking

        query = Assignment.query.join(Booking).filter(Booking.company_id == company_id)
        if start_datetime is not None:
            query = query.filter(Booking.scheduled_time >= start_datetime)
        if end_datetime is not None:
            query = query.filter(Booking.scheduled_time < end_datetime)
        return query

    def find_previous_assignment_for_driver_before_booking(
        self,
        driver_id: int,
        company_id: int,
        current_date_start: Any,
        current_date_end: Any,
        current_scheduled_time: Any,
        excluded_statuses: list[Any],
    ) -> Assignment | None:
        """Trouve l'assignation précédente d'un driver avant une booking spécifique.

        Args:
            driver_id: ID du driver
            company_id: ID de l'entreprise
            current_date_start: Début de la journée
            current_date_end: Fin de la journée
            current_scheduled_time: Heure de la booking courante
            excluded_statuses: Liste de statuts de booking à exclure

        Returns:
            Assignment précédent avec booking chargé, ou None si aucun
        """
        from sqlalchemy.orm import joinedload

        from models import Booking
        from shared.time_utils import (
            _booking_time_expr,
        )

        time_expr = _booking_time_expr()
        previous_assignments = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .options(joinedload(Assignment.booking))
            .filter(
                Assignment.driver_id == driver_id,
                Booking.company_id == company_id,
                time_expr >= current_date_start,
                time_expr < current_date_end,
                time_expr < current_scheduled_time,
                cast("Any", Booking.status).notin_(excluded_statuses),
            )
            .order_by(Booking.scheduled_time.desc())
            .limit(1)
            .all()
        )

        return previous_assignments[0] if previous_assignments else None

    def find_models_by_company_with_active_statuses_eager_loading(
        self, company_id: int, statuses: list[Any]
    ) -> list[Assignment]:
        """Trouve les assignments actifs d'une entreprise avec eager loading.

        Args:
            company_id: ID de l'entreprise
            statuses: Liste de statuts d'assignment actifs

        Returns:
            Liste de Assignment avec booking chargé
        """
        from sqlalchemy.orm import joinedload

        from models import Booking

        return (
            Assignment.query.join(Booking)
            .options(joinedload(Assignment.booking))
            .filter(
                Booking.company_id == company_id,
                Assignment.status.in_(statuses),
            )
            .all()
        )

    def find_models_by_company_and_date_with_status_eager_loading(
        self,
        company_id: int,
        for_date: Any,
        statuses: list[Any],
    ) -> list[Assignment]:
        """Trouve les assignments d'une entreprise pour une date avec filtres de statut.

        Args:
            company_id: ID de l'entreprise
            for_date: Date du jour
            statuses: Liste de statuts d'assignment à filtrer

        Returns:
            Liste de Assignment avec booking, driver et user chargés
        """
        from datetime import datetime

        from sqlalchemy.orm import joinedload

        from models import Booking, Driver

        return (
            Assignment.query.options(
                joinedload(Assignment.booking),
                joinedload(Assignment.driver).joinedload(Driver.user),
            )
            .join(Booking)
            .filter(
                Booking.company_id == company_id,
                Booking.scheduled_time
                >= datetime.combine(for_date, datetime.min.time()),
                Booking.scheduled_time
                < datetime.combine(for_date, datetime.max.time()),
                Assignment.status.in_(statuses),
            )
            .all()
        )

    def find_models_with_full_eager_loading_by_company(
        self, company_id: int, statuses: list[Any] | None = None
    ) -> list[Assignment]:
        """Trouve les assignments d'une entreprise avec eager loading complet.

        Args:
            company_id: ID de l'entreprise
            statuses: Liste de statuts à filtrer (optionnel)

        Returns:
            Liste de Assignment avec booking, driver et user chargés
        """
        from sqlalchemy.orm import joinedload

        from models import Booking, Driver

        query = (
            Assignment.query.options(
                joinedload(Assignment.booking),
                joinedload(Assignment.driver).joinedload(Driver.user),
            )
            .join(Booking)
            .filter(Booking.company_id == company_id)
        )
        if statuses:
            query = query.filter(Assignment.status.in_(statuses))
        return query.all()

    def find_model_by_id(self, assignment_id: int) -> Assignment | None:
        """Trouve un assignment par son ID (retourne le modèle SQLAlchemy).

        Args:
            assignment_id: ID de l'assignment

        Returns:
            Assignment ou None si non trouvé
        """
        return Assignment.query.get(assignment_id)

    def find_models_by_driver_with_time_expr_and_excluded_statuses(
        self,
        driver_id: int,
        company_id: int,
        time_expr: Any,
        start_datetime: Any,
        end_datetime: Any,
        before_time: Any | None,
        excluded_statuses: list[Any],
        limit: int | None = None,
    ) -> list[Assignment]:
        """Trouve les assignments d'un driver avec expression temporelle et statuts exclus.

        Args:
            driver_id: ID du driver
            company_id: ID de l'entreprise
            time_expr: Expression SQLAlchemy pour le temps
            start_datetime: Date/heure de début
            end_datetime: Date/heure de fin
            before_time: Date/heure avant laquelle filtrer (optionnel)
            excluded_statuses: Liste de statuts de booking à exclure
            limit: Nombre maximum de résultats (optionnel)

        Returns:
            Liste de Assignment avec booking chargé, triés par scheduled_time descendant
        """
        from typing import cast

        from sqlalchemy.orm import joinedload

        from models import Booking

        query = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .options(joinedload(Assignment.booking))
            .filter(
                Assignment.driver_id == driver_id,
                Booking.company_id == company_id,
                time_expr >= start_datetime,
                time_expr < end_datetime,
                cast("Any", Booking.status).notin_(excluded_statuses),
            )
        )

        if before_time:
            query = query.filter(time_expr < before_time)

        query = query.order_by(Booking.scheduled_time.desc())

        if limit:
            query = query.limit(limit)

        return query.all()

    def find_models_by_company_with_time_expr_and_time_window(
        self,
        company_id: int,
        time_expr: Any,
        day_start: Any,
        day_end: Any,
        window_start: Any,
        window_end: Any,
        excluded_statuses: list[Any],
        limit: int = 50,
    ) -> list[Assignment]:
        """Trouve les assignments d'une entreprise avec fenêtre temporelle et statuts exclus.

        Args:
            company_id: ID de l'entreprise
            time_expr: Expression SQLAlchemy pour le temps
            day_start: Date/heure de début du jour
            day_end: Date/heure de fin du jour
            window_start: Date/heure de début de la fenêtre
            window_end: Date/heure de fin de la fenêtre
            excluded_statuses: Liste de statuts de booking à exclure
            limit: Nombre maximum de résultats (défaut: 50)

        Returns:
            Liste de Assignment
        """
        from typing import cast

        from models import Booking

        return (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(
                Booking.company_id == company_id,
                time_expr >= day_start,
                time_expr < day_end,
                time_expr >= window_start,
                time_expr <= window_end,
                cast("Any", Booking.status).notin_(excluded_statuses),
            )
            .limit(limit)
            .all()
        )

    def find_models_by_company_with_time_range_and_excluded_statuses_eager_loading(
        self,
        company_id: int,
        start_datetime: Any,
        end_datetime: Any,
        excluded_statuses: list[Any],
    ) -> list[Assignment]:
        """Trouve les assignments d'une entreprise avec statuts exclus et eager loading.

        Args:
            company_id: ID de l'entreprise
            start_datetime: Date/heure de début
            end_datetime: Date/heure de fin
            excluded_statuses: Liste de statuts de booking à exclure

        Returns:
            Liste de Assignment avec booking chargé
        """
        from typing import cast

        from sqlalchemy.orm import joinedload

        from models import Booking

        return (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .options(joinedload(Assignment.booking))
            .filter(
                Booking.company_id == company_id,
                Booking.scheduled_time >= start_datetime,
                Booking.scheduled_time < end_datetime,
                cast("Any", Booking.status).notin_(excluded_statuses),
            )
            .all()
        )

    def find_active_by_driver_and_time_range(
        self,
        driver_id: int,
        booking_id: int | None = None,
    ) -> list[Assignment]:
        """Trouve les assignations actives d'un chauffeur avec eager loading du booking.

        ✅ Exclut les bookings terminés (COMPLETED) ou annulés (CANCELED)
        car le chauffeur est libre dans ces cas.

        Args:
            driver_id: ID du chauffeur
            booking_id: ID du booking à exclure (optionnel, pour les modifications)

        Returns:
            Liste d'Assignment actives avec booking chargé
        """
        from sqlalchemy.orm import joinedload

        from models import AssignmentStatus, Booking, BookingStatus

        query = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .options(joinedload(Assignment.booking))
            .filter(
                Assignment.driver_id == driver_id,
                Assignment.status.in_(
                    [
                        AssignmentStatus.SCHEDULED,
                        AssignmentStatus.EN_ROUTE_PICKUP,
                        AssignmentStatus.ARRIVED_PICKUP,
                        AssignmentStatus.ONBOARD,
                        AssignmentStatus.EN_ROUTE_DROPOFF,
                    ]
                ),
                # ✅ Exclure les bookings terminés ou annulés (chauffeur libre)
                Booking.status.notin_(
                    [
                        BookingStatus.COMPLETED,
                        BookingStatus.CANCELED,
                        BookingStatus.RETURN_COMPLETED,  # Retour terminé aussi
                    ]
                ),
            )
        )

        if booking_id is not None:
            query = query.filter(Booking.id != booking_id)

        return query.all()

    def find_active_by_driver(
        self,
        driver_id: int,
        exclude_booking_id: int | None = None,
    ) -> list[Assignment]:
        """Trouve les assignations actives d'un chauffeur avec eager loading du booking.

        ✅ Exclut les bookings terminés (COMPLETED) ou annulés (CANCELED)
        car le chauffeur est libre dans ces cas.

        Args:
            driver_id: ID du chauffeur
            exclude_booking_id: ID du booking à exclure (optionnel)

        Returns:
            Liste d'Assignment actives triées par scheduled_time avec booking chargé
        """
        from sqlalchemy.orm import joinedload

        from models import AssignmentStatus, Booking, BookingStatus

        query = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .options(joinedload(Assignment.booking))
            .filter(
                Assignment.driver_id == driver_id,
                Assignment.status.in_(
                    [
                        AssignmentStatus.SCHEDULED,
                        AssignmentStatus.EN_ROUTE_PICKUP,
                        AssignmentStatus.ARRIVED_PICKUP,
                        AssignmentStatus.ONBOARD,
                        AssignmentStatus.EN_ROUTE_DROPOFF,
                    ]
                ),
                # ✅ Exclure les bookings terminés ou annulés (chauffeur libre)
                Booking.status.notin_(
                    [
                        BookingStatus.COMPLETED,
                        BookingStatus.CANCELED,
                        BookingStatus.RETURN_COMPLETED,  # Retour terminé aussi
                    ]
                ),
            )
        )

        if exclude_booking_id is not None:
            query = query.filter(Assignment.booking_id != exclude_booking_id)

        return query.order_by(Booking.scheduled_time).all()
