"""Repository pour l'accès aux données Booking avec conversion DTO.

Ce repository découple les services de l'implémentation SQLAlchemy
en retournant des DTOs au lieu de modèles SQLAlchemy directs.
"""

from datetime import datetime, timedelta
from typing import Any

from domain.booking_dto import BookingDTO
from models import Booking, BookingStatus

logger = __import__("logging").getLogger(__name__)


class BookingRepository:
    """Repository pour l'accès aux données Booking avec conversion DTO."""

    @staticmethod
    def _company_visibility_filter(company_id: int):
        """Crée un filtre SQLAlchemy pour la visibilité des courses selon les règles de transfert.

        Règles de visibilité :
        - L'entreprise voit ses propres courses (company_id == company_id)
        - L'entreprise voit les courses transférées en PENDING où elle est l'exécutante
        - L'entreprise voit les courses transférées acceptées/assignées où elle est l'exécutante

        Args:
            company_id: ID de l'entreprise

        Returns:
            Filtre SQLAlchemy pour la visibilité
        """
        from sqlalchemy import and_, or_

        return or_(
            # L'entreprise voit ses propres courses
            (Booking.company_id == company_id),
            # Courses transférées en PENDING où l'entreprise est l'exécutante
            and_(
                (Booking.executing_company_id == company_id),
                (Booking.status == BookingStatus.PENDING),
            ),
            # Courses transférées acceptées/assignées où l'entreprise est l'exécutante
            and_(
                (Booking.executing_company_id == company_id),
                (Booking.status.in_([BookingStatus.ACCEPTED, BookingStatus.ASSIGNED])),
            ),
        )

    def _to_dto(self, booking: Booking) -> BookingDTO:
        """Convertit un modèle Booking SQLAlchemy en DTO.

        Args:
            booking: Modèle Booking SQLAlchemy

        Returns:
            BookingDTO correspondant
        """
        return BookingDTO(
            id=booking.id,
            company_id=booking.company_id,  # type: ignore[reportGeneralTypeIssues]
            client_id=booking.client_id,  # type: ignore[reportGeneralTypeIssues]
            user_id=booking.user_id,  # type: ignore[reportGeneralTypeIssues]
            executing_company_id=booking.executing_company_id,
            driver_id=booking.driver_id,
            customer_name=booking.customer_name,
            pickup_location=booking.pickup_location,
            dropoff_location=booking.dropoff_location,
            booking_type=getattr(booking, "booking_type", "standard"),
            scheduled_time=booking.scheduled_time,
            boarded_at=booking.boarded_at,
            completed_at=booking.completed_at,
            status=booking.status,
            amount=booking.amount,
            pickup_lat=booking.pickup_lat,  # type: ignore[reportGeneralTypeIssues]
            pickup_lon=booking.pickup_lon,  # type: ignore[reportGeneralTypeIssues]
            dropoff_lat=booking.dropoff_lat,  # type: ignore[reportGeneralTypeIssues]
            dropoff_lon=booking.dropoff_lon,  # type: ignore[reportGeneralTypeIssues]
            distance_meters=booking.distance_meters,  # type: ignore[reportGeneralTypeIssues]
            duration_seconds=booking.duration_seconds,  # type: ignore[reportGeneralTypeIssues]
            is_round_trip=getattr(booking, "is_round_trip", False),
            is_return=getattr(booking, "is_return", False),
            is_urgent=getattr(booking, "is_urgent", False),
            time_confirmed=getattr(booking, "time_confirmed", True),
            parent_booking_id=getattr(booking, "parent_booking_id", None),
        )

    def find_by_id(self, booking_id: int) -> BookingDTO | None:
        """Trouve un booking par son ID.

        Args:
            booking_id: ID du booking

        Returns:
            BookingDTO ou None si non trouvé
        """
        booking = Booking.query.get(booking_id)
        if booking is None:
            return None
        return self._to_dto(booking)

    def find_for_dispatch(
        self, company_id: int, horizon_minutes: int
    ) -> list[BookingDTO]:
        """Trouve les bookings éligibles pour le dispatch.

        Args:
            company_id: ID de l'entreprise
            horizon_minutes: Horizon temporel en minutes

        Returns:
            Liste de BookingDTO éligibles pour le dispatch
        """
        from shared.time_utils import now_local

        now_ts = now_local()
        horizon_local = now_ts + timedelta(minutes=horizon_minutes)

        # Utiliser le filtre de visibilité pour les courses transférées
        # Pour le dispatch, on veut seulement ACCEPTED et ASSIGNED
        bookings = (
            Booking.query.filter(
                self._company_visibility_filter(company_id),
                Booking.status.in_([BookingStatus.ACCEPTED, BookingStatus.ASSIGNED]),
                Booking.scheduled_time.isnot(None),
                Booking.scheduled_time <= horizon_local,
            )
            .order_by(Booking.scheduled_time.asc())
            .all()
        )

        # Filtrer les retours avec heure à confirmer
        filtered_bookings = []
        for booking in bookings:
            if booking.scheduled_time is None:
                continue

            is_return = bool(getattr(booking, "is_return", False))
            time_confirmed = bool(getattr(booking, "time_confirmed", True))

            if is_return and not time_confirmed:
                continue

            filtered_bookings.append(booking)

        return [self._to_dto(b) for b in filtered_bookings]

    def find_for_day(self, company_id: int, day_str: str) -> list[BookingDTO]:
        """Trouve les bookings pour un jour donné.

        Args:
            company_id: ID de l'entreprise
            day_str: Date du jour au format YYYY-MM-DD

        Returns:
            Liste de BookingDTO pour ce jour
        """
        from shared.time_utils import day_local_bounds

        day_start, day_end = day_local_bounds(day_str)

        # Utiliser le filtre de visibilité pour les courses transférées
        bookings = (
            Booking.query.filter(
                self._company_visibility_filter(company_id),
                Booking.scheduled_time >= day_start,
                Booking.scheduled_time < day_end,
            )
            .order_by(Booking.scheduled_time.asc())
            .all()
        )

        return [self._to_dto(b) for b in bookings]

    def find_by_ids(self, booking_ids: list[int]) -> list[BookingDTO]:
        """Trouve plusieurs bookings par leurs IDs.

        Args:
            booking_ids: Liste d'IDs de bookings

        Returns:
            Liste de BookingDTO correspondants
        """
        if not booking_ids:
            return []

        bookings = Booking.query.filter(Booking.id.in_(booking_ids)).all()
        return [self._to_dto(b) for b in bookings]

    def find_recent_completed_by_driver(
        self,
        driver_id: int,
        cutoff_date: datetime,
        limit: int = 50,
    ) -> list[BookingDTO]:
        """Trouve les bookings récemment complétés pour un driver.

        Args:
            driver_id: ID du driver
            cutoff_date: Date de coupure (bookings après cette date)
            limit: Nombre maximum de bookings à retourner

        Returns:
            Liste de BookingDTO triés par completed_at décroissant
        """
        from sqlalchemy import and_

        bookings = (
            Booking.query.filter(
                and_(
                    Booking.driver_id == driver_id,
                    Booking.status == BookingStatus.COMPLETED,
                    Booking.completed_at >= cutoff_date,
                )
            )
            .order_by(Booking.completed_at.desc())
            .limit(limit)
            .all()
        )
        return [self._to_dto(b) for b in bookings]

    def find_model_by_id(self, booking_id: int) -> Booking | None:
        """Trouve un booking par son ID (retourne le modèle SQLAlchemy).

        Args:
            booking_id: ID du booking

        Returns:
            Booking ou None si non trouvé
        """
        return Booking.query.get(booking_id)

    def find_model_by_id_and_company(
        self, booking_id: int, company_id: int
    ) -> Booking | None:
        """Trouve un booking par son ID et company_id (retourne le modèle SQLAlchemy).

        Args:
            booking_id: ID du booking
            company_id: ID de l'entreprise

        Returns:
            Booking ou None si non trouvé
        """
        return Booking.query.filter_by(id=booking_id, company_id=company_id).first()

    def find_models_by_company_with_filters(
        self,
        company_id: int,
        day_str: str | None = None,
        status: str | None = None,
    ) -> list[Booking]:
        """Trouve les bookings d'une entreprise avec filtres optionnels (retourne les modèles SQLAlchemy).

        Règles de visibilité :
        - L'entreprise voit ses propres courses (company_id == company_id)
        - L'entreprise voit les courses transférées en PENDING où elle est l'exécutante
        - L'entreprise voit les courses transférées acceptées/assignées où elle est l'exécutante

        Args:
            company_id: ID de l'entreprise
            day_str: Date au format YYYY-MM-DD (optionnel)
            status: Statut à filtrer (optionnel)

        Returns:
            Liste de Booking
        """
        # Utiliser le filtre de visibilité pour les courses transférées
        query = Booking.query.filter(self._company_visibility_filter(company_id))

        logger.info(
            (
                "[BookingRepository] find_models_by_company_with_filters: company_id=%s, "
                "day_str=%s, status=%s"
            ),
            company_id,
            day_str,
            status,
        )

        if day_str:
            from shared.time_utils import day_local_bounds

            try:
                start_local, end_local = day_local_bounds(day_str)
                query = query.filter(
                    Booking.scheduled_time >= start_local,
                    Booking.scheduled_time < end_local,
                )
                logger.info(
                    "[BookingRepository] Date filter applied: %s to %s",
                    start_local,
                    end_local,
                )
            except ValueError:
                # Retourner une liste vide si la date est invalide
                logger.warning("[BookingRepository] Invalid date format: %s", day_str)
                return []

        if status:
            query = query.filter(Booking.status == status)
            logger.info("[BookingRepository] Status filter applied: %s", status)

        bookings = query.order_by(Booking.scheduled_time.desc()).all()

        logger.info(
            (
                "[BookingRepository] Found %s bookings for company_id=%s. "
                "Bookings details: %s"
            ),
            len(bookings),
            company_id,
            [
                {
                    "id": b.id,
                    "company_id": b.company_id,
                    "executing_company_id": b.executing_company_id,
                    "status": (
                        b.status.value if hasattr(b.status, "value") else str(b.status)
                    ),
                }
                for b in bookings
            ],
        )

        return bookings

    def find_models_by_client_id(
        self, client_id: int, limit: int | None = None
    ) -> list[Booking]:
        """Trouve les bookings d'un client (retourne les modèles SQLAlchemy).

        Args:
            client_id: ID du client
            limit: Nombre maximum de bookings à retourner (optionnel)

        Returns:
            Liste de Booking triées par scheduled_time décroissant
        """
        query = Booking.query.filter_by(client_id=client_id).order_by(
            Booking.scheduled_time.desc()
        )
        if limit:
            query = query.limit(limit)
        return query.all()

    def find_models_by_client_and_company(
        self, client_id: int, company_id: int
    ) -> list[Booking]:
        """Trouve les bookings d'un client pour une entreprise (retourne les modèles SQLAlchemy).

        Args:
            client_id: ID du client
            company_id: ID de l'entreprise

        Returns:
            Liste de Booking triées par scheduled_time décroissant
        """
        return (
            Booking.query.filter_by(client_id=client_id, company_id=company_id)
            .order_by(Booking.scheduled_time.desc())
            .all()
        )

    def find_models_by_driver_and_company(
        self,
        driver_id: int,
        company_id: int,
        statuses: list[BookingStatus] | None = None,
    ) -> list[Booking]:
        """Trouve les bookings d'un driver pour une entreprise (retourne les modèles SQLAlchemy).

        Args:
            driver_id: ID du driver
            company_id: ID de l'entreprise
            statuses: Liste de statuts à filtrer (optionnel)

        Returns:
            Liste de Booking
        """
        query = Booking.query.filter_by(driver_id=driver_id, company_id=company_id)

        if statuses:
            query = query.filter(Booking.status.in_(statuses))

        return query.all()

    def count_by_client_id(self, client_id: int) -> int:
        """Compte les bookings d'un client.

        Args:
            client_id: ID du client

        Returns:
            Nombre de bookings
        """
        return Booking.query.filter_by(client_id=client_id).count()

    def find_model_by_parent_booking_id_and_company(
        self, parent_booking_id: int, company_id: int, is_return: bool = True
    ) -> Booking | None:
        """Trouve un booking par parent_booking_id, company_id et is_return.

        Args:
            parent_booking_id: ID du booking parent
            company_id: ID de l'entreprise
            is_return: Si True, cherche un booking de retour

        Returns:
            Booking ou None si non trouvé
        """
        return Booking.query.filter_by(
            parent_booking_id=parent_booking_id,
            is_return=is_return,
            company_id=company_id,
        ).first()

    def find_models_by_company_with_driver_and_user(
        self, company_id: int, statuses: list[BookingStatus] | None = None
    ) -> list[Booking]:
        """Trouve les bookings d'une entreprise avec eager loading de driver et user.

        Args:
            company_id: ID de l'entreprise
            statuses: Liste de statuts à filtrer (optionnel)

        Returns:
            Liste de Booking avec driver et user chargés
        """
        from sqlalchemy.orm import joinedload

        from models import Driver

        # Utiliser le filtre de visibilité pour les courses transférées
        query = Booking.query.options(
            joinedload(Booking.driver).joinedload(Driver.user)
        ).filter(self._company_visibility_filter(company_id))

        if statuses:
            query = query.filter(Booking.status.in_(statuses))

        return query.all()

    def find_models_by_company_with_filters_query(
        self,
        company_id: int,
        booking_ids: list[int] | None = None,
        start_datetime: Any | None = None,
        end_datetime: Any | None = None,
    ):
        """Retourne une query Booking filtrée par company avec filtres optionnels.

        Args:
            company_id: ID de l'entreprise
            booking_ids: Liste d'IDs de bookings à filtrer (optionnel)
            start_datetime: Date/heure de début (optionnel)
            end_datetime: Date/heure de fin (optionnel)

        Returns:
            Query SQLAlchemy filtrée
        """
        # Utiliser le filtre de visibilité pour les courses transférées
        query = Booking.query.filter(self._company_visibility_filter(company_id))
        if booking_ids:
            query = query.filter(Booking.id.in_(booking_ids))
        if start_datetime is not None:
            query = query.filter(Booking.scheduled_time >= start_datetime)
        if end_datetime is not None:
            query = query.filter(Booking.scheduled_time < end_datetime)
        return query

    def find_models_by_driver_id(self, driver_id: int) -> list[Booking]:
        """Trouve les bookings d'un driver (retourne les modèles SQLAlchemy).

        Args:
            driver_id: ID du driver

        Returns:
            Liste de Booking
        """
        return Booking.query.filter_by(driver_id=driver_id).all()

    def find_model_by_id_and_driver(
        self, booking_id: int, driver_id: int
    ) -> Booking | None:
        """Trouve un booking par son ID et driver_id (retourne le modèle SQLAlchemy).

        Args:
            booking_id: ID du booking
            driver_id: ID du driver

        Returns:
            Booking ou None si non trouvé
        """
        return Booking.query.filter_by(id=booking_id, driver_id=driver_id).one_or_none()

    def find_model_by_id_and_client(
        self, booking_id: int, client_id: int
    ) -> Booking | None:
        """Trouve un booking par son ID et client_id (retourne le modèle SQLAlchemy).

        Args:
            booking_id: ID du booking
            client_id: ID du client

        Returns:
            Booking ou None si non trouvé
        """
        return Booking.query.filter_by(id=booking_id, client_id=client_id).one_or_none()

    def _coerce_booking_statuses(self, statuses: list[Any]) -> list[BookingStatus]:
        """Accepte des statuts sous forme d'enum BookingStatus OU de str (ex: 'ASSIGNED')."""
        coerced: list[BookingStatus] = []
        for s in statuses:
            if isinstance(s, BookingStatus):
                coerced.append(s)
                continue
            try:
                coerced.append(BookingStatus(str(s)))
            except Exception:
                # Ignorer les valeurs inconnues (évite de casser des appels legacy)
                continue
        return coerced

    def find_models_by_driver_with_statuses_and_time_range(
        self,
        driver_id: int,
        statuses: list[Any],
        start_time: Any,
        end_time: Any,
    ) -> list[Booking]:
        """Trouve les bookings d'un driver avec filtres de statut et plage temporelle.

        Args:
            driver_id: ID du driver
            statuses: Liste de statuts à filtrer
            start_time: Date/heure de début
            end_time: Date/heure de fin

        Returns:
            Liste de Booking triés par scheduled_time ascendant
        """
        statuses_enum = self._coerce_booking_statuses(statuses)
        if not statuses_enum:
            return []

        return (
            Booking.query.filter_by(driver_id=driver_id)
            .filter(
                Booking.scheduled_time >= start_time,
                Booking.scheduled_time < end_time,
                Booking.status.in_(statuses_enum),
            )
            .order_by(Booking.scheduled_time.asc())
            .all()
        )

    def find_models_by_driver_with_statuses(
        self, driver_id: int, statuses: list[BookingStatus]
    ) -> list[Booking]:
        """Trouve les bookings d'un driver avec filtres de statut.

        Args:
            driver_id: ID du driver
            statuses: Liste de statuts à filtrer

        Returns:
            Liste de Booking triés par scheduled_time ascendant
        """
        return (
            Booking.query.filter_by(driver_id=driver_id)
            .filter(Booking.status.in_(statuses))
            .order_by(Booking.scheduled_time.asc())
            .all()
        )

    def find_models_by_driver_with_status_clause(
        self, driver_id: int, status_clause: Any
    ) -> list[Booking]:
        """Trouve les bookings d'un driver avec une clause de statut personnalisée.

        Args:
            driver_id: ID du driver
            status_clause: Clause SQLAlchemy pour filtrer les statuts

        Returns:
            Liste de Booking triés par scheduled_time descendant
        """
        return (
            Booking.query.filter(Booking.driver_id == driver_id)
            .filter(status_clause)
            .order_by(Booking.scheduled_time.desc())
            .all()
        )

    def find_models_unbilled_by_company_and_client(
        self,
        company_id: int,
        client_id: int,
        period_year: int | None = None,
        period_month: int | None = None,
    ) -> list[Booking]:
        """Trouve les bookings terminés non facturés pour un client et une entreprise.

        Args:
            company_id: ID de l'entreprise
            client_id: ID du client
            period_year: Année de la période (optionnel)
            period_month: Mois de la période (optionnel)

        Returns:
            Liste de Booking triés par scheduled_time ascendant
        """
        # #region agent log
        import json
        from pathlib import Path

        log_path = Path(r"c:\Users\jasiq\atmr\.cursor\debug.log")
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "C",
                            "location": "booking_repository.py:find_by_company_and_client_and_period",
                            "message": "Recherche bookings pour facturation",
                            "data": {
                                "company_id": company_id,
                                "client_id": client_id,
                                "period_year": period_year,
                                "period_month": period_month,
                            },
                            "timestamp": int(
                                __import__("datetime")
                                .datetime.now(__import__("datetime").UTC)
                                .timestamp()
                                * 1000
                            ),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion

        # ✅ Pour les courses transférées :
        # - L'entreprise propriétaire (company_id) peut facturer le client
        # - L'entreprise exécutante (executing_company_id) peut facturer l'entreprise propriétaire
        # Ici, on cherche les courses où l'entreprise est propriétaire (peut facturer le client)
        query = Booking.query.filter(
            Booking.company_id == company_id,  # Entreprise propriétaire
            Booking.client_id == client_id,
            Booking.status.in_(
                [BookingStatus.COMPLETED, BookingStatus.RETURN_COMPLETED]
            ),
            Booking.invoice_line_id.is_(None),  # Pas encore facturé
        )
        # Note: Les courses transférées ont toujours company_id = entreprise propriétaire,
        # donc elles sont incluses automatiquement dans cette requête

        # #region agent log
        # Vérifier s'il y a des bookings transférés pour ce client
        try:
            transferred_count = Booking.query.filter(
                Booking.executing_company_id == company_id,
                Booking.company_id != company_id,
                Booking.client_id == client_id,
                Booking.status.in_(
                    [BookingStatus.COMPLETED, BookingStatus.RETURN_COMPLETED]
                ),
                Booking.invoice_line_id.is_(None),
            ).count()
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "D",
                            "location": "booking_repository.py:find_by_company_and_client_and_period",
                            "message": "Bookings transférés trouvés pour ce client",
                            "data": {
                                "company_id": company_id,
                                "client_id": client_id,
                                "transferred_count": transferred_count,
                                "note": "Bookings où cette entreprise est l'exécutante mais pas le propriétaire",
                            },
                            "timestamp": int(
                                __import__("datetime")
                                .datetime.now(__import__("datetime").UTC)
                                .timestamp()
                                * 1000
                            ),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion

        # Filtrer par période si fournie
        if period_year and period_month:
            PERIOD_MONTH_THRESHOLD = 12
            start_date = datetime(period_year, period_month, 1)
            if period_month == PERIOD_MONTH_THRESHOLD:
                end_date = datetime(period_year + 1, 1, 1)
            else:
                end_date = datetime(period_year, period_month + 1, 1)

            query = query.filter(
                Booking.scheduled_time >= start_date,
                Booking.scheduled_time < end_date,
            )

        return query.order_by(Booking.scheduled_time.asc()).all()

    def find_model_by_id_with_full_eager_loading(
        self, booking_id: int, company_id: int
    ) -> Booking | None:
        """Trouve un booking par son ID avec eager loading complet (retourne le modèle SQLAlchemy).

        Args:
            booking_id: ID du booking
            company_id: ID de l'entreprise

        Returns:
            Booking ou None si non trouvé (avec driver, user, assignments chargés)
        """
        from sqlalchemy.orm import selectinload

        from models import Assignment, Driver

        return (
            Booking.query.options(
                selectinload(Booking.driver).selectinload(Driver.user),
                selectinload(Booking.assignments)
                .selectinload(Assignment.driver)
                .selectinload(Driver.user),
            )
            .filter(Booking.id == booking_id, Booking.company_id == company_id)
            .first()
        )

    def find_model_by_id_with_eager_loading(self, booking_id: int) -> Booking | None:
        """Trouve un booking par son ID avec eager loading (retourne le modèle SQLAlchemy).

        Args:
            booking_id: ID du booking

        Returns:
            Booking ou None si non trouvé (avec driver, client, company chargés)
        """
        from sqlalchemy.orm import selectinload

        from models import Client, Driver

        return (
            Booking.query.filter_by(id=booking_id)
            .options(
                selectinload(Booking.driver).selectinload(Driver.user),
                selectinload(Booking.client).selectinload(Client.user),
                selectinload(Booking.company),
            )
            .first()
        )

    def find_all_with_eager_loading_query(self, status_filter: str | None = None):
        """Retourne une query Booking avec eager loading pour tous les bookings (admin).

        Args:
            status_filter: Statut à filtrer (optionnel)

        Returns:
            Query SQLAlchemy avec eager loading
        """
        from sqlalchemy.orm import selectinload

        from models import Client, Driver

        query = Booking.query.options(
            selectinload(Booking.driver).selectinload(Driver.user),
            selectinload(Booking.client).selectinload(Client.user),
            selectinload(Booking.company),
        )
        if status_filter:
            query = query.filter_by(status=status_filter)
        return query

    def find_by_client_id_with_eager_loading_query(
        self, client_id: int, status_filter: str | None = None
    ):
        """Retourne une query Booking filtrée par client_id avec eager loading.

        Args:
            client_id: ID du client
            status_filter: Statut à filtrer (optionnel)

        Returns:
            Query SQLAlchemy avec eager loading, triée par scheduled_time desc
        """
        from sqlalchemy.orm import joinedload

        from models import Client, Driver

        query = (
            Booking.query.options(
                joinedload(Booking.client).joinedload(Client.user),
                joinedload(Booking.driver).joinedload(Driver.user),
                joinedload(Booking.company),
            )
            .filter_by(client_id=client_id)
            .order_by(Booking.scheduled_time.desc())
        )
        if status_filter:
            query = query.filter_by(status=status_filter)
        return query

    def find_models_by_company_with_time_range_and_statuses(
        self,
        company_id: int,
        start_datetime: Any,
        end_datetime: Any,
        statuses: list[BookingStatus],
    ) -> list[Booking]:
        """Trouve les bookings d'une entreprise dans une plage temporelle avec statuts.

        Args:
            company_id: ID de l'entreprise
            start_datetime: Date/heure de début
            end_datetime: Date/heure de fin
            statuses: Liste de statuts à filtrer

        Returns:
            Liste de Booking
        """
        # Utiliser le filtre de visibilité pour les courses transférées
        return Booking.query.filter(
            self._company_visibility_filter(company_id),
            Booking.scheduled_time >= start_datetime,
            Booking.scheduled_time < end_datetime,
            Booking.status.in_(statuses),
        ).all()

    def find_models_by_company_with_time_range_or_none_and_statuses(
        self,
        company_id: int,
        start_datetime: Any,
        end_datetime: Any,
        statuses: list[BookingStatus] | None = None,
    ) -> list[Booking]:
        """Trouve les bookings d'une entreprise avec plage temporelle optionnelle et statuts.

        Args:
            company_id: ID de l'entreprise
            start_datetime: Date/heure de début
            end_datetime: Date/heure de fin
            statuses: Liste de statuts à filtrer (optionnel)

        Returns:
            Liste de Booking avec eager loading complet
        """
        from sqlalchemy import and_, or_
        from sqlalchemy.orm import selectinload

        from models import Assignment, Driver

        query = Booking.query.options(
            selectinload(Booking.driver).selectinload(Driver.user),
            selectinload(Booking.assignments)
            .selectinload(Assignment.driver)
            .selectinload(Driver.user),
        ).filter(
            self._company_visibility_filter(company_id),
            or_(
                Booking.scheduled_time.is_(None),
                and_(
                    Booking.scheduled_time >= start_datetime,
                    Booking.scheduled_time < end_datetime,
                ),
            ),
        )
        if statuses:
            query = query.filter(Booking.status.in_(statuses))
        return query.all()

    def find_models_by_company_with_time_range_and_statuses_query(
        self,
        company_id: int,
        start_datetime: Any,
        end_datetime: Any,
        statuses: list[BookingStatus],
    ):
        """Retourne une query Booking filtrée par company, plage temporelle et statuts.

        Args:
            company_id: ID de l'entreprise
            start_datetime: Date/heure de début
            end_datetime: Date/heure de fin
            statuses: Liste de statuts à filtrer

        Returns:
            Query SQLAlchemy filtrée
        """
        # Utiliser le filtre de visibilité pour les courses transférées
        return Booking.query.filter(
            self._company_visibility_filter(company_id),
            Booking.scheduled_time >= start_datetime,
            Booking.scheduled_time < end_datetime,
            Booking.status.in_(statuses),
        )

    def find_models_by_company_with_time_range_or_none_and_statuses_query(
        self,
        company_id: int,
        start_datetime: Any,
        end_datetime: Any,
        statuses: list[BookingStatus] | None = None,
    ):
        """Retourne une query Booking filtrée par company avec plage temporelle optionnelle et statuts.

        Args:
            company_id: ID de l'entreprise
            start_datetime: Date/heure de début
            end_datetime: Date/heure de fin
            statuses: Liste de statuts à filtrer (optionnel)

        Returns:
            Query SQLAlchemy filtrée
        """
        from sqlalchemy import and_, or_

        # Utiliser le filtre de visibilité pour les courses transférées
        query = Booking.query.filter(
            self._company_visibility_filter(company_id),
            or_(
                Booking.scheduled_time.is_(None),
                and_(
                    Booking.scheduled_time >= start_datetime,
                    Booking.scheduled_time < end_datetime,
                ),
            ),
        )
        if statuses:
            query = query.filter(Booking.status.in_(statuses))
        return query

    def find_models_by_company_with_time_expr_and_excluded_statuses(
        self,
        company_id: int,
        time_expr: Any,
        start_datetime: Any,
        end_datetime: Any,
        excluded_statuses: list[BookingStatus],
    ) -> list[Booking]:
        """Trouve les bookings d'une entreprise avec expression temporelle et statuts exclus.

        Args:
            company_id: ID de l'entreprise
            time_expr: Expression SQLAlchemy pour le temps (ex: Booking.scheduled_time)
            start_datetime: Date/heure de début
            end_datetime: Date/heure de fin
            excluded_statuses: Liste de statuts à exclure

        Returns:
            Liste de Booking avec eager loading complet
        """
        from typing import cast

        from sqlalchemy.orm import selectinload

        from models import Client, Driver

        return (
            Booking.query.options(
                selectinload(Booking.driver).selectinload(Driver.user),
                selectinload(Booking.client).selectinload(Client.user),
                selectinload(Booking.company),
            )
            .filter(
                self._company_visibility_filter(company_id),
                time_expr >= start_datetime,
                time_expr < end_datetime,
                cast("Any", Booking.status).notin_(excluded_statuses),
            )
            .all()
        )

    def count_all(self) -> int:
        """Compte tous les bookings.

        Returns:
            Nombre total de bookings
        """
        return Booking.query.count()

    def count_by_date_range(self, start_date: Any, end_date: Any) -> int:
        """Compte les bookings dans une plage de dates.

        Args:
            start_date: Date de début
            end_date: Date de fin

        Returns:
            Nombre de bookings dans la plage
        """
        return Booking.query.filter(
            Booking.created_at >= start_date, Booking.created_at <= end_date
        ).count()

    def find_recent_with_client_and_user(self, limit: int = 5) -> list[Booking]:
        """Trouve les bookings récents avec eager loading de client et user.

        Args:
            limit: Nombre maximum de bookings à retourner (défaut: 5)

        Returns:
            Liste de Booking triés par scheduled_time décroissant avec client et user chargés
        """
        from sqlalchemy.orm import joinedload

        from models import Client

        return (
            Booking.query.options(joinedload(Booking.client).joinedload(Client.user))
            .order_by(Booking.scheduled_time.desc())
            .limit(limit)
            .all()
        )

    def find_completed_after_date(self, cutoff_date: datetime) -> list[Booking]:
        """Trouve les bookings complétés après une date donnée.

        Args:
            cutoff_date: Date de coupure (bookings complétés après cette date)

        Returns:
            Liste de Booking complétés après la date de coupure
        """
        from sqlalchemy import and_

        return Booking.query.filter(
            and_(
                Booking.status == BookingStatus.COMPLETED,
                Booking.completed_at >= cutoff_date,
            )
        ).all()

    def find_by_company_and_client_and_period(
        self,
        company_id: int,
        client_id: int,
        start_date: datetime,
        end_date: datetime,
        statuses: list[str] | None = None,
    ) -> list[Booking]:
        """Trouve les bookings d'un client pour une période donnée.

        Args:
            company_id: ID de l'entreprise
            client_id: ID du client
            start_date: Date de début de la période
            end_date: Date de fin de la période
            statuses: Liste des statuts à inclure (défaut: ["COMPLETED", "RETURN_COMPLETED"])

        Returns:
            Liste de Booking correspondant aux critères
        """
        from sqlalchemy import and_

        if statuses is None:
            statuses = ["COMPLETED", "RETURN_COMPLETED"]

        return Booking.query.filter(
            and_(
                Booking.company_id == company_id,
                Booking.client_id == client_id,
                Booking.scheduled_time >= start_date,
                Booking.scheduled_time < end_date,
                Booking.status.in_(statuses),
                Booking.invoice_line_id.is_(None),
            )
        ).all()

    def find_pending_by_company_and_time_window(
        self,
        company_id: int,
        now: datetime,
        time_window_end: datetime,
        exclude_booking_id: int | None = None,
    ) -> list[Booking]:
        """Trouve les bookings en attente pour une entreprise dans une fenêtre temporelle.

        Args:
            company_id: ID de l'entreprise
            now: Date/heure actuelle (début de la fenêtre)
            time_window_end: Date/heure de fin de la fenêtre
            exclude_booking_id: ID du booking à exclure (optionnel)

        Returns:
            Liste de Booking en attente dans la fenêtre temporelle
        """
        from typing import cast

        # Utiliser le filtre de visibilité pour les courses transférées
        query = Booking.query.filter(
            self._company_visibility_filter(company_id),
            cast("Any", Booking.status).in_(
                [BookingStatus.PENDING, BookingStatus.ACCEPTED]
            ),
            Booking.scheduled_time >= now,
            Booking.scheduled_time <= time_window_end,
        )

        if exclude_booking_id is not None:
            query = query.filter(cast("Any", Booking.id) != exclude_booking_id)

        return query.all()

    def find_by_zone_and_period(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        cutoff_date: datetime,
        statuses: list[BookingStatus] | None = None,
    ) -> int:
        """Compte les bookings dans une zone géographique pour une période donnée.

        Args:
            lat_min: Latitude minimale
            lat_max: Latitude maximale
            lon_min: Longitude minimale
            lon_max: Longitude maximale
            cutoff_date: Date de coupure (bookings après cette date)
            statuses: Liste des statuts à inclure (défaut: [COMPLETED, ACCEPTED])

        Returns:
            Nombre de bookings dans la zone
        """
        from sqlalchemy import and_

        if statuses is None:
            statuses = [BookingStatus.COMPLETED, BookingStatus.ACCEPTED]

        return Booking.query.filter(
            and_(
                Booking.pickup_lat.between(lat_min, lat_max),
                Booking.pickup_lon.between(lon_min, lon_max),
                Booking.status.in_(statuses),
                Booking.scheduled_time >= cutoff_date,
            )
        ).count()
