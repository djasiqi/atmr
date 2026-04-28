"""Repository pour l'accès aux données Booking avec conversion DTO.

Ce repository découple les services de l'implémentation SQLAlchemy
en retournant des DTOs au lieu de modèles SQLAlchemy directs.
"""

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import joinedload

from domain.booking_dto import BookingDTO
from models import Booking, BookingStatus

logger = __import__("logging").getLogger(__name__)


class BookingRepository:
    """Repository pour l'accès aux données Booking avec conversion DTO."""

    @staticmethod
    def _company_visibility_filter(company_id: int):
        """Crée un filtre SQLAlchemy pour la visibilité des courses (modèle Owner vs Executor).

        Règles de visibilité :
        - Owner : l'entreprise voit ses propres courses (company_id == company_id)
        - Owner via transfer : l'entreprise voit les courses où elle est owner_company_id
          dans un BookingTransfer accepté/complété (robustesse si company_id a été changé par le passé)
        - Executor : l'entreprise voit les courses où elle est exécutante (executing_company_id)

        Args:
            company_id: ID de l'entreprise

        Returns:
            Filtre SQLAlchemy pour la visibilité
        """
        from sqlalchemy import and_, exists, or_

        from models.booking_transfer import BookingTransfer
        from models.enums import DispatchOfferStatus, TransferStatus
        from models.service_area_pricing import DispatchOffer

        # Seuls ACCEPTED/COMPLETED : exclut PENDING (pas encore accepté) et REJECTED (refusé/annulé)
        owner_via_transfer = exists().where(
            and_(
                BookingTransfer.booking_id == Booking.id,
                BookingTransfer.owner_company_id == company_id,
                BookingTransfer.status.in_(
                    [TransferStatus.ACCEPTED, TransferStatus.COMPLETED]
                ),
            )
        )
        return or_(
            # Owner : l'entreprise voit ses propres courses
            (Booking.company_id == company_id),
            # Owner via transfer (robustesse pour données où company_id avait été mis à l'exécutant)
            owner_via_transfer,
            # Executor : courses transférées en PENDING où l'entreprise est l'exécutante
            and_(
                (Booking.executing_company_id == company_id),
                (Booking.status == BookingStatus.PENDING),
            ),
            # Executor : courses transférées acceptées/en cours où l'entreprise est l'exécutante
            and_(
                (Booking.executing_company_id == company_id),
                (
                    Booking.status.in_(
                        [
                            BookingStatus.ACCEPTED,
                            BookingStatus.ASSIGNED,
                            BookingStatus.EN_ROUTE,
                            BookingStatus.IN_PROGRESS,
                            BookingStatus.COMPLETED,
                            BookingStatus.RETURN_COMPLETED,
                        ]
                    )
                ),
            ),
            # Demande ouverte : pas encore de transporteur, mais offre PROPOSED pour cette entreprise
            and_(
                Booking.company_id.is_(None),
                Booking.status.in_(
                    [
                        BookingStatus.PENDING,
                        BookingStatus.AWAITING_CLIENT_PAYMENT,
                    ]
                ),
                exists().where(
                    and_(
                        DispatchOffer.booking_id == Booking.id,
                        DispatchOffer.company_id == company_id,
                        DispatchOffer.status == DispatchOfferStatus.PROPOSED,
                    )
                ),
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
        # ✅ NOUVEAU : Inclure les courses retour sans heure attribuée
        from sqlalchemy import case, or_

        bookings = (
            Booking.query.filter(
                self._company_visibility_filter(company_id),
                Booking.status.in_([BookingStatus.ACCEPTED, BookingStatus.ASSIGNED]),
                # Inclure les courses avec heure planifiée OU les retours sans heure
                or_(
                    Booking.scheduled_time.isnot(None),
                    Booking.is_return,  # Inclure les retours même sans heure
                ),
            )
            .filter(
                # Si scheduled_time existe, doit être <= horizon
                or_(
                    Booking.scheduled_time.is_(None),
                    Booking.scheduled_time <= horizon_local,
                )
            )
            # Trier : courses avec heure d'abord (par heure), puis courses sans heure à la fin
            .order_by(
                case((Booking.scheduled_time.is_(None), 1), else_=0),
                Booking.scheduled_time.asc().nullslast(),
                Booking.id.asc(),
            )
            .all()
        )

        # Filtrer uniquement les retours avec time_confirmed=False (mais garder ceux avec scheduled_time=None)
        filtered_bookings = []
        for booking in bookings:
            is_return = bool(getattr(booking, "is_return", False))
            time_confirmed = bool(getattr(booking, "time_confirmed", True))

            # Exclure seulement les retours avec time_confirmed=False ET scheduled_time défini
            # (les retours sans heure sont OK)
            if is_return and not time_confirmed and booking.scheduled_time is not None:
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

    def find_model_by_id_with_visibility(
        self, booking_id: int, company_id: int
    ) -> Booking | None:
        """Trouve un booking par son ID avec visibilité transferts partenaires.

        Utilise le filtre de visibilité pour supporter les transferts partenaires.
        Une entreprise peut voir une réservation si :
        - Elle est propriétaire (company_id)
        - Elle est exécutante d'un transfert (executing_company_id)

        Args:
            booking_id: ID du booking
            company_id: ID de l'entreprise

        Returns:
            Booking ou None si non trouvé
        """
        return Booking.query.filter(
            Booking.id == booking_id,
            self._company_visibility_filter(company_id),
        ).first()

    def find_model_by_id_with_visibility_for_update(
        self, booking_id: int, company_id: int
    ) -> Booking | None:
        """Trouve un booking visible et verrouille la ligne pour mise à jour atomique.

        Utilisé pour les opérations critiques concurrentes (ex: assignation chauffeur).
        """
        return (
            Booking.query.filter(
                Booking.id == booking_id,
                self._company_visibility_filter(company_id),
            )
            .with_for_update()
            .first()
        )

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

        logger.warning(
            (
                "[BookingRepository] find_models_by_company_with_filters: company_id=%s, "
                "day_str=%s, status=%s"
            ),
            company_id,
            day_str,
            status,
        )

        if day_str:
            from sqlalchemy import or_

            from shared.time_utils import day_local_bounds

            try:
                start_local, end_local = day_local_bounds(day_str)

                # ÉTAPE 1 : Récupérer d'abord les IDs des courses aller du jour
                outbound_bookings_query = Booking.query.filter(
                    self._company_visibility_filter(company_id),
                    Booking.scheduled_time >= start_local,
                    Booking.scheduled_time < end_local,
                    ~Booking.is_return,
                )
                outbound_bookings = outbound_bookings_query.all()
                outbound_ids = [b.id for b in outbound_bookings]

                logger.warning(
                    "[BookingRepository] Found %s outbound bookings for date range: %s",
                    len(outbound_ids),
                    outbound_ids[:5] if outbound_ids else [],
                )

                # Vérifier si des retours sans heure existent pour ces courses aller
                if outbound_ids:
                    # ✅ FIX: Inclure les retours avec scheduled_time=None OU time_confirmed=False
                    return_bookings_query = Booking.query.filter(
                        self._company_visibility_filter(company_id),
                        Booking.is_return,
                        or_(
                            Booking.scheduled_time.is_(None),
                            ~Booking.time_confirmed,  # Inclure les retours avec heure non confirmée
                        ),
                        Booking.parent_booking_id.in_(outbound_ids),
                    )
                    return_bookings = return_bookings_query.all()
                    logger.warning(
                        "[BookingRepository] Found %s return bookings without time linked to outbound bookings: %s",
                        len(return_bookings),
                        [
                            {
                                "id": b.id,
                                "parent_id": b.parent_booking_id,
                                "status": b.status.value,
                            }
                            for b in return_bookings[:3]
                        ],
                    )

                # ÉTAPE 2 : Inclure les courses avec heure dans la plage OU les retours liés aux courses aller du jour
                if outbound_ids:
                    # Si on a des courses aller, inclure aussi leurs retours sans heure ou heure non confirmée
                    query = query.filter(
                        or_(
                            # Courses avec heure planifiée dans la plage
                            (Booking.scheduled_time >= start_local)
                            & (Booking.scheduled_time < end_local),
                            # Courses retour sans heure confirmée dont la course aller est aujourd'hui
                            Booking.is_return
                            & or_(
                                Booking.scheduled_time.is_(None),
                                ~Booking.time_confirmed,
                            )
                            & (Booking.parent_booking_id.in_(outbound_ids)),
                        )
                    )
                else:
                    # Pas de courses aller, filtrer normalement par date
                    query = query.filter(
                        (Booking.scheduled_time >= start_local)
                        & (Booking.scheduled_time < end_local)
                    )
                logger.warning(
                    "[BookingRepository] Date filter applied: %s to %s (including returns without time linked to %s outbound bookings)",
                    start_local,
                    end_local,
                    len(outbound_ids),
                )
            except ValueError:
                # Retourner une liste vide si la date est invalide
                logger.warning("[BookingRepository] Invalid date format: %s", day_str)
                return []

        if status:
            query = query.filter(Booking.status == status)
            logger.warning("[BookingRepository] Status filter applied: %s", status)

        # Trier : courses avec heure en premier (desc), puis courses sans heure à la fin
        from sqlalchemy import case

        bookings = query.order_by(
            case((Booking.scheduled_time.is_(None), 1), else_=0),
            Booking.scheduled_time.desc().nullslast(),
            Booking.id.desc(),
        ).all()

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
        query = (
            Booking.query.options(joinedload(Booking.return_trip))
            .filter_by(client_id=client_id)
            .order_by(Booking.scheduled_time.desc())
        )
        if limit:
            query = query.limit(limit)
        return query.all()

    def find_models_by_client_and_company(
        self, client_id: int, company_id: int, limit: int | None = None
    ) -> list[Booking]:
        """Trouve les bookings d'un client pour une entreprise (retourne les modèles SQLAlchemy).

        Args:
            client_id: ID du client
            company_id: ID de l'entreprise

        Returns:
            Liste de Booking triées par scheduled_time décroissant
        """
        query = Booking.query.filter_by(client_id=client_id, company_id=company_id).order_by(
            Booking.scheduled_time.desc()
        )
        if limit:
            query = query.limit(limit)
        return query.all()

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

    def find_children_by_parent_booking_id(
        self, parent_booking_id: int
    ) -> list[Booking]:
        """Trouve tous les segments enfants (retours) d'un booking parent.

        Utilisé pour l'annulation scope=reservation (aller + retour).

        Args:
            parent_booking_id: ID du booking parent (aller)

        Returns:
            Liste des bookings avec parent_booking_id = parent_booking_id
        """
        return (
            Booking.query.filter_by(parent_booking_id=parent_booking_id)
            .order_by(Booking.scheduled_time.asc())
            .all()
        )

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
        billed_to_type: str | None = None,
    ) -> list[Booking]:
        """Trouve les bookings terminés ou annulations facturables non facturés pour un client.

        Aligné sur l'éligibilité facturation : COMPLETED, RETURN_COMPLETED, ou
        CANCELED + is_cancellation_billable + amount > 0 (si billed_to_type=patient).
        Période sur scheduled_time. Politique : la règle « aller annulé ⇒ retour exclu »
        s'applique uniquement pour billed_to_type=patient (facturation directe).
        Pour clinic (S2), la même règle est gérée côté routes / génération facture.

        Args:
            company_id: ID de l'entreprise
            client_id: ID du client
            period_year: Année de la période (optionnel)
            period_month: Mois de la période (optionnel)
            billed_to_type: "patient", "clinic", "insurance" (optionnel)

        Returns:
            Liste de Booking triés par scheduled_time ascendant
        """
        from sqlalchemy import or_
        from sqlalchemy.orm import aliased

        target_statuses = [
            BookingStatus.COMPLETED.value,
            BookingStatus.RETURN_COMPLETED.value,
        ]
        status_filter = Booking.status.in_(target_statuses)
        # Annulations facturables : patient ou clinique (S2)
        if billed_to_type in ("patient", "clinic"):
            canceled_eligible = (
                (Booking.status == BookingStatus.CANCELED.value)
                & (Booking.amount > 0)
                & (
                    (Booking.is_cancellation_billable == True)  # noqa: E712
                    | (
                        Booking.billing_override_reason.isnot(None)
                        & (Booking.billing_override_reason != "")
                    )
                )
            )
            status_filter = or_(status_filter, canceled_eligible)

        query = Booking.query.filter(
            Booking.company_id == company_id,
            Booking.client_id == client_id,
            status_filter,
            Booking.invoice_line_id.is_(None),
        )

        if billed_to_type and billed_to_type in ("patient", "clinic", "insurance"):
            query = query.filter(Booking.billed_to_type == billed_to_type)

        # Règle « aller annulé ⇒ retour exclu » uniquement pour facturation directe patient.
        if billed_to_type == "patient":
            parent_alias = aliased(Booking)
            query = query.outerjoin(
                parent_alias, parent_alias.id == Booking.parent_booking_id
            ).filter(
                or_(
                    Booking.is_return == False,  # noqa: E712
                    parent_alias.id.is_(None),
                    parent_alias.status != BookingStatus.CANCELED.value,
                )
            )

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

    def find_unbilled_ids_by_company_and_client(
        self,
        company_id: int,
        client_id: int,
        period_year: int | None = None,
        period_month: int | None = None,
        billed_to_type: str | None = None,
    ) -> list[int]:
        """Trouve uniquement les IDs des bookings terminés non facturés pour un client et une entreprise.

        Optimisation pour "select all" : retourne uniquement les IDs sans charger les objets complets.

        Args:
            company_id: ID de l'entreprise
            client_id: ID du client
            period_year: Année de la période (optionnel)
            period_month: Mois de la période (optionnel)
            billed_to_type: Type de facturation ("patient", "clinic", "insurance") (optionnel)

        Returns:
            Liste d'IDs de Booking triés par scheduled_time ascendant
        """
        from datetime import datetime

        from sqlalchemy import or_
        from sqlalchemy.orm import aliased

        from models import BookingStatus, Invoice, InvoiceLine, InvoiceStatus, db

        target_statuses = [
            BookingStatus.COMPLETED.value,
            BookingStatus.RETURN_COMPLETED.value,
        ]
        status_filter = Booking.status.in_(target_statuses)
        if billed_to_type in ("patient", "clinic"):
            canceled_eligible = (
                (Booking.status == BookingStatus.CANCELED.value)
                & (Booking.amount > 0)
                & (
                    (Booking.is_cancellation_billable == True)  # noqa: E712
                    | (
                        Booking.billing_override_reason.isnot(None)
                        & (Booking.billing_override_reason != "")
                    )
                )
            )
            status_filter = or_(status_filter, canceled_eligible)

        query = (
            db.session.query(Booking.id)
            .outerjoin(InvoiceLine, Booking.invoice_line_id == InvoiceLine.id)
            .outerjoin(Invoice, InvoiceLine.invoice_id == Invoice.id)
            .filter(
                Booking.company_id == company_id,
                Booking.client_id == client_id,
                status_filter,
                or_(
                    Booking.invoice_line_id.is_(None),
                    Invoice.status == InvoiceStatus.CANCELLED,
                ),
            )
        )

        if billed_to_type and billed_to_type in ("patient", "clinic", "insurance"):
            query = query.filter(Booking.billed_to_type == billed_to_type)

        # Même politique que find_models_* : règle parent uniquement pour patient.
        if billed_to_type == "patient":
            parent_alias = aliased(Booking)
            query = query.outerjoin(
                parent_alias, parent_alias.id == Booking.parent_booking_id
            ).filter(
                or_(
                    Booking.is_return == False,  # noqa: E712
                    parent_alias.id.is_(None),
                    parent_alias.status != BookingStatus.CANCELED.value,
                )
            )

        # Filtrer par période si fournie (scheduled_time pour inclure annulations)
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

        return [row[0] for row in query.order_by(Booking.scheduled_time.asc()).all()]

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
                selectinload(Booking.executing_company),
                selectinload(Booking.return_trip),
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
        from sqlalchemy.orm import joinedload, selectinload

        from models import Client, Driver
        from models.transport_request import TransportRequest

        query = Booking.query.options(
            joinedload(Booking.driver).joinedload(Driver.user),
            joinedload(Booking.client).joinedload(Client.user),
            joinedload(Booking.company),
            joinedload(Booking.executing_company),
            joinedload(Booking.billed_to_company),
            selectinload(Booking.return_trip),
            selectinload(Booking.payments),
            selectinload(Booking.source_request).selectinload(
                TransportRequest.accepted_by_company
            ),
            selectinload(Booking.source_request).selectinload(
                TransportRequest.created_by
            ),
            selectinload(Booking.source_request).selectinload(
                TransportRequest.institution
            ),
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
        from sqlalchemy.orm import joinedload, selectinload

        from models import Client, Driver
        from models.transport_request import TransportRequest

        query = (
            Booking.query.options(
                joinedload(Booking.client).joinedload(Client.user),
                joinedload(Booking.driver).joinedload(Driver.user),
                joinedload(Booking.company),
                joinedload(Booking.executing_company),
                joinedload(Booking.billed_to_company),
                selectinload(Booking.return_trip),
                selectinload(Booking.payments),
                selectinload(Booking.source_request).selectinload(
                    TransportRequest.accepted_by_company
                ),
                selectinload(Booking.source_request).selectinload(
                    TransportRequest.created_by
                ),
                selectinload(Booking.source_request).selectinload(
                    TransportRequest.institution
                ),
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
        from sqlalchemy import func

        from ext import db

        result = (
            db.session.query(func.count(Booking.id))
            .filter(
                Booking.created_at >= start_date,
                Booking.created_at <= end_date,
            )
            .scalar()
        )
        return int(result or 0)

    def get_monthly_booking_counts(
        self, start_date: Any, end_date: Any
    ) -> dict[str, int]:
        """Compte les bookings par mois calendaire (created_at) en une requête.

        Utilise ``date_trunc`` (PostgreSQL). Les clés sont au format ``YYYY-MM``.

        Args:
            start_date: Borne inférieure inclusive sur ``created_at``
            end_date: Borne supérieure inclusive sur ``created_at``

        Returns:
            Dictionnaire mois -> nombre de bookings
        """
        from sqlalchemy import func

        from ext import db

        month_expr = func.date_trunc("month", Booking.created_at)

        rows = (
            db.session.query(
                month_expr.label("month"),
                func.count(Booking.id).label("count"),
            )
            .filter(
                Booking.created_at >= start_date,
                Booking.created_at <= end_date,
            )
            .group_by(month_expr)
            .order_by(month_expr)
            .all()
        )

        out: dict[str, int] = {}
        for row in rows:
            if row.month is None:
                continue
            month_key = row.month.strftime("%Y-%m")
            # row.count est la méthode Row.count(), pas la colonne agrégée label("count")
            cnt = row._mapping["count"]
            out[month_key] = int(cnt or 0)
        return out

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
