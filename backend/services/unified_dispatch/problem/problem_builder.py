"""Constructeur du problème de dispatch VRPTW."""

import logging
from typing import Any

from models import Company
from services.unified_dispatch import data
from services.unified_dispatch.core import settings as ud_settings
from shared.geo_utils import GeoValidator

logger = logging.getLogger(__name__)

# Constantes pour éviter les valeurs magiques
MAX_BOOKING_IDS_TO_LOG = 20  # Limite le nombre de booking IDs dans les logs


class ProblemBuilder:
    """Construit le problème de dispatch VRPTW à partir des données de l'entreprise."""

    def build(
        self,
        company: Company,  # noqa: ARG002 - Conservé pour compatibilité API future
        company_id: int,
        settings: ud_settings.Settings,
        for_date: str,
        regular_first: bool,
        allow_emergency: bool,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Construit le problème de dispatch.

        Args:
            company: Objet Company (non utilisé actuellement, conservé pour compatibilité)
            company_id: ID de l'entreprise
            settings: Paramètres de dispatch
            for_date: Date au format YYYY-MM-DD
            regular_first: Prioriser les courses régulières
            allow_emergency: Autoriser les courses d'urgence
            overrides: Paramètres de surcharge optionnels

        Returns:
            Dictionnaire contenant bookings, drivers, et métadonnées du problème
        """
        problem = (
            data.build_problem_data(
                company_id=company_id,
                settings=settings,
                for_date=for_date,
                regular_first=bool(regular_first),
                allow_emergency=allow_emergency,
                overrides=overrides or {},
            )
            or {}
        )

        n_b = len(problem.get("bookings", []))
        n_d = len(problem.get("drivers", []))
        logger.info(
            "[ProblemBuilder] Problem built: bookings=%d drivers=%d for_date=%s",
            n_b,
            n_d,
            for_date,
        )

        # Valider coordonnées géographiques avant dispatch
        self._validate_booking_coordinates(problem)

        return problem

    def build_vrptw(
        self,
        company: Company,
        bookings: list[Any],
        drivers: list[Any],
        settings: ud_settings.Settings,
        base_time: Any | None = None,
        for_date: str | None = None,
    ) -> dict[str, Any]:
        """Construit un problème VRPTW à partir de bookings et drivers donnés.

        Args:
            company: Objet Company
            bookings: Liste des bookings
            drivers: Liste des drivers
            settings: Paramètres de dispatch
            base_time: Temps de base optionnel
            for_date: Date optionnelle

        Returns:
            Dictionnaire contenant le problème VRPTW
        """
        return data.build_vrptw_problem(
            company,
            bookings,
            drivers,
            settings=settings,
            base_time=base_time,
            for_date=for_date,
        )

    def _validate_booking_coordinates(self, problem: dict[str, Any]) -> None:
        """Valide les coordonnées géographiques des bookings.

        Args:
            problem: Dictionnaire contenant les bookings
        """
        bookings_list = problem.get("bookings", [])
        if not bookings_list:
            return

        bookings_without_coords: list[int] = []
        bookings_with_invalid_coords: list[int] = []

        for booking in bookings_list:
            pickup_lat = getattr(booking, "pickup_lat", None)
            pickup_lon = getattr(booking, "pickup_lon", None)
            dropoff_lat = getattr(booking, "dropoff_lat", None)
            dropoff_lon = getattr(booking, "dropoff_lon", None)

            # Vérifier pickup
            pickup_missing = pickup_lat is None or pickup_lon is None
            pickup_invalid = False
            if not pickup_missing and pickup_lat is not None and pickup_lon is not None:
                try:
                    lat_float = float(pickup_lat)
                    lon_float = float(pickup_lon)
                    if not GeoValidator.is_valid(lat_float, lon_float):
                        pickup_invalid = True
                except (ValueError, TypeError):
                    pickup_invalid = True

            # Vérifier dropoff
            dropoff_missing = dropoff_lat is None or dropoff_lon is None
            dropoff_invalid = False
            if (
                not dropoff_missing
                and dropoff_lat is not None
                and dropoff_lon is not None
            ):
                try:
                    lat_float = float(dropoff_lat)
                    lon_float = float(dropoff_lon)
                    if not GeoValidator.is_valid(lat_float, lon_float):
                        dropoff_invalid = True
                except (ValueError, TypeError):
                    dropoff_invalid = True

            booking_id = getattr(booking, "id", None)
            if booking_id:
                if pickup_missing or dropoff_missing:
                    bookings_without_coords.append(booking_id)
                elif pickup_invalid or dropoff_invalid:
                    bookings_with_invalid_coords.append(booking_id)

        if bookings_without_coords:
            logger.warning(
                (
                    "[ProblemBuilder] ⚠️ %d booking(s) sans coordonnées "
                    "géographiques (pickup ou dropoff manquantes) : %s"
                ),
                len(bookings_without_coords),
                bookings_without_coords[:MAX_BOOKING_IDS_TO_LOG],
            )

        if bookings_with_invalid_coords:
            logger.warning(
                (
                    "[ProblemBuilder] ⚠️ %d booking(s) avec coordonnées "
                    "géographiques invalides (hors plages valides) : %s"
                ),
                len(bookings_with_invalid_coords),
                bookings_with_invalid_coords[:MAX_BOOKING_IDS_TO_LOG],
            )
