"""Use-case: liste des réservations (bookings) selon le rôle.

Migration progressive vers Clean Architecture:
- La logique de lecture est portée par ce module Application
- Filtrage selon le rôle (admin/client) géré ici
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from domain.client_dto import ClientDTO
from models import UserRole

logger = logging.getLogger(__name__)


class BookingLike(Protocol):
    """Protocole pour un booking avec méthode serialize."""

    id: int
    serialize: dict[str, Any]


class BookingQueryLike(Protocol):
    """Protocole pour une query SQLAlchemy paginable."""

    def paginate(
        self, *, page: int, per_page: int, error_out: bool = False
    ) -> Any:  # Pagination object
        """Pagine les résultats."""
        ...


class BookingRepoPort(Protocol):
    """Port pour récupérer des bookings."""

    def find_all_with_eager_loading_query(
        self, *, status_filter: str | None = None
    ) -> BookingQueryLike:
        """Retourne une query pour tous les bookings (admin)."""
        ...

    def find_by_client_id_with_eager_loading_query(
        self, *, client_id: int, status_filter: str | None = None
    ) -> BookingQueryLike:
        """Retourne une query pour les bookings d'un client."""
        ...


class ClientRepoPort(Protocol):
    """Port pour récupérer un client."""

    def find_by_id(self, client_id: int) -> ClientDTO | None: ...

    def find_by_user_id(self, user_id: int) -> ClientDTO | None: ...


@dataclass(frozen=True, slots=True)
class ListBookingsInput:
    """Input pour lister les réservations.

    Attributes:
        user_role: Rôle de l'utilisateur (admin ou client)
        user_id: ID de l'utilisateur
        page: Numéro de page (commence à 1)
        per_page: Nombre de résultats par page
        status_filter: Filtre optionnel par statut
    """

    user_role: UserRole
    user_id: int
    page: int
    per_page: int
    status_filter: str | None = None


@dataclass(frozen=True, slots=True)
class ListBookingsOutput:
    """Output pour lister les réservations.

    Attributes:
        success: True si l'opération a réussi
        bookings: Liste des réservations
        total: Nombre total de réservations
        page: Numéro de page actuel
        per_page: Nombre de résultats par page
        total_pages: Nombre total de pages
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    bookings: list[BookingLike] | None = None
    total: int | None = None
    page: int | None = None
    per_page: int | None = None
    total_pages: int | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class ListBookingsUseCase:
    """Use-case Application: lister les réservations selon le rôle.

    Gère la pagination et le filtrage selon le rôle:
    - Admin: tous les bookings
    - Client: seulement ses bookings

    Exemple:
        >>> uc = ListBookingsUseCase(
        ...     booking_repo=BookingRepository(),
        ...     client_repo=ClientRepository(),
        ... )
        >>> input_data = ListBookingsInput(
        ...     user_role=UserRole.admin,
        ...     user_id=1,
        ...     page=1,
        ...     per_page=100,
        ...     status_filter=None,
        ... )
        >>> result = uc.execute(input_data)
    """

    MAX_PER_PAGE = 100

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        *,
        booking_repo: BookingRepoPort,
        client_repo: ClientRepoPort | None = None,
    ) -> None:
        """Initialise le use-case.

        Args:
            booking_repo: Repository pour récupérer les bookings.
            client_repo: Repository pour récupérer les clients (requis pour rôle client).
        """
        self.booking_repo = booking_repo
        self.client_repo = client_repo

    def execute(self, input_data: ListBookingsInput) -> ListBookingsOutput:  # noqa: PLR0911
        """Exécute la liste des réservations.

        Args:
            input_data: Input avec filtres et pagination

        Returns:
            ListBookingsOutput avec les bookings paginés
        """
        # Validation
        if input_data.page < 1:
            return ListBookingsOutput(
                success=False,
                error={"page": "Le numéro de page doit être >= 1"},
                status_code=400,
            )

        if input_data.per_page < 1 or input_data.per_page > self.MAX_PER_PAGE:
            return ListBookingsOutput(
                success=False,
                error={
                    "per_page": f"Le nombre par page doit être entre 1 et {self.MAX_PER_PAGE}"
                },
                status_code=400,
            )

        try:
            if input_data.user_role == UserRole.admin:
                query = self.booking_repo.find_all_with_eager_loading_query(
                    status_filter=input_data.status_filter
                )
            elif input_data.user_role == UserRole.client:
                if self.client_repo is None:
                    logger.error(
                        "ListBookingsUseCase: client_repo requis pour rôle client"
                    )
                    return ListBookingsOutput(
                        success=False,
                        error={"error": "Configuration invalide"},
                        status_code=500,
                    )
                client = self.client_repo.find_by_user_id(input_data.user_id)
                if not client:
                    logger.warning(
                        "ListBookingsUseCase: client non trouvé pour user_id=%s",
                        input_data.user_id,
                    )
                    return ListBookingsOutput(
                        success=False,
                        error={"error": "Client non trouvé"},
                        status_code=404,
                    )
                query = self.booking_repo.find_by_client_id_with_eager_loading_query(
                    client_id=client.id, status_filter=input_data.status_filter
                )
            else:
                logger.warning(
                    "ListBookingsUseCase: rôle non supporté: %s", input_data.user_role
                )
            return ListBookingsOutput(
                success=False,
                error={"error": "Rôle non supporté"},
                status_code=400,
            )

            pagination = query.paginate(
                page=input_data.page, per_page=input_data.per_page, error_out=False
            )
            total = pagination.total or 0
            bookings = pagination.items or []
            total_pages = (
                (total + input_data.per_page - 1) // input_data.per_page
                if total > 0
                else 0
            )

            return ListBookingsOutput(
                success=True,
                bookings=list(bookings),
                total=total,
                page=input_data.page,
                per_page=input_data.per_page,
                total_pages=total_pages,
            )
        except Exception:
            logger.exception("Erreur lors de la liste des réservations")
            return ListBookingsOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )
