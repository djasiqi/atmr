"""Use-case: liste des réservations (bookings) selon le rôle.

Migration progressive vers Clean Architecture:
- La logique de lecture est portée par ce module Application
- Filtrage selon le rôle (admin/client) géré ici
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from models import UserRole

logger = logging.getLogger(__name__)


class BookingLike(Protocol):
    """Protocole pour un booking avec méthode serialize."""

    id: int
    serialize: dict[str, Any]


class BookingQueryLike(Protocol):
    """Protocole pour une query SQLAlchemy paginable."""

    def paginate(self, *, page: int, per_page: int, error_out: bool = False) -> Any:
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

    def find_by_user_id(self, user_id: int) -> Any | None:
        """Récupère un client par user_id."""
        ...


@dataclass(frozen=True, slots=True)
class ListBookingsResult:
    """Résultat du use-case ListBookings."""

    bookings: list[BookingLike]
    total: int
    page: int
    per_page: int
    total_pages: int


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
        >>> result = uc.execute(
        ...     user_role=UserRole.admin,
        ...     user_id=1,
        ...     page=1,
        ...     per_page=100,
        ...     status_filter=None,
        ... )
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        *,
        booking_repo: BookingRepoPort,
        client_repo: ClientRepoPort | None = None,
    ) -> None:
        """Initialise le use-case.

        Args:
            booking_repo: Repository pour récupérer les bookings.
            client_repo: Repository pour récupérer les clients
                (requis pour rôle client).
        """
        self.booking_repo = booking_repo
        self.client_repo = client_repo

    def execute(
        self,
        *,
        user_role: UserRole,
        user_id: int,
        page: int,
        per_page: int,
        status_filter: str | None = None,
    ) -> ListBookingsResult | None:
        """Exécute la liste des réservations.

        Args:
            user_role: Rôle de l'utilisateur (admin ou client).
            user_id: ID de l'utilisateur.
            page: Numéro de page (commence à 1).
            per_page: Nombre de résultats par page.
            status_filter: Filtre optionnel par statut.

        Returns:
            ListBookingsResult avec les bookings paginés, ou None si erreur
                (client non trouvé).
        """
        if user_role == UserRole.admin:
            query = self.booking_repo.find_all_with_eager_loading_query(
                status_filter=status_filter
            )
        elif user_role == UserRole.client:
            if self.client_repo is None:
                logger.error("ListBookingsUseCase: client_repo requis pour rôle client")
                return None
            client = self.client_repo.find_by_user_id(user_id)
            if not client:
                logger.warning(
                    "ListBookingsUseCase: client non trouvé pour user_id=%s", user_id
                )
                return None
            query = self.booking_repo.find_by_client_id_with_eager_loading_query(
                client_id=client.id, status_filter=status_filter
            )
        else:
            logger.warning("ListBookingsUseCase: rôle non supporté: %s", user_role)
            return None

        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        total = pagination.total or 0
        bookings = pagination.items or []
        total_pages = (total + per_page - 1) // per_page if total > 0 else 0

        return ListBookingsResult(
            bookings=list(bookings),
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
        )
