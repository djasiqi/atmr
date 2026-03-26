"""Repository pour l'accès aux données Client."""

from __future__ import annotations

import re
from typing import Protocol

from domain.client_dto import ClientDTO
from models import Client, User

logger = __import__("logging").getLogger(__name__)

DATE_DDMMYYYY_RE = re.compile(r"^(\d{2})[./-](\d{2})[./-](\d{4})$")


def _build_search_patterns(search: str) -> list[str]:
    term = search.strip()
    if not term:
        return []
    patterns = {f"%{term}%"}
    match = DATE_DDMMYYYY_RE.match(term)
    if match:
        day, month, year = match.groups()
        patterns.add(f"%{year}-{month}-{day}%")
    return list(patterns)


class ClientRepositoryPort(Protocol):
    """Port (interface) pour le repository Client.

    Cette interface définit le contrat que doit respecter toute implémentation
    du repository. Elle permet de découpler la couche Application de l'implémentation
    concrète (SQLAlchemy, MongoDB, etc.).
    """

    def find_by_id(self, client_id: int) -> ClientDTO | None:
        """Trouve un client par son ID.

        Args:
            client_id: ID du client

        Returns:
            ClientDTO ou None si non trouvé
        """
        ...

    def find_by_user_id(self, user_id: int) -> ClientDTO | None:
        """Trouve un client par l'ID de son utilisateur.

        Args:
            user_id: ID de l'utilisateur propriétaire du client

        Returns:
            ClientDTO ou None si non trouvé
        """
        ...

    def find_by_public_id(self, public_id: str) -> ClientDTO | None:
        """Trouve un client par l'ID public de son utilisateur.

        Args:
            public_id: ID public de l'utilisateur propriétaire du client

        Returns:
            ClientDTO ou None si non trouvé
        """
        ...

    def find_by_company_with_user_and_search(
        self, company_id: int, search: str | None = None
    ) -> list[ClientDTO]:
        """Trouve les clients d'une entreprise avec eager loading de user et recherche optionnelle.

        Args:
            company_id: ID de l'entreprise
            search: Terme de recherche optionnel (filtre sur first_name/last_name)

        Returns:
            Liste de ClientDTO avec données user chargées
        """
        ...


class ClientRepository:
    """Repository SQLAlchemy pour Client.

    Implémentation concrète du port ClientRepositoryPort utilisant SQLAlchemy.
    Cette classe convertit les modèles SQLAlchemy en DTOs pour maintenir
    le découplage avec la couche Application.
    """

    def _to_dto(self, client: Client, include_user: bool = False) -> ClientDTO:
        """Convertit un modèle SQLAlchemy Client en DTO.

        Args:
            client: Modèle SQLAlchemy Client
            include_user: Si True, inclut les données du user (si chargé)

        Returns:
            ClientDTO correspondant
        """
        user_data = {}
        if include_user and client.user:
            user_data = {
                "user_first_name": client.user.first_name,
                "user_last_name": client.user.last_name,
                "user_email": client.user.email,
                "user_phone": client.user.phone,
                "user_public_id": client.user.public_id,
                "user_gender": (
                    client.user.gender.value if client.user.gender else None
                ),
                "user_birth_date": (
                    client.user.birth_date.isoformat()
                    if client.user.birth_date
                    else None
                ),
            }

        return ClientDTO(
            id=client.id,
            user_id=client.user_id,
            company_id=client.company_id,
            client_type=client.client_type,
            billing_address=client.billing_address,
            billing_lat=client.billing_lat,
            billing_lon=client.billing_lon,
            contact_email=client.contact_email,
            contact_phone=client.contact_phone,
            domicile_address=client.domicile_address,
            domicile_zip=client.domicile_zip,
            domicile_city=client.domicile_city,
            domicile_lat=client.domicile_lat,
            domicile_lon=client.domicile_lon,
            door_code=client.door_code,
            floor=client.floor,
            access_notes=client.access_notes,
            institution_name=client.institution_name,
            institution_contact=getattr(client, "institution_contact", None),
            institution_phone=getattr(client, "institution_phone", None),
            is_institution=client.is_institution,  # type: ignore[reportGeneralTypeIssues]
            is_active=client.is_active,
            residence_facility=getattr(client, "residence_facility", None),
            preferential_rate=getattr(client, "preferential_rate", None),
            avs_number=getattr(client, "avs_number", None),
            created_at=getattr(client, "created_at", None),
            **user_data,
        )

    def find_by_id(self, client_id: int) -> ClientDTO | None:
        """Trouve un client par son ID.

        Args:
            client_id: ID du client

        Returns:
            ClientDTO ou None si non trouvé
        """
        client = Client.query.get(client_id)
        if client is None:
            return None
        return self._to_dto(client)

    def find_by_user_id(self, user_id: int) -> ClientDTO | None:
        """Trouve un client par l'ID de son utilisateur.

        Args:
            user_id: ID de l'utilisateur propriétaire du client

        Returns:
            ClientDTO ou None si non trouvé
        """
        client = Client.query.filter_by(user_id=user_id).first()
        if client is None:
            return None
        return self._to_dto(client)

    def find_by_public_id(self, public_id: str) -> ClientDTO | None:
        """Trouve un client par l'ID public de son utilisateur.

        Args:
            public_id: ID public de l'utilisateur propriétaire du client

        Returns:
            ClientDTO ou None si non trouvé
        """
        client = (
            Client.query.join(User, Client.user_id == User.id)
            .filter(User.public_id == public_id)
            .one_or_none()
        )
        if client is None:
            return None
        return self._to_dto(client)

    # Méthodes legacy - retournent des modèles SQLAlchemy pour compatibilité
    def find_model_by_id(self, client_id: int) -> Client | None:
        """Trouve un client par son ID.

        Args:
            client_id: ID du client

        Returns:
            Client ou None si non trouvé
        """
        return Client.query.get(client_id)

    def find_by_company_with_user_and_search(
        self, company_id: int, search: str | None = None
    ) -> list[ClientDTO]:
        """Trouve les clients d'une entreprise avec eager loading de user et recherche optionnelle.

        Args:
            company_id: ID de l'entreprise
            search: Terme de recherche optionnel (filtre sur first_name/last_name)

        Returns:
            Liste de ClientDTO avec données user chargées
        """
        from sqlalchemy import String, cast, func, or_
        from sqlalchemy.orm import joinedload

        from models import ClientType, User

        query = Client.query.options(joinedload(Client.user)).filter(
            Client.company_id == company_id,
            Client.client_type != ClientType.SELF_SERVICE,
        )

        patterns = _build_search_patterns(search or "")
        if patterns:
            user_fields = [
                User.first_name,
                User.last_name,
                User.email,
                User.phone,
                User.username,
                cast(User.birth_date, String),
                func.concat(
                    func.coalesce(User.first_name, ""), " ", func.coalesce(User.last_name, "")
                ),
                func.concat(
                    func.coalesce(User.last_name, ""), " ", func.coalesce(User.first_name, "")
                ),
            ]
            client_fields = [
                Client.contact_email,
                Client.contact_phone,
                Client.domicile_address,
                Client.domicile_zip,
                Client.domicile_city,
                Client.residence_facility,
                Client.institution_name,
                Client.billing_address,
            ]
            conditions = []
            for pattern in patterns:
                for field in user_fields:
                    conditions.append(Client.user.has(field.ilike(pattern)))
                for field in client_fields:
                    conditions.append(field.ilike(pattern))
            query = query.filter(or_(*conditions))

        clients = query.all()
        return [self._to_dto(client, include_user=True) for client in clients]

    # Méthodes legacy - retournent des modèles SQLAlchemy pour compatibilité
    def find_models_by_company_with_user_and_search(
        self, company_id: int, search: str | None = None
    ) -> list[Client]:
        """Trouve les clients d'une entreprise avec eager loading de user et recherche optionnelle.

        Args:
            company_id: ID de l'entreprise
            search: Terme de recherche optionnel (filtre sur first_name/last_name)

        Returns:
            Liste de Client avec user chargé (modèle SQLAlchemy)

        Note:
            Méthode legacy - utiliser find_by_company_with_user_and_search() pour obtenir des DTOs
            Cette méthode charge aussi default_billed_to_company pour inclure default_billing dans serialize
        """
        from sqlalchemy import String, cast, func, or_
        from sqlalchemy.orm import joinedload

        from models import ClientType, User

        # Charger user et default_billed_to_company pour que serialize inclue default_billing
        query = Client.query.options(
            joinedload(Client.user),
            joinedload(Client.default_billed_to_company)
        ).filter(
            Client.company_id == company_id,
            Client.client_type != ClientType.SELF_SERVICE,
        )

        patterns = _build_search_patterns(search or "")
        if patterns:
            user_fields = [
                User.first_name,
                User.last_name,
                User.email,
                User.phone,
                User.username,
                cast(User.birth_date, String),
                func.concat(
                    func.coalesce(User.first_name, ""), " ", func.coalesce(User.last_name, "")
                ),
                func.concat(
                    func.coalesce(User.last_name, ""), " ", func.coalesce(User.first_name, "")
                ),
            ]
            client_fields = [
                Client.contact_email,
                Client.contact_phone,
                Client.domicile_address,
                Client.domicile_zip,
                Client.domicile_city,
                Client.residence_facility,
                Client.institution_name,
                Client.billing_address,
            ]
            conditions = []
            for pattern in patterns:
                for field in user_fields:
                    conditions.append(Client.user.has(field.ilike(pattern)))
                for field in client_fields:
                    conditions.append(field.ilike(pattern))
            query = query.filter(or_(*conditions))

        return query.all()

    def find_models_by_company_with_user_and_search_paginated(
        self,
        company_id: int,
        search: str | None,
        page: int,
        per_page: int,
    ) -> tuple[list[Client], int]:
        """Liste paginée en SQL (OFFSET/LIMIT) + total, sans charger toute la table."""
        from sqlalchemy import String, cast, func, or_
        from sqlalchemy.orm import joinedload

        from models import ClientType

        query = Client.query.options(
            joinedload(Client.user),
            joinedload(Client.default_billed_to_company),
        ).filter(
            Client.company_id == company_id,
            Client.client_type != ClientType.SELF_SERVICE,
        )

        patterns = _build_search_patterns(search or "")
        if patterns:
            user_fields = [
                User.first_name,
                User.last_name,
                User.email,
                User.phone,
                User.username,
                cast(User.birth_date, String),
                func.concat(
                    func.coalesce(User.first_name, ""), " ", func.coalesce(User.last_name, "")
                ),
                func.concat(
                    func.coalesce(User.last_name, ""), " ", func.coalesce(User.first_name, "")
                ),
            ]
            client_fields = [
                Client.contact_email,
                Client.contact_phone,
                Client.domicile_address,
                Client.domicile_zip,
                Client.domicile_city,
                Client.residence_facility,
                Client.institution_name,
                Client.billing_address,
            ]
            conditions = []
            for pattern in patterns:
                for field in user_fields:
                    conditions.append(Client.user.has(field.ilike(pattern)))
                for field in client_fields:
                    conditions.append(field.ilike(pattern))
            query = query.filter(or_(*conditions))

        query = query.order_by(Client.id.asc())
        total = query.order_by(None).count()
        offset = max(page - 1, 0) * per_page
        page_clients = query.offset(offset).limit(per_page).all()
        return page_clients, total

    def find_models_by_company_and_institution_status(
        self, company_id: int, is_institution: bool = True, is_active: bool = True
    ) -> list[Client]:
        """Trouve les clients d'une entreprise par statut d'institution.

        Args:
            company_id: ID de l'entreprise
            is_institution: Si True, cherche les institutions (défaut: True)
            is_active: Si True, cherche uniquement les clients actifs (défaut: True)

        Returns:
            Liste de Client
        """
        return Client.query.filter_by(
            company_id=company_id,
            is_institution=is_institution,
            is_active=is_active,
        ).all()

    def find_model_by_id_and_company(
        self, client_id: int, company_id: int
    ) -> Client | None:
        """Trouve un client par son ID et company_id (retourne le modèle SQLAlchemy).

        Args:
            client_id: ID du client
            company_id: ID de l'entreprise

        Returns:
            Client ou None si non trouvé
        """
        return Client.query.filter_by(id=client_id, company_id=company_id).first()

    def find_model_by_id_with_user(
        self, client_id: int, company_id: int
    ) -> Client | None:
        """Trouve un client par son ID avec eager loading de user (retourne le modèle SQLAlchemy).

        Args:
            client_id: ID du client
            company_id: ID de l'entreprise

        Returns:
            Client ou None si non trouvé (avec user chargé)
        """
        from sqlalchemy.orm import joinedload

        return (
            Client.query.options(joinedload(Client.user))
            .filter_by(id=client_id, company_id=company_id)
            .first()
        )

    def find_by_public_id_with_user(self, public_id: str) -> Client | None:
        """Trouve un client par l'ID public de son utilisateur avec eager loading de user.

        Args:
            public_id: ID public de l'utilisateur propriétaire du client

        Returns:
            Client ou None si non trouvé (avec user chargé)
        """
        from sqlalchemy.orm import joinedload

        return (
            Client.query.options(joinedload(Client.user))
            .join(User, Client.user_id == User.id)
            .filter(User.public_id == public_id)
            .one_or_none()
        )

    def find_by_user_id_with_user(self, user_id: int) -> Client | None:
        """Trouve un client par l'ID de son utilisateur avec eager loading de user.

        Args:
            user_id: ID de l'utilisateur propriétaire du client

        Returns:
            Client ou None si non trouvé (avec user chargé)
        """
        from sqlalchemy.orm import joinedload

        return (
            Client.query.options(joinedload(Client.user))
            .filter_by(user_id=user_id)
            .first()
        )

    def find_by_public_id_with_payments(self, public_id: str) -> Client | None:
        """Trouve un client par l'ID public de son utilisateur avec eager loading de payments.

        Args:
            public_id: ID public de l'utilisateur propriétaire du client

        Returns:
            Client ou None si non trouvé (avec payments chargé)
        """
        from sqlalchemy.orm import joinedload

        return (
            Client.query.options(joinedload(Client.payments))
            .join(User, Client.user_id == User.id)
            .filter(User.public_id == public_id)
            .one_or_none()
        )

    def find_by_user_id_with_bookings(self, user_id: int) -> Client | None:
        """Trouve un client par l'ID de son utilisateur avec eager loading de bookings.

        Args:
            user_id: ID de l'utilisateur propriétaire du client

        Returns:
            Client ou None si non trouvé (avec bookings chargé)
        """
        from sqlalchemy.orm import joinedload

        return (
            Client.query.options(joinedload(Client.bookings))
            .filter_by(user_id=user_id)
            .first()
        )

    def find_by_search_with_user(self, search: str) -> list[Client]:
        """Trouve les clients dont le prénom ou le nom contient le terme de recherche.

        Args:
            search: Terme de recherche (filtre sur first_name/last_name)

        Returns:
            Liste de Client avec user chargé
        """
        from sqlalchemy import or_

        return (
            Client.query.join(User)
            .filter(
                or_(
                    User.first_name.ilike(f"%{search}%"),
                    User.last_name.ilike(f"%{search}%"),
                )
            )
            .all()
        )
