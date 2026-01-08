"""Repository pour l'accès aux données User."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, cast

from domain.user_dto import UserDTO
from models import User, UserRole

logger = __import__("logging").getLogger(__name__)


class UserRepositoryPort(Protocol):
    """Port (interface) pour le repository User.

    Cette interface définit le contrat que doit respecter toute implémentation
    du repository. Elle permet de découpler la couche Application de l'implémentation
    concrète (SQLAlchemy, MongoDB, etc.).
    """

    def find_by_id(self, user_id: int) -> UserDTO | None:
        """Trouve un utilisateur par son ID.

        Args:
            user_id: ID de l'utilisateur

        Returns:
            UserDTO ou None si non trouvé
        """
        ...

    def find_by_public_id(self, public_id: str) -> UserDTO | None:
        """Trouve un utilisateur par son ID public.

        Args:
            public_id: ID public de l'utilisateur

        Returns:
            UserDTO ou None si non trouvé
        """
        ...

    def find_by_email(self, email: str) -> UserDTO | None:
        """Trouve un utilisateur par son email.

        Args:
            email: Email de l'utilisateur

        Returns:
            UserDTO ou None si non trouvé
        """
        ...

    def find_by_username(self, username: str) -> UserDTO | None:
        """Trouve un utilisateur par son username.

        Args:
            username: Username de l'utilisateur

        Returns:
            UserDTO ou None si non trouvé
        """
        ...


class UserRepository:
    """Repository SQLAlchemy pour User.

    Implémentation concrète du port UserRepositoryPort utilisant SQLAlchemy.
    Cette classe convertit les modèles SQLAlchemy en DTOs pour maintenir
    le découplage avec la couche Application.
    """

    def _to_dto(self, user: User) -> UserDTO:
        """Convertit un modèle SQLAlchemy User en DTO.

        Args:
            user: Modèle SQLAlchemy User

        Returns:
            UserDTO correspondant
        """
        return UserDTO(
            id=user.id,
            public_id=cast(str, user.public_id),
            username=cast(str, user.username),
            email=cast(str | None, user.email),
            first_name=user.first_name,
            last_name=user.last_name,
            phone=user.phone,
            address=user.address,
            zip_code=user.zip_code,
            city=user.city,
            birth_date=user.birth_date,
            gender=user.gender,
            profile_image=user.profile_image,
            role=user.role,
            created_at=cast(datetime | None, user.created_at),
            updated_at=cast(datetime | None, user.updated_at),
        )

    def find_by_id(self, user_id: int) -> UserDTO | None:
        """Trouve un utilisateur par son ID.

        Args:
            user_id: ID de l'utilisateur

        Returns:
            UserDTO ou None si non trouvé
        """
        user = User.query.get(user_id)
        if user is None:
            return None
        return self._to_dto(user)

    def find_by_public_id(self, public_id: str) -> UserDTO | None:
        """Trouve un utilisateur par son ID public.

        Args:
            public_id: ID public de l'utilisateur

        Returns:
            UserDTO ou None si non trouvé
        """
        user = User.query.filter_by(public_id=public_id).one_or_none()
        if user is None:
            return None
        return self._to_dto(user)

    def find_by_email(self, email: str) -> UserDTO | None:
        """Trouve un utilisateur par son email.

        Args:
            email: Email de l'utilisateur

        Returns:
            UserDTO ou None si non trouvé
        """
        user = User.query.filter_by(email=email).one_or_none()
        if user is None:
            return None
        return self._to_dto(user)

    def find_by_email_and_role(self, email: str, role: UserRole) -> User | None:
        """Trouve un utilisateur par email et rôle.

        Args:
            email: Email de l'utilisateur
            role: Rôle de l'utilisateur

        Returns:
            User ou None si non trouvé
        """
        return User.query.filter_by(email=email).filter(User.role == role).first()

    def find_by_email_with_role_filter(
        self, email: str, roles: tuple[UserRole, ...]
    ) -> User | None:
        """Trouve un utilisateur par email avec filtre sur les rôles.

        Args:
            email: Email de l'utilisateur
            roles: Tuple de rôles acceptés

        Returns:
            User ou None si non trouvé
        """
        user = User.query.filter(User.email == email).first()
        if user and user.role in roles:
            return user
        return None

    def find_by_email_with_driver_join(
        self, email: str, company_id: int, driver_type: Any
    ) -> User | None:
        """Trouve un utilisateur par email avec join sur Driver.

        Args:
            email: Email de l'utilisateur
            company_id: ID de l'entreprise
            driver_type: Type de driver

        Returns:
            User ou None si non trouvé
        """
        from models import Driver

        return (
            User.query.join(Driver, User.id == Driver.user_id)
            .filter(
                Driver.company_id == company_id,
                User.email == email,
                Driver.driver_type == driver_type,
            )
            .first()
        )

    def find_by_id_with_company(self, user_id: int) -> User | None:
        """Trouve un utilisateur par son ID avec eager loading de la relation company.

        Args:
            user_id: ID de l'utilisateur

        Returns:
            User ou None si non trouvé (avec company chargé)
        """
        from sqlalchemy.orm import joinedload

        return (
            User.query.options(joinedload(User.company))
            .filter_by(id=user_id)
            .one_or_none()
        )

    def count_all(self) -> int:
        """Compte tous les utilisateurs.

        Returns:
            Nombre total d'utilisateurs
        """
        return User.query.count()

    def find_all(self) -> list[User]:
        """Trouve tous les utilisateurs.

        Returns:
            Liste de tous les utilisateurs
        """
        return User.query.all()

    def find_recent(self, limit: int = 5) -> list[User]:
        """Trouve les utilisateurs récents.

        Args:
            limit: Nombre maximum d'utilisateurs à retourner (défaut: 5)

        Returns:
            Liste d'utilisateurs triés par created_at décroissant
        """
        return User.query.order_by(User.created_at.desc()).limit(limit).all()

    def find_by_id_with_clients_and_company(self, user_id: int) -> User | None:
        """Trouve un utilisateur par son ID avec eager loading de clients et company.

        Args:
            user_id: ID de l'utilisateur

        Returns:
            User ou None si non trouvé (avec clients et company chargés)
        """
        from sqlalchemy.orm import joinedload

        return (
            User.query.options(
                joinedload(User.clients),
                joinedload(User.company),
            )
            .filter_by(id=user_id)
            .one_or_none()
        )

    def find_by_id_with_driver_and_company(self, user_id: int) -> User | None:
        """Trouve un utilisateur par son ID avec eager loading de driver et company.

        Args:
            user_id: ID de l'utilisateur

        Returns:
            User ou None si non trouvé (avec driver et company chargés)
        """
        from sqlalchemy.orm import joinedload

        return (
            User.query.options(
                joinedload(User.driver),
                joinedload(User.company),
            )
            .filter_by(id=user_id)
            .one_or_none()
        )

    def find_by_public_id_first(self, public_id: str) -> User | None:
        """Trouve un utilisateur par son ID public (retourne le premier résultat).

        Args:
            public_id: ID public de l'utilisateur

        Returns:
            User ou None si non trouvé
        """
        return User.query.filter_by(public_id=public_id).first()

    def find_by_public_id_with_driver_and_company(self, public_id: str) -> User | None:
        """Trouve un utilisateur par son ID public avec eager loading de
        driver et company.

        Args:
            public_id: ID public de l'utilisateur

        Returns:
            User ou None si non trouvé (avec driver et company chargés)
        """
        from typing import cast

        from sqlalchemy.orm import joinedload

        return (
            User.query.options(
                joinedload(cast("Any", User.driver)),
                joinedload(cast("Any", User.company)),
            )
            .filter_by(public_id=public_id)
            .first()
        )

    def find_by_email_excluding_user(
        self, email: str, exclude_user_id: int
    ) -> User | None:
        """Trouve un utilisateur par email en excluant un user_id spécifique.

        Args:
            email: Email de l'utilisateur
            exclude_user_id: ID de l'utilisateur à exclure

        Returns:
            User ou None si non trouvé
        """
        return User.query.filter(
            User.email == email,
            User.id != exclude_user_id,
        ).first()

    def find_by_username(self, username: str) -> UserDTO | None:
        """Trouve un utilisateur par son username.

        Args:
            username: Username de l'utilisateur

        Returns:
            UserDTO ou None si non trouvé
        """
        user = User.query.filter_by(username=username).first()
        if user is None:
            return None
        return self._to_dto(user)

    # Méthodes legacy - retournent des modèles SQLAlchemy pour compatibilité
    def find_model_by_id(self, user_id: int) -> User | None:
        """Trouve un utilisateur par son ID (retourne le modèle SQLAlchemy).

        Args:
            user_id: ID de l'utilisateur

        Returns:
            User ou None si non trouvé (modèle SQLAlchemy)

        Note:
            Méthode legacy - utiliser find_by_id() pour obtenir un DTO
        """
        return User.query.get(user_id)

    def find_model_by_public_id(self, public_id: str) -> User | None:
        """Trouve un utilisateur par son ID public (retourne le modèle SQLAlchemy).

        Args:
            public_id: ID public de l'utilisateur

        Returns:
            User ou None si non trouvé (modèle SQLAlchemy)

        Note:
            Méthode legacy - utiliser find_by_public_id() pour obtenir un DTO
        """
        return User.query.filter_by(public_id=public_id).one_or_none()

    def find_model_by_email(self, email: str) -> User | None:
        """Trouve un utilisateur par son email (retourne le modèle SQLAlchemy).

        Args:
            email: Email de l'utilisateur

        Returns:
            User ou None si non trouvé (modèle SQLAlchemy)

        Note:
            Méthode legacy - utiliser find_by_email() pour obtenir un DTO
        """
        return User.query.filter_by(email=email).one_or_none()
