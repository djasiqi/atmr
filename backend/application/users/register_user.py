"""Use-case: enregistrer un nouvel utilisateur (register).

⚠️ TODO: Ce use case encapsule temporairement la logique d'inscription
dans routes/auth.py pour permettre une migration progressive.
La logique métier devrait être migrée progressivement
vers ce use case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RegisterUserInput:
    """Input pour enregistrer un nouvel utilisateur.

    Attributes:
        username: Nom d'utilisateur
        email: Email de l'utilisateur
        password: Mot de passe de l'utilisateur
        first_name: Prénom (optionnel)
        last_name: Nom (optionnel)
        phone: Téléphone (optionnel)
        address: Adresse (optionnel)
        birth_date: Date de naissance (optionnel)
        gender: Genre (optionnel)
        profile_image: Image de profil (optionnel)
    """

    username: str
    email: str
    password: str
    first_name: str | None = None
    last_name: str | None = None
    phone: str | None = None
    address: str | None = None
    birth_date: Any | None = None
    gender: str | None = None
    profile_image: str | None = None


@dataclass(frozen=True, slots=True)
class RegisterUserOutput:
    """Output pour enregistrer un nouvel utilisateur.

    Attributes:
        success: True si l'opération a réussi
        user_id: ID de l'utilisateur créé (si succès)
        user: Utilisateur créé (si succès)
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    user_id: int | None = None
    user: Any | None = None  # User model
    error: dict[str, str] | None = None
    status_code: int | None = None


class RegisterUserUseCase:
    """Use-case Application: enregistrer un nouvel utilisateur (register).

    ⚠️ TODO: Migration progressive - Ce use case encapsule la logique d'inscription.
    La logique métier devrait être migrée progressivement ici.
    """

    def execute(self, input_data: RegisterUserInput) -> RegisterUserOutput:
        """Enregistre un nouvel utilisateur.

        Args:
            input_data: Données d'entrée du use case

        Returns:
            RegisterUserOutput avec l'utilisateur créé

        Side-effects:
            - DB: Crée User et commit
        """
        import logging

        from models import User, db
        from models.enums import UserRole

        logger = logging.getLogger(__name__)

        # Validation
        validation_error = self._validate_input(input_data)
        if validation_error:
            return RegisterUserOutput(
                success=False,
                error=validation_error,
                status_code=400,
            )

        # Vérifier que l'email n'existe pas déjà
        from repositories.user_repository import UserRepository

        user_repo = UserRepository()
        existing_user = user_repo.find_by_email(input_data.email)
        if existing_user:
            return RegisterUserOutput(
                success=False,
                error={"error": "Un utilisateur avec cet email existe déjà"},
                status_code=400,
            )

        # Vérifier que le username n'existe pas déjà
        existing_user_by_username = user_repo.find_by_username(input_data.username)
        if existing_user_by_username:
            return RegisterUserOutput(
                success=False,
                error={"error": "Un utilisateur avec ce nom d'utilisateur existe déjà"},
                status_code=400,
            )

        try:
            # Créer l'utilisateur
            user = User()
            user.username = input_data.username
            user.email = input_data.email
            user.set_password(input_data.password)
            user.role = UserRole.CLIENT  # Par défaut, rôle client
            user.account_status = "pending_activation"

            if input_data.first_name:
                user.first_name = input_data.first_name
            if input_data.last_name:
                user.last_name = input_data.last_name
            if input_data.phone:
                user.phone = input_data.phone
            if input_data.address:
                user.address = input_data.address
            if input_data.birth_date:
                user.birth_date = input_data.birth_date
            if input_data.gender:
                from contextlib import suppress

                from models.enums import GenderEnum

                with suppress(ValueError, TypeError):
                    user.gender = GenderEnum(
                        input_data.gender
                    )  # Ignorer si le genre est invalide
            if input_data.profile_image:
                user.profile_image = input_data.profile_image

            db.session.add(user)
            db.session.commit()

            return RegisterUserOutput(success=True, user_id=user.id, user=user)
        except Exception:
            db.session.rollback()
            logger.exception("Erreur lors de l'inscription")
            return RegisterUserOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )

    def _validate_input(self, input_data: RegisterUserInput) -> dict[str, str] | None:
        """Valide les inputs du use case.

        Args:
            input_data: Input à valider

        Returns:
            None si valide, dict d'erreurs sinon
        """
        errors: dict[str, str] = {}

        if not input_data.username or len(input_data.username.strip()) == 0:
            errors["username"] = "Le nom d'utilisateur est requis"

        if not input_data.email or len(input_data.email.strip()) == 0:
            errors["email"] = "L'email est requis"

        if not input_data.password or len(input_data.password.strip()) == 0:
            errors["password"] = "Le mot de passe est requis"

        return errors if errors else None
