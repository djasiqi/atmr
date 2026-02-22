"""Service centralisé pour la gestion de l'autorisation.

Ce service centralise toutes les vérifications d'autorisation pour améliorer
la maintenabilité et la cohérence du code.

✅ S2: Centralisation de la logique d'autorisation
"""

import logging
from typing import Literal, Tuple

from flask import abort
from flask_jwt_extended import get_jwt_identity

from ext import db
from models import Booking, Client, Company, Driver, Institution, User, UserRole

logger = logging.getLogger(__name__)


class AuthorizationService:
    """Service centralisé pour la gestion de l'autorisation."""

    @staticmethod
    def get_current_user() -> User | None:
        """Récupère l'utilisateur actuel depuis le token JWT.

        Returns:
            User si trouvé, None sinon
        """
        user_public_id = get_jwt_identity()
        if not user_public_id:
            return None

        user = User.query.filter_by(public_id=user_public_id).first()
        if not user:
            logger.warning(
                "[Authorization] Utilisateur non trouvé pour public_id: %s",
                user_public_id,
            )
        return user

    @staticmethod
    def require_user() -> User:
        """Récupère l'utilisateur actuel et lève une exception si non trouvé.

        Returns:
            User actuel

        Raises:
            HTTPException: 404 si l'utilisateur n'est pas trouvé
        """
        user = AuthorizationService.get_current_user()
        if not user:
            logger.warning(
                "[Authorization] Tentative d'accès sans utilisateur authentifié"
            )
            abort(404, description="Utilisateur non trouvé")
        return user

    @staticmethod
    def require_role(*allowed_roles: UserRole | str) -> User:
        """Vérifie que l'utilisateur actuel a l'un des rôles autorisés.

        Args:
            *allowed_roles: Rôles autorisés (UserRole ou str)

        Returns:
            User actuel

        Raises:
            HTTPException: 404 si l'utilisateur n'est pas trouvé
            HTTPException: 403 si l'utilisateur n'a pas le rôle requis
        """
        user = AuthorizationService.require_user()

        # Normaliser les rôles (convertir str en UserRole si nécessaire)
        normalized_roles: list[UserRole] = []
        for role in allowed_roles:
            # Type narrowing: role peut être UserRole ou str
            if hasattr(role, "value"):  # UserRole a un attribut value
                normalized_roles.append(role)  # type: ignore[arg-type]
            else:  # C'est un str
                try:
                    normalized_roles.append(UserRole[str(role).upper()])
                except KeyError:
                    logger.warning(
                        "[Authorization] Rôle invalide dans la configuration : %s", role
                    )

        if user.role not in normalized_roles:
            roles_str = ", ".join([r.value for r in normalized_roles])
            msg = (
                "[Authorization] ⛔ Accès refusé : %s (%s) a tenté d'accéder "
                "à une route restreinte (rôles autorisés: %s)"
            )
            logger.warning(msg, user.username, user.role, roles_str)
            abort(403, description="Accès non autorisé")

        return user

    @staticmethod
    def require_company() -> Tuple[Company, User]:
        """Récupère la company de l'utilisateur actuel.

        Returns:
            Tuple (Company, User)

        Raises:
            HTTPException: 404 si l'utilisateur ou la company n'est pas trouvée
        """
        from sqlalchemy.orm import joinedload

        user = AuthorizationService.require_role(UserRole.company)

        # Charger la company avec eager loading
        user_with_company = (
            User.query.options(joinedload(User.company))
            .filter_by(public_id=user.public_id)
            .first()
        )

        if not user_with_company:
            logger.error("[Authorization] User disappeared after role check")
            abort(404, description="Utilisateur non trouvé")

        company: Company | None = getattr(user_with_company, "company", None)

        # Si l'utilisateur est de rôle company mais n'a pas encore d'objet Company,
        # on le crée automatiquement (compatibilité avec get_company_from_token)
        if company is None:
            msg = (
                "[Authorization] ⚠️ Aucun objet Company associé à l'utilisateur %s - "
                "tentative de création"
            )
            logger.warning(msg, user.username)
            try:
                # Créer l'instance Company et assigner les attributs
                company = Company()
                company.name = user.username or "Company"
                company.user_id = user.id
                company.address = ""
                company.latitude = None
                company.longitude = None
                company.contact_email = user.email
                company.contact_phone = ""
                company.service_area = ""
                company.max_daily_bookings = 50
                company.is_approved = False
                db.session.add(company)
                db.session.commit()
                logger.info(
                    "[Authorization] ✅ Company créée automatiquement pour user %s",
                    user.username,
                )
            except Exception as e:
                logger.exception(
                    "[Authorization] ❌ Erreur lors de la création automatique de Company : %s",
                    e,
                )
                db.session.rollback()
                abort(500, description="Failed to create default company")

        # company ne peut plus être None ici (soit existait déjà, soit créé ci-dessus)
        assert company is not None, "Company should not be None at this point"

        return company, user

    @staticmethod
    def require_driver() -> Tuple[Driver, User]:
        """Récupère le driver de l'utilisateur actuel.

        Returns:
            Tuple (Driver, User)

        Raises:
            HTTPException: 404 si l'utilisateur ou le driver n'est pas trouvé
        """
        from sqlalchemy.orm import joinedload

        user = AuthorizationService.require_user()

        # Charger le driver avec eager loading
        user_with_driver = (
            User.query.options(joinedload(User.driver))
            .filter_by(public_id=user.public_id)
            .first()
        )

        if not user_with_driver:
            logger.error("[Authorization] User disappeared after authentication")
            abort(404, description="Utilisateur non trouvé")

        driver: Driver | None = getattr(user_with_driver, "driver", None)

        if driver is None:
            logger.warning(
                "[Authorization] ⚠️ Aucun objet Driver associé à l'utilisateur %s",
                user.username,
            )
            abort(404, description="No driver associated with this user.")

        return driver, user

    @staticmethod
    def require_client() -> Tuple[Client, User]:
        """Récupère le client de l'utilisateur actuel.

        Returns:
            Tuple (Client, User)

        Raises:
            HTTPException: 404 si l'utilisateur ou le client n'est pas trouvé
        """
        from sqlalchemy.orm import joinedload

        user = AuthorizationService.require_role(UserRole.client)

        # Charger le client avec eager loading
        user_with_client = (
            User.query.options(joinedload(User.client))
            .filter_by(public_id=user.public_id)
            .first()
        )

        if not user_with_client:
            logger.error("[Authorization] User disappeared after role check")
            abort(404, description="Utilisateur non trouvé")

        client: Client | None = getattr(user_with_client, "client", None)

        if client is None:
            logger.warning(
                "[Authorization] ⚠️ Aucun objet Client associé à l'utilisateur %s",
                user.username,
            )
            abort(404, description="No client associated with this user.")

        return client, user

    @staticmethod
    def require_institution() -> Tuple["Institution", User]:
        """Récupère l'institution de l'utilisateur actuel.

        Returns:
            Tuple (Institution, User)

        Raises:
            HTTPException: 404 si l'utilisateur ou l'institution n'est pas trouvée
            HTTPException: 403 si l'utilisateur n'a pas de claim institution_id
        """
        from flask_jwt_extended import get_jwt

        user = AuthorizationService.require_role(UserRole.INSTITUTION)

        # Récupérer institution_id depuis le JWT claim
        jwt_claims = get_jwt()
        institution_id = jwt_claims.get("institution_id")

        if not institution_id:
            logger.warning(
                "[Authorization] ⚠️ User %s has institution role but no institution_id claim",
                user.username,
            )
            abort(403, description="No institution_id claim in token")

        # Charger l'institution
        institution = Institution.query.get(institution_id)

        if institution is None:
            logger.warning(
                "[Authorization] ⚠️ Institution %s not found for user %s",
                institution_id,
                user.username,
            )
            abort(404, description="Institution not found")

        return institution, user

    @staticmethod
    def get_institution_role_from_jwt() -> str | None:
        """Récupère le rôle institution depuis le JWT claim.

        Returns:
            Le rôle institution (institution_admin, institution_requester, etc.) ou None
        """
        from flask_jwt_extended import get_jwt

        try:
            jwt_claims = get_jwt()
            return jwt_claims.get("institution_role")
        except Exception:
            return None

    @staticmethod
    def require_institution_role(*allowed_roles: str) -> Tuple["Institution", User]:
        """Vérifie que l'utilisateur a un des rôles institution autorisés.

        Args:
            *allowed_roles: Rôles autorisés (institution_admin, institution_requester, etc.)

        Returns:
            Tuple (Institution, User)

        Raises:
            HTTPException: 403 si le rôle institution n'est pas autorisé
        """
        institution, user = AuthorizationService.require_institution()

        institution_role = AuthorizationService.get_institution_role_from_jwt()

        if not institution_role or institution_role not in allowed_roles:
            roles_str = ", ".join(allowed_roles)
            msg = (
                "[Authorization] ⛔ Accès refusé : %s (institution_role=%s) a tenté d'accéder "
                "à une route restreinte (rôles autorisés: %s)"
            )
            logger.warning(msg, user.username, institution_role, roles_str)
            abort(403, description="Institution role not authorized")

        return institution, user

    @staticmethod
    def check_institution_resource_access(
        resource_institution_id: int | None,
        user: User,
    ) -> Tuple[bool, Tuple[dict[str, str], int] | None]:
        """Vérifie si l'utilisateur a accès à une ressource appartenant à une institution.

        Args:
            resource_institution_id: ID de l'institution propriétaire de la ressource
            user: L'utilisateur authentifié

        Returns:
            (has_access: bool, error_response_tuple_or_none)
        """
        from flask_jwt_extended import get_jwt

        if not resource_institution_id:
            return False, ({"error": "Ressource sans institution_id"}, 400)

        # Admin a tous les droits
        if user.role == UserRole.ADMIN:
            return True, None

        # Institution doit être la propriétaire
        if user.role == UserRole.INSTITUTION:
            jwt_claims = get_jwt()
            user_institution_id = jwt_claims.get("institution_id")
            if user_institution_id and user_institution_id == resource_institution_id:
                return True, None
            # IDOR attempt détecté
            msg = (
                "[Authorization] ⚠️ IDOR attempt: Institution user %s (institution_id=%s) "
                "a tenté d'accéder à une ressource de institution_id=%s"
            )
            logger.warning(msg, user.public_id, user_institution_id, resource_institution_id)
            return False, ({"error": "Accès non autorisé à cette ressource"}, 403)

        return False, ({"error": "Accès non autorisé"}, 403)

    @staticmethod
    def check_booking_ownership(  # noqa: PLR0911
        booking: Booking,
        user: User,
        action: Literal["read", "modify", "delete"] = "read",
    ) -> Tuple[bool, Tuple[dict[str, str], int] | None]:
        """Vérifie si l'utilisateur a le droit d'accéder/modifier ce booking.

        Args:
            booking: Le booking à vérifier
            user: L'utilisateur authentifié
            action: Type d'action ("read", "modify", "delete")

        Returns:
            (has_access: bool, error_response_tuple_or_none)

        Exemple:
            has_access, error = AuthorizationService.check_booking_ownership(booking, user, "modify")
            if not has_access:
                return error  # ({"error": "..."}, 403)
        """
        user_role_value = str(getattr(user.role, "value", user.role))
        error_response = ({"error": f"Accès non autorisé ({action})"}, 403)

        # Admin a tous les droits
        if user_role_value == UserRole.admin.value:
            return True, None

        # Company a accès à tous ses bookings
        if user_role_value == UserRole.company.value:
            company = Company.query.filter_by(user_id=user.id).first()
            has_access = company is not None and company.id == booking.company_id
            if has_access:
                return True, None
            # IDOR attempt détecté
            company_id_str = str(company.id) if company else "None"
            msg = (
                "[Authorization] ⚠️ IDOR attempt: Company %s (user %s) a tenté d'accéder "
                "au booking %s (company_id=%s)"
            )
            logger.warning(
                msg, company_id_str, user.public_id, booking.id, booking.company_id
            )
            return False, error_response

        # Client propriétaire
        if user_role_value == UserRole.client.value:
            client = Client.query.filter_by(user_id=user.id).first()
            if not client:
                logger.warning(
                    "[Authorization] ⚠️ User %s has client role but no Client record",
                    user.public_id,
                )
                return False, error_response
            if client.id == booking.client_id:
                return True, None
            # IDOR attempt détecté
            msg = (
                "[Authorization] ⚠️ IDOR attempt: Client %s (user %s) a tenté d'accéder "
                "au booking %s (client_id=%s)"
            )
            logger.warning(
                msg, str(client.id), user.public_id, booking.id, booking.client_id
            )
            return False, error_response

        # Driver (peut voir les bookings assignés)
        if user_role_value == UserRole.driver.value:
            driver = Driver.query.filter_by(user_id=user.id).first()
            if not driver:
                logger.warning(
                    "[Authorization] ⚠️ User %s has driver role but no Driver record",
                    user.public_id,
                )
                return False, error_response

            # Vérifier si le booking est assigné à ce driver
            from models import Assignment

            assignment = Assignment.query.filter_by(
                driver_id=driver.id, booking_id=booking.id
            ).first()
            if assignment:
                return True, None

            # IDOR attempt détecté
            msg = (
                "[Authorization] ⚠️ IDOR attempt: Driver %s (user %s) a tenté d'accéder "
                "au booking %s (non assigné)"
            )
            logger.warning(msg, str(driver.id), user.public_id, booking.id)
            return False, error_response

        # Rôle non reconnu
        logger.warning(
            "[Authorization] ⚠️ Rôle non reconnu pour vérification ownership: %s",
            user_role_value,
        )
        return False, error_response

    @staticmethod
    def require_booking_ownership(
        booking: Booking,
        user: User,
        action: Literal["read", "modify", "delete"] = "read",
    ) -> None:
        """Vérifie que l'utilisateur a le droit d'accéder/modifier ce booking.
        Lève une exception si l'accès est refusé.

        Args:
            booking: Le booking à vérifier
            user: L'utilisateur authentifié
            action: Type d'action ("read", "modify", "delete")

        Raises:
            HTTPException: 403 si l'accès est refusé
        """
        has_access, error_response = AuthorizationService.check_booking_ownership(
            booking, user, action
        )
        if not has_access:
            if error_response:
                error_dict, status_code = error_response
                abort(
                    status_code,
                    description=error_dict.get("error", "Accès non autorisé"),
                )
            abort(403, description="Accès non autorisé")

    @staticmethod
    def check_company_resource_access(
        resource_company_id: int | None,
        user: User,
    ) -> Tuple[bool, Tuple[dict[str, str], int] | None]:
        """Vérifie si l'utilisateur a accès à une ressource appartenant à une company.

        Args:
            resource_company_id: ID de la company propriétaire de la ressource
            user: L'utilisateur authentifié

        Returns:
            (has_access: bool, error_response_tuple_or_none)
        """
        if not resource_company_id:
            return False, ({"error": "Ressource sans company_id"}, 400)

        # Admin a tous les droits
        if user.role == UserRole.admin:
            return True, None

        # Company doit être la propriétaire
        if user.role == UserRole.company:
            company = Company.query.filter_by(user_id=user.id).first()
            if company and company.id == resource_company_id:
                return True, None
            # IDOR attempt détecté
            company_id_str = str(company.id) if company else "None"
            msg = (
                "[Authorization] ⚠️ IDOR attempt: Company %s (user %s) a tenté d'accéder "
                "à une ressource de company_id=%s"
            )
            logger.warning(msg, company_id_str, user.public_id, resource_company_id)
            return False, ({"error": "Accès non autorisé à cette ressource"}, 403)

        return False, ({"error": "Accès non autorisé"}, 403)

    @staticmethod
    def require_company_resource_access(
        resource_company_id: int | None,
        user: User,
    ) -> None:
        """Vérifie que l'utilisateur a accès à une ressource appartenant à une company.
        Lève une exception si l'accès est refusé.

        Args:
            resource_company_id: ID de la company propriétaire de la ressource
            user: L'utilisateur authentifié

        Raises:
            HTTPException: 403 si l'accès est refusé
        """
        has_access, error_response = AuthorizationService.check_company_resource_access(
            resource_company_id, user
        )
        if not has_access:
            if error_response:
                error_dict, status_code = error_response
                abort(
                    status_code,
                    description=error_dict.get("error", "Accès non autorisé"),
                )
            abort(403, description="Accès non autorisé")


# ── Helpers Curatelle ─────────────────────────────────────────────────────


def get_user_team_ids(user_id: int) -> list[int]:
    """Retourne les IDs des équipes de curateurs auxquelles l'utilisateur appartient."""
    from models.curator_team import CuratorTeamMember

    return [m.team_id for m in CuratorTeamMember.query.filter_by(user_id=user_id).all()]


# Instance globale du service (singleton)
_authorization_service = AuthorizationService()


def get_authorization_service() -> AuthorizationService:
    """Récupère l'instance du service d'autorisation.

    Returns:
        Instance du service d'autorisation
    """
    return _authorization_service
