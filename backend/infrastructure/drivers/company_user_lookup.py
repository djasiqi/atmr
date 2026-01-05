from __future__ import annotations

from typing import Any, cast

from models import User, UserRole
from repositories.user_repository import UserRepository


def find_company_user_for_driver(driver: Any, company: Any) -> User | None:
    """Adapter Infrastructure: retrouver l'utilisateur entreprise lié à un driver.

    Retourne le modèle User SQLAlchemy pour compatibilité avec le Protocol _UserLike.
    """

    user_repo = UserRepository()

    # Utiliser find_model_by_id pour obtenir le modèle User au lieu du DTO
    company_user = user_repo.find_model_by_id(user_id=company.user_id)

    if not company_user or company_user.role != UserRole.COMPANY:
        driver_user = user_repo.find_model_by_id(user_id=driver.user_id)
        if driver_user and bool(driver_user.email):
            company_user = user_repo.find_by_email_and_role(
                email=cast(str, driver_user.email), role=UserRole.COMPANY
            )

    if not company_user:
        company_user = user_repo.find_model_by_id(user_id=company.user_id)

    if not company_user or company_user.role != UserRole.COMPANY:
        return None

    return company_user
