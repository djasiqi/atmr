"""Provisionne la fiche chauffeur liée au compte entreprise (double casquette mobile)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError

from ext import db
from models import Driver, User
from models.enums import DriverType, UserRole
from repositories.driver_repository import DriverRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EnsureCompanyOperatorDriverResult:
    driver: Driver | None
    created: bool = False


class EnsureCompanyOperatorDriverUseCase:
    """Crée un profil chauffeur sur le même user que l'entreprise si absent.

    Cible : comptes entreprise transport (manuel, semi-auto ou fully-auto) qui basculent
    entreprise ↔ chauffeur dans l'app unifiée via ``/auth/bootstrap`` et
    ``/auth/switch-context``. Indépendant de ``dispatch_enabled`` (moteur d'assignation
    auto) et de ``dispatch_mode``.
    """

    def execute(self, user: User) -> EnsureCompanyOperatorDriverResult:
        if user.role is not UserRole.COMPANY:
            return EnsureCompanyOperatorDriverResult(driver=None, created=False)

        company = getattr(user, "company", None)
        if company is None or company.id is None:
            return EnsureCompanyOperatorDriverResult(driver=None, created=False)

        existing = getattr(user, "driver", None)
        if existing is None:
            existing = DriverRepository().find_model_by_user_id(int(user.id))
        if existing is not None:
            return EnsureCompanyOperatorDriverResult(driver=existing, created=False)

        # Attribution par attributs : les Column SQLAlchemy ne sont pas dans __init__ typé
        driver = Driver()
        driver.user_id = int(user.id)
        driver.company_id = int(company.id)
        driver.is_active = True
        driver.is_available = True
        driver.driver_type = DriverType.REGULAR
        try:
            with db.session.begin_nested():
                db.session.add(driver)
                db.session.flush()
        except IntegrityError:
            logger.info(
                "company operator driver already provisioned (race)",
                extra={"user_id": user.id, "company_id": company.id},
            )
            raced = DriverRepository().find_model_by_user_id(int(user.id))
            return EnsureCompanyOperatorDriverResult(driver=raced, created=False)

        logger.info(
            "company operator driver provisioned",
            extra={
                "user_id": user.id,
                "company_id": company.id,
                "driver_id": driver.id,
            },
        )
        return EnsureCompanyOperatorDriverResult(driver=driver, created=True)
