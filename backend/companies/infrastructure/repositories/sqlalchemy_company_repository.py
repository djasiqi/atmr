"""Implémentation SQLAlchemy du repository Company."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from companies.domain.company import Company
from companies.domain.company_id import CompanyId
from companies.domain.value_objects import (
    BillingSettings,
    CompanySettings,
    DispatchMode,
    PlanningSettings,
)

if TYPE_CHECKING:
    from models import Company as SQLAlchemyCompany
else:
    SQLAlchemyCompany = Any

logger = __import__("logging").getLogger(__name__)


class SqlAlchemyCompanyRepository:
    """Implémentation SQLAlchemy du repository Company.

    Adapte les modèles SQLAlchemy vers les agrégats du domaine.
    """

    def _to_aggregate(self, sa_company: SQLAlchemyCompany) -> Company:
        """Convertit un modèle SQLAlchemy en agrégat Company."""
        # Construire PlanningSettings
        planning_settings = PlanningSettings(
            max_daily_bookings=sa_company.max_daily_bookings,
            service_area=sa_company.service_area,
        )

        # Construire BillingSettings
        billing_settings = BillingSettings(
            billing_email=sa_company.billing_email,
            billing_notes=sa_company.billing_notes,
            iban=cast(
                str | None,
                sa_company._iban_raw if hasattr(sa_company, "_iban_raw") else None,
            ),
        )

        # Construire DispatchMode
        dispatch_mode = DispatchMode(str(sa_company.dispatch_mode.value))

        # Construire CompanySettings
        settings = CompanySettings(
            dispatch_enabled=bool(sa_company.dispatch_enabled),
            dispatch_mode=dispatch_mode,
            planning_settings=planning_settings,
            billing_settings=billing_settings,
        )

        return Company(
            id=CompanyId(sa_company.id),
            name=sa_company.name,
            user_id=cast(int, sa_company.user_id),
            settings=settings,
            is_approved=bool(sa_company.is_approved),
            is_partner=bool(sa_company.is_partner),
            uid_ide=sa_company.uid_ide,
            logo_url=sa_company.logo_url,
            created_at=cast(
                Any | None,
                sa_company.created_at if hasattr(sa_company, "created_at") else None,
            ),
            accepted_at=sa_company.accepted_at,
        )

    def _from_aggregate(self, company: Company) -> dict[str, Any]:
        """Convertit un agrégat Company en dictionnaire pour SQLAlchemy."""
        data: dict[str, Any] = {
            "id": company.id.value,
            "name": company.name,
            "user_id": company.user_id,
            "is_approved": company.is_approved,
            "is_partner": company.is_partner,
            "uid_ide": company.uid_ide,
            "logo_url": company.logo_url,
            "accepted_at": company.accepted_at,
            "dispatch_enabled": company.settings.dispatch_enabled,
            "dispatch_mode": company.settings.dispatch_mode.value,
            "max_daily_bookings": company.settings.planning_settings.max_daily_bookings,
            "service_area": company.settings.planning_settings.service_area,
            "billing_email": company.settings.billing_settings.billing_email,
            "billing_notes": company.settings.billing_settings.billing_notes,
        }

        # IBAN est stocké dans _iban_raw (chiffré)
        if company.settings.billing_settings.iban:
            data["_iban_raw"] = company.settings.billing_settings.iban

        return data

    def save(self, company: Company) -> None:
        """Sauvegarde une entreprise."""
        from ext import db
        from models import Company as SQLAlchemyCompany
        from models.enums import DispatchMode as SQLAlchemyDispatchMode

        data = self._from_aggregate(company)
        company_id = data.pop("id")

        sa_company = SQLAlchemyCompany.query.get(company_id)
        if sa_company:
            # Update
            for key, value in data.items():
                if key == "dispatch_mode":
                    # Convertir string en enum
                    mode_enum = SQLAlchemyDispatchMode(value)
                    setattr(sa_company, key, mode_enum)
                else:
                    setattr(sa_company, key, value)
        else:
            # Create
            # Convertir dispatch_mode en enum
            mode_enum = SQLAlchemyDispatchMode(data["dispatch_mode"])
            data["dispatch_mode"] = mode_enum
            sa_company = SQLAlchemyCompany(**data)
            db.session.add(sa_company)

        db.session.commit()

    def find_by_id(self, company_id: CompanyId) -> Company | None:
        """Trouve une entreprise par ID."""
        from models import Company as SQLAlchemyCompany

        sa_company = SQLAlchemyCompany.query.get(company_id.value)
        if sa_company is None:
            return None
        return self._to_aggregate(sa_company)

    def find_by_user_id(self, user_id: int) -> Company | None:
        """Trouve une entreprise par user_id."""
        from models import Company as SQLAlchemyCompany

        sa_company = SQLAlchemyCompany.query.filter_by(user_id=user_id).first()
        if sa_company is None:
            return None
        return self._to_aggregate(sa_company)

    def find_model_by_user_id(self, user_id: int) -> Any | None:
        """Trouve une entreprise SQLAlchemy par user_id (compatibilité)."""
        from models import Company as SQLAlchemyCompany

        return SQLAlchemyCompany.query.filter_by(user_id=user_id).first()
