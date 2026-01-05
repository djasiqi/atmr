"""Repository pour l'accès aux données Company avec conversion DTO.

Ce repository découple les services de l'implémentation SQLAlchemy
en retournant des DTOs au lieu de modèles SQLAlchemy directs.
"""

from domain.company_dto import CompanyDTO
from models import Company
from models.enums import DispatchMode

logger = __import__("logging").getLogger(__name__)


class CompanyRepository:
    """Repository pour l'accès aux données Company avec conversion DTO."""

    def _to_dto(self, company: Company) -> CompanyDTO:
        """Convertit un modèle Company SQLAlchemy en DTO.

        Args:
            company: Modèle Company SQLAlchemy

        Returns:
            CompanyDTO correspondant
        """
        return CompanyDTO(
            id=company.id,
            user_id=company.user_id,  # type: ignore[reportGeneralTypeIssues]
            name=company.name,
            address=company.address,
            latitude=company.latitude,
            longitude=company.longitude,
            contact_email=company.contact_email,
            contact_phone=company.contact_phone,
            uid_ide=company.uid_ide,
            billing_email=company.billing_email,
            billing_notes=company.billing_notes,
            is_approved=company.is_approved,
            dispatch_enabled=company.dispatch_enabled,
            dispatch_mode=company.dispatch_mode
            if hasattr(company, "dispatch_mode")
            else DispatchMode.SEMI_AUTO,
            autonomous_config=getattr(company, "autonomous_config", None),
            max_daily_bookings=company.max_daily_bookings,
            service_area=company.service_area,
            created_at=company.created_at,  # type: ignore[reportGeneralTypeIssues]
            accepted_at=company.accepted_at,
            is_partner=getattr(company, "is_partner", False),
            logo_url=company.logo_url,
        )

    def find_model_by_id(self, company_id: int) -> Company | None:
        """Trouve une entreprise par son ID (retourne le modèle SQLAlchemy).

        Args:
            company_id: ID de l'entreprise

        Returns:
            Company ou None si non trouvé
        """
        return Company.query.get(company_id)

    def find_by_id(self, company_id: int) -> CompanyDTO | None:
        """Trouve une entreprise par son ID.

        Args:
            company_id: ID de l'entreprise

        Returns:
            CompanyDTO ou None si non trouvé
        """
        company = Company.query.get(company_id)
        if company is None:
            return None
        return self._to_dto(company)

    def find_by_user_id(self, user_id: int) -> CompanyDTO | None:
        """Trouve une entreprise par l'ID de son utilisateur.

        Args:
            user_id: ID de l'utilisateur

        Returns:
            CompanyDTO ou None si non trouvé
        """
        company = Company.query.filter_by(user_id=user_id).first()
        if company is None:
            return None
        return self._to_dto(company)

    def find_first_model(self) -> Company | None:
        """Trouve la première entreprise (retourne le modèle SQLAlchemy).

        Returns:
            Company ou None si aucune entreprise trouvée
        """
        return Company.query.first()

    def find_model_by_user_id(self, user_id: int) -> Company | None:
        """Trouve une entreprise par l'ID de son utilisateur (retourne le modèle SQLAlchemy).

        Args:
            user_id: ID de l'utilisateur

        Returns:
            Company ou None si non trouvé
        """
        return Company.query.filter_by(user_id=user_id).first()

    def find_all_models_ordered_by_name(self) -> list[Company]:
        """Trouve toutes les entreprises triées par nom (retourne les modèles SQLAlchemy).

        Returns:
            Liste de Company triées par nom ascendant
        """
        return Company.query.order_by(Company.name.asc()).all()
