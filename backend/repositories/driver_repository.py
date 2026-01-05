"""Repository pour l'accès aux données Driver avec conversion DTO.

Ce repository découple les services de l'implémentation SQLAlchemy
en retournant des DTOs au lieu de modèles SQLAlchemy directs.
"""

from typing import Any, cast

from domain.driver_dto import DriverDTO
from models import Driver

logger = __import__("logging").getLogger(__name__)


class DriverRepository:
    """Repository pour l'accès aux données Driver avec conversion DTO."""

    def _to_dto(self, driver: Driver) -> DriverDTO:
        """Convertit un modèle Driver SQLAlchemy en DTO.

        Args:
            driver: Modèle Driver SQLAlchemy

        Returns:
            DriverDTO correspondant
        """
        return DriverDTO(
            id=driver.id,
            user_id=cast(int, driver.user_id),
            company_id=cast(int, driver.company_id),
            vehicle_assigned=driver.vehicle_assigned,
            brand=driver.brand,
            license_plate=driver.license_plate,
            is_active=getattr(driver, "is_active", True),
            is_available=getattr(driver, "is_available", True),
            driver_type=driver.driver_type,
            latitude=driver.latitude,
            longitude=driver.longitude,
            last_position_update=driver.last_position_update,
            driver_photo=driver.driver_photo,
            push_token=driver.push_token,
            contract_type=getattr(driver, "contract_type", "CDI"),
            weekly_hours=driver.weekly_hours,
            hourly_rate_cents=driver.hourly_rate_cents,
            employment_start_date=driver.employment_start_date,
            employment_end_date=driver.employment_end_date,
            license_categories=(
                list(cast(Any, driver.license_categories))
                if bool(driver.license_categories)
                else None
            ),
            license_valid_until=driver.license_valid_until,
            trainings=(
                list(cast(Any, driver.trainings))
                if bool(driver.trainings)
                else None
            ),
            medical_valid_until=driver.medical_valid_until,
        )

    def find_by_id(self, driver_id: int) -> DriverDTO | None:
        """Trouve un driver par son ID.

        Args:
            driver_id: ID du driver

        Returns:
            DriverDTO ou None si non trouvé
        """
        driver = Driver.query.get(driver_id)
        if driver is None:
            return None
        return self._to_dto(driver)

    def find_by_company_id(
        self, company_id: int, active_only: bool = True
    ) -> list[DriverDTO]:
        """Trouve les drivers d'une entreprise.

        Args:
            company_id: ID de l'entreprise
            active_only: Si True, ne retourne que les drivers actifs

        Returns:
            Liste de DriverDTO
        """
        query = Driver.query.filter_by(company_id=company_id)
        if active_only:
            query = query.filter_by(is_active=True)
        drivers = query.all()
        return [self._to_dto(d) for d in drivers]

    def find_available_by_company_id(self, company_id: int) -> list[DriverDTO]:
        """Trouve les drivers disponibles d'une entreprise.

        Args:
            company_id: ID de l'entreprise

        Returns:
            Liste de DriverDTO disponibles
        """
        drivers = Driver.query.filter_by(
            company_id=company_id, is_active=True, is_available=True
        ).all()
        return [self._to_dto(d) for d in drivers]

    def find_by_ids(self, driver_ids: list[int]) -> list[DriverDTO]:
        """Trouve les drivers par leurs IDs.

        Args:
            driver_ids: Liste d'IDs de drivers

        Returns:
            Liste de DriverDTO correspondants
        """
        if not driver_ids:
            return []
        drivers = Driver.query.filter(Driver.id.in_(driver_ids)).all()
        return [self._to_dto(d) for d in drivers]

    def find_model_by_id(self, driver_id: int) -> Driver | None:
        """Trouve un driver par son ID (retourne le modèle SQLAlchemy).

        Args:
            driver_id: ID du driver

        Returns:
            Driver ou None si non trouvé
        """
        return Driver.query.get(driver_id)

    def find_model_by_id_and_company(
        self, driver_id: int, company_id: int
    ) -> Driver | None:
        """Trouve un driver par son ID et company_id (retourne le modèle SQLAlchemy).

        Args:
            driver_id: ID du driver
            company_id: ID de l'entreprise

        Returns:
            Driver ou None si non trouvé
        """
        return Driver.query.filter_by(id=driver_id, company_id=company_id).one_or_none()

    def find_model_by_id_with_user(
        self, driver_id: int, company_id: int
    ) -> Driver | None:
        """Trouve un driver par son ID avec eager loading de user (retourne le modèle SQLAlchemy).

        Args:
            driver_id: ID du driver
            company_id: ID de l'entreprise

        Returns:
            Driver ou None si non trouvé (avec user chargé)
        """
        from sqlalchemy.orm import joinedload

        return (
            Driver.query.options(joinedload(Driver.user))
            .filter_by(id=driver_id, company_id=company_id)
            .one_or_none()
        )

    def find_models_by_ids_with_user_and_vacations(
        self, driver_ids: list[int]
    ) -> list[Driver]:
        """Trouve les drivers par leurs IDs avec eager loading de user et vacations.

        Args:
            driver_ids: Liste d'IDs de drivers

        Returns:
            Liste de Driver avec user et vacations chargés
        """
        from sqlalchemy.orm import joinedload

        return (
            Driver.query.options(joinedload(Driver.user), joinedload(Driver.vacations))
            .filter(Driver.id.in_(driver_ids))
            .all()
        )

    def find_models_by_ids_with_user_eager_loading(
        self, driver_ids: list[int]
    ) -> list[Driver]:
        """Trouve les drivers par IDs avec eager loading de user (retourne les modèles SQLAlchemy).

        Args:
            driver_ids: Liste d'IDs de drivers

        Returns:
            Liste de Driver avec user chargé
        """
        from sqlalchemy.orm import joinedload

        if not driver_ids:
            return []
        return (
            Driver.query.filter(Driver.id.in_(driver_ids))
            .options(joinedload(Driver.user))
            .all()
        )

    def find_models_by_company_available_with_user_eager_loading(
        self, company_id: int
    ) -> list[Driver]:
        """Trouve les drivers disponibles d'une entreprise avec eager loading de user.

        Args:
            company_id: ID de l'entreprise

        Returns:
            Liste de Driver avec user chargé, triés par driver_type desc
        """
        from sqlalchemy.orm import joinedload

        return (
            Driver.query.options(joinedload(Driver.user))
            .filter(
                Driver.company_id == company_id,
                Driver.is_available == True,  # noqa: E712
            )
            .order_by(Driver.driver_type.desc())
            .all()
        )

    def find_models_by_company_active_available_with_user_eager_loading(
        self, company_id: int
    ) -> list[Driver]:
        """Trouve les drivers actifs et disponibles d'une entreprise avec eager loading de user.

        Args:
            company_id: ID de l'entreprise

        Returns:
            Liste de Driver avec user chargé
        """
        from sqlalchemy.orm import joinedload

        return (
            Driver.query.options(joinedload(Driver.user))
            .filter(
                Driver.company_id == company_id,
                Driver.is_active.is_(True),
                Driver.is_available.is_(True),
            )
            .all()
        )

    def find_models_by_company_available_with_user_eager_loading_limited(
        self, company_id: int, limit: int = 10
    ) -> list[Driver]:
        """Trouve les drivers disponibles d'une entreprise avec eager loading, limités.

        Args:
            company_id: ID de l'entreprise
            limit: Nombre maximum de résultats (défaut: 10)

        Returns:
            Liste de Driver avec user chargé, triés par driver_type desc
        """
        from sqlalchemy.orm import joinedload

        return (
            Driver.query.options(joinedload(Driver.user))
            .filter(
                Driver.company_id == company_id,
                Driver.is_available == True,  # noqa: E712
            )
            .order_by(Driver.driver_type.desc())
            .limit(limit)
            .all()
        )

    def find_model_by_user_id(self, user_id: int) -> Driver | None:
        """Trouve un driver par user_id (retourne le modèle SQLAlchemy).

        Args:
            user_id: ID de l'utilisateur

        Returns:
            Driver ou None si non trouvé
        """
        return Driver.query.filter_by(user_id=user_id).first()

    def find_model_by_company_and_type(
        self, company_id: int, driver_type: Any
    ) -> Driver | None:
        """Trouve un driver par company_id et driver_type (retourne le modèle SQLAlchemy).

        Args:
            company_id: ID de l'entreprise
            driver_type: Type de driver

        Returns:
            Driver ou None si non trouvé
        """
        return Driver.query.filter_by(
            company_id=company_id, driver_type=driver_type
        ).first()

    def count_by_company_and_type(self, company_id: int, driver_type: Any) -> int:
        """Compte les drivers par company_id et driver_type.

        Args:
            company_id: ID de l'entreprise
            driver_type: Type de driver

        Returns:
            Nombre de drivers correspondants
        """
        return Driver.query.filter_by(
            company_id=company_id, driver_type=driver_type
        ).count()

    def find_models_by_company_id(self, company_id: int) -> list[Driver]:
        """Trouve les drivers d'une entreprise (retourne les modèles SQLAlchemy).

        Args:
            company_id: ID de l'entreprise

        Returns:
            Liste de Driver
        """
        return Driver.query.filter_by(company_id=company_id).all()

    def find_model_by_id_and_company_available(
        self, driver_id: int, company_id: int
    ) -> Driver | None:
        """Trouve un driver par son ID et company_id avec is_available=True (retourne le modèle SQLAlchemy).

        Args:
            driver_id: ID du driver
            company_id: ID de l'entreprise

        Returns:
            Driver ou None si non trouvé
        """
        return Driver.query.filter_by(
            id=driver_id, company_id=company_id, is_available=True
        ).first()

    def find_models_by_company_available(self, company_id: int) -> list[Driver]:
        """Trouve les drivers disponibles d'une entreprise (retourne les modèles SQLAlchemy).

        Args:
            company_id: ID de l'entreprise

        Returns:
            Liste de Driver disponibles
        """
        return Driver.query.filter_by(company_id=company_id, is_available=True).all()
